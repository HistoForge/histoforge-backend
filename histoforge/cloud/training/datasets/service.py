"""Dataset-route orchestration around short control cuts and mounted byte IO."""

from __future__ import annotations

import hashlib
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from histoforge.canonical_json import require_sha256
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.training.datasets.archive import (
    extract_training_dataset_archive,
)
from histoforge.cloud.training.datasets.content import (
    AdmittedTrainingDataset,
    TrainingRoiLabelError,
    admit_extracted_training_dataset,
)
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.manifest import roi_geometry_path
from histoforge.cloud.training.datasets.transfer import (
    validate_training_dataset_seal_request,
    validate_training_dataset_start_request,
)
from histoforge.cloud.training.limits import (
    validate_training_api_limits,
    validate_training_service_limits,
)

_SEAL_GATE = threading.Lock()


class TrainingDatasetService:
    """Keep archive work outside the serialized training-state owner."""

    def __init__(
        self,
        files: TrainingDatasetFiles,
        control: Any,
        api_limits: Any,
        service_limits: Any,
        source_limits: Any,
        capability_resolver: Callable[[str], Any],
    ) -> None:
        self._files = files
        self._control = control
        self._api_limits = validate_training_api_limits(api_limits)
        self._service_limits = validate_training_service_limits(service_limits)
        self._source_limits = source_limits
        self._capability_resolver = capability_resolver

    def start(self, request: Any) -> dict[str, Any]:
        """Reclaim bounded expired uploads, then reserve or reuse one dataset."""

        start = self._validate_start_request(request)
        self.cleanup_stale_artifacts()
        result = self._control.reserve(start)
        if result["type"] == "existing":
            self._delete_upload(result["dataset"]["dataset_id"])
        return result

    def cleanup_stale_artifacts(self) -> None:
        """Run one bounded pass over expired and unowned upload storage."""

        self._cleanup_unowned_upload_roots()
        for _ in range(self._service_limits["max_active_dataset_uploads"]):
            expired = self._control.claim_expired_upload()
            if expired is None:
                break
            self._cleanup_expired_upload(expired)

    def put_chunk(
        self,
        dataset_id: str,
        chunk_index: int,
        body: bytes,
        declared_digest: str,
    ) -> dict[str, Any]:
        """Claim, durably publish, and acknowledge one immutable archive range."""

        digest = hashlib.sha256(body).hexdigest()
        if require_sha256(declared_digest, "chunk digest") != digest:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "chunk body digest does not match X-HistoForge-Chunk-Sha256",
                resource_id=dataset_id,
            )
        claim = self._control.claim_chunk(
            dataset_id,
            chunk_index,
            len(body),
            digest,
        )
        if "accepted" in claim:
            try:
                self._files.require_chunk(dataset_id, claim["accepted"])
            except (ArtifactBytesCorrupt, ArtifactBytesMissing, OSError) as error:
                raise _storage_failure(dataset_id) from error
            return {
                "dataset_id": dataset_id,
                "chunk": claim["accepted"],
                "expires_at": claim["expires_at"],
            }

        commit = claim["commit"]
        try:
            durable = self._files.publish_chunk(dataset_id, body, commit)
            return self._control.commit_chunk(dataset_id, commit, durable)
        except Exception as error:
            self._control.abort_chunk(dataset_id, commit)
            if (
                isinstance(error, GatewayFailure)
                and error.error["code"]
                == GatewayErrorCode.TRAINING_UPLOAD_NOT_FOUND.value
            ):
                self._delete_upload(dataset_id)
            raise

    def seal(self, dataset_id: str, request: Any) -> dict[str, Any]:
        """Validate and publish one complete immutable dataset, then expose it."""

        seal_request = self._validate_seal_request(dataset_id, request)
        claim = self._control.claim_seal(dataset_id, seal_request)
        if "sealed" in claim:
            self._delete_upload(dataset_id)
            return claim["sealed"]

        start_digest = claim["start_request_digest"]
        claim_token = claim["claim_token"]
        if not _SEAL_GATE.acquire(blocking=False):
            self._control.abort_seal(dataset_id, start_digest, claim_token)
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "another training dataset is being sealed by this worker",
                resource_id=dataset_id,
            )
        committed = False
        try:
            with tempfile.TemporaryDirectory(prefix="histoforge-training-seal-") as raw:
                temporary = Path(raw)
                archive = self._copy_archive(dataset_id, claim, temporary)
                admitted = self._extract_and_admit(dataset_id, archive, temporary)
                if (
                    admitted.manifest["dataset_digest"]
                    != claim["start_request"]["dataset_digest"]
                ):
                    raise GatewayFailure(
                        GatewayErrorCode.VALIDATION_ERROR,
                        "dataset_digest does not match the uploaded dataset contents",
                        resource_id=dataset_id,
                    )
                self._publish_dataset(dataset_id, admitted)
                record = self._control.commit_seal(
                    dataset_id,
                    start_digest,
                    claim_token,
                    admitted.manifest,
                    admitted.retained_bytes,
                    admitted.task_display_name,
                )
                committed = True
            self._delete_upload(dataset_id)
            return record
        finally:
            try:
                if not committed:
                    self._control.abort_seal(dataset_id, start_digest, claim_token)
            finally:
                _SEAL_GATE.release()

    def get(self, dataset_id: str) -> dict[str, Any]:
        return self._control.get(dataset_id)

    def page(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        return self._control.page(limit, cursor)

    def delete(self, dataset_id: str) -> None:
        """Fence a retained dataset, remove its exact root, then remove its record."""

        cleanup = self._control.begin_delete(dataset_id)
        if cleanup is None:
            return
        try:
            self._files.delete_dataset(dataset_id)
            self._files.delete_upload(dataset_id)
            if self._remaining_bytes(dataset_id) != 0:
                raise OSError("training dataset bytes remain after cleanup")
            self._control.finish_delete(dataset_id)
        except Exception as error:
            failure = unexpected_gateway_failure(error)
            remaining = cleanup["last_known_remaining_bytes"]
            try:
                remaining = self._remaining_bytes(dataset_id)
            except Exception:
                pass
            self._control.record_delete_failure(
                dataset_id,
                failure.error,
                remaining,
            )
            raise failure from error

    def _copy_archive(
        self,
        dataset_id: str,
        claim: dict[str, Any],
        temporary: Path,
    ) -> Path:
        try:
            return self._files.copy_archive(
                dataset_id,
                claim["start_request"],
                self._api_limits,
                temporary / "dataset.zip",
            )
        except (
            ArtifactBytesCorrupt,
            ArtifactBytesMissing,
            OSError,
            ValueError,
        ) as error:
            raise _storage_failure(dataset_id) from error

    def _extract_and_admit(
        self,
        dataset_id: str,
        archive: Path,
        temporary: Path,
    ) -> AdmittedTrainingDataset:
        try:
            extracted = extract_training_dataset_archive(
                archive,
                temporary / "extracted",
                self._api_limits,
            )
            return admit_extracted_training_dataset(
                extracted,
                self._api_limits,
                self._source_limits,
                self._capability_resolver,
            )
        except GatewayFailure:
            raise
        except TrainingRoiLabelError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                str(error),
                resource_id=roi_geometry_path(error.roi_index),
            ) from error
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                f"training dataset validation failed: {error}",
                resource_id=dataset_id,
            ) from error

    def _publish_dataset(
        self, dataset_id: str, admitted: AdmittedTrainingDataset
    ) -> None:
        try:
            self._files.publish_dataset(
                dataset_id,
                admitted.root,
                admitted.manifest,
                self._api_limits,
            )
            self._files.require_dataset(
                dataset_id,
                admitted.manifest,
                self._api_limits,
            )
        except (ArtifactBytesCorrupt, ArtifactBytesMissing, OSError) as error:
            raise _storage_failure(dataset_id) from error
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.INTERNAL_ERROR,
                "training dataset publication failed validation",
                resource_id=dataset_id,
            ) from error

    def _cleanup_unowned_upload_roots(self) -> None:
        """Reclaim a bounded batch of upload roots with no persisted owner."""

        cleanup_limit = self._service_limits["max_active_dataset_uploads"]
        scan_limit = self._service_limits["max_control_records"] + cleanup_limit
        try:
            dataset_ids = self._files.upload_ids(scan_limit)
        except (OSError, ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
            raise _storage_failure("training-datasets") from error
        if not dataset_ids:
            return

        permissions = self._control.upload_root_cleanup_permissions(dataset_ids)
        if (
            not isinstance(permissions, Mapping)
            or set(permissions) != set(dataset_ids)
            or any(type(value) is not bool for value in permissions.values())
        ):
            raise GatewayFailure(
                GatewayErrorCode.INTERNAL_ERROR,
                "training dataset cleanup permissions are invalid",
            )

        cleaned = 0
        for dataset_id in dataset_ids:
            if not permissions[dataset_id]:
                continue
            self._delete_upload(dataset_id)
            cleaned += 1
            if cleaned == cleanup_limit:
                return

    def _delete_upload(self, dataset_id: str) -> None:
        try:
            self._files.delete_upload(dataset_id)
        except (OSError, ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
            raise _storage_failure(dataset_id) from error

    def _cleanup_expired_upload(self, record: dict[str, Any]) -> None:
        dataset_id = record["dataset_id"]
        start_digest = record["start_request_digest"]
        try:
            self._files.delete_upload(dataset_id)
            self._files.delete_dataset(dataset_id)
            self._control.finish_expired_upload_cleanup(dataset_id, start_digest)
        except Exception:
            self._control.abort_expired_upload_cleanup(dataset_id, start_digest)
            raise

    def _remaining_bytes(self, dataset_id: str) -> int:
        return self._files.dataset_remaining_bytes(
            dataset_id
        ) + self._files.upload_remaining_bytes(dataset_id)

    def _validate_start_request(self, value: Any) -> dict[str, Any]:
        try:
            return validate_training_dataset_start_request(value, self._api_limits)
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                f"invalid training dataset start request: {error}",
            ) from error

    @staticmethod
    def _validate_seal_request(dataset_id: str, value: Any) -> dict[str, Any]:
        try:
            return validate_training_dataset_seal_request(value)
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                f"invalid training dataset seal request: {error}",
                resource_id=dataset_id,
            ) from error


def _storage_failure(resource_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.SERVICE_UNAVAILABLE,
        "training dataset storage is temporarily unavailable",
        resource_id=resource_id,
    )
