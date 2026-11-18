"""Serialized training-dataset lifecycle decisions over persisted records."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_canonical_uuid,
    require_sha256,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud._time import iso_z
from histoforge.cloud.training.contract import require_training_text
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.registry import TrainingDatasetRegistry
from histoforge.cloud.training.datasets.transfer import (
    training_dataset_chunk_range,
    training_dataset_start_digest,
    validate_training_dataset_chunk_commit,
    validate_training_dataset_seal_request,
    validate_training_dataset_start_request,
    validate_training_dataset_start_result,
)
from histoforge.cloud.training.limits import (
    validate_training_api_limits,
    validate_training_service_limits,
)


class TrainingDatasetState:
    """Own dataset identity, state transitions, claims, and list projections."""

    def __init__(
        self,
        records: TrainingControlRecords,
        api_limits: Any,
        service_limits: Any,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        claim_token_factory: Callable[[], uuid.UUID] = uuid.uuid4,
        dataset_in_use: Callable[[str], bool] = lambda _dataset_id: False,
    ) -> None:
        self._records = records
        self._api_limits = validate_training_api_limits(api_limits)
        self._service_limits = validate_training_service_limits(service_limits)
        self._clock = clock
        self._id_factory = id_factory
        self._claim_token_factory = claim_token_factory
        self._registry = TrainingDatasetRegistry(
            records, self._api_limits, dataset_in_use=dataset_in_use
        )
        self._chunk_claims: dict[tuple[str, int], dict[str, Any]] = {}
        self._expired_cleanup_claims: dict[str, dict[str, Any]] = {}

    def upload_records(self) -> list[dict[str, Any]]:
        """Return bounded unsealed records for outside-lane receipt inspection."""

        return [
            record
            for record in self._records.datasets()
            if record["type"] == "uploading"
        ]

    def upload_root_cleanup_permissions(self, dataset_ids: Any) -> dict[str, bool]:
        """Allow deletion only when persisted state cannot own an upload root."""

        if not isinstance(dataset_ids, list):
            raise ValueError("dataset_ids must be an array")
        maximum = (
            self._service_limits["max_control_records"]
            + self._service_limits["max_active_dataset_uploads"]
        )
        if len(dataset_ids) > maximum:
            raise ValueError("dataset_ids exceeds the cleanup scan bound")
        canonical_ids = [
            require_canonical_uuid(dataset_id, "dataset_id")
            for dataset_id in dataset_ids
        ]
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("dataset_ids must be unique")

        records = {record["dataset_id"]: record for record in self._records.datasets()}
        return {
            dataset_id: records.get(dataset_id) is None
            or records[dataset_id]["type"] == "retained"
            for dataset_id in canonical_ids
        }

    def reserve(
        self,
        value: Any,
        commits_by_dataset: Mapping[str, Mapping[int, Any]],
    ) -> dict[str, Any]:
        """Reuse one digest or reserve one new unsealed dataset ID."""

        start = validate_training_dataset_start_request(value, self._api_limits)
        records = self._records.datasets()
        uploads = self._validated_upload_observations(records, commits_by_dataset)
        now = self._now()

        for record in records:
            if _record_dataset_digest(record) != start["dataset_digest"]:
                continue
            if record["type"] == "retained":
                return validate_training_dataset_start_result(
                    {"type": "existing", "dataset": record}, self._api_limits
                )
            if record["type"] == "uploading":
                if self._expired(record, uploads[record["dataset_id"]], now):
                    raise self._conflict(
                        record["dataset_id"],
                        "an expired upload must be cleaned before this dataset "
                        "can restart",
                    )
                expected = training_dataset_start_digest(start, self._api_limits)
                if record["start_request_digest"] != expected:
                    raise self._conflict(
                        record["dataset_id"],
                        "this dataset digest already has a different live upload",
                    )
                return self._upload_acceptance(record, uploads[record["dataset_id"]])

        active = [
            record
            for record in records
            if record["type"] == "uploading"
            and not self._expired(record, uploads[record["dataset_id"]], now)
        ]
        if len(active) >= self._service_limits["max_active_dataset_uploads"]:
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "the deployment has reached its active training-upload limit",
            )

        dataset_id = str(self._id_factory())
        require_canonical_uuid(dataset_id, "dataset_id")
        record = canonical_tree(
            {
                "type": "uploading",
                "dataset_id": dataset_id,
                "request": start,
                "start_request_digest": training_dataset_start_digest(
                    start, self._api_limits
                ),
                "created_at": iso_z(now),
            }
        )
        winner, created = self._records.reserve_dataset(record)
        if not created:
            raise self._invalid(dataset_id)
        return self._upload_acceptance(winner, {})

    def claim_chunk(
        self,
        dataset_id: str,
        chunk_index: Any,
        body_size: Any,
        digest: Any,
        commits: Mapping[int, Any],
    ) -> dict[str, Any]:
        """Claim one immutable chunk identity before mounted byte publication."""

        self._purge_transient_claims()
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        record = self._live_upload(canonical_id, commits)
        if canonical_id in self._expired_cleanup_claims:
            raise self._upload_not_found(canonical_id)
        if self._active_seal_claim(record) is not None:
            raise self._conflict(canonical_id, "dataset sealing is already in progress")

        expected = training_dataset_chunk_range(
            record["request"], chunk_index, self._api_limits
        )
        if body_size != expected["size_bytes"]:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "chunk body length does not match its canonical range",
                resource_id=canonical_id,
            )
        canonical_digest = require_sha256(digest, "chunk digest")
        observed = self._normalize_commits(record, commits)
        existing = observed.get(expected["chunk_index"])
        if existing is not None:
            if existing["sha256"] != canonical_digest:
                raise self._conflict(canonical_id, "chunk bytes differ")
            return {
                "accepted": existing,
                "expires_at": self._expires_at(record, observed),
            }

        key = (canonical_id, expected["chunk_index"])
        current = self._chunk_claims.get(key)
        if current is None:
            commit = validate_training_dataset_chunk_commit(
                {
                    "start_request_digest": record["start_request_digest"],
                    "chunk_index": expected["chunk_index"],
                    "size_bytes": expected["size_bytes"],
                    "sha256": canonical_digest,
                    "accepted_at": iso_z(self._now()),
                }
            )
            current = {
                "commit": commit,
                "deadline": self._writer_claim_deadline(),
            }
            self._chunk_claims[key] = current
        if current["commit"]["sha256"] != canonical_digest:
            raise self._conflict(canonical_id, "chunk bytes differ")
        return {
            "start_request": record["request"],
            "commit": current["commit"],
        }

    def commit_chunk(
        self,
        dataset_id: str,
        value: Any,
        commits: Mapping[int, Any],
    ) -> dict[str, Any]:
        """Acknowledge one already durable chunk and release its transient claim."""

        self._purge_transient_claims()
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        commit = validate_training_dataset_chunk_commit(value)
        record = self._live_upload(canonical_id, commits)
        key = (canonical_id, commit["chunk_index"])
        claim = self._chunk_claims.get(key)
        if claim is None or claim["commit"] != commit:
            raise self._invalid(canonical_id)
        observed = self._normalize_commits(record, commits)
        durable = observed.get(commit["chunk_index"])
        if durable is None or _chunk_identity(durable) != _chunk_identity(commit):
            raise self._invalid(canonical_id)
        del self._chunk_claims[key]
        return canonical_tree(
            {
                "dataset_id": canonical_id,
                "chunk": durable,
                "expires_at": self._expires_at(record, observed),
            }
        )

    def abort_chunk(self, dataset_id: str, value: Any) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        commit = validate_training_dataset_chunk_commit(value)
        key = (canonical_id, commit["chunk_index"])
        current = self._chunk_claims.get(key)
        if current is not None and current["commit"] == commit:
            del self._chunk_claims[key]

    def claim_seal(
        self,
        dataset_id: str,
        value: Any,
        commits: Mapping[int, Any],
    ) -> dict[str, Any]:
        """Claim one complete upload before archive validation and publication."""

        self._purge_transient_claims()
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        if canonical_id in self._expired_cleanup_claims:
            raise self._upload_not_found(canonical_id)
        request = validate_training_dataset_seal_request(value)
        current = self._records.get_dataset(canonical_id)
        if current is not None and current["type"] == "retained":
            return {"sealed": current}
        record = self._live_upload(canonical_id, commits)
        if request["start_request_digest"] != record["start_request_digest"]:
            raise self._conflict(canonical_id, "seal request differs")
        if any(key[0] == canonical_id for key in self._chunk_claims):
            raise self._conflict(
                canonical_id, "a chunk publication is still in progress"
            )
        observed = self._normalize_commits(record, commits)
        expected_count = self._chunk_count(record)
        if set(observed) != set(range(expected_count)):
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "training dataset upload is incomplete",
                resource_id=canonical_id,
            )
        claim = self._active_seal_claim(record)
        if claim is not None:
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "dataset sealing is already in progress",
                resource_id=canonical_id,
            )
        claim_token = require_canonical_uuid(
            str(self._claim_token_factory()), "seal claim token"
        )
        seal_claim = {
            "claim_token": claim_token,
            # Keep control ownership beyond the longest gateway input. The
            # opaque token prevents this owner from mutating successor state.
            "expires_at": iso_z(self._writer_claim_deadline()),
        }
        self._records.replace_dataset({**record, "seal_claim": seal_claim})
        return {
            "start_request": record["request"],
            "start_request_digest": record["start_request_digest"],
            "claim_token": claim_token,
        }

    def commit_seal(
        self,
        dataset_id: str,
        start_request_digest: Any,
        claim_token: Any,
        manifest: Any,
        retained_bytes: Any,
        task_display_name: Any,
    ) -> dict[str, Any]:
        """Replace one upload record with its already published retained dataset."""

        self._purge_transient_claims()
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        current = self._records.get_dataset(canonical_id)
        if current is not None and current["type"] == "retained":
            return current
        if current is None or current["type"] != "uploading":
            raise self._upload_not_found(canonical_id)
        digest = require_sha256(start_request_digest, "start_request_digest")
        token = require_canonical_uuid(claim_token, "seal claim token")
        claim = self._active_seal_claim(current)
        if (
            digest != current["start_request_digest"]
            or claim is None
            or claim["claim_token"] != token
        ):
            raise self._invalid(canonical_id)
        dataset = validate_training_dataset_manifest(manifest, self._api_limits)
        if dataset["dataset_digest"] != current["request"]["dataset_digest"]:
            raise self._invalid(canonical_id)
        if (
            isinstance(retained_bytes, bool)
            or not isinstance(retained_bytes, int)
            or retained_bytes < 0
        ):
            raise ValueError("retained_bytes must be a nonnegative integer")
        record = canonical_tree(
            {
                "type": "retained",
                "dataset_id": canonical_id,
                "dataset_digest": dataset["dataset_digest"],
                "retained_bytes": retained_bytes,
                "task_display_name": require_training_text(
                    task_display_name,
                    "task_display_name",
                    self._api_limits["report"]["max_text_bytes"],
                ),
                "image_count": len(dataset["images"]),
                "roi_count": len(dataset["rois"]),
                "created_at": current["created_at"],
            }
        )
        published = self._records.replace_dataset(record)
        self._drop_chunk_claims(canonical_id)
        return published

    def abort_seal(
        self,
        dataset_id: str,
        start_request_digest: Any,
        claim_token: Any,
    ) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        digest = require_sha256(start_request_digest, "start_request_digest")
        token = require_canonical_uuid(claim_token, "seal claim token")
        current = self._records.get_dataset(canonical_id)
        if current is None or current["type"] != "uploading":
            return
        claim = current.get("seal_claim")
        if (
            claim is not None
            and current["start_request_digest"] == digest
            and claim["claim_token"] == token
        ):
            self._records.replace_dataset(
                {key: value for key, value in current.items() if key != "seal_claim"}
            )

    def get(self, dataset_id: str) -> dict[str, Any]:
        return self._registry.get(dataset_id)

    def page(self, limit: Any, cursor: str | None = None) -> dict[str, Any]:
        return self._registry.page(limit, cursor)

    def begin_delete(self, dataset_id: str) -> dict[str, Any] | None:
        """Fence one retained dataset before outside-lane recursive deletion."""

        return self._registry.begin_delete(dataset_id)

    def finish_delete(self, dataset_id: str) -> None:
        self._registry.finish_delete(dataset_id)

    def record_delete_failure(
        self,
        dataset_id: str,
        error: Any,
        remaining_bytes: Any,
    ) -> dict[str, Any]:
        return self._registry.record_delete_failure(dataset_id, error, remaining_bytes)

    def claim_expired_upload(
        self,
        dataset_id: str,
        commits: Mapping[int, Any],
    ) -> dict[str, Any] | None:
        """Fence one expired upload before deleting only its owned roots."""

        self._purge_transient_claims()
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        record = self._records.get_dataset(canonical_id)
        if record is None or record["type"] != "uploading":
            return None
        if canonical_id in self._expired_cleanup_claims:
            return None
        observed = self._normalize_commits(record, commits)
        if not self._expired(record, observed, self._now()):
            return None
        if any(key[0] == canonical_id for key in self._chunk_claims):
            return None
        if self._active_seal_claim(record) is not None:
            return None
        self._expired_cleanup_claims[canonical_id] = {
            "start_request_digest": record["start_request_digest"],
            "deadline": self._writer_claim_deadline(),
        }
        return record

    def finish_expired_upload_cleanup(
        self, dataset_id: str, start_request_digest: Any
    ) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        digest = require_sha256(start_request_digest, "start_request_digest")
        current = self._records.get_dataset(canonical_id)
        claim = self._expired_cleanup_claims.get(canonical_id)
        if current is None:
            self._expired_cleanup_claims.pop(canonical_id, None)
            return
        if (
            current["type"] != "uploading"
            or current["start_request_digest"] != digest
            or claim is None
            or claim["start_request_digest"] != digest
        ):
            raise self._invalid(canonical_id)
        self._records.remove_dataset(canonical_id)
        self._expired_cleanup_claims.pop(canonical_id, None)

    def abort_expired_upload_cleanup(
        self, dataset_id: str, start_request_digest: Any
    ) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        digest = require_sha256(start_request_digest, "start_request_digest")
        claim = self._expired_cleanup_claims.get(canonical_id)
        if claim is not None and claim["start_request_digest"] == digest:
            del self._expired_cleanup_claims[canonical_id]

    def _validated_upload_observations(
        self,
        records: list[Mapping[str, Any]],
        values: Mapping[str, Mapping[int, Any]],
    ) -> dict[str, dict[int, dict[str, Any]]]:
        upload_ids = {
            record["dataset_id"] for record in records if record["type"] == "uploading"
        }
        if set(values) != upload_ids:
            raise ValueError("chunk observations must exactly cover active uploads")
        by_id = {record["dataset_id"]: record for record in records}
        return {
            dataset_id: self._normalize_commits(by_id[dataset_id], commits)
            for dataset_id, commits in values.items()
        }

    def _normalize_commits(
        self,
        record: Mapping[str, Any],
        commits: Mapping[int, Any],
    ) -> dict[int, dict[str, Any]]:
        if not isinstance(commits, Mapping):
            raise ValueError("chunk commits must be an object")
        result = {}
        for key, value in commits.items():
            if isinstance(key, bool) or not isinstance(key, int) or key < 0:
                raise ValueError("chunk commit index must be a nonnegative integer")
            commit = validate_training_dataset_chunk_commit(value)
            expected = training_dataset_chunk_range(
                record["request"], key, self._api_limits
            )
            if (
                commit["start_request_digest"] != record["start_request_digest"]
                or commit["chunk_index"] != key
                or commit["size_bytes"] != expected["size_bytes"]
            ):
                raise self._invalid(record["dataset_id"])
            result[key] = commit
        return result

    def _live_upload(
        self, dataset_id: str, commits: Mapping[int, Any]
    ) -> dict[str, Any]:
        record = self._records.get_dataset(dataset_id)
        if record is None or record["type"] != "uploading":
            raise self._upload_not_found(dataset_id)
        observed = self._normalize_commits(record, commits)
        if self._expired(record, observed, self._now()):
            raise self._upload_not_found(dataset_id)
        return record

    def _upload_acceptance(
        self, record: Mapping[str, Any], commits: Mapping[int, Any]
    ) -> dict[str, Any]:
        return validate_training_dataset_start_result(
            {
                "type": "upload",
                "dataset_id": record["dataset_id"],
                "start_request_digest": record["start_request_digest"],
                "expires_at": self._expires_at(record, commits),
            },
            self._api_limits,
        )

    def _expires_at(self, record: Mapping[str, Any], commits: Mapping[int, Any]) -> str:
        observed = self._normalize_commits(record, commits)
        activity = [_parse_z(record["created_at"])]
        activity.extend(_parse_z(item["accepted_at"]) for item in observed.values())
        return iso_z(
            max(activity)
            + timedelta(
                seconds=self._service_limits["dataset_upload_idle_timeout_seconds"]
            )
        )

    def _expired(
        self,
        record: Mapping[str, Any],
        commits: Mapping[int, Any],
        now: datetime,
    ) -> bool:
        return now >= _parse_z(self._expires_at(record, commits))

    def _chunk_count(self, record: Mapping[str, Any]) -> int:
        request = record["request"]
        return (request["total_bytes"] - 1) // request["chunk_size_bytes"] + 1

    def _purge_transient_claims(self) -> None:
        now = self._now()
        self._chunk_claims = {
            key: claim
            for key, claim in self._chunk_claims.items()
            if claim["deadline"] > now
        }
        self._expired_cleanup_claims = {
            key: claim
            for key, claim in self._expired_cleanup_claims.items()
            if claim["deadline"] > now
        }

    def _drop_chunk_claims(self, dataset_id: str) -> None:
        self._chunk_claims = {
            key: claim
            for key, claim in self._chunk_claims.items()
            if key[0] != dataset_id
        }

    def _writer_claim_deadline(self) -> datetime:
        """Return the shared deadline for one still-running dataset writer."""

        return self._now() + timedelta(
            seconds=self._service_limits["dataset_writer_claim_seconds"]
        )

    def _active_seal_claim(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        claim = record.get("seal_claim")
        if claim is None or _parse_z(claim["expires_at"]) <= self._now():
            return None
        return claim

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("training dataset clock must be timezone-aware")
        return value.astimezone(timezone.utc).replace(
            microsecond=value.microsecond // 1000 * 1000
        )

    @staticmethod
    def _conflict(dataset_id: str, message: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.TRAINING_STATE_CONFLICT,
            message,
            resource_id=dataset_id,
        )

    @staticmethod
    def _upload_not_found(dataset_id: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.TRAINING_UPLOAD_NOT_FOUND,
            "the training dataset upload does not exist or has expired",
            resource_id=dataset_id,
        )

    @staticmethod
    def _invalid(dataset_id: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "training dataset state is inconsistent",
            resource_id=dataset_id,
        )


def _record_dataset_digest(record: Mapping[str, Any]) -> str | None:
    if record["type"] == "uploading":
        return record["request"]["dataset_digest"]
    if record["type"] == "retained":
        return record["dataset_digest"]
    return None


def _chunk_identity(value: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        value["start_request_digest"],
        value["chunk_index"],
        value["size_bytes"],
        value["sha256"],
    )


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
