"""Single-lane adapter joining dataset state to durable chunk observations."""

from __future__ import annotations

from typing import Any

from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.state import TrainingDatasetState


class TrainingDatasetControl:
    """Keep the lane's receipt projection synchronized with Artifact Volume."""

    def __init__(
        self,
        state: TrainingDatasetState,
        files: TrainingDatasetFiles,
        api_limits: Any,
    ) -> None:
        self._state = state
        self._files = files
        self._api_limits = api_limits
        self._commits = {
            record["dataset_id"]: files.chunk_receipts(
                record["dataset_id"], record["request"], api_limits
            )
            for record in state.upload_records()
        }

    def reserve(self, request: Any) -> dict[str, Any]:
        result = self._state.reserve(request, self._commits)
        if result["type"] == "upload":
            self._commits.setdefault(result["dataset_id"], {})
        return result

    def upload_root_cleanup_permissions(self, dataset_ids: Any) -> dict[str, bool]:
        return self._state.upload_root_cleanup_permissions(dataset_ids)

    def claim_chunk(
        self,
        dataset_id: str,
        chunk_index: int,
        body_size: int,
        digest: str,
    ) -> dict[str, Any]:
        durable = self._files.chunk_receipt(dataset_id, chunk_index)
        if durable is not None:
            self._commits.setdefault(dataset_id, {})[chunk_index] = durable
        return self._state.claim_chunk(
            dataset_id,
            chunk_index,
            body_size,
            digest,
            self._commits.get(dataset_id, {}),
        )

    def commit_chunk(
        self,
        dataset_id: str,
        claimed_commit: Any,
        durable_commit: Any,
    ) -> dict[str, Any]:
        observed = {
            **self._commits.get(dataset_id, {}),
            durable_commit["chunk_index"]: durable_commit,
        }
        receipt = self._state.commit_chunk(dataset_id, claimed_commit, observed)
        self._commits[dataset_id] = observed
        return receipt

    def abort_chunk(self, dataset_id: str, commit: Any) -> None:
        self._state.abort_chunk(dataset_id, commit)

    def claim_seal(self, dataset_id: str, request: Any) -> dict[str, Any]:
        return self._state.claim_seal(
            dataset_id,
            request,
            self._commits.get(dataset_id, {}),
        )

    def commit_seal(
        self,
        dataset_id: str,
        start_request_digest: str,
        claim_token: str,
        manifest: Any,
        retained_bytes: int,
        task_display_name: str,
    ) -> dict[str, Any]:
        result = self._state.commit_seal(
            dataset_id,
            start_request_digest,
            claim_token,
            manifest,
            retained_bytes,
            task_display_name,
        )
        self._commits.pop(dataset_id, None)
        return result

    def abort_seal(
        self,
        dataset_id: str,
        start_request_digest: str,
        claim_token: str,
    ) -> None:
        self._state.abort_seal(dataset_id, start_request_digest, claim_token)

    def get(self, dataset_id: str) -> dict[str, Any]:
        return self._state.get(dataset_id)

    def page(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        return self._state.page(limit, cursor)

    def begin_delete(self, dataset_id: str) -> dict[str, Any] | None:
        return self._state.begin_delete(dataset_id)

    def finish_delete(self, dataset_id: str) -> None:
        self._state.finish_delete(dataset_id)

    def record_delete_failure(
        self,
        dataset_id: str,
        error: Any,
        remaining_bytes: int,
    ) -> dict[str, Any]:
        return self._state.record_delete_failure(dataset_id, error, remaining_bytes)

    def claim_expired_upload(self) -> dict[str, Any] | None:
        for record in self._state.upload_records():
            dataset_id = record["dataset_id"]
            self._commits[dataset_id] = self._files.chunk_receipts(
                dataset_id, record["request"], self._api_limits
            )
            claimed = self._state.claim_expired_upload(
                dataset_id,
                self._commits.get(dataset_id, {}),
            )
            if claimed is not None:
                return claimed
        return None

    def finish_expired_upload_cleanup(
        self,
        dataset_id: str,
        start_request_digest: str,
    ) -> None:
        self._state.finish_expired_upload_cleanup(
            dataset_id,
            start_request_digest,
        )
        self._commits.pop(dataset_id, None)

    def abort_expired_upload_cleanup(
        self,
        dataset_id: str,
        start_request_digest: str,
    ) -> None:
        self._state.abort_expired_upload_cleanup(
            dataset_id,
            start_request_digest,
        )
