"""Authenticated Modal adapter for the sole training-state owner."""

import modal

from histoforge.cloud._config import DEPLOYMENT_CONFIG


class TrainingStateClient:
    """Bind one authenticated request to the serialized training-state lane."""

    def __init__(self, state: dict) -> None:
        self._lane = modal.Cls.from_name(
            DEPLOYMENT_CONFIG.gateway_app_name,
            DEPLOYMENT_CONFIG.training_state_class_name,
        )()
        self._deployment_id = state["deployment_id"]
        self._auth_revision = state["auth_revision"]

    def _call(self, method: str, *args):
        return getattr(self._lane, method).remote(
            self._deployment_id,
            self._auth_revision,
            *args,
        )

    def reserve(self, request: dict) -> dict:
        return self._call("reserve", request)

    def upload_root_cleanup_permissions(
        self, dataset_ids: list[str]
    ) -> dict[str, bool]:
        return self._call("upload_root_cleanup_permissions", dataset_ids)

    def claim_chunk(
        self,
        dataset_id: str,
        chunk_index: int,
        body_size: int,
        digest: str,
    ) -> dict:
        return self._call("claim_chunk", dataset_id, chunk_index, body_size, digest)

    def commit_chunk(
        self,
        dataset_id: str,
        claimed_commit: dict,
        durable_commit: dict,
    ) -> dict:
        return self._call("commit_chunk", dataset_id, claimed_commit, durable_commit)

    def abort_chunk(self, dataset_id: str, commit: dict) -> None:
        self._call("abort_chunk", dataset_id, commit)

    def claim_seal(self, dataset_id: str, request: dict) -> dict:
        return self._call("claim_seal", dataset_id, request)

    def commit_seal(
        self,
        dataset_id: str,
        start_request_digest: str,
        claim_token: str,
        manifest: dict,
        retained_bytes: int,
        task_display_name: str,
    ) -> dict:
        return self._call(
            "commit_seal",
            dataset_id,
            start_request_digest,
            claim_token,
            manifest,
            retained_bytes,
            task_display_name,
        )

    def abort_seal(
        self,
        dataset_id: str,
        start_request_digest: str,
        claim_token: str,
    ) -> None:
        self._call("abort_seal", dataset_id, start_request_digest, claim_token)

    def get(self, dataset_id: str) -> dict:
        return self._call("get", dataset_id)

    def page(self, limit: int, cursor: str | None = None) -> dict:
        return self._call("page", limit, cursor)

    def begin_delete(self, dataset_id: str) -> dict | None:
        return self._call("begin_delete", dataset_id)

    def finish_delete(self, dataset_id: str) -> None:
        self._call("finish_delete", dataset_id)

    def record_delete_failure(
        self,
        dataset_id: str,
        error: dict,
        remaining_bytes: int,
    ) -> dict:
        return self._call("record_delete_failure", dataset_id, error, remaining_bytes)

    def claim_expired_upload(self) -> dict | None:
        return self._call("claim_expired_upload")

    def finish_expired_upload_cleanup(
        self,
        dataset_id: str,
        start_request_digest: str,
    ) -> None:
        self._call("finish_expired_upload_cleanup", dataset_id, start_request_digest)

    def abort_expired_upload_cleanup(
        self,
        dataset_id: str,
        start_request_digest: str,
    ) -> None:
        self._call("abort_expired_upload_cleanup", dataset_id, start_request_digest)

    def lookup_run(self, request: dict) -> dict | None:
        return self._call("lookup_run", request)

    def reserve_run(self, request: dict) -> dict:
        return self._call("reserve_run", request)

    def claim_abandoned_run_admission(self) -> dict | None:
        return self._call("claim_abandoned_run_admission")

    def training_run_transient_cleanup_scopes(
        self, run_ids: list[str]
    ) -> dict[str, dict]:
        return self._call("training_run_transient_cleanup_scopes", run_ids)

    def accept_run(
        self,
        request: dict,
        trainer_build: dict,
        total_epochs: int,
        trainer_execution_seconds: int,
        report_digest: str,
    ) -> dict:
        return self._call(
            "accept_run",
            request,
            trainer_build,
            total_epochs,
            trainer_execution_seconds,
            report_digest,
        )

    def attach_run_worker(self, run_id: str, worker_call_id: str) -> dict:
        return self._call("attach_run_worker", run_id, worker_call_id)

    def get_run(self, run_id: str) -> dict:
        return self._call("get_run", run_id)

    def get_run_control(self, run_id: str) -> dict:
        return self._call("get_run_control", run_id)

    def page_run_records(self, limit: int, cursor: str | None = None) -> list[dict]:
        return self._call("page_run_records", limit, cursor)

    def cancel_run(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
    ) -> dict:
        return self._call("cancel_run", run_id, report_revision, report_digest)

    def fail_run(
        self,
        run_id: str,
        expected_state: str,
        report_revision: int,
        report_digest: str,
        finalization_revision: int | None = None,
    ) -> dict:
        return self._call(
            "fail_run",
            run_id,
            expected_state,
            report_revision,
            report_digest,
            finalization_revision,
        )

    def advance_run_progress(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
    ) -> dict:
        return self._call(
            "advance_run_progress", run_id, report_revision, report_digest
        )

    def begin_run_finalization(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
        succeeded_report: dict,
        module_content_digest: str,
    ) -> dict:
        return self._call(
            "begin_run_finalization",
            run_id,
            report_revision,
            report_digest,
            succeeded_report,
            module_content_digest,
        )

    def claim_run_finalization(self, run_id: str, expected_revision: int) -> dict:
        return self._call("claim_run_finalization", run_id, expected_revision)

    def succeed_run(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
        finalization_revision: int,
    ) -> dict:
        return self._call(
            "succeed_run", run_id, report_revision, report_digest, finalization_revision
        )

    def begin_run_delete(self, run_id: str) -> dict | None:
        return self._call("begin_run_delete", run_id)

    def finish_run_delete(self, run_id: str) -> None:
        self._call("finish_run_delete", run_id)

    def training_result_visible(self, run_id: str) -> bool:
        return self._call("training_result_visible", run_id)
