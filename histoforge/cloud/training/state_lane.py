"""Serialized control-record owner for managed training.

Heavy dataset and trainer work runs outside this class. This lane owns only the
short persisted cuts that validate already-published artifacts and make their
results visible.
"""

from __future__ import annotations

import logging
from typing import Any

import modal

from histoforge.cloud._app import app, artifact_volume, image
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.gateway_state_client import GatewayStateClient
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.datasets.control import TrainingDatasetControl
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.state import TrainingDatasetState
from histoforge.cloud.training.runs.control import TrainingRunControl
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.state import TrainingRunState

_ARTIFACT_ROOT = "/artifacts"
_LOG = logging.getLogger(__name__)


@app.function(
    image=image,
    memory=DEPLOYMENT_CONFIG.gateway_memory_mib,
    timeout=300,
    volumes={_ARTIFACT_ROOT: artifact_volume},
    retries=0,
)
def cleanup_training_storage() -> None:
    """Perform one bounded cleanup pass outside the serialized state lane."""

    from histoforge.cloud.training.gateway import TrainingGatewayServices

    state = GatewayStateClient().current()
    if state is None:
        return
    TrainingGatewayServices(
        _ARTIFACT_ROOT,
        reload_volume=artifact_volume.reload,
        commit_volume=artifact_volume.commit,
    ).cleanup(state)


@app.cls(
    image=image,
    timeout=300,
    volumes={_ARTIFACT_ROOT: artifact_volume},
    max_containers=1,
    retries=0,
)
@modal.concurrent(max_inputs=1)
class TrainingStateLane:
    """Sole writer of persisted training control records."""

    @modal.enter()
    def schedule_startup_cleanup(self) -> None:
        """Trigger bounded cleanup without delaying state-service readiness."""

        try:
            cleanup_training_storage.spawn()
        except Exception:
            _LOG.warning(
                "training startup cleanup could not be scheduled",
                exc_info=True,
            )

    @modal.method()
    def reserve(
        self,
        deployment_id: str,
        auth_revision: int,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().reserve(request)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def upload_root_cleanup_permissions(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_ids: list[str],
    ) -> dict[str, bool]:
        """Fence a bounded batch before deleting unowned upload roots."""

        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().upload_root_cleanup_permissions(dataset_ids)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def claim_chunk(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        chunk_index: int,
        body_size: int,
        digest: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().claim_chunk(
            dataset_id, chunk_index, body_size, digest
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def commit_chunk(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        claimed_commit: dict[str, Any],
        durable_commit: dict[str, Any],
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().commit_chunk(
            dataset_id, claimed_commit, durable_commit
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def abort_chunk(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        commit: dict[str, Any],
    ) -> None:
        _authorize(deployment_id, auth_revision)
        self._dataset_control().abort_chunk(dataset_id, commit)
        _authorize(deployment_id, auth_revision)

    @modal.method()
    def claim_seal(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().claim_seal(dataset_id, request)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def commit_seal(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        start_request_digest: str,
        claim_token: str,
        manifest: dict[str, Any],
        retained_bytes: int,
        task_display_name: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().commit_seal(
            dataset_id,
            start_request_digest,
            claim_token,
            manifest,
            retained_bytes,
            task_display_name,
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def abort_seal(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        start_request_digest: str,
        claim_token: str,
    ) -> None:
        _authorize(deployment_id, auth_revision)
        self._dataset_control().abort_seal(
            dataset_id, start_request_digest, claim_token
        )
        _authorize(deployment_id, auth_revision)

    @modal.method()
    def get(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().get(dataset_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def page(
        self,
        deployment_id: str,
        auth_revision: int,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().page(limit, cursor)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def begin_delete(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
    ) -> dict[str, Any] | None:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().begin_delete(dataset_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def finish_delete(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
    ) -> None:
        _authorize(deployment_id, auth_revision)
        self._dataset_control().finish_delete(dataset_id)
        _authorize(deployment_id, auth_revision)

    @modal.method()
    def record_delete_failure(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        error: dict[str, Any],
        remaining_bytes: int,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().record_delete_failure(
            dataset_id, error, remaining_bytes
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def claim_expired_upload(
        self,
        deployment_id: str,
        auth_revision: int,
    ) -> dict[str, Any] | None:
        _authorize(deployment_id, auth_revision)
        result = self._dataset_control().claim_expired_upload()
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def finish_expired_upload_cleanup(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        start_request_digest: str,
    ) -> None:
        _authorize(deployment_id, auth_revision)
        self._dataset_control().finish_expired_upload_cleanup(
            dataset_id, start_request_digest
        )
        _authorize(deployment_id, auth_revision)

    @modal.method()
    def abort_expired_upload_cleanup(
        self,
        deployment_id: str,
        auth_revision: int,
        dataset_id: str,
        start_request_digest: str,
    ) -> None:
        _authorize(deployment_id, auth_revision)
        self._dataset_control().abort_expired_upload_cleanup(
            dataset_id, start_request_digest
        )
        _authorize(deployment_id, auth_revision)

    @modal.method()
    def lookup_run(
        self,
        deployment_id: str,
        auth_revision: int,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return an exact run replay without reserving mutable state."""

        _authorize(deployment_id, auth_revision)
        result = self._run_control().lookup(request)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def reserve_run(
        self,
        deployment_id: str,
        auth_revision: int,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().reserve(request)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def claim_abandoned_run_admission(
        self,
        deployment_id: str,
        auth_revision: int,
    ) -> dict[str, Any] | None:
        """Fence one expired run preparation before outside-lane cleanup."""

        _authorize(deployment_id, auth_revision)
        result = self._run_control().claim_abandoned_admission()
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def training_run_transient_cleanup_scopes(
        self,
        deployment_id: str,
        auth_revision: int,
        run_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Classify a bounded run-ID batch before deleting temporary roots."""

        _authorize(deployment_id, auth_revision)
        result = self._run_control().transient_cleanup_scopes(run_ids)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def accept_run(
        self,
        deployment_id: str,
        auth_revision: int,
        request: dict[str, Any],
        trainer_build: dict[str, Any],
        total_epochs: int,
        trainer_execution_seconds: int,
        report_digest: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().accept(
            request,
            trainer_build,
            total_epochs,
            trainer_execution_seconds,
            report_digest,
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def attach_run_worker(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        worker_call_id: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().attach_worker(run_id, worker_call_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def claim_run_worker(
        self,
        run_id: str,
        executing_build: dict[str, Any],
    ) -> dict[str, Any]:
        """Claim an accepted run through the Modal workspace trust boundary.

        Only the internal worker uses this method. Its durable authority is the
        accepted run and exact build, independent of a user's later token rotation.
        HTTP callers continue through the authenticated gateway methods.
        """

        return self._run_control().claim_worker(run_id, executing_build)

    @modal.method()
    def get_worker_run_control(self, run_id: str) -> dict[str, Any]:
        """Let a workspace worker observe its attachment before claiming."""

        return self._run_control().get_control(run_id)

    @modal.method()
    def advance_worker_run_progress(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
    ) -> dict[str, Any]:
        """Validate a worker's report through the same serialized state owner.

        Like the physical worker claim, this internal workspace call does not
        depend on the user's current gateway token. It grants no direct access to
        control files and cannot bypass the claimed-run or report-history checks.
        """

        return self._run_control().advance_progress(
            run_id, report_revision, report_digest
        )

    @modal.method()
    def advance_run_progress(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        report_revision: int,
        report_digest: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().advance_progress(
            run_id, report_revision, report_digest
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def cancel_run(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        report_revision: int,
        report_digest: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().cancel(run_id, report_revision, report_digest)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def fail_run(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        expected_state: str,
        report_revision: int,
        report_digest: str,
        finalization_revision: int | None = None,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().fail(
            run_id,
            expected_state,
            report_revision,
            report_digest,
            finalization_revision,
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def begin_run_finalization(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        report_revision: int,
        report_digest: str,
        succeeded_report: dict[str, Any],
        module_content_digest: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().begin_finalization(
            run_id,
            report_revision,
            report_digest,
            succeeded_report,
            module_content_digest,
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def claim_run_finalization(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Commit the next private save fence before the module writer acts."""

        _authorize(deployment_id, auth_revision)
        result = self._run_control().claim_finalization(run_id, expected_revision)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def succeed_run(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
        report_revision: int,
        report_digest: str,
        finalization_revision: int,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().succeed(
            run_id, report_revision, report_digest, finalization_revision
        )
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def get_run(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().get(run_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def page_run_records(
        self,
        deployment_id: str,
        auth_revision: int,
        limit: int,
        cursor: str | None,
    ) -> list[dict[str, Any]]:
        """Return ordered run snapshots without reading report artifacts."""

        _authorize(deployment_id, auth_revision)
        result = self._run_control().page_records(limit, cursor)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def get_run_control(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
    ) -> dict[str, Any]:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().get_control(run_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def training_result_visible(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
    ) -> bool:
        """Gate a stored result on SUCCEEDED while its Jobs row exists."""

        _authorize(deployment_id, auth_revision)
        result = self._run_control().training_result_visible(run_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def begin_run_delete(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
    ) -> dict[str, Any] | None:
        _authorize(deployment_id, auth_revision)
        result = self._run_control().begin_delete(run_id)
        _authorize(deployment_id, auth_revision)
        return result

    @modal.method()
    def finish_run_delete(
        self,
        deployment_id: str,
        auth_revision: int,
        run_id: str,
    ) -> None:
        _authorize(deployment_id, auth_revision)
        self._run_control().finish_delete(run_id)
        _authorize(deployment_id, auth_revision)

    @modal.method()
    def deployment_barrier(self) -> None:
        """Return after every earlier training-state mutation has completed."""

    def _dataset_control(self) -> TrainingDatasetControl:
        return self._owners()[0]

    def _run_control(self) -> TrainingRunControl:
        return self._owners()[1]

    def _owners(
        self,
    ) -> tuple[TrainingDatasetControl, TrainingRunControl]:
        current = getattr(self, "_training_owners", None)
        if current is not None:
            return current
        api_limits = DEPLOYMENT_CONFIG.training_api_limits.wire_value()
        service_limits = DEPLOYMENT_CONFIG.training_service_limits.wire_value()
        files = TrainingDatasetFiles(
            _ARTIFACT_ROOT,
            reload_volume=artifact_volume.reload,
            commit_volume=artifact_volume.commit,
        )
        records = TrainingControlRecords(
            _ARTIFACT_ROOT,
            api_limits,
            service_limits,
            reload_volume=artifact_volume.reload,
            commit_volume=artifact_volume.commit,
        )
        run_state = TrainingRunState(records, api_limits, service_limits)
        run_files = TrainingRunFiles(
            _ARTIFACT_ROOT,
            api_limits,
            reload_volume=artifact_volume.reload,
            commit_volume=artifact_volume.commit,
        )
        source_files = TrainingSourceFiles(
            _ARTIFACT_ROOT,
            DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes,
            reload_volume=artifact_volume.reload,
            commit_volume=artifact_volume.commit,
        )
        run_control = TrainingRunControl(run_state, run_files, source_files, api_limits)
        dataset_state = TrainingDatasetState(
            records,
            api_limits,
            service_limits,
            dataset_in_use=run_state.dataset_in_use,
        )
        current = (
            TrainingDatasetControl(dataset_state, files, api_limits),
            run_control,
        )
        self._training_owners = current
        return current


def _authorize(deployment_id: str, auth_revision: int) -> None:
    from histoforge.cloud._auth import require_current_auth_generation

    auth = GatewayStateClient().current
    require_current_auth_generation(auth, deployment_id, auth_revision)
