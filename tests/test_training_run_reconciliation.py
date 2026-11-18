from __future__ import annotations

import copy
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.dinov3_training import project_dinov3_training_model
from histoforge.cloud.architectures.registry import resolved_fixed_field_profile
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt
from histoforge.cloud.modules.store import ModuleStore
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.runs.control import TrainingRunControl
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.finalizer import (
    TrainingRunFinalizer,
    finalize_training_result,
)
from histoforge.cloud.training.runs.reconciliation import TrainingRunReconciler
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_preparation import TrainingSourcePreparation
from histoforge.cloud.training.runs.state import TrainingRunState
from histoforge.cloud.training.runs.terminalizer import TrainingRunTerminalizer
from histoforge.cloud.training.runs.worker import run_training_worker
from modal.exception import RemoteError
from tests.test_module_store import _training_result_source
from tests.test_training_capabilities import _entry, _registry
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_state import service_limits
from tests.test_training_result_module import _success
from tests.test_training_run_contract import trainer_build
from tests.test_training_source_preparation import (
    RUN_ID,
    install_module,
    request_for,
    workflow_recipe,
)

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


class _ControlClient:
    def __init__(self, control: TrainingRunControl) -> None:
        self.control = control

    def get_run_control(self, run_id: str) -> dict:
        return self.control.get_control(run_id)

    def claim_run_worker(self, run_id: str, build: dict) -> dict:
        return self.control.claim_worker(run_id, build)

    def advance_run_progress(
        self, run_id: str, report_revision: int, report_digest: str
    ) -> dict:
        return self.control.advance_progress(run_id, report_revision, report_digest)

    def begin_run_finalization(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
        succeeded_report: dict,
        module_content_digest: str,
    ) -> dict:
        return self.control.begin_finalization(
            run_id,
            report_revision,
            report_digest,
            succeeded_report,
            module_content_digest,
        )

    def succeed_run(
        self,
        run_id: str,
        report_revision: int,
        report_digest: str,
        finalization_revision: int,
    ) -> dict:
        return self.control.succeed(
            run_id, report_revision, report_digest, finalization_revision
        )

    def claim_run_finalization(self, run_id: str, expected_revision: int) -> dict:
        return self.control.claim_finalization(run_id, expected_revision)

    def cancel_run(self, run_id: str, report_revision: int, report_digest: str) -> dict:
        return self.control.cancel(run_id, report_revision, report_digest)

    def fail_run(
        self,
        run_id: str,
        expected_state: str,
        revision: int,
        digest: str,
        finalization_revision: int | None = None,
    ) -> dict:
        return self.control.fail(
            run_id, expected_state, revision, digest, finalization_revision
        )


class _Call:
    def __init__(self, result: object = None, *, pending: bool = False) -> None:
        self.result = result
        self.pending = pending
        self.cancelled = False

    def get(self, *, timeout: int) -> object:
        assert timeout == 0
        if self.pending:
            raise TimeoutError
        return self.result

    def cancel(self) -> None:
        self.cancelled = True


def test_completed_call_adopts_candidate_installs_module_and_succeeds(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    call = _Call({"published": True})
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda call_id: call if call_id == "fc-training" else None,
        training_api_limits(),
        clock=lambda: NOW,
    )

    succeeded = reconciler.reconcile(RUN_ID)

    assert succeeded["state"] == "SUCCEEDED"
    assert succeeded["progress"] == {"completed_epochs": 3, "total_epochs": 3}
    assert context["control"].get_report(RUN_ID)["report"] == context["success_report"]
    record = context["result_store"].current.read_record(f"workspace/run-{RUN_ID}")
    assert (
        record["content_digest"] == succeeded["finalization"]["module_content_digest"]
    )
    assert not context["source_root"].exists()
    assert context["result_files"].inspect_candidate(RUN_ID) is None
    _assert_only_advertised_report(context, succeeded)
    assert reconciler.reconcile(RUN_ID) == succeeded


def test_blocked_module_install_does_not_block_training_control_reads(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_install(run_id: str):
        entered.set()
        assert release.wait(timeout=5)
        return context["finalize_result"](run_id)

    finalizer = TrainingRunFinalizer(
        context["run_files"],
        context["result_files"],
        context["source_files"],
        context["client"],
        blocked_install,
        training_api_limits(),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(finalizer.reconcile_candidate, RUN_ID)
        try:
            assert entered.wait(timeout=5)
            current = context["client"].get_run_control(RUN_ID)
            assert current["run"]["state"] == "FINALIZING"
        finally:
            release.set()
        succeeded = pending.result(timeout=5)

    assert succeeded is not None
    assert succeeded["state"] == "SUCCEEDED"


def test_worker_claim_through_candidate_reconciliation_creates_one_private_module(
    tmp_path: Path,
) -> None:
    context = _accepted_context(tmp_path)
    context["control"].attach_worker(RUN_ID, "fc-training")

    def publish(run, _entry):
        assert context["control"].get_control(RUN_ID)["worker"]["worker_claimed"]
        assert run["request"] == context["request"]
        context["result_files"].publish_candidate(
            RUN_ID,
            source_root=context["source_root"],
            source_marker=context["marker"],
            selected_weights=context["weights"],
            succeeded_report=context["success_report"],
            request=run["request"],
            project_model=project_dinov3_training_model,
        )

    run_training_worker(
        RUN_ID,
        control=context["client"],
        capabilities=_registry(_entry()),
        api_limits=training_api_limits(),
        build_entry=lambda _entry: trainer_build(),
        train_and_publish=publish,
        clock=lambda: NOW,
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    succeeded = reconciler.reconcile(RUN_ID)
    assert succeeded["state"] == "SUCCEEDED"
    assert reconciler.reconcile(RUN_ID) == succeeded
    assert context["result_store"].current.list_module_refs() == [
        f"workspace/run-{RUN_ID}"
    ]


def test_worker_build_rejection_becomes_the_visible_failed_report(tmp_path: Path):
    context = _accepted_context(tmp_path)
    context["control"].attach_worker(RUN_ID, "fc-training")
    with pytest.raises(GatewayFailure) as rejected:
        run_training_worker(
            RUN_ID,
            control=context["client"],
            capabilities=_registry(_entry()),
            api_limits=training_api_limits(),
            build_entry=lambda _: {**trainer_build(), "artifact_digest": "0" * 64},
            train_and_publish=lambda *_: pytest.fail("rejected build must not train"),
            clock=lambda: NOW,
        )

    class FailedCall:
        def get(self, *, timeout):
            raise rejected.value

    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _: FailedCall(),
        training_api_limits(),
        clock=lambda: NOW,
    )
    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert failed["result"]["error"] == rejected.value.error
    assert context["control"].get_report(RUN_ID)["report"]["failure"] == (
        rejected.value.error
    )
    assert context["result_store"].current.list_module_refs() == []
    _assert_only_advertised_report(context, failed)


def test_active_call_stays_running_without_scanning_for_a_result(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    call = _Call(pending=True)
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: call,
        training_api_limits(),
        clock=lambda: NOW,
    )

    assert reconciler.reconcile(RUN_ID)["state"] == "RUNNING"
    assert context["result_files"].inspect_candidate(RUN_ID) is None


@pytest.mark.parametrize("failure_type", [MemoryError, RemoteError])
def test_worker_memory_failure_is_terminal_and_does_not_publish_a_module(
    tmp_path: Path, failure_type
) -> None:
    context = _running_context(tmp_path)

    class FailedCall:
        def get(self, *, timeout):
            assert timeout == 0
            raise failure_type("worker exhausted memory")

    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _: FailedCall(),
        training_api_limits(),
        clock=lambda: NOW,
    )
    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert failed["result"]["error"]["code"] == "RUNNER_FAILED"
    assert context["control"].get_report(RUN_ID)["report"]["state"] == "FAILED"
    assert context["result_store"].current.list_module_refs() == []
    assert not context["source_root"].exists()
    _assert_only_advertised_report(context, failed)
    assert reconciler.reconcile(RUN_ID) == failed


def test_completed_call_without_candidate_fails_with_a_terminal_report(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    context["run_files"].publish_report_candidate(
        context["control"].get_report(RUN_ID)["report"]
    )
    result_staging = (
        context["artifact_root"] / "training/runs/result-staging" / RUN_ID / "stale"
    )
    result_staging.mkdir(parents=True)
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert failed["result"]["error"]["code"] == "RUNNER_FAILED"
    report = context["control"].get_report(RUN_ID)
    assert report["report"]["failure"] == failed["result"]["error"]
    assert report["report_digest"] == failed["result"]["report_digest"]
    assert not context["source_root"].exists()
    assert context["result_files"].inspect_candidate(RUN_ID) is None
    assert not result_staging.exists()
    assert not (
        context["artifact_root"] / "training/runs/report-candidates" / RUN_ID
    ).exists()
    assert context["control"].get_report(RUN_ID)["report"] == report["report"]
    _assert_only_advertised_report(context, failed)


def test_cancelled_run_removes_transients_but_preserves_its_report(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    context["run_files"].publish_report_candidate(
        context["control"].get_report(RUN_ID)["report"]
    )
    result_staging = (
        context["artifact_root"] / "training/runs/result-staging" / RUN_ID / "stale"
    )
    result_staging.mkdir(parents=True)
    cancelled = context["terminalizer"].cancel(RUN_ID)["run"]
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(pending=True),
        training_api_limits(),
        clock=lambda: NOW,
    )

    assert reconciler.reconcile(RUN_ID) == cancelled
    assert not context["source_root"].exists()
    assert context["result_files"].inspect_candidate(RUN_ID) is None
    assert not result_staging.exists()
    assert not (
        context["artifact_root"] / "training/runs/report-candidates" / RUN_ID
    ).exists()
    assert context["control"].get_report(RUN_ID)["report"]["state"] == "CANCELLED"
    _assert_only_advertised_report(context, cancelled)


def test_terminal_read_retries_transient_cleanup_after_storage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _running_context(tmp_path)
    delete_sources = context["source_files"].delete_source_roots_after_fence
    attempts = 0

    def fail_once(run_id: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("storage unavailable")
        delete_sources(run_id)

    monkeypatch.setattr(
        context["source_files"], "delete_source_roots_after_fence", fail_once
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    failed = reconciler.reconcile(RUN_ID)
    assert failed["state"] == "FAILED"
    assert context["source_root"].exists()

    assert reconciler.reconcile(RUN_ID) == failed
    assert attempts == 2
    assert not context["source_root"].exists()
    assert context["control"].get_report(RUN_ID)["report"]["state"] == "FAILED"


def test_terminal_read_retries_report_history_cleanup_after_storage_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _running_context(tmp_path)
    prune_reports = context["run_files"].prune_report_history
    attempts = 0

    def fail_once(run_id: str, revision: int, digest: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("storage unavailable")
        prune_reports(run_id, revision, digest)

    monkeypatch.setattr(context["run_files"], "prune_report_history", fail_once)
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    failed = reconciler.reconcile(RUN_ID)
    report_root = context["artifact_root"] / "training/runs" / RUN_ID / "reports"
    assert failed["state"] == "FAILED"
    assert len(list(report_root.iterdir())) == 2

    assert reconciler.reconcile(RUN_ID) == failed
    assert attempts == 2
    _assert_only_advertised_report(context, failed)


def test_attachment_and_execution_deadlines_fail_without_retrying(
    tmp_path: Path,
) -> None:
    queued = _accepted_context(tmp_path / "queued")
    attachment_deadline = datetime.fromisoformat(
        queued["control"]
        .get_control(RUN_ID)["worker"]["call_attachment_deadline_at"]
        .replace("Z", "+00:00")
    )
    queued_reconciler = TrainingRunReconciler(
        queued["client"],
        queued["finalizer"],
        queued["terminalizer"],
        lambda _call_id: None,
        training_api_limits(),
        clock=lambda: attachment_deadline,
    )
    assert queued_reconciler.reconcile(RUN_ID)["state"] == "FAILED"

    running = _running_context(tmp_path / "running")
    call = _Call(pending=True)
    execution_deadline = datetime.fromisoformat(
        running["control"]
        .get_control(RUN_ID)["worker"]["execution_deadline_at"]
        .replace("Z", "+00:00")
    )
    running_reconciler = TrainingRunReconciler(
        running["client"],
        running["finalizer"],
        running["terminalizer"],
        lambda _call_id: call,
        training_api_limits(),
        clock=lambda: execution_deadline,
    )

    expired = running_reconciler.reconcile(RUN_ID)
    assert expired["state"] == "FAILED"
    assert call.cancelled is True


def test_candidate_first_observed_at_execution_deadline_cannot_succeed(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    deadline = datetime.fromisoformat(
        context["control"]
        .get_control(RUN_ID)["worker"]["execution_deadline_at"]
        .replace("Z", "+00:00")
    )
    call = _Call(pending=True)
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: call,
        training_api_limits(),
        clock=lambda: deadline,
    )

    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert call.cancelled is True
    assert context["result_store"].current.list_module_refs() == []


def test_candidate_that_rewrites_advertised_epoch_history_fails(
    tmp_path: Path,
) -> None:
    context = _running_context(tmp_path)
    running = copy.deepcopy(context["success_report"])
    running["state"] = "RUNNING"
    running["epochs"] = running["epochs"][:1]
    running["unavailable_metrics"] = []
    running.pop("model_selection")
    context["control"].advance_progress(
        RUN_ID,
        2,
        context["run_files"].publish_report_candidate(running)["report_digest"],
    )
    changed = copy.deepcopy(context["success_report"])
    changed["epochs"][0]["metrics"][0]["value"] += 0.01
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=changed,
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert failed["progress"]["completed_epochs"] == 1
    assert (
        "contradicts its advertised report history"
        in failed["result"]["error"]["message"]
    )
    assert context["result_store"].current.list_module_refs() == []


def test_success_report_publication_failure_happens_before_module_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _running_context(tmp_path)
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    publish = context["run_files"].publish_report_candidate

    def fail_success_report(report: dict) -> dict:
        if report["state"] == "SUCCEEDED":
            raise ArtifactBytesCorrupt("success report publication failed")
        return publish(report)

    monkeypatch.setattr(
        context["run_files"], "publish_report_candidate", fail_success_report
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert context["result_store"].current.list_module_refs() == []


@pytest.mark.parametrize("lost_response", [OSError, ValueError])
def test_postcommit_install_error_stays_finalizing_and_replays_exact_module(
    tmp_path: Path, lost_response: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _running_context(tmp_path)
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    attempts = 0
    install = context["result_store"].install_from_directory

    def lose_install_response(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        installed = install(*args, **kwargs)
        assert installed["module_ref"] == f"workspace/run-{RUN_ID}"
        raise lost_response("lost response after module commit")

    monkeypatch.setattr(
        context["result_store"], "install_from_directory", lose_install_response
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _call_id: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    assert reconciler.reconcile(RUN_ID)["state"] == "FINALIZING"
    assert context["control"].get_report(RUN_ID)["report"]["state"] == "FINALIZING"
    installed = context["result_store"].current.read_record(f"workspace/run-{RUN_ID}")
    assert (
        installed["content_digest"]
        == context["control"].get(RUN_ID)["finalization"]["module_content_digest"]
    )
    assert reconciler.reconcile(RUN_ID)["state"] == "SUCCEEDED"
    assert attempts == 1  # Recovery inspects the installed module, not the candidate.


@pytest.mark.parametrize("invalid_record", [False, True])
def test_bad_install_response_does_not_fail_a_valid_installed_module(
    tmp_path: Path, invalid_record: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _running_context(tmp_path)
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )

    install = context["result_store"].install_from_directory

    def wrong_installed_record(*args, **kwargs):
        record = copy.deepcopy(install(*args, **kwargs))
        if invalid_record:
            record.pop("content_digest")
        else:
            record["resolved_commit_sha"] = "e" * 40
        return record

    monkeypatch.setattr(
        context["result_store"], "install_from_directory", wrong_installed_record
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    assert reconciler.reconcile(RUN_ID)["state"] == "FINALIZING"
    assert context["control"].get_report(RUN_ID)["report"]["state"] == "FINALIZING"
    assert not context["control"].training_result_visible(RUN_ID)
    assert reconciler.reconcile(RUN_ID)["state"] == "SUCCEEDED"
    assert context["control"].training_result_visible(RUN_ID)


def test_different_stored_result_fails_and_preserves_existing_module(tmp_path: Path):
    context = _running_context(tmp_path)
    store = context["result_store"]
    source, ref, attestation = _training_result_source(tmp_path / "other-result")
    assert ref == f"workspace/run-{RUN_ID}"
    existing = store.install_from_directory(
        source,
        module_ref=ref,
        resolved_commit_sha=None,
        job_id=RUN_ID,
        publish=lambda: None,
        training_result_attestation=attestation,
    )
    context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    reconciler = TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )

    failed = reconciler.reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert failed["result"]["error"]["code"] == "INTERNAL_ERROR"
    assert store.current.revalidate(ref) == existing
    assert not context["control"].training_result_visible(RUN_ID)
    assert reconciler.reconcile(RUN_ID) == failed


def _accepted_context(tmp_path: Path) -> dict:
    limits = training_api_limits()
    private_limits = service_limits()
    artifact_root = tmp_path / "artifacts"
    source_store = install_module(tmp_path, workflow_recipe())
    request = request_for(source_store, result_form="complete_source_workflow")
    source_recipe = source_store.current.load_recipe("owner/workflow")
    request["field_profile"] = resolved_fixed_field_profile(
        source_recipe["models"][request["selected_model_id"]]
    )
    records = TrainingControlRecords(artifact_root, limits, private_limits)
    records.reserve_dataset(
        {
            "type": "retained",
            "dataset_id": request["dataset_id"],
            "dataset_digest": request["dataset_digest"],
            "retained_bytes": 100,
            "task_display_name": "Leydig training",
            "image_count": 2,
            "roi_count": 2,
            "created_at": "2026-09-04T11:00:00.000Z",
        }
    )
    state = TrainingRunState(
        records,
        limits,
        private_limits,
        clock=lambda: NOW,
        id_factory=lambda: uuid.UUID(RUN_ID),
    )
    run_files = TrainingRunFiles(artifact_root, limits)
    source_files = TrainingSourceFiles(artifact_root, 8 * 1024**2)
    result_files = TrainingResultFiles(artifact_root, limits, 8 * 1024**2)
    control = TrainingRunControl(state, run_files, source_files, limits)
    client = _ControlClient(control)
    result_store = ModuleStore(
        tmp_path / "result-modules",
        max_module_bytes=10_000_000,
        max_installed_bytes=20_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
        training_api_limits=limits,
    )

    def finalize_result(run_id: str):
        return finalize_training_result(
            result_store,
            run_files,
            result_files,
            client,
            run_id,
            lambda: None,
            limits,
        )

    finalizer = TrainingRunFinalizer(
        run_files,
        result_files,
        source_files,
        client,
        finalize_result,
        limits,
    )
    terminalizer = TrainingRunTerminalizer(run_files, client, limits)

    control.reserve(request)
    preparation = TrainingSourcePreparation(
        lambda: nullcontext(source_store.current), source_files, limits
    )
    with preparation.prepare(request) as prepared:
        marker_seed = {"starting_weight_digest": prepared.starting_weight_digest}
        success_report, weights = _success(tmp_path, request, marker_seed)
        initial_report = copy.deepcopy(success_report)
        initial_report["state"] = "QUEUED"
        initial_report["epochs"] = []
        del initial_report["model_selection"]
        report_digest = run_files.publish_report_candidate(initial_report)[
            "report_digest"
        ]
        marker = prepared.publish(RUN_ID, report_digest)
    accepted = control.accept(request, trainer_build(), 3, 3_600, report_digest)[
        "accepted"
    ]
    return {
        "artifact_root": artifact_root,
        "accepted": accepted,
        "request": request,
        "marker": marker,
        "success_report": success_report,
        "weights": weights,
        "source_root": artifact_root / "training/runs/pending" / RUN_ID,
        "control": control,
        "state": state,
        "client": client,
        "run_files": run_files,
        "source_files": source_files,
        "result_files": result_files,
        "result_store": result_store,
        "finalize_result": finalize_result,
        "finalizer": finalizer,
        "terminalizer": terminalizer,
    }


def _running_context(tmp_path: Path) -> dict:
    context = _accepted_context(tmp_path)
    context["control"].attach_worker(RUN_ID, "fc-training")
    context["control"].claim_worker(RUN_ID, trainer_build())
    return context


def _finalizing_context(tmp_path: Path) -> dict:
    context = _running_context(tmp_path)
    context["candidate"] = context["result_files"].publish_candidate(
        RUN_ID,
        source_root=context["source_root"],
        source_marker=context["marker"],
        selected_weights=context["weights"],
        succeeded_report=context["success_report"],
        request=context["request"],
        project_model=project_dinov3_training_model,
    )
    # Exercise the real adoption path, stopping at the serialized-writer boundary.
    adoption = TrainingRunFinalizer(
        context["run_files"],
        context["result_files"],
        context["source_files"],
        context["client"],
        context["control"].get,
        training_api_limits(),
    )
    assert adoption.reconcile_candidate(RUN_ID)["state"] == "FINALIZING"
    return context


def _assert_only_advertised_report(context: dict, run: dict) -> None:
    report_root = context["artifact_root"] / "training/runs" / RUN_ID / "reports"
    assert sorted(path.name for path in report_root.iterdir()) == [
        f"{run['report_revision']}.json"
    ]
