from __future__ import annotations

import copy
from pathlib import Path

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.runs.access import TrainingRunAccess
from histoforge.cloud.training.runs.control import TrainingRunControl
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.terminalizer import TrainingRunTerminalizer
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_control import (
    accept,
    make_control,
    publish,
    running_report,
)


class _ControlClient:
    def __init__(self, control: TrainingRunControl) -> None:
        self.control = control

    def get_run(self, run_id: str) -> dict:
        return self.control.get(run_id)

    def get_run_control(self, run_id: str) -> dict:
        return self.control.get_control(run_id)

    def page_run_records(self, limit: int, cursor: str | None) -> list[dict]:
        return self.control.page_records(limit, cursor)

    def cancel_run(self, run_id: str, revision: int, digest: str) -> dict:
        return self.control.cancel(run_id, revision, digest)

    def begin_run_delete(self, run_id: str) -> dict | None:
        return self.control.begin_delete(run_id)

    def finish_run_delete(self, run_id: str) -> None:
        self.control.finish_delete(run_id)


class _Call:
    def __init__(self, *, broken: bool = False) -> None:
        self.broken = broken
        self.cancelled = False

    def cancel(self) -> None:
        if self.broken:
            raise OSError("provider unavailable")
        self.cancelled = True


class _Reconciler:
    def __init__(self, control: TrainingRunControl) -> None:
        self.control = control

    def reconcile(self, run_id: str) -> dict:
        return self.control.get(run_id)


def _access(
    control: TrainingRunControl,
    files: TrainingRunFiles,
    calls: dict[str, _Call] | None = None,
) -> TrainingRunAccess:
    available = calls or {}
    client = _ControlClient(control)
    return TrainingRunAccess(
        files,
        client,
        _Reconciler(control),
        TrainingRunTerminalizer(files, client, training_api_limits()),
        available.__getitem__,
        training_api_limits(),
    )


def test_get_page_and_report_return_advertised_run_evidence(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    access = _access(control, files)

    assert access.get(run["run_id"]) == run
    assert access.report(run["run_id"])["report"] == queued_report()
    page = access.page(10)
    assert page == {
        "items": [
            {
                "run_id": run["run_id"],
                "result_display_name": "Fine-tuned Leydig workflow",
                "source_module_ref": "histoforge/leydig",
                "selected_model_id": "leydig_nuclear",
                "result_form": "complete_source_workflow",
                "state": "QUEUED",
                "progress": {"completed_epochs": 0, "total_epochs": 3},
                "report_revision": 1,
                "created_at": run["created_at"],
                "updated_at": run["updated_at"],
            }
        ]
    }


def test_page_reads_report_only_after_state_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    client = _ControlClient(control)
    access = TrainingRunAccess(
        files,
        client,
        _Reconciler(control),
        TrainingRunTerminalizer(files, client, training_api_limits()),
        {}.__getitem__,
        training_api_limits(),
    )
    events = []
    original_page = client.page_run_records
    original_report = files.read_report

    def page_records(limit: int, cursor: str | None):
        events.append("state snapshot")
        return original_page(limit, cursor)

    def read_report(run_id: str, revision: int):
        events.append("report file")
        return original_report(run_id, revision)

    monkeypatch.setattr(client, "page_run_records", page_records)
    monkeypatch.setattr(files, "read_report", read_report)

    assert access.page(10)["items"][0]["run_id"] == run["run_id"]
    assert events == ["state snapshot", "report file"]


def test_page_uses_the_report_revision_committed_during_reconciliation(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    accept(control, files, sources, tmp_path)

    class _AdvancingReconciler:
        def __init__(self) -> None:
            self.advanced = False

        def reconcile(self, run_id: str) -> dict:
            if not self.advanced:
                control.attach_worker(run_id, "fc-progress")
                control.claim_worker(run_id, trainer_build())
                control.advance_progress(
                    run_id,
                    2,
                    publish(files, running_report()),
                )
                self.advanced = True
            return control.get(run_id)

    access = TrainingRunAccess(
        files,
        _ControlClient(control),
        _AdvancingReconciler(),
        TrainingRunTerminalizer(files, _ControlClient(control), training_api_limits()),
        {}.__getitem__,
        training_api_limits(),
    )

    item = access.page(10)["items"][0]

    assert item["state"] == "RUNNING"
    assert item["report_revision"] == 2
    assert item["progress"] == {"completed_epochs": 1, "total_epochs": 3}
    assert item["primary_metric"]["epoch"] == 1


def test_cancel_commits_report_fence_before_stopping_attached_call(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    call = _Call()
    control.attach_worker(run["run_id"], "fc-123")
    access = _access(control, files, {"fc-123": call})

    cancelled = access.cancel(run["run_id"])

    assert cancelled["state"] == "CANCELLED"
    assert call.cancelled is True
    response = access.report(run["run_id"])
    assert response["report"]["state"] == "CANCELLED"
    assert response["report_digest"] == cancelled["result"]["report_digest"]
    call.cancelled = False
    assert access.cancel(run["run_id"]) == cancelled
    assert call.cancelled is True


def test_cancel_uses_latest_progress_and_worker_attachment(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-latest")
    control.claim_worker(run["run_id"], trainer_build())
    control.advance_progress(run["run_id"], 2, publish(files, running_report()))
    call = _Call()
    access = _access(control, files, {"fc-latest": call})

    cancelled = access.cancel(run["run_id"])

    assert cancelled["state"] == "CANCELLED"
    assert cancelled["progress"]["completed_epochs"] == 1
    assert access.report(run["run_id"])["report"]["epochs"][-1]["epoch"] == 1
    assert call.cancelled is True


def test_committed_cancellation_survives_physical_call_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    access = _access(control, files, {"fc-123": _Call(broken=True)})

    cancelled = access.cancel(run["run_id"])

    assert cancelled["state"] == "CANCELLED"
    assert "physical training-call cancellation failed" in caplog.text


def test_committed_cancellation_survives_candidate_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    access = _access(control, files)

    monkeypatch.setattr(
        files,
        "discard_report_candidate",
        lambda *_args: (_ for _ in ()).throw(OSError("storage unavailable")),
    )
    cancelled = access.cancel(run["run_id"])

    assert cancelled["state"] == "CANCELLED"
    assert "candidate cleanup failed" in caplog.text


def test_cancel_rejects_terminal_non_cancelled_run(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    failed_report = copy.deepcopy(queued_report())
    failed_report["state"] = "FAILED"
    failed_report["failure"] = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training failed",
    }
    control.fail(run["run_id"], "QUEUED", 2, publish(files, failed_report))
    access = _access(control, files)

    with pytest.raises(GatewayFailure) as raised:
        access.cancel(run["run_id"])

    assert raised.value.error["code"] == "TRAINING_STATE_CONFLICT"


def test_delete_is_terminal_only_and_preserves_row_when_report_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    access = _access(control, files)

    with pytest.raises(GatewayFailure, match="active training run cannot be cleared"):
        access.delete(run["run_id"])

    access.cancel(run["run_id"])

    def fail_cleanup(_run_id: str) -> None:
        raise OSError("storage unavailable")

    monkeypatch.setattr(files, "delete_reports", fail_cleanup)
    with pytest.raises(OSError, match="storage unavailable"):
        access.delete(run["run_id"])
    assert access.get(run["run_id"])["state"] == "CANCELLED"

    monkeypatch.undo()
    access.delete(run["run_id"])
    access.delete(run["run_id"])
    with pytest.raises(GatewayFailure) as raised:
        access.get(run["run_id"])
    assert raised.value.error["code"] == "TRAINING_RUN_NOT_FOUND"
