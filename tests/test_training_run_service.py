from __future__ import annotations

import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.runs.control import TrainingRunControl
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from histoforge.cloud.training.runs.service import TrainingRunService
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_preparation import TrainingSourcePreparation
from histoforge.cloud.training.runs.terminalizer import TrainingRunTerminalizer
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_state import service_limits
from tests.test_training_report import queued_report
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request
from tests.test_training_run_state import RUN_ID, SECOND_RUN_ID, make_state
from tests.test_training_source_preparation import (
    install_module,
    request_for,
    workflow_recipe,
)


class _ControlClient:
    def __init__(self, control: TrainingRunControl) -> None:
        self.control = control
        self._lock = threading.Lock()

    def lookup_run(self, request: dict) -> dict | None:
        with self._lock:
            return self.control.lookup(request)

    def reserve_run(self, request: dict) -> dict:
        with self._lock:
            return self.control.reserve(request)

    def claim_abandoned_run_admission(self) -> dict | None:
        with self._lock:
            return self.control.claim_abandoned_admission()

    def training_run_transient_cleanup_scopes(
        self, run_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        with self._lock:
            return self.control.transient_cleanup_scopes(run_ids)

    def get_run_control(self, run_id: str) -> dict:
        with self._lock:
            return self.control.get_control(run_id)

    def fail_run(
        self,
        run_id: str,
        expected_state: str,
        report_revision: int,
        report_digest: str,
    ) -> dict:
        with self._lock:
            return self.control.fail(
                run_id,
                expected_state,
                report_revision,
                report_digest,
            )

    def accept_run(
        self,
        request: dict,
        build: dict,
        total_epochs: int,
        execution_seconds: int,
        report_digest: str,
    ) -> dict:
        with self._lock:
            return self.control.accept(
                request,
                build,
                total_epochs,
                execution_seconds,
                report_digest,
            )

    def attach_run_worker(self, run_id: str, call_id: str) -> dict:
        with self._lock:
            return self.control.attach_worker(run_id, call_id)


@dataclass
class _PreparedRun:
    request: dict
    starting_weight_digest: str
    trainer_build: dict
    total_epochs: int = 3
    trainer_execution_seconds: int = 3_600
    worker_resources: dict = field(
        default_factory=lambda: {
            "gpu": "L4",
            "cpu": 4.0,
            "memory_mib": 16_384,
        }
    )

    def initial_report(self, run_id: str) -> dict:
        report = copy.deepcopy(queued_report())
        report.update(
            {
                "run_id": run_id,
                "dataset_digest": self.request["dataset_digest"],
                "split_plan": self.request["split_plan"],
                "source_module": self.request["source_module"],
                "selected_model_id": self.request["selected_model_id"],
                "starting_weight_digest": self.starting_weight_digest,
                "result_form": self.request["result_form"],
                "result_display_name": self.request["result_display_name"],
                "result_inference_gpu": self.request["result_inference_gpu"],
                "trainer_build": self.trainer_build,
                "training_seed": self.request["training_seed"],
                "trainer_settings": self.request["trainer_settings"],
                "field_profile": self.request["field_profile"],
                "augmentation": self.request["augmentation"],
            }
        )
        report["trainer_configuration"]["preprocessing"]["profile_id"] = self.request[
            "field_profile"
        ]["preprocessing"]
        return report


class _BrokenPreparedRun(_PreparedRun):
    def initial_report(self, run_id: str) -> dict:
        raise ValueError(f"broken report for {run_id}")


class _Call:
    def __init__(self, object_id: Any = "fc-training-1") -> None:
        self.object_id = object_id
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_submit_publishes_accepts_cleans_and_dispatches_once(tmp_path: Path) -> None:
    service, control, files, sources, request, prepare_calls, spawned = _service(
        tmp_path
    )

    accepted = service.submit(request)

    run = control.control.get(accepted["run_id"])
    assert accepted == {
        "run_id": RUN_ID,
        "request_digest": run["request_digest"],
    }
    assert run["state"] == "RUNNING"
    assert prepare_calls == [request]
    assert len(spawned) == 1
    assert spawned[0][0] == RUN_ID
    assert spawned[0][2] == {
        "gpu": "L4",
        "cpu": 4.0,
        "memory_mib": 16_384,
    }
    assert spawned[0][3] == 3_600
    assert sources.require_ready(RUN_ID)["run_id"] == RUN_ID
    assert files.read_report(RUN_ID, 1)["report"]["state"] == "QUEUED"
    assert not _source_candidate(tmp_path, RUN_ID).exists()
    assert not _report_candidates(tmp_path, RUN_ID).exists()

    assert service.submit(request) == accepted
    assert prepare_calls == [request]
    assert len(spawned) == 1


def test_concurrent_identical_submissions_accept_and_dispatch_once(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    prepared_requests: list[dict] = []

    def concurrent_preflight(request, _recipe, starting_weight_digest):
        prepared_requests.append(request)
        barrier.wait(timeout=5)
        return _PreparedRun(request, starting_weight_digest, trainer_build())

    service, control, _files, _sources, request, _calls, spawned = _service(
        tmp_path,
        prepare_run=concurrent_preflight,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                service.submit,
                [copy.deepcopy(request), copy.deepcopy(request)],
            )
        )

    assert results[0] == results[1]
    assert results[0]["run_id"] == RUN_ID
    assert len(prepared_requests) == 2
    assert len(spawned) == 1
    assert control.control.get(RUN_ID)["state"] == "RUNNING"


def test_blocked_source_copy_does_not_block_training_control_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, control, _files, sources, request, _calls, _spawned = _service(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    publish_candidate = sources.publish_candidate

    def blocked_publish(marker, source_paths):
        entered.set()
        assert release.wait(timeout=5)
        return publish_candidate(marker, source_paths)

    monkeypatch.setattr(sources, "publish_candidate", blocked_publish)
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(service.submit, request)
        try:
            assert entered.wait(timeout=5)
            reservation = control.lookup_run(request)
            assert reservation is not None
            assert reservation["type"] == "preparing"
        finally:
            release.set()
        accepted = pending.result(timeout=5)

    assert accepted["run_id"] == RUN_ID


def test_invalid_preflight_does_not_reserve_run(tmp_path: Path) -> None:
    service, _control, _files, _sources, request, _calls, _spawned = _service(
        tmp_path,
        prepare_run=lambda *_args: (_ for _ in ()).throw(
            ValueError("unsupported trainer")
        ),
    )

    with pytest.raises(GatewayFailure) as raised:
        service.submit(request)

    assert raised.value.error["code"] == "VALIDATION_ERROR"
    assert _control.control.lookup(request) is None


def test_stale_source_identity_is_a_closed_module_mismatch(tmp_path: Path) -> None:
    service, control, _files, _sources, request, _calls, _spawned = _service(tmp_path)
    request["source_module"]["content_digest"] = "f" * 64

    with pytest.raises(GatewayFailure) as raised:
        service.submit(request)

    assert raised.value.error["code"] == "MODULE_IDENTITY_MISMATCH"
    assert control.control.lookup(request) is None


def test_report_builder_defect_is_not_misreported_as_invalid_user_input(
    tmp_path: Path,
) -> None:
    def broken_preflight(request, _recipe, starting_weight_digest):
        return _BrokenPreparedRun(request, starting_weight_digest, trainer_build())

    service, _control, _files, _sources, request, _calls, _spawned = _service(
        tmp_path,
        prepare_run=broken_preflight,
    )

    with pytest.raises(ValueError, match="broken report"):
        service.submit(request)


def test_attachment_failure_cancels_the_spawned_call(tmp_path: Path) -> None:
    service, control, _files, _sources, request, _calls, spawned = _service(tmp_path)

    def fail_attachment(_run_id: str, _call_id: str) -> dict:
        raise OSError("lane unavailable")

    control.attach_run_worker = fail_attachment
    with pytest.raises(GatewayFailure) as raised:
        service.submit(request)

    assert raised.value.error["code"] == "SERVICE_UNAVAILABLE"
    assert spawned[0][1].cancelled is True
    assert control.control.get(RUN_ID)["state"] == "QUEUED"


def test_closed_provider_dispatch_failure_is_preserved_after_acceptance(
    tmp_path: Path,
) -> None:
    failure = GatewayFailure(
        "PROVIDER_CREDIT_EXHAUSTED",
        "safe credit guidance",
        resource_id=RUN_ID,
    )
    service, control, _files, _sources, request, _calls, _spawned = _service(
        tmp_path,
        spawn_run=lambda _run_id, _resources, _timeout: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(GatewayFailure) as raised:
        service.submit(request)

    assert raised.value is failure
    assert raised.value.error == {
        "schema_version": 1,
        "code": "PROVIDER_CREDIT_EXHAUSTED",
        "message": "safe credit guidance",
        "resource_id": RUN_ID,
    }
    assert control.control.get(RUN_ID)["state"] == "QUEUED"


def test_dispatch_attaches_before_candidate_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)]
    service, control, _files, sources, request, _calls, spawned = _service(
        tmp_path,
        now=now,
    )
    discard = sources.discard_candidate

    def delayed_discard(run_id: str, report_digest: str) -> None:
        now[0] += timedelta(seconds=61)
        discard(run_id, report_digest)

    monkeypatch.setattr(sources, "discard_candidate", delayed_discard)

    accepted = service.submit(request)

    assert control.control.get(accepted["run_id"])["state"] == "RUNNING"
    assert spawned[0][1].cancelled is False


def test_invalid_spawn_identity_cancels_call_and_fails_closed(tmp_path: Path) -> None:
    invalid_call = _Call("")
    service, _control, _files, _sources, request, _calls, _spawned = _service(
        tmp_path,
        spawn_run=lambda _run_id, _resources, _timeout: invalid_call,
    )

    with pytest.raises(GatewayFailure) as raised:
        service.submit(request)

    assert raised.value.error["code"] == "INTERNAL_ERROR"
    assert invalid_call.cancelled is True


def test_abandoned_admission_is_fenced_and_its_roots_are_removed(
    tmp_path: Path,
) -> None:
    now = [datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)]
    private_limits = service_limits(pending_snapshot_cleanup_age_seconds=10)
    service, control, _files, _sources, request, _calls, _spawned = _service(
        tmp_path,
        now=now,
        ids=[RUN_ID, SECOND_RUN_ID],
        private_limits=private_limits,
    )
    old = run_request()
    old["source_module"] = request["source_module"]
    control.reserve_run(old)
    source_root = _artifact_root(tmp_path) / "training" / "runs" / "pending" / RUN_ID
    report_root = _artifact_root(tmp_path) / "training" / "runs" / RUN_ID / "reports"
    source_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (source_root / "orphan").write_bytes(b"source")
    (report_root / "1.json").write_bytes(b"report")

    now[0] += timedelta(seconds=10)
    request["training_seed"] += 1
    accepted = service.submit(request)

    assert accepted["run_id"] == SECOND_RUN_ID
    assert not source_root.exists()
    assert not report_root.exists()


def test_cleanup_crash_leaves_discoverable_roots_for_the_next_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)]
    private_limits = service_limits(pending_snapshot_cleanup_age_seconds=10)
    service, control, _files, sources, request, _calls, _spawned = _service(
        tmp_path,
        now=now,
        ids=[RUN_ID, SECOND_RUN_ID],
        private_limits=private_limits,
    )
    old = run_request()
    old["source_module"] = request["source_module"]
    control.reserve_run(old)
    source_root = _artifact_root(tmp_path) / "training" / "runs" / "pending" / RUN_ID
    report_root = _artifact_root(tmp_path) / "training" / "runs" / RUN_ID / "reports"
    source_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (source_root / "orphan").write_bytes(b"source")
    (report_root / "1.json").write_bytes(b"report")
    now[0] += timedelta(seconds=10)
    request["training_seed"] += 1

    original_delete = sources.delete_source_roots_after_fence
    failed = False

    def fail_once(run_id: str) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated cleanup interruption")
        original_delete(run_id)

    monkeypatch.setattr(sources, "delete_source_roots_after_fence", fail_once)
    with pytest.raises(OSError, match="simulated cleanup interruption"):
        service.submit(request)

    assert control.control.lookup(old) is None
    assert source_root.exists()
    accepted = service.submit(request)

    assert accepted["run_id"] == SECOND_RUN_ID
    assert not source_root.exists()
    assert not report_root.exists()


def test_cleanup_retries_a_report_root_after_source_deletion_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)]
    private_limits = service_limits(pending_snapshot_cleanup_age_seconds=10)
    service, control, files, _sources, request, _calls, _spawned = _service(
        tmp_path,
        now=now,
        ids=[RUN_ID, SECOND_RUN_ID],
        private_limits=private_limits,
    )
    old = run_request()
    old["source_module"] = request["source_module"]
    control.reserve_run(old)
    source_root = _artifact_root(tmp_path) / "training" / "runs" / "pending" / RUN_ID
    report_root = _artifact_root(tmp_path) / "training" / "runs" / RUN_ID / "reports"
    source_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (source_root / "orphan").write_bytes(b"source")
    (report_root / "1.json").write_bytes(b"report")
    now[0] += timedelta(seconds=10)
    request["training_seed"] += 1

    original_delete = files.delete_reports
    failed = False

    def fail_once(run_id: str) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated report cleanup interruption")
        original_delete(run_id)

    monkeypatch.setattr(files, "delete_reports", fail_once)
    with pytest.raises(OSError, match="simulated report cleanup interruption"):
        service.submit(request)

    assert not source_root.exists()
    assert report_root.exists()
    accepted = service.submit(request)

    assert accepted["run_id"] == SECOND_RUN_ID
    assert not report_root.exists()


def test_cleanup_removes_an_unowned_training_result_candidate(tmp_path: Path) -> None:
    service, _control, _files, _sources, request, _calls, _spawned = _service(tmp_path)
    orphan_id = SECOND_RUN_ID
    candidate = (
        _artifact_root(tmp_path) / "training" / "runs" / "result-candidates" / orphan_id
    )
    candidate.mkdir(parents=True)
    (candidate / "orphan").write_bytes(b"candidate")

    service.submit(request)

    assert not candidate.exists()


def test_cleanup_preserves_a_result_candidate_owned_by_an_active_run(
    tmp_path: Path,
) -> None:
    service, _control, _files, _sources, request, _calls, _spawned = _service(tmp_path)
    accepted = service.submit(request)
    candidate = (
        _artifact_root(tmp_path)
        / "training"
        / "runs"
        / "result-candidates"
        / accepted["run_id"]
    )
    candidate.mkdir(parents=True)
    (candidate / "owned").write_bytes(b"candidate")

    service._cleanup_transient_roots()

    assert candidate.exists()


def test_admission_cleanup_prunes_interrupted_terminal_report_history(
    tmp_path: Path,
) -> None:
    service, control, files, _sources, request, _calls, _spawned = _service(tmp_path)
    service.submit(request)
    failed = TrainingRunTerminalizer(files, control, training_api_limits()).fail(
        control.get_run_control(RUN_ID),
        {
            "schema_version": 1,
            "code": "RUNNER_FAILED",
            "message": "training failed",
            "resource_id": RUN_ID,
        },
    )["run"]
    report_root = _artifact_root(tmp_path) / "training/runs" / RUN_ID / "reports"
    assert sorted(path.name for path in report_root.iterdir()) == ["1.json", "2.json"]

    service._cleanup_transient_roots()

    assert sorted(path.name for path in report_root.iterdir()) == [
        f"{failed['report_revision']}.json"
    ]


def _service(
    tmp_path: Path,
    *,
    prepare_run=None,
    spawn_run=None,
    now: list[datetime] | None = None,
    ids: list[str] | None = None,
    private_limits: dict | None = None,
):
    root = _artifact_root(tmp_path)
    limits = training_api_limits()
    selected_service_limits = private_limits or service_limits()
    state, _records = make_state(
        root,
        now=now,
        ids=ids,
        service_overrides={
            key: value
            for key, value in selected_service_limits.items()
            if value != service_limits()[key]
        },
    )
    files = TrainingRunFiles(root, limits)
    sources = TrainingSourceFiles(root, 8 * 1024**2)
    results = TrainingResultFiles(root, limits, 8 * 1024**2)
    control = _ControlClient(TrainingRunControl(state, files, sources, limits))
    store = install_module(tmp_path, workflow_recipe())
    request = request_for(store)
    source_preparation = TrainingSourcePreparation(
        lambda: nullcontext(store.current), sources, limits
    )
    prepare_calls: list[dict] = []

    def default_prepare(request_value, _recipe, starting_weight_digest):
        prepare_calls.append(request_value)
        return _PreparedRun(request_value, starting_weight_digest, trainer_build())

    spawned: list[tuple[str, _Call, dict, int]] = []

    def default_spawn(run_id: str, resources: dict, timeout: int) -> _Call:
        call = _Call()
        spawned.append((run_id, call, resources, timeout))
        return call

    return (
        TrainingRunService(
            source_preparation,
            files,
            sources,
            results,
            control,
            prepare_run or default_prepare,
            spawn_run or default_spawn,
            limits,
            selected_service_limits,
        ),
        control,
        files,
        sources,
        request,
        prepare_calls,
        spawned,
    )


def _artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def _source_candidate(tmp_path: Path, run_id: str) -> Path:
    return _artifact_root(tmp_path) / "training" / "runs" / "source-candidates" / run_id


def _report_candidates(tmp_path: Path, run_id: str) -> Path:
    return _artifact_root(tmp_path) / "training" / "runs" / "report-candidates" / run_id
