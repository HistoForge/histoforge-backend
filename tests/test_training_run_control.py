from __future__ import annotations

import copy
from pathlib import Path

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactBytesMissing
from histoforge.cloud.training.runs.control import TrainingRunControl
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.terminalizer import TrainingRunTerminalizer
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report, succeeded_report
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request
from tests.test_training_run_state import make_state
from tests.test_training_source_files import source_paths
from tests.test_training_source_snapshot import source_ready_marker


def make_control(
    tmp_path: Path,
) -> tuple[TrainingRunControl, TrainingRunFiles, TrainingSourceFiles]:
    state, _ = make_state(tmp_path)
    files = TrainingRunFiles(tmp_path, training_api_limits())
    sources = TrainingSourceFiles(tmp_path, 8 * 1024**2)
    return (
        TrainingRunControl(state, files, sources, training_api_limits()),
        files,
        sources,
    )


def publish(files: TrainingRunFiles, report: dict) -> str:
    return files.publish_report_candidate(report)["report_digest"]


def accept(
    control: TrainingRunControl,
    files: TrainingRunFiles,
    sources: TrainingSourceFiles,
    tmp_path: Path,
) -> dict:
    digest = publish(files, queued_report())
    publish_initial_source(control, sources, tmp_path, digest)
    return control.accept(run_request(), trainer_build(), 3, 3_600, digest)["accepted"]


def publish_initial_source(
    control: TrainingRunControl,
    sources: TrainingSourceFiles,
    tmp_path: Path,
    report_digest: str,
) -> dict:
    reservation = control.reserve(run_request())
    marker = source_ready_marker()
    marker["run_id"] = reservation["admission"]["run_id"]
    marker["report_digest"] = report_digest
    return sources.publish_candidate(marker, source_paths(tmp_path))


def running_report(*, completed_epochs: int = 1) -> dict:
    report = succeeded_report()
    report["state"] = "RUNNING"
    report["epochs"] = report["epochs"][:completed_epochs]
    del report["model_selection"]
    return report


def report_path(root: Path, run_id: str, revision: int) -> Path:
    return root / "training" / "runs" / run_id / "reports" / f"{revision}.json"


def test_accept_promotes_initial_report_before_run_visibility(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    digest = publish(files, queued_report())
    publish_initial_source(control, sources, tmp_path, digest)
    result = control.accept(run_request(), trainer_build(), 3, 3_600, digest)
    run = result["accepted"]

    assert result["should_dispatch"] is True
    assert run["state"] == "QUEUED"
    response = control.get_report(run["run_id"])
    assert response["report_revision"] == 1
    assert response["report"] == queued_report()
    assert control.accept(run_request(), trainer_build(), 3, 3_600, digest) == {
        "accepted": run,
        "should_dispatch": False,
    }


@pytest.mark.parametrize(
    "field,value",
    [("result_display_name", "Different"), ("result_inference_gpu", "L4")],
)
def test_invalid_initial_report_leaves_only_reusable_admission_and_candidate(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    control, files, sources = make_control(tmp_path)
    invalid = queued_report()
    invalid[field] = value
    digest = publish(files, invalid)
    publish_initial_source(control, sources, tmp_path, digest)

    with pytest.raises(GatewayFailure, match=f"initial report {field}"):
        control.accept(run_request(), trainer_build(), 3, 3_600, digest)

    reserved = control.reserve(run_request())
    assert reserved["type"] == "preparing"
    assert not report_path(tmp_path, reserved["admission"]["run_id"], 1).exists()


def test_accept_requires_existing_admission_and_complete_source_snapshot(
    tmp_path: Path,
) -> None:
    control, files, _ = make_control(tmp_path)
    digest = publish(files, queued_report())

    with pytest.raises(GatewayFailure, match="was not reserved"):
        control.accept(run_request(), trainer_build(), 3, 3_600, digest)
    assert control.reserve(run_request())["type"] == "preparing"

    with pytest.raises(ArtifactBytesMissing):
        control.accept(run_request(), trainer_build(), 3, 3_600, digest)


def test_accept_binds_source_marker_report_and_starting_weights(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    report = queued_report()
    report["starting_weight_digest"] = "f" * 64
    digest = publish(files, report)
    publish_initial_source(control, sources, tmp_path, digest)

    with pytest.raises(GatewayFailure, match="starting_weight_digest differs"):
        control.accept(run_request(), trainer_build(), 3, 3_600, digest)


def test_accept_requires_source_candidate_for_the_same_report(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    digest = publish(files, queued_report())
    publish_initial_source(control, sources, tmp_path, "f" * 64)

    with pytest.raises(ArtifactBytesMissing):
        control.accept(run_request(), trainer_build(), 3, 3_600, digest)


def test_acceptance_uses_only_the_committed_source_marker_in_the_state_cut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, files, sources = make_control(tmp_path)
    digest = publish(files, queued_report())
    publish_initial_source(control, sources, tmp_path, digest)

    def reject_full_source_read(_run_id: str) -> dict:
        raise AssertionError("acceptance must not hash source files in the state lane")

    monkeypatch.setattr(sources, "require_ready", reject_full_source_read)

    assert (
        control.accept(run_request(), trainer_build(), 3, 3_600, digest)["accepted"][
            "state"
        ]
        == "QUEUED"
    )


def test_later_build_can_finish_after_earlier_source_promotion(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    reservation = control.reserve(run_request())
    run_id = reservation["admission"]["run_id"]

    first_report = queued_report()
    first_digest = publish(files, first_report)
    first_marker = source_ready_marker()
    first_marker["run_id"] = run_id
    first_marker["report_digest"] = first_digest
    sources.publish_candidate(first_marker, source_paths(tmp_path))
    sources.promote_candidate(run_id, first_digest)

    later_build = {**trainer_build(), "artifact_digest": "c" * 64}
    later_report = queued_report()
    later_report["trainer_build"] = later_build
    later_digest = publish(files, later_report)
    later_marker = {**first_marker, "report_digest": later_digest}
    sources.publish_candidate(later_marker, source_paths(tmp_path))

    accepted = control.accept(run_request(), later_build, 3, 3_600, later_digest)[
        "accepted"
    ]

    assert accepted["trainer_build"] == later_build
    assert sources.require_ready(run_id) == later_marker
    assert control.get_report(run_id)["report"] == later_report


def test_progress_promotes_report_then_advances_record(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    report = running_report()

    updated = control.advance_progress(run["run_id"], 2, publish(files, report))

    assert updated["progress"]["completed_epochs"] == 1
    response = control.get_report(run["run_id"])
    assert response["report_revision"] == 2
    assert response["report"]["epochs"][-1]["epoch"] == 1


def test_progress_exact_replay_does_not_publish_another_revision(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    report = running_report()
    digest = publish(files, report)

    first = control.advance_progress(run["run_id"], 2, digest)
    replay = control.advance_progress(run["run_id"], 2, digest)

    assert replay == first
    assert replay["report_revision"] == 2
    assert not report_path(tmp_path, run["run_id"], 3).exists()


def test_running_and_finalizing_state_survive_owner_reconstruction(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    progress = running_report()
    control.advance_progress(run["run_id"], 2, publish(files, progress))

    running_control, running_files, _ = make_control(tmp_path)
    assert running_control.get(run["run_id"])["state"] == "RUNNING"
    assert running_control.get_report(run["run_id"])["report"] == progress
    complete = running_report(completed_epochs=3)
    running_control.advance_progress(run["run_id"], 3, publish(running_files, complete))
    finalizing_report = succeeded_report()
    finalizing_report["state"] = "FINALIZING"
    finalizing_digest = publish(running_files, finalizing_report)
    finalizing = running_control.begin_finalization(
        run["run_id"],
        4,
        finalizing_digest,
        succeeded_report(),
        "e" * 64,
    )

    finalizing_control, _, _ = make_control(tmp_path)
    assert finalizing_control.get(run["run_id"]) == finalizing
    assert finalizing_control.get_report(run["run_id"])["report"] == (finalizing_report)


def test_invalid_transition_does_not_promote_candidate(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    digest = publish(files, running_report())

    with pytest.raises(GatewayFailure, match="no longer accepted"):
        control.advance_progress(run["run_id"], 2, digest)

    assert not report_path(tmp_path, run["run_id"], 2).exists()
    assert control.get_report(run["run_id"])["report"]["state"] == "QUEUED"


def test_report_cannot_rewrite_immutable_run_or_epoch_history(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    first = running_report()
    control.advance_progress(run["run_id"], 2, publish(files, first))

    changed = running_report(completed_epochs=2)
    changed["training_counts"]["target_object_counts"] = {"Leydig cell": 51}
    with pytest.raises(GatewayFailure, match="training_counts changed"):
        control.advance_progress(run["run_id"], 3, publish(files, changed))

    changed = running_report(completed_epochs=2)
    changed["min_instance_pixels"] = 0
    with pytest.raises(GatewayFailure, match="report min_instance_pixels differs"):
        control.advance_progress(run["run_id"], 3, publish(files, changed))

    changed = running_report(completed_epochs=2)
    changed["epochs"][0]["metrics"][0]["value"] = 0.9
    with pytest.raises(GatewayFailure, match="changed completed epochs"):
        control.advance_progress(run["run_id"], 3, publish(files, changed))


def test_report_read_rejects_control_progress_that_points_to_stale_bytes(
    tmp_path: Path,
) -> None:
    state, records = make_state(tmp_path)
    files = TrainingRunFiles(tmp_path, training_api_limits())
    sources = TrainingSourceFiles(tmp_path, 8 * 1024**2)
    control = TrainingRunControl(state, files, sources, training_api_limits())
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    report = running_report()
    control.advance_progress(run["run_id"], 2, publish(files, report))

    persisted = records.get_run(run["run_id"])
    persisted["run"]["report_revision"] = 1
    records.replace_run(persisted)

    with pytest.raises(GatewayFailure, match="report progress differs"):
        control.get_report(run["run_id"])


def test_result_inference_gpu_cannot_change_in_progress_or_stored_reports(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    changed = running_report()
    changed["result_inference_gpu"] = "L4"
    digest = publish(files, changed)

    with pytest.raises(GatewayFailure, match="report result_inference_gpu differs"):
        control.advance_progress(run["run_id"], 2, digest)
    assert control.get(run["run_id"])["report_revision"] == 1
    assert not report_path(tmp_path, run["run_id"], 2).exists()

    # Simulate retained bytes written without the run owner's binding checks.
    changed = queued_report()
    changed["result_inference_gpu"] = "L4"
    files.promote_report(run["run_id"], 1, publish(files, changed))
    with pytest.raises(GatewayFailure, match="report result_inference_gpu differs"):
        control.get_report(run["run_id"])


def test_cancellation_report_and_state_commit_together(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    report = queued_report()
    report["state"] = "CANCELLED"
    digest = publish(files, report)

    cancelled = control.cancel(run["run_id"], 2, digest)
    assert cancelled["state"] == "CANCELLED"
    response = control.get_report(run["run_id"])
    assert response["report"]["state"] == "CANCELLED"
    assert cancelled["result"]["report_digest"] == response["report_digest"]

    assert control.cancel(run["run_id"], 2, digest) == cancelled
    assert not report_path(tmp_path, run["run_id"], 3).exists()


def test_finalization_binds_selected_weights_success_report_and_module(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    complete = running_report(completed_epochs=3)
    control.advance_progress(run["run_id"], 2, publish(files, complete))

    finalizing_report = succeeded_report()
    finalizing_report["state"] = "FINALIZING"
    finalizing_digest = publish(files, finalizing_report)
    success_report = succeeded_report()
    finalizing = control.begin_finalization(
        run["run_id"],
        3,
        finalizing_digest,
        success_report,
        "e" * 64,
    )
    assert finalizing["state"] == "FINALIZING"
    assert finalizing["finalization"]["selected_weight_digest"] == "d" * 64
    assert (
        control.begin_finalization(
            run["run_id"],
            3,
            finalizing_digest,
            success_report,
            "e" * 64,
        )
        == finalizing
    )
    assert not report_path(tmp_path, run["run_id"], 4).exists()

    success_digest = publish(files, success_report)
    control.claim_finalization(run["run_id"], 0)
    succeeded = control.succeed(run["run_id"], 4, success_digest, 1)
    assert succeeded["state"] == "SUCCEEDED"
    response = control.get_report(run["run_id"])
    assert response["report"]["state"] == "SUCCEEDED"
    assert response["report_digest"] == succeeded["finalization"]["report_digest"]
    assert control.succeed(run["run_id"], 4, success_digest, 1) == succeeded
    assert not report_path(tmp_path, run["run_id"], 5).exists()


def test_failure_report_error_becomes_exact_terminal_result(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    report = queued_report()
    report["state"] = "FAILED"
    report["failure"] = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training failed",
    }

    failed = control.fail(run["run_id"], "QUEUED", 2, publish(files, report))
    assert failed["state"] == "FAILED"
    assert failed["result"]["error"] == report["failure"]


def test_terminalizer_failure_uses_latest_report_and_is_exactly_replayable(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    control.advance_progress(run["run_id"], 2, publish(files, running_report()))
    error = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training worker stopped",
        "resource_id": run["run_id"],
    }

    class _Client:
        def get_run_control(self, run_id: str) -> dict:
            return control.get_control(run_id)

        def fail_run(self, run_id: str, state: str, revision: int, digest: str) -> dict:
            return control.fail(run_id, state, revision, digest)

    terminalizer = TrainingRunTerminalizer(files, _Client(), training_api_limits())
    first = terminalizer.fail(control.get_control(run["run_id"]), error)
    replay = terminalizer.fail(
        first,
        {
            "schema_version": 1,
            "code": "INTERNAL_ERROR",
            "message": "a later explanation cannot replace the winner",
        },
    )

    assert replay == first
    assert first["run"]["state"] == "FAILED"
    assert first["run"]["progress"]["completed_epochs"] == 1
    response = control.get_report(run["run_id"])
    assert response["report"]["epochs"][-1]["epoch"] == 1
    assert response["report"]["failure"] == error
    assert response["report_digest"] == first["run"]["result"]["report_digest"]


def test_stale_attachment_failure_cannot_overwrite_a_started_worker(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)

    class _RacingClient:
        def __init__(self) -> None:
            self.injected = False

        def get_run_control(self, run_id: str) -> dict:
            return control.get_control(run_id)

        def fail_run(self, run_id: str, state: str, revision: int, digest: str) -> dict:
            if not self.injected:
                self.injected = True
                control.attach_worker(run_id, "fc-started")
                control.claim_worker(run_id, trainer_build())
            return control.fail(run_id, state, revision, digest)

    terminalizer = TrainingRunTerminalizer(
        files, _RacingClient(), training_api_limits()
    )
    result = terminalizer.fail(
        control.get_control(run["run_id"]),
        {
            "schema_version": 1,
            "code": "RUNNER_FAILED",
            "message": "worker did not attach before its deadline",
        },
    )

    assert result["run"]["state"] == "RUNNING"
    assert result["run"]["report_revision"] == 1
    assert control.get_report(run["run_id"])["report"]["state"] == "QUEUED"


def test_failure_rebuilds_after_concurrent_progress_in_the_same_state(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())

    class _RacingClient:
        def __init__(self) -> None:
            self.injected = False

        def get_run_control(self, run_id: str) -> dict:
            return control.get_control(run_id)

        def fail_run(self, run_id: str, state: str, revision: int, digest: str) -> dict:
            if not self.injected:
                self.injected = True
                control.advance_progress(
                    run_id, 2, publish(files, running_report(completed_epochs=1))
                )
            return control.fail(run_id, state, revision, digest)

    terminalizer = TrainingRunTerminalizer(
        files, _RacingClient(), training_api_limits()
    )
    error = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training worker stopped",
    }
    result = terminalizer.fail(control.get_control(run["run_id"]), error)

    assert result["run"]["state"] == "FAILED"
    assert result["run"]["progress"]["completed_epochs"] == 1
    response = control.get_report(run["run_id"])
    assert response["report_revision"] == 3
    assert response["report"]["epochs"][-1]["epoch"] == 1
    assert response["report"]["failure"] == error


def test_failure_cannot_invent_or_change_model_selection(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    premature = succeeded_report()
    premature["state"] = "FAILED"
    premature["failure"] = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training failed",
    }
    with pytest.raises(GatewayFailure, match="before finalization"):
        control.fail(run["run_id"], "QUEUED", 2, publish(files, premature))

    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    complete = running_report(completed_epochs=3)
    control.advance_progress(run["run_id"], 2, publish(files, complete))
    finalizing_report = succeeded_report()
    finalizing_report["state"] = "FINALIZING"
    control.begin_finalization(
        run["run_id"],
        3,
        publish(files, finalizing_report),
        succeeded_report(),
        "e" * 64,
    )

    failed = copy.deepcopy(finalizing_report)
    failed["state"] = "FAILED"
    failed["failure"] = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "publication failed",
    }
    failed["model_selection"]["selected_weight_digest"] = "f" * 64
    control.claim_finalization(run["run_id"], 0)
    with pytest.raises(GatewayFailure, match="differs from finalization"):
        control.fail(run["run_id"], "FINALIZING", 4, publish(files, failed), 1)


def test_interrupted_promotion_can_be_replaced_before_advertisement(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    stale = queued_report()
    stale["state"] = "CANCELLED"
    stale_digest = publish(files, stale)
    files.promote_report(run["run_id"], 2, stale_digest)

    report = running_report()
    updated = control.advance_progress(run["run_id"], 2, publish(files, report))

    assert updated["report_revision"] == 2
    assert files.read_report(run["run_id"], 2)["report"]["state"] == "RUNNING"


def test_interrupted_promotion_completes_with_the_same_report(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    report = running_report()
    digest = publish(files, report)
    files.promote_report(run["run_id"], 2, digest)

    updated = control.advance_progress(run["run_id"], 2, digest)

    assert updated["report_revision"] == 2
    assert updated["progress"]["completed_epochs"] == 1


def test_losing_candidate_cannot_replace_an_advertised_revision(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    running = running_report()
    running_digest = publish(files, running)
    control.advance_progress(run["run_id"], 2, running_digest)
    cancelled = copy.deepcopy(running)
    cancelled["state"] = "CANCELLED"
    cancelled_digest = publish(files, cancelled)

    with pytest.raises(GatewayFailure, match="digest differs"):
        control.cancel(run["run_id"], 2, cancelled_digest)

    assert files.read_report(run["run_id"], 2)["report_digest"] == running_digest


def test_clear_removes_terminal_report_and_row_in_two_steps(tmp_path: Path) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    report = queued_report()
    report["state"] = "CANCELLED"
    control.cancel(run["run_id"], 2, publish(files, report))

    assert control.begin_delete(run["run_id"])["state"] == "CANCELLED"
    files.delete_reports(run["run_id"])
    control.finish_delete(run["run_id"])
    with pytest.raises(GatewayFailure) as raised:
        control.get(run["run_id"])
    assert raised.value.error["code"] == "TRAINING_RUN_NOT_FOUND"


def test_success_report_digest_cannot_change_after_finalization(
    tmp_path: Path,
) -> None:
    control, files, sources = make_control(tmp_path)
    run = accept(control, files, sources, tmp_path)
    control.attach_worker(run["run_id"], "fc-123")
    control.claim_worker(run["run_id"], trainer_build())
    complete = running_report(completed_epochs=3)
    control.advance_progress(run["run_id"], 2, publish(files, complete))
    finalizing_report = succeeded_report()
    finalizing_report["state"] = "FINALIZING"
    control.begin_finalization(
        run["run_id"],
        3,
        publish(files, finalizing_report),
        succeeded_report(),
        "e" * 64,
    )

    changed = copy.deepcopy(succeeded_report())
    changed["messages"] = [{"severity": "info", "message": "Changed"}]
    control.claim_finalization(run["run_id"], 0)
    with pytest.raises(GatewayFailure, match="differs from finalization"):
        control.succeed(run["run_id"], 4, publish(files, changed), 1)
