from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.datasets.registry import TrainingDatasetRegistry
from histoforge.cloud.training.runs.state import TrainingRunState
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_state import service_limits
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request

RUN_ID = "44444444-4444-4444-8444-444444444444"
SECOND_RUN_ID = "55555555-5555-4555-8555-555555555555"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def make_state(
    root: Path,
    *,
    now: list[datetime] | None = None,
    ids: list[str] | None = None,
    service_overrides: dict[str, int] | None = None,
) -> tuple[TrainingRunState, TrainingControlRecords]:
    times = now or [NOW]
    generated_ids = iter(ids or [RUN_ID])
    private_limits = service_limits(**(service_overrides or {}))
    records = TrainingControlRecords(root, training_api_limits(), private_limits)
    request = run_request()
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
    return (
        TrainingRunState(
            records,
            training_api_limits(),
            private_limits,
            clock=lambda: times[0],
            id_factory=lambda: uuid.UUID(next(generated_ids)),
        ),
        records,
    )


def accept_run(state: TrainingRunState) -> dict:
    request = run_request()
    state.reserve(request)
    return state.accept(request, trainer_build(), 3, 3_600)["accepted"]


def test_reserve_accept_and_exact_replay_are_one_run(tmp_path: Path) -> None:
    state, records = make_state(tmp_path)
    request = run_request()

    first = state.reserve(request)
    assert first["type"] == "preparing"
    assert first["admission"]["run_id"] == RUN_ID
    assert state.reserve(request) == first

    result = state.accept(request, trainer_build(), 3, 3_600)
    accepted = result["accepted"]
    assert result["should_dispatch"] is True
    assert accepted["state"] == "QUEUED"
    assert accepted["report_revision"] == 1
    assert state.reserve(request) == {"type": "accepted", "run": accepted}
    assert state.accept(request, trainer_build(), 3, 3_600) == {
        "accepted": accepted,
        "should_dispatch": False,
    }
    assert records.admissions() == []
    assert len(records.runs()) == 1


def test_lookup_never_creates_a_missing_admission(tmp_path: Path) -> None:
    state, records = make_state(tmp_path)

    assert state.lookup(run_request()) is None
    assert records.admissions() == []

    reserved = state.reserve(run_request())
    assert state.lookup(run_request()) == reserved


def test_abandoned_admission_is_fenced_before_artifact_cleanup(
    tmp_path: Path,
) -> None:
    now = [NOW]
    state, records = make_state(
        tmp_path,
        now=now,
        service_overrides={"pending_snapshot_cleanup_age_seconds": 10},
    )
    request = run_request()
    admission = state.reserve(request)["admission"]

    now[0] += timedelta(seconds=9)
    assert state.claim_abandoned_admission() is None
    assert state.transient_cleanup_scopes([admission["run_id"]]) == {
        admission["run_id"]: {"type": "blocked"}
    }

    now[0] += timedelta(seconds=1)
    assert state.claim_abandoned_admission() == admission
    assert state.transient_cleanup_scopes([admission["run_id"]]) == {
        admission["run_id"]: {"type": "all"}
    }
    assert records.admissions() == []
    with pytest.raises(GatewayFailure, match="was not reserved"):
        state.accept(request, trainer_build(), 3, 3_600)


def test_abandoned_cleanup_never_claims_an_accepted_run(tmp_path: Path) -> None:
    now = [NOW]
    state, _ = make_state(
        tmp_path,
        now=now,
        service_overrides={"pending_snapshot_cleanup_age_seconds": 10},
    )
    accepted = accept_run(state)

    now[0] += timedelta(seconds=10)
    assert state.claim_abandoned_admission() is None
    assert state.transient_cleanup_scopes([accepted["run_id"]]) == {
        accepted["run_id"]: {"type": "blocked"}
    }

    cancelled = state.cancel(accepted["run_id"], "d" * 64, 2)
    assert cancelled["state"] == "CANCELLED"
    assert state.transient_cleanup_scopes([accepted["run_id"]]) == {
        accepted["run_id"]: {
            "type": "terminal",
            "report_revision": 2,
            "report_digest": "d" * 64,
        }
    }


def test_admissions_and_active_runs_share_one_capacity_limit(tmp_path: Path) -> None:
    state, _ = make_state(
        tmp_path,
        ids=[RUN_ID, SECOND_RUN_ID],
        service_overrides={"max_active_training_runs": 1},
    )
    state.reserve(run_request())
    other = run_request()
    other["training_seed"] += 1
    with pytest.raises(GatewayFailure) as raised:
        state.reserve(other)
    assert raised.value.error["code"] == "RATE_LIMITED"


def test_worker_attachment_claim_and_progress_are_fenced(tmp_path: Path) -> None:
    state, _ = make_state(tmp_path)
    accepted = accept_run(state)

    running = state.attach_worker(accepted["run_id"], "fc-123")
    assert running["state"] == "RUNNING"
    assert state.attach_worker(accepted["run_id"], "fc-123") == running
    with pytest.raises(GatewayFailure, match="build does not match"):
        state.claim_worker(
            accepted["run_id"],
            {**trainer_build(), "artifact_digest": "0" * 64},
        )

    state.claim_worker(accepted["run_id"], trainer_build())
    with pytest.raises(GatewayFailure, match="already claimed"):
        state.claim_worker(accepted["run_id"], trainer_build())
    epoch = state.advance_progress(accepted["run_id"], 1, 2)
    assert epoch["progress"]["completed_epochs"] == 1
    assert epoch["report_revision"] == 2
    with pytest.raises(GatewayFailure, match="must advance monotonically"):
        state.advance_progress(accepted["run_id"], 1, 3)


def test_attachment_and_execution_deadlines_reject_late_workers(tmp_path: Path) -> None:
    now = [NOW]
    state, _ = make_state(
        tmp_path,
        now=now,
        service_overrides={"worker_call_attachment_timeout_seconds": 10},
    )
    accepted = accept_run(state)
    now[0] += timedelta(seconds=10)
    with pytest.raises(GatewayFailure, match="attachment deadline"):
        state.attach_worker(accepted["run_id"], "fc-late")


def test_complete_result_cannot_finalize_after_execution_deadline(
    tmp_path: Path,
) -> None:
    now = [NOW]
    state, _ = make_state(tmp_path, now=now)
    accepted = accept_run(state)
    state.attach_worker(accepted["run_id"], "fc-123")
    state.claim_worker(accepted["run_id"], trainer_build())
    state.advance_progress(accepted["run_id"], 3, 2)
    deadline = state.get_control(accepted["run_id"])["worker"]["execution_deadline_at"]
    now[0] = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

    with pytest.raises(GatewayFailure, match="execution deadline"):
        state.begin_finalization(
            accepted["run_id"],
            {
                "selected_weight_digest": "c" * 64,
                "report_digest": "d" * 64,
                "module_content_digest": "e" * 64,
            },
            3,
        )


def test_cancel_wins_before_finalization_and_blocks_late_worker(tmp_path: Path) -> None:
    state, _ = make_state(tmp_path)
    accepted = accept_run(state)
    state.attach_worker(accepted["run_id"], "fc-123")
    state.claim_worker(accepted["run_id"], trainer_build())

    cancelled = state.cancel(accepted["run_id"], "c" * 64, 2)
    assert cancelled["state"] == "CANCELLED"
    assert state.cancel(accepted["run_id"], "c" * 64, 2) == cancelled
    with pytest.raises(GatewayFailure, match="no longer accepted"):
        state.advance_progress(accepted["run_id"], 1, 3)
    with pytest.raises(GatewayFailure, match="cannot begin finalization"):
        state.begin_finalization(
            accepted["run_id"],
            {
                "selected_weight_digest": "d" * 64,
                "report_digest": "e" * 64,
                "module_content_digest": "f" * 64,
            },
            3,
        )


def test_complete_run_closes_cancellation_and_preserves_dataset(tmp_path: Path) -> None:
    state, records = make_state(tmp_path)
    accepted = accept_run(state)
    state.attach_worker(accepted["run_id"], "fc-123")
    state.claim_worker(accepted["run_id"], trainer_build())
    state.advance_progress(accepted["run_id"], 3, 2)
    finalization = {
        "selected_weight_digest": "c" * 64,
        "report_digest": "d" * 64,
        "module_content_digest": "e" * 64,
    }
    finalizing = state.begin_finalization(accepted["run_id"], finalization, 3)
    assert finalizing["state"] == "FINALIZING"
    assert state.training_result_visible(accepted["run_id"]) is False
    with pytest.raises(GatewayFailure, match="can no longer be cancelled"):
        state.cancel(accepted["run_id"], "f" * 64, 4)

    claimed = state.claim_finalization(accepted["run_id"], 0)
    assert claimed["finalization_revision"] == 1
    succeeded = state.succeed(accepted["run_id"], 4, 1)
    assert succeeded["state"] == "SUCCEEDED"
    assert state.training_result_visible(accepted["run_id"]) is True
    assert succeeded["result"] == {"type": "succeeded"}
    assert state.dataset_in_use(run_request()["dataset_id"]) is False
    assert records.get_dataset(run_request()["dataset_id"])["type"] == "retained"


def finalizing_state(root: Path) -> tuple[TrainingRunState, TrainingControlRecords]:
    state, records = make_state(root)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-123")
    state.claim_worker(RUN_ID, trainer_build())
    state.advance_progress(RUN_ID, 3, 2)
    state.begin_finalization(
        RUN_ID,
        {
            "selected_weight_digest": "c" * 64,
            "report_digest": "d" * 64,
            "module_content_digest": "e" * 64,
        },
        3,
    )
    return state, records


def test_finalization_claim_is_private_durable_and_compare_and_swap(
    tmp_path: Path,
) -> None:
    state, _ = finalizing_state(tmp_path)
    before = state.get_control(RUN_ID)
    claimed = state.claim_finalization(RUN_ID, 0)
    assert claimed == {**before, "finalization_revision": 1}

    restarted, _ = make_state(tmp_path)
    assert restarted.get_control(RUN_ID) == claimed
    with pytest.raises(GatewayFailure, match="revision is not current"):
        restarted.claim_finalization(RUN_ID, 0)
    assert restarted.claim_finalization(RUN_ID, 1) == {
        **before,
        "finalization_revision": 2,
    }


def test_finalization_claim_requires_finalizing_and_rejects_overflow(
    tmp_path: Path,
) -> None:
    state, records = finalizing_state(tmp_path)
    current = state.get_control(RUN_ID)
    current["finalization_revision"] = 2**31 - 1
    records.replace_run(current)
    with pytest.raises(ValueError, match="finalization_revision"):
        state.claim_finalization(RUN_ID, 2**31 - 1)
    assert state.get_control(RUN_ID) == current
    state.succeed(RUN_ID, 4, 2**31 - 1)
    with pytest.raises(GatewayFailure, match="not finalizing"):
        state.claim_finalization(RUN_ID, 2**31 - 1)


def test_stale_finalization_outcomes_cannot_mutate_state(tmp_path: Path) -> None:
    state, _ = finalizing_state(tmp_path)
    state.claim_finalization(RUN_ID, 0)
    current = state.claim_finalization(RUN_ID, 1)
    error = {"schema_version": 1, "code": "INTERNAL_ERROR", "message": "save failed"}
    with pytest.raises(GatewayFailure, match="revision is not current"):
        state.fail(RUN_ID, error, "f" * 64, 4, 1)
    with pytest.raises(GatewayFailure, match="revision is not current"):
        state.succeed(RUN_ID, 4, 1)
    assert state.get_control(RUN_ID) == current
    assert state.fail(RUN_ID, error, "f" * 64, 4, 2)["state"] == "FAILED"


def test_active_run_cannot_clear_and_terminal_run_can_clear(
    tmp_path: Path,
) -> None:
    state, _ = make_state(tmp_path)
    accepted = accept_run(state)
    assert state.dataset_in_use(run_request()["dataset_id"]) is True
    with pytest.raises(GatewayFailure, match="active training run"):
        state.begin_delete(accepted["run_id"])

    failed = state.fail(
        accepted["run_id"],
        {
            "schema_version": 1,
            "code": "RUNNER_FAILED",
            "message": "training failed",
        },
        "f" * 64,
        2,
    )
    assert failed["state"] == "FAILED"
    assert state.begin_delete(accepted["run_id"]) == failed
    state.finish_delete(accepted["run_id"])
    assert state.begin_delete(accepted["run_id"]) is None
    assert state.training_result_visible(accepted["run_id"]) is True


def test_active_run_blocks_its_dataset_deletion(tmp_path: Path) -> None:
    state, records = make_state(tmp_path)
    accepted = accept_run(state)
    registry = TrainingDatasetRegistry(
        records,
        training_api_limits(),
        dataset_in_use=state.dataset_in_use,
    )
    dataset_id = run_request()["dataset_id"]

    with pytest.raises(GatewayFailure, match="dataset is in use"):
        registry.begin_delete(dataset_id)

    state.fail(
        accepted["run_id"],
        {
            "schema_version": 1,
            "code": "RUNNER_FAILED",
            "message": "training failed",
        },
        "f" * 64,
        2,
    )
    assert registry.begin_delete(dataset_id)["type"] == "cleanup-required"


def test_run_admission_shares_the_control_record_capacity(tmp_path: Path) -> None:
    state, _ = make_state(
        tmp_path,
        service_overrides={"max_control_records": 1},
    )

    with pytest.raises(GatewayFailure) as raised:
        state.reserve(run_request())

    assert raised.value.error["code"] == "RATE_LIMITED"


def test_noncanonical_run_record_fails_closed(tmp_path: Path) -> None:
    state, _ = make_state(tmp_path)
    accepted = accept_run(state)
    path = (
        tmp_path / "training" / "control" / "v1" / "runs" / f"{accepted['run_id']}.json"
    )
    parsed = json.loads(path.read_text())
    path.write_text(json.dumps(parsed, indent=2))
    with pytest.raises(GatewayFailure) as raised:
        state.get(accepted["run_id"])
    assert raised.value.error["code"] == "INTERNAL_ERROR"


def test_accept_rejects_deleted_or_replaced_dataset(tmp_path: Path) -> None:
    state, records = make_state(tmp_path)
    request = run_request()
    state.reserve(request)
    records.remove_dataset(request["dataset_id"])
    with pytest.raises(GatewayFailure, match="unavailable or has changed"):
        state.accept(request, trainer_build(), 3, 3_600)

    changed = copy.deepcopy(request)
    changed["training_seed"] += 1
    with pytest.raises(GatewayFailure, match="was not reserved"):
        state.accept(changed, trainer_build(), 3, 3_600)
