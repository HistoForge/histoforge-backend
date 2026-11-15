from __future__ import annotations

import copy

import pytest
from histoforge.cloud.training.runs.contract import (
    validate_training_run_admission_record,
    validate_training_run_control_record,
)
from histoforge.cloud.training.runs.request import training_run_request_digest
from tests.test_training_dataset import training_api_limits
from tests.test_training_run_request import run_request


def trainer_build() -> dict:
    return {
        "version": "phase8-test",
        "artifact_digest": "a" * 64,
        "manifest_digest": "b" * 64,
    }


def queued_control() -> dict:
    request = run_request()
    return {
        "finalization_revision": 0,
        "run": {
            "run_id": "44444444-4444-4444-8444-444444444444",
            "request": request,
            "request_digest": training_run_request_digest(
                request, training_api_limits()
            ),
            "trainer_build": trainer_build(),
            "state": "QUEUED",
            "progress": {"completed_epochs": 0, "total_epochs": 80},
            "report_revision": 1,
            "created_at": "2026-09-04T12:00:00.000Z",
            "updated_at": "2026-09-04T12:00:00.000Z",
        },
        "worker": {
            "call_attachment_deadline_at": "2026-09-04T12:01:00.000Z",
            "execution_deadline_at": "2026-09-05T12:10:00.000Z",
            "worker_claimed": False,
        },
    }


def test_admission_and_queued_control_round_trip() -> None:
    request = run_request()
    admission = {
        "type": "preparing",
        "run_id": "44444444-4444-4444-8444-444444444444",
        "request_digest": training_run_request_digest(request, training_api_limits()),
        "created_at": "2026-09-04T12:00:00.000Z",
    }

    assert validate_training_run_admission_record(admission) == admission
    assert (
        validate_training_run_control_record(queued_control(), training_api_limits())
        == queued_control()
    )


def test_control_enforces_worker_and_state_relationships() -> None:
    running = queued_control()
    running["run"]["state"] = "RUNNING"
    with pytest.raises(ValueError, match="requires an attached worker"):
        validate_training_run_control_record(running, training_api_limits())

    running["worker"]["worker_call_id"] = "fc-123"
    running["worker"]["worker_claimed"] = True
    running["run"]["progress"]["completed_epochs"] = 1
    assert (
        validate_training_run_control_record(running, training_api_limits()) == running
    )

    running["worker"]["worker_call_id"] = "x" * 129
    with pytest.raises(ValueError, match="exceeds 128"):
        validate_training_run_control_record(running, training_api_limits())


def test_control_enforces_terminal_payloads_and_finalization() -> None:
    succeeded = queued_control()
    succeeded["finalization_revision"] = 1
    succeeded["run"].update(
        {
            "state": "SUCCEEDED",
            "progress": {"completed_epochs": 80, "total_epochs": 80},
            "finalization": {
                "selected_weight_digest": "c" * 64,
                "report_digest": "d" * 64,
                "module_content_digest": "e" * 64,
            },
            "result": {"type": "succeeded"},
        }
    )
    succeeded["worker"].update({"worker_call_id": "fc-123", "worker_claimed": True})
    assert (
        validate_training_run_control_record(succeeded, training_api_limits())
        == succeeded
    )

    invalid = copy.deepcopy(succeeded)
    invalid["finalization_revision"] = 0
    with pytest.raises(ValueError, match="completed finalization requires"):
        validate_training_run_control_record(invalid, training_api_limits())

    invalid = copy.deepcopy(succeeded)
    del invalid["run"]["finalization"]
    with pytest.raises(ValueError, match="SUCCEEDED requires finalization"):
        validate_training_run_control_record(invalid, training_api_limits())

    failed = queued_control()
    failed["run"].update(
        {
            "state": "FAILED",
            "result": {
                "type": "failed",
                "error": {
                    "schema_version": 1,
                    "code": "RUNNER_FAILED",
                    "message": "training failed",
                },
                "report_digest": "f" * 64,
            },
        }
    )
    assert validate_training_run_control_record(failed, training_api_limits()) == failed

    oversized_error = copy.deepcopy(failed)
    oversized_error["run"]["result"]["error"]["message"] = "x" * 4097
    with pytest.raises(ValueError, match="result.error.message exceeds"):
        validate_training_run_control_record(oversized_error, training_api_limits())

    incomplete = copy.deepcopy(succeeded)
    incomplete["run"]["state"] = "FAILED"
    incomplete["run"]["progress"]["completed_epochs"] = 1
    incomplete["run"]["result"] = {
        "type": "failed",
        "error": {
            "schema_version": 1,
            "code": "RUNNER_FAILED",
            "message": "publication failed",
        },
        "report_digest": "f" * 64,
    }
    with pytest.raises(ValueError, match="finalization requires every training epoch"):
        validate_training_run_control_record(incomplete, training_api_limits())


def test_control_rejects_request_digest_or_epoch_drift() -> None:
    control = queued_control()
    control["run"]["request_digest"] = "0" * 64
    with pytest.raises(ValueError, match="does not match request"):
        validate_training_run_control_record(control, training_api_limits())

    control = queued_control()
    control["run"]["progress"]["total_epochs"] = 1_001
    with pytest.raises(ValueError, match="exceeds the training report limit"):
        validate_training_run_control_record(control, training_api_limits())


@pytest.mark.parametrize("revision", [-1, True, 1.0, None, 2**31])
def test_control_rejects_invalid_finalization_revision(revision: object) -> None:
    control = queued_control()
    control["finalization_revision"] = revision
    with pytest.raises(ValueError, match="finalization_revision"):
        validate_training_run_control_record(control, training_api_limits())


def test_control_requires_explicit_zero_before_finalization() -> None:
    control = queued_control()
    control["finalization_revision"] = 1
    with pytest.raises(ValueError, match="zero before finalization"):
        validate_training_run_control_record(control, training_api_limits())
    del control["finalization_revision"]
    with pytest.raises(ValueError, match="missing or unknown fields"):
        validate_training_run_control_record(control, training_api_limits())
