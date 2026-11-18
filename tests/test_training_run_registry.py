from __future__ import annotations

import copy

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.runs.contract import validate_training_run_page
from histoforge.cloud.training.runs.registry import (
    training_run_page,
    validate_training_run_record_selection,
)
from histoforge.cloud.training.runs.report import (
    training_report_digest,
    validate_training_report_response,
    validate_training_report_response_for_run,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report, succeeded_report
from tests.test_training_run_contract import queued_control


def _run(run_id: str, created_at: str, *, completed_epochs: int = 0) -> dict:
    run = copy.deepcopy(queued_control()["run"])
    run.update(
        {
            "run_id": run_id,
            "progress": {
                "completed_epochs": completed_epochs,
                "total_epochs": 3,
            },
            "created_at": created_at,
            "updated_at": created_at,
        }
    )
    if completed_epochs:
        run["state"] = "RUNNING"
        run["report_revision"] = 2
    return run


def _report(run_id: str, *, completed_epochs: int = 0) -> dict:
    report = copy.deepcopy(queued_report())
    report["run_id"] = run_id
    if completed_epochs:
        report = copy.deepcopy(succeeded_report())
        report["run_id"] = run_id
        report["state"] = "RUNNING"
        report["epochs"] = report["epochs"][:completed_epochs]
        del report["model_selection"]
    return report


def _response(run: dict, report: dict, limits: dict) -> dict:
    return {
        "report_revision": run["report_revision"],
        "report_digest": training_report_digest(report, limits),
        "report": report,
    }


def test_page_orders_newest_first_and_uses_an_exclusive_cursor() -> None:
    limits = training_api_limits()
    older = _run("44444444-4444-4444-8444-444444444444", "2026-09-04T12:00:00.000Z")
    newer = _run(
        "55555555-5555-4555-8555-555555555555",
        "2026-09-04T13:00:00.000Z",
        completed_epochs=2,
    )
    reports = {
        older["run_id"]: _report(older["run_id"]),
        newer["run_id"]: _report(newer["run_id"], completed_epochs=2),
    }

    first = training_run_page(
        [older, newer],
        lambda run_id: _response(
            {older["run_id"]: older, newer["run_id"]: newer}[run_id],
            reports[run_id],
            limits,
        ),
        1,
        None,
        limits,
    )
    second = training_run_page(
        [older, newer],
        lambda run_id: _response(
            {older["run_id"]: older, newer["run_id"]: newer}[run_id],
            reports[run_id],
            limits,
        ),
        1,
        first["next_cursor"],
        limits,
    )

    assert first["items"][0]["run_id"] == newer["run_id"]
    assert first["items"][0]["primary_metric"] == {
        "series_id": "validation-instance-msa.v1",
        "display_name": "Validation mSA",
        "value": 0.7,
        "epoch": 2,
    }
    assert second["items"][0]["run_id"] == older["run_id"]
    assert "next_cursor" not in second


def test_page_rejects_invalid_cursor_limit_and_inconsistent_report() -> None:
    limits = training_api_limits()
    run = _run(
        "44444444-4444-4444-8444-444444444444",
        "2026-09-04T12:00:00.000Z",
    )
    report = _report(run["run_id"])

    with pytest.raises(GatewayFailure) as cursor:
        training_run_page(
            [run], lambda _run_id: _response(run, report, limits), 1, "!", limits
        )
    assert cursor.value.error["code"] == "VALIDATION_ERROR"

    with pytest.raises(GatewayFailure) as maximum:
        training_run_page(
            [run],
            lambda _run_id: _response(run, report, limits),
            limits["max_page_size"] + 1,
            None,
            limits,
        )
    assert maximum.value.error["code"] == "VALIDATION_ERROR"

    changed = copy.deepcopy(report)
    changed["run_id"] = "55555555-5555-4555-8555-555555555555"
    with pytest.raises(GatewayFailure) as inconsistent:
        training_run_page(
            [run], lambda _run_id: _response(run, changed, limits), 1, None, limits
        )
    assert inconsistent.value.error["code"] == "INTERNAL_ERROR"

    too_small = copy.deepcopy(limits)
    too_small["max_page_encoded_bytes"] = 1
    with pytest.raises(GatewayFailure) as encoded:
        training_run_page(
            [run],
            lambda _run_id: _response(run, report, limits),
            1,
            None,
            too_small,
        )
    assert encoded.value.error["code"] == "INTERNAL_ERROR"


def test_page_validator_rejects_metric_after_advertised_progress() -> None:
    limits = training_api_limits()
    run = _run(
        "44444444-4444-4444-8444-444444444444",
        "2026-09-04T12:00:00.000Z",
        completed_epochs=2,
    )
    report = _report(run["run_id"], completed_epochs=2)
    page = training_run_page(
        [run], lambda _run_id: _response(run, report, limits), 1, None, limits
    )
    page["items"][0]["primary_metric"]["epoch"] = 3

    with pytest.raises(ValueError, match="exceeds completed"):
        validate_training_run_page(page, limits)


def test_page_validator_enforces_order_tiebreak_and_encoded_cap() -> None:
    limits = training_api_limits()
    older = _run(
        "44444444-4444-4444-8444-444444444444",
        "2026-09-04T12:00:00.000Z",
    )
    newer = _run(
        "55555555-5555-4555-8555-555555555555",
        "2026-09-04T13:00:00.000Z",
    )
    reports = {run["run_id"]: _report(run["run_id"]) for run in (older, newer)}
    page = training_run_page(
        [older, newer],
        lambda run_id: _response(
            {older["run_id"]: older, newer["run_id"]: newer}[run_id],
            reports[run_id],
            limits,
        ),
        2,
        None,
        limits,
    )

    reversed_page = copy.deepcopy(page)
    reversed_page["items"].reverse()
    with pytest.raises(ValueError, match="uniquely ordered"):
        validate_training_run_page(reversed_page, limits)

    tied = copy.deepcopy(page)
    tied["items"][1]["created_at"] = tied["items"][0]["created_at"]
    tied["items"][1]["updated_at"] = tied["items"][0]["created_at"]
    tied["items"].sort(key=lambda item: item["run_id"], reverse=True)
    with pytest.raises(ValueError, match="uniquely ordered"):
        validate_training_run_page(tied, limits)

    duplicate_id = copy.deepcopy(page)
    duplicate_id["items"][1]["run_id"] = duplicate_id["items"][0]["run_id"]
    with pytest.raises(ValueError, match="uniquely ordered"):
        validate_training_run_page(duplicate_id, limits)

    incomplete_success = copy.deepcopy(page["items"][0])
    incomplete_success["state"] = "SUCCEEDED"
    incomplete_success["result"] = {"type": "succeeded"}
    incomplete_success["progress"]["completed_epochs"] = 0
    with pytest.raises(ValueError, match="every training epoch"):
        validate_training_run_page({"items": [incomplete_success]}, limits)

    below_page_size = copy.deepcopy(limits)
    below_page_size["max_page_encoded_bytes"] = len(canonical_bytes(page)) - 1
    with pytest.raises(ValueError, match="max_page_encoded_bytes"):
        validate_training_run_page(page, below_page_size)


def test_state_lane_selection_is_ordered_and_bounded() -> None:
    limits = training_api_limits()
    older = _run(
        "44444444-4444-4444-8444-444444444444",
        "2026-09-04T12:00:00.000Z",
    )
    newer = _run(
        "55555555-5555-4555-8555-555555555555",
        "2026-09-04T13:00:00.000Z",
    )
    newest = _run(
        "66666666-6666-4666-8666-666666666666",
        "2026-09-04T14:00:00.000Z",
    )

    assert validate_training_run_record_selection([newer, older], 1, limits) == [
        newer,
        older,
    ]
    with pytest.raises(ValueError, match="uniquely ordered"):
        validate_training_run_record_selection([older, newer], 1, limits)
    with pytest.raises(ValueError, match="not bounded"):
        validate_training_run_record_selection([newest, newer, older], 1, limits)

    duplicate_id = copy.deepcopy(older)
    duplicate_id["created_at"] = newest["created_at"]
    duplicate_id["updated_at"] = newest["updated_at"]
    with pytest.raises(ValueError, match="uniquely ordered"):
        validate_training_run_record_selection([duplicate_id, older], 1, limits)


def test_report_response_digest_is_the_report_digest() -> None:
    limits = training_api_limits()
    report = queued_report()
    expected = training_report_digest(report, limits)

    response = {
        "report_revision": 1,
        "report_digest": expected,
        "report": report,
    }
    assert validate_training_report_response(response, limits) == response

    response["report_digest"] = "f" * 64
    with pytest.raises(ValueError, match="does not match report"):
        validate_training_report_response(response, limits)


def test_report_response_is_bound_to_the_advertised_run() -> None:
    limits = training_api_limits()
    run = _run(
        "44444444-4444-4444-8444-444444444444",
        "2026-09-04T12:00:00.000Z",
    )
    report = _report(run["run_id"])
    response = _response(run, report, limits)

    assert validate_training_report_response_for_run(response, run, limits) == response

    wrong_revision = copy.deepcopy(response)
    wrong_revision["report_revision"] = 2
    with pytest.raises(ValueError, match="advertised run state"):
        validate_training_report_response_for_run(wrong_revision, run, limits)

    finalizing_report = copy.deepcopy(succeeded_report())
    finalizing_report["run_id"] = run["run_id"]
    finalizing_report["state"] = "FINALIZING"
    finalizing_run = _run(run["run_id"], run["created_at"], completed_epochs=3)
    finalizing_run.update(
        {
            "state": "FINALIZING",
            "report_revision": 3,
            "finalization": {
                "selected_weight_digest": "f" * 64,
                "report_digest": "e" * 64,
                "module_content_digest": "d" * 64,
            },
        }
    )
    with pytest.raises(ValueError, match="selection differs"):
        validate_training_report_response_for_run(
            _response(finalizing_run, finalizing_report, limits),
            finalizing_run,
            limits,
        )
