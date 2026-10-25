from __future__ import annotations

import json
from pathlib import Path

from histoforge.canonical_json import canonical_digest
from histoforge.cloud.training.runs.contract import (
    validate_training_run_page,
    validate_training_run_record,
)
from histoforge.cloud.training.runs.report import (
    validate_training_report,
    validate_training_report_response,
)
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)
from tests.test_training_dataset import training_api_limits

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "contracts"


def test_shared_dense_wire_fixture_preserves_task_specific_absence() -> None:
    fixture = _load("training-dense-v1.json")
    limits = fixture["trainingApiLimits"]
    for name, validate in (
        ("request", validate_training_run_request),
        ("report", validate_training_report),
        ("run", validate_training_run_record),
        ("report_response", validate_training_report_response),
    ):
        case = fixture[name]
        resolved = validate(case["value"], limits)
        assert resolved == case["value"]
        assert canonical_digest(resolved) == case["sha256"]
    assert "min_instance_pixels" not in fixture["request"]["value"]
    assert "min_instance_pixels" not in fixture["report"]["value"]
    assert fixture["run"]["value"]["request_digest"] == fixture["request"]["sha256"]


def test_shared_training_run_wire_fixture() -> None:
    limits = training_api_limits()
    fixture = _load("training-runs-v1.json")

    accepted = fixture["accepted"]
    assert canonical_digest(accepted["value"]) == accepted["sha256"]

    records = {}
    for case in fixture["records"]:
        record = validate_training_run_record(case["value"], limits)
        assert record == case["value"]
        assert canonical_digest(record) == case["sha256"]
        records[case["name"]] = record

    queued = records["queued"]
    assert accepted["value"] == {
        "run_id": queued["run_id"],
        "request_digest": training_run_request_digest(queued["request"], limits),
    }
    assert records["finalizing"]["state"] == "FINALIZING"
    assert "result" not in records["finalizing"]
    assert records["cancelled"]["result"]["type"] == "cancelled"

    page = validate_training_run_page(fixture["page"]["value"], limits)
    assert canonical_digest(page) == fixture["page"]["sha256"]
    tied = page["items"][1:]
    assert tied[0]["created_at"] == tied[1]["created_at"]
    assert tied[0]["run_id"] < tied[1]["run_id"]

    report_fixture = _load("module-training-report-v1.json")
    response_fixture = fixture["report_response"]
    response = {
        "report_revision": response_fixture["report_revision"],
        "report_digest": response_fixture["report_digest"],
        "report": report_fixture["response"]["report"],
    }
    response = validate_training_report_response(response, limits)
    assert canonical_digest(response) == response_fixture["sha256"]
    succeeded = records["succeeded"]
    assert response["report_revision"] == succeeded["report_revision"]
    assert response["report"]["run_id"] == succeeded["run_id"]
    assert response["report_digest"] == succeeded["finalization"]["report_digest"]

    dataset_manifest = _load("training-datasets-v1.json")["manifest"]["value"]
    assert (
        response["report"]["dataset_digest"]
        == succeeded["request"]["dataset_digest"]
        == dataset_manifest["dataset_digest"]
    )
    assert response["report"]["task"] == dataset_manifest["task"]


def _load(name: str) -> dict:
    return json.loads((_FIXTURE_ROOT / name).read_text(encoding="utf-8"))
