"""Python and QuPath agree on image assignments and complete-result evidence."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest
from histoforge.cloud.inference.finite.contract import (
    finite_input_identity,
    validate_finite_inference_request,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
    validate_model_result_manifest,
)
from histoforge.cloud.inference.runs.request import (
    model_run_request_digest,
    validate_model_run_request,
)
from histoforge.cloud.inference.runs.source_bindings import (
    validate_run_source_bindings,
)

FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures/contracts/model-source-delivery-v1.json"
    ).read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_shared_requests_bindings_execution_and_result_identities(case):
    request = validate_model_run_request(json.loads(case["requestCanonical"]))
    assert canonical_bytes(request).decode() == case["requestCanonical"]
    assert model_run_request_digest(request) == case["requestDigest"]
    plans = build_expected_model_execution_plans(request, FIXTURE["modelRunLimits"])
    delivery = json.loads(case["deliveryCanonical"])
    assert delivery["request_digest"] == case["requestDigest"]
    assert delivery["execution_plans"] == [
        {"input_id": input_id, "plan": plan} for input_id, plan in sorted(plans.items())
    ]
    inputs = {item["input_id"]: item for item in request["inputs"]}
    bindings = validate_run_source_bindings(delivery["source_bindings"], inputs, plans)
    assert bindings == delivery["source_bindings"]
    input_id = request["inputs"][0]["input_id"]
    for unit, expected in enumerate(case["unitExecutionDigests"]):
        assert (
            model_run_execution_identity_digest(
                request, input_id, unit, FIXTURE["modelRunLimits"], bindings
            )
            == expected
        )
    manifest = validate_model_result_manifest(
        json.loads(case["manifestCanonical"]), request, FIXTURE["modelRunLimits"]
    )
    assert canonical_bytes(manifest).decode() == case["manifestCanonical"]
    if "finiteRequestCanonical" in case:
        finite = validate_finite_inference_request(
            json.loads(case["finiteRequestCanonical"])
        )
        assert canonical_bytes(finite).decode() == case["finiteRequestCanonical"]
        assert canonical_digest(finite) == case["finiteRequestDigest"]
        assert (
            finite_input_identity(finite)
            == manifest["completed_inputs"][0]["input_identity"]
        )


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("schema_version", [2, 4])
def test_captured_durable_requests_reject_old_and_future_versions(case, schema_version):
    request = json.loads(case["requestCanonical"])
    request["schema_version"] = schema_version
    with pytest.raises(ValueError, match="unsupported model run schema_version"):
        validate_model_run_request(request)


@pytest.mark.parametrize("schema_version", [1, 3])
def test_captured_finite_requests_reject_old_and_future_versions(schema_version):
    request = json.loads(FIXTURE["cases"][0]["finiteRequestCanonical"])
    request["schema_version"] = schema_version
    with pytest.raises(ValueError, match="unsupported finite inference schema_version"):
        validate_finite_inference_request(request)


def test_later_uploads_do_not_change_an_already_completed_units_identity():
    case = FIXTURE["cases"][1]
    request = json.loads(case["requestCanonical"])
    bindings = json.loads(case["deliveryCanonical"])["source_bindings"]
    input_id = request["inputs"][0]["input_id"]
    assert (
        model_run_execution_identity_digest(
            request, input_id, 0, FIXTURE["modelRunLimits"], bindings[:1]
        )
        == case["unitExecutionDigests"][0]
    )
    changed = copy.deepcopy(bindings)
    changed[0]["source"]["artifact"]["bytes_digest"] = "f" * 64
    assert (
        model_run_execution_identity_digest(
            request, input_id, 0, FIXTURE["modelRunLimits"], changed
        )
        != case["unitExecutionDigests"][0]
    )
