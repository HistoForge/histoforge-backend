"""The gateway encodes validated worker output once, without changing its bytes."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference.finite import runtime

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "finite-inference-v3.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _outcome(case):
    request = json.loads(case["requestCanonical"])
    result = json.loads(case["resultCanonical"])
    meta = result["meta"]
    return {
        "type": "result",
        **{
            key: meta[key]
            for key in (
                "action_id",
                "output_generation_id",
                "input_id",
                "producer",
                "backend",
                "input_identity",
            )
        },
        "raster": request["raster_input"]["frame"]["transform"]["raster"],
        "result_budget": request["result_budget"],
        "declared_classes": case["declaredClasses"],
        "geojson": result["geojson"],
    }, result


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_shared_result_bytes_are_exact_and_the_worker_outcome_is_not_modified(case):
    outcome, result = _outcome(case)
    before = copy.deepcopy(outcome)

    encoded = runtime.encode_finite_result(outcome, result["meta"]["backend_call_id"])

    assert encoded == case["resultCanonical"].encode("utf-8")
    assert outcome == before


def test_gateway_encodes_features_once_without_a_deep_validation_pass(monkeypatch):
    outcome, result = _outcome(CASES[0])
    encodings = []

    def encode(value):
        encodings.append(value)
        return canonical_bytes(value)

    monkeypatch.setattr(runtime, "canonical_bytes", encode)
    monkeypatch.setattr(
        runtime, "canonical_tree", lambda _value: pytest.fail("no result round trip")
    )
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.validate_inference_features",
        lambda *_args: pytest.fail("features were already validated in the worker"),
    )
    encoded = runtime.encode_finite_result(outcome, result["meta"]["backend_call_id"])

    assert encoded == canonical_bytes(result)
    assert len(encodings) == 1
    assert encodings[0]["geojson"] is outcome["geojson"]


@pytest.mark.parametrize("field", ["max_transferred_bytes", "max_decoded_bytes"])
@pytest.mark.parametrize("call_id", ["normal", "\x01" * 128, 'é/"?\\'])
def test_size_check_uses_exact_encoded_bytes_including_escaped_call_id(field, call_id):
    outcome, result = _outcome(CASES[0])
    result["meta"]["backend_call_id"] = call_id
    expected = canonical_bytes(result)
    outcome["result_budget"][field] = len(expected)
    assert runtime.encode_finite_result(outcome, call_id) == expected

    outcome["result_budget"][field] -= 1
    with pytest.raises(GatewayFailure) as failure:
        runtime.encode_finite_result(outcome, call_id)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    "field,value",
    [
        ("unexpected", True),
        ("action_id", "not-a-uuid"),
        ("input_id", -1),
        ("raster", None),
        ("declared_classes", None),
        ("geojson", {"type": "FeatureCollection", "features": [], "unexpected": True}),
        ("geojson", {"type": "FeatureCollection", "features": None}),
    ],
)
def test_gateway_rejects_malformed_worker_envelope(field, value):
    outcome, _result = _outcome(CASES[0])
    outcome[field] = value
    with pytest.raises(GatewayFailure) as failure:
        runtime.encode_finite_result(outcome, "call")
    assert failure.value.error["code"] == "RUNNER_FAILED"


@pytest.mark.parametrize(
    "field,value", [("step_id", "other-model"), ("module_ref", "other/module")]
)
def test_gateway_rejects_producer_backend_disagreement(field, value):
    outcome, _result = _outcome(CASES[0])
    outcome["producer"][field] = value
    with pytest.raises(GatewayFailure) as failure:
        runtime.encode_finite_result(outcome, "call")
    assert failure.value.error["code"] == "RUNNER_FAILED"


def test_unencodable_worker_result_becomes_a_closed_runner_failure():
    outcome, _result = _outcome(CASES[0])
    outcome["geojson"]["features"] = [object()]
    with pytest.raises(GatewayFailure) as failure:
        runtime.encode_finite_result(outcome, "call")
    assert failure.value.error["code"] == "RUNNER_FAILED"
    assert failure.value.error["message"] == "runner returned an unencodable result"
