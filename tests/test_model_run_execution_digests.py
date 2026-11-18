"""Common mask/plan encoding must preserve the complete original JCS preimage."""

import copy

import pytest
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.inference.runs import publications
from histoforge.cloud.inference.runs.request import validate_model_run_resource_limits

from test_model_run_publications import _case_values, _limits, _request
from test_model_run_source_delivery import _binding, _captured_request


def _original_digest(request, model_input, plan, unit, bindings):
    # Independent pre-optimization formula, including the whole common objects.
    preimage = {
        "action_id": request["action_id"],
        "producer": request["producer"],
        "expected_backend": request["expected_backend"],
        "input": model_input,
        "execution_plan": plan,
        "unit_id": unit,
    }
    if "expected_profile" in request:
        preimage["expected_profile"] = request["expected_profile"]
    if "capture" in model_input["raster_input"]["source_view"]:
        if plan["type"] == "core-halo" and request["expected_backend"][
            "architecture"
        ] in {
            "dinov3-adapter-cellpose",
            "dinov3-adapter-hovernet",
            "dinov3-adapter-semantic",
        }:
            preimage["source_bindings"] = [
                {"unit_id": item["unit_id"], "source": item["source"]}
                for item in bindings
                if item["input_id"] == model_input["input_id"]
            ]
        else:
            preimage["source_binding"] = next(
                item["source"]
                for item in bindings
                if (item["input_id"], item["unit_id"])
                == (model_input["input_id"], unit)
            )
    return canonical_digest(preimage)


def _keys_and_expected(request, bindings):
    normalized = validate_model_run_resource_limits(request, _limits())
    plans = publications.build_expected_model_execution_plans(normalized, _limits())
    return {
        (item["input_id"], unit): _original_digest(
            normalized, item, plans[item["input_id"]], unit, bindings
        )
        for item in normalized["inputs"]
        for unit in range(len(plans[item["input_id"]].get("windows", [None])))
    }


@pytest.mark.parametrize("request_case", range(5))
def test_sealed_finite_core_and_global_preimages_are_identical(request_case):
    request = _request(request_case)
    expected = _keys_and_expected(request, [])
    assert (
        publications.model_run_execution_identity_digests(
            request, list(expected), _limits()
        )
        == expected
    )
    for (input_id, unit), digest in expected.items():
        assert (
            publications.model_run_execution_identity_digest(
                request, input_id, unit, _limits()
            )
            == digest
        )


@pytest.mark.parametrize("core_halo", [False, True])
def test_captured_sources_preserve_inner_binding_preimage(core_halo):
    request = _captured_request(core_halo)
    bindings = [_binding(request, unit) for unit in range(2 if core_halo else 1)]
    expected = _keys_and_expected(request, bindings)
    assert (
        publications.model_run_execution_identity_digests(
            request, list(expected), _limits(), bindings
        )
        == expected
    )
    for (input_id, unit), digest in expected.items():
        assert (
            publications.model_run_execution_identity_digest(
                request, input_id, unit, _limits(), bindings
            )
            == digest
        )
    assert (
        publications.model_run_execution_identity_digests(
            request, list(expected), _limits(), []
        )
        == {}
    )
    with pytest.raises(ValueError, match="no accepted source|every input source"):
        publications.model_run_execution_identity_digest(
            request, request["inputs"][0]["input_id"], 0, _limits(), []
        )


def _count_common_encodings(monkeypatch):
    encoded = []
    original = publications.canonical_bytes

    def encode(value):
        if isinstance(value, dict) and "execution_plan" in value and "input" in value:
            encoded.append(copy.deepcopy(value))
        return original(value)

    monkeypatch.setattr(publications, "canonical_bytes", encode)
    return encoded


def test_batch_encodes_each_used_input_once_and_never_reuses_between_calls(monkeypatch):
    request = _captured_request()
    first = request["inputs"][0]
    second = copy.deepcopy(first)
    second["input_id"] = first["input_id"] + 1
    second["output_generation_id"] = "88888888-8888-4888-8888-888888888888"
    second["output"]["scope"].update(
        type="parent", parent_id="77777777-7777-4777-8777-777777777777"
    )
    request["inputs"].append(second)
    bindings = [_binding(request, unit) for unit in (0, 1)]
    bindings += [
        {**copy.deepcopy(item), "input_id": second["input_id"]} for item in bindings[:]
    ]
    expected = _keys_and_expected(request, bindings)
    encoded = _count_common_encodings(monkeypatch)
    keys = list(expected)
    assert (
        publications.model_run_execution_identity_digests(
            request, keys + keys + [(999, 0)], _limits(), bindings
        )
        == expected
    )
    assert len(encoded) == 2

    # A new verification operation must see changes, not an earlier hash state.
    changed = copy.deepcopy(request)
    changed["action_id"] = "99999999-9999-4999-8999-999999999999"
    next_expected = _keys_and_expected(changed, bindings)
    assert next_expected != expected
    assert (
        publications.model_run_execution_identity_digests(
            changed, keys, _limits(), bindings
        )
        == next_expected
    )
    assert len(encoded) == 4


def test_publication_collection_reuses_one_prefix_without_changing_fixture(monkeypatch):
    _, request, plan, manifest, _ = _case_values(1)
    # The fixture has one core; repeated validation still demonstrates that the
    # collection validator shares an operation-local prefix for all publications.
    publication = manifest["completed_units"][0]
    values = {
        "source_bindings": [],
        "completed_units": [publication, copy.deepcopy(publication)],
        "checkpoints": [],
        "failed_units": [],
        "completed_inputs": [],
    }
    encoded = _count_common_encodings(monkeypatch)
    assert (
        publications.validate_model_run_publications(values, request, _limits())
        == values
    )
    assert len(encoded) == 1
    assert encoded[0]["execution_plan"] == plan
    # Uniqueness remains the manifest owner's responsibility; it still rejects.
    invalid = {**manifest, "completed_units": values["completed_units"]}
    with pytest.raises(ValueError, match="duplicate"):
        publications.validate_model_result_manifest(invalid, request, _limits())
