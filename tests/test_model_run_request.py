"""Cross-language durable model-run request contract tests."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud.execution_limits import (
    fits_target_raster,
    validate_target_raster_limits,
)
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.request import (
    model_run_request_digest,
    validate_model_run_acceptance,
    validate_model_run_accepted,
    validate_model_run_admission,
    validate_model_run_plan_family,
    validate_model_run_request,
    validate_model_run_resource_limits,
    validate_model_run_route_request,
)
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION
from histoforge.cloud.rasterization.inputs import geometry_vertex_count

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _json_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _case_request(case):
    finite_case = _json_fixture("finite-inference-v1.json")["cases"][
        case["finiteFixtureCase"]
    ]
    finite = json.loads(finite_case["requestCanonical"])
    region = finite["raster_input"]["frame"]["region"]
    request = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": case["actionId"],
        "model_run_id": case["modelRunId"],
        "producer": copy.deepcopy(finite["producer"]),
        "expected_backend": copy.deepcopy(finite["expected_backend"]),
        "inputs": [
            {
                "input_id": case["inputId"],
                "output_generation_id": case["outputGenerationId"],
                "output": {
                    "producer": copy.deepcopy(finite["producer"]),
                    "scope": {"type": "field", "region": region},
                },
                "raster_input": copy.deepcopy(finite["raster_input"]),
            }
        ],
        "result_budget": copy.deepcopy(finite["result_budget"]),
    }
    profile = None
    if "largeRegionProfileCase" in case:
        profile = _json_fixture("large-region-execution-v1.json")["profiles"][
            case["largeRegionProfileCase"]
        ]["input"]
        request["expected_profile"] = profile
    if "resumeFrom" in case:
        request["resume_from"] = case["resumeFrom"]
    if "haloPxOverride" in case:
        request["halo_px_override"] = case["haloPxOverride"]
    return request, profile, finite_case["route"]


def _accepted(case):
    return {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": case["actionId"],
        "model_run_id": case["modelRunId"],
        "request_digest": case["requestSha256"],
    }


def test_shared_requests_plan_families_and_acceptances_are_exact():
    fixture = _json_fixture("model-run-request-v1.json")
    assert fixture["schemaVersion"] == 1
    for case in fixture["cases"]:
        raw, profile, route = _case_request(case)
        request = validate_model_run_admission(
            raw,
            route["moduleRef"],
            route["modelId"],
            raw["expected_backend"],
            case["targetRasterLimits"],
            profile,
            fixture["testModelRunLimits"],
        )
        encoded = canonical_bytes(request)
        assert sha256_bytes(encoded) == case["requestSha256"], case["name"]
        assert model_run_request_digest(request) == case["requestSha256"]
        accepted = validate_model_run_acceptance(request, _accepted(case))
        assert sha256_bytes(canonical_bytes(accepted)) == case["acceptedSha256"]


def test_target_raster_fit_uses_all_three_inclusive_limits():
    case = _json_fixture("model-run-request-v1.json")["cases"][0]
    request, _, _ = _case_request(case)
    raster = request["inputs"][0]["raster_input"]["frame"]["transform"]["raster"]
    exact = validate_target_raster_limits(case["targetRasterLimits"])
    assert fits_target_raster(raster, exact)
    for field in ("max_width", "max_height", "max_pixels"):
        too_small = dict(exact)
        too_small[field] -= 1
        assert not fits_target_raster(raster, too_small)


def test_request_rejects_identity_scope_and_input_collection_mismatch():
    case = _json_fixture("model-run-request-v1.json")["cases"][0]
    request, _, _ = _case_request(case)

    invalid = copy.deepcopy(request)
    invalid["inputs"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_model_run_request(invalid)

    second = copy.deepcopy(request["inputs"][0])
    second["output_generation_id"] = "11111111-1111-4111-8111-111111111111"
    second["output"]["scope"] = {
        "type": "parent",
        "region": copy.deepcopy(second["raster_input"]["frame"]["region"]),
        "parent_id": "22222222-2222-4222-8222-222222222222",
    }
    invalid = copy.deepcopy(request)
    invalid["inputs"].append(second)
    with pytest.raises(ValueError, match="duplicate input_id"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["inputs"][0]["input_id"] = 11
    second["input_id"] = 3
    invalid["inputs"].append(second)
    with pytest.raises(ValueError, match="strictly increasing input_id"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["inputs"][0]["output"]["producer"]["step_id"] = "lumen"
    with pytest.raises(ValueError, match="output producer"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["inputs"][0]["output"]["scope"]["region"]["roi_id"] = (
        "11111111-1111-4111-8111-111111111111"
    )
    with pytest.raises(ValueError, match="scope region"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["expected_backend"]["rasterizer"]["rasterization_contract_digest"] = (
        "1" * 64
    )
    with pytest.raises(ValueError, match="input rasterizer"):
        validate_model_run_request(invalid)

    second = copy.deepcopy(request["inputs"][0])
    second["input_id"] = 1
    invalid = copy.deepcopy(request)
    invalid["inputs"].append(second)
    with pytest.raises(ValueError, match="duplicate output_generation_id"):
        validate_model_run_request(invalid)

    second["output_generation_id"] = "11111111-1111-4111-8111-111111111111"
    invalid = copy.deepcopy(request)
    invalid["inputs"].append(second)
    with pytest.raises(ValueError, match="duplicate output"):
        validate_model_run_request(invalid)


def test_route_plan_family_profile_and_acceptance_mismatch_fail_closed():
    fixture = _json_fixture("model-run-request-v1.json")
    finite, _, route = _case_request(fixture["cases"][0])
    oversized, profile, _ = _case_request(fixture["cases"][1])

    with pytest.raises(ValueError, match="route module_ref"):
        validate_model_run_route_request(finite, "histoforge/leydig", route["modelId"])
    with pytest.raises(ValueError, match="route model_id"):
        validate_model_run_route_request(finite, route["moduleRef"], "lumen")

    wrong_backend = copy.deepcopy(finite["expected_backend"])
    wrong_backend["artifact_digests"]["weights"] = "4" * 64
    with pytest.raises(ValueError, match="deployed backend identity"):
        validate_model_run_admission(
            finite,
            route["moduleRef"],
            route["modelId"],
            wrong_backend,
            fixture["cases"][0]["targetRasterLimits"],
            None,
            fixture["testModelRunLimits"],
        )

    invalid = copy.deepcopy(finite)
    invalid["expected_profile"] = profile
    with pytest.raises(ValueError, match="must omit expected_profile"):
        validate_model_run_plan_family(
            invalid, fixture["cases"][0]["targetRasterLimits"], profile
        )

    invalid = copy.deepcopy(oversized)
    del invalid["expected_profile"]
    with pytest.raises(ValueError, match="must carry expected_profile"):
        validate_model_run_plan_family(
            invalid, fixture["cases"][1]["targetRasterLimits"], profile
        )

    with pytest.raises(ValueError, match="does not advertise"):
        validate_model_run_plan_family(
            oversized, fixture["cases"][1]["targetRasterLimits"], None
        )
    wrong_profile = _json_fixture("large-region-execution-v1.json")["profiles"][0][
        "input"
    ]
    with pytest.raises(ValueError, match="advertised profile"):
        validate_model_run_plan_family(
            oversized,
            fixture["cases"][1]["targetRasterLimits"],
            wrong_profile,
        )

    replacements = {
        "action_id": "11111111-1111-4111-8111-111111111111",
        "model_run_id": "22222222-2222-4222-8222-222222222222",
        "request_digest": "0" * 64,
    }
    for field, replacement in replacements.items():
        accepted = _accepted(fixture["cases"][0])
        accepted[field] = replacement
        with pytest.raises(ValueError, match=field):
            validate_model_run_acceptance(finite, accepted)


def test_mixed_finite_and_oversized_inputs_are_rejected_and_results_detach():
    fixture = _json_fixture("model-run-request-v1.json")
    raw, _, _ = _case_request(fixture["cases"][0])
    raw["inputs"][0]["input_id"] = 3
    second_finite = json.loads(
        _json_fixture("finite-inference-v1.json")["cases"][1]["requestCanonical"]
    )
    second = {
        "input_id": 11,
        "output_generation_id": "11111111-1111-4111-8111-111111111111",
        "output": {
            "producer": raw["producer"],
            "scope": {
                "type": "field",
                "region": second_finite["raster_input"]["frame"]["region"],
            },
        },
        "raster_input": second_finite["raster_input"],
    }
    raw["inputs"].append(second)
    result = validate_model_run_request(raw)
    assert [item["input_id"] for item in result["inputs"]] == [3, 11]
    raw["inputs"][0]["output_generation_id"] = "22222222-2222-4222-8222-222222222222"
    assert (
        result["inputs"][0]["output_generation_id"]
        == (fixture["cases"][0]["outputGenerationId"])
    )
    limits = {"max_width": 512, "max_height": 768, "max_pixels": 600000}
    with pytest.raises(ValueError, match="homogeneous plan family"):
        validate_model_run_plan_family(result, limits, None)


@pytest.mark.parametrize("schema_version", [3, 5])
def test_old_and_future_request_and_acceptance_schemas_are_rejected(schema_version):
    case = _json_fixture("model-run-request-v1.json")["cases"][0]
    request, _, _ = _case_request(case)
    accepted = _accepted(case)
    for value, validate in (
        (request, validate_model_run_request),
        (accepted, validate_model_run_accepted),
    ):
        value["schema_version"] = schema_version
        with pytest.raises(ValueError, match="unsupported model run schema_version"):
            validate(value)


def test_request_schema_and_optional_fields_are_closed():
    case = _json_fixture("model-run-request-v1.json")["cases"][0]
    request, _, _ = _case_request(case)

    invalid = copy.deepcopy(request)
    invalid["schema_version"] = 1
    with pytest.raises(ValueError, match="unsupported model run schema_version"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["resume_from"] = None
    with pytest.raises(ValueError, match="resume source must be an object"):
        validate_model_run_request(invalid)

    invalid = copy.deepcopy(request)
    invalid["resume_from"] = {
        "previous_run_id": invalid["model_run_id"],
        "previous_manifest_digest": "a" * 64,
    }
    with pytest.raises(ValueError, match="fresh model_run_id"):
        validate_model_run_request(invalid)


@pytest.mark.parametrize("halo", [None, -1, True, 0.5, 2**31])
def test_halo_override_rejects_invalid_numbers(halo):
    request, _, _ = _case_request(
        _json_fixture("model-run-request-v1.json")["cases"][1]
    )
    request["halo_px_override"] = halo
    with pytest.raises(ValueError, match="halo_px_override"):
        validate_model_run_request(request)


@pytest.mark.parametrize("case_index", [0, 2])
def test_halo_override_is_illegal_for_finite_and_global_requests(case_index):
    request, _, _ = _case_request(
        _json_fixture("model-run-request-v1.json")["cases"][case_index]
    )
    request["halo_px_override"] = 0
    with pytest.raises(ValueError, match="requires a core-halo"):
        validate_model_run_request(request)


def test_every_durable_run_admission_limit_is_enforced():
    fixture = _json_fixture("model-run-request-v1.json")
    raw, _, _ = _case_request(fixture["cases"][0])
    raw["inputs"][0]["input_id"] = 3
    second_finite = json.loads(
        _json_fixture("finite-inference-v1.json")["cases"][1]["requestCanonical"]
    )
    raw["inputs"].append(
        {
            "input_id": 11,
            "output_generation_id": "11111111-1111-4111-8111-111111111111",
            "output": {
                "producer": copy.deepcopy(raw["producer"]),
                "scope": {
                    "type": "field",
                    "region": copy.deepcopy(
                        second_finite["raster_input"]["frame"]["region"]
                    ),
                },
            },
            "raster_input": copy.deepcopy(second_finite["raster_input"]),
        }
    )
    request = validate_model_run_request(raw)
    base_limits = fixture["testModelRunLimits"]
    assert validate_model_run_resource_limits(request, base_limits) == request

    request_bytes = len(canonical_bytes(request))
    aggregate_pixels = sum(
        item["raster_input"]["frame"]["transform"]["raster"]["width"]
        * item["raster_input"]["frame"]["transform"]["raster"]["height"]
        for item in request["inputs"]
    )
    aggregate_mask_vertices = sum(
        geometry_vertex_count(item["raster_input"]["frame"]["inclusion_mask_raster"])
        for item in request["inputs"]
    )
    mutations = (
        ("max_request_canonical_bytes", request_bytes - 1, "max_request"),
        ("max_inputs", 1, "max_inputs"),
        ("max_units", 1, "max_units"),
        (
            "max_aggregate_target_raster_pixels",
            aggregate_pixels - 1,
            "aggregate target-raster pixels",
        ),
        (
            "max_aggregate_inclusion_mask_vertices",
            aggregate_mask_vertices - 1,
            "aggregate inclusion-mask vertices",
        ),
    )
    for field, value, pattern in mutations:
        limits = copy.deepcopy(base_limits)
        limits[field] = value
        with pytest.raises(ValueError, match=pattern):
            validate_model_run_resource_limits(request, limits)

    limits = copy.deepcopy(base_limits)
    limits["target_raster_per_input"]["max_width"] = 1023
    with pytest.raises(ValueError, match="target_raster_per_input"):
        validate_model_run_resource_limits(request, limits)

    for field, requested in request["result_budget"].items():
        limits = copy.deepcopy(base_limits)
        limits["result"][field] = requested - 1
        with pytest.raises(ValueError, match="result_budget"):
            validate_model_run_resource_limits(request, limits)


def test_processing_raster_limits_use_exact_clipped_custom_halo_windows():
    fixture = _json_fixture("model-run-request-v1.json")
    request, _, _ = _case_request(fixture["cases"][1])
    request["expected_profile"] = build_core_halo_profile(256, 192, 4, "e" * 64)
    request["halo_px_override"] = 0
    limits = copy.deepcopy(fixture["testModelRunLimits"])
    limits["processing_raster"] = {
        "max_width": 256,
        "max_height": 192,
        "max_pixels": 256 * 192,
    }
    assert validate_model_run_resource_limits(request, limits) == request
    request["halo_px_override"] = 1
    with pytest.raises(ValueError, match="processing window exceeds processing_raster"):
        validate_model_run_resource_limits(request, limits)

    # A huge nominal halo is legal when the actual clipped allocation fits exactly.
    request["halo_px_override"] = 2**31 - 1
    limits["processing_raster"] = {
        "max_width": 512,
        "max_height": 384,
        "max_pixels": 512 * 384,
    }
    assert validate_model_run_resource_limits(request, limits) == request
    for field in ("max_width", "max_height", "max_pixels"):
        too_small = copy.deepcopy(limits)
        too_small["processing_raster"][field] -= 1
        with pytest.raises(ValueError, match="processing_raster"):
            validate_model_run_resource_limits(request, too_small)


@pytest.mark.parametrize("case_index", [0, 2])
def test_finite_and_global_processing_limits_bound_the_complete_raster(case_index):
    fixture = _json_fixture("model-run-request-v1.json")
    request, _, _ = _case_request(fixture["cases"][case_index])
    limits = copy.deepcopy(fixture["testModelRunLimits"])
    limits["processing_raster"] = {
        "max_width": 511,
        "max_height": 384,
        "max_pixels": 512 * 384,
    }
    with pytest.raises(ValueError, match="processing raster exceeds processing_raster"):
        validate_model_run_resource_limits(request, limits)
