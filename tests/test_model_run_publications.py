"""Cross-language durable model-run publication and status contract tests."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud._errors import build_gateway_error
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    build_model_result_manifest,
    model_run_execution_identity_digest,
    preprocessing_context_digest,
    validate_model_result_manifest,
)
from histoforge.cloud.inference.runs.request import (
    validate_model_run_admission,
    validate_model_run_resource_limits,
)
from histoforge.cloud.inference.runs.status import (
    validate_model_run_status,
    validate_model_run_status_projection,
)
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _request(request_case_index):
    request_fixture = _fixture("model-run-request-v1.json")
    case = request_fixture["cases"][request_case_index]
    finite_case = _fixture("finite-inference-v1.json")["cases"][
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
                    "scope": {
                        "type": "field",
                        "region": copy.deepcopy(region),
                    },
                },
                "raster_input": copy.deepcopy(finite["raster_input"]),
            }
        ],
        "result_budget": copy.deepcopy(finite["result_budget"]),
    }
    profile = None
    if "largeRegionProfileCase" in case:
        profile = _fixture("large-region-execution-v1.json")["profiles"][
            case["largeRegionProfileCase"]
        ]["input"]
        request["expected_profile"] = copy.deepcopy(profile)
    if "resumeFrom" in case:
        request["resume_from"] = copy.deepcopy(case["resumeFrom"])
    return validate_model_run_admission(
        request,
        finite_case["route"]["moduleRef"],
        finite_case["route"]["modelId"],
        request["expected_backend"],
        case["targetRasterLimits"],
        profile,
        request_fixture["testModelRunLimits"],
    )


def _limits():
    return _fixture("model-run-request-v1.json")["testModelRunLimits"]


def _artifact(artifact_id, media_type, size_bytes, sha256, content_encoding=None):
    value = {
        "artifact_id": artifact_id,
        "media_type": media_type,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }
    if content_encoding is not None:
        value["content_encoding"] = content_encoding
    return value


def _manifest_preimage(test_case, request, plan):
    input_id = request["inputs"][0]["input_id"]
    preimage = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "model_run_id": request["model_run_id"],
        "action_id": request["action_id"],
        "backend": copy.deepcopy(request["expected_backend"]),
        "request_digest": sha256_bytes(canonical_bytes(request)),
        "completed_units": [],
        "checkpoints": [],
        "failed_units": [],
        "completed_inputs": [],
        "complete": test_case["phase"] == "COMPLETE",
    }
    if test_case["name"] == "complete-finite":
        preimage["completed_inputs"] = [
            {
                "input_id": input_id,
                "input_identity": build_backend_input_identity(
                    request["inputs"][0]["raster_input"]
                ),
                "execution_plan": copy.deepcopy(plan),
                "feature_stream_artifact": copy.deepcopy(
                    test_case["featureStreamArtifact"]
                ),
                "feature_count": 0,
                "vertex_count": 0,
            }
        ]
    elif test_case["name"] == "partial-core-halo":
        preimage["completed_units"] = [
            {
                "input_id": input_id,
                "unit_id": 0,
                "reusable_state": copy.deepcopy(test_case["reusableStateArtifact"]),
                "execution_identity_digest": test_case["executionIdentityDigest"],
            }
        ]
    else:
        preimage["checkpoints"] = [
            {
                "input_id": input_id,
                "unit_id": 0,
                "sequence": test_case["checkpointSequence"],
                "state": copy.deepcopy(test_case["checkpointStateArtifact"]),
                "execution_identity_digest": test_case["executionIdentityDigest"],
            }
        ]
        preimage["failed_units"] = [
            {
                "input_id": input_id,
                "unit_id": 0,
                "failure": {
                    "schema_version": 1,
                    "code": "RUNNER_FAILED",
                    "message": "model tile failed",
                },
            }
        ]
    return preimage


def _status(test_case, request, manifest):
    request_bytes = canonical_bytes(request)
    manifest_bytes = canonical_bytes(manifest)
    return {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "model_run_id": request["model_run_id"],
        "phase": test_case["phase"],
        "request_artifact": _artifact(
            test_case["requestArtifactId"],
            "application/json",
            len(request_bytes),
            sha256_bytes(request_bytes),
        ),
        "completed_unit_count": 0
        if test_case["name"] == "partial-global-checkpoint"
        else 1,
        "total_unit_count": 1,
        "terminal_manifest": _artifact(
            test_case["terminalManifestArtifactId"],
            "application/json",
            len(manifest_bytes),
            sha256_bytes(manifest_bytes),
        ),
        "phase_deadline": test_case["phaseDeadline"],
    }


def _case_values(index):
    test_case = _fixture("model-run-publication-v1.json")["cases"][index]
    request = _request(test_case["requestFixtureCase"])
    plan = build_expected_model_execution_plans(request, _limits())[
        request["inputs"][0]["input_id"]
    ]
    manifest = build_model_result_manifest(
        _manifest_preimage(test_case, request, plan), request, _limits()
    )
    return test_case, request, plan, manifest, _status(test_case, request, manifest)


def test_shared_manifests_plans_and_statuses_are_exact():
    fixture = _fixture("model-run-publication-v1.json")
    assert fixture["schemaVersion"] == 1
    for index, test_case in enumerate(fixture["cases"]):
        _, request, plan, manifest, status = _case_values(index)
        assert sha256_bytes(canonical_bytes(plan)) == test_case["planSha256"]
        if "executionIdentityDigest" in test_case:
            assert (
                model_run_execution_identity_digest(
                    request, request["inputs"][0]["input_id"], 0, _limits()
                )
                == test_case["executionIdentityDigest"]
            )
        assert manifest["manifest_digest"] == test_case["manifestDigest"]
        manifest_bytes = canonical_bytes(manifest)
        assert len(manifest_bytes) == test_case["manifestArtifactSize"]
        assert sha256_bytes(manifest_bytes) == test_case["manifestArtifactSha256"]
        assert validate_model_result_manifest(manifest, request, _limits()) == manifest
        validated_status = validate_model_run_status_projection(
            status, request, _limits(), manifest
        )
        assert (
            sha256_bytes(canonical_bytes(validated_status)) == test_case["statusSha256"]
        )


@pytest.mark.parametrize("schema_version", [3, 5])
def test_old_and_future_manifest_and_status_schemas_are_rejected(schema_version):
    _, request, _, manifest, status = _case_values(0)
    manifest["schema_version"] = schema_version
    with pytest.raises(
        ValueError, match="unsupported model run publication schema_version"
    ):
        validate_model_result_manifest(manifest, request, _limits())
    status["schema_version"] = schema_version
    with pytest.raises(ValueError, match="unsupported model run status schema_version"):
        validate_model_run_status(status)


def test_preprocessing_and_execution_identity_bind_the_settled_inputs():
    _, request, plan, _, _ = _case_values(1)
    item = request["inputs"][0]
    first = preprocessing_context_digest(item, request["expected_backend"])
    assert first == plan["preprocessing_context_digest"]

    changed_backend = copy.deepcopy(request["expected_backend"])
    changed_backend["preprocessing_contract_digest"] = "5" * 64
    assert preprocessing_context_digest(item, changed_backend) != first

    changed_request = copy.deepcopy(request)
    changed_request["action_id"] = "51515151-5151-4151-8151-515151515151"
    assert model_run_execution_identity_digest(
        changed_request, item["input_id"], 0, _limits()
    ) != model_run_execution_identity_digest(request, item["input_id"], 0, _limits())


def test_changed_halo_cannot_reuse_a_unit_from_the_prior_plan():
    case, request, plan, _, _ = _case_values(1)
    changed_request = copy.deepcopy(request)
    changed_request["halo_px_override"] = plan["resolved_halo_px"] + 1
    changed_plan = build_expected_model_execution_plans(changed_request, _limits())[
        request["inputs"][0]["input_id"]
    ]
    assert changed_plan["resolved_halo_px"] != plan["resolved_halo_px"]
    assert changed_plan["plan_digest"] != plan["plan_digest"]
    preimage = _manifest_preimage(case, changed_request, changed_plan)
    with pytest.raises(ValueError, match="execution_identity_digest"):
        build_model_result_manifest(preimage, changed_request, _limits())


@pytest.mark.parametrize("code", ["INSUFFICIENT_CONTEXT", "RESULT_LIMIT_EXCEEDED"])
def test_partial_failure_is_digest_bound_and_matches_status(code):
    case, request, plan, without_failure, _ = _case_values(1)
    failure = build_gateway_error(code, "the actual execution failure")
    preimage = _manifest_preimage(case, request, plan)
    preimage["failure"] = failure
    manifest = build_model_result_manifest(preimage, request, _limits())
    assert manifest["failure"] == failure
    assert manifest["manifest_digest"] != without_failure["manifest_digest"]
    assert validate_model_result_manifest(manifest, request, _limits()) == manifest

    status = _status(case, request, manifest)
    status["failure"] = failure
    assert (
        validate_model_run_status_projection(status, request, _limits(), manifest)
        == status
    )
    for replacement in (None, build_gateway_error("RUNNER_FAILED", "wrong cause")):
        invalid_manifest = copy.deepcopy(manifest)
        invalid_status = copy.deepcopy(status)
        if replacement is None:
            del invalid_manifest["failure"]
            del invalid_status["failure"]
        else:
            invalid_manifest["failure"] = replacement
            invalid_status["failure"] = replacement
        with pytest.raises(ValueError, match="manifest_digest"):
            validate_model_result_manifest(invalid_manifest, request, _limits())
        with pytest.raises(ValueError, match="status failure"):
            validate_model_run_status_projection(
                invalid_status, request, _limits(), manifest
            )
    status_without_failure = _status(case, request, without_failure)
    status_without_failure["failure"] = failure
    with pytest.raises(ValueError, match="status failure"):
        validate_model_run_status_projection(
            status_without_failure, request, _limits(), without_failure
        )


def test_complete_forbids_failure_and_partial_failure_cannot_be_null():
    case, request, plan, _, status = _case_values(0)
    failure = build_gateway_error("RUNNER_FAILED", "actual failure")
    preimage = _manifest_preimage(case, request, plan)
    preimage["failure"] = failure
    with pytest.raises(ValueError, match="complete manifest cannot carry failure"):
        build_model_result_manifest(preimage, request, _limits())
    status["failure"] = failure
    with pytest.raises(ValueError, match="COMPLETE status requires only"):
        validate_model_run_status(status)

    case, request, plan, _, status = _case_values(1)
    preimage = _manifest_preimage(case, request, plan)
    preimage["failure"] = None
    with pytest.raises(ValueError, match="must be an object"):
        build_model_result_manifest(preimage, request, _limits())
    status["failure"] = None
    with pytest.raises(ValueError, match="must be an object"):
        validate_model_run_status(status)


def test_manifest_rejects_cross_record_plan_identity_and_coverage_mismatch():
    _, request, _, manifest, _ = _case_values(1)

    invalid = copy.deepcopy(manifest)
    invalid["action_id"] = "51515151-5151-4151-8151-515151515151"
    invalid["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="action_id"):
        validate_model_result_manifest(invalid, request, _limits())

    preimage = {key: copy.deepcopy(value) for key, value in manifest.items()}
    del preimage["manifest_digest"]
    preimage["action_id"] = "51515151-5151-4151-8151-515151515151"
    with pytest.raises(ValueError, match="action_id"):
        build_model_result_manifest(preimage, request, _limits())

    preimage = _manifest_preimage(
        _fixture("model-run-publication-v1.json")["cases"][1],
        request,
        build_expected_model_execution_plans(request, _limits())[
            request["inputs"][0]["input_id"]
        ],
    )
    preimage["completed_units"][0]["execution_identity_digest"] = "0" * 64
    with pytest.raises(ValueError, match="execution_identity_digest"):
        build_model_result_manifest(preimage, request, _limits())

    preimage = _manifest_preimage(
        _fixture("model-run-publication-v1.json")["cases"][1],
        request,
        build_expected_model_execution_plans(request, _limits())[
            request["inputs"][0]["input_id"]
        ],
    )
    preimage["complete"] = True
    with pytest.raises(ValueError, match="cover every requested input"):
        build_model_result_manifest(preimage, request, _limits())


def test_plan_specific_publications_and_feature_stream_contract_fail_closed():
    global_case, global_request, _, _, _ = _case_values(2)
    global_plan = build_expected_model_execution_plans(global_request, _limits())[
        global_request["inputs"][0]["input_id"]
    ]
    preimage = _manifest_preimage(global_case, global_request, global_plan)
    checkpoint = preimage["checkpoints"].pop()
    preimage["completed_units"] = [
        {
            "input_id": checkpoint["input_id"],
            "unit_id": 0,
            "reusable_state": checkpoint["state"],
            "execution_identity_digest": checkpoint["execution_identity_digest"],
        }
    ]
    with pytest.raises(ValueError, match="core-halo"):
        build_model_result_manifest(preimage, global_request, _limits())

    complete_case, request, plan, _, _ = _case_values(0)
    preimage = _manifest_preimage(complete_case, request, plan)
    completed_input = preimage["completed_inputs"].pop()
    preimage["complete"] = False
    preimage["completed_units"] = [
        {
            "input_id": request["inputs"][0]["input_id"],
            "unit_id": 0,
            "reusable_state": completed_input["feature_stream_artifact"],
            "execution_identity_digest": model_run_execution_identity_digest(
                request,
                request["inputs"][0]["input_id"],
                0,
                _limits(),
            ),
        }
    ]
    with pytest.raises(ValueError, match="core-halo"):
        build_model_result_manifest(preimage, request, _limits())

    preimage = _manifest_preimage(complete_case, request, plan)
    preimage["completed_inputs"][0]["feature_stream_artifact"]["media_type"] = (
        "application/json"
    )
    with pytest.raises(ValueError, match="media_type"):
        build_model_result_manifest(preimage, request, _limits())

    preimage = _manifest_preimage(complete_case, request, plan)
    preimage["completed_inputs"][0]["vertex_count"] = 3
    with pytest.raises(ValueError, match="zero together"):
        build_model_result_manifest(preimage, request, _limits())


def test_resource_and_manifest_byte_budgets_are_enforced():
    complete_case, request, plan, _, _ = _case_values(0)
    limits = copy.deepcopy(_limits())
    limits["max_request_canonical_bytes"] = len(canonical_bytes(request)) - 1
    with pytest.raises(ValueError, match="max_request_canonical_bytes"):
        validate_model_run_resource_limits(request, limits)

    limits = copy.deepcopy(_limits())
    limits["result"]["max_features"] = request["result_budget"]["max_features"] - 1
    with pytest.raises(ValueError, match="result_budget"):
        validate_model_run_resource_limits(request, limits)

    preimage = _manifest_preimage(complete_case, request, plan)
    preimage["completed_inputs"][0]["feature_stream_artifact"]["size_bytes"] = (
        request["result_budget"]["max_transferred_bytes"] + 1
    )
    with pytest.raises(ValueError, match="max_transferred_bytes"):
        build_model_result_manifest(preimage, request, _limits())

    core_case, core_request, core_plan, _, _ = _case_values(1)
    preimage = _manifest_preimage(core_case, core_request, core_plan)
    limits = copy.deepcopy(_limits())
    limits["max_recovery_state_bytes_per_unit"] = (
        preimage["completed_units"][0]["reusable_state"]["size_bytes"] - 1
    )
    with pytest.raises(ValueError, match="per-unit byte limit"):
        build_model_result_manifest(preimage, core_request, limits)

    limits = copy.deepcopy(_limits())
    limits["max_recovery_state_bytes_per_run"] = (
        preimage["completed_units"][0]["reusable_state"]["size_bytes"] - 1
    )
    with pytest.raises(ValueError, match="run byte limit"):
        build_model_result_manifest(preimage, core_request, limits)


def test_status_phase_payload_and_artifact_bindings_fail_closed():
    _, request, _, manifest, status = _case_values(0)

    invalid = copy.deepcopy(status)
    invalid["phase"] = "FAILED"
    with pytest.raises(ValueError, match="FAILED status requires"):
        validate_model_run_status(invalid)

    invalid = copy.deepcopy(status)
    invalid["request_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="request_artifact"):
        validate_model_run_status_projection(invalid, request, _limits(), manifest)

    invalid = copy.deepcopy(status)
    invalid["total_unit_count"] = 2
    with pytest.raises(ValueError, match="total_unit_count"):
        validate_model_run_status_projection(invalid, request, _limits(), manifest)

    invalid = copy.deepcopy(status)
    invalid["phase"] = "PARTIAL"
    with pytest.raises(ValueError, match="completeness"):
        validate_model_run_status_projection(invalid, request, _limits(), manifest)

    queued = {
        key: copy.deepcopy(status[key])
        for key in (
            "schema_version",
            "model_run_id",
            "request_artifact",
            "total_unit_count",
            "phase_deadline",
        )
    }
    queued["phase"] = "QUEUED"
    queued["completed_unit_count"] = 0
    assert (
        validate_model_run_status_projection(queued, request, _limits())["phase"]
        == "QUEUED"
    )
    queued["completed_unit_count"] = 1
    with pytest.raises(ValueError, match="zero completed"):
        validate_model_run_status_projection(queued, request, _limits())
