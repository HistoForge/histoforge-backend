"""Cross-language deployment health and capability contract fixtures."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud import capabilities as runtime_capabilities
from histoforge.cloud._config import (
    DEPLOYMENT_CONFIG,
    MAX_ACTIVE_ADMIN_JOBS,
    MAX_ADMIN_REQUEST_BYTES,
    MAX_CATALOG_PAGE_SIZE,
    MAX_INSTALLED_BYTES,
    MAX_JSON_REQUEST_BYTES,
    MAX_MODULE_BYTES,
    validate_gateway_capabilities,
    validate_gateway_health,
)
from histoforge.cloud.inference.gpu import INFERENCE_GPU_OVERRIDES

FIXTURE = (
    Path(__file__).parent / "fixtures" / "contracts" / "gateway-capabilities-v1.json"
)


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_shared_health_fixture_has_exact_bytes_digest_and_roundtrip():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["healthCases"]:
        value = validate_gateway_health(case["value"])
        encoded = canonical_bytes(value)
        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert validate_gateway_health(value) == value, case["name"]


def test_shared_capability_fixtures_have_exact_bytes_digest_and_roundtrip():
    for case in _fixture()["capabilityCases"]:
        value = validate_gateway_capabilities(case["value"])
        encoded = canonical_bytes(value)
        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert validate_gateway_capabilities(value) == value, case["name"]


def test_synthetic_field_binding_case_detects_swapped_fields():
    fixture = _fixture()
    source = copy.deepcopy(fixture["capabilityCases"][0]["value"])
    for mutation in fixture["fieldBindingCase"]["mutations"]:
        _set_path(source, mutation["path"], mutation["value"])

    assert validate_gateway_capabilities(source) == source


def test_shared_invalid_handshake_mutations_fail_closed():
    fixture = _fixture()
    bases = {
        "health": fixture["healthCases"][0]["value"],
        "capabilities": fixture["capabilityCases"][0]["value"],
    }
    validators = {
        "health": validate_gateway_health,
        "capabilities": validate_gateway_capabilities,
    }

    for case in fixture["invalidMutations"]:
        value = copy.deepcopy(bases[case["target"]])
        if case["operation"] == "remove":
            parent = value
            for segment in case["path"][:-1]:
                parent = parent[segment]
            field = case["path"][-1]
            del parent[field]
        else:
            _set_path(value, case["path"], case["value"])
        with pytest.raises(ValueError):
            validators[case["target"]](value)


def test_validated_capabilities_are_detached_from_input():
    source = copy.deepcopy(_fixture()["capabilityCases"][0]["value"])
    validated = validate_gateway_capabilities(source)
    source["limits"]["max_active_admin_jobs"] = 99
    source["gateway_build"]["version"] = "mutated"

    assert validated["limits"]["max_active_admin_jobs"] == 4
    assert validated["gateway_build"]["version"] == "0.1.0+abc123"


def test_runtime_limits_equal_the_frozen_target_fixture():
    target = _fixture()["capabilityCases"][0]["value"]
    limits = target["limits"]
    assert MAX_JSON_REQUEST_BYTES == limits["max_json_request_bytes"]
    assert MAX_CATALOG_PAGE_SIZE == limits["max_catalog_page_size"]
    assert MAX_ADMIN_REQUEST_BYTES == limits["max_admin_request_bytes"]
    assert MAX_MODULE_BYTES == limits["max_module_bytes"]
    assert MAX_INSTALLED_BYTES == limits["max_installed_bytes"]
    assert MAX_ACTIVE_ADMIN_JOBS == limits["max_active_admin_jobs"]
    assert DEPLOYMENT_CONFIG.limits.wire_value() == limits
    assert (
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
        == target["source_artifact_limits"]
    )
    assert (
        DEPLOYMENT_CONFIG.training_api_limits.wire_value()
        == target["training_api_limits"]
    )


def test_uploaded_image_allowance_keeps_execution_and_byte_limits_separate():
    assert DEPLOYMENT_CONFIG.limits.max_active_source_artifacts == 1_024
    assert DEPLOYMENT_CONFIG.limits.max_active_model_runs == 4
    assert DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes == 1024**4
    assert DEPLOYMENT_CONFIG.source_artifact_limits.max_artifact_bytes == 32 * 1024**3
    assert DEPLOYMENT_CONFIG.source_artifact_limits.max_chunks == 4_096
    assert DEPLOYMENT_CONFIG.source_artifact_limits.max_chunk_bytes == 8 * 1024**2


def test_runtime_capabilities_bind_state_registry_build_and_config(monkeypatch):
    registry = {"registry_digest": "c" * 64, "definitions": {}}
    build = {
        "version": "runtime",
        "artifact_digest": "a" * 64,
        "manifest_digest": "b" * 64,
    }
    captured = {}

    def fake_runtime_build(component, policy):
        captured["component"] = component
        captured["policy"] = policy
        return SimpleNamespace(identity=build)

    monkeypatch.setattr(runtime_capabilities, "runner_registry", lambda: registry)
    monkeypatch.setattr(runtime_capabilities, "runtime_build", fake_runtime_build)

    value = runtime_capabilities.gateway_capabilities(
        {
            "deployment_id": "deployment-a",
            "auth_revision": 9,
            "token_sha256": "d" * 64,
        }
    )

    assert value["deployment_id"] == "deployment-a"
    assert value["auth_revision"] == 9
    assert value["gateway_build"] == build
    assert value["runner_registry_digest"] == registry["registry_digest"]
    assert value["inference_gpus"] == list(INFERENCE_GPU_OVERRIDES)
    assert value["limits"] == DEPLOYMENT_CONFIG.limits.wire_value()
    assert captured["component"] == "histoforge-gateway"
    assert captured["policy"]["runner_registry_digest"] == "c" * 64
    assert captured["policy"]["inference_gpus"] == list(INFERENCE_GPU_OVERRIDES)
    versions = {
        "api_major": 1,
        "module_schema_version": 3,
        "source_artifact_schema_version": 2,
        "finite_inference_schema_version": 2,
        "durable_model_run_schema_version": 3,
        "training_api_schema_version": 1,
    }
    for field, version in versions.items():
        assert value[field] == version
        assert captured["policy"][field] == version
    assert runtime_capabilities.gateway_health() == {
        "api_major": 1,
        "status": "ok",
    }


def test_supported_handshake_requires_every_current_schema_version():
    value = _fixture()["capabilityCases"][0]["value"]
    assert runtime_capabilities.require_supported_gateway_capabilities(value) == value

    for field in (
        "api_major",
        "module_schema_version",
        "source_artifact_schema_version",
        "finite_inference_schema_version",
        "durable_model_run_schema_version",
        "training_api_schema_version",
    ):
        for unsupported in {value[field] - 1, value[field] + 1} - {0}:
            incompatible = copy.deepcopy(value)
            incompatible[field] = unsupported
            with pytest.raises(
                ValueError, match="unsupported gateway capability versions"
            ):
                runtime_capabilities.require_supported_gateway_capabilities(
                    incompatible
                )


def _set_path(value, path, replacement):
    parent = value
    for segment in path[:-1]:
        parent = parent[segment]
    parent[path[-1]] = replacement
