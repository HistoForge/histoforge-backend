"""Runtime projection of one authenticated deployment handshake."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from histoforge.cloud._auth import parse_gateway_state
from histoforge.cloud._config import (
    DEPLOYMENT_CONFIG,
    validate_gateway_capabilities,
    validate_gateway_health,
)
from histoforge.cloud.architectures.registry import (
    KNOWN_ARCHITECTURES,
    runner_registry,
)
from histoforge.cloud.inference.gpu import INFERENCE_GPU_OVERRIDES
from histoforge.cloud.protocol_versions import (
    FINITE_INFERENCE_SCHEMA_VERSION,
    GATEWAY_API_MAJOR,
    MODEL_RUN_SCHEMA_VERSION,
    MODULE_SCHEMA_VERSION,
    SOURCE_ARTIFACT_SCHEMA_VERSION,
    TRAINING_API_SCHEMA_VERSION,
)
from histoforge.cloud.runtime_identity import runtime_build


def gateway_health() -> dict[str, Any]:
    """Return the exact unauthenticated liveness value."""

    return validate_gateway_health({"api_major": GATEWAY_API_MAJOR, "status": "ok"})


def gateway_capabilities(state: Any) -> dict[str, Any]:
    """Bind authenticated deployment state to the exact build and limits."""

    authenticated = parse_gateway_state(state)
    registry = runner_registry()
    policy = {
        "schema_version": 1,
        "api_major": GATEWAY_API_MAJOR,
        "module_schema_version": MODULE_SCHEMA_VERSION,
        "source_artifact_schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "finite_inference_schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
        "durable_model_run_schema_version": MODEL_RUN_SCHEMA_VERSION,
        "training_api_schema_version": TRAINING_API_SCHEMA_VERSION,
        "known_architectures": sorted(KNOWN_ARCHITECTURES),
        "runner_registry_digest": registry["registry_digest"],
        "inference_gpus": list(INFERENCE_GPU_OVERRIDES),
        "deployment_config": asdict(DEPLOYMENT_CONFIG),
    }
    gateway_build = runtime_build("histoforge-gateway", policy).identity
    return validate_gateway_capabilities(
        {
            "api_major": GATEWAY_API_MAJOR,
            "module_schema_version": MODULE_SCHEMA_VERSION,
            "source_artifact_schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "finite_inference_schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
            "durable_model_run_schema_version": MODEL_RUN_SCHEMA_VERSION,
            "training_api_schema_version": TRAINING_API_SCHEMA_VERSION,
            "deployment_id": authenticated["deployment_id"],
            "auth_revision": authenticated["auth_revision"],
            "gateway_build": gateway_build,
            "runner_registry_digest": registry["registry_digest"],
            "inference_gpus": list(INFERENCE_GPU_OVERRIDES),
            "source_artifact_limits": (
                DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
            ),
            "training_api_limits": DEPLOYMENT_CONFIG.training_api_limits.wire_value(),
            "limits": DEPLOYMENT_CONFIG.limits.wire_value(),
        }
    )


def gateway_protocol_mismatches(
    capabilities: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Compare validated capabilities with the helper's execution protocols."""

    expected_versions = {
        "api_major": GATEWAY_API_MAJOR,
        "module_schema_version": MODULE_SCHEMA_VERSION,
        "source_artifact_schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "finite_inference_schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
        "durable_model_run_schema_version": MODEL_RUN_SCHEMA_VERSION,
        "training_api_schema_version": TRAINING_API_SCHEMA_VERSION,
    }
    return {
        field: {"gateway": capabilities[field], "helper": expected}
        for field, expected in expected_versions.items()
        if capabilities[field] != expected
    }


def require_supported_gateway_capabilities(value: Any) -> dict[str, Any]:
    """Require a handshake this client/runtime version can execute exactly."""

    capabilities = validate_gateway_capabilities(value)
    mismatches = gateway_protocol_mismatches(capabilities)
    if mismatches:
        raise ValueError(f"unsupported gateway capability versions: {mismatches}")
    return capabilities
