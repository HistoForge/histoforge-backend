"""Closed runner-capability wire contract shared by catalog and dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_tree,
    require_positive_int32,
    require_positive_safe_integer,
    require_sha256,
)
from histoforge.cloud.execution_limits import (
    validate_model_run_limits,
    validate_result_limits,
    validate_target_raster_limits,
)
from histoforge.cloud.inference.identity import validate_rasterizer_identity
from histoforge.cloud.sources.contract import SOURCE_CONTAINERS

_FIELDS = {
    "modes",
    "output_types",
    "source_containers",
    "runner_build",
    "rasterizer",
    "preprocessing_contract_digest",
    "finite",
}
_FIELDS_WITH_DURABLE = _FIELDS | {"durable_run"}
_FINITE_FIELDS = {
    "max_attempt_execution_seconds",
    "max_request_canonical_bytes",
    "target_raster",
    "max_inclusion_mask_vertices",
    "result",
}
_DURABLE_FIELDS = {"run", "terminal_retention_seconds"}
_MODES = {"FINITE_CALL", "DURABLE_MODEL_RUN"}
_OUTPUT_TYPES = {
    "INSTANCE_SEGMENTATION",
    "INSTANCE_SEGMENTATION_AND_CLASSIFICATION",
    "DENSE_CLASSIFICATION",
    "REGION_CLASSIFICATION",
}


def validate_runner_capabilities(value: Any) -> dict[str, Any]:
    """Validate exact executable behavior and limits advertised by one runner."""

    capabilities = _object(value, "runner capabilities")
    expected = _FIELDS_WITH_DURABLE if "durable_run" in capabilities else _FIELDS
    _fields(capabilities, expected, "runner capabilities")
    modes = _enum_set(capabilities["modes"], _MODES, "modes")
    if "FINITE_CALL" not in modes:
        raise ValueError("runner capabilities must advertise FINITE_CALL")
    output_types = _enum_set(
        capabilities["output_types"], _OUTPUT_TYPES, "output_types"
    )
    source_containers = _enum_set(
        capabilities["source_containers"],
        SOURCE_CONTAINERS,
        "source_containers",
    )
    result = {
        "modes": modes,
        "output_types": output_types,
        "source_containers": source_containers,
        "runner_build": parse_software_build_identity(capabilities["runner_build"]),
        "rasterizer": validate_rasterizer_identity(capabilities["rasterizer"]),
        "preprocessing_contract_digest": require_sha256(
            capabilities["preprocessing_contract_digest"],
            "preprocessing_contract_digest",
        ),
        "finite": validate_finite_call_limits(capabilities["finite"]),
    }
    if "durable_run" in capabilities:
        if "DURABLE_MODEL_RUN" not in modes:
            raise ValueError("durable_run requires DURABLE_MODEL_RUN mode")
        durable = _exact_object(
            capabilities["durable_run"], _DURABLE_FIELDS, "durable_run"
        )
        result["durable_run"] = {
            "run": validate_model_run_limits(durable["run"]),
            "terminal_retention_seconds": require_positive_int32(
                durable["terminal_retention_seconds"],
                "terminal_retention_seconds",
            ),
        }
    elif "DURABLE_MODEL_RUN" in modes:
        raise ValueError("DURABLE_MODEL_RUN mode requires durable_run limits")
    return canonical_tree(result)


def validate_finite_call_limits(value: Any) -> dict[str, Any]:
    """Validate bounded finite-request, raster, mask, timeout, and result caps."""

    limits = _exact_object(value, _FINITE_FIELDS, "finite call limits")
    return canonical_tree(
        {
            "max_attempt_execution_seconds": require_positive_int32(
                limits["max_attempt_execution_seconds"],
                "max_attempt_execution_seconds",
            ),
            "max_request_canonical_bytes": require_positive_safe_integer(
                limits["max_request_canonical_bytes"],
                "max_request_canonical_bytes",
            ),
            "target_raster": validate_target_raster_limits(limits["target_raster"]),
            "max_inclusion_mask_vertices": require_positive_safe_integer(
                limits["max_inclusion_mask_vertices"],
                "max_inclusion_mask_vertices",
            ),
            "result": validate_result_limits(limits["result"]),
        }
    )


def _enum_set(value: Any, allowed: set[str], path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty array")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise ValueError(f"{path} contains an unsupported value")
    if value != sorted(set(value)):
        raise ValueError(f"{path} must be unique and sorted")
    return list(value)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    result = _object(value, path)
    _fields(result, fields, path)
    return result


def _fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
