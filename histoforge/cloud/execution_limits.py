"""Shared execution-limit contracts and overflow-safe fit predicates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_positive_int32,
    require_positive_safe_integer,
)

_TARGET_RASTER_LIMIT_FIELDS = {"max_width", "max_height", "max_pixels"}
_RESULT_LIMIT_FIELDS = {
    "max_transferred_bytes",
    "max_decoded_bytes",
    "max_record_bytes",
    "max_features",
    "max_vertices",
}
_MODEL_RUN_LIMIT_FIELDS = {
    "max_execution_seconds",
    "max_request_canonical_bytes",
    "max_inputs",
    "max_units",
    "target_raster_per_input",
    "processing_raster",
    "max_aggregate_target_raster_pixels",
    "max_aggregate_inclusion_mask_vertices",
    "max_recovery_state_bytes_per_unit",
    "max_recovery_state_bytes_per_run",
    "intermediate_result",
    "result",
}


def validate_target_raster_limits(value: Any) -> dict[str, int]:
    """Validate the exact per-raster spatial ceilings advertised by a runner."""

    limits = _require_exact_object(
        value, _TARGET_RASTER_LIMIT_FIELDS, "target raster limits"
    )
    return {
        "max_width": require_positive_int32(limits["max_width"], "max_width"),
        "max_height": require_positive_int32(limits["max_height"], "max_height"),
        "max_pixels": require_positive_safe_integer(limits["max_pixels"], "max_pixels"),
    }


def validate_result_limits(value: Any) -> dict[str, int]:
    """Validate one closed set of transferred, decoded, and biological caps."""

    limits = _require_exact_object(value, _RESULT_LIMIT_FIELDS, "result limits")
    return {
        field: require_positive_safe_integer(limits[field], field)
        for field in (
            "max_transferred_bytes",
            "max_decoded_bytes",
            "max_record_bytes",
            "max_features",
            "max_vertices",
        )
    }


def validate_model_run_limits(value: Any) -> dict[str, Any]:
    """Validate durable-run limits for input, processing, storage and results.

    ``target_raster_per_input`` bounds the complete ROI or slide at model resolution.
    ``processing_raster`` bounds the image read for a single prediction, including
    halo. ``intermediate_result`` bounds aggregate completed-core artifacts before
    assembly; ``result`` bounds the final assembled output. The runner registry
    separately owns the build-pinned candidate limits for one processing image.
    """

    limits = _require_exact_object(value, _MODEL_RUN_LIMIT_FIELDS, "model run limits")
    return canonical_tree(
        {
            "max_execution_seconds": require_positive_int32(
                limits["max_execution_seconds"], "max_execution_seconds"
            ),
            "max_request_canonical_bytes": require_positive_safe_integer(
                limits["max_request_canonical_bytes"],
                "max_request_canonical_bytes",
            ),
            "max_inputs": require_positive_int32(limits["max_inputs"], "max_inputs"),
            "max_units": require_positive_int32(limits["max_units"], "max_units"),
            "target_raster_per_input": validate_target_raster_limits(
                limits["target_raster_per_input"]
            ),
            "processing_raster": validate_target_raster_limits(
                limits["processing_raster"]
            ),
            "max_aggregate_target_raster_pixels": require_positive_safe_integer(
                limits["max_aggregate_target_raster_pixels"],
                "max_aggregate_target_raster_pixels",
            ),
            "max_aggregate_inclusion_mask_vertices": require_positive_safe_integer(
                limits["max_aggregate_inclusion_mask_vertices"],
                "max_aggregate_inclusion_mask_vertices",
            ),
            "max_recovery_state_bytes_per_unit": require_positive_safe_integer(
                limits["max_recovery_state_bytes_per_unit"],
                "max_recovery_state_bytes_per_unit",
            ),
            "max_recovery_state_bytes_per_run": require_positive_safe_integer(
                limits["max_recovery_state_bytes_per_run"],
                "max_recovery_state_bytes_per_run",
            ),
            "intermediate_result": validate_result_limits(
                limits["intermediate_result"]
            ),
            "result": validate_result_limits(limits["result"]),
        }
    )


def fits_target_raster(raster: Mapping[str, Any], limits: Any) -> bool:
    """Return the frozen finite-plan selection predicate for one raster."""

    normalized = validate_target_raster_limits(limits)
    width = require_positive_int32(raster.get("width"), "raster.width")
    height = require_positive_int32(raster.get("height"), "raster.height")
    return (
        width <= normalized["max_width"]
        and height <= normalized["max_height"]
        and width * height <= normalized["max_pixels"]
    )


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return value
