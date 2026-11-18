"""Source/raster identity and result contracts shared by inference protocols.

Finite calls and durable runs use these exact schemas and feature checks.
This module owns no HTTP envelopes, execution state or artifact lifecycle.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    require_canonical_uuid,
    require_nonnegative_int32,
    require_positive_safe_integer,
    require_sha256,
)
from histoforge.cloud.inference.features import (
    FeatureGeometryKind,
    ResultLimitExceeded,
    validate_feature_collection,
)
from histoforge.cloud.rasterization.inputs import (
    geometry_vertex_count,
    require_geometry_inside_raster,
    validate_model_raster_input,
    validate_raster_descriptor,
)
from histoforge.cloud.sources.capture import (
    validate_captured_source_binding,
    validate_source_binding,
)

_RESULT_BUDGET_FIELDS = {
    "max_transferred_bytes",
    "max_decoded_bytes",
    "max_record_bytes",
    "max_features",
    "max_vertices",
}


_BACKEND_INPUT_FIELDS = {
    "source_artifact_id",
    "source_manifest_digest",
    "source_view_digest",
    "model_raster_input_digest",
}


def validate_client_result_budget(value: Any) -> dict[str, int]:
    """Validate positive client acceptance ceilings for one complete result."""

    budget = _require_exact_object(value, _RESULT_BUDGET_FIELDS, "result budget")
    return {
        field: require_positive_safe_integer(budget[field], field)
        for field in (
            "max_transferred_bytes",
            "max_decoded_bytes",
            "max_record_bytes",
            "max_features",
            "max_vertices",
        )
    }


def validate_backend_input_identity(value: Any) -> dict[str, Any]:
    """Validate one compact source-view and model-raster identity echo."""

    if isinstance(value, dict) and "source_bindings" in value:
        identity = _require_exact_object(
            value,
            {"source_view_digest", "model_raster_input_digest", "source_bindings"},
            "backend input identity",
        )
        return {
            "source_view_digest": require_sha256(
                identity["source_view_digest"], "source_view_digest"
            ),
            "model_raster_input_digest": require_sha256(
                identity["model_raster_input_digest"], "model_raster_input_digest"
            ),
            "source_bindings": validate_input_source_bindings(
                identity["source_bindings"]
            ),
        }
    identity = _require_exact_object(
        value, _BACKEND_INPUT_FIELDS, "backend input identity"
    )
    return {
        "source_artifact_id": require_canonical_uuid(
            identity["source_artifact_id"], "source_artifact_id"
        ),
        "source_manifest_digest": require_sha256(
            identity["source_manifest_digest"], "source_manifest_digest"
        ),
        "source_view_digest": require_sha256(
            identity["source_view_digest"], "source_view_digest"
        ),
        "model_raster_input_digest": require_sha256(
            identity["model_raster_input_digest"], "model_raster_input_digest"
        ),
    }


def build_backend_input_identity(
    raster_input: Any, source_bindings=()
) -> dict[str, Any]:
    """Derive the only compact backend echo for one validated raster input."""

    model_input = validate_model_raster_input(raster_input)
    view = model_input["source_view"]
    if "capture" in view:
        bindings = validate_input_source_bindings(source_bindings)
        for item in bindings:
            validate_captured_source_binding(item["source"], view["capture"])
        return {
            "source_view_digest": view["view_digest"],
            "model_raster_input_digest": model_input["input_digest"],
            "source_bindings": bindings,
        }
    if source_bindings:
        raise ValueError("sealed-file input identity cannot carry source bindings")
    artifact = model_input["source_view"]["artifact"]
    return {
        "source_artifact_id": artifact["artifact_id"],
        "source_manifest_digest": artifact["manifest_digest"],
        "source_view_digest": model_input["source_view"]["view_digest"],
        "model_raster_input_digest": model_input["input_digest"],
    }


def validate_input_source_bindings(value: Any) -> list[dict[str, Any]]:
    """Validate a completed input's ordered, contiguous processing-unit sources."""

    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("completed input source_bindings must be a nonempty array")
    result = []
    for unit_id, entry in enumerate(value):
        item = _require_exact_object(
            entry, {"unit_id", "source"}, "input source binding"
        )
        if require_nonnegative_int32(item["unit_id"], "unit_id") != unit_id:
            raise ValueError(
                "input source bindings must cover consecutive units from zero"
            )
        result.append(
            {"unit_id": unit_id, "source": validate_source_binding(item["source"])}
        )
    return result


def validate_inference_features(
    value: Any,
    raster: Any,
    result_budget: Any,
    declared_classes: Collection[str],
    *,
    geometry_kind: FeatureGeometryKind = "polygonal",
) -> dict[str, Any]:
    """Validate fixed features, raster-local bounds, and decoded result caps."""

    descriptor = validate_raster_descriptor(raster)
    budget = validate_client_result_budget(result_budget)
    geojson = validate_feature_collection(
        value, declared_classes, geometry_kind=geometry_kind
    )
    if len(geojson["features"]) > budget["max_features"]:
        raise ResultLimitExceeded("inference result exceeds max_features")
    vertex_count = 0
    for feature in geojson["features"]:
        require_geometry_inside_raster(
            feature["geometry"], descriptor, "inference feature geometry"
        )
        if len(canonical_bytes(feature)) > budget["max_record_bytes"]:
            raise ResultLimitExceeded("inference feature exceeds max_record_bytes")
        vertex_count += geometry_vertex_count(feature["geometry"])
        if vertex_count > budget["max_vertices"]:
            raise ResultLimitExceeded("inference result exceeds max_vertices")
    return geojson


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
