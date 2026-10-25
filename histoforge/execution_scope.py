"""Canonical logical region, execution-scope, and output ownership contracts.

These identities are shared by action selection, raster inputs, model runs,
provenance, Fill presence, and Redo replacement.  They deliberately contain no
geometry, crop, source-artifact, raster, processing-window, or model-tile data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import canonical_tree, require_canonical_uuid
from histoforge.cloud.inference.identity import validate_model_step_identity

_ROI_REGION_FIELDS = {"type", "roi_id"}
_SLIDE_REGION_FIELDS = {"type"}
_FIELD_SCOPE_FIELDS = {"type", "region"}
_PARENT_SCOPE_FIELDS = {"type", "region", "parent_id"}
_OUTPUT_KEY_FIELDS = {"producer", "scope"}


def validate_region_key(value: Any) -> dict[str, str]:
    """Validate one stable logical ROI or slide-region identity."""

    region = _require_object(value, "region key")
    region_type = region.get("type")
    if region_type == "roi":
        _require_fields(region, _ROI_REGION_FIELDS, "ROI region key")
        return {
            "type": "roi",
            "roi_id": require_canonical_uuid(region["roi_id"], "roi_id"),
        }
    if region_type == "slide":
        _require_fields(region, _SLIDE_REGION_FIELDS, "slide region key")
        return {"type": "slide"}
    raise ValueError(f"unsupported region key type: {region_type!r}")


def validate_execution_scope(value: Any) -> dict[str, Any]:
    """Validate one logical model-output scope independent of pixel framing."""

    scope = _require_object(value, "execution scope")
    scope_type = scope.get("type")
    if scope_type == "field":
        _require_fields(scope, _FIELD_SCOPE_FIELDS, "field scope")
        return canonical_tree(
            {"type": "field", "region": validate_region_key(scope["region"])}
        )
    if scope_type == "parent":
        _require_fields(scope, _PARENT_SCOPE_FIELDS, "parent scope")
        return canonical_tree(
            {
                "type": "parent",
                "region": validate_region_key(scope["region"]),
                "parent_id": require_canonical_uuid(scope["parent_id"], "parent_id"),
            }
        )
    raise ValueError(f"unsupported execution scope type: {scope_type!r}")


def validate_output_key(value: Any) -> dict[str, Any]:
    """Validate the sole logical key for Fill presence and Redo replacement."""

    output = _require_exact_object(value, _OUTPUT_KEY_FIELDS, "output key")
    return canonical_tree(
        {
            "producer": validate_model_step_identity(output["producer"]),
            "scope": validate_execution_scope(output["scope"]),
        }
    )


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    result = _require_object(value, path)
    _require_fields(result, fields, path)
    return result


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
