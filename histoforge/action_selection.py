"""Canonical user-action and target-selection contracts.

This structural codec owns exact variant shapes, set normalization, and the
single action-level region boundary.  Recipe-aware validation remains with the
planner: only a captured module can decide whether a step exists, whether its
input is field or crop, and whether selected steps are in recipe order.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
)
from histoforge.cloud.modules.schema import validate_model_id
from histoforge.execution_scope import validate_region_key

_ROI_REGIONS_FIELDS = {"type", "roi_ids"}
_SLIDE_REGION_FIELDS = {"type", "region"}
_FULL_SLIDE_FIELDS = {"type"}
_ANNOTATION_UNION_FIELDS = {"type", "annotation_ids"}
_ROI_TARGET_FIELDS = {"type", "roi_id"}
_PARENT_TARGET_FIELDS = {"type", "region", "parent_id"}
_SLIDE_TARGET_FIELDS = {"type"}
_TARGET_SELECTION_FIELDS = {"selectors"}
_STEP_TARGETS_FIELDS = {"step_id", "targets"}
_ALL_MODELS_FIELDS = {"type", "regions"}
_SELECTED_MODELS_FIELDS = {"type", "regions", "steps"}
_MODEL_ACTION_FIELDS = {
    "type",
    "work",
    "remove_overlapping_same_class_duplicates",
}
_METRICS_ACTION_FIELDS = {"type", "targets"}


def validate_region_selection(value: Any) -> dict[str, Any]:
    """Validate one nonempty ROI set or one captured slide-region selector."""

    selection = _require_object(value, "region selection")
    selection_type = selection.get("type")
    if selection_type == "roi-regions":
        _require_fields(selection, _ROI_REGIONS_FIELDS, "ROI region selection")
        return {
            "type": "roi-regions",
            "roi_ids": _canonical_uuid_set(selection["roi_ids"], "roi_ids"),
        }
    if selection_type == "slide-region":
        _require_fields(selection, _SLIDE_REGION_FIELDS, "slide region selection")
        return canonical_tree(
            {
                "type": "slide-region",
                "region": validate_slide_region_selection(selection["region"]),
            }
        )
    raise ValueError(f"unsupported region selection type: {selection_type!r}")


def validate_slide_region_selection(value: Any) -> dict[str, Any]:
    """Validate the complete slide rectangle or one annotation-union selector."""

    selection = _require_object(value, "slide region selector")
    selection_type = selection.get("type")
    if selection_type == "full-slide":
        _require_fields(selection, _FULL_SLIDE_FIELDS, "full-slide selector")
        return {"type": "full-slide"}
    if selection_type == "annotation-union":
        _require_fields(
            selection, _ANNOTATION_UNION_FIELDS, "annotation-union selector"
        )
        return {
            "type": "annotation-union",
            "annotation_ids": _canonical_uuid_set(
                selection["annotation_ids"], "annotation_ids"
            ),
        }
    raise ValueError(f"unsupported slide region selector type: {selection_type!r}")


def validate_target_selector(value: Any) -> dict[str, Any]:
    """Validate one ROI, exact parent, or enclosing slide target marker."""

    selector = _require_object(value, "target selector")
    selector_type = selector.get("type")
    if selector_type == "roi-target":
        _require_fields(selector, _ROI_TARGET_FIELDS, "ROI target")
        return {
            "type": "roi-target",
            "roi_id": require_canonical_uuid(selector["roi_id"], "roi_id"),
        }
    if selector_type == "parent-target":
        _require_fields(selector, _PARENT_TARGET_FIELDS, "parent target")
        return canonical_tree(
            {
                "type": "parent-target",
                "region": validate_region_key(selector["region"]),
                "parent_id": require_canonical_uuid(selector["parent_id"], "parent_id"),
            }
        )
    if selector_type == "slide-target":
        _require_fields(selector, _SLIDE_TARGET_FIELDS, "slide target")
        return {"type": "slide-target"}
    raise ValueError(f"unsupported target selector type: {selector_type!r}")


def validate_target_selection(value: Any) -> dict[str, Any]:
    """Validate and canonically order one nonempty duplicate-free selector set."""

    selection = _require_exact_object(
        value, _TARGET_SELECTION_FIELDS, "target selection"
    )
    raw = _require_nonempty_list(selection["selectors"], "selectors")
    selectors = [validate_target_selector(item) for item in raw]
    encoded = [canonical_bytes(item) for item in selectors]
    if len(set(encoded)) != len(encoded):
        raise ValueError("selectors must not contain duplicates")
    selectors.sort(key=canonical_bytes)
    return {"selectors": selectors}


def validate_model_work_selection(value: Any) -> dict[str, Any]:
    """Validate all-model or explicit-step work under one region owner."""

    work = _require_object(value, "model work selection")
    work_type = work.get("type")
    if work_type == "all-models":
        _require_fields(work, _ALL_MODELS_FIELDS, "all-models selection")
        return canonical_tree(
            {
                "type": "all-models",
                "regions": validate_region_selection(work["regions"]),
            }
        )
    if work_type == "selected-models":
        _require_fields(work, _SELECTED_MODELS_FIELDS, "selected-models selection")
        regions = validate_region_selection(work["regions"])
        raw_steps = _require_nonempty_list(work["steps"], "steps")
        steps = [_validate_step_targets(item, regions) for item in raw_steps]
        step_ids = [step["step_id"] for step in steps]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("steps must not contain duplicate step_id values")
        return canonical_tree(
            {"type": "selected-models", "regions": regions, "steps": steps}
        )
    raise ValueError(f"unsupported model work selection type: {work_type!r}")


def validate_image_action(value: Any) -> dict[str, Any]:
    """Validate one explicit Fill, Redo, or Metrics user action."""

    action = _require_object(value, "image action")
    action_type = action.get("type")
    if action_type in {"fill", "redo"}:
        _require_fields(action, _MODEL_ACTION_FIELDS, f"{action_type} action")
        duplicate_setting = action["remove_overlapping_same_class_duplicates"]
        if not isinstance(duplicate_setting, bool):
            raise ValueError(
                "remove_overlapping_same_class_duplicates must be a boolean"
            )
        return canonical_tree(
            {
                "type": action_type,
                "work": validate_model_work_selection(action["work"]),
                "remove_overlapping_same_class_duplicates": duplicate_setting,
            }
        )
    if action_type == "metrics":
        _require_fields(action, _METRICS_ACTION_FIELDS, "metrics action")
        return canonical_tree(
            {"type": "metrics", "targets": validate_region_selection(action["targets"])}
        )
    raise ValueError(f"unsupported image action type: {action_type!r}")


def _validate_step_targets(value: Any, regions: Mapping[str, Any]) -> dict[str, Any]:
    step = _require_exact_object(value, _STEP_TARGETS_FIELDS, "step targets")
    targets = validate_target_selection(step["targets"])
    _require_targets_within_regions(targets["selectors"], regions)
    return canonical_tree(
        {
            "step_id": validate_model_id(step["step_id"], "step_id"),
            "targets": targets,
        }
    )


def _require_targets_within_regions(
    selectors: list[dict[str, Any]], regions: Mapping[str, Any]
) -> None:
    if regions["type"] == "slide-region":
        if len(selectors) != 1 or selectors[0]["type"] != "slide-target":
            raise ValueError(
                "slide-region selected steps require exactly one slide-target"
            )
        return

    allowed = set(regions["roi_ids"])
    for selector in selectors:
        selector_type = selector["type"]
        if selector_type == "slide-target":
            raise ValueError("ROI-region selected steps cannot contain slide-target")
        if selector_type == "roi-target":
            roi_id = selector["roi_id"]
        else:
            region = selector["region"]
            if region["type"] != "roi":
                raise ValueError("ROI-region parent-target must name an ROI region")
            roi_id = region["roi_id"]
        if roi_id not in allowed:
            raise ValueError(
                "step target lies outside the action-level region selection"
            )


def _canonical_uuid_set(value: Any, path: str) -> list[str]:
    raw = _require_nonempty_list(value, path)
    normalized = [require_canonical_uuid(item, path) for item in raw]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{path} must not contain duplicates")
    return sorted(normalized)


def _require_nonempty_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a nonempty array")
    return value


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
