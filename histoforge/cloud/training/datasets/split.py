"""Exact training and validation assignment contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_int32,
    require_nonnegative_int32,
    require_sha256,
)
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)

_NO_VALIDATION_FIELDS = {"type"}
_SPLIT_FIELDS = {"type", "unit", "assignment_source", "units"}
_GENERATED_SPLIT_FIELDS = _SPLIT_FIELDS | {"requested_validation_percent"}
_ASSIGNMENT_FIELDS = {"unit_index", "partition"}
_SIDECAR_FIELDS = {"schema_version", "dataset_digest", "split_plan"}
_UNITS = {"roi", "image", "project_group"}
_SOURCES = {"generated", "custom"}
_PARTITIONS = {"training", "validation"}


@dataclass(frozen=True)
class ExpandedTrainingSplit:
    """Canonical ROI-index partitions consumed by training and validation."""

    training_roi_indices: tuple[int, ...]
    validation_roi_indices: tuple[int, ...]


def validate_training_split_plan(
    value: Any, manifest: Any, api_limits: Any
) -> dict[str, Any]:
    """Validate one exact final assignment against its immutable dataset."""

    dataset = validate_training_dataset_manifest(manifest, api_limits)
    return _validate_training_split_plan_for_dataset(value, dataset)


def expand_training_split_plan(
    value: Any, manifest: Any, api_limits: Any
) -> ExpandedTrainingSplit:
    """Expand one validated unit assignment into canonical ROI indices."""

    dataset = validate_training_dataset_manifest(manifest, api_limits)
    split = _validate_training_split_plan_for_dataset(value, dataset)
    if split["type"] == "none":
        return ExpandedTrainingSplit(tuple(range(len(dataset["rois"]))), ())

    assignments = tuple(item["partition"] for item in split["units"])
    training = []
    validation = []
    for roi_index, roi in enumerate(dataset["rois"]):
        unit_index = _roi_unit_index(split["unit"], roi_index, roi, dataset)
        target = validation if assignments[unit_index] == "validation" else training
        target.append(roi_index)
    return ExpandedTrainingSplit(tuple(training), tuple(validation))


def _validate_training_split_plan_for_dataset(
    value: Any, dataset: Mapping[str, Any]
) -> dict[str, Any]:
    result = _validate_training_split_plan_syntax(value, require_both_partitions=False)
    if result["type"] == "none":
        return result

    expected_count = _unit_count(result["unit"], dataset)
    if len(result["units"]) != expected_count:
        raise ValueError("split units must assign every unit once in index order")
    if {assignment["partition"] for assignment in result["units"]} != _PARTITIONS:
        raise ValueError("validation split requires nonempty training and validation")
    if result["unit"] == "project_group" and any(
        "project_group_index" not in image for image in dataset["images"]
    ):
        raise ValueError("project-group splitting requires every image to have a group")
    return result


def validate_training_split_plan_syntax(value: Any) -> dict[str, Any]:
    """Validate a persisted split without requiring retained dataset bytes."""

    return _validate_training_split_plan_syntax(value, require_both_partitions=True)


def _validate_training_split_plan_syntax(
    value: Any, *, require_both_partitions: bool
) -> dict[str, Any]:
    item = _object(value, "training split plan")
    kind = item.get("type")
    if kind == "none":
        _require_fields(item, _NO_VALIDATION_FIELDS, "no-validation plan")
        return {"type": "none"}
    if kind != "split":
        raise ValueError(f"unsupported split plan type: {kind!r}")

    source = item.get("assignment_source")
    expected_fields = (
        _GENERATED_SPLIT_FIELDS if source == "generated" else _SPLIT_FIELDS
    )
    _require_fields(item, expected_fields, "validation split plan")
    if source not in _SOURCES:
        raise ValueError("unsupported assignment_source")
    unit = item["unit"]
    if not isinstance(unit, str) or unit not in _UNITS:
        raise ValueError("unsupported split unit")

    assignments = []
    for value_item in _array(item["units"], "units"):
        assignment = _exact_object(
            value_item, _ASSIGNMENT_FIELDS, "split unit assignment"
        )
        index = require_nonnegative_int32(assignment["unit_index"], "unit_index")
        partition = assignment["partition"]
        if not isinstance(partition, str) or partition not in _PARTITIONS:
            raise ValueError("unsupported split partition")
        assignments.append({"unit_index": index, "partition": partition})
    indices = [assignment["unit_index"] for assignment in assignments]
    if indices != list(range(len(assignments))):
        raise ValueError("split units must assign every unit once in index order")
    partitions = {assignment["partition"] for assignment in assignments}
    if require_both_partitions and partitions != _PARTITIONS:
        raise ValueError("validation split requires nonempty training and validation")
    result: dict[str, Any] = {
        "type": "split",
        "unit": unit,
        "assignment_source": source,
        "units": assignments,
    }
    if source == "generated":
        percent = require_int32(
            item["requested_validation_percent"],
            "requested_validation_percent",
        )
        if percent < 1 or percent > 99:
            raise ValueError("requested_validation_percent must be from 1 through 99")
        result["requested_validation_percent"] = percent
    return canonical_tree(result)


def validate_training_split_sidecar(
    value: Any, manifest: Any, api_limits: Any
) -> dict[str, Any]:
    """Validate the local export companion without changing dataset identity."""

    item = _exact_object(value, _SIDECAR_FIELDS, "training split sidecar")
    if require_int32(item["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported training split sidecar schema_version")
    dataset = validate_training_dataset_manifest(manifest, api_limits)
    digest = require_sha256(item["dataset_digest"], "dataset_digest")
    if digest != dataset["dataset_digest"]:
        raise ValueError("split sidecar dataset_digest does not match manifest")
    return canonical_tree(
        {
            "schema_version": 1,
            "dataset_digest": digest,
            "split_plan": validate_training_split_plan(
                item["split_plan"], dataset, api_limits
            ),
        }
    )


def _unit_count(unit: str, manifest: Mapping[str, Any]) -> int:
    if unit == "roi":
        return len(manifest["rois"])
    if unit == "image":
        return len(manifest["images"])
    return len(manifest["project_groups"])


def _roi_unit_index(
    unit: str,
    roi_index: int,
    roi: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> int:
    if unit == "roi":
        return roi_index
    image_index = roi["image_index"]
    if unit == "image":
        return image_index
    return manifest["images"][image_index]["project_group_index"]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    item = _object(value, path)
    _require_fields(item, fields, path)
    return item


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value
