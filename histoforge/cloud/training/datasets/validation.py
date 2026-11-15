"""Bounded full-ROI validation reads, shared by instance and dense tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from shapely.geometry import shape

from histoforge.cloud.execution_limits import (
    fits_target_raster,
    validate_target_raster_limits,
)
from histoforge.cloud.training.datasets.field_reader import read_training_field_inputs
from histoforge.cloud.training.datasets.io import read_training_roi_source
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_training_geometry,
    training_geometry_on_model_grid,
    training_real_pixel_mask,
)
from histoforge.cloud.training.datasets.rasters import (
    training_roi_raster,
    validate_training_raster_profile,
)
from histoforge.cloud.training.datasets.split import expand_training_split_plan

if TYPE_CHECKING:
    import numpy as np

    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset


def plan_validation_rois(
    dataset: RetainedTrainingDataset,
    split_plan: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    api_limits: Mapping[str, Any],
    target_raster_limits: Mapping[str, int],
) -> dict[int, dict[str, Any]]:
    """Validate every ROI grid before prediction; never resize or truncate to fit."""

    raster_limits = validate_target_raster_limits(target_raster_limits)
    indices = expand_training_split_plan(
        split_plan, dataset.manifest, api_limits
    ).validation_roi_indices
    if not indices:
        return {}
    task = dataset.manifest["task"]
    profile = validate_training_raster_profile(
        task,
        capability,
        field_profile,
        api_limits["report"]["max_text_bytes"],
    )
    grids = {}
    for index in indices:
        geometry, origin = read_training_roi_source(
            dataset.root,
            dataset.manifest,
            index,
            api_limits["dataset"]["max_json_member_bytes"],
        )
        grids[index] = training_roi_raster(
            shape(geometry).bounds, origin, task, capability, profile
        )
    if any(
        not fits_target_raster(grid["raster"], raster_limits) for grid in grids.values()
    ):
        raise ValueError("validation ROI exceeds bounded inference limits")
    return grids


def read_validation_roi(
    dataset: RetainedTrainingDataset,
    roi_index: int,
    target: Mapping[str, Any],
    *,
    source_limits: Mapping[str, int],
    max_json_member_bytes: int,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Close source handles before returning raw RGB, annotations and masks."""

    raster = target["raster"]
    rectangle = {"x": 0, "y": 0, "width": raster["width"], "height": raster["height"]}
    pixels, geometry, objects, bounds = read_training_field_inputs(
        dataset,
        roi_index,
        target,
        rectangle,
        source_limits,
        max_json_member_bytes,
    )
    left, top, right, bottom = training_geometry_on_model_grid(geometry, target).bounds
    if left < 0 or top < 0 or right > raster["width"] or bottom > raster["height"]:
        raise ValueError("validation grid does not contain the complete ROI")
    real = training_real_pixel_mask(bounds, target, rectangle)
    support = rasterize_training_geometry(geometry, target, rectangle) & real
    if not support.any():
        raise ValueError("validation ROI has no supervised real pixels")
    return pixels, objects, real, support
