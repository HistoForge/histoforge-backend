"""Training ROI framing under the shared target-pixel grid contract.

Dataset admission reserves the maximum permitted real context. Execution uses
only the accepted profile's context. These pure calculations never choose a
source level, resize an ROI, or add model/alignment padding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from histoforge.cloud.rasterization.inputs import validate_raster_transform
from histoforge.cloud.sources.views import source_slide_bounds
from histoforge.cloud.training.contract import (
    training_task_from_capability,
    validate_training_field_profile_for_capability,
)


def validate_training_raster_profile(
    task: Mapping[str, Any],
    capability: Mapping[str, Any],
    profile: Mapping[str, Any],
    max_text_bytes: int,
) -> dict[str, Any]:
    """Bind a run's accepted field profile to the retained dataset's exact task."""

    if task != training_task_from_capability(
        capability,
        task["model_output"],
        task["model_microns_per_pixel"],
        max_text_bytes,
        annotation_class_map=task["annotation_class_map"],
    ):
        raise ValueError("dataset task does not match its training capability")
    return validate_training_field_profile_for_capability(
        profile, capability, max_text_bytes
    )


def training_roi_raster(
    roi_bounds: tuple[float, float, float, float],
    origin: Mapping[str, Any],
    task: Mapping[str, Any],
    capability: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Frame one complete ROI using its already-validated accepted context rule."""

    fixed = profile["planner"] == "fixed-field-v1"
    frame = _context_frame(
        roi_bounds,
        origin,
        task,
        capability,
        profile["side_model_px"] if fixed else profile["minSideModelPx"],
        profile["small_roi_context"] if fixed else profile["smallRoiContext"],
    )
    return training_raster_transform(frame, task, origin["source"])


def maximum_training_source_frame(
    roi_bounds: tuple[float, float, float, float],
    origin: Mapping[str, Any],
    task: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    """Bound every real-context choice that this captured capability permits."""

    profile = capability["default_field_profile"]
    configured = (
        profile["small_roi_context"]
        if profile["planner"] == "fixed-field-v1"
        else profile["smallRoiContext"]
    )
    return _context_frame(
        roi_bounds,
        origin,
        task,
        capability,
        capability["allowed_field_side_model_px"]["maximum"],
        configured or capability["small_roi_context_editable"],
    )


def _context_frame(roi_bounds, origin, task, capability, minimum_side, context):
    """Expand in physical units, then clip without shifting along an image edge."""

    calibration = origin["source"]["calibration"]
    mpp = task["model_microns_per_pixel"]
    if task["input_kind"] == "crop":
        # Preserve the inference operation order: model pixels -> microns ->
        # source-axis pixels. Crop framing is the sole real-context owner.
        padding_um = capability["crop_padding_model_px"] * mpp
        padding_x = padding_um / calibration["mpp_x"]
        padding_y = padding_um / calibration["mpp_y"]
    elif context:
        minimum_width = minimum_side * mpp / calibration["mpp_x"]
        minimum_height = minimum_side * mpp / calibration["mpp_y"]
        padding_x = max(0.0, (minimum_width - (roi_bounds[2] - roi_bounds[0])) / 2.0)
        padding_y = max(0.0, (minimum_height - (roi_bounds[3] - roi_bounds[1])) / 2.0)
    else:
        return roi_bounds
    bounds = source_slide_bounds(origin)
    frame = (
        max(bounds[0], roi_bounds[0] - padding_x),
        max(bounds[1], roi_bounds[1] - padding_y),
        min(bounds[2], roi_bounds[2] + padding_x),
        min(bounds[3], roi_bounds[3] + padding_y),
    )
    if not (frame[2] > frame[0] and frame[3] > frame[1]):
        raise ValueError("model input frame is empty after source clipping")
    return frame


def training_raster_transform(
    frame: tuple[float, float, float, float],
    task: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep exact target MPP and origin; round only the raster's pixel counts."""

    step_x = task["model_microns_per_pixel"] / source["calibration"]["mpp_x"]
    step_y = task["model_microns_per_pixel"] / source["calibration"]["mpp_y"]
    return validate_raster_transform(
        {
            "raster": {
                "width": max(1, math.ceil((frame[2] - frame[0]) / step_x)),
                "height": max(1, math.ceil((frame[3] - frame[1]) / step_y)),
                "plane": {"z": 0, "t": 0},
                "target_mpp": task["model_microns_per_pixel"],
                "resampling": "BILINEAR",
                "pixel_format": "SRGB_UINT8",
            },
            "raster_to_slide": {
                "scale_x": step_x,
                "scale_y": step_y,
                "translate_x": frame[0],
                "translate_y": frame[1],
            },
        }
    )
