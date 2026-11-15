"""Assemble one field's targets from complete captured object geometries.

The field and ROI boundaries limit supervision, not instance target generation.
Each intersecting object is rasterized on its complete model grid and its targets
are derived before being copied into the requested field. Only
one object-local grid and the field arrays are resident at a time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from shapely.geometry.base import BaseGeometry

from histoforge._vendor.histoforge_models.pathology.managed_training import (
    HeadType,
    ManagedSegmentationTargets,
    build_managed_targets,
)
from histoforge.cloud.modules.schema import model_output_target_channel_by_class
from histoforge.cloud.rasterization.inputs import validate_raster_transform
from histoforge.cloud.sources.identity import validate_int_rect
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_model_geometry,
    rasterize_training_geometry,
    training_geometry_on_model_grid,
    training_real_pixel_mask,
)


@dataclass(frozen=True)
class TrainingFieldTargets:
    """Unmasked dense targets plus the two independent spatial support masks."""

    targets: ManagedSegmentationTargets
    supervision_mask: np.ndarray
    real_pixel_mask: np.ndarray


def build_training_field_targets(
    *,
    roi_geometry: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    model_output: Mapping[str, Any],
    head_type: HeadType,
    raster_transform: Mapping[str, Any],
    output_rect: Mapping[str, int],
    physical_slide_bounds: Sequence[float],
    flow_device: torch.device | None = None,
) -> TrainingFieldTargets:
    """Produce flow or HoVer supervision for one already-planned model field.

    Inputs are the admitted dataset's complete polygons and exact model grid.
    Class targets use the recipe's explicit channel numbers, including
    training-only channels. This function performs no sampling, image
    preprocessing, augmentation, loss reduction, or file IO.
    """

    if head_type not in {"flow", "hovernet"}:
        raise ValueError("head_type must be 'flow' or 'hovernet'")
    output_type = model_output["type"]
    if output_type not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
    }:
        raise ValueError("flow and HoVer field targets require instance segmentation")
    class_indices = model_output_target_channel_by_class(model_output)
    target = validate_raster_transform(raster_transform)
    field = validate_int_rect(output_rect)
    supervision = rasterize_training_geometry(roi_geometry, target, field)
    real_pixels = training_real_pixel_mask(physical_slide_bounds, target, field)
    effective = supervision & real_pixels
    if not effective.any():
        raise ValueError("training field has no supervised real pixels")

    field_shape = (field["height"], field["width"])
    directional = np.zeros((2, *field_shape), dtype=np.float32)
    foreground = np.zeros(field_shape, dtype=np.float32)
    classes = (
        np.zeros(field_shape, dtype=np.int64)
        if output_type == "instance_segmentation_and_classification"
        else None
    )
    instance_flow_weights = (
        np.zeros(field_shape, dtype=np.float32) if head_type == "flow" else None
    )
    for item in objects:
        geometry = training_geometry_on_model_grid(item["geometry"], target)
        object_rect = _complete_object_rect(geometry)
        origin_x, origin_y = object_rect["x"], object_rect["y"]
        width, height = object_rect["width"], object_rect["height"]
        x0, y0 = max(origin_x, field["x"]), max(origin_y, field["y"])
        x1 = min(origin_x + width, field["x"] + field["width"])
        y1 = min(origin_y + height, field["y"] + field["height"])
        if x0 >= x1 or y0 >= y1:
            continue
        source = (
            slice(y0 - origin_y, y1 - origin_y),
            slice(x0 - origin_x, x1 - origin_x),
        )
        destination = (
            slice(y0 - field["y"], y1 - field["y"]),
            slice(x0 - field["x"], x1 - field["x"]),
        )
        instance = rasterize_model_geometry(geometry, object_rect).astype(np.int32)
        visible = instance[source] != 0
        if not np.any(visible & effective[destination]):
            del instance, visible
            continue
        if np.any(visible & (foreground[destination] != 0) & effective[destination]):
            raise ValueError("training instances share a supervised real pixel")

        complete = build_managed_targets(
            instance, head_type=head_type, flow_device=flow_device
        )
        for axis in range(2):
            directional[axis][destination][visible] = complete.directional[axis][
                source
            ][visible]
        foreground[destination][visible] = 1.0
        if instance_flow_weights is not None:
            assert complete.instance_flow_weights is not None
            instance_flow_weights[destination][visible] = (
                complete.instance_flow_weights[source][visible]
            )
        if classes is not None:
            classes[destination][visible] = class_indices[
                item["properties"]["path_class"]
            ]
        del complete, instance, visible

    return TrainingFieldTargets(
        ManagedSegmentationTargets(
            directional=directional,
            foreground=foreground,
            classes=classes,
            instance_flow_weights=instance_flow_weights,
        ),
        supervision,
        real_pixels,
    )


def _complete_object_rect(geometry: BaseGeometry) -> dict[str, int]:
    """Keep the target's pixel phase while enclosing the entire object.

    A two-pixel background margin supplies the existing HoVer transform's
    bounding-box context. Cellpose computes its own diffusion border. Negative
    local origins are allowed: a captured object may cross the target boundary.
    No image pixels are read for this target-only grid.
    """

    left, top, right, bottom = geometry.bounds
    x0, y0 = math.floor(left) - 2, math.floor(top) - 2
    x1, y1 = math.ceil(right) + 2, math.ceil(bottom) + 2
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
