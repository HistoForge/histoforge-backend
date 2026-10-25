"""TRAIN-only pixel populations for the frozen classification objective.

Counting uses the accepted ROI grid, not overlapping execution fields. It reads
only canonical geometry and source metadata; RGB and instance-dependent dense
targets have no role in these class populations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from histoforge.canonical_json import (
    require_finite_number,
    require_nonnegative_safe_integer,
)
from histoforge.cloud.modules.schema import (
    model_output_class_channels,
    model_output_target_channel_by_class,
)
from histoforge.cloud.sources.views import source_slide_bounds
from histoforge.cloud.training.datasets.class_raster import rasterize_class_targets
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.datasets.io import read_training_roi_labels
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_model_geometry,
    training_geometry_on_model_grid,
    training_real_pixel_mask,
)
from histoforge.cloud.training.datasets.rasters import training_roi_raster

# Bounds temporary label arrays, independently of the model's field profile.
_COUNT_WINDOW_SIDE_PX = 1024


def count_training_class_pixels(
    dataset: RetainedTrainingDataset,
    training_roi_indices: Sequence[int],
    *,
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    max_json_member_bytes: int,
) -> tuple[int, ...]:
    """Count each supervised real pixel once per TRAIN ROI.

    The caller supplies a verified retained dataset, its expanded TRAIN indices,
    and the already-validated capability and accepted field profile. Channel zero
    is background. Every target uses the explicit channel declared by the model
    output, including training-only classes. Separate ROIs retain their own pixel
    occurrences.
    """

    task = dataset.manifest["task"]
    model_output = task["model_output"]
    class_indices = model_output_target_channel_by_class(model_output)
    counts = [0] * len(model_output_class_channels(model_output))
    for roi_index in training_roi_indices:
        roi, objects, origin = read_training_roi_labels(
            dataset.root, dataset.manifest, roi_index, max_json_member_bytes
        )
        # All file handles and the mounted Volume gate are released before
        # mapping or rasterizing a potentially large ROI.
        target = training_roi_raster(
            shape(roi).bounds, origin, task, capability, field_profile
        )
        roi_on_grid = training_geometry_on_model_grid(roi, target)
        geometries = [
            training_geometry_on_model_grid(item["geometry"], target)
            for item in objects
        ]
        object_classes = [
            class_indices[item["properties"]["path_class"]] for item in objects
        ]
        tree = STRtree(geometries)
        bounds = source_slide_bounds(origin)
        raster = target["raster"]
        has_support = False
        for y in range(0, raster["height"], _COUNT_WINDOW_SIDE_PX):
            for x in range(0, raster["width"], _COUNT_WINDOW_SIDE_PX):
                rectangle = {
                    "x": x,
                    "y": y,
                    "width": min(_COUNT_WINDOW_SIDE_PX, raster["width"] - x),
                    "height": min(_COUNT_WINDOW_SIDE_PX, raster["height"] - y),
                }
                window = box(x, y, x + rectangle["width"], y + rectangle["height"])
                if not roi_on_grid.intersects(window):
                    continue
                support = rasterize_model_geometry(roi_on_grid, rectangle)
                support &= training_real_pixel_mask(bounds, target, rectangle)
                if not support.any():
                    continue
                has_support = True
                labels = rasterize_class_targets(
                    (
                        (geometries[index], object_classes[index])
                        for index in tree.query(window, predicate="intersects")
                    ),
                    rectangle,
                    support,
                    allow_same_class_overlap=model_output["type"]
                    == "dense_classification",
                )
                window_counts = np.bincount(labels[support], minlength=len(counts))
                for index, count in enumerate(window_counts):
                    counts[index] += int(count)
        if not has_support:
            raise ValueError("training ROI has no supervised real pixels")
    return tuple(counts)


def smoothed_class_weights(raw_counts: Sequence[int]) -> list[float]:
    """Resolve sqrt(median(max(count, 1)) / max(count, 1)) as float32 class weights."""

    floored = np.maximum(np.asarray(raw_counts, dtype=np.float64), 1)
    return np.sqrt(np.median(floored) / floored).astype(np.float32).tolist()


def validated_smoothed_class_weights(
    weighting: Mapping[str, Any],
    model_output: Mapping[str, Any],
    class_count: int,
) -> list[float]:
    """Validate channel-bound count evidence and return tensor-order class weights."""

    channels = model_output_class_channels(model_output)
    if len(channels) != class_count:
        raise ValueError(
            "classification objective does not match the complete model head"
        )
    records = weighting.get("channels")
    if not isinstance(records, list) or len(records) != class_count:
        raise ValueError("class weighting does not cover the complete model head")
    by_channel: dict[int, tuple[int, float]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise ValueError("class weighting channel must be an object")
        channel = require_nonnegative_safe_integer(value.get("channel"), "channel")
        if channel >= class_count or channel in by_channel:
            raise ValueError("class weighting channels must cover the model head once")
        expected = channels[channel]
        fields = {"channel", "role", "raw_count", "weight"}
        if expected["role"] != "background":
            fields.add("path_class")
        if set(value) != fields:
            raise ValueError("class weighting channel fields do not match model output")
        if value["channel"] != expected["channel"] or value["role"] != expected["role"]:
            raise ValueError(
                "class weighting channel identity differs from model output"
            )
        if (
            expected["role"] != "background"
            and value["path_class"] != expected["pathClass"]
        ):
            raise ValueError("class weighting PathClass differs from model output")
        by_channel[channel] = (
            require_nonnegative_safe_integer(value["raw_count"], "raw_count"),
            require_finite_number(value["weight"], "weight"),
        )
    if set(by_channel) != set(range(class_count)):
        raise ValueError("class weighting channels must cover the model head once")
    raw_counts = [by_channel[channel][0] for channel in range(class_count)]
    weights = [by_channel[channel][1] for channel in range(class_count)]
    if weights != smoothed_class_weights(raw_counts):
        raise ValueError("class weighting values do not match their recorded counts")
    return weights
