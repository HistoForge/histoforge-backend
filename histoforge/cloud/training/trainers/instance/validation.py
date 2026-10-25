"""Evaluate complete bounded validation ROIs through normal model inference.

The caller owns the accepted field profile, predictor, and resource profile.
This adapter owns dataset reads and conversion of captured objects and decoded
instances to the existing metric inputs. It chooses no fields, padding, model
parameters, metric series, checkpoints, or report revisions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from histoforge.cloud.modules.schema import (
    model_output_class_channels,
    model_output_target_channel_by_class,
)
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_model_geometry,
    training_geometry_on_model_grid,
)
from histoforge.cloud.training.datasets.validation import (
    plan_validation_rois,
    read_validation_roi,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.trainers.instance.metrics import (
    InstanceMetricResult,
    InstanceMetricRoi,
    evaluate_instance_metrics,
)


def evaluate_bounded_training_validation(
    dataset: RetainedTrainingDataset,
    split_plan: Mapping[str, Any],
    predict: Callable[[RasterizedImage], Mapping[str, Any]],
    *,
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, int],
    target_raster_limits: Mapping[str, int],
    max_assignment_matrix_entries: int,
) -> InstanceMetricResult | None:
    """Score every held-out ROI once, or return ``None`` for no validation.

    Each grid derives from the retained ROI, source calibration, and accepted
    field profile, including only permitted real context. ``predict`` is the configured
    model's ordinary decoded-mask inference, returning ``instance_mask`` and,
    for classified output, ``votes_per_instance`` in network-channel indices.
    It receives raw RGB and real-pixel support, not pre-normalized ROI pixels.
    The caller owns the model's evaluation mode and restoration of training mode.

    The worker retains the dataset's active-run reference. Each image is read
    and closed before prediction; no Volume gate spans model or metric work.
    All raster plans must fit the supplied bounded-inference limits. An oversized
    ROI needs the separate large-region path, never a resized or truncated score.
    A read, prediction, or metric failure propagates without a partial result.
    """

    limits = validate_training_api_limits(api_limits)
    grids = plan_validation_rois(
        dataset,
        split_plan,
        capability=capability,
        field_profile=field_profile,
        api_limits=limits,
        target_raster_limits=target_raster_limits,
    )
    if not grids:
        return None
    task = dataset.manifest["task"]
    model_output = task["model_output"]
    output_type = model_output["type"]
    if output_type not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
    }:
        raise ValueError("instance validation requires an instance-output task")
    classified = output_type == "instance_segmentation_and_classification"
    channels = model_output_class_channels(model_output)
    valid_channels = frozenset(item["channel"] for item in channels)
    emitted_channel_to_index = {
        item["channel"]: index
        for index, item in enumerate(
            (item for item in channels if item["role"] == "emitted")
        )
    }
    rois = (
        _predict_metric_roi(
            dataset,
            index,
            grids[index],
            predict,
            classified=classified,
            class_by_name=(
                model_output_target_channel_by_class(model_output)
                if classified
                else None
            ),
            valid_class_channels=valid_channels,
            source_limits=source_limits,
            max_json_member_bytes=limits["dataset"]["max_json_member_bytes"],
        )
        for index in grids
    )
    return evaluate_instance_metrics(
        rois,
        max_assignment_matrix_entries=max_assignment_matrix_entries,
        emitted_channel_to_index=(emitted_channel_to_index if classified else None),
        valid_class_channels=(valid_channels if classified else None),
    )


def _predict_metric_roi(
    dataset: RetainedTrainingDataset,
    roi_index: int,
    target: Mapping[str, Any],
    predict: Callable[[RasterizedImage], Mapping[str, Any]],
    *,
    classified: bool,
    class_by_name: Mapping[str, int] | None,
    valid_class_channels: frozenset[int],
    source_limits: Mapping[str, int],
    max_json_member_bytes: int,
) -> InstanceMetricRoi:
    """Read one complete ROI, predict it, and retain only metric-owned arrays."""

    pixels, objects, real, support = read_validation_roi(
        dataset,
        roi_index,
        target,
        source_limits=source_limits,
        max_json_member_bytes=max_json_member_bytes,
    )
    prediction = predict(RasterizedImage(pixels, real))
    mask = np.asarray(prediction["instance_mask"])
    if mask.shape != support.shape:
        raise ValueError("prediction must retain the exact validation raster shape")
    ground_truth, classes = _ground_truth_instances(
        objects, class_by_name, target, support
    )
    return InstanceMetricRoi(
        ground_truth,
        mask,
        support,
        ground_truth_classes=classes if classified else None,
        prediction_classes=(
            _prediction_classes(prediction["votes_per_instance"], valid_class_channels)
            if classified
            else None
        ),
    )


def _ground_truth_instances(
    objects: Sequence[Mapping[str, Any]],
    class_by_name: Mapping[str, int] | None,
    target: Mapping[str, Any],
    support: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Rasterize complete polygons on one fixed grid without clipping to the ROI.

    Package order supplies stable positive IDs. One complete captured object
    keeps one ID even if the later support mask splits it into fragments.
    Object-local windows bound temporary masks; their origins are integer offsets
    in the original target grid, never newly computed slide-space transforms.
    """

    height, width = support.shape
    labels = np.zeros((height, width), dtype=np.int64)
    classes = {}
    for instance_id, item in enumerate(objects, start=1):
        if class_by_name is not None:
            classes[instance_id] = class_by_name[item["properties"]["path_class"]]
        geometry = training_geometry_on_model_grid(item["geometry"], target)
        left, top, right, bottom = geometry.bounds
        x0, y0 = max(0, math.floor(left) - 1), max(0, math.floor(top) - 1)
        x1, y1 = min(width, math.ceil(right) + 1), min(height, math.ceil(bottom) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        rectangle = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
        instance = rasterize_model_geometry(geometry, rectangle)
        destination = labels[y0:y1, x0:x1]
        if np.any(instance & (destination != 0) & support[y0:y1, x0:x1]):
            raise ValueError("validation instances share a supervised real pixel")
        destination[instance] = instance_id
    return labels, classes


def _prediction_classes(
    votes: Mapping[int, int], valid_channels: frozenset[int]
) -> dict[int, int]:
    """Validate ordinary decoder votes while preserving neural channel IDs."""

    result = {}
    for instance_id, channel in votes.items():
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or channel not in valid_channels
        ):
            raise ValueError("prediction has an invalid instance class channel")
        result[instance_id] = channel
    return result
