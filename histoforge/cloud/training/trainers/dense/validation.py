"""Full-ROI semantic validation with pooled neural-channel confusion counts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from histoforge._vendor.histoforge_models.pathology.metric_aggregation import (
    class_f1,
    mean_defined,
)
from histoforge.cloud.modules.schema import (
    model_output_class_channels,
    model_output_target_channel_by_class,
)
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.training.datasets.class_raster import rasterize_class_targets
from histoforge.cloud.training.datasets.label_rasterizer import (
    training_geometry_on_model_grid,
)
from histoforge.cloud.training.datasets.validation import (
    plan_validation_rois,
    read_validation_roi,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.trainers.dense.policy import (
    VALIDATION_PIXEL_MACRO_F1_SERIES_ID,
    pixel_class_f1_series_id,
)

if TYPE_CHECKING:
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset


def evaluate_dense_validation(
    dataset: RetainedTrainingDataset,
    split_plan: Mapping[str, Any],
    predict: Callable[[RasterizedImage], Mapping[str, Any]],
    *,
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, int],
    target_raster_limits: Mapping[str, int],
) -> dict[str, float]:
    """Infer each held-out ROI once, using full-head argmax before role filtering."""

    limits = validate_training_api_limits(api_limits)
    output = dataset.manifest["task"]["model_output"]
    if output["type"] != "dense_classification":
        raise ValueError("dense validation requires a dense-classification task")
    grids = plan_validation_rois(
        dataset,
        split_plan,
        capability=capability,
        field_profile=field_profile,
        api_limits=limits,
        target_raster_limits=target_raster_limits,
    )
    if not grids:
        return {}
    channels = model_output_class_channels(output)
    by_name = model_output_target_channel_by_class(output)
    confusion = np.zeros((len(channels), len(channels)), dtype=np.int64)
    for index, target in grids.items():
        pixels, objects, real, support = read_validation_roi(
            dataset,
            index,
            target,
            source_limits=source_limits,
            max_json_member_bytes=limits["dataset"]["max_json_member_bytes"],
        )
        rectangle = {
            "x": 0,
            "y": 0,
            "width": support.shape[1],
            "height": support.shape[0],
        }
        truth = rasterize_class_targets(
            (
                (
                    training_geometry_on_model_grid(item["geometry"], target),
                    by_name[item["properties"]["path_class"]],
                )
                for item in objects
            ),
            rectangle,
            support,
            allow_same_class_overlap=True,
        )
        prediction = np.asarray(predict(RasterizedImage(pixels, real))["class_map"])
        confusion += pixel_confusion(truth, prediction, support, len(channels))
    return dense_metric_values(confusion, channels)


def pixel_confusion(
    truth: np.ndarray, prediction: np.ndarray, support: np.ndarray, class_count: int
) -> np.ndarray:
    """Rows are truth, columns prediction; retain background and training-only roles."""

    if prediction.shape != support.shape or truth.shape != support.shape:
        raise ValueError("prediction must retain the exact validation raster shape")
    for array in (truth, prediction):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("dense labels must be integer neural channel indices")
        if np.any((array[support] < 0) | (array[support] >= class_count)):
            raise ValueError("dense label is outside the complete model head")
    pairs = truth[support].astype(np.int64) * class_count + prediction[support]
    return np.bincount(pairs, minlength=class_count**2).reshape(
        class_count, class_count
    )


def dense_metric_values(
    confusion: np.ndarray, channels: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    """Omit undefined F1; prediction-only and target-only classes receive zero."""

    tp = np.diag(confusion)
    values = class_f1(tp, confusion.sum(axis=0) - tp, confusion.sum(axis=1) - tp)
    emitted = [item["channel"] for item in channels if item["role"] == "emitted"]
    result = {
        pixel_class_f1_series_id(channel): float(values[channel])
        for channel in emitted
        if np.isfinite(values[channel])
    }
    macro = mean_defined(values[emitted])
    if np.isfinite(macro):
        result[VALIDATION_PIXEL_MACRO_F1_SERIES_ID] = float(macro)
    return result
