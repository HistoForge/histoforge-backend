"""Exact instance metrics used by managed-training validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from histoforge._vendor.histoforge_models.pathology.metric_aggregation import (
    class_f1,
    mean_defined,
    segmentation_accuracy,
)

_MSA_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


@dataclass(frozen=True)
class InstanceMetricRoi:
    """One ROI's decoded instances, ground truth, and supervised pixel support.

    Class mappings use the model's explicit neural channel numbers. They are
    absent for unclassified instance segmentation. Extra mapping entries are
    permitted because an instance can have no pixels inside metric support.
    """

    ground_truth: NDArray[np.integer]
    prediction: NDArray[np.integer]
    support: NDArray[np.bool_]
    ground_truth_classes: Mapping[int, int] | None = None
    prediction_classes: Mapping[int, int] | None = None


@dataclass(frozen=True)
class InstanceMetricResult:
    """Run-level validation values; ``None`` means mathematically undefined."""

    ap50: float | None
    msa: float | None
    object_macro_f1: float | None
    object_class_f1: Mapping[int, float | None]


@dataclass(frozen=True)
class _IoUComponent:
    """One independent connected component of the instance-overlap graph."""

    ground_truth_indices: NDArray[np.int64]
    prediction_indices: NDArray[np.int64]
    values: NDArray[np.float64]


@dataclass(frozen=True)
class _PairwiseIoU:
    """Sparse component decomposition with global instance counts."""

    ground_truth_count: int
    prediction_count: int
    components: tuple[_IoUComponent, ...]


def evaluate_instance_metrics(
    rois: Iterable[InstanceMetricRoi],
    *,
    max_assignment_matrix_entries: int,
    emitted_channel_to_index: Mapping[int, int] | None = None,
    valid_class_channels: frozenset[int] | None = None,
) -> InstanceMetricResult:
    """Evaluate AP50, mSA, and optional object macro F1 across validation ROIs.

    AP50 and mSA pool counts across ROIs separately for every IoU threshold.
    Classification counts are pooled across ROIs before per-class F1 values
    are macro-averaged. Empty-empty populations are undefined, not zero.
    Every match uses Cellpose's threshold-first linear
    assignment cost and the exact supervised support supplied for that ROI.
    ROIs may be produced lazily; ROI rasters and IoU matrices are released before
    requesting the next ROI. No validation-set-wide raster stack is needed.
    """

    if (
        isinstance(max_assignment_matrix_entries, bool)
        or not isinstance(max_assignment_matrix_entries, int)
        or max_assignment_matrix_entries <= 0
    ):
        raise ValueError("max_assignment_matrix_entries must be a positive integer")
    classified = (
        emitted_channel_to_index is not None or valid_class_channels is not None
    )
    if classified:
        if not emitted_channel_to_index or not valid_class_channels:
            raise ValueError(
                "classified metrics require emitted and valid class-channel mappings"
            )
        if not set(emitted_channel_to_index) <= valid_class_channels:
            raise ValueError("emitted metric channels must be valid model channels")
        emitted_indices = sorted(emitted_channel_to_index.values())
        if emitted_indices != list(range(len(emitted_indices))):
            raise ValueError("emitted metric indices must be contiguous from zero")
        emitted_class_count = len(emitted_indices)

    instance_counts = np.zeros((3, len(_MSA_THRESHOLDS)), dtype=np.int64)
    true_positive = (
        np.zeros(emitted_class_count, dtype=np.int64) if classified else None
    )
    false_positive = (
        np.zeros(emitted_class_count, dtype=np.int64) if classified else None
    )
    false_negative = (
        np.zeros(emitted_class_count, dtype=np.int64) if classified else None
    )

    roi_count = 0
    for roi in rois:
        roi_count += 1
        ground_truth, prediction = _supported_instances(roi)
        if classified:
            ground_truth, ground_truth_ids, ground_truth_classes = _emitted_instances(
                ground_truth,
                roi.ground_truth_classes,
                emitted_channel_to_index,
                valid_class_channels,
                "ground truth",
            )
            prediction, prediction_ids, prediction_classes = _emitted_instances(
                prediction,
                roi.prediction_classes,
                emitted_channel_to_index,
                valid_class_channels,
                "prediction",
            )
        else:
            if (
                roi.ground_truth_classes is not None
                or roi.prediction_classes is not None
            ):
                raise ValueError(
                    "class mappings require a classified instance metric policy"
                )
            ground_truth_ids = _positive_ids(ground_truth)
            prediction_ids = _positive_ids(prediction)
            ground_truth_classes = prediction_classes = None

        counts = (len(ground_truth_ids), len(prediction_ids))
        if classified:
            _require_bounded_dense_work(
                *counts,
                max_assignment_matrix_entries,
            )
        iou = _pairwise_iou(
            ground_truth,
            prediction,
            ground_truth_ids,
            prediction_ids,
            max_assignment_matrix_entries,
        )
        for index, threshold in enumerate(_MSA_THRESHOLDS):
            pairs = (
                _global_cellpose_matches(
                    iou,
                    threshold,
                    max_assignment_matrix_entries,
                )
                if classified and threshold == 0.50
                else _component_cellpose_matches(iou, threshold)
            )
            matched = len(pairs[0])
            instance_counts[:, index] += (
                matched,
                counts[1] - matched,
                counts[0] - matched,
            )
            if classified and threshold == 0.50:
                _add_class_counts(
                    pairs,
                    ground_truth_classes,
                    prediction_classes,
                    true_positive,
                    false_positive,
                    false_negative,
                )

        del roi, ground_truth, prediction, iou

    if roi_count == 0:
        raise ValueError("instance metrics require at least one ROI")

    instance_scores = segmentation_accuracy(*instance_counts)
    class_scores = (
        class_f1(true_positive, false_positive, false_negative)
        if classified
        else np.empty(0, dtype=np.float64)
    )
    return InstanceMetricResult(
        ap50=_value_or_none(instance_scores[0]),
        msa=_value_or_none(mean_defined(instance_scores)),
        object_macro_f1=_value_or_none(mean_defined(class_scores)),
        object_class_f1={
            channel: _value_or_none(class_scores[index])
            for channel, index in (emitted_channel_to_index or {}).items()
        },
    )


def _supported_instances(
    roi: InstanceMetricRoi,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Validate one ROI and remove every pixel outside its metric support."""

    ground_truth = _instance_labels(roi.ground_truth, "ground truth")
    prediction = _instance_labels(roi.prediction, "prediction")
    support = np.asarray(roi.support)
    if support.dtype != np.bool_ or support.ndim != 2:
        raise ValueError("metric support must be a two-dimensional boolean array")
    if ground_truth.shape != prediction.shape or ground_truth.shape != support.shape:
        raise ValueError("metric masks and support must have the same shape")
    return (
        np.where(support, ground_truth, 0),
        np.where(support, prediction, 0),
    )


def _instance_labels(value: NDArray[np.integer], name: str) -> NDArray[np.int64]:
    """Normalize a nonnegative two-dimensional integer instance raster."""

    labels = np.asarray(value)
    if (
        labels.ndim != 2
        or labels.dtype == np.bool_
        or not np.issubdtype(labels.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be a two-dimensional integer array")
    if labels.size and np.min(labels) < 0:
        raise ValueError(f"{name} contains a negative instance ID")
    return labels.astype(np.int64, copy=False)


def _emitted_instances(
    labels: NDArray[np.int64],
    classes: Mapping[int, int] | None,
    emitted_channel_to_index: Mapping[int, int],
    valid_class_channels: frozenset[int],
    name: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Project one classified raster to classes emitted by normal inference."""

    if classes is None:
        raise ValueError(f"{name} class mapping is required")
    ids = _positive_ids(labels)
    class_values = []
    emitted_ids = []
    missing = object()
    for instance_id in ids:
        class_id = classes.get(int(instance_id), missing)
        if (
            class_id is missing
            or isinstance(class_id, bool)
            or not isinstance(class_id, int)
            or class_id not in valid_class_channels
        ):
            raise ValueError(f"{name} instance class is missing or invalid")
        if class_id in emitted_channel_to_index:
            emitted_ids.append(instance_id)
            class_values.append(emitted_channel_to_index[class_id])
    selected_ids = np.asarray(emitted_ids, dtype=np.int64)
    if len(selected_ids) == len(ids):
        projected = labels
    else:
        projected = np.where(np.isin(labels, selected_ids), labels, 0)
    return projected, selected_ids, np.asarray(class_values, dtype=np.int64)


def _positive_ids(labels: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return sorted positive IDs without assuming that IDs are contiguous."""

    return np.unique(labels[labels > 0])


def _pairwise_iou(
    ground_truth: NDArray[np.int64],
    prediction: NDArray[np.int64],
    ground_truth_ids: NDArray[np.int64],
    prediction_ids: NDArray[np.int64],
    max_assignment_matrix_entries: int,
) -> _PairwiseIoU:
    """Build independent dense IoU components from sparse pixel overlaps."""

    true_count = len(ground_truth_ids)
    predicted_count = len(prediction_ids)
    if true_count == 0 or predicted_count == 0:
        return _PairwiseIoU(true_count, predicted_count, ())

    true_positive_pixels = ground_truth > 0
    predicted_positive_pixels = prediction > 0
    true_positions = np.searchsorted(
        ground_truth_ids, ground_truth[true_positive_pixels]
    )
    predicted_positions = np.searchsorted(
        prediction_ids, prediction[predicted_positive_pixels]
    )
    true_areas = np.bincount(true_positions, minlength=true_count)
    predicted_areas = np.bincount(predicted_positions, minlength=predicted_count)

    shared = true_positive_pixels & predicted_positive_pixels
    raw_pair_indices = np.searchsorted(
        ground_truth_ids, ground_truth[shared]
    ) * predicted_count + np.searchsorted(prediction_ids, prediction[shared])
    if not len(raw_pair_indices):
        return _PairwiseIoU(true_count, predicted_count, ())

    pair_indices, intersections = np.unique(raw_pair_indices, return_counts=True)
    edge_truth = pair_indices // predicted_count
    edge_prediction = pair_indices % predicted_count
    prediction_nodes = true_count + edge_prediction
    rows = np.concatenate((edge_truth, prediction_nodes))
    columns = np.concatenate((prediction_nodes, edge_truth))
    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(true_count + predicted_count, true_count + predicted_count),
    )
    _, labels = connected_components(graph, directed=False)
    edge_components = labels[edge_truth]
    order = np.argsort(edge_components, kind="stable")
    ordered_components = edge_components[order]
    boundaries = np.flatnonzero(np.diff(ordered_components)) + 1

    components = []
    retained_dense_cells = 0
    for edge_positions in np.split(order, boundaries):
        true_indices = np.unique(edge_truth[edge_positions])
        predicted_indices = np.unique(edge_prediction[edge_positions])
        retained_dense_cells += len(true_indices) * len(predicted_indices)
        if retained_dense_cells > max_assignment_matrix_entries:
            raise ValueError("validation instance matching exceeds its resource limit")
        local_truth = np.searchsorted(true_indices, edge_truth[edge_positions])
        local_prediction = np.searchsorted(
            predicted_indices, edge_prediction[edge_positions]
        )
        local_intersections = intersections[edge_positions]
        local_unions = (
            true_areas[edge_truth[edge_positions]]
            + predicted_areas[edge_prediction[edge_positions]]
            - local_intersections
        )
        values = np.zeros((len(true_indices), len(predicted_indices)), dtype=np.float64)
        values[local_truth, local_prediction] = local_intersections / local_unions
        components.append(_IoUComponent(true_indices, predicted_indices, values))
    return _PairwiseIoU(true_count, predicted_count, tuple(components))


def _component_cellpose_matches(
    iou: _PairwiseIoU, threshold: float
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return count-equivalent Cellpose matches from independent overlap components."""

    # Zero-IoU edges have zero weight in Cellpose's objective. Removing them splits
    # the maximum qualified-match count into independent positive-overlap graphs.
    # Exact pair identity can still differ in a tie, so classified matching uses the
    # global matrix below instead.
    minimum_count = min(iou.ground_truth_count, iou.prediction_count)
    if minimum_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    accepted_truth = []
    accepted_predictions = []
    for component in iou.components:
        costs = -(component.values >= threshold).astype(float) - component.values / (
            2 * minimum_count
        )
        true_indices, predicted_indices = linear_sum_assignment(costs)
        accepted = component.values[true_indices, predicted_indices] >= threshold
        accepted_truth.append(component.ground_truth_indices[true_indices[accepted]])
        accepted_predictions.append(
            component.prediction_indices[predicted_indices[accepted]]
        )
    if not accepted_truth:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(accepted_truth), np.concatenate(accepted_predictions)


def _global_cellpose_matches(
    iou: _PairwiseIoU,
    threshold: float,
    max_assignment_matrix_entries: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Preserve global Cellpose pair identity for classification metrics."""

    minimum_count = min(iou.ground_truth_count, iou.prediction_count)
    if minimum_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    _require_bounded_dense_work(
        iou.ground_truth_count,
        iou.prediction_count,
        max_assignment_matrix_entries,
    )
    values = np.zeros(
        (iou.ground_truth_count, iou.prediction_count),
        dtype=np.float64,
    )
    for component in iou.components:
        values[np.ix_(component.ground_truth_indices, component.prediction_indices)] = (
            component.values
        )
    costs = -(values >= threshold).astype(float) - values / (2 * minimum_count)
    true_indices, predicted_indices = linear_sum_assignment(costs)
    accepted = values[true_indices, predicted_indices] >= threshold
    return true_indices[accepted], predicted_indices[accepted]


def _require_bounded_dense_work(
    ground_truth_count: int,
    prediction_count: int,
    max_assignment_matrix_entries: int,
) -> None:
    """Fail before allocating a pairwise matrix beyond the worker's resource profile."""

    if ground_truth_count * prediction_count > max_assignment_matrix_entries:
        raise ValueError("validation instance matching exceeds its resource limit")


def _add_class_counts(
    pairs: tuple[NDArray[np.int64], NDArray[np.int64]],
    ground_truth_classes: NDArray[np.int64],
    prediction_classes: NDArray[np.int64],
    true_positive: NDArray[np.int64],
    false_positive: NDArray[np.int64],
    false_negative: NDArray[np.int64],
) -> None:
    """Pool correct, wrong-class, and unmatched object counts for one ROI."""

    matched_truth = set(map(int, pairs[0]))
    matched_predictions = set(map(int, pairs[1]))
    for true_index, predicted_index in zip(*pairs, strict=True):
        true_class = ground_truth_classes[true_index]
        predicted_class = prediction_classes[predicted_index]
        if true_class == predicted_class:
            true_positive[true_class] += 1
        else:
            false_negative[true_class] += 1
            false_positive[predicted_class] += 1
    for index, class_id in enumerate(ground_truth_classes):
        if index not in matched_truth:
            false_negative[class_id] += 1
    for index, class_id in enumerate(prediction_classes):
        if index not in matched_predictions:
            false_positive[class_id] += 1


def _value_or_none(value: float) -> float | None:
    """Project the reducer's zero-denominator marker to the report's sparse values."""

    return None if np.isnan(value) else float(value)
