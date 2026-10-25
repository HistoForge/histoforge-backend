"""Validation aggregation releases each ROI before requesting the next."""

import weakref

import numpy as np
import pytest
from histoforge.cloud.training.trainers.instance.metrics import (
    InstanceMetricRoi,
    evaluate_instance_metrics,
)


@pytest.mark.parametrize("classified", [False, True])
def test_lazy_roi_metrics_equal_eager_values_without_retaining_previous_rasters(
    classified,
):
    references = []

    def next_roi(index, *, track=False):
        if track:
            assert all(reference() is None for reference in references)
        labels = np.array([[1, 1], [0, 0]], np.int32)
        predicted = labels.copy() if index == 0 else np.zeros_like(labels)
        support = np.ones_like(labels, bool)
        if track:
            references.extend(
                weakref.ref(item) for item in (labels, predicted, support)
            )
        return InstanceMetricRoi(
            labels,
            predicted,
            support,
            {1: 1} if classified else None,
            {1: 1} if classified else None,
        )

    kwargs = {"max_assignment_matrix_entries": 100}
    if classified:
        kwargs.update(
            emitted_channel_to_index={1: 0},
            valid_class_channels=frozenset({0, 1}),
        )
    eager = evaluate_instance_metrics([next_roi(index) for index in range(3)], **kwargs)
    lazy = evaluate_instance_metrics(
        (next_roi(index, track=True) for index in range(3)), **kwargs
    )
    assert eager == lazy
    assert all(reference() is None for reference in references)


def test_empty_iterable_is_not_misreported_as_an_all_empty_validation_set():
    with pytest.raises(ValueError, match="at least one ROI"):
        evaluate_instance_metrics(iter(()), max_assignment_matrix_entries=100)


def test_failure_reading_a_later_roi_does_not_return_partial_metrics():
    def rois():
        labels = np.ones((2, 2), np.int32)
        yield InstanceMetricRoi(labels, labels, np.ones((2, 2), bool))
        raise OSError("injected next-ROI failure")

    with pytest.raises(OSError, match="injected"):
        evaluate_instance_metrics(rois(), max_assignment_matrix_entries=100)
