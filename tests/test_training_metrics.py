from __future__ import annotations

import numpy as np
import pytest
from histoforge.cloud.training.trainers.instance.metrics import (
    InstanceMetricRoi,
    evaluate_instance_metrics,
)

_MAX_ASSIGNMENT_MATRIX_ENTRIES = 100_000_000
_ONE_EMITTED_CHANNEL = {1: 0}
_TWO_EMITTED_CHANNELS = {1: 0, 2: 1}
_ONE_EMITTED_HEAD = frozenset({0, 1})
_TWO_EMITTED_HEAD = frozenset({0, 1, 2})
_TWO_EMITTED_ONE_TRAINING_ONLY_HEAD = frozenset({0, 1, 2, 3})


def _evaluate(
    rois: list[InstanceMetricRoi],
    *,
    emitted_channel_to_index: dict[int, int] | None = None,
    valid_class_channels: frozenset[int] | None = None,
):
    return evaluate_instance_metrics(
        rois,
        max_assignment_matrix_entries=_MAX_ASSIGNMENT_MATRIX_ENTRIES,
        emitted_channel_to_index=emitted_channel_to_index,
        valid_class_channels=valid_class_channels,
    )


def test_ap50_and_msa_exclude_empty_rois_and_respect_support() -> None:
    ground_truth = np.array([[7, 7, 8], [7, 7, 8]], dtype=np.int32)
    prediction = np.array([[42, 42, 9], [42, 42, 9]], dtype=np.int32)
    support = np.array([[True, True, False], [True, True, False]])
    empty = np.zeros((2, 3), dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(ground_truth, prediction, support),
            InstanceMetricRoi(empty, empty, np.ones_like(empty, dtype=bool)),
        ]
    )

    assert result.ap50 == 1.0
    assert result.msa == 1.0
    assert result.object_macro_f1 is None


def test_msa_averages_pooled_threshold_scores() -> None:
    ground_truth = np.array([[1, 1, 1, 1, 0]], dtype=np.int32)
    prediction = np.array([[2, 2, 2, 0, 2]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
            )
        ]
    )

    assert result.ap50 == 1.0
    assert result.msa == pytest.approx(0.3)


def test_empty_ground_truth_with_a_prediction_is_a_defined_zero() -> None:
    ground_truth = np.zeros((2, 2), dtype=np.int32)
    prediction = np.array([[9, 9], [0, 0]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
            )
        ]
    )

    assert result.ap50 == 0.0
    assert result.msa == 0.0


def test_all_empty_validation_metrics_are_unavailable() -> None:
    empty = np.zeros((2, 2), dtype=np.int32)

    result = _evaluate(
        [InstanceMetricRoi(empty, empty, np.ones_like(empty, dtype=bool))]
    )

    assert result.ap50 is None
    assert result.msa is None

    classified = _evaluate(
        [
            InstanceMetricRoi(
                empty,
                empty,
                np.ones_like(empty, dtype=bool),
                ground_truth_classes={},
                prediction_classes={},
            )
        ],
        emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
        valid_class_channels=frozenset({0, 1, 2}),
    )
    assert classified.object_macro_f1 is None
    assert classified.object_class_f1 == {1: None}


def test_instance_scores_pool_counts_instead_of_averaging_rois() -> None:
    perfect = np.array([[1]], dtype=np.int32)
    nine_ground_truth_instances = np.arange(1, 10, dtype=np.int32).reshape(3, 3)

    result = _evaluate(
        [
            InstanceMetricRoi(perfect, perfect, np.ones_like(perfect, dtype=bool)),
            InstanceMetricRoi(
                nine_ground_truth_instances,
                np.zeros((3, 3), dtype=np.int32),
                np.ones((3, 3), dtype=bool),
            ),
        ]
    )

    assert result.ap50 == 0.1
    assert result.msa == 0.1


def test_object_macro_f1_uses_cellpose_tie_assignment() -> None:
    ground_truth = np.array([[10, 10, 20, 20]], dtype=np.int32)
    prediction = np.array([[30, 30, 30, 30]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
                ground_truth_classes={10: 1, 20: 2},
                prediction_classes={30: 1},
            )
        ],
        emitted_channel_to_index=_TWO_EMITTED_CHANNELS,
        valid_class_channels=_TWO_EMITTED_HEAD,
    )

    assert result.ap50 == 0.5
    assert result.msa == pytest.approx(0.05)
    assert result.object_macro_f1 == 0.5


def test_threshold_first_assignment_wins_over_larger_total_iou() -> None:
    ground_truth = np.array([[1, 1, 1, 2]], dtype=np.int32)
    prediction = np.array([[1, 1, 2, 1]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
            )
        ]
    )

    assert result.ap50 == pytest.approx(1 / 3)
    assert result.msa == pytest.approx(1 / 30)


def test_support_mask_preserves_one_id_across_disconnected_fragments() -> None:
    ground_truth = np.array([[7, 8, 7]], dtype=np.int32)
    prediction = np.array([[42, 0, 43]], dtype=np.int32)
    support = np.array([[True, False, True]])

    result = _evaluate([InstanceMetricRoi(ground_truth, prediction, support)])

    assert result.ap50 == 0.5
    assert result.msa == pytest.approx(0.05)


def test_instance_scores_include_every_disconnected_overlap_component() -> None:
    ground_truth = np.array([[1, 1, 0, 2, 2]], dtype=np.int32)
    prediction = np.array([[3, 3, 0, 4, 0]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
            )
        ]
    )

    assert result.ap50 == 1.0
    assert result.msa == pytest.approx(0.4)


def test_wrong_class_match_is_segmentation_tp_but_classification_error() -> None:
    ground_truth = np.array([[1, 1], [1, 1]], dtype=np.int32)
    prediction = np.array([[2, 2], [2, 2]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
                ground_truth_classes={1: 1},
                prediction_classes={2: 2},
            )
        ],
        emitted_channel_to_index=_TWO_EMITTED_CHANNELS,
        valid_class_channels=_TWO_EMITTED_HEAD,
    )

    assert result.ap50 == 1.0
    assert result.msa == 1.0
    assert result.object_macro_f1 == 0.0
    assert result.object_class_f1 == {1: 0.0, 2: 0.0}


def test_training_only_instances_are_removed_before_matching() -> None:
    emitted_as_training_only = InstanceMetricRoi(
        np.array([[1, 1]], dtype=np.int32),
        np.array([[2, 2]], dtype=np.int32),
        np.ones((1, 2), dtype=bool),
        ground_truth_classes={1: 1},
        prediction_classes={2: 3},
    )
    training_only_as_emitted = InstanceMetricRoi(
        np.array([[3, 3]], dtype=np.int32),
        np.array([[4, 4]], dtype=np.int32),
        np.ones((1, 2), dtype=bool),
        ground_truth_classes={3: 3},
        prediction_classes={4: 2},
    )
    correctly_training_only = InstanceMetricRoi(
        np.array([[5, 5]], dtype=np.int32),
        np.array([[6, 6]], dtype=np.int32),
        np.ones((1, 2), dtype=bool),
        ground_truth_classes={5: 3},
        prediction_classes={6: 3},
    )

    result = _evaluate(
        [emitted_as_training_only, training_only_as_emitted, correctly_training_only],
        emitted_channel_to_index=_TWO_EMITTED_CHANNELS,
        valid_class_channels=_TWO_EMITTED_ONE_TRAINING_ONLY_HEAD,
    )

    assert result.ap50 == 0.0
    assert result.msa == 0.0
    assert result.object_macro_f1 == 0.0


def test_object_f1_pools_counts_across_rois_before_averaging_classes() -> None:
    perfect = np.array([[1]], dtype=np.int32)
    nine_ground_truth_instances = np.arange(1, 10, dtype=np.int32).reshape(3, 3)

    result = _evaluate(
        [
            InstanceMetricRoi(
                perfect,
                perfect,
                np.ones_like(perfect, dtype=bool),
                ground_truth_classes={1: 1},
                prediction_classes={1: 1},
            ),
            InstanceMetricRoi(
                nine_ground_truth_instances,
                np.zeros((3, 3), dtype=np.int32),
                np.ones((3, 3), dtype=bool),
                ground_truth_classes={index: 1 for index in range(1, 10)},
                prediction_classes={},
            ),
        ],
        emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
        valid_class_channels=_ONE_EMITTED_HEAD,
    )

    assert result.object_macro_f1 == pytest.approx(2 / 11)
    assert result.object_class_f1 == {1: pytest.approx(2 / 11)}


def test_per_class_f1_uses_explicit_channels_and_prediction_only_support() -> None:
    truth = np.array([[0, 1, 0, 0]], dtype=np.int32)
    prediction = np.array([[0, 7, 0, 8]], dtype=np.int32)
    result = _evaluate(
        [
            InstanceMetricRoi(
                truth,
                prediction,
                np.ones_like(truth, dtype=bool),
                ground_truth_classes={1: 5},
                prediction_classes={7: 5, 8: 2},
            )
        ],
        emitted_channel_to_index={5: 1, 2: 0, 8: 2},
        valid_class_channels=frozenset({0, 1, 2, 5, 8}),
    )
    assert result.object_class_f1 == {2: 0.0, 5: 1.0, 8: None}
    assert result.object_macro_f1 == 0.5


def test_pooled_scores_are_invariant_to_roi_order_and_partition() -> None:
    # Each object lies entirely in one sub-ROI: no changed matching/support.
    truth = np.array([[0, 1, 0, 2, 0, 3]], dtype=np.int32)
    prediction = np.array([[0, 1, 0, 0, 0, 3]], dtype=np.int32)
    whole = _evaluate([InstanceMetricRoi(truth, prediction, np.ones_like(truth, bool))])
    parts = [
        InstanceMetricRoi(t, p, np.ones_like(t, bool))
        for t, p in zip(
            np.split(truth, 3, axis=1), np.split(prediction, 3, axis=1), strict=True
        )
    ]
    assert _evaluate(parts) == whole == _evaluate(reversed(parts))
    assert whole.ap50 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "roi",
    [
        InstanceMetricRoi(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.int32),
            np.ones((2, 2), dtype=bool),
        ),
        InstanceMetricRoi(
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((2, 3), dtype=np.int32),
            np.ones((2, 2), dtype=bool),
        ),
        InstanceMetricRoi(
            np.zeros((2, 2), dtype=np.int32),
            np.zeros((2, 2), dtype=np.int32),
            np.ones((2, 2), dtype=np.uint8),
        ),
    ],
)
def test_metric_inputs_are_closed_and_shape_safe(roi: InstanceMetricRoi) -> None:
    with pytest.raises(ValueError):
        _evaluate([roi])


def test_classified_metrics_require_complete_valid_mappings() -> None:
    labels = np.array([[7]], dtype=np.int32)
    support = np.ones_like(labels, dtype=bool)

    with pytest.raises(ValueError, match="mapping is required"):
        _evaluate(
            [InstanceMetricRoi(labels, labels, support)],
            emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
            valid_class_channels=_ONE_EMITTED_HEAD,
        )
    with pytest.raises(ValueError, match="missing or invalid"):
        _evaluate(
            [
                InstanceMetricRoi(
                    labels,
                    labels,
                    support,
                    ground_truth_classes={7: -1},
                    prediction_classes={7: 1},
                )
            ],
            emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
            valid_class_channels=_ONE_EMITTED_HEAD,
        )
    with pytest.raises(ValueError, match="indices must be contiguous"):
        _evaluate(
            [
                InstanceMetricRoi(
                    labels,
                    labels,
                    support,
                    ground_truth_classes={7: 1},
                    prediction_classes={7: 1},
                )
            ],
            emitted_channel_to_index={1: 1},
            valid_class_channels=_ONE_EMITTED_HEAD,
        )


def test_classified_metrics_reject_unknown_classes_and_discard_background() -> None:
    labels = np.array([[7]], dtype=np.int32)
    support = np.ones_like(labels, dtype=bool)

    with pytest.raises(ValueError, match="missing or invalid"):
        _evaluate(
            [
                InstanceMetricRoi(
                    labels,
                    labels,
                    support,
                    ground_truth_classes={7: 1},
                    prediction_classes={7: 999},
                )
            ],
            emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
            valid_class_channels=_ONE_EMITTED_HEAD,
        )

    background = _evaluate(
        [
            InstanceMetricRoi(
                labels,
                labels,
                support,
                ground_truth_classes={7: 1},
                prediction_classes={7: 0},
            )
        ],
        emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
        valid_class_channels=_ONE_EMITTED_HEAD,
    )
    assert background.ap50 == 0.0
    assert background.msa == 0.0
    assert background.object_macro_f1 == 0.0


def test_classified_tie_uses_one_global_cellpose_assignment() -> None:
    ground_truth = np.array([[0, 1, 2, 2]], dtype=np.int32)
    prediction = np.array([[0, 0, 1, 2]], dtype=np.int32)

    result = _evaluate(
        [
            InstanceMetricRoi(
                ground_truth,
                prediction,
                np.ones_like(ground_truth, dtype=bool),
                ground_truth_classes={1: 2, 2: 1},
                prediction_classes={1: 1, 2: 2},
            )
        ],
        emitted_channel_to_index=_TWO_EMITTED_CHANNELS,
        valid_class_channels=_TWO_EMITTED_HEAD,
    )

    assert result.ap50 == pytest.approx(1 / 3)
    assert result.msa == pytest.approx(1 / 30)
    assert result.object_macro_f1 == 0.0


def test_component_dense_matching_limit_fails_before_solver(monkeypatch) -> None:
    ground_truth = np.array([[1, 1, 2, 2]], dtype=np.int32)
    prediction = np.array([[3, 4, 3, 4]], dtype=np.int32)

    def unexpected_solver(_costs):
        raise AssertionError("solver must not run beyond the dense-work limit")

    monkeypatch.setattr(
        "histoforge.cloud.training.trainers.instance.metrics.linear_sum_assignment",
        unexpected_solver,
    )
    with pytest.raises(ValueError, match="resource limit"):
        evaluate_instance_metrics(
            [
                InstanceMetricRoi(
                    ground_truth,
                    prediction,
                    np.ones_like(ground_truth, dtype=bool),
                )
            ],
            max_assignment_matrix_entries=3,
        )


def test_global_dense_matching_limit_fails_before_solver(monkeypatch) -> None:
    ground_truth = np.array([[1, 0, 2]], dtype=np.int32)
    prediction = np.array([[3, 0, 4]], dtype=np.int32)

    def unexpected_solver(_costs):
        raise AssertionError("solver must not run beyond the dense-work limit")

    monkeypatch.setattr(
        "histoforge.cloud.training.trainers.instance.metrics.linear_sum_assignment",
        unexpected_solver,
    )
    with pytest.raises(ValueError, match="resource limit"):
        evaluate_instance_metrics(
            [
                InstanceMetricRoi(
                    ground_truth,
                    prediction,
                    np.ones_like(ground_truth, dtype=bool),
                    ground_truth_classes={1: 1, 2: 1},
                    prediction_classes={3: 1, 4: 1},
                )
            ],
            max_assignment_matrix_entries=3,
            emitted_channel_to_index=_ONE_EMITTED_CHANNEL,
            valid_class_channels=_ONE_EMITTED_HEAD,
        )


def test_component_dense_matching_limit_is_cumulative(monkeypatch) -> None:
    ground_truth = np.array([[1, 1, 2, 2, 0, 5, 5, 6, 6]], dtype=np.int32)
    prediction = np.array([[3, 4, 3, 4, 0, 7, 8, 7, 8]], dtype=np.int32)

    def unexpected_solver(_costs):
        raise AssertionError("solver must not run beyond the dense-work limit")

    monkeypatch.setattr(
        "histoforge.cloud.training.trainers.instance.metrics.linear_sum_assignment",
        unexpected_solver,
    )
    with pytest.raises(ValueError, match="resource limit"):
        evaluate_instance_metrics(
            [
                InstanceMetricRoi(
                    ground_truth,
                    prediction,
                    np.ones_like(ground_truth, dtype=bool),
                )
            ],
            max_assignment_matrix_entries=4,
        )
