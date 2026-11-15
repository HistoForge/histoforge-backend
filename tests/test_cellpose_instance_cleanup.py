"""Cellpose instance cleanup preserves product area and context semantics."""

import numpy as np
import pytest
from histoforge._vendor.histoforge_models.pathology.instance_cleanup import (
    cleanup_cellpose_instances,
    remove_instances_below_area,
)


def test_shared_final_area_filter_preserves_zero_and_exact_thresholds():
    mask = np.array(
        [
            [4, 4, 0, 0],
            [4, 0, 9, 9],
            [0, 0, 9, 9],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(remove_instances_below_area(mask, 0), mask)
    np.testing.assert_array_equal(
        remove_instances_below_area(mask, 3),
        np.array([[1, 1, 0, 0], [1, 0, 2, 2], [0, 0, 2, 2]]),
    )
    np.testing.assert_array_equal(
        remove_instances_below_area(mask, 4),
        np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
    )


def test_shared_final_area_filter_handles_no_background_and_invalid_values():
    mask = np.full((2, 3), 7, dtype=np.uint16)

    np.testing.assert_array_equal(remove_instances_below_area(mask, 6), np.ones((2, 3)))
    assert not remove_instances_below_area(mask, 7).any()
    for value in (-1, 1.0, True):
        with pytest.raises(ValueError, match="nonnegative integer"):
            remove_instances_below_area(mask, value)


def test_shared_final_area_filter_preserves_ids_for_a_later_context_check():
    mask = np.array(
        [[4, 4, 0, 9], [4, 0, 9, 9], [0, 0, 0, 9]],
        dtype=np.int32,
    )

    observed = remove_instances_below_area(
        mask,
        4,
        protected_instance_ids={4},
    )

    np.testing.assert_array_equal(
        observed,
        np.array([[1, 1, 0, 2], [1, 0, 2, 2], [0, 0, 0, 2]]),
    )


def test_upstream_cleanup_filters_a_background_free_lowest_label():
    mask = np.array([[5, 9, 9], [9, 9, 9]], dtype=np.int32)

    np.testing.assert_array_equal(
        cleanup_cellpose_instances(mask, 2),
        np.array([[0, 1, 1], [1, 1, 1]], dtype=np.int32),
    )


def test_background_frame_remains_through_upstream_hole_filling():
    mask = np.array(
        [
            [1, 3, 3, 3, 3],
            [3, 2, 2, 2, 3],
            [3, 2, 1, 2, 3],
            [3, 2, 0, 2, 3],
            [3, 2, 2, 2, 3],
        ],
        dtype=np.int32,
    )

    result = cleanup_cellpose_instances(mask, 2)

    assert result[0, 0] == 0
    np.testing.assert_array_equal(np.unique(result), np.array([0, 1, 2]))
