"""Exact pixel-coordinate and vector fixtures for every admitted spatial transform."""

from itertools import product

import numpy as np
import pytest
from histoforge.cloud.training.trainers.instance.field_reader import TrainingField
from histoforge.cloud.training.trainers.instance.spatial_augmentation import (
    transform_training_field,
)
from tests.test_training_batches import _field
from tests.test_training_field_targets import _build
from tests.test_training_label_rasterizer import _grid, _object, _rectangle

TRANSFORMS = list(product(range(4), (False, True), (False, True)))


def _arrays(field):
    """Give each spatial array a name without replacing the production record."""

    arrays = {
        "pixels": field.pixels,
        "directional": field.labels.targets.directional,
        "foreground": field.labels.targets.foreground,
        "supervision": field.labels.supervision_mask,
        "real": field.labels.real_pixel_mask,
    }
    if field.labels.targets.classes is not None:
        arrays["classes"] = field.labels.targets.classes
    if field.labels.targets.instance_flow_weights is not None:
        arrays["instance_flow_weights"] = field.labels.targets.instance_flow_weights
    return arrays


def _coordinate_map(height, width, turns, horizontal, vertical):
    """Independent affine-coordinate oracle, not NumPy's array rotations/flips."""

    matrix = np.array(
        (
            ((1, 0), (0, 1)),
            ((0, -1), (1, 0)),
            ((-1, 0), (0, -1)),
            ((0, 1), (-1, 0)),
        )[turns],
        dtype=np.int64,
    )
    if horizontal:
        matrix[1] *= -1
    if vertical:
        matrix[0] *= -1
    corners = np.array(list(product((0, height - 1), (0, width - 1)))) @ matrix.T
    offset = -corners.min(axis=0)
    shape = tuple(corners.max(axis=0) + offset + 1)
    return matrix, offset, shape


def _assert_transform(field, actual, turns, horizontal, vertical):
    """Map each original pixel and its vector by the same affine transform."""

    height, width = field.pixels.shape[:2]
    matrix, offset, shape = _coordinate_map(height, width, turns, horizontal, vertical)
    source, transformed = _arrays(field), _arrays(actual)
    assert source.keys() == transformed.keys()
    assert actual.pixels.shape == (*shape, 3)
    assert actual.labels.targets.directional.shape == (2, *shape)
    for name, value in source.items():
        assert transformed[name].dtype == value.dtype
        assert not np.shares_memory(transformed[name], value)
        assert transformed[name].flags.writeable
        if name not in {"pixels", "directional"}:
            assert transformed[name].shape == shape
    for y, x in product(range(height), range(width)):
        out_y, out_x = matrix @ (y, x) + offset
        for name, value in source.items():
            if name == "directional":
                expected = matrix.astype(np.float32) @ value[:, y, x]
                observed = transformed[name][:, out_y, out_x]
            else:
                expected = value[y, x]
                observed = transformed[name][out_y, out_x]
            np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("turns,horizontal,vertical", TRANSFORMS)
@pytest.mark.parametrize("classified", [False, True])
@pytest.mark.parametrize("pixel_dtype", [np.uint8, np.float32])
def test_every_transform_keeps_pixels_targets_and_masks_aligned(
    turns, horizontal, vertical, classified, pixel_dtype
):
    # Odd, rectangular dimensions exercise the natural field before alignment.
    field = _field(7, 11, classified=classified)
    if pixel_dtype == np.uint8:
        field = TrainingField(
            np.nan_to_num(field.pixels, nan=255).astype(np.uint8), field.labels
        )
    before = {name: value.copy() for name, value in _arrays(field).items()}
    for value in _arrays(field).values():
        value.flags.writeable = False
    actual = transform_training_field(
        field,
        quarter_turns=turns,
        flip_horizontal=horizontal,
        flip_vertical=vertical,
    )

    _assert_transform(field, actual, turns, horizontal, vertical)
    # A later transform may mutate its private result, never a retained input.
    for value in _arrays(actual).values():
        value[:] = 0
    for name, value in _arrays(field).items():
        np.testing.assert_array_equal(value, before[name])


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("turns,horizontal,vertical", TRANSFORMS)
def test_complete_object_directions_survive_field_and_roi_edges(
    head, turns, horizontal, vertical
):
    labels = _build(
        head_type=head,
        classified=True,
        grid=_grid(64, 64),
        roi=_rectangle(16, 17, 28, 29),
        field={"x": 14, "y": 13, "width": 18, "height": 21},
        objects=[_object(_rectangle(8, 7, 51, 42), "Cell")],
        bounds=(0, 0, 29, 30),
    )
    y, x = np.mgrid[:21, :18]
    field = TrainingField(np.stack((y, x, x + y), axis=-1).astype(np.uint8), labels)
    assert np.any(labels.targets.foreground[~labels.supervision_mask])
    assert np.any(labels.targets.foreground[~labels.real_pixel_mask])
    actual = transform_training_field(
        field,
        quarter_turns=turns,
        flip_horizontal=horizontal,
        flip_vertical=vertical,
    )
    _assert_transform(field, actual, turns, horizontal, vertical)
    assert actual.labels.supervision_mask.sum() == labels.supervision_mask.sum()
    assert actual.labels.real_pixel_mask.sum() == labels.real_pixel_mask.sum()


@pytest.mark.parametrize("turns,equivalent", [(-1, 3), (4, 0), (5, 1)])
def test_integer_quarter_turns_compose_without_resampling(turns, equivalent):
    field = _field()
    actual = transform_training_field(field, quarter_turns=turns)
    expected = transform_training_field(field, quarter_turns=equivalent)
    for name, value in _arrays(actual).items():
        np.testing.assert_array_equal(value, _arrays(expected)[name])
