"""Complete-instance targets must survive model-field and ROI boundaries."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    build_managed_targets,
)
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_training_geometry,
)
from histoforge.cloud.training.trainers.instance import (
    field_targets as training_field_targets,
)
from histoforge.cloud.training.trainers.instance.field_targets import (
    build_training_field_targets,
)
from tests.test_training_label_rasterizer import _grid, _object, _rectangle


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
@pytest.mark.parametrize("classified", [False, True])
def test_complete_object_targets_survive_field_and_roi_clipping(head_type, classified):
    grid = _grid(64, 64)
    geometry = _rectangle(8, 7, 51, 42)
    roi = _rectangle(16, 17, 28, 29)
    field = {"x": 14, "y": 13, "width": 18, "height": 21}
    objects = [_object(geometry, "Cell")]
    before = copy.deepcopy(objects)

    actual = build_training_field_targets(
        roi_geometry=roi,
        objects=objects,
        model_output=_model_output(("Cell",), classified),
        head_type=head_type,
        raster_transform=grid,
        output_rect=field,
        physical_slide_bounds=(0, 0, 64, 64),
    )
    full_mask = rasterize_training_geometry(
        geometry, grid, {"x": 0, "y": 0, "width": 64, "height": 64}
    ).astype(np.int32)
    reference = build_managed_targets(full_mask, head_type=head_type)
    rows, columns = slice(13, 34), slice(14, 32)
    assert np.array_equal(
        actual.targets.foreground, reference.foreground[rows, columns]
    )
    assert np.array_equal(
        actual.targets.directional, reference.directional[:, rows, columns]
    )
    premature = build_managed_targets(full_mask[rows, columns], head_type=head_type)
    assert not np.array_equal(actual.targets.directional, premature.directional)
    assert np.any(actual.targets.foreground[~actual.supervision_mask])
    assert actual.supervision_mask.sum() == 12 * 12
    assert actual.real_pixel_mask.all()
    if classified:
        assert np.array_equal(actual.targets.classes, actual.targets.foreground)
    else:
        assert actual.targets.classes is None
    if head_type == "flow":
        assert np.array_equal(
            actual.targets.instance_flow_weights,
            reference.instance_flow_weights[rows, columns],
        )
    else:
        assert actual.targets.instance_flow_weights is None
    assert objects == before


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_complete_geometry_can_extend_beyond_the_declared_target_grid(head_type):
    geometry = _rectangle(20, 5, 48, 40)
    field = {"x": 18, "y": 8, "width": 14, "height": 16}
    actual = _build(
        head_type=head_type,
        grid=_grid(32, 32),
        roi=_rectangle(0, 0, 32, 32),
        field=field,
        objects=[_object(geometry, "Cell")],
    )
    reference_mask = rasterize_training_geometry(
        geometry, _grid(64, 64), {"x": 0, "y": 0, "width": 64, "height": 64}
    ).astype(np.int32)
    reference = build_managed_targets(reference_mask, head_type=head_type)
    assert np.array_equal(
        actual.targets.directional, reference.directional[:, 8:24, 18:32]
    )


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_fractional_anisotropic_grid_keeps_the_same_pixel_phase(head_type):
    grid = _grid(32, 32, translate_x=100.1, translate_y=200.3)
    grid["raster_to_slide"].update(scale_x=0.3, scale_y=0.7)
    geometry = _rectangle(101.05, 201.4, 107.05, 215.0)
    field = {"x": 4, "y": 3, "width": 17, "height": 18}
    actual = _build(
        head_type=head_type,
        grid=grid,
        roi=_rectangle(100.1, 200.3, 109.7, 222.7),
        field=field,
        objects=[_object(geometry, "Cell")],
        bounds=(0, 0, 1000, 1000),
    )
    reference_mask = rasterize_training_geometry(
        geometry, grid, {"x": 0, "y": 0, "width": 32, "height": 32}
    ).astype(np.int32)
    reference = build_managed_targets(reference_mask, head_type=head_type)
    assert np.array_equal(actual.targets.foreground, reference.foreground[3:21, 4:21])
    assert np.array_equal(
        actual.targets.directional, reference.directional[:, 3:21, 4:21]
    )


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
@pytest.mark.parametrize("offset", [100.1, 1_000_000.3])
def test_object_local_half_pixel_edges_match_the_original_grid(head_type, offset):
    scale = 0.3
    grid = _grid(
        32, 32, translate_x=offset, translate_y=offset, scale_x=scale, scale_y=scale
    )
    geometry = _rectangle(
        offset + 3.5 * scale,
        offset + 5.5 * scale,
        offset + 18.5 * scale,
        offset + 21.5 * scale,
    )
    field = {"x": 0, "y": 0, "width": 32, "height": 32}
    actual = _build(
        head_type=head_type,
        grid=grid,
        roi=_rectangle(offset, offset, offset + 32 * scale, offset + 32 * scale),
        field=field,
        objects=[_object(geometry, "Cell")],
        bounds=(0, 0, 2_000_000, 2_000_000),
    )
    complete_mask = rasterize_training_geometry(geometry, grid, field).astype(np.int32)
    reference = build_managed_targets(complete_mask, head_type=head_type)
    assert np.array_equal(actual.targets.foreground, reference.foreground)
    assert np.array_equal(actual.targets.directional, reference.directional)


@pytest.mark.parametrize("head_type", ["flow", "hovernet"])
def test_complete_object_grid_can_start_before_the_target_origin(head_type):
    geometry = _rectangle(8, 9, 42, 41)
    actual = _build(
        head_type=head_type,
        grid=_grid(16, 16, translate_x=16, translate_y=16),
        roi=_rectangle(16, 16, 32, 32),
        objects=[_object(geometry, "Cell")],
    )
    complete_mask = rasterize_training_geometry(
        geometry, _grid(64, 64), {"x": 0, "y": 0, "width": 64, "height": 64}
    ).astype(np.int32)
    reference = build_managed_targets(complete_mask, head_type=head_type)
    assert np.array_equal(
        actual.targets.directional, reference.directional[:, 16:32, 16:32]
    )


def test_training_only_class_and_background_are_preserved():
    actual = _build(
        objects=[
            _object(_rectangle(2, 2, 5, 5), "Cell"),
            _object(_rectangle(8, 2, 11, 5), "Other"),
        ],
        classes=["Cell", "Other"],
        classified=True,
    )
    assert np.all(actual.targets.classes[2:5, 2:5] == 1)
    assert np.all(actual.targets.classes[2:5, 8:11] == 2)
    assert np.all(actual.targets.classes[actual.targets.foreground == 0] == 0)


@pytest.mark.parametrize("classified", [False, True])
def test_negative_fields_have_supported_background_and_no_foreground(classified):
    actual = _build(objects=[], classified=classified)
    assert actual.supervision_mask.all()
    assert actual.real_pixel_mask.all()
    assert not actual.targets.directional.any()
    assert not actual.targets.foreground.any()
    if classified:
        assert not actual.targets.classes.any()


def test_roi_holes_and_image_overhang_are_independent_support_masks():
    roi = {
        "type": "Polygon",
        "coordinates": [
            _rectangle(0, 0, 16, 16)["coordinates"][0],
            list(reversed(_rectangle(5, 5, 8, 8)["coordinates"][0])),
        ],
    }
    actual = _build(
        roi=roi,
        objects=[_object(_rectangle(3, 3, 14, 14), "Cell")],
        bounds=(0, 0, 12, 16),
    )
    assert not actual.supervision_mask[5:8, 5:8].any()
    assert actual.targets.foreground[5:8, 5:8].all()
    assert not actual.real_pixel_mask[:, 12:].any()
    assert actual.targets.foreground[3:14, 12:14].all()
    assert actual.real_pixel_mask[5:8, 5:8].all()


@pytest.mark.parametrize("excluded_by", ["roi", "image"])
def test_overlap_is_rejected_only_on_supervised_real_pixels(excluded_by):
    objects = [
        _object(_rectangle(1, 1, 8, 8), "Cell"),
        _object(_rectangle(7, 7, 13, 13), "Cell"),
    ]
    with pytest.raises(ValueError, match="share a supervised real pixel"):
        _build(objects=objects)
    boundary = _rectangle(0, 0, 16, 6)
    actual = _build(
        objects=objects,
        roi=boundary if excluded_by == "roi" else None,
        bounds=(0, 0, 16, 6) if excluded_by == "image" else (0, 0, 16, 16),
    )
    assert actual.targets.foreground[1:6, 1:8].all()


def test_object_holes_and_disconnected_parts_are_not_filled_or_split():
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [
            [
                _rectangle(1, 1, 7, 7)["coordinates"][0],
                list(reversed(_rectangle(3, 3, 5, 5)["coordinates"][0])),
            ],
            _rectangle(10, 2, 13, 5)["coordinates"],
        ],
    }
    actual = _build(objects=[_object(geometry, "Cell")])
    expected = rasterize_training_geometry(
        geometry, _grid(16, 16), {"x": 0, "y": 0, "width": 16, "height": 16}
    )
    assert np.array_equal(actual.targets.foreground, expected)
    assert not actual.targets.foreground[3:5, 3:5].any()


def test_large_roi_allocates_only_the_requested_field_and_complete_object(monkeypatch):
    calls = []
    generator = training_field_targets.build_managed_targets

    def measured(labels, **kwargs):
        calls.append(labels.shape)
        return generator(labels, **kwargs)

    monkeypatch.setattr(training_field_targets, "build_managed_targets", measured)
    offset = 1_000_000
    actual = _build(
        grid=_grid(2_000_000, 2_000_000),
        roi=_rectangle(0, 0, 2_000_000, 2_000_000),
        field={"x": offset, "y": offset, "width": 16, "height": 16},
        bounds=(0, 0, 2_000_000, 2_000_000),
        objects=[
            _object(_rectangle(offset + 2, offset + 3, offset + 8, offset + 9), "Cell")
        ],
    )
    assert calls == [(10, 10)]
    assert actual.targets.directional.shape == (2, 16, 16)


def test_zero_support_is_rejected_before_target_generation(monkeypatch):
    monkeypatch.setattr(
        training_field_targets,
        "build_managed_targets",
        lambda *_args, **_kwargs: pytest.fail("no target work for unsupported field"),
    )
    with pytest.raises(ValueError, match="no supervised real pixels"):
        _build(objects=[], roi=_rectangle(100, 100, 101, 101))


def _build(
    *,
    objects,
    grid=None,
    roi=None,
    field=None,
    bounds=(0, 0, 64, 64),
    classes=("Cell",),
    classified=False,
    head_type="flow",
):
    return build_training_field_targets(
        roi_geometry=roi or _rectangle(0, 0, 16, 16),
        objects=objects,
        model_output=_model_output(classes, classified),
        head_type=head_type,
        raster_transform=grid or _grid(16, 16),
        output_rect=field or {"x": 0, "y": 0, "width": 16, "height": 16},
        physical_slide_bounds=bounds,
    )


def _model_output(classes, classified):
    if not classified:
        return {"type": "instance_segmentation", "classes": list(classes)}
    return {
        "type": "instance_segmentation_and_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            *[
                {
                    "channel": index,
                    "role": "emitted" if index == 1 else "training_only",
                    "pathClass": path_class,
                }
                for index, path_class in enumerate(classes, start=1)
            ],
        ],
    }
