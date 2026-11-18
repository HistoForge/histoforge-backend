from __future__ import annotations

import numpy as np
import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.cloud.training.datasets import (
    label_rasterizer as training_label_rasterizer,
)
from histoforge.cloud.training.datasets.label_rasterizer import (
    inspect_training_roi_labels,
    rasterize_training_geometry,
    training_real_pixel_mask,
)


def _model_output(output_type, classes):
    if output_type == "instance_segmentation":
        return {"type": output_type, "classes": list(classes)}
    return {
        "type": output_type,
        "classChannels": [
            {"channel": 0, "role": "background"},
            *[
                {"channel": index, "role": "emitted", "pathClass": path_class}
                for index, path_class in enumerate(classes, start=1)
            ],
        ],
    }


def test_polygon_holes_and_subpixel_shapes_follow_pixel_centers() -> None:
    grid = _grid(3, 3)
    polygon_with_hole = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]],
            [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]],
        ],
    }

    actual = rasterize_training_geometry(
        polygon_with_hole,
        grid,
        {"x": 0, "y": 0, "width": 3, "height": 3},
    )

    assert np.array_equal(
        actual,
        np.array(
            [[True, True, True], [True, False, True], [True, True, True]],
            dtype=bool,
        ),
    )
    subpixel = rasterize_training_geometry(
        _rectangle(0.1, 0.1, 0.4, 0.4),
        grid,
        {"x": 0, "y": 0, "width": 1, "height": 1},
    )
    assert not subpixel.any()

    multipolygon = {
        "type": "MultiPolygon",
        "coordinates": [
            _rectangle(0, 0, 1, 1)["coordinates"],
            _rectangle(2, 2, 3, 3)["coordinates"],
        ],
    }
    actual_multi = rasterize_training_geometry(
        multipolygon,
        grid,
        {"x": 0, "y": 0, "width": 3, "height": 3},
    )
    assert np.array_equal(
        actual_multi,
        np.array(
            [[True, False, False], [False, False, False], [False, False, True]],
            dtype=bool,
        ),
    )


def test_real_pixel_mask_uses_half_open_physical_bounds() -> None:
    grid = _grid(3, 3, translate_x=-1, translate_y=-1)

    actual = training_real_pixel_mask(
        (0, 0, 2, 2),
        grid,
        {"x": 0, "y": 0, "width": 3, "height": 3},
    )

    assert np.array_equal(
        actual,
        np.array(
            [[False, False, False], [False, True, True], [False, True, True]],
            dtype=bool,
        ),
    )
    exact_boundaries = training_real_pixel_mask(
        (0, 0, 2, 1),
        _grid(3, 1, translate_x=-0.5, translate_y=-0.5),
        {"x": 0, "y": 0, "width": 3, "height": 1},
    )
    assert np.array_equal(
        exact_boundaries,
        np.array([[True, True, False]], dtype=bool),
    )


def test_summary_counts_only_raster_visible_objects_in_class_order() -> None:
    summary = inspect_training_roi_labels(
        roi_geometry=_rectangle(0, 0, 4, 4),
        objects=[
            _object(_rectangle(0, 0, 1, 1), "Cell"),
            _object(_rectangle(2.1, 2.1, 2.4, 2.4), "Cell"),
            _object(_rectangle(3, 3, 4, 4), "Other"),
        ],
        model_output=_model_output(
            "instance_segmentation_and_classification", ["Cell", "Other"]
        ),
        raster_transform=_grid(4, 4),
        physical_slide_bounds=(0, 0, 4, 4),
    )

    assert summary.visible_object_indices == (0, 2)
    assert summary.target_object_counts == {"Cell": 1, "Other": 1}
    assert not summary.is_negative


def test_instance_overlap_is_checked_after_rasterization() -> None:
    arguments = {
        "roi_geometry": _rectangle(0, 0, 4, 4),
        "model_output": _model_output("instance_segmentation", ["Cell"]),
        "raster_transform": _grid(4, 4),
        "physical_slide_bounds": (0, 0, 4, 4),
    }
    with pytest.raises(
        ValueError, match=r"Two training objects overlap near x=1\.50, y=1\.50"
    ):
        inspect_training_roi_labels(
            **arguments,
            objects=[
                _object(_rectangle(0, 0, 2, 2), "Cell"),
                _object(_rectangle(1, 1, 3, 3), "Cell"),
            ],
        )

    touching = inspect_training_roi_labels(
        **arguments,
        objects=[
            _object(_rectangle(0, 0, 1, 1), "Cell"),
            _object(_rectangle(1, 0, 2, 1), "Cell"),
        ],
    )
    assert touching.target_object_counts == {"Cell": 2}


@pytest.mark.parametrize("excluded_by", ["roi", "physical_bounds"])
def test_instance_overlap_outside_effective_support_is_allowed(
    excluded_by: str,
) -> None:
    left = _multipolygon(
        _rectangle(0, 0, 1, 1),
        _rectangle(4, 0, 5, 1),
    )
    right = _multipolygon(
        _rectangle(2, 0, 3, 1),
        _rectangle(4, 0, 5, 1),
    )
    roi = _rectangle(0, 0, 3, 1) if excluded_by == "roi" else _rectangle(0, 0, 6, 1)
    physical_bounds = (0, 0, 6, 1) if excluded_by == "roi" else (0, 0, 3, 1)

    summary = inspect_training_roi_labels(
        roi_geometry=roi,
        objects=[_object(left, "Cell"), _object(right, "Cell")],
        model_output=_model_output("instance_segmentation", ["Cell"]),
        raster_transform=_grid(6, 1),
        physical_slide_bounds=physical_bounds,
    )

    assert summary.visible_object_indices == (0, 1)


def test_instance_overlap_across_collision_partition_is_rejected() -> None:
    with pytest.raises(ValueError, match="Two training objects overlap"):
        inspect_training_roi_labels(
            roi_geometry=_rectangle(0, 0, 2050, 1),
            objects=[
                _object(
                    _multipolygon(
                        _rectangle(0, 0, 1, 1),
                        _rectangle(1023, 0, 1026, 1),
                    ),
                    "Cell",
                ),
                _object(
                    _multipolygon(
                        _rectangle(1024, 0, 1025, 1),
                        _rectangle(2049, 0, 2050, 1),
                    ),
                    "Cell",
                ),
            ],
            model_output=_model_output("instance_segmentation", ["Cell"]),
            raster_transform=_grid(2050, 1),
            physical_slide_bounds=(0, 0, 2050, 1),
        )


@pytest.mark.parametrize(
    "output_type",
    ["instance_segmentation_and_classification", "dense_classification"],
)
def test_overlap_scan_matches_separately_rasterized_masks(
    output_type: str,
) -> None:
    rng = np.random.default_rng(8421)
    grid = _grid(
        12,
        9,
        translate_x=10,
        translate_y=20,
        scale_x=0.75,
        scale_y=1.25,
    )
    roi = _rectangle(10, 20, 19, 31.25)
    output_rect = {"x": 0, "y": 0, "width": 12, "height": 9}
    observed_outcomes: set[bool] = set()

    for _case in range(30):
        geometries = []
        classes = []
        for _object_index in range(8):
            x0 = float(rng.uniform(9.5, 18.5))
            y0 = float(rng.uniform(19.5, 30.5))
            geometries.append(
                _rectangle(
                    x0,
                    y0,
                    x0 + float(rng.uniform(0.2, 2.0)),
                    y0 + float(rng.uniform(0.2, 2.0)),
                )
            )
            classes.append("A" if rng.integers(0, 2) == 0 else "B")

        masks = [
            rasterize_training_geometry(geometry, grid, output_rect)
            for geometry in geometries
        ]
        if output_type == "dense_classification":
            planes = []
            for selected in ("A", "B"):
                selected_masks = [
                    mask
                    for mask, path_class in zip(masks, classes, strict=True)
                    if path_class == selected
                ]
                planes.append(
                    np.logical_or.reduce(selected_masks, axis=0)
                    if selected_masks
                    else np.zeros((9, 12), dtype=bool)
                )
        else:
            planes = masks
        expected_overlap = bool(np.any(np.sum(planes, axis=0) > 1))
        observed_outcomes.add(expected_overlap)
        arguments = {
            "roi_geometry": roi,
            "objects": [
                _object(geometry, path_class)
                for geometry, path_class in zip(geometries, classes, strict=True)
            ],
            "model_output": _model_output(output_type, ["A", "B"]),
            "raster_transform": grid,
            "physical_slide_bounds": (10, 20, 19, 31.25),
        }
        if expected_overlap:
            with pytest.raises(ValueError, match="overlap near"):
                inspect_training_roi_labels(**arguments)
        else:
            inspect_training_roi_labels(**arguments)

    assert observed_outcomes == {False, True}


def test_dense_overlap_unions_same_class_and_rejects_cross_class() -> None:
    common = {
        "roi_geometry": _rectangle(0, 0, 4, 4),
        "model_output": _model_output("dense_classification", ["Tubule", "Lumen"]),
        "raster_transform": _grid(4, 4),
        "physical_slide_bounds": (0, 0, 4, 4),
    }
    same_class = inspect_training_roi_labels(
        **common,
        objects=[
            _object(_rectangle(0, 0, 2, 2), "Tubule"),
            _object(_rectangle(1, 1, 3, 3), "Tubule"),
        ],
    )
    assert same_class.target_object_counts == {"Tubule": 2, "Lumen": 0}

    with pytest.raises(ValueError, match="Two training classes overlap"):
        inspect_training_roi_labels(
            **common,
            objects=[
                _object(_rectangle(0, 0, 2, 2), "Tubule"),
                _object(_rectangle(1, 1, 3, 3), "Lumen"),
            ],
        )


def test_summary_scans_large_grids_in_bounded_windows() -> None:
    summary = inspect_training_roi_labels(
        roi_geometry=_rectangle(0, 0, 2050, 2),
        objects=[_object(_rectangle(2049, 0, 2050, 1), "Cell")],
        model_output=_model_output("instance_segmentation", ["Cell"]),
        raster_transform=_grid(2050, 2),
        physical_slide_bounds=(0, 0, 2050, 2),
    )

    assert summary.target_object_counts == {"Cell": 1}


def test_shifted_anisotropic_subwindow_preserves_complete_geometry() -> None:
    grid = _grid(
        6,
        4,
        translate_x=10,
        translate_y=20,
        scale_x=2,
        scale_y=0.5,
    )
    crossing = _multipolygon(
        _rectangle(12.5, 20.4, 17.5, 21.1),
        _rectangle(18.5, 21.1, 21.5, 21.9),
    )

    complete = rasterize_training_geometry(
        crossing,
        grid,
        {"x": 1, "y": 0, "width": 4, "height": 4},
    )
    support = rasterize_training_geometry(
        _rectangle(14.5, 20.6, 19.5, 21.6),
        grid,
        {"x": 1, "y": 0, "width": 4, "height": 4},
    )

    assert np.array_equal(
        complete,
        np.array(
            [
                [False, False, False, False],
                [True, True, True, False],
                [False, False, False, True],
                [False, False, False, True],
            ],
            dtype=bool,
        ),
    )
    assert np.array_equal(
        support,
        np.array(
            [
                [False, False, False, False],
                [False, True, True, True],
                [False, True, True, True],
                [False, False, False, False],
            ],
            dtype=bool,
        ),
    )
    assert np.any(complete & support)
    assert np.any(complete & ~support)


@pytest.mark.parametrize("scale", [0.1, 0.3, 0.7, 1.0, 4.0])
@pytest.mark.parametrize("offset", [0.0, 100.1, 1_000_000.3])
def test_pixel_center_boundaries_are_independent_of_window_partition(
    scale: float, offset: float
) -> None:
    grid = _grid(
        32, 32, translate_x=offset, translate_y=offset, scale_x=scale, scale_y=scale
    )
    geometry = _rectangle(
        offset + 3.5 * scale,
        offset + 5.5 * scale,
        offset + 18.5 * scale,
        offset + 21.5 * scale,
    )
    complete = rasterize_training_geometry(
        geometry, grid, {"x": 0, "y": 0, "width": 32, "height": 32}
    )
    for y in range(0, 32, 7):
        for x in range(0, 32, 11):
            height, width = min(7, 32 - y), min(11, 32 - x)
            part = rasterize_training_geometry(
                geometry, grid, {"x": x, "y": y, "width": width, "height": height}
            )
            assert np.array_equal(part, complete[y : y + height, x : x + width])


def test_declared_per_roi_instance_limit_avoids_pairwise_overlap_work() -> None:
    object_count = 10_000
    summary = inspect_training_roi_labels(
        roi_geometry=_rectangle(0, 0, object_count, 1),
        objects=[
            _object(_rectangle(index, 0, index + 1, 1), "Cell")
            for index in range(object_count)
        ],
        model_output=_model_output("instance_segmentation", ["Cell"]),
        raster_transform=_grid(object_count, 1),
        physical_slide_bounds=(0, 0, object_count, 1),
    )

    assert summary.target_object_counts == {"Cell": object_count}


def test_disconnected_roi_prunes_unsupported_wsi_scale_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raster_calls = 0
    rasterize_shape = training_label_rasterizer._rasterize_shape

    def counted_rasterize_shape(*args, **kwargs):
        nonlocal raster_calls
        raster_calls += 1
        if raster_calls > 100:
            pytest.fail("rasterization scanned unsupported WSI-scale space")
        return rasterize_shape(*args, **kwargs)

    monkeypatch.setattr(
        training_label_rasterizer,
        "_rasterize_shape",
        counted_rasterize_shape,
    )
    roi = _multipolygon(
        _rectangle(0, 0, 5, 5),
        _rectangle(99_995, 99_995, 100_000, 100_000),
    )
    common_outside_support = _rectangle(100, 100, 99_900, 99_900)
    summary = inspect_training_roi_labels(
        roi_geometry=roi,
        objects=[
            _object(
                _multipolygon(
                    _rectangle(1, 1, 2, 2),
                    common_outside_support,
                ),
                "Cell",
            ),
            _object(
                _multipolygon(
                    common_outside_support,
                    _rectangle(99_998, 99_998, 99_999, 99_999),
                ),
                "Cell",
            ),
        ],
        model_output=_model_output("instance_segmentation", ["Cell"]),
        raster_transform=_grid(100_000, 100_000),
        physical_slide_bounds=(0, 0, 100_000, 100_000),
    )

    assert summary.target_object_counts == {"Cell": 2}
    assert raster_calls < 100


def test_roi_without_one_supervised_real_pixel_fails() -> None:
    with pytest.raises(ValueError, match="no supervised real pixels"):
        inspect_training_roi_labels(
            roi_geometry=_rectangle(0.1, 0.1, 0.4, 0.4),
            objects=[],
            model_output=_model_output("instance_segmentation", ["Cell"]),
            raster_transform=_grid(1, 1),
            physical_slide_bounds=(0, 0, 1, 1),
        )


def _object(geometry: dict, path_class: str) -> dict:
    return {
        "type": "Feature",
        "geometry": canonical_geometry(geometry),
        "properties": {"path_class": path_class},
    }


def _grid(
    width: int,
    height: int,
    *,
    translate_x: float = 0,
    translate_y: float = 0,
    scale_x: float = 1,
    scale_y: float = 1,
) -> dict:
    return {
        "raster": {
            "width": width,
            "height": height,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 1,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "translate_x": translate_x,
            "translate_y": translate_y,
        },
    }


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }


def _multipolygon(*polygons: dict) -> dict:
    return {
        "type": "MultiPolygon",
        "coordinates": [polygon["coordinates"] for polygon in polygons],
    }
