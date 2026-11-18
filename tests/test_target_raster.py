"""Scientific golden tests for exact source-view target-grid sampling."""

import base64
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from histoforge.canonical_json import canonical_digest, sha256_bytes
from histoforge.cloud.rasterization.rasterizer import (
    rasterize_srgb_uint8,
    required_source_pixel_rect,
)
from histoforge.cloud.sources.views import (
    derive_derived_view_transform,
    derive_original_view_transform,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "target-raster-v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _source_pixels(fixture):
    source = fixture["sourceLevel"]
    raw = base64.b64decode(source["rgbBase64"], validate=True)
    assert sha256_bytes(raw) == source["rgbSha256"]
    dimensions = source["dimensions"]
    return np.frombuffer(raw, dtype=np.uint8).reshape(
        dimensions["height"], dimensions["width"], 3
    )


def _case_inputs(fixture, case):
    source = _source_pixels(fixture)
    rectangle = case["sourcePixelRect"]
    crop = source[
        rectangle["y"] : rectangle["y"] + rectangle["height"],
        rectangle["x"] : rectangle["x"] + rectangle["width"],
    ].copy()
    original_transform = derive_original_view_transform(
        fixture["levelZeroPixelsToSlide"],
        fixture["selectedLevelPixelsToLevelZero"],
        fixture["sourceLevel"]["dimensions"],
        rectangle,
    )
    derived_transform = derive_derived_view_transform(
        case["expectedViewPixelsToSlide"],
        {"width": rectangle["width"], "height": rectangle["height"]},
        {"x": 0, "y": 0, "width": rectangle["width"], "height": rectangle["height"]},
    )
    dimensions = fixture["sourceLevel"]["dimensions"]
    physical_bounds = {
        "min_x": -rectangle["x"],
        "min_y": -rectangle["y"],
        "max_x": dimensions["width"] - rectangle["x"],
        "max_y": dimensions["height"] - rectangle["y"],
    }
    return crop, original_transform, derived_transform, physical_bounds


def test_shared_fixture_freezes_view_geometry_and_exact_target_pixels():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1
    selected_dimensions = fixture["sourceLevel"]["dimensions"]
    level_zero_dimensions = fixture["levelZeroDimensions"]
    selected_transform = fixture["selectedLevelPixelsToLevelZero"]
    assert selected_transform["scale_x"] != (
        level_zero_dimensions["width"] / selected_dimensions["width"]
    )
    assert selected_transform["scale_y"] != (
        level_zero_dimensions["height"] / selected_dimensions["height"]
    )
    digest_preimage = copy.deepcopy(fixture)
    digest = digest_preimage.pop("rasterizationContractDigest")
    assert canonical_digest(digest_preimage) == digest

    for case in fixture["cases"]:
        crop, original, derived, physical_bounds = _case_inputs(fixture, case)
        assert original == case["expectedViewPixelsToSlide"], case["name"]
        assert derived == original, case["name"]
        _assert_target_grid(case, fixture["sourceCalibration"])

        original_pixels = rasterize_srgb_uint8(
            crop,
            original,
            case["targetTransform"],
            physical_bounds,
        )
        derived_pixels = rasterize_srgb_uint8(
            crop,
            derived,
            case["targetTransform"],
            physical_bounds,
        )
        expected = base64.b64decode(case["expectedRgbBase64"], validate=True)
        assert original_pixels.tobytes() == expected, case["name"]
        assert derived_pixels.tobytes() == expected, case["name"]
        assert sha256_bytes(expected) == case["expectedRgbSha256"], case["name"]


def test_absolute_output_rects_are_partition_invariant():
    fixture = _fixture()
    for case in fixture["cases"]:
        crop, view_transform, _, physical_bounds = _case_inputs(fixture, case)
        raster = case["targetTransform"]["raster"]
        assembled = np.empty((raster["height"], raster["width"], 3), dtype=np.uint8)
        covered = np.zeros((raster["height"], raster["width"]), dtype=bool)
        for rectangle in case["outputPartitions"]:
            result = rasterize_srgb_uint8(
                crop,
                view_transform,
                case["targetTransform"],
                physical_bounds,
                rectangle,
            )
            y0, x0 = rectangle["y"], rectangle["x"]
            y1, x1 = y0 + rectangle["height"], x0 + rectangle["width"]
            assert not covered[y0:y1, x0:x1].any()
            assembled[y0:y1, x0:x1] = result
            covered[y0:y1, x0:x1] = True
        assert covered.all(), case["name"]
        assert assembled.tobytes() == base64.b64decode(case["expectedRgbBase64"])


def test_decoded_crop_keeps_the_fixed_source_view_float32_origin():
    source = np.array([[[17, 29, 43]]], dtype=np.uint8)
    fixed_view = {
        "scale_x": 8.16711260996145,
        "scale_y": 1.0,
        "translate_x": 698110.5514846945,
        "translate_y": 0.0,
    }
    decoded_x = 496_935
    target_center_x = 4_756_638.7398722
    target = {
        "raster": {
            "width": 1,
            "height": 1,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 1.0,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "translate_x": target_center_x - 0.5,
            "translate_y": 0.0,
        },
    }
    physical_bounds = {"min_x": 0, "min_y": 0, "max_x": 1_000_000, "max_y": 1}

    actual = rasterize_srgb_uint8(
        source,
        fixed_view,
        target,
        physical_bounds,
        decoded_rect_in_view={"x": decoded_x, "y": 0, "width": 1, "height": 1},
    )

    assert np.array_equal(actual, source)

    # Reconstructing a new view origin for the same crop moves the canonical
    # float32 coordinate just above zero and falsely requires a second pixel.
    recomposed_view = {
        **fixed_view,
        "translate_x": fixed_view["translate_x"] + decoded_x * fixed_view["scale_x"],
    }
    with pytest.raises(ValueError, match="omits an in-image"):
        rasterize_srgb_uint8(
            source,
            recomposed_view,
            target,
            {
                "min_x": -decoded_x,
                "min_y": 0,
                "max_x": 1_000_000 - decoded_x,
                "max_y": 1,
            },
        )


def test_fixture_crosses_declared_source_tiles_and_exercises_white_border():
    fixture = _fixture()
    interior, edge = fixture["cases"]
    rectangle = interior["sourcePixelRect"]
    source = fixture["sourceLevel"]
    assert (
        rectangle["x"] // source["tileWidth"]
        != (rectangle["x"] + rectangle["width"] - 1) // source["tileWidth"]
    )
    assert (
        rectangle["y"] // source["tileHeight"]
        != (rectangle["y"] + rectangle["height"] - 1) // source["tileHeight"]
    )

    crop, transform, _, bounds = _case_inputs(fixture, edge)
    pixels = rasterize_srgb_uint8(crop, transform, edge["targetTransform"], bounds)
    assert np.array_equal(pixels[-1, -1], np.array([255, 255, 255], dtype=np.uint8))
    assert np.any(pixels != 255)


def test_missing_in_image_source_pixels_and_invalid_shapes_fail_closed():
    fixture = _fixture()
    case = fixture["cases"][0]
    crop, transform, _, bounds = _case_inputs(fixture, case)

    with pytest.raises(ValueError, match="omits an in-image"):
        rasterize_srgb_uint8(crop[:, :-1], transform, case["targetTransform"], bounds)
    with pytest.raises(ValueError, match="uint8 shape"):
        rasterize_srgb_uint8(
            crop.astype(np.float32), transform, case["targetTransform"], bounds
        )
    with pytest.raises(ValueError, match="outside the target"):
        rasterize_srgb_uint8(
            crop,
            transform,
            case["targetTransform"],
            bounds,
            {"x": 0, "y": 0, "width": 8, "height": 36},
        )

    outside = dict(case["sourcePixelRect"])
    outside["width"] = fixture["sourceLevel"]["dimensions"]["width"]
    with pytest.raises(ValueError, match="outside its selected level"):
        derive_original_view_transform(
            fixture["levelZeroPixelsToSlide"],
            fixture["selectedLevelPixelsToLevelZero"],
            fixture["sourceLevel"]["dimensions"],
            outside,
        )


def test_float_map_neighbor_support_fails_closed_on_both_sides_of_an_integer():
    source = np.zeros((1, 2, 3), dtype=np.uint8)
    view_transform = {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }
    bounds = {"min_x": 0, "min_y": 0, "max_x": 2, "max_y": 1}

    def target(source_x):
        return {
            "raster": {
                "width": 1,
                "height": 1,
                "plane": {"z": 0, "t": 0},
                "target_mpp": 1.0,
                "resampling": "BILINEAR",
                "pixel_format": "SRGB_UINT8",
            },
            "raster_to_slide": {
                "scale_x": 1.0,
                "scale_y": 1.0,
                "translate_x": source_x,
                "translate_y": 0.0,
            },
        }

    # At an exact source center, the second pixel has zero weight.
    exact = rasterize_srgb_uint8(source[:, :1], view_transform, target(0.0), bounds)
    assert np.array_equal(exact, np.zeros((1, 1, 3), dtype=np.uint8))

    # OpenCV 5 retains these small float-map contributions. Omitting the
    # in-image second pixel must fail instead of blending the white border.
    for source_x in (0.01, 0.99):
        with pytest.raises(ValueError, match="omits an in-image"):
            rasterize_srgb_uint8(
                source[:, :1], view_transform, target(source_x), bounds
            )


def test_minimal_view_uses_view_local_float32_coordinates():
    rectangle = {"x": 650, "y": 0, "width": 5, "height": 1}
    view_transform = {
        "scale_x": 2.0,
        "scale_y": 1.0,
        "translate_x": -63.15894033791618 + rectangle["x"] * 2.0,
        "translate_y": 0.0,
    }
    target = {
        "raster": {
            "width": 2,
            "height": 1,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 1.0,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": {
            "scale_x": 6.0,
            "scale_y": 1.0,
            "translate_x": 1236.8410532038954,
            "translate_y": 0.0,
        },
    }

    assert (
        required_source_pixel_rect(
            view_transform,
            target,
            rectangle,
            {"width": 1_000, "height": 1},
        )
        == rectangle
    )


def test_view_transforms_compose_nonzero_level_and_subview_offsets():
    original = derive_original_view_transform(
        {
            "scale_x": 0.75,
            "scale_y": 1.25,
            "translate_x": -1.5,
            "translate_y": 2.25,
        },
        {
            "scale_x": 2.0,
            "scale_y": 3.0,
            "translate_x": 0.25,
            "translate_y": -0.5,
        },
        {"width": 12, "height": 10},
        {"x": 2, "y": 1, "width": 4, "height": 3},
    )
    assert original == {
        "scale_x": 1.5,
        "scale_y": 3.75,
        "translate_x": 1.6875,
        "translate_y": 5.375,
    }

    derived = derive_derived_view_transform(
        {
            "scale_x": 1.5,
            "scale_y": 3.75,
            "translate_x": 1.5,
            "translate_y": 6.0,
        },
        {"width": 8, "height": 8},
        {"x": 1, "y": 2, "width": 4, "height": 3},
    )
    assert derived == {
        "scale_x": 1.5,
        "scale_y": 3.75,
        "translate_x": 3.0,
        "translate_y": 13.5,
    }


def test_view_transforms_preserve_the_frozen_binary64_operation_order():
    level_zero = {
        "scale_x": 0.9402327520733241,
        "scale_y": 1.0,
        "translate_x": 9.120685437784989,
        "translate_y": 0.0,
    }
    selected_level = {
        "scale_x": 0.659858540495406,
        "scale_y": 1.0,
        "translate_x": 8.956549741186986,
        "translate_y": 0.0,
    }
    rectangle = {"x": 2770, "y": 0, "width": 1, "height": 1}
    original = derive_original_view_transform(
        level_zero,
        selected_level,
        {"width": 4000, "height": 1},
        rectangle,
    )
    compose_first = (
        level_zero["translate_x"]
        + selected_level["translate_x"] * level_zero["scale_x"]
        + rectangle["x"] * (level_zero["scale_x"] * selected_level["scale_x"])
    )

    assert original["translate_x"] == 1736.1070207301818
    assert original["translate_x"] != compose_first

    base_translate = -731.2715117751975
    scale = 24.532112507341665
    artifact_offset = 33086
    view_offset = 33432
    derived = derive_derived_view_transform(
        {
            "scale_x": scale,
            "scale_y": 1.0,
            "translate_x": base_translate + artifact_offset * scale,
            "translate_y": 0.0,
        },
        {"width": 70000, "height": 1},
        {"x": view_offset, "y": 0, "width": 1, "height": 1},
    )
    combine_offsets_first = base_translate + (artifact_offset + view_offset) * scale

    assert derived["translate_x"] == 1631095.7882515779
    assert derived["translate_x"] != combine_offsets_first


def _assert_target_grid(case, calibration):
    frame = case["frameBoundsSlide"]
    transform = case["targetTransform"]
    raster = transform["raster"]
    mapping = transform["raster_to_slide"]
    step_x = raster["target_mpp"] / calibration["mpp_x"]
    step_y = raster["target_mpp"] / calibration["mpp_y"]
    assert mapping["scale_x"] == step_x
    assert mapping["scale_y"] == step_y
    assert mapping["translate_x"] == frame["min_x"]
    assert mapping["translate_y"] == frame["min_y"]
    assert raster["width"] == max(
        1, math.ceil((frame["max_x"] - frame["min_x"]) / step_x)
    )
    assert raster["height"] == max(
        1, math.ceil((frame["max_y"] - frame["min_y"]) / step_y)
    )
