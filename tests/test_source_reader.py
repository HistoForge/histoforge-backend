"""Bounded source-container reads feeding the pinned target rasterizer."""

import copy
import json
import struct
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import tifffile
from histoforge.canonical_json import canonical_digest, sha256_bytes
from histoforge.cloud.rasterization.inputs import build_raster_frame
from histoforge.cloud.rasterization.rasterizer import rasterize_srgb_uint8
from histoforge.cloud.sources.reader import SourceReader, _qupath_level_downsamples
from histoforge.cloud.sources.views import (
    derive_derived_view_transform,
    derive_original_view_transform,
)

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def test_png_and_derived_ome_bigtiff_decode_exact_cross_tile_windows(tmp_path):
    pixels = _pixels(530, 520)
    rectangle = {"x": 500, "y": 500, "width": 20, "height": 25}
    expected = pixels[500:525, 500:520]

    png = tmp_path / "derived.png"
    _write_png(png, pixels)
    with SourceReader(png, _derived_start(png, pixels, "PNG"), _limits()) as reader:
        assert reader.validated_levels == [_level(0, 0, 520, 530)]
        assert np.array_equal(reader.read_rgb_view(0, 0, rectangle), expected)

    ome_tiff = tmp_path / "derived.ome.tif"
    _write_derived_tiff(ome_tiff, pixels)
    with SourceReader(
        ome_tiff,
        _derived_start(ome_tiff, pixels, "TIFF_FAMILY"),
        _limits(),
    ) as reader:
        assert reader.validated_levels == [_level(0, 0, 520, 530)]
        assert np.array_equal(reader.read_rgb_view(0, 0, rectangle), expected)


def test_openslide_original_matches_qupath_levels_and_exact_rgb_windows():
    fixture = _openslide_fixture()
    path = _openslide_path()
    assert sha256_bytes(path.read_bytes()) == fixture["source_file"]["sha256"]

    with SourceReader(path, _openslide_start(path), _limits()) as reader:
        assert reader.validated_levels == [
            _level(0, value["resolution_level"], value["width"], value["height"])
            for value in fixture["levels"]
        ]
        for level in fixture["levels"]:
            mapping = reader.selected_level_pixels_to_level_zero(
                0, level["resolution_level"]
            )
            assert mapping == {
                "scale_x": level["qupath_downsample"],
                "scale_y": level["qupath_downsample"],
                "translate_x": 0.0,
                "translate_y": 0.0,
            }
        for case in fixture["window_cases"]:
            pixels = reader.read_rgb_view(
                0, case["resolution_level"], case["pixel_rect"]
            )
            assert sha256_bytes(pixels.tobytes()) == case["rgb_sha256"]


def test_irregular_pyramid_keeps_qupath_native_level_mapping():
    assert _qupath_level_downsamples(((100, 80), (67, 53))) == (
        1.0,
        (100 / 67 + 80 / 53) / 2,
    )


def test_original_and_derived_lossless_paths_produce_the_same_target(tmp_path):
    fixture = _openslide_fixture()
    target_case = fixture["target_case"]
    window = next(
        value
        for value in fixture["window_cases"]
        if value["name"] == target_case["source_window"]
    )
    original = _openslide_path()
    original_start = _openslide_start(original)
    rectangle = window["pixel_rect"]

    with SourceReader(original, original_start, _limits()) as reader:
        level_mapping = reader.selected_level_pixels_to_level_zero(0, 1)
        original_pixels = reader.read_rgb_view(0, 1, rectangle)

    original_transform = derive_original_view_transform(
        original_start["origin"]["level_zero_pixels_to_slide"],
        level_mapping,
        {"width": 257, "height": 193},
        rectangle,
    )
    assert original_transform == target_case["source_view_to_slide"]

    derived_png = tmp_path / "selected.png"
    _write_png(derived_png, original_pixels)
    artifact_transform = original_transform
    derived_start = _derived_start(
        derived_png,
        original_pixels,
        "PNG",
        artifact_transform,
        source=original_start["origin"]["source"],
        source_resolution_level=1,
    )
    derived_start["origin"]["source_rect_at_level"] = dict(rectangle)
    derived_rectangle = {
        "x": 0,
        "y": 0,
        "width": rectangle["width"],
        "height": rectangle["height"],
    }
    with SourceReader(derived_png, derived_start, _limits()) as reader:
        derived_pixels = reader.read_rgb_view(0, 0, derived_rectangle)
    derived_transform = derive_derived_view_transform(
        artifact_transform,
        {"width": rectangle["width"], "height": rectangle["height"]},
        derived_rectangle,
    )

    assert np.array_equal(original_pixels, derived_pixels)
    assert original_transform == derived_transform
    assert derived_start["origin"]["source"] == original_start["origin"]["source"]
    assert derived_start["origin"]["source_resolution_level"] == 1
    target = target_case["target_grid"]
    build_raster_frame(
        {
            "source_revision": {
                "image_data_revision": {
                    "image_id": "source-reader-equivalence",
                    "image_source_digest": canonical_digest(
                        original_start["origin"]["source"]
                    ),
                    "image_data_content_digest": "a" * 64,
                },
                "pixel_source": original_start["origin"]["source"],
            },
            "region": {
                "type": "roi",
                "roi_id": "11111111-1111-4111-8111-111111111111",
            },
            "transform": target,
            "inclusion_mask_raster": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [80, 0], [80, 70], [0, 70], [0, 0]]],
            },
        }
    )
    original_target = rasterize_srgb_uint8(
        original_pixels,
        original_transform,
        target,
        target_case["original_physical_bounds_in_view"],
    )
    derived_target = rasterize_srgb_uint8(
        derived_pixels,
        derived_transform,
        target,
        target_case["derived_physical_bounds_in_view"],
    )
    assert np.array_equal(original_target, derived_target)
    assert sha256_bytes(original_pixels.tobytes()) == window["rgb_sha256"]
    assert sha256_bytes(original_target.tobytes()) == target_case["target_rgb_sha256"]


def test_tiff_window_read_does_not_use_complete_page_decode(tmp_path):
    pixels = _pixels(530, 520)
    path = tmp_path / "derived.ome.tif"
    _write_derived_tiff(path, pixels)
    start = _derived_start(path, pixels, "TIFF_FAMILY")

    with SourceReader(path, start, _limits()) as reader:
        with patch.object(
            tifffile.TiffPage,
            "asarray",
            side_effect=AssertionError("complete page decode is forbidden"),
        ):
            actual = reader.read_rgb_view(
                0, 0, {"x": 510, "y": 510, "width": 10, "height": 15}
            )
    assert np.array_equal(actual, pixels[510:525, 510:520])


def test_reader_rejects_wrong_required_shape_and_original_identity(tmp_path):
    pixels = _pixels(32, 48)
    path = tmp_path / "derived.ome.tif"
    _write_derived_tiff(path, pixels)
    wrong_shape = _derived_start(path, pixels, "TIFF_FAMILY")
    wrong_shape["required_levels"][0]["width"] += 1
    wrong_shape["origin"]["source_rect_at_level"]["width"] += 1
    with pytest.raises(ValueError, match="required level dimensions"):
        SourceReader(path, wrong_shape, _limits())

    wrong_adapter = _openslide_start(_openslide_path())
    wrong_adapter["origin"]["source"]["source_adapter_id"] = "qupath.bioformats.local"
    with pytest.raises(ValueError, match="reader_settings"):
        SourceReader(_openslide_path(), wrong_adapter, _limits())

    multiplane_claim = _openslide_start(_openslide_path())
    multiplane_claim["origin"]["source"]["z_slices"] = 2
    with pytest.raises(ValueError, match="one z=0, t=0 plane"):
        SourceReader(_openslide_path(), multiplane_claim, _limits())


def test_png_dimensions_fail_before_complete_decode(tmp_path):
    width = 500_001
    height = 2
    header = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        + b"\x00\x00\x00\x00"
    )
    path = tmp_path / "oversized.png"
    path.write_bytes(header)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    start = _derived_start(path, pixels, "PNG")
    with patch("histoforge.cloud.sources.reader.cv2.imdecode") as decode:
        with pytest.raises(ValueError, match="complete-decode limits"):
            SourceReader(path, start, _limits())
    decode.assert_not_called()


def test_tiff_metadata_caps_fail_before_window_decode():
    path = _openslide_path()
    start = _openslide_start(path)

    level_cap = _limits()
    level_cap["max_pyramid_levels_per_series"] = 2
    with patch(
        "histoforge.cloud.sources.reader._read_premultiplied_openslide_rgb"
    ) as read:
        with pytest.raises(ValueError, match="pyramid level count"):
            SourceReader(path, start, level_cap)
    read.assert_not_called()

    dimension_cap = _limits()
    dimension_cap["max_level_width_px"] = 512
    with patch(
        "histoforge.cloud.sources.reader._read_premultiplied_openslide_rgb"
    ) as read:
        with pytest.raises(ValueError, match="level dimensions"):
            SourceReader(path, start, dimension_cap)
    read.assert_not_called()

    tile_cap = _limits()
    tile_cap["max_complete_decode_pixels"] = 128 * 128 - 1
    with patch(
        "histoforge.cloud.sources.reader._read_premultiplied_openslide_rgb"
    ) as read:
        with pytest.raises(ValueError, match="tile dimensions"):
            SourceReader(path, start, tile_cap)
    read.assert_not_called()


def test_source_view_pixel_cap_fails_before_dense_window_allocation():
    path = _openslide_path()
    limits = _limits()
    limits["max_complete_decode_pixels"] = 128 * 128

    with SourceReader(path, _openslide_start(path), limits) as reader:
        with patch(
            "histoforge.cloud.sources.reader._read_premultiplied_openslide_rgb"
        ) as read:
            with pytest.raises(ValueError, match="source pixel_rect.*complete-decode"):
                reader.read_rgb_view(
                    0,
                    0,
                    {"x": 0, "y": 0, "width": 129, "height": 128},
                )
    read.assert_not_called()


def test_reader_rejects_tiff_orientation_outside_scale_translate_contract(tmp_path):
    pixels = _pixels(32, 48)
    path = tmp_path / "rotated.tif"
    tifffile.imwrite(
        path,
        pixels,
        bigtiff=True,
        ome=True,
        photometric="rgb",
        tile=(512, 512),
        compression="zlib",
        metadata={"axes": "YXS"},
        extratags=[(274, "H", 1, 3, False)],
    )
    with pytest.raises(ValueError, match="top-left orientation"):
        SourceReader(path, _derived_start(path, pixels, "TIFF_FAMILY"), _limits())


def _write_png(path, pixels):
    success, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    assert success
    path.write_bytes(encoded.tobytes())


def _write_derived_tiff(path, pixels):
    tifffile.imwrite(
        path,
        pixels,
        bigtiff=True,
        ome=True,
        photometric="rgb",
        tile=(512, 512),
        compression="zlib",
        metadata={"axes": "YXS"},
    )


def _derived_start(
    path,
    pixels,
    container,
    transform=None,
    *,
    source=None,
    source_resolution_level=None,
):
    start = copy.deepcopy(_start_cases()["derived-region-s3-etag"])
    start["schema_version"] = 2
    height, width, _ = pixels.shape
    start["container"] = container
    start["total_bytes"] = path.stat().st_size
    start["origin"]["source_rect_at_level"].update(
        {"x": 0, "y": 0, "width": width, "height": height}
    )
    start["origin"]["artifact_pixels_to_slide"] = transform or {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }
    if source is not None:
        start["origin"]["source"] = copy.deepcopy(source)
    if source_resolution_level is not None:
        start["origin"]["source_resolution_level"] = source_resolution_level
    start["required_levels"] = [_level(0, 0, width, height)]
    return start


def _openslide_start(path):
    start = copy.deepcopy(_start_cases()["original-local-multilevel-normalizes-order"])
    start["schema_version"] = 2
    start["container"] = "SLIDE_FILE"
    fixture = _openslide_fixture()
    start["origin"]["entry_path"] = path.name
    start["origin"]["reader_settings"] = {
        "levels": [
            {
                "resolution_level": level["resolution_level"],
                "native_resolution_index": level["resolution_level"],
                "pixels_to_level_zero": {
                    "scale_x": level["qupath_downsample"],
                    "scale_y": level["qupath_downsample"],
                    "translate_x": 0.0,
                    "translate_y": 0.0,
                },
            }
            for level in fixture["levels"]
        ],
        "crop_x": 0,
        "crop_y": 0,
        "background_rgb": [0, 0, 0],
        "tile_width": 128,
        "tile_height": 128,
    }
    start["total_bytes"] = path.stat().st_size
    start["origin"]["pyramid_series_index"] = 0
    source = start["origin"]["source"]
    source["source_adapter_id"] = fixture["source_file"]["source_adapter_id"]
    source["width"] = fixture["levels"][0]["width"]
    source["height"] = fixture["levels"][0]["height"]
    source["calibration"] = {
        "mpp_x": fixture["source_file"]["mpp_x"],
        "mpp_y": fixture["source_file"]["mpp_y"],
    }
    source["content_version"]["size_bytes"] = path.stat().st_size
    start["origin"]["level_zero_pixels_to_slide"] = {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }
    start["required_levels"] = [
        _level(0, value["resolution_level"], value["width"], value["height"])
        for value in fixture["levels"]
    ]
    return start


def _openslide_path():
    return Path(__file__).parent / "fixtures" / "openslide" / "generic-tiff-v1.tif"


def _openslide_fixture():
    return json.loads(
        (FIXTURES / "openslide-original-v1.json").read_text(encoding="utf-8")
    )


def _limits():
    fixture = json.loads(
        (FIXTURES / "gateway-capabilities-v1.json").read_text(encoding="utf-8")
    )
    return copy.deepcopy(
        fixture["capabilityCases"][0]["value"]["source_artifact_limits"]
    )


def _start_cases():
    fixture = json.loads(
        (FIXTURES / "source-artifacts-v2.json").read_text(encoding="utf-8")
    )
    return {case["name"]: case["value"] for case in fixture["startCases"]}


def _pixels(height, width):
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        (
            (x * 17 + y * 3) % 256,
            (x * 5 + y * 29) % 256,
            (x * 11 + y * 7) % 256,
        ),
        axis=2,
    ).astype(np.uint8)


def _level(series, resolution, width, height):
    return {
        "series_index": series,
        "resolution_level": resolution,
        "width": width,
        "height": height,
    }
