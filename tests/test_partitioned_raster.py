"""Bounded source reads preserve exact pixels, including source-free border blocks."""

from contextlib import contextmanager, nullcontext

import numpy as np
import pytest
from histoforge.cloud.rasterization.partitioned import PartitionedRaster
from histoforge.cloud.rasterization.rasterizer import (
    InsufficientSourceContext,
    rasterize_srgb_uint8,
)


@pytest.mark.parametrize("decode_cap", [2, 4, 9])
def test_partitions_with_only_white_border_need_no_source_read(decode_cap):
    source = np.arange(3 * 3 * 3, dtype=np.uint8).reshape(3, 3, 3)
    target = _target(9, 8, -3.0, -2.0)
    source_rect = {"x": 0, "y": 0, "width": 3, "height": 3}
    reads = []

    def read(rectangle):
        reads.append(rectangle)
        assert rectangle["width"] * rectangle["height"] <= decode_cap
        return source[
            rectangle["y"] : rectangle["y"] + rectangle["height"],
            rectangle["x"] : rectangle["x"] + rectangle["width"],
        ].copy()

    raster = PartitionedRaster(
        target,
        _identity(),
        source_rect,
        {"width": 3, "height": 3},
        decode_cap,
        lambda: nullcontext(read),
        (0.0, 0.0, 3.0, 3.0),
    )
    expected = rasterize_srgb_uint8(
        source,
        _identity(),
        target,
        {"min_x": 0, "min_y": 0, "max_x": 3, "max_y": 3},
    )
    actual = raster.read({"x": 0, "y": 0, "width": 9, "height": 8})
    assert np.array_equal(actual.pixels, expected)
    expected_support = np.zeros((8, 9), bool)
    expected_support[2:5, 3:6] = True
    np.testing.assert_array_equal(actual.real_pixel_mask, expected_support)
    assert reads

    reads.clear()
    border = raster.read({"x": 0, "y": 0, "width": 2, "height": 2})
    assert np.all(border.pixels == 255)
    assert not border.real_pixel_mask.any()
    assert reads == []


def test_missing_in_image_source_support_is_not_treated_as_white_border():
    def unexpected_read(_rectangle):
        pytest.fail("missing required source pixels must fail before IO")

    raster = PartitionedRaster(
        _target(3, 3, 0, 0),
        _identity(),
        {"x": 0, "y": 0, "width": 2, "height": 3},
        {"width": 3, "height": 3},
        9,
        lambda: nullcontext(unexpected_read),
        (0.0, 0.0, 3.0, 3.0),
    )
    with pytest.raises(InsufficientSourceContext, match="omits required"):
        raster.read({"x": 0, "y": 0, "width": 3, "height": 3})


@pytest.mark.parametrize("fail", [False, True])
def test_reader_scope_is_lazy_per_window_and_closes_after_success_or_failure(fail):
    events = []
    reads = []

    @contextmanager
    def open_pixels():
        events.append("open")

        def read(rectangle):
            reads.append(rectangle)
            assert events[-1] == "open"
            if fail and len(reads) == 2:
                raise OSError("decode failed")
            return np.zeros((rectangle["height"], rectangle["width"], 3), np.uint8)

        try:
            yield read
        finally:
            events.append("close")

    raster = PartitionedRaster(
        _target(9, 8, -3, -2),
        _identity(),
        {"x": 0, "y": 0, "width": 3, "height": 3},
        {"width": 3, "height": 3},
        4,
        open_pixels,
        (0.0, 0.0, 3.0, 3.0),
    )
    raster.read({"x": 0, "y": 0, "width": 2, "height": 2})
    assert events == []
    for _ in range(2):
        reads.clear()
        if fail:
            with pytest.raises(OSError, match="decode failed"):
                raster.read({"x": 0, "y": 0, "width": 9, "height": 8})
        else:
            result = raster.read({"x": 0, "y": 0, "width": 9, "height": 8})
            assert result.real_pixel_mask.sum() == 9
        assert len(reads) >= 2
    assert events == ["open", "close", "open", "close"]


@pytest.mark.parametrize("fail", [False, True])
def test_multiple_target_blocks_share_one_reader_and_close_on_failure(fail):
    source = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    events = []
    read_count = 0

    @contextmanager
    def open_pixels():
        nonlocal read_count
        events.append("open")

        def read(rectangle):
            nonlocal read_count
            read_count += 1
            if fail and read_count == 2:
                raise OSError("second block failed")
            return source[
                rectangle["y"] : rectangle["y"] + rectangle["height"],
                rectangle["x"] : rectangle["x"] + rectangle["width"],
            ].copy()

        try:
            yield read
        finally:
            events.append("close")

    raster = PartitionedRaster(
        _target(9, 8, -3, -2),
        _identity(),
        {"x": 0, "y": 0, "width": 3, "height": 3},
        {"width": 3, "height": 3},
        9,
        open_pixels,
        (0.0, 0.0, 3.0, 3.0),
    )

    def read_rows(read):
        border = read({"x": 0, "y": 0, "width": 2, "height": 2})
        assert not border.real_pixel_mask.any()
        assert events == []
        rows = []
        for index in range(3):
            image = read({"x": 3, "y": 2 + index, "width": 3, "height": 1})
            assert image.real_pixel_mask.all()
            rows.append(image.pixels)
            assert events == ["open"]
        np.testing.assert_array_equal(np.concatenate(rows), source)

    if fail:
        with pytest.raises(OSError, match="second block failed"):
            with raster.open_reader() as read:
                read_rows(read)
    else:
        with raster.open_reader() as read:
            read_rows(read)
    assert events == ["open", "close"]


def _identity():
    return {"scale_x": 1.0, "scale_y": 1.0, "translate_x": 0.0, "translate_y": 0.0}


def _target(width, height, x, y):
    return {
        "raster": {
            "width": width,
            "height": height,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 1.0,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": {**_identity(), "translate_x": x, "translate_y": y},
    }
