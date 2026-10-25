"""Source-footprint bounds agree with every actual float32 sampling position."""

import numpy as np
import pytest
from histoforge.cloud.rasterization.rasterizer import (
    InsufficientSourceContext,
    required_source_pixel_rect,
)


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [(-2.0, 7, (0, 3)), (-1.0, 3, (0, 2)), (1.0, 3, (1, 2))],
)
def test_sampling_support_includes_interior_pixels_when_endpoints_are_outside(
    offset, length, expected
):
    target = _target(length, offset, 1.0)
    actual = required_source_pixel_rect(
        _view(), target, _rect(0, 3), {"width": 3, "height": 1}
    )
    assert actual == _rect(*expected)


@pytest.mark.parametrize(("offset", "step"), [(-1.0, 4.0), (-2.0, 6.0)])
def test_sparse_samples_that_jump_over_the_image_have_no_footprint(offset, step):
    with pytest.raises(InsufficientSourceContext, match="no bilinear"):
        required_source_pixel_rect(
            _view(),
            _target(2, offset - (step - 1) / 2, step),
            _rect(0, 3),
            {"width": 3, "height": 1},
        )


def test_footprint_matches_exhaustive_float32_support_for_absolute_subrectangles():
    rng = np.random.default_rng(4081)
    for _ in range(250):
        extent = int(rng.integers(1, 65))
        view_origin = int(rng.integers(0, extent))
        view_scale = float(rng.choice([0.3, 1.0, 2.0, 7.5]))
        view_translate = float(rng.choice([0.0, 100.1, 1_000_000.3]))
        target_scale = float(rng.choice([0.1, 0.3, 1.0, 4.0, 25.0]))
        target_translate = view_translate + float(rng.uniform(-150, 150))
        length = int(rng.integers(1, 65))
        start = int(rng.integers(0, 10))
        view = _view(view_translate, view_scale)
        target = _target(length + start + 3, target_translate, target_scale)
        submitted = _rect(view_origin, extent - view_origin)
        output = _rect(start, length)

        positions = np.arange(start, start + length, dtype=np.float64)
        coordinates = (
            (target_translate + (positions + 0.5) * target_scale - view_translate)
            / view_scale
            - 0.5
        ).astype(np.float32)
        base = np.floor(coordinates).astype(np.int64)
        fraction = coordinates.astype(np.float64) - base
        neighbors = np.concatenate(
            [base + view_origin, (base + 1 + view_origin)[fraction > 0]]
        )
        neighbors = neighbors[(neighbors >= 0) & (neighbors < extent)]
        if neighbors.size == 0:
            with pytest.raises(InsufficientSourceContext, match="no bilinear"):
                required_source_pixel_rect(
                    view, target, submitted, {"width": extent, "height": 1}, output
                )
        else:
            actual = required_source_pixel_rect(
                view, target, submitted, {"width": extent, "height": 1}, output
            )
            assert actual == _rect(
                int(neighbors.min()), int(neighbors.max() - neighbors.min() + 1)
            )


def test_large_target_footprint_does_not_allocate_the_target_axis():
    actual = required_source_pixel_rect(
        _view(),
        _target(2_000_000_000, -10.0, 1.0),
        _rect(0, 3),
        {"width": 3, "height": 1},
    )
    assert actual == _rect(0, 3)


def test_large_source_offset_does_not_round_away_the_final_pixels():
    # Match SourceCaptureCodecTest in Java. Only view-local sampling is float32.
    extent = 33_554_435
    origin = extent - 8
    for start, length in [(0, 8), (6, 2)]:
        actual = required_source_pixel_rect(
            _view(origin),
            _target(8, origin, 1.0),
            _rect(origin, 8),
            {"width": extent, "height": 1},
            _rect(start, length),
        )
        assert actual == _rect(origin + start, length)


def _rect(x, width):
    return {"x": x, "y": 0, "width": width, "height": 1}


def _view(translate=0.0, scale=1.0):
    return {
        "scale_x": scale,
        "scale_y": 1.0,
        "translate_x": translate,
        "translate_y": 0.0,
    }


def _target(width, translate, scale):
    return {
        "raster": {
            "width": width,
            "height": 1,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 1.0,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": _view(translate, scale),
    }
