"""Deterministic target-grid sampling after source-view validation.

The source reader supplies one decoded RGB view and physical-image bounds in
that view's local integer pixel coordinates. This module owns only the pure
pixel-center mapping and the pinned OpenCV ``INTER_LINEAR`` sampling kernel;
source IO and model preprocessing remain separate owners.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from histoforge.cloud.rasterization.inputs import validate_raster_transform
from histoforge.cloud.sources.identity import (
    validate_int_rect,
    validate_scale_translate_2d,
)

_PHYSICAL_BOUNDS_FIELDS = {"min_x", "min_y", "max_x", "max_y"}
_DIMENSION_FIELDS = {"width", "height"}


@dataclass(frozen=True)
class RasterizedImage:
    """Raw HWC uint8 RGB and HW real-pixel support on the same target grid."""

    pixels: np.ndarray
    real_pixel_mask: np.ndarray


def real_pixel_mask(
    bounds: tuple[float, float, float, float],
    target: Mapping[str, Any],
    rectangle: Mapping[str, int],
) -> np.ndarray:
    """Select centers inside validated half-open physical slide bounds.

    Training and inference share this operation. Source pixel color, bilinear
    neighbor availability, ROI supervision, and IO partitions do not select
    real support. Callers supply the validated absolute target grid.
    """

    mapping = target["raster_to_slide"]
    x = (
        mapping["translate_x"]
        + (np.arange(rectangle["x"], rectangle["x"] + rectangle["width"]) + 0.5)
        * mapping["scale_x"]
    )
    y = (
        mapping["translate_y"]
        + (np.arange(rectangle["y"], rectangle["y"] + rectangle["height"]) + 0.5)
        * mapping["scale_y"]
    )
    return (
        (y[:, None] >= bounds[1])
        & (y[:, None] < bounds[3])
        & (x[None, :] >= bounds[0])
        & (x[None, :] < bounds[2])
    )


def require_real_pixel_coverage(
    physical_bounds: tuple[float, float, float, float],
    image_pixels_to_slide: Mapping[str, float],
    dimensions: Mapping[str, int],
    target: Mapping[str, Any],
) -> None:
    """Reject missing target centers that normalization would count as real.

    All inputs are already validated. The complete prepared image must cover
    every target center inside the physical slide, even when a pyramid level's
    dimensions were rounded down. Binary search uses the same half-open center
    mapping as real_pixel_mask without allocating a slide-sized mask. Bilinear
    neighbor coverage is checked separately by required_source_pixel_rect.
    """

    mapping = target["raster_to_slide"]
    covered_axes = []
    for index, (axis, extent) in enumerate((("x", "width"), ("y", "height"))):
        positions = range(target["raster"][extent])
        translate = mapping[f"translate_{axis}"]
        scale = mapping[f"scale_{axis}"]

        def center(position: int) -> float:
            return translate + (position + 0.5) * scale

        first = bisect_left(positions, physical_bounds[index], key=center)
        stop = bisect_left(positions, physical_bounds[index + 2], key=center)
        if first == stop:
            return  # No target pixel is real unless both axes have support.
        minimum = image_pixels_to_slide[f"translate_{axis}"]
        maximum = minimum + dimensions[extent] * image_pixels_to_slide[f"scale_{axis}"]
        covered_axes.append(minimum <= center(first) and center(stop - 1) < maximum)
    if not all(covered_axes):
        raise ValueError("source image omits real target pixels")


class InsufficientSourceContext(ValueError):
    """The decoded view omits source pixels required by the target grid."""


class EmptySourceFootprint(InsufficientSourceContext):
    """This target rectangle samples only the rasterizer's white image border."""


def required_source_pixel_rect(
    view_pixels_to_slide: Any,
    raster_transform: Any,
    submitted_pixel_rect: Any,
    selected_level_dimensions: Any,
    output_rect: Any | None = None,
) -> dict[str, int]:
    """Return the smallest selected-level rectangle needed by bilinear sampling.

    The calculation uses the same float32 source coordinates passed to OpenCV
    and includes only nonzero-weight neighbors that lie inside the physical
    selected level. The submitted rectangle supplies the local-view origin; it
    is not trusted as the answer.
    """

    view_transform = validate_scale_translate_2d(view_pixels_to_slide)
    target = validate_raster_transform(raster_transform)
    submitted = validate_int_rect(submitted_pixel_rect)
    dimensions = _validate_dimensions(selected_level_dimensions)
    if (
        submitted["x"] + submitted["width"] > dimensions["width"]
        or submitted["y"] + submitted["height"] > dimensions["height"]
    ):
        raise ValueError("source view lies outside its selected level")

    raster = target["raster"]
    rectangle = _validate_output_rect(output_rect, raster)
    required_x = _required_axis_bounds(
        range(rectangle["x"], rectangle["x"] + rectangle["width"]),
        target["raster_to_slide"]["translate_x"],
        target["raster_to_slide"]["scale_x"],
        view_transform["translate_x"],
        view_transform["scale_x"],
        submitted["x"],
        dimensions["width"],
    )
    required_y = _required_axis_bounds(
        range(rectangle["y"], rectangle["y"] + rectangle["height"]),
        target["raster_to_slide"]["translate_y"],
        target["raster_to_slide"]["scale_y"],
        view_transform["translate_y"],
        view_transform["scale_y"],
        submitted["y"],
        dimensions["height"],
    )
    if required_x is None or required_y is None:
        raise EmptySourceFootprint(
            "target raster has no bilinear sampling footprint inside the source"
        )
    return {
        "x": required_x[0],
        "y": required_y[0],
        "width": required_x[1] - required_x[0] + 1,
        "height": required_y[1] - required_y[0] + 1,
    }


def _required_axis_bounds(
    positions: range,
    target_translate: float,
    target_scale: float,
    view_translate: float,
    view_scale: float,
    view_offset: int,
    physical_extent: int,
) -> tuple[int, int] | None:
    """Find the first and last samples with nonzero in-image support.

    The float32 mapping is monotonic. Binary search keeps memory constant for
    slide-sized grids and finds interior support even when both ends are outside
    the image. The open interval excludes exact zero-weight border neighbors.
    """

    def coordinate(position: int) -> float:
        return float(
            _target_axis_in_view(
                np.float64(position),
                target_translate,
                target_scale,
                view_translate,
                view_scale,
            )
        )

    first = bisect_right(positions, -view_offset - 1, key=coordinate)
    after_last = bisect_left(positions, physical_extent - view_offset, key=coordinate)
    if first >= after_last:
        return None
    endpoints = np.asarray(
        [coordinate(positions[first]), coordinate(positions[after_last - 1])],
        dtype=np.float32,
    )
    required = _required_physical_axis_pixels(endpoints, view_offset, physical_extent)
    return int(required.min()), int(required.max())


def rasterize_srgb_uint8(
    source_rgb: np.ndarray,
    view_pixels_to_slide: Any,
    raster_transform: Any,
    physical_bounds_in_view: Any,
    output_rect: Any | None = None,
    *,
    decoded_rect_in_view: Any | None = None,
) -> np.ndarray:
    """Sample one absolute target-grid rectangle from a decoded source view.

    ``output_rect`` is expressed in the complete target raster.  Using those
    absolute coordinates makes whole-raster and partitioned calls identical.
    ``decoded_rect_in_view`` locates a bounded decoded crop inside the one fixed
    submitted SourceView coordinate system.  The float32 sampling map is always
    derived in that fixed system before the integer crop offset is removed, so
    source IO partitioning cannot change a target pixel.  Missing decoded pixels
    that still lie inside ``physical_bounds_in_view`` are rejected; only true
    out-of-image bilinear contributions become white.
    """

    source = _validate_source_pixels(source_rgb)
    view_transform = validate_scale_translate_2d(view_pixels_to_slide)
    target = validate_raster_transform(raster_transform)
    decoded = _validate_decoded_rect(decoded_rect_in_view, source.shape)
    bounds = _validate_physical_bounds(physical_bounds_in_view, decoded)
    rectangle = _validate_output_rect(output_rect, target["raster"])

    x = np.arange(rectangle["x"], rectangle["x"] + rectangle["width"], dtype=np.float64)
    y = np.arange(
        rectangle["y"], rectangle["y"] + rectangle["height"], dtype=np.float64
    )
    raster_to_slide = target["raster_to_slide"]
    source_x = _target_axis_in_view(
        x,
        raster_to_slide["translate_x"],
        raster_to_slide["scale_x"],
        view_transform["translate_x"],
        view_transform["scale_x"],
    )
    source_y = _target_axis_in_view(
        y,
        raster_to_slide["translate_y"],
        raster_to_slide["scale_y"],
        view_transform["translate_y"],
        view_transform["scale_y"],
    )

    fixed_map_x = np.broadcast_to(
        source_x.astype(np.float32)[None, :], (len(y), len(x))
    ).copy()
    fixed_map_y = np.broadcast_to(
        source_y.astype(np.float32)[:, None], (len(y), len(x))
    ).copy()

    # OpenCV 5 samples float maps without the table quantization used by
    # OpenCV 4. Derive support from the exact float32 coordinates passed to
    # remap; convertMaps would incorrectly hide small nonzero contributions.
    if not np.isfinite(fixed_map_x).all() or not np.isfinite(fixed_map_y).all():
        raise ValueError("target-to-source mapping is not finite")
    x0 = np.floor(fixed_map_x).astype(np.int64)
    y0 = np.floor(fixed_map_y).astype(np.int64)
    fx = fixed_map_x.astype(np.float64) - x0
    fy = fixed_map_y.astype(np.float64) - y0

    _require_neighbor_available(x0, y0, (1.0 - fx) * (1.0 - fy), bounds, decoded)
    _require_neighbor_available(x0 + 1, y0, fx * (1.0 - fy), bounds, decoded)
    _require_neighbor_available(x0, y0 + 1, (1.0 - fx) * fy, bounds, decoded)
    _require_neighbor_available(x0 + 1, y0 + 1, fx * fy, bounds, decoded)

    map_x = _localize_float_map(fixed_map_x, decoded["x"])
    map_y = _localize_float_map(fixed_map_y, decoded["y"])

    return cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _require_neighbor_available(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bounds: Mapping[str, int],
    decoded: Mapping[str, int],
) -> None:
    in_view = (
        (x >= decoded["x"])
        & (x < decoded["x"] + decoded["width"])
        & (y >= decoded["y"])
        & (y < decoded["y"] + decoded["height"])
    )
    in_physical_image = (
        (x >= bounds["min_x"])
        & (x < bounds["max_x"])
        & (y >= bounds["min_y"])
        & (y < bounds["max_y"])
    )
    if np.any((weight > 0.0) & in_physical_image & ~in_view):
        raise InsufficientSourceContext(
            "source view omits an in-image bilinear contribution"
        )


def _localize_float_map(value: np.ndarray, offset: int) -> np.ndarray:
    localized = (value.astype(np.float64) - offset).astype(np.float32)
    restored = (localized.astype(np.float64) + offset).astype(np.float32)
    if not np.array_equal(restored, value):
        raise ValueError(
            "decoded source partition cannot preserve the fixed float32 map"
        )
    return localized


def _target_axis_in_view(
    positions_or_count: Any,
    target_translate: float,
    target_scale: float,
    view_translate: float,
    view_scale: float,
) -> np.ndarray:
    positions = (
        np.arange(positions_or_count, dtype=np.float64)
        if isinstance(positions_or_count, int)
        else positions_or_count
    )
    coordinates = (
        target_translate + (positions + 0.5) * target_scale - view_translate
    ) / view_scale - 0.5
    result = coordinates.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("target-to-source mapping is not finite")
    return result


def _required_physical_axis_pixels(
    coordinates: np.ndarray, view_offset: int, physical_extent: int
) -> np.ndarray:
    base = np.floor(coordinates).astype(np.int64)
    fraction = coordinates.astype(np.float64) - base
    candidates = [base + view_offset]
    if np.any(fraction > 0.0):
        candidates.append((base + 1 + view_offset)[fraction > 0.0])
    combined = np.concatenate(candidates)
    return combined[(combined >= 0) & (combined < physical_extent)]


def _validate_source_pixels(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError("source_rgb must be a numpy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("source_rgb must have uint8 shape (height, width, 3)")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError("source_rgb dimensions must be positive")
    return value


def _validate_decoded_rect(value: Any | None, shape: tuple[int, ...]) -> dict[str, int]:
    rectangle = (
        {"x": 0, "y": 0, "width": shape[1], "height": shape[0]}
        if value is None
        else validate_int_rect(value)
    )
    if rectangle["width"] != shape[1] or rectangle["height"] != shape[0]:
        raise ValueError("decoded_rect_in_view dimensions must match source_rgb")
    return rectangle


def _validate_physical_bounds(value: Any, decoded: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _PHYSICAL_BOUNDS_FIELDS:
        raise ValueError("physical_bounds_in_view has missing or unknown fields")
    result = {
        field: _require_int(value[field], field) for field in _PHYSICAL_BOUNDS_FIELDS
    }
    if result["max_x"] <= result["min_x"] or result["max_y"] <= result["min_y"]:
        raise ValueError("physical_bounds_in_view must have positive area")
    if (
        result["min_x"] > decoded["x"]
        or result["min_y"] > decoded["y"]
        or result["max_x"] < decoded["x"] + decoded["width"]
        or result["max_y"] < decoded["y"] + decoded["height"]
    ):
        raise ValueError("physical_bounds_in_view must contain the decoded source view")
    return result


def _validate_dimensions(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _DIMENSION_FIELDS:
        raise ValueError("selected_level_dimensions must contain width and height")
    result = {
        field: _require_int(value[field], f"selected_level_dimensions.{field}")
        for field in _DIMENSION_FIELDS
    }
    if result["width"] <= 0 or result["height"] <= 0:
        raise ValueError("selected-level dimensions must be positive")
    return result


def _validate_output_rect(
    value: Any | None, raster: Mapping[str, Any]
) -> dict[str, int]:
    rectangle = (
        {"x": 0, "y": 0, "width": raster["width"], "height": raster["height"]}
        if value is None
        else validate_int_rect(value)
    )
    if (
        rectangle["x"] + rectangle["width"] > raster["width"]
        or rectangle["y"] + rectangle["height"] > raster["height"]
    ):
        raise ValueError("output_rect lies outside the target raster")
    return rectangle


def _require_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value
