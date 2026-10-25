"""Pixel-exact training-label rasterization shared by seal and execution.

This module owns the scientific conversion from canonical slide-space polygons
to one model-space target grid.  It performs no package IO and stores no label
rasters.  Callers choose bounded output windows; dataset admission uses the
streaming summary to validate an entire ROI without allocating an ROI-sized
array.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from shapely import transform
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from histoforge.canonical_geometry import canonical_geometry
from histoforge.cloud.modules.schema import model_output_target_classes
from histoforge.cloud.rasterization.inputs import validate_raster_transform
from histoforge.cloud.rasterization.rasterizer import real_pixel_mask
from histoforge.cloud.sources.identity import validate_int_rect

_INSTANCE_OUTPUT_TYPES = {
    "instance_segmentation",
    "instance_segmentation_and_classification",
}
_DENSE_OUTPUT_TYPE = "dense_classification"
_SUMMARY_CHUNK_SIDE_PX = 1024


@dataclass(frozen=True)
class TrainingRoiLabelSummary:
    """Raster-visible target occurrences for one supervised training ROI."""

    target_object_counts: dict[str, int]
    visible_object_indices: tuple[int, ...]

    @property
    def is_negative(self) -> bool:
        """Return whether no captured target has a supervised real pixel."""

        return not self.visible_object_indices


def rasterize_training_geometry(
    geometry: Any,
    raster_transform: Any,
    output_rect: Any,
) -> np.ndarray:
    """Rasterize one Polygon or MultiPolygon with the fixed pixel-center rule.

    The returned boolean array covers ``output_rect`` in the complete target
    grid.  Geometry is never clipped before this operation, so callers may
    derive complete-instance targets and apply spatial support afterward.
    """

    target = validate_raster_transform(raster_transform)
    rectangle = _bounded_output_rect(output_rect, target)
    polygon = shape(canonical_geometry(geometry))
    return _rasterize_shape(polygon, target, rectangle)


def training_geometry_on_model_grid(
    geometry: Any, raster_transform: Any
) -> BaseGeometry:
    """Map complete geometry to the target grid before choosing a local window.

    Apply the inverse mapping once in the fixed target coordinate system.
    Recomputing a slide-space affine origin for each window can move a boundary
    across a pixel center through floating-point cancellation.
    """

    target = validate_raster_transform(raster_transform)
    return _model_geometry(shape(canonical_geometry(geometry)), target)


def rasterize_model_geometry(
    geometry: BaseGeometry, rectangle: Mapping[str, int]
) -> np.ndarray:
    """Rasterize already-mapped geometry at an integer grid offset.

    The trusted caller supplies positive dimensions and may request negative
    origins for complete objects crossing the target. This allocates only the
    requested window and does not clip the supplied geometry.
    """

    return _rasterize_model_shapes([geometry], rectangle).astype(bool, copy=False)


def training_real_pixel_mask(
    physical_slide_bounds: Any,
    raster_transform: Any,
    output_rect: Any,
) -> np.ndarray:
    """Return target pixels whose centers lie inside the physical image."""

    target = validate_raster_transform(raster_transform)
    rectangle = _bounded_output_rect(output_rect, target)
    bounds = _physical_bounds(physical_slide_bounds)
    return real_pixel_mask(bounds, target, rectangle)


def inspect_training_roi_labels(
    *,
    roi_geometry: Any,
    objects: Sequence[Mapping[str, Any]],
    model_output: Mapping[str, Any],
    raster_transform: Any,
    physical_slide_bounds: Any,
) -> TrainingRoiLabelSummary:
    """Validate raster support and summarize one complete training ROI.

    Objects are counted only when at least one of their pixels lies inside both
    the ROI supervision mask and the physical image.  Instance tasks reject a
    supervised pixel shared by two instances.  Dense tasks union same-class
    overlap and reject cross-class overlap.  The scan is bounded by fixed-size
    windows and therefore never constructs an ROI-sized dense label raster.
    """

    target = validate_raster_transform(raster_transform)
    bounds = _physical_bounds(physical_slide_bounds)
    roi = shape(canonical_geometry(roi_geometry))
    class_names = model_output_target_classes(model_output)
    output_type = model_output["type"]
    if output_type not in _INSTANCE_OUTPUT_TYPES | {_DENSE_OUTPUT_TYPE}:
        raise ValueError("training label rasterization requires a segmentation output")

    if not _geometry_has_supported_pixel(roi, roi, target, bounds):
        raise ValueError("training ROI has no supervised real pixels")

    class_indices = {name: index for index, name in enumerate(class_names)}
    geometries: list[BaseGeometry] = []
    object_class_indices: list[int] = []
    for item in objects:
        try:
            object_geometry = item["geometry"]
            path_class = item["properties"]["path_class"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "training object is missing geometry or path_class"
            ) from error
        if path_class not in class_indices:
            raise ValueError("training object PathClass is not a target class")
        geometries.append(shape(canonical_geometry(object_geometry)))
        object_class_indices.append(class_indices[path_class])

    visible = tuple(
        index
        for index, geometry in enumerate(geometries)
        if _geometry_has_supported_pixel(geometry, roi, target, bounds)
    )
    _require_unambiguous_overlap(
        geometries,
        object_class_indices,
        visible,
        roi,
        target,
        bounds,
        instance_output=output_type in _INSTANCE_OUTPUT_TYPES,
    )

    counts = {name: 0 for name in class_names}
    for index in visible:
        counts[class_names[object_class_indices[index]]] += 1
    return TrainingRoiLabelSummary(counts, visible)


def _require_unambiguous_overlap(
    geometries: Sequence[BaseGeometry],
    class_indices: Sequence[int],
    visible: Sequence[int],
    roi: BaseGeometry,
    target: Mapping[str, Any],
    physical_bounds: tuple[float, float, float, float],
    *,
    instance_output: bool,
) -> None:
    if len(visible) < 2:
        return
    if instance_output:
        _require_no_instance_overlap(geometries, visible, roi, target, physical_bounds)
        return
    _require_no_cross_class_overlap(
        geometries,
        class_indices,
        visible,
        roi,
        target,
        physical_bounds,
    )


def _require_no_instance_overlap(
    geometries: Sequence[BaseGeometry],
    visible: Sequence[int],
    roi: BaseGeometry,
    target: Mapping[str, Any],
    physical_bounds: tuple[float, float, float, float],
) -> None:
    selected = [geometries[index] for index in visible]
    for rectangle, candidates in _collision_leaves(
        selected, roi, target, physical_bounds
    ):
        support = _effective_support(roi, target, physical_bounds, rectangle)
        if not support.any():
            continue
        counts = _rasterize_shapes(
            [selected[index] for index in candidates],
            target,
            rectangle,
            count_overlaps=True,
        )
        if np.any((counts > 1) & support):
            raise ValueError("training target instances share a supervised real pixel")


def _require_no_cross_class_overlap(
    geometries: Sequence[BaseGeometry],
    class_indices: Sequence[int],
    visible: Sequence[int],
    roi: BaseGeometry,
    target: Mapping[str, Any],
    physical_bounds: tuple[float, float, float, float],
) -> None:
    selected = [geometries[index] for index in visible]
    selected_classes = [class_indices[index] for index in visible]
    if len(set(selected_classes)) < 2:
        return
    for rectangle, candidates in _collision_leaves(
        selected, roi, target, physical_bounds
    ):
        support = _effective_support(roi, target, physical_bounds, rectangle)
        if not support.any():
            continue
        by_class: dict[int, list[BaseGeometry]] = {}
        for index in candidates:
            by_class.setdefault(selected_classes[index], []).append(selected[index])
        if len(by_class) < 2:
            continue
        occupied = np.zeros(support.shape, dtype=bool)
        for class_index in sorted(by_class):
            current = _rasterize_shapes(
                by_class[class_index], target, rectangle
            ).astype(bool, copy=False)
            if np.any(occupied & current & support):
                raise ValueError(
                    "training target classes share a supervised real pixel"
                )
            occupied |= current


def _geometry_has_supported_pixel(
    geometry: BaseGeometry,
    roi: BaseGeometry,
    target: Mapping[str, Any],
    physical_bounds: tuple[float, float, float, float],
) -> bool:
    candidate = _candidate_rect(
        _intersect_bounds(geometry.bounds, roi.bounds, physical_bounds), target
    )
    if candidate is None:
        return False
    for rectangle in _intersecting_leaves(candidate, (geometry, roi), target):
        geometry_mask = _rasterize_shape(geometry, target, rectangle)
        if not geometry_mask.any():
            continue
        support = _rasterize_shape(roi, target, rectangle)
        support &= real_pixel_mask(physical_bounds, target, rectangle)
        if np.any(geometry_mask & support):
            return True
    return False


def _collision_leaves(
    geometries: Sequence[BaseGeometry],
    roi: BaseGeometry,
    target: Mapping[str, Any],
    physical_bounds: tuple[float, float, float, float],
) -> Iterator[tuple[dict[str, int], tuple[int, ...]]]:
    root = _candidate_rect(
        _intersect_bounds(_union_bounds(geometries), roi.bounds, physical_bounds),
        target,
    )
    if root is None:
        return
    tree = STRtree(geometries)
    pending = [root]
    while pending:
        rectangle = pending.pop()
        slide_box = _rect_slide_box(rectangle, target)
        if not roi.intersects(slide_box):
            continue
        candidates = tuple(
            int(index) for index in tree.query(slide_box, predicate="intersects")
        )
        if len(candidates) < 2:
            continue
        if (
            rectangle["width"] <= _SUMMARY_CHUNK_SIDE_PX
            and rectangle["height"] <= _SUMMARY_CHUNK_SIDE_PX
        ):
            yield rectangle, candidates
        else:
            pending.extend(_split_rect(rectangle))


def _intersecting_leaves(
    root: Mapping[str, int],
    geometries: Sequence[BaseGeometry],
    target: Mapping[str, Any],
) -> Iterator[dict[str, int]]:
    pending = [dict(root)]
    while pending:
        rectangle = pending.pop()
        slide_box = _rect_slide_box(rectangle, target)
        if any(not geometry.intersects(slide_box) for geometry in geometries):
            continue
        if (
            rectangle["width"] <= _SUMMARY_CHUNK_SIDE_PX
            and rectangle["height"] <= _SUMMARY_CHUNK_SIDE_PX
        ):
            yield rectangle
        else:
            pending.extend(_split_rect(rectangle))


def _effective_support(
    roi: BaseGeometry,
    target: Mapping[str, Any],
    physical_bounds: tuple[float, float, float, float],
    rectangle: Mapping[str, int],
) -> np.ndarray:
    support = _rasterize_shape(roi, target, rectangle)
    support &= real_pixel_mask(physical_bounds, target, rectangle)
    return support


def _rasterize_shape(
    geometry: BaseGeometry,
    target: Mapping[str, Any],
    rectangle: Mapping[str, int],
) -> np.ndarray:
    return _rasterize_shapes([geometry], target, rectangle).astype(bool, copy=False)


def _rasterize_shapes(
    geometries: Sequence[BaseGeometry],
    target: Mapping[str, Any],
    rectangle: Mapping[str, int],
    *,
    count_overlaps: bool = False,
) -> np.ndarray:
    return _rasterize_model_shapes(
        [_model_geometry(geometry, target) for geometry in geometries],
        rectangle,
        count_overlaps=count_overlaps,
    )


def _rasterize_model_shapes(
    geometries: Sequence[BaseGeometry],
    rectangle: Mapping[str, int],
    *,
    count_overlaps: bool = False,
) -> np.ndarray:
    return rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=(rectangle["height"], rectangle["width"]),
        transform=Affine.translation(rectangle["x"], rectangle["y"]),
        fill=0,
        default_value=1,
        dtype="uint32" if count_overlaps else "uint8",
        all_touched=False,
        merge_alg=MergeAlg.add if count_overlaps else MergeAlg.replace,
    )


def _model_geometry(geometry: BaseGeometry, target: Mapping[str, Any]) -> BaseGeometry:
    """Use one inverse coordinate calculation for every window of a target."""

    mapping = target["raster_to_slide"]
    offset = np.array([mapping["translate_x"], mapping["translate_y"]])
    scale = np.array([mapping["scale_x"], mapping["scale_y"]])
    return transform(
        geometry,
        lambda coordinates: (coordinates - offset) / scale,
    )


def _bounded_output_rect(value: Any, target: Mapping[str, Any]) -> dict[str, int]:
    rectangle = validate_int_rect(value)
    raster = target["raster"]
    if (
        rectangle["x"] + rectangle["width"] > raster["width"]
        or rectangle["y"] + rectangle["height"] > raster["height"]
    ):
        raise ValueError("training raster output_rect lies outside the target grid")
    return rectangle


def _physical_bounds(value: Any) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        raise ValueError("physical_slide_bounds must contain four numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("physical_slide_bounds must be finite")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("physical_slide_bounds must have positive extents")
    return result


def _intersect_bounds(
    *values: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    result = (
        max(value[0] for value in values),
        max(value[1] for value in values),
        min(value[2] for value in values),
        min(value[3] for value in values),
    )
    if result[2] < result[0] or result[3] < result[1]:
        return None
    return result


def _union_bounds(
    geometries: Sequence[BaseGeometry],
) -> tuple[float, float, float, float]:
    return (
        min(geometry.bounds[0] for geometry in geometries),
        min(geometry.bounds[1] for geometry in geometries),
        max(geometry.bounds[2] for geometry in geometries),
        max(geometry.bounds[3] for geometry in geometries),
    )


def _candidate_rect(
    bounds: tuple[float, float, float, float] | None,
    target: Mapping[str, Any],
) -> dict[str, int] | None:
    if bounds is None:
        return None
    mapping = target["raster_to_slide"]
    raster = target["raster"]
    x0 = max(
        0, math.floor((bounds[0] - mapping["translate_x"]) / mapping["scale_x"]) - 1
    )
    y0 = max(
        0, math.floor((bounds[1] - mapping["translate_y"]) / mapping["scale_y"]) - 1
    )
    x1 = min(
        raster["width"],
        math.ceil((bounds[2] - mapping["translate_x"]) / mapping["scale_x"]) + 1,
    )
    y1 = min(
        raster["height"],
        math.ceil((bounds[3] - mapping["translate_y"]) / mapping["scale_y"]) + 1,
    )
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _rect_slide_box(
    rectangle: Mapping[str, int], target: Mapping[str, Any]
) -> BaseGeometry:
    mapping = target["raster_to_slide"]
    x0 = mapping["translate_x"] + rectangle["x"] * mapping["scale_x"]
    y0 = mapping["translate_y"] + rectangle["y"] * mapping["scale_y"]
    x1 = x0 + rectangle["width"] * mapping["scale_x"]
    y1 = y0 + rectangle["height"] * mapping["scale_y"]
    return box(x0, y0, x1, y1)


def _split_rect(rectangle: Mapping[str, int]) -> tuple[dict[str, int], ...]:
    x = rectangle["x"]
    y = rectangle["y"]
    width = rectangle["width"]
    height = rectangle["height"]
    if width > _SUMMARY_CHUNK_SIDE_PX and (
        width >= height or height <= _SUMMARY_CHUNK_SIDE_PX
    ):
        first_width = width // 2
        return (
            {"x": x, "y": y, "width": first_width, "height": height},
            {
                "x": x + first_width,
                "y": y,
                "width": width - first_width,
                "height": height,
            },
        )
    first_height = height // 2
    return (
        {"x": x, "y": y, "width": width, "height": first_height},
        {
            "x": x,
            "y": y + first_height,
            "width": width,
            "height": height - first_height,
        },
    )
