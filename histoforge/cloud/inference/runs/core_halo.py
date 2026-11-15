"""Retain complete instances or core-cropped regions in target-image coordinates."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from histoforge.canonical_json import canonical_bytes, require_positive_int32
from histoforge.cloud.inference.features import (
    CanonicalFeatureGeometry,
    CanonicalFeatureRecords,
    build_feature_collection,
    build_feature_collection_from_canonical_geometries,
    build_feature_records_from_canonical_geometries,
    validate_feature_collection,
)
from histoforge.cloud.inference.plans import validate_processing_window


class InsufficientCoreHaloContext(ValueError):
    """An owned object reaches an artificial processing-window boundary."""


def analyze_core_halo_unit(
    window_value: Any,
    window_collection: Any,
    declared_classes: Collection[str],
    *,
    target_width: Any,
    target_height: Any,
    result_budget: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Select instances from one window without hiding incomplete objects.

    The returned population contains only features whose GEOS point on surface
    lies in the unit's half-open ownership core. Diagnostics retain the number
    of owned features touching an artificial input edge; the production caller
    must reject a nonzero count before publishing the unit.
    """

    collection = validate_feature_collection(window_collection, declared_classes)
    instances, diagnostics = _owned_instance_geometries(
        window_value, collection, target_width, target_height
    )
    if result_budget is None:
        owned = build_feature_collection(
            [(path_class, geometry.value) for path_class, geometry in instances],
            declared_classes,
        )
    else:
        owned = build_feature_collection_from_canonical_geometries(
            instances, declared_classes, result_budget
        )
    return owned, diagnostics


def select_canonical_core_halo_unit_features(
    window_value: Any,
    window_collection: Mapping[str, Any],
    declared_classes: Collection[str],
    *,
    target_width: int,
    target_height: int,
    result_budget: Mapping[str, int],
) -> CanonicalFeatureRecords:
    """Select builder-owned instances and retain final coordinate bytes.

    The collection must come from a feature builder in the same call stack.
    External values use ``select_core_halo_unit_features`` instead. Only the
    repeated feature-contract validation is omitted: input bounds, core ownership,
    artificial-edge rejection and translated-output limits are still checked.
    """

    instances, diagnostics = _owned_instance_geometries(
        window_value, window_collection, target_width, target_height
    )
    owned = build_feature_records_from_canonical_geometries(
        instances, declared_classes, result_budget
    )
    if diagnostics["insufficient_context_count"]:
        raise InsufficientCoreHaloContext(
            "owned model output reaches an artificial CoreHalo input edge"
        )
    return owned


def _owned_instance_geometries(
    window_value: Any,
    collection: Mapping[str, Any],
    target_width: int,
    target_height: int,
) -> tuple[list[tuple[str, CanonicalFeatureGeometry]], dict[str, int]]:
    window = validate_processing_window(window_value)
    width = require_positive_int32(target_width, "target_width")
    height = require_positive_int32(target_height, "target_height")
    _require_window_inside_target(window, width, height)
    input_rect = window["input_rect"]
    core_rect = window["core_rect"]

    instances: list[tuple[str, CanonicalFeatureGeometry]] = []
    discarded = 0
    context_failures = 0
    for feature in collection["features"]:
        local = shape(feature["geometry"])
        _require_supplied_geometry(local, input_rect)
        interior = local.representative_point()
        interior_x = interior.x + input_rect["x"]
        interior_y = interior.y + input_rect["y"]
        if not _core_owns(interior_x, interior_y, core_rect):
            discarded += 1
            continue
        if _touches_nonlogical_input_edge(local, input_rect, width, height):
            context_failures += 1
        translated = _translate_canonical_geometry(
            feature["geometry"], input_rect["x"], input_rect["y"]
        )
        instances.append(
            (
                feature["properties"]["path_class"],
                translated,
            )
        )

    return instances, {
        "unit_id": window["unit_id"],
        "decoded_feature_count": len(collection["features"]),
        "owned_feature_count": len(instances),
        "discarded_feature_count": discarded,
        "insufficient_context_count": context_failures,
    }


def select_core_halo_unit_features(
    window_value: Any,
    window_collection: Any,
    declared_classes: Collection[str],
    *,
    target_width: Any,
    target_height: Any,
    result_budget: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Keep complete instances assigned to this core, rejecting artificial cuts."""

    owned, diagnostics = analyze_core_halo_unit(
        window_value,
        window_collection,
        declared_classes,
        target_width=target_width,
        target_height=target_height,
        result_budget=result_budget,
    )
    if diagnostics["insufficient_context_count"]:
        raise InsufficientCoreHaloContext(
            "owned model output reaches an artificial CoreHalo input edge"
        )
    return owned


def translate_core_regions(
    window_value: Any,
    window_collection: Any,
    declared_classes: Collection[str],
    *,
    target_width: int,
    target_height: int,
    result_budget: Mapping[str, int],
) -> dict[str, Any]:
    """Place core-cropped semantic regions into the complete target image.

    The predictor has already cropped class labels to the ownership core.
    Geometry still uses processing-image coordinates. Every fragment must stay
    inside that core; no interior-point selection or instance edge check applies.
    """

    collection = validate_feature_collection(window_collection, declared_classes)
    return translate_canonical_core_regions(
        window_value,
        collection,
        declared_classes,
        target_width=target_width,
        target_height=target_height,
        result_budget=result_budget,
    ).as_collection()


def translate_canonical_core_regions(
    window_value: Any,
    collection: Mapping[str, Any],
    declared_classes: Collection[str],
    *,
    target_width: int,
    target_height: int,
    result_budget: Mapping[str, int],
) -> CanonicalFeatureRecords:
    """Translate builder-owned core-cropped regions, retaining final coordinate bytes.

    Use ``translate_core_regions`` for external values. The predictor's existing
    builder owns canonical geometry; this step still enforces core containment
    and translated-output limits. No instance ownership rule applies to regions.
    """

    window = validate_processing_window(window_value)
    _require_window_inside_target(window, target_width, target_height)
    input_rect = window["input_rect"]
    core = window["core_rect"]
    left = core["x"] - input_rect["x"]
    top = core["y"] - input_rect["y"]
    regions = []
    for feature in collection["features"]:
        local = shape(feature["geometry"])
        min_x, min_y, max_x, max_y = local.bounds
        if (
            min_x < left
            or min_y < top
            or max_x > left + core["width"]
            or max_y > top + core["height"]
        ):
            raise ValueError("semantic region lies outside its ownership core")
        geometry = _translate_canonical_geometry(
            feature["geometry"], input_rect["x"], input_rect["y"]
        )
        regions.append(
            (
                feature["properties"]["path_class"],
                geometry,
            )
        )
    return build_feature_records_from_canonical_geometries(
        regions, declared_classes, result_budget
    )


def _require_window_inside_target(
    window: Mapping[str, Any], target_width: int, target_height: int
) -> None:
    for name in ("core_rect", "input_rect"):
        rectangle = window[name]
        if (
            _rect_end(rectangle, "x") > target_width
            or _rect_end(rectangle, "y") > target_height
        ):
            raise ValueError(f"{name} lies outside the complete target raster")


def _rect_end(rectangle: Mapping[str, Any], axis: str) -> int:
    dimension = "width" if axis == "x" else "height"
    return int(rectangle[axis]) + int(rectangle[dimension])


def _core_owns(x: float, y: float, core: Mapping[str, Any]) -> bool:
    return core["x"] <= x < _rect_end(core, "x") and core["y"] <= y < _rect_end(
        core, "y"
    )


def _touches_nonlogical_input_edge(
    geometry: BaseGeometry,
    input_rect: Mapping[str, Any],
    target_width: int,
    target_height: int,
) -> bool:
    min_x, min_y, max_x, max_y = geometry.bounds
    input_max_x = _rect_end(input_rect, "x")
    input_max_y = _rect_end(input_rect, "y")
    return (
        (input_rect["x"] > 0 and min_x <= 0)
        or (input_max_x < target_width and max_x >= input_rect["width"])
        or (input_rect["y"] > 0 and min_y <= 0)
        or (input_max_y < target_height and max_y >= input_rect["height"])
    )


def _require_supplied_geometry(
    geometry: BaseGeometry, input_rect: Mapping[str, int]
) -> None:
    min_x, min_y, max_x, max_y = geometry.bounds
    if (
        geometry.geom_type not in {"Polygon", "MultiPolygon"}
        or geometry.is_empty
        or not geometry.is_valid
        or min_x < 0
        or min_y < 0
        or max_x > input_rect["width"]
        or max_y > input_rect["height"]
    ):
        raise ValueError("window feature geometry must be valid and inside input_rect")


def _translate_canonical_geometry(
    geometry: Mapping[str, Any], x_offset: int, y_offset: int
) -> CanonicalFeatureGeometry:
    """Translate coordinates and encode each ring once for ordering and output."""

    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    translated = []
    for polygon in polygons:
        rings = [
            [[point[0] + x_offset, point[1] + y_offset] for point in ring]
            for ring in polygon
        ]
        encoded_rings = [(canonical_bytes(ring), ring) for ring in rings]
        encoded_rings[1:] = sorted(encoded_rings[1:], key=lambda item: item[0])
        translated.append(
            (
                b"[" + b",".join(encoded for encoded, _ in encoded_rings) + b"]",
                [ring for _, ring in encoded_rings],
            )
        )
    if geometry["type"] == "MultiPolygon":
        translated.sort(key=lambda item: item[0])
        encoded_coordinates = b"[" + b",".join(item[0] for item in translated) + b"]"
        coordinates = [item[1] for item in translated]
    else:
        encoded_coordinates, coordinates = translated[0]
    return CanonicalFeatureGeometry(
        {"type": geometry["type"], "coordinates": coordinates}, encoded_coordinates
    )
