"""Convert decoded label masks into the fixed inference feature contract."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import shapes

from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference.features import (
    CanonicalFeatureGeometry,
    ResultLimitExceeded,
    build_feature_collection,
    build_feature_collection_from_canonical_geometries,
    encoded_feature_collection_size,
    encoded_feature_record_size,
)


def feature_collection_from_mask(
    mask: Any,
    classifications: str | Mapping[int, str],
    declared_classes: Collection[str],
    result_budget: Mapping[str, int] | None = None,
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> dict[str, Any]:
    """Emit one Polygon/MultiPolygon feature per nonbackground instance ID.

    Pixel offsets place a cropped mask in its caller's raster coordinates before
    canonical geometry, feature identities, and encoded-byte limits are computed.
    """

    labels = _label_mask(mask)
    single_class, class_by_id = _classifications(classifications)
    # Validate the complete inputs first, then keep non-emitted IDs out of
    # Rasterio's native polygon layer. NumPy bounds its automatic lookup-table
    # choice, so sparse IDs do not require an allocation up to the largest ID.
    eligible_pixels = (
        labels > 0 if single_class is not None else np.isin(labels, list(class_by_id))
    )
    polygons_by_id: dict[int, list[list[Any]]] = {}
    encoded_polygons_by_id: dict[int, list[bytes]] = {}
    vertex_count = 0
    polygon_bytes_by_id: dict[int, int] = {}
    record_bytes_by_id: dict[int, int] = {}
    retained_feature_bytes = 0
    byte_limit = (
        None
        if result_budget is None
        else min(
            result_budget["max_transferred_bytes"],
            result_budget["max_decoded_bytes"],
        )
    )
    for geometry, raw_value in shapes(
        labels,
        mask=eligible_pixels,
        connectivity=4,
        transform=Affine.translation(x_offset, y_offset),
    ):
        instance_id = int(raw_value)
        if instance_id <= 0 or (
            single_class is None and instance_id not in class_by_id
        ):
            continue
        is_new = instance_id not in polygons_by_id
        if is_new:
            if (
                result_budget is not None
                and len(polygons_by_id) >= result_budget["max_features"]
            ):
                raise ResultLimitExceeded("inference result exceeds max_features")

        canonical = canonical_geometry(geometry)
        component_polygons = (
            [canonical["coordinates"]]
            if canonical["type"] == "Polygon"
            else canonical["coordinates"]
        )
        component_vertices = sum(
            len(ring) - 1 for polygon in component_polygons for ring in polygon
        )
        next_vertex_count = vertex_count + component_vertices
        encoded_component_polygons = [
            canonical_bytes(polygon) for polygon in component_polygons
        ]
        component_polygon_bytes = sum(map(len, encoded_component_polygons))
        prior_polygon_count = len(polygons_by_id.get(instance_id, ()))
        next_polygon_count = prior_polygon_count + len(component_polygons)
        next_polygon_bytes = (
            polygon_bytes_by_id.get(instance_id, 0) + component_polygon_bytes
        )
        geometry_type = "Polygon" if next_polygon_count == 1 else "MultiPolygon"
        encoded_coordinates_size = (
            next_polygon_bytes
            if next_polygon_count == 1
            else 2 + next_polygon_bytes + next_polygon_count - 1
        )

        if result_budget is not None:
            path_class = single_class or class_by_id[instance_id]
            next_record_bytes = encoded_feature_record_size(
                path_class, geometry_type, encoded_coordinates_size
            )
            if next_record_bytes > result_budget["max_record_bytes"]:
                raise ResultLimitExceeded("inference feature exceeds max_record_bytes")
            next_feature_bytes = (
                retained_feature_bytes
                - record_bytes_by_id.get(instance_id, 0)
                + next_record_bytes
            )
            next_feature_count = len(polygons_by_id) + int(is_new)
            if (
                encoded_feature_collection_size(next_feature_bytes, next_feature_count)
                > byte_limit
            ):
                raise ResultLimitExceeded(
                    "inference feature collection exceeds transferred or decoded "
                    "byte budget"
                )
            if next_vertex_count > result_budget["max_vertices"]:
                raise ResultLimitExceeded("inference result exceeds max_vertices")

        if is_new:
            polygons_by_id[instance_id] = []
            encoded_polygons_by_id[instance_id] = []
        polygons_by_id[instance_id].extend(component_polygons)
        encoded_polygons_by_id[instance_id].extend(encoded_component_polygons)
        polygon_bytes_by_id[instance_id] = next_polygon_bytes
        vertex_count = next_vertex_count
        if result_budget is not None:
            retained_feature_bytes = next_feature_bytes
            record_bytes_by_id[instance_id] = next_record_bytes

    instances = []
    for instance_id, polygons in polygons_by_id.items():
        encoded_polygons = encoded_polygons_by_id[instance_id]
        if len(polygons) > 1:
            ordered = sorted(
                zip(encoded_polygons, polygons, strict=True), key=lambda item: item[0]
            )
            encoded_polygons = [item[0] for item in ordered]
            polygons = [item[1] for item in ordered]
        geometry = (
            {"type": "Polygon", "coordinates": polygons[0]}
            if len(polygons) == 1
            else {"type": "MultiPolygon", "coordinates": polygons}
        )
        encoded_coordinates = (
            encoded_polygons[0]
            if len(encoded_polygons) == 1
            else b"[" + b",".join(encoded_polygons) + b"]"
        )
        instances.append(
            (
                single_class or class_by_id[instance_id],
                CanonicalFeatureGeometry(geometry, encoded_coordinates),
            )
        )
    if result_budget is None:
        return build_feature_collection(
            [(path_class, canonical.value) for path_class, canonical in instances],
            declared_classes,
        )
    return build_feature_collection_from_canonical_geometries(
        instances, declared_classes, result_budget
    )


def _label_mask(value: Any) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError("instance mask must be a two-dimensional numpy array")
    if value.size == 0 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("instance mask must contain integer pixels")
    minimum = int(value.min())
    maximum = int(value.max())
    if minimum < 0 or maximum > np.iinfo(np.int32).max:
        raise ValueError("instance mask IDs must fit a nonnegative int32")
    return value.astype(np.int32, copy=False)


def _classifications(
    value: str | Mapping[int, str],
) -> tuple[str | None, Mapping[int, str]]:
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, Mapping):
        raise ValueError("classifications must be one class or an ID-to-class map")
    result: dict[int, str] = {}
    for instance_id, path_class in value.items():
        if (
            isinstance(instance_id, bool)
            or not isinstance(instance_id, int)
            or instance_id <= 0
        ):
            raise ValueError("classification instance IDs must be positive integers")
        if not isinstance(path_class, str):
            raise ValueError("classification values must be PathClass strings")
        result[instance_id] = path_class
    return None, result
