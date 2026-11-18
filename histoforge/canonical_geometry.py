"""Canonical Polygon/MultiPolygon coordinates and identities.

Topology validation is delegated to GEOS through Shapely. This module owns only
the deterministic representation layered on top: structural ring cleanup,
winding, cyclic start, component order, JCS bytes, and SHA-256.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from shapely import MultiPolygon, Polygon, is_valid_reason
from shapely.errors import GEOSException

from histoforge.canonical_json import canonical_bytes, sha256_bytes

Coordinate = tuple[float, float]
Ring = list[Coordinate]
PolygonCoordinates = list[Ring]


def canonical_geometry(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized GeoJSON Polygon or MultiPolygon geometry.

    Only representation-preserving cleanup is performed. Invalid topology is
    rejected rather than repaired, merged, clipped, or quantized.
    """

    geometry_type, polygons, topology = _validated_polygons(geometry)
    components = [topology] if geometry_type == "Polygon" else topology.geoms
    canonical_polygons = [
        _canonical_polygon(polygon, component)
        for polygon, component in zip(polygons, components, strict=True)
    ]
    if geometry_type == "MultiPolygon":
        canonical_polygons.sort(key=canonical_bytes)
        coordinates: Any = canonical_polygons
    else:
        coordinates = canonical_polygons[0]
    return {"type": geometry_type, "coordinates": coordinates}


def canonical_geometry_coordinates(geometry: Mapping[str, Any]) -> list[Any]:
    """Return the canonical RFC 7946 coordinate array for ``geometry``."""

    return canonical_geometry(geometry)["coordinates"]


def canonical_geometry_bytes(geometry: Mapping[str, Any]) -> bytes:
    """Return JCS bytes for only the canonical coordinate array."""

    return canonical_bytes(canonical_geometry_coordinates(geometry))


def canonical_geometry_digest(geometry: Mapping[str, Any]) -> str:
    """Return lowercase SHA-256 over canonical coordinate-array bytes."""

    return sha256_bytes(canonical_geometry_bytes(geometry))


def _validated_polygons(
    geometry: Mapping[str, Any],
) -> tuple[str, list[PolygonCoordinates], Polygon | MultiPolygon]:
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be an object")

    geometry_type = geometry.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("geometry type must be Polygon or MultiPolygon")
    if "coordinates" not in geometry:
        raise ValueError("geometry coordinates are required")

    if geometry_type == "Polygon":
        polygons = [_polygon(geometry["coordinates"], "coordinates")]
    else:
        raw_polygons = _sequence(geometry["coordinates"], "coordinates")
        if not raw_polygons:
            raise ValueError("MultiPolygon must contain at least one polygon")
        polygons = [
            _polygon(value, f"coordinates[{index}]")
            for index, value in enumerate(raw_polygons)
        ]

    try:
        topology = (
            Polygon(polygons[0][0], polygons[0][1:])
            if geometry_type == "Polygon"
            else MultiPolygon(
                [Polygon(polygon[0], polygon[1:]) for polygon in polygons]
            )
        )
    except (GEOSException, TypeError, ValueError) as exc:
        raise ValueError(f"invalid {geometry_type} geometry: {exc}") from exc

    if topology.is_empty or not topology.is_valid:
        raise ValueError(
            f"invalid {geometry_type} geometry: {is_valid_reason(topology)}"
        )
    return geometry_type, polygons, topology


def _polygon(value: Any, path: str) -> PolygonCoordinates:
    raw_rings = _sequence(value, path)
    if not raw_rings:
        raise ValueError(f"{path} must contain a shell")
    return [_ring(ring, f"{path}[{index}]") for index, ring in enumerate(raw_rings)]


def _ring(value: Any, path: str) -> Ring:
    raw_points = _sequence(value, path)
    points: Ring = []
    for index, raw_point in enumerate(raw_points):
        point = _coordinate(raw_point, f"{path}[{index}]")
        if not points or point != points[-1]:
            points.append(point)

    if len(points) > 1 and points[-1] == points[0]:
        points.pop()
    if len(points) < 3:
        raise ValueError(f"{path} must contain at least three distinct ring vertices")
    return points


def _coordinate(value: Any, path: str) -> Coordinate:
    ordinates = _sequence(value, path)
    if len(ordinates) != 2:
        raise ValueError(f"{path} must be an exact 2D coordinate")
    return (
        _ordinate(ordinates[0], f"{path}[0]"),
        _ordinate(ordinates[1], f"{path}[1]"),
    )


def _ordinate(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite JSON number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite JSON number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite JSON number")
    return 0.0 if number == 0.0 else number


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    raise ValueError(f"{path} must be an array")


def _canonical_polygon(
    polygon: PolygonCoordinates, topology: Polygon
) -> list[list[list[float]]]:
    # Validation already constructed these exact cleaned rings. Read their
    # winding instead of converting the Python coordinates to GEOS again.
    shell = _canonical_ring(polygon[0], reverse=not topology.exterior.is_ccw)
    holes = [
        _canonical_ring(ring, reverse=interior.is_ccw)
        for ring, interior in zip(polygon[1:], topology.interiors, strict=True)
    ]
    holes.sort(key=canonical_bytes)
    return [shell, *holes]


def _canonical_ring(ring: Ring, *, reverse: bool) -> list[list[float]]:
    cycle = list(ring)
    if reverse:
        cycle.reverse()

    start = _least_rotation(cycle)
    cycle = cycle[start:] + cycle[:start]
    closed = cycle + [cycle[0]]
    return [[x, y] for x, y in closed]


def _least_rotation(values: Ring) -> int:
    """Return the start of the lexicographically least cyclic sequence in O(n)."""

    size = len(values)
    left = 0
    right = 1
    offset = 0
    while left < size and right < size and offset < size:
        first = values[(left + offset) % size]
        second = values[(right + offset) % size]
        if first == second:
            offset += 1
            continue
        if first > second:
            left += offset + 1
            if left <= right:
                left = right + 1
        else:
            right += offset + 1
            if right <= left:
                right = left + 1
        offset = 0
    return min(left, right)
