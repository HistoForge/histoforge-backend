"""Deterministic fixed GeoJSON features for model inference results.

The backend owns feature construction.  The extension independently validates
the same closed representation before any geometry can become a QuPath object.
This module contains no runner, transport, or persistence policy.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from histoforge.canonical_geometry import (
    canonical_geometry,
)
from histoforge.canonical_json import (
    canonical_bytes,
    require_path_class_name,
    require_sha256,
    sha256_bytes,
)

_COLLECTION_FIELDS = frozenset({"type", "features"})
_FEATURE_FIELDS = frozenset({"type", "id", "geometry", "properties"})
_GEOMETRY_FIELDS = frozenset({"type", "coordinates"})
_PROPERTY_FIELDS = frozenset({"object_type", "path_class"})
_EMPTY_COLLECTION_BYTES = len(
    canonical_bytes({"type": "FeatureCollection", "features": []})
)


class ResultLimitExceeded(ValueError):
    """A structurally valid inference result exceeds its accepted budget."""


@dataclass(frozen=True)
class CanonicalFeatureGeometry:
    """Backend-owned canonical geometry with its exact coordinate-array bytes."""

    value: Mapping[str, Any]
    encoded_coordinates: bytes


@dataclass(frozen=True)
class CanonicalFeatureRecords:
    """Constructed or validated records with matching canonical coordinate bytes.

    Coordinates are encoded once for identity checks and reused by stream
    validation and assembly in the same call stack. Consumers must not change
    the geometries before using their encoded coordinates.
    """

    features: list[dict[str, Any]]
    encoded_coordinates: list[bytes]
    vertex_count: int

    def as_collection(self) -> dict[str, Any]:
        """Expose the existing feature tree without copying or changing geometry."""

        return {"type": "FeatureCollection", "features": self.features}

    def encoded_records(self) -> Iterator[tuple[bytes, int]]:
        """Yield exact records and vertex counts while the geometries are unchanged."""

        for feature, coordinates in zip(
            self.features, self.encoded_coordinates, strict=True
        ):
            geometry = feature["geometry"]
            yield (
                encode_feature_record(
                    feature["id"],
                    feature["properties"]["path_class"],
                    geometry["type"],
                    coordinates,
                ),
                _geometry_vertex_count(geometry),
            )


def build_feature_collection(
    instances: Iterable[tuple[str, Mapping[str, Any]]],
    declared_classes: Collection[str],
) -> dict[str, Any]:
    """Build the closed deterministic feature collection for decoded instances.

    Each input tuple is ``(path_class, polygonal_geojson_geometry)``.  Input
    iteration order is deliberately irrelevant, including when two decoded
    instances have byte-identical class and geometry.
    """

    declared = _declared_classes(declared_classes)
    normalized: dict[tuple[str, bytes], dict[str, Any]] = {}
    multiplicities: Counter[tuple[str, bytes]] = Counter()

    for index, instance in enumerate(instances):
        if not isinstance(instance, tuple) or len(instance) != 2:
            raise ValueError(
                f"instances[{index}] must be a (path_class, geometry) tuple"
            )
        path_class = _emitted_class(instance[0], declared, f"instances[{index}]")
        geometry = canonical_geometry(instance[1])
        key = (path_class, canonical_bytes(geometry["coordinates"]))
        normalized.setdefault(key, geometry)
        multiplicities[key] += 1

    features: list[dict[str, Any]] = []
    for (path_class, geometry_bytes), count in multiplicities.items():
        canonical = normalized[(path_class, geometry_bytes)]
        for occurrence_index in range(count):
            geometry = copy.deepcopy(canonical)
            features.append(_feature(path_class, geometry, occurrence_index))

    features.sort(key=lambda feature: feature["id"])
    return {"type": "FeatureCollection", "features": features}


def build_bounded_feature_collection(
    instances: Iterable[tuple[str, Mapping[str, Any]]],
    declared_classes: Collection[str],
    result_budget: Mapping[str, int],
) -> dict[str, Any]:
    """Build deterministic features without first materializing an oversized result."""

    declared = _declared_classes(declared_classes)
    return _build_bounded_feature_records(
        _prepared_instances(instances, declared, canonicalize=True), result_budget
    ).as_collection()


def build_feature_collection_from_canonical_geometries(
    instances: Iterable[tuple[str, CanonicalFeatureGeometry]],
    declared_classes: Collection[str],
    result_budget: Mapping[str, int],
) -> dict[str, Any]:
    """Build bounded features from backend-owned canonical geometries.

    This is a trusted internal seam for geometries that were already produced
    by ``canonical_geometry`` in the same call stack. Wire, persisted, and
    otherwise external values must use ``build_bounded_feature_collection`` so
    topology and representation are independently validated.
    """

    return build_feature_records_from_canonical_geometries(
        instances, declared_classes, result_budget
    ).as_collection()


def build_feature_records_from_canonical_geometries(
    instances: Iterable[tuple[str, CanonicalFeatureGeometry]],
    declared_classes: Collection[str],
    result_budget: Mapping[str, int],
) -> CanonicalFeatureRecords:
    """Retain constructed records and their coordinate bytes for immediate encoding.

    Inputs are canonical geometries owned by this call stack, never wire or stored
    values. Consumers must not mutate their geometry before encoding the records.
    """

    return _build_bounded_feature_records(
        _prepared_instances(
            instances, _declared_classes(declared_classes), canonicalize=False
        ),
        result_budget,
    )


def _prepared_instances(
    instances: Iterable[tuple[str, Any]],
    declared: frozenset[str],
    *,
    canonicalize: bool,
) -> Iterable[tuple[str, CanonicalFeatureGeometry]]:
    for index, instance in enumerate(instances):
        if not isinstance(instance, tuple) or len(instance) != 2:
            raise ValueError(
                f"instances[{index}] must be a (path_class, geometry) tuple"
            )
        path_class = _emitted_class(instance[0], declared, f"instances[{index}]")
        if canonicalize:
            geometry = canonical_geometry(instance[1])
            canonical = CanonicalFeatureGeometry(
                geometry, canonical_bytes(geometry["coordinates"])
            )
        else:
            canonical = instance[1]
            if not isinstance(canonical, CanonicalFeatureGeometry):
                raise ValueError(
                    f"instances[{index}].geometry must be a CanonicalFeatureGeometry"
                )
        yield path_class, canonical


def _build_bounded_feature_records(
    instances: Iterable[tuple[str, CanonicalFeatureGeometry]],
    result_budget: Mapping[str, int],
) -> CanonicalFeatureRecords:
    occurrences: Counter[tuple[str, bytes]] = Counter()
    records: list[tuple[dict[str, Any], bytes]] = []
    feature_bytes = 0
    vertex_count = 0
    byte_limit = min(
        result_budget["max_transferred_bytes"],
        result_budget["max_decoded_bytes"],
    )

    for path_class, canonical in instances:
        if len(records) >= result_budget["max_features"]:
            raise ResultLimitExceeded("inference result exceeds max_features")

        geometry = canonical.value
        geometry_bytes = canonical.encoded_coordinates
        key = (path_class, geometry_bytes)
        occurrence_index = occurrences[key]
        occurrences[key] += 1
        feature = _feature_from_encoded_coordinates(
            path_class, geometry, geometry_bytes, occurrence_index
        )
        record_bytes = encoded_feature_record_size(
            path_class, geometry["type"], len(geometry_bytes)
        )
        if record_bytes > result_budget["max_record_bytes"]:
            raise ResultLimitExceeded("inference feature exceeds max_record_bytes")

        vertex_count += _geometry_vertex_count(geometry)
        if vertex_count > result_budget["max_vertices"]:
            raise ResultLimitExceeded("inference result exceeds max_vertices")

        feature_bytes += record_bytes
        collection_bytes = encoded_feature_collection_size(
            feature_bytes, len(records) + 1
        )
        if collection_bytes > byte_limit:
            raise ResultLimitExceeded(
                "inference feature collection exceeds transferred or decoded "
                "byte budget"
            )
        records.append((feature, geometry_bytes))

    records.sort(key=lambda record: record[0]["id"])
    return CanonicalFeatureRecords(
        [feature for feature, _ in records],
        [coordinates for _, coordinates in records],
        vertex_count,
    )


def encoded_feature_record_size(
    path_class: str, geometry_type: str, encoded_coordinates_size: int
) -> int:
    """Return exact JCS bytes for a feature from its coordinate-array byte size."""

    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("geometry_type must be Polygon or MultiPolygon")
    if (
        isinstance(encoded_coordinates_size, bool)
        or not isinstance(encoded_coordinates_size, int)
        or encoded_coordinates_size < 2
    ):
        raise ValueError("encoded_coordinates_size must describe a JSON array")
    return (
        _feature_record_envelope_size(
            require_path_class_name(path_class, "path_class"), geometry_type
        )
        + encoded_coordinates_size
    )


def encoded_feature_collection_size(
    encoded_feature_bytes: int, feature_count: int
) -> int:
    """Return exact JCS bytes for a collection from complete feature sizes."""

    for value, name in (
        (encoded_feature_bytes, "encoded_feature_bytes"),
        (feature_count, "feature_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if feature_count == 0:
        if encoded_feature_bytes != 0:
            raise ValueError("an empty collection cannot contain feature bytes")
        return _EMPTY_COLLECTION_BYTES
    if encoded_feature_bytes == 0:
        raise ValueError("a nonempty collection must contain feature bytes")
    return _EMPTY_COLLECTION_BYTES + encoded_feature_bytes + feature_count - 1


def validate_feature_collection(
    value: Mapping[str, Any], declared_classes: Collection[str]
) -> dict[str, Any]:
    """Validate a complete fixed feature collection and return its canonical tree.

    Validation is closed and verifies normalized geometry, deterministic IDs,
    response order, duplicate IDs, object type, and declared PathClasses.  An
    exact empty ``FeatureCollection`` is valid.
    """

    collection = _object(value, "feature_collection", _COLLECTION_FIELDS)
    if collection["type"] != "FeatureCollection":
        raise ValueError("feature_collection.type must be FeatureCollection")
    raw_features = collection["features"]
    if not isinstance(raw_features, list):
        raise ValueError("feature_collection.features must be an array")

    records = validate_feature_records(raw_features, declared_classes)
    return records.as_collection()


def validate_feature_records(
    raw_features: Iterable[Any], declared_classes: Collection[str]
) -> CanonicalFeatureRecords:
    """Validate every record and occurrence ID before returning any population.

    An iterable permits a stream reader to parse one record at a time without
    retaining a second, unvalidated feature population. The returned features
    have independent mutable geometry values, just as collection validation does.
    """

    declared = _declared_classes(declared_classes)
    normalized_features: list[dict[str, Any]] = []
    encoded_coordinates: list[bytes] = []
    vertex_count = 0
    feature_ids_by_key: dict[tuple[str, bytes], set[str]] = {}
    feature_ids: set[str] = set()
    prior_id: str | None = None
    for index, raw_feature in enumerate(raw_features):
        path = f"feature_collection.features[{index}]"
        feature = _object(raw_feature, path, _FEATURE_FIELDS)
        if feature["type"] != "Feature":
            raise ValueError(f"{path}.type must be Feature")

        feature_id = require_sha256(feature["id"], f"{path}.id")
        if feature_id in feature_ids:
            raise ValueError("feature_collection contains a duplicate feature id")
        if prior_id is not None and prior_id >= feature_id:
            raise ValueError(
                "feature_collection.features must be in ascending id order"
            )
        feature_ids.add(feature_id)
        prior_id = feature_id

        geometry_value = _object(
            feature["geometry"], f"{path}.geometry", _GEOMETRY_FIELDS
        )
        geometry = canonical_geometry(geometry_value)
        if geometry_value != geometry:
            raise ValueError(f"{path}.geometry must use canonical coordinates")

        properties = _object(
            feature["properties"], f"{path}.properties", _PROPERTY_FIELDS
        )
        if properties["object_type"] != "DETECTION":
            raise ValueError(f"{path}.properties.object_type must be DETECTION")
        path_class = _emitted_class(
            properties["path_class"], declared, f"{path}.properties"
        )
        coordinates = canonical_bytes(geometry["coordinates"])
        key = (path_class, coordinates)
        feature_ids_by_key.setdefault(key, set()).add(feature_id)
        encoded_coordinates.append(coordinates)
        vertex_count += _geometry_vertex_count(geometry)
        normalized_features.append(
            {
                "type": "Feature",
                "id": feature_id,
                "geometry": geometry,
                "properties": {
                    "object_type": "DETECTION",
                    "path_class": path_class,
                },
            }
        )

    for key, observed_ids in feature_ids_by_key.items():
        path_class, geometry_bytes = key
        expected_ids = {
            feature_id_from_encoded_coordinates(
                path_class, geometry_bytes, occurrence_index
            )
            for occurrence_index in range(len(observed_ids))
        }
        if observed_ids != expected_ids:
            raise ValueError(
                "feature_collection ids do not match their class and geometry"
            )
    return CanonicalFeatureRecords(
        normalized_features, encoded_coordinates, vertex_count
    )


def _feature_id(
    path_class: str, geometry: Mapping[str, Any], occurrence_index: int
) -> str:
    preimage = [path_class, geometry["coordinates"], occurrence_index]
    return sha256_bytes(canonical_bytes(preimage))


def feature_id_from_encoded_coordinates(
    path_class: str, encoded_coordinates: bytes, occurrence_index: int
) -> str:
    """Derive an occurrence ID from already validated canonical coordinates."""

    preimage = b"".join(
        (
            b"[",
            canonical_bytes(path_class),
            b",",
            encoded_coordinates,
            b",",
            canonical_bytes(occurrence_index),
            b"]",
        )
    )
    return sha256_bytes(preimage)


def _feature(
    path_class: str, geometry: Mapping[str, Any], occurrence_index: int
) -> dict[str, Any]:
    return _feature_with_id(
        path_class, geometry, _feature_id(path_class, geometry, occurrence_index)
    )


def _feature_from_encoded_coordinates(
    path_class: str,
    geometry: Mapping[str, Any],
    encoded_coordinates: bytes,
    occurrence_index: int,
) -> dict[str, Any]:
    return _feature_with_id(
        path_class,
        geometry,
        feature_id_from_encoded_coordinates(
            path_class, encoded_coordinates, occurrence_index
        ),
    )


def _feature_with_id(
    path_class: str, geometry: Mapping[str, Any], feature_id: str
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": geometry,
        "properties": {
            "object_type": "DETECTION",
            "path_class": path_class,
        },
    }


@cache
def _feature_record_envelope_size(path_class: str, geometry_type: str) -> int:
    # Every real feature ID is a lowercase 64-character SHA-256 value.
    return len(encode_feature_record("0" * 64, path_class, geometry_type, b"[]")) - 2


def encode_feature_record(
    feature_id: str,
    path_class: str,
    geometry_type: str,
    encoded_coordinates: bytes,
) -> bytes:
    """Encode the fixed feature envelope without reparsing canonical coordinates.

    This is an internal encoder, not a validator. Coordinates must already be
    validated, and the ID must come from feature_id_from_encoded_coordinates.
    Keys have canonical JSON order; the shared JSON encoder escapes scalar values.
    """

    geometry_suffix, feature_suffix = _feature_record_suffixes(
        path_class, geometry_type
    )
    return b"".join(
        (
            b'{"geometry":{"coordinates":',
            encoded_coordinates,
            geometry_suffix,
            feature_id.encode("ascii"),
            feature_suffix,
        )
    )


@cache
def _feature_record_suffixes(
    path_class: str, geometry_type: str
) -> tuple[bytes, bytes]:
    return (
        b',"type":' + canonical_bytes(geometry_type) + b'},"id":"',
        b'","properties":{"object_type":"DETECTION","path_class":'
        + canonical_bytes(path_class)
        + b'},"type":"Feature"}',
    )


def _geometry_vertex_count(geometry: Mapping[str, Any]) -> int:
    coordinates = geometry["coordinates"]
    polygons = [coordinates] if geometry["type"] == "Polygon" else coordinates
    return sum(len(ring) - 1 for polygon in polygons for ring in polygon)


def _declared_classes(values: Collection[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("declared_classes must be a collection of PathClass names")
    result: set[str] = set()
    for index, value in enumerate(values):
        path_class = require_path_class_name(value, f"declared_classes[{index}]")
        if path_class in result:
            raise ValueError(f"duplicate declared PathClass: {path_class}")
        result.add(path_class)
    return frozenset(result)


def _emitted_class(value: Any, declared: frozenset[str], path: str) -> str:
    path_class = require_path_class_name(value, f"{path}.path_class")
    if path_class not in declared:
        raise ValueError(f"{path}.path_class is not declared by the model")
    return path_class


def _object(value: Any, path: str, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    actual = frozenset(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise ValueError(
            f"{path} has missing fields {missing} and unknown fields {unknown}"
        )
    return value
