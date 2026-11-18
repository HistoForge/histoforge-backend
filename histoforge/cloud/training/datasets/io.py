"""Canonical package-member IO shared by dataset admission and training.

These readers do not publish or retain datasets. Execution callers obtain a
verified RetainedTrainingDataset from TrainingDatasetFiles first. A source
handle is scoped to one operation and never stored in a DataLoader sample.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import openslide
from shapely.geometry import shape

from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
    require_path_class_name,
)
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.contract import validate_source_artifact_origin
from histoforge.cloud.sources.original import OPENSLIDE_ADAPTER
from histoforge.cloud.sources.reader import SourceReader
from histoforge.cloud.training.datasets.manifest import (
    artifact_origin_path,
    artifact_source_path,
    roi_geometry_path,
    roi_objects_path,
)

_FEATURE_FIELDS = {"type", "geometry", "properties"}
_COLLECTION_FIELDS = {"type", "features"}
_OBJECT_PROPERTIES = {"source_object_uuid", "object_type", "path_class"}
_TRAINING_OBJECT_TYPES = frozenset({"annotation", "detection"})
_NAMESPACE = uuid.UUID("6193a38e-0f5d-4889-97be-65d747f0fbef")


@contextmanager
def open_training_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    artifact_index: int,
    max_json_member_bytes: int,
    source_limits: Mapping[str, int],
) -> Iterator[tuple[dict[str, Any], SourceReader]]:
    """Open one package image through the existing validated source reader.

    The temporary start declaration adapts package provenance to SourceReader's
    existing input; it is never persisted or submitted to the source-upload
    lifecycle. Admission and execution use the same declaration.
    """

    artifact = manifest["artifacts"][artifact_index]
    origin = read_training_artifact_origin(root, artifact_index, max_json_member_bytes)
    representation = artifact["representation"]
    expected_origin = (
        "original-wsi" if representation == "original" else "derived-region"
    )
    if origin["type"] != expected_origin:
        raise ValueError("training artifact representation disagrees with its origin")

    source_path = root / artifact_source_path(artifact_index, representation)
    total_bytes = source_path.stat().st_size
    if representation == "original":
        _require_training_original(source_path, origin)
        series = origin["pyramid_series_index"]
        required_levels = [
            {
                "series_index": series,
                "resolution_level": 0,
                "width": origin["source"]["width"],
                "height": origin["source"]["height"],
            }
        ]
        container = "SLIDE_FILE"
    else:
        rectangle = origin["source_rect_at_level"]
        required_levels = [
            {
                "series_index": 0,
                "resolution_level": 0,
                "width": rectangle["width"],
                "height": rectangle["height"],
            }
        ]
        container = "PNG" if representation == "png" else "TIFF_FAMILY"

    start = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "creation_action_id": str(
            uuid.uuid5(_NAMESPACE, f"{manifest['dataset_digest']}:capture")
        ),
        "artifact_id": str(
            uuid.uuid5(_NAMESPACE, f"{manifest['dataset_digest']}:{artifact_index}")
        ),
        "container": container,
        "total_bytes": total_bytes,
        "chunk_size_bytes": min(total_bytes, source_limits["max_chunk_bytes"]),
        "origin": origin,
        "required_levels": required_levels,
    }
    if (
        representation == "ome_bigtiff"
        and required_levels[0]["width"] * required_levels[0]["height"]
        <= source_limits["max_complete_decode_pixels"]
    ):
        raise ValueError("small derived artifacts must use lossless PNG")
    with SourceReader(source_path, start, source_limits) as reader:
        yield origin, reader


def _require_training_original(path: Path, origin: Mapping[str, Any]) -> None:
    """Keep training originals on its qualified single-file OpenSlide TIFF path."""

    source = origin["source"]
    if (
        source["source_adapter_id"] != OPENSLIDE_ADAPTER
        or source["content_version"]["type"] != "local-file"
        or "members" in origin
        or openslide.OpenSlide.detect_format(path) != "generic-tiff"
    ):
        raise ValueError("training originals require one OpenSlide generic-TIFF file")
    settings = origin["reader_settings"]
    if settings["background_rgb"] != [0, 0, 0] or any(
        not float(level["pixels_to_level_zero"]["scale_x"]).is_integer()
        for level in settings["levels"]
    ):
        raise ValueError(
            "training originals require black background and integer downsamples"
        )


def read_training_artifact_origin(
    root: Path, artifact_index: int, max_json_member_bytes: int
) -> dict[str, Any]:
    """Read source geometry without opening or decoding its image payload."""

    return _read_canonical(
        root / artifact_origin_path(artifact_index),
        validate_source_artifact_origin,
        max_json_member_bytes,
    )


def read_training_roi_geometry(
    root: Path, roi_index: int, max_json_member_bytes: int
) -> dict[str, Any]:
    """Read an axis-aligned administrative rectangle without reshaping it."""

    def validate(value: Any) -> dict[str, Any]:
        item = _exact_object(value, _FEATURE_FIELDS, "training ROI feature")
        if item["type"] != "Feature" or item["properties"] != {}:
            raise ValueError("training ROI must be a Feature with empty properties")
        geometry = canonical_geometry(item["geometry"])
        boundary = shape(geometry)
        if not boundary.equals(boundary.envelope):
            raise ValueError("training ROI must be an axis-aligned rectangle")
        return {
            "type": "Feature",
            "geometry": geometry,
            "properties": {},
        }

    return _read_canonical(
        root / roi_geometry_path(roi_index), validate, max_json_member_bytes
    )["geometry"]


@mounted_artifact_volume_operation
def read_training_roi_source(
    root: Path,
    manifest: Mapping[str, Any],
    roi_index: int,
    max_json_member_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one ROI and its origin under the Volume gate, without image IO."""

    artifact_index = manifest["rois"][roi_index]["source_envelope"]["artifact_index"]
    return (
        read_training_roi_geometry(root, roi_index, max_json_member_bytes),
        read_training_artifact_origin(root, artifact_index, max_json_member_bytes),
    )


def read_training_objects(
    root: Path,
    roi_index: int,
    roi_geometry: Mapping[str, Any],
    class_map: Mapping[str, str],
    max_json_member_bytes: int,
) -> list[dict[str, Any]]:
    """Validate original-name bytes, then map detached targets exactly once."""

    roi_shape = shape(roi_geometry)

    def validate(value: Any) -> dict[str, Any]:
        item = _exact_object(value, _COLLECTION_FIELDS, "training objects")
        if item["type"] != "FeatureCollection" or not isinstance(
            item["features"], list
        ):
            raise ValueError("training objects must be a FeatureCollection")
        features = []
        for value_feature in item["features"]:
            feature = _exact_object(
                value_feature, _FEATURE_FIELDS, "training object feature"
            )
            if feature["type"] != "Feature":
                raise ValueError("training object must have type Feature")
            properties = _exact_object(
                feature["properties"], _OBJECT_PROPERTIES, "training object properties"
            )
            source_id = require_canonical_uuid(
                properties["source_object_uuid"], "source_object_uuid"
            )
            object_type = properties["object_type"]
            if object_type not in _TRAINING_OBJECT_TYPES:
                raise ValueError("training object type must be annotation or detection")
            path_class = require_path_class_name(properties["path_class"], "path_class")
            if path_class not in class_map:
                raise ValueError("training object PathClass has no mapped target class")
            geometry = canonical_geometry(feature["geometry"])
            if shape(geometry).intersection(roi_shape).area <= 0:
                raise ValueError(
                    "training object has no positive-area ROI intersection"
                )
            features.append(
                {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "source_object_uuid": source_id,
                        "object_type": object_type,
                        "path_class": path_class,
                    },
                }
            )
        ids = [item["properties"]["source_object_uuid"] for item in features]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("training objects must be unique and sorted by UUID")
        return {"type": "FeatureCollection", "features": features}

    features = _read_canonical(
        root / roi_objects_path(roi_index), validate, max_json_member_bytes
    )["features"]
    for feature in features:
        properties = feature["properties"]
        properties["path_class"] = class_map[properties["path_class"]]
    return features


@mounted_artifact_volume_operation
def read_training_roi_labels(
    root: Path,
    manifest: Mapping[str, Any],
    roi_index: int,
    max_json_member_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read one ROI's labels and origin, releasing the gate before rasterization."""

    roi = read_training_roi_geometry(root, roi_index, max_json_member_bytes)
    artifact_index = manifest["rois"][roi_index]["source_envelope"]["artifact_index"]
    origin = read_training_artifact_origin(root, artifact_index, max_json_member_bytes)
    objects = read_training_objects(
        root,
        roi_index,
        roi,
        manifest["task"]["annotation_class_map"],
        max_json_member_bytes,
    )
    return roi, objects, origin


def _read_canonical(
    path: Path,
    validator: Callable[[Any], dict[str, Any]],
    maximum_bytes: int,
) -> dict[str, Any]:
    with path.open("rb") as stream:
        encoded = stream.read(maximum_bytes + 1)
        if len(encoded) > maximum_bytes or stream.read(1):
            raise ValueError(
                f"training package JSON exceeds its byte limit: {path.name}"
            )
    value = validator(parse_json(encoded))
    if encoded != canonical_bytes(value):
        raise ValueError(f"training package JSON is not canonical: {path.name}")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value
