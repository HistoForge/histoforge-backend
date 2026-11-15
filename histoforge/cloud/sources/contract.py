"""Validate source-upload reservations shared with the QuPath extension.

These contracts describe source identity, start requests, acceptance, and byte
ranges. Source services own storage, transport, and deployment admission limits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonblank_string,
    require_nonnegative_int32,
    require_positive_int32,
    require_positive_safe_integer,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.file_sets import relative_slide_path, validate_members
from histoforge.cloud.sources.identity import (
    validate_image_plane_index,
    validate_image_source_identity,
    validate_int_rect,
    validate_scale_translate_2d,
)
from histoforge.cloud.sources.original import validate_reader_settings

_ORIGINAL_FIELDS = {
    "type",
    "source",
    "pyramid_series_index",
    "level_zero_pixels_to_slide",
    "entry_path",
    "reader_settings",
}
_DERIVED_FIELDS = {
    "type",
    "source",
    "source_resolution_level",
    "plane",
    "source_rect_at_level",
    "artifact_pixels_to_slide",
    "writer_build",
}
_LEVEL_FIELDS = {"series_index", "resolution_level", "width", "height"}
_START_FIELDS = {
    "schema_version",
    "creation_action_id",
    "artifact_id",
    "container",
    "total_bytes",
    "chunk_size_bytes",
    "origin",
    "required_levels",
}
_ACCEPTED_FIELDS = {
    "schema_version",
    "creation_action_id",
    "artifact_id",
    "start_request_digest",
    "absolute_expires_at",
}
SOURCE_CONTAINERS = frozenset({"PNG", "TIFF_FAMILY", "SLIDE_FILE", "SLIDE_FILE_SET"})


def validate_source_container(value: Any) -> str:
    """Validate one closed encoded-image family."""

    container = require_nonblank_string(value, "container")
    if container not in SOURCE_CONTAINERS:
        raise ValueError(f"unsupported source container: {container!r}")
    return container


def validate_source_artifact_origin(value: Any) -> dict[str, Any]:
    """Validate original-WSI or derived-region provenance and slide mapping."""

    origin = _require_object(value, "source artifact origin")
    origin_type = _required_string(origin, "type", "source artifact origin")
    if origin_type == "original-wsi":
        _require_fields(
            origin,
            _ORIGINAL_FIELDS | ({"members"} if "members" in origin else set()),
            "original WSI origin",
        )
        source = validate_image_source_identity(origin["source"])
        if "content_version" not in source:
            raise ValueError("original WSI source must have a stable content_version")
        result = {
            "type": origin_type,
            "source": source,
            "pyramid_series_index": require_nonnegative_int32(
                origin["pyramid_series_index"], "pyramid_series_index"
            ),
            "level_zero_pixels_to_slide": validate_scale_translate_2d(
                origin["level_zero_pixels_to_slide"]
            ),
            "entry_path": relative_slide_path(origin["entry_path"]),
            "reader_settings": validate_reader_settings(
                origin["reader_settings"], source["source_adapter_id"]
            ),
        }
        if "members" in origin:
            result["members"] = validate_members(origin["members"])
        return canonical_tree(result)
    if origin_type == "derived-region":
        _require_fields(origin, _DERIVED_FIELDS, "derived region origin")
        source = validate_image_source_identity(origin["source"])
        plane = validate_image_plane_index(origin["plane"])
        if plane != {"z": 0, "t": 0}:
            raise ValueError("source artifacts support only plane z=0, t=0")
        rectangle = validate_int_rect(origin["source_rect_at_level"])
        return canonical_tree(
            {
                "type": origin_type,
                "source": source,
                "source_resolution_level": require_nonnegative_int32(
                    origin["source_resolution_level"],
                    "source_resolution_level",
                ),
                "plane": plane,
                "source_rect_at_level": rectangle,
                "artifact_pixels_to_slide": validate_scale_translate_2d(
                    origin["artifact_pixels_to_slide"]
                ),
                "writer_build": parse_software_build_identity(origin["writer_build"]),
            }
        )
    raise ValueError(f"unsupported source artifact origin type: {origin_type!r}")


def validate_source_level_requirement(value: Any) -> dict[str, int]:
    """Validate one decoded source-level shape required at seal."""

    level = _require_exact_object(value, _LEVEL_FIELDS, "source level requirement")
    return {
        "series_index": require_nonnegative_int32(
            level["series_index"], "series_index"
        ),
        "resolution_level": require_nonnegative_int32(
            level["resolution_level"], "resolution_level"
        ),
        "width": require_positive_int32(level["width"], "width"),
        "height": require_positive_int32(level["height"], "height"),
    }


def validate_source_artifact_start_request(value: Any) -> dict[str, Any]:
    """Validate and normalize one replayable immutable-upload reservation."""

    request = _require_exact_object(value, _START_FIELDS, "source artifact start")
    if (
        require_int32(request["schema_version"], "schema_version")
        != SOURCE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported source artifact schema_version")
    artifact_id = require_canonical_uuid(request["artifact_id"], "artifact_id")
    container = validate_source_container(request["container"])
    total_bytes = require_positive_safe_integer(request["total_bytes"], "total_bytes")
    chunk_size = require_positive_int32(request["chunk_size_bytes"], "chunk_size_bytes")
    origin = validate_source_artifact_origin(request["origin"])
    levels = _normalize_required_levels(request["required_levels"])
    _validate_origin_requirements(container, total_bytes, origin, levels)
    canonical_chunk_count(total_bytes, chunk_size)
    return canonical_tree(
        {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "creation_action_id": require_canonical_uuid(
                request["creation_action_id"], "creation_action_id"
            ),
            "artifact_id": artifact_id,
            "container": container,
            "total_bytes": total_bytes,
            "chunk_size_bytes": chunk_size,
            "origin": origin,
            "required_levels": levels,
        }
    )


def source_artifact_start_digest(value: Any) -> str:
    """Return the JCS digest that fences idempotent start under an artifact ID."""

    return canonical_digest(validate_source_artifact_start_request(value))


def validate_source_artifact_accepted(value: Any) -> dict[str, Any]:
    """Validate the backend's durable reservation acknowledgement."""

    accepted = _require_exact_object(
        value, _ACCEPTED_FIELDS, "source artifact acceptance"
    )
    if (
        require_int32(accepted["schema_version"], "schema_version")
        != SOURCE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported source artifact schema_version")
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "creation_action_id": require_canonical_uuid(
            accepted["creation_action_id"], "creation_action_id"
        ),
        "artifact_id": require_canonical_uuid(accepted["artifact_id"], "artifact_id"),
        "start_request_digest": require_sha256(
            accepted["start_request_digest"], "start_request_digest"
        ),
        "absolute_expires_at": require_rfc3339_millis(
            accepted["absolute_expires_at"], "absolute_expires_at"
        ),
    }


def validate_source_artifact_acceptance(request: Any, accepted: Any) -> dict[str, Any]:
    """Require an acceptance to echo and bind the exact normalized start request."""

    start = validate_source_artifact_start_request(request)
    result = validate_source_artifact_accepted(accepted)
    if result["creation_action_id"] != start["creation_action_id"]:
        raise ValueError("acceptance creation_action_id does not match request")
    if result["artifact_id"] != start["artifact_id"]:
        raise ValueError("acceptance artifact_id does not match request")
    if result["start_request_digest"] != canonical_digest(start):
        raise ValueError("acceptance start_request_digest does not match request")
    return result


def canonical_chunk_count(total_bytes: Any, chunk_size_bytes: Any) -> int:
    """Return the number of ranges in the unique contiguous byte cover."""

    total = require_positive_safe_integer(total_bytes, "total_bytes")
    chunk_size = require_positive_int32(chunk_size_bytes, "chunk_size_bytes")
    count = (total + chunk_size - 1) // chunk_size
    if count > 2**31 - 1:
        raise ValueError("canonical chunk count exceeds SourceByteChunkId range")
    return count


def canonical_chunk_range(
    total_bytes: Any, chunk_size_bytes: Any, chunk_id: Any
) -> dict[str, int]:
    """Return one canonical range without materializing the complete plan."""

    total = require_positive_safe_integer(total_bytes, "total_bytes")
    chunk_size = require_positive_int32(chunk_size_bytes, "chunk_size_bytes")
    count = canonical_chunk_count(total, chunk_size)
    index = require_nonnegative_int32(chunk_id, "chunk_id")
    if index >= count:
        raise ValueError("chunk_id lies outside the canonical chunk plan")
    offset = index * chunk_size
    return {
        "chunk_id": index,
        "offset": offset,
        "size": min(chunk_size, total - offset),
    }


def _normalize_required_levels(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list) or not value:
        raise ValueError("required_levels must be a nonempty array")
    levels = [validate_source_level_requirement(level) for level in value]
    levels.sort(key=lambda level: (level["series_index"], level["resolution_level"]))
    keys = [(level["series_index"], level["resolution_level"]) for level in levels]
    if len(keys) != len(set(keys)):
        raise ValueError("required_levels contains a duplicate series/level key")
    return levels


def _validate_origin_requirements(
    container: str,
    total_bytes: int,
    origin: Mapping[str, Any],
    levels: list[dict[str, int]],
) -> None:
    if origin["type"] == "original-wsi":
        if container not in {"SLIDE_FILE", "SLIDE_FILE_SET"}:
            raise ValueError("original WSI must use SLIDE_FILE or SLIDE_FILE_SET")
        series = origin["pyramid_series_index"]
        if any(level["series_index"] != series for level in levels):
            raise ValueError("original WSI requirements must use pyramid_series_index")
        source = origin["source"]
        content_version = source["content_version"]
        if container == "SLIDE_FILE_SET":
            if "members" not in origin or content_version["type"] != "local-file-set":
                raise ValueError(
                    "SLIDE_FILE_SET requires member bytes and member versions"
                )
            sizes = {
                member["path"]: member["size_bytes"] for member in origin["members"]
            }
            versions = {
                member["path"]: member["size_bytes"]
                for member in content_version["members"]
            }
            if sizes != versions or sizes.get(origin["entry_path"], 0) <= 0:
                raise ValueError(
                    "fileset entry/membership/sizes differ from source version"
                )
        elif (
            "members" in origin
            or content_version["type"] != "local-file"
            or "/" in origin["entry_path"]
        ):
            raise ValueError(
                "SLIDE_FILE requires one unchanged local file name/version"
            )
        if (
            content_version["type"] == "local-file"
            and content_version["size_bytes"] != total_bytes
        ):
            raise ValueError("local-file size_bytes must equal total_bytes")
        if not any(
            level["resolution_level"] == 0
            and level["width"] == source["width"]
            and level["height"] == source["height"]
            for level in levels
        ):
            raise ValueError("original WSI requirements must contain exact level zero")
        configured = origin["reader_settings"]["levels"]
        if any(level["resolution_level"] >= len(configured) for level in levels):
            raise ValueError("required source level has no captured reader mapping")
        return

    if container not in {"PNG", "TIFF_FAMILY"}:
        raise ValueError("derived regions must use PNG or TIFF_FAMILY")
    if len(levels) != 1:
        raise ValueError("derived region must have exactly one required level")
    level = levels[0]
    rectangle = origin["source_rect_at_level"]
    if (
        level["series_index"] != 0
        or level["resolution_level"] != 0
        or level["width"] != rectangle["width"]
        or level["height"] != rectangle["height"]
    ):
        raise ValueError("derived region requires exact series-zero level-zero shape")


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    result = _require_object(value, path)
    _require_fields(result, fields, path)
    return result


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise ValueError(
            f"{path} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _required_string(value: Mapping[str, Any], field: str, path: str) -> str:
    if field not in value:
        raise ValueError(f"{path} is missing {field}")
    return require_nonblank_string(value[field], field)
