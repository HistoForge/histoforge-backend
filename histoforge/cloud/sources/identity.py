"""Credential-free image-source identity and spatial primitives.

These pure validators are shared by source-artifact reservations and later
source-view/raster contracts. They normalize wire values but do not inspect a
live QuPath source or decide which adapter identifiers are supported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_finite_number,
    require_nonblank_string,
    require_nonnegative_int32,
    require_nonnegative_safe_integer,
    require_positive_finite_number,
    require_positive_int32,
    require_positive_safe_integer,
    require_safe_integer,
    require_sha256,
)
from histoforge.cloud.sources.file_sets import (
    MAX_MEMBERS,
    member_order,
    relative_slide_path,
    validate_member_paths,
)

_LOCAL_FILE_FIELDS = {"type", "size_bytes", "last_modified_epoch_millis"}
_LOCAL_FILE_FIELDS_WITH_KEY = _LOCAL_FILE_FIELDS | {"file_key"}
_OPAQUE_VERSION_FIELDS = {"type", "value"}
_S3_ETAG_FIELDS = {"type", "quoted_value"}
_CALIBRATION_FIELDS = {"mpp_x", "mpp_y"}
_SOURCE_FIELDS = {
    "source_adapter_id",
    "credential_free_descriptor_digest",
    "width",
    "height",
    "z_slices",
    "timepoints",
    "calibration",
}
_SOURCE_FIELDS_WITH_VERSION = _SOURCE_FIELDS | {"content_version"}
_TRANSFORM_FIELDS = {"scale_x", "scale_y", "translate_x", "translate_y"}
_PLANE_FIELDS = {"z", "t"}
_RECT_FIELDS = {"x", "y", "width", "height"}


def validate_source_content_version(value: Any) -> dict[str, Any]:
    """Validate stable local-file or S3 source-version evidence."""

    version = _require_object(value, "source content version")
    version_type = _required_string(version, "type", "source content version")
    if version_type == "local-file-set":
        _require_fields(version, {"type", "members"}, "local-file-set version")
        raw = version["members"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_MEMBERS:
            raise ValueError("local-file-set members must be a bounded nonempty array")
        members = []
        for item in raw:
            member = _require_object(item, "local-file-set member")
            fields = {"path", "size_bytes", "last_modified_epoch_millis"}
            _require_fields(
                member,
                fields | ({"file_key"} if "file_key" in member else set()),
                "local-file-set member",
            )
            result = {
                "path": relative_slide_path(member["path"]),
                "size_bytes": require_nonnegative_safe_integer(
                    member["size_bytes"], "member.size_bytes"
                ),
                "last_modified_epoch_millis": require_safe_integer(
                    member["last_modified_epoch_millis"],
                    "member.last_modified_epoch_millis",
                ),
            }
            if "file_key" in member:
                result["file_key"] = require_nonblank_string(
                    member["file_key"], "file_key"
                )
            members.append(result)
        validate_member_paths([member["path"] for member in members])
        return canonical_tree(
            {
                "type": version_type,
                "members": sorted(members, key=lambda item: member_order(item["path"])),
            }
        )
    if version_type == "local-file":
        _require_fields(
            version,
            _LOCAL_FILE_FIELDS_WITH_KEY
            if "file_key" in version
            else _LOCAL_FILE_FIELDS,
            "local-file source content version",
        )
        result: dict[str, Any] = {
            "type": version_type,
            "size_bytes": require_positive_safe_integer(
                version["size_bytes"], "size_bytes"
            ),
            "last_modified_epoch_millis": require_safe_integer(
                version["last_modified_epoch_millis"],
                "last_modified_epoch_millis",
            ),
        }
        if "file_key" in version:
            result["file_key"] = require_nonblank_string(
                version["file_key"], "file_key"
            )
        return canonical_tree(result)
    if version_type == "s3-version-id":
        _require_fields(version, _OPAQUE_VERSION_FIELDS, "S3 version ID")
        return {
            "type": version_type,
            "value": require_nonblank_string(version["value"], "value"),
        }
    if version_type == "s3-etag":
        _require_fields(version, _S3_ETAG_FIELDS, "S3 ETag")
        quoted = require_nonblank_string(version["quoted_value"], "quoted_value")
        if len(quoted) < 3 or quoted[0] != '"' or quoted[-1] != '"':
            raise ValueError("quoted_value must be a quoted S3 ETag")
        return {"type": version_type, "quoted_value": quoted}
    raise ValueError(f"unsupported source content version type: {version_type!r}")


def validate_pixel_calibration(value: Any) -> dict[str, float]:
    """Validate independent positive source-pixel calibration axes."""

    calibration = _require_exact_object(value, _CALIBRATION_FIELDS, "pixel calibration")
    return {
        "mpp_x": _positive_finite(calibration["mpp_x"], "mpp_x"),
        "mpp_y": _positive_finite(calibration["mpp_y"], "mpp_y"),
    }


def validate_image_source_identity(value: Any) -> dict[str, Any]:
    """Validate one credential-free pixel-source identity and dimensions."""

    source = _require_object(value, "image source identity")
    _require_fields(
        source,
        _SOURCE_FIELDS_WITH_VERSION if "content_version" in source else _SOURCE_FIELDS,
        "image source identity",
    )
    result = {
        "source_adapter_id": require_nonblank_string(
            source["source_adapter_id"], "source_adapter_id"
        ),
        "credential_free_descriptor_digest": require_sha256(
            source["credential_free_descriptor_digest"],
            "credential_free_descriptor_digest",
        ),
        "width": require_positive_int32(source["width"], "width"),
        "height": require_positive_int32(source["height"], "height"),
        "z_slices": require_positive_int32(source["z_slices"], "z_slices"),
        "timepoints": require_positive_int32(source["timepoints"], "timepoints"),
        "calibration": validate_pixel_calibration(source["calibration"]),
    }
    if "content_version" in source:
        result["content_version"] = validate_source_content_version(
            source["content_version"]
        )
    return canonical_tree(result)


def validate_scale_translate_2d(value: Any) -> dict[str, float]:
    """Validate a finite axis-aligned transform with positive scales."""

    transform = _require_exact_object(value, _TRANSFORM_FIELDS, "scale transform")
    return {
        "scale_x": _positive_finite(transform["scale_x"], "scale_x"),
        "scale_y": _positive_finite(transform["scale_y"], "scale_y"),
        "translate_x": _finite(transform["translate_x"], "translate_x"),
        "translate_y": _finite(transform["translate_y"], "translate_y"),
    }


def validate_image_plane_index(value: Any) -> dict[str, int]:
    """Validate one nonnegative source z/t plane index."""

    plane = _require_exact_object(value, _PLANE_FIELDS, "image plane index")
    return {
        "z": require_nonnegative_int32(plane["z"], "z"),
        "t": require_nonnegative_int32(plane["t"], "t"),
    }


def validate_int_rect(value: Any) -> dict[str, int]:
    """Validate one nonnegative half-open rectangle with positive extents."""

    rectangle = _require_exact_object(value, _RECT_FIELDS, "integer rectangle")
    return {
        "x": require_nonnegative_int32(rectangle["x"], "x"),
        "y": require_nonnegative_int32(rectangle["y"], "y"),
        "width": require_positive_int32(rectangle["width"], "width"),
        "height": require_positive_int32(rectangle["height"], "height"),
    }


def _finite(value: Any, path: str) -> float:
    return require_finite_number(value, path)


def _positive_finite(value: Any, path: str) -> float:
    return require_positive_finite_number(value, path)


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
