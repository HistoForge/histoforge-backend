"""Immutable identity of the source bytes accepted for one training run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_safe_integer,
    require_positive_safe_integer,
    require_sha256,
)
from histoforge.cloud.inference.identity import validate_module_content_identity
from histoforge.cloud.modules.files import (
    normalize_relative_path,
    require_unique_casefold_paths,
)

_FILE_FIELDS = {"relative_path", "size_bytes", "sha256"}
_READY_FIELDS = {
    "schema_version",
    "run_id",
    "request_digest",
    "source_module",
    "starting_weight_path",
    "starting_weight_digest",
    "source_files",
    "report_digest",
}


def validate_training_source_ready_marker(
    value: Any,
    max_marker_bytes: Any,
) -> dict[str, Any]:
    """Validate the complete source inventory published before run acceptance."""

    maximum = require_positive_safe_integer(max_marker_bytes, "max_marker_bytes")
    item = _exact_object(value, _READY_FIELDS, "training source ready marker")
    if require_int32(item["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported training source ready marker schema_version")

    values = item["source_files"]
    if not isinstance(values, list) or not values:
        raise ValueError("source_files must be a nonempty array")
    files = [_file_identity(entry) for entry in values]
    if files != sorted(files, key=lambda entry: entry["relative_path"]):
        raise ValueError("source_files must be sorted by relative_path")
    paths = [entry["relative_path"] for entry in files]
    require_unique_casefold_paths(paths, "training source files")

    starting_path = _source_path(item["starting_weight_path"], "starting_weight_path")
    starting_digest = require_sha256(
        item["starting_weight_digest"], "starting_weight_digest"
    )
    matches = [
        entry
        for entry in files
        if entry["relative_path"] == starting_path
        and entry["sha256"] == starting_digest
    ]
    if len(matches) != 1:
        raise ValueError("starting weights do not match the source-file inventory")

    marker = canonical_tree(
        {
            "schema_version": 1,
            "run_id": require_canonical_uuid(item["run_id"], "run_id"),
            "request_digest": require_sha256(item["request_digest"], "request_digest"),
            "source_module": validate_module_content_identity(item["source_module"]),
            "starting_weight_path": starting_path,
            "starting_weight_digest": starting_digest,
            "source_files": files,
            "report_digest": require_sha256(item["report_digest"], "report_digest"),
        }
    )
    if len(canonical_bytes(marker)) > maximum:
        raise ValueError("training source ready marker exceeds its byte limit")
    return marker


def _file_identity(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _FILE_FIELDS, "training source file identity")
    return {
        "relative_path": _source_path(item["relative_path"], "relative_path"),
        "size_bytes": require_nonnegative_safe_integer(
            item["size_bytes"], "size_bytes"
        ),
        "sha256": require_sha256(item["sha256"], "sha256"),
    }


def _source_path(value: Any, path: str) -> str:
    normalized = normalize_relative_path(value)
    if not normalized.startswith("source/"):
        raise ValueError(f"{path} must be below the source directory")
    return normalized


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value
