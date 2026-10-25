"""Validate uploaded image references and completed-upload publication.

Check chunk acknowledgements, whole-file digests, reader-discovered image levels,
and immutable references against the upload reservation. These functions do not
read image bytes, store records, change upload state, or make HTTP requests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_positive_int32,
    require_positive_safe_integer,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    validate_source_artifact_acceptance,
    validate_source_artifact_start_request,
    validate_source_container,
)

_CHUNK_COMMIT_FIELDS = {"chunk_id", "sha256"}
_CHUNK_ACCEPTED_FIELDS = {"schema_version", "artifact_id", "chunk"}
_SEAL_REQUEST_FIELDS = {"bytes_digest"}
_LEVEL_FIELDS = {"series_index", "resolution_level", "width", "height"}
_MANIFEST_PREIMAGE_FIELDS = {
    "schema_version",
    "artifact_id",
    "creation_action_id",
    "container",
    "total_bytes",
    "bytes_digest",
    "validated_levels",
}
_MANIFEST_FIELDS = _MANIFEST_PREIMAGE_FIELDS | {"manifest_digest"}
_REFERENCE_FIELDS = {
    "artifact_id",
    "creation_action_id",
    "start_request_digest",
    "bytes_digest",
    "manifest_digest",
    "container",
}
_SEALED_FIELDS = {"schema_version", "artifact", "absolute_expires_at"}


def validate_source_byte_chunk_commit(value: Any) -> dict[str, Any]:
    """Validate the immutable identity of one accepted raw file range."""

    commit = _require_exact_object(value, _CHUNK_COMMIT_FIELDS, "chunk commit")
    return {
        "chunk_id": require_nonnegative_int32(commit["chunk_id"], "chunk_id"),
        "sha256": require_sha256(commit["sha256"], "sha256"),
    }


def validate_source_byte_chunk_accepted(value: Any) -> dict[str, Any]:
    """Validate one backend chunk acknowledgement without external state."""

    accepted = _require_exact_object(value, _CHUNK_ACCEPTED_FIELDS, "chunk acceptance")
    _require_current_schema(accepted["schema_version"])
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": require_canonical_uuid(accepted["artifact_id"], "artifact_id"),
        "chunk": validate_source_byte_chunk_commit(accepted["chunk"]),
    }


def validate_source_byte_chunk_acceptance(
    start_request: Any, expected_commit: Any, accepted: Any
) -> dict[str, Any]:
    """Require a chunk acknowledgement to echo the expected upload range."""

    start = validate_source_artifact_start_request(start_request)
    expected = validate_source_byte_chunk_commit(expected_commit)
    result = validate_source_byte_chunk_accepted(accepted)
    if expected["chunk_id"] >= canonical_chunk_count(
        start["total_bytes"], start["chunk_size_bytes"]
    ):
        raise ValueError("chunk_id lies outside the canonical chunk plan")
    if result["artifact_id"] != start["artifact_id"]:
        raise ValueError("chunk acceptance artifact_id does not match request")
    if result["chunk"] != expected:
        raise ValueError("chunk acceptance does not match expected commit")
    return result


def validate_source_artifact_seal_request(value: Any) -> dict[str, str]:
    """Validate the requested identity of the reconstructed encoded file."""

    request = _require_exact_object(value, _SEAL_REQUEST_FIELDS, "seal request")
    return {"bytes_digest": require_sha256(request["bytes_digest"], "bytes_digest")}


def validate_validated_source_level(value: Any) -> dict[str, int]:
    """Validate one actual source level discovered by the deployed reader."""

    level = _require_exact_object(value, _LEVEL_FIELDS, "validated source level")
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


def build_source_artifact_manifest(
    start_request: Any, seal_request: Any, validated_levels: Sequence[Any]
) -> dict[str, Any]:
    """Build the self-digested manifest for one validated sealed file."""

    start = validate_source_artifact_start_request(start_request)
    seal = validate_source_artifact_seal_request(seal_request)
    levels = _normalize_validated_levels(validated_levels)
    _validate_level_coverage(start, levels)
    preimage = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": start["artifact_id"],
        "creation_action_id": start["creation_action_id"],
        "container": start["container"],
        "total_bytes": start["total_bytes"],
        "bytes_digest": seal["bytes_digest"],
        "validated_levels": levels,
    }
    return canonical_tree({**preimage, "manifest_digest": canonical_digest(preimage)})


def validate_source_artifact_manifest(value: Any) -> dict[str, Any]:
    """Validate and normalize one self-digested source-artifact manifest."""

    manifest = _require_exact_object(value, _MANIFEST_FIELDS, "source manifest")
    _require_current_schema(manifest["schema_version"])
    container = validate_source_container(manifest["container"])
    preimage = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": require_canonical_uuid(manifest["artifact_id"], "artifact_id"),
        "creation_action_id": require_canonical_uuid(
            manifest["creation_action_id"], "creation_action_id"
        ),
        "container": container,
        "total_bytes": require_positive_safe_integer(
            manifest["total_bytes"], "total_bytes"
        ),
        "bytes_digest": require_sha256(manifest["bytes_digest"], "bytes_digest"),
        "validated_levels": _normalize_validated_levels(manifest["validated_levels"]),
    }
    digest = require_sha256(manifest["manifest_digest"], "manifest_digest")
    if digest != canonical_digest(preimage):
        raise ValueError("manifest_digest does not match manifest fields")
    return canonical_tree({**preimage, "manifest_digest": digest})


def validate_source_artifact_publication(
    start_request: Any, seal_request: Any, manifest: Any
) -> dict[str, Any]:
    """Bind one manifest to the exact start and seal requests that produced it."""

    start = validate_source_artifact_start_request(start_request)
    seal = validate_source_artifact_seal_request(seal_request)
    result = validate_source_artifact_manifest(manifest)
    expected = build_source_artifact_manifest(start, seal, result["validated_levels"])
    if result != expected:
        raise ValueError("source manifest does not match start and seal requests")
    return result


def validate_source_artifact_ref(value: Any) -> dict[str, Any]:
    """Validate the compact immutable source identity accepted by inference."""

    reference = _require_exact_object(value, _REFERENCE_FIELDS, "source artifact ref")
    return {
        "artifact_id": require_canonical_uuid(reference["artifact_id"], "artifact_id"),
        "creation_action_id": require_canonical_uuid(
            reference["creation_action_id"], "creation_action_id"
        ),
        "start_request_digest": require_sha256(
            reference["start_request_digest"], "start_request_digest"
        ),
        "bytes_digest": require_sha256(reference["bytes_digest"], "bytes_digest"),
        "manifest_digest": require_sha256(
            reference["manifest_digest"], "manifest_digest"
        ),
        "container": validate_source_container(reference["container"]),
    }


def build_source_artifact_ref(start_request: Any, manifest: Any) -> dict[str, Any]:
    """Derive the only compact reference for one start request and manifest."""

    start = validate_source_artifact_start_request(start_request)
    result = validate_source_artifact_manifest(manifest)
    _require_manifest_start_match(start, result)
    return {
        "artifact_id": start["artifact_id"],
        "creation_action_id": start["creation_action_id"],
        "start_request_digest": canonical_digest(start),
        "bytes_digest": result["bytes_digest"],
        "manifest_digest": result["manifest_digest"],
        "container": result["container"],
    }


def validate_source_artifact_reference(
    start_request: Any, manifest: Any, reference: Any
) -> dict[str, Any]:
    """Require a supplied compact reference to equal the uniquely derived value."""

    result = validate_source_artifact_ref(reference)
    if result != build_source_artifact_ref(start_request, manifest):
        raise ValueError(
            "source artifact ref does not match start request and manifest"
        )
    return result


def validate_source_artifact_sealed(value: Any) -> dict[str, Any]:
    """Validate a seal response without consulting upload state."""

    sealed = _require_exact_object(value, _SEALED_FIELDS, "source artifact sealed")
    _require_current_schema(sealed["schema_version"])
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "artifact": validate_source_artifact_ref(sealed["artifact"]),
        "absolute_expires_at": require_rfc3339_millis(
            sealed["absolute_expires_at"], "absolute_expires_at"
        ),
    }


def validate_source_artifact_seal_response(
    start_request: Any,
    accepted: Any,
    seal_request: Any,
    manifest: Any,
    sealed: Any,
) -> dict[str, Any]:
    """Bind a seal response to its reservation, manifest, and fixed deadline."""

    start = validate_source_artifact_start_request(start_request)
    acceptance = validate_source_artifact_acceptance(start, accepted)
    publication = validate_source_artifact_publication(start, seal_request, manifest)
    result = validate_source_artifact_sealed(sealed)
    if result["artifact"] != build_source_artifact_ref(start, publication):
        raise ValueError("sealed artifact ref does not match publication")
    if result["absolute_expires_at"] != acceptance["absolute_expires_at"]:
        raise ValueError("sealed deadline does not match accepted deadline")
    return result


def _normalize_validated_levels(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("validated_levels must be a nonempty array")
    levels = [validate_validated_source_level(level) for level in value]
    levels.sort(key=lambda level: (level["series_index"], level["resolution_level"]))
    keys = [(level["series_index"], level["resolution_level"]) for level in levels]
    if len(keys) != len(set(keys)):
        raise ValueError("validated_levels contains a duplicate series/level key")
    return levels


def _validate_level_coverage(
    start: Mapping[str, Any], levels: list[dict[str, int]]
) -> None:
    required = {
        (level["series_index"], level["resolution_level"]): level
        for level in start["required_levels"]
    }
    actual = {
        (level["series_index"], level["resolution_level"]): level for level in levels
    }
    for key, requirement in required.items():
        if actual.get(key) != requirement:
            raise ValueError(
                "validated_levels do not satisfy required level dimensions"
            )

    origin = start["origin"]
    if origin["type"] == "original-wsi":
        series = origin["pyramid_series_index"]
        if any(level["series_index"] != series for level in levels):
            raise ValueError(
                "original WSI validated levels must use pyramid_series_index"
            )
        return

    if len(levels) != 1 or levels[0] != start["required_levels"][0]:
        raise ValueError(
            "derived region must have exactly its required validated level"
        )


def _require_manifest_start_match(
    start: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    for field in (
        "artifact_id",
        "creation_action_id",
        "container",
        "total_bytes",
    ):
        if manifest[field] != start[field]:
            raise ValueError(f"source manifest {field} does not match start request")
    _validate_level_coverage(start, manifest["validated_levels"])


def _require_current_schema(value: Any) -> None:
    if require_int32(value, "schema_version") != SOURCE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported source artifact schema_version")


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return value
