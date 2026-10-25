"""Validate source-upload status and its reservation and seal references.

Status summarizes upload progress and references published image bytes.
Source services own transfer, persistence, expiry, and HTTP behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    source_artifact_start_digest,
    validate_source_artifact_acceptance,
    validate_source_artifact_start_request,
)
from histoforge.cloud.sources.publication import (
    validate_source_artifact_ref,
    validate_source_artifact_seal_request,
    validate_source_artifact_sealed,
)

_STATUS_BASE_FIELDS = {
    "schema_version",
    "artifact_id",
    "start_request_digest",
    "phase",
    "missing_chunks",
    "absolute_expires_at",
}
_UPLOADING_FIELDS = _STATUS_BASE_FIELDS | {"upload_idle_expires_at"}
_SEALED_FIELDS = _STATUS_BASE_FIELDS | {"sealed_artifact"}
_PHASES = {"UPLOADING", "SEALED"}


def validate_source_artifact_status(value: Any) -> dict[str, Any]:
    """Validate one bounded status without consulting external upload state.

    The status binds a source reservation ID, its canonical request digest,
    sorted missing ranges, phase-specific sealed/idle data, and one fixed
    absolute deadline. It never embeds uploaded image bytes or execution state.
    """

    status = _require_object(value, "source artifact status")
    phase_value = status.get("phase")
    if not isinstance(phase_value, str) or phase_value not in _PHASES:
        raise ValueError(f"unsupported source artifact phase: {phase_value!r}")
    expected = _UPLOADING_FIELDS if phase_value == "UPLOADING" else _SEALED_FIELDS
    if set(status) != expected:
        raise ValueError(
            "source artifact status fields mismatch: "
            f"missing={sorted(expected - set(status))}, "
            f"unknown={sorted(set(status) - expected)}"
        )
    if (
        require_int32(status["schema_version"], "schema_version")
        != SOURCE_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported source artifact schema_version")

    missing = _normalize_missing_chunks(status["missing_chunks"])
    result = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": require_canonical_uuid(status["artifact_id"], "artifact_id"),
        "start_request_digest": require_sha256(
            status["start_request_digest"], "start_request_digest"
        ),
        "phase": phase_value,
        "missing_chunks": missing,
        "absolute_expires_at": require_rfc3339_millis(
            status["absolute_expires_at"], "absolute_expires_at"
        ),
    }
    if phase_value == "UPLOADING":
        result["upload_idle_expires_at"] = require_rfc3339_millis(
            status["upload_idle_expires_at"], "upload_idle_expires_at"
        )
    else:
        if missing:
            raise ValueError("SEALED status must have no missing_chunks")
        result["sealed_artifact"] = validate_source_artifact_ref(
            status["sealed_artifact"]
        )
    return canonical_tree(result)


def validate_source_artifact_status_projection(
    start_request: Any,
    accepted: Any,
    status: Any,
    expected_seal_request: Any | None = None,
    sealed: Any | None = None,
) -> dict[str, Any]:
    """Bind status to its reservation, requested digest, and seal publication."""

    start = validate_source_artifact_start_request(start_request)
    acceptance = validate_source_artifact_acceptance(start, accepted)
    result = validate_source_artifact_status(status)
    if result["artifact_id"] != start["artifact_id"]:
        raise ValueError("status artifact_id does not match request")
    if result["absolute_expires_at"] != acceptance["absolute_expires_at"]:
        raise ValueError("status absolute deadline does not match acceptance")
    if result["start_request_digest"] != source_artifact_start_digest(start):
        raise ValueError("status start_request_digest does not match request")

    chunk_count = canonical_chunk_count(start["total_bytes"], start["chunk_size_bytes"])
    if any(chunk_id >= chunk_count for chunk_id in result["missing_chunks"]):
        raise ValueError("status missing_chunks lies outside the canonical chunk plan")

    if result["phase"] == "UPLOADING":
        if sealed is not None:
            raise ValueError("UPLOADING status cannot bind a seal publication")
        if result["upload_idle_expires_at"] > result["absolute_expires_at"]:
            raise ValueError("upload idle deadline cannot exceed absolute deadline")
        return result
    reference = result["sealed_artifact"]
    if expected_seal_request is None:
        raise ValueError("SEALED status requires the expected seal request")
    expected_seal = validate_source_artifact_seal_request(expected_seal_request)
    if reference["artifact_id"] != start["artifact_id"]:
        raise ValueError("status sealed_artifact artifact_id does not match request")
    if reference["creation_action_id"] != start["creation_action_id"]:
        raise ValueError(
            "status sealed_artifact creation_action_id does not match request"
        )
    if reference["start_request_digest"] != source_artifact_start_digest(start):
        raise ValueError(
            "status sealed_artifact start_request_digest does not match request"
        )
    if reference["container"] != start["container"]:
        raise ValueError("status sealed_artifact container does not match request")
    if reference["bytes_digest"] != expected_seal["bytes_digest"]:
        raise ValueError(
            "status sealed_artifact bytes_digest does not match seal request"
        )
    if sealed is not None:
        publication = validate_source_artifact_sealed(sealed)
        if reference != publication["artifact"]:
            raise ValueError("status sealed_artifact does not match seal publication")
        if result["absolute_expires_at"] != publication["absolute_expires_at"]:
            raise ValueError("status absolute deadline does not match seal publication")
    return result


def _normalize_missing_chunks(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("missing_chunks must be an array")
    chunks = [require_nonnegative_int32(item, "missing_chunks item") for item in value]
    chunks.sort()
    if len(chunks) != len(set(chunks)):
        raise ValueError("missing_chunks contains a duplicate chunk ID")
    return chunks


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value
