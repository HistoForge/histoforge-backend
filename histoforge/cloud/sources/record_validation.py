"""Validate the stored components needed by an upload operation.

These checks bind reservations, chunk digests and sealed files to the same image
and its original deadline. Callers select the components they need; a successful
seal does not depend on obsolete upload history, and absolute expiry bypasses it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.contract import validate_source_artifact_start_request
from histoforge.cloud.sources.publication import (
    validate_source_artifact_manifest,
    validate_source_artifact_seal_request,
    validate_source_artifact_seal_response,
    validate_source_byte_chunk_accepted,
)

_RESERVATION_FIELDS = {
    "schema_version",
    "start_request",
    "start_request_digest",
    "accepted_at",
    "absolute_expires_at",
}
_CHUNK_CLAIM_FIELDS = {
    "schema_version",
    "artifact_id",
    "chunk_id",
    "sha256",
    "claimed_at",
}
_CHUNK_COMMIT_FIELDS = {"schema_version", "artifact_id", "chunk", "accepted_at"}
_SEAL_CLAIM_FIELDS = {"schema_version", "artifact_id", "seal_request", "claimed_at"}
_SEALED_FIELDS = {
    "schema_version",
    "artifact_id",
    "seal_request",
    "manifest",
    "sealed",
}


def validate_reservation(value: Any, artifact_id: str) -> dict[str, Any]:
    try:
        item = _exact_object(value, _RESERVATION_FIELDS, "source reservation")
        if item["schema_version"] != SOURCE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported private source reservation schema")
        start = validate_source_artifact_start_request(item["start_request"])
        if start["artifact_id"] != artifact_id:
            raise ValueError("source reservation key does not match request")
        digest = require_sha256(item["start_request_digest"], "start_request_digest")
        if digest != canonical_digest(start):
            raise ValueError("source reservation digest does not match request")
        accepted_at = require_rfc3339_millis(item["accepted_at"], "accepted_at")
        absolute = require_rfc3339_millis(
            item["absolute_expires_at"], "absolute_expires_at"
        )
        # Exact UTC-millisecond RFC3339 text sorts chronologically.
        if accepted_at >= absolute:
            raise ValueError("source reservation deadline is not after acceptance")
        return canonical_tree(
            {
                "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                "start_request": start,
                "start_request_digest": digest,
                "accepted_at": accepted_at,
                "absolute_expires_at": absolute,
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise invalid_source_record(artifact_id) from error


def validate_chunk_claim(value: Any, artifact_id: str) -> dict[str, Any]:
    try:
        item = _exact_object(value, _CHUNK_CLAIM_FIELDS, "source chunk claim")
        result = {
            "schema_version": item["schema_version"],
            "artifact_id": require_canonical_uuid(item["artifact_id"], "artifact_id"),
            "chunk_id": item["chunk_id"],
            "sha256": require_sha256(item["sha256"], "sha256"),
            "claimed_at": require_rfc3339_millis(item["claimed_at"], "claimed_at"),
        }
        if (
            result["schema_version"] != SOURCE_ARTIFACT_SCHEMA_VERSION
            or result["artifact_id"] != artifact_id
            or isinstance(result["chunk_id"], bool)
            or not isinstance(result["chunk_id"], int)
            or result["chunk_id"] < 0
        ):
            raise ValueError("invalid source chunk claim")
        return canonical_tree(result)
    except (KeyError, TypeError, ValueError) as error:
        raise invalid_source_record(artifact_id) from error


def validate_chunk_commit(value: Any, artifact_id: str) -> dict[str, Any]:
    try:
        item = _exact_object(value, _CHUNK_COMMIT_FIELDS, "source chunk record")
        accepted = validate_source_byte_chunk_accepted(
            {
                "schema_version": item["schema_version"],
                "artifact_id": item["artifact_id"],
                "chunk": item["chunk"],
            }
        )
        if accepted["artifact_id"] != artifact_id:
            raise ValueError("source chunk key does not match artifact")
        return canonical_tree(
            {
                **accepted,
                "accepted_at": require_rfc3339_millis(
                    item["accepted_at"], "accepted_at"
                ),
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise invalid_source_record(artifact_id) from error


def validate_seal_claim(value: Any, artifact_id: str) -> dict[str, Any]:
    try:
        item = _exact_object(value, _SEAL_CLAIM_FIELDS, "source seal claim")
        result = {
            "schema_version": item["schema_version"],
            "artifact_id": require_canonical_uuid(item["artifact_id"], "artifact_id"),
            "seal_request": validate_source_artifact_seal_request(item["seal_request"]),
            "claimed_at": require_rfc3339_millis(item["claimed_at"], "claimed_at"),
        }
        if (
            result["schema_version"] != SOURCE_ARTIFACT_SCHEMA_VERSION
            or result["artifact_id"] != artifact_id
        ):
            raise ValueError("source seal claim identity differs")
        return canonical_tree(result)
    except (KeyError, TypeError, ValueError) as error:
        raise invalid_source_record(artifact_id) from error


def validate_sealed_publication(
    value: Any, reservation: Mapping[str, Any]
) -> dict[str, Any]:
    artifact_id = reservation["start_request"]["artifact_id"]
    try:
        item = _exact_object(value, _SEALED_FIELDS, "sealed source record")
        if (
            item["schema_version"] != SOURCE_ARTIFACT_SCHEMA_VERSION
            or require_canonical_uuid(item["artifact_id"], "artifact_id") != artifact_id
        ):
            raise ValueError("sealed source record identity differs")
        start = reservation["start_request"]
        accepted = {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "creation_action_id": start["creation_action_id"],
            "artifact_id": artifact_id,
            "start_request_digest": reservation["start_request_digest"],
            "absolute_expires_at": reservation["absolute_expires_at"],
        }
        # Check the records together, including the original fixed deadline.
        sealed = validate_source_artifact_seal_response(
            start, accepted, item["seal_request"], item["manifest"], item["sealed"]
        )
        return canonical_tree(
            {
                "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "seal_request": validate_source_artifact_seal_request(
                    item["seal_request"]
                ),
                "manifest": validate_source_artifact_manifest(item["manifest"]),
                "sealed": sealed,
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise invalid_source_record(artifact_id) from error


def validate_activity(value: Any, artifact_id: str) -> str:
    try:
        return require_rfc3339_millis(value, "last_activity_at")
    except (TypeError, ValueError) as error:
        raise invalid_source_record(artifact_id) from error


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def invalid_source_record(artifact_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.INTERNAL_ERROR,
        "the source-artifact store contains an invalid record",
        resource_id=artifact_id,
    )
