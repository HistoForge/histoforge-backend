"""Validate model-run status against its request and terminal manifest.

Status summarizes progress and references immutable request and result artifacts.
Run persistence and publication belong to ModelRunStore.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_rfc3339_millis,
    sha256_bytes,
)
from histoforge.cloud._errors import validate_gateway_error
from histoforge.cloud.artifacts.refs import validate_artifact_ref
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_result_manifest_completed_unit_count,
    validate_model_result_manifest,
)
from histoforge.cloud.inference.runs.request import validate_model_run_resource_limits
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION

_REQUIRED_FIELDS = {
    "schema_version",
    "model_run_id",
    "phase",
    "request_artifact",
    "completed_unit_count",
    "total_unit_count",
    "phase_deadline",
}
_OPTIONAL_FIELDS = {"terminal_manifest", "failure"}
_PHASES = {
    "QUEUED",
    "RUNNING",
    "PARTIAL",
    "COMPLETE",
    "FAILED",
    "CANCELLED",
    "TIMED_OUT",
}


def validate_model_run_status(value: Any) -> dict[str, Any]:
    """Validate the closed status shape and phase/payload combination."""

    status = _require_object(value, "model run status")
    actual = set(status)
    if not _REQUIRED_FIELDS <= actual or not actual <= (
        _REQUIRED_FIELDS | _OPTIONAL_FIELDS
    ):
        raise ValueError("model run status has missing or unknown fields")
    if (
        require_int32(status["schema_version"], "schema_version")
        != MODEL_RUN_SCHEMA_VERSION
    ):
        raise ValueError("unsupported model run status schema_version")
    phase = status["phase"]
    if not isinstance(phase, str) or phase not in _PHASES:
        raise ValueError(f"unsupported model run phase: {phase!r}")
    result: dict[str, Any] = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "model_run_id": require_canonical_uuid(status["model_run_id"], "model_run_id"),
        "phase": phase,
        "request_artifact": validate_artifact_ref(status["request_artifact"]),
        "completed_unit_count": require_nonnegative_int32(
            status["completed_unit_count"], "completed_unit_count"
        ),
        "total_unit_count": require_nonnegative_int32(
            status["total_unit_count"], "total_unit_count"
        ),
        "phase_deadline": require_rfc3339_millis(
            status["phase_deadline"], "phase_deadline"
        ),
    }
    if result["completed_unit_count"] > result["total_unit_count"]:
        raise ValueError("completed_unit_count cannot exceed total_unit_count")
    if "terminal_manifest" in status:
        result["terminal_manifest"] = validate_artifact_ref(status["terminal_manifest"])
    if "failure" in status:
        result["failure"] = validate_gateway_error(status["failure"])

    has_manifest = "terminal_manifest" in result
    has_failure = "failure" in result
    if phase in {"QUEUED", "RUNNING", "CANCELLED", "TIMED_OUT"}:
        if has_manifest or has_failure:
            raise ValueError(f"{phase} status cannot carry terminal payloads")
    elif phase == "PARTIAL":
        if not has_manifest:
            raise ValueError("PARTIAL status requires terminal_manifest")
    elif phase == "COMPLETE":
        if not has_manifest or has_failure:
            raise ValueError(f"{phase} status requires only terminal_manifest")
    elif not has_failure or has_manifest:
        raise ValueError("FAILED status requires only failure")
    return canonical_tree(result)


def validate_model_run_status_projection(
    value: Any,
    request: Any,
    model_run_limits: Any,
    terminal_manifest: Any | None = None,
) -> dict[str, Any]:
    """Bind status to the exact request and optional downloaded manifest bytes."""

    status = validate_model_run_status(value)
    submitted = validate_model_run_resource_limits(request, model_run_limits)
    if status["model_run_id"] != submitted["model_run_id"]:
        raise ValueError("status model_run_id does not match model run request")
    _require_canonical_json_artifact(
        status["request_artifact"], submitted, "request_artifact"
    )

    plans = build_expected_model_execution_plans(submitted, model_run_limits)
    total_units = sum(_plan_unit_count(plan) for plan in plans.values())
    if status["total_unit_count"] != total_units:
        raise ValueError("status total_unit_count does not match generated plans")
    phase = status["phase"]
    if phase in {"QUEUED", "TIMED_OUT"} and status["completed_unit_count"] != 0:
        raise ValueError(f"{phase} status must report zero completed units")

    if phase in {"PARTIAL", "COMPLETE"}:
        if terminal_manifest is None:
            raise ValueError(f"{phase} status requires downloaded terminal manifest")
        manifest = validate_model_result_manifest(
            terminal_manifest, submitted, model_run_limits
        )
        expected_complete = phase == "COMPLETE"
        if manifest["complete"] != expected_complete:
            raise ValueError(
                "status phase disagrees with terminal manifest completeness"
            )
        if status.get("failure") != manifest.get("failure"):
            raise ValueError("status failure does not match terminal manifest")
        _require_canonical_json_artifact(
            status["terminal_manifest"], manifest, "terminal_manifest"
        )
        expected_completed = model_result_manifest_completed_unit_count(manifest)
        if status["completed_unit_count"] != expected_completed:
            raise ValueError(
                "status completed_unit_count does not match terminal manifest"
            )
    elif terminal_manifest is not None:
        raise ValueError(f"{phase} status cannot be paired with a terminal manifest")
    return status


def _require_canonical_json_artifact(
    artifact: Mapping[str, Any], value: Any, path: str
) -> None:
    encoded = canonical_bytes(value)
    if artifact["media_type"] != "application/json":
        raise ValueError(f"{path} must use application/json")
    if "content_encoding" in artifact:
        raise ValueError(f"{path} must not use content encoding")
    if artifact["size_bytes"] != len(encoded) or artifact["sha256"] != sha256_bytes(
        encoded
    ):
        raise ValueError(f"{path} does not bind the supplied canonical JSON bytes")


def _plan_unit_count(plan: Mapping[str, Any]) -> int:
    return len(plan["windows"]) if plan["type"] == "core-halo" else 1


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value
