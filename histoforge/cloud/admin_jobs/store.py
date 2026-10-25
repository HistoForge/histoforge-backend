"""Transient admin-job storage, admission, transitions, and pagination.

The Dict is operational UI state, not a transaction ledger. The mutation lane is the
sole phase writer after acceptance. Dismissing a row never cancels work or proves that
an external side effect did not occur.
"""

from __future__ import annotations

import base64
import binascii
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import modal

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    parse_json,
    require_canonical_uuid,
    require_rfc3339_millis,
)
from histoforge.cloud._config import DEPLOYMENT_CONFIG, MAX_ACTIVE_ADMIN_JOBS
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud._time import now_z
from histoforge.cloud.admin_jobs.contract import (
    ACTIVE_PHASES,
    TERMINAL_PHASES,
    job_type,
    module_ref,
    project_job_status,
    validate_admin_job_result,
    validate_job_accepted,
    validate_job_entry,
)

MAX_RETAINED_ADMIN_JOBS = 1_000

_CURSOR_FIELDS = frozenset({"updated_at", "job_id"})
_CURSOR_TEXT = re.compile(r"[A-Za-z0-9_-]+")


def submit_job(
    *,
    deployment_id: str,
    auth_revision: int,
    job_type_value: str,
    requested_module_ref: str,
    conflict_refs: Sequence[str],
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Submit one request through the deployment's serialized admission owner."""

    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.admin_job_admission_class_name,
    )()
    accepted = lane.accept.remote(
        deployment_id,
        auth_revision,
        job_type_value,
        requested_module_ref,
        list(conflict_refs),
        canonical_tree(dict(payload)),
    )
    return validate_job_accepted(accepted)


def accept_job(
    jobs,
    *,
    deployment_id: str,
    auth_revision: int,
    job_type_value: str,
    requested_module_ref: str,
    conflict_refs: Sequence[str],
    payload: Mapping[str, Any],
    spawn_runner: Callable[[str, str, dict[str, Any], str, int], None] | None = None,
) -> dict[str, str]:
    """Admit and spawn one job; caller serializes this complete transaction."""

    normalized_type = job_type(job_type_value)
    requested = module_ref(requested_module_ref, "requested_module_ref")
    conflicts = sorted({module_ref(value, "conflict_refs") for value in conflict_refs})
    if not conflicts:
        raise ValueError("conflict_refs must not be empty")

    records = _validated_records(jobs)
    active = [item for item in records if item[1]["phase"] in ACTIVE_PHASES]
    if len(active) >= MAX_ACTIVE_ADMIN_JOBS:
        raise GatewayFailure(
            GatewayErrorCode.RATE_LIMITED,
            "the deployment has reached its active administrative-job limit",
        )
    requested_conflicts = set(conflicts)
    for _, entry in active:
        overlap = requested_conflicts.intersection(entry["conflict_refs"])
        if overlap:
            resource = min(overlap)
            raise GatewayFailure(
                GatewayErrorCode.MODULE_BUSY,
                f"an observed administrative job already owns {resource}",
                resource_id=resource,
            )

    _prune_old_terminal_jobs(jobs, records)
    job_id = str(uuid.uuid4())
    jobs[job_id] = validate_job_entry(
        {
            "type": normalized_type,
            "requested_module_ref": requested,
            "phase": "QUEUED",
            "updated_at": now_z(),
            "conflict_refs": conflicts,
        }
    )

    try:
        (spawn_runner or _spawn_runner)(
            job_id,
            normalized_type,
            canonical_tree(dict(payload)),
            deployment_id,
            auth_revision,
        )
    except Exception as error:
        # Spawn failure can be ambiguous after platform acceptance. Keep the visible
        # QUEUED row; the user may inspect/dismiss it and retry explicitly.
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "the administrative worker could not be started",
            resource_id=job_id,
        ) from error
    return {"job_id": job_id}


def _spawn_runner(
    job_id: str,
    job_type_value: str,
    payload: dict[str, Any],
    deployment_id: str,
    auth_revision: int,
) -> None:
    modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.module_mutation_class_name,
    )().run.spawn(
        job_id,
        job_type_value,
        payload,
        deployment_id,
        auth_revision,
    )


def transition_running(jobs, job_id: str) -> bool:
    """Move QUEUED to RUNNING; return false when the row was dismissed."""

    return _transition(jobs, job_id, "RUNNING")


def transition_succeeded(jobs, job_id: str, result: Any) -> bool:
    """Publish one successful result."""

    return _transition(
        jobs, job_id, "SUCCEEDED", result=validate_admin_job_result(result)
    )


def transition_partial(jobs, job_id: str, result: Any) -> bool:
    """Publish successful HF publication with failed requested local sync."""

    normalized = validate_admin_job_result(result)
    if (
        normalized.get("type") != "published"
        or normalized["local_sync"].get("type") != "failed"
    ):
        raise ValueError("PARTIAL requires a published result with failed local sync")
    return _transition(jobs, job_id, "PARTIAL", result=normalized)


def transition_failed(jobs, job_id: str, failure: Any) -> bool:
    """Publish one closed terminal failure."""

    return _transition(jobs, job_id, "FAILED", failure=failure)


def _transition(
    jobs,
    job_id: str,
    phase: str,
    *,
    result: dict[str, Any] | None = None,
    failure: dict[str, Any] | None = None,
) -> bool:
    canonical_id = require_canonical_uuid(job_id, "job_id")
    try:
        entry = validate_job_entry(jobs[canonical_id])
    except KeyError:
        return False
    if entry["phase"] in TERMINAL_PHASES:
        raise ValueError("a terminal administrative job cannot transition")
    if phase == "RUNNING" and entry["phase"] != "QUEUED":
        raise ValueError("RUNNING requires a QUEUED administrative job")
    if phase != "RUNNING" and entry["phase"] != "RUNNING":
        raise ValueError("a terminal phase requires a RUNNING administrative job")
    updated = {
        key: value for key, value in entry.items() if key not in {"result", "failure"}
    }
    updated.update({"phase": phase, "updated_at": now_z()})
    if result is not None:
        updated["result"] = result
    if failure is not None:
        updated["failure"] = failure
    jobs[canonical_id] = validate_job_entry(updated)
    return True


def get_status(jobs, job_id: str) -> dict[str, Any]:
    """Return one exact status or JOB_NOT_FOUND."""

    try:
        entry = jobs[job_id]
    except KeyError as error:
        raise GatewayFailure(
            GatewayErrorCode.JOB_NOT_FOUND,
            f"administrative job not found: {job_id}",
            resource_id=job_id,
        ) from error
    return _project(job_id, entry)


def forget_job(jobs, job_id: str) -> None:
    """Dismiss one transient row without cancelling its worker."""

    try:
        del jobs[job_id]
    except KeyError as error:
        raise GatewayFailure(
            GatewayErrorCode.JOB_NOT_FOUND,
            f"administrative job not found: {job_id}",
            resource_id=job_id,
        ) from error


def job_page(jobs, *, limit: int, cursor: str | None) -> dict[str, Any]:
    """Return a deterministic descending-update-time page."""

    records = [_project(job_id, entry) for job_id, entry in _validated_records(jobs)]
    records.sort(key=lambda item: item["job_id"])
    records.sort(key=lambda item: item["updated_at"], reverse=True)
    cursor_value = _decode_cursor(cursor) if cursor is not None else None
    if cursor_value is not None:
        records = [
            item
            for item in records
            if item["updated_at"] < cursor_value["updated_at"]
            or (
                item["updated_at"] == cursor_value["updated_at"]
                and item["job_id"] > cursor_value["job_id"]
            )
        ]
    page = records[:limit]
    result: dict[str, Any] = {"jobs": page}
    if len(records) > limit:
        result["next_cursor"] = _encode_cursor(page[-1])
    return canonical_tree(result)


def active_job(jobs) -> dict[str, Any] | None:
    """Return one observed active job for synchronous-delete admission."""

    active = [
        _project(job_id, entry)
        for job_id, entry in _validated_records(jobs)
        if entry["phase"] in ACTIVE_PHASES
    ]
    active.sort(key=lambda item: (item["updated_at"], item["job_id"]))
    return active[0] if active else None


def _project(job_id: str, entry: Any) -> dict[str, Any]:
    try:
        return project_job_status(job_id, entry)
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "the administrative-job store contains an invalid record",
            resource_id=str(job_id),
        ) from error


def _validated_records(jobs) -> list[tuple[str, dict[str, Any]]]:
    records: list[tuple[str, dict[str, Any]]] = []
    try:
        for job_id, entry in jobs.items():
            canonical_id = require_canonical_uuid(job_id, "job_id")
            records.append((canonical_id, validate_job_entry(entry)))
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "the administrative-job store contains an invalid record",
        ) from error
    return records


def _prune_old_terminal_jobs(jobs, records) -> None:
    excess = len(records) - MAX_RETAINED_ADMIN_JOBS + 1
    if excess <= 0:
        return
    terminal = sorted(
        (entry["updated_at"], job_id)
        for job_id, entry in records
        if entry["phase"] in TERMINAL_PHASES
    )
    if len(terminal) < excess:
        raise GatewayFailure(
            GatewayErrorCode.RATE_LIMITED,
            "the transient administrative-job store is full",
        )
    for _, job_id in terminal[:excess]:
        try:
            del jobs[job_id]
        except KeyError:
            pass


def _encode_cursor(status: Mapping[str, Any]) -> str:
    encoded = canonical_bytes(
        {"updated_at": status["updated_at"], "job_id": status["job_id"]}
    )
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> dict[str, str]:
    if not isinstance(value, str) or not _CURSOR_TEXT.fullmatch(value):
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, "invalid job cursor")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        cursor = parse_json(decoded)
    except (UnicodeDecodeError, ValueError, binascii.Error) as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR, "invalid job cursor"
        ) from error
    if not isinstance(cursor, dict) or set(cursor) != _CURSOR_FIELDS:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, "invalid job cursor")
    try:
        result = {
            "updated_at": require_rfc3339_millis(cursor["updated_at"], "updated_at"),
            "job_id": require_canonical_uuid(cursor["job_id"], "job_id"),
        }
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR, "invalid job cursor"
        ) from error
    if _encode_cursor(result) != value:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR, "noncanonical job cursor"
        )
    return result
