"""Exact wire and private-record contracts for transient administrative jobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_canonical_uuid,
    require_nonblank_string,
    require_rfc3339_millis,
)
from histoforge.cloud._errors import (
    gateway_error_from_exception,
    validate_gateway_error,
)
from histoforge.cloud.inference.identity import validate_module_content_identity
from histoforge.cloud.modules.identity import module_content_identity
from histoforge.cloud.modules.schema import validate_commit_sha
from histoforge.module_ref import ModuleRef, workspace_training_run_id

JOB_TYPES = frozenset({"INSTALL", "PUBLISH"})
JOB_PHASES = frozenset({"QUEUED", "RUNNING", "SUCCEEDED", "PARTIAL", "FAILED"})
ACTIVE_PHASES = frozenset({"QUEUED", "RUNNING"})
TERMINAL_PHASES = JOB_PHASES - ACTIVE_PHASES

_ENTRY_BASE_FIELDS = frozenset(
    {"type", "requested_module_ref", "phase", "updated_at", "conflict_refs"}
)
_STATUS_BASE_FIELDS = frozenset(
    {"job_id", "type", "requested_module_ref", "phase", "updated_at"}
)
_INSTALL_REQUIRED = frozenset({"module_ref"})
_INSTALL_OPTIONAL = frozenset({"hugging_face_revision"})
_PUBLISH_REQUIRED = frozenset(
    {
        "source_module",
        "target_repository",
        "visibility",
        "card_markdown",
        "sync_locally",
    }
)


def validate_job_accepted(value: Any) -> dict[str, str]:
    """Validate the exact small 202 admission response."""

    accepted = _exact_object(
        value, frozenset({"job_id"}), frozenset(), "admin job accepted"
    )
    return {"job_id": require_canonical_uuid(accepted["job_id"], "job_id")}


def validate_install_request(value: Any) -> dict[str, Any]:
    """Validate and detach one exact credential-free install request."""

    request = _exact_object(
        value, _INSTALL_REQUIRED, _INSTALL_OPTIONAL, "install request"
    )
    result = {"module_ref": module_ref(request["module_ref"], "module_ref")}
    if "hugging_face_revision" in request:
        revision = request["hugging_face_revision"]
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("hugging_face_revision must be a nonblank string")
        result["hugging_face_revision"] = revision
    return canonical_tree(result)


def validate_publish_request(value: Any) -> dict[str, Any]:
    """Validate and detach one exact credential-free publication request."""

    request = _exact_object(value, _PUBLISH_REQUIRED, frozenset(), "publish request")
    visibility = request["visibility"]
    if visibility not in {"PUBLIC", "PRIVATE"}:
        raise ValueError("visibility must be PUBLIC or PRIVATE")
    card = request["card_markdown"]
    if not isinstance(card, str) or not card:
        raise ValueError("card_markdown must be a nonempty string")
    sync = request["sync_locally"]
    if not isinstance(sync, bool):
        raise ValueError("sync_locally must be a boolean")
    source = module_ref(request["source_module"], "source_module")
    if sync and workspace_training_run_id(source) is not None:
        raise ValueError(
            "sync_locally is not available for a run-derived workspace module"
        )
    return canonical_tree(
        {
            "source_module": source,
            "target_repository": module_ref(
                request["target_repository"], "target_repository"
            ),
            "visibility": visibility,
            "card_markdown": card,
            "sync_locally": sync,
        }
    )


def validate_job_entry(value: Any) -> dict[str, Any]:
    """Validate one private Dict record, including admission-only conflict refs."""

    if not isinstance(value, dict):
        raise ValueError("administrative job record must be an object")
    phase = job_phase(value.get("phase"))
    expected = set(_ENTRY_BASE_FIELDS)
    if phase in {"SUCCEEDED", "PARTIAL"}:
        expected.add("result")
    elif phase == "FAILED":
        expected.add("failure")
    _require_fields(value, expected, "administrative job record")

    refs_value = value["conflict_refs"]
    if not isinstance(refs_value, list) or not refs_value:
        raise ValueError("conflict_refs must be a nonempty array")
    refs = [module_ref(item, "conflict_refs") for item in refs_value]
    if refs != sorted(set(refs)):
        raise ValueError("conflict_refs must be unique and sorted")

    normalized_type = job_type(value["type"])
    requested = module_ref(value["requested_module_ref"], "requested_module_ref")
    if requested not in refs:
        raise ValueError("conflict_refs must contain requested_module_ref")
    if normalized_type == "INSTALL" and refs != [requested]:
        raise ValueError("INSTALL conflict_refs must contain only requested_module_ref")

    normalized: dict[str, Any] = {
        "type": normalized_type,
        "requested_module_ref": requested,
        "phase": phase,
        "updated_at": require_rfc3339_millis(value["updated_at"], "updated_at"),
        "conflict_refs": refs,
    }
    if "result" in value:
        normalized["result"] = validate_admin_job_result(value["result"])
    if "failure" in value:
        normalized["failure"] = validate_gateway_error(value["failure"])
    _require_phase_semantics(normalized)
    return canonical_tree(normalized)


def project_job_status(job_id: str, entry: Any) -> dict[str, Any]:
    """Project a private record into the exact public status shape."""

    normalized = validate_job_entry(entry)
    status = {
        "job_id": require_canonical_uuid(job_id, "job_id"),
        "type": normalized["type"],
        "requested_module_ref": normalized["requested_module_ref"],
        "phase": normalized["phase"],
        "updated_at": normalized["updated_at"],
    }
    if "result" in normalized:
        status["result"] = normalized["result"]
    if "failure" in normalized:
        status["failure"] = normalized["failure"]
    return canonical_tree(status)


def validate_job_status(value: Any) -> dict[str, Any]:
    """Validate one public status independently of its private Dict record."""

    if not isinstance(value, dict):
        raise ValueError("administrative job status must be an object")
    phase = job_phase(value.get("phase"))
    expected = set(_STATUS_BASE_FIELDS)
    if phase in {"SUCCEEDED", "PARTIAL"}:
        expected.add("result")
    elif phase == "FAILED":
        expected.add("failure")
    _require_fields(value, expected, "administrative job status")
    entry = {key: item for key, item in value.items() if key != "job_id"}
    entry["conflict_refs"] = [value["requested_module_ref"]]
    normalized = project_job_status(value["job_id"], entry)
    return normalized


def validate_job_page(value: Any) -> dict[str, Any]:
    """Validate one bounded public page independently of store pagination."""

    page = _exact_object(
        value,
        frozenset({"jobs"}),
        frozenset({"next_cursor"}),
        "admin job page",
    )
    jobs_value = page["jobs"]
    if not isinstance(jobs_value, list):
        raise ValueError("jobs must be an array")
    result: dict[str, Any] = {
        "jobs": [validate_job_status(item) for item in jobs_value]
    }
    if "next_cursor" in page:
        result["next_cursor"] = require_nonblank_string(
            page["next_cursor"], "next_cursor"
        )
    return canonical_tree(result)


def validate_admin_job_result(value: Any) -> dict[str, Any]:
    """Validate the closed installed-or-published success result."""

    if not isinstance(value, dict):
        raise ValueError("admin job result must be an object")
    result_type = value.get("type")
    if result_type == "installed":
        _require_fields(value, {"type", "installed_module"}, "installed result")
        installed = validate_module_content_identity(value["installed_module"])
        _require_resolved_commit(installed, "installed result")
        return canonical_tree(
            {
                "type": "installed",
                "installed_module": installed,
            }
        )
    if result_type != "published":
        raise ValueError("unknown admin job result type")
    _require_fields(
        value,
        {"type", "target_repository", "commit_sha", "local_sync"},
        "published result",
    )
    target = module_ref(value["target_repository"], "target_repository")
    commit = validate_commit_sha(value["commit_sha"], "commit_sha")
    local_sync = _validate_local_sync(value["local_sync"])
    if local_sync.get("installed_module", {}).get("module_ref", target) != target:
        raise ValueError("local sync installed module does not match target repository")
    if (
        local_sync.get("installed_module", {}).get("resolved_commit_sha", commit)
        != commit
    ):
        raise ValueError("local sync installed module does not match published commit")
    return canonical_tree(
        {
            "type": "published",
            "target_repository": target,
            "commit_sha": commit,
            "local_sync": local_sync,
        }
    )


def installed_result(record: Any) -> dict[str, Any]:
    """Build the exact successful installation result from a stored record."""

    return validate_admin_job_result(
        {"type": "installed", "installed_module": module_content_identity(record)}
    )


def published_result(
    target_repository: str,
    commit_sha: str,
    *,
    sync_requested: bool,
    local_record: Any | None,
    local_sync_error: BaseException | None,
) -> dict[str, Any]:
    """Build publication identity plus the separately optional local-sync outcome."""

    target = module_ref(target_repository, "target_repository")
    if not sync_requested:
        if local_record is not None or local_sync_error is not None:
            raise ValueError("unrequested local sync cannot have an outcome payload")
        local_sync: dict[str, Any] = {"type": "not-requested"}
    elif local_sync_error is not None:
        if local_record is not None:
            raise ValueError("failed local sync cannot also have an installed module")
        local_sync = {
            "type": "failed",
            "failure": gateway_error_from_exception(
                local_sync_error, resource_id=target
            ),
        }
    else:
        if local_record is None:
            raise ValueError(
                "successful local sync requires its installed module record"
            )
        local_sync = {
            "type": "succeeded",
            "installed_module": module_content_identity(local_record),
        }
    return validate_admin_job_result(
        {
            "type": "published",
            "target_repository": target,
            "commit_sha": commit_sha,
            "local_sync": local_sync,
        }
    )


def job_type(value: Any) -> str:
    if value not in JOB_TYPES:
        raise ValueError(f"unknown administrative job type: {value!r}")
    return value


def job_phase(value: Any) -> str:
    if value not in JOB_PHASES:
        raise ValueError(f"unknown administrative job phase: {value!r}")
    return value


def module_ref(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be canonical owner/name")
    try:
        return str(ModuleRef.parse(value))
    except ValueError as error:
        raise ValueError(f"{path} must be canonical owner/name") from error


def _validate_local_sync(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("local_sync must be an object")
    sync_type = value.get("type")
    if sync_type == "not-requested":
        _require_fields(value, {"type"}, "not-requested local sync")
        return {"type": sync_type}
    if sync_type == "succeeded":
        _require_fields(value, {"type", "installed_module"}, "successful local sync")
        installed = validate_module_content_identity(value["installed_module"])
        _require_resolved_commit(installed, "successful local sync")
        return canonical_tree(
            {
                "type": sync_type,
                "installed_module": installed,
            }
        )
    if sync_type == "failed":
        _require_fields(value, {"type", "failure"}, "failed local sync")
        return canonical_tree(
            {"type": sync_type, "failure": validate_gateway_error(value["failure"])}
        )
    raise ValueError("unknown local_sync result type")


def _require_resolved_commit(identity: Mapping[str, Any], path: str) -> None:
    if "resolved_commit_sha" not in identity:
        raise ValueError(f"{path} requires resolved_commit_sha")


def _require_phase_semantics(value: Mapping[str, Any]) -> None:
    phase = value["phase"]
    result = value.get("result")
    failure = value.get("failure")
    if phase in ACTIVE_PHASES:
        if result is not None or failure is not None:
            raise ValueError("active administrative jobs cannot have an outcome")
        return
    if phase == "FAILED":
        if failure is None or result is not None:
            raise ValueError("FAILED requires one failure and no result")
        return
    if result is None or failure is not None:
        raise ValueError("successful administrative jobs require one result")
    job_type_value = value["type"]
    requested = value["requested_module_ref"]
    if job_type_value == "INSTALL":
        if phase != "SUCCEEDED" or result["type"] != "installed":
            raise ValueError("INSTALL success requires an installed result")
        if result["installed_module"]["module_ref"] != requested:
            raise ValueError("installed result does not match requested_module_ref")
        return
    if result["type"] != "published":
        raise ValueError("PUBLISH success requires a published result")
    if result["target_repository"] != requested:
        raise ValueError("published result does not match requested_module_ref")
    local_type = result["local_sync"]["type"]
    if phase == "PARTIAL" and local_type != "failed":
        raise ValueError("PARTIAL requires failed local sync")
    if phase == "SUCCEEDED" and local_type == "failed":
        raise ValueError("SUCCEEDED cannot contain failed local sync")


def _exact_object(
    value: Any,
    required: frozenset[str],
    optional: frozenset[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object")
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing or unknown:
        raise ValueError(
            f"{path} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
