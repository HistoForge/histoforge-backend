"""Closed persisted and public control records for managed training runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
    require_finite_number,
    require_nonblank_string,
    require_nonnegative_int32,
    require_positive_int32,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud._errors import validate_gateway_error
from histoforge.cloud.training.contract import (
    require_training_model_id,
    require_training_stable_id,
    require_training_text,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)
from histoforge.module_ref import ModuleRef

_ADMISSION_FIELDS = {"type", "run_id", "request_digest", "created_at"}
_PROGRESS_FIELDS = {"completed_epochs", "total_epochs"}
_WORKER_REQUIRED_FIELDS = {
    "call_attachment_deadline_at",
    "execution_deadline_at",
    "worker_claimed",
}
_WORKER_OPTIONAL_FIELDS = {"worker_call_id"}
_FINALIZATION_FIELDS = {
    "selected_weight_digest",
    "report_digest",
    "module_content_digest",
}
_RUN_REQUIRED_FIELDS = {
    "run_id",
    "request",
    "request_digest",
    "trainer_build",
    "state",
    "progress",
    "report_revision",
    "created_at",
    "updated_at",
}
_RUN_OPTIONAL_FIELDS = {"finalization", "result"}
_CONTROL_FIELDS = {"run", "worker", "finalization_revision"}
_RUN_STATES = {"QUEUED", "RUNNING", "FINALIZING", "SUCCEEDED", "FAILED", "CANCELLED"}
_ACTIVE_STATES = {"QUEUED", "RUNNING", "FINALIZING"}
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
_SUMMARY_REQUIRED_FIELDS = {
    "run_id",
    "result_display_name",
    "source_module_ref",
    "selected_model_id",
    "result_form",
    "state",
    "progress",
    "report_revision",
    "created_at",
    "updated_at",
}
_SUMMARY_OPTIONAL_FIELDS = {"primary_metric", "result"}
_PRIMARY_METRIC_FIELDS = {"series_id", "display_name", "value", "epoch"}
_PAGE_REQUIRED_FIELDS = {"items"}
_PAGE_OPTIONAL_FIELDS = {"next_cursor"}
_RESULT_FORMS = {"selected_model_only", "complete_source_workflow"}
_GENERATED_CURSOR_BYTES = 119


def validate_training_run_admission_record(value: Any) -> dict[str, Any]:
    """Validate one digest-keyed source-snapshot preparation reservation."""

    item = _exact_object(value, _ADMISSION_FIELDS, "training run admission")
    if item["type"] != "preparing":
        raise ValueError("training run admission type must be 'preparing'")
    return canonical_tree(
        {
            "type": "preparing",
            "run_id": require_canonical_uuid(item["run_id"], "run_id"),
            "request_digest": require_sha256(item["request_digest"], "request_digest"),
            "created_at": require_rfc3339_millis(item["created_at"], "created_at"),
        }
    )


def validate_training_progress(value: Any, api_limits: Any) -> dict[str, int]:
    """Validate the bounded completed-epoch projection used by Jobs polling."""

    item = _exact_object(value, _PROGRESS_FIELDS, "training progress")
    completed = require_nonnegative_int32(item["completed_epochs"], "completed_epochs")
    total = require_positive_int32(item["total_epochs"], "total_epochs")
    limits = validate_training_api_limits(api_limits)
    if total > limits["report"]["max_epochs"]:
        raise ValueError("total_epochs exceeds the training report limit")
    if completed > total:
        raise ValueError("completed_epochs cannot exceed total_epochs")
    return {"completed_epochs": completed, "total_epochs": total}


def validate_training_worker_control(value: Any) -> dict[str, Any]:
    """Validate private worker attachment, claim, and deadline state."""

    item = _object_with_optional(
        value,
        _WORKER_REQUIRED_FIELDS,
        _WORKER_OPTIONAL_FIELDS,
        "training worker control",
    )
    attachment_deadline = require_rfc3339_millis(
        item["call_attachment_deadline_at"], "call_attachment_deadline_at"
    )
    execution_deadline = require_rfc3339_millis(
        item["execution_deadline_at"], "execution_deadline_at"
    )
    if execution_deadline < attachment_deadline:
        raise ValueError("execution deadline cannot precede call attachment deadline")
    if not isinstance(item["worker_claimed"], bool):
        raise ValueError("worker_claimed must be a boolean")

    result: dict[str, Any] = {
        "call_attachment_deadline_at": attachment_deadline,
        "execution_deadline_at": execution_deadline,
        "worker_claimed": item["worker_claimed"],
    }
    if "worker_call_id" in item:
        result["worker_call_id"] = validate_training_worker_call_id(
            item["worker_call_id"]
        )
    if item["worker_claimed"] and "worker_call_id" not in result:
        raise ValueError("a claimed worker requires worker_call_id")
    return canonical_tree(result)


def validate_training_worker_call_id(value: Any) -> str:
    """Validate the one provider call identity stored with a training run."""

    call_id = require_nonblank_string(value, "worker_call_id")
    if len(call_id.encode("utf-8")) > 128:
        raise ValueError("worker_call_id exceeds 128 UTF-8 bytes")
    return call_id


def validate_training_finalization(value: Any) -> dict[str, str]:
    """Validate the frozen candidate and report identities used for finalization."""

    item = _exact_object(value, _FINALIZATION_FIELDS, "training finalization")
    return {
        "selected_weight_digest": require_sha256(
            item["selected_weight_digest"], "selected_weight_digest"
        ),
        "report_digest": require_sha256(item["report_digest"], "report_digest"),
        "module_content_digest": require_sha256(
            item["module_content_digest"], "module_content_digest"
        ),
    }


def validate_training_run_result(
    value: Any, state: str, api_limits: Any
) -> dict[str, Any]:
    """Validate the terminal result variant required by ``state``."""

    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ValueError("training run result must be a typed object")
    kind = value["type"]
    if state == "SUCCEEDED":
        result = _exact_object(value, {"type"}, "succeeded training result")
        if kind != "succeeded":
            raise ValueError("SUCCEEDED requires a succeeded result")
        return dict(result)
    if state == "FAILED":
        item = _exact_object(
            value, {"type", "error", "report_digest"}, "failed training result"
        )
        if kind != "failed":
            raise ValueError("FAILED requires a failed result")
        limits = validate_training_api_limits(api_limits)["report"]
        error = validate_gateway_error(item["error"])
        require_training_text(
            error["message"], "result.error.message", limits["max_text_bytes"]
        )
        if "resource_id" in error:
            require_training_text(
                error["resource_id"],
                "result.error.resource_id",
                limits["max_text_bytes"],
            )
        return canonical_tree(
            {
                "type": "failed",
                "error": error,
                "report_digest": require_sha256(item["report_digest"], "report_digest"),
            }
        )
    if state == "CANCELLED":
        item = _exact_object(
            value, {"type", "report_digest"}, "cancelled training result"
        )
        if kind != "cancelled":
            raise ValueError("CANCELLED requires a cancelled result")
        return {
            "type": "cancelled",
            "report_digest": require_sha256(item["report_digest"], "report_digest"),
        }
    raise ValueError("only a terminal training run may have a result")


def validate_primary_metric_observation(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate the latest defined primary metric projected into a Jobs row."""

    item = _exact_object(value, _PRIMARY_METRIC_FIELDS, "primary metric observation")
    limits = validate_training_api_limits(api_limits)
    return canonical_tree(
        {
            "series_id": require_training_stable_id(
                item["series_id"], "primary_metric.series_id"
            ),
            "display_name": require_training_text(
                item["display_name"],
                "primary_metric.display_name",
                limits["report"]["max_text_bytes"],
            ),
            "value": require_finite_number(item["value"], "primary_metric.value"),
            "epoch": require_positive_int32(item["epoch"], "primary_metric.epoch"),
        }
    )


def validate_training_run_summary(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate one bounded training-run row returned by list pagination."""

    item = _object_with_optional(
        value,
        _SUMMARY_REQUIRED_FIELDS,
        _SUMMARY_OPTIONAL_FIELDS,
        "training run summary",
    )
    limits = validate_training_api_limits(api_limits)
    state = item["state"]
    if not isinstance(state, str) or state not in _RUN_STATES:
        raise ValueError("unsupported training run state")
    progress = validate_training_progress(item["progress"], limits)
    if state in {"FINALIZING", "SUCCEEDED"} and (
        progress["completed_epochs"] != progress["total_epochs"]
    ):
        raise ValueError(f"{state} requires every training epoch to be complete")
    created_at = require_rfc3339_millis(item["created_at"], "created_at")
    updated_at = require_rfc3339_millis(item["updated_at"], "updated_at")
    if updated_at < created_at:
        raise ValueError("updated_at cannot precede created_at")

    result = (
        validate_training_run_result(item["result"], state, limits)
        if "result" in item
        else None
    )
    if state in _ACTIVE_STATES and result is not None:
        raise ValueError("an active training run summary cannot have a result")
    if state in _TERMINAL_STATES and result is None:
        raise ValueError("a terminal training run summary requires a result")
    primary_metric = (
        validate_primary_metric_observation(item["primary_metric"], limits)
        if "primary_metric" in item
        else None
    )
    if (
        primary_metric is not None
        and primary_metric["epoch"] > progress["completed_epochs"]
    ):
        raise ValueError("primary metric epoch exceeds completed training progress")

    result_form = item["result_form"]
    if result_form not in _RESULT_FORMS:
        raise ValueError("unsupported training result form")
    normalized: dict[str, Any] = {
        "run_id": require_canonical_uuid(item["run_id"], "run_id"),
        "result_display_name": require_training_text(
            item["result_display_name"],
            "result_display_name",
            limits["report"]["max_text_bytes"],
        ),
        "source_module_ref": str(ModuleRef.parse(item["source_module_ref"])),
        "selected_model_id": require_training_model_id(
            item["selected_model_id"], "selected_model_id"
        ),
        "result_form": result_form,
        "state": state,
        "progress": progress,
        "report_revision": require_positive_int32(
            item["report_revision"], "report_revision"
        ),
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if primary_metric is not None:
        normalized["primary_metric"] = primary_metric
    if result is not None:
        normalized["result"] = result
    return canonical_tree(normalized)


def validate_training_run_page(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate one bounded page of training-run summaries."""

    item = _object_with_optional(
        value,
        _PAGE_REQUIRED_FIELDS,
        _PAGE_OPTIONAL_FIELDS,
        "training run page",
    )
    if not isinstance(item["items"], list):
        raise ValueError("training run page items must be an array")
    limits = validate_training_api_limits(api_limits)
    if len(item["items"]) > limits["max_page_size"]:
        raise ValueError("training run page exceeds max_page_size")
    items = [validate_training_run_summary(entry, limits) for entry in item["items"]]
    keys = [(entry["created_at"], entry["run_id"]) for entry in items]
    expected_keys = sorted(
        keys, key=lambda key: (_descending_timestamp(key[0]), key[1])
    )
    run_ids = [entry["run_id"] for entry in items]
    if (
        keys != expected_keys
        or len(keys) != len(set(keys))
        or len(run_ids) != len(set(run_ids))
    ):
        raise ValueError("training run page is not uniquely ordered")
    result: dict[str, Any] = {"items": items}
    if "next_cursor" in item:
        if not items:
            raise ValueError("empty training run page cannot have next_cursor")
        result["next_cursor"] = require_nonblank_string(
            item["next_cursor"], "next_cursor"
        )
    result = canonical_tree(result)
    if len(canonical_bytes(result)) > limits["max_page_encoded_bytes"]:
        raise ValueError("training run page exceeds max_page_encoded_bytes")
    return result


def maximum_training_run_single_item_page_bytes(api_limits: Any) -> int:
    """Return an upper bound for one generated page containing one run row."""

    limits = validate_training_api_limits(api_limits)
    maximum_epoch = limits["report"]["max_epochs"]
    placeholder = "x"
    summary = {
        "run_id": "ffffffff-ffff-4fff-bfff-ffffffffffff",
        "result_display_name": placeholder,
        "source_module_ref": f"{'a' * 47}/{'b' * 48}",
        "selected_model_id": "a" * 128,
        "result_form": "complete_source_workflow",
        "state": "FAILED",
        "progress": {
            "completed_epochs": maximum_epoch,
            "total_epochs": maximum_epoch,
        },
        "report_revision": 2_147_483_647,
        "created_at": "9999-12-31T23:59:59.999Z",
        "updated_at": "9999-12-31T23:59:59.999Z",
        "primary_metric": {
            "series_id": placeholder,
            "display_name": placeholder,
            "value": -1.7976931348623157e308,
            "epoch": maximum_epoch,
        },
        "result": {
            "type": "failed",
            "error": {
                "schema_version": 1,
                "code": "REGISTRY_RATE_LIMITED",
                "message": placeholder,
                "resource_id": placeholder,
                "retry_not_before": "9999-12-31T23:59:59.999Z",
            },
            "report_digest": "f" * 64,
        },
    }
    page = {
        "items": [summary],
        "next_cursor": "x" * _GENERATED_CURSOR_BYTES,
    }
    # Four TrainingText values can each use two JSON bytes per permitted input
    # byte through quote or backslash escaping. The stable series ID has the
    # same expansion but a fixed 128-byte input limit.
    placeholder_count = 5
    return (
        len(canonical_bytes(page))
        - placeholder_count
        + 8 * limits["report"]["max_text_bytes"]
        + 2 * 128
    )


def validate_training_run_record(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate one complete public training-run record."""

    item = _object_with_optional(
        value, _RUN_REQUIRED_FIELDS, _RUN_OPTIONAL_FIELDS, "training run record"
    )
    limits = validate_training_api_limits(api_limits)
    run_id = require_canonical_uuid(item["run_id"], "run_id")
    request = validate_training_run_request(item["request"], limits)
    request_digest = require_sha256(item["request_digest"], "request_digest")
    if request_digest != training_run_request_digest(request, limits):
        raise ValueError("training run request_digest does not match request")
    state = item["state"]
    if not isinstance(state, str) or state not in _RUN_STATES:
        raise ValueError("unsupported training run state")
    progress = validate_training_progress(item["progress"], limits)
    report_revision = require_positive_int32(item["report_revision"], "report_revision")
    created_at = require_rfc3339_millis(item["created_at"], "created_at")
    updated_at = require_rfc3339_millis(item["updated_at"], "updated_at")
    if updated_at < created_at:
        raise ValueError("updated_at cannot precede created_at")

    finalization = (
        validate_training_finalization(item["finalization"])
        if "finalization" in item
        else None
    )
    result = (
        validate_training_run_result(item["result"], state, limits)
        if "result" in item
        else None
    )
    if state in _ACTIVE_STATES and result is not None:
        raise ValueError("an active training run cannot have a terminal result")
    if state in _TERMINAL_STATES and result is None:
        raise ValueError("a terminal training run requires its matching result")
    if state in {"QUEUED", "RUNNING", "CANCELLED"} and finalization is not None:
        raise ValueError(f"{state} cannot have finalization")
    if state in {"FINALIZING", "SUCCEEDED"} and finalization is None:
        raise ValueError(f"{state} requires finalization")
    if finalization is not None and (
        progress["completed_epochs"] != progress["total_epochs"]
    ):
        raise ValueError("finalization requires every training epoch to be complete")

    normalized: dict[str, Any] = {
        "run_id": run_id,
        "request": request,
        "request_digest": request_digest,
        "trainer_build": parse_software_build_identity(item["trainer_build"]),
        "state": state,
        "progress": progress,
        "report_revision": report_revision,
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if finalization is not None:
        normalized["finalization"] = finalization
    if result is not None:
        normalized["result"] = result
    return canonical_tree(normalized)


def validate_training_run_control_record(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate the public run, private worker control, and save-attempt fence."""

    item = _exact_object(value, _CONTROL_FIELDS, "training run control record")
    run = validate_training_run_record(item["run"], api_limits)
    worker = validate_training_worker_control(item["worker"])
    finalization_revision = require_nonnegative_int32(
        item["finalization_revision"], "finalization_revision"
    )
    state = run["state"]
    call_attached = "worker_call_id" in worker

    if worker["call_attachment_deadline_at"] <= run["created_at"]:
        raise ValueError("call attachment deadline must follow run acceptance")
    if worker["execution_deadline_at"] <= run["created_at"]:
        raise ValueError("execution deadline must follow run acceptance")
    if state == "QUEUED" and (call_attached or worker["worker_claimed"]):
        raise ValueError("QUEUED cannot have an attached or claimed worker")
    if state in {"RUNNING", "FINALIZING", "SUCCEEDED"} and not call_attached:
        raise ValueError(f"{state} requires an attached worker call")
    if state in {"FINALIZING", "SUCCEEDED"} and not worker["worker_claimed"]:
        raise ValueError(f"{state} requires a claimed worker")
    if run["progress"]["completed_epochs"] > 0 and not worker["worker_claimed"]:
        raise ValueError("training progress requires a claimed worker")
    if "finalization" not in run and finalization_revision != 0:
        raise ValueError("finalization_revision must be zero before finalization")
    if (
        "finalization" in run
        and state in _TERMINAL_STATES
        and not finalization_revision
    ):
        raise ValueError("completed finalization requires a positive revision")
    return canonical_tree(
        {
            "run": run,
            "worker": worker,
            "finalization_revision": finalization_revision,
        }
    )


def is_active_training_run(record: Any, api_limits: Any) -> bool:
    """Return whether a validated control record still consumes run capacity."""

    return (
        validate_training_run_control_record(record, api_limits)["run"]["state"]
        in _ACTIVE_STATES
    )


def terminal_training_report_identity(value: Any, api_limits: Any) -> dict[str, Any]:
    """Return the sole report revision and digest retained by a terminal run."""

    run = validate_training_run_record(value, api_limits)
    if run["state"] not in _TERMINAL_STATES:
        raise ValueError("terminal report identity requires a terminal run")
    digest = (
        run["finalization"]["report_digest"]
        if run["state"] == "SUCCEEDED"
        else run["result"]["report_digest"]
    )
    return canonical_tree(
        {
            "report_revision": run["report_revision"],
            "report_digest": require_sha256(digest, "report_digest"),
        }
    )


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def _object_with_optional(
    value: Any,
    required: set[str],
    optional: set[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    fields = set(value)
    if not required <= fields or not fields <= required | optional:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def _descending_timestamp(value: str) -> tuple[int, ...]:
    # Exact UTC-millisecond text has fixed width and lexical chronological order.
    return tuple(-ord(character) for character in value)
