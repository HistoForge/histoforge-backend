"""Bounded training-run list projection and exclusive cursor."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
    require_positive_int32,
    require_rfc3339_millis,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.contract import (
    validate_training_run_page,
    validate_training_run_record,
    validate_training_run_summary,
)
from histoforge.cloud.training.runs.report import (
    validate_training_report,
    validate_training_report_response_for_run,
)

_CURSOR_FIELDS = {"created_at", "run_id"}
_CURSOR_TEXT = re.compile(r"[A-Za-z0-9_-]+")


def training_run_page(
    runs: Sequence[Mapping[str, Any]],
    report_for_run: Callable[[str], Mapping[str, Any]],
    limit: Any,
    cursor: str | None,
    api_limits: Any,
) -> dict[str, Any]:
    """Return a stable, byte-bounded page of current run summaries."""

    limits = validate_training_api_limits(api_limits)
    page_limit = _page_limit(limit, limits)
    normalized = select_training_run_records(
        runs,
        page_limit,
        cursor,
        limits,
    )

    selected: list[dict[str, Any]] = []
    for index, run in enumerate(normalized[:page_limit]):
        try:
            response = validate_training_report_response_for_run(
                report_for_run(run["run_id"]), run, limits
            )
        except (TypeError, ValueError) as error:
            raise _invalid(run["run_id"]) from error
        report = response["report"]
        summary = project_training_run_summary(run, report, limits)
        candidate_items = [*selected, summary]
        candidate: dict[str, Any] = {"items": candidate_items}
        if index + 1 < len(normalized):
            candidate["next_cursor"] = _encode_cursor(run)
        if len(canonical_bytes(candidate)) > limits["max_page_encoded_bytes"]:
            break
        selected.append(summary)
    if normalized and not selected:
        raise _invalid("training-runs")

    result: dict[str, Any] = {"items": selected}
    if len(selected) < len(normalized):
        result["next_cursor"] = _encode_cursor(selected[-1])
    return validate_training_run_page(result, limits)


def select_training_run_records(
    runs: Sequence[Mapping[str, Any]],
    limit: Any,
    cursor: str | None,
    api_limits: Any,
) -> list[dict[str, Any]]:
    """Select at most one page plus one lookahead without reading reports."""

    limits = validate_training_api_limits(api_limits)
    page_limit = _page_limit(limit, limits)
    after = _decode_cursor(cursor) if cursor is not None else None
    normalized = [validate_training_run_record(run, limits) for run in runs]
    normalized.sort(key=_run_order)
    if after is not None:
        after_key = _run_order(after)
        normalized = [run for run in normalized if _run_order(run) > after_key]
    return normalized[: page_limit + 1]


def validate_training_run_record_selection(
    value: Any,
    limit: Any,
    api_limits: Any,
) -> list[dict[str, Any]]:
    """Validate the ordered state-lane snapshot consumed by list projection."""

    limits = validate_training_api_limits(api_limits)
    page_limit = _page_limit(limit, limits)
    if not isinstance(value, list) or len(value) > page_limit + 1:
        raise ValueError("training run record selection is not bounded")
    records = [validate_training_run_record(run, limits) for run in value]
    keys = [_run_order(run) for run in records]
    run_ids = [run["run_id"] for run in records]
    if (
        keys != sorted(keys)
        or len(keys) != len(set(keys))
        or len(run_ids) != len(set(run_ids))
    ):
        raise ValueError("training run record selection is not uniquely ordered")
    return records


def project_training_run_summary(
    run: Any, report: Any, api_limits: Any
) -> dict[str, Any]:
    """Project one validated run and its advertised report into a Jobs row."""

    limits = validate_training_api_limits(api_limits)
    accepted = validate_training_run_record(run, limits)
    current = validate_training_report(report, limits)
    if (
        current["run_id"] != accepted["run_id"]
        or current["state"]
        not in {
            "QUEUED": {"QUEUED"},
            "RUNNING": {"QUEUED", "RUNNING"},
            "FINALIZING": {"FINALIZING"},
            "SUCCEEDED": {"SUCCEEDED"},
            "FAILED": {"FAILED"},
            "CANCELLED": {"CANCELLED"},
        }[accepted["state"]]
        or _last_epoch(current) != accepted["progress"]["completed_epochs"]
    ):
        raise _invalid(accepted["run_id"])

    request = accepted["request"]
    summary: dict[str, Any] = {
        "run_id": accepted["run_id"],
        "result_display_name": request["result_display_name"],
        "source_module_ref": request["source_module"]["module_ref"],
        "selected_model_id": request["selected_model_id"],
        "result_form": request["result_form"],
        "state": accepted["state"],
        "progress": accepted["progress"],
        "report_revision": accepted["report_revision"],
        "created_at": accepted["created_at"],
        "updated_at": accepted["updated_at"],
    }
    primary = _latest_primary_metric(current)
    if primary is not None:
        summary["primary_metric"] = primary
    if "result" in accepted:
        summary["result"] = accepted["result"]
    return validate_training_run_summary(summary, limits)


def _latest_primary_metric(report: Mapping[str, Any]) -> dict[str, Any] | None:
    primary_id = report["metric_policy"]["primary_validation_series_id"]
    series = next(
        entry
        for entry in report["metric_policy"]["series"]
        if entry["series_id"] == primary_id
    )
    latest = None
    for epoch in report["epochs"]:
        for metric in epoch["metrics"]:
            if metric["series_id"] == primary_id:
                latest = {
                    "series_id": primary_id,
                    "display_name": series["display_name"],
                    "value": metric["value"],
                    "epoch": epoch["epoch"],
                }
    return latest


def _last_epoch(report: Mapping[str, Any]) -> int:
    return report["epochs"][-1]["epoch"] if report["epochs"] else 0


def _page_limit(value: Any, limits: Mapping[str, Any]) -> int:
    page_limit = require_positive_int32(value, "limit")
    if page_limit > limits["max_page_size"]:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "limit exceeds the training run page maximum",
        )
    return page_limit


def _run_order(run: Mapping[str, Any]) -> tuple[tuple[int, ...], str]:
    created_at = require_rfc3339_millis(run["created_at"], "created_at")
    run_id = require_canonical_uuid(run["run_id"], "run_id")
    return tuple(-ord(character) for character in created_at), run_id


def _encode_cursor(run: Mapping[str, Any]) -> str:
    encoded = canonical_bytes(
        {"created_at": run["created_at"], "run_id": run["run_id"]}
    )
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> dict[str, str]:
    if not isinstance(value, str) or not _CURSOR_TEXT.fullmatch(value):
        raise _invalid_cursor()
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        cursor = parse_json(decoded)
        if not isinstance(cursor, dict) or set(cursor) != _CURSOR_FIELDS:
            raise ValueError("invalid cursor fields")
        result = {
            "created_at": require_rfc3339_millis(cursor["created_at"], "created_at"),
            "run_id": require_canonical_uuid(cursor["run_id"], "run_id"),
        }
    except (UnicodeDecodeError, ValueError, binascii.Error) as error:
        raise _invalid_cursor() from error
    if _encode_cursor(result) != value:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "noncanonical training run cursor",
        )
    return result


def _invalid_cursor() -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.VALIDATION_ERROR, "invalid training run cursor"
    )


def _invalid(run_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.INTERNAL_ERROR,
        "training run state is inconsistent",
        resource_id=run_id,
    )
