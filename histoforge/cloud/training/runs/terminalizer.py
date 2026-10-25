"""Construct terminal run reports outside the serialized training-state lane."""

from __future__ import annotations

import logging
from typing import Any

from histoforge.canonical_json import canonical_tree, require_positive_int32
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    validate_gateway_error,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.contract import (
    validate_training_run_control_record,
)
from histoforge.cloud.training.runs.report import (
    validate_training_report_response_for_run,
)

_LOG = logging.getLogger(__name__)
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


class TrainingRunTerminalizer:
    """Publish a terminal report, then commit it against one current run revision."""

    def __init__(self, run_files: Any, control: Any, api_limits: Any) -> None:
        self._run_files = run_files
        self._control = control
        self._api_limits = validate_training_api_limits(api_limits)

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Cancel the latest still-cancellable revision and return its control."""

        current = self._current(run_id)
        while True:
            state = current["run"]["state"]
            if state == "CANCELLED":
                return current
            if state not in {"QUEUED", "RUNNING"}:
                raise self._conflict(run_id, "training run can no longer be cancelled")
            try:
                return self._commit(current, "CANCELLED")
            except GatewayFailure as failure:
                if (
                    failure.error["code"]
                    != GatewayErrorCode.TRAINING_STATE_CONFLICT.value
                ):
                    raise
                latest = self._current(run_id)
                if latest["run"]["state"] == "CANCELLED":
                    return latest
                if latest["run"]["state"] not in {"QUEUED", "RUNNING"}:
                    raise self._conflict(
                        run_id, "training run can no longer be cancelled"
                    ) from failure
                if _version(latest) == _version(current):
                    raise
                current = latest

    def fail(self, observed: Any, error: Any) -> dict[str, Any]:
        """Rebuild after progress, but never rebase a finalization decision."""

        initial = validate_training_run_control_record(observed, self._api_limits)
        expected_state = initial["run"]["state"]
        if expected_state == "FINALIZING":
            require_positive_int32(
                initial["finalization_revision"], "finalization_revision"
            )
        current = self._current(initial["run"]["run_id"])
        while True:
            state = current["run"]["state"]
            if (
                state in _TERMINAL_STATES
                or state != expected_state
                or current["finalization_revision"] != initial["finalization_revision"]
            ):
                return current
            try:
                return self._commit(current, "FAILED", error)
            except GatewayFailure as failure:
                if (
                    failure.error["code"]
                    != GatewayErrorCode.TRAINING_STATE_CONFLICT.value
                ):
                    raise
                latest = self._current(initial["run"]["run_id"])
                if (
                    latest["run"]["state"] in _TERMINAL_STATES
                    or latest["run"]["state"] != expected_state
                    or latest["finalization_revision"]
                    != initial["finalization_revision"]
                ):
                    return latest
                if _version(latest) == _version(current):
                    raise
                current = latest

    def _commit(
        self,
        control: dict[str, Any],
        state: str,
        error: Any | None = None,
    ) -> dict[str, Any]:
        run = control["run"]
        response = validate_training_report_response_for_run(
            {
                "report_revision": run["report_revision"],
                **self._run_files.read_report(run["run_id"], run["report_revision"]),
            },
            run,
            self._api_limits,
        )
        report = canonical_tree(response["report"])
        report["state"] = state
        if state == "FAILED":
            report["failure"] = validate_gateway_error(error)
        else:
            report.pop("failure", None)
        candidate = self._run_files.publish_report_candidate(report)
        try:
            if state == "CANCELLED":
                self._control.cancel_run(
                    run["run_id"],
                    run["report_revision"] + 1,
                    candidate["report_digest"],
                )
            else:
                fence = (
                    {"finalization_revision": control["finalization_revision"]}
                    if run["state"] == "FINALIZING"
                    else {}
                )
                self._control.fail_run(
                    run["run_id"],
                    run["state"],
                    run["report_revision"] + 1,
                    candidate["report_digest"],
                    **fence,
                )
        finally:
            try:
                self._run_files.discard_report_candidate(
                    run["run_id"], candidate["report_digest"]
                )
            except Exception:
                _LOG.warning(
                    "training terminal-report candidate cleanup failed for run %s",
                    run["run_id"],
                    exc_info=True,
                )
        return self._current(run["run_id"])

    def _current(self, run_id: str) -> dict[str, Any]:
        return validate_training_run_control_record(
            self._control.get_run_control(run_id), self._api_limits
        )

    @staticmethod
    def _conflict(run_id: str, message: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.TRAINING_STATE_CONFLICT,
            message,
            resource_id=run_id,
        )


def _version(control: dict[str, Any]) -> tuple[str, int]:
    run = control["run"]
    return run["state"], run["report_revision"]
