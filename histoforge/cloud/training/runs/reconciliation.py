"""Read-triggered reconciliation for one durable managed-training run."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import modal

from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    build_gateway_error,
    is_automatically_retryable,
)
from histoforge.cloud._time import iso_z
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.contract import (
    validate_training_run_control_record,
    validate_training_run_record,
)
from histoforge.cloud.training.runs.finalizer import TrainingFinalizationIndeterminate

_LOG = logging.getLogger(__name__)
_TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}


class TrainingRunReconciler:
    """Project one provider call and verified artifacts into durable run state."""

    def __init__(
        self,
        control: Any,
        finalizer: Any,
        terminalizer: Any,
        call_from_id: Callable[[str], Any],
        api_limits: Any,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._control = control
        self._finalizer = finalizer
        self._terminalizer = terminalizer
        self._call_from_id = call_from_id
        self._api_limits = validate_training_api_limits(api_limits)
        self._clock = clock

    def reconcile(self, run_id: str) -> dict[str, Any]:
        """Return the current run after one bounded reconciliation attempt."""

        control = self._current(run_id)
        run = control["run"]
        if run["state"] in _TERMINAL_STATES:
            self._finalizer.cleanup_terminal(run)
            return run
        if run["state"] == "FINALIZING":
            return self._finish_candidate(control)
        if run["state"] == "QUEUED":
            if self._deadline_reached(control["worker"]["call_attachment_deadline_at"]):
                return self._fail(
                    control,
                    GatewayErrorCode.RUNNER_FAILED,
                    "training worker was not attached before its deadline",
                )
            return run

        worker = control["worker"]
        if self._deadline_reached(worker["execution_deadline_at"]):
            return self._fail(
                control,
                GatewayErrorCode.RUNNER_FAILED,
                "training exceeded its execution deadline",
                cancel_call=True,
            )

        call_id = worker["worker_call_id"]
        try:
            call = self._call_from_id(call_id)
        except modal.exception.NotFoundError:
            return self._recover_or_wait(control)
        except (ConnectionError, OSError, modal.exception.Error):
            return self._recover_or_wait(control)
        except Exception:
            return self._recover_or_wait(control)

        try:
            call.get(timeout=0)
        except TimeoutError:
            return run
        except (modal.exception.NotFoundError, modal.exception.OutputExpiredError):
            return self._recover_or_wait(control)
        except GatewayFailure as failure:
            return self._recover_or_fail(control, error=failure.error)
        except (
            modal.exception.FunctionTimeoutError,
            modal.exception.RemoteError,
            modal.exception.InternalFailure,
            modal.exception.ExecutionError,
        ):
            return self._recover_or_fail(control)
        except (ConnectionError, OSError, modal.exception.Error):
            return self._recover_or_wait(control)
        except Exception:
            return self._recover_or_fail(control)

        return self._recover_or_fail(control)

    def _recover_or_wait(self, control: dict[str, Any]) -> dict[str, Any]:
        run = control["run"]
        recovered = self._finish_candidate(control)
        if recovered is not None:
            return recovered
        if self._deadline_reached(control["worker"]["execution_deadline_at"]):
            return self._fail(
                control,
                GatewayErrorCode.RUNNER_FAILED,
                "training ended without a retained outcome or complete result",
                cancel_call=True,
            )
        return run

    def _recover_or_fail(
        self, control: dict[str, Any], *, error: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        recovered = self._finish_candidate(control)
        if recovered is not None:
            return recovered
        return self._fail(
            control,
            GatewayErrorCode.RUNNER_FAILED,
            "training worker stopped before completing the run",
            error=error,
        )

    def _finish_candidate(
        self,
        observed: dict[str, Any],
    ) -> dict[str, Any] | None:
        previous = observed["run"]
        run_id = previous["run_id"]
        try:
            result = self._finalizer.reconcile_candidate(run_id)
        except GatewayFailure as failure:
            current = self._current(run_id)["run"]
            if (
                current["state"] in _TERMINAL_STATES
                or current["state"] != previous["state"]
                or previous["state"] == "FINALIZING"
            ):
                return current
            if failure.error["code"] == GatewayErrorCode.TRAINING_STATE_CONFLICT.value:
                if current["report_revision"] != previous["report_revision"]:
                    return current
                return self._fail(
                    observed,
                    GatewayErrorCode.INTERNAL_ERROR,
                    "the training result contradicts its advertised report history",
                )
            if is_automatically_retryable(failure.error["code"]):
                return current
            return self._fail(observed, error=failure.error)
        except (
            ArtifactBytesCorrupt,
            ArtifactBytesMissing,
            FileNotFoundError,
            ValueError,
        ):
            if previous["state"] == "FINALIZING":
                return self._current_or(previous)
            return self._fail(
                observed,
                GatewayErrorCode.INTERNAL_ERROR,
                "the training result candidate is invalid or incomplete",
            )
        except TrainingFinalizationIndeterminate:
            return self._current_or(previous)
        except (ConnectionError, TimeoutError, OSError, modal.exception.Error):
            return self._current_or(previous)
        except Exception:
            _LOG.warning(
                "training finalization is temporarily indeterminate for run %s",
                run_id,
                exc_info=True,
            )
            return self._current_or(previous)
        if result is None and previous["state"] == "FINALIZING":
            return self._current_or(previous)
        return result

    def _fail(
        self,
        control: dict[str, Any],
        code: GatewayErrorCode | None = None,
        message: str | None = None,
        *,
        error: dict[str, Any] | None = None,
        cancel_call: bool = False,
    ) -> dict[str, Any]:
        run = control["run"]
        failure = error or build_gateway_error(
            code,
            message,
            resource_id=run["run_id"],
        )
        failed = validate_training_run_control_record(
            self._terminalizer.fail(control, failure), self._api_limits
        )
        if cancel_call or self._deadline_reached(
            failed["worker"]["execution_deadline_at"]
        ):
            call_id = failed["worker"].get("worker_call_id")
            if call_id is not None:
                try:
                    self._call_from_id(call_id).cancel()
                except Exception:
                    _LOG.warning(
                        "expired training-call cancellation failed for run %s",
                        run["run_id"],
                        exc_info=True,
                    )
        result = validate_training_run_record(failed["run"], self._api_limits)
        if result["state"] in _TERMINAL_STATES:
            self._finalizer.cleanup_terminal(result)
        return result

    def _current(self, run_id: str) -> dict[str, Any]:
        return validate_training_run_control_record(
            self._control.get_run_control(run_id), self._api_limits
        )

    def _current_or(self, previous: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._current(previous["run_id"])["run"]
        except Exception:
            return previous

    def _deadline_reached(self, deadline: str) -> bool:
        return iso_z(self._clock()) >= deadline
