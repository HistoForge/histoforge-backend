"""Read, cancel, and clear managed-training runs outside the state lane."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.contract import (
    validate_training_run_control_record,
    validate_training_run_record,
)
from histoforge.cloud.training.runs.registry import (
    training_run_page,
    validate_training_run_record_selection,
)
from histoforge.cloud.training.runs.report import (
    validate_training_report_response_for_run,
)

_LOG = logging.getLogger(__name__)


class TrainingRunAccess:
    """Adapt public run operations to report files and serialized state cuts."""

    def __init__(
        self,
        run_files: Any,
        control: Any,
        reconciler: Any,
        terminalizer: Any,
        call_from_id: Callable[[str], Any],
        api_limits: Any,
    ) -> None:
        self._run_files = run_files
        self._control = control
        self._reconciler = reconciler
        self._terminalizer = terminalizer
        self._call_from_id = call_from_id
        self._api_limits = validate_training_api_limits(api_limits)

    def get(self, run_id: str) -> dict[str, Any]:
        """Return one complete public run record."""

        return validate_training_run_record(
            self._reconciler.reconcile(run_id), self._api_limits
        )

    def page(self, limit: int, cursor: str | None = None) -> dict[str, Any]:
        """Return one stable page of Jobs rows."""

        runs = validate_training_run_record_selection(
            self._control.page_run_records(limit, cursor),
            limit,
            self._api_limits,
        )
        runs = [
            validate_training_run_record(
                self._reconciler.reconcile(run["run_id"]), self._api_limits
            )
            for run in runs
        ]
        by_id = {run["run_id"]: run for run in runs}
        return training_run_page(
            runs,
            lambda run_id: {
                "report_revision": by_id[run_id]["report_revision"],
                **self._run_files.read_report(
                    run_id,
                    by_id[run_id]["report_revision"],
                ),
            },
            limit,
            None,
            self._api_limits,
        )

    def report(self, run_id: str) -> dict[str, Any]:
        """Return only the run's currently advertised immutable report."""

        run = validate_training_run_record(
            self._reconciler.reconcile(run_id), self._api_limits
        )
        return validate_training_report_response_for_run(
            {
                "report_revision": run["report_revision"],
                **self._run_files.read_report(run_id, run["report_revision"]),
            },
            run,
            self._api_limits,
        )

    def cancel(self, run_id: str) -> dict[str, Any]:
        """Commit cancellation, then stop the now-fenced physical worker call."""

        control = validate_training_run_control_record(
            self._terminalizer.cancel(run_id), self._api_limits
        )
        call_id = control["worker"].get("worker_call_id")
        if call_id is not None:
            self._cancel_physical_call(run_id, call_id)
        return validate_training_run_record(
            self._reconciler.reconcile(run_id), self._api_limits
        )

    def delete(self, run_id: str) -> None:
        """Delete one terminal report history, then remove its Jobs record."""

        run = self._control.begin_run_delete(run_id)
        if run is None:
            return
        validate_training_run_record(run, self._api_limits)
        self._run_files.delete_reports(run_id)
        self._control.finish_run_delete(run_id)

    def _cancel_physical_call(self, run_id: str, call_id: str) -> None:
        try:
            self._call_from_id(call_id).cancel()
        except Exception:
            # The committed CANCELLED state already fences every worker callback.
            # Physical cancellation only avoids unnecessary provider work.
            _LOG.warning(
                "physical training-call cancellation failed for run %s",
                run_id,
                exc_info=True,
            )
