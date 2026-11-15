"""Recover and expose one complete managed-training result candidate."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from histoforge.canonical_json import canonical_tree, require_canonical_uuid
from histoforge.cloud._errors import GatewayErrorCode, build_gateway_error
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt, ArtifactBytesMissing
from histoforge.cloud.modules.identity import validate_stored_module_record
from histoforge.cloud.modules.store import ModuleInstallRejected, TrainingResultConflict
from histoforge.cloud.modules.training_evidence import (
    validate_training_result_attestation,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.contract import (
    terminal_training_report_identity,
    validate_training_run_control_record,
    validate_training_run_record,
)
from histoforge.cloud.training.runs.terminalizer import TrainingRunTerminalizer

_LOG = logging.getLogger(__name__)


class TrainingFinalizationIndeterminate(RuntimeError):
    """A module may be installed, so reconciliation must remain FINALIZING."""


class TrainingRunFinalizer:
    """Freeze a candidate, delegate saving, and clean up terminal artifacts."""

    def __init__(
        self,
        run_files: Any,
        result_files: Any,
        source_files: Any,
        control: Any,
        finalize_result: Callable[[str], Any],
        api_limits: Any,
    ) -> None:
        self._run_files = run_files
        self._result_files = result_files
        self._source_files = source_files
        self._control = control
        self._finalize_result = finalize_result
        self._api_limits = validate_training_api_limits(api_limits)

    def reconcile_candidate(self, run_id: str) -> dict[str, Any] | None:
        """Finish a complete candidate, or return ``None`` when none exists yet."""

        control = self._current(run_id)
        run = control["run"]
        if run["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return run
        if run["state"] == "RUNNING":
            candidate = self._result_files.inspect_candidate(run_id)
            if candidate is None:
                return None
            run = self._adopt_candidate(run, candidate)
            if run["state"] != "FINALIZING":
                return run
        elif run["state"] != "FINALIZING":
            return None

        try:
            outcome = validate_training_run_record(
                self._finalize_result(run_id), self._api_limits
            )
            if outcome["run_id"] != run_id:
                raise ValueError("training finalization returned a different run")
        except Exception as error:
            raise TrainingFinalizationIndeterminate(
                "training save outcome is not yet known"
            ) from error
        if outcome["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            self.cleanup_terminal(outcome)
        return outcome

    def _adopt_candidate(self, run: dict[str, Any], candidate: Any) -> dict[str, Any]:
        success_report = canonical_tree(candidate.report)
        if run["progress"]["completed_epochs"] < run["progress"]["total_epochs"]:
            running_report = _live_report(success_report, "RUNNING")
            published = self._run_files.publish_report_candidate(running_report)
            run = validate_training_run_record(
                self._control.advance_run_progress(
                    run["run_id"],
                    run["report_revision"] + 1,
                    published["report_digest"],
                ),
                self._api_limits,
            )

        finalizing_report = _live_report(success_report, "FINALIZING")
        published = self._run_files.publish_report_candidate(finalizing_report)
        return validate_training_run_record(
            self._control.begin_run_finalization(
                run["run_id"],
                run["report_revision"] + 1,
                published["report_digest"],
                success_report,
                candidate.module_content_digest,
            ),
            self._api_limits,
        )

    def _current(self, run_id: str) -> dict[str, Any]:
        return validate_training_run_control_record(
            self._control.get_run_control(run_id), self._api_limits
        )

    def cleanup_terminal(self, value: Any) -> None:
        """Best-effort removal of bytes not retained by one terminal run."""

        run = validate_training_run_record(value, self._api_limits)
        if run["state"] not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            raise ValueError("training cleanup requires a terminal run")
        run_id = run["run_id"]
        report = terminal_training_report_identity(run, self._api_limits)

        for cleanup in (
            self._source_files.delete_source_roots_after_fence,
            self._result_files.delete_candidate_after_fence,
            self._run_files.delete_transients,
        ):
            try:
                cleanup(run_id)
            except Exception:
                _LOG.warning(
                    "training transient cleanup failed for run %s",
                    run_id,
                    exc_info=True,
                )
        try:
            self._run_files.prune_report_history(
                run_id,
                report["report_revision"],
                report["report_digest"],
            )
        except Exception:
            _LOG.warning(
                "training terminal-report cleanup failed for run %s",
                run_id,
                exc_info=True,
            )


def finalize_training_result(
    store: Any,
    run_files: Any,
    result_files: Any,
    control: Any,
    run_id: str,
    publish_volume: Callable[[], None],
    api_limits: Any,
) -> dict[str, Any]:
    """Own one save decision inside the sole serialized module writer.

    Acknowledging the next private counter fences late terminal RPCs from an
    older attempt. Only proven absence/rejection or a verified conflict may
    fail the run; all unknown reads, writes and RPC outcomes propagate for
    later reconciliation, without a new failure decision outside this lane.
    """

    canonical_id = require_canonical_uuid(run_id, "run_id")
    observed = validate_training_run_control_record(
        control.get_run_control(canonical_id), api_limits
    )
    run = observed["run"]
    if run["run_id"] != canonical_id:
        raise ValueError("training control belongs to another run")
    if run["state"] in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        return run
    if run["state"] != "FINALIZING":
        raise ValueError("training result can be saved only while finalizing")

    claimed = validate_training_run_control_record(
        control.claim_run_finalization(canonical_id, observed["finalization_revision"]),
        api_limits,
    )
    if claimed != {
        **observed,
        "finalization_revision": observed["finalization_revision"] + 1,
    }:
        raise ValueError("training save claim differs from its expected successor")

    def fail(message: str) -> dict[str, Any]:
        error = build_gateway_error(
            GatewayErrorCode.INTERNAL_ERROR, message, resource_id=canonical_id
        )
        return TrainingRunTerminalizer(run_files, control, api_limits).fail(
            claimed, error
        )["run"]

    finalization = run["finalization"]
    module_ref = f"workspace/run-{canonical_id}"
    attestation = validate_training_result_attestation(
        {
            "run_id": canonical_id,
            "module_content_digest": finalization["module_content_digest"],
            "report_digest": finalization["report_digest"],
        }
    )
    try:
        installed = store.matching_training_result(module_ref, attestation)
    except TrainingResultConflict:
        return fail("a different training result already exists at the reserved ref")

    if installed is not None:
        # Recover without transient candidate bytes or fresh-install limit checks.
        report = store.current.load_training_report(module_ref, record=installed)
        published = run_files.publish_report_candidate(report)
    else:
        try:
            candidate = result_files.require_candidate(
                canonical_id,
                finalization["module_content_digest"],
                finalization["report_digest"],
            )
            if (
                candidate.selected_weight_digest
                != finalization["selected_weight_digest"]
            ):
                raise ValueError("training result selected weights differ")
            published = run_files.publish_report_candidate(candidate.report)
            if published["report_digest"] != finalization["report_digest"]:
                raise ValueError("successful training report digest changed")
        except (
            ArtifactBytesCorrupt,
            ArtifactBytesMissing,
            FileNotFoundError,
            ValueError,
        ):
            return fail("the training result candidate is invalid or incomplete")
        try:
            installed = store.install_from_directory(
                candidate.module_dir,
                module_ref=module_ref,
                resolved_commit_sha=None,
                job_id=canonical_id,
                publish=publish_volume,
                external_artifacts=candidate.external_artifacts,
                training_result_attestation=attestation,
            )
        except ModuleInstallRejected:
            return fail(
                "the training result exceeds the module storage or catalog limits"
            )
        except TrainingResultConflict:
            return fail(
                "a different training result already exists at the reserved ref"
            )

    installed = validate_stored_module_record(installed)
    if (
        installed["module_ref"] != module_ref
        or installed["resolved_commit_sha"] is not None
        or installed.get("training_result_attestation") != attestation
        or published["report_digest"] != finalization["report_digest"]
    ):
        raise ValueError("installed training result differs from finalization")
    return validate_training_run_record(
        control.succeed_run(
            canonical_id,
            run["report_revision"] + 1,
            finalization["report_digest"],
            claimed["finalization_revision"],
        ),
        api_limits,
    )


def _live_report(success_report: dict[str, Any], state: str) -> dict[str, Any]:
    report = canonical_tree(success_report)
    report["state"] = state
    report["unavailable_metrics"] = []
    report.pop("failure", None)
    if state == "RUNNING":
        report.pop("model_selection", None)
    return report
