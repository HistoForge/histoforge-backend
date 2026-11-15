"""Run-submission orchestration around short training-state cuts."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from histoforge.canonical_json import require_positive_int32, require_sha256
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.training.limits import (
    validate_training_api_limits,
    validate_training_service_limits,
)
from histoforge.cloud.training.runs.contract import (
    validate_training_run_admission_record,
    validate_training_run_record,
    validate_training_worker_call_id,
)
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)

_LOG = logging.getLogger(__name__)


class PreparedTrainingRun(Protocol):
    """Scientific preflight result consumed by run admission."""

    trainer_build: Mapping[str, Any]
    total_epochs: int
    trainer_execution_seconds: int
    worker_resources: Mapping[str, Any]

    def initial_report(self, run_id: str) -> Mapping[str, Any]:
        """Build the QUEUED report after the state owner allocates ``run_id``."""


class TrainingRunService:
    """Publish one accepted run and dispatch its sole logical worker attempt."""

    def __init__(
        self,
        source_preparation: Any,
        run_files: Any,
        source_files: Any,
        result_files: Any,
        control: Any,
        prepare_run: Callable[
            [Mapping[str, Any], Mapping[str, Any], str], PreparedTrainingRun
        ],
        spawn_run: Callable[[str, Mapping[str, Any], int], Any],
        api_limits: Any,
        service_limits: Any,
    ) -> None:
        self._source_preparation = source_preparation
        self._run_files = run_files
        self._source_files = source_files
        self._result_files = result_files
        self._control = control
        self._prepare_run = prepare_run
        self._spawn_run = spawn_run
        self._api_limits = validate_training_api_limits(api_limits)
        self._service_limits = validate_training_service_limits(service_limits)

    def submit(self, value: Any) -> dict[str, str]:
        """Preflight, accept, dispatch, and attach one idempotent training run."""

        request = self._validate_request(value)
        existing = self._lookup(request)
        if existing is not None and existing["type"] == "accepted":
            return _accepted_response(existing["run"])

        self.cleanup_stale_artifacts()
        candidate: tuple[str, str] | None = None
        try:
            with self._source_preparation.prepare(request) as source:
                prepared = self._prepare_run_checked(
                    source.request,
                    source.recipe,
                    source.starting_weight_digest,
                )
                reservation = self._reservation(
                    self._control.reserve_run(source.request), source.request
                )
                if reservation["type"] == "accepted":
                    return _accepted_response(reservation["run"])

                run_id = reservation["admission"]["run_id"]
                report = prepared.initial_report(run_id)
                report_digest = self._run_files.publish_report_candidate(report)[
                    "report_digest"
                ]
                candidate = (run_id, report_digest)
                source.publish(run_id, report_digest)
                accepted = self._accept(
                    source.request,
                    prepared,
                    report_digest,
                )

            run = accepted["accepted"]
            if accepted["should_dispatch"]:
                self._dispatch_and_attach(run["run_id"], prepared)
            return _accepted_response(run)
        finally:
            if candidate is not None:
                self._discard_candidates(*candidate)

    def cleanup_stale_artifacts(self) -> None:
        """Run one bounded pass over abandoned and unowned run storage."""

        self._cleanup_abandoned_admissions()
        self._cleanup_transient_roots()

    def _lookup(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        value = self._control.lookup_run(request)
        if value is None:
            return None
        return self._reservation(value, request)

    def _reservation(self, value: Any, request: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise self._invalid_control_result()
        kind = value.get("type")
        try:
            if kind == "accepted" and set(value) == {"type", "run"}:
                run = validate_training_run_record(value["run"], self._api_limits)
                if run["request"] != request:
                    raise ValueError("accepted request differs")
                return {"type": kind, "run": run}
            if kind == "preparing" and set(value) == {"type", "admission"}:
                admission = validate_training_run_admission_record(value["admission"])
                if admission["request_digest"] != training_run_request_digest(
                    request, self._api_limits
                ):
                    raise ValueError("admission request digest differs")
                return {"type": kind, "admission": admission}
        except ValueError as error:
            raise self._invalid_control_result() from error
        raise self._invalid_control_result()

    def _accept(
        self,
        request: Mapping[str, Any],
        prepared: PreparedTrainingRun,
        report_digest: str,
    ) -> dict[str, Any]:
        value = self._control.accept_run(
            request,
            prepared.trainer_build,
            prepared.total_epochs,
            prepared.trainer_execution_seconds,
            report_digest,
        )
        if not isinstance(value, dict) or set(value) != {
            "accepted",
            "should_dispatch",
        }:
            raise self._invalid_control_result()
        should_dispatch = value["should_dispatch"]
        if not isinstance(should_dispatch, bool):
            raise self._invalid_control_result()
        try:
            run = validate_training_run_record(value["accepted"], self._api_limits)
        except ValueError as error:
            raise self._invalid_control_result() from error
        if run["request"] != request:
            raise self._invalid_control_result()
        return {"accepted": run, "should_dispatch": should_dispatch}

    def _dispatch_and_attach(self, run_id: str, prepared: PreparedTrainingRun) -> None:
        call = None
        try:
            call = self._spawn_run(
                run_id,
                prepared.worker_resources,
                prepared.trainer_execution_seconds,
            )
        except GatewayFailure:
            raise
        except Exception as error:
            _LOG.exception("training worker dispatch failed for run %s", run_id)
            raise GatewayFailure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "training run could not be dispatched",
                resource_id=run_id,
            ) from error
        try:
            call_id = validate_training_worker_call_id(call.object_id)
        except (AttributeError, ValueError) as error:
            _cancel_quietly(call)
            raise GatewayFailure(
                GatewayErrorCode.INTERNAL_ERROR,
                "training dispatch returned an invalid call identity",
                resource_id=run_id,
            ) from error
        try:
            self._control.attach_run_worker(run_id, call_id)
        except GatewayFailure:
            _cancel_quietly(call)
            raise
        except Exception as error:
            _cancel_quietly(call)
            raise GatewayFailure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "training worker attachment is temporarily unavailable",
                resource_id=run_id,
            ) from error

    def _cleanup_abandoned_admissions(self) -> None:
        for _ in range(self._service_limits["max_active_training_runs"]):
            admission = self._control.claim_abandoned_run_admission()
            if admission is None:
                return
            try:
                accepted = validate_training_run_admission_record(admission)
            except ValueError as error:
                raise self._invalid_control_result() from error
            run_id = accepted["run_id"]
            self._source_files.delete_source_roots_after_fence(run_id)
            self._run_files.delete_reports(run_id)

    def _cleanup_transient_roots(self) -> None:
        cleanup_limit = self._service_limits["max_active_training_runs"]
        scan_limit = self._service_limits["max_control_records"] + cleanup_limit
        source_ids = set(self._source_files.transient_run_ids(scan_limit))
        result_ids = set(self._result_files.transient_run_ids(scan_limit))
        report_transient_ids = set(self._run_files.transient_run_ids(scan_limit))
        report_ids = set(self._run_files.report_run_ids(scan_limit))
        report_history_ids = set(self._run_files.report_history_run_ids(scan_limit))
        run_ids = sorted(source_ids | result_ids | report_transient_ids | report_ids)[
            :scan_limit
        ]
        scopes = self._control.training_run_transient_cleanup_scopes(run_ids)
        if not isinstance(scopes, dict) or set(scopes) != set(run_ids):
            raise self._invalid_control_result()
        cleaned = 0
        for run_id in run_ids:
            scope = scopes[run_id]
            retained_report: tuple[int, str] | None = None
            if not isinstance(scope, dict):
                raise self._invalid_control_result()
            kind = scope.get("type")
            if kind == "blocked" and set(scope) == {"type"}:
                continue
            if kind == "all" and set(scope) == {"type"}:
                pass
            elif kind == "terminal" and set(scope) == {
                "type",
                "report_revision",
                "report_digest",
            }:
                try:
                    retained_report = (
                        require_positive_int32(
                            scope["report_revision"], "report_revision"
                        ),
                        require_sha256(scope["report_digest"], "report_digest"),
                    )
                except ValueError as error:
                    raise self._invalid_control_result() from error
            else:
                raise self._invalid_control_result()
            if (
                kind == "terminal"
                and run_id not in source_ids
                and run_id not in result_ids
                and run_id not in report_transient_ids
                and run_id not in report_history_ids
            ):
                continue
            self._source_files.delete_source_roots_after_fence(run_id)
            self._result_files.delete_candidate_after_fence(run_id)
            if kind == "all":
                self._run_files.delete_reports(run_id)
            else:
                self._run_files.delete_transients(run_id)
                assert retained_report is not None
                self._run_files.prune_report_history(
                    run_id,
                    *retained_report,
                )
            cleaned += 1
            if cleaned == cleanup_limit:
                return

    def _prepare_run_checked(
        self,
        request: Mapping[str, Any],
        recipe: Mapping[str, Any],
        starting_weight_digest: str,
    ) -> PreparedTrainingRun:
        try:
            return self._prepare_run(request, recipe, starting_weight_digest)
        except GatewayFailure:
            raise
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "training settings are incompatible with the selected trainer",
            ) from error

    def _discard_candidates(self, run_id: str, report_digest: str) -> None:
        for discard in (
            self._source_files.discard_candidate,
            self._run_files.discard_report_candidate,
        ):
            try:
                discard(run_id, report_digest)
            except Exception:
                # Acceptance no longer depends on these unreachable bytes. A later
                # bounded transient cleanup pass owns crash or cleanup leftovers.
                _LOG.warning(
                    "training candidate cleanup failed for run %s",
                    run_id,
                    exc_info=True,
                )

    def _validate_request(self, value: Any) -> dict[str, Any]:
        try:
            return validate_training_run_request(value, self._api_limits)
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                f"invalid training run request: {error}",
            ) from error

    @staticmethod
    def _invalid_control_result() -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "training state owner returned an invalid result",
        )


def _accepted_response(run: Mapping[str, Any]) -> dict[str, str]:
    return {"run_id": run["run_id"], "request_digest": run["request_digest"]}


def _cancel_quietly(call: Any) -> None:
    try:
        call.cancel()
    except Exception:
        pass
