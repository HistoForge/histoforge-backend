"""Join training-run state transitions to immutable report candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_positive_int32,
    require_sha256,
)
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.registry import select_training_run_records
from histoforge.cloud.training.runs.report import (
    training_report_digest,
    validate_training_report,
)
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.state import (
    TrainingRunState,
    require_finalization_revision,
)

_IMMUTABLE_REPORT_FIELDS = {
    "schema_version",
    "run_id",
    "dataset_digest",
    "dataset_image_count",
    "split_plan",
    "source_module",
    "selected_model_id",
    "starting_weight_digest",
    "result_form",
    "result_display_name",
    "result_inference_gpu",
    "task",
    "trainer_build",
    "training_seed",
    "trainer_settings",
    "total_epochs",
    "field_profile",
    "min_instance_pixels",
    "augmentation",
    "trainer_configuration",
    "training_counts",
    "validation_counts",
    "metric_policy",
}


class TrainingRunControl:
    """Validate and promote report candidates before advertising state cuts."""

    def __init__(
        self,
        state: TrainingRunState,
        files: TrainingRunFiles,
        source_files: TrainingSourceFiles,
        api_limits: Any,
    ) -> None:
        self._state = state
        self._files = files
        self._source_files = source_files
        self._api_limits = validate_training_api_limits(api_limits)

    def reserve(self, request: Any) -> dict[str, Any]:
        return self._state.reserve(request)

    def lookup(self, request: Any) -> dict[str, Any] | None:
        """Read an accepted replay or live admission without creating state."""

        return self._state.lookup(request)

    def claim_abandoned_admission(self) -> dict[str, Any] | None:
        return self._state.claim_abandoned_admission()

    def transient_cleanup_scopes(self, run_ids: Any) -> dict[str, dict[str, Any]]:
        return self._state.transient_cleanup_scopes(run_ids)

    def accept(
        self,
        request: Any,
        trainer_build: Any,
        total_epochs: Any,
        trainer_execution_seconds: Any,
        report_digest: Any,
    ) -> dict[str, Any]:
        """Verify report revision 1, then expose one accepted QUEUED run."""

        normalized_request = validate_training_run_request(request, self._api_limits)
        digest = require_sha256(report_digest, "report_digest")
        reservation = self._state.lookup(normalized_request)
        if reservation is None:
            raise self._conflict(
                "unknown-training-run", "training run was not reserved"
            )
        if reservation["type"] == "accepted":
            return {"accepted": reservation["run"], "should_dispatch": False}
        run_id = reservation["admission"]["run_id"]
        report = self._files.read_report_for_promotion(run_id, 1, digest)["report"]
        source = self._source_files.read_for_promotion(run_id, digest)
        self._require_source_ready(
            source,
            run_id,
            normalized_request,
            digest,
        )
        self._require_initial_report(
            report,
            run_id,
            normalized_request,
            trainer_build,
            total_epochs,
            source["starting_weight_digest"],
        )
        self._source_files.promote_candidate(run_id, digest)
        self._files.promote_report(run_id, 1, digest)
        return self._state.accept(
            normalized_request,
            trainer_build,
            total_epochs,
            trainer_execution_seconds,
        )

    def attach_worker(self, run_id: str, worker_call_id: str) -> dict[str, Any]:
        return self._state.attach_worker(run_id, worker_call_id)

    def claim_worker(self, run_id: str, executing_build: Any) -> dict[str, Any]:
        return self._state.claim_worker(run_id, executing_build)

    def advance_progress(
        self, run_id: str, report_revision: Any, report_digest: Any
    ) -> dict[str, Any]:
        """Verify one RUNNING report and advertise its completed epoch."""

        control, report, revision = self._prepare_report(
            run_id, report_revision, report_digest, "RUNNING"
        )
        run = control["run"]
        if revision is None:
            return run
        if run["state"] != "RUNNING" or not control["worker"]["worker_claimed"]:
            raise self._conflict(run_id, "training progress is no longer accepted")
        completed = _last_epoch(report)
        if not (
            run["progress"]["completed_epochs"]
            < completed
            <= run["progress"]["total_epochs"]
        ):
            raise self._conflict(run_id, "training progress must advance monotonically")
        self._promote(run_id, revision, report_digest)
        return self._state.advance_progress(run_id, completed, revision)

    def cancel(
        self, run_id: str, report_revision: Any, report_digest: Any
    ) -> dict[str, Any]:
        """Verify one CANCELLED report before committing cancellation."""

        control, report, revision = self._prepare_report(
            run_id, report_revision, report_digest, "CANCELLED"
        )
        run = control["run"]
        if revision is None:
            return run
        if run["state"] not in {"QUEUED", "RUNNING"}:
            raise self._conflict(run_id, "training run can no longer be cancelled")
        self._require_report_progress(report, run)
        self._promote(run_id, revision, report_digest)
        return self._state.cancel(run_id, require_sha256(report_digest), revision)

    def fail(
        self,
        run_id: str,
        expected_state: str,
        report_revision: Any,
        report_digest: Any,
        finalization_revision: Any = None,
    ) -> dict[str, Any]:
        """Fail only the exact lifecycle state that produced the failure decision."""

        control, report, revision = self._prepare_report(
            run_id, report_revision, report_digest, "FAILED"
        )
        run = control["run"]
        if expected_state == "FINALIZING":
            require_finalization_revision(control, finalization_revision)
        elif finalization_revision is not None:
            raise ValueError("failure before finalization cannot carry its revision")
        if revision is None:
            return run
        if run["state"] != expected_state:
            raise self._conflict(run_id, "training run advanced before failure")
        if run["state"] not in {"QUEUED", "RUNNING", "FINALIZING"}:
            raise self._conflict(run_id, "training run cannot fail from its state")
        self._require_report_progress(report, run)
        self._promote(run_id, revision, report_digest)
        return self._state.fail(
            run_id,
            report["failure"],
            require_sha256(report_digest),
            revision,
            finalization_revision,
        )

    def begin_finalization(
        self,
        run_id: str,
        report_revision: Any,
        report_digest: Any,
        succeeded_report: Any,
        module_content_digest: Any,
    ) -> dict[str, Any]:
        """Freeze candidate identities after verifying the FINALIZING report."""

        control, live_report, revision = self._prepare_report(
            run_id, report_revision, report_digest, "FINALIZING"
        )
        run = control["run"]
        success_report = validate_training_report(succeeded_report, self._api_limits)
        self._require_report_matches_run(success_report, run, "SUCCEEDED")
        self._require_report_history(success_report, live_report)
        if success_report["model_selection"] != live_report["model_selection"]:
            raise self._conflict(run_id, "success report changes model selection")
        finalization = {
            "selected_weight_digest": live_report["model_selection"][
                "selected_weight_digest"
            ],
            "report_digest": training_report_digest(success_report, self._api_limits),
            "module_content_digest": require_sha256(
                module_content_digest, "module_content_digest"
            ),
        }
        if revision is None:
            if run.get("finalization") == finalization:
                return run
            raise self._conflict(run_id, "training finalization differs")
        if (
            run["state"] != "RUNNING"
            or not control["worker"]["worker_claimed"]
            or run["progress"]["completed_epochs"] != run["progress"]["total_epochs"]
        ):
            raise self._conflict(run_id, "training run cannot begin finalization")
        self._promote(run_id, revision, report_digest)
        return self._state.begin_finalization(run_id, finalization, revision)

    def claim_finalization(self, run_id: str, expected_revision: Any) -> dict[str, Any]:
        """Advance only the private save fence, without changing report bytes."""

        return self._state.claim_finalization(run_id, expected_revision)

    def succeed(
        self,
        run_id: str,
        report_revision: Any,
        report_digest: Any,
        finalization_revision: Any,
    ) -> dict[str, Any]:
        """Advertise the frozen success report after module reconciliation."""

        control, report, revision = self._prepare_report(
            run_id, report_revision, report_digest, "SUCCEEDED"
        )
        require_finalization_revision(control, finalization_revision)
        run = control["run"]
        if run["state"] not in {"FINALIZING", "SUCCEEDED"}:
            raise self._conflict(run_id, "training run is not finalizing")
        digest = training_report_digest(report, self._api_limits)
        if digest != run["finalization"]["report_digest"]:
            raise self._conflict(run_id, "success report differs from finalization")
        if (
            report["model_selection"]["selected_weight_digest"]
            != run["finalization"]["selected_weight_digest"]
        ):
            raise self._conflict(run_id, "success report changes selected weights")
        if revision is None:
            return run
        self._promote(run_id, revision, report_digest)
        return self._state.succeed(run_id, revision, finalization_revision)

    def get(self, run_id: str) -> dict[str, Any]:
        return self._state.get(run_id)

    def get_control(self, run_id: str) -> dict[str, Any]:
        """Return private worker control to authenticated backend orchestration."""

        return self._state.get_control(run_id)

    def page_records(
        self, limit: Any, cursor: str | None = None
    ) -> list[dict[str, Any]]:
        """Select one ordered page plus lookahead without report-file IO."""

        return select_training_run_records(
            self._state.runs(),
            limit,
            cursor,
            self._api_limits,
        )

    def get_report(self, run_id: str) -> dict[str, Any]:
        run = self._state.get(run_id)
        value = self._files.read_report(run_id, run["report_revision"])
        self._require_current_report(value["report"], run)
        return canonical_tree(
            {
                "report_revision": run["report_revision"],
                "report_digest": value["report_digest"],
                "report": value["report"],
            }
        )

    def training_result_visible(self, run_id: str) -> bool:
        """Return the state owner's visibility decision for one result module."""

        return self._state.training_result_visible(run_id)

    def begin_delete(self, run_id: str) -> dict[str, Any] | None:
        return self._state.begin_delete(run_id)

    def finish_delete(self, run_id: str) -> None:
        self._state.finish_delete(run_id)

    def _prepare_report(
        self,
        run_id: str,
        report_revision: Any,
        report_digest: Any,
        expected_state: str,
    ) -> tuple[dict[str, Any], dict[str, Any], int | None]:
        control = self._state.get_control(run_id)
        run = control["run"]
        revision = require_positive_int32(report_revision, "report_revision")
        if revision not in {run["report_revision"], run["report_revision"] + 1}:
            raise self._conflict(run_id, "training report revision is not current")
        digest = require_sha256(report_digest, "report_digest")
        if revision == run["report_revision"]:
            report = self._read_report(run_id, revision, digest)
        else:
            report = self._files.read_report_for_promotion(run_id, revision, digest)[
                "report"
            ]
        self._require_report_matches_run(report, run, expected_state)
        self._require_report_selection(
            report,
            run,
            allow_new_finalization=(
                expected_state == "FINALIZING" and run["state"] == "RUNNING"
            ),
        )
        if revision == run["report_revision"]:
            self._require_current_report(report, run)
            return control, report, None
        current = self._files.read_report(run_id, run["report_revision"])["report"]
        self._require_current_report(current, run)
        self._require_report_history(report, current)
        return control, report, revision

    def _read_report(
        self, run_id: str, report_revision: Any, report_digest: Any
    ) -> dict[str, Any]:
        expected_digest = require_sha256(report_digest, "report_digest")
        published = self._files.read_report(run_id, report_revision)
        if published["report_digest"] != expected_digest:
            raise self._conflict(run_id, "training report digest differs")
        return published["report"]

    def _promote(self, run_id: str, revision: int, report_digest: Any) -> None:
        self._files.promote_report(run_id, revision, report_digest)

    def _require_initial_report(
        self,
        report: Mapping[str, Any],
        run_id: str,
        request: Mapping[str, Any],
        trainer_build: Any,
        total_epochs: Any,
        starting_weight_digest: str,
    ) -> None:
        expected = {
            "run_id": run_id,
            "state": "QUEUED",
            "dataset_digest": request["dataset_digest"],
            "split_plan": request["split_plan"],
            "source_module": request["source_module"],
            "selected_model_id": request["selected_model_id"],
            "result_form": request["result_form"],
            "result_display_name": request["result_display_name"],
            "result_inference_gpu": request["result_inference_gpu"],
            "trainer_build": trainer_build,
            "training_seed": request["training_seed"],
            "trainer_settings": request["trainer_settings"],
            "total_epochs": total_epochs,
            "field_profile": request["field_profile"],
            "min_instance_pixels": request.get("min_instance_pixels"),
            "augmentation": request["augmentation"],
            "starting_weight_digest": starting_weight_digest,
        }
        for field, expected_value in expected.items():
            if report.get(field) != canonical_tree(expected_value):
                raise self._conflict(run_id, f"initial report {field} differs")

    def _require_source_ready(
        self,
        marker: Mapping[str, Any],
        run_id: str,
        request: Mapping[str, Any],
        report_digest: str,
    ) -> None:
        expected = {
            "run_id": run_id,
            "request_digest": training_run_request_digest(request, self._api_limits),
            "source_module": request["source_module"],
            "report_digest": report_digest,
        }
        for field, expected_value in expected.items():
            if marker[field] != canonical_tree(expected_value):
                raise self._conflict(run_id, f"source snapshot {field} differs")

    def _require_report_matches_run(
        self,
        report: Mapping[str, Any],
        run: Mapping[str, Any],
        expected_state: str,
    ) -> None:
        request = run["request"]
        expected = {
            "run_id": run["run_id"],
            "state": expected_state,
            "dataset_digest": request["dataset_digest"],
            "split_plan": request["split_plan"],
            "source_module": request["source_module"],
            "selected_model_id": request["selected_model_id"],
            "result_form": request["result_form"],
            "result_display_name": request["result_display_name"],
            "result_inference_gpu": request["result_inference_gpu"],
            "trainer_build": run["trainer_build"],
            "training_seed": request["training_seed"],
            "trainer_settings": request["trainer_settings"],
            "total_epochs": run["progress"]["total_epochs"],
            "field_profile": request["field_profile"],
            "min_instance_pixels": request.get("min_instance_pixels"),
            "augmentation": request["augmentation"],
        }
        for field, expected_value in expected.items():
            if report.get(field) != expected_value:
                raise self._conflict(run["run_id"], f"report {field} differs")

    def _require_report_history(
        self, report: Mapping[str, Any], previous: Mapping[str, Any]
    ) -> None:
        for field in _IMMUTABLE_REPORT_FIELDS:
            if report.get(field) != previous.get(field):
                raise self._conflict(report["run_id"], f"report {field} changed")
        if report["epochs"][: len(previous["epochs"])] != previous["epochs"]:
            raise self._conflict(report["run_id"], "report changed completed epochs")
        if report["messages"][: len(previous["messages"])] != previous["messages"]:
            raise self._conflict(report["run_id"], "report changed earlier messages")
        if "model_selection" in previous and (
            report.get("model_selection") != previous["model_selection"]
        ):
            raise self._conflict(report["run_id"], "report changed model selection")

    def _require_report_progress(
        self, report: Mapping[str, Any], run: Mapping[str, Any]
    ) -> None:
        if _last_epoch(report) != run["progress"]["completed_epochs"]:
            raise self._conflict(run["run_id"], "report progress differs from run")

    def _require_current_report_digest(
        self, report: Mapping[str, Any], run: Mapping[str, Any]
    ) -> None:
        state = run["state"]
        expected = None
        if state in {"FAILED", "CANCELLED"}:
            expected = run["result"]["report_digest"]
        elif state == "SUCCEEDED":
            expected = run["finalization"]["report_digest"]
        if (
            expected is not None
            and training_report_digest(report, self._api_limits) != expected
        ):
            raise self._conflict(run["run_id"], "advertised report digest differs")

    def _require_current_report(
        self, report: Mapping[str, Any], run: Mapping[str, Any]
    ) -> None:
        self._require_report_matches_run(report, run, report["state"])
        allowed_report_states = {
            "QUEUED": {"QUEUED"},
            "RUNNING": {"QUEUED", "RUNNING"},
            "FINALIZING": {"FINALIZING"},
            "SUCCEEDED": {"SUCCEEDED"},
            "FAILED": {"FAILED"},
            "CANCELLED": {"CANCELLED"},
        }
        if report["state"] not in allowed_report_states[run["state"]]:
            raise self._conflict(run["run_id"], "advertised report state differs")
        if _last_epoch(report) != run["progress"]["completed_epochs"]:
            raise self._conflict(run["run_id"], "advertised report progress differs")
        self._require_report_selection(report, run, allow_new_finalization=False)
        self._require_current_report_digest(report, run)

    def _require_report_selection(
        self,
        report: Mapping[str, Any],
        run: Mapping[str, Any],
        *,
        allow_new_finalization: bool,
    ) -> None:
        selection = report.get("model_selection")
        finalization = run.get("finalization")
        if allow_new_finalization:
            return
        if finalization is None:
            if selection is not None:
                raise self._conflict(
                    run["run_id"], "report selects weights before finalization"
                )
            return
        if (
            selection is None
            or selection["selected_weight_digest"]
            != finalization["selected_weight_digest"]
        ):
            raise self._conflict(
                run["run_id"], "report selection differs from finalization"
            )

    @staticmethod
    def _conflict(run_id: str, message: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.TRAINING_STATE_CONFLICT,
            message,
            resource_id=run_id,
        )


def _last_epoch(report: Mapping[str, Any]) -> int:
    return report["epochs"][-1]["epoch"] if report["epochs"] else 0
