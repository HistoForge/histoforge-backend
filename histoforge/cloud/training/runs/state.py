"""Serialized lifecycle decisions for durable managed-training runs."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_tree,
    require_canonical_uuid,
    require_nonnegative_int32,
    require_positive_int32,
    require_sha256,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud._time import iso_z
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.limits import (
    validate_training_api_limits,
    validate_training_service_limits,
)
from histoforge.cloud.training.runs.contract import (
    is_active_training_run,
    terminal_training_report_identity,
    validate_training_finalization,
    validate_training_run_control_record,
)
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)


class TrainingRunState:
    """Own admission identity, run transitions, and worker fencing."""

    def __init__(
        self,
        records: TrainingControlRecords,
        api_limits: Any,
        service_limits: Any,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        id_factory: Callable[[], uuid.UUID] = uuid.uuid4,
    ) -> None:
        self._records = records
        self._api_limits = validate_training_api_limits(api_limits)
        self._service_limits = validate_training_service_limits(service_limits)
        self._clock = clock
        self._id_factory = id_factory

    def reserve(self, value: Any) -> dict[str, Any]:
        """Return an accepted replay or reserve one preparation admission."""

        request = validate_training_run_request(value, self._api_limits)
        digest = training_run_request_digest(request, self._api_limits)
        current = self._lookup_digest(digest)
        if current is not None:
            return current
        if self._active_count() >= self._service_limits["max_active_training_runs"]:
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "the deployment has reached its active training-run limit",
            )

        run_id = require_canonical_uuid(str(self._id_factory()), "run_id")
        admission = canonical_tree(
            {
                "type": "preparing",
                "run_id": run_id,
                "request_digest": digest,
                "created_at": iso_z(self._now()),
            }
        )
        winner, created = self._records.reserve_admission(admission)
        if not created and winner != admission:
            raise self._invalid(run_id)
        return {"type": "preparing", "admission": winner}

    def lookup(self, value: Any) -> dict[str, Any] | None:
        """Read an accepted replay or existing admission without creating one."""

        request = validate_training_run_request(value, self._api_limits)
        digest = training_run_request_digest(request, self._api_limits)
        return self._lookup_digest(digest)

    def claim_abandoned_admission(self) -> dict[str, Any] | None:
        """Remove one expired admission before its artifact roots are deleted."""

        cutoff = self._now() - timedelta(
            seconds=self._service_limits["pending_snapshot_cleanup_age_seconds"]
        )
        runs = self._records.runs()
        by_run_id = {control["run"]["run_id"]: control for control in runs}
        by_digest = {control["run"]["request_digest"]: control for control in runs}
        for admission in sorted(
            self._records.admissions(),
            key=lambda item: (item["created_at"], item["request_digest"]),
        ):
            if _parse_z(admission["created_at"]) > cutoff:
                continue
            if (
                admission["run_id"] in by_run_id
                or admission["request_digest"] in by_digest
            ):
                raise self._invalid(admission["run_id"])
            self._records.remove_admission(admission["request_digest"])
            return admission
        return None

    def transient_cleanup_scopes(self, run_ids: Any) -> dict[str, dict[str, Any]]:
        """Classify a bounded batch of run-owned roots before physical deletion."""

        if not isinstance(run_ids, list):
            raise ValueError("run_ids must be an array")
        maximum = (
            self._service_limits["max_control_records"]
            + self._service_limits["max_active_training_runs"]
        )
        if len(run_ids) > maximum:
            raise ValueError("run_ids exceeds the transient-cleanup batch limit")
        canonical_ids = [require_canonical_uuid(value, "run_id") for value in run_ids]
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("run_ids must not contain duplicates")

        admissions: dict[str, dict[str, Any]] = {}
        for admission in self._records.admissions():
            run_id = admission["run_id"]
            if run_id in admissions:
                raise self._invalid(run_id)
            admissions[run_id] = admission
        runs: dict[str, dict[str, Any]] = {}
        for control in self._records.runs():
            run_id = control["run"]["run_id"]
            if run_id in runs or run_id in admissions:
                raise self._invalid(run_id)
            runs[run_id] = control

        result: dict[str, dict[str, Any]] = {}
        for run_id in canonical_ids:
            if run_id in admissions:
                result[run_id] = {"type": "blocked"}
                continue
            control = runs.get(run_id)
            if control is None:
                result[run_id] = {"type": "all"}
            elif is_active_training_run(control, self._api_limits):
                result[run_id] = {"type": "blocked"}
            else:
                result[run_id] = {
                    "type": "terminal",
                    **terminal_training_report_identity(
                        control["run"], self._api_limits
                    ),
                }
        return result

    def accept(
        self,
        value: Any,
        trainer_build: Any,
        total_epochs: Any,
        trainer_execution_seconds: Any,
    ) -> dict[str, Any]:
        """Publish one QUEUED run and identify the sole dispatch winner."""

        request = validate_training_run_request(value, self._api_limits)
        digest = training_run_request_digest(request, self._api_limits)
        existing = self._run_for_request_digest(digest)
        if existing is not None:
            return {"accepted": existing["run"], "should_dispatch": False}
        admission = self._records.get_admission(digest)
        if admission is None:
            raise self._conflict(
                "unknown-training-run", "training run was not reserved"
            )
        dataset = self._records.get_dataset(request["dataset_id"])
        if (
            dataset is None
            or dataset["type"] != "retained"
            or dataset["dataset_digest"] != request["dataset_digest"]
        ):
            raise self._conflict(
                admission["run_id"],
                "the selected training dataset is unavailable or has changed",
            )

        epochs = require_positive_int32(total_epochs, "total_epochs")
        if epochs > self._api_limits["report"]["max_epochs"]:
            raise ValueError("total_epochs exceeds the training report limit")
        execution_seconds = require_positive_int32(
            trainer_execution_seconds, "trainer_execution_seconds"
        )
        now = self._now()
        accepted_at = iso_z(now)
        attachment_deadline = iso_z(
            now
            + timedelta(
                seconds=self._service_limits["worker_call_attachment_timeout_seconds"]
            )
        )
        bounded_execution = min(
            execution_seconds,
            self._service_limits["max_training_execution_seconds"],
        )
        execution_deadline = iso_z(
            now
            + timedelta(
                seconds=bounded_execution
                + self._service_limits["training_queue_allowance_seconds"]
            )
        )
        control = validate_training_run_control_record(
            {
                "run": {
                    "run_id": admission["run_id"],
                    "request": request,
                    "request_digest": digest,
                    "trainer_build": parse_software_build_identity(trainer_build),
                    "state": "QUEUED",
                    "progress": {
                        "completed_epochs": 0,
                        "total_epochs": epochs,
                    },
                    "report_revision": 1,
                    "created_at": accepted_at,
                    "updated_at": accepted_at,
                },
                "worker": {
                    "call_attachment_deadline_at": attachment_deadline,
                    "execution_deadline_at": execution_deadline,
                    "worker_claimed": False,
                },
                "finalization_revision": 0,
            },
            self._api_limits,
        )
        accepted = self._records.accept_run(digest, control)["run"]
        return {"accepted": accepted, "should_dispatch": True}

    def attach_worker(self, run_id: str, worker_call_id: str) -> dict[str, Any]:
        """Attach exactly one Modal call and advance QUEUED to RUNNING."""

        control = self._require_control(run_id)
        run = control["run"]
        worker = control["worker"]
        if run["state"] == "RUNNING" and worker.get("worker_call_id") == worker_call_id:
            return run
        if run["state"] != "QUEUED":
            raise self._conflict(
                run["run_id"], "training run no longer accepts a worker"
            )
        if iso_z(self._now()) >= worker["call_attachment_deadline_at"]:
            raise self._conflict(run["run_id"], "worker attachment deadline has passed")

        updated = self._copy(control)
        updated["worker"]["worker_call_id"] = worker_call_id
        updated["run"]["state"] = "RUNNING"
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def claim_worker(self, run_id: str, executing_build: Any) -> dict[str, Any]:
        """Allow only the first matching deployed build to begin expensive work."""

        control = self._require_control(run_id)
        run = control["run"]
        worker = control["worker"]
        if run["state"] != "RUNNING" or "worker_call_id" not in worker:
            raise self._conflict(run["run_id"], "training worker cannot claim this run")
        if worker["worker_claimed"]:
            raise self._conflict(run["run_id"], "training worker was already claimed")
        if parse_software_build_identity(executing_build) != run["trainer_build"]:
            raise self._conflict(run["run_id"], "training worker build does not match")
        if iso_z(self._now()) >= worker["execution_deadline_at"]:
            raise self._conflict(
                run["run_id"], "training execution deadline has passed"
            )

        updated = self._copy(control)
        updated["worker"]["worker_claimed"] = True
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def advance_progress(
        self, run_id: str, completed_epochs: Any, report_revision: Any
    ) -> dict[str, Any]:
        """Advertise one already committed report revision and epoch projection."""

        control = self._require_control(run_id)
        run = control["run"]
        if run["state"] != "RUNNING" or not control["worker"]["worker_claimed"]:
            raise self._conflict(
                run["run_id"], "training progress is no longer accepted"
            )
        completed = require_positive_int32(completed_epochs, "completed_epochs")
        revision = self._next_revision(run, report_revision)
        if (
            not run["progress"]["completed_epochs"]
            < completed
            <= run["progress"]["total_epochs"]
        ):
            raise self._conflict(
                run["run_id"], "training progress must advance monotonically"
            )

        updated = self._copy(control)
        updated["run"]["progress"]["completed_epochs"] = completed
        updated["run"]["report_revision"] = revision
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def cancel(
        self, run_id: str, report_digest: Any, report_revision: Any
    ) -> dict[str, Any]:
        """Commit the winning QUEUED or RUNNING cancellation transition."""

        control = self._require_control(run_id)
        run = control["run"]
        digest = require_sha256(report_digest, "report_digest")
        if run["state"] == "CANCELLED":
            if run["result"]["report_digest"] == digest:
                return run
            raise self._conflict(run["run_id"], "training cancellation differs")
        if run["state"] not in {"QUEUED", "RUNNING"}:
            raise self._conflict(
                run["run_id"], "training run can no longer be cancelled"
            )
        revision = self._next_revision(run, report_revision)

        updated = self._copy(control)
        updated["run"]["state"] = "CANCELLED"
        updated["run"]["report_revision"] = revision
        updated["run"]["result"] = {
            "type": "cancelled",
            "report_digest": digest,
        }
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def fail(
        self,
        run_id: str,
        error: Any,
        report_digest: Any,
        report_revision: Any,
        finalization_revision: Any = None,
    ) -> dict[str, Any]:
        """Commit one terminal failure while preserving frozen finalization evidence."""

        control = self._require_control(run_id)
        run = control["run"]
        if "finalization" in run:
            require_finalization_revision(control, finalization_revision)
        elif finalization_revision is not None:
            raise ValueError("failure before finalization cannot carry its revision")
        digest = require_sha256(report_digest, "report_digest")
        if run["state"] == "FAILED":
            if run["result"]["report_digest"] == digest:
                return run
            raise self._conflict(run["run_id"], "training failure differs")
        if run["state"] not in {"QUEUED", "RUNNING", "FINALIZING"}:
            raise self._conflict(
                run["run_id"], "training run cannot fail from its state"
            )
        revision = self._next_revision(run, report_revision)

        updated = self._copy(control)
        updated["run"]["state"] = "FAILED"
        updated["run"]["report_revision"] = revision
        updated["run"]["result"] = {
            "type": "failed",
            "error": error,
            "report_digest": digest,
        }
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def begin_finalization(
        self, run_id: str, finalization: Any, report_revision: Any
    ) -> dict[str, Any]:
        """Freeze one complete module candidate and close cancellation."""

        control = self._require_control(run_id)
        run = control["run"]
        frozen = validate_training_finalization(finalization)
        if run["state"] == "FINALIZING":
            if run["finalization"] == frozen:
                return run
            raise self._conflict(run["run_id"], "training finalization differs")
        if run["state"] != "RUNNING" or not control["worker"]["worker_claimed"]:
            raise self._conflict(
                run["run_id"], "training run cannot begin finalization"
            )
        if iso_z(self._now()) >= control["worker"]["execution_deadline_at"]:
            raise self._conflict(
                run["run_id"], "training execution deadline has passed"
            )
        if run["progress"]["completed_epochs"] != run["progress"]["total_epochs"]:
            raise self._conflict(run["run_id"], "training epochs are incomplete")
        revision = self._next_revision(run, report_revision)

        updated = self._copy(control)
        updated["run"]["state"] = "FINALIZING"
        updated["run"]["report_revision"] = revision
        updated["run"]["finalization"] = frozen
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def claim_finalization(self, run_id: str, expected_revision: Any) -> dict[str, Any]:
        """Fence older outcomes before one serialized module save attempt."""

        control = self._require_control(run_id)
        run = control["run"]
        expected = require_nonnegative_int32(expected_revision, "finalization_revision")
        if run["state"] != "FINALIZING":
            raise self._conflict(run_id, "training run is not finalizing")
        if control["finalization_revision"] != expected:
            raise self._conflict(
                run_id, "training finalization revision is not current"
            )
        successor = require_positive_int32(expected + 1, "finalization_revision")
        updated = self._copy(control)
        updated["finalization_revision"] = successor
        return self._replace(updated)

    def succeed(
        self, run_id: str, report_revision: Any, finalization_revision: Any
    ) -> dict[str, Any]:
        """Advertise the preconstructed success report after module verification."""

        control = self._require_control(run_id)
        require_finalization_revision(control, finalization_revision)
        run = control["run"]
        if run["state"] == "SUCCEEDED":
            return run
        if run["state"] != "FINALIZING":
            raise self._conflict(run["run_id"], "training run is not finalizing")
        revision = self._next_revision(run, report_revision)

        updated = self._copy(control)
        updated["run"]["state"] = "SUCCEEDED"
        updated["run"]["report_revision"] = revision
        updated["run"]["result"] = {"type": "succeeded"}
        updated["run"]["updated_at"] = iso_z(self._now())
        return self._replace(updated)["run"]

    def get(self, run_id: str) -> dict[str, Any]:
        """Return one complete public run record."""

        return self._require_control(run_id)["run"]

    def runs(self) -> list[dict[str, Any]]:
        """Return every validated public run for bounded list projection."""

        return [self._copy(control["run"]) for control in self._records.runs()]

    def get_control(self, run_id: str) -> dict[str, Any]:
        """Return one complete private control record to an internal owner."""

        return self._require_control(run_id)

    def training_result_visible(self, run_id: str) -> bool:
        """Allow an attested result after success or explicit Jobs cleanup."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        control = self._records.get_run(canonical_id)
        return control is None or control["run"]["state"] == "SUCCEEDED"

    def dataset_in_use(self, dataset_id: str) -> bool:
        """Return whether one nonterminal accepted run still references a dataset."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        return any(
            control["run"]["request"]["dataset_id"] == canonical_id
            and is_active_training_run(control, self._api_limits)
            for control in self._records.runs()
        )

    def begin_delete(self, run_id: str) -> dict[str, Any] | None:
        """Return one terminal run whose report root may now be deleted."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        control = self._records.get_run(canonical_id)
        if control is None:
            return None
        if is_active_training_run(control, self._api_limits):
            raise self._conflict(canonical_id, "active training run cannot be cleared")
        return control["run"]

    def finish_delete(self, run_id: str) -> None:
        """Remove a terminal Jobs record after its report bytes are absent."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        control = self._records.get_run(canonical_id)
        if control is None:
            return
        if is_active_training_run(control, self._api_limits):
            raise self._invalid(canonical_id)
        self._records.remove_run(canonical_id)

    def _run_for_request_digest(self, digest: str) -> dict[str, Any] | None:
        matches = [
            control
            for control in self._records.runs()
            if control["run"]["request_digest"] == digest
        ]
        if len(matches) > 1:
            raise self._invalid("training-runs")
        return matches[0] if matches else None

    def _lookup_digest(self, digest: str) -> dict[str, Any] | None:
        accepted = self._run_for_request_digest(digest)
        if accepted is not None:
            return {"type": "accepted", "run": accepted["run"]}
        admission = self._records.get_admission(digest)
        if admission is not None:
            return {"type": "preparing", "admission": admission}
        return None

    def _active_count(self) -> int:
        return len(self._records.admissions()) + sum(
            is_active_training_run(control, self._api_limits)
            for control in self._records.runs()
        )

    def _require_control(self, run_id: str) -> dict[str, Any]:
        canonical_id = require_canonical_uuid(run_id, "run_id")
        control = self._records.get_run(canonical_id)
        if control is None:
            raise GatewayFailure(
                GatewayErrorCode.TRAINING_RUN_NOT_FOUND,
                "the training run does not exist",
                resource_id=canonical_id,
            )
        return control

    def _replace(self, value: Any) -> dict[str, Any]:
        return self._records.replace_run(
            validate_training_run_control_record(value, self._api_limits)
        )

    @staticmethod
    def _next_revision(run: Mapping[str, Any], value: Any) -> int:
        revision = require_positive_int32(value, "report_revision")
        if revision != run["report_revision"] + 1:
            raise GatewayFailure(
                GatewayErrorCode.TRAINING_STATE_CONFLICT,
                "training report revision is not the next revision",
                resource_id=run["run_id"],
            )
        return revision

    @staticmethod
    def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
        return canonical_tree(value)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("training run clock must be timezone-aware")
        return value.astimezone(timezone.utc).replace(
            microsecond=value.microsecond // 1000 * 1000
        )

    @staticmethod
    def _conflict(run_id: str, message: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.TRAINING_STATE_CONFLICT,
            message,
            resource_id=run_id,
        )

    @staticmethod
    def _invalid(run_id: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "training run state is inconsistent",
            resource_id=run_id,
        )


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")


def require_finalization_revision(control: Mapping[str, Any], value: Any) -> int:
    """Require the acknowledged save fence before report or state publication."""

    revision = require_positive_int32(value, "finalization_revision")
    if revision != control["finalization_revision"]:
        raise GatewayFailure(
            GatewayErrorCode.TRAINING_STATE_CONFLICT,
            "training finalization revision is not current",
            resource_id=control["run"]["run_id"],
        )
    return revision
