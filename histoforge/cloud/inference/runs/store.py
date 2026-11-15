"""Durable model-run lifecycle and first-wins publication policy.

All mutating methods are intended to run inside the deployment's single
``ArtifactLifecycleLane``.  Artifact bytes are committed before a returned
reference enters the persisted control record.  This owner has no worker
lease, heartbeat, automatic redispatch, or partial biological result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from histoforge.canonical_json import canonical_bytes, canonical_tree
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    validate_gateway_error,
)
from histoforge.cloud._time import iso_z
from histoforge.cloud.artifacts.refs import validate_artifact_ref
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.execution_limits import validate_model_run_limits
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    build_model_result_manifest,
    model_result_manifest_completed_unit_count,
    validate_model_input_result,
    validate_model_run_checkpoint,
    validate_model_run_unit_failure,
    validate_model_run_unit_result,
)
from histoforge.cloud.inference.runs.records import (
    MODEL_RUN_RECORD_SCHEMA_VERSION,
    ModelRunRecords,
    model_run_record_artifact_ids,
    require_model_run_publication_budgets,
)
from histoforge.cloud.inference.runs.request import (
    model_run_request_digest,
    validate_model_run_acceptance,
    validate_model_run_resource_limits,
)
from histoforge.cloud.inference.runs.reuse import select_reusable_model_run_state
from histoforge.cloud.inference.runs.source_bindings import validate_run_source_binding
from histoforge.cloud.inference.runs.status import validate_model_run_status
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION


class ModelRunStore:
    """One bounded durable-run state machine over immutable artifacts."""

    def __init__(
        self,
        values: Any,
        *,
        max_active_runs: int,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if isinstance(max_active_runs, bool) or not isinstance(max_active_runs, int):
            raise ValueError("max_active_runs must be a positive integer")
        if max_active_runs <= 0:
            raise ValueError("max_active_runs must be a positive integer")
        self._records = ModelRunRecords(values)
        self._max_active_runs = max_active_runs
        self._clock = clock

    def reserve(
        self,
        request: Any,
        model_run_limits: Any,
        terminal_retention_seconds: int,
        files: ArtifactFileStore,
        *,
        require_sources_live_through: Callable[[Mapping[str, Any], datetime], None],
        require_bindings_live_through: Callable[
            [Mapping[str, Any], list, datetime], None
        ]
        | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Persist one absent run and return ``(acceptance, should_dispatch)``.

        The boolean is true only for the caller that won reservation.  An
        idempotent replay receives the same acceptance with ``False`` and must
        not start a second worker.
        """

        limits = validate_model_run_limits(model_run_limits)
        submitted = validate_model_run_resource_limits(request, limits)
        retention = _positive_int(
            terminal_retention_seconds, "terminal_retention_seconds"
        )
        model_run_id = submitted["model_run_id"]
        request_digest = model_run_request_digest(submitted)
        now = self._now()

        existing = self._records.get(model_run_id)
        if existing is not None:
            self._require_retained(existing, now)
            if existing["request_digest"] != request_digest:
                raise self._conflict(model_run_id, "model-run request differs")
            return _accepted(submitted), False

        # Reclamation and reuse adoption share the same lifecycle lane. Keep
        # existing-ID outcomes above cleanup; perform all fallible collection
        # before publishing a fresh request so a failure cannot lose dispatch.
        self.cleanup_expired(files)
        now = self._now()

        active = 0
        for record in self._records.all():
            if record["phase"] in {"QUEUED", "RUNNING"} and now < _parse_z(
                record["execution_deadline"]
            ):
                active += 1
        if active >= self._max_active_runs:
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "the deployment has reached its active model-run limit",
            )

        execution_deadline = now + timedelta(seconds=limits["max_execution_seconds"])
        # Prediction needs source pixels through execution, not through result
        # retention. Completed result files can be read after those pixels expire.
        require_sources_live_through(submitted, execution_deadline)
        reusable = self._replacement_state(submitted, limits, files, now)
        if reusable["source_bindings"]:
            if require_bindings_live_through is None:
                raise ValueError(
                    "replacement bindings require source-lifetime validation"
                )
            require_bindings_live_through(
                submitted,
                reusable["source_bindings"],
                execution_deadline,
            )
        request_artifact = files.publish_artifact_bytes(
            canonical_bytes(submitted), media_type="application/json"
        )
        record = {
            "schema_version": MODEL_RUN_RECORD_SCHEMA_VERSION,
            "request": submitted,
            "request_digest": request_digest,
            "request_artifact": request_artifact,
            "limits": limits,
            "terminal_retention_seconds": retention,
            "accepted_at": iso_z(now),
            "execution_deadline": iso_z(execution_deadline),
            "phase": "QUEUED",
            "phase_deadline": iso_z(execution_deadline),
            **reusable,
            "terminal_manifest": None,
            "manifest": None,
            "failure": None,
        }
        self._require_publication_budgets(record)
        winner, created = self._records.reserve(record)
        if not created:
            files.delete_artifact(request_artifact["artifact_id"])
            self._require_retained(winner, now)
            if winner["request_digest"] != request_digest:
                raise self._conflict(model_run_id, "model-run request differs")
        return _accepted(submitted), created

    def begin_execution(
        self, model_run_id: str, files: ArtifactFileStore
    ) -> dict[str, Any] | None:
        """Begin execution and return its authoritative immutable context.

        ``None`` means a deadline or another terminal publication already won;
        the caller must perform no model work.
        """

        record = self._active_record(model_run_id, files)
        if record is None:
            return None
        if record["phase"] == "QUEUED":
            updated = dict(record)
            updated["phase"] = "RUNNING"
            record = self._records.replace(updated)
        plans = build_expected_model_execution_plans(
            record["request"], record["limits"]
        )
        return canonical_tree(
            {
                "request": record["request"],
                "limits": record["limits"],
                "execution_deadline": record["execution_deadline"],
                "source_bindings": record["source_bindings"],
                "execution_plans": [
                    {"input_id": item["input_id"], "plan": plans[item["input_id"]]}
                    for item in record["request"]["inputs"]
                ],
                "completed_units": record["completed_units"],
                "checkpoints": record["checkpoints"],
                "failed_units": record["failed_units"],
                "completed_input_ids": [
                    item["input_id"] for item in record["completed_inputs"]
                ],
            }
        )

    def bind_source(
        self,
        model_run_id: str,
        value: Any,
        files: ArtifactFileStore,
        *,
        require_source: Callable[
            [Mapping[str, Any], Mapping[str, Any], datetime], None
        ],
    ) -> dict[str, Any]:
        """Accept one exact uploaded file, or confirm an identical retained assignment.

        Confirmation precedes live-source and terminal checks because a lost HTTP
        reply must not make an accepted upload look unaccepted after the run ends.
        The lifecycle lane serializes this method with source release and cleanup.
        """

        record = self._retained_record(model_run_id, files)
        request = record["request"]
        inputs = {item["input_id"]: item for item in request["inputs"]}
        plans = build_expected_model_execution_plans(request, record["limits"])
        item = validate_run_source_binding(value, inputs, plans)
        existing = _by_unit(record["source_bindings"], _unit_key(item))
        if existing is not None:
            if existing != item:
                raise self._conflict(
                    model_run_id, "processing unit source binding differs"
                )
            return existing
        if record["phase"] not in {"QUEUED", "RUNNING"}:
            raise self._conflict(
                model_run_id, "terminated model run cannot accept another source"
            )
        require_source(
            request,
            item,
            _parse_z(record["execution_deadline"]),
        )
        updated = dict(record)
        updated["source_bindings"] = _sorted_units([*record["source_bindings"], item])
        self._records.replace(updated)
        return item

    def source_delivery(
        self, model_run_id: str, files: ArtifactFileStore
    ) -> dict[str, Any]:
        """Read accepted plans and bindings, including those retained after failure."""

        record = self._retained_record(model_run_id, files)
        plans = build_expected_model_execution_plans(
            record["request"], record["limits"]
        )
        return canonical_tree(
            {
                "model_run_id": model_run_id,
                "request_digest": record["request_digest"],
                "active": record["phase"] in {"QUEUED", "RUNNING"},
                "execution_plans": [
                    {"input_id": input_id, "plan": plan}
                    for input_id, plan in plans.items()
                ],
                "source_bindings": record["source_bindings"],
            }
        )

    def publish_unit(
        self,
        model_run_id: str,
        publication: Any,
        files: ArtifactFileStore,
    ) -> bool:
        """Expose one complete core/halo unit after its bytes are durable."""

        record = self._active_record(model_run_id, files)
        if record is None:
            return False
        item = validate_model_run_unit_result(
            publication, record["request"], record["limits"], record["source_bindings"]
        )
        key = _unit_key(item)
        existing = _by_unit(record["completed_units"], key)
        if existing is not None:
            if existing != item:
                raise self._conflict(model_run_id, "completed unit publication differs")
            return True
        if _by_unit(record["failed_units"], key) is not None:
            raise self._conflict(model_run_id, "unit already has a terminal failure")
        updated = _running_update(record)
        updated["completed_units"] = _sorted_units([*record["completed_units"], item])
        self._require_publication_budgets(updated)
        self._records.replace(updated)
        return True

    def publish_checkpoint(
        self,
        model_run_id: str,
        publication: Any,
        files: ArtifactFileStore,
    ) -> bool:
        """Retain only the highest accepted self-contained global checkpoint."""

        record = self._active_record(model_run_id, files)
        if record is None:
            return False
        item = validate_model_run_checkpoint(
            publication, record["request"], record["limits"]
        )
        key = _unit_key(item)
        existing = _by_unit(record["checkpoints"], key)
        if existing is not None:
            if item["sequence"] < existing["sequence"]:
                return False
            if item["sequence"] == existing["sequence"]:
                if item != existing:
                    raise self._conflict(model_run_id, "checkpoint publication differs")
                return True
        retained = [value for value in record["checkpoints"] if _unit_key(value) != key]
        updated = _running_update(record)
        updated["checkpoints"] = _sorted_units([*retained, item])
        if existing is not None:
            updated["released_artifact_ids"] = _released_artifact_ids(
                record,
                [existing["state"]["artifact_id"]],
            )
        self._require_publication_budgets(updated)
        self._records.replace(updated)
        return True

    def publish_unit_failure(
        self,
        model_run_id: str,
        publication: Any,
        files: ArtifactFileStore,
    ) -> bool:
        """First-win one coded unit failure while the run remains active."""

        record = self._active_record(model_run_id, files)
        if record is None:
            return False
        item = validate_model_run_unit_failure(
            publication, record["request"], record["limits"]
        )
        key = _unit_key(item)
        if _by_unit(record["completed_units"], key) is not None:
            raise self._conflict(model_run_id, "unit already completed")
        existing = _by_unit(record["failed_units"], key)
        if existing is not None:
            if existing != item:
                raise self._conflict(model_run_id, "unit failure publication differs")
            return True
        updated = _running_update(record)
        updated["failed_units"] = _sorted_units([*record["failed_units"], item])
        self._records.replace(updated)
        return True

    def publish_input(
        self,
        model_run_id: str,
        publication: Any,
        decoded_bytes: int,
        files: ArtifactFileStore,
    ) -> bool:
        """Close one logical input after its complete feature stream is durable."""

        record = self._active_record(model_run_id, files)
        if record is None:
            return False
        item = validate_model_input_result(
            publication, record["request"], record["limits"], record["source_bindings"]
        )
        decoded = _nonnegative_safe_integer(decoded_bytes, "decoded_bytes")
        input_id = item["input_id"]
        existing = _by_input(record["completed_inputs"], input_id)
        if existing is not None:
            if existing != item:
                raise self._conflict(
                    model_run_id, "completed input publication differs"
                )
            existing_count = _by_input(
                record["completed_input_decoded_bytes"], input_id
            )
            if existing_count is None or existing_count["decoded_bytes"] != decoded:
                raise self._conflict(
                    model_run_id, "completed input decoded-byte count differs"
                )
            return True
        plan = item["execution_plan"]
        if plan["type"] == "finite":
            allowed_unit_sets = (set(),)
        elif plan["type"] == "core-halo":
            expected_units = set(range(_plan_unit_count(plan)))
            allowed_unit_sets = (expected_units,)
        else:
            allowed_unit_sets = None
        observed_units = {
            unit["unit_id"]
            for unit in record["completed_units"]
            if unit["input_id"] == input_id
        }
        if allowed_unit_sets is not None and observed_units not in allowed_unit_sets:
            raise ValueError("completed input lacks exact core/halo unit coverage")
        updated = _running_update(record)
        released = [
            unit["reusable_state"]["artifact_id"]
            for unit in record["completed_units"]
            if unit["input_id"] == input_id
        ]
        released.extend(
            checkpoint["state"]["artifact_id"]
            for checkpoint in record["checkpoints"]
            if checkpoint["input_id"] == input_id
        )
        updated["released_artifact_ids"] = _released_artifact_ids(record, released)
        updated["completed_units"] = _without_input(record["completed_units"], input_id)
        updated["checkpoints"] = _without_input(record["checkpoints"], input_id)
        updated["failed_units"] = _without_input(record["failed_units"], input_id)
        updated["completed_inputs"] = sorted(
            [*record["completed_inputs"], item], key=lambda value: value["input_id"]
        )
        updated["completed_input_decoded_bytes"] = sorted(
            [
                *record["completed_input_decoded_bytes"],
                {"input_id": input_id, "decoded_bytes": decoded},
            ],
            key=lambda value: value["input_id"],
        )
        self._require_publication_budgets(updated)
        self._records.replace(updated)
        return True

    def complete(self, model_run_id: str, files: ArtifactFileStore) -> bool:
        """Publish COMPLETE only after every requested input is closed."""

        record = self._active_record(model_run_id, files)
        if record is None:
            return False
        if not _all_inputs_complete(record):
            raise ValueError("model run cannot complete before every input completes")
        self._publish_manifest(record, files, complete=True)
        return True

    def fail(
        self,
        model_run_id: str,
        failure: Any,
        files: ArtifactFileStore,
    ) -> bool:
        """Close a run without discarding results that were published first.

        A failure report can arrive after the last input result but before the
        coordinator's completion call. In that case all requested work is already
        durable, so publish COMPLETE. Otherwise preserve the error with reusable
        computation as PARTIAL, or alone as FAILED.
        """

        record = self._active_record(model_run_id, files)
        if record is None:
            return False
        error = validate_gateway_error(failure)
        if _all_inputs_complete(record):
            self._publish_manifest(record, files, complete=True)
        elif _has_reusable_state(record):
            self._publish_manifest(record, files, complete=False, failure=error)
        else:
            updated = _terminal_record(record, "FAILED", self._now())
            updated["failed_units"] = []
            updated["failure"] = error
            self._records.replace(updated)
        return True

    def cancel(self, model_run_id: str, files: ArtifactFileStore) -> None:
        """Idempotently let cancellation win only against an active record."""

        record = self._records.get(model_run_id)
        if record is None or record["phase"] not in {"QUEUED", "RUNNING"}:
            return
        if self._now() >= _parse_z(record["execution_deadline"]):
            self._publish_deadline(record, files)
            return
        self._records.replace(_terminal_record(record, "CANCELLED", self._now()))

    def finalize_deadline(self, model_run_id: str, files: ArtifactFileStore) -> bool:
        """Publish the exact deadline cut when its application deadline is due."""

        record = self._records.get(model_run_id)
        if record is None or record["phase"] not in {"QUEUED", "RUNNING"}:
            return False
        if self._now() < _parse_z(record["execution_deadline"]):
            return False
        self._publish_deadline(record, files)
        return True

    def status(self, model_run_id: str, files: ArtifactFileStore) -> dict[str, Any]:
        """Project one retained record, finalizing an observed due deadline."""

        record = self._retained_record(model_run_id, files)
        value: dict[str, Any] = {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "model_run_id": record["request"]["model_run_id"],
            "phase": record["phase"],
            "request_artifact": record["request_artifact"],
            "completed_unit_count": _completed_unit_count(record),
            "total_unit_count": _total_unit_count(record),
            "phase_deadline": record["phase_deadline"],
        }
        if record["terminal_manifest"] is not None:
            value["terminal_manifest"] = record["terminal_manifest"]
        if record["failure"] is not None:
            value["failure"] = record["failure"]
        return validate_model_run_status(value)

    def cleanup_expired(self, files: ArtifactFileStore) -> None:
        """Delete expired artifacts before removing their retry authority."""

        now = self._now()
        for record in self._records.all():
            if record["phase"] in {"QUEUED", "RUNNING"}:
                if now >= _parse_z(record["execution_deadline"]):
                    self._publish_deadline(record, files)

        records = self._records.all()
        artifact_ids_by_run = {
            record["request"]["model_run_id"]: model_run_record_artifact_ids(record)
            for record in records
        }
        owner_counts = Counter(
            artifact_id
            for artifact_ids in artifact_ids_by_run.values()
            for artifact_id in artifact_ids
        )
        for record in records:
            if record["phase"] in {"QUEUED", "RUNNING"} or now < _parse_z(
                record["phase_deadline"]
            ):
                continue
            model_run_id = record["request"]["model_run_id"]
            artifact_ids = artifact_ids_by_run[model_run_id]
            for artifact_id in sorted(artifact_ids):
                if owner_counts[artifact_id] == 1:
                    files.delete_artifact(artifact_id)
            self._records.remove(model_run_id)
            owner_counts.subtract(artifact_ids)

    def _replacement_state(
        self,
        request: Mapping[str, Any],
        limits: Mapping[str, Any],
        files: ArtifactFileStore,
        now: datetime,
    ) -> dict[str, Any]:
        if "resume_from" not in request:
            return {
                "source_bindings": [],
                "completed_units": [],
                "checkpoints": [],
                "failed_units": [],
                "completed_inputs": [],
                "completed_input_decoded_bytes": [],
                "released_artifact_ids": [],
            }
        previous_id = request["resume_from"]["previous_run_id"]
        previous = self._records.get(previous_id)
        if previous is None:
            raise GatewayFailure(
                GatewayErrorCode.MODEL_RUN_NOT_FOUND,
                "replacement source model run does not exist",
                resource_id=previous_id,
            )
        self._require_retained(previous, now)
        return {
            **select_reusable_model_run_state(previous, request, limits, files),
            "released_artifact_ids": [],
        }

    def has_source_reference(self, artifact_id: str) -> bool:
        """Return whether active/retained durable state still names a source."""

        now = self._now()
        for record in self._records.all():
            if not _record_retains_dependencies(record, now):
                continue
            if any(
                item["raster_input"]["source_view"]
                .get("artifact", {})
                .get("artifact_id")
                == artifact_id
                for item in record["request"]["inputs"]
            ):
                return True
            if any(
                item["source"]["artifact"]["artifact_id"] == artifact_id
                for item in record["source_bindings"]
            ):
                return True
        return False

    def has_active_module_reference(self, module_ref: str) -> bool:
        """Return whether running work still needs one exact installed module."""

        now = self._now()
        return any(
            record["phase"] in {"QUEUED", "RUNNING"}
            and now < _parse_z(record["execution_deadline"])
            and record["request"]["producer"]["module_ref"] == module_ref
            for record in self._records.all()
        )

    def discard_unreferenced_artifact(
        self, reference: Any, files: ArtifactFileStore
    ) -> bool:
        """Delete one rejected late publication only when no run names it.

        The lifecycle lane calls this for the exact artifact supplied by a
        publication that lost a terminal fence. It never scans or guesses at
        concurrently produced files.
        """

        artifact = validate_artifact_ref(reference)
        artifact_id = artifact["artifact_id"]
        if any(
            artifact_id in model_run_record_artifact_ids(record)
            for record in self._records.all()
        ):
            return False
        files.delete_artifact(artifact_id)
        return True

    def _active_record(
        self, model_run_id: str, files: ArtifactFileStore
    ) -> dict[str, Any] | None:
        record = self._records.get(model_run_id)
        if record is None:
            raise GatewayFailure(
                GatewayErrorCode.MODEL_RUN_NOT_FOUND,
                "durable model run does not exist",
                resource_id=model_run_id,
            )
        if record["phase"] not in {"QUEUED", "RUNNING"}:
            return None
        if self._now() >= _parse_z(record["execution_deadline"]):
            self._publish_deadline(record, files)
            return None
        return record

    def _retained_record(self, model_run_id, files):
        record = self._records.get(model_run_id)
        if record is None:
            raise GatewayFailure(
                GatewayErrorCode.MODEL_RUN_NOT_FOUND,
                "durable model run does not exist",
                resource_id=model_run_id,
            )
        if record["phase"] in {"QUEUED", "RUNNING"} and self._now() >= _parse_z(
            record["execution_deadline"]
        ):
            record = self._publish_deadline(record, files)
        self._require_retained(record, self._now())
        return record

    def _publish_deadline(
        self, record: Mapping[str, Any], files: ArtifactFileStore
    ) -> dict[str, Any]:
        if _all_inputs_complete(record):
            return self._publish_manifest(record, files, complete=True)
        if _has_reusable_state(record):
            # A worker may stop after recording a unit failure but before it can
            # fail the run. Preserve that known cause when the deadline closes it.
            failure = (
                record["failed_units"][0]["failure"] if record["failed_units"] else None
            )
            return self._publish_manifest(
                record, files, complete=False, failure=failure
            )
        updated = _terminal_record(record, "TIMED_OUT", self._now())
        updated["failed_units"] = []
        return self._records.replace(updated)

    def _publish_manifest(
        self,
        record: Mapping[str, Any],
        files: ArtifactFileStore,
        *,
        complete: bool,
        failure: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = record["request"]
        manifest = build_model_result_manifest(
            {
                "schema_version": MODEL_RUN_SCHEMA_VERSION,
                "model_run_id": request["model_run_id"],
                "action_id": request["action_id"],
                "backend": request["expected_backend"],
                "request_digest": record["request_digest"],
                "completed_units": record["completed_units"],
                "checkpoints": record["checkpoints"],
                "failed_units": record["failed_units"],
                "completed_inputs": record["completed_inputs"],
                "complete": complete,
                **(
                    {"source_bindings": record["source_bindings"]}
                    if any(
                        "capture" in item["raster_input"]["source_view"]
                        for item in request["inputs"]
                    )
                    else {}
                ),
                **({"failure": failure} if failure is not None else {}),
            },
            request,
            record["limits"],
        )
        artifact = files.publish_artifact_bytes(
            canonical_bytes(manifest), media_type="application/json"
        )
        phase = "COMPLETE" if complete else "PARTIAL"
        updated = _terminal_record(record, phase, self._now())
        updated["terminal_manifest"] = artifact
        updated["manifest"] = manifest
        updated["failure"] = manifest.get("failure")
        return self._records.replace(updated)

    def _require_retained(self, record: Mapping[str, Any], now: datetime) -> None:
        if record["phase"] in {"QUEUED", "RUNNING"}:
            return
        if now >= _parse_z(record["phase_deadline"]):
            raise GatewayFailure(
                GatewayErrorCode.MODEL_RUN_EXPIRED,
                "durable model run has expired",
                resource_id=record["request"]["model_run_id"],
            )

    @staticmethod
    def _require_publication_budgets(record: Mapping[str, Any]) -> None:
        try:
            require_model_run_publication_budgets(
                record["request"],
                record["limits"],
                record["completed_units"],
                record["checkpoints"],
                record["completed_inputs"],
                record["completed_input_decoded_bytes"],
            )
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
                str(error),
                resource_id=record["request"]["model_run_id"],
            ) from error

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("model-run clock must be timezone-aware")
        return value.astimezone(timezone.utc).replace(
            microsecond=value.microsecond // 1000 * 1000
        )

    @staticmethod
    def _conflict(model_run_id: str, message: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.IDEMPOTENCY_CONFLICT,
            message,
            resource_id=model_run_id,
        )


def _accepted(request: Mapping[str, Any]) -> dict[str, Any]:
    return validate_model_run_acceptance(
        request,
        {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "action_id": request["action_id"],
            "model_run_id": request["model_run_id"],
            "request_digest": model_run_request_digest(request),
        },
    )


def _terminal_record(
    record: Mapping[str, Any], phase: str, published_at: datetime
) -> dict[str, Any]:
    updated = dict(record)
    updated["phase"] = phase
    updated["phase_deadline"] = iso_z(
        published_at + timedelta(seconds=record["terminal_retention_seconds"])
    )
    updated["terminal_manifest"] = None
    updated["manifest"] = None
    updated["failure"] = None
    return updated


def _running_update(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable record whose first publication has begun execution."""

    updated = dict(record)
    updated["phase"] = "RUNNING"
    return updated


def _completed_unit_count(record: Mapping[str, Any]) -> int:
    if record["manifest"] is not None:
        return model_result_manifest_completed_unit_count(record["manifest"])
    completed = len(record["completed_units"])
    for item in record["completed_inputs"]:
        completed += _plan_unit_count(item["execution_plan"])
    return completed


def _total_unit_count(record: Mapping[str, Any]) -> int:
    return sum(
        _plan_unit_count(plan)
        for plan in build_expected_model_execution_plans(
            record["request"], record["limits"]
        ).values()
    )


def _has_reusable_state(record: Mapping[str, Any]) -> bool:
    return bool(
        record["completed_units"] or record["checkpoints"] or record["completed_inputs"]
    )


def _all_inputs_complete(record: Mapping[str, Any]) -> bool:
    return {item["input_id"] for item in record["completed_inputs"]} == {
        item["input_id"] for item in record["request"]["inputs"]
    }


def _record_retains_dependencies(record: Mapping[str, Any], now: datetime) -> bool:
    """Whether source/module dependencies can still affect observable run state."""

    if record["phase"] in {"QUEUED", "RUNNING"}:
        return now < _parse_z(record["execution_deadline"])
    return now < _parse_z(record["phase_deadline"])


def _plan_unit_count(plan: Mapping[str, Any]) -> int:
    return len(plan["windows"]) if plan["type"] == "core-halo" else 1


def _unit_key(item: Mapping[str, Any]) -> tuple[int, int]:
    return item["input_id"], item["unit_id"]


def _by_unit(
    values: list[dict[str, Any]], key: tuple[int, int]
) -> dict[str, Any] | None:
    return next((item for item in values if _unit_key(item) == key), None)


def _by_input(values: list[dict[str, Any]], input_id: int) -> dict[str, Any] | None:
    return next((item for item in values if item["input_id"] == input_id), None)


def _sorted_units(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(values, key=_unit_key)


def _without_input(values: list[dict[str, Any]], input_id: int) -> list[dict[str, Any]]:
    return [item for item in values if item["input_id"] != input_id]


def _released_artifact_ids(
    record: Mapping[str, Any], artifact_ids: list[str]
) -> list[str]:
    """Retain physical cleanup ownership after logical state is superseded."""

    return sorted({*record["released_artifact_ids"], *artifact_ids})


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    if value > 2**31 - 1:
        raise ValueError(f"{path} must fit signed 32-bit range")
    return value


def _nonnegative_safe_integer(value: Any, path: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**53 - 1
    ):
        raise ValueError(f"{path} must be a nonnegative safe integer")
    return value


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
