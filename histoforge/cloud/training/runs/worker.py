"""Admit one physical worker and publish its completed training epochs."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

from histoforge.canonical_json import require_canonical_uuid
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.training.capabilities import (
    TrainingCapabilityEntry,
    TrainingCapabilityRegistry,
)
from histoforge.cloud.training.runs.contract import (
    validate_training_run_control_record,
    validate_training_run_record,
)
from histoforge.cloud.training.runs.report import (
    validate_training_report_response_for_run,
)
from histoforge.cloud.training.runtime_identity import (
    current_trainer_build,
    resolve_current_trainer,
)


class TrainingWorkerControl:
    """Translate worker calls to the internal Modal state-lane methods."""

    def __init__(self, lane: Any) -> None:
        self._lane = lane

    def get_run_control(self, run_id: str) -> dict[str, Any]:
        return self._lane.get_worker_run_control.remote(run_id)

    def claim_run_worker(
        self, run_id: str, executing_build: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self._lane.claim_run_worker.remote(run_id, dict(executing_build))

    def advance_run_progress(
        self, run_id: str, report_revision: int, report_digest: str
    ) -> dict[str, Any]:
        return self._lane.advance_worker_run_progress.remote(
            run_id, report_revision, report_digest
        )


def run_training_worker(
    run_id: str,
    *,
    control: Any,
    capabilities: TrainingCapabilityRegistry,
    api_limits: Any,
    train_and_publish: Callable[[Mapping[str, Any], TrainingCapabilityEntry], None],
    build_entry: Callable[[TrainingCapabilityEntry], Mapping[str, Any]] = (
        current_trainer_build
    ),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait for attachment, claim the exact build once, then invoke the trainer.

    ``train_and_publish`` owns dataset reads, model loading, optimization, and
    immutable candidate publication. It is invoked only after the serialized
    state owner accepts the claim. A provider retry therefore cannot restart
    optimization for a run that already began. The existing run reconciler
    converts a stopped call without a complete candidate to a terminal failure.
    """

    run_id = require_canonical_uuid(run_id, "run_id")
    while True:
        observed = validate_training_run_control_record(
            control.get_run_control(run_id), api_limits
        )
        run = observed["run"]
        if run["run_id"] != run_id:
            raise ValueError("training state returned a different run")
        if run["state"] == "RUNNING":
            break
        if run["state"] != "QUEUED":
            raise GatewayFailure(
                GatewayErrorCode.TRAINING_STATE_CONFLICT,
                "training run no longer accepts a worker",
                resource_id=run_id,
            )
        deadline = datetime.fromisoformat(
            observed["worker"]["call_attachment_deadline_at"].replace("Z", "+00:00")
        )
        remaining = (deadline - clock()).total_seconds()
        if remaining <= 0:
            raise GatewayFailure(
                GatewayErrorCode.RUNNER_FAILED,
                "training worker was not attached before its deadline",
                resource_id=run_id,
            )
        sleep(min(0.25, remaining))

    if observed["worker"]["worker_claimed"]:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "automatic restart of an interrupted training run is not supported",
            resource_id=run_id,
        )
    try:
        entry, executing_build = resolve_current_trainer(
            capabilities, run["trainer_build"], build_entry=build_entry
        )
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "the deployed trainer differs from the accepted training build",
            resource_id=run_id,
        ) from error
    claimed = validate_training_run_record(
        control.claim_run_worker(run_id, executing_build), api_limits
    )
    if (
        claimed["run_id"] != run_id
        or claimed["request_digest"] != run["request_digest"]
        or claimed["trainer_build"] != executing_build
        or claimed["state"] != "RUNNING"
    ):
        raise ValueError("training state returned an inconsistent worker claim")
    train_and_publish(claimed, entry)


def publish_training_epoch(
    run: Any,
    epoch: Any,
    *,
    control: Any,
    run_files: Any,
    api_limits: Any,
) -> dict[str, Any]:
    """Append a completed epoch and return the run that advertises its report.

    The trainer callback supplies the epoch's elapsed time and exact metric values
    under the frozen report's series IDs. This function neither computes metrics
    nor selects weights. It reloads the current immutable report rather than
    maintaining another history, and publishes bytes before asking the sole state
    owner to advertise them. The caller retains the returned run for its next epoch.

    A publication error or a lost state response stops the callback without retry.
    Cancellation and stale updates are rejected by the existing state owner; any
    unadvertised candidate remains within ordinary run-transient cleanup.
    """

    report = read_running_training_report(
        run, run_files=run_files, api_limits=api_limits
    )
    report["state"] = "RUNNING"
    report["epochs"].append(epoch)
    candidate = run_files.publish_report_candidate(report)
    revision = run["report_revision"] + 1
    updated = validate_training_run_record(
        control.advance_run_progress(
            run["run_id"], revision, candidate["report_digest"]
        ),
        api_limits,
    )
    if updated["request_digest"] != run["request_digest"]:
        raise ValueError("training progress response changed the accepted request")
    validate_training_report_response_for_run(
        {"report_revision": revision, **candidate}, updated, api_limits
    )
    return updated


def read_running_training_report(
    run: Any, *, run_files: Any, api_limits: Any
) -> dict[str, Any]:
    """Read the exact acknowledged history used by worker progress and weights."""

    run = validate_training_run_record(run, api_limits)
    if run["state"] != "RUNNING":
        raise GatewayFailure(
            GatewayErrorCode.TRAINING_STATE_CONFLICT,
            "training progress is no longer accepted",
            resource_id=run["run_id"],
        )
    response = validate_training_report_response_for_run(
        {
            "report_revision": run["report_revision"],
            **run_files.read_report(run["run_id"], run["report_revision"]),
        },
        run,
        api_limits,
    )
    return response["report"]
