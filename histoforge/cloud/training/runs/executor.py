"""Execute a claimed run and publish its immutable output candidate."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.implementations import resolve_training_implementation
from histoforge.cloud.training.runs.output import (
    publish_training_result_candidate,
    record_training_epoch,
)
from histoforge.cloud.training.runs.request import (
    validate_training_run_request_against_manifest_capability,
)
from histoforge.cloud.training.runs.result_module import TrainingResultCandidate
from histoforge.cloud.training.runs.worker import read_running_training_report


def train_and_publish_claimed_run(
    run: Mapping[str, Any],
    entry: TrainingCapabilityEntry,
    *,
    dataset_files: Any,
    source_files: Any,
    run_files: Any,
    result_files: Any,
    control: Any,
    device: Any,
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, Any],
) -> TrainingResultCandidate:
    """Train one newly claimed run and publish its complete hidden candidate.

    The caller has already won the one-time worker claim. This function reads
    only the retained dataset and accepted source snapshot, records each complete
    epoch through the existing state owner, and publishes one result candidate.
    The existing reconciler remains responsible for installation and SUCCEEDED.
    """

    implementation = resolve_training_implementation(entry.trainer_implementation_id)
    report = read_running_training_report(
        run, run_files=run_files, api_limits=api_limits
    )
    request = run["request"]
    dataset = dataset_files.read_dataset(request["dataset_id"], api_limits)
    validate_training_run_request_against_manifest_capability(
        request, dataset.manifest, entry.capability, api_limits
    )
    source = source_files.read_source(run["run_id"])

    with tempfile.TemporaryDirectory(prefix="histoforge-training-weights-") as temp:
        weights = Path(temp)
        current_run = run

        def epoch_completed(epoch: Mapping[str, Any], network: Any) -> None:
            nonlocal current_run
            current_run = record_training_epoch(
                current_run,
                epoch,
                network,
                weights_directory=weights,
                control=control,
                run_files=run_files,
                api_limits=api_limits,
            )

        network = implementation.execute(
            entry,
            run=run,
            report=report,
            dataset=dataset,
            source=source,
            device=device,
            api_limits=api_limits,
            source_limits=source_limits,
            epoch_completed=epoch_completed,
        )
        completed_report = read_running_training_report(
            current_run, run_files=run_files, api_limits=api_limits
        )
        return publish_training_result_candidate(
            current_run,
            network,
            weights_directory=weights,
            source_root=source.root,
            source_marker=source.marker,
            run_files=run_files,
            result_files=result_files,
            api_limits=api_limits,
            unavailable_metrics=_unavailable_validation_metrics(completed_report),
            load_model_state=implementation.load_model_state,
            project_model=implementation.project_model,
        )


def _unavailable_validation_metrics(
    report: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Explain only applicable validation series with no defined observation."""

    observed = {
        metric["series_id"] for epoch in report["epochs"] for metric in epoch["metrics"]
    }
    missing_training = [
        series["series_id"]
        for series in report["metric_policy"]["series"]
        if series["partition"] == "training" and series["series_id"] not in observed
    ]
    if missing_training:
        raise ValueError("completed training has no training metric observation")
    if report["split_plan"]["type"] == "none":
        return ()
    return tuple(
        {
            "series_id": series["series_id"],
            "reason": "The complete validation population had no defined value.",
        }
        for series in report["metric_policy"]["series"]
        if series["partition"] == "validation" and series["series_id"] not in observed
    )
