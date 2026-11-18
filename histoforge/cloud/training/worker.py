"""Modal entry point and service composition for managed training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import modal

from histoforge.cloud._app import app, artifact_volume, image
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.training.capabilities import (
    TrainingCapabilityEntry,
    deployed_training_registry,
)
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.runs.executor import train_and_publish_claimed_run
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.worker import (
    TrainingWorkerControl,
    run_training_worker,
)

_ARTIFACT_ROOT = "/artifacts"


@app.function(
    image=image,
    timeout=DEPLOYMENT_CONFIG.training_service_limits.max_training_execution_seconds,
    retries=0,
    volumes={_ARTIFACT_ROOT: artifact_volume},
)
def execute_training_run(run_id: str) -> None:
    """Execute one accepted run through the sole deployed training worker."""

    api_limits = DEPLOYMENT_CONFIG.training_api_limits.wire_value()
    source_limits = DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
    volume = {
        "reload_volume": artifact_volume.reload,
        "commit_volume": artifact_volume.commit,
    }
    datasets = TrainingDatasetFiles(_ARTIFACT_ROOT, **volume)
    sources = TrainingSourceFiles(
        _ARTIFACT_ROOT,
        DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes,
        **volume,
    )
    runs = TrainingRunFiles(_ARTIFACT_ROOT, api_limits, **volume)
    results = TrainingResultFiles(
        _ARTIFACT_ROOT,
        api_limits,
        DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes,
        **volume,
    )
    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.training_state_class_name,
    )()
    control = TrainingWorkerControl(lane)

    def train_and_publish(
        run: Mapping[str, Any], entry: TrainingCapabilityEntry
    ) -> None:
        train_and_publish_claimed_run(
            run,
            entry,
            dataset_files=datasets,
            source_files=sources,
            run_files=runs,
            result_files=results,
            control=control,
            device="cuda",
            api_limits=api_limits,
            source_limits=source_limits,
        )

    run_training_worker(
        run_id,
        control=control,
        capabilities=deployed_training_registry(),
        api_limits=api_limits,
        train_and_publish=train_and_publish,
    )
