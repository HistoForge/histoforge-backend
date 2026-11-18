"""Compose training HTTP services from existing file and state owners."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import modal

from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.modules.runtime import current_module_snapshot
from histoforge.cloud.training.capabilities import (
    deployed_training_registry,
    resolve_training_capability,
)
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.service import TrainingDatasetService
from histoforge.cloud.training.runs.access import TrainingRunAccess
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.finalizer import TrainingRunFinalizer
from histoforge.cloud.training.runs.preparation import TrainingRunPreparation
from histoforge.cloud.training.runs.reconciliation import TrainingRunReconciler
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from histoforge.cloud.training.runs.service import TrainingRunService
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_preparation import TrainingSourcePreparation
from histoforge.cloud.training.runs.terminalizer import TrainingRunTerminalizer
from histoforge.cloud.training.runtime_identity import current_trainer_build
from histoforge.cloud.training.state_client import TrainingStateClient

_LOG = logging.getLogger(__name__)


class TrainingGatewayServices:
    """Share mounted file adapters while binding each request's authentication."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        reload_volume: Callable[[], None],
        commit_volume: Callable[[], None],
    ) -> None:
        self._api_limits = DEPLOYMENT_CONFIG.training_api_limits.wire_value()
        self._service_limits = DEPLOYMENT_CONFIG.training_service_limits.wire_value()
        volume = {"reload_volume": reload_volume, "commit_volume": commit_volume}
        self._datasets = TrainingDatasetFiles(artifact_root, **volume)
        self._runs = TrainingRunFiles(artifact_root, self._api_limits, **volume)
        self._sources = TrainingSourceFiles(
            artifact_root, DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes, **volume
        )
        self._results = TrainingResultFiles(
            artifact_root,
            self._api_limits,
            DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes,
            **volume,
        )

    def dataset(self, state: dict[str, Any]) -> TrainingDatasetService:
        """Bind upload, seal, retained-data access, and deletion to one request."""

        return TrainingDatasetService(
            self._datasets,
            TrainingStateClient(state),
            self._api_limits,
            self._service_limits,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            resolve_training_capability,
        )

    def submission(self, state: dict[str, Any]) -> TrainingRunService:
        """Construct one submission path without resolving mutable source state."""

        control = TrainingStateClient(state)

        def prepare(request, recipe, starting_weight_digest):
            # Resolve only after the run service checks exact accepted replay.
            # Later catalog or module changes cannot invalidate that replay.
            return TrainingRunPreparation(
                self._datasets,
                control,
                deployed_training_registry(),
                current_trainer_build,
                self._api_limits,
                DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            )(request, recipe, starting_weight_digest)

        return TrainingRunService(
            TrainingSourcePreparation(
                current_module_snapshot, self._sources, self._api_limits
            ),
            self._runs,
            self._sources,
            self._results,
            control,
            prepare,
            spawn_training_run,
            self._api_limits,
            self._service_limits,
        )

    def access(self, state: dict[str, Any]) -> TrainingRunAccess:
        """Compose read, cancellation, finalization, and terminal-history cleanup."""

        control = TrainingStateClient(state)
        terminalizer = TrainingRunTerminalizer(self._runs, control, self._api_limits)
        finalizer = TrainingRunFinalizer(
            self._runs,
            self._results,
            self._sources,
            control,
            lambda run_id: _finalize_training_result(run_id, state),
            self._api_limits,
        )
        reconciler = TrainingRunReconciler(
            control,
            finalizer,
            terminalizer,
            modal.FunctionCall.from_id,
            self._api_limits,
        )
        return TrainingRunAccess(
            self._runs,
            control,
            reconciler,
            terminalizer,
            modal.FunctionCall.from_id,
            self._api_limits,
        )

    def cleanup(self, state: dict[str, Any]) -> None:
        """Run the same bounded storage cleanup used by normal admission."""

        self.dataset(state).cleanup_stale_artifacts()
        self.submission(state).cleanup_stale_artifacts()


def spawn_training_run(
    run_id: str, resources: Mapping[str, Any], execution_seconds: int
) -> Any:
    """Dispatch the accepted run once with its exact resolved compute settings."""

    try:
        return (
            modal.Function.from_name(
                DEPLOYMENT_CONFIG.gateway_app_name,
                DEPLOYMENT_CONFIG.training_executor_function_name,
            )
            .with_options(
                gpu=resources["gpu"],
                cpu=resources["cpu"],
                memory=resources["memory_mib"],
                timeout=execution_seconds,
                retries=0,
            )
            .spawn(run_id)
        )
    except modal.exception.ResourceExhaustedError as error:
        _LOG.warning(
            "Modal training worker launch rejected for run %s",
            run_id,
            exc_info=True,
        )
        raise _provider_limit_failure(error, run_id) from error


_CREDIT_EXHAUSTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\binsufficient credits?\b",
        r"\bnot enough credits?\b",
        r"\bno (?:remaining|available) credits?\b",
        r"\bout of credits?\b",
        r"\bcredits?(?: balance)? (?:is |are |has been |have been )?"
        r"(?:exhausted|depleted|too low)\b",
    )
)


def _provider_limit_failure(
    error: modal.exception.ResourceExhaustedError, run_id: str
) -> GatewayFailure:
    """Translate one rejected Modal launch without exposing provider diagnostics."""

    provider_message = " ".join(str(error).casefold().split())
    if any(pattern.search(provider_message) for pattern in _CREDIT_EXHAUSTION_PATTERNS):
        return GatewayFailure(
            GatewayErrorCode.PROVIDER_CREDIT_EXHAUSTED,
            "The Modal workspace does not have enough credit to start training. "
            "Add credit or switch to another workspace with available credit.",
            resource_id=run_id,
        )
    return GatewayFailure(
        GatewayErrorCode.PROVIDER_LIMIT_REACHED,
        "Modal could not start training because a workspace limit was reached. "
        "Check Modal Usage & Billing, spend limits, quotas, and rate limits.",
        resource_id=run_id,
    )


def _finalize_training_result(run_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Ask the sole module writer to save and commit the frozen result's outcome."""

    lane = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.module_mutation_class_name,
    )()
    return lane.finalize_training_result.remote(
        run_id, state["deployment_id"], state["auth_revision"]
    )
