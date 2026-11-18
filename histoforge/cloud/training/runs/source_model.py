"""Exact model loading from one accepted managed-training source snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from histoforge.cloud.architectures.registry import require_available_model
from histoforge.cloud.modules.schema import parse_module_recipe
from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.class_schema import require_training_source_evidence
from histoforge.cloud.training.contract import (
    model_with_training_result_settings,
    training_task_from_capability,
)
from histoforge.cloud.training.implementations import resolve_training_implementation
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.contract import validate_training_run_record
from histoforge.cloud.training.runs.report import (
    training_report_digest,
    validate_training_report,
)
from histoforge.cloud.training.runs.source_files import RetainedTrainingSource


@dataclass(frozen=True)
class LoadedTrainingSourceModel:
    """The accepted model projected to its result profile and ordinary runtime."""

    model: Mapping[str, Any]
    network: Any
    target_raster_limits: Mapping[str, int]


def load_training_source_model(
    run: Any,
    report: Any,
    source: RetainedTrainingSource,
    entry: TrainingCapabilityEntry,
    api_limits: Any,
    build_network: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
    adapt_network: Callable[[Any, Mapping[str, Any], str, int], None],
) -> LoadedTrainingSourceModel:
    """Load exact snapshotted weights only after the worker claim succeeds.

    The accepted source, report, and deployed capability must describe the same
    model behavior. The result field profile is applied before ordinary runner
    admission and predictor construction. No current module catalog or mutable
    artifact reference is consulted here.
    """

    limits = validate_training_api_limits(api_limits)
    accepted = validate_training_run_record(run, limits)
    frozen_report = validate_training_report(report, limits)
    marker = source.marker
    request = accepted["request"]
    if accepted["state"] != "RUNNING":
        raise ValueError("training model loading requires a running worker claim")
    if marker["run_id"] != accepted["run_id"]:
        raise ValueError("training source belongs to another run")
    if marker["request_digest"] != accepted["request_digest"]:
        raise ValueError("training source request differs from the accepted run")
    if marker["source_module"] != request["source_module"]:
        raise ValueError("training source module differs from the accepted run")
    if marker["report_digest"] != training_report_digest(frozen_report, limits):
        raise ValueError("training source report differs from the accepted report")
    if marker["starting_weight_digest"] != frozen_report["starting_weight_digest"]:
        raise ValueError("training source weights differ from the accepted report")

    recipe_path = source.root / "source" / "module" / "recipe.json"
    recipe = parse_module_recipe(recipe_path.read_bytes())
    model_id = request["selected_model_id"]
    try:
        source_model = recipe["models"][model_id]
    except KeyError as error:
        raise ValueError("selected model is absent from the training source") from error
    if source_model["arch"] != entry.architecture:
        raise ValueError("training source architecture differs from the trainer")
    require_training_source_evidence(source_model, frozen_report)
    task = frozen_report["task"]
    expected_task = training_task_from_capability(
        entry.capability,
        task["model_output"],
        task["model_microns_per_pixel"],
        limits["report"]["max_text_bytes"],
        annotation_class_map=task["annotation_class_map"],
    )
    if frozen_report["task"] != expected_task:
        raise ValueError("training source model differs from the accepted task")

    implementation = resolve_training_implementation(entry.trainer_implementation_id)
    target_model = implementation.project_model(
        source_model, task, request["class_initialization"]
    )
    model = model_with_training_result_settings(
        target_model,
        frozen_report["field_profile"],
        frozen_report.get("min_instance_pixels"),
        frozen_report["result_inference_gpu"],
    )
    runner_capabilities = require_available_model(model)
    weight_relative = PurePosixPath(marker["starting_weight_path"])
    weight_path = source.root.joinpath(*weight_relative.parts)

    from safetensors.torch import load_file

    state = load_file(str(weight_path))
    network = build_network(source_model["params"], state)
    adapt_network(
        network, model, request["class_initialization"], request["training_seed"]
    )
    return LoadedTrainingSourceModel(
        model=model,
        network=network,
        target_raster_limits=runner_capabilities["finite"]["target_raster"],
    )
