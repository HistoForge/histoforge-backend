"""Bind complete DINOv3 semantic models to the reusable dense trainer.

Concrete model policy and executable wiring live here; pixel targets, objectives
and validation remain task-owned. Catalog imports never initialize Torch or NumPy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from histoforge.canonical_json import canonical_bytes, canonical_tree
from histoforge.cloud.architectures.dinov3_optimization import (
    managed_dinov3_optimizer_component,
)
from histoforge.cloud.architectures.dinov3_training import adapt_dinov3_training_network
from histoforge.cloud.architectures.registry import FIELD_PREPROCESSING_ID
from histoforge.cloud.training.component_schema import managed_preprocessing_component
from histoforge.cloud.training.profiles import full_coverage_sampler_profile
from histoforge.cloud.training.schedule import managed_scheduler_component
from histoforge.cloud.training.trainers.dense.policy import (
    dense_metric_policy,
    dense_objective_component,
)

if TYPE_CHECKING:
    from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
    from histoforge.cloud.training.runs.source_files import RetainedTrainingSource


def dense_trainer_profile(
    architecture,
    capability,
    *,
    augmentation_profiles,
    gpu_resources,
    trainer_execution_seconds,
) -> dict[str, Any]:
    """Retain semantic checkpoint topology and train only its class prediction."""

    if (
        architecture != "dinov3-adapter-semantic"
        or capability["output_type"] != "dense_classification"
    ):
        raise ValueError("dense trainer requires the semantic DINOv3 model")
    if capability["default_field_profile"]["preprocessing"] != FIELD_PREPROCESSING_ID:
        raise ValueError("dense trainer requires real-field normalization")
    if "flow_loss_weighting_options" in capability:
        raise ValueError("dense training has no flow-loss option")
    if any(
        "max_validation_assignment_matrix_entries" in resource
        for resource in gpu_resources.values()
    ):
        raise ValueError("dense validation has no instance assignment matrix")
    return canonical_tree(
        {
            "trainer_components": {
                "optimizer": managed_dinov3_optimizer_component(),
                "scheduler": managed_scheduler_component(),
                "sampler": full_coverage_sampler_profile(),
                "objective": dense_objective_component(),
                "preprocessing": managed_preprocessing_component(),
            },
            "augmentation_profiles": augmentation_profiles,
            "gpu_resources": gpu_resources,
            "trainer_execution_seconds": trainer_execution_seconds,
            "metric_policy": dense_metric_policy(),
        }
    )


def validate_dense_trainer_profile(architecture, capability, profile) -> None:
    """Validate this concrete policy, independently of component identity strings."""

    expected = dense_trainer_profile(
        architecture,
        capability,
        augmentation_profiles=profile["augmentation_profiles"],
        gpu_resources=profile["gpu_resources"],
        trainer_execution_seconds=profile["trainer_execution_seconds"],
    )
    if canonical_bytes(profile) != canonical_bytes(expected):
        raise ValueError("dense trainer profile differs from its closed policy")


def execute_dense_training(
    entry: TrainingCapabilityEntry,
    *,
    run: Mapping[str, Any],
    report: Mapping[str, Any],
    dataset: RetainedTrainingDataset,
    source: RetainedTrainingSource,
    device: Any,
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, int],
    epoch_completed: Callable[[Mapping[str, Any], Any], None],
) -> Any:
    """Load the exact semantic model, train, and return its complete network."""

    # Preserve targets-before-Torch initialization before importing the builder.
    from histoforge.cloud.training.trainers.dense.loop import train_managed_dense_model

    # isort: split
    # Model construction may import Torch only after the scientific target package.
    from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
    from histoforge.cloud.architectures.dinov3_optimization import (
        build_dinov3_optimizer,
    )
    from histoforge.cloud.architectures.semantic.net import build
    from histoforge.cloud.architectures.semantic.runtime import (
        create_semantic_runtime,
        prepare_semantic_finetuning,
        semantic_training_logits,
    )
    from histoforge.cloud.training.runs.source_model import load_training_source_model

    loaded = load_training_source_model(
        run, report, source, entry, api_limits, build, adapt_dinov3_training_network
    )
    runtime = create_semantic_runtime(loaded.model, loaded.network)
    train_managed_dense_model(
        runtime.network,
        report,
        dataset,
        runtime.predict,
        build_optimizer=build_dinov3_optimizer,
        prepare_model=prepare_semantic_finetuning,
        preprocess=normalize99,
        training_forward=partial(
            semantic_training_logits, class_count=runtime.num_classes
        ),
        capability=entry.capability,
        device=device,
        api_limits=api_limits,
        source_limits=source_limits,
        target_raster_limits=loaded.target_raster_limits,
        epoch_completed=lambda epoch: epoch_completed(epoch, runtime.network),
    )
    return runtime.network
