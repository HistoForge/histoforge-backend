"""Bind complete DINOv3 instance models to the reusable instance trainer.

Admission constructs and validates this concrete policy without scientific imports.
Execution supplies model construction, optimizer groups, forward extraction,
preprocessing and ordinary inference; the task owns targets, losses and metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from histoforge.canonical_json import canonical_bytes, canonical_tree
from histoforge.cloud.architectures.dinov3_optimization import (
    managed_dinov3_optimizer_component,
)
from histoforge.cloud.architectures.dinov3_training import adapt_dinov3_training_network
from histoforge.cloud.architectures.registry import FIELD_PREPROCESSING_ID
from histoforge.cloud.training.component_schema import managed_preprocessing_component
from histoforge.cloud.training.profiles import instance_sampler_profile
from histoforge.cloud.training.schedule import managed_scheduler_component
from histoforge.cloud.training.trainers.instance.policy import (
    instance_metric_policy,
    instance_objective_component,
)

if TYPE_CHECKING:
    from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
    from histoforge.cloud.training.runs.source_files import RetainedTrainingSource


def initial_managed_trainer_profile(
    architecture: str,
    capability: Mapping[str, Any],
    *,
    augmentation_profiles: Mapping[str, Any],
    gpu_resources: Mapping[str, Any],
    trainer_execution_seconds: int,
) -> dict[str, Any]:
    """Construct the approved managed profile; advertise nothing by itself."""

    heads = {
        "dinov3-adapter-cellpose": "flow",
        "dinov3-adapter-hovernet": "hovernet",
    }
    if architecture not in heads:
        raise ValueError("initial managed preset requires a DINOv3 instance adapter")
    output_type = capability["output_type"]
    if output_type not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
    }:
        raise ValueError("initial managed preset requires an instance output")
    if capability["default_field_profile"]["preprocessing"] != FIELD_PREPROCESSING_ID:
        raise ValueError("initial managed preset requires real-field normalization")
    classified = output_type == "instance_segmentation_and_classification"
    flow_head = heads[architecture] == "flow"
    if flow_head != ("flow_loss_weighting_options" in capability):
        raise ValueError("flow weighting must be advertised only for a Cellpose head")
    return canonical_tree(
        {
            "trainer_components": {
                "optimizer": managed_dinov3_optimizer_component(),
                "scheduler": managed_scheduler_component(),
                "sampler": instance_sampler_profile(
                    capability["default_training_settings"].get("oversample", False)
                ),
                "objective": instance_objective_component(
                    head_type=heads[architecture],
                    classified=classified,
                    flow_loss_weighting=capability["default_training_settings"].get(
                        "flow_loss_weighting"
                    ),
                ),
                "preprocessing": managed_preprocessing_component(),
            },
            "augmentation_profiles": augmentation_profiles,
            "gpu_resources": gpu_resources,
            "trainer_execution_seconds": trainer_execution_seconds,
            "metric_policy": instance_metric_policy(classified),
        }
    )


def validate_initial_managed_trainer_profile(
    architecture: str,
    capability: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    """Reject changes to the closed initial preset before advertisement.

    This validator belongs to the explicitly selected DINOv3 instance trainer;
    component IDs never select or bypass it. Dynamic augmentation and resource
    records pass through unchanged because their dedicated validators have
    already closed those schemas.
    """

    if any(
        "max_validation_assignment_matrix_entries" not in resource
        for resource in profile["gpu_resources"].values()
    ):
        raise ValueError("instance validation requires an assignment matrix bound")
    expected = initial_managed_trainer_profile(
        architecture,
        capability,
        augmentation_profiles=profile["augmentation_profiles"],
        gpu_resources=profile["gpu_resources"],
        trainer_execution_seconds=profile["trainer_execution_seconds"],
    )
    if canonical_bytes(profile) != canonical_bytes(expected):
        raise ValueError(
            "initial managed trainer profile differs from its closed preset"
        )


def execute_instance_training(
    entry: TrainingCapabilityEntry,
    *,
    run: Mapping[str, Any],
    report: Mapping[str, Any],
    dataset: RetainedTrainingDataset,
    source: RetainedTrainingSource,
    device: Any,
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, Any],
    epoch_completed: Callable[[Mapping[str, Any], Any], None],
) -> Any:
    """Load and train the accepted instance model; return its complete network."""

    # Target construction initializes fastremap before model loading imports
    # Torch. Neither scientific stack is needed by catalog/admission imports.
    from histoforge.cloud.training.trainers.instance.loop import train_managed_model

    # isort: split
    from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
    from histoforge.cloud.architectures.dinov3_optimization import (
        build_dinov3_optimizer,
    )
    from histoforge.cloud.architectures.instance_runtime import (
        create_instance_runtime,
        instance_network_builder,
        instance_training_predictions,
    )
    from histoforge.cloud.training.runs.source_model import load_training_source_model

    loaded = load_training_source_model(
        run,
        report,
        source,
        entry,
        api_limits,
        instance_network_builder(entry.architecture),
        adapt_dinov3_training_network,
    )
    runtime = create_instance_runtime(loaded.model, loaded.network)
    network = runtime.network
    gpu_id = run["request"]["trainer_settings"]["training_gpu_id"]
    resources = entry.trainer_profile["gpu_resources"][gpu_id]
    train_managed_model(
        network,
        report,
        dataset,
        runtime.predict,
        build_optimizer=build_dinov3_optimizer,
        training_forward=instance_training_predictions,
        preprocess=normalize99,
        capability=entry.capability,
        device=device,
        api_limits=api_limits,
        source_limits=source_limits,
        target_raster_limits=loaded.target_raster_limits,
        max_assignment_matrix_entries=resources[
            "max_validation_assignment_matrix_entries"
        ],
        epoch_completed=lambda epoch: epoch_completed(epoch, network),
    )
    return network
