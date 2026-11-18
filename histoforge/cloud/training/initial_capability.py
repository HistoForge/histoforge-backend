"""Managed DINOv3 training candidates; deployment owns their advertisement."""

from __future__ import annotations

from typing import Any

from histoforge.cloud.architectures.registry import (
    FIELD_AGGREGATION_ID,
    FIELD_PREPROCESSING_ID,
)
from histoforge.cloud.modules.schema import (
    VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX,
    VARIABLE_FIELD_PLANNER_ID,
)
from histoforge.cloud.training.bindings.dinov3_dense import dense_trainer_profile
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.contract import FLOW_LOSS_WEIGHTINGS
from histoforge.cloud.training.implementations import (
    DINOV3_DENSE_TRAINER_ID,
    DINOV3_INSTANCE_TRAINER_ID,
)


def initial_cellpose_classification_entry(
    *,
    trainer_execution_seconds: int,
    max_validation_assignment_matrix_entries: int,
) -> TrainingCapabilityEntry:
    """Build the Cellpose instance-classification entry with deployment limits.

    ``deployed_training_registry`` supplies the limits and controls availability.
    """

    return _dinov3_entry(
        architecture="dinov3-adapter-cellpose",
        capability_id="managed-cellpose-instance-classification-v3",
        output_type="instance_segmentation_and_classification",
        trainer_execution_seconds=trainer_execution_seconds,
        max_validation_assignment_matrix_entries=(
            max_validation_assignment_matrix_entries
        ),
    )


def cellpose_segmentation_entry(
    *,
    trainer_execution_seconds: int,
    max_validation_assignment_matrix_entries: int,
) -> TrainingCapabilityEntry:
    """Build the classification-free Cellpose profile.

    Reuses the Cellpose field, flow, augmentation and optimization policies. Its
    profile contains no class-weight vector or learned classification tasks and
    selects by held-out instance mSA. The registry owns advertisement.
    """

    return _dinov3_entry(
        architecture="dinov3-adapter-cellpose",
        capability_id="managed-cellpose-instance-v1",
        output_type="instance_segmentation",
        trainer_execution_seconds=trainer_execution_seconds,
        max_validation_assignment_matrix_entries=(
            max_validation_assignment_matrix_entries
        ),
    )


def hovernet_classification_entry(
    *,
    trainer_execution_seconds: int,
    max_validation_assignment_matrix_entries: int,
) -> TrainingCapabilityEntry:
    """Build the classified HoVer profile with CE-only class weighting."""

    return _dinov3_entry(
        architecture="dinov3-adapter-hovernet",
        capability_id="managed-hovernet-instance-classification-v1",
        output_type="instance_segmentation_and_classification",
        trainer_execution_seconds=trainer_execution_seconds,
        max_validation_assignment_matrix_entries=(
            max_validation_assignment_matrix_entries
        ),
    )


def hovernet_segmentation_entry(
    *,
    trainer_execution_seconds: int,
    max_validation_assignment_matrix_entries: int,
) -> TrainingCapabilityEntry:
    """Build the classification-free HoVer profile."""

    return _dinov3_entry(
        architecture="dinov3-adapter-hovernet",
        capability_id="managed-hovernet-instance-v1",
        output_type="instance_segmentation",
        trainer_execution_seconds=trainer_execution_seconds,
        max_validation_assignment_matrix_entries=(
            max_validation_assignment_matrix_entries
        ),
    )


def semantic_dense_entry(*, trainer_execution_seconds: int) -> TrainingCapabilityEntry:
    """Build the dense profile; the registry owns advertisement."""

    return _dinov3_entry(
        architecture="dinov3-adapter-semantic",
        capability_id="managed-semantic-dense-v1",
        output_type="dense_classification",
        trainer_execution_seconds=trainer_execution_seconds,
    )


def _dinov3_entry(
    *,
    architecture: str,
    capability_id: str,
    output_type: str,
    trainer_execution_seconds: int,
    max_validation_assignment_matrix_entries: int | None = None,
) -> TrainingCapabilityEntry:
    """Share settled model-field controls; task policy owns head-specific losses.

    Candidate resources are not measurements for a newly supported combination.
    Deployment must qualify each complete path before adding it to the registry.
    """

    capability = {
        "capability_id": capability_id,
        "task_display_name": {
            "dense_classification": "Dense classification",
            "instance_segmentation": "Instance segmentation",
            "instance_segmentation_and_classification": (
                "Instance segmentation and classification"
            ),
        }[output_type],
        "input_kind": "field",
        "output_type": output_type,
        "supports_negative_rois": True,
        "crop_padding_model_px": 0,
        "default_field_profile": {
            "planner": VARIABLE_FIELD_PLANNER_ID,
            "minSideModelPx": 512,
            "maxSideModelPx": 1024,
            "preferredTileSideModelPx": 1024,
            "alignmentModelPx": 16,
            "outerArtificialPaddingModelPx": (
                VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX
            ),
            "smallRoiContext": True,
            "tileOverlap": 0.25,
            "aggregation": FIELD_AGGREGATION_ID,
            "preprocessing": FIELD_PREPROCESSING_ID,
        },
        "allowed_field_side_model_px": {"minimum": 512, "maximum": 1024},
        "small_roi_context_editable": True,
        "augmentation_profiles": [
            {
                "profile_id": "he-morphology-v1",
                "display_name": "H&E",
                "description": (
                    "Vary image orientation and appearance during training. "
                    "The stain-color changes are intended for H&E tissue."
                ),
                "operators": [
                    _augmentation_choice(
                        "rotate90-v1",
                        "Rotate by 90-degree increments",
                        "Randomly rotate training images and their annotations "
                        "together in 90-degree steps.",
                    ),
                    _augmentation_choice(
                        "flip-horizontal-v1",
                        "Flip horizontally",
                        "Randomly mirror training images and their annotations "
                        "together from left to right.",
                    ),
                    _augmentation_choice(
                        "flip-vertical-v1",
                        "Flip vertically",
                        "Randomly mirror training images and their annotations "
                        "together from top to bottom.",
                    ),
                    _augmentation_choice(
                        "hed-v1",
                        "H&E stain variation",
                        "Vary colors in H&E-stained training images to represent mild "
                        "staining differences. Not intended for PAS or tasks based "
                        "on IHC stain intensity.",
                    ),
                    _augmentation_choice(
                        "brightness-contrast-v1",
                        "Brightness and contrast variation",
                        "Slightly vary brightness and contrast in training images. "
                        "Bringing images to a common brightness range afterward "
                        "can reduce this effect.",
                    ),
                    _augmentation_choice(
                        "blur-v1",
                        "Mild blur",
                        "Occasionally blur training images slightly to represent "
                        "mild focus differences.",
                    ),
                    _augmentation_choice(
                        "noise-v1",
                        "Image noise",
                        "Add small random changes to pixel colors in training images "
                        "to represent image noise.",
                    ),
                ],
            },
        ],
        "default_augmentation_profile_id": "he-morphology-v1",
        "training_gpus": [
            {"training_gpu_id": "l40s", "display_name": "L40S"},
            {
                "training_gpu_id": "a100-40gb",
                "display_name": "A100 (40 GB)",
            },
        ],
        "total_epochs_range": {"minimum": 1, "maximum": 1_000},
        "automatic_warmup": {"divisor": 8, "maximum_epochs": 10},
        "training_batch_size_range": {"minimum": 1, "maximum": 64},
        "data_loading_workers_range": {"minimum": 0, "maximum": 4},
        "gradient_accumulation_steps_range": {"minimum": 1, "maximum": 64},
        "default_training_settings": {
            "training_gpu_id": "l40s",
            "training_batch_size": 1,
            "data_loading_workers": 4,
            "gradient_accumulation_steps": 1,
            "total_epochs": 80,
            "warmup_mode": "auto",
            "warmup_epochs": 10,
        },
    }
    if output_type == "instance_segmentation_and_classification":
        capability["default_training_settings"]["oversample"] = False
    if architecture == "dinov3-adapter-cellpose":
        capability["flow_loss_weighting_options"] = list(FLOW_LOSS_WEIGHTINGS)
        capability["default_training_settings"]["flow_loss_weighting"] = "pixel"
    augmentation_profiles = {
        "he-morphology-v1": {
            "rotate90-v1": {
                "implementation": "quarter-turn-v1",
                "quarter_turn_probabilities": [0.25, 0.25, 0.25, 0.25],
            },
            "flip-horizontal-v1": {
                "implementation": "axis-flips-v1",
                "horizontal_probability": 0.5,
                "vertical_probability": 0.0,
            },
            "flip-vertical-v1": {
                "implementation": "axis-flips-v1",
                "horizontal_probability": 0.0,
                "vertical_probability": 0.5,
            },
            "hed-v1": {
                "implementation": "masked-hed-v1",
                "probability": 0.5,
                "sigma_range": 0.05,
                "mean_intensity_range": [0.15, 0.85],
            },
            "brightness-contrast-v1": {
                "implementation": "masked-brightness-contrast-v1",
                "probability": 0.9,
                "brightness_range": [0.9, 1.1],
                "contrast_range": [0.9, 1.1],
            },
            "blur-v1": {
                "implementation": "masked-gaussian-blur-v1",
                "probability": 0.1,
                "sigma_range": [0.0, 2.0],
            },
            "noise-v1": {
                "implementation": "masked-gaussian-noise-v1",
                "probability": 0.8,
                "sigma": 0.01,
            },
        },
    }
    dense = output_type == "dense_classification"
    profile_factory = (
        dense_trainer_profile if dense else initial_managed_trainer_profile
    )
    profile = profile_factory(
        architecture,
        capability,
        augmentation_profiles=augmentation_profiles,
        gpu_resources={
            "l40s": {
                "gpu": "L40S",
                "cpu": 4,
                "memory_mib": 32_768,
                **(
                    {
                        "max_validation_assignment_matrix_entries": (
                            max_validation_assignment_matrix_entries
                        )
                    }
                    if max_validation_assignment_matrix_entries is not None
                    else {}
                ),
            },
            "a100-40gb": {
                "gpu": "A100-40GB",
                "cpu": 4,
                "memory_mib": 32_768,
                **(
                    {
                        "max_validation_assignment_matrix_entries": (
                            max_validation_assignment_matrix_entries
                        )
                    }
                    if max_validation_assignment_matrix_entries is not None
                    else {}
                ),
            },
        },
        trainer_execution_seconds=trainer_execution_seconds,
    )
    return TrainingCapabilityEntry(
        architecture,
        capability,
        profile,
        DINOV3_DENSE_TRAINER_ID if dense else DINOV3_INSTANCE_TRAINER_ID,
    )


def _augmentation_choice(
    operator_id: str,
    display_name: str,
    description: str,
) -> dict[str, Any]:
    """Construct one default-enabled H&E checkbox description."""

    return {
        "operator_id": operator_id,
        "display_name": display_name,
        "description": description,
        "default_enabled": True,
    }
