from __future__ import annotations

import copy

import pytest
from histoforge.cloud.architectures.registry import FIELD_PREPROCESSING_ID
from histoforge.cloud.training.contract import (
    training_task_from_capability,
    validate_model_training_capability,
    validate_training_augmentation_selection,
    validate_training_field_profile,
    validate_training_field_profile_for_capability,
    validate_training_task,
)


def training_capability() -> dict:
    return {
        "capability_id": "cellpose-sam-cls-fixed-v1",
        "task_display_name": "Leydig cell segmentation",
        "input_kind": "field",
        "output_type": "instance_segmentation",
        "supports_negative_rois": True,
        "crop_padding_model_px": 0,
        "default_field_profile": {
            "planner": "fixed-field-v1",
            "side_model_px": 384,
            "alignment_model_px": 16,
            "small_roi_context": True,
            "tile_overlap": 0.25,
            "aggregation": "mean-logits-v1",
            "preprocessing": FIELD_PREPROCESSING_ID,
        },
        "allowed_field_side_model_px": {"minimum": 384, "maximum": 384},
        "small_roi_context_editable": False,
        "augmentation_profiles": [
            {
                "profile_id": "he-v1",
                "display_name": "H&E",
                "description": "Geometry and bounded H&E color variation.",
                "operators": [
                    {
                        "operator_id": "rotate90-v1",
                        "display_name": "Right-angle rotations",
                        "description": "Rotate the field by multiples of 90 degrees.",
                        "default_enabled": True,
                    },
                    {
                        "operator_id": "flip-v1",
                        "display_name": "Flips",
                        "description": "Reflect the field horizontally or vertically.",
                        "default_enabled": True,
                    },
                ],
            }
        ],
        "default_augmentation_profile_id": "he-v1",
        "training_gpus": [{"training_gpu_id": "l4", "display_name": "L4"}],
        "total_epochs_range": {"minimum": 1, "maximum": 1_000},
        "automatic_warmup": {"divisor": 8, "maximum_epochs": 10},
        "training_batch_size_range": {"minimum": 1, "maximum": 8},
        "data_loading_workers_range": {"minimum": 0, "maximum": 8},
        "gradient_accumulation_steps_range": {"minimum": 1, "maximum": 16},
        "flow_loss_weighting_options": ["pixel", "equal_instance"],
        "default_training_settings": {
            "training_gpu_id": "l4",
            "training_batch_size": 1,
            "data_loading_workers": 2,
            "gradient_accumulation_steps": 1,
            "total_epochs": 80,
            "warmup_mode": "auto",
            "warmup_epochs": 10,
            "flow_loss_weighting": "pixel",
        },
    }


def training_model_output(path_class="Leydig cell") -> dict:
    return {"type": "instance_segmentation", "classes": [path_class]}


def classified_training_model_output(
    emitted=("Leydig cell",), training_only=("Other",)
) -> dict:
    channels = [{"channel": 0, "role": "background"}]
    for role, classes in (("emitted", emitted), ("training_only", training_only)):
        for path_class in classes:
            channels.append(
                {
                    "channel": len(channels),
                    "role": role,
                    "pathClass": path_class,
                }
            )
    return {
        "type": "instance_segmentation_and_classification",
        "classChannels": channels,
    }


def test_capability_derives_exact_dataset_task() -> None:
    capability = validate_model_training_capability(training_capability(), 4096)
    assert training_task_from_capability(
        capability, training_model_output(), 0.5, 4096
    ) == {
        "capability_id": "cellpose-sam-cls-fixed-v1",
        "input_kind": "field",
        "model_microns_per_pixel": 0.5,
        "model_output": training_model_output(),
        "annotation_class_map": {"Leydig cell": "Leydig cell"},
        "supports_negative_rois": True,
    }


def test_capability_rejects_control_text_and_unaligned_variable_fields() -> None:
    invalid_text = training_capability()
    invalid_text["task_display_name"] = "Leydig\nsegmentation"
    with pytest.raises(ValueError, match="control characters"):
        validate_model_training_capability(invalid_text, 4096)

    variable = {
        "planner": "variable-field-v2",
        "minSideModelPx": 256,
        "maxSideModelPx": 1024,
        "preferredTileSideModelPx": 1000,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": 8,
        "smallRoiContext": True,
        "tileOverlap": 0.25,
        "aggregation": "mean-logits-v1",
        "preprocessing": "brightfield-rgb-v1",
    }
    with pytest.raises(ValueError, match="divisible"):
        validate_training_field_profile(variable)

    field_with_crop_padding = training_capability()
    field_with_crop_padding["crop_padding_model_px"] = 16
    with pytest.raises(ValueError, match="zero crop_padding_model_px"):
        validate_model_training_capability(field_with_crop_padding, 4096)


def test_task_rejects_invalid_segmentation_targets() -> None:
    task = training_task_from_capability(
        training_capability(), training_model_output(), 0.5, 4096
    )
    task["model_output"]["classes"].append("Other")
    with pytest.raises(ValueError, match="exactly one post-decoding class"):
        validate_training_task(task)


def test_augmentation_selection_requires_profile_order() -> None:
    capability = training_capability()
    valid = {
        "profile_id": "he-v1",
        "enabled_operator_ids": ["rotate90-v1", "flip-v1"],
    }
    assert validate_training_augmentation_selection(valid, capability, 4096) == valid

    reordered = copy.deepcopy(valid)
    reordered["enabled_operator_ids"].reverse()
    with pytest.raises(ValueError, match="profile order"):
        validate_training_augmentation_selection(reordered, capability, 4096)


def test_variable_field_selection_changes_only_advertised_controls() -> None:
    capability = training_capability()
    capability["default_field_profile"] = {
        "planner": "variable-field-v2",
        "minSideModelPx": 256,
        "maxSideModelPx": 1024,
        "preferredTileSideModelPx": 512,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": 8,
        "smallRoiContext": True,
        "tileOverlap": 0.25,
        "aggregation": "mean-logits-v1",
        "preprocessing": "brightfield-rgb-v1",
    }
    capability["allowed_field_side_model_px"] = {
        "minimum": 256,
        "maximum": 1024,
    }
    capability["small_roi_context_editable"] = True
    selection = copy.deepcopy(capability["default_field_profile"])
    selection["minSideModelPx"] = 384
    selection["preferredTileSideModelPx"] = 768
    selection["smallRoiContext"] = False

    assert (
        validate_training_field_profile_for_capability(selection, capability, 4096)
        == selection
    )

    changed_algorithm = copy.deepcopy(selection)
    changed_algorithm["aggregation"] = "another-aggregation-v1"
    with pytest.raises(ValueError, match="differs from the capability"):
        validate_training_field_profile_for_capability(
            changed_algorithm, capability, 4096
        )

    changed_border = copy.deepcopy(selection)
    changed_border["outerArtificialPaddingModelPx"] = 16
    with pytest.raises(ValueError):
        validate_training_field_profile_for_capability(changed_border, capability, 4096)

    outside_range = copy.deepcopy(selection)
    outside_range["minSideModelPx"] = 128
    with pytest.raises(ValueError, match="outside the capability"):
        validate_training_field_profile_for_capability(outside_range, capability, 4096)
