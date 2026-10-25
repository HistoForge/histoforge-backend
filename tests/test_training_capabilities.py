"""Private training capability resolution and model compatibility."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from histoforge.cloud.architectures.registry import (
    FIELD_AGGREGATION_ID,
    FIELD_PREPROCESSING_ID,
    model_runner_availability,
)
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.capabilities import (
    TrainingCapabilityEntry,
    TrainingCapabilityRegistry,
)
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_state import service_limits


def _model(*, architecture: str = "dinov3-adapter-cellpose") -> dict:
    return {
        "arch": architecture,
        "gpu": "L40S",
        "weights": {"type": "vendored", "file": "model.safetensors"},
        "input": {"type": "field", "micronsPerPixel": 0.22},
        "output": {"type": "instance_segmentation", "classes": ["Cell"]},
        "params": {
            "num_classes": 0,
            "bsize": 384,
            "batch_size": 1,
            "min_instance_pixels": 15,
        },
    }


def _recipe(model: dict | None = None) -> dict:
    return {
        "schemaVersion": 3,
        "name": "Example",
        "models": {"cells": model or _model()},
        "steps": [{"model": "cells"}],
    }


def _capability() -> dict:
    return {
        "capability_id": "fixture-cells-fixed-v1",
        "task_display_name": "Cell segmentation",
        "input_kind": "field",
        "output_type": "instance_segmentation",
        "supports_negative_rois": True,
        "crop_padding_model_px": 0,
        "default_field_profile": {
            "planner": "fixed-field-v1",
            "side_model_px": 384,
            "alignment_model_px": 16,
            "small_roi_context": False,
            "tile_overlap": 0.1,
            "aggregation": "cellpose-tapered-mean-v1",
            "preprocessing": "normalize99-real-image-v1",
        },
        "allowed_field_side_model_px": {"minimum": 384, "maximum": 384},
        "small_roi_context_editable": False,
        "augmentation_profiles": [
            {
                "profile_id": "none-v1",
                "display_name": "None",
                "description": "No stochastic image transforms.",
                "operators": [],
            }
        ],
        "default_augmentation_profile_id": "none-v1",
        "training_gpus": [{"training_gpu_id": "test-gpu", "display_name": "Test"}],
        "total_epochs_range": {"minimum": 1, "maximum": 1_000},
        "automatic_warmup": {"divisor": 8, "maximum_epochs": 10},
        "training_batch_size_range": {"minimum": 1, "maximum": 1},
        "data_loading_workers_range": {"minimum": 0, "maximum": 0},
        "gradient_accumulation_steps_range": {"minimum": 1, "maximum": 1},
        "flow_loss_weighting_options": ["pixel", "equal_instance"],
        "default_training_settings": {
            "training_gpu_id": "test-gpu",
            "training_batch_size": 1,
            "data_loading_workers": 0,
            "gradient_accumulation_steps": 1,
            "total_epochs": 80,
            "warmup_mode": "auto",
            "warmup_epochs": 10,
            "flow_loss_weighting": "pixel",
        },
    }


def _profile(
    capability: dict | None = None,
    architecture: str = "dinov3-adapter-cellpose",
) -> dict:
    return initial_managed_trainer_profile(
        architecture,
        capability or _capability(),
        augmentation_profiles={"none-v1": {}},
        trainer_execution_seconds=3600,
        gpu_resources={
            "test-gpu": {
                "gpu": "L4",
                "cpu": 4,
                "memory_mib": 16384,
                "max_validation_assignment_matrix_entries": 1_000_000,
            }
        },
    )


def _variable_capability() -> dict:
    """Fixture-only field envelope, not a deployed resource or scientific preset."""

    capability = _capability()
    capability.update(
        capability_id="fixture-cells-variable-v2",
        default_field_profile={
            "planner": "variable-field-v2",
            "minSideModelPx": 256,
            "maxSideModelPx": 1024,
            "preferredTileSideModelPx": 512,
            "alignmentModelPx": 16,
            "outerArtificialPaddingModelPx": 8,
            "smallRoiContext": True,
            "tileOverlap": 0.25,
            "aggregation": FIELD_AGGREGATION_ID,
            "preprocessing": FIELD_PREPROCESSING_ID,
        },
        allowed_field_side_model_px={"minimum": 256, "maximum": 1024},
        small_roi_context_editable=True,
    )
    return capability


def _entry(
    capability: dict | None = None,
    architecture: str = "dinov3-adapter-cellpose",
) -> TrainingCapabilityEntry:
    return TrainingCapabilityEntry(
        architecture,
        capability or _capability(),
        _profile(capability, architecture),
        DINOV3_INSTANCE_TRAINER_ID,
    )


def _registry(*entries: TrainingCapabilityEntry) -> TrainingCapabilityRegistry:
    return TrainingCapabilityRegistry(
        entries,
        api_limits=training_api_limits(),
        service_limits=service_limits(),
    )


def test_registry_resolves_the_same_exact_capability_by_model_and_id() -> None:
    capability = _capability()
    registry = _registry(_entry(capability))
    recipe = _recipe()

    availability = registry.availability(
        recipe,
        "cells",
        model_runner_availability(recipe["models"]["cells"]),
    )

    assert availability == {
        "type": "available",
        "options": [{"capability": capability, "class_initializations": ["use"]}],
    }
    assert registry.resolve(capability["capability_id"]) == capability
    resolved = registry.resolve_entry(recipe, "cells", capability["capability_id"])
    assert resolved.capability == capability
    assert resolved.trainer_profile == _profile()


def test_registry_does_not_advertise_epochs_the_report_contract_cannot_store() -> None:
    capability = _capability()
    capability["total_epochs_range"]["maximum"] = 1_001

    with pytest.raises(ValueError, match="total_epochs_range exceeds report limits"):
        _registry(_entry(capability))


@pytest.mark.parametrize(
    "architecture", ["dinov3-adapter-cellpose", "dinov3-adapter-hovernet"]
)
@pytest.mark.parametrize("source_fields", ["fixed", "default", "different"])
def test_variable_capability_checks_new_fields_without_reinterpreting_source(
    architecture: str, source_fields: str
) -> None:
    capability = _variable_capability()
    model = _model(architecture=architecture)
    if architecture == "dinov3-adapter-hovernet":
        model["params"].pop("batch_size")
        capability.pop("flow_loss_weighting_options")
        capability["default_training_settings"].pop("flow_loss_weighting")
    if source_fields != "fixed":
        model["input"]["fieldProfile"] = copy.deepcopy(
            capability["default_field_profile"]
        )
        model["params"].pop("bsize")
        if source_fields == "different":
            # Source inference sizes need not fit the *new training* envelope.
            model["input"]["fieldProfile"].update(
                minSideModelPx=128,
                maxSideModelPx=2048,
                preferredTileSideModelPx=256,
                smallRoiContext=False,
                tileOverlap=0.1,
            )
    recipe = _recipe(model)
    original = copy.deepcopy(recipe)
    registry = _registry(_entry(capability, architecture))
    runner = model_runner_availability(model)
    assert runner["type"] == "available"

    assert registry.availability(recipe, "cells", runner) == {
        "type": "available",
        "options": [{"capability": capability, "class_initializations": ["use"]}],
    }
    assert (
        registry.resolve_entry(recipe, "cells", capability["capability_id"]).capability
        == capability
    )
    assert recipe == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregation", "unknown-v1"),
        ("alignmentModelPx", 32),
    ],
)
def test_variable_capability_rejects_runner_incompatible_result_defaults(field, value):
    capability = _variable_capability()
    capability["default_field_profile"][field] = value
    registry = _registry(_entry(capability))
    recipe = _recipe()
    runner = model_runner_availability(recipe["models"]["cells"])

    assert (
        registry.availability(recipe, "cells", runner)["code"]
        == "unsupported_model_configuration"
    )
    with pytest.raises(ValueError, match="does not match"):
        registry.resolve_entry(recipe, "cells", capability["capability_id"])


def test_variable_crop_capability_keeps_parent_and_context_requirements() -> None:
    capability = _variable_capability()
    capability.update(
        input_kind="crop", crop_padding_model_px=8, small_roi_context_editable=False
    )
    capability["default_field_profile"]["smallRoiContext"] = False
    model = _model()
    model["input"].update(type="crop", framing={"paddingModelPx": 8})
    recipe = _recipe(model)
    registry = _registry(_entry(capability))
    runner = model_runner_availability(model)
    assert registry.availability(recipe, "cells", runner)["type"] == "unavailable"

    recipe["models"]["parent"] = _model()
    recipe["steps"] = [
        {"model": "parent"},
        {"model": "cells", "parentModel": "parent", "parentClass": ["Cell"]},
    ]
    assert (
        registry.resolve_entry(recipe, "cells", capability["capability_id"]).capability
        == capability
    )


def test_registry_does_not_silently_prefer_fixed_or_variable_training() -> None:
    registry = _registry(_entry(), _entry(_variable_capability()))
    recipe = _recipe()
    with pytest.raises(RuntimeError, match="multiple training capabilities"):
        registry.availability(
            recipe, "cells", model_runner_availability(recipe["models"]["cells"])
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model["params"].update(num_classes=2),
        lambda model: model["params"].update(bsize=512),
        lambda model: model["input"].update(type="crop", framing={"paddingModelPx": 8}),
    ],
)
def test_registry_rejects_a_model_that_changes_task_or_field_contract(mutate) -> None:
    registry = _registry(_entry())
    model = _model()
    mutate(model)
    recipe = _recipe(model)

    availability = registry.availability(
        recipe,
        "cells",
        model_runner_availability(model),
    )

    assert availability["type"] == "unavailable"
    assert availability["code"] == "unsupported_model_configuration"


def test_registry_accepts_recipe_owned_path_class_names() -> None:
    registry = _registry(_entry())
    model = _model()
    model["output"]["classes"] = ["Other"]
    recipe = _recipe(model)

    assert (
        registry.availability(recipe, "cells", model_runner_availability(model))["type"]
        == "available"
    )


@pytest.mark.parametrize(
    "component_name",
    ["optimizer", "scheduler", "sampler", "objective", "preprocessing"],
)
def test_registry_rejects_unknown_initial_managed_component_evidence(
    component_name: str,
) -> None:
    capability = _capability()
    supplied = _profile()
    profile = initial_managed_trainer_profile(
        "dinov3-adapter-cellpose",
        capability,
        augmentation_profiles=supplied["augmentation_profiles"],
        gpu_resources=supplied["gpu_resources"],
        trainer_execution_seconds=supplied["trainer_execution_seconds"],
    )
    profile["trainer_components"][component_name]["parameters"][
        "unrecognized_runtime_dump"
    ] = "secret-or-runtime-state"

    with pytest.raises(ValueError, match="differs from its closed preset"):
        _registry(replace(_entry(capability), trainer_profile=profile))


def test_registry_does_not_treat_number_as_boolean_in_managed_preset() -> None:
    capability = _capability()
    supplied = _profile()
    profile = initial_managed_trainer_profile(
        "dinov3-adapter-cellpose",
        capability,
        augmentation_profiles=supplied["augmentation_profiles"],
        gpu_resources=supplied["gpu_resources"],
        trainer_execution_seconds=supplied["trainer_execution_seconds"],
    )
    profile["trainer_components"]["optimizer"]["parameters"][
        "trainable_parameters_only"
    ] = 1

    with pytest.raises(ValueError, match="differs from its closed preset"):
        _registry(replace(_entry(capability), trainer_profile=profile))


def test_registry_requires_the_declared_parent_for_a_crop_capability() -> None:
    capability = _capability()
    capability.update(
        {
            "capability_id": "fixture-crop-v1",
            "input_kind": "crop",
            "crop_padding_model_px": 8,
        }
    )
    model = _model()
    model["input"] = {
        "type": "crop",
        "micronsPerPixel": 0.22,
        "framing": {"paddingModelPx": 8},
    }
    recipe = _recipe(model)
    registry = _registry(_entry(capability))

    unavailable = registry.availability(
        recipe, "cells", model_runner_availability(model)
    )
    assert unavailable["type"] == "unavailable"

    recipe["models"]["parent"] = _model()
    recipe["steps"] = [
        {"model": "parent"},
        {"model": "cells", "parentModel": "parent", "parentClass": ["Cell"]},
    ]
    available = registry.availability(recipe, "cells", model_runner_availability(model))
    assert available["type"] == "available"


def test_unused_model_is_not_offered_or_resolved_for_training() -> None:
    registry = _registry(_entry())
    recipe = _recipe()
    recipe["models"]["unused"] = _model()
    runner = model_runner_availability(recipe["models"]["unused"])

    assert registry.availability(recipe, "unused", runner)["type"] == "unavailable"
    with pytest.raises(ValueError, match="does not match"):
        registry.resolve_entry(recipe, "unused", _capability()["capability_id"])
    assert registry.availability(recipe, "cells", runner)["type"] == "available"


def test_registry_distinguishes_missing_trainer_from_rejected_configuration() -> None:
    empty = _registry()
    recipe = _recipe()
    runner = model_runner_availability(recipe["models"]["cells"])
    assert empty.availability(recipe, "cells", runner)["code"] == (
        "trainer_not_deployed"
    )

    configured = _registry(_entry())
    changed = copy.deepcopy(recipe)
    changed["models"]["cells"]["params"]["num_classes"] = 2
    assert (
        configured.availability(
            changed,
            "cells",
            model_runner_availability(changed["models"]["cells"]),
        )["code"]
        == "unsupported_model_configuration"
    )


def test_registry_rejects_duplicate_ids_and_unknown_resolution() -> None:
    entry = _entry()
    with pytest.raises(ValueError, match="unique"):
        _registry(entry, entry)
    with pytest.raises(ValueError, match="not deployed"):
        _registry().resolve("missing-v1")


def test_registry_rejects_incomplete_private_trainer_profiles() -> None:
    profile = _profile()
    profile["gpu_resources"] = {}
    with pytest.raises(ValueError, match="every advertised"):
        _registry(replace(_entry(), trainer_profile=profile))

    profile = _profile()
    profile["trainer_execution_seconds"] = (
        service_limits()["max_training_execution_seconds"] + 1
    )
    with pytest.raises(ValueError, match="execution time"):
        _registry(replace(_entry(), trainer_profile=profile))


@pytest.mark.parametrize("value", [None, 0, True, 2**31])
def test_registry_requires_a_bounded_validation_assignment_profile(value) -> None:
    profile = _profile()
    resource = profile["gpu_resources"]["test-gpu"]
    if value is None:
        resource.pop("max_validation_assignment_matrix_entries")
    else:
        resource["max_validation_assignment_matrix_entries"] = value

    with pytest.raises(ValueError):
        _registry(replace(_entry(), trainer_profile=profile))
