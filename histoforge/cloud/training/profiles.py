"""Shared private trainer-profile envelopes, resources and sampler evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_nonblank_string,
    require_positive_int32,
)
from histoforge.cloud.training.augmentation import validate_augmentation_profiles
from histoforge.cloud.training.runs.report import (
    TRAINER_CONFIGURATION_FIELDS,
    validate_resolved_trainer_component,
    validate_trainer_metric_policy,
)

_PROFILE_FIELDS = {
    "trainer_components",
    "augmentation_profiles",
    "metric_policy",
    "trainer_execution_seconds",
    "gpu_resources",
}
_GPU_RESOURCE_FIELDS = {
    "gpu",
    "cpu",
    "memory_mib",
}


def full_coverage_sampler_profile() -> dict[str, Any]:
    """Declare the executable sampler without importing training libraries.

    The report already contains the accepted field profile and root seed. This
    component names full coverage, dataset/row-major base order, and the exact
    epoch shuffle without duplicating model settings or storing a ledger.
    """

    return {
        "profile_id": "managed-full-roi-coverage-v1",
        "parameters": {"seed_derivation": "sha256-canonical-json-pcg64-epoch-v1"},
    }


def instance_sampler_profile(oversample: bool) -> dict[str, Any]:
    """Freeze instance-class field sampling; counts stay in objective evidence.

    The weighted profile draws N fields with replacement from the dataset/row-major
    population. Each score sums visible foreground instances divided by original
    TRAIN class-pixel counts (floored at one), then applies the recorded score floor.
    Emitted and training-only channels participate; background contributes no score.
    This evidence is interpretable without importing the executable trainer.
    """

    if not oversample:
        return full_coverage_sampler_profile()
    return {
        "profile_id": "managed-instance-class-field-sampling-v1",
        "parameters": {
            "seed_derivation": "sha256-canonical-json-pcg64-instance-field-epoch-v1",
            "score_floor": 1e-10,
        },
    }


def validate_trainer_profile(
    value: Any,
    capability: Mapping[str, Any],
    api_limits: Mapping[str, Any],
    service_limits: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate deployment-owned components, augmentation choices, and resources."""

    if not isinstance(value, dict) or set(value) != _PROFILE_FIELDS:
        raise ValueError("trainer profile has missing or unknown fields")
    components = value["trainer_components"]
    if not isinstance(components, dict) or set(components) != (
        TRAINER_CONFIGURATION_FIELDS - {"augmentation"}
    ):
        raise ValueError(
            "trainer_components must contain the non-augmentation components"
        )
    execution_seconds = require_positive_int32(
        value["trainer_execution_seconds"], "trainer_execution_seconds"
    )
    if execution_seconds > service_limits["max_training_execution_seconds"]:
        raise ValueError("trainer execution time exceeds deployment limits")
    resources = value["gpu_resources"]
    if not isinstance(resources, dict):
        raise ValueError("gpu_resources must be an object")
    expected_ids = {item["training_gpu_id"] for item in capability["training_gpus"]}
    if set(resources) != expected_ids:
        raise ValueError("gpu_resources must cover every advertised training GPU")
    validated_components = {
        name: validate_resolved_trainer_component(component, api_limits["report"], name)
        for name, component in components.items()
    }
    if "class_count" in validated_components["objective"]["parameters"]:
        raise ValueError("trainer profile must not bind a model class count")
    return canonical_tree(
        {
            "trainer_components": validated_components,
            "augmentation_profiles": validate_augmentation_profiles(
                value["augmentation_profiles"], capability, api_limits["report"]
            ),
            "metric_policy": validate_trainer_metric_policy(
                value["metric_policy"], api_limits
            ),
            "trainer_execution_seconds": execution_seconds,
            "gpu_resources": {
                gpu_id: _validate_gpu_resource(resource)
                for gpu_id, resource in resources.items()
            },
        }
    )


def _validate_gpu_resource(value: Any) -> dict[str, Any]:
    assignment_field = "max_validation_assignment_matrix_entries"
    if (
        not isinstance(value, dict)
        or not _GPU_RESOURCE_FIELDS <= set(value)
        or (set(value) - _GPU_RESOURCE_FIELDS - {assignment_field})
    ):
        raise ValueError("training GPU resource has missing or unknown fields")
    cpu = value["cpu"]
    if (
        isinstance(cpu, bool)
        or not isinstance(cpu, (int, float))
        or not math.isfinite(cpu)
        or cpu <= 0
    ):
        raise ValueError("training GPU cpu must be finite and positive")
    return {
        "gpu": require_nonblank_string(value["gpu"], "training GPU resource gpu"),
        "cpu": float(cpu),
        "memory_mib": require_positive_int32(
            value["memory_mib"], "training GPU resource memory_mib"
        ),
        **(
            {
                assignment_field: require_positive_int32(
                    value[assignment_field], f"training GPU resource {assignment_field}"
                )
            }
            if assignment_field in value
            else {}
        ),
    }
