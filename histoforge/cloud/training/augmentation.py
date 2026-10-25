"""Validate and resolve bounded augmentation configuration without tensor imports.

The task trainers execute these recorded selections. No stain preset or
numeric probability is supplied by this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_finite_number,
)
from histoforge.cloud.training.runs.report import validate_resolved_trainer_component

AUGMENTATION_PROFILE_ID = "managed-field-augmentation-v1"
AUGMENTATION_SEED_DERIVATION = "sha256-canonical-json-pcg64-sample-operator-v1"
AUGMENTATION_INPUT_NUMERIC_DOMAIN = "raw-srgb-hwc-finite-0-255-v1"
AUGMENTATION_APPEARANCE_NUMERIC_POLICY = {
    "output_dtype": "float32",
    "real_pixel_range": [0.0, 255.0],
    "clipping": "after-each-appearance-operator-v1",
    "artificial_pixels": "preserve-input-values-v1",
}
_ROTATION_FIELDS = {"implementation", "quarter_turn_probabilities"}
_FLIP_FIELDS = {"implementation", "horizontal_probability", "vertical_probability"}
_HED_FIELDS = {
    "implementation",
    "probability",
    "sigma_range",
    "mean_intensity_range",
}
_BRIGHTNESS_CONTRAST_FIELDS = {
    "implementation",
    "probability",
    "brightness_range",
    "contrast_range",
}
_BLUR_FIELDS = {"implementation", "probability", "sigma_range"}
_NOISE_FIELDS = {"implementation", "probability", "sigma"}
_RESOLVED_PARAMETER_FIELDS = {
    "base_profile_id",
    "base_profile_display_name",
    "default_enabled_operator_ids",
    "enabled_operators",
    "input_numeric_domain",
    "appearance_numeric_policy",
    "seed_derivation",
    "operator_parameters",
}


def validate_augmentation_profiles(
    value: Any, capability: Mapping[str, Any], report_limits: Mapping[str, int]
) -> dict[str, Any]:
    """Require executable parameters for every advertised preset and operator.

    The already-validated capability owns names, order, and default checkboxes.
    Private data owns only the corresponding implementation and numeric settings.
    """

    profiles = capability["augmentation_profiles"]
    if not isinstance(value, dict) or set(value) != {
        profile["profile_id"] for profile in profiles
    }:
        raise ValueError("augmentation parameters must cover every advertised profile")
    result = {}
    for profile in profiles:
        profile_id = profile["profile_id"]
        parameters = value[profile_id]
        ids = [operator["operator_id"] for operator in profile["operators"]]
        if not isinstance(parameters, dict) or set(parameters) != set(ids):
            raise ValueError(
                "augmentation parameters must cover every advertised operator"
            )
        result[profile_id] = {
            operator_id: _validate_operator(parameters[operator_id])
            for operator_id in ids
        }
        # All enabled is the largest possible selection. Apply the existing
        # report bounds before this capability can advertise any subset.
        validate_resolved_trainer_component(
            resolve_training_augmentation(profile, result[profile_id], ids),
            report_limits,
            "augmentation",
        )
    return result


def resolve_training_augmentation(
    profile: Mapping[str, Any],
    operator_parameters: Mapping[str, Any],
    enabled_ids: list[str],
) -> dict[str, Any]:
    """Capture one admitted selection with names and only its enabled parameters.

    The capability validator has checked the preset and canonical subset order;
    the private-profile validator has checked its executable definitions.
    """

    names = {item["operator_id"]: item["display_name"] for item in profile["operators"]}
    return canonical_tree(
        {
            "profile_id": AUGMENTATION_PROFILE_ID,
            "parameters": {
                "base_profile_id": profile["profile_id"],
                "base_profile_display_name": profile["display_name"],
                "default_enabled_operator_ids": [
                    item["operator_id"]
                    for item in profile["operators"]
                    if item["default_enabled"]
                ],
                "enabled_operators": [
                    {"operator_id": operator_id, "display_name": names[operator_id]}
                    for operator_id in enabled_ids
                ],
                "input_numeric_domain": AUGMENTATION_INPUT_NUMERIC_DOMAIN,
                "appearance_numeric_policy": AUGMENTATION_APPEARANCE_NUMERIC_POLICY,
                "seed_derivation": AUGMENTATION_SEED_DERIVATION,
                "operator_parameters": {
                    operator_id: operator_parameters[operator_id]
                    for operator_id in enabled_ids
                },
            },
        }
    )


def validate_resolved_training_augmentation(
    component: Mapping[str, Any], selection: Mapping[str, Any]
) -> None:
    """Require the closed report schema for one resolved augmentation subset."""

    if component["profile_id"] != AUGMENTATION_PROFILE_ID:
        raise ValueError("resolved augmentation profile is not supported")
    parameters = component["parameters"]
    if set(parameters) != _RESOLVED_PARAMETER_FIELDS:
        raise ValueError("resolved augmentation parameters have unknown fields")
    if (
        parameters["seed_derivation"] != AUGMENTATION_SEED_DERIVATION
        or parameters["input_numeric_domain"] != AUGMENTATION_INPUT_NUMERIC_DOMAIN
        or canonical_bytes(parameters["appearance_numeric_policy"])
        != canonical_bytes(AUGMENTATION_APPEARANCE_NUMERIC_POLICY)
    ):
        raise ValueError("resolved augmentation implementation is not supported")
    enabled_ids = [item["operator_id"] for item in parameters["enabled_operators"]]
    if enabled_ids != selection["enabled_operator_ids"]:
        raise ValueError("resolved augmentation differs from the selected operators")
    operator_parameters = parameters["operator_parameters"]
    if not isinstance(operator_parameters, dict) or set(operator_parameters) != set(
        enabled_ids
    ):
        raise ValueError(
            "resolved augmentation parameters must match enabled operators"
        )
    for operator_id in enabled_ids:
        validated = _validate_operator(operator_parameters[operator_id])
        if canonical_bytes(validated) != canonical_bytes(
            operator_parameters[operator_id]
        ):
            raise ValueError("resolved augmentation operator is not canonical")


def _validate_operator(value: Any) -> dict[str, Any]:
    """Validate the closed scientific parameters of one curated implementation."""

    if not isinstance(value, dict):
        raise ValueError("augmentation operator parameters must be an object")
    implementation = value.get("implementation")
    if implementation == "quarter-turn-v1" and set(value) == _ROTATION_FIELDS:
        probabilities = value["quarter_turn_probabilities"]
        if not isinstance(probabilities, list) or len(probabilities) != 4:
            raise ValueError(
                "quarter_turn_probabilities must contain four probabilities"
            )
        probabilities = [_probability(item) for item in probabilities]
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("quarter_turn_probabilities must sum to one")
        return {
            "implementation": implementation,
            "quarter_turn_probabilities": probabilities,
        }
    if implementation == "axis-flips-v1" and set(value) == _FLIP_FIELDS:
        return {
            "implementation": implementation,
            "horizontal_probability": _probability(value["horizontal_probability"]),
            "vertical_probability": _probability(value["vertical_probability"]),
        }
    if implementation == "masked-hed-v1" and set(value) == _HED_FIELDS:
        amount = require_finite_number(value["sigma_range"], "HED sigma_range")
        if not 0 <= amount <= 1:
            raise ValueError("HED sigma_range must be in [0, 1]")
        intensity = _finite_range(value["mean_intensity_range"], "mean_intensity_range")
        if not 0 <= intensity[0] < intensity[1] <= 1:
            raise ValueError(
                "mean_intensity_range must satisfy 0 <= lower < upper <= 1"
            )
        return {
            "implementation": implementation,
            "probability": _probability(value["probability"]),
            "sigma_range": amount,
            "mean_intensity_range": intensity,
        }
    if (
        implementation == "masked-brightness-contrast-v1"
        and set(value) == _BRIGHTNESS_CONTRAST_FIELDS
    ):
        return {
            "implementation": implementation,
            "probability": _probability(value["probability"]),
            "brightness_range": _finite_range(
                value["brightness_range"], "brightness_range", nonnegative=True
            ),
            "contrast_range": _finite_range(
                value["contrast_range"], "contrast_range", nonnegative=True
            ),
        }
    if implementation == "masked-gaussian-blur-v1" and set(value) == _BLUR_FIELDS:
        return {
            "implementation": implementation,
            "probability": _probability(value["probability"]),
            "sigma_range": _finite_range(
                value["sigma_range"], "blur sigma_range", nonnegative=True
            ),
        }
    if implementation == "masked-gaussian-noise-v1" and set(value) == _NOISE_FIELDS:
        sigma = require_finite_number(value["sigma"], "noise sigma")
        if not 0 <= sigma <= 1:
            raise ValueError("noise sigma must be in [0, 1]")
        return {
            "implementation": implementation,
            "probability": _probability(value["probability"]),
            "sigma": sigma,
        }
    raise ValueError("augmentation operator has unsupported implementation or fields")


def _probability(value: Any) -> float:
    """Accept an explicit probability without clamping or inventing a default."""

    result = require_finite_number(value, "augmentation probability")
    if not 0 <= result <= 1:
        raise ValueError("augmentation probability must be in [0, 1]")
    return result


def _finite_range(value: Any, name: str, *, nonnegative: bool = False) -> list[float]:
    """Validate and canonicalize an explicit finite lower/upper pair."""

    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain lower and upper bounds")
    lower, upper = (require_finite_number(item, name) for item in value)
    if lower > upper or (nonnegative and lower < 0):
        qualifier = " ordered nonnegative" if nonnegative else " ordered"
        raise ValueError(f"{name} must contain{qualifier} bounds")
    return [lower, upper]
