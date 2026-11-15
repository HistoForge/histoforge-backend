"""Closed report schemas for the supported managed trainer component profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    require_finite_number,
    require_nonnegative_int32,
    require_positive_int32,
)
from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.augmentation import (
    validate_resolved_training_augmentation,
)
from histoforge.cloud.training.component_schema import (
    ADAMW_PROFILE_ID,
    DENSE_OBJECTIVE_PROFILE_ID,
    NO_DECAY_RULE,
    OBJECTIVE_PROFILE_ID,
    PARAMETER_GROUP_RULE,
    SCHEDULER_PROFILE_ID,
    managed_objective_rules,
    managed_preprocessing_component,
)
from histoforge.cloud.training.contract import validate_flow_loss_weighting
from histoforge.cloud.training.profiles import instance_sampler_profile

_OPTIMIZER_FIELDS = {
    "parameter_group_rule",
    "trainable_parameters_only",
    "learning_rate",
    "llrd_factor",
    "patch_embed_learning_rate",
    "adapter_learning_rate",
    "neck_learning_rate",
    "head_learning_rate",
    "weight_decay",
    "decoupled_weight_decay",
    "no_decay_rule",
    "task_learning_rate",
    "task_weight_decay",
    "betas",
    "eps",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
}
_SCHEDULER_FIELDS = {
    "total_epochs",
    "warmup_epochs",
    "warmup_start_scale",
    "warmup_end_scale",
    "warmup_endpoint",
    "cosine_epochs",
    "cosine_denominator",
    "epoch_indexing",
    "group_scale_rule",
    "zero_rate_behavior",
}
_OBJECTIVE_SETTING_FIELDS = {
    "head_type",
    "class_count",
    "classification_objective",
    "classification_loss_weight",
    "use_uncertainty_weighting",
    "weight_tversky_by_class",
    "tversky_alpha",
    "tversky_gamma",
    "max_gradient_norm",
}


def validate_managed_training_report_configuration(
    configuration: Mapping[str, Any],
    task: Mapping[str, Any],
    augmentation: Mapping[str, Any],
    total_epochs: int,
    trainer_settings: Mapping[str, Any],
) -> None:
    """Reject unknown or malformed component evidence before it is persisted."""

    _validate_optimizer(configuration["optimizer"])
    _validate_scheduler(configuration["scheduler"], total_epochs)
    if "oversample" in trainer_settings and task["model_output"]["type"] != (
        "instance_segmentation_and_classification"
    ):
        raise ValueError("oversample requires instance classification")
    if canonical_bytes(configuration["sampler"]) != canonical_bytes(
        instance_sampler_profile(trainer_settings.get("oversample", False))
    ):
        raise ValueError("resolved sampler is not supported by trainer_settings")
    _validate_objective(configuration["objective"], task["model_output"])
    validate_resolved_training_augmentation(configuration["augmentation"], augmentation)
    if canonical_bytes(configuration["preprocessing"]) != canonical_bytes(
        managed_preprocessing_component()
    ):
        raise ValueError("resolved preprocessing profile is not supported")


def _validate_optimizer(component: Mapping[str, Any]) -> None:
    parameters = _parameters(
        component, ADAMW_PROFILE_ID, _OPTIMIZER_FIELDS, "optimizer"
    )
    if (
        parameters["parameter_group_rule"] != PARAMETER_GROUP_RULE
        or parameters["no_decay_rule"] != NO_DECAY_RULE
        or parameters["trainable_parameters_only"] is not True
        or parameters["decoupled_weight_decay"] is not True
    ):
        raise ValueError("resolved optimizer implementation is not supported")
    for name in (
        "learning_rate",
        "patch_embed_learning_rate",
        "adapter_learning_rate",
        "neck_learning_rate",
        "head_learning_rate",
        "task_learning_rate",
        "eps",
    ):
        if require_finite_number(parameters[name], name) <= 0:
            raise ValueError(f"resolved optimizer {name} must be positive")
    factor = require_finite_number(parameters["llrd_factor"], "llrd_factor")
    if not 0 < factor <= 1:
        raise ValueError("resolved optimizer llrd_factor must be in (0, 1]")
    for name in ("weight_decay", "task_weight_decay"):
        if require_finite_number(parameters[name], name) < 0:
            raise ValueError(f"resolved optimizer {name} must be nonnegative")
    betas = parameters["betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("resolved optimizer betas must contain two values")
    if any(not 0 <= require_finite_number(value, "beta") < 1 for value in betas):
        raise ValueError("resolved optimizer betas must be in [0, 1)")
    for name in ("amsgrad", "maximize", "capturable", "differentiable"):
        if not isinstance(parameters[name], bool):
            raise ValueError(f"resolved optimizer {name} must be boolean")
    for name in ("foreach", "fused"):
        if parameters[name] is not None and not isinstance(parameters[name], bool):
            raise ValueError(f"resolved optimizer {name} must be boolean or null")


def _validate_scheduler(component: Mapping[str, Any], total_epochs: int) -> None:
    parameters = _parameters(
        component, SCHEDULER_PROFILE_ID, _SCHEDULER_FIELDS, "scheduler"
    )
    total = require_positive_int32(parameters["total_epochs"], "total_epochs")
    warmup = require_nonnegative_int32(parameters["warmup_epochs"], "warmup_epochs")
    cosine = require_positive_int32(parameters["cosine_epochs"], "cosine_epochs")
    denominator = require_positive_int32(
        parameters["cosine_denominator"], "cosine_denominator"
    )
    if total != total_epochs or total != warmup + cosine or denominator != cosine:
        raise ValueError("resolved scheduler epoch counts are inconsistent")
    for name in ("warmup_start_scale", "warmup_end_scale"):
        scale = require_finite_number(parameters[name], name)
        if not 0 <= scale <= 1:
            raise ValueError(f"resolved scheduler {name} must be in [0, 1]")
    if not isinstance(parameters["warmup_endpoint"], bool):
        raise ValueError("resolved scheduler warmup_endpoint must be boolean")
    if (
        parameters["epoch_indexing"] != "zero-based-before-epoch"
        or parameters["group_scale_rule"] != "multiply-initial-learning-rate"
        or parameters["zero_rate_behavior"] != "adamw-step-updates-moments"
    ):
        raise ValueError("resolved scheduler implementation is not supported")


def _validate_objective(
    component: Mapping[str, Any], model_output: Mapping[str, Any]
) -> None:
    channels = model_output_class_channels(model_output)
    classified = bool(channels)
    parameters = component["parameters"]
    dense = model_output["type"] == "dense_classification"
    if not dense and model_output["type"] not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
    }:
        raise ValueError("resolved objective does not support this model output")
    head_type = parameters.get("head_type")
    if not dense and head_type not in ("flow", "hovernet"):
        raise ValueError("resolved objective head type is not supported")
    use_uncertainty = parameters.get("use_uncertainty_weighting")
    if not isinstance(use_uncertainty, bool) or use_uncertainty != classified:
        raise ValueError("resolved objective task balancing differs from model output")
    rules = managed_objective_rules(
        classified=classified,
        use_uncertainty_weighting=use_uncertainty,
    )
    expected_fields = _OBJECTIVE_SETTING_FIELDS | set(rules)
    if dense:
        expected_fields.remove("head_type")
    if head_type == "flow":
        expected_fields.add("flow_loss_weighting")
    parameters = _parameters(
        component,
        DENSE_OBJECTIVE_PROFILE_ID if dense else OBJECTIVE_PROFILE_ID,
        expected_fields,
        "objective",
    )
    if head_type == "flow":
        validate_flow_loss_weighting(parameters["flow_loss_weighting"])
    class_count = require_nonnegative_int32(
        parameters["class_count"], "resolved objective class_count"
    )
    if class_count != len(channels):
        raise ValueError("resolved objective class count differs from model output")
    if parameters["classification_objective"] != "cross_entropy+tversky" or parameters[
        "weight_tversky_by_class"
    ] is not (classified and (dense or head_type == "flow")):
        raise ValueError("resolved classification objective is not supported")
    for name in (
        "classification_loss_weight",
        "tversky_gamma",
        "max_gradient_norm",
    ):
        if require_finite_number(parameters[name], name) <= 0:
            raise ValueError(f"resolved objective {name} must be positive")
    alpha = require_finite_number(parameters["tversky_alpha"], "tversky_alpha")
    if not 0 <= alpha <= 1:
        raise ValueError("resolved objective tversky_alpha must be in [0, 1]")
    for name, expected in rules.items():
        if name != "class_weighting" and canonical_bytes(
            parameters[name]
        ) != canonical_bytes(expected):
            raise ValueError(f"resolved objective {name} is not supported")
    _validate_class_weighting(
        parameters["class_weighting"], model_output, channels, rules
    )


def _validate_class_weighting(
    value: Any,
    model_output: Mapping[str, Any],
    channels: tuple[dict[str, Any], ...],
    rules: Mapping[str, Any],
) -> None:
    expected = rules["class_weighting"]
    if not channels:
        if value != expected:
            raise ValueError("segmentation-only objective has invalid class weighting")
        return
    if not isinstance(value, dict) or set(value) != set(expected) | {"channels"}:
        raise ValueError("resolved class weighting has unknown fields")
    if any(
        canonical_bytes(value[name]) != canonical_bytes(expected[name])
        for name in expected
    ):
        raise ValueError("resolved class weighting policy is not supported")
    # Keep the formula and channel-identity implementation in one place with
    # admission and execution. Import only for classified reports because that
    # owner also contains the NumPy-backed TRAIN pixel counter.
    from histoforge.cloud.training.datasets.class_weights import (
        validated_smoothed_class_weights,
    )

    validated_smoothed_class_weights(value, model_output, len(channels))


def _parameters(
    component: Mapping[str, Any],
    profile_id: str,
    fields: set[str],
    name: str,
) -> Mapping[str, Any]:
    if component["profile_id"] != profile_id or set(component["parameters"]) != fields:
        raise ValueError(f"resolved {name} profile has unknown fields")
    return component["parameters"]
