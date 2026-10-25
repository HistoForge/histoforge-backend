"""Shared schema-1 training capability, task, and field-profile contracts.

This module owns pure validation and model-field projection. Deployment policy
decides which curated capabilities exist; stores own persistence and lifecycle.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_finite_number,
    require_nonnegative_int32,
    require_nonnegative_safe_integer,
    require_path_class_name,
    require_positive_finite_number,
    require_positive_int32,
)
from histoforge.cloud.modules.schema import (
    VARIABLE_FIELD_PLANNER_ID,
    model_output_target_classes,
    validate_model_id,
    validate_model_output,
    validate_model_output_type,
    validate_variable_field_profile,
)
from histoforge.cloud.training.class_schema import validate_class_initialization

_STABLE_ID = re.compile(r"[!-~]{1,128}")
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

_TASK_FIELDS = {
    "capability_id",
    "input_kind",
    "model_output",
    "supports_negative_rois",
    "model_microns_per_pixel",
    "annotation_class_map",
}
_FIXED_FIELD_FIELDS = {
    "planner",
    "side_model_px",
    "alignment_model_px",
    "small_roi_context",
    "tile_overlap",
    "aggregation",
    "preprocessing",
}
_INTEGER_RANGE_FIELDS = {"minimum", "maximum"}
_AUTOMATIC_WARMUP_FIELDS = {"divisor", "maximum_epochs"}
_GPU_FIELDS = {"training_gpu_id", "display_name"}
_SETTINGS_FIELDS = {
    "training_gpu_id",
    "training_batch_size",
    "data_loading_workers",
    "gradient_accumulation_steps",
    "total_epochs",
    "warmup_mode",
    "warmup_epochs",
}
_AUGMENTATION_OPERATOR_FIELDS = {
    "operator_id",
    "display_name",
    "description",
    "default_enabled",
}
_AUGMENTATION_PROFILE_FIELDS = {
    "profile_id",
    "display_name",
    "description",
    "operators",
}
_AUGMENTATION_SELECTION_FIELDS = {"profile_id", "enabled_operator_ids"}
_CAPABILITY_FIELDS = {
    "capability_id",
    "task_display_name",
    "input_kind",
    "output_type",
    "supports_negative_rois",
    "crop_padding_model_px",
    "default_field_profile",
    "allowed_field_side_model_px",
    "small_roi_context_editable",
    "augmentation_profiles",
    "default_augmentation_profile_id",
    "training_gpus",
    "total_epochs_range",
    "automatic_warmup",
    "training_batch_size_range",
    "data_loading_workers_range",
    "gradient_accumulation_steps_range",
    "default_training_settings",
}
_AVAILABLE_FIELDS = {"type", "options"}
_TRAINING_OPTION_FIELDS = {"capability", "class_initializations"}
_UNAVAILABLE_FIELDS = {"type", "code", "message"}

_INPUT_KINDS = {"field", "crop"}
_WARMUP_MODES = {"auto", "custom"}
FLOW_LOSS_WEIGHTINGS = ("pixel", "equal_instance")
_UNAVAILABLE_CODES = {"trainer_not_deployed", "unsupported_model_configuration"}


def require_training_stable_id(value: Any, path: str) -> str:
    """Validate one compact ASCII identity used by training contracts."""

    if not isinstance(value, str) or not _STABLE_ID.fullmatch(value):
        raise ValueError(f"{path} must be 1 to 128 visible ASCII characters")
    return value


def require_training_model_id(value: Any, path: str) -> str:
    """Validate one bounded schema-owned model ID used by training DTOs."""

    return require_training_stable_id(validate_model_id(value, path), path)


def require_training_text(value: Any, path: str, max_text_bytes: int) -> str:
    """Validate bounded user-facing text without terminal control characters."""

    result = require_training_text_syntax(value, path)
    if len(result.encode("utf-8")) > max_text_bytes:
        raise ValueError(f"{path} exceeds max_text_bytes")
    return result


def require_training_text_syntax(value: Any, path: str) -> str:
    """Validate stable training-text syntax independently of deployment policy."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonblank string")
    if _CONTROL_CHARACTERS.search(value):
        raise ValueError(f"{path} must not contain control characters")
    return value


def validate_training_task(value: Any) -> dict[str, Any]:
    """Validate the immutable label and raster interpretation in a dataset."""

    task = _exact_object(value, _TASK_FIELDS, "training task")
    output = validate_model_output(task["model_output"], "model_output")
    classes = set(model_output_target_classes(output))
    mapping = _object(task["annotation_class_map"], "annotation_class_map")
    normalized_mapping = {}
    for source, target in mapping.items():
        source = require_path_class_name(source, "annotation_class_map source")
        target = require_path_class_name(target, "annotation_class_map target")
        if target not in classes:
            raise ValueError("annotation_class_map target is not a training class")
        normalized_mapping[source] = target
    return canonical_tree(
        {
            "capability_id": require_training_stable_id(
                task["capability_id"], "capability_id"
            ),
            "input_kind": _enum(task["input_kind"], _INPUT_KINDS, "input_kind"),
            "model_output": output,
            "annotation_class_map": normalized_mapping,
            "supports_negative_rois": _boolean(
                task["supports_negative_rois"], "supports_negative_rois"
            ),
            "model_microns_per_pixel": require_positive_finite_number(
                task["model_microns_per_pixel"], "model_microns_per_pixel"
            ),
        }
    )


def training_task_from_capability(
    capability: Any,
    model_output: Any,
    model_microns_per_pixel: Any,
    max_text_bytes: int,
    *,
    annotation_class_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Combine trainer constraints with the selected recipe-owned model output."""

    item = validate_model_training_capability(capability, max_text_bytes)
    output = validate_model_output(model_output, "model_output")
    if output["type"] != item["output_type"]:
        raise ValueError("model output type differs from the training capability")
    return validate_training_task(
        {
            "capability_id": item["capability_id"],
            "input_kind": item["input_kind"],
            "model_output": output,
            "supports_negative_rois": item["supports_negative_rois"],
            "model_microns_per_pixel": model_microns_per_pixel,
            "annotation_class_map": (
                dict(annotation_class_map)
                if annotation_class_map is not None
                else {name: name for name in model_output_target_classes(output)}
            ),
        }
    )


def validate_training_field_profile(value: Any) -> dict[str, Any]:
    """Validate fixed historical or module-owned variable model fields."""

    profile = _object(value, "training field profile")
    planner = profile.get("planner")
    if planner == "fixed-field-v1":
        item = _exact_object(profile, _FIXED_FIELD_FIELDS, "fixed field profile")
        side = require_positive_int32(item["side_model_px"], "side_model_px")
        alignment = require_positive_int32(
            item["alignment_model_px"], "alignment_model_px"
        )
        if side % alignment:
            raise ValueError("side_model_px must be divisible by alignment_model_px")
        return canonical_tree(
            {
                "planner": planner,
                "side_model_px": side,
                "alignment_model_px": alignment,
                "small_roi_context": _boolean(
                    item["small_roi_context"], "small_roi_context"
                ),
                "tile_overlap": _overlap(item["tile_overlap"], "tile_overlap"),
                "aggregation": require_training_stable_id(
                    item["aggregation"], "aggregation"
                ),
                "preprocessing": require_training_stable_id(
                    item["preprocessing"], "preprocessing"
                ),
            }
        )
    if planner == VARIABLE_FIELD_PLANNER_ID:
        return canonical_tree(validate_variable_field_profile(profile))
    raise ValueError(f"unsupported training field planner: {planner!r}")


def model_with_training_result_settings(
    model: Mapping[str, Any],
    profile: Mapping[str, Any],
    min_instance_pixels: Any,
    result_inference_gpu: Any,
) -> dict[str, Any]:
    """Project the trained model's immutable inference settings.

    Admission and result assembly use the same mapping. A variable result owns
    its complete profile instead of legacy runner-size parameters; a fixed result
    retains the source's fixed parameters. The architecture-owned instance-area
    threshold is independent of training-label construction. This chooses no
    training defaults and does not qualify the starting weights for the new
    inference settings.
    """

    result = canonical_tree(model)
    result["gpu"] = require_training_stable_id(
        result_inference_gpu, "result_inference_gpu"
    )
    if profile["planner"] == VARIABLE_FIELD_PLANNER_ID:
        result["input"]["fieldProfile"] = canonical_tree(profile)
        if "params" in result:
            result["params"].pop("bsize", None)
            result["params"].pop("tile_overlap", None)
    else:
        result["input"].pop("fieldProfile", None)
    settings = validate_instance_training_settings(
        {}
        if min_instance_pixels is None
        else {"min_instance_pixels": min_instance_pixels},
        model["output"],
    )
    result.setdefault("params", {}).update(settings)
    validate_instance_training_settings(result["params"], model["output"])
    return result


def validate_instance_training_settings(
    record: Mapping[str, Any], model_output: Mapping[str, Any]
) -> dict[str, int]:
    """Require the area threshold for instance tasks and forbid it for dense tasks.

    Wire absence is meaningful. Zero is an instance threshold, not a dense sentinel;
    explicit null is never a valid threshold.
    """

    if model_output["type"] == "dense_classification":
        if "min_instance_pixels" in record:
            raise ValueError("dense classification forbids min_instance_pixels")
        return {}
    if model_output["type"] not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
    }:
        raise ValueError("training settings do not support this model output")
    if "min_instance_pixels" not in record:
        raise ValueError("instance training requires min_instance_pixels")
    return {
        "min_instance_pixels": require_nonnegative_int32(
            record["min_instance_pixels"],
            "min_instance_pixels",
        )
    }


def validate_model_training_capability(
    value: Any, max_text_bytes: int
) -> dict[str, Any]:
    """Validate one complete client-visible curated training capability."""

    item = _exact_object(
        value,
        _CAPABILITY_FIELDS,
        "model training capability",
        optional={"flow_loss_weighting_options"},
    )
    output_type = validate_model_output_type(item["output_type"])
    input_kind = _enum(item["input_kind"], _INPUT_KINDS, "input_kind")
    crop_padding = require_nonnegative_int32(
        item["crop_padding_model_px"], "crop_padding_model_px"
    )
    if input_kind == "field" and crop_padding != 0:
        raise ValueError("field input must use zero crop_padding_model_px")
    default_field = validate_training_field_profile(item["default_field_profile"])
    allowed = validate_closed_integer_range(item["allowed_field_side_model_px"])
    editable = _boolean(
        item["small_roi_context_editable"], "small_roi_context_editable"
    )
    _validate_field_capability(input_kind, default_field, allowed, editable)

    profiles = _augmentation_profiles(item["augmentation_profiles"], max_text_bytes)
    if len(profiles) != 1:
        raise ValueError("augmentation_profiles must contain exactly one profile")
    default_profile = require_training_stable_id(
        item["default_augmentation_profile_id"],
        "default_augmentation_profile_id",
    )
    if default_profile not in {profile["profile_id"] for profile in profiles}:
        raise ValueError("default_augmentation_profile_id is not advertised")

    gpus = _training_gpus(item["training_gpus"], max_text_bytes)
    epoch_range = validate_closed_integer_range(item["total_epochs_range"])
    if epoch_range["minimum"] < 1:
        raise ValueError("total epochs must start at one or greater")
    automatic_warmup = validate_automatic_warmup_policy(item["automatic_warmup"])
    batch_range = validate_closed_integer_range(item["training_batch_size_range"])
    worker_range = validate_closed_integer_range(item["data_loading_workers_range"])
    accumulation_range = validate_closed_integer_range(
        item["gradient_accumulation_steps_range"]
    )
    if batch_range["minimum"] < 1 or accumulation_range["minimum"] < 1:
        raise ValueError(
            "batch size and gradient accumulation must start at one or greater"
        )
    flow_options = []
    if "flow_loss_weighting_options" in item:
        flow_options = [
            validate_flow_loss_weighting(option)
            for option in _nonempty_array(
                item["flow_loss_weighting_options"], "flow_loss_weighting_options"
            )
        ]
        if len(set(flow_options)) != len(flow_options):
            raise ValueError("flow_loss_weighting_options must be unique")
        if output_type == "dense_classification":
            raise ValueError("dense classification cannot advertise flow weighting")
    settings = validate_training_settings(item["default_training_settings"])
    if "oversample" in settings and output_type != (
        "instance_segmentation_and_classification"
    ):
        raise ValueError("oversample requires instance classification")
    _validate_settings_ranges(
        settings,
        gpus,
        epoch_range,
        automatic_warmup,
        batch_range,
        worker_range,
        accumulation_range,
        flow_options,
    )

    return canonical_tree(
        {
            "capability_id": require_training_stable_id(
                item["capability_id"], "capability_id"
            ),
            "task_display_name": require_training_text(
                item["task_display_name"], "task_display_name", max_text_bytes
            ),
            "input_kind": input_kind,
            "output_type": output_type,
            "supports_negative_rois": _boolean(
                item["supports_negative_rois"], "supports_negative_rois"
            ),
            "crop_padding_model_px": crop_padding,
            "default_field_profile": default_field,
            "allowed_field_side_model_px": allowed,
            "small_roi_context_editable": editable,
            "augmentation_profiles": profiles,
            "default_augmentation_profile_id": default_profile,
            "training_gpus": gpus,
            "total_epochs_range": epoch_range,
            "automatic_warmup": automatic_warmup,
            "training_batch_size_range": batch_range,
            "data_loading_workers_range": worker_range,
            "gradient_accumulation_steps_range": accumulation_range,
            "default_training_settings": settings,
            **({"flow_loss_weighting_options": flow_options} if flow_options else {}),
        }
    )


def validate_training_availability(value: Any, max_text_bytes: int) -> dict[str, Any]:
    item = _object(value, "training availability")
    kind = item.get("type")
    if kind == "available":
        exact = _exact_object(item, _AVAILABLE_FIELDS, "training availability")
        options = []
        for value_option in _nonempty_array(exact["options"], "training options"):
            option = _exact_object(
                value_option, _TRAINING_OPTION_FIELDS, "training option"
            )
            modes = [
                validate_class_initialization(mode)
                for mode in _nonempty_array(
                    option["class_initializations"], "class_initializations"
                )
            ]
            if len(modes) != len(set(modes)):
                raise ValueError("class_initializations must be unique")
            options.append(
                {
                    "capability": validate_model_training_capability(
                        option["capability"], max_text_bytes
                    ),
                    "class_initializations": modes,
                }
            )
        ids = [option["capability"]["capability_id"] for option in options]
        outputs = [option["capability"]["output_type"] for option in options]
        if len(ids) != len(set(ids)) or len(outputs) != len(set(outputs)):
            raise ValueError("training options must have unique IDs and target tasks")
        return {"type": kind, "options": options}
    if kind == "unavailable":
        exact = _exact_object(item, _UNAVAILABLE_FIELDS, "training availability")
        return {
            "type": kind,
            "code": _enum(exact["code"], _UNAVAILABLE_CODES, "code"),
            "message": require_training_text(
                exact["message"], "message", max_text_bytes
            ),
        }
    raise ValueError(f"unsupported training availability type: {kind!r}")


def validate_training_augmentation_selection(
    value: Any, capability: Any, max_text_bytes: int
) -> dict[str, Any]:
    selection = validate_training_augmentation_selection_syntax(value)
    profile_id = selection["profile_id"]
    resolved = validate_model_training_capability(capability, max_text_bytes)
    matches = [
        profile
        for profile in resolved["augmentation_profiles"]
        if profile["profile_id"] == profile_id
    ]
    if len(matches) != 1:
        raise ValueError("augmentation profile is not advertised")
    enabled = selection["enabled_operator_ids"]
    operator_order = [operator["operator_id"] for operator in matches[0]["operators"]]
    if enabled != [operator for operator in operator_order if operator in set(enabled)]:
        raise ValueError("enabled operators must be unique and follow profile order")
    if not set(enabled) <= set(operator_order):
        raise ValueError("enabled operator is not advertised by the profile")
    return {"profile_id": profile_id, "enabled_operator_ids": enabled}


def validate_training_augmentation_selection_syntax(value: Any) -> dict[str, Any]:
    """Validate a frozen selection without consulting a later capability build."""

    item = _exact_object(
        value, _AUGMENTATION_SELECTION_FIELDS, "training augmentation selection"
    )
    enabled = _stable_id_array(item["enabled_operator_ids"], "enabled_operator_ids")
    if len(enabled) != len(set(enabled)):
        raise ValueError("enabled_operator_ids must be unique")
    return {
        "profile_id": require_training_stable_id(item["profile_id"], "profile_id"),
        "enabled_operator_ids": enabled,
    }


def validate_training_field_profile_for_capability(
    value: Any, capability: Any, max_text_bytes: int
) -> dict[str, Any]:
    """Require one requested field profile to remain inside its capability."""

    profile = validate_training_field_profile(value)
    resolved = validate_model_training_capability(capability, max_text_bytes)
    default = resolved["default_field_profile"]
    if profile["planner"] != default["planner"]:
        raise ValueError("training field planner differs from the capability")
    if profile["planner"] == "fixed-field-v1":
        if profile != default:
            raise ValueError("fixed training field profile must equal the capability")
        return profile

    for field in (
        "alignmentModelPx",
        "outerArtificialPaddingModelPx",
        "tileOverlap",
        "aggregation",
        "preprocessing",
    ):
        if profile[field] != default[field]:
            raise ValueError(f"training field {field} differs from the capability")
    if (
        not resolved["small_roi_context_editable"]
        and profile["smallRoiContext"] != default["smallRoiContext"]
    ):
        raise ValueError("smallRoiContext is not editable for this capability")
    allowed = resolved["allowed_field_side_model_px"]
    for field in (
        "minSideModelPx",
        "preferredTileSideModelPx",
        "maxSideModelPx",
    ):
        if not allowed["minimum"] <= profile[field] <= allowed["maximum"]:
            raise ValueError(f"training field {field} is outside the capability")
    return profile


def validate_training_settings_for_capability(
    value: Any, capability: Any, max_text_bytes: int
) -> dict[str, Any]:
    """Require one user-selected compute record to remain inside its capability."""

    settings = validate_training_settings(value)
    resolved = validate_model_training_capability(capability, max_text_bytes)
    if ("oversample" in settings) != (
        "oversample" in resolved["default_training_settings"]
    ):
        raise ValueError("oversample presence must match the advertised setting")
    _validate_settings_ranges(
        settings,
        resolved["training_gpus"],
        resolved["total_epochs_range"],
        resolved["automatic_warmup"],
        resolved["training_batch_size_range"],
        resolved["data_loading_workers_range"],
        resolved["gradient_accumulation_steps_range"],
        resolved.get("flow_loss_weighting_options", []),
    )
    return settings


def validate_closed_integer_range(value: Any) -> dict[str, int]:
    item = _exact_object(value, _INTEGER_RANGE_FIELDS, "closed integer range")
    minimum = require_nonnegative_safe_integer(item["minimum"], "minimum")
    maximum = require_nonnegative_safe_integer(item["maximum"], "maximum")
    if maximum < minimum:
        raise ValueError("closed integer range maximum is below minimum")
    return {"minimum": minimum, "maximum": maximum}


def validate_automatic_warmup_policy(value: Any) -> dict[str, int]:
    """Validate the exact capability-owned Auto warmup rule."""

    item = _exact_object(value, _AUTOMATIC_WARMUP_FIELDS, "automatic warmup")
    return {
        "divisor": require_positive_int32(item["divisor"], "divisor"),
        "maximum_epochs": require_nonnegative_int32(
            item["maximum_epochs"], "maximum_epochs"
        ),
    }


def automatic_warmup_epochs(total_epochs: Any, policy: Any) -> int:
    """Resolve Auto deterministically from one advertised policy."""

    total = require_positive_int32(total_epochs, "total_epochs")
    resolved = validate_automatic_warmup_policy(policy)
    return min(resolved["maximum_epochs"], total // resolved["divisor"])


def validate_training_settings(value: Any) -> dict[str, Any]:
    item = _exact_object(
        value,
        _SETTINGS_FIELDS,
        "training settings",
        optional={"flow_loss_weighting", "oversample"},
    )
    total_epochs = require_positive_int32(item["total_epochs"], "total_epochs")
    warmup_epochs = require_nonnegative_int32(item["warmup_epochs"], "warmup_epochs")
    if warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must be less than total_epochs")
    return {
        "training_gpu_id": require_training_stable_id(
            item["training_gpu_id"], "training_gpu_id"
        ),
        "training_batch_size": require_positive_int32(
            item["training_batch_size"], "training_batch_size"
        ),
        "data_loading_workers": require_nonnegative_safe_integer(
            item["data_loading_workers"], "data_loading_workers"
        ),
        "gradient_accumulation_steps": require_positive_int32(
            item["gradient_accumulation_steps"], "gradient_accumulation_steps"
        ),
        "total_epochs": total_epochs,
        "warmup_mode": _enum(item["warmup_mode"], _WARMUP_MODES, "warmup_mode"),
        "warmup_epochs": warmup_epochs,
        **(
            {"oversample": _boolean(item["oversample"], "oversample")}
            if "oversample" in item
            else {}
        ),
        **(
            {
                "flow_loss_weighting": validate_flow_loss_weighting(
                    item["flow_loss_weighting"]
                )
            }
            if "flow_loss_weighting" in item
            else {}
        ),
    }


def validate_flow_loss_weighting(value: Any) -> str:
    """Validate the Cellpose flow-loss choice without a live capability lookup."""

    return _enum(value, set(FLOW_LOSS_WEIGHTINGS), "flow_loss_weighting")


def _augmentation_profiles(value: Any, max_text_bytes: int) -> list[dict[str, Any]]:
    values = _nonempty_array(value, "augmentation_profiles")
    result = []
    for value_item in values:
        item = _exact_object(
            value_item, _AUGMENTATION_PROFILE_FIELDS, "augmentation profile"
        )
        operators = []
        for operator_value in _array(item["operators"], "operators"):
            operator = _exact_object(
                operator_value,
                _AUGMENTATION_OPERATOR_FIELDS,
                "augmentation operator",
            )
            operators.append(
                {
                    "operator_id": require_training_stable_id(
                        operator["operator_id"], "operator_id"
                    ),
                    "display_name": require_training_text(
                        operator["display_name"], "display_name", max_text_bytes
                    ),
                    "description": require_training_text(
                        operator["description"], "description", max_text_bytes
                    ),
                    "default_enabled": _boolean(
                        operator["default_enabled"], "default_enabled"
                    ),
                }
            )
        operator_ids = [operator["operator_id"] for operator in operators]
        if len(operator_ids) != len(set(operator_ids)):
            raise ValueError("augmentation profile contains duplicate operator IDs")
        result.append(
            {
                "profile_id": require_training_stable_id(
                    item["profile_id"], "profile_id"
                ),
                "display_name": require_training_text(
                    item["display_name"], "display_name", max_text_bytes
                ),
                "description": require_training_text(
                    item["description"], "description", max_text_bytes
                ),
                "operators": operators,
            }
        )
    profile_ids = [profile["profile_id"] for profile in result]
    if len(profile_ids) != len(set(profile_ids)):
        raise ValueError("augmentation_profiles contains duplicate profile IDs")
    return result


def _training_gpus(value: Any, max_text_bytes: int) -> list[dict[str, str]]:
    values = _nonempty_array(value, "training_gpus")
    result = []
    for value_item in values:
        item = _exact_object(value_item, _GPU_FIELDS, "training GPU")
        result.append(
            {
                "training_gpu_id": require_training_stable_id(
                    item["training_gpu_id"], "training_gpu_id"
                ),
                "display_name": require_training_text(
                    item["display_name"], "display_name", max_text_bytes
                ),
            }
        )
    ids = [gpu["training_gpu_id"] for gpu in result]
    if len(ids) != len(set(ids)):
        raise ValueError("training_gpus contains duplicate IDs")
    return result


def _validate_field_capability(
    input_kind: str,
    profile: Mapping[str, Any],
    allowed: Mapping[str, int],
    editable: bool,
) -> None:
    context = (
        profile["small_roi_context"]
        if profile["planner"] == "fixed-field-v1"
        else profile["smallRoiContext"]
    )
    if input_kind == "crop" and (context or editable):
        raise ValueError("crop input cannot enable editable small-ROI context")
    if profile["planner"] == "fixed-field-v1":
        side = profile["side_model_px"]
        if allowed != {"minimum": side, "maximum": side} or editable:
            raise ValueError("fixed field capability must expose one immutable side")
        return
    sides = (
        profile["minSideModelPx"],
        profile["preferredTileSideModelPx"],
        profile["maxSideModelPx"],
    )
    if any(side < allowed["minimum"] or side > allowed["maximum"] for side in sides):
        raise ValueError("default variable field sides are outside the allowed range")


def _validate_settings_ranges(
    settings: Mapping[str, Any],
    gpus: Sequence[Mapping[str, str]],
    epoch_range: Mapping[str, int],
    automatic_warmup: Mapping[str, int],
    batch_range: Mapping[str, int],
    worker_range: Mapping[str, int],
    accumulation_range: Mapping[str, int],
    flow_options: Sequence[str],
) -> None:
    if flow_options:
        if settings.get("flow_loss_weighting") not in flow_options:
            raise ValueError("flow_loss_weighting must name an advertised option")
    elif "flow_loss_weighting" in settings:
        raise ValueError("flow_loss_weighting is not advertised")
    if settings["training_gpu_id"] not in {gpu["training_gpu_id"] for gpu in gpus}:
        raise ValueError("training GPU is not advertised")
    if not epoch_range["minimum"] <= settings["total_epochs"] <= epoch_range["maximum"]:
        raise ValueError("total_epochs is outside its advertised range")
    if settings["warmup_mode"] == "auto" and settings[
        "warmup_epochs"
    ] != automatic_warmup_epochs(settings["total_epochs"], automatic_warmup):
        raise ValueError("automatic warmup does not match the advertised policy")
    for field, bounds in (
        ("training_batch_size", batch_range),
        ("data_loading_workers", worker_range),
        ("gradient_accumulation_steps", accumulation_range),
    ):
        if not bounds["minimum"] <= settings[field] <= bounds["maximum"]:
            raise ValueError(f"{field} is outside its advertised range")


def _overlap(value: Any, path: str) -> float:
    result = require_finite_number(value, path)
    if result < 0.0 or result >= 1.0:
        raise ValueError(f"{path} must be in [0, 1)")
    return result


def _stable_id_array(value: Any, path: str) -> list[str]:
    return [
        require_training_stable_id(item, f"{path} item") for item in _array(value, path)
    ]


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _enum(value: Any, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{path} is unsupported")
    return value


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_object(
    value: Any,
    fields: set[str],
    path: str,
    *,
    optional: set[str] | frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    item = _object(value, path)
    if not fields <= set(item) <= fields | optional:
        raise ValueError(f"{path} has missing or unknown fields")
    return item


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _nonempty_array(value: Any, path: str) -> list[Any]:
    result = _array(value, path)
    if not result:
        raise ValueError(f"{path} must be nonempty")
    return result
