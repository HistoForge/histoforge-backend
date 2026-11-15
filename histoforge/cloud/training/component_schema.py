"""Frozen scientific component identities and report semantics.

Retained reports use this pure schema, never a current trainer policy or catalog.
These IDs describe behavior; they do not dispatch executable implementations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.cloud.architectures.registry import FIELD_PREPROCESSING_ID

ADAMW_PROFILE_ID = "managed-adamw-llrd-v1"
SCHEDULER_PROFILE_ID = "managed-warmup-cosine-v2"
OBJECTIVE_PROFILE_ID = "managed-segmentation-objective-v2"
DENSE_OBJECTIVE_PROFILE_ID = "managed-dense-objective-v1"
CLASS_WEIGHT_POLICY_ID = "sqrt-median-frequency-class-v1"
PARAMETER_GROUP_RULE = "dinov3-source-build-param-groups-v1"
NO_DECAY_RULE = "ndim-lt2-or-name-contains-cls-token-or-reg-token"


def managed_objective_rules(
    *, classified: bool, use_uncertainty_weighting: bool
) -> dict[str, Any]:
    """Evidence for fixed managed-loss semantics, not additional loss options.

    Numeric options that the managed objective actually consumes remain separate.
    Admission adds raw counts and resolved weights to the classification policy;
    those observations do not change its fixed population or computation rule.
    """

    return {
        "initial_log_variance": 0.0,
        "task_combination": (
            "sum-exp-negative-log-variance-loss-plus-log-variance"
            if classified and use_uncertainty_weighting
            else "sum-active-task-losses"
        ),
        "field_reduction": ("loss-specific-supported-reduction-then-equal-field-mean"),
        "effective_support": "supervision-mask-and-real-pixel-mask",
        "cross_entropy_reduction": "sum-weighted-ce-over-sum-target-weights",
        "tversky_reduction": "mean-over-all-class-channels-including-background",
        "gradient_norm_type": 2.0,
        "gradient_clip_parameters": "model-only",
        "precision": "float32-no-autocast",
        "class_weighting": (
            {
                "policy": CLASS_WEIGHT_POLICY_ID,
                "population": "each-supervised-real-pixel-once-per-training-roi",
                "overlapping_rois": "count-independently-without-deduplication",
                "channel_identity": "explicit-model-output-channel",
                "count_floor": 1,
                "formula": "sqrt(median(max(raw_count,1))/max(raw_count,1))",
                "dtype": "float32",
            }
            if classified
            else {"policy": "none"}
        ),
    }


def require_managed_objective_rules(
    component: Mapping[str, Any], *, profile_id: str
) -> None:
    """Check task loss semantics without imposing a model's optimizer or inputs.

    Numeric loss options remain task inputs. This checks only the fixed reduction,
    support and weighting rules implemented by the shared scientific losses.
    """

    options = component["parameters"]
    rules = managed_objective_rules(
        classified=bool(options["class_count"]),
        use_uncertainty_weighting=options["use_uncertainty_weighting"],
    )
    recorded = {name: options[name] for name in rules}
    if options["class_count"]:
        recorded["class_weighting"] = {
            name: value
            for name, value in options["class_weighting"].items()
            if name != "channels"
        }
    if component["profile_id"] != profile_id or recorded != rules:
        raise ValueError("resolved training objective implementation is not supported")


def managed_preprocessing_component() -> dict[str, Any]:
    """Shared image normalization applied to each augmented training sample."""

    return {
        "profile_id": FIELD_PREPROCESSING_ID,
        "parameters": {
            "channels": "RGB",
            "dtype": "float32",
            "population": "real-pixels-per-image-including-context",
            "percentiles": [1.0, 99.0],
            "percentile_method": "linear",
            "minimum_percentile_range": 0.001,
            "small_range_output": 0.0,
            "clip": False,
            "artificial_pixel_value": 0.0,
            "operation_order": [
                "read-field-and-apply-fixed-planner-padding",
                "augment",
                "normalize-real-pixels",
                "apply-model-minimum-alignment-padding",
                "bottom-right-batch-padding",
            ],
        },
    }
