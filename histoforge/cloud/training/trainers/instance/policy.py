"""Instance objective and metric policies, independent of a concrete model."""

from __future__ import annotations

from typing import Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.component_schema import (
    OBJECTIVE_PROFILE_ID,
    managed_objective_rules,
)
from histoforge.cloud.training.metric_policy import (
    training_loss_series,
)

VALIDATION_AP50_SERIES_ID = "validation-instance-ap50.pooled.v1"
VALIDATION_MSA_SERIES_ID = "validation-instance-msa.pooled.v1"
VALIDATION_OBJECT_MACRO_F1_SERIES_ID = "validation-object-macro-f1.v1"


def object_class_f1_series_id(channel: int) -> str:
    """One stable metric identity per explicit emitted neural channel."""

    return f"validation-object-f1.channel-{channel}.v1"


def prepare_instance_metric_policy(entry, model_output) -> dict[str, Any]:
    """Bind the selected model's emitted classes without mutating the profile."""

    policy = canonical_tree(entry.trainer_profile["metric_policy"])
    for item in model_output_class_channels(model_output):
        if item["role"] != "emitted":
            continue
        channel, path_class = item["channel"], item["pathClass"]
        policy["series"].append(
            {
                "series_id": object_class_f1_series_id(channel),
                "display_name": f"Validation object F1: {path_class}",
                "partition": "validation",
                "value_domain": "unit_interval",
                "definition": (
                    f"Object F1 for emitted channel {channel} ({path_class}). "
                    "Use the same supervised real support and IoU >= 0.50 "
                    "Cellpose assignment as object macro F1. Pool this class's "
                    "TP,FP,FN over all validation ROIs, then compute "
                    "2TP/(2TP+FP+FN). Wrong-class matches add true-class FN and "
                    "predicted-class FP. Undefined only when all three counts "
                    "are zero; target-only or prediction-only support yields zero."
                ),
            }
        )
    return policy


def instance_objective_component(
    *, head_type: str, classified: bool, flow_loss_weighting: str | None
) -> dict[str, Any]:
    """Describe the approved flow/HoVer objective, independent of its backbone."""

    return {
        "profile_id": OBJECTIVE_PROFILE_ID,
        "parameters": {
            "head_type": head_type,
            "classification_objective": "cross_entropy+tversky",
            "classification_loss_weight": 1.0,
            "use_uncertainty_weighting": classified,
            "weight_tversky_by_class": classified and head_type == "flow",
            "tversky_alpha": 0.3,
            "tversky_gamma": 1.33,
            "max_gradient_norm": 1.0,
            **managed_objective_rules(
                classified=classified, use_uncertainty_weighting=classified
            ),
            **(
                {"flow_loss_weighting": flow_loss_weighting}
                if head_type == "flow"
                else {}
            ),
        },
    }


def instance_metric_policy(classified: bool) -> dict[str, Any]:
    """Freeze the task-owned instance metrics, independent of decoder family."""

    support = (
        "Project ground truth and predictions to emitted classes, then restrict "
        "both to supervised real pixels. Use Cellpose threshold-first linear "
        "assignment with its IoU tie term. "
    )
    series = [
        training_loss_series(),
        {
            "series_id": VALIDATION_AP50_SERIES_ID,
            "display_name": "Validation AP50",
            "partition": "validation",
            "value_domain": "unit_interval",
            "definition": support
            + (
                "Match within each validation ROI at IoU >= 0.50, then pool "
                "TP,FP,FN across the complete validation set before calculating "
                "TP/(TP+FP+FN). Undefined only for a zero pooled denominator. This is "
                "not confidence-integrated COCO AP."
            ),
        },
        {
            "series_id": VALIDATION_MSA_SERIES_ID,
            "display_name": "Validation mSA",
            "partition": "validation",
            "value_domain": "unit_interval",
            "definition": support
            + (
                "Match within each validation ROI and pool TP,FP,FN separately "
                "at IoU thresholds 0.50,0.55,...,0.95. Mean the resulting defined "
                "TP/(TP+FP+FN) threshold scores, never ROI scores. Undefined "
                "if all pooled denominators are zero."
            ),
        },
    ]
    if classified:
        series.append(
            {
                "series_id": VALIDATION_OBJECT_MACRO_F1_SERIES_ID,
                "display_name": "Validation object macro F1",
                "partition": "validation",
                "value_domain": "unit_interval",
                "definition": support
                + (
                    "Match at IoU >= 0.50 and pool emitted-class TP,FP,FN across "
                    "ROIs. Wrong-class matches add a true-class FN and a "
                    "predicted-class FP; unmatched objects add FN or FP. "
                    "Average 2TP/(2TP+FP+FN) across classes with ground-truth "
                    "or predicted support; unavailable if none are supported."
                ),
            }
        )
    return {
        "series": series,
        "primary_validation_series_id": (
            VALIDATION_OBJECT_MACRO_F1_SERIES_ID
            if classified
            else VALIDATION_MSA_SERIES_ID
        ),
        "primary_selection_direction": "maximize",
    }
