"""Pixel-classification objective and metrics, independent of a concrete model."""

from __future__ import annotations

from typing import Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.component_schema import (
    DENSE_OBJECTIVE_PROFILE_ID,
    managed_objective_rules,
)
from histoforge.cloud.training.metric_policy import training_loss_series

VALIDATION_PIXEL_MACRO_F1_SERIES_ID = "validation-pixel-macro-f1.pooled.v1"


def pixel_class_f1_series_id(channel: int) -> str:
    """Stable metric identity for one explicit emitted neural channel."""

    return f"validation-pixel-f1.channel-{channel}.v1"


def dense_objective_component() -> dict[str, Any]:
    """Describe the approved pixel-classification objective for any bound model."""

    return {
        "profile_id": DENSE_OBJECTIVE_PROFILE_ID,
        "parameters": {
            "classification_objective": "cross_entropy+tversky",
            "classification_loss_weight": 1.0,
            "use_uncertainty_weighting": True,
            "weight_tversky_by_class": True,
            "tversky_alpha": 0.3,
            "tversky_gamma": 1.33,
            "max_gradient_norm": 1.0,
            **managed_objective_rules(classified=True, use_uncertainty_weighting=True),
        },
    }


def dense_metric_policy() -> dict[str, Any]:
    """Describe full-ROI pixel metrics without model or decoder assumptions."""

    return {
        "series": [
            training_loss_series(),
            {
                "series_id": VALIDATION_PIXEL_MACRO_F1_SERIES_ID,
                "display_name": "Validation pixel macro F1",
                "partition": "validation",
                "value_domain": "unit_interval",
                "definition": (
                    "Run ordinary full-head argmax on every validation ROI. "
                    "Pool supervised real-pixel TP,FP,FN across all ROIs "
                    "for each emitted channel. Average defined class F1 "
                    "values 2TP/(2TP+FP+FN). Undefined only if all emitted "
                    "denominators are zero. Background and training-only "
                    "winners remain in the confusion counts; no "
                    "connected-component matching or size filter."
                ),
            },
        ],
        "primary_validation_series_id": VALIDATION_PIXEL_MACRO_F1_SERIES_ID,
        "primary_selection_direction": "maximize",
    }


def prepare_dense_metric_policy(entry, model_output) -> dict[str, Any]:
    """Bind per-class series before the run's first report is created."""

    policy = canonical_tree(entry.trainer_profile["metric_policy"])
    for item in model_output_class_channels(model_output):
        if item["role"] == "emitted":
            policy["series"].append(
                {
                    "series_id": pixel_class_f1_series_id(item["channel"]),
                    "display_name": f"Validation pixel F1: {item['pathClass']}",
                    "partition": "validation",
                    "value_domain": "unit_interval",
                    "definition": (
                        f"Pixel F1 for emitted channel {item['channel']} "
                        f"({item['pathClass']}). Pool TP,FP,FN over the complete "
                        "validation set's supervised real pixels, then compute "
                        "2TP/(2TP+FP+FN). Undefined only for a zero denominator; "
                        "target-only or prediction-only support yields zero."
                    ),
                }
            )
    return policy
