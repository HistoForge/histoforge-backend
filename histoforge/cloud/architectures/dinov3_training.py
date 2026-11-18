"""DINOv3 class-output projection and initialization for managed fine-tuning.

Pure projection is usable during admission and result assembly. Scientific
imports occur only when adapting an already strictly loaded network.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import canonical_tree, require_positive_finite_number
from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.class_schema import validate_training_class_transition


def dinov3_class_initializations(
    source_output_type: str, target_output_type: str
) -> tuple[str, ...]:
    """Advertise only the existing task/head combinations this adapter can adapt."""

    if source_output_type == "instance_segmentation":
        if target_output_type == source_output_type:
            return ("use",)
        if target_output_type == "instance_segmentation_and_classification":
            return ("add", "define")
    if source_output_type == target_output_type and source_output_type in {
        "instance_segmentation_and_classification",
        "dense_classification",
    }:
        return ("use", "add", "define")
    return ()


def project_dinov3_training_model(
    source_model: Mapping[str, Any], task: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    """Project only class schema, class-count parameter, and target resolution."""

    output = validate_training_class_transition(
        source_model["output"], task["model_output"], mode
    )
    if mode not in dinov3_class_initializations(
        source_model["output"]["type"], output["type"]
    ):
        raise ValueError("this DINOv3 source cannot train the selected target task")
    result = canonical_tree(source_model)
    result["output"] = output
    result["params"]["num_classes"] = len(model_output_class_channels(output))
    result["input"]["micronsPerPixel"] = require_positive_finite_number(
        task["model_microns_per_pixel"], "model_microns_per_pixel"
    )
    return result


def adapt_dinov3_training_network(
    network: Any, model: Mapping[str, Any], mode: str, seed: int
) -> None:
    """Adapt after strict source loading and before constructing the optimizer."""

    import torch

    from histoforge._vendor.histoforge_models.pathology.class_head import (
        adapt_class_head,
    )

    torch.manual_seed(seed)
    adapt_class_head(network, model["params"]["num_classes"], mode=mode)
