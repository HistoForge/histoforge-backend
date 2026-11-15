"""Bind shared schedule, augmentation and TRAIN-only class-weight evidence.

Task adapters choose their objective and bind its applicable options. This owner
does not choose a trainer, head, loss, model architecture or validation metric.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.augmentation import resolve_training_augmentation
from histoforge.cloud.training.component_schema import CLASS_WEIGHT_POLICY_ID
from histoforge.cloud.training.contract import automatic_warmup_epochs

if TYPE_CHECKING:
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset


def resolve_training_components(
    profile: Mapping[str, Any],
    capability: Mapping[str, Any],
    augmentation_selection: Mapping[str, Any],
    model_output: Mapping[str, Any],
    trainer_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy a profile and resolve user-selected schedule and augmentation."""

    if model_output.get("type") != capability["output_type"]:
        raise ValueError("model output type differs from the training capability")
    profile_id = augmentation_selection["profile_id"]
    choice = next(
        item
        for item in capability["augmentation_profiles"]
        if item["profile_id"] == profile_id
    )
    components = canonical_tree(profile["trainer_components"])
    total_epochs = trainer_settings["total_epochs"]
    warmup_epochs = trainer_settings["warmup_epochs"]
    if trainer_settings["warmup_mode"] == "auto" and warmup_epochs != (
        automatic_warmup_epochs(total_epochs, capability["automatic_warmup"])
    ):
        raise ValueError("automatic warmup does not match the advertised policy")
    cosine_epochs = total_epochs - warmup_epochs
    components["scheduler"]["parameters"].update(
        {
            "total_epochs": total_epochs,
            "warmup_epochs": warmup_epochs,
            "cosine_epochs": cosine_epochs,
            "cosine_denominator": cosine_epochs,
        }
    )
    components["augmentation"] = resolve_training_augmentation(
        choice,
        profile["augmentation_profiles"][profile_id],
        augmentation_selection["enabled_operator_ids"],
    )
    return components


def bind_training_class_weights(
    configuration: dict[str, Any],
    *,
    dataset: RetainedTrainingDataset,
    training_roi_indices: Sequence[int],
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    model_output: Mapping[str, Any],
    max_json_member_bytes: int,
) -> None:
    """Record frequencies once per TRAIN ROI before sampling or augmentation."""

    from histoforge.cloud.training.datasets.class_weights import (
        count_training_class_pixels,
        smoothed_class_weights,
    )

    weighting = configuration["objective"]["parameters"].get("class_weighting")
    if weighting is None or weighting["policy"] != CLASS_WEIGHT_POLICY_ID:
        return
    counts = count_training_class_pixels(
        dataset,
        training_roi_indices,
        capability=capability,
        field_profile=field_profile,
        max_json_member_bytes=max_json_member_bytes,
    )
    weighting["channels"] = [
        {
            "channel": channel["channel"],
            "role": channel["role"],
            "raw_count": count,
            "weight": weight,
            **({"path_class": channel["pathClass"]} if "pathClass" in channel else {}),
        }
        for channel, count, weight in zip(
            model_output_class_channels(model_output),
            counts,
            smoothed_class_weights(counts),
            strict=True,
        )
    ]
