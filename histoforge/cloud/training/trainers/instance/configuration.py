"""Resolve instance settings and bind TRAIN-only class statistics at admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.configuration import (
    bind_training_class_weights,
    resolve_training_components,
)
from histoforge.cloud.training.profiles import instance_sampler_profile

if TYPE_CHECKING:
    from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset


def prepare_instance_configuration(
    entry: TrainingCapabilityEntry,
    *,
    request: Mapping[str, Any],
    model_output: Mapping[str, Any],
    dataset: RetainedTrainingDataset,
    training_roi_indices: Sequence[int],
    api_limits: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the executable configuration before a run becomes visible."""

    configuration = resolve_instance_configuration(
        entry.trainer_profile,
        entry.capability,
        request["augmentation"],
        model_output,
        request["trainer_settings"],
    )
    bind_training_class_weights(
        configuration,
        dataset=dataset,
        training_roi_indices=training_roi_indices,
        capability=entry.capability,
        field_profile=request["field_profile"],
        model_output=model_output,
        max_json_member_bytes=api_limits["dataset"]["max_json_member_bytes"],
    )
    return configuration


def resolve_instance_configuration(
    profile: Mapping[str, Any],
    capability: Mapping[str, Any],
    augmentation_selection: Mapping[str, Any],
    model_output: Mapping[str, Any],
    trainer_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve instance schedule, class count and augmentation without dataset IO."""

    components = resolve_training_components(
        profile,
        capability,
        augmentation_selection,
        model_output,
        trainer_settings,
    )
    parameters = components["objective"]["parameters"]
    if "class_count" in parameters:
        raise ValueError("trainer profile must not bind a model class count")
    parameters["class_count"] = len(model_output_class_channels(model_output))
    if parameters["head_type"] == "flow":
        parameters["flow_loss_weighting"] = trainer_settings["flow_loss_weighting"]
    components["sampler"] = instance_sampler_profile(
        trainer_settings.get("oversample", False)
    )
    return canonical_tree(components)
