"""Resolve dense settings and bind TRAIN pixel-frequency evidence at admission."""

from histoforge.cloud.modules.schema import model_output_class_channels
from histoforge.cloud.training.configuration import (
    bind_training_class_weights,
    resolve_training_components,
)


def prepare_dense_configuration(
    entry, *, request, model_output, dataset, training_roi_indices, api_limits
):
    """No instance size, flow target, foreground task or validation-derived weight."""

    configuration = resolve_training_components(
        entry.trainer_profile,
        entry.capability,
        request["augmentation"],
        model_output,
        request["trainer_settings"],
    )
    parameters = configuration["objective"]["parameters"]
    if "class_count" in parameters:
        raise ValueError("trainer profile must not bind a model class count")
    parameters["class_count"] = len(model_output_class_channels(model_output))
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
