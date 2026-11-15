"""Execute the recorded per-field augmentation without consulting defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from histoforge.cloud.training.fields.augmentation import (
    apply_rgb_augmentation,
    sample_spatial_transform,
    training_augmentation_operators,
)

if TYPE_CHECKING:
    from histoforge.cloud.training.trainers.instance.field_reader import TrainingField


def apply_training_augmentation(
    field: TrainingField,
    component: Mapping[str, Any],
    *,
    training_seed: int,
    epoch_index: int,
    sample_index: int,
) -> TrainingField:
    """Execute the resolved run component without consulting current presets.

    ``sample_index`` is the ordinal in the sampler's unchanged epoch sequence,
    never a batch-local or worker-local counter. Each operator gets its own PCG64
    generator from the full SHA-256 integer of the canonical seed record below.
    Disabling another operator cannot consume or shift this operator's draws.
    Quarter-turn probabilities describe outcomes 0, 1, 2, 3 counterclockwise;
    flips draw horizontal first and vertical second, independently.
    """

    from histoforge.cloud.training.trainers.instance.spatial_augmentation import (
        transform_training_field,
    )

    result = field
    for settings, rng in training_augmentation_operators(
        component,
        training_seed=training_seed,
        epoch_index=epoch_index,
        sample_index=sample_index,
    ):
        spatial = sample_spatial_transform(settings, rng)
        if spatial is not None:
            result = transform_training_field(result, **spatial)
        else:
            # Appearance operators change raw real RGB only. Detach all target
            # arrays before the first appearance operation if no earlier spatial
            # operation has already done so.
            if result is field:
                result = transform_training_field(result)
            result = replace(
                result,
                pixels=apply_rgb_augmentation(
                    result.pixels, result.labels.real_pixel_mask, settings, rng
                ),
            )
    return transform_training_field(field) if result is field else result
