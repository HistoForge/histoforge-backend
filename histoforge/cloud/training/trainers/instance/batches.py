"""Adapt selected, augmented fields to the managed trainer's CPU batch contract.

Sampling and augmentation happen upstream. A normal DataLoader owns batch grouping
and worker scheduling. This adapter normalizes each field independently, applies
its minimum/alignment padding, and pads to the largest resulting shape in a batch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedSegmentationBatch,
)
from histoforge.canonical_json import require_nonnegative_int32
from histoforge.cloud.training.fields.batches import (
    field_tensor,
    pad_field_tensors,
    prepare_field_batch,
)
from histoforge.cloud.training.trainers.instance.field_reader import TrainingField

if TYPE_CHECKING:
    import numpy as np


def collate_training_fields(
    fields: Sequence[TrainingField],
    *,
    field_profile: Mapping[str, Any],
    class_count: int,
    preprocess: Callable[..., np.ndarray],
) -> ManagedSegmentationBatch:
    """Normalize fields, center model padding, then pad the batch bottom/right.

    Fields already have their sampled size, complete-object targets, geometric
    augmentation, and masks. RGB may be uint8 or appearance-augmented floating
    point. No field is resized, tiled, reordered, dropped, or given new context.
    Variable fields below their minimum or off alignment receive centered padding;
    fixed fields must already have their historical square size. Each field keeps
    its own centered offset when further batch padding is appended bottom/right.
    One destination allocation per tensor combines both padding steps without
    changing their meaning. The caller moves the returned CPU tensors to its chosen
    training device; this function never selects a GPU or changes batch size.
    """

    class_count = require_nonnegative_int32(class_count, "class_count")
    images, paddings, (height, width) = prepare_field_batch(
        [field.pixels for field in fields],
        [field.labels.real_pixel_mask for field in fields],
        field_profile,
        preprocess=preprocess,
    )
    samples = []
    for field, image in zip(fields, images):
        labels = field.labels
        targets = labels.targets
        sample = ManagedSegmentationBatch(
            images=image,
            directional_targets=field_tensor(targets.directional),
            foreground_targets=field_tensor(targets.foreground),
            supervision_mask=field_tensor(labels.supervision_mask),
            real_pixel_mask=field_tensor(labels.real_pixel_mask),
            class_targets=(
                None if targets.classes is None else field_tensor(targets.classes)
            ),
            instance_flow_weights=(
                None
                if targets.instance_flow_weights is None
                else field_tensor(targets.instance_flow_weights)
            ),
        )
        sample.validate(class_count)
        samples.append(sample)
    names = [
        "images",
        "directional_targets",
        "foreground_targets",
        "supervision_mask",
        "real_pixel_mask",
    ]
    if class_count:
        names.append("class_targets")
    weighted = [sample.instance_flow_weights is not None for sample in samples]
    if any(weighted):
        if not all(weighted):
            raise ValueError(
                "one batch cannot mix fields with and without flow weights"
            )
        names.append("instance_flow_weights")
    return ManagedSegmentationBatch(
        **{
            name: pad_field_tensors(
                [getattr(sample, name) for sample in samples], paddings, height, width
            )
            for name in names
        }
    )
