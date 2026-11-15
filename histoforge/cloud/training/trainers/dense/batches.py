"""Normalize and pad dense fields into their own scientific batch contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedDenseBatch,
)
from histoforge.cloud.training.fields.batches import (
    field_tensor,
    pad_field_tensors,
    prepare_field_batch,
)

if TYPE_CHECKING:
    import numpy as np

    from histoforge.cloud.training.trainers.dense.fields import DenseTrainingField


def collate_dense_fields(
    fields: Sequence[DenseTrainingField],
    *,
    field_profile: Mapping[str, Any],
    class_count: int,
    preprocess: Callable[..., np.ndarray],
) -> ManagedDenseBatch:
    """Preserve per-field normalization and centered padding before batch padding."""

    images, paddings, (height, width) = prepare_field_batch(
        [field.pixels for field in fields],
        [field.real_pixel_mask for field in fields],
        field_profile,
        preprocess=preprocess,
    )
    samples = []
    for field, image in zip(fields, images, strict=True):
        batch = ManagedDenseBatch(
            image,
            field_tensor(field.class_targets),
            field_tensor(field.supervision_mask),
            field_tensor(field.real_pixel_mask),
        )
        batch.validate(class_count)
        samples.append(batch)
    return ManagedDenseBatch(
        **{
            name: pad_field_tensors(
                [getattr(sample, name) for sample in samples],
                paddings,
                height,
                width,
            )
            for name in (
                "images",
                "class_targets",
                "supervision_mask",
                "real_pixel_mask",
            )
        }
    )
