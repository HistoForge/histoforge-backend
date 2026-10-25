"""Read and augment dense class targets on the same grid as raw RGB."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from histoforge.cloud.modules.schema import model_output_target_channel_by_class
from histoforge.cloud.training.datasets.class_raster import rasterize_class_targets
from histoforge.cloud.training.datasets.field_reader import read_training_field_inputs
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_training_geometry,
    training_geometry_on_model_grid,
    training_real_pixel_mask,
)
from histoforge.cloud.training.fields.augmentation import (
    apply_rgb_augmentation,
    sample_spatial_transform,
    training_augmentation_operators,
    transform_spatial_array,
)

if TYPE_CHECKING:
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset


@dataclass(frozen=True)
class DenseTrainingField:
    """Unnormalized HWC RGB, neural class indices and independent support masks."""

    pixels: np.ndarray
    class_targets: np.ndarray
    supervision_mask: np.ndarray
    real_pixel_mask: np.ndarray


def read_dense_field(
    dataset: RetainedTrainingDataset,
    roi_index: int,
    target: Mapping[str, Any],
    rectangle: Mapping[str, int],
    source_limits: Mapping[str, int],
    max_json_member_bytes: int,
) -> DenseTrainingField:
    """Read bounded pixels; release IO handles before rasterizing annotations."""

    pixels, roi, objects, bounds = read_training_field_inputs(
        dataset,
        roi_index,
        target,
        rectangle,
        source_limits,
        max_json_member_bytes,
    )
    supervision = rasterize_training_geometry(roi, target, rectangle)
    real = training_real_pixel_mask(bounds, target, rectangle)
    support = supervision & real
    if not support.any():
        raise ValueError("training field has no supervised real pixels")
    channels = model_output_target_channel_by_class(
        dataset.manifest["task"]["model_output"]
    )
    labels = rasterize_class_targets(
        (
            (
                training_geometry_on_model_grid(item["geometry"], target),
                channels[item["properties"]["path_class"]],
            )
            for item in objects
        ),
        rectangle,
        support,
        allow_same_class_overlap=True,
    )
    return DenseTrainingField(pixels, labels, supervision, real)


def pad_dense_field(
    field: DenseTrainingField, padding: tuple[int, int, int, int]
) -> DenseTrainingField:
    """Apply fixed-field offsets, keeping all added support false."""

    left, right, top, bottom = padding
    spatial = ((top, bottom), (left, right))
    return DenseTrainingField(
        np.pad(field.pixels, (*spatial, (0, 0))),
        np.pad(field.class_targets, spatial),
        np.pad(field.supervision_mask, spatial),
        np.pad(field.real_pixel_mask, spatial),
    )


def augment_dense_field(
    field: DenseTrainingField,
    augmentation: Mapping[str, Any],
    *,
    training_seed: int,
    epoch_index: int,
    sample_index: int,
) -> DenseTrainingField:
    """Apply the same spatial draw to pixels, class indices and both masks."""

    for settings, rng in training_augmentation_operators(
        augmentation,
        training_seed=training_seed,
        epoch_index=epoch_index,
        sample_index=sample_index,
    ):
        spatial = sample_spatial_transform(settings, rng)
        if spatial is not None:
            field = DenseTrainingField(
                **{
                    name: transform_spatial_array(getattr(field, name), **spatial)
                    for name in (
                        "pixels",
                        "class_targets",
                        "supervision_mask",
                        "real_pixel_mask",
                    )
                }
            )
        else:
            field = replace(
                field,
                pixels=apply_rgb_augmentation(
                    field.pixels,
                    field.real_pixel_mask,
                    settings,
                    rng,
                ),
            )
    return field
