"""Preprocess RGB fields and plan model padding plus bottom/right batch padding.

These functions receive arrays, not task records. Concrete collators construct and
validate their own target tensors before stacking them at these shared offsets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from histoforge._vendor.histoforge_models.pathology.field_geometry import (
    centered_field_padding,
)
from histoforge.cloud.training.contract import validate_training_field_profile


def prepare_field_batch(
    pixels: Sequence[np.ndarray],
    real_pixel_masks: Sequence[np.ndarray],
    field_profile: Mapping[str, Any],
    *,
    preprocess: Callable[..., np.ndarray],
) -> tuple[list[torch.Tensor], list[tuple[int, int, int, int]], tuple[int, int]]:
    """Return unpadded normalized tensors, field offsets and final batch shape.

    The binding supplies a CPU, spawn-picklable function accepting HWC RGB and
    keyword real_pixel_mask. It must preserve shape, return finite float32, and
    leave artificial support zero. Its statistics may use only real pixels,
    including unsupervised context. No model or device is captured by the callable.
    Fixed fields must already have their historical square size. Variable fields
    receive centered minimum/alignment padding; a taller/wider peer adds only
    bottom/right padding. No input is resized or tiled and no inference-only outer
    border is added. Artificial pixels never enter normalization statistics.
    """

    if not pixels:
        raise ValueError("a training batch must contain at least one field")
    profile = validate_training_field_profile(field_profile)
    if profile["planner"] == "fixed-field-v1":
        minimum = maximum = profile["side_model_px"]
        alignment = profile["alignment_model_px"]
    else:
        minimum, maximum = profile["minSideModelPx"], profile["maxSideModelPx"]
        alignment = profile["alignmentModelPx"]

    images = []
    paddings = []
    for image, real_mask in zip(pixels, real_pixel_masks, strict=True):
        height, width, channels = image.shape
        if channels != 3:
            raise ValueError("training fields must contain three RGB channels")
        if not (0 < height <= maximum and 0 < width <= maximum) or (
            profile["planner"] == "fixed-field-v1"
            and (height != minimum or width != minimum)
        ):
            raise ValueError("training field dimensions violate the field profile")
        paddings.append(
            centered_field_padding(
                height, width, minimum_side=minimum, alignment=alignment
            )
        )
        if real_mask.dtype != np.bool_ or real_mask.shape != image.shape[:2]:
            raise ValueError(
                "real_pixel_mask must be boolean with the field's HW shape"
            )
        if not np.isfinite(image[real_mask]).all():
            raise ValueError("real training pixels must be finite")
        normalized = preprocess(image, real_pixel_mask=real_mask)
        if normalized.shape != image.shape or normalized.dtype != np.float32:
            raise ValueError("preprocessing must preserve HWC shape and return float32")
        if not np.isfinite(normalized).all() or np.any(normalized[~real_mask] != 0):
            raise ValueError(
                "preprocessing must return finite pixels and zero artificial support"
            )
        images.append(field_tensor(normalized.transpose(2, 0, 1)))
    height = max(
        image.shape[-2] + padding[2] + padding[3]
        for image, padding in zip(images, paddings)
    )
    width = max(
        image.shape[-1] + padding[0] + padding[1]
        for image, padding in zip(images, paddings)
    )
    return images, paddings, (height, width)


def field_tensor(value: np.ndarray) -> torch.Tensor:
    """Wrap a contiguous field with a batch axis, including reflected arrays."""

    return torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0)


def pad_field_tensors(
    values: Sequence[torch.Tensor],
    paddings: Sequence[tuple[int, int, int, int]],
    height: int,
    width: int,
) -> torch.Tensor:
    """Copy each field at its model offset; all remaining batch support is zero."""

    result = values[0].new_zeros((len(values), *values[0].shape[1:-2], height, width))
    for index, (value, padding) in enumerate(zip(values, paddings, strict=True)):
        left, _, top, _ = padding
        result[
            index, ..., top : top + value.shape[-2], left : left + value.shape[-1]
        ] = value[0]
    return result
