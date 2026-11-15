"""Recorded augmentation decisions and real-pixel RGB operators.

This owner knows arrays, operator parameters and seeds, not instance or dense
targets. Concrete trainers apply each spatial decision to their own field type.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

from histoforge.canonical_json import canonical_digest
from histoforge.cloud.training.augmentation import (
    AUGMENTATION_APPEARANCE_NUMERIC_POLICY,
    AUGMENTATION_INPUT_NUMERIC_DOMAIN,
    AUGMENTATION_PROFILE_ID,
    AUGMENTATION_SEED_DERIVATION,
)

if TYPE_CHECKING:
    import numpy as np


def training_augmentation_operators(
    component: Mapping[str, Any],
    *,
    training_seed: int,
    epoch_index: int,
    sample_index: int,
) -> Iterator[tuple[Mapping[str, Any], np.random.Generator]]:
    """Yield each recorded operator with its independent PCG64 generator.

    ``sample_index`` is the epoch ordinal, never a batch or worker counter.
    Disabling another operator cannot consume or shift this operator's draws.
    """

    import numpy as np

    parameters = component["parameters"]
    if (
        component["profile_id"] != AUGMENTATION_PROFILE_ID
        or parameters["seed_derivation"] != AUGMENTATION_SEED_DERIVATION
        or parameters["input_numeric_domain"] != AUGMENTATION_INPUT_NUMERIC_DOMAIN
        or parameters["appearance_numeric_policy"]
        != AUGMENTATION_APPEARANCE_NUMERIC_POLICY
    ):
        raise ValueError("resolved augmentation implementation is not supported")
    for operator in parameters["enabled_operators"]:
        operator_id = operator["operator_id"]
        seed = int(
            canonical_digest(
                {
                    "domain": AUGMENTATION_SEED_DERIVATION,
                    "training_seed": training_seed,
                    "epoch_index": epoch_index,
                    "sample_index": sample_index,
                    "operator_id": operator_id,
                }
            ),
            16,
        )
        yield (
            parameters["operator_parameters"][operator_id],
            np.random.Generator(np.random.PCG64(seed)),
        )


def sample_spatial_transform(
    settings: Mapping[str, Any], rng: np.random.Generator
) -> dict[str, int | bool] | None:
    """Resolve quarter turns or flips; return None for an appearance operator.

    Turns are counterclockwise outcomes 0, 1, 2, 3. Flips draw horizontal first,
    then vertical. The task adapter applies this exact decision to every target.
    """

    if settings["implementation"] == "quarter-turn-v1":
        return {
            "quarter_turns": int(
                rng.choice(4, p=settings["quarter_turn_probabilities"])
            )
        }
    if settings["implementation"] == "axis-flips-v1":
        return {
            "flip_horizontal": rng.random() < settings["horizontal_probability"],
            "flip_vertical": rng.random() < settings["vertical_probability"],
        }
    return None


def transform_spatial_array(
    value: np.ndarray,
    *,
    axes: tuple[int, int] = (0, 1),
    quarter_turns: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    """Permute spatial coordinates without interpolation and own the result memory."""

    import numpy as np

    value = np.rot90(value, quarter_turns % 4, axes=axes)
    if flip_horizontal:
        value = np.flip(value, axis=axes[1])
    if flip_vertical:
        value = np.flip(value, axis=axes[0])
    return value.copy()


def apply_rgb_augmentation(
    pixels: np.ndarray,
    real_pixel_mask: np.ndarray,
    settings: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Delegate real-pixel RGB math to the exact vendored scientific functions."""

    from histoforge._vendor.histoforge_models.pathology.managed_training import (
        augment_blur,
        augment_brightness_contrast,
        augment_hed,
        augment_noise,
    )

    implementation = settings["implementation"]
    common = {
        "real_pixel_mask": real_pixel_mask,
        "rng": rng,
        "probability": settings["probability"],
    }
    if implementation == "masked-hed-v1":
        return augment_hed(
            pixels,
            **common,
            sigma_range=settings["sigma_range"],
            mean_intensity_range=settings["mean_intensity_range"],
        )
    if implementation == "masked-brightness-contrast-v1":
        return augment_brightness_contrast(
            pixels,
            **common,
            brightness_range=settings["brightness_range"],
            contrast_range=settings["contrast_range"],
        )
    if implementation == "masked-gaussian-blur-v1":
        return augment_blur(pixels, **common, sigma_range=settings["sigma_range"])
    if implementation == "masked-gaussian-noise-v1":
        return augment_noise(pixels, **common, sigma=settings["sigma"])
    raise ValueError("augmentation operator implementation is not supported")
