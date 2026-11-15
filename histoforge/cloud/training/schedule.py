"""Evaluate the recorded warmup/cosine schedule without choosing its settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.cloud.training.component_schema import SCHEDULER_PROFILE_ID


def managed_scheduler_component() -> dict[str, Any]:
    """Shared warmup/cosine behavior; admission binds the selected epoch counts."""

    return {
        "profile_id": SCHEDULER_PROFILE_ID,
        "parameters": {
            "warmup_start_scale": 0.0,
            "warmup_end_scale": 1.0,
            "warmup_endpoint": True,
            "epoch_indexing": "zero-based-before-epoch",
            "group_scale_rule": "multiply-initial-learning-rate",
            "zero_rate_behavior": "adamw-step-updates-moments",
        },
    }


def managed_learning_rate_scales(
    scheduler_component: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return normalized schedule scales with the source's epoch indexing.

    Nonempty warmup includes both endpoints; the scales at its boundaries are
    unchanged. The first cosine epoch starts at scale one. The final cosine epoch
    is positive, not an added zero endpoint.
    A zero-rate epoch still performs AdamW steps and updates optimizer moments.
    """

    import numpy as np

    parameters = scheduler_component["parameters"]
    if (
        scheduler_component["profile_id"] != SCHEDULER_PROFILE_ID
        or parameters["epoch_indexing"] != "zero-based-before-epoch"
        or parameters["group_scale_rule"] != "multiply-initial-learning-rate"
        or parameters["zero_rate_behavior"] != "adamw-step-updates-moments"
        or parameters["total_epochs"]
        != parameters["warmup_epochs"] + parameters["cosine_epochs"]
        or parameters["cosine_denominator"] != parameters["cosine_epochs"]
        or parameters["warmup_epochs"] < 0
        or parameters["cosine_epochs"] < 1
    ):
        raise ValueError("resolved scheduler implementation is not supported")
    warmup = np.linspace(
        parameters["warmup_start_scale"],
        parameters["warmup_end_scale"],
        parameters["warmup_epochs"],
        endpoint=parameters["warmup_endpoint"],
    )
    cosine = 0.5 * (
        1
        + np.cos(
            np.pi
            * np.arange(parameters["cosine_epochs"])
            / parameters["cosine_denominator"]
        )
    )
    return tuple(float(value) for value in np.concatenate((warmup, cosine)))
