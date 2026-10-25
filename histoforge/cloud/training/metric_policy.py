"""Metric declarations shared by concrete training policies."""

from typing import Any

TRAINING_LOSS_SERIES_ID = "training-loss.v1"


def training_loss_series() -> dict[str, Any]:
    """The engine reports the field-equal mean of its active objective."""

    return {
        "series_id": TRAINING_LOSS_SERIES_ID,
        "display_name": "Training loss",
        "partition": "training",
        "value_domain": "real",
        "definition": (
            "Mean resolved training objective over all actual model fields "
            "in this epoch, evaluated during optimization with the current "
            "model and learned task weights."
        ),
    }
