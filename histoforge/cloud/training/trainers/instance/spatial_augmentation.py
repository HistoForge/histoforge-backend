"""Rotate and reflect prepared fields without interpolation or new image support.

The trainer's resolved augmentation profile owns operator selection and random
draws. This transform receives those choices; it does not sample them, regenerate
targets, normalize RGB, or change the training ROI's supervision.
"""

from __future__ import annotations

import numpy as np

from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedSegmentationTargets,
)
from histoforge.cloud.training.fields.augmentation import transform_spatial_array
from histoforge.cloud.training.trainers.instance.field_reader import TrainingField
from histoforge.cloud.training.trainers.instance.field_targets import (
    TrainingFieldTargets,
)


def transform_training_field(
    field: TrainingField,
    *,
    quarter_turns: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> TrainingField:
    """Apply counterclockwise quarter turns, then horizontal and vertical flips.

    RGB is HWC; scalar targets and masks are HW. Both supported heads store
    directional targets in vertical, horizontal order: flow [dY, dX] and HoVer
    [V, H]. A quarter turn maps these components to [-horizontal, vertical]; a
    reflection reverses the component along that image axis.

    Targets were derived from complete instances before the field was cropped.
    Only their coordinates and directions change here. All returned arrays own
    their memory, including the identity transform, so later appearance changes
    cannot modify the source field. Dimensions may be unaligned at this stage.
    """

    turns = quarter_turns % 4

    def spatial(value: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
        """Permute the same two image axes and return an independent array."""

        return transform_spatial_array(
            value,
            axes=axes,
            quarter_turns=turns,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )

    labels = field.labels
    targets = labels.targets
    components = targets.directional[::-1] if turns % 2 else targets.directional
    directional = spatial(components, (1, 2))
    if turns in (1, 2):
        directional[0] *= -1
    if turns in (2, 3):
        directional[1] *= -1
    if flip_vertical:
        directional[0] *= -1
    if flip_horizontal:
        directional[1] *= -1

    return TrainingField(
        spatial(field.pixels, (0, 1)),
        TrainingFieldTargets(
            ManagedSegmentationTargets(
                directional=directional,
                foreground=spatial(targets.foreground, (0, 1)),
                classes=(
                    None
                    if targets.classes is None
                    else spatial(targets.classes, (0, 1))
                ),
                instance_flow_weights=(
                    None
                    if targets.instance_flow_weights is None
                    else spatial(targets.instance_flow_weights, (0, 1))
                ),
            ),
            spatial(labels.supervision_mask, (0, 1)),
            spatial(labels.real_pixel_mask, (0, 1)),
        ),
    )
