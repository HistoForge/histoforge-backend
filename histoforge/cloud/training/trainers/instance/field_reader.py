"""Read raw RGB and complete-object targets for one planned training field.

The run owner first verifies the retained dataset and holds its existing active
run reference. This adapter only reads that immutable package. It does not
select fields, normalize pixels, augment samples, or change source levels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from histoforge._vendor.histoforge_models.pathology.managed_training import HeadType
from histoforge.cloud.rasterization.inputs import validate_raster_transform
from histoforge.cloud.training.datasets.field_reader import (
    read_training_field_inputs,
)
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.trainers.instance.field_targets import (
    TrainingFieldTargets,
    build_training_field_targets,
)


@dataclass(frozen=True)
class TrainingField:
    """Unnormalized HWC RGB and aligned targets with separate masks.

    The reader returns uint8. Appearance augmentation may replace RGB with
    floating-point pixels before the batch adapter normalizes the field.
    """

    pixels: np.ndarray
    labels: TrainingFieldTargets


def read_training_field(
    dataset: RetainedTrainingDataset,
    roi_index: int,
    *,
    raster_transform: Mapping[str, Any],
    output_rect: Mapping[str, int],
    head_type: HeadType,
    source_limits: Mapping[str, int],
    max_json_member_bytes: int,
    flow_device: torch.device | None = None,
) -> TrainingField:
    """Pair a bounded image read and targets on the exact same absolute grid.

    ``dataset`` comes from ``TrainingDatasetFiles.read_dataset`` after the run
    is claimed. The caller supplies the planned grid and field. All file handles
    close before target generation; no Volume gate is held during flow/HoVer
    construction, and no open reader is retained between samples.
    """

    target = validate_raster_transform(raster_transform)
    pixels, roi_geometry, objects, physical_bounds = read_training_field_inputs(
        dataset,
        roi_index,
        target,
        output_rect,
        source_limits,
        max_json_member_bytes,
    )
    task = dataset.manifest["task"]
    labels = build_training_field_targets(
        roi_geometry=roi_geometry,
        objects=objects,
        model_output=task["model_output"],
        head_type=head_type,
        raster_transform=target,
        output_rect=output_rect,
        physical_slide_bounds=physical_bounds,
        flow_device=flow_device,
    )
    return TrainingField(pixels, labels)
