"""Read raw RGB and captured geometry on an already planned training grid.

All source handles and the Volume gate close before the caller constructs task
targets, runs inference or evaluates metrics. No model or target tensors are
constructed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import numpy as np

from histoforge.cloud._config import validate_source_artifact_limits
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.rasterization.partitioned import PartitionedRaster
from histoforge.cloud.sources.views import (
    derive_derived_view_transform,
    derive_original_view_transform,
    source_slide_bounds,
)
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.datasets.io import (
    open_training_artifact,
    read_training_objects,
    read_training_roi_geometry,
)


@mounted_artifact_volume_operation
def read_training_field_inputs(
    dataset: RetainedTrainingDataset,
    roi_index: int,
    target: Mapping[str, Any],
    output_rect: Mapping[str, int],
    source_limits: Mapping[str, int],
    max_json_member_bytes: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], tuple[float, ...]]:
    """Read pixels and complete annotations without interpreting their task.

    Training and validation supply an already-validated target grid and an
    absolute output rectangle. The run owner holds the retained dataset's active
    reference; this reader opens no mutable project or live source module.
    """

    limits = validate_source_artifact_limits(source_limits)
    manifest = dataset.manifest
    task = manifest["task"]
    roi = manifest["rois"][roi_index]
    envelope = roi["source_envelope"]
    roi_geometry = read_training_roi_geometry(
        dataset.root, roi_index, max_json_member_bytes
    )
    objects = read_training_objects(
        dataset.root,
        roi_index,
        roi_geometry,
        task["annotation_class_map"],
        max_json_member_bytes,
    )
    with open_training_artifact(
        dataset.root,
        manifest,
        envelope["artifact_index"],
        max_json_member_bytes,
        limits,
    ) as (origin, reader):
        mpp = task["model_microns_per_pixel"]
        calibration = origin["source"]["calibration"]
        if (
            target["raster"]["target_mpp"] != mpp
            or target["raster_to_slide"]["scale_x"] != mpp / calibration["mpp_x"]
            or target["raster_to_slide"]["scale_y"] != mpp / calibration["mpp_y"]
        ):
            raise ValueError("training field grid disagrees with dataset calibration")
        series = envelope["series_index"]
        resolution = envelope["resolution_level"]
        mapping = reader.selected_level_pixels_to_level_zero(series, resolution)
        level = next(
            value
            for value in reader.validated_levels
            if value["series_index"] == series
            and value["resolution_level"] == resolution
        )
        dimensions = {"width": level["width"], "height": level["height"]}
        rectangle = envelope["pixel_rect"]
        if origin["type"] == "original-wsi":
            transform = derive_original_view_transform(
                origin["level_zero_pixels_to_slide"], mapping, dimensions, rectangle
            )
        else:
            transform = derive_derived_view_transform(
                origin["artifact_pixels_to_slide"], dimensions, rectangle
            )
        raster = PartitionedRaster(
            raster_transform=target,
            view_pixels_to_slide=transform,
            source_pixel_rect=rectangle,
            source_dimensions=dimensions,
            max_complete_decode_pixels=limits["max_complete_decode_pixels"],
            open_pixels=lambda: nullcontext(
                lambda required: reader.read_rgb_view(series, resolution, required)
            ),
            physical_slide_bounds=source_slide_bounds(origin),
        )
        pixels = raster.read(output_rect).pixels
    return pixels, roi_geometry, objects, source_slide_bounds(origin)
