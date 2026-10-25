"""Validate source identity and read RGB on the model's exact pixel grid.

Original slide files and prepared images use the same bounded source reader.
Reading in pieces does not change the model's pixel grid or split prediction
into separate calls. This module does not select a source file or run a model.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from histoforge.cloud._config import validate_source_artifact_limits
from histoforge.cloud.rasterization.captured_source import (
    prepare_captured_source_raster,
)
from histoforge.cloud.rasterization.inputs import validate_model_raster_input
from histoforge.cloud.rasterization.partitioned import PartitionedRaster
from histoforge.cloud.rasterization.rasterizer import (
    RasterizedImage,
    require_real_pixel_coverage,
    required_source_pixel_rect,
)
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.views import (
    derive_derived_view_transform,
    derive_original_view_transform,
    source_slide_bounds,
)


def prepare_partitioned_model_raster(
    raster_input: Any,
    resolved_source: Mapping[str, Any],
    files: SourceFileStore,
    source_limits: Any,
    supported_containers: Collection[str],
    *,
    source_binding: Any = None,
    output_rect: Any = None,
) -> PartitionedRaster:
    """Check source identity and geometry without decoding pixels; prepare reads."""

    model_input = validate_model_raster_input(raster_input)
    if "capture" in model_input["source_view"]:
        target = model_input["frame"]["transform"]
        rectangle = (
            output_rect
            if output_rect is not None
            else {
                "x": 0,
                "y": 0,
                "width": target["raster"]["width"],
                "height": target["raster"]["height"],
            }
        )
        return prepare_captured_source_raster(
            model_input["source_view"],
            target,
            rectangle,
            source_binding,
            resolved_source,
            files,
            source_limits,
            supported_containers,
        )
    limits = validate_source_artifact_limits(source_limits)
    start, manifest = _require_source_binding(
        model_input, resolved_source, supported_containers
    )
    view = model_input["source_view"]
    metadata = files.source_level_metadata(
        start,
        manifest,
        view["series_index"],
        view["resolution_level"],
        limits,
    )
    _require_view_geometry(model_input, start, metadata)

    @contextmanager
    def open_pixels():
        with files.source_view_reader(
            start, manifest, view["series_index"], view["resolution_level"], limits
        ) as read_view:

            def read_pixels(required: Mapping[str, int]) -> np.ndarray:
                read = read_view(required)
                if (
                    read["dimensions"] != metadata["dimensions"]
                    or read["selected_level_pixels_to_level_zero"]
                    != metadata["selected_level_pixels_to_level_zero"]
                ):
                    raise ValueError("source geometry changed after source validation")
                return read["pixels"]

            yield read_pixels

    return PartitionedRaster(
        raster_transform=model_input["frame"]["transform"],
        view_pixels_to_slide=view["view_pixels_to_slide"],
        source_pixel_rect=view["pixel_rect"],
        source_dimensions=metadata["dimensions"],
        max_complete_decode_pixels=limits["max_complete_decode_pixels"],
        open_pixels=open_pixels,
        physical_slide_bounds=source_slide_bounds(start["origin"]),
    )


def rasterize_model_input(
    raster_input: Any,
    resolved_source: Mapping[str, Any],
    files: SourceFileStore,
    source_limits: Any,
    supported_containers: Collection[str],
    source_binding: Any = None,
) -> RasterizedImage:
    """Assemble one complete model image and its real-pixel mask from bounded reads."""

    prepared = prepare_partitioned_model_raster(
        raster_input,
        resolved_source,
        files,
        source_limits,
        supported_containers,
        source_binding=source_binding,
    )
    raster = prepared.raster_transform["raster"]
    return prepared.read(
        {"x": 0, "y": 0, "width": raster["width"], "height": raster["height"]}
    )


def _require_source_binding(
    model_input: Mapping[str, Any],
    resolved_source: Mapping[str, Any],
    supported_containers: Collection[str],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_fields = {
        "start_request",
        "manifest",
        "artifact",
        "absolute_expires_at",
    }
    if (
        not isinstance(resolved_source, Mapping)
        or set(resolved_source) != expected_fields
    ):
        raise ValueError("resolved source has missing or unknown fields")

    start = resolved_source["start_request"]
    manifest = resolved_source["manifest"]
    artifact = resolved_source["artifact"]
    view = model_input["source_view"]
    frame = model_input["frame"]
    if view["artifact"] != artifact:
        raise ValueError("source view artifact does not match sealed source state")
    if frame["source_revision"]["pixel_source"] != start["origin"]["source"]:
        raise ValueError("raster source revision does not match source artifact origin")
    if start["container"] != manifest["container"]:
        raise ValueError("source start and manifest containers differ")
    if start["container"] not in _container_set(supported_containers):
        raise ValueError("source container is not supported by the selected runner")
    if view["plane"] != {"z": 0, "t": 0}:
        raise ValueError("source artifact schema 1 supports only plane z=0, t=0")
    return start, manifest


def _require_view_geometry(
    model_input: Mapping[str, Any],
    start: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    view = model_input["source_view"]
    origin = start["origin"]
    if origin["type"] == "original-wsi":
        if view["series_index"] != origin["pyramid_series_index"]:
            raise ValueError("source view uses the wrong original pyramid series")
    else:
        if view["series_index"] != 0 or view["resolution_level"] != 0:
            raise ValueError("derived source view must use series zero and level zero")
        if view["plane"] != origin["plane"]:
            raise ValueError("source view plane does not match derived source origin")
        require_real_pixel_coverage(
            source_slide_bounds(origin),
            origin["artifact_pixels_to_slide"],
            metadata["dimensions"],
            model_input["frame"]["transform"],
        )
    expected = _derive_view_transform(start, metadata, view["pixel_rect"])
    if view["view_pixels_to_slide"] != expected:
        raise ValueError(
            "source view transform does not match validated source geometry"
        )
    required = required_source_pixel_rect(
        expected,
        model_input["frame"]["transform"],
        view["pixel_rect"],
        metadata["dimensions"],
    )
    if view["pixel_rect"] != required:
        raise ValueError(
            "source view pixel_rect is not the smallest bilinear sampling footprint"
        )


def _derive_view_transform(
    start: Mapping[str, Any], metadata: Mapping[str, Any], pixel_rect: Any
) -> dict[str, float]:
    origin = start["origin"]
    if origin["type"] == "original-wsi":
        return derive_original_view_transform(
            origin["level_zero_pixels_to_slide"],
            metadata["selected_level_pixels_to_level_zero"],
            metadata["dimensions"],
            pixel_rect,
        )
    return derive_derived_view_transform(
        origin["artifact_pixels_to_slide"],
        metadata["dimensions"],
        pixel_rect,
    )


def _container_set(values: Collection[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("supported_containers must be a collection")
    result = frozenset(values)
    if not result or any(not isinstance(value, str) for value in result):
        raise ValueError("supported_containers must contain strings")
    return result
