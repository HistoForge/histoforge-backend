"""Read a processing image from a bounded upload on its fixed target grid.

The uploaded file is a lossless rectangle of the captured source-pyramid level.
Only IO addresses are translated into that file. Sampling coordinates, physical
image edges and normalization support remain those of the complete captured view.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from contextlib import contextmanager
from typing import Any

from histoforge.cloud._config import validate_source_artifact_limits
from histoforge.cloud.rasterization.inputs import validate_raster_transform
from histoforge.cloud.rasterization.partitioned import PartitionedRaster
from histoforge.cloud.rasterization.rasterizer import (
    EmptySourceFootprint,
    InsufficientSourceContext,
    require_real_pixel_coverage,
    required_source_pixel_rect,
)
from histoforge.cloud.sources.capture import (
    captured_image_origin,
    captured_view_geometry,
    validate_captured_source_binding,
    validate_captured_source_view,
)
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.publication import validate_source_artifact_reference
from histoforge.cloud.sources.views import source_slide_bounds


def captured_source_footprint(
    view: Any, raster_transform: Any, output_rect: Any | None = None
) -> dict[str, int] | None:
    """Return required source-level pixels, or None for an entirely white border.

    The model's full input and each processing rectangle use the same view origin.
    Bilinear neighbors are counted before clipping to an uploaded file; missing
    interior context must fail rather than become an artificial white edge.
    """

    captured_view = validate_captured_source_view(view)
    target = _require_target(captured_view, raster_transform)
    rectangle, mapping = captured_view_geometry(captured_view)
    level = captured_view["capture"]["source_level"]
    try:
        required = required_source_pixel_rect(
            mapping,
            target,
            rectangle,
            {"width": level["width"], "height": level["height"]},
            output_rect,
        )
    except EmptySourceFootprint:
        return None
    _require_coverage(required, rectangle)
    return required


def prepare_captured_source_raster(
    view: Any,
    raster_transform: Any,
    output_rect: Any,
    binding: Any,
    resolved_source: Mapping[str, Any],
    files: SourceFileStore,
    source_limits: Any,
    supported_containers: Collection[str],
) -> PartitionedRaster:
    """Validate one uploaded image before reading any target pixels.

    ``output_rect`` is a finite input or core-plus-halo rectangle on the complete
    target grid. The returned reader can split source reads to respect the decode
    limit. It cannot read outside the bound file, even if a caller later asks for
    another target rectangle. No source files are combined here.
    """

    captured_view = validate_captured_source_view(view)
    capture = captured_view["capture"]
    target = _require_target(captured_view, raster_transform)
    envelope = capture["source_rect_at_level"]
    capture_origin = captured_image_origin(capture, envelope)
    physical_bounds = source_slide_bounds(
        {
            "type": "original-wsi",
            "source": capture["source"],
            "level_zero_pixels_to_slide": capture["level_zero_pixels_to_slide"],
        }
    )
    require_real_pixel_coverage(
        physical_bounds,
        capture_origin["artifact_pixels_to_slide"],
        {"width": envelope["width"], "height": envelope["height"]},
        target,
    )
    rectangle, mapping = captured_view_geometry(captured_view)
    if captured_source_footprint(captured_view, target) != rectangle:
        raise ValueError(
            "captured source view is not the smallest bilinear sampling footprint"
        )
    bound = validate_captured_source_binding(binding, capture)
    required = captured_source_footprint(captured_view, target, output_rect)
    file_rectangle = bound["source_rect_at_level"]
    if required is not None:
        _require_coverage(required, file_rectangle)
    validate_captured_source_upload(
        capture, bound, resolved_source, supported_containers
    )
    start = resolved_source["start_request"]
    manifest = resolved_source["manifest"]
    limits = validate_source_artifact_limits(source_limits)
    metadata = files.source_level_metadata(start, manifest, 0, 0, limits)
    dimensions = {"width": file_rectangle["width"], "height": file_rectangle["height"]}
    if metadata["dimensions"] != dimensions:
        raise ValueError("uploaded image dimensions do not match its source rectangle")
    identity = {"scale_x": 1.0, "scale_y": 1.0, "translate_x": 0.0, "translate_y": 0.0}
    if metadata["selected_level_pixels_to_level_zero"] != identity:
        raise ValueError("uploaded image must use its unscaled level-zero grid")

    @contextmanager
    def open_pixels():
        with files.source_view_reader(start, manifest, 0, 0, limits) as read_view:

            def read_pixels(rectangle):
                _require_coverage(rectangle, file_rectangle)
                local = {
                    **rectangle,
                    "x": rectangle["x"] - file_rectangle["x"],
                    "y": rectangle["y"] - file_rectangle["y"],
                }
                read = read_view(local)
                if (
                    read["dimensions"] != dimensions
                    or read["selected_level_pixels_to_level_zero"] != identity
                ):
                    raise ValueError("uploaded image geometry changed after validation")
                return read["pixels"]

            yield read_pixels

    level = capture["source_level"]
    return PartitionedRaster(
        raster_transform=target,
        view_pixels_to_slide=mapping,
        source_pixel_rect=rectangle,
        source_dimensions={"width": level["width"], "height": level["height"]},
        max_complete_decode_pixels=limits["max_complete_decode_pixels"],
        open_pixels=open_pixels,
        physical_slide_bounds=physical_bounds,
    )


def validate_captured_source_upload(
    capture, binding, resolved_source, supported_containers
):
    """Check sealed metadata during binding without opening image bytes in the lane."""

    bound = validate_captured_source_binding(binding, capture)
    file_rectangle = bound["source_rect_at_level"]
    expected_fields = {"start_request", "manifest", "artifact", "absolute_expires_at"}
    if (
        not isinstance(resolved_source, Mapping)
        or set(resolved_source) != expected_fields
    ):
        raise ValueError("resolved source has missing or unknown fields")
    if bound["artifact"] != resolved_source["artifact"]:
        raise ValueError("source binding does not match the sealed source")
    start = resolved_source["start_request"]
    manifest = resolved_source["manifest"]
    validate_source_artifact_reference(start, manifest, bound["artifact"])
    expected_origin = captured_image_origin(capture, file_rectangle)
    if start["origin"] != expected_origin:
        raise ValueError("uploaded image origin does not match the captured pixels")
    if start["container"] != manifest["container"]:
        raise ValueError("source start and manifest containers differ")
    if start["container"] not in supported_containers:
        raise ValueError("source container is not supported by the selected runner")
    expected_level = {
        "series_index": 0,
        "resolution_level": 0,
        "width": file_rectangle["width"],
        "height": file_rectangle["height"],
    }
    if manifest["validated_levels"] != [expected_level]:
        raise ValueError("sealed source dimensions differ from its captured rectangle")


def _require_target(view, raster_transform):
    target = validate_raster_transform(raster_transform)
    capture = view["capture"]
    raster = target["raster"]
    if raster["plane"] != capture["plane"]:
        raise ValueError("target plane differs from the source capture")
    calibration = capture["source"]["calibration"]
    mapping = target["raster_to_slide"]
    if (
        mapping["scale_x"] != raster["target_mpp"] / calibration["mpp_x"]
        or mapping["scale_y"] != raster["target_mpp"] / calibration["mpp_y"]
    ):
        raise ValueError("target scale differs from the captured source calibration")
    return target


def _require_coverage(required, supplied):
    if any(
        required[axis] < supplied[axis]
        or required[axis] + required[extent] > supplied[axis] + supplied[extent]
        for axis, extent in (("x", "width"), ("y", "height"))
    ):
        raise InsufficientSourceContext("uploaded image omits required source pixels")
