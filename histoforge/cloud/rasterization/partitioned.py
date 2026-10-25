"""Read a fixed target grid through bounded source-rectangle callbacks.

Inference and training share this partition algorithm and the pinned rasterizer.
Callers own source identity, handle lifetime, and the chosen target geometry.
The declared source rectangle is an allowed envelope, not necessarily the
smallest footprint for a requested field.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np

from histoforge.cloud.rasterization.rasterizer import (
    EmptySourceFootprint,
    InsufficientSourceContext,
    RasterizedImage,
    rasterize_srgb_uint8,
    real_pixel_mask,
    required_source_pixel_rect,
)
from histoforge.cloud.sources.identity import validate_int_rect


@dataclass(frozen=True)
class PartitionedRaster:
    """Validated logical input whose absolute target rectangles are read on demand."""

    raster_transform: Mapping[str, Any]
    view_pixels_to_slide: Mapping[str, float]
    source_pixel_rect: Mapping[str, int]
    source_dimensions: Mapping[str, int]
    max_complete_decode_pixels: int
    open_pixels: Callable[
        [], AbstractContextManager[Callable[[Mapping[str, int]], np.ndarray]]
    ]
    physical_slide_bounds: tuple[float, float, float, float]

    def read(self, output_rect: Any) -> RasterizedImage:
        """Read raw RGB and physical support on one absolute target rectangle."""

        rectangle = _target_output_rect(
            output_rect,
            self.raster_transform["raster"],
        )
        output = np.empty((rectangle["height"], rectangle["width"], 3), dtype=np.uint8)
        with ExitStack() as readers:
            read_pixels = None
            pending = [(rectangle, 0, 0)]
            while pending:
                block, local_x, local_y = pending.pop()
                destination = output[
                    local_y : local_y + block["height"],
                    local_x : local_x + block["width"],
                ]
                try:
                    required = required_source_pixel_rect(
                        self.view_pixels_to_slide,
                        self.raster_transform,
                        self.source_pixel_rect,
                        self.source_dimensions,
                        block,
                    )
                except EmptySourceFootprint:
                    destination.fill(255)
                    continue
                if not (
                    self.source_pixel_rect["x"] <= required["x"]
                    and self.source_pixel_rect["y"] <= required["y"]
                    and required["x"] + required["width"]
                    <= self.source_pixel_rect["x"] + self.source_pixel_rect["width"]
                    and required["y"] + required["height"]
                    <= self.source_pixel_rect["y"] + self.source_pixel_rect["height"]
                ):
                    raise InsufficientSourceContext(
                        "source envelope omits required in-image pixels"
                    )
                if (
                    required["width"] * required["height"]
                    > self.max_complete_decode_pixels
                ):
                    first, second = _split_target_output_rect(block)
                    if first is None or second is None:
                        raise ValueError(
                            "one target pixel source footprint exceeds "
                            "complete-decode limits"
                        )
                    if first["x"] != block["x"]:
                        raise RuntimeError(
                            "partition split changed the first block origin"
                        )
                    if first["y"] != block["y"]:
                        raise RuntimeError(
                            "partition split changed the first block origin"
                        )
                    second_local_x = local_x + second["x"] - block["x"]
                    second_local_y = local_y + second["y"] - block["y"]
                    pending.append((second, second_local_x, second_local_y))
                    pending.append((first, local_x, local_y))
                    continue
                # White-border blocks and rejected source footprints do not need
                # to decode pixels or retain a mounted reader.
                if read_pixels is None:
                    read_pixels = readers.enter_context(self.open_pixels())
                destination[:] = self._read_block(block, required, read_pixels)
        return RasterizedImage(
            output,
            real_pixel_mask(
                self.physical_slide_bounds, self.raster_transform, rectangle
            ),
        )

    def _read_block(
        self,
        output_rect: Mapping[str, int],
        required: Mapping[str, int],
        read_pixels: Callable[[Mapping[str, int]], np.ndarray],
    ) -> np.ndarray:
        """Read and rasterize one source-cap-admissible absolute block."""

        submitted = self.source_pixel_rect
        dimensions = self.source_dimensions
        physical_bounds = {
            "min_x": -submitted["x"],
            "min_y": -submitted["y"],
            "max_x": dimensions["width"] - submitted["x"],
            "max_y": dimensions["height"] - submitted["y"],
        }
        decoded_rect_in_view = {
            "x": required["x"] - submitted["x"],
            "y": required["y"] - submitted["y"],
            "width": required["width"],
            "height": required["height"],
        }
        return rasterize_srgb_uint8(
            read_pixels(required),
            self.view_pixels_to_slide,
            self.raster_transform,
            physical_bounds,
            output_rect,
            decoded_rect_in_view=decoded_rect_in_view,
        )


def _target_output_rect(value: Any, raster: Mapping[str, int]) -> dict[str, int]:
    rectangle = validate_int_rect(value)
    if (
        rectangle["x"] + rectangle["width"] > raster["width"]
        or rectangle["y"] + rectangle["height"] > raster["height"]
    ):
        raise ValueError("output_rect lies outside the complete target raster")
    return rectangle


def _split_target_output_rect(
    rectangle: Mapping[str, int],
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    if rectangle["width"] >= rectangle["height"] and rectangle["width"] > 1:
        first_width = rectangle["width"] // 2
        return (
            {**rectangle, "width": first_width},
            {
                **rectangle,
                "x": rectangle["x"] + first_width,
                "width": rectangle["width"] - first_width,
            },
        )
    if rectangle["height"] > 1:
        first_height = rectangle["height"] // 2
        return (
            {**rectangle, "height": first_height},
            {
                **rectangle,
                "y": rectangle["y"] + first_height,
                "height": rectangle["height"] - first_height,
            },
        )
    return None, None
