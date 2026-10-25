"""The captured QuPath interpretation required to read an original slide.

Reader family, scene, dimensions, calibration and slide mapping keep their
existing owners. Settings here contain only reader-specific choices and the
correspondence between QuPath's logical levels and native decoder resolutions.
"""

from __future__ import annotations

from typing import Any

from histoforge.canonical_json import (
    require_nonnegative_int32,
    require_positive_int32,
)
from histoforge.cloud.sources.identity import validate_scale_translate_2d

OPENSLIDE_ADAPTER = "qupath.openslide.local"
BIOFORMATS_ADAPTER = "qupath.bioformats.local"


def validate_reader_settings(value: Any, adapter: str) -> dict[str, Any]:
    """Validate closed settings; arbitrary Java options/classes are not accepted."""

    fields = {"levels"}
    if adapter == OPENSLIDE_ADAPTER:
        fields |= {"crop_x", "crop_y", "background_rgb", "tile_width", "tile_height"}
    elif adapter != BIOFORMATS_ADAPTER:
        raise ValueError(
            "original source requires the captured OpenSlide/Bio-Formats adapter"
        )
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(
            "original reader_settings fields differ from its adapter contract"
        )
    raw_levels = value["levels"]
    if not isinstance(raw_levels, list) or not raw_levels or len(raw_levels) > 64:
        raise ValueError("original reader levels must be a bounded nonempty array")
    levels = []
    for item in raw_levels:
        if not isinstance(item, dict) or set(item) != {
            "resolution_level",
            "native_resolution_index",
            "pixels_to_level_zero",
        }:
            raise ValueError("original reader level fields differ from the contract")
        levels.append(
            {
                "resolution_level": require_nonnegative_int32(
                    item["resolution_level"], "resolution_level"
                ),
                "native_resolution_index": require_nonnegative_int32(
                    item["native_resolution_index"], "native_resolution_index"
                ),
                "pixels_to_level_zero": validate_scale_translate_2d(
                    item["pixels_to_level_zero"]
                ),
            }
        )
    levels.sort(key=lambda level: level["resolution_level"])
    if [level["resolution_level"] for level in levels] != list(range(len(levels))):
        raise ValueError("original logical reader levels must be contiguous from zero")
    if [level["native_resolution_index"] for level in levels] != list(
        range(len(levels))
    ):
        raise ValueError("skipped or remapped native resolutions are not qualified")
    last = 0.0
    for index, level in enumerate(levels):
        mapping = level["pixels_to_level_zero"]
        downsample = mapping["scale_x"]
        if (
            mapping["scale_y"] != downsample
            or mapping["translate_x"] != 0
            or mapping["translate_y"] != 0
            or downsample <= last
            or (index == 0 and downsample != 1)
        ):
            raise ValueError(
                "original levels require increasing isotropic native mappings"
            )
        last = downsample
    result = {"levels": levels}
    if adapter == OPENSLIDE_ADAPTER:
        for name in ("crop_x", "crop_y"):
            result[name] = require_nonnegative_int32(value[name], name)
        for name in ("tile_width", "tile_height"):
            result[name] = require_positive_int32(value[name], name)
        background = value["background_rgb"]
        if not isinstance(background, list) or len(background) != 3:
            raise ValueError("background_rgb must contain three uint8 channels")
        result["background_rgb"] = [
            require_nonnegative_int32(channel, "background_rgb channel")
            for channel in background
        ]
        if any(channel > 255 for channel in result["background_rgb"]):
            raise ValueError("background_rgb channel exceeds uint8")
    return result
