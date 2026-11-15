"""Exact source-level view transforms shared by planning and admission.

These functions derive the mapping carried by ``SourceView`` from trusted
source geometry and container metadata. They perform no image IO and do
not choose a source level or representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import require_positive_int32
from histoforge.cloud.sources.identity import (
    validate_int_rect,
    validate_scale_translate_2d,
)

_DIMENSION_FIELDS = {"width", "height"}


def derive_original_view_transform(
    level_zero_pixels_to_slide: Any,
    selected_level_pixels_to_level_zero: Any,
    selected_level_dimensions: Any,
    pixel_rect: Any,
) -> dict[str, float]:
    """Map one original-pyramid view's local pixel edges into slide space."""

    level_zero_transform = validate_scale_translate_2d(level_zero_pixels_to_slide)
    level_transform = validate_scale_translate_2d(selected_level_pixels_to_level_zero)
    selected = _validate_dimensions(
        selected_level_dimensions, "selected-level dimensions"
    )
    rectangle = _contained_rect(pixel_rect, selected, "original source view")

    scale_x = level_zero_transform["scale_x"] * level_transform["scale_x"]
    scale_y = level_zero_transform["scale_y"] * level_transform["scale_y"]
    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "translate_x": level_zero_transform["translate_x"]
        + (level_transform["translate_x"] + rectangle["x"] * level_transform["scale_x"])
        * level_zero_transform["scale_x"],
        "translate_y": level_zero_transform["translate_y"]
        + (level_transform["translate_y"] + rectangle["y"] * level_transform["scale_y"])
        * level_zero_transform["scale_y"],
    }


def derive_derived_view_transform(
    artifact_pixels_to_slide: Any,
    artifact_dimensions: Any,
    pixel_rect: Any,
) -> dict[str, float]:
    """Map one derived-artifact view's local pixel edges into slide space."""

    artifact_transform = validate_scale_translate_2d(artifact_pixels_to_slide)
    dimensions = _validate_dimensions(artifact_dimensions, "artifact dimensions")
    rectangle = _contained_rect(pixel_rect, dimensions, "derived source view")
    return {
        "scale_x": artifact_transform["scale_x"],
        "scale_y": artifact_transform["scale_y"],
        "translate_x": artifact_transform["translate_x"]
        + rectangle["x"] * artifact_transform["scale_x"],
        "translate_y": artifact_transform["translate_y"]
        + rectangle["y"] * artifact_transform["scale_y"],
    }


def _validate_dimensions(value: Any, path: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != _DIMENSION_FIELDS:
        raise ValueError(f"{path} must contain exactly width and height")
    return {
        "width": require_positive_int32(value["width"], f"{path}.width"),
        "height": require_positive_int32(value["height"], f"{path}.height"),
    }


def _contained_rect(
    value: Any, dimensions: Mapping[str, int], path: str
) -> dict[str, int]:
    rectangle = validate_int_rect(value)
    if (
        rectangle["x"] + rectangle["width"] > dimensions["width"]
        or rectangle["y"] + rectangle["height"] > dimensions["height"]
    ):
        raise ValueError(f"{path} lies outside its selected level")
    return rectangle


def source_slide_bounds(origin: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return physical level-zero image edges in slide coordinates."""

    source = origin["source"]
    if origin["type"] == "derived-region":
        return (0.0, 0.0, float(source["width"]), float(source["height"]))
    transform = origin["level_zero_pixels_to_slide"]
    return (
        transform["translate_x"],
        transform["translate_y"],
        transform["translate_x"] + source["width"] * transform["scale_x"],
        transform["translate_y"] + source["height"] * transform["scale_y"],
    )
