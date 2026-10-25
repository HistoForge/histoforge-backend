"""Image geometry captured before its lossless source files are uploaded.

A capture describes one rectangle at a fixed source-pyramid level. A view is
the part needed by one model input. Both keep the same coordinates whether the
pixels arrive in one file or in several self-contained processing images.
Bindings identify sealed files; they do not change the capture or its views.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import canonical_digest, canonical_tree, require_sha256
from histoforge.cloud.sources.contract import validate_source_level_requirement
from histoforge.cloud.sources.identity import (
    validate_image_plane_index,
    validate_image_source_identity,
    validate_int_rect,
    validate_scale_translate_2d,
)
from histoforge.cloud.sources.publication import validate_source_artifact_ref
from histoforge.cloud.sources.views import (
    derive_derived_view_transform,
    derive_original_view_transform,
)

_CAPTURE_FIELDS = {
    "source",
    "plane",
    "source_level",
    "source_level_pixels_to_level_zero",
    "level_zero_pixels_to_slide",
    "source_rect_at_level",
    "writer_build",
}
_VIEW_FIELDS = {"capture", "pixel_rect"}
_BINDING_FIELDS = {"capture_digest", "artifact", "source_rect_at_level"}


def build_source_capture(value: Any) -> dict[str, Any]:
    """Capture source identity, reader geometry and the complete planned envelope."""

    item = _exact(value, _CAPTURE_FIELDS, "source capture")
    source = validate_image_source_identity(item["source"])
    plane = validate_image_plane_index(item["plane"])
    if plane != {"z": 0, "t": 0}:
        raise ValueError("source capture supports only plane z=0, t=0")
    level = validate_source_level_requirement(item["source_level"])
    rectangle = validate_int_rect(item["source_rect_at_level"])
    _require_contained(
        rectangle,
        {"x": 0, "y": 0, "width": level["width"], "height": level["height"]},
    )
    level_mapping = validate_scale_translate_2d(
        item["source_level_pixels_to_level_zero"]
    )
    slide_mapping = validate_scale_translate_2d(item["level_zero_pixels_to_slide"])
    if level["resolution_level"] == 0 and (
        level["width"] != source["width"]
        or level["height"] != source["height"]
        or level_mapping
        != {"scale_x": 1.0, "scale_y": 1.0, "translate_x": 0.0, "translate_y": 0.0}
    ):
        raise ValueError("source level zero must match the original image grid")
    return _digested(
        {
            "source": source,
            "plane": plane,
            "source_level": level,
            "source_level_pixels_to_level_zero": level_mapping,
            "level_zero_pixels_to_slide": slide_mapping,
            "source_rect_at_level": rectangle,
            "writer_build": parse_software_build_identity(item["writer_build"]),
        },
        "capture_digest",
    )


def validate_source_capture(value: Any) -> dict[str, Any]:
    """Validate immutable capture metadata without requiring an uploaded file."""

    item = _exact(value, _CAPTURE_FIELDS | {"capture_digest"}, "source capture")
    expected = build_source_capture({key: item[key] for key in _CAPTURE_FIELDS})
    _require_digest(item, expected, "capture_digest")
    return expected


def build_captured_source_view(capture: Any, pixel_rect: Any) -> dict[str, Any]:
    """Locate a fixed model view relative to the complete capture's top-left pixel.

    The rectangle is not relative to an individual uploaded file. Its origin
    fixes the float32 sampling coordinates for every later processing image.
    """

    captured = validate_source_capture(capture)
    rectangle = validate_int_rect(pixel_rect)
    envelope = captured["source_rect_at_level"]
    _require_contained(
        rectangle,
        {"x": 0, "y": 0, "width": envelope["width"], "height": envelope["height"]},
    )
    return _digested({"capture": captured, "pixel_rect": rectangle}, "view_digest")


def validate_captured_source_view(value: Any) -> dict[str, Any]:
    """Validate the fixed view and its complete captured coordinate authority."""

    item = _exact(value, _VIEW_FIELDS | {"view_digest"}, "captured source view")
    expected = build_captured_source_view(item["capture"], item["pixel_rect"])
    _require_digest(item, expected, "view_digest")
    return expected


def captured_view_geometry(view: Any) -> tuple[dict[str, int], dict[str, float]]:
    """Return the view rectangle in source-level pixels and its slide mapping.

    Preserve the complete prepared-image composition order: first place the
    capture in the source level, then place the view within that capture. Moving
    this calculation to a processing-image origin can change rounding.
    """

    item = validate_captured_source_view(view)
    capture = item["capture"]
    envelope = capture["source_rect_at_level"]
    rectangle = item["pixel_rect"]
    capture_mapping = _source_rectangle_mapping(capture, envelope)
    return (
        {
            **rectangle,
            "x": envelope["x"] + rectangle["x"],
            "y": envelope["y"] + rectangle["y"],
        },
        derive_derived_view_transform(
            capture_mapping,
            {"width": envelope["width"], "height": envelope["height"]},
            rectangle,
        ),
    )


def validate_captured_source_binding(value: Any, capture: Any) -> dict[str, Any]:
    """Bind one sealed file's exact source-level rectangle to a capture.

    The file may supply several processing units. Unit assignment belongs to
    the run plan, not to this reusable description of the file's pixels.
    """

    captured = validate_source_capture(capture)
    item = validate_source_binding(value)
    if item["capture_digest"] != captured["capture_digest"]:
        raise ValueError("source binding names a different capture")
    _require_contained(item["source_rect_at_level"], captured["source_rect_at_level"])
    return item


def validate_source_binding(value: Any) -> dict[str, Any]:
    """Validate file evidence before checking it against a particular capture."""

    item = _exact(value, _BINDING_FIELDS, "captured source binding")
    return canonical_tree(
        {
            "capture_digest": require_sha256(item["capture_digest"], "capture_digest"),
            "artifact": validate_source_artifact_ref(item["artifact"]),
            "source_rect_at_level": validate_int_rect(item["source_rect_at_level"]),
        }
    )


def captured_image_origin(capture: Any, source_rect_at_level: Any) -> dict[str, Any]:
    """Describe bytes written by the existing lossless QuPath source writer."""

    captured = validate_source_capture(capture)
    rectangle = validate_int_rect(source_rect_at_level)
    _require_contained(rectangle, captured["source_rect_at_level"])
    return canonical_tree(
        {
            "type": "derived-region",
            "source": captured["source"],
            "source_resolution_level": captured["source_level"]["resolution_level"],
            "plane": captured["plane"],
            "source_rect_at_level": rectangle,
            "artifact_pixels_to_slide": _source_rectangle_mapping(captured, rectangle),
            "writer_build": captured["writer_build"],
        }
    )


def _source_rectangle_mapping(capture, rectangle):
    level = capture["source_level"]
    return derive_original_view_transform(
        capture["level_zero_pixels_to_slide"],
        capture["source_level_pixels_to_level_zero"],
        {"width": level["width"], "height": level["height"]},
        rectangle,
    )


def _require_contained(inner, outer):
    if any(
        inner[axis] < outer[axis]
        or inner[axis] + inner[extent] > outer[axis] + outer[extent]
        for axis, extent in (("x", "width"), ("y", "height"))
    ):
        raise ValueError("source rectangle lies outside the captured image bounds")


def _exact(value, fields, path):
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value


def _digested(value, field):
    return canonical_tree({**value, field: canonical_digest(value)})


def _require_digest(item, expected, field):
    if require_sha256(item[field], field) != expected[field]:
        raise ValueError(f"{field} does not match its fields")
