"""Cross-language image-capture identities, placement and strict boundary validation."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest
from histoforge.cloud.sources.capture import (
    build_captured_source_view,
    build_source_capture,
    captured_image_origin,
    captured_view_geometry,
    validate_captured_source_binding,
    validate_captured_source_view,
    validate_source_capture,
)

FIXTURE = Path(__file__).parent / "fixtures/contracts/source-capture-v1.json"


def test_shared_fixture_captures_geometry_without_an_uploaded_file():
    fixture = json.loads(FIXTURE.read_text())
    capture = validate_source_capture(fixture["capture"])
    view = validate_captured_source_view(fixture["view"])
    assert canonical_bytes(capture).decode() == fixture["captureCanonical"]
    assert canonical_bytes(view).decode() == fixture["viewCanonical"]
    assert captured_view_geometry(view) == (
        fixture["sourceRectAtLevel"],
        fixture["viewPixelsToSlide"],
    )
    assert (
        captured_image_origin(capture, fixture["sourceRectAtLevel"])
        == fixture["imageOrigin"]
    )
    binding = validate_captured_source_binding(fixture["binding"], capture)
    assert canonical_digest(binding) == fixture["bindingSha256"]


@pytest.mark.parametrize("field", ["capture_digest", "writer_build", "source_level"])
def test_capture_rejects_modified_or_missing_authority(field):
    capture = json.loads(FIXTURE.read_text())["capture"]
    if field == "capture_digest":
        capture[field] = "0" * 64
    else:
        capture.pop(field)
    with pytest.raises(ValueError):
        validate_source_capture(capture)


def test_view_and_binding_do_not_redefine_the_capture():
    fixture = json.loads(FIXTURE.read_text())
    capture = fixture["capture"]
    changed_view = copy.deepcopy(fixture["view"])
    changed_view["pixel_rect"]["x"] += 1
    with pytest.raises(ValueError, match="view_digest"):
        validate_captured_source_view(changed_view)
    binding = copy.deepcopy(fixture["binding"])
    binding["capture_digest"] = "0" * 64
    with pytest.raises(ValueError, match="different capture"):
        validate_captured_source_binding(binding, capture)
    binding = copy.deepcopy(fixture["binding"])
    binding["source_rect_at_level"]["x"] = 0
    with pytest.raises(ValueError, match="outside"):
        validate_captured_source_binding(binding, capture)
    with pytest.raises(ValueError, match="outside"):
        build_captured_source_view(
            capture, {"x": 0, "y": 0, "width": 100000, "height": 1}
        )


def test_capture_level_zero_is_the_original_image_grid():
    capture = json.loads(FIXTURE.read_text())["capture"]
    capture.pop("capture_digest")
    capture["source_level"]["resolution_level"] = 0
    with pytest.raises(ValueError, match="level zero"):
        build_source_capture(capture)


def test_capture_rejects_nonzero_plane_and_rectangles_outside_the_source():
    capture = json.loads(FIXTURE.read_text())["capture"]
    capture.pop("capture_digest")
    capture["plane"]["z"] = 1
    with pytest.raises(ValueError, match="plane"):
        build_source_capture(capture)
    capture["plane"]["z"] = 0
    capture["source_rect_at_level"]["width"] = 100000
    with pytest.raises(ValueError, match="outside"):
        build_source_capture(capture)
