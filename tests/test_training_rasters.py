"""Accepted model fields derive one calibrated ROI grid without source IO."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.cloud.training.contract import training_task_from_capability
from histoforge.cloud.training.datasets.rasters import (
    maximum_training_source_frame,
    training_raster_transform,
    training_roi_raster,
    validate_training_raster_profile,
)
from tests.test_training_batches import _profile
from tests.test_training_contract import training_capability, training_model_output

_CASES = json.loads(
    (Path(__file__).parent / "fixtures/training-raster-v1.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_roi_framing_and_qupath_raster_fixture(case):
    """The Java fixture executes its actual raster kernel on the same frames."""
    origin, task, capability, profile = _inputs(case)
    profile = validate_training_raster_profile(task, capability, profile, 4096)
    actual = training_roi_raster(
        tuple(case["roi_bounds"]), origin, task, capability, profile
    )
    assert actual == case["expected_raster"]
    assert (
        training_raster_transform(tuple(case["frame_slide"]), task, origin["source"])
        == case["expected_raster"]
    )


def test_capture_reserves_maximum_context_but_execution_uses_accepted_minimum():
    case = {
        **_CASES[2],
        "roi_bounds": [70, 40, 82, 50],
    }
    origin, task, capability, profile = _inputs(case)
    # Context may be enabled after capture, within the same admitted capability.
    capability["default_field_profile"]["smallRoiContext"] = False
    bounds = tuple(case["roi_bounds"])
    assert maximum_training_source_frame(bounds, origin, task, capability) == (
        44,
        13,
        108,
        77,
    )
    accepted = validate_training_raster_profile(task, capability, profile, 4096)
    raster = training_roi_raster(bounds, origin, task, capability, accepted)
    assert raster["raster"]["width"] == raster["raster"]["height"] == 32
    assert raster["raster_to_slide"]["translate_x"] == 60
    assert raster["raster_to_slide"]["translate_y"] == 29
    without_context = {**accepted, "smallRoiContext": False}
    raster = training_roi_raster(bounds, origin, task, capability, without_context)
    assert (raster["raster"]["width"], raster["raster"]["height"]) == (12, 10)
    assert raster["raster_to_slide"]["translate_x"] == 70
    assert raster["raster_to_slide"]["translate_y"] == 40


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_reserved_frame_contains_exact_execution_frame_without_reselecting_source(case):
    origin, task, capability, profile = _inputs(case)
    before = copy.deepcopy((origin, task, capability, profile))
    maximum = maximum_training_source_frame(
        tuple(case["roi_bounds"]), origin, task, capability
    )
    frame = case["frame_slide"]
    assert maximum[0] <= frame[0] and maximum[1] <= frame[1]
    assert maximum[2] >= frame[2] and maximum[3] >= frame[3]
    if case["input_kind"] == "crop":
        assert maximum == tuple(frame)  # Crop padding is the only context rule.
    training_roi_raster(tuple(case["roi_bounds"]), origin, task, capability, profile)
    assert (origin, task, capability, profile) == before


def _inputs(case):
    """Build trusted geometry inputs using real task/capability validation."""
    fixed = case["planner"] == "fixed-field-v1"
    profile = (
        {
            **training_capability()["default_field_profile"],
            "side_model_px": case["minimum_side_model_px"],
            "small_roi_context": case["small_roi_context"],
        }
        if fixed
        else {
            **_profile(),
            "minSideModelPx": case["minimum_side_model_px"],
            "smallRoiContext": case["small_roi_context"],
        }
    )
    capability = _capability(profile)
    capability["input_kind"] = case["input_kind"]
    capability["crop_padding_model_px"] = case["crop_padding_model_px"]
    if case["input_kind"] == "crop":
        capability["small_roi_context_editable"] = False
    origin = {
        "type": "derived-region",
        "source": {
            "width": case["source_dimensions"][0],
            "height": case["source_dimensions"][1],
            "calibration": {
                "mpp_x": case["source_mpp"][0],
                "mpp_y": case["source_mpp"][1],
            },
        },
    }
    if "level_zero_pixels_to_slide" in case:
        origin.update(
            type="original-wsi",
            level_zero_pixels_to_slide=case["level_zero_pixels_to_slide"],
        )
    task = training_task_from_capability(
        capability, training_model_output(), case["target_mpp"], 4096
    )
    return origin, task, capability, profile


def _capability(profile):
    """Advertise a small fixture profile, never a production field-size default."""
    capability = training_capability()
    capability["default_field_profile"] = copy.deepcopy(profile)
    fixed = profile["planner"] == "fixed-field-v1"
    capability["allowed_field_side_model_px"] = (
        {"minimum": profile["side_model_px"], "maximum": profile["side_model_px"]}
        if fixed
        else {"minimum": 16, "maximum": 64}
    )
    capability["small_roi_context_editable"] = not fixed
    return capability
