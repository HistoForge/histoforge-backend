"""Cross-language large-region profile and execution-plan fixtures."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud.inference.plans import (
    build_core_halo_plan,
    build_core_halo_profile,
    build_expected_model_execution_plan,
    build_finite_plan,
    build_global_decode_plan,
    build_global_decode_profile,
    model_execution_plan_unit_count,
    validate_large_region_profile,
    validate_model_execution_plan,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "contracts" / "large-region-execution-v1.json"
)


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _assert_exact(case, value):
    encoded = canonical_bytes(value)
    assert encoded.decode("utf-8") == case["canonical"], case["name"]
    assert sha256_bytes(encoded) == case["sha256"], case["name"]


def test_shared_fixture_validates_profiles_and_plans_exactly():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["profiles"]:
        profile = validate_large_region_profile(case["input"])
        _assert_exact(case, profile)
        assert validate_large_region_profile(profile) == profile

    for case in fixture["plans"]:
        plan = validate_model_execution_plan(case["input"])
        _assert_exact(case, plan)
        assert validate_model_execution_plan(plan) == plan


def test_builders_reproduce_every_self_digested_fixture():
    fixture = _fixture()
    global_profile = fixture["profiles"][0]["input"]
    core_profile = fixture["profiles"][1]["input"]
    assert build_global_decode_profile(
        global_profile["validation_evidence_digest"]
    ) == validate_large_region_profile(global_profile)
    assert build_core_halo_profile(
        core_profile["core_width_px"],
        core_profile["core_height_px"],
        core_profile["halo_px"],
        core_profile["validation_evidence_digest"],
    ) == validate_large_region_profile(core_profile)

    finite = fixture["plans"][0]["input"]
    global_plan = fixture["plans"][1]["input"]
    core_plan = fixture["plans"][2]["input"]
    assert build_finite_plan(
        finite["preprocessing_context_digest"]
    ) == validate_model_execution_plan(finite)
    assert build_global_decode_plan(
        global_plan["profile_digest"],
        global_plan["preprocessing_context_digest"],
    ) == validate_model_execution_plan(global_plan)
    assert build_core_halo_plan(
        core_plan["profile_digest"],
        core_plan["preprocessing_context_digest"],
        core_plan["windows"],
        resolved_halo_px=core_plan["resolved_halo_px"],
    ) == validate_model_execution_plan(core_plan)

    raster = {"width": 1800, "height": 1300}
    assert model_execution_plan_unit_count(core_profile, raster) == 4
    assert build_expected_model_execution_plan(
        core_profile,
        raster,
        core_plan["preprocessing_context_digest"],
        4,
    ) == validate_model_execution_plan(core_plan)
    with pytest.raises(ValueError, match="max_units"):
        build_expected_model_execution_plan(
            core_profile,
            raster,
            core_plan["preprocessing_context_digest"],
            3,
        )


def test_profile_shapes_and_self_digests_fail_closed():
    global_profile = copy.deepcopy(_fixture()["profiles"][0]["input"])
    global_profile["validation_evidence_digest"] = "9" * 64
    with pytest.raises(ValueError, match="profile_digest does not match"):
        validate_large_region_profile(global_profile)

    core_profile = copy.deepcopy(_fixture()["profiles"][1]["input"])
    core_profile["core_width_px"] = 0
    with pytest.raises(ValueError, match="positive signed 32-bit"):
        validate_large_region_profile(core_profile)

    core_profile = copy.deepcopy(_fixture()["profiles"][1]["input"])
    core_profile["halo_px"] = -1
    with pytest.raises(ValueError, match="nonnegative signed 32-bit"):
        validate_large_region_profile(core_profile)

    core_profile = copy.deepcopy(_fixture()["profiles"][1]["input"])
    core_profile["core_width_px"] = 2048
    with pytest.raises(ValueError, match="profile_digest does not match"):
        validate_large_region_profile(core_profile)

    global_profile = copy.deepcopy(_fixture()["profiles"][0]["input"])
    global_profile["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_large_region_profile(global_profile)

    global_profile = copy.deepcopy(_fixture()["profiles"][0]["input"])
    global_profile["type"] = "tiled"
    with pytest.raises(ValueError, match="unsupported large-region profile"):
        validate_large_region_profile(global_profile)


def test_core_windows_require_canonical_complete_contained_cover():
    core_plan = _fixture()["plans"][2]["input"]

    invalid = copy.deepcopy(core_plan)
    invalid["windows"][1]["unit_id"] = 0
    with pytest.raises(ValueError, match="contiguous unit_id"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    invalid["windows"][1]["core_rect"]["x"] += 1
    invalid["windows"][1]["core_rect"]["width"] -= 1
    with pytest.raises(ValueError, match="columns without gaps or overlap"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    for window in invalid["windows"][2:]:
        window["core_rect"]["y"] += 1
        window["core_rect"]["height"] -= 1
    with pytest.raises(ValueError, match="rows without gaps or overlap"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    invalid["windows"][0]["input_rect"] = {
        "x": 1,
        "y": 0,
        "width": 1119,
        "height": 864,
    }
    with pytest.raises(ValueError, match="input_rect must contain core_rect"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    invalid["windows"][0]["input_rect"]["width"] = 1801
    with pytest.raises(ValueError, match="inside the logical target"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    invalid["windows"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    invalid["windows"][0]["core_rect"]["x"] = 2**31 - 1
    with pytest.raises(ValueError, match="endpoint"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(core_plan)
    invalid["windows"][0], invalid["windows"][1] = (
        invalid["windows"][1],
        invalid["windows"][0],
    )
    invalid["windows"][0]["unit_id"] = 0
    invalid["windows"][1]["unit_id"] = 1
    with pytest.raises(ValueError, match="ordered row-major"):
        validate_model_execution_plan(invalid)


@pytest.mark.parametrize(
    "rectangles",
    [
        [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)],
        [(0, 0, 2, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
    ],
)
def test_core_windows_reject_non_grid_t_junctions_in_both_orientations(rectangles):
    windows = [
        {
            "unit_id": index,
            "core_rect": {"x": x, "y": y, "width": width, "height": height},
            "input_rect": {"x": x, "y": y, "width": width, "height": height},
        }
        for index, (x, y, width, height) in enumerate(rectangles)
    ]
    with pytest.raises(ValueError, match="same height|same column partition"):
        build_core_halo_plan("1" * 64, "2" * 64, windows, resolved_halo_px=0)


def test_custom_halo_is_resolved_without_mutating_the_advertised_profile():
    profile = _fixture()["profiles"][1]["input"]
    raster = {"width": 1800, "height": 1300}
    original = copy.deepcopy(profile)
    zero = build_expected_model_execution_plan(
        profile, raster, "5" * 64, 4, halo_px_override=0
    )
    assert zero["resolved_halo_px"] == 0
    assert all(
        window["input_rect"] == window["core_rect"] for window in zero["windows"]
    )
    large = build_expected_model_execution_plan(
        profile, raster, "5" * 64, 4, halo_px_override=2000
    )
    larger = build_expected_model_execution_plan(
        profile, raster, "5" * 64, 4, halo_px_override=2**31 - 1
    )
    assert large["windows"] == larger["windows"]
    assert large["plan_digest"] != larger["plan_digest"]
    assert profile == original
    assert large["profile_digest"] == profile["profile_digest"]
    for invalid in (-1, True, 0.5, 2**31):
        with pytest.raises(ValueError, match="halo_px_override"):
            build_expected_model_execution_plan(
                profile, raster, "5" * 64, 4, halo_px_override=invalid
            )
    for unsupported in (None, _fixture()["profiles"][0]["input"]):
        with pytest.raises(ValueError, match="requires a core-halo"):
            build_expected_model_execution_plan(
                unsupported, raster, "5" * 64, 4, halo_px_override=0
            )


def test_plan_shapes_digests_and_returned_collections_fail_closed():
    core_plan = copy.deepcopy(_fixture()["plans"][2]["input"])
    result = validate_model_execution_plan(core_plan)
    core_plan["windows"][0]["core_rect"]["width"] = 1
    assert result["windows"][0]["core_rect"]["width"] == 1024

    invalid = copy.deepcopy(_fixture()["plans"][0]["input"])
    invalid["preprocessing_context_digest"] = "6" * 64
    with pytest.raises(ValueError, match="plan_digest does not match"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(_fixture()["plans"][1]["input"])
    invalid["type"] = "global"
    with pytest.raises(ValueError, match="unsupported model execution plan"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(_fixture()["plans"][0]["input"])
    invalid["profile_digest"] = "7" * 64
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_model_execution_plan(invalid)

    invalid = copy.deepcopy(_fixture()["plans"][2]["input"])
    for left, right in (
        (invalid["windows"][0], invalid["windows"][1]),
        (invalid["windows"][2], invalid["windows"][3]),
    ):
        left["core_rect"]["width"] = 1023
        right["core_rect"]["x"] = 1023
        right["core_rect"]["width"] = 777
    with pytest.raises(ValueError, match="plan_digest does not match"):
        validate_model_execution_plan(invalid)
