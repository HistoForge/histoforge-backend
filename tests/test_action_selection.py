"""Cross-language action, region-selection, scope, and output-key fixtures."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.action_selection import validate_image_action
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.execution_scope import validate_output_key

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "action-selection-v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _assert_exact(case, value):
    encoded = canonical_bytes(value)
    assert encoded.decode("utf-8") == case["canonical"], case["name"]
    assert sha256_bytes(encoded) == case["sha256"], case["name"]


def test_shared_fixture_normalizes_actions_and_output_keys_exactly():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["actions"]:
        action = validate_image_action(case["input"])
        _assert_exact(case, action)
        assert validate_image_action(action) == action

    for case in fixture["outputKeys"]:
        output = validate_output_key(case["input"])
        _assert_exact(case, output)
        assert validate_output_key(output) == output


def test_validation_returns_detached_normalized_collections():
    raw = copy.deepcopy(_fixture()["actions"][1]["input"])
    action = validate_image_action(raw)
    raw["work"]["regions"]["roi_ids"][0] = "99999999-9999-4999-8999-999999999999"
    raw["work"]["steps"][0]["targets"]["selectors"].clear()

    assert action["work"]["regions"]["roi_ids"] == [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    ]
    assert len(action["work"]["steps"][0]["targets"]["selectors"]) == 2


def test_region_and_target_membership_fail_closed():
    selected = copy.deepcopy(_fixture()["actions"][1]["input"])

    invalid = copy.deepcopy(selected)
    invalid["work"]["regions"]["roi_ids"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(selected)
    invalid["work"]["regions"]["roi_ids"].append(
        invalid["work"]["regions"]["roi_ids"][0]
    )
    with pytest.raises(ValueError, match="duplicates"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(selected)
    invalid["work"]["steps"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(selected)
    invalid["work"]["steps"].append(copy.deepcopy(invalid["work"]["steps"][0]))
    with pytest.raises(ValueError, match="duplicate step_id"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(selected)
    selector = invalid["work"]["steps"][0]["targets"]["selectors"][0]
    invalid["work"]["steps"][0]["targets"]["selectors"].append(selector)
    with pytest.raises(ValueError, match="selectors must not contain duplicates"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(selected)
    invalid["work"]["steps"][0]["targets"]["selectors"][0]["roi_id"] = (
        "99999999-9999-4999-8999-999999999999"
    )
    with pytest.raises(ValueError, match="outside"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(selected)
    invalid["work"]["steps"][0]["targets"]["selectors"] = [{"type": "slide-target"}]
    with pytest.raises(ValueError, match="cannot contain slide-target"):
        validate_image_action(invalid)

    slide = copy.deepcopy(_fixture()["actions"][2]["input"])
    slide["work"]["steps"][0]["targets"]["selectors"] = [
        {
            "type": "roi-target",
            "roi_id": "11111111-1111-4111-8111-111111111111",
        }
    ]
    with pytest.raises(ValueError, match="exactly one slide-target"):
        validate_image_action(slide)

    invalid = copy.deepcopy(selected)
    invalid["work"]["steps"][1]["targets"]["selectors"][1]["region"] = {"type": "slide"}
    with pytest.raises(ValueError, match="must name an ROI region"):
        validate_image_action(invalid)


def test_closed_action_and_output_shapes_fail_closed():
    fill = copy.deepcopy(_fixture()["actions"][0]["input"])

    invalid = copy.deepcopy(fill)
    invalid["type"] = "run"
    with pytest.raises(ValueError, match="unsupported image action"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(fill)
    invalid["remove_overlapping_same_class_duplicates"] = 1
    with pytest.raises(ValueError, match="boolean"):
        validate_image_action(invalid)

    invalid = copy.deepcopy(fill)
    invalid["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_image_action(invalid)

    metrics = copy.deepcopy(_fixture()["actions"][3]["input"])
    metrics["remove_overlapping_same_class_duplicates"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_image_action(metrics)

    metrics = copy.deepcopy(_fixture()["actions"][3]["input"])
    metrics["targets"]["region"]["annotation_ids"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_image_action(metrics)

    metrics = copy.deepcopy(_fixture()["actions"][3]["input"])
    annotation_ids = metrics["targets"]["region"]["annotation_ids"]
    annotation_ids.append(annotation_ids[0])
    with pytest.raises(ValueError, match="duplicates"):
        validate_image_action(metrics)

    output = copy.deepcopy(_fixture()["outputKeys"][0]["input"])
    output["scope"]["type"] = "crop"
    with pytest.raises(ValueError, match="unsupported execution scope"):
        validate_output_key(output)

    output = copy.deepcopy(_fixture()["outputKeys"][0]["input"])
    output["producer"]["step_id"] = "not-kebab"
    with pytest.raises(ValueError, match="canonical model id"):
        validate_output_key(output)
