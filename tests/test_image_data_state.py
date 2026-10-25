"""Cross-language persisted ImageData state fixtures."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.image_data_state import (
    ACTION_COMMIT_WITNESS_PROPERTY_KEY,
    decode_action_commit_witness_property,
    encode_action_commit_witness,
    validate_action_commit_witness,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "contracts" / "action-commit-witness-v1.json"
)


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_shared_witness_fixture_has_exact_property_bytes_and_digest():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1
    assert fixture["propertyKey"] == ACTION_COMMIT_WITNESS_PROPERTY_KEY

    for case in fixture["cases"]:
        witness = validate_action_commit_witness(case["input"])
        encoded = canonical_bytes(witness)

        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert encode_action_commit_witness(witness) == case["canonical"]
        assert decode_action_commit_witness_property(case["canonical"]) == witness


def test_shared_invalid_witness_mutations_fail_closed():
    fixture = _fixture()
    base = fixture["cases"][0]["input"]

    for mutation in fixture["invalidMutations"]:
        value = copy.deepcopy(base)
        field = mutation["path"][0]
        if mutation["operation"] == "remove":
            del value[field]
        else:
            value[field] = mutation["value"]
        with pytest.raises(ValueError):
            validate_action_commit_witness(value)


def test_property_load_requires_a_canonical_string():
    canonical = _fixture()["cases"][0]["canonical"]

    with pytest.raises(ValueError, match="must be a string"):
        decode_action_commit_witness_property(json.loads(canonical))
    with pytest.raises(ValueError, match="canonical JSON"):
        decode_action_commit_witness_property(" " + canonical)
    with pytest.raises(ValueError, match="duplicate JSON object member"):
        decode_action_commit_witness_property(canonical[:-1] + ',"schema_version":1}')
