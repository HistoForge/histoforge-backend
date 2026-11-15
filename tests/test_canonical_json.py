import json
import math
import random
import struct
from pathlib import Path

import pytest
import rfc8785
from histoforge.canonical_json import (
    canonical_bytes,
    canonicalize_json,
    parse_json,
    sha256_bytes,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "canonical-json-v1.json"


def test_canonical_json_fixtures():
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["schemaVersion"] == 1
    for case in fixture["cases"]:
        encoded = canonicalize_json(case["input"])
        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]


@pytest.mark.parametrize(
    "source",
    [
        '{"a": 1, "a": 2}',
        '{"nested": {"a": 1, "a": 2}}',
        '{"n": NaN}',
        '{"n": Infinity}',
        '{"a": 1,}',
        "{'a': 1}",
        "[1,]",
        "1 2",
        '"\\ud800"',
    ],
)
def test_parse_json_rejects_noncanonical_extensions(source):
    with pytest.raises(ValueError):
        parse_json(source)


@pytest.mark.parametrize("source", ['{"n": 1e400}'])
def test_canonical_json_rejects_values_outside_jcs_domain(source):
    with pytest.raises(ValueError):
        canonicalize_json(source)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.125,
        -0.125,
        1e-7,
        1e-6,
        1e20,
        1e21,
        float(2**53 - 1),
        float(-(2**53 - 1)),
        float(2**53),
        float(-(2**53)),
        math.nextafter(float(2**53), math.inf),
        math.nextafter(float(-(2**53)), -math.inf),
        math.nextafter(1.0, 0.0),
        math.nextafter(1.0, math.inf),
        math.ulp(0.0),
        -math.ulp(0.0),
        float.fromhex("0x1.fffffffffffffp+1023"),
    ],
)
def test_number_encoding_matches_rfc8785_at_binary64_boundaries(value):
    assert canonical_bytes(value) == rfc8785.dumps(value)


def test_number_encoding_matches_rfc8785_for_random_binary64_values():
    generator = random.Random(42)
    for _ in range(4096):
        value = struct.unpack(">d", generator.getrandbits(64).to_bytes(8, "big"))[0]
        if math.isfinite(value):
            assert canonical_bytes(value) == rfc8785.dumps(value), value.hex()


def test_number_preparation_preserves_nested_input_types_and_signed_zero():
    values = [0.0, -0.0, 1.0, -2.0, 0.25, float(2**53)]
    tree = {"points": [values], "flags": [True, False, None]}
    before = [(type(value), struct.pack(">d", value)) for value in values]

    assert canonical_bytes(tree) == rfc8785.dumps(tree)
    assert [(type(value), struct.pack(">d", value)) for value in values] == before
    assert tree["points"][0] is values
    assert tree["flags"] == [True, False, None]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_numbers_still_fail_after_number_preparation(value):
    with pytest.raises(ValueError, match="cannot be encoded"):
        canonical_bytes({"coordinate": value})
