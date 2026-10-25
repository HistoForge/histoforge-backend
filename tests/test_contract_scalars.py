import json
from pathlib import Path

import pytest
from histoforge.canonical_json import (
    canonical_bytes,
    canonical_map_entries,
    canonical_unordered,
    require_canonical_uuid,
    require_nonblank_string,
    require_nonnegative_int32,
    require_path_class_name,
    require_sha256,
    scalar_map_key,
    sha256_bytes,
)
from histoforge.module_ref import ModuleRef

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "canonical-scalars-v1.json"


def _fixture():
    return json.loads(FIXTURE.read_text())


def _validate_scalar(kind, value):
    if kind == "nonblank-string":
        return require_nonblank_string(value)
    if kind == "sha256":
        return require_sha256(value)
    if kind == "path-class-name":
        return require_path_class_name(value)
    if kind == "module-ref":
        return str(ModuleRef.parse(value))
    if kind == "nonnegative-int32":
        return require_nonnegative_int32(value)
    if kind == "canonical-uuid":
        return require_canonical_uuid(value)
    raise AssertionError(f"unknown fixture scalar kind: {kind}")


def test_shared_scalar_fixtures_encode_exact_values_and_map_keys():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["scalarCases"]:
        value = _validate_scalar(case["kind"], case["input"])
        encoded = canonical_bytes(value)
        assert encoded.decode("utf-8") == case["canonicalJson"], case["name"]
        assert scalar_map_key(value) == case["mapKey"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]


def test_shared_invalid_scalar_fixtures_fail_closed():
    for case in _fixture()["invalidScalarCases"]:
        with pytest.raises(ValueError):
            _validate_scalar(case["kind"], case["input"])


def test_shared_module_route_fixtures_require_canonical_raw_segments():
    for case in _fixture()["moduleRefRouteCases"]:
        if "canonical" in case:
            ref = ModuleRef.from_route_segments(case["owner"], case["module"])
            assert str(ref) == case["canonical"], case["name"]
        else:
            with pytest.raises(ValueError):
                ModuleRef.from_route_segments(case["owner"], case["module"])


def test_module_ref_enforces_segment_and_combined_boundaries():
    assert str(ModuleRef("a" * 47, "b" * 48)) == f"{'a' * 47}/{'b' * 48}"
    with pytest.raises(ValueError):
        ModuleRef("a" * 48, "b" * 48)
    with pytest.raises(ValueError):
        ModuleRef("é", "module")


def test_shared_compound_map_and_unordered_collection_fixtures():
    fixture = _fixture()
    compound = fixture["compoundMap"]
    entries = [(item["key"], item["value"]) for item in compound["entries"]]
    encoded = canonical_bytes(canonical_map_entries(entries))
    assert encoded.decode("utf-8") == compound["canonicalJson"]
    assert sha256_bytes(encoded) == compound["sha256"]

    unordered = fixture["unordered"]
    encoded = canonical_bytes(canonical_unordered(unordered["values"]))
    assert encoded.decode("utf-8") == unordered["canonicalJson"]
    assert sha256_bytes(encoded) == unordered["sha256"]


def test_canonical_collections_reject_identity_collisions():
    with pytest.raises(ValueError, match="duplicate canonical key"):
        canonical_map_entries(
            [
                (9_007_199_254_740_992, "left"),
                (9_007_199_254_740_993, "right"),
            ]
        )
    with pytest.raises(ValueError, match="duplicate canonical value"):
        canonical_unordered([9_007_199_254_740_992, 9_007_199_254_740_993])
