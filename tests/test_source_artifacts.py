"""Cross-language source-artifact reservation contract fixtures."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    canonical_chunk_range,
    source_artifact_start_digest,
    validate_source_artifact_acceptance,
    validate_source_artifact_accepted,
    validate_source_artifact_start_request,
)
from histoforge.cloud.sources.identity import (
    validate_image_source_identity,
    validate_source_content_version,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "source-artifacts-v2.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _start_cases(fixture):
    return {case["name"]: case for case in fixture["startCases"]}


def test_shared_source_identity_fixture_has_exact_bytes_and_digest():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["sourceIdentityCases"]:
        value = validate_image_source_identity(case["value"])
        encoded = canonical_bytes(value)

        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert validate_image_source_identity(value) == value, case["name"]


def test_shared_start_fixtures_normalize_and_bind_acceptance_exactly():
    for case in _fixture()["startCases"]:
        request = validate_source_artifact_start_request(case["value"])
        encoded = canonical_bytes(request)

        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert source_artifact_start_digest(request) == case["sha256"], case["name"]
        assert validate_source_artifact_start_request(request) == request, case["name"]

        accepted = validate_source_artifact_acceptance(request, case["accepted"])
        accepted_bytes = canonical_bytes(accepted)
        assert accepted_bytes.decode("utf-8") == case["accepted_canonical"]
        assert sha256_bytes(accepted_bytes) == case["accepted_sha256"]
        assert validate_source_artifact_accepted(accepted) == accepted


def test_shared_chunk_plan_cases_are_exact_and_lazy():
    for case in _fixture()["startCases"]:
        request = validate_source_artifact_start_request(case["value"])
        assert (
            canonical_chunk_count(request["total_bytes"], request["chunk_size_bytes"])
            == case["chunk_count"]
        ), case["name"]
        assert [
            canonical_chunk_range(
                request["total_bytes"],
                request["chunk_size_bytes"],
                expected["chunk_id"],
            )
            for expected in case["chunk_ranges"]
        ] == case["chunk_ranges"], case["name"]


def test_shared_invalid_start_mutations_fail_closed():
    fixture = _fixture()
    cases = _start_cases(fixture)

    for mutation in fixture["invalidStartMutations"]:
        value = copy.deepcopy(cases[mutation["base"]]["value"])
        _apply_mutation(value, mutation)
        with pytest.raises(ValueError):
            validate_source_artifact_start_request(value)


def test_shared_invalid_acceptance_mutations_fail_closed():
    case = _fixture()["startCases"][0]
    request = validate_source_artifact_start_request(case["value"])

    for mutation in _fixture()["invalidAcceptedMutations"]:
        accepted = copy.deepcopy(case["accepted"])
        _apply_mutation(accepted, mutation)
        with pytest.raises(ValueError):
            validate_source_artifact_acceptance(request, accepted)


def test_validation_is_detached_and_does_not_mutate_input_order():
    source = copy.deepcopy(_fixture()["startCases"][0]["value"])
    original = copy.deepcopy(source)
    validated = validate_source_artifact_start_request(source)

    assert source == original
    assert [item["resolution_level"] for item in source["required_levels"]] == [
        2,
        0,
        1,
    ]
    assert [item["resolution_level"] for item in validated["required_levels"]] == [
        0,
        1,
        2,
    ]
    source["required_levels"][0]["width"] = 1
    assert validated["required_levels"][2]["width"] == 2500


def test_duplicate_level_keys_and_nonfinite_transforms_fail_closed():
    case = copy.deepcopy(_fixture()["startCases"][0]["value"])
    case["required_levels"].append(copy.deepcopy(case["required_levels"][0]))
    with pytest.raises(ValueError, match="duplicate series/level"):
        validate_source_artifact_start_request(case)

    case = copy.deepcopy(_fixture()["startCases"][2]["value"])
    case["origin"]["artifact_pixels_to_slide"]["translate_x"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        validate_source_artifact_start_request(case)


def test_derived_source_may_omit_version_but_original_source_may_not():
    derived = copy.deepcopy(_fixture()["startCases"][2]["value"])
    del derived["origin"]["source"]["content_version"]
    assert (
        "content_version"
        not in validate_source_artifact_start_request(derived)["origin"]["source"]
    )

    original = copy.deepcopy(_fixture()["startCases"][1]["value"])
    del original["origin"]["source"]["content_version"]
    with pytest.raises(ValueError, match="stable content_version"):
        validate_source_artifact_start_request(original)


def test_local_modification_time_uses_the_signed_jcs_safe_range():
    value = {
        "type": "local-file",
        "size_bytes": 1,
        "last_modified_epoch_millis": -9_007_199_254_740_991,
    }
    assert validate_source_content_version(value) == value

    value["last_modified_epoch_millis"] -= 1
    with pytest.raises(ValueError, match="safe integer"):
        validate_source_content_version(value)


def test_s3_etag_requires_a_nonempty_quoted_identifier():
    with pytest.raises(ValueError, match="quoted S3 ETag"):
        validate_source_content_version({"type": "s3-etag", "quoted_value": '""'})


def test_chunk_plan_rejects_unaddressable_count_and_out_of_range_id():
    with pytest.raises(ValueError, match="exceeds SourceByteChunkId"):
        canonical_chunk_count(2**31, 1)

    with pytest.raises(ValueError, match="outside the canonical chunk plan"):
        canonical_chunk_range(10, 4, 3)


def _apply_mutation(value, mutation):
    parent = value
    for segment in mutation["path"][:-1]:
        parent = parent[segment]
    field = mutation["path"][-1]
    if mutation["operation"] == "remove":
        del parent[field]
    else:
        parent[field] = mutation["value"]
