"""Cross-language source-artifact publication contract fixtures."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud.artifacts.refs import validate_artifact_ref
from histoforge.cloud.sources.publication import (
    build_source_artifact_manifest,
    build_source_artifact_ref,
    validate_source_artifact_manifest,
    validate_source_artifact_publication,
    validate_source_artifact_ref,
    validate_source_artifact_reference,
    validate_source_artifact_seal_request,
    validate_source_artifact_seal_response,
    validate_source_artifact_sealed,
    validate_source_byte_chunk_acceptance,
    validate_source_byte_chunk_accepted,
    validate_source_byte_chunk_commit,
)
from histoforge.cloud.sources.status import (
    validate_source_artifact_status,
    validate_source_artifact_status_projection,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "contracts"
PUBLICATION_FIXTURE = FIXTURE_DIR / "source-artifact-publication-v2.json"
START_FIXTURE = FIXTURE_DIR / "source-artifacts-v2.json"


def _fixture(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _start_cases():
    return {case["name"]: case for case in _fixture(START_FIXTURE)["startCases"]}


def test_shared_publication_fixture_binds_exact_bytes_and_digests():
    fixture = _fixture(PUBLICATION_FIXTURE)
    starts = _start_cases()
    assert fixture["schemaVersion"] == 1

    encoded = fixture["encodedArtifactRef"]
    reference = validate_artifact_ref(encoded["value"])
    assert canonical_bytes(reference).decode("utf-8") == encoded["canonical"]
    assert sha256_bytes(canonical_bytes(reference)) == encoded["sha256"]

    for case in fixture["cases"]:
        start_case = starts[case["startCaseName"]]
        start = start_case["value"]
        accepted = start_case["accepted"]

        commit = validate_source_byte_chunk_commit(case["chunkCommit"])
        chunk_accepted = validate_source_byte_chunk_acceptance(
            start, commit, case["chunkAccepted"]
        )
        _assert_exact(case, "chunkAccepted", chunk_accepted)

        seal_request = validate_source_artifact_seal_request(case["sealRequest"])
        _assert_exact(case, "sealRequest", seal_request)

        manifest = build_source_artifact_manifest(
            start, seal_request, case["validatedLevels"]
        )
        assert manifest == case["manifest"], case["name"]
        assert validate_source_artifact_manifest(case["manifest"]) == manifest
        assert (
            validate_source_artifact_publication(start, seal_request, manifest)
            == manifest
        )
        _assert_exact(case, "manifest", manifest)

        reference = build_source_artifact_ref(start, manifest)
        assert reference == case["artifactRef"], case["name"]
        assert validate_source_artifact_ref(case["artifactRef"]) == reference
        assert (
            validate_source_artifact_reference(start, manifest, reference) == reference
        )
        _assert_exact(case, "artifactRef", reference)

        sealed = validate_source_artifact_sealed(case["sealed"])
        assert (
            validate_source_artifact_seal_response(
                start, accepted, seal_request, manifest, sealed
            )
            == sealed
        )
        _assert_exact(case, "sealed", sealed)

        status = validate_source_artifact_status(case["status"])
        if status["phase"] == "SEALED":
            assert (
                validate_source_artifact_status_projection(
                    start, accepted, status, seal_request, sealed
                )
                == status
            )
        else:
            assert (
                validate_source_artifact_status_projection(start, accepted, status)
                == status
            )
        _assert_exact(case, "status", status)


def test_validated_levels_are_sorted_detached_and_unique():
    case = copy.deepcopy(_fixture(PUBLICATION_FIXTURE)["cases"][0])
    start = _start_cases()[case["startCaseName"]]["value"]
    original = copy.deepcopy(case["validatedLevels"])

    manifest = build_source_artifact_manifest(
        start, case["sealRequest"], case["validatedLevels"]
    )

    assert case["validatedLevels"] == original
    assert [level["resolution_level"] for level in manifest["validated_levels"]] == [
        0,
        1,
        2,
        3,
    ]
    case["validatedLevels"][0]["width"] = 1
    assert manifest["validated_levels"][3]["width"] == 1250

    duplicated = copy.deepcopy(original)
    duplicated.append(copy.deepcopy(original[0]))
    with pytest.raises(ValueError, match="duplicate series/level"):
        build_source_artifact_manifest(start, case["sealRequest"], duplicated)


def test_manifest_rejects_missing_required_or_unrelated_levels():
    fixture = _fixture(PUBLICATION_FIXTURE)
    starts = _start_cases()
    original = fixture["cases"][0]
    original_start = starts[original["startCaseName"]]["value"]

    missing_required = copy.deepcopy(original["validatedLevels"])
    del missing_required[2]
    with pytest.raises(ValueError, match="required level dimensions"):
        build_source_artifact_manifest(
            original_start, original["sealRequest"], missing_required
        )

    wrong_series = copy.deepcopy(original["validatedLevels"])
    wrong_series[0]["series_index"] = 3
    with pytest.raises(ValueError, match="pyramid_series_index"):
        build_source_artifact_manifest(
            original_start, original["sealRequest"], wrong_series
        )

    derived = fixture["cases"][1]
    derived_start = starts[derived["startCaseName"]]["value"]
    extra = copy.deepcopy(derived["validatedLevels"])
    extra.append(
        {"series_index": 0, "resolution_level": 1, "width": 2048, "height": 1536}
    )
    with pytest.raises(ValueError, match="exactly its required"):
        build_source_artifact_manifest(derived_start, derived["sealRequest"], extra)


def test_cross_record_identity_mutations_fail_closed():
    case = _fixture(PUBLICATION_FIXTURE)["cases"][0]
    start_case = _start_cases()[case["startCaseName"]]
    start = start_case["value"]

    wrong_chunk = copy.deepcopy(case["chunkAccepted"])
    wrong_chunk["chunk"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="expected commit"):
        validate_source_byte_chunk_acceptance(start, case["chunkCommit"], wrong_chunk)

    wrong_manifest = copy.deepcopy(case["manifest"])
    wrong_manifest["total_bytes"] += 1
    with pytest.raises(ValueError, match="manifest_digest"):
        validate_source_artifact_manifest(wrong_manifest)

    wrong_reference = copy.deepcopy(case["artifactRef"])
    wrong_reference["start_request_digest"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_source_artifact_reference(start, case["manifest"], wrong_reference)

    wrong_sealed = copy.deepcopy(case["sealed"])
    wrong_sealed["absolute_expires_at"] = "2026-08-13T12:34:56.788Z"
    with pytest.raises(ValueError, match="deadline"):
        validate_source_artifact_seal_response(
            start,
            start_case["accepted"],
            case["sealRequest"],
            case["manifest"],
            wrong_sealed,
        )

    wrong_status = copy.deepcopy(case["status"])
    wrong_status["start_request_digest"] = "f" * 64
    with pytest.raises(ValueError, match="does not match request"):
        validate_source_artifact_status_projection(
            start, start_case["accepted"], wrong_status
        )

    outside_plan = copy.deepcopy(case["status"])
    outside_plan["missing_chunks"] = [3]
    with pytest.raises(ValueError, match="outside the canonical chunk plan"):
        validate_source_artifact_status_projection(
            start, start_case["accepted"], outside_plan
        )


def test_sealed_status_is_authoritative_and_bound_to_its_start_request():
    case = copy.deepcopy(_fixture(PUBLICATION_FIXTURE)["cases"][1])
    start_case = _start_cases()[case["startCaseName"]]
    start = start_case["value"]
    accepted = start_case["accepted"]
    status = case["status"]

    with pytest.raises(ValueError, match="expected seal request"):
        validate_source_artifact_status_projection(start, accepted, status)
    assert (
        validate_source_artifact_status_projection(
            start, accepted, status, case["sealRequest"]
        )
        == status
    )

    ref_mutations = {
        "artifact_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "creation_action_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        "start_request_digest": "f" * 64,
        "container": "TIFF_FAMILY",
    }
    for field, replacement in ref_mutations.items():
        wrong = copy.deepcopy(status)
        wrong["sealed_artifact"][field] = replacement
        with pytest.raises(ValueError, match=field):
            validate_source_artifact_status_projection(
                start, accepted, wrong, case["sealRequest"]
            )

    wrong_deadline = copy.deepcopy(status)
    wrong_deadline["absolute_expires_at"] = "2026-08-15T23:59:59.998Z"
    with pytest.raises(ValueError, match="deadline"):
        validate_source_artifact_status_projection(
            start, accepted, wrong_deadline, case["sealRequest"]
        )

    wrong_status_digest = copy.deepcopy(status)
    wrong_status_digest["sealed_artifact"]["bytes_digest"] = "f" * 64
    with pytest.raises(ValueError, match="seal request"):
        validate_source_artifact_status_projection(
            start,
            accepted,
            wrong_status_digest,
            case["sealRequest"],
        )

    for field in ("bytes_digest", "manifest_digest"):
        wrong_seal = copy.deepcopy(case["sealed"])
        wrong_seal["artifact"][field] = "f" * 64
        with pytest.raises(ValueError, match="seal publication"):
            validate_source_artifact_status_projection(
                start, accepted, status, case["sealRequest"], wrong_seal
            )


def test_status_phase_payloads_and_request_digest_are_closed():
    fixture = _fixture(PUBLICATION_FIXTURE)
    starts = _start_cases()
    uploading = fixture["cases"][0]
    sealed = fixture["cases"][1]

    wrong_phase = copy.deepcopy(uploading["status"])
    wrong_phase["sealed_artifact"] = uploading["sealed"]["artifact"]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_source_artifact_status(wrong_phase)

    missing_sealed_ref = copy.deepcopy(sealed["status"])
    del missing_sealed_ref["sealed_artifact"]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_source_artifact_status(missing_sealed_ref)

    duplicate_chunks = copy.deepcopy(uploading["status"])
    duplicate_chunks["missing_chunks"] = [0, 0]
    with pytest.raises(ValueError, match="duplicate chunk"):
        validate_source_artifact_status(duplicate_chunks)

    legacy_request_artifact = copy.deepcopy(uploading["status"])
    legacy_request_artifact["start_request_artifact"] = {}
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_source_artifact_status(legacy_request_artifact)

    wrong_digest = copy.deepcopy(uploading["status"])
    wrong_digest["start_request_digest"] = "F" * 64
    with pytest.raises(ValueError, match="lowercase"):
        validate_source_artifact_status(wrong_digest)

    wrong_deadline = copy.deepcopy(uploading["status"])
    wrong_deadline["upload_idle_expires_at"] = "2026-08-14T00:00:00.000Z"
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_source_artifact_status_projection(
            starts[uploading["startCaseName"]]["value"],
            starts[uploading["startCaseName"]]["accepted"],
            wrong_deadline,
        )


def test_unknown_fields_and_noncanonical_scalars_fail_closed():
    case = _fixture(PUBLICATION_FIXTURE)["cases"][0]

    chunk = copy.deepcopy(case["chunkAccepted"])
    chunk["extra"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_source_byte_chunk_accepted(chunk)

    seal = {"bytes_digest": case["sealRequest"]["bytes_digest"].upper()}
    with pytest.raises(ValueError, match="lowercase"):
        validate_source_artifact_seal_request(seal)

    manifest = copy.deepcopy(case["manifest"])
    manifest["schema_version"] = 1.0
    with pytest.raises(ValueError, match="32-bit"):
        validate_source_artifact_manifest(manifest)


def _assert_exact(case, field, value):
    encoded = canonical_bytes(value)
    assert encoded.decode("utf-8") == case[f"{field}Canonical"], case["name"]
    assert sha256_bytes(encoded) == case[f"{field}Sha256"], case["name"]
