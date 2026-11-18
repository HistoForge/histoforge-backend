"""Stored upload identities stay canonical, detached, and corruption-checked."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources import contract, publication, record_validation, records

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"
OTHER_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
OTHER_DIGEST = "f" * 64


def _start_cases():
    return json.loads((FIXTURES / "source-artifacts-v2.json").read_text())["startCases"]


def _publication_cases():
    return json.loads((FIXTURES / "source-artifact-publication-v2.json").read_text())[
        "cases"
    ]


def _reservation(case):
    return {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "start_request": copy.deepcopy(case["value"]),
        "start_request_digest": case["sha256"],
        "accepted_at": "2026-08-12T00:00:00.000Z",
        "absolute_expires_at": case["accepted"]["absolute_expires_at"],
    }


def _stored_publication(case):
    start = next(
        item for item in _start_cases() if item["name"] == case["startCaseName"]
    )
    reservation = _reservation(start)
    sealed = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "artifact_id": start["value"]["artifact_id"],
        "seal_request": copy.deepcopy(case["sealRequest"]),
        "manifest": copy.deepcopy(case["manifest"]),
        "sealed": copy.deepcopy(case["sealed"]),
    }
    return reservation, sealed


@pytest.mark.parametrize("case", _start_cases(), ids=lambda case: case["name"])
def test_reservation_read_preserves_fixture_digest_and_detached_normalization(case):
    source = _reservation(case)
    values = {}
    store = records.SourceArtifactRecords(values)
    artifact_id = source["start_request"]["artifact_id"]
    normalized, created = store.reserve(source)
    assert created
    key = next(iter(values))
    # Exercise raw persisted ordering, not only the already-normalized writer output.
    values[key]["reservation"] = copy.deepcopy(source)
    before = copy.deepcopy(values)

    result = store.load(artifact_id)["reservation"]
    expected = {**source, "start_request": json.loads(case["canonical"])}
    assert canonical_bytes(result) == canonical_bytes(expected)
    assert canonical_digest(result["start_request"]) == case["sha256"]
    assert result == normalized["reservation"]
    assert values == before

    result["start_request"]["required_levels"][0]["width"] = 1
    assert values == before
    winner, created = store.reserve(copy.deepcopy(source))
    assert not created
    assert winner == normalized


@pytest.mark.parametrize("case", _publication_cases(), ids=lambda case: case["name"])
def test_sealed_read_preserves_exact_fixture_and_detaches_nested_records(case):
    reservation, source = _stored_publication(case)
    artifact_id = source["artifact_id"]
    values = {}
    store = records.SourceArtifactRecords(values)
    store.reserve(reservation)
    normalized = store.publish_sealed(store.load(artifact_id), source)
    key = records.source_record_key(artifact_id)
    values[key]["sealed"] = copy.deepcopy(source)
    values[key]["sealed"]["manifest"]["validated_levels"].reverse()
    before = copy.deepcopy(values)

    result = store.sealed(store.load(artifact_id))
    assert canonical_bytes(result) == canonical_bytes(source)
    assert result == normalized
    assert values == before
    result["manifest"]["validated_levels"][0]["width"] = 1
    result["sealed"]["artifact"]["container"] = "PNG"
    assert values == before

    winner = store.publish_sealed(store.load(artifact_id), source)
    assert winner == normalized


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("start_request_digest",), OTHER_DIGEST),
        (("start_request", "artifact_id"), OTHER_ID),
        (("start_request", "origin", "unexpected"), True),
        (("start_request", "required_levels", 0, "width"), True),
        (("accepted_at",), "2026-08-16T00:00:00.000Z"),
        (("absolute_expires_at",), "invalid"),
    ],
)
def test_corrupt_reservation_read_fails_without_mutating_storage(path, replacement):
    source = _reservation(_start_cases()[0])
    artifact_id = source["start_request"]["artifact_id"]
    values = {}
    store = records.SourceArtifactRecords(values)
    store.reserve(source)
    key = next(iter(values))
    _replace(values[key]["reservation"], path, replacement)
    before = copy.deepcopy(values)

    with pytest.raises(GatewayFailure) as failure:
        store.load(artifact_id)
    assert failure.value.error["code"] == "INTERNAL_ERROR"
    assert values == before


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("artifact_id",), OTHER_ID),
        (("seal_request", "bytes_digest"), OTHER_DIGEST),
        (("manifest", "manifest_digest"), OTHER_DIGEST),
        (("manifest", "validated_levels", 0, "width"), True),
        (("sealed", "schema_version"), 1.0),
        (("sealed", "artifact", "artifact_id"), OTHER_ID),
        (("sealed", "artifact", "creation_action_id"), OTHER_ID),
        (("sealed", "artifact", "start_request_digest"), OTHER_DIGEST),
        (("sealed", "artifact", "manifest_digest"), OTHER_DIGEST),
        (("sealed", "artifact", "bytes_digest"), OTHER_DIGEST),
        (("sealed", "absolute_expires_at"), "2026-08-15T23:59:59.998Z"),
        (("sealed", "unexpected"), True),
    ],
)
def test_corrupt_sealed_read_fails_without_mutating_storage(path, replacement):
    reservation, source = _stored_publication(_publication_cases()[1])
    values = {}
    store = records.SourceArtifactRecords(values)
    store.reserve(reservation)
    store.publish_sealed(store.load(source["artifact_id"]), source)
    key = next(iter(values))
    _replace(values[key]["sealed"], path, replacement)
    before = copy.deepcopy(values)

    with pytest.raises(GatewayFailure) as failure:
        store.sealed(store.load(source["artifact_id"]))
    assert failure.value.error["code"] == "INTERNAL_ERROR"
    assert values == before


def test_self_digested_manifest_still_must_match_the_reserved_upload():
    reservation, source = _stored_publication(_publication_cases()[1])
    values = {}
    store = records.SourceArtifactRecords(values)
    store.reserve(reservation)
    store.publish_sealed(store.load(source["artifact_id"]), source)
    key = next(iter(values))
    manifest = values[key]["sealed"]["manifest"]
    manifest["total_bytes"] += 1
    manifest["manifest_digest"] = canonical_digest(
        {
            field: value
            for field, value in manifest.items()
            if field != "manifest_digest"
        }
    )
    values[key]["sealed"]["sealed"]["artifact"]["manifest_digest"] = manifest[
        "manifest_digest"
    ]

    with pytest.raises(GatewayFailure) as failure:
        store.sealed(store.load(source["artifact_id"]))
    assert failure.value.error["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("start_request_digest",), OTHER_DIGEST),
        (("start_request", "origin", "unexpected"), True),
        (("absolute_expires_at",), "2026-08-15T23:59:59.998Z"),
    ],
)
def test_sealed_read_rejects_a_mismatched_reservation(path, replacement):
    reservation, source = _stored_publication(_publication_cases()[1])
    values = {}
    store = records.SourceArtifactRecords(values)
    store.reserve(reservation)
    store.publish_sealed(store.load(source["artifact_id"]), source)
    _replace(
        values[records.source_record_key(source["artifact_id"])]["reservation"],
        path,
        replacement,
    )

    with pytest.raises(GatewayFailure) as failure:
        store.sealed(store.load(source["artifact_id"]))
    assert failure.value.error["code"] == "INTERNAL_ERROR"


def test_reservation_digest_uses_one_start_validation(monkeypatch):
    case = _start_cases()[0]
    values = {}
    store = records.SourceArtifactRecords(values)
    store.reserve(_reservation(case))
    calls = []
    validate = contract.validate_source_artifact_start_request

    def counted(value):
        calls.append(value)
        return validate(value)

    monkeypatch.setattr(
        record_validation, "validate_source_artifact_start_request", counted
    )
    monkeypatch.setattr(contract, "validate_source_artifact_start_request", counted)
    result = store.load(case["value"]["artifact_id"])["reservation"]
    assert len(calls) == 1
    assert result["start_request_digest"] == case["sha256"]


def test_reference_digest_uses_one_start_validation(monkeypatch):
    case = _publication_cases()[0]
    reservation, source = _stored_publication(case)
    calls = []
    validate = contract.validate_source_artifact_start_request

    def counted(value):
        calls.append(value)
        return validate(value)

    monkeypatch.setattr(publication, "validate_source_artifact_start_request", counted)
    monkeypatch.setattr(contract, "validate_source_artifact_start_request", counted)
    result = publication.build_source_artifact_ref(
        reservation["start_request"], source["manifest"]
    )
    assert len(calls) == 1
    assert canonical_bytes(result).decode() == case["artifactRefCanonical"]


def _replace(value, path, replacement):
    parent = value
    for field in path[:-1]:
        parent = parent[field]
    parent[path[-1]] = replacement
