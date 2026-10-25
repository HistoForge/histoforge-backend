"""Unit coverage for the shared immutable-download reference shape."""

import copy

import pytest
from histoforge.cloud.artifacts.refs import validate_artifact_ref


def test_artifact_ref_accepts_empty_and_encoded_representations():
    empty = {
        "artifact_id": "99999999-9999-4999-8999-999999999999",
        "media_type": "application/geo+json-seq",
        "size_bytes": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    encoded = {
        **empty,
        "artifact_id": "99999999-9999-4999-8999-999999999998",
        "content_encoding": "gzip",
        "size_bytes": 20,
        "sha256": "a" * 64,
    }

    assert validate_artifact_ref(empty) == empty
    assert validate_artifact_ref(encoded) == encoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", "not-a-uuid"),
        ("media_type", "Application/JSON"),
        ("media_type", "application/json; charset=utf-8"),
        ("content_encoding", "GZip"),
        ("size_bytes", -1),
        ("size_bytes", 1.0),
        ("sha256", "A" * 64),
    ],
)
def test_artifact_ref_rejects_noncanonical_fields(field, value):
    reference = {
        "artifact_id": "99999999-9999-4999-8999-999999999999",
        "media_type": "application/json",
        "content_encoding": "gzip",
        "size_bytes": 12,
        "sha256": "a" * 64,
    }
    reference[field] = value
    with pytest.raises(ValueError):
        validate_artifact_ref(reference)


def test_artifact_ref_rejects_null_or_unknown_optional_fields():
    reference = {
        "artifact_id": "99999999-9999-4999-8999-999999999999",
        "media_type": "application/json",
        "size_bytes": 12,
        "sha256": "a" * 64,
    }
    null_encoding = {**reference, "content_encoding": None}
    unknown = copy.deepcopy(reference)
    unknown["path"] = "/v1/artifacts/anything"

    with pytest.raises(ValueError):
        validate_artifact_ref(null_encoding)
    with pytest.raises(ValueError):
        validate_artifact_ref(unknown)
