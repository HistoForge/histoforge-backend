"""Deterministic durable feature-stream encoding tests."""

from __future__ import annotations

import base64
import gzip
import io
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud.inference.feature_stream import (
    collection_budget_for_stream,
    decode_feature_stream,
    encode_canonical_feature_records,
    encode_canonical_feature_stream,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_bounded_feature_collection,
    build_feature_collection,
)


def _geometry(offset: float = 0.0):
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [offset, 0.0],
                [offset + 2.0, 0.0],
                [offset + 2.0, 2.0],
                [offset, 0.0],
            ]
        ],
    }


def test_feature_stream_is_deterministic_rfc8142_gzip():
    collection = build_feature_collection(
        [("Tubule", _geometry(4.0)), ("Tubule", _geometry())], {"Tubule"}
    )

    first = encode_feature_stream(collection, {"Tubule"})
    second = encode_feature_stream(collection, {"Tubule"})

    assert first == second
    assert first.decoded_bytes == len(gzip.decompress(first.data))
    assert first.feature_count == 2
    assert first.vertex_count == 6
    assert first.data[:10] == bytes.fromhex("1f8b08000000000002ff")
    assert gzip.decompress(first.data) == b"".join(
        b"\x1e" + canonical_bytes(feature) + b"\n" for feature in collection["features"]
    )


def test_valid_empty_population_is_a_nonempty_empty_gzip_member():
    encoded = encode_feature_stream(
        {"type": "FeatureCollection", "features": []}, {"Tubule"}
    )

    assert encoded.feature_count == 0
    assert encoded.vertex_count == 0
    assert encoded.decoded_bytes == 0
    assert encoded.data
    assert gzip.decompress(encoded.data) == b""


@pytest.mark.parametrize("count", [0, 1, 1000])
def test_temporary_collection_preserves_exact_stream_acceptance(count):
    instances = [("Tubule", _geometry(float(index * 4))) for index in range(count)]
    expected = encode_feature_stream(
        build_feature_collection(instances, {"Tubule"}), {"Tubule"}
    )
    budget = _budget(expected)
    budget["max_decoded_bytes"] = max(1, expected.decoded_bytes)
    before = budget.copy()
    temporary = collection_budget_for_stream(budget)
    collection = build_bounded_feature_collection(instances, {"Tubule"}, temporary)

    assert budget == before
    assert encode_canonical_feature_stream(collection, budget) == expected
    assert all(
        temporary[field] == budget[field]
        for field in ("max_features", "max_vertices", "max_record_bytes")
    )
    if count:
        # The compressed ceiling is not a bound on temporary uncompressed JSON.
        assert len(canonical_bytes(collection)) > budget["max_transferred_bytes"]
        for field in ("max_transferred_bytes", "max_decoded_bytes"):
            too_small = {**budget, field: budget[field] - 1}
            with pytest.raises(ResultLimitExceeded, match=field):
                encode_canonical_feature_stream(collection, too_small)


def test_feature_stream_decoder_round_trips_and_enforces_each_byte_boundary():
    collection = build_feature_collection(
        [("Tubule", _geometry()), ("Tubule", _geometry(4.0))], {"Tubule"}
    )
    encoded = encode_feature_stream(collection, {"Tubule"})
    budget = _budget(encoded)

    decoded = decode_feature_stream(encoded.data, {"Tubule"}, budget)

    assert decoded.collection == collection
    assert decoded.decoded_bytes == encoded.decoded_bytes
    assert decoded.feature_count == encoded.feature_count
    assert decoded.vertex_count == encoded.vertex_count
    assert decoded.encoded_coordinates == [
        canonical_bytes(feature["geometry"]["coordinates"])
        for feature in collection["features"]
    ]

    for field, decrement, message in (
        ("max_transferred_bytes", 1, "max_transferred_bytes"),
        ("max_decoded_bytes", 1, "max_decoded_bytes"),
        ("max_record_bytes", 1, "max_record_bytes"),
        ("max_features", 1, "max_features"),
        ("max_vertices", 1, "max_vertices"),
    ):
        too_small = dict(budget)
        too_small[field] = budget[field] - decrement
        with pytest.raises(ResultLimitExceeded, match=message):
            decode_feature_stream(encoded.data, {"Tubule"}, too_small)


def test_trusted_canonical_encoder_is_exact_and_enforces_each_budget():
    collection = build_feature_collection(
        [("Tubule", _geometry()), ("Tubule", _geometry(4.0))], {"Tubule"}
    )
    expected = encode_feature_stream(collection, {"Tubule"})
    budget = _budget(expected)

    assert encode_canonical_feature_stream(collection, budget) == expected

    for field, message in (
        ("max_transferred_bytes", "max_transferred_bytes"),
        ("max_decoded_bytes", "max_decoded_bytes"),
        ("max_record_bytes", "max_record_bytes"),
        ("max_features", "max_features"),
        ("max_vertices", "max_vertices"),
    ):
        too_small = dict(budget)
        too_small[field] -= 1
        with pytest.raises(ResultLimitExceeded, match=message):
            encode_canonical_feature_stream(collection, too_small)


@pytest.mark.parametrize("count", [0, 2, 1000])
def test_incremental_encoder_preserves_single_write_gzip_and_every_budget(count):
    collection = build_feature_collection(
        [("Tubule", _geometry(float(index * 4))) for index in range(count)], {"Tubule"}
    )
    records = [(canonical_bytes(feature), 3) for feature in collection["features"]]
    payload = b"".join(b"\x1e" + record + b"\n" for record, _ in records)
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9
    ) as stream:
        stream.write(payload)
    expected = output.getvalue()
    budget = {
        "max_transferred_bytes": len(expected),
        "max_decoded_bytes": max(1, len(payload)),
        "max_record_bytes": max((len(record) for record, _ in records), default=1),
        "max_features": max(1, count),
        "max_vertices": max(1, 3 * count),
    }

    encoded = encode_canonical_feature_records(iter(records), budget)

    assert encoded.data == expected
    assert encoded.decoded_bytes == len(payload)
    assert encoded.feature_count == count
    assert encoded.vertex_count == 3 * count
    for field, ceiling in budget.items():
        if ceiling > 1:
            with pytest.raises(ResultLimitExceeded, match=field):
                encode_canonical_feature_records(
                    iter(records), {**budget, field: ceiling - 1}
                )


def test_record_encoder_stops_consuming_when_compressed_output_exceeds_budget():
    collection = build_feature_collection([("Tubule", _geometry())], {"Tubule"})
    encoded = canonical_bytes(collection["features"][0])

    def records():
        yield encoded, 3
        pytest.fail(
            "encoder consumed another record after its output exceeded the budget"
        )

    budget = {
        **_budget(encode_feature_stream(collection, {"Tubule"})),
        "max_transferred_bytes": 1,
    }
    with pytest.raises(ResultLimitExceeded, match="max_transferred_bytes"):
        encode_canonical_feature_records(records(), budget)


def test_feature_stream_decoder_rejects_noncanonical_or_malformed_gzip():
    collection = build_feature_collection([("Tubule", _geometry())], {"Tubule"})
    encoded = encode_feature_stream(collection, {"Tubule"})
    payload = gzip.decompress(encoded.data)
    noncanonical = gzip.compress(payload, mtime=1)

    with pytest.raises(ValueError, match="deterministic gzip"):
        decode_feature_stream(noncanonical, {"Tubule"}, _budget(encoded))
    with pytest.raises(ValueError, match="valid gzip"):
        decode_feature_stream(b"not gzip", {"Tubule"}, _budget(encoded))


def test_shared_feature_stream_fixture_is_exact():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "contracts" / "feature-stream-v1.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["schemaVersion"] == 1
    for case in fixture["cases"]:
        collection = json.loads(case["collectionCanonical"])
        encoded = encode_feature_stream(collection, fixture["declaredClasses"])
        assert encoded.data == base64.b64decode(case["gzipBase64"])
        assert len(encoded.data) == case["sizeBytes"]
        assert sha256_bytes(encoded.data) == case["sha256"]
        assert encoded.feature_count == case["featureCount"]
        assert encoded.vertex_count == case["vertexCount"]
        assert (
            decode_feature_stream(
                encoded.data, fixture["declaredClasses"], _budget(encoded)
            ).collection
            == collection
        )


def _budget(encoded):
    payload = gzip.decompress(encoded.data)
    record_sizes = [len(record) - 1 for record in payload.split(b"\n") if record]
    return {
        "max_transferred_bytes": len(encoded.data),
        "max_decoded_bytes": max(1, encoded.decoded_bytes),
        "max_record_bytes": max(record_sizes, default=1),
        "max_features": encoded.feature_count or 1,
        "max_vertices": encoded.vertex_count or 1,
    }
