"""Strict record and bounded single-member gzip checks without recompression."""

import gzip
import io
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference import feature_stream
from histoforge.cloud.inference.feature_stream import (
    decode_feature_stream,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
)
from tests.test_feature_stream import _budget, _geometry

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _gzip(payload, *, flush_at=None):
    output = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9
    ) as stream:
        if flush_at is not None:
            stream.write(payload[:flush_at])
            stream.flush()
            payload = payload[flush_at:]
        stream.write(payload)
    return output.getvalue()


def _budget_for_payload(data, payload):
    return {
        "max_transferred_bytes": len(data),
        "max_decoded_bytes": max(1, len(payload)),
        "max_record_bytes": max(1, len(payload)),
        "max_features": 1000,
        "max_vertices": 100_000,
    }


def _decode_payload(payload, classes=("Tubule",)):
    data = _gzip(payload)
    return decode_feature_stream(data, classes, _budget_for_payload(data, payload))


def _example():
    collection = build_feature_collection([("Tubule", _geometry())], {"Tubule"})
    return collection, encode_feature_stream(collection, {"Tubule"})


def test_alternate_deflate_body_is_accepted_without_recompression(monkeypatch):
    collection, encoded = _example()
    payload = gzip.decompress(encoded.data)
    alternate = _gzip(payload, flush_at=len(payload) // 2)
    assert alternate != encoded.data
    assert alternate[:10] == encoded.data[:10]
    assert gzip.decompress(alternate) == payload

    monkeypatch.setattr(
        feature_stream,
        "_encode_canonical_feature_stream",
        lambda *_args: pytest.fail("decoding must not recompress"),
    )
    budget = {**_budget(encoded), "max_transferred_bytes": len(alternate)}
    assert decode_feature_stream(alternate, {"Tubule"}, budget).collection == collection
    with pytest.raises(ResultLimitExceeded, match="max_transferred_bytes"):
        decode_feature_stream(
            alternate,
            {"Tubule"},
            {**budget, "max_transferred_bytes": len(alternate) - 1},
        )


@pytest.mark.parametrize(
    "damage",
    [
        "truncated_header",
        "truncated_body",
        "truncated_trailer",
        "crc",
        "isize",
        "deflate",
        "timestamp",
        "filename_flag",
        "compression_level",
        "os",
        "second_member",
        "zero_padding",
        "junk",
    ],
)
def test_invalid_gzip_never_reaches_record_validation(damage, monkeypatch):
    _collection, encoded = _example()
    data = bytearray(encoded.data)
    if damage == "truncated_header":
        del data[5:]
    elif damage == "truncated_body":
        del data[len(data) // 2 :]
    elif damage == "truncated_trailer":
        del data[-1:]
    elif damage in {"crc", "isize"}:
        data[-8 if damage == "crc" else -4] ^= 1
    elif damage == "deflate":
        data[10] = (data[10] & ~6) | 6  # Reserved DEFLATE block type.
    elif damage in {"timestamp", "filename_flag", "compression_level", "os"}:
        data[
            {"timestamp": 4, "filename_flag": 3, "compression_level": 8, "os": 9}[
                damage
            ]
        ] ^= 1
    elif damage == "second_member":
        data.extend(_gzip(b""))
    elif damage == "zero_padding":
        data.extend(b"\0")
    else:
        data.extend(b"trailing junk")
    monkeypatch.setattr(
        feature_stream,
        "parse_json",
        lambda *_args: pytest.fail("invalid gzip must fail before parsing records"),
    )
    with pytest.raises(ValueError, match="gzip"):
        decode_feature_stream(
            bytes(data),
            {"Tubule"},
            {**_budget(encoded), "max_transferred_bytes": len(data)},
        )


@pytest.mark.parametrize("payload_size", [0, 1, 1024 * 1024, 3 * 1024 * 1024])
def test_inflate_is_bounded_at_exact_decoded_limit(payload_size):
    payload = b"a" * payload_size
    data = _gzip(payload)
    # Test the framing-independent reader so repeated bytes exercise bounded
    # inflate output and trailer consumption, not invalid GeoJSON.
    assert feature_stream._decode_gzip_member(data, max(1, payload_size)) == payload
    if payload_size > 1:
        with pytest.raises(ResultLimitExceeded, match="max_decoded_bytes"):
            feature_stream._decode_gzip_member(data, payload_size - 1)


def test_multiple_compressed_input_chunks_are_read_completely():
    # SHA digests form reproducible low-compressibility bytes without random input.
    import hashlib

    payload = b"".join(hashlib.sha256(str(i).encode()).digest() for i in range(4000))
    data = _gzip(payload)
    assert len(data) > 64 * 1024
    assert feature_stream._decode_gzip_member(data, len(payload)) == payload
    with pytest.raises(ValueError, match="one gzip member"):
        feature_stream._decode_gzip_member(data + b"\0", len(payload))


@pytest.mark.parametrize(
    "payload",
    [b"\n", b"\x1e\n", b"{}\n", b"\x1e{}", b"\x1e{}\n\n"],
)
def test_bad_record_framing_is_rejected(payload):
    with pytest.raises(ValueError):
        _decode_payload(payload)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"[0,0]", b"[-0,0]"),
        (b"[0,0]", b"[0.0,0]"),
        (b'"geometry":', b'"geometry" : '),
        (b'"type":"Feature"', b'"type":"Feature","type":"Feature"'),
        (b'"type":"Feature"', b'"unknown":1,"type":"Feature"'),
        (b"[0,0]", b"[false,0]"),
        (b"[0,0]", b"[1e400,0]"),
    ],
)
def test_noncanonical_or_invalid_record_values_are_rejected(old, new):
    _collection, encoded = _example()
    payload = gzip.decompress(encoded.data)
    changed = payload.replace(old, new)
    assert changed != payload
    with pytest.raises(ValueError):
        _decode_payload(changed)


def test_reordered_keys_and_escaped_unicode_are_rejected():
    classes = {"Tübule"}
    collection = build_feature_collection([("Tübule", _geometry())], classes)
    feature = collection["features"][0]
    reordered = json.dumps(feature, separators=(",", ":"), ensure_ascii=False).encode()
    canonical = canonical_bytes(feature)
    assert reordered != canonical
    escaped = canonical.replace("ü".encode(), b"\\u00fc")
    for record in (reordered, escaped):
        with pytest.raises(ValueError, match="not canonical JSON"):
            _decode_payload(b"\x1e" + record + b"\n", classes)


@pytest.mark.parametrize(
    "case",
    json.loads((FIXTURES / "canonical-geometry-v1.json").read_text())["validCases"],
    ids=lambda case: case["name"],
)
def test_geometry_fixtures_keep_the_existing_binary64_and_topology_contract(case):
    collection = build_feature_collection([("Tubule", case["geometry"])], {"Tubule"})
    encoded = encode_feature_stream(collection, {"Tubule"})
    decoded = decode_feature_stream(encoded.data, {"Tubule"}, _budget(encoded))
    assert decoded.collection == collection
    assert decoded.encoded_coordinates == [case["canonical"].encode()]

    payload = gzip.decompress(encoded.data)
    if case["name"] == "fractional-triangle":
        alternatives = [(b"1e-7", b"0.0000001"), (b"0.1", b"0.10000000000000001")]
    elif case["name"] == "binary64-coordinate-rounding":
        alternatives = [(b"9007199254740992", b"9007199254740993")]
    else:
        alternatives = []
    for before, after in alternatives:
        changed = payload.replace(before, after)
        assert changed != payload
        with pytest.raises(ValueError):
            _decode_payload(changed)


@pytest.mark.parametrize("damage", ["duplicate", "reversed", "missing_occurrence"])
def test_stream_checks_complete_feature_identity_and_order(damage):
    collection = build_feature_collection(
        [("Tubule", _geometry()), ("Tubule", _geometry())], {"Tubule"}
    )
    features = collection["features"]
    if damage == "duplicate":
        features = [features[0], features[0]]
    elif damage == "reversed":
        features = list(reversed(features))
    else:
        # Keep occurrence 1 only: no duplicate or order error, but incomplete IDs.
        single = build_feature_collection([("Tubule", _geometry())], {"Tubule"})
        zero_id = single["features"][0]["id"]
        features = [feature for feature in features if feature["id"] != zero_id]
    payload = b"".join(
        b"\x1e" + canonical_bytes(feature) + b"\n" for feature in features
    )
    with pytest.raises(ValueError):
        _decode_payload(payload)


def test_feature_limit_stops_before_parsing_another_record(monkeypatch):
    _collection, encoded = _example()
    payload = gzip.decompress(encoded.data) * 2
    data = _gzip(payload)
    parse = feature_stream.parse_json
    calls = 0

    def counted_parse(record):
        nonlocal calls
        calls += 1
        assert calls == 1
        return parse(record)

    monkeypatch.setattr(feature_stream, "parse_json", counted_parse)
    with pytest.raises(ResultLimitExceeded, match="max_features"):
        decode_feature_stream(
            data, {"Tubule"}, {**_budget_for_payload(data, payload), "max_features": 1}
        )
