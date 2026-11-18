"""Deterministic durable-result feature streams.

The backend is the sole encoder.  A stream is one canonical fixed GeoJSON
``Feature`` per RFC 8142 record, compressed as one gzip member with a zero
timestamp.  The immutable artifact digest binds the exact compressed bytes;
clients independently validate every decoded record and biological limit.
"""

from __future__ import annotations

import gzip
import io
import zlib
from collections.abc import Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from histoforge.canonical_json import canonical_bytes, parse_json
from histoforge.cloud.inference.contract import (
    validate_client_result_budget,
)
from histoforge.cloud.inference.features import (
    FeatureGeometryKind,
    ResultLimitExceeded,
    encode_feature_record,
    encoded_feature_collection_size,
    geometry_vertex_count,
    validate_feature_collection,
    validate_feature_records,
)

RECORD_SEPARATOR = b"\x1e"
RECORD_TERMINATOR = b"\n"
GZIP_COMPRESSION_LEVEL = 9
# GzipFile's unchanged level-9, zero-time, unnamed-file writer envelope.
_GZIP_HEADER = bytes.fromhex("1f8b08000000000002ff")


@dataclass(frozen=True)
class EncodedFeatureStream:
    """Exact compressed bytes plus manifest counts derived from those records."""

    data: bytes
    decoded_bytes: int
    feature_count: int
    vertex_count: int


@dataclass(frozen=True)
class DecodedFeatureStream:
    """Validated records, their aligned coordinate bytes and derived counts.

    The coordinate bytes may be reused only before modifying the collection.
    """

    collection: dict[str, Any]
    encoded_coordinates: list[bytes]
    decoded_bytes: int
    feature_count: int
    vertex_count: int


def collection_budget_for_stream(budget: Mapping[str, int]) -> dict[str, int]:
    """Bound temporary geometry without treating gzip limits as JSON limits.

    Feature builders hold a JSON collection; durable publication stores framed
    feature records without that collection envelope. Bound the temporary JSON
    by the decoded stream ceiling plus one envelope. Publication must still
    check the original budget against actual framed and compressed stream bytes.
    Feature, vertex and individual-record limits remain unchanged.
    """

    collection_bytes = budget["max_decoded_bytes"] + encoded_feature_collection_size(
        0, 0
    )
    return {
        **budget,
        "max_transferred_bytes": collection_bytes,
        "max_decoded_bytes": collection_bytes,
    }


def encode_feature_stream(
    feature_collection: Any,
    declared_classes: Collection[str],
    *,
    geometry_kind: FeatureGeometryKind = "polygonal",
) -> EncodedFeatureStream:
    """Validate and encode one complete logical-input population.

    Empty output is represented by a valid nonempty gzip member containing no
    RFC 8142 records.  No FeatureCollection envelope is stored.
    """

    collection = validate_feature_collection(
        feature_collection, declared_classes, geometry_kind=geometry_kind
    )
    return _encode_canonical_feature_stream(collection, None)


def encode_canonical_feature_stream(
    feature_collection: Mapping[str, Any],
    result_budget: Any,
) -> EncodedFeatureStream:
    """Encode a backend-owned canonical collection under exact stream limits.

    The caller must pass a collection returned by the feature builder or
    validator in the same trusted call stack. External values must use
    ``encode_feature_stream`` so the full feature contract is validated first.
    """

    return _encode_canonical_feature_stream(
        feature_collection,
        validate_client_result_budget(result_budget),
    )


def _encode_canonical_feature_stream(
    collection: Mapping[str, Any],
    budget: Mapping[str, int] | None,
) -> EncodedFeatureStream:
    return _encode_feature_records(
        (
            (canonical_bytes(feature), geometry_vertex_count(feature["geometry"]))
            for feature in collection["features"]
        ),
        budget,
    )


def encode_canonical_feature_records(
    records: Iterable[tuple[bytes, int]], result_budget: Any
) -> EncodedFeatureStream:
    """Compress trusted canonical records without a complete decoded-output buffer.

    Each item contains one encoded feature and its vertex count. The caller owns
    validated geometry, global occurrence IDs and ascending feature-ID order;
    this encoder owns framing, compression and the exact output limits.
    """

    return _encode_feature_records(
        records, validate_client_result_budget(result_budget)
    )


def _encode_feature_records(
    records: Iterable[tuple[bytes, int]], budget: Mapping[str, int] | None
) -> EncodedFeatureStream:
    decoded_bytes = feature_count = vertex_count = 0

    output = io.BytesIO()
    with gzip.GzipFile(
        filename="",
        mode="wb",
        compresslevel=GZIP_COMPRESSION_LEVEL,
        fileobj=output,
        mtime=0,
    ) as stream:
        for encoded, vertices in records:
            feature_count += 1
            vertex_count += vertices
            decoded_bytes += (
                len(RECORD_SEPARATOR) + len(encoded) + len(RECORD_TERMINATOR)
            )
            if budget is not None:
                if feature_count > budget["max_features"]:
                    raise ResultLimitExceeded("feature stream exceeds max_features")
                if len(encoded) > budget["max_record_bytes"]:
                    raise ResultLimitExceeded(
                        "feature stream record exceeds max_record_bytes"
                    )
                if decoded_bytes > budget["max_decoded_bytes"]:
                    raise ResultLimitExceeded(
                        "feature stream exceeds max_decoded_bytes"
                    )
                if vertex_count > budget["max_vertices"]:
                    raise ResultLimitExceeded("feature stream exceeds max_vertices")
            stream.write(RECORD_SEPARATOR)
            stream.write(encoded)
            stream.write(RECORD_TERMINATOR)
            if budget is not None and output.tell() > budget["max_transferred_bytes"]:
                raise ResultLimitExceeded(
                    "feature stream exceeds max_transferred_bytes"
                )
    result = EncodedFeatureStream(
        data=output.getvalue(),
        decoded_bytes=decoded_bytes,
        feature_count=feature_count,
        vertex_count=vertex_count,
    )
    if budget is not None and len(result.data) > budget["max_transferred_bytes"]:
        raise ResultLimitExceeded("feature stream exceeds max_transferred_bytes")
    return result


def decode_feature_stream(
    data: Any,
    declared_classes: Collection[str],
    result_budget: Any,
    *,
    geometry_kind: FeatureGeometryKind = "polygonal",
) -> DecodedFeatureStream:
    """Validate one bounded gzip member and every canonical feature record.

    The artifact store checks compressed byte identity. Here the fixed gzip
    envelope, CRC and length are checked without recompressing the decoded data;
    equivalent valid DEFLATE encodings under that envelope are accepted.
    """

    if not isinstance(data, bytes) or not data:
        raise ValueError("feature stream must be nonempty bytes")
    budget = validate_client_result_budget(result_budget)
    if len(data) > budget["max_transferred_bytes"]:
        raise ResultLimitExceeded("feature stream exceeds max_transferred_bytes")

    payload = _decode_gzip_member(data, budget["max_decoded_bytes"])
    records = validate_feature_records(
        (parse_json(encoded) for encoded in _encoded_records(payload, budget)),
        declared_classes,
        geometry_kind=geometry_kind,
    )
    if records.vertex_count > budget["max_vertices"]:
        raise ResultLimitExceeded("feature stream exceeds max_vertices")

    # Revisit record boundaries, not parsed JSON or coordinate formatting. The
    # exact comparison also rejects alternate numeric and string spellings that
    # compare equal as Python values (for example 0, -0 and 0.0).
    for feature, coordinates, encoded in zip(
        records.features,
        records.encoded_coordinates,
        _encoded_records(payload, budget),
        strict=True,
    ):
        canonical = encode_feature_record(
            feature["id"],
            feature["properties"]["path_class"],
            feature["geometry"]["type"],
            coordinates,
            score=feature["properties"].get("score"),
        )
        if canonical != encoded:
            raise ValueError("feature stream record is not canonical JSON")
    return DecodedFeatureStream(
        collection={"type": "FeatureCollection", "features": records.features},
        encoded_coordinates=records.encoded_coordinates,
        decoded_bytes=len(payload),
        feature_count=len(records.features),
        vertex_count=records.vertex_count,
    )


def _decode_gzip_member(data: bytes, decoded_limit: int) -> bytearray:
    if len(data) < 18 or data[:3] != _GZIP_HEADER[:3]:
        raise ValueError("feature stream is not valid gzip")
    if data[:10] != _GZIP_HEADER:
        raise ValueError("feature stream does not use the deterministic gzip header")

    payload = bytearray()
    decoder = zlib.decompressobj(wbits=31)
    offset = 0
    pending = b""
    try:
        while not decoder.eof:
            if not pending and offset < len(data):
                # Bound unconsumed_tail copies as well as decoded output.
                pending = data[offset : offset + 64 * 1024]
                offset += len(pending)
            remaining = decoded_limit - len(payload)
            block = decoder.decompress(pending, min(1024 * 1024, remaining + 1))
            pending = decoder.unconsumed_tail
            payload.extend(block)
            if len(payload) > decoded_limit:
                raise ResultLimitExceeded("feature stream exceeds max_decoded_bytes")
            if not block and not pending and offset == len(data) and not decoder.eof:
                raise ValueError("feature stream is not valid gzip: incomplete member")
    except zlib.error as error:
        raise ValueError("feature stream is not valid gzip") from error
    # eof proves that zlib consumed and checked the CRC32 and ISIZE trailer.
    # GzipFile would silently accept concatenated members and zero padding.
    if decoder.unused_data or pending or offset < len(data):
        raise ValueError(
            "feature stream must contain one gzip member without trailing bytes"
        )
    return payload


def _encoded_records(payload: bytearray, budget: Mapping[str, int]) -> Iterator[bytes]:
    """Read bounded records without splitting or copying the entire payload."""

    offset = 0
    count = 0
    while offset < len(payload):
        end = payload.find(RECORD_TERMINATOR, offset)
        if end < 0:
            raise ValueError("feature stream record is missing its terminator")
        if payload[offset] != RECORD_SEPARATOR[0] or end == offset + 1:
            raise ValueError("feature stream record has invalid RFC 8142 framing")
        if end - offset - 1 > budget["max_record_bytes"]:
            raise ResultLimitExceeded("feature stream record exceeds max_record_bytes")
        count += 1
        if count > budget["max_features"]:
            raise ResultLimitExceeded("feature stream exceeds max_features")
        yield bytes(payload[offset + 1 : end])
        offset = end + 1
