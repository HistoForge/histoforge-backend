"""Scored-point identity, closed admission and cross-language byte fixtures."""

from __future__ import annotations

import base64
import copy
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud.inference.feature_stream import (
    decode_feature_stream,
    encode_canonical_feature_records,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_bounded_feature_collection,
    build_feature_collection,
    build_point_feature_records,
    encoded_feature_record_size,
    validate_feature_collection,
)

FIXTURE = Path(__file__).parent / "fixtures/contracts/inference-point-features-v1.json"
BUDGET = dict.fromkeys(
    (
        "max_transferred_bytes",
        "max_decoded_bytes",
        "max_record_bytes",
        "max_features",
        "max_vertices",
    ),
    100_000,
)
CLASSES = {"Mitotic figure"}


@pytest.mark.parametrize("field", ["max_transferred_bytes", "max_decoded_bytes"])
def test_empty_collection_still_requires_its_exact_envelope_budget(field):
    size = len(canonical_bytes({"type": "FeatureCollection", "features": []}))
    for builder in [build_point_feature_records, build_bounded_feature_collection]:
        builder([], CLASSES, {**BUDGET, field: size})
        with pytest.raises(ResultLimitExceeded):
            builder([], CLASSES, {**BUDGET, field: size - 1})


def _records(detections=None, budget=BUDGET):
    if detections is None:
        detections = [("Mitotic figure", (10.125, 20.25), 0.0)]
    return build_point_feature_records(detections, CLASSES, budget)


def _stream_budget(encoded, records):
    return {
        "max_transferred_bytes": len(encoded.data),
        "max_decoded_bytes": max(1, encoded.decoded_bytes),
        "max_record_bytes": max(
            (len(record) for record, _ in records.encoded_records()), default=1
        ),
        "max_features": max(1, encoded.feature_count),
        "max_vertices": max(1, encoded.vertex_count),
    }


def test_shared_points_have_exact_ids_records_and_gzip_bytes():
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["schemaVersion"] == 1
    for case in fixture["cases"]:
        instances = [
            (path_class, tuple(xy), score)
            for path_class, xy, score in case["instances"]
        ]
        records = build_point_feature_records(
            instances, fixture["declaredClasses"], BUDGET
        )
        collection = records.as_collection()
        assert canonical_bytes(collection).decode() == case["collectionCanonical"]
        assert sha256_bytes(canonical_bytes(collection)) == case["collectionSha256"]
        assert records.vertex_count == len(instances) == case["vertexCount"]
        assert (
            validate_feature_collection(
                collection, fixture["declaredClasses"], geometry_kind="point"
            )
            == collection
        )
        shuffled = build_point_feature_records(
            reversed(instances), fixture["declaredClasses"], BUDGET
        )
        assert shuffled.as_collection() == collection
        encoded = encode_feature_stream(
            collection, fixture["declaredClasses"], geometry_kind="point"
        )
        assert encoded.data == base64.b64decode(case["gzipBase64"])
        assert sha256_bytes(encoded.data) == case["streamSha256"]
        budget = _stream_budget(encoded, records)
        assert (
            encode_canonical_feature_records(records.encoded_records(), budget)
            == encoded
        )
        decoded = decode_feature_stream(
            encoded.data, fixture["declaredClasses"], budget, geometry_kind="point"
        )
        assert decoded.collection == collection
        for feature, (record, vertices) in zip(
            collection["features"], records.encoded_records(), strict=True
        ):
            assert record == canonical_bytes(feature)
            assert vertices == 1
            assert len(record) == encoded_feature_record_size(
                feature["properties"]["path_class"],
                "Point",
                len(canonical_bytes(feature["geometry"]["coordinates"])),
                score=feature["properties"]["score"],
            )


def test_geometry_kind_is_explicit_and_existing_polygon_admission_stays_closed():
    collection = _records().as_collection()
    with pytest.raises(ValueError):
        validate_feature_collection(collection, CLASSES)
    with pytest.raises(ValueError):
        encode_feature_stream(collection, CLASSES)
    polygon = build_feature_collection(
        [
            (
                "Mitotic figure",
                {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [0, 2], [0, 0]]]},
            )
        ],
        CLASSES,
    )
    with pytest.raises(ValueError):
        validate_feature_collection(polygon, CLASSES, geometry_kind="point")
    with pytest.raises(ValueError):
        validate_feature_collection(collection, CLASSES, geometry_kind="unknown")


@pytest.mark.parametrize(
    "score", [None, True, "0.5", -0.1, 1.1, float("nan"), float("inf"), 0.9]
)
def test_point_scores_are_required_finite_float32_values(score):
    collection = _records().as_collection()
    collection["features"][0]["properties"]["score"] = score
    with pytest.raises(ValueError):
        validate_feature_collection(collection, CLASSES, geometry_kind="point")
    with pytest.raises(ValueError):
        _records([("Mitotic figure", (1, 2), score)])


@pytest.mark.parametrize(
    "damage",
    [
        "missing_score",
        "extra_property",
        "z",
        "empty",
        "boolean_xy",
        "nonfinite_xy",
        "multipoint",
        "undeclared_class",
        "score_tamper",
    ],
)
def test_invalid_point_results_fail_before_any_population_is_returned(damage):
    collection = _records().as_collection()
    feature = collection["features"][0]
    if damage == "missing_score":
        del feature["properties"]["score"]
    elif damage == "extra_property":
        feature["properties"]["class_probability"] = 0.5
    elif damage in {"z", "empty", "boolean_xy", "nonfinite_xy"}:
        feature["geometry"]["coordinates"] = {
            "z": [1, 2, 3],
            "empty": [],
            "boolean_xy": [True, 2],
            "nonfinite_xy": [1, float("nan")],
        }[damage]
    elif damage == "multipoint":
        feature["geometry"] = {"type": "MultiPoint", "coordinates": [[1, 2]]}
    elif damage == "undeclared_class":
        feature["properties"]["path_class"] = "Other"
    else:
        feature["properties"]["score"] = 0.5
    with pytest.raises(ValueError):
        validate_feature_collection(collection, CLASSES, geometry_kind="point")


def test_score_is_in_the_identity_and_coincident_points_keep_multiplicity():
    instances = [("Mitotic figure", (1, 2), score) for score in (0.0, 0.5, 0.5)]
    records = _records(instances)
    assert len({feature["id"] for feature in records.features}) == 3
    assert sorted(feature["properties"]["score"] for feature in records.features) == [
        0.0,
        0.5,
        0.5,
    ]
    singleton = _records(instances[:1]).features[0]
    assert singleton in records.features
    changed = _records([("Mitotic figure", (1, 2), 1.0)]).features[0]
    assert singleton["id"] != changed["id"]
    missing_occurrence = copy.deepcopy(records.as_collection())
    zero = _records([instances[1]]).features[0]
    missing_occurrence["features"].remove(zero)
    with pytest.raises(ValueError, match="ids do not match"):
        validate_feature_collection(missing_occurrence, CLASSES, geometry_kind="point")


@pytest.mark.parametrize("field", list(BUDGET))
def test_point_collection_and_stream_enforce_exact_limits(field):
    records = _records([("Mitotic figure", (1, 2), score) for score in (0.0, 0.5)])
    instances = [("Mitotic figure", (1, 2), score) for score in (0.0, 0.5)]
    size = len(canonical_bytes(records.as_collection()))
    budget = {
        **BUDGET,
        "max_transferred_bytes": size,
        "max_decoded_bytes": size,
        "max_record_bytes": max(len(record) for record, _ in records.encoded_records()),
        "max_features": 2,
        "max_vertices": 2,
    }
    assert _records(instances, budget) == records
    with pytest.raises(ResultLimitExceeded):
        _records(instances, {**budget, field: budget[field] - 1})
    encoded = encode_feature_stream(
        records.as_collection(), CLASSES, geometry_kind="point"
    )
    stream_budget = _stream_budget(encoded, records)
    reduced = {**stream_budget, field: stream_budget[field] - 1}
    with pytest.raises(ResultLimitExceeded):
        decode_feature_stream(encoded.data, CLASSES, reduced, geometry_kind="point")
    with pytest.raises(ResultLimitExceeded):
        encode_canonical_feature_records(records.encoded_records(), reduced)


def test_float32_model_score_round_trips_without_decimal_rounding():
    score = struct.unpack(">f", struct.pack(">f", 0.9))[0]
    records = _records([("Mitotic figure", (1, 2), score)])
    assert records.features[0]["properties"]["score"] == score
    encoded = encode_feature_stream(
        records.as_collection(), CLASSES, geometry_kind="point"
    )
    decoded = decode_feature_stream(
        encoded.data, CLASSES, _stream_budget(encoded, records), geometry_kind="point"
    )
    assert decoded.collection == records.as_collection()


def test_trusted_builder_accepts_model_array_scalars_without_widening_wire_types():
    records = _records(
        [("Mitotic figure", (np.float32(1.5), np.float64(2.5)), np.float32(0.9))]
    )
    assert type(records.features[0]["properties"]["score"]) is float
    assert records.features[0]["properties"]["score"] == float(np.float32(0.9))
    assert records.features[0]["geometry"]["coordinates"] == [1.5, 2.5]
    for bad in [np.bool_(True), True, "0.5"]:
        with pytest.raises(ValueError):
            _records([("Mitotic figure", (1, 2), bad)])
    records.features[0]["properties"]["score"] = np.float32(0.9)
    with pytest.raises(ValueError):
        validate_feature_collection(
            records.as_collection(), CLASSES, geometry_kind="point"
        )
