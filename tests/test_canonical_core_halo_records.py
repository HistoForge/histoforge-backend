"""Trusted core publication keeps exact bytes without repeating feature validation."""

import copy

import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference import feature_stream, features
from histoforge.cloud.inference.feature_stream import (
    encode_canonical_feature_records,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
)
from histoforge.cloud.inference.runs import core_halo
from tests.test_core_halo_runtime import _geometry

CLASSES = ("Tubule", 'Tübu"le\\cell: 🧪')
BUDGET = {
    "max_transferred_bytes": 1_000_000,
    "max_decoded_bytes": 1_000_000,
    "max_record_bytes": 100_000,
    "max_features": 100,
    "max_vertices": 1_000,
}


def _window(offset):
    return {
        "unit_id": 1,
        "input_rect": {"x": offset, "y": offset, "width": 80, "height": 80},
        "core_rect": {"x": offset + 10, "y": offset + 10, "width": 60, "height": 60},
    }


def _collection(empty=False):
    hole = _geometry(12, 12, width=30, height=30)
    hole["coordinates"].append([[20, 20], [20, 25], [25, 25], [25, 20], [20, 20]])
    multipart = {
        "type": "MultiPolygon",
        "coordinates": [
            _geometry(45, 20)["coordinates"],
            _geometry(60, 20)["coordinates"],
        ],
    }
    populations = [
        (CLASSES[0], hole),
        (CLASSES[0], hole),
        (CLASSES[1], multipart),
        (CLASSES[1], _geometry(3, 3)),  # context, not owned by the instance core
    ]
    return build_feature_collection([] if empty else populations, CLASSES)


@pytest.mark.parametrize("offset", [0, 90, 999])
@pytest.mark.parametrize("empty", [False, True])
def test_trusted_instances_preserve_bytes_ids_order_holes_and_duplicates(
    monkeypatch, offset, empty
):
    collection = _collection(empty)
    before = copy.deepcopy(collection)
    kwargs = dict(target_width=2000, target_height=2000, result_budget=BUDGET)
    expected = encode_feature_stream(
        core_halo.select_core_halo_unit_features(
            _window(offset), collection, CLASSES, **kwargs
        ),
        CLASSES,
    )
    monkeypatch.setattr(
        core_halo,
        "validate_feature_collection",
        lambda *_: pytest.fail("builder output was fully revalidated"),
    )
    owned = core_halo.select_canonical_core_halo_unit_features(
        _window(offset), collection, CLASSES, **kwargs
    )
    assert collection == before
    assert owned.vertex_count == expected.vertex_count
    assert len(owned.features) == expected.feature_count
    assert owned.encoded_coordinates == [
        canonical_bytes(feature["geometry"]["coordinates"])
        for feature in owned.features
    ]

    encode = features.canonical_bytes

    def scalar_only(value):
        assert not isinstance(value, (list, dict)), "coordinates were encoded twice"
        return encode(value)

    monkeypatch.setattr(features, "canonical_bytes", scalar_only)
    monkeypatch.setattr(feature_stream, "canonical_bytes", scalar_only)
    assert encode_canonical_feature_records(owned.encoded_records(), BUDGET) == expected


def test_trusted_semantic_regions_preserve_core_containment_and_exact_bytes(
    monkeypatch,
):
    collection = build_feature_collection(
        [(CLASSES[0], _geometry(10, 10, width=60, height=60))], CLASSES
    )
    kwargs = dict(target_width=2000, target_height=2000, result_budget=BUDGET)
    expected = encode_feature_stream(
        core_halo.translate_core_regions(_window(90), collection, CLASSES, **kwargs),
        CLASSES,
    )
    monkeypatch.setattr(
        core_halo,
        "validate_feature_collection",
        lambda *_: pytest.fail("builder output was fully revalidated"),
    )
    records = core_halo.translate_canonical_core_regions(
        _window(90), collection, CLASSES, **kwargs
    )
    assert (
        encode_canonical_feature_records(records.encoded_records(), BUDGET) == expected
    )
    outside = build_feature_collection([(CLASSES[0], _geometry(3, 3))], CLASSES)
    with pytest.raises(ValueError, match="outside its ownership core"):
        core_halo.translate_canonical_core_regions(
            _window(90), outside, CLASSES, **kwargs
        )


@pytest.mark.parametrize("field", list(BUDGET))
def test_preencoded_publication_still_enforces_each_exact_stream_limit(field):
    records = core_halo.select_canonical_core_halo_unit_features(
        _window(90),
        _collection(),
        CLASSES,
        target_width=2000,
        target_height=2000,
        result_budget=BUDGET,
    )
    expected = encode_feature_stream(records.as_collection(), CLASSES)
    exact = {
        "max_transferred_bytes": len(expected.data),
        "max_decoded_bytes": expected.decoded_bytes,
        "max_record_bytes": max(len(value) for value, _ in records.encoded_records()),
        "max_features": expected.feature_count,
        "max_vertices": expected.vertex_count,
    }
    assert (
        encode_canonical_feature_records(records.encoded_records(), exact) == expected
    )
    exact[field] -= 1
    with pytest.raises(ResultLimitExceeded, match=field):
        encode_canonical_feature_records(records.encoded_records(), exact)


@pytest.mark.parametrize(
    "select",
    [
        core_halo.select_core_halo_unit_features,
        core_halo.select_canonical_core_halo_unit_features,
    ],
)
def test_both_instance_paths_still_reject_artificial_edges_and_outside_input(select):
    kwargs = dict(target_width=2000, target_height=2000, result_budget=BUDGET)
    edge = build_feature_collection([("Tubule", _geometry(0, 20, width=30))], CLASSES)
    with pytest.raises(core_halo.InsufficientCoreHaloContext):
        select(_window(90), edge, CLASSES, **kwargs)
    outside = build_feature_collection(
        [("Tubule", _geometry(72, 20, width=10))], CLASSES
    )
    with pytest.raises(ValueError, match="inside input_rect"):
        select(_window(90), outside, CLASSES, **kwargs)


@pytest.mark.parametrize("semantic", [False, True])
def test_external_paths_continue_rejecting_invalid_feature_ids(semantic):
    collection = build_feature_collection([("Tubule", _geometry(20, 20))], CLASSES)
    collection["features"][0]["id"] = "0" * 64
    operation = (
        core_halo.translate_core_regions
        if semantic
        else core_halo.select_core_halo_unit_features
    )
    with pytest.raises(ValueError, match="ids do not match"):
        operation(
            _window(90),
            collection,
            CLASSES,
            target_width=2000,
            target_height=2000,
            result_budget=BUDGET,
        )


@pytest.mark.parametrize("offset", [0, 1, 8, 90, 999])
@pytest.mark.parametrize("multipart", [False, True])
def test_translation_encodes_each_ring_once_and_preserves_final_canonical_order(
    monkeypatch, offset, multipart
):
    # Adding 1 reverses lexicographic hole/component order: 2 < 9 as JSON text,
    # but 10 < 3 after translation. Numeric sorting would be incorrect.
    polygon = _geometry(0, 0, width=15, height=15)["coordinates"]
    polygon.extend(
        [
            _geometry(2, 2, width=1, height=1)["coordinates"][0],
            _geometry(9, 2, width=1, height=1)["coordinates"][0],
        ]
    )
    second = _geometry(2, 30, width=1, height=1)["coordinates"]
    third = _geometry(9, 30, width=1, height=1)["coordinates"]
    geometry = canonical_geometry(
        {
            "type": "MultiPolygon" if multipart else "Polygon",
            "coordinates": [polygon, second, third] if multipart else polygon,
        }
    )
    before = copy.deepcopy(geometry)
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    translated = [
        [[[x + offset, y + offset] for x, y in ring] for ring in component]
        for component in polygons
    ]
    expected = canonical_geometry(
        {
            "type": geometry["type"],
            "coordinates": translated if multipart else translated[0],
        }
    )
    expected_bytes = canonical_bytes(expected["coordinates"])
    encoded_rings = []

    def encode_ring_only(ring):
        assert all(
            len(point) == 2 and all(isinstance(x, (int, float)) for x in point)
            for point in ring
        ), "an already-encoded ring was traversed again as part of a polygon"
        encoded_rings.append(True)
        return canonical_bytes(ring)

    monkeypatch.setattr(core_halo, "canonical_bytes", encode_ring_only)
    actual = core_halo._translate_canonical_geometry(geometry, offset, offset)

    assert actual.value == expected
    assert actual.encoded_coordinates == expected_bytes
    assert geometry == before
    assert len(encoded_rings) == sum(map(len, polygons))
