"""Decoded label masks become deterministic one-feature-per-instance output."""

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference import mask_features
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
    validate_feature_collection,
)
from histoforge.cloud.inference.mask_features import feature_collection_from_mask
from shapely.affinity import translate
from shapely.geometry import mapping, shape


def test_disconnected_components_of_one_instance_form_one_multipolygon():
    mask = np.zeros((6, 8), dtype=np.int32)
    mask[1:3, 1:3] = 7
    mask[3:5, 5:7] = 7

    result = feature_collection_from_mask(mask, {7: "Tubule"}, ["Tubule"])

    assert len(result["features"]) == 1
    assert result["features"][0]["geometry"]["type"] == "MultiPolygon"
    assert len(result["features"][0]["geometry"]["coordinates"]) == 2
    assert validate_feature_collection(result, ["Tubule"]) == result


def test_class_mapping_drops_background_votes_and_single_class_applies_to_all():
    mask = np.array([[1, 1, 0, 2, 2]], dtype=np.int32)
    classified = feature_collection_from_mask(mask, {1: "Positive"}, ["Positive"])
    unclassified = feature_collection_from_mask(mask, "Cell", ["Cell"])

    assert len(classified["features"]) == 1
    assert len(unclassified["features"]) == 2
    assert {
        feature["properties"]["path_class"] for feature in unclassified["features"]
    } == {"Cell"}


def test_mask_holes_remain_holes_in_the_instance_polygon():
    mask = np.ones((5, 5), dtype=np.int32)
    mask[2, 2] = 0

    result = feature_collection_from_mask(mask, "Tubule", ["Tubule"])

    geometry = result["features"][0]["geometry"]
    assert geometry["type"] == "Polygon"
    assert len(geometry["coordinates"]) == 2


def test_mask_vectorization_accepts_exact_multipolygon_byte_limits():
    mask = np.zeros((6, 8), dtype=np.int32)
    mask[1:3, 1:3] = 7
    mask[3:5, 5:7] = 7
    expected = feature_collection_from_mask(mask, {7: "Tubule"}, ["Tubule"])
    budget = _budget(len(canonical_bytes(expected)))
    budget["max_record_bytes"] = len(canonical_bytes(expected["features"][0]))

    assert (
        feature_collection_from_mask(
            mask, {7: "Tubule"}, ["Tubule"], result_budget=budget
        )
        == expected
    )


def test_bounded_mask_vectorization_does_not_repeat_geometry_validation(monkeypatch):
    mask = np.zeros((6, 8), dtype=np.int32)
    mask[1:3, 1:3] = 7
    mask[3:5, 5:7] = 7
    expected = feature_collection_from_mask(mask, {7: "Tubule"}, ["Tubule"])
    budget = _budget(len(canonical_bytes(expected)))

    def repeated_validation(_geometry):
        raise AssertionError("canonical mask geometry was validated twice")

    monkeypatch.setattr(
        "histoforge.cloud.inference.features.canonical_geometry", repeated_validation
    )

    assert (
        feature_collection_from_mask(
            mask, {7: "Tubule"}, ["Tubule"], result_budget=budget
        )
        == expected
    )


def test_invalid_instance_ids_fail_before_vectorization():
    with pytest.raises(ValueError, match="nonnegative int32"):
        feature_collection_from_mask(np.array([[-1]], dtype=np.int32), "Cell", ["Cell"])


@pytest.mark.parametrize("absent_classes", [{}, {10**30: "Tubule"}])
def test_sparse_emitted_ids_filter_native_polygonization_without_changing_bytes(
    monkeypatch,
    absent_classes,
):
    labels = np.full((12, 16), 7, np.int32)
    labels[1:8, 1:8] = 2_000_000_000
    labels[3:5, 3:5] = 17  # Non-emitted instance remains a hole in the emitted one.
    labels[9:11, 12:15] = 2_000_000_000  # Disconnected part of the same instance.
    labels[1:3, 11:14] = 101
    classes = {101: "Lumen", 2_000_000_000: "Tubule", **absent_classes}
    native_shapes = mask_features.shapes

    def prior_polygonization(source, **kwargs):
        return native_shapes(source, **{**kwargs, "mask": source > 0})

    monkeypatch.setattr(mask_features, "shapes", prior_polygonization)
    expected = feature_collection_from_mask(labels, classes, ["Tubule", "Lumen"])
    budget = _budget(len(canonical_bytes(expected)))
    budget["max_features"] = 2
    budget["max_record_bytes"] = max(
        len(canonical_bytes(feature)) for feature in expected["features"]
    )
    calls = []

    def emitted_polygonization(source, **kwargs):
        assert source is labels
        np.testing.assert_array_equal(
            kwargs["mask"], (labels == 101) | (labels == 2_000_000_000)
        )
        calls.append(True)
        return native_shapes(source, **kwargs)

    monkeypatch.setattr(mask_features, "shapes", emitted_polygonization)
    actual = feature_collection_from_mask(labels, classes, ["Tubule", "Lumen"], budget)

    assert calls == [True]
    assert canonical_bytes(actual) == canonical_bytes(expected)
    tubule = next(
        item
        for item in actual["features"]
        if item["properties"]["path_class"] == "Tubule"
    )
    assert tubule["geometry"]["type"] == "MultiPolygon"
    assert sorted(len(polygon) for polygon in tubule["geometry"]["coordinates"]) == [
        1,
        2,
    ]


@pytest.mark.parametrize(
    "labels,classifications,error",
    [
        (np.array([[-1, 1]], np.int32), {}, "nonnegative int32"),
        (np.array([[0, 1]], np.int32), {99: None}, "PathClass strings"),
        (np.array([[0, 1]], np.int32), {False: "Cell"}, "positive integers"),
    ],
)
def test_suppressed_pixels_and_absent_classifications_are_still_validated(
    monkeypatch, labels, classifications, error
):
    def reject_polygonization(*_args, **_kwargs):
        pytest.fail("invalid complete inputs reached native polygonization")

    monkeypatch.setattr(mask_features, "shapes", reject_polygonization)
    with pytest.raises(ValueError, match=error):
        feature_collection_from_mask(labels, classifications, ["Cell"])


def test_empty_classification_map_has_no_native_eligible_pixels(monkeypatch):
    native_shapes = mask_features.shapes

    def empty_eligibility(labels, **kwargs):
        assert not kwargs["mask"].any()
        return native_shapes(labels, **kwargs)

    monkeypatch.setattr(mask_features, "shapes", empty_eligibility)
    assert (
        feature_collection_from_mask(np.ones((4, 4), np.int32), {}, ["Cell"])[
            "features"
        ]
        == []
    )


def test_pixel_offsets_preserve_holes_components_and_exact_feature_identity():
    mask = np.zeros((8, 12), np.int32)
    mask[1:6, 1:6] = 1
    mask[3, 3] = 0
    mask[6:8, 10:12] = 1
    original = feature_collection_from_mask(mask, "Tubule", ["Tubule"])
    expected = build_feature_collection(
        [
            (
                "Tubule",
                mapping(
                    translate(
                        shape(original["features"][0]["geometry"]), xoff=17, yoff=31
                    )
                ),
            )
        ],
        ["Tubule"],
    )
    budget = _budget(len(canonical_bytes(expected)))
    budget["max_record_bytes"] = len(canonical_bytes(expected["features"][0]))

    actual = feature_collection_from_mask(
        mask, "Tubule", ["Tubule"], budget, x_offset=17, y_offset=31
    )

    assert actual == expected
    assert actual["features"][0]["id"] != original["features"][0]["id"]
    assert validate_feature_collection(actual, ["Tubule"]) == actual


def test_pixel_offsets_are_applied_before_encoded_record_budget():
    mask = np.ones((2, 2), np.int32)
    original = feature_collection_from_mask(mask, "Tubule", ["Tubule"])
    budget = _budget(1_000_000)
    budget["max_record_bytes"] = len(canonical_bytes(original["features"][0]))

    with pytest.raises(ResultLimitExceeded, match="max_record_bytes"):
        feature_collection_from_mask(
            mask, "Tubule", ["Tubule"], budget, x_offset=1_000_000, y_offset=1_000_000
        )


@pytest.mark.parametrize(
    ("field", "limit", "message"),
    [
        ("max_features", 1, "max_features"),
        ("max_vertices", 3, "max_vertices"),
        ("max_record_bytes", 1, "max_record_bytes"),
        ("max_transferred_bytes", 1, "transferred or decoded"),
    ],
)
def test_result_budgets_stop_mask_vectorization_before_publication(
    field, limit, message
):
    mask = np.array([[1, 1, 0, 2, 2]], dtype=np.int32)
    budget = {
        "max_transferred_bytes": 1_000_000,
        "max_decoded_bytes": 1_000_000,
        "max_record_bytes": 100_000,
        "max_features": 100,
        "max_vertices": 1_000,
    }
    budget[field] = limit

    with pytest.raises(ResultLimitExceeded, match=message):
        feature_collection_from_mask(mask, "Cell", ["Cell"], budget)


def test_collection_byte_budget_stops_at_first_complete_feature_overflow(monkeypatch):
    first = _square(0)
    second = _square(2)
    one_feature = build_feature_collection([("Cell", first)], ["Cell"])
    budget = _budget(len(canonical_bytes(one_feature)))
    yielded = []

    def generated_shapes(*_args, **_kwargs):
        yielded.append(1)
        yield first, 1
        yielded.append(2)
        yield second, 2
        raise AssertionError("vectorization continued past the exact byte limit")

    monkeypatch.setattr(mask_features, "shapes", generated_shapes)

    with pytest.raises(ResultLimitExceeded, match="transferred or decoded"):
        feature_collection_from_mask(
            np.ones((1, 1), dtype=np.int32), "Cell", ["Cell"], budget
        )
    assert yielded == [1, 2]


def test_record_byte_budget_stops_at_first_component_overflow(monkeypatch):
    first = _square(0)
    second = _square(2)
    one_feature = build_feature_collection([("Cell", first)], ["Cell"])
    budget = _budget(1_000_000)
    budget["max_record_bytes"] = len(canonical_bytes(one_feature["features"][0]))
    yielded = []

    def generated_shapes(*_args, **_kwargs):
        yielded.append(1)
        yield first, 1
        yielded.append(2)
        yield second, 1
        raise AssertionError("vectorization continued past the exact record limit")

    monkeypatch.setattr(mask_features, "shapes", generated_shapes)

    with pytest.raises(ResultLimitExceeded, match="max_record_bytes"):
        feature_collection_from_mask(
            np.ones((1, 1), dtype=np.int32), "Cell", ["Cell"], budget
        )
    assert yielded == [1, 2]


def _square(offset):
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [offset, 0],
                [offset + 1, 0],
                [offset + 1, 1],
                [offset, 1],
                [offset, 0],
            ]
        ],
    }


def _budget(byte_limit):
    return {
        "max_transferred_bytes": byte_limit,
        "max_decoded_bytes": byte_limit,
        "max_record_bytes": 1_000_000,
        "max_features": 100,
        "max_vertices": 1_000,
    }
