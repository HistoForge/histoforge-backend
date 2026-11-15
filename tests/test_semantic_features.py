"""Semantic labels retain finite geometry and join exact core fragments."""

from itertools import product

import numpy as np
import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference import mask_features
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
)
from histoforge.cloud.inference.mask_features import feature_collection_from_mask
from histoforge.cloud.inference.semantic_features import (
    assemble_semantic_fragments,
    feature_collection_from_class_map,
)
from scipy import ndimage
from shapely.geometry import MultiPolygon, box, mapping, shape

_CLASSES = {2: "Tubule", 3: "Lumen"}
_BUDGET = {
    "max_features": 10_000,
    "max_vertices": 100_000,
    "max_record_bytes": 1_000_000,
    "max_transferred_bytes": 10_000_000,
    "max_decoded_bytes": 10_000_000,
}


def _finite_reference(class_map, class_by_channel=_CLASSES):
    """The previous finite runner's component-to-label assignment, unchanged."""

    labels = np.zeros_like(class_map, dtype=np.int32)
    classifications = {}
    next_id = 1
    for class_id, path_class in class_by_channel.items():
        labeled, count = ndimage.label(class_map == class_id)
        for local_id in range(1, count + 1):
            labels[labeled == local_id] = next_id
            classifications[next_id] = path_class
            next_id += 1
    return feature_collection_from_mask(
        labels, classifications, class_by_channel.values(), _BUDGET
    )


def _fragments(class_map, *, core_width, core_height):
    fragments = []
    for y in range(0, class_map.shape[0], core_height):
        for x in range(0, class_map.shape[1], core_width):
            crop = class_map[y : y + core_height, x : x + core_width]
            collection = feature_collection_from_class_map(
                crop,
                _CLASSES,
                _CLASSES.values(),
                _BUDGET,
                x_offset=x,
                y_offset=y,
            )
            fragments.extend(
                (feature["properties"]["path_class"], feature["geometry"])
                for feature in collection["features"]
            )
    return fragments


def _assert_same_regions(expected, assembled):
    remaining = [
        (feature["properties"]["path_class"], shape(feature["geometry"]))
        for feature in expected["features"]
    ]
    assert len(remaining) == len(assembled)
    for path_class, geometry in assembled:
        actual = shape(geometry)
        assert actual.geom_type == "Polygon"
        matches = [
            index
            for index, (expected_class, expected_geometry) in enumerate(remaining)
            if path_class == expected_class and actual.equals(expected_geometry)
        ]
        assert len(matches) == 1
        remaining.pop(matches[0])
    assert remaining == []


def test_every_binary_three_by_three_map_preserves_finite_and_assembled_regions():
    for pixels in product((0, 2), repeat=9):
        class_map = np.asarray(pixels, np.int32).reshape(3, 3)
        expected = _finite_reference(class_map)
        actual = feature_collection_from_class_map(
            class_map, _CLASSES, _CLASSES.values(), _BUDGET
        )
        assert canonical_bytes(actual) == canonical_bytes(expected), pixels
        assembled = assemble_semantic_fragments(
            _fragments(class_map, core_width=2, core_height=2)
        )
        _assert_same_regions(expected, assembled)


def test_multiclass_finite_geometry_is_byte_identical_and_order_independent():
    generator = np.random.default_rng(7382)
    for _ in range(25):
        class_map = generator.integers(0, 4, size=(11, 13), dtype=np.int32)
        expected = _finite_reference(class_map)
        actual = feature_collection_from_class_map(
            class_map, dict(reversed(_CLASSES.items())), _CLASSES.values(), _BUDGET
        )
        assert canonical_bytes(actual) == canonical_bytes(expected)


@pytest.mark.parametrize("core_width,core_height", [(3, 4), (5, 3), (1, 1)])
def test_core_assembly_preserves_multiple_classes_holes_and_junctions(
    core_width, core_height
):
    class_map = np.zeros((13, 17), np.int32)
    class_map[1:12, 1:16] = 2
    class_map[3:10, 4:13] = 0
    class_map[5:8, 6:11] = 3
    class_map[0, 0] = 2  # Only a corner touches the main Tubule region.
    class_map[11:13, 0] = 1  # A non-emitted winner does not become Tubule.
    expected = _finite_reference(class_map)

    assembled = assemble_semantic_fragments(
        _fragments(class_map, core_width=core_width, core_height=core_height)
    )

    _assert_same_regions(expected, assembled)
    assert len(assembled) == 3
    assert sum(len(shape(geometry).interiors) for _, geometry in assembled) == 1


@pytest.mark.parametrize("map_values", [[[2, 0], [0, 2]], [[2, 3], [3, 2]]])
def test_corner_only_contact_keeps_same_class_regions_separate(map_values):
    class_map = np.asarray(map_values, np.int32)
    assembled = assemble_semantic_fragments(
        _fragments(class_map, core_width=1, core_height=1)
    )

    assert len(assembled) == np.count_nonzero(class_map)
    _assert_same_regions(_finite_reference(class_map), assembled)


def test_shared_edges_join_only_with_the_same_class():
    class_map = np.array([[2, 2, 3, 3]], np.int32)
    assembled = assemble_semantic_fragments(
        _fragments(class_map, core_width=1, core_height=1)
    )

    assert len(assembled) == 2
    assert {path_class for path_class, _ in assembled} == set(_CLASSES.values())
    assert [shape(geometry).area for _, geometry in assembled] == [2, 2]


def test_cropped_map_offset_changes_coordinates_and_feature_identity():
    class_map = np.array([[2, 2, 0], [2, 0, 3]], np.int32)
    padded = np.zeros((10, 12), np.int32)
    padded[7:9, 9:12] = class_map
    expected = _finite_reference(padded)

    actual = feature_collection_from_class_map(
        class_map, _CLASSES, _CLASSES.values(), _BUDGET, x_offset=9, y_offset=7
    )

    assert canonical_bytes(actual) == canonical_bytes(expected)


@pytest.mark.parametrize("second_class", [0, 3])
def test_excessive_regions_fail_before_any_native_polygonization(
    monkeypatch, second_class
):
    checkerboard = np.indices((16, 16)).sum(axis=0) % 2
    class_map = np.where(checkerboard, second_class, 2).astype(np.int32)
    limit = 100 if second_class == 0 else 200

    def reject_shapes(*_args, **_kwargs):
        pytest.fail("over-count must fail before GDAL creates its polygon layer")

    monkeypatch.setattr(mask_features, "shapes", reject_shapes)

    with pytest.raises(ResultLimitExceeded, match="max_features"):
        feature_collection_from_class_map(
            class_map, _CLASSES, _CLASSES.values(), {**_BUDGET, "max_features": limit}
        )


def test_many_regions_use_no_per_component_label_raster_comparisons(monkeypatch):
    class NoComponentComparisons(np.ndarray):
        def __eq__(self, _other):
            pytest.fail("component assignment rescanned the complete label raster")

    label = ndimage.label

    def labeled_without_component_scans(*args, **kwargs):
        result, count = label(*args, **kwargs)
        return result.view(NoComponentComparisons), count

    monkeypatch.setattr(ndimage, "label", labeled_without_component_scans)
    class_map = np.where(np.indices((32, 32)).sum(axis=0) % 2, 0, 2).astype(np.int32)

    actual = feature_collection_from_class_map(
        class_map, _CLASSES, _CLASSES.values(), {**_BUDGET, "max_features": 512}
    )

    assert len(actual["features"]) == 512
    assert all(shape(feature["geometry"]).area == 1 for feature in actual["features"])


def test_non_emitted_winners_remain_empty_and_do_not_consume_region_budget():
    class_map = np.indices((16, 16)).sum(axis=0) % 2
    actual = feature_collection_from_class_map(
        class_map, _CLASSES, _CLASSES.values(), {**_BUDGET, "max_features": 1}
    )

    assert actual == {"type": "FeatureCollection", "features": []}
    assert assemble_semantic_fragments(iter(())) == []


def test_fragment_assembly_is_deterministic_and_splits_multipolygons():
    fragments = [
        (
            "Tubule",
            canonical_geometry(
                mapping(MultiPolygon([box(0, 0, 2, 2), box(8, 0, 9, 1)]))
            ),
        ),
        ("Tubule", canonical_geometry(mapping(box(2, 0, 4, 2)))),
        ("Lumen", canonical_geometry(mapping(box(0, 3, 2, 4)))),
    ]
    expected = assemble_semantic_fragments(fragments)

    for order in [
        fragments,
        fragments[::-1],
        [fragments[1], fragments[2], fragments[0]],
    ]:
        actual = assemble_semantic_fragments(iter(order))
        assert canonical_bytes(actual) == canonical_bytes(expected)
        assert canonical_bytes(
            build_feature_collection(actual, _CLASSES.values())
        ) == canonical_bytes(build_feature_collection(expected, _CLASSES.values()))
    assert len(expected) == 3
    assert all(geometry["type"] == "Polygon" for _, geometry in expected)


def test_assembly_keeps_exact_coordinates_without_simplifying_collinear_vertices():
    geometry = canonical_geometry(
        {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [2, 0], [2, 1], [0, 1], [0, 0]]],
        }
    )

    assert assemble_semantic_fragments([("Tubule", geometry)]) == [("Tubule", geometry)]
