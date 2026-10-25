import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes, parse_json, sha256_bytes
from histoforge.cloud.inference.features import (
    CanonicalFeatureGeometry,
    ResultLimitExceeded,
    build_bounded_feature_collection,
    build_feature_collection,
    build_feature_collection_from_canonical_geometries,
    encode_feature_record,
    encoded_feature_record_size,
    validate_feature_collection,
)

FIXTURE = (
    Path(__file__).parent / "fixtures" / "contracts" / "inference-features-v1.json"
)


def _fixture():
    return json.loads(FIXTURE.read_text())


def _instances(case):
    return [(item["pathClass"], item["geometry"]) for item in case["instances"]]


def test_shared_valid_fixtures_match_exact_bytes_digests_and_validation():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["validCases"]:
        collection = build_feature_collection(_instances(case), case["declaredClasses"])
        encoded = canonical_bytes(collection)

        assert encoded.decode("utf-8") == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]
        assert (
            canonical_bytes(
                validate_feature_collection(collection, case["declaredClasses"])
            )
            == encoded
        ), case["name"]


def test_builder_is_independent_of_decoder_iteration_order():
    case = _fixture()["validCases"][1]
    forward = build_feature_collection(_instances(case), case["declaredClasses"])
    reverse = build_feature_collection(
        list(reversed(_instances(case))), case["declaredClasses"]
    )
    assert canonical_bytes(forward) == canonical_bytes(reverse)


def test_encoded_coordinate_envelope_matches_every_canonical_feature_fixture():
    for case in _fixture()["validCases"]:
        collection = build_feature_collection(_instances(case), case["declaredClasses"])
        for feature in collection["features"]:
            geometry = feature["geometry"]
            path_class = feature["properties"]["path_class"]
            coordinates = canonical_bytes(geometry["coordinates"])
            encoded = encode_feature_record(
                feature["id"], path_class, geometry["type"], coordinates
            )
            assert encoded == canonical_bytes(feature)
            assert len(encoded) == encoded_feature_record_size(
                path_class, geometry["type"], len(coordinates)
            )


def test_preencoded_canonical_builder_is_byte_identical_to_validating_builder():
    case = _fixture()["validCases"][1]
    instances = _instances(case)
    expected = build_feature_collection(instances, case["declaredClasses"])
    budget = {
        "max_transferred_bytes": len(canonical_bytes(expected)),
        "max_decoded_bytes": len(canonical_bytes(expected)),
        "max_record_bytes": max(
            len(canonical_bytes(feature)) for feature in expected["features"]
        ),
        "max_features": len(expected["features"]),
        "max_vertices": 1_000,
    }
    canonical_instances = []
    for path_class, geometry_value in instances:
        geometry = canonical_geometry(geometry_value)
        canonical_instances.append(
            (
                path_class,
                CanonicalFeatureGeometry(
                    geometry,
                    canonical_bytes(geometry["coordinates"]),
                ),
            )
        )

    assert (
        build_feature_collection_from_canonical_geometries(
            canonical_instances, case["declaredClasses"], budget
        )
        == expected
    )


def test_shared_invalid_fixtures_fail_closed():
    for case in _fixture()["invalidCases"]:
        with pytest.raises(ValueError):
            validate_feature_collection(
                case["featureCollection"], case["declaredClasses"]
            )


def test_feature_construction_and_validation_do_not_mutate_inputs():
    case = _fixture()["validCases"][1]
    instances = _instances(case)
    original_instances = copy.deepcopy(instances)
    collection = build_feature_collection(instances, case["declaredClasses"])
    original_collection = copy.deepcopy(collection)

    validate_feature_collection(collection, case["declaredClasses"])

    assert instances == original_instances
    assert collection == original_collection


@pytest.mark.parametrize("validate", [False, True])
def test_duplicate_features_have_independent_geometry_values(validate):
    case = _fixture()["validCases"][1]
    collection = build_feature_collection(_instances(case), case["declaredClasses"])
    if validate:
        collection = validate_feature_collection(collection, case["declaredClasses"])

    duplicates = [
        feature
        for feature in collection["features"]
        if feature["properties"]["path_class"] == "Tubule"
    ]
    assert len(duplicates) == 2
    assert duplicates[0]["geometry"] is not duplicates[1]["geometry"]

    untouched = copy.deepcopy(duplicates[1]["geometry"])
    duplicates[0]["geometry"]["coordinates"][0][0][0] = 99
    assert duplicates[1]["geometry"] == untouched


@pytest.mark.parametrize("ordinate", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_feature_geometry_fails_closed(ordinate):
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [0, ordinate], [0, 0]]],
    }
    with pytest.raises(ValueError, match="finite JSON number"):
        build_feature_collection([("Tubule", geometry)], ["Tubule"])


def test_validator_accepts_only_json_arrays_for_features():
    with pytest.raises(ValueError, match="must be an array"):
        validate_feature_collection(
            {"type": "FeatureCollection", "features": ()}, ["Tubule"]
        )


def test_fixture_canonical_values_are_duplicate_free_json():
    for case in _fixture()["validCases"]:
        assert (
            canonical_bytes(parse_json(case["canonical"])).decode("utf-8")
            == case["canonical"]
        )


@pytest.mark.parametrize("instance_count", [1, 4])
def test_bounded_builder_accepts_exact_record_and_collection_byte_limits(
    instance_count,
):
    case = _fixture()["validCases"][1]
    instances = _instances(case)[:instance_count]
    expected = build_feature_collection(instances, case["declaredClasses"])
    encoded = canonical_bytes(expected)
    record_bytes = max(len(canonical_bytes(value)) for value in expected["features"])
    budget = {
        "max_transferred_bytes": len(encoded),
        "max_decoded_bytes": len(encoded),
        "max_record_bytes": record_bytes,
        "max_features": instance_count,
        "max_vertices": 1_000,
    }

    assert (
        build_bounded_feature_collection(instances, case["declaredClasses"], budget)
        == expected
    )

    too_small = dict(budget)
    too_small["max_transferred_bytes"] -= 1
    with pytest.raises(ResultLimitExceeded, match="transferred or decoded"):
        build_bounded_feature_collection(instances, case["declaredClasses"], too_small)
