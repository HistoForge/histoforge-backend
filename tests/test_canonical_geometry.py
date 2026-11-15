import copy
import json
from pathlib import Path

import pytest
import shapely
from histoforge.canonical_geometry import (
    canonical_geometry,
    canonical_geometry_bytes,
    canonical_geometry_coordinates,
    canonical_geometry_digest,
)
from histoforge.canonical_json import parse_json

FIXTURE = (
    Path(__file__).parent / "fixtures" / "contracts" / "canonical-geometry-v1.json"
)


def _fixture():
    return json.loads(FIXTURE.read_text())


def test_shared_geometry_fixtures_match_exact_bytes_and_digests():
    fixture = _fixture()
    assert fixture["schemaVersion"] == 1

    for case in fixture["validCases"]:
        geometry = case["geometry"]
        expected_coordinates = parse_json(case["canonical"])

        assert canonical_geometry_coordinates(geometry) == expected_coordinates, case[
            "name"
        ]
        assert canonical_geometry(geometry) == {
            "type": geometry["type"],
            "coordinates": expected_coordinates,
        }, case["name"]
        assert (
            canonical_geometry_bytes(geometry).decode("utf-8") == case["canonical"]
        ), case["name"]
        assert canonical_geometry_digest(geometry) == case["sha256"], case["name"]


def test_shared_invalid_geometry_fixtures_fail_closed():
    for case in _fixture()["invalidCases"]:
        with pytest.raises(ValueError):
            canonical_geometry(case["geometry"])


@pytest.mark.parametrize("ordinate", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coordinates_fail_closed(ordinate):
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [0, ordinate], [0, 0]]],
    }
    with pytest.raises(ValueError, match="finite JSON number"):
        canonical_geometry(geometry)


def test_canonicalization_does_not_mutate_input():
    geometry = _fixture()["validCases"][0]["geometry"]
    original = copy.deepcopy(geometry)
    canonical_geometry(geometry)
    assert geometry == original


@pytest.mark.parametrize("multipart", [False, True])
def test_winding_reuses_rings_constructed_for_topology_validation(
    monkeypatch, multipart
):
    polygon = [
        [[0, 0], [0, 9], [9, 9], [9, 0], [0, 0]],
        [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
    ]
    second = [[[20, 0], [20, 9], [29, 9], [29, 0], [20, 0]]]
    geometry = {
        "type": "MultiPolygon" if multipart else "Polygon",
        "coordinates": [second, polygon] if multipart else polygon,
    }
    expected = canonical_geometry(geometry)
    construct = shapely.linearrings
    constructed = []

    def record_construction(*args, **kwargs):
        constructed.append(True)
        return construct(*args, **kwargs)

    monkeypatch.setattr(shapely, "linearrings", record_construction)

    assert canonical_geometry(geometry) == expected
    # One shell and one hole, plus the second shell for a multipart instance.
    assert len(constructed) == 2 + int(multipart)


def test_reversed_rotated_rings_and_reordered_components_keep_fixture_bytes():
    for case in _fixture()["validCases"]:
        geometry = copy.deepcopy(case["geometry"])
        polygons = (
            [geometry["coordinates"]]
            if geometry["type"] == "Polygon"
            else geometry["coordinates"]
        )
        for polygon in polygons:
            for index, ring in enumerate(polygon):
                points = ring[:-1] if ring[0] == ring[-1] else ring
                points = list(reversed(points[1:] + points[:1]))
                polygon[index] = points + [points[0]]
            polygon[1:] = reversed(polygon[1:])
        if geometry["type"] == "MultiPolygon":
            polygons.reverse()

        assert canonical_geometry_bytes(geometry).decode() == case["canonical"]
