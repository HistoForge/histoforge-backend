"""Indexed HoVer selection preserves exact overlap rules and bounds query work."""

from itertools import combinations
from random import Random

import pytest
from histoforge.cloud.architectures.hovernet.reconciliation import (
    _pairwise_overlaps,
    reconcile_window_predictions,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
)
from histoforge.cloud.inference.plans import (
    build_core_halo_profile,
    build_expected_model_execution_plan,
)
from shapely.affinity import translate
from shapely.geometry import Polygon, box, mapping


@pytest.mark.parametrize("seed", range(8))
def test_index_matches_an_independent_all_pairs_reference(seed):
    random = Random(seed)
    candidates = [
        {"unit_id": index, "geometry": box(x, y, x + width, y + height)}
        for index in range(32)
        for x, y, width, height in [
            [
                random.randrange(60),
                random.randrange(60),
                random.randrange(1, 25),
                random.randrange(1, 25),
            ]
        ]
    ]
    expected = {}
    for i, j in combinations(range(len(candidates)), 2):
        a, b = candidates[i]["geometry"], candidates[j]["geometry"]
        intersection = a.intersection(b).area
        if intersection > 0:
            expected[i, j] = intersection / a.union(b).area
    budget = {"limit": 10_000, "used": 0}
    assert _pairwise_overlaps(candidates, budget) == expected
    assert budget["used"] <= len(candidates) ** 2


def test_work_guard_counts_bbox_hits_even_when_polygons_do_not_overlap():
    ring = Polygon(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [[(10, 10), (90, 10), (90, 90), (10, 90)]],
    )
    candidates = [
        {"unit_id": 0, "geometry": ring},
        {"unit_id": 1, "geometry": box(20, 20, 30, 30)},
    ]
    assert _pairwise_overlaps(candidates, {"limit": 4, "used": 0}) == {}
    with pytest.raises(ResultLimitExceeded, match="spatial query"):
        _pairwise_overlaps(candidates, {"limit": 3, "used": 0})


@pytest.mark.parametrize("touch_only", [False, True])
def test_index_keeps_same_window_disjointness_before_any_class_filter(touch_only):
    candidates = [
        {"unit_id": 0, "geometry": box(0, 0, 10, 10)},
        {"unit_id": 0, "geometry": box(10 if touch_only else 9, 0, 20, 10)},
    ]
    if touch_only:
        assert _pairwise_overlaps(candidates, {"limit": 10, "used": 0}) == {}
    else:
        with pytest.raises(ValueError, match="from one window overlap"):
            _pairwise_overlaps(candidates, {"limit": 10, "used": 0})


def _plan(width, height, core, halo):
    return build_expected_model_execution_plan(
        build_core_halo_profile(core, core, halo, "a" * 64),
        {"width": width, "height": height},
        "b" * 64,
        64,
    )


def test_core_coverage_queries_share_the_same_closed_work_limit():
    plan = _plan(128, 128, 64, 48)
    geometry = box(32, 32, 96, 96)
    collections = [
        build_feature_collection(
            [
                (
                    "Tubule",
                    mapping(
                        translate(
                            geometry,
                            -window["input_rect"]["x"],
                            -window["input_rect"]["y"],
                        )
                    ),
                )
            ],
            {"Tubule"},
        )
        for window in plan["windows"]
    ]
    output, trace = reconcile_window_predictions(
        plan, collections, {"Tubule"}, max_spatial_query_hits=32
    )
    assert len(output["features"]) == 1
    assert trace["spatial_query_hits"] == 32
    with pytest.raises(ResultLimitExceeded, match="spatial query"):
        reconcile_window_predictions(
            plan, collections, {"Tubule"}, max_spatial_query_hits=31
        )


def test_disjoint_population_requires_only_linear_bbox_work():
    plan = _plan(256, 256, 256, 0)
    collection = build_feature_collection(
        [
            ("Tubule", mapping(box(x * 4 + 1, y * 4 + 1, x * 4 + 3, y * 4 + 3)))
            for x in range(20)
            for y in range(20)
        ],
        {"Tubule"},
    )
    output, trace = reconcile_window_predictions(
        plan, [collection], {"Tubule"}, max_spatial_query_hits=800
    )
    assert len(output["features"]) == 400
    assert trace["spatial_query_hits"] == 800


def test_repeated_fragment_diagnostics_share_one_group_vector():
    plan = _plan(8192, 8192, 4096, 2048)
    populations = [[] for _ in plan["windows"]]
    for index in range(32):
        x = 2055 + 3 * index
        populations[index % 4].append(box(x, 2049, x + 10, 2053))
        if index % 4 == 2:
            populations[2].append(box(x + 10, 2048, x + 12, 2052))
    collections = [
        build_feature_collection(
            [
                (
                    "Tubule",
                    mapping(
                        translate(
                            geometry,
                            -window["input_rect"]["x"],
                            -window["input_rect"]["y"],
                        )
                    ),
                )
                for geometry in population
            ],
            {"Tubule"},
        )
        for window, population in zip(plan["windows"], populations, strict=True)
    ]
    output, trace = reconcile_window_predictions(plan, collections, {"Tubule"})
    assert output is None
    assert len(trace["groups"]) == 1
    members = trace["groups"][0]["members"]
    assert len(members) == 32
    assert len(trace["fragment_decisions"]) == 8
    # Invalid chains still need readable diagnostics. Their repeated fragment
    # references must not retain another full group vector per fragment.
    assert all(
        fragment["complete_groups"][0] is members
        for fragment in trace["fragment_decisions"]
    )
