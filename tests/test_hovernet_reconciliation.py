"""Complete HoVer window selection and partial-observation regression fixtures."""

from copy import deepcopy

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.architectures.hovernet.reconciliation import (
    _artificial_input_edges,
    _partial_observation_failure,
    reconcile_window_predictions,
)
from histoforge.cloud.inference.features import build_feature_collection
from histoforge.cloud.inference.plans import (
    build_core_halo_profile,
    build_expected_model_execution_plan,
)
from shapely.affinity import translate
from shapely.geometry import MultiPolygon, Polygon, box, mapping
from shapely.ops import unary_union

CLASSES = ("Glomerulus", "Other")


def _plan(width=192, height=96, core=64, halo=16):
    return build_expected_model_execution_plan(
        build_core_halo_profile(core, min(core, height), halo, "a" * 64),
        {"width": width, "height": height},
        "b" * 64,
        64,
    )


def _collections(plan, populations):
    return [
        build_feature_collection(
            [
                (name, mapping(translate(geometry, -rect["x"], -rect["y"])))
                for name, geometry in population
            ],
            CLASSES,
        )
        for window, population in zip(plan["windows"], populations, strict=True)
        for rect in [window["input_rect"]]
    ]


def _offset_coordinates(coordinates, x, y):
    if isinstance(coordinates[0], (float, int)):
        return [coordinates[0] + x, coordinates[1] + y]
    return [_offset_coordinates(part, x, y) for part in coordinates]


def _run(plan, populations):
    values = _collections(plan, populations)
    original_values = deepcopy(values)
    output, trace = reconcile_window_predictions(plan, values, CLASSES)
    assert values == original_values
    repeat, repeat_trace = reconcile_window_predictions(
        plan,
        _collections(plan, [list(reversed(p)) for p in populations]),
        list(reversed(CLASSES)),
    )
    assert canonical_bytes(repeat) == canonical_bytes(output)
    assert canonical_bytes(repeat_trace) == canonical_bytes(trace)
    assert trace["accepted"] == (output is not None) == (not trace["failures"])
    if output is not None:
        originals = {
            f"{window['unit_id']}:{feature['id']}": (
                feature["properties"]["path_class"],
                {
                    "type": feature["geometry"]["type"],
                    "coordinates": _offset_coordinates(
                        feature["geometry"]["coordinates"],
                        window["input_rect"]["x"],
                        window["input_rect"]["y"],
                    ),
                },
            )
            for window, collection in zip(plan["windows"], values, strict=True)
            for feature in collection["features"]
        }
        info = {
            candidate["key"]: candidate for candidate in trace["candidate_evidence"]
        }
        chosen = []
        for decision in trace["groups"]:
            key = decision["provisional_winner"]
            assert not info[key]["artificial_edge"]
            chosen.append(originals[key])
        assert canonical_bytes(output) == canonical_bytes(
            build_feature_collection(chosen, CLASSES)
        )
    return output, trace


def _g(*geometries):
    return [(CLASSES[0], geometry) for geometry in geometries]


@pytest.mark.parametrize("mode", ["zero", "four"])
def test_one_complete_outline_survives_a_four_core_ownership_disagreement(mode):
    plan = _plan(128, 128, 64, 48)
    centers = (
        [(65, 65), (63, 65), (65, 63), (63, 63)]
        if mode == "zero"
        else [(63, 63), (65, 63), (63, 65), (65, 65)]
    )
    output, trace = _run(
        plan, [_g(box(x - 10, y - 10, x + 10, y + 10)) for x, y in centers]
    )
    assert len(output["features"]) == 1
    assert trace["groups"][0]["owner_count"] == (0 if mode == "zero" else 4)
    assert trace["groups"][0]["provisional_winner"].startswith("0:")


@pytest.mark.parametrize("kind", ["polygon", "hole", "concave", "multipart"])
def test_existing_single_owner_retains_its_whole_geometry(kind):
    plan = _plan(128, 64, 64, 48)
    geometries = {
        "polygon": box(40, 10, 90, 55),
        "hole": Polygon(
            [(40, 10), (90, 10), (90, 55), (40, 55)],
            [[(50, 20), (80, 20), (80, 40), (50, 40)]],
        ),
        "concave": Polygon(
            [(40, 10), (90, 10), (90, 30), (57, 30), (57, 55), (40, 55)]
        ),
        "multipart": MultiPolygon([box(42, 10, 52, 22), box(69, 35, 80, 55)]),
    }
    geometry = geometries[kind]
    output, trace = _run(plan, [_g(geometry), _g(geometry)])
    assert len(output["features"]) == 1
    assert trace["groups"][0]["selection"] == "preserve_single_owner"
    assert canonical_bytes(output) == canonical_bytes(
        build_feature_collection([(CLASSES[0], mapping(geometry))], CLASSES)
    )


@pytest.mark.parametrize("fragment_class", CLASSES)
def test_excluded_fragment_cannot_invent_a_missing_complete_counterpart(fragment_class):
    plan = _plan(128, 64, 64, 16)
    output, trace = _run(
        plan,
        [
            _g(box(52, 20, 70, 40)),
            [(fragment_class, box(48, 20, 60, 40))],
        ],
    )
    assert output is None
    assert "missing_core_counterpart" in {x["reason"] for x in trace["failures"]}


def test_ordinary_partial_copy_is_excluded_without_changing_complete_output():
    plan = _plan(192, 64)
    complete = box(45, 20, 57, 50)
    output, trace = _run(plan, [_g(complete), _g(box(48, 20, 57, 50)), []])
    assert len(output["features"]) == 1
    assert trace["fragment_decisions"][0]["reasons"] == []
    assert canonical_bytes(output) == canonical_bytes(
        build_feature_collection([(CLASSES[0], mapping(complete))], CLASSES)
    )


@pytest.mark.parametrize("redundant_vertex", [False, True])
def test_cut_intervals_do_not_depend_on_redundant_polygon_vertices(redundant_vertex):
    plan = _plan(192, 64)
    fragment = (
        Polygon([(48, 20), (58, 20), (58, 30), (48, 30), (48, 24)])
        if redundant_vertex
        else box(48, 20, 58, 30)
    )
    complete = unary_union([box(48, 20, 58, 30), box(44, 20, 50, 24)])
    output, trace = _run(plan, [_g(complete), _g(fragment), []])
    assert output is not None
    assert trace["fragment_decisions"][0]["reasons"] == []
    assert len(output["features"]) == 1


@pytest.mark.parametrize(
    "mode", ["merged-fragment", "split-fragments", "exact-two-object-control"]
)
def test_partial_observations_do_not_conceal_split_merge_disagreement(mode):
    plan = _plan(192, 64)
    full = box(45, 20, 57, 50)
    parts = [box(45, 20, 57, 38), box(45, 38, 57, 50)]
    partial = [geometry.intersection(box(48, 0, 144, 64)) for geometry in parts]
    if mode == "merged-fragment":
        populations = [_g(*parts), _g(full.intersection(box(48, 0, 144, 64))), []]
    elif mode == "split-fragments":
        populations = [_g(full), _g(*partial), []]
    else:
        populations = [_g(*parts), _g(*partial), []]
    output, trace = _run(plan, populations)
    assert trace["accepted"] == (mode == "exact-two-object-control")
    if output is not None:
        assert len(output["features"]) == 2


def test_cross_core_split_is_visible_even_when_area_and_class_are_unchanged():
    plan = _plan(128, 128, 64, 16)
    output, trace = _run(
        plan,
        [
            _g(box(58, 44, 64, 54)),
            _g(box(64, 44, 68, 54)),
            _g(box(58, 48, 68, 54)),
            [],
        ],
    )
    # These same predictions could be two correct objects. A partial observation
    # cannot choose that biological interpretation; the whole input is withheld.
    assert output is None
    assert "fragment_overlaps_multiple_groups" in {
        x["reason"] for x in trace["failures"]
    }


def test_irrelevant_halo_group_remains_an_ambiguity_challenger():
    plan = _plan(128, 128, 64, 16)
    output, trace = _run(
        plan,
        [
            _g(box(52, 44, 58, 54)),
            [(CLASSES[1], box(58, 44, 62, 54))],
            _g(box(52, 48, 62, 54)),
            [],
        ],
    )
    assert output is None
    assert "fragment_overlaps_multiple_groups" in {
        x["reason"] for x in trace["failures"]
    }
    # The diagnostic must name both groups, including the one omitted from the
    # relevant-output decisions. Stable candidate keys make it self-contained.
    assert len(trace["groups"]) == 1
    competing = trace["fragment_decisions"][0]["complete_groups"]
    assert len(competing) == 2
    known = {item["key"] for item in trace["candidate_evidence"]}
    assert all(key in known for group in competing for key in group)


def test_core_veto_is_not_reported_as_an_explained_halo_observation():
    plan = _plan(192, 64)
    output, trace = _run(
        plan,
        [
            _g(box(45, 20, 63, 30)),
            _g(box(48, 20, 65, 30)),
            [],
        ],
    )
    assert output is None
    assert "fragment_affects_core" in {item["reason"] for item in trace["failures"]}
    assert trace["fragment_decisions"] == []


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize("reach", ["halo", "boundary", "core"])
def test_artificial_fragments_reaching_their_core_always_fail(side, reach):
    plan = _plan(192, 192)
    edge = {"halo": 60, "boundary": 64, "core": 65}[reach]
    geometries = {
        "left": box(48, 80, edge, 90),
        "right": box(192 - edge, 80, 144, 90),
        "top": box(80, 48, 90, edge),
        "bottom": box(80, 192 - edge, 90, 144),
    }
    populations = [[] for _ in plan["windows"]]
    populations[4] = _g(geometries[side])
    output, trace = _run(plan, populations)
    assert trace["accepted"] == (reach != "core")
    if output is not None:
        assert output["features"] == []


def test_partial_class_disagreement_cannot_change_a_complete_class():
    plan = _plan(192, 64)
    output, trace = _run(
        plan,
        [
            _g(box(45, 20, 57, 50)),
            [(CLASSES[1], box(48, 20, 57, 50))],
            [],
        ],
    )
    assert output is None
    assert "fragment_class_conflict" in {x["reason"] for x in trace["failures"]}


def _partial_cases():
    source = box(0, 0, 100, 100)
    fragment = box(0, 20, 10, 30)
    complete = box(-10, 20, 10, 30)
    outside = box(30, 110, 40, 120)
    island = box(70, 70, 71, 71)
    multi_contact = unary_union(
        [box(0, 20, 20, 30), box(0, 60, 20, 61), box(15, 20, 20, 61)]
    )
    partial_continuation = unary_union([multi_contact, box(-5, 20, 5, 30)])
    hole = Polygon(
        [(-10, 10), (20, 10), (20, 40), (-10, 40)],
        [[(-5, 20), (5, 20), (5, 25), (-5, 25)]],
    )
    yield "exact", fragment, complete, None
    yield "boundary-only", fragment, fragment, "fragment_cut_not_explained"
    yield (
        "wrong-side",
        fragment,
        MultiPolygon([fragment, outside]),
        "fragment_cut_not_explained",
    )
    yield (
        "unrelated-same-side",
        fragment,
        MultiPolygon([fragment, box(-20, 75, -10, 85)]),
        "fragment_cut_not_explained",
    )
    yield (
        "off-window-component-allowed",
        fragment,
        MultiPolygon([complete, outside]),
        None,
    )
    yield (
        "unexplained-fragment-island",
        MultiPolygon([fragment, island]),
        complete,
        "multipart_partial_observation",
    )
    yield (
        "unexplained-complete-island",
        fragment,
        MultiPolygon([complete, island]),
        "multipart_partial_observation",
    )
    yield (
        "every-contact-interval",
        multi_contact,
        partial_continuation,
        "fragment_cut_not_explained",
    )
    yield (
        "point-contact",
        Polygon([(0, 20), (10, 10), (20, 20), (10, 30)]),
        Polygon([(-1, 20), (10, 10), (20, 20), (10, 30)]),
        "fragment_cut_not_explained",
    )
    yield "hole-opened-by-cut", hole.intersection(source), hole, None
    yield "half-iou", box(0, 20, 20, 30), complete, "fragment_footprint_disagreement"


@pytest.mark.parametrize(
    "_name,fragment,complete,reason",
    list(_partial_cases()),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_partial_footprint_and_actual_cut_witness(_name, fragment, complete, reason):
    rect = {"x": 100, "y": 100, "width": 100, "height": 100}
    result = _partial_observation_failure(
        translate(fragment, 100, 100),
        translate(complete, 100, 100),
        box(100, 100, 200, 200),
        _artificial_input_edges(rect, 300, 300),
    )
    assert result == reason
