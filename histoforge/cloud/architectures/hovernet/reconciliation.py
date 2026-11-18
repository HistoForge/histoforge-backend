"""Select intact HoVer instances from complete, bounded window observations.

This CPU owner consumes decoded polygons, not images or networks. It rejects
unresolved correspondence rather than publishing a partial biological population.
Geometric agreement does not certify biological identity or prediction accuracy.
"""

from shapely.affinity import translate
from shapely.geometry import LineString, box, mapping, shape
from shapely.ops import linemerge, unary_union
from shapely.strtree import STRtree

from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
    validate_feature_collection,
)
from histoforge.cloud.inference.plans import (
    validate_model_execution_plan,
    validate_processing_window,
)
from histoforge.cloud.inference.runs.core_halo import (
    InsufficientCoreHaloContext,
    _require_supplied_geometry,
    _require_window_inside_target,
)

POLICY = "majority-clique-partial-observations-v3"
# Build-owned work guard, including bbox hits that have zero polygon overlap.
# This provisional ceiling still needs resource qualification before activation.
MAX_SPATIAL_QUERY_HITS = 1_000_000


def reconcile_window_predictions(
    plan,
    window_collections,
    declared_classes,
    *,
    max_spatial_query_hits=MAX_SPATIAL_QUERY_HITS,
):
    """Return one complete population, or None plus unresolved-observation evidence.

    Each collection uses its processing window's local pixel-edge coordinates.
    All windows must be present. The caller owns per-unit and aggregate decoded
    population budgets; this owner separately bounds indexed spatial work.
    No population is returned if any relevant group is unresolved.
    """
    plan = validate_model_execution_plan(plan)
    if plan["type"] != "core-halo":
        raise ValueError("reconciliation requires a core-halo plan")
    if len(plan["windows"]) != len(window_collections):
        raise ValueError("one decoded collection is required for every plan window")
    candidates, original_owned_count, cores = _collect_candidates(
        plan, window_collections, declared_classes
    )
    spatial_budget = {"limit": max_spatial_query_hits, "used": 0}
    overlaps = _pairwise_overlaps(candidates, spatial_budget)
    groups, strong_degrees = _complete_groups(candidates, overlaps)
    # Reuse group evidence when many fragments refer to the same population.
    # Copying these vectors per fragment can grow quadratically on a failed input.
    group_keys = [[candidates[i]["key"] for i in group] for group in groups]
    core_index = STRtree(cores)
    group_of = {
        index: group for group, members in enumerate(groups) for index in members
    }
    relevant = [any(candidates[i]["relevant"] for i in group) for group in groups]
    units = [{candidates[i]["unit_id"] for i in group} for group in groups]
    correspondence_failures = []
    for members, member_units in zip(groups, units, strict=True):
        reasons = []
        if len(member_units) != len(members):
            reasons.append("multiple_candidates_from_one_window")
        if sum(strong_degrees[i] for i in members) != len(members) * (len(members) - 1):
            reasons.append("non_clique_correspondence")
        if len({candidates[i]["path_class"] for i in members}) != 1:
            reasons.append("class_conflict")
        correspondence_failures.append(reasons)
    failures = []
    decisions = []
    winners = []
    valid_groups = set()

    for group_id, members in enumerate(groups):
        if not relevant[group_id]:
            continue
        group = [candidates[i] for i in members]
        member_keys = group_keys[group_id]
        reasons = list(correspondence_failures[group_id])
        required_units = {
            unit_id
            for item in group
            for unit_id in _spatial_hits(core_index, item["geometry"], spatial_budget)
            if item["geometry"].intersection(cores[unit_id]).area > 0
        }
        if missing := required_units - units[group_id]:
            reasons.append("missing_core_counterpart")
        owners = [item for item in group if item["owned"]]
        winner = owners[0] if len(owners) == 1 else min(group, key=_winner_rank)
        decisions.append(
            {
                "members": member_keys,
                "owner_count": len(owners),
                "required_units": sorted(required_units),
                "missing_units": sorted(missing),
                "provisional_winner": winner["key"],
                "selection": "preserve_single_owner"
                if len(owners) == 1
                else "context_clearance",
                "reasons": reasons,
            }
        )
        winners.append(winner)
        if not reasons:
            valid_groups.add(group_id)
        failures.extend(
            {"reason": reason, "candidates": member_keys} for reason in reasons
        )

    fragment_groups = {}
    for (i, j), iou in overlaps.items():
        first, second = candidates[i], candidates[j]
        if first["artificial_edge"] or second["artificial_edge"]:
            fragment, complete = (i, j) if first["artificial_edge"] else (j, i)
            if complete in group_of:
                fragment_groups.setdefault(fragment, set()).add(group_of[complete])
            continue
        a, b = group_of[i], group_of[j]
        if a == b or not (relevant[a] or relevant[b]):
            continue
        # Small overlaps between two independently established correspondences
        # are allowed. A leftover split branch has no counterpart in one window.
        if not (
            not correspondence_failures[a]
            and not correspondence_failures[b]
            and second["unit_id"] in units[a]
            and first["unit_id"] in units[b]
        ):
            failures.append(
                {
                    "reason": "unresolved_cross_group_overlap",
                    "candidates": [first["key"], second["key"]],
                    "iou": iou,
                }
            )
    for item in candidates:
        if item["artificial_edge"] and item["relevant"]:
            failures.append(
                {
                    "reason": "owned_artificial_edge"
                    if item["owned"]
                    else "fragment_affects_core",
                    "candidates": [item["key"]],
                }
            )

    fragment_decisions = []
    target_width, target_height = cores[-1].bounds[2:]
    for fragment_index, matching_groups in sorted(fragment_groups.items()):
        fragment = candidates[fragment_index]
        # Core-intersecting fragments already have an authoritative failure
        # above. Partial-observation checks concern only discarded halo data.
        if fragment["relevant"] or not any(
            relevant[group_id] for group_id in matching_groups
        ):
            continue
        reasons = []
        if len(matching_groups) != 1:
            reasons.append("fragment_overlaps_multiple_groups")
        else:
            group_id = next(iter(matching_groups))
            if group_id not in valid_groups:
                reasons.append("fragment_overlaps_invalid_group")
            else:
                window = plan["windows"][fragment["unit_id"]]
                rect = window["input_rect"]
                source = box(
                    rect["x"],
                    rect["y"],
                    rect["x"] + rect["width"],
                    rect["y"] + rect["height"],
                )
                edges = _artificial_input_edges(rect, target_width, target_height)
                for member in groups[group_id]:
                    complete = candidates[member]
                    if fragment["path_class"] != complete["path_class"]:
                        reasons.append("fragment_class_conflict")
                        break
                    reason = _partial_observation_failure(
                        fragment["geometry"], complete["geometry"], source, edges
                    )
                    if reason:
                        reasons.append(reason)
                        break
        fragment_decisions.append(
            {
                "fragment": fragment["key"],
                "complete_groups": [
                    group_keys[group_id] for group_id in sorted(matching_groups)
                ],
                "reasons": reasons,
            }
        )
        failures.extend(
            {"reason": reason, "candidates": [fragment["key"]]} for reason in reasons
        )

    collection = (
        None
        if failures
        else build_feature_collection(
            [(item["path_class"], mapping(item["geometry"])) for item in winners],
            declared_classes,
        )
    )
    selected = {item["key"] for item in winners}
    winner_overlaps = [
        {
            "candidates": [candidates[i]["key"], candidates[j]["key"]],
            "area_model_px2": candidates[i]["geometry"]
            .intersection(candidates[j]["geometry"])
            .area,
        }
        for i, j in overlaps
        if candidates[i]["key"] in selected and candidates[j]["key"] in selected
    ]
    return collection, {
        "policy": POLICY,
        "accepted": not failures,
        "decoded_candidate_count": len(candidates),
        "original_owned_count": original_owned_count,
        "spatial_query_hits": spatial_budget["used"],
        "output_count": None if collection is None else len(collection["features"]),
        "failures": failures,
        "groups": decisions,
        "fragment_decisions": fragment_decisions,
        "provisional_winner_overlaps": winner_overlaps,
        "candidate_evidence": [
            {
                key: item[key]
                for key in (
                    "key",
                    "unit_id",
                    "path_class",
                    "owned",
                    "relevant",
                    "artificial_edge",
                    "context_clearance",
                )
            }
            for item in candidates
        ],
    }


def _collect_candidates(plan, collections, declared_classes):
    windows = plan["windows"]
    width = max(w["core_rect"]["x"] + w["core_rect"]["width"] for w in windows)
    height = max(w["core_rect"]["y"] + w["core_rect"]["height"] for w in windows)
    candidates, cores = [], []
    for window, value in zip(windows, collections, strict=True):
        core = window["core_rect"]
        core_geometry = box(
            core["x"], core["y"], core["x"] + core["width"], core["y"] + core["height"]
        )
        cores.append(core_geometry)
        candidates.extend(
            _window_candidates(window, value, declared_classes, width, height)
        )
    return candidates, sum(item["owned"] for item in candidates), cores


def _window_candidates(window, value, declared_classes, width, height):
    window = validate_processing_window(window)
    _require_window_inside_target(window, width, height)
    collection = validate_feature_collection(value, declared_classes)
    rect, core = window["input_rect"], window["core_rect"]
    core_geometry = box(
        core["x"], core["y"], core["x"] + core["width"], core["y"] + core["height"]
    )
    for feature in sorted(collection["features"], key=lambda item: item["id"]):
        local = shape(feature["geometry"])
        _require_supplied_geometry(local, rect)
        point = local.representative_point()
        x, y = point.x + rect["x"], point.y + rect["y"]
        is_owned = (
            core["x"] <= x < core["x"] + core["width"]
            and core["y"] <= y < core["y"] + core["height"]
        )
        geometry = translate(local, rect["x"], rect["y"])
        left, top, right, bottom = local.bounds
        clearances = []
        if rect["x"] > 0:
            clearances.append(left)
        if rect["y"] > 0:
            clearances.append(top)
        if rect["x"] + rect["width"] < width:
            clearances.append(rect["width"] - right)
        if rect["y"] + rect["height"] < height:
            clearances.append(rect["height"] - bottom)
        clearance = min(clearances) if clearances else None
        yield {
            "key": f"{window['unit_id']}:{feature['id']}",
            "unit_id": window["unit_id"],
            "geometry": geometry,
            "path_class": feature["properties"]["path_class"],
            "owned": is_owned,
            "relevant": is_owned or geometry.intersection(core_geometry).area > 0,
            "artificial_edge": clearance is not None and clearance <= 0,
            "context_clearance": clearance,
        }


def validate_window_observations(window, collection, declared_classes, target_raster):
    """Reject locally decidable context failures before a unit becomes reusable."""
    for candidate in _window_candidates(
        window,
        collection,
        declared_classes,
        target_raster["width"],
        target_raster["height"],
    ):
        if candidate["artificial_edge"] and candidate["relevant"]:
            raise InsufficientCoreHaloContext(
                "A predicted object reaches the edge of the supplied context. "
                "Increase the context in Inference settings and start a new analysis, "
                "or use a smaller region."
            )


def _spatial_hits(index, geometry, budget):
    hits = index.query(geometry)
    budget["used"] += len(hits)
    if budget["used"] > budget["limit"]:
        raise ResultLimitExceeded(
            "HoVer reconciliation exceeds spatial query work limit"
        )
    return sorted(int(hit) for hit in hits)


def _pairwise_overlaps(candidates, spatial_budget):
    overlaps = {}
    index = STRtree([item["geometry"] for item in candidates])
    for i, first in enumerate(candidates):
        for j in _spatial_hits(index, first["geometry"], spatial_budget):
            if j <= i:
                continue
            second = candidates[j]
            intersection = first["geometry"].intersection(second["geometry"]).area
            if intersection == 0:
                continue
            if first["unit_id"] == second["unit_id"]:
                raise ValueError("decoded instances from one window overlap in area")
            overlaps[i, j] = intersection / (
                first["geometry"].area + second["geometry"].area - intersection
            )
    return overlaps


def _complete_groups(candidates, overlaps):
    neighbors = {i: set() for i, c in enumerate(candidates) if not c["artificial_edge"]}
    for (i, j), iou in overlaps.items():
        if i in neighbors and j in neighbors and iou > 0.5:
            neighbors[i].add(j)
            neighbors[j].add(i)
    seen, groups = set(), []
    for start in neighbors:
        if start in seen:
            continue
        component, todo = set(), [start]
        while todo:
            index = todo.pop()
            if index not in component:
                component.add(index)
                todo.extend(neighbors[index] - component)
        seen.update(component)
        groups.append(sorted(component))
    return groups, {index: len(values) for index, values in neighbors.items()}


def _winner_rank(candidate):
    # No artificial edge is best; otherwise maximize the nearest edge clearance.
    # The captured unit ID resolves exact ties without favoring class or area.
    clearance = candidate["context_clearance"]
    return (clearance is not None, -(clearance or 0), candidate["unit_id"])


def _geometry_parts(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type in {
        "MultiPolygon",
        "MultiLineString",
        "MultiPoint",
        "GeometryCollection",
    }:
        for part in geometry.geoms:
            yield from _geometry_parts(part)
    else:
        yield geometry


def _cut_intervals(contact):
    """Join contiguous cut segments without changing the predicted polygon.

    GEOS can split a straight contact at a redundant ring vertex. Those pieces
    describe one interval. Preserve isolated point contacts for rejection.
    """

    parts = list(_geometry_parts(contact))
    lines = [part for part in parts if part.length > 0]
    if not lines:
        yield from parts
        return
    merged = linemerge(list(_geometry_parts(unary_union(lines))))
    yield from _geometry_parts(merged)
    yield from (part for part in parts if part.length == 0 and part.disjoint(merged))


def _artificial_input_edges(rect, target_width, target_height):
    left, top = rect["x"], rect["y"]
    right, bottom = left + rect["width"], top + rect["height"]
    return [
        LineString(points)
        for artificial, points in (
            (left > 0, ((left, top), (left, bottom))),
            (right < target_width, ((right, top), (right, bottom))),
            (top > 0, ((left, top), (right, top))),
            (bottom < target_height, ((left, bottom), (right, bottom))),
        )
        if artificial
    ]


def _partial_observation_failure(fragment, complete, source, artificial_edges):
    """Check one discarded partial observation without supplying output geometry.

    Work in global model-pixel coordinates. Restrict this exception to one
    visible polygon on each side; multiple visible pieces need an explicit
    failure rather than a guessed component matching. Holes remain meaningful.
    """

    fragment_parts = [part for part in _geometry_parts(fragment) if part.area > 0]
    visible = complete.intersection(source)
    visible_parts = [part for part in _geometry_parts(visible) if part.area > 0]
    if len(fragment_parts) != 1 or len(visible_parts) != 1:
        return "multipart_partial_observation"
    intersection = fragment.intersection(visible).area
    union = fragment.area + visible.area - intersection
    if union <= 0 or intersection / union <= 0.5:
        return "fragment_footprint_disagreement"
    parent = next(
        part
        for part in _geometry_parts(complete)
        if part.intersection(visible_parts[0]).area > 0
    )
    touched = False
    for edge in artificial_edges:
        for contact in _cut_intervals(fragment_parts[0].intersection(edge)):
            touched = True
            if (
                contact.length <= 0
                or contact.intersection(parent).difference(parent.boundary).length <= 0
            ):
                return "fragment_cut_not_explained"
    return None if touched else "fragment_cut_not_explained"
