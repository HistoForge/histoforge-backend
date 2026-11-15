"""Convert semantic labels and assemble their exact pixel-edge regions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np
from scipy import ndimage
from shapely import union_all
from shapely.geometry import mapping, shape

from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.inference.features import ResultLimitExceeded
from histoforge.cloud.inference.mask_features import feature_collection_from_mask


def feature_collection_from_class_map(
    class_map: np.ndarray,
    class_by_channel: Mapping[int, str],
    declared_classes: Collection[str],
    result_budget: Mapping[str, int],
    *,
    x_offset: int = 0,
    y_offset: int = 0,
) -> dict[str, Any]:
    """Emit one feature per four-connected region of an emitted winning class.

    The class map already contains the argmax over all class-head channels. Classes
    absent from ``class_by_channel`` remain excluded; they are never reassigned.
    Offsets place a core crop back into its supplied processing image.
    """

    region_labels = np.zeros_like(class_map, dtype=np.int32)
    classifications: dict[int, str] = {}
    next_id = 1
    for class_id, path_class in class_by_channel.items():
        foreground = class_map == class_id
        if not foreground.any():
            continue
        labeled, count = ndimage.label(foreground)
        # Rasterio creates its native polygon layer before yielding its first
        # geometry, so this count limit must run before polygonization starts.
        if next_id - 1 + count > result_budget["max_features"]:
            raise ResultLimitExceeded("inference result exceeds max_features")
        np.add(labeled, next_id - 1, out=labeled, where=foreground)
        np.copyto(region_labels, labeled, where=foreground)
        classifications.update(
            (region_id, path_class) for region_id in range(next_id, next_id + count)
        )
        next_id += count
    return feature_collection_from_mask(
        region_labels,
        classifications,
        declared_classes,
        result_budget,
        x_offset=x_offset,
        y_offset=y_offset,
    )


def assemble_semantic_fragments(
    fragments: Iterable[tuple[str, Mapping[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    """Join same-class core fragments into separate edge-connected polygons.

    The caller has validated the canonical geometries and bounded their aggregate
    size. Exact union joins shared edges, leaves corner-only contacts separate,
    and preserves holes. No simplification, repair, or coordinate rounding occurs.
    Final result limits belong to the caller after regions have been assembled.
    """

    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for path_class, geometry in fragments:
        by_class[path_class].append(geometry)

    assembled = []
    for path_class, geometries in sorted(by_class.items()):
        # Fix union input order as well as the returned order so recovery artifact
        # iteration cannot change the representation of the same fragments.
        geometries.sort(key=canonical_bytes)
        merged = union_all([shape(geometry) for geometry in geometries])
        polygons = [merged] if merged.geom_type == "Polygon" else merged.geoms
        canonical_polygons = [
            canonical_geometry(mapping(polygon)) for polygon in polygons
        ]
        canonical_polygons.sort(key=canonical_bytes)
        assembled.extend((path_class, polygon) for polygon in canonical_polygons)
    return assembled
