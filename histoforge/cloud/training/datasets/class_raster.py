"""Paint neural class channels on the canonical annotation raster grid."""

from collections.abc import Iterable, Mapping

import numpy as np
from shapely.geometry.base import BaseGeometry

from histoforge.cloud.training.datasets.label_rasterizer import rasterize_model_geometry


def rasterize_class_targets(
    shapes: Iterable[tuple[BaseGeometry, int]],
    rectangle: Mapping[str, int],
    support: np.ndarray,
    *,
    allow_same_class_overlap: bool,
) -> np.ndarray:
    """Background is zero; reject contradictory labels only on effective support.

    Inputs are already mapped to the same absolute model grid. Dense same-class
    annotations form a union; instance class-frequency counting rejects any
    overlap. Both TRAIN frequencies and dense targets use this single rule.
    """

    labels = np.zeros(support.shape, dtype=np.int32)
    for geometry, channel in shapes:
        mask = rasterize_model_geometry(geometry, rectangle) & support
        occupied = labels[mask]
        conflict = occupied != 0
        if allow_same_class_overlap:
            conflict &= occupied != channel
        if np.any(conflict):
            kind = "classes" if allow_same_class_overlap else "instances"
            raise ValueError(f"training target {kind} share a supervised real pixel")
        labels[mask] = channel
    return labels
