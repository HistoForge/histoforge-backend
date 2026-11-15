"""Plan TRAIN fields without decoding RGB or constructing task targets.

The plan contains only model grids and field coordinates. Concrete task
datasets read those coordinates and construct their own samples on demand.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from shapely.geometry import shape

from histoforge._vendor.histoforge_models.pathology.field_geometry import (
    fixed_field_rectangles,
    variable_field_rectangles,
)
from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_nonnegative_safe_integer,
)
from histoforge.cloud._config import validate_source_artifact_limits
from histoforge.cloud.modules.schema import VARIABLE_FIELD_PLANNER_ID
from histoforge.cloud.sources.views import source_slide_bounds
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.datasets.io import read_training_roi_source
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_training_geometry,
    training_real_pixel_mask,
)
from histoforge.cloud.training.datasets.rasters import (
    training_roi_raster,
    validate_training_raster_profile,
)
from histoforge.cloud.training.datasets.split import expand_training_split_plan
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.profiles import full_coverage_sampler_profile


@dataclass(frozen=True)
class EpochFieldInputs:
    """Spawn-safe epoch coordinates and read/augmentation settings, never pixels."""

    training_seed: int
    epoch_index: int
    augmentation: Mapping[str, Any]
    source_limits: Mapping[str, int]
    max_json_member_bytes: int
    grids: Mapping[int, Mapping[str, Any]]
    field_locations: tuple[tuple[int, tuple[int, int, int, int]], ...]


def prepare_epoch_field_inputs(
    dataset: RetainedTrainingDataset,
    split_plan: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    augmentation: Mapping[str, Any],
    training_seed: int,
    epoch_index: int,
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, int],
) -> EpochFieldInputs:
    """Bind an unsampled field population and detached settings for either task.

    This does not read pixels or select task targets. The concrete dataset owns
    its read, fixed-overhang padding and augmentation sequence at each ordinal.
    """

    training_seed = require_nonnegative_safe_integer(training_seed, "training_seed")
    epoch_index = require_nonnegative_safe_integer(epoch_index, "epoch_index")
    limits = validate_training_api_limits(api_limits)
    augmentation = canonical_tree(augmentation)
    source_limits = validate_source_artifact_limits(source_limits)
    _, grids, locations = plan_training_fields(
        dataset,
        split_plan,
        capability=capability,
        field_profile=field_profile,
        api_limits=limits,
    )
    return EpochFieldInputs(
        training_seed,
        epoch_index,
        augmentation,
        source_limits,
        limits["dataset"]["max_json_member_bytes"],
        grids,
        locations,
    )


def plan_training_fields(
    dataset: RetainedTrainingDataset,
    split_plan: Mapping[str, Any],
    *,
    capability: Mapping[str, Any],
    field_profile: Mapping[str, Any],
    api_limits: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    tuple[tuple[int, tuple[int, int, int, int]], ...],
]:
    """Return the validated profile, ROI grids and dataset/row-major fields.

    Coordinates depend only on the retained TRAIN ROIs and field profile.
    Negative fields remain; only fields with no supervised real pixel are omitted.
    No worker count, batch size, head type or GPU can change this population.
    """

    limits = validate_training_api_limits(api_limits)
    split = expand_training_split_plan(split_plan, dataset.manifest, limits)
    profile = validate_training_raster_profile(
        dataset.manifest["task"],
        capability,
        field_profile,
        limits["report"]["max_text_bytes"],
    )
    grids = {}
    locations = []
    for roi_index in split.training_roi_indices:
        geometry, origin = read_training_roi_source(
            dataset.root,
            dataset.manifest,
            roi_index,
            limits["dataset"]["max_json_member_bytes"],
        )
        target = training_roi_raster(
            shape(geometry).bounds,
            origin,
            dataset.manifest["task"],
            capability,
            profile,
        )
        grids[roi_index] = target
        bounds = source_slide_bounds(origin)
        raster = target["raster"]
        first = len(locations)
        for rectangle in _field_rectangles(raster, profile):
            read_rect, _ = bounded_field_read(rectangle, raster)
            if not read_rect["width"] or not read_rect["height"]:
                continue
            support = rasterize_training_geometry(geometry, target, read_rect)
            support &= training_real_pixel_mask(bounds, target, read_rect)
            if support.any():
                locations.append((roi_index, rectangle))
        if len(locations) == first:
            raise ValueError("training ROI has no supervised real pixels")
    return profile, grids, tuple(locations)


def epoch_random_generator(
    sampler: Mapping[str, Any], training_seed: int, epoch_index: int
) -> np.random.Generator:
    """Derive a local generator without depending on worker or global RNG state."""

    seed = int(
        canonical_digest(
            {
                "domain": sampler["parameters"]["seed_derivation"],
                "training_seed": require_nonnegative_safe_integer(
                    training_seed, "training_seed"
                ),
                "epoch_index": require_nonnegative_safe_integer(
                    epoch_index, "epoch_index"
                ),
            }
        ),
        16,
    )
    return np.random.Generator(np.random.PCG64(seed))


def permute_epoch_fields(
    inputs: EpochFieldInputs, sampler: Mapping[str, Any]
) -> EpochFieldInputs:
    """Preserve the full-coverage permutation over the canonical population."""

    if sampler != full_coverage_sampler_profile():
        raise ValueError("resolved sampler implementation is not supported")
    order = epoch_random_generator(
        sampler, inputs.training_seed, inputs.epoch_index
    ).permutation(len(inputs.field_locations))
    return replace(
        inputs,
        field_locations=tuple(inputs.field_locations[int(index)] for index in order),
    )


def _field_rectangles(raster: Mapping[str, int], profile: Mapping[str, Any]):
    """Plan optimizer fields on the natural supervised raster.

    ``outerArtificialPaddingModelPx`` belongs only to evaluation/inference target
    framing. It must not expand or shift the full-coverage training population.
    """

    if profile["planner"] == "fixed-field-v1":
        return fixed_field_rectangles(
            raster["height"],
            raster["width"],
            side=profile["side_model_px"],
            overlap=profile["tile_overlap"],
        )
    if profile["planner"] != VARIABLE_FIELD_PLANNER_ID:
        raise ValueError("unsupported variable training field planner")
    return variable_field_rectangles(
        raster["height"],
        raster["width"],
        maximum_side=profile["maxSideModelPx"],
        tile_side=profile["preferredTileSideModelPx"],
        overlap=profile["tileOverlap"],
    )


def bounded_field_read(
    rectangle: tuple[int, int, int, int], raster: Mapping[str, int]
) -> tuple[dict[str, int], tuple[int, int, int, int]]:
    """Intersect a model field with its input, preserving fixed-field overhang."""

    x, y, width, height = rectangle
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(raster["width"], x + width), min(raster["height"], y + height)
    return (
        {"x": x0, "y": y0, "width": max(0, x1 - x0), "height": max(0, y1 - y0)},
        (x0 - x, x + width - x1, y0 - y, y + height - y1),
    )
