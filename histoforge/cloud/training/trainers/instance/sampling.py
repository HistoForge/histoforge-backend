"""Full-coverage or class-oversampled fields from a retained training dataset.

The worker owns the dataset reference, accepted model settings, and DataLoader.
This map-style dataset owns a deterministic field list and per-sample reads.
It retains coordinates, never decoded ROI images or open source handles.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from itertools import groupby
from typing import Any

import numpy as np
from shapely.geometry import box
from shapely.strtree import STRtree
from torch.utils.data import Dataset

from histoforge._vendor.histoforge_models.pathology.managed_training import (
    HeadType,
    ManagedSegmentationTargets,
)
from histoforge.canonical_json import canonical_tree
from histoforge.cloud.modules.schema import model_output_target_channel_by_class
from histoforge.cloud.sources.views import source_slide_bounds
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.datasets.io import read_training_roi_labels
from histoforge.cloud.training.datasets.label_rasterizer import (
    rasterize_model_geometry,
    training_geometry_on_model_grid,
    training_real_pixel_mask,
)
from histoforge.cloud.training.fields.sampling import (
    EpochFieldInputs,
    bounded_field_read,
    epoch_random_generator,
    permute_epoch_fields,
    prepare_epoch_field_inputs,
)
from histoforge.cloud.training.profiles import instance_sampler_profile
from histoforge.cloud.training.trainers.instance.augmentation import (
    apply_training_augmentation,
)
from histoforge.cloud.training.trainers.instance.field_reader import (
    TrainingField,
    read_training_field,
)
from histoforge.cloud.training.trainers.instance.field_targets import (
    TrainingFieldTargets,
)


class TrainingEpochDataset(Dataset[TrainingField]):
    """Read the same ordered, augmented fields regardless of worker or batch count.

    Each grid derives from the retained ROI, source calibration, and accepted
    field profile, including only its permitted real context. A new instance
    represents one zero-based epoch. Use DataLoader with ``shuffle=False`` and
    ``drop_last=False``; this dataset has already ordered its draws. Epoch copies
    reuse only the canonical coordinates and optional sampling probabilities.
    All tensors and generated flow/HoVer targets remain on CPU in data workers.
    """

    def __init__(
        self,
        dataset: RetainedTrainingDataset,
        split_plan: Mapping[str, Any],
        *,
        capability: Mapping[str, Any],
        field_profile: Mapping[str, Any],
        sampler: Mapping[str, Any],
        augmentation: Mapping[str, Any],
        training_seed: int,
        epoch_index: int,
        head_type: HeadType,
        api_limits: Mapping[str, Any],
        source_limits: Mapping[str, int],
        class_pixel_counts: Mapping[int, int] | None = None,
    ) -> None:
        if head_type not in {"flow", "hovernet"}:
            raise ValueError("training sampler requires a flow or HoVer head")
        self.dataset = dataset
        self.head_type = head_type
        self.sampler = canonical_tree(sampler)
        oversample = self.sampler == instance_sampler_profile(True)
        if not oversample and self.sampler != instance_sampler_profile(False):
            raise ValueError("resolved sampler implementation is not supported")
        if oversample and (
            dataset.manifest["task"]["model_output"]["type"]
            != "instance_segmentation_and_classification"
            or class_pixel_counts is None
        ):
            raise ValueError("oversampling requires instance classes and TRAIN counts")
        self._population = prepare_epoch_field_inputs(
            dataset,
            split_plan,
            capability=capability,
            field_profile=field_profile,
            augmentation=augmentation,
            training_seed=training_seed,
            epoch_index=epoch_index,
            api_limits=api_limits,
            source_limits=source_limits,
        )
        self.probabilities = (
            _field_sampling_probabilities(
                dataset,
                self._population,
                class_pixel_counts,
                self.sampler["parameters"]["score_floor"],
            )
            if oversample
            else None
        )
        self.inputs = self._epoch_inputs(epoch_index)

    def for_epoch(self, epoch_index: int) -> TrainingEpochDataset:
        """Make a spawn-safe epoch snapshot without rereading labels or RGB."""

        epoch = copy(self)
        epoch.inputs = self._epoch_inputs(epoch_index)
        return epoch

    def _epoch_inputs(self, epoch_index: int) -> EpochFieldInputs:
        inputs = replace(self._population, epoch_index=epoch_index)
        if self.probabilities is None:
            return permute_epoch_fields(inputs, self.sampler)
        count = len(inputs.field_locations)
        order = epoch_random_generator(
            self.sampler, inputs.training_seed, epoch_index
        ).choice(count, size=count, replace=True, p=self.probabilities)
        return replace(
            inputs,
            field_locations=tuple(inputs.field_locations[int(i)] for i in order),
        )

    def __len__(self) -> int:
        return len(self.inputs.field_locations)

    def __getitem__(self, index: int) -> TrainingField:
        """Read a field and augment by its epoch ordinal, never by a worker RNG."""

        inputs = self.inputs
        roi_index, rectangle = inputs.field_locations[index]
        target = inputs.grids[roi_index]
        read_rect, padding = bounded_field_read(rectangle, target["raster"])
        field = read_training_field(
            self.dataset,
            roi_index,
            raster_transform=target,
            output_rect=read_rect,
            head_type=self.head_type,
            source_limits=inputs.source_limits,
            max_json_member_bytes=inputs.max_json_member_bytes,
        )
        if any(padding):
            field = _pad_fixed_field(field, padding)
        return apply_training_augmentation(
            field,
            inputs.augmentation,
            training_seed=inputs.training_seed,
            epoch_index=inputs.epoch_index,
            sample_index=index,
        )


def _field_sampling_probabilities(
    dataset: RetainedTrainingDataset,
    population: EpochFieldInputs,
    class_pixel_counts: Mapping[int, int],
    score_floor: float,
) -> np.ndarray:
    """Score raster-visible instances, never whole-ROI counts or contextual labels.

    The admitted original TRAIN pixel counts already live in objective evidence.
    Overlapping fields repeat visibility, not those denominators. One ROI's
    geometry and one bounded field mask are resident; no RGB or flow/HV targets.
    """

    class_indices = model_output_target_channel_by_class(
        dataset.manifest["task"]["model_output"]
    )
    channels = sorted(class_indices.values())
    inverse_counts = {c: 1.0 / max(class_pixel_counts[c], 1) for c in channels}
    scores = []
    for roi_index, locations in groupby(population.field_locations, key=lambda f: f[0]):
        roi, objects, origin = read_training_roi_labels(
            dataset.root,
            dataset.manifest,
            roi_index,
            population.max_json_member_bytes,
        )
        target = population.grids[roi_index]
        roi_on_grid = training_geometry_on_model_grid(roi, target)
        geometries = [
            training_geometry_on_model_grid(item["geometry"], target)
            for item in objects
        ]
        object_classes = [
            class_indices[item["properties"]["path_class"]] for item in objects
        ]
        tree = STRtree(geometries)
        bounds = source_slide_bounds(origin)
        for _, rectangle in locations:
            field, _ = bounded_field_read(rectangle, target["raster"])
            x, y = field["x"], field["y"]
            window = box(x, y, x + field["width"], y + field["height"])
            support = rasterize_model_geometry(roi_on_grid, field)
            support &= training_real_pixel_mask(bounds, target, field)
            counts = dict.fromkeys(channels, 0)
            for index in tree.query(window, predicate="intersects"):
                visible = rasterize_model_geometry(geometries[index], field) & support
                if visible.any():
                    counts[object_classes[index]] += 1
            scores.append(sum(counts[c] * inverse_counts[c] for c in channels))
    probabilities = np.maximum(np.asarray(scores, dtype=np.float64), score_floor)
    probabilities /= probabilities.sum()
    return probabilities


def _pad_fixed_field(
    field: TrainingField, padding: tuple[int, int, int, int]
) -> TrainingField:
    """Place real pixels/targets at the historical offsets with false added masks."""

    left, right, top, bottom = padding
    spatial = ((top, bottom), (left, right))
    targets = field.labels.targets
    return TrainingField(
        np.pad(field.pixels, (*spatial, (0, 0))),
        TrainingFieldTargets(
            ManagedSegmentationTargets(
                directional=np.pad(targets.directional, ((0, 0), *spatial)),
                foreground=np.pad(targets.foreground, spatial),
                classes=(
                    None
                    if targets.classes is None
                    else np.pad(targets.classes, spatial)
                ),
                instance_flow_weights=(
                    None
                    if targets.instance_flow_weights is None
                    else np.pad(targets.instance_flow_weights, spatial)
                ),
            ),
            np.pad(field.labels.supervision_mask, spatial),
            np.pad(field.labels.real_pixel_mask, spatial),
        ),
    )
