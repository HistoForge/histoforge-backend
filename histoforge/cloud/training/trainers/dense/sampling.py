"""Deterministic full-coverage dense fields; no open readers or cached ROI pixels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset

from histoforge.cloud.training.fields.sampling import (
    bounded_field_read,
    permute_epoch_fields,
    prepare_epoch_field_inputs,
)
from histoforge.cloud.training.trainers.dense.fields import (
    DenseTrainingField,
    augment_dense_field,
    pad_dense_field,
    read_dense_field,
)

if TYPE_CHECKING:
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset


class DenseEpochDataset(Dataset[DenseTrainingField]):
    """A CPU-only map dataset with randomness bound to epoch field ordinal."""

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
        api_limits: Mapping[str, Any],
        source_limits: Mapping[str, int],
    ) -> None:
        self.dataset = dataset
        if dataset.manifest["task"]["model_output"]["type"] != "dense_classification":
            raise ValueError("dense sampler requires a dense-classification task")
        self.inputs = prepare_epoch_field_inputs(
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
        self.inputs = permute_epoch_fields(self.inputs, sampler)

    def __len__(self) -> int:
        return len(self.inputs.field_locations)

    def __getitem__(self, index: int) -> DenseTrainingField:
        inputs = self.inputs
        roi_index, rectangle = inputs.field_locations[index]
        target = inputs.grids[roi_index]
        read_rect, padding = bounded_field_read(rectangle, target["raster"])
        field = read_dense_field(
            self.dataset,
            roi_index,
            target,
            read_rect,
            inputs.source_limits,
            inputs.max_json_member_bytes,
        )
        if any(padding):
            field = pad_dense_field(field, padding)
        return augment_dense_field(
            field,
            inputs.augmentation,
            training_seed=inputs.training_seed,
            epoch_index=inputs.epoch_index,
            sample_index=index,
        )
