"""Load task batches and publish epochs only after complete task validation.

The task supplies its dataset, collator, optimization binding and validator.
This owner sequences them without choosing targets, model topology or metrics.
Run state, checkpoint selection and durable publication remain in runs/.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, TypeVar

from histoforge.cloud.training.engine import TrainingCancelled, TrainingEpochResult
from histoforge.cloud.training.metric_policy import TRAINING_LOSS_SERIES_ID

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    from histoforge.cloud.rasterization.rasterizer import RasterizedImage
    from histoforge.cloud.training.datasets.files import RetainedTrainingDataset

Field = TypeVar("Field")
Batch = TypeVar("Batch")


def require_fresh_training_dataset(
    report: Mapping[str, Any], dataset: RetainedTrainingDataset
) -> None:
    """Reject a resumed report or a dataset other than the accepted snapshot."""

    if report["epochs"]:
        raise ValueError("managed training does not resume completed epochs")
    if (
        dataset.manifest["dataset_digest"] != report["dataset_digest"]
        or dataset.manifest["task"] != report["task"]
    ):
        raise ValueError("training dataset differs from the accepted report")


def run_training_epochs(
    report: Mapping[str, Any],
    *,
    device: torch.device | str,
    epoch_dataset: Callable[[int], Dataset[Field]],
    collate_fields: Callable[[Sequence[Field]], Batch],
    train_fields: Callable[..., tuple[TrainingEpochResult, ...]],
    validate: Callable[
        [Callable[[RasterizedImage], Mapping[str, Any]]], Mapping[str, float | None]
    ],
    predict: Callable[[RasterizedImage], Mapping[str, Any]],
    epoch_completed: Callable[[Mapping[str, Any]], None],
    cancellation_requested: Callable[[], bool],
) -> tuple[dict[str, Any], ...]:
    """Preserve seeded loading, validation mode and synchronous epoch publication.

    ``train_fields`` already binds its model, optimizer, objective and schedule.
    Only top-level task datasets/collators enter spawned workers; the callbacks
    and model stay here. Every batch is a concrete dataclass of tensors/None.
    Failures propagate without retry or publication of a partially validated epoch.
    """

    import torch
    from torch.utils.data import DataLoader

    settings = report["trainer_settings"]
    completed = []
    started = time.monotonic()

    def check_cancelled() -> None:
        if cancellation_requested():
            raise TrainingCancelled("managed training was cancelled")

    def batches(epoch_index: int):
        workers = settings["data_loading_workers"]
        loader = DataLoader(
            epoch_dataset(epoch_index),
            batch_size=settings["training_batch_size"],
            num_workers=workers,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fields,
            # Never fork a process containing a CUDA model; workers return CPU data.
            multiprocessing_context="spawn" if workers else None,
            generator=torch.Generator().manual_seed(report["training_seed"]),
        )
        for batch in loader:
            check_cancelled()
            yield replace(
                batch,
                **{
                    field.name: value.to(device)
                    for field in fields(batch)
                    if (value := getattr(batch, field.name)) is not None
                },
            )

    def checked_predict(image: RasterizedImage) -> Mapping[str, Any]:
        check_cancelled()
        return predict(image)

    def finish_epoch(result: TrainingEpochResult, model: nn.Module) -> None:
        check_cancelled()
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                values = dict(validate(checked_predict))
        finally:
            model.train(was_training)
        check_cancelled()
        values[TRAINING_LOSS_SERIES_ID] = result.mean_loss
        epoch = {
            "epoch": result.epoch,
            "elapsed_seconds": int(time.monotonic() - started),
            "metrics": [
                {"series_id": item["series_id"], "value": values[item["series_id"]]}
                for item in report["metric_policy"]["series"]
                if values.get(item["series_id"]) is not None
            ],
        }
        epoch_completed(epoch)
        completed.append(epoch)

    train_fields(
        batches,
        cancellation_requested=cancellation_requested,
        epoch_completed=finish_epoch,
    )
    return tuple(completed)
