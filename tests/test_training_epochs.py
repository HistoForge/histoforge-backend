"""Shared sequencing and task bindings do not assume a backbone output layout."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedDenseBatch,
)
from histoforge.cloud.training.engine import TrainingEpochResult
from histoforge.cloud.training.epochs import run_training_epochs
from histoforge.cloud.training.trainers.dense.loop import (
    train_dense_fields,
    train_managed_dense_model,
)
from histoforge.cloud.training.trainers.instance.loop import (
    train_instance_fields,
    train_managed_model,
)
from tests.test_training_batches import single_thread_torch as single_thread_torch
from tests.test_training_dataset import training_api_limits
from tests.test_training_execution import _setup
from tests.test_training_field_reader import _source_limits
from tests.test_training_instance_loop import _batch
from tests.training_dense_fixtures import dense_setup
from torch import nn


class _DictionaryModel(nn.Module):
    """No DINO module names, tuple output or unused segmentation tail."""

    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(3, channels, 1)

    def forward(self, images):
        return {"scores": self.projection(images)}


def _fixture_preprocess(pixels, *, real_pixel_mask):
    """Distinct test-only preprocessing; top-level for spawned CPU workers."""

    result = np.zeros(pixels.shape, dtype=np.float32)
    result[real_pixel_mask] = 0.25
    return result


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("workers", [0, 1])
def test_complete_task_loop_reuses_fields_losses_and_validation_for_another_model(
    tmp_path, monkeypatch, dense, workers, single_thread_torch
):
    """Task reuse is not admission or qualification of a new production model.

    Use real retained fields, collation, optimizer updates, full-ROI validation
    and publication sequencing. Explicit fixture component identities prevent
    this test from claiming that a non-DINO optimizer implements the DINO policy.
    """

    if dense:
        setup = dense_setup(tmp_path)
        report, dataset, capability = (
            setup.report,
            setup.dataset,
            setup.entry.capability,
        )
    else:
        _, report, dataset, capability = _setup(tmp_path)
    options = report["trainer_configuration"]["objective"]["parameters"]
    report["trainer_configuration"]["optimizer"] = {
        "profile_id": "test-only-plain-adamw-v1",
        "parameters": {"learning_rate": 0.01},
    }
    report["trainer_configuration"]["preprocessing"] = {
        "profile_id": "test-only-constant-real-rgb-v1",
        "parameters": {},
    }
    report["field_profile"]["preprocessing"] = "test-only-constant-real-rgb-v1"
    capability["default_field_profile"] = dict(report["field_profile"])
    report["trainer_settings"]["data_loading_workers"] = workers
    torch.manual_seed(37)
    model = _DictionaryModel(options["class_count"] + (0 if dense else 3))
    original = model.projection.weight.detach().clone()
    optimizers, objectives, forwarded, predicted, published = [], [], [], [], []

    def unexpected(*_args, **_kwargs):
        pytest.fail(
            "a task invoked concrete DINO behavior or a production report validator"
        )

    monkeypatch.setattr(
        "histoforge.cloud.architectures.dinov3_optimization.build_dinov3_optimizer",
        unexpected,
    )
    monkeypatch.setattr(
        "histoforge.cloud.training.profile_schema.validate_managed_training_report_configuration",
        unexpected,
    )

    def build_optimizer(network, objective, component):
        assert component["profile_id"] == "test-only-plain-adamw-v1"
        assert list(dict(network.named_parameters())) == [
            "projection.weight",
            "projection.bias",
        ]
        optimizer = torch.optim.AdamW(
            [*network.parameters(), *objective.parameters()],
            lr=component["parameters"]["learning_rate"],
        )
        optimizers.append(optimizer)
        objectives.append(objective)
        return optimizer

    def training_forward(network, images):
        assert set(images.unique().tolist()) == {0.0, 0.25}
        forwarded.append(images.shape)
        return network(images)["scores"]

    def predict(image):
        assert not model.training and torch.is_inference_mode_enabled()
        predicted.append(image.pixels.shape)
        pixels = _fixture_preprocess(
            image.pixels, real_pixel_mask=image.real_pixel_mask
        )
        scores = model(torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0))[
            "scores"
        ][0]
        if dense:
            return {"class_map": scores.argmax(0).numpy().astype(np.int32)}
        mask = (scores[2] > 0).numpy().astype(np.int32)
        return {
            "instance_mask": mask,
            "votes_per_instance": {1: 1} if mask.any() else {},
        }

    def publish(epoch):
        assert len(predicted) == 2 * epoch["epoch"]
        assert model.training
        published.append(epoch)

    train = train_managed_dense_model if dense else train_managed_model
    epochs = train(
        model,
        report,
        dataset,
        predict,
        build_optimizer=build_optimizer,
        training_forward=training_forward,
        preprocess=_fixture_preprocess,
        capability=capability,
        device="cpu",
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
        target_raster_limits={"max_width": 128, "max_height": 128, "max_pixels": 16384},
        epoch_completed=publish,
        **(
            {"prepare_model": lambda _: None}
            if dense
            else {"max_assignment_matrix_entries": 100}
        ),
    )
    assert list(epochs) == published
    assert len(forwarded) == len(epochs) == report["total_epochs"]
    assert all(shape == (20, 20, 3) for shape in predicted)
    assert not torch.equal(model.projection.weight, original)
    assert objectives[0].log_variances.abs().sum() > 0
    assert len(optimizers) == 1
    assert all(
        state["step"].item() == len(epochs) for state in optimizers[0].state.values()
    )
    assert all(
        epoch["metrics"][0]["series_id"] == "training-loss.v1" for epoch in epochs
    )


@pytest.mark.parametrize("dense", [False, True])
def test_task_loss_accepts_explicit_backbone_output_extraction(
    dense, single_thread_torch
):
    model = _DictionaryModel(3)
    original = model.projection.bias.detach().clone()
    calls = []

    def training_forward(network, images):
        calls.append(images.shape)
        return network(images)["scores"]

    batch = _batch([1.0, 2.0, 3.0])
    if dense:
        batch = ManagedDenseBatch(
            batch.images,
            torch.zeros((1, 1, 1), dtype=torch.long),
            batch.supervision_mask,
            batch.real_pixel_mask,
        )
    train = train_dense_fields if dense else train_instance_fields
    result = train(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        lambda _: [batch],
        training_forward=training_forward,
        class_count=3 if dense else 0,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=1,
        **({} if dense else {"head_type": "flow"}),
    )
    assert calls == [batch.images.shape]
    assert result[0].field_count == 1 and result[0].optimizer_steps == 1
    assert not torch.equal(model.projection.bias, original)


@dataclass(frozen=True)
class _TensorBatch:
    images: torch.Tensor
    optional: torch.Tensor | None = None


@pytest.mark.parametrize("training", [False, True])
def test_epoch_sequence_preserves_mode_sparse_metrics_and_callback_order(training):
    model = nn.Linear(1, 1)
    model.train(training)
    events = []
    values = {"defined-zero": 0.0, "undefined": None}
    report = {
        "training_seed": 37,
        "trainer_settings": {"data_loading_workers": 0, "training_batch_size": 2},
        "metric_policy": {
            "series": [
                {"series_id": name}
                for name in ("training-loss.v1", "defined-zero", "undefined", "missing")
            ]
        },
    }

    def train(batches, *, cancellation_requested, epoch_completed):
        for index in range(2):
            events.append(("train", index + 1))
            batch = list(batches(index))
            assert len(batch) == 1 and batch[0].optional is None
            assert torch.equal(batch[0].images, torch.tensor([0, 1]))
            epoch_completed(TrainingEpochResult(index + 1, 2, 1, 0.5, {}), model)

    def predict(image):
        assert image == "complete-roi"
        assert not model.training and torch.is_inference_mode_enabled()
        events.append(("predict", None))
        return {}

    def validate(checked_predict):
        checked_predict("complete-roi")
        return values

    def publish(epoch):
        assert model.training is training and not torch.is_inference_mode_enabled()
        assert epoch["metrics"] == [
            {"series_id": "training-loss.v1", "value": 0.5},
            {"series_id": "defined-zero", "value": 0.0},
        ]
        events.append(("publish", epoch["epoch"]))

    completed = run_training_epochs(
        report,
        device="cpu",
        epoch_dataset=lambda _: [0, 1],
        collate_fields=lambda items: _TensorBatch(torch.tensor(items)),
        train_fields=train,
        validate=validate,
        predict=predict,
        epoch_completed=publish,
        cancellation_requested=lambda: False,
    )
    assert events == [
        ("train", 1),
        ("predict", None),
        ("publish", 1),
        ("train", 2),
        ("predict", None),
        ("publish", 2),
    ]
    assert len(completed) == 2
    assert values == {"defined-zero": 0.0, "undefined": None}
