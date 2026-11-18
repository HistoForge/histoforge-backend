import copy

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedObjective,
    ManagedSegmentationBatch,
)
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.training.engine import TrainingCancelled
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields
from torch import nn


def test_task_binding_preserves_scientific_tensors_and_field_count(monkeypatch):
    """The boundary converts records, not losses, diagnostics or their gradients."""

    from histoforge.cloud.training.trainers.instance import loop

    model = _TupleModel(5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    batch = _batch([1.0, 2.0, 3.0, 4.0, 7.0, 9.0], class_count=2)
    scientific_results = []
    original_loss = loop.managed_segmentation_loss

    def scientific_loss(*args, **kwargs):
        result = original_loss(*args, **kwargs)
        scientific_results.append(result)
        return result

    def inspect_boundary(network, passed_optimizer, batches, compute_loss, **kwargs):
        assert network is model and passed_optimizer is optimizer
        result = compute_loss(network, next(iter(batches(0))))
        scientific = scientific_results[0]
        assert result.total_sum is scientific.total_sum
        assert result.component_sums is scientific.component_sums
        assert result.field_count == scientific.field_count == 2
        result.total_sum.backward()
        assert model.projection.weight.grad is not None
        assert torch.isfinite(model.projection.weight.grad).all()
        assert torch.count_nonzero(model.projection.weight.grad) > 0
        return ()

    monkeypatch.setattr(loop, "managed_segmentation_loss", scientific_loss)
    monkeypatch.setattr(loop, "optimize", inspect_boundary)
    assert (
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [batch],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=2,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            classification_objective="cross_entropy+tversky",
        )
        == ()
    )


class _TupleModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.Conv2d(3, channels, kernel_size=1, bias=False)

    def forward(self, images):
        return self.projection(images), None


class _StatefulModel(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(3)
        self.projection = nn.Conv2d(3, channels, kernel_size=1)
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        return self.projection(self.batch_norm(images))


class _TwoParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.offset = nn.Parameter(torch.tensor(0.5))
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        return images * self.scale + self.offset


def _batch(values, *, class_count=0, support=None):
    images = torch.tensor(values, dtype=torch.float32).reshape(-1, 3, 1, 1)
    batch = images.shape[0]
    if support is None:
        support = torch.ones((batch, 1, 1), dtype=torch.bool)
    return ManagedSegmentationBatch(
        images=images,
        directional_targets=torch.zeros((batch, 2, 1, 1)),
        foreground_targets=torch.zeros((batch, 1, 1)),
        supervision_mask=support,
        real_pixel_mask=torch.ones_like(support),
        class_targets=(
            torch.zeros((batch, 1, 1), dtype=torch.long) if class_count else None
        ),
    )


def test_accumulation_matches_one_combined_batch_with_unequal_field_losses():
    torch.manual_seed(7)
    combined = _TupleModel(3)
    accumulated = copy.deepcopy(combined)
    combined_optimizer = torch.optim.SGD(combined.parameters(), lr=0.05)
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.05)
    values = [1.0, 2.0, 3.0, 4.0, 7.0, 9.0, 11.0, 13.0, 17.0]

    train_instance_fields(
        combined,
        combined_optimizer,
        lambda _epoch: [_batch(values)],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=0,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=1,
    )
    history = train_instance_fields(
        accumulated,
        accumulated_optimizer,
        lambda _epoch: [
            _batch(values[:3]),
            _batch(values[3:6]),
            _batch(values[6:]),
        ],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=0,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=3,
    )

    assert torch.allclose(
        combined.projection.weight, accumulated.projection.weight, atol=1e-7
    )
    assert history[0].field_count == 3
    assert history[0].optimizer_steps == 1


def test_partial_accumulation_group_uses_its_actual_field_count():
    torch.manual_seed(11)
    partial = _TupleModel(3)
    expected = copy.deepcopy(partial)
    partial_optimizer = torch.optim.SGD(partial.parameters(), lr=0.05)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=0.05)
    values = [1.0, 2.0, 3.0, 5.0, 7.0, 11.0]

    history = train_instance_fields(
        partial,
        partial_optimizer,
        lambda _epoch: [_batch(values[:3]), _batch(values[3:])],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=0,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=8,
    )
    train_instance_fields(
        expected,
        expected_optimizer,
        lambda _epoch: [_batch(values)],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=0,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=1,
    )

    assert torch.allclose(partial.projection.weight, expected.projection.weight)
    assert history[0].optimizer_steps == 1


def test_uncertainty_weighting_preserves_accumulation_equivalence():
    torch.manual_seed(13)
    combined = _TupleModel(5)
    accumulated = copy.deepcopy(combined)
    combined_objective = ManagedObjective(
        class_count=2,
        classification_objective="cross_entropy+tversky",
        use_uncertainty_weighting=True,
    )
    accumulated_objective = copy.deepcopy(combined_objective)
    combined_optimizer = torch.optim.SGD(
        [
            {"params": combined.parameters()},
            {"params": combined_objective.parameters(), "lr": 0.005},
        ],
        lr=0.05,
    )
    accumulated_optimizer = torch.optim.SGD(
        [
            {"params": accumulated.parameters()},
            {"params": accumulated_objective.parameters(), "lr": 0.005},
        ],
        lr=0.05,
    )
    values = [1.0, 2.0, 3.0, 4.0, 7.0, 9.0, 11.0, 13.0, 17.0]

    train_instance_fields(
        combined,
        combined_optimizer,
        lambda _epoch: [_batch(values, class_count=2)],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=2,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=1,
        classification_objective="cross_entropy+tversky",
        objective=combined_objective,
    )
    train_instance_fields(
        accumulated,
        accumulated_optimizer,
        lambda _epoch: [
            _batch(values[:3], class_count=2),
            _batch(values[3:6], class_count=2),
            _batch(values[6:], class_count=2),
        ],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=2,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=3,
        classification_objective="cross_entropy+tversky",
        objective=accumulated_objective,
    )

    assert torch.allclose(
        combined.projection.weight, accumulated.projection.weight, atol=1e-7
    )
    assert torch.allclose(
        combined_objective.log_variances,
        accumulated_objective.log_variances,
        atol=1e-7,
    )
    assert torch.count_nonzero(combined_objective.log_variances) > 0


def test_learning_rate_schedule_preserves_parameter_group_ratios():
    model = _TwoParameterModel()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.scale], "lr": 0.1},
            {"params": [model.offset], "lr": 0.01},
        ]
    )
    observed = []

    def batches(_epoch):
        observed.append([group["lr"] for group in optimizer.param_groups])
        return [_batch([1.0, 2.0, 3.0])]

    train_instance_fields(
        model,
        optimizer,
        batches,
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=0,
        learning_rate_scales=[1.0, 0.25],
        gradient_accumulation_steps=1,
    )

    assert observed == [[0.1, 0.01], [0.025, 0.0025]]


def test_cancellation_before_step_prevents_optimizer_update():
    model = _TupleModel(3)
    before = model.projection.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(TrainingCancelled):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0])],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            cancellation_requested=cancelled,
        )

    assert torch.equal(model.projection.weight, before)


def test_cancellation_fences_epoch_callback():
    model = _TupleModel(3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    calls = 0
    published = False

    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 4

    def epoch_completed(_result, _model):
        nonlocal published
        published = True

    with pytest.raises(TrainingCancelled):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0])],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            cancellation_requested=cancelled,
            epoch_completed=epoch_completed,
        )

    assert not published


def test_invalid_field_is_rejected_before_stateful_model_forward():
    model = _StatefulModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    invalid = ManagedSegmentationBatch(
        images=torch.zeros((1, 3, 2, 2)),
        directional_targets=torch.zeros((1, 2, 2, 2)),
        foreground_targets=torch.zeros((1, 2, 2)),
        supervision_mask=torch.zeros((1, 2, 2), dtype=torch.bool),
        real_pixel_mask=torch.ones((1, 2, 2), dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="supervised real pixels"):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [invalid],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
        )

    assert model.forward_calls == 0
    assert model.batch_norm.num_batches_tracked.item() == 0


def test_trainable_objective_must_belong_to_optimizer_before_forward():
    model = _StatefulModel(channels=5)
    objective = ManagedObjective(
        class_count=2,
        classification_objective="cross_entropy+tversky",
        use_uncertainty_weighting=True,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="trainable objective parameters"):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0], class_count=2)],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=2,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            classification_objective="cross_entropy+tversky",
            objective=objective,
        )

    assert model.forward_calls == 0
    assert model.batch_norm.num_batches_tracked.item() == 0


def test_all_trainable_model_parameters_must_belong_to_optimizer():
    model = _TwoParameterModel()
    optimizer = torch.optim.SGD([model.scale], lr=0.05)

    with pytest.raises(ValueError, match="trainable model parameters"):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0])],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
        )

    assert model.forward_calls == 0


def test_invalid_loss_configuration_is_rejected_before_model_forward():
    model = _StatefulModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="head_type"):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0])],
            training_forward=instance_training_predictions,
            head_type="invalid",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
        )

    assert model.forward_calls == 0
    assert model.batch_norm.num_batches_tracked.item() == 0


def test_missing_equal_instance_weights_are_rejected_before_model_forward():
    model = _StatefulModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    with pytest.raises(ValueError, match="derived from complete instances"):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0])],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            flow_loss_weighting="equal_instance",
        )

    assert model.forward_calls == 0
    assert model.batch_norm.num_batches_tracked.item() == 0


def test_nonfinite_loss_fails_without_step_or_epoch_callback():
    class NonfiniteModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, images):
            return torch.full_like(images, float("nan")) * self.weight

    model = NonfiniteModel()
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    published = False

    def epoch_completed(_result, _model):
        nonlocal published
        published = True

    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train_instance_fields(
            model,
            optimizer,
            lambda _epoch: [_batch([1.0, 2.0, 3.0])],
            training_forward=instance_training_predictions,
            head_type="flow",
            class_count=0,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            epoch_completed=epoch_completed,
        )

    assert torch.equal(model.weight, before)
    assert model.weight.grad is None
    assert not published


@pytest.mark.parametrize(
    ("head_type", "class_count", "channels"),
    [("flow", 0, 3), ("flow", 3, 6), ("hovernet", 0, 4), ("hovernet", 3, 7)],
)
def test_all_shared_head_shapes_train(head_type, class_count, channels):
    model = _TupleModel(channels)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _batch([1.0, 2.0, 3.0], class_count=class_count)

    history = train_instance_fields(
        model,
        optimizer,
        lambda _epoch: [batch],
        training_forward=instance_training_predictions,
        head_type=head_type,
        class_count=class_count,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=1,
    )

    assert history[0].field_count == 1
    assert history[0].mean_loss > 0
