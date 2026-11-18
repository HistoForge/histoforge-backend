"""Fail-closed optimizer gates, including parameters outside the model."""

import copy
from unittest.mock import Mock

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedObjective,
    ManagedSegmentationBatch,
)
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields
from torch import nn
from torch.nn import functional as F


class _Model(nn.Module):
    """A finite, stateful forward with independently controllable gradients."""

    def __init__(self, channels=3):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.forward_calls = 0

    def forward(self, images):
        self.forward_calls += 1
        return self.weight[None, :, None, None].expand(
            images.shape[0], -1, images.shape[2], images.shape[3]
        )


def _batch(class_count):
    """Return a synthetic supported negative field; no dataset is accessed."""

    return ManagedSegmentationBatch(
        images=torch.zeros(1, 3, 4, 4),
        directional_targets=torch.zeros(1, 2, 4, 4),
        foreground_targets=torch.zeros(1, 4, 4),
        supervision_mask=torch.ones(1, 4, 4, dtype=torch.bool),
        real_pixel_mask=torch.ones(1, 4, 4, dtype=torch.bool),
        class_targets=torch.zeros(1, 4, 4, dtype=torch.long) if class_count else None,
    )


def _train(model, optimizer, callback, *, class_count=0, **options):
    """Exercise the public entry point with one finite field."""

    arguments = dict(
        head_type="flow",
        class_count=class_count,
        learning_rate_scales=[1.0],
        gradient_accumulation_steps=1,
        epoch_completed=callback,
    )
    arguments.update(options)
    return train_instance_fields(
        model,
        optimizer,
        lambda _epoch: [_batch(class_count)],
        training_forward=instance_training_predictions,
        **arguments,
    )


@pytest.mark.parametrize("bad_gradient", [float("inf"), float("nan")])
@pytest.mark.parametrize("max_norm", [None, 1.0])
@pytest.mark.parametrize("accumulation", [1, 8])
def test_nonfinite_backward_rejected_without_update_or_callback(
    bad_gradient, max_norm, accumulation
):
    """A finite loss is insufficient: full and partial groups must check gradients."""

    model = _Model()
    model.weight.register_hook(lambda grad: torch.full_like(grad, bad_gradient))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    before = model.weight.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    callback = Mock()

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        _train(
            model,
            optimizer,
            callback,
            max_gradient_norm=max_norm,
            gradient_accumulation_steps=accumulation,
        )

    optimizer.step.assert_not_called()
    callback.assert_not_called()
    assert torch.equal(model.weight, before)
    assert optimizer.state_dict() == state
    assert model.weight.grad is None


@pytest.mark.parametrize("max_norm", [None, 1.0])
def test_uncertainty_objective_gradient_is_also_checked(max_norm):
    """The optimizer also owns learned task weights outside model.parameters()."""

    model = _Model(channels=5)
    objective = ManagedObjective(class_count=2, use_uncertainty_weighting=True)
    objective.log_variances.register_hook(
        lambda grad: torch.full_like(grad, float("inf"))
    )
    parameters = [*model.parameters(), *objective.parameters()]
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.Adam(parameters, lr=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    state = copy.deepcopy(optimizer.state_dict())
    callback = Mock()

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        _train(
            model,
            optimizer,
            callback,
            class_count=2,
            objective=objective,
            max_gradient_norm=max_norm,
        )

    optimizer.step.assert_not_called()
    callback.assert_not_called()
    assert optimizer.state_dict() == state
    for parameter, previous in zip(parameters, before):
        assert torch.equal(parameter, previous)
        assert parameter.grad is None


@pytest.mark.parametrize(
    "options,initial_lr",
    [
        ({"learning_rate_scales": [float("nan")]}, 0.1),
        ({"learning_rate_scales": [float("inf")]}, 0.1),
        ({"learning_rate_scales": [1.0, float("inf")]}, 0.1),
        ({"learning_rate_scales": [-1.0]}, 0.1),
        ({"learning_rate_scales": [1.0, 2.0]}, 1.0e308),
        ({"max_gradient_norm": float("nan")}, 0.1),
        ({"max_gradient_norm": float("inf")}, 0.1),
        ({"max_gradient_norm": 0.0}, 0.1),
        ({"max_gradient_norm": -1.0}, 0.1),
        ({}, float("nan")),
        ({}, float("inf")),
        ({}, -0.1),
        ({"gradient_accumulation_steps": 1.5}, 0.1),
        ({"classification_loss_weight": float("nan")}, 0.1),
        ({"classification_loss_weight": float("inf")}, 0.1),
        ({"tversky_gamma": float("nan")}, 0.1),
        ({"tversky_gamma": float("inf")}, 0.1),
    ],
)
def test_invalid_configuration_rejected_before_forward_or_optimizer_mutation(
    options, initial_lr
):
    """Validate even future epoch rates before any training state is changed."""

    model = _Model()
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer.param_groups[0]["lr"] = initial_lr
    optimizer.step = Mock(wraps=optimizer.step)
    state = repr(optimizer.state_dict())
    before = model.weight.detach().clone()
    callback = Mock()

    with pytest.raises(ValueError):
        _train(model, optimizer, callback, **options)

    assert model.forward_calls == 0
    assert not model.training
    optimizer.step.assert_not_called()
    callback.assert_not_called()
    assert repr(optimizer.state_dict()) == state
    assert torch.equal(model.weight, before)


def test_clipping_preserves_its_domain_and_rechecks_all_gradients(monkeypatch):
    """Preserve model-only clipping; still reject corruption in any owned gradient."""

    model = _Model(channels=5)
    objective = ManagedObjective(class_count=2, use_uncertainty_weighting=True)
    parameters = [*model.parameters(), *objective.parameters()]
    before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.Adam(parameters, lr=0.1)
    optimizer.step = Mock(wraps=optimizer.step)
    callback = Mock()

    def corrupt_clipping(values, *_args, **_kwargs):
        values = tuple(values)
        assert {id(p) for p in values} == {id(p) for p in model.parameters()}
        objective.log_variances.grad.fill_(float("nan"))
        return torch.tensor(1.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", corrupt_clipping)
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        _train(
            model,
            optimizer,
            callback,
            class_count=2,
            objective=objective,
            max_gradient_norm=1.0,
        )

    optimizer.step.assert_not_called()
    callback.assert_not_called()
    for parameter, previous in zip(parameters, before):
        assert torch.equal(parameter, previous)
        assert parameter.grad is None


@pytest.mark.parametrize("weight", [float("nan"), float("inf")])
def test_objective_constructor_rejects_nonfinite_weight(weight):
    """Direct use of the vendorable objective also enforces finite configuration."""

    with pytest.raises(ValueError, match="finite"):
        ManagedObjective(class_count=2, classification_loss_weight=weight)


@pytest.mark.parametrize("zero_scale", [0, 0.0])
@pytest.mark.parametrize("accumulation", [1, 8])
def test_adamw_zero_rate_advances_state_and_matches_reference(zero_scale, accumulation):
    """Zero-rate steps retain moments for later updates, including task weights."""

    model = nn.Conv2d(3, 5, kernel_size=1)
    objective = ManagedObjective(class_count=2, use_uncertainty_weighting=True)
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.copy_(torch.tensor([-1.25, -1.85, 0.5, -4.5, -1.4]))
        objective.log_variances.copy_(torch.tensor([0.2, -0.1]))
    reference_model = copy.deepcopy(model)
    reference_objective = copy.deepcopy(objective)
    peak_lrs = [0.02, 0.01, 0.003]
    scales = [zero_scale, 1.0, 0.25]

    def make_optimizer(network, task_objective):
        return torch.optim.AdamW(
            [
                {"params": [network.weight], "lr": peak_lrs[0]},
                {"params": [network.bias], "lr": peak_lrs[1]},
                {"params": task_objective.parameters(), "lr": peak_lrs[2]},
            ],
            weight_decay=0.1,
        )

    optimizer = make_optimizer(model, objective)
    reference_optimizer = make_optimizer(reference_model, reference_objective)
    parameters = [*model.parameters(), *objective.parameters()]
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    batch = _batch(2)
    batch.images.fill_(1.0)
    observed_lrs = []

    def compare_epoch(result, trained_model):
        assert trained_model is model
        scale = scales[result.epoch - 1]
        expected_lrs = [lr * scale for lr in peak_lrs]
        observed_lrs.append([group["lr"] for group in optimizer.param_groups])
        assert observed_lrs[-1] == expected_lrs
        assert result.field_count == 1
        assert result.optimizer_steps == 1

        # Independent Torch forward/loss/step; do not reuse the managed trainer,
        # loss reduction, or objective combination as the reference algorithm.
        for group, learning_rate in zip(reference_optimizer.param_groups, expected_lrs):
            group["lr"] = learning_rate
        reference_optimizer.zero_grad(set_to_none=True)
        predictions = reference_model(batch.images)
        segmentation = predictions[:, 2:4].square().mean() / 2.0
        segmentation = segmentation + F.binary_cross_entropy_with_logits(
            predictions[:, 4], batch.foreground_targets
        )
        classification = F.cross_entropy(predictions[:, :2], batch.class_targets)
        task_losses = torch.stack([segmentation, classification])
        log_variances = reference_objective.log_variances
        loss = (torch.exp(-log_variances) * task_losses + log_variances).sum()
        loss.backward()
        # The fixture genuinely triggers clipping, which must exclude the
        # independently optimizer-owned objective parameters.
        norm = torch.nn.utils.clip_grad_norm_(reference_model.parameters(), 1.0)
        assert norm > 1.0
        reference_optimizer.step()
        reference_optimizer.zero_grad(set_to_none=True)

        for actual_group, expected_group in zip(
            optimizer.param_groups, reference_optimizer.param_groups
        ):
            for actual, expected in zip(
                actual_group["params"], expected_group["params"]
            ):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                assert actual.grad is None
                actual_state = optimizer.state[actual]
                expected_state = reference_optimizer.state[expected]
                assert actual_state.keys() == expected_state.keys()
                assert actual_state["step"].item() == result.epoch
                for name in actual_state:
                    torch.testing.assert_close(
                        actual_state[name], expected_state[name], rtol=1e-6, atol=1e-7
                    )
                assert torch.count_nonzero(actual_state["exp_avg"]) > 0
                assert torch.count_nonzero(actual_state["exp_avg_sq"]) > 0

        for parameter, initial in zip(parameters, initial_parameters):
            if result.epoch == 1:
                assert torch.equal(parameter, initial)
            else:
                assert not torch.equal(parameter, initial)

    history = train_instance_fields(
        model,
        optimizer,
        lambda _epoch: [batch],
        training_forward=instance_training_predictions,
        head_type="flow",
        class_count=2,
        learning_rate_scales=scales,
        gradient_accumulation_steps=accumulation,
        max_gradient_norm=1.0,
        objective=objective,
        epoch_completed=compare_epoch,
    )

    assert len(history) == 3
    assert observed_lrs == [[0.0, 0.0, 0.0], peak_lrs, [0.005, 0.0025, 0.00075]]
