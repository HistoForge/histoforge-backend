"""Task-independent optimization without instance tensors or output assumptions."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from histoforge.cloud.training.engine import TrainingStepResult, optimize
from torch import nn


def test_engine_runs_without_scientific_vendor_or_cloud_provider(tmp_path):
    """A clean interpreter performs a real update without the instance stack."""

    repository_root = Path(__file__).resolve().parents[1]
    script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(repository_root)!r})

class ForbidTaskAndProviderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = ('cellpose', 'histoforge_models', 'histoforge._vendor', 'modal',
                 'histoforge.cloud.training.trainers',
                 'histoforge.cloud.training.datasets',
                 'histoforge.cloud.training.runs', 'histoforge.cloud.architectures')
        if any(fullname == root or fullname.startswith(root + '.') for root in roots):
            raise AssertionError('engine crossed its ownership boundary: ' + fullname)

sys.meta_path.insert(0, ForbidTaskAndProviderImports())
from histoforge.cloud.training.engine import TrainingStepResult, optimize
import torch

class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
    def forward(self, inputs):
        return {{'scores': self.weight * inputs}}

model = Regression()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
def loss_step(network, values):
    field_losses = network(values)['scores'].square()
    total = field_losses.sum()
    return TrainingStepResult(total, {{'squared_error': total}}, len(values))

history = optimize(model, optimizer, lambda epoch: [torch.tensor([1.0, 2.0])],
    loss_step, learning_rate_scales=[1.0], gradient_accumulation_steps=1)
assert len(history) == 1 and history[0].field_count == 2
assert history[0].mean_loss == 2.5 and history[0].optimizer_steps == 1
torch.testing.assert_close(model.weight, torch.tensor(0.5))
"""
    subprocess.run(
        [sys.executable, "-B", "-I", "-c", script],
        cwd=tmp_path,
        check=True,
        timeout=60,
    )


class RegressionModel(nn.Module):
    """A deliberately non-segmentation test model with a task-owned output."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.7))
        self.forward_count = 0

    def forward(self, inputs):
        self.forward_count += 1
        return {"estimates": inputs * self.weight}


def _loss_step(objective):
    def compute_loss(model, batch):
        inputs, targets = batch
        if inputs.shape != targets.shape or inputs.shape[0] == 0:
            raise ValueError("regression fields require matching nonempty targets")
        predictions = model(inputs)["estimates"]
        # One mean per field, then a sum over actual fields, not all pixels.
        field_losses = (predictions - targets).square().flatten(1).mean(1)
        task_sum = field_losses.sum()
        total = objective["scale"].exp() * task_sum
        return TrainingStepResult(
            total_sum=total,
            # Overlapping diagnostics must never be summed into the objective.
            component_sums={"error": task_sum, "twice_error": 2 * task_sum},
            field_count=len(inputs),
        )

    return compute_loss


@pytest.mark.parametrize("accumulation", [2, 3])
def test_arbitrary_task_matches_independent_field_mean_updates(accumulation):
    """Unequal microbatches and a partial group preserve model/objective updates."""

    model = RegressionModel()
    objective = nn.ParameterDict({"scale": nn.Parameter(torch.tensor(-0.2))})
    reference = copy.deepcopy(model)
    reference_objective = copy.deepcopy(objective)
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [8.0, 16.0]])
    targets = torch.tensor([[0.5, 1.5], [1.0, 2.0], [4.0, 7.0]])

    def optimizer_for(network, loss_parameters):
        return torch.optim.AdamW(
            [
                {"params": network.parameters(), "lr": 0.01},
                {"params": loss_parameters.parameters(), "lr": 0.003},
            ],
            weight_decay=0.1,
        )

    optimizer = optimizer_for(model, objective)
    expected_optimizer = optimizer_for(reference, reference_objective)
    snapshots = []

    def finish_epoch(result, trained_model):
        assert trained_model is model
        assert model.training and objective.training
        assert result.field_count == 3
        assert result.optimizer_steps == 1
        assert result.component_means["twice_error"] == pytest.approx(
            2 * result.component_means["error"]
        )
        snapshots.append(
            (
                result,
                model.weight.detach().clone(),
                objective["scale"].detach().clone(),
                copy.deepcopy(optimizer.state_dict()),
            )
        )

    model.eval()
    objective.eval()
    history = optimize(
        model,
        optimizer,
        lambda epoch: [(inputs[:2], targets[:2]), (inputs[2:], targets[2:])],
        _loss_step(objective),
        learning_rate_scales=[0.0, 1.0, 0.25],
        gradient_accumulation_steps=accumulation,
        max_gradient_norm=0.1,
        objective=objective,
        epoch_completed=finish_epoch,
    )
    assert tuple(snapshot[0] for snapshot in snapshots) == history
    for scale, (result, weight, loss_scale, state) in zip([0.0, 1.0, 0.25], snapshots):
        for group, peak in zip(expected_optimizer.param_groups, [0.01, 0.003]):
            group["lr"] = scale * peak
        expected_optimizer.zero_grad(set_to_none=True)
        per_field = (reference(inputs)["estimates"] - targets).square().mean(1)
        expected_loss = reference_objective["scale"].exp() * per_field.mean()
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
        expected_optimizer.step()
        assert result.mean_loss == pytest.approx(float(expected_loss.detach()))
        torch.testing.assert_close(weight, reference.weight)
        torch.testing.assert_close(loss_scale, reference_objective["scale"])
        expected_state = expected_optimizer.state_dict()
        assert state["param_groups"] == expected_state["param_groups"]
        assert state["state"].keys() == expected_state["state"].keys()
        for key, parameter_state in state["state"].items():
            for name, value in parameter_state.items():
                torch.testing.assert_close(value, expected_state["state"][key][name])


def test_task_validation_runs_before_forward_or_optimizer_mutation():
    model = RegressionModel()
    objective = nn.ParameterDict({"scale": nn.Parameter(torch.tensor(0.0))})
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(objective.parameters())
    )
    with pytest.raises(ValueError, match="matching nonempty targets"):
        optimize(
            model,
            optimizer,
            lambda epoch: [(torch.ones(1, 2), torch.ones(2, 2))],
            _loss_step(objective),
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=1,
            objective=objective,
        )
    assert model.forward_count == 0
    assert optimizer.state == {}
    assert model.weight.item() == pytest.approx(0.7)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_callback_failure_stops_before_next_epoch_without_retry():
    model = RegressionModel()
    objective = nn.ParameterDict({"scale": nn.Parameter(torch.tensor(0.0))})
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(objective.parameters()), lr=0.1
    )
    events = []

    def batches(epoch):
        events.append(("batches", epoch))
        return [(torch.ones(1, 2), torch.zeros(1, 2))]

    def callback(result, trained_model):
        assert trained_model is model
        events.append(("callback", result.epoch))
        raise RuntimeError("validation or publication failed")

    with pytest.raises(RuntimeError, match="validation or publication failed"):
        optimize(
            model,
            optimizer,
            batches,
            _loss_step(objective),
            learning_rate_scales=[1.0, 1.0],
            gradient_accumulation_steps=1,
            objective=objective,
            epoch_completed=callback,
        )
    assert events == [("batches", 0), ("callback", 1)]
    assert model.forward_count == 1
