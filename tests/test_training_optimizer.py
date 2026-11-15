"""Source-equivalent parameter ownership, learning rates, and decay rules."""

from __future__ import annotations

import copy

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology import optimization
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedObjective,
    ManagedSegmentationBatch,
    managed_segmentation_loss,
)
from histoforge.cloud.architectures.dinov3_optimization import (
    build_dinov3_optimizer,
)
from histoforge.cloud.training.trainers.instance.configuration import (
    resolve_instance_configuration,
)
from tests.test_training_preset import _model_output, initial_profile
from torch import nn


class _Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.backbone.patch_embed = nn.Linear(2, 2)
        self.backbone.norm = nn.LayerNorm(2)
        self.backbone.cls_token = nn.Parameter(torch.ones(1, 1, 2))
        self.backbone.reg_token = nn.Parameter(torch.ones(1, 4, 2))
        self.backbone.mask_token = nn.Parameter(torch.ones(1, 1, 2))
        self.spm = nn.Linear(2, 2)
        self.cti_modules = nn.ModuleList([nn.Linear(2, 2)])
        self.neck = nn.Linear(2, 2)
        self.out = nn.Linear(2, 2)
        self.out_cls = nn.Linear(2, 2)
        self.other = nn.Linear(2, 2)
        self.W2 = nn.Parameter(torch.eye(2), requires_grad=False)
        self.W3 = nn.Parameter(torch.eye(2), requires_grad=False)
        self.diam_mean = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.diam_labels = nn.Parameter(torch.tensor([30.0]), requires_grad=False)
        self.frozen = nn.Linear(2, 2).requires_grad_(False)


@pytest.mark.parametrize(
    "class_count,classification",
    [(0, "cross_entropy"), (3, "cross_entropy"), (3, "cross_entropy+tversky")],
)
def test_optimizer_exact_rates_decay_and_ownership(class_count, classification):
    model = _Adapter()
    objective = ManagedObjective(
        class_count=class_count,
        classification_objective=classification,
        use_uncertainty_weighting=True,
    )
    _, profile = initial_profile(classified=bool(class_count))
    component = profile["trainer_components"]["optimizer"]
    original_component = copy.deepcopy(component)
    original_parameters = {
        name: (parameter.detach().clone(), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    optimizer = build_dinov3_optimizer(model, objective, component)
    assert component == original_component
    actual = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert id(parameter) not in actual
            actual[id(parameter)] = (group["lr"], group["weight_decay"])
    assert set(actual) == {
        id(parameter)
        for module in (model, objective)
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    for name, parameter in model.named_parameters():
        before, requires_grad = original_parameters[name]
        assert parameter.requires_grad is requires_grad
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        if not requires_grad:
            assert id(parameter) not in actual
            continue
        if name.startswith("backbone.blocks."):
            rate = [5e-5 * 0.8**2, 5e-5 * 0.8, 5e-5][int(name.split(".")[2])]
        elif name.startswith("backbone.patch_embed."):
            rate = 3e-4
        elif name.startswith(("spm.", "cti_modules.", "neck.", "out")):
            rate = 1e-3
        else:
            rate = 5e-5
        decay = (
            0.0
            if parameter.ndim < 2 or "cls_token" in name or "reg_token" in name
            else 0.1
        )
        assert actual[id(parameter)] == (rate, decay)
    for parameter in objective.parameters():
        assert actual[id(parameter)] == (5e-6, 0.1)
        torch.testing.assert_close(
            parameter, torch.zeros_like(parameter), rtol=0, atol=0
        )
    assert len(list(objective.parameters())) == (1 if class_count else 0)
    expected_defaults = {
        "lr": 5e-5,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.1,
        "amsgrad": False,
        "foreach": None,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
    }
    # Newer Torch exposes AdamW's always-decoupled invariant in its defaults;
    # older supported releases encode it in the AdamW implementation itself.
    reference = torch.optim.AdamW([nn.Parameter(torch.zeros(1))])
    if "decoupled_weight_decay" in reference.defaults:
        expected_defaults["decoupled_weight_decay"] = True
    assert optimizer.defaults == expected_defaults
    assert component["parameters"]["decoupled_weight_decay"] is True
    for module in (model, objective):
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        before, requires_grad = original_parameters[name]
        if not requires_grad:
            torch.testing.assert_close(parameter, before, rtol=0, atol=0)


def test_optimizer_consumes_recorded_numeric_values_not_hidden_defaults():
    model = _Adapter()
    objective = ManagedObjective(class_count=3, use_uncertainty_weighting=True)
    _, profile = initial_profile(classified=True)
    component = profile["trainer_components"]["optimizer"]
    parameters = component["parameters"]
    parameters.update(
        learning_rate=0.01,
        llrd_factor=0.5,
        patch_embed_learning_rate=0.02,
        adapter_learning_rate=0.03,
        neck_learning_rate=0.04,
        head_learning_rate=0.05,
        task_learning_rate=0.06,
        weight_decay=0.2,
        task_weight_decay=0.3,
        betas=[0.8, 0.99],
        eps=1e-7,
    )
    optimizer = build_dinov3_optimizer(model, objective, component)
    actual = {
        id(parameter): (group["lr"], group["weight_decay"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert actual[id(model.backbone.blocks[0].weight)] == (0.0025, 0.2)
    assert actual[id(model.backbone.patch_embed.weight)] == (0.02, 0.2)
    assert actual[id(model.spm.weight)] == (0.03, 0.2)
    assert actual[id(model.neck.weight)] == (0.04, 0.2)
    assert actual[id(model.out.weight)] == (0.05, 0.2)
    assert actual[id(objective.log_variances)] == (0.06, 0.3)
    assert optimizer.defaults["betas"] == (0.8, 0.99)
    assert optimizer.defaults["eps"] == 1e-7


def test_shared_network_groups_keep_order_and_objective_group_is_separate(monkeypatch):
    model = _Adapter()
    objective = ManagedObjective(class_count=3, use_uncertainty_weighting=True)
    _, profile = initial_profile(classified=True)
    source_groups = optimization.build_param_groups
    captured = []

    def group_network(network, **kwargs):
        assert network is model
        groups = source_groups(network, **kwargs)
        captured.append(
            [
                (tuple(id(p) for p in g["params"]), g["lr"], g["weight_decay"])
                for g in groups
            ]
        )
        return groups

    monkeypatch.setattr(optimization, "build_param_groups", group_network)
    optimizer = build_dinov3_optimizer(
        model, objective, profile["trainer_components"]["optimizer"]
    )
    assert len(captured) == 1
    assert [
        (tuple(id(p) for p in g["params"]), g["lr"], g["weight_decay"])
        for g in optimizer.param_groups[:-1]
    ] == captured[0]
    task = optimizer.param_groups[-1]
    assert [id(p) for p in task["params"]] == [id(objective.log_variances)]
    assert (task["lr"], task["weight_decay"]) == (5e-6, 0.1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("parameter_group_rule", "head-only"),
        ("no_decay_rule", "all-parameters"),
        ("trainable_parameters_only", False),
        ("decoupled_weight_decay", False),
    ],
)
def test_optimizer_rejects_unsupported_recorded_parameter_rules(key, value):
    _, profile = initial_profile()
    component = profile["trainer_components"]["optimizer"]
    component["parameters"][key] = value
    with pytest.raises(ValueError, match="optimizer implementation"):
        build_dinov3_optimizer(_Adapter(), ManagedObjective(class_count=0), component)


def test_classified_cellpose_weights_change_both_classification_losses_only():
    capability, profile = initial_profile(classified=True)
    configuration = resolve_instance_configuration(
        profile,
        capability,
        {"profile_id": "none-v1", "enabled_operator_ids": []},
        _model_output(True),
        capability["default_training_settings"],
    )
    options = configuration["objective"]["parameters"]
    assert options["weight_tversky_by_class"] is True
    labels = torch.tensor([[[0, 0], [1, 2]]])
    batch = ManagedSegmentationBatch(
        images=torch.zeros(1, 3, 2, 2),
        directional_targets=torch.zeros(1, 2, 2, 2),
        foreground_targets=(labels > 0).float(),
        supervision_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        real_pixel_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        class_targets=labels,
    )
    prediction = torch.zeros(1, 6, 2, 2)
    prediction[:, 0] = 2.0
    kwargs = {
        name: options[name]
        for name in (
            "head_type",
            "class_count",
            "classification_objective",
            "classification_loss_weight",
            "weight_tversky_by_class",
            "tversky_alpha",
            "tversky_gamma",
        )
    }
    uniform = managed_segmentation_loss(
        prediction, batch, class_weights=torch.ones(3), **kwargs
    )
    weighted = managed_segmentation_loss(
        prediction, batch, class_weights=torch.tensor([1.0, 2.0, 3.0]), **kwargs
    )
    for name in ("classification_cross_entropy", "classification_tversky"):
        assert weighted.task_sums[name] != uniform.task_sums[name]
    torch.testing.assert_close(
        weighted.task_sums["segmentation"],
        uniform.task_sums["segmentation"],
        rtol=0,
        atol=0,
    )
