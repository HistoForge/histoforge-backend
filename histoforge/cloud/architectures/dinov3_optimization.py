"""Construct the DINOv3 optimizer from recorded parameter-group policy.

The scientific source owns network grouping. This architecture adapter binds it
and learned objective parameters to the exact managed AdamW configuration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from histoforge.cloud.training.component_schema import (
    ADAMW_PROFILE_ID,
    NO_DECAY_RULE,
    PARAMETER_GROUP_RULE,
)

if TYPE_CHECKING:
    import torch
    from torch import nn


def managed_dinov3_optimizer_component() -> dict[str, Any]:
    """The shared managed DINOv3 parameter-group and AdamW policy."""

    return {
        "profile_id": ADAMW_PROFILE_ID,
        "parameters": {
            "parameter_group_rule": PARAMETER_GROUP_RULE,
            "trainable_parameters_only": True,
            "learning_rate": 5e-5,
            "llrd_factor": 0.8,
            "patch_embed_learning_rate": 3e-4,
            "adapter_learning_rate": 1e-3,
            "neck_learning_rate": 1e-3,
            "head_learning_rate": 1e-3,
            "weight_decay": 0.1,
            "decoupled_weight_decay": True,
            "no_decay_rule": NO_DECAY_RULE,
            "task_learning_rate": 5e-6,
            "task_weight_decay": 0.1,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
        },
    }


def build_dinov3_optimizer(
    model: nn.Module,
    objective: nn.Module,
    optimizer_component: Mapping[str, Any],
) -> torch.optim.AdamW:
    """Own each trainable parameter once, without unfreezing fixed model state.

    Reuses the source ``build_param_groups`` name/rank rules. The separately
    learned task parameters retain AdamW decay, including their rank-one vector.
    The caller constructs the model and objective on the training device first.
    """

    import torch

    from histoforge._vendor.histoforge_models.pathology.optimization import (
        build_param_groups,
    )

    parameters = optimizer_component["parameters"]
    if (
        optimizer_component["profile_id"] != ADAMW_PROFILE_ID
        or parameters["parameter_group_rule"] != PARAMETER_GROUP_RULE
        or parameters["no_decay_rule"] != NO_DECAY_RULE
        or parameters["trainable_parameters_only"] is not True
        or parameters["decoupled_weight_decay"] is not True
    ):
        raise ValueError("resolved optimizer implementation is not supported")
    groups = build_param_groups(
        model,
        base_lr=parameters["learning_rate"],
        llrd_factor=parameters["llrd_factor"],
        patch_embed_lr=parameters["patch_embed_learning_rate"],
        weight_decay=parameters["weight_decay"],
        neck_lr=parameters["neck_learning_rate"],
        head_lr=parameters["head_learning_rate"],
        adapter_lr=parameters["adapter_learning_rate"],
    )
    task_parameters = [p for p in objective.parameters() if p.requires_grad]
    if task_parameters:
        groups.append(
            {
                "params": task_parameters,
                "lr": parameters["task_learning_rate"],
                "weight_decay": parameters["task_weight_decay"],
            }
        )
    return torch.optim.AdamW(
        groups,
        lr=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
        betas=tuple(parameters["betas"]),
        **{
            key: parameters[key]
            for key in (
                "eps",
                "amsgrad",
                "maximize",
                "foreach",
                "capturable",
                "differentiable",
                "fused",
            )
        },
    )
