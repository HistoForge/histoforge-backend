"""Task-independent optimization over equally weighted model input fields.

The task callback validates each batch before its forward pass and computes the
loss. This module owns scheduling, backward, accumulation, clipping, cancellation
checks and synchronous epoch callbacks, not targets, inference or persistence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import Tensor, nn

Batch = TypeVar("Batch")


@dataclass(frozen=True)
class TrainingStepResult:
    """Task-combined sums over the actual fields in one microbatch.

    The task owns within-field reduction and the differentiable ``total_sum``.
    Diagnostics may overlap; the engine never sums ``component_sums`` into loss.
    ``field_count`` counts actual fields, not pixels or nominal batch capacity.
    """

    total_sum: Tensor
    component_sums: Mapping[str, Tensor]
    field_count: int


@dataclass(frozen=True)
class TrainingEpochResult:
    """Completed epoch values after equal-field reduction."""

    epoch: int
    field_count: int
    optimizer_steps: int
    mean_loss: float
    component_means: Mapping[str, float]


class TrainingCancelled(RuntimeError):
    """Raised after the caller's cancellation fence becomes active."""


def optimize(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_batches: Callable[[int], Iterable[Batch]],
    compute_loss: Callable[[nn.Module, Batch], TrainingStepResult],
    *,
    learning_rate_scales: Sequence[float],
    gradient_accumulation_steps: int,
    max_gradient_norm: float | None = None,
    objective: nn.Module | None = None,
    cancellation_requested: Callable[[], bool] = lambda: False,
    epoch_completed: Callable[[TrainingEpochResult, nn.Module], None] | None = None,
) -> tuple[TrainingEpochResult, ...]:
    """Optimize task-combined field sums without knowing the batch or model output.

    ``compute_loss`` must validate supervision before running the model, then
    return a differentiable sum over its actual fields. The loop divides all
    accumulated gradients, including learned objective gradients, by that field
    count. Only model gradients are clipped. An optional learned ``objective``
    must already belong to the optimizer; the loop validates ownership and sets
    train mode without choosing its scientific terms or calling it.

    The caller supplies fresh batches in its chosen order each epoch and owns
    validation and persistence in the synchronous callback. Errors propagate;
    no step or callback is retried. A zero learning-rate scale still runs
    backward and optimizer steps, preserving moments and step counters.
    """

    if not learning_rate_scales:
        raise ValueError("learning_rate_scales must be nonempty")
    if any(
        not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 0
        for scale in learning_rate_scales
    ):
        raise ValueError("learning_rate_scales must contain finite nonnegative values")
    if (
        not isinstance(gradient_accumulation_steps, int)
        or isinstance(gradient_accumulation_steps, bool)
        or gradient_accumulation_steps < 1
    ):
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if max_gradient_norm is not None and (
        not math.isfinite(max_gradient_norm) or max_gradient_norm <= 0
    ):
        raise ValueError("max_gradient_norm must be finite and positive when present")
    _require_optimizer_parameters(optimizer, model, owner="model")
    if objective is not None:
        _require_optimizer_parameters(optimizer, objective, owner="objective")

    initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if any(not math.isfinite(lr) or lr < 0 for lr in initial_lrs):
        raise ValueError("optimizer learning rates must be finite and nonnegative")
    scheduled_lrs = [
        [lr * float(scale) for lr in initial_lrs] for scale in learning_rate_scales
    ]
    if any(not math.isfinite(lr) for rates in scheduled_lrs for lr in rates):
        raise ValueError("effective optimizer learning rates must be finite")
    history: list[TrainingEpochResult] = []
    for epoch_index, learning_rates in enumerate(scheduled_lrs):
        _raise_if_cancelled(cancellation_requested)
        for group, learning_rate in zip(optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate

        model.train()
        if objective is not None:
            objective.train()
        optimizer.zero_grad(set_to_none=True)
        pending_microbatches = 0
        pending_fields = 0
        epoch_fields = 0
        optimizer_steps = 0
        total_sum = 0.0
        component_sums: dict[str, float] = {}

        for batch in epoch_batches(epoch_index):
            _raise_if_cancelled(cancellation_requested)
            result = compute_loss(model, batch)
            if not bool(torch.isfinite(result.total_sum)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("managed objective produced a non-finite loss")
            result.total_sum.backward()
            pending_microbatches += 1
            pending_fields += result.field_count
            epoch_fields += result.field_count
            total_sum += float(result.total_sum.detach())
            for name, value in result.component_sums.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(
                    value.detach()
                )
            if pending_microbatches == gradient_accumulation_steps:
                _optimizer_step(
                    model,
                    optimizer,
                    pending_fields,
                    max_gradient_norm,
                    cancellation_requested,
                )
                pending_microbatches = 0
                pending_fields = 0
                optimizer_steps += 1

        if pending_microbatches:
            _optimizer_step(
                model,
                optimizer,
                pending_fields,
                max_gradient_norm,
                cancellation_requested,
            )
            optimizer_steps += 1
        if epoch_fields == 0:
            raise ValueError("every epoch must contain at least one field")

        epoch_result = TrainingEpochResult(
            epoch=epoch_index + 1,
            field_count=epoch_fields,
            optimizer_steps=optimizer_steps,
            mean_loss=total_sum / epoch_fields,
            component_means={
                name: value / epoch_fields for name, value in component_sums.items()
            },
        )
        _raise_if_cancelled(cancellation_requested)
        history.append(epoch_result)
        if epoch_completed is not None:
            epoch_completed(epoch_result, model)
    return tuple(history)


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    field_count: int,
    max_gradient_norm: float | None,
    cancellation_requested: Callable[[], bool],
) -> None:
    """Normalize accumulated gradients by fields, clip, and update once.

    Cancellation is checked immediately before mutation so a cancelled run
    cannot publish one additional optimizer step after observing the fence.
    Clipping retains the established model-only norm. Objective parameters are
    still normalized and checked for finiteness, but do not change that norm.
    """

    parameters = _optimizer_parameters(optimizer)
    try:
        _raise_if_cancelled(cancellation_requested)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(field_count)
        _require_finite_gradients(parameters)
        if max_gradient_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_gradient_norm, error_if_nonfinite=False
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(
                    "managed objective produced a non-finite gradient norm"
                )
            _require_finite_gradients(parameters)
        _raise_if_cancelled(cancellation_requested)
        optimizer.step()
    finally:
        optimizer.zero_grad(set_to_none=True)


def _require_finite_gradients(parameters: Sequence[nn.Parameter]) -> None:
    """Check every optimizer-owned gradient with one synchronization per device."""

    checks: dict[torch.device, list[Tensor]] = {}
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad
            if gradient.is_sparse:
                gradient = gradient._values()
            checks.setdefault(gradient.device, []).append(
                torch.isfinite(gradient).all()
            )
    if any(not bool(torch.stack(values).all()) for values in checks.values()):
        raise FloatingPointError("managed objective produced a non-finite gradient")


def _raise_if_cancelled(cancellation_requested: Callable[[], bool]) -> None:
    """Raise the managed cancellation signal when the caller's fence is set."""

    if cancellation_requested():
        raise TrainingCancelled("managed training was cancelled")


def _optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Parameter, ...]:
    """Return every optimizer-owned parameter exactly once."""

    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                parameters.append(parameter)
    return tuple(parameters)


def _require_optimizer_parameters(
    optimizer: torch.optim.Optimizer, module: nn.Module, *, owner: str
) -> None:
    """Ensure every trainable module parameter belongs to the optimizer."""

    owned = {id(parameter) for parameter in _optimizer_parameters(optimizer)}
    missing = [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad and id(parameter) not in owned
    ]
    if missing:
        raise ValueError(f"optimizer does not own all trainable {owner} parameters")
