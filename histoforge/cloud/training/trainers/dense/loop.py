"""Bind dense fields and full-ROI validation to the common optimization engine."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

# Establish the source's targets-before-Torch import order.
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ClassificationObjective,
    ManagedDenseBatch,
    ManagedObjective,
    managed_dense_loss,
)
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.training.component_schema import (
    DENSE_OBJECTIVE_PROFILE_ID,
    require_managed_objective_rules,
)
from histoforge.cloud.training.datasets.class_weights import (
    validated_smoothed_class_weights,
)
from histoforge.cloud.training.datasets.files import RetainedTrainingDataset
from histoforge.cloud.training.engine import (
    TrainingEpochResult,
    TrainingStepResult,
    optimize,
)
from histoforge.cloud.training.epochs import (
    require_fresh_training_dataset,
    run_training_epochs,
)
from histoforge.cloud.training.schedule import managed_learning_rate_scales
from histoforge.cloud.training.trainers.dense.batches import collate_dense_fields
from histoforge.cloud.training.trainers.dense.sampling import DenseEpochDataset
from histoforge.cloud.training.trainers.dense.validation import (
    evaluate_dense_validation,
)

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch import Tensor, nn


def train_managed_dense_model(
    model: nn.Module,
    report: Mapping[str, Any],
    dataset: RetainedTrainingDataset,
    predict: Callable[[RasterizedImage], Mapping[str, Any]],
    *,
    build_optimizer: Callable[
        [nn.Module, nn.Module, Mapping[str, Any]], torch.optim.Optimizer
    ],
    prepare_model: Callable[[nn.Module], None],
    training_forward: Callable[[nn.Module, Tensor], Tensor],
    preprocess: Callable[..., np.ndarray],
    capability: Mapping[str, Any],
    device: torch.device | str,
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, int],
    target_raster_limits: Mapping[str, int],
    epoch_completed: Callable[[Mapping[str, Any]], None],
    cancellation_requested: Callable[[], bool] = lambda: False,
) -> tuple[dict[str, Any], ...]:
    """Execute immutable settings and publish only fully validated epochs.

    The caller owns worker claims, checkpoints and durable state. Validation uses
    the same network and ordinary inference profile, not training batch size.
    Failure or cancellation prevents the epoch callback; no partial score or retry.
    """

    import torch

    require_fresh_training_dataset(report, dataset)
    if report["task"]["model_output"]["type"] != "dense_classification":
        raise ValueError("dense training requires a dense-classification task")
    configuration = report["trainer_configuration"]
    require_managed_objective_rules(
        configuration["objective"], profile_id=DENSE_OBJECTIVE_PROFILE_ID
    )
    scales = managed_learning_rate_scales(configuration["scheduler"])
    if len(scales) != report["total_epochs"]:
        raise ValueError("epoch schedule differs from the accepted report")
    options = configuration["objective"]["parameters"]
    class_count = options["class_count"]
    weights = validated_smoothed_class_weights(
        options["class_weighting"],
        report["task"]["model_output"],
        class_count,
    )
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(report["training_seed"])
    model.to(device=device, dtype=torch.float32)
    prepare_model(model)
    objective = ManagedObjective(
        class_count=class_count,
        classification_objective=options["classification_objective"],
        classification_loss_weight=options["classification_loss_weight"],
        use_uncertainty_weighting=options["use_uncertainty_weighting"],
        include_segmentation=False,
    ).to(device)
    optimizer = build_optimizer(model, objective, configuration["optimizer"])
    settings, profile = report["trainer_settings"], report["field_profile"]

    return run_training_epochs(
        report,
        device=device,
        epoch_dataset=lambda epoch_index: DenseEpochDataset(
            dataset,
            report["split_plan"],
            capability=capability,
            field_profile=profile,
            sampler=configuration["sampler"],
            augmentation=configuration["augmentation"],
            training_seed=report["training_seed"],
            epoch_index=epoch_index,
            api_limits=api_limits,
            source_limits=source_limits,
        ),
        collate_fields=partial(
            collate_dense_fields,
            field_profile=profile,
            class_count=class_count,
            preprocess=preprocess,
        ),
        train_fields=partial(
            train_dense_fields,
            model,
            optimizer,
            training_forward=training_forward,
            class_count=class_count,
            learning_rate_scales=scales,
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            class_weights=class_weights,
            objective=objective,
            classification_objective=options["classification_objective"],
            classification_loss_weight=options["classification_loss_weight"],
            weight_tversky_by_class=options["weight_tversky_by_class"],
            tversky_alpha=options["tversky_alpha"],
            tversky_gamma=options["tversky_gamma"],
            max_gradient_norm=options["max_gradient_norm"],
        ),
        validate=partial(
            evaluate_dense_validation,
            dataset,
            report["split_plan"],
            capability=capability,
            field_profile=profile,
            api_limits=api_limits,
            source_limits=source_limits,
            target_raster_limits=target_raster_limits,
        ),
        predict=predict,
        cancellation_requested=cancellation_requested,
        epoch_completed=epoch_completed,
    )


def train_dense_fields(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_batches: Callable[[int], Iterable[ManagedDenseBatch]],
    *,
    training_forward: Callable[[nn.Module, Tensor], Tensor],
    class_count: int,
    learning_rate_scales: Sequence[float],
    gradient_accumulation_steps: int,
    class_weights: Tensor | None = None,
    objective: ManagedObjective | None = None,
    classification_objective: ClassificationObjective = "cross_entropy+tversky",
    classification_loss_weight: float = 1.0,
    weight_tversky_by_class: bool = True,
    tversky_alpha: float = 0.3,
    tversky_gamma: float = 1.33,
    max_gradient_norm: float | None = None,
    cancellation_requested: Callable[[], bool] = lambda: False,
    epoch_completed: Callable[[TrainingEpochResult, nn.Module], None] | None = None,
) -> tuple[TrainingEpochResult, ...]:
    """Bind only semantic class logits to field-equal CE/Tversky optimization."""

    def compute_loss(
        trained_model: nn.Module, batch: ManagedDenseBatch
    ) -> TrainingStepResult:
        batch.validate(class_count)
        logits = training_forward(trained_model, batch.images)
        result = managed_dense_loss(
            logits,
            batch,
            class_count=class_count,
            class_weights=class_weights,
            objective=objective,
            classification_objective=classification_objective,
            classification_loss_weight=classification_loss_weight,
            weight_tversky_by_class=weight_tversky_by_class,
            tversky_alpha=tversky_alpha,
            tversky_gamma=tversky_gamma,
        )
        return TrainingStepResult(
            result.total_sum, result.component_sums, result.field_count
        )

    return optimize(
        model,
        optimizer,
        epoch_batches,
        compute_loss,
        learning_rate_scales=learning_rate_scales,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_gradient_norm=max_gradient_norm,
        objective=objective,
        cancellation_requested=cancellation_requested,
        epoch_completed=epoch_completed,
    )
