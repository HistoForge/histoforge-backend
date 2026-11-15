"""Bind instance fields, losses and validation to shared epoch execution.

The concrete trainer supplies model behavior and optimizer construction. This
module owns task configuration and scientific loss/validation bindings; epochs.py
owns loading and publication order, and engine.py owns optimization. Run state,
checkpoints and durable publication remain with the worker and runs/.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ClassificationObjective,
    FlowLossWeighting,
    HeadType,
    ManagedObjective,
    ManagedSegmentationBatch,
    managed_segmentation_loss,
    validate_managed_loss_configuration,
)
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.training.component_schema import (
    OBJECTIVE_PROFILE_ID,
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
from histoforge.cloud.training.trainers.instance.batches import collate_training_fields
from histoforge.cloud.training.trainers.instance.policy import (
    VALIDATION_AP50_SERIES_ID,
    VALIDATION_MSA_SERIES_ID,
    VALIDATION_OBJECT_MACRO_F1_SERIES_ID,
    object_class_f1_series_id,
)
from histoforge.cloud.training.trainers.instance.sampling import TrainingEpochDataset
from histoforge.cloud.training.trainers.instance.validation import (
    evaluate_bounded_training_validation,
)

if TYPE_CHECKING:
    import numpy as np
    import torch
    from torch import Tensor, nn


def train_managed_model(
    model: nn.Module,
    report: Mapping[str, Any],
    dataset: RetainedTrainingDataset,
    predict: Callable[[RasterizedImage], Mapping[str, Any]],
    *,
    build_optimizer: Callable[
        [nn.Module, nn.Module, Mapping[str, Any]], torch.optim.Optimizer
    ],
    training_forward: Callable[[nn.Module, Tensor], Tensor],
    preprocess: Callable[..., np.ndarray],
    capability: Mapping[str, Any],
    device: torch.device | str,
    api_limits: Mapping[str, Any],
    source_limits: Mapping[str, int],
    target_raster_limits: Mapping[str, int],
    max_assignment_matrix_entries: int,
    epoch_completed: Callable[[Mapping[str, Any]], None],
    cancellation_requested: Callable[[], bool] = lambda: False,
) -> tuple[dict[str, Any], ...]:
    """Train from the immutable report and publish only fully evaluated epochs.

    ``report`` is the acknowledged, validated report for a freshly claimed run.
    ``model`` is its exact-loaded canonical model, not an inference wrapper.
    ``predict`` must decode that same model using its accepted field profile and
    normal inference settings. Every held-out ROI is evaluated after every epoch;
    validation failure or cancellation prevents that epoch's output callback.

    The callback connects to the existing checkpoint/report publication owner.
    An exception stops execution; neither training nor publication is retried.
    No checkpoint is reloaded, module installed, or run state changed here.
    """

    # Let the managed package establish its targets-before-Torch import order.
    import torch

    require_fresh_training_dataset(report, dataset)
    configuration = report["trainer_configuration"]
    component = configuration["objective"]
    require_managed_objective_rules(component, profile_id=OBJECTIVE_PROFILE_ID)
    options = component["parameters"]
    class_count = options["class_count"]
    weighting = options["class_weighting"]
    scales = managed_learning_rate_scales(configuration["scheduler"])
    if len(scales) != report["total_epochs"]:
        raise ValueError("epoch schedule differs from the accepted report")
    # The count pass ran before acceptance. Never recompute it from shuffled or
    # augmented fields, a validation set, or the contents of a later dataset.
    class_weights = None
    if class_count:
        resolved_weights = validated_smoothed_class_weights(
            weighting, report["task"]["model_output"], class_count
        )
        class_weights = torch.tensor(
            resolved_weights, dtype=torch.float32, device=device
        )
    torch.manual_seed(report["training_seed"])
    model.to(device=device, dtype=torch.float32)
    objective = ManagedObjective(
        class_count=class_count,
        classification_objective=options["classification_objective"],
        classification_loss_weight=options["classification_loss_weight"],
        use_uncertainty_weighting=options["use_uncertainty_weighting"],
    ).to(device)
    optimizer = build_optimizer(model, objective, configuration["optimizer"])
    settings = report["trainer_settings"]
    profile = report["field_profile"]
    fields = None

    def epoch_dataset(epoch_index):
        # Prepare once, inside the existing cancellation/timing boundary. Retain
        # only coordinates and probabilities, not image pixels or target arrays.
        nonlocal fields
        if fields is None:
            fields = TrainingEpochDataset(
                dataset,
                report["split_plan"],
                capability=capability,
                field_profile=profile,
                sampler=configuration["sampler"],
                augmentation=configuration["augmentation"],
                training_seed=report["training_seed"],
                epoch_index=epoch_index,
                head_type=options["head_type"],
                api_limits=api_limits,
                source_limits=source_limits,
                class_pixel_counts=(
                    {
                        item["channel"]: item["raw_count"]
                        for item in weighting["channels"]
                    }
                    if class_count
                    else None
                ),
            )
            return fields
        return fields.for_epoch(epoch_index)

    def validate(checked_predict):
        validation = evaluate_bounded_training_validation(
            dataset,
            report["split_plan"],
            checked_predict,
            capability=capability,
            field_profile=profile,
            api_limits=api_limits,
            source_limits=source_limits,
            target_raster_limits=target_raster_limits,
            max_assignment_matrix_entries=max_assignment_matrix_entries,
        )
        if validation is None:
            return {}
        return {
            VALIDATION_AP50_SERIES_ID: validation.ap50,
            VALIDATION_MSA_SERIES_ID: validation.msa,
            VALIDATION_OBJECT_MACRO_F1_SERIES_ID: validation.object_macro_f1,
            **{
                object_class_f1_series_id(channel): value
                for channel, value in validation.object_class_f1.items()
            },
        }

    return run_training_epochs(
        report,
        device=device,
        epoch_dataset=epoch_dataset,
        collate_fields=partial(
            collate_training_fields,
            field_profile=profile,
            class_count=class_count,
            preprocess=preprocess,
        ),
        train_fields=partial(
            train_instance_fields,
            model,
            optimizer,
            training_forward=training_forward,
            head_type=options["head_type"],
            class_count=class_count,
            learning_rate_scales=scales,
            gradient_accumulation_steps=settings["gradient_accumulation_steps"],
            classification_objective=options["classification_objective"],
            classification_loss_weight=options["classification_loss_weight"],
            class_weights=class_weights,
            weight_tversky_by_class=options["weight_tversky_by_class"],
            tversky_alpha=options["tversky_alpha"],
            tversky_gamma=options["tversky_gamma"],
            **(
                {"flow_loss_weighting": options["flow_loss_weighting"]}
                if options["head_type"] == "flow"
                else {}
            ),
            max_gradient_norm=options["max_gradient_norm"],
            objective=objective,
        ),
        validate=validate,
        predict=predict,
        cancellation_requested=cancellation_requested,
        epoch_completed=epoch_completed,
    )


def train_instance_fields(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_batches: Callable[[int], Iterable[ManagedSegmentationBatch]],
    *,
    training_forward: Callable[[nn.Module, Tensor], Tensor],
    head_type: HeadType,
    class_count: int,
    learning_rate_scales: Sequence[float],
    gradient_accumulation_steps: int,
    classification_objective: ClassificationObjective = "cross_entropy",
    classification_loss_weight: float = 1.0,
    class_weights: Tensor | None = None,
    weight_tversky_by_class: bool = True,
    tversky_alpha: float = 0.3,
    tversky_gamma: float = 1.33,
    max_gradient_norm: float | None = None,
    objective: ManagedObjective | None = None,
    cancellation_requested: Callable[[], bool] = lambda: False,
    epoch_completed: Callable[[TrainingEpochResult, nn.Module], None] | None = None,
    flow_loss_weighting: FlowLossWeighting = "pixel",
) -> tuple[TrainingEpochResult, ...]:
    """Bind flow/HoVer supervision and losses to the common optimizer engine.

    Validate configuration before optimization and every batch before forward.
    The scientific loss owns within-field reduction and task combination; this
    binding preserves its tensors and field count without detaching or reweighting.
    The caller owns sampling, validation, checkpointing and durable state.
    """

    validate_managed_loss_configuration(
        head_type,
        class_count,
        classification_objective,
        classification_loss_weight,
        class_weights,
        weight_tversky_by_class,
        tversky_alpha,
        tversky_gamma,
        flow_loss_weighting,
    )
    if objective is not None:
        objective.validate_loss_configuration(
            class_count=class_count,
            classification_objective=classification_objective,
            classification_loss_weight=classification_loss_weight,
        )

    def compute_loss(
        trained_model: nn.Module, batch: ManagedSegmentationBatch
    ) -> TrainingStepResult:
        batch.validate(class_count, flow_loss_weighting=flow_loss_weighting)
        predictions = training_forward(trained_model, batch.images)
        result = managed_segmentation_loss(
            predictions,
            batch,
            head_type=head_type,
            class_count=class_count,
            classification_objective=classification_objective,
            classification_loss_weight=classification_loss_weight,
            class_weights=class_weights,
            weight_tversky_by_class=weight_tversky_by_class,
            tversky_alpha=tversky_alpha,
            tversky_gamma=tversky_gamma,
            objective=objective,
            flow_loss_weighting=flow_loss_weighting,
        )
        return TrainingStepResult(
            total_sum=result.total_sum,
            component_sums=result.component_sums,
            field_count=result.field_count,
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
