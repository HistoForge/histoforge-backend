"""Saved settings drive real losses, optimizer steps, and complete validation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.cloud.architectures import dinov3_optimization
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.modules.schema import (
    model_output_class_channels,
    model_output_target_classes,
)
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.datasets.class_weights import (
    count_training_class_pixels,
    smoothed_class_weights,
    validated_smoothed_class_weights,
)
from histoforge.cloud.training.engine import TrainingCancelled
from histoforge.cloud.training.metric_policy import TRAINING_LOSS_SERIES_ID
from histoforge.cloud.training.runs.report import validate_training_report
from histoforge.cloud.training.trainers.instance.configuration import (
    resolve_instance_configuration,
)
from histoforge.cloud.training.trainers.instance.loop import train_managed_model
from histoforge.cloud.training.trainers.instance.policy import (
    prepare_instance_metric_policy,
)
from tests.test_training_batches import _profile
from tests.test_training_capabilities import _capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _rectangle
from tests.test_training_field_reader import _source_limits
from tests.test_training_report import queued_report
from tests.test_training_validation import _capture, _object
from torch import nn


class _TinyModel(nn.Module):
    """Use actual gradients with the production optimizer's parameter names."""

    def __init__(self, channels):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Conv2d(3, 4, 1)
        self.backbone.blocks = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.spm = nn.Conv2d(4, 4, 1)
        self.neck = nn.Conv2d(4, 4, 1)
        self.out = nn.Conv2d(4, channels, 1)
        self.W2 = nn.Parameter(torch.eye(2), requires_grad=False)

    def forward(self, image):
        value = self.backbone.patch_embed(image)
        for block in self.backbone.blocks:
            value = torch.tanh(block(value))
        return self.out(self.neck(self.spm(value)))


def _setup(tmp_path, *, head="flow", classified=True):
    dataset, split, capability = _capture(
        tmp_path,
        profile=_profile(),
        classified=classified,
        validation_rois=2,
        objects=[_object(1, _rectangle(12, 12, 20, 20))],
    )
    # Reuse ordinary no-augmentation presentation while retaining captured task.
    control = _capability()
    for name in (
        "augmentation_profiles",
        "default_augmentation_profile_id",
        "default_training_settings",
        "training_gpus",
        "total_epochs_range",
        "automatic_warmup",
    ):
        capability[name] = control[name]
    if head == "flow":
        capability["flow_loss_weighting_options"] = control[
            "flow_loss_weighting_options"
        ]
    else:
        capability.pop("flow_loss_weighting_options", None)
        capability["default_training_settings"].pop("flow_loss_weighting")
    profile = initial_managed_trainer_profile(
        "dinov3-adapter-cellpose" if head == "flow" else "dinov3-adapter-hovernet",
        capability,
        augmentation_profiles={"none-v1": {}},
        gpu_resources={},
        trainer_execution_seconds=3600,
    )
    # Small execution fixture only. Independent schedule tests cover the default.
    settings = {
        **capability["default_training_settings"],
        "total_epochs": 3,
        "warmup_mode": "custom",
        "warmup_epochs": 2,
    }
    selection = {"profile_id": "none-v1", "enabled_operator_ids": []}
    configuration = resolve_instance_configuration(
        profile,
        capability,
        selection,
        dataset.manifest["task"]["model_output"],
        settings,
    )
    options = configuration["objective"]["parameters"]
    if classified:
        counts = count_training_class_pixels(
            dataset,
            (0,),
            capability=capability,
            field_profile=capability["default_field_profile"],
            max_json_member_bytes=training_api_limits()["dataset"][
                "max_json_member_bytes"
            ],
        )
        weights = smoothed_class_weights(counts)
        options["class_weighting"]["channels"] = [
            {
                "channel": channel["channel"],
                "role": channel["role"],
                **(
                    {"path_class": channel["pathClass"]}
                    if "pathClass" in channel
                    else {}
                ),
                "raw_count": count,
                "weight": weight,
            }
            for channel, count, weight in zip(
                model_output_class_channels(dataset.manifest["task"]["model_output"]),
                counts,
                weights,
                strict=True,
            )
        ]
    report = queued_report()
    report.update(
        task=dataset.manifest["task"],
        source_model_output=copy.deepcopy(dataset.manifest["task"]["model_output"]),
        source_model_microns_per_pixel=dataset.manifest["task"][
            "model_microns_per_pixel"
        ],
        dataset_digest=dataset.manifest["dataset_digest"],
        dataset_image_count=1,
        split_plan=split,
        field_profile=capability["default_field_profile"],
        trainer_settings=settings,
        augmentation=selection,
        trainer_configuration=configuration,
        metric_policy=prepare_instance_metric_policy(
            SimpleNamespace(trainer_profile=profile),
            dataset.manifest["task"]["model_output"],
        ),
        total_epochs=3,
    )
    for key, roi_count in (("training_counts", 1), ("validation_counts", 2)):
        classes = model_output_target_classes(dataset.manifest["task"]["model_output"])
        report[key].update(
            roi_count=roi_count,
            target_object_counts={
                path_class: roi_count if index == 0 else 0
                for index, path_class in enumerate(classes)
            },
        )
    report = validate_training_report(report, training_api_limits())
    torch.manual_seed(55)
    model = _TinyModel((3 if head == "flow" else 4) + options["class_count"])
    return model, report, dataset, capability


def _execute(setup, predict, completed, **kwargs):
    model, report, dataset, capability = setup
    return train_managed_model(
        model,
        report,
        dataset,
        predict,
        preprocess=normalize99,
        capability=capability,
        device="cpu",
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
        target_raster_limits={"max_width": 128, "max_height": 128, "max_pixels": 16384},
        max_assignment_matrix_entries=100,
        epoch_completed=completed,
        **{
            "build_optimizer": dinov3_optimization.build_dinov3_optimizer,
            "training_forward": instance_training_predictions,
            **kwargs,
        },
    )


def test_ce_weight_records_are_bound_by_channel_not_array_position():
    output = {
        "type": "instance_segmentation_and_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "training_only", "pathClass": "Other"},
            {"channel": 2, "role": "emitted", "pathClass": "Cell"},
        ],
    }
    counts = [100, 25, 4]
    weights = smoothed_class_weights(counts)
    records = [
        {
            "channel": 2,
            "role": "emitted",
            "path_class": "Cell",
            "raw_count": counts[2],
            "weight": weights[2],
        },
        {
            "channel": 0,
            "role": "background",
            "raw_count": counts[0],
            "weight": weights[0],
        },
        {
            "channel": 1,
            "role": "training_only",
            "path_class": "Other",
            "raw_count": counts[1],
            "weight": weights[1],
        },
    ]

    weighting = {"channels": records}
    assert validated_smoothed_class_weights(weighting, output, 3) == weights

    records[0]["path_class"] = "Other"
    with pytest.raises(ValueError, match="PathClass differs"):
        validated_smoothed_class_weights(weighting, output, 3)


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("classified", [False, True])
def test_real_optimization_validates_every_roi_before_each_epoch(
    tmp_path, monkeypatch, head, classified
):
    setup = _setup(tmp_path, head=head, classified=classified)
    model, report, _, _ = setup
    original_report = copy.deepcopy(report)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    optimizer_holder = []
    ordinary_build = dinov3_optimization.build_dinov3_optimizer

    def build(*args):
        result = ordinary_build(*args)
        optimizer_holder.append(result)
        return result

    monkeypatch.setattr(dinov3_optimization, "build_dinov3_optimizer", build)
    predicted, epochs = [], []

    def predict(image):
        assert not model.training and torch.is_inference_mode_enabled()
        assert image.pixels.shape == (20, 20, 3)
        predicted.append(image.pixels.shape)
        mask = np.zeros((20, 20), np.int32)
        mask[2:10, 2:10] = 7
        return {"instance_mask": mask, "votes_per_instance": {7: 1}}

    def complete(epoch):
        assert len(predicted) == epoch["epoch"] * 2
        assert model.training
        epochs.append(copy.deepcopy(epoch))
        if epoch["epoch"] == 1:
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, original[name], rtol=0, atol=0)
            optimizer = optimizer_holder[0]
            assert all(group["lr"] == 0 for group in optimizer.param_groups)
            assert all(state["step"].item() == 1 for state in optimizer.state.values())
            assert any(
                state["exp_avg"].abs().sum() > 0 for state in optimizer.state.values()
            )

    result = _execute(setup, predict, complete)
    assert tuple(epochs) == result
    assert report == original_report
    assert [e["epoch"] for e in epochs] == [1, 2, 3]
    for epoch in epochs:
        assert [m["series_id"] for m in epoch["metrics"]] == [
            s["series_id"] for s in report["metric_policy"]["series"]
        ]
        assert all(m["value"] == 1 for m in epoch["metrics"][1:])
    assert any(
        not torch.equal(value, original[name])
        for name, value in model.state_dict().items()
    )
    torch.testing.assert_close(model.W2, original["W2"], rtol=0, atol=0)
    validate_training_report(
        {**report, "state": "RUNNING", "epochs": epochs}, training_api_limits()
    )


@pytest.mark.parametrize("failure", ["validation", "cancellation", "publication"])
def test_failed_validation_cancellation_or_publication_does_not_continue(
    tmp_path, failure
):
    setup = _setup(tmp_path)
    predictions, publications = [], []
    cancelled = False

    def predict(image):
        nonlocal cancelled
        predictions.append(1)
        if failure == "validation" and len(predictions) == 2:
            raise RuntimeError("validation failed")
        if failure == "cancellation" and len(predictions) == 2:
            cancelled = True
        return {
            "instance_mask": np.zeros(image.pixels.shape[:2], np.int32),
            "votes_per_instance": {},
        }

    def complete(epoch):
        publications.append(epoch)
        raise RuntimeError("publication failed")

    exception = TrainingCancelled if failure == "cancellation" else RuntimeError
    with pytest.raises(exception):
        _execute(setup, predict, complete, cancellation_requested=lambda: cancelled)
    assert len(predictions) == 2
    assert len(publications) == (1 if failure == "publication" else 0)
    assert setup[0].training


def test_zero_validation_is_supported_without_invented_metrics(tmp_path):
    setup = _setup(tmp_path, classified=False)
    setup[1]["split_plan"] = {"type": "none"}
    setup[1].pop("validation_counts")
    setup[1]["training_counts"].update(
        roi_count=3, target_object_counts={"Leydig cell": 3}
    )

    def unexpected(_image):
        pytest.fail("no validation was selected")

    epochs = _execute(setup, unexpected, lambda _epoch: None)
    assert all(
        [m["series_id"] for m in e["metrics"]] == [TRAINING_LOSS_SERIES_ID]
        for e in epochs
    )


@pytest.mark.parametrize("workers", [0, 1, 4])
def test_user_batch_accumulation_and_worker_settings_preserve_field_weighting(
    tmp_path, workers
):
    """Three real fields form one update, including the partial second batch."""

    reference = _setup(tmp_path / "reference")
    grouped = _setup(tmp_path / "grouped")
    for setup, batch_size, accumulation, worker_count in (
        (reference, 1, 3, 0),
        (grouped, 2, 2, workers),
    ):
        report = setup[1]
        report["split_plan"] = {"type": "none"}
        report.pop("validation_counts")
        report["training_counts"].update(
            roi_count=3,
            target_object_counts={"Leydig cell": 3, "Other": 0},
        )
        report["trainer_settings"].update(
            training_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            data_loading_workers=worker_count,
        )
        _execute(setup, lambda _: pytest.fail("no validation"), lambda _: None)
    for name, value in reference[0].state_dict().items():
        torch.testing.assert_close(
            value, grouped[0].state_dict()[name], rtol=1e-5, atol=1e-7
        )
