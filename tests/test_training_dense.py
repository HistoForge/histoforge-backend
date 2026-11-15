"""Dense scientific adapter tests from retained capture to actual CPU optimization."""

import copy
from functools import partial

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedObjective,
)
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.cloud.architectures.dinov3_optimization import build_dinov3_optimizer
from histoforge.cloud.architectures.semantic.runtime import (
    create_semantic_runtime,
    load_semantic_model_state,
    prepare_semantic_finetuning,
    semantic_training_logits,
)
from histoforge.cloud.training.engine import TrainingCancelled
from histoforge.cloud.training.runs.report import (
    validate_training_report,
)
from histoforge.cloud.training.runs.request import (
    validate_training_run_request,
    validate_training_run_request_against_manifest_capability,
)
from histoforge.cloud.training.trainers.dense.batches import collate_dense_fields
from histoforge.cloud.training.trainers.dense.fields import DenseTrainingField
from histoforge.cloud.training.trainers.dense.loop import (
    train_dense_fields,
    train_managed_dense_model,
)
from histoforge.cloud.training.trainers.dense.policy import (
    VALIDATION_PIXEL_MACRO_F1_SERIES_ID,
    pixel_class_f1_series_id,
)
from histoforge.cloud.training.trainers.dense.sampling import DenseEpochDataset
from histoforge.cloud.training.trainers.dense.validation import (
    dense_metric_values,
    evaluate_dense_validation,
    pixel_confusion,
)
from tests.test_training_batches import single_thread_torch as single_thread_torch
from tests.test_training_dataset import training_api_limits
from tests.test_training_field_reader import _source_limits
from tests.training_dense_fixtures import dense_output, dense_setup
from torch import nn


class TinySemantic(nn.Module):
    """Small network with the class-prefix/flow-tail contract; not DINO numerics."""

    def __init__(self):
        super().__init__()
        self.nclasses = 4
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Conv2d(3, 4, 1)
        self.backbone.blocks = nn.ModuleList([nn.Conv2d(4, 4, 1)])
        self.out_class = nn.Conv2d(4, 4, 1)
        self.out = nn.Conv2d(4, 3, 1)
        self.W2 = nn.Parameter(torch.ones(3))
        self.register_buffer("class_count", torch.tensor(4))

    @property
    def device(self):
        return self.out.weight.device

    @property
    def dtype(self):
        return self.out.weight.dtype

    def forward(self, image):
        value = self.backbone.patch_embed(image)
        for block in self.backbone.blocks:
            value = torch.tanh(block(value))
        output = torch.cat(
            (self.out_class(value), self.out(value) * self.W2[None, :, None, None]),
            dim=1,
        )
        return output, torch.zeros(len(image), 256, device=image.device)


def _validation_kwargs(setup):
    return dict(
        capability=setup.entry.capability,
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
        target_raster_limits={"max_width": 128, "max_height": 128, "max_pixels": 16384},
    )


def _training_kwargs(setup):
    return {
        **_validation_kwargs(setup),
        "build_optimizer": build_dinov3_optimizer,
        "prepare_model": prepare_semantic_finetuning,
        "training_forward": partial(semantic_training_logits, class_count=4),
    }


@pytest.mark.parametrize("fixed", [False, True])
def test_dense_admission_and_field_targets_preserve_true_threshold_absence(
    tmp_path, fixed
):
    setup = dense_setup(tmp_path, fixed=fixed)
    assert "min_instance_pixels" not in setup.report
    assert "min_instance_pixels" not in setup.request
    assert validate_training_report(setup.report, training_api_limits()) == setup.report
    assert setup.report["training_counts"]["target_object_counts"]["Leydig cell"] == 1
    assert setup.report["training_counts"]["target_object_counts"]["Other"] == 1
    options = setup.report["trainer_configuration"]["objective"]["parameters"]
    assert "head_type" not in options and "flow_loss_weighting" not in options
    assert options["use_uncertainty_weighting"] and options["weight_tversky_by_class"]
    assert all(
        "max_validation_assignment_matrix_entries" not in r
        for r in setup.entry.trainer_profile["gpu_resources"].values()
    )
    configuration = setup.report["trainer_configuration"]
    fields = DenseEpochDataset(
        setup.dataset,
        setup.request["split_plan"],
        capability=setup.entry.capability,
        field_profile=setup.request["field_profile"],
        sampler=configuration["sampler"],
        augmentation=configuration["augmentation"],
        training_seed=17,
        epoch_index=0,
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
    )
    field = fields[0]
    batch = collate_dense_fields(
        [field],
        preprocess=normalize99,
        field_profile=setup.request["field_profile"],
        class_count=4,
    )
    assert not hasattr(batch, "directional_targets")
    counts = np.bincount(
        field.class_targets[field.supervision_mask & field.real_pixel_mask], minlength=4
    )
    assert counts.tolist() == [
        c["raw_count"] for c in options["class_weighting"]["channels"]
    ]
    assert counts[2] == 36  # Training-only class remains in targets and TRAIN weights.
    for value in (0, 15, None):
        invalid = {**setup.request, "min_instance_pixels": value}
        with pytest.raises(ValueError):
            validate_training_run_request_against_manifest_capability(
                invalid,
                setup.dataset.manifest,
                setup.entry.capability,
                training_api_limits(),
            )
        with pytest.raises(ValueError):
            validate_training_report(
                {**setup.report, "min_instance_pixels": value}, training_api_limits()
            )
    with pytest.raises(ValueError):
        validate_training_run_request(
            {**setup.request, "min_instance_pixels": None}, training_api_limits()
        )


@pytest.mark.parametrize(
    "turns,horizontal,vertical", [(1, True, False), (3, False, True), (2, True, True)]
)
def test_dense_spatial_augmentation_and_mixed_shape_padding(
    turns, horizontal, vertical
):
    from histoforge.cloud.training.trainers.dense.fields import augment_dense_field
    from tests.test_training_augmentation import _component
    from tests.test_training_batches import _profile

    labels = np.arange(8 * 20, dtype=np.int32).reshape(8, 20) % 4
    real = np.ones(labels.shape, bool)
    real[:2, :3] = False
    supervision = np.ones(labels.shape, bool)
    supervision[5:, 10:] = False
    field = DenseTrainingField(
        np.repeat(labels[:, :, None], 3, axis=2).astype(np.uint8),
        labels,
        supervision,
        real,
    )
    component = _component()
    operators = component["parameters"]["operator_parameters"]
    operators["rotate-v1"]["quarter_turn_probabilities"] = [
        float(k == turns) for k in range(4)
    ]
    operators["flip-v1"].update(
        horizontal_probability=float(horizontal), vertical_probability=float(vertical)
    )
    augmented = augment_dense_field(
        field, component, training_seed=7, epoch_index=0, sample_index=0
    )
    for name in ("pixels", "class_targets", "supervision_mask", "real_pixel_mask"):
        expected = np.rot90(getattr(field, name), turns)
        if horizontal:
            expected = expected[:, ::-1]
        if vertical:
            expected = expected[::-1]
        np.testing.assert_array_equal(getattr(augmented, name), expected)
    batch = collate_dense_fields(
        [augmented, field],
        preprocess=normalize99,
        field_profile=_profile(),
        class_count=4,
    )
    batch.validate(4)
    original_counts = np.bincount(labels[real & supervision], minlength=4)
    for index, source_field in enumerate((augmented, field)):
        support = batch.supervision_mask[index] & batch.real_pixel_mask[index]
        np.testing.assert_array_equal(
            torch.bincount(batch.class_targets[index][support], minlength=4),
            original_counts,
        )
        # Center padding and larger-peer padding add no supervision or real pixels.
        assert batch.real_pixel_mask[index].sum() == source_field.real_pixel_mask.sum()
        assert (
            batch.supervision_mask[index].sum() == source_field.supervision_mask.sum()
        )


def test_validation_masks_context_and_preflights_every_roi_before_inference(tmp_path):
    setup = dense_setup(tmp_path)
    calls = []

    def predict(image):
        calls.append(image.pixels.shape)
        labels = np.zeros(image.pixels.shape[:2], np.int32)
        # Expanded context starts at source (0,0); only (10,10)-(30,30) is scored.
        labels[:5, :5] = 3
        labels[12:20, 12:20] = 1
        labels[22:28, 22:28] = 2
        return {"class_map": labels}

    kwargs = {
        **_validation_kwargs(setup),
        "field_profile": {
            **setup.request["field_profile"],
            "minSideModelPx": 64,
            "preferredTileSideModelPx": 64,
            "smallRoiContext": True,
        },
    }
    values = evaluate_dense_validation(
        setup.dataset, setup.request["split_plan"], predict, **kwargs
    )
    assert len(calls) == 2
    assert values == {
        pixel_class_f1_series_id(1): 1.0,
        VALIDATION_PIXEL_MACRO_F1_SERIES_ID: 1.0,
    }
    calls.clear()
    with pytest.raises(ValueError, match="exceeds bounded inference limits"):
        evaluate_dense_validation(
            setup.dataset,
            setup.request["split_plan"],
            predict,
            **{
                **kwargs,
                "target_raster_limits": {
                    "max_width": 1,
                    "max_height": 1,
                    "max_pixels": 1,
                },
            },
        )
    assert calls == []


def test_pooled_full_head_confusion_handles_all_roles_and_undefined_classes():
    channels = dense_output()["classChannels"]
    confusion = pixel_confusion(
        np.array([[1, 1, 0, 2]]), np.array([[1, 2, 3, 3]]), np.ones((1, 4), bool), 4
    )
    # Class 1: TP1 FN1. Class3: no GT but FP2, hence zero, not undefined.
    values = dense_metric_values(confusion, channels)
    assert values[pixel_class_f1_series_id(1)] == pytest.approx(2 / 3)
    assert values[pixel_class_f1_series_id(3)] == 0
    assert values[VALIDATION_PIXEL_MACRO_F1_SERIES_ID] == pytest.approx(1 / 3)
    assert dense_metric_values(np.diag([20, 0, 5, 0]), channels) == {}
    second = np.zeros((4, 4), np.int64)
    second[1, 1] = 99
    pooled = dense_metric_values(confusion + second, channels)
    assert pooled[pixel_class_f1_series_id(1)] == pytest.approx(200 / 201)


@pytest.mark.parametrize("predicted_channel", [0, 1, 2])
def test_negative_validation_rois_keep_defined_zero_without_coverage_guard(
    tmp_path, predicted_channel
):
    setup = dense_setup(tmp_path, objects=[])
    values = evaluate_dense_validation(
        setup.dataset,
        setup.request["split_plan"],
        lambda image: {
            "class_map": np.full(image.pixels.shape[:2], predicted_channel, np.int32)
        },
        field_profile=setup.request["field_profile"],
        **_validation_kwargs(setup),
    )
    assert values == (
        {pixel_class_f1_series_id(1): 0.0, VALIDATION_PIXEL_MACRO_F1_SERIES_ID: 0.0}
        if predicted_channel == 1
        else {}
    )


def test_no_validation_trains_without_inference_or_invented_metrics(
    tmp_path, single_thread_torch
):
    setup = dense_setup(tmp_path, validation_rois=0)
    epochs = []
    train_managed_dense_model(
        TinySemantic(),
        setup.report,
        setup.dataset,
        lambda _: pytest.fail("no-validation run invoked inference"),
        preprocess=normalize99,
        device="cpu",
        epoch_completed=epochs.append,
        **_training_kwargs(setup),
    )
    assert len(epochs) == 2
    assert all(
        [m["series_id"] for m in e["metrics"]] == ["training-loss.v1"] for e in epochs
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_dense_train_validate_and_strict_fresh_reload(
    tmp_path, fixed, single_thread_torch
):
    setup = dense_setup(tmp_path, fixed=fixed)
    torch.manual_seed(18)
    model = TinySemantic()
    original = copy.deepcopy(model.state_dict())
    runtime = create_semantic_runtime(setup.recipe["models"]["nuclei"], model)
    calls, epochs = [], []

    def predict(image):
        assert not model.training and torch.is_inference_mode_enabled()
        calls.append(image.pixels.shape)
        return runtime.predict(image)

    train_managed_dense_model(
        model,
        setup.report,
        setup.dataset,
        predict,
        preprocess=normalize99,
        device="cpu",
        epoch_completed=epochs.append,
        **_training_kwargs(setup),
    )
    assert len(calls) == 4 and [e["epoch"] for e in epochs] == [1, 2]
    for name in ("out.weight", "out.bias", "W2"):
        assert not dict(model.named_parameters())[name].requires_grad
        torch.testing.assert_close(
            model.state_dict()[name], original[name], rtol=0, atol=0
        )
    assert not torch.equal(model.out_class.weight, original["out_class.weight"])
    validate_training_report(
        {**setup.report, "state": "RUNNING", "epochs": epochs}, training_api_limits()
    )
    fresh = TinySemantic()
    load_semantic_model_state(fresh, model.state_dict())
    from histoforge.cloud.rasterization.rasterizer import RasterizedImage

    pixels = np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3)
    image = RasterizedImage(pixels, np.ones((20, 20), bool))
    actual = create_semantic_runtime(setup.recipe["models"]["nuclei"], fresh).predict(
        image
    )
    expected = runtime.predict(image)
    assert set(actual) == set(expected) == {"class_map"}
    np.testing.assert_array_equal(actual["class_map"], expected["class_map"])


def test_partial_microbatches_keep_field_equal_model_and_objective_gradients(
    single_thread_torch,
):
    setup_fields = [
        DenseTrainingField(
            np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3),
            np.full((h, w), index % 4, np.int32),
            np.ones((h, w), bool),
            np.ones((h, w), bool),
        )
        for index, (h, w) in enumerate(((16, 32), (32, 16), (16, 16)))
    ]
    from tests.test_training_batches import _profile

    models, objectives = [], []
    for batch_size, accumulation in ((1, 3), (2, 2)):
        torch.manual_seed(33)
        model = TinySemantic()
        prepare_semantic_finetuning(model)
        objective = ManagedObjective(
            class_count=4,
            classification_objective="cross_entropy+tversky",
            use_uncertainty_weighting=True,
            include_segmentation=False,
        )
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad]
            + list(objective.parameters()),
            lr=0.01,
        )
        batches = [
            collate_dense_fields(
                setup_fields[i : i + batch_size],
                preprocess=normalize99,
                field_profile=_profile(),
                class_count=4,
            )
            for i in range(0, 3, batch_size)
        ]
        train_dense_fields(
            model,
            optimizer,
            lambda _: batches,
            training_forward=partial(semantic_training_logits, class_count=4),
            class_count=4,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=accumulation,
            objective=objective,
        )
        models.append(model)
        objectives.append(objective)
    for name, value in models[0].state_dict().items():
        torch.testing.assert_close(
            value, models[1].state_dict()[name], rtol=1e-5, atol=1e-7
        )
    torch.testing.assert_close(objectives[0].log_variances, objectives[1].log_variances)


@pytest.mark.parametrize("failure", ["validation", "cancellation", "publication"])
def test_validation_cancellation_and_publication_failures_stop_next_epoch(
    tmp_path, failure, single_thread_torch
):
    setup = dense_setup(tmp_path)
    cancelled = False
    published = []
    predictions = []
    model = TinySemantic()

    def predict(image):
        nonlocal cancelled
        predictions.append(image.pixels.shape)
        if failure == "validation" and len(predictions) == 2:
            raise RuntimeError("injected validation failure")
        if failure == "cancellation" and len(predictions) == 2:
            cancelled = True
        return {"class_map": np.zeros(image.pixels.shape[:2], np.int32)}

    def publish(epoch):
        published.append(epoch)
        raise RuntimeError("injected publication failure")

    with pytest.raises(
        TrainingCancelled if failure == "cancellation" else RuntimeError
    ):
        train_managed_dense_model(
            model,
            setup.report,
            setup.dataset,
            predict,
            preprocess=normalize99,
            device="cpu",
            epoch_completed=publish,
            cancellation_requested=lambda: cancelled,
            **_training_kwargs(setup),
        )
    assert len(predictions) == 2
    assert len(published) == (1 if failure == "publication" else 0)
    assert model.training


def test_dense_data_workers_preserve_all_augmented_fields(tmp_path):
    from dataclasses import fields

    from tests.test_training_augmentation import _appearance_configured, _component
    from torch.utils.data import DataLoader

    setup = dense_setup(tmp_path)
    capability, profile = _appearance_configured()
    augmentation = _component(
        [
            "rotate-v1",
            "flip-v1",
            "hed-v1",
            "brightness-contrast-v1",
            "blur-v1",
            "noise-v1",
        ],
        capability=capability,
        profile=profile,
    )
    epoch = DenseEpochDataset(
        setup.dataset,
        {"type": "none"},
        capability=setup.entry.capability,
        field_profile=setup.report["field_profile"],
        sampler=setup.report["trainer_configuration"]["sampler"],
        augmentation=augmentation,
        training_seed=1937,
        epoch_index=0,
        api_limits=training_api_limits(),
        source_limits=_source_limits(),
    )
    collate = partial(
        collate_dense_fields,
        preprocess=normalize99,
        field_profile=setup.report["field_profile"],
        class_count=4,
    )
    results = [
        list(
            DataLoader(
                epoch,
                batch_size=2,
                collate_fn=collate,
                shuffle=False,
                num_workers=workers,
                multiprocessing_context="spawn" if workers else None,
            )
        )
        for workers in (0, 2)
    ]
    assert [batch.images.shape[0] for batch in results[0]] == [2, 1]
    for first, second in zip(*results, strict=True):
        for field in fields(first):
            assert torch.equal(getattr(first, field.name), getattr(second, field.name))
