"""Selected dataset fields reach the trainer without changed pixels or weights."""

import copy
from dataclasses import replace
from functools import partial

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedSegmentationTargets,
)
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.architectures.registry import FIELD_PREPROCESSING_ID
from histoforge.cloud.training.trainers.instance.augmentation import (
    apply_training_augmentation,
)
from histoforge.cloud.training.trainers.instance.batches import collate_training_fields
from histoforge.cloud.training.trainers.instance.field_reader import TrainingField
from histoforge.cloud.training.trainers.instance.field_targets import (
    TrainingFieldTargets,
)
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields
from tests.test_training_contract import (
    classified_training_model_output,
    training_capability,
    training_model_output,
)
from tests.test_training_dataset_content import _dataset, _object_feature, _rectangle
from tests.test_training_field_reader import (
    _origin,
    _pattern,
    _read,
    _retain,
    _target,
    _write_png,
)
from torch import nn
from torch.utils.data import DataLoader


def _profile():
    """Small shapes exercise the contract, not a qualified production range."""
    return {
        "planner": "variable-field-v2",
        "minSideModelPx": 16,
        "maxSideModelPx": 64,
        "preferredTileSideModelPx": 32,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": 8,
        "smallRoiContext": False,
        "tileOverlap": 0.25,
        "aggregation": "test-only-aggregation",
        "preprocessing": FIELD_PREPROCESSING_ID,
    }


def _field(height=16, width=32, *, classified=True):
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((3 * x + y, x + 5 * y, (x * y) % 239), axis=-1).astype(np.float32)
    foreground = ((x > 4) & (y > 4) & (x < width - 5)).astype(np.float32)
    targets = ManagedSegmentationTargets(
        directional=np.stack((y / height, x / width)).astype(np.float32),
        foreground=foreground,
        classes=(foreground * (1 + x % 2)).astype(np.int64) if classified else None,
    )
    real = np.ones((height, width), dtype=bool)
    real[-2:] = False
    pixels[~real] = np.nan  # Artificial pixels cannot enter normalization.
    supervision = (x > 2) & (y > 2)
    return TrainingField(pixels, TrainingFieldTargets(targets, supervision, real))


def _normalized_oracle(field):
    pixels = field.pixels.astype(np.float32)
    support = field.labels.real_pixel_mask
    expected = np.zeros_like(pixels)
    for channel in range(3):
        values = pixels[..., channel][support]
        lower, upper = np.percentile(values, (1.0, 99.0))
        if upper - lower > 0.001:
            expected[..., channel][support] = (values - lower) / (upper - lower)
    return expected.transpose(2, 0, 1)


@pytest.mark.parametrize("classified", [False, True])
def test_mixed_shapes_preserve_each_field_and_pad_only_bottom_and_right(classified):
    fields = [
        _field(16, 48, classified=classified),
        _field(32, 16, classified=classified),
    ]
    original = copy.deepcopy(fields)
    batch = collate_training_fields(
        fields,
        preprocess=normalize99,
        field_profile=_profile(),
        class_count=3 if classified else 0,
    )

    assert batch.images.shape == (2, 3, 32, 48)
    assert batch.images.dtype == torch.float32
    assert batch.images.device.type == "cpu"
    assert batch.supervision_mask.dtype == batch.real_pixel_mask.dtype == torch.bool
    batch.validate(3 if classified else 0)
    for index, field in enumerate(fields):
        height, width = field.pixels.shape[:2]
        np.testing.assert_array_equal(
            batch.images[index, :, :height, :width], _normalized_oracle(field)
        )
        expected = {
            "directional_targets": field.labels.targets.directional,
            "foreground_targets": field.labels.targets.foreground,
            "supervision_mask": field.labels.supervision_mask,
            "real_pixel_mask": field.labels.real_pixel_mask,
        }
        if classified:
            expected["class_targets"] = field.labels.targets.classes
        for name, value in expected.items():
            np.testing.assert_array_equal(
                getattr(batch, name)[index, ..., :height, :width], value
            )
        for name in ("images", *expected):
            padded = getattr(batch, name)[index]
            assert not padded[..., height:, :].any()
            assert not padded[..., :, width:].any()
        # Context remains real and influences preprocessing without becoming labels.
        context = field.labels.real_pixel_mask & ~field.labels.supervision_mask
        assert context.any()
        assert batch.images[index, :, :height, :width][:, context].abs().sum() > 0
    assert (batch.class_targets is None) == (not classified)

    # Returned tensors own their memory; neither collation nor later tensor writes
    # can rewrite source arrays shared by another epoch or a DataLoader worker.
    for name in ("images", "directional_targets", "foreground_targets"):
        getattr(batch, name).zero_()
    batch.supervision_mask.zero_()
    batch.real_pixel_mask.zero_()
    if batch.class_targets is not None:
        batch.class_targets.zero_()
    for field, before in zip(fields, original):
        np.testing.assert_array_equal(field.pixels, before.pixels)
        for name in ("directional", "foreground", "classes"):
            np.testing.assert_array_equal(
                getattr(field.labels.targets, name),
                getattr(before.labels.targets, name),
            )
        np.testing.assert_array_equal(
            field.labels.supervision_mask, before.labels.supervision_mask
        )
        np.testing.assert_array_equal(
            field.labels.real_pixel_mask, before.labels.real_pixel_mask
        )


def test_one_field_adds_no_padding_and_reflected_arrays_remain_exact():
    field = _field(32, 16)
    targets = field.labels.targets
    field = TrainingField(
        field.pixels[:, ::-1],
        TrainingFieldTargets(
            ManagedSegmentationTargets(
                targets.directional[..., ::-1],
                targets.foreground[:, ::-1],
                targets.classes[:, ::-1],
            ),
            field.labels.supervision_mask[:, ::-1],
            field.labels.real_pixel_mask[:, ::-1],
        ),
    )
    batch = collate_training_fields(
        [field], preprocess=normalize99, field_profile=_profile(), class_count=3
    )
    assert batch.images.shape == (1, 3, 32, 16)
    np.testing.assert_array_equal(batch.images[0], _normalized_oracle(field))
    np.testing.assert_array_equal(
        batch.directional_targets[0], field.labels.targets.directional
    )


def test_optional_complete_instance_flow_weights_follow_batch_padding():
    field = _field(17, 23, classified=False)
    weights = np.where(
        field.labels.targets.foreground != 0,
        1.0 / field.labels.targets.foreground.sum(),
        0.0,
    ).astype(np.float32)
    field = replace(
        field,
        labels=replace(
            field.labels,
            targets=replace(
                field.labels.targets,
                instance_flow_weights=weights,
            ),
        ),
    )
    batch = collate_training_fields(
        [field], preprocess=normalize99, field_profile=_profile(), class_count=0
    )
    assert batch.instance_flow_weights is not None
    assert batch.instance_flow_weights.shape == (1, 32, 32)
    np.testing.assert_array_equal(batch.instance_flow_weights[0, 7:24, 4:27], weights)
    assert not batch.instance_flow_weights[0, :7].any()
    assert not batch.instance_flow_weights[0, 24:].any()
    assert not batch.instance_flow_weights[0, :, :4].any()
    assert not batch.instance_flow_weights[0, :, 27:].any()
    batch.validate(0, flow_loss_weighting="equal_instance")


def test_fixed_profile_accepts_only_its_actual_square_model_fields():
    profile = training_capability()["default_field_profile"]
    profile.update(side_model_px=32, preprocessing=FIELD_PREPROCESSING_ID)
    batch = collate_training_fields(
        [_field(32, 32)], preprocess=normalize99, field_profile=profile, class_count=3
    )
    assert batch.images.shape == (1, 3, 32, 32)
    with pytest.raises(ValueError, match="dimensions violate"):
        collate_training_fields(
            [_field(16, 32)],
            preprocess=normalize99,
            field_profile=profile,
            class_count=3,
        )


@pytest.mark.parametrize("shape", [(0, 16), (16, 0), (65, 16), (16, 80)])
def test_field_is_rejected_instead_of_resized_or_clamped(shape):
    with pytest.raises(ValueError, match="dimensions violate"):
        collate_training_fields(
            [_field(*shape)],
            preprocess=normalize99,
            field_profile=_profile(),
            class_count=3,
        )


@pytest.mark.parametrize("classified", [False, True])
def test_natural_fields_receive_centered_padding_before_cross_batch_padding(classified):
    """Small/unaligned fields preserve labels; only their tensor offsets change."""
    fields = [
        _field(9, 22, classified=classified),
        _field(33, 17, classified=classified),
    ]
    profile = {**_profile(), "minSideModelPx": 32}
    batch = collate_training_fields(
        fields,
        preprocess=normalize99,
        field_profile=profile,
        class_count=3 if classified else 0,
    )
    assert batch.images.shape == (2, 3, 48, 32)
    for index, (field, top, left) in enumerate(zip(fields, (11, 7), (5, 7))):
        height, width = field.pixels.shape[:2]
        expected = {
            "images": _normalized_oracle(field),
            "directional_targets": field.labels.targets.directional,
            "foreground_targets": field.labels.targets.foreground,
            "supervision_mask": field.labels.supervision_mask,
            "real_pixel_mask": field.labels.real_pixel_mask,
        }
        if classified:
            expected["class_targets"] = field.labels.targets.classes
        outside = np.ones((48, 32), dtype=bool)
        outside[top : top + height, left : left + width] = False
        for name, value in expected.items():
            actual = getattr(batch, name)[index]
            np.testing.assert_array_equal(
                actual[..., top : top + height, left : left + width], value
            )
            assert not actual.numpy()[..., outside].any()
        alone = collate_training_fields(
            [field],
            preprocess=normalize99,
            field_profile=profile,
            class_count=3 if classified else 0,
        )
        # A taller peer changes only final batch padding, never this field's offset.
        np.testing.assert_array_equal(
            batch.images[index, :, : alone.images.shape[-2], : alone.images.shape[-1]],
            alone.images[0],
        )


@pytest.mark.parametrize("head", ["flow", "hovernet"])
def test_centering_does_not_normalize_over_or_supervise_added_support(head):
    """Padding cannot change the per-field loss of a pointwise model."""
    from histoforge._vendor.histoforge_models.pathology.managed_training import (
        managed_segmentation_loss,
    )

    field = _field(31, 21)
    ordinary = collate_training_fields(
        [field], preprocess=normalize99, field_profile=_profile(), class_count=3
    )
    larger = collate_training_fields(
        [field],
        preprocess=normalize99,
        field_profile={
            **_profile(),
            "minSideModelPx": 48,
            "preferredTileSideModelPx": 48,
        },
        class_count=3,
    )
    model = nn.Conv2d(3, 6 if head == "flow" else 7, 1)
    for batch in (ordinary, larger):
        batch.validate(3)
    reference = managed_segmentation_loss(
        model(ordinary.images),
        ordinary,
        head_type=head,
        class_count=3,
        classification_objective="cross_entropy+tversky",
    )
    actual = managed_segmentation_loss(
        model(larger.images),
        larger,
        head_type=head,
        class_count=3,
        classification_objective="cross_entropy+tversky",
    )
    torch.testing.assert_close(actual.total_sum, reference.total_sum)
    for name in reference.component_sums:
        torch.testing.assert_close(
            actual.component_sums[name], reference.component_sums[name]
        )
    expected_gradients = torch.autograd.grad(
        reference.total_sum, tuple(model.parameters())
    )
    actual_gradients = torch.autograd.grad(actual.total_sum, tuple(model.parameters()))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="at least one field"):
        collate_training_fields(
            [], preprocess=normalize99, field_profile=_profile(), class_count=0
        )


@pytest.mark.parametrize(
    "defect",
    ["support", "mask_type", "target_shape", "class_type", "class_count", "real_nan"],
)
def test_invalid_scientific_samples_fail_before_a_forward(defect):
    field = _field()
    count = 3
    if defect == "support":
        field.labels.supervision_mask[:] = False
    elif defect == "mask_type":
        field = replace(
            field,
            labels=replace(
                field.labels,
                supervision_mask=field.labels.supervision_mask.astype(np.uint8),
            ),
        )
    elif defect == "target_shape":
        field = replace(
            field,
            labels=replace(
                field.labels,
                targets=replace(
                    field.labels.targets,
                    foreground=field.labels.targets.foreground[:, :1],
                ),
            ),
        )
    elif defect == "class_type":
        field = replace(
            field,
            labels=replace(
                field.labels,
                targets=replace(
                    field.labels.targets,
                    classes=field.labels.targets.classes.astype(np.float32),
                ),
            ),
        )
    elif defect == "class_count":
        count = 0
    else:
        field.pixels[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        collate_training_fields(
            [field], preprocess=normalize99, field_profile=_profile(), class_count=count
        )


@pytest.fixture
def single_thread_torch():
    """Keep tiny CPU convolutions proportional without changing trainer settings."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("classified", [False, True])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("augmented", [False, True])
def test_retained_dataset_fields_train_with_equal_weight_across_uneven_batches(
    tmp_path, head, classified, negative, augmented, single_thread_torch
):
    """Exercise real files, rasterization, targets, collation, losses, and SGD.

    A pointwise network deliberately removes attention effects: this proves
    equal-field reduction, not identical DINO predictions across padded shapes.
    """
    capability = training_capability()
    if classified:
        capability["output_type"] = "instance_segmentation_and_classification"
    objects = (
        []
        if negative
        else [
            _object_feature(
                "11111111-1111-4111-8111-111111111111", _rectangle(12, 12, 20, 22)
            )
        ]
    )
    if classified and objects:
        objects[0]["properties"]["path_class"] = "Other"
    model_output = (
        classified_training_model_output() if classified else training_model_output()
    )
    extracted = _dataset(
        tmp_path / "capture",
        capability=capability,
        model_output=model_output,
        objects=objects,
    )
    _write_png(extracted.root / "artifacts/0/source.png", _pattern(64, 64))
    dataset = _retain(tmp_path, extracted, _origin(extracted), capability=capability)
    fields = [
        _read(dataset, _target(width, height, 8, 8), head=head)
        for width, height in ((22, 9), (17, 33), (48, 31))
    ]
    if augmented:
        from tests.test_training_augmentation import _component

        augmentation = _component()
        fields = [
            apply_training_augmentation(
                field,
                augmentation,
                training_seed=1937,
                epoch_index=0,
                sample_index=index,
            )
            for index, field in enumerate(fields)
        ]
    class_count = 3 if classified else 0
    channels = class_count + (3 if head == "flow" else 4)
    base = nn.Conv2d(3, channels, 1)
    with torch.no_grad():
        base.weight.fill_(0.05)
        base.bias.fill_(0.1)
    results = []
    for batch_size, accumulation in ((3, 1), (2, 2), (1, 4)):
        model = copy.deepcopy(base)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = DataLoader(
            fields,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(
                collate_training_fields,
                preprocess=normalize99,
                field_profile=_profile(),
                class_count=class_count,
            ),
        )
        epochs = train_instance_fields(
            model,
            optimizer,
            lambda _epoch: loader,
            training_forward=instance_training_predictions,
            head_type=head,
            class_count=class_count,
            learning_rate_scales=[1.0],
            gradient_accumulation_steps=accumulation,
            classification_objective="cross_entropy+tversky",
        )
        assert epochs[0].field_count == 3
        assert epochs[0].optimizer_steps == 1
        assert not torch.equal(model.weight, base.weight)
        results.append((model, epochs[0]))
    reference, epoch = results[0]
    for model, actual in results[1:]:
        torch.testing.assert_close(model.weight, reference.weight, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(model.bias, reference.bias, atol=2e-7, rtol=2e-5)
        assert actual.mean_loss == pytest.approx(epoch.mean_loss, abs=1e-6)
        assert actual.component_means == pytest.approx(epoch.component_means, abs=1e-6)
