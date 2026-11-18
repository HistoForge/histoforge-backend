"""Synthetic shared-core field-to-update qualification, not a model benchmark."""

import copy

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.field_geometry import (
    centered_field_padding,
)
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedSegmentationBatch,
    augment_blur,
    augment_brightness_contrast,
    augment_hed,
    augment_noise,
    build_managed_targets,
)
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields


@pytest.mark.parametrize(
    "head,weighting",
    [
        ("flow", "pixel"),
        ("flow", "equal_instance"),
        ("hovernet", "pixel"),
    ],
)
@pytest.mark.parametrize("class_count", [0, 3])
def test_complete_targets_appearance_padding_warmup_and_update(
    head, weighting, class_count
):
    """Trace both heads and B1 through two differently sized accumulated fields."""
    labels = np.zeros((31, 37), dtype=np.int32)
    labels[3:18, 6:21] = 12
    labels[21:30, 25:35] = 39
    classes = np.zeros_like(labels)
    classes[labels == 12], classes[labels == 39] = 1, 2
    targets = build_managed_targets(
        labels, head_type=head, class_labels=classes if class_count else None
    )
    previous = copy.deepcopy(targets)
    image = np.random.default_rng(51).integers(20, 235, (31, 37, 3), dtype=np.uint8)
    batches = []
    for sample_index, crop in enumerate(
        [(slice(6, 24), slice(5, 30)), (slice(0, 31), slice(0, 37))]
    ):
        pixels = image[crop].copy()
        real = np.ones(pixels.shape[:2], bool)
        real[:2, :3] = False
        support = real.copy()
        support[-2:] = False  # Real context is not the same as supervised support.
        for index, (operator, settings) in enumerate(
            [
                (
                    augment_hed,
                    {"sigma_range": 0.1, "mean_intensity_range": (0.15, 0.85)},
                ),
                (
                    augment_brightness_contrast,
                    {"brightness_range": (0.9, 1.1), "contrast_range": (0.9, 1.1)},
                ),
                (augment_blur, {"sigma_range": (0.1, 0.5)}),
                (augment_noise, {"sigma": 0.01}),
            ]
        ):
            pixels = operator(
                pixels,
                real_pixel_mask=real,
                rng=np.random.default_rng(sample_index * 10 + index),
                probability=1,
                **settings,
            )
        normalized = normalize99(pixels, real_pixel_mask=real)
        assert np.isfinite(normalized).all() and not normalized[~real].any()
        left, right, top, bottom = centered_field_padding(
            *real.shape, minimum_side=32, alignment=16
        )
        padding = ((top, bottom), (left, right))

        def scalar(value):
            return torch.from_numpy(np.pad(value[crop], padding)[None].copy())

        batch = ManagedSegmentationBatch(
            images=torch.from_numpy(
                np.pad(normalized.transpose(2, 0, 1), ((0, 0), *padding))[None].copy()
            ),
            directional_targets=torch.from_numpy(
                np.pad(targets.directional[(slice(None), *crop)], ((0, 0), *padding))[
                    None
                ].copy()
            ),
            foreground_targets=scalar(targets.foreground),
            class_targets=scalar(targets.classes) if class_count else None,
            supervision_mask=torch.from_numpy(np.pad(support, padding)[None].copy()),
            real_pixel_mask=torch.from_numpy(np.pad(real, padding)[None].copy()),
            instance_flow_weights=None
            if targets.instance_flow_weights is None
            else scalar(targets.instance_flow_weights),
        )
        batch.validate(class_count, flow_loss_weighting=weighting)
        assert torch.count_nonzero(batch.images[:, :, ~batch.real_pixel_mask[0]]) == 0
        batches.append(batch)

    assert batches[0].images.shape[-2:] == (32, 32)
    assert batches[1].images.shape[-2:] == (32, 48)
    for name in vars(targets):
        actual, expected = getattr(targets, name), getattr(previous, name)
        if actual is not None:
            np.testing.assert_array_equal(actual, expected)

    torch.manual_seed(87)
    model = torch.nn.Conv2d(
        3, class_count + (3 if head == "flow" else 4), kernel_size=1
    )
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    completed = []

    def verify_epoch(result, trained):
        completed.append(result.epoch)
        assert result.field_count == 2 and result.optimizer_steps == 1
        assert np.isfinite(result.mean_loss)
        for name, value in trained.state_dict().items():
            assert torch.isfinite(value).all()
            if result.epoch == 1:
                assert torch.equal(value, initial[name])
        for parameter in trained.parameters():
            assert optimizer.state[parameter]["step"] == result.epoch

    train_instance_fields(
        model,
        optimizer,
        lambda _epoch: batches,
        training_forward=instance_training_predictions,
        head_type=head,
        class_count=class_count,
        flow_loss_weighting=weighting,
        learning_rate_scales=[0, 1],
        gradient_accumulation_steps=2,
        epoch_completed=verify_epoch,
    )
    assert completed == [1, 2]
    assert any(
        not torch.equal(value, initial[name])
        for name, value in model.state_dict().items()
    )
