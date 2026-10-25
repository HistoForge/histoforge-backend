"""Independent arithmetic/gradient oracles for managed classification weighting."""

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.managed_training import (
    ManagedObjective,
    ManagedSegmentationBatch,
    managed_segmentation_loss,
)


def _batch():
    classes = torch.tensor(
        [
            [[0, 1, 1, 0], [2, 2, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 2], [1, 0, 2, 2], [0, 0, 0, 0]],
        ]
    )
    foreground = (classes != 0).float()
    supervision = torch.ones_like(classes, dtype=torch.bool)
    supervision[0, 2] = False
    supervision[1, 1:] = False
    real = torch.ones_like(supervision)
    real[2, :, 3] = False
    areas = torch.where(classes == 1, 1 / 16, 1 / 64) * foreground
    classes[~(supervision & real)] = 255
    return ManagedSegmentationBatch(
        images=torch.zeros(3, 3, 3, 4),
        directional_targets=torch.linspace(-0.2, 0.3, 72).reshape(3, 2, 3, 4),
        foreground_targets=foreground,
        supervision_mask=supervision,
        real_pixel_mask=real,
        class_targets=classes,
        instance_flow_weights=areas,
    )


def _classification_oracle(logits, target, weights, weight_tversky):
    """Independent CE and focal Tversky arithmetic over selected valid pixels."""

    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    probabilities = log_probabilities.exp()
    pixel_weights = weights[target]
    ce = (
        -(pixel_weights * log_probabilities.gather(1, target[:, None])[:, 0]).sum()
        / pixel_weights.sum()
    )
    tv = []
    for channel in range(3):
        truth = target == channel
        prob = probabilities[:, channel]
        tp = prob[truth].sum()
        fp = prob[~truth].sum()
        fn = (1 - prob[truth]).sum()
        score = tp / (tp + 0.3 * fp + 0.7 * fn).clamp_min(
            torch.finfo(torch.float32).eps
        )
        focal = (1 - score).clamp(1e-6, 1 - 1e-6).pow(1 / 1.33)
        tv.append(focal * weights[channel] if weight_tversky else focal)
    return ce, torch.stack(tv).sum() / 3  # Not divided by the sum of class weights.


def _oracle(predictions, batch, weights, flow_mode, weight_tversky):
    """Index supported pixels explicitly; do not call the managed loss helpers."""

    fields = []
    for i, support in enumerate(batch.effective_support):
        target = batch.class_targets[i][support]
        logits = predictions[i, :3, support].T
        ce, tv = _classification_oracle(logits, target, weights, weight_tversky)
        directions = predictions[i, 3:5, support]
        errors = (directions - 5 * batch.directional_targets[i, :, support]).square()
        foreground = batch.foreground_targets[i][support]
        flow_weights = torch.ones_like(foreground)
        if flow_mode == "equal_instance" and foreground.any():
            fg = foreground.bool()
            inverse_areas = batch.instance_flow_weights[i][support][fg]
            flow_weights[fg] = inverse_areas * fg.sum() / inverse_areas.sum()
        flow = (errors * flow_weights).sum() / (4 * len(target))
        foreground_logits = predictions[i, 5, support]
        bce = (
            torch.logaddexp(torch.zeros_like(foreground_logits), foreground_logits)
            - foreground * foreground_logits
        ).mean()
        fields.append(torch.stack([flow + bce, ce, tv]))
    return torch.stack(fields).sum(dim=0)


@pytest.mark.parametrize("flow_mode", ["pixel", "equal_instance"])
def test_both_weighted_losses_and_logits_and_learned_balance_gradients(flow_mode):
    batch = _batch()
    weights = torch.tensor([0.25, 1.0, 2.0])  # TRAIN counts 64, 4, 0, floored at 1.
    logits = torch.linspace(-1.1, 0.9, 216).reshape(3, 6, 3, 4).requires_grad_()
    expected_logits = logits.detach().clone().requires_grad_()
    objective = ManagedObjective(
        class_count=3,
        classification_objective="cross_entropy+tversky",
        use_uncertainty_weighting=True,
    )
    with torch.no_grad():
        objective.log_variances.copy_(torch.tensor([0.2, -0.3, 0.4]))
    expected_log_variances = objective.log_variances.detach().clone().requires_grad_()
    result = managed_segmentation_loss(
        logits,
        batch,
        head_type="flow",
        class_count=3,
        classification_objective="cross_entropy+tversky",
        class_weights=weights,
        weight_tversky_by_class=True,
        objective=objective,
        flow_loss_weighting=flow_mode,
    )
    task_sums = _oracle(expected_logits, batch, weights, flow_mode, True)
    expected = (
        torch.exp(-expected_log_variances) * task_sums + 3 * expected_log_variances
    ).sum() / 3
    actual = result.total_sum / result.field_count
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.stack(list(result.task_sums.values())), task_sums)
    actual_gradients = torch.autograd.grad(actual, (logits, objective.log_variances))
    expected_gradients = torch.autograd.grad(
        expected, (expected_logits, expected_log_variances)
    )
    for observed, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-7)
    assert (
        torch.count_nonzero(
            actual_gradients[0].permute(0, 2, 3, 1)[~batch.effective_support]
        )
        == 0
    )

    # Both-weighted is a real change in Tversky, not a second application to CE.
    ce_only = _oracle(expected_logits, batch, weights, flow_mode, False)
    torch.testing.assert_close(task_sums[:2], ce_only[:2])
    assert not torch.isclose(task_sums[2], ce_only[2])
    alternate_flow = _oracle(
        expected_logits,
        batch,
        weights,
        "pixel" if flow_mode == "equal_instance" else "equal_instance",
        True,
    )
    torch.testing.assert_close(task_sums[1:], alternate_flow[1:])
    assert not torch.isclose(task_sums[0], alternate_flow[0])


def test_hovernet_retains_weighted_ce_and_unweighted_tversky_values_and_gradients():
    batch = _batch()
    weights = torch.tensor([0.25, 1.0, 2.0])
    logits = torch.linspace(-1.1, 0.9, 252).reshape(3, 7, 3, 4).requires_grad_()
    expected_logits = logits.detach().clone().requires_grad_()
    result = managed_segmentation_loss(
        logits,
        batch,
        head_type="hovernet",
        class_count=3,
        classification_objective="cross_entropy+tversky",
        class_weights=weights,
        weight_tversky_by_class=False,
    )
    expected = torch.stack(
        [
            torch.stack(
                _classification_oracle(
                    expected_logits[index, :3, support].T,
                    batch.class_targets[index][support],
                    weights,
                    False,
                )
            )
            for index, support in enumerate(batch.effective_support)
        ]
    ).sum(dim=0)
    actual = torch.stack(
        [
            result.task_sums["classification_cross_entropy"],
            result.task_sums["classification_tversky"],
        ]
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), logits)[0],
        torch.autograd.grad(expected.sum(), expected_logits)[0],
        rtol=1e-5,
        atol=1e-7,
    )
    no_weights = managed_segmentation_loss(
        logits,
        batch,
        head_type="hovernet",
        class_count=3,
        classification_objective="cross_entropy+tversky",
        weight_tversky_by_class=False,
    )
    torch.testing.assert_close(
        result.task_sums["classification_tversky"],
        no_weights.task_sums["classification_tversky"],
    )
    assert not torch.isclose(
        result.task_sums["classification_cross_entropy"],
        no_weights.task_sums["classification_cross_entropy"],
    )
