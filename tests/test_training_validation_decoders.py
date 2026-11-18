"""Exercise both ordinary instance decoders on retained validation ROI inputs."""

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.flow_inference import (
    run_flow_inference,
)
from histoforge._vendor.histoforge_models.pathology.hovernet_inference import (
    run_hovernet_inference,
)
from histoforge._vendor.histoforge_models.pathology.managed_training.targets import (
    build_managed_targets,
)
from histoforge.cloud.training.trainers.instance.metrics import InstanceMetricResult
from tests.test_training_dataset_content import _rectangle
from tests.test_training_validation import _capture, _evaluate, _object
from tests.test_variable_field_inference import _profile


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("classified", [False, True])
@pytest.mark.parametrize("positive", [False, True])
@pytest.mark.parametrize("variable", [False, True])
def test_normal_tiler_and_decoder_feed_complete_roi_validation(
    tmp_path, head, classified, positive, variable
):
    """CPU fixture logits exercise the real tiler, mask decoder, and class vote.

    This is integration evidence, not qualification of trained model weights.
    Fixed inference uses a 128-pixel field around the 64-pixel retained raster.
    Variable inference expands the ROI to available context (52 pixels), embeds
    it in the eight-pixel virtual border, and evaluates four 64-pixel fields. The
    predicted object crosses the ROI edge; only its inside pixels enter the metric.
    Field placement and blending must preserve those exact coordinates.
    """
    objects = [_object(1, _rectangle(24, 24, 40, 40))] if positive else []
    dataset, split, capability = _capture(
        tmp_path,
        objects=objects,
        classified=classified,
        profile=_profile(
            minSideModelPx=64, preferredTileSideModelPx=64, smallRoiContext=True
        )
        if variable
        else None,
    )
    num_classes = 3 if classified else 0
    object_starts = (
        [
            (32 - field_x, 32 - field_y)
            for field_x, field_y in ((0, 0), (4, 0), (0, 4), (4, 4))
        ]
        if variable
        else [(56, 56)]
    )
    logits = [
        _field_logits(
            64 if variable else 128,
            head,
            num_classes,
            positive,
            object_x=object_x,
            object_y=object_y,
        )
        for object_x, object_y in object_starts
    ]
    net = _FixtureNet(np.stack(logits))
    function = run_flow_inference if head == "flow" else run_hovernet_inference
    calls = []

    def predict(image):
        fields = (
            {"field_profile": capability["default_field_profile"]}
            if variable
            else {"bsize": 128, "tile_overlap": 0.25}
        )
        output = function(
            net,
            image.pixels,
            num_classes=num_classes,
            batch_size=1,
            real_pixel_mask=image.real_pixel_mask,
            **fields,
        )
        calls.append(output["instance_mask"])
        return output

    result = _evaluate(dataset, split, capability, predict)
    assert len(calls) == 1
    assert calls[0].shape == ((52, 52) if variable else (64, 64))
    assert bool(calls[0].any()) == positive
    expected = (
        InstanceMetricResult(
            1, 1, 1 if classified else None, {1: 1} if classified else {}
        )
        if positive
        else InstanceMetricResult(None, None, None, {1: None} if classified else {})
    )
    assert result == expected
    assert net.forward_count == (4 if variable else 1)


def _field_logits(side, head, num_classes, positive, *, object_x, object_y):
    logits = np.zeros(
        (num_classes + (3 if head == "flow" else 4), side, side), np.float32
    )
    foreground = np.zeros((side, side), np.int32)
    if positive:
        foreground[object_y : object_y + 16, object_x : object_x + 16] = 1
    targets = build_managed_targets(foreground, head_type=head)
    logits[-1] = np.where(foreground, 10, -10)
    if head == "flow":
        logits[-3:-1] = targets.directional * 5
    if head == "hovernet":
        logits[-4:-2] = targets.directional
        logits[-2] = -logits[-1]
    if num_classes:
        logits[1] = 10  # Emitted class; channel 0 is background, channel 2 is Other.
    return logits


class _FixtureNet(torch.nn.Module):
    """Return one known field's raw logits through the normal Torch forward path."""

    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.register_buffer("logits", torch.from_numpy(logits))
        self.forward_count = 0
        self.field_index = 0

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, batch):
        assert not self.training
        assert batch.shape[1:] == (3, *self.logits.shape[2:])
        assert torch.isfinite(batch).all()
        stop = self.field_index + len(batch)
        assert stop <= len(self.logits)
        output = self.logits[self.field_index : stop]
        self.field_index = stop
        self.forward_count += 1
        return output, batch.new_zeros(len(batch), 256)
