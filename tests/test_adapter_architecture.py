"""The vendored runner exposes one fixed DINOv3 adapter architecture."""

import inspect

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology import common, dinov3
from torch import nn


def test_spatial_prior_returns_consumed_levels_and_retains_checkpoint_stem(monkeypatch):
    spatial_prior = dinov3.SPM(3, 64, 256)
    retained_stem = {
        name: tuple(tensor.shape)
        for name, tensor in spatial_prior.state_dict().items()
        if name.startswith("stem_s1.")
    }
    assert retained_stem == {
        "stem_s1.0.weight": (32, 3, 3, 3),
        "stem_s1.1.weight": (32,),
        "stem_s1.1.bias": (32,),
    }

    def unused_stem(*_args):
        pytest.fail("the retained full-resolution stem must not execute")

    monkeypatch.setattr(spatial_prior.stem_s1, "forward", unused_stem)
    stride_2_shapes = []
    spatial_prior.stem_s2.register_forward_hook(
        lambda _module, _args, output: stride_2_shapes.append(tuple(output.shape))
    )

    outputs = spatial_prior(torch.randn(1, 3, 128, 128))

    assert stride_2_shapes == [(1, 64, 64, 64)]
    assert [tuple(output.shape) for output in outputs] == [
        (1, 256, 32, 32),
        (1, 256, 16, 16),
        (1, 256, 8, 8),
    ]
    assert any(
        isinstance(module, dinov3.LayerNorm2d) for module in spatial_prior.modules()
    )
    assert not any(
        isinstance(module, nn.BatchNorm2d) for module in spatial_prior.modules()
    )


def test_adapter_constructor_exposes_no_architecture_switches():
    parameters = inspect.signature(dinov3.DinoV3AdapterBackbone).parameters

    assert set(parameters) == {"head_type", "dtype", "num_classes"}
    assert dinov3.DinoV3AdapterBackbone.ADAPTER_DIM == 256
    assert dinov3.DinoV3AdapterBackbone.READOUT_STRIDE == 4
    for retired_name in (
        "SPMLegacy",
        "UNetDecoder",
        "CoMerCNN",
        "DinoV3Backbone",
        "_build_fm_encoder",
    ):
        assert not hasattr(dinov3, retired_name)


@pytest.mark.parametrize(
    "retired_key",
    [
        "adapter_dim",
        "enable_boundary_loss",
        "fm_model",
        "ln_spm",
        "use_comer",
        "use_unet_decoder",
    ],
)
def test_network_builder_rejects_retired_architecture_keys(retired_key):
    with pytest.raises(ValueError, match="unsupported network construction"):
        common.build(
            {
                "head_type": "flow",
                "num_classes": 0,
                retired_key: False,
            }
        )
