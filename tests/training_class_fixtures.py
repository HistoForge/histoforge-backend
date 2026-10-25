"""Tiny image encoder with the real adapter readout and strict checkpoint layout.

Only the expensive feature extractor is replaced. Head construction, adaptation,
loading, readout and the ordinary inference pipeline remain production code.
"""

import torch
from histoforge._vendor.histoforge_models.pathology.class_head import (
    initialize_decoupled_class_head,
)
from histoforge._vendor.histoforge_models.pathology.dinov3 import DinoV3AdapterBackbone
from torch import nn


class TinyClassAdapter(nn.Module):
    def __init__(self, *, head_type="flow", dtype=torch.float32):
        super().__init__()
        self.ps = 4
        self.nclasses = 0
        self.backbone = nn.Module()
        self.backbone.patch_embed = nn.Conv2d(3, 8, 4, stride=4)
        self.backbone.blocks = nn.ModuleList([nn.Conv2d(8, 8, 1)])
        self.neck = nn.Sequential(nn.Conv2d(8, 256, 1))
        self.register_buffer("head_type_code", torch.tensor(0))
        self.register_buffer("adapter_dim_buf", torch.tensor(256))
        DinoV3AdapterBackbone.reset_segmentation_head(self, head_type)
        self.to(dtype=dtype)

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
        output = DinoV3AdapterBackbone._readout(self, self.neck(value))
        return output, torch.zeros(len(image), 256, device=image.device)


def initialized_adapter(class_count, *, head_type="flow"):
    network = TinyClassAdapter(head_type=head_type)
    initialize_decoupled_class_head(network, class_count)
    return network
