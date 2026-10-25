"""Per-architecture net factory for `dinov3-adapter-hovernet`.

The runner reads the state dict via safetensors and hands it here. The arch key
fixes the complete network architecture; only the recipe's class count enters
construction.
"""

from torch import nn

from histoforge._vendor.histoforge_models.pathology.common import build_and_load


def build(config: dict, state_dict: dict) -> nn.Module:
    """Construct + load a `dinov3-adapter-hovernet` net from config + state_dict."""
    return build_and_load(
        {"head_type": "hovernet", "num_classes": config["num_classes"]},
        state_dict,
    )
