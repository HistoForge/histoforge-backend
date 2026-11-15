"""Per-architecture net factory for `dinov3-adapter-semantic` (DS).

Dense semantic inference uses the fixed flow-shaped backbone with a class head;
flow channels are ignored. The arch key fixes the complete network
architecture, and only the recipe's class count enters construction.
"""

from torch import nn

from histoforge._vendor.histoforge_models.pathology.common import build_and_load


def build(config: dict, state_dict: dict) -> nn.Module:
    """Construct + load a `dinov3-adapter-semantic` net from config + state_dict."""
    return build_and_load(
        {"head_type": "flow", "num_classes": config["num_classes"]},
        state_dict,
    )
