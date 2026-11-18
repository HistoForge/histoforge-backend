"""Strict construction of the recipe-declared point head and complete checkpoint."""

from histoforge._vendor.histoforge_models.pathology.common import build_and_load


def build(config, state_dict):
    return build_and_load(
        {"head_type": "point_detection", "num_classes": config["num_classes"]},
        state_dict,
    )
