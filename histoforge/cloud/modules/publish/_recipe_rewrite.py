"""Pure transform: local (working-form) recipe → the recipe published to HF.

The vendor transform deep-copies the recipe and materializes each `local`
weight as `{type: "vendored", file: "<model-id>.safetensors"}`. Existing
`vendored` filenames and pinned `referenced` declarations remain unchanged.

Deliberately narrow otherwise — no other fields added or removed. Future
contributors should NOT "helpfully" sanitize author-declared fields between
publishes; that silently breaks reproducibility.
"""

import copy


def to_published_recipe(local: dict) -> dict:
    """Deep-copy ``local`` and materialize only local weight declarations."""
    recipe = copy.deepcopy(local)
    for model_id, model in recipe.get("models", {}).items():
        if model.get("weights", {}).get("type") == "local":
            model["weights"] = {
                "type": "vendored",
                "file": f"{model_id}.safetensors",
            }
    return recipe
