"""Versioned Modal inference-GPU policy shared by handshake and admission."""

from __future__ import annotations

from typing import Any

from histoforge.canonical_json import require_nonblank_string

INFERENCE_GPU_OVERRIDES = ("A10G", "L40S", "A100", "H200")


def validate_inference_gpu_choices(value: Any) -> list[str]:
    """Validate one nonempty, unique, ordered capability projection."""

    if not isinstance(value, list) or not value:
        raise ValueError("inference_gpus must be a nonempty array")
    result = [
        require_nonblank_string(item, f"inference_gpus[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ValueError("inference_gpus contains duplicate values")
    return result


def require_supported_inference_gpu(requested: Any, recipe_gpu: Any) -> str:
    """Accept the recipe default or one explicit HistoForge-curated override."""

    selected = require_nonblank_string(requested, "inference GPU")
    default = require_nonblank_string(recipe_gpu, "recipe inference GPU")
    if selected != default and selected not in INFERENCE_GPU_OVERRIDES:
        raise ValueError("inference GPU is not supported by this deployment")
    return selected
