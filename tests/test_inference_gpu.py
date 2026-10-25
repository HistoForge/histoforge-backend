"""Inference GPU choices are explicit, bounded, and recipe-preserving."""

import pytest
from histoforge.cloud.inference.gpu import (
    INFERENCE_GPU_OVERRIDES,
    require_supported_inference_gpu,
    validate_inference_gpu_choices,
)


def test_capability_choices_are_nonempty_unique_strings_and_detached():
    source = list(INFERENCE_GPU_OVERRIDES)
    assert validate_inference_gpu_choices(source) == source
    source[0] = "changed"
    assert validate_inference_gpu_choices(list(INFERENCE_GPU_OVERRIDES)) == [
        "A10G",
        "L40S",
        "A100",
    ]

    for invalid in (None, {}, [], [""], ["A100", "A100"]):
        with pytest.raises(ValueError):
            validate_inference_gpu_choices(invalid)


def test_resolution_accepts_recipe_gpu_or_one_curated_override():
    assert require_supported_inference_gpu("H200", "H200") == "H200"
    for choice in INFERENCE_GPU_OVERRIDES:
        assert require_supported_inference_gpu(choice, "H200") == choice

    with pytest.raises(ValueError, match="not supported"):
        require_supported_inference_gpu("T4", "H200")
