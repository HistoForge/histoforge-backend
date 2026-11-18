"""Finite point prediction with model-owned settings and caller-owned count caps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud.architectures.registry import require_available_model
from histoforge.cloud.modules.schema import model_output_emitted_classes


@dataclass(frozen=True)
class PointModelRuntime:
    """One loaded FP32 network and its captured finite inference arguments."""

    network: Any
    field_profile: dict[str, Any]
    emitted_classes: tuple[str, ...]
    score_threshold: float
    peak_radius_model_px: int
    batch_size: int

    def predict(self, image, *, max_detections: int):
        from histoforge._vendor.histoforge_models.pathology.point_inference import (
            run_point_inference,
        )

        return run_point_inference(
            self.network,
            image.pixels,
            field_profile=self.field_profile,
            score_threshold=self.score_threshold,
            peak_radius_model_px=self.peak_radius_model_px,
            max_detections=max_detections,
            batch_size=self.batch_size,
            real_pixel_mask=image.real_pixel_mask,
        )


def create_point_runtime(model, network) -> PointModelRuntime:
    """Validate configuration before freezing independent sigmoid class meaning."""

    if model["arch"] != "dinov3-adapter-point":
        raise ValueError("point runtime requires the point architecture")
    require_available_model(model)
    params = model["params"]
    return PointModelRuntime(
        network=network,
        field_profile=canonical_tree(model["input"]["fieldProfile"]),
        emitted_classes=model_output_emitted_classes(model["output"]),
        score_threshold=params["score_threshold"],
        peak_radius_model_px=params["peak_radius_model_px"],
        batch_size=params.get("batch_size", 8),
    )
