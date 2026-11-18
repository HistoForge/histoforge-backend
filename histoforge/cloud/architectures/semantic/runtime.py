"""Ordinary semantic inference shared by project prediction and dense validation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from histoforge._vendor.histoforge_models.pathology.network_execution import (
    NetworkPrecision,
)
from histoforge.cloud.architectures.registry import runner_field_configuration
from histoforge.cloud.modules.schema import (
    model_output_class_channels,
    model_output_emitted_classes,
)

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor, nn

    from histoforge.cloud.rasterization.rasterizer import RasterizedImage

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SemanticModelRuntime:
    """Exact loaded network and field settings, before output polygon projection."""

    network: Any
    num_classes: int
    field_options: Mapping[str, Any]
    warmup_side: int
    batch_size: int
    emitted_classes: tuple[str, ...]
    class_by_channel: Mapping[int, str]
    network_precision: NetworkPrecision = "fp32"

    def predict(self, image: RasterizedImage) -> dict[str, Any]:
        return self._predict(image.pixels, image.real_pixel_mask)

    def predict_window(
        self,
        image: RasterizedImage,
        plan: Mapping[str, Any],
        window: Mapping[str, Any],
        *,
        channel_percentiles=None,
    ) -> dict[str, Any]:
        """Return independent core labels selected over every class channel.

        The caller places these cropped labels at the core's window-local offset
        during geometry conversion. Dense fields and full-window labels do not
        escape this call.
        """

        from histoforge._vendor.histoforge_models.pathology.network_execution import (
            forward_network,
        )
        from histoforge._vendor.histoforge_models.pathology.semantic_inference import (
            class_labels_from_maps,
        )
        from histoforge._vendor.histoforge_models.pathology.window_maps import (
            predict_window_maps,
        )

        target_width = max(
            item["core_rect"]["x"] + item["core_rect"]["width"]
            for item in plan["windows"]
        )
        target_height = max(
            item["core_rect"]["y"] + item["core_rect"]["height"]
            for item in plan["windows"]
        )
        rect = window["input_rect"]
        started = perf_counter()
        maps = predict_window_maps(
            image.pixels,
            real_pixel_mask=image.real_pixel_mask,
            target_height=target_height,
            target_width=target_width,
            window_x=rect["x"],
            window_y=rect["y"],
            field_options=self.field_options,
            batch_size=self.batch_size,
            forward_tiles=lambda tiles: forward_network(
                self.network, tiles, network_precision=self.network_precision
            )[0],
            channel_percentiles=channel_percentiles,
        )
        # Includes preprocessing, forward execution, transfers and blending.
        _LOGGER.info(
            "stage=field_prediction elapsed_seconds=%.6f", perf_counter() - started
        )
        started = perf_counter()
        labels = class_labels_from_maps(maps, self.num_classes)
        core = window["core_rect"]
        left, top = core["x"] - rect["x"], core["y"] - rect["y"]
        core_labels = labels[
            top : top + core["height"], left : left + core["width"]
        ].copy()
        _LOGGER.info(
            "stage=classify_and_crop elapsed_seconds=%.6f", perf_counter() - started
        )
        return {"class_map": core_labels}

    def warmup(self) -> None:
        import numpy as np

        self._predict(
            np.zeros((self.warmup_side, self.warmup_side, 3), dtype=np.uint8), None
        )

    def _predict(
        self, pixels: np.ndarray, real_pixel_mask: np.ndarray | None
    ) -> dict[str, Any]:
        from histoforge._vendor.histoforge_models.pathology.semantic_inference import (
            run_semantic_inference,
        )

        return run_semantic_inference(
            self.network,
            pixels,
            num_classes=self.num_classes,
            batch_size=self.batch_size,
            real_pixel_mask=real_pixel_mask,
            include_diagnostics=False,
            network_precision=self.network_precision,
            **self.field_options,
        )


def create_semantic_runtime(
    model: Mapping[str, Any],
    network: nn.Module,
    *,
    network_precision: NetworkPrecision = "fp32",
) -> SemanticModelRuntime:
    """Freeze predictor arguments, retaining FP32 for managed validation.

    Serving callers pass precision explicitly; this factory does not consult the
    serving registry and cannot silently change training checkpoint selection.
    """

    if (
        model["arch"] != "dinov3-adapter-semantic"
        or model["output"]["type"] != "dense_classification"
    ):
        raise ValueError("semantic runtime requires a dense semantic model")
    field_options, warmup_side = runner_field_configuration(model)
    return SemanticModelRuntime(
        network,
        model["params"]["num_classes"],
        field_options,
        warmup_side,
        model["params"].get("batch_size", 8),
        model_output_emitted_classes(model["output"]),
        {
            item["channel"]: item["pathClass"]
            for item in model_output_class_channels(model["output"])
            if item["role"] == "emitted"
        },
        network_precision=network_precision,
    )


def prepare_semantic_finetuning(network: nn.Module) -> None:
    """Freeze unused flow readouts, preserving the full semantic checkpoint layout.

    This matches standalone dense training. A zero gradient is insufficient:
    AdamW could otherwise decay an unused but still-trainable tail.
    """

    for parameter in network.out.parameters():
        parameter.requires_grad_(False)
    network.W2.requires_grad_(False)


def load_semantic_model_state(
    network: nn.Module, tensor_state: Mapping[str, Any]
) -> None:
    """Selected checkpoints use the same strict loader as normal reconstruction."""

    from histoforge._vendor.histoforge_models.pathology.common import (
        load_state_dict_exact,
    )

    load_state_dict_exact(network, tensor_state)


def semantic_training_logits(
    network: nn.Module, images: Tensor, class_count: int
) -> Tensor:
    """Expose the semantic prefix without changing the source network topology."""

    from torch import Tensor

    predictions = network(images)
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    if not isinstance(predictions, Tensor) or predictions.shape != (
        images.shape[0],
        class_count + 3,
        *images.shape[2:],
    ):
        raise ValueError(
            "semantic model must return its complete class-plus-flow layout"
        )
    return predictions[:, :class_count]
