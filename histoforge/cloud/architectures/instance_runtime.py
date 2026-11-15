"""Shared construction, output extraction and decoding for instance models.

The Modal runners and managed trainer use this owner so a validation prediction
cannot drift from normal finite inference. It owns model-field and decoder
arguments plus training-tensor extraction. Source reads, training losses/state,
feature serialization and large-region ownership remain with their existing
components.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
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
    from torch import Tensor, nn

    from histoforge.cloud.rasterization.rasterizer import RasterizedImage

_CELLPOSE = "dinov3-adapter-cellpose"
_HOVERNET = "dinov3-adapter-hovernet"
_INSTANCE_ARCHITECTURES = {_CELLPOSE, _HOVERNET}
_HOVERNET_DECODER_PARAMETERS = (
    "edge_threshold",
    "fg_threshold",
    "object_size",
    "morph_kernel",
    "ksize",
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstanceModelRuntime:
    """One loaded network plus its exact ordinary inference arguments."""

    network: Any
    architecture: str
    num_classes: int
    field_options: Mapping[str, Any]
    warmup_side: int
    batch_size: int | None
    decoder_options: Mapping[str, Any]
    emitted_classes: tuple[str, ...]
    class_by_channel: Mapping[int, str]
    network_precision: NetworkPrecision = "fp32"

    def warmup(self) -> None:
        """Exercise one preferred-size field using ordinary decoding."""

        import numpy as np

        pixels = np.zeros((self.warmup_side, self.warmup_side, 3), dtype=np.uint8)
        self._predict(pixels, None)

    def predict(self, image: RasterizedImage) -> dict[str, Any]:
        """Return complete instance labels and all winning class channels."""

        return self._predict(image.pixels, image.real_pixel_mask)

    def predict_window(
        self,
        image: RasterizedImage,
        plan: Mapping[str, Any],
        window: Mapping[str, Any],
        *,
        channel_percentiles=None,
    ) -> dict[str, Any]:
        """Decode a target-anchored window, retaining its complete halo labels.

        Field arrays stay inside this call. Class votes still include background
        and training-only channels; projection and ownership belong to callers.
        Cellpose integrates flows in target coordinates, unlike finite decoding.
        """

        import numpy as np

        from histoforge._vendor.histoforge_models.pathology.hovernet_inference import (
            DEFAULT_BATCH_SIZE,
        )
        from histoforge._vendor.histoforge_models.pathology.network_execution import (
            forward_network,
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
        fields = predict_window_maps(
            image.pixels,
            real_pixel_mask=image.real_pixel_mask,
            target_height=target_height,
            target_width=target_width,
            window_x=rect["x"],
            window_y=rect["y"],
            field_options=self.field_options,
            batch_size=(
                DEFAULT_BATCH_SIZE if self.batch_size is None else self.batch_size
            ),
            forward_tiles=lambda tiles: forward_network(
                self.network, tiles, network_precision=self.network_precision
            )[0],
            channel_percentiles=channel_percentiles,
        )
        # Wall time includes normalization, network execution, transfers and
        # blending. It is not an isolated GPU kernel measurement.
        _LOGGER.info(
            "stage=field_prediction elapsed_seconds=%.6f", perf_counter() - started
        )
        if self.architecture == _CELLPOSE:
            from histoforge._vendor.histoforge_models.pathology import (
                cellpose_window_decode,
            )
            from histoforge._vendor.histoforge_models.pathology.common import (
                per_instance_majority_vote,
            )

            started = perf_counter()
            if fields.shape[2] < self.num_classes + 3:
                raise ValueError("Cellpose CoreHalo output has too few channels")
            flows = np.stack([fields[..., -3], fields[..., -2]], axis=0).astype(
                np.float32, copy=False
            )
            cell_probability = fields[..., -1].astype(np.float32, copy=False)
            instance_mask = (
                cellpose_window_decode.compute_cellpose_masks_in_target_coordinates(
                    flows,
                    cell_probability,
                    target_height=target_height,
                    target_width=target_width,
                    window_x=rect["x"],
                    window_y=rect["y"],
                    **self.decoder_options,
                    device=self.network.device,
                )
            )
            result = {"instance_mask": instance_mask}
            if self.num_classes:
                result["votes_per_instance"] = per_instance_majority_vote(
                    fields[..., : self.num_classes].transpose(2, 0, 1), instance_mask
                )
            _LOGGER.info(
                "stage=decode_and_classify elapsed_seconds=%.6f",
                perf_counter() - started,
            )
            return result
        if self.architecture == _HOVERNET:
            from histoforge._vendor.histoforge_models.pathology import (
                hovernet_inference,
            )

            # Truncated instances must survive area filtering until the existing
            # ownership check can reject incomplete owned geometry.
            protected_edges = tuple(
                edge
                for edge, artificial in (
                    ("top", rect["y"] > 0),
                    ("bottom", rect["y"] + rect["height"] < target_height),
                    ("left", rect["x"] > 0),
                    ("right", rect["x"] + rect["width"] < target_width),
                )
                if artificial
            )
            return hovernet_inference.decode_hovernet_maps(
                fields,
                num_classes=self.num_classes,
                **self.decoder_options,
                protected_edges=protected_edges,
                include_diagnostics=False,
            )
        raise ValueError(f"unsupported instance architecture: {self.architecture}")

    def classifications(self, result: Mapping[str, Any]) -> Mapping[int, str] | str:
        """Project decoded tensor channels to emitted QuPath classes."""

        if not self.num_classes:
            return self.emitted_classes[0]
        return {
            int(instance_id): self.class_by_channel[channel]
            for instance_id, channel in result["votes_per_instance"].items()
            if channel in self.class_by_channel
        }

    def _predict(self, pixels: Any, real_pixel_mask: Any | None) -> dict[str, Any]:
        options = {
            "num_classes": self.num_classes,
            **self.field_options,
            **self.decoder_options,
            "include_diagnostics": False,
            "network_precision": self.network_precision,
        }
        if self.batch_size is not None:
            options["batch_size"] = self.batch_size
        if real_pixel_mask is not None:
            options["real_pixel_mask"] = real_pixel_mask
        if self.architecture == _CELLPOSE:
            from histoforge._vendor.histoforge_models.pathology.flow_inference import (
                run_flow_inference,
            )

            return run_flow_inference(self.network, pixels, **options)
        if self.architecture == _HOVERNET:
            from histoforge._vendor.histoforge_models.pathology import (
                hovernet_inference,
            )

            return hovernet_inference.run_hovernet_inference(
                self.network, pixels, **options
            )
        raise ValueError(f"unsupported instance architecture: {self.architecture}")


def create_instance_runtime(
    model: Mapping[str, Any],
    network: Any,
    *,
    network_precision: NetworkPrecision = "fp32",
) -> InstanceModelRuntime:
    """Capture predictor arguments; training validation keeps the FP32 default.

    Serving callers explicitly pass their build-selected precision. Looking it
    up here would silently change managed validation when serving policy changes.
    """

    architecture = model.get("arch")
    if architecture not in _INSTANCE_ARCHITECTURES:
        raise ValueError(f"unsupported instance architecture: {architecture}")
    parameters = model["params"]
    output = model["output"]
    field_options, warmup_side = runner_field_configuration(model)
    channels = model_output_class_channels(output)
    return InstanceModelRuntime(
        network=network,
        network_precision=network_precision,
        architecture=architecture,
        num_classes=int(parameters.get("num_classes", 0)),
        field_options=field_options,
        warmup_side=warmup_side,
        batch_size=(
            int(parameters.get("batch_size", 8)) if architecture == _CELLPOSE else None
        ),
        decoder_options={
            "min_instance_pixels": parameters["min_instance_pixels"],
            **(
                {
                    name: parameters[name]
                    for name in _HOVERNET_DECODER_PARAMETERS
                    if name in parameters
                }
                if architecture == _HOVERNET
                # Product predictions must not remove an instance merely because
                # it occupies over 40% of its ROI or core-plus-halo image. The
                # shared scientific wrapper keeps its default for other callers.
                else {"max_size_fraction": 1.0}
            ),
        },
        emitted_classes=model_output_emitted_classes(output),
        class_by_channel={
            item["channel"]: item["pathClass"]
            for item in channels
            if item["role"] == "emitted"
        },
    )


def load_instance_model_state(network: Any, tensor_state: Mapping[str, Any]) -> None:
    """Reload the selected checkpoint through the ordinary exact tensor loader."""

    from histoforge._vendor.histoforge_models.pathology.common import (
        load_state_dict_exact,
    )

    load_state_dict_exact(network, tensor_state)


def instance_training_predictions(network: nn.Module, images: Tensor) -> Tensor:
    """Extract the flow/HoVer tensor from the supported network return layouts."""

    from torch import Tensor

    output = network(images)
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("model returned an empty output sequence")
        output = output[0]
    if not isinstance(output, Tensor):
        raise ValueError(
            "model must return a tensor or a sequence whose first item is a tensor"
        )
    return output


def instance_network_builder(
    architecture: str,
) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    """Return the ordinary exact network builder for one instance family."""

    if architecture == _CELLPOSE:
        from histoforge.cloud.architectures.cellpose.net import build

        return build
    if architecture == _HOVERNET:
        from histoforge.cloud.architectures.hovernet.net import build

        return build
    raise ValueError(f"unsupported instance architecture: {architecture}")
