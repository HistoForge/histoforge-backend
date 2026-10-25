"""Modal adapter for dense semantic segmentation.

The model assigns one class to each pixel. Only recipe-declared emitted classes
become region geometry; background and other winning classes remain excluded.
Finite inputs and processing windows share class selection and polygon conversion.
"""

import logging
from time import perf_counter

import modal

from histoforge.cloud._app import app, artifact_volume, image, model_volume
from histoforge.cloud.architectures.registry import (
    runner_build_for_precision,
    runner_network_precision,
    runner_resources,
)

_ARCHITECTURE = "dinov3-adapter-semantic"
_RESOURCES = runner_resources(_ARCHITECTURE)
_LOGGER = logging.getLogger(__name__)


@app.cls(
    image=image,
    gpu=_RESOURCES.gpu,
    timeout=_RESOURCES.attempt_timeout_seconds,
    volumes={"/modules": model_volume, "/artifacts": artifact_volume},
    max_containers=_RESOURCES.max_containers,
    retries=0,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=_RESOURCES.max_concurrent_inputs)
class SemanticRunner:
    ref: str = modal.parameter()
    model_id: str = modal.parameter()
    content_digest: str = modal.parameter()
    gpu: str = modal.parameter()
    inference_precision: str = modal.parameter(default="default")
    resolved_commit_sha: str = modal.parameter(default="")

    @modal.enter(snap=True)
    def load_model(self):
        import torch

        from histoforge.cloud._errors import GatewayFailure
        from histoforge.cloud.architectures.semantic.net import build as build_net
        from histoforge.cloud.architectures.semantic.runtime import (
            create_semantic_runtime,
        )
        from histoforge.cloud.inference.activation import (
            activate_model,
            activation_failure,
        )

        print(f"CUDA available: {torch.cuda.is_available()}")

        try:
            activation = activate_model(
                self.ref,
                self.model_id,
                self.content_digest,
                self.resolved_commit_sha or None,
                _ARCHITECTURE,
                build_net,
                self.gpu,
                runner_build_for_precision(_ARCHITECTURE, self.inference_precision),
            )
            network = activation["network"].to("cuda").eval()
            self.runtime = create_semantic_runtime(
                activation["model"],
                network,
                network_precision=runner_network_precision(
                    _ARCHITECTURE, self.inference_precision
                ),
            )
            self.runtime.warmup()
            self.activation = activation
            print(f"[{self.ref}] warmup complete, snapshot ready")
        except GatewayFailure:
            raise
        except Exception as error:
            raise activation_failure(error, f"{self.ref}/{self.model_id}") from error

    @modal.enter(snap=False)
    def post_restore(self):
        import torch

        from histoforge.cloud.inference.execution import mounted_inference_stores

        self.source_store, self.source_files, self.artifact_files = (
            mounted_inference_stores()
        )
        print(f"[{self.ref}] restored — CUDA: {torch.cuda.is_available()}")

    @modal.method()
    def infer(self, request: dict) -> dict:
        from histoforge.cloud.inference.finite.runtime import execute_finite_request

        return execute_finite_request(
            request,
            self.activation,
            self.source_store,
            self.source_files,
            self._infer_features,
        )

    @modal.method()
    def infer_durable_input(
        self, request: dict, input_id: int, model_run_limits: dict, source_bindings=()
    ) -> dict:
        """Publish one complete finite-plan input for a durable run."""

        from histoforge.cloud.inference.runs.runtime import (
            execute_durable_finite_input,
        )

        return execute_durable_finite_input(
            request,
            input_id,
            model_run_limits,
            self.activation,
            self.source_store,
            self.source_files,
            self.artifact_files,
            self._infer_features,
            source_bindings=source_bindings,
        )

    def _infer_features(self, image, result_budget):
        """Run this architecture and return its complete feature collection."""

        from histoforge.cloud.inference.semantic_features import (
            feature_collection_from_class_map,
        )

        result = self.runtime.predict(image)
        started = perf_counter()
        features = feature_collection_from_class_map(
            result["class_map"],
            self.runtime.class_by_channel,
            self.runtime.emitted_classes,
            result_budget,
        )
        _LOGGER.info(
            "stage=geometry_conversion elapsed_seconds=%.6f", perf_counter() - started
        )
        return features

    @modal.method()
    def infer_durable_core_halo_unit(
        self,
        request: dict,
        input_id: int,
        unit_id: int,
        model_run_limits: dict,
        source_bindings=(),
    ) -> dict:
        """Predict a core and its context, then save only the core's regions."""

        from histoforge.cloud.inference.runs.core_halo_output import (
            encode_semantic_window,
        )
        from histoforge.cloud.inference.runs.core_halo_runtime import (
            execute_core_halo_unit,
        )

        return execute_core_halo_unit(
            request,
            input_id,
            unit_id,
            model_run_limits,
            self.activation,
            self.source_store,
            self.source_files,
            self.artifact_files,
            self._predict_core_halo_window,
            encode_semantic_window,
            source_bindings=source_bindings,
        )

    def _predict_core_halo_window(self, read_window_image, plan, window):
        """Return the already cropped core labels and their class mapping."""

        result = self.runtime.predict_window(read_window_image(), plan, window)
        return result["class_map"], self.runtime.class_by_channel
