"""Modal runner for the `dinov3-adapter-cellpose` architecture family.

The warm key includes the module content, GPU and inference precision. Exact
activation loads the installed model and its fixed or variable field settings.
The shared `run_flow_inference` wrapper aggregates raw field outputs, decodes
instances, and applies per-instance class voting.

For a class head, the recipe maps every explicit tensor channel to background,
emitted, or training-only behavior. Voting uses the complete head before only
emitted winners are projected to PathClasses. A pure instance model instead
applies its sole post-decoding class to every instance.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from threading import Event, Lock
from time import perf_counter

import modal

from histoforge.cloud._app import app, artifact_volume, image, model_volume
from histoforge.cloud.architectures.instance_runtime import (
    create_instance_runtime,
    instance_network_builder,
)
from histoforge.cloud.architectures.registry import (
    runner_build_for_precision,
    runner_network_precision,
    runner_resources,
)

_ARCHITECTURE = "dinov3-adapter-cellpose"
_RESOURCES = runner_resources(_ARCHITECTURE)
_LOGGER = logging.getLogger(__name__)


@app.cls(
    image=image,
    gpu=_RESOURCES.gpu,
    cpu=_RESOURCES.cpu_cores,
    memory=_RESOURCES.memory_mib,
    timeout=_RESOURCES.attempt_timeout_seconds,
    volumes={"/modules": model_volume, "/artifacts": artifact_volume},
    max_containers=_RESOURCES.max_containers,
    retries=0,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=_RESOURCES.max_concurrent_inputs)
class CellposeRunner:
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
                instance_network_builder(_ARCHITECTURE),
                self.gpu,
                runner_build_for_precision(_ARCHITECTURE, self.inference_precision),
            )
            network = activation["network"].to("cuda").eval()
            model = activation["model"]
            self.runtime = create_instance_runtime(
                model,
                network,
                network_precision=runner_network_precision(
                    _ARCHITECTURE, self.inference_precision
                ),
            )

            print(
                f"[{self.ref}/{self.model_id}] loaded "
                f"num_classes={self.runtime.num_classes} "
                f"fields={self.runtime.field_options}"
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
        from histoforge.cloud.inference.runs.core_halo_output import (
            initialize_geometry_worker,
        )

        self.source_store, self.source_files, self.artifact_files = (
            mounted_inference_stores()
        )
        # Live locks/processes must not be captured in the GPU memory snapshot.
        self._model_operation_lock = Lock()
        self._postprocessing_unavailable = Event()
        self._geometry_executor = ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn")
        )
        try:
            self._geometry_executor.submit(initialize_geometry_worker).result()
        except Exception:
            self._geometry_executor.shutdown(wait=False, cancel_futures=True)
            raise
        print(f"[{self.ref}] restored — CUDA: {torch.cuda.is_available()}")

    @modal.exit()
    def shutdown(self):
        """Close the worker after inputs drain; Modal owns forced termination."""

        executor = getattr(self, "_geometry_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    @modal.method()
    def infer(self, request: dict) -> dict:
        from histoforge.cloud.inference.finite.runtime import execute_finite_request

        with self._model_operation_lock:
            self._require_postprocessing_available()
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

        with self._model_operation_lock:
            self._require_postprocessing_available()
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

    @modal.method()
    def infer_durable_core_halo_unit(
        self,
        request: dict,
        input_id: int,
        unit_id: int,
        model_run_limits: dict,
        source_bindings=(),
        channel_percentiles=None,
    ) -> dict:
        """Publish one complete fixed CoreHalo ownership unit."""

        from histoforge.cloud.inference.runs.core_halo_runtime import (
            execute_core_halo_unit,
        )

        self._require_postprocessing_available()
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
            self._encode_core_halo_window,
            source_bindings=source_bindings,
            channel_percentiles=channel_percentiles,
        )

    def _infer_features(self, image, result_budget):
        """Run this architecture and return its complete feature collection."""

        from histoforge.cloud.inference.mask_features import (
            feature_collection_from_mask,
        )

        result = self.runtime.predict(image)
        started = perf_counter()
        features = feature_collection_from_mask(
            result["instance_mask"],
            self.runtime.classifications(result),
            self.runtime.emitted_classes,
            result_budget,
        )
        _LOGGER.info(
            "stage=geometry_conversion elapsed_seconds=%.6f", perf_counter() - started
        )
        return features

    def _predict_core_halo_window(
        self, read_window_image, plan, window, *, channel_percentiles
    ):
        """Decode one window, retaining only labels and emitted class decisions."""

        # Acquire before allocating pixels, not just before the network forward.
        # The reader closes before predict_window; fields leave scope on return.
        with self._model_operation_lock:
            self._require_postprocessing_available()
            result = self.runtime.predict_window(
                read_window_image(),
                plan,
                window,
                channel_percentiles=channel_percentiles,
            )
            return result["instance_mask"], self.runtime.classifications(result)

    def _encode_core_halo_window(
        self, prediction, window, declared_classes, target_raster, processing_limits
    ):
        """Wait for this input's CPU conversion without holding compute or IO locks.

        Modal admits at most two inputs. Each submits once and waits here, so
        the single worker has at most two outstanding conversions. Only compact
        decoded arrays and geometry/limit metadata cross the process boundary.
        """

        from histoforge.cloud.inference.runs.core_halo_output import (
            encode_instance_window,
        )

        try:
            return self._geometry_executor.submit(
                encode_instance_window,
                prediction,
                window,
                declared_classes,
                target_raster,
                processing_limits,
            ).result()
        except BrokenProcessPool as error:
            from modal.experimental import stop_fetching_inputs

            from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure

            self._postprocessing_unavailable.set()
            stop_fetching_inputs()
            raise GatewayFailure(
                GatewayErrorCode.RUNNER_FAILED,
                "the CPU output worker is unavailable",
            ) from error

    def _require_postprocessing_available(self):
        """Reject newly admitted work after a child failure, including finite calls."""

        if self._postprocessing_unavailable.is_set():
            from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure

            raise GatewayFailure(
                GatewayErrorCode.RUNNER_FAILED,
                "the CPU output worker is unavailable",
            )
