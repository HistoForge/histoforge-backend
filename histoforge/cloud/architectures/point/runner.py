"""Exact finite point inference, without BF16 or durable/large-region support."""

import modal

from histoforge.cloud._app import app, artifact_volume, image, model_volume
from histoforge.cloud.architectures.registry import (
    runner_build_for_precision,
    runner_resources,
)

_ARCHITECTURE = "dinov3-adapter-point"
_RESOURCES = runner_resources(_ARCHITECTURE)


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
class PointRunner:
    ref: str = modal.parameter()
    model_id: str = modal.parameter()
    content_digest: str = modal.parameter()
    gpu: str = modal.parameter()
    inference_precision: str = modal.parameter(default="default")
    resolved_commit_sha: str = modal.parameter(default="")

    @modal.enter(snap=True)
    def load_model(self):
        import numpy as np

        from histoforge._vendor.histoforge_models.pathology.point_inference import (
            PointDetectionLimitExceeded,
        )
        from histoforge.cloud._errors import GatewayFailure
        from histoforge.cloud.architectures.point.net import build
        from histoforge.cloud.architectures.point.runtime import create_point_runtime
        from histoforge.cloud.inference.activation import (
            activate_model,
            activation_failure,
        )
        from histoforge.cloud.rasterization.rasterizer import RasterizedImage

        try:
            activation = activate_model(
                self.ref,
                self.model_id,
                self.content_digest,
                self.resolved_commit_sha or None,
                _ARCHITECTURE,
                build,
                self.gpu,
                runner_build_for_precision(_ARCHITECTURE, self.inference_precision),
            )
            network = activation["network"].to("cuda").eval()
            self.runtime = create_point_runtime(activation["model"], network)
            side = self.runtime.field_profile["preferredTileSideModelPx"]
            try:
                self.runtime.predict(
                    RasterizedImage(
                        np.zeros((side, side, 3), np.uint8),
                        np.ones((side, side), bool),
                    ),
                    max_detections=activation["capabilities"]["finite"]["result"][
                        "max_features"
                    ],
                )
            except PointDetectionLimitExceeded:
                # Synthetic warmup density does not decide a real input's result cap.
                pass
            self.activation = activation
        except GatewayFailure:
            raise
        except Exception as error:
            raise activation_failure(error, f"{self.ref}/{self.model_id}") from error

    @modal.enter(snap=False)
    def post_restore(self):
        from histoforge.cloud.inference.execution import mounted_inference_stores

        self.source_store, self.source_files, _ = mounted_inference_stores()

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

    def _infer_features(self, image, result_budget):
        from histoforge._vendor.histoforge_models.pathology.point_inference import (
            PointDetectionLimitExceeded,
        )
        from histoforge.cloud.inference.features import (
            ResultLimitExceeded,
            build_point_feature_records,
        )

        try:
            result = self.runtime.predict(
                image,
                max_detections=min(
                    result_budget["max_features"], result_budget["max_vertices"]
                ),
            )
        except PointDetectionLimitExceeded as error:
            raise ResultLimitExceeded(str(error)) from error
        classes = self.runtime.emitted_classes
        return build_point_feature_records(
            (
                (classes[channel], xy, score)
                for xy, channel, score in zip(
                    result["xy"], result["class_indices"], result["scores"], strict=True
                )
            ),
            classes,
            result_budget,
        ).as_collection()
