"""Bounded CoreHalo raster, unit-state, and final-input execution.

Architecture runners decode labels before pure output conversion constructs
polygons. Instance labels keep the complete processing-image shape; semantic
labels are cropped to the core. This module owns deferred bounded image reads,
parent-process publication, and final assembly of accepted completed windows.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

from shapely.geometry import box, shape

from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.architectures.registry import processing_result_limits
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
    ArtifactFileStore,
)
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
)
from histoforge.cloud.inference.execution import (
    model_declared_classes,
    require_activation_identity,
)
from histoforge.cloud.inference.feature_stream import EncodedFeatureStream
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_feature_collection,
)
from histoforge.cloud.inference.runs.core_halo import (
    InsufficientCoreHaloContext,
)
from histoforge.cloud.inference.runs.core_halo_output import encode_core_halo_collection
from histoforge.cloud.inference.runs.core_halo_state import (
    merge_core_halo_unit_states,
    publish_core_halo_unit_state,
)
from histoforge.cloud.inference.runs.publications import (
    FEATURE_STREAM_CONTENT_ENCODING,
    FEATURE_STREAM_MEDIA_TYPE,
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
)
from histoforge.cloud.inference.runs.request import validate_model_run_resource_limits
from histoforge.cloud.inference.runs.source_bindings import (
    input_source_bindings,
    unit_source_binding,
)
from histoforge.cloud.rasterization.inputs import model_input_source_artifact
from histoforge.cloud.rasterization.model_input import (
    prepare_partitioned_model_raster,
)
from histoforge.cloud.rasterization.partitioned import PartitionedRaster
from histoforge.cloud.rasterization.rasterizer import (
    InsufficientSourceContext,
    RasterizedImage,
)
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.store import SourceArtifactStore

_LOGGER = logging.getLogger(__name__)
_Prediction = TypeVar("_Prediction")


@dataclass(frozen=True)
class _CoreHaloContext:
    request: dict[str, Any]
    model_input: dict[str, Any]
    plan: dict[str, Any]
    classes: list[str]
    output_type: str
    input_id: int


def execute_core_halo_unit(
    request_value: Any,
    input_id: int,
    unit_id: int,
    model_run_limits: Any,
    activation: Mapping[str, Any],
    source_store: SourceArtifactStore,
    source_files: SourceFileStore,
    files: ArtifactFileStore,
    predict_window: Callable[
        [Callable[[], RasterizedImage], Mapping[str, Any], Mapping[str, Any]],
        _Prediction,
    ],
    encode_window: Callable[
        [
            _Prediction,
            Mapping[str, Any],
            list[str],
            Mapping[str, int],
            Mapping[str, int],
        ],
        EncodedFeatureStream,
    ],
    source_bindings=(),
) -> dict[str, Any]:
    """Execute one complete CoreHalo unit and publish its owned feature state.

    ``predict_window`` controls when to read pixels and releases image/field
    buffers before ``encode_window`` converts its compact decoded output. Both
    callbacks are synchronous. Mounted IO and publication remain in this process.
    """

    context = _context(request_value, input_id, model_run_limits, activation)
    window = _window(context, unit_id)
    binding = unit_source_binding(source_bindings, input_id, unit_id)
    started = perf_counter()
    prepared = _prepare_raster(
        context, activation, source_store, source_files, binding, window["input_rect"]
    )
    _LOGGER.info("stage=source_prepare elapsed_seconds=%.6f", perf_counter() - started)
    processing_limits = processing_result_limits(
        context.request["expected_backend"]["architecture"]
    )
    processing_limits["max_transferred_bytes"] = min(
        processing_limits["max_transferred_bytes"],
        model_run_limits["max_recovery_state_bytes_per_unit"],
    )
    raster = context.model_input["raster_input"]["frame"]["transform"]["raster"]
    try:
        if _field_instance_window_misses_mask(context, window["input_rect"]):
            encoded = encode_core_halo_collection(
                build_feature_collection((), context.classes),
                window,
                context.classes,
                context.output_type,
                raster,
                processing_limits,
            )
        else:
            artifact = model_input_source_artifact(
                context.model_input["raster_input"], binding
            )

            def read_window_image():
                started = perf_counter()
                image = _read(prepared, window["input_rect"], artifact["artifact_id"])
                _LOGGER.info(
                    "stage=source_read_and_rasterize elapsed_seconds=%.6f",
                    perf_counter() - started,
                )
                return image

            encoded = encode_window(
                predict_window(read_window_image, context.plan, window),
                window,
                context.classes,
                raster,
                processing_limits,
            )
        started = perf_counter()
        state = publish_core_halo_unit_state(encoded, files)
        _LOGGER.info(
            "stage=unit_state_publish elapsed_seconds=%.6f",
            perf_counter() - started,
        )
    except ResultLimitExceeded as error:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            str(error),
        ) from error
    except InsufficientCoreHaloContext as error:
        raise GatewayFailure(
            GatewayErrorCode.INSUFFICIENT_CONTEXT,
            str(error),
        ) from error
    except GatewayFailure:
        raise
    except Exception as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "processing-image prediction or output conversion failed",
            resource_id=_resource_id(context.request),
        ) from error

    return {
        "input_id": context.input_id,
        "unit_id": window["unit_id"],
        "reusable_state": state,
        "execution_identity_digest": model_run_execution_identity_digest(
            context.request,
            context.input_id,
            window["unit_id"],
            model_run_limits,
            source_bindings,
        ),
    }


def _field_instance_window_misses_mask(context, input_rect) -> bool:
    """Skip only FIELD instances whose entire supplied image misses the target.

    Valid instance outlines cannot extend outside their supplied input rectangle.
    If that rectangle (including halo) is disjoint from the target mask, none can
    survive target clipping. Closed rectangles deliberately retain boundary
    contacts. Parent admission and dense-region assembly have different semantics.
    """

    if context.model_input["output"]["scope"]["type"] != "field" or (
        context.output_type
        not in {"instance_segmentation", "instance_segmentation_and_classification"}
    ):
        return False
    mask = context.model_input["raster_input"]["frame"]["inclusion_mask_raster"]
    x, y = input_rect["x"], input_rect["y"]
    rectangle = box(x, y, x + input_rect["width"], y + input_rect["height"])
    return rectangle.disjoint(shape(mask))


def finalize_core_halo_input(
    request_value: Any,
    input_id: int,
    model_run_limits: Any,
    unit_publications: Sequence[Any],
    activation: Mapping[str, Any],
    files: ArtifactFileStore,
    source_bindings=(),
) -> dict[str, Any]:
    """Merge exact completed units and publish one complete biological result."""

    context = _context(request_value, input_id, model_run_limits, activation)
    budget = context.request["result_budget"]
    try:
        started = perf_counter()
        raster = context.model_input["raster_input"]["frame"]["transform"]["raster"]
        encoded = merge_core_halo_unit_states(
            unit_publications,
            request=context.request,
            input_id=context.input_id,
            plan=context.plan,
            model_run_limits=model_run_limits,
            declared_classes=context.classes,
            output_type=context.output_type,
            result_budget=budget,
            target_raster=raster,
            files=files,
            source_bindings=source_bindings,
        )
        _LOGGER.info(
            "stage=unit_state_read_merge_and_encode elapsed_seconds=%.6f",
            perf_counter() - started,
        )
    except ResultLimitExceeded as error:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            str(error),
        ) from error
    except (ArtifactBytesCorrupt, ArtifactBytesMissing, OSError) as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "completed CoreHalo unit bytes are missing or corrupt",
            resource_id=_resource_id(context.request),
        ) from error
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "completed CoreHalo unit state is invalid",
            resource_id=_resource_id(context.request),
        ) from error

    started = perf_counter()
    artifact = files.publish_artifact_bytes(
        encoded.data,
        media_type=FEATURE_STREAM_MEDIA_TYPE,
        content_encoding=FEATURE_STREAM_CONTENT_ENCODING,
    )
    _LOGGER.info("stage=result_publish elapsed_seconds=%.6f", perf_counter() - started)
    return {
        "input": {
            "input_id": context.input_id,
            "input_identity": build_backend_input_identity(
                context.model_input["raster_input"],
                input_source_bindings(source_bindings, input_id),
            ),
            "execution_plan": context.plan,
            "feature_stream_artifact": artifact,
            "feature_count": encoded.feature_count,
            "vertex_count": encoded.vertex_count,
        },
        "decoded_bytes": encoded.decoded_bytes,
    }


def _context(
    request_value: Any,
    input_id: int,
    model_run_limits: Any,
    activation: Mapping[str, Any],
) -> _CoreHaloContext:
    request = validate_model_run_resource_limits(request_value, model_run_limits)
    require_activation_identity(request["expected_backend"], activation)
    if isinstance(input_id, bool) or not isinstance(input_id, int) or input_id < 0:
        raise ValueError("input_id must be a nonnegative integer")
    inputs = {item["input_id"]: item for item in request["inputs"]}
    try:
        model_input = inputs[input_id]
    except KeyError as error:
        raise ValueError("durable worker input_id is absent from request") from error
    plan = build_expected_model_execution_plans(request, model_run_limits)[input_id]
    if plan["type"] != "core-halo":
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            "the selected durable input is not a CoreHalo plan",
            resource_id=_resource_id(request),
        )
    output_type = activation["model"]["output"]["type"]
    if output_type not in {
        "instance_segmentation",
        "instance_segmentation_and_classification",
        "dense_classification",
    }:
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            "large-region execution does not support this model output type",
            resource_id=_resource_id(request),
        )
    return _CoreHaloContext(
        request=request,
        model_input=model_input,
        plan=plan,
        classes=model_declared_classes(activation["model"]["output"]),
        output_type=output_type,
        input_id=input_id,
    )


def _window(context: _CoreHaloContext, unit_id: int) -> dict[str, Any]:
    if isinstance(unit_id, bool) or not isinstance(unit_id, int) or unit_id < 0:
        raise ValueError("unit_id must be a nonnegative integer")
    try:
        return context.plan["windows"][unit_id]
    except IndexError as error:
        raise ValueError("durable worker unit_id is absent from plan") from error


def _prepare_raster(
    context: _CoreHaloContext,
    activation: Mapping[str, Any],
    source_store: SourceArtifactStore,
    files: SourceFileStore,
    source_binding=None,
    output_rect=None,
) -> PartitionedRaster:
    raster_input = context.model_input["raster_input"]
    artifact = model_input_source_artifact(raster_input, source_binding)
    try:
        resolved = source_store.resolve_sealed_reference(artifact)
        return prepare_partitioned_model_raster(
            raster_input,
            resolved,
            files,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            activation["capabilities"]["source_containers"],
            source_binding=source_binding,
            output_rect=output_rect,
        )
    except GatewayFailure:
        raise
    except (ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "sealed source bytes are missing or corrupt; recreate the artifact",
            resource_id=artifact["artifact_id"],
        ) from error
    except OSError as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "sealed source bytes could not be read",
            resource_id=artifact["artifact_id"],
        ) from error
    except InsufficientSourceContext as error:
        raise GatewayFailure(
            GatewayErrorCode.INSUFFICIENT_CONTEXT,
            str(error),
        ) from error
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error


def _read(
    prepared: PartitionedRaster,
    rectangle: Mapping[str, int],
    artifact_id: str,
) -> RasterizedImage:
    try:
        return prepared.read(rectangle)
    except GatewayFailure:
        raise
    except (ArtifactBytesCorrupt, ArtifactBytesMissing, OSError) as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "sealed source bytes could not be read",
            resource_id=artifact_id,
        ) from error
    except InsufficientSourceContext as error:
        raise GatewayFailure(
            GatewayErrorCode.INSUFFICIENT_CONTEXT, str(error)
        ) from error
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error


def _resource_id(request: Mapping[str, Any]) -> str:
    return f"{request['producer']['module_ref']}/{request['producer']['step_id']}"
