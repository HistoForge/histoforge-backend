"""Finite-call admission, runner execution, and result-response encoding.

This is one bounded route family, not a durable workflow.  The gateway checks
the current module and source before dispatch; the warm runner independently
rechecks the same identities before producing one typed result or failure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
    require_nonnegative_int32,
)
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    build_gateway_error,
    validate_gateway_error,
)
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.inference.contract import (
    validate_backend_input_identity,
    validate_client_result_budget,
)
from histoforge.cloud.inference.execution import (
    declared_classes,
    execute_model_input,
    model_declared_classes,
    require_activation_identity,
)
from histoforge.cloud.inference.finite.contract import (
    finite_input_identity,
    require_backend_call_id,
    validate_finite_route_request,
)
from histoforge.cloud.inference.identity import (
    validate_backend_model_identity,
    validate_model_step_identity,
)
from histoforge.cloud.inference.route_activation import (
    resolve_requested_inference_activation,
)
from histoforge.cloud.protocol_versions import FINITE_INFERENCE_SCHEMA_VERSION
from histoforge.cloud.rasterization.inputs import (
    geometry_vertex_count,
    model_input_source_artifact,
    validate_raster_descriptor,
)
from histoforge.cloud.rasterization.model_input import prepare_partitioned_model_raster
from histoforge.cloud.runner_contract import validate_runner_capabilities
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.store import SourceArtifactStore

_SUCCESS_FIELDS = {
    "type",
    "action_id",
    "output_generation_id",
    "input_id",
    "producer",
    "backend",
    "input_identity",
    "raster",
    "result_budget",
    "declared_classes",
    "geojson",
}
_FAILURE_FIELDS = {"type", "error"}
_MAX_CANONICAL_CALL_ID = "\x01" * 128


def validate_finite_admission(
    value: Any,
    route_module_ref: Any,
    route_model_id: Any,
    backend: Any,
    capabilities: Any,
    source_store: SourceArtifactStore,
    files: SourceFileStore,
) -> dict[str, Any]:
    """Apply exact current-model, capability, source, and raster admission."""

    try:
        request = validate_finite_route_request(value, route_module_ref, route_model_id)
        current_backend = validate_backend_model_identity(backend)
        runner = validate_runner_capabilities(capabilities)
        resolved_activation = resolve_requested_inference_activation(
            request["expected_backend"],
            {"backend": current_backend, "capabilities": runner},
        )
        require_activation_identity(
            request["expected_backend"],
            resolved_activation,
        )
        _require_finite_limits(request, runner)
        artifact = model_input_source_artifact(
            request["raster_input"], request.get("source_binding")
        )
        resolved_source = source_store.resolve_sealed_reference(artifact)
        prepare_partitioned_model_raster(
            request["raster_input"],
            resolved_source,
            files,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            runner["source_containers"],
            source_binding=request.get("source_binding"),
        )
    except (ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "sealed source bytes are missing or corrupt; recreate the artifact",
            resource_id=artifact["artifact_id"],
        ) from error
    except GatewayFailure:
        raise
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            str(error),
        ) from error
    return request


def execute_finite_request(
    request_value: Any,
    activation: Mapping[str, Any],
    source_store: SourceArtifactStore,
    files: SourceFileStore,
    infer: Callable[[Any, Mapping[str, int]], dict[str, Any]],
) -> dict[str, Any]:
    """Run one validated finite input and always return a typed worker outcome."""

    backend = activation["backend"]
    route_ref = backend["module"]["module_ref"]
    route_model_id = backend["model_id"]
    try:
        request = validate_finite_route_request(
            request_value, route_ref, route_model_id
        )
        require_activation_identity(request["expected_backend"], activation)
        _require_finite_limits(request, activation["capabilities"])
    except GatewayFailure as error:
        return _failure_outcome(error.error)
    except ValueError as error:
        return _failure_outcome(
            build_gateway_error(GatewayErrorCode.VALIDATION_ERROR, str(error))
        )

    try:
        features = execute_model_input(
            request["raster_input"],
            request["result_budget"],
            activation,
            source_store,
            files,
            infer,
            source_binding=request.get("source_binding"),
        )
    except GatewayFailure as error:
        return _failure_outcome(error.error)
    try:
        return _success_outcome(
            request,
            backend,
            model_declared_classes(activation["model"]["output"]),
            features,
        )
    except GatewayFailure as error:
        return _failure_outcome(error.error)
    except ValueError:
        return _failure_outcome(
            build_gateway_error(
                GatewayErrorCode.RUNNER_FAILED,
                "model output violates the finite result envelope",
                resource_id=f"{route_ref}/{route_model_id}",
            )
        )


def encode_finite_result(value: Any, backend_call_id: Any) -> bytes:
    """Encode an addressed response from the worker's validated features.

    The worker owns feature validation. The gateway checks its result envelope,
    supplies the call identity, and checks the exact bytes it sends to the client.
    QuPath independently validates the received features before using them.
    """

    if not isinstance(value, Mapping):
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "runner returned an invalid result envelope",
        )
    if value.get("type") == "failure":
        if set(value) != _FAILURE_FIELDS:
            raise GatewayFailure(
                GatewayErrorCode.RUNNER_FAILED,
                "runner returned an invalid failure envelope",
            )
        try:
            error = validate_gateway_error(value["error"])
        except ValueError as validation_error:
            raise GatewayFailure(
                GatewayErrorCode.RUNNER_FAILED,
                "runner returned an invalid failure envelope",
            ) from validation_error
        raise _gateway_failure(error)
    success = _validate_result_envelope(value)
    call_id = require_backend_call_id(backend_call_id)
    try:
        encoded = canonical_bytes(_wire_result(success, call_id))
    except (ValueError, TypeError) as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "runner returned an unencodable result",
        ) from error
    _require_result_size(len(encoded), success["result_budget"])
    return encoded


def _success_outcome(
    request: Mapping[str, Any],
    backend: Mapping[str, Any],
    classes: list[str],
    features: Any,
) -> dict[str, Any]:
    raster = request["raster_input"]["frame"]["transform"]["raster"]
    outcome = canonical_tree(
        {
            "type": "result",
            "action_id": request["action_id"],
            "output_generation_id": request["output_generation_id"],
            "input_id": request["input_id"],
            "producer": request["producer"],
            "backend": backend,
            "input_identity": finite_input_identity(request),
            "raster": raster,
            "result_budget": request["result_budget"],
            "declared_classes": classes,
            "geojson": features,
        }
    )
    _require_result_size(
        len(canonical_bytes(_wire_result(outcome, _MAX_CANONICAL_CALL_ID))),
        outcome["result_budget"],
    )
    return outcome


def _validate_result_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _SUCCESS_FIELDS or value.get("type") != "result":
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "runner returned an invalid result envelope",
        )
    try:
        producer = validate_model_step_identity(value["producer"])
        backend = validate_backend_model_identity(value["backend"])
        if producer["module_ref"] != backend["module"]["module_ref"]:
            raise ValueError("result producer module does not match backend")
        if producer["step_id"] != backend["model_id"]:
            raise ValueError("result producer step does not match backend")
        raster = validate_raster_descriptor(value["raster"])
        budget = validate_client_result_budget(value["result_budget"])
        classes = declared_classes(value["declared_classes"])
        features = value["geojson"]
        if (
            not isinstance(features, dict)
            or set(features) != {"type", "features"}
            or features["type"] != "FeatureCollection"
            or not isinstance(features["features"], list)
        ):
            raise ValueError("result must contain a feature collection")
        return {
            "type": "result",
            "action_id": require_canonical_uuid(value["action_id"], "action_id"),
            "output_generation_id": require_canonical_uuid(
                value["output_generation_id"], "output_generation_id"
            ),
            "input_id": require_nonnegative_int32(value["input_id"], "input_id"),
            "producer": producer,
            "backend": backend,
            "input_identity": validate_backend_input_identity(value["input_identity"]),
            "raster": raster,
            "result_budget": budget,
            "declared_classes": classes,
            "geojson": features,
        }
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "runner returned an invalid result envelope",
        ) from error


def _wire_result(success: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "meta": {
            "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
            "action_id": success["action_id"],
            "output_generation_id": success["output_generation_id"],
            "backend_call_id": call_id,
            "input_id": success["input_id"],
            "producer": success["producer"],
            "backend": success["backend"],
            "input_identity": success["input_identity"],
            "coordinate_space": "RASTER_LOCAL",
        },
        "geojson": success["geojson"],
    }


def _require_finite_limits(request: Mapping[str, Any], capabilities: Any) -> None:
    runner = validate_runner_capabilities(capabilities)
    finite = runner["finite"]
    if len(canonical_bytes(request)) > finite["max_request_canonical_bytes"]:
        raise GatewayFailure(
            GatewayErrorCode.PAYLOAD_TOO_LARGE,
            "finite request exceeds the selected runner's canonical byte limit",
        )
    raster = request["raster_input"]["frame"]["transform"]["raster"]
    raster_limits = finite["target_raster"]
    if (
        raster["width"] > raster_limits["max_width"]
        or raster["height"] > raster_limits["max_height"]
        or raster["width"] * raster["height"] > raster_limits["max_pixels"]
    ):
        raise GatewayFailure(
            GatewayErrorCode.PAYLOAD_TOO_LARGE,
            "target raster exceeds the selected runner's finite limits",
        )
    mask = request["raster_input"]["frame"]["inclusion_mask_raster"]
    if geometry_vertex_count(mask) > finite["max_inclusion_mask_vertices"]:
        raise GatewayFailure(
            GatewayErrorCode.PAYLOAD_TOO_LARGE,
            "inclusion mask exceeds the selected runner's vertex limit",
        )
    if (
        model_input_source_artifact(
            request["raster_input"], request.get("source_binding")
        )["container"]
        not in runner["source_containers"]
    ):
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            "source container is not supported by the selected runner",
        )
    for field, value in request["result_budget"].items():
        if value > finite["result"][field]:
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                f"client result budget exceeds runner limit {field}",
            )


def _require_result_size(size: int, budget: Mapping[str, int]) -> None:
    if size > budget["max_transferred_bytes"] or size > budget["max_decoded_bytes"]:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            "inference result exceeds transferred or decoded byte budget",
        )


def _failure_outcome(error: Any) -> dict[str, Any]:
    return {"type": "failure", "error": validate_gateway_error(error)}


def _gateway_failure(error: Mapping[str, Any]) -> GatewayFailure:
    return GatewayFailure(
        error["code"],
        error["message"],
        resource_id=error.get("resource_id"),
        retry_not_before=error.get("retry_not_before"),
    )
