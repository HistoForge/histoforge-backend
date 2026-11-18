"""Durable-run admission and one complete finite-plan input execution."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from histoforge.canonical_json import require_nonnegative_int32
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
)
from histoforge.cloud.inference.execution import (
    execute_model_input,
    model_declared_classes,
    require_activation_identity,
)
from histoforge.cloud.inference.feature_stream import (
    collection_budget_for_stream,
    encode_feature_stream,
)
from histoforge.cloud.inference.route_activation import (
    resolve_requested_inference_activation,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
)
from histoforge.cloud.inference.runs.request import (
    validate_model_run_admission,
    validate_model_run_resource_limits,
    validate_model_run_route_request,
)
from histoforge.cloud.inference.runs.source_bindings import (
    input_source_bindings,
    unit_source_binding,
)
from histoforge.cloud.runner_contract import validate_runner_capabilities
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.store import SourceArtifactStore

_LOGGER = logging.getLogger(__name__)


def validate_durable_admission(
    value: Any,
    route_module_ref: str,
    route_model_id: str,
    activation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Validate one request against the exact installed durable capability.

    Returns the normalized request, durable run limits, and terminal retention
    because those values are the complete source-bound reservation input.
    """

    try:
        request = validate_model_run_route_request(
            value, route_module_ref, route_model_id
        )
        activation = resolve_requested_inference_activation(
            request["expected_backend"], activation
        )
        capabilities = validate_runner_capabilities(activation["capabilities"])
        if "DURABLE_MODEL_RUN" not in capabilities["modes"]:
            raise GatewayFailure(
                GatewayErrorCode.UNSUPPORTED_CAPABILITY,
                "the selected runner does not implement durable model runs",
                resource_id=f"{route_module_ref}/{route_model_id}",
            )
        require_activation_identity(request["expected_backend"], activation)
        durable = capabilities["durable_run"]
        admitted = validate_model_run_admission(
            request,
            route_module_ref,
            route_model_id,
            activation["backend"],
            capabilities["finite"]["target_raster"],
            activation.get("large_region_profile"),
            durable["run"],
        )
        unsupported = [
            item["raster_input"]["source_view"]["artifact"]["container"]
            for item in admitted["inputs"]
            if "artifact" in item["raster_input"]["source_view"]
            and item["raster_input"]["source_view"]["artifact"]["container"]
            not in capabilities["source_containers"]
        ]
        if unsupported:
            raise GatewayFailure(
                GatewayErrorCode.UNSUPPORTED_CAPABILITY,
                "source container is not supported by the selected runner",
                resource_id=f"{route_module_ref}/{route_model_id}",
            )
        return admitted, durable["run"], durable["terminal_retention_seconds"]
    except GatewayFailure:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            str(error),
        ) from error


def execute_durable_finite_input(
    request_value: Any,
    input_id: int,
    model_run_limits: Any,
    activation: Mapping[str, Any],
    source_store: SourceArtifactStore,
    source_files: SourceFileStore,
    files: ArtifactFileStore,
    infer: Callable[[Any, Mapping[str, int]], dict[str, Any]],
    source_bindings=(),
) -> dict[str, Any]:
    """Execute and publish one finite-plan input through shared model science."""

    request = validate_model_run_resource_limits(request_value, model_run_limits)
    require_activation_identity(request["expected_backend"], activation)
    selected_input_id = require_nonnegative_int32(input_id, "input_id")
    inputs = {item["input_id"]: item for item in request["inputs"]}
    try:
        model_input = inputs[selected_input_id]
    except KeyError as error:
        raise ValueError("durable worker input_id is absent from request") from error
    plan = build_expected_model_execution_plans(request, model_run_limits)[
        selected_input_id
    ]
    if plan["type"] != "finite":
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            "this runner build has not enabled the requested oversized plan",
            resource_id=(
                f"{request['producer']['module_ref']}/{request['producer']['step_id']}"
            ),
        )

    # A durable finite plan still predicts one small image. A larger whole-slide
    # result allowance must not enlarge its existing finite candidate budget.
    budget = {
        field: min(value, activation["capabilities"]["finite"]["result"][field])
        for field, value in request["result_budget"].items()
    }
    features = execute_model_input(
        model_input["raster_input"],
        budget,
        activation,
        source_store,
        source_files,
        lambda raster, budget: infer(raster, collection_budget_for_stream(budget)),
        source_binding=unit_source_binding(source_bindings, input_id, 0),
    )
    started = perf_counter()
    classes = model_declared_classes(activation["model"]["output"])
    stream = encode_feature_stream(features, classes)
    if len(stream.data) > budget["max_transferred_bytes"]:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            "feature stream exceeds max_transferred_bytes",
        )
    if stream.decoded_bytes > budget["max_decoded_bytes"]:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            "feature stream exceeds max_decoded_bytes",
        )
    artifact = files.publish_artifact_bytes(
        stream.data,
        media_type="application/geo+json-seq",
        content_encoding="gzip",
    )
    _LOGGER.info(
        "stage=result_encode_and_publish elapsed_seconds=%.6f",
        perf_counter() - started,
    )
    result = {
        "input_id": selected_input_id,
        "input_identity": build_backend_input_identity(
            model_input["raster_input"],
            input_source_bindings(source_bindings, input_id),
        ),
        "execution_plan": plan,
        "feature_stream_artifact": artifact,
        "feature_count": stream.feature_count,
        "vertex_count": stream.vertex_count,
    }
    return {
        "input": result,
        "decoded_bytes": stream.decoded_bytes,
    }
