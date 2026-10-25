"""Pure durable model-run request and acceptance contracts.

The plugin constructs a request for one exact backend model. The backend
validates its route, logical outputs, raster identities, homogeneous plan
family, and optional resume source before reserving the supplied run ID. This
module performs no persistence, dispatch, source IO, or result publication.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_sha256,
)
from histoforge.cloud.execution_limits import (
    fits_target_raster,
    validate_model_run_limits,
)
from histoforge.cloud.inference.contract import (
    validate_client_result_budget,
)
from histoforge.cloud.inference.identity import (
    validate_backend_model_identity,
    validate_model_step_identity,
)
from histoforge.cloud.inference.plans import (
    model_execution_plan_unit_count,
    validate_large_region_profile,
    validate_processing_raster_limits,
)
from histoforge.cloud.modules.schema import validate_model_id
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION
from histoforge.cloud.rasterization.inputs import (
    geometry_vertex_count,
    validate_model_raster_input,
)
from histoforge.execution_scope import validate_output_key
from histoforge.module_ref import ModuleRef

_INPUT_FIELDS = {"input_id", "output_generation_id", "output", "raster_input"}
_RESUME_FIELDS = {"previous_run_id", "previous_manifest_digest"}
_REQUEST_REQUIRED_FIELDS = {
    "schema_version",
    "action_id",
    "model_run_id",
    "producer",
    "expected_backend",
    "inputs",
    "result_budget",
}
_REQUEST_OPTIONAL_FIELDS = {"expected_profile", "resume_from", "halo_px_override"}
_ACCEPTED_FIELDS = {
    "schema_version",
    "action_id",
    "model_run_id",
    "request_digest",
}


def validate_model_run_input(value: Any) -> dict[str, Any]:
    """Validate one logical output and its exact source-to-raster request."""

    item = _require_exact_object(value, _INPUT_FIELDS, "model run input")
    output = validate_output_key(item["output"])
    raster_input = validate_model_raster_input(item["raster_input"])
    if output["scope"]["region"] != raster_input["frame"]["region"]:
        raise ValueError("output scope region does not match raster input region")
    return canonical_tree(
        {
            "input_id": require_nonnegative_int32(item["input_id"], "input_id"),
            "output_generation_id": require_canonical_uuid(
                item["output_generation_id"], "output_generation_id"
            ),
            "output": output,
            "raster_input": raster_input,
        }
    )


def validate_model_run_resume_source(value: Any) -> dict[str, str]:
    """Validate the exact prior terminal manifest proposed for explicit reuse."""

    source = _require_exact_object(value, _RESUME_FIELDS, "model run resume source")
    return {
        "previous_run_id": require_canonical_uuid(
            source["previous_run_id"], "previous_run_id"
        ),
        "previous_manifest_digest": require_sha256(
            source["previous_manifest_digest"], "previous_manifest_digest"
        ),
    }


def validate_model_run_request(value: Any) -> dict[str, Any]:
    """Validate one closed request and its context-free identity agreement."""

    request = _require_object(value, "model run request")
    _require_fields(
        request,
        _REQUEST_REQUIRED_FIELDS,
        _REQUEST_OPTIONAL_FIELDS,
        "model run request",
    )
    _require_current_schema(request["schema_version"])
    producer = validate_model_step_identity(request["producer"])
    backend = validate_backend_model_identity(request["expected_backend"])
    if producer["module_ref"] != backend["module"]["module_ref"]:
        raise ValueError("producer module_ref does not match expected backend module")
    if producer["step_id"] != backend["model_id"]:
        raise ValueError("producer step_id does not match expected backend model_id")

    inputs = _normalize_inputs(request["inputs"], producer, backend)
    model_run_id = require_canonical_uuid(request["model_run_id"], "model_run_id")
    result: dict[str, Any] = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": require_canonical_uuid(request["action_id"], "action_id"),
        "model_run_id": model_run_id,
        "producer": producer,
        "expected_backend": backend,
        "inputs": inputs,
        "result_budget": validate_client_result_budget(request["result_budget"]),
    }
    if "expected_profile" in request:
        result["expected_profile"] = validate_large_region_profile(
            request["expected_profile"]
        )
    if result.get("expected_profile", {}).get("type") == "global-decode" and any(
        "capture" in item["raster_input"]["source_view"] for item in inputs
    ):
        raise ValueError(
            "global-decode plans do not support incremental image delivery"
        )
    if "halo_px_override" in request:
        result["halo_px_override"] = require_nonnegative_int32(
            request["halo_px_override"], "halo_px_override"
        )
        if result.get("expected_profile", {}).get("type") != "core-halo":
            raise ValueError("halo_px_override requires a core-halo profile")
    if "resume_from" in request:
        resume_from = validate_model_run_resume_source(request["resume_from"])
        if resume_from["previous_run_id"] == model_run_id:
            raise ValueError("replacement run must use a fresh model_run_id")
        result["resume_from"] = resume_from
    return canonical_tree(result)


def validate_model_run_route_request(
    value: Any, route_module_ref: Any, route_model_id: Any
) -> dict[str, Any]:
    """Bind a request to the exact canonical module/model route receiving it."""

    request = validate_model_run_request(value)
    module_ref = _module_ref(route_module_ref, "route module_ref")
    model_id = validate_model_id(route_model_id, "route model_id")
    if request["producer"]["module_ref"] != module_ref:
        raise ValueError("request producer does not match route module_ref")
    if request["producer"]["step_id"] != model_id:
        raise ValueError("request producer does not match route model_id")
    return request


def validate_model_run_admission(
    value: Any,
    route_module_ref: Any,
    route_model_id: Any,
    actual_backend: Any,
    finite_target_limits: Any,
    advertised_profile: Any | None,
    model_run_limits: Any,
) -> dict[str, Any]:
    """Validate every codec-owned request fence before durable admission."""

    request = validate_model_run_route_request(value, route_module_ref, route_model_id)
    if request["expected_backend"] != validate_backend_model_identity(actual_backend):
        raise ValueError(
            "expected_backend does not match the deployed backend identity"
        )
    request = validate_model_run_plan_family(
        request, finite_target_limits, advertised_profile
    )
    return validate_model_run_resource_limits(request, model_run_limits)


def validate_model_run_plan_family(
    value: Any, finite_target_limits: Any, advertised_profile: Any | None
) -> dict[str, Any]:
    """Bind profile presence to the runner's exact finite-raster predicate."""

    request = validate_model_run_request(value)
    fits = [
        fits_target_raster(
            item["raster_input"]["frame"]["transform"]["raster"],
            finite_target_limits,
        )
        for item in request["inputs"]
    ]
    if any(fits) and not all(fits):
        raise ValueError("model run inputs must use one homogeneous plan family")
    expected_profile = request.get("expected_profile")
    if all(fits):
        if expected_profile is not None:
            raise ValueError("finite-plan request must omit expected_profile")
        return request
    if expected_profile is None:
        raise ValueError("oversized request must carry expected_profile")
    if advertised_profile is None:
        raise ValueError("runner does not advertise a large-region profile")
    if expected_profile != validate_large_region_profile(advertised_profile):
        raise ValueError("expected_profile does not match the advertised profile")
    return request


def validate_model_run_resource_limits(
    value: Any, model_run_limits: Any
) -> dict[str, Any]:
    """Enforce the durable-run admission caps that do not select plan family."""

    request = validate_model_run_request(value)
    limits = validate_model_run_limits(model_run_limits)
    if len(canonical_bytes(request)) > limits["max_request_canonical_bytes"]:
        raise ValueError("model run request exceeds max_request_canonical_bytes")
    if len(request["inputs"]) > limits["max_inputs"]:
        raise ValueError("model run request exceeds max_inputs")

    aggregate_pixels = 0
    aggregate_mask_vertices = 0
    aggregate_units = 0
    for item in request["inputs"]:
        raster = item["raster_input"]["frame"]["transform"]["raster"]
        if not fits_target_raster(raster, limits["target_raster_per_input"]):
            raise ValueError("model run input exceeds target_raster_per_input")
        aggregate_pixels += raster["width"] * raster["height"]
        aggregate_mask_vertices += geometry_vertex_count(
            item["raster_input"]["frame"]["inclusion_mask_raster"]
        )
        aggregate_units += model_execution_plan_unit_count(
            request.get("expected_profile"), raster
        )

    if aggregate_pixels > limits["max_aggregate_target_raster_pixels"]:
        raise ValueError("model run request exceeds aggregate target-raster pixels")
    if aggregate_mask_vertices > limits["max_aggregate_inclusion_mask_vertices"]:
        raise ValueError("model run request exceeds aggregate inclusion-mask vertices")
    if aggregate_units > limits["max_units"]:
        raise ValueError("model run request exceeds max_units")
    for field, requested in request["result_budget"].items():
        if requested > limits["result"][field]:
            raise ValueError(f"model run result_budget.{field} exceeds runner limit")
    # The admitted unit-count limit bounds this traversal before source access.
    for item in request["inputs"]:
        validate_processing_raster_limits(
            request.get("expected_profile"),
            item["raster_input"]["frame"]["transform"]["raster"],
            limits["processing_raster"],
            halo_px_override=request.get("halo_px_override"),
        )
    return request


def model_run_request_digest(value: Any) -> str:
    """Return the idempotency digest of the exact accepted canonical request."""

    return canonical_digest(validate_model_run_request(value))


def validate_model_run_accepted(value: Any) -> dict[str, Any]:
    """Validate one durable reservation acknowledgement without request context."""

    accepted = _require_exact_object(value, _ACCEPTED_FIELDS, "model run accepted")
    _require_current_schema(accepted["schema_version"])
    return {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": require_canonical_uuid(accepted["action_id"], "action_id"),
        "model_run_id": require_canonical_uuid(
            accepted["model_run_id"], "model_run_id"
        ),
        "request_digest": require_sha256(accepted["request_digest"], "request_digest"),
    }


def validate_model_run_acceptance(request: Any, accepted: Any) -> dict[str, Any]:
    """Require an acknowledgement to echo and digest the submitted request."""

    submitted = validate_model_run_request(request)
    result = validate_model_run_accepted(accepted)
    expected = {
        "action_id": submitted["action_id"],
        "model_run_id": submitted["model_run_id"],
        "request_digest": canonical_digest(submitted),
    }
    for field, expected_value in expected.items():
        if result[field] != expected_value:
            raise ValueError(f"accepted {field} does not match model run request")
    return result


def _normalize_inputs(
    value: Any, producer: Mapping[str, str], backend: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("inputs must be a nonempty array")
    inputs = [validate_model_run_input(item) for item in value]
    input_ids = [item["input_id"] for item in inputs]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("inputs contain duplicate input_id")
    if input_ids != sorted(input_ids):
        raise ValueError("inputs must use strictly increasing input_id order")
    if any(item["output"]["producer"] != producer for item in inputs):
        raise ValueError("input output producer does not match request producer")
    if any(
        item["raster_input"]["rasterizer"] != backend["rasterizer"] for item in inputs
    ):
        raise ValueError("input rasterizer does not match expected backend rasterizer")
    if len({item["output_generation_id"] for item in inputs}) != len(inputs):
        raise ValueError("inputs contain duplicate output_generation_id")
    output_keys = {canonical_bytes(item["output"]) for item in inputs}
    if len(output_keys) != len(inputs):
        raise ValueError("inputs contain duplicate output")
    return inputs


def _module_ref(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    try:
        return str(ModuleRef.parse(value))
    except ValueError as exc:
        raise ValueError(f"{path} must be a canonical owner/name module ref") from exc


def _require_current_schema(value: Any) -> None:
    if require_int32(value, "schema_version") != MODEL_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported model run schema_version")


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    result = _require_object(value, path)
    actual = set(result)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return result


def _require_fields(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    path: str,
) -> None:
    actual = set(value)
    if not required <= actual or not actual <= required | optional:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - required - optional)}"
        )
