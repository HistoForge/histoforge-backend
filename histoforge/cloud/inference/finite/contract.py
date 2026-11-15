"""Pure finite-inference request and response contracts.

This module binds one exact model/raster request to accepted, pending,
cancelled, and terminal result envelopes. It also owns the pure query-component
encoding used to address an opaque call ID. It performs no HTTP IO, Modal
dispatch, polling, source IO, rasterization, or QuPath mutation.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any
from urllib.parse import quote_from_bytes

from histoforge.canonical_json import (
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonblank_string,
    require_nonnegative_int32,
)
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
    validate_backend_input_identity,
    validate_client_result_budget,
    validate_inference_features,
)
from histoforge.cloud.inference.identity import (
    validate_backend_model_identity,
    validate_model_step_identity,
)
from histoforge.cloud.modules.schema import validate_model_id
from histoforge.cloud.protocol_versions import FINITE_INFERENCE_SCHEMA_VERSION
from histoforge.cloud.rasterization.inputs import (
    validate_model_raster_input,
)
from histoforge.cloud.sources.capture import validate_captured_source_binding
from histoforge.module_ref import ModuleRef

MAX_BACKEND_CALL_ID_UTF8_BYTES = 128
BACKEND_CALL_QUERY_PARAMETER = "backend_call_id"

_REQUEST_FIELDS = {
    "schema_version",
    "action_id",
    "output_generation_id",
    "input_id",
    "producer",
    "expected_backend",
    "raster_input",
    "result_budget",
}
_ACCEPTED_FIELDS = {
    "schema_version",
    "action_id",
    "output_generation_id",
    "backend_call_id",
}
_CALL_STATUS_FIELDS = {"schema_version", "backend_call_id"}
_RESULT_META_FIELDS = {
    "schema_version",
    "action_id",
    "output_generation_id",
    "backend_call_id",
    "input_id",
    "producer",
    "backend",
    "input_identity",
    "coordinate_space",
}
_RESULT_FIELDS = {"meta", "geojson"}


def validate_finite_inference_request(value: Any) -> dict[str, Any]:
    """Validate one complete finite request and all internal identity agreement."""

    request = _require_object(value, "finite request")
    _require_exact_object(
        request,
        _REQUEST_FIELDS
        | ({"source_binding"} if "source_binding" in request else set()),
        "finite request",
    )
    _require_current_schema(request["schema_version"])
    producer = validate_model_step_identity(request["producer"])
    backend = validate_backend_model_identity(request["expected_backend"])
    raster_input = validate_model_raster_input(request["raster_input"])
    captured = "capture" in raster_input["source_view"]
    if captured != ("source_binding" in request):
        raise ValueError(
            "captured finite inputs require exactly one sealed source_binding"
        )
    binding = (
        validate_captured_source_binding(
            request["source_binding"], raster_input["source_view"]["capture"]
        )
        if captured
        else None
    )
    if raster_input["frame"]["region"]["type"] != "roi":
        raise ValueError("finite request region must be roi")
    if producer["module_ref"] != backend["module"]["module_ref"]:
        raise ValueError("producer module_ref does not match expected backend module")
    if producer["step_id"] != backend["model_id"]:
        raise ValueError("producer step_id does not match expected backend model_id")
    if backend["rasterizer"] != raster_input["rasterizer"]:
        raise ValueError("expected backend rasterizer does not match raster input")
    return canonical_tree(
        {
            "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
            "action_id": require_canonical_uuid(request["action_id"], "action_id"),
            "output_generation_id": require_canonical_uuid(
                request["output_generation_id"], "output_generation_id"
            ),
            "input_id": require_nonnegative_int32(request["input_id"], "input_id"),
            "producer": producer,
            "expected_backend": backend,
            "raster_input": raster_input,
            "result_budget": validate_client_result_budget(request["result_budget"]),
            **({"source_binding": binding} if captured else {}),
        }
    )


def validate_finite_route_request(
    value: Any, route_module_ref: Any, route_model_id: Any
) -> dict[str, Any]:
    """Bind a request to the exact canonical module/model route that received it."""

    request = validate_finite_inference_request(value)
    module_ref = _module_ref(route_module_ref, "route module_ref")
    model_id = validate_model_id(route_model_id, "route model_id")
    if request["producer"]["module_ref"] != module_ref:
        raise ValueError("request producer does not match route module_ref")
    if request["producer"]["step_id"] != model_id:
        raise ValueError("request producer does not match route model_id")
    return request


def validate_inference_accepted(value: Any) -> dict[str, Any]:
    """Validate one finite submission acknowledgement without request context."""

    accepted = _require_exact_object(value, _ACCEPTED_FIELDS, "inference accepted")
    _require_current_schema(accepted["schema_version"])
    return {
        "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
        "action_id": require_canonical_uuid(accepted["action_id"], "action_id"),
        "output_generation_id": require_canonical_uuid(
            accepted["output_generation_id"], "output_generation_id"
        ),
        "backend_call_id": require_backend_call_id(
            accepted["backend_call_id"], "backend_call_id"
        ),
    }


def validate_inference_acceptance(request: Any, accepted: Any) -> dict[str, Any]:
    """Require an acceptance to echo the exact action and output generation."""

    submitted = validate_finite_inference_request(request)
    result = validate_inference_accepted(accepted)
    for field in ("action_id", "output_generation_id"):
        if result[field] != submitted[field]:
            raise ValueError(f"accepted {field} does not match finite request")
    return result


def validate_inference_pending(value: Any) -> dict[str, Any]:
    """Validate one nonterminal poll response."""

    return _validate_call_status(value, "inference pending")


def validate_inference_pending_response(
    value: Any, addressed_backend_call_id: Any
) -> dict[str, Any]:
    """Bind a pending response to the call ID addressed by the poll."""

    return _validate_addressed_call_status(
        validate_inference_pending(value), addressed_backend_call_id
    )


def validate_inference_cancellation_accepted(value: Any) -> dict[str, Any]:
    """Validate one idempotent best-effort cancellation acknowledgement."""

    return _validate_call_status(value, "inference cancellation")


def validate_inference_cancellation_acceptance(
    value: Any, addressed_backend_call_id: Any
) -> dict[str, Any]:
    """Bind a cancellation acknowledgement to the call ID addressed by DELETE."""

    return _validate_addressed_call_status(
        validate_inference_cancellation_accepted(value), addressed_backend_call_id
    )


def validate_backend_result_meta(value: Any) -> dict[str, Any]:
    """Validate result metadata without a retained request or addressed call."""

    meta = _require_exact_object(value, _RESULT_META_FIELDS, "backend result meta")
    _require_current_schema(meta["schema_version"])
    if meta["coordinate_space"] != "RASTER_LOCAL":
        raise ValueError("coordinate_space must be RASTER_LOCAL")
    return canonical_tree(
        {
            "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
            "action_id": require_canonical_uuid(meta["action_id"], "action_id"),
            "output_generation_id": require_canonical_uuid(
                meta["output_generation_id"], "output_generation_id"
            ),
            "backend_call_id": require_backend_call_id(
                meta["backend_call_id"], "backend_call_id"
            ),
            "input_id": require_nonnegative_int32(meta["input_id"], "input_id"),
            "producer": validate_model_step_identity(meta["producer"]),
            "backend": validate_backend_model_identity(meta["backend"]),
            "input_identity": validate_backend_input_identity(meta["input_identity"]),
            "coordinate_space": "RASTER_LOCAL",
        }
    )


def validate_inference_result(
    value: Any,
    request: Any,
    backend_call_id: Any,
    declared_classes: Collection[str],
) -> dict[str, Any]:
    """Bind one parsed deterministic result to its request and addressed call.

    The transport owns transferred/decoded byte counting while it reads the
    response. This parsed-tree boundary rechecks record, feature, and vertex
    ceilings before candidates can be used.
    """

    result = _require_exact_object(value, _RESULT_FIELDS, "inference result")
    submitted = validate_finite_inference_request(request)
    call_id = require_backend_call_id(backend_call_id, "backend_call_id")
    meta = validate_backend_result_meta(result["meta"])
    expected_input = finite_input_identity(submitted)
    expected = {
        "action_id": submitted["action_id"],
        "output_generation_id": submitted["output_generation_id"],
        "backend_call_id": call_id,
        "input_id": submitted["input_id"],
        "producer": submitted["producer"],
        "backend": submitted["expected_backend"],
        "input_identity": expected_input,
    }
    for field, expected_value in expected.items():
        if meta[field] != expected_value:
            raise ValueError(f"result meta {field} does not match submitted request")
    raster = submitted["raster_input"]["frame"]["transform"]["raster"]
    geojson = validate_inference_features(
        result["geojson"], raster, submitted["result_budget"], declared_classes
    )
    return canonical_tree({"meta": meta, "geojson": geojson})


def finite_input_identity(request):
    """Bind finite results to the one already-sealed image that supplied them."""

    bindings = (
        [{"unit_id": 0, "source": request["source_binding"]}]
        if "source_binding" in request
        else []
    )
    return build_backend_input_identity(request["raster_input"], bindings)


def _validate_call_status(value: Any, path: str) -> dict[str, Any]:
    status = _require_exact_object(value, _CALL_STATUS_FIELDS, path)
    _require_current_schema(status["schema_version"])
    return {
        "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
        "backend_call_id": require_backend_call_id(
            status["backend_call_id"], "backend_call_id"
        ),
    }


def _validate_addressed_call_status(
    status: dict[str, Any], addressed_backend_call_id: Any
) -> dict[str, Any]:
    call_id = require_backend_call_id(
        addressed_backend_call_id, "addressed_backend_call_id"
    )
    if status["backend_call_id"] != call_id:
        raise ValueError("response backend_call_id does not match addressed call")
    return status


def require_backend_call_id(value: Any, path: str = "backend_call_id") -> str:
    """Validate one opaque finite-call ID against its protocol byte ceiling."""

    call_id = require_nonblank_string(value, path)
    if len(call_id.encode("utf-8")) > MAX_BACKEND_CALL_ID_UTF8_BYTES:
        raise ValueError(
            f"{path} must be at most {MAX_BACKEND_CALL_ID_UTF8_BYTES} UTF-8 bytes"
        )
    return call_id


def encode_backend_call_query(value: Any) -> str:
    """Encode one validated opaque call ID as its canonical RFC 3986 query."""

    call_id = require_backend_call_id(value)
    encoded = quote_from_bytes(call_id.encode("utf-8"), safe="-._~")
    return f"{BACKEND_CALL_QUERY_PARAMETER}={encoded}"


def _module_ref(value: Any, path: str) -> str:
    text = require_nonblank_string(value, path)
    try:
        return str(ModuleRef.parse(text))
    except ValueError as exc:
        raise ValueError(f"{path} must be a canonical owner/name module ref") from exc


def _require_current_schema(value: Any) -> None:
    if require_int32(value, "schema_version") != FINITE_INFERENCE_SCHEMA_VERSION:
        raise ValueError("unsupported finite inference schema_version")


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
