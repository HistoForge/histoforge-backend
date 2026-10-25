"""Authenticated HTTP adaptation for bounded finite model calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import modal
from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.architectures.registry import runner_precision_for_build
from histoforge.cloud.http_input import read_json_body, strict_single_query_value
from histoforge.cloud.inference.finite.contract import (
    BACKEND_CALL_QUERY_PARAMETER,
    require_backend_call_id,
    validate_inference_accepted,
    validate_inference_cancellation_accepted,
    validate_inference_pending,
)
from histoforge.cloud.inference.finite.runtime import (
    encode_finite_result,
    validate_finite_admission,
)
from histoforge.cloud.inference.route_activation import (
    load_current_model_activation,
    model_route_identity,
)
from histoforge.cloud.modules.runtime import current_module_snapshot
from histoforge.cloud.protocol_versions import FINITE_INFERENCE_SCHEMA_VERSION


def register_routes(
    web: Any,
    *,
    require_token: Callable[..., Any],
    source_store: Any,
    files: Any,
    max_json_request_bytes: int,
    training_state_factory: Callable[[dict[str, Any]], Any],
    current_modules: Callable[[], Any] = current_module_snapshot,
    spawn_call: Callable[..., Any] | None = None,
    call_from_id: Callable[[str], Any] | None = None,
) -> None:
    """Register the strict submit, poll, and best-effort cancel routes."""

    spawn = spawn_call or _spawn_call
    recover = call_from_id or modal.FunctionCall.from_id

    @web.post("/v1/inference/{owner}/{name}/{model_id}")
    async def submit(
        owner: str,
        name: str,
        model_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        module_ref, canonical_model_id = model_route_identity(
            request, owner, name, model_id
        )
        value = await read_json_body(request, max_json_request_bytes)
        try:
            activation = await run_in_threadpool(
                load_current_model_activation,
                current_modules,
                module_ref,
                canonical_model_id,
                training_state_factory(state).training_result_visible,
            )
            admitted = await run_in_threadpool(
                validate_finite_admission,
                value,
                module_ref,
                canonical_model_id,
                activation["backend"],
                activation["capabilities"],
                source_store,
                files,
            )
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        try:
            call = await run_in_threadpool(
                spawn,
                activation["modal_class_name"],
                admitted["expected_backend"]["gpu"],
                module_ref,
                canonical_model_id,
                activation["backend"]["module"]["content_digest"],
                activation["backend"]["module"].get("resolved_commit_sha"),
                admitted,
            )
        except Exception as error:
            raise GatewayFailure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "finite inference could not be dispatched",
            ) from error
        try:
            call_id = require_backend_call_id(call.object_id)
        except (AttributeError, ValueError) as error:
            _cancel_quietly(call)
            raise GatewayFailure(
                GatewayErrorCode.INTERNAL_ERROR,
                "finite inference returned an invalid server call identity",
            ) from error
        accepted = validate_inference_accepted(
            {
                "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
                "action_id": admitted["action_id"],
                "output_generation_id": admitted["output_generation_id"],
                "backend_call_id": call_id,
            }
        )
        return _json_response(Response, accepted, 202)

    @web.get("/v1/inference/calls")
    def poll(request: Request, _state=Depends(require_token)):
        call_id = _addressed_call_id(request)
        try:
            call = recover(call_id)
        except modal.exception.NotFoundError as error:
            raise _call_failure(
                GatewayErrorCode.CALL_NOT_FOUND,
                "finite call does not exist",
                call_id,
                error,
            )
        except Exception as error:
            raise _call_failure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "finite call status is temporarily unavailable",
                call_id,
                error,
            )
        try:
            outcome = call.get(timeout=0)
        except GatewayFailure:
            raise
        except modal.exception.NotFoundError as error:
            raise _call_failure(
                GatewayErrorCode.CALL_NOT_FOUND,
                "finite call does not exist",
                call_id,
                error,
            )
        except modal.exception.OutputExpiredError as error:
            raise _call_failure(
                GatewayErrorCode.CALL_EXPIRED,
                "finite call output has expired",
                call_id,
                error,
            )
        except modal.exception.FunctionTimeoutError as error:
            raise _call_failure(
                GatewayErrorCode.RUNNER_FAILED,
                "finite runner exceeded its attempt timeout",
                call_id,
                error,
            )
        except modal.exception.RemoteError as error:
            raise _call_failure(
                GatewayErrorCode.RUNNER_FAILED,
                "finite runner failed before publishing a typed result",
                call_id,
                error,
            )
        except (
            modal.exception.InternalFailure,
            modal.exception.ExecutionError,
        ) as error:
            raise _call_failure(
                GatewayErrorCode.RUNNER_FAILED,
                "finite runner failed before publishing a typed result",
                call_id,
                error,
            )
        except TimeoutError:
            pending = validate_inference_pending(
                {
                    "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
                    "backend_call_id": call_id,
                }
            )
            return _json_response(Response, pending, 202)
        except (ConnectionError, OSError, modal.exception.Error) as error:
            raise _call_failure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "finite call status is temporarily unavailable",
                call_id,
                error,
            )
        except Exception as error:
            raise _call_failure(
                GatewayErrorCode.RUNNER_FAILED,
                "finite runner failed before publishing a typed result",
                call_id,
                error,
            )
        return Response(
            content=encode_finite_result(outcome, call_id),
            media_type="application/json",
            status_code=200,
        )

    @web.delete("/v1/inference/calls")
    def cancel(request: Request, _state=Depends(require_token)):
        call_id = _addressed_call_id(request)
        try:
            recover(call_id).cancel()
        except modal.exception.NotFoundError as error:
            raise _call_failure(
                GatewayErrorCode.CALL_NOT_FOUND,
                "finite call does not exist",
                call_id,
                error,
            )
        except Exception as error:
            raise _call_failure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "finite call cancellation is temporarily unavailable",
                call_id,
                error,
            )
        accepted = validate_inference_cancellation_accepted(
            {
                "schema_version": FINITE_INFERENCE_SCHEMA_VERSION,
                "backend_call_id": call_id,
            }
        )
        return _json_response(Response, accepted, 202)


def _addressed_call_id(request: Any) -> str:
    value = strict_single_query_value(request, BACKEND_CALL_QUERY_PARAMETER)
    try:
        return require_backend_call_id(value)
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            str(error),
        ) from error


def _spawn_call(
    class_name: str,
    gpu: str,
    module_ref: str,
    model_id: str,
    content_digest: str,
    resolved_commit_sha: str | None,
    request: dict[str, Any],
) -> Any:
    backend = request["expected_backend"]
    precision = runner_precision_for_build(
        backend["architecture"], backend["runner_build"], gpu
    )
    runner_cls = modal.Cls.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        class_name,
    ).with_options(gpu=gpu)
    runner = runner_cls(
        ref=module_ref,
        model_id=model_id,
        content_digest=content_digest,
        resolved_commit_sha=resolved_commit_sha or "",
        gpu=gpu,
        inference_precision=precision,
    )
    return runner.infer.spawn(request)


def _cancel_quietly(call: Any) -> None:
    if call is None:
        return
    try:
        call.cancel()
    except Exception:
        pass


def _call_failure(
    code: GatewayErrorCode,
    message: str,
    call_id: str,
    cause: BaseException,
) -> GatewayFailure:
    failure = GatewayFailure(code, message, resource_id=call_id)
    failure.__cause__ = cause
    return failure


def _json_response(response_type: Any, value: Any, status_code: int) -> Any:
    return response_type(
        content=canonical_bytes(value),
        media_type="application/json",
        status_code=status_code,
    )
