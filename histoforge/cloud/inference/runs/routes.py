"""Authenticated HTTP adaptation for durable exact-model runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import modal
from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import (
    canonical_bytes,
    require_canonical_uuid,
)
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    build_gateway_error,
    unexpected_gateway_failure,
    validate_gateway_error,
)
from histoforge.cloud.http_input import canonical_path_segment, read_json_body
from histoforge.cloud.inference.route_activation import (
    load_current_model_activation,
    model_route_identity,
)
from histoforge.cloud.inference.runs.request import validate_model_run_acceptance
from histoforge.cloud.inference.runs.runtime import validate_durable_admission
from histoforge.cloud.modules.runtime import current_module_snapshot


def register_routes(
    web: Any,
    *,
    require_token: Callable[..., Any],
    max_json_request_bytes: int,
    training_state_factory: Callable[[dict[str, Any]], Any],
    current_modules: Callable[[], Any] = current_module_snapshot,
    lifecycle_factory: Callable[[dict[str, Any]], Any] | None = None,
    spawn_run: Callable[[str], Any] | None = None,
) -> None:
    """Register strict start, retained status, and idempotent cancel routes."""

    make_lifecycle = lifecycle_factory or _ModelRunLifecycleClient
    spawn = spawn_run or _spawn_run

    @web.post("/v1/inference/{owner}/{name}/{model_id}/runs")
    async def submit(
        owner: str,
        name: str,
        model_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        module_ref, canonical_model_id = model_route_identity(
            request,
            owner,
            name,
            model_id,
            trailing_segments=1,
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
            admitted, limits, retention = await run_in_threadpool(
                validate_durable_admission,
                value,
                module_ref,
                canonical_model_id,
                activation,
            )
            reservation = await run_in_threadpool(
                make_lifecycle(state).reserve_model_run,
                admitted,
                limits,
                retention,
            )
            accepted, should_dispatch = _reservation(reservation, admitted)
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error

        if should_dispatch:
            try:
                await run_in_threadpool(spawn, admitted["model_run_id"])
            except Exception as error:
                failure = build_gateway_error(
                    GatewayErrorCode.SERVICE_UNAVAILABLE,
                    "durable model run could not be dispatched",
                    resource_id=admitted["model_run_id"],
                )
                try:
                    await run_in_threadpool(
                        make_lifecycle(state).fail,
                        admitted["model_run_id"],
                        failure,
                    )
                except Exception:
                    pass
                raise GatewayFailure(
                    failure["code"],
                    failure["message"],
                    resource_id=failure["resource_id"],
                ) from error
        return _json_response(accepted, 202)

    @web.get("/v1/inference/runs/{model_run_id}")
    async def status(
        model_run_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _route_run_id(request, model_run_id)
        try:
            value = await run_in_threadpool(
                make_lifecycle(state).status,
                canonical_id,
            )
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        return _json_response(value, 200)

    @web.delete("/v1/inference/runs/{model_run_id}")
    async def cancel(
        model_run_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _route_run_id(request, model_run_id)
        try:
            await run_in_threadpool(
                make_lifecycle(state).cancel,
                canonical_id,
            )
        except GatewayFailure:
            raise
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        return Response(status_code=204)

    @web.get("/v1/inference/runs/{model_run_id}/sources")
    async def sources(
        model_run_id: str, request: Request, state=Depends(require_token)
    ):
        canonical_id = _route_run_id(request, model_run_id, trailing_segments=1)
        try:
            value = await run_in_threadpool(make_lifecycle(state).sources, canonical_id)
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        return _json_response(value, 200)

    @web.post("/v1/inference/runs/{model_run_id}/sources")
    async def bind_source(
        model_run_id: str, request: Request, state=Depends(require_token)
    ):
        canonical_id = _route_run_id(request, model_run_id, trailing_segments=1)
        value = await read_json_body(request, max_json_request_bytes)
        try:
            accepted = await run_in_threadpool(
                make_lifecycle(state).bind_source, canonical_id, value
            )
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR, str(error)
            ) from error
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        return _json_response(accepted, 200)

    @web.post("/v1/inference/runs/{model_run_id}/source-failure")
    async def source_failure(
        model_run_id: str, request: Request, state=Depends(require_token)
    ):
        canonical_id = _route_run_id(request, model_run_id, trailing_segments=1)
        value = await read_json_body(request, max_json_request_bytes)
        try:
            failure = validate_gateway_error(value)
            await run_in_threadpool(
                make_lifecycle(state).report_source_failure, canonical_id, failure
            )
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR, str(error)
            ) from error
        except Exception as error:
            raise unexpected_gateway_failure(error) from error
        return Response(status_code=204)


class _ModelRunLifecycleClient:
    """Bind authenticated run admission/status/cancel and failure publication."""

    def __init__(self, state: dict[str, Any]) -> None:
        self._lane = modal.Cls.from_name(
            DEPLOYMENT_CONFIG.gateway_app_name,
            DEPLOYMENT_CONFIG.artifact_lifecycle_class_name,
        )()
        self._deployment_id = state["deployment_id"]
        self._auth_revision = state["auth_revision"]

    def reserve_model_run(
        self,
        request: dict,
        model_run_limits: dict,
        terminal_retention_seconds: int,
    ) -> dict:
        """Reserve in the same lane as source release and artifact collection."""

        return self._lane.reserve_model_run.remote(
            self._deployment_id,
            self._auth_revision,
            request,
            model_run_limits,
            terminal_retention_seconds,
        )

    def status(self, model_run_id: str) -> dict[str, Any]:
        return self._lane.status.remote(
            self._deployment_id,
            self._auth_revision,
            model_run_id,
        )

    def cancel(self, model_run_id: str) -> None:
        self._lane.cancel.remote(
            self._deployment_id,
            self._auth_revision,
            model_run_id,
        )

    def fail(self, model_run_id: str, failure: dict[str, Any]) -> bool:
        return self._lane.fail.remote(model_run_id, failure)

    def sources(self, model_run_id: str) -> dict[str, Any]:
        return self._lane.model_run_sources.remote(
            self._deployment_id, self._auth_revision, model_run_id
        )

    def bind_source(self, model_run_id: str, binding: dict[str, Any]) -> dict[str, Any]:
        return self._lane.bind_model_run_source.remote(
            self._deployment_id, self._auth_revision, model_run_id, binding
        )

    def report_source_failure(self, model_run_id: str, failure: dict[str, Any]) -> None:
        self._lane.report_source_failure.remote(
            self._deployment_id, self._auth_revision, model_run_id, failure
        )


def _reservation(value: Any, request: Any) -> tuple[dict[str, Any], bool]:
    if not isinstance(value, dict) or set(value) != {"accepted", "should_dispatch"}:
        raise GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "durable reservation returned an invalid result",
        )
    should_dispatch = value["should_dispatch"]
    if not isinstance(should_dispatch, bool):
        raise GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "durable reservation returned an invalid dispatch decision",
        )
    try:
        accepted = validate_model_run_acceptance(request, value["accepted"])
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "durable reservation returned an invalid acceptance",
        ) from error
    return accepted, should_dispatch


def _route_run_id(
    request: Any, model_run_id: str, *, trailing_segments: int = 0
) -> str:
    raw = canonical_path_segment(
        request, model_run_id, trailing_segments=trailing_segments
    )
    try:
        return require_canonical_uuid(raw, "model_run_id")
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            str(error),
        ) from error


def _spawn_run(model_run_id: str) -> Any:
    worker = modal.Function.from_name(
        DEPLOYMENT_CONFIG.gateway_app_name,
        DEPLOYMENT_CONFIG.model_run_executor_function_name,
    )
    return worker.spawn(model_run_id)


def _json_response(value: Any, status_code: int) -> Response:
    return Response(
        content=canonical_bytes(value),
        media_type="application/json",
        status_code=status_code,
    )
