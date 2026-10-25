"""Authenticated HTTP adaptation for managed-training runs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import canonical_bytes, require_canonical_uuid
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.http_input import (
    canonical_path_segment,
    read_json_body,
    strict_page_query,
)

_LOG = logging.getLogger(__name__)


def register_routes(
    web: Any,
    *,
    require_token: Callable[..., Any],
    submission_service_factory: Callable[[dict[str, Any]], Any],
    access_service_factory: Callable[[dict[str, Any]], Any],
    max_json_request_bytes: int,
) -> None:
    """Register the complete authenticated training-run route family."""

    @web.post("/v1/training/runs")
    async def submit(request: Request, state=Depends(require_token)):
        value = await read_json_body(request, max_json_request_bytes)
        try:
            accepted = await run_in_threadpool(
                lambda: submission_service_factory(state).submit(value)
            )
        except GatewayFailure:
            raise
        except Exception as error:
            _LOG.exception("managed-training run submission failed unexpectedly")
            raise unexpected_gateway_failure(error) from error
        return Response(
            content=canonical_bytes(accepted),
            media_type="application/json",
            status_code=202,
        )

    @web.get("/v1/training/runs")
    async def page(request: Request, state=Depends(require_token)):
        limit, cursor = strict_page_query(request, "training run list")
        return await _run_json(
            lambda: access_service_factory(state).page(limit, cursor)
        )

    @web.get("/v1/training/runs/{run_id}")
    async def get(
        run_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _run_id(request, run_id, trailing_segments=0)
        return await _run_json(lambda: access_service_factory(state).get(canonical_id))

    @web.post("/v1/training/runs/{run_id}/cancel")
    async def cancel(
        run_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _run_id(request, run_id, trailing_segments=1)
        return await _run_json(
            lambda: access_service_factory(state).cancel(canonical_id)
        )

    @web.get("/v1/training/runs/{run_id}/report")
    async def report(
        run_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _run_id(request, run_id, trailing_segments=1)
        return await _run_json(
            lambda: access_service_factory(state).report(canonical_id)
        )

    @web.delete("/v1/training/runs/{run_id}")
    async def delete(
        run_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _run_id(request, run_id, trailing_segments=0)
        await _run(lambda: access_service_factory(state).delete(canonical_id))
        return Response(status_code=204)


async def _run_json(operation: Callable[[], Any]) -> Response:
    return Response(
        content=canonical_bytes(await _run(operation)),
        media_type="application/json",
        status_code=200,
    )


async def _run(operation: Callable[[], Any]) -> Any:
    try:
        return await run_in_threadpool(operation)
    except GatewayFailure:
        raise
    except Exception as error:
        raise unexpected_gateway_failure(error) from error


def _run_id(request: Request, value: str, *, trailing_segments: int) -> str:
    raw = canonical_path_segment(
        request,
        value,
        trailing_segments=trailing_segments,
    )
    try:
        return require_canonical_uuid(raw, "run_id")
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error
