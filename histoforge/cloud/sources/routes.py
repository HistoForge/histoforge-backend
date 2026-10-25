"""FastAPI adaptation for source-artifact upload and lifecycle routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import require_canonical_uuid
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.http_input import (
    canonical_path_segment,
    read_bounded_body,
    read_json_body,
)
from histoforge.cloud.sources.contract import validate_source_artifact_start_request
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.publication import (
    validate_source_artifact_seal_request,
)
from histoforge.cloud.sources.store import SourceArtifactStore


def register_routes(
    web,
    *,
    require_token,
    store: SourceArtifactStore,
    files: SourceFileStore,
    max_json_request_bytes: int,
    lifecycle: Callable[[dict[str, Any]], Any],
) -> None:
    """Register source routes against one authenticated lifecycle owner."""

    @web.post("/v1/source-artifacts")
    async def start(request: Request, state=Depends(require_token)):
        value = _request_value(
            validate_source_artifact_start_request,
            await read_json_body(request, max_json_request_bytes),
        )
        return await _run(lambda: lifecycle(state).reserve(value))

    @web.put("/v1/source-artifacts/{artifact_id}/chunks/{chunk_id}")
    async def put_chunk(
        artifact_id: str,
        chunk_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _artifact_id(request, artifact_id, trailing_segments=2)
        canonical_chunk = _chunk_id(request, chunk_id)
        _require_octet_stream(request)
        expected = await _run(lambda: store.chunk_range(canonical_id, canonical_chunk))
        body = await read_bounded_body(request, expected["size"])
        control = lifecycle(state)
        return await _run(
            lambda: store.accept_chunk(
                canonical_id,
                canonical_chunk,
                body,
                files,
                control=control,
            )
        )

    @web.post("/v1/source-artifacts/{artifact_id}/seal")
    async def seal(
        artifact_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _artifact_id(request, artifact_id, trailing_segments=1)
        value = _request_value(
            validate_source_artifact_seal_request,
            await read_json_body(request, max_json_request_bytes),
        )
        control = lifecycle(state)
        return await _run(
            lambda: store.seal(
                canonical_id,
                value,
                files,
                control=control,
            )
        )

    @web.get("/v1/source-artifacts/{artifact_id}")
    async def status(
        artifact_id: str,
        request: Request,
        _state=Depends(require_token),
    ):
        canonical_id = _artifact_id(request, artifact_id, trailing_segments=0)
        return await _run(lambda: store.status(canonical_id))

    @web.delete("/v1/source-artifacts/{artifact_id}")
    async def delete(
        artifact_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _artifact_id(request, artifact_id, trailing_segments=0)
        await _run(lambda: lifecycle(state).release(canonical_id))
        return Response(status_code=204)


async def _run(operation: Callable[[], Any]) -> Any:
    try:
        return await run_in_threadpool(operation)
    except GatewayFailure:
        raise
    except Exception as error:
        raise unexpected_gateway_failure(error) from error


def _request_value(validator, value: Any) -> dict[str, Any]:
    try:
        return validator(value)
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error


def _artifact_id(request: Request, value: str, *, trailing_segments: int) -> str:
    raw = canonical_path_segment(
        request,
        value,
        trailing_segments=trailing_segments,
    )
    try:
        return require_canonical_uuid(raw, "artifact_id")
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error


def _chunk_id(request: Request, value: str) -> int:
    raw = canonical_path_segment(request, value, trailing_segments=0)
    if (
        not raw.isascii()
        or not raw.isdecimal()
        or (len(raw) > 1 and raw.startswith("0"))
        or len(raw) > 10
    ):
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "chunk_id must be a canonical nonnegative 32-bit integer",
        )
    result = int(raw)
    if result > 2**31 - 1:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "chunk_id must be a canonical nonnegative 32-bit integer",
        )
    return result


def _require_octet_stream(request: Request) -> None:
    values = [
        value
        for key, value in request.scope.get("headers", [])
        if key.lower() == b"content-type"
    ]
    if values != [b"application/octet-stream"]:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "source chunks require Content-Type: application/octet-stream",
        )
