"""Authenticated HTTP adaptation for immutable training datasets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request, Response
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import (
    canonical_bytes,
    require_canonical_uuid,
    require_sha256,
)
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.http_input import (
    canonical_path_segment,
    read_bounded_body,
    read_json_body,
    strict_page_query,
)


def register_routes(
    web: Any,
    *,
    require_token: Callable[..., Any],
    service_factory: Callable[[dict[str, Any]], Any],
    max_json_request_bytes: int,
    max_chunk_bytes: int,
) -> None:
    """Register the complete training-dataset route family."""

    @web.post("/v1/training/datasets")
    async def start(request: Request, state=Depends(require_token)):
        value = await read_json_body(request, max_json_request_bytes)
        return await _run_json(lambda: service_factory(state).start(value))

    @web.put("/v1/training/datasets/{dataset_id}/chunks/{chunk_index}")
    async def put_chunk(
        dataset_id: str,
        chunk_index: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _dataset_id(request, dataset_id, trailing_segments=2)
        canonical_index = _chunk_index(request, chunk_index)
        _require_octet_stream(request)
        digest = _chunk_digest(request)
        body = await read_bounded_body(request, max_chunk_bytes)
        return await _run_json(
            lambda: service_factory(state).put_chunk(
                canonical_id,
                canonical_index,
                body,
                digest,
            )
        )

    @web.post("/v1/training/datasets/{dataset_id}/seal")
    async def seal(
        dataset_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _dataset_id(request, dataset_id, trailing_segments=1)
        value = await read_json_body(request, max_json_request_bytes)
        return await _run_json(lambda: service_factory(state).seal(canonical_id, value))

    @web.get("/v1/training/datasets")
    async def page(request: Request, state=Depends(require_token)):
        limit, cursor = strict_page_query(request, "dataset list")
        return await _run_json(lambda: service_factory(state).page(limit, cursor))

    @web.get("/v1/training/datasets/{dataset_id}")
    async def get(
        dataset_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _dataset_id(request, dataset_id, trailing_segments=0)
        return await _run_json(lambda: service_factory(state).get(canonical_id))

    @web.delete("/v1/training/datasets/{dataset_id}")
    async def delete(
        dataset_id: str,
        request: Request,
        state=Depends(require_token),
    ):
        canonical_id = _dataset_id(request, dataset_id, trailing_segments=0)
        await _run(lambda: service_factory(state).delete(canonical_id))
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


def _dataset_id(request: Request, value: str, *, trailing_segments: int) -> str:
    raw = canonical_path_segment(
        request,
        value,
        trailing_segments=trailing_segments,
    )
    try:
        return require_canonical_uuid(raw, "dataset_id")
    except ValueError as error:
        raise GatewayFailure(GatewayErrorCode.VALIDATION_ERROR, str(error)) from error


def _chunk_index(request: Request, value: str) -> int:
    raw = canonical_path_segment(request, value, trailing_segments=0)
    if (
        not raw.isascii()
        or not raw.isdecimal()
        or (len(raw) > 1 and raw.startswith("0"))
        or len(raw) > 10
    ):
        raise _invalid_chunk_index()
    result = int(raw)
    if result > 2**31 - 1:
        raise _invalid_chunk_index()
    return result


def _invalid_chunk_index() -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.VALIDATION_ERROR,
        "chunk index must be a canonical nonnegative 32-bit integer",
    )


def _require_octet_stream(request: Request) -> None:
    values = _header_values(request, b"content-type")
    if values != [b"application/octet-stream"]:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "training chunks require Content-Type: application/octet-stream",
        )


def _chunk_digest(request: Request) -> str:
    values = _header_values(request, b"x-histoforge-chunk-sha256")
    if len(values) != 1:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "training chunks require one X-HistoForge-Chunk-Sha256 header",
        )
    try:
        value = values[0].decode("ascii")
        return require_sha256(value, "X-HistoForge-Chunk-Sha256")
    except (UnicodeDecodeError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "X-HistoForge-Chunk-Sha256 must be lowercase hexadecimal SHA-256",
        ) from error


def _header_values(request: Request, name: bytes) -> list[bytes]:
    return [
        value for key, value in request.scope.get("headers", []) if key.lower() == name
    ]
