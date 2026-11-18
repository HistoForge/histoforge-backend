"""Authenticated HTTP reads for immutable opaque-ID artifacts."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from histoforge.canonical_json import require_canonical_uuid
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    unexpected_gateway_failure,
)
from histoforge.cloud.artifacts.store import (
    DEFAULT_READ_BLOCK_BYTES,
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
    ArtifactFileStore,
)
from histoforge.cloud.http_input import (
    canonical_path_segment,
    parse_byte_range,
)


def register_routes(web, *, require_token, files: ArtifactFileStore) -> None:
    """Register the one complete/range byte-read endpoint."""

    @web.get("/v1/artifacts/{artifact_id}")
    async def read_artifact(
        artifact_id: str,
        request: Request,
        _state=Depends(require_token),
    ):
        raw_id = canonical_path_segment(request, artifact_id, trailing_segments=0)
        try:
            canonical_id = require_canonical_uuid(raw_id, "artifact_id")
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                str(error),
            ) from error
        reference = await _artifact_ref(files, canonical_id)
        interval = _requested_interval(request, reference)
        start, end = interval or (0, reference["size_bytes"] - 1)
        headers = _response_headers(reference, start, end, interval is not None)
        return StreamingResponse(
            _blocks(files, canonical_id, start, end),
            status_code=206 if interval is not None else 200,
            headers=headers,
        )


async def _artifact_ref(files: ArtifactFileStore, artifact_id: str) -> dict:
    try:
        return await run_in_threadpool(files.load_artifact_ref, artifact_id)
    except ArtifactBytesMissing as error:
        raise GatewayFailure(
            GatewayErrorCode.ARTIFACT_NOT_FOUND,
            "artifact bytes do not exist",
            resource_id=artifact_id,
        ) from error
    except ArtifactBytesCorrupt as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "artifact bytes are corrupt; recreate the owning operation",
            resource_id=artifact_id,
        ) from error
    except GatewayFailure:
        raise
    except Exception as error:
        raise unexpected_gateway_failure(error) from error


def _requested_interval(request: Request, reference: dict) -> tuple[int, int] | None:
    ranges = _header_values(request, b"range")
    if not ranges:
        return None
    if len(ranges) != 1:
        raise _range_failure(reference["artifact_id"])

    matches = _header_values(request, b"if-match")
    expected = f'"{reference["sha256"]}"'
    if len(matches) != 1 or matches[0] != expected:
        raise GatewayFailure(
            GatewayErrorCode.ARTIFACT_PRECONDITION_FAILED,
            "artifact range requires its exact strong ETag",
            resource_id=reference["artifact_id"],
        )
    try:
        return parse_byte_range(ranges[0], reference["size_bytes"])
    except ValueError as error:
        raise _range_failure(reference["artifact_id"]) from error


def _header_values(request: Request, name: bytes) -> list[str]:
    values = []
    for key, value in request.scope.get("headers", []):
        if key.lower() != name:
            continue
        try:
            values.append(value.decode("ascii"))
        except UnicodeDecodeError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "request header must be ASCII",
            ) from error
    return values


def _response_headers(
    reference: dict,
    start: int,
    end: int,
    partial: bool,
) -> dict[str, str]:
    headers = {
        "Content-Type": reference["media_type"],
        "Content-Length": str(end - start + 1),
        "Accept-Ranges": "bytes",
        "ETag": f'"{reference["sha256"]}"',
    }
    if "content_encoding" in reference:
        headers["Content-Encoding"] = reference["content_encoding"]
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{reference['size_bytes']}"
    return headers


async def _blocks(
    files: ArtifactFileStore,
    artifact_id: str,
    start: int,
    end: int,
) -> AsyncIterator[bytes]:
    offset = start
    while offset <= end:
        size = min(DEFAULT_READ_BLOCK_BYTES, end - offset + 1)
        # The worker retains ArtifactFileStore's process lock through close,
        # even if the response coroutine is cancelled while awaiting it.
        yield await run_in_threadpool(
            files.read_artifact_block,
            artifact_id,
            offset,
            size,
        )
        offset += size


def _range_failure(artifact_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.RANGE_NOT_SATISFIABLE,
        "artifact byte range is not satisfiable",
        resource_id=artifact_id,
    )
