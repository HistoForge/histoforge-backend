"""Bounded HTTP request-body and canonical path-scalar adaptation.

Domain routes validate parsed values themselves.  This module owns the two
transport invariants shared by those routes: never read an unbounded body, and
never let ASGI percent-decoding turn a noncanonical path spelling into a valid
identity.
"""

from __future__ import annotations

import re
from typing import Any

from histoforge.canonical_json import parse_json
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure

_BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_PAGE_CURSOR = re.compile(rb"[A-Za-z0-9_-]+")


async def read_bounded_body(request: Any, limit: int) -> bytes:
    """Read at most ``limit`` bytes without trusting ``Content-Length`` alone."""

    if limit <= 0:
        raise ValueError("request-body limit must be positive")
    _require_content_length(request.headers.get("content-length"), limit)

    body = bytearray()
    async for block in request.stream():
        if len(body) + len(block) > limit:
            raise _payload_too_large()
        body.extend(block)
    return bytes(body)


async def read_json_body(request: Any, limit: int) -> Any:
    """Read bounded UTF-8 JSON and return the duplicate-key-free parsed tree."""

    encoded = await read_bounded_body(request, limit)
    try:
        return parse_json(encoded)
    except (UnicodeDecodeError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            f"invalid JSON request: {error}",
        ) from error


def canonical_path_segment(
    request: Any,
    decoded_value: str,
    *,
    trailing_segments: int,
) -> str:
    """Return one canonical ASCII path segment from available ASGI evidence.

    ``trailing_segments`` counts path components after the requested segment.
    When the optional ``raw_path`` is present, reject any percent-encoded alias.
    Modal omits that optional field, so the deployed fallback validates the exact
    decoded ``path`` segment and its already-decoded route value instead.
    """

    raw_path = request.scope.get("raw_path")
    if isinstance(raw_path, bytes):
        segments: list[bytes | str] = raw_path.split(b"?", 1)[0].split(b"/")
    else:
        path = request.scope.get("path")
        if not isinstance(path, str):
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "request path is unavailable",
            )
        segments = path.split("?")[0].split("/")
    index = len(segments) - trailing_segments - 1
    if index < 0 or index >= len(segments):
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "request path has an invalid shape",
        )
    segment = segments[index]
    try:
        value = segment.decode("ascii") if isinstance(segment, bytes) else segment
    except UnicodeDecodeError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "request path segment must be canonical ASCII",
        ) from error
    if not value.isascii():
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "request path segment must be canonical ASCII",
        )
    if value != decoded_value:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "request path segment is not canonically encoded",
        )
    return value


def strict_single_query_value(request: Any, parameter: str) -> str:
    """Decode one RFC 3986 query value exactly once with strict UTF-8.

    Starlette's form-style query adapter accepts malformed percent escapes,
    replaces invalid UTF-8, and treats ``+`` as a space. Opaque call identity
    cannot use those rewrites, so this boundary reads the original ASGI bytes.
    """

    if not parameter.isascii() or not parameter:
        raise ValueError("query parameter name must be nonempty ASCII")
    raw = request.scope.get("query_string")
    if not isinstance(raw, bytes):
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "request query bytes are unavailable",
        )
    expected = parameter.encode("ascii") + b"="
    if b"&" in raw or not raw.startswith(expected):
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            f"query must contain exactly one {parameter} parameter",
        )
    encoded = raw[len(expected) :]
    decoded = bytearray()
    index = 0
    while index < len(encoded):
        byte = encoded[index]
        if byte != ord("%"):
            decoded.append(byte)
            index += 1
            continue
        if index + 2 >= len(encoded):
            raise _invalid_percent_encoding(parameter)
        pair = encoded[index + 1 : index + 3]
        if any(value not in b"0123456789abcdefABCDEF" for value in pair):
            raise _invalid_percent_encoding(parameter)
        decoded.append(int(pair, 16))
        index += 3
    try:
        return decoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            f"{parameter} must contain valid UTF-8",
        ) from error


def strict_page_query(request: Any, resource: str) -> tuple[int, str | None]:
    """Parse one literal ``limit`` and optional opaque ``cursor`` query.

    Pagination identities are already URL-safe. Reading the original ASGI bytes
    prevents a framework decoder from accepting percent-encoded aliases for the
    parameter names, decimal limit, or cursor.
    """

    raw = request.scope.get("query_string")
    if not isinstance(raw, bytes):
        raise _invalid_page_query(resource)
    values: dict[bytes, bytes] = {}
    for component in raw.split(b"&") if raw else ():
        name, separator, value = component.partition(b"=")
        if (
            not separator
            or name not in {b"limit", b"cursor"}
            or name in values
            or any(byte > 0x7F for byte in value)
        ):
            raise _invalid_page_query(resource)
        values[name] = value
    if b"limit" not in values:
        raise _invalid_page_query(resource)

    raw_limit = values[b"limit"]
    if not raw_limit.isdigit() or raw_limit.startswith(b"0") or len(raw_limit) > 10:
        raise _invalid_page_limit(resource)
    limit = int(raw_limit)
    if limit > 2**31 - 1:
        raise _invalid_page_limit(resource)

    encoded_cursor = values.get(b"cursor")
    if encoded_cursor is not None and not _PAGE_CURSOR.fullmatch(encoded_cursor):
        raise _invalid_page_query(resource)
    cursor = encoded_cursor.decode("ascii") if encoded_cursor is not None else None
    return limit, cursor


def parse_byte_range(value: str, size_bytes: int) -> tuple[int, int]:
    """Parse one RFC 9110 byte range and return an inclusive clipped interval."""

    match = _BYTE_RANGE.fullmatch(value) if isinstance(value, str) else None
    if match is None or size_bytes <= 0:
        raise ValueError("range is not satisfiable")
    first, last = match.groups()
    if not first and not last:
        raise ValueError("range is not satisfiable")

    if first:
        start, start_exceeds = _decimal_at_most(first, size_bytes - 1)
        end = _decimal_at_most(last, size_bytes - 1)[0] if last else size_bytes - 1
        if start_exceeds or start > end:
            raise ValueError("range is not satisfiable")
        return start, end

    suffix_size, _ = _decimal_at_most(last, size_bytes)
    if suffix_size <= 0:
        raise ValueError("range is not satisfiable")
    return max(0, size_bytes - suffix_size), size_bytes - 1


def _payload_too_large() -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.PAYLOAD_TOO_LARGE,
        "request body exceeds the configured byte limit",
    )


def _invalid_percent_encoding(parameter: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.VALIDATION_ERROR,
        f"{parameter} contains malformed percent encoding",
    )


def _invalid_page_query(resource: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.VALIDATION_ERROR,
        f"{resource} requires one limit and at most one cursor",
    )


def _invalid_page_limit(resource: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.VALIDATION_ERROR,
        f"{resource} limit must be a canonical positive 32-bit integer",
    )


def _require_content_length(value: str | None, limit: int) -> None:
    if value is None:
        return
    if not value.isascii() or not value.isdecimal():
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "invalid Content-Length header",
        )
    normalized = value.lstrip("0") or "0"
    maximum = str(limit)
    if len(normalized) > len(maximum) or (
        len(normalized) == len(maximum) and normalized > maximum
    ):
        raise _payload_too_large()


def _decimal_at_most(value: str, maximum: int) -> tuple[int, bool]:
    normalized = value.lstrip("0") or "0"
    maximum_text = str(maximum)
    exceeds = len(normalized) > len(maximum_text) or (
        len(normalized) == len(maximum_text) and normalized > maximum_text
    )
    return (maximum, True) if exceeds else (int(normalized), False)
