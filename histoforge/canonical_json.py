"""RFC 8785/JCS JSON bytes and SHA-256 identities.

This module is the sole Python owner of JSON canonicalization used by HistoForge
contracts. Parsing is deliberately strict: duplicate object members and JSON
extensions such as NaN or Infinity fail before a value can acquire an identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any
from uuid import UUID

import rfc8785

_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_MAX_SIGNED_INT = 2**31 - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RFC3339_MILLIS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}Z"
)


def is_blank(value: str) -> bool:
    """Return whether contract text is empty or contains only Unicode whitespace."""

    return not value or all(
        _is_unicode_whitespace(ord(character)) for character in value
    )


def _is_unicode_whitespace(code_point: int) -> bool:
    """Implement Unicode White_Space independently of runtime tables."""

    return (
        0x0009 <= code_point <= 0x000D
        or code_point
        in {0x0020, 0x0085, 0x00A0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
        or 0x2000 <= code_point <= 0x200A
    )


def parse_json(source: str | bytes) -> Any:
    """Parse duplicate-key-free UTF-8 JSON.

    The returned value is an ordinary Python JSON tree. Call :func:`canonical_bytes`
    before treating it as identity-bearing; that also enforces the RFC 8785 number
    domain.
    """

    if isinstance(source, bytes):
        source = source.decode("utf-8")

    def object_from_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object member: {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON number: {value}")

    value = json.loads(
        source,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
    _require_unicode_scalars(value)
    return value


def _require_unicode_scalars(value: Any) -> None:
    if isinstance(value, str):
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON contains an unpaired Unicode surrogate")
    elif isinstance(value, list):
        for item in value:
            _require_unicode_scalars(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _require_unicode_scalars(key)
            _require_unicode_scalars(item)


def canonical_bytes(value: Any) -> bytes:
    """Return the RFC 8785/JCS UTF-8 representation of a JSON value."""

    try:
        return rfc8785.dumps(_binary64_numbers(value))
    except (rfc8785.CanonicalizationError, rfc8785.FloatDomainError) as exc:
        raise ValueError(f"value cannot be encoded as RFC 8785 JSON: {exc}") from exc


def canonical_tree(value: Any) -> Any:
    """Return the JSON tree represented by ``value`` under JCS number semantics."""

    return parse_json(canonical_bytes(value))


def _binary64_numbers(value: Any) -> Any:
    """Prepare equivalent JCS numbers without changing the original JSON tree."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and abs(value) > _MAX_SAFE_INTEGER:
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError("JSON number is outside the RFC 8785 domain") from exc
        if not math.isfinite(number):
            raise ValueError("JSON number is outside the RFC 8785 domain")
        return number
    if (
        isinstance(value, float)
        and value.is_integer()
        and abs(value) <= _MAX_SAFE_INTEGER
    ):
        # Pixel-edge coordinates commonly arrive as integral floats. JCS emits
        # the same bytes for safe integers, avoiding its general float formatter.
        return int(value)
    if isinstance(value, list):
        return [_binary64_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _binary64_numbers(item) for key, item in value.items()}
    return value


def canonicalize_json(source: str | bytes) -> bytes:
    """Strictly parse JSON and return its canonical UTF-8 representation."""

    return canonical_bytes(parse_json(source))


def sha256_bytes(value: bytes) -> str:
    """Return lowercase SHA-256 hex for exact bytes."""

    return hashlib.sha256(value).hexdigest()


def canonical_digest(value: Any) -> str:
    """Return lowercase SHA-256 hex over a JSON value's canonical bytes."""

    return sha256_bytes(canonical_bytes(value))


def require_nonblank_string(value: Any, path: str = "value") -> str:
    """Return a contract string after rejecting non-strings and Unicode blanks."""

    if not isinstance(value, str) or is_blank(value):
        raise ValueError(f"{path} must be a nonblank string")
    _require_unicode_scalars(value)
    return value


def require_sha256(value: Any, path: str = "value") -> str:
    """Return an exact lowercase 64-character SHA-256 value."""

    digest = require_nonblank_string(value, path)
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{path} must be lowercase 64-character SHA-256")
    return digest


def require_canonical_uuid(value: Any, path: str = "value") -> str:
    """Return lowercase canonical UUID text without alternate spellings."""

    text = require_nonblank_string(value, path)
    try:
        parsed = UUID(text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{path} must be a canonical UUID") from exc
    if str(parsed) != text:
        raise ValueError(f"{path} must be a lowercase canonical UUID")
    return text


def require_rfc3339_millis(value: Any, path: str = "value") -> str:
    """Return an exact UTC RFC 3339 timestamp with millisecond precision."""

    text = require_nonblank_string(value, path)
    if not _RFC3339_MILLIS.fullmatch(text):
        raise ValueError(f"{path} must be UTC RFC 3339 with exactly milliseconds")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(
            f"{path} must be UTC RFC 3339 with exactly milliseconds"
        ) from exc
    canonical = (
        f"{parsed.year:04d}-{parsed.month:02d}-{parsed.day:02d}"
        f"T{parsed.hour:02d}:{parsed.minute:02d}:{parsed.second:02d}"
        f".{parsed.microsecond // 1000:03d}Z"
    )
    if text != canonical:
        raise ValueError(f"{path} must be a canonical UTC millisecond timestamp")
    return text


def require_int32(value: Any, path: str = "value") -> int:
    """Return an exact JSON integer representable by Java's signed ``int``."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(2**31) <= value <= _MAX_SIGNED_INT
    ):
        raise ValueError(f"{path} must be a signed 32-bit integer")
    return value


def require_nonnegative_int32(value: Any, path: str = "value") -> int:
    """Return a nonnegative JSON integer representable by Java's signed ``int``."""

    result = require_int32(value, path)
    if result < 0:
        raise ValueError(f"{path} must be a nonnegative signed 32-bit integer")
    return result


def require_positive_int32(value: Any, path: str = "value") -> int:
    """Return a positive JSON integer representable by Java's signed ``int``."""

    result = require_int32(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be a positive signed 32-bit integer")
    return result


def require_finite_number(value: Any, path: str = "value") -> float:
    """Return one finite binary64 value, normalizing negative zero to positive zero."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return 0.0 if result == 0.0 else result


def require_positive_finite_number(value: Any, path: str = "value") -> float:
    """Return one positive finite binary64 value."""

    result = require_finite_number(value, path)
    if result <= 0.0:
        raise ValueError(f"{path} must be a positive finite number")
    return result


def require_nonnegative_safe_integer(value: Any, path: str = "value") -> int:
    """Return an exact nonnegative integer in the RFC 8785 safe range."""

    result = require_safe_integer(value, path)
    if result < 0:
        raise ValueError(f"{path} must be a nonnegative safe integer")
    return result


def require_safe_integer(value: Any, path: str = "value") -> int:
    """Return an exact signed integer in the RFC 8785 safe range."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER
    ):
        raise ValueError(f"{path} must be a safe integer")
    return value


def require_positive_safe_integer(value: Any, path: str = "value") -> int:
    """Return an exact positive integer in the RFC 8785 safe range."""

    result = require_nonnegative_safe_integer(value, path)
    if result == 0:
        raise ValueError(f"{path} must be a positive safe integer")
    return result


def require_path_class_name(value: Any, path: str = "value") -> str:
    """Require QuPath 0.6's canonical ``PathClass.toString()`` spelling.

    Java activation performs the authoritative ``fromString``/``toString`` round trip.
    This mirrors that default-delimiter normalization so backend module
    validation accepts the same canonical class vocabulary.
    """

    text = require_nonblank_string(value, path)
    components = [_java_strip(component) for component in text.split(":")]
    if any(
        not component or "\n" in component or "\r" in component
        for component in components
    ):
        raise ValueError(f"{path} must be a canonical QuPath PathClass name")
    if ": ".join(components) != text:
        raise ValueError(f"{path} must be a canonical QuPath PathClass name")
    return text


def _java_strip(value: str) -> str:
    """Apply the closed whitespace set used by Java 21 ``String.strip``."""

    start = 0
    end = len(value)
    while start < end and _is_java_whitespace(ord(value[start])):
        start += 1
    while end > start and _is_java_whitespace(ord(value[end - 1])):
        end -= 1
    return value[start:end]


def _is_java_whitespace(code_point: int) -> bool:
    return (
        0x0009 <= code_point <= 0x000D
        or 0x001C <= code_point <= 0x001F
        or code_point in {0x0020, 0x1680, 0x2028, 0x2029, 0x205F, 0x3000}
        or 0x2000 <= code_point <= 0x2006
        or 0x2008 <= code_point <= 0x200A
    )


def scalar_map_key(value: str | int) -> str:
    """Encode a validated schema-1 scalar identity as a JSON object-member key."""

    if isinstance(value, bool):
        raise ValueError("boolean is not a schema-1 scalar map key")
    if isinstance(value, str):
        _require_unicode_scalars(value)
        return value
    if isinstance(value, int) and 0 <= value <= _MAX_SIGNED_INT:
        return str(value)
    raise ValueError("scalar map key must be a validated string or nonnegative int32")


def canonical_map_entries(entries: Iterable[tuple[Any, Any]]) -> list[dict[str, Any]]:
    """Encode a compound-key map as entries sorted by canonical key bytes."""

    ordered: list[tuple[bytes, Any, Any]] = []
    seen: set[bytes] = set()
    for key, value in entries:
        key_bytes = canonical_bytes(key)
        if key_bytes in seen:
            raise ValueError("compound map contains a duplicate canonical key")
        seen.add(key_bytes)
        ordered.append((key_bytes, canonical_tree(key), canonical_tree(value)))
    ordered.sort(key=lambda item: item[0])
    return [{"key": key, "value": value} for _, key, value in ordered]


def canonical_unordered(values: Iterable[Any]) -> list[Any]:
    """Encode a semantic set as values sorted by canonical element bytes."""

    ordered: list[tuple[bytes, Any]] = []
    seen: set[bytes] = set()
    for value in values:
        value_bytes = canonical_bytes(value)
        if value_bytes in seen:
            raise ValueError(
                "unordered collection contains a duplicate canonical value"
            )
        seen.add(value_bytes)
        ordered.append((value_bytes, canonical_tree(value)))
    ordered.sort(key=lambda item: item[0])
    return [value for _, value in ordered]
