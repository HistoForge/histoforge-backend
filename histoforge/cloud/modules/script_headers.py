"""Strict schema-1 contracts for HistoForge Groovy script headers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    is_blank,
    parse_json,
    require_nonnegative_int32,
    require_path_class_name,
)

_BLOCK_COMMENT = re.compile(r"/\*(.*?)\*/", re.DOTALL)
_HEADER_SPACE = r"[ \t\n\x0b\f\r]"
_HEADER = re.compile(
    rf"^{_HEADER_SPACE}*@histoforge-(classes|metrics)"
    rf"{_HEADER_SPACE}+(\[.*\]){_HEADER_SPACE}*$",
    re.DOTALL,
)

_METRIC_FIELDS = {
    "name",
    "label",
    "type",
    "unit",
    "decimals",
    "aggregation",
}
_REQUIRED_METRIC_FIELDS = {
    "name",
    "label",
    "type",
    "decimals",
    "aggregation",
}


def parse_ordinary_script_header(source: str) -> tuple[str, ...]:
    """Return an ordinary script's declared classes after strict validation."""

    header = _extract_header(source)
    if header is None:
        return ()
    kind, payload = header
    if kind != "classes":
        raise ValueError("ordinary script cannot declare @histoforge-metrics")

    value = parse_json(payload)
    if not isinstance(value, list) or not value:
        raise ValueError("@histoforge-classes must be a nonempty JSON array")
    classes = tuple(require_path_class_name(item, "class") for item in value)
    if len(set(classes)) != len(classes):
        raise ValueError("@histoforge-classes contains a duplicate class")
    return classes


def parse_metric_script_header(source: str) -> tuple[dict[str, Any], ...]:
    """Return the normalized metric definitions declared by metrics.groovy."""

    header = _extract_header(source)
    if header is None:
        raise ValueError("metrics.groovy is missing @histoforge-metrics")
    kind, payload = header
    if kind != "metrics":
        raise ValueError("metrics.groovy cannot declare @histoforge-classes")

    value = parse_json(payload)
    if not isinstance(value, list) or not value:
        raise ValueError("@histoforge-metrics must be a nonempty JSON array")

    definitions = tuple(
        _metric_definition(item, index) for index, item in enumerate(value)
    )
    names = [definition["name"] for definition in definitions]
    if len(set(names)) != len(names):
        raise ValueError("@histoforge-metrics contains a duplicate metric name")
    return definitions


def _extract_header(source: str) -> tuple[str, str] | None:
    if not isinstance(source, str):
        raise ValueError("script source must be UTF-8 text")

    comments = list(_BLOCK_COMMENT.finditer(source))
    parsed: list[tuple[int, str, str]] = []
    for index, comment in enumerate(comments):
        body = comment.group(1)
        match = _HEADER.fullmatch(body)
        if match:
            parsed.append((index, match.group(1), match.group(2)))
        elif "@histoforge-" in body:
            raise ValueError("malformed @histoforge header block")

    if not parsed:
        return None
    if len(parsed) != 1:
        raise ValueError("script must contain at most one @histoforge header")
    index, kind, payload = parsed[0]
    if index != 0:
        raise ValueError("@histoforge header must be the first block comment")
    return kind, payload


def _metric_definition(value: Any, index: int) -> dict[str, Any]:
    path = f"metric[{index}]"
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    fields = set(value)
    unknown = fields - _METRIC_FIELDS
    missing = _REQUIRED_METRIC_FIELDS - fields
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")

    metric_type = _nonblank_string(value["type"], f"{path}.type")
    aggregation = _nonblank_string(value["aggregation"], f"{path}.aggregation")
    if metric_type == "scalar":
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"{path}: scalar metrics require sum or mean")
    elif metric_type == "distribution":
        if aggregation != "pool":
            raise ValueError(f"{path}: distribution metrics require pool")
    else:
        raise ValueError(f"{path}.type must be scalar or distribution")

    decimals = require_nonnegative_int32(value["decimals"], f"{path}.decimals")

    definition: dict[str, Any] = {
        "name": _nonblank_string(value["name"], f"{path}.name"),
        "label": _nonblank_string(value["label"], f"{path}.label"),
        "type": metric_type,
        "decimals": decimals,
        "aggregation": aggregation,
    }
    if "unit" in value:
        definition["unit"] = _nonblank_string(value["unit"], f"{path}.unit")

    # This also enforces the JCS number/string domain on the normalized value.
    canonical_bytes(definition)
    return definition


def _nonblank_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or is_blank(value):
        raise ValueError(f"{path} must be a nonblank string")
    return value
