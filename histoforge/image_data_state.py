"""Validate and encode the action commit witness stored in QuPath ImageData.

The extension uses this record to resolve an ambiguous save outcome.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    parse_json,
    require_canonical_uuid,
    require_int32,
)

ACTION_COMMIT_WITNESS_PROPERTY_KEY = "histoforge:action-commit-witness"
ACTION_COMMIT_WITNESS_SCHEMA_VERSION = 1

_ACTION_COMMIT_WITNESS_FIELDS = {
    "schema_version",
    "action_id",
    "handle_generation_id",
}


def validate_action_commit_witness(value: Any) -> dict[str, Any]:
    """Validate one exact schema-1 large-action save witness."""

    witness = _require_exact_object(
        value,
        _ACTION_COMMIT_WITNESS_FIELDS,
        "action commit witness",
    )
    schema_version = require_int32(witness["schema_version"], "schema_version")
    if schema_version != ACTION_COMMIT_WITNESS_SCHEMA_VERSION:
        raise ValueError("unsupported action commit witness schema_version")
    return canonical_tree(
        {
            "schema_version": schema_version,
            "action_id": require_canonical_uuid(witness["action_id"], "action_id"),
            "handle_generation_id": require_canonical_uuid(
                witness["handle_generation_id"], "handle_generation_id"
            ),
        }
    )


def encode_action_commit_witness(value: Any) -> str:
    """Encode one validated witness as the exact String stored in ImageData."""

    return canonical_bytes(validate_action_commit_witness(value)).decode("utf-8")


def decode_action_commit_witness_property(value: Any) -> dict[str, Any]:
    """Decode a canonical ImageData String property and reject alternate JSON."""

    if not isinstance(value, str):
        raise ValueError("action commit witness property must be a string")
    witness = validate_action_commit_witness(parse_json(value))
    if encode_action_commit_witness(witness) != value:
        raise ValueError("action commit witness property must be canonical JSON")
    return witness


def _require_exact_object(
    value: Any,
    fields: set[str],
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return value
