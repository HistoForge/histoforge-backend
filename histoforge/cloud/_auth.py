"""Deployment-bound gateway state and verifier-only bearer authentication."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_sha256,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure

STATE_KEY = "state"
_STATE_FIELDS = {"deployment_id", "auth_revision", "token_sha256"}


def parse_gateway_state(value: Any) -> dict[str, Any]:
    """Validate the exact replaceable server-side state record."""

    if not isinstance(value, Mapping):
        raise ValueError("gateway state must be an object")
    actual = set(value)
    if actual != _STATE_FIELDS:
        raise ValueError(
            "gateway state fields mismatch: "
            f"missing={sorted(_STATE_FIELDS - actual)}, "
            f"unknown={sorted(actual - _STATE_FIELDS)}"
        )
    return canonical_tree(
        {
            "deployment_id": require_nonblank_string(
                value["deployment_id"], "deployment_id"
            ),
            "auth_revision": require_nonnegative_safe_integer(
                value["auth_revision"], "auth_revision"
            ),
            "token_sha256": require_sha256(value["token_sha256"], "token_sha256"),
        }
    )


def current_state(store: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the current valid state, or ``None`` when unprovisioned."""

    value = store.get(STATE_KEY)
    return None if value is None else parse_gateway_state(value)


def provision_new(store: MutableMapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Create a new deployment identity and its first local-only raw token."""

    token = secrets.token_urlsafe(32)
    state = _state(str(uuid.uuid4()), 1, token)
    _replace(store, state)
    return state, token


def rotate(store: MutableMapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Replace the verifier and increment the existing deployment revision."""

    previous = current_state(store)
    if previous is None:
        raise ValueError(
            "gateway identity is missing; reconnect cannot create a deployment"
        )
    token = secrets.token_urlsafe(32)
    state = _state(previous["deployment_id"], previous["auth_revision"] + 1, token)
    _replace(store, state)
    return state, token


def matches_local_token(state: Mapping[str, Any], token: str) -> bool:
    """Return whether a raw local token matches one validated server verifier."""

    parsed = parse_gateway_state(state)
    return hmac.compare_digest(parsed["token_sha256"], _token_digest(token))


def authenticate_token(store: Mapping[str, Any], token: str) -> dict[str, Any] | None:
    """Return authenticated state, or ``None`` without exposing match detail."""

    try:
        state = current_state(store)
        if state is None or not token:
            return None
        return state if matches_local_token(state, token) else None
    except ValueError:
        return None


def require_authenticated_request(
    headers: Iterable[tuple[bytes, bytes]], store: Mapping[str, Any]
) -> dict[str, Any]:
    """Require exactly one syntactically valid bearer header and matching state."""

    values = [value for name, value in headers if name.lower() == b"authorization"]
    token = _bearer_token(values)
    state = authenticate_token(store, token) if token is not None else None
    if state is None:
        raise GatewayFailure(
            GatewayErrorCode.AUTHENTICATION_REQUIRED,
            "a valid deployment-bound bearer credential is required",
        )
    return state


def require_current_auth_generation(
    store: Mapping[str, Any], deployment_id: str, auth_revision: int
) -> dict[str, Any]:
    """Require that an earlier authenticated mutation still belongs to current auth."""

    expected_id = require_nonblank_string(deployment_id, "deployment_id")
    expected_revision = require_nonnegative_safe_integer(auth_revision, "auth_revision")
    try:
        state = current_state(store)
    except ValueError:
        state = None
    if (
        state is None
        or state["deployment_id"] != expected_id
        or state["auth_revision"] != expected_revision
    ):
        raise GatewayFailure(
            GatewayErrorCode.AUTHENTICATION_REQUIRED,
            "the authenticated deployment generation is no longer current",
        )
    return state


def _state(deployment_id: str, auth_revision: int, token: str) -> dict[str, Any]:
    return parse_gateway_state(
        {
            "deployment_id": deployment_id,
            "auth_revision": auth_revision,
            "token_sha256": _token_digest(token),
        }
    )


def _replace(store: MutableMapping[str, Any], state: dict[str, Any]) -> None:
    # This is a Modal-authorized administrative operation. Concurrent rotations
    # may race, but only the final verifier authenticates and only a matching
    # capability probe may publish a local credential.
    # One Dict value is the whole identity. Never clear it before replacement:
    # a failed second RPC would otherwise destroy the deployment identity.
    store[STATE_KEY] = state


def _token_digest(token: str) -> str:
    if (
        not isinstance(token, str)
        or not token
        or any(ord(character) <= 0x20 or ord(character) >= 0x7F for character in token)
    ):
        raise ValueError("bearer token must be nonempty visible ASCII without spaces")
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _bearer_token(values: list[bytes]) -> str | None:
    if len(values) != 1:
        return None
    try:
        text = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    if " " not in text:
        return None
    scheme, token = text.split(" ", 1)
    if scheme.lower() != "bearer" or not token or " " in token or "\t" in token:
        return None
    return token
