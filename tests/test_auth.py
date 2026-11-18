"""Verifier-only deployment state and strict bearer authentication."""

import hashlib

import pytest
from histoforge.cloud import _auth
from histoforge.cloud._errors import GatewayFailure
from tests.auth_fixtures import provision_legacy_state


def test_provision_and_rotation_store_only_verifiers():
    store = {}

    first, first_token = provision_legacy_state(store)
    assert store == {_auth.STATE_KEY: first}
    assert first_token not in repr(store)
    assert first["auth_revision"] == 1
    assert (
        first["token_sha256"] == hashlib.sha256(first_token.encode("ascii")).hexdigest()
    )
    assert _auth.matches_local_token(first, first_token)

    second, second_token = _auth.legacy_rotate(store)
    assert second["deployment_id"] == first["deployment_id"]
    assert second["auth_revision"] == 2
    assert not _auth.matches_local_token(second, first_token)
    assert _auth.matches_local_token(second, second_token)


def test_strict_bearer_header_authenticates_current_revision():
    store = {}
    state, token = provision_legacy_state(store)

    assert (
        _auth.require_authenticated_request(
            [(b"authorization", f"Bearer {token}".encode("ascii"))],
            lambda: _auth.legacy_current_state(store),
        )
        == state
    )
    assert (
        _auth.require_authenticated_request(
            [(b"Authorization", f"bearer {token}".encode("ascii"))],
            lambda: _auth.legacy_current_state(store),
        )
        == state
    )


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"authorization", b"Basic value")],
        [(b"authorization", b"Bearer")],
        [(b"authorization", b"Bearer wrong")],
        [(b"authorization", b"Bearer token with-space")],
        [
            (b"authorization", b"Bearer one"),
            (b"authorization", b"Bearer two"),
        ],
        [(b"authorization", b"Bearer \xff")],
    ],
)
def test_missing_malformed_or_duplicate_authorization_fails_closed(headers):
    store = {}
    provision_legacy_state(store)

    with pytest.raises(GatewayFailure) as caught:
        _auth.require_authenticated_request(
            headers, lambda: _auth.legacy_current_state(store)
        )
    assert caught.value.error["code"] == "AUTHENTICATION_REQUIRED"


def test_corrupt_or_absent_gateway_state_never_authenticates():
    for store in ({}, {_auth.STATE_KEY: {"token_sha256": "bad"}}):
        with pytest.raises(GatewayFailure) as caught:
            _auth.require_authenticated_request(
                [(b"authorization", b"Bearer token")],
                lambda: _auth.legacy_current_state(store),
            )
        assert caught.value.error["code"] == "AUTHENTICATION_REQUIRED"


def test_mutation_generation_must_still_match_current_auth_state():
    store = {}
    first, _first_token = provision_legacy_state(store)
    assert (
        _auth.require_current_auth_generation(
            lambda: _auth.legacy_current_state(store),
            first["deployment_id"],
            first["auth_revision"],
        )
        == first
    )

    second, _second_token = _auth.legacy_rotate(store)
    with pytest.raises(GatewayFailure) as stale:
        _auth.require_current_auth_generation(
            lambda: _auth.legacy_current_state(store),
            first["deployment_id"],
            first["auth_revision"],
        )
    assert stale.value.error["code"] == "AUTHENTICATION_REQUIRED"
    assert (
        _auth.require_current_auth_generation(
            lambda: _auth.legacy_current_state(store),
            second["deployment_id"],
            second["auth_revision"],
        )
        == second
    )


def test_gateway_state_is_closed_and_detached():
    value = {
        "deployment_id": "deployment-a",
        "auth_revision": 7,
        "token_sha256": "a" * 64,
    }
    parsed = _auth.parse_gateway_state(value)
    value["deployment_id"] = "changed"
    assert parsed["deployment_id"] == "deployment-a"

    with pytest.raises(ValueError):
        _auth.parse_gateway_state({**parsed, "extra": True})


def test_rotation_cannot_recreate_a_missing_identity():
    store = {}
    with pytest.raises(ValueError, match="identity is missing"):
        _auth.legacy_rotate(store)
    assert store == {}


def test_failed_rotation_preserves_identity_without_a_clear_rpc():
    class FailingWrite(dict):
        def clear(self):
            pytest.fail("rotation must replace one record, never clear the store")

        def __setitem__(self, key, value):
            raise OSError("write unavailable")

    original = {}
    state, token = provision_legacy_state(original)
    store = FailingWrite(original)
    with pytest.raises(OSError, match="write unavailable"):
        _auth.legacy_rotate(store)
    assert _auth.legacy_current_state(store) == state
    assert (
        _auth.authenticate_token(lambda: _auth.legacy_current_state(store), token)
        == state
    )
