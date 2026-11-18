"""Only committed cache data or the serialized owner supplies live auth state."""

import ast
import inspect
from types import SimpleNamespace

import modal
import pytest
from histoforge.cloud import _auth
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.gateway_state_client import GatewayStateClient


@pytest.fixture
def transport(monkeypatch):
    state = SimpleNamespace(cache={}, record=None, calls=[], error=None)

    def mapping(name, *, create_if_missing):
        assert name == DEPLOYMENT_CONFIG.gateway_state_cache_dict_name
        assert create_if_missing is True
        return state.cache

    def hydrate():
        if state.error is not None:
            raise state.error

    def load():
        state.calls.append("load")
        return state.record

    def rotate(*args):
        state.calls.append(("rotate", args))
        return state.record

    lane = SimpleNamespace(
        load_current=SimpleNamespace(remote=load),
        rotate=SimpleNamespace(remote=rotate),
    )

    class RemoteClass:
        def __call__(self):
            return lane

        def hydrate(self):
            hydrate()

    def lookup(app, name):
        assert app == DEPLOYMENT_CONFIG.gateway_app_name
        assert name == DEPLOYMENT_CONFIG.gateway_state_class_name
        return RemoteClass()

    monkeypatch.setattr(modal.Dict, "from_name", mapping)
    monkeypatch.setattr(modal.Cls, "from_name", lookup)
    return state


def test_cache_hit_is_detached_and_does_not_call_state_owner(transport):
    transport.record = _auth.initial_state()
    transport.cache[_auth.STATE_KEY] = transport.record
    current = GatewayStateClient().current()
    current["deployment_id"] = "changed"
    assert transport.cache[_auth.STATE_KEY] == transport.record
    assert transport.calls == []


@pytest.mark.parametrize("cache", [{}, {_auth.STATE_KEY: {"bad": True}}])
def test_cache_miss_uses_owner_and_missing_record_never_uses_legacy(transport, cache):
    transport.cache = cache
    client = GatewayStateClient()
    assert client.current() is None
    transport.record = _auth.initial_state()
    assert client.current() == transport.record
    assert transport.calls == ["load", "load"]


@pytest.mark.parametrize("error", [OSError("offline"), PermissionError("denied")])
def test_lookup_failure_never_enables_legacy_fallback(transport, error):
    transport.error = error
    with pytest.raises(type(error)):
        GatewayStateClient().available()


def test_only_confirmed_missing_class_enables_bridge(transport):
    transport.error = modal.exception.NotFoundError("class absent")
    assert GatewayStateClient().available() is False


def test_rotation_transmits_only_hash_and_expected_identity(transport):
    transport.record = _auth.initial_state() | {"auth_revision": 1}
    verifier = _auth.token_digest("raw-credential")
    assert (
        GatewayStateClient().rotate(
            verifier,
            expected_deployment_id=transport.record["deployment_id"],
            expected_revision=0,
        )
        == transport.record
    )
    assert transport.calls == [
        ("rotate", (verifier, transport.record["deployment_id"], 0))
    ]
    assert "raw-credential" not in repr(transport.calls)


def test_rotation_rejects_foreign_response_even_at_same_revision(transport):
    transport.record = _auth.initial_state()
    with pytest.raises(ValueError, match="different deployment identity"):
        GatewayStateClient().rotate(
            _auth.token_digest("raw-credential"), expected_deployment_id="another"
        )


def test_rotation_rejects_a_response_without_the_expected_successor_revision(transport):
    transport.record = _auth.initial_state()
    with pytest.raises(ValueError, match="different auth revision"):
        GatewayStateClient().rotate(
            _auth.token_digest("raw-credential"),
            expected_deployment_id=transport.record["deployment_id"],
            expected_revision=0,
        )


def test_deployment_registers_one_serialized_gateway_state_owner():
    from histoforge.cloud import deploy, gateway_state_service

    assert DEPLOYMENT_CONFIG.gateway_state_class_name in deploy.app.registered_classes
    tree = ast.parse(inspect.getsource(gateway_state_service))
    lane = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    decorators = {
        call.func.attr: {item.arg: item.value for item in call.keywords}
        for call in lane.decorator_list
        if isinstance(call, ast.Call)
    }
    assert ast.literal_eval(decorators["cls"]["max_containers"]) == 1
    assert ast.literal_eval(decorators["cls"]["retries"]) == 0
    assert ast.literal_eval(decorators["concurrent"]["max_inputs"]) == 1
