"""A mounted module view cannot be reloaded while another reader uses it."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest
from histoforge.cloud import _app
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference import activation
from histoforge.cloud.modules import runtime


def test_reload_waits_for_the_complete_previous_read(monkeypatch):
    first_reading = Event()
    release_first = Event()
    second_attempted = Event()
    second_reloaded = Event()
    calls = []
    content = b'{"modules":[{"module_ref":"owner/module"}]}\n'
    current = SimpleNamespace(content=content)

    def reload():
        assert not first_reading.is_set() or release_first.is_set()
        calls.append("reload")
        if len(calls) == 2:
            second_reloaded.set()

    monkeypatch.setattr(_app, "model_volume", SimpleNamespace(reload=reload))
    monkeypatch.setattr(
        runtime, "mounted_module_store", lambda: SimpleNamespace(current=current)
    )

    def first():
        with runtime.current_module_snapshot() as view:
            first_reading.set()
            assert release_first.wait(5)
            return view.content

    def second():
        second_attempted.set()
        with runtime.current_module_snapshot() as view:
            return view.content

    with ThreadPoolExecutor(max_workers=2) as executor:
        initial = executor.submit(first)
        try:
            assert first_reading.wait(5)
            following = executor.submit(second)
            assert second_attempted.wait(5)
            assert not second_reloaded.wait(0.05)
        finally:
            release_first.set()
        assert initial.result(timeout=5) == content
        assert following.result(timeout=5) == content
    assert calls == ["reload", "reload"]


@pytest.mark.parametrize("where", ["reload", "consumer"])
def test_failure_releases_the_mount_and_never_returns_empty_success(monkeypatch, where):
    def reload():
        if where == "reload":
            raise OSError("storage unavailable")

    monkeypatch.setattr(_app, "model_volume", SimpleNamespace(reload=reload))
    monkeypatch.setattr(
        runtime, "mounted_module_store", lambda: SimpleNamespace(current=object())
    )
    with pytest.raises(OSError, match="storage unavailable"):
        with runtime.current_module_snapshot():
            raise OSError("storage unavailable")
    assert runtime._module_volume_lock.acquire(timeout=1)
    runtime._module_volume_lock.release()


@pytest.mark.parametrize("read_failure", [None, OSError("checkpoint unavailable")])
def test_checkpoint_bytes_are_read_before_releasing_the_mount(
    monkeypatch, tmp_path, read_failure
):
    import safetensors.torch

    active = False
    events = []
    current = object()
    path = tmp_path / "weights.safetensors"
    state_dict = {"test": object()}
    network = object()

    @contextmanager
    def snapshot():
        nonlocal active
        active = True
        events.append("enter")
        try:
            yield current
        finally:
            active = False
            events.append("exit")

    def resolve(view, *_args):
        assert active and view is current
        events.append("resolve")
        return {"model": {"params": {"test": True}}}, path

    def read(file):
        assert active and file == str(path)
        events.append("read")
        if read_failure is not None:
            raise read_failure
        return state_dict

    def build(params, state):
        assert not active and params == {"test": True} and state is state_dict
        events.append("build")
        return network

    monkeypatch.setattr(runtime, "current_module_snapshot", snapshot)
    monkeypatch.setattr(activation, "resolve_installed_model_activation", resolve)
    monkeypatch.setattr(safetensors.torch, "load_file", read)
    if read_failure is None:
        assert (
            activation.activate_model(
                "owner/module", "model", "c" * 64, "a" * 40, "test", build
            )["network"]
            is network
        )
        assert events == ["enter", "resolve", "read", "exit", "build"]
    else:
        with pytest.raises(GatewayFailure) as failure:
            activation.activate_model(
                "owner/module", "model", "c" * 64, "a" * 40, "test", build
            )
        assert failure.value.error["code"] == "SERVICE_UNAVAILABLE"
        assert events == ["enter", "resolve", "read", "exit"]
    assert not active
