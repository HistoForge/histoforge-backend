"""Exact local Modal-profile selection before importing the SDK."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import toml
from histoforge import _modal_setup


def _write_config(path, value):
    path.write_text(toml.dumps(value), encoding="utf-8")


def _isolated_environment(tmp_path, monkeypatch):
    path = tmp_path / ".modal.toml"
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(path))
    for name in ("MODAL_PROFILE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_explicit_profile_does_not_confuse_profile_with_workspace(
    tmp_path, monkeypatch
):
    path = _isolated_environment(tmp_path, monkeypatch)
    _write_config(
        path,
        {
            "research-lab": {
                "token_id": "id",
                "token_secret": "secret",
                "active": True,
            },
            "other": {"token_id": "other-id", "token_secret": "other-secret"},
        },
    )
    monkeypatch.setattr(
        _modal_setup.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("setup ran")),
    )

    assert _modal_setup.ensure_auth("research-lab") == "research-lab"
    assert os.environ["MODAL_PROFILE"] == "research-lab"


def test_credentials_from_another_profile_do_not_satisfy_selected_profile(
    tmp_path, monkeypatch
):
    path = _isolated_environment(tmp_path, monkeypatch)
    _write_config(
        path,
        {"other": {"token_id": "other-id", "token_secret": "other-secret"}},
    )
    commands = []

    def setup(command, check, stdin, stdout, env):
        assert stdin == _modal_setup.subprocess.DEVNULL
        assert stdout is _modal_setup.sys.stderr
        assert env == {**os.environ, "PYTHONIOENCODING": "utf-8"}
        commands.append((command, check))
        _write_config(
            path,
            {
                "other": {"token_id": "other-id", "token_secret": "other-secret"},
                "research-lab": {
                    "token_id": "new-id",
                    "token_secret": "new-secret",
                    "active": True,
                },
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_modal_setup.subprocess, "run", setup)

    assert _modal_setup.ensure_auth("research-lab") == "research-lab"
    assert commands == [
        (
            [
                _modal_setup.sys.executable,
                "-m",
                "modal",
                "setup",
                "--profile",
                "research-lab",
            ],
            True,
        )
    ]


def test_browser_setup_without_override_reloads_the_new_active_profile(
    tmp_path, monkeypatch
):
    path = _isolated_environment(tmp_path, monkeypatch)
    commands = []

    def setup(command, check, stdin, stdout, env):
        assert stdin == _modal_setup.subprocess.DEVNULL
        assert stdout is _modal_setup.sys.stderr
        assert env == {**os.environ, "PYTHONIOENCODING": "utf-8"}
        commands.append((command, check))
        _write_config(
            path,
            {
                "workspace-generated-name": {
                    "token_id": "new-id",
                    "token_secret": "new-secret",
                    "active": True,
                }
            },
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_modal_setup.subprocess, "run", setup)

    assert _modal_setup.ensure_auth() == "workspace-generated-name"
    assert os.environ["MODAL_PROFILE"] == "workspace-generated-name"
    assert commands == [([_modal_setup.sys.executable, "-m", "modal", "setup"], True)]


def test_noninteractive_selection_never_launches_login(tmp_path, monkeypatch):
    _isolated_environment(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _modal_setup.subprocess, "run", lambda *args, **kwargs: pytest.fail("login ran")
    )
    with pytest.raises(RuntimeError, match="Modal login is required"):
        _modal_setup.ensure_auth("lab", interactive=False)


@pytest.mark.parametrize("inherited_encoding", [None, "cp1252", "ascii"])
def test_browser_child_uses_utf8_without_changing_parent(
    tmp_path, monkeypatch, inherited_encoding
):
    path = _isolated_environment(tmp_path, monkeypatch)
    if inherited_encoding is None:
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    else:
        monkeypatch.setenv("PYTHONIOENCODING", inherited_encoding)
    parent_environment = dict(os.environ)
    run_child = subprocess.run

    def setup(command, check, stdin, stdout, env):
        assert command == [sys.executable, "-m", "modal", "setup"]
        assert check and stdin == subprocess.DEVNULL and stdout is sys.stderr
        assert env == {**parent_environment, "PYTHONIOENCODING": "utf-8"}
        # Exercise real Python pipe encoding with harmless Unicode, not Modal,
        # browser authorization or a network request.
        child = run_child(
            [
                sys.executable,
                "-c",
                "import sys; print('█░'); print('μm²', file=sys.stderr)",
            ],
            env=env,
            capture_output=True,
            check=True,
        )
        assert child.stdout.decode("utf-8").strip() == "█░"
        assert child.stderr.decode("utf-8").strip() == "μm²"
        _write_config(
            path,
            {
                "lab": {
                    "active": True,
                    "token_id": "test-id",
                    "token_secret": "test-secret",
                }
            },
        )
        return child

    monkeypatch.setattr(_modal_setup.subprocess, "run", setup)
    assert _modal_setup.ensure_auth() == "lab"
    # Selecting the newly authenticated profile is the existing parent mutation;
    # the child's encoding override must not leak into it.
    assert dict(os.environ) == {**parent_environment, "MODAL_PROFILE": "lab"}
