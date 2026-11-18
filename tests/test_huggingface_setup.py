"""Credential provisioning through stdin, with no real HF or Modal mutation."""

import ast
import io
import json
import logging
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
from histoforge import _huggingface_setup as setup
from histoforge import cli

TOKEN = "hf_test_credential_do_not_log"


@pytest.fixture(autouse=True)
def isolated_context(monkeypatch, tmp_path):
    monkeypatch.setattr(cli._deployment, "PATH", tmp_path / "deployment.json")
    monkeypatch.setattr(
        setup._modal_setup, "ensure_environment", lambda value=None: value or "research"
    )


class Response(io.BytesIO):
    status = 200


def _identity(monkeypatch, body=b'{"name":"researcher"}', status=200):
    requests = []

    def open_request(request, timeout):
        requests.append((request, timeout))
        response = Response(body)
        response.status = status
        return response

    def opener(*handlers):
        assert len(handlers) == 2
        assert isinstance(handlers[0], setup._RejectRedirect)
        assert isinstance(handlers[1], setup.urllib.request.HTTPSHandler)
        assert (
            handlers[0].redirect_request(
                None, None, 302, None, None, "https://evil.invalid"
            )
            is None
        )
        return SimpleNamespace(open=open_request)

    monkeypatch.setattr(setup.urllib.request, "build_opener", opener)
    return requests


def _modal(monkeypatch, events, existing=None, fail=False):
    saved = {} if existing is None else dict(existing)

    class AlreadyExistsError(Exception):
        pass

    class NotFoundError(Exception):
        pass

    def create(name, values):
        events.append("create")
        assert name == setup.DEPLOYMENT_CONFIG.huggingface_secret_name
        if fail:
            print(TOKEN)
            print(TOKEN, file=sys.stderr)
            logging.error(TOKEN)
            raise RuntimeError(TOKEN)
        if saved:
            raise AlreadyExistsError(TOKEN)
        saved.update(values)

    def update(values):
        events.append("update")
        saved.update(values)

    def from_name(name):
        assert name == setup.DEPLOYMENT_CONFIG.huggingface_secret_name
        return SimpleNamespace(update=update)

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Secret=SimpleNamespace(
                objects=SimpleNamespace(create=create), from_name=from_name
            ),
            exception=SimpleNamespace(
                AlreadyExistsError=AlreadyExistsError, NotFoundError=NotFoundError
            ),
        ),
    )
    return saved


@pytest.mark.parametrize(
    "existing", [None, {"HF_TOKEN": "previous", "unrelated": "keep"}]
)
def test_verify_then_authenticate_then_create_or_update_without_local_storage(
    monkeypatch, existing, capsys, tmp_path
):
    events = []
    requests = _identity(monkeypatch)
    source = io.StringIO(TOKEN)

    def authenticate(profile, *, interactive):
        assert interactive is False, (
            "Sensitive operation must never start browser login"
        )
        assert requests, "HF must accept the candidate before any Modal operation"
        assert source.read() == "", "Credential input must be fully consumed"
        assert profile == "research-profile"
        events.append("auth")
        return profile

    monkeypatch.setattr(setup._modal_setup, "ensure_auth", authenticate)
    saved = _modal(monkeypatch, events, existing)
    monkeypatch.chdir(tmp_path)
    before_environment = dict(setup.os.environ)

    assert setup.save_token("research-profile", source) == (
        "research-profile",
        "research",
    )
    assert saved == {**(existing or {}), "HF_TOKEN": TOKEN}
    assert events == ["auth", "create"] + (["update"] if existing else [])
    assert dict(setup.os.environ) == before_environment
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr() == ("", "")
    request, timeout = requests[0]
    assert request.full_url == "https://huggingface.co/api/whoami-v2"
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert timeout == 30


@pytest.mark.parametrize(
    "token", ["", " ", "a b", "a\nb", "a\rb", "a\x00b", "ü", "a" * 4097]
)
def test_invalid_input_never_contacts_a_provider(monkeypatch, token):
    monkeypatch.setattr(
        setup, "_verify_token", lambda _: pytest.fail("HF was contacted")
    )
    with pytest.raises(setup.TokenSetupFailure) as failure:
        setup.save_token(None, io.StringIO(token))
    assert failure.value.code == "HF_TOKEN_INVALID"


@pytest.mark.parametrize(
    "body,status",
    [
        (b"not JSON", 200),
        (b"{}", 200),
        (b"[]", 200),
        (b'{"name":42}', 200),
        (b'{"name":""}', 200),
        (b"x" * (64 * 1024 + 1), 200),
        (b'{"name":"ok"}', 201),
    ],
)
def test_invalid_identity_does_not_authenticate_or_change_modal(
    monkeypatch, body, status
):
    _identity(monkeypatch, body, status)
    monkeypatch.setattr(
        setup._modal_setup,
        "ensure_auth",
        lambda *a, **kw: pytest.fail("Modal was contacted"),
    )
    with pytest.raises(setup.TokenSetupFailure) as failure:
        setup.save_token(None, io.StringIO(TOKEN))
    assert failure.value.code == "HF_VERIFICATION_FAILED"


@pytest.mark.parametrize("status", [301, 302, 307, 308, 401, 403, 429, 500])
def test_http_failure_is_closed_and_preserves_saved_token(monkeypatch, status, capsys):
    def reject(*args, **kwargs):
        raise urllib.error.HTTPError(
            setup._IDENTITY_URL, status, TOKEN, {}, io.BytesIO(TOKEN.encode())
        )

    monkeypatch.setattr(
        setup.urllib.request, "build_opener", lambda *args: SimpleNamespace(open=reject)
    )
    monkeypatch.setattr(
        setup._modal_setup,
        "ensure_auth",
        lambda *a, **kw: pytest.fail("Modal was contacted"),
    )
    with pytest.raises(setup.TokenSetupFailure) as failure:
        setup.save_token(None, io.StringIO(TOKEN))
    assert failure.value.code == (
        "HF_TOKEN_REJECTED" if status in {401, 403} else "HF_VERIFICATION_FAILED"
    )
    assert TOKEN not in str(failure.value)
    assert capsys.readouterr() == ("", "")


def test_sdk_failure_does_not_echo_token_or_change_logging_configuration(
    monkeypatch, capsys, caplog
):
    _identity(monkeypatch)
    monkeypatch.setattr(
        setup._modal_setup, "ensure_auth", lambda profile, **kw: profile
    )
    _modal(monkeypatch, [], fail=True)
    previous = logging.root.manager.disable
    with pytest.raises(setup.TokenSetupFailure) as failure:
        setup.save_token("lab", io.StringIO(TOKEN))
    assert failure.value.code == "HF_SECRET_SAVE_FAILED"
    assert TOKEN not in str(failure.value)
    assert capsys.readouterr() == ("", "")
    assert TOKEN not in caplog.text
    assert logging.root.manager.disable == previous


def test_login_is_separate_and_keeps_manual_instructions_visible(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["histoforge", "login", "--json", "-p", "lab"])

    def authenticate(profile):
        print("https://modal.com/login/fixture", file=sys.stderr)
        print("Enter this code: ABCD", file=sys.stderr)
        return profile

    monkeypatch.setattr(cli._modal_setup, "ensure_auth", authenticate)
    cli.main()
    stdout, stderr = capsys.readouterr()
    assert json.loads(stdout) == {
        "status": "authenticated",
        "profile": "lab",
        "environment": "research",
    }
    assert "https://modal.com/login/fixture" in stderr
    assert "Enter this code: ABCD" in stderr


def test_auth_failure_does_not_save_or_expose_diagnostics(monkeypatch, capsys):
    _identity(monkeypatch)

    def reject(*args, **kwargs):
        raise RuntimeError(TOKEN)

    monkeypatch.setattr(setup._modal_setup, "ensure_auth", reject)
    monkeypatch.setattr(setup, "_save_secret", lambda _: pytest.fail("Secret changed"))
    with pytest.raises(setup.TokenSetupFailure) as failure:
        setup.save_token("lab", io.StringIO(TOKEN))
    assert failure.value.code == "MODAL_AUTH_FAILED"
    assert TOKEN not in str(failure.value)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("bound", [False, True])
def test_missing_secret_environment_preserves_target_failure_and_closed_output(
    monkeypatch, capsys, bound
):
    _identity(monkeypatch)
    events = []
    saved = _modal(monkeypatch, events, existing={"HF_TOKEN": "previous"})
    monkeypatch.setattr(
        setup._modal_setup, "ensure_auth", lambda profile, **kw: profile
    )

    def missing(_name):
        print(TOKEN)
        print(TOKEN, file=sys.stderr)
        raise sys.modules["modal"].exception.NotFoundError(TOKEN)

    monkeypatch.setattr(setup._modal_setup, "ensure_environment", missing)
    with pytest.raises(setup.TokenSetupFailure) as failure:
        setup.save_token(
            "lab",
            io.StringIO(TOKEN),
            environment="missing",
            verify_target=(lambda _: pytest.fail("identity read ran"))
            if bound
            else None,
        )
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert events == [] and saved == {"HF_TOKEN": "previous"}
    assert TOKEN not in str(failure.value)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("fails", [False, True])
def test_cli_uses_stdin_and_emits_only_safe_result(monkeypatch, capsys, fails):
    monkeypatch.setattr(
        sys, "argv", ["histoforge", "huggingface-token", "--json", "-p", "lab"]
    )
    source = io.StringIO(TOKEN)
    monkeypatch.setattr(sys, "stdin", source)

    def save(profile, supplied, **kwargs):
        assert profile == "lab" and supplied is source
        if fails:
            raise RuntimeError(TOKEN)
        return profile, "research"

    monkeypatch.setattr(setup, "save_token", save)
    if fails:
        with pytest.raises(SystemExit) as exit_info:
            cli.main()
        assert exit_info.value.code == 1
    else:
        cli.main()
    stdout, stderr = capsys.readouterr()
    assert TOKEN not in stdout + stderr
    result = json.loads(stdout)
    assert result["status"] == ("error" if fails else "saved")
    if not fails:
        assert result == {
            "status": "saved",
            "profile": "lab",
            "environment": "research",
        }


def test_module_lane_restarts_after_each_input_for_current_secret():
    from histoforge.cloud.modules import mutation

    tree = ast.parse(Path(mutation.__file__).read_text())
    lane = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ModuleMutationLane"
    )
    options = {
        kw.arg: ast.literal_eval(kw.value)
        for kw in lane.decorator_list[0].keywords
        if kw.arg in {"single_use_containers", "max_containers", "retries"}
    }
    assert options == {"single_use_containers": True, "max_containers": 1, "retries": 0}
    assert ast.literal_eval(lane.decorator_list[1].keywords[0].value) == 1
