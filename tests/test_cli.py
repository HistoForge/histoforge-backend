"""Deployment-handshake CLI orchestration without network calls."""

import http.server
import json
import sys
import threading
import urllib.error
from types import ModuleType, SimpleNamespace

import modal
import pytest
from histoforge import __version__, _deployment, _modal_setup, cli


def _capabilities(deployment_id="deployment-a", auth_revision=4):
    return {
        "deployment_id": deployment_id,
        "auth_revision": auth_revision,
        "api_major": 1,
        "gateway_build": {"version": __version__, "artifact_digest": "a" * 64},
    }


def test_publish_deployed_credential_only_after_matching_authenticated_probe(
    tmp_path, monkeypatch
):
    path = tmp_path / "deployment.json"
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_deployment, "DIR", path.parent)
    monkeypatch.setattr(cli, "_capabilities", lambda _url, _token: _capabilities())

    value = cli._publish_deployed_credential(
        "https://research--histoforge-gateway-fastapi-app.modal.run",
        {
            "deployment_id": "deployment-a",
            "auth_revision": 4,
            "token_sha256": "a" * 64,
        },
        "secret-token",
        "research-lab",
        expected_artifact_digest="a" * 64,
        environment="research",
    )

    assert _deployment.read() == value
    assert value["modal_profile"] == "research-lab"
    assert value["modal_app_name"] == "histoforge-gateway"
    assert value["token"] == "secret-token"
    assert "token" not in cli._public(value)


def test_authenticated_capability_probe_never_follows_redirect():
    received_authorization = []

    class Sink(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            received_authorization.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, _format, *_args):
            pass

    sink = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Sink)

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{sink.server_port}/captured",
            )
            self.end_headers()

        def log_message(self, _format, *_args):
            pass

    redirect = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (sink, redirect)
    ]
    for thread in threads:
        thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as failure:
            cli._capabilities(
                f"http://127.0.0.1:{redirect.server_port}", "secret-token"
            )
        assert failure.value.code == 302
        assert received_authorization == []
    finally:
        redirect.shutdown()
        sink.shutdown()
        redirect.server_close()
        sink.server_close()


@pytest.mark.parametrize(
    ("capability_field", "replacement"),
    [("deployment_id", "other"), ("auth_revision", 5)],
)
def test_mismatched_probe_never_replaces_local_credential(
    tmp_path, monkeypatch, capability_field, replacement
):
    path = tmp_path / "deployment.json"
    path.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_deployment, "DIR", path.parent)
    response = _capabilities()
    response[capability_field] = replacement
    monkeypatch.setattr(cli, "_capabilities", lambda _url, _token: response)

    with pytest.raises(RuntimeError, match="different"):
        cli._publish_deployed_credential(
            "https://research--histoforge-gateway-fastapi-app.modal.run",
            {
                "deployment_id": "deployment-a",
                "auth_revision": 4,
                "token_sha256": "a" * 64,
            },
            "secret-token",
            "research-lab",
            expected_artifact_digest="a" * 64,
            environment="research",
        )

    assert path.read_text(encoding="utf-8") == "existing"


def test_deployed_gateway_lookup_distinguishes_only_absence(monkeypatch):
    monkeypatch.setattr(cli, "_gateway_url", lambda: "https://gateway.example")
    assert cli._deployed_gateway_exists()

    monkeypatch.setattr(
        cli,
        "_gateway_url",
        lambda: (_ for _ in ()).throw(modal.exception.NotFoundError("not deployed")),
    )
    assert not cli._deployed_gateway_exists()

    monkeypatch.setattr(
        cli,
        "_gateway_url",
        lambda: (_ for _ in ()).throw(RuntimeError("lookup failed")),
    )
    with pytest.raises(RuntimeError, match="lookup failed"):
        cli._deployed_gateway_exists()


def test_missing_training_lane_is_not_silently_skipped(monkeypatch):
    monkeypatch.setattr(
        modal.Cls,
        "from_name",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            modal.exception.NotFoundError("not deployed")
        ),
    )

    with pytest.raises(modal.exception.NotFoundError):
        cli._await_training_state_lane()


def test_connect_reports_only_confirmed_gateway_absence_with_stable_code(
    monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["histoforge", "connect", "--json"])
    monkeypatch.setattr(_modal_setup, "ensure_auth", lambda _profile: "research-lab")
    monkeypatch.setattr(
        _modal_setup, "ensure_environment", lambda _environment=None: "research"
    )
    monkeypatch.setattr(
        cli,
        "_gateway_url",
        lambda: (_ for _ in ()).throw(modal.exception.NotFoundError("not deployed")),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "error",
        "code": "GATEWAY_NOT_FOUND",
        "error": "No HistoForge gateway exists for this Modal profile.",
    }


def test_self_check_constructs_deployment_graph_and_reports_runtime(monkeypatch):
    fake_deploy = ModuleType("histoforge.cloud.deploy")
    fake_deploy.app = SimpleNamespace(registered_classes={})
    monkeypatch.setitem(sys.modules, "histoforge.cloud.deploy", fake_deploy)
    emitted = {}
    monkeypatch.setattr(cli, "_emit", lambda _args, payload: emitted.update(payload))

    cli.cmd_self_check(SimpleNamespace(json=True))

    assert emitted["status"] == "ok"
    assert emitted["version"] == __version__
    assert emitted["python"] == (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def test_deployment_graph_rejects_invalid_modal_parameter_order(monkeypatch):
    class InvalidParameters:
        optional: str = modal.parameter(default="")
        required: str = modal.parameter()

    deployed_class = SimpleNamespace(_get_user_cls=lambda: InvalidParameters)
    fake_deploy = ModuleType("histoforge.cloud.deploy")
    fake_deploy.app = SimpleNamespace(
        registered_classes={"InvalidParameters": deployed_class}
    )
    monkeypatch.setitem(sys.modules, "histoforge.cloud.deploy", fake_deploy)

    with pytest.raises(
        RuntimeError,
        match=(
            "Modal class 'InvalidParameters' has an invalid parameter signature: "
            "non-default argument follows default argument"
        ),
    ):
        cli._validated_deployment_app()


def test_artifact_lane_preflight_propagates_transport_failure(monkeypatch):
    def hydrate():
        raise OSError("lookup unavailable")

    monkeypatch.setattr(
        modal.Cls, "from_name", lambda *_args: SimpleNamespace(hydrate=hydrate)
    )
    with pytest.raises(OSError, match="lookup unavailable"):
        cli._require_artifact_lifecycle_lane()


def test_status_requires_authenticated_identity_match(monkeypatch):
    local = {
        "deployment_id": "deployment-a",
        "gateway_url": "https://gateway.example",
        "modal_profile": "research-lab",
        "modal_app_name": "histoforge-gateway",
        "auth_revision": 4,
        "token": "secret",
    }
    emitted = {}
    monkeypatch.setattr(_deployment, "read", lambda: local)
    monkeypatch.setattr(cli, "_health", lambda _url: {"api_major": 1, "status": "ok"})
    monkeypatch.setattr(
        cli,
        "_capabilities",
        lambda _url, _token: _capabilities(auth_revision=5),
    )
    monkeypatch.setattr(cli, "_emit", lambda _args, payload: emitted.update(payload))

    cli.cmd_status(SimpleNamespace(json=True))

    assert emitted["reachable"] is True
    assert emitted["connected"] is False
    assert emitted["auth_revision"] == 4


def test_status_projects_invalid_local_credential_as_replaceable_setup_state(
    monkeypatch,
):
    emitted = {}
    monkeypatch.setattr(
        _deployment,
        "read",
        lambda: (_ for _ in ()).throw(ValueError("invalid credential")),
    )
    monkeypatch.setattr(cli, "_emit", lambda _args, payload: emitted.update(payload))

    cli.cmd_status(SimpleNamespace(json=True))

    assert emitted["status"] == "not_configured"
    assert "invalid" in emitted["hint"]
    assert "connect" in emitted["hint"]
