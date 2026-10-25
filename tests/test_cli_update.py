"""Confirmed cloud replacement: drain, fence, redeploy, verify, publish credentials."""

import contextlib
import json
import os
import subprocess
import sys
import urllib.error
import warnings
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest
from histoforge import __version__, _cloud_update, _deployment, _modal_setup, cli
from histoforge.cloud import _auth, runtime_identity


@pytest.fixture
def deployment(monkeypatch):
    store = {}
    original, token = _auth.provision_new(store)
    url = "https://research--histoforge-gateway-fastapi-app.modal.run"
    credential = _deployment.build(
        deployment_id=original["deployment_id"],
        gateway_url=url,
        modal_profile="research-lab",
        modal_app_name="histoforge-gateway",
        auth_revision=original["auth_revision"],
        token=token,
    )
    events = []
    state = SimpleNamespace(
        store=store,
        original=original,
        credential=credential,
        events=events,
        exists=True,
        failure=None,
        published=[],
        creates=[],
        args=SimpleNamespace(profile="research-lab", json=True, cancel_work=True),
    )

    def deploy(**kwargs):
        assert os.environ["MODAL_IMAGE_BUILDER_VERSION"] == "2025.06"
        events.append(("deploy", kwargs))
        if state.failure:
            raise state.failure

    def publish(url, current, token, profile, *, expected_artifact_digest):
        assert expected_artifact_digest == "a" * 64
        assert _auth.matches_local_token(current, token)
        state.published.append(current)
        return credential | {"auth_revision": current["auth_revision"], "token": token}

    def stop(url, token, *, wait_for_module_writer):
        events.append(("stop", store[_auth.STATE_KEY]["auth_revision"]))
        assert _auth.matches_local_token(store[_auth.STATE_KEY], token)
        wait_for_module_writer()

    monkeypatch.setattr(_modal_setup, "ensure_auth", lambda profile: profile)
    monkeypatch.setattr(
        cli, "_validated_deployment_app", lambda: SimpleNamespace(deploy=deploy)
    )
    monkeypatch.setattr(runtime_identity, "bundle_artifact_digest", lambda: "a" * 64)
    monkeypatch.setattr(cli, "_deployed_gateway_exists", lambda: state.exists)
    monkeypatch.setattr(cli, "_gateway_url", lambda: url)
    monkeypatch.setattr(cli, "_require_artifact_lifecycle_lane", lambda: None)
    monkeypatch.setattr(
        cli,
        "_gateway_state_dict",
        lambda *, create=False: state.creates.append(create) or store,
    )
    monkeypatch.setattr(_deployment, "read", lambda: credential)
    monkeypatch.setattr(_cloud_update, "stop_cloud_work", stop)
    for name in ("module_mutation", "artifact_lifecycle", "training_state"):
        monkeypatch.setattr(
            cli, f"_await_{name}_lane", lambda name=name: events.append((name, None))
        )
    monkeypatch.setattr(modal, "enable_output", contextlib.nullcontext)
    monkeypatch.setattr(cli, "_publish_local_credential", publish)
    monkeypatch.setattr(
        cli, "_emit", lambda _args, value: events.append(("emit", value["status"]))
    )
    return state


@pytest.mark.parametrize("previous_builder", [None, "PREVIEW"])
def test_update_drains_before_rotation_then_barriers_and_rechecks(
    deployment, monkeypatch, previous_builder
):
    if previous_builder is None:
        monkeypatch.delenv("MODAL_IMAGE_BUILDER_VERSION", raising=False)
    else:
        monkeypatch.setenv("MODAL_IMAGE_BUILDER_VERSION", previous_builder)
    cli._deploy(deployment.args, fresh=False)
    assert deployment.events == [
        ("stop", 1),
        ("module_mutation", None),
        ("module_mutation", None),
        ("artifact_lifecycle", None),
        ("training_state", None),
        ("stop", 2),
        ("module_mutation", None),
        ("deploy", {"strategy": "recreate"}),
        ("emit", "upgraded"),
    ]
    assert (
        deployment.store[_auth.STATE_KEY]["deployment_id"]
        == deployment.original["deployment_id"]
    )
    assert deployment.creates == [False]
    assert len(deployment.published) == 1
    assert os.environ.get("MODAL_IMAGE_BUILDER_VERSION") == previous_builder


def test_update_requires_confirmation_before_authentication(deployment, monkeypatch):
    deployment.args.cancel_work = False
    monkeypatch.setattr(
        _modal_setup, "ensure_auth", lambda _: pytest.fail("not confirmed")
    )
    with pytest.raises(cli.CliFailure) as failure:
        cli._deploy(deployment.args, fresh=False)
    assert failure.value.code == "UPDATE_CONFIRMATION_REQUIRED"


@pytest.mark.parametrize(
    ("fresh", "exists", "code"),
    [
        (True, True, "GATEWAY_EXISTS"),
        (False, False, "GATEWAY_NOT_FOUND"),
    ],
)
def test_setup_never_resets_existing_gateway_and_update_never_creates(
    deployment, fresh, exists, code
):
    deployment.exists = exists
    with pytest.raises(cli.CliFailure) as failure:
        cli._deploy(deployment.args, fresh=fresh)
    assert failure.value.code == code
    assert deployment.store[_auth.STATE_KEY] == deployment.original
    assert deployment.creates == []


@pytest.mark.parametrize("has_identity", [False, True])
def test_setup_retry_reuses_identity_without_a_recovery_journal(
    deployment, has_identity
):
    deployment.exists = False
    if not has_identity:
        deployment.store.clear()
    cli._deploy(deployment.args, fresh=True)
    current = deployment.store[_auth.STATE_KEY]
    if has_identity:
        assert current["deployment_id"] == deployment.original["deployment_id"]
        assert current["auth_revision"] == 2
    else:
        assert current["auth_revision"] == 1
    assert deployment.creates == [True]
    assert deployment.events == [
        ("deploy", {"strategy": "recreate"}),
        ("emit", "deployed"),
    ]


@pytest.mark.parametrize(
    "problem", ["missing", "token", "modal_profile", "deployment_id", "gateway_url"]
)
def test_update_requires_current_local_credential_before_cancellation(
    deployment, monkeypatch, problem
):
    value = None if problem == "missing" else deployment.credential | {problem: "wrong"}
    monkeypatch.setattr(_deployment, "read", lambda: value)
    with pytest.raises(cli.CliFailure) as failure:
        cli._deploy(deployment.args, fresh=False)
    assert failure.value.code == "RECONNECT_REQUIRED"
    assert deployment.events == []
    assert deployment.store[_auth.STATE_KEY] == deployment.original


@pytest.mark.parametrize(
    "failure_at",
    [
        "first_stop",
        "second_stop",
        "module_barrier",
        "deploy",
        "acknowledgement",
        "credential_write",
    ],
)
def test_failed_stop_or_deployment_never_publishes_success_or_resets_identity(
    deployment, monkeypatch, failure_at
):
    calls = []

    def stop(*_args, **_kwargs):
        calls.append(True)
        if (failure_at == "first_stop" and len(calls) == 1) or (
            failure_at == "second_stop" and len(calls) == 2
        ):
            raise RuntimeError("injected stop failure")

    monkeypatch.setattr(_cloud_update, "stop_cloud_work", stop)
    if failure_at == "module_barrier":

        def barrier():
            raise RuntimeError("injected barrier failure")

        monkeypatch.setattr(cli, "_await_module_mutation_lane", barrier)
    if failure_at == "deploy":
        deployment.failure = RuntimeError("injected deploy failure")
    if failure_at in {"acknowledgement", "credential_write"}:

        def publish(*_args, **_kwargs):
            if failure_at == "acknowledgement":
                raise TimeoutError("injected lost post-deployment acknowledgement")
            raise OSError("injected local credential write failure")

        monkeypatch.setattr(cli, "_publish_local_credential", publish)
    before_builder = os.environ.get("MODAL_IMAGE_BUILDER_VERSION")
    with pytest.raises((RuntimeError, OSError), match="injected"):
        cli._deploy(deployment.args, fresh=False)
    assert deployment.published == []
    assert not any(event[0] == "emit" for event in deployment.events)
    assert (
        deployment.store[_auth.STATE_KEY]["deployment_id"]
        == deployment.original["deployment_id"]
    )
    assert deployment.store[_auth.STATE_KEY]["auth_revision"] == (
        1 if failure_at == "first_stop" else 2
    )
    assert deployment.credential["auth_revision"] == 1
    assert os.environ.get("MODAL_IMAGE_BUILDER_VERSION") == before_builder


def test_concurrent_credential_change_aborts_before_own_rotation(
    deployment, monkeypatch
):
    def stop(*_args, **_kwargs):
        _auth.rotate(deployment.store)

    monkeypatch.setattr(_cloud_update, "stop_cloud_work", stop)
    with pytest.raises(RuntimeError, match="access changed"):
        cli._deploy(deployment.args, fresh=False)
    assert deployment.store[_auth.STATE_KEY]["auth_revision"] == 2
    assert deployment.events == []
    assert deployment.published == []


def test_incomplete_recreate_is_not_verified_success(deployment, monkeypatch):
    def deploy(**_kwargs):
        warnings.warn("Some old containers did not all terminate", UserWarning)

    monkeypatch.setattr(
        cli, "_validated_deployment_app", lambda: SimpleNamespace(deploy=deploy)
    )
    with pytest.raises(cli.CliFailure) as failure:
        cli._deploy(deployment.args, fresh=False)
    assert failure.value.code == "DEPLOYMENT_INCOMPLETE"
    assert deployment.published == []


def test_graph_failure_precedes_any_cloud_mutation(deployment, monkeypatch):
    def graph():
        raise ValueError("broken deployment graph")

    monkeypatch.setattr(cli, "_validated_deployment_app", graph)
    with pytest.raises(ValueError, match="broken deployment"):
        cli._deploy(deployment.args, fresh=False)
    assert deployment.events == []
    assert deployment.creates == []


def test_missing_artifact_lane_fails_before_credential_rotation(monkeypatch):
    # Exercise the real preflight independently, without cloud access.
    def hydrate():
        raise modal.exception.NotFoundError("not deployed")

    monkeypatch.setattr(
        modal.Cls, "from_name", lambda *_args: SimpleNamespace(hydrate=hydrate)
    )
    with pytest.raises(cli.CliFailure) as failure:
        cli._require_artifact_lifecycle_lane()
    assert failure.value.code == "DEPLOYMENT_CUTOVER_REQUIRED"


def test_cli_import_does_not_capture_modal_profile_before_login():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import histoforge.cli; assert 'modal' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("version", "digest", "accepted"),
    [
        (__version__, "a" * 64, True),
        (__version__, "b" * 64, False),
        ("99.0.0", "a" * 64, False),
    ],
)
def test_post_update_requires_exact_bundled_artifact_before_writing_credential(
    tmp_path, monkeypatch, version, digest, accepted
):
    path = tmp_path / "deployment.json"
    path.write_text("previous credential", encoding="utf-8")
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_deployment, "DIR", tmp_path)
    current, token = _auth.provision_new({})
    monkeypatch.setattr(
        cli,
        "_capabilities",
        lambda *_: (
            current | {"gateway_build": {"version": version, "artifact_digest": digest}}
        ),
    )
    if accepted:
        cli._publish_local_credential(
            "https://gateway.example",
            current,
            token,
            "lab",
            expected_artifact_digest="a" * 64,
        )
        assert _deployment.read()["deployment_id"] == current["deployment_id"]
    else:
        with pytest.raises(RuntimeError, match="expected bundled backend"):
            cli._publish_local_credential(
                "https://gateway.example",
                current,
                token,
                "lab",
                expected_artifact_digest="a" * 64,
            )
        assert path.read_text(encoding="utf-8") == "previous credential"


@pytest.mark.parametrize(
    ("field", "version"),
    [
        ("finite_inference_schema_version", 1),
        ("finite_inference_schema_version", 3),
        ("durable_model_run_schema_version", 2),
        ("durable_model_run_schema_version", 4),
    ],
)
@pytest.mark.parametrize("after_deployment", [False, True])
def test_reconnect_and_update_reject_mismatched_schemas_before_writing_credential(
    tmp_path, monkeypatch, field, version, after_deployment
):
    path = tmp_path / "deployment.json"
    path.write_text("previous credential", encoding="utf-8")
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_deployment, "DIR", tmp_path)
    current, token = _auth.provision_new({})
    fixture = Path(__file__).parent / "fixtures/contracts/gateway-capabilities-v1.json"
    capabilities = json.loads(fixture.read_text(encoding="utf-8"))["capabilityCases"][
        0
    ]["value"]
    capabilities["deployment_id"] = current["deployment_id"]
    capabilities["auth_revision"] = current["auth_revision"]
    capabilities["gateway_build"]["version"] = __version__
    capabilities["gateway_build"]["artifact_digest"] = "a" * 64
    capabilities[field] = version
    probes = []

    def gateway_request(_url, _token, route):
        probes.append(route)
        return capabilities

    monkeypatch.setattr(_cloud_update, "gateway_request", gateway_request)
    monkeypatch.setattr(
        cli.time, "sleep", lambda _: pytest.fail("schema mismatch must not retry")
    )
    with pytest.raises(ValueError, match="unsupported gateway capability versions"):
        cli._publish_local_credential(
            "https://gateway.example",
            current,
            token,
            "lab",
            expected_artifact_digest="a" * 64 if after_deployment else None,
        )
    assert probes == ["/v1/capabilities"]
    assert path.read_text(encoding="utf-8") == "previous credential"


def test_post_deploy_readiness_recovers_from_a_cold_gateway(monkeypatch):
    calls = []
    expected = {"deployment_id": "same", "auth_revision": 4}

    def capabilities(_url, _token):
        calls.append("probe")
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                "https://gateway.example/v1/capabilities", 500, "cold gateway", {}, None
            )
        return expected

    monkeypatch.setattr(cli, "_capabilities", capabilities)
    monkeypatch.setattr(
        cli.time, "sleep", lambda seconds: calls.append(("wait", seconds))
    )
    assert (
        cli._capabilities_after_deploy("https://gateway.example", "token") is expected
    )
    assert calls == ["probe", ("wait", 2), "probe"]


@pytest.mark.parametrize("code", [401, 403])
def test_post_deploy_readiness_does_not_retry_rejected_credentials(monkeypatch, code):
    calls = []

    def capabilities(_url, _token):
        calls.append("probe")
        raise urllib.error.HTTPError(
            "https://gateway.example/v1/capabilities", code, "rejected", {}, None
        )

    monkeypatch.setattr(cli, "_capabilities", capabilities)
    monkeypatch.setattr(cli.time, "sleep", lambda _: pytest.fail("must not retry"))
    with pytest.raises(urllib.error.HTTPError) as failure:
        cli._capabilities_after_deploy("https://gateway.example", "token")
    assert failure.value.code == code
    assert calls == ["probe"]


def test_post_deploy_readiness_stops_after_four_transient_failures(monkeypatch):
    calls = []

    def capabilities(_url, _token):
        calls.append("probe")
        raise urllib.error.HTTPError(
            "https://gateway.example/v1/capabilities", 500, "unavailable", {}, None
        )

    monkeypatch.setattr(cli, "_capabilities", capabilities)
    monkeypatch.setattr(
        cli.time, "sleep", lambda seconds: calls.append(("wait", seconds))
    )
    with pytest.raises(urllib.error.HTTPError) as failure:
        cli._capabilities_after_deploy("https://gateway.example", "token")
    assert failure.value.code == 500
    assert calls == [
        "probe",
        ("wait", 2),
        "probe",
        ("wait", 2),
        "probe",
        ("wait", 2),
        "probe",
    ]
