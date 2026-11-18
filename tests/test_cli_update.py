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
from histoforge import (
    __version__,
    _cloud_update,
    _deployment,
    _gateway_state_setup,
    _modal_setup,
    cli,
)
from histoforge.cloud import _auth, gateway_state_client, runtime_identity
from histoforge.cloud import capabilities as runtime_capabilities
from tests.auth_fixtures import provision_legacy_state


def _capability_fixture():
    fixture = Path(__file__).parent / "fixtures/contracts/gateway-capabilities-v2.json"
    return json.loads(fixture.read_text())["capabilityCases"][0]["value"]


@pytest.fixture
def deployment(monkeypatch):
    store = {}
    original, token = provision_legacy_state(store)
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
        legacy=False,
        staged=None,
        record_cutover=False,
        capability_overrides={},
        real_publisher=cli._publish_deployed_credential,
        args=SimpleNamespace(profile="research-lab", json=True, cancel_work=True),
        default_environment="research",
        gateway_environment="research",
        selected_environments=[],
    )

    def deploy(**kwargs):
        assert os.environ["MODAL_IMAGE_BUILDER_VERSION"] == "2025.06"
        events.append(("deploy", kwargs))
        if state.failure:
            raise state.failure
        state.exists = True

    def write(value):
        state.credential = value
        state.published.append(value)
        return value

    def publish(url, current, token, profile, *, expected_artifact_digest, environment):
        assert expected_artifact_digest == "a" * 64
        assert _auth.matches_local_token(current, token)
        return write(
            credential
            | {
                "deployment_id": current["deployment_id"],
                "auth_revision": current["auth_revision"],
                "token": token,
                "schema_version": 2,
                "modal_environment": environment,
            }
        )

    def capabilities(_url, supplied_token, path):
        assert path == "/v1/capabilities"
        current = store[_auth.STATE_KEY]
        assert _auth.matches_local_token(current, supplied_token)
        value = _capability_fixture()
        value.update(
            deployment_id=current["deployment_id"],
            auth_revision=current["auth_revision"],
            gateway_build={
                "version": __version__,
                "artifact_digest": "a" * 64,
                "manifest_digest": "b" * 64,
            },
        )
        value.update(state.capability_overrides)
        return value

    def stop(url, token, *, wait_for_module_writer):
        events.append(("stop", store[_auth.STATE_KEY]["auth_revision"]))
        assert _auth.matches_local_token(store[_auth.STATE_KEY], token)
        wait_for_module_writer()

    monkeypatch.setattr(_modal_setup, "ensure_auth", lambda profile: profile)

    def select_environment(environment=None):
        selected = environment or state.default_environment
        monkeypatch.setenv("MODAL_ENVIRONMENT", selected)
        state.selected_environments.append(selected)
        return selected

    monkeypatch.setattr(_modal_setup, "ensure_environment", select_environment)
    monkeypatch.setattr(
        cli, "_validated_deployment_app", lambda: SimpleNamespace(deploy=deploy)
    )
    monkeypatch.setattr(runtime_identity, "bundle_artifact_digest", lambda: "a" * 64)
    monkeypatch.setattr(cli, "_deployed_gateway_exists", lambda: state.exists)

    def gateway_url():
        if not state.exists:
            raise modal.exception.NotFoundError("not deployed")
        if os.environ.get("MODAL_ENVIRONMENT") != state.gateway_environment:
            return "https://unrelated-environment.example"
        return url

    monkeypatch.setattr(cli, "_gateway_url", gateway_url)
    monkeypatch.setattr(cli, "_require_artifact_lifecycle_lane", lambda: None)

    class Client:
        def available(self):
            return not state.legacy

        def current(self):
            return _auth.legacy_current_state(store)

        def rotate(
            self, token_sha256, *, expected_deployment_id=None, expected_revision=None
        ):
            previous = self.current()
            if expected_deployment_id is not None and (
                previous["deployment_id"] != expected_deployment_id
                or (
                    expected_revision is not None
                    and previous["auth_revision"] != expected_revision
                )
            ):
                raise ValueError("gateway access changed during update preparation")
            current = _auth.next_state(previous, token_sha256)
            store[_auth.STATE_KEY] = current
            return current

    class Setup:
        def __init__(self, *, fresh=False):
            state.creates.append(fresh)

        def bootstrap(self):
            if _auth.STATE_KEY not in store:
                store[_auth.STATE_KEY] = _auth.initial_state()

        def legacy_current(self):
            return _auth.legacy_current_state(store)

        def legacy_rotate(self, *, expected_deployment_id=None):
            return _auth.legacy_rotate(
                store, expected_deployment_id=expected_deployment_id
            )

        def prepare_cutover(self, previous):
            staged = state.staged
            if (
                staged is not None
                and staged["deployment_id"] != previous["deployment_id"]
            ):
                raise ValueError("foreign staged identity")
            return staged

        def cut_over(self, previous, staged, verifier):
            if state.record_cutover:
                events.append(("cutover", None))
            base = previous | {
                "auth_revision": max(
                    previous["auth_revision"],
                    0 if staged is None else staged["auth_revision"],
                )
            }
            current = _auth.next_state(base, verifier)
            store[_auth.STATE_KEY] = state.staged = current
            return current

    monkeypatch.setattr(gateway_state_client, "GatewayStateClient", Client)
    monkeypatch.setattr(_gateway_state_setup, "GatewayStateSetup", Setup)
    monkeypatch.setattr(_deployment, "read", lambda: state.credential)
    monkeypatch.setattr(_deployment, "write", write)
    monkeypatch.setattr(_cloud_update, "gateway_request", capabilities)
    monkeypatch.setattr(_cloud_update, "stop_cloud_work", stop)
    for name in ("module_mutation", "artifact_lifecycle", "training_state"):
        monkeypatch.setattr(
            cli, f"_await_{name}_lane", lambda name=name: events.append((name, None))
        )
    monkeypatch.setattr(modal, "enable_output", contextlib.nullcontext)
    monkeypatch.setattr(cli, "_publish_deployed_credential", publish)
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
    assert deployment.creates == []
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
        (False, False, "GATEWAY_TARGET_MISMATCH"),
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
    "problem",
    [
        "missing",
        "token",
        "modal_profile",
        "deployment_id",
        "auth_revision",
        "gateway_url",
    ],
)
def test_update_requires_current_local_credential_before_cancellation(
    deployment, monkeypatch, problem
):
    value = None if problem == "missing" else deployment.credential | {problem: "wrong"}
    monkeypatch.setattr(_deployment, "read", lambda: value)
    with pytest.raises(cli.CliFailure) as failure:
        cli._deploy(deployment.args, fresh=False)
    assert failure.value.code == (
        "GATEWAY_TARGET_MISMATCH"
        if problem in {"modal_profile", "deployment_id", "gateway_url"}
        else "RECONNECT_REQUIRED"
    )
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

        monkeypatch.setattr(cli, "_publish_deployed_credential", publish)
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
        _auth.legacy_rotate(deployment.store)

    monkeypatch.setattr(_cloud_update, "stop_cloud_work", stop)
    with pytest.raises(ValueError, match="access changed"):
        cli._deploy(deployment.args, fresh=False)
    assert deployment.store[_auth.STATE_KEY]["auth_revision"] == 2
    assert deployment.events == []
    assert deployment.published == []


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_identity_never_triggers_setup_or_credential_publication(
    deployment, legacy
):
    deployment.legacy = legacy
    deployment.store.clear()
    with pytest.raises(cli.CliFailure) as caught:
        cli.cmd_connect(deployment.args)
    assert caught.value.code == "IDENTITY_RECOVERY_REQUIRED"
    assert deployment.store == {}
    assert deployment.published == []
    assert deployment.events == []


def test_interrupted_fresh_deployment_can_reconnect_to_same_bootstrap_identity(
    deployment,
):
    deployment.exists = False
    deployment.store.clear()
    deployment.failure = TimeoutError("deploy acknowledgement lost")
    with pytest.raises(TimeoutError):
        cli.cmd_deploy(deployment.args)
    bootstrap = deployment.store[_auth.STATE_KEY]
    assert bootstrap["auth_revision"] == 0
    assert deployment.published == []
    deployment.exists = True
    cli.cmd_connect(deployment.args)
    assert deployment.published[-1]["deployment_id"] == bootstrap["deployment_id"]
    assert deployment.published[-1]["auth_revision"] == 1


def test_update_cutover_advances_past_an_interrupted_staged_revision(deployment):
    deployment.legacy = True
    deployment.record_cutover = True
    deployment.staged = deployment.original | {"auth_revision": 8}
    cli.cmd_upgrade(deployment.args)
    assert deployment.published[-1]["auth_revision"] == 9
    assert (
        deployment.published[-1]["deployment_id"]
        == deployment.original["deployment_id"]
    )
    assert deployment.creates == [False]
    assert deployment.events == [
        ("stop", 1),
        ("module_mutation", None),
        ("cutover", None),
        ("module_mutation", None),
        ("artifact_lifecycle", None),
        ("training_state", None),
        ("stop", 9),
        ("module_mutation", None),
        ("deploy", {"strategy": "recreate"}),
        ("emit", "upgraded"),
    ]


def test_legacy_reconnect_publishes_only_a_matching_same_identity(deployment):
    deployment.legacy = True
    cli.cmd_connect(deployment.args)
    assert (
        deployment.published[-1]["deployment_id"]
        == deployment.original["deployment_id"]
    )
    assert deployment.published[-1]["auth_revision"] == 2
    assert deployment.events == [("emit", "connected")]


@pytest.mark.parametrize(
    ("helper_versions", "gateway_versions"),
    [((4, 3), (3, 2)), ((3, 2), (4, 3)), ((4, 3), (4, 3))],
)
def test_reconnect_saves_verified_login_and_reports_execution_mismatches(
    deployment, monkeypatch, helper_versions, gateway_versions
):
    monkeypatch.setattr(
        runtime_capabilities, "MODULE_SCHEMA_VERSION", helper_versions[0]
    )
    monkeypatch.setattr(
        runtime_capabilities, "FINITE_INFERENCE_SCHEMA_VERSION", helper_versions[1]
    )
    deployment.capability_overrides.update(
        module_schema_version=gateway_versions[0],
        finite_inference_schema_version=gateway_versions[1],
    )
    previous_token = deployment.credential["token"]
    emitted = []
    monkeypatch.setattr(cli, "_emit", lambda _args, value: emitted.append(value))

    cli.cmd_connect(deployment.args)

    current = deployment.store[_auth.STATE_KEY]
    assert len(deployment.published) == 1
    assert deployment.credential["deployment_id"] == current["deployment_id"]
    assert deployment.credential["auth_revision"] == current["auth_revision"] == 2
    assert _auth.matches_local_token(current, deployment.credential["token"])
    assert not _auth.matches_local_token(current, previous_token)
    assert emitted[0]["status"] == "connected"
    if helper_versions == gateway_versions:
        assert "protocol_mismatches" not in emitted[0]
    else:
        assert emitted[0]["protocol_mismatches"] == {
            field: {"gateway": gateway, "helper": helper}
            for field, gateway, helper in zip(
                ["module_schema_version", "finite_inference_schema_version"],
                gateway_versions,
                helper_versions,
                strict=True,
            )
        }
    assert deployment.credential["token"] not in json.dumps(emitted)
    assert previous_token not in json.dumps(emitted)
    assert deployment.events == []


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        (
            "deployment_id",
            "foreign-deployment",
            RuntimeError,
            "different deployment ID",
        ),
        ("auth_revision", 99, RuntimeError, "different auth revision"),
        ("api_major", 2, ValueError, "unsupported gateway API major"),
        ("module_schema_version", True, ValueError, "module_schema_version"),
        ("gateway_build", {}, ValueError, "software build identity"),
    ],
)
def test_reconnect_rejects_foreign_identity_api_or_malformed_capabilities(
    deployment, field, value, error_type, message
):
    previous = dict(deployment.credential)
    deployment.capability_overrides[field] = value

    with pytest.raises(error_type, match=message):
        cli.cmd_connect(deployment.args)

    assert deployment.credential == previous
    current = deployment.store[_auth.STATE_KEY]
    assert current["auth_revision"] == 2
    assert not _auth.matches_local_token(current, previous["token"])
    assert deployment.published == []
    assert deployment.events == []


@pytest.mark.parametrize("failure", [401, 403, "network", "write"])
def test_reconnect_failure_does_not_publish_success_or_retry(
    deployment, monkeypatch, failure
):
    previous = dict(deployment.credential)
    calls = []

    def request(_url, _token, _path):
        calls.append("probe")
        if isinstance(failure, int):
            raise urllib.error.HTTPError(_url, failure, "rejected", {}, None)
        if failure == "network":
            raise urllib.error.URLError("unavailable")
        value = _capability_fixture()
        current = deployment.store[_auth.STATE_KEY]
        return value | {
            "deployment_id": current["deployment_id"],
            "auth_revision": current["auth_revision"],
        }

    def write(_value):
        raise OSError("write failed")

    monkeypatch.setattr(_cloud_update, "gateway_request", request)
    if failure == "write":
        monkeypatch.setattr(_deployment, "write", write)
    with pytest.raises((urllib.error.URLError, OSError)):
        cli.cmd_connect(deployment.args)
    assert calls == ["probe"]
    assert deployment.credential == previous
    assert deployment.published == []
    assert deployment.events == []


@pytest.mark.parametrize("failure_at", ["second_stop", "deploy"])
def test_reconnect_recovers_an_interrupted_cross_version_update(
    deployment, monkeypatch, failure_at
):
    deployed = False
    fail_once = True
    stop_calls = []

    def capabilities(_url, token, path):
        assert path == "/v1/capabilities"
        current = deployment.store[_auth.STATE_KEY]
        assert _auth.matches_local_token(current, token)
        value = _capability_fixture()
        value.update(
            deployment_id=current["deployment_id"],
            auth_revision=current["auth_revision"],
        )
        if deployed:
            value["gateway_build"].update(version=__version__, artifact_digest="a" * 64)
        else:
            value.update(module_schema_version=3, finite_inference_schema_version=2)
        return value

    def stop(_url, token, *, wait_for_module_writer):
        assert _auth.matches_local_token(deployment.store[_auth.STATE_KEY], token)
        stop_calls.append(deployment.store[_auth.STATE_KEY]["auth_revision"])
        if fail_once and failure_at == "second_stop" and len(stop_calls) == 2:
            raise RuntimeError("injected stop failure")
        wait_for_module_writer()

    def deploy(**kwargs):
        nonlocal deployed
        assert kwargs == {"strategy": "recreate"}
        if fail_once and failure_at == "deploy":
            raise RuntimeError("injected deploy failure")
        deployed = True

    monkeypatch.setattr(_cloud_update, "gateway_request", capabilities)
    monkeypatch.setattr(_cloud_update, "stop_cloud_work", stop)
    monkeypatch.setattr(
        cli, "_validated_deployment_app", lambda: SimpleNamespace(deploy=deploy)
    )
    monkeypatch.setattr(cli, "_publish_deployed_credential", deployment.real_publisher)
    with pytest.raises(RuntimeError, match="injected"):
        cli.cmd_upgrade(deployment.args)
    assert deployment.credential["auth_revision"] == 1
    assert deployment.store[_auth.STATE_KEY]["auth_revision"] == 2
    with pytest.raises(cli.CliFailure) as stale:
        cli.cmd_upgrade(deployment.args)
    assert stale.value.code == "RECONNECT_REQUIRED"

    cli.cmd_connect(deployment.args)
    assert deployment.credential["auth_revision"] == 3
    fail_once = False
    cli.cmd_upgrade(deployment.args)

    assert deployed
    assert (
        deployment.credential["deployment_id"] == deployment.original["deployment_id"]
    )
    assert deployment.credential["auth_revision"] == 4
    assert stop_calls == [1, 2, 3, 4]
    assert len(deployment.published) == 2
    assert deployment.events[-1] == ("emit", "upgraded")


def test_post_update_rejects_the_published_protocol_pair(deployment, monkeypatch):
    deployment.capability_overrides.update(
        module_schema_version=3, finite_inference_schema_version=2
    )
    monkeypatch.setattr(cli, "_publish_deployed_credential", deployment.real_publisher)
    with pytest.raises(ValueError, match="unsupported gateway capability versions"):
        cli._publish_deployed_credential(
            deployment.credential["gateway_url"],
            deployment.original,
            deployment.credential["token"],
            deployment.args.profile,
            expected_artifact_digest="a" * 64,
            environment="research",
        )
    assert deployment.published == []
    assert deployment.credential["auth_revision"] == 1
    assert deployment.store[_auth.STATE_KEY]["auth_revision"] == 1


def test_foreign_cutover_record_blocks_before_cloud_work_is_stopped(deployment):
    deployment.legacy = True
    deployment.staged = _auth.initial_state()
    with pytest.raises(ValueError, match="foreign staged identity"):
        cli.cmd_upgrade(deployment.args)
    assert deployment.events == []
    assert deployment.store[_auth.STATE_KEY] == deployment.original


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
    current, token = provision_legacy_state({})
    monkeypatch.setattr(
        cli,
        "_capabilities",
        lambda *_: (
            current | {"gateway_build": {"version": version, "artifact_digest": digest}}
        ),
    )
    if accepted:
        cli._publish_deployed_credential(
            "https://gateway.example",
            current,
            token,
            "lab",
            expected_artifact_digest="a" * 64,
            environment="research",
        )
        assert _deployment.read()["deployment_id"] == current["deployment_id"]
    else:
        with pytest.raises(RuntimeError, match="expected bundled backend"):
            cli._publish_deployed_credential(
                "https://gateway.example",
                current,
                token,
                "lab",
                expected_artifact_digest="a" * 64,
                environment="research",
            )
        assert path.read_text(encoding="utf-8") == "previous credential"


@pytest.mark.parametrize(
    ("field", "version"),
    [
        ("finite_inference_schema_version", 1),
        ("finite_inference_schema_version", 2),
        ("finite_inference_schema_version", 4),
        ("durable_model_run_schema_version", 3),
        ("durable_model_run_schema_version", 5),
    ],
)
def test_update_rejects_mismatched_schemas_before_writing_credential(
    tmp_path, monkeypatch, field, version
):
    path = tmp_path / "deployment.json"
    path.write_text("previous credential", encoding="utf-8")
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_deployment, "DIR", tmp_path)
    current, token = provision_legacy_state({})
    fixture = Path(__file__).parent / "fixtures/contracts/gateway-capabilities-v2.json"
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
        cli._publish_deployed_credential(
            "https://gateway.example",
            current,
            token,
            "lab",
            expected_artifact_digest="a" * 64,
            environment="research",
        )
    assert probes == ["/v1/capabilities"]
    assert path.read_text(encoding="utf-8") == "previous credential"


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError(
            "https://gateway.example/v1/capabilities", 500, "cold gateway", {}, None
        ),
        urllib.error.URLError("connection unavailable"),
        TimeoutError("The read operation timed out"),
    ],
)
def test_post_deploy_readiness_recovers_from_a_cold_gateway(monkeypatch, failure):
    calls = []
    expected = {"deployment_id": "same", "auth_revision": 4}

    def capabilities(_url, _token):
        calls.append("probe")
        if len(calls) == 1:
            raise failure
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


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.HTTPError(
            "https://gateway.example/v1/capabilities", 500, "unavailable", {}, None
        ),
        urllib.error.URLError("connection unavailable"),
        TimeoutError("The read operation timed out"),
    ],
)
def test_post_deploy_readiness_stops_after_four_transient_failures(
    monkeypatch, failure
):
    calls = []

    def capabilities(_url, _token):
        calls.append("probe")
        raise failure

    monkeypatch.setattr(cli, "_capabilities", capabilities)
    monkeypatch.setattr(
        cli.time, "sleep", lambda seconds: calls.append(("wait", seconds))
    )
    with pytest.raises(type(failure)) as caught:
        cli._capabilities_after_deploy("https://gateway.example", "token")
    assert caught.value is failure
    assert calls == [
        "probe",
        ("wait", 2),
        "probe",
        ("wait", 2),
        "probe",
        ("wait", 2),
        "probe",
    ]


def _bind_saved_target(state):
    state.args.expected_gateway = state.credential["gateway_url"]
    state.args.expected_deployment_id = state.credential["deployment_id"]


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check"])
@pytest.mark.parametrize("target", ["different", "absent"])
def test_bound_management_rejects_wrong_target_before_state_lookup(
    deployment, monkeypatch, operation, target
):
    _bind_saved_target(deployment)
    deployment.args.check = operation == "check"
    before = dict(deployment.credential)
    if target == "different":
        monkeypatch.setattr(cli, "_gateway_url", lambda: "https://other.example")
    else:
        deployment.exists = False

    def forbidden():
        pytest.fail("wrong target must fail before any cloud state lookup")

    monkeypatch.setattr(gateway_state_client, "GatewayStateClient", forbidden)
    monkeypatch.setattr(_gateway_state_setup, "GatewayStateSetup", forbidden)
    monkeypatch.setattr(cli, "_require_artifact_lifecycle_lane", forbidden)
    command = cli.cmd_connect if operation == "connect" else cli.cmd_upgrade
    with pytest.raises(cli.CliFailure) as failure:
        command(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert deployment.credential == before
    assert deployment.store[_auth.STATE_KEY] == deployment.original
    assert deployment.published == deployment.events == deployment.creates == []


@pytest.mark.parametrize("operation", ["connect", "upgrade"])
def test_bound_management_rejects_another_identity_at_the_same_url(
    deployment, operation
):
    _bind_saved_target(deployment)
    foreign = _auth.initial_state()
    deployment.store[_auth.STATE_KEY] = foreign
    command = cli.cmd_connect if operation == "connect" else cli.cmd_upgrade
    with pytest.raises(cli.CliFailure) as failure:
        command(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert deployment.store[_auth.STATE_KEY] == foreign
    assert deployment.published == deployment.events == []


@pytest.mark.parametrize("legacy", [False, True])
def test_bound_reconnect_recovers_stale_access_without_replacing_identity(
    deployment, legacy
):
    _bind_saved_target(deployment)
    deployment.legacy = legacy
    deployment.store[_auth.STATE_KEY] = deployment.original | {
        "auth_revision": 8,
        "token_sha256": _auth.token_digest("newer-test-client"),
    }
    deployment.capability_overrides.update(
        module_schema_version=3, finite_inference_schema_version=2
    )
    cli.cmd_connect(deployment.args)
    assert len(deployment.published) == 1
    assert (
        deployment.credential["deployment_id"] == deployment.original["deployment_id"]
    )
    assert deployment.credential["auth_revision"] == 9
    assert _auth.matches_local_token(
        deployment.store[_auth.STATE_KEY], deployment.credential["token"]
    )


def test_bound_reconnect_rechecks_identity_inside_rotation(deployment, monkeypatch):
    _bind_saved_target(deployment)
    original_client = gateway_state_client.GatewayStateClient
    foreign = _auth.initial_state()

    class ReplacedBeforeRotation(original_client):
        def rotate(self, token_sha256, **expected):
            deployment.store[_auth.STATE_KEY] = foreign
            return super().rotate(token_sha256, **expected)

    monkeypatch.setattr(
        gateway_state_client, "GatewayStateClient", ReplacedBeforeRotation
    )
    with pytest.raises(ValueError, match="access changed"):
        cli.cmd_connect(deployment.args)
    assert deployment.store[_auth.STATE_KEY] == foreign
    assert deployment.published == []


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check"])
def test_bound_management_uses_saved_profile_when_none_is_explicit(
    deployment, monkeypatch, operation
):
    _bind_saved_target(deployment)
    deployment.args.profile = None
    deployment.args.check = operation == "check"
    profiles = []

    def authenticate(profile):
        profiles.append(profile)
        return profile

    monkeypatch.setattr(_modal_setup, "ensure_auth", authenticate)
    command = cli.cmd_connect if operation == "connect" else cli.cmd_upgrade
    command(deployment.args)
    assert profiles == ["research-lab"]


@pytest.mark.parametrize("legacy", [False, True])
def test_update_preflight_does_not_cancel_rotate_deploy_or_write(deployment, legacy):
    _bind_saved_target(deployment)
    deployment.legacy = legacy
    deployment.args.check = True
    deployment.args.cancel_work = False
    before = dict(deployment.credential)
    cli.cmd_upgrade(deployment.args)
    assert deployment.events == [("emit", "ready_to_update")]
    assert deployment.published == []
    assert deployment.credential == before
    assert deployment.store[_auth.STATE_KEY] == deployment.original


def test_actual_update_rechecks_access_after_successful_preflight(deployment):
    _bind_saved_target(deployment)
    deployment.args.check = True
    cli.cmd_upgrade(deployment.args)
    deployment.events.clear()
    _auth.legacy_rotate(deployment.store)
    deployment.args.check = False
    with pytest.raises(cli.CliFailure) as failure:
        cli.cmd_upgrade(deployment.args)
    assert failure.value.code == "RECONNECT_REQUIRED"
    assert deployment.events == deployment.published == []
    assert deployment.store[_auth.STATE_KEY]["auth_revision"] == 2
    assert deployment.credential["auth_revision"] == 1


def test_saved_target_change_after_preflight_cannot_redirect_the_update(deployment):
    _bind_saved_target(deployment)
    deployment.args.check = True
    cli.cmd_upgrade(deployment.args)
    deployment.events.clear()
    deployment.credential = deployment.credential | {
        "gateway_url": "https://other.example",
        "deployment_id": "another-deployment",
    }
    deployment.args.check = False
    with pytest.raises(cli.CliFailure) as failure:
        cli.cmd_upgrade(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert deployment.events == deployment.published == []
    assert deployment.store[_auth.STATE_KEY] == deployment.original


def test_bound_reconnect_requires_both_expected_fields_before_authentication(
    deployment, monkeypatch
):
    deployment.args.expected_gateway = deployment.credential["gateway_url"]
    monkeypatch.setattr(
        _modal_setup, "ensure_auth", lambda _: pytest.fail("incomplete target")
    )
    with pytest.raises(ValueError, match="must be used together"):
        cli.cmd_connect(deployment.args)
    assert deployment.events == deployment.published == []


def test_cli_parses_bound_update_check_without_cancellation_confirmation(
    deployment, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "histoforge",
            "upgrade",
            "--json",
            "--check",
            "--expected-gateway",
            deployment.credential["gateway_url"],
            "--expected-deployment-id",
            deployment.credential["deployment_id"],
        ],
    )
    cli.main()
    assert deployment.events == [("emit", "ready_to_update")]
    assert deployment.published == []


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check", "login"])
def test_saved_environment_overrides_later_provider_defaults(deployment, operation):
    _bind_saved_target(deployment)
    deployment.credential |= {"schema_version": 2, "modal_environment": "analysis-qa"}
    deployment.gateway_environment = "analysis-qa"
    deployment.default_environment = "unrelated"
    deployment.args.check = operation == "check"
    command = {"connect": cli.cmd_connect, "login": cli.cmd_login}.get(
        operation, cli.cmd_upgrade
    )
    command(deployment.args)
    assert deployment.selected_environments == ["analysis-qa"]
    assert os.environ["MODAL_ENVIRONMENT"] == "analysis-qa"
    if deployment.published:
        assert deployment.published[0]["modal_environment"] == "analysis-qa"


def test_v1_preflight_freezes_resolved_environment_without_writing(
    deployment, monkeypatch
):
    _bind_saved_target(deployment)
    deployment.args.check = True
    results = []
    monkeypatch.setattr(cli, "_emit", lambda _args, value: results.append(value))
    before = dict(deployment.credential)
    cli.cmd_upgrade(deployment.args)
    assert results[0]["environment"] == "research"
    assert deployment.credential == before and not deployment.published
    deployment.default_environment = "unrelated"
    deployment.args.environment = results[0]["environment"]
    deployment.args.check = False
    cli.cmd_upgrade(deployment.args)
    assert deployment.selected_environments == ["research", "research"]
    assert deployment.credential["schema_version"] == 2
    assert deployment.credential["modal_environment"] == "research"


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check", "login"])
def test_explicit_location_cannot_override_a_v2_saved_backend(deployment, operation):
    _bind_saved_target(deployment)
    deployment.credential |= {"schema_version": 2, "modal_environment": "analysis-qa"}
    deployment.gateway_environment = "analysis-qa"
    deployment.args.environment = "unrelated"
    deployment.args.check = operation == "check"
    command = {"connect": cli.cmd_connect, "login": cli.cmd_login}.get(
        operation, cli.cmd_upgrade
    )
    with pytest.raises(cli.CliFailure) as failure:
        command(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert not deployment.selected_environments
    assert not deployment.events and not deployment.published
    assert deployment.store[_auth.STATE_KEY] == deployment.original


def test_old_connection_adopts_explicit_location_only_after_verified_reconnect(
    deployment,
):
    _bind_saved_target(deployment)
    deployment.default_environment = "unrelated"
    deployment.gateway_environment = "analysis-qa"
    deployment.args.environment = "analysis-qa"
    cli.cmd_connect(deployment.args)
    assert deployment.credential["modal_environment"] == "analysis-qa"
    assert deployment.credential["schema_version"] == 2
    assert (
        deployment.credential["deployment_id"] == deployment.original["deployment_id"]
    )


def test_bound_legacy_reconnect_checks_the_identity_used_by_the_writer(
    deployment, monkeypatch
):
    _bind_saved_target(deployment)
    deployment.legacy = True
    original = _gateway_state_setup.GatewayStateSetup
    foreign = _auth.initial_state()

    class ReplacedBeforeRotation(original):
        def legacy_rotate(self, **expected):
            deployment.store[_auth.STATE_KEY] = foreign
            return super().legacy_rotate(**expected)

    monkeypatch.setattr(
        _gateway_state_setup, "GatewayStateSetup", ReplacedBeforeRotation
    )
    with pytest.raises(ValueError, match="identity changed"):
        cli.cmd_connect(deployment.args)
    assert deployment.store[_auth.STATE_KEY] == foreign
    assert not deployment.published


@pytest.mark.parametrize(
    "wrong_target", [None, "url", "identity", "app", "missing_environment"]
)
def test_token_save_uses_saved_location_and_checks_identity_before_secret_write(
    deployment, monkeypatch, tmp_path, capsys, wrong_target
):
    import io

    from histoforge import _huggingface_setup

    _bind_saved_target(deployment)
    deployment.credential |= {"schema_version": 2, "modal_environment": "analysis-qa"}
    deployment.gateway_environment = "analysis-qa"
    deployment.default_environment = "unrelated"
    path = tmp_path / "deployment.json"
    path.write_text("fixture-only; read is mocked")
    monkeypatch.setattr(_deployment, "PATH", path)
    monkeypatch.setattr(_modal_setup, "ensure_auth", lambda profile, **kw: profile)
    monkeypatch.setattr(sys, "stdin", io.StringIO("hf_fixture_token"))
    monkeypatch.setattr(_huggingface_setup, "_verify_token", lambda token: None)
    writes = []
    monkeypatch.setattr(
        _huggingface_setup,
        "_save_secret",
        lambda token: writes.append((os.environ["MODAL_ENVIRONMENT"], token)),
    )
    if wrong_target == "url":
        monkeypatch.setattr(cli, "_gateway_url", lambda: "https://other.example")
    elif wrong_target == "identity":
        deployment.store[_auth.STATE_KEY] = _auth.initial_state()
    elif wrong_target == "app":
        deployment.credential["modal_app_name"] = "different-app"
    elif wrong_target == "missing_environment":

        def missing(_name):
            print("hf_fixture_token")
            print("hf_fixture_token", file=sys.stderr)
            raise modal.exception.NotFoundError("hf_fixture_token")

        monkeypatch.setattr(_modal_setup, "ensure_environment", missing)
    before = dict(deployment.credential)
    if wrong_target:
        with pytest.raises(cli.CliFailure) as failure:
            cli.cmd_huggingface_token(deployment.args)
        assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
        assert "hf_fixture_token" not in str(failure.value)
        assert capsys.readouterr() == ("", "")
        assert writes == []
    else:
        cli.cmd_huggingface_token(deployment.args)
        assert writes == [("analysis-qa", "hf_fixture_token")]
    assert deployment.credential == before and not deployment.published


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check", "login"])
def test_wrong_ambient_environment_fails_and_explicit_v1_location_repairs_it(
    deployment, operation
):
    _bind_saved_target(deployment)
    deployment.default_environment = "unrelated"
    deployment.args.check = operation == "check"
    command = {"connect": cli.cmd_connect, "login": cli.cmd_login}.get(
        operation, cli.cmd_upgrade
    )
    before = dict(deployment.credential)
    with pytest.raises(cli.CliFailure) as failure:
        command(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert deployment.credential == before
    assert deployment.store[_auth.STATE_KEY] == deployment.original
    assert not deployment.published and not deployment.events and not deployment.creates
    deployment.args.environment = "research"
    command(deployment.args)
    assert deployment.selected_environments == ["unrelated", "research"]
    if operation in {"connect", "upgrade"}:
        assert deployment.credential["modal_environment"] == "research"
        assert deployment.credential["schema_version"] == 2
    else:
        assert deployment.credential == before and not deployment.published


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check", "login"])
def test_missing_environment_keeps_v1_recovery_action_available(
    deployment, monkeypatch, operation
):
    _bind_saved_target(deployment)
    deployment.args.check = operation == "check"

    def missing(_name):
        raise modal.exception.NotFoundError("environment does not exist")

    monkeypatch.setattr(_modal_setup, "ensure_environment", missing)
    monkeypatch.setattr(cli, "_gateway_url", lambda: pytest.fail("gateway lookup ran"))
    command = {"connect": cli.cmd_connect, "login": cli.cmd_login}.get(
        operation, cli.cmd_upgrade
    )
    with pytest.raises(cli.CliFailure) as failure:
        command(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert "Locate saved backend" in str(failure.value)
    assert not deployment.published and not deployment.events
    assert deployment.store[_auth.STATE_KEY] == deployment.original


@pytest.mark.parametrize(
    "failure", [PermissionError("access denied"), TimeoutError("transport")]
)
def test_environment_access_and_transport_failure_are_not_missing_target(
    deployment, monkeypatch, failure
):
    _bind_saved_target(deployment)

    def unavailable(_name):
        raise failure

    monkeypatch.setattr(_modal_setup, "ensure_environment", unavailable)
    with pytest.raises(type(failure)) as caught:
        cli.cmd_connect(deployment.args)
    assert caught.value is failure
    assert not deployment.published and not deployment.events


@pytest.mark.parametrize("operation", ["connect", "upgrade", "check", "login"])
def test_inconsistent_app_name_fails_before_lookup_or_mutation(
    deployment, monkeypatch, operation
):
    _bind_saved_target(deployment)
    deployment.credential |= {"modal_app_name": "different-app"}
    deployment.args.check = operation == "check"
    monkeypatch.setattr(cli, "_gateway_url", lambda: pytest.fail("gateway lookup ran"))
    command = {"connect": cli.cmd_connect, "login": cli.cmd_login}.get(
        operation, cli.cmd_upgrade
    )
    with pytest.raises(cli.CliFailure) as failure:
        command(deployment.args)
    assert failure.value.code == "GATEWAY_TARGET_MISMATCH"
    assert not deployment.published and not deployment.events
    assert deployment.store[_auth.STATE_KEY] == deployment.original
