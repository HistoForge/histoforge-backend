"""Gateway adaptation of the canonical ModuleRef contract.

Path segments on `DELETE /v1/modules/{owner}/{name}` are attacker-controlled;
a slack regex would let `x/..` resolve outside the intended module path.
"""

import asyncio
import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes, canonical_digest
from histoforge.cloud import _auth
from histoforge.cloud import capabilities as runtime_capabilities
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.admin_jobs.contract import (
    validate_install_request,
    validate_publish_request,
)
from histoforge.cloud.gateway import _GatewayAdmission, fastapi_app
from histoforge.cloud.gateway_state import GATEWAY_STATE_PATH, GatewayStateStore
from histoforge.cloud.modules.routes import (
    _canonical_route_module_ref,
    _delete_module_via_lane,
)
from histoforge.cloud.sources.client import SourceArtifactLifecycleClient
from tests.auth_fixtures import provision_legacy_state
from tests.test_training_report import succeeded_report


@pytest.fixture
def route_client():
    web = FastAPI()

    @web.get("/v1/modules/{owner}/{name}")
    async def module_route(owner: str, name: str, request: Request):
        ref = _canonical_route_module_ref(request, owner, name, trailing_segments=0)
        if ref is None:
            raise HTTPException(400, "invalid module route")
        return {"ref": ref}

    @web.get("/v1/inference/{owner}/{name}/{model_id}")
    async def inference_route(owner: str, name: str, model_id: str, request: Request):
        ref = _canonical_route_module_ref(request, owner, name, trailing_segments=1)
        if ref is None:
            raise HTTPException(400, "invalid module route")
        return {"ref": ref, "model_id": model_id}

    return TestClient(web)


def test_real_routes_accept_only_canonical_module_segments(route_client):
    assert route_client.get("/v1/modules/histoforge/leydig").json() == {
        "ref": "histoforge/leydig"
    }
    assert route_client.get("/v1/inference/histoforge/leydig/model").json() == {
        "ref": "histoforge/leydig",
        "model_id": "model",
    }

    for case in _route_cases():
        if "canonical" in case:
            continue
        module_path = f"/v1/modules/{case['owner']}/{case['module']}"
        inference_path = f"/v1/inference/{case['owner']}/{case['module']}/model"
        assert route_client.get(module_path).status_code in {400, 404}, case["name"]
        assert route_client.get(inference_path).status_code in {400, 404}, case["name"]


def _route_cases():
    fixture = (
        Path(__file__).parent / "fixtures" / "contracts" / "canonical-scalars-v1.json"
    )
    return json.loads(fixture.read_text())["moduleRefRouteCases"]


def test_gateway_resources_match_the_verified_bounded_upload_topology():
    assert DEPLOYMENT_CONFIG.gateway_max_concurrent_inputs == 8
    assert DEPLOYMENT_CONFIG.gateway_memory_mib == 1024
    assert DEPLOYMENT_CONFIG.gateway_timeout_seconds == 1800
    assert (
        DEPLOYMENT_CONFIG.training_service_limits.dataset_writer_claim_seconds
        > DEPLOYMENT_CONFIG.gateway_timeout_seconds
    )
    assert (
        DEPLOYMENT_CONFIG.training_service_limits.dataset_writer_claim_seconds
        < DEPLOYMENT_CONFIG.training_service_limits.dataset_upload_idle_timeout_seconds
    )
    assert fastapi_app.spec.memory == DEPLOYMENT_CONFIG.gateway_memory_mib


def test_real_gateway_health_and_capabilities_enforce_live_auth(monkeypatch):
    state_store = {}
    jobs_store = {}
    source_store = {}
    first_state, first_token = provision_legacy_state(state_store)
    observed_states = []
    current_marker = object()
    snapshot_active = False

    @contextmanager
    def snapshot():
        nonlocal snapshot_active
        snapshot_active = True
        try:
            yield current_marker
        finally:
            snapshot_active = False

    def from_name(name, **_kwargs):
        if name == DEPLOYMENT_CONFIG.gateway_state_cache_dict_name:
            return state_store
        if name == DEPLOYMENT_CONFIG.admin_jobs_dict_name:
            return jobs_store
        if name == DEPLOYMENT_CONFIG.source_artifacts_dict_name:
            return source_store
        raise AssertionError(f"unexpected Dict: {name}")

    def capabilities(state):
        observed_states.append(state)
        return {
            "deployment_id": state["deployment_id"],
            "auth_revision": state["auth_revision"],
        }

    monkeypatch.setattr("modal.Dict.from_name", from_name)
    monkeypatch.setattr(runtime_capabilities, "gateway_capabilities", capabilities)
    monkeypatch.setattr(
        "histoforge.cloud.modules.runtime.current_module_snapshot",
        snapshot,
    )
    client = TestClient(fastapi_app.get_raw_f()())

    assert client.get("/v1/health").json() == {"api_major": 1, "status": "ok"}
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    unauthorized = client.get("/v1/capabilities")
    assert unauthorized.status_code == 401
    assert unauthorized.json()["code"] == "AUTHENTICATION_REQUIRED"
    opaque_id = str(uuid.uuid4())
    assert client.get(f"/v1/source-artifacts/{opaque_id}").status_code == 401
    assert client.get(f"/v1/artifacts/{opaque_id}").status_code == 401

    accepted = client.get(
        "/v1/capabilities",
        headers={"Authorization": f"Bearer {first_token}"},
    )
    assert accepted.status_code == 200
    assert accepted.json() == {
        "deployment_id": first_state["deployment_id"],
        "auth_revision": 1,
    }
    assert observed_states[-1] == first_state

    second_state, second_token = _auth.legacy_rotate(state_store)
    assert (
        client.get(
            "/v1/capabilities",
            headers={"Authorization": f"Bearer {first_token}"},
        ).status_code
        == 401
    )
    rotated = client.get(
        "/v1/capabilities",
        headers={"Authorization": f"Bearer {second_token}"},
    )
    assert rotated.status_code == 200
    assert rotated.json()["auth_revision"] == 2
    assert observed_states[-1] == second_state

    catalog_fixture = json.loads(
        (
            Path(__file__).parent / "fixtures" / "contracts" / "module-catalog-v1.json"
        ).read_text(encoding="utf-8")
    )["page"]
    observed_catalog = []

    def catalog_page(current, **kwargs):
        assert snapshot_active
        observed_catalog.append((current, kwargs))
        return catalog_fixture

    monkeypatch.setattr("histoforge.cloud.modules.catalog.catalog_page", catalog_page)
    catalog = client.get(
        "/v1/modules?limit=1",
        headers={"Authorization": f"Bearer {second_token}"},
    )
    assert catalog.status_code == 200
    assert catalog.content == canonical_bytes(catalog_fixture)
    assert not snapshot_active
    visibility = observed_catalog[0][1].pop("training_result_visible")
    assert callable(visibility)
    assert visibility.__self__._deployment_id == second_state["deployment_id"]
    assert visibility.__self__._auth_revision == second_state["auth_revision"]
    assert observed_catalog == [
        (
            current_marker,
            {
                "limit": 1,
                "cursor": None,
                "max_page_size": DEPLOYMENT_CONFIG.limits.max_catalog_page_size,
                "max_encoded_bytes": (
                    DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes
                ),
            },
        )
    ]

    duplicate = client.get(
        "/v1/modules?limit=1&limit=2",
        headers={"Authorization": f"Bearer {second_token}"},
    )
    assert duplicate.status_code == 422
    assert duplicate.json()["code"] == "VALIDATION_ERROR"

    oversized_decimal = client.get(
        "/v1/modules?limit=" + "9" * 5_000,
        headers={"Authorization": f"Bearer {second_token}"},
    )
    assert oversized_decimal.status_code == 422
    assert oversized_decimal.json()["code"] == "VALIDATION_ERROR"


def test_module_training_report_is_authenticated_and_content_fenced(monkeypatch):
    state_store = {}
    jobs_store = {}
    source_store = {}
    _state, token = provision_legacy_state(state_store)
    report = succeeded_report()
    ref = "owner/trained"
    content_digest = "c" * 64
    snapshot_active = False
    record = {
        "module_ref": ref,
        "resolved_commit_sha": "a" * 40,
        "content_digest": content_digest,
    }

    class Current:
        def __init__(self):
            self.report = report
            self.error = None
            self.load_error = None
            self.record = record

        def read_record(self, module_ref):
            assert snapshot_active
            assert module_ref == self.record["module_ref"]
            if self.error is not None:
                raise self.error
            return self.record

        def revalidate(self, _module_ref):
            raise AssertionError("training report lookup must not rehash model weights")

        def load_training_report(self, module_ref, *, record):
            assert snapshot_active
            assert module_ref == self.record["module_ref"]
            assert record is self.record
            if self.load_error is not None:
                raise self.load_error
            return self.report

    current = Current()

    @contextmanager
    def snapshot():
        nonlocal snapshot_active
        snapshot_active = True
        try:
            yield current
        finally:
            snapshot_active = False

    def from_name(name, **_kwargs):
        if name == DEPLOYMENT_CONFIG.gateway_state_cache_dict_name:
            return state_store
        if name == DEPLOYMENT_CONFIG.admin_jobs_dict_name:
            return jobs_store
        if name == DEPLOYMENT_CONFIG.source_artifacts_dict_name:
            return source_store
        raise AssertionError(f"unexpected Dict: {name}")

    monkeypatch.setattr("modal.Dict.from_name", from_name)
    monkeypatch.setattr(
        "histoforge.cloud.modules.runtime.current_module_snapshot", snapshot
    )
    identity = {
        "module_ref": ref,
        "resolved_commit_sha": "a" * 40,
        "recipe_digest": "b" * 64,
        "content_digest": content_digest,
    }

    def identity_for(value):
        result = {
            "module_ref": value["module_ref"],
            "recipe_digest": "b" * 64,
            "content_digest": value["content_digest"],
        }
        if value["resolved_commit_sha"] is not None:
            result["resolved_commit_sha"] = value["resolved_commit_sha"]
        return result

    monkeypatch.setattr(
        "histoforge.cloud.modules.routes.module_content_identity",
        identity_for,
    )
    client = TestClient(fastapi_app.get_raw_f()())
    path = (
        "/v1/modules/owner/trained/training-report?expected_content_digest="
        f"{content_digest}"
    )

    assert client.get(path).status_code == 401
    response = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    expected = {
        "module_identity": identity,
        "report_digest": canonical_digest(report),
        "report": report,
    }
    assert response.content == canonical_bytes(expected)

    missing = client.get(
        "/v1/modules/owner/trained/training-report",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert missing.status_code == 422
    assert missing.json()["code"] == "VALIDATION_ERROR"
    duplicate = client.get(
        path + f"&expected_content_digest={content_digest}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert duplicate.status_code == 422
    invalid = client.get(
        "/v1/modules/owner/trained/training-report?expected_content_digest=ABC",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert invalid.status_code == 422

    stale = client.get(
        "/v1/modules/owner/trained/training-report?expected_content_digest=" + "d" * 64,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert stale.status_code == 409
    assert stale.json()["code"] == "MODULE_IDENTITY_MISMATCH"

    current.report = None
    unavailable = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert unavailable.status_code == 409
    assert unavailable.json()["code"] == "MODULE_VERSION_UNAVAILABLE"

    current.report = report
    current.error = FileNotFoundError(ref)
    absent = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert absent.status_code == 404
    assert absent.json()["code"] == "MODULE_NOT_FOUND"

    current.error = ValueError("private path must not leak")
    corrupt = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert corrupt.status_code == 409
    assert corrupt.json()["code"] == "MODULE_IDENTITY_MISMATCH"
    assert "private path" not in corrupt.text

    current.error = OSError("private storage path")
    unavailable_storage = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert unavailable_storage.status_code == 503
    assert unavailable_storage.json()["code"] == "SERVICE_UNAVAILABLE"
    assert "private storage path" not in unavailable_storage.text

    current.error = None
    current.load_error = FileNotFoundError("private report path")
    vanished_report = client.get(path, headers={"Authorization": f"Bearer {token}"})
    assert vanished_report.status_code == 409
    assert vanished_report.json()["code"] == "MODULE_IDENTITY_MISMATCH"
    assert "private report path" not in vanished_report.text

    current.load_error = None
    current.record = {
        "module_ref": "workspace/run-44444444-4444-4444-8444-444444444444",
        "resolved_commit_sha": None,
        "content_digest": content_digest,
    }
    unattested = client.get(
        "/v1/modules/workspace/run-44444444-4444-4444-8444-444444444444/"
        f"training-report?expected_content_digest={content_digest}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert unattested.status_code == 409
    assert unattested.json()["code"] == "MODULE_IDENTITY_MISMATCH"

    current.record["training_result_attestation"] = {
        "run_id": "44444444-4444-4444-8444-444444444444",
        "module_content_digest": content_digest,
        "report_digest": canonical_digest(report),
    }
    workspace = client.get(
        "/v1/modules/workspace/run-44444444-4444-4444-8444-444444444444/"
        f"training-report?expected_content_digest={content_digest}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert workspace.status_code == 200
    assert not snapshot_active
    assert (
        workspace.json()["module_identity"]["module_ref"]
        == current.record["module_ref"]
    )


def test_real_gateway_refills_expired_cache_from_permanent_identity(
    tmp_path, monkeypatch
):
    state, token = provision_legacy_state({})
    path = tmp_path / GATEWAY_STATE_PATH
    path.parent.mkdir()
    path.write_bytes(canonical_bytes(state))
    cache = {}
    reloads = []
    owner = GatewayStateStore(
        tmp_path,
        cache,
        reload_volume=lambda: reloads.append(True),
        commit_volume=lambda: pytest.fail("authentication must not rotate identity"),
    )
    stores = {
        DEPLOYMENT_CONFIG.gateway_state_cache_dict_name: cache,
        DEPLOYMENT_CONFIG.admin_jobs_dict_name: {},
        DEPLOYMENT_CONFIG.source_artifacts_dict_name: {},
    }
    monkeypatch.setattr("modal.Dict.from_name", lambda name, **_kwargs: stores[name])

    class RemoteClass:
        def __call__(self):
            return SimpleNamespace(
                load_current=SimpleNamespace(remote=owner.load_current)
            )

    def lookup(app, name):
        assert app == DEPLOYMENT_CONFIG.gateway_app_name
        assert name == DEPLOYMENT_CONFIG.gateway_state_class_name
        return RemoteClass()

    monkeypatch.setattr("modal.Cls.from_name", lookup)
    monkeypatch.setattr(
        runtime_capabilities, "gateway_capabilities", lambda value: value
    )
    client = TestClient(fastapi_app.get_raw_f()())
    for _ in range(2):
        cache.clear()
        response = client.get(
            "/v1/capabilities", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json() == state
        assert cache == {_auth.STATE_KEY: state}
    assert len(reloads) == 2
    path.unlink()
    cache.clear()
    assert (
        client.get(
            "/v1/capabilities", headers={"Authorization": f"Bearer {token}"}
        ).status_code
        == 401
    )
    assert cache == {}


def test_gateway_authenticates_before_reading_any_request_body():
    state_store = {}
    _state, token = provision_legacy_state(state_store)

    async def exercise(headers, bodies):
        calls = {"app": 0, "receive": 0}
        messages = []
        pending = list(bodies)

        async def receive():
            calls["receive"] += 1
            body = pending.pop(0)
            return {
                "type": "http.request",
                "body": body,
                "more_body": bool(pending),
            }

        async def send(message):
            messages.append(message)

        async def app(scope, receive_downstream, send_downstream):
            calls["app"] += 1
            while True:
                message = await receive_downstream()
                if not message.get("more_body"):
                    break
            await send_downstream(
                {"type": "http.response.start", "status": 204, "headers": []}
            )
            await send_downstream({"type": "http.response.body", "body": b""})

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/v1/inference/owner/module/model",
            "raw_path": b"/v1/inference/owner/module/model",
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("gateway", 443),
            "root_path": "",
        }
        await _GatewayAdmission(app, lambda: _auth.legacy_current_state(state_store))(
            scope, receive, send
        )
        return calls, messages

    unauthorized_calls, unauthorized_messages = asyncio.run(
        exercise([], [b"body must remain unread"])
    )
    assert unauthorized_calls == {"app": 0, "receive": 0}
    assert unauthorized_messages[0]["status"] == 401

    authenticated = [(b"authorization", f"Bearer {token}".encode("ascii"))]
    admitted_calls, admitted_messages = asyncio.run(
        exercise(authenticated, [b"abc", b"de"])
    )
    assert admitted_calls == {"app": 1, "receive": 2}
    assert admitted_messages[0]["status"] == 204


def test_delete_uses_the_single_module_mutation_lane(monkeypatch):
    called = {}

    class DeleteMethod:
        def remote(self, job_id, module_ref, deployment_id, auth_revision):
            called["args"] = (
                job_id,
                module_ref,
                deployment_id,
                auth_revision,
            )

    class Lane:
        delete = DeleteMethod()

    def from_name(app_name, class_name):
        called["address"] = (app_name, class_name)
        return lambda: Lane()

    monkeypatch.setattr("modal.Cls.from_name", from_name)

    _delete_module_via_lane("owner/module", "deployment-a", 7)

    assert called["address"] == ("histoforge-gateway", "ModuleMutationLane")
    assert uuid.UUID(called["args"][0]).version == 4
    assert called["args"][1:] == ("owner/module", "deployment-a", 7)


def test_all_source_control_publications_use_the_single_lifecycle_lane(monkeypatch):
    called = []

    class Method:
        def __init__(self, name):
            self.name = name

        def remote(self, *args):
            called.append((self.name, args))
            return {"accepted": True} if self.name == "reserve" else None

    class Lane:
        reserve = Method("reserve")
        release = Method("release")
        claim_chunk = Method("claim_chunk")
        commit_chunk = Method("commit_chunk")
        claim_seal = Method("claim_seal")
        commit_seal = Method("commit_seal")

    def from_name(app_name, class_name):
        called.append(("address", (app_name, class_name)))
        return lambda: Lane()

    monkeypatch.setattr("modal.Cls.from_name", from_name)
    state = {"deployment_id": "deployment-a", "auth_revision": 7}
    start = {"schema_version": 1}
    lifecycle = SourceArtifactLifecycleClient(state)

    assert lifecycle.reserve(start) == {"accepted": True}
    lifecycle.release("artifact-a")
    lifecycle.claim_chunk("artifact-a", 2, 10, "a" * 64)
    lifecycle.commit_chunk("artifact-a", 2, "a" * 64, "b" * 64)
    lifecycle.claim_seal("artifact-a", {"bytes_digest": "c" * 64})
    lifecycle.commit_seal(
        "artifact-a",
        {"bytes_digest": "c" * 64},
        "b" * 64,
        {"manifest_digest": "d" * 64},
    )

    expected_address = (
        "histoforge-gateway",
        "ArtifactLifecycleLane",
    )
    assert called == [
        ("address", expected_address),
        ("reserve", ("deployment-a", 7, start)),
        ("release", ("deployment-a", 7, "artifact-a")),
        ("claim_chunk", ("deployment-a", 7, "artifact-a", 2, 10, "a" * 64)),
        (
            "commit_chunk",
            ("deployment-a", 7, "artifact-a", 2, "a" * 64, "b" * 64),
        ),
        (
            "claim_seal",
            ("deployment-a", 7, "artifact-a", {"bytes_digest": "c" * 64}),
        ),
        (
            "commit_seal",
            (
                "deployment-a",
                7,
                "artifact-a",
                {"bytes_digest": "c" * 64},
                "b" * 64,
                {"manifest_digest": "d" * 64},
            ),
        ),
    ]


def test_install_request_is_exact_and_credential_free():
    assert validate_install_request({"module_ref": "owner/module"}) == {
        "module_ref": "owner/module"
    }
    assert validate_install_request(
        {
            "module_ref": "owner/module",
            "hugging_face_revision": "main",
        }
    ) == {
        "module_ref": "owner/module",
        "hugging_face_revision": "main",
    }

    with pytest.raises(ValueError, match="unknown=.*hf_token"):
        validate_install_request(
            {
                "module_ref": "owner/module",
                "hf_token": "client-secret",
            }
        )

    with pytest.raises(ValueError, match="must be a nonblank string"):
        validate_install_request(
            {
                "module_ref": "owner/module",
                "hugging_face_revision": None,
            }
        )


def test_publish_request_is_exact_and_credential_free():
    payload = {
        "source_module": "owner/source",
        "target_repository": "owner/target",
        "visibility": "PRIVATE",
        "card_markdown": "# Card\n",
        "sync_locally": False,
    }
    assert validate_publish_request(payload) == payload

    with pytest.raises(ValueError, match="PUBLIC.*PRIVATE"):
        validate_publish_request({**payload, "visibility": "private"})

    with pytest.raises(ValueError, match="unknown=.*hf_token"):
        validate_publish_request({**payload, "hf_token": "client-secret"})


def test_artifact_inventories_leave_gateway_identity_outside_their_namespaces(tmp_path):
    from histoforge.cloud.artifacts.store import ArtifactFileStore
    from histoforge.cloud.sources.files import SourceFileStore
    from histoforge.cloud.training.control_records import TrainingControlRecords
    from histoforge.cloud.training.datasets.files import TrainingDatasetFiles

    record = tmp_path / GATEWAY_STATE_PATH
    record.parent.mkdir()
    encoded = canonical_bytes(_auth.initial_state())
    record.write_bytes(encoded)
    assert ArtifactFileStore(tmp_path).artifact_ids() == set()
    assert SourceFileStore(tmp_path).source_ids() == set()
    control = TrainingControlRecords(
        tmp_path,
        DEPLOYMENT_CONFIG.training_api_limits.wire_value(),
        DEPLOYMENT_CONFIG.training_service_limits.wire_value(),
    )
    assert control.datasets() == control.admissions() == control.runs() == []
    assert TrainingDatasetFiles(tmp_path).upload_ids(10) == []
    assert record.read_bytes() == encoded
