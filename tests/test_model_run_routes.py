"""Durable HTTP routes preserve reservation, dispatch, and lifecycle ownership."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import (
    GatewayFailure,
    build_gateway_error,
    gateway_error_http_status,
)
from histoforge.cloud.inference.runs import routes as model_run_routes
from histoforge.cloud.inference.runs.request import (
    model_run_request_digest,
    validate_model_run_admission,
)
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class _RunLifecycle:
    def __init__(self, accepted):
        self.accepted = accepted
        self.should_dispatch = True
        self.calls = []
        self._init_status()

    def reserve_model_run(self, request, limits, retention):
        self.calls.append((request, limits, retention))
        return {
            "accepted": self.accepted,
            "should_dispatch": self.should_dispatch,
        }

    def _init_status(self):
        self.cancelled = []
        self.failures = []
        self.status_value = {
            "schema_version": MODEL_RUN_SCHEMA_VERSION,
            "model_run_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "phase": "QUEUED",
        }

    def status(self, model_run_id):
        assert model_run_id == self.status_value["model_run_id"]
        return self.status_value

    def cancel(self, model_run_id):
        self.cancelled.append(model_run_id)

    def fail(self, model_run_id, failure):
        self.failures.append((model_run_id, failure))
        return True

    def sources(self, model_run_id):
        self.calls.append(("sources", model_run_id))
        return self.source_delivery

    def bind_source(self, model_run_id, binding):
        self.calls.append(("bind", model_run_id, binding))
        return binding

    def report_source_failure(self, model_run_id, failure):
        self.failures.append((model_run_id, failure))


def test_start_dispatches_once_and_status_cancel_use_the_reserved_id(monkeypatch):
    request = _request()
    accepted = _accepted(request)
    lifecycle = _RunLifecycle(accepted)
    spawned = []
    _patch_admission(monkeypatch, request)
    client = _client(lifecycle, lambda run_id: spawned.append(run_id))

    response = client.post(
        "/v1/inference/histoforge/tubule/tubule_semantic/runs",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 202
    assert response.content == canonical_bytes(accepted)
    assert spawned == [request["model_run_id"]]
    assert lifecycle.calls == [(request, {"limits": True}, 3600)]

    lifecycle.should_dispatch = False
    replay = client.post(
        "/v1/inference/histoforge/tubule/tubule_semantic/runs",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )
    assert replay.content == canonical_bytes(accepted)
    assert spawned == [request["model_run_id"]]

    status = client.get(f"/v1/inference/runs/{request['model_run_id']}")
    assert status.status_code == 200
    assert status.content == canonical_bytes(lifecycle.status_value)
    cancelled = client.delete(f"/v1/inference/runs/{request['model_run_id']}")
    assert cancelled.status_code == 204
    assert lifecycle.cancelled == [request["model_run_id"]]


def test_dispatch_failure_publishes_typed_terminal_failure(monkeypatch):
    request = _request()
    lifecycle = _RunLifecycle(_accepted(request))
    _patch_admission(monkeypatch, request)

    def fail_spawn(_run_id):
        raise OSError("Modal unavailable")

    response = _client(lifecycle, fail_spawn).post(
        "/v1/inference/histoforge/tubule/tubule_semantic/runs",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 503
    assert response.json()["code"] == "SERVICE_UNAVAILABLE"
    assert lifecycle.failures[0][0] == request["model_run_id"]
    assert lifecycle.failures[0][1]["code"] == "SERVICE_UNAVAILABLE"


def test_durable_routes_reject_noncanonical_path_spelling(monkeypatch):
    request = _request()
    _patch_admission(monkeypatch, request)
    client = _client(
        _RunLifecycle(_accepted(request)),
        lambda _run_id: None,
    )

    encoded_module = client.post(
        "/v1/inference/%68istoforge/tubule/tubule_semantic/runs",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )
    assert encoded_module.status_code == 422
    encoded_id = client.get("/v1/inference/runs/%62bbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    assert encoded_id.status_code == 422


def test_source_routes_keep_exact_assignments_and_failure_reports_in_lifecycle():
    case = _fixture("model-source-delivery-v1.json")["cases"][1]
    request = json.loads(case["requestCanonical"])
    lifecycle = _RunLifecycle(_accepted(request))
    lifecycle.source_delivery = json.loads(case["deliveryCanonical"])
    client = _client(lifecycle, lambda _: pytest.fail("binding cannot dispatch a run"))
    path = f"/v1/inference/runs/{request['model_run_id']}"
    response = client.get(path + "/sources")
    assert response.status_code == 200
    assert response.content == case["deliveryCanonical"].encode()
    binding = lifecycle.source_delivery["source_bindings"][0]
    for _ in range(2):
        response = client.post(path + "/sources", json=binding)
        assert response.status_code == 200
        assert response.content == canonical_bytes(binding)
    failure = build_gateway_error("SERVICE_UNAVAILABLE", "image preparation failed")
    response = client.post(path + "/source-failure", json=failure)
    assert response.status_code == 204
    assert lifecycle.failures == [(request["model_run_id"], failure)]
    assert lifecycle.calls == [
        ("sources", request["model_run_id"]),
        ("bind", request["model_run_id"], binding),
        ("bind", request["model_run_id"], binding),
    ]


@pytest.mark.parametrize(
    "method,suffix",
    [("get", "sources"), ("post", "sources"), ("post", "source-failure")],
)
def test_image_delivery_routes_require_authentication_and_canonical_run_id(
    method, suffix
):
    lifecycle = _RunLifecycle(_accepted(_request()))

    def deny():
        raise GatewayFailure("AUTHENTICATION_REQUIRED", "not authenticated")

    path = f"/v1/inference/runs/bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb/{suffix}"
    response = getattr(_client(lifecycle, lambda _: None, require_token=deny), method)(
        path
    )
    assert response.status_code == 401
    assert lifecycle.calls == []
    encoded = path.replace("/bbbbbbbb", "/%62bbbbbbb")
    response = getattr(_client(lifecycle, lambda _: None), method)(encoded)
    assert response.status_code == 422
    assert lifecycle.calls == []


def test_binding_validation_and_lifecycle_failure_codes_are_not_hidden():
    lifecycle = _RunLifecycle(_accepted(_request()))
    client = _client(lifecycle, lambda _: None)
    path = "/v1/inference/runs/bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb/sources"

    def invalid(*_):
        raise ValueError("source image omits required pixels")

    lifecycle.bind_source = invalid
    response = client.post(path, json={})
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"

    def expired(*_):
        raise GatewayFailure("ARTIFACT_PRECONDITION_FAILED", "source expired")

    lifecycle.bind_source = expired
    response = client.post(path, json={})
    assert response.json()["code"] == "ARTIFACT_PRECONDITION_FAILED"
    response = client.post(
        path.removesuffix("sources") + "source-failure", json={"message": "incomplete"}
    )
    assert response.status_code == 422
    assert lifecycle.failures == []


def _patch_admission(monkeypatch, request):
    monkeypatch.setattr(
        model_run_routes,
        "load_current_model_activation",
        lambda *_args: {"activation": True},
    )
    monkeypatch.setattr(
        model_run_routes,
        "validate_durable_admission",
        lambda value, *_args: (value, {"limits": True}, 3600),
    )


def _client(lifecycle, spawn, *, require_token=None):
    web = FastAPI()

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request, failure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    if require_token is None:

        def require_token():
            return {"deployment_id": "deployment-a", "auth_revision": 7}

    model_run_routes.register_routes(
        web,
        require_token=require_token,
        max_json_request_bytes=1_000_000,
        training_state_factory=lambda _state: SimpleNamespace(
            training_result_visible=lambda _run_id: True
        ),
        current_modules=lambda: object(),
        lifecycle_factory=lambda _state: lifecycle,
        spawn_run=spawn,
    )
    return TestClient(web)


def _accepted(request):
    return {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": request["action_id"],
        "model_run_id": request["model_run_id"],
        "request_digest": model_run_request_digest(request),
    }


def _request():
    request_fixture = _fixture("model-run-request-v1.json")
    case = request_fixture["cases"][0]
    finite_case = _fixture("finite-inference-v1.json")["cases"][
        case["finiteFixtureCase"]
    ]
    finite = json.loads(finite_case["requestCanonical"])
    region = finite["raster_input"]["frame"]["region"]
    request = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": case["actionId"],
        "model_run_id": case["modelRunId"],
        "producer": copy.deepcopy(finite["producer"]),
        "expected_backend": copy.deepcopy(finite["expected_backend"]),
        "inputs": [
            {
                "input_id": case["inputId"],
                "output_generation_id": case["outputGenerationId"],
                "output": {
                    "producer": copy.deepcopy(finite["producer"]),
                    "scope": {"type": "field", "region": copy.deepcopy(region)},
                },
                "raster_input": copy.deepcopy(finite["raster_input"]),
            }
        ],
        "result_budget": copy.deepcopy(finite["result_budget"]),
    }
    return validate_model_run_admission(
        request,
        finite_case["route"]["moduleRef"],
        finite_case["route"]["modelId"],
        request["expected_backend"],
        case["targetRasterLimits"],
        None,
        request_fixture["testModelRunLimits"],
    )


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
