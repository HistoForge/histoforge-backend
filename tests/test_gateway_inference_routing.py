"""The composed gateway keeps durable image delivery out of finite inference."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud import _auth
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure, build_gateway_error
from histoforge.cloud.gateway import fastapi_app
from histoforge.cloud.inference.finite import routes as finite_routes
from histoforge.cloud.inference.runs import routes as run_routes


@pytest.fixture
def gateway(monkeypatch):
    stores = {
        DEPLOYMENT_CONFIG.gateway_state_dict_name: {},
        DEPLOYMENT_CONFIG.admin_jobs_dict_name: {},
        DEPLOYMENT_CONFIG.source_artifacts_dict_name: {},
    }
    _state, token = _auth.provision_new(
        stores[DEPLOYMENT_CONFIG.gateway_state_dict_name]
    )
    monkeypatch.setattr("modal.Dict.from_name", lambda name, **_: stores[name])
    calls = []

    def bind_source(run_id, binding):
        calls.append(("bind_source", run_id, binding))
        return binding

    def report_source_failure(run_id, failure):
        calls.append(("report_source_failure", run_id, failure))

    lifecycle = SimpleNamespace(
        bind_source=bind_source, report_source_failure=report_source_failure
    )
    monkeypatch.setattr(run_routes, "_ModelRunLifecycleClient", lambda _: lifecycle)

    def finite_activation(_current, module_ref, model_id, _visible):
        calls.append(("finite_activation", module_ref, model_id))
        raise GatewayFailure("MODULE_NOT_FOUND", "model is not installed")

    monkeypatch.setattr(
        finite_routes, "load_current_model_activation", finite_activation
    )

    def dispatch(*args):
        calls.append(("finite_dispatch", args[2], args[3], args[-1]))
        return SimpleNamespace(object_id="fc-test")

    monkeypatch.setattr(finite_routes, "_spawn_call", dispatch)
    client = TestClient(fastapi_app.get_raw_f()())
    return client, {"Authorization": f"Bearer {token}"}, calls


@pytest.fixture
def source_delivery():
    fixture = Path(__file__).parent / "fixtures/contracts/model-source-delivery-v1.json"
    case = json.loads(fixture.read_text())["cases"][1]
    request = json.loads(case["requestCanonical"])
    binding = json.loads(case["deliveryCanonical"])["source_bindings"][0]
    return request["model_run_id"], binding


@pytest.mark.parametrize("suffix", ["sources", "source-failure"])
def test_real_gateway_routes_image_delivery_to_run_lifecycle(
    gateway, source_delivery, suffix
):
    client, headers, calls = gateway
    run_id, binding = source_delivery
    path = f"/v1/inference/runs/{run_id}/{suffix}"
    failure = build_gateway_error("SERVICE_UNAVAILABLE", "image preparation failed")
    body = binding if suffix == "sources" else failure

    unauthorized = client.post(path, content=b"not JSON")
    assert unauthorized.status_code == 401
    assert calls == []

    response = client.post(path, json=body, headers=headers)
    if suffix == "sources":
        assert response.status_code == 200
        assert response.content == canonical_bytes(binding)
        assert calls == [("bind_source", run_id, binding)]
    else:
        assert response.status_code == 204
        assert response.content == b""
        assert calls == [("report_source_failure", run_id, failure)]


@pytest.mark.parametrize("suffix", ["sources", "source-failure"])
def test_real_gateway_keeps_canonical_run_id_checks(gateway, source_delivery, suffix):
    client, headers, calls = gateway
    run_id, binding = source_delivery
    body = (
        binding
        if suffix == "sources"
        else build_gateway_error("SERVICE_UNAVAILABLE", "image preparation failed")
    )
    encoded = "%" + format(ord(run_id[0]), "02x") + run_id[1:]
    for addressed_id in (encoded, "not-a-run-id"):
        response = client.post(
            f"/v1/inference/runs/{addressed_id}/{suffix}",
            json=body,
            headers=headers,
        )
        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"
        assert calls == []


def test_real_gateway_still_routes_ordinary_models_to_finite_inference(
    gateway, monkeypatch
):
    client, headers, calls = gateway
    fixture = Path(__file__).parent / "fixtures/contracts/finite-inference-v1.json"
    case = json.loads(fixture.read_text())["cases"][0]
    request = json.loads(case["requestCanonical"])
    module_ref = case["route"]["moduleRef"]
    model_id = case["route"]["modelId"]

    def activate(_current, addressed_module, addressed_model, _visible):
        calls.append(("finite_activation", addressed_module, addressed_model))
        return {
            "backend": request["expected_backend"],
            "capabilities": {},
            "modal_class_name": "test-runner",
        }

    monkeypatch.setattr(finite_routes, "load_current_model_activation", activate)
    monkeypatch.setattr(
        finite_routes, "validate_finite_admission", lambda value, *_: value
    )
    response = client.post(
        f"/v1/inference/{module_ref}/{model_id}", json=request, headers=headers
    )
    assert response.status_code == 202
    assert response.json() == {
        "schema_version": request["schema_version"],
        "action_id": request["action_id"],
        "output_generation_id": request["output_generation_id"],
        "backend_call_id": "fc-test",
    }
    assert calls == [
        ("finite_activation", module_ref, model_id),
        ("finite_dispatch", module_ref, model_id, request),
    ]
