"""Finite HTTP routes preserve exact addressing, statuses, and call ownership."""

import json
from pathlib import Path
from types import SimpleNamespace

import modal
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure, gateway_error_http_status
from histoforge.cloud.architectures.registry import runner_build_for_precision
from histoforge.cloud.inference.finite import routes as finite_inference_routes

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "finite-inference-v3.json"


def test_submit_poll_pending_and_cancel_use_the_exact_finite_contract(monkeypatch):
    case = _case()
    request = case["request"]
    call_id = case["accepted"]["backend_call_id"]
    calls = _Calls(_outcome(case))
    source = _Source()
    spawn_args = []

    monkeypatch.setattr(
        finite_inference_routes,
        "load_current_model_activation",
        lambda *_args: {
            "backend": request["expected_backend"],
            "capabilities": {"marker": "capabilities"},
            "modal_class_name": "HovernetRunner",
        },
    )

    def admit(value, module_ref, model_id, backend, capabilities, store, files):
        assert value == request
        assert (module_ref, model_id) == (
            "histoforge/tubule",
            "tubule_semantic",
        )
        assert backend == request["expected_backend"]
        assert capabilities == {"marker": "capabilities"}
        assert store is source
        assert files == "files"
        return request

    monkeypatch.setattr(finite_inference_routes, "validate_finite_admission", admit)

    def spawn(*args):
        spawn_args.append(args)
        return SimpleNamespace(object_id=call_id, cancel=lambda: None)

    client = _client(source, spawn, calls.from_id)
    submitted = client.post(
        "/v1/inference/histoforge/tubule/tubule_semantic",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )
    assert submitted.status_code == 202
    assert submitted.content == canonical_bytes(case["accepted"])
    assert spawn_args[0][:4] == (
        "HovernetRunner",
        request["expected_backend"]["gpu"],
        "histoforge/tubule",
        "tubule_semantic",
    )
    assert spawn_args[0][4] == request["expected_backend"]["module"]["content_digest"]
    assert (
        spawn_args[0][5] == request["expected_backend"]["module"]["resolved_commit_sha"]
    )
    assert spawn_args[0][6] == request

    completed = client.get(f"/v1/inference/calls?backend_call_id={call_id}")
    assert completed.status_code == 200
    assert completed.content == canonical_bytes(case["result"])

    calls.mode = "pending"
    pending = client.get(f"/v1/inference/calls?backend_call_id={call_id}")
    assert pending.status_code == 202
    assert pending.content == canonical_bytes(case["pending"])

    calls.mode = "result"
    cancelled = client.delete(f"/v1/inference/calls?backend_call_id={call_id}")
    assert cancelled.status_code == 202
    assert cancelled.content == canonical_bytes(case["cancellation"])
    assert calls.cancelled == [call_id]


@pytest.mark.parametrize(
    ("gpu", "precision"),
    [("A10G", "default"), ("A10G", "bf16"), ("H200", "default")],
)
def test_modal_spawn_captures_gpu_and_precision_in_the_warm_key(
    monkeypatch, gpu, precision
):
    observed = {}

    class RunnerMethod:
        def spawn(self, request):
            observed["request"] = request
            return "call"

    class Runner:
        infer = RunnerMethod()

    class RunnerClass:
        def with_options(self, **options):
            observed["options"] = options
            return self

        def __call__(self, **parameters):
            observed["parameters"] = parameters
            return Runner()

    monkeypatch.setattr(
        finite_inference_routes.modal.Cls,
        "from_name",
        lambda app_name, class_name: (
            observed.update(app_name=app_name, class_name=class_name) or RunnerClass()
        ),
    )

    request = {
        "expected_backend": {
            "architecture": "dinov3-adapter-cellpose",
            "runner_build": runner_build_for_precision(
                "dinov3-adapter-cellpose", precision
            ),
        }
    }
    result = finite_inference_routes._spawn_call(
        "CellposeRunner",
        gpu,
        "histoforge/leydig",
        "leydig_nuclear",
        "a" * 64,
        "b" * 40,
        request,
    )

    assert result == "call"
    assert observed["options"] == {"gpu": gpu}
    assert observed["parameters"] == {
        "ref": "histoforge/leydig",
        "model_id": "leydig_nuclear",
        "content_digest": "a" * 64,
        "resolved_commit_sha": "b" * 40,
        "gpu": gpu,
        "inference_precision": precision,
    }
    assert observed["request"] == request


def test_raw_query_decoder_rejects_ambiguous_or_malformed_identity(monkeypatch):
    case = _case()
    calls = _Calls(_outcome(case))
    client = _client(_Source(), lambda *_args: None, calls.from_id)

    for query in (
        "backend_call_id=a&backend_call_id=b",
        "backend_call_id=%",
        "backend_call_id=%GG",
        "backend_call_id=%2",
        "backend_call_id=%FF",
        "wrong=a",
    ):
        response = client.get(f"/v1/inference/calls?{query}")
        assert response.status_code == 422, query
        assert response.json()["code"] == "VALIDATION_ERROR", query

    plus = client.get("/v1/inference/calls?backend_call_id=a+b")
    assert plus.status_code == 200
    assert plus.json()["meta"]["backend_call_id"] == "a+b"


def test_every_shared_opaque_call_address_round_trips_through_the_real_route():
    case = _case()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    calls = _Calls(_outcome(case))
    client = _client(_Source(), lambda *_args: None, calls.from_id)

    for address in fixture["backendCallAddressCases"]:
        call_id = address["unit"] * address["repetitions"]
        encoded = address["encodedUnit"] * address["repetitions"]
        response = client.get(f"/v1/inference/calls?backend_call_id={encoded}")
        assert response.status_code == 200, address["name"]
        assert response.json()["meta"]["backend_call_id"] == call_id


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (modal.exception.OutputExpiredError(), 410, "CALL_EXPIRED"),
        (modal.exception.FunctionTimeoutError(), 502, "RUNNER_FAILED"),
        (modal.exception.InternalFailure(), 502, "RUNNER_FAILED"),
        (modal.exception.RemoteError(), 502, "RUNNER_FAILED"),
    ],
)
def test_poll_maps_terminal_modal_states_before_generic_timeout(error, status, code):
    case = _case()

    class FailedCall:
        def get(self, *, timeout):
            assert timeout == 0
            raise error

    client = _client(_Source(), lambda *_args: None, lambda _call_id: FailedCall())
    response = client.get(
        f"/v1/inference/calls?backend_call_id={case['accepted']['backend_call_id']}"
    )

    assert response.status_code == status
    assert response.json()["code"] == code


def test_submit_rejects_noncanonical_route_spelling_before_admission():
    request = _case()["request"]
    client = _client(_Source(), lambda *_args: None, lambda _call_id: None)

    for path in (
        "/v1/inference/%68istoforge/tubule/tubule_semantic",
        "/v1/inference/histoforge/%74ubule/tubule_semantic",
        "/v1/inference/histoforge/tubule/%74ubule_semantic",
    ):
        response = client.post(
            path,
            content=canonical_bytes(request),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422, path
        assert response.json()["code"] == "VALIDATION_ERROR", path


def test_submit_maps_malformed_request_to_the_closed_validation_error(monkeypatch):
    request = _case()["request"]
    malformed = dict(request)
    malformed.pop("action_id")
    monkeypatch.setattr(
        finite_inference_routes,
        "load_current_model_activation",
        lambda *_args: {
            "backend": request["expected_backend"],
            "capabilities": {},
            "modal_class_name": "HovernetRunner",
        },
    )
    client = _client(_Source(), lambda *_args: None, lambda _call_id: None)

    response = client.post(
        "/v1/inference/histoforge/tubule/tubule_semantic",
        content=canonical_bytes(malformed),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (OSError("volume unavailable"), 503, "SERVICE_UNAVAILABLE"),
        (RuntimeError("programming fault"), 500, "INTERNAL_ERROR"),
    ],
)
def test_submit_closes_unexpected_activation_failures(monkeypatch, error, status, code):
    request = _case()["request"]

    def fail(*_args):
        raise error

    monkeypatch.setattr(finite_inference_routes, "load_current_model_activation", fail)
    client = _client(_Source(), lambda *_args: None, lambda _call_id: None)
    response = client.post(
        "/v1/inference/histoforge/tubule/tubule_semantic",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == status
    assert response.json()["code"] == code


def test_poll_preserves_typed_runner_failure_and_closes_untyped_failure():
    case = _case()

    class FailedCall:
        def __init__(self, error):
            self.error = error

        def get(self, *, timeout):
            assert timeout == 0
            raise self.error

    call_id = case["accepted"]["backend_call_id"]
    typed = GatewayFailure(
        "SERVICE_UNAVAILABLE",
        "runner activation could not read required storage",
        resource_id="histoforge/tubule/tubule_semantic",
    )
    typed_response = _client(
        _Source(), lambda *_args: None, lambda _call_id: FailedCall(typed)
    ).get(f"/v1/inference/calls?backend_call_id={call_id}")
    assert typed_response.status_code == 503
    assert typed_response.json() == typed.error

    untyped_response = _client(
        _Source(),
        lambda *_args: None,
        lambda _call_id: FailedCall(ValueError("remote secret")),
    ).get(f"/v1/inference/calls?backend_call_id={call_id}")
    assert untyped_response.status_code == 502
    assert untyped_response.json()["code"] == "RUNNER_FAILED"
    assert "secret" not in untyped_response.json()["message"]


def test_submit_fails_closed_and_cancels_an_invalid_server_call_id(monkeypatch):
    case = _case()
    request = case["request"]
    cancelled = []

    monkeypatch.setattr(
        finite_inference_routes,
        "load_current_model_activation",
        lambda *_args: {
            "backend": request["expected_backend"],
            "capabilities": {"marker": "capabilities"},
            "modal_class_name": "HovernetRunner",
        },
    )
    monkeypatch.setattr(
        finite_inference_routes,
        "validate_finite_admission",
        lambda value, *_args: value,
    )

    call = SimpleNamespace(
        object_id="x" * 129,
        cancel=lambda: cancelled.append(True),
    )
    client = _client(_Source(), lambda *_args: call, lambda _call_id: None)
    response = client.post(
        "/v1/inference/histoforge/tubule/tubule_semantic",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert cancelled == [True]


def _client(source, spawn, recover):
    web = FastAPI()

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request, failure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    finite_inference_routes.register_routes(
        web,
        require_token=lambda: {"authenticated": True},
        source_store=source,
        files="files",
        max_json_request_bytes=1_000_000,
        training_state_factory=lambda _state: SimpleNamespace(
            training_result_visible=lambda _run_id: True
        ),
        current_modules=lambda: object(),
        spawn_call=spawn,
        call_from_id=recover,
    )
    return TestClient(web)


class _Source:
    def __init__(self):
        self.references = []

    def resolve_sealed_reference(self, reference):
        self.references.append(reference)
        return {"sealed": True}


class _Calls:
    def __init__(self, outcome):
        self.outcome = outcome
        self.mode = "result"
        self.cancelled = []

    def from_id(self, call_id):
        owner = self

        class Call:
            def get(self, *, timeout):
                assert timeout == 0
                if owner.mode == "pending":
                    raise TimeoutError
                return owner.outcome

            def cancel(self):
                owner.cancelled.append(call_id)

        return Call()


def _case():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][0]
    return {
        name: json.loads(raw[f"{name}Canonical"])
        for name in ("request", "accepted", "pending", "cancellation", "result")
    } | {"declared_classes": raw["declaredClasses"]}


def _outcome(case):
    request = case["request"]
    meta = case["result"]["meta"]
    return {
        "type": "result",
        "action_id": meta["action_id"],
        "output_generation_id": meta["output_generation_id"],
        "input_id": meta["input_id"],
        "producer": meta["producer"],
        "backend": meta["backend"],
        "input_identity": meta["input_identity"],
        "raster": request["raster_input"]["frame"]["transform"]["raster"],
        "result_budget": request["result_budget"],
        "declared_classes": case["declared_classes"],
        "geojson": case["result"]["geojson"],
    }
