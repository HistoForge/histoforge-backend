from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
    gateway_error_http_status,
)
from histoforge.cloud.training.runs.routes import register_routes
from tests.test_training_run_request import run_request


class _Service:
    def __init__(self) -> None:
        self.requests = []
        self.calls = []
        self.accepted = {
            "run_id": "44444444-4444-4444-8444-444444444444",
            "request_digest": "a" * 64,
        }

    def submit(self, request):
        self.requests.append(request)
        return self.accepted

    def page(self, limit, cursor):
        self.calls.append(("page", limit, cursor))
        return {"items": []}

    def get(self, run_id):
        self.calls.append(("get", run_id))
        return {"run_id": run_id}

    def cancel(self, run_id):
        self.calls.append(("cancel", run_id))
        return {"run_id": run_id, "state": "CANCELLED"}

    def report(self, run_id):
        self.calls.append(("report", run_id))
        return {"report_revision": 2, "run_id": run_id}

    def delete(self, run_id):
        self.calls.append(("delete", run_id))


def test_submit_route_returns_canonical_accepted_record() -> None:
    service = _Service()
    client = _client(service)
    request = run_request()

    response = client.post(
        "/v1/training/runs",
        content=canonical_bytes(request),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 202
    assert response.content == canonical_bytes(service.accepted)
    assert service.requests == [request]


def test_submit_route_preserves_closed_service_failure() -> None:
    class _FailingService:
        def submit(self, _request):
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "training capacity is full",
            )

    response = _client(_FailingService()).post(
        "/v1/training/runs",
        content=canonical_bytes(run_request()),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 429
    assert response.json()["code"] == "RATE_LIMITED"


def test_submit_route_closes_and_logs_an_unexpected_service_defect(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _BrokenService:
        def submit(self, _request):
            raise ValueError("private implementation detail")

    with caplog.at_level(
        logging.ERROR,
        logger="histoforge.cloud.training.runs.routes",
    ):
        response = _client(_BrokenService()).post(
            "/v1/training/runs",
            content=canonical_bytes(run_request()),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "private implementation detail" not in response.text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "managed-training run submission failed unexpectedly"
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)
    assert str(record.exc_info[1]) == "private implementation detail"


def test_submit_route_logs_an_unavailable_service_cause_without_exposing_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _UnavailableService:
        def submit(self, _request):
            raise OSError("private storage endpoint detail")

    with caplog.at_level(
        logging.ERROR,
        logger="histoforge.cloud.training.runs.routes",
    ):
        response = _client(_UnavailableService()).post(
            "/v1/training/runs",
            content=canonical_bytes(run_request()),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 503
    assert response.json()["code"] == "SERVICE_UNAVAILABLE"
    assert "private storage endpoint detail" not in response.text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "managed-training run submission failed unexpectedly"
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], OSError)
    assert str(record.exc_info[1]) == "private storage endpoint detail"


def test_submit_route_rejects_oversized_or_invalid_json_body() -> None:
    client = _client(_Service(), max_json_request_bytes=16)

    oversized = client.post(
        "/v1/training/runs",
        content=b"{}" * 9,
        headers={"Content-Type": "application/json"},
    )
    invalid_json = client.post(
        "/v1/training/runs",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )

    assert oversized.status_code == 413
    assert invalid_json.status_code == 422


def test_run_access_routes_preserve_shapes_and_statuses() -> None:
    service = _Service()
    client = _client(service)
    run_id = service.accepted["run_id"]

    page = client.get("/v1/training/runs?limit=25&cursor=next")
    get = client.get(f"/v1/training/runs/{run_id}")
    cancel = client.post(f"/v1/training/runs/{run_id}/cancel")
    report = client.get(f"/v1/training/runs/{run_id}/report")
    delete = client.delete(f"/v1/training/runs/{run_id}")

    assert page.status_code == 200
    assert page.content == canonical_bytes({"items": []})
    assert get.content == canonical_bytes({"run_id": run_id})
    assert cancel.content == canonical_bytes({"run_id": run_id, "state": "CANCELLED"})
    assert report.content == canonical_bytes({"report_revision": 2, "run_id": run_id})
    assert delete.status_code == 204
    assert delete.content == b""
    assert service.calls == [
        ("page", 25, "next"),
        ("get", run_id),
        ("cancel", run_id),
        ("report", run_id),
        ("delete", run_id),
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/v1/training/runs",
        "/v1/training/runs?limit=0",
        "/v1/training/runs?limit=01",
        "/v1/training/runs?limit=1&limit=2",
        "/v1/training/runs?limit=1&unknown=x",
        "/v1/training/runs?limit=%31",
        "/v1/training/runs?%6cimit=1",
        "/v1/training/runs?limit=1&cursor=next%5Fpage",
        "/v1/training/runs?limit=1&cursor=%",
    ],
)
def test_run_page_rejects_noncanonical_or_unknown_query(path: str) -> None:
    response = _client(_Service()).get(path)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_run_routes_reject_noncanonical_ids_and_close_access_failures() -> None:
    client = _client(_Service())

    invalid = client.get("/v1/training/runs/not-a-uuid")
    encoded = client.get("/v1/training/runs/44444444-4444-4444-8444-444444444%34%34%34")

    assert invalid.status_code == 422
    assert encoded.status_code == 422

    class _BrokenAccess(_Service):
        def get(self, _run_id):
            raise ValueError("private access detail")

    broken = _client(_BrokenAccess()).get(
        "/v1/training/runs/44444444-4444-4444-8444-444444444444"
    )
    assert broken.status_code == 500
    assert "private access detail" not in broken.text


def _client(service, *, max_json_request_bytes: int = 1024**2) -> TestClient:
    web = FastAPI(redirect_slashes=False)

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request: Request, failure: GatewayFailure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    def require_token():
        return {"deployment_id": "test", "auth_revision": 1}

    register_routes(
        web,
        require_token=require_token,
        submission_service_factory=lambda _state: service,
        access_service_factory=lambda _state: service,
        max_json_request_bytes=max_json_request_bytes,
    )
    return TestClient(web)
