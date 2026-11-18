from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure, gateway_error_http_status
from histoforge.cloud.training.datasets.routes import register_routes
from tests.test_training_dataset_service import (
    _archive,
    _chunks,
    _service,
    _start_request,
)
from tests.test_training_dataset_state import DATASET_ID


def test_training_dataset_http_workflow_uses_canonical_transport(
    tmp_path: Path,
) -> None:
    service, _control, _files = _service(tmp_path)
    client = _client(service)
    archive, manifest = _archive(tmp_path)
    request = _start_request(archive, manifest["dataset_digest"], chunk_size=211)

    started = client.post("/v1/training/datasets", json=request)
    assert started.status_code == 200
    assert started.content == canonical_bytes(started.json())
    for index, body in _chunks(archive, request["chunk_size_bytes"]):
        response = client.put(
            f"/v1/training/datasets/{DATASET_ID}/chunks/{index}",
            content=body,
            headers={
                "Content-Type": "application/octet-stream",
                "X-HistoForge-Chunk-Sha256": hashlib.sha256(body).hexdigest(),
            },
        )
        assert response.status_code == 200
        assert response.content == canonical_bytes(response.json())

    sealed = client.post(
        f"/v1/training/datasets/{DATASET_ID}/seal",
        json={
            "schema_version": 1,
            "start_request_digest": started.json()["start_request_digest"],
        },
    )
    assert sealed.status_code == 200
    assert sealed.json()["dataset_digest"] == manifest["dataset_digest"]
    assert client.get("/v1/training/datasets?limit=10").status_code == 200
    assert client.get(f"/v1/training/datasets/{DATASET_ID}").status_code == 200
    assert client.delete(f"/v1/training/datasets/{DATASET_ID}").status_code == 204
    assert client.get(f"/v1/training/datasets/{DATASET_ID}").status_code == 404


def test_chunk_route_rejects_noncanonical_or_inconsistent_transport(
    tmp_path: Path,
) -> None:
    service, _control, _files = _service(tmp_path)
    client = _client(service, max_chunk_bytes=4)
    archive = b"abcd"
    request = _start_request(archive, "1" * 64, chunk_size=4)
    assert client.post("/v1/training/datasets", json=request).status_code == 200
    route = f"/v1/training/datasets/{DATASET_ID}/chunks/0"
    valid = {
        "Content-Type": "application/octet-stream",
        "X-HistoForge-Chunk-Sha256": hashlib.sha256(archive).hexdigest(),
    }

    missing = client.put(
        route, content=archive, headers={"Content-Type": valid["Content-Type"]}
    )
    assert missing.status_code == 422
    wrong_type = client.put(
        route,
        content=archive,
        headers={**valid, "Content-Type": "application/octet-stream; charset=binary"},
    )
    assert wrong_type.status_code == 422
    wrong_digest = client.put(
        route,
        content=archive,
        headers={**valid, "X-HistoForge-Chunk-Sha256": "2" * 64},
    )
    assert wrong_digest.status_code == 422
    oversized = client.put(route, content=archive + b"e", headers=valid)
    assert oversized.status_code == 413
    alias = client.put(
        route[:-1] + "00",
        content=archive,
        headers=valid,
    )
    assert alias.status_code == 422


def test_dataset_routes_require_canonical_ids_and_bounded_list_query(
    tmp_path: Path,
) -> None:
    service, _control, _files = _service(tmp_path)
    client = _client(service)

    assert client.get("/v1/training/datasets").status_code == 422
    assert client.get("/v1/training/datasets?limit=0").status_code == 422
    assert client.get("/v1/training/datasets?limit=01").status_code == 422
    assert client.get("/v1/training/datasets?limit=1&extra=x").status_code == 422
    assert client.get("/v1/training/datasets?limit=1&limit=2").status_code == 422
    assert client.get("/v1/training/datasets?limit=%31").status_code == 422
    assert client.get("/v1/training/datasets?%6cimit=1").status_code == 422
    assert (
        client.get("/v1/training/datasets?limit=1&cursor=next%5Fpage").status_code
        == 422
    )
    assert client.get("/v1/training/datasets?limit=1").status_code == 200

    encoded_alias = DATASET_ID.replace("1", "%31", 1)
    assert client.get(f"/v1/training/datasets/{encoded_alias}").status_code == 422
    assert client.get(f"/v1/training/datasets/{DATASET_ID}/").status_code == 404


def test_invalid_start_and_seal_documents_are_validation_errors(tmp_path: Path) -> None:
    service, _control, _files = _service(tmp_path)
    client = _client(service)

    invalid_start = client.post("/v1/training/datasets", json={"schema_version": 1})
    assert invalid_start.status_code == 422
    assert invalid_start.json()["code"] == "VALIDATION_ERROR"

    invalid_seal = client.post(
        f"/v1/training/datasets/{DATASET_ID}/seal",
        json={"schema_version": 1},
    )
    assert invalid_seal.status_code == 422
    assert invalid_seal.json()["code"] == "VALIDATION_ERROR"


def _client(service, *, max_chunk_bytes: int = 8 * 1024**2) -> TestClient:
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
        service_factory=lambda _state: service,
        max_json_request_bytes=1024**2,
        max_chunk_bytes=max_chunk_bytes,
    )
    return TestClient(web)
