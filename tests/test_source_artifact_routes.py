"""Authenticated source lifecycle and generic artifact HTTP behavior."""

import hashlib
import io
import uuid
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure, gateway_error_http_status
from histoforge.cloud.artifacts.routes import (
    register_routes as register_artifact_routes,
)
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.original import BIOFORMATS_ADAPTER
from histoforge.cloud.sources.routes import (
    register_routes as register_source_routes,
)
from histoforge.cloud.sources.store import SourceArtifactStore


class FakeDict(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class LocalLifecycle:
    """Route-test adapter for the production lifecycle-owner contract."""

    def __init__(self, store, files):
        self._store = store
        self._files = files

    def reserve(self, start_request):
        return self._store.reserve(start_request, files=self._files)

    def release(self, artifact_id):
        return self._store.release(artifact_id, self._files)

    def claim_chunk(self, artifact_id, chunk_id, body_size, digest):
        return self._store.claim_chunk(artifact_id, chunk_id, body_size, digest)

    def commit_chunk(self, artifact_id, chunk_id, digest, start_request_digest):
        return self._store.commit_chunk(
            artifact_id,
            chunk_id,
            digest,
            start_request_digest,
        )

    def claim_seal(self, artifact_id, seal_request):
        return self._store.claim_seal(artifact_id, seal_request)

    def commit_seal(
        self,
        artifact_id,
        seal_request,
        start_request_digest,
        manifest,
    ):
        return self._store.commit_seal(
            artifact_id,
            seal_request,
            start_request_digest,
            manifest,
        )


def test_source_http_workflow_and_release_are_exact(tmp_path):
    client, _files, _store = _client(tmp_path)
    start, encoded = _png_source()

    accepted = client.post("/v1/source-artifacts", json=start)
    assert accepted.status_code == 200
    assert accepted.json()["artifact_id"] == start["artifact_id"]

    chunk_size = start["chunk_size_bytes"]
    for chunk_id, offset in enumerate(range(0, len(encoded), chunk_size)):
        response = client.put(
            f"/v1/source-artifacts/{start['artifact_id']}/chunks/{chunk_id}",
            content=encoded[offset : offset + chunk_size],
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200
        assert response.json()["chunk"]["chunk_id"] == chunk_id

    status = client.get(f"/v1/source-artifacts/{start['artifact_id']}")
    assert status.status_code == 200
    assert status.json()["missing_chunks"] == []

    sealed = client.post(
        f"/v1/source-artifacts/{start['artifact_id']}/seal",
        json={"bytes_digest": hashlib.sha256(encoded).hexdigest()},
    )
    assert sealed.status_code == 200
    assert sealed.json()["artifact"]["artifact_id"] == start["artifact_id"]
    assert (
        client.get(f"/v1/source-artifacts/{start['artifact_id']}").json()["phase"]
        == "SEALED"
    )

    assert (
        client.delete(f"/v1/source-artifacts/{start['artifact_id']}").status_code == 204
    )
    assert (
        client.delete(f"/v1/source-artifacts/{start['artifact_id']}").status_code == 204
    )
    missing = client.get(f"/v1/source-artifacts/{start['artifact_id']}")
    assert missing.status_code == 404
    assert missing.json()["code"] == "ARTIFACT_NOT_FOUND"


def test_source_routes_reject_noncanonical_paths_and_chunk_transport(tmp_path):
    client, _files, _store = _client(tmp_path)
    start, encoded = _png_source()
    assert client.post("/v1/source-artifacts", json=start).status_code == 200
    base = f"/v1/source-artifacts/{start['artifact_id']}/chunks"

    assert (
        client.put(
            f"{base}/00",
            content=encoded[: start["chunk_size_bytes"]],
            headers={"Content-Type": "application/octet-stream"},
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"{base}/0",
            content=encoded[: start["chunk_size_bytes"]],
            headers={"Content-Type": "application/octet-stream; charset=binary"},
        ).status_code
        == 422
    )
    encoded_alias = start["artifact_id"].replace("a", "%61", 1)
    if encoded_alias != start["artifact_id"]:
        assert client.get(f"/v1/source-artifacts/{encoded_alias}").status_code == 422
    assert (
        client.get(f"/v1/source-artifacts/{start['artifact_id']}/").status_code == 404
    )


def test_chunk_route_enforces_the_exact_bounded_body(tmp_path):
    client, _files, _store = _client(tmp_path)
    start, encoded = _png_source()
    assert client.post("/v1/source-artifacts", json=start).status_code == 200
    path = f"/v1/source-artifacts/{start['artifact_id']}/chunks/0"
    expected = encoded[: start["chunk_size_bytes"]]

    oversized = client.put(
        path,
        content=expected + b"x",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert oversized.status_code == 413
    assert oversized.json()["code"] == "PAYLOAD_TOO_LARGE"

    short = client.put(
        path,
        content=expected[:-1],
        headers={"Content-Type": "application/octet-stream"},
    )
    assert short.status_code == 422
    assert short.json()["code"] == "VALIDATION_ERROR"
    assert (
        0
        in client.get(f"/v1/source-artifacts/{start['artifact_id']}").json()[
            "missing_chunks"
        ]
    )


def test_expired_status_is_410_until_a_later_reserve_cleans_it(tmp_path):
    now = [datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)]
    client, _files, _store = _client(tmp_path, clock=lambda: now[0])
    start, _encoded = _png_source()
    assert client.post("/v1/source-artifacts", json=start).status_code == 200

    now[0] = now[0].replace(hour=14)
    expired = client.get(f"/v1/source-artifacts/{start['artifact_id']}")
    assert expired.status_code == 410
    assert expired.json()["code"] == "ARTIFACT_EXPIRED"

    fresh, _fresh_bytes = _png_source()
    assert client.post("/v1/source-artifacts", json=fresh).status_code == 200
    absent = client.get(f"/v1/source-artifacts/{start['artifact_id']}")
    assert absent.status_code == 404
    assert absent.json()["code"] == "ARTIFACT_NOT_FOUND"


def test_seal_rejects_bytes_that_do_not_match_the_declared_container(tmp_path):
    client, _files, _store = _client(tmp_path)
    start, encoded = _png_source()
    invalid = b"not-png!" + encoded[8:]
    assert len(invalid) == len(encoded)
    assert client.post("/v1/source-artifacts", json=start).status_code == 200

    chunk_size = start["chunk_size_bytes"]
    for chunk_id, offset in enumerate(range(0, len(invalid), chunk_size)):
        response = client.put(
            f"/v1/source-artifacts/{start['artifact_id']}/chunks/{chunk_id}",
            content=invalid[offset : offset + chunk_size],
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 200

    rejected = client.post(
        f"/v1/source-artifacts/{start['artifact_id']}/seal",
        json={"bytes_digest": hashlib.sha256(invalid).hexdigest()},
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "VALIDATION_ERROR"
    assert (
        client.get(f"/v1/source-artifacts/{start['artifact_id']}").json()["phase"]
        == "UPLOADING"
    )


def test_malformed_fileset_zip_is_a_422_validation_error(tmp_path):
    client, _files, _store = _client(tmp_path)
    start, _encoded = _png_source()
    contents = {"sidecars/empty.dat": b"", "source.vsi": b"entry"}
    members = [
        {
            "path": name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in contents.items()
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    # Preserve the bounded central directory, but disagree with one local header.
    invalid = buffer.getvalue().replace(b"source.vsi", b"broken.vsi", 1)
    source = start["origin"]["source"]
    source["source_adapter_id"] = BIOFORMATS_ADAPTER
    source["content_version"] = {
        "type": "local-file-set",
        "members": [
            {
                "path": member["path"],
                "size_bytes": member["size_bytes"],
                "last_modified_epoch_millis": 1,
            }
            for member in members
        ],
    }
    start.update(
        container="SLIDE_FILE_SET",
        total_bytes=len(invalid),
        chunk_size_bytes=len(invalid),
        origin={
            "type": "original-wsi",
            "source": source,
            "pyramid_series_index": 0,
            "level_zero_pixels_to_slide": start["origin"]["artifact_pixels_to_slide"],
            "entry_path": "source.vsi",
            "reader_settings": {
                "levels": [
                    {
                        "resolution_level": 0,
                        "native_resolution_index": 0,
                        "pixels_to_level_zero": start["origin"][
                            "artifact_pixels_to_slide"
                        ],
                    }
                ]
            },
            "members": members,
        },
        required_levels=[
            {
                "series_index": 0,
                "resolution_level": 0,
                "width": source["width"],
                "height": source["height"],
            }
        ],
    )
    assert client.post("/v1/source-artifacts", json=start).status_code == 200
    base = f"/v1/source-artifacts/{start['artifact_id']}"
    assert (
        client.put(
            f"{base}/chunks/0",
            content=invalid,
            headers={"Content-Type": "application/octet-stream"},
        ).status_code
        == 200
    )
    rejected = client.post(
        f"{base}/seal", json={"bytes_digest": hashlib.sha256(invalid).hexdigest()}
    )
    assert rejected.status_code == 422
    assert rejected.json()["code"] == "VALIDATION_ERROR"
    assert client.get(base).json()["phase"] == "UPLOADING"
    root = tmp_path / "sources" / start["artifact_id"]
    assert not (root / "slide").exists()
    assert not (root / ".slide-stage").exists()


def test_generic_artifact_complete_and_range_reads(tmp_path):
    client, files, _store = _client(tmp_path)
    payload = bytes(range(251)) * 9000
    reference = files.publish_artifact_bytes(
        payload,
        media_type="application/octet-stream",
    )
    path = f"/v1/artifacts/{reference['artifact_id']}"

    complete = client.get(path)
    assert complete.status_code == 200
    assert complete.content == payload
    assert complete.headers["content-length"] == str(len(payload))
    assert complete.headers["etag"] == f'"{reference["sha256"]}"'
    assert complete.headers["accept-ranges"] == "bytes"

    partial = client.get(
        path,
        headers={
            "Range": "bytes=17-999999999999999999999999999999999",
            "If-Match": f'"{reference["sha256"]}"',
        },
    )
    assert partial.status_code == 206
    assert partial.content == payload[17:]
    assert (
        partial.headers["content-range"]
        == f"bytes 17-{len(payload) - 1}/{len(payload)}"
    )

    missing_precondition = client.get(path, headers={"Range": "bytes=0-9"})
    assert missing_precondition.status_code == 412
    assert missing_precondition.json()["code"] == "ARTIFACT_PRECONDITION_FAILED"

    malformed = client.get(
        path,
        headers={
            "Range": "bytes=0-1,3-4",
            "If-Match": f'"{reference["sha256"]}"',
        },
    )
    assert malformed.status_code == 416
    assert malformed.json()["code"] == "RANGE_NOT_SATISFIABLE"

    absent = client.get(f"/v1/artifacts/{uuid.uuid4()}")
    assert absent.status_code == 404
    assert absent.json()["code"] == "ARTIFACT_NOT_FOUND"

    empty = files.publish_artifact_bytes(
        b"",
        media_type="application/octet-stream",
    )
    empty_response = client.get(f"/v1/artifacts/{empty['artifact_id']}")
    assert empty_response.status_code == 200
    assert empty_response.content == b""
    assert empty_response.headers["content-length"] == "0"


def test_generic_artifact_rejects_same_length_corruption_before_headers(tmp_path):
    client, files, _store = _client(tmp_path)
    payload = b"correct immutable bytes"
    reference = files.publish_artifact_bytes(
        payload,
        media_type="application/octet-stream",
    )
    data_path = tmp_path / "files" / reference["artifact_id"] / "data.bin"
    data_path.write_bytes(b"x" * len(payload))
    path = f"/v1/artifacts/{reference['artifact_id']}"

    complete = client.get(path)
    assert complete.status_code == 503
    assert complete.json()["code"] == "SERVICE_UNAVAILABLE"

    partial = client.get(
        path,
        headers={
            "Range": "bytes=0-3",
            "If-Match": f'"{reference["sha256"]}"',
        },
    )
    assert partial.status_code == 503
    assert partial.json()["code"] == "SERVICE_UNAVAILABLE"


def _client(
    tmp_path,
    *,
    clock=lambda: datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc),
):
    records = FakeDict()
    files = ArtifactFileStore(tmp_path)
    source_files = SourceFileStore(tmp_path)
    store = SourceArtifactStore(
        records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=clock,
    )
    state = {"deployment_id": "deployment-a", "auth_revision": 1}
    web = FastAPI(redirect_slashes=False)

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request: Request, failure: GatewayFailure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    def require_token():
        return state

    register_source_routes(
        web,
        require_token=require_token,
        store=store,
        files=source_files,
        max_json_request_bytes=DEPLOYMENT_CONFIG.limits.max_json_request_bytes,
        lifecycle=lambda _state: LocalLifecycle(store, source_files),
    )
    register_artifact_routes(web, require_token=require_token, files=files)
    return TestClient(web), files, store


def _png_source():
    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    value = encoded.tobytes()
    return (
        {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "creation_action_id": str(uuid.uuid4()),
            "artifact_id": str(uuid.uuid4()),
            "container": "PNG",
            "total_bytes": len(value),
            "chunk_size_bytes": max(1, len(value) // 2),
            "origin": {
                "type": "derived-region",
                "source": {
                    "source_adapter_id": "qupath.test",
                    "credential_free_descriptor_digest": "1" * 64,
                    "width": 100,
                    "height": 80,
                    "z_slices": 1,
                    "timepoints": 1,
                    "calibration": {"mpp_x": 0.25, "mpp_y": 0.25},
                    "content_version": {
                        "type": "s3-etag",
                        "quoted_value": '"source-v1"',
                    },
                },
                "source_resolution_level": 0,
                "plane": {"z": 0, "t": 0},
                "source_rect_at_level": {
                    "x": 0,
                    "y": 0,
                    "width": 5,
                    "height": 4,
                },
                "artifact_pixels_to_slide": {
                    "scale_x": 1,
                    "scale_y": 1,
                    "translate_x": 0,
                    "translate_y": 0,
                },
                "writer_build": {
                    "version": "test",
                    "artifact_digest": "2" * 64,
                    "manifest_digest": "3" * 64,
                },
            },
            "required_levels": [
                {
                    "series_index": 0,
                    "resolution_level": 0,
                    "width": 5,
                    "height": 4,
                }
            ],
        },
        value,
    )
