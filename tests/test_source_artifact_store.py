"""Upload retries, committed image bytes, and cancellation checks."""

import copy
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    canonical_chunk_range,
    source_artifact_start_digest,
)
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.records import source_record_key
from histoforge.cloud.sources.store import SourceArtifactStore


class FakeDict(dict):
    """Modal Dict subset with detached writes and injectable publication failure."""

    def __init__(self):
        super().__init__()
        self.write_failure = None
        self.items_calls = 0
        self.get_calls = []
        self.put_calls = []

    def get(self, key, default=None):
        self.get_calls.append(key)
        return super().get(key, default)

    def put(self, key, value, *, skip_if_exists=False):
        self.put_calls.append(key)
        if self.write_failure is not None and self.write_failure(value):
            self.write_failure = None
            raise RuntimeError("injected Dict publication failure")
        if skip_if_exists and key in self:
            return False
        self[key] = copy.deepcopy(value)
        return True

    def items(self):
        self.items_calls += 1
        return super().items()


def test_source_limit_accepts_32_gib_without_allocating_image_bytes():
    limits = DEPLOYMENT_CONFIG.source_artifact_limits
    start, _ = _png_source()
    start.update(
        container="TIFF_FAMILY",
        total_bytes=32 * 1024**3,
        chunk_size_bytes=limits.max_chunk_bytes,
    )
    records = FakeDict()
    store = _store(records)

    accepted = store.reserve(start)

    assert accepted["artifact_id"] == start["artifact_id"]
    assert (
        canonical_chunk_count(start["total_bytes"], start["chunk_size_bytes"]) == 4096
    )
    last = canonical_chunk_range(start["total_bytes"], start["chunk_size_bytes"], 4095)
    assert last["offset"] + last["size"] == 32 * 1024**3

    too_large = copy.deepcopy(start)
    too_large["artifact_id"] = str(uuid.uuid4())
    too_large["total_bytes"] += 1
    with pytest.raises(GatewayFailure) as failure:
        store.reserve(too_large)
    assert failure.value.error["code"] == "PAYLOAD_TOO_LARGE"
    assert len(records) == 1


def test_reservation_replay_conflict_and_capacity_are_exact(tmp_path):
    records = FakeDict()
    gateway_limits = DEPLOYMENT_CONFIG.limits.wire_value()
    gateway_limits["max_active_source_artifacts"] = 1
    store = SourceArtifactStore(
        records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        gateway_limits,
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=lambda: _instant(),
    )
    start, _ = _png_source()

    accepted = store.reserve(start)
    assert store.reserve(copy.deepcopy(start)) == accepted

    changed = copy.deepcopy(start)
    changed["creation_action_id"] = str(uuid.uuid4())
    with pytest.raises(GatewayFailure) as conflict:
        store.reserve(changed)
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"

    second, _ = _png_source()
    with pytest.raises(GatewayFailure) as limited:
        store.reserve(second)
    assert limited.value.error["code"] == "RATE_LIMITED"


def test_backend_only_peak_limit_bounds_active_sources_not_shared_volume():
    start, _ = _png_source()
    records = FakeDict()
    store = SourceArtifactStore(
        records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            2 * start["total_bytes"]
            + DEPLOYMENT_CONFIG.source_artifact_limits.max_start_canonical_bytes
        ),
        clock=lambda: _instant(),
    )
    accepted = store.reserve(start)
    assert store.reserve(copy.deepcopy(start)) == accepted
    second, _ = _png_source()
    with pytest.raises(GatewayFailure) as limited:
        store.reserve(second)
    assert limited.value.error["code"] == "RATE_LIMITED"
    assert "peak-byte" in limited.value.error["message"]
    assert len(records) == 1
    assert "max_artifact_stored_bytes" not in DEPLOYMENT_CONFIG.limits.wire_value()


def test_chunk_bytes_commit_before_record_and_replay_after_control_failure(tmp_path):
    records = FakeDict()
    events = []
    files = SourceFileStore(
        tmp_path,
        reload_volume=lambda: events.append("reload"),
        commit_volume=lambda: events.append("commit"),
    )
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    first = encoded[: start["chunk_size_bytes"]]

    records.write_failure = lambda value: bool(value["chunk_commits"])
    with pytest.raises(RuntimeError, match="injected Dict"):
        store.accept_chunk(start["artifact_id"], 0, first, files)

    chunk_path = tmp_path / "sources" / start["artifact_id"] / "chunks" / "0.bin"
    assert chunk_path.read_bytes() == first
    assert events[-1] == "commit"

    accepted = store.accept_chunk(start["artifact_id"], 0, first, files)
    assert accepted["chunk"]["sha256"] == hashlib.sha256(first).hexdigest()
    assert store.accept_chunk(start["artifact_id"], 0, first, files) == accepted

    records.items_calls = 0
    store.accept_chunk(start["artifact_id"], 0, first, files)
    assert records.items_calls == 0

    different = bytes([first[0] ^ 1]) + first[1:]
    with pytest.raises(GatewayFailure) as conflict:
        store.accept_chunk(start["artifact_id"], 0, different, files)
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"


def test_exact_chunk_replay_reports_committed_storage_corruption(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    body = encoded[: start["chunk_size_bytes"]]
    store.accept_chunk(start["artifact_id"], 0, body, files)

    chunk_path = tmp_path / "sources" / start["artifact_id"] / "chunks" / "0.bin"
    chunk_path.write_bytes(bytes([body[0] ^ 1]) + body[1:])

    with pytest.raises(GatewayFailure) as unavailable:
        store.accept_chunk(start["artifact_id"], 0, body, files)
    assert unavailable.value.error["code"] == "SERVICE_UNAVAILABLE"

    chunk_path.unlink()
    with pytest.raises(GatewayFailure) as missing:
        store.accept_chunk(start["artifact_id"], 0, body, files)
    assert missing.value.error["code"] == "SERVICE_UNAVAILABLE"


def test_release_before_chunk_control_commit_fences_the_late_publication(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    body = encoded[: start["chunk_size_bytes"]]

    def release_then_commit(artifact_id, chunk_id, digest, start_digest):
        store.release(artifact_id, files)
        return store.commit_chunk(artifact_id, chunk_id, digest, start_digest)

    with pytest.raises(GatewayFailure) as absent:
        store.accept_chunk(
            start["artifact_id"],
            0,
            body,
            files,
            control=SimpleNamespace(
                claim_chunk=store.claim_chunk,
                commit_chunk=release_then_commit,
            ),
        )
    assert absent.value.error["code"] == "ARTIFACT_NOT_FOUND"
    assert not any(start["artifact_id"] in str(key) for key in records)
    assert not (tmp_path / "sources" / start["artifact_id"]).exists()


def test_stale_authorization_keeps_unreferenced_bytes_for_safe_exact_replay(
    tmp_path,
):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    calls = 0

    def authorize():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise GatewayFailure("AUTHENTICATION_REQUIRED", "stale generation")

    with pytest.raises(GatewayFailure):
        store.accept_chunk(
            start["artifact_id"],
            0,
            encoded[: start["chunk_size_bytes"]],
            files,
            authorize=authorize,
        )
    chunk_path = tmp_path / "sources" / start["artifact_id"] / "chunks" / "0.bin"
    assert chunk_path.is_file()
    assert store.status(start["artifact_id"])["missing_chunks"][0] == 0

    accepted = store.accept_chunk(
        start["artifact_id"],
        0,
        encoded[: start["chunk_size_bytes"]],
        files,
    )
    assert (
        accepted["chunk"]["sha256"]
        == hashlib.sha256(chunk_path.read_bytes()).hexdigest()
    )


def test_losing_chunk_request_cannot_delete_a_concurrent_winner(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    body = encoded[: start["chunk_size_bytes"]]
    calls = 0

    def authorize():
        nonlocal calls
        calls += 1
        if calls == 2:
            winner = store.accept_chunk(start["artifact_id"], 0, body, files)
            assert winner["chunk"]["sha256"] == hashlib.sha256(body).hexdigest()
            raise GatewayFailure("AUTHENTICATION_REQUIRED", "stale generation")

    with pytest.raises(GatewayFailure):
        store.accept_chunk(
            start["artifact_id"],
            0,
            body,
            files,
            authorize=authorize,
        )

    chunk_path = tmp_path / "sources" / start["artifact_id"] / "chunks" / "0.bin"
    assert chunk_path.read_bytes() == body
    assert 0 not in store.status(start["artifact_id"])["missing_chunks"]


def test_complete_upload_seals_once_and_status_recovers_lost_response(tmp_path):
    records = FakeDict()
    commits = []
    files = SourceFileStore(
        tmp_path,
        commit_volume=lambda: commits.append("commit"),
    )
    store = _store(records)
    start, encoded = _png_source()
    accepted = store.reserve(start)
    _upload_all(store, files, start, encoded)

    records.items_calls = 0
    uploading = store.status(start["artifact_id"])
    assert uploading["phase"] == "UPLOADING"
    assert uploading["missing_chunks"] == []
    assert records.items_calls == 0

    request = {"bytes_digest": hashlib.sha256(encoded).hexdigest()}
    sealed = store.seal(start["artifact_id"], request, files)
    assert sealed["absolute_expires_at"] == accepted["absolute_expires_at"]
    assert sealed["artifact"]["bytes_digest"] == request["bytes_digest"]
    assert (
        tmp_path / "sources" / start["artifact_id"] / "source.bin"
    ).read_bytes() == encoded
    assert not (tmp_path / "sources" / start["artifact_id"] / "chunks").exists()

    records.items_calls = 0
    status = store.status(start["artifact_id"])
    assert status["phase"] == "SEALED"
    assert status["sealed_artifact"] == sealed["artifact"]
    assert records.items_calls == 0

    first = encoded[: start["chunk_size_bytes"]]
    replay = store.accept_chunk(start["artifact_id"], 0, first, files)
    assert replay["chunk"]["sha256"] == hashlib.sha256(first).hexdigest()
    assert not (tmp_path / "sources" / start["artifact_id"] / "chunks").exists()
    different = bytes([first[0] ^ 1]) + first[1:]
    with pytest.raises(GatewayFailure) as conflict:
        store.accept_chunk(start["artifact_id"], 0, different, files)
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"
    resolved = store.resolve_sealed_reference(sealed["artifact"])
    assert resolved["start_request"] == start
    assert resolved["artifact"] == sealed["artifact"]
    assert resolved["absolute_expires_at"] == sealed["absolute_expires_at"]
    assert (
        resolved["manifest"]["manifest_digest"] == sealed["artifact"]["manifest_digest"]
    )
    assert store.seal(start["artifact_id"], request, files) == sealed
    assert commits

    wrong_reference = copy.deepcopy(sealed["artifact"])
    wrong_reference["bytes_digest"] = "f" * 64
    with pytest.raises(GatewayFailure) as mismatch:
        store.resolve_sealed_reference(wrong_reference)
    assert mismatch.value.error["code"] == "ARTIFACT_PRECONDITION_FAILED"


def test_sealed_status_uses_validated_manifest_not_old_upload_commits(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    sealed = store.seal(
        start["artifact_id"],
        {"bytes_digest": hashlib.sha256(encoded).hexdigest()},
        files,
    )
    key = source_record_key(start["artifact_id"])
    records[key]["chunk_commits"]["0"] = {"invalid": True}

    records.items_calls = 0
    assert store.status(start["artifact_id"])["sealed_artifact"] == sealed["artifact"]
    assert records.items_calls == 0
    with pytest.raises(GatewayFailure) as invalid_chunk:
        store.accept_chunk(
            start["artifact_id"], 0, encoded[: start["chunk_size_bytes"]], files
        )
    assert invalid_chunk.value.error["code"] == "INTERNAL_ERROR"

    records[key]["sealed"]["manifest"]["manifest_digest"] = "f" * 64
    with pytest.raises(GatewayFailure) as invalid_manifest:
        store.status(start["artifact_id"])
    assert invalid_manifest.value.error["code"] == "INTERNAL_ERROR"
    assert records.items_calls == 0


def test_sealed_status_still_rejects_absolute_expiry(tmp_path):
    now = [_instant()]
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records, clock=lambda: now[0])
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    store.seal(
        start["artifact_id"],
        {"bytes_digest": hashlib.sha256(encoded).hexdigest()},
        files,
    )

    now[0] += timedelta(
        seconds=DEPLOYMENT_CONFIG.source_artifact_limits.absolute_lifetime_seconds
    )
    records.items_calls = 0
    with pytest.raises(GatewayFailure) as expired:
        store.status(start["artifact_id"])
    assert expired.value.error["code"] == "ARTIFACT_EXPIRED"
    assert records.items_calls == 0


@pytest.mark.parametrize("fail_removal", [False, True])
def test_release_uses_one_pop_and_can_retry_interrupted_removal(tmp_path, fail_removal):
    events = []

    class DeletionDict(FakeDict):
        fail_removal_key = None

        def pop(self, key, default=None):
            events.append(("remove", key))
            if self.fail_removal_key == key:
                self.fail_removal_key = None
                raise RuntimeError("injected Dict removal failure")
            return super().pop(key, default)

    class OrderedFiles(SourceFileStore):
        def delete_source(self, artifact_id):
            assert not removable.intersection(records)
            events.append(("delete_source", artifact_id))
            super().delete_source(artifact_id)

    records = DeletionDict()
    files = OrderedFiles(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    artifact_id = start["artifact_id"]
    key = source_record_key(artifact_id)
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    store.seal(
        artifact_id, {"bytes_digest": hashlib.sha256(encoded).hexdigest()}, files
    )

    # Removal also reclaims incomplete records without interpreting their contents.
    records[key]["chunk_claims"]["orphan"] = {"invalid": True}
    removable = set(records)
    unrelated = {
        "unrelated": {"keep": True},
        42: "non-string key",
        source_record_key(str(uuid.uuid4())): {"unrelated source": True},
    }
    records.update(unrelated)
    events.clear()
    records.items_calls = 0
    source_path = tmp_path / "sources" / artifact_id / "source.bin"

    if fail_removal:
        records.fail_removal_key = key
        with pytest.raises(RuntimeError, match="injected Dict removal"):
            store.release(artifact_id, files)
        assert key in records
        assert source_path.exists()
        assert not any(event[0] == "delete_source" for event in events)
        assert records.items_calls == 0
        records.items_calls = 0

    store.release(artifact_id, files)
    assert records.items_calls == 0
    assert events[0] == ("remove", key)
    assert events[-1] == ("delete_source", artifact_id)
    assert records == unrelated
    assert not source_path.exists()


def test_unsealed_reference_cannot_be_resolved_for_inference(tmp_path):
    records = FakeDict()
    store = _store(records)
    start, _encoded = _png_source()
    store.reserve(start)
    reference = {
        "artifact_id": start["artifact_id"],
        "creation_action_id": start["creation_action_id"],
        "start_request_digest": source_artifact_start_digest(start),
        "bytes_digest": "a" * 64,
        "manifest_digest": "b" * 64,
        "container": start["container"],
    }

    with pytest.raises(GatewayFailure) as unsealed:
        store.resolve_sealed_reference(reference)
    assert unsealed.value.error["code"] == "ARTIFACT_PRECONDITION_FAILED"


def test_sealed_control_is_visible_only_after_final_volume_commit(tmp_path):
    records = FakeDict()
    committed = False

    def commit_volume():
        nonlocal committed
        committed = True

    class OrderedDict(FakeDict):
        def put(self, key, value, *, skip_if_exists=False):
            if value["sealed"] is not None:
                assert committed
                root = tmp_path / "sources" / start["artifact_id"]
                assert (root / "source.bin").is_file()
                assert (root / "manifest.json").is_file()
            return super().put(key, value, skip_if_exists=skip_if_exists)

    records = OrderedDict()
    files = SourceFileStore(tmp_path, commit_volume=commit_volume)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    committed = False

    store.seal(
        start["artifact_id"],
        {"bytes_digest": hashlib.sha256(encoded).hexdigest()},
        files,
    )


def test_seal_publication_failure_is_recoverable_by_exact_replay(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    request = {"bytes_digest": hashlib.sha256(encoded).hexdigest()}

    records.write_failure = lambda value: value["sealed"] is not None
    with pytest.raises(RuntimeError, match="injected Dict"):
        store.seal(start["artifact_id"], request, files)
    assert store.status(start["artifact_id"])["phase"] == "UPLOADING"

    sealed = store.seal(start["artifact_id"], request, files)
    assert store.status(start["artifact_id"])["sealed_artifact"] == sealed["artifact"]


def test_release_before_seal_control_commit_fences_the_late_publication(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    request = {"bytes_digest": hashlib.sha256(encoded).hexdigest()}

    def release_then_commit(artifact_id, seal_request, start_digest, manifest):
        store.release(artifact_id, files)
        return store.commit_seal(
            artifact_id,
            seal_request,
            start_digest,
            manifest,
        )

    with pytest.raises(GatewayFailure) as absent:
        store.seal(
            start["artifact_id"],
            request,
            files,
            control=SimpleNamespace(
                claim_seal=store.claim_seal,
                commit_seal=release_then_commit,
            ),
        )
    assert absent.value.error["code"] == "ARTIFACT_NOT_FOUND"
    assert not any(start["artifact_id"] in str(key) for key in records)
    assert not (tmp_path / "sources" / start["artifact_id"]).exists()


def test_losing_seal_request_cannot_delete_a_concurrent_winner(tmp_path):
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = _store(records)
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    request = {"bytes_digest": hashlib.sha256(encoded).hexdigest()}
    calls = 0

    def authorize():
        nonlocal calls
        calls += 1
        if calls == 3:
            winner = store.seal(start["artifact_id"], request, files)
            assert winner["artifact"]["bytes_digest"] == request["bytes_digest"]
            raise GatewayFailure("AUTHENTICATION_REQUIRED", "stale generation")

    with pytest.raises(GatewayFailure):
        store.seal(start["artifact_id"], request, files, authorize=authorize)

    source_path = tmp_path / "sources" / start["artifact_id"] / "source.bin"
    assert source_path.read_bytes() == encoded
    assert store.status(start["artifact_id"])["phase"] == "SEALED"


def test_expired_present_then_cleaned_absent_and_durable_release(tmp_path):
    now = [_instant()]
    source_limits = DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
    source_limits["upload_idle_timeout_seconds"] = 5
    source_limits["absolute_lifetime_seconds"] = 20
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = SourceArtifactStore(
        records,
        source_limits,
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=lambda: now[0],
    )
    start, encoded = _png_source()
    store.reserve(start)
    store.accept_chunk(
        start["artifact_id"],
        0,
        encoded[: start["chunk_size_bytes"]],
        files,
    )

    now[0] += timedelta(seconds=6)
    with pytest.raises(GatewayFailure) as expired:
        store.status(start["artifact_id"])
    assert expired.value.error["code"] == "ARTIFACT_EXPIRED"

    store.cleanup_expired(files)
    with pytest.raises(GatewayFailure) as absent:
        store.status(start["artifact_id"])
    assert absent.value.error["code"] == "ARTIFACT_NOT_FOUND"
    assert not (tmp_path / "sources" / start["artifact_id"]).exists()

    fresh, fresh_bytes = _png_source()
    store.reserve(fresh)
    _upload_all(store, files, fresh, fresh_bytes)
    store.seal(
        fresh["artifact_id"],
        {"bytes_digest": hashlib.sha256(fresh_bytes).hexdigest()},
        files,
    )
    store.release(fresh["artifact_id"], files, has_durable_reference=lambda _id: True)
    assert store.status(fresh["artifact_id"])["phase"] == "SEALED"
    store.release(fresh["artifact_id"], files)
    with pytest.raises(GatewayFailure) as released:
        store.status(fresh["artifact_id"])
    assert released.value.error["code"] == "ARTIFACT_NOT_FOUND"


def test_stale_activity_cache_cannot_expire_a_recent_chunk(tmp_path):
    now = [_instant()]
    source_limits = DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
    source_limits["upload_idle_timeout_seconds"] = 5
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    store = SourceArtifactStore(
        records,
        source_limits,
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=lambda: now[0],
    )
    start, encoded = _png_source()
    store.reserve(start)

    now[0] += timedelta(seconds=4)
    store.accept_chunk(
        start["artifact_id"],
        0,
        encoded[: start["chunk_size_bytes"]],
        files,
    )
    records[source_record_key(start["artifact_id"])]["last_activity_at"] = (
        "2026-08-15T12:00:00.000Z"
    )

    now[0] += timedelta(seconds=2)
    status = store.status(start["artifact_id"])
    assert status["phase"] == "UPLOADING"
    assert status["upload_idle_expires_at"] == "2026-08-15T12:00:09.000Z"


def _store(records, *, clock=lambda: _instant()):
    return SourceArtifactStore(
        records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=clock,
    )


def _upload_all(store, files, start, encoded):
    chunk_size = start["chunk_size_bytes"]
    for chunk_id, offset in enumerate(range(0, len(encoded), chunk_size)):
        store.accept_chunk(
            start["artifact_id"],
            chunk_id,
            encoded[offset : offset + chunk_size],
            files,
        )


def _png_source():
    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    value = encoded.tobytes()
    start = {
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
            "source_rect_at_level": {"x": 0, "y": 0, "width": 5, "height": 4},
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
            {"series_index": 0, "resolution_level": 0, "width": 5, "height": 4}
        ],
    }
    return start, value


def _instant():
    return datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
