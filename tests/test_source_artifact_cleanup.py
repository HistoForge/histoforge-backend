"""Upload cleanup, quota accounting and interrupted-deletion regressions."""

import copy
import hashlib
import shutil
import uuid
from datetime import timedelta

import pytest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.records import source_record_key
from histoforge.cloud.sources.store import SourceArtifactStore

from test_source_artifact_store import FakeDict, _instant, _png_source, _upload_all


def test_fresh_reservation_checks_existing_uploads_once_before_rejecting_quota(
    tmp_path,
):
    now, records, files, store = _setup(tmp_path, source_count_limit=1)
    existing = _sealed_source(store, files, 1)
    fresh, _ = _source(2)
    before = copy.deepcopy(records)
    records.items_calls = 0
    records.get_calls.clear()

    with pytest.raises(GatewayFailure) as failure:
        store.reserve(fresh, files=files)

    assert failure.value.error["code"] == "RATE_LIMITED"
    assert records == before
    assert records.items_calls == 1
    assert records.get_calls.count(source_record_key(existing["artifact_id"])) == 0
    assert records.get_calls == [source_record_key(fresh["artifact_id"])]


def test_cleanup_checks_reservations_once_without_changing_live_uploads(tmp_path):
    now, records, files, store = _setup(tmp_path)
    existing = _sealed_source(store, files, 1)
    before = copy.deepcopy(records)
    records.items_calls = 0
    records.get_calls.clear()

    assert store.cleanup_expired(files) is None

    assert records == before
    assert records.items_calls == 1
    assert records.get_calls == []
    assert (tmp_path / "sources" / existing["artifact_id"] / "source.bin").exists()


def test_cleanup_removes_expired_and_orphan_sources_but_preserves_live_state(tmp_path):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    now[0] += _lifetime() / 2
    live = _sealed_source(store, files, 2)
    orphan, _ = _source(3)
    files.publish_source_chunk(orphan["artifact_id"], 0, b"orphan", _digest(b"orphan"))
    unrelated = {"unrelated": {"keep": True}, 42: "non-string key"}
    records.update(unrelated)
    expected = {
        key: copy.deepcopy(value)
        for key, value in records.items()
        if live["artifact_id"] in str(key) or key in unrelated
    }
    now[0] = _instant() + _lifetime()

    assert store.cleanup_expired(files) is None

    assert records == expected
    assert files.source_ids() == {live["artifact_id"]}
    assert not (tmp_path / "sources" / expired["artifact_id"]).exists()


@pytest.mark.parametrize("absolute_expired", [False, True])
def test_excluded_upload_skips_expiry_dependent_corruption_checks(
    tmp_path, absolute_expired
):
    now, records, files, store = _setup(tmp_path)
    protected = _sealed_source(store, files, 1)
    _record(records, protected)["sealed"]["manifest"]["manifest_digest"] = "f" * 64
    _record(records, protected)["last_activity_at"] = "invalid"
    if absolute_expired:
        now[0] += _lifetime()
    before = copy.deepcopy(records)

    assert store.cleanup_expired(files, exclude=protected["artifact_id"]) is None

    assert records == before
    assert files.source_ids() == {protected["artifact_id"]}


def test_invalid_excluded_reservation_prevents_any_cleanup_mutation(tmp_path):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    protected = _sealed_source(store, files, 2)
    _record(records, protected)["reservation"]["start_request_digest"] = "f" * 64
    now[0] += _lifetime()
    before = copy.deepcopy(records)

    with pytest.raises(GatewayFailure) as failure:
        store.cleanup_expired(files, exclude=protected["artifact_id"])

    assert failure.value.error["code"] == "INTERNAL_ERROR"
    assert failure.value.error["resource_id"] == protected["artifact_id"]
    assert records == before
    assert files.source_ids() == {expired["artifact_id"], protected["artifact_id"]}


def test_absolute_expiry_removes_corrupt_seal_and_upload_metadata(tmp_path):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    _record(records, expired)["sealed"] = {"invalid": True}
    _record(records, expired)["last_activity_at"] = "invalid"
    _record(records, expired)["chunk_commits"] = {"0": {"invalid": True}}
    now[0] += _lifetime()

    assert store.cleanup_expired(files) is None

    assert records == {}
    assert files.source_ids() == set()


@pytest.mark.parametrize("operation", ["cleanup", "reserve"])
def test_later_corrupt_seal_does_not_undo_an_earlier_expired_deletion(
    tmp_path, operation
):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    now[0] += _lifetime() / 2
    corrupt = _sealed_source(store, files, 2)
    _record(records, corrupt)["sealed"]["manifest"]["manifest_digest"] = "f" * 64
    retained = copy.deepcopy(records)
    retained = {
        key: value
        for key, value in retained.items()
        if expired["artifact_id"] not in key
    }
    now[0] = _instant() + _lifetime()
    fresh, _ = _source(3)

    with pytest.raises(GatewayFailure) as failure:
        if operation == "cleanup":
            store.cleanup_expired(files)
        else:
            store.reserve(fresh, files=files)

    assert failure.value.error["code"] == "INTERNAL_ERROR"
    assert failure.value.error["resource_id"] == corrupt["artifact_id"]
    assert records == retained
    assert files.source_ids() == {corrupt["artifact_id"]}


@pytest.mark.parametrize("failure_stage", ["control", "files"])
def test_interrupted_cleanup_prevents_admission_and_retries_orphan_deletion(
    tmp_path, monkeypatch, failure_stage
):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    fresh, _ = _source(2)
    now[0] += _lifetime()
    failed = False
    deleted = []
    original_pop = records.pop
    original_delete = shutil.rmtree

    def pop(key, default=None):
        nonlocal failed
        if (
            failure_stage == "control"
            and key == source_record_key(expired["artifact_id"])
            and not failed
        ):
            failed = True
            raise RuntimeError("injected control removal failure")
        return original_pop(key, default)

    def delete(path):
        nonlocal failed
        artifact_id = path.name
        assert not any(artifact_id in str(key) for key in records)
        if failure_stage == "files" and not failed:
            failed = True
            raise RuntimeError("injected file removal failure")
        deleted.append(artifact_id)
        return original_delete(path)

    monkeypatch.setattr(records, "pop", pop)
    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    with pytest.raises(RuntimeError, match="injected"):
        store.reserve(fresh, files=files)

    assert (source_record_key(expired["artifact_id"]) in records) == (
        failure_stage == "control"
    )
    assert source_record_key(fresh["artifact_id"]) not in records
    assert deleted == []
    assert files.source_ids() == {expired["artifact_id"]}

    accepted = store.reserve(fresh, files=files)

    assert accepted["artifact_id"] == fresh["artifact_id"]
    assert deleted == [expired["artifact_id"]]
    assert files.source_ids() == set()
    assert set(records) == {source_record_key(fresh["artifact_id"])}


@pytest.mark.parametrize("operation", ["cleanup", "reserve"])
def test_stale_activity_cache_keeps_recent_authoritative_chunk_live(
    tmp_path, operation
):
    now, records, files, store = _setup(tmp_path, source_count_limit=1)
    start, encoded = _source(1)
    store.reserve(start)
    idle = timedelta(
        seconds=DEPLOYMENT_CONFIG.source_artifact_limits.upload_idle_timeout_seconds
    )
    now[0] += idle - timedelta(seconds=1)
    store.accept_chunk(
        start["artifact_id"], 0, encoded[: start["chunk_size_bytes"]], files
    )
    _record(records, start)["last_activity_at"] = "2026-08-15T12:00:00.000Z"
    before = copy.deepcopy(records)
    now[0] += timedelta(seconds=2)

    if operation == "cleanup":
        assert store.cleanup_expired(files) is None
    else:
        fresh, _ = _source(2)
        with pytest.raises(GatewayFailure) as failure:
            store.reserve(fresh, files=files)
        assert failure.value.error["code"] == "RATE_LIMITED"

    assert records == before
    assert files.source_ids() == {start["artifact_id"]}


def test_expired_upload_no_longer_consumes_count_or_peak_quota(tmp_path):
    source, _ = _source(1)
    peak = 2 * source["total_bytes"] + (
        DEPLOYMENT_CONFIG.source_artifact_limits.max_start_canonical_bytes
    )
    now, records, files, store = _setup(
        tmp_path, source_count_limit=1, peak_byte_limit=peak
    )
    expired = _sealed_source(store, files, 1)
    now[0] += _lifetime()
    fresh, _ = _source(2)

    accepted = store.reserve(fresh, files=files)

    assert accepted["artifact_id"] == fresh["artifact_id"]
    assert set(records) == {source_record_key(fresh["artifact_id"])}
    assert not (tmp_path / "sources" / expired["artifact_id"]).exists()


def test_live_upload_still_consumes_peak_quota(tmp_path):
    source, _ = _source(1)
    peak = 2 * source["total_bytes"] + (
        DEPLOYMENT_CONFIG.source_artifact_limits.max_start_canonical_bytes
    )
    now, records, files, store = _setup(tmp_path, peak_byte_limit=peak)
    _sealed_source(store, files, 1)
    before = copy.deepcopy(records)
    fresh, _ = _source(2)

    with pytest.raises(GatewayFailure) as failure:
        store.reserve(fresh, files=files)

    assert failure.value.error["code"] == "RATE_LIMITED"
    assert "peak-byte" in failure.value.error["message"]
    assert records == before


def test_next_admission_observes_records_added_after_cleanup(tmp_path):
    now, records, files, store = _setup(tmp_path, source_count_limit=2)
    _sealed_source(store, files, 1)
    store.cleanup_expired(files)
    second, _ = _source(2)
    store.reserve(second)
    before = copy.deepcopy(records)
    third, _ = _source(3)

    with pytest.raises(GatewayFailure) as failure:
        store.reserve(third, files=files)

    assert failure.value.error["code"] == "RATE_LIMITED"
    assert records == before


def test_fresh_reservation_preserves_its_unreserved_source_directory(tmp_path):
    now, records, files, store = _setup(tmp_path)
    fresh, encoded = _source(1)
    body = encoded[: fresh["chunk_size_bytes"]]
    files.publish_source_chunk(fresh["artifact_id"], 0, body, _digest(body))

    accepted = store.reserve(fresh, files=files)

    assert accepted["artifact_id"] == fresh["artifact_id"]
    path = tmp_path / "sources" / fresh["artifact_id"] / "chunks" / "0.bin"
    assert path.read_bytes() == body


def test_same_id_replay_does_not_collect_other_expired_uploads(tmp_path):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    now[0] += _lifetime() / 2
    live = _sealed_source(store, files, 2)
    now[0] = _instant() + _lifetime()
    expected = store.reserve(live)
    before = copy.deepcopy(records)
    records.items_calls = 0

    assert store.reserve(live, files=files) == expected

    assert records == before
    assert records.items_calls == 0
    assert (tmp_path / "sources" / expired["artifact_id"]).exists()


def test_admission_without_files_filters_expiry_without_deleting_records(tmp_path):
    now, records, files, store = _setup(tmp_path, source_count_limit=1)
    expired = _sealed_source(store, files, 1)
    now[0] += _lifetime()
    fresh, _ = _source(2)

    assert store.reserve(fresh)["artifact_id"] == fresh["artifact_id"]

    assert source_record_key(expired["artifact_id"]) in records
    assert (tmp_path / "sources" / expired["artifact_id"]).exists()


def _setup(tmp_path, *, source_count_limit=None, peak_byte_limit=None):
    now = [_instant()]
    records = FakeDict()
    files = SourceFileStore(tmp_path)
    gateway_limits = DEPLOYMENT_CONFIG.limits.wire_value()
    if source_count_limit is not None:
        gateway_limits["max_active_source_artifacts"] = source_count_limit
    store = SourceArtifactStore(
        records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        gateway_limits,
        max_active_source_artifact_peak_bytes=(
            peak_byte_limit
            if peak_byte_limit is not None
            else DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=lambda: now[0],
    )
    return now, records, files, store


def _source(number):
    start, encoded = _png_source()
    start["artifact_id"] = str(uuid.UUID(int=number, version=4))
    return start, encoded


def _sealed_source(store, files, number):
    start, encoded = _source(number)
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    store.seal(start["artifact_id"], {"bytes_digest": _digest(encoded)}, files)
    return start


def _digest(body):
    return hashlib.sha256(body).hexdigest()


def _record(records, start):
    return records[source_record_key(start["artifact_id"])]


def _lifetime():
    return timedelta(
        seconds=DEPLOYMENT_CONFIG.source_artifact_limits.absolute_lifetime_seconds
    )
