"""Expired-record requests finish before file cleanup or lifecycle failure."""

import copy
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.sources.records import source_record_key

from test_source_artifact_cleanup import (
    _lifetime,
    _record,
    _sealed_source,
    _setup,
    _source,
)


def test_cleanup_overlaps_at_most_eight_removals_and_deletes_files_after_the_group(
    tmp_path, monkeypatch
):
    now, records, files, store = _setup(tmp_path, source_count_limit=32)
    expired = [_sealed_source(store, files, number) for number in range(1, 21)]
    now[0] += _lifetime()
    first_group = {source_record_key(item["artifact_id"]) for item in expired[:8]}
    ready = Event()
    release = Event()
    lock = Lock()
    active = peak = 0
    started = []
    original_pop = records.pop
    original_delete = shutil.rmtree

    def pop(key, default=None):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(key)
            if set(started) == first_group:
                ready.set()
        try:
            if key in first_group:
                assert release.wait(5), "test did not release the first group"
            return original_pop(key, default)
        finally:
            with lock:
                active -= 1

    def delete(path):
        with lock:
            assert active == 0
        assert source_record_key(path.name) not in records
        original_delete(path)

    monkeypatch.setattr(records, "pop", pop)
    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    with ThreadPoolExecutor(max_workers=1) as caller:
        cleanup = caller.submit(store.cleanup_expired, files)
        try:
            assert ready.wait(5), "cleanup did not dispatch eight requests"
            assert len(started) == 8
            assert {path.name for path in (tmp_path / "sources").iterdir()} == {
                item["artifact_id"] for item in expired
            }
        finally:
            release.set()
        cleanup.result(timeout=5)

    assert peak == 8
    assert active == 0
    assert len(started) == 20
    assert records == {}
    assert files.source_ids() == set()


@pytest.mark.parametrize("removed_before_error", [False, True])
def test_failed_group_waits_for_other_requests_blocks_admission_and_retries_orphans(
    tmp_path, monkeypatch, removed_before_error
):
    now, records, files, store = _setup(tmp_path, source_count_limit=32)
    expired = [_sealed_source(store, files, number) for number in range(1, 10)]
    fresh, _ = _source(10)
    now[0] += _lifetime()
    failed_key = source_record_key(expired[1]["artifact_id"])
    first_group = {source_record_key(item["artifact_id"]) for item in expired[:8]}
    ready = Event()
    release = Event()
    error_reported = Event()
    lock = Lock()
    started = set()
    finished = set()
    original_pop = records.pop

    def pop(key, default=None):
        with lock:
            started.add(key)
            if started == first_group:
                ready.set()
        try:
            if key == failed_key:
                if removed_before_error:
                    original_pop(key, default)
                error_reported.set()
                raise RuntimeError("expired record removal failed")
            assert release.wait(5), "test did not release pending requests"
            return original_pop(key, default)
        finally:
            with lock:
                finished.add(key)

    monkeypatch.setattr(records, "pop", pop)
    with ThreadPoolExecutor(max_workers=1) as caller:
        admission = caller.submit(store.reserve, fresh, files=files)
        try:
            assert ready.wait(5)
            assert error_reported.wait(5)
            assert not admission.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="expired record removal"):
            admission.result(timeout=5)

    assert started == finished == first_group
    assert source_record_key(fresh["artifact_id"]) not in records
    assert (failed_key in records) != removed_before_error
    assert files.source_ids() == {item["artifact_id"] for item in expired}
    assert source_record_key(expired[8]["artifact_id"]) in records

    monkeypatch.setattr(records, "pop", original_pop)
    assert store.reserve(fresh, files=files)["artifact_id"] == fresh["artifact_id"]
    assert set(records) == {source_record_key(fresh["artifact_id"])}
    assert files.source_ids() == set()


def test_concurrent_cleanup_never_dispatches_past_a_known_corrupt_record(
    tmp_path, monkeypatch
):
    now, records, files, store = _setup(tmp_path, source_count_limit=16)
    before = [_sealed_source(store, files, number) for number in range(1, 5)]
    later = _sealed_source(store, files, 6)
    now[0] += _lifetime() / 2
    corrupt = _sealed_source(store, files, 5)
    _record(records, corrupt)["sealed"]["manifest"]["manifest_digest"] = "f" * 64
    now[0] += _lifetime() / 2
    expected = {
        source_record_key(item["artifact_id"]): copy.deepcopy(_record(records, item))
        for item in [corrupt, later]
    }
    removed = []
    original_pop = records.pop

    def pop(key, default=None):
        removed.append(key)
        return original_pop(key, default)

    monkeypatch.setattr(records, "pop", pop)
    with pytest.raises(GatewayFailure) as failure:
        store.cleanup_expired(files)

    assert failure.value.error["resource_id"] == corrupt["artifact_id"]
    assert set(removed) == {source_record_key(item["artifact_id"]) for item in before}
    assert records == expected
    assert files.source_ids() == {corrupt["artifact_id"], later["artifact_id"]}


def test_failed_second_group_commits_first_group_and_preserves_the_removal_error(
    tmp_path, monkeypatch
):
    now, records, files, store = _setup(tmp_path, source_count_limit=16)
    expired = [_sealed_source(store, files, number) for number in range(1, 10)]
    fresh, _ = _source(10)
    now[0] += _lifetime()
    failed_key = source_record_key(expired[8]["artifact_id"])
    original_pop = records.pop
    commit_states = []

    def pop(key, default=None):
        if key == failed_key:
            raise RuntimeError("second group removal failed")
        return original_pop(key, default)

    def commit():
        commit_states.append({path.name for path in (tmp_path / "sources").iterdir()})
        raise OSError("partial deletion commit failed")

    monkeypatch.setattr(records, "pop", pop)
    files._commit_volume = commit
    with pytest.raises(RuntimeError, match="second group") as failure:
        store.reserve(fresh, files=files)

    assert isinstance(failure.value.__cause__, OSError)
    assert str(failure.value.__cause__) == "partial deletion commit failed"
    assert commit_states == [{expired[8]["artifact_id"]}]
    assert set(records) == {failed_key}
    assert source_record_key(fresh["artifact_id"]) not in records
