"""Expired-file deletions settle before the next group, commit or admission."""

import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import pytest
from histoforge.cloud.artifacts.volume import artifact_volume_gate
from histoforge.cloud.sources.records import source_record_key

from test_source_artifact_cleanup import (
    _lifetime,
    _record,
    _sealed_source,
    _setup,
    _source,
)


def test_file_deletions_overlap_at_most_eight_and_finish_before_next_record_group(
    tmp_path, monkeypatch
):
    now, records, files, store = _setup(tmp_path, source_count_limit=32)
    expired = [_sealed_source(store, files, number) for number in range(1, 21)]
    now[0] += _lifetime()
    first_group = {item["artifact_id"] for item in expired[:8]}
    ready = Event()
    release = Event()
    lock = Lock()
    active = peak = 0
    started = set()
    finished = set()
    commits = []
    original_delete = shutil.rmtree
    original_pop = records.pop

    def pop(key, default=None):
        if key == source_record_key(expired[8]["artifact_id"]):
            with lock:
                assert first_group <= finished
                assert active == 0
        return original_pop(key, default)

    def delete(path):
        nonlocal active, peak
        assert artifact_volume_gate.locked()
        assert not any(source_record_key(item) in records for item in first_group)
        with lock:
            active += 1
            peak = max(peak, active)
            started.add(path.name)
            if started == first_group:
                ready.set()
        try:
            if path.name in first_group:
                assert release.wait(5), "test did not release file deletions"
            original_delete(path)
        finally:
            with lock:
                active -= 1
                finished.add(path.name)

    def commit():
        with lock:
            assert active == 0
            commits.append(finished.copy())
        assert artifact_volume_gate.locked()

    monkeypatch.setattr(records, "pop", pop)
    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    files._commit_volume = commit
    with ThreadPoolExecutor(max_workers=1) as caller:
        cleanup = caller.submit(store.cleanup_expired, files)
        try:
            assert ready.wait(5), "cleanup did not dispatch eight file deletions"
            assert started == first_group
            assert set(records) == {
                source_record_key(item["artifact_id"]) for item in expired[8:]
            }
            assert not cleanup.done()
            assert commits == []
            assert artifact_volume_gate.locked()
        finally:
            release.set()
        cleanup.result(timeout=5)

    assert active == 0
    assert peak == 8
    assert commits == [{item["artifact_id"] for item in expired}]
    assert records == {}
    assert files.source_ids() == set()


@pytest.mark.parametrize("commit_fails", [False, True])
def test_file_failure_waits_for_workers_commits_partial_changes_and_retries_orphans(
    tmp_path, monkeypatch, commit_fails
):
    now, records, files, store = _setup(tmp_path, source_count_limit=16)
    expired = [_sealed_source(store, files, number) for number in range(1, 10)]
    fresh, _ = _source(10)
    now[0] += _lifetime()
    first_group = {item["artifact_id"] for item in expired[:8]}
    failed_id = expired[0]["artifact_id"]
    ready = Event()
    release = Event()
    error_reported = Event()
    lock = Lock()
    started = set()
    finished = set()
    commits = []
    original_delete = shutil.rmtree

    def delete(path):
        with lock:
            started.add(path.name)
            if started == first_group:
                ready.set()
        try:
            if path.name == failed_id:
                (path / "source.bin").unlink()
                error_reported.set()
                raise RuntimeError("partial directory deletion failed")
            assert release.wait(5), "test did not release file deletions"
            original_delete(path)
        finally:
            with lock:
                finished.add(path.name)

    def commit():
        with lock:
            assert finished == first_group
        assert artifact_volume_gate.locked()
        commits.append({path.name for path in (tmp_path / "sources").iterdir()})
        if commit_fails:
            raise OSError("partial deletion commit failed")

    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    files._commit_volume = commit
    with ThreadPoolExecutor(max_workers=1) as caller:
        admission = caller.submit(store.reserve, fresh, files=files)
        try:
            assert ready.wait(5)
            assert error_reported.wait(5)
            assert not admission.done()
            assert commits == []
            assert source_record_key(expired[8]["artifact_id"]) in records
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="partial directory") as failure:
            admission.result(timeout=5)

    if commit_fails:
        assert isinstance(failure.value.__cause__, OSError)
        assert str(failure.value.__cause__) == "partial deletion commit failed"
    assert started == finished == first_group
    assert commits == [{failed_id, expired[8]["artifact_id"]}]
    assert set(records) == {source_record_key(expired[8]["artifact_id"])}
    assert source_record_key(fresh["artifact_id"]) not in records

    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", original_delete)
    files._commit_volume = lambda: None
    assert store.reserve(fresh, files=files)["artifact_id"] == fresh["artifact_id"]
    assert files.source_ids() == set()
    assert set(records) == {source_record_key(fresh["artifact_id"])}


def test_file_failure_precedes_a_later_iterator_validation_error(tmp_path, monkeypatch):
    now, records, files, store = _setup(tmp_path, source_count_limit=16)
    expired = [_sealed_source(store, files, number) for number in range(1, 5)]
    now[0] += _lifetime() / 2
    corrupt = _sealed_source(store, files, 5)
    _record(records, corrupt)["sealed"]["manifest"]["manifest_digest"] = "f" * 64
    now[0] += _lifetime() / 2
    fresh, _ = _source(6)
    failed_id = expired[1]["artifact_id"]
    original_delete = shutil.rmtree
    commits = []

    def delete(path):
        if path.name == failed_id:
            raise RuntimeError("earlier directory deletion failed")
        original_delete(path)

    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    files._commit_volume = lambda: commits.append(
        {path.name for path in (tmp_path / "sources").iterdir()}
    )
    with pytest.raises(RuntimeError, match="earlier directory"):
        store.reserve(fresh, files=files)

    assert commits == [{failed_id, corrupt["artifact_id"]}]
    assert set(records) == {source_record_key(corrupt["artifact_id"])}
    assert source_record_key(fresh["artifact_id"]) not in records


def test_missing_directories_do_not_prefetch_the_next_record_group(
    tmp_path, monkeypatch
):
    now, records, files, store = _setup(tmp_path, source_count_limit=16)
    expired = [_sealed_source(store, files, number) for number in range(1, 10)]
    now[0] += _lifetime()
    for item in expired[:7]:
        shutil.rmtree(tmp_path / "sources" / item["artifact_id"])
    blocked_id = expired[7]["artifact_id"]
    ready = Event()
    release = Event()
    original_delete = shutil.rmtree

    def delete(path):
        if path.name == blocked_id:
            ready.set()
            assert release.wait(5), "test did not release file deletion"
        original_delete(path)

    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    with ThreadPoolExecutor(max_workers=1) as caller:
        cleanup = caller.submit(store.cleanup_expired, files)
        try:
            assert ready.wait(5)
            assert source_record_key(expired[8]["artifact_id"]) in records
            assert not cleanup.done()
        finally:
            release.set()
        cleanup.result(timeout=5)

    assert records == {}
    assert files.source_ids() == set()
