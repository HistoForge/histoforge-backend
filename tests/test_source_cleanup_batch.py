"""Batched file deletion preserves upload fencing and interrupted-cleanup retry."""

import copy
import shutil

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.sources.records import source_record_key

from test_source_artifact_cleanup import (
    _digest,
    _lifetime,
    _record,
    _sealed_source,
    _setup,
    _source,
)
from test_source_artifact_store import _instant


def test_expired_and_orphan_files_use_one_commit_without_duplicate_control_removal(
    tmp_path, monkeypatch
):
    now, records, files, store = _setup(tmp_path)
    first = _sealed_source(store, files, 1)
    second = _sealed_source(store, files, 2)
    now[0] += _lifetime() / 2
    live = _sealed_source(store, files, 3)
    orphan, _ = _source(4)
    files.publish_source_chunk(orphan["artifact_id"], 0, b"orphan", _digest(b"orphan"))
    now[0] = _instant() + _lifetime()
    events = []
    original_pop = records.pop
    original_delete = shutil.rmtree

    def pop(key, default=None):
        events.append(("pop", key))
        return original_pop(key, default)

    def delete(path):
        assert source_record_key(path.name) not in records
        events.append(("delete", path.name))
        original_delete(path)

    monkeypatch.setattr(records, "pop", pop)
    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    files._reload_volume = lambda: events.append(("reload", None))
    files._commit_volume = lambda: events.append(("commit", None))

    store.cleanup_expired(files)

    expected = {first["artifact_id"], second["artifact_id"], orphan["artifact_id"]}
    assert {value for event, value in events if event == "pop"} == {
        source_record_key(first["artifact_id"]),
        source_record_key(second["artifact_id"]),
    }
    assert {value for event, value in events if event == "delete"} == expected
    assert events.count(("reload", None)) == 2
    assert events.count(("commit", None)) == 1
    assert events[-1] == ("commit", None)
    assert set(records) == {source_record_key(live["artifact_id"])}
    assert {path.name for path in (tmp_path / "sources").iterdir()} == {
        live["artifact_id"]
    }


@pytest.mark.parametrize("failure_stage", ["control", "partial_file_delete"])
def test_later_failure_commits_deleted_prefix_and_retry_removes_remaining_orphan(
    tmp_path, monkeypatch, failure_stage
):
    now, records, files, store = _setup(tmp_path)
    first = _sealed_source(store, files, 1)
    second = _sealed_source(store, files, 2)
    fresh, _ = _source(3)
    now[0] += _lifetime()
    second_root = tmp_path / "sources" / second["artifact_id"]
    commits = []
    failed = False
    original_pop = records.pop
    original_delete = shutil.rmtree

    def pop(key, default=None):
        nonlocal failed
        if (
            failure_stage == "control"
            and key == source_record_key(second["artifact_id"])
            and not failed
        ):
            failed = True
            raise RuntimeError("later control removal failed")
        return original_pop(key, default)

    def delete(path):
        nonlocal failed
        assert source_record_key(path.name) not in records
        if (
            failure_stage == "partial_file_delete"
            and path == second_root
            and not failed
        ):
            failed = True
            (path / "source.bin").unlink()
            raise RuntimeError("later partial file deletion failed")
        original_delete(path)

    monkeypatch.setattr(records, "pop", pop)
    monkeypatch.setattr("histoforge.cloud.sources.files.shutil.rmtree", delete)
    files._commit_volume = lambda: commits.append(
        {path.name for path in (tmp_path / "sources").iterdir()}
    )

    with pytest.raises(RuntimeError, match="later"):
        store.reserve(fresh, files=files)

    first_root = tmp_path / "sources" / first["artifact_id"]
    assert commits == ([] if failure_stage == "control" else [{second["artifact_id"]}])
    assert first_root.exists() == (failure_stage == "control")
    assert source_record_key(fresh["artifact_id"]) not in records
    assert (source_record_key(second["artifact_id"]) in records) == (
        failure_stage == "control"
    )
    assert store.reserve(fresh, files=files)["artifact_id"] == fresh["artifact_id"]
    assert commits == (
        [set()] if failure_stage == "control" else [{second["artifact_id"]}, set()]
    )
    assert not first_root.exists()
    assert not second_root.exists()
    assert set(records) == {source_record_key(fresh["artifact_id"])}


@pytest.mark.parametrize("publication_happened", [False, True])
def test_failed_final_commit_blocks_admission_and_retry_handles_ambiguous_publication(
    tmp_path, publication_happened
):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    fresh, _ = _source(2)
    now[0] += _lifetime()
    root = tmp_path / "sources" / expired["artifact_id"]
    original_files = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    restore_on_reload = False
    commits = 0

    def reload():
        nonlocal restore_on_reload
        if restore_on_reload:
            # An unacknowledged deletion may not have reached durable storage.
            for relative, content in original_files.items():
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            restore_on_reload = False

    def commit():
        nonlocal commits, restore_on_reload
        commits += 1
        if commits == 1:
            restore_on_reload = not publication_happened
            raise RuntimeError("final commit acknowledgement failed")

    files._reload_volume = reload
    files._commit_volume = commit

    with pytest.raises(RuntimeError, match="final commit"):
        store.reserve(fresh, files=files)

    assert source_record_key(expired["artifact_id"]) not in records
    assert source_record_key(fresh["artifact_id"]) not in records
    assert store.reserve(fresh, files=files)["artifact_id"] == fresh["artifact_id"]
    assert not root.exists()
    assert commits == (1 if publication_happened else 2)
    assert set(records) == {source_record_key(fresh["artifact_id"])}


def test_later_record_failure_remains_primary_if_prefix_commit_also_fails(tmp_path):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    now[0] += _lifetime() / 2
    corrupt = _sealed_source(store, files, 2)
    _record(records, corrupt)["sealed"]["manifest"]["manifest_digest"] = "f" * 64
    now[0] = _instant() + _lifetime()
    fresh, _ = _source(3)

    def commit():
        raise OSError("prefix commit failed")

    files._commit_volume = commit

    with pytest.raises(GatewayFailure) as failure:
        store.reserve(fresh, files=files)

    assert failure.value.error["resource_id"] == corrupt["artifact_id"]
    assert isinstance(failure.value.__cause__, OSError)
    assert str(failure.value.__cause__) == "prefix commit failed"
    assert not (tmp_path / "sources" / expired["artifact_id"]).exists()
    assert source_record_key(fresh["artifact_id"]) not in records
    assert (tmp_path / "sources" / corrupt["artifact_id"]).exists()


@pytest.mark.parametrize("failed_reload", [1, 2])
def test_reload_failure_does_not_remove_metadata_or_admit_an_image(
    tmp_path, failed_reload
):
    now, records, files, store = _setup(tmp_path)
    expired = _sealed_source(store, files, 1)
    fresh, _ = _source(2)
    now[0] += _lifetime()
    before = copy.deepcopy(records)
    reloads = 0

    def reload():
        nonlocal reloads
        reloads += 1
        if reloads == failed_reload:
            raise OSError("reload failed")

    files._reload_volume = reload

    with pytest.raises(OSError, match="reload failed"):
        store.reserve(fresh, files=files)

    assert records == before
    assert (tmp_path / "sources" / expired["artifact_id"]).exists()


def test_live_sources_need_no_commit_and_single_delete_invalidates_digest_check(
    tmp_path,
):
    now, records, files, store = _setup(tmp_path)
    live = _sealed_source(store, files, 1)
    path = tmp_path / "sources" / live["artifact_id"] / "source.bin"
    files._verified_sources[path] = ("unused test digest", (0, 0, 0, 0, 0))
    commits = []
    files._commit_volume = lambda: commits.append(True)

    store.cleanup_expired(files)

    assert commits == []
    assert path in files._verified_sources
    store.release(live["artifact_id"], files)
    assert commits == [True]
    assert path not in files._verified_sources
    assert not path.parent.exists()
