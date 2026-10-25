"""All-file retention through installation, result construction and publication."""

import uuid
from types import SimpleNamespace

import pytest
from histoforge.canonical_json import canonical_bytes, parse_json
from histoforge.cloud.modules.files import module_file_paths
from histoforge.cloud.modules.install import _installer
from histoforge.cloud.training.runs.result_module import read_training_result_candidate
from huggingface_hub import RepoFile
from tests.test_installer import _recipe, _write_repository, fake_hf
from tests.test_module_store import _source, _store
from tests.test_publisher import (
    _FakeHub,
    _install_fake_hub,
    _publish,
    _training_result_store,
)
from tests.test_training_dataset import training_api_limits

__all__ = ["fake_hf"]

RETAINED = {
    "README.md": b"---\nlicense: mit\n---\n# Original card\n",
    "LICENSE": b"Original license bytes\r\n",
    "NOTICE": b"Original notice",
    "docs/arbitrary-name.bin": b"\x00\xff\x80binary document",
    "examples/preview.png": b"\x89PNG\x00",
    "spare.safetensors": b"undeclared checkpoint, intentionally retained",
    "unused.groovy": b"not executable; no script header required",
}


def _write_files(root, files):
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def test_installer_retains_all_files_once_at_the_same_revision(fake_hf):
    root, calls = fake_hf
    _write_repository(root, _recipe(), weights={"model.safetensors": b"weights"})
    _write_files(root, RETAINED)
    download = _installer.download_module_snapshot(
        "owner/module", "main", "token", max_module_bytes=1_000_000
    )
    try:
        assert set(module_file_paths(download.directory)) == {
            "recipe.json",
            "model.safetensors",
            *RETAINED,
        }
        for path, expected in RETAINED.items():
            assert (download.directory / path).read_bytes() == expected
        filenames = [call["filename"] for call in calls["download"]]
        assert len(filenames) == len(set(filenames))
        assert {call["revision"] for call in calls["download"]} == {
            download.resolved_commit_sha
        }
    finally:
        _installer.remove_download(download)


def test_repository_budget_is_checked_before_any_file_download(fake_hf):
    root, calls = fake_hf
    _write_repository(root, _recipe(), weights={"model.safetensors": b"weights"})
    _write_files(root, {"docs/large.bin": b"x" * 1024})
    with pytest.raises(ValueError, match="max_module_bytes"):
        _installer.download_module_snapshot(
            "owner/module", None, "token", max_module_bytes=1000
        )
    assert calls["download"] == []


@pytest.mark.parametrize(
    "paths",
    [
        ["../escape"],
        ["/absolute"],
        ["bad\\path"],
        ["a//b"],
        [".histoforge/record.json"],
        [".HistoForge/record.json"],
        ["LICENSE", "license"],
        ["docs/a", "Docs/b"],
        ["docs", "docs/a"],
    ],
)
def test_remote_paths_are_rejected_before_filesystem_access(paths):
    api = SimpleNamespace(
        list_repo_tree=lambda **_: [
            RepoFile(path=path, size=1, oid="a" * 40) for path in paths
        ]
    )
    with pytest.raises(ValueError):
        _installer.repository_file_sizes(api, "owner/module", "a" * 40, "token")


@pytest.mark.parametrize("change", ["modify", "remove", "add"])
def test_saved_extra_files_are_part_of_exact_revalidation(tmp_path, change):
    store = _store(tmp_path / "modules")
    source = _source(tmp_path / "source")
    _write_files(source, RETAINED)
    store.install_from_directory(
        source,
        module_ref="owner/module",
        resolved_commit_sha="a" * 40,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
    )
    saved = store.current.module_dir("owner/module")
    path = saved / "docs/arbitrary-name.bin"
    if change == "modify":
        path.write_bytes(b"changed")
    elif change == "remove":
        (saved / "LICENSE").unlink()
    else:
        (saved / "extra.txt").write_bytes(b"new")
    with pytest.raises(ValueError, match="does not match installed bytes"):
        store.current.revalidate("owner/module")


@pytest.mark.parametrize("form", ["selected_model_only", "complete_source_workflow"])
def test_train_publish_install_round_trip_retains_nonmodel_files(
    tmp_path, monkeypatch, form
):
    store, candidate, _ = _training_result_store(
        tmp_path, result_form=form, retained_files=RETAINED
    )
    original = store.current.revalidate(candidate.module_ref)
    recipe = parse_json((candidate.module_dir / "recipe.json").read_bytes())
    assert ("cells" in recipe["models"]) == (form == "complete_source_workflow")
    assert (candidate.module_dir / "cells.safetensors").exists() == (
        form == "complete_source_workflow"
    )
    assert (
        candidate.module_dir / "nuclei.safetensors"
    ).read_bytes() == b"trained weights"
    assert (candidate.module_dir / "classify.groovy").exists()
    if form == "selected_model_only":
        assert recipe["steps"] == [{"model": "nuclei"}]
        assert store.current.script_sources(candidate.module_ref) == {}
    for path, expected in RETAINED.items():
        assert (candidate.module_dir / path).read_bytes() == expected
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)
    publication = _publish(
        store,
        source_ref=candidate.module_ref,
        card_markdown="# Do not replace source card",
    )
    assert store.current.revalidate(candidate.module_ref) == original
    for path, expected in RETAINED.items():
        assert hub.files[path] == expected
    download = _installer.download_module_snapshot(
        "owner/target", publication.commit_sha, "token", max_module_bytes=1_000_000
    )
    try:
        record = store.install_from_directory(
            download.directory,
            module_ref="owner/target",
            resolved_commit_sha=publication.commit_sha,
            job_id=str(uuid.uuid4()),
            publish=lambda: None,
        )
        assert record["readme_markdown"] == RETAINED["README.md"].decode()
        assert store.current.revalidate("owner/target") == record
        snapshot = tmp_path / "reinstalled-snapshot"
        store.current.snapshot("owner/target", snapshot)
        assert {
            path: (snapshot / path).read_bytes() for path in module_file_paths(snapshot)
        } == hub.files
    finally:
        _installer.remove_download(download)


def test_preserved_binary_is_bound_to_training_finalization(tmp_path):
    _, candidate, _ = _training_result_store(tmp_path, retained_files=RETAINED)
    (candidate.module_dir / "docs/arbitrary-name.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="module digest differs"):
        read_training_result_candidate(
            candidate.root,
            expected_run_id=candidate.report["run_id"],
            expected_module_content_digest=candidate.module_content_digest,
            expected_report_digest=candidate.report_digest,
            api_limits=training_api_limits(),
        )


def test_target_supporting_files_win_and_sync_matches_fresh_install(
    tmp_path, monkeypatch
):
    store = _store(tmp_path / "modules")
    source = _source(tmp_path / "source")
    _write_files(source, RETAINED)
    store.install_from_directory(
        source,
        module_ref="owner/source",
        resolved_commit_sha="a" * 40,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
    )
    target_only = {
        "user-notes.pdf": b"private notes",
        "unused-target.groovy": b"keep",
        "target-spare.safetensors": b"keep",
    }
    target_docs = {
        "README.md": b"# User edits on HF",
        "LICENSE": b"Target terms",
        "docs/arbitrary-name.bin": b"Target binary",
    }
    hub = _FakeHub(
        tmp_path,
        exists=True,
        files={
            "recipe.json": canonical_bytes(_recipe()),
            "model.safetensors": b"old weights",
            **target_only,
            **target_docs,
        },
    )
    _install_fake_hub(monkeypatch, hub)
    publication = _publish(store, sync_local=True)
    assert publication.local_sync_error is None
    assert hub.files["model.safetensors"] == b"weights"
    for path, expected in {**RETAINED, **target_only, **target_docs}.items():
        assert hub.files[path] == expected
    first = publication.local_record
    assert first["readme_markdown"] == "# User edits on HF"
    download = _installer.download_module_snapshot(
        "owner/target", publication.commit_sha, "token", max_module_bytes=1_000_000
    )
    try:
        fresh = store.install_from_directory(
            download.directory,
            module_ref="owner/target",
            resolved_commit_sha=publication.commit_sha,
            job_id=str(uuid.uuid4()),
            publish=lambda: None,
        )
        assert fresh["content_digest"] == first["content_digest"]
        assert fresh["additional_files"] == first["additional_files"]
    finally:
        _installer.remove_download(download)
    replay = _publish(store)
    assert replay.commit_sha == publication.commit_sha


def test_source_readme_fills_a_missing_path_in_existing_target(tmp_path, monkeypatch):
    store, candidate, _ = _training_result_store(tmp_path, retained_files=RETAINED)
    hub = _FakeHub(
        tmp_path,
        exists=True,
        files={
            "recipe.json": canonical_bytes(_recipe()),
            "model.safetensors": b"obsolete",
        },
    )
    _install_fake_hub(monkeypatch, hub)
    _publish(store, source_ref=candidate.module_ref)
    assert hub.files["README.md"] == RETAINED["README.md"]
    assert "model.safetensors" not in hub.files


@pytest.mark.parametrize("target_path", ["License", "docs"])
def test_merged_file_collisions_fail_before_upload(tmp_path, monkeypatch, target_path):
    store, candidate, _ = _training_result_store(tmp_path, retained_files=RETAINED)
    hub = _FakeHub(tmp_path, exists=True, files={target_path: b"keep me"})
    _install_fake_hub(monkeypatch, hub)
    with pytest.raises(ValueError, match="case-fold collision"):
        _publish(store, source_ref=candidate.module_ref)
    assert hub.commit_calls == []
    assert hub.files == {target_path: b"keep me"}
