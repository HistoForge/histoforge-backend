"""Saved README bytes stay pinned and bounded without changing model behavior."""

import json
import uuid
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.modules.catalog import (
    installed_module_descriptor,
    validate_installed_module_descriptor,
)
from histoforge.cloud.modules.identity import module_content_identity
from histoforge.cloud.modules.install import _installer
from histoforge.cloud.modules.mutation import execute_admin_job
from histoforge.cloud.modules.readme import MAX_README_BYTES, validate_readme_markdown
from tests.test_installer import (
    COMMIT,
    _recipe,
    _write_repository,
    fake_hf,
)
from tests.test_module_mutation import _jobs
from tests.test_module_store import _source, _store
from tests.test_publisher import (
    _FakeHub,
    _install_fake_hub,
    _publish,
)
from tests.test_publisher import (
    _store as _publication_store,
)

__all__ = ["fake_hf"]


def test_shared_readme_wire_shape_and_invalid_values():
    fixtures = Path(__file__).parent / "fixtures" / "contracts"
    fixture = json.loads((fixtures / "module-readme-v1.json").read_text())
    original = json.loads((fixtures / "module-catalog-v1.json").read_text())["page"][
        "modules"
    ][0]
    descriptor = {**original, "readme_markdown": fixture["readme_markdown"]}
    assert validate_installed_module_descriptor(descriptor) == descriptor
    assert (
        canonical_bytes({"readme_markdown": descriptor["readme_markdown"]}).decode()
        == fixture["canonical_field"]
    )
    assert descriptor["identity"] == original["identity"]
    for value in fixture["invalid_values"] + ["é" * MAX_README_BYTES, "\ud800"]:
        with pytest.raises(ValueError):
            validate_installed_module_descriptor({**original, "readme_markdown": value})
    descriptor["identity"] = {
        **original["identity"],
        "module_ref": "workspace/run-44444444-4444-4444-8444-444444444444",
    }
    del descriptor["identity"]["resolved_commit_sha"]
    with pytest.raises(ValueError, match="resolved HF commit"):
        validate_installed_module_descriptor(descriptor)


def _install(store, source, readme, *, commit=COMMIT):
    if readme is None:
        (source / "README.md").unlink(missing_ok=True)
    else:
        (source / "README.md").write_bytes(
            readme.encode("utf-8") if isinstance(readme, str) else readme
        )
    return store.install_from_directory(
        source,
        module_ref="owner/module",
        resolved_commit_sha=commit,
        installed_at="2026-08-15T12:34:56.789Z",
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
    )


def test_capture_uses_private_token_and_exact_commit_and_retains_readme_file(
    fake_hf,
):
    repository, calls = fake_hf
    _write_repository(repository, _recipe(), weights={"model.safetensors": b"weights"})
    markdown = "---\nlicense: mit\n---\n# Cells\n\nResolution: 0.22 µm/px.\n"
    (repository / "README.md").write_text(markdown, encoding="utf-8")
    download = _installer.download_module_snapshot(
        "owner/module", "branch", "secret", max_module_bytes=1_000_000
    )
    try:
        assert (download.directory / "README.md").read_text() == markdown
        assert calls["paths"] == [
            {
                "repo_id": "owner/module",
                "recursive": True,
                "revision": COMMIT,
                "repo_type": "model",
                "token": "secret",
            }
        ]
        assert sum(call["filename"] == "README.md" for call in calls["download"]) == 1
        assert all(call["revision"] == COMMIT for call in calls["download"])
        assert all(call["token"] == "secret" for call in calls["download"])
    finally:
        _installer.remove_download(download)


@pytest.mark.parametrize("content", [b"\xff", b"x" * (MAX_README_BYTES + 1)])
def test_invalid_card_fails_capture_and_removes_private_stage(
    fake_hf, monkeypatch, tmp_path, content
):
    repository, calls = fake_hf
    _write_repository(repository, _recipe(), weights={"model.safetensors": b"weights"})
    (repository / "README.md").write_bytes(content)
    stage = tmp_path / "download"
    monkeypatch.setattr(_installer.tempfile, "mkdtemp", lambda **_: str(stage))
    with pytest.raises(ValueError, match="README.md"):
        _installer.download_module_snapshot(
            "owner/module", "main", "secret", max_module_bytes=1_000_000
        )
    assert not stage.exists()
    if len(content) > MAX_README_BYTES:
        assert "README.md" not in [call["filename"] for call in calls["download"]]


def test_readme_bytes_change_content_identity_but_not_recipe_identity(tmp_path):
    store = _store(tmp_path / "modules")
    source = _source(tmp_path / "source")
    original = _install(store, source, "# Original\n")
    changed = _install(store, source, "# Updated\n")
    assert module_content_identity(changed) != module_content_identity(original)
    assert changed["recipe_digest"] == original["recipe_digest"]
    assert changed["executable_files"] == original["executable_files"]
    assert store.current.revalidate("owner/module") == changed
    descriptor = installed_module_descriptor(store.current, "owner/module")
    assert descriptor["readme_markdown"] == "# Updated\n"
    assert validate_installed_module_descriptor(descriptor) == descriptor
    assert descriptor["installed_module_file_bytes"] == sum(
        item["size_bytes"]
        for item in changed["executable_files"] + changed["additional_files"]
    )
    snapshot = tmp_path / "snapshot"
    store.current.snapshot("owner/module", snapshot)
    assert sorted(path.name for path in snapshot.iterdir()) == [
        "README.md",
        "model.safetensors",
        "recipe.json",
    ]
    without_card = _install(store, source, None, commit="d" * 40)
    assert "readme_markdown" not in without_card
    assert "readme_markdown" not in installed_module_descriptor(
        store.current, "owner/module"
    )
    store.delete("owner/module", job_id=str(uuid.uuid4()), publish=lambda: None)
    assert not store.current.list_module_refs()


def test_invalid_or_over_budget_card_preserves_current_module(tmp_path):
    store = _store(tmp_path / "modules")
    source = _source(tmp_path / "source")
    original = _install(store, source, "# Original")
    before_bytes = store.stored_bytes()
    for invalid in [b"\xff", b"x" * (MAX_README_BYTES + 1)]:
        with pytest.raises(ValueError):
            _install(store, source, invalid)
        assert store.current.revalidate("owner/module") == original
    store.max_catalog_encoded_bytes = len("# Small".encode())
    with pytest.raises(ValueError, match="catalog byte limit"):
        _install(store, source, "# Replacement")
    assert store.current.revalidate("owner/module") == original
    assert store.stored_bytes() == before_bytes


def test_utf8_limit_and_commit_requirement_are_enforced(tmp_path):
    assert validate_readme_markdown("") == ""
    assert validate_readme_markdown("é" * (MAX_README_BYTES // 2))
    with pytest.raises(ValueError, match="256 KiB"):
        validate_readme_markdown("é" * (MAX_README_BYTES // 2 + 1))
    record = _install(
        _store(tmp_path / "modules"),
        _source(tmp_path / "source"),
        "# Source card",
        commit=None,
    )
    assert "readme_markdown" not in record
    assert record["additional_files"][0]["relative_path"] == "README.md"


@pytest.mark.parametrize("existing", [b"# User card\n", None])
def test_sync_captures_preserved_target_card_not_the_source_or_draft(
    tmp_path, monkeypatch, existing
):
    store = _publication_store(tmp_path)
    files = {"recipe.json": canonical_bytes(_recipe())}
    if existing is not None:
        files["README.md"] = existing
    hub = _FakeHub(tmp_path, exists=True, files=files)
    _install_fake_hub(monkeypatch, hub)
    result = _publish(store, sync_local=True, card_markdown="# Do not use me")
    assert result.local_sync_error is None
    assert result.local_record.get("readme_markdown") == (
        existing.decode("utf-8") if existing is not None else None
    )
    assert store.current.revalidate("owner/target") == result.local_record


def test_capture_failure_after_publication_is_partial_and_preserves_current_target(
    tmp_path, monkeypatch
):
    store = _publication_store(tmp_path)
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)
    first = _publish(store, sync_local=True)
    hub.files["README.md"] = b"\xff"
    result = _publish(store, sync_local=True)
    assert result.commit_sha == hub.head
    assert result.local_record is None
    assert isinstance(result.local_sync_error, ValueError)
    assert store.current.revalidate("owner/target") == first.local_record


@pytest.mark.parametrize("failure_at", ["metadata", "download", "encoding"])
def test_readme_capture_failure_fails_install_job_and_preserves_current(
    fake_hf, tmp_path, monkeypatch, failure_at
):
    repository, _ = fake_hf
    _write_repository(repository, _recipe(), weights={"model.safetensors": b"weights"})
    (repository / "README.md").write_bytes(
        b"\xff" if failure_at == "encoding" else b"# Card"
    )
    store = _store(tmp_path / "modules")
    original = _install(store, _source(tmp_path / "source"), "# Original")
    stage = tmp_path / "download"
    monkeypatch.setattr(_installer.tempfile, "mkdtemp", lambda **_: str(stage))

    def unavailable(*args, **kwargs):
        raise OSError("private provider request failed")

    if failure_at == "metadata":
        monkeypatch.setattr(_installer.HfApi, "list_repo_tree", unavailable)
    elif failure_at == "download":
        monkeypatch.setattr(_installer, "hf_hub_download", unavailable)
    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id)
    published = []
    execute_admin_job(
        store,
        jobs,
        job_id,
        "INSTALL",
        {"module_ref": "owner/module"},
        lambda: published.append(True),
        hf_token="secret",
    )
    assert jobs[job_id]["phase"] == "FAILED"
    assert store.current.revalidate("owner/module") == original
    assert not stage.exists()
    assert published == []


def test_readme_download_failure_after_publish_reports_partial_job(
    tmp_path, monkeypatch
):
    store = _publication_store(tmp_path)
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)
    original = _publish(store, sync_local=True).local_record

    def unavailable(**kwargs):
        raise OSError("private README download failed")

    monkeypatch.setattr(_installer, "hf_hub_download", unavailable)
    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id, "PUBLISH", "owner/target")
    execute_admin_job(
        store,
        jobs,
        job_id,
        "PUBLISH",
        {
            "source_module": "owner/source",
            "target_repository": "owner/target",
            "visibility": "PRIVATE",
            "card_markdown": "# Draft",
            "sync_locally": True,
        },
        lambda: None,
        hf_token="secret",
    )
    assert jobs[job_id]["phase"] == "PARTIAL"
    assert jobs[job_id]["result"]["local_sync"]["type"] == "failed"
    assert store.current.revalidate("owner/target") == original
