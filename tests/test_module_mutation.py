"""One-lane module administration and typed terminal outcomes."""

import json
import uuid

from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.modules.install._installer import DownloadedModule
from histoforge.cloud.modules.mutation import execute_admin_job
from histoforge.cloud.modules.publish._publisher import PublicationResult
from histoforge.cloud.modules.store import ModuleStore


def _recipe():
    return {
        "schemaVersion": 3,
        "name": "Example",
        "models": {
            "model": {
                "arch": "dinov3-adapter-cellpose",
                "gpu": "A10G",
                "weights": {"type": "vendored", "file": "model.safetensors"},
                "input": {"type": "field", "micronsPerPixel": 0.22},
                "output": {
                    "type": "instance_segmentation",
                    "classes": ["Cell"],
                },
                "params": {"num_classes": 0, "min_instance_pixels": 15},
            }
        },
        "steps": [{"model": "model"}],
    }


def _source(path):
    path.mkdir()
    (path / "recipe.json").write_text(json.dumps(_recipe()), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def _store(tmp_path):
    return ModuleStore(
        tmp_path / "modules",
        max_module_bytes=1_000_000,
        max_installed_bytes=10_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
    )


def _jobs(job_id, job_type="INSTALL", requested="owner/module"):
    return {
        job_id: {
            "type": job_type,
            "requested_module_ref": requested,
            "phase": "QUEUED",
            "updated_at": "2026-08-15T01:02:03.004Z",
            "conflict_refs": [requested],
        }
    }


def test_install_downloads_then_commits_through_store(
    tmp_path,
    monkeypatch,
):
    from histoforge.cloud.modules.install import _installer

    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id)
    source = _source(tmp_path / "download")
    removed = []
    credential = {}
    readme = "# Saved private card\n"
    (source / "README.md").write_text(readme)

    def download(module_ref, revision, token, *, max_module_bytes):
        assert max_module_bytes == 1_000_000
        credential["download"] = (module_ref, revision, token)
        return DownloadedModule("owner/module", "a" * 40, source)

    monkeypatch.setattr(
        _installer,
        "download_module_snapshot",
        download,
    )
    monkeypatch.setattr(_installer, "remove_download", removed.append)
    publications = []
    store = _store(tmp_path)

    execute_admin_job(
        store,
        jobs,
        job_id,
        "INSTALL",
        {"module_ref": "owner/module", "hugging_face_revision": "main"},
        lambda: publications.append(True),
        hf_token="server-token",
    )

    assert jobs[job_id]["phase"] == "SUCCEEDED"
    result = jobs[job_id]["result"]
    assert result["type"] == "installed"
    assert result["installed_module"]["module_ref"] == "owner/module"
    assert len(result["installed_module"]["content_digest"]) == 64
    assert removed == [DownloadedModule("owner/module", "a" * 40, source)]
    assert store.current.revalidate("owner/module")["readme_markdown"] == readme
    assert publications
    assert credential["download"] == ("owner/module", "main", "server-token")


def test_publish_projects_success_and_separate_local_sync_failure(
    tmp_path,
    monkeypatch,
):
    from histoforge.cloud.modules.publish import _publisher

    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id, "PUBLISH", "owner/target")
    commit = "b" * 40

    called = {}

    def publish(*args, **kwargs):
        called.update(kwargs)
        return PublicationResult(commit, None, OSError("local sync failed"))

    monkeypatch.setattr(
        _publisher,
        "publish_module",
        publish,
    )

    execute_admin_job(
        _store(tmp_path),
        jobs,
        job_id,
        "PUBLISH",
        {
            "source_module": "owner/source",
            "target_repository": "owner/target",
            "visibility": "PRIVATE",
            "card_markdown": "# Card\n",
            "sync_locally": True,
        },
        lambda: None,
        hf_token="server-token",
    )

    assert jobs[job_id]["phase"] == "PARTIAL"
    result = jobs[job_id]["result"]
    assert result["type"] == "published"
    assert result["commit_sha"] == commit
    assert result["target_repository"] == "owner/target"
    assert result["local_sync"]["type"] == "failed"
    assert result["local_sync"]["failure"]["code"] == "SERVICE_UNAVAILABLE"
    assert called["token"] == "server-token"
    assert called["source_ref"] == "owner/source"
    assert called["target_repo"] == "owner/target"


def test_unknown_operation_fails_visibly(tmp_path):
    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id)

    execute_admin_job(
        _store(tmp_path),
        jobs,
        job_id,
        "TRAIN",
        {},
        lambda: None,
        hf_token=None,
    )

    assert jobs[job_id]["phase"] == "FAILED"
    assert jobs[job_id]["failure"]["code"] == "INTERNAL_ERROR"
    assert "TRAIN" not in jobs[job_id]["failure"]["message"]


def test_missing_server_credential_fails_job_without_registry_call(
    tmp_path, monkeypatch
):
    from histoforge.cloud.modules.install import _installer

    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id)
    called = []
    monkeypatch.setattr(
        _installer,
        "download_module_snapshot",
        lambda *args: called.append(args),
    )

    execute_admin_job(
        _store(tmp_path),
        jobs,
        job_id,
        "INSTALL",
        {"module_ref": "owner/module"},
        lambda: None,
        hf_token="",
    )

    assert called == []
    assert jobs[job_id]["phase"] == "FAILED"
    assert jobs[job_id]["failure"]["code"] == "SERVICE_UNAVAILABLE"
    assert jobs[job_id]["failure"]["message"] == (
        "server Hugging Face credential is not configured"
    )


def test_stale_auth_generation_fails_before_registry_or_module_mutation(
    tmp_path, monkeypatch
):
    from histoforge.cloud.modules.install import _installer

    job_id = str(uuid.uuid4())
    jobs = _jobs(job_id)
    called = []
    monkeypatch.setattr(
        _installer,
        "download_module_snapshot",
        lambda *args: called.append(args),
    )

    def reject_stale_generation():
        raise GatewayFailure(
            GatewayErrorCode.AUTHENTICATION_REQUIRED,
            "the authenticated deployment generation is no longer current",
        )

    execute_admin_job(
        _store(tmp_path),
        jobs,
        job_id,
        "INSTALL",
        {"module_ref": "owner/module"},
        lambda: None,
        hf_token="server-token",
        authorize=reject_stale_generation,
    )

    assert called == []
    assert jobs[job_id]["phase"] == "FAILED"
    assert jobs[job_id]["failure"]["code"] == "AUTHENTICATION_REQUIRED"
