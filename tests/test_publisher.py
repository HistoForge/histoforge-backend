"""Atomic verified Hugging Face module publication and local-sync tests."""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.dinov3_training import project_dinov3_training_model
from histoforge.cloud.inference.route_activation import current_model_activation
from histoforge.cloud.modules.install import _installer
from histoforge.cloud.modules.publish import _publisher
from histoforge.cloud.modules.store import ModuleStore
from histoforge.cloud.training.runs.result_module import build_training_result_candidate
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, RepoFile
from huggingface_hub.errors import HfHubHTTPError
from tests.test_training_dataset import training_api_limits
from tests.test_training_result_module import _accepted_source, _success
from tests.test_training_source_preparation import RUN_ID

SOURCE_COMMIT = "a" * 40


def _recipe(name="Example"):
    return {
        "schemaVersion": 3,
        "name": name,
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


def _source(path: Path, *, recipe=None, weights=b"weights") -> Path:
    path.mkdir()
    (path / "recipe.json").write_text(json.dumps(recipe or _recipe()), encoding="utf-8")
    if weights is not None:
        (path / "model.safetensors").write_bytes(weights)
    return path


def _store(tmp_path) -> ModuleStore:
    store = ModuleStore(
        tmp_path / "modules",
        max_module_bytes=1_000_000,
        max_installed_bytes=10_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
    )
    store.install_from_directory(
        _source(tmp_path / "source"),
        module_ref="owner/source",
        resolved_commit_sha=SOURCE_COMMIT,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        installed_at="2026-08-15T12:34:56.789Z",
    )
    return store


class _FakeHub:
    def __init__(
        self, tmp_path: Path, *, exists=False, files=None, cache_symlinks=False
    ):
        self.tmp_path = tmp_path
        self.exists = exists
        self.files = dict(files or {})
        self.cache_symlinks = cache_symlinks
        self.commit_index = 1
        self.head = "1" * 40
        self.create_calls = []
        self.commit_calls = []

    def api_class(self, token):
        hub = self

        class Api:
            def create_repo(self, **kwargs):
                hub.create_calls.append(kwargs)
                if hub.exists:
                    request = httpx.Request("POST", "https://huggingface.test")
                    response = httpx.Response(409, request=request)
                    raise HfHubHTTPError("exists", response=response)
                hub.exists = True
                hub.files.setdefault(".gitattributes", b"platform")
                return SimpleNamespace(repo_id=kwargs["repo_id"])

            def repo_info(self, **kwargs):
                return SimpleNamespace(sha=hub.head)

            def list_repo_files(self, **kwargs):
                return sorted(hub.files)

            def list_repo_tree(self, **kwargs):
                assert kwargs["revision"] == hub.head
                return [
                    RepoFile(path=path, size=len(value), oid="c" * 40)
                    for path, value in hub.files.items()
                ]

            def get_paths_info(self, **kwargs):
                assert kwargs["revision"] == hub.head
                (name,) = kwargs["paths"]
                if name not in hub.files:
                    return []
                return [RepoFile(path=name, size=len(hub.files[name]), oid="c" * 40)]

            def create_commit(self, **kwargs):
                assert kwargs["parent_commit"] == hub.head
                hub.commit_calls.append(kwargs)
                next_files = dict(hub.files)
                for operation in kwargs["operations"]:
                    if isinstance(operation, CommitOperationDelete):
                        next_files.pop(operation.path_in_repo, None)
                    elif isinstance(operation, CommitOperationAdd):
                        source = operation.path_or_fileobj
                        next_files[operation.path_in_repo] = (
                            source
                            if isinstance(source, bytes)
                            else Path(source).read_bytes()
                        )
                    else:
                        raise AssertionError(type(operation))
                if next_files != hub.files:
                    hub.commit_index += 1
                    hub.head = f"{hub.commit_index:040x}"
                    hub.files = next_files
                return SimpleNamespace(oid=hub.head)

        return Api()

    def download(self, **kwargs):
        path = self.tmp_path / "downloads" / kwargs["revision"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[kwargs["filename"]])
        if self.cache_symlinks:
            link = self.tmp_path / "snapshots" / kwargs["revision"] / kwargs["filename"]
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.is_symlink():
                try:
                    link.symlink_to(path)
                except OSError as error:
                    pytest.skip(f"cache symlinks unavailable: {error}")
            return str(link)
        return str(path)


def _install_fake_hub(monkeypatch, hub):
    monkeypatch.setattr(_publisher, "HfApi", hub.api_class)
    monkeypatch.setattr(_publisher, "hf_hub_download", hub.download)
    monkeypatch.setattr(_installer, "HfApi", hub.api_class)
    monkeypatch.setattr(_installer, "hf_hub_download", hub.download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", hub.download)


def _publish(store, **overrides):
    values = {
        "source_ref": "owner/source",
        "target_repo": "owner/target",
        "visibility": "PRIVATE",
        "card_markdown": "# Card\n",
        "sync_local": False,
        "token": "server-token",
        "job_id": str(uuid.uuid4()),
        "publish_volume": lambda: None,
    }
    values.update(overrides)
    return _publisher.publish_module(store, **values)


def _training_result_store(
    tmp_path, *, result_form="selected_model_only", retained_files=None
):
    request, marker, source_root = _accepted_source(
        tmp_path, result_form=result_form, retained_files=retained_files
    )
    report, weights = _success(tmp_path, request, marker)
    candidate = build_training_result_candidate(
        tmp_path / "result-candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )
    store = ModuleStore(
        tmp_path / "trained-modules",
        max_module_bytes=1_000_000,
        max_installed_bytes=10_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
        training_api_limits=training_api_limits(),
    )
    record = store.install_from_directory(
        candidate.module_dir,
        module_ref=candidate.module_ref,
        resolved_commit_sha=None,
        job_id=RUN_ID,
        publish=lambda: None,
        installed_at="2026-09-07T12:00:00.000Z",
        external_artifacts=candidate.external_artifacts,
        training_result_attestation=candidate.attestation,
    )
    assert record["content_digest"] == candidate.module_content_digest
    return store, candidate, report


def test_first_publish_commits_exact_snapshot_and_initial_card(tmp_path, monkeypatch):
    store = _store(tmp_path)
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)

    result = _publish(store)

    assert result.commit_sha == hub.head
    assert result.local_record is None and result.local_sync_error is None
    assert hub.files["README.md"] == b"# Card\n"
    assert hub.files["model.safetensors"] == b"weights"
    assert json.loads(hub.files["recipe.json"])["name"] == "Example"
    assert len(hub.commit_calls) == 1
    assert hub.commit_calls[0]["commit_message"] == "histoforge publish owner/target"
    assert hub.create_calls[0]["private"] is True


@pytest.mark.parametrize("sync_local", [False, True])
def test_publication_verifies_hub_cache_symlinks(tmp_path, monkeypatch, sync_local):
    store = _store(tmp_path)
    hub = _FakeHub(tmp_path, cache_symlinks=True)
    _install_fake_hub(monkeypatch, hub)

    result = _publish(store, sync_local=sync_local)

    assert result.commit_sha == hub.head
    assert result.local_sync_error is None
    if sync_local:
        assert result.local_record == store.current.read_record("owner/target")
        installed_weights = store.current.artifact_path("owner/target", "model")
        assert not installed_weights.is_symlink()
        assert installed_weights.read_bytes() == b"weights"
    else:
        assert result.local_record is None


@pytest.mark.parametrize(
    "result_form", ["selected_model_only", "complete_source_workflow"]
)
def test_run_derived_result_publishes_without_installing_a_second_module(
    tmp_path, monkeypatch, result_form
):
    store, candidate, report = _training_result_store(tmp_path, result_form=result_form)
    source_ref = candidate.module_ref
    # Compare against the private candidate, not the publisher's own snapshot.
    # This also catches omitted workflow files or accidentally included run data.
    module_files = {
        path.relative_to(candidate.module_dir).as_posix(): path.read_bytes()
        for path in candidate.module_dir.rglob("*")
        if path.is_file()
    }
    original = store.current.revalidate(source_ref)
    original_activation = current_model_activation(
        store.current,
        source_ref,
        "nuclei",
        lambda run_id: run_id == RUN_ID,
    )
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)

    result = _publish(store, source_ref=source_ref)

    assert result.commit_sha == hub.head
    assert result.local_record is None and result.local_sync_error is None
    assert hub.files == {
        **module_files,
        ".gitattributes": b"platform",
        "README.md": b"# Card\n",
    }
    assert hub.files["training-report.json"] == canonical_bytes(report)
    assert hub.files["nuclei.safetensors"] == b"trained weights"
    assert len(hub.commit_calls) == 1
    assert hub.create_calls[0]["private"] is True
    assert store.current.list_module_refs() == [source_ref]
    assert store.current.revalidate(source_ref) == original
    assert (
        current_model_activation(
            store.current,
            source_ref,
            "nuclei",
            lambda run_id: run_id == RUN_ID,
        )
        == original_activation
    )


@pytest.mark.parametrize(
    "result_form", ["selected_model_only", "complete_source_workflow"]
)
def test_failed_run_derived_publication_leaves_private_result_runnable(
    tmp_path, monkeypatch, result_form
):
    store, candidate, _report = _training_result_store(
        tmp_path, result_form=result_form
    )
    source_ref = candidate.module_ref
    original = store.current.revalidate(source_ref)
    original_weights = store.current.artifact_path(
        source_ref, "nuclei", "weights"
    ).read_bytes()
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)
    download = hub.download

    def corrupt_report(**kwargs):
        path = Path(download(**kwargs))
        if kwargs["filename"] == "training-report.json":
            path.write_bytes(b"corrupt")
        return str(path)

    monkeypatch.setattr(_publisher, "hf_hub_download", corrupt_report)

    with pytest.raises(GatewayFailure) as caught:
        _publish(store, source_ref=source_ref)

    assert caught.value.error["code"] == "REGISTRY_PUBLICATION_CONFLICT"
    assert store.current.list_module_refs() == [source_ref]
    assert store.current.revalidate(source_ref) == original
    assert (
        store.current.artifact_path(source_ref, "nuclei", "weights").read_bytes()
        == original_weights
    )
    activation = current_model_activation(
        store.current,
        source_ref,
        "nuclei",
        lambda run_id: run_id == RUN_ID,
    )
    assert activation["backend"]["module"]["module_ref"] == source_ref


def test_retry_preserves_supporting_files_and_deletes_only_declared_old_weights(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    existing_card = b"# User-edited card\n"
    previous = _recipe()
    previous["models"]["old"] = previous["models"].pop("model")
    previous["models"]["old"]["weights"]["file"] = "old.safetensors"
    previous["steps"] = [{"model": "old"}]
    hub = _FakeHub(
        tmp_path,
        exists=True,
        files={
            ".gitattributes": b"platform",
            "README.md": existing_card,
            "recipe.json": canonical_bytes(previous),
            "old.groovy": b"stale",
            "old.safetensors": b"stale",
            "notes.txt": b"preserve",
        },
    )
    _install_fake_hub(monkeypatch, hub)

    _publish(store, card_markdown="# Replacement card\n")

    assert hub.files["README.md"] == existing_card
    assert hub.files["notes.txt"] == b"preserve"
    assert hub.files["old.groovy"] == b"stale"
    assert "old.safetensors" not in hub.files
    deletes = {
        operation.path_in_repo
        for operation in hub.commit_calls[0]["operations"]
        if isinstance(operation, CommitOperationDelete)
    }
    assert deletes == {"old.safetensors"}


def test_exact_replay_accepts_existing_verified_head_without_new_commit(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)
    first = _publish(store)
    first_head = hub.head

    second = _publish(store)

    assert first.commit_sha == second.commit_sha == first_head
    assert hub.head == first_head
    assert len(hub.commit_calls) == 2


def test_sync_local_installs_exact_verified_target_commit(tmp_path, monkeypatch):
    store = _store(tmp_path)
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)

    result = _publish(store, sync_local=True)

    assert result.local_sync_error is None
    assert result.local_record == store.current.read_record("owner/target")
    assert result.local_record["resolved_commit_sha"] == result.commit_sha
    assert (
        store.current.artifact_path("owner/target", "model").read_bytes() == b"weights"
    )


def test_publish_materializes_local_working_weights(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.install_from_directory(
        _source(tmp_path / "local-source"),
        module_ref="local/source",
        resolved_commit_sha=None,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        installed_at="2026-08-15T12:34:56.789Z",
    )
    working_recipe = _recipe("Working")
    working_recipe["models"]["model"]["weights"] = {
        "type": "local",
        "ref": "local/source",
    }
    working = _source(
        tmp_path / "working-source",
        recipe=working_recipe,
        weights=None,
    )
    store.install_from_directory(
        working,
        module_ref="owner/working",
        resolved_commit_sha=None,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        external_artifacts={
            ("model", "weights"): store.current.artifact_path("local/source", "model")
        },
        installed_at="2026-08-15T12:34:56.789Z",
    )
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)

    result = _publish(store, source_ref="owner/working")

    assert result.commit_sha == hub.head
    assert json.loads(hub.files["recipe.json"])["models"]["model"]["weights"] == {
        "type": "vendored",
        "file": "model.safetensors",
    }
    assert hub.files["model.safetensors"] == b"weights"


def test_post_publish_local_sync_failure_is_a_separate_partial_result(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    hub = _FakeHub(tmp_path)
    _install_fake_hub(monkeypatch, hub)
    original = store.install_from_directory

    def fail_target(*args, **kwargs):
        if kwargs.get("module_ref") == "owner/target":
            raise OSError("local sync failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "install_from_directory", fail_target)

    result = _publish(store, sync_local=True)

    assert result.commit_sha == hub.head
    assert result.local_record is None
    assert isinstance(result.local_sync_error, OSError)
    assert hub.files["model.safetensors"] == b"weights"


@pytest.mark.parametrize("cache_symlinks", [False, True])
def test_verification_rejects_wrong_returned_bytes_before_local_sync(
    tmp_path, monkeypatch, cache_symlinks
):
    store = _store(tmp_path)
    hub = _FakeHub(tmp_path, cache_symlinks=cache_symlinks)
    _install_fake_hub(monkeypatch, hub)
    original_download = hub.download

    def corrupt_weight(**kwargs):
        path = Path(original_download(**kwargs))
        if kwargs["filename"] == "model.safetensors":
            path.write_bytes(b"corrupt")
        return str(path)

    monkeypatch.setattr(_publisher, "hf_hub_download", corrupt_weight)

    with pytest.raises(GatewayFailure) as caught:
        _publish(store, sync_local=True)
    assert caught.value.error["code"] == "REGISTRY_PUBLICATION_CONFLICT"
    assert caught.value.error["resource_id"] == "owner/target"
    assert store.current.list_module_refs() == ["owner/source"]
