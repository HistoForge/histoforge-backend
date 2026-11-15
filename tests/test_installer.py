"""Exact immutable Hugging Face snapshot download adapter."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from histoforge.cloud.modules.install import _installer
from huggingface_hub import RepoFile

COMMIT = "a" * 40


def _recipe(*, weights=None, steps=None):
    return {
        "schemaVersion": 3,
        "name": "Example",
        "models": {
            "model": {
                "arch": "dinov3-adapter-cellpose",
                "gpu": "A10G",
                "weights": weights or {"type": "vendored", "file": "model.safetensors"},
                "input": {"type": "field", "micronsPerPixel": 0.22},
                "output": {
                    "type": "instance_segmentation",
                    "classes": ["Cell"],
                },
                "params": {"num_classes": 0, "min_instance_pixels": 15},
            }
        },
        "steps": steps or [{"model": "model"}],
    }


@pytest.fixture
def fake_hf(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    calls = {"info": [], "download": [], "paths": []}

    class FakeApi:
        def __init__(self, token):
            calls["api_token"] = token

        def repo_info(self, **kwargs):
            calls["info"].append(kwargs)
            return SimpleNamespace(sha=COMMIT)

        def get_paths_info(self, **kwargs):
            calls["paths"].append(kwargs)
            name = kwargs["paths"][0]
            path = repository / name
            return (
                [RepoFile(path=name, size=path.stat().st_size, oid="b" * 40)]
                if path.exists()
                else []
            )

        def list_repo_files(self, **kwargs):
            return [
                path.relative_to(repository).as_posix()
                for path in repository.rglob("*")
                if path.is_file()
            ]

        def list_repo_tree(self, **kwargs):
            calls["paths"].append(kwargs)
            return [
                RepoFile(
                    path=path.relative_to(repository).as_posix(),
                    size=path.stat().st_size,
                    oid="b" * 40,
                )
                for path in repository.rglob("*")
                if path.is_file()
            ]

    def download(**kwargs):
        calls["download"].append(kwargs)
        path = repository / kwargs["filename"]
        if not path.exists():
            raise FileNotFoundError(kwargs["filename"])
        return str(path)

    monkeypatch.setattr(_installer, "HfApi", FakeApi)
    monkeypatch.setattr(_installer, "hf_hub_download", download)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return repository, calls


def _write_repository(repository: Path, recipe, *, weights=None, scripts=None):
    (repository / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    for name, content in (weights or {}).items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for name, content in (scripts or {}).items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_resolves_once_then_downloads_every_file_from_exact_commit(fake_hf):
    repository, calls = fake_hf
    _write_repository(
        repository,
        _recipe(),
        weights={"model.safetensors": b"weights"},
    )

    download = _installer.download_module_snapshot(
        "owner/module", "main", "token", max_module_bytes=1_000_000
    )
    try:
        assert download.module_ref == "owner/module"
        assert download.resolved_commit_sha == COMMIT
        assert (download.directory / "model.safetensors").read_bytes() == b"weights"
        assert calls["info"] == [
            {
                "repo_id": "owner/module",
                "revision": "main",
                "repo_type": "model",
                "token": "token",
            }
        ]
        assert [call["filename"] for call in calls["download"]] == [
            "recipe.json",
            "model.safetensors",
        ]
        assert {call["revision"] for call in calls["download"]} == {COMMIT}
    finally:
        _installer.remove_download(download)
    assert not download.directory.exists()


def test_scripts_are_downloaded_once_in_deterministic_path_order(fake_hf):
    repository, calls = fake_hf
    metric_source = (
        "/* @histoforge-metrics\n"
        '[{"name":"count","label":"Count","type":"scalar",'
        '"decimals":0,"aggregation":"sum"}]\n'
        "*/\nreturn [count: 0]\n"
    )
    _write_repository(
        repository,
        _recipe(steps=[{"model": "model"}, {"script": "metrics.groovy"}]),
        weights={"model.safetensors": b"weights"},
        scripts={"metrics.groovy": metric_source},
    )

    download = _installer.download_module_snapshot(
        "owner/module", None, "server-token", max_module_bytes=1_000_000
    )
    try:
        assert [call["filename"] for call in calls["download"]] == [
            "recipe.json",
            "metrics.groovy",
            "model.safetensors",
        ]
    finally:
        _installer.remove_download(download)


def test_external_artifact_recipe_fails_visibly_without_inventing_locator(fake_hf):
    repository, calls = fake_hf
    _write_repository(
        repository,
        _recipe(
            weights={
                "type": "referenced",
                "ref": "foundation/encoder",
                "revision": "b" * 40,
            }
        ),
    )

    with pytest.raises(ValueError, match="cannot yet resolve external artifact"):
        _installer.download_module_snapshot(
            "owner/module", None, "server-token", max_module_bytes=1_000_000
        )

    assert [call["filename"] for call in calls["download"]] == ["recipe.json"]


def test_invalid_resolved_commit_cleans_private_download(
    fake_hf, monkeypatch, tmp_path
):
    repository, _ = fake_hf
    _write_repository(
        repository,
        _recipe(),
        weights={"model.safetensors": b"weights"},
    )
    destination = tmp_path / "download"
    monkeypatch.setattr(_installer.tempfile, "mkdtemp", lambda **_: str(destination))

    class InvalidApi:
        def __init__(self, token):
            pass

        def repo_info(self, **kwargs):
            return SimpleNamespace(sha="main")

    monkeypatch.setattr(_installer, "HfApi", InvalidApi)

    with pytest.raises(ValueError, match="full lowercase"):
        _installer.download_module_snapshot(
            "owner/module", None, "server-token", max_module_bytes=1_000_000
        )
    assert not destination.exists()


def test_mid_download_failure_cleans_private_snapshot(fake_hf, monkeypatch, tmp_path):
    repository, _ = fake_hf
    _write_repository(repository, _recipe())
    destination = tmp_path / "download"
    monkeypatch.setattr(_installer.tempfile, "mkdtemp", lambda **_: str(destination))

    with pytest.raises(ValueError, match="missing recipe-declared"):
        _installer.download_module_snapshot(
            "owner/module", None, "server-token", max_module_bytes=1_000_000
        )
    assert not destination.exists()


def test_invalid_module_ref_fails_before_hugging_face(fake_hf):
    _, calls = fake_hf
    with pytest.raises(ValueError):
        _installer.download_module_snapshot(
            "owner/../module", None, "server-token", max_module_bytes=1_000_000
        )
    assert calls["info"] == []
