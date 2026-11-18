"""Exact Hugging Face module snapshot adapter for ``ModuleStore``.

This adapter resolves a branch/tag/requested SHA once, downloads every
repository file from that immutable commit, and returns a private
temporary directory. It never mutates the Module Volume and never decides
replacement or capacity policy.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, RepoFile, hf_hub_download

from histoforge.canonical_json import (
    require_nonnegative_safe_integer,
    require_positive_safe_integer,
)
from histoforge.cloud.modules.files import (
    is_private_module_path,
    validate_module_file_paths,
)
from histoforge.cloud.modules.readme import MAX_README_BYTES, read_module_readme
from histoforge.cloud.modules.schema import (
    model_artifact_declarations,
    module_content_paths,
    parse_module_recipe,
    validate_commit_sha,
)
from histoforge.module_ref import ModuleRef


@dataclass(frozen=True)
class DownloadedModule:
    """One exact immutable repository snapshot in a private local directory."""

    module_ref: str
    resolved_commit_sha: str
    directory: Path


def download_module_snapshot(
    module_ref: str,
    revision: str | None,
    token: str,
    *,
    max_module_bytes: int,
    external_artifacts: Mapping[tuple[str, str], Path] | None = None,
) -> DownloadedModule:
    """Resolve once, bound the whole repository, and download every regular file.

    External artifact bytes may be supplied by publication's verified source;
    ordinary installation still requires vendored artifacts.
    """

    canonical_ref = str(ModuleRef.parse(module_ref))
    api = HfApi(token=token)
    info = api.repo_info(
        repo_id=canonical_ref,
        revision=revision,
        repo_type="model",
        token=token,
    )
    commit = validate_commit_sha(info.sha, "resolved Hugging Face commit")
    sizes = repository_file_sizes(api, canonical_ref, commit, token)
    if "recipe.json" not in sizes:
        raise ValueError("repository is missing recipe.json")
    maximum = require_positive_safe_integer(max_module_bytes, "max_module_bytes")
    if sum(sizes.values()) > maximum:
        raise ValueError("repository files exceed max_module_bytes")
    if sizes.get("README.md", 0) > MAX_README_BYTES:
        raise ValueError("README.md exceeds the 256 KiB module-card limit")
    destination = Path(tempfile.mkdtemp(prefix="histoforge-module-"))
    try:
        _download_file(
            canonical_ref,
            commit,
            token,
            "recipe.json",
            destination,
            sizes["recipe.json"],
        )
        recipe = parse_module_recipe((destination / "recipe.json").read_bytes())
        unsupported = [
            declaration
            for declaration in model_artifact_declarations(recipe)
            if declaration["source"]["type"] != "vendored"
            and (
                declaration["source"]["type"] != "referenced"
                or (declaration["model_id"], declaration["role"])
                not in (external_artifacts or {})
            )
        ]
        if unsupported:
            roles = [
                f"{item['model_id']}.{item['role']}:{item['source']['type']}"
                for item in unsupported
            ]
            raise ValueError(
                "this install slice cannot yet resolve external artifact bytes: "
                f"{roles}"
            )
        missing = set(module_content_paths(recipe)) - set(sizes)
        if missing:
            raise ValueError(
                f"repository is missing recipe-declared files: {sorted(missing)}"
            )
        for relative_path in sorted(sizes):
            if relative_path != "recipe.json":
                _download_file(
                    canonical_ref,
                    commit,
                    token,
                    relative_path,
                    destination,
                    sizes[relative_path],
                )
        read_module_readme(destination)
        return DownloadedModule(canonical_ref, commit, destination)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def repository_file_sizes(
    api: HfApi, module_ref: str, commit: str, token: str
) -> dict[str, int]:
    """Inventory exact-revision regular files before joining or downloading paths."""

    files = [
        entry
        for entry in api.list_repo_tree(
            repo_id=module_ref,
            revision=commit,
            recursive=True,
            repo_type="model",
            token=token,
        )
        if isinstance(entry, RepoFile)
    ]
    paths = [entry.path for entry in files]
    validate_module_file_paths(paths)
    if any(is_private_module_path(path) for path in paths):
        raise ValueError("repository contains reserved backend-private paths")
    return {
        path: require_nonnegative_safe_integer(entry.size, f"{path} size")
        for path, entry in zip(paths, files, strict=True)
    }


def remove_download(download: DownloadedModule) -> None:
    """Remove the adapter-owned private snapshot after store consumption."""

    shutil.rmtree(download.directory, ignore_errors=True)


def _download_file(
    module_ref: str,
    commit: str,
    token: str,
    relative_path: str,
    destination: Path,
    expected_size: int,
) -> None:
    """Copy exact cache bytes into the adapter-owned strict file tree."""

    source = Path(
        hf_hub_download(
            repo_id=module_ref,
            filename=relative_path,
            revision=commit,
            token=token,
            repo_type="model",
        )
    )
    if source.stat().st_size != expected_size:
        raise ValueError(
            f"downloaded size differs from the resolved revision: {relative_path}"
        )
    target = destination.joinpath(*PurePosixPath(relative_path).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
