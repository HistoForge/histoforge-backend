"""Atomic Hugging Face publication from one exact current module snapshot."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)
from huggingface_hub.errors import HfHubHTTPError

from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.modules.files import (
    module_file_paths,
    stream_file_identity,
    validate_module_file_paths,
)
from histoforge.cloud.modules.identity import build_stored_module_record
from histoforge.cloud.modules.install._installer import (
    download_module_snapshot,
    remove_download,
    repository_file_sizes,
)
from histoforge.cloud.modules.publish._recipe_rewrite import to_published_recipe
from histoforge.cloud.modules.schema import (
    TRAINING_REPORT_NAME,
    model_artifact_declarations,
    module_content_paths,
    parse_module_recipe,
    validate_commit_sha,
)
from histoforge.cloud.modules.store import ModuleStore, Publish
from histoforge.module_ref import ModuleRef


@dataclass(frozen=True)
class PublicationResult:
    """Verified HF result plus the separately optional local-sync outcome."""

    commit_sha: str
    local_record: dict[str, Any] | None
    local_sync_error: Exception | None


def publish_module(
    store: ModuleStore,
    *,
    source_ref: str,
    target_repo: str,
    visibility: str,
    card_markdown: str,
    sync_local: bool,
    token: str,
    job_id: str,
    publish_volume: Publish,
) -> PublicationResult:
    """Snapshot, atomically publish, verify, and optionally install exact bytes."""

    canonical_source = str(ModuleRef.parse(source_ref))
    canonical_target = str(ModuleRef.parse(target_repo))
    if visibility not in {"PUBLIC", "PRIVATE"}:
        raise ValueError("visibility must be PUBLIC or PRIVATE")
    if not isinstance(card_markdown, str) or not card_markdown:
        raise ValueError("card_markdown must be nonempty")

    workspace = Path(tempfile.mkdtemp(prefix="histoforge-publish-"))
    source_snapshot = workspace / "source"
    published_snapshot = workspace / "published"
    try:
        source_record = store.current.snapshot(canonical_source, source_snapshot)
        published_recipe, referenced = _materialize_published_snapshot(
            store,
            canonical_source,
            source_record,
            source_snapshot,
            published_snapshot,
        )
        source_files = _file_identities(published_snapshot)

        api = HfApi(token=token)
        try:
            api.create_repo(
                repo_id=canonical_target,
                private=visibility == "PRIVATE",
                exist_ok=False,
                repo_type="model",
                token=token,
            )
        except HfHubHTTPError as error:
            if getattr(error.response, "status_code", None) != 409:
                raise

        head = api.repo_info(
            repo_id=canonical_target,
            repo_type="model",
            token=token,
        )
        parent = validate_commit_sha(head.sha, "Hugging Face parent commit")
        current_sizes = repository_file_sizes(api, canonical_target, parent, token)
        current_files = set(current_sizes)
        include_card = (
            "README.md" not in source_files
            and "README.md" not in current_files
            and "recipe.json" not in current_files
        )
        if include_card:
            (published_snapshot / "README.md").write_text(
                card_markdown, encoding="utf-8"
            )
            source_files = _file_identities(published_snapshot)
        replaceable = set(module_content_paths(published_recipe))
        uploads = {
            path: identity
            for path, identity in source_files.items()
            if path in replaceable or path not in current_files
        }
        stale = _obsolete_target_files(
            canonical_target,
            parent,
            token,
            current_files,
            set(source_files),
            published_recipe,
        )
        final_paths = (current_files | set(uploads)) - stale
        validate_module_file_paths(final_paths)
        final_size = sum(
            uploads[path]["size_bytes"] if path in uploads else current_sizes[path]
            for path in final_paths
        )
        if final_size > store.max_module_bytes:
            raise ValueError("published repository files exceed max_module_bytes")
        operations = _publication_operations(published_snapshot, uploads, stale)
        committed = api.create_commit(
            repo_id=canonical_target,
            repo_type="model",
            commit_message=f"histoforge publish {canonical_target}",
            parent_commit=parent,
            operations=operations,
            token=token,
        )
        commit = validate_commit_sha(committed.oid, "published commit")
        _verify_publication(
            api,
            token,
            canonical_target,
            commit,
            uploads,
            final_paths,
            workspace / "verification-cache",
        )

        if not sync_local:
            return PublicationResult(commit, None, None)
        try:
            # Target-preferred and target-only files are not in source staging.
            download = download_module_snapshot(
                canonical_target,
                commit,
                token,
                max_module_bytes=store.max_module_bytes,
                external_artifacts=referenced,
            )
            try:
                if download.resolved_commit_sha != commit:
                    raise ValueError("local sync resolved a different HF commit")
                local_record = store.install_from_directory(
                    download.directory,
                    module_ref=canonical_target,
                    resolved_commit_sha=commit,
                    job_id=job_id,
                    publish=publish_volume,
                    external_artifacts=referenced,
                )
            finally:
                remove_download(download)
            return PublicationResult(commit, local_record, None)
        except Exception as error:
            return PublicationResult(commit, None, error)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _materialize_published_snapshot(
    store: ModuleStore,
    source_ref: str,
    source_record: dict[str, Any],
    source: Path,
    destination: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], Path]]:
    local_recipe = parse_module_recipe((source / "recipe.json").read_bytes())
    published_recipe = to_published_recipe(local_recipe)
    destination.mkdir()

    recipe_bytes = (source / "recipe.json").read_bytes()
    if published_recipe != local_recipe:
        recipe_bytes = (
            json.dumps(published_recipe, indent=2, ensure_ascii=False).encode("utf-8")
            + b"\n"
        )
    (destination / "recipe.json").write_bytes(recipe_bytes)

    for path in set(module_file_paths(source)) - set(
        module_content_paths(local_recipe)
    ):
        _copy_relative(source, destination, path)
    for step in published_recipe["steps"]:
        if "script" in step:
            _copy_relative(source, destination, step["script"])
    if "trainingReportDigest" in published_recipe:
        _copy_relative(source, destination, TRAINING_REPORT_NAME)
    published_declarations = {
        (item["model_id"], item["role"]): item["source"]
        for item in model_artifact_declarations(published_recipe)
    }
    for declaration in model_artifact_declarations(local_recipe):
        model_id = declaration["model_id"]
        role = declaration["role"]
        artifact = declaration["source"]
        if artifact["type"] == "referenced":
            continue
        if artifact["type"] == "vendored":
            artifact_path = source.joinpath(*PurePosixPath(artifact["file"]).parts)
        else:
            artifact_path = store.current.artifact_path(
                source_ref,
                model_id,
                role,
            )
        published_path = published_declarations[(model_id, role)]["file"]
        _copy_file(artifact_path, destination, published_path)

    referenced: dict[tuple[str, str], Path] = {}
    for declaration in model_artifact_declarations(published_recipe):
        if declaration["source"]["type"] == "referenced":
            key = (declaration["model_id"], declaration["role"])
            referenced[key] = store.current.artifact_path(
                source_ref, declaration["model_id"], declaration["role"]
            )

    build_stored_module_record(
        destination,
        module_ref=source_record["module_ref"],
        resolved_commit_sha=None,
        installed_at=source_record["installed_at"],
        external_artifacts=referenced,
        training_api_limits=store.training_api_limits,
    )
    return published_recipe, referenced


def _file_identities(snapshot: Path) -> dict[str, dict[str, Any]]:
    identities = {}
    for relative_path in module_file_paths(snapshot):
        size_bytes, digest = stream_file_identity(
            snapshot.joinpath(*PurePosixPath(relative_path).parts),
            relative_path,
        )
        identities[relative_path] = {
            "size_bytes": size_bytes,
            "sha256": digest,
        }
    return identities


def _publication_operations(
    snapshot: Path,
    expected: dict[str, dict[str, Any]],
    stale: set[str],
) -> list[Any]:
    operations: list[Any] = [
        CommitOperationAdd(
            path_in_repo=relative_path,
            path_or_fileobj=str(snapshot.joinpath(*PurePosixPath(relative_path).parts)),
        )
        for relative_path in sorted(expected)
    ]
    operations.extend(
        CommitOperationDelete(path_in_repo=path) for path in sorted(stale)
    )
    return operations


def _verify_publication(
    api: HfApi,
    token: str,
    repo_id: str,
    revision: str,
    expected: dict[str, dict[str, Any]],
    expected_paths: set[str],
    cache_dir: Path,
) -> None:
    files = set(
        api.list_repo_files(
            repo_id=repo_id,
            revision=revision,
            repo_type="model",
            token=token,
        )
    )
    if files != expected_paths:
        raise GatewayFailure(
            GatewayErrorCode.REGISTRY_PUBLICATION_CONFLICT,
            "published repository module-file closure does not match snapshot",
            resource_id=repo_id,
        )
    for relative_path, identity in expected.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=relative_path,
                revision=revision,
                repo_type="model",
                token=token,
                cache_dir=cache_dir,
            )
        )
        # HF cache snapshots may link to shared blobs; module trees must not.
        size_bytes, digest = stream_file_identity(
            downloaded.resolve(strict=True), relative_path
        )
        if size_bytes != identity["size_bytes"]:
            raise GatewayFailure(
                GatewayErrorCode.REGISTRY_PUBLICATION_CONFLICT,
                f"published file size mismatch: {relative_path}",
                resource_id=repo_id,
            )
        if digest != identity["sha256"]:
            raise GatewayFailure(
                GatewayErrorCode.REGISTRY_PUBLICATION_CONFLICT,
                f"published file digest mismatch: {relative_path}",
                resource_id=repo_id,
            )


def _obsolete_target_files(
    repo_id: str,
    revision: str,
    token: str,
    current_paths: set[str],
    result_paths: set[str],
    recipe: dict[str, Any],
) -> set[str]:
    """Remove only declared weights absent from the result and superseded reports."""

    stale = set()
    if "recipe.json" in current_paths:
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename="recipe.json",
                revision=revision,
                repo_type="model",
                token=token,
            )
        )
        previous = parse_module_recipe(path.read_bytes())
        stale = {
            item["source"]["file"]
            for item in model_artifact_declarations(previous)
            if item["source"]["type"] == "vendored"
        } - result_paths
    if TRAINING_REPORT_NAME in current_paths and "trainingReportDigest" not in recipe:
        stale.add(TRAINING_REPORT_NAME)
    return stale & current_paths


def _copy_relative(source: Path, destination: Path, relative_path: str) -> None:
    _copy_file(
        source.joinpath(*PurePosixPath(relative_path).parts), destination, relative_path
    )


def _copy_file(source: Path, destination: Path, relative_path: str) -> None:
    relative = PurePosixPath(relative_path)
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
