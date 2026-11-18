"""Current-ref Module Volume storage and its only mutation algorithm.

``ModuleStore`` owns paths, capacity, private same-Volume staging, immutable
referenced-artifact cache entries, visible replacement, delete, and cache sweep.
It deliberately knows nothing about Modal jobs, Hugging Face, HTTP, or catalog
presentation. A caller supplies the Volume publication callback; the store owns
its exact ordering so callers cannot expose a partial visible directory.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    require_canonical_uuid,
    require_positive_safe_integer,
)
from histoforge.cloud.modules.current import CurrentModules
from histoforge.cloud.modules.files import (
    contained_path,
    copy_exact_file,
    read_regular_file,
    require_real_directory_outside,
)
from histoforge.cloud.modules.identity import (
    build_stored_module_record,
    module_content_file_identities,
    validate_module_directory,
    write_stored_module_record,
)
from histoforge.cloud.modules.schema import parse_module_recipe
from histoforge.module_ref import ModuleRef

Publish = Callable[[], None]


class TrainingResultConflict(ValueError):
    """The reserved ref contains verified content from a different result."""


class ModuleInstallRejected(ValueError):
    """A configured limit rejected installation before any current-ref change."""


class ModuleStore:
    """One filesystem owner for current modules and private store state."""

    def __init__(
        self,
        root: Path,
        *,
        max_module_bytes: int,
        max_installed_bytes: int,
        max_catalog_encoded_bytes: int,
        training_api_limits: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.max_module_bytes = require_positive_safe_integer(
            max_module_bytes, "max_module_bytes"
        )
        self.max_installed_bytes = require_positive_safe_integer(
            max_installed_bytes, "max_installed_bytes"
        )
        self.max_catalog_encoded_bytes = require_positive_safe_integer(
            max_catalog_encoded_bytes, "max_catalog_encoded_bytes"
        )
        self.training_api_limits = training_api_limits
        if self.max_module_bytes > self.max_installed_bytes:
            raise ValueError("max_module_bytes cannot exceed max_installed_bytes")
        self.current = CurrentModules(
            self.root,
            training_api_limits=self.training_api_limits,
        )

    def install_from_directory(
        self,
        source_dir: Path,
        *,
        module_ref: str,
        resolved_commit_sha: str | None,
        job_id: str,
        publish: Publish,
        external_artifacts: Mapping[tuple[str, str], Path] | None = None,
        installed_at: str | None = None,
        training_result_attestation: Any | None = None,
    ) -> dict[str, Any]:
        """Validate, stage, publish, and replace one exact current module."""

        parsed_ref = ModuleRef.parse(module_ref)
        canonical_ref = str(parsed_ref)
        if parsed_ref.owner == "workspace" and training_result_attestation is None:
            raise ValueError("workspace module refs are reserved for training results")
        canonical_job_id = require_canonical_uuid(job_id, "job_id")
        source = Path(source_dir)
        require_real_directory_outside(source, self.root, "module source")
        timestamp = installed_at or _now_millis()
        external = dict(external_artifacts or {})

        cleaned = self._remove_stale_attempts()
        swept = self.sweep_referenced_cache()
        if cleaned or swept:
            publish()

        record = build_stored_module_record(
            source,
            module_ref=canonical_ref,
            resolved_commit_sha=resolved_commit_sha,
            installed_at=timestamp,
            external_artifacts=external,
            training_api_limits=self.training_api_limits,
            training_result_attestation=training_result_attestation,
        )
        if training_result_attestation is not None:
            existing = self.matching_training_result(
                canonical_ref, record["training_result_attestation"]
            )
            if existing is not None:
                return existing

        from histoforge.cloud.modules.catalog import (
            CatalogDescriptorTooLarge,
            require_directory_descriptor_fits,
        )

        try:
            require_directory_descriptor_fits(
                source,
                record,
                self.max_catalog_encoded_bytes,
            )
        except CatalogDescriptorTooLarge as error:
            raise ModuleInstallRejected(str(error)) from error
        recipe = parse_module_recipe(read_regular_file(source, "recipe.json"))
        content_identities = module_content_file_identities(source, recipe, record)
        content_bytes = sum(identity["size_bytes"] for identity in content_identities)
        record_size = len(canonical_bytes(record))
        if content_bytes + record_size > self.max_module_bytes:
            raise ModuleInstallRejected(
                "installed module directory exceeds max_module_bytes"
            )

        attempt = self._attempt_dir(canonical_job_id)
        candidate = self._candidate_dir(canonical_job_id, canonical_ref)
        if attempt.exists():
            raise ValueError("module mutation attempt directory already exists")

        referenced_inputs = self._referenced_inputs(record, external)
        missing_cache_bytes = self.current.artifact_cache.missing_bytes(
            referenced_inputs
        )
        additional = content_bytes + record_size + missing_cache_bytes
        self._require_capacity(additional)

        try:
            self._copy_module_files(source, candidate, content_identities)
            cached_artifacts = self.current.artifact_cache.publish_or_reuse(
                self._attempt_dir(canonical_job_id), referenced_inputs
            )
            write_stored_module_record(candidate, record)
            validation_artifacts = self._validation_artifacts(
                record, external, cached_artifacts
            )
            validate_module_directory(
                candidate,
                expected_module_ref=canonical_ref,
                external_artifacts=validation_artifacts,
                training_api_limits=self.training_api_limits,
            )
            self._require_capacity(0)
            publish()

            if training_result_attestation is None:
                self._replace_visible(
                    canonical_job_id,
                    canonical_ref,
                    candidate,
                    publish,
                )
            else:
                self._create_visible(canonical_ref, candidate, record, publish)
            self._remove_attempt(canonical_job_id)
            self.sweep_referenced_cache()
            publish()
            return self.current.read_record(canonical_ref)
        except Exception:
            # Private remnants are safe and may already be visible through a
            # background Volume commit. The next serialized attempt removes them.
            raise

    def delete(self, module_ref: str, *, job_id: str, publish: Publish) -> None:
        """Atomically retire current visibility, then remove private bytes."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        canonical_job_id = require_canonical_uuid(job_id, "job_id")
        cleaned = self._remove_stale_attempts()
        swept = self.sweep_referenced_cache()
        if cleaned or swept:
            publish()

        visible = self.current.module_dir(canonical_ref)
        if not visible.exists():
            raise FileNotFoundError(canonical_ref)
        self.current.read_record(canonical_ref)

        retired = self._retired_dir(canonical_job_id, canonical_ref)
        retired.parent.mkdir(parents=True, exist_ok=True)
        visible.rename(retired)
        publish()
        shutil.rmtree(retired)
        self._remove_attempt(canonical_job_id)
        self.sweep_referenced_cache()
        publish()

    def sweep_referenced_cache(self) -> set[str]:
        """Remove cache entries not referenced by any current module record."""

        roots: set[str] = set()
        for ref in self.current.list_module_refs():
            record = self.current.read_record(ref)
            roots.update(
                artifact["sha256"]
                for artifact in record["resolved_artifacts"]
                if artifact["artifact_source_type"] == "referenced"
            )

        return self.current.artifact_cache.sweep(roots)

    def stored_bytes(self) -> int:
        """Return exact bytes under the Module Volume root, including private state."""

        if not self.root.exists():
            if self.root.is_symlink():
                raise ValueError("module store root is a broken link")
            return 0
        if not self.root.is_dir():
            raise ValueError("module store root is not a directory")
        total = 0
        for directory, names, files in os.walk(self.root, followlinks=False):
            directory_path = Path(directory)
            # Modal exposes the trusted configured Volume mount point itself as
            # a link. Links below that authority remain forbidden.
            if directory_path != self.root and directory_path.is_symlink():
                raise ValueError("module store contains a directory link")
            for name in names:
                path = directory_path / name
                if path.is_symlink():
                    raise ValueError("module store contains a directory link")
                if not path.is_dir():
                    raise ValueError("module store contains an invalid directory entry")
            for name in files:
                path = directory_path / name
                mode = path.lstat().st_mode
                if path.is_symlink() or not stat.S_ISREG(mode):
                    raise ValueError("module store contains a non-regular file")
                total += path.stat().st_size
        return total

    def _replace_visible(
        self,
        job_id: str,
        module_ref: str,
        candidate: Path,
        publish: Publish,
    ) -> None:
        visible = self.current.module_dir(module_ref)
        visible.parent.mkdir(parents=True, exist_ok=True)
        if visible.exists():
            retired = self._retired_dir(job_id, module_ref)
            retired.parent.mkdir(parents=True, exist_ok=True)
            visible.rename(retired)
            publish()
        candidate.rename(visible)
        publish()

    def _copy_module_files(
        self,
        source: Path,
        candidate: Path,
        identities: list[Mapping[str, Any]],
    ) -> None:
        candidate.mkdir(parents=True)
        for identity in identities:
            relative = Path(*identity["relative_path"].split("/"))
            source_path = source / relative
            target_path = candidate / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            copy_exact_file(source_path, target_path, identity)

    def matching_training_result(
        self, module_ref: str, expected_attestation: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Revalidate an exact installed result, or prove its ref is absent."""

        visible = self.current.module_dir(module_ref)
        if not visible.exists():
            return None
        actual = self.current.revalidate(module_ref)
        if actual.get("training_result_attestation") != expected_attestation:
            raise TrainingResultConflict(
                "different training result already exists at reserved ref"
            )
        return actual

    def _create_visible(
        self,
        module_ref: str,
        candidate: Path,
        expected: Mapping[str, Any],
        publish: Publish,
    ) -> None:
        visible = self.current.module_dir(module_ref)
        visible.parent.mkdir(parents=True, exist_ok=True)
        if visible.exists():
            self.matching_training_result(
                module_ref, expected["training_result_attestation"]
            )
            return
        candidate.rename(visible)
        publish()

    @staticmethod
    def _referenced_inputs(
        record: Mapping[str, Any],
        external: Mapping[tuple[str, str], Path],
    ) -> dict[tuple[str, str], tuple[Path, Mapping[str, Any]]]:
        result = {}
        for identity in record["resolved_artifacts"]:
            if identity["artifact_source_type"] != "referenced":
                continue
            key = (identity["model_id"], identity["role"])
            result[key] = (external[key], identity)
        return result

    @staticmethod
    def _validation_artifacts(
        record: Mapping[str, Any],
        external: Mapping[tuple[str, str], Path],
        cached: Mapping[tuple[str, str], Path],
    ) -> dict[tuple[str, str], Path]:
        result = {}
        for identity in record["resolved_artifacts"]:
            source_type = identity["artifact_source_type"]
            if source_type == "vendored":
                continue
            key = (identity["model_id"], identity["role"])
            result[key] = cached[key] if source_type == "referenced" else external[key]
        return result

    def _require_capacity(self, additional_bytes: int) -> None:
        used = self.stored_bytes()
        if used + additional_bytes > self.max_installed_bytes:
            raise ModuleInstallRejected("module store would exceed max_installed_bytes")

    def _remove_stale_attempts(self) -> bool:
        staging = self.root / ".staging"
        if not staging.exists():
            return False
        removed = False
        for attempt in staging.iterdir():
            if attempt.is_symlink() or not attempt.is_dir():
                raise ValueError("module staging root contains an invalid entry")
            shutil.rmtree(attempt)
            removed = True
        return removed

    def _remove_attempt(self, job_id: str) -> None:
        attempt = self._attempt_dir(job_id)
        if attempt.exists():
            shutil.rmtree(attempt)

    def _candidate_dir(self, job_id: str, module_ref: str) -> Path:
        ref = ModuleRef.parse(module_ref)
        return self._attempt_dir(job_id).joinpath("candidate", *ref.filesystem_parts())

    def _retired_dir(self, job_id: str, module_ref: str) -> Path:
        ref = ModuleRef.parse(module_ref)
        return self._attempt_dir(job_id).joinpath("retired", *ref.filesystem_parts())

    def _attempt_dir(self, job_id: str) -> Path:
        canonical = require_canonical_uuid(job_id, "job_id")
        candidate = self.root / ".staging" / canonical
        return contained_path(self.root, candidate, "module staging path")


def _now_millis() -> str:
    now = datetime.now(timezone.utc)
    milliseconds = now.microsecond // 1000
    return (
        f"{now.year:04d}-{now.month:02d}-{now.day:02d}"
        f"T{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
        f".{milliseconds:03d}Z"
    )
