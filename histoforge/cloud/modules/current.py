"""Read-only view of complete current modules on the Module Volume.

``CurrentModules`` owns path resolution, installed-record reads, exact artifact
lookup, and detached publication snapshots. It never mutates the mounted store;
all Volume changes remain serialized through :class:`ModuleStore`.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from histoforge.canonical_json import canonical_digest, sha256_bytes
from histoforge.cloud.modules.artifact_cache import ReferencedArtifactCache
from histoforge.cloud.modules.files import (
    contained_path,
    copy_exact_file,
    read_regular_file,
    require_file_closure,
    require_real_directory_outside,
)
from histoforge.cloud.modules.identity import (
    STORED_MODULE_RECORD_PATH,
    module_content_file_identities,
    read_stored_module_record,
    validate_module_directory,
)
from histoforge.cloud.modules.schema import parse_module_recipe
from histoforge.cloud.modules.training_evidence import validate_module_training_report
from histoforge.module_ref import ModuleRef


class CurrentModules:
    """Resolve and verify only fully published current module directories."""

    def __init__(self, root: Path, *, training_api_limits: Any | None = None) -> None:
        self.root = Path(root)
        self.training_api_limits = training_api_limits
        self.artifact_cache = ReferencedArtifactCache(self.root)

    def module_dir(self, module_ref: str) -> Path:
        """Return one contained current-ref path after canonical validation."""

        ref = ModuleRef.parse(module_ref)
        candidate = self.root.joinpath(*ref.filesystem_parts())
        return contained_path(self.root, candidate, "module path")

    def list_module_refs(self) -> list[str]:
        """Return current refs that contain a canonical stored record."""

        if not self.root.exists():
            return []
        refs: list[str] = []
        for owner in self.root.iterdir():
            if owner.name.startswith(".") or not owner.is_dir() or owner.is_symlink():
                continue
            for module in owner.iterdir():
                if not module.is_dir() or module.is_symlink():
                    continue
                try:
                    ref = str(ModuleRef(owner.name, module.name))
                    contained_path(self.root, module, "module path")
                    if (module / STORED_MODULE_RECORD_PATH).is_file():
                        refs.append(ref)
                except ValueError:
                    continue
        return sorted(refs)

    def read_record(self, module_ref: str) -> dict[str, Any]:
        """Read a current module record without rehashing large weight files."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        directory = self.module_dir(canonical_ref)
        if not directory.exists():
            raise FileNotFoundError(canonical_ref)
        record = read_stored_module_record(directory)
        if record["module_ref"] != canonical_ref:
            raise ValueError("stored module record does not match current path")
        return record

    def load_recipe(
        self,
        module_ref: str,
        *,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Read the recipe against one supplied or current installed record."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        current = record or self.read_record(canonical_ref)
        if current["module_ref"] != canonical_ref:
            raise ValueError("stored module record does not match current path")
        recipe_bytes = self.read_executable(
            canonical_ref,
            "recipe.json",
            record=current,
        )
        recipe = parse_module_recipe(recipe_bytes)
        if canonical_digest(recipe) != current["recipe_digest"]:
            raise ValueError("current recipe does not match recorded recipe digest")
        return recipe

    def load_training_report(
        self,
        module_ref: str,
        *,
        record: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Read and validate the optional immutable report owned by one module."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        current = record or self.read_record(canonical_ref)
        if current["module_ref"] != canonical_ref:
            raise ValueError("stored module record does not match current path")
        if self.training_api_limits is None:
            raise ValueError("training_api_limits are required for a training report")
        recipe = self.load_recipe(canonical_ref, record=current)
        report = validate_module_training_report(
            self.module_dir(canonical_ref),
            recipe,
            current["resolved_artifacts"],
            self.training_api_limits,
        )
        attestation = current.get("training_result_attestation")
        if attestation is not None and report is not None:
            if attestation["run_id"] != report["run_id"]:
                raise ValueError(
                    "training result attestation run_id differs from report"
                )
            if attestation["report_digest"] != recipe["trainingReportDigest"]:
                raise ValueError(
                    "training result attestation report_digest differs from recipe"
                )
        return report

    def read_executable(
        self,
        module_ref: str,
        relative_path: str,
        *,
        record: dict[str, Any] | None = None,
    ) -> bytes:
        """Read one executable only when its bytes match the current record."""

        current = record or self.read_record(module_ref)
        matches = [
            item
            for item in current["executable_files"]
            if item["relative_path"] == relative_path
        ]
        if len(matches) != 1:
            raise KeyError(f"module has no unique executable {relative_path!r}")
        content = read_regular_file(self.module_dir(module_ref), relative_path)
        identity = matches[0]
        if len(content) != identity["size_bytes"]:
            raise ValueError(
                "current executable size does not match installed identity"
            )
        if sha256_bytes(content) != identity["sha256"]:
            raise ValueError(
                "current executable digest does not match installed identity"
            )
        return content

    def script_sources(self, module_ref: str) -> dict[str, str]:
        """Return every exact recipe-declared Groovy source by relative path."""

        recipe = self.load_recipe(module_ref)
        names = sorted({step["script"] for step in recipe["steps"] if "script" in step})
        result = {}
        for name in names:
            try:
                result[name] = self.read_executable(module_ref, name).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"module script is not valid UTF-8: {name}") from exc
        return result

    def artifact_path(
        self,
        module_ref: str,
        model_id: str,
        role: str = "weights",
    ) -> Path:
        """Resolve one installed artifact from its recorded schema role."""

        return self._artifact_path(module_ref, model_id, role, frozenset())

    def _artifact_path(
        self,
        module_ref: str,
        model_id: str,
        role: str,
        visited: frozenset[tuple[str, str, str]],
    ) -> Path:
        """Resolve local references recursively while rejecting cycles."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        key = (canonical_ref, model_id, role)
        if key in visited:
            raise ValueError("local artifact references contain a cycle")
        record = self.read_record(canonical_ref)
        matches = [
            artifact
            for artifact in record["resolved_artifacts"]
            if artifact["model_id"] == model_id and artifact["role"] == role
        ]
        if len(matches) != 1:
            raise KeyError(f"module has no unique artifact {model_id}.{role}")
        artifact = matches[0]
        if artifact["artifact_source_type"] == "vendored":
            recipe = self.load_recipe(module_ref)
            source = recipe["models"][model_id]
            if role == "backbone.weights":
                source = source["backbone"]
            path = self.module_dir(canonical_ref) / source["weights"]["file"]
            self.artifact_cache.require_exact(path, artifact)
            return path
        if artifact["artifact_source_type"] == "referenced":
            path = self.artifact_cache.path(artifact["sha256"])
            self.artifact_cache.require_exact(path, artifact)
            return path
        path = self._artifact_path(
            artifact["artifact_source_ref"],
            model_id,
            role,
            visited | {key},
        )
        self.artifact_cache.require_exact(path, artifact)
        return path

    def external_artifact_paths(
        self,
        module_ref: str,
        record: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], Path]:
        """Resolve non-vendored artifact bytes needed for full revalidation."""

        current = record or self.read_record(module_ref)
        result: dict[tuple[str, str], Path] = {}
        for artifact in current["resolved_artifacts"]:
            source_type = artifact["artifact_source_type"]
            if source_type == "vendored":
                continue
            key = (artifact["model_id"], artifact["role"])
            if source_type == "referenced":
                result[key] = self.artifact_cache.path(artifact["sha256"])
            else:
                result[key] = self.artifact_path(module_ref, *key)
        return result

    def snapshot(self, module_ref: str, destination: Path) -> dict[str, Any]:
        """Copy one fully revalidated current module into a private snapshot."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        source = self.module_dir(canonical_ref)
        record = self.revalidate(canonical_ref)

        target = Path(destination)
        require_real_directory_outside(
            target.parent,
            self.root,
            "module snapshot parent",
        )
        if target.exists():
            raise ValueError("module snapshot destination already exists")
        target.mkdir(parents=True)
        recipe = self.load_recipe(canonical_ref, record=record)
        identities = module_content_file_identities(source, recipe, record)
        for identity in identities:
            relative = PurePosixPath(identity["relative_path"])
            source_path = source.joinpath(*relative.parts)
            target_path = target.joinpath(*relative.parts)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            copy_exact_file(source_path, target_path, identity)
        require_file_closure(
            target,
            tuple(identity["relative_path"] for identity in identities),
        )
        return record

    def revalidate(self, module_ref: str) -> dict[str, Any]:
        """Recompute one current module and every external artifact identity."""

        canonical_ref = str(ModuleRef.parse(module_ref))
        record = self.read_record(canonical_ref)
        external = self.external_artifact_paths(canonical_ref, record)
        return validate_module_directory(
            self.module_dir(canonical_ref),
            expected_module_ref=canonical_ref,
            external_artifacts=external,
            training_api_limits=self.training_api_limits,
        )
