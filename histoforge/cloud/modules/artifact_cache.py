"""Immutable content-addressed cache for referenced module artifacts."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from histoforge.canonical_json import require_sha256
from histoforge.cloud.modules.files import contained_path, require_exact_file

ArtifactKey = tuple[str, str]
ArtifactInput = tuple[Path, Mapping[str, Any]]


class ReferencedArtifactCache:
    """Publish, verify, address, and sweep exact referenced artifact bytes."""

    def __init__(self, module_store_root: Path) -> None:
        self._store_root = Path(module_store_root)

    def path(self, digest: str) -> Path:
        """Return the private cache path for one exact SHA-256 identity."""

        canonical = require_sha256(digest, "referenced artifact digest")
        return self._contained(self._store_root / ".artifacts" / "sha256" / canonical)

    def missing_bytes(self, inputs: Mapping[ArtifactKey, ArtifactInput]) -> int:
        """Return bytes that would be newly added, deduplicated by digest."""

        missing: dict[str, int] = {}
        for _, identity in inputs.values():
            target = self.path(identity["sha256"])
            if target.exists():
                self.require_exact(target, identity)
            else:
                missing[identity["sha256"]] = identity["size_bytes"]
        return sum(missing.values())

    def publish_or_reuse(
        self,
        attempt_dir: Path,
        inputs: Mapping[ArtifactKey, ArtifactInput],
    ) -> dict[ArtifactKey, Path]:
        """Publish fully forced files by atomic rename, or verify existing bytes."""

        result: dict[ArtifactKey, Path] = {}
        for key, (source, identity) in inputs.items():
            target = self.path(identity["sha256"])
            if target.exists():
                self.require_exact(target, identity)
                result[key] = target
                continue
            staged = attempt_dir / "artifacts" / identity["sha256"]
            staged.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, staged.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
                writer.flush()
                os.fsync(writer.fileno())
            self.require_exact(staged, identity)
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.rename(target)
            result[key] = target
        return result

    def sweep(self, referenced_digests: Iterable[str]) -> set[str]:
        """Remove cache files not named by the supplied current-record roots."""

        roots = {
            require_sha256(digest, "referenced artifact digest")
            for digest in referenced_digests
        }
        cache_root = self._store_root / ".artifacts" / "sha256"
        if not cache_root.exists():
            return set()
        self._contained(cache_root)
        removed: set[str] = set()
        for entry in cache_root.iterdir():
            if entry.is_symlink() or not entry.is_file():
                raise ValueError("referenced-artifact cache contains an invalid entry")
            require_sha256(entry.name, "referenced artifact cache key")
            if entry.name not in roots:
                entry.unlink()
                removed.add(entry.name)
        return removed

    @staticmethod
    def require_exact(path: Path, identity: Mapping[str, Any]) -> None:
        """Require one cache file to equal its recorded length and digest."""

        require_exact_file(path, identity, "artifact cache")

    def _contained(self, candidate: Path) -> Path:
        return contained_path(self._store_root, candidate, "artifact cache path")
