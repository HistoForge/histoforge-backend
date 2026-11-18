"""Immutable Artifact-Volume publication and bounded byte reads.

The store owns physical paths, reload/open/close/commit ordering, and exact
byte identity.  Source-artifact and model-run stores retain lifecycle and
scientific meaning; this owner never decides when their references expire.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
)
from histoforge.cloud.artifacts.refs import validate_artifact_ref
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation

DEFAULT_READ_BLOCK_BYTES = 1024 * 1024


class ArtifactBytesCorrupt(OSError):
    """Committed artifact bytes or private metadata fail their immutable identity."""


class ArtifactBytesMissing(FileNotFoundError):
    """A committed artifact path is absent from the mounted Volume view."""


class ArtifactFileStore:
    """Physical owner of generic immutable artifacts."""

    def __init__(
        self,
        root: str | Path,
        *,
        reload_volume: Callable[[], None] = lambda: None,
        commit_volume: Callable[[], None] = lambda: None,
    ) -> None:
        self._root = Path(root)
        self._reload_volume = reload_volume
        self._commit_volume = commit_volume

    @mounted_artifact_volume_operation
    def publish_artifact_bytes(
        self,
        value: bytes,
        *,
        media_type: str,
        content_encoding: str | None = None,
    ) -> dict[str, Any]:
        """Publish one fresh generic representation and return its opaque ref."""

        artifact_id = str(uuid.uuid4())
        reference: dict[str, Any] = {
            "artifact_id": artifact_id,
            "media_type": media_type,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        if content_encoding is not None:
            reference["content_encoding"] = content_encoding
        reference = validate_artifact_ref(reference)

        self._reload_volume()
        root = self._artifact_root(artifact_id)
        changed = publish_bytes(root / "data.bin", value, reference["sha256"])
        changed |= publish_bytes(
            root / "descriptor.json",
            canonical_bytes(reference),
            hashlib.sha256(canonical_bytes(reference)).hexdigest(),
        )
        if changed:
            self._commit_volume()
        return reference

    @mounted_artifact_volume_operation
    def load_artifact_ref(self, artifact_id: str) -> dict[str, Any]:
        """Reload and validate one generic descriptor plus its exact data digest."""

        canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
        self._reload_volume()
        root = self._artifact_root(canonical_id)
        descriptor_path = root / "descriptor.json"
        data_path = root / "data.bin"
        if not descriptor_path.is_file() or not data_path.is_file():
            raise ArtifactBytesMissing(canonical_id)
        try:
            reference = validate_artifact_ref(parse_json(descriptor_path.read_bytes()))
        except (UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt(
                "stored artifact descriptor is invalid"
            ) from error
        if reference["artifact_id"] != canonical_id:
            raise ArtifactBytesCorrupt(
                "stored artifact descriptor names a different artifact"
            )
        if data_path.stat().st_size != reference["size_bytes"]:
            raise ArtifactBytesCorrupt(
                "stored artifact length does not match its descriptor"
            )
        if sha256_file(data_path) != reference["sha256"]:
            raise ArtifactBytesCorrupt(
                "stored artifact digest does not match its descriptor"
            )
        return reference

    @mounted_artifact_volume_operation
    def delete_artifact(self, artifact_id: str) -> None:
        """Remove one generic immutable artifact after its owner releases it."""

        canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
        self._reload_volume()
        root = self._artifact_root(canonical_id)
        if root.exists():
            shutil.rmtree(root)
            self._commit_volume()

    @mounted_artifact_volume_operation
    def artifact_ids(self) -> set[str]:
        """Return canonical generic-artifact directories visible after reload."""

        self._reload_volume()
        artifact_root = self._root / "files"
        if not artifact_root.exists():
            return set()
        result = set()
        for path in artifact_root.iterdir():
            if not path.is_dir():
                continue
            try:
                result.add(require_canonical_uuid(path.name, "artifact_id"))
            except ValueError:
                continue
        return result

    @mounted_artifact_volume_operation
    def read_artifact_block(
        self, artifact_id: str, offset: int, size_bytes: int
    ) -> bytes:
        """Open, copy, and close one bounded immutable response block."""

        canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
        if offset < 0 or size_bytes <= 0:
            raise ValueError("artifact block range must be positive and contained")
        path = self._artifact_root(canonical_id) / "data.bin"
        try:
            with path.open("rb") as stream:
                stream.seek(offset)
                value = stream.read(size_bytes)
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(canonical_id) from error
        if len(value) != size_bytes:
            raise ArtifactBytesMissing("artifact ended before its declared length")
        return value

    def _artifact_root(self, artifact_id: str) -> Path:
        return self._root / "files" / require_canonical_uuid(artifact_id, "artifact_id")


def force_file(stream: Any) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(DEFAULT_READ_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_bytes(path: Path, value: bytes, digest: str) -> bool:
    if hashlib.sha256(value).hexdigest() != digest:
        raise ValueError("caller-supplied immutable byte digest is wrong")
    if path.exists():
        if path.stat().st_size != len(value) or sha256_file(path) != digest:
            raise ArtifactBytesCorrupt(
                f"immutable artifact path contains different bytes: {path.name}"
            )
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.{uuid.uuid4().hex}.stage")
    try:
        with stage.open("xb") as stream:
            stream.write(value)
            force_file(stream)
        os.replace(stage, path)
        return True
    finally:
        if stage.exists():
            stage.unlink()
