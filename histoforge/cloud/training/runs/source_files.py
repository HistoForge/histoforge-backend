"""Immutable Artifact Volume source snapshots for managed training runs."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
    require_sha256,
)
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.modules.files import (
    copy_exact_file,
    require_exact_file,
    require_file_closure,
)
from histoforge.cloud.training.artifact_directories import (
    discover_training_artifact_ids,
)
from histoforge.cloud.training.runs.source_snapshot import (
    validate_training_source_ready_marker,
)

_READY_FILE = "ready.json"


@dataclass(frozen=True)
class RetainedTrainingSource:
    """One fully verified accepted source snapshot on Artifact Volume."""

    root: Path
    marker: dict[str, Any]


class TrainingSourceFiles:
    """Publish and verify one complete source closure before run visibility."""

    def __init__(
        self,
        root: str | Path,
        max_marker_bytes: int,
        *,
        reload_volume: Callable[[], None] = lambda: None,
        commit_volume: Callable[[], None] = lambda: None,
    ) -> None:
        self._root = Path(root)
        self._max_marker_bytes = max_marker_bytes
        self._reload_volume = reload_volume
        self._commit_volume = commit_volume

    @mounted_artifact_volume_operation
    def publish_candidate(
        self,
        marker: Any,
        source_paths: Mapping[str, Path],
    ) -> dict[str, Any]:
        """Publish one immutable source candidate keyed by its report digest."""

        ready = validate_training_source_ready_marker(marker, self._max_marker_bytes)
        sources = {str(path): Path(source) for path, source in source_paths.items()}
        identities = {
            identity["relative_path"]: identity for identity in ready["source_files"]
        }
        if set(sources) != set(identities):
            raise ValueError("source paths do not match the ready-marker inventory")

        run_id = ready["run_id"]
        self._reload_volume()
        target = self._candidate_root(run_id, ready["report_digest"])
        if target.exists():
            existing = self._read_root(target, run_id)
            if existing != ready:
                raise ArtifactBytesCorrupt("training source candidate differs")
            return existing

        stage = self._staging_root(run_id) / uuid.uuid4().hex
        try:
            stage.mkdir(parents=True)
            for relative_path, identity in identities.items():
                relative = PurePosixPath(relative_path)
                destination = stage.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                copy_exact_file(sources[relative_path], destination, identity)
            _write_fsynced(stage / _READY_FILE, canonical_bytes(ready))
            self._read_root(stage, run_id)

            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                stage.rename(target)
            except OSError:
                if not target.exists():
                    raise
                existing = self._read_root(target, run_id)
                if existing != ready:
                    raise ArtifactBytesCorrupt("training source candidate differs")
            self._commit_volume()
            return ready
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            _remove_empty_directory(self._staging_root(run_id))

    @mounted_artifact_volume_operation
    def read_for_promotion(
        self,
        run_id: str,
        report_digest: Any,
    ) -> dict[str, Any]:
        """Read a candidate, or the matching pending root left by an earlier cut."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        digest = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        candidate = self._candidate_root(canonical_id, digest)
        root = candidate if candidate.exists() else self._pending_root(canonical_id)
        marker = self._read_marker(root, canonical_id)
        if marker["report_digest"] != digest:
            raise ArtifactBytesCorrupt("training source report digest differs")
        return marker

    @mounted_artifact_volume_operation
    def promote_candidate(
        self,
        run_id: str,
        report_digest: Any,
    ) -> dict[str, Any]:
        """Move one accepted candidate to the sole run-owned pending root."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        digest = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        candidate = self._candidate_root(canonical_id, digest)
        target = self._pending_root(canonical_id)

        if not candidate.exists():
            marker = self._read_marker(target, canonical_id)
            if marker["report_digest"] != digest:
                raise ArtifactBytesCorrupt("training source report digest differs")
            return marker

        marker = self._read_marker(candidate, canonical_id)
        if marker["report_digest"] != digest:
            raise ArtifactBytesCorrupt("training source report digest differs")
        if target.exists():
            existing = self._read_marker(target, canonical_id)
            if _source_identity(existing) != _source_identity(marker):
                raise ArtifactBytesCorrupt("accepted training source differs")
            if existing != marker:
                self._replace_pending_marker(canonical_id, target, marker)
            return marker

        target.parent.mkdir(parents=True, exist_ok=True)
        candidate.rename(target)
        self._commit_volume()
        _remove_empty_directory(self._candidate_run_root(canonical_id))
        return marker

    @mounted_artifact_volume_operation
    def discard_candidate(self, run_id: str, report_digest: Any) -> None:
        """Delete one exact unaccepted candidate outside the serialized lane."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        digest = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        candidate = self._candidate_root(canonical_id, digest)
        if candidate.exists():
            if candidate.is_symlink() or not candidate.is_dir():
                raise ArtifactBytesCorrupt("training source candidate is invalid")
            shutil.rmtree(candidate)
            self._commit_volume()
        _remove_empty_directory(self._candidate_run_root(canonical_id))
        _remove_empty_directory(self._source_candidates_root())

    @mounted_artifact_volume_operation
    def require_ready(self, run_id: str) -> dict[str, Any]:
        """Reload and verify the marker, closure, and every accepted source file."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        return self._read_root(self._pending_root(canonical_id), canonical_id)

    @mounted_artifact_volume_operation
    def read_source(self, run_id: str) -> RetainedTrainingSource:
        """Reload and return one verified accepted source root and marker."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        root = self._pending_root(canonical_id)
        return RetainedTrainingSource(root, self._read_root(root, canonical_id))

    @mounted_artifact_volume_operation
    def read_ready_marker(self, run_id: str) -> dict[str, Any]:
        """Read only the committed marker during a short serialized state cut."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        return self._read_marker(self._pending_root(canonical_id), canonical_id)

    @mounted_artifact_volume_operation
    def delete_source_roots_after_fence(self, run_id: str) -> None:
        """Delete source roots after the state owner permits transient cleanup."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        changed = False
        for path in (
            self._pending_root(canonical_id),
            self._staging_root(canonical_id),
            self._candidate_run_root(canonical_id),
        ):
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise ArtifactBytesCorrupt("training source root is invalid")
                shutil.rmtree(path)
                changed = True
        if changed:
            self._commit_volume()
        _remove_empty_directory(self._source_staging_root())
        _remove_empty_directory(self._source_candidates_root())

    @mounted_artifact_volume_operation
    def transient_run_ids(self, max_items: Any) -> list[str]:
        """Discover a bounded set of run IDs with source-owned temporary roots."""

        self._reload_volume()
        return discover_training_artifact_ids(
            (
                self._pending_runs_root(),
                self._source_staging_root(),
                self._source_candidates_root(),
            ),
            max_items,
            identity_field="run_id",
        )

    def _read_root(self, root: Path, run_id: str) -> dict[str, Any]:
        try:
            marker = self._read_marker(root, run_id)
            expected = tuple(
                identity["relative_path"] for identity in marker["source_files"]
            )
            require_file_closure(root, (*expected, _READY_FILE))
            for identity in marker["source_files"]:
                relative = PurePosixPath(identity["relative_path"])
                require_exact_file(
                    root.joinpath(*relative.parts),
                    identity,
                    identity["relative_path"],
                )
            return marker
        except ArtifactBytesMissing:
            raise
        except ArtifactBytesCorrupt:
            raise
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(run_id) from error
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt("training source snapshot is invalid") from error

    def _read_marker(self, root: Path, run_id: str) -> dict[str, Any]:
        try:
            if root.is_symlink() or not root.is_dir():
                raise ArtifactBytesMissing(run_id)
            marker_path = root / _READY_FILE
            if marker_path.is_symlink() or not marker_path.is_file():
                raise ArtifactBytesMissing(run_id)
            if marker_path.stat().st_size > self._max_marker_bytes:
                raise ArtifactBytesCorrupt(
                    "training source ready marker exceeds its byte limit"
                )
            encoded = marker_path.read_bytes()
            marker = validate_training_source_ready_marker(
                parse_json(encoded), self._max_marker_bytes
            )
            if encoded != canonical_bytes(marker):
                raise ArtifactBytesCorrupt(
                    "training source ready marker is not canonical"
                )
            if marker["run_id"] != run_id:
                raise ArtifactBytesCorrupt(
                    "training source snapshot belongs to another run"
                )
            return marker
        except ArtifactBytesMissing:
            raise
        except ArtifactBytesCorrupt:
            raise
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(run_id) from error
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt(
                "training source ready marker is invalid"
            ) from error

    def _pending_root(self, run_id: str) -> Path:
        return self._pending_runs_root() / run_id

    def _pending_runs_root(self) -> Path:
        return self._root / "training" / "runs" / "pending"

    def _source_staging_root(self) -> Path:
        return self._root / "training" / "runs" / "source-staging"

    def _staging_root(self, run_id: str) -> Path:
        return self._source_staging_root() / run_id

    def _source_candidates_root(self) -> Path:
        return self._root / "training" / "runs" / "source-candidates"

    def _candidate_run_root(self, run_id: str) -> Path:
        return self._source_candidates_root() / run_id

    def _candidate_root(self, run_id: str, report_digest: str) -> Path:
        return self._candidate_run_root(run_id) / report_digest

    def _replace_pending_marker(
        self,
        run_id: str,
        pending_root: Path,
        marker: Mapping[str, Any],
    ) -> None:
        staging_root = self._staging_root(run_id)
        stage = staging_root / f"ready-{uuid.uuid4().hex}.json"
        try:
            staging_root.mkdir(parents=True, exist_ok=True)
            _write_fsynced(stage, canonical_bytes(marker))
            os.replace(stage, pending_root / _READY_FILE)
            self._commit_volume()
        finally:
            try:
                stage.unlink()
            except FileNotFoundError:
                pass
            _remove_empty_directory(staging_root)


def _write_fsynced(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        pass


def _source_identity(marker: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in marker.items() if key != "report_digest"}
