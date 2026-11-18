"""Immutable Artifact Volume candidates for run-derived private modules."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from histoforge.canonical_json import require_canonical_uuid, require_sha256
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.training.artifact_directories import (
    discover_training_artifact_ids,
)
from histoforge.cloud.training.runs.result_module import (
    TrainingResultCandidate,
    build_training_result_candidate,
    inspect_training_result_candidate,
    read_training_result_candidate,
)


class TrainingResultFiles:
    """Publish, verify, and remove one hidden complete candidate per run."""

    def __init__(
        self,
        root: str | Path,
        api_limits: Any,
        max_source_marker_bytes: int,
        *,
        reload_volume: Callable[[], None] = lambda: None,
        commit_volume: Callable[[], None] = lambda: None,
    ) -> None:
        self._root = Path(root)
        self._api_limits = api_limits
        self._max_source_marker_bytes = max_source_marker_bytes
        self._reload_volume = reload_volume
        self._commit_volume = commit_volume

    @mounted_artifact_volume_operation
    def publish_candidate(
        self,
        run_id: str,
        *,
        source_root: str | Path,
        source_marker: Any,
        selected_weights: str | Path,
        succeeded_report: Any,
        request: Any,
        project_model: Callable[..., dict[str, Any]],
    ) -> TrainingResultCandidate:
        """First-win one fully verified candidate and commit it before return."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        target = self._candidate_root(canonical_id)
        staging_parent = self._staging_root(canonical_id)
        stage = staging_parent / uuid.uuid4().hex
        try:
            candidate = build_training_result_candidate(
                stage,
                source_root=source_root,
                source_marker=source_marker,
                selected_weights=selected_weights,
                succeeded_report=succeeded_report,
                request=request,
                api_limits=self._api_limits,
                max_source_marker_bytes=self._max_source_marker_bytes,
                project_model=project_model,
            )
            if candidate.attestation["run_id"] != canonical_id:
                raise ValueError("training result candidate belongs to another run")
            if target.exists():
                return self._read_candidate(
                    target,
                    canonical_id,
                    candidate.module_content_digest,
                    candidate.report_digest,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                stage.rename(target)
            except OSError:
                if not target.exists():
                    raise
                existing = self._read_candidate(
                    target,
                    canonical_id,
                    candidate.module_content_digest,
                    candidate.report_digest,
                )
                return existing
            self._commit_volume()
            return self._read_candidate(
                target,
                canonical_id,
                candidate.module_content_digest,
                candidate.report_digest,
            )
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            _remove_empty_directory(staging_parent)
            _remove_empty_directory(self._all_staging_root())

    @mounted_artifact_volume_operation
    def require_candidate(
        self,
        run_id: str,
        module_content_digest: Any,
        report_digest: Any,
    ) -> TrainingResultCandidate:
        """Reload and verify the candidate frozen by run finalization."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        module_digest = require_sha256(module_content_digest, "module_content_digest")
        report = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        return self._read_candidate(
            self._candidate_root(canonical_id),
            canonical_id,
            module_digest,
            report,
        )

    @mounted_artifact_volume_operation
    def inspect_candidate(self, run_id: str) -> TrainingResultCandidate | None:
        """Return a complete self-verified candidate, or ``None`` if absent."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        root = self._candidate_root(canonical_id)
        if not root.exists():
            return None
        return inspect_training_result_candidate(
            root,
            expected_run_id=canonical_id,
            api_limits=self._api_limits,
        )

    def _read_candidate(
        self,
        root: Path,
        run_id: str,
        module_content_digest: str,
        report_digest: str,
    ) -> TrainingResultCandidate:
        return read_training_result_candidate(
            root,
            expected_run_id=run_id,
            expected_module_content_digest=module_content_digest,
            expected_report_digest=report_digest,
            api_limits=self._api_limits,
        )

    @mounted_artifact_volume_operation
    def delete_candidate_after_fence(self, run_id: str) -> None:
        """Remove a result candidate only after its run state permits cleanup."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        changed = False
        for path in (
            self._candidate_root(canonical_id),
            self._staging_root(canonical_id),
        ):
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("training result candidate root is invalid")
                shutil.rmtree(path)
                changed = True
        if changed:
            self._commit_volume()
        _remove_empty_directory(self._all_candidates_root())
        _remove_empty_directory(self._all_staging_root())

    @mounted_artifact_volume_operation
    def transient_run_ids(self, max_items: Any) -> list[str]:
        """Discover a bounded set of run IDs with hidden result roots."""

        self._reload_volume()
        return discover_training_artifact_ids(
            (self._all_candidates_root(), self._all_staging_root()),
            max_items,
            identity_field="run_id",
        )

    def _candidate_root(self, run_id: str) -> Path:
        return self._all_candidates_root() / run_id

    def _all_candidates_root(self) -> Path:
        return self._root / "training" / "runs" / "result-candidates"

    def _staging_root(self, run_id: str) -> Path:
        return self._all_staging_root() / run_id

    def _all_staging_root(self) -> Path:
        return self._root / "training" / "runs" / "result-staging"


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        pass
