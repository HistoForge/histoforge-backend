"""Immutable Artifact Volume reports owned by managed training runs."""

from __future__ import annotations

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
    require_positive_int32,
    require_sha256,
)
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.training.artifact_directories import (
    discover_training_artifact_ids,
)
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.report import (
    training_report_digest,
    validate_training_report,
)


class TrainingRunFiles:
    """Stage report candidates and manage immutable advertised revisions."""

    def __init__(
        self,
        root: str | Path,
        api_limits: Any,
        *,
        reload_volume: Callable[[], None] = lambda: None,
        commit_volume: Callable[[], None] = lambda: None,
    ) -> None:
        self._root = Path(root)
        self._api_limits = validate_training_api_limits(api_limits)
        self._reload_volume = reload_volume
        self._commit_volume = commit_volume

    @mounted_artifact_volume_operation
    def publish_report_candidate(self, value: Any) -> dict[str, Any]:
        """Publish canonical bytes under their digest outside the state lane."""

        report = validate_training_report(value, self._api_limits)
        run_id = report["run_id"]
        encoded = canonical_bytes(report)
        digest = training_report_digest(report, self._api_limits)

        self._reload_volume()
        target = self._candidate_path(run_id, digest)
        if target.exists():
            existing = self._read_report_path(target, run_id, digest)
            if existing != report:
                raise ArtifactBytesCorrupt("training report candidate digest collides")
            return {"report": existing, "report_digest": digest}

        staging_root = self._staging_root(run_id)
        stage = staging_root / f"{digest}-{uuid.uuid4().hex}.json"
        try:
            staging_root.mkdir(parents=True, exist_ok=True)
            _write_fsynced(stage, encoded)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, target)
            self._commit_volume()
        finally:
            _remove_if_present(stage)
            _remove_empty_directory(staging_root)
            _remove_empty_directory(self._all_staging_root())
        return {"report": report, "report_digest": digest}

    @mounted_artifact_volume_operation
    def read_report_for_promotion(
        self, run_id: str, revision: Any, report_digest: Any
    ) -> dict[str, Any]:
        """Read a candidate, or a matching revision left by an interrupted cut."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        number = require_positive_int32(revision, "report_revision")
        digest = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        candidate = self._candidate_path(canonical_id, digest)
        path = (
            candidate if candidate.exists() else self._report_path(canonical_id, number)
        )
        report = self._read_report_path(path, canonical_id, digest)
        return {"report": report, "report_digest": digest}

    @mounted_artifact_volume_operation
    def promote_report(
        self, run_id: str, revision: Any, report_digest: Any
    ) -> dict[str, Any]:
        """Move one caller-proven unadvertised candidate to its revision path."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        number = require_positive_int32(revision, "report_revision")
        digest = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        candidate = self._candidate_path(canonical_id, digest)
        target = self._report_path(canonical_id, number)

        if not candidate.exists():
            report = self._read_report_path(target, canonical_id, digest)
            return {"report": report, "report_digest": digest}

        report = self._read_report_path(candidate, canonical_id, digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(candidate, target)
        self._commit_volume()
        _remove_empty_directory(self._candidates_root(canonical_id))
        _remove_empty_directory(self._all_candidates_root())
        return {"report": report, "report_digest": digest}

    @mounted_artifact_volume_operation
    def read_report(self, run_id: str, revision: Any) -> dict[str, Any]:
        """Reload and verify one exact immutable report revision."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        number = require_positive_int32(revision, "report_revision")
        self._reload_volume()
        report = self._read_report_path(
            self._report_path(canonical_id, number), canonical_id
        )
        return {
            "report": report,
            "report_digest": training_report_digest(report, self._api_limits),
        }

    @mounted_artifact_volume_operation
    def discard_report_candidate(self, run_id: str, report_digest: Any) -> None:
        """Remove one unadvertised content-addressed candidate."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        digest = require_sha256(report_digest, "report_digest")
        self._reload_volume()
        path = self._candidate_path(canonical_id, digest)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ArtifactBytesCorrupt("training report candidate path is invalid")
            path.unlink()
            self._commit_volume()
        _remove_empty_directory(self._candidates_root(canonical_id))
        _remove_empty_directory(self._all_candidates_root())

    @mounted_artifact_volume_operation
    def delete_reports(self, run_id: str) -> None:
        """Delete only the immutable report and report-staging roots for one run."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        changed = False
        for path in (
            self._reports_root(canonical_id),
            self._candidates_root(canonical_id),
            self._staging_root(canonical_id),
        ):
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise ArtifactBytesCorrupt("training report root is invalid")
                shutil.rmtree(path)
                changed = True
        if changed:
            self._commit_volume()
        _remove_empty_directory(self._run_root(canonical_id))
        _remove_empty_directory(self._all_candidates_root())
        _remove_empty_directory(self._all_staging_root())

    @mounted_artifact_volume_operation
    def delete_transients(self, run_id: str) -> None:
        """Delete report candidates and staging while preserving report history."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        changed = False
        for path in (
            self._candidates_root(canonical_id),
            self._staging_root(canonical_id),
        ):
            if path.exists():
                if path.is_symlink() or not path.is_dir():
                    raise ArtifactBytesCorrupt("training report root is invalid")
                shutil.rmtree(path)
                changed = True
        if changed:
            self._commit_volume()
        _remove_empty_directory(self._all_candidates_root())
        _remove_empty_directory(self._all_staging_root())

    @mounted_artifact_volume_operation
    def prune_report_history(
        self,
        run_id: str,
        retained_revision: Any,
        retained_digest: Any,
    ) -> None:
        """Keep only the exact report advertised by a terminal run."""

        canonical_id = require_canonical_uuid(run_id, "run_id")
        revision = require_positive_int32(retained_revision, "report_revision")
        digest = require_sha256(retained_digest, "report_digest")
        self._reload_volume()
        retained = self._report_path(canonical_id, revision)
        self._read_report_path(retained, canonical_id, digest)

        reports_root = self._reports_root(canonical_id)
        paths = list(reports_root.iterdir())
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ArtifactBytesCorrupt("training report revision path is invalid")

        changed = False
        for path in paths:
            if path != retained:
                path.unlink()
                changed = True
        if changed:
            self._commit_volume()

    @mounted_artifact_volume_operation
    def transient_run_ids(self, max_items: Any) -> list[str]:
        """Discover a bounded set of run IDs with report-owned temporary roots."""

        self._reload_volume()
        return discover_training_artifact_ids(
            (self._all_candidates_root(), self._all_staging_root()),
            max_items,
            identity_field="run_id",
        )

    @mounted_artifact_volume_operation
    def report_run_ids(self, max_items: Any) -> list[str]:
        """Discover a bounded set of run IDs that own a report-revision root."""

        self._reload_volume()
        reserved = {
            "pending",
            "report-candidates",
            "report-staging",
            "result-candidates",
            "result-staging",
            "source-candidates",
            "source-staging",
        }
        return discover_training_artifact_ids(
            (self._runs_root(),),
            max_items,
            identity_field="run_id",
            ignored_names=reserved,
        )

    @mounted_artifact_volume_operation
    def report_history_run_ids(self, max_items: Any) -> list[str]:
        """Return bounded run IDs whose report root has obsolete revisions."""

        self._reload_volume()
        run_ids = discover_training_artifact_ids(
            (self._runs_root(),),
            max_items,
            identity_field="run_id",
            ignored_names={
                "pending",
                "report-candidates",
                "report-staging",
                "result-candidates",
                "result-staging",
                "source-candidates",
                "source-staging",
            },
        )
        result = []
        for run_id in run_ids:
            root = self._reports_root(run_id)
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                raise ArtifactBytesCorrupt("training report root is invalid")
            with os.scandir(root) as entries:
                for index, _entry in enumerate(entries):
                    if index == 1:
                        result.append(run_id)
                        break
        return result

    def _read_report_path(
        self, path: Path, run_id: str, expected_digest: str | None = None
    ) -> dict[str, Any]:
        try:
            if path.is_symlink() or not path.is_file():
                raise ArtifactBytesMissing(run_id)
            size = path.stat().st_size
            if size > self._api_limits["report"]["max_report_bytes"]:
                raise ArtifactBytesCorrupt("training report exceeds its byte limit")
            encoded = path.read_bytes()
            report = validate_training_report(parse_json(encoded), self._api_limits)
        except ArtifactBytesMissing:
            raise
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(run_id) from error
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt("training report is invalid") from error
        if report["run_id"] != run_id:
            raise ArtifactBytesCorrupt("training report belongs to another run")
        if encoded != canonical_bytes(report):
            raise ArtifactBytesCorrupt("training report is not canonical")
        if (
            expected_digest is not None
            and training_report_digest(report, self._api_limits) != expected_digest
        ):
            raise ArtifactBytesCorrupt("training report digest differs")
        return report

    def _run_root(self, run_id: str) -> Path:
        return self._runs_root() / run_id

    def _runs_root(self) -> Path:
        return self._root / "training" / "runs"

    def _reports_root(self, run_id: str) -> Path:
        return self._run_root(run_id) / "reports"

    def _report_path(self, run_id: str, revision: int) -> Path:
        return self._reports_root(run_id) / f"{revision}.json"

    def _candidates_root(self, run_id: str) -> Path:
        return self._all_candidates_root() / run_id

    def _all_candidates_root(self) -> Path:
        return self._root / "training" / "runs" / "report-candidates"

    def _candidate_path(self, run_id: str, digest: str) -> Path:
        return self._candidates_root(run_id) / f"{digest}.json"

    def _staging_root(self, run_id: str) -> Path:
        return self._all_staging_root() / run_id

    def _all_staging_root(self) -> Path:
        return self._root / "training" / "runs" / "report-staging"


def _write_fsynced(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except (FileNotFoundError, OSError):
        pass
