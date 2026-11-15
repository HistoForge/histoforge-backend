from __future__ import annotations

import json
from pathlib import Path

import pytest
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from tests.test_training_source_snapshot import source_ready_marker


def source_paths(tmp_path: Path) -> dict[str, Path]:
    weights = tmp_path / "model.safetensors"
    metadata = tmp_path / "selected-model.json"
    weights.write_bytes(b"starting weights")
    metadata.write_bytes(b"{}")
    return {
        "source/model.safetensors": weights,
        "source/selected-model.json": metadata,
    }


def test_snapshot_publishes_complete_source_and_reuses_exact_result(
    tmp_path: Path,
) -> None:
    commits = []
    files = TrainingSourceFiles(
        tmp_path / "artifacts",
        8 * 1024**2,
        commit_volume=lambda: commits.append("commit"),
    )
    marker = source_ready_marker()

    assert files.publish_candidate(marker, source_paths(tmp_path)) == marker
    assert files.publish_candidate(marker, source_paths(tmp_path)) == marker
    with pytest.raises(ArtifactBytesMissing):
        files.require_ready(marker["run_id"])
    assert files.read_for_promotion(marker["run_id"], marker["report_digest"]) == marker
    assert files.promote_candidate(marker["run_id"], marker["report_digest"]) == marker
    assert files.require_ready(marker["run_id"]) == marker
    root = tmp_path / "artifacts" / "training" / "runs" / "pending" / marker["run_id"]
    retained = files.read_source(marker["run_id"])
    assert retained.root == root
    assert retained.marker == marker
    assert files.promote_candidate(marker["run_id"], marker["report_digest"]) == marker
    assert commits == ["commit", "commit"]

    assert (root / "source" / "model.safetensors").read_bytes() == b"starting weights"
    assert json.loads((root / "ready.json").read_text()) == marker


def test_snapshot_rejects_tampered_or_different_content(tmp_path: Path) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    marker = source_ready_marker()
    files.publish_candidate(marker, source_paths(tmp_path))
    root = (
        tmp_path
        / "artifacts"
        / "training"
        / "runs"
        / "source-candidates"
        / marker["run_id"]
        / marker["report_digest"]
    )
    (root / "source" / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(ArtifactBytesCorrupt):
        files.publish_candidate(marker, source_paths(tmp_path))
    files.promote_candidate(marker["run_id"], marker["report_digest"])
    with pytest.raises(ArtifactBytesCorrupt, match="invalid"):
        files.require_ready(marker["run_id"])


def test_short_state_cut_reads_marker_without_rehashing_source(tmp_path: Path) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    marker = source_ready_marker()
    files.publish_candidate(marker, source_paths(tmp_path))
    files.promote_candidate(marker["run_id"], marker["report_digest"])
    root = tmp_path / "artifacts" / "training" / "runs" / "pending" / marker["run_id"]
    (root / "source" / "model.safetensors").unlink()

    assert files.read_ready_marker(marker["run_id"]) == marker
    with pytest.raises(ArtifactBytesCorrupt, match="snapshot is invalid"):
        files.require_ready(marker["run_id"])


def test_snapshot_rejects_missing_or_extra_files(tmp_path: Path) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    marker = source_ready_marker()
    paths = source_paths(tmp_path)
    del paths["source/selected-model.json"]

    with pytest.raises(ValueError, match="do not match"):
        files.publish_candidate(marker, paths)
    with pytest.raises(ArtifactBytesMissing):
        files.require_ready(marker["run_id"])


def test_delete_source_roots_after_fence_removes_only_source_roots(
    tmp_path: Path,
) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    marker = source_ready_marker()
    files.publish_candidate(marker, source_paths(tmp_path))
    files.promote_candidate(marker["run_id"], marker["report_digest"])
    report = (
        tmp_path
        / "artifacts"
        / "training"
        / "runs"
        / marker["run_id"]
        / "reports"
        / "1.json"
    )
    report.parent.mkdir(parents=True)
    report.write_bytes(b"report")

    files.delete_source_roots_after_fence(marker["run_id"])

    assert report.read_bytes() == b"report"
    with pytest.raises(ArtifactBytesMissing):
        files.require_ready(marker["run_id"])


def test_transient_run_ids_cover_candidates_and_promoted_sources(
    tmp_path: Path,
) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    marker = source_ready_marker()
    files.publish_candidate(marker, source_paths(tmp_path))

    assert files.transient_run_ids(2) == [marker["run_id"]]

    files.promote_candidate(marker["run_id"], marker["report_digest"])
    assert files.transient_run_ids(2) == [marker["run_id"]]


def test_losing_candidate_cannot_replace_the_accepted_source(tmp_path: Path) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    first = source_ready_marker()
    second = {**first, "request_digest": "f" * 64, "report_digest": "f" * 64}

    files.publish_candidate(first, source_paths(tmp_path))
    files.publish_candidate(second, source_paths(tmp_path))
    files.promote_candidate(first["run_id"], first["report_digest"])

    with pytest.raises(ArtifactBytesCorrupt, match="accepted training source differs"):
        files.promote_candidate(second["run_id"], second["report_digest"])
    assert files.require_ready(first["run_id"]) == first


def test_interrupted_promotion_rebinds_identical_source_to_the_winning_report(
    tmp_path: Path,
) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    first = source_ready_marker()
    second = {**first, "report_digest": "f" * 64}
    paths = source_paths(tmp_path)

    files.publish_candidate(first, paths)
    files.publish_candidate(second, paths)
    files.promote_candidate(first["run_id"], first["report_digest"])

    assert files.promote_candidate(second["run_id"], second["report_digest"]) == second
    assert files.require_ready(second["run_id"]) == second
    files.discard_candidate(second["run_id"], second["report_digest"])


def test_republished_accepted_candidate_is_discarded_outside_promotion(
    tmp_path: Path,
) -> None:
    commits = []
    files = TrainingSourceFiles(
        tmp_path / "artifacts",
        8 * 1024**2,
        commit_volume=lambda: commits.append("commit"),
    )
    marker = source_ready_marker()
    paths = source_paths(tmp_path)

    files.publish_candidate(marker, paths)
    files.promote_candidate(marker["run_id"], marker["report_digest"])
    files.publish_candidate(marker, paths)
    candidate = (
        tmp_path
        / "artifacts"
        / "training"
        / "runs"
        / "source-candidates"
        / marker["run_id"]
        / marker["report_digest"]
    )
    assert candidate.exists()

    assert files.promote_candidate(marker["run_id"], marker["report_digest"]) == marker
    assert candidate.exists()
    files.discard_candidate(marker["run_id"], marker["report_digest"])
    assert not candidate.exists()
    assert files.require_ready(marker["run_id"]) == marker
    assert commits == ["commit", "commit", "commit", "commit"]


def test_redundant_candidate_promotion_never_recursively_deletes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    marker = source_ready_marker()
    paths = source_paths(tmp_path)

    files.publish_candidate(marker, paths)
    files.promote_candidate(marker["run_id"], marker["report_digest"])
    files.publish_candidate(marker, paths)

    def reject_recursive_delete(_path: Path) -> None:
        raise AssertionError("promotion must not recursively delete source bytes")

    monkeypatch.setattr(
        "histoforge.cloud.training.runs.source_files.shutil.rmtree",
        reject_recursive_delete,
    )

    assert files.promote_candidate(marker["run_id"], marker["report_digest"]) == marker
