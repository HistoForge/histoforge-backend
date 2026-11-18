from __future__ import annotations

from pathlib import Path

import pytest
from histoforge.canonical_json import sha256_bytes
from histoforge.cloud.architectures.dinov3_training import project_dinov3_training_model
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from tests.test_training_dataset import training_api_limits
from tests.test_training_result_module import _accepted_source, _success
from tests.test_training_source_preparation import RUN_ID

OTHER_RUN_ID = "55555555-5555-4555-8555-555555555555"


def test_publish_is_committed_and_exactly_replayable(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)
    publications: list[str] = []
    files = TrainingResultFiles(
        tmp_path / "artifacts",
        training_api_limits(),
        8 * 1024**2,
        commit_volume=lambda: publications.append("commit"),
    )

    first = files.publish_candidate(
        RUN_ID,
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        project_model=project_dinov3_training_model,
    )
    replay = files.publish_candidate(
        RUN_ID,
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        project_model=project_dinov3_training_model,
    )

    assert first.attestation == replay.attestation
    assert files.inspect_candidate(RUN_ID) == first
    assert files.inspect_candidate(OTHER_RUN_ID) is None
    assert publications == ["commit"]
    assert files.transient_run_ids(10) == [RUN_ID]
    assert (
        files.require_candidate(
            RUN_ID, first.module_content_digest, first.report_digest
        ).attestation
        == first.attestation
    )


def test_existing_different_candidate_conflicts(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)
    files = TrainingResultFiles(
        tmp_path / "artifacts", training_api_limits(), 8 * 1024**2
    )
    files.publish_candidate(
        RUN_ID,
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        project_model=project_dinov3_training_model,
    )
    weights.write_bytes(b"different trained weights")
    report["model_selection"]["selected_weight_digest"] = sha256_bytes(
        weights.read_bytes()
    )

    with pytest.raises(ValueError, match="digest differs|weights"):
        files.publish_candidate(
            RUN_ID,
            source_root=source_root,
            source_marker=marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            project_model=project_dinov3_training_model,
        )


def test_run_identity_mismatch_commits_nothing(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)
    publications: list[str] = []
    files = TrainingResultFiles(
        tmp_path / "artifacts",
        training_api_limits(),
        8 * 1024**2,
        commit_volume=lambda: publications.append("commit"),
    )

    with pytest.raises(ValueError, match="belongs to another run"):
        files.publish_candidate(
            OTHER_RUN_ID,
            source_root=source_root,
            source_marker=marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            project_model=project_dinov3_training_model,
        )

    assert publications == []
    assert not (
        tmp_path
        / "artifacts"
        / "training"
        / "runs"
        / "result-candidates"
        / OTHER_RUN_ID
    ).exists()


def test_cleanup_removes_only_the_named_result_root(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)
    publications: list[str] = []
    files = TrainingResultFiles(
        tmp_path / "artifacts",
        training_api_limits(),
        8 * 1024**2,
        commit_volume=lambda: publications.append("commit"),
    )
    candidate = files.publish_candidate(
        RUN_ID,
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        project_model=project_dinov3_training_model,
    )

    files.delete_candidate_after_fence(RUN_ID)

    assert not candidate.root.exists()
    assert source_root.exists()
    assert publications == ["commit", "commit"]
