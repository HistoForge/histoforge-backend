from __future__ import annotations

import json
from pathlib import Path

import pytest
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.training.runs.files import TrainingRunFiles
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report


def test_report_candidate_is_content_addressed_and_promoted_once(
    tmp_path: Path,
) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    report = queued_report()

    candidate = files.publish_report_candidate(report)
    assert files.publish_report_candidate(report) == candidate
    assert (
        files.read_report_for_promotion(report["run_id"], 1, candidate["report_digest"])
        == candidate
    )
    assert (
        files.promote_report(report["run_id"], 1, candidate["report_digest"])
        == candidate
    )
    assert files.read_report(report["run_id"], 1) == candidate
    assert (
        files.promote_report(report["run_id"], 1, candidate["report_digest"])
        == candidate
    )


def test_invalid_report_profile_is_rejected_before_candidate_publication(
    tmp_path: Path,
) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    report = queued_report()
    report["trainer_configuration"]["optimizer"]["parameters"]["api_key"] = "secret"

    with pytest.raises(ValueError, match="unknown fields"):
        files.publish_report_candidate(report)

    assert not (tmp_path / "training").exists()


def test_competing_report_candidates_do_not_overwrite_each_other(
    tmp_path: Path,
) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    first_report = queued_report()
    second_report = queued_report()
    second_report["messages"] = [{"severity": "info", "message": "Different"}]

    first = files.publish_report_candidate(first_report)
    second = files.publish_report_candidate(second_report)

    assert first["report_digest"] != second["report_digest"]
    assert (
        files.read_report_for_promotion(
            first_report["run_id"], 1, first["report_digest"]
        )
        == first
    )
    assert (
        files.read_report_for_promotion(
            second_report["run_id"], 1, second["report_digest"]
        )
        == second
    )


def test_report_read_fails_closed_on_noncanonical_bytes(tmp_path: Path) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    report = queued_report()
    candidate = files.publish_report_candidate(report)
    files.promote_report(report["run_id"], 1, candidate["report_digest"])
    path = tmp_path / "training" / "runs" / report["run_id"] / "reports" / "1.json"
    path.write_text(json.dumps(report, indent=2))

    with pytest.raises(ArtifactBytesCorrupt, match="not canonical"):
        files.read_report(report["run_id"], 1)


def test_delete_reports_removes_candidates_but_not_other_run_content(
    tmp_path: Path,
) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    report = queued_report()
    candidate = files.publish_report_candidate(report)
    run_root = tmp_path / "training" / "runs" / report["run_id"]
    run_root.mkdir(parents=True)
    module_candidate = run_root / "candidate.bin"
    module_candidate.write_bytes(b"candidate")

    files.delete_reports(report["run_id"])

    assert module_candidate.read_bytes() == b"candidate"
    assert files.report_history_run_ids(2) == []
    with pytest.raises(ArtifactBytesMissing):
        files.read_report_for_promotion(report["run_id"], 1, candidate["report_digest"])


def test_transient_cleanup_preserves_advertised_report_history(tmp_path: Path) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    report = queued_report()
    accepted = files.publish_report_candidate(report)
    files.promote_report(report["run_id"], 1, accepted["report_digest"])
    competing = {**report, "messages": [{"severity": "info", "message": "later"}]}
    files.publish_report_candidate(competing)

    assert files.transient_run_ids(2) == [report["run_id"]]
    assert files.report_run_ids(2) == [report["run_id"]]

    files.delete_transients(report["run_id"])

    assert files.transient_run_ids(2) == []
    assert files.read_report(report["run_id"], 1) == accepted


def test_terminal_cleanup_keeps_only_the_exact_advertised_report(
    tmp_path: Path,
) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    first_report = queued_report()
    first = files.publish_report_candidate(first_report)
    files.promote_report(first_report["run_id"], 1, first["report_digest"])
    final_report = {
        **first_report,
        "messages": [{"severity": "info", "message": "Final report"}],
    }
    final = files.publish_report_candidate(final_report)
    files.promote_report(first_report["run_id"], 2, final["report_digest"])
    assert files.report_history_run_ids(2) == [first_report["run_id"]]

    files.prune_report_history(first_report["run_id"], 2, final["report_digest"])

    reports_root = tmp_path / "training" / "runs" / first_report["run_id"] / "reports"
    assert [path.name for path in reports_root.iterdir()] == ["2.json"]
    assert files.report_history_run_ids(2) == []
    assert files.read_report(first_report["run_id"], 2) == final


def test_terminal_cleanup_verifies_retained_report_before_deleting_history(
    tmp_path: Path,
) -> None:
    files = TrainingRunFiles(tmp_path, training_api_limits())
    report = queued_report()
    first = files.publish_report_candidate(report)
    files.promote_report(report["run_id"], 1, first["report_digest"])

    with pytest.raises(ArtifactBytesCorrupt, match="digest differs"):
        files.prune_report_history(report["run_id"], 1, "0" * 64)

    assert files.read_report(report["run_id"], 1) == first
