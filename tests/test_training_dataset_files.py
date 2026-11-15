from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.manifest import build_training_dataset_manifest
from histoforge.cloud.training.datasets.transfer import (
    training_dataset_start_digest,
)
from tests.test_training_dataset import (
    manifest_preimage,
    training_api_limits,
)

DATASET_ID = "11111111-1111-4111-8111-111111111111"


def test_chunk_pairs_replay_and_assemble_exact_archive(tmp_path: Path) -> None:
    commits = []
    reloads = []
    store = TrainingDatasetFiles(
        tmp_path / "volume",
        reload_volume=lambda: reloads.append(True),
        commit_volume=lambda: commits.append(True),
    )
    archive = b"abcdefghij"
    start = _start_request(archive, chunk_size=4)
    start_digest = training_dataset_start_digest(start, training_api_limits())
    for index, body in enumerate((b"abcd", b"efgh", b"ij")):
        receipt = {
            "start_request_digest": start_digest,
            "chunk_index": index,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "accepted_at": f"2026-09-02T12:00:0{index}.000Z",
        }
        assert store.publish_chunk(DATASET_ID, body, receipt) == receipt
        assert store.publish_chunk(DATASET_ID, body, receipt) == receipt

    assert len(commits) == 3
    assert set(store.chunk_commits(DATASET_ID, start, training_api_limits())) == {
        0,
        1,
        2,
    }
    destination = tmp_path / "assembled.zip"
    assert (
        store.copy_archive(
            DATASET_ID, start, training_api_limits(), destination
        ).read_bytes()
        == archive
    )
    assert reloads


def test_chunk_pair_rejects_different_replay(tmp_path: Path) -> None:
    store = TrainingDatasetFiles(tmp_path / "volume")
    body = b"abcd"
    start = _start_request(body, chunk_size=4)
    receipt = {
        "start_request_digest": training_dataset_start_digest(
            start, training_api_limits()
        ),
        "chunk_index": 0,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "accepted_at": "2026-09-02T12:00:00.000Z",
    }
    store.publish_chunk(DATASET_ID, body, receipt)
    different = {**receipt, "sha256": hashlib.sha256(b"wxyz").hexdigest()}
    with pytest.raises(ArtifactBytesCorrupt, match="different|match"):
        store.publish_chunk(DATASET_ID, b"wxyz", different)


@pytest.mark.parametrize(
    "fault",
    (
        "missing-data",
        "missing-receipt",
        "extra-member",
        "extra-index",
        "mismatched-receipt",
        "corrupt-body",
    ),
)
def test_chunk_plan_rejects_incomplete_or_corrupt_pairs(
    tmp_path: Path,
    fault: str,
) -> None:
    store = TrainingDatasetFiles(tmp_path / "volume")
    archive = b"abcdefghij"
    start = _start_request(archive, chunk_size=4)
    start_digest = training_dataset_start_digest(start, training_api_limits())
    bodies = (b"abcd", b"efgh", b"ij")
    receipts = []
    for index, body in enumerate(bodies):
        receipt = {
            "start_request_digest": start_digest,
            "chunk_index": index,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "accepted_at": f"2026-09-02T12:00:0{index}.000Z",
        }
        receipts.append(receipt)
        store.publish_chunk(DATASET_ID, body, receipt)

    chunks = tmp_path / "volume" / "training" / "uploads" / DATASET_ID / "chunks"
    first = chunks / "0"
    if fault == "missing-data":
        (first / "data.bin").unlink()
    elif fault == "missing-receipt":
        (first / "receipt.json").unlink()
    elif fault == "extra-member":
        (first / "unexpected.bin").write_bytes(b"unexpected")
    elif fault == "extra-index":
        shutil.copytree(first, chunks / "3")
    elif fault == "mismatched-receipt":
        (first / "receipt.json").write_bytes(
            canonical_bytes({**receipts[0], "chunk_index": 1})
        )
    elif fault == "corrupt-body":
        (first / "data.bin").write_bytes(b"wxyz")
    else:  # pragma: no cover - parameter list is closed above
        raise AssertionError(f"unsupported test fault: {fault}")

    with pytest.raises(ArtifactBytesCorrupt):
        store.chunk_commits(DATASET_ID, start, training_api_limits())


def test_chunk_replay_adopts_the_existing_durable_acceptance_time(
    tmp_path: Path,
) -> None:
    store = TrainingDatasetFiles(tmp_path / "volume")
    body = b"abcd"
    start = _start_request(body, chunk_size=4)
    first = {
        "start_request_digest": training_dataset_start_digest(
            start, training_api_limits()
        ),
        "chunk_index": 0,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "accepted_at": "2026-09-02T12:00:00.000Z",
    }
    replay = {**first, "accepted_at": "2026-09-02T12:05:01.000Z"}

    store.publish_chunk(DATASET_ID, body, first)

    assert store.publish_chunk(DATASET_ID, body, replay) == first


def test_receipt_observation_is_bounded_and_replay_checks_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TrainingDatasetFiles(tmp_path / "volume")
    body = b"abcd"
    start = _start_request(body, chunk_size=4)
    receipt = {
        "start_request_digest": training_dataset_start_digest(
            start, training_api_limits()
        ),
        "chunk_index": 0,
        "size_bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "accepted_at": "2026-09-02T12:00:00.000Z",
    }
    store.publish_chunk(DATASET_ID, body, receipt)

    def reject_body_hash(_path: Path):
        raise AssertionError("receipt observation must not hash chunk bodies")

    monkeypatch.setattr(
        "histoforge.cloud.training.datasets.files._stream_identity",
        reject_body_hash,
    )
    assert store.chunk_receipt(DATASET_ID, 0) == receipt
    assert store.chunk_receipts(DATASET_ID, start, training_api_limits()) == {
        0: receipt
    }

    monkeypatch.undo()
    receipt_path = (
        tmp_path
        / "volume"
        / "training"
        / "uploads"
        / DATASET_ID
        / "chunks"
        / "0"
        / "receipt.json"
    )
    divergent = {**receipt, "accepted_at": "2026-09-02T12:01:00.000Z"}
    receipt_path.write_bytes(canonical_bytes(divergent))
    with pytest.raises(ArtifactBytesCorrupt, match="receipt differs"):
        store.require_chunk(DATASET_ID, receipt)


def test_retained_dataset_is_first_wins_and_exactly_deletable(tmp_path: Path) -> None:
    commits = []
    store = TrainingDatasetFiles(
        tmp_path / "volume", commit_volume=lambda: commits.append(True)
    )
    local = tmp_path / "local-dataset"
    local.mkdir()
    payloads = {
        "artifacts/0/origin.json": b"origin",
        "artifacts/0/source.png": b"png",
        "rois/0/objects.geojson": b"objects",
        "rois/0/roi.geojson": b"roi",
    }
    preimage = manifest_preimage()
    preimage["files"] = [
        {
            "path": path,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for path, value in sorted(payloads.items())
    ]
    manifest = build_training_dataset_manifest(preimage, training_api_limits())
    (local / "dataset.json").write_bytes(canonical_bytes(manifest))
    for relative, value in payloads.items():
        path = local.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    assert (
        store.publish_dataset(DATASET_ID, local, manifest, training_api_limits())
        == manifest
    )
    assert (
        store.publish_dataset(DATASET_ID, local, manifest, training_api_limits())
        == manifest
    )
    assert len(commits) == 1
    assert store.read_manifest(DATASET_ID, training_api_limits()) == manifest
    assert store.dataset_remaining_bytes(DATASET_ID) > 0

    store.delete_dataset(DATASET_ID)
    assert store.dataset_remaining_bytes(DATASET_ID) == 0
    assert len(commits) == 2


def test_retained_manifest_read_is_bounded_before_parsing(tmp_path: Path) -> None:
    store = TrainingDatasetFiles(tmp_path / "volume")
    local = tmp_path / "local-dataset"
    local.mkdir()
    payloads = {
        "artifacts/0/origin.json": b"origin",
        "artifacts/0/source.png": b"png",
        "rois/0/objects.geojson": b"objects",
        "rois/0/roi.geojson": b"roi",
    }
    preimage = manifest_preimage()
    preimage["files"] = [
        {
            "path": path,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for path, value in sorted(payloads.items())
    ]
    provisional = build_training_dataset_manifest(preimage, training_api_limits())
    limits = training_api_limits()
    limits["dataset"]["max_json_member_bytes"] = len(canonical_bytes(provisional))
    manifest = build_training_dataset_manifest(preimage, limits)
    (local / "dataset.json").write_bytes(canonical_bytes(manifest))
    for relative, value in payloads.items():
        path = local.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    store.publish_dataset(DATASET_ID, local, manifest, limits)

    retained_manifest = (
        tmp_path / "volume" / "training" / "datasets" / DATASET_ID / "dataset.json"
    )
    retained_manifest.write_bytes(canonical_bytes(manifest) + b"x")
    with pytest.raises(ArtifactBytesCorrupt, match="max_json_member_bytes"):
        store.read_manifest(DATASET_ID, limits)
    with pytest.raises(ArtifactBytesCorrupt, match="max_json_member_bytes"):
        store.require_dataset(DATASET_ID, manifest, limits)


def test_upload_ids_discovers_only_the_requested_bounded_batch(tmp_path: Path) -> None:
    store = TrainingDatasetFiles(tmp_path / "volume")
    upload_root = tmp_path / "volume" / "training" / "uploads"
    dataset_ids = sorted(str(uuid.uuid4()) for _ in range(4))
    for dataset_id in dataset_ids:
        (upload_root / dataset_id).mkdir(parents=True)

    discovered = store.upload_ids(2)

    assert len(discovered) == 2
    assert set(discovered).issubset(dataset_ids)


def _start_request(value: bytes, *, chunk_size: int) -> dict:
    return {
        "schema_version": 1,
        "dataset_digest": "1" * 64,
        "archive_bytes_digest": hashlib.sha256(value).hexdigest(),
        "total_bytes": len(value),
        "chunk_size_bytes": chunk_size,
    }
