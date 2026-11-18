from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.datasets.manifest import build_training_dataset_manifest
from histoforge.cloud.training.datasets.registry import TrainingDatasetRegistry
from histoforge.cloud.training.datasets.state import TrainingDatasetState
from tests.test_training_dataset import (
    manifest_preimage,
    training_api_limits,
)

DATASET_ID = "11111111-1111-4111-8111-111111111111"
SECOND_DATASET_ID = "22222222-2222-4222-8222-222222222222"
NOW = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)


def service_limits(**overrides: int) -> dict[str, int]:
    limits = {
        "max_active_dataset_uploads": 4,
        "dataset_upload_idle_timeout_seconds": 3600,
        "dataset_writer_claim_seconds": 1860,
        "max_active_training_runs": 2,
        "worker_call_attachment_timeout_seconds": 60,
        "max_training_execution_seconds": 86_400,
        "training_queue_allowance_seconds": 600,
        "pending_snapshot_cleanup_age_seconds": 3600,
        "max_control_records": 100,
        "max_control_stored_bytes": 1024**2,
    }
    limits.update(overrides)
    return limits


def start_request(
    *,
    dataset_digest: str = "1" * 64,
    archive_digest: str = "2" * 64,
    total_bytes: int = 10,
    chunk_size_bytes: int = 4,
) -> dict:
    return {
        "schema_version": 1,
        "dataset_digest": dataset_digest,
        "archive_bytes_digest": archive_digest,
        "total_bytes": total_bytes,
        "chunk_size_bytes": chunk_size_bytes,
    }


def make_state(
    root: Path,
    *,
    now: list[datetime] | None = None,
    ids: list[str] | None = None,
    in_use=lambda _dataset_id: False,
    service_overrides: dict[str, int] | None = None,
) -> TrainingDatasetState:
    times = now or [NOW]
    generated_ids = iter(ids or [DATASET_ID])
    limits = service_limits(**(service_overrides or {}))
    records = TrainingControlRecords(root, training_api_limits(), limits)
    return TrainingDatasetState(
        records,
        training_api_limits(),
        limits,
        clock=lambda: times[0],
        id_factory=lambda: uuid.UUID(next(generated_ids)),
        dataset_in_use=in_use,
    )


def test_reservation_replays_exact_request_and_rejects_digest_collision(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    accepted = state.reserve(start_request(), {})
    assert accepted["type"] == "upload"
    assert accepted["dataset_id"] == DATASET_ID

    observations = {DATASET_ID: {}}
    assert state.reserve(start_request(), observations) == accepted
    different_archive = start_request(archive_digest="3" * 64)
    with pytest.raises(GatewayFailure) as raised:
        state.reserve(different_archive, observations)
    assert raised.value.error["code"] == "TRAINING_STATE_CONFLICT"


def test_reservation_enforces_active_and_control_capacity(tmp_path: Path) -> None:
    state = make_state(
        tmp_path,
        ids=[DATASET_ID, SECOND_DATASET_ID],
        service_overrides={"max_active_dataset_uploads": 1},
    )
    state.reserve(start_request(), {})
    with pytest.raises(GatewayFailure) as raised:
        state.reserve(
            start_request(dataset_digest="3" * 64),
            {DATASET_ID: {}},
        )
    assert raised.value.error["code"] == "RATE_LIMITED"

    other = make_state(
        tmp_path / "control-cap",
        ids=[DATASET_ID, SECOND_DATASET_ID],
        service_overrides={"max_control_records": 1},
    )
    other.reserve(start_request(), {})
    with pytest.raises(GatewayFailure) as raised:
        other.reserve(
            start_request(dataset_digest="4" * 64),
            {DATASET_ID: {}},
        )
    assert raised.value.error["code"] == "RATE_LIMITED"


def test_upload_root_cleanup_requires_absent_or_retained_ownership(
    tmp_path: Path,
) -> None:
    state = make_state(tmp_path)
    assert state.upload_root_cleanup_permissions([DATASET_ID]) == {DATASET_ID: True}

    manifest = build_training_dataset_manifest(
        manifest_preimage(), training_api_limits()
    )
    archive = b"data"
    request = start_request(
        dataset_digest=manifest["dataset_digest"],
        archive_digest=hashlib.sha256(archive).hexdigest(),
        total_bytes=len(archive),
        chunk_size_bytes=len(archive),
    )
    accepted = state.reserve(request, {})
    assert state.upload_root_cleanup_permissions([DATASET_ID]) == {DATASET_ID: False}

    claim = state.claim_chunk(
        DATASET_ID,
        0,
        len(archive),
        hashlib.sha256(archive).hexdigest(),
        {},
    )
    commits = {0: claim["commit"]}
    state.commit_chunk(DATASET_ID, claim["commit"], commits)
    seal_claim = state.claim_seal(
        DATASET_ID,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
        commits,
    )
    retained = state.commit_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        seal_claim["claim_token"],
        manifest,
        retained_bytes=1234,
        task_display_name="Leydig cell segmentation",
    )
    assert retained["type"] == "retained"
    assert state.upload_root_cleanup_permissions([DATASET_ID]) == {DATASET_ID: True}

    state.begin_delete(DATASET_ID)
    assert state.upload_root_cleanup_permissions([DATASET_ID]) == {DATASET_ID: False}


def test_upload_root_cleanup_rejects_duplicate_ids(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        state.upload_root_cleanup_permissions([DATASET_ID, DATASET_ID])


def test_chunk_claim_publication_commit_and_replay_are_exact(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.reserve(start_request(), {})
    body = b"abcd"
    digest = hashlib.sha256(body).hexdigest()

    claim = state.claim_chunk(DATASET_ID, 0, len(body), digest, {})
    assert claim["commit"]["chunk_index"] == 0
    commits = {0: claim["commit"]}
    receipt = state.commit_chunk(DATASET_ID, claim["commit"], commits)
    assert receipt["chunk"] == claim["commit"]

    replay = state.claim_chunk(DATASET_ID, 0, len(body), digest, commits)
    assert replay == {
        "accepted": claim["commit"],
        "expires_at": receipt["expires_at"],
    }
    with pytest.raises(GatewayFailure) as raised:
        state.claim_chunk(
            DATASET_ID,
            0,
            len(body),
            hashlib.sha256(b"wxyz").hexdigest(),
            commits,
        )
    assert raised.value.error["code"] == "TRAINING_STATE_CONFLICT"


def test_chunk_commit_adopts_matching_earlier_durable_receipt(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.reserve(start_request(), {})
    body = b"abcd"
    digest = hashlib.sha256(body).hexdigest()
    claim = state.claim_chunk(DATASET_ID, 0, len(body), digest, {})
    durable = {
        **claim["commit"],
        "accepted_at": "2026-09-02T11:55:00.000Z",
    }

    receipt = state.commit_chunk(DATASET_ID, claim["commit"], {0: durable})

    assert receipt["chunk"] == durable


def test_maximum_chunk_plan_does_not_grow_the_upload_control_record(
    tmp_path: Path,
) -> None:
    limits = training_api_limits()
    chunk_size = limits["transfer"]["max_chunk_bytes"]
    chunk_count = limits["transfer"]["max_chunks_per_upload"]
    request = start_request(
        total_bytes=chunk_size * chunk_count,
        chunk_size_bytes=chunk_size,
    )
    state = make_state(tmp_path)
    accepted = state.reserve(request, {})
    record_path = (
        tmp_path / "training" / "control" / "v1" / "datasets" / f"{DATASET_ID}.json"
    )
    initial_record = record_path.read_bytes()
    final_index = chunk_count - 1
    commits = {
        index: {
            "start_request_digest": accepted["start_request_digest"],
            "chunk_index": index,
            "size_bytes": chunk_size,
            "sha256": "3" * 64,
            "accepted_at": "2026-09-02T12:10:00.000Z",
        }
        for index in range(final_index)
    }

    final_claim = state.claim_chunk(
        DATASET_ID,
        final_index,
        chunk_size,
        "3" * 64,
        commits,
    )
    assert record_path.read_bytes() == initial_record
    final_commit = final_claim["commit"]
    complete_commits = {**commits, final_index: final_commit}
    final_receipt = state.commit_chunk(
        DATASET_ID,
        final_commit,
        complete_commits,
    )
    assert final_receipt["chunk"] == final_commit
    assert record_path.read_bytes() == initial_record
    commits = complete_commits

    replay = state.reserve(request, {DATASET_ID: commits})
    assert replay["type"] == "upload"
    assert record_path.read_bytes() == initial_record

    seal_request = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }
    with pytest.raises(GatewayFailure) as raised:
        state.claim_seal(DATASET_ID, seal_request, dict(list(commits.items())[:-1]))
    assert raised.value.error["code"] == "VALIDATION_ERROR"

    extra = {**commits, chunk_count: commits[0]}
    with pytest.raises(
        ValueError,
        match="chunk_id lies outside the canonical chunk plan",
    ):
        state.claim_seal(DATASET_ID, seal_request, extra)

    mismatched = {**commits, 0: {**commits[0], "chunk_index": 1}}
    with pytest.raises(GatewayFailure) as raised:
        state.claim_seal(DATASET_ID, seal_request, mismatched)
    assert raised.value.error["code"] == "INTERNAL_ERROR"

    claim = state.claim_seal(DATASET_ID, seal_request, commits)
    assert claim["start_request"] == request
    persisted = json.loads(record_path.read_text())
    assert set(persisted) == {
        "created_at",
        "dataset_id",
        "request",
        "seal_claim",
        "start_request_digest",
        "type",
    }


def test_expired_upload_is_fenced_until_exact_cleanup_finishes(
    tmp_path: Path,
) -> None:
    now = [NOW]
    state = make_state(
        tmp_path,
        now=now,
        ids=[DATASET_ID, SECOND_DATASET_ID],
        service_overrides={"dataset_upload_idle_timeout_seconds": 10},
    )
    state.reserve(start_request(), {})
    now[0] += timedelta(seconds=10)

    assert state.claim_expired_upload(DATASET_ID, {}) is not None
    assert state.claim_expired_upload(DATASET_ID, {}) is None
    with pytest.raises(GatewayFailure) as raised:
        state.claim_chunk(
            DATASET_ID,
            0,
            4,
            hashlib.sha256(b"abcd").hexdigest(),
            {},
        )
    assert raised.value.error["code"] == "TRAINING_UPLOAD_NOT_FOUND"
    with pytest.raises(GatewayFailure) as raised:
        state.reserve(start_request(), {DATASET_ID: {}})
    assert raised.value.error["code"] == "TRAINING_STATE_CONFLICT"

    now[0] += timedelta(seconds=service_limits()["dataset_writer_claim_seconds"] + 1)
    assert state.claim_expired_upload(DATASET_ID, {}) is not None

    record = state.upload_records()[0]
    state.finish_expired_upload_cleanup(DATASET_ID, record["start_request_digest"])
    accepted = state.reserve(start_request(), {})
    assert accepted["dataset_id"] == SECOND_DATASET_ID


def test_expiry_cleanup_rejects_seal_after_a_late_receipt_refreshes_upload(
    tmp_path: Path,
) -> None:
    now = [NOW]
    state = make_state(tmp_path, now=now)
    body = b"data"
    request = start_request(
        archive_digest=hashlib.sha256(body).hexdigest(),
        total_bytes=len(body),
        chunk_size_bytes=len(body),
    )
    accepted = state.reserve(request, {})

    now[0] += timedelta(seconds=3_590)
    late_commit = state.claim_chunk(
        DATASET_ID,
        0,
        len(body),
        hashlib.sha256(body).hexdigest(),
        {},
    )["commit"]
    now[0] += timedelta(seconds=service_limits()["dataset_writer_claim_seconds"] + 1)
    assert state.claim_expired_upload(DATASET_ID, {}) is not None

    with pytest.raises(GatewayFailure) as raised:
        state.claim_seal(
            DATASET_ID,
            {
                "schema_version": 1,
                "start_request_digest": accepted["start_request_digest"],
            },
            {0: late_commit},
        )
    assert raised.value.error["code"] == "TRAINING_UPLOAD_NOT_FOUND"


def test_complete_upload_seals_once_and_becomes_reusable(tmp_path: Path) -> None:
    manifest = build_training_dataset_manifest(
        manifest_preimage(), training_api_limits()
    )
    archive = b"data"
    request = start_request(
        dataset_digest=manifest["dataset_digest"],
        archive_digest=hashlib.sha256(archive).hexdigest(),
        total_bytes=len(archive),
        chunk_size_bytes=len(archive),
    )
    state = make_state(tmp_path)
    accepted = state.reserve(request, {})
    claim = state.claim_chunk(
        DATASET_ID,
        0,
        len(archive),
        hashlib.sha256(archive).hexdigest(),
        {},
    )
    commits = {0: claim["commit"]}
    state.commit_chunk(DATASET_ID, claim["commit"], commits)
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }
    claim = state.claim_seal(DATASET_ID, seal, commits)
    assert claim["start_request"] == request
    with pytest.raises(GatewayFailure) as raised:
        state.claim_seal(DATASET_ID, seal, commits)
    assert raised.value.error["code"] == "RATE_LIMITED"

    with pytest.raises(ValueError, match="max_text_bytes"):
        state.commit_seal(
            DATASET_ID,
            accepted["start_request_digest"],
            claim["claim_token"],
            manifest,
            retained_bytes=1234,
            task_display_name="x"
            * (training_api_limits()["report"]["max_text_bytes"] + 1),
        )

    retained = state.commit_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        claim["claim_token"],
        manifest,
        retained_bytes=1234,
        task_display_name="Leydig cell segmentation",
    )
    assert retained["type"] == "retained"
    assert retained["image_count"] == 1
    assert retained["roi_count"] == 1
    assert state.reserve(request, {}) == {"type": "existing", "dataset": retained}
    assert state.claim_seal(DATASET_ID, seal, {}) == {"sealed": retained}


def test_lower_limits_do_not_block_existing_dataset_reads_or_cleanup(
    tmp_path: Path,
) -> None:
    high = TrainingControlRecords(tmp_path, training_api_limits(), service_limits())
    for dataset_id, digest in (
        (DATASET_ID, "1" * 64),
        (SECOND_DATASET_ID, "2" * 64),
    ):
        high.reserve_dataset(
            {
                "type": "retained",
                "dataset_id": dataset_id,
                "dataset_digest": digest,
                "retained_bytes": 100,
                "task_display_name": "Existing training dataset",
                "image_count": 1,
                "roi_count": 1,
                "created_at": "2026-09-02T12:00:00.000Z",
            }
        )

    low = TrainingControlRecords(
        tmp_path,
        training_api_limits(),
        service_limits(max_control_records=1, max_control_stored_bytes=1),
    )
    assert len(low.datasets()) == 2
    registry = TrainingDatasetRegistry(low, training_api_limits())
    assert registry.begin_delete(DATASET_ID)["type"] == "cleanup-required"
    registry.finish_delete(DATASET_ID)
    assert low.get_dataset(DATASET_ID) is None
    assert low.get_dataset(SECOND_DATASET_ID)["type"] == "retained"


def test_expired_seal_owner_cannot_mutate_successor_claim(tmp_path: Path) -> None:
    now = [NOW]
    state = make_state(tmp_path, now=now)
    manifest = build_training_dataset_manifest(
        manifest_preimage(), training_api_limits()
    )
    archive = b"data"
    request = start_request(
        dataset_digest=manifest["dataset_digest"],
        archive_digest=hashlib.sha256(archive).hexdigest(),
        total_bytes=len(archive),
        chunk_size_bytes=len(archive),
    )
    accepted = state.reserve(request, {})
    chunk = state.claim_chunk(
        DATASET_ID,
        0,
        len(archive),
        hashlib.sha256(archive).hexdigest(),
        {},
    )["commit"]
    commits = {0: chunk}
    state.commit_chunk(DATASET_ID, chunk, commits)
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }

    first = state.claim_seal(DATASET_ID, seal, commits)
    now[0] += timedelta(seconds=service_limits()["dataset_writer_claim_seconds"] - 1)
    state = make_state(tmp_path, now=now)
    with pytest.raises(GatewayFailure) as raised:
        state.claim_seal(DATASET_ID, seal, commits)
    assert raised.value.error["code"] == "RATE_LIMITED"

    now[0] += timedelta(seconds=2)
    state = make_state(tmp_path, now=now)
    second = state.claim_seal(DATASET_ID, seal, commits)
    assert second["claim_token"] != first["claim_token"]

    state = make_state(tmp_path, now=now)
    state.abort_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        first["claim_token"],
    )
    with pytest.raises(GatewayFailure) as raised:
        state.claim_seal(DATASET_ID, seal, commits)
    assert raised.value.error["code"] == "RATE_LIMITED"

    with pytest.raises(GatewayFailure) as raised:
        state.commit_seal(
            DATASET_ID,
            accepted["start_request_digest"],
            first["claim_token"],
            manifest,
            retained_bytes=1234,
            task_display_name="Leydig cell segmentation",
        )
    assert raised.value.error["code"] == "INTERNAL_ERROR"

    retained = state.commit_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        second["claim_token"],
        manifest,
        retained_bytes=1234,
        task_display_name="Leydig cell segmentation",
    )
    assert retained["type"] == "retained"


def test_seal_claim_survives_state_owner_reconstruction(tmp_path: Path) -> None:
    now = [NOW]
    state = make_state(tmp_path, now=now)
    manifest = build_training_dataset_manifest(
        manifest_preimage(), training_api_limits()
    )
    archive = b"data"
    request = start_request(
        dataset_digest=manifest["dataset_digest"],
        archive_digest=hashlib.sha256(archive).hexdigest(),
        total_bytes=len(archive),
        chunk_size_bytes=len(archive),
    )
    accepted = state.reserve(request, {})
    chunk = state.claim_chunk(
        DATASET_ID,
        0,
        len(archive),
        hashlib.sha256(archive).hexdigest(),
        {},
    )["commit"]
    commits = {0: chunk}
    state.commit_chunk(DATASET_ID, chunk, commits)
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }
    claim = state.claim_seal(DATASET_ID, seal, commits)

    reconstructed = make_state(tmp_path, now=now)
    with pytest.raises(GatewayFailure) as raised:
        reconstructed.claim_seal(DATASET_ID, seal, commits)
    assert raised.value.error["code"] == "RATE_LIMITED"
    replay = reconstructed.reserve(request, {DATASET_ID: commits})
    assert replay["type"] == "upload"
    assert "seal_claim" not in replay

    retained = reconstructed.commit_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        claim["claim_token"],
        manifest,
        retained_bytes=1234,
        task_display_name="Leydig cell segmentation",
    )
    assert retained["type"] == "retained"


def test_reconstructed_owner_preserves_seal_cleanup_fence(tmp_path: Path) -> None:
    now = [NOW]
    state = make_state(
        tmp_path,
        now=now,
        service_overrides={"dataset_upload_idle_timeout_seconds": 10},
    )
    archive = b"data"
    request = start_request(
        archive_digest=hashlib.sha256(archive).hexdigest(),
        total_bytes=len(archive),
        chunk_size_bytes=len(archive),
    )
    accepted = state.reserve(request, {})
    chunk = state.claim_chunk(
        DATASET_ID,
        0,
        len(archive),
        hashlib.sha256(archive).hexdigest(),
        {},
    )["commit"]
    commits = {0: chunk}
    state.commit_chunk(DATASET_ID, chunk, commits)
    seal = {
        "schema_version": 1,
        "start_request_digest": accepted["start_request_digest"],
    }
    claim = state.claim_seal(DATASET_ID, seal, commits)
    now[0] += timedelta(seconds=11)

    reconstructed = make_state(
        tmp_path,
        now=now,
        service_overrides={"dataset_upload_idle_timeout_seconds": 10},
    )
    assert reconstructed.claim_expired_upload(DATASET_ID, commits) is None
    reconstructed.abort_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        "33333333-3333-4333-8333-333333333333",
    )
    assert reconstructed.claim_expired_upload(DATASET_ID, commits) is None

    reconstructed.abort_seal(
        DATASET_ID,
        accepted["start_request_digest"],
        claim["claim_token"],
    )
    expired = reconstructed.claim_expired_upload(DATASET_ID, commits)
    assert expired is not None


def test_page_and_delete_project_only_retained_state(tmp_path: Path) -> None:
    state = make_state(tmp_path, in_use=lambda dataset_id: dataset_id == DATASET_ID)
    manifest = build_training_dataset_manifest(
        manifest_preimage(), training_api_limits()
    )
    _seal_single_chunk(state, DATASET_ID, manifest)
    page = state.page(1)
    assert page["items"][0]["record"]["dataset_id"] == DATASET_ID
    assert page["items"][0]["in_use"] is True
    with pytest.raises(GatewayFailure) as raised:
        state.begin_delete(DATASET_ID)
    assert raised.value.error["code"] == "TRAINING_STATE_CONFLICT"

    deletable = make_state(tmp_path, in_use=lambda _dataset_id: False)
    cleanup = deletable.begin_delete(DATASET_ID)
    assert cleanup is not None and cleanup["type"] == "cleanup-required"
    failure = {
        "schema_version": 1,
        "code": "SERVICE_UNAVAILABLE",
        "message": "dataset cleanup did not complete",
    }
    updated = deletable.record_delete_failure(DATASET_ID, failure, 100)
    assert updated["last_error"] == failure
    assert deletable.get(DATASET_ID)["in_use"] is False
    deletable.finish_delete(DATASET_ID)
    assert deletable.page(10) == {"items": []}
    assert deletable.begin_delete(DATASET_ID) is None


def test_noncanonical_persisted_record_fails_closed(tmp_path: Path) -> None:
    state = make_state(tmp_path)
    state.reserve(start_request(), {})
    path = tmp_path / "training" / "control" / "v1" / "datasets" / f"{DATASET_ID}.json"
    parsed = json.loads(path.read_text())
    path.write_text(json.dumps(parsed, indent=2))
    with pytest.raises(GatewayFailure) as raised:
        state.upload_records()
    assert raised.value.error["code"] == "INTERNAL_ERROR"


def _seal_single_chunk(
    state: TrainingDatasetState,
    dataset_id: str,
    manifest: dict,
) -> dict:
    archive = b"data"
    request = start_request(
        dataset_digest=manifest["dataset_digest"],
        archive_digest=hashlib.sha256(archive).hexdigest(),
        total_bytes=len(archive),
        chunk_size_bytes=len(archive),
    )
    accepted = state.reserve(request, {})
    claim = state.claim_chunk(
        dataset_id,
        0,
        len(archive),
        hashlib.sha256(archive).hexdigest(),
        {},
    )
    commits = {0: claim["commit"]}
    state.commit_chunk(dataset_id, claim["commit"], commits)
    seal_claim = state.claim_seal(
        dataset_id,
        {
            "schema_version": 1,
            "start_request_digest": accepted["start_request_digest"],
        },
        commits,
    )
    return state.commit_seal(
        dataset_id,
        accepted["start_request_digest"],
        seal_claim["claim_token"],
        manifest,
        retained_bytes=1234,
        task_display_name="Leydig cell segmentation",
    )
