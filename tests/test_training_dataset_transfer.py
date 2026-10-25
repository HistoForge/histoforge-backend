from __future__ import annotations

import copy

import pytest
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.training.datasets.transfer import (
    training_dataset_chunk_range,
    training_dataset_start_digest,
    validate_training_dataset_chunk_receipt,
    validate_training_dataset_cleanup_record,
    validate_training_dataset_page,
    validate_training_dataset_record,
    validate_training_dataset_start_request,
    validate_training_dataset_upload_record,
)
from tests.test_training_dataset import training_api_limits


def start_request() -> dict:
    return {
        "schema_version": 1,
        "dataset_digest": "1" * 64,
        "archive_bytes_digest": "2" * 64,
        "total_bytes": 20_000_001,
        "chunk_size_bytes": 8 * 1024**2,
    }


def retained_record(dataset_id: str, created_at: str) -> dict:
    return {
        "type": "retained",
        "dataset_id": dataset_id,
        "dataset_digest": "1" * 64,
        "retained_bytes": 19_000_000,
        "task_display_name": "Leydig cell segmentation",
        "image_count": 2,
        "roi_count": 5,
        "created_at": created_at,
    }


def test_start_and_chunk_plan_are_exact() -> None:
    request = validate_training_dataset_start_request(
        start_request(), training_api_limits()
    )
    assert training_dataset_start_digest(
        request, training_api_limits()
    ) == canonical_digest(request)
    assert training_dataset_chunk_range(request, 2, training_api_limits()) == {
        "chunk_index": 2,
        "offset": 16_777_216,
        "size_bytes": 3_222_785,
    }


def test_upload_record_binds_complete_start_request() -> None:
    request = start_request()
    digest = training_dataset_start_digest(request, training_api_limits())
    record = {
        "type": "uploading",
        "dataset_id": "11111111-1111-4111-8111-111111111111",
        "request": request,
        "start_request_digest": digest,
        "created_at": "2026-09-02T12:00:00.000Z",
    }
    assert (
        validate_training_dataset_upload_record(record, training_api_limits()) == record
    )
    wrong = copy.deepcopy(record)
    wrong["start_request_digest"] = "f" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_training_dataset_upload_record(wrong, training_api_limits())

    claimed = {
        **record,
        "seal_claim": {
            "claim_token": "22222222-2222-4222-8222-222222222222",
            "expires_at": "2026-09-02T12:06:00.000Z",
        },
    }
    assert (
        validate_training_dataset_upload_record(claimed, training_api_limits())
        == claimed
    )

    malformed = copy.deepcopy(claimed)
    malformed["seal_claim"]["claim_token"] = "not-a-uuid"
    with pytest.raises(ValueError, match="canonical UUID"):
        validate_training_dataset_upload_record(malformed, training_api_limits())


def test_chunk_receipt_retains_length_digest_and_acceptance_time() -> None:
    receipt = {
        "dataset_id": "11111111-1111-4111-8111-111111111111",
        "chunk": {
            "start_request_digest": "3" * 64,
            "chunk_index": 1,
            "size_bytes": 8 * 1024**2,
            "sha256": "4" * 64,
            "accepted_at": "2026-09-02T12:01:00.000Z",
        },
        "expires_at": "2026-09-02T13:01:00.000Z",
    }
    assert validate_training_dataset_chunk_receipt(receipt) == receipt


def test_cleanup_record_has_no_reusable_dataset_digest() -> None:
    cleanup = {
        "type": "cleanup-required",
        "dataset_id": "11111111-1111-4111-8111-111111111111",
        "task_display_name": "Leydig cell segmentation",
        "image_count": 2,
        "roi_count": 5,
        "last_known_remaining_bytes": 1024,
        "created_at": "2026-09-02T12:00:00.000Z",
    }
    assert (
        validate_training_dataset_cleanup_record(cleanup, training_api_limits())
        == cleanup
    )
    cleanup["dataset_digest"] = "1" * 64
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_training_dataset_cleanup_record(cleanup, training_api_limits())


def test_dataset_page_is_created_descending_then_id_ascending() -> None:
    first = retained_record(
        "11111111-1111-4111-8111-111111111111", "2026-09-02T13:00:00.000Z"
    )
    second = retained_record(
        "22222222-2222-4222-8222-222222222222", "2026-09-02T12:00:00.000Z"
    )
    assert validate_training_dataset_record(first, training_api_limits()) == first
    page = {
        "items": [
            {"record": first, "in_use": False},
            {"record": second, "in_use": True},
        ]
    }
    assert validate_training_dataset_page(page, training_api_limits()) == page
    page["items"].reverse()
    with pytest.raises(ValueError, match="uniquely ordered"):
        validate_training_dataset_page(page, training_api_limits())


def test_retained_record_remains_readable_after_admission_limits_change() -> None:
    record = retained_record(
        "11111111-1111-4111-8111-111111111111", "2026-09-02T13:00:00.000Z"
    )
    record["image_count"] = 3_000
    record["roi_count"] = 3_000
    assert validate_training_dataset_record(record, training_api_limits()) == record


def test_retained_display_text_remains_readable_after_limit_reduction() -> None:
    limits = training_api_limits()
    limits["report"]["max_text_bytes"] = 8
    record = retained_record(
        "11111111-1111-4111-8111-111111111111",
        "2026-09-02T13:00:00.000Z",
    )
    record["task_display_name"] = "123456789"

    assert validate_training_dataset_record(record, limits) == record
