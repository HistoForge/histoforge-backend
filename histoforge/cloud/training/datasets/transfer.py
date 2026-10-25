"""Shared training-dataset upload, retention, and list DTO contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_nonnegative_safe_integer,
    require_positive_safe_integer,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud._errors import validate_gateway_error
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    canonical_chunk_range,
)
from histoforge.cloud.training.contract import require_training_text_syntax
from histoforge.cloud.training.limits import validate_training_api_limits

_START_FIELDS = {
    "schema_version",
    "dataset_digest",
    "archive_bytes_digest",
    "total_bytes",
    "chunk_size_bytes",
}
_UPLOAD_FIELDS = {
    "type",
    "dataset_id",
    "start_request_digest",
    "expires_at",
}
_EXISTING_FIELDS = {"type", "dataset"}
_SEAL_FIELDS = {"schema_version", "start_request_digest"}
_CHUNK_FIELDS = {
    "start_request_digest",
    "chunk_index",
    "size_bytes",
    "sha256",
    "accepted_at",
}
_RECEIPT_FIELDS = {"dataset_id", "chunk", "expires_at"}
_UPLOAD_RECORD_FIELDS = {
    "type",
    "dataset_id",
    "request",
    "start_request_digest",
    "created_at",
}
_UPLOAD_RECORD_FIELDS_WITH_SEAL_CLAIM = _UPLOAD_RECORD_FIELDS | {"seal_claim"}
_SEAL_CLAIM_RECORD_FIELDS = {"claim_token", "expires_at"}
_RETAINED_FIELDS = {
    "type",
    "dataset_id",
    "dataset_digest",
    "retained_bytes",
    "task_display_name",
    "image_count",
    "roi_count",
    "created_at",
}
_CLEANUP_FIELDS = {
    "type",
    "dataset_id",
    "task_display_name",
    "image_count",
    "roi_count",
    "last_known_remaining_bytes",
    "created_at",
}
_CLEANUP_FIELDS_WITH_ERROR = _CLEANUP_FIELDS | {"last_error"}
_SUMMARY_FIELDS = {"record", "in_use"}
_PAGE_FIELDS = {"items"}
_PAGE_FIELDS_WITH_CURSOR = {"items", "next_cursor"}
_GENERATED_CURSOR_BYTES = 119


def validate_training_dataset_start_request(
    value: Any, api_limits: Any
) -> dict[str, Any]:
    item = _exact_object(value, _START_FIELDS, "training dataset start")
    if require_int32(item["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported training dataset start schema_version")
    limits = validate_training_api_limits(api_limits)
    total = require_positive_safe_integer(item["total_bytes"], "total_bytes")
    chunk_size = require_positive_safe_integer(
        item["chunk_size_bytes"], "chunk_size_bytes"
    )
    if total > limits["dataset"]["max_archive_bytes"]:
        raise ValueError("training dataset archive exceeds max_archive_bytes")
    if chunk_size > limits["transfer"]["max_chunk_bytes"]:
        raise ValueError("chunk_size_bytes exceeds max_chunk_bytes")
    count = canonical_chunk_count(total, chunk_size)
    if count > limits["transfer"]["max_chunks_per_upload"]:
        raise ValueError("training dataset archive exceeds max_chunks_per_upload")
    return canonical_tree(
        {
            "schema_version": 1,
            "dataset_digest": require_sha256(item["dataset_digest"], "dataset_digest"),
            "archive_bytes_digest": require_sha256(
                item["archive_bytes_digest"], "archive_bytes_digest"
            ),
            "total_bytes": total,
            "chunk_size_bytes": chunk_size,
        }
    )


def training_dataset_start_digest(value: Any, api_limits: Any) -> str:
    return canonical_digest(validate_training_dataset_start_request(value, api_limits))


def training_dataset_chunk_range(
    start_request: Any, chunk_index: Any, api_limits: Any
) -> dict[str, int]:
    start = validate_training_dataset_start_request(start_request, api_limits)
    source_range = canonical_chunk_range(
        start["total_bytes"], start["chunk_size_bytes"], chunk_index
    )
    return {
        "chunk_index": source_range["chunk_id"],
        "offset": source_range["offset"],
        "size_bytes": source_range["size"],
    }


def validate_training_dataset_start_result(
    value: Any, api_limits: Any
) -> dict[str, Any]:
    item = _object(value, "training dataset start result")
    kind = item.get("type")
    if kind == "existing":
        exact = _exact_object(item, _EXISTING_FIELDS, "existing training dataset")
        return {
            "type": kind,
            "dataset": validate_training_dataset_record(exact["dataset"], api_limits),
        }
    if kind == "upload":
        exact = _exact_object(item, _UPLOAD_FIELDS, "training dataset upload accepted")
        return {
            "type": kind,
            "dataset_id": require_canonical_uuid(exact["dataset_id"], "dataset_id"),
            "start_request_digest": require_sha256(
                exact["start_request_digest"], "start_request_digest"
            ),
            "expires_at": require_rfc3339_millis(exact["expires_at"], "expires_at"),
        }
    raise ValueError(f"unsupported training dataset start result: {kind!r}")


def validate_training_dataset_seal_request(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _SEAL_FIELDS, "training dataset seal request")
    if require_int32(item["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported training dataset seal schema_version")
    return {
        "schema_version": 1,
        "start_request_digest": require_sha256(
            item["start_request_digest"], "start_request_digest"
        ),
    }


def validate_training_dataset_chunk_commit(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _CHUNK_FIELDS, "training dataset chunk commit")
    return canonical_tree(
        {
            "start_request_digest": require_sha256(
                item["start_request_digest"], "start_request_digest"
            ),
            "chunk_index": require_nonnegative_int32(
                item["chunk_index"], "chunk_index"
            ),
            "size_bytes": require_positive_safe_integer(
                item["size_bytes"], "size_bytes"
            ),
            "sha256": require_sha256(item["sha256"], "sha256"),
            "accepted_at": require_rfc3339_millis(item["accepted_at"], "accepted_at"),
        }
    )


def validate_training_dataset_chunk_receipt(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _RECEIPT_FIELDS, "training dataset chunk receipt")
    return canonical_tree(
        {
            "dataset_id": require_canonical_uuid(item["dataset_id"], "dataset_id"),
            "chunk": validate_training_dataset_chunk_commit(item["chunk"]),
            "expires_at": require_rfc3339_millis(item["expires_at"], "expires_at"),
        }
    )


def validate_training_dataset_upload_record(
    value: Any, api_limits: Any
) -> dict[str, Any]:
    raw = _object(value, "training dataset upload record")
    expected = (
        _UPLOAD_RECORD_FIELDS_WITH_SEAL_CLAIM
        if "seal_claim" in raw
        else _UPLOAD_RECORD_FIELDS
    )
    item = _exact_object(raw, expected, "training dataset upload record")
    if item["type"] != "uploading":
        raise ValueError("training dataset upload record type must be uploading")
    dataset_id = require_canonical_uuid(item["dataset_id"], "dataset_id")
    request = validate_training_dataset_start_request(item["request"], api_limits)
    digest = require_sha256(item["start_request_digest"], "start_request_digest")
    if digest != canonical_digest(request):
        raise ValueError("upload record start_request_digest does not match request")
    result = {
        "type": "uploading",
        "dataset_id": dataset_id,
        "request": request,
        "start_request_digest": digest,
        "created_at": require_rfc3339_millis(item["created_at"], "created_at"),
    }
    if "seal_claim" in item:
        claim = _exact_object(
            item["seal_claim"],
            _SEAL_CLAIM_RECORD_FIELDS,
            "training dataset seal claim",
        )
        result["seal_claim"] = {
            "claim_token": require_canonical_uuid(
                claim["claim_token"], "seal claim token"
            ),
            "expires_at": require_rfc3339_millis(
                claim["expires_at"], "seal claim expires_at"
            ),
        }
    return canonical_tree(result)


def validate_training_dataset_record(value: Any, api_limits: Any) -> dict[str, Any]:
    item = _exact_object(value, _RETAINED_FIELDS, "training dataset record")
    validate_training_api_limits(api_limits)
    if item["type"] != "retained":
        raise ValueError("training dataset record type must be retained")
    image_count = require_positive_safe_integer(item["image_count"], "image_count")
    roi_count = require_positive_safe_integer(item["roi_count"], "roi_count")
    return canonical_tree(
        {
            "type": "retained",
            "dataset_id": require_canonical_uuid(item["dataset_id"], "dataset_id"),
            "dataset_digest": require_sha256(item["dataset_digest"], "dataset_digest"),
            "retained_bytes": require_nonnegative_safe_integer(
                item["retained_bytes"], "retained_bytes"
            ),
            "task_display_name": require_training_text_syntax(
                item["task_display_name"], "task_display_name"
            ),
            "image_count": image_count,
            "roi_count": roi_count,
            "created_at": require_rfc3339_millis(item["created_at"], "created_at"),
        }
    )


def validate_training_dataset_cleanup_record(
    value: Any, api_limits: Any
) -> dict[str, Any]:
    item = _object(value, "training dataset cleanup record")
    validate_training_api_limits(api_limits)
    expected = _CLEANUP_FIELDS_WITH_ERROR if "last_error" in item else _CLEANUP_FIELDS
    _require_fields(item, expected, "training dataset cleanup record")
    if item["type"] != "cleanup-required":
        raise ValueError(
            "training dataset cleanup record type must be cleanup-required"
        )
    image_count = require_positive_safe_integer(item["image_count"], "image_count")
    roi_count = require_positive_safe_integer(item["roi_count"], "roi_count")
    result: dict[str, Any] = {
        "type": "cleanup-required",
        "dataset_id": require_canonical_uuid(item["dataset_id"], "dataset_id"),
        "task_display_name": require_training_text_syntax(
            item["task_display_name"], "task_display_name"
        ),
        "image_count": image_count,
        "roi_count": roi_count,
        "last_known_remaining_bytes": require_nonnegative_safe_integer(
            item["last_known_remaining_bytes"], "last_known_remaining_bytes"
        ),
        "created_at": require_rfc3339_millis(item["created_at"], "created_at"),
    }
    if "last_error" in item:
        result["last_error"] = validate_gateway_error(item["last_error"])
    return canonical_tree(result)


def validate_training_dataset_summary(value: Any, api_limits: Any) -> dict[str, Any]:
    item = _exact_object(value, _SUMMARY_FIELDS, "training dataset summary")
    record_value = _object(item["record"], "training dataset summary record")
    if record_value.get("type") == "retained":
        record = validate_training_dataset_record(record_value, api_limits)
    elif record_value.get("type") == "cleanup-required":
        record = validate_training_dataset_cleanup_record(record_value, api_limits)
    else:
        raise ValueError("training dataset summary has an unsupported record")
    if not isinstance(item["in_use"], bool):
        raise ValueError("in_use must be a boolean")
    if record["type"] == "cleanup-required" and item["in_use"]:
        raise ValueError("cleanup-required dataset cannot be in use")
    return canonical_tree({"record": record, "in_use": item["in_use"]})


def validate_training_dataset_page(value: Any, api_limits: Any) -> dict[str, Any]:
    item = _object(value, "training dataset page")
    expected = _PAGE_FIELDS_WITH_CURSOR if "next_cursor" in item else _PAGE_FIELDS
    _require_fields(item, expected, "training dataset page")
    limits = validate_training_api_limits(api_limits)
    raw_items = _array(item["items"], "items")
    if len(raw_items) > limits["max_page_size"]:
        raise ValueError("training dataset page exceeds max_page_size")
    items = [validate_training_dataset_summary(value, limits) for value in raw_items]
    keys = [
        (summary["record"]["created_at"], summary["record"]["dataset_id"])
        for summary in items
    ]
    expected_keys = sorted(
        keys, key=lambda key: (_descending_timestamp(key[0]), key[1])
    )
    if keys != expected_keys or len(keys) != len(set(keys)):
        raise ValueError("training dataset page is not uniquely ordered")
    result: dict[str, Any] = {"items": items}
    if "next_cursor" in item:
        cursor = item["next_cursor"]
        if not isinstance(cursor, str) or not cursor:
            raise ValueError("next_cursor must be a nonblank string")
        if not items:
            raise ValueError("empty page cannot have next_cursor")
        result["next_cursor"] = cursor
    result = canonical_tree(result)
    if len(canonical_bytes(result)) > limits["max_page_encoded_bytes"]:
        raise ValueError("training dataset page exceeds max_page_encoded_bytes")
    return result


def maximum_training_dataset_single_item_page_bytes(api_limits: Any) -> int:
    """Bound a generated one-item page under current dataset admission limits."""

    limits = validate_training_api_limits(api_limits)
    placeholder = "x"
    page = {
        "items": [
            {
                "record": {
                    "type": "cleanup-required",
                    "dataset_id": "ffffffff-ffff-4fff-bfff-ffffffffffff",
                    "task_display_name": placeholder,
                    "image_count": 9_007_199_254_740_991,
                    "roi_count": 9_007_199_254_740_991,
                    "last_known_remaining_bytes": 9_007_199_254_740_991,
                    "created_at": "9999-12-31T23:59:59.999Z",
                    "last_error": {
                        "schema_version": 1,
                        "code": "REGISTRY_RATE_LIMITED",
                        "message": placeholder,
                        "resource_id": placeholder,
                        "retry_not_before": "9999-12-31T23:59:59.999Z",
                    },
                },
                "in_use": False,
            }
        ],
        "next_cursor": "x" * _GENERATED_CURSOR_BYTES,
    }
    # Three admitted TrainingText values can each double in JSON through
    # permitted quote or backslash escaping.
    return len(canonical_bytes(page)) - 3 + 6 * limits["report"]["max_text_bytes"]


def _descending_timestamp(value: str) -> tuple[int, ...]:
    # Exact UTC-millisecond text has fixed width and lexical chronological order.
    return tuple(-ord(character) for character in value)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    item = _object(value, path)
    _require_fields(item, fields, path)
    return item


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value
