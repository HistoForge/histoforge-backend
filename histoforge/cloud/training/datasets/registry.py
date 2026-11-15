"""Retained training-dataset queries and deletion transitions."""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Callable, Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    parse_json,
    require_canonical_uuid,
    require_positive_int32,
    require_rfc3339_millis,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.datasets.transfer import (
    validate_training_dataset_page,
    validate_training_dataset_summary,
)
from histoforge.cloud.training.limits import validate_training_api_limits

_CURSOR_FIELDS = {"created_at", "dataset_id"}
_CURSOR_TEXT = re.compile(r"[A-Za-z0-9_-]+")


class TrainingDatasetRegistry:
    """Project retained rows and fence exact-root dataset deletion."""

    def __init__(
        self,
        records: TrainingControlRecords,
        api_limits: Any,
        *,
        dataset_in_use: Callable[[str], bool] = lambda _dataset_id: False,
    ) -> None:
        self._records = records
        self._api_limits = validate_training_api_limits(api_limits)
        self._dataset_in_use = dataset_in_use

    def get(self, dataset_id: str) -> dict[str, Any]:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        record = self._records.get_dataset(canonical_id)
        if record is None or record["type"] == "uploading":
            raise _dataset_not_found(canonical_id)
        return self._summary(record)

    def page(self, limit: Any, cursor: str | None = None) -> dict[str, Any]:
        page_limit = require_positive_int32(limit, "limit")
        if page_limit > self._api_limits["max_page_size"]:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "limit exceeds the training dataset page maximum",
            )
        after = _decode_cursor(cursor) if cursor is not None else None
        records = [
            record
            for record in self._records.datasets()
            if record["type"] in {"retained", "cleanup-required"}
        ]
        records.sort(key=_record_order)
        if after is not None:
            after_key = _record_order(after)
            records = [
                record for record in records if _record_order(record) > after_key
            ]

        summaries = [self._summary(record) for record in records]
        selected: list[dict[str, Any]] = []
        for index, summary in enumerate(summaries[:page_limit]):
            candidate_items = [*selected, summary]
            candidate: dict[str, Any] = {"items": candidate_items}
            if index + 1 < len(summaries):
                candidate["next_cursor"] = _encode_cursor(candidate_items[-1]["record"])
            if (
                len(canonical_bytes(candidate))
                > self._api_limits["max_page_encoded_bytes"]
            ):
                break
            selected.append(summary)
        if summaries and not selected:
            raise _invalid("training-datasets")

        result: dict[str, Any] = {"items": selected}
        if len(selected) < len(summaries):
            result["next_cursor"] = _encode_cursor(selected[-1]["record"])
        return validate_training_dataset_page(result, self._api_limits)

    def begin_delete(self, dataset_id: str) -> dict[str, Any] | None:
        """Fence one retained dataset before outside-lane recursive deletion."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        record = self._records.get_dataset(canonical_id)
        if record is None:
            return None
        if record["type"] == "uploading":
            raise _dataset_not_found(canonical_id)
        if record["type"] == "cleanup-required":
            return record
        if self._dataset_in_use(canonical_id):
            raise GatewayFailure(
                GatewayErrorCode.TRAINING_STATE_CONFLICT,
                "the training dataset is in use",
                resource_id=canonical_id,
            )
        cleanup = canonical_tree(
            {
                "type": "cleanup-required",
                "dataset_id": canonical_id,
                "task_display_name": record["task_display_name"],
                "image_count": record["image_count"],
                "roi_count": record["roi_count"],
                "last_known_remaining_bytes": record["retained_bytes"],
                "created_at": record["created_at"],
            }
        )
        return self._records.replace_dataset(cleanup)

    def finish_delete(self, dataset_id: str) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        current = self._records.get_dataset(canonical_id)
        if current is None:
            return
        if current["type"] != "cleanup-required":
            raise _invalid(canonical_id)
        self._records.remove_dataset(canonical_id)

    def record_delete_failure(
        self,
        dataset_id: str,
        error: Any,
        remaining_bytes: Any,
    ) -> dict[str, Any]:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        current = self._records.get_dataset(canonical_id)
        if current is None or current["type"] != "cleanup-required":
            raise _invalid(canonical_id)
        updated = dict(current)
        updated["last_error"] = error
        updated["last_known_remaining_bytes"] = remaining_bytes
        return self._records.replace_dataset(updated)

    def _summary(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return validate_training_dataset_summary(
            {
                "record": record,
                "in_use": record["type"] == "retained"
                and self._dataset_in_use(record["dataset_id"]),
            },
            self._api_limits,
        )


def _record_order(record: Mapping[str, Any]) -> tuple[tuple[int, ...], str]:
    created_at = require_rfc3339_millis(record["created_at"], "created_at")
    dataset_id = require_canonical_uuid(record["dataset_id"], "dataset_id")
    return tuple(-ord(character) for character in created_at), dataset_id


def _encode_cursor(record: Mapping[str, Any]) -> str:
    encoded = canonical_bytes(
        {"created_at": record["created_at"], "dataset_id": record["dataset_id"]}
    )
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> dict[str, str]:
    if not isinstance(value, str) or not _CURSOR_TEXT.fullmatch(value):
        raise _invalid_cursor()
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        cursor = parse_json(decoded)
        if not isinstance(cursor, dict) or set(cursor) != _CURSOR_FIELDS:
            raise ValueError("invalid cursor fields")
        result = {
            "created_at": require_rfc3339_millis(cursor["created_at"], "created_at"),
            "dataset_id": require_canonical_uuid(cursor["dataset_id"], "dataset_id"),
        }
    except (UnicodeDecodeError, ValueError, binascii.Error) as error:
        raise _invalid_cursor() from error
    if _encode_cursor(result) != value:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "noncanonical training dataset cursor",
        )
    return result


def _invalid_cursor() -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.VALIDATION_ERROR, "invalid training dataset cursor"
    )


def _dataset_not_found(dataset_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.TRAINING_DATASET_NOT_FOUND,
        "the training dataset does not exist",
        resource_id=dataset_id,
    )


def _invalid(dataset_id: str) -> GatewayFailure:
    return GatewayFailure(
        GatewayErrorCode.INTERNAL_ERROR,
        "training dataset state is inconsistent",
        resource_id=dataset_id,
    )
