"""Closed client-visible and backend-private training limit records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_positive_int32,
    require_positive_safe_integer,
)

_DATASET_FIELDS = {
    "max_rois",
    "max_objects_per_roi",
    "max_total_objects",
    "max_vertices_per_geometry",
    "max_total_vertices",
    "max_json_member_bytes",
    "max_archive_bytes",
    "max_extracted_bytes",
}
_TRANSFER_FIELDS = {"max_chunk_bytes", "max_chunks_per_upload"}
_REPORT_FIELDS = {
    "max_metric_series",
    "max_epochs",
    "max_messages",
    "max_text_bytes",
    "max_component_parameters_bytes",
    "max_component_parameter_depth",
    "max_component_collection_entries",
    "max_report_bytes",
}
_API_FIELDS = {
    "dataset",
    "transfer",
    "report",
    "max_page_size",
    "max_page_encoded_bytes",
}
_SERVICE_FIELDS = {
    "max_active_dataset_uploads",
    "dataset_upload_idle_timeout_seconds",
    "dataset_writer_claim_seconds",
    "max_active_training_runs",
    "worker_call_attachment_timeout_seconds",
    "max_training_execution_seconds",
    "training_queue_allowance_seconds",
    "pending_snapshot_cleanup_age_seconds",
    "max_control_records",
    "max_control_stored_bytes",
}


@dataclass(frozen=True)
class TrainingDatasetLimits:
    max_rois: int
    max_objects_per_roi: int
    max_total_objects: int
    max_vertices_per_geometry: int
    max_total_vertices: int
    max_json_member_bytes: int
    max_archive_bytes: int
    max_extracted_bytes: int

    def wire_value(self) -> dict[str, int]:
        return validate_training_dataset_limits(asdict(self))


@dataclass(frozen=True)
class TrainingDatasetTransferLimits:
    max_chunk_bytes: int
    max_chunks_per_upload: int

    def wire_value(self) -> dict[str, int]:
        return validate_training_transfer_limits(asdict(self))


@dataclass(frozen=True)
class TrainingReportLimits:
    max_metric_series: int
    max_epochs: int
    max_messages: int
    max_text_bytes: int
    max_component_parameters_bytes: int
    max_component_parameter_depth: int
    max_component_collection_entries: int
    max_report_bytes: int

    def wire_value(self) -> dict[str, int]:
        return validate_training_report_limits(asdict(self))


@dataclass(frozen=True)
class TrainingApiLimits:
    dataset: TrainingDatasetLimits
    transfer: TrainingDatasetTransferLimits
    report: TrainingReportLimits
    max_page_size: int
    max_page_encoded_bytes: int

    def wire_value(self) -> dict[str, Any]:
        return validate_training_api_limits(asdict(self))

    def validated(self) -> TrainingApiLimits:
        validate_training_api_deployment_limits(asdict(self))
        return self


@dataclass(frozen=True)
class TrainingServiceLimits:
    max_active_dataset_uploads: int
    dataset_upload_idle_timeout_seconds: int
    dataset_writer_claim_seconds: int
    max_active_training_runs: int
    worker_call_attachment_timeout_seconds: int
    max_training_execution_seconds: int
    training_queue_allowance_seconds: int
    pending_snapshot_cleanup_age_seconds: int
    max_control_records: int
    max_control_stored_bytes: int

    def wire_value(self) -> dict[str, int]:
        return validate_training_service_limits(asdict(self))

    def validated(self) -> TrainingServiceLimits:
        self.wire_value()
        return self


def validate_training_dataset_limits(value: Any) -> dict[str, int]:
    item = _exact_object(value, _DATASET_FIELDS, "training dataset limits")
    result = {field: _positive(item, field) for field in _DATASET_FIELDS}
    if result["max_json_member_bytes"] > result["max_extracted_bytes"]:
        raise ValueError("max_json_member_bytes exceeds max_extracted_bytes")
    return canonical_tree(result)


def validate_training_transfer_limits(value: Any) -> dict[str, int]:
    item = _exact_object(value, _TRANSFER_FIELDS, "training transfer limits")
    return canonical_tree(
        {
            "max_chunk_bytes": _positive(item, "max_chunk_bytes"),
            "max_chunks_per_upload": _positive_int32(item, "max_chunks_per_upload"),
        }
    )


def validate_training_report_limits(value: Any) -> dict[str, int]:
    item = _exact_object(value, _REPORT_FIELDS, "training report limits")
    return canonical_tree({field: _positive(item, field) for field in _REPORT_FIELDS})


def validate_training_api_limits(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _API_FIELDS, "training API limits")
    dataset = validate_training_dataset_limits(item["dataset"])
    transfer = validate_training_transfer_limits(item["transfer"])
    if (
        dataset["max_archive_bytes"]
        > transfer["max_chunk_bytes"] * transfer["max_chunks_per_upload"]
    ):
        raise ValueError("max_archive_bytes exceeds the canonical chunk plan")
    return canonical_tree(
        {
            "dataset": dataset,
            "transfer": transfer,
            "report": validate_training_report_limits(item["report"]),
            "max_page_size": _positive_int32(item, "max_page_size"),
            "max_page_encoded_bytes": _positive(item, "max_page_encoded_bytes"),
        }
    )


def validate_training_api_deployment_limits(value: Any) -> dict[str, Any]:
    """Require configured page bytes to hold one maximum legal list item."""

    limits = validate_training_api_limits(value)
    # Local imports keep scalar limit parsing independent from the DTO modules.
    from histoforge.cloud.training.datasets.transfer import (
        maximum_training_dataset_single_item_page_bytes,
    )
    from histoforge.cloud.training.runs.contract import (
        maximum_training_run_single_item_page_bytes,
    )

    required = max(
        maximum_training_dataset_single_item_page_bytes(limits),
        maximum_training_run_single_item_page_bytes(limits),
    )
    if limits["max_page_encoded_bytes"] < required:
        raise ValueError(
            "max_page_encoded_bytes cannot hold one maximum training list item"
        )
    return limits


def validate_training_service_limits(value: Any) -> dict[str, int]:
    item = _exact_object(value, _SERVICE_FIELDS, "training service limits")
    return canonical_tree(
        {
            field: _positive_int32(item, field)
            if field != "max_control_stored_bytes"
            else _positive(item, field)
            for field in _SERVICE_FIELDS
        }
    )


def _positive(value: Mapping[str, Any], field: str) -> int:
    return require_positive_safe_integer(value[field], field)


def _positive_int32(value: Mapping[str, Any], field: str) -> int:
    return require_positive_int32(value[field], field)


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return value
