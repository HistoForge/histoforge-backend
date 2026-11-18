"""Immutable training-run request and admission-time scientific checks.

Request syntax remains readable after its dataset or capability is deleted.
Admission separately binds that request to the retained dataset and the current
curated capability. Source-module and selected-model resolution belong to the
catalog-facing admission service, not this pure contract module.
"""

from __future__ import annotations

from typing import Any

from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_nonnegative_safe_integer,
    require_sha256,
)
from histoforge.cloud.inference.identity import validate_module_content_identity
from histoforge.cloud.training.class_schema import validate_class_initialization
from histoforge.cloud.training.contract import (
    require_training_model_id,
    require_training_stable_id,
    require_training_text,
    training_task_from_capability,
    validate_instance_training_settings,
    validate_training_augmentation_selection,
    validate_training_augmentation_selection_syntax,
    validate_training_field_profile,
    validate_training_field_profile_for_capability,
    validate_training_settings,
    validate_training_settings_for_capability,
)
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.split import (
    validate_training_split_plan,
    validate_training_split_plan_syntax,
)
from histoforge.cloud.training.limits import validate_training_api_limits

_REQUEST_FIELDS = {
    "schema_version",
    "dataset_id",
    "dataset_digest",
    "source_module",
    "selected_model_id",
    "result_form",
    "result_display_name",
    "result_inference_gpu",
    "split_plan",
    "field_profile",
    "augmentation",
    "trainer_settings",
    "training_seed",
    "class_initialization",
}
_RESULT_FORMS = {"selected_model_only", "complete_source_workflow"}


def validate_training_run_request(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate a complete request without requiring mutable external state."""

    if (
        not isinstance(value, dict)
        or not _REQUEST_FIELDS <= set(value)
        or (set(value) - _REQUEST_FIELDS - {"min_instance_pixels"})
    ):
        raise ValueError("training run request has missing or unknown fields")
    item = value
    if require_int32(item["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported training run request schema_version")
    limits = validate_training_api_limits(api_limits)
    return canonical_tree(
        {
            "schema_version": 1,
            "class_initialization": validate_class_initialization(
                item["class_initialization"]
            ),
            "dataset_id": require_canonical_uuid(item["dataset_id"], "dataset_id"),
            "dataset_digest": require_sha256(item["dataset_digest"], "dataset_digest"),
            "source_module": validate_module_content_identity(item["source_module"]),
            "selected_model_id": require_training_model_id(
                item["selected_model_id"], "selected_model_id"
            ),
            "result_form": _enum(item["result_form"], _RESULT_FORMS, "result_form"),
            "result_display_name": require_training_text(
                item["result_display_name"],
                "result_display_name",
                limits["report"]["max_text_bytes"],
            ),
            "result_inference_gpu": require_training_stable_id(
                item["result_inference_gpu"], "result_inference_gpu"
            ),
            "split_plan": validate_training_split_plan_syntax(item["split_plan"]),
            "field_profile": validate_training_field_profile(item["field_profile"]),
            **(
                {
                    "min_instance_pixels": require_nonnegative_int32(
                        item["min_instance_pixels"], "min_instance_pixels"
                    )
                }
                if "min_instance_pixels" in item
                else {}
            ),
            "augmentation": validate_training_augmentation_selection_syntax(
                item["augmentation"]
            ),
            "trainer_settings": validate_training_settings(item["trainer_settings"]),
            "training_seed": require_nonnegative_safe_integer(
                item["training_seed"], "training_seed"
            ),
        }
    )


def training_run_request_digest(value: Any, api_limits: Any) -> str:
    """Return the replay identity of one validated run request."""

    return canonical_digest(validate_training_run_request(value, api_limits))


def validate_training_run_request_against_manifest_capability(
    value: Any,
    manifest: Any,
    capability: Any,
    api_limits: Any,
) -> dict[str, Any]:
    """Bind request choices to one dataset manifest and curated capability.

    The admission service must first resolve ``dataset_id`` to a retained
    record whose digest matches this manifest. It must also resolve and
    validate ``source_module`` and ``selected_model_id`` against the installed
    module catalog.
    """

    request = validate_training_run_request(value, api_limits)
    limits = validate_training_api_limits(api_limits)
    dataset = validate_training_dataset_manifest(manifest, limits)
    validate_instance_training_settings(request, dataset["task"]["model_output"])
    if request["dataset_digest"] != dataset["dataset_digest"]:
        raise ValueError("training run dataset_digest does not match the dataset")

    validate_training_split_plan(request["split_plan"], dataset, limits)
    expected_task = training_task_from_capability(
        capability,
        dataset["task"]["model_output"],
        dataset["task"]["model_microns_per_pixel"],
        limits["report"]["max_text_bytes"],
        annotation_class_map=dataset["task"]["annotation_class_map"],
    )
    if dataset["task"] != expected_task:
        raise ValueError("training dataset task does not match the capability")
    validate_training_field_profile_for_capability(
        request["field_profile"], capability, limits["report"]["max_text_bytes"]
    )
    validate_training_augmentation_selection(
        request["augmentation"], capability, limits["report"]["max_text_bytes"]
    )
    validate_training_settings_for_capability(
        request["trainer_settings"], capability, limits["report"]["max_text_bytes"]
    )
    return request


def _enum(value: Any, choices: set[str], path: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"unsupported {path}")
    return value
