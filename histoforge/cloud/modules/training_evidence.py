"""Validation of the optional training report owned by one module root."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
    require_sha256,
    sha256_bytes,
)
from histoforge.cloud.modules.files import require_regular_path
from histoforge.cloud.modules.schema import TRAINING_REPORT_NAME
from histoforge.cloud.training.contract import validate_instance_training_settings
from histoforge.cloud.training.runs.report import validate_training_report

_TRAINING_RESULT_ATTESTATION_FIELDS = {
    "run_id",
    "module_content_digest",
    "report_digest",
}


def validate_module_training_report(
    module_dir: Path,
    recipe: Mapping[str, Any],
    resolved_artifacts: Sequence[Mapping[str, Any]],
    training_api_limits: Any,
) -> dict[str, Any] | None:
    """Validate canonical report bytes and bind them to the selected model artifact."""

    declared_digest = recipe.get("trainingReportDigest")
    if declared_digest is None:
        return None

    maximum_bytes = training_api_limits["report"]["max_report_bytes"]
    report_path = require_regular_path(
        module_dir / TRAINING_REPORT_NAME,
        TRAINING_REPORT_NAME,
    )
    with report_path.open("rb") as stream:
        encoded = stream.read(maximum_bytes + 1)
        if len(encoded) > maximum_bytes or stream.read(1):
            raise ValueError("training report exceeds max_report_bytes")
    if sha256_bytes(encoded) != declared_digest:
        raise ValueError("training report digest does not match recipe")
    report = validate_training_report(parse_json(encoded), training_api_limits)
    if canonical_bytes(report) != encoded:
        raise ValueError("training report must use canonical JSON bytes")
    if report["state"] != "SUCCEEDED":
        raise ValueError("module training report must describe a successful run")

    model_id = report["selected_model_id"]
    try:
        model = recipe["models"][model_id]
    except KeyError as error:
        raise ValueError(
            "training report selected_model_id is absent from recipe"
        ) from error
    selected = report["model_selection"]
    artifact = _selected_weight_artifact(resolved_artifacts, model_id)
    if artifact["artifact_source_type"] != "vendored":
        raise ValueError("trained module selected weights must be vendored")
    if artifact["sha256"] != selected["selected_weight_digest"]:
        raise ValueError("training report selected weights do not match module bytes")

    task = report["task"]
    if task["input_kind"] != model["input"]["type"]:
        raise ValueError("training report input kind does not match selected model")
    if task["model_output"] != model["output"]:
        raise ValueError("training report model output does not match selected model")
    if task["model_microns_per_pixel"] != model["input"]["micronsPerPixel"]:
        raise ValueError("training report resolution does not match selected model")
    model_settings = validate_instance_training_settings(
        model["params"], model["output"]
    )
    report_settings = validate_instance_training_settings(report, task["model_output"])
    if report_settings != model_settings:
        raise ValueError(
            "training report minimum instance area does not match selected model"
        )
    if report["result_inference_gpu"] != model["gpu"]:
        raise ValueError(
            "training report result inference GPU does not match selected model"
        )
    model_profile = model["input"].get("fieldProfile")
    report_profile = report["field_profile"]
    if model_profile is not None:
        if report_profile != model_profile:
            raise ValueError(
                "training report field profile does not match selected model"
            )
    elif report_profile["planner"] != "fixed-field-v1":
        raise ValueError("fixed model requires a fixed training field profile")
    else:
        # Imported lazily to keep schema/report parsing independent of the
        # deployed runner registry.
        from histoforge.cloud.architectures.registry import (
            resolved_fixed_field_profile,
        )

        expected_profile = resolved_fixed_field_profile(model)
        if expected_profile is not None and report_profile != expected_profile:
            raise ValueError(
                "training report fixed field profile does not match selected runner"
            )
    return report


def validate_training_result_attestation(value: Any) -> dict[str, str]:
    """Validate the private identity of one run-derived workspace module."""

    if not isinstance(value, dict):
        raise ValueError("training result attestation must be an object")
    if set(value) != _TRAINING_RESULT_ATTESTATION_FIELDS:
        raise ValueError(
            "training result attestation fields mismatch: "
            f"missing={sorted(_TRAINING_RESULT_ATTESTATION_FIELDS - set(value))}, "
            f"unknown={sorted(set(value) - _TRAINING_RESULT_ATTESTATION_FIELDS)}"
        )
    return {
        "run_id": require_canonical_uuid(value["run_id"], "run_id"),
        "module_content_digest": require_sha256(
            value["module_content_digest"], "module_content_digest"
        ),
        "report_digest": require_sha256(value["report_digest"], "report_digest"),
    }


def training_result_is_visible(
    record: Mapping[str, Any],
    run_result_is_visible: Callable[[str], bool],
) -> bool:
    """Apply the one training-state gate for an attested workspace result."""

    attestation = record.get("training_result_attestation")
    if attestation is None:
        return True
    visible = run_result_is_visible(attestation["run_id"])
    if not isinstance(visible, bool):
        raise ValueError("training result visibility owner returned a non-boolean")
    return visible


def _selected_weight_artifact(
    artifacts: Sequence[Mapping[str, Any]], model_id: str
) -> Mapping[str, Any]:
    matches = [
        artifact
        for artifact in artifacts
        if artifact["model_id"] == model_id and artifact["role"] == "weights"
    ]
    if len(matches) != 1:
        raise ValueError("selected model has no unique primary weight artifact")
    return matches[0]
