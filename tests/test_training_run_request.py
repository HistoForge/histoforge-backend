from __future__ import annotations

import copy

import pytest
from histoforge.cloud.training.contract import training_task_from_capability
from histoforge.cloud.training.datasets.manifest import build_training_dataset_manifest
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
    validate_training_run_request_against_manifest_capability,
)
from tests.test_training_contract import training_capability, training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_split import two_image_manifest


def run_request() -> dict:
    trainer_settings = {
        **training_capability()["default_training_settings"],
        "total_epochs": 3,
        "warmup_mode": "custom",
        "warmup_epochs": 1,
    }
    return {
        "schema_version": 1,
        "class_initialization": "use",
        "dataset_id": "33333333-3333-4333-8333-333333333333",
        "dataset_digest": two_image_manifest()["dataset_digest"],
        "source_module": {
            "module_ref": "histoforge/leydig",
            "resolved_commit_sha": "1" * 40,
            "recipe_digest": "2" * 64,
            "content_digest": "3" * 64,
        },
        "selected_model_id": "leydig_nuclear",
        "result_form": "complete_source_workflow",
        "result_display_name": "Fine-tuned Leydig workflow",
        "result_inference_gpu": "A100",
        "split_plan": {
            "type": "split",
            "unit": "image",
            "assignment_source": "generated",
            "requested_validation_percent": 20,
            "units": [
                {"unit_index": 0, "partition": "training"},
                {"unit_index": 1, "partition": "validation"},
            ],
        },
        "field_profile": training_capability()["default_field_profile"],
        "min_instance_pixels": 15,
        "augmentation": {
            "profile_id": "he-v1",
            "enabled_operator_ids": ["rotate90-v1", "flip-v1"],
        },
        "trainer_settings": trainer_settings,
        "training_seed": 1234,
    }


def test_request_round_trip_and_digest_are_external_state_independent() -> None:
    request = run_request()
    validated = validate_training_run_request(request, training_api_limits())

    assert validated == request
    assert training_run_request_digest(request, training_api_limits()) == (
        training_run_request_digest(validated, training_api_limits())
    )


def test_request_binds_dataset_task_and_capability_choices() -> None:
    request = run_request()
    manifest = two_image_manifest()
    capability = training_capability()

    assert (
        validate_training_run_request_against_manifest_capability(
            request, manifest, capability, training_api_limits()
        )
        == request
    )

    wrong_digest = copy.deepcopy(request)
    wrong_digest["dataset_digest"] = "f" * 64
    with pytest.raises(ValueError, match="does not match the dataset"):
        validate_training_run_request_against_manifest_capability(
            wrong_digest, manifest, capability, training_api_limits()
        )

    wrong_settings = copy.deepcopy(request)
    wrong_settings["trainer_settings"]["training_batch_size"] = 9
    with pytest.raises(ValueError, match="outside its advertised range"):
        validate_training_run_request_against_manifest_capability(
            wrong_settings, manifest, capability, training_api_limits()
        )


def test_request_rejects_capability_or_fixed_field_drift() -> None:
    request = run_request()
    manifest = two_image_manifest()
    capability = training_capability()

    changed_task = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "dataset_digest"
    }
    changed_task["task"] = training_task_from_capability(
        capability, training_model_output(), 0.5, 4096
    )
    changed_task["task"]["supports_negative_rois"] = False
    drifted_manifest = build_training_dataset_manifest(
        changed_task, training_api_limits()
    )
    request_for_drifted = copy.deepcopy(request)
    request_for_drifted["dataset_digest"] = drifted_manifest["dataset_digest"]
    with pytest.raises(ValueError, match="task does not match"):
        validate_training_run_request_against_manifest_capability(
            request_for_drifted,
            drifted_manifest,
            capability,
            training_api_limits(),
        )

    changed_field = copy.deepcopy(request)
    changed_field["field_profile"]["small_roi_context"] = False
    with pytest.raises(ValueError, match="must equal the capability"):
        validate_training_run_request_against_manifest_capability(
            changed_field, manifest, capability, training_api_limits()
        )


def test_request_rejects_invalid_persisted_syntax_without_live_lookup() -> None:
    request = run_request()
    request["augmentation"]["enabled_operator_ids"] = [
        "flip-v1",
        "flip-v1",
    ]
    with pytest.raises(ValueError, match="must be unique"):
        validate_training_run_request(request, training_api_limits())

    request = run_request()
    request["selected_model_id"] = "a" * 129
    with pytest.raises(ValueError, match="1 to 128"):
        validate_training_run_request(request, training_api_limits())

    request = run_request()
    request["min_instance_pixels"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_training_run_request(request, training_api_limits())

    request = run_request()
    request["min_instance_pixels"] = 1.5
    with pytest.raises(ValueError, match="integer"):
        validate_training_run_request(request, training_api_limits())

    request = run_request()
    request["result_inference_gpu"] = "contains spaces"
    with pytest.raises(ValueError, match="visible ASCII"):
        validate_training_run_request(request, training_api_limits())


def test_request_accepts_no_validation_plan_without_external_lookup() -> None:
    request = run_request()
    request["split_plan"] = {"type": "none"}

    assert validate_training_run_request(request, training_api_limits()) == request

    request["split_plan"]["units"] = []
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_training_run_request(request, training_api_limits())
