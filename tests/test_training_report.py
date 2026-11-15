from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.modules.schema import (
    model_output_class_channels,
    model_output_target_classes,
)
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.contract import training_task_from_capability
from histoforge.cloud.training.runs.report import (
    training_report_digest,
    validate_training_report,
)
from histoforge.cloud.training.trainers.instance.configuration import (
    resolve_instance_configuration,
)
from tests.test_training_contract import (
    classified_training_model_output,
    training_capability,
    training_model_output,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request

STARTING_WEIGHT_BYTES = b"starting weights"
STARTING_WEIGHT_DIGEST = hashlib.sha256(STARTING_WEIGHT_BYTES).hexdigest()


def test_shared_module_training_report_fixture() -> None:
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "contracts"
            / "module-training-report-v1.json"
        ).read_text(encoding="utf-8")
    )
    response = fixture["response"]
    report = validate_training_report(response["report"], training_api_limits())

    assert report == response["report"]
    assert (
        training_report_digest(report, training_api_limits())
        == fixture["report_digest"]
    )
    assert response["report_digest"] == fixture["report_digest"]
    assert canonical_digest(response) == fixture["response_sha256"]


def queued_report(model_output: dict | None = None) -> dict:
    request = run_request()
    model_output = model_output or training_model_output()
    capability = copy.deepcopy(training_capability())
    capability["output_type"] = model_output["type"]
    profile = initial_managed_trainer_profile(
        "dinov3-adapter-cellpose",
        capability,
        augmentation_profiles={
            "he-v1": {
                "rotate90-v1": {
                    "implementation": "quarter-turn-v1",
                    "quarter_turn_probabilities": [0.25, 0.25, 0.25, 0.25],
                },
                "flip-v1": {
                    "implementation": "axis-flips-v1",
                    "horizontal_probability": 0.5,
                    "vertical_probability": 0.5,
                },
            }
        },
        gpu_resources={},
        trainer_execution_seconds=3600,
    )
    settings = {
        **capability["default_training_settings"],
        "total_epochs": 3,
        "warmup_mode": "custom",
        "warmup_epochs": 1,
    }
    request["trainer_settings"] = settings
    task = training_task_from_capability(capability, model_output, 0.5, 4096)
    configuration = resolve_instance_configuration(
        profile,
        capability,
        request["augmentation"],
        task["model_output"],
        settings,
    )
    channels = model_output_class_channels(model_output)
    if channels:
        configuration["objective"]["parameters"]["class_weighting"]["channels"] = [
            {
                "channel": channel["channel"],
                "role": channel["role"],
                **(
                    {"path_class": channel["pathClass"]}
                    if "pathClass" in channel
                    else {}
                ),
                "raw_count": 10,
                "weight": 1.0,
            }
            for channel in channels
        ]
    target_classes = model_output_target_classes(model_output)

    return {
        "schema_version": 1,
        "class_initialization": "use",
        "source_model_output": copy.deepcopy(model_output),
        "source_model_microns_per_pixel": 0.5,
        "run_id": "44444444-4444-4444-8444-444444444444",
        "state": "QUEUED",
        "dataset_digest": request["dataset_digest"],
        "dataset_image_count": 2,
        "split_plan": request["split_plan"],
        "source_module": request["source_module"],
        "selected_model_id": request["selected_model_id"],
        "starting_weight_digest": STARTING_WEIGHT_DIGEST,
        "result_form": request["result_form"],
        "result_display_name": request["result_display_name"],
        "result_inference_gpu": request["result_inference_gpu"],
        "task": task,
        "trainer_build": trainer_build(),
        "training_seed": request["training_seed"],
        "trainer_settings": request["trainer_settings"],
        "total_epochs": 3,
        "field_profile": capability["default_field_profile"],
        "min_instance_pixels": request["min_instance_pixels"],
        "augmentation": request["augmentation"],
        "trainer_configuration": configuration,
        "training_counts": {
            "image_count": 1,
            "roi_count": 1,
            "negative_roi_count": 0,
            "target_object_counts": {path_class: 50 for path_class in target_classes},
        },
        "validation_counts": {
            "image_count": 1,
            "roi_count": 1,
            "negative_roi_count": 0,
            "target_object_counts": {path_class: 20 for path_class in target_classes},
        },
        "metric_policy": {
            "series": [
                {
                    "series_id": "training-loss.v1",
                    "display_name": "Training loss",
                    "partition": "training",
                    "value_domain": "real",
                    "definition": "Equal-field mean training objective.",
                },
                {
                    "series_id": "validation-instance-msa.v1",
                    "display_name": "Validation mSA",
                    "partition": "validation",
                    "value_domain": "unit_interval",
                    "definition": "Mean segmentation accuracy over IoU thresholds.",
                },
            ],
            "primary_validation_series_id": "validation-instance-msa.v1",
            "primary_selection_direction": "maximize",
        },
        "epochs": [],
        "unavailable_metrics": [],
        "messages": [],
    }


def succeeded_report() -> dict:
    report = queued_report()
    report["state"] = "SUCCEEDED"
    report["epochs"] = [
        {
            "epoch": 1,
            "elapsed_seconds": 10,
            "metrics": [
                {"series_id": "training-loss.v1", "value": 0.8},
                {"series_id": "validation-instance-msa.v1", "value": 0.6},
            ],
        },
        {
            "epoch": 2,
            "elapsed_seconds": 20,
            "metrics": [
                {"series_id": "training-loss.v1", "value": 0.5},
                {"series_id": "validation-instance-msa.v1", "value": 0.7},
            ],
        },
        {
            "epoch": 3,
            "elapsed_seconds": 30,
            "metrics": [
                {"series_id": "training-loss.v1", "value": 0.4},
                {"series_id": "validation-instance-msa.v1", "value": 0.7},
            ],
        },
    ]
    report["model_selection"] = {
        "type": "best-validation",
        "series_id": "validation-instance-msa.v1",
        "value": 0.7,
        "epoch": 2,
        "selected_weight_digest": "d" * 64,
    }
    return report


def test_report_round_trip_and_digest() -> None:
    report = queued_report()
    assert validate_training_report(report, training_api_limits()) == report
    assert training_report_digest(
        report, training_api_limits()
    ) == training_report_digest(copy.deepcopy(report), training_api_limits())

    report["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        validate_training_report(report, training_api_limits())


def test_success_selects_earliest_best_primary_observation() -> None:
    report = succeeded_report()
    assert validate_training_report(report, training_api_limits()) == report

    report["model_selection"]["epoch"] = 3
    with pytest.raises(ValueError, match="does not match report observations"):
        validate_training_report(report, training_api_limits())


def test_report_state_controls_failure_selection_and_final_epoch() -> None:
    report = succeeded_report()
    report["epochs"].pop()
    with pytest.raises(ValueError, match="requires the final epoch record"):
        validate_training_report(report, training_api_limits())

    failed = queued_report()
    failed["state"] = "FAILED"
    with pytest.raises(ValueError, match="requires failure"):
        validate_training_report(failed, training_api_limits())
    failed["failure"] = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training failed",
    }
    assert validate_training_report(failed, training_api_limits()) == failed

    queued = queued_report()
    queued["epochs"] = succeeded_report()["epochs"][:1]
    with pytest.raises(ValueError, match="QUEUED report cannot contain"):
        validate_training_report(queued, training_api_limits())


def test_failed_report_bounds_public_resource_identifier() -> None:
    report = queued_report()
    report["state"] = "FAILED"
    report["failure"] = {
        "schema_version": 1,
        "code": "RUNNER_FAILED",
        "message": "training failed",
        "resource_id": "/secret/path\nname",
    }
    with pytest.raises(ValueError, match="failure.resource_id.*control characters"):
        validate_training_report(report, training_api_limits())

    report["failure"]["resource_id"] = "a" * 4097
    with pytest.raises(ValueError, match="failure.resource_id.*max_text_bytes"):
        validate_training_report(report, training_api_limits())


def test_no_validation_uses_final_weights_and_no_validation_values() -> None:
    report = queued_report()
    report["split_plan"] = {"type": "none"}
    del report["validation_counts"]
    report["state"] = "SUCCEEDED"
    report["epochs"] = [
        {
            "epoch": 3,
            "elapsed_seconds": 30,
            "metrics": [{"series_id": "training-loss.v1", "value": 0.4}],
        }
    ]
    report["model_selection"] = {
        "type": "final-weights",
        "reason": "no_validation",
        "epoch": 3,
        "selected_weight_digest": "d" * 64,
    }
    assert validate_training_report(report, training_api_limits()) == report

    report["epochs"][0]["metrics"].append(
        {"series_id": "validation-instance-msa.v1", "value": 0.7}
    )
    with pytest.raises(ValueError, match="cannot contain validation values"):
        validate_training_report(report, training_api_limits())


def test_report_rejects_metric_order_and_unbounded_component_data() -> None:
    report = succeeded_report()
    report["epochs"][0]["metrics"].reverse()
    with pytest.raises(ValueError, match="policy order"):
        validate_training_report(report, training_api_limits())

    limits = training_api_limits()
    limits["report"]["max_component_collection_entries"] = 1
    oversized = queued_report()
    oversized["trainer_configuration"]["optimizer"]["parameters"]["weight_decay"] = 0.01
    with pytest.raises(ValueError, match="collection-entry limit"):
        validate_training_report(oversized, limits)


def test_metric_domains_are_explicit_and_unit_interval_values_are_bounded() -> None:
    report = succeeded_report()
    report["epochs"][0]["metrics"][0]["value"] = -0.125
    assert validate_training_report(report, training_api_limits()) == report

    report["epochs"][0]["metrics"][1]["value"] = 1.0001
    with pytest.raises(ValueError, match="unit-interval metric values"):
        validate_training_report(report, training_api_limits())

    report = succeeded_report()
    report["metric_policy"]["series"][1]["value_domain"] = "percentage"
    with pytest.raises(ValueError, match="value_domain"):
        validate_training_report(report, training_api_limits())


def test_report_binds_augmentation_presentation_to_the_frozen_selection() -> None:
    report = queued_report()
    parameters = report["trainer_configuration"]["augmentation"]["parameters"]

    parameters["base_profile_id"] = "pas-v1"
    with pytest.raises(ValueError, match="base profile differs"):
        validate_training_report(report, training_api_limits())

    report = queued_report()
    parameters = report["trainer_configuration"]["augmentation"]["parameters"]
    parameters["enabled_operators"].pop()
    with pytest.raises(ValueError, match="enabled augmentation operators differ"):
        validate_training_report(report, training_api_limits())


def test_report_requires_preprocessing_identity_and_target_count_shape() -> None:
    report = queued_report()
    report["trainer_configuration"]["preprocessing"]["profile_id"] = "other-v1"
    with pytest.raises(ValueError, match="differs from field_profile"):
        validate_training_report(report, training_api_limits())

    report = queued_report()
    report["training_counts"]["target_object_counts"] = [1, 2]
    with pytest.raises(ValueError, match="must be an object"):
        validate_training_report(report, training_api_limits())


def test_terminal_report_cannot_mark_an_observed_metric_unavailable() -> None:
    report = succeeded_report()
    report["unavailable_metrics"] = [
        {
            "series_id": "validation-instance-msa.v1",
            "reason": "No valid validation regions.",
        }
    ]

    with pytest.raises(ValueError, match="observed metric cannot also be unavailable"):
        validate_training_report(report, training_api_limits())


def test_unavailable_primary_selection_requires_a_reason() -> None:
    report = succeeded_report()
    report["epochs"] = [
        {
            "epoch": epoch["epoch"],
            "elapsed_seconds": epoch["elapsed_seconds"],
            "metrics": [epoch["metrics"][0]],
        }
        for epoch in report["epochs"]
    ]
    report["model_selection"] = {
        "type": "final-weights",
        "reason": "primary_metric_unavailable",
        "epoch": 3,
        "selected_weight_digest": "d" * 64,
    }

    with pytest.raises(
        ValueError, match="unavailable primary metric requires a reason"
    ):
        validate_training_report(report, training_api_limits())

    report["unavailable_metrics"] = [
        {
            "series_id": "validation-instance-msa.v1",
            "reason": "No validation ROI produced a defined score.",
        }
    ]
    assert validate_training_report(report, training_api_limits()) == report


def test_finalizing_report_can_select_final_weights_before_unavailable_reason() -> None:
    report = succeeded_report()
    report["state"] = "FINALIZING"
    report["epochs"] = [
        {
            "epoch": epoch["epoch"],
            "elapsed_seconds": epoch["elapsed_seconds"],
            "metrics": [epoch["metrics"][0]],
        }
        for epoch in report["epochs"]
    ]
    report["model_selection"] = {
        "type": "final-weights",
        "reason": "primary_metric_unavailable",
        "epoch": 3,
        "selected_weight_digest": "d" * 64,
    }

    assert validate_training_report(report, training_api_limits()) == report


def test_success_explains_every_metric_never_observed() -> None:
    report = succeeded_report()
    report["metric_policy"]["series"].append(
        {
            "series_id": "validation-instance-ap50.v1",
            "display_name": "Validation AP50",
            "partition": "validation",
            "value_domain": "unit_interval",
            "definition": "Instance score at IoU 0.50.",
        }
    )

    with pytest.raises(ValueError, match="observe or explain every applicable metric"):
        validate_training_report(report, training_api_limits())

    report["unavailable_metrics"] = [
        {
            "series_id": "validation-instance-ap50.v1",
            "reason": "No validation ROI produced a defined score.",
        }
    ]
    assert validate_training_report(report, training_api_limits()) == report


@pytest.mark.parametrize(
    ("component", "field", "value"),
    [
        ("optimizer", "api_key", "secret"),
        ("scheduler", "environment_values", {"TOKEN": "secret"}),
        ("sampler", "source_filename", "/private/source.svs"),
        ("objective", "raw_exception", "secret traceback"),
        ("augmentation", "runtime", {"nested": {"secret": "value"}}),
        ("preprocessing", "credentials", ["secret"]),
    ],
)
def test_report_rejects_unknown_component_fields(
    component: str, field: str, value: object
) -> None:
    report = queued_report()
    report["trainer_configuration"][component]["parameters"][field] = value

    with pytest.raises(ValueError, match="unknown fields|not supported"):
        validate_training_report(report, training_api_limits())


@pytest.mark.parametrize(
    "component",
    [
        "optimizer",
        "scheduler",
        "sampler",
        "objective",
        "augmentation",
        "preprocessing",
    ],
)
def test_report_rejects_unknown_component_profile_ids(component: str) -> None:
    report = queued_report()
    report["trainer_configuration"][component]["profile_id"] = "unknown-v1"

    with pytest.raises(ValueError):
        validate_training_report(report, training_api_limits())


@pytest.mark.parametrize(
    ("component", "field", "value"),
    [
        ("optimizer", "learning_rate", True),
        ("scheduler", "total_epochs", True),
        ("objective", "class_count", False),
        ("objective", "initial_log_variance", False),
        ("preprocessing", "clip", 0),
        ("preprocessing", "percentiles", [True, 99.0]),
    ],
)
def test_report_rejects_json_boolean_number_confusion(
    component: str, field: str, value: object
) -> None:
    report = queued_report()
    report["trainer_configuration"][component]["parameters"][field] = value

    with pytest.raises(ValueError):
        validate_training_report(report, training_api_limits())


def test_report_rejects_invalid_class_weight_channel_and_formula() -> None:
    output = classified_training_model_output()
    report = queued_report(output)
    weighting = report["trainer_configuration"]["objective"]["parameters"][
        "class_weighting"
    ]
    weighting["channels"][0]["channel"] = False
    with pytest.raises(ValueError, match="channel"):
        validate_training_report(report, training_api_limits())

    report = queued_report(output)
    weighting = report["trainer_configuration"]["objective"]["parameters"][
        "class_weighting"
    ]
    weighting["channels"][0]["weight"] = 2.0
    with pytest.raises(ValueError, match="recorded counts"):
        validate_training_report(report, training_api_limits())


def test_report_rejects_control_text_in_component_presentation() -> None:
    report = queued_report()
    report["trainer_configuration"]["augmentation"]["parameters"][
        "base_profile_display_name"
    ] = "H&E\nsecret"

    with pytest.raises(ValueError, match="control characters"):
        validate_training_report(report, training_api_limits())
