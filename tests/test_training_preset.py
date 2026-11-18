"""Executable numeric preset and bounded report evidence, without deployment."""

from __future__ import annotations

import copy
import subprocess
import sys

import numpy as np
import pytest
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.contract import (
    training_task_from_capability,
    validate_training_settings_for_capability,
)
from histoforge.cloud.training.profiles import (
    full_coverage_sampler_profile,
    validate_trainer_profile,
)
from histoforge.cloud.training.runs.report import validate_training_report
from histoforge.cloud.training.schedule import managed_learning_rate_scales
from histoforge.cloud.training.trainers.instance.configuration import (
    resolve_instance_configuration,
)
from tests.test_training_capabilities import _capability, _profile
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_state import service_limits
from tests.test_training_report import queued_report


def _model_output(classified: bool) -> dict:
    if not classified:
        return {"type": "instance_segmentation", "classes": ["Cell"]}
    return {
        "type": "instance_segmentation_and_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Cell"},
            {"channel": 2, "role": "training_only", "pathClass": "Other"},
        ],
    }


def initial_profile(architecture="dinov3-adapter-cellpose", *, classified=False):
    capability = _capability()
    if classified:
        capability["output_type"] = "instance_segmentation_and_classification"
    if architecture == "dinov3-adapter-hovernet":
        capability.pop("flow_loss_weighting_options")
        capability["default_training_settings"].pop("flow_loss_weighting")
    supplied = _profile()
    profile = initial_managed_trainer_profile(
        architecture,
        capability,
        augmentation_profiles=supplied["augmentation_profiles"],
        gpu_resources=supplied["gpu_resources"],
        trainer_execution_seconds=supplied["trainer_execution_seconds"],
    )
    return capability, profile


@pytest.mark.parametrize(
    "architecture,head",
    [
        ("dinov3-adapter-cellpose", "flow"),
        ("dinov3-adapter-hovernet", "hovernet"),
    ],
)
@pytest.mark.parametrize("classified", [False, True])
def test_initial_profile_resolves_task_and_preserves_supplied_controls(
    architecture, head, classified
):
    capability, profile = initial_profile(architecture, classified=classified)
    supplied = _profile()
    assert capability["default_training_settings"]["total_epochs"] == 80
    for name in ("augmentation_profiles", "gpu_resources", "trainer_execution_seconds"):
        assert profile[name] == supplied[name]
    assert capability["default_field_profile"]["side_model_px"] == 384
    components = profile["trainer_components"]
    assert components["sampler"] == full_coverage_sampler_profile()
    objective = components["objective"]["parameters"]
    assert objective["head_type"] == head
    assert "class_count" not in objective
    assert objective["classification_objective"] == "cross_entropy+tversky"
    assert objective["classification_loss_weight"] == 1.0
    assert objective["use_uncertainty_weighting"] is classified
    assert objective["weight_tversky_by_class"] is (classified and head == "flow")
    assert objective["initial_log_variance"] == 0.0
    assert objective["tversky_alpha"] == 0.3
    assert objective["tversky_gamma"] == 1.33
    assert objective.get("flow_loss_weighting") == ("pixel" if head == "flow" else None)
    assert objective["max_gradient_norm"] == 1.0
    assert objective["gradient_norm_type"] == 2.0
    assert objective["gradient_clip_parameters"] == "model-only"
    assert objective["precision"] == "float32-no-autocast"
    weighting = objective["class_weighting"]
    assert "channels" not in weighting
    assert weighting["policy"] == (
        "sqrt-median-frequency-class-v1" if classified else "none"
    )
    if classified:
        assert weighting["count_floor"] == 1
        assert weighting["dtype"] == "float32"
        assert weighting["channel_identity"] == "explicit-model-output-channel"
    preprocessing = components["preprocessing"]
    assert (
        preprocessing["profile_id"]
        == capability["default_field_profile"]["preprocessing"]
    )
    assert preprocessing["parameters"]["percentiles"] == [1.0, 99.0]
    assert preprocessing["parameters"]["clip"] is False
    policy = profile["metric_policy"]
    assert policy["primary_validation_series_id"] == (
        "validation-object-macro-f1.v1"
        if classified
        else "validation-instance-msa.pooled.v1"
    )
    assert policy["primary_selection_direction"] == "maximize"
    assert [series["series_id"] for series in policy["series"]] == [
        "training-loss.v1",
        "validation-instance-ap50.pooled.v1",
        "validation-instance-msa.pooled.v1",
        *(["validation-object-macro-f1.v1"] if classified else []),
    ]
    assert (
        validate_trainer_profile(
            profile, capability, training_api_limits(), service_limits()
        )
        == profile
    )


def test_profile_returns_detached_supplied_values_and_no_numeric_field_defaults():
    capability = _capability()
    supplied = _profile()
    original_capability = copy.deepcopy(capability)
    original_supplied = copy.deepcopy(supplied)
    profile = initial_managed_trainer_profile(
        "dinov3-adapter-cellpose",
        capability,
        augmentation_profiles=supplied["augmentation_profiles"],
        gpu_resources=supplied["gpu_resources"],
        trainer_execution_seconds=supplied["trainer_execution_seconds"],
    )
    profile["gpu_resources"]["test-gpu"]["cpu"] = 999
    profile["augmentation_profiles"]["none-v1"]["new"] = {}
    assert supplied == original_supplied
    assert capability == original_capability
    assert "field_profile" not in profile


def test_selected_model_head_count_is_bound_only_in_resolved_configuration():
    capability, profile = initial_profile(classified=True)
    selection = {"profile_id": "none-v1", "enabled_operator_ids": []}
    three_channels = _model_output(True)
    four_channels = copy.deepcopy(three_channels)
    four_channels["classChannels"].append(
        {"channel": 3, "role": "emitted", "pathClass": "Leydig cell"}
    )
    original = copy.deepcopy(profile)

    assert (
        resolve_instance_configuration(
            profile,
            capability,
            selection,
            three_channels,
            capability["default_training_settings"],
        )["objective"]["parameters"]["class_count"]
        == 3
    )
    assert (
        resolve_instance_configuration(
            profile,
            capability,
            selection,
            four_channels,
            capability["default_training_settings"],
        )["objective"]["parameters"]["class_count"]
        == 4
    )
    assert profile == original


def test_registry_profile_rejects_a_frozen_model_class_count():
    capability, profile = initial_profile(classified=True)
    profile["trainer_components"]["objective"]["parameters"]["class_count"] = 3

    with pytest.raises(ValueError, match="must not bind"):
        validate_trainer_profile(
            profile, capability, training_api_limits(), service_limits()
        )


@pytest.mark.parametrize("classified", [False, True])
def test_initial_components_fit_existing_complete_queued_report(classified):
    capability, profile = initial_profile(classified=classified)
    report = queued_report()
    selection = {"profile_id": "none-v1", "enabled_operator_ids": []}
    settings = capability["default_training_settings"]
    report.update(
        source_model_output=_model_output(classified),
        task=training_task_from_capability(
            capability, _model_output(classified), 0.5, 4096
        ),
        total_epochs=settings["total_epochs"],
        trainer_settings=settings,
        field_profile=capability["default_field_profile"],
        augmentation=selection,
        metric_policy=profile["metric_policy"],
        trainer_configuration=resolve_instance_configuration(
            profile,
            capability,
            selection,
            _model_output(classified),
            settings,
        ),
    )
    target_counts = {"Cell": 20, "Other": 20} if classified else {"Cell": 20}
    for key in ("training_counts", "validation_counts"):
        report[key]["target_object_counts"] = target_counts
    if classified:
        # Admission supplies these dataset-dependent values before revision one.
        report["trainer_configuration"]["objective"]["parameters"]["class_weighting"][
            "channels"
        ] = [
            {"channel": 0, "role": "background", "raw_count": 10, "weight": 1.0},
            {
                "channel": 1,
                "role": "emitted",
                "path_class": "Cell",
                "raw_count": 10,
                "weight": 1.0,
            },
            {
                "channel": 2,
                "role": "training_only",
                "path_class": "Other",
                "raw_count": 10,
                "weight": 1.0,
            },
        ]
    assert validate_training_report(report, training_api_limits()) == report


def test_schedule_is_the_exact_source_80_epoch_sequence():
    capability, profile = initial_profile()
    scheduler = resolve_instance_configuration(
        profile,
        capability,
        {"profile_id": "none-v1", "enabled_operator_ids": []},
        _model_output(False),
        capability["default_training_settings"],
    )["scheduler"]
    observed = managed_learning_rate_scales(scheduler)
    expected = np.concatenate(
        (np.linspace(0, 1, 10), 0.5 * (1 + np.cos(np.pi * np.arange(70) / 70)))
    )
    np.testing.assert_array_equal(observed, expected)
    assert len(observed) == 80
    assert observed[0] == 0.0
    assert observed[9] == observed[10] == 1.0
    assert 0 < observed[-1] < observed[-2]
    parameters = scheduler["parameters"]
    assert parameters["epoch_indexing"] == "zero-based-before-epoch"
    assert parameters["group_scale_rule"] == "multiply-initial-learning-rate"
    assert parameters["zero_rate_behavior"] == "adamw-step-updates-moments"


@pytest.mark.parametrize(
    ("total_epochs", "warmup_epochs"),
    [(1, 0), (8, 1), (40, 5), (80, 10), (160, 10)],
)
def test_auto_warmup_resolves_the_advertised_schedule(total_epochs, warmup_epochs):
    capability, profile = initial_profile()
    settings = {
        **capability["default_training_settings"],
        "total_epochs": total_epochs,
        "warmup_epochs": warmup_epochs,
    }
    validate_training_settings_for_capability(settings, capability, 4_096)
    scheduler = resolve_instance_configuration(
        profile,
        capability,
        {"profile_id": "none-v1", "enabled_operator_ids": []},
        _model_output(False),
        settings,
    )["scheduler"]
    parameters = scheduler["parameters"]
    assert parameters["total_epochs"] == total_epochs
    assert parameters["warmup_epochs"] == warmup_epochs
    assert parameters["cosine_epochs"] == total_epochs - warmup_epochs
    assert len(managed_learning_rate_scales(scheduler)) == total_epochs


def test_custom_zero_warmup_starts_cosine_at_peak_rate():
    capability, profile = initial_profile()
    settings = {
        **capability["default_training_settings"],
        "total_epochs": 3,
        "warmup_mode": "custom",
        "warmup_epochs": 0,
    }
    scheduler = resolve_instance_configuration(
        profile,
        capability,
        {"profile_id": "none-v1", "enabled_operator_ids": []},
        _model_output(False),
        settings,
    )["scheduler"]
    observed = managed_learning_rate_scales(scheduler)
    assert observed[0] == 1.0
    assert len(observed) == 3


def test_warmup_is_rejected_instead_of_silently_clamped():
    capability, _ = initial_profile()
    invalid = {
        **capability["default_training_settings"],
        "total_epochs": 8,
        "warmup_mode": "custom",
        "warmup_epochs": 8,
    }
    with pytest.raises(ValueError, match="less than total_epochs"):
        validate_training_settings_for_capability(invalid, capability, 4_096)

    wrong_auto = {
        **capability["default_training_settings"],
        "total_epochs": 40,
        "warmup_epochs": 4,
    }
    with pytest.raises(ValueError, match="automatic warmup"):
        validate_training_settings_for_capability(wrong_auto, capability, 4_096)


@pytest.mark.parametrize(
    "key,value",
    [
        ("total_epochs", 81),
        ("cosine_denominator", 69),
        ("warmup_epochs", -1),
        ("epoch_indexing", "after-epoch"),
        ("group_scale_rule", "replace-learning-rate"),
        ("zero_rate_behavior", "skip-optimizer-step"),
    ],
)
def test_schedule_rejects_conflicting_recorded_execution_rules(key, value):
    capability, profile = initial_profile()
    scheduler = resolve_instance_configuration(
        profile,
        capability,
        {"profile_id": "none-v1", "enabled_operator_ids": []},
        _model_output(False),
        capability["default_training_settings"],
    )["scheduler"]
    scheduler["parameters"][key] = value
    with pytest.raises(ValueError, match="scheduler implementation"):
        managed_learning_rate_scales(scheduler)


def test_profile_construction_imports_neither_torch_nor_numpy():
    script = """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise RuntimeError('training import during preset construction')

sys.meta_path.insert(0, BlockTrainingImports())
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.architectures.dinov3_optimization import (
    build_dinov3_optimizer,
)

profile = initial_managed_trainer_profile(
    'dinov3-adapter-cellpose',
    {'output_type': 'instance_segmentation',
     'flow_loss_weighting_options': ['pixel', 'equal_instance'],
     'default_training_settings': {'flow_loss_weighting': 'pixel'},
     'default_field_profile': {'preprocessing': 'normalize99-real-image-v1'}},
    augmentation_profiles={'none-v1': {}},
    gpu_resources={'test': {'gpu': 'L4', 'cpu': 4, 'memory_mib': 16384,
                            'max_validation_assignment_matrix_entries': 1000000}},
    trainer_execution_seconds=3600,
)
assert 'total_epochs' not in profile
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_initial_preset_does_not_claim_other_architectures_or_preprocessing():
    capability = _capability()
    supplied = _profile()
    kwargs = {
        name: supplied[name]
        for name in (
            "augmentation_profiles",
            "gpu_resources",
            "trainer_execution_seconds",
        )
    }
    with pytest.raises(ValueError, match="DINOv3 instance adapter"):
        initial_managed_trainer_profile("other", capability, **kwargs)
    capability["default_field_profile"]["preprocessing"] = "other-v1"
    with pytest.raises(ValueError, match="real-field normalization"):
        initial_managed_trainer_profile("dinov3-adapter-cellpose", capability, **kwargs)
