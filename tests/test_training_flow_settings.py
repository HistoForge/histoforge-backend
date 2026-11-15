"""Flow applicability is capability-owned and frozen evidence is catalog-free."""

import pytest
from histoforge.cloud.training.contract import (
    validate_model_training_capability,
    validate_training_settings_for_capability,
)
from histoforge.cloud.training.runs.report import validate_training_report
from tests.test_training_capabilities import _capability, _entry, _registry
from tests.test_training_contract import classified_training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report


@pytest.mark.parametrize("options", [[], None, ["pixel", "pixel"], ["other"], [0]])
def test_capability_rejects_malformed_flow_options(options):
    capability = _capability()
    capability["flow_loss_weighting_options"] = options
    with pytest.raises(ValueError):
        validate_model_training_capability(capability, 4096)


def test_capability_requires_an_advertised_default_without_backfill():
    capability = _capability()
    capability["flow_loss_weighting_options"] = ["equal_instance"]
    with pytest.raises(ValueError, match="advertised"):
        validate_model_training_capability(capability, 4096)
    capability["default_training_settings"]["flow_loss_weighting"] = "equal_instance"
    assert validate_model_training_capability(capability, 4096) == capability
    settings = {
        **capability["default_training_settings"],
        "flow_loss_weighting": "pixel",
    }
    with pytest.raises(ValueError, match="advertised"):
        validate_training_settings_for_capability(settings, capability, 4096)


def test_hover_and_dense_do_not_accept_a_cellpose_option():
    capability = _capability()
    with pytest.raises(ValueError, match="only for a Cellpose head"):
        _entry(capability, "dinov3-adapter-hovernet")
    capability.pop("flow_loss_weighting_options")
    capability["default_training_settings"].pop("flow_loss_weighting")
    _registry(_entry(capability, "dinov3-adapter-hovernet"))
    for output_type in ("instance_segmentation", "dense_classification"):
        capability["output_type"] = output_type
        with pytest.raises(ValueError, match="not advertised"):
            validate_training_settings_for_capability(
                {
                    **capability["default_training_settings"],
                    "flow_loss_weighting": "pixel",
                },
                capability,
                4096,
            )
    capability["flow_loss_weighting_options"] = ["pixel"]
    with pytest.raises(ValueError, match="dense classification"):
        validate_model_training_capability(capability, 4096)


@pytest.mark.parametrize("flow_mode", ["pixel", "equal_instance"])
def test_classified_report_freezes_both_weighting_and_matching_flow_without_catalog(
    flow_mode,
):
    report = queued_report(classified_training_model_output())
    report["trainer_settings"]["flow_loss_weighting"] = flow_mode
    objective = report["trainer_configuration"]["objective"]["parameters"]
    objective["flow_loss_weighting"] = flow_mode
    assert objective["weight_tversky_by_class"] is True
    assert validate_training_report(report, training_api_limits()) == report
    report["trainer_settings"].pop("flow_loss_weighting")
    with pytest.raises(ValueError, match="differs from trainer_settings"):
        validate_training_report(report, training_api_limits())


@pytest.mark.parametrize(
    "defect", ["mismatch", "missing_objective", "ce_only", "old_profile", "old_weights"]
)
def test_report_rejects_conflicting_or_superseded_objective_evidence(defect):
    report = queued_report(classified_training_model_output())
    component = report["trainer_configuration"]["objective"]
    options = component["parameters"]
    if defect == "mismatch":
        options["flow_loss_weighting"] = "equal_instance"
    elif defect == "missing_objective":
        options.pop("flow_loss_weighting")
    elif defect == "ce_only":
        options["weight_tversky_by_class"] = False
    elif defect == "old_profile":
        component["profile_id"] = "managed-segmentation-objective-v1"
    else:
        options["class_weighting"]["policy"] = "sqrt-median-frequency-ce-v1"
    with pytest.raises(ValueError):
        validate_training_report(report, training_api_limits())


@pytest.mark.parametrize("head", [None, [], {}, "other"])
def test_report_rejects_malformed_head_evidence(head):
    report = queued_report(classified_training_model_output())
    report["trainer_configuration"]["objective"]["parameters"]["head_type"] = head
    with pytest.raises(ValueError, match="head type is not supported"):
        validate_training_report(report, training_api_limits())
