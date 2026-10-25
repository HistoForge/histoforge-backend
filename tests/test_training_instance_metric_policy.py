"""Class-bound metric catalogs remain ordinary, immutable report evidence."""

import copy
from types import SimpleNamespace

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.training.metric_policy import TRAINING_LOSS_SERIES_ID
from histoforge.cloud.training.runs.report import (
    select_training_weight_epoch,
    validate_trainer_metric_policy,
    validate_training_report,
)
from histoforge.cloud.training.trainers.instance.policy import (
    VALIDATION_AP50_SERIES_ID,
    VALIDATION_MSA_SERIES_ID,
    VALIDATION_OBJECT_MACRO_F1_SERIES_ID,
    object_class_f1_series_id,
    prepare_instance_metric_policy,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_preset import initial_profile
from tests.test_training_report import queued_report


def _output():
    return {
        "type": "instance_segmentation_and_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Cell"},
            {"channel": 2, "role": "training_only", "pathClass": "Other"},
            {"channel": 3, "role": "emitted", "pathClass": "Sertoli cell"},
            {"channel": 4, "role": "emitted", "pathClass": "Leydig cell"},
        ],
    }


def _policy():
    _, profile = initial_profile(classified=True)
    return prepare_instance_metric_policy(
        SimpleNamespace(trainer_profile=profile), _output()
    )


@pytest.mark.parametrize(
    "architecture", ["dinov3-adapter-cellpose", "dinov3-adapter-hovernet"]
)
def test_bound_series_use_explicit_emitted_channels_without_changing_profile(
    architecture,
):
    _, profile = initial_profile(architecture, classified=True)
    original = canonical_bytes(profile)
    output = _output()
    output["classChannels"].reverse()  # Normalization owns numeric display order.
    policy = prepare_instance_metric_policy(
        SimpleNamespace(trainer_profile=profile), output
    )
    assert canonical_bytes(profile) == original
    assert len(profile["metric_policy"]["series"]) == 4
    assert [item["series_id"] for item in policy["series"][4:]] == [
        object_class_f1_series_id(channel) for channel in (1, 3, 4)
    ]
    assert [item["display_name"] for item in policy["series"][4:]] == [
        "Validation object F1: Cell",
        "Validation object F1: Sertoli cell",
        "Validation object F1: Leydig cell",
    ]
    assert validate_trainer_metric_policy(policy, training_api_limits()) == policy
    policy["series"][0]["definition"] = "changed after resolution"
    assert canonical_bytes(profile) == original


def test_bound_catalog_uses_existing_report_limits():
    limits = training_api_limits()
    limits["report"]["max_metric_series"] = 6
    with pytest.raises(ValueError, match="limit"):
        validate_trainer_metric_policy(_policy(), limits)


def test_sparse_class_series_and_zero_gt_do_not_add_checkpoint_safeguards():
    report = queued_report(_output())
    report["metric_policy"] = _policy()
    policy = report["metric_policy"]
    # A researcher supplied an all-negative VAL set. Epoch 1 hallucinated objects;
    # later epochs have no GT or predictions, hence no defined primary score.
    # The existing best-defined rule must still select epoch 1, as agreed.
    report["validation_counts"]["target_object_counts"] = {
        key: 0 for key in report["validation_counts"]["target_object_counts"]
    }
    report["validation_counts"]["negative_roi_count"] = 1
    initial = {
        TRAINING_LOSS_SERIES_ID: 0.5,
        VALIDATION_AP50_SERIES_ID: 0.0,
        VALIDATION_MSA_SERIES_ID: 0.0,
        VALIDATION_OBJECT_MACRO_F1_SERIES_ID: 0.0,
        object_class_f1_series_id(1): 0.0,
        object_class_f1_series_id(3): 0.0,
    }
    report["epochs"] = [
        {
            "epoch": epoch,
            "elapsed_seconds": epoch * 10,
            "metrics": [
                {"series_id": item["series_id"], "value": values[item["series_id"]]}
                for item in policy["series"]
                if item["series_id"] in values
            ],
        }
        for epoch, values in enumerate(
            [initial, {TRAINING_LOSS_SERIES_ID: 0.4}, {TRAINING_LOSS_SERIES_ID: 0.3}],
            start=1,
        )
    ]
    selection = select_training_weight_epoch(
        policy, report["epochs"], validation_enabled=True, total_epochs=3
    )
    assert selection == {
        "type": "best-validation",
        "series_id": VALIDATION_OBJECT_MACRO_F1_SERIES_ID,
        "value": 0.0,
        "epoch": 1,
    }
    report["state"] = "SUCCEEDED"
    report["model_selection"] = {**selection, "selected_weight_digest": "d" * 64}
    report["unavailable_metrics"] = [
        {
            "series_id": object_class_f1_series_id(4),
            "reason": "No ground-truth or predicted objects of this class.",
        }
    ]
    assert validate_training_report(report, training_api_limits()) == report
    invalid = copy.deepcopy(report)
    invalid["unavailable_metrics"].insert(
        0,
        {"series_id": object_class_f1_series_id(1), "reason": "Absent at final epoch."},
    )
    with pytest.raises(ValueError, match="observed metric cannot also be unavailable"):
        validate_training_report(invalid, training_api_limits())
