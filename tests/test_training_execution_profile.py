"""Fixed report evidence must describe the implementation that will execute."""

import copy
import subprocess
import sys

import pytest
from histoforge.cloud.architectures import dinov3_optimization
from histoforge.cloud.training.bindings.dinov3_instance import execute_instance_training
from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.component_schema import (
    managed_objective_rules,
)
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID
from histoforge.cloud.training.runs.report import validate_training_report
from histoforge.cloud.training.trainers.instance import loop as training_execution
from tests.test_training_dataset import training_api_limits
from tests.test_training_execution import _execute, _setup


def test_execution_entry_imports_managed_targets_before_torch_in_fresh_interpreter():
    script = """
import importlib.abc
import sys

targets = 'histoforge._vendor.histoforge_models.pathology.managed_training.targets'
observed = []

class TraceTorchImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            observed.append(targets in sys.modules)

sys.meta_path.insert(0, TraceTorchImport())
from histoforge.cloud.training.trainers.instance.loop import train_managed_model

assert callable(train_managed_model)
assert observed == [True]
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


@pytest.mark.parametrize(
    "component,field,value",
    [
        ("preprocessing", "percentiles", [5.0, 95.0]),
        ("preprocessing", "channels", "BGR"),
        ("preprocessing", "clip", True),
        ("preprocessing", "artificial_pixel_value", 1.0),
        ("preprocessing", "percentile_method", "nearest"),
        ("preprocessing", "population", "all-roi-pixels"),
        ("preprocessing", "operation_order", ["normalize", "augment"]),
        ("objective", "effective_support", "all-pixels"),
        ("objective", "field_reduction", "pixel-weighted-mean"),
        ("objective", "task_combination", "sum-active-task-losses"),
        ("objective", "cross_entropy_reduction", "unweighted-pixel-mean"),
        ("objective", "tversky_reduction", "foreground-classes-only"),
        ("objective", "initial_log_variance", 1.0),
        ("objective", "gradient_norm_type", 1.0),
        ("objective", "gradient_clip_parameters", "model-and-objective"),
        ("objective", "precision", "bfloat16-autocast"),
        ("class_weighting", "policy", "inverse-frequency"),
        ("class_weighting", "population", "training-and-validation-pixels"),
        ("class_weighting", "overlapping_rois", "globally-deduplicate"),
        ("class_weighting", "channel_identity", "target-array-position"),
        ("class_weighting", "count_floor", 2),
        ("class_weighting", "formula", "median(count)/count"),
        ("class_weighting", "dtype", "float64"),
    ],
)
def test_fixed_evidence_drift_fails_before_model_or_dataset_work(
    tmp_path, monkeypatch, component, field, value
):
    setup = _setup(tmp_path)
    report = setup[1]
    components = report["trainer_configuration"]
    parameters = (
        components["objective"]["parameters"]["class_weighting"]
        if component == "class_weighting"
        else components[component]["parameters"]
    )
    parameters[field] = value
    # Persisted evidence and the direct execution boundary both reject drift.
    with pytest.raises(ValueError):
        validate_training_report(report, training_api_limits())
    captured = copy.deepcopy(report)

    def unexpected(*_args, **_kwargs):
        pytest.fail("inconsistent evidence reached execution")

    monkeypatch.setattr(setup[0], "to", unexpected)
    monkeypatch.setattr(training_execution, "TrainingEpochDataset", unexpected)
    monkeypatch.setattr(dinov3_optimization, "build_dinov3_optimizer", unexpected)
    if component != "preprocessing":
        # The task owns objective rules, not a concrete model's preprocessing.
        with pytest.raises(ValueError, match="implementation is not supported"):
            _execute(setup, unexpected, unexpected)
    else:
        from tests.test_training_output import _session

        session = _session(tmp_path / "claimed", validation=False)
        with pytest.raises(ValueError, match="preprocessing profile is not supported"):
            execute_instance_training(
                TrainingCapabilityEntry(
                    "dinov3-adapter-cellpose", setup[3], {}, DINOV3_INSTANCE_TRAINER_ID
                ),
                run=session.run,
                report=report,
                dataset=setup[2],
                source=None,
                device="cpu",
                api_limits=training_api_limits(),
                source_limits={},
                epoch_completed=unexpected,
            )
    assert report == captured


def test_shared_rules_preserve_consumed_objective_options(tmp_path, monkeypatch):
    setup = _setup(tmp_path)
    options = setup[1]["trainer_configuration"]["objective"]["parameters"]
    options.update(
        classification_objective="cross_entropy",
        classification_loss_weight=2.0,
        use_uncertainty_weighting=False,
        tversky_alpha=0.4,
        tversky_gamma=1.5,
        max_gradient_norm=0.5,
    )
    # Fixed descriptions must agree with the option actually supplied to Torch.
    options["task_combination"] = managed_objective_rules(
        classified=bool(options["class_count"]), use_uncertainty_weighting=False
    )["task_combination"]
    observed = []

    def train(_model, _optimizer, _batches, **kwargs):
        observed.append(kwargs)
        return ()

    monkeypatch.setattr(training_execution, "train_instance_fields", train)
    assert _execute(setup, lambda _image: None, lambda _epoch: None) == ()
    assert len(observed) == 1
    for name in (
        "classification_objective",
        "classification_loss_weight",
        "tversky_alpha",
        "tversky_gamma",
        "max_gradient_norm",
        "flow_loss_weighting",
    ):
        assert observed[0][name] == options[name]
    assert not observed[0]["objective"].uses_uncertainty_weighting
    assert observed[0]["class_weights"].tolist() == [
        item["weight"] for item in options["class_weighting"]["channels"]
    ]
