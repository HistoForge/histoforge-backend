"""Concrete trainer binding is explicit, private and independent of component IDs."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.training import capabilities, implementations
from histoforge.cloud.training.implementations import (
    DINOV3_INSTANCE_TRAINER_ID,
    TrainingImplementation,
    resolve_training_implementation,
)
from histoforge.cloud.training.profiles import instance_sampler_profile
from histoforge.cloud.training.runs.report import validate_training_report
from tests.test_training_capabilities import _entry, _registry
from tests.test_training_contract import classified_training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report


def alternate_entry(monkeypatch, entry, events):
    """Test-only trainer sharing component schemas, but not the initial preset.

    The initial whole-profile validator rejects this weight decay. The existing
    component evidence schema and optimizer support it. This proves that only
    the explicit binding, not a shared component ID, selects trainer policy.
    """

    original = resolve_training_implementation(entry.trainer_implementation_id)
    profile = copy.deepcopy(entry.trainer_profile)
    profile["trainer_components"]["optimizer"]["parameters"]["weight_decay"] = 0.2
    alternate = replace(
        entry,
        trainer_implementation_id="fixture-alternate-instance-v1",
        trainer_profile=profile,
    )

    def validate(architecture, capability, supplied):
        assert architecture == alternate.architecture
        assert capability == alternate.capability
        assert canonical_bytes(supplied) == canonical_bytes(profile)
        events.append("validate")

    def prepare(selected, **kwargs):
        assert selected == alternate
        result = original.prepare_configuration(selected, **kwargs)
        assert result["optimizer"]["parameters"]["weight_decay"] == 0.2
        events.append("prepare")
        return result

    def execute(selected, **kwargs):
        assert selected == alternate
        assert (
            kwargs["report"]["trainer_configuration"]["optimizer"]["parameters"][
                "weight_decay"
            ]
            == 0.2
        )
        events.append("execute")
        result = original.execute(selected, **kwargs)
        events.append("completed")
        return result

    def load_model_state(network, tensor_state):
        original.load_model_state(network, tensor_state)
        events.append("reload")

    monkeypatch.setitem(
        implementations._IMPLEMENTATIONS,
        alternate.trainer_implementation_id,
        TrainingImplementation(
            validate,
            prepare,
            original.prepare_metric_policy,
            execute,
            load_model_state,
            original.class_initializations,
            original.project_model,
        ),
    )
    return alternate


def test_component_reuse_does_not_select_the_initial_whole_profile(monkeypatch):
    entry = _entry()
    events = []
    alternate = alternate_entry(monkeypatch, entry, events)
    assert {
        name: value["profile_id"]
        for name, value in alternate.trainer_profile["trainer_components"].items()
    } == {
        name: value["profile_id"]
        for name, value in entry.trainer_profile["trainer_components"].items()
    }

    assert _registry(alternate).entries() == (alternate,)
    assert events == ["validate"]
    with pytest.raises(ValueError, match="differs from its closed preset"):
        _registry(
            replace(alternate, trainer_implementation_id=DINOV3_INSTANCE_TRAINER_ID)
        )


@pytest.mark.parametrize("implementation_id", ["missing-v1", "", None, []])
def test_unknown_binding_cannot_advertise_a_capability(implementation_id):
    with pytest.raises(ValueError, match="implementation is not deployed"):
        _registry(replace(_entry(), trainer_implementation_id=implementation_id))


def test_initial_binding_rejects_another_architecture():
    with pytest.raises(ValueError, match="requires a DINOv3 instance adapter"):
        _registry(replace(_entry(), architecture="another-backbone"))


def test_invented_component_ids_cannot_bypass_initial_profile_validation():
    entry = _entry()
    for name, component in entry.trainer_profile["trainer_components"].items():
        component["profile_id"] = f"invented-{name}-v1"
    with pytest.raises(ValueError, match="differs from its closed preset"):
        _registry(entry)


def test_initial_binding_rejects_unsupported_preprocessing():
    entry = _entry()
    entry.capability["default_field_profile"]["preprocessing"] = "unsupported-v1"
    with pytest.raises(ValueError, match="requires real-field normalization"):
        _registry(entry)


@pytest.mark.parametrize("classified", [False, True])
def test_retained_report_is_readable_without_its_trainer_catalog(
    monkeypatch, classified
):
    report = queued_report(classified_training_model_output() if classified else None)
    before = canonical_bytes(report)
    empty = _registry()
    monkeypatch.setattr(capabilities, "deployed_training_registry", lambda: empty)

    def not_an_execution(*_args, **_kwargs):
        pytest.fail("retained evidence must not resolve a deployed implementation")

    monkeypatch.setattr(
        capabilities, "resolve_training_implementation", not_an_execution
    )
    assert (
        canonical_bytes(validate_training_report(report, training_api_limits()))
        == before
    )


@pytest.mark.parametrize(
    "classified,oversample", [(False, None), (True, None), (True, False), (True, True)]
)
def test_retained_report_does_not_import_concrete_trainers(classified, oversample):
    report = queued_report(classified_training_model_output() if classified else None)
    if oversample is not None:
        report["trainer_settings"]["oversample"] = oversample
        report["trainer_configuration"]["sampler"] = instance_sampler_profile(
            oversample
        )
    script = """
import json
import sys
from importlib.abc import MetaPathFinder

class RejectConcreteTrainer(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith((
            "histoforge.cloud.training.bindings",
            "histoforge.cloud.training.trainers",
            "histoforge.cloud.training.implementations",
            "histoforge.cloud.training.capabilities",
        )):
            raise AssertionError(f"retained report imported {fullname}")
        return None

sys.meta_path.insert(0, RejectConcreteTrainer())
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.training.runs.report import validate_training_report
report, limits = json.loads(sys.argv[1])
validated = validate_training_report(report, limits)
assert canonical_bytes(validated) == canonical_bytes(report)
"""
    subprocess.run(
        [sys.executable, "-c", script, json.dumps([report, training_api_limits()])],
        check=True,
        capture_output=True,
        text=True,
    )
