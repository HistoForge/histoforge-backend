"""Trainer identity binds code and the exact deployed behavior configuration."""

from dataclasses import replace

import pytest
from histoforge.cloud.training import implementations
from histoforge.cloud.training.runtime_identity import (
    current_trainer_build,
    resolve_current_trainer,
)
from tests.test_runtime_identity import _install_inputs
from tests.test_training_capabilities import _entry, _registry


def test_current_trainer_matches_only_its_exact_build(monkeypatch) -> None:
    _install_inputs(monkeypatch)
    entry = _entry()
    accepted = current_trainer_build(entry)
    registry = _registry(entry)

    resolved, executing = resolve_current_trainer(registry, accepted)

    assert resolved == entry
    assert executing == accepted
    changed = replace(
        entry,
        trainer_profile={**entry.trainer_profile, "trainer_execution_seconds": 3_599},
    )
    assert current_trainer_build(changed) != accepted
    with pytest.raises(ValueError, match="unavailable in this deployment"):
        resolve_current_trainer(_registry(changed), accepted)


def test_deployment_change_rejects_the_accepted_trainer(monkeypatch) -> None:
    _install_inputs(monkeypatch)
    entry = _entry()
    accepted = current_trainer_build(entry)

    from histoforge.cloud import runtime_identity

    monkeypatch.setattr(
        runtime_identity, "_source_files", lambda: (("histoforge/a.py", b"new-code"),)
    )
    runtime_identity._runtime_build.cache_clear()
    with pytest.raises(ValueError, match="unavailable in this deployment"):
        resolve_current_trainer(_registry(entry), accepted)


def test_registry_entries_cannot_mutate_the_deployed_policy() -> None:
    entry = _entry()
    registry = _registry(entry)
    external = registry.entries()[0]
    external.capability["default_training_settings"]["training_batch_size"] = 9
    external.trainer_profile["trainer_execution_seconds"] = 9

    assert registry.entries()[0] == entry


def test_binding_alone_changes_build_even_when_profiles_and_code_are_identical(
    monkeypatch,
):
    _install_inputs(monkeypatch)
    entry = _entry()
    accepted = current_trainer_build(entry)
    alternate = replace(entry, trainer_implementation_id="test-other-binding-v1")
    monkeypatch.setitem(
        implementations._IMPLEMENTATIONS,
        alternate.trainer_implementation_id,
        implementations.resolve_training_implementation(
            entry.trainer_implementation_id
        ),
    )

    changed = current_trainer_build(alternate)
    assert changed["artifact_digest"] == accepted["artifact_digest"]
    assert changed["manifest_digest"] != accepted["manifest_digest"]
    with pytest.raises(ValueError, match="unavailable in this deployment"):
        resolve_current_trainer(_registry(alternate), accepted)
