"""Exact deployed build identity for one curated managed trainer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import canonical_tree
from histoforge.cloud.runtime_identity import runtime_build
from histoforge.cloud.training.capabilities import (
    TrainingCapabilityEntry,
    TrainingCapabilityRegistry,
)

_COMPONENT = "histoforge-managed-training"


def trainer_behavior_policy(entry: TrainingCapabilityEntry) -> dict[str, Any]:
    """Return every curated behavior input bound into one trainer build."""

    return canonical_tree(
        {
            "schema_version": 1,
            "trainer_implementation_id": entry.trainer_implementation_id,
            "architecture": entry.architecture,
            "capability": entry.capability,
            "trainer_profile": entry.trainer_profile,
        }
    )


def current_trainer_build(entry: TrainingCapabilityEntry) -> dict[str, str]:
    """Recompute the compact identity of this deployed trainer entry."""

    return runtime_build(_COMPONENT, trainer_behavior_policy(entry)).identity


def resolve_current_trainer(
    registry: TrainingCapabilityRegistry,
    expected_build: Any,
    *,
    build_entry: Callable[[TrainingCapabilityEntry], Mapping[str, Any]] = (
        current_trainer_build
    ),
) -> tuple[TrainingCapabilityEntry, dict[str, str]]:
    """Find the sole current entry whose recomputed build equals acceptance."""

    expected = parse_software_build_identity(expected_build)
    matches = []
    for entry in registry.entries():
        current = parse_software_build_identity(build_entry(entry))
        if current == expected:
            matches.append((entry, current))
    if not matches:
        raise ValueError("accepted trainer build is unavailable in this deployment")
    if len(matches) != 1:
        raise RuntimeError("multiple deployed trainers share one build identity")
    return matches[0]
