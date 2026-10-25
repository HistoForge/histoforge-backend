"""Private deployed-training capability resolution.

The installed-module catalog and dataset seal use this same in-process owner.
Entries bind one reviewed trainer implementation to compatible model behavior
and configuration. The selected recipe remains the sole owner of class meaning.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud.architectures.registry import (
    model_runner_availability,
    resolved_fixed_field_profile,
)
from histoforge.cloud.training.contract import (
    model_with_training_result_settings,
    validate_model_training_capability,
    validate_training_availability,
)
from histoforge.cloud.training.implementations import resolve_training_implementation
from histoforge.cloud.training.limits import (
    validate_training_api_limits,
    validate_training_service_limits,
)
from histoforge.cloud.training.profiles import validate_trainer_profile


@dataclass(frozen=True)
class TrainingCapabilityEntry:
    """One private compatibility rule and its client-visible projection."""

    architecture: str
    capability: Mapping[str, Any]
    trainer_profile: Mapping[str, Any]
    trainer_implementation_id: str


class TrainingCapabilityRegistry:
    """Resolve curated capabilities by ID or installed model configuration."""

    def __init__(
        self,
        entries: Iterable[TrainingCapabilityEntry],
        *,
        api_limits: Any,
        service_limits: Any,
    ) -> None:
        limits = validate_training_api_limits(api_limits)
        private_limits = validate_training_service_limits(service_limits)
        max_text_bytes = limits["report"]["max_text_bytes"]
        resolved: dict[str, TrainingCapabilityEntry] = {}
        for entry in entries:
            implementation = resolve_training_implementation(
                entry.trainer_implementation_id
            )
            if not isinstance(entry.architecture, str) or not entry.architecture:
                raise ValueError("training architecture must be a nonempty string")
            capability = validate_model_training_capability(
                entry.capability, max_text_bytes
            )
            if (
                capability["total_epochs_range"]["maximum"]
                > limits["report"]["max_epochs"]
            ):
                raise ValueError(
                    "training capability total_epochs_range exceeds report limits"
                )
            capability_id = capability["capability_id"]
            if capability_id in resolved:
                raise ValueError("training capability IDs must be unique")
            profile = validate_trainer_profile(
                entry.trainer_profile,
                capability,
                limits,
                private_limits,
            )
            implementation.validate_profile(entry.architecture, capability, profile)
            resolved[capability_id] = replace(
                entry,
                capability=capability,
                trainer_profile=profile,
            )
        self._entries = resolved
        self._max_text_bytes = max_text_bytes

    def entries(self) -> tuple[TrainingCapabilityEntry, ...]:
        """Return detached entries in stable capability-ID order."""

        return tuple(
            replace(
                entry,
                capability=canonical_tree(entry.capability),
                trainer_profile=canonical_tree(entry.trainer_profile),
            )
            for _capability_id, entry in sorted(self._entries.items())
        )

    def resolve(self, capability_id: str) -> dict[str, Any]:
        """Return one deployed capability, or reject an unknown stable ID."""

        try:
            return canonical_tree(self._entries[capability_id].capability)
        except (KeyError, TypeError) as error:
            raise ValueError("training capability is not deployed") from error

    def resolve_entry(
        self,
        recipe: Mapping[str, Any],
        model_id: str,
        capability_id: str,
    ) -> TrainingCapabilityEntry:
        """Require compatible source and result configurations for this capability."""

        try:
            entry = self._entries[capability_id]
            model = recipe["models"][model_id]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "training capability or selected model is unavailable"
            ) from error
        if (
            model_runner_availability(model).get("type") != "available"
            or entry.architecture != model.get("arch")
            or not _matches_capability(recipe, model_id, entry)
        ):
            raise ValueError("selected model does not match its training capability")
        return replace(
            entry,
            capability=canonical_tree(entry.capability),
            trainer_profile=canonical_tree(entry.trainer_profile),
        )

    def availability(
        self,
        recipe: Mapping[str, Any],
        model_id: str,
        runner_availability: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Project the exact selected model against all deployed entries."""

        model = recipe["models"][model_id]
        if runner_availability.get("type") != "available":
            return self._unavailable(
                "unsupported_model_configuration",
                "This model cannot be fine-tuned by the deployed trainer.",
            )

        matches = [
            {
                "capability": entry.capability,
                "class_initializations": list(
                    resolve_training_implementation(
                        entry.trainer_implementation_id
                    ).class_initializations(
                        model["output"]["type"], entry.capability["output_type"]
                    )
                ),
            }
            for entry in sorted(
                self._entries.values(),
                key=lambda item: item.capability["capability_id"],
            )
            if entry.architecture == model["arch"]
            and _matches_capability(recipe, model_id, entry)
        ]
        outputs = [option["capability"]["output_type"] for option in matches]
        if len(outputs) != len(set(outputs)):
            raise RuntimeError("multiple training capabilities match one target task")
        if matches:
            return validate_training_availability(
                {"type": "available", "options": matches},
                self._max_text_bytes,
            )

        code = (
            "unsupported_model_configuration"
            if any(
                entry.architecture == model["arch"] for entry in self._entries.values()
            )
            else "trainer_not_deployed"
        )
        message = (
            "This model configuration has no compatible deployed trainer."
            if code == "unsupported_model_configuration"
            else "Training is not available for this model in this backend build."
        )
        return self._unavailable(code, message)

    def _unavailable(self, code: str, message: str) -> dict[str, Any]:
        return validate_training_availability(
            {"type": "unavailable", "code": code, "message": message},
            self._max_text_bytes,
        )


def _matches_capability(
    recipe: Mapping[str, Any],
    model_id: str,
    entry: TrainingCapabilityEntry,
) -> bool:
    """Check model behavior and configuration against trainer capabilities."""

    model = recipe["models"][model_id]
    capability = entry.capability
    implementation = resolve_training_implementation(entry.trainer_implementation_id)
    if not any(step.get("model") == model_id for step in recipe["steps"]):
        return False
    if model["input"]["type"] != capability["input_kind"]:
        return False
    if not implementation.class_initializations(
        model["output"]["type"], capability["output_type"]
    ):
        return False

    profile = capability["default_field_profile"]
    if profile["planner"] == "fixed-field-v1":
        if resolved_fixed_field_profile(dict(model)) != profile:
            return False
    else:
        # The capability bounds the new training/result fields, not the source
        # model's previous field sizes. Its permitted variants share this
        # default's planner, alignment, overlap, aggregation, and preprocessing.
        candidate = model_with_training_result_settings(
            model,
            profile,
            model["params"].get("min_instance_pixels"),
            model["gpu"],
        )
        if model_runner_availability(candidate)["type"] != "available":
            return False

    padding = (
        model["input"]["framing"]["paddingModelPx"]
        if model["input"]["type"] == "crop"
        else 0
    )
    if padding != capability["crop_padding_model_px"]:
        return False
    if model["input"]["type"] == "crop" and not _has_crop_parent(recipe, model_id):
        return False
    return True


def _has_crop_parent(recipe: Mapping[str, Any], model_id: str) -> bool:
    return any(
        step.get("model") == model_id
        and "parentModel" in step
        and "parentClass" in step
        for step in recipe["steps"]
    )


@lru_cache(maxsize=1)
def deployed_training_registry() -> TrainingCapabilityRegistry:
    """Return the current private registry.

    Entries are added only with their trainer adapter and qualification
    evidence. Catalog, dataset, and worker routes all use this same owner.
    """

    from histoforge.cloud._config import DEPLOYMENT_CONFIG
    from histoforge.cloud.training.initial_capability import (
        cellpose_segmentation_entry,
        hovernet_classification_entry,
        hovernet_segmentation_entry,
        initial_cellpose_classification_entry,
        semantic_dense_entry,
    )

    instance_entries = tuple(
        build(
            trainer_execution_seconds=44_100,
            max_validation_assignment_matrix_entries=1_400_000,
        )
        for build in (
            initial_cellpose_classification_entry,
            cellpose_segmentation_entry,
            hovernet_classification_entry,
            hovernet_segmentation_entry,
        )
    )

    return TrainingCapabilityRegistry(
        (
            *instance_entries,
            semantic_dense_entry(trainer_execution_seconds=44_100),
        ),
        api_limits=DEPLOYMENT_CONFIG.training_api_limits.wire_value(),
        service_limits=DEPLOYMENT_CONFIG.training_service_limits.wire_value(),
    )


def model_training_availability(
    recipe: Mapping[str, Any],
    model_id: str,
    runner_availability: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deployed catalog projection for one installed model."""

    return canonical_tree(
        deployed_training_registry().availability(recipe, model_id, runner_availability)
    )


def resolve_training_capability(capability_id: str) -> dict[str, Any]:
    """Resolve the same deployed capability used by catalog projection."""

    return canonical_tree(deployed_training_registry().resolve(capability_id))


def resolve_training_capability_entry(
    recipe: Mapping[str, Any], model_id: str, capability_id: str
) -> TrainingCapabilityEntry:
    """Resolve the private trainer profile for one exact installed model."""

    return deployed_training_registry().resolve_entry(recipe, model_id, capability_id)
