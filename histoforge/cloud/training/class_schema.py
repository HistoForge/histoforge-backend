"""Explicit source-to-target class semantics, independent of network construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.cloud.modules.schema import validate_model_output

CLASS_INITIALIZATIONS = ("use", "add", "define")


def validate_class_initialization(value: Any) -> str:
    if not isinstance(value, str) or value not in CLASS_INITIALIZATIONS:
        raise ValueError("class_initialization must be use, add, or define")
    return value


def validate_training_class_transition(
    source_output: Any, target_output: Any, mode: Any
) -> dict[str, Any]:
    """Validate channel identity/role preservation, not architecture support.

    PathClass names describe target labels, not tensor identity. Use/Add preserve
    the numeric channel prefix and roles even when a displayed label is renamed.
    Implementations separately advertise the task conversions they can execute.
    """

    source = validate_model_output(source_output, "source_model_output")
    target = validate_model_output(target_output, "model_output")
    initialization = validate_class_initialization(mode)
    source_type, target_type = source["type"], target["type"]
    if source_type == "instance_segmentation":
        expected = (
            source_type
            if initialization == "use"
            else "instance_segmentation_and_classification"
        )
        if target_type != expected:
            raise ValueError("class initialization does not match the target task")
        return target
    if target_type != source_type:
        raise ValueError("class initialization cannot change this model's task")
    old, new = source["classChannels"], target["classChannels"]
    if initialization == "use" and len(new) != len(old):
        raise ValueError("use requires the complete existing class-channel count")
    if initialization == "add" and len(new) <= len(old):
        raise ValueError("add requires new channels after the complete source prefix")
    if initialization in {"use", "add"} and any(
        before["channel"] != after["channel"] or before["role"] != after["role"]
        for before, after in zip(old, new, strict=False)
    ):
        raise ValueError("use/add must preserve existing channel identities and roles")
    if initialization == "add" and any(
        channel["role"] == "background" for channel in new[len(old) :]
    ):
        raise ValueError("added class channels must be foreground classes")
    return target


def require_training_source_evidence(
    source_model: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    """Bind the reported original taxonomy/resolution to the exact source recipe."""

    if source_model["output"] != report["source_model_output"]:
        raise ValueError("training source output differs from the accepted report")
    if (
        source_model["input"]["micronsPerPixel"]
        != report["source_model_microns_per_pixel"]
    ):
        raise ValueError("training source resolution differs from the accepted report")
