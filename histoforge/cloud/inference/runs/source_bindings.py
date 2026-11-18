"""Assign exact uploaded image files to accepted processing units.

The request and processing grid never change as files arrive. Each assignment
names one input, one unit and its sealed file. The existing run record owns these
assignments independently of completed computation, including after failure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from histoforge.canonical_json import canonical_tree, require_nonnegative_int32
from histoforge.cloud.sources.capture import validate_captured_source_binding


def validate_run_source_binding(
    value: Any, inputs: Mapping[int, Mapping[str, Any]], plans: Mapping[int, Any]
) -> dict[str, Any]:
    """Check an assignment's input, unit, capture and required pixel coverage."""

    from histoforge.cloud.rasterization.captured_source import captured_source_footprint

    if not isinstance(value, Mapping) or set(value) != {
        "input_id",
        "unit_id",
        "source",
    }:
        raise ValueError("run source binding has missing or unknown fields")
    input_id = require_nonnegative_int32(value["input_id"], "input_id")
    unit_id = require_nonnegative_int32(value["unit_id"], "unit_id")
    if input_id not in inputs or input_id not in plans:
        raise ValueError("source binding input_id is absent from the accepted request")
    raster_input = inputs[input_id]["raster_input"]
    view = raster_input["source_view"]
    if "capture" not in view:
        raise ValueError("sealed-file inputs cannot accept incremental source bindings")
    rectangle = processing_rectangle(plans[input_id], unit_id)
    source = validate_captured_source_binding(value["source"], view["capture"])
    required = captured_source_footprint(
        view, raster_input["frame"]["transform"], rectangle
    )
    supplied = source["source_rect_at_level"]
    if required is not None and any(
        required[axis] < supplied[axis]
        or required[axis] + required[extent] > supplied[axis] + supplied[extent]
        for axis, extent in (("x", "width"), ("y", "height"))
    ):
        raise ValueError("source binding omits pixels required by its processing unit")
    return canonical_tree({"input_id": input_id, "unit_id": unit_id, "source": source})


def validate_run_source_bindings(value, inputs, plans) -> list[dict[str, Any]]:
    """Require at most one immutable assignment per unit, in input/unit order."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("source_bindings must be an array")
    bindings = [validate_run_source_binding(item, inputs, plans) for item in value]
    keys = [(item["input_id"], item["unit_id"]) for item in bindings]
    if keys != sorted(set(keys)):
        raise ValueError("source_bindings must be unique and ordered by input and unit")
    return bindings


def unit_source_binding(
    bindings: Sequence[Mapping[str, Any]], input_id: int, unit_id: int
):
    """Find accepted file evidence without substituting another unit's source."""

    return next(
        (
            item["source"]
            for item in bindings
            if item["input_id"] == input_id and item["unit_id"] == unit_id
        ),
        None,
    )


def input_source_bindings(bindings: Sequence[Mapping[str, Any]], input_id: int):
    """Project the complete ordered unit-to-file evidence for one result."""

    return [
        {"unit_id": item["unit_id"], "source": item["source"]}
        for item in bindings
        if item["input_id"] == input_id
    ]


def processing_rectangle(
    plan: Mapping[str, Any], unit_id: int
) -> dict[str, int] | None:
    """Return a core-plus-halo rectangle, or None for the complete finite input."""

    if plan["type"] == "global-decode":
        raise ValueError(
            "global-decode plans do not support incremental image delivery"
        )
    if plan["type"] == "core-halo":
        if unit_id >= len(plan["windows"]):
            raise ValueError("source binding unit_id is absent from the accepted plan")
        return plan["windows"][unit_id]["input_rect"]
    if unit_id != 0:
        raise ValueError("finite input has only processing unit zero")
    return None
