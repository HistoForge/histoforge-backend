import copy
import json
import math
from pathlib import Path

from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import require_canonical_uuid, require_path_class_name
from histoforge.cloud.execution_limits import (
    fits_target_raster,
    validate_target_raster_limits,
)
from histoforge.cloud.modules.schema import validate_model_id
from histoforge.module_ref import ModuleRef
from shapely.geometry import shape

FIXTURE = (
    Path(__file__).parent / "fixtures" / "contracts" / "execution-protocol-v1.json"
)


def test_shared_fixture_pins_every_spatial_selection_rule():
    fixture = _fixture()
    _validate_fixture(fixture)

    for case in fixture["cases"]:
        assert _select_protocol(case, fixture) == case["expectedProtocol"], case["name"]


def test_parent_count_cannot_select_the_protocol():
    fixture = _fixture()
    case = copy.deepcopy(
        _named_case(
            fixture, "small-current-parents-and-excluded-large-objects-stay-finite"
        )
    )
    parents = case["cropWork"]["child-current"]["currentParents"]
    template = parents[0]
    for _ in range(500):
        parents.append(copy.deepcopy(template))

    assert _select_protocol(case, fixture) == "finite"


def _select_protocol(case, fixture):
    source = fixture["source"]
    models = fixture["models"]
    selected_names = case["selectedModels"]
    selected = [models[name] for name in selected_names]
    selected_by_step = {model["stepId"]: model for model in selected}

    if case["region"]["type"] == "slide":
        return "recoverable"

    target_frame = _clipped_frame(_bounds(case["targetGeometry"]), 0, 0, source)
    for model_name, model in zip(selected_names, selected, strict=True):
        if model["type"] == "field":
            if not _fits(target_frame, model, source):
                return "recoverable"
            continue

        crop_work = case["cropWork"][model_name]
        for parent in crop_work["currentParents"]:
            if _can_supply_parent(parent, model):
                parent_frame = _crop_frame(_bounds(parent["geometry"]), model, source)
                if not _fits(parent_frame, model, source):
                    return "recoverable"

        upstream = selected_by_step.get(model["parentStepId"])
        if crop_work["discoverParentsAcrossRegion"] and upstream is not None:
            assert upstream["type"] == "field"
            possible_parent = _raster_domain(target_frame, upstream, source)
            if not _fits(_crop_frame(possible_parent, model, source), model, source):
                return "recoverable"

    return "finite"


def _can_supply_parent(parent, crop_model):
    producer = parent.get("producerStepId")
    if producer is not None:
        return producer == crop_model["parentStepId"]
    path_class = parent.get("pathClass")
    return path_class is not None and path_class in (
        set(crop_model["parentClasses"]) | set(crop_model["scriptVisibleClasses"])
    )


def _crop_frame(frame, model, source):
    padding_um = model["paddingModelPx"] * model["targetMpp"]
    return _clipped_frame(
        frame,
        padding_um / source["mppX"],
        padding_um / source["mppY"],
        source,
    )


def _raster_domain(frame, model, source):
    width, height = _raster_size(frame, model["targetMpp"], source)
    step_x = model["targetMpp"] / source["mppX"]
    step_y = model["targetMpp"] / source["mppY"]
    return (
        frame[0],
        frame[1],
        frame[0] + width * step_x,
        frame[1] + height * step_y,
    )


def _fits(frame, model, source):
    width, height = _raster_size(frame, model["targetMpp"], source)
    return fits_target_raster(
        {"width": width, "height": height}, model["finiteTargetRasterLimits"]
    )


def _raster_size(frame, target_mpp, source):
    width = max(1, math.ceil((frame[2] - frame[0]) / (target_mpp / source["mppX"])))
    height = max(1, math.ceil((frame[3] - frame[1]) / (target_mpp / source["mppY"])))
    return width, height


def _clipped_frame(frame, padding_x, padding_y, source):
    result = (
        max(0, frame[0] - padding_x),
        max(0, frame[1] - padding_y),
        min(source["width"], frame[2] + padding_x),
        min(source["height"], frame[3] + padding_y),
    )
    if not result[2] > result[0] or not result[3] > result[1]:
        raise ValueError("model input frame is empty after image clipping")
    return result


def _bounds(geometry):
    return shape(canonical_geometry(geometry)).bounds


def _validate_fixture(fixture):
    assert fixture["schemaVersion"] == 1
    ModuleRef.parse(fixture["moduleRef"])
    source = fixture["source"]
    assert source["width"] > 0 and source["height"] > 0
    assert source["mppX"] > 0 and source["mppY"] > 0

    for model in fixture["models"].values():
        assert model["type"] in {"field", "crop"}
        validate_model_id(model["stepId"])
        assert math.isfinite(model["targetMpp"]) and model["targetMpp"] > 0
        validate_target_raster_limits(model["finiteTargetRasterLimits"])
        if model["type"] == "crop":
            validate_model_id(model["parentStepId"])
            assert model["paddingModelPx"] >= 0
            assert model["parentClasses"]
            for path_class in model["parentClasses"] + model["scriptVisibleClasses"]:
                require_path_class_name(path_class)

    for case in fixture["cases"]:
        assert case["expectedProtocol"] in {"finite", "recoverable"}
        assert case["selectedModels"]
        for model_name in case["selectedModels"]:
            assert model_name in fixture["models"]
        selected_crop_models = {
            name
            for name in case["selectedModels"]
            if fixture["models"][name]["type"] == "crop"
        }
        assert set(case["cropWork"]) == selected_crop_models
        canonical_geometry(case["targetGeometry"])
        if case["region"]["type"] == "roi":
            require_canonical_uuid(case["region"]["roi_id"])
        else:
            assert case["region"] == {"type": "slide"}
        for crop_work in case["cropWork"].values():
            assert set(crop_work) == {
                "discoverParentsAcrossRegion",
                "currentParents",
            }
            assert isinstance(crop_work["discoverParentsAcrossRegion"], bool)
            for parent in crop_work["currentParents"]:
                if "pathClass" in parent:
                    require_path_class_name(parent["pathClass"])
                if "producerStepId" in parent:
                    validate_model_id(parent["producerStepId"])
                canonical_geometry(parent["geometry"])


def _named_case(fixture, name):
    return next(case for case in fixture["cases"] if case["name"] == name)


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))
