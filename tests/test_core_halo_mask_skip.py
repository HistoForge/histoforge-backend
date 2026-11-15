"""FIELD instance windows may skip pixels only when mask and halo are disjoint."""

import copy

import numpy as np
import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.feature_stream import decode_feature_stream
from histoforge.cloud.inference.features import build_feature_collection
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.core_halo_output import encode_core_halo_collection
from histoforge.cloud.inference.runs.core_halo_runtime import (
    execute_core_halo_unit,
    finalize_core_halo_input,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
)
from histoforge.cloud.rasterization.inputs import (
    build_model_raster_input,
    build_raster_frame,
)
from histoforge.cloud.sources.files import SourceFileStore

from test_core_halo_runtime import _geometry, _PartitionedRaster, _values


def _masked_request(mask):
    request, activation, limits, _ = _values()
    item = request["inputs"][0]
    raster = item["raster_input"]
    frame = build_raster_frame(
        {
            **{
                key: raster["frame"][key]
                for key in ("source_revision", "region", "transform")
            },
            "inclusion_mask_raster": mask,
        }
    )
    item["raster_input"] = build_model_raster_input(
        raster["source_view"], frame, raster["rasterizer"]
    )
    return request, activation, limits


def _execute(request, activation, limits, unit, tmp_path, infer):
    def predict_window(read_window_image, plan, window, *, channel_percentiles):
        return infer(read_window_image(), plan, window)

    def encode_window(collection, window, declared_classes, target_raster, processing):
        return encode_core_halo_collection(
            collection,
            window,
            declared_classes,
            activation["model"]["output"]["type"],
            target_raster,
            processing,
        )

    return execute_core_halo_unit(
        request,
        0,
        unit,
        limits,
        activation,
        object(),
        SourceFileStore(tmp_path),
        ArtifactFileStore(tmp_path),
        predict_window,
        encode_window,
        channel_percentiles=[[0.0, 255.0]] * 3,
    )


@pytest.mark.parametrize(
    "output_type", ["instance_segmentation", "instance_segmentation_and_classification"]
)
def test_disjoint_window_publishes_ordinary_empty_state_and_assembles(
    tmp_path, monkeypatch, output_type
):
    request, activation, limits = _masked_request(
        _geometry(0, 0, width=200, height=384)
    )
    activation["model"]["output"]["type"] = output_type
    if output_type == "instance_segmentation_and_classification":
        activation["model"]["output"] = {
            "type": output_type,
            "classChannels": [{"channel": 0, "role": "emitted", "pathClass": "Tubule"}],
        }
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    preparations = []

    def prepare(*args):
        preparations.append(args[-1])
        return prepared

    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster", prepare
    )
    predicted_units = []

    def infer(_image, _plan, window):
        predicted_units.append(window["unit_id"])
        return build_feature_collection([("Tubule", _geometry(4, 4))], {"Tubule"})

    publications = [
        _execute(request, activation, limits, unit, tmp_path, infer)
        for unit in range(2)
    ]
    plan = build_expected_model_execution_plans(request, limits)[0]
    assert preparations == [window["input_rect"] for window in plan["windows"]]
    assert prepared.reads == [plan["windows"][0]["input_rect"]]
    assert predicted_units == [0]
    files = ArtifactFileStore(tmp_path)
    for unit, publication in enumerate(publications):
        assert publication["unit_id"] == unit
        assert publication[
            "execution_identity_digest"
        ] == model_run_execution_identity_digest(request, 0, unit, limits)
        reference = publication["reusable_state"]
        assert reference["size_bytes"] > 0
        decoded = decode_feature_stream(
            files.read_artifact_block(
                reference["artifact_id"], 0, reference["size_bytes"]
            ),
            {"Tubule"},
            request["result_budget"],
        )
        assert decoded.feature_count == (1 if unit == 0 else 0)
    final = finalize_core_halo_input(
        request, 0, limits, publications, activation, files
    )
    assert final["input"]["execution_plan"] == plan
    assert final["input"]["feature_count"] == 1
    with pytest.raises(GatewayFailure, match="invalid"):
        finalize_core_halo_input(
            request, 0, limits, publications[:1], activation, files
        )


@pytest.mark.parametrize(
    "mask, should_predict",
    [
        (_geometry(10, 4, width=213.5), False),
        (_geometry(10, 4, width=214), True),  # edge at input x=224
        (_geometry(10, 4, width=215), True),  # only halo, not ownership core
        (
            {
                "type": "Polygon",
                "coordinates": [[[220, 384], [220, 380], [224, 384], [220, 384]]],
            },
            True,
        ),
        (
            {
                "type": "MultiPolygon",
                "coordinates": [
                    _geometry(8, 8)["coordinates"],
                    _geometry(100, 100)["coordinates"],
                ],
            },
            False,
        ),
        (
            {
                "type": "MultiPolygon",
                "coordinates": [
                    _geometry(8, 8)["coordinates"],
                    _geometry(230, 100)["coordinates"],
                ],
            },
            True,
        ),
    ],
)
def test_closed_input_rectangle_includes_halo_edge_corner_and_all_islands(
    mask, should_predict, tmp_path, monkeypatch
):
    request, activation, limits = _masked_request(mask)
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_: prepared,
    )
    calls = []

    def infer(*_):
        calls.append(True)
        return build_feature_collection((), {"Tubule"})

    _execute(request, activation, limits, 1, tmp_path, infer)
    assert bool(calls) is should_predict
    assert bool(prepared.reads) is should_predict


def test_window_wholly_inside_mask_hole_skips(tmp_path, monkeypatch):
    mask = _geometry(0, 0, width=512, height=384)
    mask["coordinates"].append(
        _geometry(100, 100, width=200, height=200)["coordinates"][0]
    )
    request, activation, limits = _masked_request(mask)
    request["expected_profile"] = build_core_halo_profile(128, 128, 16, "2" * 64)
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_: object(),  # Any attempt to read pixels fails.
    )
    _execute(
        request,
        activation,
        limits,
        5,
        tmp_path,
        lambda *_: pytest.fail("predicted hole"),
    )


@pytest.mark.parametrize("kind", ["parent", "dense"])
def test_other_output_semantics_never_skip(kind, tmp_path, monkeypatch):
    request, activation, limits = _masked_request(_geometry(4, 4))
    if kind == "parent":
        scope = request["inputs"][0]["output"]["scope"]
        scope.update(type="parent", parent_id="22222222-2222-4222-8222-222222222222")
    else:
        activation["model"]["output"] = {
            "type": "dense_classification",
            "classChannels": [{"channel": 0, "role": "emitted", "pathClass": "Tubule"}],
        }
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_: prepared,
    )
    calls = []

    def infer(*_):
        calls.append(True)
        return build_feature_collection((), {"Tubule"})

    _execute(request, activation, limits, 1, tmp_path, infer)
    assert calls == [True]
    assert len(prepared.reads) == 1


@pytest.mark.parametrize("invalid", ["source", "mask", "activation"])
def test_disjoint_window_still_rejects_invalid_inputs(invalid, tmp_path, monkeypatch):
    request, activation, limits = _masked_request(_geometry(4, 4))
    if invalid == "mask":
        request["inputs"][0]["raster_input"]["frame"]["inclusion_mask_digest"] = (
            "0" * 64
        )
    if invalid == "activation":
        activation = copy.deepcopy(activation)
        activation["backend"]["runner_build"] = "0" * 64

    def prepare(*_):
        if invalid != "source":
            pytest.fail("invalid identity reached source preparation")
        raise GatewayFailure("ARTIFACT_PRECONDITION_FAILED", "source expired")

    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster", prepare
    )
    with pytest.raises((ValueError, GatewayFailure)):
        _execute(
            request,
            activation,
            limits,
            1,
            tmp_path,
            lambda *_: pytest.fail("predicted"),
        )
    assert not ArtifactFileStore(tmp_path).artifact_ids()
