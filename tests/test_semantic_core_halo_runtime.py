"""Semantic core publication and final assembly enforce distinct resource limits."""

import copy

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.feature_stream import decode_feature_stream
from histoforge.cloud.inference.features import build_feature_collection
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.core_halo_output import (
    encode_core_halo_collection,
    encode_semantic_window,
)
from histoforge.cloud.inference.runs.core_halo_runtime import (
    execute_core_halo_unit,
    finalize_core_halo_input,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
)
from histoforge.cloud.inference.semantic_features import (
    feature_collection_from_class_map,
)
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.sources.files import SourceFileStore
from shapely.geometry import box, mapping, shape
from tests.test_core_halo_runtime import _PartitionedRaster, _read_then_return, _values
from tests.test_durable_runtime import _Source
from tests.test_durable_runtime import _values as _finite_values
from tests.test_semantic_core_halo import (
    _BUDGET,
    _CLASSES,
    _maps,
    _runner,
    _SyntheticSemanticNet,
)


def _output():
    return {
        "type": "dense_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "training_only", "pathClass": "Other"},
            *[
                {"channel": channel, "role": "emitted", "pathClass": name}
                for channel, name in _CLASSES.items()
            ],
        ],
    }


def _case(tmp_path, monkeypatch, labels, *, halo=32):
    request, activation, limits, _ = _values()
    request["expected_backend"]["architecture"] = "dinov3-adapter-semantic"
    activation["backend"]["architecture"] = "dinov3-adapter-semantic"
    request["expected_profile"] = build_core_halo_profile(256, 192, 32, "2" * 64)
    request["halo_px_override"] = halo
    activation["model"]["output"] = _output()
    plan = build_expected_model_execution_plans(request, limits)[0]
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )

    def predict(_pixels, *, window_x, window_y, **_kwargs):
        height, width = _pixels.shape[:2]
        return _maps(labels[window_y : window_y + height, window_x : window_x + width])

    monkeypatch.setattr(
        "histoforge._vendor.histoforge_models.pathology.window_maps.predict_window_maps",
        predict,
    )
    runner = _runner()
    runner.activation = activation
    runner.source_store, runner.source_files = object(), object()
    runner.artifact_files = ArtifactFileStore(tmp_path)
    runner_type = type(runner)

    def publish():
        return [
            runner_type.__dict__["infer_durable_core_halo_unit"]._get_raw_f()(
                runner,
                request,
                0,
                window["unit_id"],
                limits,
                channel_percentiles=[[0.0, 255.0]] * 3,
            )
            for window in plan["windows"]
        ]

    def finalize(publications):
        return finalize_core_halo_input(
            request, 0, limits, publications, activation, runner.artifact_files
        )

    return request, activation, limits, plan, prepared, runner, publish, finalize


def _read(reference, files, budget):
    return decode_feature_stream(
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"]),
        _CLASSES.values(),
        budget,
    )


def _assert_same_published_result(first, second, files):
    # Publication creates a fresh artifact ID. Content and input evidence must
    # still repeat exactly when finalization reuses the same completed cores.
    before, after = copy.deepcopy(first), copy.deepcopy(second)
    first_ref = before["input"]["feature_stream_artifact"]
    second_ref = after["input"]["feature_stream_artifact"]
    first_id, second_id = first_ref.pop("artifact_id"), second_ref.pop("artifact_id")
    assert before == after
    assert files.read_artifact_block(first_id, 0, first_ref["size_bytes"]) == (
        files.read_artifact_block(second_id, 0, second_ref["size_bytes"])
    )


def test_semantic_finite_and_durable_entry_points_execute_the_same_predictor(
    tmp_path, monkeypatch
):
    finite, request, activation, limits = _finite_values()
    for backend in (
        finite["expected_backend"],
        request["expected_backend"],
        activation["backend"],
    ):
        backend["architecture"] = "dinov3-adapter-semantic"
    activation["model"]["output"] = _output()
    y, x = np.mgrid[:384, :512]
    pixels = np.stack([x, y, (x + y) / 2], axis=-1).astype(np.float32)
    image = RasterizedImage(pixels, np.ones((384, 512), bool))
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: image,
    )
    network = _SyntheticSemanticNet()
    runner = _runner(network=network)
    runner.activation, runner.source_store = activation, _Source()
    runner.source_files, runner.artifact_files = (
        SourceFileStore(tmp_path),
        ArtifactFileStore(tmp_path),
    )
    runner_type = type(runner)

    ordinary = runner_type.__dict__["infer"]._get_raw_f()(runner, finite)
    ordinary_batches = network.batches.copy()
    network.batches.clear()
    durable = runner_type.__dict__["infer_durable_input"]._get_raw_f()(
        runner, request, request["inputs"][0]["input_id"], limits
    )
    actual = _read(
        durable["input"]["feature_stream_artifact"],
        runner.artifact_files,
        request["result_budget"],
    )

    assert actual.collection["features"]
    assert canonical_bytes(actual.collection) == canonical_bytes(ordinary["geojson"])
    assert durable["input"]["execution_plan"]["type"] == "finite"
    assert len(network.batches) == len(ordinary_batches)
    for actual_batch, expected_batch in zip(
        network.batches, ordinary_batches, strict=True
    ):
        np.testing.assert_array_equal(actual_batch, expected_batch)


@pytest.mark.parametrize("halo", [0, 32, 64])
def test_core_artifacts_assemble_exact_multiclass_regions(tmp_path, monkeypatch, halo):
    labels = np.zeros((384, 512), np.int32)
    labels[32:352, 32:480] = 2
    labels[96:288, 96:416] = 0
    labels[144:240, 192:320] = 3
    labels[31, 31] = 2  # Only a corner touches the main region.
    labels[200:230, 400:440] = 1  # Non-emitted wins are not reassigned.
    request, _, limits, plan, prepared, runner, publish, finalize = _case(
        tmp_path, monkeypatch, labels, halo=halo
    )

    publications = publish()
    final = finalize(publications)
    actual = _read(
        final["input"]["feature_stream_artifact"],
        runner.artifact_files,
        request["result_budget"],
    )
    expected = feature_collection_from_class_map(
        labels, _CLASSES, _CLASSES.values(), _BUDGET
    )
    assert actual.feature_count == len(expected["features"])
    for feature in actual.collection["features"]:
        assert any(
            feature["properties"]["path_class"] == other["properties"]["path_class"]
            and shape(feature["geometry"]).equals(shape(other["geometry"]))
            for other in expected["features"]
        )
    assert final["input"]["execution_plan"] == plan
    assert plan["resolved_halo_px"] == halo
    assert final["decoded_bytes"] == actual.decoded_bytes
    assert prepared.reads == [window["input_rect"] for window in plan["windows"]]
    fragments = [
        _read(
            item["reusable_state"], runner.artifact_files, limits["intermediate_result"]
        )
        for item in publications
    ]
    assert sum(item.feature_count for item in fragments) > actual.feature_count
    _assert_same_published_result(finalize(publications), final, runner.artifact_files)
    assert len(prepared.reads) == len(plan["windows"])


@pytest.mark.parametrize("empty", [False, True])
def test_final_region_limit_does_not_count_separate_core_fragments(
    tmp_path, monkeypatch, empty
):
    labels = np.full((384, 512), 0 if empty else 2, np.int32)
    request, _, _, _, _, runner, publish, finalize = _case(
        tmp_path, monkeypatch, labels
    )
    request["result_budget"]["max_features"] = 1
    final = finalize(publish())
    decoded = _read(
        final["input"]["feature_stream_artifact"],
        runner.artifact_files,
        request["result_budget"],
    )

    assert decoded.feature_count == (0 if empty else 1)
    if not empty:
        assert shape(decoded.collection["features"][0]["geometry"]).equals(
            box(0, 0, 512, 384)
        )


@pytest.mark.parametrize(
    "field",
    ["max_record_bytes", "max_vertices", "max_decoded_bytes", "max_transferred_bytes"],
)
def test_actual_assembled_output_limits_accept_exact_size_and_reject_one_less(
    tmp_path, monkeypatch, field
):
    labels = np.full((384, 512), 2, np.int32)
    request, _, _, _, prepared, runner, publish, finalize = _case(
        tmp_path, monkeypatch, labels
    )
    publications = publish()
    baseline = finalize(publications)
    reference = baseline["input"]["feature_stream_artifact"]
    decoded = _read(reference, runner.artifact_files, request["result_budget"])
    boundary = {
        "max_record_bytes": max(
            len(canonical_bytes(feature)) for feature in decoded.collection["features"]
        ),
        "max_vertices": decoded.vertex_count,
        "max_decoded_bytes": decoded.decoded_bytes,
        "max_transferred_bytes": reference["size_bytes"],
    }[field]
    request["result_budget"][field] = boundary
    _assert_same_published_result(
        finalize(publications), baseline, runner.artifact_files
    )
    request["result_budget"][field] -= 1

    with pytest.raises(GatewayFailure, match=field) as failure:
        finalize(publications)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert len(prepared.reads) == 4
    # Failure does not erase successfully saved prediction work.
    for item in publications:
        assert (
            runner.artifact_files.load_artifact_ref(
                item["reusable_state"]["artifact_id"]
            )
            == item["reusable_state"]
        )


def test_intermediate_limit_precedes_union_even_when_final_regions_would_fit(
    tmp_path, monkeypatch
):
    labels = np.full((384, 512), 2, np.int32)
    request, _, limits, _, _, _, publish, finalize = _case(
        tmp_path, monkeypatch, labels
    )
    request["result_budget"]["max_features"] = 1
    limits["intermediate_result"]["max_features"] = 3
    publications = publish()
    monkeypatch.setattr(
        "histoforge.cloud.inference.semantic_features.assemble_semantic_fragments",
        lambda *_args: pytest.fail("union must not receive over-budget fragments"),
    )
    with pytest.raises(GatewayFailure, match="max_features") as failure:
        finalize(publications)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"


def test_semantic_callback_cannot_publish_geometry_outside_its_core(
    tmp_path, monkeypatch
):
    request, activation, limits, _, _, runner, _, _ = _case(
        tmp_path, monkeypatch, np.zeros((384, 512), np.int32)
    )

    def encode_window(collection, window, declared_classes, target_raster, processing):
        return encode_core_halo_collection(
            collection,
            window,
            declared_classes,
            "dense_classification",
            target_raster,
            processing,
        )

    # Core 1 begins 32 pixels into this input image. This fragment is halo-only.
    with pytest.raises(GatewayFailure) as failure:
        execute_core_halo_unit(
            request,
            0,
            1,
            limits,
            activation,
            object(),
            object(),
            runner.artifact_files,
            _read_then_return(
                build_feature_collection(
                    [("Tubule", mapping(box(1, 1, 10, 10)))], _CLASSES.values()
                )
            ),
            encode_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
    assert failure.value.error["code"] == "RUNNER_FAILED"
    assert "outside its ownership core" in str(failure.value.__cause__)


def test_unsupported_output_rejected_before_reading_source(tmp_path, monkeypatch):
    request, activation, limits, _, _, runner, _, _ = _case(
        tmp_path, monkeypatch, np.zeros((384, 512), np.int32)
    )
    activation = copy.deepcopy(activation)
    activation["model"]["output"]["type"] = "region_classification"
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: pytest.fail("unsupported task reached source read"),
    )
    with pytest.raises(GatewayFailure) as failure:
        execute_core_halo_unit(
            request,
            0,
            0,
            limits,
            activation,
            object(),
            object(),
            runner.artifact_files,
            lambda *_args: pytest.fail("unsupported task reached prediction"),
            lambda *_args: pytest.fail("unsupported task reached encoding"),
        )
    assert failure.value.error["code"] == "UNSUPPORTED_CAPABILITY"


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_network_maps_fail_without_publishing_a_core(
    tmp_path, monkeypatch, invalid
):
    request, activation, limits, _ = _values()
    request["expected_backend"]["architecture"] = "dinov3-adapter-semantic"
    activation["backend"]["architecture"] = "dinov3-adapter-semantic"
    activation["model"]["output"] = _output()
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )

    def forward(_net, batch):
        output = np.zeros((len(batch), 7, *batch.shape[2:]), np.float32)
        output[:, 2] = invalid  # An invalid emitted-class logit must not win argmax.
        return output, np.zeros((len(batch), 256), np.float32)

    monkeypatch.setattr("histoforge._vendor.histoforge_models.core._forward", forward)
    files = ArtifactFileStore(tmp_path)
    monkeypatch.setattr(
        files,
        "publish_artifact_bytes",
        lambda *_args, **_kwargs: pytest.fail("invalid prediction reached publication"),
    )
    with pytest.raises(GatewayFailure) as failure:
        execute_core_halo_unit(
            request,
            0,
            0,
            limits,
            activation,
            object(),
            object(),
            files,
            _runner()._predict_core_halo_window,
            encode_semantic_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
    assert failure.value.error["code"] == "RUNNER_FAILED"
    assert "finite" in str(failure.value.__cause__)
