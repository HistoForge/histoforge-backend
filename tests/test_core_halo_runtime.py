"""Durable CoreHalo unit execution and immutable-state tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.registry import processing_result_limits
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt, ArtifactFileStore
from histoforge.cloud.inference.feature_stream import (
    collection_budget_for_stream,
    decode_feature_stream,
    encode_canonical_feature_records,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import (
    ResultLimitExceeded,
    build_bounded_feature_collection,
    build_feature_collection,
    validate_feature_records,
)
from histoforge.cloud.inference.mask_features import feature_collection_from_mask
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.core_halo_output import (
    encode_core_halo_collection,
    encode_instance_window,
)
from histoforge.cloud.inference.runs.core_halo_runtime import (
    execute_core_halo_unit,
    finalize_core_halo_input,
)
from histoforge.cloud.inference.runs.core_halo_state import (
    merge_core_halo_unit_states,
    publish_core_halo_unit_state,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
)
from histoforge.cloud.inference.runs.request import validate_model_run_resource_limits
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.sources.files import SourceFileStore

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class _PartitionedRaster:
    def __init__(self, image):
        self.image = image
        self.support = np.ones(image.shape[:2], bool)
        self.support[:, -7:] = False
        self.reads = []

    def read(self, rectangle):
        self.reads.append(copy.deepcopy(rectangle))
        x = rectangle["x"]
        y = rectangle["y"]
        region = (slice(y, y + rectangle["height"]), slice(x, x + rectangle["width"]))
        return RasterizedImage(self.image[region], self.support[region])


def _encode_instance_collection(
    collection, window, declared_classes, target_raster, processing_limits
):
    """Keep ownership/encoding real when a test supplies polygonized candidates."""

    return encode_core_halo_collection(
        collection,
        window,
        declared_classes,
        "instance_segmentation",
        target_raster,
        processing_limits,
    )


def _read_then_return(prediction):
    """Read the admitted window before returning isolated geometry fixtures."""

    def predict(read_window_image, _plan, _window, *, channel_percentiles=None):
        read_window_image()
        return prediction

    return predict


def test_unit_state_merge_requires_exact_order_and_preserves_features(tmp_path):
    request, _activation, limits, plan = _values()
    files = ArtifactFileStore(tmp_path)
    publications = _publish_units(request, limits, files)

    encoded = _merge(publications, request, limits, plan, files)
    decoded = decode_feature_stream(encoded.data, {"Tubule"}, request["result_budget"])

    assert decoded.feature_count == 2
    assert {
        _minimum_x(item["geometry"]) for item in decoded.collection["features"]
    } == {
        4,
        260,
    }
    with pytest.raises(ValueError, match="exactly cover"):
        _merge(publications[:1], request, limits, plan, files)
    with pytest.raises(ValueError, match="exactly cover"):
        _merge(list(reversed(publications)), request, limits, plan, files)


def test_unit_state_merge_rejects_corruption_and_aggregate_budget(tmp_path):
    request, _activation, limits, plan = _values()
    files = ArtifactFileStore(tmp_path)
    publications = _publish_units(request, limits, files)
    bad_media = copy.deepcopy(publications)
    bad_media[0]["reusable_state"]["media_type"] = "application/json"
    with pytest.raises(ValueError, match="media contract"):
        _merge(bad_media, request, limits, plan, files)

    reference = publications[0]["reusable_state"]
    data_path = tmp_path / "files" / reference["artifact_id"] / "data.bin"
    data_path.write_bytes(b"x" * reference["size_bytes"])
    with pytest.raises(ArtifactBytesCorrupt, match="digest"):
        _merge(publications, request, limits, plan, files)

    one_feature_budget = copy.deepcopy(request["result_budget"])
    one_feature_budget["max_features"] = 1
    request, _activation, limits, plan = _values(one_feature_budget)
    files = ArtifactFileStore(tmp_path / "aggregate")
    publications = _publish_units(request, limits, files)
    with pytest.raises(ResultLimitExceeded, match="max_features"):
        _merge(publications, request, limits, plan, files)


def test_merge_limits_actual_final_stream_not_collection_wrapper(tmp_path):
    request, _activation, limits, plan = _values()
    files = ArtifactFileStore(tmp_path)
    publications = _publish_units(request, limits, files)
    collection = build_feature_collection(
        [("Tubule", _geometry(x, 4)) for x in (4, 260)], {"Tubule"}
    )
    expected = encode_feature_stream(collection, {"Tubule"})
    stream_sizes = {
        "max_transferred_bytes": len(expected.data),
        "max_decoded_bytes": expected.decoded_bytes,
    }
    assert len(canonical_bytes(collection)) > max(stream_sizes.values())
    request["result_budget"].update(stream_sizes)

    assert _merge(publications, request, limits, plan, files) == expected
    for field, size in stream_sizes.items():
        too_small = copy.deepcopy(request)
        too_small["result_budget"][field] = size - 1
        with pytest.raises(ResultLimitExceeded, match=field):
            _merge(publications, too_small, limits, plan, files)


def test_runtime_reads_raw_bounded_windows_and_merges_owned_results(
    tmp_path, monkeypatch
):
    request, activation, limits, plan = _values()
    y, x = np.indices((384, 512), dtype=np.uint16)
    image = np.stack(
        (
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=-1,
    )
    prepared = _PartitionedRaster(image)
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )

    observed = []

    def predict_window(read_window_image, _plan, window, *, channel_percentiles):
        raw = read_window_image()
        observed.append((raw.pixels.shape, raw.pixels.dtype))
        input_rect = window["input_rect"]
        region = (
            slice(input_rect["y"], input_rect["y"] + input_rect["height"]),
            slice(input_rect["x"], input_rect["x"] + input_rect["width"]),
        )
        np.testing.assert_array_equal(raw.pixels, image[region])
        np.testing.assert_array_equal(raw.real_pixel_mask, prepared.support[region])
        core_rect = window["core_rect"]
        local_x = core_rect["x"] - input_rect["x"] + 4
        local_y = core_rect["y"] - input_rect["y"] + 4
        return build_feature_collection(
            [("Tubule", _geometry(local_x, local_y))], {"Tubule"}
        )

    def encode_window(collection, window, declared_classes, target_raster, processing):
        observed[-1] += (collection_budget_for_stream(processing),)
        return _encode_instance_collection(
            collection, window, declared_classes, target_raster, processing
        )

    files = ArtifactFileStore(tmp_path)
    publications = [
        execute_core_halo_unit(
            request,
            0,
            unit_id,
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            files,
            predict_window,
            encode_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
        for unit_id in range(2)
    ]
    final = finalize_core_halo_input(
        request, 0, limits, publications, activation, files
    )
    reference = final["input"]["feature_stream_artifact"]
    encoded = files.read_artifact_block(
        reference["artifact_id"], 0, reference["size_bytes"]
    )
    decoded = decode_feature_stream(encoded, {"Tubule"}, request["result_budget"])

    assert final["input"]["execution_plan"] == plan
    assert final["input"]["feature_count"] == 2
    assert final["decoded_bytes"] == decoded.decoded_bytes
    assert {
        _minimum_x(item["geometry"]) for item in decoded.collection["features"]
    } == {
        4,
        260,
    }
    assert prepared.reads == [
        plan["windows"][0]["input_rect"],
        plan["windows"][1]["input_rect"],
    ]
    assert observed == [
        (
            (
                plan["windows"][unit_id]["input_rect"]["height"],
                plan["windows"][unit_id]["input_rect"]["width"],
                3,
            ),
            np.dtype(np.uint8),
            collection_budget_for_stream(
                processing_result_limits(request["expected_backend"]["architecture"])
            ),
        )
        for unit_id in range(2)
    ]


def test_runtime_rejects_owned_geometry_on_an_artificial_edge(tmp_path, monkeypatch):
    request, activation, limits, _plan = _values()
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )

    def edge_touching(read_window_image, _plan, _window, *, channel_percentiles):
        read_window_image()
        # Unit 1 has input x=224 and ownership core x=256.  The translated
        # interior point belongs to unit 1, but the geometry reaches the artificial
        # left input edge, so publishing it would silently clip an instance.
        return build_feature_collection(
            [("Tubule", _geometry(0, 4, width=70))], {"Tubule"}
        )

    with pytest.raises(GatewayFailure) as failure:
        execute_core_halo_unit(
            request,
            0,
            1,
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            ArtifactFileStore(tmp_path),
            edge_touching,
            _encode_instance_collection,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
    assert failure.value.error["code"] == "INSUFFICIENT_CONTEXT"


def test_core_candidates_and_ownership_accept_exact_processing_stream_limits(
    tmp_path, monkeypatch
):
    request, activation, limits, _plan = _values()
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )
    labels = np.zeros((34, 49), dtype=np.int32)
    labels[4:29, 4:44] = np.arange(1, 1001, dtype=np.int32).reshape(25, 40)
    expected = encode_feature_stream(
        feature_collection_from_mask(labels, "Tubule", {"Tubule"}), {"Tubule"}
    )
    processing = processing_result_limits(request["expected_backend"]["architecture"])
    processing.update(
        max_transferred_bytes=len(expected.data),
        max_decoded_bytes=expected.decoded_bytes,
    )
    files = ArtifactFileStore(tmp_path)
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime.processing_result_limits",
        lambda _architecture: processing,
    )

    def predict_window(read_window_image, _plan, _window, *, channel_percentiles):
        read_window_image()
        return labels, "Tubule"

    def run(current_limits):
        return execute_core_halo_unit(
            request,
            0,
            0,
            current_limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            files,
            predict_window,
            encode_instance_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )

    publication = run(limits)
    reference = publication["reusable_state"]
    assert (
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"])
        == expected.data
    )
    for field in ("max_transferred_bytes", "max_decoded_bytes"):
        processing[field] -= 1
        with pytest.raises(GatewayFailure, match=field) as failure:
            run(limits)
        assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
        processing[field] += 1


def test_core_recovery_file_limit_applies_before_publication(tmp_path, monkeypatch):
    request, activation, limits, _plan = _values()
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )
    collection = build_feature_collection([("Tubule", _geometry(4, 4))], {"Tubule"})
    expected = encode_feature_stream(collection, {"Tubule"})
    files = ArtifactFileStore(tmp_path)

    def run():
        return execute_core_halo_unit(
            request,
            0,
            0,
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            files,
            _read_then_return(collection),
            _encode_instance_collection,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )

    limits["max_recovery_state_bytes_per_unit"] = len(expected.data)
    assert run()["reusable_state"]["size_bytes"] == len(expected.data)
    original_ids = set(files.artifact_ids())
    limits["max_recovery_state_bytes_per_unit"] -= 1
    with pytest.raises(GatewayFailure, match="max_transferred_bytes") as failure:
        run()
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert set(files.artifact_ids()) == original_ids


def test_custom_halo_processing_limit_fails_before_source_preparation(
    tmp_path, monkeypatch
):
    request, activation, limits, _plan = _values()
    request["halo_px_override"] = 64
    limits["processing_raster"] = {
        "max_width": 288,
        "max_height": 384,
        "max_pixels": 288 * 384,
    }

    def unexpected_source_read(*_args):
        pytest.fail("processing allocation must be admitted before source preparation")

    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        unexpected_source_read,
    )
    with pytest.raises(ValueError, match="processing_raster"):
        execute_core_halo_unit(
            request,
            0,
            0,
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            ArtifactFileStore(tmp_path),
            unexpected_source_read,
            unexpected_source_read,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )


def test_halo_candidates_use_processing_not_final_feature_limit(tmp_path, monkeypatch):
    request, activation, limits, _plan = _values()
    request["result_budget"]["max_features"] = 1
    prepared = _PartitionedRaster(np.zeros((384, 512, 3), dtype=np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )

    def encode_window(_prediction, window, declared_classes, target_raster, processing):
        # Both candidates are inside the processing image. Only the first
        # belongs to core 0; the second is context from the neighboring core.
        collection = build_bounded_feature_collection(
            [("Tubule", _geometry(4, 4)), ("Tubule", _geometry(270, 4))],
            declared_classes,
            collection_budget_for_stream(processing),
        )
        return _encode_instance_collection(
            collection, window, declared_classes, target_raster, processing
        )

    files = ArtifactFileStore(tmp_path)
    publication = execute_core_halo_unit(
        request,
        0,
        0,
        limits,
        activation,
        object(),
        SourceFileStore(tmp_path),
        files,
        _read_then_return(None),
        encode_window,
        channel_percentiles=[[0.0, 255.0]] * 3,
    )
    reference = publication["reusable_state"]
    decoded = decode_feature_stream(
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"]),
        {"Tubule"},
        request["result_budget"],
    )
    assert decoded.feature_count == 1
    assert _minimum_x(decoded.collection["features"][0]["geometry"]) == 4


def test_runtime_rejects_large_processing_image_before_reading_source(
    tmp_path, monkeypatch
):
    request, activation, limits, _plan = _values()
    limits["processing_raster"]["max_width"] = 287
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: pytest.fail("oversized processing image reached source reading"),
    )
    with pytest.raises(ValueError, match="processing window exceeds"):
        execute_core_halo_unit(
            request,
            0,
            0,
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            ArtifactFileStore(tmp_path),
            lambda *_args: pytest.fail("oversized image reached prediction"),
            lambda *_args: pytest.fail("oversized image reached encoding"),
            channel_percentiles=[[0.0, 255.0]] * 3,
        )


@pytest.mark.parametrize("field", ["max_features", "max_vertices", "max_decoded_bytes"])
def test_intermediate_geometry_limits_apply_before_final_assembly(tmp_path, field):
    request, _activation, limits, plan = _values()
    files = ArtifactFileStore(tmp_path)
    publications = _publish_units(request, limits, files)
    reference = publications[0]["reusable_state"]
    first = decode_feature_stream(
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"]),
        {"Tubule"},
        limits["intermediate_result"],
    )
    # One completed core fits; collecting the second exceeds the intermediate
    # cap, even though the client's final-result budget would allow both.
    limits["intermediate_result"][field] = {
        "max_features": first.feature_count,
        "max_vertices": first.vertex_count,
        "max_decoded_bytes": first.decoded_bytes,
    }[field]
    with pytest.raises(ResultLimitExceeded, match=field):
        _merge(publications, request, limits, plan, files)


def test_intermediate_artifact_bytes_checked_before_loading(tmp_path, monkeypatch):
    request, _activation, limits, plan = _values()
    files = ArtifactFileStore(tmp_path)
    publications = _publish_units(request, limits, files)
    limits["intermediate_result"]["max_transferred_bytes"] = (
        sum(item["reusable_state"]["size_bytes"] for item in publications) - 1
    )
    monkeypatch.setattr(
        files,
        "read_artifact_block",
        lambda *_args: pytest.fail("oversized intermediate artifacts were loaded"),
    )
    with pytest.raises(ResultLimitExceeded, match="intermediate max_transferred_bytes"):
        _merge(publications, request, limits, plan, files)


def _publish_units(request, limits, files):
    publications = []
    for unit_id, x in enumerate((4, 260)):
        collection = build_feature_collection([("Tubule", _geometry(x, 4))], {"Tubule"})
        reference = publish_core_halo_unit_state(
            encode_canonical_feature_records(
                validate_feature_records(
                    collection["features"], {"Tubule"}
                ).encoded_records(),
                limits["intermediate_result"],
            ),
            files,
        )
        publications.append(
            {
                "input_id": 0,
                "unit_id": unit_id,
                "reusable_state": reference,
                "execution_identity_digest": model_run_execution_identity_digest(
                    request, 0, unit_id, limits
                ),
            }
        )
    return publications


def _merge(publications, request, limits, plan, files):
    return merge_core_halo_unit_states(
        publications,
        request=request,
        input_id=0,
        plan=plan,
        model_run_limits=limits,
        declared_classes={"Tubule"},
        output_type="instance_segmentation",
        result_budget=request["result_budget"],
        target_raster=request["inputs"][0]["raster_input"]["frame"]["transform"][
            "raster"
        ],
        files=files,
    )


def _values(result_budget=None, *, architecture="dinov3-adapter-cellpose"):
    finite_case = _fixture("finite-inference-v1.json")["cases"][0]
    finite = json.loads(finite_case["requestCanonical"])
    backend = copy.deepcopy(finite["expected_backend"])
    # These shared fixtures describe globally translated, already-owned unit
    # records. HoVer's new full-window format has separate explicit fixtures.
    backend["architecture"] = architecture
    profile = build_core_halo_profile(256, 384, 32, "2" * 64)
    model_case = _fixture("model-run-request-v1.json")["cases"][0]
    region = finite["raster_input"]["frame"]["region"]
    request = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": model_case["actionId"],
        "model_run_id": model_case["modelRunId"],
        "producer": copy.deepcopy(finite["producer"]),
        "expected_backend": backend,
        "expected_profile": profile,
        "inputs": [
            {
                "input_id": 0,
                "output_generation_id": model_case["outputGenerationId"],
                "output": {
                    "producer": copy.deepcopy(finite["producer"]),
                    "scope": {"type": "field", "region": copy.deepcopy(region)},
                },
                "raster_input": copy.deepcopy(finite["raster_input"]),
            }
        ],
        "result_budget": copy.deepcopy(result_budget or finite["result_budget"]),
    }
    limits = copy.deepcopy(_fixture("model-run-request-v1.json")["testModelRunLimits"])
    request = validate_model_run_resource_limits(request, limits)
    activation = {
        "backend": backend,
        "capabilities": {
            "runner_build": backend["runner_build"],
            "rasterizer": backend["rasterizer"],
            "preprocessing_contract_digest": backend["preprocessing_contract_digest"],
            "source_containers": ["PNG", "TIFF_FAMILY"],
        },
        "model": {
            "output": {
                "type": "instance_segmentation",
                "classes": ["Tubule"],
            }
        },
    }
    plan = build_expected_model_execution_plans(request, limits)[0]
    assert len(plan["windows"]) == 2
    return request, activation, limits, plan


def _geometry(x, y, *, width=4, height=4):
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [x, y],
                [x + width, y],
                [x + width, y + height],
                [x, y + height],
                [x, y],
            ]
        ],
    }


def _minimum_x(geometry):
    polygons = (
        [geometry["coordinates"]]
        if geometry["type"] == "Polygon"
        else geometry["coordinates"]
    )
    return min(point[0] for polygon in polygons for ring in polygon for point in ring)


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
