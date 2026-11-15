"""Real HoVer unit encoding, final assembly and failure-preserving reuse."""

from copy import deepcopy

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.hovernet import reconciliation
from histoforge.cloud.architectures.registry import processing_result_limits
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.feature_stream import decode_feature_stream
from histoforge.cloud.inference.features import build_feature_collection
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.core_halo_output import encode_hovernet_window
from histoforge.cloud.inference.runs.core_halo_runtime import (
    execute_core_halo_unit,
    finalize_core_halo_input,
)
from histoforge.cloud.inference.runs.core_halo_state import publish_core_halo_unit_state
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
)
from histoforge.cloud.inference.runs.request import validate_model_run_resource_limits
from histoforge.cloud.rasterization.inputs import (
    build_model_raster_input,
    build_raster_frame,
)
from histoforge.cloud.sources.files import SourceFileStore
from shapely.geometry import box, mapping
from shapely.ops import unary_union
from tests.test_core_halo_runtime import _PartitionedRaster, _values


def _case(tmp_path, monkeypatch, *, mask=None, halo=16):
    request, activation, limits, _ = _values(architecture="dinov3-adapter-hovernet")
    request["expected_profile"] = build_core_halo_profile(64, 64, halo, "2" * 64)
    item = request["inputs"][0]
    original = item["raster_input"]
    transform = deepcopy(original["frame"]["transform"])
    transform["raster"].update(width=128, height=128)
    frame = build_raster_frame(
        {
            "source_revision": original["frame"]["source_revision"],
            "region": original["frame"]["region"],
            "transform": transform,
            "inclusion_mask_raster": mapping(
                mask if mask is not None else box(0, 0, 128, 128)
            ),
        }
    )
    item["raster_input"] = build_model_raster_input(
        original["source_view"], frame, original["rasterizer"]
    )
    request = validate_model_run_resource_limits(request, limits)
    plan = build_expected_model_execution_plans(request, limits)[0]
    prepared = _PartitionedRaster(np.zeros((128, 128, 3), np.uint8))
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime._prepare_raster",
        lambda *_args: prepared,
    )
    return request, activation, limits, plan, ArtifactFileStore(tmp_path), prepared


def _publish(case, tmp_path, populations):
    request, activation, limits, plan, files, _prepared = case
    predicted = []

    def predict(read_image, _plan, window, *, channel_percentiles):
        read_image()
        unit = window["unit_id"]
        predicted.append(unit)
        rect = window["input_rect"]
        labels = np.zeros((rect["height"], rect["width"]), np.int32)
        for identifier, bounds in enumerate(populations[unit], 1):
            left, top, right, bottom = bounds
            labels[
                top - rect["y"] : bottom - rect["y"],
                left - rect["x"] : right - rect["x"],
            ] = identifier
        return labels, "Tubule"

    publications = [
        execute_core_halo_unit(
            request,
            0,
            window["unit_id"],
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            files,
            predict,
            encode_hovernet_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
        for window in plan["windows"]
    ]
    return publications, predicted


def _finalize(case, publications):
    request, activation, limits, _plan, files, _prepared = case
    return finalize_core_halo_input(request, 0, limits, publications, activation, files)


def _read(case, reference):
    files = case[4]
    return decode_feature_stream(
        files.read_artifact_block(reference["artifact_id"], 0, reference["size_bytes"]),
        {"Tubule"},
        case[2]["intermediate_result"],
    )


def test_all_complete_copies_survive_unit_encoding_before_one_final_selection(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, halo=48)
    case[0]["result_budget"]["max_features"] = 1
    centers = [(65, 65), (63, 65), (65, 63), (63, 63)]
    populations = [[(x - 10, y - 10, x + 10, y + 10)] for x, y in centers]
    publications, predicted = _publish(case, tmp_path, populations)
    assert predicted == [0, 1, 2, 3]
    assert all(
        _read(case, item["reusable_state"]).feature_count == 1 for item in publications
    )
    # No individual window owns its point, but all four complete observations
    # must remain recoverable. Final-result limits apply after reconciliation.
    result = _finalize(case, publications)
    final = _read(case, result["input"]["feature_stream_artifact"])
    assert result["input"]["feature_count"] == final.feature_count == 1
    assert final.collection == _read(case, publications[0]["reusable_state"]).collection


def test_sparse_mask_cannot_skip_an_observer_that_exposes_a_split(
    tmp_path, monkeypatch
):
    mask = unary_union([box(0, 0, 128, 47), box(100, 40, 128, 128)])
    assert mask.bounds == (0, 0, 128, 128)
    case = _case(tmp_path, monkeypatch, mask=mask)
    populations = [[(58, 44, 64, 54)], [(64, 44, 68, 54)], [(58, 48, 68, 54)], []]
    publications, predicted = _publish(case, tmp_path, populations)
    assert predicted == [0, 1, 2, 3]
    assert len(case[5].reads) == 4
    third = case[3]["windows"][2]["input_rect"]
    assert box(
        third["x"],
        third["y"],
        third["x"] + third["width"],
        third["y"] + third["height"],
    ).disjoint(mask)
    original_ids = set(case[4].artifact_ids())
    for _ in range(2):
        with pytest.raises(GatewayFailure) as failure:
            _finalize(case, publications)
        assert failure.value.error["code"] == "INSUFFICIENT_CONTEXT"
        assert "conflicting object outlines" in failure.value.error["message"]
        assert set(case[4].artifact_ids()) == original_ids
    assert predicted == [0, 1, 2, 3]


def test_local_context_failure_does_not_publish_unit_state(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    request, activation, limits, plan, files, _ = case
    window = plan["windows"][1]

    def predict(read_image, *_, channel_percentiles):
        read_image()
        labels = np.zeros(
            (window["input_rect"]["height"], window["input_rect"]["width"]), np.int32
        )
        # x=48 is an artificial input edge. The shape reaches x=66, so it
        # affects this unit's core even though its point remains outside x>=64.
        labels[20:30, :18] = 1
        return labels, "Tubule"

    with pytest.raises(GatewayFailure) as failure:
        execute_core_halo_unit(
            request,
            0,
            1,
            limits,
            activation,
            object(),
            SourceFileStore(tmp_path),
            files,
            predict,
            encode_hovernet_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
    assert failure.value.error["code"] == "INSUFFICIENT_CONTEXT"
    assert not files.artifact_ids()


def test_final_budget_failure_reuses_full_window_states_without_prediction(
    tmp_path, monkeypatch
):
    case = _case(tmp_path, monkeypatch, halo=48)
    populations = [[(55, 55, 75, 75)]] * 4
    publications, predicted = _publish(case, tmp_path, populations)
    good = _finalize(case, publications)
    reference = good["input"]["feature_stream_artifact"]
    expected = _read(case, reference)
    data = case[4].read_artifact_block(
        reference["artifact_id"], 0, reference["size_bytes"]
    )
    case[0]["result_budget"].update(
        max_transferred_bytes=len(data), max_decoded_bytes=expected.decoded_bytes
    )
    assert (
        _read(
            case, _finalize(case, publications)["input"]["feature_stream_artifact"]
        ).collection
        == expected.collection
    )
    original_ids = set(case[4].artifact_ids())
    case[0]["result_budget"]["max_features"] = 1
    case[0]["result_budget"]["max_vertices"] = 3
    with pytest.raises(GatewayFailure) as failure:
        _finalize(case, publications)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert set(case[4].artifact_ids()) == original_ids
    case[0]["result_budget"]["max_vertices"] = expected.vertex_count
    assert (
        _read(
            case, _finalize(case, publications)["input"]["feature_stream_artifact"]
        ).collection
        == expected.collection
    )
    assert predicted == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "failure_kind", ["missing", "reversed", "local-frame", "corrupt", "aggregate"]
)
def test_incomplete_or_invalid_state_never_publishes_final_output(
    tmp_path, monkeypatch, failure_kind
):
    case = _case(tmp_path, monkeypatch, halo=48)
    publications, _ = _publish(case, tmp_path, [[(55, 55, 75, 75)]] * 4)
    if failure_kind == "missing":
        publications = publications[:-1]
    elif failure_kind == "reversed":
        publications = list(reversed(publications))
    elif failure_kind == "aggregate":
        case[2]["intermediate_result"]["max_features"] = 3
    else:
        reference = publications[-1]["reusable_state"]
        if failure_kind == "corrupt":
            (tmp_path / "files" / reference["artifact_id"] / "data.bin").write_bytes(
                b"x" * reference["size_bytes"]
            )
        else:
            from histoforge.cloud.inference.feature_stream import encode_feature_stream

            # The global target admits this x coordinate; this window's local
            # input has width112 and does not admit x=120.
            encoded = encode_feature_stream(
                build_feature_collection(
                    [("Tubule", mapping(box(120, 20, 124, 24)))], {"Tubule"}
                ),
                {"Tubule"},
            )
            publications[-1]["reusable_state"] = publish_core_halo_unit_state(
                encoded, case[4]
            )
            publications[-1]["execution_identity_digest"] = (
                model_run_execution_identity_digest(case[0], 0, 3, case[2])
            )
    original_ids = set(case[4].artifact_ids())
    with pytest.raises(GatewayFailure) as failure:
        _finalize(case, publications)
    assert (
        failure.value.error["code"]
        == {
            "missing": "RUNNER_FAILED",
            "reversed": "RUNNER_FAILED",
            "local-frame": "RUNNER_FAILED",
            "corrupt": "SERVICE_UNAVAILABLE",
            "aggregate": "RESULT_LIMIT_EXCEEDED",
        }[failure_kind]
    )
    assert set(case[4].artifact_ids()) == original_ids


def test_spatial_work_failure_keeps_all_completed_units(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch, halo=48)
    publications, predicted = _publish(case, tmp_path, [[(55, 55, 75, 75)]] * 4)
    original_ids = set(case[4].artifact_ids())
    reconcile = reconciliation.reconcile_window_predictions
    monkeypatch.setattr(
        reconciliation,
        "reconcile_window_predictions",
        lambda *args: reconcile(*args, max_spatial_query_hits=1),
    )
    with pytest.raises(GatewayFailure) as failure:
        _finalize(case, publications)
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert set(case[4].artifact_ids()) == original_ids
    assert predicted == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "field",
    [
        "max_transferred_bytes",
        "max_decoded_bytes",
        "max_record_bytes",
        "max_features",
        "max_vertices",
    ],
)
def test_full_window_processing_limit_precedes_unit_publication(
    tmp_path, monkeypatch, field
):
    case = _case(tmp_path, monkeypatch)
    request, activation, limits, plan, files, _ = case
    window = plan["windows"][0]
    labels = np.zeros((80, 80), np.int32)
    labels[4:8, 4:8] = 1
    labels[4:8, 68:72] = 2  # Complete halo candidate, not owned by this core.
    processing = processing_result_limits("dinov3-adapter-hovernet")
    expected = encode_hovernet_window(
        (labels, "Tubule"),
        window,
        {"Tubule"},
        {"width": 128, "height": 128},
        processing,
    )
    decoded = decode_feature_stream(expected.data, {"Tubule"}, processing)
    ceiling = {
        "max_transferred_bytes": len(expected.data),
        "max_decoded_bytes": expected.decoded_bytes,
        "max_record_bytes": max(
            len(canonical_bytes(item)) for item in decoded.collection["features"]
        ),
        "max_features": 2,
        "max_vertices": expected.vertex_count,
    }[field]
    processing[field] = ceiling
    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.core_halo_runtime.processing_result_limits",
        lambda _architecture: processing,
    )

    def predict(read_image, *_args, channel_percentiles):
        read_image()
        return labels, "Tubule"

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
            predict,
            encode_hovernet_window,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )

    publication = run()
    assert _read(case, publication["reusable_state"]).feature_count == 2
    original_ids = set(files.artifact_ids())
    processing[field] -= 1
    with pytest.raises(GatewayFailure) as failure:
        run()
    assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert field in failure.value.error["message"]
    assert set(files.artifact_ids()) == original_ids
