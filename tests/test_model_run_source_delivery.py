"""Accepted image assignments survive failure, reuse and completed-input cleanup."""

import copy
from datetime import timedelta

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure, build_gateway_error
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.contract import build_backend_input_identity
from histoforge.cloud.inference.feature_stream import encode_feature_stream
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
    validate_model_input_result,
)
from histoforge.cloud.inference.runs.source_bindings import (
    input_source_bindings,
    processing_rectangle,
)
from histoforge.cloud.inference.runs.store import ModelRunStore
from histoforge.cloud.rasterization.captured_source import captured_source_footprint
from histoforge.cloud.rasterization.inputs import build_model_raster_input
from histoforge.cloud.sources.capture import (
    build_captured_source_view,
    build_source_capture,
)

from test_model_run_store import _artifact_json, _Clock, _limits, _request, _reserve


def test_accept_before_upload_and_bind_without_changing_plan_or_request(tmp_path):
    request = _captured_request()
    files, store, clock = _store(tmp_path)
    accepted, dispatch = _reserve(store, request, _limits(), 600, files)
    assert dispatch
    initial = store.source_delivery(request["model_run_id"], files)
    assert initial["source_bindings"] == []
    assert initial["request_digest"] == accepted["request_digest"]
    assert initial["active"]
    context = store.begin_execution(request["model_run_id"], files)
    assert context["source_bindings"] == []
    required_until = []

    def validate_source(submitted, item, deadline):
        assert submitted == request
        assert item == _binding(request, item["unit_id"])
        required_until.append(deadline)

    first = _bind(store, files, request, 0, validate_source)
    with pytest.raises(ValueError, match="every input source"):
        model_run_execution_identity_digest(
            request, first["input_id"], 0, _limits(), [first]
        )
    clock.advance(seconds=120)
    second = _bind(store, files, request, 1, validate_source)
    delivery = store.source_delivery(request["model_run_id"], files)
    assert delivery["execution_plans"] == initial["execution_plans"]
    assert delivery["source_bindings"] == [first, second]
    assert delivery["request_digest"] == initial["request_digest"]
    assert (
        len(
            model_run_execution_identity_digest(
                request, first["input_id"], 0, _limits(), [first, second]
            )
        )
        == 64
    )
    assert required_until[0] == required_until[1]
    assert required_until[0] == clock.value + timedelta(seconds=21600 - 120)
    status = store.status(request["model_run_id"], files)
    assert _artifact_json(files, status["request_artifact"]) == request


@pytest.mark.parametrize("termination", ["failure", "cancel", "deadline"])
def test_exact_binding_confirmation_survives_termination(tmp_path, termination):
    request = _captured_request()
    files, store, clock = _store(tmp_path)
    _reserve(store, request, _limits(), 600, files)
    binding = _bind(store, files, request, 0)
    if termination == "failure":
        store.fail(request["model_run_id"], _failure(), files)
    elif termination == "cancel":
        store.cancel(request["model_run_id"], files)
    else:
        clock.advance(seconds=21600)

    assert (
        store.bind_source(
            request["model_run_id"],
            binding,
            files,
            require_source=lambda *_: pytest.fail(
                "accepted binding must not recheck upload"
            ),
        )
        == binding
    )
    delivery = store.source_delivery(request["model_run_id"], files)
    assert not delivery["active"]
    assert delivery["source_bindings"] == [binding]
    # A binding is not completed computation and therefore does not create PARTIAL.
    status = store.status(request["model_run_id"], files)
    assert (
        status["phase"]
        == {"failure": "FAILED", "cancel": "CANCELLED", "deadline": "TIMED_OUT"}[
            termination
        ]
    )
    assert "terminal_manifest" not in status
    with pytest.raises(GatewayFailure, match="cannot accept another source"):
        _bind(store, files, request, 1)


def test_bound_assignment_cannot_be_replaced_and_checks_before_acceptance(tmp_path):
    request = _captured_request()
    files, store, _ = _store(tmp_path)
    _reserve(store, request, _limits(), 600, files)
    binding = _binding(request, 0)

    def expired(*_):
        raise GatewayFailure("ARTIFACT_PRECONDITION_FAILED", "image expires too soon")

    with pytest.raises(GatewayFailure):
        _bind(store, files, request, 0, expired)
    assert (
        store.source_delivery(request["model_run_id"], files)["source_bindings"] == []
    )
    _bind(store, files, request, 0)
    changed = copy.deepcopy(binding)
    changed["source"]["artifact"]["bytes_digest"] = "0" * 64
    with pytest.raises(GatewayFailure, match="binding differs"):
        store.bind_source(
            request["model_run_id"], changed, files, require_source=expired
        )
    invalid = _binding(request, 1)
    invalid["source"]["source_rect_at_level"]["width"] -= 1
    with pytest.raises(ValueError, match="omits pixels"):
        store.bind_source(
            request["model_run_id"], invalid, files, require_source=expired
        )


def test_bindings_outlive_unit_records_and_do_not_own_source_file_deletion(tmp_path):
    request = _captured_request()
    files, store, clock = _store(tmp_path)
    _reserve(store, request, _limits(), 2, files)
    bindings = [_bind(store, files, request, unit) for unit in (0, 1)]
    for unit in (0, 1):
        store.publish_unit(
            request["model_run_id"], _unit(request, files, bindings, unit), files
        )
    result = _input_result(request, files, bindings)
    store.publish_input(request["model_run_id"], result, 0, files)
    context = store.begin_execution(request["model_run_id"], files)
    assert context["completed_units"] == []
    assert context["source_bindings"] == bindings
    assert store.complete(request["model_run_id"], files)
    manifest = _artifact_json(
        files, store.status(request["model_run_id"], files)["terminal_manifest"]
    )
    assert manifest["source_bindings"] == bindings
    assert manifest["completed_inputs"] == [result]
    for binding in bindings:
        assert store.has_source_reference(binding["source"]["artifact"]["artifact_id"])
        assert binding["source"]["artifact"]["artifact_id"] not in files.artifact_ids()
    clock.advance(seconds=2)
    store.cleanup_expired(files)
    assert files.artifact_ids() == set()
    assert not store.has_source_reference(
        bindings[0]["source"]["artifact"]["artifact_id"]
    )


@pytest.mark.parametrize("complete", [False, True])
def test_replacement_retains_unfinished_bindings_and_validates_new_lifetime(
    tmp_path, complete
):
    request = _captured_request()
    files, store, clock = _store(tmp_path)
    _reserve(store, request, _limits(), 600, files)
    bindings = [_bind(store, files, request, unit) for unit in (0, 1)]
    first_unit = _unit(request, files, bindings, 0)
    store.publish_unit(request["model_run_id"], first_unit, files)
    if complete:
        store.publish_unit(
            request["model_run_id"], _unit(request, files, bindings, 1), files
        )
        result = _input_result(request, files, bindings)
        store.publish_input(request["model_run_id"], result, 0, files)
    store.fail(request["model_run_id"], _failure(), files)
    manifest = _artifact_json(
        files, store.status(request["model_run_id"], files)["terminal_manifest"]
    )
    assert manifest["source_bindings"] == bindings
    replacement = copy.deepcopy(request)
    replacement["model_run_id"] = "99999999-9999-4999-8999-999999999999"
    replacement["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": manifest["manifest_digest"],
    }
    clock.advance(seconds=300)
    deadlines = []

    def require_live(submitted, restored, deadline):
        assert submitted == replacement
        assert restored == bindings
        deadlines.append(deadline)

    store.reserve(
        replacement,
        _limits(),
        600,
        files,
        require_sources_live_through=lambda *_: None,
        require_bindings_live_through=require_live,
    )
    assert deadlines == [clock.value + timedelta(seconds=21600)]
    restored = store.begin_execution(replacement["model_run_id"], files)
    assert restored["source_bindings"] == bindings
    assert restored["completed_units"] == ([] if complete else [first_unit])
    assert restored["completed_input_ids"] == (
        [bindings[0]["input_id"]] if complete else []
    )


def test_complete_result_requires_all_sources_not_only_the_first(tmp_path):
    request = _captured_request()
    files, _, _ = _store(tmp_path)
    bindings = [_binding(request, unit) for unit in (0, 1)]
    result = _input_result(request, files, bindings)
    assert validate_model_input_result(result, request, _limits(), bindings) == result
    with pytest.raises(ValueError):
        validate_model_input_result(result, request, _limits(), bindings[:1])
    changed = copy.deepcopy(bindings)
    changed[1]["source"]["artifact"]["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_model_input_result(result, request, _limits(), changed)


def _captured_request(core_halo=True):
    request = _request(1 if core_halo else 0)
    item = request["inputs"][0]
    raster = item["raster_input"]
    frame = raster["frame"]
    source = frame["source_revision"]["pixel_source"]
    rectangle = {"x": 0, "y": 0, "width": source["width"], "height": source["height"]}
    identity = {"scale_x": 1, "scale_y": 1, "translate_x": 0, "translate_y": 0}
    capture = build_source_capture(
        {
            "source": source,
            "plane": frame["transform"]["raster"]["plane"],
            "source_level": {
                "series_index": 0,
                "resolution_level": 0,
                "width": source["width"],
                "height": source["height"],
            },
            "source_level_pixels_to_level_zero": identity,
            "level_zero_pixels_to_slide": identity,
            "source_rect_at_level": rectangle,
            "writer_build": request["expected_backend"]["rasterizer"]["build"],
        }
    )
    view = build_captured_source_view(capture, rectangle)
    for _ in range(8):
        required = captured_source_footprint(view, frame["transform"])
        if required == view["pixel_rect"]:
            break
        view = build_captured_source_view(capture, required)
    else:
        raise AssertionError("fixture view failed to converge")
    item["raster_input"] = build_model_raster_input(view, frame, raster["rasterizer"])
    if core_halo:
        request["expected_profile"] = build_core_halo_profile(256, 384, 16, "a" * 64)
    return request


def _binding(request, unit_id):
    model_input = request["inputs"][0]
    raster = model_input["raster_input"]
    plan = build_expected_model_execution_plans(request, _limits())[
        model_input["input_id"]
    ]
    rectangle = captured_source_footprint(
        raster["source_view"],
        raster["frame"]["transform"],
        processing_rectangle(plan, unit_id),
    )
    artifact = copy.deepcopy(
        _request(0)["inputs"][0]["raster_input"]["source_view"]["artifact"]
    )
    artifact["artifact_id"] = f"22222222-2222-4222-8222-{unit_id:012d}"
    return {
        "input_id": model_input["input_id"],
        "unit_id": unit_id,
        "source": {
            "capture_digest": raster["source_view"]["capture"]["capture_digest"],
            "artifact": artifact,
            "source_rect_at_level": rectangle,
        },
    }


def _bind(store, files, request, unit_id, validator=lambda *_: None):
    return store.bind_source(
        request["model_run_id"],
        _binding(request, unit_id),
        files,
        require_source=validator,
    )


def _unit(request, files, bindings, unit_id):
    input_id = request["inputs"][0]["input_id"]
    return {
        "input_id": input_id,
        "unit_id": unit_id,
        "reusable_state": files.publish_artifact_bytes(
            canonical_bytes({"unit": unit_id}), media_type="application/octet-stream"
        ),
        "execution_identity_digest": model_run_execution_identity_digest(
            request, input_id, unit_id, _limits(), bindings
        ),
    }


def _input_result(request, files, bindings):
    item = request["inputs"][0]
    stream = encode_feature_stream(
        {"type": "FeatureCollection", "features": []}, {"Tubule"}
    )
    return {
        "input_id": item["input_id"],
        "input_identity": build_backend_input_identity(
            item["raster_input"], input_source_bindings(bindings, item["input_id"])
        ),
        "execution_plan": build_expected_model_execution_plans(request, _limits())[
            item["input_id"]
        ],
        "feature_stream_artifact": files.publish_artifact_bytes(
            stream.data, media_type="application/geo+json-seq", content_encoding="gzip"
        ),
        "feature_count": 0,
        "vertex_count": 0,
    }


def _store(tmp_path):
    clock = _Clock()
    return (
        ArtifactFileStore(tmp_path),
        ModelRunStore({}, max_active_runs=4, clock=clock),
        clock,
    )


def _failure():
    return build_gateway_error("SERVICE_UNAVAILABLE", "image delivery failed")
