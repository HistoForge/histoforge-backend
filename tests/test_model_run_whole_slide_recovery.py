"""512-core metadata recovery without slide reads or model prediction."""

import copy
from unittest.mock import patch

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest
from histoforge.cloud._errors import build_gateway_error
from histoforge.cloud.architectures.registry import runner_registry
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.contract import build_backend_input_identity
from histoforge.cloud.inference.feature_stream import encode_feature_stream
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs import publications
from histoforge.cloud.inference.runs.core_halo_state import merge_core_halo_unit_states
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digests,
)
from histoforge.cloud.inference.runs.records import validate_model_run_record
from histoforge.cloud.inference.runs.reuse import select_reusable_model_run_state
from histoforge.cloud.inference.runs.source_bindings import (
    input_source_bindings,
    processing_rectangle,
)
from histoforge.cloud.inference.runs.store import ModelRunStore
from histoforge.cloud.rasterization.captured_source import captured_source_footprint
from histoforge.cloud.rasterization.inputs import (
    build_model_raster_input,
    build_raster_frame,
)
from histoforge.cloud.sources.capture import (
    build_captured_source_view,
    build_source_capture,
)

from test_model_run_store import _artifact_json, _Clock, _request, _reserve


@pytest.fixture(scope="module")
def whole_slide_metadata():
    """Use the AWS slide's dimensions, not its pixels or biological predictions."""

    request = _request(1)
    raster_input = request["inputs"][0]["raster_input"]
    old_frame = raster_input["frame"]
    source = copy.deepcopy(old_frame["source_revision"]["pixel_source"])
    source.update(width=126976, height=253952)
    source["calibration"] = {"mpp_x": 0.22220247089147632, "mpp_y": 0.22220247089147632}
    revision = copy.deepcopy(old_frame["source_revision"])
    revision["pixel_source"] = source
    revision["image_data_revision"]["image_source_digest"] = canonical_digest(source)
    width, height = 128248, 256495
    transform = copy.deepcopy(old_frame["transform"])
    transform["raster"].update(width=width, height=height)
    scale = transform["raster"]["target_mpp"] / source["calibration"]["mpp_x"]
    transform["raster_to_slide"] = {
        "scale_x": scale,
        "scale_y": scale,
        "translate_x": 0,
        "translate_y": 0,
    }
    frame = build_raster_frame(
        {
            "source_revision": revision,
            "region": old_frame["region"],
            "transform": transform,
            "inclusion_mask_raster": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
                ],
            },
        }
    )
    rectangle = {"x": 0, "y": 0, "width": source["width"], "height": source["height"]}
    identity = {"scale_x": 1, "scale_y": 1, "translate_x": 0, "translate_y": 0}
    capture = build_source_capture(
        {
            "source": source,
            "plane": transform["raster"]["plane"],
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
    request["inputs"][0]["raster_input"] = build_model_raster_input(
        view, frame, raster_input["rasterizer"]
    )
    request["expected_profile"] = build_core_halo_profile(8192, 8192, 256, "a" * 64)
    limits = copy.deepcopy(
        runner_registry()["definitions"]["dinov3-adapter-cellpose"]["capabilities"][
            "durable_run"
        ]["run"]
    )
    plan = build_expected_model_execution_plans(request, limits)[
        request["inputs"][0]["input_id"]
    ]
    artifact = _request(0)["inputs"][0]["raster_input"]["source_view"]["artifact"]
    bindings = [
        {
            "input_id": request["inputs"][0]["input_id"],
            "unit_id": unit_id,
            "source": {
                "capture_digest": capture["capture_digest"],
                "artifact": {
                    **artifact,
                    "artifact_id": f"22222222-2222-4222-8222-{unit_id:012d}",
                },
                "source_rect_at_level": captured_source_footprint(
                    view, transform, processing_rectangle(plan, unit_id)
                ),
            },
        }
        for unit_id in range(len(plan["windows"]))
    ]
    assert len(bindings) == 512
    return request, limits, plan, bindings


def test_512_sources_partial_failure_replacement_and_complete_input(
    tmp_path, whole_slide_metadata
):
    request, limits, plan, bindings = copy.deepcopy(whole_slide_metadata)
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=4, clock=_Clock())
    _reserve(store, request, limits, 600, files)
    record_key, record = next(iter(values.items()))
    record = copy.deepcopy(record)
    record.update(phase="RUNNING", source_bindings=bindings)
    input_id = request["inputs"][0]["input_id"]
    identities = model_run_execution_identity_digests(
        request, [(input_id, unit_id) for unit_id in range(32)], limits, bindings
    )
    record["completed_units"] = [
        {
            "input_id": input_id,
            "unit_id": unit_id,
            "reusable_state": files.publish_artifact_bytes(
                b"unit", media_type="application/octet-stream"
            ),
            "execution_identity_digest": identities[(input_id, unit_id)],
        }
        for unit_id in range(32)
    ]
    with patch.object(
        publications, "_publication_context", wraps=publications._publication_context
    ) as context:
        with patch.object(
            publications,
            "validate_run_source_bindings",
            wraps=publications.validate_run_source_bindings,
        ) as sources:
            validated = validate_model_run_record(record)
    assert context.call_count == sources.call_count == 1
    assert canonical_bytes(validated) == canonical_bytes(record)
    values[record_key] = validated
    failure = build_gateway_error(
        "INSUFFICIENT_CONTEXT", "owned instance touches an artificial processing edge"
    )
    store.fail(request["model_run_id"], failure, files)
    status = store.status(request["model_run_id"], files)
    assert status["phase"] == "PARTIAL"
    assert status["completed_unit_count"] == 32
    assert status["total_unit_count"] == 512
    assert status["failure"] == failure
    manifest = _artifact_json(files, status["terminal_manifest"])
    assert len(manifest["source_bindings"]) == 512
    replacement = copy.deepcopy(request)
    replacement["model_run_id"] = "99999999-9999-4999-8999-999999999999"
    replacement["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": manifest["manifest_digest"],
    }
    required_bindings = []
    store.reserve(
        replacement,
        limits,
        600,
        files,
        require_sources_live_through=lambda *_: None,
        require_bindings_live_through=lambda _request, restored, _deadline: (
            required_bindings.append(restored)
        ),
    )
    assert required_bindings == [bindings]
    restored = store.begin_execution(replacement["model_run_id"], files)
    assert restored["source_bindings"] == bindings
    assert restored["completed_units"] == record["completed_units"]
    assert restored["failed_units"] == []
    replacement_key, replacement_record = next(
        (key, value)
        for key, value in values.items()
        if value["request"]["model_run_id"] == replacement["model_run_id"]
    )
    stream = encode_feature_stream(
        {"type": "FeatureCollection", "features": []}, {"Tubule"}
    )
    result = {
        "input_id": input_id,
        "input_identity": build_backend_input_identity(
            request["inputs"][0]["raster_input"],
            input_source_bindings(bindings, input_id),
        ),
        "execution_plan": plan,
        "feature_stream_artifact": files.publish_artifact_bytes(
            stream.data, media_type="application/geo+json-seq", content_encoding="gzip"
        ),
        "feature_count": 0,
        "vertex_count": 0,
    }
    before_publication = canonical_bytes(values[replacement_key])
    with pytest.raises(ValueError, match="lacks exact core/halo unit coverage"):
        store.publish_input(replacement["model_run_id"], result, 0, files)
    assert canonical_bytes(values[replacement_key]) == before_publication
    all_identities = model_run_execution_identity_digests(
        replacement,
        [(input_id, unit_id) for unit_id in range(512)],
        limits,
        bindings,
    )
    completed_record = copy.deepcopy(replacement_record)
    # Seed the remaining synthetic computation, not 480 cloud model calls.
    # Final input publication must still require every core to be complete.
    completed_record["completed_units"].extend(
        {
            "input_id": input_id,
            "unit_id": unit_id,
            "reusable_state": files.publish_artifact_bytes(
                b"unit", media_type="application/octet-stream"
            ),
            "execution_identity_digest": all_identities[(input_id, unit_id)],
        }
        for unit_id in range(32, 512)
    )
    values[replacement_key] = validate_model_run_record(completed_record)
    store.publish_input(replacement["model_run_id"], result, 0, files)
    assert store.complete(replacement["model_run_id"], files)
    complete = store.status(replacement["model_run_id"], files)
    assert complete["phase"] == "COMPLETE"
    assert complete["completed_unit_count"] == 512
    completed_manifest = _artifact_json(files, complete["terminal_manifest"])
    assert completed_manifest["completed_units"] == []
    assert completed_manifest["completed_inputs"] == [result]
    assert completed_manifest["source_bindings"] == bindings
    previous = next(
        item
        for item in values.values()
        if item["request"]["model_run_id"] == replacement["model_run_id"]
    )
    next_request = copy.deepcopy(replacement)
    next_request["model_run_id"] = "88888888-8888-4888-8888-888888888888"
    next_request["resume_from"] = {
        "previous_run_id": replacement["model_run_id"],
        "previous_manifest_digest": completed_manifest["manifest_digest"],
    }
    with patch.object(
        publications, "_publication_context", wraps=publications._publication_context
    ) as context:
        selected = select_reusable_model_run_state(
            previous, next_request, limits, files
        )
    assert (
        context.call_count == 3
    )  # Prior identities, replacement identities, complete-input validation.
    assert selected["completed_inputs"] == [result]
    assert selected["completed_units"] == []
    assert selected["source_bindings"] == bindings


def test_512_core_finalization_validates_the_shared_plan_once(
    tmp_path, whole_slide_metadata
):
    request, limits, plan, bindings = copy.deepcopy(whole_slide_metadata)
    files = ArtifactFileStore(tmp_path)
    empty = encode_feature_stream({"type": "FeatureCollection", "features": []}, set())
    reference = files.publish_artifact_bytes(
        empty.data, media_type="application/geo+json-seq", content_encoding="gzip"
    )
    input_id = request["inputs"][0]["input_id"]
    identities = model_run_execution_identity_digests(
        request, [(input_id, unit_id) for unit_id in range(512)], limits, bindings
    )
    units = [
        {
            "input_id": input_id,
            "unit_id": unit_id,
            "reusable_state": reference,
            "execution_identity_digest": identities[(input_id, unit_id)],
        }
        for unit_id in range(512)
    ]
    with patch.object(
        publications, "_publication_context", wraps=publications._publication_context
    ) as context:
        actual = merge_core_halo_unit_states(
            units,
            request=request,
            input_id=input_id,
            plan=plan,
            model_run_limits=limits,
            declared_classes=set(),
            output_type="instance_segmentation",
            result_budget=request["result_budget"],
            target_raster=request["inputs"][0]["raster_input"]["frame"]["transform"][
                "raster"
            ],
            files=files,
            source_bindings=bindings,
        )
    assert actual == empty
    assert context.call_count == 1
