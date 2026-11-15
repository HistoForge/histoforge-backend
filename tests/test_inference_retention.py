"""Uploaded pixels and terminal results have independent, fixed expiry times."""

import copy
import gzip
import hashlib
from datetime import timedelta

import cv2
import numpy as np
import pytest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure, build_gateway_error
from histoforge.cloud.architectures.registry import MODEL_RUN_MAX_EXECUTION_SECONDS
from histoforge.cloud.artifacts.lifecycle import (
    _require_bound_source,
    _require_bound_sources_live_through,
    _require_sources_live_through,
)
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.contract import build_backend_input_identity
from histoforge.cloud.inference.feature_stream import encode_feature_stream
from histoforge.cloud.inference.runs.executor import _COORDINATOR_TIMEOUT_SECONDS
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
)
from histoforge.cloud.inference.runs.source_bindings import input_source_bindings
from histoforge.cloud.inference.runs.store import ModelRunStore
from histoforge.cloud.sources.capture import captured_image_origin
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.store import SourceArtifactStore

from test_model_run_source_delivery import _binding, _captured_request, _unit
from test_model_run_store import (
    _artifact_json,
    _Clock,
    _finite_publications,
    _limits,
    _one_feature_population,
    _request,
    _reserve,
)
from test_source_artifact_store import FakeDict, _png_source, _upload_all

RETENTION = 7 * 24 * 60 * 60


def test_execution_source_and_result_durations_have_separate_owners():
    assert MODEL_RUN_MAX_EXECUTION_SECONDS == 12 * 60 * 60
    assert _COORDINATOR_TIMEOUT_SECONDS == MODEL_RUN_MAX_EXECUTION_SECONDS + 60
    assert DEPLOYMENT_CONFIG.source_artifact_limits.absolute_lifetime_seconds == (
        48 * 60 * 60
    )


@pytest.mark.parametrize(
    "phase", ["COMPLETE", "PARTIAL", "FAILED", "CANCELLED", "TIMED_OUT"]
)
def test_every_terminal_phase_retains_its_original_seven_day_deadline(tmp_path, phase):
    clock = _Clock()
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=1, clock=clock)
    request = _request(1 if phase == "PARTIAL" else 0)
    limits = _run_limits()
    _reserve(store, request, limits, RETENTION, files)
    run_id = request["model_run_id"]
    clock.advance(hours=1)
    error = build_gateway_error("RUNNER_FAILED", "prediction failed")
    if phase == "COMPLETE":
        _, result = _finite_publications(request, files, limits)
        store.publish_input(run_id, result, 0, files)
        store.complete(run_id, files)
    elif phase == "PARTIAL":
        unit, _ = _finite_publications(request, files, limits)
        store.publish_unit(run_id, unit, files)
        store.fail(run_id, error, files)
    elif phase == "FAILED":
        store.fail(run_id, error, files)
    elif phase == "CANCELLED":
        store.cancel(run_id, files)
    else:
        clock.advance(hours=11)
        assert store.finalize_deadline(run_id, files)
    stopped_at = clock.value
    terminal = store.status(run_id, files)
    assert terminal["phase"] == phase
    assert terminal["phase_deadline"] == _iso(stopped_at + timedelta(days=7))
    artifact_ids = files.artifact_ids()

    # Polling, late worker outcomes and repeated cancellation cannot extend expiry.
    clock.advance(days=6)
    store.cancel(run_id, files)
    assert not store.fail(run_id, error, files)
    assert not store.complete(run_id, files)
    reopened = ModelRunStore(values, max_active_runs=1, clock=clock)
    assert reopened.status(run_id, files) == terminal
    reopened.cleanup_expired(files)
    assert files.artifact_ids() == artifact_ids
    clock.advance(days=1)
    with pytest.raises(GatewayFailure) as expired:
        reopened.status(run_id, files)
    assert expired.value.error["code"] == "MODEL_RUN_EXPIRED"
    reopened.cleanup_expired(files)
    assert files.artifact_ids() == set()
    assert values == {}


def test_completed_geometry_survives_physical_source_cleanup_and_reopening(tmp_path):
    clock = _Clock()
    source_records = FakeDict()
    sources = _sources(source_records, clock)
    source_files = SourceFileStore(tmp_path)
    files = ArtifactFileStore(tmp_path)
    values = {}
    runs = ModelRunStore(values, max_active_runs=1, clock=clock)
    request = _captured_request(False)
    limits = _run_limits()
    binding = _sealed_binding(request, 0, sources, source_files)
    _reserve(runs, request, limits, RETENTION, files)
    _bind(runs, files, request, binding, sources)
    population = _one_feature_population()
    stream = encode_feature_stream(population, {"Tubule"})
    item = request["inputs"][0]
    result = {
        "input_id": item["input_id"],
        "input_identity": build_backend_input_identity(
            item["raster_input"], input_source_bindings([binding], item["input_id"])
        ),
        "execution_plan": build_expected_model_execution_plans(request, limits)[
            item["input_id"]
        ],
        "feature_stream_artifact": files.publish_artifact_bytes(
            stream.data, media_type="application/geo+json-seq", content_encoding="gzip"
        ),
        "feature_count": stream.feature_count,
        "vertex_count": stream.vertex_count,
    }
    clock.advance(hours=1)
    run_id = request["model_run_id"]
    runs.publish_input(run_id, result, stream.decoded_bytes, files)
    runs.complete(run_id, files)
    status = runs.status(run_id, files)
    reference = binding["source"]["artifact"]
    assert reference["artifact_id"] in source_files.source_ids()

    clock.advance(hours=47)
    sources.cleanup_expired(source_files)
    assert source_files.source_ids() == set()
    with pytest.raises(GatewayFailure) as removed:
        sources.resolve_sealed_reference(reference)
    assert removed.value.error["code"] == "ARTIFACT_NOT_FOUND"
    reopened = ModelRunStore(values, max_active_runs=1, clock=clock)
    reopened.cleanup_expired(files)
    assert reopened.status(run_id, files) == status
    manifest = _artifact_json(files, status["terminal_manifest"])
    assert manifest["source_bindings"] == [binding]
    retained = manifest["completed_inputs"][0]["feature_stream_artifact"]
    decoded = gzip.decompress(
        files.read_artifact_block(retained["artifact_id"], 0, retained["size_bytes"])
    )
    assert decoded == gzip.decompress(stream.data)
    assert manifest["completed_inputs"][0]["feature_count"] == 1

    clock.advance(days=5, minutes=59, seconds=59)
    assert reopened.status(run_id, files) == status
    assert files.load_artifact_ref(retained["artifact_id"]) == retained
    clock.advance(seconds=1)
    with pytest.raises(GatewayFailure) as expired:
        reopened.status(run_id, files)
    assert expired.value.error["code"] == "MODEL_RUN_EXPIRED"
    reopened.cleanup_expired(files)
    assert files.artifact_ids() == set()


@pytest.mark.parametrize("offset_ms", [-1, 0, 1])
def test_partial_replacement_requires_all_exact_images_for_another_twelve_hours(
    tmp_path, offset_ms
):
    clock = _Clock()
    reserved_at = clock.value
    sources = _sources(FakeDict(), clock)
    source_files = SourceFileStore(tmp_path)
    files = ArtifactFileStore(tmp_path)
    values = {}
    runs = ModelRunStore(values, max_active_runs=2, clock=clock)
    request = _captured_request()
    limits = _run_limits()
    bindings = [
        _sealed_binding(request, unit, sources, source_files) for unit in (0, 1)
    ]
    _reserve(runs, request, limits, RETENTION, files)
    for binding in bindings:
        _bind(runs, files, request, binding, sources)
    first = _unit(request, files, bindings, 0)
    runs.publish_unit(request["model_run_id"], first, files)
    clock.advance(hours=1)
    runs.fail(
        request["model_run_id"],
        build_gateway_error("RUNNER_FAILED", "second core failed"),
        files,
    )
    prior = runs.status(request["model_run_id"], files)
    manifest = _artifact_json(files, prior["terminal_manifest"])
    replacement = copy.deepcopy(request)
    replacement["model_run_id"] = "99999999-9999-4999-8999-999999999999"
    replacement["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": manifest["manifest_digest"],
    }
    clock.value = reserved_at + timedelta(hours=36, milliseconds=offset_ms)
    checked = []

    def require_bindings(current, restored, deadline):
        checked.append(copy.deepcopy(restored))
        _require_bound_sources_live_through(sources, current, restored, deadline)

    before = copy.deepcopy(values)
    artifact_ids = files.artifact_ids()

    def reserve():
        return runs.reserve(
            replacement,
            limits,
            RETENTION,
            files,
            require_sources_live_through=lambda current, deadline: (
                _require_sources_live_through(sources, current, deadline)
            ),
            require_bindings_live_through=require_bindings,
        )

    if offset_ms <= 0:
        assert reserve()[1]
        context = runs.begin_execution(replacement["model_run_id"], files)
        assert context["source_bindings"] == bindings
        assert context["completed_units"] == [first]
        assert context["execution_deadline"] == _iso(clock.value + timedelta(hours=12))
    else:
        with pytest.raises(GatewayFailure) as expired:
            reserve()
        assert expired.value.error["code"] == "ARTIFACT_PRECONDITION_FAILED"
        assert values == before
        assert files.artifact_ids() == artifact_ids
        assert runs.status(request["model_run_id"], files) == prior
    assert checked == [bindings]


def test_existing_records_keep_their_captured_retention_when_policy_changes(tmp_path):
    clock = _Clock()
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=1, clock=clock)
    request = _request(0)
    _reserve(store, request, _limits(), 2, files)
    store.cancel(request["model_run_id"], files)
    before = store.status(request["model_run_id"], files)
    assert not _reserve(store, request, _run_limits(), RETENTION, files)[1]
    assert store.status(request["model_run_id"], files) == before
    assert before["phase_deadline"] == _iso(clock.value + timedelta(seconds=2))
    clock.advance(seconds=2)
    with pytest.raises(GatewayFailure) as expired:
        store.status(request["model_run_id"], files)
    assert expired.value.error["code"] == "MODEL_RUN_EXPIRED"


def _run_limits():
    limits = _limits()
    limits["max_execution_seconds"] = MODEL_RUN_MAX_EXECUTION_SECONDS
    return limits


def _sources(records, clock):
    return SourceArtifactStore(
        records,
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        DEPLOYMENT_CONFIG.limits.wire_value(),
        max_active_source_artifact_peak_bytes=(
            DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
        ),
        clock=clock,
    )


def _sealed_binding(request, unit_id, sources, files):
    binding = _binding(request, unit_id)
    rectangle = binding["source"]["source_rect_at_level"]
    capture = request["inputs"][0]["raster_input"]["source_view"]["capture"]
    pixels = np.zeros((rectangle["height"], rectangle["width"], 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    encoded = encoded.tobytes()
    start, _ = _png_source()
    start.update(
        total_bytes=len(encoded),
        chunk_size_bytes=len(encoded),
        origin=captured_image_origin(capture, rectangle),
        required_levels=[
            {
                "series_index": 0,
                "resolution_level": 0,
                "width": rectangle["width"],
                "height": rectangle["height"],
            }
        ],
    )
    sources.reserve(start)
    _upload_all(sources, files, start, encoded)
    sealed = sources.seal(
        start["artifact_id"],
        {"bytes_digest": hashlib.sha256(encoded).hexdigest()},
        files,
    )
    binding["source"]["artifact"] = sealed["artifact"]
    return binding


def _bind(runs, files, request, binding, sources):
    item = request["inputs"][0]
    return runs.bind_source(
        request["model_run_id"],
        binding,
        files,
        require_source=lambda _request, current, deadline: _require_bound_source(
            sources, item, current["source"], deadline, {"PNG", "TIFF_FAMILY"}
        ),
    )


def _iso(instant):
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")
