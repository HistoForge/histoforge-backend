"""Durable finite-plan admission and finite-call science parity tests."""

from __future__ import annotations

import copy
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.registry import runner_registry
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.feature_stream import (
    encode_canonical_feature_records,
    encode_feature_stream,
)
from histoforge.cloud.inference.features import (
    CanonicalFeatureRecords,
    build_bounded_feature_collection,
    build_feature_collection,
    validate_feature_records,
)
from histoforge.cloud.inference.finite.runtime import execute_finite_request
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.inference.runs.core_halo_runtime import (
    finalize_core_halo_input,
)
from histoforge.cloud.inference.runs.core_halo_state import (
    publish_core_halo_unit_state,
)
from histoforge.cloud.inference.runs.executor import (
    execute_reserved_model_run,
)
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
)
from histoforge.cloud.inference.runs.runtime import (
    execute_durable_finite_input,
    validate_durable_admission,
)
from histoforge.cloud.inference.runs.store import ModelRunStore
from histoforge.cloud.inference.semantic_features import assemble_semantic_fragments
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION
from histoforge.cloud.rasterization.inputs import build_model_raster_input
from histoforge.cloud.sources.files import SourceFileStore

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class _Source:
    def resolve_sealed_reference(self, reference):
        return {"artifact": reference}


class _SimulatedWorkerLoss(BaseException):
    """Abrupt container loss that cannot publish a terminal failure."""


class _StoreLifecycle:
    def __init__(
        self,
        store,
        files,
        *,
        stop_before_complete=False,
        stop_after_unit=False,
        stop_after_unit_failure=False,
    ):
        self.store = store
        self.files = files
        self.stop_before_complete = stop_before_complete
        self.stop_after_unit = stop_after_unit
        self.stop_after_unit_failure = stop_after_unit_failure

    def begin_execution(self, model_run_id):
        return self.store.begin_execution(model_run_id, self.files)

    def source_delivery(self, model_run_id):
        return self.store.source_delivery(model_run_id, self.files)

    def publish_input(self, model_run_id, publication, decoded_bytes):
        return self.store.publish_input(
            model_run_id,
            publication,
            decoded_bytes,
            self.files,
        )

    def publish_unit(self, model_run_id, publication):
        accepted = self.store.publish_unit(model_run_id, publication, self.files)
        if self.stop_after_unit:
            raise _SimulatedWorkerLoss
        return accepted

    def publish_unit_failure(self, model_run_id, publication):
        accepted = self.store.publish_unit_failure(
            model_run_id, publication, self.files
        )
        if self.stop_after_unit_failure:
            raise _SimulatedWorkerLoss
        return accepted

    def complete(self, model_run_id):
        if self.stop_before_complete:
            raise _SimulatedWorkerLoss
        return self.store.complete(model_run_id, self.files)

    def fail(self, model_run_id, failure):
        return self.store.fail(model_run_id, failure, self.files)


def test_durable_finite_input_uses_exact_finite_science_and_publishes_stream(
    tmp_path, monkeypatch
):
    finite, request, activation, limits = _values()
    raster = np.zeros((384, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: raster,
    )

    def infer(_image, _budget):
        return build_feature_collection(
            [
                (
                    "Tubule",
                    {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]],
                    },
                )
            ],
            {"Tubule"},
        )

    admitted, admitted_limits, retention = validate_durable_admission(
        request,
        "histoforge/tubule",
        "tubule_semantic",
        activation,
    )
    assert admitted == request
    assert admitted_limits == limits
    assert retention == 7 * 24 * 60 * 60

    files = ArtifactFileStore(tmp_path)
    durable = execute_durable_finite_input(
        admitted,
        admitted["inputs"][0]["input_id"],
        admitted_limits,
        activation,
        _Source(),
        SourceFileStore(tmp_path),
        files,
        infer,
    )
    finite_outcome = execute_finite_request(
        finite,
        activation,
        _Source(),
        SourceFileStore(tmp_path),
        infer,
    )

    stream = durable["input"]["feature_stream_artifact"]
    compressed = files.read_artifact_block(
        stream["artifact_id"], 0, stream["size_bytes"]
    )
    decoded = gzip.decompress(compressed)
    features = [json.loads(record) for record in decoded.split(b"\x1e") if record]
    assert features == finite_outcome["geojson"]["features"]
    assert durable["decoded_bytes"] == len(decoded)
    assert durable["input"]["execution_plan"]["type"] == "finite"


def test_durable_admission_resolves_gpu_but_worker_identity_remains_exact(tmp_path):
    _finite, request, activation, limits = _values()
    request["expected_backend"]["gpu"] = "A100"

    admitted, admitted_limits, _retention = validate_durable_admission(
        request,
        "histoforge/tubule",
        "tubule_semantic",
        activation,
    )
    assert admitted == request

    with pytest.raises(GatewayFailure, match="no longer matches"):
        execute_durable_finite_input(
            admitted,
            admitted["inputs"][0]["input_id"],
            admitted_limits,
            activation,
            _Source(),
            SourceFileStore(tmp_path),
            ArtifactFileStore(tmp_path),
            lambda *_args: pytest.fail("GPU mismatch must precede inference"),
        )


def test_durable_finite_builds_geometry_under_actual_stream_byte_limits(
    tmp_path, monkeypatch
):
    _finite, request, activation, limits = _values()
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: np.zeros((384, 512, 3), dtype=np.uint8),
    )
    instances = [
        (
            "Tubule",
            {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]]},
        )
    ]
    expected = encode_feature_stream(
        build_feature_collection(instances, {"Tubule"}), {"Tubule"}
    )
    request["result_budget"].update(
        max_transferred_bytes=len(expected.data),
        max_decoded_bytes=expected.decoded_bytes,
    )

    def run(current):
        return execute_durable_finite_input(
            current,
            current["inputs"][0]["input_id"],
            limits,
            activation,
            _Source(),
            SourceFileStore(tmp_path),
            ArtifactFileStore(tmp_path),
            lambda _image, budget: build_bounded_feature_collection(
                instances, {"Tubule"}, budget
            ),
        )

    actual = run(request)
    assert actual["decoded_bytes"] == expected.decoded_bytes
    assert actual["input"]["feature_stream_artifact"]["size_bytes"] == len(
        expected.data
    )
    for field in ("max_transferred_bytes", "max_decoded_bytes"):
        too_small = copy.deepcopy(request)
        too_small["result_budget"][field] -= 1
        with pytest.raises(GatewayFailure, match=field) as failure:
            run(too_small)
        assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"


def test_durable_finite_does_not_inherit_whole_slide_candidate_limits(
    tmp_path, monkeypatch
):
    _finite, request, activation, limits = _values()
    request["result_budget"] = copy.deepcopy(
        runner_registry()["definitions"]["dinov3-adapter-cellpose"]["capabilities"][
            "durable_run"
        ]["run"]["result"]
    )
    limits["result"] = copy.deepcopy(request["result_budget"])
    observed = []

    def execute(_raster, budget, *_args, **_kwargs):
        observed.append(budget)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(
        "histoforge.cloud.inference.runs.runtime.execute_model_input", execute
    )
    execute_durable_finite_input(
        request,
        request["inputs"][0]["input_id"],
        limits,
        activation,
        _Source(),
        SourceFileStore(tmp_path),
        ArtifactFileStore(tmp_path),
        lambda *_args: pytest.fail("prediction is stubbed"),
    )
    assert observed == [activation["capabilities"]["finite"]["result"]]


def test_durable_finite_restart_reuses_published_input_without_reinference(
    tmp_path, monkeypatch
):
    _finite, request, activation, limits = _values()
    raster = np.zeros((384, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: raster,
    )
    inference_calls = []

    def infer(_image, _budget):
        inference_calls.append("infer")
        return build_feature_collection(
            [
                (
                    "Tubule",
                    {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 0]]],
                    },
                )
            ],
            {"Tubule"},
        )

    class Runner:
        def execute_input(self, current, input_id, current_limits, source_bindings):
            assert source_bindings == []
            return execute_durable_finite_input(
                current,
                input_id,
                current_limits,
                activation,
                _Source(),
                SourceFileStore(tmp_path),
                files,
                infer,
                source_bindings=source_bindings,
            )

    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=1)
    accepted, should_dispatch = store.reserve(
        request,
        limits,
        activation["capabilities"]["durable_run"]["terminal_retention_seconds"],
        files,
        require_sources_live_through=lambda _request, _deadline: None,
    )
    assert accepted["model_run_id"] == request["model_run_id"]
    assert should_dispatch is True

    with pytest.raises(_SimulatedWorkerLoss):
        execute_reserved_model_run(
            request["model_run_id"],
            _StoreLifecycle(store, files, stop_before_complete=True),
            lambda _request: Runner(),
        )
    interrupted = store.status(request["model_run_id"], files)
    assert interrupted["phase"] == "RUNNING"
    assert interrupted["completed_unit_count"] == 1
    assert inference_calls == ["infer"]

    assert execute_reserved_model_run(
        request["model_run_id"],
        _StoreLifecycle(store, files),
        lambda _request: Runner(),
    )
    completed = store.status(request["model_run_id"], files)
    assert completed["phase"] == "COMPLETE"
    assert completed["completed_unit_count"] == 1
    assert completed["terminal_manifest"] is not None
    assert inference_calls == ["infer"]


@pytest.mark.parametrize("semantic_output", [False, True], ids=["instance", "semantic"])
def test_durable_core_restart_reuses_units_and_recomputes_missing_input_statistics(
    tmp_path, semantic_output
):
    request, activation, limits = (
        _semantic_core_values() if semantic_output else _core_values()
    )
    files = ArtifactFileStore(tmp_path)
    calls = {"unit": [], "finalize": [], "normalization": 0}

    class Runner:
        def calculate_core_halo_percentiles(self, *_args):
            # This test supplies immutable empty unit state, not model pixels.
            calls["normalization"] += 1
            return [[0.0, 255.0]] * 3

        def execute_core_halo_unit(
            self,
            current,
            input_id,
            unit_id,
            current_limits,
            source_bindings,
            channel_percentiles=None,
        ):
            assert source_bindings == []
            assert channel_percentiles == [[0.0, 255.0]] * 3
            calls["unit"].append(unit_id)
            reference = publish_core_halo_unit_state(
                encode_canonical_feature_records(
                    CanonicalFeatureRecords([], [], 0).encoded_records(),
                    current_limits["intermediate_result"],
                ),
                files,
            )
            return {
                "input_id": input_id,
                "unit_id": unit_id,
                "reusable_state": reference,
                "execution_identity_digest": model_run_execution_identity_digest(
                    current, input_id, unit_id, current_limits
                ),
            }

        def finalize_core_halo_input(
            self, current, input_id, current_limits, units, source_bindings
        ):
            assert source_bindings == []
            calls["finalize"].append([item["unit_id"] for item in units])
            return finalize_core_halo_input(
                current,
                input_id,
                current_limits,
                units,
                activation,
                files,
                source_bindings=source_bindings,
            )

    store = ModelRunStore({}, max_active_runs=1)
    accepted, should_dispatch = store.reserve(
        request,
        limits,
        3600,
        files,
        require_sources_live_through=lambda _request, _deadline: None,
    )
    assert accepted["model_run_id"] == request["model_run_id"]
    assert should_dispatch

    with pytest.raises(_SimulatedWorkerLoss):
        execute_reserved_model_run(
            request["model_run_id"],
            _StoreLifecycle(store, files, stop_after_unit=True),
            lambda _request: Runner(),
        )
    assert store.status(request["model_run_id"], files)["completed_unit_count"] == 1

    assert execute_reserved_model_run(
        request["model_run_id"],
        _StoreLifecycle(store, files),
        lambda _request: Runner(),
    )
    status = store.status(request["model_run_id"], files)
    assert status["phase"] == "COMPLETE"
    assert status["completed_unit_count"] == 2
    assert calls == {
        "unit": [0, 1],
        "finalize": [[0, 1]],
        "normalization": 2,
    }
    assert "failure" not in status
    artifact = status["terminal_manifest"]
    manifest = json.loads(
        files.read_artifact_block(artifact["artifact_id"], 0, artifact["size_bytes"])
    )
    complete_input = manifest["completed_inputs"][0]
    assert complete_input["feature_count"] == complete_input["vertex_count"] == 0
    stream = complete_input["feature_stream_artifact"]
    assert (
        gzip.decompress(
            files.read_artifact_block(stream["artifact_id"], 0, stream["size_bytes"])
        )
        == b""
    )


@pytest.mark.parametrize(
    ("retain_unit", "terminalizer"),
    [(False, "worker_restart"), (True, "worker_restart"), (True, "deadline")],
)
def test_interrupted_worker_preserves_retained_unit_failure(
    tmp_path, retain_unit, terminalizer
):
    request, _activation, limits = _core_values()
    files = ArtifactFileStore(tmp_path)
    now = [datetime(2026, 8, 17, 12, tzinfo=timezone.utc)]

    class FailingRunner:
        def calculate_core_halo_percentiles(self, *_args):
            return [[0.0, 255.0]] * 3

        def execute_core_halo_unit(
            self,
            current,
            input_id,
            unit_id,
            current_limits,
            source_bindings,
            channel_percentiles,
        ):
            assert source_bindings == []
            assert channel_percentiles == [[0.0, 255.0]] * 3
            if retain_unit and unit_id == 0:
                return {
                    "input_id": input_id,
                    "unit_id": unit_id,
                    "reusable_state": publish_core_halo_unit_state(
                        encode_canonical_feature_records(
                            CanonicalFeatureRecords([], [], 0).encoded_records(),
                            current_limits["intermediate_result"],
                        ),
                        files,
                    ),
                    "execution_identity_digest": model_run_execution_identity_digest(
                        current, input_id, unit_id, current_limits
                    ),
                }
            raise GatewayFailure("INSUFFICIENT_CONTEXT", "unit context is insufficient")

    store = ModelRunStore({}, max_active_runs=1, clock=lambda: now[0])
    _accepted, should_dispatch = store.reserve(
        request,
        limits,
        3600,
        files,
        require_sources_live_through=lambda _request, _deadline: None,
    )
    assert should_dispatch

    with pytest.raises(_SimulatedWorkerLoss):
        execute_reserved_model_run(
            request["model_run_id"],
            _StoreLifecycle(store, files, stop_after_unit_failure=True),
            lambda _request: FailingRunner(),
        )
    assert store.status(request["model_run_id"], files)["phase"] == "RUNNING"

    if terminalizer == "worker_restart":
        created = []
        assert not execute_reserved_model_run(
            request["model_run_id"],
            _StoreLifecycle(store, files),
            lambda current: created.append(current),
        )
        assert created == []
    else:
        now[0] += timedelta(seconds=limits["max_execution_seconds"])
        assert store.finalize_deadline(request["model_run_id"], files)
    status = store.status(request["model_run_id"], files)
    assert status["phase"] == ("PARTIAL" if retain_unit else "FAILED")
    assert status["failure"]["code"] == "INSUFFICIENT_CONTEXT"
    if retain_unit:
        artifact = status["terminal_manifest"]
        manifest = json.loads(
            files.read_artifact_block(
                artifact["artifact_id"], 0, artifact["size_bytes"]
            )
        )
        assert manifest["failure"] == status["failure"]
        assert manifest["failed_units"][0]["failure"] == status["failure"]
        assert [item["unit_id"] for item in manifest["completed_units"]] == [0]


@pytest.mark.parametrize("code", ["INSUFFICIENT_CONTEXT", "RESULT_LIMIT_EXCEEDED"])
def test_partial_failure_and_reusable_units_survive_explicit_same_settings_resume(
    tmp_path, code
):
    request, activation, limits = _core_values()
    request["halo_px_override"] = 40
    request["result_budget"]["max_features"] = 1
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=1)
    calls = {"unit": [], "finalize": []}

    class Runner:
        def calculate_core_halo_percentiles(self, *_args):
            return [[0.0, 255.0]] * 3

        def execute_core_halo_unit(
            self,
            current,
            input_id,
            unit_id,
            current_limits,
            source_bindings,
            channel_percentiles,
        ):
            assert source_bindings == []
            assert channel_percentiles == [[0.0, 255.0]] * 3
            assert current["halo_px_override"] == 40
            assert current["result_budget"] == request["result_budget"]
            calls["unit"].append(unit_id)
            if code == "INSUFFICIENT_CONTEXT" and unit_id == 1:
                raise GatewayFailure(code, "owned object reaches the processing edge")
            x = unit_id * 256
            collection = build_feature_collection(
                [
                    (
                        "Tubule",
                        {
                            "type": "Polygon",
                            "coordinates": [[[x, 0], [x + 2, 0], [x + 2, 2], [x, 0]]],
                        },
                    )
                ],
                {"Tubule"},
            )
            return {
                "input_id": input_id,
                "unit_id": unit_id,
                "reusable_state": publish_core_halo_unit_state(
                    encode_canonical_feature_records(
                        validate_feature_records(
                            collection["features"], {"Tubule"}
                        ).encoded_records(),
                        current_limits["intermediate_result"],
                    ),
                    files,
                ),
                "execution_identity_digest": model_run_execution_identity_digest(
                    current, input_id, unit_id, current_limits
                ),
            }

        def finalize_core_halo_input(
            self, current, input_id, current_limits, units, source_bindings
        ):
            assert source_bindings == []
            calls["finalize"].append([item["unit_id"] for item in units])
            # Each unit fits the budget, but the real final merge exceeds it.
            return finalize_core_halo_input(
                current,
                input_id,
                current_limits,
                units,
                activation,
                files,
                source_bindings=source_bindings,
            )

    prior_manifest = None
    for attempt in range(2):
        current = copy.deepcopy(request)
        if attempt:
            current["model_run_id"] = "89898989-8989-4989-8989-898989898989"
            current["resume_from"] = {
                "previous_run_id": request["model_run_id"],
                "previous_manifest_digest": prior_manifest["manifest_digest"],
            }
        _, should_dispatch = store.reserve(
            current,
            limits,
            3600,
            files,
            require_sources_live_through=lambda _request, _deadline: None,
        )
        assert should_dispatch
        assert "failure" not in store.status(current["model_run_id"], files)
        assert not execute_reserved_model_run(
            current["model_run_id"],
            _StoreLifecycle(store, files),
            lambda _request: Runner(),
        )
        store = ModelRunStore(values, max_active_runs=1)
        status = store.status(current["model_run_id"], files)
        artifact = status["terminal_manifest"]
        manifest = json.loads(
            files.read_artifact_block(
                artifact["artifact_id"], 0, artifact["size_bytes"]
            )
        )
        assert status["phase"] == "PARTIAL"
        assert status["failure"] == manifest["failure"]
        assert status["failure"]["code"] == code
        assert manifest["completed_inputs"] == []
        if code == "INSUFFICIENT_CONTEXT":
            assert [item["unit_id"] for item in manifest["completed_units"]] == [0]
            assert manifest["failed_units"][0]["failure"] == status["failure"]
        else:
            assert [item["unit_id"] for item in manifest["completed_units"]] == [0, 1]
            assert manifest["failed_units"] == []
        prior_manifest = manifest

    # Resume makes one explicit attempt under captured settings; it neither grows
    # the halo/output budget nor reruns completed units or invents a failed unit.
    if code == "INSUFFICIENT_CONTEXT":
        assert calls == {"unit": [0, 1, 1], "finalize": []}
    else:
        assert calls == {"unit": [0, 1], "finalize": [[0, 1], [0, 1]]}


@pytest.mark.parametrize("limited_field", ["max_vertices", "max_record_bytes"])
def test_joined_semantic_geometry_limit_survives_partial_storage_and_core_reuse(
    tmp_path, limited_field
):
    request, activation, limits = _semantic_core_values()
    fragments = [
        (
            "Tubule",
            {
                "type": "Polygon",
                "coordinates": [
                    [[254, 10], [256, 10], [256, 14], [254, 14], [254, 10]]
                ],
            },
        ),
        (
            "Tubule",
            {
                "type": "Polygon",
                "coordinates": [
                    [[256, 12], [259, 12], [259, 14], [256, 14], [256, 12]]
                ],
            },
        ),
    ]
    core_collections = [
        build_feature_collection([fragment], {"Tubule"}) for fragment in fragments
    ]
    assembled = build_feature_collection(
        assemble_semantic_fragments(fragments), {"Tubule"}
    )
    assert len(assembled["features"]) == 1
    joined = assembled["features"][0]
    if limited_field == "max_vertices":
        joined_size = len(joined["geometry"]["coordinates"][0]) - 1
        fragment_sizes = [
            len(item["features"][0]["geometry"]["coordinates"][0]) - 1
            for item in core_collections
        ]
    else:
        joined_size = len(canonical_bytes(joined))
        fragment_sizes = [
            len(canonical_bytes(item["features"][0])) for item in core_collections
        ]
    assert max(fragment_sizes) < joined_size
    request["result_budget"]["max_features"] = 1
    request["result_budget"][limited_field] = joined_size - 1
    captured = copy.deepcopy(request)
    input_id = request["inputs"][0]["input_id"]
    captured_plan = build_expected_model_execution_plans(request, limits)[input_id]
    assert captured_plan["resolved_halo_px"] == 40
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=1)
    calls = {"unit": [], "finalize": []}

    class Runner:
        def calculate_core_halo_percentiles(self, *_args):
            return [[0.0, 255.0]] * 3

        def execute_core_halo_unit(
            self,
            current,
            current_input_id,
            unit_id,
            current_limits,
            source_bindings,
            channel_percentiles,
        ):
            assert source_bindings == []
            assert channel_percentiles == [[0.0, 255.0]] * 3
            assert unit_id not in calls["unit"], (
                "completed semantic cores must not be predicted again"
            )
            calls["unit"].append(unit_id)
            return {
                "input_id": current_input_id,
                "unit_id": unit_id,
                "reusable_state": publish_core_halo_unit_state(
                    encode_canonical_feature_records(
                        validate_feature_records(
                            core_collections[unit_id]["features"], {"Tubule"}
                        ).encoded_records(),
                        current_limits["intermediate_result"],
                    ),
                    files,
                ),
                "execution_identity_digest": model_run_execution_identity_digest(
                    current, current_input_id, unit_id, current_limits
                ),
            }

        def finalize_core_halo_input(
            self, current, current_input_id, current_limits, units, source_bindings
        ):
            assert source_bindings == []
            assert {
                key: value
                for key, value in current.items()
                if key not in {"model_run_id", "resume_from"}
            } == {
                key: value for key, value in captured.items() if key != "model_run_id"
            }
            assert current_limits == limits
            assert (
                build_expected_model_execution_plans(current, current_limits)[
                    current_input_id
                ]
                == captured_plan
            )
            calls["finalize"].append([item["unit_id"] for item in units])
            return finalize_core_halo_input(
                current,
                current_input_id,
                current_limits,
                units,
                activation,
                files,
                source_bindings=source_bindings,
            )

    prior_manifest = None
    retained_bytes = {}
    for attempt in range(2):
        current = copy.deepcopy(request)
        if attempt:
            current["model_run_id"] = "89898989-8989-4989-8989-898989898989"
            current["resume_from"] = {
                "previous_run_id": request["model_run_id"],
                "previous_manifest_digest": prior_manifest["manifest_digest"],
            }
        _, should_dispatch = store.reserve(
            current,
            limits,
            3600,
            files,
            require_sources_live_through=lambda _request, _deadline: None,
        )
        assert should_dispatch
        queued = store.status(current["model_run_id"], files)
        assert "failure" not in queued
        assert queued["completed_unit_count"] == (2 if attempt else 0)
        assert not execute_reserved_model_run(
            current["model_run_id"],
            _StoreLifecycle(store, files),
            lambda _request: Runner(),
        )
        # Reopen the control store before reading the retained failure/artifacts.
        store = ModelRunStore(values, max_active_runs=1)
        status = store.status(current["model_run_id"], files)
        artifact = status["terminal_manifest"]
        manifest = json.loads(
            files.read_artifact_block(
                artifact["artifact_id"], 0, artifact["size_bytes"]
            )
        )
        assert status["phase"] == "PARTIAL"
        assert status["completed_unit_count"] == 2
        assert status["failure"] == manifest["failure"]
        assert status["failure"]["code"] == "RESULT_LIMIT_EXCEEDED"
        assert limited_field in status["failure"]["message"]
        assert manifest["completed_inputs"] == []
        assert manifest["failed_units"] == []
        assert [item["unit_id"] for item in manifest["completed_units"]] == [0, 1]
        if prior_manifest is not None:
            assert manifest["failure"] == prior_manifest["failure"]
            assert manifest["completed_units"] == prior_manifest["completed_units"]
        for unit in manifest["completed_units"]:
            reference = unit["reusable_state"]
            assert files.load_artifact_ref(reference["artifact_id"]) == reference
            data = files.read_artifact_block(
                reference["artifact_id"], 0, reference["size_bytes"]
            )
            if attempt:
                assert data == retained_bytes[unit["unit_id"]]
            else:
                retained_bytes[unit["unit_id"]] = data
        prior_manifest = manifest

    assert calls == {"unit": [0, 1], "finalize": [[0, 1], [0, 1]]}


def test_durable_admission_rejects_runner_without_the_mode():
    _finite, request, activation, _limits = _values()
    capabilities = copy.deepcopy(
        runner_registry()["definitions"]["dinov3-adapter-semantic"]["capabilities"]
    )
    capabilities["modes"] = ["FINITE_CALL"]
    capabilities.pop("durable_run")
    # Keep identities consistent so admission reaches the missing durable mode.
    activation["capabilities"] = capabilities
    activation["backend"].update(
        architecture="dinov3-adapter-semantic",
        runner_build=capabilities["runner_build"],
        rasterizer=capabilities["rasterizer"],
        preprocessing_contract_digest=capabilities["preprocessing_contract_digest"],
    )
    request["expected_backend"] = copy.deepcopy(activation["backend"])

    with pytest.raises(GatewayFailure) as failure:
        validate_durable_admission(
            request,
            "histoforge/tubule",
            "tubule_semantic",
            activation,
        )
    assert failure.value.error["code"] == "UNSUPPORTED_CAPABILITY"


def test_durable_worker_rejects_boolean_or_absent_input_id_before_inference(tmp_path):
    _finite, request, activation, limits = _values()

    def unexpected_inference(_image, _budget):
        pytest.fail("invalid input identity must fail before inference")

    for input_id in (True, 999):
        with pytest.raises(ValueError, match="input_id"):
            execute_durable_finite_input(
                request,
                input_id,
                limits,
                activation,
                _Source(),
                SourceFileStore(tmp_path),
                ArtifactFileStore(tmp_path),
                unexpected_inference,
            )


def _values():
    finite_case = _fixture("finite-inference-v1.json")["cases"][0]
    finite = json.loads(finite_case["requestCanonical"])
    capabilities = copy.deepcopy(
        runner_registry()["definitions"]["dinov3-adapter-hovernet"]["capabilities"]
    )
    limits = copy.deepcopy(capabilities["durable_run"]["run"])
    backend = copy.deepcopy(finite["expected_backend"])
    backend["runner_build"] = capabilities["runner_build"]
    backend["rasterizer"] = capabilities["rasterizer"]
    backend["preprocessing_contract_digest"] = capabilities[
        "preprocessing_contract_digest"
    ]
    finite["expected_backend"] = copy.deepcopy(backend)
    raster_input = finite["raster_input"]
    finite["raster_input"] = build_model_raster_input(
        raster_input["source_view"],
        raster_input["frame"],
        capabilities["rasterizer"],
    )
    case = _fixture("model-run-request-v1.json")["cases"][0]
    region = finite["raster_input"]["frame"]["region"]
    request = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": case["actionId"],
        "model_run_id": case["modelRunId"],
        "producer": copy.deepcopy(finite["producer"]),
        "expected_backend": copy.deepcopy(backend),
        "inputs": [
            {
                "input_id": case["inputId"],
                "output_generation_id": case["outputGenerationId"],
                "output": {
                    "producer": copy.deepcopy(finite["producer"]),
                    "scope": {"type": "field", "region": copy.deepcopy(region)},
                },
                "raster_input": copy.deepcopy(finite["raster_input"]),
            }
        ],
        "result_budget": copy.deepcopy(finite["result_budget"]),
    }
    activation = {
        "backend": backend,
        "capabilities": capabilities,
        "model": {
            "output": {
                "type": "instance_segmentation",
                "classes": ["Tubule"],
            }
        },
    }
    return finite, request, activation, limits


def _core_values():
    _finite, request, activation, limits = _values()
    request["expected_profile"] = build_core_halo_profile(256, 384, 32, "2" * 64)
    limits["max_units"] = 2
    limits["max_recovery_state_bytes_per_unit"] = 1024 * 1024
    limits["max_recovery_state_bytes_per_run"] = 2 * 1024 * 1024
    return request, activation, limits


def _semantic_core_values():
    request, activation, limits = _core_values()
    request["halo_px_override"] = 40
    request["expected_backend"]["architecture"] = "dinov3-adapter-semantic"
    activation["backend"]["architecture"] = "dinov3-adapter-semantic"
    activation["model"]["output"] = {
        "type": "dense_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Tubule"},
        ],
    }
    return request, activation, limits


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
