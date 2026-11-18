"""Finite runtime repeats identity/limit checks and projects typed outcomes."""

import copy
import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.registry import runner_registry
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt
from histoforge.cloud.inference import activation as model_activation
from histoforge.cloud.inference import execution as model_execution
from histoforge.cloud.inference.activation import activate_model
from histoforge.cloud.inference.features import ResultLimitExceeded
from histoforge.cloud.inference.finite import runtime as finite_runtime
from histoforge.cloud.inference.finite.runtime import (
    encode_finite_result,
    execute_finite_request,
    validate_finite_admission,
)
from histoforge.cloud.inference.identity import validate_backend_model_identity
from histoforge.cloud.modules import runtime as module_runtime
from histoforge.cloud.rasterization.inputs import build_model_raster_input
from histoforge.cloud.rasterization.rasterizer import InsufficientSourceContext

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "finite-inference-v3.json"


def test_admission_binds_current_backend_limits_and_reader_owned_view_geometry():
    request, backend, capabilities, resolved, files = _admission_values()
    source_store = _SourceStore(resolved)

    admitted = validate_finite_admission(
        request,
        "histoforge/tubule",
        "tubule_semantic",
        backend,
        capabilities,
        source_store,
        files,
    )

    assert admitted == request
    assert files.calls == 1

    too_large = copy.deepcopy(request)
    too_large["result_budget"]["max_features"] = (
        capabilities["finite"]["result"]["max_features"] + 1
    )
    with pytest.raises(GatewayFailure) as limited:
        validate_finite_admission(
            too_large,
            "histoforge/tubule",
            "tubule_semantic",
            backend,
            capabilities,
            source_store,
            files,
        )
    assert limited.value.error["code"] == "PAYLOAD_TOO_LARGE"
    assert source_store.references == [
        request["raster_input"]["source_view"]["artifact"]
    ]


def test_admission_accepts_only_a_pure_curated_inference_gpu_override():
    request, backend, capabilities, resolved, files = _admission_values()
    overridden = copy.deepcopy(request)
    overridden["expected_backend"]["gpu"] = "A100"

    assert (
        validate_finite_admission(
            overridden,
            "histoforge/tubule",
            "tubule_semantic",
            backend,
            capabilities,
            _SourceStore(resolved),
            files,
        )
        == overridden
    )

    unsupported = copy.deepcopy(overridden)
    unsupported["expected_backend"]["gpu"] = "T4"
    with pytest.raises(GatewayFailure) as failure:
        validate_finite_admission(
            unsupported,
            "histoforge/tubule",
            "tubule_semantic",
            backend,
            capabilities,
            _SourceStore(resolved),
            files,
        )
    assert failure.value.error["code"] == "VALIDATION_ERROR"


@pytest.mark.parametrize("failure", [None, "geometry", "corruption"])
def test_bounded_source_reads_finish_before_single_finite_prediction(
    monkeypatch, failure
):
    request, backend, capabilities, resolved, files = _admission_values()
    source_store = _SourceStore(resolved)
    config = replace(
        DEPLOYMENT_CONFIG,
        source_artifact_limits=replace(
            DEPLOYMENT_CONFIG.source_artifact_limits, max_complete_decode_pixels=4_096
        ),
    )
    footprint = request["raster_input"]["source_view"]["pixel_rect"]
    assert footprint["width"] * footprint["height"] > 4_096
    monkeypatch.setattr(finite_runtime, "DEPLOYMENT_CONFIG", config)
    monkeypatch.setattr(model_execution, "DEPLOYMENT_CONFIG", config)
    reads = []
    predictions = []

    scopes = []

    def read_source_view(rectangle):
        assert (
            rectangle["width"] * rectangle["height"]
            <= config.source_artifact_limits.max_complete_decode_pixels
        )
        reads.append(dict(rectangle))
        if len(reads) == 2 and failure == "corruption":
            raise ArtifactBytesCorrupt("source changed during bounded reads")
        read = copy.deepcopy(files.metadata)
        if len(reads) == 2 and failure == "geometry":
            read["dimensions"]["width"] += 1
        read["pixels"] = np.zeros(
            (rectangle["height"], rectangle["width"], 3), dtype=np.uint8
        )
        return read

    @contextmanager
    def source_view_reader(*_args):
        scopes.append("open")
        try:
            yield read_source_view
        finally:
            scopes.append("closed")

    def infer(image, budget):
        assert scopes == ["open", "closed"]
        assert len(reads) > 1
        assert image.pixels.shape == (384, 512, 3)
        assert not image.pixels.any()
        assert image.real_pixel_mask.all()
        assert budget == request["result_budget"]
        predictions.append(True)
        return {"type": "FeatureCollection", "features": []}

    monkeypatch.setattr(files, "source_view_reader", source_view_reader, raising=False)
    assert (
        validate_finite_admission(
            request,
            "histoforge/tubule",
            "tubule_semantic",
            backend,
            capabilities,
            source_store,
            files,
        )
        == request
    )
    assert reads == []
    outcome = execute_finite_request(
        request,
        {
            "backend": backend,
            "capabilities": capabilities,
            "model": {
                "output": {"type": "instance_segmentation", "classes": ["Tubule"]}
            },
        },
        source_store,
        files,
        infer,
    )
    assert scopes == ["open", "closed"]

    if failure is None:
        assert predictions == [True]
        assert outcome["type"] == "result"
        assert outcome["geojson"] == {"type": "FeatureCollection", "features": []}
    else:
        assert len(reads) == 2
        assert predictions == []
        assert outcome["type"] == "failure"
        assert outcome["error"]["code"] == (
            "VALIDATION_ERROR" if failure == "geometry" else "SERVICE_UNAVAILABLE"
        )
        assert "geojson" not in outcome


def test_worker_rejects_a_gpu_identity_different_from_its_activation():
    request, backend, capabilities, resolved, _files = _admission_values()
    activation = {
        "backend": copy.deepcopy(backend),
        "capabilities": capabilities,
        "model": {
            "output": {
                "type": "instance_segmentation",
                "classes": ["Tubule"],
            }
        },
    }
    request["expected_backend"]["gpu"] = "A100"

    outcome = execute_finite_request(
        request,
        activation,
        _SourceStore(resolved),
        object(),
        lambda *_args: pytest.fail("GPU mismatch must precede inference"),
    )

    assert outcome["type"] == "failure"
    assert outcome["error"]["code"] == "MODULE_IDENTITY_MISMATCH"


def test_backend_identity_accepts_only_the_reserved_commitless_result_ref():
    backend = copy.deepcopy(_wire_case()["request"]["expected_backend"])
    backend["module"].pop("resolved_commit_sha")
    backend["module"]["module_ref"] = (
        "workspace/run-44444444-4444-4444-8444-444444444444"
    )

    assert validate_backend_model_identity(backend)["module"] == backend["module"]

    backend["module"]["module_ref"] = "owner/module"
    with pytest.raises(ValueError, match="resolved commit or reserved"):
        validate_backend_model_identity(backend)


def test_runner_returns_compact_typed_outcome_and_gateway_adds_call_identity(
    monkeypatch,
):
    request, backend, capabilities, resolved, _files = _admission_values()
    source_store = _SourceStore(resolved)
    raster = np.zeros((384, 512, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: raster,
    )
    activation = {
        "backend": backend,
        "capabilities": capabilities,
        "model": {
            "output": {
                "type": "instance_segmentation_and_classification",
                "classChannels": [
                    {"channel": 0, "role": "background"},
                    {"channel": 1, "role": "emitted", "pathClass": "Tubule"},
                    {
                        "channel": 2,
                        "role": "emitted",
                        "pathClass": "Tubule (Not Intact)",
                    },
                ],
            }
        },
    }
    geojson = _wire_case()["result"]["geojson"]

    outcome = execute_finite_request(
        request,
        activation,
        source_store,
        object(),
        lambda actual, budget: (
            geojson if actual is raster and budget == request["result_budget"] else None
        ),
    )
    result = json.loads(encode_finite_result(outcome, "server/call?opaque"))

    assert outcome["type"] == "result"
    assert "backend_call_id" not in outcome
    assert result["meta"]["backend_call_id"] == "server/call?opaque"
    assert result["meta"]["backend"] == backend
    assert source_store.references == [
        request["raster_input"]["source_view"]["artifact"]
    ]


def test_runner_failures_are_closed_and_never_publish_partial_features(monkeypatch):
    request, backend, capabilities, resolved, _files = _admission_values()
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: np.zeros((384, 512, 3), dtype=np.uint8),
    )
    activation = {
        "backend": backend,
        "capabilities": capabilities,
        "model": {
            "output": {
                "type": "instance_segmentation_and_classification",
                "classChannels": [
                    {"channel": 0, "role": "background"},
                    {"channel": 1, "role": "emitted", "pathClass": "Tubule"},
                    {
                        "channel": 2,
                        "role": "emitted",
                        "pathClass": "Tubule (Not Intact)",
                    },
                ],
            }
        },
    }

    outcome = execute_finite_request(
        request,
        activation,
        _SourceStore(resolved),
        object(),
        lambda _image, _budget: (_ for _ in ()).throw(RuntimeError("secret detail")),
    )

    assert outcome == {
        "type": "failure",
        "error": {
            "schema_version": 1,
            "code": "RUNNER_FAILED",
            "message": "model prediction or output conversion failed",
            "resource_id": "histoforge/tubule/tubule_semantic",
        },
    }
    with pytest.raises(GatewayFailure) as failure:
        encode_finite_result(outcome, "call-id")
    assert failure.value.error["code"] == "RUNNER_FAILED"

    with pytest.raises(GatewayFailure) as malformed:
        encode_finite_result(
            {"type": "failure", "error": {"code": "RUNNER_FAILED"}},
            "call-id",
        )
    assert malformed.value.error["code"] == "RUNNER_FAILED"

    limited = execute_finite_request(
        request,
        activation,
        _SourceStore(resolved),
        object(),
        lambda _image, _budget: (_ for _ in ()).throw(
            ResultLimitExceeded("inference result exceeds max_features")
        ),
    )
    assert limited["error"]["code"] == "RESULT_LIMIT_EXCEEDED"


def test_runner_reports_missing_source_halo_as_insufficient_context(monkeypatch):
    request, backend, capabilities, resolved, _files = _admission_values()
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            InsufficientSourceContext(
                "source view omits an in-image bilinear contribution"
            )
        ),
    )

    outcome = execute_finite_request(
        request,
        {
            "backend": backend,
            "capabilities": capabilities,
            "model": {
                "output": {
                    "type": "instance_segmentation",
                    "classes": ["Tubule"],
                }
            },
        },
        _SourceStore(resolved),
        object(),
        lambda _image, _budget: None,
    )

    assert outcome["type"] == "failure"
    assert outcome["error"]["code"] == "INSUFFICIENT_CONTEXT"


@pytest.mark.parametrize(
    "corruption", ["class", "geometry", "id", "bounds", "record_limit"]
)
def test_worker_rejects_invalid_features_before_returning_a_success(
    monkeypatch, corruption
):
    request, backend, capabilities, resolved, _files = _admission_values()
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.rasterize_model_input",
        lambda *_args, **_kwargs: np.zeros((384, 512, 3), dtype=np.uint8),
    )
    geojson = _wire_case()["result"]["geojson"]
    feature = geojson["features"][0]
    if corruption == "class":
        feature["properties"]["path_class"] = "Undeclared"
    elif corruption == "geometry":
        feature["geometry"]["coordinates"] = [[[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]]
    elif corruption == "id":
        feature["id"] = "f" * 64
    elif corruption == "bounds":
        raw = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][0]
        geojson = json.loads(raw["outOfBoundsGeojsonCanonical"])
    else:
        request["result_budget"]["max_record_bytes"] = 1

    outcome = execute_finite_request(
        request,
        {
            "backend": backend,
            "capabilities": capabilities,
            "model": {
                "output": {
                    "type": "instance_segmentation_and_classification",
                    "classChannels": [
                        {"channel": 0, "role": "background"},
                        {"channel": 1, "role": "emitted", "pathClass": "Tubule"},
                        {
                            "channel": 2,
                            "role": "emitted",
                            "pathClass": "Tubule (Not Intact)",
                        },
                    ],
                }
            },
        },
        _SourceStore(resolved),
        object(),
        lambda *_args: geojson,
    )

    assert outcome["type"] == "failure"
    assert outcome["error"]["code"] == (
        "RESULT_LIMIT_EXCEEDED" if corruption == "record_limit" else "RUNNER_FAILED"
    )
    assert "geojson" not in outcome


def test_worker_reserves_space_for_the_longest_escaped_call_id():
    request = _wire_case()["request"]
    backend = request["expected_backend"]
    features = _wire_case()["result"]["geojson"]
    outcome = finite_runtime._success_outcome(
        request, backend, ["Tubule", "Tubule (Not Intact)"], features
    )
    reserved = len(encode_finite_result(outcome, "\x01" * 128))
    assert len(encode_finite_result(outcome, "short")) < reserved

    for field in ("max_transferred_bytes", "max_decoded_bytes"):
        bounded = copy.deepcopy(request)
        bounded["result_budget"][field] = reserved
        finite_runtime._success_outcome(
            bounded, backend, outcome["declared_classes"], features
        )
        bounded["result_budget"][field] -= 1
        with pytest.raises(GatewayFailure) as failure:
            finite_runtime._success_outcome(
                bounded, backend, outcome["declared_classes"], features
            )
        assert failure.value.error["code"] == "RESULT_LIMIT_EXCEEDED"


def test_admission_rejects_changed_backend_before_opening_source():
    request, backend, capabilities, resolved, files = _admission_values()

    changed_module = copy.deepcopy(backend)
    changed_module["module"]["content_digest"] = "0" * 64
    module_source = _SourceStore(resolved)
    with pytest.raises(GatewayFailure) as unavailable:
        validate_finite_admission(
            request,
            "histoforge/tubule",
            "tubule_semantic",
            changed_module,
            capabilities,
            module_source,
            files,
        )
    assert unavailable.value.error["code"] == "MODULE_VERSION_UNAVAILABLE"
    assert module_source.references == []

    changed_runner = copy.deepcopy(backend)
    changed_runner["runner_build"]["version"] = "different-runner"
    runner_source = _SourceStore(resolved)
    with pytest.raises(GatewayFailure) as mismatch:
        validate_finite_admission(
            request,
            "histoforge/tubule",
            "tubule_semantic",
            changed_runner,
            capabilities,
            runner_source,
            files,
        )
    assert mismatch.value.error["code"] == "MODULE_IDENTITY_MISMATCH"
    assert runner_source.references == []


def test_activation_binds_warm_key_to_current_content_and_exact_artifact(monkeypatch):
    model = {
        "params": {"num_classes": 0, "min_instance_pixels": 15},
        "output": {"type": "instance_segmentation", "classes": ["Cell"]},
    }
    current = _CurrentModules(model)
    capabilities = {"runner_build": {"version": "fixture"}}
    backend = {
        "architecture": "dinov3-adapter-cellpose",
        "runner_build": capabilities["runner_build"],
    }
    state = {"weight": object()}
    built = []

    monkeypatch.setattr(
        module_runtime, "current_module_snapshot", lambda: nullcontext(current)
    )
    monkeypatch.setattr(
        model_activation,
        "installed_model_descriptor",
        lambda *_args: {
            "runner_availability": {
                "type": "available",
                "capabilities": capabilities,
            }
        },
    )
    monkeypatch.setattr(
        model_activation, "installed_backend_identity", lambda *_args: backend
    )
    monkeypatch.setattr(safetensors.torch, "load_file", lambda path: state)

    def build(parameters, supplied):
        built.append((parameters, supplied))
        return "network"

    activation = activate_model(
        "owner/module",
        "model",
        "a" * 64,
        "c" * 40,
        "dinov3-adapter-cellpose",
        build,
    )

    assert activation == {
        "network": "network",
        "model": model,
        "backend": backend,
        "capabilities": capabilities,
    }
    assert built == [({"num_classes": 0, "min_instance_pixels": 15}, state)]
    assert current.artifact_requests == [("owner/module", "model")]

    with pytest.raises(GatewayFailure) as stale:
        activate_model(
            "owner/module",
            "model",
            "b" * 64,
            "c" * 40,
            "dinov3-adapter-cellpose",
            build,
        )
    assert stale.value.error["code"] == "MODULE_VERSION_UNAVAILABLE"

    with pytest.raises(GatewayFailure) as moved:
        activate_model(
            "owner/module",
            "model",
            "a" * 64,
            "d" * 40,
            "dinov3-adapter-cellpose",
            build,
        )
    assert moved.value.error["code"] == "MODULE_VERSION_UNAVAILABLE"


class _SourceStore:
    def __init__(self, resolved):
        self.resolved = resolved
        self.references = []

    def resolve_sealed_reference(self, reference):
        self.references.append(reference)
        return self.resolved


class _LevelFiles:
    def __init__(self, metadata):
        self.metadata = metadata
        self.calls = 0

    def source_level_metadata(self, *_args):
        self.calls += 1
        return self.metadata


class _CurrentModules:
    def __init__(self, model):
        self.model = model
        self.artifact_requests = []

    def read_record(self, _module_ref):
        return {"content_digest": "a" * 64, "resolved_commit_sha": "c" * 40}

    def load_recipe(self, _module_ref, *, record=None):
        assert record == self.read_record(_module_ref)
        return {"models": {"model": self.model}}

    def artifact_path(self, module_ref, model_id):
        self.artifact_requests.append((module_ref, model_id))
        return Path("model.safetensors")


def _admission_values():
    case = _wire_case()
    request = copy.deepcopy(case["request"])
    capabilities = runner_registry()["definitions"]["dinov3-adapter-hovernet"][
        "capabilities"
    ]
    backend = copy.deepcopy(request["expected_backend"])
    for field in (
        "runner_build",
        "rasterizer",
        "preprocessing_contract_digest",
    ):
        backend[field] = capabilities[field]
    request["expected_backend"] = backend
    raster_input = request["raster_input"]
    request["raster_input"] = build_model_raster_input(
        raster_input["source_view"],
        raster_input["frame"],
        capabilities["rasterizer"],
    )
    source = request["raster_input"]["frame"]["source_revision"]["pixel_source"]
    start = {
        "origin": {
            "type": "original-wsi",
            "source": source,
            "pyramid_series_index": 2,
            "level_zero_pixels_to_slide": {
                "scale_x": 1,
                "scale_y": 1,
                "translate_x": -12.5,
                "translate_y": 3.25,
            },
        },
        "container": "TIFF_FAMILY",
    }
    resolved = {
        "start_request": start,
        "manifest": {"container": "TIFF_FAMILY"},
        "artifact": request["raster_input"]["source_view"]["artifact"],
        "absolute_expires_at": "2026-08-23T00:00:00.000Z",
    }
    files = _LevelFiles(
        {
            "dimensions": {"width": 5000, "height": 3000},
            "selected_level_pixels_to_level_zero": {
                "scale_x": 2,
                "scale_y": 2,
                "translate_x": 0,
                "translate_y": 0,
            },
        }
    )
    return request, backend, capabilities, resolved, files


def _wire_case():
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][0]
    return {
        "request": json.loads(raw["requestCanonical"]),
        "result": json.loads(raw["resultCanonical"]),
    }
