"""Production timing records are coarse completed wall spans, not GPU claims."""

import io
import logging
import runpy
from types import SimpleNamespace

import pytest
from histoforge.cloud import _app
from histoforge.cloud.inference import execution
from histoforge.cloud.inference.runs import runtime


def test_worker_app_exposes_backend_and_scientific_logs_without_duplicate_handlers(
    monkeypatch,
):
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    root = logging.getLogger()
    root_configuration = (root.level, list(root.handlers))
    shared_app, shared_image = _app.app, _app.image
    stage_loggers = [
        logging.getLogger(name)
        for name in (
            "histoforge.cloud.architectures",
            "histoforge.cloud.inference",
            "histoforge._vendor.histoforge_models.pathology",
        )
    ]
    for logger in stage_loggers:
        monkeypatch.setattr(logger, "handlers", [handler])
        monkeypatch.setattr(logger, "level", logging.WARNING)
        monkeypatch.setattr(logger, "propagate", True)

    for _ in range(2):
        # Execute the actual app entry without replacing the Modal objects that
        # already-imported runners and lifecycle functions share.
        runpy.run_path(_app.__file__, run_name="inference_stage_logging_test")
        logging.getLogger("histoforge.cloud.architectures.instance_runtime").info(
            "stage=field_prediction elapsed_seconds=1.0"
        )
        logging.getLogger(
            "histoforge._vendor.histoforge_models.pathology.flow_inference"
        ).info("stage=decode_and_classify elapsed_seconds=2.0")

    assert all(logger.handlers == [handler] for logger in stage_loggers)
    assert all(not logger.propagate for logger in stage_loggers)
    assert (
        output.getvalue().splitlines()
        == [
            "stage=field_prediction elapsed_seconds=1.0",
            "stage=decode_and_classify elapsed_seconds=2.0",
        ]
        * 2
    )
    assert (root.level, root.handlers) == root_configuration
    assert _app.app is shared_app
    assert _app.image is shared_image


def test_model_input_logs_completed_source_and_validation_spans(monkeypatch, caplog):
    monkeypatch.setattr(
        logging.getLogger("histoforge.cloud.inference"), "handlers", [caplog.handler]
    )
    clock = iter([1.0, 4.0, 10.0, 12.0])
    monkeypatch.setattr(execution, "perf_counter", lambda: next(clock))
    raster = object()
    events = []
    features = {"type": "FeatureCollection", "features": []}

    def resolve(_artifact):
        events.append("resolve")
        return object()

    def rasterize(*_args, **_kwargs):
        events.append("rasterize")
        return raster

    def infer(image, budget):
        assert image is raster
        assert budget == {}
        events.append("infer")
        return features

    def validate(value, *_args):
        assert value is features
        events.append("validate")
        return value

    monkeypatch.setattr(execution, "model_input_source_artifact", lambda *_: {})
    monkeypatch.setattr(execution, "rasterize_model_input", rasterize)
    monkeypatch.setattr(execution, "model_declared_classes", lambda *_: ["Cell"])
    monkeypatch.setattr(execution, "validate_inference_features", validate)
    actual = execution.execute_model_input(
        {"frame": {"transform": {"raster": {}}}},
        {},
        {
            "backend": {"module": {"module_ref": "owner/module"}, "model_id": "cells"},
            "capabilities": {"source_containers": []},
            "model": {"output": {}},
        },
        SimpleNamespace(resolve_sealed_reference=resolve),
        object(),
        infer,
    )
    assert actual is features
    assert events == ["resolve", "rasterize", "infer", "validate"]
    assert caplog.messages == [
        "stage=source_read_and_rasterize elapsed_seconds=3.000000",
        "stage=feature_validation elapsed_seconds=2.000000",
    ]


@pytest.mark.parametrize("publication_fails", [False, True])
def test_publication_timing_waits_for_storage_and_never_logs_failed_success(
    monkeypatch, caplog, publication_fails
):
    monkeypatch.setattr(
        logging.getLogger("histoforge.cloud.inference"), "handlers", [caplog.handler]
    )
    clock = iter([10.0, 17.0])
    monkeypatch.setattr(runtime, "perf_counter", lambda: next(clock))
    budget = {"max_transferred_bytes": 100, "max_decoded_bytes": 100}
    request = {
        "expected_backend": {},
        "inputs": [{"input_id": 0, "raster_input": {}}],
        "result_budget": budget,
    }
    monkeypatch.setattr(
        runtime, "validate_model_run_resource_limits", lambda *_: request
    )
    monkeypatch.setattr(runtime, "require_activation_identity", lambda *_: None)
    monkeypatch.setattr(
        runtime, "build_expected_model_execution_plans", lambda *_: [{"type": "finite"}]
    )
    monkeypatch.setattr(runtime, "execute_model_input", lambda *_a, **_k: {})
    monkeypatch.setattr(runtime, "model_declared_classes", lambda *_: ["Cell"])
    monkeypatch.setattr(runtime, "build_backend_input_identity", lambda *_: {})
    monkeypatch.setattr(
        runtime,
        "encode_feature_stream",
        lambda *_: SimpleNamespace(
            data=b"encoded", decoded_bytes=7, feature_count=0, vertex_count=0
        ),
    )

    def publish(data, **_kwargs):
        assert data == b"encoded"
        assert caplog.messages == []
        if publication_fails:
            raise OSError("injected storage failure")
        return {"published": True}

    args = (
        request,
        0,
        {},
        {"capabilities": {"finite": {"result": budget}}, "model": {"output": {}}},
        object(),
        object(),
        SimpleNamespace(publish_artifact_bytes=publish),
        lambda *_: {},
    )
    if publication_fails:
        with pytest.raises(OSError, match="injected storage failure"):
            runtime.execute_durable_finite_input(*args)
        assert caplog.messages == []
    else:
        result = runtime.execute_durable_finite_input(*args)
        assert result["input"]["feature_stream_artifact"] == {"published": True}
        assert caplog.messages == [
            "stage=result_encode_and_publish elapsed_seconds=7.000000"
        ]
