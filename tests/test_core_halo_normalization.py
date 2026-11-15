"""One complete-target pixel population feeds every supported window family."""

from contextlib import contextmanager

import numpy as np
import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference.runs import core_halo_runtime as runtime
from histoforge.cloud.rasterization.rasterizer import RasterizedImage

from test_core_halo_runtime import _geometry, _values


@pytest.mark.parametrize(
    "architecture",
    ["dinov3-adapter-cellpose", "dinov3-adapter-hovernet", "dinov3-adapter-semantic"],
)
def test_percentiles_include_mask_disjoint_cores_once_and_exclude_padding(
    monkeypatch, architecture
):
    request, activation, limits, plan = _values(architecture=architecture)
    # This mask excludes most pixels. Real context must still set the scaling.
    request["inputs"][0]["raster_input"]["frame"]["inclusion_mask_raster"] = _geometry(
        4, 4
    )
    # Preserve all frame and input digests through the normal builder.
    from histoforge.cloud.rasterization.inputs import (
        build_model_raster_input,
        build_raster_frame,
    )

    raster = request["inputs"][0]["raster_input"]
    frame = {
        key: value
        for key, value in raster["frame"].items()
        if key not in {"frame_digest", "inclusion_mask_digest"}
    }
    request["inputs"][0]["raster_input"] = build_model_raster_input(
        raster["source_view"], build_raster_frame(frame), raster["rasterizer"]
    )
    pixels = np.random.default_rng(137).integers(0, 256, (384, 512, 3), np.uint8)
    support = np.ones((384, 512), bool)
    support[:, -13:] = False
    support[:3] = False
    pixels[~support] = 255
    coverage = np.zeros(support.shape, np.uint8)
    opened = closed = 0
    monkeypatch.setattr(runtime, "_NORMALIZATION_BLOCK_SIDE", 61)

    class Reader:
        @contextmanager
        def open_reader(self):
            nonlocal opened, closed
            opened += 1

            def read(rect):
                assert rect["width"] <= 61 and rect["height"] <= 61
                region = (
                    slice(rect["y"], rect["y"] + rect["height"]),
                    slice(rect["x"], rect["x"] + rect["width"]),
                )
                coverage[region] += 1
                return RasterizedImage(pixels[region], support[region])

            try:
                yield read
            finally:
                closed += 1

    prepared_cores = []

    def prepare(*args):
        prepared_cores.append(args[-1])
        return Reader()

    monkeypatch.setattr(runtime, "_prepare_raster", prepare)
    values = runtime.collect_core_halo_percentiles(
        request, 0, limits, activation, object(), object(), is_active=lambda: True
    )
    expected = np.percentile(pixels.astype(np.float32)[support], (1, 99), axis=0).T
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))
    assert prepared_cores == [window["core_rect"] for window in plan["windows"]]
    assert opened == closed == len(plan["windows"])


def test_cancellation_closes_reader_and_never_continues_scan(monkeypatch):
    request, activation, limits, _ = _values()
    events = []
    checks = iter([True, True, False])
    monkeypatch.setattr(runtime, "_NORMALIZATION_BLOCK_SIDE", 61)

    class Reader:
        @contextmanager
        def open_reader(self):
            events.append("open")

            def read(rect):
                events.append("read")
                return RasterizedImage(
                    np.zeros((rect["height"], rect["width"], 3), np.uint8),
                    np.ones((rect["height"], rect["width"]), bool),
                )

            try:
                yield read
            finally:
                events.append("close")

    monkeypatch.setattr(runtime, "_prepare_raster", lambda *_: Reader())
    assert (
        runtime.collect_core_halo_percentiles(
            request,
            0,
            limits,
            activation,
            object(),
            object(),
            is_active=lambda: next(checks),
        )
        is None
    )
    assert events == ["open", "read", "close"]


def test_source_failure_closes_reader_and_retains_typed_failure(monkeypatch):
    request, activation, limits, _ = _values()
    closed = []

    class Reader:
        @contextmanager
        def open_reader(self):
            def read(_):
                raise OSError("unreadable source")

            try:
                yield read
            finally:
                closed.append(True)

    monkeypatch.setattr(runtime, "_prepare_raster", lambda *_: Reader())
    with pytest.raises(GatewayFailure, match="could not be read") as caught:
        runtime.collect_core_halo_percentiles(
            request, 0, limits, activation, object(), object(), is_active=lambda: True
        )
    assert caught.value.error["code"] == "SERVICE_UNAVAILABLE"
    assert closed == [True]


@pytest.mark.parametrize(
    "architecture",
    ["dinov3-adapter-cellpose", "dinov3-adapter-hovernet", "dinov3-adapter-semantic"],
)
def test_core_execution_refuses_local_normalization_fallback(monkeypatch, architecture):
    request, activation, limits, _ = _values(architecture=architecture)
    monkeypatch.setattr(
        runtime,
        "_prepare_raster",
        lambda *_: pytest.fail("read before statistics check"),
    )
    with pytest.raises(ValueError, match="requires target percentiles"):
        runtime.execute_core_halo_unit(
            request,
            0,
            0,
            limits,
            activation,
            object(),
            object(),
            object(),
            lambda *_: pytest.fail("predicted"),
            lambda *_: pytest.fail("encoded"),
        )


@pytest.mark.parametrize(
    "architecture",
    ["dinov3-adapter-cellpose", "dinov3-adapter-hovernet", "dinov3-adapter-semantic"],
)
def test_captured_population_requires_complete_sources_before_any_read(
    monkeypatch, architecture
):
    from test_model_run_source_delivery import _binding, _captured_request
    from test_model_run_store import _limits

    request = _captured_request()
    request["expected_backend"]["architecture"] = architecture
    activation = {
        "backend": request["expected_backend"],
        "capabilities": {
            key: request["expected_backend"][key]
            for key in ("runner_build", "rasterizer", "preprocessing_contract_digest")
        },
        "model": {"output": {"type": "instance_segmentation", "classes": ["Tubule"]}},
    }
    monkeypatch.setattr(
        runtime,
        "_prepare_raster",
        lambda *_: pytest.fail("incomplete target reached source read"),
    )
    with pytest.raises(ValueError, match="every input source"):
        runtime.collect_core_halo_percentiles(
            request,
            request["inputs"][0]["input_id"],
            _limits(),
            activation,
            object(),
            object(),
            [_binding(request, 0)],
            is_active=lambda: True,
        )


@pytest.mark.parametrize("family", ["cellpose", "hovernet"])
def test_runner_passes_statistics_per_call_and_cellpose_keeps_allocation_lock(family):
    from importlib import import_module
    from threading import Event, Lock
    from types import SimpleNamespace

    wrapper = import_module(f"histoforge.cloud.architectures.{family}.runner")
    cls = wrapper.CellposeRunner if family == "cellpose" else wrapper.HovernetRunner
    runner = cls._get_user_cls()()
    runner._model_operation_lock = Lock()
    runner._postprocessing_unavailable = Event()
    seen = []
    values = []

    def read():
        if family == "cellpose":
            assert runner._model_operation_lock.locked()
        return "raw"

    def predict(image, plan, window, *, channel_percentiles):
        assert image == "raw" and plan == "plan" and window == "window"
        assert channel_percentiles is values[-1]
        if family == "cellpose":
            assert runner._model_operation_lock.locked()
        seen.append(channel_percentiles)
        return {"instance_mask": "decoded"}

    runner.runtime = SimpleNamespace(
        predict_window=predict, classifications=lambda _: "Tubule"
    )
    for high in (141.75, 243.25):
        values.append([[7.0, high]] * 3)
        assert runner._predict_core_halo_window(
            read, "plan", "window", channel_percentiles=values[-1]
        ) == ("decoded", "Tubule")
        assert not runner._model_operation_lock.locked()
    assert seen == values
    assert not hasattr(runner, "channel_percentiles")
