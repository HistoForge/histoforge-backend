"""Offline checks of Cellpose's bounded prediction/geometry overlap.

Controlled futures check ordering; tiny spawned workers check actual process
failure. These tests do not qualify Modal admission, GPU snapshots, large IPC
memory, or forced container shutdown.
"""

import gc
import logging
import operator
import os
import threading
import weakref
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from multiprocessing import get_context
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.cellpose import runner as cellpose_runner
from histoforge.cloud.architectures.instance_runtime import InstanceModelRuntime
from histoforge.cloud.artifacts.volume import artifact_volume_gate
from histoforge.cloud.inference.feature_stream import EncodedFeatureStream
from histoforge.cloud.inference.features import ResultLimitExceeded
from histoforge.cloud.inference.runs import core_halo_output
from histoforge.cloud.inference.runs.core_halo import InsufficientCoreHaloContext
from histoforge.cloud.rasterization.rasterizer import RasterizedImage

_WAIT_SECONDS = 5
_CLASSES = ("Cell",)
_RASTER = {"width": 10, "height": 2}
_RECT = {"x": 0, "y": 0, **_RASTER}
_WINDOW = {"unit_id": 0, "core_rect": _RECT, "input_rect": _RECT}
_PLAN = {"windows": [_WINDOW]}
_LIMITS = {
    "max_transferred_bytes": 1_000_000,
    "max_decoded_bytes": 1_000_000,
    "max_record_bytes": 100_000,
    "max_features": 100,
    "max_vertices": 1_000,
}


class _QueuedExecutor:
    """Expose submitted work so a test controls completion without a process."""

    def __init__(self):
        self.submitted = Queue()
        self.futures = []
        self.shutdown_calls = []

    def submit(self, function, *args, **kwargs):
        future = Future()
        self.futures.append(future)
        self.submitted.put((function, args, kwargs, future))
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)

    def finish_pending(self):
        for future in self.futures:
            if not future.done():
                future.set_exception(RuntimeError("test cleanup"))


class _ObservedLock:
    """Report attempted acquisition before blocking, avoiding timing sleeps."""

    def __init__(self):
        self.lock = threading.Lock()
        self.attempts = Queue()

    def __enter__(self):
        self.attempts.put(None)
        assert self.lock.acquire(timeout=_WAIT_SECONDS)
        return self

    def __exit__(self, *_args):
        self.lock.release()


def _runner():
    runner = cellpose_runner.CellposeRunner._get_user_cls()()
    runner.ref = "owner/module"
    runner.model_id = "model"
    runner.gpu = "L40S"
    runner.content_digest = "a" * 64
    runner.resolved_commit_sha = "b" * 40
    runner.activation = {}
    runner.source_store = object()
    runner.source_files = object()
    runner.artifact_files = object()
    runner._model_operation_lock = threading.Lock()
    runner._postprocessing_unavailable = threading.Event()
    runner._geometry_executor = _QueuedExecutor()
    runner.runtime = SimpleNamespace(
        emitted_classes=_CLASSES,
        predict_window=lambda image, _plan, _window, **_kwargs: {
            "instance_mask": image
        },
        classifications=lambda _result: {1: "Cell"},
    )
    return runner


def _invoke(runner, name, *args, **kwargs):
    method = type(runner).__dict__[name]._get_raw_f()
    return method(runner, *args, **kwargs)


def _encode(runner, prediction=None):
    if prediction is None:
        prediction = (np.ones((2, 10), np.uint16), {1: "Cell"})
    return runner._encode_core_halo_window(
        prediction, _WINDOW, _CLASSES, _RASTER, _LIMITS
    )


def _encoded(data=b"encoded"):
    return EncodedFeatureStream(data, decoded_bytes=12, feature_count=1, vertex_count=4)


def _patch_finite_execution(monkeypatch, name, execute):
    path = (
        "histoforge.cloud.inference.finite.runtime.execute_finite_request"
        if name == "infer"
        else "histoforge.cloud.inference.runs.runtime.execute_durable_finite_input"
    )
    monkeypatch.setattr(path, execute)


def _finite(runner, name):
    arguments = ({},) if name == "infer" else ({}, 0, {})
    return _invoke(runner, name, *arguments)


def test_post_restore_starts_one_spawn_worker_and_waits_for_initialization(monkeypatch):
    runner = _runner()
    old_lock = runner._model_operation_lock
    old_signal = runner._postprocessing_unavailable
    old_signal.set()
    executor = _QueuedExecutor()
    context = object()
    constructed = []
    stores = (object(), object(), object())
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.mounted_inference_stores", lambda: stores
    )

    def get_context(method):
        assert method == "spawn"
        return context

    def create_executor(**kwargs):
        constructed.append(kwargs)
        return executor

    monkeypatch.setattr(cellpose_runner, "get_context", get_context)
    monkeypatch.setattr(cellpose_runner, "ProcessPoolExecutor", create_executor)
    with ThreadPoolExecutor(max_workers=1) as calls:
        restored = calls.submit(_invoke, runner, "post_restore")
        try:
            function, args, kwargs, ready = executor.submitted.get(
                timeout=_WAIT_SECONDS
            )
            assert function is core_halo_output.initialize_geometry_worker
            assert args == () and kwargs == {}
            assert constructed == [{"max_workers": 1, "mp_context": context}]
            assert not restored.done()
            ready.set_result(12345)
            restored.result(timeout=_WAIT_SECONDS)
        finally:
            executor.finish_pending()
    assert runner._geometry_executor is executor
    assert runner._model_operation_lock is not old_lock
    assert runner._postprocessing_unavailable is not old_signal
    assert not runner._postprocessing_unavailable.is_set()
    assert (runner.source_store, runner.source_files, runner.artifact_files) == stores


def test_worker_initialization_failure_aborts_post_restore(monkeypatch):
    runner = _runner()
    executor = _QueuedExecutor()
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.mounted_inference_stores",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr(cellpose_runner, "ProcessPoolExecutor", lambda **_kw: executor)
    with ThreadPoolExecutor(max_workers=1) as calls:
        restored = calls.submit(_invoke, runner, "post_restore")
        try:
            _function, _args, _kwargs, ready = executor.submitted.get(
                timeout=_WAIT_SECONDS
            )
            ready.set_exception(BrokenProcessPool("worker import failed"))
            with pytest.raises(BrokenProcessPool, match="worker import failed"):
                restored.result(timeout=_WAIT_SECONDS)
        finally:
            executor.finish_pending()


def test_real_spawn_initializer_failure_aborts_runner_startup(monkeypatch):
    runner = _runner()
    executors = []
    monkeypatch.setattr(
        "histoforge.cloud.inference.execution.mounted_inference_stores",
        lambda: (object(), object(), object()),
    )

    def create_failing_executor(**kwargs):
        assert kwargs["max_workers"] == 1
        assert kwargs["mp_context"].get_start_method() == "spawn"
        # A standard-library callable fails before the child accepts any work.
        # No model, mounted store, or test-module helper runs in the child.
        executor = ProcessPoolExecutor(
            **kwargs, initializer=partial(operator.truediv, 1, 0)
        )
        executors.append(executor)
        return executor

    monkeypatch.setattr(cellpose_runner, "ProcessPoolExecutor", create_failing_executor)
    try:
        with pytest.raises(BrokenProcessPool):
            _invoke(runner, "post_restore")
        assert len(executors) == 1
        assert runner._geometry_executor is executors[0]
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def test_snapshot_warmup_does_not_create_process_or_live_overlap_state(monkeypatch):
    runner = _runner()
    for name in (
        "_geometry_executor",
        "_model_operation_lock",
        "_postprocessing_unavailable",
    ):
        delattr(runner, name)
    warmups = []
    network = SimpleNamespace()
    network.to = lambda _device: network
    network.eval = lambda: network
    activation = {"network": network, "model": {}}
    monkeypatch.setattr(
        "histoforge.cloud.inference.activation.activate_model", lambda *_a: activation
    )
    monkeypatch.setattr(
        cellpose_runner, "instance_network_builder", lambda _a: object()
    )
    monkeypatch.setattr(
        cellpose_runner,
        "create_instance_runtime",
        lambda *_a, **_k: SimpleNamespace(
            num_classes=1, field_options={}, warmup=lambda: warmups.append(True)
        ),
    )

    def unexpected_pool(**_kwargs):
        pytest.fail("a process must not be created before snapshot restoration")

    monkeypatch.setattr(cellpose_runner, "ProcessPoolExecutor", unexpected_pool)
    _invoke(runner, "load_model")
    assert warmups == [True]
    assert runner.activation is activation
    assert not any(
        name in vars(runner)
        for name in (
            "_geometry_executor",
            "_model_operation_lock",
            "_postprocessing_unavailable",
        )
    )


def test_read_prediction_and_class_projection_are_one_serial_region():
    runner = _runner()
    observed = _ObservedLock()
    runner._model_operation_lock = observed
    entered = threading.Event()
    release = threading.Event()
    events = []

    def read(name):
        assert observed.lock.locked()
        events.append(("read", name))
        return name

    def predict(name, _plan, _window, *, channel_percentiles):
        assert observed.lock.locked()
        events.append(("predict", name))
        if name == "A":
            entered.set()
            assert release.wait(_WAIT_SECONDS)
        return {"instance_mask": name}

    def classify(result):
        assert observed.lock.locked()
        events.append(("classify", result["instance_mask"]))
        return {1: "Cell"}

    runner.runtime.predict_window = predict
    runner.runtime.classifications = classify
    with ThreadPoolExecutor(max_workers=2) as calls:
        first = calls.submit(
            runner._predict_core_halo_window,
            lambda: read("A"),
            _PLAN,
            _WINDOW,
            channel_percentiles=[[0.0, 255.0]] * 3,
        )
        try:
            observed.attempts.get(timeout=_WAIT_SECONDS)
            assert entered.wait(_WAIT_SECONDS)
            second = calls.submit(
                runner._predict_core_halo_window,
                lambda: read("B"),
                _PLAN,
                _WINDOW,
                channel_percentiles=[[0.0, 255.0]] * 3,
            )
            observed.attempts.get(timeout=_WAIT_SECONDS)
            assert events == [("read", "A"), ("predict", "A")]
            release.set()
            assert first.result(timeout=_WAIT_SECONDS) == ("A", {1: "Cell"})
            assert second.result(timeout=_WAIT_SECONDS) == ("B", {1: "Cell"})
        finally:
            release.set()
    assert events == [
        (phase, name)
        for name in ("A", "B")
        for phase in ("read", "predict", "classify")
    ]


def test_geometry_wait_allows_next_prediction_and_holds_no_volume_gate():
    runner = _runner()
    executor = runner._geometry_executor

    def execute(name):
        prediction = runner._predict_core_halo_window(
            lambda: name, _PLAN, _WINDOW, channel_percentiles=[[0.0, 255.0]] * 3
        )
        return _encode(runner, prediction)

    with ThreadPoolExecutor(max_workers=2) as calls:
        first = calls.submit(execute, "A")
        try:
            function, arguments, keywords, first_geometry = executor.submitted.get(
                timeout=_WAIT_SECONDS
            )
            assert function is core_halo_output.encode_instance_window
            assert arguments == (
                ("A", {1: "Cell"}),
                _WINDOW,
                _CLASSES,
                _RASTER,
                _LIMITS,
            )
            assert keywords == {}
            assert not first.done()
            assert artifact_volume_gate.acquire(timeout=_WAIT_SECONDS)
            artifact_volume_gate.release()
            second = calls.submit(execute, "B")
            _fn, args, _kw, second_geometry = executor.submitted.get(
                timeout=_WAIT_SECONDS
            )
            assert args[0] == ("B", {1: "Cell"})
            assert len(executor.futures) == 2
            assert not first.done() and not second.done()
            first_value, second_value = _encoded(b"A"), _encoded(b"B")
            first_geometry.set_result(first_value)
            second_geometry.set_result(second_value)
            assert first.result(timeout=_WAIT_SECONDS) is first_value
            assert second.result(timeout=_WAIT_SECONDS) is second_value
        finally:
            executor.finish_pending()


@pytest.mark.parametrize("name", ["infer", "infer_durable_input"])
def test_finite_execution_uses_same_lock_through_complete_operation(monkeypatch, name):
    runner = _runner()
    observed = _ObservedLock()
    runner._model_operation_lock = observed
    entered, release = threading.Event(), threading.Event()
    reads = []
    runner.runtime.predict = lambda _image: {"instance_mask": "labels"}
    monkeypatch.setattr(
        "histoforge.cloud.inference.mask_features.feature_collection_from_mask",
        lambda *_a: "features",
    )

    def execute(*args, **_kwargs):
        assert observed.lock.locked()
        entered.set()
        assert release.wait(_WAIT_SECONDS)
        # The callback must not acquire this non-reentrant lock a second time.
        assert args[-1](object(), {}) == "features"
        assert observed.lock.locked()
        return "complete finite publication"

    _patch_finite_execution(monkeypatch, name, execute)
    with ThreadPoolExecutor(max_workers=2) as calls:
        finite = calls.submit(_finite, runner, name)
        try:
            observed.attempts.get(timeout=_WAIT_SECONDS)
            assert entered.wait(_WAIT_SECONDS)
            prediction = calls.submit(
                runner._predict_core_halo_window,
                lambda: reads.append(True) or "image",
                _PLAN,
                _WINDOW,
                channel_percentiles=[[0.0, 255.0]] * 3,
            )
            observed.attempts.get(timeout=_WAIT_SECONDS)
            assert reads == []
            release.set()
            assert finite.result(timeout=_WAIT_SECONDS) == "complete finite publication"
            assert prediction.result(timeout=_WAIT_SECONDS) == ("image", {1: "Cell"})
        finally:
            release.set()


@pytest.mark.parametrize("name", ["infer", "infer_durable_input"])
def test_finite_execution_can_finish_while_geometry_peer_waits(monkeypatch, name):
    runner = _runner()
    executor = runner._geometry_executor

    def execute(*_args, **_kwargs):
        assert runner._model_operation_lock.locked()
        return "finite result"

    _patch_finite_execution(monkeypatch, name, execute)
    with ThreadPoolExecutor(max_workers=2) as calls:
        geometry = calls.submit(_encode, runner)
        try:
            _function, _args, _kwargs, pending = executor.submitted.get(
                timeout=_WAIT_SECONDS
            )
            finite = calls.submit(_finite, runner, name)
            assert finite.result(timeout=_WAIT_SECONDS) == "finite result"
            assert not geometry.done()
            value = _encoded()
            pending.set_result(value)
            assert geometry.result(timeout=_WAIT_SECONDS) is value
        finally:
            executor.finish_pending()


@pytest.mark.parametrize("name", ["window", "infer", "infer_durable_input"])
def test_unavailable_signal_is_checked_inside_lock_before_reading(monkeypatch, name):
    runner = _runner()
    observed = _ObservedLock()
    runner._model_operation_lock = observed

    def forbidden(*_args, **_kwargs):
        pytest.fail("unavailable worker must reject before reading or execution")

    if name != "window":
        _patch_finite_execution(monkeypatch, name, forbidden)
    observed.lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as calls:
        result = (
            calls.submit(
                runner._predict_core_halo_window,
                forbidden,
                _PLAN,
                _WINDOW,
                channel_percentiles=[[0.0, 255.0]] * 3,
            )
            if name == "window"
            else calls.submit(_finite, runner, name)
        )
        try:
            observed.attempts.get(timeout=_WAIT_SECONDS)
            runner._postprocessing_unavailable.set()
        finally:
            observed.lock.release()
        with pytest.raises(GatewayFailure) as failure:
            result.result(timeout=_WAIT_SECONDS)
    assert failure.value.error["code"] == "RUNNER_FAILED"


@pytest.mark.parametrize("failure_at", ["submit", "result"])
def test_broken_pool_marks_unavailable_and_retires_container(monkeypatch, failure_at):
    runner = _runner()
    broken = BrokenProcessPool("CPU worker died")
    future = Future()
    future.set_exception(broken)
    submitted, retirements = [], []

    def submit(*_args):
        submitted.append(True)
        if failure_at == "submit":
            raise broken
        return future

    def stop_fetching():
        assert runner._postprocessing_unavailable.is_set()
        retirements.append(True)

    runner._geometry_executor = SimpleNamespace(submit=submit)
    monkeypatch.setattr("modal.experimental.stop_fetching_inputs", stop_fetching)
    with pytest.raises(GatewayFailure) as failure:
        _encode(runner)
    assert failure.value.error["code"] == "RUNNER_FAILED"
    assert runner._postprocessing_unavailable.is_set()
    assert retirements == [True] and submitted == [True]


def test_broken_worker_fails_both_admitted_futures_without_replacement(monkeypatch):
    runner = _runner()
    executor = runner._geometry_executor
    retirements = []
    monkeypatch.setattr(
        "modal.experimental.stop_fetching_inputs", lambda: retirements.append(True)
    )
    with ThreadPoolExecutor(max_workers=2) as calls:
        results = [calls.submit(_encode, runner) for _ in range(2)]
        try:
            pending = [
                executor.submitted.get(timeout=_WAIT_SECONDS)[3] for _ in range(2)
            ]
            for future in pending:
                future.set_exception(BrokenProcessPool("worker terminated"))
            for result in results:
                with pytest.raises(GatewayFailure) as failure:
                    result.result(timeout=_WAIT_SECONDS)
                assert failure.value.error["code"] == "RUNNER_FAILED"
        finally:
            executor.finish_pending()
    assert runner._postprocessing_unavailable.is_set()
    assert retirements
    assert runner._geometry_executor is executor
    assert len(executor.futures) == 2 and executor.submitted.empty()


def test_real_spawn_child_death_reaches_runner_retirement(monkeypatch):
    runner = _runner()
    retirements = []

    def stop_fetching():
        assert runner._postprocessing_unavailable.is_set()
        retirements.append(True)

    monkeypatch.setattr("modal.experimental.stop_fetching_inputs", stop_fetching)
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn")
    ) as executor:
        runner._geometry_executor = executor
        child_pid = executor.submit(os.getpid).result(timeout=_WAIT_SECONDS)
        assert child_pid != os.getpid()
        # FIFO execution in the sole worker terminates it before the runner's
        # conversion. Submit may detect the death immediately, or its queued
        # future may report it; both must reach the same retirement boundary.
        death = executor.submit(os._exit, 17)
        with pytest.raises(GatewayFailure) as failure:
            _encode(runner)
        assert failure.value.error["code"] == "RUNNER_FAILED"
        assert isinstance(failure.value.__cause__, BrokenProcessPool)
        with pytest.raises(BrokenProcessPool):
            death.result(timeout=_WAIT_SECONDS)
        assert runner._geometry_executor is executor
        assert runner._postprocessing_unavailable.is_set()
        assert retirements == [True]


@pytest.mark.parametrize(
    "error",
    [
        ResultLimitExceeded("feature limit"),
        InsufficientCoreHaloContext("artificial boundary"),
        ValueError("invalid geometry"),
    ],
)
def test_ordinary_geometry_failure_preserves_successful_peer(monkeypatch, error):
    runner = _runner()
    executor = runner._geometry_executor

    def unexpected_retirement():
        pytest.fail("ordinary conversion failure must not retire a healthy worker")

    monkeypatch.setattr(
        "modal.experimental.stop_fetching_inputs", unexpected_retirement
    )
    with ThreadPoolExecutor(max_workers=2) as calls:
        failed = calls.submit(_encode, runner)
        try:
            failed_geometry = executor.submitted.get(timeout=_WAIT_SECONDS)[3]
            successful = calls.submit(_encode, runner)
            successful_geometry = executor.submitted.get(timeout=_WAIT_SECONDS)[3]
            failed_geometry.set_exception(error)
            value = _encoded()
            successful_geometry.set_result(value)
            with pytest.raises(type(error)) as failure:
                failed.result(timeout=_WAIT_SECONDS)
            assert failure.value is error
            assert successful.result(timeout=_WAIT_SECONDS) is value
        finally:
            executor.finish_pending()
    assert not runner._postprocessing_unavailable.is_set()


def test_rgb_support_and_fields_are_released_before_geometry_wait(monkeypatch):
    from histoforge._vendor.histoforge_models.pathology import cellpose_window_decode

    runner = _runner()
    executor = runner._geometry_executor
    references = []
    labels = np.repeat(np.arange(1, 6, dtype=np.int32), 2)[None, :].repeat(2, axis=0)
    classes = np.array([[0, 0, 1, 1, 2, 2, 3, 3, 2, 3]]).repeat(2, axis=0)

    def read_image():
        image = RasterizedImage(np.zeros((2, 10, 3), np.uint8), np.ones((2, 10), bool))
        references.extend(
            weakref.ref(value) for value in (image, image.pixels, image.real_pixel_mask)
        )
        return image

    def fields(*_args, **_kwargs):
        output = np.zeros((2, 10, 7), np.float32)
        np.put_along_axis(output, classes[..., None], 2, axis=-1)
        references.append(weakref.ref(output))
        return output

    monkeypatch.setattr(
        "histoforge._vendor.histoforge_models.pathology.window_maps."
        "predict_window_maps",
        fields,
    )
    monkeypatch.setattr(
        cellpose_window_decode,
        "compute_cellpose_masks_in_target_coordinates",
        lambda *_a, **_k: labels.copy(),
    )
    runner.runtime = InstanceModelRuntime(
        network=SimpleNamespace(device="cpu"),
        architecture="dinov3-adapter-cellpose",
        num_classes=4,
        field_options={"bsize": 32, "tile_overlap": 0.25},
        warmup_side=32,
        batch_size=1,
        decoder_options={"min_instance_pixels": 0},
        emitted_classes=("Positive", "Negative"),
        class_by_channel={2: "Positive", 3: "Negative"},
    )

    def execute():
        prediction = runner._predict_core_halo_window(
            read_image, _PLAN, _WINDOW, channel_percentiles=[[0.0, 255.0]] * 3
        )
        return _encode(runner, prediction)

    with ThreadPoolExecutor(max_workers=1) as calls:
        result = calls.submit(execute)
        try:
            _function, args, _kwargs, pending = executor.submitted.get(
                timeout=_WAIT_SECONDS
            )
            mask, classifications = args[0]
            np.testing.assert_array_equal(mask, labels)
            assert classifications == {3: "Positive", 4: "Negative", 5: "Positive"}
            gc.collect()
            assert len(references) == 4
            assert all(reference() is None for reference in references)
            assert not result.done()
            value = _encoded()
            pending.set_result(value)
            assert result.result(timeout=_WAIT_SECONDS) is value
        finally:
            executor.finish_pending()


def test_exit_requests_nonblocking_executor_cleanup():
    runner = _runner()
    executor = runner._geometry_executor
    _invoke(runner, "shutdown")
    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]


def test_worker_initializer_configures_only_its_logger_once(monkeypatch, capsys):
    logger = logging.Logger("geometry-worker-test")
    root = logging.getLogger()
    root_state = root.level, tuple(root.handlers)
    monkeypatch.setattr(core_halo_output, "_LOGGER", logger)

    assert core_halo_output.initialize_geometry_worker() == os.getpid()
    assert logger.level == logging.INFO
    assert not logger.propagate
    assert len(logger.handlers) == 1
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert core_halo_output.initialize_geometry_worker() == os.getpid()
    assert logger.handlers == [handler]
    logger.info("geometry-worker-stage")
    assert capsys.readouterr().err.count("geometry-worker-stage") == 1
    assert (root.level, tuple(root.handlers)) == root_state
