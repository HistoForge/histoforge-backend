"""Generic durable worker resumes only complete authoritative boundaries."""

import threading
from concurrent.futures import wait

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.registry import runner_build_for_precision
from histoforge.cloud.inference.runs import executor as model_run_executor
from histoforge.cloud.inference.runs.executor import (
    execute_reserved_model_run,
)


class _Lifecycle:
    def __init__(self, context):
        self.context = context
        self.published = []
        self.published_units = []
        self.published_unit_failures = []
        self.completed = []
        self.failed = []

    def begin_execution(self, model_run_id):
        assert model_run_id == "run-id"
        return self.context

    def publish_input(self, model_run_id, publication, decoded_bytes):
        self.published.append((model_run_id, publication, decoded_bytes))
        return True

    def publish_unit(self, model_run_id, publication):
        self.published_units.append((model_run_id, publication))
        return True

    def publish_unit_failure(self, model_run_id, publication):
        self.published_unit_failures.append((model_run_id, publication))
        return True

    def complete(self, model_run_id):
        self.completed.append(model_run_id)
        return True

    def fail(self, model_run_id, failure):
        self.failed.append((model_run_id, failure))
        return True


class _Runner:
    def __init__(self, failure=None):
        self.failure = failure
        self.inputs = []

    def execute_input(self, request, input_id, limits, source_bindings):
        assert source_bindings == []
        self.inputs.append((request, input_id, limits))
        if self.failure is not None:
            raise self.failure
        return {"input": {"input_id": input_id}, "decoded_bytes": input_id + 10}


def test_executor_skips_completed_inputs_and_completes_once():
    context = _context()
    context["completed_input_ids"] = [3]
    lifecycle = _Lifecycle(context)
    runner = _Runner()

    assert execute_reserved_model_run("run-id", lifecycle, lambda request: runner)

    assert [item[1] for item in runner.inputs] == [7]
    assert lifecycle.published == [
        ("run-id", {"input_id": 7}, 17),
    ]
    assert lifecycle.completed == ["run-id"]
    assert lifecycle.failed == []


def test_executor_closes_typed_and_untyped_runner_failure():
    for error, expected in (
        (
            GatewayFailure("SERVICE_UNAVAILABLE", "source unavailable"),
            "SERVICE_UNAVAILABLE",
        ),
        (RuntimeError("private failure"), "RUNNER_FAILED"),
    ):
        lifecycle = _Lifecycle(_context())
        runner = _Runner(error)

        assert not execute_reserved_model_run(
            "run-id", lifecycle, lambda request: runner
        )
        assert lifecycle.completed == []
        assert lifecycle.failed[0][1]["code"] == expected
        assert "private" not in lifecycle.failed[0][1]["message"]


def test_executor_does_nothing_after_terminal_fence():
    lifecycle = _Lifecycle(None)
    created = []

    assert not execute_reserved_model_run(
        "run-id", lifecycle, lambda request: created.append(request)
    )
    assert created == []
    assert lifecycle.published == []
    assert lifecycle.completed == []


@pytest.mark.parametrize("precision", ["default", "bf16"])
def test_remote_runner_captures_gpu_and_precision_in_the_warm_key(
    monkeypatch, precision
):
    observed = {}

    class RunnerClass:
        def with_options(self, **options):
            observed["options"] = options
            return self

        def __call__(self, **parameters):
            observed["parameters"] = parameters
            return object()

    monkeypatch.setattr(
        model_run_executor.modal.Cls,
        "from_name",
        lambda app_name, class_name: (
            observed.update(app_name=app_name, class_name=class_name) or RunnerClass()
        ),
    )
    backend = {
        "architecture": "dinov3-adapter-cellpose",
        "runner_build": runner_build_for_precision(
            "dinov3-adapter-cellpose", precision
        ),
        "gpu": "A10G",
        "model_id": "leydig_nuclear",
        "module": {
            "module_ref": "histoforge/leydig",
            "content_digest": "a" * 64,
            "resolved_commit_sha": "b" * 40,
        },
    }

    runner = model_run_executor._remote_runner({"expected_backend": backend})

    assert runner.core_halo_parallelism == 8
    assert observed["options"] == {"gpu": "A10G"}
    assert observed["parameters"] == {
        "ref": "histoforge/leydig",
        "model_id": "leydig_nuclear",
        "content_digest": "a" * 64,
        "resolved_commit_sha": "b" * 40,
        "gpu": "A10G",
        "inference_precision": precision,
    }


def test_executor_reuses_completed_core_without_a_source_wide_pass():
    context = _context()
    context["request"]["inputs"] = [_input(3)]
    retained = {"input_id": 3, "unit_id": 0, "state": "retained"}
    context["completed_units"] = [retained]
    plan = _core_halo_plan()
    context["execution_plans"] = [{"input_id": 3, "plan": plan}]
    lifecycle = _Lifecycle(context)

    class CoreRunner:
        def __init__(self):
            self.units = []
            self.finalized = []

        def execute_core_halo_unit(
            self, request, input_id, unit_id, limits, source_bindings
        ):
            assert source_bindings == []
            self.units.append((request, input_id, unit_id, limits))
            return {"input_id": input_id, "unit_id": unit_id, "state": "new"}

        def finalize_core_halo_input(
            self, request, input_id, limits, units, source_bindings
        ):
            assert source_bindings == []
            self.finalized.append((request, input_id, limits, units))
            return {"input": {"input_id": input_id}, "decoded_bytes": 0}

    runner = CoreRunner()

    assert execute_reserved_model_run("run-id", lifecycle, lambda _request: runner)

    assert [item[2] for item in runner.units] == [1]
    assert lifecycle.published_units == [
        ("run-id", {"input_id": 3, "unit_id": 1, "state": "new"})
    ]
    assert runner.finalized[0][3] == [
        retained,
        {"input_id": 3, "unit_id": 1, "state": "new"},
    ]
    assert lifecycle.published == [("run-id", {"input_id": 3}, 0)]
    assert lifecycle.completed == ["run-id"]


def test_executor_publishes_core_failure_before_terminalizing():
    context = _context()
    context["request"]["inputs"] = [_input(3)]
    plan = _core_halo_plan() | {"windows": [_core_halo_plan()["windows"][0]]}
    context["execution_plans"] = [{"input_id": 3, "plan": plan}]
    lifecycle = _Lifecycle(context)

    class FailingRunner:
        def execute_core_halo_unit(self, *_args, source_bindings):
            assert source_bindings == []
            raise GatewayFailure("SERVICE_UNAVAILABLE", "runner unavailable")

    assert not execute_reserved_model_run(
        "run-id", lifecycle, lambda _request: FailingRunner()
    )
    assert lifecycle.published_unit_failures == [
        (
            "run-id",
            {
                "input_id": 3,
                "unit_id": 0,
                "failure": lifecycle.failed[0][1],
            },
        )
    ]
    assert lifecycle.failed[0][1]["code"] == "SERVICE_UNAVAILABLE"
    assert lifecycle.completed == []


@pytest.mark.parametrize("parallelism", [4, 8])
def test_executor_bounds_parallel_core_work_and_finalizes_in_canonical_order(
    parallelism,
):
    context = _context()
    context["request"]["inputs"] = [_input(3)]
    windows = [
        {
            "unit_id": unit_id,
            "core_rect": {"x": unit_id, "y": 0, "width": 1, "height": 1},
        }
        for unit_id in range(16)
    ]
    context["execution_plans"] = [
        {"input_id": 3, "plan": _core_halo_plan() | {"windows": windows}}
    ]
    lifecycle = _Lifecycle(context)

    class ParallelRunner:
        core_halo_parallelism = parallelism

        def __init__(self):
            self.barrier = threading.Barrier(parallelism)
            self.lock = threading.Lock()
            self.active = 0
            self.peak = 0
            self.finalized = []

        def _bounded(self):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            self.barrier.wait(timeout=5)
            with self.lock:
                self.active -= 1

        def execute_core_halo_unit(
            self, _request, input_id, unit_id, _limits, source_bindings
        ):
            assert source_bindings == []
            self._bounded()
            return {"input_id": input_id, "unit_id": unit_id, "state": "new"}

        def finalize_core_halo_input(
            self, _request, input_id, _limits, units, source_bindings
        ):
            assert source_bindings == []
            self.finalized = units
            return {"input": {"input_id": input_id}, "decoded_bytes": 0}

    runner = ParallelRunner()

    assert execute_reserved_model_run("run-id", lifecycle, lambda _request: runner)

    assert runner.peak == parallelism
    assert [unit["unit_id"] for unit in runner.finalized] == list(range(16))
    assert sorted(item[1]["unit_id"] for item in lifecycle.published_units) == list(
        range(16)
    )


def test_executor_drains_ready_peers_without_refill_after_observed_core_failure(
    monkeypatch,
):
    # Make all four initial completions visible together, including the failure.
    monkeypatch.setattr(model_run_executor, "wait", lambda futures, **_: wait(futures))
    context = _context()
    context["request"]["inputs"] = [_input(3)]
    windows = [
        {
            "unit_id": unit_id,
            "core_rect": {"x": unit_id, "y": 0, "width": 1, "height": 1},
        }
        for unit_id in range(8)
    ]
    context["execution_plans"] = [
        {"input_id": 3, "plan": _core_halo_plan() | {"windows": windows}}
    ]
    lifecycle = _Lifecycle(context)

    class FailingParallelRunner:
        core_halo_parallelism = 4

        def __init__(self):
            self.barrier = threading.Barrier(4)
            self.executed = []
            self.lock = threading.Lock()

        def execute_core_halo_unit(
            self, _request, input_id, unit_id, _limits, source_bindings
        ):
            assert source_bindings == []
            with self.lock:
                self.executed.append(unit_id)
            self.barrier.wait(timeout=5)
            if unit_id == 0:
                raise GatewayFailure("SERVICE_UNAVAILABLE", "unit unavailable")
            return {"input_id": input_id, "unit_id": unit_id, "state": "new"}

    runner = FailingParallelRunner()

    assert not execute_reserved_model_run("run-id", lifecycle, lambda _request: runner)

    assert sorted(runner.executed) == [0, 1, 2, 3]
    assert sorted(item[1]["unit_id"] for item in lifecycle.published_units) == [1, 2, 3]
    assert lifecycle.published_unit_failures == [
        (
            "run-id",
            {
                "input_id": 3,
                "unit_id": 0,
                "failure": lifecycle.failed[0][1],
            },
        )
    ]
    assert lifecycle.failed[0][1]["code"] == "SERVICE_UNAVAILABLE"
    assert lifecycle.completed == []


def test_executor_terminalizes_retained_failed_unit_without_dispatch():
    context = _context()
    context["failed_units"] = [
        {
            "input_id": 3,
            "unit_id": 0,
            "failure": {
                "schema_version": 1,
                "code": "RUNNER_FAILED",
                "message": "retained unit failure",
            },
        }
    ]
    lifecycle = _Lifecycle(context)
    created = []

    assert not execute_reserved_model_run(
        "run-id", lifecycle, lambda request: created.append(request)
    )

    assert created == []
    assert lifecycle.failed == [("run-id", context["failed_units"][0]["failure"])]


def _context():
    request = {
        "producer": {"module_ref": "owner/module", "step_id": "model"},
        # Exercise independent-window coordination without a real runner's
        # shared-normalization policy. Admission is outside this injected fixture.
        "expected_backend": {"architecture": "test-independent-windows"},
        "inputs": [_input(3), _input(7)],
    }
    return {
        "request": request,
        "limits": {"limit": True},
        "execution_deadline": "2026-08-17T18:00:00.000Z",
        "execution_plans": [
            {"input_id": 3, "plan": {"type": "finite"}},
            {"input_id": 7, "plan": {"type": "finite"}},
        ],
        "completed_units": [],
        "checkpoints": [],
        "failed_units": [],
        "completed_input_ids": [],
        "source_bindings": [],
    }


def _input(input_id):
    return {"input_id": input_id, "raster_input": {"source_view": {}}}


def _core_halo_plan():
    return {
        "type": "core-halo",
        "windows": [
            {
                "unit_id": 0,
                "core_rect": {"x": 0, "y": 0, "width": 1, "height": 1},
            },
            {
                "unit_id": 1,
                "core_rect": {"x": 1, "y": 0, "width": 1, "height": 1},
            },
        ],
    }
