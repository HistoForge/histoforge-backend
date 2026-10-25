"""Image delivery waits on the CPU without delaying retained GPU results."""

from threading import Event

from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference.runs.executor import execute_reserved_model_run

from test_model_run_executor import _context, _core_halo_plan, _Lifecycle


def _captured_context(core_halo=False):
    context = _context()
    context["request"]["inputs"] = [
        {"input_id": 3, "raster_input": {"source_view": {"capture": {}}}}
    ]
    context["execution_plans"] = [
        {"input_id": 3, "plan": _core_halo_plan() if core_halo else {"type": "finite"}}
    ]
    return context


def _binding(unit_id):
    return {"input_id": 3, "unit_id": unit_id, "source": {"file": unit_id}}


def test_finite_unit_is_dispatched_only_after_binding_arrives():
    context = _captured_context()

    class Lifecycle(_Lifecycle):
        polls = 0

        def source_delivery(self, _run_id):
            self.polls += 1
            return {
                "active": True,
                "source_bindings": [] if self.polls == 1 else [_binding(0)],
            }

    lifecycle = Lifecycle(context)

    class Runner:
        def execute_input(self, _request, input_id, _limits, source_bindings):
            assert lifecycle.polls == 2
            assert source_bindings == [_binding(0)]
            return {"input": {"input_id": input_id}, "decoded_bytes": 0}

    assert execute_reserved_model_run("run-id", lifecycle, lambda _: Runner())
    assert lifecycle.completed == ["run-id"]


def test_terminal_run_stops_waiting_without_a_gpu_call():
    class Lifecycle(_Lifecycle):
        def source_delivery(self, _run_id):
            return {"active": False, "source_bindings": []}

    lifecycle = Lifecycle(_captured_context())
    assert not execute_reserved_model_run("run-id", lifecycle, lambda _: object())
    assert lifecycle.failed == lifecycle.completed == lifecycle.published == []


def test_ready_core_is_published_while_its_peer_waits_for_an_image():
    class Lifecycle(_Lifecycle):
        polls = 0

        def source_delivery(self, _run_id):
            self.polls += 1
            # The second image arrives only after the first result is visible.
            # A bound prevents a broken coordinator from hanging this test.
            bindings = [_binding(0)]
            if self.published_units:
                bindings.append(_binding(1))
            return {"active": self.polls < 10, "source_bindings": bindings}

    lifecycle = Lifecycle(_captured_context(core_halo=True))

    class Runner:
        core_halo_parallelism = 2

        def execute_core_halo_unit(
            self, _request, input_id, unit_id, _limits, source_bindings
        ):
            assert source_bindings == [_binding(unit_id)]
            if unit_id == 1:
                assert lifecycle.published_units[0][1]["unit_id"] == 0
            return {"input_id": input_id, "unit_id": unit_id}

        def finalize_core_halo_input(
            self, _request, input_id, _limits, units, source_bindings
        ):
            assert [item["unit_id"] for item in units] == [0, 1]
            assert source_bindings == [_binding(0), _binding(1)]
            return {"input": {"input_id": input_id}, "decoded_bytes": 0}

    assert execute_reserved_model_run("run-id", lifecycle, lambda _: Runner())
    assert lifecycle.failed == []


def test_failed_gpu_peer_interrupts_upload_wait_without_losing_its_error():
    first_started = Event()
    second_waiting = Event()

    class Lifecycle(_Lifecycle):
        polls = 0

        def source_delivery(self, _run_id):
            self.polls += 1
            if first_started.is_set():
                second_waiting.set()
            return {"active": self.polls < 6, "source_bindings": [_binding(0)]}

    lifecycle = Lifecycle(_captured_context(core_halo=True))

    class Runner:
        core_halo_parallelism = 2

        def execute_core_halo_unit(
            self, _request, _input_id, unit_id, _limits, source_bindings
        ):
            assert unit_id == 0
            first_started.set()
            assert second_waiting.wait(5)
            raise GatewayFailure("INSUFFICIENT_CONTEXT", "instance reaches image edge")

    assert not execute_reserved_model_run("run-id", lifecycle, lambda _: Runner())
    assert lifecycle.polls < 6
    assert lifecycle.failed[0][1]["code"] == "INSUFFICIENT_CONTEXT"
    assert lifecycle.published_unit_failures[0][1]["failure"] == lifecycle.failed[0][1]
    assert lifecycle.completed == []
