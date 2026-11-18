"""Rolling slots retain publication and failure semantics without batch barriers."""

from concurrent.futures import Future
from threading import Event, Lock, get_ident

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference.runs import executor


class Lifecycle:
    def __init__(self, accepted=True):
        self.failures = []
        self.accepted = accepted

    def publish_unit_failure(self, run, publication):
        assert run == "run"
        self.failures.append(publication)
        return self.accepted


def test_free_slot_refills_while_an_earlier_unit_is_still_running():
    fifth_started = Event()
    lock = Lock()
    active = 0
    peak = 0
    accepted = []
    coordinator = get_ident()

    def operation(unit, stop):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            assert not stop.is_set()
            if unit == 0:
                assert fifth_started.wait(5), "a batch barrier left a slot idle"
            if unit == 4:
                fifth_started.set()
            return unit
        finally:
            with lock:
                active -= 1

    def accept(unit, value):
        assert get_ident() == coordinator
        assert unit == value
        accepted.append(unit)
        return True

    result = executor._run_unit_operations(
        "run", 3, list(range(9)), 4, Lifecycle(), operation, accept
    )
    assert result == dict(enumerate(range(9)))
    assert sorted(accepted) == list(range(9))
    assert active == 0
    assert peak <= 4


@pytest.mark.parametrize("accept_failure", [True, False])
def test_failure_after_refill_stops_waiters_drains_peers_and_selects_lowest_failed_id(
    accept_failure,
):
    submitted = []
    accepted = []
    lifecycle = Lifecycle(accept_failure)

    def operation(unit, stop):
        submitted.append(unit)
        if unit == 1:
            return unit  # Its freed slot must start unit 4.
        if unit == 4:
            raise GatewayFailure("RUNNER_FAILED", "later unit failed first")
        assert stop.wait(5), "failure did not release source waiters"
        if unit == 0:
            raise GatewayFailure("SERVICE_UNAVAILABLE", "lowest failed unit")
        return unit  # Already-running successes must still be accepted.

    def run():
        return executor._run_unit_operations(
            "run",
            3,
            list(range(9)),
            4,
            lifecycle,
            operation,
            lambda unit, value: accepted.append(unit) is None,
        )

    if accept_failure:
        with pytest.raises(GatewayFailure, match="lowest failed unit"):
            run()
    else:
        assert run() is None
    assert sorted(submitted) == [0, 1, 2, 3, 4]
    assert sorted(accepted) == [1, 2, 3]
    assert len(lifecycle.failures) == 1
    assert lifecycle.failures[0]["unit_id"] == 0
    assert lifecycle.failures[0]["failure"]["code"] == "SERVICE_UNAVAILABLE"


@pytest.mark.parametrize("rejection", ["false", "exception", "source-ended"])
def test_publication_rejection_or_source_end_stops_refill_and_drains_submitted(
    rejection,
):
    submitted = []
    accepted = []
    lifecycle = Lifecycle()

    def operation(unit, stop):
        submitted.append(unit)
        if unit != 0:
            assert stop.wait(5)
        return None if unit == 0 and rejection == "source-ended" else unit

    def accept(unit, value):
        accepted.append((unit, value))
        if unit == 0:
            if rejection == "exception":
                raise OSError("publication failed")
            return False
        return True

    def run():
        return executor._run_unit_operations(
            "run", 3, list(range(9)), 4, lifecycle, operation, accept
        )

    if rejection == "exception":
        with pytest.raises(OSError, match="publication failed"):
            run()
    else:
        assert run() is None
    assert sorted(submitted) == [0, 1, 2, 3]
    assert sorted(unit for unit, _ in accepted) == [0, 1, 2, 3]
    assert lifecycle.failures == []


@pytest.mark.parametrize("publication_error", [False, True])
def test_all_ready_completions_are_examined_before_refill(
    monkeypatch, publication_error
):
    submitted = []
    accepted = []
    lifecycle = Lifecycle()

    class ReadyPool:
        def __init__(self, max_workers):
            assert max_workers == 4

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def submit(self, operation, unit, stop):
            submitted.append(unit)
            result = Future()
            if unit == 3:
                result.set_exception(RuntimeError("worker failed"))
            else:
                result.set_result(unit)
            return result

    def accept(unit, value):
        accepted.append(unit)
        if unit == 0 and publication_error:
            raise OSError("publication wins over worker failure")
        return True

    monkeypatch.setattr(executor, "ThreadPoolExecutor", ReadyPool)
    expected = OSError if publication_error else GatewayFailure
    with pytest.raises(expected) as failure:
        executor._run_unit_operations(
            "run", 3, list(range(9)), 4, lifecycle, lambda *_: None, accept
        )
    assert submitted == [0, 1, 2, 3]
    assert accepted == [0, 1, 2]
    if publication_error:
        assert lifecycle.failures == []
    else:
        assert lifecycle.failures[0]["unit_id"] == 3
        assert isinstance(failure.value.__cause__, RuntimeError)
