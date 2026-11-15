"""Physical workers cannot read training data before winning the durable claim."""

from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training import implementations
from histoforge.cloud.training.runs.worker import (
    TrainingWorkerControl,
    run_training_worker,
)
from histoforge.cloud.training.runtime_identity import current_trainer_build
from tests.test_runtime_identity import _install_inputs
from tests.test_training_capabilities import _entry, _registry
from tests.test_training_dataset import training_api_limits
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request
from tests.test_training_run_state import NOW, RUN_ID, accept_run, make_state


class _Control:
    def __init__(self, state) -> None:
        self.state = state
        self.claims = []

    def get_run_control(self, run_id):
        return self.state.get_control(run_id)

    def claim_run_worker(self, run_id, build):
        self.claims.append((run_id, build))
        return self.state.claim_worker(run_id, build)


def _run(control, train, **kwargs) -> None:
    run_training_worker(
        RUN_ID,
        control=control,
        capabilities=_registry(_entry()),
        api_limits=training_api_limits(),
        train_and_publish=train,
        build_entry=kwargs.pop("build_entry", lambda _entry: trainer_build()),
        clock=kwargs.pop("clock", lambda: NOW),
        **kwargs,
    )


def test_worker_waits_for_attachment_and_claim_before_loading_data(tmp_path: Path):
    state, records = make_state(tmp_path)
    accept_run(state)
    control = _Control(state)
    events = []

    def sleep(seconds):
        assert 0 < seconds <= 0.25
        assert not control.claims
        assert not records.get_run(RUN_ID)["worker"]["worker_claimed"]
        events.append("attach")
        state.attach_worker(RUN_ID, "fc-training")

    def train(run, entry):
        assert records.get_run(RUN_ID)["worker"]["worker_claimed"]
        assert run["run_id"] == RUN_ID
        assert entry == _entry()
        events.append("load dataset and model")

    _run(control, train, sleep=sleep)

    assert events == ["attach", "load dataset and model"]
    assert control.claims == [(RUN_ID, trainer_build())]


def test_worker_with_changed_build_never_loads_data(tmp_path: Path):
    state, records = make_state(tmp_path)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-training")
    control = _Control(state)

    with pytest.raises(GatewayFailure, match="differs from the accepted"):
        _run(
            control,
            lambda *_: pytest.fail("training must not start"),
            build_entry=lambda _: {**trainer_build(), "artifact_digest": "0" * 64},
        )

    assert not control.claims
    assert not records.get_run(RUN_ID)["worker"]["worker_claimed"]


def test_changed_binding_rejects_before_claim_or_dataset_model_loading(
    tmp_path, monkeypatch
):
    _install_inputs(monkeypatch)
    entry = _entry()
    accepted_build = current_trainer_build(entry)
    replacement = replace(entry, trainer_implementation_id="test-rebound-trainer-v1")
    monkeypatch.setitem(
        implementations._IMPLEMENTATIONS,
        replacement.trainer_implementation_id,
        implementations.resolve_training_implementation(
            entry.trainer_implementation_id
        ),
    )
    state, records = make_state(tmp_path)
    request = run_request()
    state.reserve(request)
    state.accept(request, accepted_build, 3, 3_600)
    state.attach_worker(RUN_ID, "fc-training")
    control = _Control(state)

    with pytest.raises(GatewayFailure, match="differs from the accepted"):
        run_training_worker(
            RUN_ID,
            control=control,
            capabilities=_registry(replacement),
            api_limits=training_api_limits(),
            train_and_publish=lambda *_: pytest.fail(
                "old run cannot load data or model"
            ),
            clock=lambda: NOW,
        )

    assert not control.claims
    assert not records.get_run(RUN_ID)["worker"]["worker_claimed"]


@pytest.mark.parametrize("failure_type", [RuntimeError, MemoryError])
def test_physical_retry_cannot_restart_training(tmp_path: Path, failure_type):
    state, _ = make_state(tmp_path)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-training")
    control = _Control(state)

    def interrupted(*_):
        raise failure_type("container interrupted")

    with pytest.raises(failure_type, match="container interrupted"):
        _run(control, interrupted)

    with pytest.raises(GatewayFailure, match="automatic restart"):
        _run(control, lambda *_: pytest.fail("training must not restart"))
    assert control.claims == [(RUN_ID, trainer_build())]


def test_atomic_claim_rejects_a_competing_physical_worker(tmp_path: Path):
    state, _ = make_state(tmp_path)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-training")

    def competing_worker(_entry):
        state.claim_worker(RUN_ID, trainer_build())
        return trainer_build()

    with pytest.raises(GatewayFailure, match="already claimed"):
        _run(
            _Control(state),
            lambda *_: pytest.fail("second physical worker must not train"),
            build_entry=competing_worker,
        )


@pytest.mark.parametrize("cancel_during_claim", [False, True])
def test_cancellation_prevents_model_loading(tmp_path: Path, cancel_during_claim):
    state, _ = make_state(tmp_path)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-training")

    def cancel(_entry):
        state.cancel(RUN_ID, "d" * 64, 2)
        return trainer_build()

    if not cancel_during_claim:
        cancel(None)
    with pytest.raises(GatewayFailure):
        _run(
            _Control(state),
            lambda *_: pytest.fail("cancelled run must not load training data"),
            build_entry=cancel,
        )


def test_unattached_call_stops_at_its_deadline(tmp_path: Path):
    now = [NOW]
    state, _ = make_state(
        tmp_path,
        now=now,
        service_overrides={"worker_call_attachment_timeout_seconds": 1},
    )
    accept_run(state)
    control = _Control(state)
    waits = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += timedelta(seconds=seconds)

    with pytest.raises(GatewayFailure, match="not attached before its deadline"):
        _run(
            control,
            lambda *_: pytest.fail("unattached worker must not train"),
            clock=lambda: now[0],
            sleep=sleep,
        )

    assert sum(waits) == 1
    assert not control.claims


def test_worker_cannot_claim_after_execution_deadline(tmp_path: Path):
    now = [NOW]
    state, records = make_state(tmp_path, now=now)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-training")
    deadline = records.get_run(RUN_ID)["worker"]["execution_deadline_at"]
    now[0] = datetime.fromisoformat(deadline.replace("Z", "+00:00"))

    with pytest.raises(GatewayFailure, match="execution deadline"):
        _run(
            _Control(state),
            lambda *_: pytest.fail("expired run must not train"),
            clock=lambda: now[0],
        )


def test_internal_worker_methods_claim_without_user_credentials(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from histoforge.cloud.training import state_lane as training_state_lane

    state, records = make_state(tmp_path)
    accept_run(state)
    state.attach_worker(RUN_ID, "fc-training")
    lane_type = training_state_lane.TrainingStateLane._get_user_cls()
    lane = lane_type()
    monkeypatch.setattr(lane, "_run_control", lambda: state)
    monkeypatch.setattr(
        training_state_lane,
        "_authorize",
        lambda *_: pytest.fail("workers must not need a user's auth generation"),
    )
    remote = SimpleNamespace(
        get_worker_run_control=SimpleNamespace(
            remote=lambda *args: lane_type.__dict__[
                "get_worker_run_control"
            ]._get_raw_f()(lane, *args)
        ),
        claim_run_worker=SimpleNamespace(
            remote=lambda *args: lane_type.__dict__["claim_run_worker"]._get_raw_f()(
                lane, *args
            )
        ),
    )

    _run(
        TrainingWorkerControl(remote),
        lambda *_: records.get_run(RUN_ID)["worker"]["worker_claimed"],
    )

    assert records.get_run(RUN_ID)["worker"]["worker_claimed"]
