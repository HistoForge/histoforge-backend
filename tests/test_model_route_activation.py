"""Inference admission observes the training-result visibility cut."""

from contextlib import contextmanager

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.inference import route_activation
from histoforge.cloud.inference.route_activation import current_model_activation


def test_finalizing_training_result_cannot_activate() -> None:
    run_id = "44444444-4444-4444-8444-444444444444"
    module_ref = f"workspace/run-{run_id}"

    class Current:
        @staticmethod
        def read_record(_module_ref):
            return {
                "training_result_attestation": {"run_id": run_id},
            }

        @staticmethod
        def load_recipe(_module_ref):
            return {"models": {}}

    with pytest.raises(GatewayFailure) as raised:
        current_model_activation(
            Current(),
            module_ref,
            "model",
            lambda _run_id: False,
        )

    assert raised.value.error["code"] == "MODULE_VERSION_UNAVAILABLE"


def test_admission_returns_detached_metadata_only_after_the_snapshot_closes(
    monkeypatch,
):
    active = False
    current = object()
    expected = {"backend": {"module": "exact detached identity"}}

    def visible(_run):
        return True

    @contextmanager
    def snapshot():
        nonlocal active
        active = True
        try:
            yield current
        finally:
            active = False

    def resolve(view, ref, model, training_visible):
        assert active and view is current
        assert (ref, model) == ("owner/module", "model")
        assert training_visible is visible
        return expected

    monkeypatch.setattr(route_activation, "current_model_activation", resolve)
    assert (
        route_activation.load_current_model_activation(
            snapshot, "owner/module", "model", visible
        )
        is expected
    )
    assert not active
