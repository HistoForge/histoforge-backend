"""Source-lifecycle and durable-run reservation boundary tests."""

from datetime import datetime, timedelta, timezone

import modal
import pytest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts import lifecycle
from histoforge.cloud.inference.runs.executor import _RemoteLifecycle
from histoforge.cloud.inference.runs.routes import _ModelRunLifecycleClient
from histoforge.cloud.sources.client import SourceArtifactLifecycleClient


def test_source_run_and_worker_clients_share_one_serialized_lifecycle(monkeypatch):
    addresses = []
    lane = object()

    def from_name(app_name, class_name):
        addresses.append((app_name, class_name))
        return lambda: lane

    monkeypatch.setattr(modal.Cls, "from_name", from_name)
    state = {"deployment_id": "deployment", "auth_revision": 7}
    clients = [
        SourceArtifactLifecycleClient(state),
        _ModelRunLifecycleClient(state),
        _RemoteLifecycle(),
    ]
    assert all(client._lane is lane for client in clients)
    assert (
        addresses == [(DEPLOYMENT_CONFIG.gateway_app_name, "ArtifactLifecycleLane")] * 3
    )


class _SourceStore:
    def __init__(self, expires_at):
        self.expires_at = expires_at
        self.resolved = []

    def resolve_sealed_reference(self, reference):
        self.resolved.append(reference)
        return {"artifact": reference, "absolute_expires_at": self.expires_at}


def test_source_admission_requires_every_exact_artifact_through_execution():
    expires = datetime(2026, 8, 18, tzinfo=timezone.utc)
    reference = {"artifact_id": "11111111-1111-4111-8111-111111111111"}
    request = {
        "inputs": [
            {"raster_input": {"source_view": {"artifact": reference}}},
            {"raster_input": {"source_view": {"artifact": reference}}},
        ]
    }
    source = _SourceStore("2026-08-18T00:00:00.000Z")

    lifecycle._require_sources_live_through(source, request, expires)
    assert source.resolved == [reference, reference]

    with pytest.raises(GatewayFailure) as failure:
        lifecycle._require_sources_live_through(
            source, request, expires + timedelta(milliseconds=1)
        )
    assert failure.value.error["code"] == "ARTIFACT_PRECONDITION_FAILED"


def test_source_lane_reserves_run_with_source_fence_and_single_dispatch(monkeypatch):
    source = _SourceStore("2026-08-18T00:00:00.000Z")
    files = object()
    auth = object()
    request = {
        "inputs": [
            {
                "raster_input": {
                    "source_view": {
                        "artifact": {
                            "artifact_id": "11111111-1111-4111-8111-111111111111"
                        }
                    }
                }
            }
        ]
    }
    authorizations = []

    class _Runs:
        def reserve(
            self,
            submitted,
            limits,
            retention,
            actual_files,
            *,
            require_sources_live_through,
            require_bindings_live_through,
        ):
            assert submitted is request
            assert limits == {"limit": 1}
            assert retention == 60
            assert actual_files is files
            required_until = datetime(2026, 8, 17, 23, 59, tzinfo=timezone.utc)
            require_sources_live_through(submitted, required_until)
            assert callable(require_bindings_live_through)
            return {"model_run_id": "run"}, True

    monkeypatch.setattr(lifecycle, "_sources", lambda: (source, auth))
    monkeypatch.setattr(lifecycle, "_model_runs", _Runs)
    monkeypatch.setattr(lifecycle, "_artifact_files", lambda: files)
    monkeypatch.setattr(
        lifecycle,
        "_authorize",
        lambda *values: authorizations.append(values),
    )

    with pytest.warns(UserWarning, match="executing locally"):
        result = lifecycle.ArtifactLifecycleLane().reserve_model_run.local(
            "deployment", 7, request, {"limit": 1}, 60
        )

    assert result == {
        "accepted": {"model_run_id": "run"},
        "should_dispatch": True,
    }
    assert len(authorizations) == 2
    assert source.resolved == [
        request["inputs"][0]["raster_input"]["source_view"]["artifact"]
    ]
