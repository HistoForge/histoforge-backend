"""Scheduled cleanup delegates to the existing serialized storage owner."""

import runpy
from types import SimpleNamespace
from unittest.mock import Mock

import modal
import pytest
from histoforge.cloud import _app
from histoforge.cloud.artifacts import lifecycle


def test_cleanup_runs_before_sources_and_preserves_failures(monkeypatch):
    events = []
    run_files, source_files = object(), object()
    runs, sources = Mock(), Mock()
    runs.cleanup_expired.side_effect = lambda files: events.append(("runs", files))
    sources.cleanup_expired.side_effect = lambda files: events.append(
        ("sources", files)
    )
    monkeypatch.setattr(lifecycle, "_model_runs", lambda: runs)
    monkeypatch.setattr(lifecycle, "_artifact_files", lambda: run_files)
    monkeypatch.setattr(lifecycle, "_sources", lambda: (sources, None))
    monkeypatch.setattr(lifecycle, "_source_files", lambda: source_files)

    with pytest.warns(UserWarning, match="executing locally"):
        lifecycle.ArtifactLifecycleLane().cleanup_expired.local()
    assert events == [("runs", run_files), ("sources", source_files)]

    events.clear()
    runs.cleanup_expired.side_effect = OSError("run cleanup unavailable")
    with pytest.warns(UserWarning, match="executing locally"):
        with pytest.raises(OSError, match="run cleanup unavailable"):
            lifecycle.ArtifactLifecycleLane().cleanup_expired.local()
    assert events == []


def test_scheduled_trigger_waits_for_owner_and_propagates_failure(monkeypatch):
    cleanup = Mock()
    owner = SimpleNamespace(cleanup_expired=SimpleNamespace(remote=cleanup))
    monkeypatch.setattr(lifecycle, "ArtifactLifecycleLane", lambda: owner)

    lifecycle.cleanup_expired_inference_files.local()
    cleanup.assert_called_once_with()
    cleanup.side_effect = OSError("storage unavailable")
    with pytest.raises(OSError, match="storage unavailable"):
        lifecycle.cleanup_expired_inference_files.local()


def test_registration_is_hourly_cpu_only_and_deployed(monkeypatch):
    # Capture the public decorator arguments instead of Modal's private metadata.
    settings = {}

    class RecordingApp:
        def cls(self, **_settings):
            return lambda cls: cls

        def function(self, **configuration):
            def register(function):
                settings[function.__name__] = configuration
                return function

            return register

    period = Mock(return_value=modal.Period(hours=1))
    with monkeypatch.context() as patch:
        patch.setattr(_app, "app", RecordingApp())
        patch.setattr(modal, "Period", period)
        runpy.run_path(
            lifecycle.__file__, run_name="inference_cleanup_registration_test"
        )

    period.assert_called_once_with(hours=1)
    assert settings["cleanup_expired_inference_files"] == {
        "image": lifecycle.image,
        "schedule": period.return_value,
        "timeout": 300,
        "max_containers": 1,
        "retries": 0,
    }

    from histoforge.cloud import deploy

    assert (
        deploy.cleanup_expired_inference_files
        is lifecycle.cleanup_expired_inference_files
    )
    assert "cleanup_expired_inference_files" in _app.app.registered_functions
