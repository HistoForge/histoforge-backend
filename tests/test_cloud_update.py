"""Update preparation reuses cancellation and lets final publications finish."""

import copy
import json
import time
import urllib.error
from pathlib import Path

import modal
import pytest
from histoforge import _cloud_update as update
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure

LIMITS = DEPLOYMENT_CONFIG.training_api_limits.wire_value()
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/contracts/training-runs-v1.json").read_text()
)


def record(name):
    return copy.deepcopy(
        next(item["value"] for item in FIXTURE["records"] if item["name"] == name)
    )


def summary(run):
    row = {
        key: run[key]
        for key in (
            "run_id",
            "state",
            "progress",
            "report_revision",
            "created_at",
            "updated_at",
        )
    }
    row.update(
        {
            key: run["request"][key]
            for key in ("result_display_name", "selected_model_id", "result_form")
        }
    )
    row["source_module_ref"] = run["request"]["source_module"]["module_ref"]
    if "result" in run:
        row["result"] = run["result"]
    return row


@pytest.mark.parametrize("state", ["QUEUED", "RUNNING"])
def test_cancel_active_training_without_touching_terminal_runs(monkeypatch, state):
    queued = record("queued") | {"state": state}
    cancelled = record("cancelled") | {"run_id": queued["run_id"]}
    calls = []

    def request(_url, _token, path, *, method="GET", **_kwargs):
        calls.append((method, path))
        if method == "POST":
            return cancelled
        return {"items": [summary(record("failed")), summary(queued)]}

    monkeypatch.setattr(update, "gateway_request", request)
    update._stop_training("https://gateway", "token", LIMITS, time.monotonic() + 5)
    assert calls == [
        ("GET", f"/v1/training/runs?limit={LIMITS['max_page_size']}"),
        ("POST", f"/v1/training/runs/{queued['run_id']}/cancel"),
    ]


@pytest.mark.parametrize("cancel_race", [False, True])
def test_final_save_is_reconciled_not_cancelled(monkeypatch, cancel_race):
    finalizing = record("finalizing")
    queued = record("queued") | {"run_id": finalizing["run_id"]}
    complete = record("succeeded") | {"run_id": finalizing["run_id"]}
    pages = iter([queued if cancel_race else finalizing, complete])
    calls, waits = [], []

    def request(_url, _token, path, *, method="GET", **_kwargs):
        calls.append((method, path))
        if "?limit=" in path:
            return {"items": [summary(next(pages))]}
        if method == "POST":
            raise urllib.error.HTTPError(path, 409, "already saving", {}, None)
        return finalizing

    monkeypatch.setattr(update, "gateway_request", request)
    monkeypatch.setattr(update, "_wait", lambda *args: waits.append(args))
    update._stop_training("https://gateway", "token", LIMITS, time.monotonic() + 5)
    assert len(waits) == 1
    assert sum(method == "POST" for method, _ in calls) == int(cancel_race)
    assert sum("?limit=" in path for _, path in calls) == 2


@pytest.mark.parametrize(
    "response", ["wrong_run", "still_running", "invalid", "unauthorized"]
)
def test_unconfirmed_cancellation_aborts(monkeypatch, response):
    queued = record("queued")

    def request(_url, _token, path, *, method="GET", **_kwargs):
        if method == "GET":
            return {"items": [summary(queued)]}
        if response == "wrong_run":
            return record("cancelled")
        if response == "still_running":
            return queued
        if response == "unauthorized":
            raise urllib.error.HTTPError(path, 401, "unauthorized", {}, None)
        return {}

    monkeypatch.setattr(update, "gateway_request", request)
    with pytest.raises((ValueError, RuntimeError, urllib.error.HTTPError)):
        update._stop_training("https://gateway", "token", LIMITS, time.monotonic() + 5)


def test_training_inventory_visits_all_pages(monkeypatch):
    first = copy.deepcopy(FIXTURE["page"]["value"])
    second = {"items": [summary(record("queued"))]}
    pages, paths = iter([first, second]), []

    def request(_url, _token, path, **_kwargs):
        paths.append(path)
        return next(pages)

    monkeypatch.setattr(update, "gateway_request", request)
    result = update._training_runs(
        "https://gateway", "token", LIMITS, time.monotonic() + 5
    )
    assert len(result) == 4
    assert f"cursor={first['next_cursor']}" in paths[1]


def test_repeated_cursor_is_not_an_infinite_update_wait(monkeypatch):
    monkeypatch.setattr(
        update, "gateway_request", lambda *_, **__: FIXTURE["page"]["value"]
    )
    with pytest.raises(ValueError, match="repeated a page cursor"):
        update._training_runs("https://gateway", "token", LIMITS, time.monotonic() + 5)


@pytest.mark.parametrize("problem", ["missing", "transport", "invalid"])
def test_optional_stores_only_treat_confirmed_absence_as_empty(monkeypatch, problem):
    class Store:
        def items(self):
            if problem == "missing":
                raise modal.exception.NotFoundError("not yet created")
            if problem == "transport":
                raise OSError("unavailable")
            return [("model-run:v1:66666666-6666-4666-8666-666666666666", {})]

    monkeypatch.setattr(modal.Dict, "from_name", lambda _name: Store())
    calls = []
    monkeypatch.setattr(update, "_stop_training", lambda *_: calls.append("training"))

    def run():
        update.stop_cloud_work(
            "https://gateway",
            "token",
            wait_for_module_writer=lambda: calls.append("writer"),
        )

    if problem == "missing":
        run()
        assert calls == ["training", "writer", "training"]
    else:
        with pytest.raises((OSError, GatewayFailure)):
            run()
        assert calls == []


def test_inference_cancel_and_module_writer_barrier_reuse_existing_owners(monkeypatch):
    monkeypatch.setattr(modal.Dict, "from_name", lambda _name: {})
    monkeypatch.setattr(
        update.ModelRunRecords,
        "active_ids",
        lambda _: ["active"],
    )
    events = []
    monkeypatch.setattr(
        update,
        "gateway_request",
        lambda _url, _token, path, **kwargs: events.append((path, kwargs["method"])),
    )
    monkeypatch.setattr(update, "_stop_training", lambda *_: events.append("training"))
    update.stop_cloud_work(
        "https://gateway",
        "token",
        wait_for_module_writer=lambda: events.append("writer"),
    )
    assert events == [
        ("/v1/inference/runs/active", "DELETE"),
        "training",
        "writer",
        "training",
    ]


def test_writer_failure_prevents_update_even_with_no_visible_jobs(monkeypatch):
    monkeypatch.setattr(modal.Dict, "from_name", lambda _name: {})
    monkeypatch.setattr(update, "_stop_training", lambda *_: None)

    def fail():
        raise RuntimeError("writer failed")

    with pytest.raises(RuntimeError, match="writer failed"):
        update.stop_cloud_work("https://gateway", "token", wait_for_module_writer=fail)


def test_unfinished_admin_job_is_waited_and_timeout_is_visible(monkeypatch):
    job = {
        "type": "PUBLISH",
        "requested_module_ref": "owner/module",
        "phase": "RUNNING",
        "updated_at": "2026-09-29T12:00:00.000Z",
        "conflict_refs": ["owner/module"],
    }
    jobs = {"66666666-6666-4666-8666-666666666666": job}
    monkeypatch.setattr(
        modal.Dict,
        "from_name",
        lambda name: jobs if name == DEPLOYMENT_CONFIG.admin_jobs_dict_name else {},
    )
    monkeypatch.setattr(update, "_stop_training", lambda *_: None)
    monkeypatch.setattr(update, "_WAIT_SECONDS", -1)
    with pytest.raises(TimeoutError):
        update.stop_cloud_work(
            "https://gateway", "token", wait_for_module_writer=lambda: None
        )
