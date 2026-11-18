"""Typed admin-job contract, transient store, admission, and pagination."""

from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from histoforge.canonical_json import canonical_bytes, sha256_bytes
from histoforge.cloud._errors import GatewayFailure, gateway_error_http_status
from histoforge.cloud.admin_jobs import routes as admin_job_routes
from histoforge.cloud.admin_jobs import store as admin_jobs
from histoforge.cloud.admin_jobs.contract import (
    validate_install_request,
    validate_job_accepted,
    validate_job_page,
    validate_job_status,
    validate_publish_request,
)

FIXTURE = Path(__file__).parent / "fixtures" / "contracts" / "admin-jobs-v1.json"
AUTH_GENERATION = {"deployment_id": "deployment-a", "auth_revision": 7}


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _entry(
    job_type: str = "INSTALL",
    requested: str = "owner/module",
    phase: str = "QUEUED",
    updated_at: str = "2026-08-15T01:02:03.004Z",
    **outcome,
):
    return {
        "type": job_type,
        "requested_module_ref": requested,
        "phase": phase,
        "updated_at": updated_at,
        "conflict_refs": [requested],
        **outcome,
    }


def test_shared_status_contract_matches_exact_bytes_and_digests():
    fixture = _fixture()
    accepted = validate_job_accepted(fixture["accepted"]["value"])
    encoded = canonical_bytes(accepted)
    assert encoded.decode() == fixture["accepted"]["canonical"]
    assert sha256_bytes(encoded) == fixture["accepted"]["sha256"]

    for case in fixture["statusCases"]:
        status = validate_job_status(case["value"])
        encoded = canonical_bytes(status)
        assert encoded.decode() == case["canonical"], case["name"]
        assert sha256_bytes(encoded) == case["sha256"], case["name"]

    statuses = {case["name"]: case["value"] for case in fixture["statusCases"]}
    page_case = fixture["page"]
    page = validate_job_page(
        {
            "jobs": [statuses[name] for name in page_case["statusNames"]],
            "next_cursor": page_case["nextCursor"],
        }
    )
    encoded = canonical_bytes(page)
    assert encoded.decode() == page_case["canonical"]
    assert sha256_bytes(encoded) == page_case["sha256"]


@pytest.mark.parametrize(
    ("case_name", "change"),
    [
        ("queued-install", lambda value: value.update(result={})),
        ("succeeded-install", lambda value: value.update(phase="PARTIAL")),
        (
            "succeeded-publish",
            lambda value: value["result"].update(target_repository="owner/other"),
        ),
        (
            "partial-publish",
            lambda value: value["result"].update(local_sync={"type": "not-requested"}),
        ),
        ("failed-install", lambda value: value.pop("failure")),
    ],
)
def test_status_phase_and_identity_mismatches_fail_closed(case_name, change):
    value = copy.deepcopy(
        next(
            case["value"]
            for case in _fixture()["statusCases"]
            if case["name"] == case_name
        )
    )
    change(value)
    with pytest.raises(ValueError):
        validate_job_status(value)


def test_success_results_require_the_resolved_commit_identity():
    fixture = _fixture()
    installed = copy.deepcopy(
        next(
            case["value"]
            for case in fixture["statusCases"]
            if case["name"] == "succeeded-install"
        )
    )
    installed["result"]["installed_module"].pop("resolved_commit_sha")
    with pytest.raises(ValueError, match="requires resolved_commit_sha"):
        validate_job_status(installed)

    published = copy.deepcopy(
        next(
            case["value"]
            for case in fixture["statusCases"]
            if case["name"] == "succeeded-publish-with-local-sync"
        )
    )
    published["result"]["local_sync"]["installed_module"]["resolved_commit_sha"] = (
        "e" * 40
    )
    with pytest.raises(ValueError, match="does not match published commit"):
        validate_job_status(published)


def test_admin_requests_are_exact_and_credential_free():
    assert validate_install_request({"module_ref": "owner/module"}) == {
        "module_ref": "owner/module"
    }
    assert (
        validate_install_request(
            {"module_ref": "owner/module", "hugging_face_revision": "main"}
        )["hugging_face_revision"]
        == "main"
    )
    publish = {
        "source_module": "owner/source",
        "target_repository": "owner/target",
        "visibility": "PRIVATE",
        "card_markdown": "# Card\n",
        "sync_locally": False,
    }
    assert validate_publish_request(publish) == publish
    with pytest.raises(ValueError, match="unknown=.*hf_token"):
        validate_install_request({"module_ref": "owner/module", "hf_token": "secret"})
    with pytest.raises(ValueError, match="nonblank"):
        validate_install_request(
            {"module_ref": "owner/module", "hugging_face_revision": None}
        )
    with pytest.raises(ValueError, match="PUBLIC or PRIVATE"):
        validate_publish_request({**publish, "visibility": "private"})
    with pytest.raises(ValueError, match="run-derived workspace module"):
        validate_publish_request(
            {
                **publish,
                "source_module": ("workspace/run-44444444-4444-4444-8444-444444444444"),
                "sync_locally": True,
            }
        )


def test_accept_writes_before_spawn_and_copies_identity():
    jobs = {}
    observed = {}

    def spawn(job_id, job_type, payload, deployment_id, auth_revision):
        observed.update(
            present=job_id in jobs,
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            deployment_id=deployment_id,
            auth_revision=auth_revision,
        )

    payload = {"module_ref": "owner/module"}
    accepted = admin_jobs.accept_job(
        jobs,
        **AUTH_GENERATION,
        job_type_value="INSTALL",
        requested_module_ref="owner/module",
        conflict_refs=["owner/module"],
        payload=payload,
        spawn_runner=spawn,
    )
    payload["module_ref"] = "owner/changed"

    job_id = accepted["job_id"]
    assert observed == {
        "present": True,
        "job_id": job_id,
        "job_type": "INSTALL",
        "payload": {"module_ref": "owner/module"},
        **AUTH_GENERATION,
    }
    assert jobs[job_id]["phase"] == "QUEUED"
    assert jobs[job_id]["updated_at"].endswith("Z")
    assert "." in jobs[job_id]["updated_at"]


def test_submit_uses_the_short_serialized_admission_lane(monkeypatch):
    called = {}
    job_id = "11111111-1111-4111-8111-111111111111"

    class AcceptMethod:
        def remote(self, *args):
            called["args"] = args
            return {"job_id": job_id}

    class Lane:
        accept = AcceptMethod()

    def from_name(app_name, class_name):
        called["address"] = (app_name, class_name)
        return lambda: Lane()

    monkeypatch.setattr("modal.Cls.from_name", from_name)

    payload = {"module_ref": "owner/module"}
    accepted = admin_jobs.submit_job(
        **AUTH_GENERATION,
        job_type_value="INSTALL",
        requested_module_ref="owner/module",
        conflict_refs=["owner/module"],
        payload=payload,
    )
    payload["module_ref"] = "owner/changed"

    assert accepted == {"job_id": job_id}
    assert called["address"] == ("histoforge-gateway", "AdminJobAdmissionLane")
    assert called["args"] == (
        "deployment-a",
        7,
        "INSTALL",
        "owner/module",
        ["owner/module"],
        {"module_ref": "owner/module"},
    )


def test_ambiguous_spawn_failure_keeps_visible_queued_row():
    jobs = {}

    def fail(*_args):
        raise TimeoutError("response lost")

    with pytest.raises(GatewayFailure) as caught:
        admin_jobs.accept_job(
            jobs,
            **AUTH_GENERATION,
            job_type_value="INSTALL",
            requested_module_ref="owner/module",
            conflict_refs=["owner/module"],
            payload={"module_ref": "owner/module"},
            spawn_runner=fail,
        )
    assert caught.value.error["code"] == "SERVICE_UNAVAILABLE"
    assert len(jobs) == 1
    assert next(iter(jobs.values()))["phase"] == "QUEUED"


def test_observed_conflict_and_active_limit_reject_admission():
    jobs = {}
    for index in range(4):
        job_id = str(uuid.UUID(int=index + 1))
        jobs[job_id] = _entry(requested=f"owner/module-{index}")

    with pytest.raises(GatewayFailure) as capacity:
        admin_jobs.accept_job(
            jobs,
            **AUTH_GENERATION,
            job_type_value="INSTALL",
            requested_module_ref="owner/new",
            conflict_refs=["owner/new"],
            payload={},
            spawn_runner=lambda *_args: None,
        )
    assert capacity.value.error["code"] == "RATE_LIMITED"

    jobs.pop(str(uuid.UUID(int=4)))
    with pytest.raises(GatewayFailure) as conflict:
        admin_jobs.accept_job(
            jobs,
            **AUTH_GENERATION,
            job_type_value="PUBLISH",
            requested_module_ref="owner/target",
            conflict_refs=["owner/module-0", "owner/target"],
            payload={},
            spawn_runner=lambda *_args: None,
        )
    assert conflict.value.error["code"] == "MODULE_BUSY"
    assert conflict.value.error["resource_id"] == "owner/module-0"


def test_serialized_admission_stops_at_the_active_limit():
    jobs = {
        str(uuid.UUID(int=index + 1)): _entry(requested=f"owner/existing-{index}")
        for index in range(3)
    }

    admin_jobs.accept_job(
        jobs,
        **AUTH_GENERATION,
        job_type_value="INSTALL",
        requested_module_ref="owner/fourth",
        conflict_refs=["owner/fourth"],
        payload={},
        spawn_runner=lambda *_args: None,
    )
    with pytest.raises(GatewayFailure) as capacity:
        admin_jobs.accept_job(
            jobs,
            **AUTH_GENERATION,
            job_type_value="INSTALL",
            requested_module_ref="owner/fifth",
            conflict_refs=["owner/fifth"],
            payload={},
            spawn_runner=lambda *_args: None,
        )

    assert capacity.value.error["code"] == "RATE_LIMITED"
    assert len(jobs) == 4


def test_rejected_admission_does_not_prune_terminal_history(monkeypatch):
    active_id = "11111111-1111-4111-8111-111111111111"
    terminal_id = "22222222-2222-4222-8222-222222222222"
    jobs = {
        active_id: _entry(requested="owner/busy"),
        terminal_id: _entry(
            requested="owner/old",
            phase="FAILED",
            failure={
                "schema_version": 1,
                "code": "INTERNAL_ERROR",
                "message": "failed",
            },
        ),
    }
    monkeypatch.setattr(admin_jobs, "MAX_RETAINED_ADMIN_JOBS", 2)

    with pytest.raises(GatewayFailure, match="observed administrative job"):
        admin_jobs.accept_job(
            jobs,
            **AUTH_GENERATION,
            job_type_value="INSTALL",
            requested_module_ref="owner/busy",
            conflict_refs=["owner/busy"],
            payload={},
            spawn_runner=lambda *_args: None,
        )

    assert terminal_id in jobs


def test_runner_is_sole_phase_writer_and_dismissal_is_a_noop():
    job_id = "11111111-1111-4111-8111-111111111111"
    jobs = {job_id: _entry()}
    assert admin_jobs.transition_running(jobs, job_id)
    assert jobs[job_id]["phase"] == "RUNNING"
    failure = {
        "schema_version": 1,
        "code": "SERVICE_UNAVAILABLE",
        "message": "unavailable",
    }
    assert admin_jobs.transition_failed(jobs, job_id, failure)
    assert admin_jobs.get_status(jobs, job_id)["failure"] == failure
    with pytest.raises(ValueError, match="terminal"):
        admin_jobs.transition_running(jobs, job_id)

    del jobs[job_id]
    assert admin_jobs.transition_running(jobs, job_id) is False
    assert admin_jobs.transition_failed(jobs, job_id, failure) is False


def test_pages_are_stable_and_cursor_is_exclusive():
    ids = [
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
        "33333333-3333-4333-8333-333333333333",
    ]
    jobs = {
        ids[0]: _entry(updated_at="2026-08-15T03:00:00.000Z"),
        ids[1]: _entry(updated_at="2026-08-15T03:00:00.000Z"),
        ids[2]: _entry(updated_at="2026-08-15T02:00:00.000Z"),
    }
    first = admin_jobs.job_page(jobs, limit=2, cursor=None)
    assert [job["job_id"] for job in first["jobs"]] == ids[:2]
    second = admin_jobs.job_page(jobs, limit=2, cursor=first["next_cursor"])
    assert [job["job_id"] for job in second["jobs"]] == [ids[2]]
    assert "next_cursor" not in second


def test_active_job_never_guesses_staleness():
    job_id = "11111111-1111-4111-8111-111111111111"
    jobs = {
        job_id: _entry(updated_at="2000-01-01T00:00:00.000Z"),
    }
    assert admin_jobs.active_job(jobs)["job_id"] == job_id


def test_http_routes_return_202_page_and_closed_errors(monkeypatch):
    jobs = {}
    spawned = []
    snapshot_active = False

    def submit(**kwargs):
        assert not snapshot_active
        return admin_jobs.accept_job(
            jobs,
            **kwargs,
            spawn_runner=lambda *args: spawned.append(args),
        )

    monkeypatch.setattr(admin_jobs, "submit_job", submit)

    class CurrentModules:
        def load_recipe(self, module_ref):
            assert snapshot_active
            if module_ref != "owner/source":
                raise FileNotFoundError(module_ref)
            return {}

    @contextmanager
    def snapshot():
        nonlocal snapshot_active
        snapshot_active = True
        try:
            yield CurrentModules()
        finally:
            snapshot_active = False

    web = FastAPI()

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request, failure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    admin_job_routes.register_routes(
        web,
        require_token=lambda: AUTH_GENERATION,
        jobs=jobs,
        current_modules=snapshot,
    )
    client = TestClient(web)

    accepted = client.post("/v1/install", json={"module_ref": "owner/module"})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    assert spawned[0][0] == job_id
    assert client.get(f"/v1/jobs/{job_id}").json()["phase"] == "QUEUED"
    assert client.get("/v1/jobs?limit=100").json()["jobs"][0]["job_id"] == job_id
    assert client.get("/v1/jobs?limit=" + "9" * 5_000).status_code == 422
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204

    invalid = client.post(
        "/v1/install",
        json={"module_ref": "owner/module", "hf_token": "secret"},
    )
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "VALIDATION_ERROR"
    missing = client.post(
        "/v1/publish",
        json={
            "source_module": "owner/missing",
            "target_repository": "owner/target",
            "visibility": "PRIVATE",
            "card_markdown": "# Card\n",
            "sync_locally": False,
        },
    )
    assert missing.status_code == 404
    assert missing.json()["code"] == "MODULE_NOT_FOUND"
    assert not snapshot_active
    valid = client.post(
        "/v1/publish",
        json={
            "source_module": "owner/source",
            "target_repository": "owner/target",
            "visibility": "PRIVATE",
            "card_markdown": "# Card\n",
            "sync_locally": False,
        },
    )
    assert valid.status_code == 202
    assert not snapshot_active


def test_http_route_enforces_raw_and_canonical_request_caps(monkeypatch):
    web = FastAPI()

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request, failure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    admin_job_routes.register_routes(
        web,
        require_token=lambda: None,
        jobs={},
        current_modules=lambda: nullcontext(None),
    )
    client = TestClient(web)

    monkeypatch.setattr(admin_job_routes, "MAX_JSON_REQUEST_BYTES", 8)
    raw = client.post("/v1/install", content=b'{"module_ref":"owner/module"}')
    assert raw.status_code == 413
    assert raw.json()["code"] == "PAYLOAD_TOO_LARGE"

    monkeypatch.setattr(admin_job_routes, "MAX_JSON_REQUEST_BYTES", 1_000)
    monkeypatch.setattr(admin_job_routes, "MAX_ADMIN_REQUEST_BYTES", 8)
    canonical = client.post("/v1/install", json={"module_ref": "owner/module"})
    assert canonical.status_code == 413
    assert canonical.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_http_routes_close_unexpected_store_failures():
    class BrokenJobs(dict):
        def items(self):
            raise OSError("private storage path")

    web = FastAPI()

    @web.exception_handler(GatewayFailure)
    async def failure_handler(_request, failure):
        return JSONResponse(
            status_code=gateway_error_http_status(failure.error["code"]),
            content=failure.error,
        )

    admin_job_routes.register_routes(
        web,
        require_token=lambda: None,
        jobs=BrokenJobs(),
        current_modules=lambda: nullcontext(None),
    )
    response = TestClient(web).get("/v1/jobs?limit=100")

    assert response.status_code == 503
    assert response.json() == {
        "schema_version": 1,
        "code": "SERVICE_UNAVAILABLE",
        "message": "a required gateway service is temporarily unavailable",
    }
