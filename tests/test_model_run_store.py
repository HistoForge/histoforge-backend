"""Durable run persistence, publication ordering, and terminal fencing tests."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from histoforge.canonical_json import parse_json
from histoforge.cloud._errors import GatewayFailure, build_gateway_error
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.inference.contract import (
    build_backend_input_identity,
)
from histoforge.cloud.inference.feature_stream import encode_feature_stream
from histoforge.cloud.inference.features import build_feature_collection
from histoforge.cloud.inference.runs.publications import (
    build_expected_model_execution_plans,
    model_run_execution_identity_digest,
)
from histoforge.cloud.inference.runs.records import ModelRunRecords
from histoforge.cloud.inference.runs.request import validate_model_run_admission
from histoforge.cloud.inference.runs.store import ModelRunStore
from histoforge.cloud.protocol_versions import MODEL_RUN_SCHEMA_VERSION

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **delta) -> None:
        self.value += timedelta(**delta)


def test_reservation_publishes_request_before_record_and_is_idempotent(tmp_path):
    request = _request(0)
    clock = _Clock()
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=1, clock=clock)

    required_until = []
    accepted, should_dispatch = store.reserve(
        request,
        _limits(),
        600,
        files,
        require_sources_live_through=lambda _request, deadline: required_until.append(
            deadline
        ),
    )
    replay, replay_should_dispatch = store.reserve(
        copy.deepcopy(request),
        _limits(),
        600,
        files,
        require_sources_live_through=lambda _request, _deadline: pytest.fail(
            "idempotent replay must not re-admit source state"
        ),
    )

    assert replay == accepted
    assert should_dispatch is True
    assert replay_should_dispatch is False
    assert required_until == [clock.value + timedelta(seconds=21600)]
    status = store.status(request["model_run_id"], files)
    assert status["phase"] == "QUEUED"
    assert (
        files.load_artifact_ref(status["request_artifact"]["artifact_id"])
        == status["request_artifact"]
    )
    assert _artifact_json(files, status["request_artifact"]) == request

    changed = copy.deepcopy(request)
    changed["action_id"] = "51515151-5151-4151-8151-515151515151"
    with pytest.raises(GatewayFailure) as conflict:
        _reserve(store, changed, _limits(), 600, files)
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"

    second = copy.deepcopy(request)
    second["model_run_id"] = "52525252-5252-4252-8252-525252525252"
    with pytest.raises(GatewayFailure) as capacity:
        _reserve(store, second, _limits(), 600, files)
    assert capacity.value.error["code"] == "RATE_LIMITED"

    context = store.begin_execution(request["model_run_id"], files)
    assert context == {
        "checkpoints": [],
        "completed_input_ids": [],
        "completed_units": [],
        "execution_deadline": "2026-08-17T18:00:00.000Z",
        "execution_plans": [
            {
                "input_id": request["inputs"][0]["input_id"],
                "plan": build_expected_model_execution_plans(request, _limits())[
                    request["inputs"][0]["input_id"]
                ],
            }
        ],
        "failed_units": [],
        "limits": _limits(),
        "request": request,
        "source_bindings": [],
    }
    assert store.begin_execution(request["model_run_id"], files) == context


@pytest.mark.parametrize("schema_version", [1, 3])
def test_old_and_future_private_records_fail_closed_on_every_read(
    tmp_path, schema_version
):
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=1, clock=_Clock())
    request = _request(0)
    _reserve(store, request, _limits(), 600, files)
    record = next(iter(values.values()))
    assert record["schema_version"] == 2
    record["schema_version"] = schema_version
    before = copy.deepcopy(values)
    artifacts = files.artifact_ids()
    records = ModelRunRecords(values)
    for read in (
        lambda: records.get(request["model_run_id"]),
        records.all,
        records.active_ids,
    ):
        with pytest.raises(GatewayFailure) as failure:
            read()
        assert failure.value.error["code"] == "INTERNAL_ERROR"
    assert values == before
    assert files.artifact_ids() == artifacts


def test_fresh_admission_collects_expired_run_before_publishing_new_request(
    tmp_path, monkeypatch
):
    clock = _Clock()
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=1, clock=clock)
    old = _request(0)
    _reserve(store, old, _limits(), 2, files)
    store.cancel(old["model_run_id"], files)
    old_ids = files.artifact_ids()
    clock.advance(seconds=2)

    # Existing-ID outcomes are not a cleanup trigger, including expired IDs.
    with pytest.raises(GatewayFailure) as expired:
        _reserve(store, old, _limits(), 2, files)
    assert expired.value.error["code"] == "MODEL_RUN_EXPIRED"
    assert files.artifact_ids() == old_ids

    fresh = copy.deepcopy(old)
    fresh["model_run_id"] = "71717171-7171-4171-8171-717171717171"
    original_delete = files.delete_artifact

    def failed_delete(_artifact_id):
        raise OSError("cleanup failed")

    monkeypatch.setattr(files, "delete_artifact", failed_delete)
    with pytest.raises(OSError, match="cleanup failed"):
        _reserve(store, fresh, _limits(), 600, files)
    assert len(values) == 1
    assert files.artifact_ids() == old_ids
    assert next(iter(values.values()))["request"]["model_run_id"] == old["model_run_id"]

    def slow_delete(artifact_id):
        original_delete(artifact_id)
        clock.advance(seconds=5)

    monkeypatch.setattr(files, "delete_artifact", slow_delete)
    required_until = []
    accepted, should_dispatch = store.reserve(
        fresh,
        _limits(),
        600,
        files,
        require_sources_live_through=lambda _request, deadline: required_until.append(
            deadline
        ),
    )
    assert accepted["model_run_id"] == fresh["model_run_id"]
    assert should_dispatch
    assert len(values) == 1
    assert not old_ids & files.artifact_ids()
    assert required_until == [clock.value + timedelta(seconds=21600)]
    record = next(iter(values.values()))
    assert record["accepted_at"] == "2026-08-17T12:00:07.000Z"

    def unexpected_cleanup(_files):
        pytest.fail("idempotent replay must not collect unrelated records")

    monkeypatch.setattr(store, "cleanup_expired", unexpected_cleanup)
    assert _reserve(store, fresh, _limits(), 600, files) == (accepted, False)


def test_core_unit_publication_is_idempotent_and_fences_conflicts(tmp_path):
    request = _request(1)
    clock = _Clock()
    files = ArtifactFileStore(tmp_path)
    values = {}
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    _reserve(store, request, _limits(), 600, files)

    publication, _input_result = _finite_publications(request, files)
    assert store.publish_unit(request["model_run_id"], publication, files)
    assert store.publish_unit(request["model_run_id"], publication, files)
    conflicting_unit = copy.deepcopy(publication)
    conflicting_unit["reusable_state"] = files.publish_artifact_bytes(
        b"different complete unit",
        media_type="application/octet-stream",
    )
    with pytest.raises(GatewayFailure) as unit_conflict:
        store.publish_unit(request["model_run_id"], conflicting_unit, files)
    assert unit_conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"

    running = store.status(request["model_run_id"], files)
    assert running["phase"] == "RUNNING"
    assert running["completed_unit_count"] == 1
    failure = build_gateway_error("INSUFFICIENT_CONTEXT", "owned object reaches halo")
    assert store.fail(request["model_run_id"], failure, files)
    # Re-open the persistence owner: status must not depend on worker memory.
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    status = store.status(request["model_run_id"], files)
    manifest = _artifact_json(files, status["terminal_manifest"])
    assert status["phase"] == "PARTIAL"
    assert manifest["complete"] is False
    assert manifest["completed_units"] == [publication]
    assert status["failure"] == manifest["failure"] == failure
    assert next(iter(values.values()))["failure"] == failure
    assert not store.publish_unit(request["model_run_id"], publication, files)
    assert not store.fail(
        request["model_run_id"], build_gateway_error("RUNNER_FAILED", "late"), files
    )
    store.cancel(request["model_run_id"], files)
    assert store.status(request["model_run_id"], files) == status


@pytest.mark.parametrize("replacement", [None, "different"])
def test_partial_record_failure_must_match_the_immutable_manifest(
    tmp_path, replacement
):
    request = _request(1)
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    publication, _input = _finite_publications(request, files)
    store.publish_unit(request["model_run_id"], publication, files)
    store.fail(
        request["model_run_id"],
        build_gateway_error("INSUFFICIENT_CONTEXT", "owned object reaches halo"),
        files,
    )
    record = next(iter(values.values()))
    record["failure"] = (
        None
        if replacement is None
        else build_gateway_error("RUNNER_FAILED", "different failure")
    )
    with pytest.raises(GatewayFailure) as corrupt:
        store.status(request["model_run_id"], files)
    assert corrupt.value.error["code"] == "INTERNAL_ERROR"


def test_replacement_run_reuses_exact_unit_and_shared_cleanup_is_safe(
    tmp_path, monkeypatch
):
    request = _request(1)
    clock = _Clock()
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=clock)
    _reserve(store, request, _limits(), 2, files)
    unit, _input = _finite_publications(request, files)
    assert store.publish_unit(request["model_run_id"], unit, files)
    assert store.fail(
        request["model_run_id"],
        build_gateway_error("RUNNER_FAILED", "later unit failed"),
        files,
    )
    prior_status = store.status(request["model_run_id"], files)
    prior_manifest = _artifact_json(files, prior_status["terminal_manifest"])

    replacement = copy.deepcopy(request)
    replacement["model_run_id"] = "59595959-5959-4959-8959-595959595959"
    replacement["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": prior_manifest["manifest_digest"],
    }
    # Expiry occurs after reuse validation but before admission publication.
    # The one lifecycle lane keeps collection outside this complete operation.
    original_publish = files.publish_artifact_bytes

    def publish_after_expiry(*args, **kwargs):
        clock.advance(seconds=2)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(files, "publish_artifact_bytes", publish_after_expiry)
    accepted, should_dispatch = _reserve(store, replacement, _limits(), 600, files)
    monkeypatch.setattr(files, "publish_artifact_bytes", original_publish)
    assert accepted["model_run_id"] == replacement["model_run_id"]
    assert should_dispatch
    context = store.begin_execution(replacement["model_run_id"], files)
    assert context["completed_units"] == [unit]

    shared_id = unit["reusable_state"]["artifact_id"]
    next_request = _request(0)
    next_request["model_run_id"] = "72727272-7272-4272-8272-727272727272"
    _reserve(store, next_request, _limits(), 600, files)
    with pytest.raises(GatewayFailure) as absent:
        store.status(request["model_run_id"], files)
    assert absent.value.error["code"] == "MODEL_RUN_NOT_FOUND"
    assert files.load_artifact_ref(shared_id) == unit["reusable_state"]
    assert store.status(replacement["model_run_id"], files)["phase"] == "RUNNING"


def test_replacement_run_reuses_complete_input_but_not_changed_generation(tmp_path):
    request = _request(0)
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    _unit, complete_input = _finite_publications(request, files)
    assert store.publish_input(request["model_run_id"], complete_input, 0, files)
    assert store.complete(request["model_run_id"], files)
    manifest = _artifact_json(
        files, store.status(request["model_run_id"], files)["terminal_manifest"]
    )

    replacement = copy.deepcopy(request)
    replacement["model_run_id"] = "60606060-6060-4060-8060-606060606060"
    replacement["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": manifest["manifest_digest"],
    }
    _reserve(store, replacement, _limits(), 600, files)
    context = store.begin_execution(replacement["model_run_id"], files)
    assert context["completed_input_ids"] == [request["inputs"][0]["input_id"]]

    changed = copy.deepcopy(replacement)
    changed["model_run_id"] = "61616161-6161-4161-8161-616161616161"
    changed["inputs"][0]["output_generation_id"] = (
        "62626262-6262-4262-8262-626262626262"
    )
    _reserve(store, changed, _limits(), 600, files)
    changed_context = store.begin_execution(changed["model_run_id"], files)
    assert changed_context["completed_input_ids"] == []


def test_replacement_run_requires_exact_retained_manifest_and_budget(tmp_path):
    request = _request(1)
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    unit, _input = _finite_publications(request, files)
    store.publish_unit(request["model_run_id"], unit, files)
    store.fail(
        request["model_run_id"],
        build_gateway_error("RUNNER_FAILED", "later unit failed"),
        files,
    )
    manifest = _artifact_json(
        files, store.status(request["model_run_id"], files)["terminal_manifest"]
    )

    wrong = copy.deepcopy(request)
    wrong["model_run_id"] = "63636363-6363-4363-8363-636363636363"
    wrong["resume_from"] = {
        "previous_run_id": request["model_run_id"],
        "previous_manifest_digest": "0" * 64,
    }
    with pytest.raises(GatewayFailure) as conflict:
        _reserve(store, wrong, _limits(), 600, files)
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"

    too_small = copy.deepcopy(wrong)
    too_small["model_run_id"] = "64646464-6464-4464-8464-646464646464"
    too_small["resume_from"]["previous_manifest_digest"] = manifest["manifest_digest"]
    limits = _limits()
    limits["max_recovery_state_bytes_per_unit"] = (
        unit["reusable_state"]["size_bytes"] - 1
    )
    with pytest.raises(GatewayFailure) as exceeded:
        _reserve(store, too_small, limits, 600, files)
    assert exceeded.value.error["code"] == "RESULT_LIMIT_EXCEEDED"


def test_complete_finite_input_is_one_atomic_reusable_boundary(tmp_path):
    request = _request(0)
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    _unit, input_result = _finite_publications(request, files)

    assert store.publish_input(request["model_run_id"], input_result, 0, files)
    assert store.complete(request["model_run_id"], files)
    status = store.status(request["model_run_id"], files)
    assert status["phase"] == "COMPLETE"
    manifest = _artifact_json(files, status["terminal_manifest"])
    assert manifest["completed_units"] == []
    assert manifest["completed_inputs"] == [input_result]

    retained_stream = input_result["feature_stream_artifact"]
    assert not store.discard_unreferenced_artifact(retained_stream, files)
    late = files.publish_artifact_bytes(
        b"late result after terminal publication",
        media_type="application/geo+json-seq",
        content_encoding="gzip",
    )
    assert store.discard_unreferenced_artifact(late, files)
    assert late["artifact_id"] not in files.artifact_ids()
    assert retained_stream["artifact_id"] in files.artifact_ids()


@pytest.mark.parametrize("plan_kind", [0, 1, 2], ids=["finite", "core-halo", "global"])
@pytest.mark.parametrize("completed_count", [0, 1, 2])
def test_failure_uses_the_results_already_published(
    tmp_path, plan_kind, completed_count
):
    request = _two_input_request(plan_kind)
    values = {}
    files = ArtifactFileStore(tmp_path)
    clock = _Clock()
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    _reserve(store, request, _limits(), 600, files)
    completed = []
    for index in range(completed_count):
        unit, result = _finite_publications(request, files, input_index=index)
        if plan_kind == 1:
            assert store.publish_unit(request["model_run_id"], unit, files)
        assert store.publish_input(request["model_run_id"], result, 0, files)
        completed.append(result)

    failure = build_gateway_error("RUNNER_FAILED", "image preparation failed")
    assert store.fail(request["model_run_id"], failure, files)
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    status = store.status(request["model_run_id"], files)
    assert status["phase"] == ("FAILED", "PARTIAL", "COMPLETE")[completed_count]
    if completed_count:
        manifest = _artifact_json(files, status["terminal_manifest"])
        assert manifest["completed_inputs"] == completed
        assert manifest["complete"] is (completed_count == 2)
    if completed_count == 2:
        assert "failure" not in status
        assert "failure" not in manifest
    else:
        assert status["failure"] == failure

    # Neither a duplicate failure nor a later completion/cancellation changes
    # the terminal decision or starts another retention interval.
    clock.advance(seconds=1)
    assert not store.fail(request["model_run_id"], failure, files)
    assert not store.complete(request["model_run_id"], files)
    store.cancel(request["model_run_id"], files)
    assert store.status(request["model_run_id"], files) == status


def test_completed_core_input_releases_unit_artifacts_at_run_expiry(tmp_path):
    request = _request(1)
    clock = _Clock()
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    _reserve(store, request, _limits(), 2, files)
    unit, _unused_input = _finite_publications(request, files)
    _unused_unit, input_result = _finite_publications(request, files)

    assert store.publish_unit(request["model_run_id"], unit, files)
    assert store.publish_input(request["model_run_id"], input_result, 0, files)
    assert store.complete(request["model_run_id"], files)
    record = next(iter(values.values()))
    unit_artifact_id = unit["reusable_state"]["artifact_id"]
    assert record["released_artifact_ids"] == [unit_artifact_id]
    assert unit_artifact_id in files.artifact_ids()

    clock.advance(seconds=2)
    store.cleanup_expired(files)
    assert files.artifact_ids() == set()


def test_expiry_cleanup_retries_before_removing_artifact_authority(
    tmp_path, monkeypatch
):
    request = _request(1)
    clock = _Clock()
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    _reserve(store, request, _limits(), 2, files)
    unit, _unused_input = _finite_publications(request, files)
    _unused_unit, input_result = _finite_publications(request, files)
    store.publish_unit(request["model_run_id"], unit, files)
    store.publish_input(request["model_run_id"], input_result, 0, files)
    store.complete(request["model_run_id"], files)

    owned_ids = sorted(files.artifact_ids())
    failure_id = owned_ids[1]
    original_delete = files.delete_artifact
    failed = False

    def fail_once(artifact_id):
        nonlocal failed
        if artifact_id == failure_id and not failed:
            failed = True
            raise OSError("injected artifact deletion failure")
        original_delete(artifact_id)

    clock.advance(seconds=2)
    monkeypatch.setattr(files, "delete_artifact", fail_once)
    with pytest.raises(OSError, match="injected artifact deletion failure"):
        store.cleanup_expired(files)

    assert values
    assert owned_ids[0] not in files.artifact_ids()
    monkeypatch.setattr(files, "delete_artifact", original_delete)
    store.cleanup_expired(files)
    assert values == {}
    assert files.artifact_ids() == set()


def test_finite_plan_rejects_redundant_completed_unit_state(tmp_path):
    request = _request(0)
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, _limits(), 600, files)
    unit, _input = _finite_publications(request, files)

    with pytest.raises(ValueError, match="core-halo"):
        store.publish_unit(request["model_run_id"], unit, files)


def test_global_checkpoint_retains_only_highest_sequence_and_replays_exactly(
    tmp_path,
):
    request = _request(2)
    limits = _limits()
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, limits, 600, files)
    input_id = request["inputs"][0]["input_id"]
    identity = model_run_execution_identity_digest(request, input_id, 0, limits)

    first = {
        "input_id": input_id,
        "unit_id": 0,
        "sequence": 64,
        "state": files.publish_artifact_bytes(
            b"checkpoint-64", media_type="application/octet-stream"
        ),
        "execution_identity_digest": identity,
    }
    assert store.publish_checkpoint(request["model_run_id"], first, files)
    assert store.publish_checkpoint(request["model_run_id"], first, files)

    conflicting = copy.deepcopy(first)
    conflicting["state"] = files.publish_artifact_bytes(
        b"different-checkpoint-64", media_type="application/octet-stream"
    )
    with pytest.raises(GatewayFailure) as conflict:
        store.publish_checkpoint(request["model_run_id"], conflicting, files)
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"

    lower = copy.deepcopy(first)
    lower["sequence"] = 63
    assert not store.publish_checkpoint(request["model_run_id"], lower, files)

    higher = copy.deepcopy(first)
    higher["sequence"] = 65
    higher["state"] = files.publish_artifact_bytes(
        b"checkpoint-65", media_type="application/octet-stream"
    )
    assert store.publish_checkpoint(request["model_run_id"], higher, files)
    assert store.fail(
        request["model_run_id"],
        build_gateway_error("RUNNER_FAILED", "global decode interrupted"),
        files,
    )
    status = store.status(request["model_run_id"], files)
    manifest = _artifact_json(files, status["terminal_manifest"])
    assert status["phase"] == "PARTIAL"
    assert manifest["checkpoints"] == [higher]


def test_deadline_selects_partial_or_timed_out_and_cancellation_cannot_override_it(
    tmp_path,
):
    limits = _limits()
    limits["max_execution_seconds"] = 2
    clock = _Clock()
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=clock)

    partial_request = _request(1)
    _reserve(store, partial_request, limits, 600, files)
    unit, _input = _finite_publications(partial_request, files, limits)
    store.publish_unit(partial_request["model_run_id"], unit, files)
    clock.advance(seconds=2)
    partial = store.status(partial_request["model_run_id"], files)
    assert partial["phase"] == "PARTIAL"
    partial_manifest = _artifact_json(files, partial["terminal_manifest"])
    assert partial_manifest["complete"] is False
    assert "failure" not in partial
    assert "failure" not in partial_manifest
    assert not store.fail(
        partial_request["model_run_id"],
        build_gateway_error("INSUFFICIENT_CONTEXT", "arrived after deadline"),
        files,
    )
    assert store.status(partial_request["model_run_id"], files) == partial

    timed_request = _request(0)
    timed_request["model_run_id"] = "53535353-5353-4353-8353-535353535353"
    _reserve(store, timed_request, limits, 600, files)
    clock.advance(seconds=2)
    store.cancel(timed_request["model_run_id"], files)
    timed = store.status(timed_request["model_run_id"], files)
    assert timed["phase"] == "TIMED_OUT"
    assert timed["completed_unit_count"] == 0

    cancelled_request = _request(0)
    cancelled_request["model_run_id"] = "54545454-5454-4454-8454-545454545454"
    _reserve(store, cancelled_request, limits, 600, files)
    store.cancel(cancelled_request["model_run_id"], files)
    assert store.status(cancelled_request["model_run_id"], files)["phase"] == (
        "CANCELLED"
    )
    assert not store.fail(
        cancelled_request["model_run_id"],
        build_gateway_error("RESULT_LIMIT_EXCEEDED", "arrived after cancellation"),
        files,
    )
    assert "failure" not in store.status(cancelled_request["model_run_id"], files)


def test_failure_expiry_cleanup_and_source_retention_are_explicit(tmp_path):
    request = _request(0)
    limits = _limits()
    clock = _Clock()
    values = {}
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore(values, max_active_runs=4, clock=clock)
    _reserve(store, request, limits, 2, files)

    source_id = request["inputs"][0]["raster_input"]["source_view"]["artifact"][
        "artifact_id"
    ]
    assert store.has_source_reference(source_id)
    store.fail(
        request["model_run_id"],
        build_gateway_error("RUNNER_FAILED", "model execution failed"),
        files,
    )
    failed = store.status(request["model_run_id"], files)
    assert failed["phase"] == "FAILED"
    assert failed["failure"]["code"] == "RUNNER_FAILED"
    assert store.has_source_reference(source_id)

    clock.advance(seconds=2)
    with pytest.raises(GatewayFailure) as expired:
        store.status(request["model_run_id"], files)
    assert expired.value.error["code"] == "MODEL_RUN_EXPIRED"
    assert not store.has_source_reference(source_id)

    store.cleanup_expired(files)
    assert files.artifact_ids() == set()
    with pytest.raises(GatewayFailure) as absent:
        store.status(request["model_run_id"], files)
    assert absent.value.error["code"] == "MODEL_RUN_NOT_FOUND"


def test_due_unobserved_run_no_longer_consumes_capacity_or_retains_dependencies(
    tmp_path,
):
    limits = _limits()
    limits["max_execution_seconds"] = 2
    clock = _Clock()
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=1, clock=clock)
    first = _request(0)
    source_id = first["inputs"][0]["raster_input"]["source_view"]["artifact"][
        "artifact_id"
    ]
    module_ref = first["producer"]["module_ref"]
    _reserve(store, first, limits, 600, files)

    assert store.has_source_reference(source_id)
    assert store.has_active_module_reference(module_ref)
    clock.advance(seconds=2)
    assert not store.has_source_reference(source_id)
    assert not store.has_active_module_reference(module_ref)

    second = copy.deepcopy(first)
    second["model_run_id"] = "56565656-5656-4656-8656-565656565656"
    accepted, should_dispatch = _reserve(store, second, limits, 600, files)
    assert accepted["model_run_id"] == second["model_run_id"]
    assert should_dispatch is True
    assert store.status(first["model_run_id"], files)["phase"] == "TIMED_OUT"


@pytest.mark.parametrize(
    ("budget_field", "single_value"),
    [
        ("max_transferred_bytes", lambda stream: len(stream.data)),
        ("max_decoded_bytes", lambda stream: stream.decoded_bytes),
        ("max_features", lambda stream: stream.feature_count),
        ("max_vertices", lambda stream: stream.vertex_count),
    ],
)
def test_result_budgets_are_aggregate_across_completed_inputs(
    tmp_path, budget_field, single_value
):
    request = _two_input_request()
    population = _one_feature_population()
    stream = encode_feature_stream(population, {"Tubule"})
    request["result_budget"][budget_field] = 2 * single_value(stream) - 1
    limits = _limits()
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, limits, 600, files)

    _unit, first = _finite_publications(
        request, files, limits, input_index=0, population=population
    )
    _unit, second = _finite_publications(
        request, files, limits, input_index=1, population=population
    )
    assert store.publish_input(
        request["model_run_id"], first, stream.decoded_bytes, files
    )

    with pytest.raises(GatewayFailure) as exceeded:
        store.publish_input(
            request["model_run_id"], second, stream.decoded_bytes, files
        )
    assert exceeded.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert store.status(request["model_run_id"], files)["completed_unit_count"] == 1


def test_recovery_budget_is_aggregate_across_complete_units(tmp_path):
    request = _two_input_request(1)
    limits = _limits()
    limits["max_recovery_state_bytes_per_unit"] = 10
    limits["max_recovery_state_bytes_per_run"] = 15
    files = ArtifactFileStore(tmp_path)
    store = ModelRunStore({}, max_active_runs=4, clock=_Clock())
    _reserve(store, request, limits, 600, files)

    first, _input = _finite_publications(request, files, limits, input_index=0)
    first["reusable_state"] = files.publish_artifact_bytes(
        b"12345678", media_type="application/octet-stream"
    )
    second, _input = _finite_publications(request, files, limits, input_index=1)
    second["reusable_state"] = files.publish_artifact_bytes(
        b"abcdefgh", media_type="application/octet-stream"
    )
    assert store.publish_unit(request["model_run_id"], first, files)

    with pytest.raises(GatewayFailure) as exceeded:
        store.publish_unit(request["model_run_id"], second, files)
    assert exceeded.value.error["code"] == "RESULT_LIMIT_EXCEEDED"
    assert store.status(request["model_run_id"], files)["completed_unit_count"] == 1


def _finite_publications(
    request, files, limits=None, *, input_index=0, population=None
):
    active_limits = limits or _limits()
    selected_input = request["inputs"][input_index]
    input_id = selected_input["input_id"]
    plan = build_expected_model_execution_plans(request, active_limits)[input_id]
    stream = encode_feature_stream(
        population or {"type": "FeatureCollection", "features": []}, {"Tubule"}
    )
    artifact = files.publish_artifact_bytes(
        stream.data,
        media_type="application/geo+json-seq",
        content_encoding="gzip",
    )
    unit = {
        "input_id": input_id,
        "unit_id": 0,
        "reusable_state": artifact,
        "execution_identity_digest": model_run_execution_identity_digest(
            request, input_id, 0, active_limits
        ),
    }
    input_result = {
        "input_id": input_id,
        "input_identity": build_backend_input_identity(selected_input["raster_input"]),
        "execution_plan": plan,
        "feature_stream_artifact": artifact,
        "feature_count": stream.feature_count,
        "vertex_count": stream.vertex_count,
    }
    return unit, input_result


def _two_input_request(index=0):
    request = _request(index)
    second = copy.deepcopy(request["inputs"][0])
    second["input_id"] += 2
    second["output_generation_id"] = "57575757-5757-4757-8757-575757575757"
    second["output"]["scope"] = {
        "type": "parent",
        "region": copy.deepcopy(second["raster_input"]["frame"]["region"]),
        "parent_id": "58585858-5858-4858-8858-585858585858",
    }
    request["inputs"].append(second)
    return request


def _one_feature_population():
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]],
    }
    return build_feature_collection([("Tubule", geometry)], {"Tubule"})


def _reserve(store, request, limits, retention, files):
    return store.reserve(
        request,
        limits,
        retention,
        files,
        require_sources_live_through=lambda _request, _deadline: None,
    )


def _artifact_json(files, reference):
    value = files.read_artifact_block(
        reference["artifact_id"], 0, reference["size_bytes"]
    )
    return parse_json(value)


def _request(index, *, with_resume=False):
    request_fixture = _fixture("model-run-request-v1.json")
    case = request_fixture["cases"][index]
    finite_case = _fixture("finite-inference-v3.json")["cases"][
        case["finiteFixtureCase"]
    ]
    finite = json.loads(finite_case["requestCanonical"])
    region = finite["raster_input"]["frame"]["region"]
    request = {
        "schema_version": MODEL_RUN_SCHEMA_VERSION,
        "action_id": case["actionId"],
        "model_run_id": case["modelRunId"],
        "producer": copy.deepcopy(finite["producer"]),
        "expected_backend": copy.deepcopy(finite["expected_backend"]),
        "inputs": [
            {
                "input_id": case["inputId"],
                "output_generation_id": case["outputGenerationId"],
                "output": {
                    "producer": copy.deepcopy(finite["producer"]),
                    "scope": {"type": "field", "region": copy.deepcopy(region)},
                },
                "raster_input": copy.deepcopy(finite["raster_input"]),
            }
        ],
        "result_budget": copy.deepcopy(finite["result_budget"]),
    }
    profile = None
    if "largeRegionProfileCase" in case:
        profile = _fixture("large-region-execution-v1.json")["profiles"][
            case["largeRegionProfileCase"]
        ]["input"]
        request["expected_profile"] = copy.deepcopy(profile)
    if with_resume and "resumeFrom" in case:
        request["resume_from"] = copy.deepcopy(case["resumeFrom"])
    return validate_model_run_admission(
        request,
        finite_case["route"]["moduleRef"],
        finite_case["route"]["modelId"],
        request["expected_backend"],
        case["targetRasterLimits"],
        profile,
        request_fixture["testModelRunLimits"],
    )


def _limits():
    return copy.deepcopy(_fixture("model-run-request-v1.json")["testModelRunLimits"])


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
