"""Finalization counter and report publication must agree at every race cut."""

import copy

import pytest
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt
from tests.test_training_finalization import reconciler
from tests.test_training_run_reconciliation import RUN_ID, _finalizing_context

ERROR = {"schema_version": 1, "code": "INTERNAL_ERROR", "message": "save rejected"}


@pytest.mark.parametrize("outcome", ["FAILED", "SUCCEEDED"])
@pytest.mark.parametrize("interrupted_promotion", [False, True])
def test_stale_outcome_never_promotes_or_overwrites_report_bytes(
    tmp_path, outcome, interrupted_promotion
):
    context = _finalizing_context(tmp_path)
    control, files = context["control"], context["run_files"]
    before = control.get(RUN_ID)
    next_revision = before["report_revision"] + 1
    control.claim_finalization(RUN_ID, 0)
    current = control.claim_finalization(RUN_ID, 1)
    path = (
        context["artifact_root"]
        / "training/runs"
        / RUN_ID
        / "reports"
        / f"{next_revision}.json"
    )
    if interrupted_promotion:
        digest = files.publish_report_candidate(context["success_report"])[
            "report_digest"
        ]
        files.promote_report(RUN_ID, next_revision, digest)
    before_bytes = path.read_bytes() if path.exists() else None
    report = copy.deepcopy(context["success_report"])
    report["state"] = outcome
    if outcome == "FAILED":
        report["failure"] = ERROR
    digest = files.publish_report_candidate(report)["report_digest"]

    with pytest.raises(GatewayFailure, match="revision is not current"):
        if outcome == "FAILED":
            control.fail(RUN_ID, "FINALIZING", next_revision, digest, 1)
        else:
            control.succeed(RUN_ID, next_revision, digest, 1)

    assert control.get_control(RUN_ID) == current
    assert control.get(RUN_ID) == before
    assert (path.read_bytes() if path.exists() else None) == before_bytes
    # Even a stale *success* cannot be accepted just because its report matches.
    assert (
        files.read_report_for_promotion(RUN_ID, next_revision, digest)["report"]
        == report
    )


@pytest.mark.parametrize("race_during_rpc", [False, True])
def test_terminalizer_does_not_rebase_a_stale_finalization_failure(
    tmp_path, monkeypatch, race_during_rpc
):
    context = _finalizing_context(tmp_path)
    control, client = context["control"], context["client"]
    observed = control.claim_finalization(RUN_ID, 0)
    original_fail = client.fail_run
    calls = []

    def advance_before_failure(*args, **kwargs):
        calls.append(kwargs["finalization_revision"])
        control.claim_finalization(RUN_ID, 1)
        return original_fail(*args, **kwargs)

    monkeypatch.setattr(client, "fail_run", advance_before_failure)
    if not race_during_rpc:
        control.claim_finalization(RUN_ID, 1)
    latest = context["terminalizer"].fail(observed, ERROR)

    assert latest["run"] == observed["run"]
    assert latest["finalization_revision"] == 2
    assert calls == ([1] if race_during_rpc else [])
    assert context["terminalizer"].fail(observed, ERROR) == latest
    assert calls == ([1] if race_during_rpc else [])
    next_revision = observed["run"]["report_revision"] + 1
    path = (
        context["artifact_root"]
        / "training/runs"
        / RUN_ID
        / "reports"
        / f"{next_revision}.json"
    )
    assert not path.exists()


@pytest.mark.parametrize("outcome", ["FAILED", "SUCCEEDED"])
def test_terminal_outcome_replays_only_with_its_exact_finalization_counter(
    tmp_path, outcome
):
    context = _finalizing_context(tmp_path)
    control = context["control"]
    control.claim_finalization(RUN_ID, 0)
    current = control.claim_finalization(RUN_ID, 1)
    revision = current["run"]["report_revision"] + 1
    report = copy.deepcopy(context["success_report"])
    report["state"] = outcome
    if outcome == "FAILED":
        report["failure"] = ERROR
    digest = context["run_files"].publish_report_candidate(report)["report_digest"]

    def commit(counter):
        if outcome == "FAILED":
            return control.fail(RUN_ID, "FINALIZING", revision, digest, counter)
        return control.succeed(RUN_ID, revision, digest, counter)

    first = commit(2)
    assert first["state"] == outcome
    assert commit(2) == first
    with pytest.raises(GatewayFailure, match="revision is not current"):
        commit(1)
    with pytest.raises(GatewayFailure, match="not finalizing"):
        control.claim_finalization(RUN_ID, 2)
    assert control.get(RUN_ID) == first


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("bad outside-writer candidate"),
        ArtifactBytesCorrupt("unknown bytes"),
        GatewayFailure(GatewayErrorCode.INTERNAL_ERROR, "outside writer failure"),
    ],
)
def test_reconciler_never_decides_a_finalizing_failure_outside_writer(
    tmp_path, monkeypatch, failure
):
    context = _finalizing_context(tmp_path)
    before = context["control"].get_control(RUN_ID)

    def reject(*args):
        raise failure

    monkeypatch.setattr(context["finalizer"], "reconcile_candidate", reject)
    assert reconciler(context).reconcile(RUN_ID) == before["run"]
    assert context["control"].get_control(RUN_ID) == before


def test_running_failure_observation_does_not_rebase_onto_finalizing(
    tmp_path, monkeypatch
):
    context = _finalizing_context(tmp_path)
    # Recreate an observation from before adoption, while retaining the real
    # finalizing owner. The observer's invalid-candidate error is already stale.
    old = copy.deepcopy(context["control"].get_control(RUN_ID))
    old["run"]["state"] = "RUNNING"
    old["run"].pop("finalization")
    get_control = context["client"].get_run_control
    reads = 0

    def initial_stale_read(run_id):
        nonlocal reads
        reads += 1
        return old if reads == 1 else get_control(run_id)

    def reject(*args):
        raise ValueError("candidate validation failed before noticing adoption")

    monkeypatch.setattr(context["client"], "get_run_control", initial_stale_read)
    monkeypatch.setattr(context["finalizer"], "reconcile_candidate", reject)
    result = reconciler(context).reconcile(RUN_ID)
    assert result["state"] == "FINALIZING"
    assert context["control"].get_control(RUN_ID)["finalization_revision"] == 0
