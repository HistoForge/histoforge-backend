"""Real-owner save decisions, storage rejection and uncertain-response recovery."""

from dataclasses import replace

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.training.runs.reconciliation import TrainingRunReconciler
from tests.test_module_store import _install, _source
from tests.test_training_dataset import training_api_limits
from tests.test_training_run_reconciliation import (
    NOW,
    RUN_ID,
    _assert_only_advertised_report,
    _Call,
    _finalizing_context,
    _running_context,
)


def reconciler(context):
    return TrainingRunReconciler(
        context["client"],
        context["finalizer"],
        context["terminalizer"],
        lambda _: _Call(),
        training_api_limits(),
        clock=lambda: NOW,
    )


def fill_module_store(context, tmp_path):
    store = context["result_store"]
    _install(store, _source(tmp_path / "unrelated", weights=b"x" * 100_000))
    store.max_module_bytes = store.stored_bytes()
    store.max_installed_bytes = store.stored_bytes()


def test_frozen_result_is_saved_once_and_terminal_replay_does_not_install(
    tmp_path, monkeypatch
):
    context = _finalizing_context(tmp_path)
    store = context["result_store"]
    first = context["finalize_result"](RUN_ID)
    assert first["state"] == "SUCCEEDED"
    record = store.current.revalidate(f"workspace/run-{RUN_ID}")
    assert record["training_result_attestation"] == context["candidate"].attestation
    assert record["content_digest"] == context["candidate"].module_content_digest
    monkeypatch.setattr(
        store, "install_from_directory", lambda *a, **kw: pytest.fail("installed twice")
    )
    assert context["finalize_result"](RUN_ID) == first
    assert context["control"].get_control(RUN_ID)["finalization_revision"] == 1


def test_save_requires_finalizing_owner_state(tmp_path):
    context = _running_context(tmp_path)
    with pytest.raises(ValueError, match="only while finalizing"):
        context["finalize_result"](RUN_ID)
    assert context["result_store"].current.list_module_refs() == []
    assert context["control"].get_control(RUN_ID)["finalization_revision"] == 0


def test_frozen_weight_drift_fails_before_install(tmp_path, monkeypatch):
    context = _finalizing_context(tmp_path)
    changed = replace(context["candidate"], selected_weight_digest="f" * 64)
    monkeypatch.setattr(
        context["result_files"], "require_candidate", lambda *a: changed
    )
    failed = reconciler(context).reconcile(RUN_ID)
    assert failed["state"] == "FAILED"
    assert context["result_store"].current.list_module_refs() == []
    assert (
        context["control"].get_report(RUN_ID)["report"]["failure"]
        == (failed["result"]["error"])
    )


@pytest.mark.parametrize("limit", ["module", "installed", "catalog"])
def test_definite_limit_rejection_fails_and_releases_dataset(tmp_path, limit):
    context = _finalizing_context(tmp_path)
    store = context["result_store"]
    if limit == "module":
        store.max_module_bytes = 1
    elif limit == "catalog":
        store.max_catalog_encoded_bytes = 1
    else:
        fill_module_store(context, tmp_path)
    before_refs = store.current.list_module_refs()
    assert context["state"].dataset_in_use(context["request"]["dataset_id"])

    failed = reconciler(context).reconcile(RUN_ID)

    assert failed["state"] == "FAILED"
    assert "storage or catalog limits" in failed["result"]["error"]["message"]
    assert store.current.list_module_refs() == before_refs
    assert not context["state"].dataset_in_use(context["request"]["dataset_id"])
    assert context["result_files"].inspect_candidate(RUN_ID) is None
    _assert_only_advertised_report(context, failed)


def test_missing_candidate_fails_only_after_writer_proves_no_installed_result(tmp_path):
    context = _finalizing_context(tmp_path)
    context["result_files"].delete_candidate_after_fence(RUN_ID)
    assert reconciler(context).reconcile(RUN_ID)["state"] == "FAILED"
    assert context["result_store"].current.list_module_refs() == []


@pytest.mark.parametrize("committed", [False, True])
def test_unknown_claim_ack_never_installs_before_acknowledgment(
    tmp_path, monkeypatch, committed
):
    context = _finalizing_context(tmp_path)
    claim = context["client"].claim_run_finalization

    def lose_claim(*args):
        if committed:
            claim(*args)
        raise ConnectionError("claim response lost")

    monkeypatch.setattr(context["client"], "claim_run_finalization", lose_claim)
    before = context["control"].get(RUN_ID)
    assert reconciler(context).reconcile(RUN_ID) == before
    assert context["result_store"].current.list_module_refs() == []
    assert context["control"].get_control(RUN_ID)["finalization_revision"] == int(
        committed
    )
    monkeypatch.setattr(context["client"], "claim_run_finalization", claim)
    assert reconciler(context).reconcile(RUN_ID)["state"] == "SUCCEEDED"
    assert context["control"].get_control(RUN_ID)["finalization_revision"] == (
        1 + int(committed)
    )


@pytest.mark.parametrize("committed", [False, True])
def test_unknown_success_ack_recovers_exact_module_without_transient_candidate(
    tmp_path, monkeypatch, committed
):
    context = _finalizing_context(tmp_path)
    succeed = context["client"].succeed_run
    pending = []

    def lose_success(*args):
        pending.append(args)
        if committed:
            succeed(*args)
        raise ConnectionError("success response lost")

    monkeypatch.setattr(context["client"], "succeed_run", lose_success)
    first = reconciler(context).reconcile(RUN_ID)
    assert first["state"] == ("SUCCEEDED" if committed else "FINALIZING")
    store = context["result_store"]
    installed = store.current.revalidate(f"workspace/run-{RUN_ID}")
    context["result_files"].delete_candidate_after_fence(RUN_ID)
    store.max_module_bytes = store.max_installed_bytes = 1
    store.max_catalog_encoded_bytes = 1
    monkeypatch.setattr(
        context["result_files"],
        "require_candidate",
        lambda *a: pytest.fail("candidate read"),
    )
    monkeypatch.setattr(
        store,
        "install_from_directory",
        lambda *a, **kw: pytest.fail("new installation"),
    )
    monkeypatch.setattr(context["client"], "succeed_run", succeed)

    succeeded = reconciler(context).reconcile(RUN_ID)

    assert succeeded["state"] == "SUCCEEDED"
    assert store.current.revalidate(f"workspace/run-{RUN_ID}") == installed
    assert context["control"].get_report(RUN_ID)["report"] == context["success_report"]
    _assert_only_advertised_report(context, succeeded)
    if not committed:
        with pytest.raises(GatewayFailure, match="revision is not current"):
            succeed(*pending[0])
    else:
        assert succeed(*pending[0]) == succeeded


def test_unknown_installed_inspection_preserves_reports_across_save_attempts(
    tmp_path, monkeypatch
):
    context = _finalizing_context(tmp_path)
    store = context["result_store"]
    inspect = store.matching_training_result
    report_root = context["artifact_root"] / "training/runs" / RUN_ID / "reports"
    before_bytes = {path.name: path.read_bytes() for path in report_root.iterdir()}
    before_run = context["control"].get(RUN_ID)

    def unknown(*args):
        raise OSError("module volume unavailable")

    monkeypatch.setattr(store, "matching_training_result", unknown)
    for revision in range(1, 5):
        assert reconciler(context).reconcile(RUN_ID) == before_run
        assert (
            context["control"].get_control(RUN_ID)["finalization_revision"] == revision
        )
        assert {
            path.name: path.read_bytes() for path in report_root.iterdir()
        } == before_bytes
    assert context["state"].dataset_in_use(context["request"]["dataset_id"])
    monkeypatch.setattr(store, "matching_training_result", inspect)
    assert reconciler(context).reconcile(RUN_ID)["state"] == "SUCCEEDED"


@pytest.mark.parametrize("cut", ["failure_first", "before_install", "after_install"])
def test_delayed_capacity_failure_cannot_defeat_a_new_save_attempt(
    tmp_path, monkeypatch, cut
):
    context = _finalizing_context(tmp_path)
    fill_module_store(context, tmp_path)
    client = context["client"]
    fail_run = client.fail_run
    pending = []

    def delay_failure(*args, **kwargs):
        pending.append((args, kwargs))
        raise ConnectionError("failure request queued but response lost")

    def leave_report_candidate(*args):
        raise OSError("report candidate cleanup unavailable")

    monkeypatch.setattr(client, "fail_run", delay_failure)
    monkeypatch.setattr(
        context["run_files"], "discard_report_candidate", leave_report_candidate
    )
    assert reconciler(context).reconcile(RUN_ID)["state"] == "FINALIZING"
    assert len(pending) == 1
    args, kwargs = pending[0]
    assert kwargs["finalization_revision"] == 1
    # The delayed request can still read its candidate: cleanup is NOT our fence.
    context["run_files"].read_report_for_promotion(RUN_ID, args[2], args[3])
    store = context["result_store"]
    store.delete("owner/module", job_id=RUN_ID, publish=lambda: None)
    monkeypatch.setattr(client, "fail_run", fail_run)

    if cut == "failure_first":
        assert fail_run(*args, **kwargs)["state"] == "FAILED"
        assert reconciler(context).reconcile(RUN_ID)["state"] == "FAILED"
        assert store.current.list_module_refs() == []
        assert context["control"].get_control(RUN_ID)["finalization_revision"] == 1
        return

    install = store.install_from_directory

    def rejected_delayed_failure():
        assert context["control"].get_control(RUN_ID)["finalization_revision"] == 2
        with pytest.raises(GatewayFailure, match="revision is not current"):
            fail_run(*args, **kwargs)
        assert context["control"].get(RUN_ID)["state"] == "FINALIZING"
        report = context["artifact_root"] / "training/runs" / RUN_ID / "reports/4.json"
        assert not report.exists()

    def save_with_late_failure(*install_args, **install_kwargs):
        if cut == "before_install":
            rejected_delayed_failure()
        installed = install(*install_args, **install_kwargs)
        if cut == "after_install":
            rejected_delayed_failure()
        return installed

    monkeypatch.setattr(store, "install_from_directory", save_with_late_failure)
    succeeded = reconciler(context).reconcile(RUN_ID)
    assert succeeded["state"] == "SUCCEEDED"
    assert store.current.list_module_refs() == [f"workspace/run-{RUN_ID}"]
    assert not context["state"].dataset_in_use(context["request"]["dataset_id"])
    assert context["control"].get_report(RUN_ID)["report"] == context["success_report"]
    _assert_only_advertised_report(context, succeeded)


def test_late_claim_cannot_invalidate_a_new_writer_after_lost_response(
    tmp_path, monkeypatch
):
    context = _finalizing_context(tmp_path)
    claim = context["client"].claim_run_finalization
    pending = []

    def delay_claim(*args):
        pending.append(args)
        raise ConnectionError("claim queued but response lost")

    monkeypatch.setattr(context["client"], "claim_run_finalization", delay_claim)
    assert reconciler(context).reconcile(RUN_ID)["state"] == "FINALIZING"
    monkeypatch.setattr(context["client"], "claim_run_finalization", claim)
    store = context["result_store"]
    install = store.install_from_directory

    def save(*args, **kwargs):
        with pytest.raises(GatewayFailure, match="revision is not current"):
            claim(*pending[0])
        assert context["control"].get_control(RUN_ID)["finalization_revision"] == 1
        return install(*args, **kwargs)

    monkeypatch.setattr(store, "install_from_directory", save)
    assert reconciler(context).reconcile(RUN_ID)["state"] == "SUCCEEDED"
