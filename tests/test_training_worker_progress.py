"""Completed model epochs reach reports through the sole training-state writer."""

import copy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
)
from histoforge.cloud.training import state_lane as training_state_lane
from histoforge.cloud.training.runs.request import training_run_request_digest
from histoforge.cloud.training.runs.worker import (
    TrainingWorkerControl,
    publish_training_epoch,
)
from histoforge.cloud.training.trainers.instance.batches import collate_training_fields
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields
from tests.test_training_batches import _field, _profile
from tests.test_training_batches import single_thread_torch as single_thread_torch
from tests.test_training_contract import classified_training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_control import make_control, publish, report_path
from tests.test_training_run_request import run_request
from tests.test_training_source_files import source_paths
from tests.test_training_source_snapshot import source_ready_marker
from tests.test_training_worker import _run
from torch import nn
from torch.utils.data import DataLoader


def _worker(tmp_path, monkeypatch, *, request=None, report=None, claim=True):
    """Use real files, record replacement, and the raw internal Modal methods."""
    request = request if request is not None else run_request()
    report = report if report is not None else queued_report()
    control, files, sources = make_control(tmp_path)
    run_id = control.reserve(request)["admission"]["run_id"]
    digest = publish(files, report)
    marker = source_ready_marker()
    marker.update(
        run_id=run_id,
        request_digest=training_run_request_digest(request, training_api_limits()),
        report_digest=digest,
    )
    sources.publish_candidate(marker, source_paths(tmp_path))
    control.accept(request, trainer_build(), report["total_epochs"], 3_600, digest)
    run = control.attach_worker(run_id, "fc-training")
    lane_type = training_state_lane.TrainingStateLane._get_user_cls()
    lane = lane_type()
    monkeypatch.setattr(lane, "_run_control", lambda: control)
    monkeypatch.setattr(
        training_state_lane,
        "_authorize",
        lambda *_: pytest.fail("worker progress must not depend on user credentials"),
    )
    remote = SimpleNamespace(
        **{
            name: SimpleNamespace(
                remote=partial(lane_type.__dict__[name]._get_raw_f(), lane)
            )
            for name in (
                "get_worker_run_control",
                "claim_run_worker",
                "advance_worker_run_progress",
            )
        }
    )
    worker = TrainingWorkerControl(remote)
    if claim:
        run = worker.claim_run_worker(run_id, trainer_build())
    return run, worker, control, files


def _epoch(number=1, *, value=0.8123456789012345):
    return {
        "epoch": number,
        "elapsed_seconds": number * 10,
        "metrics": [{"series_id": "training-loss.v1", "value": value}],
    }


def _publish(run, epoch, worker, files):
    return publish_training_epoch(
        run, epoch, control=worker, run_files=files, api_limits=training_api_limits()
    )


def test_epochs_keep_frozen_report_content_and_exact_sparse_series(
    tmp_path, monkeypatch
):
    initial = queued_report()
    initial["metric_policy"]["series"][0]["series_id"] = "custom-objective.v3"
    initial["messages"] = [{"severity": "info", "message": "Captured at acceptance"}]
    run, worker, control, files = _worker(tmp_path, monkeypatch, report=initial)
    original_run = copy.deepcopy(run)
    first_epoch = _epoch()
    first_epoch["metrics"][0]["series_id"] = "custom-objective.v3"
    expected_first = copy.deepcopy(first_epoch)

    first = _publish(run, first_epoch, worker, files)
    first_report = control.get_report(run["run_id"])
    assert run == original_run
    assert first["report_revision"] == 2
    assert first["progress"]["completed_epochs"] == 1
    first_epoch["metrics"][0]["value"] = -100  # No caller alias reaches stored history.
    second_epoch = _epoch(2)
    second_epoch["metrics"] = [
        {"series_id": "validation-instance-msa.v1", "value": 0.6181234567890123}
    ]
    second = _publish(first, second_epoch, worker, files)
    final_report = control.get_report(run["run_id"])["report"]

    assert second["report_revision"] == 3
    assert second["progress"]["completed_epochs"] == 2
    assert final_report["epochs"] == [expected_first, second_epoch]
    assert final_report == {
        **initial,
        "state": "RUNNING",
        "epochs": final_report["epochs"],
    }
    assert files.read_report(run["run_id"], 2)["report"] == first_report["report"]
    assert files.read_report(run["run_id"], 1)["report"] == initial


@pytest.mark.parametrize(
    "defect",
    [
        "unknown_series",
        "nan",
        "infinite",
        "duplicate_series",
        "old_epoch",
        "overrun",
        "elapsed",
    ],
)
def test_invalid_epoch_leaves_the_advertised_report_untouched(
    tmp_path, monkeypatch, defect
):
    run, worker, control, files = _worker(tmp_path, monkeypatch)
    run = _publish(run, _epoch(), worker, files)
    original = control.get_report(run["run_id"])
    invalid = _epoch(2)
    if defect == "unknown_series":
        invalid["metrics"][0]["series_id"] = "not-in-the-captured-policy"
    elif defect in {"nan", "infinite"}:
        invalid["metrics"][0]["value"] = float("nan" if defect == "nan" else "inf")
    elif defect == "duplicate_series":
        invalid["metrics"] *= 2
    elif defect == "old_epoch":
        invalid["epoch"] = 1
    elif defect == "overrun":
        invalid["epoch"] = 4
    else:
        invalid["elapsed_seconds"] = 9

    with pytest.raises(ValueError):
        _publish(run, invalid, worker, files)

    assert control.get_report(run["run_id"]) == original
    assert not report_path(tmp_path, run["run_id"], 3).exists()
    assert not list((tmp_path / "training/runs/report-candidates").glob("*/*.json"))


def test_unclaimed_worker_cannot_advertise_an_epoch(tmp_path, monkeypatch):
    run, worker, control, files = _worker(tmp_path, monkeypatch, claim=False)
    with pytest.raises(GatewayFailure, match="progress is no longer accepted"):
        _publish(run, _epoch(), worker, files)
    assert control.get(run["run_id"])["progress"]["completed_epochs"] == 0
    assert not report_path(tmp_path, run["run_id"], 2).exists()


@pytest.mark.parametrize("when", ["before_candidate", "after_candidate"])
def test_cancellation_wins_over_late_worker_progress(tmp_path, monkeypatch, when):
    run, worker, control, files = _worker(tmp_path, monkeypatch)
    run = _publish(run, _epoch(), worker, files)

    def cancel():
        cancelled = control.get_report(run["run_id"])["report"]
        cancelled["state"] = "CANCELLED"
        control.cancel(run["run_id"], 3, publish(files, cancelled))

    if when == "before_candidate":
        cancel()
    else:
        advance = worker.advance_run_progress

        def racing_advance(*args):
            cancel()
            return advance(*args)

        monkeypatch.setattr(worker, "advance_run_progress", racing_advance)
    with pytest.raises(GatewayFailure):
        _publish(run, _epoch(2), worker, files)

    current = control.get(run["run_id"])
    retained = control.get_report(run["run_id"])
    assert current["state"] == retained["report"]["state"] == "CANCELLED"
    assert current["progress"]["completed_epochs"] == 1
    assert retained["report"]["epochs"] == [_epoch()]
    assert retained["report_digest"] == current["result"]["report_digest"]


@pytest.mark.parametrize("cut", ["candidate_commit", "before_state", "after_state"])
def test_publication_failure_never_retries_or_advertises_missing_bytes(
    tmp_path, monkeypatch, cut
):
    run, worker, control, files = _worker(tmp_path, monkeypatch)
    calls = []
    advance = worker.advance_run_progress

    def interrupted(*args):
        calls.append(cut)
        if cut == "after_state":
            advance(*args)
        raise OSError("interrupted publication")

    monkeypatch.setattr(
        files if cut == "candidate_commit" else worker,
        "_commit_volume" if cut == "candidate_commit" else "advance_run_progress",
        interrupted,
    )
    with pytest.raises(OSError, match="interrupted publication"):
        _publish(run, _epoch(), worker, files)

    assert calls == [cut]
    current = control.get(run["run_id"])
    response = control.get_report(run["run_id"])
    expected_epoch = 1 if cut == "after_state" else 0
    assert (
        current["report_revision"] == response["report_revision"] == expected_epoch + 1
    )
    assert current["progress"]["completed_epochs"] == expected_epoch
    assert response["report"]["epochs"] == ([_epoch()] if expected_epoch else [])


@pytest.mark.parametrize("defect", ["request", "run_id", "revision"])
def test_worker_rejects_an_inconsistent_state_response(tmp_path, monkeypatch, defect):
    run, worker, _control, files = _worker(tmp_path, monkeypatch)
    advance = worker.advance_run_progress

    def inconsistent(*args):
        updated = advance(*args)
        if defect == "request":
            updated["request"]["training_seed"] += 1
            updated["request_digest"] = training_run_request_digest(
                updated["request"], training_api_limits()
            )
        elif defect == "run_id":
            updated["run_id"] = "55555555-5555-4555-8555-555555555555"
        else:
            updated["report_revision"] -= 1
        return updated

    monkeypatch.setattr(worker, "advance_run_progress", inconsistent)
    with pytest.raises(ValueError):
        _publish(run, _epoch(), worker, files)


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("classified", [False, True])
@pytest.mark.parametrize("outcome", ["complete", "lost_progress_response", "cancelled"])
def test_managed_optimizer_epochs_reach_durable_reports(
    tmp_path, monkeypatch, head, classified, outcome, single_thread_torch
):
    """Tiny CPU networks verify the worker bridge, not production model accuracy."""
    request = run_request()
    request["split_plan"] = {"type": "none"}
    request["field_profile"] = _profile()
    request["trainer_settings"].update(
        training_batch_size=2, gradient_accumulation_steps=2
    )
    initial = queued_report(classified_training_model_output() if classified else None)
    for field in ("split_plan", "field_profile", "trainer_settings"):
        initial[field] = copy.deepcopy(request[field])
    initial["trainer_configuration"]["preprocessing"]["profile_id"] = _profile()[
        "preprocessing"
    ]
    del initial["validation_counts"]
    if classified:
        initial["training_counts"]["target_object_counts"] = {
            "Leydig cell": 50,
            "Other": 10,
        }
    run, worker, control, files = _worker(
        tmp_path, monkeypatch, request=request, report=initial, claim=False
    )
    count = 3 if classified else 0
    model = nn.Conv2d(3, count + (3 if head == "flow" else 4), 1)
    with torch.no_grad():
        model.weight.fill_(0.05)
        model.bias.fill_(0.1)
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(
        [
            _field(h, w, classified=classified)
            for h, w in ((16, 32), (32, 16), (16, 48))
        ],
        batch_size=request["trainer_settings"]["training_batch_size"],
        collate_fn=partial(
            collate_training_fields,
            preprocess=normalize99,
            field_profile=_profile(),
            class_count=count,
        ),
    )
    observations = []
    sampled_epochs = []
    if outcome != "complete":
        advance = worker.advance_run_progress

        def interrupted_progress(*args):
            if outcome == "cancelled":
                cancelled = control.get_report(run["run_id"])["report"]
                cancelled["state"] = "CANCELLED"
                control.cancel(run["run_id"], 2, publish(files, cancelled))
            advance(*args)
            raise OSError("lost progress response")

        monkeypatch.setattr(worker, "advance_run_progress", interrupted_progress)

    def epoch_batches(index):
        sampled_epochs.append(index)
        return loader

    def train(claimed, _entry):
        current = claimed

        def epoch_completed(result, trained_model):
            nonlocal current
            assert trained_model is model
            assert result.field_count == 3 and result.optimizer_steps == 1
            record = _epoch(result.epoch, value=result.mean_loss)
            observations.append(result)
            current = _publish(current, record, worker, files)
            assert control.get_report(run["run_id"])["report"]["epochs"][-1] == record

        train_instance_fields(
            model,
            optimizer,
            epoch_batches,
            training_forward=instance_training_predictions,
            head_type=head,
            class_count=count,
            learning_rate_scales=[1.0] * initial["total_epochs"],
            gradient_accumulation_steps=request["trainer_settings"][
                "gradient_accumulation_steps"
            ],
            classification_objective="cross_entropy+tversky",
            epoch_completed=epoch_completed,
        )

    if outcome == "complete":
        _run(worker, train)
    else:
        with pytest.raises(GatewayFailure if outcome == "cancelled" else OSError):
            _run(worker, train)

    assert not torch.equal(model.weight, before)
    stored = control.get_report(run["run_id"])["report"]
    completed = 3 if outcome == "complete" else 1
    assert sampled_epochs == list(range(completed))
    assert len(observations) == completed
    advertised = 0 if outcome == "cancelled" else completed
    assert stored["epochs"] == [
        _epoch(result.epoch, value=result.mean_loss)
        for result in observations[:advertised]
    ]
    # Completing optimization alone must not claim successful module publication.
    assert stored["state"] == ("CANCELLED" if outcome == "cancelled" else "RUNNING")
    assert "model_selection" not in stored
    assert control.get(run["run_id"])["progress"] == {
        "completed_epochs": advertised,
        "total_epochs": 3,
    }
