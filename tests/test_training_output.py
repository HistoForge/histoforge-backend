"""Live epochs select reloadable weights without a checkpoint-resume store."""

import copy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from histoforge._vendor.histoforge_models.pathology.common import load_state_dict_exact
from histoforge._vendor.histoforge_models.pathology.normalization import normalize99
from histoforge.canonical_json import parse_json, sha256_bytes
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.architectures.dinov3_training import project_dinov3_training_model
from histoforge.cloud.architectures.instance_runtime import (
    instance_training_predictions,
    load_instance_model_state,
)
from histoforge.cloud.training.runs.output import (
    publish_training_result_candidate,
    record_training_epoch,
)
from histoforge.cloud.training.runs.request import training_run_request_digest
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from histoforge.cloud.training.trainers.instance.batches import collate_training_fields
from histoforge.cloud.training.trainers.instance.loop import train_instance_fields
from safetensors.torch import load_file, save_file
from tests.test_training_batches import _field, _profile
from tests.test_training_batches import single_thread_torch as single_thread_torch
from tests.test_training_contract import classified_training_model_output
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import queued_report
from tests.test_training_result_module import _accepted_source, _success
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_control import make_control, publish
from tests.test_training_source_preparation import workflow_recipe
from torch import nn
from torch.utils.data import DataLoader


class TinyModel(nn.Module):
    """Small trainable tensor fixture, not a replacement DINO architecture."""

    def __init__(self, head="flow", classified=False):
        super().__init__()
        count = 3 if classified else 0
        self.out = nn.Conv2d(3, 3 if head == "flow" else 4, 1)
        self.backbone = nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer("head_type_code", torch.tensor(int(head == "hovernet")))
        if count:
            self.out_class = nn.Conv2d(3, count, 1)
            self.W3 = nn.Parameter(torch.eye(count), requires_grad=False)
        self.classified = classified

    def forward(self, image):
        image = image * self.backbone[None, :, None, None]
        segmentation = self.out(image)
        return (
            torch.cat((self.out_class(image), segmentation), dim=1)
            if self.classified
            else segmentation
        )


def _session(
    tmp_path, *, validation=True, direction="maximize", head="flow", classified=False
):
    """Real accepted source, immutable reports, state cuts, and result storage."""
    recipe = workflow_recipe()
    selected = recipe["models"]["nuclei"]
    if head == "hovernet":
        selected["arch"] = "dinov3-adapter-hovernet"
    if classified:
        selected["params"]["num_classes"] = 3
        selected["output"] = classified_training_model_output(emitted=("Nuclei",))
    request, marker, source_root = _accepted_source(
        tmp_path / "prepared-source", recipe=recipe
    )
    if not validation:
        request["split_plan"] = {"type": "none"}
    report, _ = _success(tmp_path, request, marker)
    report.update(state="QUEUED", epochs=[])
    del report["model_selection"]
    if not validation:
        del report["validation_counts"]
    if classified:
        classified_output = classified_training_model_output(emitted=("Nuclei",))
        classified_template = queued_report(classified_output)
        report["task"]["model_output"] = classified_output
        report["source_model_output"] = copy.deepcopy(classified_output)
        report["task"]["annotation_class_map"] = {"Nuclei": "Nuclei", "Other": "Other"}
        report["trainer_configuration"]["objective"] = classified_template[
            "trainer_configuration"
        ]["objective"]
        for partition in ("training_counts", "validation_counts"):
            if partition in report:
                report[partition]["target_object_counts"] = {
                    "Nuclei": report[partition]["target_object_counts"].get(
                        "Leydig cell", 0
                    ),
                    "Other": 2,
                }
        policy = report["metric_policy"]
        policy["series"][1].update(
            series_id="validation-object-f1.v1",
            display_name="Object macro F1",
            definition="Macro F1 over emitted object classes.",
        )
        policy["primary_validation_series_id"] = "validation-object-f1.v1"
    report["metric_policy"]["primary_selection_direction"] = direction
    root = tmp_path / "artifacts"
    control, files, sources = make_control(root)
    control.reserve(request)
    marker["report_digest"] = publish(files, report)
    marker["request_digest"] = training_run_request_digest(
        request, training_api_limits()
    )
    sources.publish_candidate(
        marker,
        {
            item["relative_path"]: source_root / item["relative_path"]
            for item in marker["source_files"]
        },
    )
    control.accept(request, trainer_build(), 3, 3_600, marker["report_digest"])
    control.attach_worker(report["run_id"], "fc-training")
    run = control.claim_worker(report["run_id"], trainer_build())
    source_root = root / "training/runs/pending" / run["run_id"]
    worker = SimpleNamespace(advance_run_progress=control.advance_progress)
    weights = tmp_path / "weights"
    weights.mkdir()
    return SimpleNamespace(
        run=run,
        primary_series_id=report["metric_policy"]["primary_validation_series_id"],
        control=control,
        files=files,
        weights=weights,
        model=TinyModel(head, classified),
        result_files=TrainingResultFiles(root, training_api_limits(), 8 * 1024**2),
        record=partial(
            record_training_epoch,
            weights_directory=weights,
            control=worker,
            run_files=files,
            api_limits=training_api_limits(),
        ),
        finish=partial(
            publish_training_result_candidate,
            load_model_state=load_instance_model_state,
            project_model=project_dinov3_training_model,
            weights_directory=weights,
            source_root=source_root,
            source_marker=marker,
            run_files=files,
            result_files=TrainingResultFiles(root, training_api_limits(), 8 * 1024**2),
            api_limits=training_api_limits(),
        ),
    )


def _epoch(number, primary=None, *, primary_id="validation-instance-msa.v1"):
    metrics = [{"series_id": "training-loss.v1", "value": 1 / number}]
    if primary is not None:
        metrics.append({"series_id": primary_id, "value": primary})
    return {"epoch": number, "elapsed_seconds": number * 10, "metrics": metrics}


def _snapshot(model):
    return {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }


def _assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype
        assert torch.equal(actual[name], value), name


@pytest.mark.parametrize(
    "values,direction,validation,selected_epoch,reason",
    [
        ([0.6, 0.7, 0.7], "maximize", True, 2, None),
        ([0.4, 0.3, 0.3], "minimize", True, 2, None),
        ([0.0, None, None], "maximize", True, 1, None),
        ([None, 0.8, 0.9], "maximize", True, 3, None),
        ([None, None, None], "maximize", True, 3, "primary_metric_unavailable"),
        ([None, None, None], "maximize", False, 3, "no_validation"),
    ],
)
def test_actual_weight_files_follow_the_frozen_report_selection(
    tmp_path, monkeypatch, values, direction, validation, selected_epoch, reason
):
    session = _session(tmp_path, validation=validation, direction=direction)
    run = session.run
    original = copy.deepcopy(run)
    states = []
    saved_paths = []

    def counted_save(state, path):
        saved_paths.append(path)
        save_file(state, path)

    monkeypatch.setattr("safetensors.torch.save_file", counted_save)
    for number, value in enumerate(values, 1):
        with torch.no_grad():
            session.model.out.weight.fill_(number)
        states.append(_snapshot(session.model))
        run = session.record(run, _epoch(number, value), session.model)
        if number == 3:
            _assert_state(
                load_file(str(session.weights / "final.safetensors")), states[-1]
            )
        else:
            assert not (session.weights / "final.safetensors").exists()
        assert len(list(session.weights.iterdir())) <= 2
    # One write for each improving score before the final epoch, then one final
    # write. A final-epoch best reuses those bytes without another tensor snapshot.
    expected_best_writes = {
        (0.6, 0.7, 0.7): 2,
        (0.4, 0.3, 0.3): 2,
        (0.0, None, None): 1,
        (None, 0.8, 0.9): 1,
        (None, None, None): 0,
    }[tuple(values)]
    assert saved_paths == [
        str(session.weights / "best.safetensors")
    ] * expected_best_writes + [str(session.weights / "final.safetensors")]
    explanations = (
        [
            {
                "series_id": "validation-instance-msa.v1",
                "reason": "No defined validation score.",
            }
        ]
        if reason == "primary_metric_unavailable"
        else []
    )
    reloads = []

    def reload_selected(network, tensor_state):
        assert network is session.model
        _assert_state(tensor_state, states[selected_epoch - 1])
        reloads.append(selected_epoch)
        load_instance_model_state(network, tensor_state)

    candidate = session.finish(
        run,
        session.model,
        unavailable_metrics=explanations,
        load_model_state=reload_selected,
    )
    assert reloads == [selected_epoch]
    selection = candidate.report["model_selection"]
    selected_bytes = (candidate.module_dir / "nuclei.safetensors").read_bytes()

    assert selection["epoch"] == selected_epoch
    assert selection.get("reason") == reason
    assert selection["selected_weight_digest"] == sha256_bytes(selected_bytes)
    _assert_state(
        load_file(str(candidate.module_dir / "nuclei.safetensors")),
        states[selected_epoch - 1],
    )
    _assert_state(session.model.state_dict(), states[selected_epoch - 1])
    assert session.run == original
    assert session.control.get(run["run_id"])["state"] == "RUNNING"
    assert "model_selection" not in session.control.get_report(run["run_id"])["report"]
    assert session.result_files.inspect_candidate(run["run_id"]) == candidate
    assert (
        parse_json((candidate.module_dir / "recipe.json").read_bytes())[
            "trainingReportDigest"
        ]
        == candidate.report_digest
    )


def test_snapshot_preserves_frozen_buffers_shared_storage_and_noncontiguous_tensors(
    tmp_path,
):
    session = _session(tmp_path)
    model = session.model
    model.register_buffer("view", torch.arange(6).reshape(2, 3).T)
    model.register_buffer("shared", model.view)
    before = _snapshot(model)
    session.record(session.run, _epoch(1, 0.5), model)
    with torch.no_grad():
        model.out.weight.add_(100)
        model.backbone.add_(100)
        model.view.add_(100)
    saved = load_file(str(session.weights / "best.safetensors"))
    _assert_state(saved, before)
    load_state_dict_exact(model, saved)
    _assert_state(model.state_dict(), before)


@pytest.mark.parametrize("head", ["flow", "hovernet"])
@pytest.mark.parametrize("classified", [False, True])
def test_managed_optimizer_callback_publishes_exact_selected_inference_weights(
    tmp_path, head, classified, single_thread_torch
):
    session = _session(tmp_path, head=head, classified=classified)
    model = session.model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = _snapshot(model)
    loader = DataLoader(
        [_field(16, 32, classified=classified), _field(32, 16, classified=classified)],
        batch_size=2,
        collate_fn=partial(
            collate_training_fields,
            preprocess=normalize99,
            field_profile=_profile(),
            class_count=3 if classified else 0,
        ),
    )
    run = session.run
    selected_state = None

    def completed(epoch, network):
        nonlocal run, selected_state
        # Controlled scores isolate weight selection. The metric kernel and
        # normal-decoder validation adapter have their own scientific fixtures.
        record = _epoch(
            epoch.epoch,
            [0.4, 0.6, 0.5][epoch.epoch - 1],
            primary_id=session.primary_series_id,
        )
        record["metrics"][0]["value"] = epoch.mean_loss
        run = session.record(run, record, network)
        if epoch.epoch == 2:
            selected_state = _snapshot(network)

    train_instance_fields(
        model,
        optimizer,
        lambda _epoch: loader,
        training_forward=instance_training_predictions,
        head_type=head,
        class_count=3 if classified else 0,
        learning_rate_scales=[1.0] * 3,
        gradient_accumulation_steps=1,
        classification_objective="cross_entropy+tversky",
        epoch_completed=completed,
    )
    assert not torch.equal(model.out.weight, before["out.weight"])
    candidate = session.finish(run, model, unavailable_metrics=[])
    restored = TinyModel(head, classified)
    saved = load_file(str(candidate.module_dir / "nuclei.safetensors"))
    load_state_dict_exact(restored, saved)
    _assert_state(restored.state_dict(), selected_state)
    with torch.inference_mode():
        image = torch.randn(1, 3, 16, 32)
        assert torch.equal(restored(image), model(image))
    assert "backbone" in saved and "head_type_code" in saved
    assert ("out_class.weight" in saved) == classified
    assert ("W3" in saved) == classified


@pytest.mark.parametrize("failure", ["invalid_metric", "save", "copy", "progress"])
def test_epoch_failure_cannot_publish_a_result_or_continue_silently(
    tmp_path, monkeypatch, failure
):
    session = _session(tmp_path)
    record = _epoch(3 if failure == "copy" else 1, 0.7)
    if failure == "invalid_metric":
        record["metrics"][0]["value"] = float("nan")
    else:

        def fail(*args, **kwargs):
            raise OSError("injected output failure")

        path = {"save": "safetensors.torch.save_file", "copy": "shutil.copyfile"}.get(
            failure
        )
        if path:
            monkeypatch.setattr(path, fail)
        else:
            monkeypatch.setattr(session.files, "publish_report_candidate", fail)
    with pytest.raises(ValueError if failure == "invalid_metric" else OSError):
        session.record(session.run, record, session.model)
    assert (
        session.control.get(session.run["run_id"])["progress"]["completed_epochs"] == 0
    )
    assert session.result_files.inspect_candidate(session.run["run_id"]) is None
    if failure == "invalid_metric":
        assert not list(session.weights.iterdir())


@pytest.mark.parametrize(
    "failure",
    ["unfinished", "missing_metric_reason", "bad_weights", "reload", "publication"],
)
def test_incomplete_or_invalid_result_never_becomes_a_candidate(
    tmp_path, monkeypatch, failure
):
    session = _session(tmp_path)
    run = session.run
    for number in range(1, 2 if failure == "unfinished" else 4):
        run = session.record(
            run,
            _epoch(number, None if failure == "missing_metric_reason" else 0.5),
            session.model,
        )
    if failure == "bad_weights":
        save_file(
            {"incorrect": torch.zeros(1)}, str(session.weights / "best.safetensors")
        )
    if failure == "publication":

        def fail(*args, **kwargs):
            raise OSError("candidate publication failed")

        monkeypatch.setattr(
            session.finish.keywords["result_files"], "publish_candidate", fail
        )
    if failure == "reload":

        def fail_reload(network, state):
            assert network is session.model
            raise ValueError("selected checkpoint is incompatible")

        def unexpected_publication(*args, **kwargs):
            pytest.fail("a failed reload must stop before candidate publication")

        session.finish.keywords["load_model_state"] = fail_reload
        monkeypatch.setattr(
            session.finish.keywords["result_files"],
            "publish_candidate",
            unexpected_publication,
        )
    with pytest.raises((ValueError, RuntimeError, OSError)):
        session.finish(run, session.model, unavailable_metrics=[])
    assert session.control.get(run["run_id"])["state"] == "RUNNING"
    assert session.result_files.inspect_candidate(run["run_id"]) is None


def test_cancellation_rejects_the_next_saved_epoch(tmp_path):
    session = _session(tmp_path)
    run = session.record(session.run, _epoch(1, 0.6), session.model)
    cancelled = session.control.get_report(run["run_id"])["report"]
    cancelled["state"] = "CANCELLED"
    session.control.cancel(run["run_id"], 3, publish(session.files, cancelled))
    with pytest.raises(GatewayFailure):
        session.record(run, _epoch(2, 0.9), session.model)
    current = session.control.get(run["run_id"])
    assert current["state"] == "CANCELLED"
    assert current["progress"]["completed_epochs"] == 1
    assert session.result_files.inspect_candidate(run["run_id"]) is None
