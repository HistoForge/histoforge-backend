"""One claimed worker wires immutable inputs to one result candidate."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from histoforge.canonical_json import sha256_bytes
from histoforge.cloud.training import worker as training_executor
from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID
from histoforge.cloud.training.runs.executor import (
    _unavailable_validation_metrics,
    train_and_publish_claimed_run,
)
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_model import LoadedTrainingSourceModel
from tests.test_training_contract import training_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_output import _epoch, _session


def test_modal_entrypoint_passes_the_claimed_entry_to_shared_execution(monkeypatch):
    root = "/test-artifacts"
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    lane = object()
    registry = object()
    constructed = {}
    published = {}

    def file_factory(name):
        def create(*args, **kwargs):
            value = SimpleNamespace(name=name, args=args, kwargs=kwargs)
            constructed[name] = value
            return value

        return create

    class RemoteClass:
        def __call__(self):
            return lane

    def from_name(app_name, class_name):
        assert app_name == training_executor.DEPLOYMENT_CONFIG.gateway_app_name
        assert (
            class_name == training_executor.DEPLOYMENT_CONFIG.training_state_class_name
        )
        return RemoteClass()

    def run_worker(run_id, **kwargs):
        assert run_id == "run-id"
        assert kwargs["capabilities"] is registry
        assert kwargs["api_limits"] == (
            training_executor.DEPLOYMENT_CONFIG.training_api_limits.wire_value()
        )
        assert kwargs["control"]._lane is lane
        entry = SimpleNamespace(
            trainer_profile={
                "gpu_resources": {
                    "qualified-gpu": {"max_validation_assignment_matrix_entries": 4567}
                }
            }
        )
        kwargs["train_and_publish"](
            {"request": {"trainer_settings": {"training_gpu_id": "qualified-gpu"}}},
            entry,
        )

    monkeypatch.setattr(training_executor, "_ARTIFACT_ROOT", root)
    monkeypatch.setattr(training_executor, "artifact_volume", volume)
    monkeypatch.setattr(training_executor.modal.Cls, "from_name", from_name)
    monkeypatch.setattr(
        training_executor, "deployed_training_registry", lambda: registry
    )
    monkeypatch.setattr(training_executor, "run_training_worker", run_worker)
    for name in (
        "TrainingDatasetFiles",
        "TrainingSourceFiles",
        "TrainingRunFiles",
        "TrainingResultFiles",
    ):
        monkeypatch.setattr(training_executor, name, file_factory(name))

    def train_and_publish(run, entry, **kwargs):
        published.update(run=run, entry=entry, kwargs=kwargs)

    monkeypatch.setattr(
        training_executor, "train_and_publish_claimed_run", train_and_publish
    )

    training_executor.execute_training_run.get_raw_f()("run-id")

    assert set(constructed) == {
        "TrainingDatasetFiles",
        "TrainingSourceFiles",
        "TrainingRunFiles",
        "TrainingResultFiles",
    }
    api_limits = training_executor.DEPLOYMENT_CONFIG.training_api_limits.wire_value()
    marker_limit = training_executor.DEPLOYMENT_CONFIG.limits.max_catalog_encoded_bytes
    assert constructed["TrainingDatasetFiles"].args == (root,)
    assert constructed["TrainingSourceFiles"].args == (root, marker_limit)
    assert constructed["TrainingRunFiles"].args == (root, api_limits)
    assert constructed["TrainingResultFiles"].args == (root, api_limits, marker_limit)
    assert all(
        value.kwargs == {"reload_volume": volume.reload, "commit_volume": volume.commit}
        for value in constructed.values()
    )
    options = published["kwargs"]
    assert options["dataset_files"] is constructed["TrainingDatasetFiles"]
    assert options["source_files"] is constructed["TrainingSourceFiles"]
    assert options["run_files"] is constructed["TrainingRunFiles"]
    assert options["result_files"] is constructed["TrainingResultFiles"]
    assert options["control"]._lane is lane
    assert options["device"] == "cuda"
    assert options["api_limits"] == (
        training_executor.DEPLOYMENT_CONFIG.training_api_limits.wire_value()
    )
    assert options["source_limits"] == (
        training_executor.DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
    )
    assert "max_assignment_matrix_entries" not in options


def test_claimed_run_records_epochs_and_publishes_one_hidden_candidate(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, validation=False)
    source_files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    source = source_files.read_source(session.run["run_id"])
    runtime = SimpleNamespace(
        network=session.model,
        predict=lambda image: image,
    )
    loaded = LoadedTrainingSourceModel(
        model={"fixture": "accepted model"},
        network=runtime.network,
        target_raster_limits={
            "max_width": 2048,
            "max_height": 2048,
            "max_pixels": 2048**2,
        },
    )
    events = []

    class DatasetFiles:
        def read_dataset(self, dataset_id, limits):
            events.append(("dataset", dataset_id, limits))
            return SimpleNamespace(manifest={"fixture": "manifest"})

    def validate(request, manifest, capability, limits):
        events.append(("validate", request, manifest, capability, limits))

    def load(run, report, accepted_source, entry, limits, build_network, adapt_network):
        assert accepted_source == source
        events.append(("model", run, report, entry, limits))
        return loaded

    def train(model, report, dataset, predict, **kwargs):
        assert model is session.model
        assert predict is runtime.predict
        assert kwargs["device"] == "cpu"
        assert kwargs["target_raster_limits"] == loaded.target_raster_limits
        assert kwargs["max_assignment_matrix_entries"] == 1234
        for number in range(1, 4):
            with torch.no_grad():
                model.out.weight.fill_(number)
            kwargs["epoch_completed"](_epoch(number))
        events.append(("trained", dataset, report))

    monkeypatch.setattr(
        "histoforge.cloud.training.runs.executor."
        "validate_training_run_request_against_manifest_capability",
        validate,
    )
    monkeypatch.setattr(
        "histoforge.cloud.training.runs.source_model.load_training_source_model", load
    )

    def create_runtime(model, network):
        assert model is loaded.model and network is runtime.network
        return runtime

    monkeypatch.setattr(
        "histoforge.cloud.architectures.instance_runtime.create_instance_runtime",
        create_runtime,
    )
    monkeypatch.setattr(
        "histoforge.cloud.training.trainers.instance.loop.train_managed_model", train
    )
    gpu_id = session.run["request"]["trainer_settings"]["training_gpu_id"]
    entry = TrainingCapabilityEntry(
        "dinov3-adapter-cellpose",
        training_capability(),
        {"gpu_resources": {gpu_id: {"max_validation_assignment_matrix_entries": 1234}}},
        DINOV3_INSTANCE_TRAINER_ID,
    )

    candidate = train_and_publish_claimed_run(
        session.run,
        entry,
        dataset_files=DatasetFiles(),
        source_files=source_files,
        run_files=session.files,
        result_files=session.result_files,
        control=session.record.keywords["control"],
        device="cpu",
        api_limits=training_api_limits(),
        source_limits={"fixture": "limits"},
    )

    current = session.control.get(session.run["run_id"])
    assert current["state"] == "RUNNING"
    assert current["progress"]["completed_epochs"] == 3
    assert candidate == session.result_files.inspect_candidate(session.run["run_id"])
    selected = candidate.module_dir / "nuclei.safetensors"
    assert candidate.selected_weight_digest == sha256_bytes(selected.read_bytes())
    assert candidate.report["model_selection"] == {
        "type": "final-weights",
        "reason": "no_validation",
        "epoch": 3,
        "selected_weight_digest": candidate.selected_weight_digest,
    }
    assert [event[0] for event in events] == [
        "dataset",
        "validate",
        "model",
        "trained",
    ]


def test_only_unobserved_applicable_validation_metrics_receive_explanations():
    report = {
        "split_plan": {"type": "split"},
        "metric_policy": {
            "series": [
                {"series_id": "loss", "partition": "training"},
                {"series_id": "msa", "partition": "validation"},
                {"series_id": "ap50", "partition": "validation"},
            ]
        },
        "epochs": [
            {
                "metrics": [
                    {"series_id": "loss", "value": 1.0},
                    {"series_id": "msa", "value": 0.5},
                ]
            }
        ],
    }

    assert _unavailable_validation_metrics(report) == (
        {
            "series_id": "ap50",
            "reason": "The complete validation population had no defined value.",
        },
    )
