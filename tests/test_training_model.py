"""Managed training loads one exact accepted source model and predictor."""

from __future__ import annotations

from dataclasses import replace

import pytest
from histoforge.cloud.architectures.instance_runtime import create_instance_runtime
from histoforge.cloud.training.capabilities import TrainingCapabilityEntry
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID
from histoforge.cloud.training.runs.report import training_report_digest
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_model import load_training_source_model
from histoforge.cloud.training.runs.worker import read_running_training_report
from tests.test_training_contract import training_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_output import _session


def _entry() -> TrainingCapabilityEntry:
    return TrainingCapabilityEntry(
        "dinov3-adapter-cellpose", training_capability(), {}, DINOV3_INSTANCE_TRAINER_ID
    )


def test_loader_uses_only_the_accepted_snapshot_and_ordinary_runtime(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, validation=False)
    source = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2).read_source(
        session.run["run_id"]
    )
    report = read_running_training_report(
        session.run,
        run_files=session.files,
        api_limits=training_api_limits(),
    )
    loaded_paths = []
    built = []
    adaptation = []
    state = {"tensor": object()}
    network = object()

    def load_file(path):
        loaded_paths.append(path)
        return state

    def builder_for(architecture):
        assert architecture == "dinov3-adapter-cellpose"

        def build(parameters, supplied):
            built.append((parameters, supplied))
            return network

        return build

    monkeypatch.setattr("safetensors.torch.load_file", load_file)

    def adapt(loaded_network, target_model, mode, seed):
        assert len(built) == 1
        assert loaded_network is network
        adaptation.append((target_model, mode, seed))

    loaded = load_training_source_model(
        session.run,
        report,
        source,
        _entry(),
        training_api_limits(),
        build_network=builder_for(_entry().architecture),
        adapt_network=adapt,
    )
    runtime = create_instance_runtime(loaded.model, loaded.network)
    assert runtime.network is network
    assert runtime.field_options == {"bsize": 256, "tile_overlap": 0.1}
    assert runtime.emitted_classes == ("Nuclei",)
    assert runtime.decoder_options["max_size_fraction"] == 1.0
    assert loaded_paths == [str(source.root / source.marker["starting_weight_path"])]
    assert built == [({"min_instance_pixels": 15, "num_classes": 0}, state)]
    assert adaptation == [
        (loaded.model, "use", session.run["request"]["training_seed"])
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_model_microns_per_pixel", 0.37),
        (
            "source_model_output",
            {"type": "instance_segmentation", "classes": ["Incorrect source"]},
        ),
    ],
)
def test_loader_binds_source_taxonomy_and_resolution_before_reading_weights(
    tmp_path, monkeypatch, field, value
):
    session = _session(tmp_path, validation=False)
    source = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2).read_source(
        session.run["run_id"]
    )
    report = read_running_training_report(
        session.run, run_files=session.files, api_limits=training_api_limits()
    )
    report[field] = value
    source = replace(
        source,
        marker={
            **source.marker,
            "report_digest": training_report_digest(report, training_api_limits()),
        },
    )
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda *_: pytest.fail("source mismatch reached weight loading"),
    )
    with pytest.raises(ValueError, match="training source (output|resolution) differs"):
        load_training_source_model(
            session.run,
            report,
            source,
            _entry(),
            training_api_limits(),
            build_network=lambda *_: pytest.fail(
                "source mismatch reached construction"
            ),
            adapt_network=lambda *_: pytest.fail("source mismatch reached adaptation"),
        )


@pytest.mark.parametrize("mismatch", ["run", "request", "module", "weights"])
def test_loader_rejects_source_identity_mismatch_before_reading_weights(
    tmp_path, monkeypatch, mismatch
):
    session = _session(tmp_path, validation=False)
    source = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2).read_source(
        session.run["run_id"]
    )
    report = read_running_training_report(
        session.run,
        run_files=session.files,
        api_limits=training_api_limits(),
    )
    marker = dict(source.marker)
    if mismatch == "run":
        marker["run_id"] = "55555555-5555-4555-8555-555555555555"
    elif mismatch == "request":
        marker["request_digest"] = "f" * 64
    elif mismatch == "module":
        marker["source_module"] = {
            **marker["source_module"],
            "content_digest": "f" * 64,
        }
    else:
        marker["starting_weight_digest"] = "f" * 64
    source = replace(source, marker=marker)
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda *_: pytest.fail("identity mismatch must precede weight loading"),
    )

    with pytest.raises(ValueError, match="training source"):
        load_training_source_model(
            session.run,
            report,
            source,
            _entry(),
            training_api_limits(),
            build_network=lambda *_: pytest.fail(
                "identity mismatch reached model construction"
            ),
            adapt_network=lambda *_args, **_kwargs: pytest.fail(
                "identity mismatch reached adaptation"
            ),
        )


def test_loader_rejects_a_source_model_that_differs_from_the_accepted_task(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, validation=False)
    source = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2).read_source(
        session.run["run_id"]
    )
    report = read_running_training_report(
        session.run,
        run_files=session.files,
        api_limits=training_api_limits(),
    )
    capability = training_capability()
    capability["supports_negative_rois"] = False
    entry = replace(_entry(), capability=capability)
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda *_: pytest.fail("task mismatch must precede weight loading"),
    )

    with pytest.raises(ValueError, match="differs from the accepted task"):
        load_training_source_model(
            session.run,
            report,
            source,
            entry,
            training_api_limits(),
            build_network=lambda *_: pytest.fail(
                "task mismatch reached model construction"
            ),
            adapt_network=lambda *_args, **_kwargs: pytest.fail(
                "task mismatch reached adaptation"
            ),
        )


def test_loader_rejects_a_report_other_than_the_source_bound_initial_report(
    tmp_path, monkeypatch
):
    session = _session(tmp_path, validation=False)
    source = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2).read_source(
        session.run["run_id"]
    )
    report = read_running_training_report(
        session.run,
        run_files=session.files,
        api_limits=training_api_limits(),
    )
    report["trainer_configuration"]["optimizer"]["parameters"]["weight_decay"] = 0.2
    monkeypatch.setattr(
        "safetensors.torch.load_file",
        lambda *_: pytest.fail("report mismatch must precede weight loading"),
    )

    with pytest.raises(ValueError, match="source report differs"):
        load_training_source_model(
            session.run,
            report,
            source,
            _entry(),
            training_api_limits(),
            build_network=lambda *_: pytest.fail(
                "report mismatch reached model construction"
            ),
            adapt_network=lambda *_args, **_kwargs: pytest.fail(
                "report mismatch reached adaptation"
            ),
        )
