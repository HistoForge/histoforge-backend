"""CPU evidence across real dense admission, worker, publication and installation.

Only the expensive architecture constructor is substituted. This qualifies the
ownership contracts, not DINOv3 numerics, GPU memory or research-model quality.
"""

import copy
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
import torch
from histoforge._vendor.histoforge_models.pathology import common
from histoforge.cloud.architectures.semantic.net import build as build_semantic
from histoforge.cloud.architectures.semantic.runtime import (
    create_semantic_runtime,
)
from histoforge.cloud.modules.identity import module_content_identity
from histoforge.cloud.modules.store import ModuleStore
from histoforge.cloud.modules.training_evidence import validate_module_training_report
from histoforge.cloud.rasterization.rasterizer import RasterizedImage
from histoforge.cloud.training.control_records import TrainingControlRecords
from histoforge.cloud.training.runs.control import TrainingRunControl
from histoforge.cloud.training.runs.executor import train_and_publish_claimed_run
from histoforge.cloud.training.runs.files import TrainingRunFiles
from histoforge.cloud.training.runs.finalizer import (
    TrainingRunFinalizer,
    finalize_training_result,
)
from histoforge.cloud.training.runs.report import select_training_weight_epoch
from histoforge.cloud.training.runs.result_files import TrainingResultFiles
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from histoforge.cloud.training.runs.source_preparation import TrainingSourcePreparation
from histoforge.cloud.training.runs.state import TrainingRunState
from safetensors.torch import load_file, save
from tests.test_training_batches import single_thread_torch as single_thread_torch
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_state import service_limits
from tests.test_training_field_reader import _source_limits
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_reconciliation import NOW, _ControlClient
from tests.test_training_source_preparation import (
    RUN_ID,
    install_module,
    workflow_recipe,
)
from tests.training_class_fixtures import TinyClassAdapter, initialized_adapter
from tests.training_dense_fixtures import dense_setup


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("mode", ["use", "add", "define"])
@pytest.mark.parametrize(
    "result_form", ["selected_model_only", "complete_source_workflow"]
)
def test_dense_capture_to_installed_weights_and_fresh_inference(
    tmp_path, monkeypatch, fixed, mode, result_form, single_thread_torch
):
    setup = dense_setup(tmp_path / "dataset", fixed=fixed)
    limits = training_api_limits()
    torch.manual_seed(33)
    source_count = 2 if mode == "add" else 4
    starting = initialized_adapter(source_count)
    recipe = copy.deepcopy(setup.recipe)
    source_model = recipe["models"]["nuclei"]
    source_model["output"]["classChannels"] = source_model["output"]["classChannels"][
        :source_count
    ]
    source_model["params"]["num_classes"] = source_count
    source_model["input"]["micronsPerPixel"] = 0.22
    workflow = workflow_recipe()
    recipe["models"]["cells"] = workflow["models"]["cells"]
    recipe["steps"].extend(workflow["steps"][1:])
    store = install_module(
        tmp_path / "source",
        recipe,
        vendored_weight_bytes={"nuclei": save(starting.state_dict())},
    )
    request = {
        **setup.request,
        "class_initialization": mode,
        "result_form": result_form,
        "source_module": module_content_identity(
            store.current.read_record("owner/workflow")
        ),
    }
    root = tmp_path / "artifacts"
    records = TrainingControlRecords(root, limits, service_limits())
    records.reserve_dataset(
        {
            "type": "retained",
            "dataset_id": request["dataset_id"],
            "dataset_digest": request["dataset_digest"],
            "retained_bytes": setup.dataset.retained_bytes,
            "task_display_name": "Dense fixture",
            "image_count": 1,
            "roi_count": 3,
            "created_at": "2026-09-04T11:00:00.000Z",
        }
    )
    state = TrainingRunState(
        records,
        limits,
        service_limits(),
        clock=lambda: NOW,
        id_factory=lambda: UUID(RUN_ID),
    )
    run_files = TrainingRunFiles(root, limits)
    source_files = TrainingSourceFiles(root, 8 * 1024**2)
    result_files = TrainingResultFiles(root, limits, 8 * 1024**2)
    control = TrainingRunControl(state, run_files, source_files, limits)
    client = _ControlClient(control)
    assert control.reserve(request)["admission"]["run_id"] == RUN_ID
    source_preparation = TrainingSourcePreparation(
        lambda: nullcontext(store.current), source_files, limits
    )
    with source_preparation.prepare(request) as source:
        prepared = setup.preparation(
            source.request, source.recipe, source.starting_weight_digest
        )
        initial = prepared.initial_report(RUN_ID)
        digest = run_files.publish_report_candidate(initial)["report_digest"]
        source.publish(RUN_ID, digest)
    control.accept(request, trainer_build(), 2, 3600, digest)
    control.attach_worker(RUN_ID, "fc-dense-fixture")
    run = control.claim_worker(RUN_ID, trainer_build())

    constructed = []
    monkeypatch.setattr(common, "DinoV3AdapterBackbone", TinyClassAdapter)

    def build(parameters, weights):
        assert parameters["num_classes"] == (source_count if not constructed else 4)
        assert parameters["batch_size"] == 2
        assert "min_instance_pixels" not in parameters
        network = build_semantic(parameters, weights)
        constructed.append(network)
        return network

    monkeypatch.setattr("histoforge.cloud.architectures.semantic.net.build", build)
    candidate = train_and_publish_claimed_run(
        run,
        setup.entry,
        dataset_files=SimpleNamespace(read_dataset=lambda *_: setup.dataset),
        source_files=source_files,
        run_files=run_files,
        result_files=result_files,
        control=client,
        device="cpu",
        api_limits=limits,
        source_limits=_source_limits(),
    )
    assert len(constructed) == 1
    assert control.get(RUN_ID)["state"] == "RUNNING"
    assert control.get(RUN_ID)["progress"]["completed_epochs"] == 2
    report = candidate.report
    assert report["class_initialization"] == mode
    assert report["source_model_output"] == source_model["output"]
    assert report["source_model_microns_per_pixel"] == 0.22
    assert report["task"]["model_microns_per_pixel"] == 0.5
    assert len(report["task"]["model_output"]["classChannels"]) == 4
    assert (
        report["model_selection"]["epoch"]
        == select_training_weight_epoch(
            report["metric_policy"],
            report["epochs"],
            validation_enabled=True,
            total_epochs=report["total_epochs"],
        )["epoch"]
    )
    assert "min_instance_pixels" not in report
    expected_state = {
        name: value.clone() for name, value in constructed[0].state_dict().items()
    }
    for name in ("out.weight", "out.bias", "W2"):
        torch.testing.assert_close(
            expected_state[name], starting.state_dict()[name], rtol=0, atol=0
        )
    assert not torch.equal(
        expected_state["out_class.weight"], starting.out_class.weight
    )

    result_store = ModuleStore(
        tmp_path / "results",
        max_module_bytes=10_000_000,
        max_installed_bytes=20_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
        training_api_limits=limits,
    )
    finalize = partial(
        finalize_training_result,
        result_store,
        run_files,
        result_files,
        client,
        publish_volume=lambda: None,
        api_limits=limits,
    )
    finalizer = TrainingRunFinalizer(
        run_files, result_files, source_files, client, finalize, limits
    )
    succeeded = finalizer.reconcile_candidate(RUN_ID)
    assert succeeded["state"] == "SUCCEEDED"
    assert finalizer.reconcile_candidate(RUN_ID) == succeeded
    assert result_files.inspect_candidate(RUN_ID) is None
    module_ref = f"workspace/run-{RUN_ID}"
    record = result_store.current.revalidate(module_ref)
    assert record["training_result_attestation"] == candidate.attestation
    assert result_store.current.load_training_report(module_ref) == report
    installed = result_store.current.load_recipe(module_ref)
    assert set(installed["models"]) == (
        {"nuclei", "cells"} if result_form == "complete_source_workflow" else {"nuclei"}
    )
    assert installed["models"]["nuclei"]["params"]["batch_size"] == 2
    assert installed["models"]["nuclei"]["params"]["num_classes"] == 4
    assert installed["models"]["nuclei"]["input"]["micronsPerPixel"] == 0.5
    assert installed["models"]["nuclei"]["output"] == report["task"]["model_output"]
    assert "min_instance_pixels" not in installed["models"]["nuclei"]["params"]
    module_dir = result_store.current.module_dir(module_ref)
    weights = load_file(
        str(module_dir / installed["models"]["nuclei"]["weights"]["file"])
    )
    fresh = build(installed["models"]["nuclei"]["params"], weights)
    for name, value in expected_state.items():
        torch.testing.assert_close(fresh.state_dict()[name], value, rtol=0, atol=0)
    image = RasterizedImage(
        np.arange(20 * 20 * 3, dtype=np.uint8).reshape(20, 20, 3),
        np.ones((20, 20), bool),
    )
    reference = create_semantic_runtime(
        installed["models"]["nuclei"], constructed[0]
    ).predict(image)
    reloaded = create_semantic_runtime(installed["models"]["nuclei"], fresh).predict(
        image
    )
    assert set(reloaded) == set(reference) == {"class_map"}
    np.testing.assert_array_equal(reloaded["class_map"], reference["class_map"])

    # Imported module evidence must reject forbidden presence, not compare None == None.
    for illegal in (None, 0, 15):
        changed = copy.deepcopy(installed)
        changed["models"]["nuclei"]["params"]["min_instance_pixels"] = illegal
        with pytest.raises(ValueError, match="dense classification forbids"):
            validate_module_training_report(
                module_dir, changed, record["resolved_artifacts"], limits
            )
