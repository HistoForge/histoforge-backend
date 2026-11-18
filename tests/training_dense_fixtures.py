"""Synthetic dense capture and real admission evidence; no research data."""

from dataclasses import replace
from types import SimpleNamespace

from histoforge.cloud.architectures.registry import FIELD_AGGREGATION_ID
from histoforge.cloud.training.initial_capability import semantic_dense_entry
from histoforge.cloud.training.runs.preparation import TrainingRunPreparation
from tests.test_training_batches import _profile
from tests.test_training_capabilities import _registry
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _rectangle
from tests.test_training_field_reader import _source_limits
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request
from tests.test_training_validation import _capture, _object


def dense_output():
    return {
        "type": "dense_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Leydig cell"},
            {"channel": 2, "role": "training_only", "pathClass": "Other"},
            {"channel": 3, "role": "emitted", "pathClass": "Sertoli cell"},
        ],
    }


def dense_setup(tmp_path, *, fixed=False, validation_rois=2, objects=None):
    entry = semantic_dense_entry(trainer_execution_seconds=3600)
    profile = {**_profile(), "aggregation": FIELD_AGGREGATION_ID}
    if fixed:
        profile = {
            "planner": "fixed-field-v1",
            "side_model_px": 64,
            "alignment_model_px": 16,
            "small_roi_context": False,
            "tile_overlap": 0.1,
            "aggregation": FIELD_AGGREGATION_ID,
            "preprocessing": profile["preprocessing"],
        }
    capability = {
        **entry.capability,
        "default_field_profile": profile,
        "allowed_field_side_model_px": {"minimum": 64 if fixed else 16, "maximum": 64},
        "small_roi_context_editable": not fixed,
    }
    entry = replace(entry, capability=capability)
    if objects is None:
        other = _object(2, _rectangle(22, 22, 28, 28))
        other["properties"]["path_class"] = "Other"
        objects = [_object(1, _rectangle(12, 12, 20, 20)), other]
    dataset, split, _ = _capture(
        tmp_path,
        capability=capability,
        model_output=dense_output(),
        validation_rois=validation_rois,
        objects=objects,
    )
    request = run_request()
    request.pop("min_instance_pixels")
    if validation_rois == 0:
        split = {"type": "none"}
    request.update(
        dataset_digest=dataset.manifest["dataset_digest"],
        split_plan=split,
        field_profile=profile,
        selected_model_id="nuclei",
        augmentation={"profile_id": "he-morphology-v1", "enabled_operator_ids": []},
        trainer_settings={
            **capability["default_training_settings"],
            "total_epochs": 2,
            "warmup_mode": "custom",
            "warmup_epochs": 0,
            "data_loading_workers": 0,
        },
    )
    model = {
        "arch": "dinov3-adapter-semantic",
        "gpu": "L40S",
        "weights": {"type": "vendored", "file": "nuclei.safetensors"},
        "input": {"type": "field", "micronsPerPixel": 0.5},
        "output": dense_output(),
        "params": {"num_classes": 4, "batch_size": 2},
    }
    if fixed:
        model["params"].update(bsize=64, tile_overlap=0.1)
    else:
        model["input"]["fieldProfile"] = profile
    recipe = {
        "schemaVersion": 3,
        "name": "Dense fixture",
        "models": {"nuclei": model},
        "steps": [{"model": "nuclei"}],
    }
    preparation = TrainingRunPreparation(
        SimpleNamespace(read_dataset=lambda *_: dataset),
        SimpleNamespace(
            get=lambda _: {
                "record": {
                    "type": "retained",
                    "dataset_digest": dataset.manifest["dataset_digest"],
                    "retained_bytes": dataset.retained_bytes,
                }
            }
        ),
        _registry(entry),
        lambda _: trainer_build(),
        training_api_limits(),
        _source_limits(),
    )
    prepared = preparation(request, recipe, "a" * 64)
    return SimpleNamespace(
        entry=entry,
        dataset=dataset,
        request=request,
        recipe=recipe,
        preparation=preparation,
        report=prepared.initial_report("44444444-4444-4444-8444-444444444444"),
    )
