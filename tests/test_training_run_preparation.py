from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.capabilities import (
    TrainingCapabilityEntry,
    TrainingCapabilityRegistry,
)
from histoforge.cloud.training.datasets import content
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.implementations import DINOV3_INSTANCE_TRAINER_ID
from histoforge.cloud.training.runs.preparation import TrainingRunPreparation
from histoforge.cloud.training.runtime_identity import trainer_behavior_policy
from tests.test_training_augmentation import _apply, _assert_equal, _configured
from tests.test_training_batches import _field
from tests.test_training_capabilities import (
    _capability,
    _model,
    _profile,
    _variable_capability,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import (
    _dataset,
    _object_feature,
    _rectangle,
)
from tests.test_training_dataset_state import service_limits
from tests.test_training_run_contract import trainer_build
from tests.test_training_run_request import run_request

DATASET_ID = "33333333-3333-4333-8333-333333333333"
RUN_ID = "44444444-4444-4444-8444-444444444444"
STARTING_WEIGHT_DIGEST = "f" * 64


class _DatasetControl:
    def __init__(self, record: dict) -> None:
        self.record = record

    def get(self, dataset_id: str) -> dict:
        assert dataset_id == DATASET_ID
        return {"record": self.record, "in_use": False}


def _retained_dataset(
    tmp_path: Path, capability: dict | None = None
) -> tuple[TrainingDatasetFiles, dict, dict]:
    capability = _capability() if capability is None else capability
    label = _object_feature(
        "11111111-1111-4111-8111-111111111111",
        _rectangle(12, 12, 20, 20),
    )
    label["properties"]["path_class"] = "Cell"
    extracted = _dataset(
        tmp_path / "prepared-dataset",
        objects=[label],
        capability=capability,
        model_output=_model()["output"],
    )
    files = TrainingDatasetFiles(tmp_path / "artifact-volume")
    files.publish_dataset(
        DATASET_ID,
        extracted.root,
        extracted.manifest,
        training_api_limits(),
    )
    record = {
        "type": "retained",
        "dataset_id": DATASET_ID,
        "dataset_digest": extracted.manifest["dataset_digest"],
        "retained_bytes": extracted.retained_bytes,
    }
    return files, record, capability


def _recipe() -> dict:
    model = _model()
    model["input"]["micronsPerPixel"] = 0.5
    model["weights"]["file"] = "cells.safetensors"
    return {
        "schemaVersion": 3,
        "name": "Fixture module",
        "models": {"cells": model},
        "steps": [{"model": "cells"}],
    }


def _request(dataset_digest: str, capability: dict) -> dict:
    request = run_request()
    request.update(
        {
            "dataset_id": DATASET_ID,
            "dataset_digest": dataset_digest,
            "selected_model_id": "cells",
            "result_form": "selected_model_only",
            "split_plan": {"type": "none"},
            "field_profile": copy.deepcopy(capability["default_field_profile"]),
            "augmentation": {
                "profile_id": "none-v1",
                "enabled_operator_ids": [],
            },
            "trainer_settings": copy.deepcopy(capability["default_training_settings"]),
        }
    )
    return request


def _reportable_profile(capability: dict, profile: dict | None = None) -> dict:
    """Build valid persisted component evidence around test-owned dynamic values."""

    supplied = _profile() if profile is None else profile
    return initial_managed_trainer_profile(
        "dinov3-adapter-cellpose",
        capability,
        augmentation_profiles=supplied["augmentation_profiles"],
        gpu_resources=supplied["gpu_resources"],
        trainer_execution_seconds=supplied["trainer_execution_seconds"],
    )


def _preparation(
    files: TrainingDatasetFiles,
    record: dict,
    capability: dict,
    policies: list[dict],
    profile: dict | None = None,
) -> TrainingRunPreparation:
    reportable_profile = _reportable_profile(capability, profile)
    registry = TrainingCapabilityRegistry(
        [
            TrainingCapabilityEntry(
                "dinov3-adapter-cellpose",
                capability,
                reportable_profile,
                DINOV3_INSTANCE_TRAINER_ID,
            )
        ],
        api_limits=training_api_limits(),
        service_limits=service_limits(),
    )

    def build(entry: TrainingCapabilityEntry) -> dict:
        policies.append(trainer_behavior_policy(entry))
        return trainer_build()

    return TrainingRunPreparation(
        files,
        _DatasetControl(record),
        registry,
        build,
        training_api_limits(),
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
    )


def test_preparation_binds_retained_dataset_capability_split_and_report(
    tmp_path: Path,
) -> None:
    files, record, capability = _retained_dataset(tmp_path)
    policies: list[dict] = []
    request = _request(record["dataset_digest"], capability)

    prepared = _preparation(files, record, capability, policies)(
        request,
        _recipe(),
        STARTING_WEIGHT_DIGEST,
    )
    report = prepared.initial_report(RUN_ID)

    assert prepared.request == request
    assert prepared.starting_weight_digest == STARTING_WEIGHT_DIGEST
    assert prepared.trainer_build == trainer_build()
    assert prepared.total_epochs == 80
    assert prepared.trainer_execution_seconds == 3600
    assert prepared.worker_resources == {
        "gpu": "L4",
        "cpu": 4.0,
        "memory_mib": 16384,
        "max_validation_assignment_matrix_entries": 1_000_000,
    }
    assert policies == [
        {
            "schema_version": 1,
            "trainer_implementation_id": DINOV3_INSTANCE_TRAINER_ID,
            "architecture": "dinov3-adapter-cellpose",
            "capability": capability,
            "trainer_profile": _reportable_profile(capability),
        }
    ]
    assert report["run_id"] == RUN_ID
    assert report["state"] == "QUEUED"
    assert report["dataset_image_count"] == 1
    assert report["training_counts"] == {
        "image_count": 1,
        "roi_count": 1,
        "negative_roi_count": 0,
        "target_object_counts": {"Cell": 1},
    }
    assert "validation_counts" not in report


@pytest.mark.parametrize("read_failure", [False, True])
def test_preparation_keeps_retained_admission_reads_inside_the_volume_gate(
    tmp_path: Path, monkeypatch, read_failure: bool
) -> None:
    files, record, capability = _retained_dataset(tmp_path)
    reading = Event()
    finish_reading = Event()
    reload_attempted = Event()
    reloaded = Event()
    original_inspect = content._inspect_artifact

    def inspect(*args):
        reading.set()
        assert finish_reading.wait(5)
        assert not reloaded.is_set()
        result = original_inspect(*args)
        if read_failure:
            raise OSError("admission source read failed")
        return result

    monkeypatch.setattr(content, "_inspect_artifact", inspect)
    artifacts = ArtifactFileStore(
        tmp_path / "artifact-volume", reload_volume=reloaded.set
    )

    def reload_other_store():
        reload_attempted.set()
        return artifacts.artifact_ids()

    with ThreadPoolExecutor(max_workers=2) as executor:
        prepared = executor.submit(
            _preparation(files, record, capability, []),
            _request(record["dataset_digest"], capability),
            _recipe(),
            STARTING_WEIGHT_DIGEST,
        )
        try:
            assert reading.wait(5)
            following = executor.submit(reload_other_store)
            assert reload_attempted.wait(5)
            assert not reloaded.wait(0.05)
        finally:
            finish_reading.set()
        if read_failure:
            with pytest.raises(OSError, match="admission source read failed"):
                prepared.result(timeout=5)
        else:
            assert (
                prepared.result(timeout=5).initial_report(RUN_ID)["state"] == "QUEUED"
            )
        assert following.result(timeout=5) == set()
    assert reloaded.is_set()


def test_preparation_rejects_stale_dataset_identity_before_reading_bytes(
    tmp_path: Path,
) -> None:
    files, record, capability = _retained_dataset(tmp_path)
    stale = copy.deepcopy(record)
    stale["dataset_digest"] = "e" * 64

    with pytest.raises(GatewayFailure) as raised:
        _preparation(files, stale, capability, [])(
            _request(record["dataset_digest"], capability),
            _recipe(),
            STARTING_WEIGHT_DIGEST,
        )

    assert raised.value.error["code"] == "TRAINING_DATASET_NOT_FOUND"


@pytest.mark.parametrize("variable_source", [False, True])
def test_preparation_freezes_edited_variable_fields_from_either_source_planner(
    tmp_path: Path, variable_source: bool
) -> None:
    files, record, capability = _retained_dataset(tmp_path, _variable_capability())
    recipe = _recipe()
    if variable_source:
        source = recipe["models"]["cells"]
        source["params"].pop("bsize")
        source["input"]["fieldProfile"] = {
            **capability["default_field_profile"],
            "minSideModelPx": 128,
            "maxSideModelPx": 2048,
        }
    original = copy.deepcopy(recipe)
    request = _request(record["dataset_digest"], capability)
    request["field_profile"] = {
        **capability["default_field_profile"],
        "minSideModelPx": 384,
        "preferredTileSideModelPx": 768,
        "maxSideModelPx": 768,
        "smallRoiContext": False,
    }
    prepared = _preparation(files, record, capability, [])(
        request, recipe, STARTING_WEIGHT_DIGEST
    )

    assert prepared.initial_report(RUN_ID)["field_profile"] == request["field_profile"]
    assert prepared.request == request
    assert recipe == original


def test_preparation_records_source_and_user_selected_target_resolution(
    tmp_path: Path,
) -> None:
    files, record, capability = _retained_dataset(tmp_path)
    recipe = _recipe()
    recipe["models"]["cells"]["input"]["micronsPerPixel"] = 0.22

    original = copy.deepcopy(recipe)
    prepared = _preparation(files, record, capability, [])(
        _request(record["dataset_digest"], capability), recipe, STARTING_WEIGHT_DIGEST
    )
    report = prepared.initial_report(RUN_ID)
    assert report["source_model_microns_per_pixel"] == 0.22
    assert report["task"]["model_microns_per_pixel"] == 0.5
    assert recipe == original


def test_preparation_allows_target_class_rename_and_keeps_source_evidence(
    tmp_path: Path,
) -> None:
    files, record, capability = _retained_dataset(tmp_path)
    recipe = _recipe()
    recipe["models"]["cells"]["output"]["classes"] = ["Different cell"]

    original = copy.deepcopy(recipe)
    prepared = _preparation(files, record, capability, [])(
        _request(record["dataset_digest"], capability), recipe, STARTING_WEIGHT_DIGEST
    )
    report = prepared.initial_report(RUN_ID)
    assert report["class_initialization"] == "use"
    assert report["source_model_output"]["classes"] == ["Different cell"]
    assert report["task"]["model_output"]["classes"] == ["Cell"]
    assert report["task"]["annotation_class_map"] == {"Cell": "Cell"}
    assert report["training_counts"]["target_object_counts"] == {"Cell": 1}
    assert recipe == original


def test_preparation_fails_closed_when_retained_bytes_are_missing(
    tmp_path: Path,
) -> None:
    files, record, capability = _retained_dataset(tmp_path)
    files.delete_dataset(DATASET_ID)

    with pytest.raises(GatewayFailure) as raised:
        _preparation(files, record, capability, [])(
            _request(record["dataset_digest"], capability),
            _recipe(),
            STARTING_WEIGHT_DIGEST,
        )

    assert raised.value.error["code"] == "SERVICE_UNAVAILABLE"


@pytest.mark.parametrize("enabled", [[], ["flip-v1"], ["rotate-v1", "flip-v1"]])
def test_preparation_resolves_a_custom_preset_into_executable_report_evidence(
    tmp_path, enabled
):
    files, record, _ = _retained_dataset(tmp_path)
    capability, profile = _configured()
    request = _request(record["dataset_digest"], capability)
    request["augmentation"] = {
        "profile_id": "spatial-fixture-v1",
        "enabled_operator_ids": enabled,
    }
    prepared = _preparation(files, record, capability, [], profile)(
        request, _recipe(), STARTING_WEIGHT_DIGEST
    )
    report = prepared.initial_report(RUN_ID)
    component = report["trainer_configuration"]["augmentation"]
    assert report["augmentation"] == request["augmentation"]
    assert set(component["parameters"]["operator_parameters"]) == set(enabled)
    field = _field()
    before = _apply(field, component, seed=request["training_seed"])
    profile["augmentation_profiles"].clear()
    after = _apply(
        field,
        prepared.initial_report(RUN_ID)["trainer_configuration"]["augmentation"],
        seed=request["training_seed"],
    )
    _assert_equal(before, after)
