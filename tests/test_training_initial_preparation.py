"""Managed policy resolves immutable class weights from retained TRAIN data."""

import copy

import numpy as np
import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes, parse_json
from histoforge.cloud.training.bindings.dinov3_instance import (
    initial_managed_trainer_profile,
)
from histoforge.cloud.training.component_schema import (
    CLASS_WEIGHT_POLICY_ID,
)
from histoforge.cloud.training.datasets import class_weights
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from tests.test_training_capabilities import _profile, _variable_capability
from tests.test_training_class_weights import _object
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import _original_dataset, _rectangle
from tests.test_training_field_reader import _origin, _retain
from tests.test_training_run_preparation import (
    DATASET_ID,
    RUN_ID,
    STARTING_WEIGHT_DIGEST,
    _preparation,
    _recipe,
    _request,
    _retained_dataset,
)


def _classified_output():
    return {
        "type": "instance_segmentation_and_classification",
        "classChannels": [
            {"channel": 0, "role": "background"},
            {"channel": 1, "role": "emitted", "pathClass": "Cell"},
            {"channel": 2, "role": "emitted", "pathClass": "Sertoli cell"},
            {"channel": 3, "role": "training_only", "pathClass": "Other"},
        ],
    }


@pytest.mark.parametrize("context", [False, True])
@pytest.mark.parametrize("flow_weighting", ["pixel", "equal_instance"])
def test_initial_report_freezes_train_pixel_populations_not_object_or_field_counts(
    tmp_path, monkeypatch, context, flow_weighting
):
    capability = _variable_capability()
    capability["output_type"] = "instance_segmentation_and_classification"
    model_output = _classified_output()
    files, record = _retained_classified_dataset(tmp_path, capability, model_output)
    profile = _initial_profile(capability)
    profile_before = copy.deepcopy(profile)
    request = _request(record["dataset_digest"], capability)
    request["trainer_settings"]["flow_loss_weighting"] = flow_weighting
    request["field_profile"] = {
        **request["field_profile"],
        "smallRoiContext": context,
    }
    request["split_plan"] = {
        "type": "split",
        "unit": "roi",
        "assignment_source": "custom",
        "units": [
            {"unit_index": 0, "partition": "training"},
            {"unit_index": 1, "partition": "validation"},
        ],
    }
    recipe = _recipe()
    recipe["models"]["cells"]["output"] = model_output
    recipe["models"]["cells"]["params"]["num_classes"] = 4
    count = class_weights.count_training_class_pixels
    counted_indices = []

    def count_actual_train_pixels(dataset, indices, **kwargs):
        counted_indices.append(tuple(indices))
        return count(dataset, indices, **kwargs)

    monkeypatch.setattr(
        class_weights,
        "count_training_class_pixels",
        count_actual_train_pixels,
    )
    prepared = _preparation(files, record, capability, [], profile)(
        request, recipe, STARTING_WEIGHT_DIGEST
    )
    report = prepared.initial_report(RUN_ID)
    objective = report["trainer_configuration"]["objective"]["parameters"]
    weighting = objective["class_weighting"]

    # At model MPP 0.5 this original's TRAIN ROI is 10 x 10: Cell occupies
    # 16 pixels and Other four. Neither validation's 25 Other pixels nor real
    # context nor the minimum 256 x 256 input padding enters these populations.
    assert counted_indices == [(0,)]
    assert objective["class_count"] == 4
    assert "class_count" not in profile["trainer_components"]["objective"]["parameters"]
    assert weighting["policy"] == CLASS_WEIGHT_POLICY_ID
    expected_weights = np.sqrt(np.array([10 / 80, 10 / 16, 10, 10 / 4]))
    assert weighting["channels"] == [
        {
            "channel": 0,
            "role": "background",
            "raw_count": 80,
            "weight": expected_weights.astype(np.float32).tolist()[0],
        },
        {
            "channel": 1,
            "role": "emitted",
            "path_class": "Cell",
            "raw_count": 16,
            "weight": expected_weights.astype(np.float32).tolist()[1],
        },
        {
            "channel": 2,
            "role": "emitted",
            "path_class": "Sertoli cell",
            "raw_count": 0,
            "weight": expected_weights.astype(np.float32).tolist()[2],
        },
        {
            "channel": 3,
            "role": "training_only",
            "path_class": "Other",
            "raw_count": 4,
            "weight": expected_weights.astype(np.float32).tolist()[3],
        },
    ]
    assert objective["weight_tversky_by_class"] is True
    assert objective["flow_loss_weighting"] == flow_weighting
    assert report["trainer_settings"]["flow_loss_weighting"] == flow_weighting
    assert report["training_counts"] == {
        "image_count": 1,
        "roi_count": 1,
        "negative_roi_count": 0,
        "target_object_counts": {"Cell": 1, "Sertoli cell": 0, "Other": 1},
    }
    assert report["validation_counts"]["target_object_counts"] == {
        "Cell": 0,
        "Sertoli cell": 0,
        "Other": 1,
    }
    assert report["dataset_image_count"] == 1
    assert report["field_profile"] == request["field_profile"]
    assert prepared.total_epochs == report["total_epochs"] == 80
    assert profile == profile_before  # Resolution never mutates registry input.
    class_series = report["metric_policy"]["series"][4:]
    assert [item["series_id"] for item in class_series] == [
        "validation-object-f1.channel-1.v1",
        "validation-object-f1.channel-2.v1",
    ]
    assert [item["display_name"] for item in class_series] == [
        "Validation object F1: Cell",
        "Validation object F1: Sertoli cell",
    ]
    assert len(profile["metric_policy"]["series"]) == 4

    frozen = canonical_bytes(report)
    decoded = parse_json(frozen)
    assert decoded["trainer_configuration"]["objective"]["parameters"] == objective
    assert all(type(item["weight"]) is float for item in weighting["channels"])
    profile["trainer_components"]["objective"]["parameters"]["class_weighting"] = {
        "policy": "none"
    }
    profile["trainer_execution_seconds"] = 1
    profile["metric_policy"]["series"].clear()
    request["field_profile"]["smallRoiContext"] = not context
    weighting["channels"][0]["raw_count"] = 0
    weighting["channels"][0]["weight"] = 99.0
    assert canonical_bytes(prepared.initial_report(RUN_ID)) == frozen


def test_segmentation_only_initial_policy_does_not_count_or_resolve_ce_weights(
    tmp_path, monkeypatch
):
    files, record, capability = _retained_dataset(tmp_path, _variable_capability())
    profile = _initial_profile(capability)

    def unexpected_count(*_args, **_kwargs):
        pytest.fail("segmentation-only preparation must not count CE populations")

    monkeypatch.setattr(class_weights, "count_training_class_pixels", unexpected_count)
    prepared = _preparation(files, record, capability, [], profile)(
        _request(record["dataset_digest"], capability),
        _recipe(),
        STARTING_WEIGHT_DIGEST,
    )
    report = prepared.initial_report(RUN_ID)
    objective = report["trainer_configuration"]["objective"]["parameters"]

    assert objective["class_weighting"] == {"policy": "none"}
    assert objective["class_count"] == 0
    assert objective["use_uncertainty_weighting"] is False
    assert objective["weight_tversky_by_class"] is False
    assert report["training_counts"]["target_object_counts"] == {"Cell": 1}
    assert "validation_counts" not in report
    assert report["total_epochs"] == 80


def _initial_profile(capability):
    fixture = _profile()
    return initial_managed_trainer_profile(
        "dinov3-adapter-cellpose",
        capability,
        augmentation_profiles=fixture["augmentation_profiles"],
        gpu_resources=fixture["gpu_resources"],
        trainer_execution_seconds=fixture["trainer_execution_seconds"],
    )


def _retained_classified_dataset(
    tmp_path, capability, model_output, *, validation_class="Other"
):
    extracted = _original_dataset(
        tmp_path / "source",
        resolution_level=0,
        capability=capability,
        model_output=model_output,
    )
    validation_roi = copy.deepcopy(extracted.manifest["rois"][0])
    validation_roi["source_roi_uuid"] = "22222222-2222-4222-8222-222222222222"
    extracted.manifest["rois"].append(validation_roi)
    rois = [
        (
            _rectangle(10, 10, 30, 30),
            [
                _object(1, _rectangle(12, 12, 20, 20), "Cell"),
                _object(2, _rectangle(24, 24, 28, 28), "Other"),
            ],
        ),
        (
            _rectangle(40, 40, 50, 50),
            [_object(3, _rectangle(40, 40, 50, 50), validation_class)],
        ),
    ]
    for index, (geometry, objects) in enumerate(rois):
        directory = extracted.root / f"rois/{index}"
        directory.mkdir(exist_ok=True)
        for name, value in (
            (
                "roi.geojson",
                {
                    "type": "Feature",
                    "geometry": canonical_geometry(geometry),
                    "properties": {},
                },
            ),
            ("objects.geojson", {"type": "FeatureCollection", "features": objects}),
        ):
            (directory / name).write_bytes(canonical_bytes(value))
            if index:
                extracted.manifest["files"].append(
                    {
                        "path": f"rois/{index}/{name}",
                        "size_bytes": 1,
                        "sha256": "0" * 64,
                    }
                )
    retained = _retain(tmp_path, extracted, _origin(extracted), capability=capability)
    files = TrainingDatasetFiles(tmp_path / "artifact-volume")
    files.publish_dataset(
        DATASET_ID, retained.root, retained.manifest, training_api_limits()
    )
    return files, {
        "type": "retained",
        "dataset_id": DATASET_ID,
        "dataset_digest": retained.manifest["dataset_digest"],
        "retained_bytes": retained.retained_bytes,
    }
