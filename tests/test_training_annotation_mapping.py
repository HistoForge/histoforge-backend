"""Raw annotation names remain immutable; mapped targets own counts and rasters."""

import copy
from dataclasses import replace

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.training.contract import validate_training_task
from histoforge.cloud.training.datasets.io import read_training_roi_labels
from histoforge.cloud.training.datasets.manifest import build_training_dataset_manifest
from histoforge.cloud.training.datasets.rasters import training_raster_transform
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import (
    _admit,
    _dataset,
    _object_feature,
    _rectangle,
)


def _with_task(dataset, **changes):
    preimage = copy.deepcopy(dataset.manifest)
    preimage.pop("dataset_digest")
    preimage["task"].update(changes)
    manifest = build_training_dataset_manifest(preimage, training_api_limits())
    encoded = canonical_bytes(manifest)
    (dataset.root / "dataset.json").write_bytes(encoded)
    return replace(
        dataset,
        manifest=manifest,
        retained_bytes=len(encoded) + sum(f["size_bytes"] for f in manifest["files"]),
    )


def test_many_to_one_mapping_is_applied_once_after_canonical_validation(tmp_path):
    first = _object_feature(
        "11111111-1111-4111-8111-111111111111", _rectangle(12, 12, 16, 16)
    )
    second = _object_feature(
        "22222222-2222-4222-8222-222222222222", _rectangle(20, 20, 24, 24)
    )
    first["properties"]["path_class"] = "Raw A"
    second["properties"]["path_class"] = "Raw B"
    dataset = _with_task(
        _dataset(tmp_path, objects=[first, second]),
        annotation_class_map={"Raw A": "Leydig cell", "Raw B": "Leydig cell"},
    )
    raw_path = dataset.root / "rois/0/objects.geojson"
    original = raw_path.read_bytes()
    admitted = _admit(dataset)
    assert admitted.roi_label_summaries[0].target_object_counts == {"Leydig cell": 2}
    for _ in range(2):
        _, labels, _ = read_training_roi_labels(
            dataset.root, dataset.manifest, 0, 100_000
        )
        assert [f["properties"]["path_class"] for f in labels] == ["Leydig cell"] * 2
    assert raw_path.read_bytes() == original
    raw_path.write_bytes(b" " + original)
    with pytest.raises(ValueError, match="not canonical"):
        _admit(dataset)


def test_background_mapping_is_explicit_absence_and_produces_negative_roi(tmp_path):
    dataset = _with_task(_dataset(tmp_path, objects=[]), annotation_class_map={})
    assert _admit(dataset).roi_label_summaries[0].is_negative
    _, labels, _ = read_training_roi_labels(dataset.root, dataset.manifest, 0, 100_000)
    assert labels == []
    # Capture omits background polygons. A contradictory uploaded label is rejected.
    inconsistent = _with_task(_dataset(tmp_path / "invalid"), annotation_class_map={})
    with pytest.raises(ValueError, match="no mapped target"):
        _admit(inconsistent)


def test_mapping_and_resolution_are_dataset_identity_and_raster_inputs(tmp_path):
    original = _dataset(tmp_path)
    remapped = _with_task(
        original, annotation_class_map={"Other source label": "Leydig cell"}
    )
    resampled = _with_task(original, model_microns_per_pixel=1.0)
    assert (
        len({d.manifest["dataset_digest"] for d in (original, remapped, resampled)})
        == 3
    )
    source = {"calibration": {"mpp_x": 0.5, "mpp_y": 0.5}}
    before = training_raster_transform(
        (10, 10, 30, 30), original.manifest["task"], source
    )
    after = training_raster_transform(
        (10, 10, 30, 30), resampled.manifest["task"], source
    )
    assert before["raster"]["width"] == 20
    assert after["raster"]["width"] == 10
    assert after["raster_to_slide"]["scale_x"] == 2
    invalid = {
        **original.manifest["task"],
        "annotation_class_map": {"Raw": "Not a target"},
    }
    with pytest.raises(ValueError, match="not a training class"):
        validate_training_task(invalid)
