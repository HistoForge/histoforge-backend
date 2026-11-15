"""Retained PNG, OME-BigTIFF, and original sources feed aligned training fields."""

import copy
import hashlib
import json

import cv2
import numpy as np
import pytest
import tifffile
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.artifacts import volume as artifact_volume
from histoforge.cloud.rasterization.rasterizer import rasterize_srgb_uint8
from histoforge.cloud.sources.reader import SourceReader
from histoforge.cloud.sources.views import derive_original_view_transform
from histoforge.cloud.training.datasets.archive import ExtractedTrainingDataset
from histoforge.cloud.training.datasets.content import admit_extracted_training_dataset
from histoforge.cloud.training.datasets.files import TrainingDatasetFiles
from histoforge.cloud.training.datasets.io import open_training_artifact
from histoforge.cloud.training.datasets.manifest import build_training_dataset_manifest
from histoforge.cloud.training.trainers.instance import (
    field_reader as training_field_reader,
)
from histoforge.cloud.training.trainers.instance.field_reader import read_training_field
from tests.test_training_contract import (
    classified_training_model_output,
    training_capability,
)
from tests.test_training_dataset import training_api_limits
from tests.test_training_dataset_content import (
    _dataset,
    _object_feature,
    _original_dataset,
    _rectangle,
)


@pytest.mark.parametrize("head", ["flow", "hovernet"])
def test_derived_offset_view_pairs_exact_raw_rgb_and_complete_class_targets(
    tmp_path, head
):
    capability = training_capability()
    capability["default_field_profile"]["small_roi_context"] = False
    capability["output_type"] = "instance_segmentation_and_classification"
    objects = [
        _object_feature(
            "11111111-1111-4111-8111-111111111111", _rectangle(112, 212, 120, 220)
        ),
        _object_feature(
            "22222222-2222-4222-8222-222222222222", _rectangle(124, 224, 128, 228)
        ),
    ]
    objects[1]["properties"]["path_class"] = "Other"
    extracted = _dataset(
        tmp_path / "capture",
        capability=capability,
        model_output=classified_training_model_output(),
        objects=objects,
    )
    origin = _origin(extracted)
    origin["source"].update(width=512, height=512)
    origin["source"]["calibration"] = {"mpp_x": 0.4, "mpp_y": 0.8}
    origin["source_rect_at_level"].update(x=100, y=200)
    origin["artifact_pixels_to_slide"].update(translate_x=100, translate_y=200)
    extracted.manifest["rois"][0]["source_envelope"]["pixel_rect"] = {
        "x": 4,
        "y": 6,
        "width": 56,
        "height": 54,
    }
    _write_roi(extracted, _rectangle(110, 210, 130, 230))
    pixels = _pattern(64, 64)
    _write_png(extracted.root / "artifacts/0/source.png", pixels)
    dataset = _retain(tmp_path, extracted, origin, capability=capability)
    target = _target(28, 40, 108.1, 208.4, 1.25, 0.625)
    field = {"x": 1, "y": 2, "width": 22, "height": 36}

    actual = _read(dataset, target, field, head=head)
    expected = rasterize_srgb_uint8(
        pixels[6:, 4:],
        {"scale_x": 1, "scale_y": 1, "translate_x": 104, "translate_y": 206},
        target,
        {"min_x": -4, "min_y": -6, "max_x": 60, "max_y": 58},
        field,
    )
    assert np.array_equal(actual.pixels, expected)
    assert actual.pixels.dtype == np.uint8
    assert actual.labels.targets.directional.shape == (2, 36, 22)
    assert set(np.unique(actual.labels.targets.classes)) == {0, 1, 2}
    assert actual.labels.real_pixel_mask.all()
    assert not actual.labels.supervision_mask.all()
    # Context RGB stays available without becoming supervised background.
    assert np.any(actual.pixels[~actual.labels.supervision_mask] != 0)


def test_tiled_ome_reads_partitioned_fields_without_decoding_the_complete_artifact(
    tmp_path, monkeypatch
):
    extracted = _dataset(tmp_path / "capture")
    origin = _origin(extracted)
    origin["source"].update(width=1024, height=1024)
    origin["source_rect_at_level"].update(width=1024, height=1024)
    extracted.manifest["artifacts"][0]["representation"] = "ome_bigtiff"
    extracted.manifest["rois"][0]["source_envelope"]["pixel_rect"].update(
        width=1024, height=1024
    )
    pixels = _pattern(1024, 1024)
    source_path = extracted.root / "artifacts/0/source.ome.tif"
    tifffile.imwrite(
        source_path,
        pixels,
        bigtiff=True,
        ome=True,
        photometric="rgb",
        tile=(512, 512),
        compression="deflate",
        metadata={"axes": "YXS"},
    )
    for item in extracted.manifest["files"]:
        if item["path"] == "artifacts/0/source.png":
            item["path"] = "artifacts/0/source.ome.tif"
    (extracted.root / "artifacts/0/source.png").unlink()
    limits = _source_limits()
    limits["max_complete_decode_pixels"] = 512 * 512
    dataset = _retain(tmp_path, extracted, origin, source_limits=limits)
    reads = []
    ordinary_read = SourceReader.read_rgb_view

    def record_read(reader, series, level, rectangle):
        reads.append(dict(rectangle))
        return ordinary_read(reader, series, level, rectangle)

    monkeypatch.setattr(SourceReader, "read_rgb_view", record_read)
    actual = _read(dataset, _target(768, 512), source_limits=limits)
    assert np.array_equal(actual.pixels, pixels[:512, :768])
    assert len(reads) > 1
    assert all(rect["width"] * rect["height"] <= 512 * 512 for rect in reads)


def test_original_training_read_uses_the_pinned_selected_pyramid_level(tmp_path):
    extracted = _original_dataset(tmp_path / "capture", resolution_level=1)
    origin = _origin(extracted)
    dataset = _retain(tmp_path, extracted, origin)
    calibration = origin["source"]["calibration"]
    target = _target(
        12, 11, 9.25, 9.75, 0.5 / calibration["mpp_x"], 0.5 / calibration["mpp_y"]
    )
    envelope = dataset.manifest["rois"][0]["source_envelope"]
    rectangle = envelope["pixel_rect"]
    with open_training_artifact(
        dataset.root, dataset.manifest, 0, 8 * 1024**2, _source_limits()
    ) as (_, reader):
        mapping = reader.selected_level_pixels_to_level_zero(0, 1)
        assert mapping["scale_x"] != 1
        source = reader.read_rgb_view(0, 1, rectangle)
        transform = derive_original_view_transform(
            origin["level_zero_pixels_to_slide"],
            mapping,
            {"width": rectangle["width"], "height": rectangle["height"]},
            rectangle,
        )
    expected = rasterize_srgb_uint8(
        source,
        transform,
        target,
        {
            "min_x": 0,
            "min_y": 0,
            "max_x": rectangle["width"],
            "max_y": rectangle["height"],
        },
    )
    actual = _read(dataset, target)
    assert np.array_equal(actual.pixels, expected)
    assert actual.labels.real_pixel_mask.all()
    assert actual.labels.supervision_mask.any()


def test_fractional_coarse_roi_accepts_the_extension_planners_covering_envelope(
    tmp_path,
):
    """Match TrainingSourcePlannerTest's level-zero, two-pixel sampling case."""
    capability = training_capability()
    capability["default_field_profile"]["small_roi_context"] = False
    extracted = _dataset(tmp_path / "capture", capability=capability)
    origin = _origin(extracted)
    origin["source"].update(width=100, height=80)
    origin["source_rect_at_level"] = {"x": 10, "y": 10, "width": 21, "height": 21}
    origin["artifact_pixels_to_slide"].update(translate_x=10, translate_y=10)
    extracted.manifest["task"]["model_microns_per_pixel"] = 1.0
    extracted.manifest["rois"][0]["source_envelope"]["pixel_rect"] = {
        "x": 0,
        "y": 0,
        "width": 21,
        "height": 21,
    }
    _write_roi(extracted, _rectangle(10.6, 10.6, 30.6, 30.6))
    pixels = _pattern(21, 21)
    _write_png(extracted.root / "artifacts/0/source.png", pixels)
    dataset = _retain(tmp_path, extracted, origin, capability=capability)
    target = _target(10, 10, 10.6, 10.6, 2.0, 2.0)
    target["raster"]["target_mpp"] = 1.0
    actual = _read(dataset, target)
    expected = rasterize_srgb_uint8(
        pixels,
        origin["artifact_pixels_to_slide"],
        target,
        {"min_x": 0, "min_y": 0, "max_x": 21, "max_y": 21},
    )
    assert np.array_equal(actual.pixels, expected)
    assert actual.labels.real_pixel_mask.all()
    assert actual.labels.supervision_mask.all()


def test_image_edge_keeps_real_interpolated_pixels_and_masks_only_outside_centers(
    tmp_path,
):
    extracted = _dataset(tmp_path / "capture", objects=[])
    origin = _origin(extracted)
    origin["source"]["calibration"] = {"mpp_x": 0.4, "mpp_y": 0.4}
    _write_roi(extracted, _rectangle(0, 0, 64, 64))
    dataset = _retain(tmp_path, extracted, origin)
    actual = _read(dataset, _target(12, 12, 50.4, 50.4, 1.25, 1.25))
    expected_real = np.zeros((12, 12), dtype=bool)
    expected_real[:11, :11] = True
    assert np.array_equal(actual.labels.real_pixel_mask, expected_real)
    assert 0 < actual.pixels[10, 10, 0] < 255
    assert np.all(actual.pixels[-1] == 255)
    assert np.all(actual.pixels[:, -1] == 255)


@pytest.mark.parametrize("fail_read", [False, True])
def test_source_handle_closes_before_target_work_or_when_pixel_read_fails(
    tmp_path, monkeypatch, fail_read
):
    extracted = _dataset(tmp_path / "capture")
    dataset = _retain(tmp_path, extracted, _origin(extracted))
    closed = []
    ordinary_close = SourceReader.close
    ordinary_targets = training_field_reader.build_training_field_targets

    def record_close(reader):
        ordinary_close(reader)
        closed.append(True)

    def checked_targets(**kwargs):
        assert closed == [True]
        gate = artifact_volume.artifact_volume_gate
        assert gate.acquire(blocking=False)
        gate.release()
        return ordinary_targets(**kwargs)

    monkeypatch.setattr(SourceReader, "close", record_close)
    monkeypatch.setattr(
        training_field_reader, "build_training_field_targets", checked_targets
    )
    if fail_read:

        def failure(*_args):
            raise OSError("injected image read failure")

        monkeypatch.setattr(SourceReader, "read_rgb_view", failure)
        with pytest.raises(OSError, match="injected"):
            _read(dataset, _target(32, 32))
    else:
        _read(dataset, _target(32, 32))
    assert closed == [True]


@pytest.mark.parametrize("changed", ["target_mpp", "scale_x", "scale_y"])
def test_training_grid_must_match_the_dataset_calibration(tmp_path, changed):
    extracted = _dataset(tmp_path / "capture")
    dataset = _retain(tmp_path, extracted, _origin(extracted))
    target = _target(32, 32)
    owner = target["raster"] if changed == "target_mpp" else target["raster_to_slide"]
    owner[changed] *= 2
    with pytest.raises(ValueError, match="dataset calibration"):
        _read(dataset, target)


def _read(dataset, target, rectangle=None, *, head="hovernet", source_limits=None):
    rectangle = rectangle or {
        "x": 0,
        "y": 0,
        "width": target["raster"]["width"],
        "height": target["raster"]["height"],
    }
    return read_training_field(
        dataset,
        0,
        raster_transform=target,
        output_rect=rectangle,
        head_type=head,
        source_limits=source_limits or _source_limits(),
        max_json_member_bytes=8 * 1024**2,
    )


def _retain(tmp_path, extracted, origin, *, capability=None, source_limits=None):
    """Admit, durably publish, and reverify the same fixture used by field reads."""
    (extracted.root / "artifacts/0/origin.json").write_bytes(canonical_bytes(origin))
    preimage = copy.deepcopy(extracted.manifest)
    preimage.pop("dataset_digest")
    for item in preimage["files"]:
        value = (extracted.root / item["path"]).read_bytes()
        item.update(size_bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
    preimage["files"].sort(key=lambda item: item["path"])
    manifest = build_training_dataset_manifest(preimage, training_api_limits())
    encoded = canonical_bytes(manifest)
    (extracted.root / "dataset.json").write_bytes(encoded)
    updated = ExtractedTrainingDataset(
        extracted.root,
        manifest,
        len(encoded) + sum(item["size_bytes"] for item in manifest["files"]),
    )
    admitted = admit_extracted_training_dataset(
        updated,
        training_api_limits(),
        source_limits or _source_limits(),
        lambda _id: capability or training_capability(),
    )
    files = TrainingDatasetFiles(tmp_path / "volume")
    dataset_id = "aaaaaaaa-1111-4111-8111-111111111111"
    files.publish_dataset(
        dataset_id, admitted.root, admitted.manifest, training_api_limits()
    )
    return files.read_dataset(dataset_id, training_api_limits())


def _origin(extracted):
    return json.loads((extracted.root / "artifacts/0/origin.json").read_bytes())


def _write_roi(extracted, geometry):
    (extracted.root / "rois/0/roi.geojson").write_bytes(
        canonical_bytes(
            {
                "type": "Feature",
                "geometry": canonical_geometry(geometry),
                "properties": {},
            }
        )
    )


def _pattern(width, height):
    y, x = np.mgrid[:height, :width]
    return np.stack((x % 251, y % 241, (3 * x + 7 * y) % 239), axis=2).astype(np.uint8)


def _write_png(path, pixels):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    assert ok
    path.write_bytes(encoded.tobytes())


def _target(width, height, x=0.0, y=0.0, step_x=1.0, step_y=1.0):
    return {
        "raster": {
            "width": width,
            "height": height,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 0.5,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": {
            "scale_x": step_x,
            "scale_y": step_y,
            "translate_x": x,
            "translate_y": y,
        },
    }


def _source_limits():
    return DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
