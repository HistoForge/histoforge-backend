from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from histoforge.canonical_geometry import canonical_geometry
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.rasterization.inputs import geometry_vertex_count
from histoforge.cloud.training.contract import training_task_from_capability
from histoforge.cloud.training.datasets.archive import ExtractedTrainingDataset
from histoforge.cloud.training.datasets.content import (
    TrainingRoiLabelError,
    admit_extracted_training_dataset,
)
from histoforge.cloud.training.datasets.io import open_training_artifact
from histoforge.cloud.training.datasets.manifest import (
    TRAINING_DATASET_SCHEMA_VERSION,
    build_training_dataset_manifest,
)
from tests.test_training_contract import training_capability, training_model_output
from tests.test_training_dataset import manifest_preimage, training_api_limits


@pytest.mark.parametrize("schema_version", [1, TRAINING_DATASET_SCHEMA_VERSION])
@pytest.mark.parametrize(
    "adapter", ["qupath.openslide.local", "qupath.bioformats.local"]
)
def test_complete_derived_dataset_is_admitted(tmp_path, schema_version, adapter):
    extracted = _dataset(tmp_path, schema_version=schema_version, adapter=adapter)
    admitted = _admit(extracted)
    assert admitted.manifest == extracted.manifest
    assert admitted.retained_bytes == extracted.retained_bytes
    assert admitted.task_display_name == "Leydig cell segmentation"


def test_original_dataset_accepts_a_discovered_nonzero_pyramid_level(
    tmp_path: Path,
) -> None:
    admitted = _admit(_original_dataset(tmp_path, resolution_level=1))

    assert admitted.manifest["artifacts"] == [
        {"image_index": 0, "representation": "original"}
    ]
    assert admitted.manifest["rois"][0]["source_envelope"]["resolution_level"] == 1


def test_single_original_keeps_logical_entry_name_and_fixed_package_path(tmp_path):
    extracted = _original_dataset(tmp_path, resolution_level=1)
    with open_training_artifact(
        extracted.root,
        extracted.manifest,
        0,
        training_api_limits()["dataset"]["max_json_member_bytes"],
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
    ) as (origin, reader):
        assert origin["entry_path"] == "generic-tiff-v1.tif"
        assert (extracted.root / "artifacts/0/source.tif").is_file()
        pixels = reader.read_rgb_view(0, 1, {"x": 1, "y": 2, "width": 5, "height": 4})
    assert pixels.shape == (4, 5, 3)
    assert pixels.flags.owndata


@pytest.mark.parametrize(
    "kind", ["bioformats", "companions", "vendor", "background", "levels"]
)
def test_training_original_does_not_expand_with_inference_readers(
    tmp_path, monkeypatch, kind
):
    extracted = _original_dataset(tmp_path, resolution_level=0)
    origin_path = extracted.root / "artifacts/0/origin.json"
    origin = json.loads(origin_path.read_bytes())
    if kind == "bioformats":
        origin["source"]["source_adapter_id"] = "qupath.bioformats.local"
        origin["reader_settings"] = {"levels": origin["reader_settings"]["levels"]}
    elif kind == "companions":
        origin["members"] = [
            {"path": origin["entry_path"], "size_bytes": 1, "sha256": "a" * 64}
        ]
    elif kind == "vendor":
        monkeypatch.setattr("openslide.OpenSlide.detect_format", lambda _path: "aperio")
    elif kind == "background":
        origin["reader_settings"]["background_rgb"] = [255, 255, 255]
    else:
        mapping = origin["reader_settings"]["levels"][1]["pixels_to_level_zero"]
        mapping.update(scale_x=1.5, scale_y=1.5)
    origin_path.write_bytes(canonical_bytes(origin))

    def unexpected_reader(*_args, **_kwargs):
        pytest.fail("training opened an unqualified original reader")

    monkeypatch.setattr(
        "histoforge.cloud.training.datasets.io.SourceReader", unexpected_reader
    )
    with pytest.raises(ValueError, match="training originals require"):
        _admit(extracted)


@pytest.mark.parametrize("right", [513.0, 513.95])
def test_rounded_pyramid_envelope_does_not_extend_the_physical_training_roi(
    tmp_path, right
):
    """Coarse-level pixels extend past x=513; the training ROI may not."""
    extracted = _original_dataset(tmp_path, resolution_level=1)
    (extracted.root / "rois/0/roi.geojson").write_bytes(
        canonical_bytes(
            {
                "type": "Feature",
                "geometry": canonical_geometry(_rectangle(452.1, 10, right, 20)),
                "properties": {},
            }
        )
    )
    (extracted.root / "rois/0/objects.geojson").write_bytes(
        canonical_bytes({"type": "FeatureCollection", "features": []})
    )
    if right > 513:
        with pytest.raises(ValueError, match="ROI lies outside the physical image"):
            _admit(extracted)
    else:
        assert _admit(extracted).roi_label_summaries[0].is_negative


def test_label_requires_target_class_type_and_positive_area_intersection(
    tmp_path: Path,
) -> None:
    outside = _object_feature(
        "11111111-1111-4111-8111-111111111111",
        _rectangle(30, 12, 40, 20),
    )
    with pytest.raises(ValueError, match="positive-area"):
        _admit(_dataset(tmp_path / "touching", objects=[outside]))

    wrong_class = copy.deepcopy(outside)
    wrong_class["geometry"] = canonical_geometry(_rectangle(12, 12, 20, 20))
    wrong_class["properties"]["path_class"] = "Tubule"
    with pytest.raises(ValueError, match="target class"):
        _admit(_dataset(tmp_path / "class", objects=[wrong_class]))

    annotation = copy.deepcopy(outside)
    annotation["geometry"] = canonical_geometry(_rectangle(12, 12, 20, 20))
    annotation["properties"]["object_type"] = "annotation"
    assert _admit(_dataset(tmp_path / "annotation", objects=[annotation]))

    for object_type in ("cell", "tile"):
        wrong_type = copy.deepcopy(outside)
        wrong_type["geometry"] = canonical_geometry(_rectangle(12, 12, 20, 20))
        wrong_type["properties"]["object_type"] = object_type
        with pytest.raises(ValueError, match="must be annotation or detection"):
            _admit(_dataset(tmp_path / object_type, objects=[wrong_type]))


def test_negative_roi_requires_capability_support(tmp_path: Path) -> None:
    capability = training_capability()
    capability["supports_negative_rois"] = False
    extracted = _dataset(tmp_path, objects=[], capability=capability)
    with pytest.raises(ValueError, match="negative ROIs"):
        _admit(extracted, capability=capability)

    subpixel = _object_feature(
        "11111111-1111-4111-8111-111111111111",
        _rectangle(12.1, 12.1, 12.4, 12.4),
    )
    with pytest.raises(ValueError, match="negative ROIs"):
        _admit(
            _dataset(
                tmp_path / "raster-negative",
                objects=[subpixel],
                capability=capability,
            ),
            capability=capability,
        )


def test_instance_overlap_fails_dataset_admission(tmp_path: Path) -> None:
    objects = [
        _object_feature(
            "11111111-1111-4111-8111-111111111111",
            _rectangle(12, 12, 20, 20),
        ),
        _object_feature(
            "22222222-2222-4222-8222-222222222222",
            _rectangle(16, 16, 24, 24),
        ),
    ]

    with pytest.raises(
        TrainingRoiLabelError, match="Two training objects overlap"
    ) as raised:
        _admit(_dataset(tmp_path, objects=objects))
    assert raised.value.roi_index == 0


def test_source_envelope_must_contain_complete_roi(tmp_path: Path) -> None:
    extracted = _dataset(
        tmp_path,
        envelope={
            "artifact_index": 0,
            "series_index": 0,
            "resolution_level": 0,
            "pixel_rect": {"x": 0, "y": 0, "width": 8, "height": 8},
        },
    )
    with pytest.raises(ValueError, match="does not contain"):
        _admit(extracted)


def test_source_envelope_must_retain_maximum_permitted_context(
    tmp_path: Path,
) -> None:
    tight = {
        "artifact_index": 0,
        "series_index": 0,
        "resolution_level": 0,
        "pixel_rect": {"x": 10, "y": 10, "width": 20, "height": 20},
    }
    with pytest.raises(ValueError, match="omits permitted context"):
        _admit(_dataset(tmp_path / "context", envelope=tight))

    capability = training_capability()
    capability["default_field_profile"]["small_roi_context"] = False
    _admit(
        _dataset(
            tmp_path / "no-context",
            envelope=tight,
            capability=capability,
        ),
        capability=capability,
    )

    capability["default_field_profile"] = {
        "planner": "variable-field-v2",
        "minSideModelPx": 128,
        "maxSideModelPx": 384,
        "preferredTileSideModelPx": 256,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": 8,
        "smallRoiContext": False,
        "tileOverlap": 0.25,
        "aggregation": "mean-logits-v1",
        "preprocessing": "brightfield-rgb-v1",
    }
    capability["allowed_field_side_model_px"] = {"minimum": 128, "maximum": 384}
    capability["small_roi_context_editable"] = True
    with pytest.raises(ValueError, match="omits permitted context"):
        _admit(
            _dataset(
                tmp_path / "editable-context",
                envelope=tight,
                capability=capability,
            ),
            capability=capability,
        )


def test_crop_source_envelope_must_retain_recipe_padding(tmp_path: Path) -> None:
    capability = training_capability()
    capability["input_kind"] = "crop"
    capability["crop_padding_model_px"] = 8
    capability["default_field_profile"]["small_roi_context"] = False

    tight = {
        "artifact_index": 0,
        "series_index": 0,
        "resolution_level": 0,
        "pixel_rect": {"x": 10, "y": 10, "width": 20, "height": 20},
    }
    with pytest.raises(ValueError, match="omits permitted context"):
        _admit(
            _dataset(
                tmp_path / "missing-padding", envelope=tight, capability=capability
            ),
            capability=capability,
        )

    padded = copy.deepcopy(tight)
    padded["pixel_rect"] = {"x": 0, "y": 0, "width": 40, "height": 40}
    _admit(
        _dataset(tmp_path / "complete-padding", envelope=padded, capability=capability),
        capability=capability,
    )


@pytest.mark.parametrize("width", [31, 40, 50, 63])
def test_truncated_derived_artifact_cannot_hide_missing_real_context(tmp_path, width):
    extracted = _dataset(tmp_path)
    origin_path = tmp_path / "artifacts/0/origin.json"
    origin = json.loads(origin_path.read_bytes())
    origin["source_rect_at_level"]["width"] = width
    origin_path.write_bytes(canonical_bytes(origin))
    ok, encoded = cv2.imencode(".png", np.zeros((64, width, 3), dtype=np.uint8))
    assert ok
    (tmp_path / "artifacts/0/source.png").write_bytes(encoded.tobytes())
    manifest = copy.deepcopy(extracted.manifest)
    manifest.pop("dataset_digest")
    manifest["rois"][0]["source_envelope"]["pixel_rect"]["width"] = width
    for item in manifest["files"]:
        value = (tmp_path / item["path"]).read_bytes()
        item.update(size_bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
    manifest = build_training_dataset_manifest(manifest, training_api_limits())
    encoded_manifest = canonical_bytes(manifest)
    (tmp_path / "dataset.json").write_bytes(encoded_manifest)
    truncated = ExtractedTrainingDataset(
        tmp_path,
        manifest,
        len(encoded_manifest) + sum(item["size_bytes"] for item in manifest["files"]),
    )
    with pytest.raises(ValueError, match="omits permitted context"):
        _admit(truncated)


def test_geometry_files_must_be_canonical_and_vertex_bounded(tmp_path: Path) -> None:
    extracted = _dataset(tmp_path)
    roi_path = extracted.root / "rois/0/roi.geojson"
    roi_path.write_bytes(b"  " + roi_path.read_bytes())
    with pytest.raises(ValueError, match="canonical"):
        _admit(extracted)

    extracted = _dataset(tmp_path / "vertices")
    limits = training_api_limits()
    limits["dataset"]["max_vertices_per_geometry"] = 3
    with pytest.raises(ValueError, match="max_vertices_per_geometry"):
        admit_extracted_training_dataset(
            extracted,
            limits,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            lambda _capability_id: training_capability(),
        )


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "object_count", "message"),
    [
        ("max_objects_per_roi", 1, 2, "max_objects_per_roi"),
        ("max_total_objects", 1, 2, "max_total_objects"),
        ("max_total_vertices", 7, 1, "max_total_vertices"),
    ],
)
def test_content_admission_recomputes_object_and_total_vertex_limits(
    tmp_path: Path,
    limit_name: str,
    limit_value: int,
    object_count: int,
    message: str,
) -> None:
    objects = [
        _object_feature(
            f"{index + 1:08d}-1111-4111-8111-111111111111",
            _rectangle(
                12 + 10 * index,
                12 + 10 * index,
                20 + 8 * index,
                20 + 8 * index,
            ),
        )
        for index in range(object_count)
    ]
    extracted = _dataset(tmp_path, objects=objects)
    limits = training_api_limits()
    limits["dataset"][limit_name] = limit_value

    with pytest.raises(ValueError, match=message):
        admit_extracted_training_dataset(
            extracted,
            limits,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            lambda _capability_id: training_capability(),
        )


def test_vertex_limit_rejects_before_label_rasterization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extracted = _dataset(tmp_path)
    limits = training_api_limits()
    limits["dataset"]["max_total_vertices"] = 7

    def unexpected_rasterization(**_kwargs) -> None:
        pytest.fail("label rasterization ran before vertex admission")

    monkeypatch.setattr(
        "histoforge.cloud.training.datasets.content.inspect_training_roi_labels",
        unexpected_rasterization,
    )
    with pytest.raises(ValueError, match="max_total_vertices"):
        admit_extracted_training_dataset(
            extracted,
            limits,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            lambda _capability_id: training_capability(),
        )


def test_vertex_budget_excludes_geojson_ring_closures() -> None:
    polygon_with_hole = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[2, 2], [2, 4], [4, 4], [4, 2], [2, 2]],
        ],
    }
    multi = {
        "type": "MultiPolygon",
        "coordinates": [
            polygon_with_hole["coordinates"],
            [[[20, 20], [21, 20], [21, 21], [20, 21], [20, 20]]],
        ],
    }

    assert geometry_vertex_count(polygon_with_hole) == 8
    assert geometry_vertex_count(multi) == 12


def _admit(
    extracted: ExtractedTrainingDataset,
    *,
    capability: dict | None = None,
):
    selected = capability or training_capability()
    return admit_extracted_training_dataset(
        extracted,
        training_api_limits(),
        DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
        lambda capability_id: (
            selected if capability_id == selected["capability_id"] else None
        ),
    )


def _dataset(
    root: Path,
    *,
    objects: list[dict] | None = None,
    envelope: dict | None = None,
    capability: dict | None = None,
    model_output: dict | None = None,
    schema_version: int = 1,
    adapter: str = "qupath.openslide.local",
) -> ExtractedTrainingDataset:
    root.mkdir(parents=True, exist_ok=True)
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    source = encoded.tobytes()
    origin = {
        "type": "derived-region",
        "source": {
            "source_adapter_id": adapter,
            "credential_free_descriptor_digest": "1" * 64,
            "width": 64,
            "height": 64,
            "z_slices": 1,
            "timepoints": 1,
            "calibration": {"mpp_x": 0.5, "mpp_y": 0.5},
        },
        "source_resolution_level": 0,
        "plane": {"z": 0, "t": 0},
        "source_rect_at_level": {"x": 0, "y": 0, "width": 64, "height": 64},
        "artifact_pixels_to_slide": {
            "scale_x": 1,
            "scale_y": 1,
            "translate_x": 0,
            "translate_y": 0,
        },
        "writer_build": {
            "version": "0.1.0+test",
            "artifact_digest": "2" * 64,
            "manifest_digest": "3" * 64,
        },
    }
    roi_geometry = canonical_geometry(_rectangle(10, 10, 30, 30))
    roi = {"type": "Feature", "geometry": roi_geometry, "properties": {}}
    object_values = objects
    if object_values is None:
        object_values = [
            _object_feature(
                "11111111-1111-4111-8111-111111111111",
                _rectangle(12, 12, 20, 20),
            )
        ]
    collection = {"type": "FeatureCollection", "features": object_values}
    payloads = {
        "artifacts/0/origin.json": canonical_bytes(origin),
        "artifacts/0/source.png": source,
        "rois/0/objects.geojson": canonical_bytes(collection),
        "rois/0/roi.geojson": canonical_bytes(roi),
    }
    for relative, value in payloads.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    preimage = manifest_preimage()
    preimage["schema_version"] = schema_version
    if capability is not None:
        preimage["task"] = training_task_from_capability(
            capability, model_output or training_model_output(), 0.5, 4096
        )
    preimage["rois"][0]["source_envelope"] = envelope or {
        "artifact_index": 0,
        "series_index": 0,
        "resolution_level": 0,
        "pixel_rect": {"x": 0, "y": 0, "width": 64, "height": 64},
    }
    preimage["files"] = [
        {
            "path": path,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for path, value in sorted(payloads.items())
    ]
    manifest = build_training_dataset_manifest(preimage, training_api_limits())
    manifest_bytes = canonical_bytes(manifest)
    (root / "dataset.json").write_bytes(manifest_bytes)
    return ExtractedTrainingDataset(
        root,
        manifest,
        len(manifest_bytes) + sum(len(value) for value in payloads.values()),
    )


def _original_dataset(
    root: Path,
    *,
    resolution_level: int,
    capability: dict | None = None,
    model_output: dict | None = None,
) -> ExtractedTrainingDataset:
    root.mkdir(parents=True, exist_ok=True)
    fixtures = Path(__file__).parent / "fixtures"
    source_path = fixtures / "openslide" / "generic-tiff-v1.tif"
    source = source_path.read_bytes()
    description = json.loads(
        (fixtures / "contracts" / "openslide-original-v1.json").read_text(
            encoding="utf-8"
        )
    )
    level_zero = description["levels"][0]
    selected = description["levels"][resolution_level]
    origin = {
        "type": "original-wsi",
        "entry_path": "generic-tiff-v1.tif",
        "reader_settings": {
            "crop_x": 0,
            "crop_y": 0,
            "background_rgb": [0, 0, 0],
            "tile_width": 256,
            "tile_height": 256,
            "levels": [
                {
                    "resolution_level": index,
                    "native_resolution_index": index,
                    "pixels_to_level_zero": {
                        "scale_x": level["qupath_downsample"],
                        "scale_y": level["qupath_downsample"],
                        "translate_x": 0,
                        "translate_y": 0,
                    },
                }
                for index, level in enumerate(description["levels"])
            ],
        },
        "source": {
            "source_adapter_id": description["source_file"]["source_adapter_id"],
            "credential_free_descriptor_digest": "1" * 64,
            "width": level_zero["width"],
            "height": level_zero["height"],
            "z_slices": 1,
            "timepoints": 1,
            "calibration": {
                "mpp_x": description["source_file"]["mpp_x"],
                "mpp_y": description["source_file"]["mpp_y"],
            },
            "content_version": {
                "type": "local-file",
                "file_key": "generic-tiff-v1.tif",
                "last_modified_epoch_millis": 1,
                "size_bytes": len(source),
            },
        },
        "pyramid_series_index": 0,
        "level_zero_pixels_to_slide": {
            "scale_x": 1,
            "scale_y": 1,
            "translate_x": 0,
            "translate_y": 0,
        },
    }
    roi_geometry = canonical_geometry(_rectangle(10, 10, 30, 30))
    payloads = {
        "artifacts/0/origin.json": canonical_bytes(origin),
        "artifacts/0/source.tif": source,
        "rois/0/objects.geojson": canonical_bytes(
            {
                "type": "FeatureCollection",
                "features": [
                    _object_feature(
                        "11111111-1111-4111-8111-111111111111",
                        _rectangle(12, 12, 20, 20),
                    )
                ],
            }
        ),
        "rois/0/roi.geojson": canonical_bytes(
            {"type": "Feature", "geometry": roi_geometry, "properties": {}}
        ),
    }
    for relative, value in payloads.items():
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    preimage = manifest_preimage()
    preimage["schema_version"] = TRAINING_DATASET_SCHEMA_VERSION
    if capability is not None or model_output is not None:
        preimage["task"] = training_task_from_capability(
            capability or training_capability(),
            model_output or training_model_output(),
            0.5,
            4096,
        )
    preimage["artifacts"] = [{"image_index": 0, "representation": "original"}]
    preimage["rois"][0]["source_envelope"] = {
        "artifact_index": 0,
        "series_index": 0,
        "resolution_level": resolution_level,
        "pixel_rect": {
            "x": 0,
            "y": 0,
            "width": selected["width"],
            "height": selected["height"],
        },
    }
    preimage["files"] = [
        {
            "path": path,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for path, value in sorted(payloads.items())
    ]
    manifest = build_training_dataset_manifest(preimage, training_api_limits())
    manifest_bytes = canonical_bytes(manifest)
    (root / "dataset.json").write_bytes(manifest_bytes)
    return ExtractedTrainingDataset(
        root,
        manifest,
        len(manifest_bytes) + sum(len(value) for value in payloads.values()),
    )


def _object_feature(source_id: str, geometry: dict) -> dict:
    return {
        "type": "Feature",
        "geometry": canonical_geometry(geometry),
        "properties": {
            "source_object_uuid": source_id,
            "object_type": "detection",
            "path_class": "Leydig cell",
        },
    }


def _rectangle(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]],
    }
