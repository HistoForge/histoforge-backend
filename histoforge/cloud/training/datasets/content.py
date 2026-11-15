"""Scientific and source-reader admission for one extracted training dataset."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from histoforge.cloud._config import validate_source_artifact_limits
from histoforge.cloud.rasterization.inputs import geometry_vertex_count
from histoforge.cloud.rasterization.rasterizer import required_source_pixel_rect
from histoforge.cloud.sources.views import (
    derive_derived_view_transform,
    derive_original_view_transform,
    source_slide_bounds,
)
from histoforge.cloud.training.contract import (
    training_task_from_capability,
    validate_model_training_capability,
)
from histoforge.cloud.training.datasets.archive import ExtractedTrainingDataset
from histoforge.cloud.training.datasets.io import (
    open_training_artifact,
    read_training_objects,
    read_training_roi_geometry,
)
from histoforge.cloud.training.datasets.label_rasterizer import (
    TrainingRoiLabelSummary,
    inspect_training_roi_labels,
)
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.rasters import (
    maximum_training_source_frame,
    training_raster_transform,
)
from histoforge.cloud.training.limits import validate_training_api_limits


@dataclass(frozen=True)
class AdmittedTrainingDataset:
    """One completely validated dataset ready for immutable publication."""

    root: Path
    manifest: dict[str, Any]
    retained_bytes: int
    task_display_name: str
    roi_label_summaries: tuple[TrainingRoiLabelSummary, ...]


@dataclass(frozen=True)
class _ArtifactInspection:
    origin: dict[str, Any]
    levels: dict[tuple[int, int], dict[str, int]]
    mappings: dict[tuple[int, int], dict[str, float]]


def admit_extracted_training_dataset(
    extracted: ExtractedTrainingDataset,
    api_limits: Any,
    source_limits: Any,
    capability_resolver: Callable[[str], Any],
) -> AdmittedTrainingDataset:
    """Validate task, source files, geometry, labels, and cross-file references."""

    limits = validate_training_api_limits(api_limits)
    accepted_source_limits = validate_source_artifact_limits(source_limits)
    manifest = validate_training_dataset_manifest(extracted.manifest, limits)
    capability = validate_model_training_capability(
        capability_resolver(manifest["task"]["capability_id"]),
        limits["report"]["max_text_bytes"],
    )
    expected_task = training_task_from_capability(
        capability,
        manifest["task"]["model_output"],
        manifest["task"]["model_microns_per_pixel"],
        limits["report"]["max_text_bytes"],
        annotation_class_map=manifest["task"]["annotation_class_map"],
    )
    if manifest["task"] != expected_task:
        raise ValueError("dataset task does not match its training capability")

    inspections = [
        _inspect_artifact(
            extracted.root,
            manifest,
            index,
            limits["dataset"]["max_json_member_bytes"],
            accepted_source_limits,
        )
        for index in range(len(manifest["artifacts"]))
    ]
    _require_image_source_consistency(manifest, inspections)

    object_count = 0
    vertex_count = 0
    label_summaries = []
    dataset_limits = limits["dataset"]
    class_map = manifest["task"]["annotation_class_map"]
    for roi_index, roi in enumerate(manifest["rois"]):
        roi_geometry = read_training_roi_geometry(
            extracted.root,
            roi_index,
            dataset_limits["max_json_member_bytes"],
        )
        roi_vertices = geometry_vertex_count(roi_geometry)
        _require_geometry_vertices(roi_vertices, dataset_limits)
        vertex_count += roi_vertices
        _require_roi_source_support(
            roi_geometry,
            roi,
            inspections,
            manifest["task"],
            capability,
        )

        objects = read_training_objects(
            extracted.root,
            roi_index,
            roi_geometry,
            class_map,
            dataset_limits["max_json_member_bytes"],
        )
        if len(objects) > dataset_limits["max_objects_per_roi"]:
            raise ValueError("ROI exceeds max_objects_per_roi")
        object_count += len(objects)
        if object_count > dataset_limits["max_total_objects"]:
            raise ValueError("dataset exceeds max_total_objects")

        for item in objects:
            count = geometry_vertex_count(item["geometry"])
            _require_geometry_vertices(count, dataset_limits)
            vertex_count += count
        if vertex_count > dataset_limits["max_total_vertices"]:
            raise ValueError("dataset exceeds max_total_vertices")

        origin = inspections[roi["source_envelope"]["artifact_index"]].origin
        label_summary = inspect_training_roi_labels(
            roi_geometry=roi_geometry,
            objects=objects,
            model_output=manifest["task"]["model_output"],
            raster_transform=training_raster_transform(
                shape(roi_geometry).bounds,
                manifest["task"],
                origin["source"],
            ),
            physical_slide_bounds=source_slide_bounds(origin),
        )
        label_summaries.append(label_summary)
        if label_summary.is_negative and not manifest["task"]["supports_negative_rois"]:
            raise ValueError("training capability does not accept negative ROIs")

    return AdmittedTrainingDataset(
        root=extracted.root,
        manifest=manifest,
        retained_bytes=extracted.retained_bytes,
        task_display_name=capability["task_display_name"],
        roi_label_summaries=tuple(label_summaries),
    )


def _inspect_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    artifact_index: int,
    max_json_member_bytes: int,
    source_limits: Mapping[str, int],
) -> _ArtifactInspection:
    with open_training_artifact(
        root, manifest, artifact_index, max_json_member_bytes, source_limits
    ) as (origin, reader):
        levels = {
            (item["series_index"], item["resolution_level"]): item
            for item in reader.validated_levels
        }
        mappings = {
            key: reader.selected_level_pixels_to_level_zero(*key) for key in levels
        }
    return _ArtifactInspection(origin, levels, mappings)


def _require_image_source_consistency(
    manifest: Mapping[str, Any],
    inspections: list[_ArtifactInspection],
) -> None:
    by_image: dict[int, dict[str, Any]] = {}
    for artifact, inspection in zip(manifest["artifacts"], inspections, strict=True):
        image_index = artifact["image_index"]
        source = inspection.origin["source"]
        existing = by_image.setdefault(image_index, source)
        if existing != source:
            raise ValueError("artifacts from one image have different source identity")


def _require_roi_source_support(
    roi_geometry: Mapping[str, Any],
    roi: Mapping[str, Any],
    inspections: list[_ArtifactInspection],
    task: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> None:
    envelope = roi["source_envelope"]
    inspection = inspections[envelope["artifact_index"]]
    key = (envelope["series_index"], envelope["resolution_level"])
    level = inspection.levels.get(key)
    if level is None:
        raise ValueError("training source envelope names an unavailable source level")
    origin = inspection.origin
    if origin["type"] == "derived-region":
        if key != (0, 0):
            raise ValueError("derived training envelopes must use series 0, level 0")
        transform = derive_derived_view_transform(
            origin["artifact_pixels_to_slide"],
            {"width": level["width"], "height": level["height"]},
            envelope["pixel_rect"],
        )
    else:
        if key[0] != origin["pyramid_series_index"]:
            raise ValueError("original training envelope uses the wrong series")
        transform = derive_original_view_transform(
            origin["level_zero_pixels_to_slide"],
            inspection.mappings[key],
            {"width": level["width"], "height": level["height"]},
            envelope["pixel_rect"],
        )
    rectangle = envelope["pixel_rect"]
    envelope_bounds = (
        transform["translate_x"],
        transform["translate_y"],
        transform["translate_x"] + rectangle["width"] * transform["scale_x"],
        transform["translate_y"] + rectangle["height"] * transform["scale_y"],
    )
    roi_bounds = shape(roi_geometry).bounds
    if not _bounds_contain(source_slide_bounds(origin), roi_bounds):
        raise ValueError("training ROI lies outside the physical image")
    if not _bounds_contain(envelope_bounds, roi_bounds):
        raise ValueError("training source envelope does not contain its ROI")

    frame = maximum_training_source_frame(roi_bounds, origin, task, capability)
    if not _bounds_contain(envelope_bounds, frame):
        raise ValueError("training source envelope omits permitted context")
    raster_transform = training_raster_transform(frame, task, origin["source"])
    required = required_source_pixel_rect(
        transform,
        raster_transform,
        rectangle,
        {"width": level["width"], "height": level["height"]},
    )
    if not _rect_contains(rectangle, required):
        raise ValueError(
            "training source envelope omits permitted context or rasterizer support"
        )


def _rect_contains(outer: Mapping[str, int], inner: Mapping[str, int]) -> bool:
    return (
        outer["x"] <= inner["x"]
        and outer["y"] <= inner["y"]
        and outer["x"] + outer["width"] >= inner["x"] + inner["width"]
        and outer["y"] + outer["height"] >= inner["y"] + inner["height"]
    )


def _require_geometry_vertices(count: int, limits: Mapping[str, int]) -> None:
    if count > limits["max_vertices_per_geometry"]:
        raise ValueError("geometry exceeds max_vertices_per_geometry")


def _bounds_contain(outer: tuple[float, ...], inner: tuple[float, ...]) -> bool:
    tolerance = max(1.0, *(abs(value) for value in (*outer, *inner))) * 1e-12
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] + tolerance >= inner[2]
        and outer[3] + tolerance >= inner[3]
    )
