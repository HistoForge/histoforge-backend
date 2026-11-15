"""Canonical immutable training-dataset manifest and package graph.

The extension constructs these records and the backend validates them again at
seal. This module owns identity and cross-reference rules, but no archive IO or
storage lifecycle.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    require_canonical_uuid,
    require_int32,
    require_nonnegative_int32,
    require_positive_safe_integer,
    require_sha256,
)
from histoforge.cloud.sources.identity import validate_int_rect
from histoforge.cloud.training.contract import (
    require_training_text,
    validate_training_task,
)
from histoforge.cloud.training.limits import validate_training_api_limits

TRAINING_DATASET_SCHEMA_VERSION = 2

_MANIFEST_FIELDS = {
    "schema_version",
    "dataset_digest",
    "task",
    "artifacts",
    "project_groups",
    "images",
    "rois",
    "files",
}
_MANIFEST_PREIMAGE_FIELDS = _MANIFEST_FIELDS - {"dataset_digest"}
_IMAGE_FIELDS = {"project_entry_id_digest"}
_IMAGE_FIELDS_WITH_GROUP = _IMAGE_FIELDS | {"project_group_index"}
_ROI_FIELDS = {"source_roi_uuid", "image_index", "source_envelope"}
_ARTIFACT_FIELDS = {"image_index", "representation"}
_ENVELOPE_FIELDS = {
    "artifact_index",
    "series_index",
    "resolution_level",
    "pixel_rect",
}
_FILE_FIELDS = {"path", "size_bytes", "sha256"}
_REPRESENTATIONS = {"png", "ome_bigtiff", "original"}
_SOURCE_BASENAME = {
    "png": "source.png",
    "ome_bigtiff": "source.ome.tif",
    "original": "source.tif",
}


def build_training_dataset_manifest(value: Any, api_limits: Any) -> dict[str, Any]:
    """Normalize a digest-free manifest and add its scientific identity."""

    preimage = _normalize_manifest_preimage(value, api_limits)
    result = canonical_tree({**preimage, "dataset_digest": canonical_digest(preimage)})
    _require_extracted_size(result, api_limits)
    return result


def validate_training_dataset_manifest(value: Any, api_limits: Any) -> dict[str, Any]:
    """Validate a self-digested manifest and every package-local reference."""

    item = _exact_object(value, _MANIFEST_FIELDS, "training dataset manifest")
    supplied_digest = require_sha256(item["dataset_digest"], "dataset_digest")
    preimage = _normalize_manifest_preimage(
        {key: item[key] for key in _MANIFEST_PREIMAGE_FIELDS}, api_limits
    )
    if supplied_digest != canonical_digest(preimage):
        raise ValueError("dataset_digest does not match the manifest preimage")
    result = canonical_tree({**preimage, "dataset_digest": supplied_digest})
    _require_extracted_size(result, api_limits)
    return result


def training_dataset_digest(value: Any, api_limits: Any) -> str:
    """Return the verified scientific identity of one complete manifest."""

    return validate_training_dataset_manifest(value, api_limits)["dataset_digest"]


def required_training_package_paths(manifest: Any, api_limits: Any) -> tuple[str, ...]:
    """Return every fixed non-manifest member path in canonical order."""

    value = validate_training_dataset_manifest(manifest, api_limits)
    return tuple(item["path"] for item in value["files"])


def artifact_origin_path(artifact_index: int) -> str:
    return f"artifacts/{_index(artifact_index, 'artifact_index')}/origin.json"


def artifact_source_path(artifact_index: int, representation: str) -> str:
    index = _index(artifact_index, "artifact_index")
    if representation not in _REPRESENTATIONS:
        raise ValueError("unsupported training artifact representation")
    return f"artifacts/{index}/{_SOURCE_BASENAME[representation]}"


def roi_geometry_path(roi_index: int) -> str:
    return f"rois/{_index(roi_index, 'roi_index')}/roi.geojson"


def roi_objects_path(roi_index: int) -> str:
    return f"rois/{_index(roi_index, 'roi_index')}/objects.geojson"


def _normalize_manifest_preimage(value: Any, api_limits: Any) -> dict[str, Any]:
    item = _exact_object(value, _MANIFEST_PREIMAGE_FIELDS, "training dataset preimage")
    limits = validate_training_api_limits(api_limits)
    dataset_limits = limits["dataset"]
    schema_version = require_int32(item["schema_version"], "schema_version")
    if schema_version not in {1, TRAINING_DATASET_SCHEMA_VERSION}:
        raise ValueError("unsupported training dataset schema_version")

    project_groups = _project_groups(
        item["project_groups"], limits["report"]["max_text_bytes"]
    )
    images = _images(item["images"], len(project_groups))
    rois = _rois(item["rois"], len(images))
    artifacts = _artifacts(item["artifacts"], len(images))
    if schema_version == 1 and any(
        artifact["representation"] == "original" for artifact in artifacts
    ):
        raise ValueError(
            "schema-1 original training artifacts must be recaptured "
            "with reader settings"
        )
    files = _package_files(item["files"], dataset_limits)

    _require_count("rois", len(rois), dataset_limits["max_rois"])
    _require_manifest_graph(project_groups, images, rois, artifacts, files)

    return canonical_tree(
        {
            "schema_version": schema_version,
            "task": validate_training_task(item["task"]),
            "artifacts": artifacts,
            "project_groups": project_groups,
            "images": images,
            "rois": rois,
            "files": files,
        }
    )


def _project_groups(value: Any, max_text_bytes: int) -> list[str]:
    groups = [
        require_training_text(item, "project_groups item", max_text_bytes)
        for item in _array(value, "project_groups")
    ]
    if len(groups) != len(set(groups)):
        raise ValueError("project_groups contains duplicates")
    if groups != sorted(groups, key=canonical_bytes):
        raise ValueError("project_groups is not in canonical JCS-string order")
    return groups


def _images(value: Any, group_count: int) -> list[dict[str, Any]]:
    raw = _nonempty_array(value, "images")
    result = []
    for value_item in raw:
        item = _object(value_item, "training image")
        expected = (
            _IMAGE_FIELDS_WITH_GROUP if "project_group_index" in item else _IMAGE_FIELDS
        )
        if set(item) != expected:
            raise ValueError("training image has missing or unknown fields")
        image: dict[str, Any] = {
            "project_entry_id_digest": require_sha256(
                item["project_entry_id_digest"], "project_entry_id_digest"
            )
        }
        if "project_group_index" in item:
            group_index = require_nonnegative_int32(
                item["project_group_index"], "project_group_index"
            )
            if group_index >= group_count:
                raise ValueError("project_group_index is outside project_groups")
            image["project_group_index"] = group_index
        result.append(image)
    digests = [item["project_entry_id_digest"] for item in result]
    if digests != sorted(digests) or len(digests) != len(set(digests)):
        raise ValueError("images must be unique and sorted by project entry digest")
    return result


def _rois(value: Any, image_count: int) -> list[dict[str, Any]]:
    result = []
    for value_item in _nonempty_array(value, "rois"):
        item = _exact_object(value_item, _ROI_FIELDS, "training ROI")
        image_index = require_nonnegative_int32(item["image_index"], "image_index")
        if image_index >= image_count:
            raise ValueError("ROI image_index is outside images")
        result.append(
            {
                "source_roi_uuid": require_canonical_uuid(
                    item["source_roi_uuid"], "source_roi_uuid"
                ),
                "image_index": image_index,
                "source_envelope": _source_envelope(item["source_envelope"]),
            }
        )
    keys = [(item["image_index"], item["source_roi_uuid"]) for item in result]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("rois must be unique and sorted by image and UUID")
    return result


def _artifacts(value: Any, image_count: int) -> list[dict[str, Any]]:
    result = []
    for value_item in _nonempty_array(value, "artifacts"):
        item = _exact_object(value_item, _ARTIFACT_FIELDS, "training artifact")
        image_index = require_nonnegative_int32(item["image_index"], "image_index")
        if image_index >= image_count:
            raise ValueError("artifact image_index is outside images")
        representation = item["representation"]
        if (
            not isinstance(representation, str)
            or representation not in _REPRESENTATIONS
        ):
            raise ValueError("unsupported training artifact representation")
        result.append({"image_index": image_index, "representation": representation})
    image_order = [item["image_index"] for item in result]
    if image_order != sorted(image_order):
        raise ValueError("artifacts must be grouped by image_index")
    return result


def _source_envelope(value: Any) -> dict[str, Any]:
    item = _exact_object(value, _ENVELOPE_FIELDS, "training source envelope")
    return {
        "artifact_index": require_nonnegative_int32(
            item["artifact_index"], "artifact_index"
        ),
        "series_index": require_nonnegative_int32(item["series_index"], "series_index"),
        "resolution_level": require_nonnegative_int32(
            item["resolution_level"], "resolution_level"
        ),
        "pixel_rect": validate_int_rect(item["pixel_rect"]),
    }


def _package_files(value: Any, limits: Mapping[str, int]) -> list[dict[str, Any]]:
    result = []
    for value_item in _nonempty_array(value, "files"):
        item = _exact_object(value_item, _FILE_FIELDS, "package file")
        path = _package_path(item["path"])
        size = require_positive_safe_integer(item["size_bytes"], "size_bytes")
        if (
            path.endswith((".json", ".geojson"))
            and size > limits["max_json_member_bytes"]
        ):
            raise ValueError("JSON package member exceeds max_json_member_bytes")
        result.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": require_sha256(item["sha256"], "sha256"),
            }
        )
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("files must be unique and sorted by ASCII path")
    if sum(item["size_bytes"] for item in result) > limits["max_extracted_bytes"]:
        raise ValueError("package files exceed max_extracted_bytes")
    return result


def _require_manifest_graph(
    project_groups: list[str],
    images: list[dict[str, Any]],
    rois: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    files: list[dict[str, Any]],
) -> None:
    rois_by_image: dict[int, list[int]] = defaultdict(list)
    for roi_index, roi in enumerate(rois):
        rois_by_image[roi["image_index"]].append(roi_index)
    artifacts_by_image: dict[int, list[int]] = defaultdict(list)
    for artifact_index, artifact in enumerate(artifacts):
        artifacts_by_image[artifact["image_index"]].append(artifact_index)

    for image_index in range(len(images)):
        image_rois = rois_by_image[image_index]
        image_artifacts = artifacts_by_image[image_index]
        if not image_rois or not image_artifacts:
            raise ValueError("every image must own at least one ROI and artifact")
        representations = {
            artifacts[index]["representation"] for index in image_artifacts
        }
        if "original" in representations:
            if representations != {"original"} or len(image_artifacts) != 1:
                raise ValueError(
                    "original mode requires one original artifact per image"
                )
            expected_artifact = image_artifacts[0]
            if any(
                rois[index]["source_envelope"]["artifact_index"] != expected_artifact
                for index in image_rois
            ):
                raise ValueError("original-mode ROIs must share the image artifact")
        else:
            if len(image_rois) != len(image_artifacts):
                raise ValueError("derived mode requires one artifact per ROI")
            for roi_index, artifact_index in zip(
                image_rois, image_artifacts, strict=True
            ):
                if (
                    rois[roi_index]["source_envelope"]["artifact_index"]
                    != artifact_index
                ):
                    raise ValueError(
                        "derived ROI does not reference its paired artifact"
                    )

    for roi in rois:
        artifact_index = roi["source_envelope"]["artifact_index"]
        if artifact_index >= len(artifacts):
            raise ValueError("source envelope artifact_index is outside artifacts")
        if artifacts[artifact_index]["image_index"] != roi["image_index"]:
            raise ValueError(
                "source envelope references an artifact from another image"
            )

    used_groups = {
        image["project_group_index"]
        for image in images
        if "project_group_index" in image
    }
    if used_groups != set(range(len(project_groups))):
        raise ValueError("every project group must be referenced by an image")

    expected_paths = set()
    for artifact_index, artifact in enumerate(artifacts):
        expected_paths.add(artifact_origin_path(artifact_index))
        expected_paths.add(
            artifact_source_path(artifact_index, artifact["representation"])
        )
    for roi_index in range(len(rois)):
        expected_paths.add(roi_geometry_path(roi_index))
        expected_paths.add(roi_objects_path(roi_index))
    if {item["path"] for item in files} != expected_paths:
        raise ValueError("files do not equal the required package member set")


def _require_extracted_size(manifest: Mapping[str, Any], api_limits: Any) -> None:
    limits = validate_training_api_limits(api_limits)["dataset"]
    manifest_bytes = len(canonical_bytes(manifest))
    if manifest_bytes > limits["max_json_member_bytes"]:
        raise ValueError("dataset.json exceeds max_json_member_bytes")
    total = manifest_bytes + sum(item["size_bytes"] for item in manifest["files"])
    if total > limits["max_extracted_bytes"]:
        raise ValueError("dataset exceeds max_extracted_bytes")


def _require_count(path: str, count: int, maximum: int) -> None:
    if count > maximum:
        raise ValueError(f"{path} exceeds its configured maximum")


def _package_path(value: Any) -> str:
    if not isinstance(value, str) or not value or not value.isascii():
        raise ValueError("package path must be nonempty ASCII")
    if value.startswith("/") or "\\" in value or "//" in value:
        raise ValueError("package path is not normalized relative ASCII")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("package path is not normalized relative ASCII")
    return value


def _index(value: Any, path: str) -> int:
    return require_nonnegative_int32(value, path)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    item = _object(value, path)
    if set(item) != fields:
        raise ValueError(f"{path} has missing or unknown fields")
    return item


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _nonempty_array(value: Any, path: str) -> list[Any]:
    result = _array(value, path)
    if not result:
        raise ValueError(f"{path} must be nonempty")
    return result
