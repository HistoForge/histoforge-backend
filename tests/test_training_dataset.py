from __future__ import annotations

import copy

import pytest
from histoforge.canonical_json import canonical_digest
from histoforge.cloud.training.contract import training_task_from_capability
from histoforge.cloud.training.datasets.manifest import (
    TRAINING_DATASET_SCHEMA_VERSION,
    build_training_dataset_manifest,
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.transfer import (
    maximum_training_dataset_single_item_page_bytes,
)
from histoforge.cloud.training.limits import (
    validate_training_api_deployment_limits,
    validate_training_api_limits,
)
from histoforge.cloud.training.runs.contract import (
    maximum_training_run_single_item_page_bytes,
)
from tests.test_training_contract import training_capability, training_model_output


def training_api_limits() -> dict:
    return {
        "dataset": {
            "max_rois": 20,
            "max_objects_per_roi": 1000,
            "max_total_objects": 5000,
            "max_vertices_per_geometry": 10000,
            "max_total_vertices": 100000,
            "max_json_member_bytes": 8 * 1024**2,
            "max_archive_bytes": 8 * 1024**3,
            "max_extracted_bytes": 9 * 1024**3,
        },
        "transfer": {
            "max_chunk_bytes": 8 * 1024**2,
            "max_chunks_per_upload": 1024,
        },
        "report": {
            "max_metric_series": 100,
            "max_epochs": 1000,
            "max_messages": 1000,
            "max_text_bytes": 4096,
            "max_component_parameters_bytes": 65536,
            "max_component_parameter_depth": 16,
            "max_component_collection_entries": 1024,
            "max_report_bytes": 8 * 1024**2,
        },
        "max_page_size": 100,
        "max_page_encoded_bytes": 1024**2,
    }


def manifest_preimage() -> dict:
    digest = "a" * 64
    return {
        "schema_version": 1,
        "task": training_task_from_capability(
            training_capability(), training_model_output(), 0.5, 4096
        ),
        "artifacts": [{"image_index": 0, "representation": "png"}],
        "project_groups": ["Control"],
        "images": [{"project_entry_id_digest": "1" * 64, "project_group_index": 0}],
        "rois": [
            {
                "source_roi_uuid": "11111111-1111-4111-8111-111111111111",
                "image_index": 0,
                "source_envelope": {
                    "artifact_index": 0,
                    "series_index": 0,
                    "resolution_level": 0,
                    "pixel_rect": {"x": 0, "y": 0, "width": 100, "height": 80},
                },
            }
        ],
        "files": [
            {"path": "artifacts/0/origin.json", "size_bytes": 100, "sha256": digest},
            {"path": "artifacts/0/source.png", "size_bytes": 500, "sha256": digest},
            {"path": "rois/0/objects.geojson", "size_bytes": 120, "sha256": digest},
            {"path": "rois/0/roi.geojson", "size_bytes": 80, "sha256": digest},
        ],
    }


def test_training_limits_require_archive_to_fit_the_chunk_plan() -> None:
    limits = training_api_limits()
    limits["dataset"]["max_archive_bytes"] += 1
    with pytest.raises(ValueError, match="canonical chunk plan"):
        validate_training_api_limits(limits)


def test_deployment_page_limit_fits_one_maximum_legal_list_item() -> None:
    limits = training_api_limits()
    required = max(
        maximum_training_dataset_single_item_page_bytes(limits),
        maximum_training_run_single_item_page_bytes(limits),
    )
    limits["max_page_encoded_bytes"] = required - 1
    with pytest.raises(ValueError, match="cannot hold one maximum"):
        validate_training_api_deployment_limits(limits)

    limits["max_page_encoded_bytes"] = required
    assert validate_training_api_deployment_limits(limits) == limits


@pytest.mark.parametrize("schema_version", [1, TRAINING_DATASET_SCHEMA_VERSION])
def test_manifest_builds_and_verifies_scientific_identity(schema_version) -> None:
    source = manifest_preimage()
    source["schema_version"] = schema_version
    manifest = build_training_dataset_manifest(source, training_api_limits())
    preimage = {
        key: value for key, value in manifest.items() if key != "dataset_digest"
    }
    assert manifest["dataset_digest"] == canonical_digest(preimage)
    assert manifest["schema_version"] == schema_version
    assert (
        validate_training_dataset_manifest(manifest, training_api_limits()) == manifest
    )


def test_manifest_rejects_digest_drift_and_wrong_member_graph() -> None:
    manifest = build_training_dataset_manifest(
        manifest_preimage(), training_api_limits()
    )
    drifted = copy.deepcopy(manifest)
    drifted["task"]["model_microns_per_pixel"] = 0.25
    with pytest.raises(ValueError, match="dataset_digest"):
        validate_training_dataset_manifest(drifted, training_api_limits())

    wrong = manifest_preimage()
    wrong["files"][1]["path"] = "artifacts/0/source.tif"
    with pytest.raises(ValueError, match="required package member set"):
        build_training_dataset_manifest(wrong, training_api_limits())


def test_manifest_applies_json_limit_only_to_json_members() -> None:
    limits = training_api_limits()
    limits["dataset"]["max_json_member_bytes"] = 1024
    too_large = manifest_preimage()
    too_large["files"][2]["size_bytes"] = 1025
    with pytest.raises(ValueError, match="max_json_member_bytes"):
        build_training_dataset_manifest(too_large, limits)

    limits["dataset"]["max_json_member_bytes"] = 4096
    binary = manifest_preimage()
    binary["files"][1]["size_bytes"] = 4097
    assert (
        build_training_dataset_manifest(binary, limits)["files"][1]["size_bytes"]
        == 4097
    )


def test_manifest_enforces_roi_limit_before_member_processing() -> None:
    limits = training_api_limits()
    limits["dataset"]["max_rois"] = 1
    oversized = manifest_preimage()
    duplicate = copy.deepcopy(oversized["rois"][0])
    duplicate["source_roi_uuid"] = "22222222-2222-4222-8222-222222222222"
    oversized["rois"].append(duplicate)

    with pytest.raises(ValueError, match="rois exceeds"):
        build_training_dataset_manifest(oversized, limits)


def test_manifest_enforces_one_source_mode_per_image() -> None:
    wrong = manifest_preimage()
    wrong["schema_version"] = TRAINING_DATASET_SCHEMA_VERSION
    wrong["artifacts"].append({"image_index": 0, "representation": "original"})
    wrong["files"].extend(
        [
            {
                "path": "artifacts/1/origin.json",
                "size_bytes": 100,
                "sha256": "b" * 64,
            },
            {
                "path": "artifacts/1/source.tif",
                "size_bytes": 500,
                "sha256": "b" * 64,
            },
        ]
    )
    wrong["files"].sort(key=lambda item: item["path"])
    with pytest.raises(ValueError, match="one original artifact"):
        build_training_dataset_manifest(wrong, training_api_limits())


def test_schema_one_original_requires_recapture_not_an_inferred_reader() -> None:
    original = manifest_preimage()
    original["artifacts"][0]["representation"] = "original"
    original["files"][1]["path"] = "artifacts/0/source.tif"
    with pytest.raises(ValueError, match="must be recaptured with reader settings"):
        build_training_dataset_manifest(original, training_api_limits())

    original["schema_version"] = TRAINING_DATASET_SCHEMA_VERSION
    assert build_training_dataset_manifest(original, training_api_limits())


def test_manifest_rejects_unreferenced_group_and_cross_image_envelope() -> None:
    unreferenced = manifest_preimage()
    unreferenced["project_groups"].append("Treated")
    with pytest.raises(ValueError, match="every project group"):
        build_training_dataset_manifest(unreferenced, training_api_limits())

    cross_image = manifest_preimage()
    cross_image["images"].append({"project_entry_id_digest": "2" * 64})
    with pytest.raises(ValueError, match="at least one ROI"):
        build_training_dataset_manifest(cross_image, training_api_limits())
