from __future__ import annotations

import copy

import pytest
from histoforge.cloud.training.datasets.manifest import (
    TRAINING_DATASET_SCHEMA_VERSION,
    build_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.split import (
    ExpandedTrainingSplit,
    expand_training_split_plan,
    validate_training_split_plan,
    validate_training_split_sidecar,
)
from tests.test_training_dataset import manifest_preimage, training_api_limits


def two_image_manifest() -> dict:
    preimage = manifest_preimage()
    preimage["images"].append(
        {"project_entry_id_digest": "2" * 64, "project_group_index": 0}
    )
    preimage["rois"].append(
        {
            "source_roi_uuid": "22222222-2222-4222-8222-222222222222",
            "image_index": 1,
            "source_envelope": {
                "artifact_index": 1,
                "series_index": 0,
                "resolution_level": 0,
                "pixel_rect": {"x": 0, "y": 0, "width": 60, "height": 40},
            },
        }
    )
    preimage["artifacts"].append({"image_index": 1, "representation": "png"})
    preimage["files"].extend(
        [
            {
                "path": "artifacts/1/origin.json",
                "size_bytes": 100,
                "sha256": "b" * 64,
            },
            {
                "path": "artifacts/1/source.png",
                "size_bytes": 400,
                "sha256": "b" * 64,
            },
            {
                "path": "rois/1/objects.geojson",
                "size_bytes": 120,
                "sha256": "b" * 64,
            },
            {
                "path": "rois/1/roi.geojson",
                "size_bytes": 80,
                "sha256": "b" * 64,
            },
        ]
    )
    preimage["files"].sort(key=lambda item: item["path"])
    return build_training_dataset_manifest(preimage, training_api_limits())


def grouped_manifest(image_group_indices: list[int], rois_per_image: list[int]) -> dict:
    preimage = manifest_preimage()
    preimage["schema_version"] = TRAINING_DATASET_SCHEMA_VERSION
    preimage["project_groups"] = [
        ["Control", "Treated"][index] for index in range(max(image_group_indices) + 1)
    ]
    preimage["images"] = []
    preimage["rois"] = []
    preimage["artifacts"] = []
    preimage["files"] = []
    roi_number = 1
    for image_index, (group_index, roi_count) in enumerate(
        zip(image_group_indices, rois_per_image, strict=True)
    ):
        preimage["images"].append(
            {
                "project_entry_id_digest": f"{image_index + 1:064x}",
                "project_group_index": group_index,
            }
        )
        preimage["artifacts"].append(
            {"image_index": image_index, "representation": "original"}
        )
        preimage["files"].extend(
            [
                {
                    "path": f"artifacts/{image_index}/origin.json",
                    "size_bytes": 100,
                    "sha256": "a" * 64,
                },
                {
                    "path": f"artifacts/{image_index}/source.tif",
                    "size_bytes": 500,
                    "sha256": "a" * 64,
                },
            ]
        )
        for _ in range(roi_count):
            roi_index = len(preimage["rois"])
            preimage["rois"].append(
                {
                    "source_roi_uuid": (f"00000000-0000-4000-8000-{roi_number:012x}"),
                    "image_index": image_index,
                    "source_envelope": {
                        "artifact_index": image_index,
                        "series_index": 0,
                        "resolution_level": 0,
                        "pixel_rect": {
                            "x": 0,
                            "y": 0,
                            "width": 100,
                            "height": 80,
                        },
                    },
                }
            )
            preimage["files"].extend(
                [
                    {
                        "path": f"rois/{roi_index}/objects.geojson",
                        "size_bytes": 120,
                        "sha256": "a" * 64,
                    },
                    {
                        "path": f"rois/{roi_index}/roi.geojson",
                        "size_bytes": 80,
                        "sha256": "a" * 64,
                    },
                ]
            )
            roi_number += 1
    preimage["files"].sort(key=lambda item: item["path"])
    return build_training_dataset_manifest(preimage, training_api_limits())


def test_generated_split_and_sidecar_round_trip() -> None:
    manifest = two_image_manifest()
    split = {
        "type": "split",
        "unit": "image",
        "assignment_source": "generated",
        "requested_validation_percent": 20,
        "units": [
            {"unit_index": 0, "partition": "training"},
            {"unit_index": 1, "partition": "validation"},
        ],
    }
    assert validate_training_split_plan(split, manifest, training_api_limits()) == split
    sidecar = {
        "schema_version": 1,
        "dataset_digest": manifest["dataset_digest"],
        "split_plan": split,
    }
    assert (
        validate_training_split_sidecar(sidecar, manifest, training_api_limits())
        == sidecar
    )


def test_no_validation_plan_is_closed() -> None:
    manifest = two_image_manifest()
    assert validate_training_split_plan(
        {"type": "none"}, manifest, training_api_limits()
    ) == {"type": "none"}

    with pytest.raises(ValueError, match="missing or unknown"):
        validate_training_split_plan(
            {"type": "none", "units": []}, manifest, training_api_limits()
        )


def test_custom_split_omits_generated_percentage() -> None:
    manifest = two_image_manifest()
    split = {
        "type": "split",
        "unit": "roi",
        "assignment_source": "custom",
        "units": [
            {"unit_index": 0, "partition": "validation"},
            {"unit_index": 1, "partition": "training"},
        ],
    }
    assert validate_training_split_plan(split, manifest, training_api_limits()) == split
    wrong = copy.deepcopy(split)
    wrong["requested_validation_percent"] = 20
    with pytest.raises(ValueError, match="missing or unknown"):
        validate_training_split_plan(wrong, manifest, training_api_limits())


def test_split_requires_complete_nonempty_partitions() -> None:
    manifest = two_image_manifest()
    incomplete = {
        "type": "split",
        "unit": "image",
        "assignment_source": "generated",
        "requested_validation_percent": 20,
        "units": [{"unit_index": 0, "partition": "training"}],
    }
    with pytest.raises(ValueError, match="assign every unit"):
        validate_training_split_plan(incomplete, manifest, training_api_limits())

    one_sided = copy.deepcopy(incomplete)
    one_sided["units"].append({"unit_index": 1, "partition": "training"})
    with pytest.raises(ValueError, match="nonempty training and validation"):
        validate_training_split_plan(one_sided, manifest, training_api_limits())


def test_project_group_split_requires_every_image_grouped() -> None:
    manifest = two_image_manifest()
    preimage = {
        key: value for key, value in manifest.items() if key != "dataset_digest"
    }
    del preimage["images"][1]["project_group_index"]
    ungrouped = build_training_dataset_manifest(preimage, training_api_limits())
    split = {
        "type": "split",
        "unit": "project_group",
        "assignment_source": "custom",
        "units": [{"unit_index": 0, "partition": "training"}],
    }
    with pytest.raises(
        ValueError, match="nonempty training and validation|every image"
    ):
        validate_training_split_plan(split, ungrouped, training_api_limits())


def test_no_validation_expands_every_roi_to_training() -> None:
    manifest = grouped_manifest([0, 0], [2, 1])

    assert expand_training_split_plan(
        {"type": "none"}, manifest, training_api_limits()
    ) == ExpandedTrainingSplit(
        training_roi_indices=(0, 1, 2),
        validation_roi_indices=(),
    )


def test_roi_split_expands_the_exact_stored_assignment() -> None:
    manifest = grouped_manifest([0, 0], [2, 1])
    split = {
        "type": "split",
        "unit": "roi",
        "assignment_source": "custom",
        "units": [
            {"unit_index": 0, "partition": "training"},
            {"unit_index": 1, "partition": "validation"},
            {"unit_index": 2, "partition": "training"},
        ],
    }

    assert expand_training_split_plan(
        split, manifest, training_api_limits()
    ) == ExpandedTrainingSplit(
        training_roi_indices=(0, 2),
        validation_roi_indices=(1,),
    )


def test_image_split_keeps_every_image_roi_together() -> None:
    manifest = grouped_manifest([0, 0], [2, 1])
    split = {
        "type": "split",
        "unit": "image",
        "assignment_source": "generated",
        "requested_validation_percent": 50,
        "units": [
            {"unit_index": 0, "partition": "validation"},
            {"unit_index": 1, "partition": "training"},
        ],
    }

    assert expand_training_split_plan(
        split, manifest, training_api_limits()
    ) == ExpandedTrainingSplit(
        training_roi_indices=(2,),
        validation_roi_indices=(0, 1),
    )


def test_project_group_split_keeps_every_group_roi_together() -> None:
    manifest = grouped_manifest([0, 0, 1], [2, 1, 2])
    split = {
        "type": "split",
        "unit": "project_group",
        "assignment_source": "custom",
        "units": [
            {"unit_index": 0, "partition": "training"},
            {"unit_index": 1, "partition": "validation"},
        ],
    }

    assert expand_training_split_plan(
        split, manifest, training_api_limits()
    ) == ExpandedTrainingSplit(
        training_roi_indices=(0, 1, 2),
        validation_roi_indices=(3, 4),
    )


def test_expansion_rejects_an_incomplete_assignment_before_returning_indices() -> None:
    manifest = grouped_manifest([0, 0], [2, 1])
    split = {
        "type": "split",
        "unit": "image",
        "assignment_source": "custom",
        "units": [{"unit_index": 0, "partition": "training"}],
    }

    with pytest.raises(ValueError, match="assign every unit"):
        expand_training_split_plan(split, manifest, training_api_limits())
