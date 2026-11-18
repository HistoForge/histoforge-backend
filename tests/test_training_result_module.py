from __future__ import annotations

import copy
import uuid
from pathlib import Path

import pytest
from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    parse_json,
    sha256_bytes,
)
from histoforge.cloud.architectures.dinov3_training import project_dinov3_training_model
from histoforge.cloud.architectures.registry import (
    FIELD_AGGREGATION_ID,
    FIELD_PREPROCESSING_ID,
    require_available_model,
    resolved_fixed_field_profile,
)
from histoforge.cloud.inference.activation import installed_model_descriptor
from histoforge.cloud.inference.plans import build_core_halo_profile
from histoforge.cloud.modules.identity import (
    build_stored_module_record,
    module_content_identity,
)
from histoforge.cloud.modules.store import ModuleStore
from histoforge.cloud.training.runs.request import training_run_request_digest
from histoforge.cloud.training.runs.result_module import (
    build_training_result_candidate,
    inspect_training_result_candidate,
    read_training_result_candidate,
)
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles
from tests.test_training_capabilities import _entry, _registry, _variable_capability
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import succeeded_report
from tests.test_training_source_preparation import (
    RUN_ID,
    install_module,
    prepare,
    request_for,
    workflow_recipe,
)

WORKSPACE_SOURCE_RUN_ID = "11111111-1111-4111-8111-111111111111"
REFINED_RUN_ID = "22222222-2222-4222-8222-222222222222"


def test_complete_result_replaces_only_selected_model_content(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(
        tmp_path, result_form="complete_source_workflow"
    )
    report, weights = _success(tmp_path, request, marker)

    candidate = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    recipe = parse_json((candidate.module_dir / "recipe.json").read_bytes())
    assert recipe["name"] == request["result_display_name"]
    assert recipe["upstream"] == f"owner/workflow@{'a' * 40}"
    assert recipe["models"]["nuclei"]["weights"] == {
        "type": "vendored",
        "file": "nuclei.safetensors",
    }
    assert (
        recipe["models"]["nuclei"]["params"]["min_instance_pixels"]
        == request["min_instance_pixels"]
    )
    assert recipe["models"]["nuclei"]["gpu"] == request["result_inference_gpu"]
    assert report["result_inference_gpu"] == request["result_inference_gpu"]
    assert recipe["models"]["cells"] == workflow_recipe()["models"]["cells"]
    assert recipe["steps"] == workflow_recipe()["steps"]
    assert (candidate.module_dir / "classify.groovy").read_bytes() == (
        source_root / "source/module/classify.groovy"
    ).read_bytes()
    assert (candidate.module_dir / "nuclei.safetensors").read_bytes() == (
        b"trained weights"
    )
    assert (candidate.module_dir / "cells.safetensors").read_bytes() == (
        b"cells weights"
    )
    assert candidate.report_digest == recipe["trainingReportDigest"]
    assert candidate.attestation == {
        "run_id": RUN_ID,
        "module_content_digest": candidate.module_content_digest,
        "report_digest": candidate.report_digest,
    }

    record = build_stored_module_record(
        candidate.module_dir,
        module_ref=candidate.module_ref,
        resolved_commit_sha=None,
        installed_at="2026-09-04T12:00:00.000Z",
        external_artifacts=candidate.external_artifacts,
        training_api_limits=training_api_limits(),
        training_result_attestation=candidate.attestation,
    )
    assert record["content_digest"] == candidate.module_content_digest


def test_complete_result_replaces_a_previous_training_report(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(
        tmp_path, result_form="complete_source_workflow"
    )
    first_report, weights = _success(tmp_path, request, marker)
    first = build_training_result_candidate(
        tmp_path / "first-candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=first_report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )
    store = ModuleStore(
        tmp_path / "trained-modules",
        max_module_bytes=1_000_000,
        max_installed_bytes=10_000_000,
        max_catalog_encoded_bytes=8 * 1024**2,
        training_api_limits=training_api_limits(),
    )
    record = store.install_from_directory(
        first.module_dir,
        module_ref=first.module_ref,
        resolved_commit_sha=None,
        job_id=str(uuid.uuid4()),
        publish=lambda: None,
        external_artifacts=first.external_artifacts,
        installed_at="2026-09-07T12:00:00.000Z",
        training_result_attestation=first.attestation,
    )
    second_request = copy.deepcopy(request)
    second_request["source_module"] = module_content_identity(record)
    second_request["result_display_name"] = "Refined workflow"
    second_root = tmp_path / "second"
    source_files = TrainingSourceFiles(second_root / "artifacts", 8 * 1024**2)
    with prepare(second_root, store).prepare(second_request) as prepared:
        second_marker = prepared.publish(REFINED_RUN_ID, "c" * 64)
    source_files.promote_candidate(REFINED_RUN_ID, "c" * 64)
    second_source = (
        second_root / "artifacts" / "training" / "runs" / "pending" / REFINED_RUN_ID
    )
    second_report, second_weights = _success(second_root, second_request, second_marker)
    second_report["run_id"] = REFINED_RUN_ID

    refined = build_training_result_candidate(
        second_root / "candidate",
        source_root=second_source,
        source_marker=second_marker,
        selected_weights=second_weights,
        succeeded_report=second_report,
        request=second_request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    assert (
        parse_json((refined.module_dir / "training-report.json").read_bytes())
        == second_report
    )
    assert refined.report_digest != first.report_digest


def test_standalone_result_omits_other_weights_but_retains_inactive_scripts(
    tmp_path: Path,
) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)

    candidate = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    recipe = parse_json((candidate.module_dir / "recipe.json").read_bytes())
    assert list(recipe["models"]) == ["nuclei"]
    assert recipe["steps"] == [{"model": "nuclei"}]
    assert sorted(path.name for path in candidate.module_dir.iterdir()) == [
        "classify.groovy",
        "nuclei.safetensors",
        "recipe.json",
        "training-report.json",
    ]


def test_complete_result_retains_a_referenced_artifact(tmp_path: Path) -> None:
    recipe = workflow_recipe()
    recipe["models"]["cells"]["weights"] = {
        "type": "referenced",
        "ref": "foundation/cells",
        "revision": "c" * 40,
    }
    retained_weights = tmp_path / "retained-cells.safetensors"
    retained_weights.write_bytes(b"retained cells weights")
    request, marker, source_root = _accepted_source(
        tmp_path,
        result_form="complete_source_workflow",
        recipe=recipe,
        external={("cells", "weights"): retained_weights},
    )
    report, weights = _success(tmp_path, request, marker)

    candidate = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    assert candidate.external_artifacts[("cells", "weights")].read_bytes() == (
        b"retained cells weights"
    )


def test_variable_result_is_available_with_its_exact_profile(tmp_path: Path) -> None:
    recipe = workflow_recipe()
    large_region = {
        "coreWidthModelPx": 8192,
        "coreHeightModelPx": 8192,
        "haloPerSideModelPx": 256,
    }
    recipe["models"]["nuclei"]["input"]["largeRegion"] = large_region
    recipe["models"]["nuclei"]["params"].update({"bsize": 384, "tile_overlap": 0.1})
    field_profile = {
        "planner": "variable-field-v2",
        "minSideModelPx": 384,
        "maxSideModelPx": 768,
        "preferredTileSideModelPx": 768,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": 8,
        "smallRoiContext": False,
        "tileOverlap": 0.25,
        "aggregation": FIELD_AGGREGATION_ID,
        "preprocessing": FIELD_PREPROCESSING_ID,
    }
    request, marker, source_root = _accepted_source(
        tmp_path,
        recipe=recipe,
        field_profile=field_profile,
    )
    report, weights = _success(tmp_path, request, marker)
    candidate = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    result_recipe = parse_json((candidate.module_dir / "recipe.json").read_bytes())
    model = result_recipe["models"]["nuclei"]
    assert model["input"]["fieldProfile"] == request["field_profile"]
    assert model["input"]["largeRegion"] == large_region
    assert model["params"]["min_instance_pixels"] == request["min_instance_pixels"]
    assert "bsize" not in model["params"]
    assert "tile_overlap" not in model["params"]
    record = build_stored_module_record(
        candidate.module_dir,
        module_ref=candidate.module_ref,
        resolved_commit_sha=None,
        installed_at="2026-09-06T12:00:00.000Z",
        external_artifacts=candidate.external_artifacts,
        training_api_limits=training_api_limits(),
        training_result_attestation=candidate.attestation,
    )
    descriptor = installed_model_descriptor(record, result_recipe, "nuclei")
    assert descriptor["runner_availability"]["type"] == "available"
    assert descriptor["model_config_digest"] == canonical_digest(model)
    profile = descriptor["large_region_profile"]
    assert profile == build_core_halo_profile(
        8192, 8192, 256, profile["validation_evidence_digest"]
    )

    # Editing the first run's sizes must not prevent another fine-tuning run.
    capability = _variable_capability()
    registry = _registry(_entry(capability))
    assert registry.availability(
        result_recipe, "nuclei", descriptor["runner_availability"]
    ) == {
        "type": "available",
        "options": [{"capability": capability, "class_initializations": ["use"]}],
    }
    assert (
        registry.resolve_entry(
            result_recipe, "nuclei", capability["capability_id"]
        ).capability
        == capability
    )
    assert recipe["models"]["nuclei"]["params"]["bsize"] == 384
    assert "fieldProfile" not in recipe["models"]["nuclei"]["input"]


def test_variable_result_requires_an_available_inference_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    field_profile = {
        "planner": "variable-field-v2",
        "minSideModelPx": 256,
        "maxSideModelPx": 1024,
        "preferredTileSideModelPx": 512,
        "alignmentModelPx": 16,
        "outerArtificialPaddingModelPx": 8,
        "smallRoiContext": True,
        "tileOverlap": 0.25,
        "aggregation": FIELD_AGGREGATION_ID,
        "preprocessing": FIELD_PREPROCESSING_ID,
    }
    request, marker, source_root = _accepted_source(
        tmp_path,
        field_profile=field_profile,
    )
    report, weights = _success(tmp_path, request, marker)

    def unavailable_after_admission(model: dict) -> dict:
        assert model["input"]["fieldProfile"] == field_profile
        raise ValueError("variable runner is no longer available")

    monkeypatch.setattr(
        "histoforge.cloud.training.runs.result_module.require_available_model",
        unavailable_after_admission,
    )
    with pytest.raises(ValueError, match="variable runner is no longer available"):
        build_training_result_candidate(
            tmp_path / "candidate",
            source_root=source_root,
            source_marker=marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            api_limits=training_api_limits(),
            max_source_marker_bytes=8 * 1024**2,
            project_model=project_dinov3_training_model,
        )


def test_complete_result_rechecks_retained_runners_after_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, marker, source_root = _accepted_source(
        tmp_path,
        result_form="complete_source_workflow",
    )
    report, weights = _success(tmp_path, request, marker)

    def changed_runner(model: dict) -> dict:
        # Admission already checked every step. Finalization must still verify
        # the complete result against the runner available when it is built.
        if model["output"]["classes"] == ["Cells"]:
            raise ValueError("retained cells runner is no longer available")
        return require_available_model(model)

    monkeypatch.setattr(
        "histoforge.cloud.training.runs.result_module.require_available_model",
        changed_runner,
    )
    with pytest.raises(ValueError, match="retained cells runner"):
        build_training_result_candidate(
            tmp_path / "candidate",
            source_root=source_root,
            source_marker=marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            api_limits=training_api_limits(),
            max_source_marker_bytes=8 * 1024**2,
            project_model=project_dinov3_training_model,
        )


def test_fixed_result_preserves_source_runner_fields(tmp_path: Path) -> None:
    recipe = workflow_recipe()
    large_region = {
        "coreWidthModelPx": 8192,
        "coreHeightModelPx": 8192,
        "haloPerSideModelPx": 256,
    }
    recipe["models"]["nuclei"]["input"]["largeRegion"] = large_region
    recipe["models"]["nuclei"]["params"].update({"bsize": 384, "tile_overlap": 0.1})
    request, marker, source_root = _accepted_source(tmp_path, recipe=recipe)
    report, weights = _success(tmp_path, request, marker)

    candidate = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    model = parse_json((candidate.module_dir / "recipe.json").read_bytes())["models"][
        "nuclei"
    ]
    assert model["params"]["bsize"] == 384
    assert model["params"]["tile_overlap"] == 0.1
    assert "fieldProfile" not in model["input"]
    assert model["input"]["largeRegion"] == large_region


def test_workspace_source_removes_stale_transitive_upstream(tmp_path: Path) -> None:
    recipe = workflow_recipe()
    recipe["upstream"] = f"older/source@{'c' * 40}"
    request, marker, source_root = _accepted_source(tmp_path, recipe=recipe)
    workspace_source = {
        key: value
        for key, value in request["source_module"].items()
        if key != "resolved_commit_sha"
    }
    workspace_source["module_ref"] = f"workspace/run-{WORKSPACE_SOURCE_RUN_ID}"
    request["source_module"] = workspace_source
    marker["source_module"] = request["source_module"]
    marker["request_digest"] = training_run_request_digest(
        request, training_api_limits()
    )
    (source_root / "ready.json").write_bytes(canonical_bytes(marker))
    report, weights = _success(tmp_path, request, marker)

    candidate = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    result_recipe = parse_json((candidate.module_dir / "recipe.json").read_bytes())
    result_report = parse_json(
        (candidate.module_dir / "training-report.json").read_bytes()
    )
    assert "upstream" not in result_recipe
    assert result_report["source_module"] == request["source_module"]


def test_candidate_rejects_weight_or_source_identity_drift(tmp_path: Path) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)
    weights.write_bytes(b"different")

    with pytest.raises(ValueError, match="trained weights differ"):
        build_training_result_candidate(
            tmp_path / "bad-weight",
            source_root=source_root,
            source_marker=marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            api_limits=training_api_limits(),
            max_source_marker_bytes=8 * 1024**2,
            project_model=project_dinov3_training_model,
        )

    weights.write_bytes(b"trained weights")
    changed_marker = copy.deepcopy(marker)
    changed_marker["source_module"]["content_digest"] = "f" * 64
    with pytest.raises(ValueError, match="source snapshot module identity"):
        build_training_result_candidate(
            tmp_path / "bad-source",
            source_root=source_root,
            source_marker=changed_marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            api_limits=training_api_limits(),
            max_source_marker_bytes=8 * 1024**2,
            project_model=project_dinov3_training_model,
        )

    changed_marker = copy.deepcopy(marker)
    changed_marker["request_digest"] = "f" * 64
    with pytest.raises(ValueError, match="request digest differs"):
        build_training_result_candidate(
            tmp_path / "bad-request",
            source_root=source_root,
            source_marker=changed_marker,
            selected_weights=weights,
            succeeded_report=report,
            request=request,
            api_limits=training_api_limits(),
            max_source_marker_bytes=8 * 1024**2,
            project_model=project_dinov3_training_model,
        )


def test_published_candidate_can_be_reverified_by_frozen_digests(
    tmp_path: Path,
) -> None:
    request, marker, source_root = _accepted_source(tmp_path)
    report, weights = _success(tmp_path, request, marker)
    built = build_training_result_candidate(
        tmp_path / "candidate",
        source_root=source_root,
        source_marker=marker,
        selected_weights=weights,
        succeeded_report=report,
        request=request,
        api_limits=training_api_limits(),
        max_source_marker_bytes=8 * 1024**2,
        project_model=project_dinov3_training_model,
    )

    reread = read_training_result_candidate(
        built.root,
        expected_run_id=RUN_ID,
        expected_module_content_digest=built.module_content_digest,
        expected_report_digest=built.report_digest,
        api_limits=training_api_limits(),
    )
    assert reread.attestation == built.attestation
    assert reread.report == report

    inspected = inspect_training_result_candidate(
        built.root,
        expected_run_id=RUN_ID,
        api_limits=training_api_limits(),
    )
    assert inspected.module_content_digest == built.module_content_digest
    assert inspected.report_digest == built.report_digest
    assert inspected.report == report

    (built.module_dir / "nuclei.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="selected weights|digest"):
        read_training_result_candidate(
            built.root,
            expected_run_id=RUN_ID,
            expected_module_content_digest=built.module_content_digest,
            expected_report_digest=built.report_digest,
            api_limits=training_api_limits(),
        )


def _accepted_source(
    tmp_path: Path,
    *,
    result_form: str = "selected_model_only",
    recipe: dict | None = None,
    external: dict[tuple[str, str], Path] | None = None,
    field_profile: dict | None = None,
    retained_files: dict[str, bytes] | None = None,
) -> tuple[dict, dict, Path]:
    store = install_module(
        tmp_path,
        recipe or workflow_recipe(),
        external=external,
        retained_files=retained_files,
    )
    request = request_for(store, result_form=result_form)
    source_recipe = store.current.load_recipe("owner/workflow")
    request["field_profile"] = field_profile or resolved_fixed_field_profile(
        source_recipe["models"][request["selected_model_id"]]
    )
    source_files = TrainingSourceFiles(tmp_path / "artifacts", 8 * 1024**2)
    with prepare(tmp_path, store).prepare(request) as prepared:
        marker = prepared.publish(RUN_ID, "b" * 64)
    source_files.promote_candidate(RUN_ID, "b" * 64)
    root = tmp_path / "artifacts/training/runs/pending" / RUN_ID
    return request, marker, root


def _success(
    tmp_path: Path,
    request: dict,
    marker: dict,
) -> tuple[dict, Path]:
    weights = tmp_path / "trained.safetensors"
    weights.write_bytes(b"trained weights")
    report = succeeded_report()
    report.update(
        {
            "run_id": RUN_ID,
            "dataset_digest": request["dataset_digest"],
            "split_plan": request["split_plan"],
            "source_module": request["source_module"],
            "selected_model_id": request["selected_model_id"],
            "starting_weight_digest": marker["starting_weight_digest"],
            "result_form": request["result_form"],
            "result_display_name": request["result_display_name"],
            "result_inference_gpu": request["result_inference_gpu"],
            "training_seed": request["training_seed"],
            "trainer_settings": request["trainer_settings"],
            "field_profile": request["field_profile"],
            "min_instance_pixels": request["min_instance_pixels"],
            "augmentation": request["augmentation"],
        }
    )
    report["task"].update(
        {
            "input_kind": "field",
            "model_output": {
                "type": "instance_segmentation",
                "classes": ["Nuclei"],
            },
            "model_microns_per_pixel": 0.22,
            "annotation_class_map": {"Nuclei": "Nuclei"},
        }
    )
    report["source_model_output"] = copy.deepcopy(report["task"]["model_output"])
    report["source_model_microns_per_pixel"] = 0.22
    for partition in ("training_counts", "validation_counts"):
        report[partition]["target_object_counts"] = {
            "Nuclei": report[partition]["target_object_counts"]["Leydig cell"]
        }
    report["trainer_configuration"]["preprocessing"]["profile_id"] = request[
        "field_profile"
    ]["preprocessing"]
    report["model_selection"]["selected_weight_digest"] = sha256_bytes(
        weights.read_bytes()
    )
    return report, weights
