"""Exact installed-module identity and directory-closure tests."""

import copy
import json
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest, sha256_bytes
from histoforge.cloud.modules.identity import (
    STORED_MODULE_RECORD_PATH,
    build_stored_module_record,
    module_content_identity,
    validate_module_directory,
    validate_stored_module_record,
    write_stored_module_record,
)
from histoforge.cloud.modules.training_evidence import validate_module_training_report
from tests.test_training_dataset import training_api_limits
from tests.test_training_report import succeeded_report

COMMIT = "1" * 40
INSTALLED_AT = "2026-08-15T12:34:56.789Z"


def _recipe(weights=None):
    return {
        "schemaVersion": 3,
        "name": "Example",
        "models": {
            "model": {
                "arch": "dinov3-adapter-cellpose",
                "gpu": "A10G",
                "weights": weights or {"type": "vendored", "file": "model.safetensors"},
                "input": {"type": "field", "micronsPerPixel": 0.22},
                "output": {
                    "type": "instance_segmentation",
                    "classes": ["Cell"],
                },
                "params": {"num_classes": 0, "min_instance_pixels": 15},
            }
        },
        "steps": [{"model": "model"}],
    }


def _module(directory: Path, *, recipe=None, weights=b"weights") -> Path:
    directory.mkdir(parents=True)
    (directory / "recipe.json").write_text(
        json.dumps(recipe or _recipe()), encoding="utf-8"
    )
    if weights is not None:
        (directory / "model.safetensors").write_bytes(weights)
    return directory


def _build(directory: Path, **kwargs):
    return build_stored_module_record(
        directory,
        module_ref=kwargs.pop("module_ref", "owner/module"),
        resolved_commit_sha=kwargs.pop("resolved_commit_sha", COMMIT),
        installed_at=kwargs.pop("installed_at", INSTALLED_AT),
        **kwargs,
    )


def _training_result_module(directory: Path) -> tuple[Path, dict, dict]:
    weights = b"trained weights"
    report = succeeded_report()
    report["selected_model_id"] = "model"
    report["model_selection"]["selected_weight_digest"] = sha256_bytes(weights)
    report["task"]["model_microns_per_pixel"] = 0.22
    report["task"]["model_output"] = {
        "type": "instance_segmentation",
        "classes": ["Cell"],
    }
    report["task"]["annotation_class_map"] = {"Cell": "Cell"}
    report["source_model_output"] = copy.deepcopy(report["task"]["model_output"])
    report["source_model_microns_per_pixel"] = 0.22
    for partition in ("training_counts", "validation_counts"):
        report[partition]["target_object_counts"] = {
            "Cell": report[partition]["target_object_counts"]["Leydig cell"]
        }
    report["field_profile"] = {
        "planner": "fixed-field-v1",
        "side_model_px": 256,
        "alignment_model_px": 16,
        "small_roi_context": False,
        "tile_overlap": 0.1,
        "aggregation": "cellpose-tapered-mean-v1",
        "preprocessing": "normalize99-real-image-v1",
    }
    report["trainer_configuration"]["preprocessing"]["profile_id"] = (
        "normalize99-real-image-v1"
    )
    recipe = _recipe()
    recipe["models"]["model"]["gpu"] = report["result_inference_gpu"]
    recipe["trainingReportDigest"] = canonical_digest(report)
    module = _module(directory, recipe=recipe, weights=weights)
    (module / "training-report.json").write_bytes(canonical_bytes(report))
    preview = _build(
        module,
        module_ref=f"workspace/run-{report['run_id']}",
        resolved_commit_sha=None,
        training_api_limits=training_api_limits(),
    )
    attestation = {
        "run_id": report["run_id"],
        "module_content_digest": preview["content_digest"],
        "report_digest": recipe["trainingReportDigest"],
    }
    return module, report, attestation


def test_builds_exact_sorted_file_artifact_and_content_identity(tmp_path):
    directory = _module(tmp_path / "module")

    record = _build(directory)

    assert record["schema_version"] == 1
    assert record["module_ref"] == "owner/module"
    assert record["resolved_commit_sha"] == COMMIT
    assert record["recipe_digest"] == canonical_digest(_recipe())
    assert record["executable_files"] == [
        {
            "relative_path": "model.safetensors",
            "size_bytes": 7,
            "sha256": sha256_bytes(b"weights"),
        },
        {
            "relative_path": "recipe.json",
            "size_bytes": (directory / "recipe.json").stat().st_size,
            "sha256": sha256_bytes((directory / "recipe.json").read_bytes()),
        },
    ]
    assert record["resolved_artifacts"] == [
        {
            "model_id": "model",
            "role": "weights",
            "artifact_source_type": "vendored",
            "artifact_source_ref": None,
            "resolved_revision": None,
            "size_bytes": 7,
            "sha256": sha256_bytes(b"weights"),
        }
    ]
    assert record["content_digest"] == canonical_digest(
        {
            "recipe_digest": record["recipe_digest"],
            "executable_files": record["executable_files"],
            "resolved_artifacts": record["resolved_artifacts"],
        }
    )


def test_weight_identity_is_streamed_without_path_read_bytes(tmp_path, monkeypatch):
    directory = _module(tmp_path / "module", weights=b"x" * (2 * 1024 * 1024 + 7))
    original_read_bytes = Path.read_bytes

    def reject_weight_materialization(path):
        if path.suffix == ".safetensors":
            raise AssertionError("weight identity must not materialize the whole file")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_weight_materialization)

    record = _build(directory)

    assert record["resolved_artifacts"][0]["size_bytes"] == 2 * 1024 * 1024 + 7


def test_record_roundtrip_rehashes_directory_and_projects_shared_identity(tmp_path):
    directory = _module(tmp_path / "module")
    record = _build(directory)
    write_stored_module_record(directory, record)

    assert (
        validate_module_directory(directory, expected_module_ref="owner/module")
        == record
    )
    assert (directory / STORED_MODULE_RECORD_PATH).read_bytes() == canonical_bytes(
        record
    )
    assert module_content_identity(record) == {
        "module_ref": "owner/module",
        "resolved_commit_sha": COMMIT,
        "recipe_digest": record["recipe_digest"],
        "content_digest": record["content_digest"],
    }


def test_content_digest_is_byte_identity_not_ref_commit_or_install_time(tmp_path):
    directory = _module(tmp_path / "module")
    first = _build(directory)
    second = _build(
        directory,
        module_ref="other/module",
        resolved_commit_sha="2" * 40,
        installed_at="2026-08-15T12:35:00.000Z",
    )

    assert first["content_digest"] == second["content_digest"]
    assert first["module_ref"] != second["module_ref"]
    assert first["resolved_commit_sha"] != second["resolved_commit_sha"]


def test_referenced_artifact_binds_role_ref_revision_and_exact_bytes(tmp_path):
    recipe = _recipe(
        {
            "type": "referenced",
            "ref": "foundation/encoder",
            "revision": "3" * 40,
        }
    )
    directory = _module(tmp_path / "module", recipe=recipe, weights=None)
    artifact = tmp_path / "encoder.safetensors"
    artifact.write_bytes(b"encoder")

    record = _build(directory, external_artifacts={("model", "weights"): artifact})

    assert record["executable_files"] == [
        {
            "relative_path": "recipe.json",
            "size_bytes": (directory / "recipe.json").stat().st_size,
            "sha256": sha256_bytes((directory / "recipe.json").read_bytes()),
        }
    ]
    assert record["resolved_artifacts"] == [
        {
            "model_id": "model",
            "role": "weights",
            "artifact_source_type": "referenced",
            "artifact_source_ref": "foundation/encoder",
            "resolved_revision": "3" * 40,
            "size_bytes": 7,
            "sha256": sha256_bytes(b"encoder"),
        }
    ]


def test_missing_or_extra_external_artifact_inputs_fail(tmp_path):
    recipe = _recipe(
        {
            "type": "referenced",
            "ref": "foundation/encoder",
            "revision": "3" * 40,
        }
    )
    directory = _module(tmp_path / "module", recipe=recipe, weights=None)
    artifact = tmp_path / "encoder.safetensors"
    artifact.write_bytes(b"encoder")

    with pytest.raises(ValueError, match="missing resolved artifact"):
        _build(directory)
    with pytest.raises(ValueError, match="unexpected resolved artifact"):
        _build(
            directory,
            external_artifacts={
                ("model", "weights"): artifact,
                ("model", "backbone.weights"): artifact,
            },
        )


def test_tampered_executable_or_record_fails_closed(tmp_path):
    directory = _module(tmp_path / "module")
    record = _build(directory)
    write_stored_module_record(directory, record)
    (directory / "model.safetensors").write_bytes(b"changed")

    with pytest.raises(ValueError, match="does not match installed bytes"):
        validate_module_directory(directory)

    (directory / "model.safetensors").write_bytes(b"weights")
    mutated = copy.deepcopy(record)
    mutated["content_digest"] = "f" * 64
    (directory / STORED_MODULE_RECORD_PATH).write_bytes(canonical_bytes(mutated))
    with pytest.raises(ValueError, match="content_digest"):
        validate_module_directory(directory)


def test_noncanonical_record_bytes_fail(tmp_path):
    directory = _module(tmp_path / "module")
    record = _build(directory)
    path = directory / STORED_MODULE_RECORD_PATH
    path.parent.mkdir()
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical JSON"):
        validate_module_directory(directory)


def test_additional_files_are_retained_but_empty_directories_are_rejected(tmp_path):
    directory = _module(tmp_path / "module")
    (directory / "README.md").write_text("retained", encoding="utf-8")
    record = _build(directory)
    assert record["additional_files"] == [
        {
            "relative_path": "README.md",
            "size_bytes": 8,
            "sha256": sha256_bytes(b"retained"),
        }
    ]

    (directory / "README.md").unlink()
    (directory / "unused").mkdir()
    with pytest.raises(ValueError, match="directory closure mismatch"):
        _build(directory)


def test_symlink_and_casefold_collision_fail_before_identity(tmp_path):
    directory = _module(tmp_path / "module")
    try:
        (directory / "alias.safetensors").symlink_to("model.safetensors")
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with pytest.raises(ValueError, match="link|closure|non-regular"):
        _build(directory)

    (directory / "alias.safetensors").unlink()
    (directory / "Recipe.json").write_bytes(b"different case")
    if (directory / "recipe.json").read_bytes() == b"different case":
        pytest.skip("case-distinct filenames unavailable on this filesystem")
    with pytest.raises(ValueError, match="closure|case-fold"):
        _build(directory)


def test_record_validator_rejects_wrong_source_shape_and_order(tmp_path):
    record = _build(_module(tmp_path / "module"))

    wrong_source = copy.deepcopy(record)
    wrong_source["resolved_artifacts"][0]["artifact_source_ref"] = "owner/source"
    with pytest.raises(ValueError, match="vendored artifact"):
        validate_stored_module_record(wrong_source)

    wrong_order = copy.deepcopy(record)
    wrong_order["executable_files"].reverse()
    wrong_order["content_digest"] = canonical_digest(
        {
            "recipe_digest": wrong_order["recipe_digest"],
            "executable_files": wrong_order["executable_files"],
            "resolved_artifacts": wrong_order["resolved_artifacts"],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        validate_stored_module_record(wrong_order)


def test_training_result_report_and_attestation_are_bound_to_module_bytes(tmp_path):
    module, report, attestation = _training_result_module(tmp_path / "result")
    ref = f"workspace/run-{report['run_id']}"

    record = _build(
        module,
        module_ref=ref,
        resolved_commit_sha=None,
        training_api_limits=training_api_limits(),
        training_result_attestation=attestation,
    )
    write_stored_module_record(module, record)

    assert record["training_result_attestation"] == attestation
    assert record["resolved_commit_sha"] is None
    assert (
        validate_module_directory(
            module,
            expected_module_ref=ref,
            training_api_limits=training_api_limits(),
        )
        == record
    )
    assert "training-report.json" not in {
        item["relative_path"] for item in record["executable_files"]
    }


def test_training_report_read_is_bounded_before_parsing(tmp_path):
    module = tmp_path / "module"
    module.mkdir()
    encoded = b"x" * 33
    (module / "training-report.json").write_bytes(encoded)
    limits = training_api_limits()
    limits["report"]["max_report_bytes"] = 32

    with pytest.raises(ValueError, match="max_report_bytes"):
        validate_module_training_report(
            module,
            {"trainingReportDigest": sha256_bytes(encoded)},
            [],
            limits,
        )


def test_training_report_model_output_matches_recipe_exactly(tmp_path):
    module, _report, _attestation = _training_result_module(tmp_path / "result")
    recipe = json.loads((module / "recipe.json").read_text(encoding="utf-8"))
    recipe["models"]["model"]["output"]["classes"] = ["Different cell"]
    (module / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")

    with pytest.raises(ValueError, match="model output does not match"):
        _build(module, training_api_limits=training_api_limits())


def test_training_report_minimum_instance_area_matches_recipe(tmp_path):
    module, _report, _attestation = _training_result_module(tmp_path / "result")
    recipe = json.loads((module / "recipe.json").read_text(encoding="utf-8"))
    recipe["models"]["model"]["params"]["min_instance_pixels"] = 0
    (module / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")

    with pytest.raises(ValueError, match="minimum instance area"):
        _build(module, training_api_limits=training_api_limits())


def test_training_report_profile_schema_is_checked_after_digest_binding(tmp_path):
    module, report, _attestation = _training_result_module(tmp_path / "result")
    report["trainer_configuration"]["optimizer"]["parameters"]["api_key"] = "secret"
    recipe = json.loads((module / "recipe.json").read_text(encoding="utf-8"))
    recipe["trainingReportDigest"] = canonical_digest(report)
    (module / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    (module / "training-report.json").write_bytes(canonical_bytes(report))

    with pytest.raises(ValueError, match="unknown fields"):
        _build(module, training_api_limits=training_api_limits())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda module, report, attestation: (
                module / "training-report.json"
            ).write_bytes(b"{}"),
            "training report digest",
        ),
        (
            lambda module, report, attestation: attestation.update(
                {"run_id": "55555555-5555-4555-8555-555555555555"}
            ),
            "run_id",
        ),
        (
            lambda module, report, attestation: attestation.update(
                {"module_content_digest": "e" * 64}
            ),
            "content_digest",
        ),
    ],
)
def test_training_result_identity_mismatches_fail_closed(tmp_path, mutate, message):
    module, report, attestation = _training_result_module(tmp_path / "result")
    mutate(module, report, attestation)

    with pytest.raises(ValueError, match=message):
        _build(
            module,
            module_ref=f"workspace/run-{report['run_id']}",
            resolved_commit_sha=None,
            training_api_limits=training_api_limits(),
            training_result_attestation=attestation,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("side_model_px", 384),
        ("aggregation", "different-aggregation-v1"),
    ],
)
def test_training_result_fixed_field_profile_matches_selected_runner(
    tmp_path, field, value
):
    module, report, _attestation = _training_result_module(tmp_path / "result")
    report["field_profile"][field] = value
    recipe = json.loads((module / "recipe.json").read_text(encoding="utf-8"))
    recipe["trainingReportDigest"] = canonical_digest(report)
    (module / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
    (module / "training-report.json").write_bytes(canonical_bytes(report))

    with pytest.raises(ValueError, match="fixed field profile"):
        _build(
            module,
            training_api_limits=training_api_limits(),
        )
