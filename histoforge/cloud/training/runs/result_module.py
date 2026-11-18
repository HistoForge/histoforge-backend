"""Build and verify one complete run-derived module candidate.

This module owns the deterministic transformation from an accepted source
snapshot plus selected trained weights into the private module content that can
be installed by the sole Module Volume writer. It performs no Volume commits
and no training-run state transitions.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from histoforge.canonical_json import canonical_bytes, parse_json, require_sha256
from histoforge.cloud.architectures.registry import require_available_model
from histoforge.cloud.modules.files import (
    copy_exact_file,
    file_identity,
    module_file_paths,
    require_exact_file,
    require_file_closure,
)
from histoforge.cloud.modules.identity import build_stored_module_record
from histoforge.cloud.modules.schema import (
    TRAINING_REPORT_NAME,
    model_artifact_declarations,
    parse_module_recipe,
)
from histoforge.cloud.modules.training_evidence import (
    validate_training_result_attestation,
)
from histoforge.cloud.training.class_schema import require_training_source_evidence
from histoforge.cloud.training.contract import model_with_training_result_settings
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.report import (
    training_report_digest,
    validate_training_report,
)
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)
from histoforge.cloud.training.runs.source_snapshot import (
    validate_training_source_ready_marker,
)

_PREVIEW_INSTALLED_AT = "1970-01-01T00:00:00.000Z"


@dataclass(frozen=True)
class TrainingResultCandidate:
    """Verified private module candidate and its non-vendored artifact inputs."""

    root: Path
    module_dir: Path
    module_ref: str
    module_content_digest: str
    report_digest: str
    selected_weight_digest: str
    report: Mapping[str, Any]
    external_artifacts: Mapping[tuple[str, str], Path]
    attestation: Mapping[str, str]


def build_training_result_candidate(
    destination: str | Path,
    *,
    source_root: str | Path,
    source_marker: Any,
    selected_weights: str | Path,
    succeeded_report: Any,
    request: Any,
    api_limits: Any,
    max_source_marker_bytes: int,
    project_model: Callable[..., dict[str, Any]],
) -> TrainingResultCandidate:
    """Create one closed result candidate in an absent destination directory."""

    limits = validate_training_api_limits(api_limits)
    marker = validate_training_source_ready_marker(
        source_marker, max_source_marker_bytes
    )
    normalized_request = validate_training_run_request(request, limits)
    report = validate_training_report(succeeded_report, limits)
    _require_result_identity(report, normalized_request, marker, limits)

    source = Path(source_root)
    _require_source_closure(source, marker)
    source_module = source / "source" / "module"
    recipe = parse_module_recipe((source_module / "recipe.json").read_bytes())
    model_id = normalized_request["selected_model_id"]
    try:
        source_model = recipe["models"][model_id]
    except KeyError as error:
        raise ValueError("selected model is absent from the source snapshot") from error

    require_training_source_evidence(source_model, report)
    result_recipe = _result_recipe(
        recipe, normalized_request, report, limits, project_model
    )
    validated_models: set[str] = set()
    for step in result_recipe["steps"]:
        executable_model_id = step.get("model")
        if (
            executable_model_id is not None
            and executable_model_id not in validated_models
        ):
            require_available_model(result_recipe["models"][executable_model_id])
            validated_models.add(executable_model_id)

    target_root = Path(destination)
    target_root.mkdir(parents=True)
    module_dir = target_root / "module"
    module_dir.mkdir()

    old_selected_weight = (
        source_model["weights"]["file"]
        if source_model["weights"]["type"] == "vendored"
        else None
    )
    for identity in marker["source_files"]:
        relative = PurePosixPath(identity["relative_path"])
        if relative.parts[:2] != ("source", "module"):
            continue
        module_relative = PurePosixPath(*relative.parts[2:])
        if module_relative.as_posix() in {"recipe.json", TRAINING_REPORT_NAME} or (
            old_selected_weight is not None
            and module_relative.as_posix() == old_selected_weight
        ):
            continue
        target = module_dir.joinpath(*module_relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_exact_file(
            source.joinpath(*relative.parts),
            target,
            {
                **identity,
                "relative_path": module_relative.as_posix(),
            },
        )

    weight_path = Path(selected_weights)
    weight_identity = file_identity(f"{model_id}.safetensors", weight_path)
    selected_digest = report["model_selection"]["selected_weight_digest"]
    if weight_identity["sha256"] != selected_digest:
        raise ValueError("selected trained weights differ from the training report")
    copy_exact_file(
        weight_path,
        module_dir / weight_identity["relative_path"],
        weight_identity,
    )

    _write_fsynced(module_dir / "recipe.json", canonical_bytes(result_recipe))
    _write_fsynced(module_dir / TRAINING_REPORT_NAME, canonical_bytes(report))

    external = _copy_external_artifacts(
        source,
        target_root,
        marker,
        result_recipe,
    )
    return _candidate_from_root(target_root, external, report, limits)


def read_training_result_candidate(
    root: str | Path,
    *,
    expected_run_id: str,
    expected_module_content_digest: Any,
    expected_report_digest: Any,
    api_limits: Any,
) -> TrainingResultCandidate:
    """Verify an already published result candidate against frozen identities."""

    candidate = inspect_training_result_candidate(
        root,
        expected_run_id=expected_run_id,
        api_limits=api_limits,
    )
    if candidate.module_content_digest != require_sha256(
        expected_module_content_digest, "module_content_digest"
    ):
        raise ValueError("training result candidate module digest differs")
    if candidate.report_digest != require_sha256(
        expected_report_digest, "report_digest"
    ):
        raise ValueError("training result candidate report digest differs")
    return candidate


def inspect_training_result_candidate(
    root: str | Path,
    *,
    expected_run_id: str,
    api_limits: Any,
) -> TrainingResultCandidate:
    """Verify a published candidate and derive its own immutable identities."""

    limits = validate_training_api_limits(api_limits)
    candidate_root = Path(root)
    module_dir = candidate_root / "module"
    recipe = parse_module_recipe((module_dir / "recipe.json").read_bytes())
    expected_paths = [f"module/{path}" for path in module_file_paths(module_dir)]
    external = _external_artifact_paths(candidate_root, recipe)
    expected_paths.extend(
        _resolved_artifact_relative_path(model_id, role) for model_id, role in external
    )
    require_file_closure(candidate_root, expected_paths)

    report_path = module_dir / TRAINING_REPORT_NAME
    encoded_report = _read_bounded(report_path, limits["report"]["max_report_bytes"])
    report = validate_training_report(parse_json(encoded_report), limits)
    if encoded_report != canonical_bytes(report):
        raise ValueError("training result report is not canonical")
    if report["run_id"] != expected_run_id:
        raise ValueError("training result candidate belongs to another run")
    return _candidate_from_root(candidate_root, external, report, limits)


def _candidate_from_root(
    root: Path,
    external: Mapping[tuple[str, str], Path],
    report: Mapping[str, Any],
    api_limits: Any,
) -> TrainingResultCandidate:
    module_dir = root / "module"
    report_digest = training_report_digest(report, api_limits)
    module_ref = f"workspace/run-{report['run_id']}"
    preview = build_stored_module_record(
        module_dir,
        module_ref=module_ref,
        resolved_commit_sha=None,
        installed_at=_PREVIEW_INSTALLED_AT,
        external_artifacts=external,
        training_api_limits=api_limits,
    )
    module_digest = preview["content_digest"]
    attestation = validate_training_result_attestation(
        {
            "run_id": report["run_id"],
            "module_content_digest": module_digest,
            "report_digest": report_digest,
        }
    )
    return TrainingResultCandidate(
        root=root,
        module_dir=module_dir,
        module_ref=module_ref,
        module_content_digest=module_digest,
        report_digest=report_digest,
        selected_weight_digest=report["model_selection"]["selected_weight_digest"],
        report=report,
        external_artifacts=dict(external),
        attestation=attestation,
    )


def _result_recipe(
    source_recipe: Mapping[str, Any],
    request: Mapping[str, Any],
    report: Mapping[str, Any],
    api_limits: Any,
    project_model: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    recipe = {**source_recipe, "models": dict(source_recipe["models"])}
    model_id = request["selected_model_id"]
    model = model_with_training_result_settings(
        project_model(
            recipe["models"][model_id], report["task"], request["class_initialization"]
        ),
        request["field_profile"],
        request.get("min_instance_pixels"),
        request["result_inference_gpu"],
    )
    model["weights"] = {
        "type": "vendored",
        "file": f"{model_id}.safetensors",
    }
    recipe["models"][model_id] = model
    recipe["name"] = request["result_display_name"]
    recipe["trainingReportDigest"] = training_report_digest(report, api_limits)
    source = request["source_module"]
    if "resolved_commit_sha" in source:
        recipe["upstream"] = f"{source['module_ref']}@{source['resolved_commit_sha']}"
    else:
        recipe.pop("upstream", None)
    return parse_module_recipe(canonical_bytes(recipe))


def _copy_external_artifacts(
    source_root: Path,
    target_root: Path,
    marker: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[tuple[str, str], Path]:
    identities = {entry["relative_path"]: entry for entry in marker["source_files"]}
    result: dict[tuple[str, str], Path] = {}
    for declaration in model_artifact_declarations(recipe):
        source_type = declaration["source"]["type"]
        if source_type == "vendored":
            continue
        if source_type != "referenced":
            raise ValueError("training source contains an unsupported local artifact")
        model_id = declaration["model_id"]
        role = declaration["role"]
        relative = f"source/artifacts/{model_id}/{role}.safetensors"
        try:
            identity = identities[relative]
        except KeyError as error:
            raise ValueError(
                "source snapshot omits a required model artifact"
            ) from error
        destination_relative = _resolved_artifact_relative_path(model_id, role)
        destination = target_root.joinpath(*PurePosixPath(destination_relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy_exact_file(
            source_root.joinpath(*PurePosixPath(relative).parts),
            destination,
            {**identity, "relative_path": destination_relative},
        )
        result[(model_id, role)] = destination
    return result


def _external_artifact_paths(
    root: Path, recipe: Mapping[str, Any]
) -> dict[tuple[str, str], Path]:
    result = {}
    for declaration in model_artifact_declarations(recipe):
        source_type = declaration["source"]["type"]
        if source_type == "vendored":
            continue
        if source_type != "referenced":
            raise ValueError("training result contains an unsupported local artifact")
        key = (declaration["model_id"], declaration["role"])
        result[key] = root.joinpath(
            *PurePosixPath(_resolved_artifact_relative_path(*key)).parts
        )
    return result


def _resolved_artifact_relative_path(model_id: str, role: str) -> str:
    return f"resolved/{model_id}/{role}.safetensors"


def _require_result_identity(
    report: Mapping[str, Any],
    request: Mapping[str, Any],
    marker: Mapping[str, Any],
    api_limits: Any,
) -> None:
    if report["state"] != "SUCCEEDED":
        raise ValueError("result candidate requires a successful training report")
    expected = {
        "dataset_digest": request["dataset_digest"],
        "source_module": request["source_module"],
        "selected_model_id": request["selected_model_id"],
        "class_initialization": request["class_initialization"],
        "result_form": request["result_form"],
        "result_display_name": request["result_display_name"],
        "result_inference_gpu": request["result_inference_gpu"],
        "split_plan": request["split_plan"],
        "field_profile": request["field_profile"],
        "min_instance_pixels": request.get("min_instance_pixels"),
        "augmentation": request["augmentation"],
        "trainer_settings": request["trainer_settings"],
        "training_seed": request["training_seed"],
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"training report {field} differs from the request")
    if marker["run_id"] != report["run_id"]:
        raise ValueError("source snapshot belongs to another training run")
    if marker["request_digest"] != training_run_request_digest(request, api_limits):
        raise ValueError("source snapshot request digest differs from the request")
    if marker["source_module"] != request["source_module"]:
        raise ValueError("source snapshot module identity differs from the request")
    if marker["starting_weight_digest"] != report["starting_weight_digest"]:
        raise ValueError("source snapshot starting weights differ from the report")


def _require_source_closure(root: Path, marker: Mapping[str, Any]) -> None:
    expected = [entry["relative_path"] for entry in marker["source_files"]]
    expected.append("ready.json")
    require_file_closure(root, expected)
    if (root / "ready.json").read_bytes() != canonical_bytes(marker):
        raise ValueError("source ready marker bytes differ")
    for identity in marker["source_files"]:
        relative = PurePosixPath(identity["relative_path"])
        require_exact_file(
            root.joinpath(*relative.parts), identity, relative.as_posix()
        )


def _read_bounded(path: Path, maximum: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
        if len(value) > maximum or stream.read(1):
            raise ValueError("training report exceeds its byte limit")
    return value


def _write_fsynced(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
