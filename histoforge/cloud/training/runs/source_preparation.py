"""Prepare the exact installed-module closure used by one training run."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_sha256,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.architectures.registry import require_available_model
from histoforge.cloud.inference.gpu import require_supported_inference_gpu
from histoforge.cloud.modules.current import CurrentModules
from histoforge.cloud.modules.files import file_identity
from histoforge.cloud.modules.identity import (
    module_content_file_identities,
    module_content_identity,
)
from histoforge.cloud.modules.schema import (
    TRAINING_REPORT_NAME,
    model_artifact_declarations,
    parse_module_recipe,
)
from histoforge.cloud.training.contract import model_with_training_result_settings
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.request import (
    training_run_request_digest,
    validate_training_run_request,
)
from histoforge.cloud.training.runs.source_files import TrainingSourceFiles


class TrainingSourcePreparation:
    """Resolve an installed source and publish one result-form-specific snapshot."""

    def __init__(
        self,
        current_modules: Callable[[], AbstractContextManager[CurrentModules]],
        files: TrainingSourceFiles,
        api_limits: Any,
    ) -> None:
        self._current_modules = current_modules
        self._files = files
        self._api_limits = validate_training_api_limits(api_limits)

    @contextmanager
    def prepare(
        self,
        request: Any,
    ) -> Iterator[PreparedTrainingSource]:
        """Resolve one exact source subset before reserving a training run."""

        normalized = validate_training_run_request(request, self._api_limits)
        module_ref = normalized["source_module"]["module_ref"]
        with self._current_modules() as modules:
            try:
                record = modules.revalidate(module_ref)
            except FileNotFoundError as error:
                raise GatewayFailure(
                    GatewayErrorCode.MODULE_NOT_FOUND,
                    "the selected source module is not installed",
                    resource_id=module_ref,
                ) from error
            if module_content_identity(record) != normalized["source_module"]:
                raise GatewayFailure(
                    GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
                    "the selected source module changed after it was selected",
                    resource_id=module_ref,
                )
            recipe = modules.load_recipe(module_ref, record=record)
            model_id = normalized["selected_model_id"]
            if model_id not in recipe["models"] or not any(
                step.get("model") == model_id for step in recipe["steps"]
            ):
                raise GatewayFailure(
                    GatewayErrorCode.VALIDATION_ERROR,
                    "the selected model is not an executable step in the source module",
                    resource_id=f"{module_ref}/{model_id}",
                )
            _require_result_form(
                recipe,
                model_id,
                normalized["result_form"],
                normalized["field_profile"],
                normalized.get("min_instance_pixels"),
                normalized["result_inference_gpu"],
            )

            with tempfile.TemporaryDirectory(
                prefix="histoforge-training-source-"
            ) as temp:
                source_paths, identities, starting_path = self._source_files(
                    modules,
                    Path(temp),
                    recipe,
                    record,
                    model_id,
                    normalized["result_form"],
                )
                starting = next(
                    identity
                    for identity in identities
                    if identity["relative_path"] == starting_path
                )
                # PreparedTrainingSource still names mounted files. Keep the view
                # stable through the caller's source publication, not just capture.
                yield PreparedTrainingSource(
                    request=normalized,
                    recipe=recipe,
                    starting_weight_digest=starting["sha256"],
                    _starting_weight_path=starting_path,
                    _source_files=tuple(identities),
                    _source_paths=source_paths,
                    _files=self._files,
                    _api_limits=self._api_limits,
                )

    def _source_files(
        self,
        modules: CurrentModules,
        temp_root: Path,
        recipe: Mapping[str, Any],
        record: Mapping[str, Any],
        model_id: str,
        result_form: str,
    ) -> tuple[dict[str, Path], list[dict[str, Any]], str]:
        module_ref = record["module_ref"]
        source_paths: dict[str, Path] = {}
        identities: list[dict[str, Any]] = []
        snapshot_recipe = (
            dict(recipe)
            if result_form == "complete_source_workflow"
            else _minimal_recipe(recipe, model_id)
        )
        # The previous report describes previous weights, not this new result.
        snapshot_recipe.pop("trainingReportDigest", None)
        artifact_models = set(snapshot_recipe["models"])
        excluded_weights = {
            item["source"]["file"]
            for item in model_artifact_declarations(recipe)
            if item["source"]["type"] == "vendored"
            and item["model_id"] not in artifact_models
        }
        module_dir = modules.module_dir(module_ref)
        for identity in module_content_file_identities(module_dir, recipe, record):
            path = identity["relative_path"]
            if path in excluded_weights | {"recipe.json", TRAINING_REPORT_NAME}:
                continue
            _add_source(
                source_paths,
                identities,
                f"source/module/{path}",
                module_dir / path,
                identity,
            )
        recipe_path = temp_root / "recipe.json"
        _write_fsynced(recipe_path, canonical_bytes(snapshot_recipe))
        _add_source(source_paths, identities, "source/module/recipe.json", recipe_path)

        starting_path = ""
        for artifact in record["resolved_artifacts"]:
            if artifact["model_id"] not in artifact_models:
                continue
            if artifact["artifact_source_type"] == "vendored":
                if artifact["model_id"] == model_id and artifact["role"] == "weights":
                    declaration = recipe["models"][model_id]["weights"]
                    starting_path = f"source/module/{declaration['file']}"
                continue
            relative_path = _artifact_snapshot_path(
                artifact["model_id"], artifact["role"]
            )
            source = modules.artifact_path(
                module_ref, artifact["model_id"], artifact["role"]
            )
            _add_source(source_paths, identities, relative_path, source, artifact)
            if artifact["model_id"] == model_id and artifact["role"] == "weights":
                starting_path = relative_path

        if not starting_path:
            raise ValueError("selected model has no resolved starting weights")
        identities.sort(key=lambda identity: identity["relative_path"])
        return source_paths, identities, starting_path


@dataclass(frozen=True)
class PreparedTrainingSource:
    """One temporary exact source closure that may be published before exit."""

    request: dict[str, Any]
    recipe: dict[str, Any]
    starting_weight_digest: str
    _starting_weight_path: str = field(repr=False)
    _source_files: tuple[dict[str, Any], ...] = field(repr=False)
    _source_paths: Mapping[str, Path] = field(repr=False)
    _files: TrainingSourceFiles = field(repr=False)
    _api_limits: dict[str, Any] = field(repr=False)

    def publish(self, run_id: Any, report_digest: Any) -> dict[str, Any]:
        """Publish a source candidate bound to report revision 1."""

        marker = {
            "schema_version": 1,
            "run_id": run_id,
            "request_digest": training_run_request_digest(
                self.request, self._api_limits
            ),
            "source_module": self.request["source_module"],
            "starting_weight_path": self._starting_weight_path,
            "starting_weight_digest": self.starting_weight_digest,
            "source_files": list(self._source_files),
            "report_digest": require_sha256(report_digest, "report_digest"),
        }
        return self._files.publish_candidate(marker, self._source_paths)


def _minimal_recipe(recipe: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    has_only_selected_model = len(recipe["models"]) == 1
    has_script = any("script" in step for step in recipe["steps"])
    result: dict[str, Any] = {
        "schemaVersion": recipe["schemaVersion"],
        "name": recipe["name"],
        "models": {model_id: recipe["models"][model_id]},
        "steps": [{"model": model_id}],
    }
    if has_only_selected_model and not has_script and "description" in recipe:
        result["description"] = recipe["description"]
    return parse_module_recipe(canonical_bytes(canonical_tree(result)))


def _require_result_form(
    recipe: Mapping[str, Any],
    model_id: str,
    result_form: str,
    field_profile: Mapping[str, Any],
    min_instance_pixels: int | None,
    result_inference_gpu: str,
) -> None:
    model = recipe["models"][model_id]
    if model["input"]["type"] == "crop" and result_form != "complete_source_workflow":
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "a region-gated model requires a complete source workflow result",
        )
    steps = recipe["steps"]
    if (
        len(recipe["models"]) == 1
        and len(steps) == 1
        and steps[0] == {"model": model_id}
        and result_form != "selected_model_only"
    ):
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "a one-model source requires a standalone model result",
        )
    try:
        require_available_model(model)
        require_supported_inference_gpu(result_inference_gpu, model["gpu"])
        require_available_model(
            model_with_training_result_settings(
                model,
                field_profile,
                min_instance_pixels,
                result_inference_gpu,
            )
        )
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            f"selected model {model_id!r} or its requested result settings "
            f"cannot run: {error}",
            resource_id=model_id,
        ) from error
    if result_form == "complete_source_workflow":
        retained_ids = {step["model"] for step in steps if "model" in step}
        for retained_id in retained_ids - {model_id}:
            try:
                require_available_model(recipe["models"][retained_id])
            except ValueError as error:
                raise GatewayFailure(
                    GatewayErrorCode.VALIDATION_ERROR,
                    f"copied workflow model {retained_id!r} cannot run: {error}",
                    resource_id=retained_id,
                ) from error


def _add_source(
    sources: dict[str, Path],
    identities: list[dict[str, Any]],
    relative_path: str,
    source: Path,
    expected: Mapping[str, Any] | None = None,
) -> None:
    identity = file_identity(relative_path, source)
    if expected is not None and (
        identity["size_bytes"] != expected["size_bytes"]
        or identity["sha256"] != expected["sha256"]
    ):
        raise ValueError("source file differs from its installed identity")
    if relative_path in sources:
        raise ValueError("training source path is duplicated")
    sources[relative_path] = source
    identities.append(identity)


def _artifact_snapshot_path(model_id: str, role: str) -> str:
    return f"source/artifacts/{model_id}/{role}.safetensors"


def _write_fsynced(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
