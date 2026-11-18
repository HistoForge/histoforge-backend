"""Admission-time resolution for one executable managed-training run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from histoforge.canonical_json import canonical_tree
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.architectures.registry import require_available_model
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.artifacts.volume import artifact_volume_gate
from histoforge.cloud.modules.schema import model_output_target_classes
from histoforge.cloud.training.capabilities import (
    TrainingCapabilityEntry,
    TrainingCapabilityRegistry,
)
from histoforge.cloud.training.contract import model_with_training_result_settings
from histoforge.cloud.training.datasets.archive import ExtractedTrainingDataset
from histoforge.cloud.training.datasets.content import (
    AdmittedTrainingDataset,
    admit_extracted_training_dataset,
)
from histoforge.cloud.training.datasets.files import (
    RetainedTrainingDataset,
    TrainingDatasetFiles,
)
from histoforge.cloud.training.datasets.label_rasterizer import TrainingRoiLabelSummary
from histoforge.cloud.training.datasets.split import (
    ExpandedTrainingSplit,
    expand_training_split_plan,
)
from histoforge.cloud.training.implementations import resolve_training_implementation
from histoforge.cloud.training.limits import validate_training_api_limits
from histoforge.cloud.training.runs.report import validate_training_report
from histoforge.cloud.training.runs.request import (
    validate_training_run_request,
    validate_training_run_request_against_manifest_capability,
)


@dataclass(frozen=True)
class PreparedManagedTrainingRun:
    """Complete immutable preflight result consumed by run admission."""

    request: Mapping[str, Any]
    starting_weight_digest: str
    trainer_build: Mapping[str, Any]
    total_epochs: int
    trainer_execution_seconds: int
    worker_resources: Mapping[str, Any]
    _initial_report: Mapping[str, Any]
    _api_limits: Mapping[str, Any]

    def initial_report(self, run_id: str) -> Mapping[str, Any]:
        """Bind the prevalidated QUEUED report template to its allocated run ID."""

        return validate_training_report(
            {**self._initial_report, "run_id": run_id}, self._api_limits
        )


class TrainingRunPreparation:
    """Bind source, retained dataset, split, trainer profile, and report."""

    def __init__(
        self,
        dataset_files: TrainingDatasetFiles,
        dataset_control: Any,
        capabilities: TrainingCapabilityRegistry,
        trainer_build: Callable[[TrainingCapabilityEntry], Mapping[str, Any]],
        api_limits: Any,
        source_limits: Any,
    ) -> None:
        self._dataset_files = dataset_files
        self._dataset_control = dataset_control
        self._capabilities = capabilities
        self._trainer_build = trainer_build
        self._api_limits = validate_training_api_limits(api_limits)
        self._source_limits = source_limits

    def __call__(
        self,
        value: Mapping[str, Any],
        recipe: Mapping[str, Any],
        starting_weight_digest: str,
    ) -> PreparedManagedTrainingRun:
        """Resolve every mutable dependency before the run becomes visible."""

        request = validate_training_run_request(value, self._api_limits)
        retained = self._retained_dataset(request)
        capability_id = retained.manifest["task"]["capability_id"]
        entry = self._capabilities.resolve_entry(
            recipe,
            request["selected_model_id"],
            capability_id,
        )
        model = recipe["models"][request["selected_model_id"]]
        implementation = resolve_training_implementation(
            entry.trainer_implementation_id
        )
        target_model = implementation.project_model(
            model, retained.manifest["task"], request["class_initialization"]
        )
        require_available_model(
            model_with_training_result_settings(
                target_model,
                request["field_profile"],
                request.get("min_instance_pixels"),
                request["result_inference_gpu"],
            )
        )
        validate_training_run_request_against_manifest_capability(
            request,
            retained.manifest,
            entry.capability,
            self._api_limits,
        )
        # Admission still opens mounted source and label files. Prevent another
        # gateway request from reloading this mount before those reads finish.
        with artifact_volume_gate:
            admitted = admit_extracted_training_dataset(
                ExtractedTrainingDataset(
                    retained.root,
                    retained.manifest,
                    retained.retained_bytes,
                ),
                self._api_limits,
                self._source_limits,
                self._capabilities.resolve,
            )
        split = expand_training_split_plan(
            request["split_plan"], admitted.manifest, self._api_limits
        )
        profile = entry.trainer_profile
        configuration = implementation.prepare_configuration(
            entry,
            request=request,
            model_output=target_model["output"],
            dataset=retained,
            training_roi_indices=split.training_roi_indices,
            api_limits=self._api_limits,
        )
        metric_policy = implementation.prepare_metric_policy(
            entry, target_model["output"]
        )
        build = canonical_tree(self._trainer_build(entry))
        report = _queued_report_template(
            request,
            admitted,
            split,
            starting_weight_digest,
            build,
            metric_policy,
            configuration,
            self._api_limits,
            source_model=model,
        )
        gpu_id = request["trainer_settings"]["training_gpu_id"]
        return PreparedManagedTrainingRun(
            request=request,
            starting_weight_digest=starting_weight_digest,
            trainer_build=build,
            total_epochs=request["trainer_settings"]["total_epochs"],
            trainer_execution_seconds=profile["trainer_execution_seconds"],
            worker_resources=canonical_tree(profile["gpu_resources"][gpu_id]),
            _initial_report=report,
            _api_limits=self._api_limits,
        )

    def _retained_dataset(self, request: Mapping[str, Any]) -> RetainedTrainingDataset:
        dataset_id = request["dataset_id"]
        summary = self._dataset_control.get(dataset_id)
        try:
            record = summary["record"]
        except (KeyError, TypeError) as error:
            raise GatewayFailure(
                GatewayErrorCode.INTERNAL_ERROR,
                "training dataset state is inconsistent",
                resource_id=dataset_id,
            ) from error
        if (
            record.get("type") != "retained"
            or record.get("dataset_digest") != request["dataset_digest"]
        ):
            raise GatewayFailure(
                GatewayErrorCode.TRAINING_DATASET_NOT_FOUND,
                "the selected training dataset is no longer available",
                resource_id=dataset_id,
            )
        try:
            retained = self._dataset_files.read_dataset(dataset_id, self._api_limits)
        except (ArtifactBytesCorrupt, ArtifactBytesMissing, OSError) as error:
            raise GatewayFailure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "training dataset storage is temporarily unavailable",
                resource_id=dataset_id,
            ) from error
        if (
            retained.manifest["dataset_digest"] != record["dataset_digest"]
            or retained.retained_bytes != record["retained_bytes"]
        ):
            raise GatewayFailure(
                GatewayErrorCode.INTERNAL_ERROR,
                "retained training dataset identity is inconsistent",
                resource_id=dataset_id,
            )
        return retained


def _queued_report_template(
    request: Mapping[str, Any],
    admitted: AdmittedTrainingDataset,
    split: ExpandedTrainingSplit,
    starting_weight_digest: str,
    trainer_build: Mapping[str, Any],
    metric_policy: Mapping[str, Any],
    configuration: Mapping[str, Any],
    api_limits: Any,
    *,
    source_model: Mapping[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "class_initialization": request["class_initialization"],
        "source_model_output": source_model["output"],
        "source_model_microns_per_pixel": source_model["input"]["micronsPerPixel"],
        "run_id": "00000000-0000-4000-8000-000000000000",
        "state": "QUEUED",
        "dataset_digest": request["dataset_digest"],
        "dataset_image_count": len(admitted.manifest["images"]),
        "split_plan": request["split_plan"],
        "source_module": request["source_module"],
        "selected_model_id": request["selected_model_id"],
        "starting_weight_digest": starting_weight_digest,
        "result_form": request["result_form"],
        "result_display_name": request["result_display_name"],
        "result_inference_gpu": request["result_inference_gpu"],
        "task": admitted.manifest["task"],
        "trainer_build": trainer_build,
        "training_seed": request["training_seed"],
        "trainer_settings": request["trainer_settings"],
        "total_epochs": request["trainer_settings"]["total_epochs"],
        "field_profile": request["field_profile"],
        **(
            {"min_instance_pixels": request["min_instance_pixels"]}
            if "min_instance_pixels" in request
            else {}
        ),
        "augmentation": request["augmentation"],
        "trainer_configuration": configuration,
        "training_counts": _partition_counts(
            admitted.manifest,
            admitted.roi_label_summaries,
            split.training_roi_indices,
        ),
        "metric_policy": metric_policy,
        "epochs": [],
        "unavailable_metrics": [],
        "messages": [],
    }
    if request["split_plan"]["type"] == "split":
        report["validation_counts"] = _partition_counts(
            admitted.manifest,
            admitted.roi_label_summaries,
            split.validation_roi_indices,
        )
    return validate_training_report(report, api_limits)


def _partition_counts(
    manifest: Mapping[str, Any],
    summaries: Sequence[TrainingRoiLabelSummary],
    roi_indices: Sequence[int],
) -> dict[str, Any]:
    target_counts = {
        path_class: 0
        for path_class in model_output_target_classes(manifest["task"]["model_output"])
    }
    image_indices = set()
    negative_count = 0
    for roi_index in roi_indices:
        image_indices.add(manifest["rois"][roi_index]["image_index"])
        summary = summaries[roi_index]
        negative_count += int(summary.is_negative)
        if summary.target_object_counts.keys() != target_counts.keys():
            raise ValueError("training label summary has a different class schema")
        for path_class, addition in summary.target_object_counts.items():
            target_counts[path_class] += addition
    return {
        "image_count": len(image_indices),
        "roi_count": len(roi_indices),
        "negative_roi_count": negative_count,
        "target_object_counts": target_counts,
    }
