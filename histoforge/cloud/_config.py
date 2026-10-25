"""Deployment resource names, admission limits, and capability validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_tree,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_positive_int32,
    require_positive_safe_integer,
    require_sha256,
)
from histoforge.cloud.inference.gpu import validate_inference_gpu_choices
from histoforge.cloud.training.limits import (
    TrainingApiLimits,
    TrainingDatasetLimits,
    TrainingDatasetTransferLimits,
    TrainingReportLimits,
    TrainingServiceLimits,
    validate_training_api_limits,
)

_SOURCE_ARTIFACT_LIMIT_FIELDS = {
    "max_start_canonical_bytes",
    "max_artifact_bytes",
    "max_chunks",
    "max_chunk_bytes",
    "max_image_series",
    "max_pyramid_levels_per_series",
    "max_level_width_px",
    "max_level_height_px",
    "max_complete_decode_pixels",
    "upload_idle_timeout_seconds",
    "absolute_lifetime_seconds",
}
_GATEWAY_LIMIT_FIELDS = {
    "max_json_request_bytes",
    "max_catalog_page_size",
    "max_catalog_encoded_bytes",
    "max_admin_request_bytes",
    "max_module_bytes",
    "max_installed_bytes",
    "max_active_admin_jobs",
    "max_active_source_artifacts",
    "max_active_model_runs",
}
_GATEWAY_HEALTH_FIELDS = {"api_major", "status"}
_GATEWAY_CAPABILITY_FIELDS = {
    "api_major",
    "module_schema_version",
    "source_artifact_schema_version",
    "finite_inference_schema_version",
    "durable_model_run_schema_version",
    "training_api_schema_version",
    "deployment_id",
    "auth_revision",
    "gateway_build",
    "runner_registry_digest",
    "inference_gpus",
    "source_artifact_limits",
    "training_api_limits",
    "limits",
}


@dataclass(frozen=True)
class SourceArtifactLimits:
    """Upload, source-container, decode, and lifetime limits for one deployment."""

    max_start_canonical_bytes: int
    max_artifact_bytes: int
    max_chunks: int
    max_chunk_bytes: int
    max_image_series: int
    max_pyramid_levels_per_series: int
    max_level_width_px: int
    max_level_height_px: int
    max_complete_decode_pixels: int
    upload_idle_timeout_seconds: int
    absolute_lifetime_seconds: int

    def wire_value(self) -> dict[str, int]:
        return validate_source_artifact_limits(asdict(self))


@dataclass(frozen=True)
class DeploymentLimits:
    """Request, catalog, storage, and concurrency limits for one deployment."""

    max_json_request_bytes: int
    max_catalog_page_size: int
    max_catalog_encoded_bytes: int
    max_admin_request_bytes: int
    max_module_bytes: int
    max_installed_bytes: int
    max_active_admin_jobs: int
    max_active_source_artifacts: int
    max_active_model_runs: int

    def wire_value(self) -> dict[str, int]:
        return validate_gateway_limits(asdict(self))


@dataclass(frozen=True)
class DeploymentConfig:
    """Single owner of concrete Modal resource names and deployment policy."""

    gateway_app_name: str
    gateway_memory_mib: int
    gateway_max_concurrent_inputs: int
    gateway_timeout_seconds: int
    admin_job_admission_class_name: str
    module_mutation_class_name: str
    artifact_lifecycle_class_name: str
    training_state_class_name: str
    training_executor_function_name: str
    model_run_executor_function_name: str
    modules_volume_name: str
    gateway_state_dict_name: str
    admin_jobs_dict_name: str
    source_artifacts_dict_name: str
    model_runs_dict_name: str
    artifact_volume_name: str
    huggingface_secret_name: str
    max_active_source_artifact_peak_bytes: int
    source_artifact_limits: SourceArtifactLimits
    training_api_limits: TrainingApiLimits
    training_service_limits: TrainingServiceLimits
    limits: DeploymentLimits

    def __post_init__(self) -> None:
        require_positive_safe_integer(
            self.max_active_source_artifact_peak_bytes,
            "max_active_source_artifact_peak_bytes",
        )
        gateway_timeout = require_positive_int32(
            self.gateway_timeout_seconds, "gateway_timeout_seconds"
        )
        writer_claim = self.training_service_limits.dataset_writer_claim_seconds
        if writer_claim <= gateway_timeout:
            raise ValueError(
                "dataset_writer_claim_seconds must exceed gateway_timeout_seconds"
            )


# Deployment policy shared by gateway services and capability responses.
DEPLOYMENT_CONFIG = DeploymentConfig(
    gateway_app_name="histoforge-gateway",
    # Phase-0 HTTP evidence exercised eight concurrent inputs. At that bound,
    # bounded 8 MiB body copies plus one complete PNG validation remain well
    # below the requested 1 GiB while Modal can still scale horizontally.
    gateway_memory_mib=1024,
    gateway_max_concurrent_inputs=8,
    # A complete managed-training source and dataset preflight is intentionally
    # synchronous and can stream several gigabytes before the run is accepted.
    gateway_timeout_seconds=1_800,
    admin_job_admission_class_name="AdminJobAdmissionLane",
    module_mutation_class_name="ModuleMutationLane",
    artifact_lifecycle_class_name="ArtifactLifecycleLane",
    training_state_class_name="TrainingStateLane",
    training_executor_function_name="execute_training_run",
    model_run_executor_function_name="execute_model_run",
    modules_volume_name="modules",
    gateway_state_dict_name="histoforge-gateway-state",
    admin_jobs_dict_name="histoforge-jobs",
    source_artifacts_dict_name="histoforge-source-artifacts",
    model_runs_dict_name="histoforge-model-runs",
    artifact_volume_name="histoforge-artifacts",
    huggingface_secret_name="histoforge-huggingface",
    max_active_source_artifact_peak_bytes=1 * 1024**4,
    source_artifact_limits=SourceArtifactLimits(
        max_start_canonical_bytes=128 * 1024,
        max_artifact_bytes=32 * 1024**3,
        max_chunks=4096,
        max_chunk_bytes=8 * 1024**2,
        max_image_series=16,
        max_pyramid_levels_per_series=16,
        max_level_width_px=500_000,
        max_level_height_px=500_000,
        max_complete_decode_pixels=32_000_000,
        upload_idle_timeout_seconds=3600,
        absolute_lifetime_seconds=48 * 60 * 60,
    ),
    training_api_limits=TrainingApiLimits(
        dataset=TrainingDatasetLimits(
            max_rois=2_000,
            max_objects_per_roi=10_000,
            max_total_objects=500_000,
            max_vertices_per_geometry=100_000,
            max_total_vertices=6_500_000,
            max_json_member_bytes=8 * 1024**2,
            max_archive_bytes=9 * 1024**3 + 8 * 1024**2,
            max_extracted_bytes=9 * 1024**3,
        ),
        transfer=TrainingDatasetTransferLimits(
            max_chunk_bytes=8 * 1024**2,
            max_chunks_per_upload=1_153,
        ),
        report=TrainingReportLimits(
            max_metric_series=100,
            max_epochs=1_000,
            max_messages=1_000,
            max_text_bytes=4_096,
            max_component_parameters_bytes=65_536,
            max_component_parameter_depth=16,
            max_component_collection_entries=1_024,
            max_report_bytes=8 * 1024**2,
        ),
        max_page_size=100,
        max_page_encoded_bytes=1 * 1024**2,
    ).validated(),
    training_service_limits=TrainingServiceLimits(
        max_active_dataset_uploads=4,
        dataset_upload_idle_timeout_seconds=3_600,
        # Keep control ownership beyond the gateway input budget. Modal may
        # cancel an ASGI request before its blocking file thread has stopped.
        dataset_writer_claim_seconds=1_860,
        max_active_training_runs=2,
        worker_call_attachment_timeout_seconds=60,
        max_training_execution_seconds=86_400,
        training_queue_allowance_seconds=600,
        pending_snapshot_cleanup_age_seconds=3_600,
        max_control_records=1_000,
        max_control_stored_bytes=16 * 1024**2,
    ).validated(),
    limits=DeploymentLimits(
        max_json_request_bytes=1 * 1024**2,
        max_catalog_page_size=100,
        max_catalog_encoded_bytes=8 * 1024**2,
        max_admin_request_bytes=256 * 1024,
        max_module_bytes=5 * 1024**3,
        max_installed_bytes=500 * 1024**3,
        max_active_admin_jobs=4,
        # Retained uploaded images, not parallel inference. Native upload-history
        # checks cover this allowance, including expired-file cleanup.
        max_active_source_artifacts=1_024,
        max_active_model_runs=4,
    ),
)

HUGGINGFACE_SECRET_NAME = DEPLOYMENT_CONFIG.huggingface_secret_name
MAX_JSON_REQUEST_BYTES = DEPLOYMENT_CONFIG.limits.max_json_request_bytes
MAX_CATALOG_PAGE_SIZE = DEPLOYMENT_CONFIG.limits.max_catalog_page_size
MAX_ADMIN_REQUEST_BYTES = DEPLOYMENT_CONFIG.limits.max_admin_request_bytes
MAX_ACTIVE_ADMIN_JOBS = DEPLOYMENT_CONFIG.limits.max_active_admin_jobs
MAX_MODULE_BYTES = DEPLOYMENT_CONFIG.limits.max_module_bytes
MAX_INSTALLED_BYTES = DEPLOYMENT_CONFIG.limits.max_installed_bytes


def validate_source_artifact_limits(value: Any) -> dict[str, int]:
    """Validate the exact source-upload, decode, and lifetime ceilings."""

    limits = _require_exact_object(
        value, _SOURCE_ARTIFACT_LIMIT_FIELDS, "source artifact limits"
    )
    result = {
        "max_start_canonical_bytes": _positive_safe(
            limits, "max_start_canonical_bytes"
        ),
        "max_artifact_bytes": _positive_safe(limits, "max_artifact_bytes"),
        "max_chunks": _positive_int32(limits, "max_chunks"),
        "max_chunk_bytes": _positive_safe(limits, "max_chunk_bytes"),
        "max_image_series": _positive_int32(limits, "max_image_series"),
        "max_pyramid_levels_per_series": _positive_int32(
            limits, "max_pyramid_levels_per_series"
        ),
        "max_level_width_px": _positive_int32(limits, "max_level_width_px"),
        "max_level_height_px": _positive_int32(limits, "max_level_height_px"),
        "max_complete_decode_pixels": _positive_safe(
            limits, "max_complete_decode_pixels"
        ),
        "upload_idle_timeout_seconds": _positive_int32(
            limits, "upload_idle_timeout_seconds"
        ),
        "absolute_lifetime_seconds": _positive_int32(
            limits, "absolute_lifetime_seconds"
        ),
    }
    return canonical_tree(result)


def validate_gateway_limits(value: Any) -> dict[str, int]:
    """Validate the exact request, catalog, storage, and concurrency ceilings."""

    limits = _require_exact_object(value, _GATEWAY_LIMIT_FIELDS, "gateway limits")
    result = {
        "max_json_request_bytes": _positive_safe(limits, "max_json_request_bytes"),
        "max_catalog_page_size": _positive_int32(limits, "max_catalog_page_size"),
        "max_catalog_encoded_bytes": _positive_safe(
            limits, "max_catalog_encoded_bytes"
        ),
        "max_admin_request_bytes": _positive_safe(limits, "max_admin_request_bytes"),
        "max_module_bytes": _positive_safe(limits, "max_module_bytes"),
        "max_installed_bytes": _positive_safe(limits, "max_installed_bytes"),
        "max_active_admin_jobs": _positive_int32(limits, "max_active_admin_jobs"),
        "max_active_source_artifacts": _positive_int32(
            limits, "max_active_source_artifacts"
        ),
        "max_active_model_runs": _positive_int32(limits, "max_active_model_runs"),
    }
    return canonical_tree(result)


def validate_gateway_health(value: Any) -> dict[str, Any]:
    """Validate the minimal unauthenticated liveness response."""

    health = _require_exact_object(value, _GATEWAY_HEALTH_FIELDS, "gateway health")
    api_major = _positive_int32(health, "api_major")
    status = health["status"]
    if status != "ok":
        raise ValueError("gateway health status must be 'ok'")
    return {"api_major": api_major, "status": "ok"}


def validate_gateway_capabilities(value: Any) -> dict[str, Any]:
    """Validate the authenticated deployment, schema, build, and limit handshake."""

    capabilities = _require_exact_object(
        value, _GATEWAY_CAPABILITY_FIELDS, "gateway capabilities"
    )
    result = {
        "api_major": _positive_int32(capabilities, "api_major"),
        "module_schema_version": _positive_int32(capabilities, "module_schema_version"),
        "source_artifact_schema_version": _positive_int32(
            capabilities, "source_artifact_schema_version"
        ),
        "finite_inference_schema_version": _positive_int32(
            capabilities, "finite_inference_schema_version"
        ),
        "durable_model_run_schema_version": _positive_int32(
            capabilities, "durable_model_run_schema_version"
        ),
        "training_api_schema_version": _positive_int32(
            capabilities, "training_api_schema_version"
        ),
        "deployment_id": require_nonblank_string(
            capabilities["deployment_id"], "deployment_id"
        ),
        "auth_revision": require_nonnegative_safe_integer(
            capabilities["auth_revision"], "auth_revision"
        ),
        "gateway_build": parse_software_build_identity(capabilities["gateway_build"]),
        "runner_registry_digest": require_sha256(
            capabilities["runner_registry_digest"], "runner_registry_digest"
        ),
        "inference_gpus": validate_inference_gpu_choices(
            capabilities["inference_gpus"]
        ),
        "source_artifact_limits": validate_source_artifact_limits(
            capabilities["source_artifact_limits"]
        ),
        "training_api_limits": validate_training_api_limits(
            capabilities["training_api_limits"]
        ),
        "limits": validate_gateway_limits(capabilities["limits"]),
    }
    return canonical_tree(result)


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    if actual != fields:
        missing = sorted(fields - actual)
        unknown = sorted(actual - fields)
        raise ValueError(
            f"{path} fields mismatch: missing={missing}, unknown={unknown}"
        )
    return value


def _positive_int32(value: Mapping[str, Any], field: str) -> int:
    return require_positive_int32(value[field], field)


def _positive_safe(value: Mapping[str, Any], field: str) -> int:
    return require_positive_safe_integer(value[field], field)
