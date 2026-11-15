"""Shared exact-model rasterization, inference, and feature validation.

Finite HTTP calls and durable finite-plan inputs differ in lifecycle and
publication.  They deliberately share this scientific path so equivalent
inputs cannot drift in source decoding, preprocessing, inference, or feature
validation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from time import perf_counter
from typing import Any

from histoforge.canonical_json import require_path_class_name
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
    ArtifactFileStore,
)
from histoforge.cloud.inference.contract import (
    validate_inference_features,
)
from histoforge.cloud.inference.features import ResultLimitExceeded
from histoforge.cloud.inference.identity import validate_backend_model_identity
from histoforge.cloud.modules.schema import model_output_emitted_classes
from histoforge.cloud.rasterization.inputs import model_input_source_artifact
from histoforge.cloud.rasterization.model_input import rasterize_model_input
from histoforge.cloud.rasterization.rasterizer import InsufficientSourceContext
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.store import SourceArtifactStore

_LOGGER = logging.getLogger(__name__)


def mounted_inference_stores() -> tuple[
    SourceArtifactStore, SourceFileStore, ArtifactFileStore
]:
    """Construct source readers and result-file storage for a mounted runner."""

    import modal

    from histoforge.cloud._app import artifact_volume

    records = modal.Dict.from_name(
        DEPLOYMENT_CONFIG.source_artifacts_dict_name,
        create_if_missing=True,
    )
    return (
        SourceArtifactStore(
            records,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            DEPLOYMENT_CONFIG.limits.wire_value(),
            max_active_source_artifact_peak_bytes=(
                DEPLOYMENT_CONFIG.max_active_source_artifact_peak_bytes
            ),
        ),
        SourceFileStore(
            "/artifacts",
            reload_volume=artifact_volume.reload,
            commit_volume=artifact_volume.commit,
        ),
        ArtifactFileStore(
            "/artifacts",
            reload_volume=artifact_volume.reload,
            commit_volume=artifact_volume.commit,
        ),
    )


def require_activation_identity(
    expected_backend: Mapping[str, Any], activation: Mapping[str, Any]
) -> None:
    """Require request, installed model, and executable capability identity."""

    backend = validate_backend_model_identity(activation["backend"])
    capabilities = activation["capabilities"]
    expected_capabilities = {
        "runner_build": capabilities["runner_build"],
        "rasterizer": capabilities["rasterizer"],
        "preprocessing_contract_digest": capabilities["preprocessing_contract_digest"],
    }
    if any(backend[field] != value for field, value in expected_capabilities.items()):
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "backend identity does not match selected runner capabilities",
            resource_id=(f"{backend['module']['module_ref']}/{backend['model_id']}"),
        )
    expected = validate_backend_model_identity(expected_backend)
    if expected == backend:
        return
    code = (
        GatewayErrorCode.MODULE_VERSION_UNAVAILABLE
        if expected["module"] != backend["module"]
        else GatewayErrorCode.MODULE_IDENTITY_MISMATCH
    )
    raise GatewayFailure(
        code,
        "the selected backend identity no longer matches the installed model",
        resource_id=(f"{expected['module']['module_ref']}/{expected['model_id']}"),
    )


def execute_model_input(
    raster_input: Mapping[str, Any],
    result_budget: Mapping[str, int],
    activation: Mapping[str, Any],
    source_store: SourceArtifactStore,
    files: SourceFileStore,
    infer: Callable[[Any, Mapping[str, int]], dict[str, Any]],
    source_binding: Any = None,
) -> dict[str, Any]:
    """Produce one complete validated feature population or typed failure."""

    backend = activation["backend"]
    resource_id = f"{backend['module']['module_ref']}/{backend['model_id']}"
    artifact = model_input_source_artifact(raster_input, source_binding)
    try:
        started = perf_counter()
        resolved = source_store.resolve_sealed_reference(artifact)
        raster = rasterize_model_input(
            raster_input,
            resolved,
            files,
            DEPLOYMENT_CONFIG.source_artifact_limits.wire_value(),
            activation["capabilities"]["source_containers"],
            source_binding=source_binding,
        )
        _LOGGER.info(
            "stage=source_read_and_rasterize elapsed_seconds=%.6f",
            perf_counter() - started,
        )
    except GatewayFailure:
        raise
    except (ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "sealed source bytes are missing or corrupt; recreate the artifact",
            resource_id=artifact["artifact_id"],
        ) from error
    except OSError as error:
        raise GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "sealed source bytes could not be read",
            resource_id=artifact["artifact_id"],
        ) from error
    except InsufficientSourceContext as error:
        raise GatewayFailure(
            GatewayErrorCode.INSUFFICIENT_CONTEXT,
            str(error),
        ) from error
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            str(error),
        ) from error

    try:
        geojson = infer(raster, result_budget)
    except ResultLimitExceeded as error:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            str(error),
        ) from error
    except Exception as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "model prediction or output conversion failed",
            resource_id=resource_id,
        ) from error
    try:
        started = perf_counter()
        features = validate_inference_features(
            geojson,
            raster_input["frame"]["transform"]["raster"],
            result_budget,
            model_declared_classes(activation["model"]["output"]),
        )
        _LOGGER.info(
            "stage=feature_validation elapsed_seconds=%.6f", perf_counter() - started
        )
        return features
    except ResultLimitExceeded as error:
        raise GatewayFailure(
            GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
            str(error),
        ) from error
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.RUNNER_FAILED,
            "model output violates the inference feature contract",
            resource_id=resource_id,
        ) from error


def declared_classes(value: Collection[Any]) -> list[str]:
    """Return one unique ordered PathClass vocabulary from an activated model."""

    if not isinstance(value, (list, tuple)):
        raise ValueError("declared_classes must be an array")
    result = [
        require_path_class_name(item, f"declared_classes[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ValueError("declared_classes contains a duplicate PathClass")
    return result


def model_declared_classes(output: Mapping[str, Any]) -> list[str]:
    """Return the validated emitted PathClasses from one model output record."""

    return declared_classes(model_output_emitted_classes(output))
