"""Exact installed-model projection shared by catalog and inference admission.

This owner binds one schema-valid recipe model to its stored module/artifact
identity and the deployed runner registry. Prediction also loads exact tensors;
CPU result assembly uses only the verified metadata. This module does not
dispatch Modal work or execute inference.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_sha256,
)
from histoforge.cloud._errors import (
    GatewayErrorCode,
    GatewayFailure,
)
from histoforge.cloud.architectures.registry import (
    model_runner_availability,
    runner_precision_for_build,
    runner_registry,
)
from histoforge.cloud.inference.gpu import require_supported_inference_gpu
from histoforge.cloud.inference.identity import validate_backend_model_identity
from histoforge.cloud.inference.plans import (
    build_core_halo_profile,
    build_global_decode_profile,
)
from histoforge.cloud.modules.identity import module_content_identity
from histoforge.cloud.modules.schema import validate_commit_sha, validate_model_id
from histoforge.module_ref import ModuleRef, workspace_training_run_id


def installed_model_descriptor(
    record: dict[str, Any],
    recipe: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    """Return the target catalog descriptor for one installed model entry."""

    canonical_id = validate_model_id(model_id)
    try:
        model = recipe["models"][canonical_id]
    except KeyError as error:
        raise KeyError(f"module has no model {canonical_id!r}") from error

    availability = model_runner_availability(model)
    if (
        record["resolved_commit_sha"] is None
        and "training_result_attestation" not in record
        and availability["type"] == "available"
    ):
        availability = {
            "type": "unavailable",
            "code": "UNSUPPORTED_MODEL_CONFIGURATION",
            "message": (
                "executable module content requires an immutable resolved commit"
            ),
        }

    descriptor = {
        "model_config_digest": canonical_digest(model),
        "artifact_digests": _artifact_digests(record, canonical_id),
        "runner_availability": availability,
        "precision_overrides": {},
    }
    if availability["type"] == "available":
        definition = runner_registry()["definitions"][model["arch"]]
        descriptor["precision_overrides"] = definition["precision_overrides"]
        support = definition.get("large_region_support")
        if support is not None:
            evidence = support["validation_evidence_digest"]
            if support["type"] == "global-decode":
                descriptor["large_region_profile"] = build_global_decode_profile(
                    evidence
                )
            elif (settings := model["input"].get("largeRegion")) is not None:
                # QuPath displays these recipe values as its Default choice. Keep
                # its display/tests aligned when this projection changes; Java
                # must not supply a second set of defaults.
                descriptor["large_region_profile"] = build_core_halo_profile(
                    settings["coreWidthModelPx"],
                    settings["coreHeightModelPx"],
                    settings["haloPerSideModelPx"],
                    evidence,
                )
    return canonical_tree(descriptor)


def installed_backend_identity(
    record: dict[str, Any],
    recipe: dict[str, Any],
    model_id: str,
    inference_gpu: str | None = None,
    runner_build: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return the exact executable backend identity for one available model."""

    descriptor = installed_model_descriptor(record, recipe, model_id)
    availability = descriptor["runner_availability"]
    if availability["type"] != "available":
        raise ValueError(
            f"model is unavailable: {availability['code']}: {availability['message']}"
        )
    return _backend_identity(
        record,
        recipe["models"][model_id],
        model_id,
        descriptor,
        inference_gpu,
        runner_build,
    )


def _backend_identity(
    record: dict[str, Any],
    model: dict[str, Any],
    model_id: str,
    descriptor: dict[str, Any],
    inference_gpu: str | None,
    runner_build: dict[str, str] | None,
) -> dict[str, Any]:
    capabilities = descriptor["runner_availability"]["capabilities"]
    resolved_gpu = (
        model["gpu"]
        if inference_gpu is None
        else require_supported_inference_gpu(inference_gpu, model["gpu"])
    )
    selected_build = (
        capabilities["runner_build"] if runner_build is None else runner_build
    )
    runner_precision_for_build(model["arch"], selected_build, resolved_gpu)
    return validate_backend_model_identity(
        {
            "module": module_content_identity(record),
            "model_id": model_id,
            "architecture": model["arch"],
            "gpu": resolved_gpu,
            "model_config_digest": descriptor["model_config_digest"],
            "artifact_digests": descriptor["artifact_digests"],
            "runner_build": selected_build,
            "rasterizer": capabilities["rasterizer"],
            "preprocessing_contract_digest": capabilities[
                "preprocessing_contract_digest"
            ],
        }
    )


def activate_model(
    module_ref: Any,
    model_id: Any,
    content_digest: Any,
    resolved_commit_sha: Any,
    architecture: str,
    build_network: Callable[[dict[str, Any], dict[str, Any]], Any],
    inference_gpu: str | None = None,
    runner_build: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load one exact current model and verify executable tensor identity."""

    from histoforge.cloud.modules.runtime import current_module_snapshot

    resource_id = f"{module_ref}/{model_id}"

    with current_module_snapshot() as current:
        activation, path = resolve_installed_model_activation(
            current,
            module_ref,
            model_id,
            content_digest,
            resolved_commit_sha,
            architecture,
            inference_gpu,
            runner_build,
        )
        import safetensors.torch

        try:
            state_dict = safetensors.torch.load_file(str(path))
        except OSError as error:
            raise GatewayFailure(
                GatewayErrorCode.SERVICE_UNAVAILABLE,
                "the model artifact could not be read",
                resource_id=resource_id,
            ) from error
        except Exception as error:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
                "the model artifact is not a valid exact tensor checkpoint",
                resource_id=resource_id,
            ) from error
    try:
        network = build_network(activation["model"].get("params", {}), state_dict)
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "the model configuration cannot construct its selected runner",
            resource_id=resource_id,
        ) from error
    return {**activation, "network": network}


def resolve_installed_model_activation(
    current: Any,
    module_ref: Any,
    model_id: Any,
    content_digest: Any,
    resolved_commit_sha: Any,
    architecture: str,
    inference_gpu: str | None = None,
    runner_build: dict[str, str] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Verify installed content and weight bytes without constructing a network.

    Return the model/identity/capability metadata and its verified checkpoint path.
    Prediction loads tensors from that path; result assembly needs only metadata.
    Both use the same current Module Volume view and exact artifact checks.
    """

    ref = str(ModuleRef.parse(module_ref))
    canonical_model_id = validate_model_id(model_id)
    expected_content = require_sha256(content_digest, "content_digest")
    expected_commit = (
        validate_commit_sha(resolved_commit_sha, "resolved_commit_sha")
        if resolved_commit_sha is not None
        else None
    )

    try:
        record = current.read_record(ref)
    except FileNotFoundError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_VERSION_UNAVAILABLE,
            "the requested module version is no longer installed",
            resource_id=ref,
        ) from error
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "installed module content does not match its executable identity",
            resource_id=ref,
        ) from error
    if (
        record["content_digest"] != expected_content
        or record["resolved_commit_sha"] != expected_commit
    ):
        raise GatewayFailure(
            GatewayErrorCode.MODULE_VERSION_UNAVAILABLE,
            "the requested module version is no longer installed",
            resource_id=ref,
        )

    if expected_commit is None and workspace_training_run_id(ref) is None:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_VERSION_UNAVAILABLE,
            "the requested commitless module is not an executable training result",
            resource_id=ref,
        )

    try:
        recipe = current.load_recipe(ref, record=record)
        model = recipe["models"][canonical_model_id]
        descriptor = installed_model_descriptor(record, recipe, canonical_model_id)
    except (KeyError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "installed model content does not match its executable identity",
            resource_id=f"{ref}/{canonical_model_id}",
        ) from error
    availability = descriptor["runner_availability"]
    if availability["type"] != "available":
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            availability["message"],
            resource_id=f"{ref}/{canonical_model_id}",
        )
    try:
        backend = installed_backend_identity(
            record, recipe, canonical_model_id, inference_gpu, runner_build
        )
        path = current.artifact_path(ref, canonical_model_id)
    except (KeyError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "installed model content does not match its executable identity",
            resource_id=f"{ref}/{canonical_model_id}",
        ) from error
    if backend["architecture"] != architecture:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "the model architecture does not match the selected runner",
            resource_id=f"{ref}/{canonical_model_id}",
        )

    return (
        {
            "model": model,
            "backend": backend,
            "capabilities": {
                **availability["capabilities"],
                "runner_build": backend["runner_build"],
            },
        },
        path,
    )


def activation_failure(error: BaseException, resource_id: str) -> GatewayFailure:
    """Map model validation or network startup failures to the existing errors."""

    if isinstance(error, GatewayFailure):
        return error
    from histoforge._vendor.histoforge_models.pathology.common import (
        StateDictIdentityError,
    )

    if isinstance(error, StateDictIdentityError):
        return GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "model tensor identity does not match the selected runner",
            resource_id=resource_id,
        )
    if isinstance(error, FileNotFoundError):
        return GatewayFailure(
            GatewayErrorCode.MODULE_VERSION_UNAVAILABLE,
            "the requested module version is no longer installed",
            resource_id=resource_id,
        )
    if isinstance(error, (OSError, ConnectionError, TimeoutError)):
        return GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "runner activation could not read required storage",
            resource_id=resource_id,
        )
    return GatewayFailure(
        GatewayErrorCode.RUNNER_FAILED,
        "runner activation failed",
        resource_id=resource_id,
    )


def _artifact_digests(record: dict[str, Any], model_id: str) -> dict[str, str]:
    result = {
        artifact["role"]: artifact["sha256"]
        for artifact in record["resolved_artifacts"]
        if artifact["model_id"] == model_id
    }
    if not result:
        raise ValueError(f"installed model {model_id!r} has no recorded artifacts")
    return canonical_tree(result)
