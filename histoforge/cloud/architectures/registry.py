"""Build-owned runner dispatch, resources, identities, and capabilities.

The gateway, Modal decorators, catalog, and runner startup consume this module;
no route or runner repeats base resources, concurrency, timeout, or capability
policy. A module's GPU is applied as a dynamic Modal class option at dispatch;
it is not restricted by the architecture key.
Importing it is torch-free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from histoforge._vendor.histoforge_models.pathology.network_execution import (
    NetworkPrecision,
)
from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_nonblank_string,
    require_nonnegative_int32,
    require_positive_int32,
    require_sha256,
    sha256_bytes,
)
from histoforge.cloud.modules.schema import (
    VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX,
    VARIABLE_FIELD_PLANNER_ID,
    model_output_class_channels,
)
from histoforge.cloud.runner_contract import validate_runner_capabilities
from histoforge.cloud.runtime_identity import runtime_build
from histoforge.cloud.sources.contract import SOURCE_CONTAINERS

KNOWN_ARCHITECTURES = frozenset(
    {
        "abmil",
        "ctranspath",
        "dinov3-adapter-cellpose",
        "dinov3-adapter-hovernet",
        "dinov3-adapter-semantic",
        "linear",
        "mlp",
        "torchscript-classifier",
        "vit-encoder",
    }
)

RASTERIZATION_CONTRACT_DIGEST = (
    "10087398a72117fa8f9f3e9f4d8d03203c0623f52f7ec4003fd7955a05371690"
)


@dataclass(frozen=True)
class RunnerResources:
    """Base values consumed directly by one Modal class decorator."""

    modal_class_name: str
    gpu: str
    cpu_cores: int | None
    memory_mib: int | None
    max_containers: int
    max_concurrent_inputs: int
    attempt_timeout_seconds: int


@dataclass(frozen=True)
class RunnerDefinition:
    """Build-owned resources, supported precision and large-region strategy."""

    resources: RunnerResources
    # Support and evidence belong to the runner; core/halo values belong to recipes.
    large_region_support: Mapping[str, Any] | None = None
    # The current exact checkpoint loaders construct FP32 networks. This ordinary
    # forward path uses their native dtype; it does not convert checkpoint tensors.
    native_network_precision: NetworkPrecision = "fp32"
    supports_bf16: bool = False


# SHA-256 of custom_docs/architecture/evidence/instance-large-region-20261112.md.
_INSTANCE_CORE_HALO_SUPPORT = {
    "type": "core-halo",
    "validation_evidence_digest": (
        "47e50fbdc662a2882dd8d089a3d67e327bf2f5e8c7f354afe2f4de41dff5462b"
    ),
}

# SHA-256 of custom_docs/architecture/evidence/semantic-large-region-20261114.md.
_SEMANTIC_CORE_HALO_SUPPORT = {
    "type": "core-halo",
    "validation_evidence_digest": (
        "0312780c40dac292bcdd6fa6b081694d195654085761314e33db5783d339fe65"
    ),
}

_RUNNER_DEFINITIONS = {
    "dinov3-adapter-cellpose": RunnerDefinition(
        supports_bf16=True,
        large_region_support=_INSTANCE_CORE_HALO_SUPPORT,
        resources=RunnerResources(
            modal_class_name="CellposeRunner",
            gpu="L40S",
            cpu_cores=4,
            memory_mib=32_768,
            max_containers=4,
            max_concurrent_inputs=2,
            attempt_timeout_seconds=600,
        ),
    ),
    "dinov3-adapter-hovernet": RunnerDefinition(
        supports_bf16=True,
        large_region_support=_INSTANCE_CORE_HALO_SUPPORT,
        resources=RunnerResources(
            modal_class_name="HovernetRunner",
            gpu="A10G",
            cpu_cores=None,
            memory_mib=None,
            max_containers=4,
            max_concurrent_inputs=1,
            attempt_timeout_seconds=600,
        ),
    ),
    "dinov3-adapter-semantic": RunnerDefinition(
        supports_bf16=True,
        large_region_support=_SEMANTIC_CORE_HALO_SUPPORT,
        resources=RunnerResources(
            modal_class_name="SemanticRunner",
            gpu="A10G",
            cpu_cores=None,
            memory_mib=None,
            max_containers=4,
            max_concurrent_inputs=1,
            attempt_timeout_seconds=600,
        ),
    ),
}

_OUTPUT_TYPES = {
    "dinov3-adapter-cellpose": (
        "INSTANCE_SEGMENTATION",
        "INSTANCE_SEGMENTATION_AND_CLASSIFICATION",
    ),
    "dinov3-adapter-hovernet": (
        "INSTANCE_SEGMENTATION",
        "INSTANCE_SEGMENTATION_AND_CLASSIFICATION",
    ),
    "dinov3-adapter-semantic": ("DENSE_CLASSIFICATION",),
}

# These deployed GPU options support native BF16. An arbitrary recipe GPU is not
# evidence of BF16 support; it remains usable through the native path instead.
BF16_INFERENCE_GPUS = ("A10G", "L40S", "A100")

_OUTPUT_TYPE_NAMES = {
    "instance_segmentation": "INSTANCE_SEGMENTATION",
    "instance_segmentation_and_classification": (
        "INSTANCE_SEGMENTATION_AND_CLASSIFICATION"
    ),
    "dense_classification": "DENSE_CLASSIFICATION",
    "region_classification": "REGION_CLASSIFICATION",
}

_COMMON_MODEL_PARAMS = {
    "num_classes",
    "bsize",
    "tile_overlap",
}
_PARAMS_BY_ARCHITECTURE = {
    "dinov3-adapter-cellpose": _COMMON_MODEL_PARAMS
    | {"batch_size", "min_instance_pixels"},
    "dinov3-adapter-hovernet": _COMMON_MODEL_PARAMS
    | {
        "edge_threshold",
        "fg_threshold",
        "object_size",
        "morph_kernel",
        "ksize",
        "min_instance_pixels",
    },
    "dinov3-adapter-semantic": _COMMON_MODEL_PARAMS | {"batch_size"},
}

# Durable execution uses architecture-specific spatial and geometry limits.
_DURABLE_ARCHITECTURES = frozenset(
    {
        "dinov3-adapter-cellpose",
        "dinov3-adapter-hovernet",
        "dinov3-adapter-semantic",
    }
)
MODEL_RUN_MAX_EXECUTION_SECONDS = 12 * 60 * 60
_DURABLE_TERMINAL_RETENTION_SECONDS = 7 * 24 * 60 * 60


# Both planners normalize the supplied real image before neural tiling/blending.
FIELD_AGGREGATION_ID = "cellpose-tapered-mean-v1"
FIELD_PREPROCESSING_ID = "normalize99-real-image-v1"
_TARGET_NORMALIZED_ARCHITECTURES = frozenset(
    {"dinov3-adapter-cellpose", "dinov3-adapter-hovernet", "dinov3-adapter-semantic"}
)
_FIXED_FIELD_DEFAULTS = {
    "dinov3-adapter-cellpose": (256, 0.1),
    "dinov3-adapter-hovernet": (1024, 0.25),
    "dinov3-adapter-semantic": (1024, 0.1),
}


def runner_resources(architecture: str) -> RunnerResources:
    """Return the deployed base resource policy for one runnable architecture."""

    try:
        return _RUNNER_DEFINITIONS[architecture].resources
    except KeyError as error:
        raise ValueError(
            f"architecture has no deployed runner: {architecture}"
        ) from error


def uses_target_normalization(architecture: str, plan_type: str) -> bool:
    """Select shared target statistics before CoreHalo prediction windows."""

    return plan_type == "core-halo" and architecture in _TARGET_NORMALIZED_ARCHITECTURES


def runner_network_precision(
    architecture: str, inference_precision: str = "default"
) -> NetworkPrecision:
    """Select the forward path without changing checkpoint or training dtypes."""

    definition = _RUNNER_DEFINITIONS[architecture]
    if inference_precision == "default":
        return definition.native_network_precision
    if inference_precision == "bf16" and definition.supports_bf16:
        return "bf16"
    raise ValueError(f"unsupported inference precision: {inference_precision!r}")


def runner_build_for_precision(
    architecture: str, inference_precision: str
) -> dict[str, str]:
    """Reconstruct the selected build at worker startup, independently of clients."""

    definition = runner_registry()["definitions"][architecture]
    if inference_precision == "default":
        return dict(definition["capabilities"]["runner_build"])
    if inference_precision == "bf16" and "bf16" in definition["precision_overrides"]:
        return dict(definition["precision_overrides"]["bf16"]["runner_build"])
    raise ValueError(f"unsupported inference precision: {inference_precision!r}")


def runner_precision_for_build(
    architecture: str, runner_build: Any, inference_gpu: str
) -> str:
    """Validate the captured build and GPU; never downgrade an execution request."""

    selected = parse_software_build_identity(runner_build)
    definition = runner_registry()["definitions"][architecture]
    if selected == definition["capabilities"]["runner_build"]:
        return "default"
    for precision, override in definition["precision_overrides"].items():
        if selected == override["runner_build"]:
            if inference_gpu not in override["gpus"]:
                raise ValueError(
                    f"{precision} inference is not supported on {inference_gpu}"
                )
            return precision
    raise ValueError("the requested runner build is not available")


def validate_precision_overrides(
    value: Any, native_runner_build: dict[str, str] | None
) -> dict[str, Any]:
    """Validate catalog-only choices; saved actions retain only the chosen build."""

    if not isinstance(value, dict) or set(value) - {"bf16"}:
        raise ValueError("precision_overrides must be an object with only bf16")
    if native_runner_build is None and value:
        raise ValueError("an unavailable runner cannot advertise precision overrides")
    if not value:
        return {}
    override = value["bf16"]
    if not isinstance(override, dict) or set(override) != {"runner_build", "gpus"}:
        raise ValueError("bf16 precision override has missing or unknown fields")
    build = parse_software_build_identity(override["runner_build"])
    if build == native_runner_build:
        raise ValueError("BF16 and native runner builds must differ")
    gpus = override["gpus"]
    if (
        not isinstance(gpus, list)
        or not gpus
        or any(not isinstance(gpu, str) for gpu in gpus)
        or len(gpus) != len(set(gpus))
    ):
        raise ValueError("BF16 GPUs must be nonempty, unique GPU names")
    for gpu in gpus:
        require_nonblank_string(gpu, "BF16 GPUs entry")
    return canonical_tree({"bf16": {"runner_build": build, "gpus": gpus}})


def model_run_assembly_resources() -> dict[str, int]:
    """CPU-only final assembly policy, independent of prediction attempts.

    The 3.9-million-object comparison used four CPUs / 32 GiB and took 3,283 s.
    Two hours provides a bounded qualification margin, not a latency guarantee.
    The accepted model-run deadline still fences complete-result publication.
    """

    return {
        "cpu": 4,
        "memory": 32_768,
        "timeout": 2 * 60 * 60,
        "max_containers": 4,
        "retries": 0,
    }


def processing_result_limits(architecture: str) -> dict[str, int]:
    """Bound one decoded processing image, independently of whole-run output.

    These backend-only limits are pinned by the runner build. Increasing the
    aggregate result allowance must not enlarge a worker's candidate population.
    Finite-only architectures keep their existing finite-result allowance.
    """

    if architecture == "dinov3-adapter-cellpose":
        return {
            "max_transferred_bytes": 32 * 1024**2,
            "max_decoded_bytes": 128 * 1024**2,
            "max_record_bytes": 1_000_000,
            "max_features": 200_000,
            "max_vertices": 10_000_000,
        }
    return dict(
        runner_registry()["definitions"][architecture]["capabilities"]["finite"][
            "result"
        ]
    )


def resolve_runner_cls_name(architecture: str) -> str | None:
    """Return the deployed Modal class name, or ``None`` when no runner exists."""

    definition = _RUNNER_DEFINITIONS.get(architecture)
    return definition.resources.modal_class_name if definition else None


def is_known_architecture(architecture: str) -> bool:
    """Distinguish unknown schema vocabulary from a known undeployed runner."""

    return architecture in KNOWN_ARCHITECTURES


def model_runner_availability(model: dict[str, Any]) -> dict[str, Any]:
    """Project one schema-valid model against the exact deployed registry.

    Availability is deliberately model-local: an unsupported forward feature or
    undeployed architecture does not make the containing module disappear from
    the catalog.
    """

    architecture = model["arch"]
    if architecture not in KNOWN_ARCHITECTURES:
        return _unavailable(
            "UNKNOWN_ARCHITECTURE",
            f"architecture {architecture!r} is not known to this gateway build",
        )
    definition = runner_registry()["definitions"].get(architecture)
    if definition is None:
        return _unavailable(
            "RUNNER_NOT_DEPLOYED",
            f"architecture {architecture!r} has no deployed runner",
        )
    if "backbone" in model or "acceptCriteria" in model["input"]:
        return _unavailable(
            "UNSUPPORTED_INPUT_FEATURE",
            "the model declares an input or backbone feature this runner "
            "does not implement",
        )
    output_type = _OUTPUT_TYPE_NAMES[model["output"]["type"]]
    if output_type not in definition["capabilities"]["output_types"]:
        return _unavailable(
            "UNSUPPORTED_OUTPUT_TYPE",
            f"runner {architecture!r} does not implement {output_type}",
        )
    try:
        _validate_model_configuration(model, definition)
    except ValueError as error:
        return _unavailable("UNSUPPORTED_MODEL_CONFIGURATION", str(error))
    return canonical_tree(
        {"type": "available", "capabilities": definition["capabilities"]}
    )


def require_available_model(model: dict[str, Any]) -> dict[str, Any]:
    """Return capabilities for one executable model or raise its exact reason."""

    availability = model_runner_availability(model)
    if availability["type"] != "available":
        raise ValueError(availability["message"])
    return availability["capabilities"]


def resolved_fixed_field_profile(model: dict[str, Any]) -> dict[str, Any] | None:
    """Return the deployed wrapper's exact fixed-field contract when runnable."""

    if "fieldProfile" in model["input"]:
        return None
    availability = model_runner_availability(model)
    if availability["type"] != "available":
        return None
    options, side = runner_field_configuration(model)
    return canonical_tree(
        {
            "planner": "fixed-field-v1",
            "side_model_px": side,
            "alignment_model_px": 16,
            "small_roi_context": False,
            "tile_overlap": options["tile_overlap"],
            "aggregation": FIELD_AGGREGATION_ID,
            "preprocessing": FIELD_PREPROCESSING_ID,
        }
    )


def runner_field_configuration(model: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """Return wrapper arguments and the warmup side for an already admitted model.

    Warmup uses the declared preferred tile, not a new field-size policy or a
    resource-fit guarantee. The same captured arguments serve every later call.
    This projection neither expands source context nor selects outer WSI windows.
    """

    profile = model["input"].get("fieldProfile")
    if profile is not None:
        return {"field_profile": canonical_tree(profile)}, profile[
            "preferredTileSideModelPx"
        ]
    default_side, default_overlap = _FIXED_FIELD_DEFAULTS[model["arch"]]
    params = model.get("params", {})
    side = params.get("bsize", default_side)
    return {
        "bsize": side,
        "tile_overlap": params.get("tile_overlap", default_overlap),
    }, side


@lru_cache(maxsize=1)
def runner_registry() -> dict[str, Any]:
    """Return the cached, validated runner registry and its definitions digest.

    Callers must not modify the returned dictionary.
    """

    definitions = {
        architecture: _definition(architecture)
        for architecture in sorted(_RUNNER_DEFINITIONS)
    }
    preimage = {"definitions": definitions}
    return canonical_tree(
        {
            "registry_digest": canonical_digest(preimage),
            "definitions": definitions,
        }
    )


def _runner_build(
    architecture: str, network_precision: NetworkPrecision
) -> dict[str, str]:
    return runtime_build(
        f"histoforge-runner-{architecture}",
        {
            "schema_version": 1,
            "architecture": architecture,
            "network_precision": network_precision,
            "output_types": list(_OUTPUT_TYPES[architecture]),
            "preprocessing_contract_digest": _preprocessing_contract_digest(
                architecture
            ),
            "rasterization_contract_digest": RASTERIZATION_CONTRACT_DIGEST,
        },
    ).identity


def _definition(architecture: str) -> dict[str, Any]:
    deployed = _RUNNER_DEFINITIONS[architecture]
    resources = deployed.resources
    preprocessing_digest = _preprocessing_contract_digest(architecture)
    rasterizer = runtime_build(
        "histoforge-rasterizer",
        {
            "schema_version": 1,
            "rasterization_contract_digest": RASTERIZATION_CONTRACT_DIGEST,
        },
    ).identity
    runner = _runner_build(architecture, deployed.native_network_precision)
    precision_overrides = (
        {
            "bf16": {
                "runner_build": _runner_build(architecture, "bf16"),
                "gpus": list(BF16_INFERENCE_GPUS),
            }
        }
        if deployed.supports_bf16
        else {}
    )
    finite = {
        "max_attempt_execution_seconds": resources.attempt_timeout_seconds,
        "max_request_canonical_bytes": 1_000_000,
        "target_raster": {
            "max_width": 2048,
            "max_height": 2048,
            "max_pixels": 2048 * 2048,
        },
        "max_inclusion_mask_vertices": 100_000,
        "result": {
            "max_transferred_bytes": 3 * 1024**2,
            "max_decoded_bytes": 3 * 1024**2,
            "max_record_bytes": 1_000_000,
            "max_features": 100_000,
            "max_vertices": 1_000_000,
        },
    }
    if architecture == "dinov3-adapter-cellpose":
        finite["target_raster"] = {
            "max_width": 6000,
            "max_height": 6000,
            "max_pixels": 6000**2,
        }
        finite["result"].update(
            max_transferred_bytes=64 * 1024**2,
            max_decoded_bytes=64 * 1024**2,
            max_vertices=5_000_000,
        )
    capabilities = {
        "modes": ["FINITE_CALL"],
        "output_types": list(_OUTPUT_TYPES[architecture]),
        "source_containers": sorted(SOURCE_CONTAINERS),
        "runner_build": runner,
        "rasterizer": {
            "build": rasterizer,
            "rasterization_contract_digest": RASTERIZATION_CONTRACT_DIGEST,
        },
        "preprocessing_contract_digest": preprocessing_digest,
        "finite": finite,
    }
    if architecture in _DURABLE_ARCHITECTURES:
        capabilities.update(
            {
                "modes": ["DURABLE_MODEL_RUN", "FINITE_CALL"],
                "durable_run": _durable_limits(architecture, finite),
            }
        )
    value = {
        "architecture": architecture,
        "modal_class_name": resources.modal_class_name,
        "default_gpu": resources.gpu,
        "cpu_cores": resources.cpu_cores,
        "memory_mib": resources.memory_mib,
        "max_containers": resources.max_containers,
        "max_concurrent_inputs": resources.max_concurrent_inputs,
        "attempt_timeout_seconds": resources.attempt_timeout_seconds,
        "capabilities": capabilities,
        "precision_overrides": precision_overrides,
    }
    if deployed.large_region_support is not None:
        value["large_region_support"] = deployed.large_region_support
    return _validate_definition(value)


def _durable_limits(architecture: str, finite: Mapping[str, Any]) -> dict[str, Any]:
    """Bound one complete input, its processing images, saved cores and result."""

    # Shared admission policy, not a guarantee that every population fits.
    target = {
        "max_width": 500_000,
        "max_height": 500_000,
        "max_pixels": 2**35,
    }
    durable = {
        "run": {
            "max_execution_seconds": MODEL_RUN_MAX_EXECUTION_SECONDS,
            "max_request_canonical_bytes": finite["max_request_canonical_bytes"],
            "max_inputs": 1,
            "max_units": 512,
            "target_raster_per_input": target,
            "processing_raster": {
                "max_width": 12_288,
                "max_height": 12_288,
                "max_pixels": 12_288**2,
            },
            "max_aggregate_target_raster_pixels": target["max_pixels"],
            "max_aggregate_inclusion_mask_vertices": finite[
                "max_inclusion_mask_vertices"
            ],
            "max_recovery_state_bytes_per_unit": 16 * 1024**2,
            "max_recovery_state_bytes_per_run": 4 * 1024**3,
            "intermediate_result": {
                "max_transferred_bytes": 4 * 1024**3,
                "max_decoded_bytes": 8 * 1024**3,
                "max_record_bytes": 1_000_000,
                "max_features": 6_000_000,
                "max_vertices": 400_000_000,
            },
            "result": {
                "max_transferred_bytes": 2 * 1024**3,
                "max_decoded_bytes": 8 * 1024**3,
                "max_record_bytes": 1_000_000,
                "max_features": 6_000_000,
                "max_vertices": 400_000_000,
            },
        },
        "terminal_retention_seconds": _DURABLE_TERMINAL_RETENTION_SECONDS,
    }
    if architecture == "dinov3-adapter-semantic":
        # Dense maps stay at the existing finite window scale. Finalization
        # retains fragments for exact union, so its geometry budget is smaller.
        run = durable["run"]
        geometry = {
            **finite["result"],
            "max_transferred_bytes": 64 * 1024**2,
            "max_decoded_bytes": 64 * 1024**2,
        }
        run.update(
            processing_raster=finite["target_raster"],
            max_recovery_state_bytes_per_unit=finite["result"]["max_transferred_bytes"],
            max_recovery_state_bytes_per_run=64 * 1024**2,
            intermediate_result=geometry,
            result=dict(geometry),
        )
    return durable


def _validate_definition(value: dict[str, Any]) -> dict[str, Any]:
    deployed = _RUNNER_DEFINITIONS[value["architecture"]]
    resources = deployed.resources
    if value["modal_class_name"] != resources.modal_class_name:
        raise ValueError("runner definition class does not match decorator resources")
    if value["default_gpu"] != resources.gpu:
        raise ValueError("runner definition GPU does not match decorator resources")
    resource_fields = {
        "max_containers": resources.max_containers,
        "max_concurrent_inputs": resources.max_concurrent_inputs,
        "attempt_timeout_seconds": resources.attempt_timeout_seconds,
    }
    for field, expected in resource_fields.items():
        if require_positive_int32(value[field], field) != expected:
            raise ValueError(
                f"runner definition {field} does not match decorator resources"
            )
    optional_resource_fields = {
        "cpu_cores": resources.cpu_cores,
        "memory_mib": resources.memory_mib,
    }
    for field, expected in optional_resource_fields.items():
        if value[field] != expected:
            raise ValueError(
                f"runner definition {field} does not match decorator resources"
            )
        if expected is not None:
            require_positive_int32(value[field], field)
    capabilities = validate_runner_capabilities(value["capabilities"])
    expected_modes = (
        ["DURABLE_MODEL_RUN", "FINITE_CALL"]
        if value["architecture"] in _DURABLE_ARCHITECTURES
        else ["FINITE_CALL"]
    )
    if capabilities["modes"] != expected_modes:
        raise ValueError("runner modes do not match implemented inference modes")
    finite = capabilities["finite"]
    if finite["max_attempt_execution_seconds"] != value["attempt_timeout_seconds"]:
        raise ValueError("runner and finite attempt timeouts must be equal")
    result = {
        **value,
        "capabilities": capabilities,
        "precision_overrides": validate_precision_overrides(
            value["precision_overrides"], capabilities["runner_build"]
        ),
    }
    if "large_region_support" in value:
        if "DURABLE_MODEL_RUN" not in capabilities["modes"]:
            raise ValueError("large-region processing requires durable runner support")
        support = value["large_region_support"]
        if set(support) != {"type", "validation_evidence_digest"}:
            raise ValueError("large-region support has missing or unknown fields")
        if support["type"] not in {"core-halo", "global-decode"}:
            raise ValueError("unsupported large-region processing method")
        require_sha256(
            support["validation_evidence_digest"], "validation_evidence_digest"
        )
    return canonical_tree(result)


def _validate_model_configuration(
    model: dict[str, Any], definition: dict[str, Any]
) -> None:
    architecture = model["arch"]
    if model["weights"]["type"] == "local":
        raise ValueError("working local weights must be published before execution")

    params = model.get("params", {})
    allowed = _PARAMS_BY_ARCHITECTURE[architecture]
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"runner does not consume model params: {unknown}")
    if (
        architecture
        in {
            "dinov3-adapter-cellpose",
            "dinov3-adapter-hovernet",
        }
        and "min_instance_pixels" not in params
    ):
        raise ValueError("params.min_instance_pixels is required")
    if "min_instance_pixels" in params:
        require_nonnegative_int32(
            params["min_instance_pixels"], "params.min_instance_pixels"
        )

    num_classes = require_nonnegative_int32(
        params.get("num_classes"), "params.num_classes"
    )
    output = model["output"]
    output_type = output["type"]
    if output_type == "instance_segmentation":
        if num_classes != 0 or len(output["classes"]) != 1:
            raise ValueError(
                "unclassified instance output requires num_classes=0 and one class"
            )
    elif output_type in {
        "instance_segmentation_and_classification",
        "dense_classification",
    }:
        channels = model_output_class_channels(output)
        if num_classes != len(channels):
            raise ValueError(
                "params.num_classes must equal the complete output class-channel count"
            )
        if channels[0]["role"] != "background":
            raise ValueError(
                "current DINOv3 class heads require background channel zero"
            )
        if sum(item["role"] == "background" for item in channels) != 1:
            raise ValueError(
                "current DINOv3 class heads require one background channel"
            )

    for field in ("bsize", "batch_size", "object_size", "ksize"):
        if field in params:
            require_positive_int32(params[field], f"params.{field}")
    if "ksize" in params and (params["ksize"] % 2 == 0 or params["ksize"] > 31):
        raise ValueError("params.ksize must be odd and at most 31")
    if "morph_kernel" in params:
        kernel = require_positive_int32(params["morph_kernel"], "params.morph_kernel")
        if kernel % 2 == 0:
            raise ValueError("params.morph_kernel must be odd")
    for field in ("tile_overlap", "edge_threshold", "fg_threshold"):
        if field in params:
            number = _finite_number(params[field], f"params.{field}")
            if number < 0 or number >= 1:
                raise ValueError(f"params.{field} must be in [0, 1)")
    field_profile = model["input"].get("fieldProfile")
    if field_profile is None:
        bsize = params.get("bsize", _FIXED_FIELD_DEFAULTS[architecture][0])
        if bsize % 16 != 0:
            raise ValueError("params.bsize must be divisible by 16")
    else:
        for field, expected in (
            ("planner", VARIABLE_FIELD_PLANNER_ID),
            (
                "outerArtificialPaddingModelPx",
                VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX,
            ),
            ("aggregation", FIELD_AGGREGATION_ID),
            ("preprocessing", FIELD_PREPROCESSING_ID),
        ):
            if field_profile[field] != expected:
                raise ValueError(f"unsupported input.fieldProfile.{field}")
        if field_profile["alignmentModelPx"] != 16:
            raise ValueError("input.fieldProfile.alignmentModelPx must equal 16")


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be finite")
    return number


def _unavailable(code: str, message: str) -> dict[str, str]:
    return {"type": "unavailable", "code": code, "message": message}


@lru_cache(maxsize=len(_RUNNER_DEFINITIONS))
def _preprocessing_contract_digest(architecture: str) -> str:
    source = (
        Path(__file__).parents[2]
        / "_vendor"
        / "histoforge_models"
        / "pathology"
        / "normalization.py"
    ).read_bytes()
    return canonical_digest(
        {
            "schema_version": 1,
            "input": "SRGB_UINT8",
            "normalization": FIELD_PREPROCESSING_ID,
            "scope": (
                {
                    "finite": "complete-supplied-processing-image-before-neural-tiling",
                    "core-halo": "complete-logical-target-before-processing-windows",
                }
                if uses_target_normalization(architecture, "core-halo")
                else "complete-supplied-processing-image-before-neural-tiling"
            ),
            "support": "real-target-pixel-centers-within-physical-slide",
            "padding": "excluded-from-statistics-and-normalized-zero",
            "implementation_sha256": sha256_bytes(source),
        }
    )
