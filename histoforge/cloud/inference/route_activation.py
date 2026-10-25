"""Current installed-model resolution shared by inference route families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.architectures.registry import (
    resolve_runner_cls_name,
    runner_precision_for_build,
)
from histoforge.cloud.http_input import canonical_path_segment
from histoforge.cloud.inference.activation import (
    installed_backend_identity,
    installed_model_descriptor,
)
from histoforge.cloud.inference.execution import require_activation_identity
from histoforge.cloud.inference.gpu import require_supported_inference_gpu
from histoforge.cloud.inference.identity import validate_backend_model_identity
from histoforge.cloud.modules.schema import validate_model_id
from histoforge.cloud.modules.training_evidence import training_result_is_visible
from histoforge.module_ref import ModuleRef


def load_current_model_activation(
    current_modules: Callable[[], Any],
    module_ref: str,
    model_id: str,
    training_result_visible: Callable[[str], bool],
) -> dict[str, Any]:
    """Reload and resolve one exact executable model off an event loop."""

    with current_modules() as current:
        return current_model_activation(
            current, module_ref, model_id, training_result_visible
        )


def current_model_activation(
    current: Any,
    module_ref: str,
    model_id: str,
    training_result_visible: Callable[[str], bool],
) -> dict[str, Any]:
    """Resolve one current installed model, capability set, and Modal class."""

    try:
        record = current.read_record(module_ref)
        recipe = current.load_recipe(module_ref)
    except FileNotFoundError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_NOT_FOUND,
            "the requested module does not exist",
            resource_id=module_ref,
        ) from error
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "installed module content is internally inconsistent",
            resource_id=module_ref,
        ) from error
    if not training_result_is_visible(record, training_result_visible):
        raise GatewayFailure(
            GatewayErrorCode.MODULE_VERSION_UNAVAILABLE,
            "the trained module is not available until its run succeeds",
            resource_id=module_ref,
        )
    try:
        descriptor = installed_model_descriptor(record, recipe, model_id)
    except KeyError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_NOT_FOUND,
            "the requested model does not exist in this module",
            resource_id=f"{module_ref}/{model_id}",
        ) from error
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "installed model content is internally inconsistent",
            resource_id=f"{module_ref}/{model_id}",
        ) from error
    availability = descriptor["runner_availability"]
    if availability["type"] != "available":
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            availability["message"],
            resource_id=f"{module_ref}/{model_id}",
        )
    try:
        backend = installed_backend_identity(record, recipe, model_id)
    except (KeyError, ValueError) as error:
        raise GatewayFailure(
            GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
            "installed model content is internally inconsistent",
            resource_id=f"{module_ref}/{model_id}",
        ) from error
    class_name = resolve_runner_cls_name(backend["architecture"])
    if class_name is None:
        raise GatewayFailure(
            GatewayErrorCode.UNSUPPORTED_CAPABILITY,
            "the selected model has no deployed runner",
            resource_id=f"{module_ref}/{model_id}",
        )
    result = {
        "backend": backend,
        "capabilities": availability["capabilities"],
        "modal_class_name": class_name,
    }
    if "large_region_profile" in descriptor:
        result["large_region_profile"] = descriptor["large_region_profile"]
    return result


def resolve_requested_inference_activation(
    expected_backend: Any, activation: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the captured GPU and runner build at gateway route admission.

    Runners receive the returned GPU as part of their warm-container key and
    reconstruct this exact backend identity independently. Worker-side identity
    checks remain strict.
    """

    expected = validate_backend_model_identity(expected_backend)
    current = validate_backend_model_identity(activation["backend"])
    require_activation_identity(current, activation)
    resolved_gpu = require_supported_inference_gpu(expected["gpu"], current["gpu"])
    selected_build = expected["runner_build"]
    if selected_build != current["runner_build"]:
        try:
            runner_precision_for_build(
                current["architecture"], selected_build, resolved_gpu
            )
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.MODULE_IDENTITY_MISMATCH,
                "the selected runner build or GPU is not available for this precision",
                resource_id=(
                    f"{current['module']['module_ref']}/{current['model_id']}"
                ),
            ) from error
    return {
        **activation,
        "backend": {**current, "gpu": resolved_gpu, "runner_build": selected_build},
        "capabilities": {**activation["capabilities"], "runner_build": selected_build},
    }


def model_route_identity(
    request: Any,
    owner: str,
    name: str,
    model_id: str,
    *,
    trailing_segments: int = 0,
) -> tuple[str, str]:
    """Decode one canonical module/model route without spelling aliases."""

    raw_owner = canonical_path_segment(
        request, owner, trailing_segments=trailing_segments + 2
    )
    raw_name = canonical_path_segment(
        request, name, trailing_segments=trailing_segments + 1
    )
    raw_model = canonical_path_segment(
        request, model_id, trailing_segments=trailing_segments
    )
    try:
        return str(ModuleRef(raw_owner, raw_name)), validate_model_id(raw_model)
    except ValueError as error:
        raise GatewayFailure(
            GatewayErrorCode.VALIDATION_ERROR,
            "inference route contains an invalid module or model identity",
        ) from error
