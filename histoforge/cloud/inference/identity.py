"""Exact module, producer, rasterizer, and backend execution identities.

These pure schema-1 validators are shared by finite inference, durable model
runs, and committed provenance.  They own identity shape and normalization,
not module lookup, runner dispatch, source IO, or lifecycle state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from histoforge.build_identity import parse_software_build_identity
from histoforge.canonical_json import (
    canonical_digest,
    canonical_tree,
    require_nonblank_string,
    require_sha256,
)
from histoforge.cloud.modules.schema import validate_commit_sha, validate_model_id
from histoforge.module_ref import ModuleRef, workspace_training_run_id

_MODULE_CONTENT_FIELDS = {
    "module_ref",
    "recipe_digest",
    "content_digest",
}
_MODULE_CONTENT_FIELDS_WITH_COMMIT = _MODULE_CONTENT_FIELDS | {"resolved_commit_sha"}
_STEP_IDENTITY_FIELDS = {"module_ref", "step_id"}
_RASTERIZER_FIELDS = {"build", "rasterization_contract_digest"}
_BACKEND_MODEL_FIELDS = {
    "module",
    "model_id",
    "architecture",
    "gpu",
    "model_config_digest",
    "artifact_digests",
    "runner_build",
    "rasterizer",
    "preprocessing_contract_digest",
}


def validate_module_content_identity(value: Any) -> dict[str, Any]:
    """Validate one exact installed module-content identity."""

    identity = _require_object(value, "module content identity")
    expected = (
        _MODULE_CONTENT_FIELDS_WITH_COMMIT
        if "resolved_commit_sha" in identity
        else _MODULE_CONTENT_FIELDS
    )
    _require_fields(identity, expected, "module content identity")
    result = {
        "module_ref": _module_ref(identity["module_ref"], "module_ref"),
        "recipe_digest": require_sha256(identity["recipe_digest"], "recipe_digest"),
        "content_digest": require_sha256(identity["content_digest"], "content_digest"),
    }
    if "resolved_commit_sha" in identity:
        result["resolved_commit_sha"] = validate_commit_sha(
            identity["resolved_commit_sha"], "resolved_commit_sha"
        )
    return canonical_tree(result)


def validate_model_step_identity(value: Any) -> dict[str, str]:
    """Validate the module and model step that own one inference output."""

    identity = _require_exact_object(value, _STEP_IDENTITY_FIELDS, "step identity")
    return {
        "module_ref": _module_ref(identity["module_ref"], "module_ref"),
        "step_id": validate_model_id(identity["step_id"], "step_id"),
    }


def validate_rasterizer_identity(value: Any) -> dict[str, Any]:
    """Validate the exact source-decoder and target-grid behavior identity."""

    identity = _require_exact_object(value, _RASTERIZER_FIELDS, "rasterizer identity")
    return canonical_tree(
        {
            "build": parse_software_build_identity(identity["build"]),
            "rasterization_contract_digest": require_sha256(
                identity["rasterization_contract_digest"],
                "rasterization_contract_digest",
            ),
        }
    )


def validate_backend_model_identity(value: Any) -> dict[str, Any]:
    """Validate the complete exact model/runner/rasterizer execution identity."""

    identity = _require_exact_object(
        value, _BACKEND_MODEL_FIELDS, "backend model identity"
    )
    module = validate_module_content_identity(identity["module"])
    if (
        "resolved_commit_sha" not in module
        and workspace_training_run_id(module["module_ref"]) is None
    ):
        raise ValueError(
            "executable backend module identity requires a resolved commit or "
            "reserved training-result ref"
        )
    model_id = validate_model_id(identity["model_id"], "model_id")
    artifacts = _artifact_digests(identity["artifact_digests"])
    return canonical_tree(
        {
            "module": module,
            "model_id": model_id,
            "architecture": require_nonblank_string(
                identity["architecture"], "architecture"
            ),
            "gpu": require_nonblank_string(identity["gpu"], "gpu"),
            "model_config_digest": require_sha256(
                identity["model_config_digest"], "model_config_digest"
            ),
            "artifact_digests": artifacts,
            "runner_build": parse_software_build_identity(identity["runner_build"]),
            "rasterizer": validate_rasterizer_identity(identity["rasterizer"]),
            "preprocessing_contract_digest": require_sha256(
                identity["preprocessing_contract_digest"],
                "preprocessing_contract_digest",
            ),
        }
    )


def backend_model_identity_digest(value: Any) -> str:
    """Return the canonical digest stored with committed backend identity evidence."""

    return canonical_digest(validate_backend_model_identity(value))


def _artifact_digests(value: Any) -> dict[str, str]:
    artifacts = _require_object(value, "artifact_digests")
    if not artifacts:
        raise ValueError("artifact_digests must contain at least one model artifact")
    result: dict[str, str] = {}
    for role, digest in artifacts.items():
        canonical_role = require_nonblank_string(role, "artifact role")
        result[canonical_role] = require_sha256(digest, f"artifact_digests.{role}")
    return canonical_tree(result)


def _module_ref(value: Any, path: str) -> str:
    text = require_nonblank_string(value, path)
    try:
        return str(ModuleRef.parse(text))
    except ValueError as exc:
        raise ValueError(f"{path} must be a canonical owner/name module ref") from exc


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _require_exact_object(value: Any, fields: set[str], path: str) -> Mapping[str, Any]:
    result = _require_object(value, path)
    _require_fields(result, fields, path)
    return result


def _require_fields(value: Mapping[str, Any], fields: set[str], path: str) -> None:
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{path} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
