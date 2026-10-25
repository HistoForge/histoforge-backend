"""Exact bounded module-catalog projection and exclusive cursor.

The catalog exposes validated recipe/script bytes and installed identities; it
never resolves weights for clients. Model executability comes from the shared
activation owner rather than route-specific architecture checks.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_positive_int32,
    require_positive_safe_integer,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.architectures.registry import validate_precision_overrides
from histoforge.cloud.inference.activation import installed_model_descriptor
from histoforge.cloud.inference.identity import validate_module_content_identity
from histoforge.cloud.inference.plans import validate_large_region_profile
from histoforge.cloud.modules.current import CurrentModules
from histoforge.cloud.modules.files import read_regular_file
from histoforge.cloud.modules.identity import (
    module_content_file_identities,
    module_content_identity,
)
from histoforge.cloud.modules.readme import validate_readme_markdown
from histoforge.cloud.modules.schema import (
    model_artifact_declarations,
    parse_module_definition,
    parse_module_recipe,
    validate_model_id,
)
from histoforge.cloud.modules.training_evidence import (
    training_result_is_visible,
)
from histoforge.cloud.runner_contract import validate_runner_capabilities
from histoforge.cloud.training.capabilities import model_training_availability
from histoforge.cloud.training.contract import validate_training_availability
from histoforge.module_ref import ModuleRef, workspace_training_run_id


class CatalogQueryError(ValueError):
    """Invalid client-owned catalog pagination input."""


class CatalogDescriptorTooLarge(ValueError):
    """A validated module cannot fit even one catalog page."""


_MODULE_FIELDS = {
    "identity",
    "installed_at",
    "installed_module_file_bytes",
    "recipe",
    "scripts",
    "script_set_digest",
    "models",
}
_MODULE_FIELDS_WITH_README = _MODULE_FIELDS | {"readme_markdown"}
_MODEL_FIELDS = {
    "model_config_digest",
    "artifact_digests",
    "runner_availability",
    "precision_overrides",
    "training_availability",
}
_MODEL_FIELDS_WITH_PROFILE = _MODEL_FIELDS | {"large_region_profile"}
_UNAVAILABLE_CODES = {
    "UNKNOWN_ARCHITECTURE",
    "RUNNER_NOT_DEPLOYED",
    "UNSUPPORTED_OUTPUT_TYPE",
    "UNSUPPORTED_INPUT_FEATURE",
    "UNSUPPORTED_MODEL_CONFIGURATION",
}


def catalog_page(
    current: CurrentModules,
    *,
    training_result_visible: Callable[[str], bool],
    limit: int,
    cursor: str | None,
    max_page_size: int,
    max_encoded_bytes: int,
) -> dict[str, Any]:
    """Return one ref-sorted page bounded by count and exact response bytes."""

    try:
        requested = require_positive_int32(limit, "catalog limit")
        maximum = require_positive_int32(max_page_size, "max_catalog_page_size")
        if requested > maximum:
            raise ValueError("catalog limit exceeds max_catalog_page_size")
        after = _decode_cursor(cursor) if cursor is not None else None
    except ValueError as error:
        raise CatalogQueryError(str(error)) from error
    byte_limit = require_positive_safe_integer(
        max_encoded_bytes, "max_catalog_encoded_bytes"
    )
    refs = []
    for ref in current.list_module_refs():
        if after is not None and ref <= after:
            continue
        record = current.read_record(ref)
        if training_result_is_visible(record, training_result_visible):
            refs.append(ref)

    modules: list[dict[str, Any]] = []
    for index, module_ref in enumerate(refs):
        if len(modules) == requested:
            break
        descriptor = installed_module_descriptor(current, module_ref)
        more_after_candidate = index + 1 < len(refs)
        candidate_modules = [*modules, descriptor]
        candidate = _page(
            candidate_modules,
            _encode_cursor(module_ref) if more_after_candidate else None,
        )
        if len(canonical_bytes(candidate)) > byte_limit:
            if not modules:
                raise ValueError(
                    f"catalog descriptor for {module_ref} exceeds page byte limit"
                )
            break
        modules = candidate_modules

    if not modules:
        return _page([], None)
    last_ref = modules[-1]["identity"]["module_ref"]
    has_more = any(ref > last_ref for ref in refs)
    page = _page(modules, _encode_cursor(last_ref) if has_more else None)
    if len(canonical_bytes(page)) > byte_limit:
        raise ValueError("catalog page exceeds max_catalog_encoded_bytes")
    return page


def installed_module_descriptor(
    current: CurrentModules, module_ref: str
) -> dict[str, Any]:
    """Project one complete current module after exact-byte verification."""

    record = current.read_record(module_ref)
    recipe = current.load_recipe(module_ref)
    scripts = current.script_sources(module_ref)
    module_file_bytes = sum(
        identity["size_bytes"]
        for identity in module_content_file_identities(
            current.module_dir(module_ref), recipe, record
        )
    )
    return build_installed_module_descriptor(
        record,
        recipe,
        scripts,
        installed_module_file_bytes=module_file_bytes,
    )


def build_installed_module_descriptor(
    record: dict[str, Any],
    recipe: dict[str, Any],
    scripts: dict[str, str],
    *,
    installed_module_file_bytes: int,
) -> dict[str, Any]:
    """Build one descriptor from already captured exact module content."""

    definition = parse_module_definition(canonical_bytes(recipe), scripts)
    if definition["recipe_digest"] != record["recipe_digest"]:
        raise ValueError("catalog recipe does not match installed identity")
    models = {}
    for model_id, model in recipe["models"].items():
        descriptor = installed_model_descriptor(record, recipe, model_id)
        descriptor["training_availability"] = model_training_availability(
            recipe,
            model_id,
            descriptor["runner_availability"],
        )
        models[model_id] = descriptor
    return validate_installed_module_descriptor(
        {
            "identity": module_content_identity(record),
            "installed_at": record["installed_at"],
            "installed_module_file_bytes": installed_module_file_bytes,
            "recipe": recipe,
            "scripts": scripts,
            "script_set_digest": definition["script_set_digest"],
            "models": models,
            **(
                {"readme_markdown": record["readme_markdown"]}
                if "readme_markdown" in record
                else {}
            ),
        }
    )


def validate_catalog_page(value: Any) -> dict[str, Any]:
    """Validate one complete target catalog page and all nested identities."""

    page = _object(value, "module catalog page")
    expected = {"modules", "next_cursor"} if "next_cursor" in page else {"modules"}
    _fields(page, expected, "module catalog page")
    if not isinstance(page["modules"], list):
        raise ValueError("modules must be an array")
    modules = [validate_installed_module_descriptor(item) for item in page["modules"]]
    refs = [item["identity"]["module_ref"] for item in modules]
    if refs != sorted(set(refs)):
        raise ValueError("catalog modules must be unique and ref-sorted")
    result: dict[str, Any] = {"modules": modules}
    if "next_cursor" in page:
        if not modules:
            raise ValueError("empty catalog page cannot have next_cursor")
        cursor = require_nonblank_string(page["next_cursor"], "next_cursor")
        if _decode_cursor(cursor) != refs[-1]:
            raise ValueError("next_cursor must encode the page's exclusive last ref")
        result["next_cursor"] = cursor
    return canonical_tree(result)


def validate_installed_module_descriptor(value: Any) -> dict[str, Any]:
    """Validate one descriptor and recompute recipe, script, and model digests."""

    item = _object(value, "installed module descriptor")
    _fields(
        item,
        _MODULE_FIELDS_WITH_README if "readme_markdown" in item else _MODULE_FIELDS,
        "installed module descriptor",
    )
    identity = validate_module_content_identity(item["identity"])
    readme = {}
    if "readme_markdown" in item:
        if identity.get("resolved_commit_sha") is None:
            raise ValueError("saved README requires a resolved HF commit")
        readme["readme_markdown"] = validate_readme_markdown(item["readme_markdown"])
    scripts = _string_map(item["scripts"], "scripts")
    definition = parse_module_definition(canonical_bytes(item["recipe"]), scripts)
    if definition["recipe_digest"] != identity["recipe_digest"]:
        raise ValueError("catalog recipe_digest does not match recipe")

    raw_models = _object(item["models"], "models")
    if set(raw_models) != set(definition["recipe"]["models"]):
        raise ValueError("catalog model descriptors do not match recipe models")
    expected_roles: dict[str, set[str]] = {
        model_id: set() for model_id in definition["recipe"]["models"]
    }
    for declaration in model_artifact_declarations(definition["recipe"]):
        expected_roles[declaration["model_id"]].add(declaration["role"])
    models = {
        validate_model_id(model_id): _validate_model_descriptor(
            descriptor,
            definition["model_config_digests"][model_id],
            expected_roles[model_id],
            definition["recipe"]["models"][model_id]["input"],
        )
        for model_id, descriptor in raw_models.items()
    }
    if (
        identity.get("resolved_commit_sha") is None
        and workspace_training_run_id(identity["module_ref"]) is None
        and any(
            model["runner_availability"]["type"] == "available"
            for model in models.values()
        )
    ):
        raise ValueError(
            "available catalog model requires a commit or reserved training-result ref"
        )
    script_set_digest = require_sha256(item["script_set_digest"], "script_set_digest")
    if script_set_digest != definition["script_set_digest"]:
        raise ValueError("catalog script_set_digest does not match scripts")
    return canonical_tree(
        {
            "identity": identity,
            "installed_at": require_rfc3339_millis(
                item["installed_at"], "installed_at"
            ),
            "installed_module_file_bytes": require_nonnegative_safe_integer(
                item["installed_module_file_bytes"],
                "installed_module_file_bytes",
            ),
            "recipe": definition["recipe"],
            "scripts": scripts,
            "script_set_digest": script_set_digest,
            "models": models,
            **readme,
        }
    )


def _validate_model_descriptor(
    value: Any,
    expected_digest: str,
    expected_artifact_roles: set[str],
    model_input: dict[str, Any],
) -> dict[str, Any]:
    descriptor = _object(value, "installed model descriptor")
    expected = (
        _MODEL_FIELDS_WITH_PROFILE
        if "large_region_profile" in descriptor
        else _MODEL_FIELDS
    )
    _fields(descriptor, expected, "installed model descriptor")
    digest = require_sha256(descriptor["model_config_digest"], "model_config_digest")
    if digest != expected_digest:
        raise ValueError("model_config_digest does not match recipe model entry")
    artifact_digests = _digest_map(descriptor["artifact_digests"], "artifact_digests")
    if set(artifact_digests) != expected_artifact_roles:
        raise ValueError("artifact_digests do not match recipe artifact roles")
    result = {
        "model_config_digest": digest,
        "artifact_digests": artifact_digests,
        "runner_availability": _validate_runner_availability(
            descriptor["runner_availability"]
        ),
        "training_availability": validate_training_availability(
            descriptor["training_availability"],
            DEPLOYMENT_CONFIG.training_api_limits.report.max_text_bytes,
        ),
    }
    availability = result["runner_availability"]
    result["precision_overrides"] = validate_precision_overrides(
        descriptor["precision_overrides"],
        availability["capabilities"]["runner_build"]
        if availability["type"] == "available"
        else None,
    )
    if (
        result["training_availability"]["type"] == "available"
        and result["runner_availability"]["type"] != "available"
    ):
        raise ValueError("trainable catalog model requires an available runner")
    if "large_region_profile" in descriptor:
        result["large_region_profile"] = validate_large_region_profile(
            descriptor["large_region_profile"]
        )
        profile = result["large_region_profile"]
        if profile["type"] == "core-halo":
            settings = model_input.get("largeRegion")
            if settings is None or (
                profile["core_width_px"] != settings["coreWidthModelPx"]
                or profile["core_height_px"] != settings["coreHeightModelPx"]
                or profile["halo_px"] != settings["haloPerSideModelPx"]
            ):
                raise ValueError("core-halo profile does not match input.largeRegion")
        availability = result["runner_availability"]
        if (
            availability["type"] != "available"
            or "DURABLE_MODEL_RUN" not in availability["capabilities"]["modes"]
        ):
            raise ValueError(
                "large_region_profile requires available durable runner capabilities"
            )
    return canonical_tree(result)


def _validate_runner_availability(value: Any) -> dict[str, Any]:
    availability = _object(value, "runner_availability")
    kind = availability.get("type")
    if kind == "available":
        _fields(
            availability,
            {"type", "capabilities"},
            "available runner_availability",
        )
        return canonical_tree(
            {
                "type": "available",
                "capabilities": validate_runner_capabilities(
                    availability["capabilities"]
                ),
            }
        )
    if kind == "unavailable":
        _fields(
            availability,
            {"type", "code", "message"},
            "unavailable runner_availability",
        )
        code = require_nonblank_string(availability["code"], "code")
        if code not in _UNAVAILABLE_CODES:
            raise ValueError(f"unknown runner unavailable code: {code}")
        return {
            "type": "unavailable",
            "code": code,
            "message": require_nonblank_string(availability["message"], "message"),
        }
    raise ValueError(f"unsupported runner_availability type: {kind!r}")


def require_directory_descriptor_fits(
    module_dir: Path,
    record: dict[str, Any],
    max_encoded_bytes: int,
) -> None:
    """Reject a candidate that cannot fit one target catalog response page."""

    recipe = parse_module_recipe(read_regular_file(module_dir, "recipe.json"))
    scripts = {
        step["script"]: read_regular_file(module_dir, step["script"]).decode("utf-8")
        for step in recipe["steps"]
        if "script" in step
    }
    module_file_bytes = sum(
        identity["size_bytes"]
        for identity in module_content_file_identities(module_dir, recipe, record)
    )
    descriptor = build_installed_module_descriptor(
        record,
        recipe,
        scripts,
        installed_module_file_bytes=module_file_bytes,
    )
    page = _page([descriptor], _encode_cursor(record["module_ref"]))
    if len(canonical_bytes(page)) > require_positive_safe_integer(
        max_encoded_bytes, "max_catalog_encoded_bytes"
    ):
        raise CatalogDescriptorTooLarge(
            "installed module descriptor exceeds catalog byte limit"
        )


def _page(modules: list[dict[str, Any]], cursor: str | None) -> dict[str, Any]:
    value: dict[str, Any] = {"modules": modules}
    if cursor is not None:
        value["next_cursor"] = cursor
    return canonical_tree(value)


def _string_map(value: Any, path: str) -> dict[str, str]:
    items = _object(value, path)
    result = {}
    for key, item in items.items():
        name = require_nonblank_string(key, f"{path} key")
        if not isinstance(item, str):
            raise ValueError(f"{path}.{name} must be a string")
        result[name] = item
    return canonical_tree(result)


def _digest_map(value: Any, path: str) -> dict[str, str]:
    items = _object(value, path)
    if not items:
        raise ValueError(f"{path} must be nonempty")
    return canonical_tree(
        {
            require_nonblank_string(key, f"{path} key"): require_sha256(
                item, f"{path}.{key}"
            )
            for key, item in items.items()
        }
    )


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _fields(value: dict[str, Any], fields: set[str], path: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{path} has missing or unknown fields")


def _encode_cursor(module_ref: str) -> str:
    canonical = str(ModuleRef.parse(module_ref))
    return (
        base64.urlsafe_b64encode(canonical.encode("ascii")).decode("ascii").rstrip("=")
    )


def _decode_cursor(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("catalog cursor must be a nonempty string")
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        ).decode("ascii")
    except (binascii.Error, UnicodeDecodeError) as error:
        raise ValueError("catalog cursor is malformed") from error
    canonical = str(ModuleRef.parse(decoded))
    if _encode_cursor(canonical) != value:
        raise ValueError("catalog cursor is not canonically encoded")
    return canonical
