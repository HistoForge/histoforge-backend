"""Executable module-content contract.

The author-owned schema remains ``custom_docs/contracts/modules.md``. This module
turns that document into strict parsing, recipe-order validation, stable step IDs,
and exact recipe/model/script-set identities. It does not decide whether a deployed
runner can execute an otherwise valid model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    is_blank,
    parse_json,
    require_int32,
    require_path_class_name,
    require_sha256,
    sha256_bytes,
)
from histoforge.cloud.modules.script_headers import (
    parse_metric_script_header,
    parse_ordinary_script_header,
)
from histoforge.cloud.protocol_versions import MODULE_SCHEMA_VERSION
from histoforge.module_ref import ModuleRef

METRICS_SCRIPT_NAME = "metrics.groovy"
VARIABLE_FIELD_PLANNER_ID = "variable-field-v2"
VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX = 8

_TOP_LEVEL_FIELDS = {
    "schemaVersion",
    "name",
    "description",
    "upstream",
    "trainingReportDigest",
    "models",
    "steps",
}
_MODEL_FIELDS = {"arch", "gpu", "weights", "input", "output", "params", "backbone"}
_INPUT_FIELDS = {
    "type",
    "micronsPerPixel",
    "fieldProfile",
    "largeRegion",
    "framing",
    "acceptCriteria",
}
_INSTANCE_OUTPUT_FIELDS = {"type", "classes"}
_CLASS_OUTPUT_FIELDS = {"type", "classChannels"}
_BACKGROUND_CHANNEL_FIELDS = {"channel", "role"}
_LABELED_CHANNEL_FIELDS = {"channel", "role", "pathClass"}
_OUTPUT_TYPES = {
    "point_detection",
    "instance_segmentation",
    "instance_segmentation_and_classification",
    "dense_classification",
    "region_classification",
}
_CLASS_CHANNEL_ROLES = {"background", "emitted", "training_only"}
_MODEL_ID = re.compile(r"[a-z][a-z0-9_]*")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_PROFILE_ID = re.compile(r"[!-~]{1,128}")
_COMPOSABLE_HEAD_ARCHS = {"abmil", "mlp", "linear"}
TRAINING_REPORT_NAME = "training-report.json"


def parse_module_definition(
    recipe_source: str | bytes,
    scripts: Mapping[str, str | bytes],
) -> dict[str, Any]:
    """Validate exact recipe/script bytes and return their deterministic identities."""

    recipe = parse_module_recipe(recipe_source)

    script_bytes = {
        name: _script_bytes(name, source) for name, source in scripts.items()
    }
    referenced_scripts = [
        step["script"] for step in recipe["steps"] if "script" in step
    ]
    expected = set(referenced_scripts)
    actual = set(script_bytes)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"script closure mismatch: missing={missing}, extra={extra}")

    script_contracts: dict[str, dict[str, Any]] = {}
    for filename in referenced_scripts:
        source = script_bytes[filename].decode("utf-8")
        if filename == METRICS_SCRIPT_NAME:
            script_contracts[filename] = {
                "type": "metrics",
                "definitions": list(parse_metric_script_header(source)),
            }
        else:
            script_contracts[filename] = {
                "type": "mutation",
                "classes": list(parse_ordinary_script_header(source)),
            }

    script_digests = {
        filename: sha256_bytes(source) for filename, source in script_bytes.items()
    }
    return {
        "recipe": recipe,
        "recipe_digest": canonical_digest(recipe),
        "model_config_digests": {
            model_id: canonical_digest(model)
            for model_id, model in recipe["models"].items()
        },
        "script_set_digest": canonical_digest(script_digests),
        "step_ids": [step_id(step) for step in recipe["steps"]],
        "script_contracts": script_contracts,
    }


def parse_module_recipe(recipe_source: str | bytes) -> dict[str, Any]:
    """Validate one recipe before any recipe-named file is opened.

    Validation uses the author's raw JSON tree so integer-only schema fields do not
    accept decimal spellings such as ``schemaVersion: 3.0``. Execution and identity
    use the tree reparsed from JCS bytes so equal digests cannot retain different
    numeric semantics (for example ``bsize: 1024`` versus ``1024.0``).
    """

    raw_recipe = parse_json(recipe_source)
    _validate_recipe(raw_recipe)
    return parse_json(canonical_bytes(raw_recipe))


def executable_module_paths(recipe: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the executable files for a validated recipe.

    A referenced or local artifact is not a repository file; it is represented by
    its resolved artifact identity instead. ``module_content_paths`` extends this
    closure with non-executable evidence owned by the module root.
    """

    paths = {"recipe.json"}
    for model in recipe["models"].values():
        weights = model["weights"]
        if weights["type"] == "vendored":
            paths.add(weights["file"])
    for step in recipe["steps"]:
        if "script" in step:
            paths.add(step["script"])
    return tuple(sorted(paths))


def module_content_paths(recipe: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every schema-owned file in one validated module content root."""

    paths = set(executable_module_paths(recipe))
    if "trainingReportDigest" in recipe:
        paths.add(TRAINING_REPORT_NAME)
    return tuple(sorted(paths))


def model_artifact_declarations(
    recipe: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return every schema-owned model artifact role in deterministic order."""

    declarations: list[dict[str, Any]] = []
    for model_id, model in recipe["models"].items():
        declarations.append(
            {"model_id": model_id, "role": "weights", "source": model["weights"]}
        )
        if "backbone" in model:
            declarations.append(
                {
                    "model_id": model_id,
                    "role": "backbone.weights",
                    "source": model["backbone"]["weights"],
                }
            )
    declarations.sort(key=lambda item: (item["model_id"], item["role"]))
    return tuple(declarations)


def step_id(step: Mapping[str, Any]) -> str:
    """Return the stable ID of a previously validated recipe step."""

    if "model" in step:
        return step["model"]
    return step["script"]


def validate_model_id(value: Any, path: str = "model_id") -> str:
    """Validate one schema-owned local model identifier."""

    model_id = _nonblank(value, path)
    if not _MODEL_ID.fullmatch(model_id):
        raise ValueError(f"{path} must be a canonical model id")
    return model_id


def validate_model_output_type(value: Any, path: str = "output_type") -> str:
    """Validate one module-owned closed model output type."""

    output_type = _nonblank(value, path)
    if output_type not in _OUTPUT_TYPES:
        raise ValueError(f"{path} is unsupported")
    return output_type


def validate_model_output(value: Any, path: str = "output") -> dict[str, Any]:
    """Validate and normalize one recipe-owned model output contract."""

    output = _object(value, path)
    output_type = validate_model_output_type(output.get("type"), f"{path}.type")
    if output_type == "instance_segmentation":
        _fields(
            output,
            _INSTANCE_OUTPUT_FIELDS,
            _INSTANCE_OUTPUT_FIELDS,
            path,
        )
        classes = _array(output["classes"], f"{path}.classes")
        if len(classes) != 1:
            raise ValueError(
                f"{path}.classes must contain exactly one post-decoding class"
            )
        return {
            "type": output_type,
            "classes": [require_path_class_name(classes[0], f"{path}.classes[0]")],
        }

    _fields(output, _CLASS_OUTPUT_FIELDS, _CLASS_OUTPUT_FIELDS, path)
    channels = _array(output["classChannels"], f"{path}.classChannels")
    if not channels:
        raise ValueError(f"{path}.classChannels must be nonempty")
    normalized = [
        _validate_class_channel(item, f"{path}.classChannels[{index}]")
        for index, item in enumerate(channels)
    ]
    normalized.sort(key=lambda item: item["channel"])
    actual_channels = [item["channel"] for item in normalized]
    if actual_channels != list(range(len(normalized))):
        raise ValueError(f"{path}.classChannels must be contiguous from channel zero")
    if sum(item["role"] == "background" for item in normalized) > 1:
        raise ValueError(f"{path}.classChannels permits at most one background")
    if not any(item["role"] == "emitted" for item in normalized):
        raise ValueError(f"{path}.classChannels must contain an emitted channel")
    path_classes = [item["pathClass"] for item in normalized if "pathClass" in item]
    if len(path_classes) != len(set(path_classes)):
        raise ValueError(f"{path}.classChannels contains duplicate PathClasses")
    if output_type == "point_detection" and any(
        item["role"] != "emitted" for item in normalized
    ):
        raise ValueError(f"{path}: point detection requires emitted-only channels")
    return {"type": output_type, "classChannels": normalized}


def model_output_geometry_kind(value: Mapping[str, Any]) -> str:
    """Select result geometry from the captured model, including empty results."""

    return "point" if value["type"] == "point_detection" else "polygonal"


def model_output_class_channels(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return normalized neural class-channel records, or none for a pure decoder."""

    output = validate_model_output(value)
    return tuple(output.get("classChannels", ()))


def model_output_target_channels(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return every non-background supervised channel in numeric channel order."""

    return tuple(
        item
        for item in model_output_class_channels(value)
        if item["role"] != "background"
    )


def model_output_target_classes(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every PathClass usable as a training target."""

    output = validate_model_output(value)
    if output["type"] == "instance_segmentation":
        return tuple(output["classes"])
    return tuple(item["pathClass"] for item in model_output_target_channels(output))


def model_output_target_channel_by_class(
    value: Mapping[str, Any],
) -> dict[str, int]:
    """Map each supervised class-head PathClass to its explicit neural channel."""

    output = validate_model_output(value)
    if output["type"] == "instance_segmentation":
        return {}
    return {
        item["pathClass"]: item["channel"]
        for item in model_output_target_channels(output)
    }


def model_output_emitted_classes(value: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only PathClasses that normal inference may emit."""

    output = validate_model_output(value)
    if output["type"] == "instance_segmentation":
        return tuple(output["classes"])
    return tuple(
        item["pathClass"]
        for item in model_output_class_channels(output)
        if item["role"] == "emitted"
    )


def validate_commit_sha(value: Any, path: str = "commit_sha") -> str:
    """Validate one immutable full lowercase Git commit SHA."""

    revision = _nonblank(value, path)
    if not _COMMIT_SHA.fullmatch(revision):
        raise ValueError(f"{path} must be a full lowercase 40-character commit SHA")
    return revision


def _validate_recipe(value: Any) -> None:
    recipe = _object(value, "recipe")
    _fields(
        recipe,
        _TOP_LEVEL_FIELDS,
        {"schemaVersion", "name", "models", "steps"},
        "recipe",
    )
    version = _integer(recipe["schemaVersion"], "recipe.schemaVersion")
    if version not in {3, MODULE_SCHEMA_VERSION}:
        raise ValueError("recipe.schemaVersion must be 3 or 4")
    _nonblank(recipe["name"], "recipe.name")
    if "description" in recipe:
        description = _nonblank(recipe["description"], "recipe.description")
        if "\n" in description or "\r" in description:
            raise ValueError("recipe.description must be one line")
    if "upstream" in recipe:
        _validate_upstream(recipe["upstream"])
    if "trainingReportDigest" in recipe:
        require_sha256(recipe["trainingReportDigest"], "recipe.trainingReportDigest")

    models = _object(recipe["models"], "recipe.models")
    if not models:
        raise ValueError("recipe.models must be nonempty")
    for model_id, model in models.items():
        validate_model_id(model_id, "model id")
        _validate_model(model_id, model)
        if version < 4 and model["output"]["type"] == "point_detection":
            raise ValueError("point detection requires recipe schemaVersion 4")

    steps = _array(recipe["steps"], "recipe.steps")
    if not steps:
        raise ValueError("recipe.steps must be nonempty")
    _validate_steps(steps, models)


def _validate_model(model_id: str, value: Any) -> None:
    path = f"recipe.models.{model_id}"
    model = _object(value, path)
    _fields(model, _MODEL_FIELDS, {"arch", "gpu", "weights", "input", "output"}, path)
    architecture = _nonblank(model["arch"], f"{path}.arch")
    _nonblank(model["gpu"], f"{path}.gpu")
    _validate_weights(model["weights"], f"{path}.weights", model_id=model_id)
    _validate_input(model["input"], f"{path}.input")
    model["output"] = validate_model_output(model["output"], f"{path}.output")
    if "params" in model:
        params = _object(model["params"], f"{path}.params")
        canonical_bytes(params)
        if "fieldProfile" in model["input"] and {
            "bsize",
            "tile_overlap",
        } & set(params):
            raise ValueError(
                f"{path}.params cannot declare fixed-size fields with "
                "input.fieldProfile"
            )
    if architecture in _COMPOSABLE_HEAD_ARCHS and "backbone" not in model:
        raise ValueError(f"{path}.backbone is required for a composable-head arch")
    if "backbone" in model:
        if architecture not in _COMPOSABLE_HEAD_ARCHS:
            raise ValueError(
                f"{path}.backbone is valid only for a composable-head arch"
            )
        backbone = _object(model["backbone"], f"{path}.backbone")
        _fields(backbone, {"arch", "weights"}, {"arch", "weights"}, f"{path}.backbone")
        _nonblank(backbone["arch"], f"{path}.backbone.arch")
        weights = _object(backbone["weights"], f"{path}.backbone.weights")
        if weights.get("type") != "referenced":
            raise ValueError(
                f"{path}.backbone.weights must be referenced and revision-pinned"
            )
        _validate_weights(weights, f"{path}.backbone.weights")


def _validate_weights(value: Any, path: str, *, model_id: str | None = None) -> None:
    weights = _object(value, path)
    kind = _nonblank(weights.get("type"), f"{path}.type")
    if kind == "vendored":
        _fields(weights, {"type", "file"}, {"type", "file"}, path)
        filename = _safe_relative_file(weights["file"], f"{path}.file", ".safetensors")
        if model_id is not None and filename != f"{model_id}.safetensors":
            raise ValueError(f"{path}.file must be {model_id}.safetensors")
    elif kind == "referenced":
        _fields(weights, {"type", "ref", "revision"}, {"type", "ref", "revision"}, path)
        _module_ref(weights["ref"], f"{path}.ref")
        _commit_sha(weights["revision"], f"{path}.revision")
    elif kind == "local":
        _fields(weights, {"type", "ref"}, {"type", "ref"}, path)
        ref = _module_ref(weights["ref"], f"{path}.ref")
        if not ref.startswith("local/"):
            raise ValueError(f"{path}.ref must use the local/ namespace")
    else:
        raise ValueError(f"{path}.type is not a supported weights source")


def _validate_input(value: Any, path: str) -> None:
    model_input = _object(value, path)
    _fields(model_input, _INPUT_FIELDS, {"type", "micronsPerPixel"}, path)
    input_type = _nonblank(model_input["type"], f"{path}.type")
    _positive_number(model_input["micronsPerPixel"], f"{path}.micronsPerPixel")
    if "largeRegion" in model_input:
        _validate_large_region(model_input["largeRegion"], f"{path}.largeRegion")
    field_profile = (
        validate_variable_field_profile(
            model_input["fieldProfile"], f"{path}.fieldProfile"
        )
        if "fieldProfile" in model_input
        else None
    )

    if input_type == "field":
        if "framing" in model_input or "acceptCriteria" in model_input:
            raise ValueError(f"{path}: field input cannot declare crop-only fields")
    elif input_type == "crop":
        if "framing" not in model_input:
            raise ValueError(f"{path}: crop input requires framing")
        framing = _object(model_input["framing"], f"{path}.framing")
        _fields(framing, {"paddingModelPx"}, {"paddingModelPx"}, f"{path}.framing")
        padding = _integer(framing["paddingModelPx"], f"{path}.framing.paddingModelPx")
        if padding < 0:
            raise ValueError(f"{path}.framing.paddingModelPx must be nonnegative")
        if "acceptCriteria" in model_input:
            _validate_accept_criteria(
                model_input["acceptCriteria"], f"{path}.acceptCriteria"
            )
        if field_profile is not None and field_profile["smallRoiContext"]:
            raise ValueError(
                f"{path}.fieldProfile.smallRoiContext must be false for crop input"
            )
    else:
        raise ValueError(f"{path}.type must be field or crop")


def _validate_large_region(value: Any, path: str) -> None:
    """Validate recipe-owned processing cores and context, in model pixels.

    These rectangles contain many neural fields. They need not be multiples of
    the neural network's alignment. Actual allocation limits belong to the runner.
    """

    settings = _object(value, path)
    fields = {"coreWidthModelPx", "coreHeightModelPx", "haloPerSideModelPx"}
    _fields(settings, fields, fields, path)
    for dimension in ("coreWidthModelPx", "coreHeightModelPx"):
        _positive_integer(settings[dimension], f"{path}.{dimension}")
    halo = _integer(settings["haloPerSideModelPx"], f"{path}.haloPerSideModelPx")
    if halo < 0:
        raise ValueError(f"{path}.haloPerSideModelPx must be nonnegative")


def validate_variable_field_profile(
    value: Any, path: str = "fieldProfile"
) -> dict[str, Any]:
    """Validate the exact module-owned variable-field planner contract."""

    fields = {
        "planner",
        "minSideModelPx",
        "maxSideModelPx",
        "preferredTileSideModelPx",
        "alignmentModelPx",
        "outerArtificialPaddingModelPx",
        "smallRoiContext",
        "tileOverlap",
        "aggregation",
        "preprocessing",
    }
    profile = _object(value, path)
    _fields(profile, fields, fields, path)
    if profile["planner"] != VARIABLE_FIELD_PLANNER_ID:
        raise ValueError(f"{path}.planner must be {VARIABLE_FIELD_PLANNER_ID}")
    minimum = _positive_integer(profile["minSideModelPx"], f"{path}.minSideModelPx")
    maximum = _positive_integer(profile["maxSideModelPx"], f"{path}.maxSideModelPx")
    preferred = _positive_integer(
        profile["preferredTileSideModelPx"], f"{path}.preferredTileSideModelPx"
    )
    alignment = _positive_integer(
        profile["alignmentModelPx"], f"{path}.alignmentModelPx"
    )
    outer_padding = _positive_integer(
        profile["outerArtificialPaddingModelPx"],
        f"{path}.outerArtificialPaddingModelPx",
    )
    if outer_padding != VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX:
        raise ValueError(
            f"{path}.outerArtificialPaddingModelPx must equal "
            f"{VARIABLE_FIELD_OUTER_ARTIFICIAL_PADDING_MODEL_PX}"
        )
    if any(side % alignment for side in (minimum, maximum, preferred)):
        raise ValueError(f"{path} sides must be divisible by alignmentModelPx")
    if not minimum <= preferred <= maximum:
        raise ValueError(f"{path} sides must satisfy minimum <= preferred <= maximum")
    return {
        "planner": VARIABLE_FIELD_PLANNER_ID,
        "minSideModelPx": minimum,
        "maxSideModelPx": maximum,
        "preferredTileSideModelPx": preferred,
        "alignmentModelPx": alignment,
        "outerArtificialPaddingModelPx": outer_padding,
        "smallRoiContext": _boolean(
            profile["smallRoiContext"], f"{path}.smallRoiContext"
        ),
        "tileOverlap": _overlap(profile["tileOverlap"], f"{path}.tileOverlap"),
        "aggregation": _profile_id(profile["aggregation"], f"{path}.aggregation"),
        "preprocessing": _profile_id(profile["preprocessing"], f"{path}.preprocessing"),
    }


def _validate_accept_criteria(value: Any, path: str) -> None:
    criteria = _object(value, path)
    _fields(criteria, {"sizeUmRange", "maxAspectRatio"}, set(), path)
    if not criteria:
        raise ValueError(f"{path} must declare at least one criterion")
    if "sizeUmRange" in criteria:
        size_range = _array(criteria["sizeUmRange"], f"{path}.sizeUmRange")
        if len(size_range) != 2:
            raise ValueError(f"{path}.sizeUmRange must contain [min, max]")
        minimum = _positive_number(size_range[0], f"{path}.sizeUmRange[0]")
        maximum = _positive_number(size_range[1], f"{path}.sizeUmRange[1]")
        if minimum > maximum:
            raise ValueError(f"{path}.sizeUmRange minimum exceeds maximum")
    if "maxAspectRatio" in criteria:
        ratio = _positive_number(criteria["maxAspectRatio"], f"{path}.maxAspectRatio")
        if ratio < 1:
            raise ValueError(f"{path}.maxAspectRatio must be at least 1")


def _validate_class_channel(value: Any, path: str) -> dict[str, Any]:
    channel = _object(value, path)
    role = _nonblank(channel.get("role"), f"{path}.role")
    if role not in _CLASS_CHANNEL_ROLES:
        raise ValueError(f"{path}.role is unsupported")
    expected_fields = (
        _BACKGROUND_CHANNEL_FIELDS if role == "background" else _LABELED_CHANNEL_FIELDS
    )
    _fields(channel, expected_fields, expected_fields, path)
    index = _integer(channel["channel"], f"{path}.channel")
    if index < 0:
        raise ValueError(f"{path}.channel must be nonnegative")
    result: dict[str, Any] = {"channel": index, "role": role}
    if role != "background":
        result["pathClass"] = require_path_class_name(
            channel["pathClass"], f"{path}.pathClass"
        )
    return result


def _validate_steps(steps: list[Any], models: dict[str, Any]) -> None:
    seen_ids: set[str] = set()
    earlier_plain_models: set[str] = set()
    metrics_seen = False
    for index, value in enumerate(steps):
        path = f"recipe.steps[{index}]"
        step = _object(value, path)
        fields = set(step)
        if fields == {"model"}:
            model_id = _known_model(step["model"], models, f"{path}.model")
            if models[model_id]["input"]["type"] != "field":
                raise ValueError(f"{path}: plain model must use field input")
            earlier_plain_models.add(model_id)
        elif fields == {"model", "parentModel", "parentClass"}:
            model_id = _known_model(step["model"], models, f"{path}.model")
            parent_model = _known_model(
                step["parentModel"], models, f"{path}.parentModel"
            )
            if parent_model not in earlier_plain_models:
                raise ValueError(f"{path}.parentModel must name an earlier plain model")
            if models[model_id]["input"]["type"] != "crop":
                raise ValueError(f"{path}: region-gated model must use crop input")
            parent_classes = _array(step["parentClass"], f"{path}.parentClass")
            if not parent_classes:
                raise ValueError(f"{path}.parentClass must be nonempty")
            names = [
                require_path_class_name(item, f"{path}.parentClass")
                for item in parent_classes
            ]
            if len(set(names)) != len(names):
                raise ValueError(f"{path}.parentClass contains duplicates")
        elif fields == {"script"}:
            filename = _safe_relative_file(step["script"], f"{path}.script", ".groovy")
            model_id = filename
            if filename == METRICS_SCRIPT_NAME:
                if metrics_seen or index != len(steps) - 1:
                    raise ValueError(
                        "metrics.groovy must occur once and be the final step"
                    )
                metrics_seen = True
        else:
            raise ValueError(f"{path} is not one of the supported step shapes")

        if model_id in seen_ids:
            raise ValueError(f"duplicate stable step id: {model_id!r}")
        seen_ids.add(model_id)


def _known_model(value: Any, models: dict[str, Any], path: str) -> str:
    model_id = _nonblank(value, path)
    if model_id not in models:
        raise ValueError(f"{path} names unknown model {model_id!r}")
    return model_id


def _validate_upstream(value: Any) -> None:
    upstream = _nonblank(value, "recipe.upstream")
    if upstream.count("@") != 1:
        raise ValueError("recipe.upstream must be owner/name@revision")
    ref, revision = upstream.split("@", 1)
    _module_ref(ref, "recipe.upstream")
    _commit_sha(revision, "recipe.upstream revision")


def _module_ref(value: Any, path: str) -> str:
    ref = _nonblank(value, path)
    try:
        ModuleRef.parse(ref)
    except ValueError as exc:
        raise ValueError(f"{path} must be a canonical owner/name module ref") from exc
    return ref


def _commit_sha(value: Any, path: str) -> str:
    return validate_commit_sha(value, path)


def _safe_relative_file(value: Any, path: str, suffix: str) -> str:
    filename = _nonblank(value, path)
    parts = filename.split("/")
    if (
        filename.startswith("/")
        or "\\" in filename
        or "\x00" in filename
        or any(part in {"", ".", ".."} for part in parts)
        or not filename.endswith(suffix)
    ):
        raise ValueError(f"{path} must be a safe POSIX-relative {suffix} path")
    return filename


def _script_bytes(name: Any, source: str | bytes) -> bytes:
    _safe_relative_file(name, "script filename", ".groovy")
    if isinstance(source, str):
        try:
            return source.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"script {name!r} source is not valid Unicode") from exc
    if isinstance(source, bytes):
        source.decode("utf-8")
        return source
    raise ValueError(f"script {name!r} source must be str or bytes")


def _fields(
    value: dict[str, Any],
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a JSON array")
    return value


def _nonblank(value: Any, path: str) -> str:
    if not isinstance(value, str) or is_blank(value):
        raise ValueError(f"{path} must be a nonblank string")
    return value


def _integer(value: Any, path: str) -> int:
    return require_int32(value, path)


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{path} must be finite and positive")
    return number


def _positive_integer(value: Any, path: str) -> int:
    number = _integer(value, path)
    if number <= 0:
        raise ValueError(f"{path} must be positive")
    return number


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


def _overlap(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    overlap = float(value)
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError(f"{path} must be finite and in [0, 1)")
    return overlap


def _profile_id(value: Any, path: str) -> str:
    identifier = _nonblank(value, path)
    if not _PROFILE_ID.fullmatch(identifier):
        raise ValueError(f"{path} must be 1 to 128 visible ASCII characters")
    return identifier
