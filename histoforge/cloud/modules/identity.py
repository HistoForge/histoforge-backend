"""Exact identity and validation for one installed module directory.

The author-owned module contract remains :mod:`histoforge.cloud.modules.schema`.
This module owns
the backend-private ``StoredModuleRecord`` persisted beside one current module.
It hashes all retained files and schema-owned artifact roles; it performs no
Hugging Face IO, Volume publication, catalog projection, or runner dispatch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    parse_json,
    require_int32,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_rfc3339_millis,
    require_sha256,
)
from histoforge.cloud.inference.identity import validate_module_content_identity
from histoforge.cloud.modules.files import (
    MODULE_PRIVATE_DIRECTORY,
    file_identity,
    is_private_module_path,
    module_file_paths,
    normalize_relative_path,
    read_regular_file,
    require_file_closure,
    require_unique_casefold_paths,
    stream_file_identity,
)
from histoforge.cloud.modules.readme import read_module_readme, validate_readme_markdown
from histoforge.cloud.modules.schema import (
    TRAINING_REPORT_NAME,
    executable_module_paths,
    model_artifact_declarations,
    module_content_paths,
    parse_module_definition,
    parse_module_recipe,
    validate_commit_sha,
    validate_model_id,
)
from histoforge.cloud.modules.training_evidence import (
    validate_module_training_report,
    validate_training_result_attestation,
)
from histoforge.module_ref import ModuleRef

STORED_MODULE_RECORD_SCHEMA_VERSION = 1
STORED_MODULE_RECORD_PATH = f"{MODULE_PRIVATE_DIRECTORY}/record.json"

_RECORD_REQUIRED_FIELDS = {
    "schema_version",
    "module_ref",
    "resolved_commit_sha",
    "recipe_digest",
    "content_digest",
    "executable_files",
    "resolved_artifacts",
    "installed_at",
}
_RECORD_OPTIONAL_FIELDS = {
    "training_result_attestation",
    "readme_markdown",
    "additional_files",
}
_FILE_FIELDS = {"relative_path", "size_bytes", "sha256"}
_ARTIFACT_FIELDS = {
    "model_id",
    "role",
    "artifact_source_type",
    "artifact_source_ref",
    "resolved_revision",
    "size_bytes",
    "sha256",
}
_ARTIFACT_ROLES = {"weights", "backbone.weights"}
_ARTIFACT_SOURCE_TYPES = {"vendored", "referenced", "local"}


def build_stored_module_record(
    module_dir: Path,
    *,
    module_ref: str,
    resolved_commit_sha: str | None,
    installed_at: str,
    external_artifacts: Mapping[tuple[str, str], Path] | None = None,
    training_api_limits: Any | None = None,
    training_result_attestation: Any | None = None,
) -> dict[str, Any]:
    """Build the unique record for exact validated module and artifact bytes."""

    return _build_stored_module_record(
        module_dir,
        module_ref=module_ref,
        resolved_commit_sha=resolved_commit_sha,
        installed_at=installed_at,
        external_artifacts=external_artifacts,
        training_api_limits=training_api_limits,
        training_result_attestation=training_result_attestation,
        stored_record_present=False,
    )


def _build_stored_module_record(
    module_dir: Path,
    *,
    module_ref: str,
    resolved_commit_sha: str | None,
    installed_at: str,
    external_artifacts: Mapping[tuple[str, str], Path] | None,
    training_api_limits: Any | None,
    training_result_attestation: Any | None,
    stored_record_present: bool,
) -> dict[str, Any]:
    """Implementation shared by first construction and installed revalidation."""

    canonical_ref = str(ModuleRef.parse(module_ref))
    commit = (
        validate_commit_sha(resolved_commit_sha, "resolved_commit_sha")
        if resolved_commit_sha is not None
        else None
    )
    timestamp = require_rfc3339_millis(installed_at, "installed_at")
    external = dict(external_artifacts or {})

    recipe_bytes = read_regular_file(module_dir, "recipe.json")
    recipe = parse_module_recipe(recipe_bytes)
    executable_paths = executable_module_paths(recipe)
    content_paths = module_content_paths(recipe)
    actual_paths = set(module_file_paths(module_dir))
    private_paths = {path for path in actual_paths if is_private_module_path(path)}
    expected_private = {STORED_MODULE_RECORD_PATH} if stored_record_present else set()
    if private_paths != expected_private:
        raise ValueError("module files contain unexpected backend-private paths")
    if not set(content_paths) <= actual_paths:
        raise ValueError("module file closure is missing recipe-declared files")
    if TRAINING_REPORT_NAME in actual_paths and "trainingReportDigest" not in recipe:
        raise ValueError("training-report.json requires trainingReportDigest")
    additional_files = [
        file_identity(path, module_dir / path)
        for path in sorted(actual_paths - set(content_paths) - private_paths)
    ]

    scripts = {
        step["script"]: read_regular_file(module_dir, step["script"])
        for step in recipe["steps"]
        if "script" in step
    }
    definition = parse_module_definition(recipe_bytes, scripts)

    executable_files = [
        file_identity(path, module_dir / path) for path in executable_paths
    ]
    executable_files.sort(key=lambda identity: identity["relative_path"])

    resolved_artifacts: list[dict[str, Any]] = []
    expected_external: set[tuple[str, str]] = set()
    for declaration in model_artifact_declarations(recipe):
        model_id = declaration["model_id"]
        role = declaration["role"]
        source = declaration["source"]
        source_type = source["type"]
        if source_type == "vendored":
            artifact_path = module_dir / source["file"]
        else:
            key = (model_id, role)
            expected_external.add(key)
            try:
                artifact_path = external[key]
            except KeyError as exc:
                raise ValueError(
                    f"missing resolved artifact bytes for {model_id}.{role}"
                ) from exc

        artifact_size, artifact_digest = stream_file_identity(
            artifact_path,
            f"{model_id}.{role}",
        )
        resolved_artifacts.append(
            {
                "model_id": model_id,
                "role": role,
                "artifact_source_type": source_type,
                "artifact_source_ref": source.get("ref"),
                "resolved_revision": source.get("revision"),
                "size_bytes": artifact_size,
                "sha256": artifact_digest,
            }
        )

    if set(external) != expected_external:
        extra = sorted(set(external) - expected_external)
        raise ValueError(f"unexpected resolved artifact inputs: {extra}")

    resolved_artifacts.sort(
        key=lambda identity: (
            identity["model_id"],
            identity["role"],
            identity["sha256"],
        )
    )
    content_digest = _content_digest(
        definition["recipe_digest"],
        executable_files,
        resolved_artifacts,
        additional_files,
    )
    report = None
    if "trainingReportDigest" in recipe:
        if training_api_limits is None:
            raise ValueError(
                "training_api_limits are required for a module training report"
            )
        report = validate_module_training_report(
            module_dir,
            recipe,
            resolved_artifacts,
            training_api_limits,
        )
    record = {
        "schema_version": STORED_MODULE_RECORD_SCHEMA_VERSION,
        "module_ref": canonical_ref,
        "resolved_commit_sha": commit,
        "recipe_digest": definition["recipe_digest"],
        "content_digest": content_digest,
        "executable_files": executable_files,
        "resolved_artifacts": resolved_artifacts,
        "installed_at": timestamp,
    }
    if commit is not None:
        readme = read_module_readme(module_dir)
        if readme is not None:
            record["readme_markdown"] = readme
    if additional_files:
        record["additional_files"] = additional_files
    if training_result_attestation is not None:
        attestation = validate_training_result_attestation(training_result_attestation)
        if report is None:
            raise ValueError(
                "training result attestation requires a module training report"
            )
        if attestation["run_id"] != report["run_id"]:
            raise ValueError("training result attestation run_id differs from report")
        if attestation["report_digest"] != recipe["trainingReportDigest"]:
            raise ValueError(
                "training result attestation report_digest differs from recipe"
            )
        record["training_result_attestation"] = attestation
    return validate_stored_module_record(record)


def validate_stored_module_record(value: Any) -> dict[str, Any]:
    """Validate the canonical record, retaining supporting-file byte identities.

    Current installations bind non-recipe files through ``additional_files``.
    Reading their metadata must preserve those entries and their content digest;
    this does not execute the files or change installation/publication behavior.
    """

    record = _object_with_optional(
        value,
        _RECORD_REQUIRED_FIELDS,
        _RECORD_OPTIONAL_FIELDS,
        "stored module record",
    )
    if require_int32(record["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported stored module record schema_version")
    module_ref = _module_ref(record["module_ref"])
    commit_value = record["resolved_commit_sha"]
    if commit_value is not None:
        commit_value = validate_commit_sha(commit_value, "resolved_commit_sha")

    files_value = record["executable_files"]
    if not isinstance(files_value, list) or not files_value:
        raise ValueError("executable_files must be a nonempty array")
    files = [_validate_file_identity(item) for item in files_value]
    if files != sorted(files, key=lambda item: item["relative_path"]):
        raise ValueError("executable_files must be sorted by relative_path")
    paths = [item["relative_path"] for item in files]
    require_unique_casefold_paths(paths, "executable_files")
    if "recipe.json" not in paths:
        raise ValueError("executable_files must contain recipe.json")

    artifacts_value = record["resolved_artifacts"]
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise ValueError("resolved_artifacts must be a nonempty array")
    artifacts = [_validate_artifact_identity(item) for item in artifacts_value]
    expected_artifact_order = sorted(
        artifacts,
        key=lambda item: (item["model_id"], item["role"], item["sha256"]),
    )
    if artifacts != expected_artifact_order:
        raise ValueError("resolved_artifacts are not in canonical order")
    keys = [(item["model_id"], item["role"]) for item in artifacts]
    if len(keys) != len(set(keys)):
        raise ValueError("resolved_artifacts contains a duplicate model role")

    recipe_digest = require_sha256(record["recipe_digest"], "recipe_digest")
    content_digest = require_sha256(record["content_digest"], "content_digest")
    additional_value = record.get("additional_files", [])
    if not isinstance(additional_value, list):
        raise ValueError("additional_files must be an array")
    additional_files = [_validate_file_identity(item) for item in additional_value]
    if additional_files != sorted(
        additional_files, key=lambda item: item["relative_path"]
    ):
        raise ValueError("additional_files must be sorted by relative_path")
    all_paths = paths + [item["relative_path"] for item in additional_files]
    require_unique_casefold_paths(all_paths, "module files")
    if any(is_private_module_path(path) for path in all_paths):
        raise ValueError("module files must not include backend-private paths")
    if any(item["relative_path"] == TRAINING_REPORT_NAME for item in additional_files):
        raise ValueError("training-report.json is schema-owned, not an additional file")
    expected_content_digest = _content_digest(
        recipe_digest, files, artifacts, additional_files
    )
    if content_digest != expected_content_digest:
        raise ValueError("content_digest does not match stored module fields")

    result = {
        "schema_version": STORED_MODULE_RECORD_SCHEMA_VERSION,
        "module_ref": module_ref,
        "resolved_commit_sha": commit_value,
        "recipe_digest": recipe_digest,
        "content_digest": content_digest,
        "executable_files": files,
        "resolved_artifacts": artifacts,
        "installed_at": require_rfc3339_millis(record["installed_at"], "installed_at"),
    }
    if "readme_markdown" in record:
        if commit_value is None:
            raise ValueError("saved README requires a resolved HF commit")
        result["readme_markdown"] = validate_readme_markdown(record["readme_markdown"])
    if additional_files:
        result["additional_files"] = additional_files
    if "training_result_attestation" in record:
        attestation = validate_training_result_attestation(
            record["training_result_attestation"]
        )
        expected_ref = f"workspace/run-{attestation['run_id']}"
        if module_ref != expected_ref:
            raise ValueError("training result attestation does not match module_ref")
        if commit_value is not None:
            raise ValueError("training result attestation requires a commitless module")
        if attestation["module_content_digest"] != content_digest:
            raise ValueError(
                "training result attestation does not match content_digest"
            )
        result["training_result_attestation"] = attestation
    return canonical_tree(result)


def _content_digest(recipe_digest, files, artifacts, additional_files) -> str:
    preimage = {
        "recipe_digest": recipe_digest,
        "executable_files": files,
        "resolved_artifacts": artifacts,
    }
    if additional_files:
        preimage["additional_files"] = additional_files
    return canonical_digest(preimage)


def write_stored_module_record(module_dir: Path, record: Any) -> None:
    """Write one canonical record after all retained files have been closed."""

    normalized = validate_stored_module_record(record)
    path = module_dir / STORED_MODULE_RECORD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_bytes(normalized))
        stream.flush()
        os.fsync(stream.fileno())


def validate_module_directory(
    module_dir: Path,
    *,
    expected_module_ref: str | None = None,
    external_artifacts: Mapping[tuple[str, str], Path] | None = None,
    training_api_limits: Any | None = None,
) -> dict[str, Any]:
    """Recompute every identity from disk and require the stored record to match."""

    record_bytes = read_regular_file(module_dir, STORED_MODULE_RECORD_PATH)
    record = validate_stored_module_record(parse_json(record_bytes))
    if canonical_bytes(record) != record_bytes:
        raise ValueError("stored module record must use canonical JSON bytes")
    if expected_module_ref is not None:
        expected = str(ModuleRef.parse(expected_module_ref))
        if record["module_ref"] != expected:
            raise ValueError("stored module record module_ref does not match path")

    expected = _build_stored_module_record(
        module_dir,
        module_ref=record["module_ref"],
        resolved_commit_sha=record["resolved_commit_sha"],
        installed_at=record["installed_at"],
        external_artifacts=external_artifacts,
        training_api_limits=training_api_limits,
        training_result_attestation=record.get("training_result_attestation"),
        stored_record_present=True,
    )
    if expected != record:
        raise ValueError("stored module record does not match installed bytes")

    recipe = parse_module_recipe(read_regular_file(module_dir, "recipe.json"))
    require_file_closure(
        module_dir,
        (
            *module_content_paths(recipe),
            *(item["relative_path"] for item in record.get("additional_files", [])),
            STORED_MODULE_RECORD_PATH,
        ),
    )
    return record


def module_content_identity(record: Any) -> dict[str, Any]:
    """Project the shared immutable identity from a stored module record."""

    normalized = validate_stored_module_record(record)
    identity = {
        "module_ref": normalized["module_ref"],
        "recipe_digest": normalized["recipe_digest"],
        "content_digest": normalized["content_digest"],
    }
    if normalized["resolved_commit_sha"] is not None:
        identity["resolved_commit_sha"] = normalized["resolved_commit_sha"]
    return validate_module_content_identity(identity)


def module_content_file_identities(
    module_dir: Path,
    recipe: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return exact identities for all retained files, excluding private metadata."""

    executable = {
        identity["relative_path"]: identity for identity in record["executable_files"]
    }
    return sorted(
        [
            dict(executable.get(path) or file_identity(path, module_dir / path))
            for path in module_content_paths(recipe)
        ]
        + [dict(item) for item in record.get("additional_files", [])],
        key=lambda item: item["relative_path"],
    )


def read_stored_module_record(module_dir: Path) -> dict[str, Any]:
    """Read and validate the canonical record bytes without rehashing module files."""

    encoded = read_regular_file(module_dir, STORED_MODULE_RECORD_PATH)
    record = validate_stored_module_record(parse_json(encoded))
    if encoded != canonical_bytes(record):
        raise ValueError("stored module record must use canonical JSON bytes")
    return record


def _validate_file_identity(value: Any) -> dict[str, Any]:
    identity = _exact_object(value, _FILE_FIELDS, "module file identity")
    return {
        "relative_path": normalize_relative_path(identity["relative_path"]),
        "size_bytes": require_nonnegative_safe_integer(
            identity["size_bytes"], "size_bytes"
        ),
        "sha256": require_sha256(identity["sha256"], "sha256"),
    }


def _validate_artifact_identity(value: Any) -> dict[str, Any]:
    identity = _exact_object(value, _ARTIFACT_FIELDS, "resolved artifact identity")
    model_id = validate_model_id(identity["model_id"], "model_id")
    role = require_nonblank_string(identity["role"], "role")
    if role not in _ARTIFACT_ROLES:
        raise ValueError(f"unsupported schema-owned artifact role: {role!r}")
    source_type = require_nonblank_string(
        identity["artifact_source_type"], "artifact_source_type"
    )
    if source_type not in _ARTIFACT_SOURCE_TYPES:
        raise ValueError(f"unsupported artifact_source_type: {source_type!r}")

    source_ref = identity["artifact_source_ref"]
    revision = identity["resolved_revision"]
    if source_type == "vendored":
        if source_ref is not None or revision is not None:
            raise ValueError("vendored artifact cannot name a source ref or revision")
    elif source_type == "referenced":
        source_ref = _module_ref(source_ref)
        revision = validate_commit_sha(revision, "resolved_revision")
    else:
        source_ref = _module_ref(source_ref)
        if not source_ref.startswith("local/"):
            raise ValueError("local artifact source ref must use local/ namespace")
        if revision is not None:
            raise ValueError("local artifact cannot name a resolved revision")

    return {
        "model_id": model_id,
        "role": role,
        "artifact_source_type": source_type,
        "artifact_source_ref": source_ref,
        "resolved_revision": revision,
        "size_bytes": require_nonnegative_safe_integer(
            identity["size_bytes"], "size_bytes"
        ),
        "sha256": require_sha256(identity["sha256"], "sha256"),
    }


def _module_ref(value: Any) -> str:
    text = require_nonblank_string(value, "module_ref")
    try:
        return str(ModuleRef.parse(text))
    except ValueError as exc:
        raise ValueError("module_ref must be a canonical owner/name ref") from exc


def _exact_object(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{label} fields mismatch: "
            f"missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return value


def _object_with_optional(
    value: Any,
    required: set[str],
    optional: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if not required <= actual or not actual <= required | optional:
        raise ValueError(
            f"{label} fields mismatch: "
            f"missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - required - optional)}"
        )
    return value
