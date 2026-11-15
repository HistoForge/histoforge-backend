"""Canonical build manifests and compact executable identities.

The manifest keeps behavior-relevant build evidence out of ordinary requests.
Each field is a digest of one independently inspectable input group; the
manifest's own JCS digest is the value carried by ``SoftwareBuildIdentity``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_digest,
    canonical_tree,
    require_int32,
    require_nonblank_string,
    require_sha256,
    sha256_bytes,
)

_BUILD_MANIFEST_FIELDS = {
    "schema_version",
    "component",
    "source_digest",
    "dependency_lock_digest",
    "resolved_environment_digest",
    "build_configuration_digest",
    "behavior_policy_digest",
}
_SOFTWARE_BUILD_IDENTITY_FIELDS = {
    "version",
    "artifact_digest",
    "manifest_digest",
}


def file_set_manifest(files: Iterable[tuple[str, bytes]]) -> dict[str, Any]:
    """Describe exact named bytes in deterministic path order.

    Paths are logical repository- or artifact-relative names, not host paths.
    Sorting their UTF-8 bytes makes ordering independent of Python, Java, and
    the build host's filesystem.
    """

    entries: list[tuple[bytes, dict[str, str]]] = []
    seen: set[str] = set()
    for path, content in files:
        canonical_path = _require_logical_path(path)
        if canonical_path in seen:
            raise ValueError(f"file set contains duplicate path: {canonical_path!r}")
        if not isinstance(content, bytes):
            raise ValueError(f"file {canonical_path!r} content must be bytes")
        seen.add(canonical_path)
        entries.append(
            (
                canonical_path.encode("utf-8"),
                {"path": canonical_path, "sha256": sha256_bytes(content)},
            )
        )
    entries.sort(key=lambda item: item[0])
    return {"files": [entry for _, entry in entries]}


def file_set_digest(files: Iterable[tuple[str, bytes]]) -> str:
    """Return the JCS digest of :func:`file_set_manifest`."""

    return canonical_digest(file_set_manifest(files))


def build_manifest(
    *,
    component: str,
    source_digest: str,
    dependency_lock_digest: str,
    resolved_environment_digest: str,
    build_configuration_digest: str,
    behavior_policy_digest: str,
) -> dict[str, Any]:
    """Create one closed schema-1 behavior manifest.

    ``component`` identifies the process or behavior surface. The remaining
    fields bind, respectively, executable source, dependency resolution,
    observed resolved packages/runtime, image or JAR build configuration, and
    explicit behavior policies or calibrated fixtures.
    """

    return parse_build_manifest(
        {
            "schema_version": 1,
            "component": component,
            "source_digest": source_digest,
            "dependency_lock_digest": dependency_lock_digest,
            "resolved_environment_digest": resolved_environment_digest,
            "build_configuration_digest": build_configuration_digest,
            "behavior_policy_digest": behavior_policy_digest,
        }
    )


def parse_build_manifest(value: Any) -> dict[str, Any]:
    """Validate and return a detached canonical schema-1 build manifest."""

    manifest = _require_exact_object(value, _BUILD_MANIFEST_FIELDS, "build manifest")
    if require_int32(manifest["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported build manifest schema_version")
    result = {
        "schema_version": 1,
        "component": require_nonblank_string(manifest["component"], "component"),
        "source_digest": require_sha256(manifest["source_digest"], "source_digest"),
        "dependency_lock_digest": require_sha256(
            manifest["dependency_lock_digest"], "dependency_lock_digest"
        ),
        "resolved_environment_digest": require_sha256(
            manifest["resolved_environment_digest"], "resolved_environment_digest"
        ),
        "build_configuration_digest": require_sha256(
            manifest["build_configuration_digest"], "build_configuration_digest"
        ),
        "behavior_policy_digest": require_sha256(
            manifest["behavior_policy_digest"], "behavior_policy_digest"
        ),
    }
    return canonical_tree(result)


def build_manifest_digest(value: Any) -> str:
    """Return the canonical identity of a validated build manifest."""

    return canonical_digest(parse_build_manifest(value))


def software_build_identity(
    *, version: str, artifact_digest: str, manifest: Any
) -> dict[str, str]:
    """Create the compact identity carried by requests and provenance."""

    return parse_software_build_identity(
        {
            "version": version,
            "artifact_digest": artifact_digest,
            "manifest_digest": build_manifest_digest(manifest),
        }
    )


def parse_software_build_identity(value: Any) -> dict[str, str]:
    """Validate and return a detached canonical ``SoftwareBuildIdentity``."""

    identity = _require_exact_object(
        value, _SOFTWARE_BUILD_IDENTITY_FIELDS, "software build identity"
    )
    result = {
        "version": require_nonblank_string(identity["version"], "version"),
        "artifact_digest": require_sha256(
            identity["artifact_digest"], "artifact_digest"
        ),
        "manifest_digest": require_sha256(
            identity["manifest_digest"], "manifest_digest"
        ),
    }
    return canonical_tree(result)


def build_identity_bytes(value: Any) -> bytes:
    """Return canonical bytes for a validated compact build identity."""

    return canonical_bytes(parse_software_build_identity(value))


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


def _require_logical_path(value: Any) -> str:
    path = require_nonblank_string(value, "file path")
    if path.startswith("/") or "\\" in path or "\x00" in path:
        raise ValueError("file path must be a relative POSIX path")
    segments = path.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError("file path must be canonical without empty or dot segments")
    return path
