"""Recompute compact build identities from the exact deployed Python bundle.

The Modal image contains the package sources and both resolved runtime locks.
Every gateway and runner process hashes those same bytes plus its observed
runtime inventory. No Git checkout, mutable ref, or deploy timestamp enters the
identity.
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from histoforge import __version__
from histoforge.build_identity import (
    build_manifest,
    file_set_digest,
    software_build_identity,
)
from histoforge.canonical_json import canonical_bytes, canonical_tree
from histoforge.cloud.sources.java_runtime import reader_runtime_inventory

_PACKAGE_ROOT = Path(__file__).parents[1]
_RUNTIME_ROOT = Path(__file__).with_name("runtime")
_LOCK_NAMES = ("conda-linux-64.lock", "requirements-modal.lock", "reader-runtime.json")
_VENDORED_SOURCE_EVIDENCE = (
    Path("_vendor/histoforge_models/LICENSE"),
    Path("_vendor/histoforge_models/NOTICE"),
    Path("_vendor/histoforge_models/provenance.json"),
)


@dataclass(frozen=True)
class RuntimeBuild:
    """Compact executable identity together with its reconstructible manifest."""

    identity: dict[str, str]
    manifest: dict[str, Any]


def runtime_build(component: str, behavior_policy: Any) -> RuntimeBuild:
    """Build one component identity from exact source, locks, runtime, and policy."""

    policy_bytes = canonical_bytes(behavior_policy)
    return _runtime_build(component, policy_bytes)


def bundle_artifact_digest() -> str:
    """Identify packaged sources and Modal locks, independent of the local OS."""

    return file_set_digest([*_source_files(), *_lock_files()])


@lru_cache(maxsize=None)
def _runtime_build(component: str, policy_bytes: bytes) -> RuntimeBuild:
    source_files = _source_files()
    lock_files = _lock_files()
    environment_bytes = canonical_bytes(_runtime_inventory())
    app_bytes = (_PACKAGE_ROOT / "cloud" / "_app.py").read_bytes()

    manifest = build_manifest(
        component=component,
        source_digest=file_set_digest(source_files),
        dependency_lock_digest=file_set_digest(lock_files),
        resolved_environment_digest=file_set_digest(
            [("runtime/environment.json", environment_bytes)]
        ),
        build_configuration_digest=file_set_digest(
            [("histoforge/cloud/_app.py", app_bytes), *lock_files]
        ),
        behavior_policy_digest=file_set_digest(
            [(f"policy/{component}.json", policy_bytes)]
        ),
    )
    identity = software_build_identity(
        version=__version__, artifact_digest=bundle_artifact_digest(), manifest=manifest
    )
    return RuntimeBuild(identity=identity, manifest=manifest)


@lru_cache(maxsize=1)
def _source_files() -> tuple[tuple[str, bytes], ...]:
    files = []
    for path in _PACKAGE_ROOT.rglob("*.py"):
        logical = Path("histoforge", path.relative_to(_PACKAGE_ROOT)).as_posix()
        files.append((logical, path.read_bytes()))
    for relative_path in _VENDORED_SOURCE_EVIDENCE:
        path = _PACKAGE_ROOT / relative_path
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise RuntimeError(
                "deployed vendored source evidence is missing: "
                f"{relative_path.as_posix()}"
            ) from error
        logical = Path("histoforge", relative_path).as_posix()
        files.append((logical, content))
    if not files:
        raise RuntimeError("deployed HistoForge source bundle is empty")
    return tuple(sorted(files, key=lambda item: item[0].encode("utf-8")))


@lru_cache(maxsize=1)
def _lock_files() -> tuple[tuple[str, bytes], ...]:
    files = []
    for name in _LOCK_NAMES:
        path = _RUNTIME_ROOT / name
        try:
            content = path.read_bytes()
        except FileNotFoundError as error:
            raise RuntimeError(f"deployed runtime lock is missing: {name}") from error
        files.append((f"histoforge/cloud/runtime/{name}", content))
    return tuple(files)


def _runtime_inventory() -> dict[str, Any]:
    package_names: set[str] = set()
    versions_by_root: dict[tuple[str, Path], set[str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name or not distribution.version:
            raise RuntimeError("installed runtime distribution lacks name or version")
        name = re.sub(r"[-_.]+", "-", raw_name).lower()
        package_names.add(name)
        root = Path(distribution.locate_file("")).resolve()
        versions_by_root.setdefault((name, root), set()).add(distribution.version)
    if not package_names:
        raise RuntimeError("installed runtime distribution inventory is empty")
    packages: dict[str, str] = {}
    for name in sorted(package_names):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"installed runtime distribution cannot be resolved: {name}"
            ) from error
        raw_name = distribution.metadata.get("Name")
        if (
            not raw_name
            or not distribution.version
            or re.sub(r"[-_.]+", "-", raw_name).lower() != name
        ):
            raise RuntimeError(
                f"resolved runtime distribution identity differs from inventory: {name}"
            )
        root = Path(distribution.locate_file("")).resolve()
        versions = versions_by_root[(name, root)]
        if len(versions) != 1:
            raise RuntimeError(
                f"conflicting installed versions of {name} in {root}: "
                + ", ".join(sorted(versions))
            )
        # A later sys.path root may legitimately be shadowed. Multiple versions
        # in the selected root are ambiguous: directory order is not a version
        # selection policy and can change between identical cold containers.
        packages[name] = distribution.version
    libc_name, libc_version = platform.libc_ver()
    return canonical_tree(
        {
            "schema_version": 1,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "machine": platform.machine(),
            "libc": {"name": libc_name, "version": libc_version},
            "packages": packages,
            "original_slide_readers": reader_runtime_inventory(),
        }
    )
