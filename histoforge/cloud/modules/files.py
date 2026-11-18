"""Safe exact-file primitives shared by module identity and storage."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from histoforge.canonical_json import require_nonblank_string, require_sha256

MODULE_PRIVATE_DIRECTORY = ".histoforge"


def is_private_module_path(path: str) -> bool:
    """Identify the reserved backend metadata namespace, never publisher files."""

    return path.split("/", 1)[0].casefold() == MODULE_PRIVATE_DIRECTORY


def normalize_relative_path(value: Any) -> str:
    """Require one normalized POSIX-relative module file path."""

    path = require_nonblank_string(value, "relative_path")
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise ValueError("relative_path must be a normalized POSIX-relative path")
    return path


def validate_module_file_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Validate paths together, including file/directory and case-fold collisions."""

    files = [normalize_relative_path(path) for path in paths]
    directories = {
        parent.as_posix()
        for path in files
        for parent in PurePosixPath(path).parents
        if parent.as_posix() != "."
    }
    require_unique_casefold_paths([*files, *directories], "module file tree")
    return tuple(sorted(files))


def read_regular_file(root: Path, relative_path: str) -> bytes:
    """Read one safe relative path only when it is a regular non-link file."""

    normalized = normalize_relative_path(relative_path)
    path = root.joinpath(*PurePosixPath(normalized).parts)
    return read_regular_path(path, normalized)


def read_regular_path(path: Path, label: str) -> bytes:
    """Read one explicitly addressed regular non-link file."""

    require_regular_path(path, label)
    return path.read_bytes()


def require_regular_path(path: Path, label: str) -> Path:
    """Require one explicitly addressed regular non-link file."""

    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"required file is missing: {label}") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ValueError(f"required path must be a regular non-link file: {label}")
    return path


def file_identity(relative_path: str, path: Path) -> dict[str, Any]:
    """Stream one regular file into its exact path, length, and SHA-256 identity."""

    size_bytes, digest = stream_file_identity(path, relative_path)
    return {
        "relative_path": normalize_relative_path(relative_path),
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def stream_file_identity(path: Path, label: str) -> tuple[int, str]:
    """Return exact length and SHA-256 without materializing the file in memory."""

    require_regular_path(path, label)
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            size_bytes += len(block)
            digest.update(block)
    return size_bytes, digest.hexdigest()


def require_exact_file(
    path: Path,
    identity: Mapping[str, Any],
    label: str,
) -> None:
    """Require one file to equal a recorded length and SHA-256 identity."""

    expected_digest = require_sha256(identity["sha256"], f"{label} digest")
    size_bytes, digest = stream_file_identity(path, label)
    if size_bytes != identity["size_bytes"]:
        raise ValueError(f"{label} size does not match identity")
    if digest != expected_digest:
        raise ValueError(f"{label} digest does not match identity")


def copy_exact_file(
    source: Path,
    target: Path,
    identity: Mapping[str, Any],
) -> None:
    """Copy one recorded file and fail if its bytes changed during the copy."""

    require_regular_path(source, identity["relative_path"])
    digest = hashlib.sha256()
    size_bytes = 0
    with source.open("rb") as reader, target.open("xb") as writer:
        while block := reader.read(1024 * 1024):
            size_bytes += len(block)
            digest.update(block)
            writer.write(block)
        writer.flush()
        os.fsync(writer.fileno())
    if size_bytes != identity["size_bytes"]:
        raise ValueError("copied module file size changed")
    if digest.hexdigest() != identity["sha256"]:
        raise ValueError("copied module file digest changed")


def contained_path(root: Path, candidate: Path, label: str) -> Path:
    """Reject a path that escapes or traverses links under one store root."""

    root = Path(root)
    candidate = Path(candidate)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is not under its store root") from exc
    resolved_root = root.resolve(strict=False)
    try:
        resolved = candidate.resolve(strict=False)
    except RuntimeError as exc:
        raise ValueError(f"{label} contains a symlink loop") from exc
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label} escapes its store root")
    expected = resolved_root.joinpath(*relative.parts)
    if resolved != expected:
        raise ValueError(f"{label} traverses a symlink")
    return candidate


def require_real_directory_outside(
    path: Path,
    root: Path,
    label: str,
) -> Path:
    """Require a real directory whose resolved location is outside a store."""

    path = Path(path)
    if not path.exists() or not path.is_dir() or path.is_symlink():
        raise ValueError(f"{label} must be a real directory")
    resolved_root = Path(root).resolve(strict=False)
    resolved = path.resolve(strict=True)
    if resolved == resolved_root or resolved_root in resolved.parents:
        raise ValueError(f"{label} must be outside the mutable store")
    return path


def require_file_closure(root: Path, expected_paths: tuple[str, ...]) -> None:
    """Require an exact regular-file and necessary-directory tree."""

    expected = {normalize_relative_path(path) for path in expected_paths}
    require_unique_casefold_paths(expected, "module file closure")
    actual = set(module_file_paths(root))
    if actual != expected:
        raise ValueError(
            "module file closure mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def module_file_paths(root: Path) -> tuple[str, ...]:
    """List every regular file after validating the complete module tree."""

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    if not root.exists() or not root.is_dir() or root.is_symlink():
        raise ValueError("module root must be a real directory")

    for directory, names, files in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path.is_symlink():
            raise ValueError("module tree must not contain directory links")
        for name in names:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError(f"module tree contains a link: {relative}")
            if not path.is_dir():
                raise ValueError(f"module tree contains a non-directory: {relative}")
            actual_directories.add(relative)
        for name in files:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if path.is_symlink() or not stat.S_ISREG(mode):
                raise ValueError(f"module tree contains a non-regular file: {relative}")
            actual_files.add(normalize_relative_path(relative))

    expected_directories = {
        parent.as_posix()
        for path in actual_files
        for parent in PurePosixPath(path).parents
        if parent.as_posix() != "."
    }
    if actual_directories != expected_directories:
        raise ValueError(
            "module directory closure mismatch: "
            f"missing={sorted(expected_directories - actual_directories)}, "
            f"extra={sorted(actual_directories - expected_directories)}"
        )
    require_unique_casefold_paths(actual_files | actual_directories, "module tree")
    return tuple(sorted(actual_files))


def require_unique_casefold_paths(paths: Any, label: str) -> None:
    """Reject path sets that collide on a case-insensitive filesystem."""

    values = list(paths)
    folded = [path.casefold() for path in values]
    if len(folded) != len(set(folded)):
        raise ValueError(f"{label} contains a case-fold collision")
