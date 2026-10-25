"""Vendor the shared HistoForge Models runtime with unchanged paths and bytes.

The source owns all runtime algorithms. Standalone research orchestration under
pathology/training and the upstream GUI/model loader are intentionally excluded.
Only the root namespace __init__.py is backend-owned: importing it must not load
the standalone package's version/torch dependency at gateway startup.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPOSITORY = "https://github.com/andreibeliaev/histoforge-models.git"
SOURCE_ROOT = PurePosixPath("histoforge_models")
DESTINATION_ROOT = PurePosixPath("histoforge/_vendor/histoforge_models")
ROOT_PRIMITIVES = {"core.py", "dynamics.py", "transforms.py", "utils.py", "metrics.py"}
BACKEND_OWNED_FILES = {"__init__.py"}
MANIFEST_PATH = DESTINATION_ROOT / "provenance.json"
LICENSE_SOURCE_PATH = PurePosixPath("LICENSE")
LICENSE_DESTINATION_PATH = PurePosixPath("histoforge/_vendor/histoforge_models/LICENSE")
LICENSE_DECLARATION_SOURCE_PATH = PurePosixPath("setup.py")
NOTICE_SOURCE_PATH = PurePosixPath("NOTICE")
NOTICE_DESTINATION_PATH = PurePosixPath("histoforge/_vendor/histoforge_models/NOTICE")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _git(checkout: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _source_snapshot(
    checkout: Path, commit: str
) -> tuple[str, dict[str, bytes], str, bytes, bytes]:
    checkout = checkout.resolve(strict=True)
    resolved_commit = (
        _git(checkout, "rev-parse", f"{commit}^{{commit}}").decode().strip()
    )
    paths = (
        _git(
            checkout,
            "ls-tree",
            "-r",
            "--name-only",
            resolved_commit,
            "--",
            SOURCE_ROOT.as_posix(),
        )
        .decode("utf-8")
        .splitlines()
    )
    expected_prefix = f"{SOURCE_ROOT.as_posix()}/"
    files: dict[str, bytes] = {}
    for source_path in paths:
        relative_path = source_path.removeprefix(expected_prefix)
        if not _is_shared_path(relative_path):
            continue
        _require_relative_path(relative_path)
        files[relative_path] = _git(
            checkout, "show", f"{resolved_commit}:{source_path}"
        )
    if not ROOT_PRIMITIVES <= files.keys() or "pathology/__init__.py" not in files:
        raise ValueError(
            "source commit is missing the shared HistoForge Models runtime"
        )
    _verify_import_closure(files)
    license_bytes = _git(
        checkout, "show", f"{resolved_commit}:{LICENSE_SOURCE_PATH.as_posix()}"
    )
    notice_bytes = _git(
        checkout, "show", f"{resolved_commit}:{NOTICE_SOURCE_PATH.as_posix()}"
    )
    setup_bytes = _git(
        checkout,
        "show",
        f"{resolved_commit}:{LICENSE_DECLARATION_SOURCE_PATH.as_posix()}",
    )
    license_expression = _declared_license_expression(setup_bytes)
    return resolved_commit, files, license_expression, license_bytes, notice_bytes


def _is_shared_path(value: str) -> bool:
    path = PurePosixPath(value)
    return value in ROOT_PRIMITIVES or (
        path.suffix == ".py"
        and (
            path.parent == PurePosixPath("pathology")
            or path.parent == PurePosixPath("pathology/managed_training")
        )
    )


def _verify_import_closure(files: dict[str, bytes]) -> None:
    """Reject imports of omitted standalone modules without importing ML code."""
    modules = {tuple(PurePosixPath(path).with_suffix("").parts) for path in files}
    packages = {module[:-1] for module in modules if module[-1] == "__init__"}
    packages.add(())  # backend-owned namespace initializer
    for path, content in files.items():
        package = PurePosixPath(path).parent.parts
        for node in ast.walk(ast.parse(content, filename=path)):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level > len(package) + 1:
                    raise ValueError(f"relative import escapes vendor root: {path}")
                base = package[: len(package) + 1 - node.level]
                targets = (
                    [base + tuple(node.module.split("."))]
                    if node.module
                    else [base + (alias.name,) for alias in node.names]
                )
                if any(
                    target not in modules and target not in packages
                    for target in targets
                ):
                    raise ValueError(
                        f"import leaves shared runtime: {path}:{node.lineno}"
                    )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                if any(
                    name.partition(".")[0] in {SOURCE_ROOT.name, "cellpose"}
                    for name in names
                ):
                    raise ValueError(
                        f"absolute scientific package import cannot be vendored: {path}"
                    )


def _declared_license_expression(setup_bytes: bytes) -> str:
    """Read the exact license expression declared by the source package."""

    tree = ast.parse(setup_bytes.decode("utf-8"), filename="setup.py")
    declarations = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
        for keyword in node.keywords
        if keyword.arg == "license"
        and isinstance(keyword.value, ast.Constant)
        and isinstance(keyword.value.value, str)
    ]
    if len(declarations) != 1 or not declarations[0].strip():
        raise ValueError("source setup.py must declare one literal license expression")
    return declarations[0]


def _require_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or path.as_posix() != value:
        raise ValueError(f"noncanonical vendored path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe vendored path: {value!r}")
    return value


def _manifest(
    commit: str,
    files: dict[str, bytes],
    license_expression: str,
    license_bytes: bytes,
    notice_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": commit,
        "source_root": SOURCE_ROOT.as_posix(),
        "destination_root": DESTINATION_ROOT.as_posix(),
        "files": [
            {"path": path, "sha256": _sha256(files[path])}
            for path in sorted(files, key=lambda item: item.encode("utf-8"))
        ],
        "license": {
            "declared_expression": license_expression,
            "declaration_source_path": LICENSE_DECLARATION_SOURCE_PATH.as_posix(),
            "source_path": LICENSE_SOURCE_PATH.as_posix(),
            "destination_path": LICENSE_DESTINATION_PATH.as_posix(),
            "sha256": _sha256(license_bytes),
        },
        "notice": {
            "source_path": NOTICE_SOURCE_PATH.as_posix(),
            "destination_path": NOTICE_DESTINATION_PATH.as_posix(),
            "sha256": _sha256(notice_bytes),
        },
    }


def verify_vendored_snapshot(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    source_checkout: Path | None = None,
) -> dict[str, Any]:
    """Verify the checked-in snapshot and optionally its exact source commit."""

    manifest_path = repository_root / MANIFEST_PATH
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest_bytes != _canonical_json(manifest):
        raise ValueError("managed training provenance manifest is not canonical JSON")
    if set(manifest) != {
        "schema_version",
        "source_repository",
        "source_commit",
        "source_root",
        "destination_root",
        "files",
        "license",
        "notice",
    }:
        raise ValueError("managed training provenance manifest has unexpected fields")
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported managed training provenance schema")
    if manifest["source_repository"] != SOURCE_REPOSITORY:
        raise ValueError("managed training source repository differs")
    if manifest["source_root"] != SOURCE_ROOT.as_posix():
        raise ValueError("managed training source root differs")
    if manifest["destination_root"] != DESTINATION_ROOT.as_posix():
        raise ValueError("managed training destination root differs")
    commit = manifest["source_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("managed training source commit must be a full SHA-1")

    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("managed training provenance must list source files")
    paths: list[str] = []
    destination = repository_root / DESTINATION_ROOT
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("managed training file entry has unexpected fields")
        path = _require_relative_path(entry["path"])
        if not _is_shared_path(path):
            raise ValueError(
                "manifest path is outside the shared HistoForge Models runtime"
            )
        content = (destination / path).read_bytes()
        if _sha256(content) != entry["sha256"]:
            raise ValueError(f"vendored managed training file differs: {path}")
        paths.append(path)
    expected_order = sorted(paths, key=lambda item: item.encode("utf-8"))
    if paths != expected_order or len(paths) != len(set(paths)):
        raise ValueError("managed training file paths must be unique and sorted")
    actual_paths = sorted(
        (
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*.py")
            if path.relative_to(destination).as_posix() not in BACKEND_OWNED_FILES
        ),
        key=lambda item: item.encode("utf-8"),
    )
    if actual_paths != paths:
        raise ValueError("vendored managed training file closure differs")
    _verify_import_closure({path: (destination / path).read_bytes() for path in paths})

    license_record = manifest["license"]
    if not isinstance(license_record, dict) or set(license_record) != {
        "declared_expression",
        "declaration_source_path",
        "source_path",
        "destination_path",
        "sha256",
    }:
        raise ValueError("managed training license record has unexpected fields")
    if (
        not isinstance(license_record["declared_expression"], str)
        or not license_record["declared_expression"].strip()
        or license_record["declaration_source_path"]
        != LICENSE_DECLARATION_SOURCE_PATH.as_posix()
        or license_record["source_path"] != LICENSE_SOURCE_PATH.as_posix()
        or license_record["destination_path"] != LICENSE_DESTINATION_PATH.as_posix()
    ):
        raise ValueError("managed training license identity differs")
    license_bytes = (repository_root / LICENSE_DESTINATION_PATH).read_bytes()
    if _sha256(license_bytes) != license_record["sha256"]:
        raise ValueError("vendored managed training license differs")

    notice_record = manifest["notice"]
    if not isinstance(notice_record, dict) or set(notice_record) != {
        "source_path",
        "destination_path",
        "sha256",
    }:
        raise ValueError("managed training notice record has unexpected fields")
    if notice_record != {
        "source_path": NOTICE_SOURCE_PATH.as_posix(),
        "destination_path": NOTICE_DESTINATION_PATH.as_posix(),
        "sha256": notice_record["sha256"],
    }:
        raise ValueError("managed training notice identity differs")
    notice_bytes = (repository_root / NOTICE_DESTINATION_PATH).read_bytes()
    if _sha256(notice_bytes) != notice_record["sha256"]:
        raise ValueError("vendored managed training notice differs")

    if source_checkout is not None:
        (
            resolved,
            source_files,
            source_license_expression,
            source_license,
            source_notice,
        ) = _source_snapshot(source_checkout, commit)
        if manifest != _manifest(
            resolved,
            source_files,
            source_license_expression,
            source_license,
            source_notice,
        ):
            raise ValueError(
                "vendored snapshot differs from the recorded source commit"
            )
        for path, content in source_files.items():
            if (destination / path).read_bytes() != content:
                raise ValueError(f"vendored source bytes differ: {path}")
    return manifest


def synchronize(source_checkout: Path, source_commit: str) -> None:
    """Replace the vendored package with bytes from one exact Git commit."""

    commit, files, license_expression, license_bytes, notice_bytes = _source_snapshot(
        source_checkout, source_commit
    )
    destination = REPOSITORY_ROOT / DESTINATION_ROOT
    destination.mkdir(parents=True, exist_ok=True)
    expected_paths = set(files) | BACKEND_OWNED_FILES
    for existing in destination.rglob("*.py"):
        if existing.relative_to(destination).as_posix() not in expected_paths:
            existing.unlink()
    for path, content in files.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (REPOSITORY_ROOT / LICENSE_DESTINATION_PATH).write_bytes(license_bytes)
    (REPOSITORY_ROOT / NOTICE_DESTINATION_PATH).write_bytes(notice_bytes)
    (REPOSITORY_ROOT / MANIFEST_PATH).write_bytes(
        _canonical_json(
            _manifest(
                commit,
                files,
                license_expression,
                license_bytes,
                notice_bytes,
            )
        )
    )
    verify_vendored_snapshot(
        repository_root=REPOSITORY_ROOT, source_checkout=source_checkout
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the checked-in snapshot without rewriting it",
    )
    args = parser.parse_args()
    if args.check:
        if args.source_commit is not None:
            parser.error("--source-commit is used only when synchronizing")
        verify_vendored_snapshot(source_checkout=args.source_checkout)
        return
    if args.source_checkout is None or args.source_commit is None:
        parser.error("synchronizing requires --source-checkout and --source-commit")
    synchronize(args.source_checkout, args.source_commit)


if __name__ == "__main__":
    main()
