"""Bounded display text from the README retained in a module directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from histoforge.cloud.modules.files import require_regular_path

MAX_README_BYTES = 256 * 1024


def validate_readme_markdown(value: Any) -> str:
    """Validate saved UTF-8 text without interpreting publisher Markdown."""

    if not isinstance(value, str):
        raise ValueError("readme_markdown must be a string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("README.md must be valid UTF-8") from error
    if len(encoded) > MAX_README_BYTES:
        raise ValueError("README.md exceeds the 256 KiB module-card limit")
    return value


def read_module_readme(module_dir: Path) -> str | None:
    """Read saved README bytes; only true absence is optional."""

    path = module_dir / "README.md"
    if not path.exists() and not path.is_symlink():
        return None
    require_regular_path(path, "README.md")
    with path.open("rb") as stream:
        encoded = stream.read(MAX_README_BYTES + 1)
    if len(encoded) > MAX_README_BYTES:
        raise ValueError("README.md exceeds the 256 KiB module-card limit")
    try:
        markdown = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("README.md must be valid UTF-8") from error
    return validate_readme_markdown(markdown)
