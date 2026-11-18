"""Bounded discovery of training artifact directories on the Artifact Volume."""

from __future__ import annotations

import os
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    require_canonical_uuid,
    require_positive_int32,
)
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt


def discover_training_artifact_ids(
    roots: Iterable[Path],
    max_items: Any,
    *,
    identity_field: str,
    ignored_names: Collection[str] = (),
) -> list[str]:
    """Return canonical artifact directories without scanning past the bound."""

    maximum = require_positive_int32(max_items, "max_items")
    ignored = frozenset(ignored_names)
    found: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ArtifactBytesCorrupt("training artifact namespace is invalid")
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name in ignored:
                    continue
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise ArtifactBytesCorrupt("training artifact root is invalid")
                try:
                    found.add(require_canonical_uuid(entry.name, identity_field))
                except ValueError as error:
                    raise ArtifactBytesCorrupt(
                        "training artifact root has an invalid identity"
                    ) from error
                if len(found) == maximum:
                    return sorted(found)
    return sorted(found)
