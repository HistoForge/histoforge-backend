from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
from histoforge.cloud.training.artifact_directories import (
    discover_training_artifact_ids,
)


class _CountingScandir:
    def __init__(self, delegate, visited: list[str]) -> None:
        self._delegate = delegate
        self._visited = visited

    def __enter__(self):
        self._delegate.__enter__()
        return self

    def __exit__(self, *args):
        return self._delegate.__exit__(*args)

    def __iter__(self):
        return self

    def __next__(self):
        entry = next(self._delegate)
        self._visited.append(entry.name)
        return entry


def test_discovery_stops_reading_the_namespace_at_the_requested_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for _ in range(5):
        (tmp_path / str(uuid.uuid4())).mkdir()
    original_scandir = os.scandir
    visited: list[str] = []

    def counting_scandir(path):
        return _CountingScandir(original_scandir(path), visited)

    monkeypatch.setattr(
        "histoforge.cloud.training.artifact_directories.os.scandir",
        counting_scandir,
    )

    assert (
        len(discover_training_artifact_ids((tmp_path,), 2, identity_field="run_id"))
        == 2
    )
    assert len(visited) == 2
