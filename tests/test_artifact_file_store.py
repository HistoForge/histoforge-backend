"""Immutable Artifact-Volume file ownership and HTTP scalar tests."""

import asyncio
import hashlib
import threading
import time

import pytest
from histoforge.cloud.artifacts.store import ArtifactFileStore
from histoforge.cloud.http_input import parse_byte_range
from histoforge.cloud.sources.files import SourceFileStore


def test_generic_artifact_publication_round_trips_exact_bytes(tmp_path):
    events = []
    files = ArtifactFileStore(
        tmp_path,
        reload_volume=lambda: events.append("reload"),
        commit_volume=lambda: events.append("commit"),
    )
    payload = bytes(range(251)) * 5000

    reference = files.publish_artifact_bytes(
        payload,
        media_type="application/octet-stream",
    )

    assert reference["size_bytes"] == len(payload)
    assert reference["sha256"] == hashlib.sha256(payload).hexdigest()
    assert files.load_artifact_ref(reference["artifact_id"]) == reference
    assert (
        files.read_artifact_block(reference["artifact_id"], 17, 4096)
        == payload[17:4113]
    )
    assert events.count("commit") == 1
    assert events.count("reload") == 2


def test_generic_artifact_inventory_and_release_are_idempotent(tmp_path):
    files = ArtifactFileStore(tmp_path)
    first = files.publish_artifact_bytes(b"first", media_type="application/json")
    second = files.publish_artifact_bytes(b"second", media_type="application/json")

    assert files.artifact_ids() == {first["artifact_id"], second["artifact_id"]}

    files.delete_artifact(first["artifact_id"])
    files.delete_artifact(first["artifact_id"])

    assert files.artifact_ids() == {second["artifact_id"]}


@pytest.mark.parametrize(
    ("header", "size", "expected"),
    [
        ("bytes=0-9", 100, (0, 9)),
        ("bytes=10-", 100, (10, 99)),
        ("bytes=-10", 100, (90, 99)),
        ("bytes=0-999", 100, (0, 99)),
        (f"bytes=0-{'9' * 5000}", 100, (0, 99)),
        (f"bytes=-{'9' * 5000}", 100, (0, 99)),
    ],
)
def test_byte_ranges_clip_without_unbounded_integer_conversion(header, size, expected):
    assert parse_byte_range(header, size) == expected


@pytest.mark.parametrize(
    "header",
    [
        "bytes=",
        "bytes=-0",
        "bytes=100-",
        "bytes=50-49",
        "bytes=0-1,3-4",
        "items=0-1",
        f"bytes={'9' * 5000}-",
    ],
)
def test_invalid_or_unsatisfiable_byte_ranges_fail_closed(header):
    with pytest.raises(ValueError, match="not satisfiable"):
        parse_byte_range(header, 100)


def test_worker_cancellation_cannot_release_the_volume_gate_early(tmp_path):
    first_entered = threading.Event()
    release_first = threading.Event()
    entries = []

    def reload_volume():
        entries.append(time.monotonic())
        if len(entries) == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)

    files = SourceFileStore(tmp_path, reload_volume=reload_volume)
    second_files = ArtifactFileStore(tmp_path, reload_volume=reload_volume)

    async def exercise():
        first = asyncio.create_task(asyncio.to_thread(files.source_ids))
        assert await asyncio.to_thread(first_entered.wait, 2)
        first.cancel()
        second = asyncio.create_task(asyncio.to_thread(second_files.artifact_ids))
        await asyncio.sleep(0.05)
        assert len(entries) == 1
        release_first.set()
        await second
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(exercise())
    assert len(entries) == 2
