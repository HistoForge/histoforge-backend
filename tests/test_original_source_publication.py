"""Real upload/seal/reopen with original names and immutable companion bytes."""

import copy
import hashlib
import io
import zipfile
from unittest.mock import patch

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.reader import SourceReader

from test_source_artifact_store import FakeDict, _store, _upload_all
from test_source_reader import _limits, _openslide_path, _openslide_start


def _original(fileset):
    path = _openslide_path()
    start = _openslide_start(path)
    payload = path.read_bytes()
    if fileset:
        members = {path.name: payload, "sidecars/empty.dat": b""}
        encoded = io.BytesIO()
        with zipfile.ZipFile(encoded, "w", compression=zipfile.ZIP_STORED) as package:
            for name, value in sorted(members.items()):
                package.writestr(name, value)
        payload = encoded.getvalue()
        start["container"] = "SLIDE_FILE_SET"
        start["total_bytes"] = len(payload)
        start["origin"]["members"] = [
            {
                "path": name,
                "size_bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for name, value in members.items()
        ]
        start["origin"]["source"]["content_version"] = {
            "type": "local-file-set",
            "members": [
                {
                    "path": name,
                    "size_bytes": len(value),
                    "last_modified_epoch_millis": 1,
                }
                for name, value in members.items()
            ],
        }
    start["chunk_size_bytes"] = 65536
    return start, payload


@pytest.mark.parametrize("fileset", [False, True])
def test_original_seal_preserves_names_and_exact_readback(tmp_path, fileset):
    start, payload = _original(fileset)
    files = SourceFileStore(tmp_path)
    store = _store(FakeDict())
    store.reserve(start)
    _upload_all(store, files, start, payload)
    sealed = store.seal(
        start["artifact_id"],
        {"bytes_digest": hashlib.sha256(payload).hexdigest()},
        files,
    )
    resolved = store.resolve_sealed_reference(sealed["artifact"])
    rectangle = {"x": 10, "y": 30, "width": 128, "height": 90}
    with SourceReader(
        _openslide_path(), _openslide_start(_openslide_path()), _limits()
    ) as original:
        expected = original.read_rgb_view(0, 0, rectangle)
    with files.source_view_reader(start, resolved["manifest"], 0, 0, _limits()) as read:
        actual = read(rectangle)["pixels"]
    assert (actual == expected).all()
    root = tmp_path / "sources" / start["artifact_id"]
    assert (root / "slide" / "members" / _openslide_path().name).is_file()
    assert not (root / ".slide-stage").exists()
    assert not (root / "chunks").exists()


def test_missing_or_changed_companion_invalidates_published_source(tmp_path):
    start, payload = _original(True)
    files = SourceFileStore(tmp_path)
    store = _store(FakeDict())
    store.reserve(start)
    _upload_all(store, files, start, payload)
    sealed = store.seal(
        start["artifact_id"],
        {"bytes_digest": hashlib.sha256(payload).hexdigest()},
        files,
    )
    manifest = store.resolve_sealed_reference(sealed["artifact"])["manifest"]
    companion = (
        tmp_path / "sources" / start["artifact_id"] / "slide/members/sidecars/empty.dat"
    )
    companion.unlink()
    with pytest.raises(ArtifactBytesCorrupt, match="membership"):
        files.source_level_metadata(start, manifest, 0, 0, _limits())
    companion.write_bytes(b"changed")
    with pytest.raises(ArtifactBytesCorrupt, match="length"):
        files.source_level_metadata(start, manifest, 0, 0, _limits())


def test_interrupted_original_manifest_reuses_published_bytes(tmp_path):
    start, payload = _original(True)
    files = SourceFileStore(tmp_path)
    store = _store(FakeDict())
    store.reserve(start)
    _upload_all(store, files, start, payload)
    commits = {
        i: hashlib.sha256(
            payload[offset : offset + start["chunk_size_bytes"]]
        ).hexdigest()
        for i, offset in enumerate(range(0, len(payload), start["chunk_size_bytes"]))
    }
    seal = {"bytes_digest": hashlib.sha256(payload).hexdigest()}
    with patch(
        "histoforge.cloud.sources.files.publish_bytes",
        side_effect=OSError("manifest failure"),
    ):
        with pytest.raises(OSError, match="manifest failure"):
            files.seal_source(start, seal, commits, _limits())
    with patch.object(
        files,
        "_source_chunk_path",
        side_effect=AssertionError("must reuse published bytes"),
    ):
        manifest = files.seal_source(start, seal, commits, _limits())
    assert manifest["bytes_digest"] == seal["bytes_digest"]


def test_failed_extraction_does_not_publish_or_leave_stage(tmp_path):
    start, payload = _original(True)
    start["origin"]["members"][0]["sha256"] = "0" * 64
    files = SourceFileStore(tmp_path)
    store = _store(FakeDict())
    store.reserve(start)
    _upload_all(store, files, start, payload)
    with pytest.raises(ValueError, match="SHA-256"):
        files.seal_source(
            start,
            {"bytes_digest": hashlib.sha256(payload).hexdigest()},
            {
                i: hashlib.sha256(
                    payload[offset : offset + start["chunk_size_bytes"]]
                ).hexdigest()
                for i, offset in enumerate(
                    range(0, len(payload), start["chunk_size_bytes"])
                )
            },
            _limits(),
        )
    root = tmp_path / "sources" / start["artifact_id"]
    assert not (root / ".slide-stage").exists()
    assert not (root / "slide").exists()


def test_corrupt_zip_is_a_validation_failure_and_never_publishes(tmp_path):
    start, payload = _original(True)
    name = start["origin"]["entry_path"].encode()
    # Keep central metadata and lengths intact; corrupt only the local header.
    payload = payload.replace(name, b"x" * len(name), 1)
    files = SourceFileStore(tmp_path)
    store = _store(FakeDict())
    store.reserve(start)
    _upload_all(store, files, start, payload)
    with pytest.raises(GatewayFailure) as failure:
        store.seal(
            start["artifact_id"],
            {"bytes_digest": hashlib.sha256(payload).hexdigest()},
            files,
        )
    assert failure.value.error["code"] == "VALIDATION_ERROR"
    root = tmp_path / "sources" / start["artifact_id"]
    assert not (root / ".slide-stage").exists()
    assert not (root / "slide").exists()


def test_fileset_storage_reservation_includes_extracted_members():
    start, _ = _original(True)
    store = _store(FakeDict())
    expected = 2 * start["total_bytes"] + _limits()["max_start_canonical_bytes"]
    expected += sum(member["size_bytes"] for member in start["origin"]["members"])
    assert store._peak_bytes(start) == expected
    changed = copy.deepcopy(start)
    changed["origin"]["members"][0]["size_bytes"] = _limits()["max_artifact_bytes"] + 1
    with pytest.raises(Exception, match="extracted slide"):
        store._validate_start_limits(changed)
