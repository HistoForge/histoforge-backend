from __future__ import annotations

import copy
import hashlib
import stat
import struct
import warnings
import zipfile
from pathlib import Path

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud.training.datasets.archive import (
    extract_training_dataset_archive,
)
from histoforge.cloud.training.datasets.manifest import build_training_dataset_manifest
from tests.test_training_dataset import manifest_preimage, training_api_limits


def test_archive_extracts_only_exact_manifest_members(tmp_path: Path) -> None:
    archive, manifest, payloads = _archive(tmp_path)
    extracted = extract_training_dataset_archive(
        archive, tmp_path / "extracted", training_api_limits()
    )
    assert extracted.manifest == manifest
    assert extracted.retained_bytes == len(canonical_bytes(manifest)) + sum(
        len(value) for value in payloads.values()
    )
    assert (extracted.root / "dataset.json").read_bytes() == canonical_bytes(manifest)
    for relative, value in payloads.items():
        assert extracted.root.joinpath(*relative.split("/")).read_bytes() == value


@pytest.mark.parametrize("extra_name", ["../escape", "/absolute", "a\\b"])
def test_archive_rejects_unsafe_paths(tmp_path: Path, extra_name: str) -> None:
    archive, _, _ = _archive(tmp_path, extra={extra_name: b"bad"})
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="path|members"):
        extract_training_dataset_archive(archive, destination, training_api_limits())
    assert not destination.exists()


def test_archive_rejects_duplicate_and_case_colliding_paths(tmp_path: Path) -> None:
    archive, manifest, payloads = _archive(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "a") as package:
            package.writestr("dataset.json", canonical_bytes(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        extract_training_dataset_archive(
            archive, tmp_path / "duplicate", training_api_limits()
        )

    archive = tmp_path / "case.zip"
    _write_archive(
        archive,
        canonical_bytes(manifest),
        {**payloads, "ARTIFACTS/0/origin.json": b"collision"},
    )
    with pytest.raises(ValueError, match="case-colliding"):
        extract_training_dataset_archive(
            archive, tmp_path / "case-output", training_api_limits()
        )


def test_archive_rejects_link_and_digest_mismatch(tmp_path: Path) -> None:
    archive, manifest, payloads = _archive(tmp_path)
    link_archive = tmp_path / "link.zip"
    with zipfile.ZipFile(link_archive, "w", allowZip64=True) as package:
        package.writestr("dataset.json", canonical_bytes(manifest))
        for name, value in payloads.items():
            info = zipfile.ZipInfo(name)
            if name == "artifacts/0/source.png":
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            package.writestr(info, value)
    with pytest.raises(ValueError, match="link|non-file"):
        extract_training_dataset_archive(
            link_archive, tmp_path / "link-output", training_api_limits()
        )

    wrong = dict(payloads)
    wrong["rois/0/roi.geojson"] = b"different"
    wrong_archive = tmp_path / "wrong.zip"
    _write_archive(wrong_archive, canonical_bytes(manifest), wrong)
    with pytest.raises(ValueError, match="length differs|bytes differ"):
        extract_training_dataset_archive(
            wrong_archive, tmp_path / "wrong-output", training_api_limits()
        )
    assert not (tmp_path / "wrong-output").exists()


def test_archive_rejects_noncanonical_manifest_and_extra_member(tmp_path: Path) -> None:
    archive, manifest, payloads = _archive(tmp_path)
    noncanonical = tmp_path / "noncanonical.zip"
    _write_archive(
        noncanonical,
        b"  " + canonical_bytes(manifest),
        payloads,
    )
    with pytest.raises(ValueError, match="canonical"):
        extract_training_dataset_archive(
            noncanonical, tmp_path / "noncanonical-output", training_api_limits()
        )

    with zipfile.ZipFile(archive, "a") as package:
        package.writestr("extra.txt", b"extra")
    with pytest.raises(ValueError, match="members"):
        extract_training_dataset_archive(
            archive, tmp_path / "extra-output", training_api_limits()
        )


def test_archive_rejects_excessive_entry_count_before_inventory(tmp_path: Path) -> None:
    extras = {f"extra/{index}.bin": b"x" for index in range(77)}
    archive, _, _ = _archive(tmp_path, extra=extras)

    with pytest.raises(ValueError, match="too many members"):
        extract_training_dataset_archive(
            archive, tmp_path / "too-many-members", training_api_limits()
        )


def test_archive_rejects_forged_end_record_count(tmp_path: Path) -> None:
    extras = {f"extra/{index}.bin": b"x" for index in range(77)}
    archive, _, _ = _archive(tmp_path, extra=extras)
    encoded = bytearray(archive.read_bytes())
    end_offset = encoded.rfind(b"PK\x05\x06")
    assert end_offset >= 0
    struct.pack_into("<HH", encoded, end_offset + 8, 1, 1)
    archive.write_bytes(encoded)

    with pytest.raises(ValueError, match="central directory|too many members"):
        extract_training_dataset_archive(
            archive, tmp_path / "forged-count", training_api_limits()
        )


def test_archive_rejects_oversized_central_directory_metadata(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "metadata.zip"
    with zipfile.ZipFile(archive, "w", allowZip64=True) as package:
        entry = zipfile.ZipInfo("dataset.json")
        entry.comment = b"x" * 4096
        package.writestr(entry, b"{}")

    limits = training_api_limits()
    limits["dataset"]["max_rois"] = 1
    with pytest.raises(ValueError, match="central directory is too large"):
        extract_training_dataset_archive(archive, tmp_path / "metadata", limits)


def test_archive_reads_bounded_zip64_end_record(tmp_path: Path) -> None:
    archive, manifest, payloads = _archive(tmp_path)
    _promote_end_record_to_zip64(archive)

    extracted = extract_training_dataset_archive(
        archive, tmp_path / "zip64", training_api_limits()
    )
    assert extracted.manifest == manifest
    for relative, value in payloads.items():
        assert extracted.root.joinpath(*relative.split("/")).read_bytes() == value


@pytest.mark.parametrize("comment_size", [0, 65_535])
def test_archive_reads_python_zip64_before_classic_field_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, comment_size: int
) -> None:
    """Reproduce the real 2-GiB writer boundary without a huge unit fixture."""

    with monkeypatch.context() as patch:
        patch.setattr(zipfile, "ZIP64_LIMIT", 1000)
        archive, manifest, payloads = _archive(tmp_path)
        with zipfile.ZipFile(archive, "a") as package:
            package.comment = b"x" * comment_size
    encoded = archive.read_bytes()
    end_offset = len(encoded) - 22 - comment_size
    classic = struct.unpack_from("<4s4H2LH", encoded, end_offset)
    assert classic[0] == b"PK\x05\x06"
    assert all(value != 0xFFFF for value in classic[3:5])
    assert all(value != 0xFFFFFFFF for value in classic[5:7])
    assert encoded[end_offset - 20 : end_offset - 16] == b"PK\x06\x07"

    extracted = extract_training_dataset_archive(
        archive, tmp_path / "python-zip64", training_api_limits()
    )
    assert extracted.manifest == manifest
    for relative, value in payloads.items():
        assert (extracted.root / relative).read_bytes() == value


@pytest.mark.parametrize(
    "field", ["entry count", "central directory size", "central directory offset"]
)
def test_optional_zip64_must_agree_with_classic_fields(tmp_path: Path, field: str):
    archive, _, _ = _archive(tmp_path)
    _promote_end_record_to_zip64(archive, sentinels=False)
    encoded = bytearray(archive.read_bytes())
    zip64_offset = len(encoded) - 98
    field_offset = {
        "entry count": 24,
        "central directory size": 40,
        "central directory offset": 48,
    }[field]
    actual = struct.unpack_from("<Q", encoded, zip64_offset + field_offset)[0]
    struct.pack_into("<Q", encoded, zip64_offset + field_offset, actual + 1)
    archive.write_bytes(encoded)

    with pytest.raises(ValueError, match=f"classic and ZIP64 {field} differ"):
        extract_training_dataset_archive(
            archive, tmp_path / "invalid-zip64", training_api_limits()
        )
    assert not (tmp_path / "invalid-zip64").exists()


def test_archive_rejects_oversized_manifest_before_parsing(tmp_path: Path) -> None:
    archive, manifest, _ = _archive(tmp_path)
    limits = training_api_limits()
    limits["dataset"]["max_json_member_bytes"] = len(canonical_bytes(manifest)) - 1

    with pytest.raises(ValueError, match="max_json_member_bytes"):
        extract_training_dataset_archive(archive, tmp_path / "oversized", limits)
    assert not (tmp_path / "oversized").exists()


def test_archive_applies_json_limit_without_repeating_source_artifact_limit(
    tmp_path: Path,
) -> None:
    limits = training_api_limits()
    limits["dataset"]["max_json_member_bytes"] = 4096
    large_source = b"p" * 4097
    archive, _, _ = _archive(tmp_path, source=large_source, limits=limits)

    extracted = extract_training_dataset_archive(
        archive, tmp_path / "large-binary", limits
    )
    assert (extracted.root / "artifacts/0/source.png").read_bytes() == large_source

    archive, manifest, payloads = _archive(tmp_path / "structured-source")
    oversized = tmp_path / "oversized-structured.zip"
    _write_archive(
        oversized,
        canonical_bytes(manifest),
        {**payloads, "rois/0/objects.geojson": b"x" * 4097},
    )
    with pytest.raises(ValueError, match="max_json_member_bytes"):
        extract_training_dataset_archive(
            oversized, tmp_path / "large-structured", limits
        )


def _archive(
    tmp_path: Path,
    *,
    extra: dict[str, bytes] | None = None,
    source: bytes = b"png",
    limits: dict | None = None,
) -> tuple[Path, dict, dict[str, bytes]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payloads = {
        "artifacts/0/origin.json": b"origin",
        "artifacts/0/source.png": source,
        "rois/0/objects.geojson": b"objects",
        "rois/0/roi.geojson": b"roi",
    }
    preimage = copy.deepcopy(manifest_preimage())
    preimage["files"] = [
        {
            "path": path,
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
        for path, value in sorted(payloads.items())
    ]
    manifest = build_training_dataset_manifest(
        preimage, limits if limits is not None else training_api_limits()
    )
    archive = tmp_path / "dataset.zip"
    _write_archive(
        archive,
        canonical_bytes(manifest),
        {**payloads, **(extra or {})},
    )
    return archive, manifest, payloads


def _write_archive(
    path: Path,
    manifest: bytes,
    payloads: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as package:
        package.writestr("dataset.json", manifest)
        for name, value in payloads.items():
            package.writestr(name, value)


def _promote_end_record_to_zip64(path: Path, *, sentinels: bool = True) -> None:
    encoded = path.read_bytes()
    end_offset = encoded.rfind(b"PK\x05\x06")
    assert end_offset >= 0
    end = struct.unpack_from("<4s4H2LH", encoded, end_offset)
    _, disk, central_disk, disk_entries, total_entries, size, offset, comment_size = end
    assert disk == central_disk == 0
    assert disk_entries == total_entries
    assert comment_size == 0

    zip64 = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        disk_entries,
        total_entries,
        size,
        offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, end_offset, 1)
    classic = struct.pack(
        "<4s4H2LH",
        b"PK\x05\x06",
        0,
        0,
        0xFFFF if sentinels else disk_entries,
        0xFFFF if sentinels else total_entries,
        0xFFFFFFFF if sentinels else size,
        0xFFFFFFFF if sentinels else offset,
        0,
    )
    path.write_bytes(encoded[:end_offset] + zip64 + locator + classic)
