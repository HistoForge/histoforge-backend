"""Safe ZIP64 extraction and exact training-package inventory validation."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from histoforge.canonical_json import canonical_bytes, parse_json
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.limits import validate_training_api_limits

_READ_BLOCK_BYTES = 1024 * 1024
_ALLOWED_COMPRESSION = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_END_RECORD = struct.Struct("<4s4H2LH")
_END_SIGNATURE = b"PK\x05\x06"
_ZIP64_END_RECORD = struct.Struct("<4sQ2H2L4Q")
_ZIP64_END_SIGNATURE = b"PK\x06\x06"
_ZIP64_LOCATOR = struct.Struct("<4sLQL")
_ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
_MAX_ZIP_COMMENT_BYTES = 65_535
_MAX_CENTRAL_DIRECTORY_BYTES_PER_MEMBER = 512
_UINT16_MAX = 2**16 - 1
_UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class ExtractedTrainingDataset:
    """One validated request-private dataset root ready for content admission."""

    root: Path
    manifest: dict[str, Any]
    retained_bytes: int


def extract_training_dataset_archive(
    archive: str | Path,
    destination: str | Path,
    api_limits: Any,
) -> ExtractedTrainingDataset:
    """Extract one exact package without trusting archive paths or metadata."""

    limits = validate_training_api_limits(api_limits)["dataset"]
    archive_path = Path(archive)
    target = Path(destination)
    if (
        archive_path.is_symlink()
        or not archive_path.is_file()
        or archive_path.stat().st_size > limits["max_archive_bytes"]
    ):
        raise ValueError("training dataset archive is not an admissible regular file")
    if target.exists() or not target.parent.is_dir():
        raise ValueError("training dataset extraction root must be a fresh local path")
    maximum_entries = _require_bounded_central_directory(
        archive_path, limits["max_rois"]
    )

    try:
        with zipfile.ZipFile(archive_path, "r", allowZip64=True) as package:
            members = _members(package, limits, maximum_entries)
            manifest_info = members.get("dataset.json")
            if manifest_info is None:
                raise ValueError("training dataset archive is missing dataset.json")
            manifest_bytes = _read_member(
                package,
                manifest_info,
                limits["max_json_member_bytes"],
            )
            manifest = validate_training_dataset_manifest(
                parse_json(manifest_bytes), api_limits
            )
            if manifest_bytes != canonical_bytes(manifest):
                raise ValueError("dataset.json is not canonical")

            inventory = {item["path"]: item for item in manifest["files"]}
            if set(members) != {"dataset.json", *inventory}:
                raise ValueError("archive members do not match the dataset manifest")

            target.mkdir()
            _write_bytes(target / "dataset.json", manifest_bytes)
            retained_bytes = len(manifest_bytes)
            for relative_path, expected in inventory.items():
                member = members[relative_path]
                if member.file_size != expected["size_bytes"]:
                    raise ValueError(f"archive member length differs: {relative_path}")
                output = target.joinpath(*PurePosixPath(relative_path).parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                digest, size = _extract_member(package, member, output)
                if size != expected["size_bytes"] or digest != expected["sha256"]:
                    raise ValueError(f"archive member bytes differ: {relative_path}")
                retained_bytes += size
            return ExtractedTrainingDataset(target, manifest, retained_bytes)
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as error:
        if target.exists():
            shutil.rmtree(target)
        raise ValueError("invalid training dataset archive") from error
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        raise


def _members(
    package: zipfile.ZipFile,
    limits: dict[str, int],
    maximum_entries: int,
) -> dict[str, zipfile.ZipInfo]:
    entries = package.infolist()
    if len(entries) > maximum_entries:
        raise ValueError("training archive has too many members")
    result: dict[str, zipfile.ZipInfo] = {}
    folded: set[str] = set()
    extracted_bytes = 0
    for member in entries:
        path = _member_path(member)
        if path in result:
            raise ValueError(f"duplicate archive member: {path}")
        case_key = path.casefold()
        if case_key in folded:
            raise ValueError(f"case-colliding archive member: {path}")
        if member.flag_bits & 0x1:
            raise ValueError("encrypted archive members are not supported")
        if member.compress_type not in _ALLOWED_COMPRESSION:
            raise ValueError("unsupported training archive compression")
        if member.file_size <= 0:
            raise ValueError("training archive members must be nonempty")
        if (
            path.endswith((".json", ".geojson"))
            and member.file_size > limits["max_json_member_bytes"]
        ):
            raise ValueError("archive JSON member exceeds max_json_member_bytes")
        extracted_bytes += member.file_size
        if extracted_bytes > limits["max_extracted_bytes"]:
            raise ValueError("training archive exceeds max_extracted_bytes")
        result[path] = member
        folded.add(case_key)

    return result


def _require_bounded_central_directory(archive: Path, max_rois: int) -> int:
    """Bound ZIP metadata before ``ZipFile`` allocates its member inventory."""

    maximum_entries = 4 * max_rois + 1
    archive_size = archive.stat().st_size
    tail_size = min(archive_size, _END_RECORD.size + _MAX_ZIP_COMMENT_BYTES)
    with archive.open("rb") as stream:
        stream.seek(-tail_size, os.SEEK_END)
        tail = stream.read(tail_size)

    offset = len(tail)
    record = None
    while (offset := tail.rfind(_END_SIGNATURE, 0, offset)) >= 0:
        if offset + _END_RECORD.size <= len(tail):
            candidate = _END_RECORD.unpack_from(tail, offset)
            if offset + _END_RECORD.size + candidate[-1] == len(tail):
                record = candidate
                break
    if record is None:
        raise ValueError("training dataset archive has no canonical end record")

    end_offset = archive_size - tail_size + offset
    (
        _,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        _,
    ) = record
    if disk != 0 or central_disk != 0 or disk_entries != total_entries:
        raise ValueError("multi-disk training archives are not supported")
    central_end = end_offset
    # Writers may emit ZIP64 end records before any classic field overflows.
    # Python's zipfile does so above 2 GiB, while those fields fit until 4 GiB.
    locator_present = False
    if end_offset >= _ZIP64_LOCATOR.size:
        with archive.open("rb") as stream:
            stream.seek(end_offset - _ZIP64_LOCATOR.size)
            locator_present = stream.read(4) == _ZIP64_LOCATOR_SIGNATURE
    if (
        locator_present
        or total_entries == _UINT16_MAX
        or central_size == _UINT32_MAX
        or central_offset == _UINT32_MAX
    ):
        (
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            central_end,
        ) = _read_zip64_end_record(
            archive,
            end_offset,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
        )
        if disk_entries != total_entries:
            raise ValueError("multi-disk training archives are not supported")
    if total_entries == 0 or total_entries > maximum_entries:
        raise ValueError("training archive has too many members")
    maximum_central_size = maximum_entries * _MAX_CENTRAL_DIRECTORY_BYTES_PER_MEMBER
    if central_size <= 0 or central_size > maximum_central_size:
        raise ValueError("training archive central directory is too large")
    if central_offset < 0 or central_offset + central_size != central_end:
        raise ValueError("training archive central directory is inconsistent")
    return maximum_entries


def _read_zip64_end_record(
    archive: Path,
    end_offset: int,
    classic_disk_entries: int,
    classic_total_entries: int,
    classic_central_size: int,
    classic_central_offset: int,
) -> tuple[int, int, int, int, int]:
    locator_offset = end_offset - _ZIP64_LOCATOR.size
    if locator_offset < 0:
        raise ValueError("training dataset archive has no ZIP64 locator")
    with archive.open("rb") as stream:
        stream.seek(locator_offset)
        locator_bytes = stream.read(_ZIP64_LOCATOR.size)
        if len(locator_bytes) != _ZIP64_LOCATOR.size:
            raise ValueError("training dataset archive has no ZIP64 locator")
        signature, disk, record_offset, disks = _ZIP64_LOCATOR.unpack(locator_bytes)
        if signature != _ZIP64_LOCATOR_SIGNATURE or disk != 0 or disks != 1:
            raise ValueError("invalid ZIP64 training archive locator")
        if (
            record_offset < 0
            or record_offset + _ZIP64_END_RECORD.size != locator_offset
        ):
            raise ValueError("invalid ZIP64 training archive record offset")
        stream.seek(record_offset)
        record_bytes = stream.read(_ZIP64_END_RECORD.size)
    if len(record_bytes) != _ZIP64_END_RECORD.size:
        raise ValueError("training dataset archive has no ZIP64 end record")
    (
        signature,
        record_size,
        _,
        _,
        disk,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
    ) = _ZIP64_END_RECORD.unpack(record_bytes)
    if (
        signature != _ZIP64_END_SIGNATURE
        or record_size != _ZIP64_END_RECORD.size - 12
        or disk != 0
        or central_disk != 0
    ):
        raise ValueError("invalid ZIP64 training archive end record")
    _require_classic_or_zip64_equal(
        classic_disk_entries, _UINT16_MAX, disk_entries, "entry count"
    )
    _require_classic_or_zip64_equal(
        classic_total_entries, _UINT16_MAX, total_entries, "entry count"
    )
    _require_classic_or_zip64_equal(
        classic_central_size, _UINT32_MAX, central_size, "central directory size"
    )
    _require_classic_or_zip64_equal(
        classic_central_offset,
        _UINT32_MAX,
        central_offset,
        "central directory offset",
    )
    return disk_entries, total_entries, central_size, central_offset, record_offset


def _require_classic_or_zip64_equal(
    classic: int,
    sentinel: int,
    zip64: int,
    field: str,
) -> None:
    if classic != sentinel and classic != zip64:
        raise ValueError(f"classic and ZIP64 {field} differ")


def _member_path(member: zipfile.ZipInfo) -> str:
    if member.is_dir() or member.filename != member.orig_filename:
        raise ValueError("training archive contains a directory or invalid name")
    path = member.filename
    if (
        not path
        or not path.isascii()
        or path.startswith("/")
        or "\\" in path
        or "//" in path
    ):
        raise ValueError("training archive path is not normalized relative ASCII")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("training archive path is not normalized relative ASCII")

    mode = member.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG}:
        raise ValueError("training archive contains a link or non-file member")
    return path


def _read_member(
    package: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    maximum: int,
) -> bytes:
    with package.open(member, "r") as source:
        value = source.read(maximum + 1)
        if len(value) > maximum or source.read(1):
            raise ValueError("training archive member exceeds its declared limit")
        return value


def _extract_member(
    package: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with package.open(member, "r") as source, destination.open("xb") as output:
            while block := source.read(_READ_BLOCK_BYTES):
                size += len(block)
                if size > member.file_size:
                    raise ValueError("archive member exceeds its declared length")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest(), size


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as output:
        output.write(value)
        output.flush()
        os.fsync(output.fileno())
