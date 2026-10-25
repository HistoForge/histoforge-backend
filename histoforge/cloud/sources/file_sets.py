"""Portable original-slide layouts and bounded, uncompressed ZIP extraction.

The source store owns the package and extracted directory. This module neither
publishes nor deletes a source, and never follows archive-supplied links.
"""

from __future__ import annotations

import hashlib
import stat
import struct
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    require_nonnegative_safe_integer,
    require_sha256,
)

# Shared source-wire limits, mirrored by the extension's SlideFileSetCodec.
# The existing raw/canonical start-body limits still apply independently.
MAX_MEMBERS = 256
MAX_PATH_BYTES = 512
MAX_TOTAL_PATH_BYTES = 16 * 1024
MAX_ZIP_DIRECTORY_BYTES = 256 * 1024
COPY_BLOCK_BYTES = 1024 * 1024


def relative_slide_path(value: Any) -> str:
    """Require a portable, unchanged POSIX member name, not a host path."""

    if not isinstance(value, str) or not value:
        raise ValueError("slide member path must be a nonempty string")
    if (
        len(value.encode("utf-8")) > MAX_PATH_BYTES
        or any(ord(char) < 32 for char in value)
        or any(char in value for char in "\\:")
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or unicodedata.normalize("NFC", value) != value
    ):
        raise ValueError("slide member path must be normalized portable relative text")
    return value


def validate_member_paths(paths: list[str]) -> None:
    """Reject aliases and file/directory conflicts before any filesystem access."""

    if not 1 <= len(paths) <= MAX_MEMBERS:
        raise ValueError("slide fileset member count exceeds its bound")
    checked = [relative_slide_path(path) for path in paths]
    if sum(len(path.encode("utf-8")) for path in checked) > MAX_TOTAL_PATH_BYTES:
        raise ValueError("slide fileset path metadata exceeds its bound")
    # Uppercase is the paired Java Locale.ROOT policy. Never rename a member.
    folded = [path.upper() for path in checked]
    if len(set(folded)) != len(folded):
        raise ValueError("slide fileset contains duplicate or case-aliased paths")
    names = set(folded)
    for path in folded:
        parts = path.split("/")
        if any("/".join(parts[:index]) in names for index in range(1, len(parts))):
            raise ValueError("slide member is both a file and a directory")


def member_order(path: str) -> bytes:
    """UTF-8 byte order is identical in both wire implementations."""

    return path.encode("utf-8")


def validate_members(value: Any) -> list[dict[str, Any]]:
    """Normalize exact original-member byte evidence without reading files."""

    if not isinstance(value, list) or not 1 <= len(value) <= MAX_MEMBERS:
        raise ValueError("slide members must be a bounded nonempty array")
    members = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "size_bytes", "sha256"}:
            raise ValueError("slide member fields must be path, size_bytes, sha256")
        members.append(
            {
                "path": relative_slide_path(item["path"]),
                "size_bytes": require_nonnegative_safe_integer(
                    item["size_bytes"], "member.size_bytes"
                ),
                "sha256": require_sha256(item["sha256"], "member.sha256"),
            }
        )
    validate_member_paths([member["path"] for member in members])
    return sorted(members, key=lambda member: member_order(member["path"]))


def extract_slide_files(
    package: Path, destination: Path, members: list[dict[str, Any]]
) -> None:
    """Extract a declared STORED ZIP into a new source-store-owned directory.

    Each write has a declared byte bound and a streaming SHA-256 check. The
    caller owns disposal after failure. No partial directory is a sealed source.
    """

    expected = validate_members(members)
    _bound_zip_directory(package, len(expected))
    destination.mkdir(exist_ok=False)
    try:
        with zipfile.ZipFile(package) as archive:
            entries = archive.infolist()
            if [entry.filename for entry in entries] != [m["path"] for m in expected]:
                raise ValueError(
                    "ZIP members/order differ from the declared slide layout"
                )
            for entry, member in zip(entries, expected):
                mode = entry.external_attr >> 16
                if (
                    entry.orig_filename != entry.filename
                    or entry.comment
                    or entry.is_dir()
                    or entry.flag_bits & 1
                    or (not entry.flag_bits & 0x800 and not entry.filename.isascii())
                    or entry.compress_type != zipfile.ZIP_STORED
                    or entry.file_size != member["size_bytes"]
                    or entry.compress_size != member["size_bytes"]
                    or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})
                ):
                    raise ValueError(
                        "ZIP member must be a declared regular uncompressed file"
                    )
                path = destination.joinpath(*member["path"].split("/"))
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                remaining = member["size_bytes"]
                # ZipFile.open checks the local name; EOF checks CRC, truncation
                # and overlap. Those are invalid input, not internal failures.
                with archive.open(entry) as source, path.open("xb") as target:
                    while remaining:
                        block = source.read(min(COPY_BLOCK_BYTES, remaining))
                        if not block:
                            raise ValueError(
                                "ZIP member ended before its declared byte count"
                            )
                        target.write(block)
                        digest.update(block)
                        remaining -= len(block)
                    if source.read(1):
                        raise ValueError("ZIP member exceeds its declared byte count")
                    target.flush()
                    # Store fsync/atomic publication precedes Volume commit.
                if digest.hexdigest() != member["sha256"]:
                    raise ValueError("ZIP member SHA-256 differs from its declaration")
    except zipfile.BadZipFile as error:
        raise ValueError("invalid original slide ZIP") from error


def _bound_zip_directory(package: Path, member_count: int) -> None:
    """Bound metadata before ZipFile allocates its complete central directory."""

    size = package.stat().st_size
    with package.open("rb") as source:
        source.seek(max(0, size - 65557))
        tail = source.read(65557)
        position = tail.rfind(b"PK\x05\x06")
        if position < 0 or len(tail) - position < 22:
            raise ValueError("ZIP end record is missing")
        _, disk, directory_disk, disk_count, count, length, offset, comment = (
            struct.unpack_from("<4s4H2LH", tail, position)
        )
        end_offset = size - len(tail) + position
        if disk or directory_disk or comment or position + 22 != len(tail):
            raise ValueError("split ZIPs, comments and trailing data are unsupported")
        requires_zip64 = count == 0xFFFF or length == 0xFFFFFFFF or offset == 0xFFFFFFFF
        locator = b""
        if end_offset >= 20:
            source.seek(end_offset - 20)
            locator = source.read(20)
        # ZipFile honors this locator even when the classic EOCD has no
        # sentinel fields. Bound the same authoritative directory it will read.
        if locator.startswith(b"PK\x06\x07"):
            signature, locator_disk, record_offset, disks = struct.unpack(
                "<4sLQL", locator
            )
            if signature != b"PK\x06\x07" or locator_disk or disks != 1:
                raise ValueError("invalid ZIP64 locator")
            if record_offset + 56 != end_offset - 20:
                raise ValueError("unsupported ZIP64 end record extent")
            source.seek(record_offset)
            fields = struct.unpack("<4sQ2H2L4Q", source.read(56))
            if fields[0] != b"PK\x06\x06" or fields[1] != 44 or fields[4] or fields[5]:
                raise ValueError("invalid ZIP64 end record")
            disk_count, count, length, offset = fields[6:]
            end_offset = record_offset
        elif requires_zip64:
            raise ValueError("ZIP64 locator is missing")
        if (
            count != member_count
            or disk_count != count
            or length > MAX_ZIP_DIRECTORY_BYTES
            or offset + length != end_offset
        ):
            raise ValueError(
                "ZIP directory exceeds or differs from the declared layout"
            )
