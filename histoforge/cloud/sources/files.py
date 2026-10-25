"""Source-image byte assembly, validation and reads on the Artifact Volume.

SourceArtifactStore owns admission and retention. This owner handles only source
files and readers, using the same mounted-volume gate as generic artifact files.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
)
from histoforge.cloud.artifacts.store import (
    DEFAULT_READ_BLOCK_BYTES,
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
    force_file,
    publish_bytes,
    sha256_file,
)
from histoforge.cloud.artifacts.volume import (
    artifact_volume_gate,
    mounted_artifact_volume_operation,
)
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    canonical_chunk_range,
    validate_source_artifact_start_request,
)
from histoforge.cloud.sources.file_sets import extract_slide_files
from histoforge.cloud.sources.publication import (
    build_source_artifact_manifest,
    validate_source_artifact_manifest,
    validate_source_artifact_publication,
    validate_source_artifact_seal_request,
)
from histoforge.cloud.sources.records import SOURCE_CLEANUP_GROUP_SIZE

if TYPE_CHECKING:
    from histoforge.cloud.sources.reader import SourceReader


class SourceFileStore:
    """Physical owner of immutable uploaded source images and their chunks."""

    def __init__(
        self,
        root: str | Path,
        *,
        reload_volume: Callable[[], None] = lambda: None,
        commit_volume: Callable[[], None] = lambda: None,
    ) -> None:
        self._root = Path(root)
        self._reload_volume = reload_volume
        self._commit_volume = commit_volume
        # Hash each immutable sealed source once per observed filesystem version.
        # Any replacement or in-place modification changes this signature and
        # forces a full digest check before the source is opened again.
        self._verified_sources: dict[
            Path, tuple[str, tuple[int, int, int, int, int]]
        ] = {}

    @mounted_artifact_volume_operation
    def source_level_metadata(
        self,
        start_request: Any,
        manifest: Any,
        series_index: int,
        resolution_level: int,
        source_limits: Any,
    ) -> dict[str, Any]:
        """Validate one committed source and inspect reader-owned level geometry."""

        from histoforge.cloud.sources.reader import SourceReader

        source_path, start, publication = self._validated_source_state(
            start_request, manifest
        )
        level = _manifest_level(publication, series_index, resolution_level)
        # The exact source digest proves these are the bytes fully decoded at
        # seal time, so PNG geometry can be inspected without decoding again.
        try:
            metadata = SourceReader.inspect_level(
                source_path,
                start,
                source_limits,
                series_index,
                resolution_level,
            )
        except (OSError, ValueError) as error:
            raise ArtifactBytesCorrupt(
                "published source cannot be inspected"
            ) from error
        if metadata["validated_levels"] != publication["validated_levels"]:
            raise ArtifactBytesCorrupt(
                "published source levels differ from its committed manifest"
            )
        return {
            "dimensions": {"width": level["width"], "height": level["height"]},
            "selected_level_pixels_to_level_zero": metadata[
                "selected_level_pixels_to_level_zero"
            ],
        }

    @contextmanager
    def source_view_reader(
        self,
        start_request: Any,
        manifest: Any,
        series_index: int,
        resolution_level: int,
        source_limits: Any,
    ):
        """Yield bounded RGB reads sharing one source handle within the Volume gate.

        Callers finish one processing-image read before leaving this scope. Only copied
        pixels may escape it; the callback and handle must not survive into inference.
        Each rectangle still obeys the source reader's complete-decode limit.
        """

        with artifact_volume_gate:
            reader, publication = self._open_validated_source(
                start_request, manifest, source_limits
            )
            try:
                level = _manifest_level(publication, series_index, resolution_level)
                metadata = {
                    "dimensions": {"width": level["width"], "height": level["height"]},
                    "selected_level_pixels_to_level_zero": (
                        reader.selected_level_pixels_to_level_zero(
                            series_index, resolution_level
                        )
                    ),
                }

                def read(pixel_rect):
                    return {
                        **metadata,
                        "pixels": reader.read_rgb_view(
                            series_index, resolution_level, pixel_rect
                        ),
                    }

                yield read
            finally:
                reader.close()

    @mounted_artifact_volume_operation
    def publish_source_chunk(
        self,
        artifact_id: str,
        chunk_id: int,
        value: bytes,
        expected_sha256: str,
        *,
        require_existing: bool = False,
    ) -> None:
        """Publish one immutable source range before its Dict commit is visible."""

        self._reload_volume()
        path = self._source_chunk_path(artifact_id, chunk_id)
        if require_existing and not path.is_file():
            raise ArtifactBytesMissing(f"committed source chunk is missing: {chunk_id}")
        if publish_bytes(path, value, expected_sha256):
            self._commit_volume()

    @mounted_artifact_volume_operation
    def seal_source(
        self,
        start_request: Any,
        seal_request: Any,
        chunk_commits: Mapping[int, str],
        source_limits: Any,
    ) -> dict[str, Any]:
        """Assemble, validate, and atomically publish one complete source file."""

        from histoforge.cloud.sources.reader import SourceReader

        start = validate_source_artifact_start_request(start_request)
        seal = validate_source_artifact_seal_request(seal_request)
        count = canonical_chunk_count(start["total_bytes"], start["chunk_size_bytes"])
        if set(chunk_commits) != set(range(count)):
            raise ValueError("source upload is incomplete")

        self._reload_volume()
        root = self._source_root(start["artifact_id"])
        final_path = _payload_path(root, start)
        manifest_path = root / "manifest.json"
        existing = self._validated_existing_source(
            final_path,
            manifest_path,
            start,
            seal,
        )
        if existing is not None:
            self._require_original_members(root, start)
            return existing

        # Publication can be interrupted after the data rename but before its
        # manifest commit. Revalidate those exact bytes without another assembly.
        if start["origin"]["type"] == "original-wsi" and (root / "slide").exists():
            self._require_source_digest(final_path, seal["bytes_digest"])
            if final_path.stat().st_size != start["total_bytes"]:
                raise ArtifactBytesCorrupt("unpublished original has the wrong size")
            self._require_original_members(root, start)
            with SourceReader(
                _reader_path(root, start), start, source_limits
            ) as reader:
                manifest = build_source_artifact_manifest(
                    start, seal, reader.validated_levels
                )
            encoded = canonical_bytes(manifest)
            publish_bytes(manifest_path, encoded, hashlib.sha256(encoded).hexdigest())
            self._commit_volume()
            return manifest

        root.mkdir(parents=True, exist_ok=True)
        original = start["origin"]["type"] == "original-wsi"
        stage_directory = root / ".slide-stage" if original else None
        if stage_directory is not None:
            # This directory contains only upload-owned incomplete bytes. The
            # existing source mutation owner serializes seal calls for this ID.
            if stage_directory.exists():
                shutil.rmtree(stage_directory)
            stage_directory.mkdir()
            stage_path = _payload_path(stage_directory, start)
            stage_path.parent.mkdir(parents=True)
        else:
            stage_path = root / f".source.{uuid.uuid4().hex}.stage"
        digest = hashlib.sha256()
        try:
            with stage_path.open("xb") as target:
                for chunk_id in range(count):
                    chunk_path = self._source_chunk_path(start["artifact_id"], chunk_id)
                    expected_range = canonical_chunk_range(
                        start["total_bytes"], start["chunk_size_bytes"], chunk_id
                    )
                    observed_chunk = hashlib.sha256()
                    observed_size = 0
                    try:
                        with chunk_path.open("rb") as source:
                            for block in iter(
                                lambda: source.read(DEFAULT_READ_BLOCK_BYTES), b""
                            ):
                                target.write(block)
                                digest.update(block)
                                observed_chunk.update(block)
                                observed_size += len(block)
                    except FileNotFoundError as error:
                        raise ArtifactBytesMissing(
                            f"source chunk is missing: {chunk_id}"
                        ) from error
                    if observed_size != expected_range["size"]:
                        raise ArtifactBytesMissing(
                            f"source chunk has the wrong length: {chunk_id}"
                        )
                    if observed_chunk.hexdigest() != chunk_commits[chunk_id]:
                        raise ArtifactBytesMissing(
                            f"source chunk has the wrong digest: {chunk_id}"
                        )
                force_file(target)

            if stage_path.stat().st_size != start["total_bytes"]:
                raise ArtifactBytesMissing("assembled source has the wrong length")
            observed_digest = digest.hexdigest()
            if observed_digest != seal["bytes_digest"]:
                raise ValueError(
                    "assembled source digest does not match the seal request"
                )
            read_path = stage_path
            if start["container"] == "SLIDE_FILE_SET":
                extract_slide_files(
                    stage_path,
                    stage_directory / "slide" / "members",
                    start["origin"]["members"],
                )
                for member in start["origin"]["members"]:
                    with (stage_directory / "slide" / "members" / member["path"]).open(
                        "rb"
                    ) as extracted:
                        force_file(extracted)
                read_path = _reader_path(stage_directory, start)
            with SourceReader(read_path, start, source_limits) as reader:
                manifest = build_source_artifact_manifest(
                    start,
                    seal,
                    reader.validated_levels,
                )

            if stage_directory is not None:
                os.replace(stage_directory / "slide", root / "slide")
            else:
                self._publish_staged_file(stage_path, final_path, observed_digest)
            encoded_manifest = canonical_bytes(manifest)
            publish_bytes(
                manifest_path,
                encoded_manifest,
                hashlib.sha256(encoded_manifest).hexdigest(),
            )
            self._commit_volume()
            return manifest
        finally:
            if stage_directory is not None:
                shutil.rmtree(stage_directory)
            elif stage_path.exists():
                stage_path.unlink()

    @mounted_artifact_volume_operation
    def cleanup_source_chunks(self, artifact_id: str) -> None:
        """Remove upload-only byte ranges after the sealed ref is durable."""

        self._reload_volume()
        directory = self._source_root(artifact_id) / "chunks"
        if directory.exists():
            shutil.rmtree(directory)
            self._commit_volume()

    def delete_source(self, artifact_id: str) -> None:
        """Remove every physical file for one released or expired source ID."""

        self.delete_sources((artifact_id,))

    @mounted_artifact_volume_operation
    def delete_sources(self, artifact_ids: Iterable[str]) -> None:
        """Delete source directories in groups, with one reload and one commit.

        The caller removes expired records before yielding their IDs. At most
        eight directories are deleted concurrently. Finish every dispatched
        deletion before requesting the next group, committing or reporting an
        error. Earlier IDs still get deleted if the iterator subsequently fails.
        On failure, commit completed or partial deletions; remaining files with
        no record are cleaned on retry. Cleanup or commit failure prevents a new
        image from being admitted.
        """

        self._reload_volume()
        remaining = iter(artifact_ids)
        exhausted = False
        changed = False
        try:
            with ThreadPoolExecutor(max_workers=SOURCE_CLEANUP_GROUP_SIZE) as pool:
                while not exhausted:
                    removals = []
                    pending_error = None
                    try:
                        # Count IDs, including missing directories, so fetching
                        # the next record group cannot overlap file deletions.
                        for _ in range(SOURCE_CLEANUP_GROUP_SIZE):
                            try:
                                artifact_id = next(remaining)
                            except StopIteration:
                                exhausted = True
                                break
                            root = self._source_root(artifact_id)
                            for path in tuple(self._verified_sources):
                                if path.is_relative_to(root):
                                    del self._verified_sources[path]
                            if root.exists():
                                # rmtree may change files before raising.
                                changed = True
                                removals.append(pool.submit(shutil.rmtree, root))
                    except BaseException as error:
                        pending_error = error
                    # Report an earlier file failure before a later iterator
                    # failure. Pool shutdown waits for the other deletions.
                    for removal in removals:
                        removal.result()
                    if pending_error is not None:
                        raise pending_error
        except BaseException as deletion_error:
            if changed:
                try:
                    self._commit_volume()
                except BaseException as commit_error:
                    raise deletion_error from commit_error
            raise
        if changed:
            self._commit_volume()

    @mounted_artifact_volume_operation
    def source_ids(self) -> set[str]:
        """Return canonical source directories visible after one reload."""

        self._reload_volume()
        source_root = self._root / "sources"
        if not source_root.exists():
            return set()
        result = set()
        for path in source_root.iterdir():
            if not path.is_dir():
                continue
            try:
                result.add(require_canonical_uuid(path.name, "artifact_id"))
            except ValueError:
                continue
        return result

    def _validated_existing_source(
        self,
        final_path: Path,
        manifest_path: Path,
        start: Mapping[str, Any],
        seal: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if not final_path.exists() or not manifest_path.exists():
            return None
        if final_path.stat().st_size != start["total_bytes"]:
            raise ArtifactBytesCorrupt("published source has the wrong length")
        if sha256_file(final_path) != seal["bytes_digest"]:
            raise ArtifactBytesCorrupt("published source has the wrong digest")
        try:
            manifest = parse_json(manifest_path.read_bytes())
        except (UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt(
                "published source manifest is invalid"
            ) from error
        try:
            return validate_source_artifact_publication(start, seal, manifest)
        except ValueError as error:
            raise ArtifactBytesCorrupt(
                "published source manifest does not match its immutable source"
            ) from error

    def _open_validated_source(
        self,
        start_request: Any,
        manifest: Any,
        source_limits: Any,
    ) -> tuple[SourceReader, dict[str, Any]]:
        from histoforge.cloud.sources.reader import SourceReader

        source_path, start, publication = self._validated_source_state(
            start_request, manifest
        )
        try:
            reader = SourceReader(source_path, start, source_limits)
        except (OSError, ValueError) as error:
            raise ArtifactBytesCorrupt("published source cannot be reopened") from error
        if reader.validated_levels != publication["validated_levels"]:
            reader.close()
            raise ArtifactBytesCorrupt(
                "published source levels differ from its committed manifest"
            )
        return reader, publication

    def _validated_source_state(
        self,
        start_request: Any,
        manifest: Any,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        start = validate_source_artifact_start_request(start_request)
        supplied_manifest = validate_source_artifact_manifest(manifest)
        publication = validate_source_artifact_publication(
            start,
            {"bytes_digest": supplied_manifest["bytes_digest"]},
            supplied_manifest,
        )
        self._reload_volume()
        root = self._source_root(start["artifact_id"])
        source_path = _payload_path(root, start)
        manifest_path = root / "manifest.json"
        if not source_path.is_file() or not manifest_path.is_file():
            raise ArtifactBytesMissing(start["artifact_id"])
        if source_path.stat().st_size != publication["total_bytes"]:
            raise ArtifactBytesCorrupt("published source has the wrong length")
        self._require_source_digest(source_path, publication["bytes_digest"])
        try:
            stored_manifest = manifest_path.read_bytes()
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(start["artifact_id"]) from error
        if stored_manifest != canonical_bytes(publication):
            raise ArtifactBytesCorrupt(
                "published source manifest differs from committed control state"
            )
        self._require_original_members(root, start)
        return _reader_path(root, start), start, publication

    def _require_original_members(self, root: Path, start: Mapping[str, Any]) -> None:
        if start["container"] != "SLIDE_FILE_SET":
            return
        directory = root / "slide" / "members"
        expected = {member["path"]: member for member in start["origin"]["members"]}
        observed = set()
        for path in directory.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ArtifactBytesCorrupt("published slide has a nonregular member")
            if path.is_file():
                observed.add(path.relative_to(directory).as_posix())
        if observed != set(expected):
            raise ArtifactBytesCorrupt(
                "published slide membership differs from its source"
            )
        for name, member in expected.items():
            path = directory / name
            if path.stat().st_size != member["size_bytes"]:
                raise ArtifactBytesCorrupt(
                    "published slide member has the wrong length"
                )
            self._require_source_digest(path, member["sha256"])

    def _require_source_digest(self, path: Path, expected_digest: str) -> None:
        """Verify sealed bytes once for each stable observed filesystem version."""

        try:
            before = _file_signature(path)
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(path.name) from error
        if self._verified_sources.get(path) == (expected_digest, before):
            return
        try:
            observed_digest = sha256_file(path)
            after = _file_signature(path)
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(path.name) from error
        if before != after:
            raise ArtifactBytesCorrupt(
                "published source changed while its digest was being verified"
            )
        if observed_digest != expected_digest:
            raise ArtifactBytesCorrupt(
                "published source digest does not match its committed manifest"
            )
        self._verified_sources[path] = (expected_digest, after)

    def _publish_staged_file(self, stage: Path, final: Path, digest: str) -> None:
        if final.exists():
            if (
                final.stat().st_size != stage.stat().st_size
                or sha256_file(final) != digest
            ):
                raise ArtifactBytesCorrupt(
                    "immutable source path contains different bytes"
                )
            return
        os.replace(stage, final)

    def _source_root(self, artifact_id: str) -> Path:
        return (
            self._root / "sources" / require_canonical_uuid(artifact_id, "artifact_id")
        )

    def _source_chunk_path(self, artifact_id: str, chunk_id: int) -> Path:
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 0:
            raise ValueError("chunk_id must be a nonnegative integer")
        return self._source_root(artifact_id) / "chunks" / f"{chunk_id}.bin"


def _file_signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _payload_path(root: Path, start: Mapping[str, Any]) -> Path:
    """The uploaded byte stream; only original single files retain their name."""
    if start["container"] == "SLIDE_FILE":
        return root / "slide" / "members" / start["origin"]["entry_path"]
    if start["container"] == "SLIDE_FILE_SET":
        return root / "slide" / "package.zip"
    return root / "source.bin"


def _reader_path(root: Path, start: Mapping[str, Any]) -> Path:
    if start["container"] == "SLIDE_FILE_SET":
        return root / "slide" / "members" / start["origin"]["entry_path"]
    return _payload_path(root, start)


def _manifest_level(
    manifest: Mapping[str, Any], series_index: int, resolution_level: int
) -> Mapping[str, int]:
    matches = [
        level
        for level in manifest["validated_levels"]
        if level["series_index"] == series_index
        and level["resolution_level"] == resolution_level
    ]
    if len(matches) != 1:
        raise ValueError("source series/resolution level is not validated")
    return matches[0]
