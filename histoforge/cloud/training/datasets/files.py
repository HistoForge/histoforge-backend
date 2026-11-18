"""Physical Artifact Volume ownership for training datasets and upload chunks."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
)
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.modules.files import (
    copy_exact_file,
    require_exact_file,
    require_file_closure,
    require_real_directory_outside,
)
from histoforge.cloud.training.artifact_directories import (
    discover_training_artifact_ids,
)
from histoforge.cloud.training.datasets.manifest import (
    validate_training_dataset_manifest,
)
from histoforge.cloud.training.datasets.transfer import (
    training_dataset_chunk_range,
    training_dataset_start_digest,
    validate_training_dataset_chunk_commit,
    validate_training_dataset_start_request,
)
from histoforge.cloud.training.limits import validate_training_api_limits

_READ_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RetainedTrainingDataset:
    """One fully verified retained dataset view on the mounted Volume."""

    root: Path
    manifest: dict[str, Any]
    retained_bytes: int


class TrainingDatasetFiles:
    """Publish and verify exact upload pairs and retained dataset roots."""

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

    @mounted_artifact_volume_operation
    def publish_chunk(
        self,
        dataset_id: str,
        value: bytes,
        commit: Any,
    ) -> dict[str, Any]:
        """Publish one immutable data.bin and receipt.json directory together."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        accepted = validate_training_dataset_chunk_commit(commit)
        if len(value) != accepted["size_bytes"]:
            raise ValueError("chunk body length does not match its receipt")
        if hashlib.sha256(value).hexdigest() != accepted["sha256"]:
            raise ValueError("chunk body digest does not match its receipt")

        self._reload_volume()
        target = self._chunk_root(canonical_id, accepted["chunk_index"])
        if target.exists():
            durable = self._read_chunk_receipt(target)
            self._require_chunk_pair(target, durable)
            if _chunk_identity(durable) != _chunk_identity(accepted):
                raise ArtifactBytesCorrupt(
                    "training chunk receipt does not match the existing upload"
                )
            return durable

        staging_parent = self._upload_root(canonical_id) / "chunk-staging"
        stage = staging_parent / uuid.uuid4().hex
        try:
            stage.mkdir(parents=True)
            self._write_file(stage / "data.bin", value)
            self._write_file(stage / "receipt.json", canonical_bytes(accepted))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, target)
            self._commit_volume()
            return accepted
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            _remove_empty_parents(staging_parent, self._upload_root(canonical_id))

    @mounted_artifact_volume_operation
    def chunk_commits(
        self,
        dataset_id: str,
        start_request: Any,
        api_limits: Any,
    ) -> dict[int, dict[str, Any]]:
        """Reload and verify every currently published canonical chunk pair."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        start = validate_training_dataset_start_request(start_request, api_limits)
        self._reload_volume()
        return self._chunk_commits(canonical_id, start, api_limits)

    @mounted_artifact_volume_operation
    def chunk_receipts(
        self,
        dataset_id: str,
        start_request: Any,
        api_limits: Any,
    ) -> dict[int, dict[str, Any]]:
        """Read bounded durable receipt metadata without hashing chunk bodies."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        start = validate_training_dataset_start_request(start_request, api_limits)
        self._reload_volume()
        return self._chunk_receipts(canonical_id, start, api_limits)

    @mounted_artifact_volume_operation
    def chunk_receipt(
        self,
        dataset_id: str,
        chunk_index: int,
    ) -> dict[str, Any] | None:
        """Read one durable receipt and its body length for upload replay."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise ValueError("chunk_index must be a nonnegative integer")
        self._reload_volume()
        path = self._chunk_root(canonical_id, chunk_index)
        if not path.exists():
            return None
        receipt = self._read_chunk_receipt(path)
        self._require_chunk_metadata(path, receipt)
        return receipt

    @mounted_artifact_volume_operation
    def require_chunk(
        self,
        dataset_id: str,
        commit: Any,
    ) -> dict[str, Any]:
        """Require an already published immutable chunk pair without rewriting it."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        accepted = validate_training_dataset_chunk_commit(commit)
        self._reload_volume()
        path = self._chunk_root(canonical_id, accepted["chunk_index"])
        durable = self._read_chunk_receipt(path)
        if durable != accepted:
            raise ArtifactBytesCorrupt(
                "training chunk receipt differs from the accepted upload state"
            )
        self._require_chunk_pair(path, durable)
        return durable

    @mounted_artifact_volume_operation
    def copy_archive(
        self,
        dataset_id: str,
        start_request: Any,
        api_limits: Any,
        destination: str | Path,
    ) -> Path:
        """Copy one complete verified upload to a caller-owned local file."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        start = validate_training_dataset_start_request(start_request, api_limits)
        target = Path(destination)
        if target.exists() or not target.parent.is_dir():
            raise ValueError("archive destination must be a fresh local file path")

        self._reload_volume()
        commits = self._chunk_commits(canonical_id, start, api_limits)
        expected_count = _chunk_count(start, api_limits)
        if set(commits) != set(range(expected_count)):
            raise ValueError("training dataset upload is incomplete")

        whole_digest = hashlib.sha256()
        total = 0
        try:
            with target.open("xb") as output:
                for index in range(expected_count):
                    path = self._chunk_root(canonical_id, index) / "data.bin"
                    with path.open("rb") as source:
                        while block := source.read(_READ_BLOCK_BYTES):
                            output.write(block)
                            whole_digest.update(block)
                            total += len(block)
                _force_file(output)
            if total != start["total_bytes"]:
                raise ArtifactBytesCorrupt(
                    "assembled training archive has wrong length"
                )
            if whole_digest.hexdigest() != start["archive_bytes_digest"]:
                raise ArtifactBytesCorrupt(
                    "assembled training archive has wrong digest"
                )
            return target
        except Exception:
            if target.exists():
                target.unlink()
            raise

    @mounted_artifact_volume_operation
    def publish_dataset(
        self,
        dataset_id: str,
        extracted_root: str | Path,
        manifest: Any,
        api_limits: Any,
    ) -> dict[str, Any]:
        """First-win one already validated local dataset root and commit it."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        limits = validate_training_api_limits(api_limits)
        dataset = validate_training_dataset_manifest(manifest, limits)
        source = require_real_directory_outside(
            Path(extracted_root), self._root, "validated dataset staging root"
        )
        self._require_dataset_root(source, dataset, limits)

        self._reload_volume()
        target = self._dataset_root(canonical_id)
        if target.exists():
            self._require_dataset_root(target, dataset, limits)
            return dataset

        stage_parent = self._dataset_staging_root(canonical_id)
        stage = stage_parent / uuid.uuid4().hex
        try:
            stage.mkdir(parents=True)
            self._copy_dataset_root(source, stage, dataset)
            self._require_dataset_root(stage, dataset, limits)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, target)
            self._commit_volume()
            return dataset
        finally:
            if stage.exists():
                shutil.rmtree(stage)
            _remove_empty_parents(stage_parent, self._training_root())

    @mounted_artifact_volume_operation
    def require_dataset(
        self,
        dataset_id: str,
        manifest: Any,
        api_limits: Any,
    ) -> dict[str, Any]:
        """Reload and require one retained root to match its complete manifest."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        limits = validate_training_api_limits(api_limits)
        dataset = validate_training_dataset_manifest(manifest, limits)
        self._reload_volume()
        self._require_dataset_root(self._dataset_root(canonical_id), dataset, limits)
        return dataset

    @mounted_artifact_volume_operation
    def read_manifest(self, dataset_id: str, api_limits: Any) -> dict[str, Any]:
        """Reload and validate one retained canonical dataset.json."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        limits = validate_training_api_limits(api_limits)
        self._reload_volume()
        return self._read_dataset(canonical_id, limits).manifest

    @mounted_artifact_volume_operation
    def read_dataset(self, dataset_id: str, api_limits: Any) -> RetainedTrainingDataset:
        """Reload and return one fully verified retained dataset root."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        limits = validate_training_api_limits(api_limits)
        self._reload_volume()
        return self._read_dataset(canonical_id, limits)

    def _read_dataset(
        self, dataset_id: str, limits: Mapping[str, Any]
    ) -> RetainedTrainingDataset:
        """Read one retained root after the caller refreshed its Volume view."""

        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        root = self._dataset_root(canonical_id)
        path = root / "dataset.json"
        try:
            encoded = _read_bounded_regular_file(
                path, limits["dataset"]["max_json_member_bytes"]
            )
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(canonical_id) from error
        try:
            manifest = validate_training_dataset_manifest(parse_json(encoded), limits)
        except (UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt(
                "retained dataset manifest is invalid"
            ) from error
        if encoded != canonical_bytes(manifest):
            raise ArtifactBytesCorrupt("retained dataset manifest is not canonical")
        self._require_dataset_root(root, manifest, limits)
        return RetainedTrainingDataset(
            root=root,
            manifest=manifest,
            retained_bytes=len(encoded)
            + sum(item["size_bytes"] for item in manifest["files"]),
        )

    @mounted_artifact_volume_operation
    def delete_upload(self, dataset_id: str) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        self._reload_volume()
        root = self._upload_root(canonical_id)
        if root.exists():
            shutil.rmtree(root)
            self._commit_volume()

    @mounted_artifact_volume_operation
    def upload_ids(self, max_items: Any) -> list[str]:
        """Return a bounded set of canonical dataset upload directories."""

        self._reload_volume()
        return discover_training_artifact_ids(
            (self._training_root() / "uploads",),
            max_items,
            identity_field="dataset_id",
        )

    @mounted_artifact_volume_operation
    def delete_dataset(self, dataset_id: str) -> None:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        self._reload_volume()
        roots = [
            self._dataset_root(canonical_id),
            self._dataset_staging_root(canonical_id),
        ]
        changed = False
        for root in roots:
            if root.exists():
                shutil.rmtree(root)
                changed = True
        if changed:
            self._commit_volume()

    @mounted_artifact_volume_operation
    def dataset_remaining_bytes(self, dataset_id: str) -> int:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        self._reload_volume()
        return _tree_bytes(self._dataset_root(canonical_id))

    @mounted_artifact_volume_operation
    def upload_remaining_bytes(self, dataset_id: str) -> int:
        canonical_id = require_canonical_uuid(dataset_id, "dataset_id")
        self._reload_volume()
        return _tree_bytes(self._upload_root(canonical_id))

    def _chunk_commits(
        self,
        dataset_id: str,
        start: Mapping[str, Any],
        api_limits: Any,
    ) -> dict[int, dict[str, Any]]:
        receipts = self._chunk_receipts(dataset_id, start, api_limits)
        for index, receipt in receipts.items():
            self._require_chunk_pair(self._chunk_root(dataset_id, index), receipt)
        return receipts

    def _chunk_receipts(
        self,
        dataset_id: str,
        start: Mapping[str, Any],
        api_limits: Any,
    ) -> dict[int, dict[str, Any]]:
        root = self._upload_root(dataset_id) / "chunks"
        if not root.exists():
            return {}
        expected_count = _chunk_count(start, api_limits)
        result = {}
        for path in root.iterdir():
            if not path.is_dir() or path.is_symlink():
                raise ArtifactBytesCorrupt(
                    "training upload contains an invalid chunk path"
                )
            if not path.name.isascii() or not path.name.isdecimal():
                raise ArtifactBytesCorrupt(
                    "training upload contains an invalid chunk index"
                )
            index = int(path.name)
            if str(index) != path.name or index >= expected_count:
                raise ArtifactBytesCorrupt(
                    "training upload chunk index is outside its plan"
                )
            receipt = self._read_chunk_receipt(path)
            expected = training_dataset_chunk_range(start, index, api_limits)
            if (
                receipt["start_request_digest"]
                != training_dataset_start_digest(start, api_limits)
                or receipt["chunk_index"] != index
                or receipt["size_bytes"] != expected["size_bytes"]
            ):
                raise ArtifactBytesCorrupt(
                    "training upload receipt does not match its reservation"
                )
            self._require_chunk_metadata(path, receipt)
            result[index] = receipt
        return result

    @staticmethod
    def _read_chunk_receipt(path: Path) -> dict[str, Any]:
        receipt_path = path / "receipt.json"
        try:
            encoded = receipt_path.read_bytes()
            receipt = validate_training_dataset_chunk_commit(parse_json(encoded))
        except (FileNotFoundError, UnicodeDecodeError, ValueError) as error:
            raise ArtifactBytesCorrupt("training chunk receipt is invalid") from error
        if encoded != canonical_bytes(receipt):
            raise ArtifactBytesCorrupt("training chunk receipt is not canonical")
        return receipt

    @staticmethod
    def _require_chunk_pair(path: Path, receipt: Mapping[str, Any]) -> None:
        TrainingDatasetFiles._require_chunk_metadata(path, receipt)
        data = path / "data.bin"
        size, digest = _stream_identity(data)
        if size != receipt["size_bytes"] or digest != receipt["sha256"]:
            raise ArtifactBytesCorrupt("training chunk bytes do not match receipt")

    @staticmethod
    def _require_chunk_metadata(path: Path, receipt: Mapping[str, Any]) -> None:
        expected = {"data.bin", "receipt.json"}
        try:
            names = {item.name for item in path.iterdir()}
        except FileNotFoundError as error:
            raise ArtifactBytesMissing(path.name) from error
        if names != expected:
            raise ArtifactBytesCorrupt("training chunk directory is incomplete")
        data = path / "data.bin"
        if data.is_symlink() or not data.is_file():
            raise ArtifactBytesCorrupt("training chunk data is not a regular file")
        if data.stat().st_size != receipt["size_bytes"]:
            raise ArtifactBytesCorrupt("training chunk length does not match receipt")

    @staticmethod
    def _require_dataset_root(
        root: Path,
        manifest: Mapping[str, Any],
        api_limits: Any,
    ) -> None:
        limits = validate_training_api_limits(api_limits)
        expected_paths = ("dataset.json", *(item["path"] for item in manifest["files"]))
        try:
            require_file_closure(root, expected_paths)
            encoded_manifest = _read_bounded_regular_file(
                root / "dataset.json",
                limits["dataset"]["max_json_member_bytes"],
            )
        except ArtifactBytesCorrupt:
            raise
        except (OSError, ValueError) as error:
            raise ArtifactBytesCorrupt(
                "retained dataset file closure is invalid"
            ) from error
        if encoded_manifest != canonical_bytes(manifest):
            raise ArtifactBytesCorrupt("retained dataset manifest bytes differ")
        for item in manifest["files"]:
            identity = {
                "relative_path": item["path"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            try:
                require_exact_file(
                    root.joinpath(*PurePosixPath(item["path"]).parts),
                    identity,
                    item["path"],
                )
            except (OSError, ValueError) as error:
                raise ArtifactBytesCorrupt(
                    f"retained dataset member is invalid: {item['path']}"
                ) from error
        validate_training_dataset_manifest(manifest, limits)

    @staticmethod
    def _copy_dataset_root(
        source: Path,
        target: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        manifest_bytes = canonical_bytes(manifest)
        TrainingDatasetFiles._write_file(target / "dataset.json", manifest_bytes)
        for item in manifest["files"]:
            relative = PurePosixPath(item["path"])
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_exact_file(
                source.joinpath(*relative.parts),
                destination,
                {
                    "relative_path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                },
            )

    @staticmethod
    def _write_file(path: Path, value: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(value)
            _force_file(stream)

    def _training_root(self) -> Path:
        return self._root / "training"

    def _upload_root(self, dataset_id: str) -> Path:
        return self._training_root() / "uploads" / dataset_id

    def _chunk_root(self, dataset_id: str, chunk_index: int) -> Path:
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise ValueError("chunk_index must be a nonnegative integer")
        return self._upload_root(dataset_id) / "chunks" / str(chunk_index)

    def _dataset_root(self, dataset_id: str) -> Path:
        return self._training_root() / "datasets" / dataset_id

    def _dataset_staging_root(self, dataset_id: str) -> Path:
        return self._training_root() / "dataset-staging" / dataset_id


def _chunk_count(start: Mapping[str, Any], api_limits: Any) -> int:
    last = training_dataset_chunk_range(
        start,
        (start["total_bytes"] - 1) // start["chunk_size_bytes"],
        api_limits,
    )
    return last["chunk_index"] + 1


def _chunk_identity(value: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        value["start_request_digest"],
        value["chunk_index"],
        value["size_bytes"],
        value["sha256"],
    )


def _force_file(stream: Any) -> None:
    stream.flush()
    os.fsync(stream.fileno())


def _stream_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(_READ_BLOCK_BYTES):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _read_bounded_regular_file(path: Path, maximum_bytes: int) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ArtifactBytesCorrupt("training dataset manifest is not a regular file")
    with path.open("rb") as stream:
        value = stream.read(maximum_bytes + 1)
        if len(value) > maximum_bytes or stream.read(1):
            raise ArtifactBytesCorrupt(
                "training dataset manifest exceeds max_json_member_bytes"
            )
        return value


def _tree_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for directory, names, files in os.walk(root, followlinks=False):
        path = Path(directory)
        if path.is_symlink() or any((path / name).is_symlink() for name in names):
            raise ArtifactBytesCorrupt("training dataset tree contains a link")
        for name in files:
            file = path / name
            if file.is_symlink() or not file.is_file():
                raise ArtifactBytesCorrupt("training dataset tree contains a non-file")
            total += file.stat().st_size
    return total


def _remove_empty_parents(path: Path, stop: Path) -> None:
    current = path
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
