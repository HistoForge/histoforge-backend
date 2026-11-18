"""Canonical Artifact Volume control records for managed training."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_canonical_uuid,
    require_sha256,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation
from histoforge.cloud.training.datasets.transfer import (
    validate_training_dataset_cleanup_record,
    validate_training_dataset_record,
    validate_training_dataset_upload_record,
)
from histoforge.cloud.training.limits import (
    validate_training_api_limits,
    validate_training_service_limits,
)
from histoforge.cloud.training.runs.contract import (
    validate_training_run_admission_record,
    validate_training_run_control_record,
)

_RECORD_SUFFIX = ".json"
_STAGING_SUFFIX = ".staging"
_NAMESPACES = {"datasets", "admissions", "runs"}


class TrainingControlRecords:
    """Read and atomically replace every managed-training control record."""

    def __init__(
        self,
        root: str | Path,
        api_limits: Any,
        service_limits: Any,
        *,
        reload_volume: Callable[[], None] = lambda: None,
        commit_volume: Callable[[], None] = lambda: None,
    ) -> None:
        self._root = Path(root)
        self._api_limits = validate_training_api_limits(api_limits)
        self._service_limits = validate_training_service_limits(service_limits)
        self._reload_volume = reload_volume
        self._commit_volume = commit_volume

    @mounted_artifact_volume_operation
    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        key = require_canonical_uuid(dataset_id, "dataset_id")
        self._reload_volume()
        return self._read("datasets", key, self._validate_dataset, _dataset_key)

    @mounted_artifact_volume_operation
    def datasets(self) -> list[dict[str, Any]]:
        self._reload_volume()
        return self._read_all("datasets", self._validate_dataset, _dataset_key)

    @mounted_artifact_volume_operation
    def reserve_dataset(self, record: Any) -> tuple[dict[str, Any], bool]:
        normalized = self._validate_dataset(record)
        return self._reserve("datasets", normalized, normalized["dataset_id"])

    @mounted_artifact_volume_operation
    def replace_dataset(self, record: Any) -> dict[str, Any]:
        normalized = self._validate_dataset(record)
        return self._replace("datasets", normalized, normalized["dataset_id"])

    @mounted_artifact_volume_operation
    def remove_dataset(self, dataset_id: str) -> None:
        key = require_canonical_uuid(dataset_id, "dataset_id")
        self._reload_volume()
        self._remove("datasets", key)

    @mounted_artifact_volume_operation
    def get_admission(self, request_digest: str) -> dict[str, Any] | None:
        key = require_sha256(request_digest, "request_digest")
        self._reload_volume()
        return self._read(
            "admissions", key, validate_training_run_admission_record, _admission_key
        )

    @mounted_artifact_volume_operation
    def admissions(self) -> list[dict[str, Any]]:
        self._reload_volume()
        return self._read_all(
            "admissions", validate_training_run_admission_record, _admission_key
        )

    @mounted_artifact_volume_operation
    def reserve_admission(self, record: Any) -> tuple[dict[str, Any], bool]:
        normalized = validate_training_run_admission_record(record)
        return self._reserve("admissions", normalized, normalized["request_digest"])

    @mounted_artifact_volume_operation
    def remove_admission(self, request_digest: str) -> None:
        key = require_sha256(request_digest, "request_digest")
        self._reload_volume()
        self._remove("admissions", key)

    @mounted_artifact_volume_operation
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        key = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        return self._read("runs", key, self._validate_run, _run_key)

    @mounted_artifact_volume_operation
    def runs(self) -> list[dict[str, Any]]:
        self._reload_volume()
        return self._read_all("runs", self._validate_run, _run_key)

    @mounted_artifact_volume_operation
    def replace_run(self, record: Any) -> dict[str, Any]:
        normalized = self._validate_run(record)
        return self._replace("runs", normalized, normalized["run"]["run_id"])

    @mounted_artifact_volume_operation
    def accept_run(self, request_digest: str, record: Any) -> dict[str, Any]:
        """Replace one matching admission with its accepted run in one commit."""

        digest = require_sha256(request_digest, "request_digest")
        normalized = self._validate_run(record)
        run_id = normalized["run"]["run_id"]
        if normalized["run"]["request_digest"] != digest:
            raise ValueError("accepted run request digest differs from admission")

        self._reload_volume()
        admission = self._read(
            "admissions", digest, validate_training_run_admission_record, _admission_key
        )
        if admission is None or admission["run_id"] != run_id:
            raise self._invalid(run_id)
        if self._read("runs", run_id, self._validate_run, _run_key) is not None:
            raise self._invalid(run_id)

        encoded = canonical_bytes(normalized)
        admission_path = self._record_path("admissions", digest)
        run_path = self._record_path("runs", run_id)
        stage = self._stage_path("runs", run_id)
        self._require_control_capacity(0, len(encoded) - admission_path.stat().st_size)
        run_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_fsynced(stage, encoded)
            os.replace(stage, run_path)
            admission_path.unlink()
            self._commit_volume()
        finally:
            _remove_if_present(stage)
        return normalized

    @mounted_artifact_volume_operation
    def remove_run(self, run_id: str) -> None:
        key = require_canonical_uuid(run_id, "run_id")
        self._reload_volume()
        self._remove("runs", key)

    def _reserve(
        self, namespace: str, record: dict[str, Any], key: str
    ) -> tuple[dict[str, Any], bool]:
        self._reload_volume()
        current = self._read(
            namespace, key, self._validator(namespace), self._key(namespace)
        )
        if current is not None:
            return current, False
        self._write(namespace, key, record, replacing=False)
        return record, True

    def _replace(
        self, namespace: str, record: dict[str, Any], key: str
    ) -> dict[str, Any]:
        self._reload_volume()
        if (
            self._read(namespace, key, self._validator(namespace), self._key(namespace))
            is None
        ):
            raise self._invalid(key)
        self._write(namespace, key, record, replacing=True)
        return record

    def _remove(self, namespace: str, key: str) -> None:
        path = self._record_path(namespace, key)
        stage = self._stage_path(namespace, key)
        changed = False
        for candidate in (path, stage):
            try:
                candidate.unlink()
                changed = True
            except FileNotFoundError:
                pass
        if changed:
            self._commit_volume()

    def _read_all(
        self,
        namespace: str,
        validator: Callable[[Any], dict[str, Any]],
        key_of: Callable[[Mapping[str, Any]], str],
    ) -> list[dict[str, Any]]:
        root = self._records_root(namespace)
        if not root.exists():
            return []
        if root.is_symlink() or not root.is_dir():
            raise self._invalid(namespace)

        records = []
        for path in sorted(root.iterdir(), key=lambda item: item.name):
            if path.name.endswith(_STAGING_SUFFIX):
                if path.is_symlink() or not path.is_file():
                    raise self._invalid(namespace)
                continue
            if path.is_symlink() or not path.is_file() or path.suffix != _RECORD_SUFFIX:
                raise self._invalid(namespace)
            key = path.stem
            record = self._read_file(path, key, validator, key_of)
            records.append(record)
        return records

    def _read(
        self,
        namespace: str,
        key: str,
        validator: Callable[[Any], dict[str, Any]],
        key_of: Callable[[Mapping[str, Any]], str],
    ) -> dict[str, Any] | None:
        path = self._record_path(namespace, key)
        try:
            return self._read_file(path, key, validator, key_of)
        except FileNotFoundError:
            return None

    def _read_file(
        self,
        path: Path,
        expected_key: str,
        validator: Callable[[Any], dict[str, Any]],
        key_of: Callable[[Mapping[str, Any]], str],
    ) -> dict[str, Any]:
        try:
            if not path.exists():
                raise FileNotFoundError(path)
            if path.is_symlink() or not path.is_file():
                raise ValueError("control record is not a regular file")
            encoded = path.read_bytes()
            record = validator(parse_json(encoded))
            if key_of(record) != expected_key:
                raise ValueError("control record key differs from its value")
            if encoded != canonical_bytes(record):
                raise ValueError("control record is not canonical")
            return record
        except FileNotFoundError:
            raise
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise self._invalid(expected_key) from error

    def _write(
        self,
        namespace: str,
        key: str,
        record: Mapping[str, Any],
        *,
        replacing: bool,
    ) -> None:
        encoded = canonical_bytes(record)
        path = self._record_path(namespace, key)
        if not replacing:
            self._require_control_capacity(1, len(encoded))

        path.parent.mkdir(parents=True, exist_ok=True)
        stage = self._stage_path(namespace, key)
        try:
            _write_fsynced(stage, encoded)
            os.replace(stage, path)
            self._commit_volume()
        finally:
            _remove_if_present(stage)

    def _require_control_capacity(self, added_records: int, added_bytes: int) -> None:
        record_count = 0
        stored_bytes = 0
        root = self._control_root()
        if root.exists():
            if root.is_symlink() or not root.is_dir():
                raise self._invalid("training-control")
            for child in root.iterdir():
                if (
                    child.name not in _NAMESPACES
                    or child.is_symlink()
                    or not child.is_dir()
                ):
                    raise self._invalid("training-control")
            for directory, names, files in os.walk(root, followlinks=False):
                path = Path(directory)
                if path.is_symlink() or any(
                    (path / name).is_symlink() for name in names
                ):
                    raise self._invalid("training-control")
                for name in files:
                    file = path / name
                    if file.is_symlink() or not file.is_file():
                        raise self._invalid("training-control")
                    if name.endswith(_STAGING_SUFFIX):
                        continue
                    if not name.endswith(_RECORD_SUFFIX):
                        raise self._invalid("training-control")
                    stored_bytes += file.stat().st_size
                    record_count += 1
        projected_records = record_count + added_records
        projected_bytes = stored_bytes + added_bytes
        if (
            projected_records > self._service_limits["max_control_records"]
            and projected_records > record_count
        ) or (
            projected_bytes > self._service_limits["max_control_stored_bytes"]
            and projected_bytes > stored_bytes
        ):
            raise self._capacity()

    def _validate_dataset(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("training dataset record must be an object")
        kind = value.get("type")
        if kind == "uploading":
            return validate_training_dataset_upload_record(value, self._api_limits)
        if kind == "retained":
            return validate_training_dataset_record(value, self._api_limits)
        if kind == "cleanup-required":
            return validate_training_dataset_cleanup_record(value, self._api_limits)
        raise ValueError("unsupported training dataset record type")

    def _validate_run(self, value: Any) -> dict[str, Any]:
        return validate_training_run_control_record(value, self._api_limits)

    def _validator(self, namespace: str) -> Callable[[Any], dict[str, Any]]:
        if namespace == "datasets":
            return self._validate_dataset
        if namespace == "admissions":
            return validate_training_run_admission_record
        if namespace == "runs":
            return self._validate_run
        raise ValueError("unsupported training control namespace")

    @staticmethod
    def _key(namespace: str) -> Callable[[Mapping[str, Any]], str]:
        if namespace == "datasets":
            return _dataset_key
        if namespace == "admissions":
            return _admission_key
        if namespace == "runs":
            return _run_key
        raise ValueError("unsupported training control namespace")

    def _control_root(self) -> Path:
        return self._root / "training" / "control" / "v1"

    def _records_root(self, namespace: str) -> Path:
        if namespace not in _NAMESPACES:
            raise ValueError("unsupported training control namespace")
        return self._control_root() / namespace

    def _record_path(self, namespace: str, key: str) -> Path:
        return self._records_root(namespace) / f"{key}{_RECORD_SUFFIX}"

    def _stage_path(self, namespace: str, key: str) -> Path:
        return self._records_root(namespace) / f".{key}{_STAGING_SUFFIX}"

    @staticmethod
    def _capacity() -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.RATE_LIMITED,
            "the training control store has reached its configured capacity",
        )

    @staticmethod
    def _invalid(resource_id: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.INTERNAL_ERROR,
            "the training control store contains invalid state",
            resource_id=resource_id,
        )


def _dataset_key(value: Mapping[str, Any]) -> str:
    return require_canonical_uuid(value["dataset_id"], "dataset_id")


def _admission_key(value: Mapping[str, Any]) -> str:
    return require_sha256(value["request_digest"], "request_digest")


def _run_key(value: Mapping[str, Any]) -> str:
    return require_canonical_uuid(value["run"]["run_id"], "run_id")


def _write_fsynced(path: Path, encoded: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
