"""Permanent gateway identity and a disposable copy of committed state."""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    parse_json,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_sha256,
)
from histoforge.cloud import _auth
from histoforge.cloud.artifacts.volume import mounted_artifact_volume_operation

GATEWAY_STATE_PATH = "gateway-state/v1.json"
MAX_GATEWAY_STATE_BYTES = 4096
_LOG = logging.getLogger(__name__)


def parse_record_bytes(encoded: bytes) -> dict[str, Any]:
    """Require one bounded, canonical record with the existing identity fields."""

    if len(encoded) > MAX_GATEWAY_STATE_BYTES:
        raise ValueError("the cloud connection record is too large")
    record = _auth.parse_gateway_state(parse_json(encoded))
    if canonical_bytes(record) != encoded:
        raise ValueError("the cloud connection record is not canonical")
    return record


def cached_state(cache: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a validated committed copy; malformed copies require a refill."""

    value = cache.get(_auth.STATE_KEY)
    if value is None:
        return None
    try:
        return _auth.parse_gateway_state(value)
    except ValueError:
        return None


class GatewayStateStore:
    """Own record replacement and cache publication inside the serialized lane.

    Only this owner publishes the cache, after commit or successful reload. The
    cache is invalidated before a record write. A failed write may later commit;
    the next cache miss must reload rather than assume a rollback.
    """

    def __init__(
        self,
        root: str | Path,
        cache: MutableMapping[str, Any],
        *,
        reload_volume: Callable[[], None],
        commit_volume: Callable[[], None],
    ) -> None:
        self._path = Path(root) / GATEWAY_STATE_PATH
        self._cache = cache
        self._reload = reload_volume
        self._commit = commit_volume

    @mounted_artifact_volume_operation
    def load_current(self) -> dict[str, Any] | None:
        current = cached_state(self._cache)
        if current is not None:
            return current
        self._reload()
        current = self._read_record()
        if current is None:
            self._invalidate_cache()
            return None
        self._publish_cache(current)
        return current

    @mounted_artifact_volume_operation
    def rotate(
        self,
        token_sha256: str,
        expected_deployment_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        verifier = require_sha256(token_sha256, "token_sha256")
        if expected_deployment_id is not None:
            require_nonblank_string(expected_deployment_id, "expected_deployment_id")
        if expected_revision is not None:
            require_nonnegative_safe_integer(expected_revision, "expected_revision")
        self._reload()
        current = self._read_record()
        if current is None:
            self._invalidate_cache()
            raise ValueError(
                "The cloud backend is missing its saved connection record. "
                "Reconnect cannot create a replacement deployment identity."
            )
        if (
            expected_deployment_id is not None
            and current["deployment_id"] != expected_deployment_id
        ) or (
            expected_revision is not None
            and current["auth_revision"] != expected_revision
        ):
            raise ValueError("gateway access changed during update preparation")
        replacement = _auth.next_state(current, verifier)
        encoded = canonical_bytes(replacement)
        parse_record_bytes(encoded)
        self._invalidate_cache()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        stage = self._path.with_name(".v1.json.staging")
        try:
            if stage.is_symlink():
                raise ValueError("the cloud connection staging file is invalid")
            with stage.open("wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(stage, self._path)
            self._commit()
        finally:
            stage.unlink(missing_ok=True)
        self._publish_cache(replacement)
        return replacement

    def _read_record(self) -> dict[str, Any] | None:
        try:
            if self._path.is_symlink():
                raise ValueError("the cloud connection record is not a regular file")
            status = self._path.stat()
            if not stat.S_ISREG(status.st_mode):
                raise ValueError("the cloud connection record is not a regular file")
            if status.st_size > MAX_GATEWAY_STATE_BYTES:
                raise ValueError("the cloud connection record is too large")
            return parse_record_bytes(self._path.read_bytes())
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeDecodeError):
            self._invalidate_cache()
            raise ValueError(
                "The cloud backend's saved connection record is invalid. "
                "It requires explicit recovery."
            ) from None

    def _invalidate_cache(self) -> None:
        self._cache.pop(_auth.STATE_KEY, None)

    def _publish_cache(self, state: dict[str, Any]) -> None:
        try:
            self._cache[_auth.STATE_KEY] = state
        except Exception:
            # Persistence already succeeded. A later miss can rebuild this copy.
            _LOG.warning("could not cache committed gateway state", exc_info=True)
