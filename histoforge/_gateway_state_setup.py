"""Bootstrap permanent identity and migrate a deployed Dict-era backend.

This administrative path runs on the laptop after Modal login. Once the new
state class exists, only that class may replace the record or publish its cache.
"""

from __future__ import annotations

import io
from typing import Any

import modal

from histoforge.canonical_json import canonical_bytes
from histoforge.cloud import _auth
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.gateway_state import (
    GATEWAY_STATE_PATH,
    MAX_GATEWAY_STATE_BYTES,
    parse_record_bytes,
)


class GatewayStateSetup:
    """Own the external Volume API and the limited Dict-era migration bridge."""

    def __init__(self, *, fresh: bool = False) -> None:
        self._volume = modal.Volume.from_name(
            DEPLOYMENT_CONFIG.artifact_volume_name,
            create_if_missing=fresh,
            version=2,
        )
        self._legacy = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.legacy_gateway_state_dict_name,
            create_if_missing=False,
        )
        self._cache = modal.Dict.from_name(
            DEPLOYMENT_CONFIG.gateway_state_cache_dict_name,
            create_if_missing=True,
        )

    def legacy_current(self) -> dict[str, Any] | None:
        try:
            return _auth.legacy_current_state(self._legacy)
        except modal.exception.NotFoundError:
            return None

    def legacy_rotate(
        self, *, expected_deployment_id: str | None = None
    ) -> tuple[dict[str, Any], str]:
        """Reconnect an old deployment so an interrupted cutover can be retried."""

        return _auth.legacy_rotate(
            self._legacy, expected_deployment_id=expected_deployment_id
        )

    def bootstrap(self) -> None:
        """Create only if absent; a deployment retry never replaces its identity."""

        legacy = self.legacy_current()
        existing = self.read_record()
        self._require_bootstrap_identity(legacy, existing)
        self._cache.pop(_auth.STATE_KEY, None)
        if existing is None:
            candidate = legacy if legacy is not None else _auth.initial_state()
            try:
                self._upload(candidate, force=False)
            except FileExistsError:
                # A concurrent setup won. Its record, not our candidate, is the
                # authority. The deployed class validates it before issuing a token.
                candidate = None
            observed = self.read_record()
            if observed is None or (candidate is not None and observed != candidate):
                raise ValueError(
                    "the uploaded cloud connection record was not verified"
                )
            self._require_bootstrap_identity(legacy, observed)

    def read_record(self) -> dict[str, Any] | None:
        """Read committed bytes through the external API, without mounting files."""

        encoded = bytearray()
        try:
            for chunk in self._volume.read_file(GATEWAY_STATE_PATH):
                encoded.extend(chunk)
                if len(encoded) > MAX_GATEWAY_STATE_BYTES:
                    raise ValueError("the cloud connection record is too large")
        except (FileNotFoundError, modal.exception.NotFoundError):
            return None
        return parse_record_bytes(bytes(encoded))

    def prepare_cutover(self, previous: dict[str, Any]) -> dict[str, Any] | None:
        """Reject a corrupt or foreign staged identity before stopping old work."""

        staged = self.read_record()
        self._require_same_identity(previous, staged)
        return staged

    def cut_over(
        self,
        previous: dict[str, Any],
        staged: dict[str, Any] | None,
        token_sha256: str,
    ) -> dict[str, Any]:
        """Persist a newer same-identity record, then fence the old deployment."""

        if self.legacy_current() != previous or self.read_record() != staged:
            raise ValueError("gateway access changed during update preparation")
        base = dict(previous)
        if staged is not None:
            base["auth_revision"] = max(
                previous["auth_revision"], staged["auth_revision"]
            )
        replacement = _auth.next_state(base, token_sha256)
        self._cache.pop(_auth.STATE_KEY, None)
        # Overwrite is allowed only for this explicit migration, after proving
        # the staged record belongs to the same deployment. Never lower revision.
        self._upload(replacement, force=staged is not None)
        if self.read_record() != replacement:
            raise ValueError("the uploaded cloud connection record was not verified")
        # The old Dict remains authoritative until recreate. Fence its workers
        # only after readback, so a failed upload leaves current access usable.
        self._legacy[_auth.STATE_KEY] = replacement
        return replacement

    def _upload(self, state: dict[str, Any], *, force: bool) -> None:
        encoded = canonical_bytes(_auth.parse_gateway_state(state))
        parse_record_bytes(encoded)
        with self._volume.batch_upload(force=force) as upload:
            upload.put_file(io.BytesIO(encoded), "/" + GATEWAY_STATE_PATH)

    @staticmethod
    def _require_same_identity(
        legacy: dict[str, Any] | None, staged: dict[str, Any] | None
    ) -> None:
        if (
            legacy is not None
            and staged is not None
            and legacy["deployment_id"] != staged["deployment_id"]
        ):
            raise ValueError(
                "The saved cloud connection record belongs to a different "
                "deployment. Explicit recovery is required; nothing was replaced."
            )

    @classmethod
    def _require_bootstrap_identity(
        cls, legacy: dict[str, Any] | None, staged: dict[str, Any] | None
    ) -> None:
        cls._require_same_identity(legacy, staged)
        if (
            legacy is not None
            and staged is not None
            and (staged["auth_revision"] < legacy["auth_revision"])
        ):
            raise ValueError(
                "The saved cloud connection record is older than the legacy "
                "credential generation. Explicit recovery is required; "
                "nothing was replaced."
            )
