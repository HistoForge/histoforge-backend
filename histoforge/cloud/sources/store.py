"""Source-artifact admission, upload, seal, status, and retention policy.

``SourceArtifactStore`` owns lifecycle decisions. ``SourceArtifactRecords``
stores one detached snapshot per image; ``SourceFileStore`` publishes its bytes.
The existing artifact lifecycle lane serializes control-state changes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from histoforge.canonical_json import (
    canonical_bytes,
    canonical_tree,
    require_canonical_uuid,
    require_positive_safe_integer,
)
from histoforge.cloud._config import (
    validate_gateway_limits,
    validate_source_artifact_limits,
)
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud._time import iso_z
from histoforge.cloud.artifacts.store import (
    ArtifactBytesCorrupt,
    ArtifactBytesMissing,
)
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.sources.contract import (
    canonical_chunk_count,
    canonical_chunk_range,
    source_artifact_start_digest,
    validate_source_artifact_acceptance,
    validate_source_artifact_start_request,
)
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.publication import (
    build_source_artifact_ref,
    validate_source_artifact_publication,
    validate_source_artifact_ref,
    validate_source_artifact_reference,
    validate_source_artifact_seal_request,
    validate_source_artifact_sealed,
    validate_source_byte_chunk_accepted,
)
from histoforge.cloud.sources.record_validation import invalid_source_record
from histoforge.cloud.sources.records import SourceArtifactRecords
from histoforge.cloud.sources.status import (
    validate_source_artifact_status_projection,
)


class SourceArtifactStore:
    """Coordinate source control state with immutable Artifact-Volume bytes."""

    def __init__(
        self,
        records: Any,
        source_limits: Any,
        gateway_limits: Any,
        *,
        max_active_source_artifact_peak_bytes: int,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._records = SourceArtifactRecords(records)
        self._source_limits = validate_source_artifact_limits(source_limits)
        self._gateway_limits = validate_gateway_limits(gateway_limits)
        self._max_active_source_artifact_peak_bytes = require_positive_safe_integer(
            max_active_source_artifact_peak_bytes,
            "max_active_source_artifact_peak_bytes",
        )
        self._clock = clock

    def reserve(
        self,
        start_request: Any,
        *,
        files: SourceFileStore | None = None,
    ) -> dict[str, Any]:
        """Reserve one ID or return its exact idempotent acceptance.

        The production caller serializes this short deployment-wide capacity
        decision. Dict put-if-absent remains the final same-ID identity fence.
        """

        start = validate_source_artifact_start_request(start_request)
        self._validate_start_limits(start)
        artifact_id = start["artifact_id"]
        digest = source_artifact_start_digest(start)
        now = self._now()

        current = self._records.load(artifact_id)
        if current is not None:
            self._require_live(current, now)
            self._require_same_start(current["reservation"], digest)
            return self._accepted(current["reservation"])

        if files is not None:
            # Control changes are serialized, and this ID was absent above.
            # Every reservation retained by this cleanup is therefore live.
            active = self._cleanup_expired(files, now=now, exclude=artifact_id).values()
        else:
            active = []
            for _, reservation, expired in self._retention_decisions(now):
                if isinstance(expired, GatewayFailure):
                    raise expired
                if not expired:
                    active.append(reservation)
        if len(active) >= self._gateway_limits["max_active_source_artifacts"]:
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "the deployment has reached its active source-artifact limit",
            )
        planned_bytes = sum(
            self._peak_bytes(record["start_request"]) for record in active
        ) + self._peak_bytes(start)
        if planned_bytes > self._max_active_source_artifact_peak_bytes:
            raise GatewayFailure(
                GatewayErrorCode.RATE_LIMITED,
                "the deployment has reached its active source-artifact peak-byte limit",
            )

        record = canonical_tree(
            {
                "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                "start_request": start,
                "start_request_digest": digest,
                "accepted_at": iso_z(now),
                "absolute_expires_at": iso_z(
                    now
                    + timedelta(
                        seconds=self._source_limits["absolute_lifetime_seconds"]
                    )
                ),
            }
        )
        winner, _ = self._records.reserve(record)
        self._require_live(winner, now)
        self._require_same_start(winner["reservation"], digest)
        return self._accepted(winner["reservation"])

    def chunk_range(self, artifact_id: str, chunk_id: int) -> dict[str, int]:
        """Return one live reservation's exact expected raw-body range."""

        start = self._live_source(artifact_id)["reservation"]["start_request"]
        return self._chunk_range(start, chunk_id)

    @staticmethod
    def _chunk_range(start: Mapping[str, Any], chunk_id: int) -> dict[str, int]:
        try:
            return canonical_chunk_range(
                start["total_bytes"], start["chunk_size_bytes"], chunk_id
            )
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                str(error),
                resource_id=start["artifact_id"],
            ) from error

    def accept_chunk(
        self,
        artifact_id: str,
        chunk_id: int,
        body: bytes,
        files: SourceFileStore,
        *,
        authorize: Callable[[], None] = lambda: None,
        control: Any | None = None,
    ) -> dict[str, Any]:
        """Claim one range, publish bytes, then commit through the control owner."""

        digest = hashlib.sha256(body).hexdigest()
        if control is None:
            claimed = self.claim_chunk(
                artifact_id,
                chunk_id,
                len(body),
                digest,
                authorize=authorize,
            )
        else:
            claimed = control.claim_chunk(artifact_id, chunk_id, len(body), digest)
        if "accepted" in claimed:
            return claimed["accepted"]

        try:
            files.publish_source_chunk(
                artifact_id,
                chunk_id,
                body,
                digest,
                require_existing=claimed["committed"],
            )
        except (ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
            raise self._service_loss(artifact_id) from error

        if control is None:
            return self.commit_chunk(
                artifact_id,
                chunk_id,
                digest,
                claimed["start_request_digest"],
                authorize=authorize,
            )
        return control.commit_chunk(
            artifact_id,
            chunk_id,
            digest,
            claimed["start_request_digest"],
        )

    def claim_chunk(
        self,
        artifact_id: str,
        chunk_id: int,
        body_size: int,
        digest: str,
        *,
        authorize: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Remember the first digest before writing a received chunk's bytes."""

        source = self._live_source(artifact_id)
        record = source["reservation"]
        start = record["start_request"]
        expected = self._chunk_range(start, chunk_id)
        if body_size != expected["size"]:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "chunk body length does not match its canonical range",
                resource_id=start["artifact_id"],
            )
        authorize()
        existing = self._records.chunk_commit(source, chunk_id)
        sealed = self._records.sealed(source)
        if sealed is not None:
            if existing is None:
                raise invalid_source_record(start["artifact_id"])
            self._require_same_chunk(existing, digest)
            return {
                "start_request_digest": record["start_request_digest"],
                "accepted": self._chunk_accepted(existing),
            }

        winner = self._records.claim_chunk(
            source,
            canonical_tree(
                {
                    "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                    "artifact_id": start["artifact_id"],
                    "chunk_id": chunk_id,
                    "sha256": digest,
                    "claimed_at": iso_z(self._now()),
                }
            ),
        )
        if winner["chunk_id"] != chunk_id or winner["sha256"] != digest:
            raise self._conflict(start["artifact_id"], "chunk bytes differ")
        self._require_live(source, self._now())
        authorize()
        return {
            "start_request_digest": record["start_request_digest"],
            "committed": existing is not None,
        }

    def commit_chunk(
        self,
        artifact_id: str,
        chunk_id: int,
        digest: str,
        start_request_digest: str,
        *,
        authorize: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Acknowledge a chunk after its bytes have been durably published."""

        source = self._live_source(artifact_id)
        current = source["reservation"]
        self._require_same_start(current, start_request_digest)
        authorize()
        # A delayed identical upload may finish after sealing removed claims.
        # Its durable acknowledgement remains authoritative, not the old claim.
        sealed = self._records.sealed(source)
        if sealed is not None:
            existing = self._records.chunk_commit(source, chunk_id)
            if existing is None:
                raise invalid_source_record(artifact_id)
            self._require_same_chunk(existing, digest)
            return self._chunk_accepted(existing)

        claim = self._records.chunk_claim(source, chunk_id)
        if claim is None or claim["sha256"] != digest:
            raise invalid_source_record(artifact_id)

        chunk_record = self._records.commit_chunk(
            source,
            canonical_tree(
                {
                    "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                    "artifact_id": artifact_id,
                    "chunk": {"chunk_id": chunk_id, "sha256": digest},
                    "accepted_at": iso_z(self._now()),
                }
            ),
        )
        self._require_same_chunk(chunk_record, digest)
        authorize()
        return self._chunk_accepted(chunk_record)

    def seal(
        self,
        artifact_id: str,
        seal_request: Any,
        files: SourceFileStore,
        *,
        authorize: Callable[[], None] = lambda: None,
        control: Any | None = None,
    ) -> dict[str, Any]:
        """Claim a seal, publish bytes, then commit through the control owner."""

        request = validate_source_artifact_seal_request(seal_request)
        if control is None:
            claimed = self.claim_seal(
                artifact_id,
                request,
                authorize=authorize,
            )
        else:
            claimed = control.claim_seal(artifact_id, request)
        if "sealed" in claimed:
            return claimed["sealed"]

        try:
            manifest = files.seal_source(
                claimed["start_request"],
                request,
                claimed["chunk_digests"],
                self._source_limits,
            )
        except (ArtifactBytesCorrupt, ArtifactBytesMissing) as error:
            raise self._service_loss(artifact_id) from error
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                f"source validation failed: {error}",
                resource_id=artifact_id,
            ) from error

        if control is None:
            sealed = self.commit_seal(
                artifact_id,
                request,
                claimed["start_request_digest"],
                manifest,
                authorize=authorize,
            )
        else:
            sealed = control.commit_seal(
                artifact_id,
                request,
                claimed["start_request_digest"],
                manifest,
            )

        # The complete file is already published. Release can run before or
        # after deleting temporary chunks; neither order changes the sealed file.
        files.cleanup_source_chunks(artifact_id)
        return sealed

    def claim_seal(
        self,
        artifact_id: str,
        seal_request: Any,
        *,
        authorize: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Remember the first whole-file digest before checking upload completion."""

        source = self._live_source(artifact_id)
        record = source["reservation"]
        start = record["start_request"]
        request = validate_source_artifact_seal_request(seal_request)
        authorize()
        existing = self._records.sealed(source)
        if existing is not None:
            if existing["seal_request"] != request:
                raise self._conflict(start["artifact_id"], "seal request differs")
            return {
                "start_request_digest": record["start_request_digest"],
                "sealed": existing["sealed"],
            }

        winner = self._records.claim_seal(
            source,
            canonical_tree(
                {
                    "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                    "artifact_id": start["artifact_id"],
                    "seal_request": request,
                    "claimed_at": iso_z(self._now()),
                }
            ),
        )
        if winner["seal_request"] != request:
            raise self._conflict(start["artifact_id"], "seal request differs")
        commits = self._records.chunk_commits(source)
        count = canonical_chunk_count(start["total_bytes"], start["chunk_size_bytes"])
        if set(commits) != set(range(count)):
            raise GatewayFailure(
                GatewayErrorCode.VALIDATION_ERROR,
                "source upload is incomplete",
                resource_id=start["artifact_id"],
            )
        self._require_live(source, self._now())
        authorize()
        return {
            "start_request": start,
            "start_request_digest": record["start_request_digest"],
            "chunk_digests": {
                chunk_id: value["chunk"]["sha256"]
                for chunk_id, value in commits.items()
            },
        }

    def commit_seal(
        self,
        artifact_id: str,
        seal_request: Any,
        start_request_digest: str,
        manifest: Any,
        *,
        authorize: Callable[[], None] = lambda: None,
    ) -> dict[str, Any]:
        """Publish a sealed image reference after its file and manifest are durable."""

        request = validate_source_artifact_seal_request(seal_request)
        source = self._live_source(artifact_id)
        current = source["reservation"]
        self._require_same_start(current, start_request_digest)
        start = current["start_request"]
        authorize()
        claim = self._records.seal_claim(source)
        if claim is None or claim["seal_request"] != request:
            raise invalid_source_record(artifact_id)
        publication = validate_source_artifact_publication(start, request, manifest)

        sealed = validate_source_artifact_sealed(
            {
                "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                "artifact": build_source_artifact_ref(start, publication),
                "absolute_expires_at": current["absolute_expires_at"],
            }
        )
        sealed_record = self._records.publish_sealed(
            source,
            canonical_tree(
                {
                    "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                    "artifact_id": artifact_id,
                    "seal_request": request,
                    "manifest": publication,
                    "sealed": sealed,
                }
            ),
        )
        if sealed_record["seal_request"] != request:
            raise self._conflict(artifact_id, "seal publication differs")
        authorize()
        return sealed_record["sealed"]

    def status(self, artifact_id: str) -> dict[str, Any]:
        """Return upload progress or the validated reference to a sealed image."""

        source = self._live_source(artifact_id)
        record = source["reservation"]
        start = record["start_request"]
        accepted = self._accepted(record)
        sealed = self._records.sealed(source)
        if sealed is not None:
            value = {
                "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                "artifact_id": start["artifact_id"],
                "start_request_digest": record["start_request_digest"],
                "phase": "SEALED",
                "missing_chunks": [],
                "sealed_artifact": sealed["sealed"]["artifact"],
                "absolute_expires_at": record["absolute_expires_at"],
            }
            return validate_source_artifact_status_projection(
                start,
                accepted,
                value,
                sealed["seal_request"],
                sealed["sealed"],
            )

        count = canonical_chunk_count(start["total_bytes"], start["chunk_size_bytes"])
        commits = self._records.chunk_commits(source)
        missing = [chunk_id for chunk_id in range(count) if chunk_id not in commits]
        value = {
            "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
            "artifact_id": start["artifact_id"],
            "start_request_digest": record["start_request_digest"],
            "phase": "UPLOADING",
            "missing_chunks": missing,
            "upload_idle_expires_at": iso_z(
                self._exact_idle_deadline(source, commits=commits)
            ),
            "absolute_expires_at": record["absolute_expires_at"],
        }
        return validate_source_artifact_status_projection(start, accepted, value)

    def resolve_sealed_reference(self, reference: Any) -> dict[str, Any]:
        """Resolve one exact live sealed ref to its immutable source records.

        Finite and durable inference use this read-only boundary before opening
        bytes.  A UUID match alone is insufficient: every compact reference
        field must still bind to the accepted start and committed manifest.
        """

        supplied = validate_source_artifact_ref(reference)
        source = self._live_source(supplied["artifact_id"])
        record = source["reservation"]
        sealed = self._records.sealed(source)
        if sealed is None:
            raise GatewayFailure(
                GatewayErrorCode.ARTIFACT_PRECONDITION_FAILED,
                "source artifact is not sealed",
                resource_id=supplied["artifact_id"],
            )

        start = record["start_request"]
        manifest = sealed["manifest"]
        try:
            validate_source_artifact_reference(start, manifest, supplied)
        except ValueError as error:
            raise GatewayFailure(
                GatewayErrorCode.ARTIFACT_PRECONDITION_FAILED,
                "source artifact reference does not match committed source state",
                resource_id=supplied["artifact_id"],
            ) from error
        if sealed["sealed"]["artifact"] != supplied:
            raise invalid_source_record(supplied["artifact_id"])
        return canonical_tree(
            {
                "start_request": start,
                "manifest": manifest,
                "artifact": supplied,
                "absolute_expires_at": record["absolute_expires_at"],
            }
        )

    def release(
        self,
        artifact_id: str,
        files: SourceFileStore,
        *,
        has_durable_reference: Callable[[str], bool] = lambda _artifact_id: False,
    ) -> None:
        """Idempotently release one source unless durable state retains it."""

        canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
        current = self._records.load(canonical_id)
        if (
            current is not None
            and self._records.sealed(current) is not None
            and has_durable_reference(canonical_id)
        ):
            return

        # Removing metadata first makes a later upload commit fail. The same
        # lifecycle owner excludes new reservations until file cleanup finishes.
        self._records.remove_artifact(canonical_id)
        files.delete_source(canonical_id)

    def cleanup_expired(
        self,
        files: SourceFileStore,
        *,
        now: datetime | None = None,
        exclude: str | None = None,
    ) -> None:
        """Collect expired records and source directories with no reservation."""

        instant = self._normalize_now(now or self._clock())
        canonical_exclude = (
            require_canonical_uuid(exclude, "artifact_id")
            if exclude is not None
            else None
        )
        self._cleanup_expired(files, now=instant, exclude=canonical_exclude)

    def _cleanup_expired(
        self,
        files: SourceFileStore,
        *,
        now: datetime,
        exclude: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Remove expired uploads and return this call's retained reservations.

        The caller serializes control changes. Excluded IDs keep their validated
        reservation without an expiry check; every other returned record is live.
        """

        decisions = self._retention_decisions(now, exclude)
        known_ids = {artifact_id for artifact_id, _, _ in decisions}
        # Snapshot directories before the batch so no reload interrupts its
        # uncommitted file deletions.
        orphan_ids = files.source_ids() - known_ids
        retained = {}
        expired_ids = []
        record_error = None
        for artifact_id, record, expired in decisions:
            if isinstance(expired, GatewayFailure):
                record_error = expired
                break
            if expired:
                expired_ids.append(artifact_id)
            else:
                retained[artifact_id] = record

        def expired_and_orphan_ids():
            # Do not dispatch beyond a known corrupt record. The file owner
            # commits earlier removals before propagating that validation error.
            yield from self._records.remove_artifacts(expired_ids)
            if record_error is not None:
                raise record_error
            # These directories have no record in the serialized scan, so a
            # second cloud removal is unnecessary.
            yield from sorted(orphan_ids - {exclude})

        files.delete_sources(expired_and_orphan_ids())
        return retained

    def _retention_decisions(
        self, now: datetime, exclude: str | None = None
    ) -> list[tuple[str, dict[str, Any], bool | GatewayFailure]]:
        """Validate every reservation before applying ordered cleanup decisions.

        Keep only reservations and expiry decisions, not all chunk histories.
        Nested-record failures are deferred to their sorted position, preserving
        earlier expired deletions while a corrupt reservation prevents any delete.
        """

        decisions = []
        for artifact_id, source in self._records.sources():
            try:
                expired = artifact_id != exclude and self._expired(source, now)
            except GatewayFailure as error:
                # Keep the coded failure, not traceback frames retaining this
                # image's complete chunk history until cleanup has finished.
                expired = GatewayFailure(
                    error.error["code"],
                    error.error["message"],
                    resource_id=error.error.get("resource_id"),
                    retry_not_before=error.error.get("retry_not_before"),
                )
            decisions.append((artifact_id, source["reservation"], expired))
        decisions.sort(key=lambda item: item[0])
        return decisions

    def _validate_start_limits(self, start: Mapping[str, Any]) -> None:
        if (
            len(canonical_bytes(start))
            > self._source_limits["max_start_canonical_bytes"]
        ):
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                "source start exceeds its canonical byte limit",
            )
        if start["total_bytes"] > self._source_limits["max_artifact_bytes"]:
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                "source artifact exceeds max_artifact_bytes",
            )
        if (
            sum(member["size_bytes"] for member in start["origin"].get("members", []))
            > self._source_limits["max_artifact_bytes"]
        ):
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                "extracted slide exceeds max_artifact_bytes",
            )
        if start["chunk_size_bytes"] > self._source_limits["max_chunk_bytes"]:
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                "source chunk size exceeds max_chunk_bytes",
            )
        if (
            canonical_chunk_count(start["total_bytes"], start["chunk_size_bytes"])
            > self._source_limits["max_chunks"]
        ):
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                "source artifact exceeds max_chunks",
            )
        levels_by_series: dict[int, int] = {}
        for level in start["required_levels"]:
            levels_by_series[level["series_index"]] = (
                levels_by_series.get(level["series_index"], 0) + 1
            )
            if (
                level["width"] > self._source_limits["max_level_width_px"]
                or level["height"] > self._source_limits["max_level_height_px"]
            ):
                raise GatewayFailure(
                    GatewayErrorCode.PAYLOAD_TOO_LARGE,
                    "source level dimensions exceed deployment limits",
                )
        if len(levels_by_series) > self._source_limits["max_image_series"] or any(
            count > self._source_limits["max_pyramid_levels_per_series"]
            for count in levels_by_series.values()
        ):
            raise GatewayFailure(
                GatewayErrorCode.PAYLOAD_TOO_LARGE,
                "source level requirements exceed deployment limits",
            )
        if start["container"] == "PNG":
            level = start["required_levels"][0]
            if (
                level["width"] * level["height"]
                > self._source_limits["max_complete_decode_pixels"]
            ):
                raise GatewayFailure(
                    GatewayErrorCode.PAYLOAD_TOO_LARGE,
                    "PNG source exceeds the complete-decode pixel limit",
                )

    def _live_source(self, artifact_id: str) -> dict[str, Any]:
        canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
        source = self._records.load(canonical_id)
        if source is None:
            raise GatewayFailure(
                GatewayErrorCode.ARTIFACT_NOT_FOUND,
                "source artifact does not exist",
                resource_id=canonical_id,
            )
        self._require_live(source, self._now())
        return source

    def _fast_idle_deadline(self, source: Mapping[str, Any]) -> datetime:
        record = source["reservation"]
        activity = self._records.activity(source)
        last_activity = (
            _parse_z(activity)
            if activity is not None
            else _parse_z(record["accepted_at"])
        )
        return min(
            last_activity
            + timedelta(seconds=self._source_limits["upload_idle_timeout_seconds"]),
            _parse_z(record["absolute_expires_at"]),
        )

    def _exact_idle_deadline(
        self,
        source: Mapping[str, Any],
        *,
        commits: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> datetime:
        record = source["reservation"]
        activity = [_parse_z(record["accepted_at"])]
        activity.extend(
            _parse_z(item["accepted_at"])
            for item in (
                commits if commits is not None else self._records.chunk_commits(source)
            ).values()
        )
        seal_claim = self._records.seal_claim(source)
        if seal_claim is not None:
            activity.append(_parse_z(seal_claim["claimed_at"]))
        idle = max(activity) + timedelta(
            seconds=self._source_limits["upload_idle_timeout_seconds"]
        )
        return min(idle, _parse_z(record["absolute_expires_at"]))

    def _expired(self, source: Mapping[str, Any], now: datetime) -> bool:
        record = source["reservation"]
        if now >= _parse_z(record["absolute_expires_at"]):
            return True
        if self._records.sealed(source) is not None:
            return False
        if now < self._fast_idle_deadline(source):
            return False
        # A retained activity cache may lag behind acknowledged chunks. Check
        # their original timestamps before expiring or rejecting the upload.
        return now >= self._exact_idle_deadline(source)

    def _require_live(self, source: Mapping[str, Any], now: datetime) -> None:
        if self._expired(source, now):
            artifact_id = source["reservation"]["start_request"]["artifact_id"]
            raise GatewayFailure(
                GatewayErrorCode.ARTIFACT_EXPIRED,
                "source artifact has expired",
                resource_id=artifact_id,
            )

    def _accepted(self, record: Mapping[str, Any]) -> dict[str, Any]:
        start = record["start_request"]
        return validate_source_artifact_acceptance(
            start,
            {
                "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
                "creation_action_id": start["creation_action_id"],
                "artifact_id": start["artifact_id"],
                "start_request_digest": record["start_request_digest"],
                "absolute_expires_at": record["absolute_expires_at"],
            },
        )

    @staticmethod
    def _chunk_accepted(record: Mapping[str, Any]) -> dict[str, Any]:
        return validate_source_byte_chunk_accepted(
            {
                "schema_version": record["schema_version"],
                "artifact_id": record["artifact_id"],
                "chunk": record["chunk"],
            }
        )

    @staticmethod
    def _require_same_chunk(record: Mapping[str, Any], digest: str) -> None:
        if record["chunk"]["sha256"] != digest:
            raise SourceArtifactStore._conflict(
                record["artifact_id"], "chunk bytes differ"
            )

    @staticmethod
    def _require_same_start(record: Mapping[str, Any], digest: str) -> None:
        if record["start_request_digest"] != digest:
            raise SourceArtifactStore._conflict(
                record["start_request"]["artifact_id"],
                "source reservation content differs",
            )

    def _now(self) -> datetime:
        return self._normalize_now(self._clock())

    @staticmethod
    def _normalize_now(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("source-artifact clock must be timezone-aware")
        milliseconds = value.microsecond // 1000 * 1000
        return value.astimezone(timezone.utc).replace(microsecond=milliseconds)

    def _peak_bytes(self, start: Mapping[str, Any]) -> int:
        return (
            2 * start["total_bytes"]
            + self._source_limits["max_start_canonical_bytes"]
            + sum(member["size_bytes"] for member in start["origin"].get("members", []))
        )

    @staticmethod
    def _conflict(artifact_id: str, message: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.IDEMPOTENCY_CONFLICT,
            message,
            resource_id=artifact_id,
        )

    @staticmethod
    def _service_loss(artifact_id: str) -> GatewayFailure:
        return GatewayFailure(
            GatewayErrorCode.SERVICE_UNAVAILABLE,
            "durable source bytes are missing or corrupt; recreate the artifact",
            resource_id=artifact_id,
        )


def _parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
