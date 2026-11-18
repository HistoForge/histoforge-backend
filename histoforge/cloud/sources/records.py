"""Keep one image's reservation, upload progress and seal in one Dict value.

Load a detached snapshot once per operation. The existing artifact lifecycle
lane serializes updates; chunk and seal commits must load again after byte IO.
Only components needed by the operation are validated. In particular, removing
an absolutely expired image must not depend on its old chunk metadata.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Any

from histoforge.canonical_json import require_canonical_uuid
from histoforge.cloud.sources.record_validation import (
    invalid_source_record,
    validate_activity,
    validate_chunk_claim,
    validate_chunk_commit,
    validate_reservation,
    validate_seal_claim,
    validate_sealed_publication,
)

# Private storage layout, versioned separately from the source-artifact wire contract.
SOURCE_RECORD_LAYOUT_VERSION = 2
_PREFIX = f"source:v{SOURCE_RECORD_LAYOUT_VERSION}:"
# File deletion uses the same group size, so it finishes one acknowledged
# group before requesting removal of the next group's records.
SOURCE_CLEANUP_GROUP_SIZE = 8
_FIELDS = {
    "schema_version",
    "reservation",
    "chunk_claims",
    "chunk_commits",
    "seal_claim",
    "sealed",
    "last_activity_at",
}


class SourceArtifactRecords:
    """Read and write private source records in the existing shared Dict."""

    def __init__(self, values: Any) -> None:
        self._values = values

    def load(self, artifact_id: str) -> dict[str, Any] | None:
        """Fetch one detached source, validating its envelope and reservation."""

        canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
        value = self._values.get(source_record_key(canonical_id))
        return None if value is None else _source_record(value, canonical_id)

    def sources(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Stream validated reservations without retaining every chunk history."""

        for key, value in self._values.items():
            if not isinstance(key, str) or not key.startswith("source:"):
                continue
            if not key.startswith(_PREFIX):
                # Old private layouts need an explicit offline conversion, not
                # silent exclusion from quota accounting or physical cleanup.
                raise invalid_source_record(key)
            artifact_id = key.removeprefix(_PREFIX)
            try:
                canonical_id = require_canonical_uuid(artifact_id, "artifact_id")
            except (TypeError, ValueError) as error:
                raise invalid_source_record(artifact_id) from error
            yield canonical_id, _source_record(value, canonical_id)

    def reserve(self, reservation: Any) -> tuple[dict[str, Any], bool]:
        """Create one source if absent, preserving the original same-ID winner."""

        artifact_id = reservation["start_request"]["artifact_id"]
        source = {
            "schema_version": SOURCE_RECORD_LAYOUT_VERSION,
            "reservation": validate_reservation(reservation, artifact_id),
            "chunk_claims": {},
            "chunk_commits": {},
            "seal_claim": None,
            "sealed": None,
            "last_activity_at": None,
        }
        created = _put_if_absent(
            self._values, source_record_key(artifact_id), copy.deepcopy(source)
        )
        if created:
            return source, True
        winner = self.load(artifact_id)
        if winner is None:
            raise invalid_source_record(artifact_id)
        return winner, False

    @staticmethod
    def chunk_commits(source: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        """Validate all durable chunk acknowledgements in this snapshot."""

        artifact_id = _artifact_id(source)
        result = {}
        for key, value in _chunk_map(source, "chunk_commits").items():
            item = validate_chunk_commit(value, artifact_id)
            chunk_id = item["chunk"]["chunk_id"]
            if key != str(chunk_id):
                raise invalid_source_record(artifact_id)
            result[chunk_id] = item
        return result

    @staticmethod
    def chunk_commit(source: Mapping[str, Any], chunk_id: int) -> dict[str, Any] | None:
        value = _chunk_map(source, "chunk_commits").get(str(chunk_id))
        if value is None:
            return None
        result = validate_chunk_commit(value, _artifact_id(source))
        if result["chunk"]["chunk_id"] != chunk_id:
            raise invalid_source_record(_artifact_id(source))
        return result

    @staticmethod
    def chunk_claim(source: Mapping[str, Any], chunk_id: int) -> dict[str, Any] | None:
        value = _chunk_map(source, "chunk_claims").get(str(chunk_id))
        if value is None:
            return None
        result = validate_chunk_claim(value, _artifact_id(source))
        if result["chunk_id"] != chunk_id:
            raise invalid_source_record(_artifact_id(source))
        return result

    def claim_chunk(self, source: dict[str, Any], claim: Any) -> dict[str, Any]:
        """Persist the first received digest before saving that chunk's bytes."""

        normalized = validate_chunk_claim(claim, _artifact_id(source))
        existing = self.chunk_claim(source, normalized["chunk_id"])
        if existing is not None:
            return existing
        _chunk_map(source, "chunk_claims")[str(normalized["chunk_id"])] = normalized
        self._write(source)
        return normalized

    def commit_chunk(self, source: dict[str, Any], commit: Any) -> dict[str, Any]:
        """Acknowledge durable bytes and activity in the same value replacement."""

        normalized = validate_chunk_commit(commit, _artifact_id(source))
        chunk_id = normalized["chunk"]["chunk_id"]
        existing = self.chunk_commit(source, chunk_id)
        if existing is not None:
            return existing
        _chunk_map(source, "chunk_commits")[str(chunk_id)] = normalized
        _advance_activity(source, normalized["accepted_at"])
        self._write(source)
        return normalized

    @staticmethod
    def seal_claim(source: Mapping[str, Any]) -> dict[str, Any] | None:
        value = source["seal_claim"]
        return (
            None if value is None else validate_seal_claim(value, _artifact_id(source))
        )

    def claim_seal(self, source: dict[str, Any], claim: Any) -> dict[str, Any]:
        """Persist the first file digest, even if the upload is not yet complete."""

        normalized = validate_seal_claim(claim, _artifact_id(source))
        existing = self.seal_claim(source)
        if existing is not None:
            return existing
        source["seal_claim"] = normalized
        _advance_activity(source, normalized["claimed_at"])
        self._write(source)
        return normalized

    @staticmethod
    def sealed(source: Mapping[str, Any]) -> dict[str, Any] | None:
        """Validate a successful seal without interpreting old chunk records."""

        value = source["sealed"]
        return (
            None
            if value is None
            else validate_sealed_publication(value, source["reservation"])
        )

    @staticmethod
    def activity(source: Mapping[str, Any]) -> str | None:
        value = source["last_activity_at"]
        return None if value is None else validate_activity(value, _artifact_id(source))

    def publish_sealed(
        self, source: dict[str, Any], publication: Any
    ) -> dict[str, Any]:
        """Publish the durable file and remove upload-only claims together."""

        normalized = validate_sealed_publication(publication, source["reservation"])
        existing = self.sealed(source)
        if existing is not None:
            return existing
        source["sealed"] = normalized
        source["chunk_claims"] = {}
        self._write(source)
        return normalized

    def remove_artifact(self, artifact_id: str) -> None:
        """Remove all control state before the caller removes source files."""

        self._values.pop(source_record_key(artifact_id), None)

    def remove_artifacts(self, artifact_ids: Iterable[str]) -> Iterator[str]:
        """Remove expired records in groups of eight, then yield their IDs.

        The lifecycle caller has already validated expiry and exclusions. Wait
        for every request in a group, even if one fails, before yielding IDs or
        raising. A failed group may leave files without records; the existing
        orphan cleanup removes them on retry. No requests outlive this method.
        """

        remaining = iter(artifact_ids)
        while batch := tuple(islice(remaining, SOURCE_CLEANUP_GROUP_SIZE)):
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                removals = [
                    pool.submit(self.remove_artifact, artifact_id)
                    for artifact_id in batch
                ]
                for removal in removals:
                    removal.result()
            yield from batch

    def _write(self, source: dict[str, Any]) -> None:
        # The lifecycle lane owns serialization. Its caller freshly loads this
        # snapshot; it is never reused across the chunk/seal byte-IO interval.
        key = source_record_key(_artifact_id(source))
        put = getattr(self._values, "put", None)
        value = copy.deepcopy(source)
        if callable(put):
            put(key, value)
        else:
            self._values[key] = value


def source_record_key(artifact_id: str) -> str:
    return _PREFIX + require_canonical_uuid(artifact_id, "artifact_id")


def _source_record(value: Any, artifact_id: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _FIELDS
        or type(value["schema_version"]) is not int
        or value["schema_version"] != SOURCE_RECORD_LAYOUT_VERSION
    ):
        raise invalid_source_record(artifact_id)
    result = copy.deepcopy(value)
    result["reservation"] = validate_reservation(value["reservation"], artifact_id)
    return result


def _artifact_id(source: Mapping[str, Any]) -> str:
    return source["reservation"]["start_request"]["artifact_id"]


def _chunk_map(source: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = source[field]
    if not isinstance(value, dict):
        raise invalid_source_record(_artifact_id(source))
    return value


def _advance_activity(source: dict[str, Any], observed: str) -> None:
    current = SourceArtifactRecords.activity(source)
    if current is None or current < observed:
        source["last_activity_at"] = observed


def _put_if_absent(store: Any, key: str, value: Any) -> bool:
    put = getattr(store, "put", None)
    if callable(put):
        return bool(put(key, value, skip_if_exists=True))
    if key in store:
        return False
    store[key] = value
    return True
