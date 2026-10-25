"""Single-image reads, interleaved updates and ambiguous-write recovery."""

import copy
import hashlib
from datetime import timedelta
from functools import partial

import pytest
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.records import source_record_key

from test_source_artifact_store import (
    FakeDict,
    _instant,
    _png_source,
    _store,
    _upload_all,
)


def test_each_control_operation_fetches_one_source_without_enumeration(tmp_path):
    values = FakeDict()
    store = _store(values)
    files = SourceFileStore(tmp_path)
    start, encoded = _png_source()
    store.reserve(start)
    artifact_id = start["artifact_id"]
    key = source_record_key(artifact_id)
    assert set(values) == {key}
    count = (len(encoded) + start["chunk_size_bytes"] - 1) // start["chunk_size_bytes"]

    def one_read(operation, *args):
        values.get_calls.clear()
        values.items_calls = 0
        result = operation(*args)
        assert values.get_calls == [key]
        assert values.items_calls == 0
        assert len(values) == 1
        return result

    one_read(store.reserve, start)
    one_read(store.chunk_range, artifact_id, 0)
    for chunk_id in range(count):
        offset = chunk_id * start["chunk_size_bytes"]
        body = encoded[offset : offset + start["chunk_size_bytes"]]
        digest = _digest(body)
        claimed = one_read(store.claim_chunk, artifact_id, chunk_id, len(body), digest)
        files.publish_source_chunk(artifact_id, chunk_id, body, digest)
        one_read(
            store.commit_chunk,
            artifact_id,
            chunk_id,
            digest,
            claimed["start_request_digest"],
        )
    one_read(store.status, artifact_id)
    request = {"bytes_digest": _digest(encoded)}
    claimed = one_read(store.claim_seal, artifact_id, request)
    manifest = files.seal_source(
        start, request, claimed["chunk_digests"], store._source_limits
    )
    sealed = one_read(
        store.commit_seal,
        artifact_id,
        request,
        claimed["start_request_digest"],
        manifest,
    )
    one_read(store.claim_seal, artifact_id, request)
    one_read(store.status, artifact_id)
    one_read(store.resolve_sealed_reference, sealed["artifact"])
    values.get_calls.clear()
    values.items_calls = 0
    store.release(artifact_id, files)
    assert values.get_calls == [key]
    assert values.items_calls == 0
    assert values == {}


def test_late_identical_chunk_commit_after_seal_returns_original_acknowledgement(
    tmp_path,
):
    values = FakeDict()
    store = _store(values)
    files = SourceFileStore(tmp_path)
    start, encoded = _png_source()
    accepted = store.reserve(start)
    _upload_all(store, files, start, encoded)
    body = encoded[: start["chunk_size_bytes"]]
    acknowledged = store.accept_chunk(start["artifact_id"], 0, body, files)
    store.seal(start["artifact_id"], {"bytes_digest": _digest(encoded)}, files)
    before = copy.deepcopy(values)
    assert values[source_record_key(start["artifact_id"])]["chunk_claims"] == {}

    result = store.commit_chunk(
        start["artifact_id"], 0, _digest(body), accepted["start_request_digest"]
    )
    assert result == acknowledged
    assert values == before
    with pytest.raises(GatewayFailure) as conflict:
        store.commit_chunk(
            start["artifact_id"], 0, "f" * 64, accepted["start_request_digest"]
        )
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"
    assert values == before


def test_interleaved_chunk_io_preserves_both_fresh_commits(tmp_path):
    values = FakeDict()
    store = _store(values)
    files = SourceFileStore(tmp_path)
    start, encoded = _png_source()
    accepted = store.reserve(start)
    artifact_id = start["artifact_id"]
    bodies = [
        encoded[offset : offset + start["chunk_size_bytes"]]
        for offset in (0, start["chunk_size_bytes"])
    ]
    for chunk_id, body in enumerate(bodies):
        store.claim_chunk(artifact_id, chunk_id, len(body), _digest(body))
    for chunk_id in (1, 0):
        body = bodies[chunk_id]
        files.publish_source_chunk(artifact_id, chunk_id, body, _digest(body))
        store.commit_chunk(
            artifact_id, chunk_id, _digest(body), accepted["start_request_digest"]
        )
    source = values[source_record_key(artifact_id)]
    assert set(source["chunk_claims"]) == {"0", "1"}
    assert set(source["chunk_commits"]) == {"0", "1"}
    assert store.status(artifact_id)["missing_chunks"] == [2]


def test_incomplete_seal_keeps_first_digest_and_does_not_renew_its_claim(tmp_path):
    now = [_instant()]
    values = FakeDict()
    store = _store(values, clock=lambda: now[0])
    files = SourceFileStore(tmp_path)
    start, encoded = _png_source()
    store.reserve(start)
    request = {"bytes_digest": _digest(encoded)}

    with pytest.raises(GatewayFailure, match="incomplete"):
        store.claim_seal(start["artifact_id"], request)
    before = copy.deepcopy(values)
    now[0] += timedelta(seconds=1)
    with pytest.raises(GatewayFailure, match="incomplete"):
        store.claim_seal(start["artifact_id"], request)
    assert values == before
    with pytest.raises(GatewayFailure) as conflict:
        store.claim_seal(start["artifact_id"], {"bytes_digest": "f" * 64})
    assert conflict.value.error["code"] == "IDEMPOTENCY_CONFLICT"
    assert values == before
    _upload_all(store, files, start, encoded)
    store.seal(start["artifact_id"], request, files)
    source = values[source_record_key(start["artifact_id"])]
    assert (
        source["seal_claim"]
        == before[source_record_key(start["artifact_id"])]["seal_claim"]
    )


class _LostWriteResponse(FakeDict):
    """Persist a whole value, then lose its response once at the requested step."""

    lose_response_when = None

    def put(self, key, value, *, skip_if_exists=False):
        result = super().put(key, value, skip_if_exists=skip_if_exists)
        if self.lose_response_when is not None and self.lose_response_when(value):
            self.lose_response_when = None
            raise RuntimeError("lost Dict write response")
        return result


@pytest.mark.parametrize(
    "step", ["chunk_claim", "chunk_commit", "seal_claim", "sealed"]
)
def test_lost_whole_value_response_retries_without_renewing_timestamps(tmp_path, step):
    now = [_instant()]
    values = _LostWriteResponse()
    store = _store(values, clock=lambda: now[0])
    files = SourceFileStore(tmp_path)
    start, encoded = _png_source()
    accepted = store.reserve(start)
    artifact_id = start["artifact_id"]
    body = encoded[: start["chunk_size_bytes"]]
    digest = _digest(body)
    request = {"bytes_digest": _digest(encoded)}
    if step == "chunk_claim":
        operation = partial(store.claim_chunk, artifact_id, 0, len(body), digest)
        values.lose_response_when = lambda value: bool(value["chunk_claims"])
    elif step == "chunk_commit":
        store.claim_chunk(artifact_id, 0, len(body), digest)
        files.publish_source_chunk(artifact_id, 0, body, digest)
        operation = partial(
            store.commit_chunk, artifact_id, 0, digest, accepted["start_request_digest"]
        )
        values.lose_response_when = lambda value: bool(value["chunk_commits"])
    else:
        _upload_all(store, files, start, encoded)
        if step == "seal_claim":
            operation = partial(store.claim_seal, artifact_id, request)
            values.lose_response_when = lambda value: value["seal_claim"] is not None
        else:
            claimed = store.claim_seal(artifact_id, request)
            manifest = files.seal_source(
                start, request, claimed["chunk_digests"], store._source_limits
            )
            operation = partial(
                store.commit_seal,
                artifact_id,
                request,
                accepted["start_request_digest"],
                manifest,
            )
            values.lose_response_when = lambda value: value["sealed"] is not None

    with pytest.raises(RuntimeError, match="lost Dict"):
        operation()
    before = copy.deepcopy(values)
    now[0] += timedelta(seconds=1)
    operation()
    assert values == before
    status = store.status(artifact_id)
    if step == "chunk_claim":
        assert 0 in status["missing_chunks"]
    elif step == "chunk_commit":
        assert 0 not in status["missing_chunks"]
    else:
        assert status["phase"] == ("SEALED" if step == "sealed" else "UPLOADING")


def test_deferred_cleanup_failure_keeps_no_chunk_history_in_its_traceback(tmp_path):
    values = FakeDict()
    store = _store(values)
    files = SourceFileStore(tmp_path)
    start, encoded = _png_source()
    store.reserve(start)
    _upload_all(store, files, start, encoded)
    store.seal(start["artifact_id"], {"bytes_digest": _digest(encoded)}, files)
    values[source_record_key(start["artifact_id"])]["sealed"]["manifest"][
        "manifest_digest"
    ] = "f" * 64

    decisions = store._retention_decisions(_instant())
    assert len(decisions) == 1
    _, reservation, error = decisions[0]
    assert "chunk_commits" not in reservation
    assert isinstance(error, GatewayFailure)
    assert error.__traceback__ is None
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.error["resource_id"] == start["artifact_id"]


def test_old_private_layout_cannot_be_silently_excluded_from_admission():
    values = FakeDict()
    old, _ = _png_source()
    values[f"source:v1:reservation:{old['artifact_id']}"] = {"old layout": True}
    before = copy.deepcopy(values)
    fresh, _ = _png_source()
    with pytest.raises(GatewayFailure) as failure:
        _store(values).reserve(fresh)
    assert failure.value.error["code"] == "INTERNAL_ERROR"
    assert values == before


def _digest(value):
    return hashlib.sha256(value).hexdigest()
