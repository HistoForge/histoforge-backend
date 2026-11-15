"""Durable-run publication cleanup tests."""

from __future__ import annotations

import pytest
from histoforge.cloud._errors import GatewayErrorCode, GatewayFailure
from histoforge.cloud.artifacts.lifecycle import _publish_or_discard


class _Store:
    def __init__(self, *, cleanup_fails: bool = False) -> None:
        self.cleanup_fails = cleanup_fails
        self.discarded = []

    def discard_unreferenced_artifact(self, reference, files):
        self.discarded.append((reference, files))
        if self.cleanup_fails:
            raise OSError("cleanup unavailable")
        return True


def test_rejected_publication_discards_its_exact_artifact():
    store = _Store()
    reference = {"artifact_id": "publication"}
    files = object()

    assert not _publish_or_discard(lambda: False, reference, store, files)
    assert store.discarded == [(reference, files)]


def test_publication_failure_preserves_error_after_best_effort_cleanup():
    store = _Store(cleanup_fails=True)
    reference = {"artifact_id": "publication"}
    files = object()
    failure = GatewayFailure(
        GatewayErrorCode.RESULT_LIMIT_EXCEEDED,
        "aggregate result budget exceeded",
    )

    with pytest.raises(GatewayFailure) as raised:
        _publish_or_discard(lambda: _raise(failure), reference, store, files)
    assert raised.value is failure
    assert store.discarded == [(reference, files)]


def _raise(error):
    raise error
