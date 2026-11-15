"""Permanent identity survives cache expiry; ambiguous writes never authenticate."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud import _auth
from histoforge.cloud._errors import GatewayFailure
from histoforge.cloud.gateway_state import GATEWAY_STATE_PATH, GatewayStateStore
from tests.auth_fixtures import provision_legacy_state


class MountedVolume:
    """Separate local files from durable bytes, including ambiguous commit failure."""

    def __init__(self, root, committed):
        self.path = root / GATEWAY_STATE_PATH
        self.committed = committed
        self.fail_commit = None
        self.fail_reload = False
        self.commit_pending_on_reload = False
        self.reloads = 0
        self.commits = 0
        self.reload()

    def reload(self):
        self.reloads += 1
        if self.fail_reload:
            raise OSError("reload unavailable")
        if self.commit_pending_on_reload:
            self.committed = self.path.read_bytes()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.committed is None:
            self.path.unlink(missing_ok=True)
        else:
            self.path.write_bytes(self.committed)

    def commit(self):
        self.commits += 1
        failure, self.fail_commit = self.fail_commit, None
        if failure != "before":
            self.committed = self.path.read_bytes()
        if failure is not None:
            raise OSError("commit acknowledgement lost")


@pytest.fixture
def state(tmp_path):
    original, token = provision_legacy_state({})
    cache = {_auth.STATE_KEY: original}
    volume = MountedVolume(tmp_path, canonical_bytes(original))
    owner = GatewayStateStore(
        tmp_path, cache, reload_volume=volume.reload, commit_volume=volume.commit
    )
    return SimpleNamespace(
        root=tmp_path,
        original=original,
        token=token,
        cache=cache,
        volume=volume,
        owner=owner,
    )


@pytest.mark.parametrize("cached", [None, {"incomplete": True}])
def test_missing_or_invalid_cache_refills_without_changing_identity(state, cached):
    state.cache.clear()
    if cached is not None:
        state.cache[_auth.STATE_KEY] = cached
    assert state.owner.load_current() == state.original
    assert state.cache == {_auth.STATE_KEY: state.original}
    assert _auth.authenticate_token(state.owner.load_current, state.token) == (
        state.original
    )
    assert state.volume.commits == 0


@pytest.mark.parametrize("operation", ["load", "rotate"])
def test_missing_permanent_identity_never_creates_a_replacement(state, operation):
    state.volume.committed = None
    if operation == "load":
        state.cache.clear()
        assert state.owner.load_current() is None
    else:
        with pytest.raises(ValueError, match="missing its saved connection record"):
            state.owner.rotate(_auth.token_digest("new-token"))
    assert state.cache == {}
    assert not state.volume.path.exists()
    assert state.volume.commits == 0


@pytest.mark.parametrize("corruption", ["torn", "oversize", "extra", "pretty", "utf8"])
def test_invalid_permanent_record_invalidates_cache_and_fails_closed(state, corruption):
    encoded = {
        "torn": b'{"deployment_id":',
        "oversize": b"x" * 4097,
        "extra": canonical_bytes(state.original | {"extra": True}),
        "pretty": json.dumps(state.original, indent=2).encode(),
        "utf8": b"\xff",
    }[corruption]
    state.volume.committed = encoded
    with pytest.raises(ValueError, match="saved connection record is invalid"):
        state.owner.rotate(_auth.token_digest("new-token"))
    assert state.cache == {}
    assert state.volume.committed == encoded
    assert state.volume.commits == 0


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_nonregular_record_is_rejected_without_following_it(state, monkeypatch, kind):
    state.volume.path.unlink()
    if kind == "symlink":
        target = state.root / "unrelated.json"
        target.write_bytes(canonical_bytes(state.original))
        state.volume.path.symlink_to(target)
    else:
        state.volume.path.mkdir()
    monkeypatch.setattr(state.owner, "_reload", lambda: None)
    with pytest.raises(ValueError, match="saved connection record is invalid"):
        state.owner.rotate(_auth.token_digest("new-token"))
    assert state.cache == {}


def test_committed_cache_does_not_claim_to_detect_out_of_band_record_deletion(state):
    state.volume.committed = None
    assert state.owner.load_current() == state.original
    with pytest.raises(ValueError, match="missing its saved connection record"):
        state.owner.rotate(_auth.token_digest("new-token"))
    assert state.cache == {}
    assert state.owner.load_current() is None


@pytest.mark.parametrize("failure", ["before", "after"])
@pytest.mark.parametrize("implicit_commit", [False, True])
def test_ambiguous_commit_is_not_cached_and_next_read_uses_durable_state(
    state, failure, implicit_commit
):
    state.volume.fail_commit = failure
    verifier = _auth.token_digest("new-token")
    with pytest.raises(OSError, match="acknowledgement lost"):
        state.owner.rotate(verifier)
    assert state.cache == {}
    assert not state.volume.path.with_name(".v1.json.staging").exists()
    state.volume.commit_pending_on_reload = implicit_commit
    actual = state.owner.load_current()
    assert actual == json.loads(state.volume.committed)
    assert actual["deployment_id"] == state.original["deployment_id"]
    expected_revision = 2 if failure == "after" or implicit_commit else 1
    assert actual["auth_revision"] == expected_revision
    assert state.cache == {_auth.STATE_KEY: actual}


@pytest.mark.parametrize("point", ["fsync", "replace"])
def test_local_write_failure_leaves_cache_absent_and_removes_stage(
    state, monkeypatch, point
):
    def fail(*_args):
        raise OSError("local write failed")

    monkeypatch.setattr(f"histoforge.cloud.gateway_state.os.{point}", fail)
    with pytest.raises(OSError, match="local write failed"):
        state.owner.rotate(_auth.token_digest("new-token"))
    assert state.cache == {}
    assert state.owner.load_current() == state.original
    assert state.volume.commits == 0
    assert not state.volume.path.with_name(".v1.json.staging").exists()


def test_reload_failure_is_not_misreported_as_missing_identity(state):
    state.volume.fail_reload = True
    with pytest.raises(OSError, match="reload unavailable"):
        state.owner.rotate(_auth.token_digest("new-token"))
    assert state.cache == {_auth.STATE_KEY: state.original}
    assert state.volume.commits == 0


def test_failed_cache_invalidation_prevents_record_replacement(state):
    class Cache(dict):
        def pop(self, *_args):
            raise OSError("cache unavailable")

    state.owner._cache = Cache(state.cache)
    with pytest.raises(OSError, match="cache unavailable"):
        state.owner.rotate(_auth.token_digest("new-token"))
    assert state.volume.committed == canonical_bytes(state.original)
    assert state.volume.commits == 0


def test_failed_cache_publication_does_not_undo_a_committed_rotation(state):
    class Cache(dict):
        def __setitem__(self, *_args):
            raise OSError("cache unavailable")

    cache = Cache(state.cache)
    state.owner._cache = cache
    actual = state.owner.rotate(_auth.token_digest("new-token"))
    assert actual == json.loads(state.volume.committed)
    assert actual["auth_revision"] == 2
    assert cache == {}
    assert state.owner.load_current() == actual


@pytest.mark.parametrize(
    "expected", [{"expected_deployment_id": "foreign"}, {"expected_revision": 0}]
)
def test_update_conflict_checks_identity_and_revision_before_any_write(state, expected):
    with pytest.raises(ValueError, match="access changed"):
        state.owner.rotate(_auth.token_digest("new-token"), **expected)
    assert state.cache == {_auth.STATE_KEY: state.original}
    assert state.volume.committed == canonical_bytes(state.original)
    assert state.volume.commits == 0


@pytest.mark.parametrize(
    "expected",
    [
        {"expected_deployment_id": ""},
        {"expected_revision": True},
        {"expected_revision": -1},
        {"expected_revision": 1.5},
    ],
)
def test_invalid_update_preconditions_fail_before_read_or_write(state, expected):
    reloads = state.volume.reloads
    with pytest.raises(ValueError):
        state.owner.rotate(_auth.token_digest("new-token"), **expected)
    assert state.cache == {_auth.STATE_KEY: state.original}
    assert state.volume.commits == 0
    assert state.volume.reloads == reloads


def test_concurrent_rotations_advance_unique_revisions_and_fence_old_work(state):
    tokens = ["first-token", "second-token"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(state.owner.rotate, _auth.token_digest(token))
            for token in tokens
        ]
        rotations = [future.result() for future in futures]
    assert sorted(value["auth_revision"] for value in rotations) == [2, 3]
    current = state.owner.load_current()
    assert current["auth_revision"] == 3
    assert sum(_auth.matches_local_token(current, token) for token in tokens) == 1
    assert _auth.authenticate_token(state.owner.load_current, state.token) is None
    with pytest.raises(GatewayFailure) as caught:
        _auth.require_current_auth_generation(
            state.owner.load_current,
            state.original["deployment_id"],
            state.original["auth_revision"],
        )
    assert caught.value.error["code"] == "AUTHENTICATION_REQUIRED"
