"""Create-only setup and the explicit migration bridge, without provider access."""

from types import SimpleNamespace

import modal
import pytest
from histoforge._gateway_state_setup import GatewayStateSetup
from histoforge.canonical_json import canonical_bytes
from histoforge.cloud import _auth
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.gateway_state import GATEWAY_STATE_PATH
from tests.auth_fixtures import provision_legacy_state


@pytest.fixture
def storage(monkeypatch):
    state = SimpleNamespace(
        legacy={}, cache={}, record=None, events=[], upload_failure=None
    )

    class Upload:
        def __init__(self, force):
            self.force = force

        def __enter__(self):
            return self

        def put_file(self, stream, path):
            assert path == "/" + GATEWAY_STATE_PATH
            self.encoded = stream.read()

        def __exit__(self, *_args):
            assert state.cache == {}, "invalidate the cache before writing identity"
            if state.upload_failure:
                raise state.upload_failure
            if state.record is not None and not self.force:
                raise FileExistsError("record already exists")
            state.events.append(("upload", self.force))
            state.record = self.encoded

    class Volume:
        def read_file(self, path):
            assert path == GATEWAY_STATE_PATH
            if state.record is None:
                raise FileNotFoundError(path)
            yield state.record[:10]
            yield state.record[10:]

        def batch_upload(self, *, force):
            return Upload(force)

    def volume(name, *, create_if_missing, version):
        assert name == DEPLOYMENT_CONFIG.artifact_volume_name
        assert version == 2
        state.events.append(("volume", create_if_missing))
        return Volume()

    def mapping(name, *, create_if_missing):
        if name == DEPLOYMENT_CONFIG.legacy_gateway_state_dict_name:
            assert create_if_missing is False
            return state.legacy
        assert name == DEPLOYMENT_CONFIG.gateway_state_cache_dict_name
        assert create_if_missing is True
        return state.cache

    monkeypatch.setattr(modal.Volume, "from_name", volume)
    monkeypatch.setattr(modal.Dict, "from_name", mapping)
    return state


@pytest.mark.parametrize("legacy_identity", [False, True])
def test_bootstrap_and_retry_preserve_identity_without_overwriting(
    storage, legacy_identity
):
    storage.cache[_auth.STATE_KEY] = _auth.initial_state()
    if legacy_identity:
        old, _ = provision_legacy_state(storage.legacy)
    setup = GatewayStateSetup(fresh=True)
    setup.bootstrap()
    first = setup.read_record()
    if legacy_identity:
        assert first == old
    else:
        assert first["auth_revision"] == 0
    setup.bootstrap()
    assert setup.read_record() == first
    assert storage.events == [("volume", True), ("upload", False)]
    assert storage.cache == {}


def test_competing_bootstrap_uses_winners_identity(storage, monkeypatch):
    setup = GatewayStateSetup(fresh=True)
    winner = _auth.initial_state()

    def upload(_candidate, *, force):
        assert force is False
        storage.record = canonical_bytes(winner)
        raise FileExistsError("another setup won")

    monkeypatch.setattr(setup, "_upload", upload)
    setup.bootstrap()
    assert setup.read_record() == winner


def test_bootstrap_does_not_treat_other_upload_failures_as_success(storage):
    storage.upload_failure = OSError("upload unavailable")
    with pytest.raises(OSError, match="upload unavailable"):
        GatewayStateSetup(fresh=True).bootstrap()
    assert storage.record is None
    assert storage.legacy == {}


@pytest.mark.parametrize("durable_revision", [1, 3, 8])
def test_bootstrap_never_reuses_a_higher_legacy_generation(storage, durable_revision):
    old, _ = provision_legacy_state(storage.legacy)
    storage.legacy[_auth.STATE_KEY] = old | {"auth_revision": 3}
    storage.record = canonical_bytes(old | {"auth_revision": durable_revision})
    before = storage.record
    setup = GatewayStateSetup(fresh=True)
    if durable_revision < 3:
        with pytest.raises(ValueError, match="older than the legacy"):
            setup.bootstrap()
    else:
        setup.bootstrap()
    assert storage.record == before
    assert storage.events == [("volume", True)]


@pytest.mark.parametrize("record", [b"partial", b"x" * 4097])
def test_malformed_existing_record_is_never_overwritten(storage, record):
    storage.record = record
    with pytest.raises(ValueError):
        GatewayStateSetup(fresh=True).bootstrap()
    assert storage.record == record
    assert storage.events == [("volume", True)]


@pytest.mark.parametrize("staged_revision", [None, 1, 8])
def test_cutover_preserves_identity_and_advances_past_both_revisions(
    storage, staged_revision
):
    old, token = provision_legacy_state(storage.legacy)
    storage.cache[_auth.STATE_KEY] = old
    if staged_revision is not None:
        storage.record = canonical_bytes(old | {"auth_revision": staged_revision})
    setup = GatewayStateSetup()
    staged = setup.prepare_cutover(old)
    replacement = setup.cut_over(old, staged, _auth.token_digest("replacement"))
    assert replacement["deployment_id"] == old["deployment_id"]
    assert replacement["auth_revision"] == max(1, staged_revision or 0) + 1
    assert setup.read_record() == replacement
    assert storage.legacy[_auth.STATE_KEY] == replacement
    assert storage.cache == {}
    assert not _auth.matches_local_token(replacement, token)
    assert storage.events[-1] == ("upload", staged_revision is not None)


@pytest.mark.parametrize("record", ["foreign", "corrupt"])
def test_cutover_conflict_is_detected_before_old_state_changes(storage, record):
    old, _ = provision_legacy_state(storage.legacy)
    storage.record = (
        canonical_bytes(_auth.initial_state()) if record == "foreign" else b"broken"
    )
    setup = GatewayStateSetup()
    with pytest.raises(ValueError):
        setup.prepare_cutover(old)
    assert storage.legacy == {_auth.STATE_KEY: old}
    assert storage.events == [("volume", False)]


@pytest.mark.parametrize("changed", ["legacy", "record"])
def test_concurrent_admin_change_aborts_cutover_before_own_rotation(storage, changed):
    old, _ = provision_legacy_state(storage.legacy)
    setup = GatewayStateSetup()
    staged = setup.prepare_cutover(old)
    if changed == "legacy":
        _auth.legacy_rotate(storage.legacy)
    else:
        storage.record = canonical_bytes(old | {"auth_revision": 9})
    snapshot = dict(storage.legacy), storage.record
    with pytest.raises(ValueError, match="access changed"):
        setup.cut_over(old, staged, _auth.token_digest("new-token"))
    assert (storage.legacy, storage.record) == snapshot


def test_failed_upload_keeps_old_access_and_publishes_no_new_cache(storage):
    old, _ = provision_legacy_state(storage.legacy)
    setup = GatewayStateSetup()
    storage.upload_failure = OSError("upload failed")
    with pytest.raises(OSError, match="upload failed"):
        setup.cut_over(old, None, _auth.token_digest("withheld-token"))
    assert storage.legacy[_auth.STATE_KEY] == old
    assert storage.cache == {}
    assert storage.record is None
    # Reconnect remains possible, but is not required to retry this upload.
    connected, _ = setup.legacy_rotate()
    storage.upload_failure = None
    actual = setup.cut_over(connected, None, _auth.token_digest("retry-token"))
    assert actual["auth_revision"] == 3
    assert actual["deployment_id"] == old["deployment_id"]


def test_cutover_uploads_exact_state_before_fencing_legacy_workers(storage):
    old, _ = provision_legacy_state(storage.legacy)
    storage.cache[_auth.STATE_KEY] = old

    class Legacy(dict):
        def __setitem__(self, key, value):
            assert storage.record == canonical_bytes(value)
            storage.events.append(("fence", None))
            super().__setitem__(key, value)

    storage.legacy = Legacy(storage.legacy)
    actual = GatewayStateSetup().cut_over(old, None, _auth.token_digest("new-token"))
    assert storage.legacy[_auth.STATE_KEY] == actual
    assert storage.events == [("volume", False), ("upload", False), ("fence", None)]


@pytest.mark.parametrize("operation", ["bootstrap", "cutover"])
def test_failed_cache_deletion_blocks_identity_writes(storage, operation):
    old, _ = provision_legacy_state(storage.legacy)

    class FailingCache(dict):
        def pop(self, *_args):
            raise OSError("cache unavailable")

    storage.cache = FailingCache({_auth.STATE_KEY: old})
    setup = GatewayStateSetup(fresh=operation == "bootstrap")
    with pytest.raises(OSError, match="cache unavailable"):
        if operation == "bootstrap":
            setup.bootstrap()
        else:
            setup.cut_over(old, None, _auth.token_digest("new-token"))
    assert storage.record is None
    assert storage.legacy[_auth.STATE_KEY] == old
    assert storage.cache == {_auth.STATE_KEY: old}
    assert storage.events == [("volume", operation == "bootstrap")]


@pytest.mark.parametrize("operation", ["bootstrap", "cutover"])
def test_unverified_upload_aborts_before_legacy_auth_changes(
    storage, monkeypatch, operation
):
    old, _ = provision_legacy_state(storage.legacy)
    setup = GatewayStateSetup(fresh=operation == "bootstrap")
    monkeypatch.setattr(setup, "_upload", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match="record was not verified"):
        if operation == "bootstrap":
            setup.bootstrap()
        else:
            setup.cut_over(old, None, _auth.token_digest("new-token"))
    assert storage.record is None
    assert storage.legacy[_auth.STATE_KEY] == old
    assert storage.cache == {}
