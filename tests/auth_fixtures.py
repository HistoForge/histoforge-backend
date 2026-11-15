"""Dict-era gateway records for protocol and cutover tests."""

import uuid

from histoforge.cloud import _auth


def provision_legacy_state(store):
    token = _auth.new_token()
    state = _auth.parse_gateway_state(
        {
            "deployment_id": str(uuid.uuid4()),
            "auth_revision": 1,
            "token_sha256": _auth.token_digest(token),
        }
    )
    store[_auth.STATE_KEY] = state
    return state, token
