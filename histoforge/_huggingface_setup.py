"""Provision the administration-only HF credential through the local helper.

The input is transient stdin, never argv, an environment variable or a local
credential file. Only the named Modal Secret persists it. External response
bodies, exception messages and SDK diagnostics must not leave this operation.
"""

import contextlib
import json
import logging
import os
import urllib.error
import urllib.request
from typing import TextIO

from histoforge import _modal_setup
from histoforge.cloud._config import DEPLOYMENT_CONFIG

MAX_TOKEN_CHARACTERS = 4096
_MAX_RESPONSE_BYTES = 64 * 1024
_IDENTITY_URL = "https://huggingface.co/api/whoami-v2"
_MESSAGES = {
    "HF_TOKEN_INVALID": "Enter a nonblank Hugging Face token without whitespace.",
    "HF_TOKEN_REJECTED": "Hugging Face rejected this token. Check or replace it.",
    "HF_VERIFICATION_FAILED": (
        "Could not verify the token with Hugging Face. Try again."
    ),
    "MODAL_AUTH_FAILED": "Could not authenticate with the selected Modal profile.",
    "HF_SECRET_SAVE_FAILED": (
        "Could not confirm that Modal saved the token. "
        "Check Modal access and try again."
    ),
}


class TokenSetupFailure(RuntimeError):
    """A closed diagnostic that never includes external or credential content."""

    def __init__(self, code: str):
        super().__init__(_MESSAGES[code])
        self.code = code


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _verify_token(token: str) -> None:
    request = urllib.request.Request(
        _IDENTITY_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    opener = urllib.request.build_opener(_RejectRedirect())
    try:
        with opener.open(request, timeout=30) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
            if response.status != 200 or len(body) > _MAX_RESPONSE_BYTES:
                raise ValueError("invalid identity response")
            identity = json.loads(body)
            if (
                not isinstance(identity, dict)
                or not isinstance(identity.get("name"), str)
                or not identity["name"].strip()
            ):
                raise ValueError("invalid identity response")
    except urllib.error.HTTPError as error:
        code = (
            "HF_TOKEN_REJECTED"
            if error.code in {401, 403}
            else "HF_VERIFICATION_FAILED"
        )
        error.close()
        raise TokenSetupFailure(code) from None
    except Exception:
        raise TokenSetupFailure("HF_VERIFICATION_FAILED") from None


def _save_secret(token: str) -> None:
    # Modal chooses its profile at import. The caller authenticates first.
    import modal

    name = DEPLOYMENT_CONFIG.huggingface_secret_name
    values = {"HF_TOKEN": token}
    try:
        modal.Secret.objects.create(name, values)
    except modal.exception.AlreadyExistsError:
        # Update the same Secret; never delete it, read it back or replace its
        # other keys. allow_existing=True alone would ignore the replacement.
        modal.Secret.from_name(name).update(values)


def save_token(profile: str | None, source: TextIO) -> str:
    """Verify, then create/update the selected workspace's administration Secret.

    Returns the exact selected local Modal profile. HF verification proves token
    validity, not permission for a particular private/gated repository or write.
    An invalid candidate never changes the saved Secret. An ambiguous Modal
    failure permits explicit retry, not delete/recreate or automatic rollback.
    """
    token = source.read(MAX_TOKEN_CHARACTERS + 1)
    if len(token) > MAX_TOKEN_CHARACTERS:
        raise TokenSetupFailure("HF_TOKEN_INVALID")
    token = token.strip()
    if not token or any(not (33 <= ord(char) <= 126) for char in token):
        raise TokenSetupFailure("HF_TOKEN_INVALID")

    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            _verify_token(token)
            try:
                # Browser login runs separately, without an HF credential, so
                # its manual URL/code can be visible while this output is closed.
                selected = _modal_setup.ensure_auth(profile, interactive=False)
            except Exception:
                raise TokenSetupFailure("MODAL_AUTH_FAILED") from None
            try:
                _save_secret(token)
            except Exception:
                raise TokenSetupFailure("HF_SECRET_SAVE_FAILED") from None
        return selected
    finally:
        logging.disable(previous_logging)
