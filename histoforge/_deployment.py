"""Exact private CLI-to-extension deployment credential."""

import ipaddress
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from histoforge.canonical_json import (
    parse_json,
    require_int32,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_rfc3339_millis,
)

SCHEMA_VERSION = 1
DIR = Path.home() / ".histoforge"
PATH = DIR / "deployment.json"
_FIELDS = {
    "schema_version",
    "deployment_id",
    "gateway_url",
    "modal_profile",
    "modal_app_name",
    "auth_revision",
    "token",
    "updated_at",
}
_DNS_HOST = re.compile(
    r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


def _write_private_text_atomically(path: Path, content: str) -> None:
    """Publish complete UTF-8 text without exposing partial or loose-mode bytes."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        if os.name == "posix":
            os.fchmod(fd, 0o600)

        stream = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        fd = -1  # The stream now owns the descriptor.
        with stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        # The sibling temporary file makes replacement atomic on supported filesystems.
        # An unsupported atomic replace fails here; there is no in-place fallback.
        os.replace(tmp, path)
    finally:
        if fd >= 0:
            os.close(fd)
        tmp.unlink(missing_ok=True)


def build(
    *,
    deployment_id: str,
    gateway_url: str,
    modal_profile: str,
    modal_app_name: str,
    auth_revision: int,
    token: str,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Construct one exact credential after an authenticated capability probe."""

    timestamp = updated_at or _now_millis()
    return parse(
        {
            "schema_version": SCHEMA_VERSION,
            "deployment_id": deployment_id,
            "gateway_url": gateway_url,
            "modal_profile": modal_profile,
            "modal_app_name": modal_app_name,
            "auth_revision": auth_revision,
            "token": token,
            "updated_at": timestamp,
        }
    )


def parse(value: Any) -> dict[str, Any]:
    """Validate the closed persisted credential contract."""

    if not isinstance(value, dict):
        raise ValueError("deployment credential must be an object")
    actual = set(value)
    if actual != _FIELDS:
        raise ValueError(
            "deployment credential fields mismatch: "
            f"missing={sorted(_FIELDS - actual)}, unknown={sorted(actual - _FIELDS)}"
        )
    if require_int32(value["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported deployment credential schema_version")
    return {
        "schema_version": SCHEMA_VERSION,
        "deployment_id": require_nonblank_string(
            value["deployment_id"], "deployment_id"
        ),
        "gateway_url": _gateway_url(value["gateway_url"]),
        "modal_profile": require_nonblank_string(
            value["modal_profile"], "modal_profile"
        ),
        "modal_app_name": require_nonblank_string(
            value["modal_app_name"], "modal_app_name"
        ),
        "auth_revision": require_nonnegative_safe_integer(
            value["auth_revision"], "auth_revision"
        ),
        "token": _token(value["token"]),
        "updated_at": require_rfc3339_millis(value["updated_at"], "updated_at"),
    }


def write(value: Any) -> dict[str, Any]:
    data = parse(value)
    _write_private_text_atomically(PATH, json.dumps(data, indent=2) + "\n")
    return data


def read() -> dict | None:
    if not PATH.exists():
        return None
    return parse(parse_json(PATH.read_bytes()))


def _gateway_url(value: Any) -> str:
    url = require_nonblank_string(value, "gateway_url")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise ValueError("gateway_url is invalid") from error
    host = parts.hostname
    if (
        not host
        or parts.username is not None
        or parts.password is not None
        or parts.path
        or parts.query
        or parts.fragment
    ):
        raise ValueError("gateway_url must be an origin without path or user-info")
    if not _valid_gateway_host(host):
        raise ValueError("gateway_url host must be a canonical IP or DNS name")
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("gateway_url port must be between 1 and 65535")
    if parts.scheme == "http":
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("HTTP gateway_url is permitted only for localhost")
    elif parts.scheme != "https":
        raise ValueError("gateway_url must use HTTPS")
    rendered_host = f"[{host}]" if ":" in host else host
    expected = f"{parts.scheme}://{rendered_host}"
    if port is not None:
        expected += f":{port}"
    if url != expected:
        raise ValueError("gateway_url must use canonical origin spelling")
    return url


def _valid_gateway_host(host: str) -> bool:
    if not host.isascii() or host != host.lower() or "%" in host:
        return False
    if ":" in host:
        try:
            return ipaddress.ip_address(host).version == 6
        except ValueError:
            return False
    if all(character.isdigit() or character == "." for character in host):
        try:
            return str(ipaddress.IPv4Address(host)) == host
        except ipaddress.AddressValueError:
            return False
    return _DNS_HOST.fullmatch(host) is not None


def _token(value: Any) -> str:
    token = require_nonblank_string(value, "token")
    if any(ord(character) <= 0x20 or ord(character) >= 0x7F for character in token):
        raise ValueError("token must be visible ASCII without whitespace")
    return token


def _now_millis() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
