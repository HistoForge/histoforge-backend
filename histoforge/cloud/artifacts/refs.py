"""Validate artifact references shared with the QuPath extension.

Each reference identifies one exact transferred byte representation. Storage,
retention, and downloads belong to the artifact services.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from histoforge.canonical_json import (
    canonical_tree,
    require_canonical_uuid,
    require_nonblank_string,
    require_nonnegative_safe_integer,
    require_sha256,
)

_ARTIFACT_REF_FIELDS = {
    "artifact_id",
    "media_type",
    "size_bytes",
    "sha256",
}
_ARTIFACT_REF_FIELDS_WITH_ENCODING = _ARTIFACT_REF_FIELDS | {"content_encoding"}
_HTTP_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9a-z-]+")
_MEDIA_TYPE = re.compile(r"[!#$%&'*+.^_`|~0-9a-z-]+/[!#$%&'*+.^_`|~0-9a-z-]+")


def validate_artifact_ref(value: Any) -> dict[str, Any]:
    """Validate one opaque locator plus the complete transferred-byte identity.

    ``artifact_id`` locates bytes but does not own their lifetime. ``media_type``
    and optional ``content_encoding`` describe the transferred representation;
    ``size_bytes`` and ``sha256`` record the expected byte count and digest for
    download verification.
    """

    reference = _require_object(value, "artifact ref")
    fields = set(reference)
    if fields != _ARTIFACT_REF_FIELDS and fields != _ARTIFACT_REF_FIELDS_WITH_ENCODING:
        allowed = _ARTIFACT_REF_FIELDS_WITH_ENCODING
        raise ValueError(
            "artifact ref fields mismatch: "
            f"missing={sorted(_ARTIFACT_REF_FIELDS - fields)}, "
            f"unknown={sorted(fields - allowed)}"
        )

    result = {
        "artifact_id": require_canonical_uuid(reference["artifact_id"], "artifact_id"),
        "media_type": validate_media_type(reference["media_type"]),
        "size_bytes": require_nonnegative_safe_integer(
            reference["size_bytes"], "size_bytes"
        ),
        "sha256": require_sha256(reference["sha256"], "sha256"),
    }
    if "content_encoding" in reference:
        result["content_encoding"] = validate_content_encoding(
            reference["content_encoding"]
        )
    return canonical_tree(result)


def validate_media_type(value: Any) -> str:
    """Require the contract's lowercase parameter-free HTTP media type."""

    media_type = require_nonblank_string(value, "media_type")
    if not _MEDIA_TYPE.fullmatch(media_type):
        raise ValueError(
            "media_type must be a lowercase parameter-free HTTP media type"
        )
    return media_type


def validate_content_encoding(value: Any) -> str:
    """Require one lowercase HTTP content-coding token."""

    encoding = require_nonblank_string(value, "content_encoding")
    if not _HTTP_TOKEN.fullmatch(encoding):
        raise ValueError("content_encoding must be one lowercase HTTP token")
    return encoding


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value
