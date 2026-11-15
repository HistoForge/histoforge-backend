"""Canonical and path-safe HistoForge module references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from histoforge.canonical_json import require_canonical_uuid

_SEGMENT = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")


@dataclass(frozen=True)
class ModuleRef:
    """Validated ``owner/name`` identity with no revision or URL semantics."""

    owner: str
    name: str

    def __post_init__(self) -> None:
        _require_segment(self.owner, "owner")
        _require_segment(self.name, "name")
        if len(str(self)) > 96:
            raise ValueError("module ref must be at most 96 characters")

    @classmethod
    def parse(cls, value: str) -> ModuleRef:
        """Parse only the exact canonical ``owner/name`` form."""

        if not isinstance(value, str):
            raise ValueError("module ref must be a string")
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError("module ref must be canonical owner/name")
        ref = cls(parts[0], parts[1])
        if str(ref) != value:
            raise ValueError("module ref must use its canonical owner/name spelling")
        return ref

    @classmethod
    def from_route_segments(cls, encoded_owner: str, encoded_name: str) -> ModuleRef:
        """Validate two raw URL segments and reject every noncanonical escape."""

        owner = _decode_route_segment(encoded_owner)
        name = _decode_route_segment(encoded_name)
        return cls(owner, name)

    def route_segments(self) -> tuple[str, str]:
        """Return the two already URL-safe canonical route segments."""

        return self.owner, self.name

    def filesystem_parts(self) -> tuple[str, str]:
        """Return validated path components; callers never join a raw ref string."""

        return self.owner, self.name

    def __str__(self) -> str:
        return f"{self.owner}/{self.name}"


def workspace_training_run_id(module_ref: object) -> str | None:
    """Return the run ID only for the reserved private training-result ref."""

    if not isinstance(module_ref, str):
        return None
    try:
        ref = ModuleRef.parse(module_ref)
        if ref.owner != "workspace" or not ref.name.startswith("run-"):
            return None
        return require_canonical_uuid(ref.name.removeprefix("run-"), "run_id")
    except ValueError:
        return None


def _require_segment(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 96
        or not _SEGMENT.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ValueError(f"module {name} is not a canonical path-safe segment")


def _decode_route_segment(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("module route segment must be nonempty")
    # Every schema-1 segment character is URL-unreserved, so its one canonical
    # route spelling is the raw character itself. Any escape is noncanonical.
    if "%" in value:
        raise ValueError("module route segment is not canonically encoded")
    if value in {".", ".."} or any(character in value for character in "/\\\x00"):
        raise ValueError("module route segment is not path-safe")
    return value
