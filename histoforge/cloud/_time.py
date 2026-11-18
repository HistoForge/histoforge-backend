"""Single owner of exact UTC-millisecond wire timestamps."""

from datetime import datetime, timezone


def iso_z(dt: datetime) -> str:
    """Return the RFC3339 UTC-millisecond form accepted by both codecs."""

    if dt.tzinfo is None:
        raise ValueError("wire timestamp must be timezone-aware")
    return (
        dt.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def now_z() -> str:
    """`iso_z(datetime.now(timezone.utc))` — most-common caller shorthand."""
    return iso_z(datetime.now(timezone.utc))
