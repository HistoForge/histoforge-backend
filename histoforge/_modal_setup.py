"""Ensure Modal credentials exist before the CLI makes Modal SDK calls.

First-time login runs the bundled Modal SDK's setup module (browser auth), the
only subprocess the CLI uses. Everything else goes through the Modal Python
SDK.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def ensure_auth(profile: str | None = None, *, interactive: bool = True) -> str:
    """Select a profile before SDK import; optionally complete browser login."""

    requested = profile if profile is not None else os.environ.get("MODAL_PROFILE")
    if requested is not None and not requested.strip():
        raise ValueError("Modal profile must not be blank")

    selected = requested or _active_profile()
    if _has_creds(selected):
        os.environ["MODAL_PROFILE"] = selected
        return selected

    if not interactive:
        raise RuntimeError("Modal login is required before saving a registry token")

    print(
        "No Modal credentials found. Opening a browser to log in...",
        file=sys.stderr,
    )
    # The QuPath extension installs HistoForge into a private uv tool
    # environment. Its `modal` console script is therefore not guaranteed to
    # be on PATH, but the dependency is importable by this exact interpreter.
    command = [sys.executable, "-m", "modal", "setup"]
    if requested is not None:
        command.extend(["--profile", requested])
    # The helper reserves stdout for its final JSON result. Login instructions,
    # including the manual URL/code, must stream to the plugin's progress view.
    # QuPath reads progress as UTF-8. Windows pipes otherwise use a legacy code
    # page, which cannot encode Modal's startup formatting and aborts before login.
    subprocess.run(
        command,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=sys.stderr,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    # Modal setup runs in a child process. Re-read the file it published so the
    # parent can select the exact profile before importing the Modal SDK.
    selected = requested or _active_profile()
    if not _has_creds(selected):
        raise RuntimeError(f"Modal setup did not configure profile {selected!r}")
    os.environ["MODAL_PROFILE"] = selected
    return selected


def _active_profile() -> str:
    config = _read_config()
    active = [name for name, value in config.items() if value.get("active") is True]
    if len(active) > 1:
        raise RuntimeError("more than one Modal profile is active")
    return active[0] if active else "default"


def _read_config() -> dict[str, dict[str, Any]]:
    path = Path(
        os.environ.get("MODAL_CONFIG_PATH", str(Path.home() / ".modal.toml"))
    ).expanduser()
    if not path.exists():
        return {}

    # Modal 1.4.3 depends on `toml`; importing it directly avoids importing
    # `modal`, whose active profile is fixed at module-import time.
    import toml

    value = toml.load(path)
    if not isinstance(value, dict) or not all(
        isinstance(item, dict) for item in value.values()
    ):
        raise RuntimeError("Modal config must contain one table per profile")
    return value


def _has_creds(profile: str) -> bool:
    configured = _read_config().get(profile, {})
    token_id = os.environ.get("MODAL_TOKEN_ID") or configured.get("token_id")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET") or configured.get(
        "token_secret"
    )
    return bool(token_id and token_secret)
