"""One bounded, lazily started Bio-Formats JVM per restored Python process.

Importing this module or collecting build evidence does not initialize Java.
The source store's existing Volume gate owns reader lifetimes. This lock only
owns first JVM startup; no mounted operation is called while it is held.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

_RUNTIME_ROOT = Path(__file__).parents[1] / "runtime"
_INSTALL_ROOT = Path("/opt/histoforge-readers")
_START_LOCK = threading.Lock()
_STARTED_PID: int | None = None


@lru_cache(maxsize=1)
def reader_runtime_lock() -> dict[str, Any]:
    """Read the same immutable dependency and JVM-policy file used by the image."""

    return json.loads((_RUNTIME_ROOT / "reader-runtime.json").read_text())


def _runtime_paths() -> tuple[Path, Path]:
    # Development overrides do not bypass JAR identity or Java-major validation.
    return (
        Path(os.environ.get("HISTOFORGE_JAVA_HOME", _INSTALL_ROOT / "java")),
        Path(
            os.environ.get(
                "HISTOFORGE_BIOFORMATS_JAR", _INSTALL_ROOT / "bioformats_package.jar"
            )
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def reader_runtime_inventory() -> dict[str, Any]:
    """Bind actual reader binaries without starting a JVM during a snapshot.

    An ordinary development checkout may lack the separately installed runtime;
    the deployed Linux image must contain it. Java tree hashing is streaming.
    """

    java_home, jar = _runtime_paths()
    if not java_home.is_dir() or not jar.is_file():
        if platform.system() == "Linux":
            raise RuntimeError("deployed original-slide reader runtime is missing")
        return {"installed": False}
    lock = reader_runtime_lock()
    jar_digest = _sha256(jar)
    if jar_digest != lock["bioformats"]["sha256"]:
        raise RuntimeError("Bio-Formats JAR differs from the locked artifact")
    release = (java_home / "release").read_text()
    files = []
    for path in sorted(java_home.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(java_home).as_posix(),
                    "sha256": _sha256(path),
                }
            )
    return {
        "installed": True,
        "java_release": release,
        "java_files": files,
        "bioformats_sha256": jar_digest,
    }


def ensure_bioformats_jvm() -> Any:
    """Start once after restore, refusing foreign JVMs, forks and changed pins."""

    global _STARTED_PID
    import jpype

    with _START_LOCK:
        pid = os.getpid()
        if _STARTED_PID is not None:
            if _STARTED_PID != pid or not jpype.isJVMStarted():
                raise RuntimeError("Bio-Formats JVM cannot be forked or restarted")
            return jpype
        if jpype.isJVMStarted():
            raise RuntimeError("Bio-Formats cannot adopt an unconfigured JVM")
        evidence = reader_runtime_inventory()
        if not evidence["installed"]:
            raise RuntimeError(
                "install the locked Bio-Formats runtime before original reads"
            )
        java_home, jar = _runtime_paths()
        lock = reader_runtime_lock()
        jvm = java_home / (
            "lib/server/libjvm.dylib"
            if platform.system() == "Darwin"
            else "lib/server/libjvm.so"
        )
        jpype.startJVM(
            str(jvm), *lock["jvm_options"], classpath=[str(jar)], convertStrings=True
        )
        system = jpype.JClass("java.lang.System")
        version = str(system.getProperty("java.runtime.version"))
        if str(system.getProperty("java.specification.version")) != "21":
            raise RuntimeError("Bio-Formats requires Java 21")
        if platform.system() == "Linux" and version != lock["java"]["version"]:
            raise RuntimeError(
                "Java runtime differs from the locked Linux distribution"
            )
        if (
            str(jpype.JClass("loci.formats.FormatTools").VERSION)
            != lock["bioformats"]["version"]
        ):
            raise RuntimeError(
                "loaded Bio-Formats version differs from the runtime lock"
            )
        jpype.JClass("javax.imageio.ImageIO").setUseCache(False)
        jpype.JClass("loci.common.DebugTools").setRootLevel("WARN")
        # Failed startup checks leave an unadoptable JVM, not a retry/fallback.
        _STARTED_PID = pid
        return jpype
