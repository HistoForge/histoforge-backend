"""The CI and cloud image installer must verify the same reader artifacts."""

import hashlib
import io
import json
import tarfile

import pytest
from histoforge.cloud.runtime.install_readers import install
from histoforge.cloud.sources import java_runtime


def _reader_lock(tmp_path):
    """Create tiny local artifacts; no Java process or network is needed."""
    archive = tmp_path / "java.tar.gz"
    with tarfile.open(archive, "w:gz") as package:
        for name, content in {
            "jdk-test/release": b'JAVA_VERSION="21.0.12.1"\n',
            "jdk-test/lib/server/libjvm.so": b"synthetic JVM",
        }.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            package.addfile(entry, io.BytesIO(content))
    jar = tmp_path / "bioformats.jar"
    jar.write_bytes(b"synthetic Bio-Formats")
    lock = {
        "java": {
            "archive_url": archive.as_uri(),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
        "bioformats": {
            "jar_url": jar.as_uri(),
            "sha256": hashlib.sha256(jar.read_bytes()).hexdigest(),
        },
    }
    path = tmp_path / "reader-runtime.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path, lock


def test_reader_installer_creates_runtime_bound_to_the_lock(tmp_path, monkeypatch):
    lock_path, lock = _reader_lock(tmp_path)
    destination = tmp_path / "installed"
    install(lock_path, destination)
    assert (destination / "java/lib/server/libjvm.so").read_bytes() == b"synthetic JVM"
    monkeypatch.setenv("HISTOFORGE_JAVA_HOME", str(destination / "java"))
    monkeypatch.setenv(
        "HISTOFORGE_BIOFORMATS_JAR", str(destination / "bioformats_package.jar")
    )
    monkeypatch.setattr(java_runtime, "reader_runtime_lock", lambda: lock)
    # Avoid modifying the cached inventory used by other tests in this process.
    inventory = java_runtime.reader_runtime_inventory.__wrapped__()
    assert inventory["installed"] is True
    assert inventory["bioformats_sha256"] == lock["bioformats"]["sha256"]
    assert inventory["java_release"] == 'JAVA_VERSION="21.0.12.1"\n'
    assert [item["path"] for item in inventory["java_files"]] == [
        "lib/server/libjvm.so",
        "release",
    ]


@pytest.mark.parametrize("artifact", ["java", "bioformats"])
def test_reader_installer_rejects_tampered_downloads(tmp_path, artifact):
    lock_path, lock = _reader_lock(tmp_path)
    lock[artifact]["sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its lock"):
        install(lock_path, tmp_path / "installed")


def test_reader_installer_does_not_overwrite_an_existing_runtime(tmp_path):
    lock_path, _ = _reader_lock(tmp_path)
    destination = tmp_path / "installed"
    destination.mkdir()
    sentinel = destination / "owned-data"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        install(lock_path, destination)
    assert sentinel.read_bytes() == b"preserve"
    assert list(destination.iterdir()) == [sentinel]


def test_missing_linux_reader_runtime_remains_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(java_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HISTOFORGE_JAVA_HOME", str(tmp_path / "missing-java"))
    monkeypatch.setenv("HISTOFORGE_BIOFORMATS_JAR", str(tmp_path / "missing.jar"))
    with pytest.raises(RuntimeError, match="original-slide reader runtime is missing"):
        java_runtime.reader_runtime_inventory.__wrapped__()
