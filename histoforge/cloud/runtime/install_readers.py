"""Build-time installation of the pinned Linux Java/Bio-Formats artifacts.

Run by the existing Modal image build, never by a source-reader request. No
dependency resolver, mutable release URL or runtime download is involved.
"""

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def download(url: str, destination: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        destination.open("xb") as output,
    ):
        for block in iter(lambda: response.read(1024 * 1024), b""):
            size += len(block)
            if size > 1024**3:
                raise ValueError("reader runtime download exceeds the build limit")
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("reader runtime download differs from its lock")


def install(lock_path: Path, destination: Path) -> None:
    lock = json.loads(lock_path.read_text())
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="histoforge-reader-build-") as temporary:
        stage = Path(temporary)
        archive = stage / "java.tar.gz"
        download(lock["java"]["archive_url"], archive, lock["java"]["sha256"])
        extracted = stage / "java"
        extracted.mkdir()
        with tarfile.open(archive, "r:gz") as package:
            package.extractall(extracted, filter="data")
        roots = list(extracted.iterdir())
        if len(roots) != 1 or not (roots[0] / "lib/server/libjvm.so").is_file():
            raise ValueError("locked Java archive has an unexpected layout")
        shutil.move(str(roots[0]), destination / "java")
        download(
            lock["bioformats"]["jar_url"],
            destination / "bioformats_package.jar",
            lock["bioformats"]["sha256"],
        )


if __name__ == "__main__":
    install(
        Path(__file__).with_name("reader-runtime.json"), Path("/opt/histoforge-readers")
    )
