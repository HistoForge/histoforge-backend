import json
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from histoforge.build_identity import (
    build_identity_bytes,
    build_manifest_digest,
    file_set_digest,
    file_set_manifest,
    parse_build_manifest,
    parse_software_build_identity,
    software_build_identity,
)
from histoforge.canonical_json import canonical_bytes, canonical_digest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "contracts" / "build-identities-v1.json"
RUNTIME = ROOT / "histoforge" / "cloud" / "runtime"
PACKAGE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")
CONDA_URL = re.compile(
    r"https://conda\.anaconda\.org/conda-forge/(?:linux-64|noarch)/"
    r"[^#]+#[0-9a-f]{32}"
)


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _normalized_package(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def test_shared_file_set_fixtures_have_exact_order_bytes_and_digest():
    for case in _fixture()["fileSetCases"]:
        files = [(item["path"], item["utf8"].encode()) for item in case["files"]]
        manifest = file_set_manifest(files)
        assert manifest == case["manifest"], case["name"]
        assert canonical_bytes(manifest).decode() == case["canonical"], case["name"]
        assert file_set_digest(files) == case["sha256"], case["name"]


def test_shared_build_manifest_and_identity_fixtures_are_exact():
    for case in _fixture()["manifestCases"]:
        manifest = parse_build_manifest(case["value"])
        assert canonical_bytes(manifest).decode() == case["canonical"], case["name"]
        assert build_manifest_digest(manifest) == case["sha256"], case["name"]

        identity = parse_software_build_identity(case["identity"])
        assert identity == software_build_identity(
            version=identity["version"],
            artifact_digest=identity["artifact_digest"],
            manifest=manifest,
        )
        assert build_identity_bytes(identity).decode() == case["identityCanonical"]
        assert canonical_digest(identity) == case["identitySha256"]


def test_shared_invalid_build_values_fail_closed():
    fixture = _fixture()
    for case in fixture["invalidManifestCases"]:
        with pytest.raises(ValueError):
            parse_build_manifest(case["value"])
    for case in fixture["invalidIdentityCases"]:
        with pytest.raises(ValueError):
            parse_software_build_identity(case["value"])
    for path in fixture["invalidPaths"]:
        with pytest.raises(ValueError):
            file_set_manifest([(path, b"content")])


def test_file_set_rejects_duplicate_paths_and_non_bytes():
    with pytest.raises(ValueError, match="duplicate path"):
        file_set_manifest([("same", b"one"), ("same", b"two")])
    with pytest.raises(ValueError, match="must be bytes"):
        file_set_manifest([("file", "text")])


def test_modal_pip_input_and_hash_lock_pin_the_same_complete_package_set():
    input_packages = _requirements(RUNTIME / "requirements-modal.in")
    locked_packages = _requirements(RUNTIME / "requirements-modal.lock")
    assert locked_packages == input_packages

    current = None
    hashes = 0
    seen = set()
    for line in (
        (RUNTIME / "requirements-modal.lock").read_text(encoding="utf-8").splitlines()
    ):
        match = PACKAGE.match(line)
        if match:
            if current is not None:
                assert hashes > 0, f"{current} has no artifact hash"
            current = _normalized_package(match.group(1))
            assert current not in seen
            seen.add(current)
            hashes = 0
        elif "--hash=sha256:" in line:
            hashes += 1
    assert current is not None and hashes > 0, f"{current} has no artifact hash"


def test_modal_conda_lock_is_sorted_explicit_and_hash_pinned():
    lines = (RUNTIME / "conda-linux-64.lock").read_text(encoding="utf-8").splitlines()
    urls = [line for line in lines if line.startswith("https://")]
    assert "@EXPLICIT" in lines
    assert urls == sorted(urls)
    assert len(urls) == len(set(urls))
    assert urls and all(CONDA_URL.fullmatch(url) for url in urls)

    joined = "\n".join(urls)
    for required in (
        "/python-3.10.18-",
        "/numpy-2.2.6-",
        "/scipy-1.15.2-",
        "/scikit-image-0.25.2-",
        "/opencv-5.0.0-",
        "/tifffile-2025.5.10-",
        "/imagecodecs-2025.3.30-",
        "/vigra-1.12.3-",
        "/nifty-1.2.4-",
        "/zarr-2.18.3-",
    ):
        assert required in joined
    assert "/python-elf-" not in joined


def test_modal_image_build_consumes_locks_and_training_provenance(monkeypatch):
    calls = []

    class FakeImage:
        @classmethod
        def from_registry(cls, *args, **kwargs):
            calls.append(("base", args, kwargs))
            return cls()

        def _record(self, name, *args, **kwargs):
            calls.append((name, args, kwargs))
            return self

        def apt_install(self, *args, **kwargs):
            return self._record("apt", *args, **kwargs)

        def dockerfile_commands(self, *args, **kwargs):
            return self._record("dockerfile", *args, **kwargs)

        def env(self, *args, **kwargs):
            return self._record("env", *args, **kwargs)

        def pip_install_from_requirements(self, *args, **kwargs):
            return self._record("pip-lock", *args, **kwargs)

        def run_commands(self, *args, **kwargs):
            return self._record("run", *args, **kwargs)

        def add_local_python_source(self, *args, **kwargs):
            return self._record("source", *args, **kwargs)

        def add_local_file(self, *args, **kwargs):
            return self._record("file", *args, **kwargs)

    class FakeVolume:
        @staticmethod
        def from_name(*args, **kwargs):
            return object()

    class FakeSecret:
        @staticmethod
        def from_name(*args, **kwargs):
            calls.append(("secret", args, kwargs))
            return object()

    fake_modal = SimpleNamespace(
        App=lambda name: object(),
        Image=FakeImage,
        Secret=FakeSecret,
        Volume=FakeVolume,
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    namespace = runpy.run_path(str(ROOT / "histoforge" / "cloud" / "_app.py"))

    assert namespace["IMAGE_BUILDER_VERSION"] == "2025.06"
    assert calls[0] == (
        "base",
        ("mambaorg/micromamba:2.1.1-debian12-slim",),
        {"setup_dockerfile_commands": ["USER root"]},
    )
    runs = [call for call in calls if call[0] == "run"]
    assert runs[0][1] == (
        "micromamba create --yes --name histoforge "
        "--file /root/histoforge/cloud/runtime/conda-linux-64.lock",
    )
    assert "assert sys.prefix == '/opt/conda/envs/histoforge'" in runs[1][1][0]
    pip = next(call for call in calls if call[0] == "pip-lock")
    environments = [call[1][0] for call in calls if call[0] == "env"]
    assert environments == [
        {"ENV_NAME": "histoforge", "MAMBA_DOCKERFILE_ACTIVATE": "1"},
        {"SSL_CERT_FILE": "/opt/conda/envs/histoforge/ssl/cert.pem"},
    ]
    assert next(call for call in calls if call[0] == "dockerfile")[1] == (
        "ENV PATH=/opt/conda/envs/histoforge/bin:$PATH",
    )
    assert Path(pip[1][0]) == RUNTIME / "requirements-modal.lock"
    assert pip[2]["extra_options"] == "--require-hashes --no-deps"
    copied = [call for call in calls if call[0] == "file"]
    assert copied == [
        (
            "file",
            (
                RUNTIME / "conda-linux-64.lock",
                "/root/histoforge/cloud/runtime/conda-linux-64.lock",
            ),
            {"copy": True},
        ),
        (
            "file",
            (
                RUNTIME / "requirements-modal.lock",
                "/root/histoforge/cloud/runtime/requirements-modal.lock",
            ),
            {"copy": True},
        ),
        (
            "file",
            (
                RUNTIME / "reader-runtime.json",
                "/root/histoforge/cloud/runtime/reader-runtime.json",
            ),
            {"copy": True},
        ),
        (
            "file",
            (
                RUNTIME / "install_readers.py",
                "/root/histoforge/cloud/runtime/install_readers.py",
            ),
            {"copy": True},
        ),
        (
            "file",
            (
                ROOT
                / "histoforge"
                / "_vendor"
                / "histoforge_models"
                / "provenance.json",
                "/root/histoforge/_vendor/histoforge_models/provenance.json",
            ),
            {"copy": True},
        ),
        (
            "file",
            (
                ROOT / "histoforge" / "_vendor" / "histoforge_models" / "LICENSE",
                "/root/histoforge/_vendor/histoforge_models/LICENSE",
            ),
            {"copy": True},
        ),
        (
            "file",
            (
                ROOT / "histoforge" / "_vendor" / "histoforge_models" / "NOTICE",
                "/root/histoforge/_vendor/histoforge_models/NOTICE",
            ),
            {"copy": True},
        ),
    ]
    image_calls = [
        call[0]
        for call in calls
        if call[0] in {"base", "dockerfile", "env", "pip-lock", "run", "file", "source"}
    ]
    assert image_calls == [
        "base",
        "file",
        "run",
        "env",
        "dockerfile",
        "env",
        "pip-lock",
        "run",
        "file",
        "file",
        "file",
        "run",
        "file",
        "file",
        "file",
        "source",
    ]
    assert runs[2][1] == ("python /root/histoforge/cloud/runtime/install_readers.py",)
    secret = next(call for call in calls if call[0] == "secret")
    assert secret[1] == ("histoforge-huggingface",)
    assert secret[2] == {"required_keys": ["HF_TOKEN"]}
    assert not any(call[0] in {"apt", "pip"} for call in calls)


def test_local_development_runtime_is_exactly_pinned():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"modal==1.4.3"' in project
    assert '"toml==0.10.2"' in project
    assert '"fastapi==0.139.0"' in project
    assert '"fastremap==1.20.0"' in project
    assert '"fill-voids==2.1.2"' in project
    assert '"numpy==2.2.6"' in project
    assert '"opencv-python-headless==5.0.0.93"' in project
    assert '"openslide-python==1.4.6"' in project
    assert "\"openslide-bin==4.0.1.2; sys_platform != 'darwin'\"" in project
    assert '"rasterio==1.4.4"' in project
    assert '"safetensors==0.8.0"' in project
    assert "segment-anything" not in project
    assert '"scikit-image==0.25.2"' in project
    assert '"scipy==1.15.2"' in project
    assert '"timm==1.0.27"' in project
    assert '"tifffile==2025.5.10"' in project
    assert '"torch==2.12.1"' in project
    assert '"imagecodecs==2025.3.30"' in project
    assert '"numcodecs==0.13.1"' in project
    assert '"zarr==2.18.3"' in project
    assert '"ruff==0.15.12"' in project
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "      - modal==1.4.3" in environment
    assert "      - toml==0.10.2" in environment
    assert "      - fastremap==1.20.0" in environment
    assert "      - fill-voids==2.1.2" in environment
    assert "      - numpy==2.2.6" in environment
    assert "      - opencv-python-headless==5.0.0.93" in environment
    assert "      - openslide-python==1.4.6" in environment
    assert "  - openslide=4.0.1" in environment
    assert "      - openslide-bin==" not in environment
    assert "      - rasterio==1.4.4" in environment
    assert "      - safetensors==0.8.0" in environment
    assert "segment-anything" not in environment
    assert "      - scikit-image==0.25.2" in environment
    assert "      - scipy==1.15.2" in environment
    assert "      - timm==1.0.27" in environment
    assert "      - tifffile==2025.5.10" in environment
    assert "      - torch==2.12.1" in environment
    assert "      - imagecodecs==2025.3.30" in environment
    assert "      - numcodecs==0.13.1" in environment
    assert "      - zarr==2.18.3" in environment
    assert "      - ruff==0.15.12" in environment


def _requirements(path):
    packages = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PACKAGE.match(line)
        if match:
            name = _normalized_package(match.group(1))
            assert name not in packages
            packages[name] = match.group(2)
    assert packages
    return packages
