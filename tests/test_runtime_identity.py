"""Runtime build identities bind source, locks, environment, and policy."""

from types import SimpleNamespace

import pytest
from histoforge.build_identity import build_manifest_digest, file_set_digest
from histoforge.cloud import runtime_identity


def _install_inputs(monkeypatch):
    source = (("histoforge/a.py", b"source"),)
    locks = (("histoforge/cloud/runtime/lock", b"locked"),)
    inventory = {
        "schema_version": 1,
        "python": "3.10.18",
        "implementation": "CPython",
        "platform": "linux",
        "machine": "x86_64",
        "libc": {"name": "glibc", "version": "2.31"},
        "packages": {"modal": "1.4.3"},
    }
    monkeypatch.setattr(runtime_identity, "_source_files", lambda: source)
    monkeypatch.setattr(runtime_identity, "_lock_files", lambda: locks)
    monkeypatch.setattr(runtime_identity, "_runtime_inventory", lambda: inventory)
    monkeypatch.setattr(
        runtime_identity,
        "_PACKAGE_ROOT",
        _FakePackageRoot(b"app-config"),
    )
    runtime_identity._runtime_build.cache_clear()
    return source, locks


class _FakePackageRoot:
    def __init__(self, app_content):
        self.app_content = app_content

    def __truediv__(self, _segment):
        return self

    def read_bytes(self):
        return self.app_content


def test_runtime_identity_is_deterministic_and_manifest_backed(monkeypatch):
    source, locks = _install_inputs(monkeypatch)

    first = runtime_identity.runtime_build("gateway", {"schema_version": 1})
    second = runtime_identity.runtime_build("gateway", {"schema_version": 1})

    assert first is second
    assert first.identity["artifact_digest"] == file_set_digest([*source, *locks])
    assert first.identity["manifest_digest"] == build_manifest_digest(first.manifest)
    assert first.manifest["component"] == "gateway"


def test_behavior_policy_changes_identity_without_changing_artifact(monkeypatch):
    _install_inputs(monkeypatch)

    first = runtime_identity.runtime_build("runner", {"policy": "a"})
    second = runtime_identity.runtime_build("runner", {"policy": "b"})

    assert first.identity["artifact_digest"] == second.identity["artifact_digest"]
    assert first.identity["manifest_digest"] != second.identity["manifest_digest"]


def test_deployment_artifact_check_does_not_use_laptop_runtime_inventory(monkeypatch):
    source, locks = _install_inputs(monkeypatch)
    monkeypatch.setattr(
        runtime_identity,
        "_runtime_inventory",
        lambda: pytest.fail("local OS/packages cannot identify the cloud artifact"),
    )
    assert runtime_identity.bundle_artifact_digest() == file_set_digest(
        [*source, *locks]
    )


def test_runtime_inventory_captures_every_normalized_distribution(
    monkeypatch, tmp_path
):
    distributions = [
        SimpleNamespace(metadata={"Name": "Package.One"}, version="1.0"),
        SimpleNamespace(metadata={"Name": "package-one"}, version="1.0"),
        SimpleNamespace(metadata={"Name": "Other_Package"}, version="2.5"),
    ]
    for distribution in distributions:
        distribution.locate_file = lambda _path: tmp_path
    monkeypatch.setattr(
        runtime_identity.importlib.metadata,
        "distributions",
        lambda: distributions,
    )
    selected = {
        "package-one": distributions[0],
        "other-package": distributions[2],
    }
    monkeypatch.setattr(
        runtime_identity.importlib.metadata,
        "distribution",
        selected.__getitem__,
    )
    monkeypatch.setattr(runtime_identity.platform, "python_version", lambda: "3.10.18")
    monkeypatch.setattr(
        runtime_identity.platform, "python_implementation", lambda: "CPython"
    )
    monkeypatch.setattr(runtime_identity.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        runtime_identity.platform, "libc_ver", lambda: ("glibc", "2.31")
    )
    monkeypatch.setattr(runtime_identity.sys, "platform", "linux")

    assert runtime_identity._runtime_inventory()["packages"] == {
        "other-package": "2.5",
        "package-one": "1.0",
    }


def test_runtime_inventory_uses_import_visible_distribution(monkeypatch, tmp_path):
    distributions = [
        SimpleNamespace(metadata={"Name": "Package.One"}, version="1.0"),
        SimpleNamespace(metadata={"Name": "package-one"}, version="2.0"),
    ]
    distributions[0].locate_file = lambda _path: tmp_path / "active"
    distributions[1].locate_file = lambda _path: tmp_path / "shadowed"
    monkeypatch.setattr(
        runtime_identity.importlib.metadata,
        "distributions",
        lambda: distributions,
    )
    monkeypatch.setattr(
        runtime_identity.importlib.metadata,
        "distribution",
        lambda name: distributions[0] if name == "package-one" else None,
    )

    assert runtime_identity._runtime_inventory()["packages"]["package-one"] == "1.0"


def _write_metadata(root, version, suffix):
    """Create real metadata entries without installing anything in user state."""

    directory = root / f"sample_package-{version}{suffix}"
    directory.mkdir(parents=True)
    filename = "PKG-INFO" if suffix.endswith("egg-info") else "METADATA"
    (directory / filename).write_text(
        f"Metadata-Version: 2.1\nName: sample-package\nVersion: {version}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("versions", [("1.0", "2.0"), ("2.0", "1.0")])
@pytest.mark.parametrize("suffix", ["-py3.10.egg-info", ".dist-info"])
def test_same_root_version_conflicts_fail_regardless_of_creation_order(
    tmp_path, monkeypatch, versions, suffix
):
    for version in versions:
        _write_metadata(tmp_path, version, suffix)
    monkeypatch.setattr(runtime_identity.sys, "path", [str(tmp_path)])

    with pytest.raises(RuntimeError, match="conflicting installed versions"):
        runtime_identity._runtime_inventory()


def test_identical_version_metadata_is_not_a_version_conflict(tmp_path, monkeypatch):
    _write_metadata(tmp_path, "1.0", "-py3.10.egg-info")
    _write_metadata(tmp_path, "1.0", ".dist-info")
    monkeypatch.setattr(runtime_identity.sys, "path", [str(tmp_path)])

    assert runtime_identity._runtime_inventory()["packages"] == {
        "sample-package": "1.0"
    }


def test_later_metadata_root_can_be_shadowed(tmp_path, monkeypatch):
    active, shadowed = tmp_path / "active", tmp_path / "shadowed"
    _write_metadata(active, "1.0", ".dist-info")
    _write_metadata(shadowed, "2.0", ".dist-info")
    _write_metadata(shadowed, "3.0", ".dist-info")
    monkeypatch.setattr(runtime_identity.sys, "path", [str(active), str(shadowed)])

    assert runtime_identity._runtime_inventory()["packages"] == {
        "sample-package": "1.0"
    }
