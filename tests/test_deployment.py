"""Private, atomic publication of the local CLI-to-plugin handoff."""

import json
import os
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import Event

import pytest
from histoforge import _deployment

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "contracts"
    / "local-deployment-credential-v1.json"
)


@pytest.fixture
def deployment_path(tmp_path, monkeypatch):
    path = tmp_path / ".histoforge" / "deployment.json"
    monkeypatch.setattr(_deployment, "DIR", path.parent)
    monkeypatch.setattr(_deployment, "PATH", path)
    return path


def _write():
    return _deployment.write(
        _deployment.build(
            deployment_id="deployment-a",
            gateway_url="https://gateway.example",
            modal_profile="research-lab",
            modal_app_name="histoforge-gateway",
            auth_revision=3,
            token="secret-token",
            updated_at="2026-08-15T12:34:56.789Z",
        )
    )


def _fixture(version=1):
    path = FIXTURE.with_name(f"local-deployment-credential-v{version}.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("version", [1, 2])
def test_shared_credential_fixture_round_trips_exactly(version):
    fixture = _fixture(version)
    assert fixture["schemaVersion"] == version
    for case in fixture["validCases"]:
        assert _deployment.parse(case["value"]) == case["value"], case["name"]


@pytest.mark.parametrize("version", [1, 2])
def test_shared_invalid_credential_mutations_fail_closed(version):
    fixture = _fixture(version)
    base = fixture["validCases"][0]["value"]
    for case in fixture["invalidMutations"]:
        value = deepcopy(base)
        parent = value
        for segment in case["path"][:-1]:
            parent = parent[segment]
        field = case["path"][-1]
        if case["operation"] == "remove":
            del parent[field]
        else:
            parent[field] = case["value"]
        with pytest.raises(ValueError):
            _deployment.parse(value)


def test_verified_location_builds_and_publishes_v2(deployment_path):
    value = _deployment.build(
        deployment_id="deployment-a",
        gateway_url="https://gateway.example",
        modal_profile="research-lab",
        modal_environment="analysis-qa",
        modal_app_name="histoforge-gateway",
        auth_revision=7,
        token="test-token",
    )
    assert value["schema_version"] == 2
    assert value["modal_environment"] == "analysis-qa"
    assert _deployment.write(value) == _deployment.read() == value


def test_write_publishes_complete_private_utf8_json(deployment_path):
    data = _write()

    encoded = deployment_path.read_bytes()
    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == data
    assert _deployment.read() == data
    if os.name == "posix":
        assert stat.S_IMODE(deployment_path.stat().st_mode) == 0o600
    assert list(deployment_path.parent.glob(".deployment.json.*.tmp")) == []


def test_write_replaces_an_existing_deployment(deployment_path):
    deployment_path.parent.mkdir()
    deployment_path.write_text("old deployment\n", encoding="utf-8")

    data = _write()

    assert json.loads(deployment_path.read_bytes()) == data
    if os.name == "posix":
        assert stat.S_IMODE(deployment_path.stat().st_mode) == 0o600


def test_replace_failure_preserves_old_file_and_cleans_temp(
    deployment_path, monkeypatch
):
    deployment_path.parent.mkdir()
    deployment_path.write_text("old deployment\n", encoding="utf-8")

    def fail_replace(source, target):
        assert source.parent == target.parent == deployment_path.parent
        assert source.read_text(encoding="utf-8").endswith("\n")
        if os.name == "posix":
            assert stat.S_IMODE(source.stat().st_mode) == 0o600
        raise OSError("atomic replace unavailable")

    monkeypatch.setattr(_deployment.os, "replace", fail_replace)

    with pytest.raises(OSError, match="atomic replace unavailable"):
        _write()

    assert deployment_path.read_text(encoding="utf-8") == "old deployment\n"
    assert list(deployment_path.parent.glob(".deployment.json.*.tmp")) == []


def test_flush_failure_preserves_old_file_and_cleans_temp(deployment_path, monkeypatch):
    deployment_path.parent.mkdir()
    deployment_path.write_text("old deployment\n", encoding="utf-8")

    def fail_fsync(_fd):
        raise OSError("forced flush failure")

    monkeypatch.setattr(_deployment.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="forced flush failure"):
        _write()

    assert deployment_path.read_text(encoding="utf-8") == "old deployment\n"
    assert list(deployment_path.parent.glob(".deployment.json.*.tmp")) == []


def test_publication_exposes_only_complete_generations(tmp_path):
    """Use the user-home filesystem in the matrix, an isolated temp root otherwise."""

    generation_a = '{"generation":"A","payload":"' + "α" * 262_144 + '"}'
    generation_b = '{"generation":"B","payload":"' + "β" * 262_145 + '"}'
    allowed_generations = {generation_a, generation_b}
    matrix_run = os.environ.get("HISTOFORGE_LOCAL_PERSISTENCE_MATRIX") == "true"
    with tempfile.TemporaryDirectory(
        prefix=".histoforge-deployment-", dir=Path.home() if matrix_run else tmp_path
    ) as directory:
        path = Path(directory) / "deployment.json"
        _deployment._write_private_text_atomically(path, generation_a)
        observed = {path.read_text(encoding="utf-8")}

        if os.name == "nt":
            # Python's Windows reader does not share delete access, while the real
            # plugin reader does. The extension matrix exercises that concurrent
            # Java-reader boundary; this leg verifies the Python writer on NTFS.
            for index in range(32):
                _deployment._write_private_text_atomically(
                    path, generation_a if index % 2 == 0 else generation_b
                )
                observed.add(path.read_text(encoding="utf-8"))
        else:
            writer_started = Event()
            reader_observed = Event()

            def replace_repeatedly() -> None:
                _deployment._write_private_text_atomically(path, generation_b)
                writer_started.set()
                if not reader_observed.wait(timeout=30):
                    raise TimeoutError("concurrent deployment reader did not start")
                for index in range(31):
                    _deployment._write_private_text_atomically(
                        path, generation_a if index % 2 == 0 else generation_b
                    )

            with ThreadPoolExecutor(max_workers=1) as executor:
                writer = executor.submit(replace_repeatedly)
                assert writer_started.wait(timeout=30)
                try:
                    assert not writer.done()
                    value = path.read_text(encoding="utf-8")
                    assert value in allowed_generations
                    observed.add(value)
                    concurrent_reads = 1
                finally:
                    reader_observed.set()

                while not writer.done() or concurrent_reads < 32:
                    value = path.read_text(encoding="utf-8")
                    assert value in allowed_generations
                    observed.add(value)
                    concurrent_reads += 1
                writer.result(timeout=60)

            assert concurrent_reads >= 32

        observed.add(path.read_text(encoding="utf-8"))
        assert observed == allowed_generations
        if os.name == "posix":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert list(path.parent.glob(".deployment.json.*.tmp")) == []
