"""Read current supporting-file records without weakening executable identity."""

import copy

import pytest
from histoforge.canonical_json import canonical_bytes, canonical_digest
from histoforge.cloud.modules.current import CurrentModules
from histoforge.cloud.modules.files import file_identity
from histoforge.cloud.modules.identity import (
    STORED_MODULE_RECORD_PATH,
    module_content_identity,
    read_stored_module_record,
    validate_stored_module_record,
    write_stored_module_record,
)
from tests.test_module_identity import _build, _module


def _content_digest(record):
    preimage = {
        field: record[field]
        for field in ("recipe_digest", "executable_files", "resolved_artifacts")
    }
    if record.get("additional_files"):
        preimage["additional_files"] = record["additional_files"]
    return canonical_digest(preimage)


@pytest.fixture
def supporting_record(tmp_path):
    directory = _module(tmp_path / "owner" / "module")
    record = _build(directory)
    old_digest = record["content_digest"]
    for name, data in (
        (".gitattributes", b"*.safetensors filter=lfs\n"),
        ("README.md", b"# Module documentation\n"),
    ):
        (directory / name).write_bytes(data)
    record["additional_files"] = [
        file_identity(name, directory / name)
        for name in (".gitattributes", "README.md")
    ]
    record["content_digest"] = _content_digest(record)
    write_stored_module_record(directory, record)
    return directory, record, old_digest, CurrentModules(tmp_path)


def test_current_module_reader_retains_supporting_files_and_exact_model_path(
    supporting_record,
):
    directory, record, old_digest, current = supporting_record
    assert current.read_record("owner/module") == record
    assert read_stored_module_record(directory) == record
    assert (directory / STORED_MODULE_RECORD_PATH).read_bytes() == canonical_bytes(
        record
    )
    assert record["content_digest"] != old_digest
    assert module_content_identity(record)["content_digest"] == record["content_digest"]
    recipe = current.load_recipe("owner/module")
    assert recipe["models"]["model"]["weights"]["file"] == "model.safetensors"
    assert (
        current.artifact_path("owner/module", "model")
        == directory / "model.safetensors"
    )


def test_supporting_record_cannot_claim_the_old_executable_only_digest(
    supporting_record,
):
    _directory, record, old_digest, _current = supporting_record
    changed = copy.deepcopy(record)
    changed["content_digest"] = old_digest
    with pytest.raises(ValueError, match="content_digest"):
        validate_stored_module_record(changed)


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_supporting_file_identity_change_requires_new_content_digest(
    supporting_record,
    field,
):
    _directory, record, _old_digest, _current = supporting_record
    changed = copy.deepcopy(record)
    changed["additional_files"][0][field] = 1 if field == "size_bytes" else "f" * 64
    with pytest.raises(ValueError, match="content_digest"):
        validate_stored_module_record(changed)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.md",
        "/outside.md",
        "MODEL.safetensors",
        "readme.md",
        ".HistoForge/record.json",
        "training-report.json",
    ],
)
def test_supporting_paths_reject_escape_collision_and_reserved_owners(
    supporting_record,
    path,
):
    _directory, record, _old_digest, _current = supporting_record
    changed = copy.deepcopy(record)
    changed["additional_files"].append(
        {"relative_path": path, "size_bytes": 1, "sha256": "a" * 64}
    )
    changed["additional_files"].sort(key=lambda item: item["relative_path"])
    changed["content_digest"] = _content_digest(changed)
    with pytest.raises(ValueError):
        validate_stored_module_record(changed)


def test_supporting_identity_order_is_canonical(supporting_record):
    _directory, record, _old_digest, _current = supporting_record
    changed = copy.deepcopy(record)
    changed["additional_files"].reverse()
    changed["content_digest"] = _content_digest(changed)
    with pytest.raises(ValueError, match="sorted"):
        validate_stored_module_record(changed)


@pytest.mark.parametrize("path", ["recipe.json", "model.safetensors"])
def test_supporting_metadata_does_not_bypass_actual_executable_bytes(
    supporting_record,
    path,
):
    directory, _record, _old_digest, current = supporting_record
    (directory / path).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        if path == "recipe.json":
            current.load_recipe("owner/module")
        else:
            current.artifact_path("owner/module", "model")


def test_supporting_records_still_require_canonical_encoded_bytes(supporting_record):
    directory, _record, _old_digest, _current = supporting_record
    record_path = directory / STORED_MODULE_RECORD_PATH
    record_path.write_bytes(b" " + record_path.read_bytes())
    with pytest.raises(ValueError, match="canonical JSON"):
        read_stored_module_record(directory)
