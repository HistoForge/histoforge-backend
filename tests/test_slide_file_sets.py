"""Original slide layout admission is independent of any image decoder."""

import hashlib
import stat
import struct
import zipfile

import pytest
from histoforge.cloud.sources.file_sets import (
    MAX_ZIP_DIRECTORY_BYTES,
    extract_slide_files,
    relative_slide_path,
    validate_member_paths,
    validate_members,
)
from histoforge.cloud.sources.identity import validate_source_content_version
from histoforge.cloud.sources.original import (
    BIOFORMATS_ADAPTER,
    OPENSLIDE_ADAPTER,
    validate_reader_settings,
)


def _package(tmp_path, files=None, **options):
    files = files or {
        "slide.mrxs": b"entry",
        "slide/empty": b"",
        "slide/part.dat": b"pixels",
    }
    members = [
        {
            "path": name,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in files.items()
    ]
    members = validate_members(members)
    package = tmp_path / "source.zip"
    with zipfile.ZipFile(package, "w", **options) as output:
        for member in members:
            output.writestr(member["path"], files[member["path"]])
    return package, members


def _levels():
    return [
        {
            "resolution_level": 0,
            "native_resolution_index": 0,
            "pixels_to_level_zero": {
                "scale_x": 1,
                "scale_y": 1,
                "translate_x": 0,
                "translate_y": 0,
            },
        }
    ]


def test_extract_retains_exact_names_empty_members_and_bytes(tmp_path):
    package, members = _package(tmp_path)
    destination = tmp_path / "members"
    extract_slide_files(package, destination, members)
    assert sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ) == [member["path"] for member in members]
    for member in members:
        assert (
            hashlib.sha256((destination / member["path"]).read_bytes()).hexdigest()
            == member["sha256"]
        )


@pytest.mark.parametrize(
    "path",
    [
        "/entry",
        "../entry",
        "a/../b",
        "a//b",
        "a/./b",
        "a\\b",
        "C:entry",
        "a\x00b",
        "e\u0301.tif",
    ],
)
def test_unsafe_or_renamed_path_rejected(path):
    with pytest.raises(ValueError):
        relative_slide_path(path)


@pytest.mark.parametrize(
    "paths", [["a", "a"], ["a", "A"], ["a", "a/b"], ["a/b", "A"], ["ß", "SS"]]
)
def test_alias_and_prefix_conflicts_rejected(paths):
    with pytest.raises(ValueError):
        validate_member_paths(paths)


def test_wire_order_is_utf8_not_java_utf16():
    members = [
        {"path": name, "size_bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        for name in ["\U00010000", "\ue000", "a"]
    ]
    assert [m["path"] for m in validate_members(members)] == [
        "a",
        "\ue000",
        "\U00010000",
    ]


def test_size_and_hash_rejected(tmp_path):
    package, members = _package(tmp_path)
    members[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        extract_slide_files(package, tmp_path / "members", members)


def test_compressed_member_rejected(tmp_path):
    package, members = _package(tmp_path, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ValueError, match="uncompressed"):
        extract_slide_files(package, tmp_path / "members", members)


def test_symlink_rejected(tmp_path):
    package, members = _package(tmp_path, {"slide.tif": b"entry"})
    with zipfile.ZipFile(package, "w") as output:
        link = zipfile.ZipInfo("slide.tif")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, b"entry")
    with pytest.raises(ValueError, match="regular"):
        extract_slide_files(package, tmp_path / "members", members)


def test_local_header_name_mismatch_rejected(tmp_path):
    package, members = _package(tmp_path, {"slide.tif": b"entry"})
    data = package.read_bytes()
    package.write_bytes(data.replace(b"slide.tif", b"other.tif", 1))
    with pytest.raises(ValueError, match="invalid original slide ZIP") as failure:
        extract_slide_files(package, tmp_path / "members", members)
    assert isinstance(failure.value.__cause__, zipfile.BadZipFile)


@pytest.mark.parametrize(
    "field,value", [(10, 65534), (12, MAX_ZIP_DIRECTORY_BYTES + 1)]
)
def test_directory_allocation_bound_precedes_zipfile(
    tmp_path, monkeypatch, field, value
):
    package, members = _package(tmp_path)
    data = bytearray(package.read_bytes())
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<H" if field == 10 else "<L", data, end + field, value)
    package.write_bytes(data)
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *_: pytest.fail("unbounded directory allocation")
    )
    with pytest.raises(ValueError, match="directory"):
        extract_slide_files(package, tmp_path / "members", members)


def test_source_fileset_version_allows_empty_companion_not_single_file():
    member = {"path": "slide/empty", "size_bytes": 0, "last_modified_epoch_millis": 1}
    assert validate_source_content_version(
        {"type": "local-file-set", "members": [member]}
    )["members"] == [member]
    with pytest.raises(ValueError):
        validate_source_content_version(
            {"type": "local-file", "size_bytes": 0, "last_modified_epoch_millis": 1}
        )


def test_zip64_cannot_override_a_bounded_classic_directory(tmp_path, monkeypatch):
    package, members = _package(tmp_path)
    data = package.read_bytes()
    end = data.rfind(b"PK\x05\x06")
    directory_size = struct.unpack_from("<L", data, end + 12)[0]
    directory_offset = struct.unpack_from("<L", data, end + 16)[0]
    zip64 = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        len(members),
        len(members),
        MAX_ZIP_DIRECTORY_BYTES + 1,
        directory_offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, end, 1)
    classic = bytearray(data[end:])
    # This plausible, nonsentinel classic footer passes its own extent check.
    struct.pack_into("<L", classic, 12, directory_size + 76)
    package.write_bytes(data[:end] + zip64 + locator + classic)
    monkeypatch.setattr(
        zipfile, "ZipFile", lambda *_: pytest.fail("unbounded ZIP64 allocation")
    )
    with pytest.raises(ValueError, match="directory"):
        extract_slide_files(package, tmp_path / "members", members)


def test_closed_bioformats_settings_and_no_native_remapping():
    assert validate_reader_settings({"levels": _levels()}, BIOFORMATS_ADAPTER) == {
        "levels": _levels()
    }
    with pytest.raises(ValueError, match="fields"):
        validate_reader_settings(
            {"levels": _levels(), "java_args": ["-Xmx8g"]}, BIOFORMATS_ADAPTER
        )
    levels = _levels()
    levels[0]["native_resolution_index"] = 1
    with pytest.raises(ValueError, match="remapped"):
        validate_reader_settings({"levels": levels}, BIOFORMATS_ADAPTER)


def test_openslide_settings_bind_tile_grid_and_color():
    settings = {
        "levels": _levels(),
        "crop_x": 4,
        "crop_y": 7,
        "background_rgb": [255, 255, 255],
        "tile_width": 240,
        "tile_height": 256,
    }
    assert validate_reader_settings(settings, OPENSLIDE_ADAPTER) == settings
    settings["background_rgb"][-1] = 256
    with pytest.raises(ValueError, match="uint8"):
        validate_reader_settings(settings, OPENSLIDE_ADAPTER)
