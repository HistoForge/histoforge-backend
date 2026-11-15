"""Sealed source provenance binds the exact reader view and target raster."""

import copy
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from histoforge.canonical_json import canonical_digest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.architectures.registry import runner_registry
from histoforge.cloud.artifacts.store import ArtifactBytesCorrupt
from histoforge.cloud.artifacts.volume import artifact_volume_gate
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.rasterization.inputs import (
    build_model_raster_input,
    build_raster_frame,
    build_source_view,
)
from histoforge.cloud.rasterization.model_input import (
    prepare_partitioned_model_raster,
    rasterize_model_input,
)
from histoforge.cloud.rasterization.rasterizer import required_source_pixel_rect
from histoforge.cloud.sources import files as source_file_store
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.publication import build_source_artifact_ref


def test_derived_source_reopens_and_samples_the_exact_target_grid(tmp_path):
    files, source, model_input, resolved, pixels = _derived_input(tmp_path)
    containers = ["PNG", "TIFF_FAMILY"]

    assert (
        prepare_partitioned_model_raster(
            model_input, resolved, files, _source_limits(), containers
        ).raster_transform
        == model_input["frame"]["transform"]
    )
    actual = rasterize_model_input(
        model_input, resolved, files, _source_limits(), containers
    )

    assert np.array_equal(actual.pixels, pixels)
    assert actual.real_pixel_mask.all()


@pytest.mark.parametrize("failure_at", ["metadata", "decode", "consumer"])
def test_scoped_source_reader_closes_and_releases_gate_on_every_failure(
    tmp_path, monkeypatch, failure_at
):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    original_open = files._open_validated_source
    closed = []

    def open_reader(*args):
        assert artifact_volume_gate.locked()
        reader, publication = original_open(*args)
        close = reader.close

        def record_close():
            assert artifact_volume_gate.locked()
            closed.append(True)
            close()

        def fail(*_args):
            raise OSError("scoped reader failure")

        monkeypatch.setattr(reader, "close", record_close)
        if failure_at == "metadata":
            monkeypatch.setattr(reader, "selected_level_pixels_to_level_zero", fail)
        elif failure_at == "decode":
            monkeypatch.setattr(reader, "read_rgb_view", fail)
        return reader, publication

    monkeypatch.setattr(files, "_open_validated_source", open_reader)
    if failure_at == "consumer":
        from histoforge.cloud.rasterization import partitioned

        def fail_consumer(*_args, **_kwargs):
            raise OSError("scoped reader failure")

        monkeypatch.setattr(partitioned, "rasterize_srgb_uint8", fail_consumer)
    with pytest.raises(OSError, match="scoped reader failure"):
        rasterize_model_input(model_input, resolved, files, _source_limits(), ["PNG"])
    assert closed == [True]
    assert not artifact_volume_gate.locked()


@pytest.mark.parametrize(
    "prepare",
    [rasterize_model_input, prepare_partitioned_model_raster],
)
@pytest.mark.parametrize(
    "target",
    [
        {"x": 9, "y": 20, "width": 6, "height": 4},
        {"x": 10, "y": 20, "width": 6, "height": 4},
        {"x": 10, "y": 19, "width": 5, "height": 5},
        {"x": 10, "y": 20, "width": 5, "height": 5},
    ],
)
def test_derived_crop_cannot_mark_unavailable_in_slide_pixels_as_real(
    tmp_path, monkeypatch, prepare, target
):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    expanded = _retarget(model_input, **target)
    reads = []

    def reject_read(*_args, **_kwargs):
        reads.append(True)
        raise AssertionError("missing context reached source decoding")

    monkeypatch.setattr(files, "source_view_reader", reject_read)
    with pytest.raises(ValueError, match="omits real target pixels"):
        prepare(expanded, resolved, files, _source_limits(), ["PNG"])
    assert reads == []


@pytest.mark.parametrize(
    "target",
    [
        {"x": -1, "y": -1, "width": 7, "height": 6},
        {"x": 0, "y": 0, "width": 2, "height": 2, "scale": 3},
    ],
)
def test_finite_and_partitioned_derived_slide_overhang_preserve_real_support(
    tmp_path, target
):
    files, _source, model_input, resolved, pixels = _derived_input(
        tmp_path, slide_size=(5, 4), offset=(0, 0)
    )
    expanded = _retarget(model_input, **target)
    finite = rasterize_model_input(expanded, resolved, files, _source_limits(), ["PNG"])
    partitioned = prepare_partitioned_model_raster(
        expanded, resolved, files, _source_limits(), ["PNG"]
    ).read({"x": 0, "y": 0, "width": target["width"], "height": target["height"]})
    np.testing.assert_array_equal(partitioned.pixels, finite.pixels)
    np.testing.assert_array_equal(partitioned.real_pixel_mask, finite.real_pixel_mask)
    if target.get("scale") == 3:
        # The final row's centers lie beyond the physical slide, even though
        # the continuous raster frame extends past both right and bottom edges.
        np.testing.assert_array_equal(
            finite.real_pixel_mask, [[True, True], [False, False]]
        )
    else:
        expected_mask = np.zeros((6, 7), dtype=bool)
        expected_mask[1:-1, 1:-1] = True
        np.testing.assert_array_equal(finite.real_pixel_mask, expected_mask)
        np.testing.assert_array_equal(finite.pixels[1:-1, 1:-1], pixels)
        assert (finite.pixels[~expected_mask] == 255).all()


def test_rounded_derived_level_edge_is_not_treated_as_missing_context(tmp_path):
    files, _source, model_input, resolved, pixels = _derived_input(
        tmp_path, slide_size=(13, 10), offset=(0, 0), source_scale=3
    )
    finite = rasterize_model_input(
        model_input, resolved, files, _source_limits(), ["PNG"]
    )
    partitioned = prepare_partitioned_model_raster(
        model_input, resolved, files, _source_limits(), ["PNG"]
    ).read({"x": 0, "y": 0, "width": 5, "height": 4})

    # The encoded level spans 15 x 12 slide pixels, but the physical image
    # spans 13 x 10. Its final coarse row/column are not normalization support.
    expected_mask = np.zeros((4, 5), dtype=bool)
    expected_mask[:3, :4] = True
    np.testing.assert_array_equal(finite.pixels, pixels)
    np.testing.assert_array_equal(partitioned.pixels, pixels)
    np.testing.assert_array_equal(finite.real_pixel_mask, expected_mask)
    np.testing.assert_array_equal(partitioned.real_pixel_mask, expected_mask)


def test_resolved_source_requires_the_exact_store_projection(tmp_path):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)

    missing_lifetime = copy.deepcopy(resolved)
    missing_lifetime.pop("absolute_expires_at")
    with pytest.raises(ValueError, match="missing or unknown fields"):
        prepare_partitioned_model_raster(
            model_input, missing_lifetime, files, _source_limits(), ["PNG"]
        )

    unknown_field = copy.deepcopy(resolved)
    unknown_field["unexpected"] = True
    with pytest.raises(ValueError, match="missing or unknown fields"):
        prepare_partitioned_model_raster(
            model_input, unknown_field, files, _source_limits(), ["PNG"]
        )


def test_valid_png_rasterization_decodes_the_source_once(tmp_path, monkeypatch):
    files, _source, model_input, resolved, pixels = _derived_input(tmp_path)
    real_decode = cv2.imdecode
    decode_calls = []

    def count_decode(*args, **kwargs):
        decode_calls.append(True)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr("histoforge.cloud.sources.reader.cv2.imdecode", count_decode)

    actual = rasterize_model_input(
        model_input,
        resolved,
        files,
        _source_limits(),
        ["PNG"],
    )

    assert np.array_equal(actual.pixels, pixels)
    assert actual.real_pixel_mask.all()
    assert decode_calls == [True]


@pytest.mark.parametrize("finite", [False, True])
def test_original_raster_reads_absolute_blocks_without_full_decode(
    tmp_path, monkeypatch, finite
):
    files, _source, small_input, resolved, _expected_digest = _original_input(tmp_path)
    source_revision = small_input["frame"]["source_revision"]
    view = build_source_view(
        {
            "artifact": resolved["artifact"],
            "series_index": 0,
            "resolution_level": 0,
            "plane": {"z": 0, "t": 0},
            "pixel_rect": {"x": 0, "y": 0, "width": 512, "height": 384},
            "view_pixels_to_slide": {
                "scale_x": 1,
                "scale_y": 1,
                "translate_x": 0,
                "translate_y": 0,
            },
        }
    )
    frame = build_raster_frame(
        {
            "source_revision": source_revision,
            "region": small_input["frame"]["region"],
            "transform": {
                "raster": {
                    "width": 512,
                    "height": 384,
                    "plane": {"z": 0, "t": 0},
                    "target_mpp": 0.25,
                    "resampling": "BILINEAR",
                    "pixel_format": "SRGB_UINT8",
                },
                "raster_to_slide": {
                    "scale_x": 1,
                    "scale_y": 1,
                    "translate_x": 0,
                    "translate_y": 0,
                },
            },
            "inclusion_mask_raster": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [512, 0], [512, 384], [0, 384], [0, 0]]],
            },
        }
    )
    model_input = build_model_raster_input(view, frame, small_input["rasterizer"])
    expected = rasterize_model_input(
        model_input,
        resolved,
        files,
        _source_limits(),
        ["SLIDE_FILE"],
    )
    limits = _source_limits()
    limits["max_complete_decode_pixels"] = 65_536
    reads = []
    real_reader = files.source_view_reader
    scopes = []

    @contextmanager
    def record_reader(*args, **kwargs):
        scopes.append("open")
        with real_reader(*args, **kwargs) as read_view:

            def read(rectangle):
                assert artifact_volume_gate.locked()
                reads.append(copy.deepcopy(rectangle))
                return read_view(rectangle)

            yield read
        assert not artifact_volume_gate.locked()
        scopes.append("closed")

    monkeypatch.setattr(files, "source_view_reader", record_reader)
    if finite:
        assert (
            prepare_partitioned_model_raster(
                model_input, resolved, files, limits, ["SLIDE_FILE"]
            ).raster_transform
            == model_input["frame"]["transform"]
        )
        assert reads == []
        combined = rasterize_model_input(
            model_input, resolved, files, limits, ["SLIDE_FILE"]
        )
    else:
        prepared = prepare_partitioned_model_raster(
            model_input, resolved, files, limits, ["SLIDE_FILE"]
        )
        assert reads == []
        combined = prepared.read({"x": 0, "y": 0, "width": 512, "height": 384})

    assert np.array_equal(combined.pixels, expected.pixels)
    assert scopes == ["open", "closed"]
    np.testing.assert_array_equal(combined.real_pixel_mask, expected.real_pixel_mask)
    assert reads == [
        {"x": 0, "y": 0, "width": 256, "height": 192},
        {"x": 0, "y": 192, "width": 256, "height": 192},
        {"x": 256, "y": 0, "width": 256, "height": 192},
        {"x": 256, "y": 192, "width": 256, "height": 192},
    ]
    assert all(rect["width"] * rect["height"] <= 65_536 for rect in reads)


def test_prepared_tiff_bounded_reads_preserve_fractional_sampling_and_slide_edges(
    tmp_path, monkeypatch
):
    pixels = np.random.default_rng(0).integers(0, 256, (1031, 1537, 3), dtype=np.uint8)
    files, _source, model_input, resolved, _pixels = _derived_input(
        tmp_path,
        pixels=pixels,
        container="TIFF_FAMILY",
        slide_size=(1537, 1031),
        offset=(0, 0),
    )
    model_input = _retarget(
        model_input, x=-2.3, y=-2.3, width=671, height=451, scale=2.3
    )
    expected = rasterize_model_input(
        model_input, resolved, files, _source_limits(), ["TIFF_FAMILY"]
    )
    # Captured from the unpartitioned reader before finite reads were changed.
    assert hashlib.sha256(expected.pixels.tobytes()).hexdigest() == (
        "525bca69e46c9fd31ffb22e138a1079be94f40959f76cc4b1c63ec13193266ef"
    )
    assert hashlib.sha256(expected.real_pixel_mask.tobytes()).hexdigest() == (
        "2b5af2e8d40f4dafc95d6e93739220524ae9f895055f5fd0745ace925331f0a7"
    )
    limits = _source_limits()
    limits["max_complete_decode_pixels"] = 512 * 512
    reads = []
    real_reader = files.source_view_reader

    @contextmanager
    def record_reader(*args, **kwargs):
        with real_reader(*args, **kwargs) as read_view:

            def read(rectangle):
                reads.append(copy.deepcopy(rectangle))
                return read_view(rectangle)

            yield read

    def forbid_complete_page(*_args, **_kwargs):
        pytest.fail("bounded TIFF reads must not decode the complete image")

    monkeypatch.setattr(files, "source_view_reader", record_reader)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbid_complete_page)
    assert (
        prepare_partitioned_model_raster(
            model_input, resolved, files, limits, ["TIFF_FAMILY"]
        ).raster_transform
        == model_input["frame"]["transform"]
    )
    assert reads == []
    actual = rasterize_model_input(
        model_input, resolved, files, limits, ["TIFF_FAMILY"]
    )

    assert len(reads) > 1
    assert all(rect["width"] * rect["height"] <= 512 * 512 for rect in reads)
    np.testing.assert_array_equal(actual.pixels, expected.pixels)
    np.testing.assert_array_equal(actual.real_pixel_mask, expected.real_pixel_mask)
    assert actual.real_pixel_mask.any()
    assert not actual.real_pixel_mask.all()


@pytest.mark.parametrize(
    "changed", ["dimensions", "selected_level_pixels_to_level_zero"]
)
def test_finite_read_rejects_geometry_changed_since_preflight(
    tmp_path, monkeypatch, changed
):
    files, _source, model_input, resolved, _digest = _original_input(tmp_path)
    real_reader = files.source_view_reader

    @contextmanager
    def changed_geometry(*args, **kwargs):
        with real_reader(*args, **kwargs) as read_view:

            def read(rectangle):
                result = read_view(rectangle)
                field = "width" if changed == "dimensions" else "scale_x"
                result[changed][field] += 1
                return result

            yield read

    monkeypatch.setattr(files, "source_view_reader", changed_geometry)
    with pytest.raises(ValueError, match="source geometry changed"):
        rasterize_model_input(
            model_input, resolved, files, _source_limits(), ["SLIDE_FILE"]
        )
    assert not artifact_volume_gate.locked()


def test_original_tiff_reopens_and_samples_the_exact_target_grid(tmp_path):
    files, _source, model_input, resolved, expected_digest = _original_input(tmp_path)

    actual = rasterize_model_input(
        model_input,
        resolved,
        files,
        _source_limits(),
        ["SLIDE_FILE"],
    )

    assert actual.pixels.shape == (70, 80, 3)
    assert hashlib.sha256(actual.pixels.tobytes()).hexdigest() == expected_digest
    assert actual.real_pixel_mask.all()


def test_self_consistent_wrong_view_transform_is_rejected(tmp_path):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    wrong_view = copy.deepcopy(model_input["source_view"])
    wrong_view.pop("view_digest")
    wrong_view["view_pixels_to_slide"]["translate_x"] += 1
    wrong_input = build_model_raster_input(
        build_source_view(wrong_view),
        model_input["frame"],
        model_input["rasterizer"],
    )

    with pytest.raises(ValueError, match="transform does not match"):
        prepare_partitioned_model_raster(
            wrong_input,
            resolved,
            files,
            _source_limits(),
            ["PNG"],
        )


def test_unpublished_source_level_remains_a_request_validation_error(tmp_path):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    wrong_view = copy.deepcopy(model_input["source_view"])
    wrong_view.pop("view_digest")
    wrong_view["resolution_level"] = 1
    wrong_input = build_model_raster_input(
        build_source_view(wrong_view),
        model_input["frame"],
        model_input["rasterizer"],
    )

    with pytest.raises(ValueError, match="series/resolution level is not validated"):
        prepare_partitioned_model_raster(
            wrong_input,
            resolved,
            files,
            _source_limits(),
            ["PNG"],
        )


def test_oversized_source_view_is_rejected_before_pixel_read(tmp_path, monkeypatch):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    frame = copy.deepcopy(model_input["frame"])
    frame.pop("frame_digest")
    frame.pop("inclusion_mask_digest")
    frame["transform"]["raster"]["width"] = 3
    frame["transform"]["raster"]["height"] = 2
    frame["inclusion_mask_raster"] = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]]],
    }
    oversized = build_model_raster_input(
        model_input["source_view"],
        build_raster_frame(frame),
        model_input["rasterizer"],
    )

    read_calls = []
    decode_calls = []
    real_decode = cv2.imdecode

    def fail_if_read(*_args, **_kwargs):
        read_calls.append(True)
        raise AssertionError("invalid source view reached pixel decoding")

    def count_decode(*args, **kwargs):
        decode_calls.append(True)
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(files, "source_view_reader", fail_if_read)
    monkeypatch.setattr("histoforge.cloud.sources.reader.cv2.imdecode", count_decode)
    with pytest.raises(ValueError, match="smallest bilinear sampling footprint"):
        rasterize_model_input(
            oversized,
            resolved,
            files,
            _source_limits(),
            ["PNG"],
        )
    assert read_calls == []
    assert decode_calls == []


@pytest.mark.parametrize(
    "prepare", [prepare_partitioned_model_raster, rasterize_model_input]
)
def test_png_complete_decode_cap_is_enforced_before_read(
    tmp_path, monkeypatch, prepare
):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    limits = _source_limits()
    limits["max_complete_decode_pixels"] = 19

    def forbid_read(*_args, **_kwargs):
        pytest.fail("oversized PNG must be rejected before pixel decoding")

    monkeypatch.setattr(files, "source_view_reader", forbid_read)
    monkeypatch.setattr("histoforge.cloud.sources.reader.cv2.imdecode", forbid_read)
    with pytest.raises(ArtifactBytesCorrupt) as failure:
        prepare(model_input, resolved, files, limits, ["PNG"])
    assert isinstance(failure.value.__cause__, ValueError)
    assert "complete-decode limits" in str(failure.value.__cause__)


def test_sealed_source_digest_is_rechecked_after_same_size_replacement(
    tmp_path, monkeypatch
):
    files, _source, model_input, resolved, pixels = _derived_input(tmp_path)
    containers = ["PNG"]
    real_sha256_file = source_file_store.sha256_file
    hashed = []

    def count_hash(path):
        hashed.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(source_file_store, "sha256_file", count_hash)

    assert np.array_equal(
        rasterize_model_input(
            model_input, resolved, files, _source_limits(), containers
        ).pixels,
        pixels,
    )
    assert len(hashed) == 1

    altered = (pixels + 2).astype(np.uint8)
    ok, encoded_array = cv2.imencode(".png", cv2.cvtColor(altered, cv2.COLOR_RGB2BGR))
    assert ok
    encoded = encoded_array.tobytes()
    source_path = (
        tmp_path / "sources" / resolved["start_request"]["artifact_id"] / "source.bin"
    )
    assert len(encoded) == source_path.stat().st_size
    replacement = source_path.with_name("replacement.bin")
    replacement.write_bytes(encoded)
    os.replace(replacement, source_path)

    with pytest.raises(ArtifactBytesCorrupt, match="digest"):
        rasterize_model_input(
            model_input, resolved, files, _source_limits(), containers
        )
    assert len(hashed) == 2


def test_raster_source_revision_must_equal_artifact_origin(tmp_path):
    files, _source, model_input, resolved, _pixels = _derived_input(tmp_path)
    frame = copy.deepcopy(model_input["frame"])
    frame.pop("frame_digest")
    frame.pop("inclusion_mask_digest")
    changed_source = copy.deepcopy(frame["source_revision"]["pixel_source"])
    changed_source["credential_free_descriptor_digest"] = "f" * 64
    frame["source_revision"]["pixel_source"] = changed_source
    frame["source_revision"]["image_data_revision"]["image_source_digest"] = (
        canonical_digest(changed_source)
    )
    wrong_input = build_model_raster_input(
        model_input["source_view"],
        build_raster_frame(frame),
        model_input["rasterizer"],
    )

    with pytest.raises(ValueError, match="source revision"):
        prepare_partitioned_model_raster(
            wrong_input,
            resolved,
            files,
            _source_limits(),
            ["PNG"],
        )


def _retarget(model_input, *, x, y, width, height, scale=1):
    """Build a new exact target and its smallest submitted reader view."""
    frame = copy.deepcopy(model_input["frame"])
    frame.pop("frame_digest")
    frame.pop("inclusion_mask_digest")
    frame["transform"]["raster"].update(
        width=width, height=height, target_mpp=0.25 * scale
    )
    frame["transform"]["raster_to_slide"].update(
        translate_x=x, translate_y=y, scale_x=scale, scale_y=scale
    )
    frame["inclusion_mask_raster"] = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [width, 0], [width, height], [0, height], [0, 0]]],
    }
    view = copy.deepcopy(model_input["source_view"])
    view.pop("view_digest")
    submitted = view["pixel_rect"]
    required = required_source_pixel_rect(
        view["view_pixels_to_slide"],
        frame["transform"],
        submitted,
        {
            "width": model_input["frame"]["transform"]["raster"]["width"],
            "height": model_input["frame"]["transform"]["raster"]["height"],
        },
    )
    for axis in ("x", "y"):
        view["view_pixels_to_slide"][f"translate_{axis}"] += (
            required[axis] - submitted[axis]
        ) * view["view_pixels_to_slide"][f"scale_{axis}"]
    view["pixel_rect"] = required
    return build_model_raster_input(
        build_source_view(view), build_raster_frame(frame), model_input["rasterizer"]
    )


def _derived_input(
    tmp_path,
    *,
    slide_size=(100, 80),
    offset=(10, 20),
    source_scale=1,
    pixels=None,
    container="PNG",
):
    if pixels is None:
        pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    height, width, _ = pixels.shape
    if container == "PNG":
        bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        ok, encoded_array = cv2.imencode(".png", bgr)
        assert ok
        encoded = encoded_array.tobytes()
    else:
        buffer = BytesIO()
        tifffile.imwrite(
            buffer,
            pixels,
            bigtiff=True,
            ome=True,
            photometric="rgb",
            tile=(512, 512),
            compression="zlib",
            metadata={"axes": "YXS"},
        )
        encoded = buffer.getvalue()
    source = {
        "source_adapter_id": "qupath.test",
        "credential_free_descriptor_digest": "1" * 64,
        "width": slide_size[0],
        "height": slide_size[1],
        "z_slices": 1,
        "timepoints": 1,
        "calibration": {"mpp_x": 0.25, "mpp_y": 0.25},
    }
    start = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "creation_action_id": str(uuid.uuid4()),
        "artifact_id": str(uuid.uuid4()),
        "container": container,
        "total_bytes": len(encoded),
        "chunk_size_bytes": len(encoded),
        "origin": {
            "type": "derived-region",
            "source": source,
            "source_resolution_level": 0 if source_scale == 1 else 1,
            "plane": {"z": 0, "t": 0},
            "source_rect_at_level": {
                "x": offset[0],
                "y": offset[1],
                "width": width,
                "height": height,
            },
            "artifact_pixels_to_slide": {
                "scale_x": source_scale,
                "scale_y": source_scale,
                "translate_x": offset[0] * source_scale,
                "translate_y": offset[1] * source_scale,
            },
            "writer_build": {
                "version": "test",
                "artifact_digest": "2" * 64,
                "manifest_digest": "3" * 64,
            },
        },
        "required_levels": [
            {"series_index": 0, "resolution_level": 0, "width": width, "height": height}
        ],
    }
    digest = hashlib.sha256(encoded).hexdigest()
    files = SourceFileStore(tmp_path)
    files.publish_source_chunk(start["artifact_id"], 0, encoded, digest)
    manifest = files.seal_source(
        start,
        {"bytes_digest": digest},
        {0: digest},
        _source_limits(),
    )
    artifact = build_source_artifact_ref(start, manifest)
    view = build_source_view(
        {
            "artifact": artifact,
            "series_index": 0,
            "resolution_level": 0,
            "plane": {"z": 0, "t": 0},
            "pixel_rect": {"x": 0, "y": 0, "width": width, "height": height},
            "view_pixels_to_slide": {
                "scale_x": source_scale,
                "scale_y": source_scale,
                "translate_x": offset[0] * source_scale,
                "translate_y": offset[1] * source_scale,
            },
        }
    )
    frame = build_raster_frame(
        {
            "source_revision": {
                "image_data_revision": {
                    "image_id": "test-image",
                    "image_source_digest": canonical_digest(source),
                    "image_data_content_digest": "4" * 64,
                },
                "pixel_source": source,
            },
            "region": {
                "type": "roi",
                "roi_id": "11111111-1111-4111-8111-111111111111",
            },
            "transform": {
                "raster": {
                    "width": width,
                    "height": height,
                    "plane": {"z": 0, "t": 0},
                    "target_mpp": 0.25 * source_scale,
                    "resampling": "BILINEAR",
                    "pixel_format": "SRGB_UINT8",
                },
                "raster_to_slide": {
                    "scale_x": source_scale,
                    "scale_y": source_scale,
                    "translate_x": offset[0] * source_scale,
                    "translate_y": offset[1] * source_scale,
                },
            },
            "inclusion_mask_raster": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [width, 0], [width, height], [0, height], [0, 0]]
                ],
            },
        }
    )
    rasterizer = runner_registry()["definitions"]["dinov3-adapter-cellpose"][
        "capabilities"
    ]["rasterizer"]
    model_input = build_model_raster_input(view, frame, rasterizer)
    resolved = {
        "start_request": start,
        "manifest": manifest,
        "artifact": artifact,
        "absolute_expires_at": "2026-08-23T00:00:00.000Z",
    }
    return files, source, model_input, resolved, pixels


def _original_input(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    contract = json.loads(
        (fixtures / "contracts" / "openslide-original-v1.json").read_text(
            encoding="utf-8"
        )
    )
    encoded = (fixtures / "openslide" / "generic-tiff-v1.tif").read_bytes()
    source_file = contract["source_file"]
    levels = contract["levels"]
    target_case = contract["target_case"]
    window = next(
        value
        for value in contract["window_cases"]
        if value["name"] == target_case["source_window"]
    )
    assert hashlib.sha256(encoded).hexdigest() == source_file["sha256"]
    source = {
        "source_adapter_id": source_file["source_adapter_id"],
        "credential_free_descriptor_digest": "1" * 64,
        "width": levels[0]["width"],
        "height": levels[0]["height"],
        "z_slices": 1,
        "timepoints": 1,
        "calibration": {
            "mpp_x": source_file["mpp_x"],
            "mpp_y": source_file["mpp_y"],
        },
        "content_version": {
            "type": "local-file",
            "size_bytes": len(encoded),
            "last_modified_epoch_millis": 1_700_000_000_000,
            "file_key": "generic-tiff-v1.tif",
        },
    }
    start = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "creation_action_id": str(uuid.uuid4()),
        "artifact_id": str(uuid.uuid4()),
        "container": "SLIDE_FILE",
        "total_bytes": len(encoded),
        "chunk_size_bytes": len(encoded),
        "origin": {
            "type": "original-wsi",
            "source": source,
            "pyramid_series_index": 0,
            "entry_path": "generic-tiff-v1.tif",
            "reader_settings": {
                "levels": [
                    {
                        "resolution_level": level["resolution_level"],
                        "native_resolution_index": level["resolution_level"],
                        "pixels_to_level_zero": {
                            "scale_x": level["qupath_downsample"],
                            "scale_y": level["qupath_downsample"],
                            "translate_x": 0,
                            "translate_y": 0,
                        },
                    }
                    for level in levels
                ],
                "crop_x": 0,
                "crop_y": 0,
                "background_rgb": [0, 0, 0],
                "tile_width": 128,
                "tile_height": 128,
            },
            "level_zero_pixels_to_slide": {
                "scale_x": 1,
                "scale_y": 1,
                "translate_x": 0,
                "translate_y": 0,
            },
        },
        "required_levels": [
            {
                "series_index": 0,
                "resolution_level": level["resolution_level"],
                "width": level["width"],
                "height": level["height"],
            }
            for level in levels
        ],
    }
    digest = hashlib.sha256(encoded).hexdigest()
    files = SourceFileStore(tmp_path)
    files.publish_source_chunk(start["artifact_id"], 0, encoded, digest)
    manifest = files.seal_source(
        start,
        {"bytes_digest": digest},
        {0: digest},
        _source_limits(),
    )
    artifact = build_source_artifact_ref(start, manifest)
    view = build_source_view(
        {
            "artifact": artifact,
            "series_index": 0,
            "resolution_level": window["resolution_level"],
            "plane": {"z": 0, "t": 0},
            "pixel_rect": window["pixel_rect"],
            "view_pixels_to_slide": target_case["source_view_to_slide"],
        }
    )
    frame = build_raster_frame(
        {
            "source_revision": {
                "image_data_revision": {
                    "image_id": "test-image",
                    "image_source_digest": canonical_digest(source),
                    "image_data_content_digest": "4" * 64,
                },
                "pixel_source": source,
            },
            "region": {
                "type": "roi",
                "roi_id": "11111111-1111-4111-8111-111111111111",
            },
            "transform": target_case["target_grid"],
            "inclusion_mask_raster": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [80, 0], [80, 70], [0, 70], [0, 0]]],
            },
        }
    )
    rasterizer = runner_registry()["definitions"]["dinov3-adapter-cellpose"][
        "capabilities"
    ]["rasterizer"]
    model_input = build_model_raster_input(view, frame, rasterizer)
    resolved = {
        "start_request": start,
        "manifest": manifest,
        "artifact": artifact,
        "absolute_expires_at": "2026-08-23T00:00:00.000Z",
    }
    return files, source, model_input, resolved, target_case["target_rgb_sha256"]


def _source_limits():
    return DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
