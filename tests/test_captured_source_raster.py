"""Bounded uploads preserve complete-image pixels without introducing false edges."""

import copy
import hashlib
import io
import uuid
from contextlib import contextmanager

import cv2
import numpy as np
import pytest
import tifffile
from histoforge.canonical_json import canonical_digest
from histoforge.cloud._config import DEPLOYMENT_CONFIG
from histoforge.cloud.architectures.registry import runner_registry
from histoforge.cloud.protocol_versions import SOURCE_ARTIFACT_SCHEMA_VERSION
from histoforge.cloud.rasterization.captured_source import (
    captured_source_footprint,
    prepare_captured_source_raster,
)
from histoforge.cloud.rasterization.inputs import (
    build_model_raster_input,
    build_raster_frame,
    build_source_view,
)
from histoforge.cloud.rasterization.model_input import (
    prepare_partitioned_model_raster,
    rasterize_model_input,
)
from histoforge.cloud.rasterization.rasterizer import (
    InsufficientSourceContext,
    rasterize_srgb_uint8,
)
from histoforge.cloud.sources.capture import (
    build_captured_source_view,
    build_source_capture,
    captured_image_origin,
    captured_view_geometry,
)
from histoforge.cloud.sources.files import SourceFileStore
from histoforge.cloud.sources.publication import build_source_artifact_ref


@pytest.mark.parametrize("container", ["PNG", "TIFF_FAMILY"])
@pytest.mark.parametrize(
    ("offset", "level_scale"), [((13, 7), 1.0), ((510013, 240007), 1.3)]
)
def test_bounded_uploads_equal_the_existing_complete_prepared_image(
    tmp_path, container, offset, level_scale
):
    view, target, source_pixels = _case(offset, level_scale)
    files = SourceFileStore(tmp_path)
    complete_rect = view["capture"]["source_rect_at_level"]
    complete = _seal(files, view, complete_rect, source_pixels, container)
    expected = _existing_finite_result(view, target, complete, files)
    combined = np.zeros_like(expected.pixels)
    support = np.zeros_like(expected.real_pixel_mask)

    for output in _partitions():
        required = captured_source_footprint(view, target, output)
        bounded_pixels = _crop(source_pixels, required, complete_rect)
        sealed = _seal(files, view, required, bounded_pixels, container)
        limits = _limits()
        if container == "TIFF_FAMILY":
            limits["max_complete_decode_pixels"] = 512 * 512
        raster = prepare_captured_source_raster(
            view,
            target,
            output,
            _binding(view, required, sealed),
            sealed,
            files,
            limits,
            [container],
        )
        actual = raster.read(output)
        ys = slice(output["y"], output["y"] + output["height"])
        xs = slice(output["x"], output["x"] + output["width"])
        combined[ys, xs] = actual.pixels
        support[ys, xs] = actual.real_pixel_mask

    np.testing.assert_array_equal(combined, expected.pixels)
    np.testing.assert_array_equal(support, expected.real_pixel_mask)


def test_full_file_can_supply_multiple_units_without_changing_the_view(tmp_path):
    view, target, source_pixels = _case()
    files = SourceFileStore(tmp_path)
    rectangle = view["capture"]["source_rect_at_level"]
    sealed = _seal(files, view, rectangle, source_pixels)
    binding = _binding(view, rectangle, sealed)
    expected = _existing_finite_result(view, target, sealed, files)
    for output in _partitions():
        raster = prepare_captured_source_raster(
            view, target, output, binding, sealed, files, _limits(), ["PNG"]
        )
        actual = raster.read(output)
        np.testing.assert_array_equal(actual.pixels, _crop(expected.pixels, output))


def test_nonminimal_model_view_is_rejected_before_file_access(tmp_path, monkeypatch):
    view, target, pixels = _case()
    envelope = view["capture"]["source_rect_at_level"]
    broad_view = build_captured_source_view(
        view["capture"], {**envelope, "x": 0, "y": 0}
    )
    files = SourceFileStore(tmp_path)
    sealed = _seal(files, view, envelope, pixels)
    monkeypatch.setattr(
        files, "source_level_metadata", lambda *_args: pytest.fail("premature IO")
    )
    with pytest.raises(ValueError, match="smallest bilinear sampling footprint"):
        prepare_captured_source_raster(
            broad_view,
            target,
            _partitions()[0],
            _binding(view, envelope, sealed),
            sealed,
            files,
            _limits(),
            ["PNG"],
        )


@pytest.mark.parametrize("level_extent", [4, 5])
def test_pyramid_rounding_cannot_turn_missing_real_pixels_into_white(
    tmp_path, monkeypatch, level_extent
):
    original, _, _ = _case()
    preimage = copy.deepcopy(original["capture"])
    preimage.pop("capture_digest")
    preimage["source"].update(width=13, height=13)
    preimage["source_level"].update(
        resolution_level=1, width=level_extent, height=level_extent
    )
    preimage["source_level_pixels_to_level_zero"] = _mapping(3.0)
    envelope = {"x": 0, "y": 0, "width": level_extent, "height": level_extent}
    preimage["source_rect_at_level"] = envelope
    view = build_captured_source_view(build_source_capture(preimage), envelope)
    output = {"x": 0, "y": 0, "width": 4, "height": 4}
    target = {
        "raster": {
            "width": 4,
            "height": 4,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 0.75,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": _mapping(3.0, 2.0, 2.0),
    }
    pixels = np.full((level_extent, level_extent, 3), 80, dtype=np.uint8)
    files = SourceFileStore(tmp_path)
    sealed = _seal(files, view, envelope, pixels)

    def prepare():
        return prepare_captured_source_raster(
            view,
            target,
            output,
            _binding(view, envelope, sealed),
            sealed,
            files,
            _limits(),
            ["PNG"],
        )

    # The last target center is 12.5: physically real, but absent from a level
    # rounded down to four 3-pixel cells. Rounded-up five-cell levels cover it.
    if level_extent == 4:
        with pytest.raises(ValueError, match="omits real target pixels"):
            _existing_finite_result(view, target, sealed, files)
        monkeypatch.setattr(
            files, "source_level_metadata", lambda *_args: pytest.fail("premature IO")
        )
        with pytest.raises(ValueError, match="omits real target pixels"):
            prepare()
    else:
        expected = _existing_finite_result(view, target, sealed, files)
        actual = prepare().read(output)
        np.testing.assert_array_equal(actual.pixels, expected.pixels)
        assert actual.real_pixel_mask.all()


def test_large_window_offset_does_not_rebase_float32_sampling(tmp_path):
    view, _, _ = _case((0, 0))
    preimage = copy.deepcopy(view["capture"])
    preimage.pop("capture_digest")
    preimage["source"].update(width=200000, height=10)
    preimage["source_level"].update(width=200000, height=10)
    preimage["source_rect_at_level"].update(width=200000, height=10)
    capture = build_source_capture(preimage)
    view = build_captured_source_view(
        capture, {"x": 0, "y": 0, "width": 200000, "height": 10}
    )
    target = {
        "raster": {
            "width": 150000,
            "height": 7,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 0.25 * 1.00197,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": _mapping(1.00197, 0.137, 0.27),
    }
    view = build_captured_source_view(capture, captured_source_footprint(view, target))
    pixels = np.random.default_rng(17).integers(0, 256, (10, 200000, 3), dtype=np.uint8)
    files = SourceFileStore(tmp_path)
    complete = _seal(files, view, capture["source_rect_at_level"], pixels)
    output = {"x": 130000, "y": 2, "width": 200, "height": 4}
    expected = _existing_finite_result(view, target, complete, files, output=output)
    required = captured_source_footprint(view, target, output)
    bounded_pixels = _crop(pixels, required)
    bounded = _seal(files, view, required, bounded_pixels)
    actual = prepare_captured_source_raster(
        view,
        target,
        output,
        _binding(view, required, bounded),
        bounded,
        files,
        _limits(),
        ["PNG"],
    ).read(output)
    np.testing.assert_array_equal(actual.pixels, expected.pixels)
    np.testing.assert_array_equal(actual.real_pixel_mask, expected.real_pixel_mask)

    # This tempting implementation samples the same file in crop-local float32
    # coordinates. It really changes pixels; the large-offset test detects it.
    rebased = rasterize_srgb_uint8(
        bounded_pixels,
        bounded["start_request"]["origin"]["artifact_pixels_to_slide"],
        target,
        {
            "min_x": 0,
            "min_y": 0,
            "max_x": required["width"],
            "max_y": required["height"],
        },
        output,
    )
    assert not np.array_equal(rebased, expected.pixels)


def test_tiff_source_reads_remain_bounded_inside_a_large_uploaded_image(
    tmp_path, monkeypatch
):
    view, target, pixels = _case(multiplier=32)
    files = SourceFileStore(tmp_path)
    required = captured_source_footprint(view, target)
    bounded_pixels = _crop(pixels, required, view["capture"]["source_rect_at_level"])
    sealed = _seal(files, view, required, bounded_pixels, "TIFF_FAMILY")
    complete = _seal(
        files, view, view["capture"]["source_rect_at_level"], pixels, "TIFF_FAMILY"
    )
    expected = _existing_finite_result(view, target, complete, files)
    limits = _limits()
    limits["max_complete_decode_pixels"] = 512 * 512
    output = {
        "x": 0,
        "y": 0,
        "width": target["raster"]["width"],
        "height": target["raster"]["height"],
    }
    reads = []
    original_reader = files.source_view_reader
    scopes = []

    @contextmanager
    def reader(*args):
        scopes.append("open")
        with original_reader(*args) as read_view:

            def read(rectangle):
                reads.append(rectangle)
                return read_view(rectangle)

            yield read
        scopes.append("closed")

    monkeypatch.setattr(files, "source_view_reader", reader)
    raster = prepare_captured_source_raster(
        view,
        target,
        output,
        _binding(view, required, sealed),
        sealed,
        files,
        limits,
        ["TIFF_FAMILY"],
    )
    assert reads == []
    np.testing.assert_array_equal(raster.read(output).pixels, expected.pixels)
    assert scopes == ["open", "closed"]
    assert len(reads) > 1
    assert all(rect["width"] * rect["height"] <= 512 * 512 for rect in reads)


def test_interior_bilinear_neighbor_missing_from_upload_fails_before_read(
    tmp_path, monkeypatch
):
    view, target, source_pixels = _case()
    files = SourceFileStore(tmp_path)
    output = _partitions()[0]
    required = captured_source_footprint(view, target, output)
    missing_edge = {**required, "width": required["width"] - 1}
    bounded = _crop(
        source_pixels, missing_edge, view["capture"]["source_rect_at_level"]
    )
    sealed = _seal(files, view, missing_edge, bounded)
    monkeypatch.setattr(
        files, "source_view_reader", lambda *_args: pytest.fail("premature read")
    )
    with pytest.raises(InsufficientSourceContext, match="omits required"):
        prepare_captured_source_raster(
            view,
            target,
            output,
            _binding(view, missing_edge, sealed),
            sealed,
            files,
            _limits(),
            ["PNG"],
        )


def test_a_unit_file_cannot_supply_a_different_processing_rectangle(tmp_path):
    view, target, source_pixels = _case()
    files = SourceFileStore(tmp_path)
    required = captured_source_footprint(view, target, _partitions()[0])
    bounded = _crop(source_pixels, required, view["capture"]["source_rect_at_level"])
    sealed = _seal(files, view, required, bounded)
    raster = prepare_captured_source_raster(
        view,
        target,
        _partitions()[0],
        _binding(view, required, sealed),
        sealed,
        files,
        _limits(),
        ["PNG"],
    )
    with pytest.raises(InsufficientSourceContext, match="omits required"):
        raster.read(_partitions()[-1])


def test_true_image_edges_keep_white_padding_and_real_support(tmp_path):
    view, target, source_pixels = _case((0, 0), 1.3, edge=True)
    files = SourceFileStore(tmp_path)
    envelope = view["capture"]["source_rect_at_level"]
    full = _seal(files, view, envelope, source_pixels)
    expected = _existing_finite_result(view, target, full, files)
    assert not expected.real_pixel_mask.all()
    for output in _partitions():
        required = captured_source_footprint(view, target, output)
        bounded = _crop(source_pixels, required, envelope)
        sealed = _seal(files, view, required, bounded)
        actual = prepare_captured_source_raster(
            view,
            target,
            output,
            _binding(view, required, sealed),
            sealed,
            files,
            _limits(),
            ["PNG"],
        ).read(output)
        np.testing.assert_array_equal(actual.pixels, _crop(expected.pixels, output))
        np.testing.assert_array_equal(
            actual.real_pixel_mask, _crop(expected.real_pixel_mask, output)
        )


@pytest.mark.parametrize("mismatch", ["reference", "placement", "container", "scale"])
def test_uploaded_pixels_must_match_capture_and_runner(tmp_path, mismatch):
    view, target, source_pixels = _case()
    files = SourceFileStore(tmp_path)
    required = captured_source_footprint(view, target, _partitions()[0])
    pixels = _crop(source_pixels, required, view["capture"]["source_rect_at_level"])
    sealed = _seal(files, view, required, pixels)
    binding = _binding(view, required, sealed)
    containers = ["PNG"]
    if mismatch == "reference":
        binding["artifact"] = {**binding["artifact"], "bytes_digest": "0" * 64}
    elif mismatch == "placement":
        # A larger declared rectangle would cover the request but cannot relabel
        # the already-sealed pixels as belonging to another source location.
        binding["source_rect_at_level"] = {
            **required,
            "x": required["x"] - 1,
            "width": required["width"] + 1,
        }
    elif mismatch == "container":
        containers = ["TIFF_FAMILY"]
    else:
        target = copy.deepcopy(target)
        target["raster"]["target_mpp"] *= 2
    with pytest.raises(ValueError):
        prepare_captured_source_raster(
            view,
            target,
            _partitions()[0],
            binding,
            sealed,
            files,
            _limits(),
            containers,
        )


def _case(offset=(13, 7), level_scale=1.0, *, edge=False, multiplier=1):
    width, height = (30, 18) if edge else (96 * multiplier, 80 * multiplier)
    level_width, level_height = (width, height) if edge else (600000, 300000)
    source = {
        "source_adapter_id": "qupath.test",
        "credential_free_descriptor_digest": "1" * 64,
        "width": int(level_width * level_scale),
        "height": int(level_height * level_scale),
        "z_slices": 1,
        "timepoints": 1,
        "calibration": {"mpp_x": 0.25, "mpp_y": 0.25},
    }
    capture = build_source_capture(
        {
            "source": source,
            "plane": {"z": 0, "t": 0},
            "source_level": {
                "series_index": 0,
                "resolution_level": 0 if level_scale == 1 else 1,
                "width": level_width,
                "height": level_height,
            },
            "source_level_pixels_to_level_zero": _mapping(level_scale),
            "level_zero_pixels_to_slide": _mapping(1.0),
            "source_rect_at_level": {
                "x": offset[0],
                "y": offset[1],
                "width": width,
                "height": height,
            },
            "writer_build": {
                "version": "test",
                "artifact_digest": "2" * 64,
                "manifest_digest": "3" * 64,
            },
        }
    )
    scale = 0.71 * level_scale
    target = {
        "raster": {
            "width": 49 * multiplier,
            "height": 29 * multiplier,
            "plane": {"z": 0, "t": 0},
            "target_mpp": 0.25 * scale,
            "resampling": "BILINEAR",
            "pixel_format": "SRGB_UINT8",
        },
        "raster_to_slide": _mapping(
            scale,
            -1.1 if edge else (offset[0] + 5) * level_scale + 0.13,
            -1.2 if edge else (offset[1] + 5) * level_scale + 0.17,
        ),
    }
    view = build_captured_source_view(
        capture, {"x": 0, "y": 0, "width": width, "height": height}
    )
    for _ in range(8):
        required = captured_source_footprint(view, target)
        rectangle = {
            **required,
            "x": required["x"] - offset[0],
            "y": required["y"] - offset[1],
        }
        if rectangle == view["pixel_rect"]:
            break
        view = build_captured_source_view(capture, rectangle)
    else:
        pytest.fail("fixture view did not converge")
    pixels = np.random.default_rng(13).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    return view, target, pixels


def _seal(files, view, rectangle, pixels, container="PNG"):
    if container == "PNG":
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        assert ok
        encoded = encoded.tobytes()
    else:
        buffer = io.BytesIO()
        tifffile.imwrite(
            buffer,
            pixels,
            photometric="rgb",
            compression="zlib",
            tile=(512, 512),
            bigtiff=True,
            ome=True,
            metadata={"axes": "YXS"},
        )
        encoded = buffer.getvalue()
    start = {
        "schema_version": SOURCE_ARTIFACT_SCHEMA_VERSION,
        "creation_action_id": str(uuid.uuid4()),
        "artifact_id": str(uuid.uuid4()),
        "container": container,
        "total_bytes": len(encoded),
        "chunk_size_bytes": min(len(encoded), _limits()["max_chunk_bytes"]),
        "origin": captured_image_origin(view["capture"], rectangle),
        "required_levels": [
            {
                "series_index": 0,
                "resolution_level": 0,
                "width": rectangle["width"],
                "height": rectangle["height"],
            }
        ],
    }
    digest = hashlib.sha256(encoded).hexdigest()
    chunks = {}
    for index, offset in enumerate(range(0, len(encoded), start["chunk_size_bytes"])):
        chunk = encoded[offset : offset + start["chunk_size_bytes"]]
        chunks[index] = hashlib.sha256(chunk).hexdigest()
        files.publish_source_chunk(start["artifact_id"], index, chunk, chunks[index])
    manifest = files.seal_source(start, {"bytes_digest": digest}, chunks, _limits())
    return {
        "start_request": start,
        "manifest": manifest,
        "artifact": build_source_artifact_ref(start, manifest),
        "absolute_expires_at": "2026-10-06T00:00:00.000Z",
    }


def _existing_finite_result(view, target, sealed, files, *, output=None):
    capture = view["capture"]
    source = capture["source"]
    _, mapping = captured_view_geometry(view)
    uploaded_view = build_source_view(
        {
            "artifact": sealed["artifact"],
            "series_index": 0,
            "resolution_level": 0,
            "plane": capture["plane"],
            "pixel_rect": view["pixel_rect"],
            "view_pixels_to_slide": mapping,
        }
    )
    width, height = target["raster"]["width"], target["raster"]["height"]
    frame = build_raster_frame(
        {
            "source_revision": {
                "pixel_source": source,
                "image_data_revision": {
                    "image_id": "test",
                    "image_source_digest": canonical_digest(source),
                    "image_data_content_digest": "4" * 64,
                },
            },
            "region": {"type": "roi", "roi_id": "11111111-1111-4111-8111-111111111111"},
            "transform": target,
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
    rasterize = (
        rasterize_model_input if output is None else prepare_partitioned_model_raster
    )
    result = rasterize(
        build_model_raster_input(uploaded_view, frame, rasterizer),
        sealed,
        files,
        _limits(),
        ["PNG", "TIFF_FAMILY"],
    )
    return result if output is None else result.read(output)


def _partitions():
    return [
        {"x": x, "y": y, "width": width, "height": height}
        for y, height in ((0, 15), (15, 14))
        for x, width in ((0, 25), (25, 24))
    ]


def _crop(pixels, rectangle, origin=None):
    origin = origin or {"x": 0, "y": 0}
    x, y = rectangle["x"] - origin["x"], rectangle["y"] - origin["y"]
    return pixels[y : y + rectangle["height"], x : x + rectangle["width"]].copy()


def _binding(view, rectangle, sealed):
    return {
        "capture_digest": view["capture"]["capture_digest"],
        "artifact": sealed["artifact"],
        "source_rect_at_level": rectangle,
    }


def _mapping(scale, x=0.0, y=0.0):
    return {"scale_x": scale, "scale_y": scale, "translate_x": x, "translate_y": y}


def _limits():
    return DEPLOYMENT_CONFIG.source_artifact_limits.wire_value()
