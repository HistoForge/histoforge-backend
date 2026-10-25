"""Validated prepared pixels or the captured original-slide reader.

The reader owns encoded-container inspection and bounded RGB window reads. It
does not own upload state, manifest publication, target-grid sampling, or model
preprocessing. Callers publish a manifest only after this reader's discovered
levels satisfy the accepted source-artifact request.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openslide
import openslide.lowlevel as openslide_lowlevel
import tifffile
import zarr

from histoforge.cloud._config import validate_source_artifact_limits
from histoforge.cloud.sources.contract import (
    validate_source_artifact_start_request,
)
from histoforge.cloud.sources.identity import validate_int_rect
from histoforge.cloud.sources.original import BIOFORMATS_ADAPTER
from histoforge.cloud.sources.publication import (
    build_source_artifact_manifest,
)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_IHDR_SIZE = 33
_DERIVED_TIFF_COMPRESSIONS = {"ADOBE_DEFLATE", "DEFLATE"}
_OPENSLIDE_LOCAL_ADAPTER_ID = "qupath.openslide.local"
_OPENSLIDE_BOUNDS_KEYS = (
    openslide.PROPERTY_NAME_BOUNDS_X,
    openslide.PROPERTY_NAME_BOUNDS_Y,
    openslide.PROPERTY_NAME_BOUNDS_WIDTH,
    openslide.PROPERTY_NAME_BOUNDS_HEIGHT,
)


class SourceReader:
    """One validated immutable source file and its bounded RGB read handle.

    Instances are intentionally not shared across concurrent model inputs. A
    TIFF/OpenSlide instances keep metadata and file state open until ``close``.
    Each window materializes only intersecting source tiles through its pinned
    reader path.
    """

    def __init__(self, path: str | Path, start_request: Any, limits: Any):
        self._path = Path(path)
        self._start = validate_source_artifact_start_request(start_request)
        self._limits = validate_source_artifact_limits(limits)
        self._png: np.ndarray | None = None
        self._tiff: tifffile.TiffFile | None = None
        self._openslide: openslide.OpenSlide | None = None
        self._bioformats = None
        self._openslide_downsamples: tuple[float, ...] = ()
        self._openslide_bounds = (0, 0)
        self._openslide_tile_size = (0, 0)
        self._openslide_background = (0, 0, 0)
        self._levels: list[Any] = []
        self._validated_levels: tuple[dict[str, int], ...] = ()
        self._level_mappings: dict[tuple[int, int], dict[str, float]] = {}

        try:
            self._open()
        except Exception:
            self.close()
            raise

    @property
    def validated_levels(self) -> list[dict[str, int]]:
        """Return detached reader-discovered levels in canonical order."""

        return [dict(level) for level in self._validated_levels]

    @classmethod
    def inspect_level(
        cls,
        path: str | Path,
        start_request: Any,
        limits: Any,
        series_index: Any,
        resolution_level: Any,
    ) -> dict[str, Any]:
        """Inspect level geometry without decoding PNG pixels.

        TIFF/OpenSlide inspection opens only reader metadata. PNG inspection
        validates the exact header and accepted source declaration; the
        ordinary reader still performs the full lossless decode when pixels
        are requested.
        """

        source_path = Path(path)
        start = validate_source_artifact_start_request(start_request)
        accepted_limits = validate_source_artifact_limits(limits)
        if start["container"] != "PNG":
            with cls(source_path, start, accepted_limits) as reader:
                return _level_metadata(
                    reader._validated_levels,
                    reader._level_mappings,
                    series_index,
                    resolution_level,
                )

        _validate_source_file(source_path, start, accepted_limits)
        levels, mappings = _inspect_png(source_path, start, accepted_limits)
        _validate_required_levels(start, levels)
        return _level_metadata(levels, mappings, series_index, resolution_level)

    def selected_level_pixels_to_level_zero(
        self, series_index: Any, resolution_level: Any
    ) -> dict[str, float]:
        """Return the reader-validated selected-level mapping for one level."""

        key = self._level_key(series_index, resolution_level)
        return dict(self._level_mappings[key])

    def read_rgb_view(
        self, series_index: Any, resolution_level: Any, pixel_rect: Any
    ) -> np.ndarray:
        """Read one contained half-open source rectangle as detached RGB uint8."""

        key = self._level_key(series_index, resolution_level)
        level = self._validated_levels[key[1]]
        rectangle = validate_int_rect(pixel_rect)
        if (
            rectangle["x"] + rectangle["width"] > level["width"]
            or rectangle["y"] + rectangle["height"] > level["height"]
        ):
            raise ValueError("source pixel_rect lies outside its selected level")
        if (
            rectangle["width"] * rectangle["height"]
            > self._limits["max_complete_decode_pixels"]
        ):
            raise ValueError("source pixel_rect exceeds complete-decode limits")

        y0 = rectangle["y"]
        y1 = y0 + rectangle["height"]
        x0 = rectangle["x"]
        x1 = x0 + rectangle["width"]
        if self._png is not None:
            return self._png[y0:y1, x0:x1].copy()
        if self._openslide is not None:
            return self._read_openslide_view(key[1], rectangle)
        if self._bioformats is not None:
            return self._bioformats.read_rgb(key[1], rectangle)

        level_series = self._levels[key[1]]
        store = level_series.aszarr()
        try:
            array = zarr.open(store, mode="r")
            result = np.asarray(array[y0:y1, x0:x1, :])
        finally:
            store.close()
        if result.dtype != np.uint8 or result.shape != (
            rectangle["height"],
            rectangle["width"],
            3,
        ):
            raise ValueError("source reader returned an invalid RGB window")
        return result.copy()

    def close(self) -> None:
        """Close the decoder before the caller releases the storage gate."""

        if self._tiff is not None:
            self._tiff.close()
            self._tiff = None
        if self._openslide is not None:
            self._openslide.close()
            self._openslide = None
        if self._bioformats is not None:
            decoder, self._bioformats = self._bioformats, None
            decoder.close()

    def __enter__(self) -> SourceReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _open(self) -> None:
        _validate_source_file(self._path, self._start, self._limits)

        if self._start["container"] == "PNG":
            self._open_png()
        elif self._start["origin"]["type"] == "original-wsi":
            if (
                self._start["origin"]["source"]["source_adapter_id"]
                == BIOFORMATS_ADAPTER
            ):
                self._open_bioformats()
            else:
                self._open_openslide()
        else:
            self._open_tiff()

        _validate_required_levels(self._start, self._validated_levels)

    def _open_png(self) -> None:
        levels, mappings = _inspect_png(self._path, self._start, self._limits)
        level = levels[0]
        width = level["width"]
        height = level["height"]
        with self._path.open("rb") as source:
            with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as encoded_file:
                encoded = np.frombuffer(encoded_file, dtype=np.uint8)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
                del encoded
        if (
            decoded is None
            or decoded.dtype != np.uint8
            or decoded.shape != (height, width, 3)
        ):
            raise ValueError("PNG must decode to one uint8 RGB plane")
        self._png = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        self._validated_levels = levels
        self._level_mappings = mappings

    def _open_tiff(self) -> None:
        try:
            self._tiff = tifffile.TiffFile(self._path)
        except (OSError, tifffile.TiffFileError) as error:
            raise ValueError("invalid TIFF-family source artifact") from error

        tiff = self._tiff
        series_count = len(tiff.series)
        if series_count == 0 or series_count > self._limits["max_image_series"]:
            raise ValueError("TIFF series count exceeds source-artifact limits")
        for series in tiff.series:
            if len(series.levels) > self._limits["max_pyramid_levels_per_series"]:
                raise ValueError(
                    "TIFF pyramid level count exceeds source-artifact limits"
                )

        origin = self._start["origin"]
        if origin["type"] != "derived-region":
            raise ValueError("original TIFF must use the pinned OpenSlide reader")
        if not tiff.is_bigtiff or not tiff.is_ome:
            raise ValueError("derived TIFF must be OME-BigTIFF")
        if series_count != 1 or len(tiff.series[0].levels) != 1:
            raise ValueError("derived TIFF must contain one series and one level")
        selected_series = 0
        compression = _DERIVED_TIFF_COMPRESSIONS
        required_tile = (512, 512)

        self._levels = list(tiff.series[selected_series].levels)
        validated = []
        for resolution_level, level in enumerate(self._levels):
            width, height = _validate_tiff_level(
                level,
                self._limits,
                compression,
                required_tile,
            )
            validated.append(
                {
                    "series_index": selected_series,
                    "resolution_level": resolution_level,
                    "width": width,
                    "height": height,
                }
            )
        self._validated_levels = tuple(validated)

        self._level_mappings[(0, 0)] = _identity_transform()

    def _open_openslide(self) -> None:
        origin = self._start["origin"]
        source = origin["source"]
        if source["source_adapter_id"] != _OPENSLIDE_LOCAL_ADAPTER_ID:
            raise ValueError(
                "original source was not admitted by the OpenSlide adapter"
            )
        if source["z_slices"] != 1 or source["timepoints"] != 1:
            raise ValueError("OpenSlide original must contain one z=0, t=0 plane")
        if origin["pyramid_series_index"] != 0:
            raise ValueError("OpenSlide original must use pyramid series zero")

        try:
            self._openslide = openslide.OpenSlide(self._path)
        except (openslide.OpenSlideError, OSError) as error:
            raise ValueError("invalid OpenSlide source artifact") from error
        slide = self._openslide
        properties = slide.properties
        settings = origin["reader_settings"]
        background = _openslide_background(properties)
        if list(background) != settings["background_rgb"]:
            raise ValueError("OpenSlide background differs from the captured view")
        if slide.level_count > self._limits["max_pyramid_levels_per_series"]:
            raise ValueError(
                "OpenSlide pyramid level count exceeds source-artifact limits"
            )

        full_dimensions = tuple(
            (int(width), int(height)) for width, height in slide.level_dimensions
        )
        downsamples = _qupath_level_downsamples(full_dimensions)
        full_dimensions = full_dimensions[: len(downsamples)]
        bounds_x, bounds_y, bounds_width, bounds_height, cropped = _qupath_bounds(
            properties, full_dimensions[0]
        )
        captured_bounds = (
            settings["crop_x"],
            settings["crop_y"],
            source["width"],
            source["height"],
        )
        if captured_bounds == (0, 0, *full_dimensions[0]):
            bounds_x, bounds_y, cropped = 0, 0, False
        elif captured_bounds != (bounds_x, bounds_y, bounds_width, bounds_height):
            raise ValueError("OpenSlide crop differs from the captured view")
        validated = []
        mappings: dict[tuple[int, int], dict[str, float]] = {}
        retained_downsamples = []
        for resolution_level, ((width, height), downsample) in enumerate(
            zip(full_dimensions, downsamples, strict=True)
        ):
            if cropped:
                width = int(bounds_width / downsample)
                height = int(bounds_height / downsample)
            if (
                width <= 0
                or height <= 0
                or width > self._limits["max_level_width_px"]
                or height > self._limits["max_level_height_px"]
            ):
                raise ValueError(
                    "OpenSlide level dimensions exceed source-artifact limits"
                )
            validated.append(
                {
                    "series_index": 0,
                    "resolution_level": resolution_level,
                    "width": width,
                    "height": height,
                }
            )
            mappings[(0, resolution_level)] = {
                "scale_x": downsample,
                "scale_y": downsample,
                "translate_x": 0.0,
                "translate_y": 0.0,
            }
            retained_downsamples.append(downsample)

        if (validated[0]["width"], validated[0]["height"]) != (
            source["width"],
            source["height"],
        ):
            raise ValueError(
                "OpenSlide level zero differs from the captured source identity"
            )
        self._validated_levels = tuple(validated)
        self._level_mappings = mappings
        self._require_captured_mappings()
        self._openslide_downsamples = tuple(retained_downsamples)
        self._openslide_bounds = (bounds_x, bounds_y)
        self._openslide_tile_size = (settings["tile_width"], settings["tile_height"])
        if (
            math.prod(self._openslide_tile_size)
            > self._limits["max_complete_decode_pixels"]
        ):
            raise ValueError("OpenSlide tile dimensions exceed single-decode limits")
        self._openslide_background = background

    def _open_bioformats(self) -> None:
        # Importing the shared reader or recording runtime identity does not start Java.
        from histoforge.cloud.sources.bioformats import BioFormatsReader

        origin = self._start["origin"]
        decoder = BioFormatsReader(self._path, origin, self._limits)
        self._bioformats = decoder
        self._validated_levels = tuple(decoder.levels)
        downsamples = _qupath_level_downsamples(
            tuple(decoder.dimensions), decoder.format
        )
        self._level_mappings = {
            (origin["pyramid_series_index"], index): {
                "scale_x": downsample,
                "scale_y": downsample,
                "translate_x": 0.0,
                "translate_y": 0.0,
            }
            for index, downsample in enumerate(downsamples)
        }
        self._require_captured_mappings()

    def _require_captured_mappings(self) -> None:
        origin = self._start["origin"]
        captured = origin["reader_settings"]["levels"]
        if len(captured) != len(self._validated_levels) or len(captured) != len(
            self._level_mappings
        ):
            raise ValueError("native reader levels differ from the captured view")
        for level in captured:
            key = (origin["pyramid_series_index"], level["resolution_level"])
            if self._level_mappings[key] != level["pixels_to_level_zero"]:
                raise ValueError("native level mapping differs from the captured view")

    def _read_openslide_view(
        self, resolution_level: int, rectangle: Mapping[str, int]
    ) -> np.ndarray:
        slide = self._openslide
        if slide is None:
            raise RuntimeError("OpenSlide reader is closed")
        downsample = self._openslide_downsamples[resolution_level]
        bounds_x, bounds_y = self._openslide_bounds
        tile_width, tile_height = self._openslide_tile_size
        level = self._validated_levels[resolution_level]
        result = np.empty((rectangle["height"], rectangle["width"], 3), dtype=np.uint8)
        tile_min_x = rectangle["x"] // tile_width * tile_width
        tile_min_y = rectangle["y"] // tile_height * tile_height
        tile_max_x = min(
            level["width"],
            _ceil_div(rectangle["x"] + rectangle["width"], tile_width) * tile_width,
        )
        tile_max_y = min(
            level["height"],
            _ceil_div(rectangle["y"] + rectangle["height"], tile_height) * tile_height,
        )
        for tile_y in range(tile_min_y, tile_max_y, tile_height):
            for tile_x in range(tile_min_x, tile_max_x, tile_width):
                width = min(tile_width, level["width"] - tile_x)
                height = min(tile_height, level["height"] - tile_y)
                location = (
                    _java_round(tile_x * downsample) + bounds_x,
                    _java_round(tile_y * downsample) + bounds_y,
                )
                tile = _read_premultiplied_openslide_rgb(
                    slide,
                    location,
                    resolution_level,
                    width,
                    height,
                    self._openslide_background,
                )
                source_x0 = max(rectangle["x"], tile_x)
                source_y0 = max(rectangle["y"], tile_y)
                source_x1 = min(rectangle["x"] + rectangle["width"], tile_x + width)
                source_y1 = min(rectangle["y"] + rectangle["height"], tile_y + height)
                result[
                    source_y0 - rectangle["y"] : source_y1 - rectangle["y"],
                    source_x0 - rectangle["x"] : source_x1 - rectangle["x"],
                ] = tile[
                    source_y0 - tile_y : source_y1 - tile_y,
                    source_x0 - tile_x : source_x1 - tile_x,
                ]
        if result.shape != (
            rectangle["height"],
            rectangle["width"],
            3,
        ):
            raise ValueError("OpenSlide returned an invalid RGB window")
        return result

    def _level_key(self, series_index: Any, resolution_level: Any) -> tuple[int, int]:
        return _level_key(self._level_mappings, series_index, resolution_level)


def _validate_source_file(
    path: Path, start: Mapping[str, Any], limits: Mapping[str, int]
) -> None:
    if not path.is_file():
        raise ValueError("source artifact path is not a regular file")
    size = path.stat().st_size
    expected_size = start["total_bytes"]
    if start["container"] == "SLIDE_FILE_SET":
        expected_size = next(
            member["size_bytes"]
            for member in start["origin"]["members"]
            if member["path"] == start["origin"]["entry_path"]
        )
    if size != expected_size:
        raise ValueError("source artifact size does not match start request")
    if size > limits["max_artifact_bytes"]:
        raise ValueError("source artifact exceeds max_artifact_bytes")


def _inspect_png(
    path: Path, start: Mapping[str, Any], limits: Mapping[str, int]
) -> tuple[tuple[dict[str, int], ...], dict[tuple[int, int], dict[str, float]]]:
    if start["origin"]["type"] != "derived-region":
        raise ValueError("PNG is supported only for a derived region")
    with path.open("rb") as source:
        header = source.read(_PNG_IHDR_SIZE)
    width, height = _validate_png_header(header, limits)
    return (
        (
            {
                "series_index": 0,
                "resolution_level": 0,
                "width": width,
                "height": height,
            },
        ),
        {(0, 0): _identity_transform()},
    )


def _validate_required_levels(
    start: Mapping[str, Any], levels: tuple[dict[str, int], ...]
) -> None:
    # Reuse the publication owner's exact coverage rule without publishing
    # anything. The real seal request supplies the verified whole-file digest.
    build_source_artifact_manifest(
        start,
        {"bytes_digest": "0" * 64},
        levels,
    )


def _level_key(
    mappings: Mapping[tuple[int, int], Any],
    series_index: Any,
    resolution_level: Any,
) -> tuple[int, int]:
    if isinstance(series_index, bool) or not isinstance(series_index, int):
        raise ValueError("series_index must be an integer")
    if isinstance(resolution_level, bool) or not isinstance(resolution_level, int):
        raise ValueError("resolution_level must be an integer")
    key = (series_index, resolution_level)
    if key not in mappings:
        raise ValueError("source series/resolution level is not validated")
    return key


def _level_metadata(
    levels: tuple[dict[str, int], ...],
    mappings: Mapping[tuple[int, int], Mapping[str, float]],
    series_index: Any,
    resolution_level: Any,
) -> dict[str, Any]:
    key = _level_key(mappings, series_index, resolution_level)
    level = levels[key[1]]
    return {
        "validated_levels": [dict(value) for value in levels],
        "dimensions": {"width": level["width"], "height": level["height"]},
        "selected_level_pixels_to_level_zero": dict(mappings[key]),
    }


def _validate_png_header(data: bytes, limits: Mapping[str, int]) -> tuple[int, int]:
    if len(data) < _PNG_IHDR_SIZE or data[:8] != _PNG_SIGNATURE:
        raise ValueError("source artifact is not a PNG")
    if data[8:12] != b"\x00\x00\x00\r" or data[12:16] != b"IHDR":
        raise ValueError("PNG does not begin with a canonical IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", data[16:29])
    )
    if width == 0 or height == 0:
        raise ValueError("PNG dimensions must be positive")
    if (
        width > limits["max_level_width_px"]
        or height > limits["max_level_height_px"]
        or width * height > limits["max_complete_decode_pixels"]
    ):
        raise ValueError("PNG dimensions exceed complete-decode limits")
    if (bit_depth, color_type, compression, filtering, interlace) != (8, 2, 0, 0, 0):
        raise ValueError("PNG must be noninterlaced 8-bit truecolor RGB")
    return width, height


def _validate_tiff_level(
    level: Any,
    limits: Mapping[str, int],
    allowed_compressions: set[str],
    required_tile: tuple[int, int] | None,
) -> tuple[int, int]:
    if level.axes != "YXS" or len(level.shape) != 3 or level.shape[2] != 3:
        raise ValueError("TIFF level must contain one interleaved RGB plane")
    if len(level.pages) != 1:
        raise ValueError("TIFF level must contain one z=0, t=0 page")
    height, width, _ = level.shape
    if (
        width <= 0
        or height <= 0
        or width > limits["max_level_width_px"]
        or height > limits["max_level_height_px"]
    ):
        raise ValueError("TIFF level dimensions exceed source-artifact limits")

    page = level.pages[0]
    if (
        page.dtype != np.dtype(np.uint8)
        or page.samplesperpixel != 3
        or getattr(page.photometric, "name", None) != "RGB"
        or getattr(page.planarconfig, "name", None) != "CONTIG"
    ):
        raise ValueError("TIFF level must be contiguous uint8 RGB")
    if not page.is_tiled:
        raise ValueError("TIFF source levels must be tiled")
    orientation = page.tags.get("Orientation")
    if orientation is not None and int(orientation.value) != 1:
        raise ValueError("TIFF source levels must use top-left orientation")
    if page.tilewidth * page.tilelength > limits["max_complete_decode_pixels"]:
        raise ValueError("TIFF tile dimensions exceed single-decode limits")
    compression = getattr(page.compression, "name", None)
    if compression not in allowed_compressions:
        raise ValueError(f"unsupported TIFF compression: {compression}")
    if (
        required_tile is not None
        and (
            page.tilewidth,
            page.tilelength,
        )
        != required_tile
    ):
        raise ValueError("derived TIFF must use 512-by-512 tiles")
    return width, height


def _qupath_level_downsamples(
    dimensions: tuple[tuple[int, int], ...],
    format_name: str = "",
) -> tuple[float, ...]:
    """QuPath 0.6/0.7 level normalization, including its CellSens exception."""

    if not dimensions:
        raise ValueError("OpenSlide source has no pyramid levels")
    full_width, full_height = dimensions[0]
    result = []
    last = -1.0
    for index, (width, height) in enumerate(dimensions):
        if width <= 0 or height <= 0 or width > full_width or height > full_height:
            raise ValueError("source has invalid pyramid dimensions")
        downsample_x = full_width / width
        downsample_y = full_height / height
        vsi_mismatch = format_name == "CellSens VSI" and abs(
            downsample_x - downsample_y
        ) > 0.01 * max(abs(downsample_x), abs(downsample_y))
        average = (downsample_x + downsample_y) / 2.0
        closest_power = 2.0 ** _java_round(math.log10(average) / math.log10(2.0))
        if (
            abs(full_height / closest_power - height) < 2
            and abs(full_width / closest_power - width) < 2
        ):
            downsample = closest_power
        else:
            rounded_x = _java_round(downsample_x)
            rounded_y = _java_round(downsample_y)
            if abs(full_width / rounded_x - width) <= 1:
                downsample_x = float(rounded_x)
            if abs(full_height / rounded_y - height) <= 1:
                downsample_y = float(rounded_y)
            if downsample_x == downsample_y:
                downsample = downsample_x
            elif downsample_x == closest_power or downsample_y == closest_power:
                downsample = closest_power
            else:
                downsample = (downsample_x + downsample_y) / 2.0
        if vsi_mismatch:
            downsample = 2.0**index
        if downsample <= last:
            break
        result.append(float(downsample))
        last = downsample
    if not result or result[0] != 1.0:
        raise ValueError("OpenSlide level zero does not map to itself")
    return tuple(result)


def _qupath_bounds(
    properties: Mapping[str, str], full_dimensions: tuple[int, int]
) -> tuple[int, int, int, int, bool]:
    full_width, full_height = full_dimensions
    if not all(key in properties for key in _OPENSLIDE_BOUNDS_KEYS):
        return 0, 0, full_width, full_height, False
    try:
        x = int(properties[openslide.PROPERTY_NAME_BOUNDS_X])
        y = int(properties[openslide.PROPERTY_NAME_BOUNDS_Y])
        width = int(properties[openslide.PROPERTY_NAME_BOUNDS_WIDTH])
        height = int(properties[openslide.PROPERTY_NAME_BOUNDS_HEIGHT])
    except (TypeError, ValueError):
        return 0, 0, full_width, full_height, False
    cropped = width != full_width and height != full_height
    if not cropped:
        return 0, 0, full_width, full_height, False
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > full_width
        or y + height > full_height
    ):
        raise ValueError("OpenSlide bounds are outside level zero")
    return x, y, width, height, True


def _openslide_background(properties: Mapping[str, str]) -> tuple[int, int, int]:
    value = properties.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR)
    if value is None:
        return (0, 0, 0)
    try:
        color = int(value.removeprefix("#"), 16)
        if not 0 <= color <= 0xFFFFFF:
            raise ValueError("invalid RGB color")
        return ((color >> 16) & 255, (color >> 8) & 255, color & 255)
    except ValueError:
        # QuPath leaves its TYPE_INT_RGB image black if Color.decode fails.
        return (0, 0, 0)


def _read_premultiplied_openslide_rgb(
    slide: openslide.OpenSlide,
    location: tuple[int, int],
    resolution_level: int,
    width: int,
    height: int,
    background: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Read the premultiplied C buffer painted over QuPath's captured background.

    The public Pillow adapter unpremultiplies OpenSlide's ARGB buffer. That
    conversion loses one-bit edge values and cannot reproduce QuPath's
    ``TYPE_INT_ARGB_PRE`` -> black ``TYPE_INT_RGB`` composition. The pinned
    openslide-python raw function is the same official C read with its error
    check intact; the fixture locks this narrow adapter seam.
    """

    buffer = (ctypes.c_uint32 * (width * height))()
    try:
        openslide_lowlevel._read_region(  # noqa: SLF001 - pinned exact C-buffer seam
            slide._osr,  # noqa: SLF001 - handle owned by the pinned wrapper
            buffer,
            location[0],
            location[1],
            resolution_level,
            width,
            height,
        )
    except (openslide.OpenSlideError, OSError) as error:
        raise ValueError("OpenSlide failed to read a source tile") from error
    argb = np.ctypeslib.as_array(buffer).reshape(height, width)
    rgb = np.stack(
        (
            (argb >> 16) & 0xFF,
            (argb >> 8) & 0xFF,
            argb & 0xFF,
        ),
        axis=2,
    )
    if any(background):
        inverse_alpha = 255 - (argb >> 24)
        rgb += (
            inverse_alpha[..., None] * np.asarray(background, dtype=np.uint32) + 127
        ) // 255
    return rgb.astype(np.uint8)


def _identity_transform() -> dict[str, float]:
    return {
        "scale_x": 1.0,
        "scale_y": 1.0,
        "translate_x": 0.0,
        "translate_y": 0.0,
    }


def _java_round(value: float) -> int:
    return math.floor(value + 0.5)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
