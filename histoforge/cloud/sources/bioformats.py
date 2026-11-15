"""Scoped native-RGB Bio-Formats reads matching QuPath's public pixel path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from histoforge.cloud.sources.java_runtime import ensure_bioformats_jvm

# Limits Java byte[], decoded image, RGB image and transfer-buffer coexistence.
# This does not claim to bound a format decoder's internal metadata/tile memory.
_JAVA_READ_EDGE = 512


class BioFormatsReader:
    """Own one Java reader; return only detached NumPy arrays and Python metadata.

    The caller holds the existing storage gate until close. There is no pool,
    memoizer, implicit scene choice, reader substitution or JVM shutdown here.
    """

    def __init__(
        self, path: Path, origin: Mapping[str, Any], limits: Mapping[str, int]
    ):
        self._jpype = ensure_bioformats_jvm()
        self._reader = None
        self._origin = origin
        self.levels: list[dict[str, int]] = []
        self.dimensions: list[tuple[int, int]] = []
        self.format = ""
        try:
            reader = self._jpype.JClass("loci.formats.ImageReader")()
            self._reader = reader
            reader.setFlattenedResolutions(False)
            reader.setMetadataStore(
                self._jpype.JClass("loci.formats.MetadataTools").createOMEXMLMetadata()
            )
            reader.setId(str(path))
            series = origin["pyramid_series_index"]
            if not 0 <= series < reader.getSeriesCount() <= limits["max_image_series"]:
                raise ValueError(
                    "Bio-Formats series exceeds the captured source/limits"
                )
            reader.setSeries(series)
            count = reader.getResolutionCount()
            if not 1 <= count <= limits["max_pyramid_levels_per_series"]:
                raise ValueError("Bio-Formats resolution count exceeds source limits")
            if count != len(origin["reader_settings"]["levels"]):
                raise ValueError(
                    "Bio-Formats native resolutions differ from the captured view"
                )
            self.format = str(reader.getFormat())
            for level in range(count):
                reader.setResolution(level)
                if (
                    not reader.isRGB()
                    or reader.getPixelType()
                    != self._jpype.JClass("loci.formats.FormatTools").UINT8
                    or reader.getRGBChannelCount() not in (3, 4)
                    or reader.getEffectiveSizeC() != 1
                    or reader.getSizeZ() < 1
                    or reader.getSizeT() < 1
                ):
                    raise ValueError(
                        "Bio-Formats original requires native RGB8 and captured Z/T"
                    )
                if level == 0 and (
                    reader.getSizeZ() != origin["source"]["z_slices"]
                    or reader.getSizeT() != origin["source"]["timepoints"]
                ):
                    raise ValueError(
                        "Bio-Formats source Z/T differs from captured identity"
                    )
                width, height = int(reader.getSizeX()), int(reader.getSizeY())
                if (
                    not 0 < width <= limits["max_level_width_px"]
                    or not 0 < height <= limits["max_level_height_px"]
                ):
                    raise ValueError("Bio-Formats dimensions exceed source limits")
                self.dimensions.append((width, height))
                self.levels.append(
                    {
                        "series_index": series,
                        "resolution_level": level,
                        "width": width,
                        "height": height,
                    }
                )
            source = origin["source"]
            if self.dimensions[0] != (source["width"], source["height"]):
                raise ValueError(
                    "Bio-Formats level zero differs from the captured source"
                )
            self._require_declared_dependencies(path)
        except BaseException as error:
            try:
                self.close()
            finally:
                # Keep the read failure primary if closing also fails. Python retains
                # the close failure in the exception chain; do not retry the reader.
                self._raise_decoder_error(error)

    def _require_declared_dependencies(self, entry: Path) -> None:
        root = entry
        for _ in self._origin["entry_path"].split("/"):
            root = root.parent
        declared = {
            (root / member["path"]).resolve(strict=True)
            for member in self._origin.get("members", [])
        } or {entry.resolve(strict=True)}
        used = {
            Path(str(name)).resolve(strict=True)
            for name in (self._reader.getUsedFiles(False) or [])
        }
        used.add(entry.resolve(strict=True))
        if used != declared:
            raise ValueError(
                "Bio-Formats dependencies differ from the sealed slide members"
            )

    def read_rgb(self, resolution: int, rectangle: Mapping[str, int]) -> np.ndarray:
        """Fill one bounded Python result using smaller temporary Java reads."""

        if self._reader is None:
            raise RuntimeError("Bio-Formats reader is closed")
        result = np.empty((rectangle["height"], rectangle["width"], 3), dtype=np.uint8)
        try:
            self._reader.setResolution(resolution)
            for y in range(0, rectangle["height"], _JAVA_READ_EDGE):
                height = min(_JAVA_READ_EDGE, rectangle["height"] - y)
                for x in range(0, rectangle["width"], _JAVA_READ_EDGE):
                    width = min(_JAVA_READ_EDGE, rectangle["width"] - x)
                    result[y : y + height, x : x + width] = self._read_rgb_block(
                        rectangle["x"] + x, rectangle["y"] + y, width, height
                    )
        except BaseException as error:
            self._raise_decoder_error(error)
        return result

    def _read_rgb_block(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        reader = self._reader
        encoded = reader.openBytes(reader.getIndex(0, 0, 0), x, y, width, height)
        image = self._jpype.JClass("loci.formats.gui.AWTImageTools").openImage(
            encoded, reader, width, height
        )
        buffered_image = self._jpype.JClass("java.awt.image.BufferedImage")
        rgb = buffered_image(width, height, buffered_image.TYPE_INT_RGB)
        graphics = rgb.createGraphics()
        try:
            graphics.setComposite(self._jpype.JClass("java.awt.AlphaComposite").Src)
            graphics.drawImage(image, 0, 0, None)
        finally:
            graphics.dispose()
        packed = np.asarray(
            rgb.getRaster().getDataBuffer().getData(), dtype=np.int32
        ).reshape(height, width)
        # stack+astype detach from every Java image and primitive array.
        return np.stack(
            ((packed >> 16) & 255, (packed >> 8) & 255, packed & 255), axis=2
        ).astype(np.uint8)

    def close(self) -> None:
        reader, self._reader = self._reader, None
        if reader is not None:
            try:
                reader.close()
            except self._jpype.JException as error:
                self._raise_decoder_error(error)

    def _raise_decoder_error(self, error: BaseException) -> None:
        if isinstance(error, self._jpype.JException):
            if isinstance(error, self._jpype.JClass("java.lang.VirtualMachineError")):
                raise RuntimeError(
                    "Bio-Formats exceeded or lost its Java runtime resources"
                ) from error
            raise ValueError(
                "Bio-Formats could not read the declared original slide"
            ) from error
        raise error
