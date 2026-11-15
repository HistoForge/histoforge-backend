"""Real Java adapter tests; synthetic OME data does not qualify scanner formats."""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from histoforge.cloud.sources.bioformats import BioFormatsReader
from histoforge.cloud.sources.java_runtime import ensure_bioformats_jvm
from histoforge.cloud.sources.reader import SourceReader

from test_source_reader import _limits, _openslide_start, _pixels


@pytest.mark.skipif(
    not os.environ.get("HISTOFORGE_BIOFORMATS_JAR"),
    reason="requires the locked Bio-Formats JAR and Java 21",
)
def test_real_bioformats_reads_detached_rgb_and_closes(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-B", __file__, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["heap_max_bytes"] <= 256 * 1024**2
    assert result["processors"] == 1
    assert result["exact_reads"] == 4
    assert result["version"] == "8.5.0"


def _exercise(root):
    root = Path(root)
    path = root / "two-scenes.ome.tif"
    pixels = _pixels(530, 520)
    second = 255 - pixels
    with tifffile.TiffWriter(path, ome=True) as writer:
        for image in (pixels, second):
            writer.write(image, photometric="rgb", metadata={"axes": "YXS"})
    start = _openslide_start(path)
    origin = start["origin"]
    source = origin["source"]
    source.update(source_adapter_id="qupath.bioformats.local", width=520, height=530)
    origin["pyramid_series_index"] = 1
    origin["reader_settings"] = {
        "levels": [
            {
                "resolution_level": 0,
                "native_resolution_index": 0,
                "pixels_to_level_zero": {
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "translate_x": 0.0,
                    "translate_y": 0.0,
                },
            }
        ]
    }
    start["required_levels"] = [
        {"series_index": 1, "resolution_level": 0, "width": 520, "height": 530}
    ]
    rectangle = {"x": 0, "y": 0, "width": 520, "height": 530}
    for _ in range(3):
        with SourceReader(path, start, _limits()) as reader:
            actual = reader.read_rgb_view(1, 0, rectangle)
            assert np.array_equal(actual, second)
            handle = reader._bioformats
        assert handle._reader is None
        assert actual.flags.owndata
        # A detached result remains valid after reader closure and Java collection.
        ensure_bioformats_jvm().JClass("java.lang.System").gc()
        assert np.array_equal(actual, second)
    with SourceReader(path, start, _limits()) as reader:
        assert np.array_equal(
            reader.read_rgb_view(1, 0, {"x": 500, "y": 510, "width": 20, "height": 20}),
            second[510:530, 500:520],
        )
        jpype = ensure_bioformats_jvm()
        with pytest.raises(RuntimeError, match="Java runtime resources"):
            reader._bioformats._raise_decoder_error(
                jpype.JClass("java.lang.OutOfMemoryError")("diagnostic only")
            )
        with pytest.raises(ValueError, match="could not read"):
            reader._bioformats._raise_decoder_error(
                jpype.JClass("loci.formats.FormatException")("diagnostic only")
            )
    wrong = copy.deepcopy(start)
    wrong["origin"]["pyramid_series_index"] = 2
    wrong["required_levels"][0]["series_index"] = 2
    with pytest.raises(ValueError, match="series"):
        SourceReader(path, wrong, _limits())
    close = BioFormatsReader.close

    def failing_close(reader):
        close(reader)
        raise OSError("injected close failure")

    with patch.object(BioFormatsReader, "close", failing_close):
        with pytest.raises(ValueError, match="series") as failure:
            BioFormatsReader(path, wrong["origin"], _limits())
    assert isinstance(failure.value.__context__, OSError)
    tiny_limits = {**_limits(), "max_complete_decode_pixels": 16}
    with SourceReader(path, start, tiny_limits) as reader:
        with patch.object(reader._bioformats, "read_rgb") as decode:
            with pytest.raises(ValueError, match="complete-decode"):
                reader.read_rgb_view(1, 0, rectangle)
        decode.assert_not_called()
    runtime = jpype.JClass("java.lang.Runtime").getRuntime()
    print(
        json.dumps(
            {
                "heap_max_bytes": runtime.maxMemory(),
                "processors": runtime.availableProcessors(),
                "exact_reads": 4,
                "version": str(jpype.JClass("loci.formats.FormatTools").VERSION),
            }
        )
    )


if __name__ == "__main__":
    _exercise(sys.argv[1])
