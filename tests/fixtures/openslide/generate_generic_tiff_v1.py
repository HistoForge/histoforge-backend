"""Generate the frozen OpenSlide generic-TIFF parity source.

Run only with the versions pinned by ``environment.yml``. The checked-in output
is the fixture; this script records its deterministic construction and refuses
to publish bytes that differ from the reviewed identity.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import tifffile

EXPECTED_SHA256 = "3801fbdb8e5526cf56f77b0a9db433f088804cf0428d3d33e49aff12881c0852"


def generate(path: Path) -> None:
    rng = np.random.default_rng(20_260_818)
    level_zero = rng.integers(0, 256, size=(385, 513, 3), dtype=np.uint8)
    levels = (level_zero, level_zero[::2, ::2], level_zero[::4, ::4])

    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, bigtiff=True) as output:
        for index, level in enumerate(levels):
            output.write(
                level,
                photometric="rgb",
                planarconfig="contig",
                tile=(128, 128),
                compression="jpeg",
                compressionargs={"level": 90},
                subfiletype=0 if index == 0 else 1,
                metadata=None,
                resolution=(40_000 / (2**index), 40_000 / (2**index)),
                resolutionunit="CENTIMETER",
            )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "fixture bytes differ from the reviewed identity; "
            "use the pinned environment"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
