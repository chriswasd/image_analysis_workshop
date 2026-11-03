"""Remove noise from a nucleus image with median filters of different shapes.

Run it from the top folder of the workshop repository with:

    uv run "src/session 3/median_filter_demo.py"

A median filter looks at each pixel together with its neighbours, sorts all
those values, and keeps the middle one.  The "footprint" decides which
neighbours count.  This script tries three footprints:

    cross3x3   3x3 square without the 4 corners (5 pixels)
               0 1 0
               1 1 1
               0 1 0
    square3x3  full 3x3 square (9 pixels)
    square5x5  full 5x5 square (25 pixels)

Bigger footprints remove more noise but also blur away more fine detail.

The filtered images are saved next to this script as
    Nucleus_cropped_noise_median_<footprint>.tif
with the same pixel size as the input image.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"
OUT_DIR = Path(__file__).resolve().parent

# 1 = this neighbour takes part in the median, 0 = it is ignored.
FOOTPRINTS = {
    "cross3x3": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "square3x3": np.ones((3, 3), dtype=bool),
    "square5x5": np.ones((5, 5), dtype=bool),
}

# Read the image together with its pixel size information.
with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    page = tif.pages[0]
    x_resolution = page.tags["XResolution"].value  # pixels per unit, as a fraction
    y_resolution = page.tags["YResolution"].value
    imagej_metadata = tif.imagej_metadata or {}

unit = imagej_metadata.get("unit", "micron")
pixel_size = x_resolution[1] / x_resolution[0]
print(f"Read {INPUT.name}: {image.shape[1]} x {image.shape[0]} pixels, "
      f"{image.dtype}, pixel size {pixel_size:.4f} {unit}")

for name, footprint in FOOTPRINTS.items():
    # mode="reflect" mirrors the image at its borders so edge pixels have neighbours.
    filtered = ndimage.median_filter(image, footprint=footprint, mode="reflect")

    out_path = OUT_DIR / f"{INPUT.stem}_median_{name}.tif"
    tifffile.imwrite(
        out_path,
        filtered.astype(image.dtype),
        imagej=True,
        resolution=(x_resolution, y_resolution),  # keep the pixel size
        metadata={"unit": unit},
    )
    print(f"Saved {out_path.name}  (footprint uses {footprint.sum()} pixels)")
