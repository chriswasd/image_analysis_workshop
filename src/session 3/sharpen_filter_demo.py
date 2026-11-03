"""Sharpen an actin image with a 3x3 convolution kernel.

Run it from the top folder of the workshop repository with:

    uv run "src/session 3/sharpen_filter_demo.py"

The kernel gives the centre pixel a large positive weight and its 8 neighbours
a weight of -1:

    -1 -1 -1
    -1 12 -1
    -1 -1 -1

Where the image is flat, the neighbours cancel most of the centre and the pixel
stays about the same.  Where the brightness changes (an edge or a thin
filament), the difference is amplified, so edges look crisper.

The weights add up to 12 - 8 = 4, so we divide the kernel by 4.  Otherwise the
whole image would become 4x brighter.  (ImageJ's Process > Filters > Sharpen
uses this same kernel and the same normalisation.)

Sharpening can push pixels below 0 or above the largest value the image type
can hold, so those values are clipped before saving.

The sharpened image is saved next to this script as
    Actin_cropped_sharpened.tif
with the same pixel size as the input image.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = Path(__file__).resolve().parent

KERNEL = np.array(
    [
        [-1, -1, -1],
        [-1, 12, -1],
        [-1, -1, -1],
    ],
    dtype=float,
)
KERNEL = KERNEL / KERNEL.sum()  # weights now add up to 1: brightness is kept

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

# Do the maths with decimals so negative values are not lost along the way.
# mode="reflect" mirrors the image at its borders so edge pixels have neighbours.
sharpened = ndimage.convolve(image.astype(float), KERNEL, mode="reflect")

# Clip to the range the original image type can hold (0..65535 for uint16).
limits = np.iinfo(image.dtype)
n_clipped = np.count_nonzero((sharpened < limits.min) | (sharpened > limits.max))
sharpened = np.clip(np.round(sharpened), limits.min, limits.max).astype(image.dtype)

out_path = OUT_DIR / f"{INPUT.stem}_sharpened.tif"
tifffile.imwrite(
    out_path,
    sharpened,
    imagej=True,
    resolution=(x_resolution, y_resolution),  # keep the pixel size
    metadata={"unit": unit},
)
print(f"Input  range: {image.min()} - {image.max()}")
print(f"Output range: {sharpened.min()} - {sharpened.max()}  ({n_clipped} pixels clipped)")
print(f"Saved {out_path.name}")
