"""Smooth a noisy nucleus image by convolving it with small averaging kernels.

Run it from the top folder of the workshop repository with:

    uv run "src/session 3/mean_filter_demo.py"

Convolution replaces each pixel by a weighted sum of itself and its
neighbours.  The kernel holds the weights.  This script tries three kernels:

    ones3x3     full 3x3 square of ones (9 pixels)
                1 1 1
                1 1 1
                1 1 1
    center4_3x3 3x3 square of ones, but the central pixel weighs 4
                1 1 1
                1 4 1
                1 1 1
    ones7x7     full 7x7 square of ones (49 pixels)

Each kernel is divided by the sum of its weights, so the weights add up to 1.
That way the filtered image keeps the same overall brightness: a flat region
of value 100 stays 100.

A heavier central weight keeps more of the original pixel, so it smooths less
than the plain 3x3 mean.  The 7x7 kernel averages many more pixels, so it
removes more noise but also blurs away more fine detail.

The filtered images are saved next to this script as
    Nucleus_cropped_noise_mean_<kernel>.tif
with the same pixel size as the input image.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"
OUT_DIR = Path(__file__).resolve().parent

# The weights each neighbour is multiplied with, before normalizing.
KERNELS = {
    "ones3x3": np.ones((3, 3)),
    "center4_3x3": np.array(
        [
            [1, 1, 1],
            [1, 4, 1],
            [1, 1, 1],
        ],
        dtype=float,
    ),
    "ones7x7": np.ones((7, 7)),
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

for name, kernel in KERNELS.items():
    # Normalize: divide by the sum so the weights add up to 1.
    kernel = kernel / kernel.sum()

    # Work in decimal numbers so the weighted sums are not rounded too early.
    # mode="reflect" mirrors the image at its borders so edge pixels have neighbours.
    filtered = ndimage.convolve(image.astype(float), kernel, mode="reflect")

    # Round back to whole numbers in the same type as the input image.
    filtered = np.round(filtered).astype(image.dtype)

    out_path = OUT_DIR / f"{INPUT.stem}_mean_{name}.tif"
    tifffile.imwrite(
        out_path,
        filtered,
        imagej=True,
        resolution=(x_resolution, y_resolution),  # keep the pixel size
        metadata={"unit": unit},
    )
    print(f"Saved {out_path.name}  (kernel {kernel.shape[1]}x{kernel.shape[0]}, "
          f"center weight {kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]:.3f})")
