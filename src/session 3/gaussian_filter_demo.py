"""Blur an actin image with Gaussian kernels of two different sizes.

Run it from the top folder of the workshop repository with:

    uv run "src/session 3/gaussian_filter_demo.py"

A Gaussian filter replaces each pixel with a weighted average of itself and
its neighbours.  The weights follow a bell-shaped curve: the centre pixel
counts most, and pixels further away count less and less.  This script tries
two kernel sizes:

    5x5     small kernel, light blur
    21x21   large kernel, strong blur

The width of the bell curve (sigma) is chosen so the kernel reaches 3 sigma
from the centre to its edge, which holds almost all of the curve:
    sigma = (size - 1) / 6

Subtracting a blurred image from the input leaves only the fine detail that
the blur removed.  This is called an "unsharp mask".

The images are saved next to this script, with the same pixel size as the
input image, as
    Actin_cropped_gaussian<size>.tif   blurred image
    Actin_cropped_unsharp<size>.tif    input minus blurred image (32-bit float)
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = Path(__file__).resolve().parent

KERNEL_SIZES = [5, 21]


def gaussian_kernel(size):
    """Return a size x size Gaussian kernel whose weights add up to 1."""
    sigma = (size - 1) / 6
    offsets = np.arange(size) - (size - 1) / 2  # distance from the centre
    curve = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel = np.outer(curve, curve)
    return kernel / kernel.sum(), sigma


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

for size in KERNEL_SIZES:
    kernel, sigma = gaussian_kernel(size)

    # Work in floating point so the weighted averages are not rounded early.
    # mode="reflect" mirrors the image at its borders so edge pixels have neighbours.
    blurred = ndimage.convolve(image.astype(np.float64), kernel, mode="reflect")

    # Unsharp mask: what is left after removing the blur, i.e. the fine detail.
    # It is negative wherever a pixel is darker than its blurred surroundings,
    # so it is saved as 32-bit floating point rather than the input's type.
    unsharp = image.astype(np.float64) - blurred
    unsharp_path = OUT_DIR / f"{INPUT.stem}_unsharp{size}x{size}.tif"
    tifffile.imwrite(
        unsharp_path,
        unsharp.astype(np.float32),
        imagej=True,
        resolution=(x_resolution, y_resolution),  # keep the pixel size
        metadata={"unit": unit},
    )
    print(f"Saved {unsharp_path.name}  (range {unsharp.min():.1f} to {unsharp.max():.1f})")

    # Round back to the input's data type, staying inside its allowed range.
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        blurred = np.clip(np.round(blurred), limits.min, limits.max)

    out_path = OUT_DIR / f"{INPUT.stem}_gaussian{size}x{size}.tif"
    tifffile.imwrite(
        out_path,
        blurred.astype(image.dtype),
        imagej=True,
        resolution=(x_resolution, y_resolution),  # keep the pixel size
        metadata={"unit": unit},
    )
    print(f"Saved {out_path.name}  (kernel {size}x{size}, sigma {sigma:.2f} pixels)")
