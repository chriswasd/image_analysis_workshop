"""Find edges in an actin image with the Sobel x and Sobel y kernels.

Run it from the top folder of the workshop repository with:

    uv run "src/session 3/sobel_filter_demo.py"

or use another image with:

    uv run "src/session 3/sobel_filter_demo.py" path/to/another_file.tif

Sobel x looks at how the brightness changes from left to right:

    -1  0  1
    -2  0  2
    -1  0  1

Sobel y looks at how the brightness changes from top to bottom:

    -1 -2 -1
     0  0  0
     1  2  1

Each kernel subtracts the pixels on one side from the pixels on the other side.
In a flat area the two sides cancel and the result is 0.  At an edge the result
is a large positive number (brighter on the right / bottom) or a large negative
number (brighter on the left / top).  The weights add up to 0, so there is
nothing to normalise.

The two results are combined into the gradient magnitude, the "strength" of the
edge no matter which way it points:

    magnitude = square root of (Sobel x squared + Sobel y squared)

Because Sobel x and Sobel y can be negative, all three images are saved as
32-bit decimal (float) images, so no values are lost.  They are saved next to
this script, named after the input image (for Actin_cropped.tif):
    Actin_cropped_sobel_x.tif
    Actin_cropped_sobel_y.tif
    Actin_cropped_sobel_magnitude.tif
with the same pixel size as the input image.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = Path(__file__).resolve().parent

SOBEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=float,
)

SOBEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=float,
)

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

# ndimage.correlate lays the kernel on the image exactly as written above
# (ndimage.convolve would first flip it, which swaps the sign of the result).
# mode="reflect" mirrors the image at its borders so edge pixels have neighbours.
pixels = image.astype(float)
sobel_x = ndimage.correlate(pixels, SOBEL_X, mode="reflect")
sobel_y = ndimage.correlate(pixels, SOBEL_Y, mode="reflect")
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

print(f"Input     range: {image.min()} - {image.max()}")
for name, result in [("sobel_x", sobel_x), ("sobel_y", sobel_y),
                     ("sobel_magnitude", magnitude)]:
    out_path = OUT_DIR / f"{INPUT.stem}_{name}.tif"
    tifffile.imwrite(
        out_path,
        result.astype(np.float32),
        imagej=True,
        resolution=(x_resolution, y_resolution),  # keep the pixel size
        metadata={"unit": unit},
    )
    print(f"Saved {out_path.name}  (range {result.min():.0f} - {result.max():.0f})")
