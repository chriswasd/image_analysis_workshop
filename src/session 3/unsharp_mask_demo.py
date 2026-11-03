"""Sharpen an actin image with a classical unsharp mask.

Run gaussian_filter_demo.py first to make the unsharp masks, then run this
script from the top folder of the workshop repository with:

    uv run "src/session 3/unsharp_mask_demo.py"

An unsharp mask is the input image minus a blurred copy of it: it holds only
the fine detail that the blur removed.  Adding a fraction of that detail back
onto the input makes edges and thin filaments stand out more:

    sharpened = input + amount x mask

This script uses both masks made by gaussian_filter_demo.py (5x5 and 21x21
Gaussian blur) and amounts of 20%, 40%, 60%, 80% and 100%.  The small mask
crisps up the finest edges; the large mask boosts whole fibres.

Sharpening can push pixels below 0 or above the largest value the image type
can hold, so those values are clipped before saving.

The sharpened images are saved next to this script as
    Actin_cropped_unsharp<size>_<amount>pct.tif
with the same pixel size and data type as the input image.
"""

from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = Path(__file__).resolve().parent

MASK_SIZES = [5, 21]
AMOUNTS_PERCENT = [20, 40, 60, 80, 100]

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

limits = np.iinfo(image.dtype)  # smallest and largest value a uint16 can hold

for size in MASK_SIZES:
    mask_path = OUT_DIR / f"{INPUT.stem}_unsharp{size}x{size}.tif"
    mask = tifffile.imread(mask_path)
    print(f"Read {mask_path.name}")

    for percent in AMOUNTS_PERCENT:
        amount = percent / 100
        sharpened = image.astype(np.float64) + amount * mask

        # Round, and keep values inside the range the image type can hold.
        clipped = np.sum((sharpened < limits.min) | (sharpened > limits.max))
        sharpened = np.clip(np.round(sharpened), limits.min, limits.max)

        out_path = OUT_DIR / f"{INPUT.stem}_unsharp{size}x{size}_{percent:03d}pct.tif"
        tifffile.imwrite(
            out_path,
            sharpened.astype(image.dtype),
            imagej=True,
            resolution=(x_resolution, y_resolution),  # keep the pixel size
            metadata={"unit": unit},
        )
        print(f"Saved {out_path.name}  (range {sharpened.min():.0f} to "
              f"{sharpened.max():.0f}, {clipped} pixels clipped)")
