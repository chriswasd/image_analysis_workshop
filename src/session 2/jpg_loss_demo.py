"""See what jpg compression does to an image, and what saving it again and
again does.

Run it from the top folder of the workshop repository with:

    uv run src/jpg_loss_demo.py

The script:
  1. reads testImages/Cell_cropped_2.tif (the original, lossless),
  2. saves it as a jpg and reads that jpg back in,
  3. calculates the difference between the jpg and the original,
  4. re-saves the jpg 100 times (read, save, read, save, ...) and calculates
     the difference with the original again.

The jpg files are written to testImages/jpg_demo/.  Some numbers are printed
in the terminal, and Napari opens in its own window.  Close that window to end
the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
import imageio.v3 as iio

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Cell_cropped_2.tif"
OUT_DIR = REPO / "testImages" / "jpg_demo"

# jpg quality goes from 1 (tiny file, ugly) to 100 (big file, best looking).
# 75 is a common default.  Try other values!
QUALITY = 75

# How many times to re-save the jpg.
N_RESAVES = 100


def save_and_reload_jpg(pixels, path):
    """Save pixels as a jpg file, then read the jpg back in."""
    iio.imwrite(path, pixels, quality=QUALITY)
    return iio.imread(path)


def difference(image, original):
    """Pixel-by-pixel difference: image minus original.

    The pixels are 8-bit numbers (0-255) that cannot be negative, so we first
    turn them into "int16" numbers that can be.
    """
    return image.astype(np.int16) - original.astype(np.int16)


def report(name, diff):
    """Print how big the differences are."""
    changed = np.count_nonzero(diff) / diff.size
    print(f"{name}:")
    print(f"  average size of the difference: {np.abs(diff).mean():.2f}")
    print(f"  largest difference:             {np.abs(diff).max()}")
    print(f"  numbers that changed:           {100 * changed:.1f}%")


# Read the original.  We also read the pixel size, which ImageJ stored as
# "pixels per micron" in the XResolution tag.  jpg files cannot store this, so
# we will hand it to Napari ourselves.
with tifffile.TiffFile(IMAGE) as tif:
    original = tif.asarray()
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator  # micrometers per pixel

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save as jpg once and read it back.
once_path = OUT_DIR / "Cell_cropped_2_once.jpg"
jpg_once = save_and_reload_jpg(original, once_path)
diff_once = difference(jpg_once, original)

# Re-save it many times.  Each round reads the previous jpg and saves it again,
# just like opening a jpg, editing nothing, and clicking "save".
many_path = OUT_DIR / f"Cell_cropped_2_{N_RESAVES}x.jpg"
jpg_many = jpg_once
for _ in range(N_RESAVES):
    jpg_many = save_and_reload_jpg(jpg_many, many_path)
diff_many = difference(jpg_many, original)

print(f"Original: {IMAGE.name}, {IMAGE.stat().st_size / 1024:.0f} kB")
print(f"jpg (quality {QUALITY}): {once_path.stat().st_size / 1024:.0f} kB")
print()
report("jpg saved once vs. original", diff_once)
report(f"jpg saved {N_RESAVES + 1} times vs. original", diff_many)
report(f"jpg saved {N_RESAVES + 1} times vs. jpg saved once",
       difference(jpg_many, jpg_once))

# Show everything.  "rgb=True" tells Napari that the last 3 numbers of each
# pixel are Red, Green and Blue.  "scale" tells Napari how big a pixel is, so
# the scale bar is in micrometers.
scale = (pixel_size, pixel_size)
units = ("um", "um")
viewer = napari.Viewer()
viewer.add_image(original, name="original", rgb=True, scale=scale, units=units)
viewer.add_image(jpg_once, name="jpg once", rgb=True, scale=scale, units=units)
viewer.add_image(jpg_many, name=f"jpg {N_RESAVES + 1}x", rgb=True,
                 scale=scale, units=units)

# For the difference images we keep, for every pixel, the biggest change in any
# of its 3 colors.  Black = unchanged, bright = changed a lot.  Both difference
# images get the same contrast so you can compare them fairly.
change_once = np.abs(diff_once).max(axis=-1)
change_many = np.abs(diff_many).max(axis=-1)
limits = (0, max(change_once.max(), change_many.max()))
viewer.add_image(change_once, name="difference once", colormap="magma",
                 contrast_limits=limits, scale=scale, units=units)
viewer.add_image(change_many, name=f"difference {N_RESAVES + 1}x",
                 colormap="magma", contrast_limits=limits, scale=scale,
                 units=units)

# Show the layers side by side instead of on top of each other.
viewer.grid.enabled = True
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
