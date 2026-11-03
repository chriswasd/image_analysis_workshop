"""Look at the Red, Green and Blue numbers behind a color (RGB) image.

Run it from the top folder of the workshop repository with:

    uv run src/inspect_rgb_cell.py

A table of R, G and B values for about 3% of the pixels is printed in the
terminal, and Napari opens in its own window.  Close that window to end the
script.
"""

import math
from pathlib import Path

import napari
import tifffile

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Cell_cropped_2.tif"

# What fraction of the pixels to show.
FRACTION = 0.03

# tifffile reads the pixels into a NumPy array.  For an RGB image this is a
# grid of rows x columns x 3: three numbers (R, G, B) for every pixel.  We also
# read the pixel size, which ImageJ stored as "pixels per micron" in the
# XResolution tag.
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator  # micrometers per pixel

n_rows, n_columns, n_colors = pixels.shape

# To keep 3% of the pixels on a regular grid, we keep one pixel every STEP rows
# and every STEP columns.  That keeps 1 out of STEP x STEP pixels, so STEP is
# the square root of 1 / 0.03, which rounds to 6 (1/36 = 2.8%).
STEP = round(math.sqrt(1 / FRACTION))
sample = pixels[::STEP, ::STEP]
n_shown = sample.shape[0] * sample.shape[1]

print(f"File: {IMAGE.name}")
print(f"Rows: {n_rows}, columns: {n_columns}, colors per pixel: {n_colors}")
print(f"Number type: {pixels.dtype}")
print()
print(f"Showing every {STEP}th row and every {STEP}th column:")
print(f"{n_shown} of {n_rows * n_columns} pixels "
      f"({100 * n_shown / (n_rows * n_columns):.1f}%)")
print()

# Print the table: one line per pixel, with its position and its 3 numbers.
print(f"{'Row':>5} {'Column':>7} {'R':>5} {'G':>5} {'B':>5}")
print("-" * 31)
for row in range(0, n_rows, STEP):
    for column in range(0, n_columns, STEP):
        r, g, b = pixels[row, column]
        print(f"{row:>5} {column:>7} {r:>5} {g:>5} {b:>5}")

# Show the full image.  "rgb=True" tells Napari that the last 3 numbers of
# each pixel are Red, Green and Blue, so it mixes them into one color.
# "scale" tells Napari how big a pixel is, so the scale bar is in micrometers.
viewer = napari.Viewer()
viewer.add_image(
    pixels,
    name=IMAGE.stem,
    rgb=True,
    scale=(pixel_size, pixel_size),
    units=("um", "um"),
)
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
