"""Look at a sample of the numbers behind an image, spread over the whole image.

Run it from the top folder of the workshop repository with:

    uv run src/inspect_nucleus_skip.py

The numbers are printed in the terminal, and Napari opens in its own window.
Close that window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "nucleus_cropped_2.tif"

# How many rows and columns to skip between the ones we show.  Skipping 2 means
# we keep one, skip two, keep one, ... so we keep every 3rd row and column.
SKIP = 2
STEP = SKIP + 1

# How many rows and columns to show.
COUNT = 10

# tifffile reads the pixels into a NumPy array: a grid of numbers with one
# number per pixel.  We also read the pixel size, which ImageJ stored as
# "pixels per micron" in the XResolution tag.
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator  # micrometers per pixel

# "start:stop:step" picks out rows or columns.  "0:30:3" means: start at 0,
# stop before 30, jump 3 each time, so rows 0, 3, 6, ..., 27.
stop = COUNT * STEP
sample = pixels[0:stop:STEP, 0:stop:STEP]

# Which row and column numbers we picked, so you can see what was skipped.
picked = list(range(0, stop, STEP))

# Print every number without NumPy shortening the output with "...".
np.set_printoptions(linewidth=200)

print(f"File: {IMAGE.name}")
print()
print(f"{COUNT} rows and {COUNT} columns, skipping {SKIP} each time")
print(f"Row and column numbers shown: {picked}")
print(sample)
print()
print(f"Number type: {pixels.dtype}")

# Show the full image.  Each number becomes a shade of grey: low numbers dark,
# high numbers bright.  "scale" tells Napari how big a pixel is, so the scale
# bar is in micrometers.
viewer = napari.Viewer()
viewer.add_image(
    pixels,
    name=IMAGE.stem,
    scale=(pixel_size, pixel_size),
    units=("um", "um"),
)
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
