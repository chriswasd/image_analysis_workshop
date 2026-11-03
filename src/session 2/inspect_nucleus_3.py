"""Look at the numbers behind an image, then look at the image itself.

Run it from the top folder of the workshop repository with:

    uv run src/inspect_nucleus_3.py

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
IMAGE = REPO / "testImages" / "nucleus_cropped_3.tif"

# tifffile reads the pixels into a NumPy array: a grid of numbers with one
# number per pixel.  We also keep the file open long enough to read the pixel
# size, which ImageJ stored as "pixels per micron" in the XResolution tag.
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator  # micrometers per pixel

# The first 10 rows and the first 10 columns.  Python counts from 0, and
# "0:10" means "start at 0, stop before 10", so this is rows 0-9, columns 0-9.
corner = pixels[0:10, 0:10]

# Print every number without NumPy shortening the output with "...".
np.set_printoptions(linewidth=200)

print(f"File: {IMAGE.name}")
print()
print("First 10 rows and 10 columns (top-left corner of the image):")
print(corner)
print()
rows, columns = pixels.shape
print(f"Number of rows    : {rows}")
print(f"Number of columns : {columns}")
print(f"Number type       : {pixels.dtype}")
print(f"Smallest value    : {pixels.min()}")
print(f"Largest value     : {pixels.max()}")
print(f"Pixel size        : {pixel_size:.3f} micrometer")

# Show the same numbers as an image.  Each number becomes a shade of grey:
# low numbers dark, high numbers bright.  "scale" tells Napari how big a pixel
# is, so the scale bar is in micrometers.
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
