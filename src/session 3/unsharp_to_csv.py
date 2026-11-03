"""Write the pixel values of the unsharp masked actin images to a CSV file.

Run gaussian_filter_demo.py first to make the images, then run this script
from the top folder of the workshop repository with:

    uv run "src/session 3/unsharp_to_csv.py"

Only every third pixel is written, in both directions (every third column of
every third row), so the file is about 9 times smaller than the full image.

The CSV file is saved next to this script as Actin_cropped_unsharp.csv.  It
has one line per pixel with these columns:

    x, y                   pixel position (0, 0 is the top-left corner)
    x_micron, y_micron     the same position in microns
    unsharp5x5             value in Actin_cropped_unsharp5x5.tif
    unsharp21x21           value in Actin_cropped_unsharp21x21.tif

The file opens directly in Excel, Google Sheets, Prism or R.
"""

import csv
from pathlib import Path

import tifffile

HERE = Path(__file__).resolve().parent
IMAGE_NAMES = ["unsharp5x5", "unsharp21x21"]
STEP = 3  # keep every third pixel
OUTPUT = HERE / "Actin_cropped_unsharp.csv"

# Read each unsharp masked image, and the pixel size from the first one.
images = {}
for name in IMAGE_NAMES:
    path = HERE / f"Actin_cropped_{name}.tif"
    with tifffile.TiffFile(path) as tif:
        images[name] = tif.asarray()
        x_resolution = tif.pages[0].tags["XResolution"].value  # pixels per micron
        y_resolution = tif.pages[0].tags["YResolution"].value
    print(f"Read {path.name}: {images[name].shape[1]} x {images[name].shape[0]} pixels")

pixel_width = x_resolution[1] / x_resolution[0]  # microns per pixel
pixel_height = y_resolution[1] / y_resolution[0]
height, width = images[IMAGE_NAMES[0]].shape

with open(OUTPUT, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["x", "y", "x_micron", "y_micron", *IMAGE_NAMES])
    rows_written = 0
    # range(0, height, STEP) gives 0, 3, 6, 9, ... : every third row (y),
    # and in the same way every third column (x).
    for y in range(0, height, STEP):
        for x in range(0, width, STEP):
            values = [f"{images[name][y, x]:.2f}" for name in IMAGE_NAMES]
            writer.writerow([x, y, f"{x * pixel_width:.4f}", f"{y * pixel_height:.4f}", *values])
            rows_written += 1

print(f"Saved {OUTPUT.name}: {rows_written} pixels (every {STEP}rd pixel in x and y)")
