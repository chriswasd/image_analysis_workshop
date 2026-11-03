"""Apply the Laplacian kernel to an actin image and compare it with unsharp masks.

Run gaussian_filter_demo.py and unsharp_mask_demo.py first, then run this
script from the top folder of the workshop repository with:

    uv run "src/session 3/laplacian_filter_demo.py"

The Laplacian kernel:

    -1 -1 -1
    -1  8 -1
    -1 -1 -1

It takes 8 x the centre pixel minus its 8 neighbours.  That is the same as
9 x (pixel minus the average of the 3x3 neighbourhood), so the Laplacian is
"input minus a small blur", just like an unsharp mask.  The weights add up to
0: flat areas become 0, bright thin structures become positive, and the dark
rims right next to them become negative.

The script compares the Laplacian with the 5x5 and 21x21 unsharp masks in two
ways:

1.  The correlation between the Laplacian and each mask (1 = the same pattern,
    0 = unrelated).
2.  The sharpened images: input + amount x Laplacian, next to the unsharp
    masked images input + 100% x mask.  The amount is chosen so that the
    Laplacian adds as much detail (same standard deviation) as the 5x5 mask.

Saved, with the same pixel size as the input image:
    src/session 3/Actin_cropped_laplacian.tif             (32-bit float)
    src/session 3/Actin_cropped_laplacian_sharpened.tif   (same type as input)
    Pictures/laplacian/laplacian_vs_unsharp.png           side-by-side figure
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"
OUT_DIR = Path(__file__).resolve().parent
PICTURE_DIR = REPO / "Pictures" / "laplacian"

LAPLACIAN = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ],
    dtype=float,
)
MASK_SIZES = [5, 21]

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


def save(path, data):
    """Save an image with the pixel size of the input image."""
    tifffile.imwrite(path, data, imagej=True,
                     resolution=(x_resolution, y_resolution),
                     metadata={"unit": unit})
    print(f"Saved {path.relative_to(REPO)}")


# Do the maths with decimals so negative values are not lost along the way.
# mode="reflect" mirrors the image at its borders so edge pixels have neighbours.
pixels = image.astype(float)
laplacian = ndimage.convolve(pixels, LAPLACIAN, mode="reflect")
save(OUT_DIR / f"{INPUT.stem}_laplacian.tif", laplacian.astype(np.float32))
print(f"Laplacian range: {laplacian.min():.0f} to {laplacian.max():.0f}")

# 1. How similar is the Laplacian to each unsharp mask?
masks = {}
print("\nCorrelation of the Laplacian with each unsharp mask:")
for size in MASK_SIZES:
    masks[size] = tifffile.imread(OUT_DIR / f"{INPUT.stem}_unsharp{size}x{size}.tif")
    r = np.corrcoef(laplacian.ravel(), masks[size].ravel())[0, 1]
    print(f"  {size:2d}x{size:<2d} mask: r = {r:.2f}")

# 2. Sharpen with the Laplacian, adding as much detail as the 5x5 mask does.
amount = masks[5].std() / laplacian.std()
limits = np.iinfo(image.dtype)
sharpened = np.clip(np.round(pixels + amount * laplacian), limits.min, limits.max)
sharpened = sharpened.astype(image.dtype)
print(f"\nLaplacian amount that matches the 5x5 mask: {amount:.3f}")
save(OUT_DIR / f"{INPUT.stem}_laplacian_sharpened.tif", sharpened)

# Side-by-side figure.  Top row: the detail images (gray = 0, white = positive,
# black = negative).  Bottom row: the sharpened images, all with the same
# display range as the input.
unsharp = {size: tifffile.imread(OUT_DIR / f"{INPUT.stem}_unsharp{size}x{size}_100pct.tif")
           for size in MASK_SIZES}
top = [("Laplacian", laplacian)] + [(f"Unsharp mask {s}x{s}", masks[s]) for s in MASK_SIZES]
bottom = ([(f"Input + {amount:.2f} x Laplacian", sharpened)]
          + [(f"Unsharp {s}x{s}, 100%", unsharp[s]) for s in MASK_SIZES])
low, high = np.percentile(image, [0.5, 99.5])

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
for ax, (title, data) in zip(axes[0], top):
    limit = np.percentile(np.abs(data), 99.5)
    ax.imshow(data, cmap="gray", vmin=-limit, vmax=limit)
    ax.set_title(title)
for ax, (title, data) in zip(axes[1], bottom):
    ax.imshow(data, cmap="gray", vmin=low, vmax=high)
    ax.set_title(title)
for ax in axes.ravel():
    ax.axis("off")
fig.tight_layout()
PICTURE_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PICTURE_DIR / "laplacian_vs_unsharp.png", dpi=100)
print(f"Saved {(PICTURE_DIR / 'laplacian_vs_unsharp.png').relative_to(REPO)}")
