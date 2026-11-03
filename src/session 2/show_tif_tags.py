"""Show all the tif tags stored in a tif file, in a human readable way.

Run it from the top folder of the workshop repository with:

    uv run src/show_tif_tags.py

or look at another tif file with:

    uv run src/show_tif_tags.py path/to/another_file.tif

A tif file is a list of "pages" (one 2D image each).  Every page carries its
own set of "tags": small labelled pieces of information such as the width,
the number of bits per pixel, or the pixel size.  Each tag has a number (its
"code"), a name, and a value.  This script prints them all, page by page,
with a short explanation for the most common ones.
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Cell_cropped.tif"

# Short explanations of the tags you will meet most often.  The full list is
# in the tiff specification; tifffile knows the names of hundreds of them.
EXPLANATIONS = {
    "NewSubfileType": "what kind of page this is (0 = a normal image)",
    "ImageWidth": "number of pixels from left to right",
    "ImageLength": "number of pixels from top to bottom (the height)",
    "BitsPerSample": "bits used for each pixel value (8 -> 0-255, 16 -> 0-65535)",
    "Compression": "how the pixel data is compressed (1 = not compressed)",
    "PhotometricInterpretation": "how to turn numbers into gray or color "
                                 "(MINISBLACK: 0 is black)",
    "ImageDescription": "free text; ImageJ stores its own metadata here",
    "StripOffsets": "where in the file the pixel data starts (in bytes)",
    "SamplesPerPixel": "numbers per pixel (1 = gray, 3 = RGB)",
    "RowsPerStrip": "pixel data is stored in strips of this many rows",
    "StripByteCounts": "how many bytes each strip of pixel data takes",
    "XResolution": "pixels per unit of length, horizontally",
    "YResolution": "pixels per unit of length, vertically",
    "ResolutionUnit": "unit for X/YResolution (NONE: the unit is stored "
                      "elsewhere, ImageJ puts it in ImageDescription)",
    "PlanarConfiguration": "for color: RGBRGB... (CONTIG) or RRR..GGG..BBB.. "
                           "(SEPARATE)",
    "Software": "the program that wrote the file",
    "DateTime": "when the file was written",
    "SampleFormat": "type of pixel numbers (UINT = whole numbers >= 0, "
                    "IEEEFP = decimal numbers)",
    "IJMetadataByteCounts": "sizes of the blocks of extra ImageJ metadata",
    "IJMetadata": "extra ImageJ metadata, such as display ranges and "
                  "lookup tables (LUTs)",
}


def describe(value, indent):
    """Turn a tag value into text that is pleasant to read."""
    pad = " " * indent
    # Enumerations: tifffile turns numbers like 1 into names like MINISBLACK.
    if hasattr(value, "name") and hasattr(value, "value"):
        return f"{value.name} ({value.value})"
    # Fractions: resolutions are stored as (numerator, denominator).
    if (isinstance(value, tuple) and len(value) == 2
            and all(isinstance(v, int) for v in value) and value[1] > 1):
        return f"{value[0]}/{value[1]} = {value[0] / value[1]:.6g}"
    # Text with several lines: put each line on its own row.
    if isinstance(value, str) and "\n" in value:
        lines = value.strip().splitlines()
        return "\n" + "\n".join(pad + line for line in lines)
    # Dictionaries (ImageJ metadata): one entry per row.
    if isinstance(value, dict):
        rows = [f"{pad}{key}: {describe(item, indent + 2)}"
                for key, item in value.items()]
        return "\n" + "\n".join(rows)
    # Large arrays (such as lookup tables): just say what they are.
    if isinstance(value, np.ndarray):
        return (f"array of shape {value.shape}, {value.dtype}, "
                f"values {value.min()} to {value.max()}")
    if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
        return f"{len(value)} x " + describe(value[0], indent)
    # Long lists of numbers: show the first few only.
    if isinstance(value, (tuple, list)) and len(value) > 8:
        return f"{list(value[:8])} ... ({len(value)} values)"
    return str(value)


path = Path(sys.argv[1]) if len(sys.argv) > 1 else IMAGE

with tifffile.TiffFile(path) as tif:
    print(f"File: {path.name}  ({path.stat().st_size / 1024:.0f} kB)")
    print(f"Pages (2D images) in this file: {len(tif.pages)}")
    print(f"Written by / flavor: {', '.join(sorted(tif.flags)) or 'plain tiff'}")

    for number, page in enumerate(tif.pages):
        print()
        print("=" * 70)
        print(f"Page {number}")
        print("=" * 70)
        for tag in page.tags:
            print(f"[{tag.code:5d}] {tag.name}: {describe(tag.value, 10)}")
            # Explain each tag only the first time we see it.
            if tag.name in EXPLANATIONS:
                print(f"        -> {EXPLANATIONS.pop(tag.name)}")

    # ImageJ writes the pixel size as "pixels per unit" in XResolution and
    # the unit in ImageDescription.  Put it together for you.
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "unit")
    print()
    print(f"Pixel size: {denominator / numerator:.4f} {unit} per pixel")
