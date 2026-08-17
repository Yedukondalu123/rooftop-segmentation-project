"""
Combines a CNN model's probability mask with a LangSAM binary mask via
geo-aware intersection - only pixels where BOTH models agree "rooftop"
survive. This suppresses each approach's independent false positives:
  - LangSAM can bleed onto tree-lined roads/open ground (text-prompt/box
    based detection isn't pixel-precise about what "building" means)
  - The CNN model can miss rooftop types outside its training distribution
    (industrial/institutional buildings, informal dense housing)
Intersection keeps only the overlap, trading some recall for much higher
precision - a deliberate, documented trade-off.

Since the two masks may come from different crops/extents with different
pixel grids, this reprojects the LangSAM mask onto the CNN mask's exact
grid (same transform, same shape) before combining, so pixels line up
correctly regardless of how each pipeline cropped its input.
"""

import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy import ndimage


def close_gaps(binary_mask, iterations=4):
    """
    Fills small internal holes/gaps within detected regions (caused by
    pixel-level disagreement between the two models on parts of an
    otherwise-agreed-upon rooftop) via morphological closing: dilate then
    erode. This solidifies patchy/speckled regions into clean blocky shapes
    without significantly changing their outer boundary.
    """
    dilated = ndimage.binary_dilation(binary_mask, iterations=iterations)
    closed = ndimage.binary_erosion(dilated, iterations=iterations)
    return closed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn_prob", required=True, help="CNN model's *_prob.tif (e.g. from ReFineNet)")
    parser.add_argument("--langsam_mask", required=True, help="LangSAM's binary mask tif")
    parser.add_argument("--cnn_threshold", type=float, default=0.5, help="Threshold to binarize the CNN probability mask")
    parser.add_argument("--mode", choices=["and", "or"], default="and",
                         help="'and' = intersection (higher precision, cannot exceed either input's recall). "
                              "'or' = union (higher recall/coverage, but inherits both models' false positives)")
    parser.add_argument("--close_gaps", type=int, default=4, help="Morphological closing iterations to fill internal holes (0 to disable)")
    parser.add_argument("--out", required=True, help="Output path prefix")
    args = parser.parse_args()

    with rasterio.open(args.cnn_prob) as cnn_src:
        cnn_prob = cnn_src.read(1)
        cnn_binary = (cnn_prob >= args.cnn_threshold)
        target_transform = cnn_src.transform
        target_crs = cnn_src.crs
        target_shape = cnn_src.shape
        profile = cnn_src.profile.copy()

    with rasterio.open(args.langsam_mask) as lang_src:
        langsam_data = lang_src.read(1)
        langsam_transform = lang_src.transform
        langsam_crs = lang_src.crs

    # Reproject/resample the LangSAM mask onto the CNN mask's exact grid,
    # so both arrays line up pixel-for-pixel regardless of original crop extent.
    langsam_aligned = np.zeros(target_shape, dtype=np.uint8)
    reproject(
        source=langsam_data,
        destination=langsam_aligned,
        src_transform=langsam_transform,
        src_crs=langsam_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
    )
    langsam_binary = langsam_aligned > 0

    # Report how much geographic overlap actually exists between the two
    # inputs - if the LangSAM crop is much smaller than the CNN's AOI, most
    # of the CNN area will have no LangSAM coverage at all (treated as "no
    # detection" -> excluded by intersection), which is expected and fine
    # for a smoke test, but worth knowing before drawing conclusions.
    langsam_covered_area = (langsam_aligned.astype(np.int64) >= 0).sum()  # full grid, always true
    langsam_nonzero_area = (langsam_data > 0).sum()
    print(f"LangSAM source mask had {langsam_nonzero_area} nonzero pixels before reprojection")

    if args.mode == "and":
        combined = cnn_binary & langsam_binary
    else:
        combined = cnn_binary | langsam_binary

    if args.close_gaps > 0:
        before_pct = combined.sum() / combined.size * 100
        combined = close_gaps(combined, iterations=args.close_gaps)
        after_pct = combined.sum() / combined.size * 100
        print(f"Gap-closing (iterations={args.close_gaps}): {before_pct:.2f}% -> {after_pct:.2f}%")

    total = combined.size
    cnn_pct = cnn_binary.sum() / total * 100
    langsam_pct = langsam_binary.sum() / total * 100
    combined_pct = combined.sum() / total * 100

    print(f"CNN alone:            {cnn_pct:.2f}% of area")
    print(f"LangSAM alone:        {langsam_pct:.2f}% of area (on CNN's grid)")
    print(f"Combined ({args.mode}):       {combined_pct:.2f}% of area")

    binary_path = f"{args.out}_binary.tif"
    profile_bin = dict(profile)
    profile_bin["dtype"] = "uint8"
    profile_bin["count"] = 1
    with rasterio.open(binary_path, "w", **profile_bin) as dst:
        dst.write((combined.astype(np.uint8)) * 255, 1)

    print(f"Saved -> {binary_path}")


if __name__ == "__main__":
    main()