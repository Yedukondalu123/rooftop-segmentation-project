"""
Draws yellow outline contours around detected rooftop regions (instead of a
solid color fill), for a cleaner "polygon boundary" look. Intended for use
with a precision-focused mask (e.g. the CNN+LangSAM intersection result)
rather than a noisy union/raw mask, since outlining noise just makes the
noise more visible, not less.
"""

import argparse
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Original input GeoTIFF")
    parser.add_argument("--aoi", default=None, help="AOI GeoJSON used during inference (optional, must match)")
    parser.add_argument("--binary_mask", required=True, help="Binary rooftop mask to outline")
    parser.add_argument("--out_png", required=True, help="Where to save the visualization PNG")
    parser.add_argument("--min_area", type=int, default=30, help="Minimum contour area in pixels to draw (filters tiny noise specks)")
    parser.add_argument("--line_thickness", type=int, default=3)
    args = parser.parse_args()

    with rasterio.open(args.tif) as src:
        if args.aoi is not None:
            import geopandas as gpd
            from rasterio.mask import mask as rio_mask
            aoi = gpd.read_file(args.aoi)
            if aoi.crs != src.crs:
                aoi = aoi.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in aoi.geometry]
            image, _ = rio_mask(src, geoms, crop=True, filled=True, nodata=0)
        else:
            image = src.read()
    rgb = np.moveaxis(image[:3, :, :], 0, -1).astype(np.uint8)

    with rasterio.open(args.binary_mask) as src:
        mask = src.read(1)

    binary = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= args.min_area]
    print(f"Found {len(contours)} rooftop polygons (after filtering < {args.min_area}px noise)")

    # Outline-only (no fill): draw a thicker black outline first, then a
    # slightly thinner bright yellow outline on top - this "halo" effect
    # keeps the outline visible against both light and dark rooftop colors.
    outlined = rgb.copy()
    outlined_bgr = cv2.cvtColor(outlined, cv2.COLOR_RGB2BGR)
    cv2.drawContours(outlined_bgr, contours, -1, (0, 0, 0), thickness=args.line_thickness + 2)
    cv2.drawContours(outlined_bgr, contours, -1, (0, 255, 255), thickness=args.line_thickness)
    outlined = cv2.cvtColor(outlined_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Imagery")
    axes[0].axis("off")

    axes[1].imshow(outlined)
    axes[1].set_title(f"Rooftop Boundaries ({len(contours)} polygons)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out_png}")


if __name__ == "__main__":
    main()