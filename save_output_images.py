"""
Saves ONLY the final rooftop-outlined result as a standalone PNG - no
side-by-side comparison panel, no matplotlib titles/axes. This is the
"just the output" image, sized to match the original image's pixel
dimensions, suitable as a clean deliverable.
"""

import argparse
import numpy as np
import rasterio
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Original input GeoTIFF")
    parser.add_argument("--aoi", default=None, help="AOI GeoJSON (optional, must match what was used for the mask)")
    parser.add_argument("--binary_mask", required=True, help="Binary rooftop mask")
    parser.add_argument("--out_original", required=True, help="Where to save the plain original image PNG")
    parser.add_argument("--out_result", required=True, help="Where to save the outlined result PNG")
    parser.add_argument("--min_area", type=int, default=30)
    parser.add_argument("--line_thickness", type=int, default=10)
    parser.add_argument("--style", choices=["outline", "fill"], default="outline")
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
    print(f"Found {len(contours)} rooftop polygons")

    result_bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

    if args.style == "fill":
        overlay = result_bgr.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), thickness=-1)
        result_bgr = cv2.addWeighted(overlay, 0.5, result_bgr, 0.5, 0)
        cv2.drawContours(result_bgr, contours, -1, (0, 200, 200), thickness=args.line_thickness)

        cv2.drawContours(result_bgr, contours, -1, (0, 0, 0), thickness=args.line_thickness + 2)
        cv2.drawContours(result_bgr, contours, -1, (0, 255, 255), thickness=args.line_thickness)

    # Save the plain original (RGB -> BGR for cv2.imwrite) and the result,
    # each as its own standalone image at full original resolution.
    original_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.out_original, original_bgr)
    cv2.imwrite(args.out_result, result_bgr)

    print(f"Saved original -> {args.out_original}")
    print(f"Saved result   -> {args.out_result}")


if __name__ == "__main__":
    main()