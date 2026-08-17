"""
Overlays a predicted rooftop binary mask on top of the original RGB GeoTIFF
imagery, and saves a side-by-side + overlay PNG for visual inspection.
"""

import argparse
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import matplotlib.pyplot as plt


def load_rgb(tif_path, aoi_geojson_path=None):
    with rasterio.open(tif_path) as src:
        if aoi_geojson_path is not None:
            aoi = gpd.read_file(aoi_geojson_path)
            if aoi.crs != src.crs:
                aoi = aoi.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in aoi.geometry]
            image, _ = rio_mask(src, geoms, crop=True, filled=True, nodata=0)
        else:
            image = src.read()
    image = image[:3, :, :]
    image = np.moveaxis(image, 0, -1)
    return image.astype(np.uint8)


def load_single_band(tif_path):
    with rasterio.open(tif_path) as src:
        band = src.read(1)
    return band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Original input GeoTIFF")
    parser.add_argument("--aoi", default=None, help="AOI GeoJSON used during inference (optional, must match)")
    parser.add_argument("--binary_mask", required=True, help="Path to *_binary.tif produced by inference_geotiff.py")
    parser.add_argument("--out_png", required=True, help="Where to save the visualization PNG")
    args = parser.parse_args()

    rgb = load_rgb(args.tif, args.aoi)
    mask = load_single_band(args.binary_mask)  # 0 or 255

    # Build a red overlay wherever the mask predicts "rooftop"
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255  # red channel
    alpha = 0.45
    mask_bool = mask > 0
    overlay[mask_bool] = (
        (1 - alpha) * rgb[mask_bool] + alpha * red[mask_bool]
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    axes[0].imshow(rgb)
    axes[0].set_title("Original Imagery")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Rooftop Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Rooftops Overlaid on Imagery")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    print(f"Saved visualization -> {args.out_png}")


if __name__ == "__main__":
    main()