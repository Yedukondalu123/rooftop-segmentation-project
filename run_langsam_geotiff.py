"""
Full LangSAM rooftop segmentation pipeline for a single AOI GeoTIFF:
  1. Clip the GeoTIFF to its AOI polygon (same approach as the CNN pipeline)
  2. Split the clipped image into tiles (samgeo's built-in split_raster)
  3. Run LangSAM with text_prompt="building" across all tiles
  4. Merge tile predictions into one georeferenced building mask

Usage:
    python run_langsam_geotiff.py --tif data\\dataset2.tif --aoi data\\dataset2.geojson --out output\\dataset2_langsam
"""

import argparse
import os
import shutil
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import Window
from rasterio.merge import merge
import geopandas as gpd
from samgeo.text_sam import LangSAM


def clip_to_aoi(tif_path, aoi_path, clipped_out_path):
    with rasterio.open(tif_path) as src:
        aoi = gpd.read_file(aoi_path)
        if aoi.crs != src.crs:
            aoi = aoi.to_crs(src.crs)
        geoms = [geom.__geo_interface__ for geom in aoi.geometry]
        image, transform = rio_mask(src, geoms, crop=True, filled=True, nodata=0)
        # keep RGB only (drop alpha) to match what LangSAM/SAM expects
        image = image[:3, :, :]
        profile = src.profile.copy()
        profile.update({
            "height": image.shape[1],
            "width": image.shape[2],
            "count": 3,
            "transform": transform,
        })
        with rasterio.open(clipped_out_path, "w", **profile) as dst:
            dst.write(image)
    print(f"Clipped to AOI -> {clipped_out_path}")


def tile_raster(clipped_path, tiles_dir, tile_size):
    """
    Simple rasterio-based tiler (no GDAL/osgeo dependency, unlike samgeo's
    own split_raster). Cuts the image into tile_size x tile_size windows
    (with a final partial tile at each edge if it doesn't divide evenly),
    each saved as its own small georeferenced GeoTIFF.
    """
    os.makedirs(tiles_dir, exist_ok=True)
    with rasterio.open(clipped_path) as src:
        w, h = src.width, src.height
        profile = src.profile.copy()
        count = 0
        for top in range(0, h, tile_size):
            for left in range(0, w, tile_size):
                win_w = min(tile_size, w - left)
                win_h = min(tile_size, h - top)
                window = Window(left, top, win_w, win_h)
                transform = src.window_transform(window)
                data = src.read(window=window)

                tile_profile = profile.copy()
                tile_profile.update({
                    "height": win_h,
                    "width": win_w,
                    "transform": transform,
                })
                tile_path = os.path.join(tiles_dir, f"tile_{top}_{left}.tif")
                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(data)
                count += 1
    return count


def merge_tiles(masks_dir, out_path):
    """
    Merges individually-predicted tile masks back into one georeferenced
    mosaic, using rasterio's merge (avoids needing samgeo's GDAL-based
    merge utility).
    """
    tile_paths = [
        os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith(".tif")
    ]
    srcs = [rasterio.open(p) for p in tile_paths]
    mosaic, out_transform = merge(srcs)
    out_profile = srcs[0].profile.copy()
    out_profile.update({
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
    })
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    parser.add_argument("--aoi", required=True, help="Path to AOI GeoJSON")
    parser.add_argument("--out", required=True, help="Output path prefix, e.g. output/dataset2_langsam")
    parser.add_argument("--tile_size", type=int, default=600, help="Tile size in pixels (default 600)")
    parser.add_argument("--text_prompt", default="building . roof . rooftop . house . structure", help="Text prompt(s) for LangSAM, period-separated for multiple terms")
    parser.add_argument("--box_threshold", type=float, default=0.24)
    parser.add_argument("--text_threshold", type=float, default=0.24)
    args = parser.parse_args()

    work_dir = f"{args.out}_work"
    os.makedirs(work_dir, exist_ok=True)
    tiles_dir = os.path.join(work_dir, "tiles")
    masks_dir = os.path.join(work_dir, "masks")

    clipped_path = os.path.join(work_dir, "clipped.tif")
    clip_to_aoi(args.tif, args.aoi, clipped_path)

    print(f"Tiling into {args.tile_size}x{args.tile_size} pieces...")
    if os.path.exists(tiles_dir):
        shutil.rmtree(tiles_dir)
    num_tiles = tile_raster(clipped_path, tiles_dir, args.tile_size)
    print(f"Created {num_tiles} tiles.")

    print("Loading LangSAM (sam2-hiera-tiny)...")
    sam = LangSAM(model_type="sam2-hiera-tiny")

    os.makedirs(masks_dir, exist_ok=True)
    tile_files = sorted(f for f in os.listdir(tiles_dir) if f.endswith(".tif"))

    print(f"Running prediction on each tile with text_prompt='{args.text_prompt}'...")
    for i, fname in enumerate(tile_files, 1):
        tile_path = os.path.join(tiles_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        try:
            sam.predict(
                image=tile_path,
                text_prompt=args.text_prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                output=mask_path,
                mask_multiplier=255,
                dtype="uint8",
            )
        except Exception as e:
            print(f"  tile {i}/{num_tiles} ({fname}): prediction failed ({e}), writing blank mask")
            with rasterio.open(tile_path) as src:
                profile = src.profile.copy()
            profile.update({"count": 1, "dtype": "uint8"})
            with rasterio.open(mask_path, "w", **profile) as dst:
                dst.write(np.zeros((profile["height"], profile["width"]), dtype=np.uint8), 1)
        print(f"  tile {i}/{num_tiles} done", end="\r")

    print()
    print("Merging tile masks into final mosaic...")
    merged_dst = f"{args.out}_binary.tif"
    merge_tiles(masks_dir, merged_dst)
    print(f"Saved final merged mask -> {merged_dst}")


if __name__ == "__main__":
    main()