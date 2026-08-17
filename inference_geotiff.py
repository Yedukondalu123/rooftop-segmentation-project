"""
Rooftop segmentation inference on GeoTIFF imagery using a pretrained ReFineNet
from building-footprint-segmentation, adapted to:
  - Read large GeoTIFFs directly (rasterio), preserving georeferencing
  - Optionally clip to an AOI polygon from a matching GeoJSON
  - Tile the image into overlapping windows (replaces tt_augment, which is
    incompatible with numpy>=2.0)
  - Run ReFineNet on each tile using the repo's own preprocessing utilities
  - Stitch tile predictions back together with overlap-averaging
  - Save the result as a georeferenced probability mask + binary mask GeoTIFF
"""

import argparse
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import torch

from building_footprint_segmentation.seg.binary.models import ReFineNet, DLinkNet34
from building_footprint_segmentation.helpers.normalizer import min_max_image_net
from building_footprint_segmentation.utils.py_network import (
    to_input_image_tensor,
    add_extra_dimension,
    convert_tensor_to_numpy,
    adjust_model,
)

TILE_SIZE = 256
OVERLAP = 32  # pixels of overlap between adjacent tiles, reduces seam artifacts
STRIDE = TILE_SIZE - OVERLAP


def get_model(weight_path: str, device: str, model_type: str = "refinenet"):
    if model_type == "refinenet":
        model = ReFineNet()
    elif model_type == "dlinknet":
        model = DLinkNet34()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'refinenet' or 'dlinknet'.")

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)

    # Some checkpoints (e.g. best.pt / DlinkNet) are full training checkpoints
    # bundling model weights with optimizer state, epoch, etc. Others (e.g.
    # refine.pth) are already a plain state_dict. Detect and unwrap accordingly.
    if isinstance(checkpoint, dict) and "model" in checkpoint and "optimizer" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip a "module." prefix if the checkpoint was saved from a
    # torch.nn.DataParallel-wrapped model during training.
    state_dict = adjust_model(state_dict)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_image(tif_path: str, aoi_geojson_path: str = None):
    """
    Reads a GeoTIFF, drops the alpha band (keeps RGB), and optionally clips to
    an AOI polygon. Returns the RGB array (H, W, 3) uint8, plus the transform,
    CRS, and nodata mask for saving output later.
    """
    with rasterio.open(tif_path) as src:
        if aoi_geojson_path is not None:
            aoi = gpd.read_file(aoi_geojson_path)
            if aoi.crs != src.crs:
                aoi = aoi.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in aoi.geometry]
            image, transform = rio_mask(src, geoms, crop=True, filled=True, nodata=0)
            crs = src.crs
        else:
            image = src.read()
            transform = src.transform
            crs = src.crs

    # image shape is (bands, H, W). Keep first 3 bands (RGB), drop alpha if present.
    image = image[:3, :, :]
    image = np.moveaxis(image, 0, -1)  # -> (H, W, 3)
    image = image.astype(np.uint8)
    return image, transform, crs


def is_degenerate_tile(tile: np.ndarray) -> bool:
    """
    Returns True if the tile has zero variance in any channel (e.g. fully
    black nodata regions from AOI clipping, or flat padding). Feeding such a
    tile into min_max_image_net() divides by (max - min) = 0, producing NaN.
    Safer to skip these tiles entirely - they can't contain a rooftop.
    """
    for c in range(tile.shape[2]):
        channel = tile[:, :, c]
        if channel.max() == channel.min():
            return True
    return False


def water_score(tile: np.ndarray) -> float:
    """
    Very simple heuristic water detector: water tends to be relatively dark,
    low-saturation/blue-greenish, and low local texture variance compared to
    rooftops. Returns a 0-1 "looks like water" score used to suppress false
    positive rooftop predictions over sea/ocean/large water bodies, which
    the model was never trained to reject (INRIA has no water class).
    This is a coarse heuristic, not a real water classifier - tune/replace
    with NDWI from a real multispectral source if available.
    """
    r = tile[:, :, 0].astype(np.float32)
    g = tile[:, :, 1].astype(np.float32)
    b = tile[:, :, 2].astype(np.float32)
    brightness = (r + g + b) / 3.0
    blue_dominant = (b > r) & (b >= g)
    low_texture = brightness.std() < 15  # water is visually smoother than rooftops/urban texture
    frac_blue_dominant = blue_dominant.mean()
    score = 0.0
    if low_texture:
        score += 0.5
    score += 0.5 * frac_blue_dominant
    return min(score, 1.0)


def predict_tiled(model, image: np.ndarray, device: str, suppress_water: bool = False):
    """
    Slides a TILE_SIZE x TILE_SIZE window (with OVERLAP) across the image,
    runs the model on each tile, and stitches predictions back together using
    overlap-averaging to avoid seam artifacts. Skips fully-degenerate
    (nodata) tiles to avoid NaN, and optionally suppresses likely water tiles.
    """
    h, w, _ = image.shape

    # Pad the image so it's evenly divisible into full tiles (reflect padding
    # avoids introducing hard black edges that could confuse the model).
    pad_h = (TILE_SIZE - (h - TILE_SIZE) % STRIDE) % STRIDE if h > TILE_SIZE else TILE_SIZE - h
    pad_w = (TILE_SIZE - (w - TILE_SIZE) % STRIDE) % STRIDE if w > TILE_SIZE else TILE_SIZE - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw, _ = padded.shape

    prob_sum = np.zeros((ph, pw), dtype=np.float32)
    prob_count = np.zeros((ph, pw), dtype=np.float32)

    y_positions = list(range(0, ph - TILE_SIZE + 1, STRIDE))
    x_positions = list(range(0, pw - TILE_SIZE + 1, STRIDE))
    if y_positions[-1] != ph - TILE_SIZE:
        y_positions.append(ph - TILE_SIZE)
    if x_positions[-1] != pw - TILE_SIZE:
        x_positions.append(pw - TILE_SIZE)

    total_tiles = len(y_positions) * len(x_positions)
    done = 0

    skipped = 0
    for y in y_positions:
        for x in x_positions:
            tile = padded[y:y + TILE_SIZE, x:x + TILE_SIZE, :]

            if is_degenerate_tile(tile):
                # fully-black nodata tile (outside AOI) - skip, leave probability at 0
                skipped += 1
                done += 1
                print(f"  tile {done}/{total_tiles} (skipped {skipped} nodata)", end="\r")
                continue

            norm = min_max_image_net(tile)
            tensor = to_input_image_tensor(norm)
            tensor = add_extra_dimension(tensor).to(device)

            with torch.no_grad():
                pred = model(tensor)
                pred = pred.sigmoid()

            pred_np = convert_tensor_to_numpy(pred)  # shape (1, 1, TILE_SIZE, TILE_SIZE)
            pred_np = pred_np[0, 0]

            if suppress_water:
                wscore = water_score(tile)
                if wscore > 0.5:
                    # scale down predictions on tiles that look like water
                    pred_np = pred_np * (1.0 - wscore)

            prob_sum[y:y + TILE_SIZE, x:x + TILE_SIZE] += pred_np
            prob_count[y:y + TILE_SIZE, x:x + TILE_SIZE] += 1.0

            done += 1
            print(f"  tile {done}/{total_tiles}", end="\r")

    print()
    prob_count[prob_count == 0] = 1.0
    prob_avg = prob_sum / prob_count

    # crop back to original (unpadded) size
    prob_avg = prob_avg[:h, :w]
    return prob_avg


def save_mask(prob_mask: np.ndarray, transform, crs, out_path_prefix: str, threshold: float = 0.5):
    prob_path = f"{out_path_prefix}_prob.tif"
    binary_path = f"{out_path_prefix}_binary.tif"

    h, w = prob_mask.shape

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(prob_path, "w", **profile) as dst:
        dst.write(prob_mask.astype(np.float32), 1)

    binary_mask = (prob_mask >= threshold).astype(np.uint8) * 255
    profile_bin = dict(profile)
    profile_bin["dtype"] = "uint8"
    with rasterio.open(binary_path, "w", **profile_bin) as dst:
        dst.write(binary_mask, 1)

    print(f"Saved probability mask -> {prob_path}")
    print(f"Saved binary mask      -> {binary_path}")
    return prob_path, binary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif", required=True, help="Path to input GeoTIFF")
    parser.add_argument("--aoi", default=None, help="Path to AOI GeoJSON (optional)")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth or .pt)")
    parser.add_argument("--model_type", default="refinenet", choices=["refinenet", "dlinknet"], help="Which architecture the weights belong to")
    parser.add_argument("--out", required=True, help="Output path prefix, e.g. output/dataset1")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binary mask (default 0.5)")
    parser.add_argument("--suppress_water", action="store_true", help="Apply heuristic water suppression (reduces false positives over sea/ocean)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model = get_model(args.weights, device, model_type=args.model_type)

    print(f"Loading image: {args.tif}")
    image, transform, crs = load_image(args.tif, args.aoi)
    print(f"Image shape (H, W, C): {image.shape}")

    print("Running tiled inference...")
    prob_mask = predict_tiled(model, image, device, suppress_water=args.suppress_water)

    save_mask(prob_mask, transform, crs, args.out, threshold=args.threshold)


if __name__ == "__main__":
    main()