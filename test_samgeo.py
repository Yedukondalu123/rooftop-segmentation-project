"""
Smoke test: runs SAM2 (hiera-tiny checkpoint) automatic mask generation on a
small crop of a GeoTIFF, to confirm the model downloads/loads correctly and
fits within a 4GB GPU before we scale up to full-size images.
"""

import rasterio
from rasterio.windows import Window
import numpy as np
from samgeo import SamGeo2

CROP_SIZE = 600  # small test crop, pixels


def main():
    tif_path = "data/dataset1.tif"
    crop_path = "output/smoke_test_crop.tif"

    # Cut out a small center crop to test on, keeping georeferencing intact.
    with rasterio.open(tif_path) as src:
        cx, cy = src.width // 2, src.height // 2
        window = Window(cx - CROP_SIZE // 2, cy - CROP_SIZE // 2, CROP_SIZE, CROP_SIZE)
        transform = src.window_transform(window)
        arr = src.read([1, 2, 3], window=window)  # RGB only, drop alpha
        profile = src.profile.copy()
        profile.update({
            "height": CROP_SIZE,
            "width": CROP_SIZE,
            "count": 3,
            "transform": transform,
        })
        with rasterio.open(crop_path, "w", **profile) as dst:
            dst.write(arr)

    print(f"Saved test crop -> {crop_path}")

    print("Loading SAM2 (hiera-tiny)... this downloads the checkpoint on first run.")
    sam = SamGeo2(
        model_id="sam2-hiera-tiny",
        automatic=True,
        apply_postprocessing=False,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25.0,
        use_m2m=True,
    )

    print("Running automatic mask generation on the crop...")
    mask_path = "output/smoke_test_mask.tif"
    sam.generate(crop_path, mask_path)

    print(f"Saved mask -> {mask_path}")
    print("Smoke test complete.")


if __name__ == "__main__":
    main()