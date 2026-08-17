"""
Inspects the LangSAM "building" text-prompted output: reports rooftop
coverage percentage, and saves a red-overlay visualization for a quick
visual check against the original crop.
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def main():
    with rasterio.open("output/smoke_test_crop.tif") as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)

    with rasterio.open("output/smoke_test_langsam_mask_cleaned.tif") as src:
        mask = src.read(1)

    covered_pixels = (mask > 0).sum()
    total_pixels = mask.size
    pct = covered_pixels / total_pixels * 100
    print(f"Mask value range: {mask.min()} to {mask.max()}")
    print(f"Rooftop coverage: {pct:.2f}% of the crop ({covered_pixels} / {total_pixels} pixels)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Crop")
    axes[0].axis("off")

    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    alpha = 0.5
    mask_bool = mask > 0
    overlay[mask_bool] = ((1 - alpha) * rgb[mask_bool] + alpha * red[mask_bool]).astype(np.uint8)

    axes[1].imshow(overlay)
    axes[1].set_title("LangSAM 'building' detections (red)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("output/smoke_test_langsam_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved -> output/smoke_test_langsam_visualization.png")


if __name__ == "__main__":
    main()