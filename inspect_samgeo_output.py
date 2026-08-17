"""
Inspects the smoke test output: reports how many distinct object segments
SAM2 found, and saves a colorized visualization so we can see what kinds of
things it's actually segmenting (buildings vs roads vs trees vs everything).
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt


def main():
    with rasterio.open("output/smoke_test_crop.tif") as src:
        rgb = np.moveaxis(src.read([1, 2, 3]), 0, -1)

    with rasterio.open("output/smoke_test_mask.tif") as src:
        mask = src.read(1)

    unique_ids = np.unique(mask)
    print(f"Number of distinct segments found: {len(unique_ids) - (1 if 0 in unique_ids else 0)}")
    print(f"Mask value range: {mask.min()} to {mask.max()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(rgb)
    axes[0].set_title("Original Crop")
    axes[0].axis("off")

    axes[1].imshow(rgb)
    axes[1].imshow(mask, cmap="tab20", alpha=0.5)
    axes[1].set_title(f"SAM2 Segments ({len(unique_ids)} found)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("output/smoke_test_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved -> output/smoke_test_visualization.png")


if __name__ == "__main__":
    main()