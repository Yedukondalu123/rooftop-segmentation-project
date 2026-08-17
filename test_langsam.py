"""
Smoke test: uses LangSAM (SAM + Grounding DINO) with a broadened text prompt
to segment building rooftops, then applies morphological cleanup to remove
thin false-positive bleed onto roads/tree edges while preserving blocky
rooftop shapes.
"""

from samgeo.text_sam import LangSAM
import numpy as np
import rasterio
from scipy import ndimage


def clean_mask(mask, erosion_iterations=2):
    """
    Removes thin, elongated false-positive regions (roads, tree-canopy edges)
    while preserving blocky rooftop shapes, using morphological erosion
    followed by dilation (an "opening" operation) - erosion shrinks all
    regions, which eliminates thin/narrow shapes entirely, then dilation
    grows the surviving (blocky) regions back toward their original size.
    """
    binary = mask > 0
    eroded = ndimage.binary_erosion(binary, iterations=erosion_iterations)
    opened = ndimage.binary_dilation(eroded, iterations=erosion_iterations)
    return (opened.astype(np.uint8)) * 255


def main():
    crop_path = "output/smoke_test_crop.tif"
    mask_path = "output/smoke_test_langsam_mask.tif"
    cleaned_path = "output/smoke_test_langsam_mask_cleaned.tif"

    print("Loading LangSAM (GroundingDINO + SAM)... this downloads model weights on first run.")
    sam = LangSAM(model_type="sam2-hiera-tiny")

    text_prompt = "building . roof . rooftop . house . structure"

    print(f"Running text-prompted segmentation with prompt: '{text_prompt}'")
    sam.predict(
        image=crop_path,
        text_prompt=text_prompt,
        box_threshold=0.24,
        text_threshold=0.24,
        output=mask_path,
        mask_multiplier=255,
        dtype="uint8",
    )

    print(f"Saved raw mask -> {mask_path}")

    print("Applying morphological cleanup to remove road/tree bleed...")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()

    cleaned = clean_mask(mask, erosion_iterations=6)

    with rasterio.open(cleaned_path, "w", **profile) as dst:
        dst.write(cleaned, 1)

    print(f"Saved cleaned mask -> {cleaned_path}")
    print("Smoke test complete.")


if __name__ == "__main__":
    main()