"""
Inspects a probability mask (*_prob.tif from inference_geotiff.py) to see how
confident the model actually was, and how detection rate changes at
different thresholds. Helps distinguish "model unsure, threshold too strict"
from "model genuinely didn't detect anything there".
"""

import argparse
import numpy as np
import rasterio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_tif", required=True, help="Path to *_prob.tif")
    args = parser.parse_args()

    with rasterio.open(args.prob_tif) as src:
        prob = src.read(1)

    total_pixels = prob.size
    print(f"Total pixels: {total_pixels}")
    print(f"Min: {prob.min():.4f}  Max: {prob.max():.4f}  Mean: {prob.mean():.4f}  Median: {np.median(prob):.4f}")
    print()
    print("Percentage of pixels classified as 'rooftop' at different thresholds:")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pct = (prob >= t).sum() / total_pixels * 100
        print(f"  threshold {t}: {pct:.2f}%")

    print()
    print("Histogram of probability values (10 bins):")
    hist, edges = np.histogram(prob, bins=10, range=(0, 1))
    for i in range(len(hist)):
        bar = "#" * int(hist[i] / total_pixels * 500)
        print(f"  {edges[i]:.1f}-{edges[i+1]:.1f}: {hist[i]:>10} {bar}")


if __name__ == "__main__":
    main()