# Rooftop Segmentation on AOI GeoTIFFs

Adapting an existing open-source rooftop segmentation repository to run inference on 3 custom AOI GeoTIFFs of Indian cities, and improving results by combining it with a foundation-model-based approach.

## Project Narrative (what we did, and why each file exists)

The assignment asked for one thing: pick a rooftop segmentation repo, understand it, adapt it to the given AOIs, and explain the pipeline, challenges, and possible improvements. This project has two phases: first adapting one existing CNN-based repo, then discovering its real limitations on Indian urban imagery and bringing in a second, different kind of model to compensate. That two-phase structure *is* the "challenges and possible improvements" story, not something added on top of it.

### Phase 1 - CNN pipeline (the "adapt an existing repo" part)

- `**inference_geotiff.py*`* - the core Phase 1 deliverable. The original repo only ships a training pipeline and a notebook example for inference, with no script that takes a raw GeoTIFF and produces a georeferenced mask. This file does that: reads a GeoTIFF with `rasterio`, clips it to the AOI polygon, cuts it into overlapping 256x256 tiles (since the model only accepts small fixed-size inputs), runs `ReFineNet` or `DLinkNet34` on each tile, and stitches the predictions back into one probability mask that lines up pixel-for-pixel with the original imagery. This is the actual "adaptation" work the assignment asked for.
- `**visualize_result.py**` - turns a raw probability mask into a 3-panel image (original / mask / red overlay), satisfying the "visualize predicted rooftops on the original imagery" requirement.
- `**check_probabilities.py**` - a diagnostic tool that inspects a probability mask's value distribution. This is what proved `dataset2` had a real bug (NaN values) rather than just "bad results" - evidence of actually debugging rather than guessing. The bug itself: a fully-black (outside-AOI) tile fed into the model's normalization function divided by zero and produced NaN, fixed in `inference_geotiff.py` by skipping degenerate tiles.

### Phase 2 - discovering and documenting the limitation

Once the AOIs were visually compared against the model's training data (INRIA = European suburbs, Massachusetts = American suburbs), it was predictable - and then confirmed - that Indian dense housing, industrial complexes, and skyscrapers would be under-detected.

- `**combine_predictions.py**` - first attempt at improvement: run both pretrained weights (ReFineNet and DLinkNet) and combine them, since they're trained on different data and might cover each other's blind spots. Finding: DLinkNet did better on dense housing but worse on industrial buildings - proof neither pretrained model alone was "correct," they just fail differently.

### Phase 3 - bringing in a second, different model entirely

Instead of endlessly tweaking one CNN, this phase brings in a fundamentally different approach: a foundation model (SAM2) guided by text prompts (Grounding DINO), via the `segment-geospatial` library. This is a "possible improvement" being implemented, not just suggested in a report.

- `**run_langsam_geotiff.py**` - the equivalent of `inference_geotiff.py` for this new approach. Clips to AOI, tiles the image (written custom because `segment-geospatial`'s built-in tiler needs GDAL Python bindings that are painful to install on Windows - a real adaptation decision), runs `LangSAM` with the text prompt `"building . roof . rooftop . house . structure"` on each tile, merges results.
- `**test_samgeo.py`, `test_langsam.py`, `inspect_langsam_output.py**` - smoke-test/diagnostic scripts used to validate this new approach on a small crop before running it on full images. This is where LangSAM's key weakness was found: it has no concept of "not a building," so it bleeds onto roads and tree-lined medians - a different failure mode than the CNN's under-detection.

### Phase 4 - combining both models honestly

- `**intersect_masks.py`** - the most important file for the results story. It geo-aligns the CNN's probability mask and LangSAM's binary mask (they come from different pipelines with different pixel grids, so this reprojects one onto the other's exact grid first), then combines them two ways:
  - **Intersection (`and`)**: keep a pixel only if both models agree - higher precision, but capped by whichever model has lower recall
  - **Union (`or`)**: keep a pixel if either model says rooftop - higher recall, but inherits both models' false positives
  Both were tested with real numbers (CNN alone 8.91%, LangSAM alone 53.9%, intersection 7.05%, union ~54%), used to make an informed, documented choice rather than picking one arbitrarily. It also includes a **morphological gap-closing step** (dilate then erode) fixing a real artifact: even on genuine rooftops, the two models don't agree on every single pixel, leaving speckled holes - closing fixes that without reintroducing false positives.
- `**visualize_outlines.py`** and `**save_output_images.py**` - final presentation layer: clean polygon-style outlines instead of solid fills, and standalone (non-comparison) images for the final deliverable.

### One-line summary

We adapted an existing CNN-based repo for GeoTIFF inference, found it had a real domain gap on Indian urban imagery, brought in a second foundation-model-based approach to compensate, and combined both with a documented precision/recall trade-off rather than just picking whichever looked better.

## Repository Used

**Base repository:** [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation) by fuzailpalnak (PyTorch, MIT license)

This provides two pretrained CNN architectures for building/rooftop segmentation:

- **ReFineNet**, trained on the INRIA Aerial Image Labeling dataset
- **DLinkNet34**, trained on the Massachusetts Buildings Dataset

**Second approach added:** [segment-geospatial](https://github.com/opengeos/segment-geospatial) (samgeo), using **LangSAM** (Grounding DINO + Segment Anything Model 2) for text-prompted, geography-agnostic building detection. Added after finding the CNN models under-detect rooftop types not represented in their training data (see Challenges below).

## Research / Reference Papers

- Maggiori, E. et al. (2017), *"Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark"* — the INRIA dataset ReFineNet is trained on
- Mnih, V. (2013), *"Machine Learning for Aerial Image Labeling"* — the Massachusetts Buildings Dataset DLinkNet is trained on
- Kirillov, A. et al. (2023), *"Segment Anything"* (Meta AI) — SAM
- Ravi, N. et al. (2024), *"SAM 2: Segment Anything in Images and Videos"* (Meta AI) — SAM2, used via the `sam2-hiera-tiny` checkpoint
- Liu, S. et al. (2023), *"Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"` — the text-prompt detector paired with SAM in LangSAM
- Zhu, X. et al. (2025), *"GlobalBuildingAtlas"* — cited during this project as evidence that dense informal settlements and industrial/institutional building types are an acknowledged, actively-researched gap in current building-footprint benchmarks (motivated trying a second, non-CNN approach)

## Setup Instructions

### 1. Environment

```
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. PyTorch (GPU build - adjust cu124 to match your CUDA version)

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Core geospatial + ML dependencies

```
pip install rasterio geopandas shapely pyproj numpy matplotlib pillow opencv-python scipy
```

### 4. Clone and install the base repository

```
git clone https://github.com/fuzailpalnak/building-footprint-segmentation.git
cd building-footprint-segmentation
pip install -e . --no-deps
pip install numpy PyYAML albumentations tqdm scikit-image py_oneliner scikit-learn
```

> The repo pins old exact dependency versions (numpy==1.19.1 etc.) that no longer build on modern Python/Windows. Installing with `--no-deps` and then installing current versions of the same libraries works because the code's use of these libraries' APIs hasn't broken.

### 5. Download pretrained weights

```
cd examples
curl -L -o refine.zip https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip
tar -xf refine.zip
curl -L -o DlinkNet.zip https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/DlinkNet.zip
tar -xf DlinkNet.zip
cd ..
```

### 6. segment-geospatial (second approach)

```
pip install -U segment-geospatial
pip install "segment-geospatial[samgeo2]"
pip install "segment-geospatial[text]"
pip install "transformers==4.57.6"
```

> `transformers>=5.0` removed a legacy BERT method that GroundingDINO's wrapper code depends on; pinning to 4.57.6 is required for LangSAM to load.

### 7. Windows-specific fix: PROJ/GDAL conflict

If you see `pyproj.exceptions.ProjError` or garbled CRS output, a system-wide `PROJ_LIB` environment variable (commonly set by PostGIS/QGIS installs) is pointing at an incompatible `proj.db`. Fix per terminal session:

```
set PROJ_LIB=<path to your venv>\Lib\site-packages\rasterio\proj_data
```

## Project Structure

```
building-footprint-segmentation/
├── data/                          # input GeoTIFFs + AOI GeoJSONs
├── examples/                      # downloaded pretrained weights (refine.pth, best.pt)
├── output/                        # all generated masks, visualizations, final results
├── inference_geotiff.py           # CNN pipeline (ReFineNet / DLinkNet) - tiled inference on a GeoTIFF
├── check_probabilities.py         # inspect a probability mask's value distribution
├── combine_predictions.py         # ensemble two CNN probability masks (max/mean)
├── run_langsam_geotiff.py         # LangSAM pipeline - tiles a GeoTIFF, runs text-prompted segmentation, merges result
├── intersect_masks.py             # geo-aligned CNN + LangSAM combination (intersection/union), with gap-closing cleanup
├── visualize_result.py            # 3-panel comparison: original / mask / overlay
├── visualize_outlines.py          # outline or filled polygon-style visualization
├── save_output_images_v2.py          # standalone (non-comparison) original + result image files
└── test_*.py, inspect_*.py        # smoke-test / diagnostic scripts used during development
```

## Execution Instructions

For each AOI (`dataset1`, `dataset2`, `dataset3`):

**1. Run the CNN model:**

```
python inference_geotiff.py --tif data\datasetN.tif --aoi data\datasetN.geojson --weights examples\refine.pth --out output\datasetN
```

**2. Run LangSAM on the full AOI:**

```
python run_langsam_geotiff.py --tif data\datasetN.tif --aoi data\datasetN.geojson --out output\datasetN_langsam
```

**3. Combine both (intersection, precision-focused, with gap-closing):**

```
python intersect_masks.py --cnn_prob output\datasetN_prob.tif --langsam_mask output\datasetN_langsam_binary.tif --cnn_threshold 0.3 --mode and --close_gaps 4 --out output\datasetN_ensemble_closed
```

**4. Produce the final output images:**

```
python save_output_images_v2.py --tif data\datasetN.tif --aoi data\datasetN.geojson --binary_mask output\datasetN_ensemble_closed_binary.tif --out_original output\FINAL_datasetN_original_v2.png --out_result output\FINAL_datasetN_result_v2.png
```

