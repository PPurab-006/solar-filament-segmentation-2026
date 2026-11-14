# Solar Filament Segmentation Challenge 2026

Official submission repository for the Kaggle **Solar Filament Segmentation Challenge 2026**.

This repository contains the complete, reproducible inference pipeline, model weights, post-processing rules, unit tests, and validation scripts for the winning solution.

---

## Architecture & Solution Overview

- **Model Backbone**: Segmentation Models PyTorch (SMP) `Unet` with `tu-convnext_tiny` encoder initialized from ImageNet weights.
- **Input Channels (3 Channels)**: Primary intensity-normalized full-disk solar H-alpha image plus 2 spatial context channels (polar coordinate radius $\rho$ and solar disk interior mask).
- **Intensity Normalization**: `zscore_v1` specification (mean/std normalization computed over solar disk interior pixels with 1.01 off-limb margin).
- **Inference Engine**: Seamless sliding-window inference with 2D Hann (raised-cosine) window tile blending (512x512 tile size, stride 256), eliminating boundary stitching artifacts.
- **Weights**: Half-precision (`float16`) Exponential Moving Average (EMA, $\beta=0.999$) weights stored at `weights/r1_convnext_tiny_ema_weights.pt` (~61 MB).
- **Post-Processing Pipeline**: Pure morphological rule-based filtering (binarization, gap closing, morphological hole filling, solar limb boundary clipping, connected components labeling, area and peak probability filtering).

---

## Submission Configurations

Two standardized configurations are provided, both operating on the **identical frozen weights**:

| Configuration | Config Path | Mask Threshold | Close Radius | Peak Threshold | Min Area | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Config A** | `configs/final_A.yaml` | 0.55 | 3 | 0.75 | 200 px | R1b optimal configuration (reproduces `expected/submission_R1b.csv`) |
| **Config B** | `configs/final_B.yaml` | 0.60 | 2 | 0.75 | 300 px | Plateau-robust configuration (reproduces `outputs/submission_R1_plateau.csv`) |

Neither configuration uses any learned instance scorer, test-time augmentation (TTA), or model ensembling.

---

## Environment & Requirements

- **Tested Python Versions**: Python 3.10, 3.11, 3.12
- **Hardware**: Tested on CUDA-enabled GPUs (NVIDIA P100, T4, RTX 3090, A100) and CPU fallback.

### Installation

```bash
git clone <REPO_URL>
cd solar-filament-segmentation-2026
pip install -r requirements.txt
```

---

## How to Run & Reproduce Submissions

### 1. Command-Line Reproduction

To generate predictions for the test set, run the unified entry point `solarseg.run`:

```bash
# Generate Config A (R1b Rule):
python -m solarseg.run \
    --config configs/final_A.yaml \
    --images /path/to/test_images \
    --out submission.csv

# Generate Config B (Plateau Rule):
python -m solarseg.run \
    --config configs/final_B.yaml \
    --images /path/to/test_images \
    --out submission.csv
```

### 2. Standalone Verification Notebook

A self-contained notebook is located at `notebooks/reproduce_submission.ipynb`. It:
1. Auto-detects available compute devices (CUDA GPU / CPU).
2. Executes end-to-end inference on the test set.
3. Automatically runs `scripts/check_submission.py` to assert valid header, dimensions, non-emptiness, and mutual disjointness.
4. Asserts that generated masks match `expected/submission_R1b.csv` at $\text{IoU} > 0.9$ with match fraction $> 0.999$.
5. Measures total and per-image runtime.

To run a quick smoke-test on 5 images, simply toggle `QUICK = True` in the notebook parameters.

### 3. Submission Integrity Check

Validate any submission CSV using the standalone checker:

```bash
python scripts/check_submission.py --csv-path submission.csv --test-dir /path/to/test_images
```

---

## Rules Compliance & Integrity Checklist

This solution strictly adheres to the competition rules (see `docs/RULES_CHECKLIST.md`):
- **Competition Data Only**: Only official competition images and annotations were used for model development. No external datasets, no solar observatory catalogs, and no MAGFiLO external releases.
- **Zero Test-Time Data Leakage**: At inference time, only the single test image's own pixel values are read. No filenames, timestamps, observatory site names, or neighbouring temporal frames are accessed.
- **Hard Rule Compliance**:
  - All masks are encoded as valid COCO RLE strings on fixed $2048 \times 2048$ grids.
  - Zero empty masks exist.
  - All predicted filament instances per image are strictly mutually disjoint (guaranteed by connected component labeling).
  - All file sizes are strictly under 100 MB.

---

## Evaluator Reconciliation & Methodology Note

During development, an evaluation discrepancy was identified and resolved:
1. **Multi-Annotator Separation**: In earlier exploratory scripts, annotations were improperly aggregated across annotator drawings for the same image. The evaluation code was refactored so that every validation image separates distinct annotator drawings (`drawing_id`), computing Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) per drawing before averaging across images.
2. **Scorer Exact Agreement**: The fast crop-IoU evaluator in `solarseg` was benchmarked against the competition organizers' self-evaluation notebook logic on identical validation images. The implementations match to 8 decimal places ($\Delta = 0.00000000$ at $\text{PQ} = 0.4436$), confirming absolute mathematical fidelity.
3. **Exclusion of Complex Heuristics**: Extensive cross-fit validation demonstrated that complex post-hoc instance scoring and model ensembling did not generalize stably over the clean rule-based morphological pipeline. Per the project charter, all learned scorers, TTA, and ensembling were removed from the final repository in favor of the clean, reproducible single-model architecture.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
