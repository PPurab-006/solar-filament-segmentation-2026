# Solar Filament Segmentation Challenge 2026

Official submission repository for the Kaggle **Solar Filament Segmentation Challenge 2026**.

This repository contains the complete, reproducible inference pipeline, model weights, post-processing rules, unit tests, and verification notebook for the final submission.

---

## Architecture & Solution Overview

- **Final Model**: Half-precision (`float16`) Exponential Moving Average (EMA, $\beta=0.999$) weights stored at `weights/r1_convnext_tiny_ema_weights.pt` (61.0 MB).
- **Architecture**: Segmentation Models PyTorch (SMP) `Unet` with `tu-convnext_tiny` encoder initialized from ImageNet weights.
- **Input Channels (3 Channels)**:
  1. *Channel 1 (Normalized H-alpha)*: Intensity z-score normalized over on-disk pixels.
  2. *Channel 2 (Polar Radius $\rho$)*: Normalized radial coordinate $\rho = r / R$, with values beyond $\rho > 1.01$ zeroed out.
  3. *Channel 3 (Spatial Context)*: Multi-scale context window ($1536 \times 1536$ downscaled $3\times$ to $512 \times 512$).
- **Preprocessing (`zscore_v1`)**:
  - Solar disk detection: Intensity threshold at pixel value $> 20$ extracts disk mask.
  - Limb fitting: `cv2.minEnclosingCircle` fits the minimum enclosing circle to determine disk center $(c_x, c_y)$ and radius $R$.
  - Intensity normalization: Mean and standard deviation are computed strictly over on-disk pixels ($I > 20$).
  - Coordinate channel: Radial distance normalized to disk radius; values beyond $1.01 \times R$ set strictly to zero.
  - Context channel: $1536 \times 1536$ bounding box centered on each tile is resized to $512 \times 512$.
- **Inference Engine**: Seamless sliding-window inference with 2D Hann (raised-cosine) window tile blending ($512 \times 512$ tile size, stride 256), completely eliminating boundary tile stitching artifacts.
- **Post-Processing**: Pure morphological rule-based filtering (probability thresholding, morphological closing, connected component labeling, minimum area filtering, and peak probability thresholding).
- **Exclusion of Complex Heuristics**: No learned instance scorer, no test-time augmentation (TTA), and no model ensembling are used. Both final configurations run as a fast, clean, single model.

---

## Submission Configurations

Two standardized configurations are provided, both operating on the **identical frozen weights** (`weights/r1_convnext_tiny_ema_weights.pt`):

| Configuration | Config Path | Mask Threshold ($t_m$) | Close Radius ($cr$) | Peak Threshold ($t_p$) | Min Area ($ma$) | Validation PQ (141 obs) | Public LB Score | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Config A** | `configs/final_A.yaml` | 0.55 | 3 | 0.75 | 200 px | **0.4529** | **0.37** | R1b optimal configuration (reproduces `expected/submission_R1b.csv`) |
| **Config B** | `configs/final_B.yaml` | 0.60 | 2 | 0.75 | 300 px | **0.4453** | **0.37** | Plateau-robust configuration |

- Both configurations use the exact same model weights and inference path.
- Validation metrics are evaluated on the canonical 141-observation validation split using the corrected COCO evaluator.
- Both configurations achieved an identical public leaderboard score of **0.37**.

---

## Independent Kaggle Verification Results

The pipeline was independently verified via the automated GPU verification kernel on Kaggle (`rephin/solar-filament-verify-notebook`, version 4) running on an **NVIDIA Tesla T4 GPU**:

- **Mask Match Rate**: **99.92%** ($1{,}319 / 1{,}320$ test masks matched at $\text{IoU} \ge 0.90$, passing the required $\ge 99.0\%$ threshold).
- **Instance Count Discrepancy**: **0.08%** (passing the required $\le 1.0\%$ threshold).
- **End-to-End Runtime**: **7.63 minutes** ($457.8\text{ seconds}$) for all 180 test images, averaging **2.586 seconds/image**.

---

## Environment & Requirements

- **Tested Python Versions**: Python 3.10, 3.11, 3.12
- **Hardware**: CUDA-enabled GPU (NVIDIA P100, T4, RTX 3090/4090, V100, A100) or CPU fallback.

### Installation

```bash
git clone https://github.com/PPurab-006/solar-filament-segmentation-2026.git
cd solar-filament-segmentation-2026
pip install -r requirements.txt
```

---

## How to Run & Reproduce Submissions

### 1. Command-Line Reproduction

To generate predictions for test images, run the unified entry point `solarseg.run`:

```bash
# Generate Config A (R1b Rule):
python -m solarseg.run \
    --config configs/final_A.yaml \
    --images /path/to/test_images \
    --out submission_A.csv

# Generate Config B (Plateau Rule):
python -m solarseg.run \
    --config configs/final_B.yaml \
    --images /path/to/test_images \
    --out submission_B.csv
```

### 2. Standalone Verification Notebook

A self-contained reproduction notebook is located at `notebooks/reproduce_submission.ipynb`. It:
1. Dynamically detects test data mounts in Kaggle or local environments.
2. Executes end-to-end inference on the test set.
3. Automatically runs `scripts/check_submission.py` to verify format, dimensions, non-emptiness, and mutual disjointness.
4. Compares generated predictions against `expected/submission_R1b.csv` with printed PASS/WARN checks.
5. Measures and reports GPU inference throughput.

### 3. Submission Integrity Check

Validate any submission CSV using the standalone validation script:

```bash
python scripts/check_submission.py --csv-path submission_A.csv --test-dir /path/to/test_images
```

---

## Rules Compliance & Integrity Checklist

This solution strictly adheres to all competition rules (see `docs/RULES_CHECKLIST.md`):
- **Competition Data Only**: Only official competition images and annotations were used. No external datasets, solar catalogs, or unapproved imagery were utilized.
- **Zero Test-Time Data Leakage**: At inference time, only the single test image's own pixel values are read. No filenames, timestamps, observatory site names, or neighbouring temporal frames are accessed.
- **Hard Rule Compliance**:
  - All masks are encoded as valid COCO RLE strings on fixed $2048 \times 2048$ grids.
  - Zero empty masks exist.
  - All predicted filament instances per image are strictly mutually disjoint (guaranteed by connected component labeling).
  - All weight files are strictly under 100 MB (`r1_convnext_tiny_ema_weights.pt` is 61.0 MB).

---

## Evaluator Reconciliation & Methodology Note

During development, two historical evaluation discrepancies were identified and fully resolved:
1. **Collapsed Multi-Annotator Drawings into One Canvas**: In early exploratory evaluation scripts, ground-truth annotations across multiple annotators were collapsed into a single union mask per image. This conflation artificially inflated Panoptic Quality (PQ) by approximately $+0.04$ ($0.493$ uncorrected vs $0.453$ corrected for rule A). The canonical evaluator properly isolates each annotator's distinct drawing (`drawing_id`), computing Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) per drawing before averaging across images.
2. **COCO Polygon Rasterization vs OpenCV `fillPoly`**: Early ResNet-34 scripts rasterized ground-truth polygon annotations using OpenCV `cv2.fillPoly` instead of standard COCO half-open polygon conventions (`pycocotools.mask.frPyObjects`). Due to integer boundary pixel inclusion, `cv2.fillPoly` expanded mask boundaries by ~5–10%, artificially inflating IoU and true positives ($0.437$ uncorrected vs $0.414$ corrected for the ResNet-34 rule).

**All performance numbers reported in this README and in the technical paper are computed exclusively with the canonical, corrected evaluator**, which was verified against the competition organizers' self-evaluation notebook logic to 8 decimal places ($\Delta = 0.00000000$), guaranteeing absolute mathematical fidelity and true reproducibility.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
