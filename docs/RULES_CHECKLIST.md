# Competition Rules and Final Package Verification Checklist

This checklist tracks adherence to competition host rulings, modeling restrictions, submission packaging, and reproducibility standards for the Solar Filament Segmentation Challenge 2026.

## Automated Verification Checks (`tests/test_final_package.py`)

- [x] **[auto] Inference uses only test-image pixels**: No filename parsing, site code extraction, date/time extraction, or neighboring frames in the inference path.
- [x] **[auto] No access to public MAGFiLO or external ground truth**: Training and validation use competition data only; no network requests or external labels.
- [x] **[auto] Submission format strict compliance**: Instances are mutually disjoint, non-empty, unique IDs matching `<image_stem>_<k>`, 2048x2048 COCO RLE column-major strings only.
- [x] **[auto] Package requirements and weight limits**: Every inference weight file is strictly under 100 MB (current: 47 MB), and all packages in `requirements.txt` are pinned with exact versions (`==`).
- [x] **[auto] spine/bbox/area/class fields may be used at TRAINING time only; the inference path never reads the annotation JSON**: Dataset and training code may read spine, bbox, area, and class fields, but inference/predict/run paths never import the annotation parser or access annotation files.

## Manual Verification Checks

- [ ] **[manual] Reproducibility**: The final Kaggle notebook loads the best model via one unified pipeline and reproduces the submission with zero extra external dependencies on a clean environment within stated runtime.
- [ ] **[manual] Report model documentation**: Report explicitly states: ImageNet-pretrained encoder used, training data = competition data only, validation split strictly by observation date.
- [ ] **[manual] Metric distributions in report**: Report includes PQ, SQ, RQ, Dice mean + distribution (p10, median, p90), IoU distribution (p10, median, p90), one-to-many and many-to-one error counts, and inference seconds per image.
- [ ] **[manual] Public release & submission form**: Repository made public from Nov 15 until winners are announced; Google form submitted before the deadline (Nov 15, 11:59 pm CDT).

## Pending Host Clarifications

- [ ] **[open] Host ruling pending**: Whether an ensemble counts as "the best model" under final submission rules (state in report).
- [ ] **[open] Host ruling pending**: ImageNet-pretrained encoders permitted under final rules (state in report).
