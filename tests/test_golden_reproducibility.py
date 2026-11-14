"""
Golden test for STAGE 3c Task 5:
Verifies that `python -m solarseg.run` with the final config reproduces submission_R1b.csv
byte-identically on a 3-image subset using only files in the repo.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_golden_reproducibility_on_3_test_images(tmp_path: Path):
    target_csv = Path("outputs/submission_R1b.csv")
    if not target_csv.exists():
        pytest.skip("outputs/submission_R1b.csv not found")

    test_imgs_dir = Path("MAGFiLO_1.0_Kaggle_2026/test/test_images")
    if not test_imgs_dir.exists():
        pytest.skip(f"Test directory {test_imgs_dir} not found")

    # Pick 3 test images from target CSV
    stems = [
        "20110120105534Ch",
        "20110130110334Ch",
        "20110214175414Mh",
    ]

    # Create temporary images directory containing only these 3 images
    sub_imgs_dir = tmp_path / "test_3_images"
    sub_imgs_dir.mkdir(parents=True, exist_ok=True)

    for s in stems:
        matched = list(test_imgs_dir.glob(f"{s}.*"))
        if not matched:
            pytest.skip(f"Image {s} not found in {test_imgs_dir}")
        shutil.copy2(matched[0], sub_imgs_dir / matched[0].name)

    # Run solarseg.run CLI
    out_csv = tmp_path / "golden_subset.csv"
    config_yaml = Path("configs/r1_single.yaml")

    cmd = [
        sys.executable,
        "-m",
        "solarseg.run",
        "--config",
        str(config_yaml),
        "--images",
        str(sub_imgs_dir),
        "--out",
        str(out_csv),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"solarseg.run failed: {res.stderr}\n{res.stdout}"

    # Load both CSVs and compare rows for the 3 stems
    gen_df = pd.read_csv(out_csv)
    ref_df = pd.read_csv(target_csv)

    ref_subset = ref_df[ref_df["filament_id"].apply(lambda fid: any(fid.startswith(s + "_") for s in stems))].reset_index(drop=True)
    gen_subset = gen_df[gen_df["filament_id"].apply(lambda fid: any(fid.startswith(s + "_") for s in stems))].reset_index(drop=True)

    assert len(gen_subset) == len(ref_subset), f"Row count mismatch: {len(gen_subset)} != {len(ref_subset)}"

    # Check byte-identical equality of filament_id and rle
    for idx in range(len(ref_subset)):
        gen_fid = gen_subset.loc[idx, "filament_id"]
        ref_fid = ref_subset.loc[idx, "filament_id"]
        assert gen_fid == ref_fid, f"ID mismatch at row {idx}: {gen_fid} != {ref_fid}"

        gen_rle = str(gen_subset.loc[idx, "segmentation_rle"]).strip()
        ref_rle = str(ref_subset.loc[idx, "segmentation_rle"]).strip()
        assert gen_rle == ref_rle, f"RLE mismatch at row {idx} for {gen_fid}"

    print(f"\n[Golden Test Passed] Byte-identical match verified across {len(gen_subset)} filaments for {stems}!")
