from __future__ import annotations

import json
from pathlib import Path
import pytest


def test_folds_integrity_and_rotation_group_disjointness():
    root = Path(__file__).resolve().parent.parent
    folds_path = root / "data/splits/folds.json"
    assert folds_path.exists(), f"Missing folds file: {folds_path}"

    with open(folds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    all_val_files = set()
    cr_sets = []

    for f_idx, fold_info in folds.items():
        val_files = fold_info["val_files"]
        train_files = fold_info["train_files"]
        crs = set(fold_info["carrington_rotations"])

        # Check fold sizes
        assert len(val_files) in {141, 142}, f"Fold {f_idx} has invalid size: {len(val_files)}"
        assert len(train_files) + len(val_files) == 707

        # Overlap between train and val
        assert set(val_files).isdisjoint(set(train_files)), f"Leakage within fold {f_idx}"

        # Overlap across validation folds
        assert all_val_files.isdisjoint(set(val_files)), f"Duplicate validation images in fold {f_idx}"
        all_val_files.update(val_files)

        cr_sets.append((f_idx, crs))

        # Check era span: must reach 2011 and 2022
        min_d, max_d = fold_info["date_range"]
        assert min_d.startswith("2011"), f"Fold {f_idx} does not start in 2011 (starts {min_d})"
        assert max_d.startswith("2022"), f"Fold {f_idx} does not reach 2022 (ends {max_d})"

    # Verify all 707 observations accounted for
    assert len(all_val_files) == 707

    # STRICT CHECK: No rotation group appears in two folds
    for i in range(len(cr_sets)):
        for j in range(i + 1, len(cr_sets)):
            f_i, crs_i = cr_sets[i]
            f_j, crs_j = cr_sets[j]
            overlap = crs_i.intersection(crs_j)
            assert not overlap, (
                f"CR rotation leakage! Fold {f_i} and Fold {f_j} share Carrington rotations: {overlap}"
            )
