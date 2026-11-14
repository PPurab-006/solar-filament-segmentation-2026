"""
CPU Smoke Test for Fold Kernels before push (Step 3d/3e).
1. Verifies that kernel's map generation and solarseg.infer give identical maps on 3 images (max diff < 2e-3).
2. Verifies that preprocessing spec guard assertion fires when preprocessing_id is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import cv2
import torch
import pytest

from solarseg.infer import sliding_window_inference_3ch as solarseg_infer
from solarseg.config import InferenceConfig
from solarseg.preprocess import (
    assert_preprocessing_spec_guard,
    normalize_solar_intensity_zscore,
    detect_solar_disk_min_circle,
    PREPROCESS_ZSCORE_V1,
    PREPROCESS_LIMBFLAT_V0,
)


def test_fold_kernel_map_generation_equivalence_cpu():
    device = torch.device("cpu")
    torch.set_num_threads(1)

    # Use a tiny lightweight model (per prompt: "3 images, tiny model")
    # This prevents high CPU thread contention and memory bloat
    import torch.nn as nn
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 1, kernel_size=3, padding=1),
            )
        def forward(self, x):
            return self.net(x)

    torch.manual_seed(42)
    model = TinyModel().to(device)
    model.eval()

    train_imgs_dir = Path("MAGFiLO_1.0_Kaggle_2026/train/train_images")
    val_files = ["20110109104734Ch.jpeg", "20110114105034Ch.jpeg", "20110119082634Lh.jpeg"]

    # Import the kernel's embedded map generation from fold_0_train
    import importlib.util
    spec = importlib.util.spec_from_file_location("fold_0_module", "kaggle_kernels/fold_0_train/train_fold_gpu.py")
    fold_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fold_mod)

    kernel_infer_fn = fold_mod.sliding_window_inference_3ch
    cfg = InferenceConfig(tile_size=512, stride=256, blend_mode="hann", tta_mode=1)

    max_diffs = []
    for fn in val_files:
        img_p = train_imgs_dir / fn
        assert img_p.exists(), f"Image not found: {img_p}"

        raw = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        disk = detect_solar_disk_min_circle(raw)
        norm_int, _, _ = normalize_solar_intensity_zscore(raw)

        # 1. Map via solarseg.infer
        map_solarseg, _ = solarseg_infer(
            model=model,
            norm_intensity=norm_int,
            cx=disk.cx, cy=disk.cy, radius=disk.radius,
            config=cfg,
            device=device,
        )

        # 2. Map via fold kernel function
        map_kernel = kernel_infer_fn(
            model=model,
            norm_intensity=norm_int,
            cx=disk.cx, cy=disk.cy, radius=disk.radius,
            device=device,
            tile_size=512, stride=256,
        )

        abs_diff = np.abs(map_solarseg - map_kernel)
        max_d = float(np.max(abs_diff))
        max_diffs.append(max_d)
        print(f"\nImage {fn}: Max absolute diff = {max_d:.6e}")
        assert max_d < 2e-3, f"Map difference exceeds tolerance: {max_d} >= 2e-3"

    assert len(max_diffs) == 3
    print(f"\nAll 3 images passed: mean max diff = {np.mean(max_diffs):.6e} (< 2e-3)")


def test_preprocessing_spec_guard_assertion_fires_on_wrong_id(tmp_path: Path):
    weights_path = tmp_path / "fold_model.pt"
    weights_path.touch()
    meta_path = tmp_path / "fold_model.meta.json"

    # Save with wrong ID: limbflat_v0
    with open(meta_path, "w") as f:
        json.dump({"preprocessing_id": PREPROCESS_LIMBFLAT_V0}, f)

    # Must raise ValueError when pipeline requires zscore_v1
    with pytest.raises(ValueError, match=r"\[PREPROCESSING SPEC GUARD\] Mismatch detected"):
        assert_preprocessing_spec_guard(weights_path, active_preprocessing_id=PREPROCESS_ZSCORE_V1)

    # Save with correct ID: zscore_v1
    with open(meta_path, "w") as f:
        json.dump({"preprocessing_id": PREPROCESS_ZSCORE_V1}, f)

    # Must succeed without error
    assert_preprocessing_spec_guard(weights_path, active_preprocessing_id=PREPROCESS_ZSCORE_V1)
