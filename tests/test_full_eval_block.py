#!/usr/bin/env python3
"""
Unit/smoke test for the full evaluation block on local CPU.
Verifies end-to-end:
1. Sliding-window inference with a tiny random-weight model.
2. Postprocessing with the Step 1 optimal rule.
3. Postprocessing with the Step 2 15-feature instance scorer.
4. Panoptic quality scorecard calculation on ground-truth annotator drawings.
Tests on 3 validation images. Must pass before pushing any GPU kernel.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from solarseg.metrics import evaluate_panoptic_quality
from solarseg.postprocess import postprocess_probability_map
from solarseg.preprocess import SolarDiskParams, preprocess_solar_image


class TinyMockModel(nn.Module):
    """Minimal 2-conv model for CPU smoke testing."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def run_tiny_inference(
    model: nn.Module,
    img_2d: np.ndarray,
    disk: SolarDiskParams,
    tile_size: int = 512,
    stride: int = 512,
) -> np.ndarray:
    """Fast CPU sliding window inference for smoke test."""
    model.eval()
    h, w = img_2d.shape
    accum_probs = np.zeros((h, w), dtype=np.float32)
    accum_counts = np.zeros((h, w), dtype=np.float32)

    y_starts = list(range(0, h - tile_size + 1, stride))
    if (h - tile_size) not in y_starts:
        y_starts.append(h - tile_size)
    x_starts = list(range(0, w - tile_size + 1, stride))
    if (w - tile_size) not in x_starts:
        x_starts.append(w - tile_size)

    with torch.no_grad():
        for y1 in y_starts:
            y2 = y1 + tile_size
            for x1 in x_starts:
                x2 = x1 + tile_size

                crop_int = img_2d[y1:y2, x1:x2].astype(np.float32)
                y_g, x_g = np.indices((tile_size, tile_size), dtype=np.float32)
                abs_y = y_g + y1
                abs_x = x_g + x1
                crop_rho = np.sqrt((abs_x - disk.cx) ** 2 + (abs_y - disk.cy) ** 2) / disk.radius
                crop_rho[crop_rho > 1.01] = 0.0

                # Context channel
                crop_ctx = cv2.resize(img_2d, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

                tile_3ch = np.stack([crop_int, crop_rho, crop_ctx], axis=0)
                t_input = torch.from_numpy(tile_3ch).unsqueeze(0).float()
                out = model(t_input)
                prob_tile = torch.sigmoid(out).squeeze().numpy()

                accum_probs[y1:y2, x1:x2] += prob_tile
                accum_counts[y1:y2, x1:x2] += 1.0

    return accum_probs / np.maximum(accum_counts, 1.0)


def test_full_eval_block_smoke():
    repo_root = Path(__file__).resolve().parent.parent
    train_json_path = repo_root / "MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_images_dir = repo_root / "MAGFiLO_1.0_Kaggle_2026/train/train_images"
    split_json_path = repo_root / "data/splits/split.json"
    scorer_path = repo_root / "checkpoints/scorer_model.pkl"

    assert train_json_path.exists(), f"Missing {train_json_path}"
    assert split_json_path.exists(), f"Missing {split_json_path}"
    assert scorer_path.exists(), f"Missing {scorer_path}"

    with open(split_json_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_files = split_data.get("val_files", [])[:3]
    assert len(val_files) == 3, "Expected at least 3 validation files"

    with open(train_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    img_id_to_file = {img["id"]: Path(img["file_name"]).name for img in coco_data["images"]}
    file_to_annos: Dict[str, List[dict]] = {}
    for ann in coco_data["annotations"]:
        fname = img_id_to_file.get(ann["image_id"])
        if fname in val_files:
            file_to_annos.setdefault(fname, []).append(ann)

    # 1. Instantiate tiny random-weight model
    model = TinyMockModel(in_channels=3)

    # Load trained scorer
    with open(scorer_path, "rb") as f:
        scorer_model = pickle.load(f)

    gt_drawings_all: Dict[str, Dict[str, List[np.ndarray]]] = {}
    rule_predictions: Dict[str, List[np.ndarray]] = {}
    scorer_predictions: Dict[str, List[np.ndarray]] = {}

    for fn in val_files:
        img_path = train_images_dir / fn
        assert img_path.exists(), f"Image {img_path} not found"

        raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        assert raw_img is not None, f"Failed to load image from {img_path}"
        two_channel, disk = preprocess_solar_image(raw_img)
        intensity = two_channel[0]
        assert intensity.shape == (2048, 2048)
        assert disk.radius > 500

        # Run sliding-window inference
        prob_map = run_tiny_inference(model, intensity, disk, tile_size=512, stride=512)
        assert prob_map.shape == (2048, 2048)
        assert 0.0 <= prob_map.min() and prob_map.max() <= 1.0

        # Inject a synthetic high-prob filament spot to ensure candidate components exist
        prob_map[500:540, 500:540] = 0.995

        # Test Rule Postprocessing
        rule_instances = postprocess_probability_map(
            prob_map=prob_map,
            disk=disk,
            threshold=0.35,
            min_area=200,
            close_radius=3,
            peak_threshold=0.99,
        )
        assert isinstance(rule_instances, list)
        for inst in rule_instances:
            assert inst.shape == (2048, 2048)
            assert inst.dtype == np.uint8
            assert np.count_nonzero(inst) >= 200
        rule_predictions[fn] = rule_instances

        # Test Scorer Postprocessing
        scorer_instances = postprocess_probability_map(
            prob_map=prob_map,
            disk=disk,
            threshold=0.35,
            min_area=200,
            close_radius=3,
            scorer_model=scorer_model,
            scorer_tau=0.39,
        )
        assert isinstance(scorer_instances, list)
        for inst in scorer_instances:
            assert inst.shape == (2048, 2048)
            assert inst.dtype == np.uint8
            assert np.count_nonzero(inst) >= 200
        scorer_predictions[fn] = scorer_instances

        # Collect GT drawings
        anns = file_to_annos.get(fn, [])
        drawings: Dict[str, List[np.ndarray]] = {}
        for ann in anns:
            d_id = str(ann.get("drawing_id", 0))
            seg = ann.get("segmentation", [])
            mask = np.zeros((2048, 2048), dtype=bool)
            for poly in seg:
                if len(poly) >= 6:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask.view(np.uint8), [pts], 1)
            if np.count_nonzero(mask) > 0:
                drawings.setdefault(d_id, []).append(mask)
        gt_drawings_all[fn] = drawings

    # Evaluate Scorecards
    scorecard_rule = evaluate_panoptic_quality(gt_drawings_all, rule_predictions)
    assert 0.0 <= scorecard_rule.pq <= 1.0
    assert 0.0 <= scorecard_rule.sq <= 1.0
    assert 0.0 <= scorecard_rule.rq <= 1.0
    assert scorecard_rule.total_tp >= 0
    assert scorecard_rule.total_fp >= 0
    assert scorecard_rule.total_fn >= 0

    scorecard_scorer = evaluate_panoptic_quality(gt_drawings_all, scorer_predictions)
    assert 0.0 <= scorecard_scorer.pq <= 1.0
    assert 0.0 <= scorecard_scorer.sq <= 1.0
    assert 0.0 <= scorecard_scorer.rq <= 1.0
    assert scorecard_scorer.total_tp >= 0
    assert scorecard_scorer.total_fp >= 0
    assert scorecard_scorer.total_fn >= 0

    print("\n[SMOKE TEST PASSED]")
    print(f"Rule Scorecard:   PQ={scorecard_rule.pq:.4f}, SQ={scorecard_rule.sq:.4f}, RQ={scorecard_rule.rq:.4f}")
    print(f"Scorer Scorecard: PQ={scorecard_scorer.pq:.4f}, SQ={scorecard_scorer.sq:.4f}, RQ={scorecard_scorer.rq:.4f}")


if __name__ == "__main__":
    test_full_eval_block_smoke()
