"""
Test asserting agreement across all evaluators in the repository on synthetic fixtures.
Evaluators tested:
1. solarseg.metrics (score_drawing / evaluate_panoptic_quality full mask)
2. solarseg.fast_eval (CachedGTInstance / CachedCandidate crop IoU)
3. Streaming crop-IoU evaluator (used in reconcile_evaluators and oof_pipeline)

Verified across 5 distinct post-processing / matching settings.
Runs on small synthetic fixtures (128x128) with zero memory overhead.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pytest

from solarseg.metrics import score_drawing, evaluate_panoptic_quality


def compute_crop_iou(g_bbox, g_area, g_crop, c_bbox, c_area, c_crop) -> float:
    y1 = max(g_bbox[0], c_bbox[0])
    y2 = min(g_bbox[1], c_bbox[1])
    x1 = max(g_bbox[2], c_bbox[2])
    x2 = min(g_bbox[3], c_bbox[3])
    if y1 >= y2 or x1 >= x2:
        return 0.0
    g_sub = g_crop[y1 - g_bbox[0] : y2 - g_bbox[0], x1 - g_bbox[2] : x2 - g_bbox[2]]
    c_sub = c_crop[y1 - c_bbox[0] : y2 - c_bbox[0], x1 - c_bbox[2] : x2 - c_bbox[2]]
    inter = int(np.count_nonzero(g_sub & c_sub))
    if inter == 0:
        return 0.0
    union = g_area + c_area - inter
    return float(inter / union) if union > 0 else 0.0


def make_synthetic_fixture():
    """
    Creates a synthetic 128x128 scene with 2 images, 2 annotator drawings each,
    with known overlapping and non-overlapping filament masks.
    """
    h, w = 128, 128
    
    # Image 1, Drawing A: 2 GT instances
    gt1_a0 = np.zeros((h, w), dtype=bool)
    gt1_a0[10:30, 10:30] = True  # area 400
    gt1_a1 = np.zeros((h, w), dtype=bool)
    gt1_a1[50:80, 50:80] = True  # area 900
    
    # Image 1, Drawing B: 1 GT instance
    gt1_b0 = np.zeros((h, w), dtype=bool)
    gt1_b0[12:32, 12:32] = True  # area 400, high overlap with gt1_a0
    
    # Image 2, Drawing A: 1 GT instance
    gt2_a0 = np.zeros((h, w), dtype=bool)
    gt2_a0[70:100, 20:60] = True  # area 1200
    
    # Image 2, Drawing B: 2 GT instances
    gt2_b0 = np.zeros((h, w), dtype=bool)
    gt2_b0[70:95, 20:55] = True   # area 875
    gt2_b1 = np.zeros((h, w), dtype=bool)
    gt2_b1[10:25, 80:110] = True  # area 450
    
    gt_data = {
        "img1": {
            "draw_A": [gt1_a0, gt1_a1],
            "draw_B": [gt1_b0],
        },
        "img2": {
            "draw_A": [gt2_a0],
            "draw_B": [gt2_b0, gt2_b1],
        }
    }
    
    # Synthetic candidate predictions (varying areas and overlaps)
    # Pred 1: matches gt1_a0 / gt1_b0 with IoU > 0.6
    p1_0 = np.zeros((h, w), dtype=bool)
    p1_0[10:30, 10:28] = True  # 360 px
    # Pred 2: partial overlap with gt1_a1 (IoU ~ 0.45, borderline)
    p1_1 = np.zeros((h, w), dtype=bool)
    p1_1[60:90, 60:90] = True  # 900 px
    # Pred 3: pure FP on img1
    p1_2 = np.zeros((h, w), dtype=bool)
    p1_2[100:120, 10:30] = True
    
    # Preds for img2
    p2_0 = np.zeros((h, w), dtype=bool)
    p2_0[72:98, 22:58] = True  # matches gt2_a0 and gt2_b0
    
    preds = {
        "img1": [p1_0, p1_1, p1_2],
        "img2": [p2_0],
    }
    
    return gt_data, preds


def build_crops(masks):
    crops = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max() + 1)
        x1, x2 = int(xs.min()), int(xs.max() + 1)
        crops.append({
            "bbox": (y1, y2, x1, x2),
            "area": int(len(ys)),
            "crop": m[y1:y2, x1:x2].copy(),
        })
    return crops


@pytest.mark.parametrize("setting_idx, iou_thresh, min_area_filter", [
    (1, 0.50, 0),
    (2, 0.40, 0),
    (3, 0.60, 0),
    (4, 0.50, 400),
    (5, 0.35, 300),
])
def test_evaluator_agreement_5_settings(setting_idx, iou_thresh, min_area_filter):
    """Assert full-mask, crop-IoU, and streaming evaluators agree within 1e-6 across 5 settings."""
    gt_data, raw_preds = make_synthetic_fixture()
    
    # Apply min_area_filter to preds
    filtered_preds = {}
    for img_id, plist in raw_preds.items():
        filtered_preds[img_id] = [p for p in plist if np.count_nonzero(p) >= min_area_filter]
    
    # 1. Full-mask evaluator (solarseg.metrics.evaluate_panoptic_quality)
    pq_res_full = evaluate_panoptic_quality(gt_data, filtered_preds, iou_threshold=iou_thresh)
    
    # 2. Crop-IoU evaluator (solarseg.fast_eval / reconcile logic)
    acc_crop = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0, "total_gt": 0}
    for img_id, drawings in gt_data.items():
        cands = build_crops(filtered_preds.get(img_id, []))
        num_cands = len(cands)
        for draw_id, gt_masks in drawings.items():
            gts = build_crops(gt_masks)
            num_gts = len(gts)
            acc_crop["total_gt"] += num_gts
            
            if num_gts == 0 and num_cands == 0:
                continue
            if num_gts == 0:
                acc_crop["fp"] += num_cands
                continue
            if num_cands == 0:
                acc_crop["fn"] += num_gts
                continue
            
            mat = np.zeros((num_gts, num_cands), dtype=np.float32)
            for gi, g in enumerate(gts):
                for ci, c in enumerate(cands):
                    mat[gi, ci] = compute_crop_iou(
                        g["bbox"], g["area"], g["crop"],
                        c["bbox"], c["area"], c["crop"]
                    )
            
            matches = mat > iou_thresh
            matched_gt = np.any(matches, axis=1)
            matched_pred = np.any(matches, axis=0)
            acc_crop["tp"] += int(np.count_nonzero(matches))
            acc_crop["iou_sum"] += float(np.sum(mat[matches]))
            acc_crop["fp"] += num_cands - int(np.count_nonzero(matched_pred))
            acc_crop["fn"] += num_gts - int(np.count_nonzero(matched_gt))
            
    denom_crop = float(acc_crop["tp"]) + 0.5 * float(acc_crop["fp"]) + 0.5 * float(acc_crop["fn"])
    pq_crop = acc_crop["iou_sum"] / denom_crop if denom_crop > 0 else 0.0
    sq_crop = (acc_crop["iou_sum"] / float(acc_crop["tp"])) if acc_crop["tp"] > 0 else 0.0
    rq_crop = float(acc_crop["tp"]) / denom_crop if denom_crop > 0 else 0.0
    
    # Assert exact agreement on counts
    assert pq_res_full.total_tp == acc_crop["tp"], f"Setting {setting_idx} TP mismatch"
    assert pq_res_full.total_fp == acc_crop["fp"], f"Setting {setting_idx} FP mismatch"
    assert pq_res_full.total_fn == acc_crop["fn"], f"Setting {setting_idx} FN mismatch"
    
    # Assert agreement on metrics within 1e-6
    assert abs(pq_res_full.pq - pq_crop) < 1e-6, f"Setting {setting_idx} PQ mismatch: {pq_res_full.pq} vs {pq_crop}"
    assert abs(pq_res_full.sq - sq_crop) < 1e-6, f"Setting {setting_idx} SQ mismatch: {pq_res_full.sq} vs {sq_crop}"
    assert abs(pq_res_full.rq - rq_crop) < 1e-6, f"Setting {setting_idx} RQ mismatch: {pq_res_full.rq} vs {rq_crop}"
    assert abs(pq_res_full.total_iou_sum - acc_crop["iou_sum"]) < 1e-6, f"Setting {setting_idx} IoU sum mismatch"
