"""
Unit tests for Panoptic Quality (PQ) scorer.

WHY THIS EXISTS:
Verifies that the competition scoring rules are exact:
- Known-value hand-built GT/pred pair produces mathematically exact PQ, SQ, and RQ.
- Predicting ground truth itself yields PQ = 1.0.
- Predicting nothing yields PQ = 0.0.
- Diagnostic counts of one-to-many and many-to-one overlaps.
"""

import numpy as np
import pytest

from solarseg.metrics import evaluate_panoptic_quality, score_drawing


def test_pq_known_value_case():
    """
    Test a hand-constructed case with known analytical PQ, SQ, RQ.

    Setup:
      Drawing with 2 GT instances:
        G1: 100 pixels
        G2: 100 pixels
      Predictions with 2 instances:
        P1: overlaps G1 with 80 pixels, union = 100 pixels -> IoU = 0.80 (>0.5 -> TP)
        P2: completely disjoint from G2 -> FP=1
      G2 has no match -> FN=1

    Expected:
      TP = 1, FP = 1, FN = 1
      IoU sum = 0.80
      Denominator = 1 + 0.5(1) + 0.5(1) = 2.0
      PQ = 0.80 / 2.0 = 0.40
      SQ = 0.80 / 1 = 0.80
      RQ = 1 / 2.0 = 0.50
    """
    shape = (200, 200)

    # G1: [10:20, 10:20] -> 10 x 10 = 100 px
    g1 = np.zeros(shape, dtype=np.uint8)
    g1[10:20, 10:20] = 1

    # G2: [50:60, 50:60] -> 10 x 10 = 100 px
    g2 = np.zeros(shape, dtype=np.uint8)
    g2[50:60, 50:60] = 1

    # P1: [10:20, 10:18] -> 10 x 8 = 80 px (subset of G1). Intersection = 80, Union = 100 -> IoU = 0.80
    p1 = np.zeros(shape, dtype=np.uint8)
    p1[10:20, 10:18] = 1

    # P2: [120:130, 120:130] -> 10 x 10 = 100 px (unrelated region) -> FP
    p2 = np.zeros(shape, dtype=np.uint8)
    p2[120:130, 120:130] = 1

    res = score_drawing(
        drawing_id="test_drawing_01",
        gt_masks=[g1, g2],
        pred_masks=[p1, p2],
        iou_threshold=0.5,
    )

    assert res.tp == 1
    assert res.fp == 1
    assert res.fn == 1
    assert pytest.approx(res.iou_sum, abs=1e-5) == 0.80

    # Pool across dictionary
    gt_dict = {"img1": {"d1": [g1, g2]}}
    pred_dict = {"img1": [p1, p2]}

    pq_res = evaluate_panoptic_quality(gt_dict, pred_dict, iou_threshold=0.5)

    assert pq_res.total_tp == 1
    assert pq_res.total_fp == 1
    assert pq_res.total_fn == 1
    assert pytest.approx(pq_res.pq, abs=1e-5) == 0.40
    assert pytest.approx(pq_res.sq, abs=1e-5) == 0.80
    assert pytest.approx(pq_res.rq, abs=1e-5) == 0.50


def test_pq_sanity_checks():
    """Verify edge cases: predicting GT itself gives 1.0; predicting nothing gives 0.0."""
    shape = (200, 200)

    g1 = np.zeros(shape, dtype=np.uint8)
    g1[10:30, 10:30] = 1

    g2 = np.zeros(shape, dtype=np.uint8)
    g2[60:80, 60:80] = 1

    gt_dict = {"img1": {"d1": [g1, g2]}}

    # 1. Predicting GT itself -> PQ = 1.0
    perfect_pred = {"img1": [g1.copy(), g2.copy()]}
    res_perfect = evaluate_panoptic_quality(gt_dict, perfect_pred)
    assert pytest.approx(res_perfect.pq, abs=1e-5) == 1.0
    assert pytest.approx(res_perfect.sq, abs=1e-5) == 1.0
    assert pytest.approx(res_perfect.rq, abs=1e-5) == 1.0
    assert pytest.approx(res_perfect.mean_dice, abs=1e-5) == 1.0

    # 2. Predicting nothing -> PQ = 0.0
    empty_pred = {"img1": []}
    res_empty = evaluate_panoptic_quality(gt_dict, empty_pred)
    assert res_empty.pq == 0.0
    assert res_empty.sq == 0.0
    assert res_empty.rq == 0.0
    assert res_empty.total_fn == 2
    assert res_empty.total_tp == 0


def test_overlap_diagnostics():
    """Verify that one-to-many and many-to-one overlaps are detected accurately."""
    shape = (200, 200)

    # One large GT instance
    gt_large = np.zeros(shape, dtype=np.uint8)
    gt_large[10:90, 10:30] = 1

    # Two predicted fragments overlapping the same GT instance
    p_frag1 = np.zeros(shape, dtype=np.uint8)
    p_frag1[10:40, 10:30] = 1
    p_frag2 = np.zeros(shape, dtype=np.uint8)
    p_frag2[50:90, 10:30] = 1

    res = score_drawing(
        drawing_id="frag_test",
        gt_masks=[gt_large],
        pred_masks=[p_frag1, p_frag2],
        iou_threshold=0.5,
    )

    # 1 GT overlapping 2 predictions -> one_to_many_count == 1
    assert res.one_to_many_count == 1
    assert res.many_to_one_count == 0


def test_scorecard_comprehensive():
    """Verify that scorecard computes PQ, Dice distributions, IoU distributions, and overlap rates."""
    from solarseg.metrics import scorecard

    shape = (100, 100)
    g1 = np.zeros(shape, dtype=np.uint8)
    g1[10:20, 10:20] = 1
    p1 = np.zeros(shape, dtype=np.uint8)
    p1[10:20, 10:18] = 1  # IoU = 80/100 = 0.80

    gt = {"img_0": {"d0": [g1]}}
    preds = {"img_0": [p1]}

    sc = scorecard(preds=preds, gt=gt, elapsed_seconds=0.5)

    assert sc.num_images == 1
    assert sc.num_drawings == 1
    assert sc.tp == 1
    assert sc.fp == 0
    assert sc.fn == 0
    assert sc.pq == pytest.approx(0.80)
    assert sc.sq == pytest.approx(0.80)
    assert sc.rq == pytest.approx(1.0)
    assert sc.dice_mean > 0.85
    assert sc.iou_mean == pytest.approx(0.80)
    assert sc.seconds_per_image == pytest.approx(0.5)
    assert sc.one_to_many == 0
    assert sc.many_to_one == 0
    assert "PQ: 0.8000" in sc.summary_table()
