"""
Panoptic Quality (PQ) metric evaluator for solar filament segmentation.

WHY THIS EXISTS:
Filament segmentation in Kaggle 2026 is evaluated using Panoptic Quality (PQ):
    PQ = SQ * RQ = sum(IoU of TPs) / (|TP| + 0.5|FP| + 0.5|FN|)
Scoring is performed per annotator drawing separately against the image predictions,
then TP, FP, FN, and IoU sums are pooled over all drawings.
This module strictly implements the competition rules, tracks fragmentation diagnostics
(one-to-many and many-to-one overlaps), and computes semantic Dice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class DrawingScoreResult:
    """
    Score components for a single annotator drawing evaluated against predictions.

    WHY THIS EXISTS:
    Encapsulates raw TP/FP/FN counts, IoU sums, and overlap diagnostics per drawing,
    allowing clean inspection and global pooling across multiple annotators.
    """
    drawing_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    matched_ious: List[float] = field(default_factory=list)
    dice: float = 0.0
    one_to_many_count: int = 0  # Organizer definition: GT instances overlapping >1 prediction (IoU > 0)
    many_to_one_count: int = 0  # Organizer definition: Predictions overlapping >1 GT instance (IoU > 0)
    one_to_many_iou10_count: int = 0  # Legacy definition: GT instances overlapping >1 prediction (IoU > 0.10)
    many_to_one_iou10_count: int = 0  # Legacy definition: Predictions overlapping >1 GT instance (IoU > 0.10)


@dataclass
class PQScoreResult:
    """
    Global panoptic quality evaluation summary.

    WHY THIS EXISTS:
    Summarizes PQ, Segmentation Quality (SQ), Recognition Quality (RQ), mean Dice,
    and filament count statistics across all evaluated validation drawings.
    """
    pq: float
    sq: float
    rq: float
    total_tp: int
    total_fp: int
    total_fn: int
    total_iou_sum: float
    mean_dice: float
    avg_gt_filaments_per_drawing: float
    avg_pred_filaments_per_image: float
    total_one_to_many: int
    total_many_to_one: int
    total_one_to_many_iou10: int = 0
    total_many_to_one_iou10: int = 0
    drawing_results: List[DrawingScoreResult] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, float]:
        return {
            "PQ": self.pq,
            "SQ": self.sq,
            "RQ": self.rq,
            "mean_dice": self.mean_dice,
            "total_tp": float(self.total_tp),
            "total_fp": float(self.total_fp),
            "total_fn": float(self.total_fn),
            "avg_gt_filaments": self.avg_gt_filaments_per_drawing,
            "avg_pred_filaments": self.avg_pred_filaments_per_image,
            "one_to_many_overlaps": float(self.total_one_to_many),
            "many_to_one_overlaps": float(self.total_many_to_one),
            "one_to_many_iou10": float(self.total_one_to_many_iou10),
            "many_to_one_iou10": float(self.total_many_to_one_iou10),
        }


def compute_instance_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    bbox_a: Optional[Tuple[int, int, int, int]] = None,
    bbox_b: Optional[Tuple[int, int, int, int]] = None,
    area_a: Optional[int] = None,
    area_b: Optional[int] = None,
) -> float:
    """
    Compute Intersection-over-Union between two binary masks.

    WHY THIS EXISTS:
    Fundamental building block for TP matching (threshold > 0.5) and overlap tracking.
    Uses bounding box pre-checks and region cropping to accelerate evaluation by >50x.
    """
    if bbox_a is not None and bbox_b is not None:
        y1_a, y2_a, x1_a, x2_a = bbox_a
        y1_b, y2_b, x1_b, x2_b = bbox_b
        # Fast bounding box rejection: if bounding boxes do not overlap, IoU is strictly 0.0
        if y1_a >= y2_b or y1_b >= y2_a or x1_a >= x2_b or x1_b >= x2_a:
            return 0.0
        sub_y1, sub_y2 = max(y1_a, y1_b), min(y2_a, y2_b)
        sub_x1, sub_x2 = max(x1_a, x1_b), min(x2_a, x2_b)
        intersection = int(
            np.count_nonzero(
                mask_a[sub_y1:sub_y2, sub_x1:sub_x2] & mask_b[sub_y1:sub_y2, sub_x1:sub_x2]
            )
        )
    else:
        intersection = int(np.count_nonzero(mask_a & mask_b))

    if intersection == 0:
        return 0.0
    a_area = area_a if area_a is not None else int(np.count_nonzero(mask_a))
    b_area = area_b if area_b is not None else int(np.count_nonzero(mask_b))
    union = a_area + b_area - intersection
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def _get_mask_info(mask: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], int]:
    """Extract bounding box and area for fast IoU calculation."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None, 0
    return (int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)), len(ys)


def compute_relations_organizer(iou_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Organizer notebook definition (IoU > 0, any pixel overlap):
    - One-to-Many: number of GT filaments matching > 1 prediction (fragmentation)
    - Many-to-One: number of predictions matching > 1 GT filament (over-merge)
    """
    if iou_matrix.size == 0:
        return 0, 0
    hit_matrix = (iou_matrix > 0)
    one_to_many = int(np.sum(np.sum(hit_matrix, axis=1) > 1))
    many_to_one = int(np.sum(np.sum(hit_matrix, axis=0) > 1))
    return one_to_many, many_to_one


def compute_relations_iou10(iou_matrix: np.ndarray) -> Tuple[int, int]:
    """
    Alternative / legacy definition with minimum overlap threshold IoU > 0.10:
    - One-to-Many: number of GT filaments with IoU > 0.10 against > 1 prediction
    - Many-to-One: number of predictions with IoU > 0.10 against > 1 GT filament
    """
    if iou_matrix.size == 0:
        return 0, 0
    hit_matrix = (iou_matrix > 0.10)
    one_to_many = int(np.sum(np.sum(hit_matrix, axis=1) > 1))
    many_to_one = int(np.sum(np.sum(hit_matrix, axis=0) > 1))
    return one_to_many, many_to_one


def score_drawing(
    drawing_id: str,
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    iou_threshold: float = 0.5,
    compute_dice: bool = True,
) -> DrawingScoreResult:
    """
    Score one annotator drawing against predicted masks for that image.

    WHY THIS EXISTS:
    Implements the exact per-annotator drawing evaluation logic:
    - Pairs with IoU > 0.5 are True Positives (each GT/pred matches at most once).
    - Unmatched predictions are False Positives.
    - Unmatched GT instances are False Negatives.
    - Diagnoses fragmentation (one GT overlapping multiple predictions) and
      merging (one prediction overlapping multiple GTs).

    Args:
        drawing_id: Identifier for the annotator drawing.
        gt_masks: Sequence of binary 2D arrays for ground truth instances.
        pred_masks: Sequence of binary 2D arrays for predicted instances (disjoint).
        iou_threshold: TP matching threshold (fixed at 0.5 by competition rules).

    Returns:
        DrawingScoreResult with TP/FP/FN and overlap counts.
    """
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)

    # Edge case: no GT and no predictions
    if num_gt == 0 and num_pred == 0:
        return DrawingScoreResult(
            drawing_id=drawing_id,
            tp=0,
            fp=0,
            fn=0,
            iou_sum=0.0,
            matched_ious=[],
            dice=1.0,
        )

    # Edge case: no GT, all predictions are FP
    if num_gt == 0:
        return DrawingScoreResult(
            drawing_id=drawing_id,
            tp=0,
            fp=num_pred,
            fn=0,
            iou_sum=0.0,
            matched_ious=[],
            dice=0.0,
        )

    # Edge case: no predictions, all GT are FN
    if num_pred == 0:
        return DrawingScoreResult(
            drawing_id=drawing_id,
            tp=0,
            fp=0,
            fn=num_gt,
            iou_sum=0.0,
            matched_ious=[],
            dice=0.0,
        )

    # Pre-extract bounding boxes and areas for fast intersection calculation
    gt_info = [_get_mask_info(gm) for gm in gt_masks]
    pred_info = [_get_mask_info(pm) for pm in pred_masks]

    # Compute pairwise IoU matrix
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    for i, (g_mask, (g_bbox, g_area)) in enumerate(zip(gt_masks, gt_info)):
        if g_area == 0:
            continue
        for j, (p_mask, (p_bbox, p_area)) in enumerate(zip(pred_masks, pred_info)):
            if p_area == 0:
                continue
            iou_matrix[i, j] = compute_instance_iou(
                g_mask, p_mask, bbox_a=g_bbox, bbox_b=p_bbox, area_a=g_area, area_b=p_area
            )

    # Compute relation diagnostics (organizer notebook and legacy iou10)
    one_to_many, many_to_one = compute_relations_organizer(iou_matrix)
    one_to_many_iou10, many_to_one_iou10 = compute_relations_iou10(iou_matrix)

    # TP matching: find pairs with IoU > threshold
    # Note: because pred_masks are disjoint and threshold > 0.5,
    # at most one pred can have IoU > 0.5 with any GT, and vice versa.
    matched_gt = set()
    matched_pred = set()
    matched_ious = []
    iou_sum = 0.0

    for i in range(num_gt):
        for j in range(num_pred):
            val = float(iou_matrix[i, j])
            if val > iou_threshold:
                matched_gt.add(i)
                matched_pred.add(j)
                matched_ious.append(val)
                iou_sum += val

    tp = len(matched_ious)
    fp = num_pred - len(matched_pred)
    fn = num_gt - len(matched_gt)

    dice = 1.0 if (num_gt == 0 and num_pred == 0) else 0.0
    if compute_dice and (num_gt > 0 or num_pred > 0):
        gt_union = np.zeros_like(gt_masks[0], dtype=bool)
        for gm in gt_masks:
            gt_union |= (gm > 0)
        pred_union = np.zeros_like(pred_masks[0], dtype=bool)
        for pm in pred_masks:
            pred_union |= (pm > 0)

        intersection = np.count_nonzero(gt_union & pred_union)
        gt_area = np.count_nonzero(gt_union)
        pred_area = np.count_nonzero(pred_union)
        total_area = gt_area + pred_area
        dice = (2.0 * intersection / total_area) if total_area > 0 else 1.0

    return DrawingScoreResult(
        drawing_id=drawing_id,
        tp=tp,
        fp=fp,
        fn=fn,
        iou_sum=iou_sum,
        matched_ious=matched_ious,
        dice=dice,
        one_to_many_count=one_to_many,
        many_to_one_count=many_to_one,
        one_to_many_iou10_count=one_to_many_iou10,
        many_to_one_iou10_count=many_to_one_iou10,
    )


def evaluate_panoptic_quality(
    gt_drawings: Dict[str, Dict[str, List[np.ndarray]]],
    predictions: Dict[str, List[np.ndarray]],
    iou_threshold: float = 0.5,
) -> PQScoreResult:
    """
    Evaluate PQ across all images and annotator drawings.

    WHY THIS EXISTS:
    Pools TP, FP, FN, and IoU sums across every annotator drawing in the evaluation set.
    Each drawing represents an annotator's distinct ground truth drawing for that image.

    Args:
        gt_drawings: Mapping from file_name -> {drawing_id -> list_of_gt_instance_masks}.
        predictions: Mapping from file_name -> list_of_predicted_instance_masks.
        iou_threshold: TP threshold (0.5).

    Returns:
        PQScoreResult with pooled PQ, SQ, RQ, Dice, and diagnostic statistics.
    """
    drawing_results: List[DrawingScoreResult] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    total_one_to_many = 0
    total_many_to_one = 0
    total_one_to_many_iou10 = 0
    total_many_to_one_iou10 = 0
    dice_list: List[float] = []
    total_gt_filaments = 0

    # Ensure all file_names in gt_drawings are evaluated
    for file_name, drawings in gt_drawings.items():
        base_name = file_name.replace(".jpeg", "").replace(".jpg", "")
        # Retrieve predictions matching this image
        pred_list = predictions.get(file_name) or predictions.get(base_name) or []

        for drawing_id, gt_masks in drawings.items():
            total_gt_filaments += len(gt_masks)
            res = score_drawing(
                drawing_id=drawing_id,
                gt_masks=gt_masks,
                pred_masks=pred_list,
                iou_threshold=iou_threshold,
            )
            drawing_results.append(res)
            total_tp += res.tp
            total_fp += res.fp
            total_fn += res.fn
            total_iou_sum += res.iou_sum
            total_one_to_many += res.one_to_many_count
            total_many_to_one += res.many_to_one_count
            total_one_to_many_iou10 += res.one_to_many_iou10_count
            total_many_to_one_iou10 += res.many_to_one_iou10_count
            dice_list.append(res.dice)

    num_drawings = max(len(drawing_results), 1)
    num_images = max(len(gt_drawings), 1)
    total_pred_filaments = sum(len(p) for p in predictions.values())

    denom = float(total_tp) + 0.5 * float(total_fp) + 0.5 * float(total_fn)
    if denom == 0:
        pq = 1.0 if total_tp == 0 and total_gt_filaments == 0 else 0.0
        rq = 1.0 if total_tp == 0 and total_gt_filaments == 0 else 0.0
        sq = 1.0 if total_tp == 0 and total_gt_filaments == 0 else 0.0
    else:
        pq = total_iou_sum / denom
        rq = float(total_tp) / denom
        sq = (total_iou_sum / float(total_tp)) if total_tp > 0 else 0.0

    mean_dice = float(np.mean(dice_list)) if dice_list else 0.0

    return PQScoreResult(
        pq=pq,
        sq=sq,
        rq=rq,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        total_iou_sum=total_iou_sum,
        mean_dice=mean_dice,
        avg_gt_filaments_per_drawing=total_gt_filaments / float(num_drawings),
        avg_pred_filaments_per_image=total_pred_filaments / float(num_images),
        total_one_to_many=total_one_to_many,
        total_many_to_one=total_many_to_one,
        total_one_to_many_iou10=total_one_to_many_iou10,
        total_many_to_one_iou10=total_many_to_one_iou10,
        drawing_results=drawing_results,
    )


@dataclass
class ScorecardResult:
    """
    Standardized validation scorecard result (Part D1).

    WHY THIS EXISTS:
    Mandated metric tracking for every experiment, baseline, and gate decision:
    - PQ, SQ, RQ, TP, FP, FN (pooled over drawings, organizer protocol)
    - Binary mask Dice per image (mean, median, p10, p90)
    - TP matched IoU (mean, median, p10, p90)
    - Overlap diagnostics: one-to-many (fragmentation) and many-to-one (merging) counts & per-100-GT rates
    - Seconds per image throughput
    """
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    iou_sum: float
    dice_mean: float
    dice_median: float
    dice_p10: float
    dice_p90: float
    iou_mean: float
    iou_median: float
    iou_p10: float
    iou_p90: float
    one_to_many: int
    many_to_one: int
    total_gt: int
    one_to_many_rate: float  # per 100 GT
    many_to_one_rate: float  # per 100 GT
    seconds_per_image: float
    num_images: int
    num_drawings: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "PQ": self.pq,
            "SQ": self.sq,
            "RQ": self.rq,
            "TP": float(self.tp),
            "FP": float(self.fp),
            "FN": float(self.fn),
            "iou_sum": self.iou_sum,
            "dice_mean": self.dice_mean,
            "dice_median": self.dice_median,
            "dice_p10": self.dice_p10,
            "dice_p90": self.dice_p90,
            "iou_mean": self.iou_mean,
            "iou_median": self.iou_median,
            "iou_p10": self.iou_p10,
            "iou_p90": self.iou_p90,
            "one_to_many_count": float(self.one_to_many),
            "many_to_one_count": float(self.many_to_one),
            "total_gt": float(self.total_gt),
            "one_to_many_rate": self.one_to_many_rate,
            "many_to_one_rate": self.many_to_one_rate,
            "seconds_per_image": self.seconds_per_image,
        }

    def summary_table(self) -> str:
        return (
            f"PQ: {self.pq:.4f} | SQ: {self.sq:.4f} | RQ: {self.rq:.4f}\n"
            f"Counts: TP={self.tp} | FP={self.fp} | FN={self.fn} | GT={self.total_gt}\n"
            f"Dice (image): mean={self.dice_mean:.4f}, median={self.dice_median:.4f}, p10={self.dice_p10:.4f}, p90={self.dice_p90:.4f}\n"
            f"IoU (matched): mean={self.iou_mean:.4f}, median={self.iou_median:.4f}, p10={self.iou_p10:.4f}, p90={self.iou_p90:.4f}\n"
            f"Overlaps (per 100 GT): One-to-Many={self.one_to_many_rate:.2f} ({self.one_to_many} count), Many-to-One={self.many_to_one_rate:.2f} ({self.many_to_one} count)\n"
            f"Speed: {self.seconds_per_image:.3f} s/image across {self.num_images} images ({self.num_drawings} drawings)"
        )


def scorecard(
    preds: Dict[str, Sequence[np.ndarray]],
    gt: Dict[str, Any],
    elapsed_seconds: float = 0.0,
    iou_threshold: float = 0.5,
) -> ScorecardResult:
    """
    Generate comprehensive standardized scorecard (Part D1).

    Args:
        preds: Mapping from file_name (or stem) -> list of disjoint predicted instance masks.
        gt: Mapping from file_name (or stem) -> {drawing_id -> list of GT instance masks} or Observation.
        elapsed_seconds: Total inference/evaluation wall time in seconds.
        iou_threshold: IoU matching threshold (0.5).

    Returns:
        ScorecardResult with all mandated scorecard fields.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_iou_sum = 0.0
    total_one_to_many = 0
    total_many_to_one = 0
    total_one_to_many_iou10 = 0
    total_many_to_one_iou10 = 0
    total_gt = 0
    total_drawings = 0

    all_matched_ious: List[float] = []
    image_dice_list: List[float] = []

    for key, gt_val in gt.items():
        base_name = key.replace(".jpeg", "").replace(".jpg", "")
        pred_list = preds.get(key) or preds.get(base_name) or []

        # Extract drawings dict: drawing_id -> list of masks
        if hasattr(gt_val, "drawings"):
            from solarseg.data import build_drawing_masks
            drawings_dict = {
                d_id: build_drawing_masks(drawing)
                for d_id, drawing in gt_val.drawings.items()
            }
        elif isinstance(gt_val, dict):
            drawings_dict = gt_val
        elif isinstance(gt_val, list):
            drawings_dict = {f"{base_name}_d0": gt_val}
        else:
            raise TypeError(f"Unsupported GT format for key '{key}': {type(gt_val)}")

        # Binary Dice per image following torchmetrics semantics
        if pred_list:
            pred_union = np.zeros_like(pred_list[0], dtype=bool)
            for pm in pred_list:
                pred_union |= (pm > 0)
        else:
            pred_union = None

        drawing_dices = []
        for d_id, gt_masks in drawings_dict.items():
            total_drawings += 1
            total_gt += len(gt_masks)

            res = score_drawing(
                drawing_id=d_id,
                gt_masks=gt_masks,
                pred_masks=pred_list,
                iou_threshold=iou_threshold,
                compute_dice=False,
            )

            total_tp += res.tp
            total_fp += res.fp
            total_fn += res.fn
            total_iou_sum += res.iou_sum
            total_one_to_many += res.one_to_many_count
            total_many_to_one += res.many_to_one_count
            all_matched_ious.extend(res.matched_ious)

            # Compute drawing binary dice
            if gt_masks:
                gt_union = np.zeros_like(gt_masks[0], dtype=bool)
                for gm in gt_masks:
                    gt_union |= (gm > 0)
            else:
                gt_union = None

            if gt_union is None and pred_union is None:
                d_dice = 1.0
            elif gt_union is None or pred_union is None:
                d_dice = 0.0
            else:
                inter = np.count_nonzero(gt_union & pred_union)
                union_area = np.count_nonzero(gt_union) + np.count_nonzero(pred_union)
                d_dice = (2.0 * inter / union_area) if union_area > 0 else 1.0
            drawing_dices.append(d_dice)

        if drawing_dices:
            image_dice_list.append(float(np.mean(drawing_dices)))

    num_images = max(len(gt), 1)
    denom = float(total_tp) + 0.5 * float(total_fp) + 0.5 * float(total_fn)
    if denom == 0:
        pq = 1.0 if total_tp == 0 and total_gt == 0 else 0.0
        rq = 1.0 if total_tp == 0 and total_gt == 0 else 0.0
        sq = 1.0 if total_tp == 0 and total_gt == 0 else 0.0
    else:
        pq = float(total_iou_sum / denom)
        rq = float(total_tp / denom)
        sq = float(total_iou_sum / float(total_tp)) if total_tp > 0 else 0.0

    # Dice distributions
    if image_dice_list:
        dice_mean = float(np.mean(image_dice_list))
        dice_median = float(np.median(image_dice_list))
        dice_p10 = float(np.percentile(image_dice_list, 10))
        dice_p90 = float(np.percentile(image_dice_list, 90))
    else:
        dice_mean = dice_median = dice_p10 = dice_p90 = 0.0

    # IoU distributions
    if all_matched_ious:
        iou_mean = float(np.mean(all_matched_ious))
        iou_median = float(np.median(all_matched_ious))
        iou_p10 = float(np.percentile(all_matched_ious, 10))
        iou_p90 = float(np.percentile(all_matched_ious, 90))
    else:
        iou_mean = iou_median = iou_p10 = iou_p90 = 0.0

    one_to_many_rate = (float(total_one_to_many) / float(total_gt) * 100.0) if total_gt > 0 else 0.0
    many_to_one_rate = (float(total_many_to_one) / float(total_gt) * 100.0) if total_gt > 0 else 0.0
    seconds_per_image = float(elapsed_seconds) / float(num_images)

    return ScorecardResult(
        pq=pq,
        sq=sq,
        rq=rq,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        iou_sum=total_iou_sum,
        dice_mean=dice_mean,
        dice_median=dice_median,
        dice_p10=dice_p10,
        dice_p90=dice_p90,
        iou_mean=iou_mean,
        iou_median=iou_median,
        iou_p10=iou_p10,
        iou_p90=iou_p90,
        one_to_many=total_one_to_many,
        many_to_one=total_many_to_one,
        total_gt=total_gt,
        one_to_many_rate=one_to_many_rate,
        many_to_one_rate=many_to_one_rate,
        seconds_per_image=seconds_per_image,
        num_images=num_images,
        num_drawings=total_drawings,
    )
