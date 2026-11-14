"""
Morphological post-processing pipeline for solar filament instances.

WHY THIS EXISTS:
Converts continuous per-pixel model probabilities into discrete, disjoint instance masks:
1. Threshold: Binarizes probabilities into foreground filament candidates.
2. Gap closing: Morphological closing with a circular kernel bridges small intra-filament
   breaks caused by local intensity fluctuations or faint spines.
3. Hole filling: Fills spurious interior voids within broad filament bodies.
4. Solar disk clipping: Removes false positives outside the solar photosphere.
5. Connected components: Separates disconnected filaments into distinct instances,
   guaranteeing mathematical disjointness (Hard Rule 2).
6. Area filtering: Removes small isolated noise components below min_area pixels.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from solarseg.config import PostprocessConfig
from solarseg.preprocess import SolarDiskParams


def fill_binary_holes(binary_mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary mask using morphological floodfill."""
    h, w = binary_mask.shape
    padded = cv2.copyMakeBorder(binary_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    return (flood[1:-1, 1:-1] != 2).astype(np.uint8)


def postprocess_probability_map(
    prob_map: np.ndarray,
    config: Optional[PostprocessConfig] = None,
    disk: Optional[SolarDiskParams] = None,
    threshold: Optional[float] = None,
    min_area: Optional[int] = None,
    close_radius: Optional[int] = None,
    bridge_gap: Optional[int] = None,
    clip_margin: Optional[float] = None,
    max_rho: Optional[float] = None,
    peak_threshold: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Transform a 2048x2048 probability map into disjoint binary instance masks.

    WHY THIS EXISTS:
    Baseline postprocessing pipeline fulfilling the competition requirements.
    Exposes threshold, min_area, bridge_gap, close_radius, clip_margin, and optional max_rho
    for hyperparameter tuning.

    Args:
        prob_map: 2D float array of shape (H, W) in range [0.0, 1.0].
        config: PostprocessConfig defaults.
        disk: SolarDiskParams for solar limb boundary clipping.
        threshold: Override binarization threshold (default config.threshold).
        min_area: Override minimum pixel area filter (default config.min_area).
        close_radius: Override morphological closing radius (default config.close_radius).
        bridge_gap: Configurable bridging gap parameter.
        clip_margin: Configurable margin beyond disk radius (default config.clip_margin: 1.01).
        max_rho: Optional hard cutoff on solar radius (e.g. 0.95, default config.max_rho).

    Returns:
        List of mutually disjoint 2D binary uint8 arrays of shape (H, W).
    """
    if config is None:
        config = PostprocessConfig()

    th = threshold if threshold is not None else config.threshold
    ma = min_area if min_area is not None else config.min_area
    cr = close_radius if close_radius is not None else config.close_radius
    bg = bridge_gap if bridge_gap is not None else config.bridge_gap
    cm = clip_margin if clip_margin is not None else getattr(config, "clip_margin", 1.01)
    mr = max_rho if max_rho is not None else getattr(config, "max_rho", None)
    pt = peak_threshold if peak_threshold is not None else getattr(config, "peak_threshold", None)

    # 1. Binarize
    binary = (prob_map >= th).astype(np.uint8)
    if np.count_nonzero(binary) == 0:
        return []

    # 2. Close small local gaps (fine boundary smoothing)
    if cr > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cr + 1, 2 * cr + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 3. Bridge disconnected aligned filament fragments (macro gap bridging)
    if bg > 0:
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bg + 1, 2 * bg + 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_bridge)

    # 4. Fill interior holes
    binary = fill_binary_holes(binary)

    # 5. Clip strictly to solar disk + margin / max_rho
    if disk is not None:
        h, w = prob_map.shape
        y_grid, x_grid = np.indices((h, w), dtype=np.float32)
        dist_sq = (x_grid - disk.cx) ** 2 + (y_grid - disk.cy) ** 2
        effective_limit = cm if mr is None else min(cm, mr)
        max_rad = disk.radius * effective_limit
        disk_mask = dist_sq <= (max_rad**2)
        binary[~disk_mask] = 0

    if np.count_nonzero(binary) == 0:
        return []

    # 6. Connected components labeling
    # cv2.connectedComponents guarantees disjoint instance partitions
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    # 7. Filter components by min_area and peak_threshold
    instances: List[np.ndarray] = []
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id).astype(np.uint8)
        area = int(np.count_nonzero(comp_mask))
        if area >= ma:
            if pt is not None:
                max_prob_in_comp = float(np.max(prob_map[comp_mask > 0]))
                if max_prob_in_comp < pt:
                    continue
            instances.append(comp_mask)

    return instances
