"""
Unit tests for structure signals extracted from binary instance masks.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from solarseg.structure import (
    compute_mask_area,
    compute_mask_bbox,
    compute_mask_distance_transform,
    compute_mask_elongation,
    compute_mask_mean_width,
    compute_mask_skeleton,
    compute_mask_solidity,
    extract_structure_signals,
)


def test_synthetic_rectangle_structure():
    """Test on a 100x10 horizontal bar: elongation ~ 10, solidity ~ 1.0, width ~ 10."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[95:105, 50:150] = 1  # 10 pixels high, 100 pixels wide -> area = 1000

    area = compute_mask_area(mask)
    assert area == 1000

    bbox = compute_mask_bbox(mask)
    assert bbox == (95, 105, 50, 150)

    skel, skel_len = compute_mask_skeleton(mask)
    assert skel_len > 80 and skel_len <= 100

    width = compute_mask_mean_width(area, skel_len)
    assert 9.0 <= width <= 13.0

    elongation = compute_mask_elongation(mask)
    assert elongation > 8.0, f"Expected elongation > 8 for 100x10 bar, got {elongation}"

    solidity = compute_mask_solidity(mask)
    assert solidity > 0.95

    dt = compute_mask_distance_transform(mask)
    assert dt.max() == pytest.approx(5.0, abs=0.5)


def test_synthetic_circle_structure():
    """Test on a circle: elongation ~ 1.0, solidity ~ 1.0."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 1, thickness=-1)

    area = compute_mask_area(mask)
    assert area > 4500 and area < 5500

    elongation = compute_mask_elongation(mask)
    assert 0.9 <= elongation <= 1.1, f"Expected elongation ~ 1.0 for circle, got {elongation}"

    solidity = compute_mask_solidity(mask)
    assert solidity > 0.95


def test_synthetic_l_shape_structure():
    """Test on an L-shape: solidity significantly below 1.0 due to concavity."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[20:120, 20:40] = 1   # Vertical: 100 x 20
    mask[100:120, 20:120] = 1  # Horizontal: 20 x 100

    solidity = compute_mask_solidity(mask)
    assert solidity < 0.8, f"Expected solidity < 0.8 for L-shape, got {solidity}"

    signals = extract_structure_signals(mask)
    assert signals.area > 0
    assert signals.skeleton_length > 0
    assert signals.bbox == (20, 120, 20, 120)
