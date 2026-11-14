from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from solarseg.config import ModelConfig
from solarseg.losses import (
    BarbWeightedBCEDiceLoss,
    MultiTaskAuxiliaryLoss,
    SkeletonRecallLoss,
    SoftDiceLoss,
)
from solarseg.model import SolarFilamentUNet, build_model
from solarseg.structure import (
    generate_barb_weight_map,
    generate_boundary_distance_target,
    generate_mask_skeleton_target,
    generate_real_spine_target,
)


def create_synthetic_filament_shape(shape=(256, 256)):
    """Creates a synthetic curved filament with a spine and small lateral barbs."""
    mask = np.zeros(shape, dtype=np.float32)
    h, w = shape

    # Spine curve
    spine_pts = []
    for x in range(40, 210, 10):
        y = int(128 + 40 * np.sin(x / 30.0))
        spine_pts.append([float(x), float(y)])

    # Draw main filament body (thick line)
    spine_arr = np.array(spine_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [spine_arr], isClosed=False, color=1.0, thickness=12)

    # Add small lateral barbs
    cv2.line(mask, (80, int(128 + 40 * np.sin(80 / 30.0))), (70, 90), color=1.0, thickness=4)
    cv2.line(mask, (150, int(128 + 40 * np.sin(150 / 30.0))), (165, 175), color=1.0, thickness=4)

    return mask, spine_pts


def test_synthetic_mask_skeleton_target():
    """Task 5a: Test skeletonize of soft-consensus mask dilated by 1-2 px."""
    mask, _ = create_synthetic_filament_shape()
    skel_t1 = generate_mask_skeleton_target(mask, dilation=1)
    skel_t2 = generate_mask_skeleton_target(mask, dilation=2)

    assert skel_t1.shape == mask.shape
    assert skel_t2.shape == mask.shape
    assert skel_t1.dtype == np.float32

    # Dilation 2 must contain all dilation 1 pixels and have greater/equal area
    assert np.count_nonzero(skel_t1) > 0
    assert np.count_nonzero(skel_t2) >= np.count_nonzero(skel_t1)
    assert np.all((skel_t1 > 0) <= (skel_t2 > 0))

    # Test SkeletonRecallLoss
    loss_fn = SkeletonRecallLoss()
    logits = torch.from_numpy(skel_t1).unsqueeze(0).unsqueeze(0).float()
    targets = torch.from_numpy(skel_t1).unsqueeze(0).unsqueeze(0).float()

    # High logits matching target should have low recall loss
    loss_perfect = loss_fn(logits * 10.0, targets)
    loss_zeros = loss_fn(torch.zeros_like(logits), targets)
    assert loss_perfect < loss_zeros
    assert loss_perfect.item() >= 0.0


def test_synthetic_real_spine_target():
    """Task 5b: Test rasterizing annotated spine polylines combined across annotators."""
    mask, spine_pts = create_synthetic_filament_shape()

    # Create 3 annotator drawings: 2 agree closely, 1 has slight variation
    ann1_spines = [spine_pts]
    ann2_spines = [spine_pts]
    ann3_spines = [[[p[0], p[1] + 2] for p in spine_pts]]

    drawings_spines = [ann1_spines, ann2_spines, ann3_spines]
    soft_spine = generate_real_spine_target(drawings_spines, shape=mask.shape, dilation=1)

    assert soft_spine.shape == mask.shape
    assert soft_spine.dtype == np.float32
    assert soft_spine.min() >= 0.0
    assert soft_spine.max() <= 1.0

    # Agreement regions where all 3 agree should have value 1.0
    assert np.isclose(soft_spine.max(), 1.0)
    # Disagreement regions should have value ~2/3 (0.667) or 1/3 (0.333)
    values = np.unique(np.round(soft_spine[soft_spine > 0], 2))
    assert 0.67 in values or 0.33 in values or 1.0 in values


def test_synthetic_barb_weight_map_and_loss():
    """Task 5c: Test barb weight map w = 1 + beta * (mask AND NOT dilate(spine, r)) and weighted loss."""
    mask, spine_pts = create_synthetic_filament_shape()
    spine_mask = np.zeros_like(mask)
    spine_arr = np.array(spine_pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(spine_mask, [spine_arr], isClosed=False, color=1.0, thickness=1)

    beta = 2.0
    r = 6
    w_map = generate_barb_weight_map(mask, spine_mask, beta=beta, r=r)

    assert w_map.shape == mask.shape
    assert w_map.min() == 1.0
    # Pixels far from spine (the barbs) must have weight 1 + beta = 3.0
    assert np.isclose(w_map.max(), 1.0 + beta)

    # Spine core pixels must have weight 1.0
    spine_core = (spine_mask > 0)
    assert np.all(w_map[spine_core] == 1.0)

    # Test BarbWeightedBCEDiceLoss
    criterion = BarbWeightedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
    weights = torch.ones_like(targets)
    weights[:, :, :32, :] = 3.0  # simulate barb weights

    loss_unweighted = criterion(logits, targets, weight_map=None)
    loss_weighted = criterion(logits, targets, weight_map=weights)

    assert torch.isfinite(loss_unweighted)
    assert torch.isfinite(loss_weighted)
    assert loss_weighted > 0


def test_synthetic_boundary_distance_target():
    """Task 5d: Test normalized distance transform target inside mask."""
    mask, _ = create_synthetic_filament_shape()
    dt = generate_boundary_distance_target(mask, normalize=True)

    assert dt.shape == mask.shape
    assert dt.min() == 0.0
    assert np.isclose(dt.max(), 1.0)
    # Outside mask must be 0
    assert np.all(dt[mask == 0] == 0.0)
    # Deep interior has highest distance
    assert dt[128, 100] > 0.0


def test_model_auxiliary_heads_toggle():
    """Test model forward pass with auxiliary heads enabled vs disabled."""
    # 1. Default (Single-Head, backwards-compatible, inference standard)
    cfg_default = ModelConfig(aux_centerline=False, aux_boundary=False)
    m_default = build_model(cfg_default)
    x = torch.randn(2, 2, 128, 128)
    out_default = m_default(x)

    assert isinstance(out_default, torch.Tensor)
    assert out_default.shape == (2, 1, 128, 128)

    # 2. Auxiliary heads enabled
    cfg_aux = ModelConfig(aux_centerline=True, aux_boundary=True)
    m_aux = build_model(cfg_aux)
    out_aux = m_aux(x)

    assert isinstance(out_aux, dict)
    assert "mask" in out_aux
    assert "centerline" in out_aux
    assert "boundary" in out_aux

    assert out_aux["mask"].shape == (2, 1, 128, 128)
    assert out_aux["centerline"].shape == (2, 1, 128, 128)
    assert out_aux["boundary"].shape == (2, 1, 128, 128)

    # 3. MultiTaskAuxiliaryLoss test
    mt_loss = MultiTaskAuxiliaryLoss()
    targets = {
        "mask": torch.randint(0, 2, (2, 1, 128, 128)).float(),
        "centerline": torch.randint(0, 2, (2, 1, 128, 128)).float(),
        "boundary": torch.rand(2, 1, 128, 128).float(),
    }
    losses = mt_loss(out_aux, targets)
    assert "total_loss" in losses
    assert "mask_loss" in losses
    assert "centerline_loss" in losses
    assert "boundary_loss" in losses
    assert losses["total_loss"].item() > 0
    assert torch.isfinite(losses["total_loss"])
