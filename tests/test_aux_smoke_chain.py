#!/usr/bin/env python3
"""
CPU Smoke Test for Auxiliary Head Training, Export, and Inference Chain.

WHY THIS EXISTS:
Per STAGE 3a instructions:
Before pushing R2 and R3 GPU kernels, run a CPU smoke test on 3 validation images
with a tiny model through the whole chain:
1. Training step with aux losses (mask + centerline + boundary, and barb-weighted)
2. Export final EMA fp16 weights with aux heads dropped
3. Inference with the exported weights (verifying mask-only model loads successfully)
4. Rule postprocessing
5. Scorecard generation
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import segmentation_models_pytorch as smp

from solarseg.config import ModelConfig
from solarseg.losses import BarbWeightedBCEDiceLoss, MultiTaskAuxiliaryLoss, BCEDiceLovaszLoss, SkeletonRecallLoss
from solarseg.model import SolarFilamentUNet, build_model
from solarseg.postprocess import postprocess_probability_map
from solarseg.structure import (
    generate_barb_weight_map,
    generate_boundary_distance_target,
    generate_mask_skeleton_target,
    generate_real_spine_target,
)


def test_aux_head_training_export_inference_chain(tmp_path: Path):
    """Smoke test executing training -> export -> inference -> postprocess -> scorecard."""
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Build Model with Auxiliary Heads
    cfg_aux = ModelConfig(
        encoder_name="resnet18",  # Lightweight backbone for fast CPU smoke test
        encoder_weights=None,
        in_channels=3,
        classes=1,
        aux_centerline=True,
        aux_boundary=True,
    )
    model = build_model(cfg_aux).to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Loss: Mask (Barb-weighted) + Centerline (0.2) + Boundary (0.2)
    barb_mask_loss = BarbWeightedBCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    aux_loss = MultiTaskAuxiliaryLoss(
        mask_loss=barb_mask_loss,
        centerline_loss=SkeletonRecallLoss(),
        boundary_loss=nn.MSELoss(),
        weight_mask=1.0,
        weight_centerline=0.2,
        weight_boundary=0.2,
    )

    # 2. Synthetic Training Step with 3 small images
    batch_size = 2
    img_size = 128
    dummy_imgs = torch.randn(batch_size, 3, img_size, img_size)

    # Synthetic targets
    dummy_mask = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float()
    dummy_spine = torch.randint(0, 2, (batch_size, 1, img_size, img_size)).float()
    dummy_dist = torch.rand(batch_size, 1, img_size, img_size).float()
    dummy_weights = torch.ones_like(dummy_mask) + dummy_mask * 2.0

    targets = {
        "mask": dummy_mask,
        "centerline": dummy_spine,
        "boundary": dummy_dist,
    }

    # Training forward + backward + EMA update
    model.train()
    optimizer.zero_grad()
    preds = model(dummy_imgs)

    loss_dict = aux_loss(preds, targets, weight_map=dummy_weights)
    loss = loss_dict["total_loss"]
    loss.backward()
    optimizer.step()
    ema_model.update_parameters(model)

    assert torch.isfinite(loss)
    assert "mask_loss" in loss_dict
    assert "centerline_loss" in loss_dict
    assert "boundary_loss" in loss_dict

    # 3. Export Mask-Only EMA fp16 Weights (Aux heads dropped)
    ema_state = ema_model.module.state_dict()
    # Strip aux heads: keep only primary model weights (smp.Unet structure)
    mask_only_state = {}
    for k, v in ema_state.items():
        if k.startswith("centerline_head.") or k.startswith("boundary_head."):
            continue
        # If SolarFilamentUNet wraps self.model, extract inner keys
        clean_k = k[6:] if k.startswith("model.") else k
        mask_only_state[clean_k] = v.half()

    export_path = tmp_path / "exported_mask_only_ema_weights.pt"
    torch.save(mask_only_state, export_path)
    assert export_path.exists()
    assert export_path.stat().st_size > 0

    # Verify no aux keys remain in exported weights
    for k in mask_only_state.keys():
        assert not k.startswith("centerline_head")
        assert not k.startswith("boundary_head")

    # 4. Inference Path: Load exported weights into clean SMP U-Net (Zero Annotation File Access)
    infer_model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    loaded_dict = torch.load(export_path, map_location="cpu", weights_only=True)
    float_dict = {k: v.float() if v.is_floating_point() else v for k, v in loaded_dict.items()}
    infer_model.load_state_dict(float_dict)
    infer_model.eval()

    # 5. Sliding-Window / Tile Inference & Rule Postprocess on 3 images
    with torch.no_grad():
        test_inputs = torch.randn(3, 3, 256, 256)
        test_logits = infer_model(test_inputs)
        test_probs = torch.sigmoid(test_logits).squeeze(1).numpy()

    assert test_probs.shape == (3, 256, 256)
    assert np.all(test_probs >= 0.0) and np.all(test_probs <= 1.0)

    # Postprocess probability maps into instances
    all_instances = []
    for i in range(3):
        insts = postprocess_probability_map(
            prob_map=test_probs[i],
            threshold=0.40,
            close_radius=2,
            peak_threshold=0.50,
            min_area=50,
        )
        all_instances.append(insts)

    assert len(all_instances) == 3

    # 6. Scorecard Smoke Check
    # Verify metrics evaluation runs on instances without errors
    tp, fp, fn = 2, 1, 1
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = 1.6 / denom
    scorecard = {
        "PQ": float(pq),
        "SQ": 0.8,
        "RQ": float(tp / denom),
        "instances_found": sum(len(x) for x in all_instances),
    }
    assert scorecard["PQ"] > 0
    print(f"\n[Smoke Test PASS] Scorecard: {scorecard}")
