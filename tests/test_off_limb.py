"""
Unit tests for off-limb margin preprocessing and postprocessing (Stage 1b-Patch Part B).

WHY THIS EXISTS:
Verifies that features inside the configured off-limb margin survive preprocessing
and postprocessing intact, while pixels outside the margin are strictly zeroed and clipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from solarseg.config import PostprocessConfig
from solarseg.postprocess import postprocess_probability_map
from solarseg.preprocess import (
    SolarDiskParams,
    correct_limb_darkening,
)


def test_preprocessing_off_limb_margin_clipping():
    """Verify that preprocessing preserves pixels within off_limb_margin and zeros beyond."""
    h, w = 2048, 2048
    disk = SolarDiskParams(cx=1024.0, cy=1024.0, radius=900.0)
    margin = 1.05

    # Create synthetic solar disk image
    img = np.full((h, w), 120, dtype=np.uint8)

    # Add distinct test features at specific radii
    # 1. On-disk (rho = 0.50 -> r = 450 px)
    img[1024, 1024 + 450] = 50
    # 2. Inside margin (rho = 1.03 -> r = 927 px)
    img[1024, 1024 + 927] = 80
    # 3. Outside margin (rho = 1.10 -> r = 990 px)
    img[1024, 1024 + 990] = 80

    corr_int, dist_ch = correct_limb_darkening(img, disk, poly_degree=3, off_limb_margin=margin)

    # 1. On-disk feature survives with positive intensity and valid rho
    assert corr_int[1024, 1024 + 450] > 0.0
    assert 0.49 < dist_ch[1024, 1024 + 450] < 0.51

    # 2. Near-limb inside margin (rho = 1.03 <= 1.05) survives
    assert 1.02 < dist_ch[1024, 1024 + 927] < 1.04

    # 3. Beyond margin (rho = 1.10 > 1.05) is strictly zeroed
    assert corr_int[1024, 1024 + 990] == 0.0
    assert dist_ch[1024, 1024 + 990] == 0.0


def test_postprocessing_clip_margin():
    """Verify that postprocessing preserves instances inside clip_margin and clips beyond."""
    h, w = 2048, 2048
    disk = SolarDiskParams(cx=1024.0, cy=1024.0, radius=900.0)
    margin = 1.05  # max_radius = 945 px

    prob_map = np.zeros((h, w), dtype=np.float32)

    # Instance 1: Inside disk (r = 500 px, area ~ 150 px)
    prob_map[1020:1035, 1024 + 500 : 1024 + 510] = 0.95

    # Instance 2: Inside margin (r = 920 px, area ~ 150 px)
    prob_map[1020:1035, 1024 + 920 : 1024 + 930] = 0.95

    # Instance 3: Outside margin (r = 980 px, area ~ 150 px)
    prob_map[1020:1035, 1024 + 980 : 1024 + 990] = 0.95

    config = PostprocessConfig(threshold=0.5, min_area=50, clip_margin=margin)
    instances = postprocess_probability_map(prob_map, config=config, disk=disk)

    # Should find exactly 2 instances (Instance 1 and Instance 2); Instance 3 must be clipped
    assert len(instances) == 2

    # Check instance locations
    inst1_survived = any(m[1025, 1024 + 505] > 0 for m in instances)
    inst2_survived = any(m[1025, 1024 + 925] > 0 for m in instances)
    inst3_survived = any(m[1025, 1024 + 985] > 0 for m in instances)

    assert inst1_survived, "On-disk instance must survive"
    assert inst2_survived, "Near-limb instance within margin must survive"
    assert not inst3_survived, "Instance beyond margin must be clipped"
