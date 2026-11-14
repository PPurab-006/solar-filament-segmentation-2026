"""
Unit tests for the Preprocessing Spec Guard.

Verifies:
1. Matching preprocessing ID succeeds.
2. Mismatched preprocessing ID raises ValueError.
3. preprocess_solar_image is deprecated and blocked from R1/ConvNeXt models.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from solarseg.preprocess import (
    assert_preprocessing_spec_guard,
    preprocess_solar_image,
    PREPROCESS_ZSCORE_V1,
    PREPROCESS_LIMBFLAT_V0,
)


def test_preprocessing_guard_match(tmp_path: Path):
    weights_path = tmp_path / "test_model_weights.pt"
    weights_path.touch()
    meta_path = tmp_path / "test_model_weights.meta.json"
    with open(meta_path, "w") as f:
        json.dump({"preprocessing_id": PREPROCESS_ZSCORE_V1}, f)

    # Assert matching preprocessing ID succeeds without exception
    assert_preprocessing_spec_guard(weights_path, active_preprocessing_id=PREPROCESS_ZSCORE_V1)


def test_preprocessing_guard_mismatch(tmp_path: Path):
    weights_path = tmp_path / "test_model_weights.pt"
    weights_path.touch()
    meta_path = tmp_path / "test_model_weights.meta.json"
    with open(meta_path, "w") as f:
        json.dump({"preprocessing_id": PREPROCESS_ZSCORE_V1}, f)

    # Assert mismatched preprocessing ID raises ValueError
    with pytest.raises(ValueError, match=r"\[PREPROCESSING SPEC GUARD\] Mismatch detected"):
        assert_preprocessing_spec_guard(weights_path, active_preprocessing_id=PREPROCESS_LIMBFLAT_V0)


def test_preprocess_solar_image_convnext_blocked():
    synthetic_img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\[DEPRECATION GUARD\]"):
        preprocess_solar_image(synthetic_img, target_family="convnext_tiny")

    with pytest.raises(ValueError, match=r"\[DEPRECATION GUARD\]"):
        preprocess_solar_image(synthetic_img, target_family="r1")
