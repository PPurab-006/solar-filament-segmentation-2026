"""
Unit tests for train/val splitting logic.

WHY THIS EXISTS:
Verifies that observation date grouping prevents temporal data leakage:
- Zero date overlap between train and val under calendar date grouping
- Zero date overlap under 27-day Carrington solar rotation window grouping
- All annotator drawings of the same observation remain together
"""

from datetime import datetime, timedelta
import pytest

from solarseg.data import AnnotatorDrawing, Observation
from solarseg.split import create_date_grouped_split, verify_split_leakage


def create_synthetic_observations() -> dict[str, Observation]:
    """Build synthetic observations spanning 60 consecutive days with multiple drawings."""
    base_date = datetime(2015, 6, 1)
    observations = {}

    for day_offset in range(60):
        obs_date = base_date + timedelta(days=day_offset)
        date_str = obs_date.strftime("%Y%m%d")
        file_name = f"{date_str}120000Lh.jpeg"

        # Simulate 1 to 3 annotator drawings per observation
        num_drawings = (day_offset % 3) + 1
        drawings = {}
        for d in range(num_drawings):
            d_id = f"annotator_{d}_{file_name}"
            drawings[d_id] = AnnotatorDrawing(
                drawing_id=d_id,
                file_name=file_name,
                date_captured=obs_date.strftime("%Y-%m-%d 12:00:00"),
                height=2048,
                width=2048,
                annotations=[],
            )

        observations[file_name] = Observation(
            file_name=file_name,
            date_captured=obs_date.strftime("%Y-%m-%d 12:00:00"),
            height=2048,
            width=2048,
            drawings=drawings,
        )

    return observations


def test_date_split_zero_date_overlap():
    """Verify that date-grouped splitting yields exactly 0 overlapping dates."""
    observations = create_synthetic_observations()
    split = create_date_grouped_split(observations, val_ratio=0.20, seed=42, strategy="date")

    assert len(split.train_files) > 0
    assert len(split.val_files) > 0
    assert len(split.train_files) + len(split.val_files) == len(observations)

    is_leak_free, overlap = verify_split_leakage(split.train_files, split.val_files)
    assert is_leak_free is True
    assert len(overlap) == 0


def test_rotation_split_zero_date_overlap():
    """Verify that 27-day solar rotation grouping yields zero date overlap."""
    observations = create_synthetic_observations()
    split = create_date_grouped_split(observations, val_ratio=0.20, seed=42, strategy="rotation")

    assert len(split.train_files) > 0
    assert len(split.val_files) > 0
    assert len(split.train_files) + len(split.val_files) == len(observations)

    is_leak_free, overlap = verify_split_leakage(split.train_files, split.val_files)
    assert is_leak_free is True
    assert len(overlap) == 0
