"""
Leak-free dataset partitioning grouped by observation date and solar rotation window.

WHY THIS EXISTS:
In solar physics, observations taken on the same day (or by different network sites within
hours) capture identical physical filament states. Furthermore, long-lived filaments rotate
across the solar disk over the ~27-day synodic solar rotation period. Standard random row
splitting causes severe data leakage. This module groups physical observations by calendar
date or 27-day Carrington rotation windows, guaranteeing zero temporal overlap between
train and validation partitions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from solarseg.data import Observation


@dataclass
class DatasetSplit:
    """
    Serializable train/val split manifest.

    WHY THIS EXISTS:
    Guarantees exact experiment reproducibility across local machines and cloud GPUs.
    Stores file lists, unique dates, and split metadata.
    """
    train_files: List[str]
    val_files: List[str]
    train_dates: List[str]
    val_dates: List[str]
    split_strategy: str  # 'date' or 'rotation'
    val_ratio: float
    seed: int
    created_at: str

    @property
    def val_stems(self) -> Set[str]:
        return {Path(f).stem for f in self.val_files}

    @property
    def train_stems(self) -> Set[str]:
        return {Path(f).stem for f in self.train_files}

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> DatasetSplit:
        return cls(**data)


def parse_date_from_filename(filename: str) -> datetime:
    """
    Extract datetime object from GONG observation filename.

    WHY THIS EXISTS:
    Standard GONG naming is YYYYMMDDHHMMSS<site>.jpeg. Allows reliable temporal sorting
    and calculation of 27-day solar rotation epochs.
    """
    stem = Path(filename).stem
    if len(stem) >= 8 and stem[:8].isdigit():
        return datetime.strptime(stem[:8], "%Y%m%d")
    raise ValueError(f"Unable to parse 8-digit date from filename: {filename}")


def compute_solar_rotation_window(dt: datetime, window_days: int = 27) -> int:
    """
    Calculate the solar rotation window index relative to reference epoch 2010-01-01.

    WHY THIS EXISTS:
    The synodic solar rotation period is ~27.275 days. Filaments persist for 1 to 2
    rotations. Grouping by 27-day windows prevents active region leakage.
    """
    epoch = datetime(2010, 1, 1)
    days_since_epoch = (dt - epoch).days
    return days_since_epoch // window_days


def create_date_grouped_split(
    observations: Dict[str, Observation],
    val_ratio: float = 0.20,
    seed: int = 42,
    strategy: str = "date",
    window_days: int = 27,
) -> DatasetSplit:
    """
    Partition observations into train and val with zero date overlap.

    WHY THIS EXISTS:
    Ensures that all annotator drawings of the same physical observation and all
    observations on the same observation date stay strictly within the same partition.

    Args:
        observations: Mapping of file_name -> Observation.
        val_ratio: Target fraction of observations in validation (default ~0.20).
        seed: Random seed for reproducible shuffling.
        strategy: 'date' (calendar day grouping) or 'rotation' (27-day window grouping).
        window_days: Rotation window length in days (default 27).

    Returns:
        DatasetSplit containing train/val file names and verified dates.
    """
    rng = np.random.RandomState(seed)

    # Group files by unit (either date string or rotation window index)
    unit_to_files: Dict[str, List[str]] = {}
    file_to_date: Dict[str, str] = {}

    for file_name, obs in observations.items():
        dt = parse_date_from_filename(file_name)
        date_str = dt.strftime("%Y-%m-%d")
        file_to_date[file_name] = date_str

        if strategy == "rotation":
            unit_key = str(compute_solar_rotation_window(dt, window_days=window_days))
        else:
            unit_key = date_str

        unit_to_files.setdefault(unit_key, []).append(file_name)

    unique_units = sorted(unit_to_files.keys())
    rng.shuffle(unique_units)

    total_files = len(observations)
    target_val_files = int(round(total_files * val_ratio))

    val_files: List[str] = []
    train_files: List[str] = []

    for unit in unique_units:
        files = unit_to_files[unit]
        if len(val_files) + len(files) <= target_val_files or len(val_files) == 0:
            val_files.extend(files)
        else:
            train_files.extend(files)

    train_files.sort()
    val_files.sort()

    train_dates = sorted(list({file_to_date[f] for f in train_files}))
    val_dates = sorted(list({file_to_date[f] for f in val_files}))

    # Verify zero date overlap
    date_overlap = set(train_dates).intersection(set(val_dates))
    if date_overlap:
        raise AssertionError(
            f"Leakage detected! {len(date_overlap)} dates overlap between train and val: {date_overlap}"
        )

    return DatasetSplit(
        train_files=train_files,
        val_files=val_files,
        train_dates=train_dates,
        val_dates=val_dates,
        split_strategy=strategy,
        val_ratio=float(len(val_files)) / float(total_files),
        seed=seed,
        created_at=datetime.utcnow().isoformat(),
    )


def save_split_json(split: DatasetSplit, output_path: Union[str, Path]) -> None:
    """Save split manifest to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2)


def load_split_json(split_path: Union[str, Path]) -> DatasetSplit:
    """Load split manifest from JSON file."""
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetSplit.from_dict(data)


def verify_split_leakage(
    train_files: List[str], val_files: List[str]
) -> Tuple[bool, Set[str]]:
    """
    Verify zero date overlap between arbitrary train and validation file lists.

    WHY THIS EXISTS:
    Acts as an automated assertion in CI/CD, tests, and data loading routines.
    """
    train_dates = {parse_date_from_filename(f).strftime("%Y-%m-%d") for f in train_files}
    val_dates = {parse_date_from_filename(f).strftime("%Y-%m-%d") for f in val_files}
    overlap = train_dates.intersection(val_dates)
    return len(overlap) == 0, overlap
