#!/usr/bin/env python3
"""
Submission CSV validation script.

WHY THIS EXISTS:
Strict pre-submission verification mandated by the competition rules:
- Verifies exact header columns: filament_id, segmentation_rle
- Checks unique filament_id strings with zero duplicates
- Validates that filament_ids match test image names (<imagename>_<k>)
- Decodes every RLE string to verify fixed 2048x2048 grid dimensions
- Verifies that no mask is empty (positive area > 0)
- Enforces strict mutual disjointness across all instances for every image
Prints 'submission=ok' upon passing all assertions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from solarseg.config import Config
from solarseg.rle import decode_rle_to_mask


def list_test_images(test_dir: Path | str) -> List[Path]:
    test_p = Path(test_dir)
    if not test_p.exists():
        return []
    return sorted(
        [p for p in test_p.glob("*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"}],
        key=lambda p: p.stem,
    )


def verify_submission_file(csv_path: Path, test_images_dir: Path = None) -> bool:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Submission CSV does not exist: {csv_path}")

    print(f"[check_submission] Checking: {csv_path}")

    # 1. Verify header and columns
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != ["filament_id", "segmentation_rle"]:
            raise ValueError(
                f"Invalid CSV header. Expected ['filament_id', 'segmentation_rle'], got {header}"
            )

        rows = list(reader)

    print(f"[check_submission] Total rows in submission: {len(rows)}")

    # 2. Check test image IDs if test directory is provided
    valid_test_stems: Set[str] = set()
    if test_images_dir and Path(test_images_dir).exists():
        test_imgs = list_test_images(test_images_dir)
        valid_test_stems = {p.stem for p in test_imgs}
        print(f"[check_submission] Found {len(valid_test_stems)} valid test image names.")

    seen_filament_ids: Set[str] = set()
    image_to_rles: Dict[str, List[Tuple[str, str]]] = {}

    for row_idx, row in enumerate(rows, start=2):
        if len(row) != 2:
            raise ValueError(f"Line {row_idx}: Row does not have exactly 2 columns: {row}")

        filament_id, rle_str = row[0].strip(), row[1].strip()

        # 3. Verify unique filament_id
        if filament_id in seen_filament_ids:
            raise ValueError(f"Line {row_idx}: Duplicate filament_id found: '{filament_id}'")
        seen_filament_ids.add(filament_id)

        # 4. Check format <image_name>_<k>
        parts = filament_id.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(
                f"Line {row_idx}: filament_id '{filament_id}' does not match format '<image_name>_<k>'"
            )

        img_stem, _ = parts[0], int(parts[1])
        if valid_test_stems and img_stem not in valid_test_stems:
            raise ValueError(
                f"Line {row_idx}: image stem '{img_stem}' does not match any known test image in {test_images_dir}"
            )

        image_to_rles.setdefault(img_stem, []).append((filament_id, rle_str))

    # 5. Check RLE decoding, non-emptiness, and disjointness streaming image-by-image (RAM safe)
    print(
        f"[check_submission] Streaming verification across {len(image_to_rles)} images (RAM safe)..."
    )
    for img_stem, rle_list in image_to_rles.items():
        combined = np.zeros((2048, 2048), dtype=np.uint8) if len(rle_list) > 1 else None
        for fil_id, rle_str in rle_list:
            try:
                mask = decode_rle_to_mask(rle_str, shape=(2048, 2048))
            except Exception as e:
                raise ValueError(f"Failed to decode RLE for '{fil_id}': {e}")

            if mask.shape != (2048, 2048):
                raise ValueError(
                    f"Decoded mask shape is {mask.shape}, expected (2048, 2048) for '{fil_id}'"
                )

            area = int(np.count_nonzero(mask))
            if area == 0:
                raise ValueError(f"Mask for '{fil_id}' is empty (area=0)!")

            if combined is not None:
                # Check overlap immediately
                overlap = np.count_nonzero(combined & (mask > 0))
                if overlap > 0:
                    raise ValueError(
                        f"Overlapping instances detected in image '{img_stem}' for '{fil_id}'! "
                        f"Overlap pixels: {overlap}. Hard Rule 2 violated."
                    )
                combined |= (mask > 0).astype(np.uint8)

        del combined

    print("=" * 40)
    print("ALL CHECKS PASSED SUCCESSFULLY:")
    print("  - Header columns verified")
    print("  - All filament_ids are unique")
    print("  - All IDs match test image stems")
    print("  - All RLE strings decode to (2048, 2048)")
    print("  - No empty masks")
    print("  - All instances are mutually disjoint")
    print("=" * 40)
    print("submission=ok")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify competition submission CSV.")
    parser.add_argument(
        "--csv-path",
        "--submission",
        type=Path,
        default=None,
        help="Path to submission.csv (default: outputs/submission.csv)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Path to test_images/ directory",
    )
    args = parser.parse_args()

    cfg = Config()
    csv_p = args.csv_path or (cfg.paths.output_dir / "submission.csv")
    test_d = args.test_dir or cfg.paths.test_images_dir

    verify_submission_file(csv_path=csv_p, test_images_dir=test_d)


if __name__ == "__main__":
    main()
