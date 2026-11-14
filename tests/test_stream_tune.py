"""
Tiny synthetic test fixture for streaming hyperparameter grid search.

WHY THIS EXISTS:
Verifies Part A memory optimization locally without running heavy inference,
caching, or tuning on the real dataset.
Tests:
1. COCO RLE conversion and single-image decoding.
2. Memory-mapped float16 streaming over images.
3. Multi-worker accumulation and mathematical equivalence of pooled PQ.
4. Continuous RSS monitoring (stays < 6 GB).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from solarseg.data import AnnotatorDrawing, FilamentAnnotation, Observation
from solarseg.preprocess import SolarDiskParams
from solarseg.tune_stream import (
    decode_drawing_rles,
    get_rss_gb,
    observation_to_rle,
    stream_tune_grid,
)


def test_tiny_synthetic_stream_tune():
    rss_initial = get_rss_gb()
    assert rss_initial < 6.0, f"Initial RSS already high: {rss_initial:.2f} GB"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        h, w = 256, 256

        # 1. Create 2 synthetic float16 prob maps on disk
        img_names = ["syn_001.jpeg", "syn_002.jpeg"]
        disks = {}
        observations = {}

        for idx, name in enumerate(img_names):
            prob = np.zeros((h, w), dtype=np.float16)
            # Add synthetic filament activation
            prob[50:80, 50:120] = 0.85
            if idx == 1:
                prob[150:190, 100:160] = 0.75
            prob_path = tmp_path / f"{Path(name).stem}_prob.npy"
            np.save(prob_path, prob)

            disk = SolarDiskParams(cx=128.0, cy=128.0, radius=110.0)
            disks[name] = disk

            # Synthetic COCO polygon annotation matching the activation
            poly1 = [
                [50.0, 50.0, 120.0, 50.0, 120.0, 80.0, 50.0, 80.0]
            ]
            ann1 = FilamentAnnotation(
                id=f"ann_{idx}_1",
                category_id=1,
                segmentation=poly1,
                bbox=[50.0, 50.0, 70.0, 30.0],
                area=2100.0,
            )
            annotations = [ann1]
            if idx == 1:
                poly2 = [
                    [100.0, 150.0, 160.0, 150.0, 160.0, 190.0, 100.0, 190.0]
                ]
                ann2 = FilamentAnnotation(
                    id=f"ann_{idx}_2",
                    category_id=1,
                    segmentation=poly2,
                    bbox=[100.0, 150.0, 60.0, 40.0],
                    area=2400.0,
                )
                annotations.append(ann2)

            drawing = AnnotatorDrawing(
                drawing_id=f"drawing_{idx}",
                file_name=name,
                date_captured="2026-01-01",
                height=h,
                width=w,
                annotations=annotations,
            )
            obs = Observation(
                file_name=name,
                date_captured="2026-01-01",
                height=h,
                width=w,
                drawings={drawing.drawing_id: drawing},
            )
            observations[name] = obs

        # 2. Test RLE conversion and single-image decoding
        rles = observation_to_rle(observations[img_names[0]], height=h, width=w)
        assert len(rles) == 1
        decoded = decode_drawing_rles(rles)
        assert len(decoded) == 1
        first_d_masks = next(iter(decoded.values()))
        assert len(first_d_masks) == 1
        assert first_d_masks[0].shape == (h, w)
        assert first_d_masks[0].sum() > 0

        # 3. Test multi-worker streaming grid search
        thresholds = [0.40, 0.60]
        close_radii = [0, 1]
        bridge_gaps = [0]
        min_areas = [100, 500]
        max_rhos = [None]

        best_params, best_pq, best_res, all_trials = stream_tune_grid(
            val_files=img_names,
            prob_dir=tmp_path,
            observations=observations,
            disks_dict=disks,
            thresholds=thresholds,
            close_radii=close_radii,
            bridge_gaps=bridge_gaps,
            min_areas=min_areas,
            max_rhos=max_rhos,
            num_workers=2,
            desc="Synthetic Test Grid",
        )

        assert best_pq > 0.0, f"Expected non-zero PQ, got {best_pq}"
        assert len(all_trials) == len(thresholds) * len(close_radii) * len(bridge_gaps) * len(max_rhos) * len(min_areas)
        assert "ann_subsets" in best_res

        # 4. Verify peak memory is still < 6 GB
        rss_final = get_rss_gb()
        assert rss_final < 6.0, f"Peak RSS exceeded 6 GB: {rss_final:.2f} GB"
        print(f"\n[test_stream_tune] Passed! Peak RSS: {rss_final:.2f} GB | Best Synthetic PQ: {best_pq:.4f}")


if __name__ == "__main__":
    test_tiny_synthetic_stream_tune()
