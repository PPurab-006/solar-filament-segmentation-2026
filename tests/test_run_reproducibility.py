"""
Unit test for pipeline reproducibility (Part G2).

WHY THIS EXISTS:
Verifies that executing:
`python -m solarseg.run --config configs/final_A.yaml --images <dir> --out submission.csv`
twice on the same test input generates 100% byte-identical CSV submissions.
"""

import filecmp
from pathlib import Path

import pytest

from solarseg.run import run_pipeline


def test_pipeline_reproducibility(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "configs" / "final_A.yaml"
    test_images_dir = repo_root / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"

    if not test_images_dir.exists() or not config_path.exists():
        pytest.skip("Test images or config_A not available")

    out_csv1 = tmp_path / "submission_run1.csv"
    out_csv2 = tmp_path / "submission_run2.csv"

    # Run on 1 image to keep local unit test fast (< 5s on CPU)
    run_pipeline(
        config_path=config_path,
        images_dir=test_images_dir,
        output_csv=out_csv1,
        limit=1,
        device_str="cpu",
    )

    run_pipeline(
        config_path=config_path,
        images_dir=test_images_dir,
        output_csv=out_csv2,
        limit=1,
        device_str="cpu",
    )

    assert out_csv1.exists() and out_csv2.exists()
    assert out_csv1.stat().st_size > 0

    # Strict byte-level equality check
    assert filecmp.cmp(out_csv1, out_csv2, shallow=False), (
        "Pipeline runs produced non-identical CSV outputs!"
    )
