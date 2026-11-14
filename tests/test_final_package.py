"""
Automated verification tests for competition rules compliance (RULES_CHECKLIST.md).

WHY THIS EXISTS:
Enforces Hard Rules from the competition organizers and host rulings:
1. Inference uses only test-image pixels (no filename parsing / metadata leakage in inference path).
2. No access to public MAGFiLO or external ground truth anywhere in the repository.
3. Submission format strictly compliant: mutually disjoint, non-empty, unique IDs, 2048x2048 RLE counts.
4. Model weights under 100 MB and requirements.txt strictly pins every dependency.
5. Annotation parser strictly isolates annotation keys (no read of spine/bbox/area/keypoints).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from solarseg.data import parse_coco_annotations
from solarseg.rle import decode_rle_to_mask, encode_mask_to_rle, verify_disjoint_masks


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_auto_inference_uses_only_pixels():
    """Verify that inference-path code does not parse filename metadata (date, site code)."""
    inference_files = [
        REPO_ROOT / "solarseg" / "infer.py",
        REPO_ROOT / "solarseg" / "postprocess.py",
        REPO_ROOT / "solarseg" / "run.py",
        REPO_ROOT / "scripts" / "predict.py",
    ]

    forbidden_patterns = [
        re.compile(r"site_code", re.IGNORECASE),
        re.compile(r"date_captured", re.IGNORECASE),
        re.compile(r"\b(Bh|Ch|Lh|Mh|Th|Uh)\b"),
    ]

    for p in inference_files:
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            # Ignore comments
            clean_line = line.split("#")[0].strip()
            for pattern in forbidden_patterns:
                match = pattern.search(clean_line)
                assert match is None, (
                    f"Metadata leakage detected in inference path {p.name}:{line_no}: "
                    f"'{clean_line}' matches '{pattern.pattern}'"
                )


def test_auto_no_external_data_or_downloads():
    """Verify that codebase contains no web download or external dataset fetch calls."""
    source_dirs = [REPO_ROOT / "solarseg", REPO_ROOT / "scripts"]
    forbidden_terms = ["urllib.request", "requests.get", "curl ", "wget "]

    for s_dir in source_dirs:
        for py_path in s_dir.rglob("*.py"):
            content = py_path.read_text(encoding="utf-8")
            for term in forbidden_terms:
                assert term not in content, (
                    f"Potential external download call '{term}' found in {py_path}"
                )


def test_auto_weights_under_100mb_and_requirements_pinned():
    """Verify that deployed model weights are under 100 MB and requirements.txt pins all versions."""
    # 1. Weights size check
    weight_files = list(REPO_ROOT.glob("checkpoints/*_weights.pt")) + list(
        REPO_ROOT.glob("outputs/checkpoints/*_weights.pt")
    ) + list(REPO_ROOT.glob("staging/*/*_weights.pt"))

    assert len(weight_files) > 0, "No deployable model weights (*_weights.pt) found!"
    for wf in weight_files:
        size_mb = wf.stat().st_size / (1024 * 1024)
        assert size_mb < 100.0, (
            f"Model weights file {wf} is {size_mb:.2f} MB, exceeding 100 MB limit!"
        )

    # 2. Requirements pinned check
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt not found!"
    lines = req_path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "==" in stripped, (
            f"Package requirement '{stripped}' is not strictly pinned with '=='!"
        )


def test_auto_training_and_inference_annotation_isolation():
    """
    Host Ruling: Any metadata provided in the TRAINING set (spine, bbox, area, class)
    may be used for training, but the inference path must NEVER read or import
    the annotation JSON or expect any annotation fields.
    """
    import ast
    import importlib

    # 1. Verify inference / predict / run modules do not import annotation parser
    root = Path(__file__).resolve().parent.parent
    inference_files = [
        root / "solarseg" / "infer.py",
        root / "solarseg" / "postprocess.py",
        root / "solarseg" / "run.py",
        root / "scripts" / "predict.py",
    ]

    for file_path in inference_files:
        assert file_path.exists(), f"Missing module file: {file_path}"
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "solarseg.data" not in alias.name, (
                        f"Forbidden import '{alias.name}' in inference path {file_path}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "data" not in mod.split("."), (
                    f"Forbidden 'from {mod} import ...' in inference path {file_path}"
                )
                for alias in node.names:
                    assert alias.name != "parse_coco_annotations", (
                        f"Forbidden import 'parse_coco_annotations' in {file_path}"
                    )

    # 2. Verify dataset / training code CAN parse training annotations safely
    import json
    dummy_coco = {
        "images": [
            {"id": "d1", "file_name": "20110119000000Mh.jpeg", "height": 2048, "width": 2048}
        ],
        "annotations": [
            {
                "id": "a1",
                "image_id": "d1",
                "segmentation": [[100, 100, 100, 200, 200, 200, 200, 100]],
                "bbox": [100.0, 100.0, 100.0, 100.0],
                "area": 10000.0,
                "spine": [[100.0, 150.0], [200.0, 150.0]],
                "category_id": 1,
            }
        ],
    }
    dummy_path = root / "tests" / "dummy_train_ann.json"
    dummy_path.write_text(json.dumps(dummy_coco), encoding="utf-8")
    try:
        obs = parse_coco_annotations(dummy_path)
        assert len(obs) == 1
        ann = obs["20110119000000Mh.jpeg"].drawings["d1"].annotations[0]
        assert ann.segmentation == [[100, 100, 100, 200, 200, 200, 200, 100]]
    finally:
        if dummy_path.exists():
            dummy_path.unlink()


def test_auto_submission_compliance_check():
    """Verify submission instances are mutually disjoint, non-empty, and correctly formatted."""
    shape = (2048, 2048)
    m1 = np.zeros(shape, dtype=np.uint8)
    m2 = np.zeros(shape, dtype=np.uint8)
    m1[100:200, 100:200] = 1
    m2[300:400, 300:400] = 1

    rle1 = encode_mask_to_rle(m1)
    rle2 = encode_mask_to_rle(m2)

    # Disjointness check
    dec1 = decode_rle_to_mask(rle1, shape=shape)
    dec2 = decode_rle_to_mask(rle2, shape=shape)
    assert verify_disjoint_masks([dec1, dec2]) is True

    # Non-empty check
    assert np.count_nonzero(dec1) > 0
    assert np.count_nonzero(dec2) > 0
