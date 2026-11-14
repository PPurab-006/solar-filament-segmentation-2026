"""
Single unified inference entry point for the solar filament segmentation pipeline.

WHY THIS EXISTS:
Per competition instructions (Part G2), a single standardized command:
`python -m solarseg.run --config configs/final_single.yaml --images <dir> --out submission.csv`
or
`python -m solarseg.run --config configs/final_ensemble.yaml --ensemble 5 --images <dir> --out submission.csv`
drives end-to-end inference, postprocessing, and submission generation.
Supports --ensemble {1, 5} through one unified code path (default read from config).
Running the same config on the same images is guaranteed to produce byte-identical CSVs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from solarseg.config import Config, InferenceConfig, ModelConfig, PostprocessConfig, PreprocessConfig
from solarseg.infer import list_test_images, sliding_window_inference
from solarseg.model import build_model
from solarseg.postprocess import postprocess_probability_map
from solarseg.preprocess import (
    assemble_two_channel,
    assert_preprocessing_spec_guard,
    detect_solar_disk_min_circle,
    normalize_solar_intensity_zscore,
    preprocess_and_cache,
)
from solarseg.rle import write_submission


def load_yaml_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_file_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    repo_relative = Path(__file__).resolve().parent.parent / path_str
    if repo_relative.exists():
        return repo_relative
    return p


def run_pipeline(
    config_path: Path,
    images_dir: Path,
    output_csv: Path,
    ensemble: Optional[int] = None,
    limit: Optional[int] = None,
    device_str: Optional[str] = None,
) -> Path:
    config_dict = load_yaml_config(config_path)

    # Determine ensemble mode (CLI overrides config, default=1)
    if ensemble is None:
        ensemble_mode = int(config_dict.get("ensemble", 1))
    else:
        ensemble_mode = int(ensemble)

    if ensemble_mode not in (1, 5):
        raise ValueError(f"Ensemble mode must be 1 or 5, got {ensemble_mode}")

    # Device selection
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model architecture template
    model_dict = config_dict.get("model", {})
    m_cfg = ModelConfig(
        architecture=model_dict.get("architecture", "Unet"),
        encoder_name=model_dict.get("encoder_name", "resnet34"),
        encoder_weights=None,  # Loading from checkpoint
        in_channels=model_dict.get("in_channels", 2),
        classes=model_dict.get("classes", 1),
    )

    # Determine weight paths
    if ensemble_mode == 1:
        single_path_str = config_dict.get("weights_path")
        if not single_path_str:
            raise ValueError("Config missing 'weights_path' for ensemble=1")
        weights_paths = [resolve_file_path(single_path_str)]
    else:  # ensemble_mode == 5
        paths_in_cfg = config_dict.get("weights_paths")
        if paths_in_cfg and len(paths_in_cfg) == 5:
            weights_paths = [resolve_file_path(p) for p in paths_in_cfg]
        else:
            # Fallback to standard 5-fold naming in checkpoints directory
            ckpt_dir = Path(__file__).resolve().parent.parent / "checkpoints"
            weights_paths = [resolve_file_path(str(ckpt_dir / f"fold_{k}_weights.pt")) for k in range(5)]

    # Load models
    models: List[torch.nn.Module] = []
    print(f"[solarseg.run] Operating in ENSEMBLE={ensemble_mode} mode.")
    for idx, wp in enumerate(weights_paths):
        if not wp.exists():
            # If 5-fold checkpoints are not yet on disk (e.g., dry-run before training),
            # check if single weights exist as a testing fallback
            single_fallback = resolve_file_path(config_dict.get("weights_path", "checkpoints/last_epoch_weights.pt"))
            if ensemble_mode == 5 and single_fallback.exists():
                print(f"[solarseg.run] [NOTICE] Checkpoint {wp} not found; falling back to {single_fallback.name} for fold {idx} dry-run.")
                wp = single_fallback
            else:
                raise FileNotFoundError(f"Model weights not found at: {wp}")

        print(f"  [{idx + 1}/{ensemble_mode}] Loading weights from: {wp.name}")
        active_prep_id = "zscore_v1" if m_cfg.in_channels == 3 else "limbflat_v0"
        assert_preprocessing_spec_guard(wp, active_preprocessing_id=active_prep_id)

        state_dict = torch.load(wp, map_location=device, weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        state_dict = {k: v.float() if v.is_floating_point() else v for k, v in state_dict.items()}

        m = build_model(m_cfg).to(device)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)

    # Preprocess & postprocess configs
    prep_dict = config_dict.get("preprocess", {})
    off_limb_margin = prep_dict.get("off_limb_margin", 1.01)

    post_dict = config_dict.get("postprocess", {})
    p_cfg = PostprocessConfig(
        threshold=post_dict.get("threshold", 0.35),
        min_area=post_dict.get("min_area", 200),
        close_radius=post_dict.get("close_radius", 3),
        bridge_gap=post_dict.get("bridge_gap", 0),
        max_rho=post_dict.get("max_rho", None),
        peak_threshold=post_dict.get("peak_threshold", None),
    )

    inf_dict = config_dict.get("inference", {})
    tta_mode = config_dict.get("tta_mode", inf_dict.get("tta_mode", 1))
    i_cfg = InferenceConfig(
        tile_size=inf_dict.get("tile_size", 512),
        stride=inf_dict.get("stride", 256),
        blend_mode=inf_dict.get("blend_mode", "hann"),
        tta=(tta_mode > 1),
        tta_mode=tta_mode,
    )

    # List images deterministically
    images_dir = Path(images_dir)
    image_paths = sorted(list_test_images(images_dir), key=lambda p: p.stem)
    if limit is not None:
        image_paths = image_paths[:limit]

    print(f"[solarseg.run] Found {len(image_paths)} images in {images_dir}")
    print(f"[solarseg.run] Settings: Ensemble={ensemble_mode}, TTA Mode={tta_mode}, th={p_cfg.threshold}, min_area={p_cfg.min_area}")

    predictions: Dict[str, List[np.ndarray]] = {}
    cache_dir = Path(__file__).resolve().parent.parent / "cache"

    with torch.no_grad():
        for img_p in tqdm(image_paths, desc=f"Inference (Ens={ensemble_mode})"):
            if m_cfg.in_channels == 3:
                raw_gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
                if raw_gray is None:
                    raise FileNotFoundError(f"Cannot read image: {img_p}")
                norm_intensity, _, _ = normalize_solar_intensity_zscore(raw_gray)
                disk = detect_solar_disk_min_circle(raw_gray)
                input_tensor_or_arr = norm_intensity
            else:
                intensity, disk = preprocess_and_cache(
                    image_path=img_p,
                    cache_dir=cache_dir,
                    overwrite=False,
                )
                input_tensor_or_arr = assemble_two_channel(intensity, disk, off_limb_margin=off_limb_margin)

            # Single unified code path for probability map computation
            prob_maps = []
            for m in models:
                pm, _ = sliding_window_inference(
                    model=m,
                    two_channel_image=input_tensor_or_arr,
                    config=i_cfg,
                    device=device,
                    disk=disk if m_cfg.in_channels == 3 else None,
                )
                prob_maps.append(pm)

            if len(prob_maps) == 1:
                prob_map = prob_maps[0]
            else:
                prob_map = np.mean(prob_maps, axis=0)

            instances = postprocess_probability_map(
                prob_map=prob_map,
                config=p_cfg,
                disk=disk,
            )
            predictions[img_p.stem] = instances

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = write_submission(output_csv, predictions)
    print(f"[solarseg.run] Wrote {rows_written} filament rows to: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Run filament segmentation pipeline from config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    parser.add_argument("--images", type=Path, required=True, help="Path to input images directory")
    parser.add_argument("--out", type=Path, required=True, help="Path to output submission CSV")
    parser.add_argument("--ensemble", type=int, choices=[1, 5], default=None, help="Ensemble size (1 or 5), overrides config")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")

    args = parser.parse_args()
    run_pipeline(
        config_path=args.config,
        images_dir=args.images,
        output_csv=args.out,
        ensemble=args.ensemble,
        limit=args.limit,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
