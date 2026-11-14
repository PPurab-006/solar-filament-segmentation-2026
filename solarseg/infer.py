"""
Sliding-window inference with 2D Hann-window tile blending and Test-Time Augmentation (TTA).

WHY THIS EXISTS:
Evaluating a 2048x2048 image in a single forward pass requires substantial GPU memory
and leads to domain shift from the 512x512 training crops. Tiling with abrupt box boundaries
produces visible seam artifacts where edges cut across continuous filaments.
Hann-window blending smoothly tapers tile weights to zero at borders, yielding seamless
continuous reconstructions. 8-fold D4 TTA eliminates orientation biases. Caching raw logits
enables instant post-processing threshold tuning without re-running the neural network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from solarseg.config import InferenceConfig
from solarseg.preprocess import SolarDiskParams, extract_context_crop


def generate_2d_hann_window(tile_size: int = 512) -> np.ndarray:
    """
    Generate a 2D Hann (raised cosine) weighting window of shape (tile_size, tile_size).

    WHY THIS EXISTS:
    Smoothly tapers tile contributions from 1.0 at center to near 0.0 at the perimeter,
    completely preventing tile boundary stitching seams in sliding-window inference.
    """
    n = np.arange(tile_size, dtype=np.float32)
    # Continuous Hann window with non-zero edge epsilon to avoid singular weights
    w1d = np.sin(np.pi * (n + 0.5) / float(tile_size)) ** 2
    w2d = np.outer(w1d, w1d).astype(np.float32)
    # Clip small minimum weight
    w2d = np.maximum(w2d, 1e-4)
    return w2d


def predict_tile_tta(
    model: nn.Module,
    tile: torch.Tensor,
    device: torch.device,
    tta_mode: int = 8,
) -> torch.Tensor:
    """
    Predict a single tile with test-time augmentation.

    WHY THIS EXISTS:
    High-throughput batched TTA implementation:
    - Preprocessed input tile is constructed once.
    - All augmentations are stacked into a single batch on the GPU.
    - Executes exactly ONE forward pass for the entire batch.
    - Un-augments and averages predictions in-place on the GPU.
    Runs TTA-4 and TTA-8 at near-linear or sub-linear scaling relative to TTA-1.

    Args:
        model: PyTorch segmentation model.
        tile: Input tile tensor of shape (1, C, H, W).
        device: Target execution device.
        tta_mode: 1, 4, or 8.

    Returns:
        Averaged probability tensor of shape (1, 1, H, W) in [0.0, 1.0].
    """
    use_cuda = (device.type == "cuda")

    if tta_mode == 1:
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                out = model(tile.to(device))
                return torch.sigmoid(out)

    tile_dev = tile.to(device)

    if tta_mode == 4:
        # 4 flip variants: [identity, hflip, vflip, hvflip]
        batch = torch.cat([
            tile_dev,
            torch.flip(tile_dev, dims=[3]),
            torch.flip(tile_dev, dims=[2]),
            torch.flip(tile_dev, dims=[2, 3]),
        ], dim=0)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                out = model(batch)
                probs = torch.sigmoid(out)

        # Invert flips on GPU
        p0 = probs[0:1]
        p1 = torch.flip(probs[1:2], dims=[3])
        p2 = torch.flip(probs[2:3], dims=[2])
        p3 = torch.flip(probs[3:4], dims=[2, 3])
        return (p0 + p1 + p2 + p3) / 4.0

    elif tta_mode == 8:
        # Full D4 dihedral group (8 variants batched in a single forward pass)
        flipped = torch.flip(tile_dev, dims=[3])

        batch = torch.cat([
            tile_dev,
            torch.rot90(tile_dev, 1, dims=[2, 3]),
            torch.rot90(tile_dev, 2, dims=[2, 3]),
            torch.rot90(tile_dev, 3, dims=[2, 3]),
            flipped,
            torch.rot90(flipped, 1, dims=[2, 3]),
            torch.rot90(flipped, 2, dims=[2, 3]),
            torch.rot90(flipped, 3, dims=[2, 3]),
        ], dim=0)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                out = model(batch)
                probs = torch.sigmoid(out)

        # Invert rotations and flips on GPU
        inv0 = probs[0:1]
        inv1 = torch.rot90(probs[1:2], -1, dims=[2, 3])
        inv2 = torch.rot90(probs[2:3], -2, dims=[2, 3])
        inv3 = torch.rot90(probs[3:4], -3, dims=[2, 3])
        inv4 = torch.flip(probs[4:5], dims=[3])
        inv5 = torch.flip(torch.rot90(probs[5:6], -1, dims=[2, 3]), dims=[3])
        inv6 = torch.flip(torch.rot90(probs[6:7], -2, dims=[2, 3]), dims=[3])
        inv7 = torch.flip(torch.rot90(probs[7:8], -3, dims=[2, 3]), dims=[3])

        return (inv0 + inv1 + inv2 + inv3 + inv4 + inv5 + inv6 + inv7) / 8.0

    else:
        raise ValueError(f"Unsupported tta_mode: {tta_mode}. Must be 1, 4, or 8.")


def sliding_window_inference_3ch(
    model: nn.Module,
    norm_intensity: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    config: Optional[InferenceConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    cache_logits_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window inference for 3-channel context representation:
    [Channel 0: normalized intensity, Channel 1: rho radial distance, Channel 2: 1536x1536 context downscaled 3x].

    WHY THIS EXISTS:
    Executes identical sliding-window inference with 2D Hann blending across the 3-channel
    context representation used by ConvNeXt-Tiny models (R1, R2, R3).
    """
    if config is None:
        config = InferenceConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    model.to(device)

    h, w = norm_intensity.shape[-2:]
    tile_size = config.tile_size
    stride = config.stride

    hann_weights = generate_2d_hann_window(tile_size)
    accum_probs = np.zeros((h, w), dtype=np.float32)
    accum_weights = np.zeros((h, w), dtype=np.float32)

    y_starts = list(range(0, h - tile_size + 1, stride))
    if (h - tile_size) not in y_starts:
        y_starts.append(h - tile_size)

    x_starts = list(range(0, w - tile_size + 1, stride))
    if (w - tile_size) not in x_starts:
        x_starts.append(w - tile_size)

    effective_tta = getattr(config, "tta_mode", 1)
    if getattr(config, "tta", False) and effective_tta == 1:
        effective_tta = 8

    with torch.no_grad():
        for y1 in y_starts:
            y2 = y1 + tile_size
            for x1 in x_starts:
                x2 = x1 + tile_size

                crop_int = np.array(norm_intensity[y1:y2, x1:x2], dtype=np.float32)
                y_g, x_g = np.indices((tile_size, tile_size), dtype=np.float32)
                abs_y = y_g + y1
                abs_x = x_g + x1
                crop_rho = np.sqrt((abs_x - cx) ** 2 + (abs_y - cy) ** 2) / radius
                crop_rho[crop_rho > 1.01] = 0.0

                crop_ctx = extract_context_crop(norm_intensity, y1, x1, tile_size=tile_size, context_size=1536)
                tile_3ch = np.stack([crop_int, crop_rho, crop_ctx], axis=0)
                tile_tensor = torch.from_numpy(tile_3ch).unsqueeze(0).float()

                prob_tile = predict_tile_tta(model, tile_tensor, device, tta_mode=effective_tta)

                prob_np = prob_tile.squeeze().cpu().numpy().astype(np.float32)
                accum_probs[y1:y2, x1:x2] += prob_np * hann_weights
                accum_weights[y1:y2, x1:x2] += hann_weights

    valid_mask = accum_weights > 0
    prob_map = np.zeros((h, w), dtype=np.float32)
    prob_map[valid_mask] = accum_probs[valid_mask] / accum_weights[valid_mask]

    # Mask strictly off-disk (> 1.01 solar radii)
    y_full, x_full = np.indices((h, w), dtype=np.float32)
    rho_full = np.sqrt((x_full - cx) ** 2 + (y_full - cy) ** 2) / radius
    prob_map[rho_full > 1.01] = 0.0

    clipped_p = np.clip(prob_map, 1e-6, 1.0 - 1e-6)
    blended_logits = np.log(clipped_p / (1.0 - clipped_p)).astype(np.float32)

    if cache_logits_path is not None:
        cache_logits_path = Path(cache_logits_path)
        cache_logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_logits_path, prob_map.astype(np.float16))

    return prob_map, blended_logits


def sliding_window_inference(
    model: nn.Module,
    two_channel_image: np.ndarray,
    config: Optional[InferenceConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    cache_logits_path: Optional[Union[str, Path]] = None,
    disk: Optional[SolarDiskParams] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run sliding-window inference with Hann-window blending across 2048x2048 image.
    Supports both 2-channel preprocessed representation and 3-channel context mode
    (when disk is supplied and input is 2D intensity).
    """
    if two_channel_image.ndim == 2 and disk is not None:
        return sliding_window_inference_3ch(
            model=model,
            norm_intensity=two_channel_image,
            cx=disk.cx,
            cy=disk.cy,
            radius=disk.radius,
            config=config,
            device=device,
            cache_logits_path=cache_logits_path,
        )

    if config is None:
        config = InferenceConfig()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    model.to(device)

    _, h, w = two_channel_image.shape
    tile_size = config.tile_size
    stride = config.stride

    # Create 2D Hann window
    hann_weights = generate_2d_hann_window(tile_size)

    # Accumulator buffers for weighted probabilities and weights
    accum_probs = np.zeros((h, w), dtype=np.float32)
    accum_weights = np.zeros((h, w), dtype=np.float32)

    # Generate grid coordinates
    y_starts = list(range(0, h - tile_size + 1, stride))
    if (h - tile_size) % stride != 0 and (h - tile_size) not in y_starts:
        y_starts.append(h - tile_size)

    x_starts = list(range(0, w - tile_size + 1, stride))
    if (w - tile_size) % stride != 0 and (w - tile_size) not in x_starts:
        x_starts.append(w - tile_size)

    # Determine effective TTA mode
    effective_tta = getattr(config, "tta_mode", 1)
    if getattr(config, "tta", False) and effective_tta == 1:
        effective_tta = 8

    with torch.no_grad():
        for y1 in y_starts:
            y2 = y1 + tile_size
            for x1 in x_starts:
                x2 = x1 + tile_size

                tile_np = two_channel_image[:, y1:y2, x1:x2]
                tile_tensor = torch.from_numpy(tile_np).unsqueeze(0).float()

                prob_tile = predict_tile_tta(model, tile_tensor, device, tta_mode=effective_tta)

                prob_np = prob_tile.squeeze().cpu().numpy().astype(np.float32)

                accum_probs[y1:y2, x1:x2] += prob_np * hann_weights
                accum_weights[y1:y2, x1:x2] += hann_weights

    # Normalize blended probabilities
    valid_mask = accum_weights > 0
    prob_map = np.zeros((h, w), dtype=np.float32)
    prob_map[valid_mask] = accum_probs[valid_mask] / accum_weights[valid_mask]

    # Compute corresponding pseudo-logits: log(p / (1 - p))
    clipped_p = np.clip(prob_map, 1e-6, 1.0 - 1e-6)
    blended_logits = np.log(clipped_p / (1.0 - clipped_p)).astype(np.float32)

    # Cache logits/probs if requested for rapid tuning
    if cache_logits_path is not None:
        cache_logits_path = Path(cache_logits_path)
        cache_logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_logits_path, prob_map.astype(np.float16))

    return prob_map, blended_logits


def cache_val_probabilities(
    model: nn.Module,
    val_files: List[str],
    images_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    device: torch.device,
    tile_size: int = 512,
    stride: int = 256,
    tta_mode: int = 8,
) -> Path:
    """
    Cache blended probability maps for all validation images.

    WHY THIS EXISTS:
    Allows fast local grid-search hyperparameter tuning of morphological postprocessing
    (threshold, min_area, close_radius, bridge_gap) on validation PQ without re-running the neural net.
    Saves float16 arrays (~8 MB each) on disk.
    """
    from tqdm import tqdm
    from solarseg.preprocess import assemble_two_channel, preprocess_and_cache

    output_dir.mkdir(parents=True, exist_ok=True)
    inf_config = InferenceConfig(
        tile_size=tile_size,
        stride=stride,
        tta=(tta_mode > 1),
        tta_mode=tta_mode,
    )

    print(f"[infer] Caching {tta_mode}-fold TTA probabilities for {len(val_files)} validation images...")
    for fn in tqdm(val_files, desc=f"Caching Val TTA-{tta_mode} Probs"):
        out_path = output_dir / f"{Path(fn).stem}_prob.npy"
        if out_path.exists():
            continue

        img_path = images_dir / fn
        intensity_mmap, disk = preprocess_and_cache(img_path, cache_dir=cache_dir)
        two_ch = assemble_two_channel(intensity_mmap, disk, off_limb_margin=1.01)
        prob_map, _ = sliding_window_inference(
            model=model,
            two_channel_image=two_ch,
            config=inf_config,
            device=device,
        )
        np.save(out_path, prob_map.astype(np.float16))

    print(f"[infer] Completed validation probability caching to: {output_dir}")
    return output_dir


def list_test_images(test_dir: Union[str, Path]) -> List[Path]:
    """
    Find and return sorted list of test image file paths for inference.
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        return []

    images = sorted(
        [
            p
            for p in test_dir.glob("*")
            if p.suffix.lower() in {".jpeg", ".jpg", ".png"}
        ]
    )
    return images
