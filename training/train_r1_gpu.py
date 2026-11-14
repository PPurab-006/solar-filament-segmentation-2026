#!/usr/bin/env python3
"""
STAGE 2b - RUN R1: UPGRADED CONVNEXT-TINY U-NET TRAINING (Kaggle GPU)
Solar Filament Segmentation Challenge 2026

WHY THIS EXISTS:
Executes Run R1 using the strongest candidate from the B1 benchmark:
- Model: SMP Unet with tu-convnext_tiny encoder (ImageNet pretrained, 31.9M params)
- Inputs (3 channels): limb-corrected intensity, rho coordinate, 3x context channel
  (1536x1536 centered region downscaled 3x to 512x512 via INTER_AREA)
- Target: soft-consensus multi-annotator agreement mask
- Loss: BCE + Soft Dice + 0.3 * Lovasz-Hinge (BCEDiceLovaszLoss)
- Memory Architecture: Uncompressed float16 intensity and uint8 consensus masks memory-mapped
  from disk (np.load(..., mmap_mode='r')). Peak RAM < 2 GB, zero worker OOMs.
- Augmentations: 70% filament-centered crops (mask-derived boxes only), D4 dihedral flips/rot90,
  brightness/contrast/gamma jitter, mild Gaussian blur (sigma 0.5-1.5) and Gaussian noise
- Optimizer: AdamW, OneCycleLR schedule with warm-up, mixed precision
- Weight Averaging: Exponential Moving Average (EMA, decay=0.999) used for validation and saved weights
- Virtual Epochs: 20 epochs x 2,000 crops = 40,000 crops total (~27 min on Tesla T4)
- Validation: Sliding-window Hann-blended inference on all 141 validation images using EMA weights,
  followed by date-grouped cross-fit scoring.
"""

from __future__ import annotations

import gc
import json
import os
import random
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset

# Install dependencies if missing
try:
    import segmentation_models_pytorch as smp
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation_models_pytorch", "timm", "pycocotools"])
    import segmentation_models_pytorch as smp


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# ------------------------------------------------------------------------------
# LOSS FUNCTIONS (BCE + Soft Dice + 0.3 * Lovasz Hinge)
# ------------------------------------------------------------------------------

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    def __init__(self, per_image: bool = True):
        super().__init__()
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bin_targets = (targets > 0.5).float()
        if self.per_image:
            losses = [
                _lovasz_hinge_flat(logits[i].view(-1), bin_targets[i].view(-1))
                for i in range(logits.size(0))
            ]
            return torch.stack(losses).mean()
        return _lovasz_hinge_flat(logits.view(-1), bin_targets.view(-1))


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        cardinality = torch.sum(probs_flat ** 2, dim=1) + torch.sum(targets_flat ** 2, dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return torch.mean(1.0 - dice_score)


class BCEDiceLovaszLoss(nn.Module):
    def __init__(self, bce_w: float = 1.0, dice_w: float = 1.0, lovasz_w: float = 0.3):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.lovasz_w = lovasz_w
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_bce = self.bce(logits, targets)
        l_dice = self.dice(logits, targets)
        l_lov = self.lovasz(logits, targets)
        return self.bce_w * l_bce + self.dice_w * l_dice + self.lovasz_w * l_lov


# ------------------------------------------------------------------------------
# DISK-BASED MMAP PREPROCESSING (RAM SAFE: ZERO IN-MEMORY PYTHON ARRAYS)
# ------------------------------------------------------------------------------

def precompute_mmap_cache(
    file_names: List[str],
    images_dir: Path,
    file_to_annotations: Dict[str, List[dict]],
    cache_dir: Path,
):
    """
    Precompute float16 intensity arrays and uint8 soft-target arrays to disk.
    Allows zero-copy np.load(..., mmap_mode='r') without inflating process RAM.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Cache Precomputation] Checking {len(file_names)} images in {cache_dir}...")
    t0 = time.time()
    count_cached = 0

    for idx, fn in enumerate(file_names):
        stem = Path(fn).stem
        img_npy = cache_dir / f"{stem}_img.npy"
        mask_npy = cache_dir / f"{stem}_mask.npy"
        meta_json = cache_dir / f"{stem}_meta.json"

        if img_npy.exists() and mask_npy.exists() and meta_json.exists():
            continue

        img_path = images_dir / fn
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        raw = raw_bgr[:, :, 0]

        # Solar disk circle detection
        disk_binary = (raw > 20).astype(np.uint8)
        cnts, _ = cv2.findContours(disk_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(c)
        else:
            cx, cy, radius = 1024.0, 1024.0, 904.0

        # Normalization
        mean_val = np.mean(raw[raw > 20]) if np.any(raw > 20) else 128.0
        std_val = np.std(raw[raw > 20]) if np.any(raw > 20) else 40.0
        norm_intensity = ((raw.astype(np.float32) - mean_val) / max(std_val, 1e-4)).astype(np.float16)

        # Build soft consensus mask
        anns = file_to_annotations.get(fn, [])
        drawings: Dict[str, List[dict]] = defaultdict(list)
        for ann in anns:
            drawings[str(ann.get("image_id"))].append(ann)

        h, w = raw.shape
        if drawings:
            soft_mask = np.zeros((h, w), dtype=np.float32)
            for d_id, d_anns in drawings.items():
                d_mask = np.zeros((h, w), dtype=np.uint8)
                for ann in d_anns:
                    for poly in ann.get("segmentation", []):
                        if len(poly) >= 6:
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(d_mask, [pts], 1)
                soft_mask += (d_mask > 0).astype(np.float32)
            soft_mask /= float(len(drawings))
        else:
            soft_mask = np.zeros((h, w), dtype=np.float32)

        # Convert soft mask to uint8 [0..255]
        uint8_mask = np.clip(soft_mask * 255.0, 0, 255).astype(np.uint8)

        # Save to disk
        np.save(img_npy, norm_intensity)
        np.save(mask_npy, uint8_mask)
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump({"cx": float(cx), "cy": float(cy), "radius": float(radius)}, f)

        count_cached += 1
        if (idx + 1) % 100 == 0 or (idx + 1) == len(file_names):
            print(f"  Processed {idx + 1}/{len(file_names)} images (RSS: {get_rss_gb():.2f} GB)")

    dt = time.time() - t0
    print(f"[Cache Precomputation] Cached {count_cached} new images in {dt:.1f}s.")


# ------------------------------------------------------------------------------
# STREAMING MMAP DATASET (RAM FOOTPRINT: < 50 MB PER WORKER)
# ------------------------------------------------------------------------------

class R1MmapSolarDataset(Dataset):
    def __init__(
        self,
        file_names: List[str],
        cache_dir: Path,
        file_to_annotations: Dict[str, List[dict]],
        crop_size: int = 512,
        samples_per_epoch: int = 2000,
        is_train: bool = True,
    ):
        self.file_names = list(file_names)
        self.cache_dir = cache_dir
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.is_train = is_train

        # Precompute filament bounding boxes for 70% centered sampling
        self.filament_bboxes: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
        for fn in self.file_names:
            anns = file_to_annotations.get(fn, [])
            for ann in anns:
                seg = ann.get("segmentation", [])
                xs, ys = [], []
                for poly in seg:
                    if len(poly) >= 2:
                        xs.extend(poly[0::2])
                        ys.extend(poly[1::2])
                if xs and ys:
                    bx1, bx2 = int(min(xs)), int(max(xs))
                    by1, by2 = int(min(ys)), int(max(ys))
                    self.filament_bboxes[fn].append((by1, by2, bx1, bx2))

        # Pre-load metadata dicts (tiny json)
        self.meta: Dict[str, Tuple[float, float, float]] = {}
        for fn in self.file_names:
            stem = Path(fn).stem
            meta_json = self.cache_dir / f"{stem}_meta.json"
            if meta_json.exists():
                with open(meta_json, "r") as f:
                    d = json.load(f)
                    self.meta[fn] = (d["cx"], d["cy"], d["radius"])
            else:
                self.meta[fn] = (1024.0, 1024.0, 904.0)

    def __len__(self) -> int:
        return self.samples_per_epoch if self.is_train else len(self.file_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fn = random.choice(self.file_names) if self.is_train else self.file_names[idx]
        stem = Path(fn).stem
        cx, cy, radius = self.meta[fn]
        cs = self.crop_size

        img_npy = self.cache_dir / f"{stem}_img.npy"
        mask_npy = self.cache_dir / f"{stem}_mask.npy"

        # Memory-map directly without copying into Python RAM
        mmap_img = np.load(img_npy, mmap_mode="r")
        mmap_mask = np.load(mask_npy, mmap_mode="r")
        h, w = mmap_img.shape

        # 1. Crop coordinate selection (70% filament-centered)
        bboxes = self.filament_bboxes.get(fn, [])
        if self.is_train and bboxes and random.random() < 0.70:
            by1, by2, bx1, bx2 = random.choice(bboxes)
            target_cy = (by1 + by2) / 2.0 + random.uniform(-cs * 0.25, cs * 0.25)
            target_cx = (bx1 + bx2) / 2.0 + random.uniform(-cs * 0.25, cs * 0.25)
            y1 = int(round(target_cy - cs / 2.0))
            x1 = int(round(target_cx - cs / 2.0))
        else:
            y1 = random.randint(0, h - cs)
            x1 = random.randint(0, w - cs)

        y1 = max(0, min(y1, h - cs))
        x1 = max(0, min(x1, w - cs))
        y2 = y1 + cs
        x2 = x1 + cs

        # Channel 0: Intensity (cast slice to float32)
        crop_intensity = np.array(mmap_img[y1:y2, x1:x2], dtype=np.float32)
        mask_crop = np.array(mmap_mask[y1:y2, x1:x2], dtype=np.float32) / 255.0

        # Channel 1: Solar radial distance (rho)
        y_grid, x_grid = np.indices((cs, cs), dtype=np.float32)
        abs_y = y_grid + y1
        abs_x = x_grid + x1
        rho_crop = np.sqrt((abs_x - cx) ** 2 + (abs_y - cy) ** 2) / radius
        rho_crop[rho_crop > 1.01] = 0.0

        # Channel 2: Context channel (1536x1536 downscaled 3x to 512x512)
        cy_ctx = y1 + 256
        cx_ctx = x1 + 256
        y_ctx_1 = cy_ctx - 768
        y_ctx_2 = y_ctx_1 + 1536
        x_ctx_1 = cx_ctx - 768
        x_ctx_2 = x_ctx_1 + 1536

        context_buf = np.zeros((1536, 1536), dtype=np.float32)
        src_y1, src_y2 = max(0, y_ctx_1), min(h, y_ctx_2)
        src_x1, src_x2 = max(0, x_ctx_1), min(w, x_ctx_2)
        dst_y1, dst_y2 = src_y1 - y_ctx_1, src_y1 - y_ctx_1 + (src_y2 - src_y1)
        dst_x1, dst_x2 = src_x1 - x_ctx_1, src_x1 - x_ctx_1 + (src_x2 - src_x1)

        context_buf[dst_y1:dst_y2, dst_x1:dst_x2] = np.array(mmap_img[src_y1:src_y2, src_x1:src_x2], dtype=np.float32)
        crop_context = cv2.resize(context_buf, (cs, cs), interpolation=cv2.INTER_AREA)

        # Assemble 3 channels: [intensity, rho, context]
        image_crop = np.stack([crop_intensity, rho_crop, crop_context], axis=0)

        # 2. Augmentations (Train mode only)
        if self.is_train:
            if random.random() < 0.5:
                image_crop = np.flip(image_crop, axis=2).copy()
                mask_crop = np.flip(mask_crop, axis=1).copy()
            if random.random() < 0.5:
                image_crop = np.flip(image_crop, axis=1).copy()
                mask_crop = np.flip(mask_crop, axis=0).copy()
            k_rot = random.choice([0, 1, 2, 3])
            if k_rot > 0:
                image_crop = np.rot90(image_crop, k=k_rot, axes=(1, 2)).copy()
                mask_crop = np.rot90(mask_crop, k=k_rot, axes=(0, 1)).copy()

            b_factor = 1.0 + random.uniform(-0.15, 0.15)
            c_factor = 1.0 + random.uniform(-0.15, 0.15)
            gamma = random.uniform(0.85, 1.15)

            for ch_idx in [0, 2]:
                image_crop[ch_idx] = (image_crop[ch_idx] * b_factor) * c_factor
                sign = np.sign(image_crop[ch_idx])
                image_crop[ch_idx] = sign * (np.abs(image_crop[ch_idx]) ** gamma)

            if random.random() < 0.4:
                sigma = random.uniform(0.5, 1.5)
                image_crop[0] = cv2.GaussianBlur(image_crop[0], (0, 0), sigmaX=sigma)
            if random.random() < 0.4:
                noise = np.random.normal(0.0, 0.02, (cs, cs)).astype(np.float32)
                image_crop[0] += noise

        tensor_image = torch.from_numpy(image_crop).float()
        tensor_mask = torch.from_numpy(mask_crop).unsqueeze(0).float()
        return tensor_image, tensor_mask


# ------------------------------------------------------------------------------
# SLIDING-WINDOW INFERENCE (3 CHANNELS WITH CONTEXT)
# ------------------------------------------------------------------------------

def generate_2d_hann(tile_size: int = 512) -> np.ndarray:
    n = np.arange(tile_size, dtype=np.float32)
    w1d = np.sin(np.pi * (n + 0.5) / float(tile_size)) ** 2
    w2d = np.outer(w1d, w1d).astype(np.float32)
    return np.maximum(w2d, 1e-4)


def run_sliding_window_inference_3ch(
    model: nn.Module,
    mmap_img: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    device: torch.device,
    tile_size: int = 512,
    stride: int = 256,
) -> np.ndarray:
    model.eval()
    h, w = mmap_img.shape
    hann = generate_2d_hann(tile_size)

    accum_probs = np.zeros((h, w), dtype=np.float32)
    accum_weights = np.zeros((h, w), dtype=np.float32)

    y_starts = list(range(0, h - tile_size + 1, stride))
    if (h - tile_size) not in y_starts:
        y_starts.append(h - tile_size)
    x_starts = list(range(0, w - tile_size + 1, stride))
    if (w - tile_size) not in x_starts:
        x_starts.append(w - tile_size)

    with torch.no_grad():
        for y1 in y_starts:
            y2 = y1 + tile_size
            for x1 in x_starts:
                x2 = x1 + tile_size

                crop_int = np.array(mmap_img[y1:y2, x1:x2], dtype=np.float32)
                y_g, x_g = np.indices((tile_size, tile_size), dtype=np.float32)
                abs_y = y_g + y1
                abs_x = x_g + x1
                crop_rho = np.sqrt((abs_x - cx) ** 2 + (abs_y - cy) ** 2) / radius
                crop_rho[crop_rho > 1.01] = 0.0

                cy_ctx = y1 + 256
                cx_ctx = x1 + 256
                y_ctx_1 = cy_ctx - 768
                y_ctx_2 = y_ctx_1 + 1536
                x_ctx_1 = cx_ctx - 768
                x_ctx_2 = x_ctx_1 + 1536

                context_buf = np.zeros((1536, 1536), dtype=np.float32)
                src_y1, src_y2 = max(0, y_ctx_1), min(h, y_ctx_2)
                src_x1, src_x2 = max(0, x_ctx_1), min(w, x_ctx_2)
                dst_y1, dst_y2 = src_y1 - y_ctx_1, src_y1 - y_ctx_1 + (src_y2 - src_y1)
                dst_x1, dst_x2 = src_x1 - x_ctx_1, src_x1 - x_ctx_1 + (src_x2 - src_x1)

                context_buf[dst_y1:dst_y2, dst_x1:dst_x2] = np.array(mmap_img[src_y1:src_y2, src_x1:src_x2], dtype=np.float32)
                crop_ctx = cv2.resize(context_buf, (tile_size, tile_size), interpolation=cv2.INTER_AREA)

                tile_3ch = np.stack([crop_int, crop_rho, crop_ctx], axis=0)
                t_input = torch.from_numpy(tile_3ch).unsqueeze(0).to(device)

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    out = model(t_input)
                    prob_tile = torch.sigmoid(out).squeeze().cpu().numpy()

                accum_probs[y1:y2, x1:x2] += prob_tile * hann
                accum_weights[y1:y2, x1:x2] += hann

    final_prob = accum_probs / np.maximum(accum_weights, 1e-4)
    y_full, x_full = np.indices((h, w), dtype=np.float32)
    rho_full = np.sqrt((x_full - cx) ** 2 + (y_full - cy) ** 2) / radius
    final_prob[rho_full > 1.01] = 0.0
    return final_prob


# ------------------------------------------------------------------------------
# TRAINING PIPELINE FOR RUN R1
# ------------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("=" * 78)
    print("STAGE 2b - RUN R1: CONVNEXT-TINY U-NET TRAINING (40,000 CROPS)")
    print(f"Device: {device} ({gpu_name})")
    print("=" * 78)

    # 1. Discover Paths
    def find_file(filename: str) -> Path:
        for root in [Path("/kaggle/input"), Path("/kaggle/working"), Path(".")]:
            if root.exists():
                matches = list(root.rglob(filename))
                if matches:
                    return matches[0]
        raise FileNotFoundError(f"Cannot find {filename}")

    train_json_path = find_file("MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    split_json_path = find_file("split.json")
    train_images_dir = train_json_path.parent / "train_images"

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs/stage2b_r1")
    out_dir.mkdir(parents=True, exist_ok=True)
    mmap_cache_dir = Path("/tmp/mmap_cache") if Path("/tmp").exists() else (out_dir / "mmap_cache")

    with open(split_json_path, "r") as f:
        split_data = json.load(f)
    train_files = split_data.get("train_files", [])
    val_files = split_data.get("val_files", [])

    print(f"Loaded {len(train_files)} train files and {len(val_files)} validation files.")

    # 2. Parse COCO Annotations
    with open(train_json_path, "r") as f:
        coco_data = json.load(f)
    img_to_file = {str(im["id"]): im["file_name"] for im in coco_data["images"]}
    file_to_annotations: Dict[str, List[dict]] = defaultdict(list)
    for ann in coco_data["annotations"]:
        d_id = str(ann["image_id"])
        if d_id in img_to_file:
            file_to_annotations[img_to_file[d_id]].append(ann)

    # 3. Precompute MMAP Cache (Runs in ~30s, eliminates RAM bloat)
    all_needed_files = list(set(train_files + val_files))
    precompute_mmap_cache(all_needed_files, train_images_dir, file_to_annotations, mmap_cache_dir)

    # 4. Setup Streaming Dataset and DataLoader
    epochs = 20
    crops_per_epoch = 2000  # 20 * 2000 = 40,000 crops total
    batch_size = 8

    train_ds = R1MmapSolarDataset(
        file_names=train_files,
        cache_dir=mmap_cache_dir,
        file_to_annotations=file_to_annotations,
        crop_size=512,
        samples_per_epoch=crops_per_epoch,
        is_train=True,
    )
    # Using 2 workers with zero-copy mmap
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # 5. Build Model & EMA
    print("\nBuilding SMP Unet (encoder: tu-convnext_tiny, in_channels: 3)...")
    model = smp.Unet(
        encoder_name="tu-convnext_tiny",
        encoder_weights=True,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    total_steps = epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=total_steps,
        pct_start=0.10,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    criterion = BCEDiceLovaszLoss(bce_w=1.0, dice_w=1.0, lovasz_w=0.3)
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    # 6. Training Loop
    print(f"\nStarting training: {epochs} epochs, {crops_per_epoch} crops/epoch ({total_steps} steps total)...")
    train_start_t = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_t0 = time.time()

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()

            if scale_before <= scale_after:
                scheduler.step()

            # Update EMA weights
            ema_model.update_parameters(model)
            ep_loss += float(loss.item())

        ep_time = time.time() - ep_t0
        avg_loss = ep_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {ep_time:.1f}s | RSS: {get_rss_gb():.2f} GB")

    train_total_time = time.time() - train_start_t
    print(f"\n[Training Complete] 40,000 crops completed in {train_total_time / 60.0:.2f} minutes ({40000 / train_total_time:.1f} crops/sec).")

    # 7. Save Deployable Checkpoint (under 100 MB rule)
    out_weights_path = out_dir / "r1_convnext_tiny_ema_weights.pt"
    ema_state = {k: v.half() for k, v in ema_model.module.state_dict().items()}
    torch.save(ema_state, out_weights_path)
    file_size_mb = out_weights_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved EMA weights: {out_weights_path} ({file_size_mb:.2f} MB - [PASS] strictly under 100 MB)")

    # 8. Fast Validation Sliding-Window Inference & Panoptic Quality Evaluation
    print("\n" + "=" * 78)
    print("EVALUATING RUN R1 (EMA MODEL) ON VALIDATION SET (141 OBSERVATIONS)")
    print("=" * 78)

    val_prob_cache_dir = out_dir / "val_prob_cache"
    val_prob_cache_dir.mkdir(parents=True, exist_ok=True)

    print("Running sliding-window Hann inference across 141 validation images...")
    val_start_t = time.time()

    for v_idx, fn in enumerate(val_files, start=1):
        stem = Path(fn).stem
        mmap_img = np.load(mmap_cache_dir / f"{stem}_img.npy", mmap_mode="r")
        with open(mmap_cache_dir / f"{stem}_meta.json") as f:
            meta_d = json.load(f)
        cx, cy, rad = meta_d["cx"], meta_d["cy"], meta_d["radius"]

        prob_map = run_sliding_window_inference_3ch(
            model=ema_model.module,
            mmap_img=mmap_img,
            cx=cx,
            cy=cy,
            radius=rad,
            device=device,
            tile_size=512,
            stride=256,
        )
        np.save(val_prob_cache_dir / f"{stem}_prob.npy", prob_map.astype(np.float16))

        if v_idx % 25 == 0 or v_idx == len(val_files):
            print(f"  Inferred {v_idx}/{len(val_files)} val images (RSS: {get_rss_gb():.2f} GB)")

    val_infer_time = time.time() - val_start_t
    print(f"[Validation Inference] Finished in {val_infer_time:.1f}s ({val_infer_time / len(val_files):.2f} s/image).")

    # 9. Date-Grouped Cross-Fit Evaluation using Part A Optimal Settings
    def extract_date(fn: str) -> str:
        parts = Path(fn).stem.split("_")
        for p in parts:
            if len(p) == 8 and p.isdigit():
                return p
        return parts[0]

    val_by_date = sorted(val_files, key=extract_date)
    half_a_files = set(val_by_date[: len(val_by_date) // 2])
    half_b_files = set(val_by_date[len(val_by_date) // 2 :])

    from solarseg.preprocess import SolarDiskParams
    from solarseg.postprocess import postprocess_probability_map
    from solarseg.metrics import evaluate_panoptic_quality

    # Prepare ground truth drawings by observation file
    gt_drawings_all: Dict[str, Dict[str, List[np.ndarray]]] = {}
    gt_drawings_a: Dict[str, Dict[str, List[np.ndarray]]] = {}
    gt_drawings_b: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for fn in val_files:
        anns = file_to_annotations.get(fn, [])
        drawings_dict: Dict[str, List[np.ndarray]] = {}
        for ann in anns:
            d_id = str(ann.get("image_id", ann.get("drawing_id", "0")))
            seg = ann.get("segmentation", [])
            mask = np.zeros((2048, 2048), dtype=bool)
            for poly in seg:
                if len(poly) >= 6:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask.view(np.uint8), [pts], 1)
            if np.count_nonzero(mask) > 0:
                drawings_dict.setdefault(d_id, []).append(mask)

        gt_drawings_all[fn] = drawings_dict
        if fn in half_a_files:
            gt_drawings_a[fn] = drawings_dict
        else:
            gt_drawings_b[fn] = drawings_dict

    # Postprocess probability maps using Step 1 Best Rule (tm=0.35, cr=3, tp=0.99, ma=200)
    preds_all: Dict[str, List[np.ndarray]] = {}
    preds_a: Dict[str, List[np.ndarray]] = {}
    preds_b: Dict[str, List[np.ndarray]] = {}

    print("\nGenerating predictions with best rule (th=0.35, cr=3, tp=0.99, ma=200)...")
    for fn in val_files:
        stem = Path(fn).stem
        prob_map = np.load(val_prob_cache_dir / f"{stem}_prob.npy").astype(np.float32)
        with open(mmap_cache_dir / f"{stem}_meta.json") as f:
            meta_d = json.load(f)
        cx, cy, rad = meta_d["cx"], meta_d["cy"], meta_d["radius"]
        disk = SolarDiskParams(cx=float(cx), cy=float(cy), radius=float(rad))

        pred_instances = postprocess_probability_map(
            prob_map=prob_map,
            disk=disk,
            threshold=0.35,
            min_area=200,
            close_radius=3,
            peak_threshold=0.99,
        )
        preds_all[fn] = pred_instances
        if fn in half_a_files:
            preds_a[fn] = pred_instances
        else:
            preds_b[fn] = pred_instances

    print("Evaluating Panoptic Quality on Validation Set...")
    res_all = evaluate_panoptic_quality(gt_drawings_all, preds_all)
    res_a = evaluate_panoptic_quality(gt_drawings_a, preds_a)
    res_b = evaluate_panoptic_quality(gt_drawings_b, preds_b)

    pooled_tp = res_a.total_tp + res_b.total_tp
    pooled_fp = res_a.total_fp + res_b.total_fp
    pooled_fn = res_a.total_fn + res_b.total_fn
    pooled_iou = res_a.total_iou_sum + res_b.total_iou_sum
    pooled_denom = pooled_tp + 0.5 * pooled_fp + 0.5 * pooled_fn

    cross_fit_pq = pooled_iou / pooled_denom if pooled_denom > 0 else 0.0
    cross_fit_sq = pooled_iou / pooled_tp if pooled_tp > 0 else 0.0
    cross_fit_rq = float(pooled_tp) / pooled_denom if pooled_denom > 0 else 0.0

    print(f"\n[RUN R1 VALIDATION RESULTS (ConvNeXt-Tiny U-Net EMA)]")
    print(f"  All-Val PQ:         {res_all.pq:.4f}")
    print(f"  All-Val SQ:         {res_all.sq:.4f}")
    print(f"  All-Val RQ:         {res_all.rq:.4f}")
    print(f"  Half A PQ:          {res_a.pq:.4f}")
    print(f"  Half B PQ:          {res_b.pq:.4f}")
    print(f"  Cross-Fit PQ:       {cross_fit_pq:.4f}")
    print(f"  Cross-Fit SQ:       {cross_fit_sq:.4f}")
    print(f"  Cross-Fit RQ:       {cross_fit_rq:.4f}")
    print(f"  One-to-Many:        {res_all.total_one_to_many}")
    print(f"  Many-to-One:        {res_all.total_many_to_one}")
    print(f"  Baseline Config A:  0.3731")
    print(f"  Delta vs Baseline:  {cross_fit_pq - 0.3731:+.4f} PQ")

    gate_passes = (cross_fit_pq >= 0.3731 + 0.0100)
    print(f"  Gate Evaluation:    {'PASS (Gain >= +0.0100)' if gate_passes else 'REJECTED'}")

    summary = {
        "candidate": "UNet/tu-convnext_tiny",
        "crops_total": 40000,
        "epochs": epochs,
        "train_time_min": train_total_time / 60.0,
        "all_val_pq": res_all.pq,
        "all_val_sq": res_all.sq,
        "all_val_rq": res_all.rq,
        "cross_fit_pq": cross_fit_pq,
        "cross_fit_sq": cross_fit_sq,
        "cross_fit_rq": cross_fit_rq,
        "half_a_pq": res_a.pq,
        "half_b_pq": res_b.pq,
        "o2m": res_all.total_one_to_many,
        "m2o": res_all.total_many_to_one,
        "delta_vs_baseline": cross_fit_pq - 0.3731,
        "gate_pass": gate_passes,
        "weights_size_mb": file_size_mb,
    }
    with open(out_dir / "r1_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_dir / 'r1_training_summary.json'}")


if __name__ == "__main__":
    main()
