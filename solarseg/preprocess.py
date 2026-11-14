"""
Solar disk limb detection and limb-darkening correction preprocessing.

WHY THIS EXISTS:
Solar H-alpha observations have two severe physical artifacts:
1. Solar limb darkening: center-to-limb temperature decrease causes quiet-Sun intensity
   to fall off by >40% towards the limb. Filaments near the center look very different
   from filaments near the limb if uncorrected.
2. Atmospheric & instrumental halos: Simple thresholding produces oversized disks because
   scattered light extends well beyond the true solar limb.
This module fits the true solar limb to the sharpest radial gradient (steepest intensity drop),
computes undisturbed photospheric radial profiles using median radial binning, flattens
limb-darkening, zeroes all values outside the disk, and outputs a 2-channel representation:
[corrected_intensity, distance_from_center_in_solar_radii].
"""

import hashlib
import json
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np


PREPROCESS_ZSCORE_V1 = "zscore_v1"
PREPROCESS_LIMBFLAT_V0 = "limbflat_v0"
VALID_PREPROCESSING_IDS = {PREPROCESS_ZSCORE_V1, PREPROCESS_LIMBFLAT_V0}


def assert_preprocessing_spec_guard(weights_path: Union[str, Path], active_preprocessing_id: str) -> None:
    """
    Assert that the active preprocessing pipeline matches the model checkpoint's expected preprocessing ID.
    Fails loudly with ValueError if there is a mismatch.

    WHY THIS EXISTS:
    Prevents catastrophic inference failure where a model trained on zscore_v1 (mean 0.0, std 1.0)
    is erroneously evaluated with limbflat_v0 (mean 1.0, std 0.15).
    """
    weights_p = Path(weights_path)
    meta_p = weights_p.parent / f"{weights_p.stem}.meta.json"
    if not meta_p.exists():
        meta_p = weights_p.with_suffix(".meta.json")

    if meta_p.exists():
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected_id = meta.get("preprocessing_id")
        if expected_id and expected_id != active_preprocessing_id:
            raise ValueError(
                f"[PREPROCESSING SPEC GUARD] Mismatch detected: model weights {weights_p.name} "
                f"require preprocessing_id='{expected_id}', but active pipeline is configured with '{active_preprocessing_id}'! "
                f"Halting inference to prevent catastrophic feature distribution mismatch."
            )


def compute_preprocess_hash(
    min_radius_ratio: float = 0.80,
    max_radius_ratio: float = 0.96,
    radial_step: float = 1.0,
    poly_degree: int = 5,
    off_limb_margin: float = 1.01,
) -> str:
    """Compute deterministic 8-char hex hash of preprocessing configuration."""
    key_str = f"{min_radius_ratio:.3f}_{max_radius_ratio:.3f}_{radial_step:.1f}_{poly_degree}_{off_limb_margin:.3f}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:8]


@dataclass
class SolarDiskParams:
    """Fitted solar disk geometry."""
    cx: float
    cy: float
    radius: float


def fit_circle_algebraic(x_coords: np.ndarray, y_coords: np.ndarray) -> SolarDiskParams:
    """
    Fit circle (cx, cy, R) to 2D Cartesian points using linear least-squares.

    WHY THIS EXISTS:
    Solves x^2 + y^2 + D*x + E*y + F = 0 in closed form for fast, non-iterative,
    exact circle fitting to the detected limb boundary points.
    """
    A = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)])
    b = -(x_coords**2 + y_coords**2)
    # Solve A * [D, E, F]^T = b
    res, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    d, e, f = res
    cx = -d / 2.0
    cy = -e / 2.0
    rad_sq = cx**2 + cy**2 - f
    radius = float(np.sqrt(max(rad_sq, 1.0)))
    return SolarDiskParams(cx=float(cx), cy=float(cy), radius=radius)


def detect_solar_disk_limb(
    img: np.ndarray,
    num_rays: int = 360,
    min_radius_ratio: float = 0.80,
    max_radius_ratio: float = 0.96,
    radial_step: float = 1.0,
) -> SolarDiskParams:
    """
    Detect solar limb by finding the sharpest negative radial intensity gradient.

    WHY THIS EXISTS:
    Simple thresholding fails because atmospheric halos and coronal scatter cause
    thresholded masks to bleed past the real limb. The true solar limb (photosphere)
    is marked by the sharpest drop (steepest negative derivative dI/dr).
    For 2048x2048 GONG images, the true solar limb across all sites is ~904 px,
    which corresponds to radius ratio ~0.883 of the half-dimension (1024 px).

    Args:
        img: 2D uint8 or float grayscale image (H, W).
        num_rays: Number of angular radial rays to cast from center.
        min_radius_ratio: Search start as fraction of half-image dimension (0.80 -> 819 px).
        max_radius_ratio: Search end as fraction of half-image dimension (0.96 -> 983 px).
        radial_step: Pixel sampling step along each ray.

    Returns:
        SolarDiskParams containing center coordinates (cx, cy) and radius R.
    """
    h, w = img.shape
    # Initial coarse center estimate from image dimensions
    cx_init = w / 2.0
    cy_init = h / 2.0
    half_dim = min(h, w) / 2.0

    r_min = min_radius_ratio * half_dim
    r_max = max_radius_ratio * half_dim
    radii = np.arange(r_min, r_max, radial_step, dtype=np.float32)
    thetas = np.linspace(0, 2 * np.pi, num_rays, endpoint=False, dtype=np.float32)

    # Pre-blur slightly to suppress local granulation noise before derivative
    smooth = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 1.5)

    detected_x = []
    detected_y = []
    detected_r = []

    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        sample_x = cx_init + radii * cos_t
        sample_y = cy_init + radii * sin_t

        # Map coordinates with bilinear interpolation
        # map_coordinates expects (y, x)
        coords = np.vstack([sample_y, sample_x])
        from scipy.ndimage import map_coordinates

        ray_profile = map_coordinates(smooth, coords, order=1, mode="nearest")

        # Sharpest negative radial gradient = minimum of dI/dr
        grad = np.gradient(ray_profile, radial_step)
        min_idx = np.argmin(grad)
        r_star = radii[min_idx]

        detected_r.append(r_star)
        detected_x.append(cx_init + r_star * cos_t)
        detected_y.append(cy_init + r_star * sin_t)

    detected_x = np.array(detected_x, dtype=np.float32)
    detected_y = np.array(detected_y, dtype=np.float32)
    detected_r = np.array(detected_r, dtype=np.float32)

    # Reject outlier edge points via Median Absolute Deviation (MAD)
    med_r = np.median(detected_r)
    mad = np.median(np.abs(detected_r - med_r))
    inliers = np.abs(detected_r - med_r) <= max(3.0 * mad, 10.0)

    if np.sum(inliers) < 10:
        inliers = np.ones_like(detected_r, dtype=bool)

    disk = fit_circle_algebraic(detected_x[inliers], detected_y[inliers])
    return disk


def correct_limb_darkening(
    img: np.ndarray,
    disk: SolarDiskParams,
    poly_degree: int = 5,
    off_limb_margin: float = 1.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Correct center-to-limb darkening and normalize on-disk and near-limb intensity.

    WHY THIS EXISTS:
    Quiet-Sun solar intensity naturally dims towards the limb due to viewing optical depth.
    Dividing by the median quiet-Sun radial profile flattens the background, making filaments
    have uniform contrast across disk center and limb. Pixels outside the disk but within
    off_limb_margin are normalized relative to the limb intensity. Pixels beyond the margin
    are strictly zeroed.

    Args:
        img: 2D uint8 or float grayscale image (H, W).
        disk: SolarDiskParams containing fitted cx, cy, radius.
        poly_degree: Degree of polynomial for radial profile fit.
        off_limb_margin: Radial margin factor beyond disk radius (default: 1.01 = +1%).

    Returns:
        Tuple of (corrected_intensity, distance_in_solar_radii):
        - corrected_intensity: float32, normalized on-disk/near-limb, 0.0 beyond margin.
        - distance_in_solar_radii: float32, [0, off_limb_margin] inside margin, 0.0 beyond.
    """
    h, w = img.shape
    y_grid, x_grid = np.indices((h, w), dtype=np.float32)
    dx = x_grid - disk.cx
    dy = y_grid - disk.cy
    dist_pixels = np.sqrt(dx**2 + dy**2)
    rho = dist_pixels / max(disk.radius, 1.0)  # Distance in solar radii

    disk_mask = rho <= 1.0

    # Fit undisturbed background profile using median radial bins on disk
    # Binning avoids corruption by dark absorption filaments and sunspots
    inner_mask = rho <= 0.98  # Avoid extreme boundary pixels during profile fitting
    sample_rho = rho[inner_mask]
    sample_val = img[inner_mask].astype(np.float32)

    num_bins = 50
    bin_edges = np.linspace(0.0, 0.98, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_medians = np.zeros(num_bins, dtype=np.float32)

    for b in range(num_bins):
        in_bin = (sample_rho >= bin_edges[b]) & (sample_rho < bin_edges[b + 1])
        if np.any(in_bin):
            # 75th percentile represents quiet photosphere above absorption features
            bin_medians[b] = np.percentile(sample_val[in_bin], 70)
        else:
            bin_medians[b] = bin_medians[b - 1] if b > 0 else 100.0

    # Polynomial fit B(rho)
    poly_coeffs = np.polyfit(bin_centers, bin_medians, deg=min(poly_degree, num_bins - 1))
    profile_eval = np.polyval(poly_coeffs, rho).astype(np.float32)
    profile_eval = np.maximum(profile_eval, 1.0)  # Avoid division by zero

    # Flatten intensity on disk: I_corrected = I / B(rho)
    corrected_intensity = np.zeros((h, w), dtype=np.float32)
    corrected_intensity[disk_mask] = (
        img[disk_mask].astype(np.float32) / profile_eval[disk_mask]
    )

    # Normalize on-disk intensity so median is ~1.0
    on_disk_vals = corrected_intensity[disk_mask]
    med_on_disk = 1.0
    if len(on_disk_vals) > 0:
        med_on_disk = float(np.median(on_disk_vals))
        if med_on_disk > 0:
            corrected_intensity[disk_mask] /= med_on_disk

    # Simplified off-limb margin handling: normalize relative to limb edge intensity
    margin_mask = (rho > 1.0) & (rho <= off_limb_margin)
    if np.any(margin_mask):
        limb_val = float(np.polyval(poly_coeffs, 1.0))
        corrected_intensity[margin_mask] = img[margin_mask].astype(np.float32) / max(limb_val, 1.0)
        if med_on_disk > 0:
            corrected_intensity[margin_mask] /= med_on_disk

    # Strictly zero beyond the off-limb margin
    outside_margin = rho > off_limb_margin
    corrected_intensity[outside_margin] = 0.0

    # Distance channel: rho inside off-limb margin, 0.0 beyond margin
    distance_channel = np.zeros((h, w), dtype=np.float32)
    valid_margin = rho <= off_limb_margin
    distance_channel[valid_margin] = rho[valid_margin]

    return corrected_intensity, distance_channel


def preprocess_solar_image(
    img: np.ndarray,
    num_rays: int = 360,
    min_radius_ratio: float = 0.80,
    max_radius_ratio: float = 0.96,
    poly_degree: int = 5,
    off_limb_margin: float = 1.01,
    target_family: str = "resnet34",
) -> Tuple[np.ndarray, SolarDiskParams]:
    """
    [DEPRECATED FOR R1/R2/R3/CONVNEXT FAMILY]
    Legacy 2-channel limb-darkening flattening preprocessing (limbflat_v0).
    ONLY valid for ResNet34 models.
    DO NOT use with ConvNeXt-Tiny (R1, R2, R3) models which require zscore_v1 normalization!

    Returns:
        Tuple of (2_channel_array of shape (2, H, W), SolarDiskParams).
    """
    if target_family.lower() in ("convnext_tiny", "convnext", "r1", "r2", "r3"):
        raise ValueError(
            f"[DEPRECATION GUARD] preprocess_solar_image produces limbflat_v0 normalization (mean ~ 1.0) "
            f"which is strictly prohibited for {target_family} models! Use normalize_solar_intensity_zscore (zscore_v1)."
        )

    disk = detect_solar_disk_limb(
        img,
        num_rays=num_rays,
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
    )
    intensity_ch, dist_ch = correct_limb_darkening(
        img, disk, poly_degree=poly_degree, off_limb_margin=off_limb_margin
    )
    two_channel = np.stack([intensity_ch, dist_ch], axis=0).astype(np.float32)
    return two_channel, disk


def assemble_two_channel(
    intensity: np.ndarray,
    disk: SolarDiskParams,
    off_limb_margin: float = 1.01,
) -> np.ndarray:
    """
    Compute physical rho channel on the fly and assemble (2, H, W) representation.

    WHY THIS EXISTS:
    Channel 0 is the limb-darkening-corrected photospheric intensity.
    Channel 1 is physical radial distance rho = r / R in solar radii.
    Computing Channel 1 on the fly avoids storing 33.5 MB float32 arrays on disk,
    reducing cache footprint to 8.39 MB float16 per image and enabling zero-copy mmap.
    """
    h, w = intensity.shape[-2:]
    y_grid, x_grid = np.indices((h, w), dtype=np.float32)
    dist = np.sqrt((x_grid - disk.cx) ** 2 + (y_grid - disk.cy) ** 2)
    rho = (dist / disk.radius).astype(np.float32)
    if off_limb_margin is not None:
        rho[rho > off_limb_margin] = 0.0
    return np.stack([intensity.astype(np.float32), rho], axis=0)


def preprocess_and_cache(
    image_path: Union[str, Path],
    cache_dir: Union[str, Path],
    overwrite: bool = False,
    off_limb_margin: float = 1.01,
    min_radius_ratio: float = 0.80,
    max_radius_ratio: float = 0.96,
    mmap_mode: Optional[str] = "r",
) -> Tuple[np.ndarray, SolarDiskParams]:
    """
    Preprocess image and cache ONLY the intensity channel as uncompressed float16 .npy,
    loaded via np.load(mmap_mode='r'), with a small JSON sidecar for disk geometry (cx, cy, R).

    WHY THIS EXISTS:
    Uncompressed float16 .npy enables OS page-level mmap reading. Random crop extraction
    reads only the required ~512 KB slice without loading the entire 8.4 MB into RAM,
    and without decompressing 33.5 MB from .npz on every batch.
    Includes the deterministic 8-char configuration hash in both filenames.
    """
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg_hash = compute_preprocess_hash(
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
        radial_step=1.0,
        poly_degree=5,
        off_limb_margin=off_limb_margin,
    )

    intensity_file = cache_dir / f"{image_path.stem}_{cfg_hash}_intensity.npy"
    meta_file = cache_dir / f"{image_path.stem}_{cfg_hash}_meta.json"

    if intensity_file.exists() and meta_file.exists() and not overwrite:
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            disk = SolarDiskParams(
                cx=float(meta["cx"]), cy=float(meta["cy"]), radius=float(meta["radius"])
            )
            intensity = np.load(intensity_file, mmap_mode=mmap_mode)
            return intensity, disk
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # Read and process image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    disk = detect_solar_disk_limb(
        img,
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
    )
    intensity_ch, _ = correct_limb_darkening(
        img, disk, poly_degree=5, off_limb_margin=off_limb_margin
    )
    intensity_fp16 = intensity_ch.astype(np.float16)

    # Atomic write to temporary files then rename to prevent multi-worker corruption
    pid = os.getpid()
    rnd = random.randint(0, 1000000)
    tmp_intensity = cache_dir / f"{image_path.stem}_{cfg_hash}.tmp.{pid}.{rnd}_intensity.npy"
    tmp_meta = cache_dir / f"{image_path.stem}_{cfg_hash}.tmp.{pid}.{rnd}_meta.json"

    np.save(tmp_intensity, intensity_fp16)
    tmp_intensity.replace(intensity_file)

    meta_dict = {
        "cx": float(disk.cx),
        "cy": float(disk.cy),
        "radius": float(disk.radius),
        "cfg_hash": cfg_hash,
        "off_limb_margin": float(off_limb_margin),
    }
    with open(tmp_meta, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    tmp_meta.replace(meta_file)

    intensity = np.load(intensity_file, mmap_mode=mmap_mode)
    return intensity, disk


def normalize_solar_intensity_zscore(
    raw_img: np.ndarray,
    threshold: float = 20.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Standard z-score intensity normalization for solar H-alpha observations.
    Computes mean and std strictly over pixels above threshold (solar disk).
    Used by ConvNeXt-Tiny (R1, R2, R3) models.
    """
    mask = (raw_img > threshold)
    mean_val = float(np.mean(raw_img[mask])) if np.any(mask) else 128.0
    std_val = float(np.std(raw_img[mask])) if np.any(mask) else 40.0
    norm_img = (raw_img.astype(np.float32) - mean_val) / max(std_val, 1e-4)
    return norm_img, mean_val, std_val


def detect_solar_disk_min_circle(
    raw_img: np.ndarray,
    threshold: float = 20.0,
) -> SolarDiskParams:
    """
    Fast minimum enclosing circle detection for solar disk.
    Used during 3-channel context inference and training.
    """
    disk_binary = (raw_img > threshold).astype(np.uint8)
    cnts, _ = cv2.findContours(disk_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
    else:
        cx, cy, radius = 1024.0, 1024.0, 904.0
    return SolarDiskParams(cx=float(cx), cy=float(cy), radius=float(radius))


def extract_context_crop(
    norm_img: np.ndarray,
    y1: int,
    x1: int,
    tile_size: int = 512,
    context_size: int = 1536,
) -> np.ndarray:
    """
    Extract a 1536x1536 context box centered at (y1 + tile_size/2, x1 + tile_size/2)
    with zero padding at boundaries, downscaled 3x to (tile_size, tile_size) using INTER_AREA.
    Guaranteed identical between training crops and sliding-window inference tiles.
    """
    h, w = norm_img.shape
    cy_ctx = y1 + tile_size // 2
    cx_ctx = x1 + tile_size // 2
    half_ctx = context_size // 2

    y_ctx_1 = cy_ctx - half_ctx
    y_ctx_2 = y_ctx_1 + context_size
    x_ctx_1 = cx_ctx - half_ctx
    x_ctx_2 = x_ctx_1 + context_size

    context_buf = np.zeros((context_size, context_size), dtype=np.float32)
    src_y1, src_y2 = max(0, y_ctx_1), min(h, y_ctx_2)
    src_x1, src_x2 = max(0, x_ctx_1), min(w, x_ctx_2)
    dst_y1, dst_y2 = src_y1 - y_ctx_1, src_y1 - y_ctx_1 + (src_y2 - src_y1)
    dst_x1, dst_x2 = src_x1 - x_ctx_1, src_x1 - x_ctx_1 + (src_x2 - src_x1)

    context_buf[dst_y1:dst_y2, dst_x1:dst_x2] = np.array(norm_img[src_y1:src_y2, src_x1:src_x2], dtype=np.float32)
    crop_context = cv2.resize(context_buf, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    return crop_context

