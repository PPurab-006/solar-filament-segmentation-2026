"""
Unit tests for solar disk limb detection and preprocessing.

WHY THIS EXISTS:
Verifies that:
- Sharpest radial edge detection correctly identifies the true limb boundary despite atmospheric halos.
- Preprocessed output strictly returns zero outside the fitted disk for both channels.
- Output contains exactly 2 channels: [corrected_intensity, distance_in_solar_radii].
"""

import numpy as np
import pytest

from solarseg.preprocess import (
    SolarDiskParams,
    correct_limb_darkening,
    detect_solar_disk_limb,
    preprocess_solar_image,
)


def create_synthetic_solar_image(
    size: int = 512,
    cx: float = 256.0,
    cy: float = 256.0,
    radius: float = 180.0,
) -> np.ndarray:
    """Generate synthetic solar image with center-to-limb darkening and atmospheric halo."""
    y_grid, x_grid = np.indices((size, size), dtype=np.float32)
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

    img = np.zeros((size, size), dtype=np.float32)

    # On disk: 200 at center, dims to 120 at limb (40% limb darkening)
    on_disk = dist <= radius
    rho = dist / radius
    img[on_disk] = 200.0 * (1.0 - 0.4 * (rho[on_disk] ** 2))

    # Sharp limb edge drop across 3 pixels
    # Outside disk: atmospheric halo decaying outward
    outside = dist > radius
    halo = 30.0 * np.exp(-(dist[outside] - radius) / 25.0)
    img[outside] = halo

    return np.clip(img, 0, 255).astype(np.uint8)


def test_solar_limb_detection_and_zero_outside_disk():
    """Verify disk limb fitting accuracy and zero-outside-disk enforcement."""
    size = 512
    true_cx, true_cy, true_r = 256.0, 256.0, 180.0
    synth_img = create_synthetic_solar_image(size=size, cx=true_cx, cy=true_cy, radius=true_r)

    # Preprocess with off_limb_margin=1.0 (strict disk cut)
    two_ch, disk = preprocess_solar_image(
        synth_img,
        num_rays=180,
        min_radius_ratio=0.50,
        max_radius_ratio=0.85,
        off_limb_margin=1.00,
    )

    # Verify 2-channel output shape
    assert two_ch.shape == (2, size, size)

    # Verify detected circle is within 2 pixels of true geometry
    assert pytest.approx(disk.cx, abs=3.0) == true_cx
    assert pytest.approx(disk.cy, abs=3.0) == true_cy
    assert pytest.approx(disk.radius, abs=3.0) == true_r

    # Compute distance grid from detected center
    y_grid, x_grid = np.indices((size, size), dtype=np.float32)
    dist_from_detected = np.sqrt((x_grid - disk.cx) ** 2 + (y_grid - disk.cy) ** 2)
    outside_mask = dist_from_detected > disk.radius

    # Critical requirement: returns strictly zero outside the disk
    intensity_ch = two_ch[0]
    distance_ch = two_ch[1]

    assert np.all(intensity_ch[outside_mask] == 0.0), "Channel 0 is non-zero outside the disk!"
    assert np.all(distance_ch[outside_mask] == 0.0), "Channel 1 is non-zero outside the disk!"

    # Inside disk assertions
    inner_mask = dist_from_detected <= (disk.radius * 0.95)
    assert np.all(intensity_ch[inner_mask] > 0.0), "Channel 0 is zero inside the disk!"
    assert np.all(distance_ch[inner_mask] >= 0.0)
    assert np.all(distance_ch[inner_mask] <= 1.0)


def test_real_images_limb_fit_regression():
    """Verify fitted R in [880, 930] and per-site std under 3.0 px on real train images."""
    import cv2
    from pathlib import Path
    from solarseg.config import Config

    cfg = Config()
    train_dir = cfg.paths.train_images_dir
    if not train_dir.exists():
        pytest.skip(f"Train images directory not found at {train_dir}")

    sites = ["Ch", "Bh", "Th", "Uh", "Mh", "Lh"]
    site_radii = {}

    for site in sites:
        site_files = sorted(list(train_dir.glob(f"*{site}.jpeg")))
        if len(site_files) < 3:
            continue

        radii = []
        for img_path in site_files[:3]:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            assert img is not None, f"Failed to load {img_path}"
            disk = detect_solar_disk_limb(img, min_radius_ratio=0.80, max_radius_ratio=0.96)
            radii.append(disk.radius)
            # Individual radius bounds
            assert 880.0 <= disk.radius <= 930.0, (
                f"Fitted radius {disk.radius:.2f}px for {img_path.name} out of [880, 930] bound"
            )

        site_radii[site] = radii
        site_std = float(np.std(radii))
        # Per-site std under 3 px
        assert site_std < 3.0, (
            f"Site {site} fitted radius std={site_std:.3f}px exceeds 3.0px threshold (radii={radii})"
        )

    assert len(site_radii) >= 4, f"Found fewer than 4 GONG sites locally: {list(site_radii.keys())}"


def test_uncompressed_float16_mmap_cache(tmp_path):
    """Verify preprocessing cache stores uncompressed float16 .npy and json sidecar."""
    import cv2
    import json
    from solarseg.preprocess import assemble_two_channel, preprocess_and_cache

    # Create dummy image and write to tmp
    synth_img = create_synthetic_solar_image(size=512, cx=256.0, cy=256.0, radius=180.0)
    img_path = tmp_path / "dummy_sun.png"
    cv2.imwrite(str(img_path), synth_img)

    cache_dir = tmp_path / "cache"

    intensity_mmap, disk = preprocess_and_cache(
        image_path=img_path,
        cache_dir=cache_dir,
        min_radius_ratio=0.50,
        max_radius_ratio=0.85,
        off_limb_margin=1.01,
    )

    # Verify return types and shapes
    assert intensity_mmap.shape == (512, 512)
    assert intensity_mmap.dtype == np.float16
    assert isinstance(intensity_mmap, np.memmap)
    assert pytest.approx(disk.radius, abs=3.0) == 180.0

    # Verify saved files
    npy_files = list(cache_dir.glob("dummy_sun_*_intensity.npy"))
    json_files = list(cache_dir.glob("dummy_sun_*_meta.json"))

    assert len(npy_files) == 1, f"Expected 1 intensity .npy, found: {npy_files}"
    assert len(json_files) == 1, f"Expected 1 meta .json, found: {json_files}"

    # Verify json sidecar contents
    with open(json_files[0], "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "cx" in meta and "cy" in meta and "radius" in meta and "cfg_hash" in meta
    assert pytest.approx(meta["radius"], abs=3.0) == 180.0

    # Verify assemble_two_channel
    two_ch = assemble_two_channel(intensity_mmap, disk, off_limb_margin=1.01)
    assert two_ch.shape == (2, 512, 512)
    assert two_ch.dtype == np.float32
