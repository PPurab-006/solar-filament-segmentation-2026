"""
Full-resolution crop dataset with soft multi-annotator targets and spatial augmentations.

WHY THIS EXISTS:
Filaments are thin, elongated, high-frequency structures; downscaling 2048x2048 images
to lower resolutions obliterates delicate spine details and faint endpoints.
Working with random 512x512 full-resolution crops preserves native pixel pitch.
Because filaments occupy <1% of the solar disk area, uniform random cropping yields
mostly empty background; sampling ~70% of crops centered on annotated filaments ensures
dense gradient signals for learning filament features. Soft targets reflect inter-annotator
confidence across ambiguous boundary pixels.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from solarseg.config import Config, DatasetConfig, PathConfig
from solarseg.data import Observation, build_soft_consensus_target
from solarseg.preprocess import preprocess_and_cache


class SolarFilamentDataset(Dataset):
    """
    PyTorch Dataset yielding 2-channel full-resolution crops and soft consensus masks.

    WHY THIS EXISTS:
    Supplies training batches of size (B, 2, 512, 512) preserving native instrument resolution,
    implementing biased crop centering on filaments, D4 symmetry augmentations, and
    multi-annotator agreement labels.
    """

    def __init__(
        self,
        file_names: List[str],
        observations: Dict[str, Observation],
        paths_config: PathConfig,
        dataset_config: DatasetConfig,
        is_train: bool = True,
        samples_per_epoch: Optional[int] = None,
    ):
        """
        Args:
            file_names: List of observation file names in this split.
            observations: Parsed Observation objects grouped by file name.
            paths_config: PathConfig instance containing data and cache paths.
            dataset_config: DatasetConfig instance specifying crop sizes and probabilities.
            is_train: Whether augmentations and random cropping are enabled.
            samples_per_epoch: Virtual epoch length for random sampling (default len(file_names)).
        """
        self.file_names = list(file_names)
        self.observations = observations
        self.paths = paths_config
        self.cfg = dataset_config
        self.is_train = is_train
        self.samples_per_epoch = samples_per_epoch or len(self.file_names)

        # Pre-index filament bounding boxes for fast centered sampling
        self.filament_bboxes: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for fn in self.file_names:
            obs = self.observations.get(fn)
            bboxes = []
            if obs:
                for drawing in obs.drawings.values():
                    for ann in drawing.annotations:
                        if ann.bbox and len(ann.bbox) == 4:
                            bboxes.append(tuple(ann.bbox))
            self.filament_bboxes[fn] = bboxes

    def __len__(self) -> int:
        return self.samples_per_epoch if self.is_train else len(self.file_names)

    def _get_crop_coordinates(
        self, file_name: str, h: int = 2048, w: int = 2048
    ) -> Tuple[int, int, int, int]:
        """
        Determine (y1, y2, x1, x2) crop window.

        WHY THIS EXISTS:
        ~70% of crops are centered on known filament bboxes with mild jitter;
        ~30% are random crops on the solar disk to teach background suppression.
        """
        crop_size = self.cfg.crop_size
        bboxes = self.filament_bboxes.get(file_name, [])

        # Filament-centered crop
        if self.is_train and bboxes and random.random() < self.cfg.filament_crop_prob:
            bbox = random.choice(bboxes)
            bx, by, bw, bh = bbox
            center_x = bx + bw / 2.0
            center_y = by + bh / 2.0

            # Add random jitter up to half crop size
            jitter_x = random.uniform(-crop_size * 0.25, crop_size * 0.25)
            jitter_y = random.uniform(-crop_size * 0.25, crop_size * 0.25)

            target_cx = center_x + jitter_x
            target_cy = center_y + jitter_y

            x1 = int(round(target_cx - crop_size / 2.0))
            y1 = int(round(target_cy - crop_size / 2.0))
        else:
            # Uniform random crop
            x1 = random.randint(0, max(0, w - crop_size))
            y1 = random.randint(0, max(0, h - crop_size))

        # Clamp strictly within image bounds
        x1 = max(0, min(x1, w - crop_size))
        y1 = max(0, min(y1, h - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        return y1, y2, x1, x2

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load preprocessed 2-channel array, extract crop, apply augmentations.

        Returns:
            Tuple of:
            - image_tensor: FloatTensor of shape (2, crop_size, crop_size).
            - mask_tensor: FloatTensor of shape (1, crop_size, crop_size).
            - file_name: Source observation file name string.
        """
        if self.is_train:
            file_name = random.choice(self.file_names)
        else:
            file_name = self.file_names[idx]

        obs = self.observations.get(file_name)
        img_path = self.paths.train_images_dir / file_name

        # Load preprocessed intensity channel (mmapped float16) and disk geometry
        mmap_intensity, disk = preprocess_and_cache(
            image_path=img_path,
            cache_dir=self.paths.cache_dir,
            overwrite=False,
        )
        h, w = mmap_intensity.shape

        # Build soft multi-annotator consensus target
        if obs:
            soft_mask = build_soft_consensus_target(
                observation=obs, height=h, width=w
            )
        else:
            soft_mask = np.zeros((h, w), dtype=np.float32)

        # Apply hard label mode if requested for ablation
        if not self.cfg.soft_labels:
            soft_mask = (soft_mask >= self.cfg.hard_label_threshold).astype(np.float32)

        # Crop to crop_size
        if self.is_train:
            y1, y2, x1, x2 = self._get_crop_coordinates(
                file_name, h=h, w=w
            )
            # Zero-copy kernel slice of 512 KB directly from mmap
            crop_intensity = np.array(mmap_intensity[y1:y2, x1:x2], dtype=np.float32)
            mask_crop = soft_mask[y1:y2, x1:x2].copy()

            # Compute rho channel on the fly for the 512x512 crop
            y_coords = np.arange(y1, y2, dtype=np.float32)[:, None]
            x_coords = np.arange(x1, x2, dtype=np.float32)[None, :]
            rho_crop = np.sqrt((x_coords - disk.cx) ** 2 + (y_coords - disk.cy) ** 2) / disk.radius
            off_limb_margin = getattr(self.cfg, "off_limb_margin", 1.01)
            if off_limb_margin is not None:
                rho_crop[rho_crop > off_limb_margin] = 0.0

            image_crop = np.stack([crop_intensity, rho_crop], axis=0)

            # Apply D4 symmetry augmentations (flips + rot90)
            if random.random() < 0.5:
                # Horizontal flip
                image_crop = np.flip(image_crop, axis=2).copy()
                mask_crop = np.flip(mask_crop, axis=1).copy()

            if random.random() < 0.5:
                # Vertical flip
                image_crop = np.flip(image_crop, axis=1).copy()
                mask_crop = np.flip(mask_crop, axis=0).copy()

            k_rot = random.choice([0, 1, 2, 3])
            if k_rot > 0:
                # Rot90 along spatial axes (1, 2) for image, (0, 1) for mask
                image_crop = np.rot90(image_crop, k=k_rot, axes=(1, 2)).copy()
                mask_crop = np.rot90(mask_crop, k=k_rot, axes=(0, 1)).copy()

            # Mild brightness/contrast jitter on Channel 0 (corrected intensity) only!
            # Channel 1 is the physical distance metric in solar radii and must remain geometric.
            if self.cfg.brightness_jitter > 0:
                bright_factor = 1.0 + random.uniform(
                    -self.cfg.brightness_jitter, self.cfg.brightness_jitter
                )
                image_crop[0] *= bright_factor

            if self.cfg.contrast_jitter > 0:
                contrast_factor = 1.0 + random.uniform(
                    -self.cfg.contrast_jitter, self.cfg.contrast_jitter
                )
                mean_int = np.mean(image_crop[0])
                image_crop[0] = (image_crop[0] - mean_int) * contrast_factor + mean_int

        else:
            # During evaluation, return full image
            from solarseg.preprocess import assemble_two_channel
            image_crop = assemble_two_channel(
                mmap_intensity, disk, off_limb_margin=getattr(self.cfg, "off_limb_margin", 1.01)
            )
            mask_crop = soft_mask.copy()

        # Convert to PyTorch tensors
        tensor_image = torch.from_numpy(image_crop).float()
        tensor_mask = torch.from_numpy(mask_crop).unsqueeze(0).float()

        return tensor_image, tensor_mask, file_name
