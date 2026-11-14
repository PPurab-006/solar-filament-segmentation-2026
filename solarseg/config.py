"""
Configuration management for solar filament segmentation pipeline.

WHY THIS EXISTS:
Decouples file paths, preprocessing settings, hyperparameter choices, and hardware
options from executable code. Supports Colab (T4 GPU), Kaggle notebooks, and local
workstations through dataclasses with environment variable fallbacks, preventing
path hardcoding and enabling modular pipeline orchestration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """
    Filesystem path configurations with environment variable overrides.

    WHY THIS EXISTS:
    Allows seamless execution across Colab, Kaggle kernels, and local environments
    without modifying code. All paths are resolved relative to configurable base roots.
    """
    root_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SOLARSEG_ROOT", "."))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SOLARSEG_DATA_DIR",
                os.path.join(
                    os.environ.get("SOLARSEG_ROOT", "."),
                    "MAGFiLO_1.0_Kaggle_2026",
                ),
            )
        )
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SOLARSEG_CACHE_DIR",
                os.path.join(
                    os.environ.get("SOLARSEG_ROOT", "."),
                    "cache/preprocessed",
                ),
            )
        )
    )
    splits_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SOLARSEG_SPLITS_DIR",
                os.path.join(
                    os.environ.get("SOLARSEG_ROOT", "."),
                    "data/splits",
                ),
            )
        )
    )
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "SOLARSEG_OUTPUT_DIR",
                os.path.join(os.environ.get("SOLARSEG_ROOT", "."), "outputs"),
            )
        )
    )

    _train_images_dir: Optional[Path] = None
    _train_json_path: Optional[Path] = None
    _test_images_dir: Optional[Path] = None
    _split_json_path: Optional[Path] = None

    @property
    def train_images_dir(self) -> Path:
        return self._train_images_dir if self._train_images_dir is not None else (self.data_dir / "train" / "train_images")

    @train_images_dir.setter
    def train_images_dir(self, val: Path) -> None:
        self._train_images_dir = Path(val)

    @property
    def train_json_path(self) -> Path:
        return self._train_json_path if self._train_json_path is not None else (self.data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

    @train_json_path.setter
    def train_json_path(self, val: Path) -> None:
        self._train_json_path = Path(val)

    @property
    def test_images_dir(self) -> Path:
        return self._test_images_dir if self._test_images_dir is not None else (self.data_dir / "test" / "test_images")

    @test_images_dir.setter
    def test_images_dir(self, val: Path) -> None:
        self._test_images_dir = Path(val)

    @property
    def split_json_path(self) -> Path:
        return self._split_json_path if self._split_json_path is not None else (self.splits_dir / "split.json")

    @split_json_path.setter
    def split_json_path(self, val: Path) -> None:
        self._split_json_path = Path(val)

    _checkpoint_dir: Optional[Path] = None

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir if self._checkpoint_dir is not None else (self.output_dir / "checkpoints")

    @checkpoint_dir.setter
    def checkpoint_dir(self, val: Path) -> None:
        self._checkpoint_dir = Path(val)

    def ensure_directories(self) -> None:
        """Create required output and cache directories if missing."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessConfig:
    """
    Solar disk limb detection and limb-darkening correction parameters.

    WHY THIS EXISTS:
    Solar H-alpha images suffer from solar limb darkening (center-to-limb intensity
    falloff) and atmospheric halo effects. Standard thresholding produces oversized disks.
    This config parameterizes sharpest-radial-edge limb fitting and polynomial
    radial normalization into a 2-channel representation.
    """
    num_rays: int = 360  # Radial search rays around the center
    min_radius_ratio: float = 0.80  # Search lower bound (0.80 * 1024 = 819.2 px)
    max_radius_ratio: float = 0.96  # Search upper bound (0.96 * 1024 = 983.0 px, true limb ~904 px)
    radial_step: float = 1.0  # Pixel step size along each ray
    poly_degree: int = 5  # Polynomial degree for limb darkening profile fit
    intensity_norm_center: float = 1.0  # Normalized target intensity at solar disk center
    off_limb_margin: float = 1.01  # Simplified margin beyond disk limb (1.01 = +1% margin)
    cache_format: str = "npz"  # Fast compressed disk cache format ('npz' or 'npy')


@dataclass
class DatasetConfig:
    """
    Full-resolution cropping and augmentation parameters.

    WHY THIS EXISTS:
    Filaments are long, thin, high-frequency structures; downscaling 2048x2048 images
    destroys thin spine features. Random 512x512 crops preserve resolution, while
    biased sampling (70% filament-centered) combats severe class imbalance.
    """
    crop_size: int = 512
    filament_crop_prob: float = 0.70  # Probability of centering crop on a filament
    soft_labels: bool = True  # Target is mean annotator agreement [0, 1]
    hard_label_threshold: float = 0.5  # Binarization threshold when soft_labels=False
    brightness_jitter: float = 0.15  # Mild brightness perturbation
    contrast_jitter: float = 0.15  # Mild contrast perturbation


@dataclass
class ModelConfig:
    """
    Architecture and encoder settings.

    WHY THIS EXISTS:
    Specifies U-Net backbone with 2-channel input (limb-corrected intensity + solar
    radius coordinate). Keeping encoder configurable allows rapid experimentation
    across ResNet34, EfficientNet, or ConvNeXt.
    """
    architecture: str = "Unet"
    encoder_name: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    in_channels: int = 2  # Channel 0: corrected intensity, Channel 1: solar radius
    classes: int = 1  # Binary segmentation logit

    # Auxiliary structural training targets (Stage 2c Task 5, default OFF)
    aux_skeleton: bool = False
    aux_centerline: bool = False  # Auxiliary centerline head
    aux_distance_transform: bool = False
    aux_boundary: bool = False  # Auxiliary boundary/distance-transform head
    weight_skeleton_loss: float = 0.5
    weight_centerline_loss: float = 0.5
    weight_distance_loss: float = 0.5
    weight_boundary_loss: float = 0.5
    centerline_target_type: str = "skeleton"  # "skeleton" (mask-derived) or "spine" (real annotated spine)
    skeleton_dilation: int = 1  # Dilation radius in pixels for centerline target
    skeleton_loss_type: str = "recall"  # "recall" (Skeleton Recall loss) or "dice"

    # Barb-weighted loss parameters (Task 5c, default OFF)
    barb_weighting: bool = False
    barb_beta: float = 2.0  # Multiplier beta in w = 1 + beta * (mask AND NOT dilate(spine, r))
    barb_radius: int = 6  # Dilation radius r around the spine


@dataclass
class TrainingConfig:
    """
    Optimization and schedule hyperparameters.

    WHY THIS EXISTS:
    Calibrated for 16GB VRAM (T4 on Colab or RTX 3050 locally). Uses AdamW,
    one-cycle schedule, mixed precision, and dual checkpointing (best val and last epoch).
    crops_per_image parameterizes how many random 512x512 crops are drawn per training image
    each epoch.
    """
    batch_size: int = 8
    num_workers: int = field(default_factory=lambda: min(4, os.cpu_count() or 1))
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 20
    crops_per_image: float = 2.0  # Multiplier: 2.0 * 566 = 1132 crops/epoch (~20 crops/image per 10 ep)
    train_minutes: Optional[float] = None  # Dynamic time budget in minutes (overrides crops_per_image if set)
    mixed_precision: bool = True
    seed: int = 42
    loss_name: str = "bce_dice"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    device: str = "cuda"  # Auto-falls back to cpu if cuda unavailable


@dataclass
class InferenceConfig:
    """
    Sliding-window inference and test-time augmentation parameters.

    WHY THIS EXISTS:
    Full-resolution 2048x2048 inference requires tile-based execution to avoid tile-boundary
    artifacts. Blending with a 2D Hann window gives zero-weight border transitions.
    tta_mode: 1 = none, 4 = flips only (H/V), 8 = full D4 dihedral group.
    """
    tile_size: int = 512
    stride: int = 256  # 50% overlap for seamless Hann blending
    blend_mode: str = "hann"  # 2D Hann window blending
    tta: bool = False  # Backward-compatible toggle
    tta_mode: int = 1  # 1 = single pass, 4 = flips only, 8 = full D4


@dataclass
class PostprocessConfig:
    """
    Morphological post-processing parameters.

    WHY THIS EXISTS:
    Thresholds continuous model probabilities into discrete binary instances.
    Small gap closing connects fractured filaments; hole filling cleans interiors;
    connected components isolates disjoint instances; min_area drops spurious noise.
    clip_margin restricts instance generation to the valid solar disk + margin.
    max_rho optionally masks predictions at high heliocentric angles (e.g. 0.95).
    """
    threshold: float = 0.50
    min_area: int = 100
    close_radius: int = 3
    bridge_gap: int = 0
    clip_margin: float = 1.01
    max_rho: Optional[float] = None
    peak_threshold: Optional[float] = None


@dataclass
class Config:
    """
    Master configuration combining all pipeline subsystems.

    WHY THIS EXISTS:
    Provides a single unified access point for training, validation, inference,
    and post-processing tuning.
    """
    paths: PathConfig = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    def to_dict(self) -> Dict:
        """Serialize configuration to a clean dictionary of primitive types (no Path objects)."""
        import dataclasses
        def _convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(v) for v in obj]
            return obj
        return _convert(self)


PipelineConfig = Config

