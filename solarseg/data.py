"""
Data parsing and ground truth extraction for MAGFiLO dataset.

WHY THIS EXISTS:
In MAGFiLO, up to three annotators independently marked solar filaments on the same
physical observation image. In the COCO JSON, each annotator session is given a unique
'image_id' (e.g. '010101-20160920230134Lh') sharing the same underlying 'file_name'.
This module groups annotations by physical observation (file_name), rasterizes polygons
into per-drawing instance masks for Panoptic Quality evaluation, computes soft multi-annotator
consensus masks for model training, and provides test image discovery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class FilamentAnnotation:
    """
    Single filament polygon annotation.

    WHY THIS EXISTS:
    Stores polygon vertices, bounding box, area, and category for an individual filament.
    """
    id: str
    category_id: int
    segmentation: List[List[float]]
    bbox: List[float]
    area: float


@dataclass
class AnnotatorDrawing:
    """
    Full set of filament annotations created by one annotator for one observation.

    WHY THIS EXISTS:
    Evaluation requires scoring model predictions against each annotator drawing
    separately before pooling.
    """
    drawing_id: str  # e.g., '010101-20160920230134Lh'
    file_name: str  # e.g., '20160920230134Lh.jpeg'
    date_captured: str
    height: int
    width: int
    annotations: List[FilamentAnnotation] = field(default_factory=list)


@dataclass
class Observation:
    """
    Physical solar observation grouping all annotator drawings.

    WHY THIS EXISTS:
    Prevents data leakage during splits (all annotator drawings of the same observation
    must stay in the same split) and enables soft consensus target generation.
    """
    file_name: str
    date_captured: str
    height: int
    width: int
    drawings: Dict[str, AnnotatorDrawing] = field(default_factory=dict)

    @property
    def observation_date(self) -> str:
        """Extract YYYY-MM-DD from file_name (format: YYYYMMDDHHMMSS<site>.jpeg)."""
        stem = Path(self.file_name).stem
        if len(stem) >= 8 and stem[:8].isdigit():
            return f"{stem[:4]}-{stem[4:6]}-{stem[6:8]}"
        return self.date_captured.split(" ")[0] if self.date_captured else "unknown"

    @property
    def stem(self) -> str:
        """Extract filename stem without extension."""
        return Path(self.file_name).stem

    @property
    def site_code(self) -> str:
        """Extract 2-letter GONG observatory site code from file name."""
        stem = Path(self.file_name).stem
        if len(stem) >= 16:
            return stem[14:16]
        return stem[-2:]



from pycocotools import mask as mask_util


def rasterize_polygon(
    segmentation: List[List[float]], height: int = 2048, width: int = 2048
) -> np.ndarray:
    """
    Rasterize COCO polygon coordinate lists into a binary uint8 mask using pycocotools.

    WHY THIS EXISTS:
    Converts vector polygon outlines into standard COCO 2D pixel masks for exact IoU computation,
    matching annotation areas in JSON exactly.
    """
    valid_polys = [p for p in segmentation if len(p) >= 6]
    if not valid_polys:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_util.frPyObjects(valid_polys, height, width)
    decoded = mask_util.decode(rles)
    if decoded.ndim == 3:
        mask = np.any(decoded, axis=2).astype(np.uint8)
    else:
        mask = decoded.astype(np.uint8)
    return mask


def load_grayscale_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an 8-bit grayscale solar image.

    WHY THIS EXISTS:
    Competition rule: Solar H-alpha images are 8-bit grayscale (2048x2048) and must
    NEVER be treated as RGB. Using cv2.IMREAD_GRAYSCALE prevents 3-channel memory bloat.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Solar image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image as grayscale: {image_path}")
    if img.shape != (2048, 2048):
        # Resize or pad if necessary, but standard competition images are 2048x2048
        pass
    return img


def parse_coco_annotations(json_path: Union[str, Path]) -> Dict[str, Observation]:
    """
    Parse MAGFiLO COCO JSON and group drawings by physical observation.

    WHY THIS EXISTS:
    Restructures flat COCO JSON records into an observation-centric hierarchy.
    Enables leak-free splitting and multi-annotator consensus modeling.

    Args:
        json_path: Path to MAGFiLO_1.0_Annotations_kaggle2026_train.json.

    Returns:
        Dict mapping observation file_name to Observation objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Index drawings by drawing_id and map to observation file_name
    drawing_to_file: Dict[str, str] = {}
    observations: Dict[str, Observation] = {}

    for img_info in data.get("images", []):
        d_id = str(img_info["id"])
        f_name = str(img_info["file_name"])
        drawing_to_file[d_id] = f_name

        if f_name not in observations:
            observations[f_name] = Observation(
                file_name=f_name,
                date_captured=str(img_info.get("date_captured", "")),
                height=int(img_info.get("height", 2048)),
                width=int(img_info.get("width", 2048)),
                drawings={},
            )

        observations[f_name].drawings[d_id] = AnnotatorDrawing(
            drawing_id=d_id,
            file_name=f_name,
            date_captured=str(img_info.get("date_captured", "")),
            height=int(img_info.get("height", 2048)),
            width=int(img_info.get("width", 2048)),
            annotations=[],
        )

    # Attach annotations to corresponding drawings
    for ann in data.get("annotations", []):
        d_id = str(ann["image_id"])
        if d_id in drawing_to_file:
            f_name = drawing_to_file[d_id]
            obs = observations[f_name]
            drawing = obs.drawings[d_id]

            # Derive bbox and area strictly from segmentation polygon (Hard Rule 4 / A3)
            # Never access ann["bbox"], ann["area"], ann["spine"], or keypoints.
            seg = ann.get("segmentation", [])
            xs: List[float] = []
            ys: List[float] = []
            for poly in seg:
                if len(poly) >= 2:
                    xs.extend(poly[0::2])
                    ys.extend(poly[1::2])
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                derived_bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
                derived_area = float((max_x - min_x) * (max_y - min_y))  # coarse polygon bbox area
            else:
                derived_bbox = [0.0, 0.0, 0.0, 0.0]
                derived_area = 0.0

            filament_ann = FilamentAnnotation(
                id=str(ann.get("id", "")),
                category_id=int(ann.get("category_id", 1)),
                segmentation=seg,
                bbox=derived_bbox,
                area=derived_area,
            )
            drawing.annotations.append(filament_ann)

    return observations


def build_drawing_masks(
    drawing: AnnotatorDrawing, height: int = 2048, width: int = 2048
) -> List[np.ndarray]:
    """
    Rasterize all filament polygons for a single annotator drawing into separate instance masks.

    WHY THIS EXISTS:
    PQ metric evaluation requires individual instance masks per annotator drawing.
    """
    instance_masks: List[np.ndarray] = []
    for ann in drawing.annotations:
        mask = rasterize_polygon(ann.segmentation, height=height, width=width)
        if np.count_nonzero(mask) > 0:
            instance_masks.append(mask)
    return instance_masks


def build_soft_consensus_target(
    observation: Observation, height: int = 2048, width: int = 2048
) -> np.ndarray:
    """
    Compute soft consensus label as per-pixel fraction of annotators who marked it.

    WHY THIS EXISTS:
    Different annotators have slight disagreements on filament boundary thickness
    and subtle faint tails. Soft continuous labels [0, 1] represent annotation confidence
    and prevent the network from penalizing ambiguous borderline boundaries.

    Returns:
        float32 array of shape (height, width) with values in [0.0, 1.0].
    """
    drawings = list(observation.drawings.values())
    num_drawings = len(drawings)
    if num_drawings == 0:
        return np.zeros((height, width), dtype=np.float32)

    soft_target = np.zeros((height, width), dtype=np.float32)
    for d in drawings:
        d_mask = np.zeros((height, width), dtype=np.uint8)
        for ann in d.annotations:
            d_mask |= rasterize_polygon(ann.segmentation, height=height, width=width)
        soft_target += (d_mask > 0).astype(np.float32)

    soft_target /= float(num_drawings)
    return soft_target


def list_test_images(test_dir: Union[str, Path]) -> List[Path]:
    """
    Find and return sorted list of test image file paths.

    WHY THIS EXISTS:
    Provides standardized discovery of unlabeled test images for inference.
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
