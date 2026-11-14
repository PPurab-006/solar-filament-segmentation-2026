"""
Run-Length Encoding (RLE) utilities for COCO-style mask representation.

WHY THIS EXISTS:
Kaggle submission format specifies COCO RLE count strings for instance masks on fixed
2048x2048 grids. pycocotools expects column-major (Fortran order) arrays; incorrect
memory layout silently transposes masks. This module guarantees correct Fortran ordering,
encodes/decodes masks, verifies instance disjointness, and writes verified submission CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
try:
    from pycocotools import mask as mask_util
except ImportError:
    mask_util = None


def _numpy_rle_encode(mask: np.ndarray) -> str:
    """Pure numpy fallback for COCO RLE count encoding when pycocotools is unavailable."""
    pixels = mask.T.flatten().astype(np.uint8)
    if len(pixels) == 0:
        return ""
    changes = np.diff(pixels)
    change_indices = np.where(changes != 0)[0] + 1
    split_indices = np.concatenate(([0], change_indices, [len(pixels)]))
    lengths = np.diff(split_indices)
    counts = []
    if pixels[0] != 0:
        counts.append(0)
    counts.extend(lengths.tolist())
    return " ".join(map(str, counts))


def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    Encode a 2D binary mask into a COCO RLE count string.

    WHY THIS EXISTS:
    The competition submission schema requires only the COCO RLE counts string
    (without explicit size JSON). pycocotools requires Fortran-contiguous array layout
    for correct column-major encoding.

    Args:
        mask: 2D numpy array (bool or uint8) of shape (H, W).

    Returns:
        String of encoded RLE counts.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    if mask_util is not None:
        fortran_mask = np.asfortranarray(mask.astype(np.uint8))
        encoded = mask_util.encode(fortran_mask)
        counts = encoded["counts"]
        if isinstance(counts, bytes):
            return counts.decode("utf-8")
        return str(counts)
    else:
        return _numpy_rle_encode(mask)


def _numpy_rle_decode(counts_str: Union[str, bytes], shape: Tuple[int, int]) -> np.ndarray:
    """Pure numpy fallback for COCO RLE count decoding when pycocotools is unavailable."""
    if isinstance(counts_str, bytes):
        counts_str = counts_str.decode("utf-8")
    counts = [int(x) for x in counts_str.strip().split()]
    total_pixels = shape[0] * shape[1]
    flat = np.zeros(total_pixels, dtype=np.uint8)
    idx = 0
    val = 0
    for cnt in counts:
        if val != 0:
            flat[idx : idx + cnt] = 1
        idx += cnt
        val = 1 - val
    return flat.reshape((shape[1], shape[0])).T


def decode_rle_to_mask(
    counts: Union[str, bytes], shape: Union[Tuple[int, int], int] = (2048, 2048), *args
) -> np.ndarray:
    """
    Decode a COCO RLE count string back into a 2D binary mask.

    Accepts shape as tuple `(H, W)` or positional `(counts, H, W)`.
    """
    if len(args) == 1 and isinstance(shape, int):
        shape = (shape, args[0])
    elif isinstance(shape, int) and len(args) == 0:
        shape = (shape, shape)

    if mask_util is not None:
        if isinstance(counts, str):
            counts_bytes = counts.encode("utf-8")
        else:
            counts_bytes = counts
        rle_dict = {"size": [shape[0], shape[1]], "counts": counts_bytes}
        decoded = mask_util.decode(rle_dict)
        return decoded.astype(np.uint8)
    else:
        return _numpy_rle_decode(counts, shape)


def verify_disjoint_masks(masks: List[np.ndarray]) -> bool:
    """
    Check that all masks in a collection are mutually disjoint.

    WHY THIS EXISTS:
    Hard rule 2 dictates that predicted instances cannot overlap. Any pixel
    shared between multiple instances will trigger submission or metric errors.

    Args:
        masks: List of 2D binary numpy arrays.

    Returns:
        True if all masks have zero mutual overlap, False otherwise.
    """
    if len(masks) <= 1:
        return True

    # Accumulate instance indices
    combined = np.zeros_like(masks[0], dtype=np.uint16)
    for mask in masks:
        combined += (mask > 0).astype(np.uint16)
        if np.any(combined > 1):
            return False
    return True


def write_submission(
    csv_path: Union[str, Path],
    predictions: Dict[str, List[np.ndarray]],
    mask_shape: Tuple[int, int] = (2048, 2048),
) -> int:
    """
    Write predicted instance masks to a Kaggle-compliant submission CSV.

    WHY THIS EXISTS:
    Formats predicted instances into the exact challenge format:
    columns: filament_id, segmentation_rle
    filament_id: <imagename-without-extension>_<k>
    Guarantees no empty masks and enforces instance disjointness.

    Args:
        csv_path: Target CSV file path.
        predictions: Mapping from image name (with or without extension) to list of masks.
        mask_shape: Expected mask dimensions (default 2048, 2048).

    Returns:
        Total number of written filament rows.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("filament_id,segmentation_rle\n")

        for img_name, masks in sorted(predictions.items()):
            # Strip file extension (.jpeg, .jpg, etc.)
            base_name = Path(img_name).stem

            # Verify non-empty and disjoint instances
            valid_masks = [m for m in masks if m is not None and np.count_nonzero(m) > 0]
            if not verify_disjoint_masks(valid_masks):
                raise ValueError(
                    f"Predicted masks for image '{base_name}' contain overlapping instances!"
                )

            for k, mask in enumerate(valid_masks):
                if mask.shape != mask_shape:
                    raise ValueError(
                        f"Mask for {base_name}_{k} has shape {mask.shape}, expected {mask_shape}"
                    )
                rle_str = encode_mask_to_rle(mask)
                f.write(f"{base_name}_{k},{rle_str}\n")
                rows_written += 1

    return rows_written
