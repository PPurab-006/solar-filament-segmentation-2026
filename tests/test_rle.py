"""
Unit tests for RLE encoding/decoding and instance disjointness.

WHY THIS EXISTS:
Tests Hard Rule 3: pycocotools encodes masks column-major (Fortran order).
An asymmetric mask ensures that transpose / row-major bugs are immediately detected.
"""

import numpy as np
import pytest

from solarseg.rle import decode_rle_to_mask, encode_mask_to_rle, verify_disjoint_masks


def test_asymmetric_mask_rle_roundtrip():
    """Verify that an asymmetric mask round-trips through encode -> decode without transpose."""
    shape = (2048, 2048)
    mask = np.zeros(shape, dtype=np.uint8)

    # Draw an explicitly asymmetric L-shape / rectangular pattern
    # Height != Width, y-range != x-range
    mask[200:500, 800:900] = 1   # Vertical bar of 300 x 100
    mask[450:500, 800:1400] = 1  # Horizontal bar of 50 x 600

    # Ensure mask is strictly asymmetric across transpose
    assert not np.array_equal(mask, mask.T), "Test fixture mask must be asymmetric!"

    # Encode with column-major enforcement
    rle_str = encode_mask_to_rle(mask)
    assert isinstance(rle_str, str)
    assert len(rle_str) > 0

    # Decode back with tuple shape
    decoded = decode_rle_to_mask(rle_str, shape=shape)
    assert np.array_equal(mask, decoded), "Decoded mask does not match original asymmetric mask!"

    # Decode back with 3-arg positional shape (counts, 2048, 2048) to prevent regression
    decoded_3arg = decode_rle_to_mask(rle_str, 2048, 2048)
    assert np.array_equal(mask, decoded_3arg), "3-arg positional decoded mask does not match original!"


def test_verify_disjoint_masks():
    """Verify detection of overlapping vs mutually disjoint masks."""
    shape = (2048, 2048)
    mask1 = np.zeros(shape, dtype=np.uint8)
    mask2 = np.zeros(shape, dtype=np.uint8)
    mask3 = np.zeros(shape, dtype=np.uint8)

    mask1[100:200, 100:200] = 1
    mask2[300:400, 300:400] = 1
    mask3[150:250, 150:250] = 1  # Overlaps with mask1

    # Disjoint pair
    assert verify_disjoint_masks([mask1, mask2]) is True

    # Overlapping pair
    assert verify_disjoint_masks([mask1, mask3]) is False
    assert verify_disjoint_masks([mask1, mask2, mask3]) is False
