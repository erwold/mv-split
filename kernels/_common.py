"""Shared utilities for the Triton kernels in this package."""

import triton


def calculate_settings(n):
    """Pick BLOCK_SIZE and num_warps for a reduction over `n` elements."""
    # Larger block sizes hit a Triton compilation limit; caller should fall back.
    if n > 8192:
        return None, None
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128

    # Cap warps at 8 for reduction kernels — 16 warps can cause register spill.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8

    return BLOCK_SIZE, num_warps
