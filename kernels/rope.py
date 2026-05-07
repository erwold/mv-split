# This file is derived from Unsloth (https://github.com/unslothai/unsloth),
#   unsloth/kernels/rope_embedding.py
#   Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Modifications relative to the Unsloth original:
#   - redesigned the kernel to use 4-D strided I/O over (Batch, Head, Seq, Dim)
#     with a fresh contiguous output buffer (Unsloth's design uses row-major
#     in-place writes with an inner ROPE_GROUP_SIZE=4 head loop),
#   - removed rope_embedding_indices, the joint QK kernel, and the slow /
#     in-place fallbacks.

"""Zero-copy strided RoPE (Rotary Position Embedding) Triton kernel.

Design note: this implementation uses 4-D strided indexing over
(Batch, Head, Seq, Dim) and allocates a fresh contiguous output, distinct
from Unsloth's row-major + group-iterated in-place design.
"""

import torch
import triton
import triton.language as tl

from kernels._common import calculate_settings


@triton.jit
def _rope_embedding_strided_io(
    # output pointers & strides
    ptr_O, stride_b_O, stride_h_O, stride_s_O, stride_d_O,
    # input pointers & strides
    ptr_I, stride_b_I, stride_h_I, stride_s_I, stride_d_I,
    # cos/sin (shape [L, D/2])
    ptr_cos, stride_s_cos, stride_d_cos,
    ptr_sin, stride_s_sin, stride_d_sin,
    head_dim: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Strided RoPE kernel supporting arbitrary input layouts.
    Grid: (Seq, Head, Batch) -> (s_idx, h_idx, b_idx)
    """
    s_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    b_idx = tl.program_id(2)

    in_off = b_idx * stride_b_I + h_idx * stride_h_I + s_idx * stride_s_I
    out_off = b_idx * stride_b_O + h_idx * stride_h_O + s_idx * stride_s_O

    cs_off = s_idx * stride_s_cos
    ss_off = s_idx * stride_s_sin

    HALF = head_dim // 2
    col = tl.arange(0, BLOCK_SIZE)
    mask = col < HALF

    cos = tl.load(ptr_cos + cs_off + col * stride_d_cos, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(ptr_sin + ss_off + col * stride_d_sin, mask=mask, other=0.0).to(tl.float32)

    # Backward pass: reverse the rotation by negating sin.
    if BACKWARD_PASS:
        sin = -sin

    x1 = tl.load(ptr_I + in_off + col * stride_d_I, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptr_I + in_off + (col + HALF) * stride_d_I, mask=mask, other=0.0).to(tl.float32)

    # y1 = x1*cos + x2*sin, y2 = x2*cos - x1*sin
    y1 = x1 * cos + x2 * sin
    y2 = x2 * cos - x1 * sin

    tl.store(ptr_O + out_off + col * stride_d_O, y1, mask=mask)
    tl.store(ptr_O + out_off + (col + HALF) * stride_d_O, y2, mask=mask)


class Fast_RoPE_Strided(torch.autograd.Function):
    """Autograd Function for strided RoPE supporting arbitrary input layouts."""
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Q: [B, H, L, D] (any layout — strides handle it).
        D = Q.shape[-1]
        if (D % 2) != 0:
            raise RuntimeError(f"RoPE head_dim must be even, got {D}")

        cos_v = cos.view(-1, cos.shape[-1])
        sin_v = sin.view(-1, sin.shape[-1])

        L = Q.shape[2]

        if cos_v.shape[0] < L or sin_v.shape[0] < L:
            raise RuntimeError(
                f"RoPE cos/sin length < seq_len: cos={cos_v.shape[0]}, "
                f"sin={sin_v.shape[0]}, L={L}"
            )
        # Pre-computed cos/sin may be longer than L; slice down.
        if cos_v.shape[0] != L:
            cos_v = cos_v[:L]
        if sin_v.shape[0] != L:
            sin_v = sin_v[:L]

        BLOCK_SIZE, num_warps = calculate_settings(D // 2)
        if BLOCK_SIZE is None:
            raise RuntimeError(f"RoPE head_dim/2 too large for Triton kernel: {D//2}")

        # Allocate a fresh contiguous output so downstream ops (SDPA / FlashAttn) get a contiguous tensor.
        Q_out = torch.empty(Q.shape, device=Q.device, dtype=Q.dtype)

        B, H, L, _ = Q.shape
        grid = (L, H, B)

        _rope_embedding_strided_io[grid](
            Q_out, Q_out.stride(0), Q_out.stride(1), Q_out.stride(2), Q_out.stride(3),
            Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            cos_v, cos_v.stride(0), cos_v.stride(1),
            sin_v, sin_v.stride(0), sin_v.stride(1),
            head_dim=D,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        ctx.save_for_backward(cos_v, sin_v)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.input_dtype = Q.dtype
        return Q_out

    @staticmethod
    def backward(ctx, dY):
        cos, sin = ctx.saved_tensors
        D = dY.shape[-1]

        # dQ must be a fresh allocation (cannot reuse dY storage).
        dQ = torch.empty(dY.shape, device=dY.device, dtype=dY.dtype)

        B, H, L, _ = dY.shape
        grid = (L, H, B)

        _rope_embedding_strided_io[grid](
            dQ, dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
            dY, dY.stride(0), dY.stride(1), dY.stride(2), dY.stride(3),
            cos, cos.stride(0), cos.stride(1),
            sin, sin.stride(0), sin.stride(1),
            head_dim=D,
            BACKWARD_PASS=True,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        return dQ, None, None


def fast_rope_embedding(Q, K, cos, sin):
    """Apply RoPE to Q and K. The fallback path lives outside the autograd Function so gradients flow correctly."""
    D = Q.shape[-1]
    input_dtype = Q.dtype

    # Fallback for very large head dimensions that exceed the Triton block size.
    if D // 2 > 4096:
        def manual_rope(x, c, s):
            d = x.shape[-1] // 2
            x1, x2 = x[..., :d], x[..., d:]
            out = torch.cat([x1 * c + x2 * s, x2 * c - x1 * s], -1)
            return out.to(input_dtype)
        return manual_rope(Q, cos, sin), manual_rope(K, cos, sin)

    Q_out = Fast_RoPE_Strided.apply(Q, cos, sin)
    K_out = Fast_RoPE_Strided.apply(K, cos, sin)
    return Q_out, K_out
