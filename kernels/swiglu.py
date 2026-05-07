# This file is derived from Unsloth (https://github.com/unslothai/unsloth),
#   unsloth/kernels/swiglu.py
#   Copyright 2023-present Daniel Han-Chen & the Unsloth team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications relative to the Unsloth original:
#   - reworked the API to operate on a single packed [..., 2D] gate+value tensor
#     (Unsloth takes separate `e` and `g` tensors),
#   - wrapped the kernel in a torch.autograd.Function and added a PyTorch
#     fallback for dimensions outside the Triton-supported range.

"""Packed SwiGLU activation Triton kernel."""

import torch
import triton
import triton.language as tl

from kernels._common import calculate_settings


@triton.jit
def _swiglu_packed_fwd_kernel(
    Y_ptr, stride_y_row,
    X_ptr, stride_x_row,
    dim, BLOCK_SIZE: tl.constexpr
):
    """SwiGLU forward: output = silu(gate) * val where [gate, val] = x.chunk(2)"""
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    x_row_start = X_ptr + row_idx * stride_x_row
    y_row_start = Y_ptr + row_idx * stride_y_row

    # Fused load of gate and val
    gate = tl.load(x_row_start + offs, mask=mask, other=0.0).to(tl.float32)
    val = tl.load(x_row_start + dim + offs, mask=mask, other=0.0).to(tl.float32)

    # SwiGLU: (gate * sigmoid(gate)) * val = silu(gate) * val
    res = (gate * tl.sigmoid(gate)) * val
    tl.store(y_row_start + offs, res, mask=mask)


@triton.jit
def _swiglu_packed_bwd_kernel(
    dX_ptr, stride_dx_row,
    dY_ptr, stride_dy_row,
    X_ptr, stride_x_row,
    dim, BLOCK_SIZE: tl.constexpr
):
    """SwiGLU backward: compute gradients for gate and val."""
    row_idx = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    dy = tl.load(dY_ptr + row_idx * stride_dy_row + offs, mask=mask).to(tl.float32)
    gate = tl.load(X_ptr + row_idx * stride_x_row + offs, mask=mask).to(tl.float32)
    val = tl.load(X_ptr + row_idx * stride_x_row + dim + offs, mask=mask).to(tl.float32)

    sig_gate = tl.sigmoid(gate)
    swish_gate = gate * sig_gate

    # d/d_val = dy * silu(gate)
    grad_val = dy * swish_gate
    # d/d_gate = dy * val * d_silu/d_gate = dy * val * sig * (1 + gate * (1 - sig))
    grad_gate = dy * val * (sig_gate * (1.0 + gate * (1.0 - sig_gate)))

    tl.store(dX_ptr + row_idx * stride_dx_row + offs, grad_gate, mask=mask)
    tl.store(dX_ptr + row_idx * stride_dx_row + dim + offs, grad_val, mask=mask)


class _Fast_SwiGLU_Packed_Impl(torch.autograd.Function):
    """Internal autograd Function for SwiGLU. The fallback path is handled by the wrapper, not here."""
    @staticmethod
    def forward(ctx, x):
        ctx.input_shape = x.shape
        B, L, D2 = x.shape
        D = D2 // 2
        x_flat = x.reshape(-1, D2)
        y_flat = torch.empty((x_flat.shape[0], D), device=x.device, dtype=x.dtype)

        BLOCK_SIZE, num_warps = calculate_settings(D)
        # The wrapper should have routed unsupported dims to the fallback already.
        assert BLOCK_SIZE is not None, "SwiGLU kernel called with unsupported dim"

        _swiglu_packed_fwd_kernel[(x_flat.shape[0],)](
            y_flat, y_flat.stride(0),
            x_flat, x_flat.stride(0),
            D, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        ctx.save_for_backward(x_flat)
        ctx.params = (D, BLOCK_SIZE, num_warps)
        return y_flat.view(B, L, D)

    @staticmethod
    def backward(ctx, dY):
        x_flat, = ctx.saved_tensors
        D, BLOCK_SIZE, num_warps = ctx.params
        dY_flat = dY.reshape(-1, D)
        dX_flat = torch.empty_like(x_flat)

        _swiglu_packed_bwd_kernel[(x_flat.shape[0],)](
            dX_flat, dX_flat.stride(0),
            dY_flat, dY_flat.stride(0),
            x_flat, x_flat.stride(0),
            D, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
        )
        return dX_flat.view(ctx.input_shape)


def fast_swiglu_packed(x):
    """Packed SwiGLU activation. Falls back to PyTorch ops outside the Triton-supported range."""
    D2 = x.shape[-1]
    assert D2 % 2 == 0, f"SwiGLU input last dim must be even, got {D2}"
    D = D2 // 2

    BLOCK_SIZE, _ = calculate_settings(D)
    if BLOCK_SIZE is None:
        gate, val = x.split(D, dim=-1)
        return torch.nn.functional.silu(gate) * val

    return _Fast_SwiGLU_Packed_Impl.apply(x)


class Fast_SwiGLU_Packed(torch.autograd.Function):
    """Thin alias — prefer the `fast_swiglu_packed` function entry point."""
    @staticmethod
    def forward(ctx, x):
        return fast_swiglu_packed(x)

    @staticmethod
    def backward(ctx, dY):
        raise RuntimeError("Use fast_swiglu_packed() wrapper instead of Fast_SwiGLU_Packed.apply()")
