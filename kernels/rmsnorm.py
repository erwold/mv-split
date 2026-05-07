# This file is derived from Unsloth (https://github.com/unslothai/unsloth),
#   unsloth/kernels/rms_layernorm.py
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
#   - removed the Gemma RMSNorm variant and the HuggingFace patch/test scaffolding,
#   - added a PyTorch fallback for non-contiguous or oversized inputs,
#   - added a cached-ones helper and a fused_qk_norm wrapper for QK-Norm,
#   - dX is written to a separate buffer rather than in-place over dY.

"""RMSNorm Triton kernel and QK-Norm helper."""

import torch
import triton
import triton.language as tl

from kernels._common import calculate_settings


# Cache `ones` tensors used as the constant RMSNorm weight in QK-Norm.
_ONES_CACHE = {}


def _get_ones(dim: int, device: torch.device, dtype: torch.dtype):
    """Return a cached all-ones tensor of shape (dim,) on the given device/dtype."""
    key = (device.type, device.index, dtype, dim)
    t = _ONES_CACHE.get(key)
    if t is None:
        t = torch.ones((dim,), device=device, dtype=dtype)
        _ONES_CACHE[key] = t
    return t


@triton.jit
def _rms_layernorm_forward(
    Y, Y_stride, X, X_stride, W, W_stride,
    r, r_stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    """RMSNorm forward kernel."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x = tl.load(X + row_idx * X_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + col_offsets * W_stride, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.math.rsqrt(mean_sq + eps)
    tl.store(r + row_idx * r_stride, inv_rms)

    out = x * inv_rms * w
    tl.store(Y + row_idx * Y_stride + col_offsets, out, mask=mask)


@triton.jit
def _rms_layernorm_backward(
    dX, dX_stride, dY, dY_stride, X, X_stride, W, W_stride,
    r, r_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """RMSNorm backward kernel."""
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dy = tl.load(dY + row_idx * dY_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row_idx * X_stride + col_offsets, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + col_offsets * W_stride, mask=mask, other=0.0).to(tl.float32)
    inv_rms = tl.load(r + row_idx * r_stride).to(tl.float32)

    normed = x * inv_rms
    dY_W = dy * w
    rowsum = tl.sum(dY_W * normed, axis=0)
    dx = inv_rms * (dY_W - normed * rowsum / n_cols)

    tl.store(dX + row_idx * dX_stride + col_offsets, dx, mask=mask)


class Fast_RMS_Layernorm(torch.autograd.Function):
    """Autograd Function for RMSNorm with PyTorch fallback for non-contiguous or oversized inputs."""
    @staticmethod
    def forward(ctx, X, W, eps):
        shape = X.shape
        dim = shape[-1]

        # X.reshape(-1, dim) needs X contiguous to be a view (not a copy).
        can_view = X.is_contiguous()

        BLOCK_SIZE, num_warps = calculate_settings(dim)

        # Fall back when the kernel cannot handle the dim, or when the input is non-contiguous.
        if BLOCK_SIZE is None or not can_view:
            ctx.fallback = True
            ctx.eps = eps
            ctx.save_for_backward(X, W)
            X_f = X.float()
            inv_rms = torch.rsqrt(X_f.pow(2).mean(-1, keepdim=True) + eps)
            out = X_f * inv_rms * W.float()
            return out.to(X.dtype)

        ctx.fallback = False
        X_flat = X.reshape(-1, dim)
        Y = torch.empty_like(X_flat)
        r = torch.empty(X_flat.shape[0], dtype=torch.float32, device=X.device)

        _rms_layernorm_forward[(X_flat.shape[0],)](
            Y, Y.stride(0),
            X_flat, X_flat.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            dim, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.save_for_backward(X_flat, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        if getattr(ctx, "fallback", False):
            X, W = ctx.saved_tensors
            X_f = X.float()
            dY_f = dY.float()
            W_f = W.float()

            mean_sq = X_f.pow(2).mean(-1, keepdim=True)
            inv_rms = torch.rsqrt(mean_sq + ctx.eps)
            norm = X_f * inv_rms
            dY_W = dY_f * W_f
            dX = inv_rms * (dY_W - norm * (dY_W * norm).mean(-1, keepdim=True))

            dW = None
            if ctx.needs_input_grad[1]:
                dW = (dY_f * norm).sum(tuple(range(dY.ndim - 1))).to(W.dtype)
            return dX.to(X.dtype), dW, None

        dY_flat = dY.reshape(-1, dY.shape[-1])
        X_flat, W, r = ctx.saved_tensors
        BLOCK_SIZE, num_warps = ctx.BLOCK_SIZE, ctx.num_warps

        # Skip dW when W has requires_grad=False (e.g. the cached `ones` for QK-Norm).
        dW = None
        if ctx.needs_input_grad[1]:
            inv_rms = r.unsqueeze(-1)
            normed = X_flat.float() * inv_rms
            dW = (dY_flat.float() * normed).sum(0).to(W.dtype)

        dX = torch.empty_like(dY_flat)
        _rms_layernorm_backward[(dY_flat.shape[0],)](
            dX, dX.stride(0),
            dY_flat, dY_flat.stride(0),
            X_flat, X_flat.stride(0),
            W, W.stride(0),
            r, r.stride(0),
            dY_flat.shape[-1],
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )
        return dX.view(dY.shape), dW, None


def fast_rms_layernorm(layernorm_module, X):
    """
    Apply RMSNorm using the Triton kernel.

    Args:
        layernorm_module: RMSNorm module with weight and eps attributes.
        X: Input tensor.

    Returns:
        Normalized tensor.
    """
    weight = layernorm_module.weight
    if weight is None:
        weight = _get_ones(X.shape[-1], X.device, X.dtype)
    return Fast_RMS_Layernorm.apply(X, weight, layernorm_module.eps)


def fused_qk_norm(q, k, eps=1e-6):
    """
    Apply RMSNorm to Q and K (QK-Norm).
    Uses a cached `ones` tensor as the weight; since it has requires_grad=False,
    the dW computation in the backward pass is skipped automatically.
    """
    w_q = _get_ones(q.shape[-1], q.device, q.dtype)
    w_k = _get_ones(k.shape[-1], k.device, k.dtype)
    return Fast_RMS_Layernorm.apply(q, w_q, eps), Fast_RMS_Layernorm.apply(k, w_k, eps)
