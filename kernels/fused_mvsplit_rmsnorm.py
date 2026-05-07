"""
Fused MVSplit + RMSNorm Triton kernel.

Computes:
    y = x + beta * (u - mean(u)) + alpha * (mean(u) - mean(x))
    z = y / rms(y)

Assumptions:
- RMSNorm weight is fixed to 1 (non-trainable, no dW computation).
- No bias anywhere.
- D <= 8192 (single-block-per-token design).
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


def calculate_settings(n):
    """Calculate BLOCK_SIZE and num_warps for dimension n."""
    if n > 8192:
        return None, None
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE < 128:
        BLOCK_SIZE = 128
    num_warps = 4
    if BLOCK_SIZE >= 1024:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    return BLOCK_SIZE, num_warps


def get_adaptive_chunk_l(batch_size: int, seq_len: int) -> int:
    """
    Adaptive CHUNK_L based on batch size for better GPU utilization.

    For small batches, we need more chunks to have enough parallel work.
    For large batches, larger chunks reduce output buffer size.
    """
    if batch_size >= 8:
        return 128
    elif batch_size >= 4:
        return 64
    elif batch_size >= 2:
        return 32
    else:  # B=1
        return 16


# =============================================================================
# Shared Logic (JIT Helper)
# =============================================================================
@triton.jit
def _mvsplit_fwd_logic(x, u, beta, m_u, m_x, alpha):
    """Core algebraic fusion: y = x + beta*(u-m_u) + alpha*(m_u-m_x)"""
    # Optimized form: y = x + beta*u + (alpha - beta)*m_u - alpha*m_x
    term_mu = (alpha - beta) * m_u
    term_mx = alpha * m_x
    y = x + beta * u + term_mu - term_mx
    return y


# =============================================================================
# Forward Kernel
# =============================================================================
@triton.jit
def _fwd_kernel(
    Z_ptr, stride_z_b, stride_z_l, stride_z_d,
    X_ptr, stride_x_b, stride_x_l, stride_x_d,
    U_ptr, stride_u_b, stride_u_l, stride_u_d,
    M_U_packed_ptr, stride_mu_seg, stride_mu_b, stride_mu_d,
    M_X_packed_ptr, stride_mx_seg, stride_mx_b, stride_mx_d,
    Alpha_ptr, stride_alpha,
    Beta_ptr, stride_beta,
    L_img,  # Runtime scalar
    eps: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    l_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < dim

    # Load inputs
    offset_base = b_idx * stride_x_b + l_idx * stride_x_l + col_offsets * stride_x_d
    offset_u_base = b_idx * stride_u_b + l_idx * stride_u_l + col_offsets * stride_u_d

    x = tl.load(X_ptr + offset_base, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + offset_u_base, mask=mask, other=0.0).to(tl.float32)

    # Load Stats & Params (segment selection)
    seg_idx = tl.where(l_idx < L_img, 0, 1)
    offset_stats = seg_idx * stride_mu_seg + b_idx * stride_mu_b + col_offsets * stride_mu_d

    m_u = tl.load(M_U_packed_ptr + offset_stats, mask=mask, other=0.0).to(tl.float32)
    m_x = tl.load(M_X_packed_ptr + offset_stats, mask=mask, other=0.0).to(tl.float32)

    alpha = tl.load(Alpha_ptr + col_offsets * stride_alpha, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + col_offsets * stride_beta, mask=mask, other=1.0).to(tl.float32)

    y = _mvsplit_fwd_logic(x, u, beta, m_u, m_x, alpha)

    mean_sq = tl.sum(y * y, axis=0) / dim
    inv_rms = tl.math.rsqrt(mean_sq + eps)
    z = y * inv_rms

    tl.store(Z_ptr + b_idx * stride_z_b + l_idx * stride_z_l + col_offsets * stride_z_d, z, mask=mask)


# =============================================================================
# Backward Pass A: Reduction
# =============================================================================
@triton.jit
def _bwd_pass_a_kernel(
    dZ_ptr, stride_dz_b, stride_dz_l, stride_dz_d,
    X_ptr, stride_x_b, stride_x_l, stride_x_d,
    U_ptr, stride_u_b, stride_u_l, stride_u_d,
    M_U_packed_ptr, stride_mu_seg, stride_mu_b, stride_mu_d,
    M_X_packed_ptr, stride_mx_seg, stride_mx_b, stride_mx_d,
    Alpha_ptr, stride_alpha,
    Beta_ptr, stride_beta,
    # Outputs: Partial Sums (B, NumChunks, D, 2_segments)
    Part_Sum_dY_ptr, stride_part_b, stride_part_c, stride_part_d, stride_part_s,
    Part_Sum_dYu_ptr,
    L_img, L_total,
    eps: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_L: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < dim

    # Hoist invariant params/means outside the loop.
    alpha = tl.load(Alpha_ptr + col_offsets * stride_alpha, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + col_offsets * stride_beta, mask=mask, other=1.0).to(tl.float32)

    mu0 = tl.load(M_U_packed_ptr + 0 * stride_mu_seg + b_idx * stride_mu_b + col_offsets * stride_mu_d,
                  mask=mask, other=0.0).to(tl.float32)
    mu1 = tl.load(M_U_packed_ptr + 1 * stride_mu_seg + b_idx * stride_mu_b + col_offsets * stride_mu_d,
                  mask=mask, other=0.0).to(tl.float32)
    mx0 = tl.load(M_X_packed_ptr + 0 * stride_mx_seg + b_idx * stride_mx_b + col_offsets * stride_mx_d,
                  mask=mask, other=0.0).to(tl.float32)
    mx1 = tl.load(M_X_packed_ptr + 1 * stride_mx_seg + b_idx * stride_mx_b + col_offsets * stride_mx_d,
                  mask=mask, other=0.0).to(tl.float32)

    # Per-segment accumulators: 0 = image tokens, 1 = text tokens.
    acc_dy_0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_dyu_0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_dy_1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_dyu_1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    start_l = chunk_idx * CHUNK_L

    for i in tl.static_range(CHUNK_L):
        curr_l = start_l + i

        valid = curr_l < L_total
        load_mask = mask & valid

        x_offset = b_idx * stride_x_b + curr_l * stride_x_l + col_offsets * stride_x_d
        u_offset = b_idx * stride_u_b + curr_l * stride_u_l + col_offsets * stride_u_d
        dz_offset = b_idx * stride_dz_b + curr_l * stride_dz_l + col_offsets * stride_dz_d

        x = tl.load(X_ptr + x_offset, mask=load_mask, other=0.0).to(tl.float32)
        u = tl.load(U_ptr + u_offset, mask=load_mask, other=0.0).to(tl.float32)
        dz = tl.load(dZ_ptr + dz_offset, mask=load_mask, other=0.0).to(tl.float32)

        is_seg1 = curr_l >= L_img
        m_u = tl.where(is_seg1, mu1, mu0)
        m_x = tl.where(is_seg1, mx1, mx0)

        y = _mvsplit_fwd_logic(x, u, beta, m_u, m_x, alpha)
        inv_rms = tl.math.rsqrt(tl.sum(y * y, axis=0) / dim + eps)
        normed = y * inv_rms

        # dY for the RMSNorm with weight=1 (so dz_w = dz).
        dy = inv_rms * (dz - normed * tl.sum(dz * normed, axis=0) / dim)

        # Branchless accumulation into the correct segment bucket.
        seg1_f = is_seg1.to(tl.float32)
        valid_f = valid.to(tl.float32)

        weight_seg0 = (1.0 - seg1_f) * valid_f
        acc_dy_0 += dy * weight_seg0
        acc_dyu_0 += dy * u * weight_seg0

        weight_seg1 = seg1_f * valid_f
        acc_dy_1 += dy * weight_seg1
        acc_dyu_1 += dy * u * weight_seg1

    # Store partials shaped (B, NumChunks, D, 2_segments).
    base = b_idx * stride_part_b + chunk_idx * stride_part_c + col_offsets * stride_part_d

    tl.store(Part_Sum_dY_ptr + base + 0 * stride_part_s, acc_dy_0, mask=mask)
    tl.store(Part_Sum_dYu_ptr + base + 0 * stride_part_s, acc_dyu_0, mask=mask)

    tl.store(Part_Sum_dY_ptr + base + 1 * stride_part_s, acc_dy_1, mask=mask)
    tl.store(Part_Sum_dYu_ptr + base + 1 * stride_part_s, acc_dyu_1, mask=mask)


# =============================================================================
# Backward Pass B: Write Gradients
# =============================================================================
@triton.jit
def _bwd_pass_b_kernel(
    dX_ptr, stride_dx_b, stride_dx_l, stride_dx_d,
    dU_ptr, stride_du_b, stride_du_l, stride_du_d,
    dZ_ptr, stride_dz_b, stride_dz_l, stride_dz_d,
    X_ptr, stride_x_b, stride_x_l, stride_x_d,
    U_ptr, stride_u_b, stride_u_l, stride_u_d,
    M_U_packed_ptr, stride_mu_seg, stride_mu_b, stride_mu_d,
    M_X_packed_ptr, stride_mx_seg, stride_mx_b, stride_mx_d,
    Mean_dY_packed_ptr, stride_mdy_seg, stride_mdy_b, stride_mdy_d,
    Alpha_ptr, stride_alpha,
    Beta_ptr, stride_beta,
    L_img,
    eps: tl.constexpr,
    dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    l_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < dim

    x = tl.load(X_ptr + b_idx * stride_x_b + l_idx * stride_x_l + col_offsets * stride_x_d,
                mask=mask, other=0.0).to(tl.float32)
    u = tl.load(U_ptr + b_idx * stride_u_b + l_idx * stride_u_l + col_offsets * stride_u_d,
                mask=mask, other=0.0).to(tl.float32)
    dz = tl.load(dZ_ptr + b_idx * stride_dz_b + l_idx * stride_dz_l + col_offsets * stride_dz_d,
                 mask=mask, other=0.0).to(tl.float32)

    seg_idx = tl.where(l_idx < L_img, 0, 1)

    offset_stats = seg_idx * stride_mu_seg + b_idx * stride_mu_b + col_offsets * stride_mu_d
    m_u = tl.load(M_U_packed_ptr + offset_stats, mask=mask, other=0.0).to(tl.float32)
    m_x = tl.load(M_X_packed_ptr + offset_stats, mask=mask, other=0.0).to(tl.float32)

    offset_mdy = seg_idx * stride_mdy_seg + b_idx * stride_mdy_b + col_offsets * stride_mdy_d
    mean_dy = tl.load(Mean_dY_packed_ptr + offset_mdy, mask=mask, other=0.0).to(tl.float32)

    alpha = tl.load(Alpha_ptr + col_offsets * stride_alpha, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta_ptr + col_offsets * stride_beta, mask=mask, other=1.0).to(tl.float32)

    # Recompute dY (RMSNorm weight is 1, so dz_w = dz).
    y = _mvsplit_fwd_logic(x, u, beta, m_u, m_x, alpha)
    inv_rms = tl.math.rsqrt(tl.sum(y * y, axis=0) / dim + eps)
    normed = y * inv_rms
    dy = inv_rms * (dz - normed * tl.sum(dz * normed, axis=0) / dim)

    dx = dy - alpha * mean_dy
    du = beta * dy + (alpha - beta) * mean_dy

    tl.store(dX_ptr + b_idx * stride_dx_b + l_idx * stride_dx_l + col_offsets * stride_dx_d, dx, mask=mask)
    tl.store(dU_ptr + b_idx * stride_du_b + l_idx * stride_du_l + col_offsets * stride_du_d, du, mask=mask)


# =============================================================================
# PyTorch Function
# =============================================================================
class FusedMVSplitRMSFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, u, alpha, beta, eps, L_img):
        assert x.is_contiguous(), "Input X must be contiguous"
        assert u.is_contiguous(), "Input U must be contiguous"

        B, L, D = x.shape

        BLOCK_SIZE, num_warps = calculate_settings(D)
        assert BLOCK_SIZE is not None, f"Dimension D={D} exceeds maximum supported (8192)"

        l_img_val = L if L_img is None else int(L_img)

        # torch.mean is already well-optimized; no need to compute the mean inside Triton.
        if l_img_val >= L:
            m_x_img = x.mean(1, dtype=torch.float32)
            m_u_img = u.mean(1, dtype=torch.float32)
            m_x_txt = torch.zeros_like(m_x_img)
            m_u_txt = torch.zeros_like(m_u_img)
            has_txt = False
        elif l_img_val <= 0:
            m_x_txt = x.mean(1, dtype=torch.float32)
            m_u_txt = u.mean(1, dtype=torch.float32)
            m_x_img = torch.zeros_like(m_x_txt)
            m_u_img = torch.zeros_like(m_u_txt)
            has_txt = True
            l_img_val = 0
        else:
            m_x_img = x[:, :l_img_val].mean(1, dtype=torch.float32)
            m_u_img = u[:, :l_img_val].mean(1, dtype=torch.float32)
            m_x_txt = x[:, l_img_val:].mean(1, dtype=torch.float32)
            m_u_txt = u[:, l_img_val:].mean(1, dtype=torch.float32)
            has_txt = True

        m_u_packed = torch.stack((m_u_img, m_u_txt), dim=0).contiguous()
        m_x_packed = torch.stack((m_x_img, m_x_txt), dim=0).contiguous()

        z = torch.empty_like(x)
        grid = (L, B)
        _fwd_kernel[grid](
            z, z.stride(0), z.stride(1), z.stride(2),
            x, x.stride(0), x.stride(1), x.stride(2),
            u, u.stride(0), u.stride(1), u.stride(2),
            m_u_packed, m_u_packed.stride(0), m_u_packed.stride(1), m_u_packed.stride(2),
            m_x_packed, m_x_packed.stride(0), m_x_packed.stride(1), m_x_packed.stride(2),
            alpha, alpha.stride(0),
            beta, beta.stride(0),
            L_img=l_img_val,
            eps=eps,
            dim=D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        ctx.save_for_backward(x, u, alpha, beta, m_u_packed, m_x_packed)
        ctx.params = (l_img_val, has_txt, eps, BLOCK_SIZE, num_warps, B, L, D)
        return z

    @staticmethod
    def backward(ctx, dz):
        x, u, alpha, beta, m_u_packed, m_x_packed = ctx.saved_tensors
        l_img_val, has_txt, eps, BLOCK_SIZE, num_warps, B, L, D = ctx.params

        if not dz.is_contiguous():
            dz = dz.contiguous()

        CHUNK_L = get_adaptive_chunk_l(B, L)
        num_chunks = (L + CHUNK_L - 1) // CHUNK_L

        # Per-segment partial sums shaped (B, NumChunks, D, 2).
        part_sum_dy = torch.empty((B, num_chunks, D, 2), device=x.device, dtype=torch.float32)
        part_sum_dy_u = torch.empty((B, num_chunks, D, 2), device=x.device, dtype=torch.float32)

        grid_a = (num_chunks, B)
        _bwd_pass_a_kernel[grid_a](
            dz, dz.stride(0), dz.stride(1), dz.stride(2),
            x, x.stride(0), x.stride(1), x.stride(2),
            u, u.stride(0), u.stride(1), u.stride(2),
            m_u_packed, m_u_packed.stride(0), m_u_packed.stride(1), m_u_packed.stride(2),
            m_x_packed, m_x_packed.stride(0), m_x_packed.stride(1), m_x_packed.stride(2),
            alpha, alpha.stride(0),
            beta, beta.stride(0),
            part_sum_dy, part_sum_dy.stride(0), part_sum_dy.stride(1), part_sum_dy.stride(2), part_sum_dy.stride(3),
            part_sum_dy_u,
            L_img=l_img_val,
            L_total=L,
            eps=eps,
            dim=D,
            BLOCK_SIZE=BLOCK_SIZE,
            CHUNK_L=CHUNK_L,
            num_warps=num_warps
        )

        # Aggregate partials across chunks.
        sum_dy_img = part_sum_dy[:, :, :, 0].sum(dim=1)  # (B, D)
        sum_dy_txt = part_sum_dy[:, :, :, 1].sum(dim=1)
        sum_dyu_img = part_sum_dy_u[:, :, :, 0].sum(dim=1)
        sum_dyu_txt = part_sum_dy_u[:, :, :, 1].sum(dim=1)

        # Param gradients are computed in fp32 and cast back at the end for precision.
        m_u_img, m_u_txt = m_u_packed[0], m_u_packed[1]
        m_x_img, m_x_txt = m_x_packed[0], m_x_packed[1]

        d_alpha = d_beta = None

        if ctx.needs_input_grad[2]:
            d_alpha_fp32 = torch.zeros(D, device=x.device, dtype=torch.float32)
            if l_img_val > 0:
                d_alpha_fp32.add_((sum_dy_img * (m_u_img - m_x_img)).sum(0))
            if has_txt:
                d_alpha_fp32.add_((sum_dy_txt * (m_u_txt - m_x_txt)).sum(0))
            d_alpha = d_alpha_fp32.to(alpha.dtype)

        if ctx.needs_input_grad[3]:
            d_beta_fp32 = torch.zeros(D, device=x.device, dtype=torch.float32)
            if l_img_val > 0:
                d_beta_fp32.add_(sum_dyu_img.sum(0) - (sum_dy_img * m_u_img).sum(0))
            if has_txt:
                d_beta_fp32.add_(sum_dyu_txt.sum(0) - (sum_dy_txt * m_u_txt).sum(0))
            d_beta = d_beta_fp32.to(beta.dtype)

        # Pass B: write dX, dU.
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        du = torch.empty_like(u) if ctx.needs_input_grad[1] else None

        if dx is not None or du is not None:
            mean_dy_img = sum_dy_img / l_img_val if l_img_val > 0 else torch.zeros_like(sum_dy_img)
            mean_dy_txt = sum_dy_txt / (L - l_img_val) if (L - l_img_val) > 0 else torch.zeros_like(sum_dy_txt)
            mean_dy_packed = torch.stack((mean_dy_img, mean_dy_txt), dim=0).contiguous()

            if dx is None:
                dx = torch.empty_like(x)
            if du is None:
                du = torch.empty_like(u)

            grid_b = (L, B)
            _bwd_pass_b_kernel[grid_b](
                dx, dx.stride(0), dx.stride(1), dx.stride(2),
                du, du.stride(0), du.stride(1), du.stride(2),
                dz, dz.stride(0), dz.stride(1), dz.stride(2),
                x, x.stride(0), x.stride(1), x.stride(2),
                u, u.stride(0), u.stride(1), u.stride(2),
                m_u_packed, m_u_packed.stride(0), m_u_packed.stride(1), m_u_packed.stride(2),
                m_x_packed, m_x_packed.stride(0), m_x_packed.stride(1), m_x_packed.stride(2),
                mean_dy_packed, mean_dy_packed.stride(0), mean_dy_packed.stride(1), mean_dy_packed.stride(2),
                alpha, alpha.stride(0),
                beta, beta.stride(0),
                L_img=l_img_val,
                eps=eps,
                dim=D,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps
            )

        # Returns: dx, du, d_alpha, d_beta, None (eps), None (L_img).
        return dx, du, d_alpha, d_beta, None, None


def fused_mvsplit_rmsnorm(x, u, alpha, beta, eps, L_img):
    """
    Fused MVSplit + RMSNorm operation.

    Args:
        x: Input tensor (B, L, D)
        u: Update tensor (B, L, D)
        alpha: MVSplit alpha parameter (D,)
        beta: MVSplit beta parameter (D,)
        eps: RMSNorm epsilon
        L_img: Number of image tokens (for segment split)

    Returns:
        Normalized output (B, L, D)

    Note: this kernel assumes the RMSNorm weight is fixed to 1 (non-trainable).
    """
    return FusedMVSplitRMSFunction.apply(x, u, alpha, beta, eps, L_img)


class FusedMVSplitNorm1(torch.nn.Module):
    """
    Fused MVSplit + RMSNorm module.

    Args:
        dim: Feature dimension
        eps: RMSNorm epsilon (default: 1e-6)
        init_alpha: Initial value for alpha (default: 0.0)
        init_beta: Initial value for beta (default: 1.0)

    Note: the RMSNorm weight is fixed to 1 (non-trainable buffer).
    """
    def __init__(self, dim, eps=1e-6, init_alpha=0.0, init_beta=1.0):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.alpha = torch.nn.Parameter(init_alpha * torch.ones(dim), requires_grad=True)
        self.beta = torch.nn.Parameter(init_beta * torch.ones(dim), requires_grad=True)

    def forward(self, x, u, L_img=None):
        return fused_mvsplit_rmsnorm(x, u, self.alpha, self.beta, self.eps, L_img)

    def extra_repr(self):
        return f'dim={self.dim}, eps={self.eps}'