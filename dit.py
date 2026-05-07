# DiT with text cross attention via input concatenation.
# Uses FusedMVSplitNorm1 (MVSplit residual + RMSNorm) on both attention and FFN paths.

import math
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# =====================================================================
# Triton Kernels Import (with fallback)
# =====================================================================
try:
    from kernels.rope import fast_rope_embedding
    from kernels.rmsnorm import fast_rms_layernorm, fused_qk_norm
    from kernels.swiglu import fast_swiglu_packed
    TRITON_AVAILABLE = True
    print("[Info] Triton kernels loaded.")
except ImportError as e:
    TRITON_AVAILABLE = False
    print(f"[Warning] Triton kernels unavailable, using PyTorch fallback: {e}")

# Fused MVSplit+RMSNorm Kernel (optional acceleration)
try:
    from kernels.fused_mvsplit_rmsnorm import FusedMVSplitNorm1, fused_mvsplit_rmsnorm
    FUSED_MVSPLIT_AVAILABLE = True
    print("[Info] Fused MVSplit+RMSNorm kernel loaded.")
except ImportError as e:
    FUSED_MVSPLIT_AVAILABLE = False
    print(f"[Warning] Fused MVSplit kernel unavailable, using PyTorch fallback: {e}")


# =====================================================================
# FusedMVSplitNorm1 - PyTorch Fallback Implementation
# =====================================================================
class FusedMVSplitNorm1Fallback(nn.Module):
    """
    PyTorch fallback for FusedMVSplitNorm1: MVSplit residual + RMSNorm.

    Formula:
        Z = R + beta * V(T) + alpha * (M(T) - M(R))
        Output = RMSNorm(Z)

    Where:
        - V(T) = T - mean(T)  (variance/zero-mean component)
        - M(T) = mean(T)      (mean component)
    """
    def __init__(self, dim, eps=1e-5, init_alpha=0.0, init_beta=0.03):
        super().__init__()
        self.dim = dim
        self.eps = eps

        # Learnable gates for the mean and variance branches.
        self.alpha = nn.Parameter(init_alpha * torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(init_beta * torch.ones(dim), requires_grad=True)

        self.weight = nn.Parameter(torch.ones(dim))

    def _rms_norm(self, x):
        """RMSNorm operation."""
        x_f = x.float()
        norm = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x_f * norm * self.weight.float()
        return out.to(x.dtype)

    def forward(self, x, update, L_img=None):
        """
        Args:
            x: Residual input [B, L, D]
            update: Transform output (operator or FFN) [B, L, D]
            L_img: Number of image tokens (for mixed modality)

        Returns:
            RMSNorm(x + beta*V(update) + alpha*(M(update) - M(x)))
        """
        if L_img is not None and 0 < L_img < x.shape[1]:
            # Mixed Modality (Image + Text) - process separately
            x_img, x_txt = x[:, :L_img], x[:, L_img:]
            u_img, u_txt = update[:, :L_img], update[:, L_img:]

            # Compute means
            m_x_img = x_img.mean(dim=1, keepdim=True)
            m_x_txt = x_txt.mean(dim=1, keepdim=True)
            m_u_img = u_img.mean(dim=1, keepdim=True)
            m_u_txt = u_txt.mean(dim=1, keepdim=True)

            # Variance components (zero-mean)
            v_u_img = u_img - m_u_img
            v_u_txt = u_txt - m_u_txt

            # Expand gates for broadcasting
            beta_expanded = self.beta.view(1, 1, -1)
            alpha_expanded = self.alpha.view(1, 1, -1)

            # Variance branch: controlled by Beta
            total_v_update = torch.cat([
                v_u_img * beta_expanded,
                v_u_txt * beta_expanded
            ], dim=1)

            # Mean branch: controlled by Alpha
            delta_m_img = alpha_expanded * (m_u_img - m_x_img)
            delta_m_txt = alpha_expanded * (m_u_txt - m_x_txt)
            total_m_update = torch.cat([
                delta_m_img.expand_as(x_img),
                delta_m_txt.expand_as(x_txt)
            ], dim=1)
        else:
            # Single Modality
            m_x = x.mean(dim=1, keepdim=True)
            m_u = update.mean(dim=1, keepdim=True)
            v_u = update - m_u

            # Variance branch
            total_v_update = self.beta * v_u
            # Mean branch
            total_m_update = self.alpha * (m_u - m_x).expand_as(x)

        # Combine: Z = R + variance_update + mean_update
        z = x + total_v_update + total_m_update

        # Apply RMSNorm
        return self._rms_norm(z)


if not FUSED_MVSPLIT_AVAILABLE:
    FusedMVSplitNorm1 = FusedMVSplitNorm1Fallback


# =====================================================================
# Basic Components (PatchEmbed, RoPE, Norms)
# =====================================================================
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_size = patch_size

    def forward(self, x):
        x = self.patch_proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class TwoDimRotary(torch.nn.Module):
    def __init__(self, dim, base=10000, h=256, w=256):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / (dim)))
        self.h = h
        self.w = w

        t_h = torch.arange(h).type_as(self.inv_freq)
        t_w = torch.arange(w).type_as(self.inv_freq)
        freqs_h = torch.outer(t_h, self.inv_freq).unsqueeze(1)
        freqs_w = torch.outer(t_w, self.inv_freq).unsqueeze(0)
        freqs_h = freqs_h.repeat(1, w, 1)
        freqs_w = freqs_w.repeat(h, 1, 1)
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)

        self.register_buffer("freqs_hw_cos", freqs_hw.cos(), persistent=False)
        self.register_buffer("freqs_hw_sin", freqs_hw.sin(), persistent=False)

    def forward(self, x, height_width=None):
        if height_width is not None:
            this_h, this_w = height_width
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)
        start_h, start_w = 0, 0
        cos = self.freqs_hw_cos[start_h : start_h + this_h, start_w : start_w + this_w]
        sin = self.freqs_hw_sin[start_h : start_h + this_h, start_w : start_w + this_w]
        cos = cos.clone().reshape(this_h * this_w, -1)
        sin = sin.clone().reshape(this_h * this_w, -1)
        return cos[None, None, :, :], sin[None, None, :, :]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embedding - PyTorch fallback version."""
    orig_dtype = x.dtype
    x = x.to(dtype=torch.float32)
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(dtype=orig_dtype)


class RMSNorm(nn.Module):
    """RMSNorm with optional Triton acceleration."""
    def __init__(self, dim, eps=1e-6, trainable=False):
        super().__init__()
        self.eps = eps
        self.use_triton = TRITON_AVAILABLE

        if trainable:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer("weight", torch.ones(dim))

    def forward(self, x):
        if self.use_triton and x.is_cuda:
            return fast_rms_layernorm(self, x)

        # PyTorch fallback
        x_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x * norm
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(dtype=x_dtype)


class QKNorm(nn.Module):
    """QK-Norm with optional Triton acceleration."""
    def __init__(self, dim, trainable=False):
        super().__init__()
        self.eps = 1e-6
        self.use_triton = TRITON_AVAILABLE and not trainable

        if not self.use_triton:
            self.query_norm = RMSNorm(dim, trainable=trainable)
            self.key_norm = RMSNorm(dim, trainable=trainable)

    def forward(self, q, k):
        if self.use_triton and q.is_cuda:
            return fused_qk_norm(q, k, self.eps)
        return self.query_norm(q), self.key_norm(k)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# =====================================================================
# Efficient Operators (SwiGLU FFN)
# =====================================================================
class SwiGLU(nn.Module):
    """SwiGLU FFN with optional Triton acceleration."""
    def __init__(self, dim, hidden_dim, bias=False):
        super().__init__()
        self.w13 = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.use_triton = TRITON_AVAILABLE

    def forward(self, x):
        x13 = self.w13(x)

        if self.use_triton and x13.is_cuda:
            x = fast_swiglu_packed(x13)
        else:
            gate, value = x13.chunk(2, dim=-1)
            x = F.silu(gate) * value

        x = self.w2(x)
        return x


# =====================================================================
# Attention Module - Fixed RoPE/QKNorm Order with Triton
# =====================================================================
class Attention(nn.Module):
    """
    Multi-Head Attention with QK-Norm and RoPE.

    Order: RoPE is applied before QK-Norm.
    """
    def __init__(
        self, dim, num_heads=16, num_kv_heads=None, qkv_bias=False,
        is_cross_attn=False, cross_attn_dim=None, layer_idx=None,
        use_rope=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        assert dim % num_heads == 0

        self.head_dim = dim // num_heads
        self.is_cross_attn = is_cross_attn
        self.use_rope = use_rope
        self.layer_idx = layer_idx
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if not is_cross_attn:
            if self.num_heads % self.num_kv_heads != 0:
                self.num_kv_heads = self.num_heads
            self.num_groups = self.num_heads // self.num_kv_heads
            kv_dim = self.num_kv_heads * self.head_dim
        else:
            if self.num_kv_heads != self.num_heads:
                self.num_kv_heads = self.num_heads
            self.num_groups = 1
            kv_dim = dim

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        input_dim_kv = cross_attn_dim if is_cross_attn else dim
        self.k_proj = nn.Linear(input_dim_kv, kv_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(input_dim_kv, kv_dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.qk_norm = QKNorm(self.head_dim, trainable=False)

    def forward(self, x, context=None, rope=None, L_img=None):
        input_kv = context if self.is_cross_attn else x

        q = self.q_proj(x)
        k = self.k_proj(input_kv)
        v = self.v_proj(input_kv)

        # Rearrange to [B, H, L, D]
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_kv_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_kv_heads)

        # RoPE first, then QK-Norm.
        if self.use_rope and rope is not None:
            if TRITON_AVAILABLE and q.is_cuda:
                q, k = fast_rope_embedding(q, k, rope[0], rope[1])
            else:
                q = apply_rotary_emb(q, rope[0], rope[1])
                k = apply_rotary_emb(k, rope[0], rope[1])
        else:
            if not q.is_contiguous():
                q = q.contiguous()
            if not k.is_contiguous():
                k = k.contiguous()

        q, k = self.qk_norm(q, k)

        # GQA expansion
        if self.num_groups > 1:
            k = torch.repeat_interleave(k, self.num_groups, dim=1)
            v = torch.repeat_interleave(v, self.num_groups, dim=1)

        # Standard Attention
        x_attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = rearrange(x_attn, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x


# =====================================================================
# DiT Block
# =====================================================================
class DiTBlock(nn.Module):
    """
    DiT block: Attention -> FusedMVSplitNorm1 -> FFN -> FusedMVSplitNorm1.
    Both the attention and FFN residual paths use the fused MVSplit + RMSNorm.
    """
    def __init__(
        self, hidden_size, num_heads, num_kv_heads, mlp_hidden_dim=3072,
        qkv_bias=True, layer_idx=None, block_type="attn", context_dim=None,
        use_rope=True, init_alpha=0.0, init_beta=0.03,
    ):
        super().__init__()
        self.block_type = block_type
        self.layer_idx = layer_idx

        self.fused_mvsplit_norm1 = FusedMVSplitNorm1(
            hidden_size, eps=1e-5, init_alpha=init_alpha, init_beta=init_beta
        )
        self.fused_mvsplit_norm2 = FusedMVSplitNorm1(
            hidden_size, eps=1e-5, init_alpha=init_alpha, init_beta=init_beta
        )

        if block_type == "attn" or block_type == "cross_attn":
            self.operator = Attention(
                hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
                qkv_bias=qkv_bias, is_cross_attn=(block_type == "cross_attn"),
                cross_attn_dim=context_dim if block_type == "cross_attn" else None,
                layer_idx=layer_idx,
                use_rope=use_rope,
            )
        else:
            raise ValueError(f"Unknown block type: {block_type}")

        self.ffn = SwiGLU(hidden_size, mlp_hidden_dim, bias=qkv_bias)

    def forward(self, x, context=None, rope=None, L_img=None):
        R1 = x

        if self.block_type == "attn":
            op_out = self.operator(x, rope=rope, L_img=L_img)
        elif self.block_type == "cross_attn":
            op_out = self.operator(x, context=context, rope=rope, L_img=L_img)

        x = self.fused_mvsplit_norm1(R1, op_out, L_img=L_img)

        R2 = x
        ffn_out = self.ffn(x)
        x = self.fused_mvsplit_norm2(R2, ffn_out, L_img=L_img)

        return x


# =====================================================================
# Main Model (DiT)
# =====================================================================
class DiT(nn.Module):
    """
    Diffusion Transformer with text conditioning via input concatenation.
    A flat sequence of `depth` blocks; block types cycle through `block_pattern`.
    """
    def __init__(
        self,
        in_channels=16, patch_size=2, hidden_size=1280,
        depth=100,
        block_pattern=["attn"],
        num_heads=10, num_kv_heads=10, mlp_hidden_dim=3072, qkv_bias=True,
        context_dim=1024,
        use_rope=True,
        rope_base=10000, rope_h=512, rope_w=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.use_rope = use_rope

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)

        # Input normalization layers
        self.norm_img_input = RMSNorm(hidden_size, eps=1e-5, trainable=qkv_bias)
        self.norm_text_input = RMSNorm(hidden_size, eps=1e-5, trainable=qkv_bias)

        if context_dim is not None:
            if context_dim != hidden_size:
                self.context_proj = nn.Linear(context_dim, hidden_size, bias=False)
            else:
                self.context_proj = Identity()
        else:
            self.context_proj = None
            raise ValueError("context_dim must be specified for Input Concatenation mode.")

        if self.use_rope:
            rope_dim = hidden_size // (2 * num_heads)
            self.rope = TwoDimRotary(rope_dim, base=rope_base, h=rope_h, w=rope_w)
            self.rope_dim = rope_dim
        else:
            self.rope = None
            self.rope_dim = None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_type = block_pattern[i % len(block_pattern)]
            block = DiTBlock(
                hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads,
                mlp_hidden_dim=mlp_hidden_dim, layer_idx=i,
                block_type=block_type, context_dim=None,
                qkv_bias=qkv_bias, use_rope=use_rope,
                init_alpha=0.0, init_beta=0.03,
            )
            self.blocks.append(block)

        self.final_proj = nn.Linear(
            hidden_size, patch_size * patch_size * self.out_channels, bias=True
        )

        print(f"[DiT] Initialized with {depth} blocks.")

    def forward(self, x, context):
        B, C, H_img, W_img = x.shape
        H_patch, W_patch = H_img // self.patch_size, W_img // self.patch_size

        x_img = self.patch_embed(x)
        x_img = self.norm_img_input(x_img)
        L_img = x_img.shape[1]

        if self.context_proj is not None and context is not None:
            text_context = self.context_proj(context)
            text_context = self.norm_text_input(text_context)
            L_text = text_context.shape[1]
            x = torch.cat([x_img, text_context], dim=1)
        else:
            raise ValueError("Context and context_proj must be available.")

        if self.use_rope and self.rope is not None:
            cos_img, sin_img = self.rope(x_img, height_width=(H_patch, W_patch))
            cos_text = torch.ones(
                (cos_img.shape[0], cos_img.shape[1], L_text, self.rope_dim),
                device=x.device, dtype=cos_img.dtype
            )
            sin_text = torch.zeros(
                (sin_img.shape[0], sin_img.shape[1], L_text, self.rope_dim),
                device=x.device, dtype=sin_img.dtype
            )
            cos_combined = torch.cat([cos_img, cos_text], dim=2)
            sin_combined = torch.cat([sin_img, sin_text], dim=2)
            rope = (cos_combined, sin_combined)
        else:
            rope = None

        for block in self.blocks:
            x = block(x, context=None, rope=rope, L_img=L_img)

        x_out_img = x[:, :L_img, :]
        x_out = self.final_proj(x_out_img)

        output = rearrange(
            x_out, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=H_patch, w=W_patch, p1=self.patch_size, p2=self.patch_size,
        )
        return output