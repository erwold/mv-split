# Diagnostic tool for residual-stream mean/variance dynamics in deep DiT training.
# Logs the diagnostic metrics defined in the paper: Writer GMD (G_mean, G_ctr),
# energy ratio rho_T, TR ratio, VarGain, mu_eff, TCS, MV-Split alpha/beta,
# and writer-amplification A and (T-1)*kappa.

import json
import math
import os
from collections import defaultdict
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Basic tensor metrics
# ---------------------------------------------------------------------

@torch.no_grad()
def compute_token_similarity(x: torch.Tensor) -> float:
    if x.ndim != 3:
        return 0.0
    t = x[0].float()
    if t.shape[0] > 256:
        idx = torch.randperm(t.shape[0], device=t.device)[:256]
        t = t[idx]
    t_norm = F.normalize(t, p=2, dim=-1, eps=1e-6)
    sim_matrix = torch.mm(t_norm, t_norm.t())
    mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool, device=t.device)
    return sim_matrix[mask].mean().item()


@torch.no_grad()
def modality_aware_MV_analysis(X: torch.Tensor, L_img=None):
    if X.ndim != 3:
        return None

    X_float = X.float()
    B, L, _ = X_float.shape
    L_img_int = int(L_img) if L_img is not None else None
    is_mixed = L_img_int is not None and 0 < L_img_int < L

    if is_mixed:
        X_img = X_float[:, :L_img_int, :]
        X_text = X_float[:, L_img_int:, :]

        M_img_vec = X_img.mean(dim=1, keepdim=True)
        V_img = X_img - M_img_vec
        M_img = M_img_vec.expand_as(X_img)

        if X_text.shape[1] > 0:
            M_text_vec = X_text.mean(dim=1, keepdim=True)
            V_text = X_text - M_text_vec
            M_text = M_text_vec.expand_as(X_text)
        else:
            V_text = torch.zeros_like(X_text)
            M_text = torch.zeros_like(X_text)

        M = torch.cat([M_img, M_text], dim=1)
        V = torch.cat([V_img, V_text], dim=1)
    else:
        M_vec = X_float.mean(dim=1, keepdim=True)
        V = X_float - M_vec
        M = M_vec.expand_as(X_float)

    energy_M = torch.linalg.vector_norm(M, dim=(1, 2)).mean().item()
    energy_V = torch.linalg.vector_norm(V, dim=(1, 2)).mean().item()

    if energy_V < 1e-9:
        rho = float("inf") if energy_M > 1e-9 else 1.0
    else:
        rho = energy_M / energy_V

    return {
        "M": M.to(X.dtype),
        "V": V.to(X.dtype),
        "energy_M": energy_M,
        "energy_V": energy_V,
        "rho_ratio": rho,
    }


@torch.no_grad()
def compute_mu_eff(A_flat: torch.Tensor, pi_iters: int = 5) -> float:
    if A_flat.ndim != 3 or A_flat.shape[1] != A_flat.shape[2]:
        return 0.0

    M = A_flat
    N, L, _ = M.shape

    v = torch.randn(N, L, 1, device=M.device, dtype=M.dtype)
    v = v - v.mean(dim=1, keepdim=True)
    v = F.normalize(v, p=2, dim=1, eps=1e-12)

    for _ in range(pi_iters):
        Av = torch.bmm(M, v)
        v = torch.bmm(M.transpose(1, 2), Av)
        v = v - v.mean(dim=1, keepdim=True)
        norm_v = torch.linalg.vector_norm(v, dim=1, keepdim=True)
        if torch.min(norm_v) < 1e-9:
            break
        v = v / (norm_v + 1e-12)

    Av = torch.bmm(M, v)
    eigenvalue = (Av ** 2).sum(dim=1).squeeze(-1)
    mu = torch.sqrt(torch.abs(eigenvalue)).mean().item()
    return mu


@torch.no_grad()
def compute_variance_mode_energy(X: torch.Tensor, L_img=None) -> float:
    if X.ndim != 3:
        return 0.0

    x_float = X.float()
    L = x_float.size(1)
    is_mixed = L_img is not None and 0 < L_img < L

    if is_mixed:
        x_img = x_float[:, :L_img, :]
        x_text = x_float[:, L_img:, :]
        V_img = x_img - x_img.mean(dim=1, keepdim=True)
        V_text = x_text - x_text.mean(dim=1, keepdim=True)
        V = torch.cat([V_img, V_text], dim=1)
    else:
        V = x_float - x_float.mean(dim=1, keepdim=True)

    return torch.linalg.vector_norm(V, dim=(1, 2)).mean().item()


@torch.no_grad()
def compute_cosine_similarity_tensors(t1: torch.Tensor, t2: torch.Tensor) -> float:
    if t1.numel() == 0 or t2.numel() == 0 or t1.shape != t2.shape:
        return 0.0
    v1_f = t1.float().reshape(-1, t1.shape[-1])
    v2_f = t2.float().reshape(-1, t2.shape[-1])
    v1_norm = F.normalize(v1_f, p=2, dim=-1, eps=1e-12)
    v2_norm = F.normalize(v2_f, p=2, dim=-1, eps=1e-12)
    cosine_sim = torch.sum(v1_norm * v2_norm, dim=-1)
    return torch.mean(cosine_sim).item()


@torch.no_grad()
def compute_inter_token_similarity(G: torch.Tensor, sample_size: int = 256) -> float:
    if G.ndim < 2:
        return 0.0
    G_flat = G.reshape(-1, G.shape[-1]).float()
    N, _ = G_flat.shape
    if N < 2:
        return 0.0
    if N > sample_size:
        idx = torch.randperm(N, device=G.device)[:sample_size]
        G_flat = G_flat[idx]
    G_norm = F.normalize(G_flat, p=2, dim=-1, eps=1e-12)
    Sim = G_norm @ G_norm.T
    N_sample = Sim.shape[0]
    triu_indices = torch.triu_indices(N_sample, N_sample, offset=1, device=G.device)
    off_diag = Sim[triu_indices[0], triu_indices[1]]
    return torch.mean(torch.abs(off_diag)).item()


@torch.no_grad()
def _tensor_scalar_stats(x: torch.Tensor) -> Dict[str, float]:
    if x is None or x.numel() == 0:
        return {}
    x_det = x.detach().float()
    return {
        "rms": torch.sqrt(torch.mean(x_det ** 2)).item(),
        "abs_max": x_det.abs().max().item(),
    }


@torch.no_grad()
def _per_sample_max_grad_norm(g: torch.Tensor) -> float:
    if g is None or g.ndim < 2:
        return 0.0
    g_flat = g.detach().float().reshape(g.shape[0], -1)
    norms = torch.linalg.vector_norm(g_flat, dim=1)
    return norms.max().item()


@torch.no_grad()
def compute_writer_scaling_metrics(y: torch.Tensor, delta: torch.Tensor, L_img=None, max_batch: int = 1):
    empty = {
        "amp_actual": 0.0,
        "coh_scaled": 0.0,
        "amp_minus_1": -1.0,
        "T": 0.0,
    }

    if y is None or delta is None or y.ndim != 3 or delta.ndim != 3:
        return empty

    B = min(y.shape[0], delta.shape[0], max(int(max_batch), 1))
    T_total = min(y.shape[1], delta.shape[1])
    if B <= 0 or T_total <= 0:
        return empty

    T_use = T_total if L_img is None else min(int(L_img), T_total)
    if T_use <= 0:
        return empty

    amp_vals, coh_vals, t_vals = [], [], []
    y_use = y[:B, :T_use].float()
    d_use = delta[:B, :T_use].float()

    for b in range(B):
        yb = y_use[b]
        db = d_use[b]
        T = yb.shape[0]
        if T <= 0:
            continue

        G = db.transpose(0, 1) @ yb
        grad_sq = (G ** 2).sum()
        self_term = ((db ** 2).sum(dim=-1) * (yb ** 2).sum(dim=-1)).sum()
        amp_actual = (grad_sq / (self_term + 1e-12)).item()

        if T > 1:
            yn = F.normalize(yb, p=2, dim=-1, eps=1e-12)
            dn = F.normalize(db, p=2, dim=-1, eps=1e-12)
            cos_y = yn @ yn.T
            cos_d = dn @ dn.T
            mask = ~torch.eye(T, dtype=torch.bool, device=yb.device)
            kappa_obs = (cos_y[mask].abs() * cos_d[mask].abs()).mean().item()
            coh_scaled = (T - 1) * kappa_obs
        else:
            coh_scaled = 0.0

        amp_vals.append(amp_actual)
        coh_vals.append(coh_scaled)
        t_vals.append(float(T))

    if not amp_vals:
        return empty

    amp_mean = sum(amp_vals) / len(amp_vals)
    coh_mean = sum(coh_vals) / len(coh_vals)
    t_mean = sum(t_vals) / len(t_vals)

    return {
        "amp_actual": amp_mean,
        "coh_scaled": coh_mean,
        "amp_minus_1": amp_mean - 1.0,
        "T": t_mean,
    }


# ---------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------

class SpectrumDiagnosticTool:
    def __init__(self, model, rank: int, world_size: int, output_dir: str = "./diagnostics"):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.output_dir = output_dir
        self.master_process = rank == 0
        self.write_json = os.environ.get("SPECTRUM_DIAG_WRITE_JSON", "0") == "1"

        self.tr_ratio_stats = {}
        self.attention_dynamics_stats = {}
        self.variance_gain_stats = {}
        self.token_sim_stats = {}
        self.rho_ratio_transform_stats = {}
        self.rho_ratio_stats = {}
        self.mv_branch_stats = {}
        self.writer_grad_stats = {}
        self.writer_scaling_stats = {}
        self.branch_output_mean_stats = {}

        self.hook_handles = []
        self._writer_fwd_cache = {}

        self.param_prefix_to_layer_idx = {}
        self._build_layer_idx_mapping()

        self._step_cache = {}
        self._ctx_cache = {}

        self.anchor_layers = {
            0, 1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112,
            128, 144, 160, 176, 192, 208, 224, 240, 255,
            272, 288, 304, 320, 336, 352, 368, 384,
        }
        self.step_interval = 100
        self.current_step = 0

        if self.master_process:
            os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _build_layer_idx_mapping(self):
        self.param_prefix_to_layer_idx = {}
        if self.model is None:
            return

        def _get_unwrapped_module(module):
            target = module
            wrappers = [
                "_fsdp_wrapped_module",
                "checkpoint_wrapper_module",
                "_checkpoint_wrapped_module",
                "_orig_mod",
            ]
            changed = True
            while changed:
                changed = False
                for attr in wrappers:
                    if hasattr(target, attr):
                        nxt = getattr(target, attr)
                        if nxt is not None:
                            target = nxt
                            changed = True
                            break
            return target

        block_class_names = {"DiTBlock"}

        for name, module in self.model.named_modules():
            unwrapped = _get_unwrapped_module(module)
            if type(unwrapped).__name__ in block_class_names:
                layer_idx = getattr(unwrapped, "layer_idx", None)
                if layer_idx is not None:
                    self.param_prefix_to_layer_idx[name] = layer_idx

        if self.master_process and self.param_prefix_to_layer_idx:
            print(f"[Diag] Built layer_idx mapping for {len(self.param_prefix_to_layer_idx)} blocks")

    def _get_layer_idx_from_param_name(self, param_name: str):
        prefixes = sorted(self.param_prefix_to_layer_idx.keys(), key=len, reverse=True)
        for prefix in prefixes:
            if param_name.startswith(prefix + ".") or param_name == prefix:
                return self.param_prefix_to_layer_idx[prefix]
        return "Emb/Final"

    def _classify_param_type(self, name: str) -> str:
        if "final_proj" in name:
            return "Final"
        if "patch_embed" in name:
            return "Embed"
        if "context_proj" in name:
            return "Ctx_Proj"
        if "t_embedder" in name:
            return "T_Embed"
        if "logit_scale" in name:
            return "LogitScale"

        if "operator.q_proj" in name:
            return "Q_Proj"
        if "operator.k_proj" in name:
            return "K_Proj"
        if "operator.v_proj" in name:
            return "V_Proj"
        if name.endswith("operator.proj.weight") or ".operator.proj." in name:
            return "Attn_WO"

        if "ffn.w13" in name:
            return "FFN_W13"
        if "ffn.w2" in name:
            return "FFN_W2"

        if "norm" in name:
            return "Norm"

        return "Other"

    def _should_log_layer(self, layer_idx: int) -> bool:
        if self.current_step % self.step_interval != 0:
            return False
        return layer_idx in self.anchor_layers

    @staticmethod
    def _median(values):
        if not values:
            return None
        return torch.tensor(values, dtype=torch.float32).median().item()

    # -----------------------------------------------------------------
    # Step context
    # -----------------------------------------------------------------

    def set_step_context(self, **kwargs):
        self._ctx_cache = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                self._ctx_cache[k] = v.detach().item() if v.numel() == 1 else v.detach().float().mean().item()
            elif isinstance(v, (int, float)):
                self._ctx_cache[k] = v
        self._step_cache["ctx"] = self._ctx_cache

    def set_current_step(self, step: int):
        self.current_step = step
        self._step_cache = {"step": step, "rank": self.rank}
        if self._ctx_cache:
            self._step_cache["ctx"] = self._ctx_cache
        self._writer_fwd_cache.clear()

    # -----------------------------------------------------------------
    # Output grad hook for spike autopsy
    # -----------------------------------------------------------------

    def register_output_grad_hooks(self, model):
        target = None
        target_name = "final_proj"
        if hasattr(model, "final_proj"):
            target = model.final_proj
        else:
            for n, m in model.named_modules():
                if n.endswith("final_proj"):
                    target = m
                    target_name = n
                    break

        if target is None:
            if self.master_process:
                print("[Diag] Warning: final_proj not found, output-grad hook disabled.")
            return

        def bwd_hook(mod, grad_input, grad_output):
            if not getattr(mod, "training", False):
                return
            self._step_cache.setdefault("final_grad_bwd", {})
            if grad_output and isinstance(grad_output[0], torch.Tensor):
                g_out = grad_output[0]
                stats = _tensor_scalar_stats(g_out)
                stats["max_sample_gn"] = _per_sample_max_grad_norm(g_out)
                self._step_cache["final_grad_bwd"].update(stats)

        target.register_full_backward_hook(bwd_hook)
        if self.master_process:
            print(f"[Diag] Output-grad hooks registered on: {target_name}")

    # -----------------------------------------------------------------
    # Spike diagnosis
    # -----------------------------------------------------------------

    @torch.no_grad()
    def diagnose_gradient_spike(self, model):
        if not self.param_prefix_to_layer_idx:
            self._build_layer_idx_mapping()

        local_buckets = defaultdict(float)
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            layer_idx = self._get_layer_idx_from_param_name(name)
            p_type = self._classify_param_type(name)
            g = p.grad
            g_local = g.to_local() if hasattr(g, "to_local") else g
            g_sq = g_local.detach().float().pow(2).sum().item()
            local_buckets[(layer_idx, p_type)] += g_sq

        all_buckets = [None] * self.world_size
        all_caches = [None] * self.world_size
        dist.all_gather_object(all_buckets, local_buckets)
        dist.all_gather_object(all_caches, self._step_cache)

        if not self.master_process:
            return

        total_buckets = defaultdict(float)
        for b in all_buckets:
            for k, v in b.items():
                total_buckets[k] += v

        sorted_stats = []
        for (lid, ptype), sq in total_buckets.items():
            sorted_stats.append({"layer": lid, "type": ptype, "norm": math.sqrt(sq)})
        sorted_stats.sort(key=lambda x: x["norm"], reverse=True)

        print(f"\n{'=' * 60}")
        print("GRADIENT SPIKE AUTOPSY")
        print(f"{'=' * 60}")
        print(f"{'Norm':<10} | {'Layer':<12} | {'Type':<10}")
        print("-" * 40)
        for item in sorted_stats[:15]:
            layer_str = str(item["layer"]) if isinstance(item["layer"], int) else item["layer"]
            print(f"{item['norm']:<10.4f} | {layer_str:<12} | {item['type']:<10}")

        if sorted_stats:
            top = sorted_stats[0]
            if top["type"] in ["Norm", "Norm_W", "Norm_B"]:
                print(">>> HYPOTHESIS: Post-Norm Jacobian explosion")
            elif top["type"] in ["Q_Proj", "K_Proj"]:
                print(">>> HYPOTHESIS: Basis / logits temperature spike")
            elif top["type"] in ["Attn_WO", "FFN_W2"]:
                print(">>> HYPOTHESIS: Writer/output projection sudden update")

        print(f"\n{'=' * 60}")
        print("ROOT CAUSE SNAPSHOT")
        print(f"{'=' * 60}")
        print(f"{'Rank':<4} | {'Loss':<8} | {'Scale/Wt':<8} | {'dL/dOut_RMS':<12} | {'Max_Samp_GN':<12} | {'Status'}")
        print("-" * 80)

        for r, cache in enumerate(all_caches):
            if not cache:
                continue
            ctx = cache.get("ctx", {})
            fgrad = cache.get("final_grad_bwd", {})
            loss = ctx.get("loss", -1)
            scale = ctx.get("loss_weight", -1)
            dout_rms = fgrad.get("rms", 0.0)
            max_samp_gn = fgrad.get("max_sample_gn", 0.0)

            status = ""
            if max_samp_gn > 0 and dout_rms > 0:
                if max_samp_gn > 5 * dout_rms + 1.0:
                    status = "BAD_BATCH"
                elif dout_rms > 10.0:
                    status = "GLOBAL_INJ"

            print(f"{r:<4d} | {loss:<8.4f} | {scale:<8.2f} | {dout_rms:<12.4f} | {max_samp_gn:<12.4f} | {status}")

        print("-" * 80)

    # -----------------------------------------------------------------
    # Hook factories
    # -----------------------------------------------------------------

    def _make_forward_hook(self, layer_idx: int, module_name: str, op_type: str, parent_block=None):
        def hook(module, input_tuple, output):
            if not self.master_process:
                return
            if not getattr(module, "training", False):
                return
            if not self._should_log_layer(layer_idx):
                return
            if output is None:
                return

            with torch.no_grad():
                L_img = getattr(parent_block, "L_img_cache", None) if parent_block is not None else None

                mv_stats = modality_aware_MV_analysis(output.detach(), L_img=L_img)
                if mv_stats is not None:
                    self.branch_output_mean_stats[(layer_idx, op_type)] = {
                        "layer_idx": layer_idx,
                        "op_type": op_type,
                        "rho_U": mv_stats["rho_ratio"],
                    }

                if parent_block is not None:
                    residual_mode = getattr(parent_block, "residual_mode", "none")
                    if residual_mode == "mvsplit":
                        mv_mod = None
                        if op_type in ["attn", "cross_attn", "operator"]:
                            if hasattr(parent_block, "fused_mvsplit_norm1"):
                                mv_mod = parent_block.fused_mvsplit_norm1
                        elif op_type == "ffn":
                            if hasattr(parent_block, "fused_mvsplit_norm2"):
                                mv_mod = parent_block.fused_mvsplit_norm2

                        if mv_mod is not None:
                            self.log_mv_branch_stats(
                                layer_idx=layer_idx,
                                branch_name=op_type,
                                alpha_param=mv_mod.alpha.detach(),
                                beta_param=mv_mod.beta.detach(),
                                update_tensor=output.detach(),
                                L_img=L_img,
                            )

        return hook

    def _make_writer_forward_hook(self, layer_idx: int, module_name: str, parent_block=None):
        def hook(module, input_tuple):
            if not self.master_process:
                return
            if not getattr(module, "training", False):
                return
            if not self._should_log_layer(layer_idx):
                return
            if input_tuple and input_tuple[0] is not None:
                self._writer_fwd_cache[module_name] = {
                    "y": input_tuple[0].detach(),
                    "L_img": getattr(parent_block, "L_img_cache", None) if parent_block is not None else None,
                }
        return hook

    def _make_writer_backward_hook(self, layer_idx: int, module_name: str, op_type: str):
        def hook(module, grad_input, grad_output):
            if not self.master_process:
                return
            if not getattr(module, "training", False):
                return
            if not self._should_log_layer(layer_idx):
                return

            delta = grad_output[0]
            cache_entry = self._writer_fwd_cache.pop(module_name, None)
            if cache_entry is None or delta is None:
                return

            y = cache_entry.get("y", None)
            L_img = cache_entry.get("L_img", None)
            if not isinstance(y, torch.Tensor) or not isinstance(delta, torch.Tensor):
                return
            if y.ndim != 3 or delta.ndim != 3:
                return

            with torch.no_grad():
                y_f = y.float()
                delta_f = delta.float()
                _, T, _ = y_f.shape

                y_bar = y_f.mean(dim=1, keepdim=True)
                delta_bar = delta_f.mean(dim=1, keepdim=True)
                y_tilde = y_f - y_bar
                delta_tilde = delta_f - delta_bar

                grad_mean = torch.bmm(delta_bar.transpose(1, 2), y_bar) * T
                grad_ctr = torch.bmm(delta_tilde.transpose(1, 2), y_tilde)

                grad_mean_sum = grad_mean.sum(dim=0)
                grad_ctr_sum = grad_ctr.sum(dim=0)

                self.writer_grad_stats[(layer_idx, op_type)] = {
                    "layer_idx": layer_idx,
                    "op_type": op_type,
                    "G_mean": grad_mean_sum.norm().item(),
                    "G_ctr": grad_ctr_sum.norm().item(),
                }

                scaling_stats = compute_writer_scaling_metrics(y, delta, L_img=L_img, max_batch=1)
                self.writer_scaling_stats[(layer_idx, op_type)] = {
                    "layer_idx": layer_idx,
                    "op_type": op_type,
                    **scaling_stats,
                }

        return hook

    # -----------------------------------------------------------------
    # Register hooks
    # -----------------------------------------------------------------

    def register_spectrum_hooks(self, model):
        if not self.master_process:
            return

        print("[Diag] registering hooks...")

        block_class_names = {"DiTBlock"}

        for block_name, block_module in model.named_modules():
            if type(block_module).__name__ not in block_class_names:
                continue

            layer_idx = getattr(block_module, "layer_idx", None)
            if layer_idx is None:
                continue

            if hasattr(block_module, "operator"):
                token_mixer = block_module.operator
                token_mixer_name = f"{block_name}.operator"
                block_type = getattr(block_module, "block_type", "operator")

                h1 = token_mixer.register_forward_hook(
                    self._make_forward_hook(layer_idx, token_mixer_name, block_type, block_module)
                )
                self.hook_handles.append(h1)

                if block_type in ["attn", "cross_attn"] and hasattr(token_mixer, "proj"):
                    writer_mod = token_mixer.proj
                    hw1 = writer_mod.register_forward_pre_hook(
                        self._make_writer_forward_hook(layer_idx, f"{block_name}.attn.proj", block_module)
                    )
                    hw2 = writer_mod.register_full_backward_hook(
                        self._make_writer_backward_hook(layer_idx, f"{block_name}.attn.proj", "Attn_WO")
                    )
                    self.hook_handles.extend([hw1, hw2])

            if hasattr(block_module, "ffn"):
                ffn_mod = block_module.ffn
                h3 = ffn_mod.register_forward_hook(
                    self._make_forward_hook(layer_idx, f"{block_name}.ffn", "ffn", block_module)
                )
                self.hook_handles.append(h3)

                if hasattr(ffn_mod, "w2"):
                    writer_mod = ffn_mod.w2
                    hw3 = writer_mod.register_forward_pre_hook(
                        self._make_writer_forward_hook(layer_idx, f"{block_name}.ffn.w2", block_module)
                    )
                    hw4 = writer_mod.register_full_backward_hook(
                        self._make_writer_backward_hook(layer_idx, f"{block_name}.ffn.w2", "FFN_W2")
                    )
                    self.hook_handles.extend([hw3, hw4])

        print(f"[Diag] registered {len(self.hook_handles)} hooks")

    # -----------------------------------------------------------------
    # Logging endpoints
    # -----------------------------------------------------------------

    @torch.no_grad()
    def log_attention_dynamics(self, layer_idx: int, mu_eff: float):
        if not self.master_process:
            return
        self.attention_dynamics_stats[layer_idx] = {"layer_idx": layer_idx, "mu_eff": mu_eff}

    @torch.no_grad()
    def log_token_similarity(self, layer_idx: int, sim: float):
        if not self.master_process:
            return
        self.token_sim_stats[layer_idx] = {"layer_idx": layer_idx, "sim": sim}

    @torch.no_grad()
    def log_tr_ratio(self, layer_idx: int, residual: torch.Tensor, transform: torch.Tensor, op_type: str):
        if not self.master_process or not self._should_log_layer(layer_idx):
            return
        residual_rms = torch.sqrt(torch.mean(residual.float() ** 2)).item()
        transform_rms = torch.sqrt(torch.mean(transform.float() ** 2)).item()
        tr_ratio = transform_rms / (residual_rms + 1e-9)
        self.tr_ratio_stats[(layer_idx, op_type)] = {
            "layer_idx": layer_idx,
            "op_type": op_type,
            "residual_rms": residual_rms,
            "transform_rms": transform_rms,
            "tr_ratio": tr_ratio,
        }

    @torch.no_grad()
    def log_variance_gain(self, layer_idx: int, R: torch.Tensor, T: torch.Tensor, op_type: str, L_img=None):
        if not self.master_process or not self._should_log_layer(layer_idx):
            return
        energy_R_V = compute_variance_mode_energy(R, L_img=L_img)
        energy_T_V = compute_variance_mode_energy(T, L_img=L_img)
        if energy_R_V < 1e-9:
            var_gain = float("inf") if energy_T_V > 1e-9 else 1.0
        else:
            var_gain = energy_T_V / energy_R_V
        self.variance_gain_stats[(layer_idx, op_type)] = {
            "layer_idx": layer_idx,
            "op_type": op_type,
            "var_gain": var_gain,
        }

    @torch.no_grad()
    def log_rho_ratio_transform(self, layer_idx: int, tensor: torch.Tensor, op_type: str):
        if not self.master_process or not self._should_log_layer(layer_idx):
            return
        mv_stats = modality_aware_MV_analysis(tensor)
        if mv_stats is None:
            return
        self.rho_ratio_transform_stats[(layer_idx, op_type)] = {
            "layer_idx": layer_idx,
            "op_type": op_type,
            "rho_ratio_T": mv_stats["rho_ratio"],
        }

    @torch.no_grad()
    def log_rho_ratio(self, stage_idx: int, tensor: torch.Tensor, L_img: int = None):
        if not self.master_process:
            return
        if tensor.ndim != 3:
            return
        if self.current_step % self.step_interval != 0:
            return

        x = tensor.float()

        def calc_rho(t):
            if t.shape[1] == 0:
                return 0.0
            M_vec = t.mean(dim=1, keepdim=True)
            V_t = t - M_vec
            e_m = torch.linalg.vector_norm(M_vec.expand_as(t), dim=(1, 2)).mean().item()
            e_v = torch.linalg.vector_norm(V_t, dim=(1, 2)).mean().item()
            return float("inf") if e_v < 1e-9 else e_m / e_v

        entry = {
            "stage_idx": stage_idx,
            "rho_ratio": calc_rho(x),
        }
        if L_img is not None and 0 < L_img < x.shape[1]:
            entry["rho_img"] = calc_rho(x[:, :L_img, :])
            entry["rho_text"] = calc_rho(x[:, L_img:, :])

        self.rho_ratio_stats[stage_idx] = entry

    @torch.no_grad()
    def log_mv_branch_stats(
        self,
        layer_idx: int,
        branch_name: str,
        alpha_param: torch.Tensor,
        beta_param: torch.Tensor,
        update_tensor: torch.Tensor,
        L_img=None,
    ):
        if not self.master_process:
            return

        alpha_t = alpha_param.detach().float()
        beta_t = beta_param.detach().float()
        alpha_mean = alpha_t.abs().mean().item()
        alpha_max = alpha_t.abs().max().item()
        beta_mean = beta_t.abs().mean().item()

        mv_stats = modality_aware_MV_analysis(update_tensor, L_img)
        if mv_stats is None:
            return

        e_m = mv_stats["energy_M"]
        e_v = mv_stats["energy_V"]

        self.mv_branch_stats[(layer_idx, branch_name)] = {
            "layer_idx": layer_idx,
            "branch": branch_name,
            "alpha_mean": alpha_mean,
            "alpha_max": alpha_max,
            "beta_mean": beta_mean,
            "raw_mv_ratio": e_m / (e_v + 1e-9),
            "eff_mean_energy": e_m * alpha_mean,
            "var_energy": e_v,
            "alpha": alpha_mean,
        }

    # -----------------------------------------------------------------
    # G(W_O) vs G(Q) monitoring
    # -----------------------------------------------------------------

    @torch.no_grad()
    def monitor_training_dynamics(self, step: int, num_stages: int = 4):
        grouped_tensors = defaultdict(list)

        def identify_role(name: str):
            if "operator.proj.weight" in name or ".operator.proj." in name: return "Attn_WO"
            if "operator.q_proj" in name: return "Q_Proj"
            if "operator.k_proj" in name: return "K_Proj"
            if "ffn.w2" in name: return "FFN_W2"
            return None

        int_layers = [v for v in self.param_prefix_to_layer_idx.values() if isinstance(v, int)]
        if not int_layers:
            return {}
        max_layer = max(int_layers)
        stage_size = max(1, math.ceil((max_layer + 1) / max(num_stages, 1)))
        deep_cutoff = int(0.75 * max_layer)

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            layer_idx = self._get_layer_idx_from_param_name(name)
            if not isinstance(layer_idx, int):
                continue
            role = identify_role(name)
            if role is None:
                continue

            s_idx = min(layer_idx // stage_size, num_stages - 1)

            p_local = p.to_local() if hasattr(p, "to_local") else p
            grouped_tensors[(s_idx, role, "w")].append(p_local.detach().float().flatten())

            if p.grad is not None:
                g_local = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                g_flat = g_local.detach().float().flatten()
                grouped_tensors[(s_idx, role, "g")].append(g_flat)
                grouped_tensors[("global", role, "g")].append(g_flat)
                if layer_idx >= deep_cutoff:
                    grouped_tensors[("deep", role, "g")].append(g_flat)

        results = {}
        for key, t_list in grouped_tensors.items():
            if not t_list:
                continue
            norms = torch._foreach_norm(t_list, 2.0)
            norms_t = torch.stack(norms).to(dtype=torch.float32)
            results[key] = norms_t.pow(2).sum().item()

        if not results:
            return {}

        if dist.is_initialized():
            all_sq = [None] * self.world_size
            dist.all_gather_object(all_sq, results)
            total_sq = defaultdict(float)
            for d in all_sq:
                for k, v in d.items():
                    total_sq[k] += v
        else:
            total_sq = results

        if not self.master_process:
            return {}

        metrics = {}
        for (group_id, role, kind), sq in total_sq.items():
            norm = math.sqrt(sq)
            if group_id in ["global", "deep"]:
                prefix = f"dynamics_{group_id}/{role}_{'Grad' if kind == 'g' else 'Weight'}_L2"
            else:
                prefix = f"dynamics_stage/S{group_id}_{role}_{'Grad' if kind == 'g' else 'Weight'}_L2"
            metrics[prefix] = norm

        wo_g = math.sqrt(total_sq.get(("global", "Attn_WO", "g"), 0.0))
        q_g = math.sqrt(total_sq.get(("global", "Q_Proj", "g"), 0.0))
        metrics["dynamics_global/Ratio_Grad_WO_over_Q"] = wo_g / (q_g + 1e-9)

        return metrics

    # -----------------------------------------------------------------
    # Report export
    # -----------------------------------------------------------------

    def generate_comprehensive_report(self, step: int):
        dyn_metrics = self.monitor_training_dynamics(step, num_stages=4)

        if not self.master_process:
            return None

        wandb_metrics = {}
        if dyn_metrics:
            wandb_metrics.update(dyn_metrics)

        for stat in self.tr_ratio_stats.values():
            prefix = f"tr_ratio/layer_{stat['layer_idx']:03d}_{stat['op_type']}"
            wandb_metrics[f"{prefix}/ratio"] = stat["tr_ratio"]
            wandb_metrics[f"{prefix}/transform_rms"] = stat["transform_rms"]
            wandb_metrics[f"{prefix}/residual_rms"] = stat["residual_rms"]

        for stat in self.rho_ratio_stats.values():
            s_idx = stat["stage_idx"]
            wandb_metrics[f"rho_trunk/stage_{s_idx:03d}/global"] = stat["rho_ratio"]
            if "rho_img" in stat:
                wandb_metrics[f"rho_trunk/stage_{s_idx:03d}/img"] = stat["rho_img"]
                wandb_metrics[f"rho_trunk/stage_{s_idx:03d}/text"] = stat["rho_text"]

        for stat in self.rho_ratio_transform_stats.values():
            prefix = f"rho_branch/layer_{stat['layer_idx']:03d}_{stat['op_type']}"
            wandb_metrics[f"{prefix}/ratio"] = stat["rho_ratio_T"]

        for stat in self.attention_dynamics_stats.values():
            prefix = f"attn_dynamics/layer_{stat['layer_idx']:03d}"
            wandb_metrics[f"{prefix}/mu_eff"] = stat["mu_eff"]

        for stat in self.variance_gain_stats.values():
            prefix = f"vargain/layer_{stat['layer_idx']:03d}_{stat['op_type']}"
            wandb_metrics[f"{prefix}/gain"] = stat["var_gain"]

        for stat in self.token_sim_stats.values():
            prefix = f"token_collapse/layer_{stat['layer_idx']:03d}"
            wandb_metrics[f"{prefix}/token_similarity"] = stat["sim"]

        for stat in self.mv_branch_stats.values():
            prefix = f"mv_split/layer_{stat['layer_idx']:03d}_{stat['branch']}"
            wandb_metrics[f"{prefix}/alpha_mean"] = stat["alpha_mean"]
            wandb_metrics[f"{prefix}/alpha_max"] = stat["alpha_max"]
            wandb_metrics[f"{prefix}/beta_mean"] = stat["beta_mean"]
            wandb_metrics[f"{prefix}/raw_mv_ratio"] = stat["raw_mv_ratio"]
            wandb_metrics[f"{prefix}/effective_mean_energy"] = stat["eff_mean_energy"]
            wandb_metrics[f"{prefix}/var_energy"] = stat["var_energy"]

        for stat in self.branch_output_mean_stats.values():
            prefix = f"branch_output/layer_{stat['layer_idx']:03d}_{stat['op_type']}"
            wandb_metrics[f"{prefix}/rho_U"] = stat["rho_U"]

        for stat in self.writer_grad_stats.values():
            prefix = f"writer_grad/layer_{stat['layer_idx']:03d}_{stat['op_type']}"
            wandb_metrics[f"{prefix}/G_mean"] = stat["G_mean"]
            wandb_metrics[f"{prefix}/G_ctr"] = stat["G_ctr"]
            if stat["G_ctr"] > 1e-12:
                wandb_metrics[f"{prefix}/Ratio_mean_to_ctr"] = stat["G_mean"] / stat["G_ctr"]

        for stat in self.writer_scaling_stats.values():
            prefix = f"writer_scaling/layer_{stat['layer_idx']:03d}_{stat['op_type']}"
            wandb_metrics[f"{prefix}/amp_actual"] = stat["amp_actual"]
            wandb_metrics[f"{prefix}/coh_scaled"] = stat["coh_scaled"]
            wandb_metrics[f"{prefix}/amp_minus_1"] = stat["amp_minus_1"]
            wandb_metrics[f"{prefix}/T"] = stat["T"]

        # Deep-writer median export for the attention output projection.
        attn_writer_stats = [
            s for s in self.writer_grad_stats.values()
            if s["op_type"] == "Attn_WO" and isinstance(s["layer_idx"], int)
        ]
        if attn_writer_stats:
            max_layer = max(s["layer_idx"] for s in attn_writer_stats)
            deep_stats = [s for s in attn_writer_stats if s["layer_idx"] >= int(0.75 * max_layer)]
            if deep_stats:
                g_mean_med = self._median([s["G_mean"] for s in deep_stats])
                g_ctr_med = self._median([s["G_ctr"] for s in deep_stats])
                ratio_med = self._median([
                    s["G_mean"] / (s["G_ctr"] + 1e-12) for s in deep_stats
                ])
                if g_mean_med is not None:
                    wandb_metrics["writer_grad/deep_attn_wo_median/G_mean"] = g_mean_med
                if g_ctr_med is not None:
                    wandb_metrics["writer_grad/deep_attn_wo_median/G_ctr"] = g_ctr_med
                if ratio_med is not None:
                    wandb_metrics["writer_grad/deep_attn_wo_median/Ratio_mean_to_ctr"] = ratio_med

        report_data = {
            "step": step,
            "tr_ratio": list(self.tr_ratio_stats.values()),
            "rho_trunk": list(self.rho_ratio_stats.values()),
            "rho_branch": list(self.rho_ratio_transform_stats.values()),
            "attention_dynamics": list(self.attention_dynamics_stats.values()),
            "variance_gain": list(self.variance_gain_stats.values()),
            "token_similarity": list(self.token_sim_stats.values()),
            "mv_branch": list(self.mv_branch_stats.values()),
            "branch_output": list(self.branch_output_mean_stats.values()),
            "writer_grad": list(self.writer_grad_stats.values()),
            "writer_scaling": list(self.writer_scaling_stats.values()),
        }
        self._save_json(report_data, f"comprehensive_report_step{step}.json")

        self.tr_ratio_stats.clear()
        self.attention_dynamics_stats.clear()
        self.variance_gain_stats.clear()
        self.token_sim_stats.clear()
        self.rho_ratio_transform_stats.clear()
        self.rho_ratio_stats.clear()
        self.mv_branch_stats.clear()
        self.writer_grad_stats.clear()
        self.writer_scaling_stats.clear()
        self.branch_output_mean_stats.clear()
        self._writer_fwd_cache.clear()

        return wandb_metrics

    # -----------------------------------------------------------------
    # Optional JSON dump
    # -----------------------------------------------------------------

    def _save_json(self, data, filename):
        if not self.master_process or not self.write_json:
            return
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def cleanup_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self._writer_fwd_cache.clear()
