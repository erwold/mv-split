# MVSplit-DiT (1000 layers)

A 1000-layer Diffusion Transformer trained with **FusedMVSplitNorm1** — a residual scheme that splits each block's update into a *mean* branch and a *variance* branch, each with its own learnable per-channel gate, followed by RMSNorm. Text conditioning is via input concatenation; sampling uses rectified-flow / flow-matching.

Released weights: <https://huggingface.co/StableKirito/mvsplit-dit-1000l>

## Files

```
dit.py                              # DiT model (FusedMVSplitNorm1 residual + RoPE + QK-Norm)
text_encoder.py                     # Qwen3 text encoder wrapper
vae.py                              # FLUX.2 AutoEncoder
sample.py                           # Inference / image sampling entry point
sample_prompts.txt                  # Example prompts (one per line)
kernels/
  fused_mvsplit_rmsnorm.py          # Triton kernel: MVSplit + RMSNorm
  rmsnorm.py                        # Triton RMSNorm + QK-Norm
  rope.py                           # Triton RoPE
  swiglu.py                         # Triton packed SwiGLU
  _common.py
```

All Triton kernels have PyTorch fallbacks, so the model runs on machines without Triton — just slower.

## Installation

```bash
pip install -r requirements.txt
# Triton (required for the fast path; ships with PyTorch on Linux+CUDA):
#   pip install triton
```

Tested with PyTorch 2.x on CUDA. CPU works for the fallback path but is impractical at this depth.

## Weights

Download three artifacts:

1. **DiT checkpoint** — `model.pt` from the HF repo above.
2. **FLUX.2 AE** — `flux2_ae.safetensors` (available as `ae.safetensors` in <https://huggingface.co/black-forest-labs/FLUX.2-dev>; rename or pass the original name via `--flux_vae_path`).
3. **Qwen3 text encoder** — `Qwen/Qwen3-0.6B` is fetched automatically by `transformers` on first run; or set `--qwen_model_path` to a local snapshot.

Place them anywhere and pass the paths via CLI flags (the defaults assume `/workspace/...`).

## Sampling

```bash
# Custom prompt
python sample.py \
    --checkpoint_path /path/to/model.pt \
    --flux_vae_path   /path/to/flux2_ae.safetensors \
    --qwen_model_path Qwen/Qwen3-0.6B \
    --prompt "a red panda climbing a bamboo stalk" \
    --output_dir ./samples

# Pick a specific line from the prompts file (1-indexed)
python sample.py --line 5

# Randomly sample N prompts (reproducible with --seed)
python sample.py --num_samples 4 --seed 42

# Generate every prompt in the file
python sample.py --all
```

Outputs `<name>.png` plus a `metadata.jsonl` log under `--output_dir`.

### Key sampling flags

| Flag | Default | Meaning |
|---|---|---|
| `--image_size` | 256 | Square output side in pixels (must be a multiple of 16). |
| `--num_inference_steps` | 35 | Euler steps for the flow-matching ODE. |
| `--cfg_scale` | 2.0 | Classifier-free guidance; `>1.0` enables CFG. |
| `--time_shift_alpha` | 4.0 | Time-shift in the flow schedule. **Must match training.** |
| `--batch_size` | 4 | Prompts per forward pass. |

### Architecture flags (must match the released checkpoint)

| Flag | Default | Notes |
|---|---|---|
| `--model_width` | 1024 | Hidden size. |
| `--model_head_dim` | 128 | → 8 attention heads (num_kv_heads = 8, no GQA in this config). |
| `--depth_stages` | 50 | Total depth = `depth_stages * blocks_per_stage * 2`. |
| `--blocks_per_stage` | 10 | Default config gives **1000** transformer blocks. |
| `--rope_base` | 10000 | 2-D RoPE base. |
| `--train_bias_and_rms` | off | Toggle if the checkpoint was trained with QKV bias + trainable RMSNorm. |

The default flags reproduce the released 1000-layer checkpoint.

## Third-party code

The Triton kernels under `kernels/` (`rmsnorm.py`, `swiglu.py`, `rope.py`) are
derived from [Unsloth](https://github.com/unslothai/unsloth). Each file carries
the upstream copyright header, license text, and a list of modifications. See
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md) for a top-level summary.

## Citation

A pre-print is on the way; check the HF repo for updates.
