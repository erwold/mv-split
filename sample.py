#!/usr/bin/env python3
"""
Sample images from text prompts using the trained 1000-layer DiT.

Usage:
    # Pick a specific line from the prompts file (1-indexed)
    python sample.py --line 5

    # Randomly pick N prompts (reproducible with --seed)
    python sample.py --num_samples 4 --seed 42

    # Generate images for all prompts in the file
    python sample.py --all

    # Use your own prompt
    python sample.py --prompt "a red panda climbing a bamboo stalk"
"""
import os
import argparse
import random
import logging
import torch
from PIL import Image
from tqdm import tqdm

from dit import DiT
from text_encoder import Qwen3TextEncoder
from vae import load_flux2_ae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =================================================================
# Multi-step ODE sampling (Flow Matching / Rectified Flow)
# =================================================================
@torch.no_grad()
def perform_multi_step_sampling(
    dit_model,
    latents,
    captions,
    cfg_scale,
    dtype,
    device,
    text_encoder,
    num_inference_steps=35,
    alpha=4.0,
):
    """
    Euler-method ODE sampler for Flow Matching models.

    Args:
        dit_model: The DiT velocity model.
        latents: Initial noise x_1, shape (B, C, H, W).
        captions: List of text prompts (length B).
        cfg_scale: Classifier-Free Guidance scale (>1.0 enables CFG).
        dtype: Compute dtype for the model forward pass (e.g. bfloat16).
        device: Torch device.
        text_encoder: Qwen3TextEncoder instance.
        num_inference_steps: Number of ODE solver steps.
        alpha: Time-shift parameter, must match training value.

    Returns:
        Generated latents x_0, shape (B, C, H, W).
    """
    num_samples = latents.shape[0]
    if text_encoder is None:
        raise RuntimeError("TextEncoder not provided for sampling.")

    # 1) Build prompt batch (cond + uncond if CFG)
    if cfg_scale > 1.0:
        all_prompts = list(captions) + [""] * num_samples
    else:
        all_prompts = list(captions)

    # 2) Encode text once
    context_emb, _ = text_encoder.encode(all_prompts)
    if cfg_scale > 1.0:
        context_emb_cond, context_emb_uncond = context_emb.chunk(2, dim=0)

    # 3) Euler integration in float32 from t=1 (noise) to t=0 (data)
    latents = latents.to(torch.float32)
    for i in range(num_inference_steps, 0, -1):
        t      = i / num_inference_steps
        t_next = (i - 1) / num_inference_steps

        # Time-shift (must match training)
        t_shifted      = t      * alpha / (1 + (alpha - 1) * t)
        t_next_shifted = t_next * alpha / (1 + (alpha - 1) * t_next)
        dt = t_shifted - t_next_shifted

        if cfg_scale > 1.0:
            v_cond   = dit_model(latents.to(dtype), context=context_emb_cond)
            v_uncond = dit_model(latents.to(dtype), context=context_emb_uncond)
            v_pred   = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v_pred = dit_model(latents.to(dtype), context=context_emb)

        latents = latents + dt * v_pred.to(torch.float32)

    return latents


# =================================================================
# I/O helpers
# =================================================================
def load_prompts(path):
    """Load prompts file (one prompt per line, blank lines and # comments ignored)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    logger.info(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def select_jobs(args, prompts):
    """Decide which prompts to generate. Returns list of (save_name, prompt)."""
    if args.prompt is not None:
        return [("custom", args.prompt)]

    if args.line is not None:
        if not (1 <= args.line <= len(prompts)):
            raise ValueError(f"--line {args.line} out of range [1, {len(prompts)}]")
        return [(f"line_{args.line:04d}", prompts[args.line - 1])]

    if args.all:
        return [(f"line_{i+1:04d}", p) for i, p in enumerate(prompts)]

    n = args.num_samples if args.num_samples is not None else 1
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(prompts)), k=min(n, len(prompts)))
    return [(f"line_{i+1:04d}", prompts[i]) for i in indices]


# =================================================================
# Model loading
# =================================================================
def load_models(device, args):
    dtype = torch.bfloat16

    logger.info("Loading VAE ...")
    vae = load_flux2_ae(weight_path=args.flux_vae_path, device=device, dtype=dtype).eval().requires_grad_(False)

    logger.info("Loading Qwen3 text encoder ...")
    text_encoder = Qwen3TextEncoder(model_name=args.qwen_model_path, device=device, dtype=dtype).eval()

    total_depth = args.depth_stages * args.blocks_per_stage * 2
    logger.info(f"Building DiT (depth={args.depth_stages}*{args.blocks_per_stage}*2={total_depth}) ...")
    dit = DiT(
        in_channels=128, patch_size=1,
        hidden_size=args.model_width, qkv_bias=args.train_bias_and_rms,
        depth=total_depth, block_pattern=["attn"],
        num_heads=args.model_width // args.model_head_dim,
        num_kv_heads=8, mlp_hidden_dim=3072,
        context_dim=text_encoder.hidden_size,
        use_rope=True, rope_base=args.rope_base,
    )

    logger.info(f"Loading checkpoint: {args.checkpoint_path}")
    sd = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in sd: sd = sd["model"]
    elif "state_dict" in sd: sd = sd["state_dict"]
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    if missing:    logger.warning(f"Missing {len(missing)} keys, e.g. {missing[:3]}")
    if unexpected: logger.warning(f"Unexpected {len(unexpected)} keys, e.g. {unexpected[:3]}")

    dit = dit.to(device).to(dtype).eval().requires_grad_(False)
    return dit, vae, text_encoder


@torch.no_grad()
def sample_batch(dit, vae, text_encoder, captions, args, device):
    dtype = torch.bfloat16
    bs = len(captions)
    latent_h = latent_w = args.image_size // 16
    latents = torch.randn((bs, 128, latent_h, latent_w), device=device, dtype=torch.float32)

    gen_latents = perform_multi_step_sampling(
        dit_model=dit, latents=latents, captions=captions,
        cfg_scale=args.cfg_scale, dtype=dtype, device=device,
        text_encoder=text_encoder,
        num_inference_steps=args.num_inference_steps,
        alpha=args.time_shift_alpha,
    )
    images = vae.decode(gen_latents.to(dtype))
    images = (images.mul(0.5).add(0.5).mul(255.0).clamp(0, 255)
              .type(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
    return images


# =================================================================
# Main
# =================================================================
def main():
    p = argparse.ArgumentParser(description="Sample images from text prompts.")

    # Paths
    p.add_argument("--checkpoint_path", type=str, default="/workspace/model.pt")
    p.add_argument("--prompts_file",    type=str, default="./sample_prompts.txt",
                   help="Text file with one prompt per line.")
    p.add_argument("--flux_vae_path",   type=str, default="/workspace/flux2_ae.safetensors")
    p.add_argument("--qwen_model_path", type=str, default="/workspace/qwen3")
    p.add_argument("--output_dir",      type=str, default="./samples")

    # Prompt selection (priority: --prompt > --line > --all > --num_samples)
    p.add_argument("--prompt",      type=str, default=None, help="Use a custom prompt directly.")
    p.add_argument("--line",        type=int, default=None, help="Pick a specific line (1-indexed).")
    p.add_argument("--all",         action="store_true",   help="Generate every prompt in the file.")
    p.add_argument("--num_samples", type=int, default=None, help="Randomly sample N prompts.")
    p.add_argument("--seed",        type=int, default=None, help="Random seed (controls prompt pick + noise).")

    # Model architecture (must match training!)
    p.add_argument("--model_width",        type=int, default=1024)
    p.add_argument("--model_head_dim",     type=int, default=128)
    p.add_argument("--depth_stages",       type=int, default=50,
                   help="depth_stages * blocks_per_stage * 2 = total depth. Default = 1000-layer config.")
    p.add_argument("--blocks_per_stage",   type=int, default=10)
    p.add_argument("--rope_base",          type=int, default=10000)
    p.add_argument("--train_bias_and_rms", action="store_true")

    # Sampling
    p.add_argument("--image_size",          type=int,   default=256)
    p.add_argument("--num_inference_steps", type=int,   default=35)
    p.add_argument("--cfg_scale",           type=float, default=2.0)
    p.add_argument("--time_shift_alpha",    type=float, default=4.0)
    p.add_argument("--batch_size",          type=int,   default=4)

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # 1) Load prompts and decide what to generate
    prompts = [] if args.prompt is not None else load_prompts(args.prompts_file)
    jobs = select_jobs(args, prompts)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Total jobs: {len(jobs)}, output_dir: {args.output_dir}")

    # 2) Load models
    dit, vae, text_encoder = load_models(device, args)
    logger.info(f"Sampling: size={args.image_size}, steps={args.num_inference_steps}, "
                f"cfg={args.cfg_scale}, alpha={args.time_shift_alpha}, bs={args.batch_size}")

    # 3) Run inference
    meta_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as fmeta:
        for start in tqdm(range(0, len(jobs), args.batch_size), desc="Generating"):
            chunk = jobs[start:start + args.batch_size]
            names    = [n for n, _ in chunk]
            captions = [c for _, c in chunk]

            try:
                images = sample_batch(dit, vae, text_encoder, captions, args, device)
            except Exception as e:
                logger.error(f"Batch starting at {start} failed: {e}", exc_info=True)
                continue

            for name, cap, img_np in zip(names, captions, images):
                save_path = os.path.join(args.output_dir, f"{name}.png")
                Image.fromarray(img_np).save(save_path)
                fmeta.write(f'{{"name": "{name}", "prompt": {repr(cap)}}}\n')

    logger.info(f"✅ Done. {len(jobs)} images -> {args.output_dir}")
    logger.info(f"   metadata: {meta_path}")


if __name__ == "__main__":
    main()