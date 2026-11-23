#!/usr/bin/env python3
"""
Frostmarsh Tier 4: Few-Step Real-Time Diffusion / Swappable Multi-Style Client Renderer.

Demonstrates the Tier 4 superpower:
Three players in the EXACT SAME Frostmarsh world tick rendering completely different AAA art
styles client-side from the exact same projected 3D A+ state:
  1. Dark Fantasy Photoreal
  2. Studio Ghibli Painterly
  3. Cyberpunk Neon Tactical

Uses `stabilityai/sd-turbo` (real 1-4 step adversarial diffusion distilled model) on
NVIDIA GB10 GPU, with official `EulerDiscreteScheduler` (trailing timestep spacing):
- Benchmarks exact 1-step, 2-step, and 4-step img2img latency & FPS at 512x512
  (verifying exact UNet NFE = 1, 2, 4 via forward hook!) with BOTH standard SD VAE
  (`AutoencoderKL`) AND production real-time distilled VAE (`madebyollin/taesd`).
- Measures peak GPU VRAM usage.
- Measures temporal flicker across 8 consecutive live Frostmarsh 3D battle frames:
  Raw frame-by-frame img2img vs Optical-Flow-Warped + Motion-Gated Latent-Blended Stabilization.
- Generates `tier4_multistyle.png` (4-column showcase) and `tier4_metrics.json`.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

OUT_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-tiers-3-4")
PYDEPS = OUT_DIR / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
import torch
import torch.nn.functional as F

from diffusers import AutoPipelineForImage2Image, AutoencoderTiny


STYLES = [
    {
        "id": "dark_fantasy",
        "title": "Style 1: Dark Fantasy Photoreal",
        "subtitle": "Gritty RPG, volumetric firelight, wet stone, frost",
        "prompt": (
            "isometric aerial view dark fantasy RPG video game screenshot, photorealistic Diablo 4 and Unreal Engine 5 terrain, "
            "roaring realistic volumetric orange firelight and sparks, wet jagged dark basalt cliffs and crags, "
            "glowing frosted ice bridge over dark river, realistic dark evergreen pine forest, dramatic moody lighting, ultra detailed"
        ),
        "strength": 0.82,
        "seed": 115,
    },
    {
        "id": "ghibli_painterly",
        "title": "Style 2: Studio Ghibli Painterly",
        "subtitle": "Hand-painted anime, watercolor pines, sunlit river",
        "prompt": (
            "isometric bird-eye aerial view Studio Ghibli hand-painted anime landscape by Hayao Miyazaki, "
            "lush watercolor painted emerald pine forest, luminous shimmering azure river with crystal ice bridge, "
            "warm painterly golden orange fire flames, hand-painted rocks and mossy hills, sunlit anime scenery, vibrant gouache art"
        ),
        "strength": 0.82,
        "seed": 219,
    },
    {
        "id": "cyberpunk_neon",
        "title": "Style 3: Cyberpunk Neon Tactical",
        "subtitle": "Bioluminescent river, holo-ice grid, plasma wildfire",
        "prompt": (
            "isometric aerial cyberpunk neon tactical holographic battlefield map, "
            "bioluminescent glowing electric cyan neon river, holographic laser wireframe ice bridge grid, "
            "blazing magenta and neon orange plasma wildfire, dark wet obsidian synthwave terrain, "
            "glowing neon wireframe trees, Tron sci-fi aesthetic, ultra sharp"
        ),
        "strength": 0.84,
        "seed": 303,
    },
]


def load_font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL RGB -> [1, 3, H, W] float32 in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[1, 3, H, W] or [3, H, W] float in [0, 1] -> PIL RGB."""
    if t.dim() == 4:
        t = t[0]
    arr = (torch.clamp(t, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


# =============================================================================
# 1. Pure-PyTorch GPU Multi-Scale Dense Optical Flow & Warping
# =============================================================================
def _warp_tensor(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
    """
    Warp x [B, C, H, W] using backward optical flow flow_px [B, 2, H, W]
    where flow_px[b, 0] is dx (pixels) and flow_px[b, 1] is dy (pixels).
    """
    B, C, H, W = x.shape
    device = x.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    gx = xx.unsqueeze(0) + flow_px[:, 0]
    gy = yy.unsqueeze(0) + flow_px[:, 1]
    gx_n = 2.0 * (gx / max(W - 1, 1)) - 1.0
    gy_n = 2.0 * (gy / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx_n, gy_n], dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


@torch.no_grad()
def estimate_dense_optical_flow(img_prev: torch.Tensor, img_curr: torch.Tensor) -> torch.Tensor:
    """Fast 3-level coarse-to-fine Lucas-Kanade dense optical flow on GPU."""
    w_lum = torch.tensor([0.299, 0.587, 0.114], device=img_prev.device).view(1, 3, 1, 1)
    p0 = (img_prev * w_lum).sum(dim=1, keepdim=True)
    p1 = (img_curr * w_lum).sum(dim=1, keepdim=True)

    H, W = p0.shape[-2], p0.shape[-1]
    scales = [4, 2, 1]
    flow = torch.zeros((1, 2, H // scales[0], W // scales[0]), device=p0.device, dtype=torch.float32)

    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=p0.device).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=p0.device).view(1, 1, 3, 3) / 8.0

    for idx, sc in enumerate(scales):
        hs, ws = H // sc, W // sc
        if idx > 0:
            flow = F.interpolate(flow, size=(hs, ws), mode="bilinear", align_corners=True) * 2.0
        i0 = F.interpolate(p0, size=(hs, ws), mode="bilinear", align_corners=False)
        i1 = F.interpolate(p1, size=(hs, ws), mode="bilinear", align_corners=False)

        for _ in range(4):
            i0_w = _warp_tensor(i0, flow)
            Ix = F.conv2d(F.pad(i0_w, (1, 1, 1, 1), mode="replicate"), kx)
            Iy = F.conv2d(F.pad(i0_w, (1, 1, 1, 1), mode="replicate"), ky)
            It = i1 - i0_w

            win = 9
            pad = win // 2
            Sxx = F.avg_pool2d(F.pad(Ix * Ix, (pad, pad, pad, pad), mode="replicate"), win, stride=1) + 1e-3
            Syy = F.avg_pool2d(F.pad(Iy * Iy, (pad, pad, pad, pad), mode="replicate"), win, stride=1) + 1e-3
            Sxy = F.avg_pool2d(F.pad(Ix * Iy, (pad, pad, pad, pad), mode="replicate"), win, stride=1)
            Sxt = F.avg_pool2d(F.pad(Ix * It, (pad, pad, pad, pad), mode="replicate"), win, stride=1)
            Syt = F.avg_pool2d(F.pad(Iy * It, (pad, pad, pad, pad), mode="replicate"), win, stride=1)

            det = Sxx * Syy - Sxy * Sxy + 1e-5
            du = (Syy * Sxt - Sxy * Syt) / det
            dv = (Sxx * Syt - Sxy * Sxt) / det
            delta = torch.clamp(torch.cat([du, dv], dim=1), -4.0, 4.0)
            flow = flow + delta
            flow = F.avg_pool2d(F.pad(flow, (2, 2, 2, 2), mode="replicate"), 5, stride=1)

    return flow


# =============================================================================
# 2. SD-Turbo Real-Time Engine (Official EulerDiscreteScheduler + Optional TAESD)
# =============================================================================
class SDTurboClientRenderer:
    """
    Real-time client-side SD-Turbo img2img renderer:
    - Cached style prompt embeddings (computed once on style swap!)
    - Exact 1-step, 2-step, and 4-step official `EulerDiscreteScheduler` denoising loop
    - Supports both standard `AutoencoderKL` VAE and fast distilled `madebyollin/taesd` VAE
    - Optical-flow-warped noise + motion-gated latent-space EMA temporal stabilization
    """

    def __init__(self, device: torch.device, model_id: str = "stabilityai/sd-turbo"):
        self.device = device
        self.dtype = torch.float16
        print(f"[Tier 4] Loading {model_id} in fp16 on {torch.cuda.get_device_name(0)}...")
        t0 = time.perf_counter()
        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                dtype=self.dtype,
                variant="fp16",
            )
        except Exception:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                dtype=self.dtype,
            )
        self.pipe.to(device)
        if hasattr(self.pipe, "safety_checker"):
            self.pipe.safety_checker = None
        self.pipe.set_progress_bar_config(disable=True)

        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()
        self.scheduler = self.pipe.scheduler
        # Optimize memory format for Tensor Core convs
        self.unet.to(memory_format=torch.channels_last)
        self.vae.to(memory_format=torch.channels_last)
        self.load_time_s = time.perf_counter() - t0
        print(f"[Tier 4] Loaded {model_id} in {self.load_time_s:.1f}s")

        # Load real-time distilled TAESD VAE (~9.8 MB) for low-latency streaming benchmark
        self.taesd = None
        try:
            self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype).to(device).eval()
            self.taesd.to(memory_format=torch.channels_last)
            print("[Tier 4] Loaded madebyollin/taesd fast distilled VAE (9.8 MB)")
        except Exception as ex:
            print(f"[Tier 4] TAESD unavailable ({ex}); continuing with standard VAE only")

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        tok = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.pipe.text_encoder(tok)[0].to(dtype=self.dtype)

    @torch.no_grad()
    def encode_image_to_latent(self, img_01: torch.Tensor, use_taesd: bool = False) -> torch.Tensor:
        x = (img_01 * 2.0 - 1.0).to(dtype=self.dtype).contiguous(memory_format=torch.channels_last)
        if use_taesd and self.taesd is not None:
            return self.taesd.encode(x).latents * self.vae.config.scaling_factor
        return self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent_to_image(self, lat: torch.Tensor, use_taesd: bool = False) -> torch.Tensor:
        z = (lat / self.vae.config.scaling_factor).contiguous(memory_format=torch.channels_last)
        if use_taesd and self.taesd is not None:
            img = self.taesd.decode(z).sample
        else:
            img = self.vae.decode(z, return_dict=False)[0]
        return torch.clamp((img.float() + 1.0) * 0.5, 0.0, 1.0)

    def _prepare_exact_timesteps(self, num_unet_steps: int, strength: float) -> torch.Tensor:
        """
        Configure `EulerDiscreteScheduler` (trailing spacing) so the active slice has
        EXACTLY `num_unet_steps` denoising steps starting near `strength`.
        """
        # Solve total_steps so int(total_steps * strength) == num_unet_steps
        best_total = num_unet_steps
        for cand in range(num_unet_steps, num_unet_steps * 5 + 2):
            if min(int(cand * strength), cand) == num_unet_steps:
                best_total = cand
                break
        self.scheduler.set_timesteps(best_total, device=self.device)
        init_ts = min(int(best_total * strength), best_total)
        t_start = max(best_total - init_ts, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]
        # Reset internal step index to start at t_start so scheduler.step uses exact matching sigmas!
        self.scheduler._step_index = None
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        assert len(timesteps) == num_unet_steps, f"Got {len(timesteps)} timesteps, expected {num_unet_steps}"
        return timesteps

    @torch.no_grad()
    def render_img2img_exact_steps(
        self,
        img_01: torch.Tensor,
        prompt_embeds: torch.Tensor,
        num_unet_steps: int,
        strength: float = 0.78,
        noise: torch.Tensor | None = None,
        seed: int = 108,
        use_taesd: bool = False,
        return_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | torch.Tensor:
        timesteps = self._prepare_exact_timesteps(num_unet_steps, strength)

        z0 = self.encode_image_to_latent(img_01, use_taesd=use_taesd)
        if noise is None:
            gen = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(z0.shape, generator=gen, device=self.device, dtype=self.dtype)

        latents = self.scheduler.add_noise(z0, noise, timesteps[0:1])

        nfe = 0
        for t in timesteps:
            latent_in = self.scheduler.scale_model_input(latents, t).contiguous(memory_format=torch.channels_last)
            noise_pred = self.unet(latent_in, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            nfe += 1

        out_img = self.decode_latent_to_image(latents, use_taesd=use_taesd)
        if return_latent:
            return out_img, latents, nfe
        return out_img


# =============================================================================
# 3. Benchmarks: 1 / 2 / 4 Step Latency (Standard VAE & TAESD) + Temporal Flicker
# =============================================================================
@torch.no_grad()
def benchmark_step_latencies(
    engine: SDTurboClientRenderer,
    sample_img: torch.Tensor,
    prompt_embeds: torch.Tensor,
) -> dict:
    device = engine.device

    def time_cuda(fn, warmup=10, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ms = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ms.append(s.elapsed_time(e))
        arr = np.array(ms)
        return {
            "median_ms": round(float(np.median(arr)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "fps": round(1000.0 / float(np.median(arr)), 2),
        }

    vae_enc_std = time_cuda(lambda: engine.encode_image_to_latent(sample_img, use_taesd=False))
    z_sample = engine.encode_image_to_latent(sample_img, use_taesd=False)
    vae_dec_std = time_cuda(lambda: engine.decode_latent_to_image(z_sample, use_taesd=False))

    gen = torch.Generator(device=device).manual_seed(108)
    fixed_noise = torch.randn(z_sample.shape, generator=gen, device=device, dtype=engine.dtype)

    results: dict = {
        "standard_vae": {
            "vae_encode_ms": vae_enc_std["median_ms"],
            "vae_decode_ms": vae_dec_std["median_ms"],
        }
    }

    for k_steps in (1, 2, 4):
        call_counter = [0]
        handle = engine.unet.register_forward_hook(lambda m, inp, out: call_counter.__setitem__(0, call_counter[0] + 1))
        _, _, nfe_rep = engine.render_img2img_exact_steps(
            sample_img, prompt_embeds, num_unet_steps=k_steps, strength=0.78, noise=fixed_noise,
            use_taesd=False, return_latent=True,
        )
        handle.remove()
        assert call_counter[0] == k_steps and nfe_rep == k_steps

        stats = time_cuda(
            lambda ks=k_steps: engine.render_img2img_exact_steps(
                sample_img, prompt_embeds, num_unet_steps=ks, strength=0.78, noise=fixed_noise, use_taesd=False
            ),
            warmup=8, iters=30,
        )
        unet_est_ms = max(0.1, round(stats["median_ms"] - vae_enc_std["median_ms"] - vae_dec_std["median_ms"], 2))
        stats["verified_unet_nfe"] = k_steps
        stats["unet_est_ms"] = unet_est_ms
        stats["unet_only_fps"] = round(1000.0 / unet_est_ms, 2)
        results["standard_vae"][f"steps_{k_steps}"] = stats
        print(
            f"[Tier 4] SD-Turbo 512x512 (Std VAE) {k_steps}-step [NFE={k_steps}]: "
            f"{stats['median_ms']:.2f} ms ({stats['fps']:.1f} FPS e2e | UNet {unet_est_ms:.2f} ms = {stats['unet_only_fps']:.1f} FPS)"
        )

    if engine.taesd is not None:
        tae_enc = time_cuda(lambda: engine.encode_image_to_latent(sample_img, use_taesd=True))
        z_t = engine.encode_image_to_latent(sample_img, use_taesd=True)
        tae_dec = time_cuda(lambda: engine.decode_latent_to_image(z_t, use_taesd=True))
        results["taesd_fast_vae"] = {
            "vae_encode_ms": tae_enc["median_ms"],
            "vae_decode_ms": tae_dec["median_ms"],
        }
        for k_steps in (1, 2, 4):
            stats_t = time_cuda(
                lambda ks=k_steps: engine.render_img2img_exact_steps(
                    sample_img, prompt_embeds, num_unet_steps=ks, strength=0.78, noise=fixed_noise, use_taesd=True
                ),
                warmup=8, iters=30,
            )
            stats_t["verified_unet_nfe"] = k_steps
            results["taesd_fast_vae"][f"steps_{k_steps}"] = stats_t
            print(
                f"[Tier 4] SD-Turbo 512x512 (TAESD)   {k_steps}-step [NFE={k_steps}]: "
                f"{stats_t['median_ms']:.2f} ms ({stats_t['fps']:.1f} FPS e2e | TAESD enc={tae_enc['median_ms']:.1f}ms, dec={tae_dec['median_ms']:.1f}ms)"
            )

    return results


@torch.no_grad()
def benchmark_temporal_stability(
    engine: SDTurboClientRenderer,
    frame_paths: list[str],
    prompt_embeds: torch.Tensor,
    num_steps: int = 2,
    strength: float = 0.72,
    seed: int = 108,
) -> dict:
    """
    Evaluate temporal flicker across 8 consecutive live Frostmarsh 3D battle frames:
      - Tier 3 3DGS input baseline (deterministic geometric projection)
      - Raw img2img (independent per-frame noise)
      - Fixed-seed img2img (static noise, no flow warp)
      - Stabilized img2img (Optical-Flow noise warping + Motion-Gated Latent-Space EMA)
    """
    device = engine.device
    raw_frames = [
        pil_to_tensor(ImageEnhance.Contrast(Image.open(p).convert("RGB")).enhance(1.15), device)
        for p in frame_paths
    ]
    n_frames = len(raw_frames)

    flows = []
    flow_times_ms = []
    for i in range(1, n_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fl = estimate_dense_optical_flow(raw_frames[i - 1], raw_frames[i])
        torch.cuda.synchronize()
        flow_times_ms.append((time.perf_counter() - t0) * 1000.0)
        flows.append(fl)

    # Mode A1: Raw frame-by-frame img2img (fresh seed per frame)
    raw_outs = [
        engine.render_img2img_exact_steps(
            raw_frames[i], prompt_embeds, num_unet_steps=num_steps, strength=strength, seed=seed + i * 29
        )
        for i in range(n_frames)
    ]

    # Mode A2: Fixed-seed frame-by-frame img2img (no flow warp / no latent blending)
    raw_fixed_outs = [
        engine.render_img2img_exact_steps(
            raw_frames[i], prompt_embeds, num_unet_steps=num_steps, strength=strength, seed=seed
        )
        for i in range(n_frames)
    ]

    # Mode B: Optical-Flow Noise Warp + Motion-Gated Latent EMA Stabilization
    stab_outs = []
    gen = torch.Generator(device=device).manual_seed(seed)
    prev_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=engine.dtype)
    prev_lat = None
    stab_overhead_ms = []

    for i in range(n_frames):
        if i == 0:
            out_i, lat_i, _ = engine.render_img2img_exact_steps(
                raw_frames[i], prompt_embeds, num_unet_steps=num_steps, strength=strength,
                noise=prev_noise, return_latent=True,
            )
            prev_lat = lat_i
            stab_outs.append(out_i)
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fl_512 = flows[i - 1]
            fl_64 = F.interpolate(fl_512, size=(64, 64), mode="bilinear", align_corners=True) / 8.0

            # Flow-warped noise (keeps diffusion grain attached to 3D world surfaces during camera motion!)
            warped_noise = _warp_tensor(prev_noise.float(), fl_64)
            eps_fresh = torch.randn_like(warped_noise)
            beta_n = 0.06
            cur_noise = math.sqrt(1.0 - beta_n) * warped_noise + math.sqrt(beta_n) * eps_fresh
            cur_noise = (cur_noise / torch.clamp(cur_noise.std(dim=(2, 3), keepdim=True), min=0.5)).to(engine.dtype)

            # Motion/occlusion gate from 3DGS geometry
            warped_in = _warp_tensor(raw_frames[i - 1], fl_512)
            in_err = (raw_frames[i] - warped_in).abs().mean(dim=1, keepdim=True)
            in_err_64 = F.interpolate(in_err, size=(64, 64), mode="bilinear", align_corners=False)
            ema_w = (0.62 * torch.exp(-in_err_64 * 10.0)).clamp(0.18, 0.64).to(engine.dtype)
            warped_prev_lat = _warp_tensor(prev_lat.float(), fl_64).to(engine.dtype)
            torch.cuda.synchronize()
            t_warp = (time.perf_counter() - t0) * 1000.0

            _, lat_raw, _ = engine.render_img2img_exact_steps(
                raw_frames[i], prompt_embeds, num_unet_steps=num_steps, strength=strength,
                noise=cur_noise, return_latent=True,
            )

            lat_stab = (1.0 - ema_w) * lat_raw + ema_w * warped_prev_lat
            out_i = engine.decode_latent_to_image(lat_stab)
            stab_overhead_ms.append(t_warp)

            prev_noise = cur_noise
            prev_lat = lat_stab
            stab_outs.append(out_i)

    def eval_flicker(seq: list[torch.Tensor]) -> dict:
        raw_l1s, warp_l1s, warp_rmses, warp_psnrs = [], [], [], []
        for i in range(1, len(seq)):
            cur = seq[i][:, :, 14:-14, 14:-14]
            prev = seq[i - 1][:, :, 14:-14, 14:-14]
            warped = _warp_tensor(seq[i - 1], flows[i - 1])[:, :, 14:-14, 14:-14]
            raw_l1s.append(float((cur - prev).abs().mean().item()))
            w_l1 = float((cur - warped).abs().mean().item())
            w_mse = float(((cur - warped) ** 2).mean().item())
            warp_l1s.append(w_l1)
            warp_rmses.append(math.sqrt(w_mse))
            warp_psnrs.append(-10.0 * math.log10(max(w_mse, 1e-8)))
        return {
            "unwarped_l1": round(float(np.mean(raw_l1s)), 5),
            "warped_flicker_l1": round(float(np.mean(warp_l1s)), 5),
            "warped_flicker_l1_255": round(float(np.mean(warp_l1s)) * 255.0, 2),
            "warped_flicker_rmse": round(float(np.mean(warp_rmses)), 5),
            "temporal_psnr_db": round(float(np.mean(warp_psnrs)), 2),
        }

    m_input = eval_flicker(raw_frames)
    m_raw = eval_flicker(raw_outs)
    m_fixed = eval_flicker(raw_fixed_outs)
    m_stab = eval_flicker(stab_outs)

    red_raw = (1.0 - m_stab["warped_flicker_l1"] / max(m_raw["warped_flicker_l1"], 1e-6)) * 100.0
    red_fixed = (1.0 - m_stab["warped_flicker_l1"] / max(m_fixed["warped_flicker_l1"], 1e-6)) * 100.0

    print(
        f"[Tier 4] Temporal Flicker (E_warp L1): "
        f"Tier3-3DGS={m_input['warped_flicker_l1']:.4f} ({m_input['temporal_psnr_db']:.1f} dB) | "
        f"Raw img2img={m_raw['warped_flicker_l1']:.4f} ({m_raw['temporal_psnr_db']:.1f} dB) | "
        f"Fixed-seed={m_fixed['warped_flicker_l1']:.4f} ({m_fixed['temporal_psnr_db']:.1f} dB) | "
        f"Flow+Latent-EMA={m_stab['warped_flicker_l1']:.4f} ({m_stab['temporal_psnr_db']:.1f} dB) -> "
        f"-{red_raw:.1f}% vs raw, -{red_fixed:.1f}% vs fixed-seed"
    )
    return {
        "num_frames": n_frames,
        "step_count_tested": num_steps,
        "strength_tested": strength,
        "input_3dgs_baseline": m_input,
        "raw_img2img_unseeded": m_raw,
        "raw_img2img_fixed_seed": m_fixed,
        "stabilized_flow_latent_ema": m_stab,
        "flicker_reduction_vs_raw_pct": round(red_raw, 1),
        "flicker_reduction_vs_fixed_seed_pct": round(red_fixed, 1),
        "optical_flow_median_ms": round(float(np.median(flow_times_ms)), 2),
        "latent_warp_blend_overhead_ms": round(float(np.median(stab_overhead_ms)), 2),
    }


# =============================================================================
# 4. Generate 4-Column Multi-Style Showcase (`tier4_multistyle.png`)
# =============================================================================
@torch.no_grad()
def generate_multistyle_showcase(
    engine: SDTurboClientRenderer,
    base_frame_path: str,
    step_bench: dict,
    temp_bench: dict,
    showcase_steps: int = 4,
) -> dict:
    device = engine.device
    raw_pil = Image.open(base_frame_path).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    cond_pil = ImageEnhance.Contrast(raw_pil).enhance(1.18)
    cond_t = pil_to_tensor(cond_pil, device)

    styled_pils = []
    style_meta = []
    for st in STYLES:
        emb = engine.encode_prompt(st["prompt"])
        out_t, _, nfe = engine.render_img2img_exact_steps(
            cond_t, emb, num_unet_steps=showcase_steps, strength=st["strength"],
            seed=st["seed"], use_taesd=False, return_latent=True,
        )
        out_pil = tensor_to_pil(out_t)
        indiv_path = OUT_DIR / f"tier4_style_{st['id']}.png"
        out_pil.save(indiv_path)
        styled_pils.append((st["title"], st["subtitle"], out_pil))
        style_meta.append({
            "id": st["id"],
            "title": st["title"],
            "path": str(indiv_path),
            "steps": nfe,
            "strength": st["strength"],
        })

    cols = [
        ("Raw 3D A+ Projection (Tier 3)", "Same server tick: river, ice bridge, fire, pines", raw_pil),
        *styled_pils,
    ]
    pad = 16
    header_h = 74
    cap_h = 46
    footer_h = 52
    cw, ch = 512, 512
    W_tot = pad * 5 + cw * 4
    H_tot = header_h + pad * 2 + ch + cap_h + footer_h

    canvas = Image.new("RGB", (W_tot, H_tot), (12, 16, 24))
    draw = ImageDraw.Draw(canvas)
    f_title = load_font(23)
    f_sub = load_font(14)
    f_cap = load_font(15)
    f_cap2 = load_font(13)

    draw.text(
        (pad, 12),
        "FROSTMARSH TIER 4 — Swappable Multi-Style Client Renderer (SD-Turbo on Same Server A+ State)",
        fill=(240, 245, 255),
        font=f_title,
    )
    std = step_bench["standard_vae"]
    tae = step_bench.get("taesd_fast_vae")
    if tae:
        sub_txt = (
            f"GB10 512x512 FP16 | TAESD Fast VAE: 1-step {tae['steps_1']['median_ms']:.1f}ms ({tae['steps_1']['fps']:.1f} FPS), "
            f"2-step {tae['steps_2']['median_ms']:.1f}ms ({tae['steps_2']['fps']:.1f} FPS), "
            f"4-step {tae['steps_4']['median_ms']:.1f}ms ({tae['steps_4']['fps']:.1f} FPS) | "
            f"Std VAE: 1-step {std['steps_1']['median_ms']:.1f}ms ({std['steps_1']['fps']:.1f} FPS) | 100% Client-Side Style Swap"
        )
    else:
        sub_txt = (
            f"GB10 512x512 FP16 | 1-step {std['steps_1']['median_ms']:.1f}ms ({std['steps_1']['fps']:.1f} FPS) | "
            f"2-step {std['steps_2']['median_ms']:.1f}ms ({std['steps_2']['fps']:.1f} FPS) | "
            f"4-step {std['steps_4']['median_ms']:.1f}ms ({std['steps_4']['fps']:.1f} FPS) | 100% Client-Side Style Swap"
        )
    draw.text((pad, 44), sub_txt, fill=(145, 205, 255), font=f_sub)

    for idx, (title, sub, pil_im) in enumerate(cols):
        x0 = pad + idx * (cw + pad)
        y0 = header_h + pad
        border_col = (80, 140, 220) if idx == 0 else (54, 68, 96)
        draw.rectangle([x0 - 2, y0 - 2, x0 + cw + 1, y0 + ch + cap_h + 1], outline=border_col, width=2)
        canvas.paste(pil_im, (x0, y0))
        draw.rectangle([x0, y0 + ch, x0 + cw, y0 + ch + cap_h], fill=(20, 26, 38))
        draw.text((x0 + 10, y0 + ch + 6), title, fill=(235, 242, 255), font=f_cap)
        draw.text((x0 + 10, y0 + ch + 26), sub, fill=(155, 175, 200), font=f_cap2)

    red_raw = temp_bench["flicker_reduction_vs_raw_pct"]
    red_fix = temp_bench["flicker_reduction_vs_fixed_seed_pct"]
    e_raw = temp_bench["raw_img2img_unseeded"]["warped_flicker_l1"]
    e_stab = temp_bench["stabilized_flow_latent_ema"]["warped_flicker_l1"]
    psnr_raw = temp_bench["raw_img2img_unseeded"]["temporal_psnr_db"]
    psnr_stab = temp_bench["stabilized_flow_latent_ema"]["temporal_psnr_db"]
    draw.text(
        (pad, H_tot - footer_h + 14),
        f"Temporal Stabilization (8 live frames, Optical-Flow Noise Warp + Motion-Gated Latent EMA): "
        f"E_warp L1 {e_raw:.4f} ({psnr_raw:.1f} dB) -> {e_stab:.4f} ({psnr_stab:.1f} dB) | -{red_raw:.1f}% vs raw (-{red_fix:.1f}% vs fixed-seed)",
        fill=(160, 220, 185),
        font=f_sub,
    )

    out_path = OUT_DIR / "tier4_multistyle.png"
    canvas.save(out_path)
    print(f"[Tier 4] Saved 4-column multi-style showcase -> {out_path}")
    return {"showcase_path": str(out_path), "styles": style_meta}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Tier 4 requires CUDA"
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    tier3_json = OUT_DIR / "tier3_metrics.json"
    assert tier3_json.exists(), f"Missing {tier3_json} — run tier3_3dgs.py first!"
    t3 = json.loads(tier3_json.read_text())
    frame_paths = t3["visuals"]["battle_frame_paths"]
    hero_frame = frame_paths[2]

    print(f"=== Frostmarsh Tier 4: Few-Step Real-Time Multi-Style Diffusion on {torch.cuda.get_device_name(0)} ===")
    engine = SDTurboClientRenderer(device, model_id="stabilityai/sd-turbo")

    emb_dark = engine.encode_prompt(STYLES[0]["prompt"])
    sample_t = pil_to_tensor(ImageEnhance.Contrast(Image.open(hero_frame).convert("RGB")).enhance(1.18), device)

    # 1. Exact 1-step, 2-step, 4-step GPU latency & FPS benchmark (Standard VAE + TAESD)
    step_bench = benchmark_step_latencies(engine, sample_t, emb_dark)

    # 2. Temporal flicker benchmark across 8 consecutive live Frostmarsh frames
    temp_bench = benchmark_temporal_stability(engine, frame_paths, emb_dark, num_steps=2, strength=0.72, seed=108)

    # 3. Generate 4-column multi-style showcase
    vis_info = generate_multistyle_showcase(engine, hero_frame, step_bench, temp_bench, showcase_steps=4)

    peak_alloc_mb = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 1)
    peak_res_mb = round(torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0), 1)
    print(f"[Tier 4] Peak GPU VRAM: allocated={peak_alloc_mb:.1f} MB, reserved={peak_res_mb:.1f} MB")

    tier4_results = {
        "device": torch.cuda.get_device_name(0),
        "model_id": "stabilityai/sd-turbo",
        "scheduler": "EulerDiscreteScheduler (trailing, epsilon)",
        "dtype": "float16",
        "resolution": "512x512",
        "model_load_time_s": round(engine.load_time_s, 2),
        "vram_peak_allocated_mb": peak_alloc_mb,
        "vram_peak_reserved_mb": peak_res_mb,
        "latency_fps": step_bench,
        "temporal_stability": temp_bench,
        "visuals": vis_info,
    }
    out_json = OUT_DIR / "tier4_metrics.json"
    out_json.write_text(json.dumps(tier4_results, indent=2))
    print(f"[Tier 4] Done! Metrics written to {out_json}")


if __name__ == "__main__":
    main()
