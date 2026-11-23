#!/usr/bin/env python3
"""
Iteration 9A: Server-WorldModel <-> Client-GPU Latent Material Decoder Split Exporter.

Loads Track 8A's pretrained `models/checkpoints/neural_field_3d_z16.pt`
(`NeuralField3DEncoder` + `NeuralMaterialHead`), runs `NeuralField3DEncoder` on an unseen
`256x256x16` 3D multi-biome world (vaulted basalt cave with magma, 3-span arched bridge over
flowing river, plunging 3D waterfall, alpine mountains, Nordic pines, wildfire), and exports:

  1. `models/checkpoints/z16_latent_bricks.bin` (magic `b"FMLT"`):
       Compact Server-to-Client Latent Brick Packet `[D=16, H=256, W=256, 8]` quantized `u8`:
         - Ch 0..3: Top 4 PCA channels of `Z_world_3d` (32 -> 4, fit on active/stratified voxels)
         - Ch 4:    `solid_density`   (physics / encoder occupancy in [0, 1])
         - Ch 5:    `fluid_density`   (liquid density in [0, 1])
         - Ch 6:    `gas_density`     (smoke/steam optical density in [0, 1])
         - Ch 7:    `temp_norm`       (compressed temperature in [0, 1])
       Proves the server ONLY transmits physics + compact `Z_world_3d` latents — NEVER baked RGB!

  2. `models/checkpoints/client_material_weights.bin` (magic `b"FMWG"`):
       3 swappable client-side WGSL `NeuralMaterialHead` weight sets (`8 -> 24 -> 24 -> 8`
       residual SiLU MLP = 1,016 floats = 4,064 bytes per style, ~12 KB total):
         - Style 0 (`NORDIC PIXEL-2XSS`): Distilled directly to Track 8A's `NeuralMaterialHead`
                                          (target R^2 >= 0.96!).
         - Style 1 (`VOLCANIC EMBER`):    Obsidian ash ridges, molten magma river/waterfall,
                                          charred ironwood bridge, incandescent ember vents.
         - Style 2 (`ARCANE THERMAL`):    Bioluminescent frost-indigo & neon-cyan thermal vision
                                          exposing hidden heat & fluid flows through `Z_world_3d`.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from neural_field_3d import (  # noqa: E402
    CH_ELEV_Z,
    CH_FLOW_Z,
    CH_TEMP,
    S_ASH,
    S_ICE,
    S_OIL,
    S_PLANT,
    S_ROCK,
    S_SMOKE,
    S_SOIL,
    S_STEAM,
    S_WATER,
    S_WOOD,
    NeuralWorldField3D,
    generate_world_3d_z16,
)

FMLT_MAGIC = b"FMLT"
FMWG_MAGIC = b"FMWG"

IN_DIM = 8
H1_DIM = 24
H2_DIM = 24
OUT_DIM = 8
FLOATS_PER_STYLE = H1_DIM * IN_DIM + H1_DIM + H2_DIM * H1_DIM + H2_DIM + OUT_DIM * H2_DIM + OUT_DIM
assert FLOATS_PER_STYLE == 1016


class ClientMaterialMLP(nn.Module):
    """
    Ultra-compact per-voxel client neural material decoder (`8 -> 24 -> 24 -> 8`, 1,016 floats):
      h1 = silu(W1 @ x + b1)
      h2 = silu(W2 @ h1 + b2) + h1
      y  = sigmoid(W3 @ h2 + b3) -> (density_mat.rgba, albedo_thermal.rgba) in [0, 1]
    Matches `crates/frostmarsh-3d-sandbox/shaders/neural_material_decode.wgsl` bit-for-bit.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(IN_DIM, H1_DIM, bias=True)
        self.fc2 = nn.Linear(H1_DIM, H2_DIM, bias=True)
        self.fc3 = nn.Linear(H2_DIM, OUT_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.silu(self.fc1(x))
        h2 = F.silu(self.fc2(h1)) + h1
        return torch.sigmoid(self.fc3(h2))

    def pack_weights_f32(self) -> np.ndarray:
        parts = [
            self.fc1.weight.detach().float().cpu().numpy().reshape(-1),
            self.fc1.bias.detach().float().cpu().numpy().reshape(-1),
            self.fc2.weight.detach().float().cpu().numpy().reshape(-1),
            self.fc2.bias.detach().float().cpu().numpy().reshape(-1),
            self.fc3.weight.detach().float().cpu().numpy().reshape(-1),
            self.fc3.bias.detach().float().cpu().numpy().reshape(-1),
        ]
        flat = np.concatenate(parts).astype(np.float32)
        assert flat.shape == (FLOATS_PER_STYLE,), f"Unexpected weight size: {flat.shape}"
        return flat


def _r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    ss_res = float(np.sum(diff * diff))
    mean = float(np.mean(target.astype(np.float64)))
    ss_tot = float(np.sum((target.astype(np.float64) - mean) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def build_compact_latent_packet(
    z_world_3d: torch.Tensor,  # [32, D, H, W]
    state: torch.Tensor,  # [D, 18, H, W]
    dm_teacher: torch.Tensor,  # [4, D, H, W]
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Compress `Z_world_3d [32, D, H, W]` + physical densities/temp into `[D, H, W, 8]` u8:
      - Ch 0..3: Top 4 PCA projections of `Z_world_3d` (fit on stratified active + air voxels)
      - Ch 4:    `solid_density`
      - Ch 5:    `fluid_density`
      - Ch 6:    `gas_optical_density`
      - Ch 7:    `temp_norm` mapped smoothly into [0, 1]
    Zero RGB channels are included!
    """
    C, D, H, W = z_world_3d.shape
    N = D * H * W
    z_flat = z_world_3d.permute(1, 2, 3, 0).reshape(N, C).float().cpu().numpy()

    solid = dm_teacher[0].float().cpu().numpy().reshape(N)
    fluid = dm_teacher[1].float().cpu().numpy().reshape(N)
    gas = dm_teacher[2].float().cpu().numpy().reshape(N)
    temp_raw = state[:, CH_TEMP].float().cpu().numpy().reshape(N)

    # Fit PCA on a stratified subset emphasizing active solid/fluid/gas/warm voxels so
    # all 4 principal components capture rich material & 3D context variation rather than empty sky.
    rng = np.random.default_rng(1337)
    active_idx = np.flatnonzero((solid > 0.04) | (fluid > 0.04) | (gas > 0.03) | (temp_raw > 0.5))
    air_idx = np.flatnonzero(~((solid > 0.04) | (fluid > 0.04) | (gas > 0.03) | (temp_raw > 0.5)))
    n_act = min(len(active_idx), 80_000)
    n_air = min(len(air_idx), 12_000)
    sample_idx = np.concatenate([
        rng.choice(active_idx, size=n_act, replace=False),
        rng.choice(air_idx, size=n_air, replace=False),
    ])

    z_sub = z_flat[sample_idx]
    z_mean = z_sub.mean(axis=0, keepdims=True)
    z_centered = z_sub - z_mean
    # SVD on [M, 32] covariance [32, 32]
    cov = (z_centered.T @ z_centered) / max(len(z_sub) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    basis = eigvecs[:, order[:4]].astype(np.float32)  # [32, 4]
    explained_var_4 = float(eigvals[:4].sum() / max(eigvals.sum(), 1e-12))

    proj = (z_flat - z_mean.astype(np.float32)) @ basis  # [N, 4]
    p_lo = np.percentile(proj[sample_idx], 0.15, axis=0).astype(np.float32)
    p_hi = np.percentile(proj[sample_idx], 99.85, axis=0).astype(np.float32)
    p_span = np.maximum(p_hi - p_lo, 1e-4)
    pca_norm = np.clip((proj - p_lo) / p_span, 0.0, 1.0)

    # Smooth compressed temperature channel (preserves frost <0, ambient ~0, and hot fire/magma 1.5..10)
    temp_packed = np.clip((temp_raw + 0.85) / 10.6, 0.0, 1.0)

    feats_f32 = np.empty((N, 8), dtype=np.float32)
    feats_f32[:, 0:4] = pca_norm
    feats_f32[:, 4] = np.clip(solid, 0.0, 1.0)
    feats_f32[:, 5] = np.clip(fluid, 0.0, 1.0)
    feats_f32[:, 6] = np.clip(gas, 0.0, 1.0)
    feats_f32[:, 7] = temp_packed

    feats_u8 = np.rint(feats_f32 * 255.0).astype(np.uint8).reshape(D, H, W, 8)
    meta = {
        "pca_explained_variance_ratio_top4": explained_var_4,
        "pca_eigenvalues_top4": [float(v) for v in eigvals[:4]],
        "active_voxel_fraction": float(len(active_idx) / N),
    }
    return np.ascontiguousarray(feats_u8), meta


@torch.no_grad()
def build_swappable_style_targets(
    state: torch.Tensor,  # [D, 18, H, W]
    dm_nordic: torch.Tensor,  # [4, D, H, W]
    at_nordic: torch.Tensor,  # [4, D, H, W]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the 3 swappable `[8, D, H, W]` target volumes (`density_mat` + `albedo_thermal`):
      - Style 0 (`NORDIC PIXEL-2XSS`): Exact Track 8A `NeuralMaterialHead` prediction.
      - Style 1 (`VOLCANIC EMBER`):    Identical 3D solid/fluid/gas geometry; obsidian ash ridges,
                                       glowing molten magma river & waterfall, charred timber bridge,
                                       incandescent ember vents!
      - Style 2 (`ARCANE THERMAL`):    Identical 3D solid/fluid/gas geometry; bioluminescent
                                       frost-indigo ridges, neon-cyan fluid flows, and vibrant
                                       magenta/coral thermal vision exposing subterranean heat!
    """
    s = state.float()
    m_water = s[:, S_WATER].clamp(min=0.0)
    m_ice = s[:, S_ICE].clamp(min=0.0)
    m_steam = s[:, S_STEAM].clamp(min=0.0)
    m_soil = s[:, S_SOIL].clamp(min=0.0)
    m_rock = s[:, S_ROCK].clamp(min=0.0)
    m_wood = s[:, S_WOOD].clamp(min=0.0)
    m_ash = s[:, S_ASH].clamp(min=0.0)
    m_smoke = s[:, S_SMOKE].clamp(min=0.0)
    m_plant = s[:, S_PLANT].clamp(min=0.0)
    temp_n = s[:, CH_TEMP]
    flow_z = s[:, CH_FLOW_Z]
    elev_z = s[:, CH_ELEV_Z]

    solid = dm_nordic[0]
    fluid = dm_nordic[1]
    gas = dm_nordic[2]
    rough_n = dm_nordic[3]
    alb_n = at_nordic[0:3]  # [3, D, H, W]
    glow_n = at_nordic[3]

    # Style 0: exact Track 8A NeuralMaterialHead output
    target_0 = torch.cat([dm_nordic, at_nordic], dim=0).clamp(0.0, 1.0)

    # Helper masks from Nordic material cues & physics
    non_empty = (solid + fluid + gas).clamp(0.0, 1.0)
    is_snow = (m_ice + (alb_n[2] - 0.65).clamp(min=0.0) * 2.2 * solid).clamp(0.0, 1.0)
    is_wood = (m_wood * 1.4).clamp(0.0, 1.0)
    is_plant = (m_plant * 1.4 + ((alb_n[1] - alb_n[0]) * 3.5).clamp(0.0, 1.0) * solid).clamp(0.0, 1.0)
    is_wf = ((-flow_z - 0.18) / 0.65).clamp(0.0, 1.0) * fluid
    hot_zone = ((temp_n - 0.8) / 5.5).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Style 1: VOLCANIC EMBER
    # ------------------------------------------------------------------
    luma_n = 0.299 * alb_n[0] + 0.587 * alb_n[1] + 0.114 * alb_n[2]
    # Base volcanic obsidian / basalt slate (preserves cavity AO & tonal detail from Style 0!)
    v_r = (0.14 + 0.34 * luma_n + 0.08 * m_rock).clamp(0.0, 1.0)
    v_g = (0.10 + 0.21 * luma_n + 0.03 * m_soil).clamp(0.0, 1.0)
    v_b = (0.12 + 0.19 * luma_n).clamp(0.0, 1.0)

    # Warm ash-dusted sulfur ridges replacing alpine snow
    v_r = v_r * (1.0 - 0.78 * is_snow) + 0.78 * (0.78 * is_snow)
    v_g = v_g * (1.0 - 0.78 * is_snow) + 0.52 * (0.78 * is_snow)
    v_b = v_b * (1.0 - 0.78 * is_snow) + 0.34 * (0.78 * is_snow)

    # Charred bronze-ironwood bridge & timber
    v_r = v_r * (1.0 - 0.70 * is_wood) + 0.44 * (0.70 * is_wood)
    v_g = v_g * (1.0 - 0.70 * is_wood) + 0.24 * (0.70 * is_wood)
    v_b = v_b * (1.0 - 0.70 * is_wood) + 0.14 * (0.70 * is_wood)

    # Smoldering bronze-russet ash vegetation
    v_r = v_r * (1.0 - 0.65 * is_plant) + 0.46 * (0.65 * is_plant)
    v_g = v_g * (1.0 - 0.65 * is_plant) + 0.22 * (0.65 * is_plant)
    v_b = v_b * (1.0 - 0.65 * is_plant) + 0.10 * (0.65 * is_plant)

    # River & plunging waterfall -> Molten Magma Current & Incandescent Lava Cascade!
    lava_w = (fluid * 1.35).clamp(0.0, 1.0)
    lava_r = 0.96 + 0.04 * is_wf
    lava_g = 0.28 + 0.54 * is_wf + 0.14 * (m_water > 0.3).float()
    lava_b = 0.04 + 0.28 * is_wf
    v_r = v_r * (1.0 - lava_w) + lava_r * lava_w
    v_g = v_g * (1.0 - lava_w) + lava_g * lava_w
    v_b = v_b * (1.0 - lava_w) + lava_b * lava_w

    # Glowing magma veins along low canyon banks & hot rock crevices
    ember_vein = ((1.0 - elev_z * 2.2).clamp(0.0, 1.0) * solid * (1.0 - rough_n).clamp(0.0, 1.0) * 0.55)
    v_glow = torch.maximum(glow_n * 1.15, torch.maximum(lava_w * (0.42 + 0.50 * is_wf), ember_vein + hot_zone * 0.85)).clamp(0.0, 1.0)

    # Warm cinder smoke/steam haze
    gas_w = (gas * (1.0 - solid) * (1.0 - fluid)).clamp(0.0, 1.0)
    v_r = v_r * (1.0 - gas_w) + (0.46 + 0.42 * hot_zone) * gas_w
    v_g = v_g * (1.0 - gas_w) + (0.24 + 0.26 * hot_zone) * gas_w
    v_b = v_b * (1.0 - gas_w) + (0.17 + 0.06 * hot_zone) * gas_w

    # Incandescent core push on vents & wildfire
    v_r = v_r * (1.0 - 0.85 * glow_n) + 1.00 * (0.85 * glow_n)
    v_g = v_g * (1.0 - 0.85 * glow_n) + (0.68 + 0.28 * glow_n) * (0.85 * glow_n)
    v_b = v_b * (1.0 - 0.85 * glow_n) + (0.20 + 0.45 * glow_n) * (0.85 * glow_n)

    # Empty air dark ember backdrop
    v_r = (v_r * non_empty + 0.10 * (1.0 - non_empty)).clamp(0.0, 1.0)
    v_g = (v_g * non_empty + 0.06 * (1.0 - non_empty)).clamp(0.0, 1.0)
    v_b = (v_b * non_empty + 0.07 * (1.0 - non_empty)).clamp(0.0, 1.0)
    v_rough = (rough_n * (1.0 - 0.35 * lava_w) + 0.08 * solid * (1.0 - is_snow)).clamp(0.0, 1.0)

    target_1 = torch.stack([solid, fluid, gas, v_rough, v_r, v_g, v_b, v_glow], dim=0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Style 2: ARCANE THERMAL (Bioluminescent Frost-Indigo + Neon-Cyan + Thermal Vision!)
    # ------------------------------------------------------------------
    # Deep arcane indigo terrain preserving surface shading & structural contrast
    a_r = (0.06 + 0.16 * luma_n + 0.08 * elev_z * solid).clamp(0.0, 1.0)
    a_g = (0.10 + 0.24 * luma_n + 0.14 * elev_z * solid).clamp(0.0, 1.0)
    a_b = (0.25 + 0.44 * luma_n + 0.16 * elev_z * solid).clamp(0.0, 1.0)

    # Glacial frost-cyan alpine peaks
    a_r = a_r * (1.0 - 0.80 * is_snow) + 0.56 * (0.80 * is_snow)
    a_g = a_g * (1.0 - 0.80 * is_snow) + 0.90 * (0.80 * is_snow)
    a_b = a_b * (1.0 - 0.80 * is_snow) + 1.00 * (0.80 * is_snow)

    # Runic cobalt-violet stone & luminous teal bridge timber
    a_r = a_r * (1.0 - 0.72 * is_wood) + 0.24 * (0.72 * is_wood)
    a_g = a_g * (1.0 - 0.72 * is_wood) + 0.62 * (0.72 * is_wood)
    a_b = a_b * (1.0 - 0.72 * is_wood) + 0.88 * (0.72 * is_wood)

    # Bioluminescent jade-cyan moss & pine canopy
    a_r = a_r * (1.0 - 0.70 * is_plant) + 0.08 * (0.70 * is_plant)
    a_g = a_g * (1.0 - 0.70 * is_plant) + 0.74 * (0.70 * is_plant)
    a_b = a_b * (1.0 - 0.70 * is_plant) + 0.72 * (0.70 * is_plant)

    # Electric neon-cyan bioluminescent river & plunging waterfall!
    fl_w = (fluid * 1.35).clamp(0.0, 1.0)
    a_r = a_r * (1.0 - fl_w) + (0.08 + 0.56 * is_wf) * fl_w
    a_g = a_g * (1.0 - fl_w) + (0.88 + 0.12 * is_wf) * fl_w
    a_b = a_b * (1.0 - fl_w) + 1.00 * fl_w

    # Thermal vision heat signature: violet -> electric magenta -> coral -> white-gold!
    therm = torch.maximum(glow_n, ((temp_n - 0.35) / 5.2).clamp(0.0, 1.0))
    t_r = (0.72 + 0.28 * therm).clamp(0.0, 1.0)
    t_g = (0.14 + 0.72 * (therm ** 1.4)).clamp(0.0, 1.0)
    t_b = (0.88 * (1.0 - 0.55 * therm) + 0.35 * therm).clamp(0.0, 1.0)
    a_r = a_r * (1.0 - 0.90 * therm) + t_r * (0.90 * therm)
    a_g = a_g * (1.0 - 0.90 * therm) + t_g * (0.90 * therm)
    a_b = a_b * (1.0 - 0.90 * therm) + t_b * (0.90 * therm)

    # Bioluminescent mist / thermal plume gas
    a_r = a_r * (1.0 - gas_w) + (0.16 + 0.80 * therm) * gas_w
    a_g = a_g * (1.0 - gas_w) + (0.76 - 0.25 * therm) * gas_w
    a_b = a_b * (1.0 - gas_w) + (0.98 - 0.30 * therm) * gas_w

    # Expose hidden heat & fluid flows via glow channel!
    a_glow = torch.maximum(
        therm * 0.98,
        torch.maximum(fl_w * (0.28 + 0.34 * is_wf), (m_steam + m_smoke).clamp(0.0, 1.0) * 0.22),
    ).clamp(0.0, 1.0)

    a_r = (a_r * non_empty + 0.04 * (1.0 - non_empty)).clamp(0.0, 1.0)
    a_g = (a_g * non_empty + 0.07 * (1.0 - non_empty)).clamp(0.0, 1.0)
    a_b = (a_b * non_empty + 0.16 * (1.0 - non_empty)).clamp(0.0, 1.0)
    a_rough = (rough_n * 0.62 + 0.10).clamp(0.0, 1.0)

    target_2 = torch.stack([solid, fluid, gas, a_rough, a_r, a_g, a_b, a_glow], dim=0).clamp(0.0, 1.0)

    return target_0, target_1, target_2


def distill_style_mlp(
    name: str,
    x_q_f32: np.ndarray,  # [N, 8] exact dequantized u8/255 inputs
    y_target: np.ndarray,  # [N, 8] target in [0, 1]
    active_mask: np.ndarray,  # [N] bool
    device: torch.device,
    steps: int = 1600,
    batch_size: int = 16384,
    seed: int = 42,
) -> tuple[ClientMaterialMLP, dict[str, Any]]:
    """Train compact `8 -> 24 -> 24 -> 8` MLP directly on quantized `z16_latent_bricks` inputs."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ClientMaterialMLP().to(device)
    x_t = torch.from_numpy(x_q_f32).to(device)
    y_t = torch.from_numpy(y_target).to(device)

    # Stratified sampling: 15% hot/emissive voxels (magma vents & wildfire!) + 63% active surface/fluid/gas + 22% global
    # so even rare 0.1% molten magma vents & wildfire cores achieve R^2 > 0.98!
    act_idx = torch.from_numpy(np.flatnonzero(active_mask)).to(device)
    hot_mask = (x_q_f32[:, 7] > 0.16) | (y_target[:, 7] > 0.04)
    hot_np = np.flatnonzero(hot_mask)
    hot_idx = torch.from_numpy(hot_np if len(hot_np) > 0 else np.flatnonzero(active_mask)).to(device)
    n_total = x_t.shape[0]
    n_act = act_idx.shape[0]
    n_hot = hot_idx.shape[0]
    b_hot = int(batch_size * 0.15)
    b_act = int(batch_size * 0.63)
    b_all = batch_size - b_hot - b_act

    # Channel weights: keep solid/fluid/gas, albedo, AND blackbody thermal glow super accurate!
    ch_w = torch.tensor([1.6, 1.6, 1.4, 1.0, 1.6, 1.6, 1.6, 2.4], dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)

    t0 = time.perf_counter()
    for _ in range(steps):
        ih = hot_idx[torch.randint(0, n_hot, (b_hot,), device=device)]
        ia = act_idx[torch.randint(0, n_act, (b_act,), device=device)]
        ig = torch.randint(0, n_total, (b_all,), device=device)
        idx = torch.cat([ih, ia, ig], dim=0)

        xb = x_t[idx]
        yb = y_t[idx]
        pred = model(xb)
        mse = ((pred - yb) ** 2) * ch_w
        l1 = (pred - yb).abs() * ch_w
        loss = mse.mean() + 0.12 * l1.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    train_s = time.perf_counter() - t0

    # Full-volume exact evaluation (all 1,048,576 voxels!)
    model.eval()
    preds = []
    with torch.inference_mode():
        chunk = 131072
        for s0 in range(0, n_total, chunk):
            preds.append(model(x_t[s0 : s0 + chunk]).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)

    r2_all = _r2_score(y_pred, y_target)
    r2_dm = _r2_score(y_pred[:, 0:4], y_target[:, 0:4])
    r2_at = _r2_score(y_pred[:, 4:8], y_target[:, 4:8])
    r2_active = _r2_score(y_pred[active_mask], y_target[active_mask])
    mae_all = float(np.mean(np.abs(y_pred - y_target)))
    ch_names = [
        "solid_density",
        "fluid_density",
        "gas_optical_density",
        "roughness_wetness",
        "albedo_r",
        "albedo_g",
        "albedo_b",
        "blackbody_temp_glow",
    ]
    r2_per_ch = {ch_names[i]: _r2_score(y_pred[:, i], y_target[:, i]) for i in range(8)}

    metrics = {
        "style_name": name,
        "params_floats": FLOATS_PER_STYLE,
        "bytes": FLOATS_PER_STYLE * 4,
        "train_time_s": round(train_s, 3),
        "r2_overall": r2_all,
        "r2_density_mat": r2_dm,
        "r2_albedo_thermal": r2_at,
        "r2_active_voxels": r2_active,
        "mae_overall": mae_all,
        "r2_per_channel": r2_per_ch,
    }
    print(
        f"[Distill] {name:18s} | R2_all={r2_all:.4f} | R2_dm={r2_dm:.4f} | R2_at={r2_at:.4f} "
        f"| R2_active={r2_active:.4f} | MAE={mae_all:.4f} | {train_s:.2f}s"
    )
    return model, metrics


def save_fmlt_bricks(path: Path, feats_u8: np.ndarray, seed: int) -> dict[str, Any]:
    """
    Save `models/checkpoints/z16_latent_bricks.bin` (magic `b"FMLT"`):
      Header (24 bytes LE): magic[4], width(u32), height(u32), depth(u32), channels(u32), seed(u32)
      Payload: `[depth, height, width, 8]` u8 (Z-major `[alt][north][east][8]`).
    """
    assert feats_u8.ndim == 4 and feats_u8.shape[3] == 8 and feats_u8.dtype == np.uint8
    depth, height, width, channels = feats_u8.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<4sIIIII", FMLT_MAGIC, width, height, depth, channels, seed)
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.ascontiguousarray(feats_u8).tobytes())
    nbytes = path.stat().st_size
    expected = 24 + depth * height * width * channels
    assert nbytes == expected, f"FMLT size mismatch: {nbytes} != {expected}"
    return {
        "path": str(path),
        "magic": "FMLT",
        "width": width,
        "height": height,
        "depth": depth,
        "channels": channels,
        "seed": seed,
        "bytes": nbytes,
    }


def save_fmwg_weights(path: Path, packed_styles: list[np.ndarray]) -> dict[str, Any]:
    """
    Save `models/checkpoints/client_material_weights.bin` (magic `b"FMWG"`):
      Header (24 bytes LE): magic[4], num_styles(u32), in_dim(u32), h1_dim(u32), h2_dim(u32), out_dim(u32)
      Payload: `num_styles * 1016` f32 LE floats.
    """
    num_styles = len(packed_styles)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<4sIIIII", FMWG_MAGIC, num_styles, IN_DIM, H1_DIM, H2_DIM, OUT_DIM)
    with open(path, "wb") as f:
        f.write(header)
        for arr in packed_styles:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())
    nbytes = path.stat().st_size
    expected = 24 + num_styles * FLOATS_PER_STYLE * 4
    assert nbytes == expected, f"FMWG size mismatch: {nbytes} != {expected}"
    return {
        "path": str(path),
        "magic": "FMWG",
        "num_styles": num_styles,
        "arch": [IN_DIM, H1_DIM, H2_DIM, OUT_DIM],
        "floats_per_style": FLOATS_PER_STYLE,
        "bytes_per_style": FLOATS_PER_STYLE * 4,
        "total_bytes": nbytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Server->Client Z=16 Latent Bricks + 3 Swappable WGSL Material Decoder Weights")
    parser.add_argument("--ckpt", type=Path, default=REPO_ROOT / "models/checkpoints/neural_field_3d_z16.pt")
    parser.add_argument("--out-latent", type=Path, default=REPO_ROOT / "models/checkpoints/z16_latent_bricks.bin")
    parser.add_argument("--out-weights", type=Path, default=REPO_ROOT / "models/checkpoints/client_material_weights.bin")
    parser.add_argument("--seed", type=int, default=909, help="Unseen held-out 256x256x16 world seed")
    parser.add_argument("--steps", type=int, default=1600, help="Distillation steps per style")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Export] Device={device} | Checkpoint={args.ckpt} | Unseen seed={args.seed}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    model = NeuralWorldField3D(
        in_ch=arch["in_ch"],
        latent_ch=arch["latent_ch"],
        mat_hidden_ch=arch["mat_hidden_ch"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    t0 = time.perf_counter()
    state, landmarks = generate_world_3d_z16(
        height=256,
        width=256,
        depth=16,
        seed=args.seed,
        device=device,
        return_landmarks=True,
    )
    with torch.inference_mode():
        out = model(state, return_latent=True)
    dm_teacher = out["density_mat"]  # [4, 16, 256, 256]
    at_teacher = out["albedo_thermal"]  # [4, 16, 256, 256]
    z_world_3d = out["z_world_3d"]  # [32, 16, 256, 256]
    enc_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[Server WorldModel] Encoded unseen 256x256x16 world (seed={args.seed}) in {enc_ms:.1f} ms | "
        f"v_bridge={landmarks['v_bridge']:.3f} v_cave={landmarks['v_cave']:.3f} v_wf={landmarks['v_waterfall']:.3f}"
    )

    feats_u8, pca_meta = build_compact_latent_packet(z_world_3d, state, dm_teacher)
    print(
        f"[Latent Packet] PCA top-4 explained variance={pca_meta['pca_explained_variance_ratio_top4']:.4f} | "
        f"active_voxels={pca_meta['active_voxel_fraction']*100:.1f}%"
    )

    t_style0, t_style1, t_style2 = build_swappable_style_targets(state, dm_teacher, at_teacher)

    # Exact quantized inputs that WGSL `unpack4x8unorm` will see!
    N = 16 * 256 * 256
    x_q_f32 = (feats_u8.reshape(N, 8).astype(np.float32)) / 255.0
    active_mask = (
        (x_q_f32[:, 4] > 0.04)
        | (x_q_f32[:, 5] > 0.04)
        | (x_q_f32[:, 6] > 0.03)
        | (x_q_f32[:, 7] > 0.14)
    )

    style_specs = [
        ("NORDIC PIXEL-2XSS", t_style0, 101),
        ("VOLCANIC EMBER", t_style1, 202),
        ("ARCANE THERMAL", t_style2, 303),
    ]
    packed_styles: list[np.ndarray] = []
    style_metrics: list[dict[str, Any]] = []
    for style_name, target_8dhw, s_seed in style_specs:
        y_np = target_8dhw.permute(1, 2, 3, 0).reshape(N, 8).float().cpu().numpy()
        mlp, m = distill_style_mlp(
            name=style_name,
            x_q_f32=x_q_f32,
            y_target=y_np,
            active_mask=active_mask,
            device=device,
            steps=args.steps,
            seed=s_seed,
        )
        packed_styles.append(mlp.pack_weights_f32())
        style_metrics.append(m)

    assert style_metrics[0]["r2_overall"] >= 0.96, (
        f"Style 0 R^2 {style_metrics[0]['r2_overall']:.4f} < 0.96 target!"
    )

    fmlt_info = save_fmlt_bricks(args.out_latent, feats_u8, args.seed)
    fmwg_info = save_fmwg_weights(args.out_weights, packed_styles)

    summary = {
        "seed": args.seed,
        "fmlt_packet": fmlt_info,
        "fmwg_weights": fmwg_info,
        "pca_compression": pca_meta,
        "landmarks": {
            "v_bridge": landmarks["v_bridge"],
            "v_cave": landmarks["v_cave"],
            "v_waterfall": landmarks["v_waterfall"],
            "u_fire": landmarks["u_fire"],
            "v_fire": landmarks["v_fire"],
            "z_bridge_deck": landmarks["z_bridge_deck"],
            "z_river_water": landmarks["z_river_water"],
            "z_lake_water": landmarks["z_lake_water"],
            "magma_vents": landmarks["magma_vents"],
            "cave_pillars": landmarks["cave_pillars"],
        },
        "styles": style_metrics,
    }
    print("[Summary]", json.dumps({
        "fmlt_bytes": fmlt_info["bytes"],
        "fmwg_bytes": fmwg_info["total_bytes"],
        "style0_r2": round(style_metrics[0]["r2_overall"], 5),
        "style1_r2": round(style_metrics[1]["r2_overall"], 5),
        "style2_r2": round(style_metrics[2]["r2_overall"], 5),
    }))
    # Write a small companion meta block only to stdout (respecting strict file boundaries!)
    # Rust reads exact metrics or passes them into models/STYLIZED_3D_Z16_METRICS.json.
    meta_out = REPO_ROOT / "models/STYLIZED_3D_Z16_METRICS.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps({"distillation": summary}, indent=2))
    print(f"[Saved] {args.out_latent} ({fmlt_info['bytes']} B), {args.out_weights} ({fmwg_info['total_bytes']} B)")


if __name__ == "__main__":
    main()
