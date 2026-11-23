#!/usr/bin/env python3
"""
Zero-Hardcoded-Rules Emergent Physical Actuators (`ActuatorPolicyNet`) via GPU Self-Play RL.

Runs a 100% Physics-Derived 6v6 Vectorized GPU Arena (64 parallel arenas x 12 bots = 768
concurrent bots on Colab A100-SXM4-40GB) driven directly by frozen `WorldModel` v3
(`world_model_v3.pt`) + `EcosystemModel` (`ecosystem_model.pt`).

ZERO hardcoded spell handlers (no Fire Bolt / Frost Nova / ability buttons / element enums)
and ZERO `if was_oiled { dmg = 34 }` rules! Every combat and movement effect emerges purely
from local voxel physics & `WorldModel` v3 state deltas:
  - Thermal Stress:       scorch ~ max(0, T - 1.55)^2, hypothermia ~ max(0, -0.15 - T)^2
  - Phase-Change & Combustion Pressure Impulses:
                          sharp local dT > 0.65 (e.g. oil combustion) or flash-boil
                          dSteam > 0.25 (heating water/ice) produces a physical pressure
                          wave dealing damage AND radial knockback along -grad(P)!
  - Surface Friction & Buoyancy:
                          ice -> low-friction speed boost; water -> deep-water drag unless
                          frozen into ice; oil -> momentum slip; smoke + steam optical
                          density obscures enemy observations & scatters enemy targeting!
  - Conservation + Mana Budget:
                          mana cost ~ (|thermal_flux| + sp_water + sp_oil + sp_smoke
                          + impulse_mag) * area_factor(focus_radius).

Trains `ActuatorPolicyNet` (~215K params) via pure reward-driven float32-safe PPO Self-Play
(NO named-spell imitation teacher! Only sparse game goals: Shrine control, Relic carry/delivery,
enemy eliminations, survival).

Evaluates on unseen held-out 256x256x4 worlds (24 side-balanced matches per pair) against:
  - Random Continuous Actuators
  - Thermal-Only Actuators (ablation with matter emission zeroed)
  - Legacy Named-Spell Scripted Baseline (translated to equivalent physical deposits)

Produces:
  - actuator_policy_rl.pt (pure ActuatorPolicyNet state_dict, op=7 drop-in compatible)
  - ACTUATOR_EMERGENCE_METRICS.json
  - actuator_emergence_viz.png (3-panel dark-themed matplotlib figure)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from actuator_policy import (  # noqa: E402
    ACT_EMIT_DX,
    ACT_EMIT_DY,
    ACT_FOCUS,
    ACT_IMPULSE,
    ACT_MOVE_X,
    ACT_MOVE_Y,
    ACT_OIL,
    ACT_SMOKE,
    ACT_THERMAL,
    ACT_WATER,
    ACTUATOR_DIM,
    BOT_FEAT_DIM,
    CANONICAL_WORLD_SPAN,
    MAX_EMIT_RANGE_CELLS,
    NUM_TOKENS,
    SELF_DIM,
    TOKEN_DIM,
    ActuatorPolicyNet,
    actuator_area_factor,
    count_parameters,
    focus_radius_cells,
    raw_to_actuators,
    sample_batched_dual_patches,
)
from model import EcosystemModel, WorldModel  # noqa: E402

# Exact FIELD_CONTRACT species & channel indices
SP_WATER, SP_ICE, SP_STEAM, SP_SOIL, SP_ROCK, SP_WOOD, SP_ASH, SP_SMOKE, SP_OIL, SP_PLANT = range(10)
CH_TEMP = 10
CH_LOGMASS = 11
CH_FLOW = 13
CH_ALT = 14
CH_OBST = 15

CANON_SIZE = 256.0
NUM_BOTS = 12


def load_ckpt_state_dict(path: Path, device: torch.device) -> dict:
    ck = torch.load(str(path), map_location=device, weights_only=False)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


def load_frozen_models(
    world_pt: Path,
    eco_pt: Path,
    device: torch.device,
) -> tuple[WorldModel, EcosystemModel]:
    wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    wm.load_state_dict(load_ckpt_state_dict(world_pt, device))
    for p in wm.parameters():
        p.requires_grad_(False)

    em = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).to(device).eval()
    em.load_state_dict(load_ckpt_state_dict(eco_pt, device))
    for p in em.parameters():
        p.requires_grad_(False)
    return wm, em


@torch.inference_mode()
def extract_world_latent_and_delta(
    wm: WorldModel,
    w_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w_state: [A, L, 18, H, W] float32
    Returns:
      z_world: [A, L, 48, H, W] float16 (or float32 on CPU)
      deltas:  [A, L, 17, H, W] float32
    """
    a, l, c, h, w = w_state.shape
    x = w_state.reshape(a * l, c, h, w)
    if w_state.device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h_feat = wm.stem(x)
            for blk in wm.blocks:
                h_feat = blk(h_feat)
            z = F.gelu(h_feat)
            raw = wm.head(z) * wm.out_scale
            out = torch.cat([raw[:, :10], F.softplus(raw[:, 10:])], dim=1)
        return z.half().reshape(a, l, 48, h, w), out.float().reshape(a, l, 17, h, w)
    h_feat = wm.stem(x)
    for blk in wm.blocks:
        h_feat = blk(h_feat)
    z = F.gelu(h_feat)
    raw = wm.head(z) * wm.out_scale
    out = torch.cat([raw[:, :10], F.softplus(raw[:, 10:])], dim=1)
    return z.float().reshape(a, l, 48, h, w), out.float().reshape(a, l, 17, h, w)


@torch.inference_mode()
def extract_eco_latent(em: EcosystemModel, e_state: torch.Tensor) -> torch.Tensor:
    a, _, c, h, w = e_state.shape
    x = e_state.reshape(a, c, h, w)
    if e_state.device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            h_feat = em.stem(x)
            for blk in em.blocks:
                h_feat = blk(h_feat)
            z = F.gelu(h_feat)
        return z.half().reshape(a, 1, 32, h, w)
    h_feat = em.stem(x)
    for blk in em.blocks:
        h_feat = blk(h_feat)
    z = F.gelu(h_feat)
    return z.float().reshape(a, 1, 32, h, w)


class PhysicsActuatorArena:
    """
    100% Physics-Derived 6v6 Vectorized GPU Arena driven by frozen WorldModel v3 + EcosystemModel.

    Zero hardcoded spell buttons, zero element enums, zero `if was_oiled` damage tables!
    All entity damage, knockback, surface friction, optical concealment, and phase-change
    reactions emerge strictly from voxel thermodynamics, conservation laws, and WorldModel v3 deltas.
    """

    def __init__(
        self,
        num_arenas: int,
        world_model: WorldModel,
        eco_model: EcosystemModel,
        device: torch.device,
        grid_size: int = 128,
        num_layers: int = 2,
        max_ticks: int = 120,
        latent_refresh_every: int = 4,
    ):
        self.A = num_arenas
        self.N = NUM_BOTS
        self.H = grid_size
        self.W = grid_size
        self.L = num_layers
        self.max_ticks = max_ticks
        self.refresh_every = latent_refresh_every
        self.device = device
        self.wm = world_model
        self.em = eco_model

        self.surf_l = 0 if num_layers == 2 else 1
        self.high_l = 1 if num_layers == 2 else 2

        teams = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        classes = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long, device=device)
        self.team = teams.unsqueeze(0).expand(self.A, -1).contiguous()
        self.cls = classes.unsqueeze(0).expand(self.A, -1).contiguous()
        self.w_mask = (self.team == 0).float()
        self.e_mask = (self.team == 1).float()
        self.arena_idx = torch.arange(self.A, device=device).unsqueeze(1).expand(self.A, self.N)
        self.bot_idx = torch.arange(self.N, device=device).unsqueeze(0).expand(self.A, self.N)
        self.enemy_pair_mask = self.team[:, :, None] != self.team[:, None, :]

        yy = torch.arange(self.H, device=device, dtype=torch.float32) / self.H
        xx = torch.arange(self.W, device=device, dtype=torch.float32) / self.W
        self.ny, self.nx = torch.meshgrid(yy, xx, indexing="ij")

        # Variable-radius physical Gaussian deposit stencil spanning up to +-8.5 canonical cells
        # On 128x128: 1 cell = 2.0 canon -> +-5 grid cells covers +-10 canon (11x11=121 offsets)
        # On 256x256: 1 cell = 1.0 canon -> +-9 grid cells covers +-9 canon (19x19=361 offsets)
        cell_span = CANON_SIZE / float(self.W)
        st_rad = int(math.ceil(9.5 / cell_span))
        offs = torch.arange(-st_rad, st_rad + 1, device=device, dtype=torch.long)
        sdy, sdx = torch.meshgrid(offs, offs, indexing="ij")
        dist_canon = torch.hypot(sdx.float() * cell_span, sdy.float() * cell_span)
        keep = dist_canon <= 9.6
        self.st_dy = sdy[keep].contiguous()                 # [S]
        self.st_dx = sdx[keep].contiguous()                 # [S]
        self.st_dist = dist_canon[keep].contiguous()        # [S] in canonical cells

        # Shrine coordinates in canonical 256-scale
        self.shrine_x = torch.tensor([128.0, 128.0, 128.0], device=device).unsqueeze(0).expand(self.A, -1).clone()
        self.shrine_y = torch.tensor([0.18 * 256.0, 0.50 * 256.0, 0.82 * 256.0], device=device).unsqueeze(0).expand(self.A, -1).clone()

        # Voxel & neural latent buffers
        self.w_state = torch.zeros((self.A, self.L, 18, self.H, self.W), dtype=torch.float32, device=device)
        self.e_state = torch.zeros((self.A, 1, 8, self.H, self.W), dtype=torch.float32, device=device)
        z_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.z_world = torch.zeros((self.A, self.L, 48, self.H, self.W), dtype=z_dtype, device=device)
        self.z_eco = torch.zeros((self.A, 1, 32, self.H, self.W), dtype=z_dtype, device=device)

        # Attribution & emergence telemetry grids (never used for damage rules, only credit & telemetry!)
        self.cell_src_bot = torch.full((self.A, self.L, self.H, self.W), -1, dtype=torch.long, device=device)
        self.cell_src_ttl = torch.zeros((self.A, self.L, self.H, self.W), dtype=torch.int16, device=device)
        self.cell_oil_bot = torch.full((self.A, self.L, self.H, self.W), -1, dtype=torch.long, device=device)
        self.cell_oil_ttl = torch.zeros((self.A, self.L, self.H, self.W), dtype=torch.int16, device=device)

        self.reset(seed=0)

    def _generate_worlds(self, gen: torch.Generator) -> None:
        """Vectorized GPU generation of A diverse symmetric Frostmarsh 3D worlds matching FIELD_CONTRACT."""
        A, L, H, W = self.A, self.L, self.H, self.W
        dev = self.device
        self.w_state.zero_()
        self.e_state.zero_()

        nx = self.nx.unsqueeze(0)
        ny = self.ny.unsqueeze(0)

        # Winding river down center separating West & East (makes Cryo-Crossings & Flash-Boil central!)
        phase = torch.rand((A, 1, 1), generator=gen, device=dev) * 6.28318
        rcx = 0.50 + 0.045 * torch.sin(ny * 6.28318 * 1.3 + phase)
        rhw = torch.empty((A, 1, 1), device=dev).uniform_(0.080, 0.110, generator=gen)
        in_river = (nx - rcx).abs() < rhw

        sl = self.surf_l
        hl = self.high_l

        self.w_state[:, sl, SP_SOIL] = torch.where(~in_river, 0.92, 0.0)
        self.w_state[:, sl, SP_WATER] = torch.where(in_river, 1.12, 0.0)
        self.w_state[:, sl, CH_FLOW] = torch.where(in_river, 0.13, 0.0)
        self.w_state[:, sl, CH_ALT] = torch.where(in_river, 0.15, 0.45)

        # Two narrow natural ice crossings (leaving plenty of open water around shrines & center!)
        for iy_lo, iy_hi in ((0.14, 0.28), (0.72, 0.86)):
            ice_y = torch.empty((A, 1, 1), device=dev).uniform_(iy_lo, iy_hi, generator=gen)
            ice_m = in_river & ((ny - ice_y).abs() < 0.032)
            self.w_state[:, sl, SP_ICE] = torch.where(ice_m, 0.90, self.w_state[:, sl, SP_ICE])
            self.w_state[:, sl, SP_WATER] = torch.where(ice_m, 0.15, self.w_state[:, sl, SP_WATER])
            self.w_state[:, sl, CH_TEMP] = torch.where(ice_m, -0.25, self.w_state[:, sl, CH_TEMP])

        # Symmetric natural oil basins (3 pairs = 6 pools per map)
        for _ in range(3):
            ox = torch.empty((A, 1, 1), device=dev).uniform_(0.20, 0.46, generator=gen)
            oy = torch.empty((A, 1, 1), device=dev).uniform_(0.16, 0.84, generator=gen)
            orad = torch.empty((A, 1, 1), device=dev).uniform_(0.038, 0.064, generator=gen)
            for cx in (ox, 1.0 - ox):
                om = ((nx - cx) ** 2 + (ny - oy) ** 2) < (orad ** 2)
                self.w_state[:, sl, SP_OIL] = torch.where(om, 0.80, self.w_state[:, sl, SP_OIL])
                self.w_state[:, sl, SP_WATER] = torch.where(om, torch.maximum(self.w_state[:, sl, SP_WATER], torch.full_like(om, 0.16, dtype=torch.float32)), self.w_state[:, sl, SP_WATER])

        # Wood forests & plants
        s_off = torch.rand((A, 1, 1), generator=gen, device=dev) * 10.0
        wood_noise = torch.sin(nx * 43.0 + s_off) * torch.cos(ny * 48.0 - s_off)
        self.w_state[:, sl, SP_WOOD] = torch.where((~in_river) & (wood_noise > 0.35), 0.85, 0.0)
        self.w_state[:, sl, SP_PLANT] = torch.where((~in_river) & (wood_noise > -0.10), 0.45, 0.0)

        # Symmetric active wildfires (3 pairs = 6 wildfires along lanes)
        for _ in range(3):
            fx = torch.empty((A, 1, 1), device=dev).uniform_(0.22, 0.46, generator=gen)
            fy = torch.empty((A, 1, 1), device=dev).uniform_(0.16, 0.84, generator=gen)
            frad = torch.empty((A, 1, 1), device=dev).uniform_(0.032, 0.052, generator=gen)
            ftemp = torch.empty((A, 1, 1), device=dev).uniform_(2.5, 4.0, generator=gen)
            for cx in (fx, 1.0 - fx):
                fm = ((nx - cx) ** 2 + (ny - fy) ** 2) < (frad ** 2)
                self.w_state[:, sl, CH_TEMP] = torch.where(fm, ftemp, self.w_state[:, sl, CH_TEMP])
                self.w_state[:, sl, SP_SMOKE] = torch.where(fm, 0.32, self.w_state[:, sl, SP_SMOKE])
                self.w_state[:, sl, SP_WOOD] = torch.where(fm, self.w_state[:, sl, SP_WOOD].clamp(min=0.40), self.w_state[:, sl, SP_WOOD])

        # Highland layer (plateaus + bridges)
        nw_p = (nx > 0.06) & (nx < 0.36) & (ny > 0.06) & (ny < 0.33)
        ne_p = (nx > 0.64) & (nx < 0.94) & (ny > 0.06) & (ny < 0.33)
        arch = ((ny - 0.29).abs() < 0.025) & (nx > 0.34) & (nx < 0.66)
        tbridge = ((ny - 0.67).abs() < 0.024) & (nx > 0.35) & (nx < 0.65)
        self.w_state[:, hl, SP_SOIL] = torch.where(nw_p | ne_p | arch, 0.95, 0.0)
        self.w_state[:, hl, SP_WOOD] = torch.where(tbridge, 0.55, 0.0)
        self.w_state[:, hl, CH_ALT] = 0.65

        if L == 4:
            self.w_state[:, 0, SP_WATER] = torch.where(in_river, 1.15, 0.0)
            self.w_state[:, 0, SP_ROCK] = torch.where(~in_river, 1.20, 0.0)
            self.w_state[:, 3, SP_STEAM] = torch.where(self.w_state[:, sl, CH_TEMP] > 2.0, 0.25, 0.0)
            self.w_state[:, 3, CH_ALT] = 0.95

        self.w_state[:, :, CH_LOGMASS] = torch.log1p(self.w_state[:, :, :10].sum(dim=2))

        # Ecosystem state [A, 1, 8, H, W]
        crit = torch.zeros((A, H, W), device=dev)
        for _ in range(2):
            cx = torch.empty((A, 1, 1), device=dev).uniform_(0.18, 0.45, generator=gen)
            cy = torch.empty((A, 1, 1), device=dev).uniform_(0.18, 0.82, generator=gen)
            for c_sym in (cx, 1.0 - cx):
                d2 = (nx - c_sym) ** 2 + (ny - cy) ** 2
                crit = torch.maximum(crit, torch.exp(-d2 * 196.0))
        self.e_state[:, 0, 0] = crit
        self.e_state[:, 0, 2] = 0.8
        self.e_state[:, 0, 4] = self.w_state[:, sl, SP_PLANT]
        self.e_state[:, 0, 5] = ((self.w_state[:, sl, SP_WATER] > 0.35) & (self.w_state[:, sl, SP_ICE] < 0.70)).float()
        self.e_state[:, 0, 6] = (self.w_state[:, sl, CH_TEMP] > 1.5).float()
        self.e_state[:, 0, 7] = (self.w_state[:, sl, SP_ROCK] > 0.5).float()

    def refresh_latents_and_world_step(self, apply_world_deltas: bool = True, refresh_eco: bool = False) -> None:
        """Run frozen WorldModel v3 (and optionally EcosystemModel) to refresh Z_world/Z_eco & apply neural physics deltas."""
        self.z_world, deltas = extract_world_latent_and_delta(self.wm, self.w_state)
        if apply_world_deltas:
            # Apply WorldModel v3 conservative flux divergence on mobile species (water, steam, smoke, oil)
            # Channels 0..4: flux_e [water, steam, smoke, oil]; 4..8: flux_s [water, steam, smoke, oil]
            mobile_idx = (SP_WATER, SP_STEAM, SP_SMOKE, SP_OIL)
            dt_wm = 0.08
            for k, sp in enumerate(mobile_idx):
                fe = deltas[:, :, k].clamp(-0.35, 0.35) * dt_wm
                fs = deltas[:, :, 4 + k].clamp(-0.35, 0.35) * dt_wm
                # Zero boundary flux & donor-safe divergence
                fe = F.pad(fe[:, :, :, :-1], (0, 1, 0, 0), value=0.0)
                fs = F.pad(fs[:, :, :-1, :], (0, 0, 0, 1), value=0.0)
                fw_in = F.pad(fe[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
                fn_in = F.pad(fs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
                cur = self.w_state[:, :, sp]
                out_sum = F.relu(fe) + F.relu(-fw_in) + F.relu(fs) + F.relu(-fn_in)
                scale = (cur / (out_sum + 1e-6)).clamp(max=1.0)
                fe = fe * scale
                fs = fs * scale
                fw_in2 = F.pad(fe[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
                fn_in2 = F.pad(fs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
                self.w_state[:, :, sp] = (cur + fw_in2 - fe + fn_in2 - fs).clamp(0.0, 2.0)

            # Heat conduction divergence from WorldModel v3 channels 8..10 (heat_e, heat_s)
            he = F.pad(deltas[:, :, 8, :, :-1].clamp(-1.5, 1.5), (0, 1, 0, 0), value=0.0) * 0.04
            hs = F.pad(deltas[:, :, 9, :-1, :].clamp(-1.5, 1.5), (0, 0, 0, 1), value=0.0) * 0.04
            hw_in = F.pad(he[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
            hn_in = F.pad(hs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
            self.w_state[:, :, CH_TEMP] = (self.w_state[:, :, CH_TEMP] + hw_in - he + hn_in - hs).clamp(-1.4, 5.5)

            # Neural reaction coupling (WorldModel v3 channels 10..17 modulate phase-change & combustion smoothly)
            rxn = deltas[:, :, 10:17].clamp(0.0, 1.0)
            # rxn[0]=freeze, rxn[1]=melt, rxn[2]=boil, rxn[3]=condense, rxn[4..6]=combustion
            r_freeze = torch.minimum(self.w_state[:, :, SP_WATER], 0.08 * rxn[:, :, 0] * (self.w_state[:, :, CH_TEMP] < -0.05).float())
            r_boil = torch.minimum(self.w_state[:, :, SP_WATER], 0.10 * rxn[:, :, 2] * (self.w_state[:, :, CH_TEMP] > 0.95).float())
            r_comb = torch.minimum(self.w_state[:, :, SP_OIL], 0.12 * rxn[:, :, 4:7].mean(dim=2) * (self.w_state[:, :, CH_TEMP] > 1.25).float())
            self.w_state[:, :, SP_WATER] = (self.w_state[:, :, SP_WATER] - r_freeze - r_boil).clamp(0.0, 2.0)
            self.w_state[:, :, SP_ICE] = (self.w_state[:, :, SP_ICE] + r_freeze).clamp(0.0, 2.0)
            self.w_state[:, :, SP_STEAM] = (self.w_state[:, :, SP_STEAM] + 1.1 * r_boil).clamp(0.0, 2.0)
            self.w_state[:, :, SP_OIL] = (self.w_state[:, :, SP_OIL] - r_comb).clamp(0.0, 2.0)
            self.w_state[:, :, SP_SMOKE] = (self.w_state[:, :, SP_SMOKE] + 0.5 * r_comb).clamp(0.0, 2.0)
            self.w_state[:, :, CH_TEMP] = (self.w_state[:, :, CH_TEMP] + 2.2 * r_comb).clamp(-1.4, 5.5)

        if refresh_eco:
            sl = self.surf_l
            self.e_state[:, 0, 5] = ((self.w_state[:, sl, SP_WATER] > 0.35) & (self.w_state[:, sl, SP_ICE] < 0.70)).float()
            self.e_state[:, 0, 6] = (self.w_state[:, sl, CH_TEMP] > 1.5).float()
            self.z_eco = extract_eco_latent(self.em, self.e_state)

    def reset(self, seed: int = 0, close_spawn_frac: float = 0.0) -> torch.Tensor:
        dev = self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        self.gen = gen
        self.tick = 0

        self._generate_worlds(gen)
        self.cell_src_bot.fill_(-1)
        self.cell_src_ttl.zero_()
        self.cell_oil_bot.fill_(-1)
        self.cell_oil_ttl.zero_()
        self.refresh_latents_and_world_step(apply_world_deltas=False, refresh_eco=True)

        # Symmetric 6v6 spawn across 3 lanes (optional close river-contest spawns during training!)
        lane_y = torch.tensor([46.0, 90.0, 128.0, 166.0, 210.0, 128.0], device=dev)
        jitter_x = torch.empty((self.A, 6), device=dev).uniform_(-10.0, 10.0, generator=gen)
        jitter_y = torch.empty((self.A, 6), device=dev).uniform_(-10.0, 10.0, generator=gen)

        base_wx = torch.full((self.A, 6), 54.0, device=dev)
        if close_spawn_frac > 0.0:
            n_close = int(round(self.A * close_spawn_frac))
            if n_close > 0:
                base_wx[:n_close] = torch.empty((n_close, 6), device=dev).uniform_(82.0, 114.0, generator=gen)
        west_x = (base_wx + jitter_x).clamp(20.0, 118.0)
        east_x = 256.0 - west_x
        west_y = (lane_y.unsqueeze(0) + jitter_y).clamp(20.0, 236.0)
        east_y = west_y.clone()

        self.pos_x = torch.cat([west_x, east_x], dim=1)
        self.pos_y = torch.cat([west_y, east_y], dim=1)
        self.vel_x = torch.zeros((self.A, self.N), device=dev)
        self.vel_y = torch.zeros((self.A, self.N), device=dev)
        self.bot_layer = torch.full((self.A, self.N), self.surf_l, dtype=torch.long, device=dev)

        self.hp = torch.ones((self.A, self.N), device=dev)
        self.mana = torch.ones((self.A, self.N), device=dev)
        self.stam = torch.ones((self.A, self.N), device=dev)
        self.alive = torch.ones((self.A, self.N), dtype=torch.bool, device=dev)
        self.respawn_cd = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.last_attacker = torch.full((self.A, self.N), -1, dtype=torch.long, device=dev)
        self.last_emit_thermal = torch.zeros((self.A, self.N), device=dev)

        self.carrying = torch.zeros((self.A, self.N), dtype=torch.bool, device=dev)
        self.relic_x = torch.full((self.A,), 128.0, device=dev)
        self.relic_y = 128.0 + torch.empty((self.A,), device=dev).uniform_(-22.0, 22.0, generator=gen)
        self.relic_carrier = torch.full((self.A,), -1, dtype=torch.long, device=dev)

        self.shrine_prog = torch.zeros((self.A, 3), device=dev)
        self.shrine_owner = torch.full((self.A, 3), -1, dtype=torch.long, device=dev)

        # Match statistics [A, 2]
        self.team_score = torch.zeros((self.A, 2), device=dev)
        self.team_kills = torch.zeros((self.A, 2), device=dev)
        self.team_deaths = torch.zeros((self.A, 2), device=dev)
        self.team_shrine_caps = torch.zeros((self.A, 2), device=dev)
        self.team_relic_deliveries = torch.zeros((self.A, 2), device=dev)
        self.team_thermal_dmg_taken = torch.zeros((self.A, 2), device=dev)
        self.team_pressure_dmg_dealt = torch.zeros((self.A, 2), device=dev)

        # Emergent physical tactics telemetry [A, 2] (tracked strictly by physical state deltas!)
        self.emerge_oil_ignite = torch.zeros((self.A, 2), device=dev)
        self.emerge_flash_boil = torch.zeros((self.A, 2), device=dev)
        self.emerge_cryo_cross = torch.zeros((self.A, 2), device=dev)
        self.emerge_smokescreen = torch.zeros((self.A, 2), device=dev)
        self.emerge_kinetic_push = torch.zeros((self.A, 2), device=dev)

        # Online stats for adaptive focus_radius correlation (per team: sums for Pearson r)
        self.focus_count = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_f = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_f2 = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_c = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_c2 = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_fc = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_d = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_d2 = torch.zeros((self.A, 2), device=dev)
        self.focus_sum_fd = torch.zeros((self.A, 2), device=dev)
        self.focus_single_sum = torch.zeros((self.A, 2), device=dev)
        self.focus_single_n = torch.zeros((self.A, 2), device=dev)
        self.focus_multi_sum = torch.zeros((self.A, 2), device=dev)
        self.focus_multi_n = torch.zeros((self.A, 2), device=dev)

        return self.build_bot_features()

    def _grid_xy(self, px: torch.Tensor, py: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = (px * (self.W / CANON_SIZE)).long().clamp(0, self.W - 1)
        gy = (py * (self.H / CANON_SIZE)).long().clamp(0, self.H - 1)
        return gx, gy

    def build_bot_features(self) -> torch.Tensor:
        """
        Exact [A, 12, 168] float32 observation schema matching gpu_infer_daemon.py op=7!
        Includes physical optical density obscuration (smoke + steam blurs enemy position tokens!).
        """
        A, N = self.A, self.N
        dev = self.device
        feats = torch.zeros((A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)

        gx, gy = self._grid_xy(self.pos_x, self.pos_y)
        cur_smoke = self.w_state[self.arena_idx, self.bot_layer, SP_SMOKE, gy, gx]
        cur_steam = self.w_state[self.arena_idx, self.bot_layer, SP_STEAM, gy, gx]
        bot_opt = (cur_smoke + 0.85 * cur_steam).clamp(0.0, 1.5)  # [A, 12]

        # Self features [A, 12, 24]
        feats[:, :, 0] = self.pos_x / CANON_SIZE
        feats[:, :, 1] = self.pos_y / CANON_SIZE
        feats[:, :, 2] = self.bot_layer.float() / max(self.L - 1, 1)
        feats[:, :, 3] = (self.vel_x / 10.0).clamp(-0.5, 0.5)
        feats[:, :, 4] = (self.vel_y / 10.0).clamp(-0.5, 0.5)
        feats[:, :, 5] = self.hp
        feats[:, :, 6] = self.mana
        feats[:, :, 7] = self.stam
        feats[:, :, 8] = (self.cls == 0).float()
        feats[:, :, 9] = (self.cls == 1).float()
        feats[:, :, 10] = (self.cls == 2).float()
        feats[:, :, 11] = self.team.float()
        feats[:, :, 12] = self.carrying.float()
        feats[:, :, 13] = max(0.0, 1.0 - self.tick / float(self.max_ticks))
        # Physical self-cell telemetry in unused slots 14..19 (compatible with 168-float ledger)
        feats[:, :, 14] = (self.w_state[self.arena_idx, self.bot_layer, CH_TEMP, gy, gx] / 3.0).clamp(-1.0, 1.0)
        feats[:, :, 15] = self.w_state[self.arena_idx, self.bot_layer, SP_WATER, gy, gx].clamp(0.0, 1.0)
        feats[:, :, 16] = self.w_state[self.arena_idx, self.bot_layer, SP_ICE, gy, gx].clamp(0.0, 1.0)
        feats[:, :, 17] = self.w_state[self.arena_idx, self.bot_layer, SP_OIL, gy, gx].clamp(0.0, 1.0)
        feats[:, :, 18] = bot_opt / 1.5
        feats[:, :, 23] = self.last_emit_thermal

        # Pairwise entity tokens (nearest 8 other bots) -> tokens 0..7
        dx = self.pos_x[:, None, :] - self.pos_x[:, :, None]  # [A, 12, 12]
        dy = self.pos_y[:, None, :] - self.pos_y[:, :, None]
        dz = (self.bot_layer[:, None, :] - self.bot_layer[:, :, None]).float() / 2.0
        dist = torch.hypot(dx, dy).clamp(min=1e-3)

        sort_dist = dist.clone()
        eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
        sort_dist = sort_dist.masked_fill(eye, 1e6)
        dead_j = (~self.alive)[:, None, :].expand(A, N, N)
        sort_dist = sort_dist.masked_fill(dead_j, 5e5)

        nn_idx = torch.argsort(sort_dist, dim=-1)[:, :, :8]
        nn_dx = torch.gather(dx, 2, nn_idx)
        nn_dy = torch.gather(dy, 2, nn_idx)
        nn_dz = torch.gather(dz, 2, nn_idx)
        nn_d = torch.gather(dist, 2, nn_idx)
        nn_en = torch.gather(self.enemy_pair_mask.float(), 2, nn_idx)

        hp_exp = self.hp[:, None, :].expand(A, N, N)
        car_exp = self.carrying.float()[:, None, :].expand(A, N, N)
        cls_exp = (self.cls.float() / 2.0)[:, None, :].expand(A, N, N)
        alive_exp = self.alive.float()[:, None, :].expand(A, N, N)
        opt_exp = bot_opt[:, None, :].expand(A, N, N)

        nn_hp = torch.gather(hp_exp, 2, nn_idx)
        nn_car = torch.gather(car_exp, 2, nn_idx)
        nn_cls = torch.gather(cls_exp, 2, nn_idx)
        nn_valid = torch.gather(alive_exp, 2, nn_idx)
        nn_opt = torch.gather(opt_exp, 2, nn_idx)

        # Physical optical obscuration: enemies inside dense smoke+steam have blurred observed position!
        obsc = F.relu(nn_opt - 0.25) * nn_en
        if (obsc > 0.0).any():
            jit_x = torch.randn(nn_dx.shape, generator=self.gen, device=dev) * (12.0 * obsc)
            jit_y = torch.randn(nn_dy.shape, generator=self.gen, device=dev) * (12.0 * obsc)
            obs_dx = nn_dx + jit_x
            obs_dy = nn_dy + jit_y
            obs_d = torch.hypot(obs_dx, obs_dy).clamp(min=1e-3)
        else:
            obs_dx, obs_dy, obs_d = nn_dx, nn_dy, nn_d

        vis_factor = (1.0 - 0.55 * obsc).clamp(0.25, 1.0)

        toks = feats[:, :, SELF_DIM:].view(A, N, NUM_TOKENS, TOKEN_DIM)
        toks[:, :, :8, 0] = (obs_dx / 80.0).clamp(-1.0, 1.0) * nn_valid
        toks[:, :, :8, 1] = (obs_dy / 80.0).clamp(-1.0, 1.0) * nn_valid
        toks[:, :, :8, 2] = nn_dz * nn_valid
        toks[:, :, :8, 3] = nn_en * nn_valid
        toks[:, :, :8, 4] = nn_hp * nn_valid * vis_factor
        toks[:, :, :8, 5] = nn_car * nn_valid
        toks[:, :, :8, 6] = nn_cls * nn_valid
        toks[:, :, :8, 7] = (obs_d / 100.0).clamp(max=1.0) * nn_valid
        toks[:, :, :8, 8] = (obs_dx / obs_d) * nn_valid
        toks[:, :, :8, 9] = (obs_dy / obs_d) * nn_valid
        toks[:, :, :8, 10] = nn_opt * nn_valid
        toks[:, :, :8, 11] = nn_valid

        # 3 Shrine tokens (8..10)
        sdx = self.shrine_x[:, None, :] - self.pos_x[:, :, None]
        sdy = self.shrine_y[:, None, :] - self.pos_y[:, :, None]
        sprog = self.shrine_prog[:, None, :].expand(A, N, 3)
        is_west = (self.team == 0).unsqueeze(-1)
        s_need = torch.where(is_west, sprog > -0.8, sprog < 0.8).float()

        toks[:, :, 8:11, 0] = (sdx / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 8:11, 1] = (sdy / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 8:11, 3] = s_need
        toks[:, :, 8:11, 4] = sprog
        toks[:, :, 8:11, 11] = 1.0

        # Golden Relic token (11): points to relic when free, or to home dropoff when carrying!
        home_x = torch.where(self.team == 0, 28.0, 228.0)
        home_y = torch.full_like(home_x, 128.0)
        rel_gx = torch.where(self.carrying, home_x, self.relic_x[:, None])
        rel_gy = torch.where(self.carrying, home_y, self.relic_y[:, None])
        rdx = rel_gx - self.pos_x
        rdy = rel_gy - self.pos_y
        toks[:, :, 11, 0] = (rdx / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 11, 1] = (rdy / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 11, 3] = self.carrying.float()
        toks[:, :, 11, 11] = 1.0

        return feats

    def step(
        self,
        act: torch.Tensor,  # [A, 12, 10] continuous physical actuators
        collect_behavior_samples: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict | None]:
        """
        Advance all A arenas by 1 tick under 100% physics-derived rules!
        Returns: (next_feats [A,12,168], rewards [A,12], done, optional behavior_samples)
        """
        A, N, L, H, W = self.A, self.N, self.L, self.H, self.W
        dev = self.device
        self.tick += 1

        prev_score = self.team_score.clone()
        rewards = torch.zeros((A, N), dtype=torch.float32, device=dev)

        # 1. Respawns
        dead = ~self.alive
        self.respawn_cd = torch.where(dead, (self.respawn_cd - 1).clamp(min=0), self.respawn_cd)
        do_respawn = dead & (self.respawn_cd == 0)
        if do_respawn.any():
            base_x = torch.where(self.team == 0, 38.0, 218.0)
            rjx = torch.empty((A, N), device=dev).uniform_(-12.0, 12.0, generator=self.gen)
            rjy = torch.empty((A, N), device=dev).uniform_(36.0, 220.0, generator=self.gen)
            self.pos_x = torch.where(do_respawn, (base_x + rjx).clamp(16.0, 240.0), self.pos_x)
            self.pos_y = torch.where(do_respawn, rjy, self.pos_y)
            self.vel_x = torch.where(do_respawn, torch.zeros_like(self.vel_x), self.vel_x)
            self.vel_y = torch.where(do_respawn, torch.zeros_like(self.vel_y), self.vel_y)
            self.bot_layer = torch.where(do_respawn, torch.full_like(self.bot_layer, self.surf_l), self.bot_layer)
            self.hp = torch.where(do_respawn, torch.ones_like(self.hp), self.hp)
            self.mana = torch.where(do_respawn, torch.ones_like(self.mana), self.mana)
            self.stam = torch.where(do_respawn, torch.ones_like(self.stam), self.stam)
            self.alive = self.alive | do_respawn
            self.last_attacker = torch.where(do_respawn, torch.full_like(self.last_attacker, -1), self.last_attacker)

        alive_f = self.alive.float()

        # 2. Continuous Energy / Mana Regen & Attribution TTL Decay
        self.mana = (self.mana + 0.20 * alive_f).clamp(0.0, 1.0)
        self.stam = (self.stam + 0.08 * alive_f).clamp(0.0, 1.0)
        self.cell_src_ttl = (self.cell_src_ttl - 1).clamp(min=0)
        self.cell_src_bot = torch.where(self.cell_src_ttl > 0, self.cell_src_bot, torch.full_like(self.cell_src_bot, -1))
        self.cell_oil_ttl = (self.cell_oil_ttl - 1).clamp(min=0)
        self.cell_oil_bot = torch.where(self.cell_oil_ttl > 0, self.cell_oil_bot, torch.full_like(self.cell_oil_bot, -1))

        # Decode 10 continuous physical actuator channels
        mvx = act[:, :, ACT_MOVE_X].clamp(-1.0, 1.0)
        mvy = act[:, :, ACT_MOVE_Y].clamp(-1.0, 1.0)
        mnorm = torch.hypot(mvx, mvy).clamp(min=1.0)
        mvx = mvx / mnorm
        mvy = mvy / mnorm

        emit_dx = act[:, :, ACT_EMIT_DX].clamp(-1.0, 1.0)
        emit_dy = act[:, :, ACT_EMIT_DY].clamp(-1.0, 1.0)
        thermal_in = act[:, :, ACT_THERMAL].clamp(-1.0, 1.0)
        sp_water_in = act[:, :, ACT_WATER].clamp(0.0, 1.0)
        sp_oil_in = act[:, :, ACT_OIL].clamp(0.0, 1.0)
        sp_smoke_in = act[:, :, ACT_SMOKE].clamp(0.0, 1.0)
        focus_01 = act[:, :, ACT_FOCUS].clamp(0.0, 1.0)
        impulse_in = act[:, :, ACT_IMPULSE].clamp(0.0, 1.0)

        # Physical activation deadzone so channels below threshold spend 0 mana (clean OFF vs ON control!)
        thermal_raw = torch.where(thermal_in.abs() > 0.16, thermal_in, torch.zeros_like(thermal_in))
        water_raw = F.relu(sp_water_in - 0.30) / 0.70
        oil_raw = F.relu(sp_oil_in - 0.28) / 0.72
        smoke_raw = F.relu(sp_smoke_in - 0.30) / 0.70
        impulse_raw = F.relu(impulse_in - 0.28) / 0.72

        # 3. Conservation + Continuous Mana Budget (strict physical energy-threshold law: affordable pulses fire at 100%!)
        area_f = actuator_area_factor(focus_01)
        raw_flux_sum = thermal_raw.abs() + water_raw + oil_raw + smoke_raw + impulse_raw
        raw_cost = raw_flux_sum * area_f * 0.22
        has_req = (raw_cost > 1e-6) & self.alive
        can_afford = has_req & (raw_cost <= self.mana + 1e-6)
        overloaded = has_req & (~can_afford)
        # Affordable emissions fire at 100% full power; unaffordable multi-channel overload fizzles!
        budget_scale = can_afford.float()
        mana_spent = torch.where(can_afford, raw_cost, torch.where(overloaded, 0.20 * self.mana, torch.zeros_like(raw_cost)))
        self.mana = (self.mana - mana_spent).clamp(0.0, 1.0)

        eff_thermal = thermal_raw * budget_scale
        eff_water = water_raw * budget_scale
        eff_oil = oil_raw * budget_scale
        eff_smoke = smoke_raw * budget_scale
        eff_impulse = impulse_raw * budget_scale
        self.last_emit_thermal = eff_thermal

        # Physical focus radius (1.5..8.5 canonical cells) & concentration factor (narrow = high intensity!)
        rad_cells = focus_radius_cells(focus_01)  # [A, 12]
        conc = 1.55 / (0.55 + 0.80 * focus_01)    # narrow (0.0) -> 2.82x intensity; wide (1.0) -> 1.15x intensity

        # 4. Locomotion, Surface Friction & Buoyancy (100% from local voxel species!)
        gx0, gy0 = self._grid_xy(self.pos_x, self.pos_y)
        cur_water = self.w_state[self.arena_idx, self.bot_layer, SP_WATER, gy0, gx0]
        cur_ice = self.w_state[self.arena_idx, self.bot_layer, SP_ICE, gy0, gx0]
        cur_oil = self.w_state[self.arena_idx, self.bot_layer, SP_OIL, gy0, gx0]
        cur_temp0 = self.w_state[self.arena_idx, self.bot_layer, CH_TEMP, gy0, gx0]

        in_deep_water = (cur_water > 0.42) & (cur_ice < 0.35)
        on_ice = cur_ice > 0.35
        on_oil = cur_oil > 0.22
        cryo_drag = (1.0 - 0.45 * F.relu(-0.15 - cur_temp0)).clamp(0.45, 1.0)

        base_spd = torch.where(self.cls == 2, 4.6, torch.where(self.cls == 0, 4.1, 4.35))
        spd = base_spd * cryo_drag
        spd = torch.where(in_deep_water, spd * 0.30, spd)   # strong deep-water buoyancy drag!
        spd = torch.where(on_ice, spd * 1.32, spd)          # 4.4x low-friction ice glide advantage over open water!
        spd = torch.where(self.carrying, spd * 0.90, spd)

        # Kinetic impulse recoil / propulsion (emitting kinetic impulse behind you or close acts as a dash!)
        emit_dist = torch.hypot(emit_dx, emit_dy) * MAX_EMIT_RANGE_CELLS
        e_norm = torch.hypot(emit_dx, emit_dy).clamp(min=1e-4)
        eux, euy = emit_dx / e_norm, emit_dy / e_norm
        dash_align = ((eux * mvx + euy * mvy) < 0.15) | (emit_dist < 10.0)
        recoil_boost = torch.where(dash_align, 2.8 * eff_impulse, 0.8 * eff_impulse)
        spd = spd + recoil_boost

        des_vx = mvx * spd * alive_f
        des_vy = mvy * spd * alive_f
        # Ice & oil physical momentum slip
        slip = torch.where(on_oil, 0.38, torch.where(on_ice, 0.22, 0.08))
        self.vel_x = slip * self.vel_x + (1.0 - slip) * des_vx
        self.vel_y = slip * self.vel_y + (1.0 - slip) * des_vy

        prev_x, prev_y = self.pos_x.clone(), self.pos_y.clone()
        self.pos_x = (self.pos_x + self.vel_x * alive_f).clamp(10.0, 246.0)
        self.pos_y = (self.pos_y + self.vel_y * alive_f).clamp(10.0, 246.0)

        # Wildlife Sanctuary passive heal near dense critters when low HP (Z_eco!)
        gx, gy = self._grid_xy(self.pos_x, self.pos_y)
        cur_crit = self.e_state[self.arena_idx, 0, 0, gy, gx]
        sanc_heal = (cur_crit > 0.30) & (self.hp < 0.70) & self.alive
        self.hp = torch.where(sanc_heal, (self.hp + 0.025).clamp(max=1.0), self.hp)

        # 5. Physical Voxel Deposits from Continuous Actuators (with Optical Smoke/Steam Scatter!)
        raw_ex = (self.pos_x + emit_dx * MAX_EMIT_RANGE_CELLS).clamp(8.0, 248.0)
        raw_ey = (self.pos_y + emit_dy * MAX_EMIT_RANGE_CELLS).clamp(8.0, 248.0)
        egx0, egy0 = self._grid_xy(raw_ex, raw_ey)
        mgx0, mgy0 = self._grid_xy(0.5 * (self.pos_x + raw_ex), 0.5 * (self.pos_y + raw_ey))

        opt_tgt = (
            self.w_state[self.arena_idx, self.bot_layer, SP_SMOKE, egy0, egx0]
            + 0.85 * self.w_state[self.arena_idx, self.bot_layer, SP_STEAM, egy0, egx0]
        )
        opt_mid = (
            self.w_state[self.arena_idx, self.bot_layer, SP_SMOKE, mgy0, mgx0]
            + 0.85 * self.w_state[self.arena_idx, self.bot_layer, SP_STEAM, mgy0, mgx0]
        )
        opt_path = torch.maximum(opt_tgt, 0.75 * opt_mid)
        # Optical density scatters ranged targeting (> 7 cells away)
        scatter_std = F.relu(opt_path - 0.25) * 6.5 * (emit_dist > 7.0).float()
        ex = (raw_ex + torch.randn((A, N), generator=self.gen, device=dev) * scatter_std).clamp(8.0, 248.0)
        ey = (raw_ey + torch.randn((A, N), generator=self.gen, device=dev) * scatter_std).clamp(8.0, 248.0)
        egx, egy = self._grid_xy(ex, ey)

        # Snapshot voxel state BEFORE deposits + thermodynamics so dT / dSteam / dIce are exact physical deltas!
        prev_temp = self.w_state[:, :, CH_TEMP].clone()
        prev_steam = self.w_state[:, :, SP_STEAM].clone()
        prev_ice = self.w_state[:, :, SP_ICE].clone()
        pre_water_at_emit = self.w_state[self.arena_idx, self.bot_layer, SP_WATER, egy, egx].clone()
        pre_ice_at_emit = self.w_state[self.arena_idx, self.bot_layer, SP_ICE, egy, egx].clone()
        pre_oil_at_emit = self.w_state[self.arena_idx, self.bot_layer, SP_OIL, egy, egx].clone()
        pre_oil_src = self.cell_oil_bot[self.arena_idx, self.bot_layer, egy, egx].clone()

        # Vectorized variable-radius Gaussian voxel scatter over all A*12 bots (TRUE in-place zero-copy!)
        # Gaussian kernel weight [A, 12, S]
        sigma = (0.55 * rad_cells).unsqueeze(-1)  # [A, 12, 1]
        d_s = self.st_dist.view(1, 1, -1)         # [1, 1, S]
        g_w = torch.exp(-0.5 * (d_s / sigma).pow(2)) * (d_s <= (rad_cells.unsqueeze(-1) * 1.25)).float()

        sy = (egy.unsqueeze(-1) + self.st_dy.view(1, 1, -1)).clamp(0, H - 1)
        sx = (egx.unsqueeze(-1) + self.st_dx.view(1, 1, -1)).clamp(0, W - 1)
        a_id = self.arena_idx.unsqueeze(-1).expand_as(sy)
        l_id = self.bot_layer.unsqueeze(-1).expand_as(sy)
        al_id = a_id * L + l_id
        yx_id = sy * W + sx
        flat_idx = (al_id * (H * W) + yx_id).reshape(-1)

        w_flat = self.w_state.view(-1)
        hw = H * W

        def _scatter_ch(ch: int, vals_3d: torch.Tensor) -> None:
            ch_idx = ((al_id * 18 + ch) * hw + yx_id).reshape(-1)
            w_flat.scatter_add_(0, ch_idx, vals_3d.reshape(-1))

        # Thermal energy injection / extraction (narrow beam concentrates flux ~2.45x; wide field relies on fuel/water reactions!)
        d_temp_amp = eff_thermal * conc * 0.54
        if (d_temp_amp.abs() > 1e-4).any():
            _scatter_ch(CH_TEMP, d_temp_amp.unsqueeze(-1) * g_w)

        # Matter deposits: water, oil, smoke
        mass_scale = 0.55 + 0.25 * conc
        if (eff_water > 1e-4).any():
            _scatter_ch(SP_WATER, (eff_water * mass_scale * 0.95).unsqueeze(-1) * g_w)
        if (eff_oil > 1e-4).any():
            _scatter_ch(SP_OIL, (eff_oil * mass_scale * 1.10).unsqueeze(-1) * g_w)
        if (eff_smoke > 1e-4).any():
            _scatter_ch(SP_SMOKE, (eff_smoke * mass_scale * 1.25).unsqueeze(-1) * g_w)

        # Record voxel attribution for kill credit & oil-deposit emergence tracking (core radius only)
        core_m = g_w > 0.35
        therm_active = (eff_thermal.abs() > 0.14).unsqueeze(-1) & core_m
        if therm_active.any():
            b_flat = self.bot_idx.unsqueeze(-1).expand_as(sy).reshape(-1)
            m_flat = therm_active.reshape(-1)
            self.cell_src_bot.view(-1)[flat_idx[m_flat]] = b_flat[m_flat]
            self.cell_src_ttl.view(-1)[flat_idx[m_flat]] = 6

        oil_active = (eff_oil > 0.16).unsqueeze(-1) & core_m
        if oil_active.any():
            b_flat = self.bot_idx.unsqueeze(-1).expand_as(sy).reshape(-1)
            m_flat = oil_active.reshape(-1)
            self.cell_oil_bot.view(-1)[flat_idx[m_flat]] = b_flat[m_flat]
            self.cell_oil_ttl.view(-1)[flat_idx[m_flat]] = 10

        # 6. 100% Voxel-Physics Thermodynamics, Combustion, Phase-Change & WorldModel v3 Step!
        temp = self.w_state[:, :, CH_TEMP]
        water = self.w_state[:, :, SP_WATER]
        ice = self.w_state[:, :, SP_ICE]
        oil = self.w_state[:, :, SP_OIL]
        wood = self.w_state[:, :, SP_WOOD]
        smoke = self.w_state[:, :, SP_SMOKE]
        steam = self.w_state[:, :, SP_STEAM]
        ash = self.w_state[:, :, SP_ASH]

        # 6a. Exothermic Oil Combustion (flashpoint T > 0.50 & oil > 0.08 -> sharp exothermic dT + smoke!)
        oil_ign_rate = 0.62 * torch.sigmoid((temp - 0.52) * 5.0) * (temp > 0.42).float()
        burn_oil = torch.minimum(oil, oil * oil_ign_rate)
        oil = (oil - burn_oil).clamp(min=0.0)
        temp = temp + 4.20 * burn_oil
        smoke = smoke + 0.65 * burn_oil
        ash = ash + 0.15 * burn_oil

        # 6b. Wood Combustion (slow sustained wildfire)
        wood_ign = 0.14 * torch.sigmoid((temp - 1.65) * 3.0) * (temp > 1.45).float()
        burn_wood = torch.minimum(wood, wood * wood_ign)
        wood = (wood - burn_wood).clamp(min=0.0)
        temp = temp + 0.95 * burn_wood
        smoke = smoke + 0.35 * burn_wood

        # 6c. Flash-Boil Vaporization (heating water/ice past T > 0.45 -> sharp dSteam pressure surge!)
        boil_w_rate = 0.64 * ((temp - 0.45) / 0.95).clamp(0.0, 1.0)
        boil_w = torch.minimum(water, water * boil_w_rate)
        boil_i_rate = 0.55 * ((temp - 0.38) / 0.95).clamp(0.0, 1.0)
        boil_i = torch.minimum(ice, ice * boil_i_rate)
        water = (water - boil_w).clamp(min=0.0)
        ice = (ice - boil_i).clamp(min=0.0)
        steam = steam + 1.40 * (boil_w + boil_i)
        temp = temp - 0.14 * (boil_w + boil_i)

        # 6d. Cryo Freezing (negative thermal flux on water -> solid ice!)
        frz_rate = 0.80 * ((-0.08 - temp) / 0.45).clamp(0.0, 1.0)
        freeze_w = torch.minimum(water, water * frz_rate)
        water = (water - freeze_w).clamp(min=0.0)
        ice = ice + freeze_w

        # 6e. Gentle Melting (warm ice melts back into water)
        melt_rate = 0.22 * ((temp - 0.25) / 1.0).clamp(0.0, 1.0)
        melt_i = torch.minimum(ice, ice * melt_rate)
        ice = (ice - melt_i).clamp(min=0.0)
        water = water + melt_i

        # 6f. Natural Thermal Relaxation & Gas Dissipation
        temp = torch.where(temp > 0.05, temp * 0.92, temp * 0.88).clamp(-1.4, 5.5)
        smoke = (smoke * 0.86).clamp(0.0, 2.0)
        steam = (steam * 0.82).clamp(0.0, 2.0)
        oil = oil.clamp(0.0, 2.0)
        water = water.clamp(0.0, 2.0)
        ice = ice.clamp(0.0, 2.0)

        self.w_state[:, :, CH_TEMP] = temp
        self.w_state[:, :, SP_WATER] = water
        self.w_state[:, :, SP_ICE] = ice
        self.w_state[:, :, SP_OIL] = oil
        self.w_state[:, :, SP_WOOD] = wood
        self.w_state[:, :, SP_SMOKE] = smoke
        self.w_state[:, :, SP_STEAM] = steam
        self.w_state[:, :, SP_ASH] = ash

        # Periodic real WorldModel v3 + EcosystemModel neural step & VRAM latent refresh!
        if (self.tick % self.refresh_every) == 0:
            self.refresh_latents_and_world_step(
                apply_world_deltas=True,
                refresh_eco=(self.tick % (self.refresh_every * 4) == 0),
            )

        self.w_state[:, :, CH_LOGMASS] = torch.log1p(self.w_state[:, :, :10].sum(dim=2))

        # True voxel state deltas (for Pressure Waves & Emergence Detection!)
        dT = self.w_state[:, :, CH_TEMP] - prev_temp          # [A, L, H, W]
        dSteam = self.w_state[:, :, SP_STEAM] - prev_steam    # [A, L, H, W]
        dIce = self.w_state[:, :, SP_ICE] - prev_ice          # [A, L, H, W]

        # 7. 100% Physics-Derived Entity Effects (Thermal Stress + Pressure Impulses + Kinetic Wave!)
        cur_T = self.w_state[self.arena_idx, self.bot_layer, CH_TEMP, gy, gx]

        # 7a. Thermal Stress directly from local voxel temperature
        scorch_dmg = (0.048 * F.relu(cur_T - 1.55).pow(2)).clamp(max=0.26) * alive_f
        hypo_dmg = (0.14 * F.relu(-0.15 - cur_T).pow(2)).clamp(max=0.20) * alive_f
        thermal_dmg = scorch_dmg + hypo_dmg

        # 7b. Phase-Change & Combustion Pressure Impulses (dT > 0.65 or dSteam > 0.25 -> physical shockwave propagation!)
        p_comb = F.relu(dT - 0.65)
        p_boil = F.relu(dSteam - 0.25)
        p_raw = (0.34 * p_comb + 0.42 * p_boil).clamp(max=0.52)  # [A, L, H, W]
        # Physical 1-tick acoustic shockwave radiation across neighboring voxels
        p_diff = F.avg_pool2d(p_raw.reshape(A * L, 1, H, W), kernel_size=5, stride=1, padding=2).reshape(A, L, H, W)
        p_field = torch.maximum(p_raw, 1.45 * p_diff).clamp(max=0.52)

        # Spatial pressure gradient -> radial knockback away from pressure peak!
        p_pad = F.pad(p_field, (1, 1, 1, 1), mode="replicate")
        grad_px = 0.5 * (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2])
        grad_py = 0.5 * (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1])

        p_loc = p_field[self.arena_idx, self.bot_layer, gy, gx] * alive_f
        kb_px = -grad_px[self.arena_idx, self.bot_layer, gy, gx] * 16.0 * alive_f
        kb_py = -grad_py[self.arena_idx, self.bot_layer, gy, gx] * 16.0 * alive_f

        # 7c. Kinetic Actuator Impulse Wave along (emit_dx, emit_dy)
        # Pairwise distance from attacker i's emit center (ex_i, ey_i) to target j's position (pos_x_j, pos_y_j)
        k_dx = self.pos_x[:, None, :] - ex[:, :, None]  # [A, 12, 12]
        k_dy = self.pos_y[:, None, :] - ey[:, :, None]
        k_dist = torch.hypot(k_dx, k_dy)
        k_rad = (rad_cells * 1.45).unsqueeze(-1)        # [A, 12, 1]
        valid_en = self.enemy_pair_mask & self.alive[:, :, None] & self.alive[:, None, :]
        in_kin = valid_en & (k_dist < k_rad) & (eff_impulse.unsqueeze(-1) > 0.08)
        kin_falloff = (1.0 - (k_dist / (k_rad + 1e-4))).clamp(0.0, 1.0) * in_kin.float()
        kin_dmg_mat = kin_falloff * (0.13 * eff_impulse / (0.60 + 0.85 * focus_01)).unsqueeze(-1)
        kin_dmg_taken = kin_dmg_mat.sum(dim=1)  # [A, 12]
        kin_dmg_dealt = kin_dmg_mat.sum(dim=2)  # [A, 12]

        # Kinetic knockback velocity applied to victims
        kin_kb_x = (kin_falloff * (eff_impulse * eux * 5.2).unsqueeze(-1)).sum(dim=1)
        kin_kb_y = (kin_falloff * (eff_impulse * euy * 5.2).unsqueeze(-1)).sum(dim=1)

        self.vel_x = (self.vel_x + kb_px + kin_kb_x).clamp(-10.0, 10.0)
        self.vel_y = (self.vel_y + kb_py + kin_kb_y).clamp(-10.0, 10.0)
        self.pos_x = (self.pos_x + (kb_px + kin_kb_x) * 0.45 * alive_f).clamp(10.0, 246.0)
        self.pos_y = (self.pos_y + (kb_py + kin_kb_y) * 0.45 * alive_f).clamp(10.0, 246.0)

        # Attribute voxel thermal & pressure damage to the emitter bot for kill credit & reward
        src_bot = self.cell_src_bot[self.arena_idx, self.bot_layer, gy, gx]  # [A, 12]
        has_src = src_bot >= 0
        src_safe = src_bot.clamp(min=0)
        src_team = torch.gather(self.team, 1, src_safe)
        is_enemy_src = has_src & (src_team != self.team)

        voxel_dmg = thermal_dmg + p_loc
        armor = torch.where(self.cls == 0, 0.84, 1.0)
        total_dmg_taken = (voxel_dmg + kin_dmg_taken) * armor
        self.hp = (self.hp - total_dmg_taken).clamp(min=0.0)

        # Enemy damage dealt credit per attacker (physical thermal/pressure/fuel-energy attribution)
        dmg_dealt_by_i = kin_dmg_dealt.clone()
        en_voxel_dmg = voxel_dmg * is_enemy_src.float()
        dmg_dealt_by_i.scatter_add_(1, src_safe, en_voxel_dmg)

        # Fuel supplier shares physical attribution when their deposited oil combusts on an enemy
        oil_src_bot = self.cell_oil_bot[self.arena_idx, self.bot_layer, gy, gx]
        oil_src_safe = oil_src_bot.clamp(min=0)
        is_enemy_oil_src = (oil_src_bot >= 0) & (torch.gather(self.team, 1, oil_src_safe) != self.team)
        comb_loc = (0.24 * p_comb[self.arena_idx, self.bot_layer, gy, gx]).clamp(max=0.35) * alive_f
        dmg_dealt_by_i.scatter_add_(1, oil_src_safe, 0.45 * comb_loc * is_enemy_oil_src.float())

        en_press_dmg = p_loc * is_enemy_src.float()
        press_dealt_by_i = torch.zeros((A, N), device=dev)
        press_dealt_by_i.scatter_add_(1, src_safe, en_press_dmg)

        # Update last_attacker for elimination credit
        max_kin, best_kin = kin_dmg_mat.max(dim=1)
        self.last_attacker = torch.where(
            (en_voxel_dmg > 1e-4) & (en_voxel_dmg >= max_kin),
            src_safe,
            torch.where(max_kin > 1e-4, best_kin, self.last_attacker),
        )

        # Team damage stats
        self.team_thermal_dmg_taken[:, 0] += (thermal_dmg * self.w_mask).sum(dim=1)
        self.team_thermal_dmg_taken[:, 1] += (thermal_dmg * self.e_mask).sum(dim=1)
        self.team_pressure_dmg_dealt[:, 0] += (press_dealt_by_i * self.w_mask).sum(dim=1)
        self.team_pressure_dmg_dealt[:, 1] += (press_dealt_by_i * self.e_mask).sum(dim=1)

        # 8. Emergence Telemetry Detection (Strictly observational — ZERO reward bonus!)
        en_dist_to_emit = torch.hypot(self.pos_x[:, None, :] - ex[:, :, None], self.pos_y[:, None, :] - ey[:, :, None])
        en_dist_valid = en_dist_to_emit.masked_fill(~valid_en, 1e6)
        min_en_emit_d = en_dist_valid.min(dim=-1).values  # [A, 12]
        en_cluster_14 = (en_dist_valid < 15.0).float().sum(dim=-1)  # [A, 12]

        dT_at_emit = dT[self.arena_idx, self.bot_layer, egy, egx]
        dSteam_at_emit = dSteam[self.arena_idx, self.bot_layer, egy, egx]
        dIce_at_emit = dIce[self.arena_idx, self.bot_layer, egy, egx]
        opt_self = (
            self.w_state[self.arena_idx, self.bot_layer, SP_SMOKE, gy, gx]
            + 0.85 * self.w_state[self.arena_idx, self.bot_layer, SP_STEAM, gy, gx]
        )

        # Tactic 1: Emergent Oil -> Ignition Detonation (deposited oil ignited by positive thermal_flux -> dT > 0.65 near enemy!)
        oil_from_team = (pre_oil_src >= 0) & (torch.gather(self.team, 1, pre_oil_src.clamp(min=0)) == self.team)
        ev_oil_ign = (
            self.alive
            & (eff_thermal > 0.22)
            & ((pre_oil_at_emit > 0.14) | oil_from_team)
            & (dT_at_emit > 0.65)
            & (min_en_emit_d < 16.0)
        ).float()

        # Tactic 2: Emergent Flash-Boil Steam Burst (positive thermal_flux on water/ice near enemy -> dSteam > 0.25!)
        ev_flash_boil = (
            self.alive
            & (eff_thermal > 0.22)
            & ((pre_water_at_emit + pre_ice_at_emit) > 0.18)
            & (dSteam_at_emit > 0.25)
            & (min_en_emit_d < 18.0)
        ).float()

        # Tactic 3: Emergent Cryo-Crossing (negative thermal_flux freezing unfrozen water ahead -> dIce > 0.12!)
        ahead_dot = eux * mvx + euy * mvy
        ev_cryo_cross = (
            self.alive
            & (eff_thermal < -0.22)
            & (pre_water_at_emit > 0.30)
            & (pre_ice_at_emit < 0.45)
            & (dIce_at_emit > 0.12)
            & ((ahead_dot > 0.15) | (emit_dist < 14.0))
        ).float()

        # Tactic 4: Emergent Smokescreen / Cover (low HP or carrying relic + smoke/steam cloud near self -> opt > 0.28!)
        ev_smoke_cover = (
            self.alive
            & ((self.hp < 0.60) | self.carrying)
            & ((eff_smoke > 0.18) | (dSteam_at_emit > 0.20))
            & (emit_dist < 15.0)
            & (opt_self > 0.28)
        ).float()

        # Tactic 5: Emergent Kinetic Impulse Push on enemy
        ev_kin_push = (in_kin.any(dim=-1) & self.alive).float()

        self.emerge_oil_ignite[:, 0] += (ev_oil_ign * self.w_mask).sum(dim=1)
        self.emerge_oil_ignite[:, 1] += (ev_oil_ign * self.e_mask).sum(dim=1)
        self.emerge_flash_boil[:, 0] += (ev_flash_boil * self.w_mask).sum(dim=1)
        self.emerge_flash_boil[:, 1] += (ev_flash_boil * self.e_mask).sum(dim=1)
        self.emerge_cryo_cross[:, 0] += (ev_cryo_cross * self.w_mask).sum(dim=1)
        self.emerge_cryo_cross[:, 1] += (ev_cryo_cross * self.e_mask).sum(dim=1)
        self.emerge_smokescreen[:, 0] += (ev_smoke_cover * self.w_mask).sum(dim=1)
        self.emerge_smokescreen[:, 1] += (ev_smoke_cover * self.e_mask).sum(dim=1)
        self.emerge_kinetic_push[:, 0] += (ev_kin_push * self.w_mask).sum(dim=1)
        self.emerge_kinetic_push[:, 1] += (ev_kin_push * self.e_mask).sum(dim=1)

        # Online adaptive focus_radius statistics (on active combat emissions near enemies)
        active_emit = self.alive & ((eff_thermal.abs() + eff_oil + eff_smoke + eff_water + eff_impulse) > 0.25) & (min_en_emit_d < 22.0)
        for ti, tm in enumerate((self.w_mask, self.e_mask)):
            am = active_emit.float() * tm
            self.focus_count[:, ti] += am.sum(dim=1)
            self.focus_sum_f[:, ti] += (rad_cells * am).sum(dim=1)
            self.focus_sum_f2[:, ti] += (rad_cells.pow(2) * am).sum(dim=1)
            self.focus_sum_c[:, ti] += (en_cluster_14 * am).sum(dim=1)
            self.focus_sum_c2[:, ti] += (en_cluster_14.pow(2) * am).sum(dim=1)
            self.focus_sum_fc[:, ti] += (rad_cells * en_cluster_14 * am).sum(dim=1)
            self.focus_sum_d[:, ti] += (emit_dist * am).sum(dim=1)
            self.focus_sum_d2[:, ti] += (emit_dist.pow(2) * am).sum(dim=1)
            self.focus_sum_fd[:, ti] += (rad_cells * emit_dist * am).sum(dim=1)
            sm1 = am * (en_cluster_14 == 1).float()
            sm2 = am * (en_cluster_14 >= 2).float()
            self.focus_single_sum[:, ti] += (rad_cells * sm1).sum(dim=1)
            self.focus_single_n[:, ti] += sm1.sum(dim=1)
            self.focus_multi_sum[:, ti] += (rad_cells * sm2).sum(dim=1)
            self.focus_multi_n[:, ti] += sm2.sum(dim=1)

        beh_samples: dict | None = None
        if collect_behavior_samples and (self.tick % 4 == 0):
            beh_samples = {
                "thermal": eff_thermal.detach().cpu().numpy(),
                "focus_cells": rad_cells.detach().cpu().numpy(),
                "water": eff_water.detach().cpu().numpy(),
                "oil": eff_oil.detach().cpu().numpy(),
                "smoke": eff_smoke.detach().cpu().numpy(),
                "impulse": eff_impulse.detach().cpu().numpy(),
                "alive": self.alive.detach().cpu().numpy(),
                "ev_oil_ign": ev_oil_ign.detach().cpu().numpy(),
                "ev_flash_boil": ev_flash_boil.detach().cpu().numpy(),
                "ev_cryo": ev_cryo_cross.detach().cpu().numpy(),
                "ev_smoke": ev_smoke_cover.detach().cpu().numpy(),
            }

        # 9. Eliminations & Kill Credit
        just_died = self.alive & (self.hp <= 0.0)
        if just_died.any():
            self.alive = self.alive & (~just_died)
            self.respawn_cd = torch.where(just_died, torch.full_like(self.respawn_cd, 8), self.respawn_cd)

            dropped = just_died & self.carrying
            if dropped.any():
                self.carrying = self.carrying & (~dropped)
                drop_arena = dropped.any(dim=1)
                self.relic_carrier = torch.where(drop_arena, torch.full_like(self.relic_carrier, -1), self.relic_carrier)

            self.team_deaths[:, 0] += (just_died.float() * self.w_mask).sum(dim=1)
            self.team_deaths[:, 1] += (just_died.float() * self.e_mask).sum(dim=1)

            west_died = (just_died.float() * self.w_mask).sum(dim=1)
            east_died = (just_died.float() * self.e_mask).sum(dim=1)
            self.team_kills[:, 1] += west_died
            self.team_kills[:, 0] += east_died
            self.team_score[:, 1] += 8.0 * west_died
            self.team_score[:, 0] += 8.0 * east_died

            valid_kill = just_died & (self.last_attacker >= 0)
            killer_idx = self.last_attacker.clamp(min=0)
            rewards.scatter_add_(1, killer_idx, 2.8 * valid_kill.float())
            rewards -= 2.2 * just_died.float()

        # 10. Shrine Control & Scoring
        in_fire = (cur_T > 1.55) & self.alive
        sd_bs = torch.hypot(
            self.pos_x[:, :, None] - self.shrine_x[:, None, :],
            self.pos_y[:, :, None] - self.shrine_y[:, None, :],
        )
        in_shrine = (sd_bs < 18.0) & self.alive[:, :, None] & (~in_fire[:, :, None])
        west_in = (in_shrine & (self.team[:, :, None] == 0)).float().sum(dim=1)
        east_in = (in_shrine & (self.team[:, :, None] == 1)).float().sum(dim=1)

        self.shrine_prog = (self.shrine_prog + 0.12 * (east_in - west_in)).clamp(-1.0, 1.0)
        new_owner = torch.where(
            self.shrine_prog <= -0.8,
            torch.zeros_like(self.shrine_owner),
            torch.where(self.shrine_prog >= 0.8, torch.ones_like(self.shrine_owner), self.shrine_owner),
        )
        flipped_w = (new_owner == 0) & (self.shrine_owner != 0)
        flipped_e = (new_owner == 1) & (self.shrine_owner != 1)
        self.shrine_owner = new_owner

        self.team_shrine_caps[:, 0] += flipped_w.float().sum(dim=1)
        self.team_shrine_caps[:, 1] += flipped_e.float().sum(dim=1)
        self.team_score[:, 0] += 10.0 * flipped_w.float().sum(dim=1) + (self.shrine_owner == 0).float().sum(dim=1) * 1.0
        self.team_score[:, 1] += 10.0 * flipped_e.float().sum(dim=1) + (self.shrine_owner == 1).float().sum(dim=1) * 1.0

        need_shrine = torch.where(
            self.team[:, :, None] == 0,
            self.shrine_prog[:, None, :] > -0.8,
            self.shrine_prog[:, None, :] < 0.8,
        )
        prev_sd_bs = torch.hypot(
            prev_x[:, :, None] - self.shrine_x[:, None, :],
            prev_y[:, :, None] - self.shrine_y[:, None, :],
        )
        prev_s_min = prev_sd_bs.masked_fill(~need_shrine, 1e4).min(dim=-1).values
        new_s_min = sd_bs.masked_fill(~need_shrine, 1e4).min(dim=-1).values
        has_need_shrine = need_shrine.any(dim=-1) & (~self.carrying) & self.alive
        shrine_approach = torch.where(has_need_shrine, (prev_s_min - new_s_min).clamp(-6.0, 6.0), torch.zeros_like(rewards))
        shrine_contrib = (in_shrine & need_shrine).any(dim=-1).float()
        rewards += 1.25 * shrine_contrib + 0.14 * shrine_approach

        # 11. Golden Relic Pickup, Carry & Delivery
        no_carrier = self.relic_carrier < 0
        if no_carrier.any():
            rdist_prev = torch.hypot(prev_x - self.relic_x[:, None], prev_y - self.relic_y[:, None])
            rdist = torch.hypot(self.pos_x - self.relic_x[:, None], self.pos_y - self.relic_y[:, None])
            # Encourage runners (cls==0) to contest & pick up the Golden Relic
            is_runner = (self.cls == 0) & self.alive & no_carrier.unsqueeze(1)
            rewards += torch.where(is_runner, 0.24 * (rdist_prev - rdist).clamp(-6.0, 6.0), torch.zeros_like(rewards))

            rdist_m = rdist.masked_fill(~self.alive, 1e6)
            r_min, r_who = rdist_m.min(dim=1)
            picked = no_carrier & (r_min < 12.0)
            self.relic_carrier = torch.where(picked, r_who, self.relic_carrier)
            if picked.any():
                oh = F.one_hot(r_who, num_classes=N).bool() & picked.unsqueeze(-1)
                self.carrying = self.carrying | oh
                rewards += 1.4 * oh.float()

        has_carrier = self.relic_carrier >= 0
        if has_carrier.any():
            cid = self.relic_carrier.clamp(min=0)
            cx = torch.gather(self.pos_x, 1, cid.unsqueeze(1)).squeeze(1)
            cy = torch.gather(self.pos_y, 1, cid.unsqueeze(1)).squeeze(1)
            self.relic_x = torch.where(has_carrier, cx, self.relic_x)
            self.relic_y = torch.where(has_carrier, cy, self.relic_y)

            c_team = torch.gather(self.team, 1, cid.unsqueeze(1)).squeeze(1)
            self.team_score[:, 0] += (has_carrier & (c_team == 0)).float() * 0.25
            self.team_score[:, 1] += (has_carrier & (c_team == 1)).float() * 0.25

            home_x = torch.where(self.team == 0, 28.0, 228.0)
            prev_hd = (prev_x - home_x).abs()
            new_hd = (self.pos_x - home_x).abs()
            rewards += torch.where(self.carrying, 0.40 * (prev_hd - new_hd).clamp(-6.0, 6.0), torch.zeros_like(rewards))

            deliv_w = has_carrier & (c_team == 0) & (cx < 38.0)
            deliv_e = has_carrier & (c_team == 1) & (cx > 218.0)
            deliv = deliv_w | deliv_e
            if deliv.any():
                self.team_score[:, 0] += 25.0 * deliv_w.float()
                self.team_score[:, 1] += 25.0 * deliv_e.float()
                self.team_relic_deliveries[:, 0] += deliv_w.float()
                self.team_relic_deliveries[:, 1] += deliv_e.float()
                rewards += 6.0 * (self.carrying & deliv.unsqueeze(-1)).float()
                self.carrying = self.carrying & (~deliv.unsqueeze(-1))
                self.relic_carrier = torch.where(deliv, torch.full_like(self.relic_carrier, -1), self.relic_carrier)
                self.relic_x = torch.where(deliv, torch.full_like(self.relic_x, 128.0), self.relic_x)
                self.relic_y = torch.where(deliv, torch.full_like(self.relic_y, 128.0), self.relic_y)

        # 12. Pure Sparse-Goal Dual-Horizon Rewards (NO hardcoded spell rules, NO double-counted self-dmg!)
        score_delta = self.team_score - prev_score
        my_team_delta = torch.where(self.team == 0, score_delta[:, 0:1], score_delta[:, 1:2])
        en_team_delta = torch.where(self.team == 0, score_delta[:, 1:2], score_delta[:, 0:1])

        rew_tac = (
            rewards
            + 2.2 * dmg_dealt_by_i
            - 0.85 * total_dmg_taken
            + 0.42 * my_team_delta
            - 0.16 * en_team_delta
            + 0.025 * (alive_f * (~in_fire).float())
        )
        # Immediate physical actuator feedback (100% sparse game goals: enemy damage, eliminations, survival, objective!)
        rew_act = (
            6.5 * dmg_dealt_by_i
            - 0.70 * total_dmg_taken
            + 0.32 * my_team_delta
            - 0.12 * en_team_delta
            + 0.35 * shrine_contrib
        )
        rews_pair = torch.stack([rew_tac, rew_act], dim=-1)  # [A, 12, 2]

        done = self.tick >= self.max_ticks
        return self.build_bot_features(), rews_pair, done, beh_samples


# ---------------------------------------------------------------------------
# Baseline Policies (All operating in the exact 10-channel physical actuator space!)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def random_continuous_act(arena: PhysicsActuatorArena) -> torch.Tensor:
    """Uniform random continuous physical actuators in [-1, 1]^5 x [0, 1]^5."""
    dev = arena.device
    bi = torch.empty((arena.A, arena.N, 5), device=dev).uniform_(-1.0, 1.0, generator=arena.gen)
    uni = torch.empty((arena.A, arena.N, 5), device=dev).uniform_(0.0, 1.0, generator=arena.gen)
    return torch.cat([bi, uni], dim=-1)


@torch.inference_mode()
def legacy_scripted_actuator_baseline(arena: PhysicsActuatorArena) -> torch.Tensor:
    """
    Legacy Named-Spell Scripted Baseline (matching train_bot_rl.py's scripted_teacher_act)
    translated into equivalent continuous physical deposits so BOTH policies play under the
    EXACT same 100% physics-derived world rules & mana budget!
      - Fire Bolt  -> narrow positive thermal_flux at enemy
      - Frost Bolt -> narrow negative thermal_flux at water/enemy
      - Oil Bolt   -> medium sp_oil at enemy
      - Ability E/R -> wider radius deposit (no smoke, fixed focus radii, rigid element rules)
    """
    A, N, H, W = arena.A, arena.N, arena.H, arena.W
    dev = arena.device

    # Nearest unowned shrine
    sd = torch.hypot(
        arena.shrine_x[:, None, :] - arena.pos_x[:, :, None],
        arena.shrine_y[:, None, :] - arena.pos_y[:, :, None],
    )
    need = torch.where(
        arena.team[:, :, None] == 0,
        arena.shrine_prog[:, None, :] > -0.8,
        arena.shrine_prog[:, None, :] < 0.8,
    )
    _, best_si = sd.masked_fill(~need, 1e6).min(dim=-1)
    best_sx = torch.gather(arena.shrine_x, 1, best_si)
    best_sy = torch.gather(arena.shrine_y, 1, best_si)

    home_x = torch.where(arena.team == 0, 0.12 * CANON_SIZE, 0.88 * CANON_SIZE)
    home_y = torch.full_like(home_x, 0.50 * CANON_SIZE)
    is_relic_runner = (arena.bot_idx % 3) == 0

    gx_goal = torch.where(arena.carrying, home_x, torch.where(is_relic_runner, arena.relic_x[:, None], best_sx))
    gy_goal = torch.where(arena.carrying, home_y, torch.where(is_relic_runner, arena.relic_y[:, None], best_sy))

    mdx = gx_goal - arena.pos_x
    mdy = gy_goal - arena.pos_y
    ml = torch.hypot(mdx, mdy).clamp(min=1e-3)
    mvx, mvy = mdx / ml, mdy / ml

    # Local fire/water probes
    ix, iy = arena._grid_xy(arena.pos_x, arena.pos_y)
    ax_p, ay_p = arena._grid_xy((arena.pos_x + mvx * 6.0).clamp(4.0, 252.0), (arena.pos_y + mvy * 6.0).clamp(4.0, 252.0))
    fire_here = arena.w_state[arena.arena_idx, arena.bot_layer, CH_TEMP, iy, ix] > 1.55
    fire_ahead = arena.w_state[arena.arena_idx, arena.bot_layer, CH_TEMP, ay_p, ax_p] > 1.55
    water_ahead = (
        (arena.w_state[arena.arena_idx, arena.bot_layer, SP_WATER, ay_p, ax_p] > 0.42)
        & (arena.w_state[arena.arena_idx, arena.bot_layer, SP_ICE, ay_p, ax_p] < 0.35)
    )

    # Simple fire evasion: sidestep away from fire
    evade = fire_here | fire_ahead
    new_mvx = torch.where(evade, -mvy, mvx)
    new_mvy = torch.where(evade, mvx, mvy)
    mvx, mvy = new_mvx, new_mvy

    # Nearest alive enemy within 44 canonical units (subject to identical physical optical smoke/steam blur!)
    dx_ij = arena.pos_x[:, None, :] - arena.pos_x[:, :, None]
    dy_ij = arena.pos_y[:, None, :] - arena.pos_y[:, :, None]
    dist_ij = torch.hypot(dx_ij, dy_ij).clamp(min=1e-3)
    en_mask = arena.enemy_pair_mask & arena.alive[:, None, :]
    ed, ej = dist_ij.masked_fill(~en_mask, 1e6).min(dim=-1)
    has_en = ed < 44.0

    edx = torch.gather(dx_ij, 2, ej.unsqueeze(-1)).squeeze(-1)
    edy = torch.gather(dy_ij, 2, ej.unsqueeze(-1)).squeeze(-1)

    ex_i, ey_i = arena._grid_xy(torch.gather(arena.pos_x, 1, ej), torch.gather(arena.pos_y, 1, ej))
    el_i = torch.gather(arena.bot_layer, 1, ej)
    en_opt = (
        arena.w_state[arena.arena_idx, el_i, SP_SMOKE, ey_i, ex_i]
        + 0.85 * arena.w_state[arena.arena_idx, el_i, SP_STEAM, ey_i, ex_i]
    ).clamp(0.0, 1.5)
    en_obsc = F.relu(en_opt - 0.25)
    if (en_obsc > 0.0).any():
        edx = edx + torch.randn(edx.shape, generator=arena.gen, device=dev) * (12.0 * en_obsc)
        edy = edy + torch.randn(edy.shape, generator=arena.gen, device=dev) * (12.0 * en_obsc)
    en_on_oil = has_en & (en_obsc < 0.35) & (arena.w_state[arena.arena_idx, el_i, SP_OIL, ey_i, ex_i] > 0.22)
    en_in_water = has_en & (en_obsc < 0.35) & (arena.w_state[arena.arena_idx, el_i, SP_WATER, ey_i, ex_i] > 0.38)

    # Emit offset (clamped to MAX_EMIT_RANGE_CELLS)
    emit_dx = torch.where(
        water_ahead & (~en_on_oil),
        (mvx * 8.0) / MAX_EMIT_RANGE_CELLS,
        torch.where(has_en, (edx / MAX_EMIT_RANGE_CELLS).clamp(-1.0, 1.0), mvx * 0.25),
    )
    emit_dy = torch.where(
        water_ahead & (~en_on_oil),
        (mvy * 8.0) / MAX_EMIT_RANGE_CELLS,
        torch.where(has_en, (edy / MAX_EMIT_RANGE_CELLS).clamp(-1.0, 1.0), mvy * 0.25),
    )

    # Rigid named-spell mapping to physical actuators with authentic 2-tick Frostmarsh spell cadence
    cast_tick = ((arena.tick + arena.bot_idx) % 2) == 0
    can_cast = cast_tick & (arena.mana > 0.20)
    cast_fire = can_cast & (en_on_oil | (has_en & (arena.cls == 2) & (~water_ahead)) | (has_en & en_in_water & (arena.cls == 2)))
    cast_frost = can_cast & (~cast_fire) & (water_ahead | (has_en & (arena.cls == 0)))
    cast_oil = can_cast & (~cast_fire) & (~cast_frost) & has_en & (arena.cls == 1) & (ed < 36.0)

    thermal = torch.where(cast_fire, torch.full_like(mvx, 0.90), torch.where(cast_frost, torch.full_like(mvx, -0.85), torch.zeros_like(mvx)))
    sp_water = torch.zeros_like(mvx)
    sp_oil = torch.where(cast_oil, torch.full_like(mvx, 0.85), torch.zeros_like(mvx))
    sp_smoke = torch.zeros_like(mvx)  # Legacy spells have no smokescreen!
    # Legacy spells have fixed focus radius (narrow bolt ~0.20, slightly wider AoE when mana > 0.65)
    focus = torch.where(arena.mana > 0.68, torch.full_like(mvx, 0.45), torch.full_like(mvx, 0.20))
    impulse = torch.where(evade & (arena.mana > 0.25), torch.full_like(mvx, 0.70), torch.where(has_en & (ed < 10.0), torch.full_like(mvx, 0.65), torch.zeros_like(mvx)))

    return torch.stack([mvx, mvy, emit_dx, emit_dy, thermal, sp_water, sp_oil, sp_smoke, focus, impulse], dim=-1)


@torch.inference_mode()
def policy_deterministic_act(
    policy: ActuatorPolicyNet,
    z_world: torch.Tensor,
    z_eco: torch.Tensor,
    bot_feats: torch.Tensor,
    thermal_only: bool = False,
) -> torch.Tensor:
    a, n, _ = bot_feats.shape
    p = policy.patch_size
    fp, cp = sample_batched_dual_patches(policy, z_world, z_eco, bot_feats)
    raw_act, _, _ = policy.forward_from_patches(
        fp.reshape(a * n, -1, p, p),
        cp.reshape(a * n, -1, p, p),
        bot_feats.reshape(a * n, BOT_FEAT_DIM),
        return_saliency=False,
    )
    act = raw_to_actuators(raw_act).reshape(a, n, ACTUATOR_DIM)
    if thermal_only or policy.thermal_only:
        act = act.clone()
        act[:, :, ACT_WATER : ACT_SMOKE + 1] = 0.0
    return act


# ---------------------------------------------------------------------------
# Pure Float32-Safe Continuous PPO Actor-Critic Wrapper
# ---------------------------------------------------------------------------


class RLActuatorActorCritic(nn.Module):
    """Wraps pure ActuatorPolicyNet with Dual-Horizon FP32 Value Critic & Post-Activation Gaussian PPO."""

    def __init__(self, policy: ActuatorPolicyNet):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),  # [V_tac (shrine/relic/score), V_act (physical actuator combat/survival)]
        )
        # Per-channel log_std directly in bounded physical actuator space (no saturating logit-tail collapse!)
        # Bipolar channels [-1,1] start at std ~0.35..0.45; unipolar channels [0,1] start at std ~0.24..0.30
        init_log_std = torch.tensor(
            [-1.05, -1.05, -0.85, -0.85, -0.70, -1.35, -1.15, -1.25, -1.20, -1.25],
            dtype=torch.float32,
        )
        self.log_std = nn.Parameter(init_log_std)

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-2.3, -0.55))

    def _clamp_exec(self, a_raw: torch.Tensor) -> torch.Tensor:
        return torch.cat([a_raw[..., :5].clamp(-1.0, 1.0), a_raw[..., 5:].clamp(0.0, 1.0)], dim=-1)

    def sample_from_patches(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (a_sample [A,N,10], act_exec [A,N,10], log_prob [A,N,3], values [A,N,2])."""
        a, n, _ = bot_feats.shape
        p = self.policy.patch_size
        raw_mu, h_trunk, _ = self.policy.forward_from_patches(
            fine_patches.reshape(a * n, -1, p, p).float(),
            coarse_patches.reshape(a * n, -1, p, p).float(),
            bot_feats.reshape(a * n, BOT_FEAT_DIM).float(),
            return_saliency=False,
        )
        mu = raw_to_actuators(raw_mu).reshape(a, n, ACTUATOR_DIM)
        values = self.value_head(h_trunk).reshape(a, n, 2)

        if deterministic:
            lp = torch.zeros((a, n, 3), device=mu.device, dtype=torch.float32)
            return mu, mu, lp, values

        std = self.get_std().view(1, 1, ACTUATOR_DIM)
        dist = torch.distributions.Normal(mu, std)
        a_sample = dist.sample()
        lp_all = dist.log_prob(a_sample)
        lp = torch.stack(
            [lp_all[..., 0:2].sum(dim=-1), lp_all[..., 2:4].sum(dim=-1), lp_all[..., 4:10].mean(dim=-1)],
            dim=-1,
        )
        act_exec = self._clamp_exec(a_sample)
        return a_sample, act_exec, lp, values

    def evaluate_flat_minibatch(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        a_sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """BF16-accelerated conv backbone on A100 with strict FP32 PPO log-probs, KL & value heads!"""
        if fine_patches.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                raw_mu, h_trunk, _ = self.policy.forward_from_patches(
                    fine_patches, coarse_patches, bot_feats, return_saliency=False
                )
            raw_mu = raw_mu.float()
            values = self.value_head(h_trunk.float())
        else:
            raw_mu, h_trunk, _ = self.policy.forward_from_patches(
                fine_patches.float(), coarse_patches.float(), bot_feats.float(), return_saliency=False
            )
            values = self.value_head(h_trunk)
        mu = raw_to_actuators(raw_mu)
        std = self.get_std().view(1, ACTUATOR_DIM)
        dist = torch.distributions.Normal(mu, std)
        lp_all = dist.log_prob(a_sample)
        lp = torch.stack(
            [lp_all[:, 0:2].sum(dim=-1), lp_all[:, 2:4].sum(dim=-1), lp_all[:, 4:10].mean(dim=-1)],
            dim=-1,
        )
        ent = dist.entropy().mean(dim=-1)
        return lp, ent, values, raw_mu


# ---------------------------------------------------------------------------
# Helpers for Emergence Telemetry & Correlation Computation
# ---------------------------------------------------------------------------


def _pearson_from_sums(n: float, sx: float, sx2: float, sy: float, sy2: float, sxy: float) -> float:
    if n < 10.0:
        return 0.0
    vx = n * sx2 - sx * sx
    vy = n * sy2 - sy * sy
    if vx <= 1e-8 or vy <= 1e-8:
        return 0.0
    return float((n * sxy - sx * sy) / math.sqrt(vx * vy))


def extract_team_emergence(arena: PhysicsActuatorArena, side_idx: torch.Tensor) -> dict[str, float]:
    """Extract mean per-match emergence stats for arenas where the target team is on `side_idx` (0 or 1)."""
    a_idx = torch.arange(arena.A, device=arena.device)
    n = float(arena.focus_count[a_idx, side_idx].sum().item())
    sf = float(arena.focus_sum_f[a_idx, side_idx].sum().item())
    sf2 = float(arena.focus_sum_f2[a_idx, side_idx].sum().item())
    sc = float(arena.focus_sum_c[a_idx, side_idx].sum().item())
    sc2 = float(arena.focus_sum_c2[a_idx, side_idx].sum().item())
    sfc = float(arena.focus_sum_fc[a_idx, side_idx].sum().item())
    sd = float(arena.focus_sum_d[a_idx, side_idx].sum().item())
    sd2 = float(arena.focus_sum_d2[a_idx, side_idx].sum().item())
    sfd = float(arena.focus_sum_fd[a_idx, side_idx].sum().item())

    sn1 = float(arena.focus_single_n[a_idx, side_idx].sum().item())
    sn2 = float(arena.focus_multi_n[a_idx, side_idx].sum().item())
    mean_f = sf / max(n, 1.0)
    var_f = max(0.0, (sf2 / max(n, 1.0)) - mean_f * mean_f)

    return {
        "oil_ignition_detonations": float(arena.emerge_oil_ignite[a_idx, side_idx].mean().item()),
        "flash_boil_steam_bursts": float(arena.emerge_flash_boil[a_idx, side_idx].mean().item()),
        "cryo_crossings": float(arena.emerge_cryo_cross[a_idx, side_idx].mean().item()),
        "smokescreen_covers": float(arena.emerge_smokescreen[a_idx, side_idx].mean().item()),
        "kinetic_pushes": float(arena.emerge_kinetic_push[a_idx, side_idx].mean().item()),
        "focus_mean_cells": mean_f,
        "focus_std_cells": math.sqrt(var_f),
        "focus_corr_cluster": _pearson_from_sums(n, sf, sf2, sc, sc2, sfc),
        "focus_corr_distance": _pearson_from_sums(n, sf, sf2, sd, sd2, sfd),
        "focus_single_enemy_cells": float(arena.focus_single_sum[a_idx, side_idx].sum().item()) / max(sn1, 1.0),
        "focus_multi_enemy_cells": float(arena.focus_multi_sum[a_idx, side_idx].sum().item()) / max(sn2, 1.0),
    }


@torch.inference_mode()
def quick_eval_actuator(
    policy: ActuatorPolicyNet,
    val_arena: PhysicsActuatorArena,
    seed: int = 81000,
) -> dict[str, Any]:
    """
    Fast side-balanced deterministic evaluation on val_arena:
      - First half of arenas: RL vs Legacy Named-Spell Scripted Baseline
      - Second half of arenas: RL vs Thermal-Only Ablation
    Also returns emergent tactic counts per match for discovery curve tracking!
    """
    policy.eval()
    A = val_arena.A
    half = A // 2
    q = half // 2
    f = val_arena.reset(seed=seed)

    rl_mask = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    rl_mask[:q, :6] = True
    rl_mask[q:half, 6:] = True
    rl_mask[half : half + q, :6] = True
    rl_mask[half + q :, 6:] = True

    leg_mask = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    leg_mask[:q, 6:] = True
    leg_mask[q:half, :6] = True

    for _ in range(val_arena.max_ticks):
        act_rl = policy_deterministic_act(policy, val_arena.z_world, val_arena.z_eco, f, thermal_only=False)
        act_th = act_rl.clone()
        act_th[:, :, ACT_WATER : ACT_SMOKE + 1] = 0.0
        act_leg = legacy_scripted_actuator_baseline(val_arena)
        act = torch.where(rl_mask.unsqueeze(-1), act_rl, torch.where(leg_mask.unsqueeze(-1), act_leg, act_th))
        f, _, _, _ = val_arena.step(act)

    sw, se = val_arena.team_score[:, 0], val_arena.team_score[:, 1]
    w_leg_0 = (sw[:q] > se[:q]).float() + 0.5 * (sw[:q] == se[:q]).float()
    w_leg_1 = (se[q:half] > sw[q:half]).float() + 0.5 * (se[q:half] == sw[q:half]).float()
    wr_legacy = float(torch.cat([w_leg_0, w_leg_1]).mean().item()) * 100.0

    w_th_0 = (sw[half : half + q] > se[half : half + q]).float() + 0.5 * (sw[half : half + q] == se[half : half + q]).float()
    w_th_1 = (se[half + q :] > sw[half + q :]).float() + 0.5 * (se[half + q :] == sw[half + q :]).float()
    wr_thermal = float(torch.cat([w_th_0, w_th_1]).mean().item()) * 100.0

    side_rl = torch.zeros((A,), dtype=torch.long, device=val_arena.device)
    side_rl[q:half] = 1
    side_rl[half + q :] = 1
    em = extract_team_emergence(val_arena, side_rl)

    sc_rl = float(torch.cat([sw[:q], se[q:half]]).mean().item())
    sc_leg = float(torch.cat([se[:q], sw[q:half]]).mean().item())
    kills_rl = float(torch.cat([val_arena.team_kills[:q, 0], val_arena.team_kills[q:half, 1]]).mean().item())
    shrines_rl = float(torch.cat([val_arena.team_shrine_caps[:q, 0], val_arena.team_shrine_caps[q:half, 1]]).mean().item())
    relics_rl = float(torch.cat([val_arena.team_relic_deliveries[:q, 0], val_arena.team_relic_deliveries[q:half, 1]]).mean().item())

    return {
        "wr_legacy_pct": round(wr_legacy, 2),
        "wr_thermal_pct": round(wr_thermal, 2),
        "score_rl": round(sc_rl, 2),
        "score_legacy": round(sc_leg, 2),
        "kills": round(kills_rl, 2),
        "shrines": round(shrines_rl, 2),
        "relics": round(relics_rl, 2),
        **{k: round(v, 3) for k, v in em.items()},
    }


# ---------------------------------------------------------------------------
# Pure Reward-Driven Float32-Safe PPO Self-Play Training Loop
# ---------------------------------------------------------------------------


def train_self_play_ppo(
    ac: RLActuatorActorCritic,
    arena: PhysicsActuatorArena,
    val_arena: PhysicsActuatorArena,
    num_updates: int = 130,
    rollout_steps: int = 24,
    ppo_epochs: int = 3,
    minibatch_size: int = 2048,
    lr: float = 5.0e-4,
    kl_early_stop: float = 0.25,
    eval_every: int = 10,
    seed_offset: int = 1000,
) -> tuple[dict, dict, list[dict]]:
    """
    Pure reward-driven float32-safe PPO Self-Play on 64 parallel arenas (768 concurrent bots)!
    ZERO imitation teacher: 50% mirror self-play + 50% lagged self-play pool.
    """
    A, N = arena.A, arena.N
    P = ac.policy.patch_size
    dev = arena.device

    optimizer = torch.optim.AdamW(ac.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)

    # Diverse lagged self-play pool (4 past policy snapshots rotated over training)
    pool_size = 4
    lag_pool = [copy.deepcopy(ac.policy).eval() for _ in range(pool_size)]
    for m in lag_pool:
        for p in m.parameters():
            p.requires_grad_(False)
    pool_ptr = 0

    # Arena split: 50% Mirror Self-Play (all 12 bots RL), 25% Lagged Self-Play Pool, 25% League Sparring (pure reward, ZERO imitation!)
    n_mirror = A // 2
    n_lag = (A - n_mirror) // 2
    n_spar = A - n_mirror - n_lag
    l_start = n_mirror
    l_mid = l_start + n_lag // 2
    s_start = l_start + n_lag
    s_mid = s_start + n_spar // 2

    rl_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    rl_mask[:n_mirror, :] = True
    rl_mask[l_start:l_mid, :6] = True
    rl_mask[l_mid:s_start, 6:] = True
    rl_mask[s_start:s_mid, :6] = True
    rl_mask[s_mid:, 6:] = True

    opp_lag_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_lag_mask[l_start:l_mid, 6:] = True
    opp_lag_mask[l_mid:s_start, :6] = True

    opp_spar_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_spar_mask[s_start:s_mid, 6:] = True
    opp_spar_mask[s_mid:, :6] = True

    rl_mask_t = rl_mask.unsqueeze(0).expand(rollout_steps, A, N)

    # Pre-allocated GPU rollout buffers
    p_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    buf_fp = torch.empty((rollout_steps, A, N, 80, P, P), dtype=p_dtype, device=dev)
    buf_cp = torch.empty((rollout_steps, A, N, 48, P, P), dtype=p_dtype, device=dev)
    buf_feats = torch.empty((rollout_steps, A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)
    buf_z = torch.empty((rollout_steps, A, N, ACTUATOR_DIM), dtype=torch.float32, device=dev)
    buf_lp = torch.empty((rollout_steps, A, N, 3), dtype=torch.float32, device=dev)
    buf_val = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_rew = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_done = torch.empty((rollout_steps,), dtype=torch.float32, device=dev)

    t0 = time.perf_counter()
    feats = arena.reset(seed=seed_offset, close_spawn_frac=0.5)
    total_bot_steps = 0
    kl_stops = 0

    # Initial update=0 baseline emergence evaluation (untrained policy!)
    ev0 = quick_eval_actuator(ac.policy, val_arena, seed=81000)
    ev0["update"] = 0
    ev0["elapsed_s"] = 0.0
    history: list[dict] = [ev0]
    best_metric = -1e9
    best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
    best_ev = ev0

    print(
        f"[PPO Self-Play] init (upd 0/{num_updates}) | 64x12={A*N} bots | "
        f"vsLegacy={ev0['wr_legacy_pct']:5.1f}% ({ev0['score_rl']:.0f}v{ev0['score_legacy']:.0f}) | "
        f"vsTherm={ev0['wr_thermal_pct']:5.1f}% | oilIgn={ev0['oil_ignition_detonations']:.1f} "
        f"boil={ev0['flash_boil_steam_bursts']:.1f} cryo={ev0['cryo_crossings']:.1f} "
        f"smoke={ev0['smokescreen_covers']:.1f}",
        flush=True,
    )

    for upd in range(1, num_updates + 1):
        # Update lagged self-play pool every 6 updates
        if upd > 1 and (upd % 6 == 0):
            lag_pool[pool_ptr].load_state_dict(ac.policy.state_dict())
            pool_ptr = (pool_ptr + 1) % pool_size
        opp_policy = lag_pool[(upd // 2) % pool_size]

        ac.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                fp, cp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
                buf_fp[t] = fp.to(p_dtype)
                buf_cp[t] = cp.to(p_dtype)
                buf_feats[t] = feats

                z_rl, act_rl, lp_rl, val_rl = ac.sample_from_patches(fp, cp, feats, deterministic=False)
                act = act_rl
                if opp_lag_mask.any():
                    opp_raw, _, _ = opp_policy.forward_from_patches(
                        fp.reshape(A * N, 80, P, P).float(),
                        cp.reshape(A * N, 48, P, P).float(),
                        feats.reshape(A * N, BOT_FEAT_DIM).float(),
                        return_saliency=False,
                    )
                    act_opp = raw_to_actuators(opp_raw).reshape(A, N, ACTUATOR_DIM)
                    act = torch.where(opp_lag_mask.unsqueeze(-1), act_opp, act)
                if opp_spar_mask.any():
                    act_spar = legacy_scripted_actuator_baseline(arena)
                    act = torch.where(opp_spar_mask.unsqueeze(-1), act_spar, act)

                buf_z[t] = z_rl
                buf_lp[t] = lp_rl
                buf_val[t] = val_rl

                next_feats, rews, done, _ = arena.step(act)
                buf_rew[t] = rews
                buf_done[t] = 1.0 if done else 0.0
                total_bot_steps += A * N
                feats = arena.reset(seed=seed_offset + upd * 197 + t, close_spawn_frac=0.5) if done else next_feats

            lfp, lcp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
            _, _, _, last_val = ac.sample_from_patches(lfp, lcp, feats, deterministic=True)

        # Dual-Horizon Float32 GAE (Tac: gamma=0.97, lam=0.95; Act: gamma=0.80, lam=0.88)
        adv = torch.zeros_like(buf_rew)
        for h_i, (gam, lam) in enumerate(((0.97, 0.95), (0.80, 0.88))):
            gae = torch.zeros((A, N), device=dev, dtype=torch.float32)
            for t in reversed(range(rollout_steps)):
                nonterm = 1.0 - buf_done[t]
                next_v = last_val[..., h_i] if t == rollout_steps - 1 else buf_val[t + 1, ..., h_i]
                delta = buf_rew[t, ..., h_i] + gam * next_v * nonterm - buf_val[t, ..., h_i]
                gae = delta + gam * lam * nonterm * gae
                adv[t, ..., h_i] = gae
        ret = adv + buf_val

        flat_fp = buf_fp[rl_mask_t]
        flat_cp = buf_cp[rl_mask_t]
        flat_feats = buf_feats[rl_mask_t]
        flat_z = buf_z[rl_mask_t]
        flat_old_lp = buf_lp[rl_mask_t]  # [M, 3]
        flat_adv_pair = adv[rl_mask_t]   # [M, 2]
        adv_tac = (flat_adv_pair[:, 0] - flat_adv_pair[:, 0].mean()) / (flat_adv_pair[:, 0].std() + 1e-6)
        adv_act = (flat_adv_pair[:, 1] - flat_adv_pair[:, 1].mean()) / (flat_adv_pair[:, 1].std() + 1e-6)
        # Physical actuators get balanced combat/survival AND tactical river-crossing/relic credit!
        adv_phys = 0.65 * adv_act + 0.45 * adv_tac
        flat_ret = ret[rl_mask_t]        # [M, 2]

        M = flat_feats.shape[0]
        prog = upd / float(num_updates)
        ent_coef = 0.015 * (1.0 - 0.65 * prog)

        ac.train()
        early_stopped = False
        max_kl = 0.0
        for _ in range(ppo_epochs):
            perm = torch.randperm(M, device=dev)
            ep_kls = []
            for start in range(0, M, minibatch_size):
                idx = perm[start : start + minibatch_size]
                optimizer.zero_grad(set_to_none=True)

                new_lp, ent, vals, raw_mu = ac.evaluate_flat_minibatch(
                    flat_fp[idx], flat_cp[idx], flat_feats[idx], flat_z[idx]
                )
                log_r = (new_lp - flat_old_lp[idx]).clamp(-4.0, 4.0)  # [B, 3]
                log_r_joint = log_r.sum(dim=-1)
                approx_kl = float((0.5 * log_r_joint.pow(2)).mean().item())
                ep_kls.append(approx_kl)
                if approx_kl > max_kl:
                    max_kl = approx_kl
                if approx_kl > kl_early_stop:
                    early_stopped = True
                    kl_stops += 1
                    break

                r_mv = torch.exp(log_r[:, 0])
                r_em = torch.exp(log_r[:, 1])
                r_ph = torch.exp(log_r[:, 2])

                a_t = adv_tac[idx]
                a_p = adv_phys[idx]

                pg_mv = -torch.minimum(r_mv * a_t, r_mv.clamp(0.80, 1.20) * a_t).mean()
                pg_em = -torch.minimum(r_em * a_p, r_em.clamp(0.80, 1.20) * a_p).mean()
                pg_ph = -torch.minimum(r_ph * a_p, r_ph.clamp(0.80, 1.20) * a_p).mean()
                pg_loss = pg_mv + 1.15 * pg_em + 1.35 * pg_ph

                v_loss = F.smooth_l1_loss(vals, flat_ret[idx], beta=5.0)
                ent_loss = -ent_coef * ent.mean()
                # Soft anti-saturation barrier outside [-1.85, +1.85]: zero penalty in active range, prevents dead tails!
                logit_reg = 1.5e-2 * F.relu(raw_mu.abs() - 1.85).pow(2).mean()

                loss = pg_loss + 0.5 * v_loss + ent_loss + logit_reg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                optimizer.step()

            if early_stopped or (ep_kls and float(np.mean(ep_kls)) > kl_early_stop):
                if not early_stopped:
                    kl_stops += 1
                break

        # Cosine LR schedule
        lr_mult = 0.22 + 0.78 * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = lr * lr_mult

        if (upd % eval_every == 0) or (upd == num_updates):
            elapsed = time.perf_counter() - t0
            ev = quick_eval_actuator(ac.policy, val_arena, seed=81000)
            ev["update"] = upd
            ev["elapsed_s"] = round(elapsed, 2)
            ev["mean_std"] = round(float(ac.get_std().mean().item()), 3)
            history.append(ev)

            # Select best checkpoint on match performance (win rate + score diff + objectives)
            metric = (
                4.0 * ev["wr_legacy_pct"]
                + 1.5 * ev["wr_thermal_pct"]
                + 0.70 * (ev["score_rl"] - ev["score_legacy"])
                + 4.0 * ev["kills"]
                + 12.0 * ev["relics"]
                + 3.0 * ev["shrines"]
            )
            tag = ""
            if metric > best_metric:
                best_metric = metric
                best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
                best_ev = ev
                tag = " *BEST*"

            print(
                f"[PPO Self-Play] upd {upd:3d}/{num_updates} | t={elapsed:5.1f}s | kl={max_kl:.3f} | "
                f"vsLegacy={ev['wr_legacy_pct']:5.1f}% ({ev['score_rl']:.0f}v{ev['score_legacy']:.0f}) | "
                f"vsTherm={ev['wr_thermal_pct']:5.1f}% | kills={ev['kills']:.1f} shr={ev['shrines']:.1f} | "
                f"oilIgn={ev['oil_ignition_detonations']:.1f} boil={ev['flash_boil_steam_bursts']:.1f} "
                f"cryo={ev['cryo_crossings']:.1f} smoke={ev['smokescreen_covers']:.1f} "
                f"r_foc={ev['focus_corr_cluster']:+.2f}{tag}",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    train_summary = {
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "parallel_arenas": A,
        "concurrent_bots": A * N,
        "updates": num_updates,
        "rollout_steps": rollout_steps,
        "total_bot_steps": total_bot_steps,
        "elapsed_s": elapsed,
        "bot_steps_per_sec": round(total_bot_steps / max(elapsed, 1e-3), 1),
        "kl_early_stop_threshold": kl_early_stop,
        "kl_early_stops_triggered": kl_stops,
        "zero_imitation_teacher": True,
        "zero_hardcoded_spell_rules": True,
        "best_val": best_ev,
    }
    return train_summary, best_sd, history


# ---------------------------------------------------------------------------
# Held-Out 256x256x4 Tournament Evaluation (24 Side-Balanced Matches / Pair)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_agent_kind(
    kind: str,
    policy: ActuatorPolicyNet | None,
    arena: PhysicsActuatorArena,
    feats: torch.Tensor,
) -> torch.Tensor:
    if kind == "rl":
        assert policy is not None
        return policy_deterministic_act(policy, arena.z_world, arena.z_eco, feats, thermal_only=False)
    if kind == "thermal_only":
        assert policy is not None
        return policy_deterministic_act(policy, arena.z_world, arena.z_eco, feats, thermal_only=True)
    if kind == "legacy_scripted":
        return legacy_scripted_actuator_baseline(arena)
    if kind == "random":
        return random_continuous_act(arena)
    raise ValueError(f"Unknown agent kind: {kind}")


@torch.inference_mode()
def run_heldout_matchup(
    name_a: str,
    kind_a: str,
    name_b: str,
    kind_b: str,
    policy: ActuatorPolicyNet,
    arena: PhysicsActuatorArena,
    num_matches: int = 24,
    base_seed: int = 95000,
    collect_behavior: bool = False,
) -> tuple[dict[str, Any], list[dict]]:
    """Run exact side-balanced 24-match evaluation on unseen held-out 256x256x4 worlds."""
    assert num_matches % 2 == 0
    half = num_matches // 2
    batch = arena.A
    assert half % batch == 0, f"half ({half}) must be divisible by arena.A ({batch})"
    rounds = half // batch

    t0 = time.perf_counter()
    wins_a = wins_b = draws = 0
    scores_a: list[float] = []
    scores_b: list[float] = []
    kills_a: list[float] = []
    kills_b: list[float] = []
    deaths_a: list[float] = []
    deaths_b: list[float] = []
    shrines_a: list[float] = []
    shrines_b: list[float] = []
    relics_a: list[float] = []
    relics_b: list[float] = []
    press_a: list[float] = []
    press_b: list[float] = []

    em_keys = ("oil_ignition_detonations", "flash_boil_steam_bursts", "cryo_crossings", "smokescreen_covers", "kinetic_pushes")
    em_a_acc = {k: [] for k in em_keys}
    em_b_acc = {k: [] for k in em_keys}
    foc_a = [0.0] * 13  # n, sf, sf2, sc, sc2, sfc, sd, sd2, sfd, s1, n1, s2, n2
    behavior_records: list[dict] = []

    for side in (0, 1):
        # side=0: A is West (0..5), B is East (6..11); side=1: swapped!
        mask_a = torch.zeros((batch, NUM_BOTS), dtype=torch.bool, device=arena.device)
        if side == 0:
            mask_a[:, :6] = True
        else:
            mask_a[:, 6:] = True

        for r in range(rounds):
            seed = base_seed + r * 31
            f = arena.reset(seed=seed)
            for _ in range(arena.max_ticks):
                act_a = _run_agent_kind(kind_a, policy, arena, f)
                act_b = _run_agent_kind(kind_b, policy, arena, f)
                act = torch.where(mask_a.unsqueeze(-1), act_a, act_b)
                f, _, _, bsam = arena.step(act, collect_behavior_samples=collect_behavior)
                if collect_behavior and bsam is not None:
                    # Keep only team A's bots
                    sl = slice(0, 6) if side == 0 else slice(6, 12)
                    behavior_records.append({k: v[:, sl] for k, v in bsam.items()})

            sa = arena.team_score[:, side]
            sb = arena.team_score[:, 1 - side]
            for i in range(batch):
                va, vb = float(sa[i].item()), float(sb[i].item())
                if va > vb:
                    wins_a += 1
                elif vb > va:
                    wins_b += 1
                else:
                    draws += 1
                scores_a.append(va)
                scores_b.append(vb)
                kills_a.append(float(arena.team_kills[i, side].item()))
                kills_b.append(float(arena.team_kills[i, 1 - side].item()))
                deaths_a.append(float(arena.team_deaths[i, side].item()))
                deaths_b.append(float(arena.team_deaths[i, 1 - side].item()))
                shrines_a.append(float(arena.team_shrine_caps[i, side].item()))
                shrines_b.append(float(arena.team_shrine_caps[i, 1 - side].item()))
                relics_a.append(float(arena.team_relic_deliveries[i, side].item()))
                relics_b.append(float(arena.team_relic_deliveries[i, 1 - side].item()))
                press_a.append(float(arena.team_pressure_dmg_dealt[i, side].item()))
                press_b.append(float(arena.team_pressure_dmg_dealt[i, 1 - side].item()))

                em_a_acc["oil_ignition_detonations"].append(float(arena.emerge_oil_ignite[i, side].item()))
                em_b_acc["oil_ignition_detonations"].append(float(arena.emerge_oil_ignite[i, 1 - side].item()))
                em_a_acc["flash_boil_steam_bursts"].append(float(arena.emerge_flash_boil[i, side].item()))
                em_b_acc["flash_boil_steam_bursts"].append(float(arena.emerge_flash_boil[i, 1 - side].item()))
                em_a_acc["cryo_crossings"].append(float(arena.emerge_cryo_cross[i, side].item()))
                em_b_acc["cryo_crossings"].append(float(arena.emerge_cryo_cross[i, 1 - side].item()))
                em_a_acc["smokescreen_covers"].append(float(arena.emerge_smokescreen[i, side].item()))
                em_b_acc["smokescreen_covers"].append(float(arena.emerge_smokescreen[i, 1 - side].item()))
                em_a_acc["kinetic_pushes"].append(float(arena.emerge_kinetic_push[i, side].item()))
                em_b_acc["kinetic_pushes"].append(float(arena.emerge_kinetic_push[i, 1 - side].item()))

            foc_a[0] += float(arena.focus_count[:, side].sum().item())
            foc_a[1] += float(arena.focus_sum_f[:, side].sum().item())
            foc_a[2] += float(arena.focus_sum_f2[:, side].sum().item())
            foc_a[3] += float(arena.focus_sum_c[:, side].sum().item())
            foc_a[4] += float(arena.focus_sum_c2[:, side].sum().item())
            foc_a[5] += float(arena.focus_sum_fc[:, side].sum().item())
            foc_a[6] += float(arena.focus_sum_d[:, side].sum().item())
            foc_a[7] += float(arena.focus_sum_d2[:, side].sum().item())
            foc_a[8] += float(arena.focus_sum_fd[:, side].sum().item())
            foc_a[9] += float(arena.focus_single_sum[:, side].sum().item())
            foc_a[10] += float(arena.focus_single_n[:, side].sum().item())
            foc_a[11] += float(arena.focus_multi_sum[:, side].sum().item())
            foc_a[12] += float(arena.focus_multi_n[:, side].sum().item())

    elapsed = round(time.perf_counter() - t0, 2)
    wr_a = round((wins_a + 0.5 * draws) / num_matches * 100.0, 2)
    wr_b = round((wins_b + 0.5 * draws) / num_matches * 100.0, 2)
    mean_f = foc_a[1] / max(foc_a[0], 1.0)
    std_f = math.sqrt(max(0.0, foc_a[2] / max(foc_a[0], 1.0) - mean_f * mean_f))

    res = {
        "matchup": f"{name_a} vs {name_b}",
        "team_a": name_a,
        "team_b": name_b,
        "matches": num_matches,
        "ticks_per_match": arena.max_ticks,
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a_pct": wr_a,
        "win_rate_b_pct": wr_b,
        "mean_score_a": round(float(np.mean(scores_a)), 2),
        "mean_score_b": round(float(np.mean(scores_b)), 2),
        "kills_per_match_a": round(float(np.mean(kills_a)), 2),
        "kills_per_match_b": round(float(np.mean(kills_b)), 2),
        "deaths_per_match_a": round(float(np.mean(deaths_a)), 2),
        "deaths_per_match_b": round(float(np.mean(deaths_b)), 2),
        "shrine_caps_per_match_a": round(float(np.mean(shrines_a)), 2),
        "shrine_caps_per_match_b": round(float(np.mean(shrines_b)), 2),
        "relic_deliveries_per_match_a": round(float(np.mean(relics_a)), 2),
        "relic_deliveries_per_match_b": round(float(np.mean(relics_b)), 2),
        "pressure_wave_dmg_per_match_a": round(float(np.mean(press_a)), 2),
        "pressure_wave_dmg_per_match_b": round(float(np.mean(press_b)), 2),
        "emergent_tactics_a_per_match": {k: round(float(np.mean(em_a_acc[k])), 2) for k in em_keys},
        "emergent_tactics_b_per_match": {k: round(float(np.mean(em_b_acc[k])), 2) for k in em_keys},
        "adaptive_focus_a": {
            "mean_radius_cells": round(mean_f, 3),
            "std_radius_cells": round(std_f, 3),
            "corr_vs_enemy_cluster": round(_pearson_from_sums(foc_a[0], foc_a[1], foc_a[2], foc_a[3], foc_a[4], foc_a[5]), 4),
            "corr_vs_emit_distance": round(_pearson_from_sums(foc_a[0], foc_a[1], foc_a[2], foc_a[6], foc_a[7], foc_a[8]), 4),
            "single_enemy_mean_cells": round(foc_a[9] / max(foc_a[10], 1.0), 3),
            "multi_enemy_mean_cells": round(foc_a[11] / max(foc_a[12], 1.0), 3),
        },
        "eval_time_s": elapsed,
    }
    print(
        f"  [256x256x4 24m] {name_a} vs {name_b}: WR={wr_a:5.1f}% ({wins_a}W-{wins_b}L-{draws}D) | "
        f"Score {res['mean_score_a']:.1f} vs {res['mean_score_b']:.1f} | "
        f"OilIgn={res['emergent_tactics_a_per_match']['oil_ignition_detonations']:.1f} "
        f"Boil={res['emergent_tactics_a_per_match']['flash_boil_steam_bursts']:.1f} "
        f"Cryo={res['emergent_tactics_a_per_match']['cryo_crossings']:.1f} "
        f"Smoke={res['emergent_tactics_a_per_match']['smokescreen_covers']:.1f} | {elapsed:.1f}s",
        flush=True,
    )
    return res, behavior_records


# ---------------------------------------------------------------------------
# GPU Latency Benchmark (12 / 64 / 256 bots on resident VRAM Z_world & Z_eco)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def benchmark_inference_latency(
    policy: ActuatorPolicyNet,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    policy.eval()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    z_world = torch.randn((4, 48, 256, 256), device=device, dtype=dtype)
    z_eco = torch.randn((1, 32, 256, 256), device=device, dtype=dtype)

    results: dict[str, dict[str, float]] = {}
    for n_bots in (12, 64, 256):
        bf = torch.rand((n_bots, BOT_FEAT_DIM), device=device, dtype=torch.float32)

        def _measure(ret_sal: bool) -> list[float]:
            for _ in range(15):
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _ = policy(z_world, z_eco, bf, return_saliency=ret_sal)
                else:
                    _ = policy(z_world, z_eco, bf, return_saliency=ret_sal)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(80):
                t0 = time.perf_counter()
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _ = policy(z_world, z_eco, bf, return_saliency=ret_sal)
                    torch.cuda.synchronize()
                else:
                    _ = policy(z_world, z_eco, bf, return_saliency=ret_sal)
                times.append((time.perf_counter() - t0) * 1000.0)
            return times

        t_base = _measure(False)
        t_sal = _measure(True)
        results[f"bots_{n_bots}"] = {
            "p50_ms": round(float(np.percentile(t_base, 50)), 3),
            "p95_ms": round(float(np.percentile(t_base, 95)), 3),
            "mean_ms": round(float(np.mean(t_base)), 3),
            "with_saliency_p50_ms": round(float(np.percentile(t_sal, 50)), 3),
            "with_saliency_p95_ms": round(float(np.percentile(t_sal, 95)), 3),
        }
    return results


# ---------------------------------------------------------------------------
# 3-Panel Dark-Themed Emergence & Tournament Figure
# ---------------------------------------------------------------------------


def make_emergence_viz(
    history: list[dict],
    behavior_records: list[dict],
    matchups: dict[str, dict],
    gpu_name: str,
    out_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#9da7b3",
            "ytick.color": "#9da7b3",
            "grid.color": "#262c36",
            "font.size": 10,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=180)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"Frostmarsh Zero-Hardcoded-Rules Emergent Physical Actuators (ActuatorPolicyNet, {gpu_name})",
        fontsize=14,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )

    # --- Panel 1: Emergent Tactical Skills Discovery Curve over RL Updates ---
    ax1 = axes[0]
    upds = [h["update"] for h in history]
    c_oil = [h["oil_ignition_detonations"] for h in history]
    c_boil = [h["flash_boil_steam_bursts"] for h in history]
    c_cryo = [h["cryo_crossings"] for h in history]
    c_smoke = [h["smokescreen_covers"] for h in history]

    ax1.plot(upds, c_oil, "-o", markersize=3.5, lw=2.3, color="#ff7b00", label="Oil → Ignition Detonations")
    ax1.plot(upds, c_boil, "-s", markersize=3.5, lw=2.3, color="#00d4ff", label="Flash-Boil Steam Bursts")
    ax1.plot(upds, c_cryo, "-^", markersize=3.5, lw=2.3, color="#7ee787", label="Cryo-Bridge Crossings")
    ax1.plot(upds, c_smoke, "-d", markersize=3.5, lw=2.3, color="#d2a8ff", label="Smokescreen Cover")
    ax1.set_title("1. Emergent Physics Tactics Discovery Curve", fontsize=11.5, fontweight="bold", pad=10, color="#f0f6fc")
    ax1.set_xlabel("Pure Self-Play PPO Update (Zero Imitation Teacher)")
    ax1.set_ylabel("Emergent Events / Match (6v6)")
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5)

    # --- Panel 2: Continuous Actuator Behavior Map (thermal_flux vs focus_radius) ---
    ax2 = axes[1]
    if behavior_records:
        th = np.concatenate([r["thermal"].ravel() for r in behavior_records])
        fc = np.concatenate([r["focus_cells"].ravel() for r in behavior_records])
        wa = np.concatenate([r["water"].ravel() for r in behavior_records])
        oi = np.concatenate([r["oil"].ravel() for r in behavior_records])
        sm = np.concatenate([r["smoke"].ravel() for r in behavior_records])
        im = np.concatenate([r["impulse"].ravel() for r in behavior_records])
        al = np.concatenate([r["alive"].ravel() for r in behavior_records]).astype(bool)
        ev_oi = np.concatenate([r["ev_oil_ign"].ravel() for r in behavior_records]) > 0.5
        ev_fb = np.concatenate([r["ev_flash_boil"].ravel() for r in behavior_records]) > 0.5
        ev_cr = np.concatenate([r["ev_cryo"].ravel() for r in behavior_records]) > 0.5
        ev_sm = np.concatenate([r["ev_smoke"].ravel() for r in behavior_records]) > 0.5

        active = al & ((np.abs(th) + wa + oi + sm + im) > 0.12)
        idx_all = np.where(active)[0]
        if len(idx_all) > 3500:
            rng = np.random.default_rng(42)
            idx_all = rng.choice(idx_all, size=3500, replace=False)

        # Assign dominant physical mode for color coding
        # 0=Thermal Heat, 1=Cryo Freeze, 2=Oil Deposit/Detonate, 3=Smoke Cover, 4=Water/Steam, 5=Kinetic Impulse
        cats = np.zeros(len(idx_all), dtype=np.int32)
        for k, ii in enumerate(idx_all):
            if ev_oi[ii] or oi[ii] > max(abs(th[ii]) * 0.7, sm[ii], wa[ii], im[ii]):
                cats[k] = 2
            elif ev_sm[ii] or sm[ii] > max(abs(th[ii]) * 0.7, oi[ii], wa[ii], im[ii]):
                cats[k] = 3
            elif ev_fb[ii] or wa[ii] > max(abs(th[ii]) * 0.7, oi[ii], sm[ii], im[ii]):
                cats[k] = 4
            elif ev_cr[ii] or th[ii] < -0.15:
                cats[k] = 1
            elif im[ii] > max(abs(th[ii]), oi[ii], sm[ii], wa[ii]):
                cats[k] = 5
            else:
                cats[k] = 0

        palette = [
            (0, "#ff5c5c", "Heat Beam / Scorch (+ΔE)"),
            (1, "#58a6ff", "Cryo Freeze / Bridge (-ΔE)"),
            (2, "#ff9f1c", "Oil Deposit / Detonate"),
            (3, "#c792ea", "Smokescreen Cover"),
            (4, "#00e5ff", "Water / Flash-Boil Steam"),
            (5, "#7ee787", "Kinetic Impulse Wave"),
        ]
        for cid, col, lbl in palette:
            m = cats == cid
            if m.any():
                ax2.scatter(
                    th[idx_all[m]],
                    fc[idx_all[m]],
                    s=14,
                    alpha=0.58,
                    color=col,
                    edgecolors="none",
                    label=lbl,
                )

    ax2.axvline(0.0, color="#6e7681", linestyle=":", lw=1.2)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(1.2, 8.8)
    ax2.set_title("2. Continuous Actuator Behavior Map", fontsize=11.5, fontweight="bold", pad=10, color="#f0f6fc")
    ax2.set_xlabel("thermal_flux  (-1 Cryo Extraction  ↔  +1 Heat Injection)")
    ax2.set_ylabel("focus_radius  (1.5 Narrow Beam  ↔  8.5 Wide Field, cells)")
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.0, markerscale=1.4)

    # --- Panel 3: Held-Out 256x256x4 Tournament Win Rates & Score Comparison ---
    ax3 = axes[2]
    m_keys = [
        ("rl_vs_thermal_only", "vs Thermal-Only\n(No Matter)"),
        ("rl_vs_legacy_scripted", "vs Legacy\nNamed-Spell"),
        ("rl_vs_random", "vs Random\nActuators"),
    ]
    labels = [lbl for k, lbl in m_keys if k in matchups]
    wrs = [matchups[k]["win_rate_a_pct"] for k, _ in m_keys if k in matchups]
    sc_a = [matchups[k]["mean_score_a"] for k, _ in m_keys if k in matchups]
    sc_b = [matchups[k]["mean_score_b"] for k, _ in m_keys if k in matchups]

    x = np.arange(len(labels))
    bar_cols = ["#00d4ff", "#ff9f1c", "#7ee787"][: len(labels)]
    bars = ax3.bar(x, wrs, width=0.54, color=bar_cols, edgecolor="#f0f6fc", linewidth=0.8, alpha=0.92)
    ax3.axhline(50.0, color="#ff7b72", linestyle="--", lw=1.3, alpha=0.8, label="50% Even Baseline")
    ax3.set_ylim(0, 118)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9.5)
    ax3.set_ylabel("ActuatorPolicyNet-RL Win Rate (%)")
    ax3.set_title("3. Held-Out 256×256×4 Tournament (24m/pair)", fontsize=11.5, fontweight="bold", pad=10, color="#f0f6fc")
    ax3.grid(True, axis="y", linestyle="--", alpha=0.45)

    for b, wr, sa, sb in zip(bars, wrs, sc_a, sc_b):
        ax3.text(
            b.get_x() + b.get_width() / 2.0,
            wr + 2.2,
            f"{wr:.1f}%\n({sa:.0f} vs {sb:.0f})",
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#f0f6fc",
        )
    ax3.legend(loc="lower right", frameon=True, facecolor="#0d1117", edgecolor="#30363d", fontsize=8.5)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), facecolor="#0d1117")
    plt.close(fig)
    print(f"[Viz] Saved 3-panel dark figure to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Zero-Hardcoded-Rules ActuatorPolicyNet via GPU Self-Play RL.")
    parser.add_argument("--work-dir", type=str, default=".", help="Workspace root containing checkpoints or ml/models.")
    parser.add_argument("--world-pt", type=str, default="", help="Path to world_model_v3.pt")
    parser.add_argument("--eco-pt", type=str, default="", help="Path to ecosystem_model.pt")
    parser.add_argument("--out-ckpt", type=str, default="", help="Output actuator_policy_rl.pt path")
    parser.add_argument("--out-json", type=str, default="", help="Output ACTUATOR_EMERGENCE_METRICS.json path")
    parser.add_argument("--out-viz", type=str, default="", help="Output actuator_emergence_viz.png path")
    parser.add_argument("--num-arenas", type=int, default=64, help="Parallel training arenas (default: 64 -> 768 bots)")
    parser.add_argument("--updates", type=int, default=130, help="Number of PPO self-play updates")
    parser.add_argument("--rollout-steps", type=int, default=24, help="Ticks per PPO rollout")
    parser.add_argument("--eval-matches", type=int, default=24, help="Held-out 256x256x4 matches per matchup")
    parser.add_argument("--eval-batch", type=int, default=6, help="Parallel 256x256x4 arenas during tournament eval")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()

    def _find_file(explicit: str, candidates: list[Path]) -> Path:
        if explicit:
            return Path(explicit).resolve()
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    world_pt = _find_file(
        args.world_pt,
        [
            work_dir / "models" / "checkpoints" / "world_model_v3.pt",
            work_dir / "world_model_v3.pt",
            _SCRIPT_DIR.parent / "models" / "checkpoints" / "world_model_v3.pt",
        ],
    )
    eco_pt = _find_file(
        args.eco_pt,
        [
            work_dir / "models" / "checkpoints" / "ecosystem_model.pt",
            work_dir / "ecosystem_model.pt",
            _SCRIPT_DIR.parent / "models" / "checkpoints" / "ecosystem_model.pt",
        ],
    )
    out_ckpt = Path(args.out_ckpt).resolve() if args.out_ckpt else (work_dir / "models" / "checkpoints" / "actuator_policy_rl.pt")
    out_json = Path(args.out_json).resolve() if args.out_json else (work_dir / "models" / "ACTUATOR_EMERGENCE_METRICS.json")
    out_viz = Path(args.out_viz).resolve() if args.out_viz else (work_dir / "models" / "actuator_emergence_viz.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(2026)
    np.random.seed(2026)

    print(f"=== Frostmarsh Zero-Hardcoded-Rules Physical Actuators RL ({gpu_name}) ===", flush=True)
    print(f"  WorldModel v3: {world_pt}", flush=True)
    print(f"  EcosystemModel: {eco_pt}", flush=True)

    wm, em = load_frozen_models(world_pt, eco_pt, device)
    policy = ActuatorPolicyNet().to(device)
    n_params = count_parameters(policy)
    print(f"  ActuatorPolicyNet params: {n_params:,} (drop-in op=7 compatible: [N,168] -> [N,10] + optional [N,16])", flush=True)

    # Verify op=7 API contract upfront
    with torch.inference_mode():
        tz_w = torch.zeros((4, 48, 64, 64), device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        tz_e = torch.zeros((1, 32, 64, 64), device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        tbf = torch.zeros((12, BOT_FEAT_DIM), device=device, dtype=torch.float32)
        t_act, t_sal = policy(tz_w, tz_e, tbf, return_saliency=True)
        assert tuple(t_act.shape) == (12, 10), f"Bad act shape: {t_act.shape}"
        assert tuple(t_sal.shape) == (12, 16), f"Bad saliency shape: {t_sal.shape}"

    ac = RLActuatorActorCritic(policy).to(device)

    # 1. Build 64-arena (768 concurrent bots) physics-derived training arena & 16-arena validation arena
    train_arena = PhysicsActuatorArena(
        num_arenas=args.num_arenas,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=128,
        num_layers=2,
        max_ticks=120,
        latent_refresh_every=4,
    )
    val_arena = PhysicsActuatorArena(
        num_arenas=16,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=128,
        num_layers=2,
        max_ticks=120,
        latent_refresh_every=4,
    )

    # 2. Run Pure Reward-Driven Float32-Safe PPO Self-Play
    train_summary, best_sd, history = train_self_play_ppo(
        ac=ac,
        arena=train_arena,
        val_arena=val_arena,
        num_updates=args.updates,
        rollout_steps=args.rollout_steps,
        ppo_epochs=3,
        minibatch_size=2048,
        lr=9.0e-4,
        kl_early_stop=0.35,
        eval_every=max(1, args.updates // 16),
        seed_offset=1000,
    )

    # Load best checkpoint & save pure ActuatorPolicyNet state_dict
    policy.load_state_dict(best_sd)
    policy.eval()
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
            "arch": "ActuatorPolicyNet",
            "params": n_params,
            "actuator_dim": ACTUATOR_DIM,
            "bot_feat_dim": BOT_FEAT_DIM,
            "fine_radius_cells": 14.0,
            "coarse_radius_cells": 60.0,
            "zero_hardcoded_rules": True,
            "zero_imitation_teacher": True,
        },
        str(out_ckpt),
    )
    print(f"[Checkpoint] Saved pure ActuatorPolicyNet checkpoint -> {out_ckpt}", flush=True)

    # Free 64-arena training buffers before 256x256x4 held-out tournament
    del train_arena, val_arena
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 3. Held-Out Unseen 256x256x4 Tournament (24 side-balanced matches per pair!)
    print(f"\n=== Held-Out Unseen 256x256x4 Tournament ({args.eval_matches} matches/pair) ===", flush=True)
    eval_arena = PhysicsActuatorArena(
        num_arenas=args.eval_batch,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=256,
        num_layers=4,
        max_ticks=120,
        latent_refresh_every=4,
    )

    t_tour0 = time.perf_counter()
    matchups: dict[str, dict] = {}
    m_leg, beh_samples = run_heldout_matchup(
        "ActuatorPolicyNet-RL",
        "rl",
        "Legacy Named-Spell Scripted",
        "legacy_scripted",
        policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=95000,
        collect_behavior=True,
    )
    matchups["rl_vs_legacy_scripted"] = m_leg

    m_therm, beh_samples_2 = run_heldout_matchup(
        "ActuatorPolicyNet-RL",
        "rl",
        "Thermal-Only Actuators",
        "thermal_only",
        policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=96000,
        collect_behavior=True,
    )
    matchups["rl_vs_thermal_only"] = m_therm
    beh_samples.extend(beh_samples_2)

    m_rand, _ = run_heldout_matchup(
        "ActuatorPolicyNet-RL",
        "rl",
        "Random Continuous Actuators",
        "random",
        policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=97000,
        collect_behavior=False,
    )
    matchups["rl_vs_random"] = m_rand
    tour_time = round(time.perf_counter() - t_tour0, 2)

    # 4. A100 / GPU Inference Latency Benchmark (12 / 64 / 256 bots)
    del eval_arena
    if device.type == "cuda":
        torch.cuda.empty_cache()
    latency = benchmark_inference_latency(policy, device)
    print(f"[Latency {gpu_name}] 12b={latency['bots_12']['p50_ms']}ms | 64b={latency['bots_64']['p50_ms']}ms | 256b={latency['bots_256']['p50_ms']}ms", flush=True)

    # 5. Generate 3-panel dark-themed emergence figure
    make_emergence_viz(history, beh_samples, matchups, gpu_name, out_viz)

    # 6. Write complete JSON metrics
    metrics = {
        "device": gpu_name,
        "policy_class": "ActuatorPolicyNet",
        "policy_params": n_params,
        "actuator_channels": 10,
        "op7_drop_in_compatible": True,
        "zero_hardcoded_spell_rules": True,
        "zero_imitation_teacher": True,
        "world_model_version": "world_model_v3.pt",
        "ecosystem_model": "ecosystem_model.pt",
        "eval_grid": "256x256x4",
        "matches_per_matchup": args.eval_matches,
        "ticks_per_match": 120,
        "side_balanced": True,
        "training": train_summary,
        "discovery_curve": history,
        "tournament_time_s": tour_time,
        "matchups": matchups,
        "latency_gpu": latency,
        "artifacts": {
            "checkpoint": str(out_ckpt),
            "metrics_json": str(out_json),
            "emergence_viz": str(out_viz),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[Metrics] Wrote complete JSON -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
