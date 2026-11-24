#!/usr/bin/env python3
"""
Frostmarsh Wave 20B: ONE Shared 3D Physics Environment -> 3 Radically Different Emergent "Games"
& Universal Multi-Task PPO Training + Zero-Shot Morphology Generalization Evaluation.

ZERO hardcoded spell buttons. ZERO hardcoded class perks (`+22 HP`, `-35% dmg`, etc.).
ZERO game-mode `if/else` rules inside the physics step or policy!

All 3 emergent games run on the EXACT same vectorized GPU 3D physics kernel (`UniversalPhysics3DEnv`),
reading zero-copy 3D `Z_world_3d [A, 48, L, H, W]` latents from frozen `WorldModel3D-v7`
(`models/checkpoints/world_model_3d_v7.pt`), differing ONLY in the 6-float environment law vector
`theta_world` and initial physical scarcity/pressure:

  1. `FREEZE_OUT_SURVIVAL` ("King of the Hearth"):
       `theta_world = [g=1.0, amb_T=-0.55 (-55 K), precip=0.0, diss=0.14, fric=1.0, vent_flux=0.85]`
       Local voxel `T < 265 K` (`-0.25`) drains homeostasis; scarce geothermal vents + combustible
       wood/oil patches keep agents alive. Measures: hearth ignitions, thermal survival time, vent control.
  2. `RISING_FLOOD_ESCAPE` ("Bridge-Builders"):
       `theta_world = [g=1.0, amb_T=-0.05, precip=2.6 (2.6x heavy rain), diss=0.05, fric=1.0, vent_flux=0.25]`
       Heavy precipitation steadily floods the canyon floor. Low-buoyancy agents take submergence stress
       in deep water. Measures: cryo-bridges forged (`SP_WATER -> SP_ICE`), high-ground ascents/flights,
       and flood survival rate.
  3. `PHYSICS_CORE_HEIST` ("Zero-Rule Artifact"):
       `theta_world = [g=1.0, amb_T=0.0, precip=0.0, diss=0.05, fric=0.90, vent_flux=0.45]`
       One movable dense physical core rests in the center canyon and obeys pure Newtonian physics
       (kinetic impulse push, body-mass push, oil-reduced friction, ice-slide momentum, terrain slopes &
       bored tunnels). Agents move the core into the home valley purely via the 12 physical actuators!
       Measures: core displacement, deliveries, and oil/ice slide combos.

Saves:
  - `models/checkpoints/morphology_policy_universal.pt`
  - `models/EMERGENT_GAMES_METRICS.json`
  - `models/emergent_games_viz.png`
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from actuator_policy_v3 import (  # noqa: E402
    ACT3D_DIM,
    MAX_TOKENS_3D,
    SELF3D_FEAT_DIM,
    TOKEN3D_FEAT_DIM,
    count_parameters,
)
from morphology_policy import (  # noqa: E402
    HELD_OUT_MORPHOLOGIES,
    HOMEO_DIM,
    MORPH_DIM,
    MORPH_NAMES,
    WORLD_LAW_DIM,
    WORLD_LAW_NAMES,
    MorphologyActuatorNet,
    get_held_out_morphology_tensor,
    sample_training_morphologies,
)

# Exact FIELD_CONTRACT species & channel indices (matching WorldModel3D-v7 / sim_teacher)
SP_WATER, SP_ICE, SP_STEAM, SP_SOIL, SP_ROCK, SP_WOOD, SP_ASH, SP_SMOKE, SP_OIL, SP_PLANT = range(10)
CH_TEMP = 10
CH_LOGMASS = 11
CH_FLOW = 13
CH_ALT = 14
CH_OBST = 15

GAME_FREEZE_OUT = 0
GAME_RISING_FLOOD = 1
GAME_CORE_HEIST = 2

GAME_NAMES = (
    "FREEZE_OUT_SURVIVAL",
    "RISING_FLOOD_ESCAPE",
    "PHYSICS_CORE_HEIST",
)
GAME_SUBTITLES = (
    "King of the Hearth",
    "Bridge-Builders",
    "Zero-Rule Artifact",
)

# Pure continuous 6D physical environment laws theta_world (NO game-ID one-hot!):
#   [gravity_g, ambient_temp_offset, precip_rate, thermal_dissipation, friction_coeff, vent_flux]
# Normalized temperature convention: T_Kelvin = 290.0 + 100.0 * T_norm
#   -> -0.55 == 235 K (-55 K cold ambient), -0.25 == 265 K (hypothermia threshold), 0.0 == 290 K
GAME_THETA_WORLD: tuple[tuple[float, ...], ...] = (
    (1.0, -0.55, 0.0, 0.14, 1.00, 0.85),  # 0: FREEZE_OUT_SURVIVAL (-55 K cold, scarce geothermal vents)
    (1.0, -0.05, 2.6, 0.05, 1.00, 0.25),  # 1: RISING_FLOOD_ESCAPE (2.6x heavy precipitation flooding canyon)
    (1.0,  0.00, 0.0, 0.05, 0.90, 0.45),  # 2: PHYSICS_CORE_HEIST  (dense movable core, oil/ice friction physics)
)

AIM_RANGE_XY = 9.0
AIM_RANGE_Z = 5.0


# =============================================================================
# Frozen WorldModel3D-v7 Latent Extractor (Chunked to keep peak VRAM <= 2 GB!)
# =============================================================================


class FrozenWorldModelV7Latent(nn.Module):
    """
    Loads frozen `WorldModel3D-v7` (`WorldModel3DV5` architecture, strict=True) and encodes
    `[A, 18, L, H, W]` 3D voxel worlds into `[A, 48, L, H, W]` fp16 shared latents `Z_world_3d`.
    Processes arenas in small chunks (`chunk_size=6`) so peak VRAM stays tiny on Spark GB10!
    """

    def __init__(self, ckpt_path: Path | None, device: torch.device, chunk_size: int = 6):
        super().__init__()
        self.device = device
        self.chunk_size = max(1, int(chunk_size))
        self.mode = "fallback_3d"
        self.model: nn.Module | None = None

        if ckpt_path is not None and ckpt_path.exists():
            ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            try:
                from world_model_3d_v5 import WorldModel3DV5

                wm = WorldModel3DV5(in_ch=18, out_ch=17, width=48, z_flux_ch=5).to(device).eval()
                wm.load_state_dict(sd, strict=True)
                for p in wm.parameters():
                    p.requires_grad_(False)
                self.model = wm
                self.mode = "world_model_3d_v7"
                print(
                    f"[WorldLatent] Loaded frozen WorldModel3D-v7 (strict=True) from {ckpt_path}",
                    flush=True,
                )
                return
            except Exception as exc:
                print(f"[WorldLatent] WorldModel3D-v7 load failed ({exc}); using fallback 3D conv.", flush=True)

        torch.manual_seed(2026)
        self.fallback_conv = nn.Sequential(
            nn.Conv3d(18, 48, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(48, 48, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ).to(device).eval()
        for p in self.fallback_conv.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, w_state: torch.Tensor) -> torch.Tensor:
        """
        w_state: [A, 18, L, H, W] float32
        Returns z_world_3d: [A, 48, L, H, W] float16 (CUDA) or float32 (CPU)
        """
        A, _, L, H, W = w_state.shape
        use_fp16 = self.device.type == "cuda"
        out_dtype = torch.float16 if use_fp16 else torch.float32
        z_out = torch.empty((A, 48, L, H, W), device=self.device, dtype=out_dtype)

        for s in range(0, A, self.chunk_size):
            e = min(s + self.chunk_size, A)
            w_sub = w_state[s:e]
            if self.mode == "world_model_3d_v7" and self.model is not None:
                x_bl = w_sub.permute(0, 2, 1, 3, 4).contiguous()
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    _, z_bl = self.model(x_bl, local_dt=1.0, return_z_flux=False, return_latent=True)
                z_sub = z_bl.permute(0, 2, 1, 3, 4).contiguous()
            else:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    z_sub = self.fallback_conv(w_sub)
            z_out[s:e] = z_sub.to(out_dtype)
        return z_out


# =============================================================================
# ONE Shared Vectorized 3D Physics Environment (ZERO Hardcoded Game-Mode Step!)
# =============================================================================


class UniversalPhysics3DEnv:
    """
    Single shared vectorized GPU 3D voxel physics kernel (`L=16, H=48, W=48`).
    Runs ALL 3 emergent games simultaneously across parallel arenas using ONE unified
    physics step (`step()` has ZERO `if game == ...` branching!).
    """

    def __init__(
        self,
        num_arenas: int,
        num_agents: int,
        world_encoder: FrozenWorldModelV7Latent,
        device: torch.device,
        grid_xy: int = 48,
        depth_z: int = 16,
        max_ticks: int = 64,
        latent_refresh_every: int = 4,
    ):
        self.A = int(num_arenas)
        self.N = int(num_agents)
        self.H = int(grid_xy)
        self.W = int(grid_xy)
        self.L = int(depth_z)
        self.max_ticks = int(max_ticks)
        self.refresh_every = int(latent_refresh_every)
        self.device = device
        self.world_encoder = world_encoder

        self.arena_idx_flat = torch.arange(self.A, device=device).repeat_interleave(self.N)
        self.a_idx = torch.arange(self.A, device=device).unsqueeze(1).expand(self.A, self.N).contiguous()

        # 3D voxel coordinates
        zz = torch.arange(self.L, device=device, dtype=torch.float32)
        yy = torch.arange(self.H, device=device, dtype=torch.float32) / float(self.H)
        xx = torch.arange(self.W, device=device, dtype=torch.float32) / float(self.W)
        self.gz, self.gy, self.gx = torch.meshgrid(zz, yy, xx, indexing="ij")  # [L, H, W]

        # 3D Gaussian actuator deposit stencil (3 x 5 x 5 = 75 offsets)
        oz = torch.arange(-1, 2, device=device)
        oy = torch.arange(-2, 3, device=device)
        ox = torch.arange(-2, 3, device=device)
        dz, dy, dx = torch.meshgrid(oz, oy, ox, indexing="ij")
        self.st_dz = dz.reshape(-1)
        self.st_dy = dy.reshape(-1)
        self.st_dx = dx.reshape(-1)
        self.st_r2 = self.st_dx.float() ** 2 + self.st_dy.float() ** 2 + (self.st_dz.float() * 1.4) ** 2

        # World voxel state [A, 18, L, H, W] & cached frozen WorldModel3D-v7 latent [A, 48, L, H, W]
        self.w_state = torch.zeros((self.A, 18, self.L, self.H, self.W), dtype=torch.float32, device=device)
        z_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.z_world = torch.zeros((self.A, 48, self.L, self.H, self.W), dtype=z_dtype, device=device)

        # Static reference landmarks per arena (pure physical locations of scarce vents, fuel piles, ridges)
        self.num_vents = 2
        self.num_fuels = 4
        self.num_ridges = 3
        self.vent_xyz = torch.zeros((self.A, self.num_vents, 3), device=device)
        self.fuel_xyz = torch.zeros((self.A, self.num_fuels, 3), device=device)
        self.ridge_xyz = torch.zeros((self.A, self.num_ridges, 3), device=device)
        self.vent_mask = torch.zeros((self.A, self.L, self.H, self.W), dtype=torch.bool, device=device)
        self.basin_mask = torch.zeros((self.A, self.H, self.W), dtype=torch.bool, device=device)

        # Per-arena environment law vector theta_world [A, 6] & physical scenario assignment (used ONLY at reset/logging!)
        self.arena_scenario = torch.arange(self.A, device=device) % 3
        self.theta_world = torch.zeros((self.A, WORLD_LAW_DIM), device=device)

        # Per-agent continuous morphology phi_body [A, N, 6] & homeostasis [A, N, 4]
        self.phi_body = torch.zeros((self.A, self.N, MORPH_DIM), device=device)
        self.homeostasis = torch.zeros((self.A, self.N, HOMEO_DIM), device=device)
        self.min_holdout_l2_seen = 999.0

        # Movable physical core state (active wherever core_active > 0)
        self.core_active = torch.zeros((self.A,), device=device)
        self.core_x = torch.zeros((self.A,), device=device)
        self.core_y = torch.zeros((self.A,), device=device)
        self.core_z = torch.zeros((self.A,), device=device)
        self.core_vx = torch.zeros((self.A,), device=device)
        self.core_vy = torch.zeros((self.A,), device=device)
        self.core_start_x = torch.zeros((self.A,), device=device)
        self.core_start_y = torch.zeros((self.A,), device=device)
        self.valley_x = torch.zeros((self.A,), device=device)
        self.valley_y = torch.zeros((self.A,), device=device)
        self.valley_z = torch.zeros((self.A,), device=device)

        # Active Stasis Domes (1 slot per agent)
        self.dome_x = torch.zeros((self.A, self.N), device=device)
        self.dome_y = torch.zeros((self.A, self.N), device=device)
        self.dome_z = torch.zeros((self.A, self.N), device=device)
        self.dome_ttl = torch.zeros((self.A, self.N), dtype=torch.int32, device=device)

        self.reset(seed=0)

    def _solid_support_mask(self) -> torch.Tensor:
        """[A, L, H, W] bool where voxel supports physical standing weight (rock, soil, wood, buoyant ice)."""
        return (
            self.w_state[:, SP_ROCK]
            + self.w_state[:, SP_SOIL]
            + self.w_state[:, SP_WOOD]
            + self.w_state[:, SP_ICE]
        ) >= 0.34

    def _surface_z_map(self) -> torch.Tensor:
        """[A, H, W] highest solid voxel z+1 per column."""
        solid = self._solid_support_mask()  # [A, L, H, W]
        z_idx = torch.arange(self.L, device=self.device, dtype=torch.float32).view(1, self.L, 1, 1)
        top_solid = torch.where(solid, z_idx, torch.full_like(z_idx, -1.0)).max(dim=1).values
        return (top_solid + 1.0).clamp(1.0, float(self.L - 1))

    def _water_top_z_map(self) -> torch.Tensor:
        """[A, H, W] highest water voxel z+0.6 per column (0 if no water)."""
        has_w = self.w_state[:, SP_WATER] > 0.28
        z_idx = torch.arange(self.L, device=self.device, dtype=torch.float32).view(1, self.L, 1, 1)
        top_w = torch.where(has_w, z_idx + 0.6, torch.zeros_like(z_idx)).max(dim=1).values
        return top_w

    def _support_z_near(self, px: torch.Tensor, py: torch.Tensor, cur_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find walkable solid surface z near `(px, py, cur_z)` (respects tunnels, arches, ice rafts!).
        Returns `(target_z, fall_distance)`.
        """
        ix = px.round().long().clamp(0, self.W - 1)
        iy = py.round().long().clamp(0, self.H - 1)
        solid_col = self._solid_support_mask()[self.a_idx, :, iy, ix]  # [A, N, L]
        air_above = torch.cat(
            (~solid_col[:, :, 1:], torch.ones((self.A, self.N, 1), dtype=torch.bool, device=self.device)),
            dim=-1,
        )
        standable = solid_col & air_above  # [A, N, L]
        z_vals = torch.arange(self.L, device=self.device, dtype=torch.float32).view(1, 1, self.L) + 1.0
        # Prefer standable surface closest to cur_z (mild preference for nearby levels inside tunnels)
        dist = (z_vals - cur_z.unsqueeze(-1)).abs()
        score = torch.where(standable, dist, torch.full_like(dist, 999.0))
        best_z = z_vals.expand_as(score).gather(-1, score.argmin(dim=-1, keepdim=True)).squeeze(-1)
        has_any = standable.any(dim=-1)
        best_z = torch.where(has_any, best_z.clamp(1.0, float(self.L - 1)), torch.full_like(best_z, 2.0))
        fell = (cur_z - best_z).clamp(min=0.0)
        return best_z, fell

    def reset(
        self,
        seed: int = 0,
        scenario_override: int | None = None,
        fixed_phi_body: torch.Tensor | None = None,
        min_holdout_l2: float = 0.35,
    ) -> None:
        """
        Reset all `A` parallel 3D worlds with fresh procedural canyon geometry, `theta_world`,
        and continuous randomized `phi_body` (strictly holding out unseen hybrid morphologies).
        """
        dev = self.device
        A, N, L, H, W = self.A, self.N, self.L, self.H, self.W
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        self.tick = 0
        self.w_state.zero_()
        self.dome_ttl.zero_()
        self.vent_mask.zero_()

        # Assign scenario index per arena (ONLY to set initial theta_world & physical layout — never read in step!)
        if scenario_override is not None:
            self.arena_scenario = torch.full((A,), int(scenario_override), dtype=torch.long, device=dev)
        else:
            self.arena_scenario = torch.arange(A, device=dev, dtype=torch.long) % 3

        theta_table = torch.tensor(GAME_THETA_WORLD, device=dev, dtype=torch.float32)  # [3, 6]
        self.theta_world = theta_table[self.arena_scenario].clone()
        # Subtle continuous environmental law jitter (+-5%) during training for robustness
        if scenario_override is None:
            self.theta_world[:, 0] = (self.theta_world[:, 0] + torch.empty(A, device=dev).uniform_(-0.06, 0.06, generator=gen)).clamp(0.75, 1.35)
            self.theta_world[:, 4] = (self.theta_world[:, 4] + torch.empty(A, device=dev).uniform_(-0.04, 0.04, generator=gen)).clamp(0.70, 1.15)

        # Sample continuous 6D morphologies phi_body [A, N, 6]
        if fixed_phi_body is not None:
            if fixed_phi_body.dim() == 1:
                self.phi_body = fixed_phi_body.to(dev).view(1, 1, MORPH_DIM).expand(A, N, MORPH_DIM).clone()
            elif fixed_phi_body.dim() == 2:
                self.phi_body = fixed_phi_body.to(dev).unsqueeze(0).expand(A, N, MORPH_DIM).clone()
            else:
                self.phi_body = fixed_phi_body.to(dev).clone()
        else:
            self.phi_body, d_min = sample_training_morphologies((A, N), device=dev, generator=gen, min_holdout_l2=min_holdout_l2)
            self.min_holdout_l2_seen = min(self.min_holdout_l2_seen, d_min)

        # Build shared 3D canyon terrain [A, L, H, W]
        gx = self.gx.unsqueeze(0)  # [1, L, H, W]
        gy = self.gy.unsqueeze(0)
        gz = self.gz.unsqueeze(0)

        phase = torch.rand((A, 1, 1, 1), generator=gen, device=dev) * 6.28318
        canyon_cx = 0.50 + 0.04 * torch.sin(gy * 6.28318 + phase)
        dist_c = (gx - canyon_cx).abs()

        # Steep sculpted canyon walls (floor z=2.0 in middle, sheer vertical cliffs rising sharply
        # to z=9.2..10.5 on West & East -> blocks flightless low-grip bodies unless they cryo-bridge/climb/bore!)
        is_heist = (self.arena_scenario == GAME_CORE_HEIST).view(A, 1, 1, 1)
        is_flood = (self.arena_scenario == GAME_RISING_FLOOD).view(A, 1, 1, 1)

        canyon_h = 2.0 + 7.8 * ((dist_c - 0.21) / 0.06).clamp(0.0, 1.0)
        heist_corridor = (gy - 0.50).abs() < 0.24
        heist_h = torch.where(
            heist_corridor,
            2.4 + 2.7 * torch.exp(-((gx - 0.36) / 0.042) ** 2) * ((gy - 0.50).abs() < 0.13).float(),
            5.0 + 3.4 * ((gy - 0.50).abs() - 0.24).clamp(min=0.0) / 0.26,
        )
        ridge_noise = 1.0 * torch.sin(gy * 8.5 + phase).abs() * (dist_c > 0.28).float()
        terrain_z = torch.where(is_heist, heist_h, canyon_h + ridge_noise).clamp(1.8, 10.8)

        solid_ground = gz <= terrain_z
        bedrock = gz <= 1.2
        is_berm = is_heist & ((gx - 0.36).abs() < 0.055) & ((gy - 0.50).abs() < 0.13) & (gz > 2.2) & (gz <= terrain_z)
        self.w_state[:, SP_ROCK] = torch.where(
            bedrock | (solid_ground & (gz < terrain_z - 1.3) & (~is_berm)),
            1.0,
            0.0,
        )
        self.w_state[:, SP_SOIL] = torch.where(
            (solid_ground & (~bedrock) & (gz >= terrain_z - 1.3)) | is_berm,
            0.95,
            0.0,
        )

        # Ambient temperature initialized directly from theta_world[:, 1] (ambient_temp_offset!)
        amb_t = self.theta_world[:, 1].view(A, 1, 1, 1)
        self.w_state[:, CH_TEMP] = amb_t.expand(A, L, H, W).clone()

        # Basin mask for rainfall accumulation (canyon floor / center valley)
        self.basin_mask = (dist_c[:, 0] < 0.30)

        # Initial shallow canyon water when precip_rate > 0 (Flood) + a shallow stream/ice patch in Core Heist
        init_flood_water = is_flood & (dist_c < 0.24) & (gz > terrain_z) & (gz <= 3.6)
        heist_stream = is_heist & ((gx - 0.42).abs() < 0.06) & ((gy - 0.50).abs() < 0.18) & (gz > terrain_z) & (gz <= terrain_z + 1.1)
        self.w_state[:, SP_WATER] = torch.where(init_flood_water, 0.95, torch.where(heist_stream, 0.72, 0.0))

        # Place 2 SCARCE geothermal vents (warmth scaled by theta_world.vent_flux) & 4 rich combustible wood/oil piles!
        v_fx = torch.tensor([0.30, 0.70], device=dev).unsqueeze(0).expand(A, -1)
        v_fy = torch.tensor([0.22, 0.78], device=dev).unsqueeze(0).expand(A, -1)
        v_jx = torch.empty((A, self.num_vents), device=dev).uniform_(-0.04, 0.04, generator=gen)
        v_jy = torch.empty((A, self.num_vents), device=dev).uniform_(-0.04, 0.04, generator=gen)
        vx = ((v_fx + v_jx).clamp(0.18, 0.82) * (W - 1))
        vy = ((v_fy + v_jy).clamp(0.16, 0.84) * (H - 1))

        surf_z = self._surface_z_map()  # [A, H, W]
        a_v = torch.arange(A, device=dev).unsqueeze(1).expand(A, self.num_vents)
        vz = surf_z[a_v, vy.round().long().clamp(0, H - 1), vx.round().long().clamp(0, W - 1)]
        self.vent_xyz = torch.stack((vx, vy, vz), dim=-1)  # [A, 2, 3]

        # Stamp tight scarce geothermal vents (radius ~2.6m) so agents must ALSO ignite wood/oil hearths!
        gx_w = gx * float(W - 1)
        gy_h = gy * float(H - 1)
        vent_flux_4d = self.theta_world[:, 5].view(A, 1, 1, 1)
        for k in range(self.num_vents):
            vkx = vx[:, k].view(A, 1, 1, 1)
            vky = vy[:, k].view(A, 1, 1, 1)
            vkz = vz[:, k].view(A, 1, 1, 1)
            d2 = (gx_w - vkx) ** 2 + (gy_h - vky) ** 2
            vm = (d2 <= 7.0) & (gz >= (vkz - 1.5)) & (gz <= (vkz + 5.0))
            self.vent_mask = self.vent_mask | vm
            v_temp = amb_t + 1.05 * vent_flux_4d
            self.w_state[:, CH_TEMP] = torch.where(vm, torch.maximum(self.w_state[:, CH_TEMP], v_temp), self.w_state[:, CH_TEMP])
            self.w_state[:, SP_STEAM] = torch.where(vm & (gz >= vkz), torch.maximum(self.w_state[:, SP_STEAM], 0.45 * vent_flux_4d), self.w_state[:, SP_STEAM])

        # Place 4 rich combustible wood + oil fuel piles right across the cold central canyon!
        f_fx = torch.tensor([0.46, 0.56, 0.44, 0.58], device=dev).unsqueeze(0).expand(A, -1)
        f_fy = torch.tensor([0.42, 0.58, 0.62, 0.38], device=dev).unsqueeze(0).expand(A, -1)
        f_jx = torch.empty((A, self.num_fuels), device=dev).uniform_(-0.03, 0.03, generator=gen)
        f_jy = torch.empty((A, self.num_fuels), device=dev).uniform_(-0.03, 0.03, generator=gen)
        fx = ((f_fx + f_jx).clamp(0.18, 0.82) * (W - 1))
        fy = ((f_fy + f_jy).clamp(0.16, 0.84) * (H - 1))
        a_f = torch.arange(A, device=dev).unsqueeze(1).expand(A, self.num_fuels)
        fz = surf_z[a_f, fy.round().long().clamp(0, H - 1), fx.round().long().clamp(0, W - 1)]
        self.fuel_xyz = torch.stack((fx, fy, fz), dim=-1)  # [A, 4, 3]

        for k in range(self.num_fuels):
            fkx = fx[:, k].view(A, 1, 1, 1)
            fky = fy[:, k].view(A, 1, 1, 1)
            fkz = fz[:, k].view(A, 1, 1, 1)
            d2 = (gx_w - fkx) ** 2 + (gy_h - fky) ** 2
            fm = (d2 <= 12.0) & ((gz - fkz).abs() <= 1.5) & (~self.vent_mask)
            self.w_state[:, SP_WOOD] = torch.where(fm, 1.50, self.w_state[:, SP_WOOD])
            self.w_state[:, SP_OIL] = torch.where(fm & (d2 <= 6.5), 0.95, self.w_state[:, SP_OIL])

        # High-ground safe ridge landmarks (West cliff, East cliff, North high outcrop)
        r_fx = torch.tensor([0.14, 0.86, 0.16], device=dev).unsqueeze(0).expand(A, -1)
        r_fy = torch.tensor([0.32, 0.68, 0.72], device=dev).unsqueeze(0).expand(A, -1)
        rx = r_fx * (W - 1)
        ry = r_fy * (H - 1)
        a_r = torch.arange(A, device=dev).unsqueeze(1).expand(A, self.num_ridges)
        rz = surf_z[a_r, ry.round().long().clamp(0, H - 1), rx.round().long().clamp(0, W - 1)]
        self.ridge_xyz = torch.stack((rx, ry, rz), dim=-1)  # [A, 3, 3]

        # Movable dense physical core (present when arena_scenario == GAME_CORE_HEIST)
        self.core_active = (self.arena_scenario == GAME_CORE_HEIST).float()
        self.core_x = torch.full((A,), 0.54 * (W - 1), device=dev) + torch.empty(A, device=dev).uniform_(-1.5, 1.5, generator=gen)
        self.core_y = torch.full((A,), 0.50 * (H - 1), device=dev) + torch.empty(A, device=dev).uniform_(-2.5, 2.5, generator=gen)
        a_1 = torch.arange(A, device=dev)
        self.core_z = surf_z[a_1, self.core_y.round().long().clamp(0, H - 1), self.core_x.round().long().clamp(0, W - 1)]
        self.core_vx.zero_()
        self.core_vy.zero_()
        self.core_start_x = self.core_x.clone()
        self.core_start_y = self.core_y.clone()
        self.valley_x = torch.full((A,), 0.20 * (W - 1), device=dev)
        self.valley_y = torch.full((A,), 0.50 * (H - 1), device=dev)
        self.valley_z = surf_z[a_1, self.valley_y.round().long().clamp(0, H - 1), self.valley_x.round().long().clamp(0, W - 1)]

        # Spawn agents across central canyon / valley approaches
        sx_norm = torch.empty((A, N), device=dev).uniform_(0.42, 0.70, generator=gen)
        sy_norm = torch.empty((A, N), device=dev).uniform_(0.32, 0.68, generator=gen)
        self.pos_x = sx_norm * (W - 1)
        self.pos_y = sy_norm * (H - 1)
        self.pos_z = surf_z[self.a_idx, self.pos_y.round().long().clamp(0, H - 1), self.pos_x.round().long().clamp(0, W - 1)]
        self.vel_x = torch.zeros((A, N), device=dev)
        self.vel_y = torch.zeros((A, N), device=dev)
        self.vel_z = torch.zeros((A, N), device=dev)

        # Homeostasis state: [core_temp_norm, integrity_hp, energy_mana, submergence_stress]
        self.hp = torch.ones((A, N), device=dev)
        self.mana = torch.ones((A, N), device=dev)
        self.core_temp = torch.full((A, N), 0.50, device=dev)
        self.submerge_stress = torch.zeros((A, N), device=dev)
        self.alive = torch.ones((A, N), dtype=torch.bool, device=dev)
        self._sync_homeostasis_tensor()

        # Finalize altitude & logmass channels and run initial frozen WorldModel3D-v7 encode
        self.w_state[:, CH_ALT] = gz / float(L - 1)
        self.w_state[:, CH_LOGMASS] = torch.log1p(self.w_state[:, :10].sum(dim=1))
        self.z_world = self.world_encoder.encode(self.w_state)

        # Emergent physical telemetry counters (measured across episode, per arena)
        self.stat_hearth_ignitions = torch.zeros((A,), device=dev)
        self.stat_thermal_survival_ticks = torch.zeros((A, N), device=dev)
        self.stat_vent_control_ticks = torch.zeros((A,), device=dev)
        self.stat_cryo_bridges = torch.zeros((A,), device=dev)
        self.stat_high_ground_ascents = torch.zeros((A,), device=dev)
        self.stat_flights = torch.zeros((A,), device=dev)
        self.stat_wall_climbs = torch.zeros((A,), device=dev)
        self.stat_tunnels_bored = torch.zeros((A,), device=dev)
        self.stat_core_displacement = torch.zeros((A,), device=dev)
        self.stat_core_deliveries = torch.zeros((A,), device=dev)
        self.stat_slide_combos = torch.zeros((A,), device=dev)
        self.stat_cumulative_reward = torch.zeros((A, N), device=dev)

    def _sync_homeostasis_tensor(self) -> None:
        self.homeostasis[:, :, 0] = self.core_temp
        self.homeostasis[:, :, 1] = self.hp
        self.homeostasis[:, :, 2] = self.mana
        self.homeostasis[:, :, 3] = self.submerge_stress

    def build_observations(self) -> dict[str, torch.Tensor]:
        """
        Fully vectorized observation builder.
        CRITICAL: `self_feat` and `tokens` contain ZERO `phi_body` values — morphology enters
        `MorphologyActuatorNet` strictly through its FiLM conditioning path!
        """
        A, N, L, H, W = self.A, self.N, self.L, self.H, self.W
        dev = self.device
        a_idx = self.a_idx

        ix = self.pos_x.round().long().clamp(0, W - 1)
        iy = self.pos_y.round().long().clamp(0, H - 1)
        iz = self.pos_z.round().long().clamp(0, L - 1)
        iz_m1 = (iz - 1).clamp(0, L - 1)

        surf_z = self._surface_z_map()      # [A, H, W]
        wtop_z = self._water_top_z_map()    # [A, H, W]
        h_here = surf_z[a_idx, iy, ix]
        h_e = surf_z[a_idx, iy, (ix + 2).clamp(0, W - 1)]
        h_w = surf_z[a_idx, iy, (ix - 2).clamp(0, W - 1)]
        h_n = surf_z[a_idx, (iy - 2).clamp(0, H - 1), ix]
        h_s = surf_z[a_idx, (iy + 2).clamp(0, H - 1), ix]
        w_depth = (wtop_z[a_idx, iy, ix] - h_here).clamp(min=0.0)
        flood_lvl = wtop_z.reshape(A, -1).max(dim=-1).values.unsqueeze(1).expand(A, N)

        self_feat = torch.zeros((A, N, SELF3D_FEAT_DIM), device=dev)
        self_feat[:, :, 0] = (self.pos_x / max(W - 1, 1)) * 2.0 - 1.0
        self_feat[:, :, 1] = (self.pos_y / max(H - 1, 1)) * 2.0 - 1.0
        self_feat[:, :, 2] = (self.pos_z / max(L - 1, 1)) * 2.0 - 1.0
        self_feat[:, :, 3] = (self.vel_x / 3.0).clamp(-2.0, 2.0)
        self_feat[:, :, 4] = (self.vel_y / 3.0).clamp(-2.0, 2.0)
        self_feat[:, :, 5] = (self.vel_z / 3.0).clamp(-2.0, 2.0)
        self_feat[:, :, 6] = self.w_state[a_idx, CH_TEMP, iz, iy, ix].clamp(-2.0, 4.0)
        self_feat[:, :, 7] = torch.maximum(
            self.w_state[a_idx, SP_WATER, iz, iy, ix],
            self.w_state[a_idx, SP_WATER, iz_m1, iy, ix],
        ).clamp(0.0, 2.0)
        self_feat[:, :, 8] = torch.maximum(
            self.w_state[a_idx, SP_ICE, iz, iy, ix],
            self.w_state[a_idx, SP_ICE, iz_m1, iy, ix],
        ).clamp(0.0, 2.0)
        self_feat[:, :, 9] = self.w_state[a_idx, SP_OIL, iz, iy, ix].clamp(0.0, 2.0)
        self_feat[:, :, 10] = self.w_state[a_idx, SP_WOOD, iz, iy, ix].clamp(0.0, 2.0)
        self_feat[:, :, 11] = self.w_state[a_idx, SP_STEAM, iz, iy, ix].clamp(0.0, 2.0)
        self_feat[:, :, 12] = ((h_e - h_here) / 4.0).clamp(-2.0, 2.0)
        self_feat[:, :, 13] = ((h_w - h_here) / 4.0).clamp(-2.0, 2.0)
        self_feat[:, :, 14] = ((h_n - h_here) / 4.0).clamp(-2.0, 2.0)
        self_feat[:, :, 15] = ((h_s - h_here) / 4.0).clamp(-2.0, 2.0)
        self_feat[:, :, 16] = (w_depth / 5.0).clamp(0.0, 2.0)
        self_feat[:, :, 17] = (flood_lvl / float(L)).clamp(0.0, 1.0)
        self_feat[:, :, 18] = ((self.pos_z - flood_lvl) / 5.0).clamp(-2.0, 2.0)
        self_feat[:, :, 19] = self.hp
        self_feat[:, :, 20] = self.mana
        self_feat[:, :, 21] = self.core_temp

        # Physical 3D Tokens [A, N, 14, 14]
        toks = torch.zeros((A, N, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM), device=dev)
        mask = torch.zeros((A, N, MAX_TOKENS_3D), dtype=torch.bool, device=dev)

        # Slots 0..1: Scarce Geothermal Vents / Thermal Oases (slots 2..3 masked)
        nv = self.num_vents
        vdx = ((self.vent_xyz[:, None, :, 0] - self.pos_x[:, :, None]) / 20.0).clamp(-2.0, 2.0)
        vdy = ((self.vent_xyz[:, None, :, 1] - self.pos_y[:, :, None]) / 20.0).clamp(-2.0, 2.0)
        vdz = ((self.vent_xyz[:, None, :, 2] - self.pos_z[:, :, None]) / 8.0).clamp(-2.0, 2.0)
        vdist = torch.sqrt(vdx * vdx + vdy * vdy + vdz * vdz + 1e-6).clamp(0.0, 3.0)
        toks[:, :, 0:nv, 0] = vdx
        toks[:, :, 0:nv, 1] = vdy
        toks[:, :, 0:nv, 2] = vdz
        toks[:, :, 0:nv, 3] = vdist
        toks[:, :, 0:nv, 4] = self.theta_world[:, 5].view(A, 1, 1).expand(A, N, nv)
        toks[:, :, 0:nv, 8] = 1.0  # physical type: thermal vent
        mask[:, :, 0:nv] = True

        # Slots 4..7: Combustible Fuel Caches (wood + oil)
        fdx = ((self.fuel_xyz[:, None, :, 0] - self.pos_x[:, :, None]) / 20.0).clamp(-2.0, 2.0)
        fdy = ((self.fuel_xyz[:, None, :, 1] - self.pos_y[:, :, None]) / 20.0).clamp(-2.0, 2.0)
        fdz = ((self.fuel_xyz[:, None, :, 2] - self.pos_z[:, :, None]) / 8.0).clamp(-2.0, 2.0)
        fdist = torch.sqrt(fdx * fdx + fdy * fdy + fdz * fdz + 1e-6).clamp(0.0, 3.0)
        a_f = torch.arange(A, device=dev).unsqueeze(1).expand(A, self.num_fuels)
        fx_i = self.fuel_xyz[:, :, 0].round().long().clamp(0, W - 1)
        fy_i = self.fuel_xyz[:, :, 1].round().long().clamp(0, H - 1)
        fz_i = self.fuel_xyz[:, :, 2].round().long().clamp(0, L - 1)
        fuel_rem = (self.w_state[a_f, SP_WOOD, fz_i, fy_i, fx_i] + self.w_state[a_f, SP_OIL, fz_i, fy_i, fx_i]).clamp(0.0, 2.0)
        fuel_temp = self.w_state[a_f, CH_TEMP, fz_i, fy_i, fx_i].clamp(-2.0, 4.0)
        toks[:, :, 4:8, 0] = fdx
        toks[:, :, 4:8, 1] = fdy
        toks[:, :, 4:8, 2] = fdz
        toks[:, :, 4:8, 3] = fdist
        toks[:, :, 4:8, 4] = fuel_rem[:, None, :].expand(A, N, 4)
        toks[:, :, 4:8, 5] = fuel_temp[:, None, :].expand(A, N, 4)
        toks[:, :, 4:8, 9] = 1.0  # physical type: combustible fuel/hearth
        mask[:, :, 4:8] = True

        # Slots 8..10: High-Ground Ridges
        rdx = ((self.ridge_xyz[:, None, :, 0] - self.pos_x[:, :, None]) / 20.0).clamp(-2.0, 2.0)
        rdy = ((self.ridge_xyz[:, None, :, 1] - self.pos_y[:, :, None]) / 20.0).clamp(-2.0, 2.0)
        rdz = ((self.ridge_xyz[:, None, :, 2] - self.pos_z[:, :, None]) / 8.0).clamp(-2.0, 2.0)
        rdist = torch.sqrt(rdx * rdx + rdy * rdy + rdz * rdz + 1e-6).clamp(0.0, 3.0)
        toks[:, :, 8:11, 0] = rdx
        toks[:, :, 8:11, 1] = rdy
        toks[:, :, 8:11, 2] = rdz
        toks[:, :, 8:11, 3] = rdist
        toks[:, :, 8:11, 4] = ((self.ridge_xyz[:, None, :, 2] - flood_lvl[:, :, None]) / 6.0).clamp(-1.0, 2.0)
        toks[:, :, 8:11, 10] = 1.0  # physical type: high-ground ridge
        mask[:, :, 8:11] = True

        # Slot 11: Dense Movable Physical Core (valid whenever core_active > 0)
        cdx = ((self.core_x[:, None] - self.pos_x) / 20.0).clamp(-2.0, 2.0)
        cdy = ((self.core_y[:, None] - self.pos_y) / 20.0).clamp(-2.0, 2.0)
        cdz = ((self.core_z[:, None] - self.pos_z) / 8.0).clamp(-2.0, 2.0)
        cdist = torch.sqrt(cdx * cdx + cdy * cdy + cdz * cdz + 1e-6).clamp(0.0, 3.0)
        a_1 = torch.arange(A, device=dev)
        cix = self.core_x.round().long().clamp(0, W - 1)
        ciy = self.core_y.round().long().clamp(0, H - 1)
        ciz = (self.core_z.round().long() - 1).clamp(0, L - 1)
        core_slick = (
            self.w_state[a_1, SP_OIL, ciz, ciy, cix] + self.w_state[a_1, SP_ICE, ciz, ciy, cix]
        ).clamp(0.0, 2.0)
        toks[:, :, 11, 0] = cdx
        toks[:, :, 11, 1] = cdy
        toks[:, :, 11, 2] = cdz
        toks[:, :, 11, 3] = cdist
        toks[:, :, 11, 4] = (self.core_vx[:, None] / 2.0).clamp(-2.0, 2.0)
        toks[:, :, 11, 5] = (self.core_vy[:, None] / 2.0).clamp(-2.0, 2.0)
        toks[:, :, 11, 6] = core_slick[:, None].expand(A, N)
        toks[:, :, 11, 11] = 1.0  # physical type: movable core
        mask[:, :, 11] = (self.core_active[:, None] > 0.5).expand(A, N)

        # Slot 12: Home Valley Delivery Basin (valid whenever core_active > 0)
        hdx = ((self.valley_x[:, None] - self.pos_x) / 20.0).clamp(-2.0, 2.0)
        hdy = ((self.valley_y[:, None] - self.pos_y) / 20.0).clamp(-2.0, 2.0)
        hdz = ((self.valley_z[:, None] - self.pos_z) / 8.0).clamp(-2.0, 2.0)
        hdist = torch.sqrt(hdx * hdx + hdy * hdy + hdz * hdz + 1e-6).clamp(0.0, 3.0)
        c2v_dx = ((self.valley_x - self.core_x) / 20.0).clamp(-2.0, 2.0)[:, None].expand(A, N)
        c2v_dy = ((self.valley_y - self.core_y) / 20.0).clamp(-2.0, 2.0)[:, None].expand(A, N)
        toks[:, :, 12, 0] = hdx
        toks[:, :, 12, 1] = hdy
        toks[:, :, 12, 2] = hdz
        toks[:, :, 12, 3] = hdist
        toks[:, :, 12, 4] = c2v_dx
        toks[:, :, 12, 5] = c2v_dy
        toks[:, :, 12, 12] = 1.0  # physical type: home valley
        mask[:, :, 12] = (self.core_active[:, None] > 0.5).expand(A, N)

        # Slot 13: Nearest Peer Agent
        dx_p = self.pos_x[:, None, :] - self.pos_x[:, :, None]  # [A, N, N]
        dy_p = self.pos_y[:, None, :] - self.pos_y[:, :, None]
        dz_p = self.pos_z[:, None, :] - self.pos_z[:, :, None]
        d2_p = dx_p * dx_p + dy_p * dy_p + dz_p * dz_p
        eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
        d2_p = torch.where(eye, torch.full_like(d2_p, 1e9), d2_p)
        nn_idx = d2_p.argmin(dim=-1)  # [A, N]
        pdx = (dx_p.gather(-1, nn_idx.unsqueeze(-1)).squeeze(-1) / 20.0).clamp(-2.0, 2.0)
        pdy = (dy_p.gather(-1, nn_idx.unsqueeze(-1)).squeeze(-1) / 20.0).clamp(-2.0, 2.0)
        pdz = (dz_p.gather(-1, nn_idx.unsqueeze(-1)).squeeze(-1) / 8.0).clamp(-2.0, 2.0)
        pdist = torch.sqrt(pdx * pdx + pdy * pdy + pdz * pdz + 1e-6).clamp(0.0, 3.0)
        toks[:, :, 13, 0] = pdx
        toks[:, :, 13, 1] = pdy
        toks[:, :, 13, 2] = pdz
        toks[:, :, 13, 3] = pdist
        toks[:, :, 13, 4] = self.hp.gather(-1, nn_idx)
        toks[:, :, 13, 13] = 1.0  # physical type: peer agent
        mask[:, :, 13] = N > 1

        pos_xyz = torch.stack((self.pos_x, self.pos_y, self.pos_z), dim=-1).reshape(A * N, 3)
        theta_rep = self.theta_world.unsqueeze(1).expand(A, N, WORLD_LAW_DIM).reshape(A * N, WORLD_LAW_DIM)
        return {
            "pos_xyz": pos_xyz,
            "self_feat": self_feat.reshape(A * N, SELF3D_FEAT_DIM),
            "tokens": toks.reshape(A * N, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM),
            "token_mask": mask.reshape(A * N, MAX_TOKENS_3D),
            "phi_body": self.phi_body.reshape(A * N, MORPH_DIM),
            "theta_world": theta_rep,
            "homeostasis": self.homeostasis.reshape(A * N, HOMEO_DIM),
        }

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        ONE unified vectorized 3D physics step for ALL arenas & ALL emergent games!
        ZERO `if game == ...` conditionals! ZERO hardcoded spell buttons or class perks!

        actions: [A, N, 12] bounded continuous physical actuators.
        Returns `(rewards [A, N], done)`.
        """
        A, N, L, H, W = self.A, self.N, self.L, self.H, self.W
        dev = self.device
        a_idx = self.a_idx
        self.tick += 1
        self.dome_ttl = (self.dome_ttl - 1).clamp(min=0)

        alive_f = self.alive.float()

        # Unpack continuous 6D morphology phi_body [A, N, 6]
        mass = self.phi_body[:, :, 0]
        wing_lift = self.phi_body[:, :, 1]
        buoyancy = self.phi_body[:, :, 2]
        climb_grip = self.phi_body[:, :, 3]
        bore_torque = self.phi_body[:, :, 4]
        thermal_insul = self.phi_body[:, :, 5]
        m_eff = 0.25 + 1.15 * mass  # smooth inertial mass in [0.25, 1.40]

        # Unpack continuous 6D environment laws theta_world [A, 6]
        gravity_g = self.theta_world[:, 0:1].clamp(min=0.35)      # [A, 1]
        amb_temp = self.theta_world[:, 1:2]                       # [A, 1]
        precip_rate = self.theta_world[:, 2:3].clamp(min=0.0)     # [A, 1]
        therm_diss = self.theta_world[:, 3:4].clamp(0.01, 0.35)   # [A, 1]
        fric_coeff = self.theta_world[:, 4:5].clamp(0.20, 1.50)   # [A, 1]
        vent_flux = self.theta_world[:, 5:6].clamp(min=0.0)       # [A, 1]

        # ---------------------------------------------------------------------
        # 1. Morphology-Driven 3D Physics Locomotion (Flight, Swim, Climb, Bore)
        # ---------------------------------------------------------------------
        ix = self.pos_x.round().long().clamp(0, W - 1)
        iy = self.pos_y.round().long().clamp(0, H - 1)
        iz = self.pos_z.round().long().clamp(0, L - 1)
        iz_m1 = (iz - 1).clamp(0, L - 1)
        iz_m2 = (iz - 2).clamp(0, L - 1)

        water_here = torch.maximum(
            self.w_state[a_idx, SP_WATER, iz, iy, ix],
            self.w_state[a_idx, SP_WATER, iz_m1, iy, ix],
        )
        ice_here = torch.maximum(
            self.w_state[a_idx, SP_ICE, iz, iy, ix],
            self.w_state[a_idx, SP_ICE, iz_m1, iy, ix],
        )
        in_water = (water_here > 0.30) & (ice_here < 0.34)
        on_ice = ice_here >= 0.34

        # Thermal & steam updraft beneath agent -> amplifies physical flight lift!
        t_below = torch.maximum(
            self.w_state[a_idx, CH_TEMP, iz_m1, iy, ix],
            self.w_state[a_idx, CH_TEMP, iz_m2, iy, ix],
        )
        steam_below = torch.maximum(
            self.w_state[a_idx, SP_STEAM, iz, iy, ix],
            self.w_state[a_idx, SP_STEAM, iz_m1, iy, ix],
        )
        updraft = ((t_below - amb_temp).clamp(min=0.0) * 0.70 + steam_below * 1.25).clamp(0.0, 2.2)

        # Exact physical flight lift law: wing_lift * (1 + updraft) / (m_eff * gravity_g)
        flight_lift = wing_lift * (1.0 + updraft) / (m_eff * gravity_g)
        fly_ability = ((flight_lift - 0.55) / 0.75).clamp(0.0, 1.4)

        # Hydrodynamic speed & land speed emerge continuously from buoyancy & mass
        swim_spd = 0.65 + 2.45 * buoyancy
        land_spd = (2.35 - 0.55 * mass) * torch.where(on_ice, 1.25, 1.0)
        air_bonus = 0.60 * fly_ability * (self.pos_z > 4.5).float()
        speed = (torch.where(in_water, swim_spd, land_spd) + air_bonus) * alive_f

        mv_x = actions[:, :, 0]
        mv_y = actions[:, :, 1]
        mv_jump = actions[:, :, 2]
        mv_norm = torch.hypot(mv_x, mv_y).clamp(min=1.0)
        vx = (mv_x / mv_norm) * speed
        vy = (mv_y / mv_norm) * speed

        new_x = (self.pos_x + vx).clamp(2.0, W - 3.0)
        new_y = (self.pos_y + vy).clamp(2.0, H - 3.0)
        nx_i = new_x.round().long().clamp(0, W - 1)
        ny_i = new_y.round().long().clamp(0, H - 1)

        # Mass-Conserving Voxel Boring / Tunneling (`bore_torque` displaces SP_SOIL / soft SP_ROCK!)
        moving_hard = (torch.hypot(mv_x, mv_y) > 0.30) & self.alive & (bore_torque > 0.38)
        if moving_hard.any():
            # Check voxel at target horizontal step at current z & z+1
            z_b0 = iz.clamp(2, L - 2)
            soil_ahead = self.w_state[a_idx, SP_SOIL, z_b0, ny_i, nx_i]
            rock_ahead = self.w_state[a_idx, SP_ROCK, z_b0, ny_i, nx_i]
            can_bore = moving_hard & ((soil_ahead + rock_ahead) > 0.35)
            if can_bore.any():
                bore_cap = 0.68 * bore_torque * can_bore.float()
                d_soil = torch.minimum(soil_ahead, bore_cap)
                rem_cap = (bore_cap - d_soil).clamp(min=0.0)
                d_rock = torch.minimum(rock_ahead, rem_cap * 0.70)
                # Remove from ahead & deposit exact displaced mass behind/above (strict mass conservation!)
                z_dep = (z_b0 + 1).clamp(0, L - 1)
                self.w_state[a_idx, SP_SOIL, z_b0, ny_i, nx_i] = (soil_ahead - d_soil).clamp(min=0.0)
                self.w_state[a_idx, SP_ROCK, z_b0, ny_i, nx_i] = (rock_ahead - d_rock).clamp(min=0.0)
                self.w_state[a_idx, SP_SOIL, z_dep, iy, ix] = (self.w_state[a_idx, SP_SOIL, z_dep, iy, ix] + d_soil).clamp(max=2.0)
                self.w_state[a_idx, SP_ROCK, z_dep, iy, ix] = (self.w_state[a_idx, SP_ROCK, z_dep, iy, ix] + d_rock).clamp(max=2.0)
                bored_any = (d_soil + d_rock) > 0.15
                self.stat_tunnels_bored += bored_any.float().sum(dim=1)

        # Compute ground / ice-raft / cliff-climb / flight target altitude
        climb_cap = 1.25 + 5.2 * climb_grip
        climb_boost = (mv_jump > 0.15).float() * (0.75 + 2.4 * climb_grip) * alive_f
        gnd_z, gnd_fell = self._support_z_near(new_x, new_y, self.pos_z + climb_boost)

        # Buoyant surface swimming: high-buoyancy bodies float atop water columns!
        wtop_map = self._water_top_z_map()
        wtop_at_new = wtop_map[a_idx, ny_i, nx_i]
        buoyant_float_z = torch.where(
            (wtop_at_new > gnd_z) & (buoyancy > 0.45),
            gnd_z + (wtop_at_new - gnd_z) * ((buoyancy - 0.35) / 0.65).clamp(0.0, 1.0),
            gnd_z,
        )

        # True 3D flight vertical kinematics when flight_lift > 0.55
        fly_dz = fly_ability * (mv_jump * 2.10 + 0.55 * updraft - 0.22 * gravity_g)
        fly_z = torch.maximum(self.pos_z + fly_dz, buoyant_float_z).clamp(1.0, float(L - 2))
        can_fly = fly_ability > 0.08
        target_z = torch.where(can_fly, fly_z, buoyant_float_z)

        # Cliff-wall climb blocking: non-flyers cannot step up steeper than their climb_cap!
        dz_step = target_z - self.pos_z
        blocked = (~can_fly) & (dz_step > climb_cap) & self.alive
        prev_z = self.pos_z.clone()
        self.pos_x = torch.where(blocked, self.pos_x, new_x)
        self.pos_y = torch.where(blocked, self.pos_y, new_y)
        self.pos_z = torch.where(blocked, self.pos_z, target_z)
        self.vel_x = torch.where(blocked, torch.zeros_like(vx), vx)
        self.vel_y = torch.where(blocked, torch.zeros_like(vy), vy)
        self.vel_z = self.pos_z - prev_z

        # Track emergent flight & wall-climb events
        is_flying_now = can_fly & ((self.pos_z - gnd_z) > 1.8) & self.alive
        self.stat_flights += is_flying_now.float().sum(dim=1) * 0.25
        is_wall_climbing = (~can_fly) & (~blocked) & (dz_step > 1.35) & (climb_grip > 0.35) & self.alive
        self.stat_wall_climbs += is_wall_climbing.float().sum(dim=1)

        # ---------------------------------------------------------------------
        # 2. Exact 12 Conservation-Bounded Continuous 3D Physical Actuators
        # ---------------------------------------------------------------------
        aim_dx = actions[:, :, 3]
        aim_dy = actions[:, :, 4]
        aim_dz = actions[:, :, 5]
        thermal = actions[:, :, 6]
        emit_water = actions[:, :, 7].clamp(0.0, 1.0) ** 1.4
        emit_oil = actions[:, :, 8].clamp(0.0, 1.0) ** 1.4
        emit_smoke = actions[:, :, 9].clamp(0.0, 1.0) ** 1.4
        focus = actions[:, :, 10]
        imp_dome = actions[:, :, 11]

        tx = (self.pos_x + aim_dx * AIM_RANGE_XY).clamp(1.0, W - 2.0)
        ty = (self.pos_y + aim_dy * AIM_RANGE_XY).clamp(1.0, H - 2.0)
        tz_tgt = (self.pos_z + aim_dz * AIM_RANGE_Z).clamp(1.0, L - 2.0)

        rad = 1.6 + focus * 2.4
        impulse_raw = (imp_dome - 0.10).clamp(min=0.0) / 0.90
        want_dome = (imp_dome < -0.45) & (self.dome_ttl == 0) & self.alive
        # Physical jump exertion cost when flightless / low-grip bodies spam vertical jump futilely
        jump_effort_cost = (mv_jump.clamp(min=0.0) * 0.045 * (1.0 - 0.75 * torch.maximum(wing_lift, climb_grip))).clamp(min=0.0)
        energy_cost = (
            thermal.abs() * 0.09
            + emit_water * 0.04
            + emit_oil * 0.045
            + emit_smoke * 0.03
            + impulse_raw * 0.08
            + jump_effort_cost
        ) * (0.75 + 0.25 * focus) + want_dome.float() * 0.14
        can_pay = (self.mana >= energy_cost) & self.alive
        scale_pay = torch.where(can_pay, torch.ones_like(energy_cost), (self.mana / (energy_cost + 1e-5)).clamp(0.0, 1.0)) * alive_f
        self.mana = (self.mana - energy_cost * scale_pay + 0.12 * alive_f).clamp(0.0, 1.0)

        # Stasis Dome placement (reduces local cooling / flood rate inside dome)
        place_dome = want_dome & (scale_pay > 0.45)
        if place_dome.any():
            self.dome_x = torch.where(place_dome, tx, self.dome_x)
            self.dome_y = torch.where(place_dome, ty, self.dome_y)
            self.dome_z = torch.where(place_dome, tz_tgt, self.dome_z)
            self.dome_ttl = torch.where(place_dome, torch.full_like(self.dome_ttl, 14), self.dome_ttl)

        # 3D Gaussian actuator stream deposit (65% at aimed impact point + 35% at ray midpoint ->
        # continuous fluid/thermal jet that coats both near-field water/core and aimed target!)
        tx_i = tx.round().long()
        ty_i = ty.round().long()
        tz_i = tz_tgt.round().long()
        mx_i = (0.5 * (self.pos_x + tx)).round().long().clamp(1, W - 2)
        my_i = (0.5 * (self.pos_y + ty)).round().long().clamp(1, H - 2)
        mz_i = (0.5 * (self.pos_z + tz_tgt)).round().long().clamp(1, L - 2)

        inv_2s2 = 1.0 / (2.0 * (rad * 0.60) ** 2 + 1e-4)
        w_kern = torch.exp(-self.st_r2.view(1, 1, -1) * inv_2s2.unsqueeze(-1))
        peak_scale = 8.6 - 2.2 * focus.unsqueeze(-1)
        w_kern = w_kern / (w_kern.sum(dim=-1, keepdim=True) + 1e-6) * peak_scale

        gz_idx = (tz_i.unsqueeze(-1) + self.st_dz.view(1, 1, -1)).clamp(0, L - 1)
        gy_idx = (ty_i.unsqueeze(-1) + self.st_dy.view(1, 1, -1)).clamp(0, H - 1)
        gx_idx = (tx_i.unsqueeze(-1) + self.st_dx.view(1, 1, -1)).clamp(0, W - 1)
        flat_idx_a = ((gz_idx * H + gy_idx) * W + gx_idx).reshape(A, -1)

        mgz_idx = (mz_i.unsqueeze(-1) + self.st_dz.view(1, 1, -1)).clamp(0, L - 1)
        mgy_idx = (my_i.unsqueeze(-1) + self.st_dy.view(1, 1, -1)).clamp(0, H - 1)
        mgx_idx = (mx_i.unsqueeze(-1) + self.st_dx.view(1, 1, -1)).clamp(0, W - 1)
        flat_idx_m = ((mgz_idx * H + mgy_idx) * W + mgx_idx).reshape(A, -1)

        d_temp = (thermal * scale_pay * 3.85).unsqueeze(-1) * w_kern
        d_water = (emit_water * scale_pay * 0.85).unsqueeze(-1) * w_kern
        d_oil = (emit_oil * scale_pay * 1.15).unsqueeze(-1) * w_kern
        d_smoke = (emit_smoke * scale_pay * 0.55).unsqueeze(-1) * w_kern

        def _scatter_add(ch: int, vals: torch.Tensor) -> None:
            buf = self.w_state[:, ch].reshape(A, -1)
            v_flat = vals.reshape(A, -1)
            buf.scatter_add_(1, flat_idx_a, v_flat)
            buf.scatter_add_(1, flat_idx_m, v_flat * 0.45)

        _scatter_add(CH_TEMP, d_temp)
        _scatter_add(SP_WATER, d_water)
        _scatter_add(SP_OIL, d_oil)
        _scatter_add(SP_SMOKE, d_smoke)

        # Track where agents actively deposited heat / cryo this tick for crisp causal telemetry
        agent_heat_field = torch.zeros((A, L, H, W), device=dev)
        agent_cryo_field = torch.zeros((A, L, H, W), device=dev)
        h_flat = d_temp.clamp(min=0.0).reshape(A, -1)
        c_flat = (-d_temp).clamp(min=0.0).reshape(A, -1)
        agent_heat_field.reshape(A, -1).scatter_add_(1, flat_idx_a, h_flat)
        agent_heat_field.reshape(A, -1).scatter_add_(1, flat_idx_m, h_flat * 0.45)
        agent_cryo_field.reshape(A, -1).scatter_add_(1, flat_idx_a, c_flat)
        agent_cryo_field.reshape(A, -1).scatter_add_(1, flat_idx_m, c_flat * 0.45)

        # ---------------------------------------------------------------------
        # 3. Unified 3D Voxel Thermodynamics, Precipitation & Phase Transitions
        # ---------------------------------------------------------------------
        # Heavy precipitation (governed continuously by theta_world.precip_rate!)
        if (precip_rate > 0.05).any():
            # Steadily floods canyon basin from z=2 up toward z=8.0 over 64 ticks!
            target_flood_z = (2.8 + (self.tick / float(max(self.max_ticks, 1))) * 2.05 * precip_rate).view(A, 1, 1, 1)
            gz4 = self.gz.unsqueeze(0)
            solid_any = self._solid_support_mask()
            fill_mask = self.basin_mask.unsqueeze(1) & (~solid_any) & (gz4 >= 2.0) & (gz4 <= target_flood_z)
            rain_add = (0.16 * precip_rate.view(A, 1, 1, 1)) * fill_mask.float()
            self.w_state[:, SP_WATER] = (self.w_state[:, SP_WATER] + rain_add).clamp(0.0, 2.0)

        temp = self.w_state[:, CH_TEMP].clamp(-1.8, 5.5)
        water = self.w_state[:, SP_WATER].clamp(0.0, 2.0)
        ice = self.w_state[:, SP_ICE].clamp(0.0, 2.0)
        wood = self.w_state[:, SP_WOOD].clamp(0.0, 2.0)
        oil = self.w_state[:, SP_OIL].clamp(0.0, 2.0)
        smoke = self.w_state[:, SP_SMOKE].clamp(0.0, 2.0)
        steam = self.w_state[:, SP_STEAM].clamp(0.0, 2.0)

        # Cryo-freezing: SP_WATER -> SP_ICE when T < -0.18 (272 K) -> buoyant solid ice bridge!
        r_freeze = torch.minimum(water, 0.80 * (temp < -0.18).float())
        water = water - r_freeze
        ice = (ice + r_freeze).clamp(0.0, 2.0)

        # Archimedes buoyancy for SP_ICE (ρ_ice = 0.917 < ρ_water = 1.0):
        # Solid ice submerged directly beneath liquid water floats upward one voxel per tick
        # to form a walkable floating cryo-bridge atop rising floodwaters!
        ice_below = ice[:, :-1]
        water_above = water[:, 1:]
        float_mask = (ice_below > 0.25) & (water_above > 0.25)
        if float_mask.any():
            swap_ice = torch.where(float_mask, ice_below * 0.85, torch.zeros_like(ice_below))
            swap_w = torch.where(float_mask, water_above * 0.85, torch.zeros_like(water_above))
            ice_new = ice.clone()
            water_new = water.clone()
            ice_new[:, :-1] = (ice_new[:, :-1] - swap_ice).clamp(0.0, 2.0)
            ice_new[:, 1:] = (ice_new[:, 1:] + swap_ice).clamp(0.0, 2.0)
            water_new[:, 1:] = (water_new[:, 1:] - swap_w).clamp(0.0, 2.0)
            water_new[:, :-1] = (water_new[:, :-1] + swap_w).clamp(0.0, 2.0)
            ice, water = ice_new, water_new

        # Count cryo-bridge voxels causally forged by agent cryo flux!
        causal_cryo_voxels = ((r_freeze > 0.12) & (agent_cryo_field > 0.10)).float().sum(dim=(1, 2, 3))
        self.stat_cryo_bridges += (causal_cryo_voxels / 3.5).clamp(max=4.0)

        # Melting: SP_ICE -> SP_WATER when T > 0.45 (335 K)
        r_melt = torch.minimum(ice, 0.22 * (temp > 0.45).float())
        ice = ice - r_melt
        water = (water + r_melt).clamp(0.0, 2.0)

        # Flash-boiling: SP_WATER -> SP_STEAM when T > 0.85 (375 K)
        r_boil = torch.minimum(water, 0.45 * (temp > 0.85).float())
        water = water - r_boil
        steam = (steam + 1.25 * r_boil).clamp(0.0, 2.0)

        # Exothermic Combustion: SP_OIL (T > 0.22) & SP_WOOD (T > 0.30) -> Persistent Hearth Fire!
        r_oil = torch.minimum(oil, 0.42 * (temp > 0.22).float())
        oil = oil - r_oil
        temp = (temp + 2.85 * r_oil).clamp(-1.8, 5.5)
        smoke = (smoke + 0.50 * r_oil).clamp(0.0, 2.0)

        r_wood = torch.minimum(wood, 0.12 * (temp > 0.30).float())
        wood = wood - r_wood
        temp = (temp + 2.35 * r_wood).clamp(-1.8, 5.5)
        smoke = (smoke + 0.28 * r_wood).clamp(0.0, 2.0)

        # Count causal emergent hearth ignitions (agent heat igniting combustible wood/oil!)
        causal_hearth_voxels = (((r_wood + r_oil) > 0.08) & (agent_heat_field > 0.10)).float().sum(dim=(1, 2, 3))
        self.stat_hearth_ignitions += (causal_hearth_voxels / 4.0).clamp(max=3.0)

        # Ambient thermal relaxation toward theta_world.ambient_temp_offset + geothermal vent recharge
        diss4 = therm_diss.view(A, 1, 1, 1)
        amb4 = amb_temp.view(A, 1, 1, 1)
        temp = temp + diss4 * (amb4 - temp)
        vent_t = amb4 + 1.15 * vent_flux.view(A, 1, 1, 1)
        temp = torch.where(self.vent_mask, torch.maximum(temp, vent_t), temp)

        # Buoyant rising steam & smoke plumes
        smoke = F.pad(smoke[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0) * 0.72 + smoke * 0.16
        steam = F.pad(steam[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0) * 0.70 + steam * 0.16
        steam = torch.where(
            self.vent_mask & (self.gz.unsqueeze(0) >= 3.0),
            torch.maximum(steam, 0.35 * vent_flux.view(A, 1, 1, 1)),
            steam,
        )

        self.w_state[:, CH_TEMP] = temp
        self.w_state[:, SP_WATER] = water
        self.w_state[:, SP_ICE] = ice
        self.w_state[:, SP_WOOD] = wood
        self.w_state[:, SP_OIL] = oil
        self.w_state[:, SP_SMOKE] = smoke
        self.w_state[:, SP_STEAM] = steam
        self.w_state[:, CH_LOGMASS] = torch.log1p(self.w_state[:, :10].sum(dim=1))

        # ---------------------------------------------------------------------
        # 4. Pure-Physics Movable Dense Core Dynamics (active wherever core_active > 0)
        # ---------------------------------------------------------------------
        core_progress = torch.zeros((A,), device=dev)
        core_delivered = torch.zeros((A,), device=dev)
        core_slide_bonus = torch.zeros((A,), device=dev)
        core_engage_prox = torch.zeros((A, N), device=dev)
        agent_core_work = torch.zeros((A, N), device=dev)

        if (self.core_active > 0.5).any():
            act_c = self.core_active  # [A]
            prev_dist_v = torch.hypot(self.core_x - self.valley_x, self.core_y - self.valley_y)
            c2v_x = (self.valley_x - self.core_x) / prev_dist_v.clamp(min=0.5)  # unit vector core -> valley
            c2v_y = (self.valley_y - self.core_y) / prev_dist_v.clamp(min=0.5)

            # (a) Directional kinetic impulse SHOCKWAVE BEAM onto core (hits along 3D aim ray from agent to tx,ty,tz!)
            # Compute closest-point distance from core to the 3D aim ray segment [pos -> tgt]
            ag_to_cx = self.core_x[:, None] - self.pos_x
            ag_to_cy = self.core_y[:, None] - self.pos_y
            ag_to_cz = self.core_z[:, None] - self.pos_z
            ag_c_norm = torch.hypot(ag_to_cx, ag_to_cy).clamp(min=0.5)
            ray_dx = tx - self.pos_x
            ray_dy = ty - self.pos_y
            ray_len2 = (ray_dx * ray_dx + ray_dy * ray_dy).clamp(min=0.25)
            proj_t = ((ag_to_cx * ray_dx + ag_to_cy * ray_dy) / ray_len2).clamp(0.0, 1.0)
            clos_x = self.pos_x + proj_t * ray_dx
            clos_y = self.pos_y + proj_t * ray_dy
            dist_ray_c = torch.hypot(self.core_x[:, None] - clos_x, self.core_y[:, None] - clos_y)
            dist_ag_c = torch.hypot(ag_to_cx, ag_to_cy)

            # Shockwave beam hits core if core is along aim ray within (rad + 2.6m) AND within 11.5m of agent!
            in_imp_rad = (dist_ray_c <= (rad + 2.6)) & (dist_ag_c <= 11.5) & (impulse_raw > 0.05) & self.alive
            imp_str = torch.where(
                in_imp_rad,
                impulse_raw * scale_pay * (1.0 - dist_ray_c / (rad + 3.2)).clamp(min=0.15),
                0.0,
            )
            aim_norm = torch.hypot(aim_dx, aim_dy).clamp(min=0.35)
            # Shockwave pushes cleanly along the agent's aimed force vector!
            dir_x = 0.80 * (aim_dx / aim_norm) + 0.20 * (ag_to_cx / ag_c_norm)
            dir_y = 0.80 * (aim_dy / aim_norm) + 0.20 * (ag_to_cy / ag_c_norm)
            ag_imp_fx = imp_str * dir_x * 1.85
            ag_imp_fy = imp_str * dir_y * 1.85
            fx_imp = ag_imp_fx.sum(dim=1)
            fy_imp = ag_imp_fy.sum(dim=1)

            # Newton's 3rd-law kinetic recoil on firing agent (absorbed by heavy mass!)
            recoil_mag = (imp_str * 0.45 / m_eff).clamp(0.0, 1.2)
            self.pos_x = (self.pos_x - recoil_mag * (aim_dx / aim_norm)).clamp(2.0, W - 3.0)
            self.pos_y = (self.pos_y - recoil_mag * (aim_dy / aim_norm)).clamp(2.0, H - 3.0)

            # (b) Physical body-push momentum transfer (p ∝ m_eff * (0.45 + 1.10 * bore_torque) * v_agent!)
            touching = (dist_ag_c < 3.8) & self.alive
            push_pow = m_eff * (0.45 + 1.10 * bore_torque)
            v_norm_ag = torch.hypot(self.vel_x, self.vel_y).clamp(min=0.35)
            push_dx = 0.72 * (self.vel_x / v_norm_ag) + 0.28 * (ag_to_cx / ag_c_norm)
            push_dy = 0.72 * (self.vel_y / v_norm_ag) + 0.28 * (ag_to_cy / ag_c_norm)
            ag_body_fx = torch.where(touching, push_dx * push_pow * 0.95, 0.0)
            ag_body_fy = torch.where(touching, push_dy * push_pow * 0.95, 0.0)
            fx_body = ag_body_fx.sum(dim=1)
            fy_body = ag_body_fy.sum(dim=1)

            # Pure Newtonian individual work done by each agent along the core->valley axis (F_i · d_valley_hat)
            ag_fx_tot = ag_imp_fx + ag_body_fx
            ag_fy_tot = ag_imp_fy + ag_body_fy
            agent_core_work = (ag_fx_tot * c2v_x[:, None] + ag_fy_tot * c2v_y[:, None]) * act_c[:, None]

            # (c) Terrain slope gravity & berm blocking
            surf_z = self._surface_z_map()
            a_1 = torch.arange(A, device=dev)
            cix = self.core_x.round().long().clamp(1, W - 2)
            ciy = self.core_y.round().long().clamp(1, H - 2)
            ciz = (self.core_z.round().long() - 1).clamp(0, L - 1)
            dh_dx = (surf_z[a_1, ciy, cix + 1] - surf_z[a_1, ciy, cix - 1]) * 0.5
            dh_dy = (surf_z[a_1, ciy + 1, cix] - surf_z[a_1, ciy - 1, cix]) * 0.5
            fx_slope = -0.14 * gravity_g[:, 0] * dh_dx
            fy_slope = -0.14 * gravity_g[:, 0] * dh_dy

            # (d) Local surface friction: SP_OIL (-76%) & SP_ICE (-84%) unlock high-momentum 3.5x slides!
            oil_under = torch.maximum(
                self.w_state[a_1, SP_OIL, ciz, ciy, cix],
                self.w_state[a_1, SP_OIL, (ciz + 1).clamp(0, L - 1), ciy, cix],
            )
            ice_under = torch.maximum(
                self.w_state[a_1, SP_ICE, ciz, ciy, cix],
                self.w_state[a_1, SP_ICE, (ciz + 1).clamp(0, L - 1), ciy, cix],
            )
            on_oil = oil_under > 0.10
            on_ice_c = ice_under > 0.14
            eff_fric = fric_coeff[:, 0] * torch.where(on_ice_c, 0.16, torch.where(on_oil, 0.24, 1.0))
            vel_keep = (1.0 - 0.64 * eff_fric).clamp(0.20, 0.92)

            nvx = ((self.core_vx + fx_imp + fx_body + fx_slope) * vel_keep).clamp(-4.2, 4.2)
            nvy = ((self.core_vy + fy_imp + fy_body + fy_slope) * vel_keep).clamp(-4.2, 4.2)
            ncx = (self.core_x + nvx).clamp(2.0, W - 3.0)
            ncy = (self.core_y + nvy).clamp(2.0, H - 3.0)
            ncz = surf_z[a_1, ncy.round().long().clamp(0, H - 1), ncx.round().long().clamp(0, W - 1)]

            # Steep un-bored berm blocks core until either bored through or routed around!
            core_blocked = (ncz - self.core_z) > 1.20
            self.core_x = torch.where((act_c > 0.5) & (~core_blocked), ncx, self.core_x)
            self.core_y = torch.where((act_c > 0.5) & (~core_blocked), ncy, self.core_y)
            self.core_z = torch.where((act_c > 0.5) & (~core_blocked), ncz, self.core_z)
            self.core_vx = torch.where((act_c > 0.5) & (~core_blocked), nvx, self.core_vx * -0.25)
            self.core_vy = torch.where((act_c > 0.5) & (~core_blocked), nvy, self.core_vy * -0.25)

            new_dist_v = torch.hypot(self.core_x - self.valley_x, self.core_y - self.valley_y)
            step_disp = (prev_dist_v - new_dist_v) * act_c
            core_progress = step_disp
            self.stat_core_displacement += step_disp.clamp(min=0.0)

            # Oil/Ice slide combo detection (high-velocity core sliding over oil or ice!)
            core_spd = torch.hypot(self.core_vx, self.core_vy)
            is_sliding = (act_c > 0.5) & (on_oil | on_ice_c) & (core_spd > 0.28)
            self.stat_slide_combos += is_sliding.float()
            core_slide_bonus = torch.where(is_sliding & (step_disp > 0.02), core_spd * 0.65, 0.0)

            # Delivery into Home Valley!
            delivered = (act_c > 0.5) & (new_dist_v < 5.5)
            if delivered.any():
                core_delivered = delivered.float()
                self.stat_core_deliveries += core_delivered
                # Respawn core at center canyon so high-performing agents can score multiple deliveries!
                self.core_x = torch.where(delivered, self.core_start_x, self.core_x)
                self.core_y = torch.where(delivered, self.core_start_y, self.core_y)
                self.core_z = torch.where(
                    delivered,
                    surf_z[a_1, self.core_start_y.round().long().clamp(0, H - 1), self.core_start_x.round().long().clamp(0, W - 1)],
                    self.core_z,
                )
                self.core_vx = torch.where(delivered, torch.zeros_like(self.core_vx), self.core_vx)
                self.core_vy = torch.where(delivered, torch.zeros_like(self.core_vy), self.core_vy)

            core_engage_prox = (1.0 - (dist_ag_c / 14.0).clamp(0.0, 1.0)) * act_c[:, None] * alive_f

        # ---------------------------------------------------------------------
        # 5. Physical Homeostasis Update (ZERO Hardcoded Class Perks!)
        # ---------------------------------------------------------------------
        ix_p = self.pos_x.round().long().clamp(0, W - 1)
        iy_p = self.pos_y.round().long().clamp(0, H - 1)
        iz_p = self.pos_z.round().long().clamp(0, L - 1)
        iz_pm1 = (iz_p - 1).clamp(0, L - 1)

        local_t = temp[a_idx, iz_p, iy_p, ix_p]
        local_w = torch.maximum(water[a_idx, iz_p, iy_p, ix_p], water[a_idx, iz_pm1, iy_p, ix_p])
        local_ice = torch.maximum(ice[a_idx, iz_p, iy_p, ix_p], ice[a_idx, iz_pm1, iy_p, ix_p])
        in_water_post = (local_w > 0.30) & (local_ice < 0.34)

        surf_post = self._surface_z_map()[a_idx, iy_p, ix_p]
        wtop_post = self._water_top_z_map()[a_idx, iy_p, ix_p]
        subm_depth = (wtop_post - self.pos_z + 0.6).clamp(0.0, 5.0) / 3.0

        # Check if sheltered inside any friendly Stasis Dome (dampens thermal & flood stress by 45%)
        d2_domes = (
            (self.pos_x[:, :, None] - self.dome_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.dome_y[:, None, :]) ** 2
            + (self.pos_z[:, :, None] - self.dome_z[:, None, :]) ** 2
        )
        in_dome = ((self.dome_ttl[:, None, :] > 0) & (d2_domes < 32.0)).any(dim=-1)
        dome_shield = torch.where(in_dome, 0.55, 1.0)

        # Thermal tolerance widens continuously with `thermal_insul` (pure thermodynamics!)
        t_cold_thresh = -0.25 - 0.16 * thermal_insul   # 265 K at insul=0 -> 249 K at insul=1
        t_hot_thresh = 0.70 + 0.45 * thermal_insul     # 360 K at insul=0 -> 405 K at insul=1
        cold_stress = (t_cold_thresh - local_t).clamp(min=0.0) * (1.45 - 0.75 * thermal_insul) * dome_shield
        heat_stress = (local_t - t_hot_thresh).clamp(min=0.0) * (1.05 - 0.45 * thermal_insul) * dome_shield

        # Hydrodynamic submergence stress scales continuously with (1 - buoyancy)^1.5!
        submerge_stress = (
            in_water_post.float()
            * subm_depth
            * ((1.0 - buoyancy).clamp(min=0.0) ** 1.5)
            * 1.55
            * dome_shield
        )
        self.submerge_stress = submerge_stress.clamp(0.0, 2.0)

        prev_core_temp = self.core_temp.clone()
        target_ct = (0.50 + 0.65 * local_t).clamp(0.0, 1.0)
        ct_rate = 0.18 - 0.08 * thermal_insul
        self.core_temp = (self.core_temp + ct_rate * (target_ct - self.core_temp)).clamp(0.0, 1.0)

        hp_dmg = (cold_stress * 0.28 + heat_stress * 0.10 + submerge_stress * 0.21) * alive_f
        comfortable = (cold_stress < 0.01) & (heat_stress < 0.01) & (submerge_stress < 0.05) & self.alive
        hp_regen = 0.025 * comfortable.float()
        self.hp = (self.hp - hp_dmg + hp_regen).clamp(0.0, 1.0)
        self.alive = self.hp > 0.01
        self._sync_homeostasis_tensor()

        # Telemetry: thermal survival ticks, vent control, high-ground ascents
        self.stat_thermal_survival_ticks += self.alive.float()
        near_vent = (
            torch.hypot(
                self.pos_x[:, :, None] - self.vent_xyz[:, None, :, 0],
                self.pos_y[:, :, None] - self.vent_xyz[:, None, :, 1],
            ).min(dim=-1).values
            < 5.0
        ) & self.alive & (local_t > -0.20)
        self.stat_vent_control_ticks += near_vent.float().sum(dim=1)

        flood_lvl = self._water_top_z_map().reshape(A, -1).max(dim=-1).values.unsqueeze(1)
        above_flood = self.alive & (~in_water_post) & (self.pos_z >= (flood_lvl + 0.5)) & (self.pos_z >= 5.5)
        self.stat_high_ground_ascents += above_flood.float().sum(dim=1) * 0.25

        # ---------------------------------------------------------------------
        # 6. ONE Unified Physical Homeostasis & Work Reward (ZERO Game-Mode Code!)
        # ---------------------------------------------------------------------
        warming_relief = (self.core_temp - prev_core_temp).clamp(min=0.0) * (prev_core_temp < 0.46).float() * alive_f
        warm_oasis_bonus = (local_t > -0.18).float() * (amb_temp < -0.25).float() * alive_f
        dry_elev_bonus = (
            ((~in_water_post) & (self.pos_z >= (flood_lvl + 0.2))).float()
            * (precip_rate > 0.5).float()
            * alive_f
        )

        rewards = (
            0.045 * self.alive.float() * (0.40 + 0.60 * self.hp)
            - 1.75 * hp_dmg
            + 0.90 * warming_relief
            + 0.14 * warm_oasis_bonus
            + 0.14 * dry_elev_bonus
            + 1.35 * core_progress[:, None]
            + 0.85 * agent_core_work
            + 10.0 * core_delivered[:, None]
            + 0.04 * core_engage_prox
            + 0.65 * core_slide_bonus[:, None]
        )
        self.stat_cumulative_reward += rewards

        # Refresh frozen WorldModel3D-v7 shared 3D latent trunk periodically
        if self.tick % self.refresh_every == 0:
            self.z_world = self.world_encoder.encode(self.w_state)

        done = self.tick >= self.max_ticks
        return rewards, done


# =============================================================================
# Evaluation Suite (Full Universal vs Morphology-Blind vs Random + Zero-Shot Hybrids)
# =============================================================================


@torch.no_grad()
def evaluate_policy_on_game(
    policy: MorphologyActuatorNet | None,
    env: UniversalPhysics3DEnv,
    game_id: int,
    num_episodes: int,
    base_seed: int,
    mode: str = "full",
    fixed_phi_body: torch.Tensor | None = None,
) -> dict[str, Any]:
    """
    Evaluate on held-out unseen seeds for one emergent game (`game_id` in {0, 1, 2}).
    `mode`:
      - `"full"`       : Universal `MorphologyActuatorNet` with true `phi_body`
      - `"morph_blind"`: `Morphology-Blind Ablation` (`phi_body` zeroed at eval!)
      - `"random"`     : `Random-3D` baseline (uniform random 12D physical actuators)
    """
    if policy is not None:
        policy.eval()

    dev = env.device
    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    rounds = max(1, math.ceil(num_episodes / env.A))
    ep_collected = 0

    tot_return = 0.0
    tot_survival_rate = 0.0
    tot_survival_ticks = 0.0
    tot_hearth_ignitions = 0.0
    tot_vent_control = 0.0
    tot_cryo_bridges = 0.0
    tot_high_ground = 0.0
    tot_flights = 0.0
    tot_wall_climbs = 0.0
    tot_tunnels = 0.0
    tot_core_disp = 0.0
    tot_core_deliv = 0.0
    tot_slide_combos = 0.0

    for r in range(rounds):
        env.reset(
            seed=base_seed + game_id * 997 + r * 37,
            scenario_override=game_id,
            fixed_phi_body=fixed_phi_body,
        )
        for _ in range(env.max_ticks):
            obs = env.build_observations()
            if mode == "random" or policy is None:
                act_flat = torch.cat(
                    (
                        torch.empty((env.A * env.N, 7), device=dev).uniform_(-1.0, 1.0),
                        torch.empty((env.A * env.N, 4), device=dev).uniform_(0.0, 1.0),
                        torch.empty((env.A * env.N, 1), device=dev).uniform_(-1.0, 1.0),
                    ),
                    dim=-1,
                )
            else:
                fine_raw, coarse_raw, _ = policy.sample_raw_crops(
                    env.z_world, obs["pos_xyz"], arena_idx=env.arena_idx_flat
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = policy.forward_from_crops(
                        fine_raw,
                        coarse_raw,
                        obs["self_feat"],
                        obs["tokens"],
                        token_mask=obs["token_mask"],
                        phi_body=obs["phi_body"],
                        theta_world=obs["theta_world"],
                        homeostasis=obs["homeostasis"],
                        blind_morphology=(mode == "morph_blind"),
                    )
                act_flat = out["mean"].float()
            env.step(act_flat.reshape(env.A, env.N, ACT3D_DIM))

        take = min(env.A, num_episodes - ep_collected)
        tot_return += float(env.stat_cumulative_reward[:take].mean(dim=1).sum().item())
        tot_survival_rate += float(env.alive[:take].float().mean(dim=1).sum().item())
        tot_survival_ticks += float(env.stat_thermal_survival_ticks[:take].mean(dim=1).sum().item())
        tot_hearth_ignitions += float(env.stat_hearth_ignitions[:take].sum().item())
        tot_vent_control += float(env.stat_vent_control_ticks[:take].sum().item())
        tot_cryo_bridges += float(env.stat_cryo_bridges[:take].sum().item())
        tot_high_ground += float(env.stat_high_ground_ascents[:take].sum().item())
        tot_flights += float(env.stat_flights[:take].sum().item())
        tot_wall_climbs += float(env.stat_wall_climbs[:take].sum().item())
        tot_tunnels += float(env.stat_tunnels_bored[:take].sum().item())
        tot_core_disp += float(env.stat_core_displacement[:take].sum().item())
        tot_core_deliv += float(env.stat_core_deliveries[:take].sum().item())
        tot_slide_combos += float(env.stat_slide_combos[:take].sum().item())
        ep_collected += take

    n = float(max(ep_collected, 1))
    return {
        "episodes": ep_collected,
        "mean_return": round(tot_return / n, 3),
        "survival_rate": round(tot_survival_rate / n, 4),
        "thermal_survival_ticks": round(tot_survival_ticks / n, 2),
        "hearth_ignitions": round(tot_hearth_ignitions / n, 2),
        "vent_control_ticks": round(tot_vent_control / n, 2),
        "cryo_bridges_forged": round(tot_cryo_bridges / n, 2),
        "high_ground_ascents": round(tot_high_ground / n, 2),
        "aerial_flights": round(tot_flights / n, 2),
        "wall_climbs": round(tot_wall_climbs / n, 2),
        "tunnels_bored": round(tot_tunnels / n, 2),
        "core_displacement_m": round(tot_core_disp / n, 2),
        "core_deliveries": round(tot_core_deliv / n, 3),
        "oil_ice_slide_combos": round(tot_slide_combos / n, 2),
    }


def run_full_evaluation(
    policy: MorphologyActuatorNet,
    env: UniversalPhysics3DEnv,
    eval_episodes: int,
    base_seed: int = 9200,
) -> dict[str, Any]:
    """
    Evaluate all 3 emergent games comparing:
      1. Universal `MorphologyActuatorNet` (Full)
      2. `Morphology-Blind Ablation` (`phi_body` zeroed at eval)
      3. `Random-3D` baseline
      4. Zero-Shot Unseen Hybrid Morphologies (Amphibious Glider & Insulated Heavy Excavator)
    """
    results: dict[str, Any] = {
        "per_game": {},
        "zero_shot_hybrids": {},
    }
    for gid, gname in enumerate(GAME_NAMES):
        r_full = evaluate_policy_on_game(policy, env, gid, eval_episodes, base_seed, mode="full")
        r_blind = evaluate_policy_on_game(policy, env, gid, eval_episodes, base_seed, mode="morph_blind")
        r_rand = evaluate_policy_on_game(None, env, gid, eval_episodes, base_seed, mode="random")
        results["per_game"][gname] = {
            "universal_full": r_full,
            "morphology_blind_ablation": r_blind,
            "random_3d": r_rand,
            "ablation_return_delta": round(r_full["mean_return"] - r_blind["mean_return"], 3),
            "vs_random_return_delta": round(r_full["mean_return"] - r_rand["mean_return"], 3),
        }

    # Strictly Held-Out Zero-Shot Hybrid Morphologies evaluation across ALL 3 games!
    for h_name, h_vals in HELD_OUT_MORPHOLOGIES.items():
        phi_t = torch.tensor(h_vals, device=env.device, dtype=torch.float32)
        h_res: dict[str, Any] = {"phi_body": list(h_vals), "games": {}}
        for gid, gname in enumerate(GAME_NAMES):
            rf = evaluate_policy_on_game(
                policy, env, gid, eval_episodes, base_seed + 4000, mode="full", fixed_phi_body=phi_t
            )
            rb = evaluate_policy_on_game(
                policy, env, gid, eval_episodes, base_seed + 4000, mode="morph_blind", fixed_phi_body=phi_t
            )
            rr = evaluate_policy_on_game(
                None, env, gid, eval_episodes, base_seed + 4000, mode="random", fixed_phi_body=phi_t
            )
            h_res["games"][gname] = {
                "zero_shot_full": rf,
                "morphology_blind": rb,
                "random_3d": rr,
                "morph_adaptation_gain": round(rf["mean_return"] - rb["mean_return"], 3),
            }
        results["zero_shot_hybrids"][h_name] = h_res

    return results


# =============================================================================
# Crisp 3-Panel Dark-Theme Chart (`models/emergent_games_viz.png`)
# =============================================================================


def save_emergent_games_viz(
    eval_results: dict[str, Any],
    history: list[dict[str, Any]],
    out_png: Path,
    n_params: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[Viz] matplotlib unavailable in this Python env; skipping {out_png} (render with --render-viz-from-json).", flush=True)
        return

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#30363d",
            "axes.linewidth": 0.9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=160)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"Wave 20B: Universal MorphologyActuatorNet ({n_params:,} params) — 3 Emergent 3D Games from Pure Physics (Zero Hardcoded Rules)",
        color="#f0f6fc",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9", labelsize=8.8)
        ax.grid(axis="y", color="#30363d", linestyle="--", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    short_games = ["Freeze-Out\n(Hearth)", "Rising Flood\n(Bridge-Build)", "Core Heist\n(Artifact)"]
    x = np.arange(3)

    # -------------------------------------------------------------------------
    # Panel 1: Universal Full vs Morphology-Blind Ablation vs Random-3D (Mean Return)
    # -------------------------------------------------------------------------
    ax0 = axes[0]
    w = 0.25
    full_ret = [eval_results["per_game"][g]["universal_full"]["mean_return"] for g in GAME_NAMES]
    blind_ret = [eval_results["per_game"][g]["morphology_blind_ablation"]["mean_return"] for g in GAME_NAMES]
    rand_ret = [eval_results["per_game"][g]["random_3d"]["mean_return"] for g in GAME_NAMES]

    b1 = ax0.bar(x - w, full_ret, width=w, color="#3fb950", label="Universal Full")
    b2 = ax0.bar(x, blind_ret, width=w, color="#d29922", label="Morph-Blind (φ=0)")
    b3 = ax0.bar(x + w, rand_ret, width=w, color="#f85149", label="Random-3D")

    ax0.set_xticks(x)
    ax0.set_xticklabels(short_games, color="#f0f6fc", fontsize=9)
    ax0.set_ylabel("Held-Out Episode Return", color="#c9d1d9", fontsize=9.5)
    ax0.set_title("1. All 3 Emergent Games: Full vs Ablations", color="#f0f6fc", fontsize=10.8, fontweight="bold", pad=10)
    ax0.axhline(0.0, color="#8b949e", lw=0.8)
    y_min = min(full_ret + blind_ret + rand_ret + [0.0])
    y_max = max(full_ret + blind_ret + rand_ret + [1.0])
    span = max(y_max - y_min, 1.0)
    ax0.set_ylim(y_min - 0.14 * span, y_max + 0.22 * span)
    for bars in (b1, b2, b3):
        for bar in bars:
            v = bar.get_height()
            va = "bottom" if v >= 0 else "top"
            off = 0.025 * span if v >= 0 else -0.025 * span
            ax0.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + off,
                f"{v:+.1f}",
                ha="center",
                va=va,
                color="#f0f6fc",
                fontsize=7.8,
                fontweight="bold",
            )
    ax0.legend(loc="upper left", frameon=True, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=8.2)

    # -------------------------------------------------------------------------
    # Panel 2: Zero-Shot Unseen Hybrid Morphologies (Glider & Heavy Excavator)
    # -------------------------------------------------------------------------
    ax1 = axes[1]
    glider_g = eval_results["zero_shot_hybrids"]["amphibious_glider"]["games"]
    excav_g = eval_results["zero_shot_hybrids"]["insulated_heavy_excavator"]["games"]

    g_full = [glider_g[g]["zero_shot_full"]["mean_return"] for g in GAME_NAMES]
    g_blind = [glider_g[g]["morphology_blind"]["mean_return"] for g in GAME_NAMES]
    e_full = [excav_g[g]["zero_shot_full"]["mean_return"] for g in GAME_NAMES]
    e_blind = [excav_g[g]["morphology_blind"]["mean_return"] for g in GAME_NAMES]

    w2 = 0.19
    p1 = ax1.bar(x - 1.5 * w2, g_full, width=w2, color="#58a6ff", label="Glider (Zero-Shot)")
    p2 = ax1.bar(x - 0.5 * w2, g_blind, width=w2, color="#30507a", label="Glider (φ=0)")
    p3 = ax1.bar(x + 0.5 * w2, e_full, width=w2, color="#a371f7", label="Excavator (Zero-Shot)")
    p4 = ax1.bar(x + 1.5 * w2, e_blind, width=w2, color="#4e3775", label="Excavator (φ=0)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_games, color="#f0f6fc", fontsize=9)
    ax1.set_ylabel("Zero-Shot Held-Out Return", color="#c9d1d9", fontsize=9.5)
    ax1.set_title("2. Zero-Shot Unseen Hybrid Morphologies", color="#f0f6fc", fontsize=10.8, fontweight="bold", pad=10)
    ax1.axhline(0.0, color="#8b949e", lw=0.8)
    y2_min = min(g_full + g_blind + e_full + e_blind + [0.0])
    y2_max = max(g_full + g_blind + e_full + e_blind + [1.0])
    span2 = max(y2_max - y2_min, 1.0)
    ax1.set_ylim(y2_min - 0.14 * span2, y2_max + 0.28 * span2)
    for bars in (p1, p3):
        for bar in bars:
            v = bar.get_height()
            va = "bottom" if v >= 0 else "top"
            off = 0.025 * span2 if v >= 0 else -0.025 * span2
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + off,
                f"{v:+.1f}",
                ha="center",
                va=va,
                color="#f0f6fc",
                fontsize=7.6,
                fontweight="bold",
            )
    ax1.legend(loc="upper left", ncol=2, frameon=True, facecolor="#0d1117", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=7.6)

    # -------------------------------------------------------------------------
    # Panel 3: Emergent Physical Tactic Discoveries per Game (Zero Hardcoded Rules!)
    # -------------------------------------------------------------------------
    ax2 = axes[2]
    g0 = eval_results["per_game"]["FREEZE_OUT_SURVIVAL"]["universal_full"]
    g1 = eval_results["per_game"]["RISING_FLOOD_ESCAPE"]["universal_full"]
    g2 = eval_results["per_game"]["PHYSICS_CORE_HEIST"]["universal_full"]

    t_labels = [
        "G1: Hearth\nIgnitions",
        "G1: Surv.\nRate %",
        "G2: Cryo\nBridges",
        "G2: High-Gnd\n/ Flights",
        "G2: Surv.\nRate %",
        "G3: Core\nDisp. (m)",
        "G3: Oil/Ice\nSlides",
    ]
    t_vals = [
        g0["hearth_ignitions"],
        g0["survival_rate"] * 100.0,
        g1["cryo_bridges_forged"],
        g1["high_ground_ascents"] + g1["aerial_flights"],
        g1["survival_rate"] * 100.0,
        g2["core_displacement_m"],
        g2["oil_ice_slide_combos"],
    ]
    t_cols = ["#ffa657", "#ff7b72", "#79c0ff", "#58a6ff", "#3fb950", "#d2a8ff", "#f778ba"]

    tx_pos = np.arange(len(t_labels))
    b_tac = ax2.bar(tx_pos, t_vals, color=t_cols, width=0.62)
    ax2.set_xticks(tx_pos)
    ax2.set_xticklabels(t_labels, color="#f0f6fc", fontsize=7.6)
    ax2.set_ylabel("Per-Episode Metric / %", color="#c9d1d9", fontsize=9.5)
    ax2.set_title("3. Emergent Physical Tactics per Game (Full Policy)", color="#f0f6fc", fontsize=10.8, fontweight="bold", pad=10)
    top3 = max(t_vals + [10.0]) * 1.20
    ax2.set_ylim(0.0, top3)
    for bar, v in zip(b_tac, t_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + top3 * 0.025,
            f"{v:.1f}",
            ha="center",
            color="#f0f6fc",
            fontsize=8.0,
            fontweight="bold",
        )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Viz] Saved 3-panel chart -> {out_png}", flush=True)


# =============================================================================
# Universal Multi-Task PPO Trainer (100% Pure RL — Zero Hardcoded Rules!)
# =============================================================================


def _resolve_world_ckpt(path: Path) -> Path | None:
    if path.exists():
        return path
    alt = Path("/home/gooning/.t3/scratch/frostmarsh-ml") / path.name
    if alt.exists():
        return alt
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-ckpt", type=Path, default=Path("models/checkpoints/world_model_3d_v7.pt"))
    ap.add_argument("--out-ckpt", type=Path, default=Path("models/checkpoints/morphology_policy_universal.pt"))
    ap.add_argument("--out-json", type=Path, default=Path("models/EMERGENT_GAMES_METRICS.json"))
    ap.add_argument("--out-png", type=Path, default=Path("models/emergent_games_viz.png"))
    ap.add_argument("--num-arenas", type=int, default=24)  # 8 arenas x 3 emergent games simultaneously!
    ap.add_argument("--num-agents", type=int, default=6)
    ap.add_argument("--grid-xy", type=int, default=48)
    ap.add_argument("--depth-z", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--iters", type=int, default=140)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--eval-episodes", type=int, default=24)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--render-viz-from-json", type=Path, default=None)
    args = ap.parse_args()

    if args.render_viz_from_json is not None:
        data = json.loads(args.render_viz_from_json.read_text())
        save_emergent_games_viz(data["evaluation"], data.get("history", []), args.out_png, n_params=int(data["params"]))
        return

    if args.smoke_test:
        args.num_arenas = 3
        args.num_agents = 4
        args.grid_xy = 32
        args.depth_z = 16
        args.rollout_steps = 12
        args.iters = 3
        args.ppo_epochs = 2
        args.minibatch = 64
        args.eval_episodes = 3

    # Ensure arenas is divisible by 3 so all 3 emergent games are perfectly balanced in every rollout
    if args.num_arenas % 3 != 0:
        args.num_arenas = max(3, (args.num_arenas // 3) * 3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    t0 = time.time()

    print(
        f"[Wave 20B EmergentGames] Device={device} (bf16={use_bf16}) | arenas={args.num_arenas} "
        f"(x{args.num_agents} agents) | grid={args.grid_xy}x{args.grid_xy}x{args.depth_z} | iters={args.iters} | steps={args.rollout_steps}",
        flush=True,
    )

    wm_ckpt = _resolve_world_ckpt(args.world_ckpt)
    world_enc = FrozenWorldModelV7Latent(wm_ckpt, device=device, chunk_size=6)

    env = UniversalPhysics3DEnv(
        num_arenas=args.num_arenas,
        num_agents=args.num_agents,
        world_encoder=world_enc,
        device=device,
        grid_xy=args.grid_xy,
        depth_z=args.depth_z,
        max_ticks=args.rollout_steps,
        latent_refresh_every=8,
    )

    policy = MorphologyActuatorNet(coarse_span_xy=float(args.grid_xy)).to(device)
    n_params = count_parameters(policy)
    with torch.no_grad():
        # Neutral moderate emission bias on sigmoid channels 7..9 & healthy initial exploration std
        policy.actor_mean.bias[7:10].fill_(-0.75)
        policy.actor_log_std.fill_(-0.35)

    print(
        f"[Wave 20B] MorphologyActuatorNet params={n_params:,} | WorldLatent={world_enc.mode}",
        flush=True,
    )

    film_params = (
        list(policy.cond_in_proj.parameters())
        + list(policy.cond_out_proj.parameters())
        + list(policy.film_fine.parameters())
        + list(policy.film_coarse.parameters())
        + list(policy.film_self.parameters())
        + list(policy.film_trunk.parameters())
        + list(policy.cond_actor_bias.parameters())
    )
    film_ids = {id(p) for p in film_params}
    base_params = [p for p in policy.parameters() if id(p) not in film_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": film_params, "lr": args.lr * 1.4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.iters, 1), eta_min=args.lr * 0.18
    )

    # Precompute per-game agent masks for exact equal-weight per-game advantage normalization!
    B_tot = env.A * env.N
    agent_game_id = env.arena_scenario.repeat_interleave(env.N)  # [A*N]
    game_masks = [(agent_game_id == g) for g in range(3)]

    history: list[dict[str, Any]] = []
    best_score = -1e9
    best_sd = copy.deepcopy(policy.state_dict())

    for it in range(1, args.iters + 1):
        env.reset(seed=1000 + it * 23)

        buf_fine, buf_coarse, buf_sf, buf_tok, buf_mask = [], [], [], [], []
        buf_phi, buf_theta, buf_homeo = [], [], []
        buf_u, buf_lp, buf_val, buf_rew, buf_done = [], [], [], [], []

        policy.eval()
        for step in range(args.rollout_steps):
            obs = env.build_observations()
            with torch.inference_mode():
                fine_raw, coarse_raw, _ = policy.sample_raw_crops(
                    env.z_world, obs["pos_xyz"], arena_idx=env.arena_idx_flat
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = policy.forward_from_crops(
                        fine_raw,
                        coarse_raw,
                        obs["self_feat"],
                        obs["tokens"],
                        token_mask=obs["token_mask"],
                        phi_body=obs["phi_body"],
                        theta_world=obs["theta_world"],
                        homeostasis=obs["homeostasis"],
                    )
                act_flat, u_flat, lp_flat = policy.sample_action(out)
                val_flat = out["value"]

            rew_mat, done = env.step(act_flat.reshape(env.A, env.N, ACT3D_DIM))
            rew_flat = rew_mat.reshape(-1) * 0.25

            buf_fine.append(fine_raw.detach())
            buf_coarse.append(coarse_raw.detach())
            buf_sf.append(obs["self_feat"].detach())
            buf_tok.append(obs["tokens"].detach())
            buf_mask.append(obs["token_mask"].detach())
            buf_phi.append(obs["phi_body"].detach())
            buf_theta.append(obs["theta_world"].detach())
            buf_homeo.append(obs["homeostasis"].detach())
            buf_u.append(u_flat.detach().float())
            buf_lp.append(lp_flat.detach().float())
            buf_val.append(val_flat.detach().float())
            buf_rew.append(rew_flat.detach().float())
            buf_done.append(torch.full((B_tot,), float(done), device=device))

            if done and step + 1 < args.rollout_steps:
                env.reset(seed=50000 + it * 101 + step)

        # GAE-Lambda in strict float32 + per-game advantage normalization so all 3 emergent games learn in lockstep!
        T = len(buf_rew)
        rews = torch.stack(buf_rew, dim=0)    # [T, B_tot]
        vals = torch.stack(buf_val, dim=0)    # [T, B_tot]
        dones = torch.stack(buf_done, dim=0)  # [T, B_tot]
        adv = torch.zeros_like(rews)
        last_gae = torch.zeros(B_tot, device=device)
        gamma, lam = 0.97, 0.95
        for t in reversed(range(T)):
            next_v = vals[t + 1] if t + 1 < T else torch.zeros(B_tot, device=device)
            non_term = 1.0 - dones[t]
            delta = rews[t] + gamma * next_v * non_term - vals[t]
            last_gae = delta + gamma * lam * non_term * last_gae
            adv[t] = last_gae
        returns = adv + vals

        # Normalize advantages separately per game (equal 1/3 policy gradient weight across all 3 emergent games!)
        adv_norm = torch.zeros_like(adv)
        for gm in game_masks:
            sub = adv[:, gm]
            adv_norm[:, gm] = (sub - sub.mean()) / (sub.std() + 1e-6)

        f_fine = torch.cat(buf_fine, dim=0)
        f_coarse = torch.cat(buf_coarse, dim=0)
        f_sf = torch.cat(buf_sf, dim=0)
        f_tok = torch.cat(buf_tok, dim=0)
        f_mask = torch.cat(buf_mask, dim=0)
        f_phi = torch.cat(buf_phi, dim=0)
        f_theta = torch.cat(buf_theta, dim=0)
        f_homeo = torch.cat(buf_homeo, dim=0)
        f_u = torch.cat(buf_u, dim=0)
        f_lp = torch.cat(buf_lp, dim=0)
        f_adv = adv_norm.reshape(-1)
        f_ret = returns.reshape(-1)

        policy.train()
        total_n = f_u.shape[0]
        frac = it / max(args.iters, 1)
        ent_coef = max(0.0006, 0.0045 * (1.0 - 0.85 * frac))
        max_log_std = -0.25 - 0.85 * frac

        last_loss = 0.0
        kl_stop = False
        for _ in range(args.ppo_epochs):
            if kl_stop:
                break
            perm = torch.randperm(total_n, device=device)
            for s in range(0, total_n, args.minibatch):
                mb = perm[s : s + args.minibatch]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out_mb = policy.forward_from_crops(
                        f_fine[mb],
                        f_coarse[mb],
                        f_sf[mb],
                        f_tok[mb],
                        token_mask=f_mask[mb],
                        phi_body=f_phi[mb],
                        theta_world=f_theta[mb],
                        homeostasis=f_homeo[mb],
                    )
                new_lp, ent = policy.evaluate_actions(out_mb, f_u[mb])
                log_ratio = (new_lp - f_lp[mb]).clamp(-5.0, 5.0)
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                if approx_kl > 0.055:
                    kl_stop = True
                    break
                surr1 = ratio * f_adv[mb]
                surr2 = ratio.clamp(0.80, 1.20) * f_adv[mb]
                pi_loss = -torch.minimum(surr1, surr2).mean()
                v_loss = F.smooth_l1_loss(out_mb["value"].float(), f_ret[mb])
                loss = pi_loss + 0.35 * v_loss - ent_coef * ent.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.70)
                optimizer.step()
                with torch.no_grad():
                    policy.actor_log_std.clamp_(-2.0, max_log_std)
                last_loss = float(loss.item())

        scheduler.step()

        # Per-game training rollout returns (unscaled)
        ep_ret_raw = rews.sum(dim=0) * 4.0
        g_rets = [float(ep_ret_raw[gm].mean().item()) for gm in game_masks]
        row: dict[str, Any] = {
            "iter": it,
            "loss": round(last_loss, 4),
            "g0_freeze_ret": round(g_rets[0], 3),
            "g1_flood_ret": round(g_rets[1], 3),
            "g2_heist_ret": round(g_rets[2], 3),
        }

        eval_every = 1 if args.smoke_test else 20
        if it in (1, 10) or it % eval_every == 0 or it == args.iters:
            ev_ep = 3 if args.smoke_test else min(12, args.eval_episodes)
            ev_g = [
                evaluate_policy_on_game(policy, env, g, ev_ep, base_seed=7100, mode="full")
                for g in range(3)
            ]
            # Balanced composite score across all 3 emergent games
            comp = ev_g[0]["mean_return"] + ev_g[1]["mean_return"] + 0.40 * ev_g[2]["mean_return"]
            row["eval_composite"] = round(comp, 3)
            row["eval_g0_surv"] = ev_g[0]["survival_rate"]
            row["eval_g1_surv"] = ev_g[1]["survival_rate"]
            row["eval_g2_disp"] = ev_g[2]["core_displacement_m"]
            if comp >= best_score:
                best_score = comp
                best_sd = copy.deepcopy(policy.state_dict())
            vram_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
            print(
                f"[Iter {it:3d}/{args.iters}] G0(Freeze)={ev_g[0]['mean_return']:+5.2f} (surv={ev_g[0]['survival_rate']*100:4.0f}%, hearth={ev_g[0]['hearth_ignitions']:4.1f}) "
                f"| G1(Flood)={ev_g[1]['mean_return']:+5.2f} (surv={ev_g[1]['survival_rate']*100:4.0f}%, cryo={ev_g[1]['cryo_bridges_forged']:4.1f}) "
                f"| G2(Heist)={ev_g[2]['mean_return']:+5.2f} (disp={ev_g[2]['core_displacement_m']:4.1f}m, deliv={ev_g[2]['core_deliveries']:.2f}, slide={ev_g[2]['oil_ice_slide_combos']:4.1f}) "
                f"| VRAM={vram_gb:.2f}GB | t={time.time()-t0:.1f}s",
                flush=True,
            )

        history.append(row)

    # Restore best checkpoint & run full held-out multi-game + ablation + zero-shot hybrid evaluation!
    policy.load_state_dict(best_sd)
    policy.eval()

    print("\n=== HELD-OUT MULTI-GAME & ZERO-SHOT MORPHOLOGY EVALUATION (UNSEEN SEEDS) ===", flush=True)
    eval_results = run_full_evaluation(policy, env, eval_episodes=args.eval_episodes, base_seed=9300)

    for gname in GAME_NAMES:
        pg = eval_results["per_game"][gname]
        rf = pg["universal_full"]
        rb = pg["morphology_blind_ablation"]
        rr = pg["random_3d"]
        print(
            f"  [{gname:20s}] Full={rf['mean_return']:+6.2f} | Morph-Blind(φ=0)={rb['mean_return']:+6.2f} "
            f"(Δ={pg['ablation_return_delta']:+5.2f}) | Random={rr['mean_return']:+6.2f} "
            f"| surv={rf['survival_rate']*100:5.1f}% | hearth={rf['hearth_ignitions']:.1f} | cryo={rf['cryo_bridges_forged']:.1f} "
            f"| disp={rf['core_displacement_m']:.1f}m | deliv={rf['core_deliveries']:.2f} | slide={rf['oil_ice_slide_combos']:.1f}",
            flush=True,
        )

    print("\n=== ZERO-SHOT UNSEEN HYBRID MORPHOLOGIES (STRICTLY HELD OUT FROM TRAINING) ===", flush=True)
    for h_name, h_data in eval_results["zero_shot_hybrids"].items():
        print(f"  Hybrid `{h_name}` φ={h_data['phi_body']}:", flush=True)
        for gname in GAME_NAMES:
            gh = h_data["games"][gname]
            rf = gh["zero_shot_full"]
            rb = gh["morphology_blind"]
            rr = gh["random_3d"]
            print(
                f"    {gname:20s}: ZeroShot={rf['mean_return']:+6.2f} vs Blind={rb['mean_return']:+6.2f} "
                f"(adapt={gh['morph_adaptation_gain']:+5.2f}) vs Rand={rr['mean_return']:+6.2f} "
                f"| surv={rf['survival_rate']*100:5.1f}% | flights={rf['aerial_flights']:.1f} | tunnels={rf['tunnels_bored']:.1f} "
                f"| cryo={rf['cryo_bridges_forged']:.1f} | disp={rf['core_displacement_m']:.1f}m",
                flush=True,
            )

    peak_vram_gb = round(
        (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0, 3
    )

    if not args.smoke_test:
        args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_sd,
                "config": {
                    "arch": "MorphologyActuatorNet",
                    "wave": "20B",
                    "params": n_params,
                    "z_world_ch": 48,
                    "z_eco_ch": 32,
                    "act_dim": ACT3D_DIM,
                    "morph_dim": MORPH_DIM,
                    "world_law_dim": WORLD_LAW_DIM,
                    "homeo_dim": HOMEO_DIM,
                    "self_dim": SELF3D_FEAT_DIM,
                    "tok_dim": TOKEN3D_FEAT_DIM,
                    "coarse_span_xy": float(args.grid_xy),
                    "world_latent_mode": world_enc.mode,
                    "held_out_morphologies": HELD_OUT_MORPHOLOGIES,
                },
                "evaluation": eval_results,
            },
            str(args.out_ckpt),
        )
        print(
            f"[Save] Saved checkpoint -> {args.out_ckpt} ({args.out_ckpt.stat().st_size / 1024:.1f} KB)",
            flush=True,
        )

        payload = {
            "model": "MorphologyActuatorNet",
            "wave": "20B",
            "params": n_params,
            "world_latent_mode": world_enc.mode,
            "device": str(device),
            "peak_vram_gb": peak_vram_gb,
            "train_seconds": round(time.time() - t0, 1),
            "grid": [args.grid_xy, args.grid_xy, args.depth_z],
            "num_arenas": args.num_arenas,
            "num_agents_per_arena": args.num_agents,
            "rollout_steps": args.rollout_steps,
            "iters": args.iters,
            "eval_episodes_per_game": args.eval_episodes,
            "morphology_names": list(MORPH_NAMES),
            "world_law_names": list(WORLD_LAW_NAMES),
            "game_theta_world": {
                gname: list(GAME_THETA_WORLD[gid]) for gid, gname in enumerate(GAME_NAMES)
            },
            "held_out_morphologies": {k: list(v) for k, v in HELD_OUT_MORPHOLOGIES.items()},
            "min_training_to_holdout_l2": round(env.min_holdout_l2_seen, 4),
            "evaluation": eval_results,
            "history": history,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[Save] Saved metrics -> {args.out_json}", flush=True)
        save_emergent_games_viz(eval_results, history, args.out_png, n_params=n_params)
    else:
        print(
            f"[Smoke Test] PASS in {time.time()-t0:.1f}s | params={n_params:,} | min_holdout_L2={env.min_holdout_l2_seen:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
