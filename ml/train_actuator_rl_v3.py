#!/usr/bin/env python3
"""
Wave 14A: True 3D Volumetric (Z=16) Pure-Physics Self-Play PPO Trainer & Held-Out Tournament
for Champion `ActuatorPolicyNetV3` (ZERO hardcoded spell buttons!).

Scientific Upgrades over Wave 13A:
1. Rich Multi-Archetype 3D Arena (`Physics3DArenaV3`, Z=16, 3 biomes, 6v6 = 12 bots):
   - All 4 physical archetypes spawn & operate in their natural 3D terrain:
     * `0=Strider` (Heavy Walker / Squad Anchor)
     * `1=Skywing` (True 3D Flyer + persistent geothermal/wildfire thermal updraft columns -> aerial high-ground dive hits!)
     * `2=Tidekin` (Amphibious 2.2x+ Swimmer starting along the submerged river canyon -> aquatic ambushes!)
     * `3=Emberstalker` (Solo 3.6m Cliff Climber + smoke-veil speed & stealth)
   - True multi-level 3D terrain: vaulted caves, 2 combustible timber/stone arched bridges, deep river/magma
     channel (water at walkable riverbed z=3..4), oil pockets, persistent geothermal vents.
   - Fully vectorized observation builder (10x faster GPU rollouts, exact `SELF3D_FEAT_DIM=22`, `TOKEN3D_FEAT_DIM=14`).
   - True 50/50 West+East side-balanced training inside EVERY single batch!
2. Phase 1 — 12-Step Pure-Spatial Navigation & Aiming Warmstart (NO spell rules!):
   - Supervises ONLY the 6 spatial geometry channels (`move_x, move_y, move_jump, aim_dx, aim_dy, aim_dz`)
     to navigate toward unowned shrines / deliver relic and point the 3D aim ray at the nearest enemy.
   - Leaves ALL 6 physical actuator channels (`thermal_flux, emit_water, emit_oil, emit_smoke, focus_radius,
     impulse_or_dome`) 100% UNSUPERVISED (ZERO imitation) so every physical ability & combo emerges purely from RL!
3. Phase 2 — 110-Iteration Pure-Physics PPO League Self-Play (`rollout_steps=64`, `num_arenas=24`, frozen `WorldModel3D-v5`):
   - Alternates 50% vs full-strength `Scripted-3D` + 50% vs League Self-Play Snapshot Pool.
   - Side-balanced 50/50 West & East in every rollout AND every held-out evaluation.
   - Tracks best checkpoint by `(vs Scripted-3D WR, mean_score_diff)`.

Evaluates on 24 held-out 3D matches (12 West + 12 East side-swapped) on unseen seeds vs:
  - `Scripted-3D` (strong tactical 3D baseline)
  - `Flat-2D-Ablation` (`ActuatorPolicyNetV3` with `aim_dz=0` and `impulse_or_dome>=0` forced flat-2D)
  - `Random-3D`

Saves:
  - `models/checkpoints/actuator_policy_v3_rl.pt`
  - `models/ACTUATOR_V3_TOURNAMENT.json`
  - `models/actuator_v3_viz.png`
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
    ActuatorPolicyNetV3,
    count_parameters,
)

# Exact FIELD_CONTRACT species & channel indices
SP_WATER, SP_ICE, SP_STEAM, SP_SOIL, SP_ROCK, SP_WOOD, SP_ASH, SP_SMOKE, SP_OIL, SP_PLANT = range(10)
CH_TEMP = 10
CH_LOGMASS = 11
CH_FLOW = 13
CH_ALT = 14
CH_OBST = 15

NUM_BOTS = 12
AIM_RANGE_XY = 11.0
AIM_RANGE_Z = 6.0


class FrozenWorldLatentExtractor(nn.Module):
    """
    Wraps `WorldModel3DV5` (preferred), `WorldModelDT`, or deterministic 3D projection
    for fast CPU smoke tests to produce `[A, 48, L, H, W]` 3D shared world latents.
    """

    def __init__(self, ckpt_path: Path | None, device: torch.device):
        super().__init__()
        self.device = device
        self.mode = "fallback_3d"
        self.model: nn.Module | None = None

        if ckpt_path is not None and ckpt_path.exists():
            ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            try:
                from world_model_3d_v5 import WorldModel3DV5

                wm = WorldModel3DV5(in_ch=18, out_ch=17, width=48, z_flux_ch=5).to(device).eval()
                wm.load_state_dict(sd, strict=False)
                for p in wm.parameters():
                    p.requires_grad_(False)
                self.model = wm
                self.mode = "world_model_3d_v5"
                print(f"[WorldLatent] Loaded frozen WorldModel3DV5 from {ckpt_path}", flush=True)
                return
            except Exception as exc:
                print(f"[WorldLatent] WorldModel3DV5 load skipped ({exc}); trying WorldModelDT...", flush=True)

            try:
                from world_model_dt import WorldModelDT

                wm = WorldModelDT(in_ch=18, width=48, num_res=4).to(device).eval()
                wm.load_state_dict(sd, strict=False)
                for p in wm.parameters():
                    p.requires_grad_(False)
                self.model = wm
                self.mode = "world_model_dt"
                print(f"[WorldLatent] Loaded frozen WorldModelDT from {ckpt_path}", flush=True)
                return
            except Exception as exc:
                print(f"[WorldLatent] WorldModelDT load skipped ({exc}); using lightweight 3D encoder.", flush=True)

        torch.manual_seed(1337)
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
        A, C, L, H, W = w_state.shape
        use_fp16 = self.device.type == "cuda"
        if self.mode == "world_model_3d_v5" and self.model is not None:
            x_bl = w_state.permute(0, 2, 1, 3, 4).contiguous()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                _, _, z_bl = self.model(x_bl, local_dt=1.0, return_z_flux=True, return_latent=True)
            z = z_bl.permute(0, 2, 1, 3, 4).contiguous()
            return z.half() if use_fp16 else z.float()

        if self.mode == "world_model_dt" and self.model is not None:
            x = w_state.permute(0, 2, 1, 3, 4).reshape(A * L, C, H, W)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                h = self.model.stem(x)
                for blk in self.model.blocks:
                    h = blk(h)
                z = F.gelu(h)
            z = z.reshape(A, L, 48, H, W).permute(0, 2, 1, 3, 4).contiguous()
            return z.half() if use_fp16 else z.float()

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            z = self.fallback_conv(w_state)
        return z.half() if use_fp16 else z.float()


class Physics3DArenaV3:
    """
    True 3D Volumetric (Z=16) Multi-Biome 6v6 Vectorized GPU Arena.
    Coordinates: X in [0, W-1], Y in [0, H-1], Z in [0, L-1].
    """

    def __init__(
        self,
        num_arenas: int,
        world_encoder: FrozenWorldLatentExtractor,
        device: torch.device,
        grid_xy: int = 64,
        depth_z: int = 16,
        max_ticks: int = 64,
        latent_refresh_every: int = 4,
    ):
        self.A = num_arenas
        self.N = NUM_BOTS
        self.H = grid_xy
        self.W = grid_xy
        self.L = depth_z
        self.max_ticks = max_ticks
        self.refresh_every = latent_refresh_every
        self.device = device
        self.world_encoder = world_encoder

        # 4 Physically Distinct Archetypes per team (6v6 = 12 bots, Squads vs Solos):
        #   0: Strider (Heavy Walker / Squad Anchor, high HP & knockback resist)
        #   1: Skywing (True 3D Flyer / Thermal Glider — rides geothermal/fire/steam updrafts!)
        #   2: Tidekin (Amphibious Swimmer / Diver — 2.2x+ water speed, submerged river ambushes!)
        #   3: Emberstalker (Solo Cavern Climber — 3.6m vertical wall-climb, smoke stealth!)
        teams = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        classes = torch.tensor([0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 3], dtype=torch.long, device=device)
        squads = torch.tensor([1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 0], dtype=torch.long, device=device)
        self.team = teams.unsqueeze(0).expand(self.A, -1).contiguous()
        self.cls = classes.unsqueeze(0).expand(self.A, -1).contiguous()
        self.squad = squads.unsqueeze(0).expand(self.A, -1).contiguous()
        self.is_solo = self.squad == 0
        self.same_squad_ally = (
            (self.team[:, :, None] == self.team[:, None, :])
            & (self.squad[:, :, None] == self.squad[:, None, :])
            & (self.squad[:, :, None] > 0)
            & (~torch.eye(self.N, dtype=torch.bool, device=device).unsqueeze(0))
        )
        # 3-lane assignment (-1=North Bridge, 0=Center Cave/River, +1=South Bridge)
        lanes = torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0], device=device)
        self.lane_id = lanes.unsqueeze(0).expand(self.A, -1).contiguous()
        self.lane_shrine_idx = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], dtype=torch.long, device=device)
        self.arena_idx_flat = torch.arange(self.A, device=device).repeat_interleave(self.N)
        self.a_idx = torch.arange(self.A, device=device).unsqueeze(1).expand(self.A, self.N).contiguous()
        self.bot_idx_row = torch.arange(self.N, device=device).unsqueeze(0)
        self.enemy_mask = self.team[:, :, None] != self.team[:, None, :]

        # Precomputed enemy [12, 6] and ally [12, 4] gather tables for zero-loop observation building
        e_tbl = []
        a_tbl = []
        for i in range(self.N):
            my_t = 0 if i < 6 else 1
            e_tbl.append([j for j in range(self.N) if (0 if j < 6 else 1) != my_t])
            a_tbl.append([j for j in range(self.N) if (0 if j < 6 else 1) == my_t and j != i][:4])
        self.enemy_gather = torch.tensor(e_tbl, dtype=torch.long, device=device)  # [N, 6]
        self.ally_gather = torch.tensor(a_tbl, dtype=torch.long, device=device)   # [N, 4]

        # Normalized coordinates for procedural 3D biome generation
        zz = torch.arange(self.L, device=device, dtype=torch.float32)
        yy = torch.arange(self.H, device=device, dtype=torch.float32) / self.H
        xx = torch.arange(self.W, device=device, dtype=torch.float32) / self.W
        self.gz, self.gy, self.gx = torch.meshgrid(zz, yy, xx, indexing="ij")  # [L, H, W]

        # Persistent geothermal vent mask (under & beside North/South bridges -> Skywing thermal updraft columns!)
        self.vent_mask = (
            (((self.gx - 0.35).abs() < 0.065) | ((self.gx - 0.65).abs() < 0.065))
            & (((self.gy - 0.24).abs() < 0.075) | ((self.gy - 0.76).abs() < 0.075))
            & (self.gz >= 2.0)
            & (self.gz <= 10.0)
        ).unsqueeze(0)  # [1, L, H, W]

        # 3D shrines: North Bridge (z=8.5), Center River/Cave (z=3.0), South Timber Bridge (z=8.5)
        self.shrine_x = torch.tensor([0.50, 0.50, 0.50], device=device).unsqueeze(0).expand(self.A, -1).clone() * self.W
        self.shrine_y = torch.tensor([0.24, 0.50, 0.76], device=device).unsqueeze(0).expand(self.A, -1).clone() * self.H
        self.shrine_z = torch.tensor([8.5, 3.0, 8.5], device=device).unsqueeze(0).expand(self.A, -1).clone()

        # Voxel state [A, 18, L, H, W] and neural latent [A, 48, L, H, W]
        self.w_state = torch.zeros((self.A, 18, self.L, self.H, self.W), dtype=torch.float32, device=device)
        z_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.z_world = torch.zeros((self.A, 48, self.L, self.H, self.W), dtype=z_dtype, device=device)

        # Active Stasis TimeDomes (1 slot per bot: x, y, z, ttl)
        self.dome_x = torch.zeros((self.A, self.N), device=device)
        self.dome_y = torch.zeros((self.A, self.N), device=device)
        self.dome_z = torch.zeros((self.A, self.N), device=device)
        self.dome_ttl = torch.zeros((self.A, self.N), dtype=torch.int32, device=device)

        # 3D Gaussian deposit kernel offsets (3x5x5 = 75 offsets covering +-1.5m Z and +-2.5m XY)
        oz = torch.arange(-1, 2, device=device)
        oy = torch.arange(-2, 3, device=device)
        ox = torch.arange(-2, 3, device=device)
        dz, dy, dx = torch.meshgrid(oz, oy, ox, indexing="ij")
        self.st_dz = dz.reshape(-1)
        self.st_dy = dy.reshape(-1)
        self.st_dx = dx.reshape(-1)
        self.st_r2 = self.st_dx.float() ** 2 + self.st_dy.float() ** 2 + (self.st_dz.float() * 1.4) ** 2

        self.reset(seed=0)

    def _generate_3d_biomes(self, gen: torch.Generator) -> None:
        """Vectorized GPU generation of A true-3D Z=16 multi-biome worlds (caves + arched bridges + deep river)."""
        A, L, H, W = self.A, self.L, self.H, self.W
        dev = self.device
        self.w_state.zero_()

        gx = self.gx.unsqueeze(0)  # [1, L, H, W]
        gy = self.gy.unsqueeze(0)
        gz = self.gz.unsqueeze(0)

        biome_id = torch.randint(0, 3, (A, 1, 1, 1), generator=gen, device=dev)
        self.biome_id = biome_id.view(A)

        phase = torch.rand((A, 1, 1, 1), generator=gen, device=dev) * 6.28318
        rcx = 0.50 + 0.035 * torch.sin(gy * 6.28318 + phase)
        dist_river = (gx - rcx).abs()

        # Sculpted canyon: deep river channel floor z=2.0 in center (walkable z=3.0 submerged in water!),
        # rising smoothly to highland shelves z=7.5..9.5 on West/East.
        base_h = 2.0 + 6.2 * ((dist_river - 0.11) / 0.22).clamp(0.0, 1.0)
        ridge = 2.4 * torch.sin(gy * 9.4 + phase).abs() * ((gx - 0.5).abs() > 0.34).float()
        terrain_z = (base_h + ridge).clamp(2.0, 11.2)

        solid_ground = gz <= terrain_z
        bedrock = gz <= 1.2
        self.w_state[:, SP_ROCK] = torch.where(bedrock | (solid_ground & (gz < terrain_z - 1.1)), 1.0, 0.0)
        self.w_state[:, SP_SOIL] = torch.where(solid_ground & (~bedrock) & (gz >= terrain_z - 1.1), 0.92, 0.0)

        # Vaulted subterranean basalt cave tunnel running East-West at y ~ 0.50, z=2.5..5.2
        cave_mask = (
            ((gy - 0.50).abs() < 0.095)
            & (gz >= 2.5)
            & (gz <= 5.2 - 1.7 * ((gy - 0.50) / 0.095) ** 2)
            & ((gx - 0.50).abs() < 0.34)
        )
        self.w_state[:, SP_ROCK] = torch.where(cave_mask, 0.0, self.w_state[:, SP_ROCK])
        self.w_state[:, SP_SOIL] = torch.where(cave_mask, 0.0, self.w_state[:, SP_SOIL])

        # Wide submerged River/Flooded Canyon channel (dist_river < 0.14, gz in 2.5..4.2 -> walkable z=3.0 is IN WATER!)
        # In Biome 1 (Volcanic Caldera), outer edges are thermal hot springs / river + magma center!
        river_water = (dist_river < 0.14) & (gz >= 2.5) & (gz <= 4.2)
        magma_core = (biome_id == 1) & (dist_river < 0.05) & (gz >= 2.5) & (gz <= 3.5)
        self.w_state[:, SP_WATER] = torch.where(river_water & (~magma_core), 1.10, 0.0)
        self.w_state[:, CH_TEMP] = torch.where(magma_core, 2.8, 0.25)

        # Biome 2 (Glacial Sinkhole): cold ambient + partial ice ledges along outer river
        self.w_state[:, CH_TEMP] = torch.where(biome_id == 2, self.w_state[:, CH_TEMP] - 0.50, self.w_state[:, CH_TEMP])
        ice_plate = (biome_id == 2) & (dist_river < 0.12) & ((gz - 4.0).abs() < 0.55) & ((gy - 0.50).abs() > 0.18)
        self.w_state[:, SP_ICE] = torch.where(ice_plate, 0.92, 0.0)

        # True 3D Arched Bridges spanning the canyon at y=0.24 (North) and y=0.76 (South Timber)!
        arch_curve = 8.2 - 1.4 * ((gx - 0.50) / 0.22).clamp(-1.0, 1.0) ** 2
        north_bridge = ((gy - 0.24).abs() < 0.058) & ((gx - 0.50).abs() < 0.23) & ((gz - arch_curve).abs() <= 0.65)
        south_bridge = ((gy - 0.76).abs() < 0.058) & ((gx - 0.50).abs() < 0.23) & ((gz - arch_curve).abs() <= 0.65)
        under_north = ((gy - 0.24).abs() < 0.058) & ((gx - 0.50).abs() < 0.17) & (gz >= 3.2) & (gz < arch_curve - 0.65)
        under_south = ((gy - 0.76).abs() < 0.058) & ((gx - 0.50).abs() < 0.17) & (gz >= 3.2) & (gz < arch_curve - 0.65)
        self.w_state[:, SP_ROCK] = torch.where(under_north | under_south, 0.0, self.w_state[:, SP_ROCK])
        self.w_state[:, SP_SOIL] = torch.where(under_north | under_south, 0.0, self.w_state[:, SP_SOIL])

        # Combustible timber bridge decks (can be burned/collapsed -> enemies drop into river/canyon!)
        self.w_state[:, SP_WOOD] = torch.where(north_bridge | south_bridge, 0.92, 0.0)
        self.w_state[:, SP_ROCK] = torch.where(north_bridge & ((gx - 0.50).abs() > 0.12), 0.75, self.w_state[:, SP_ROCK])

        # Symmetric oil pockets near cave mouths & bridge approaches
        oil_pockets = (
            (((gx - 0.36).abs() < 0.05) | ((gx - 0.64).abs() < 0.05))
            & ((gy - 0.50).abs() < 0.07)
            & ((gz - 3.0).abs() < 0.9)
        )
        self.w_state[:, SP_OIL] = torch.where(oil_pockets, 0.90, 0.0)

        # Persistent geothermal updraft plumes near North & South bridge approaches
        self.w_state[:, CH_TEMP] = torch.where(self.vent_mask, torch.maximum(self.w_state[:, CH_TEMP], torch.tensor(2.1, device=dev)), self.w_state[:, CH_TEMP])
        self.w_state[:, SP_STEAM] = torch.where(self.vent_mask & (gz >= 4.0), 0.60, self.w_state[:, SP_STEAM])

        self.w_state[:, CH_ALT] = gz / float(L - 1)
        self.w_state[:, CH_LOGMASS] = torch.log1p(self.w_state[:, :10].sum(dim=1))

    def _solid_mask(self) -> torch.Tensor:
        """[A, L, H, W] bool where voxel supports standing weight (rock, soil, wood, ice)."""
        return (
            self.w_state[:, SP_ROCK]
            + self.w_state[:, SP_SOIL]
            + self.w_state[:, SP_WOOD]
            + self.w_state[:, SP_ICE]
        ) > 0.42

    def _ground_z_near(self, x: torch.Tensor, y: torch.Tensor, cur_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """True 3D multi-level ground lookup (supports walking under bridges AND falling when bridges collapse!)."""
        ix = x.round().long().clamp(0, self.W - 1)
        iy = y.round().long().clamp(0, self.H - 1)
        solid = self._solid_mask()  # [A, L, H, W]
        col = solid[self.a_idx, :, iy, ix]  # [A, N, L] bool

        z_grid = torch.arange(self.L, device=self.device, dtype=torch.float32).view(1, 1, self.L)
        below_solid = F.pad(col[:, :, :-1], (1, 0), value=True)
        walkable = below_solid & (~col)

        reachable = walkable & (z_grid <= cur_z.unsqueeze(-1) + 1.8)
        score = torch.where(reachable, z_grid, torch.full_like(z_grid, -1.0))
        best_z = score.max(dim=-1).values
        any_walk = torch.where(walkable, z_grid, torch.full_like(z_grid, 99.0)).min(dim=-1).values
        target_z = torch.where(best_z >= 0.0, best_z, any_walk.clamp(1.0, self.L - 2.0))
        fell = (cur_z - target_z).clamp(min=0.0)
        return target_z, fell

    def reset(self, seed: int = 0) -> None:
        dev = self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        self.gen = gen
        self.tick = 0

        self._generate_3d_biomes(gen)
        self.z_world = self.world_encoder.encode(self.w_state)

        # Archetype-aligned 3D spawns (symmetric West & East):
        #   0=Strider (North bridge approach), 1=Skywing (North geothermal vent!),
        #   2=Tidekin (Center submerged river bank!), 3=Strider (Center cave mouth),
        #   4=Skywing (South geothermal vent!), 5=Emberstalker (South cliff/bridge raider)
        lane_x = torch.tensor([0.24, 0.34, 0.39, 0.27, 0.34, 0.24], device=dev) * self.W
        lane_y = torch.tensor([0.24, 0.24, 0.50, 0.50, 0.76, 0.76], device=dev) * self.H
        lane_z = torch.tensor([8.5, 10.5, 3.0, 3.5, 10.5, 8.5], device=dev)

        jx = torch.empty((self.A, 6), device=dev).uniform_(-2.2, 2.2, generator=gen)
        jy = torch.empty((self.A, 6), device=dev).uniform_(-1.6, 1.6, generator=gen)

        wx = (lane_x.unsqueeze(0) + jx).clamp(4.0, self.W * 0.44)
        ex = (self.W - 1.0) - wx
        wy = (lane_y.unsqueeze(0) + jy).clamp(4.0, self.H - 5.0)
        ey = wy.clone()

        self.pos_x = torch.cat([wx, ex], dim=1)
        self.pos_y = torch.cat([wy, ey], dim=1)
        init_z = torch.cat([lane_z.unsqueeze(0).expand(self.A, -1), lane_z.unsqueeze(0).expand(self.A, -1)], dim=1).clone()
        tz, _ = self._ground_z_near(self.pos_x, self.pos_y, init_z)
        # Keep Skywings airborne at high altitude on spawn
        self.pos_z = torch.where(self.cls == 1, torch.maximum(tz + 2.2, init_z).clamp(max=float(self.L - 2)), tz)

        self.vel_x = torch.zeros((self.A, self.N), device=dev)
        self.vel_y = torch.zeros((self.A, self.N), device=dev)
        self.hp = torch.ones((self.A, self.N), device=dev)
        self.mana = torch.ones((self.A, self.N), device=dev)
        self.alive = torch.ones((self.A, self.N), dtype=torch.bool, device=dev)
        self.respawn_cd = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.last_attacker = torch.full((self.A, self.N), -1, dtype=torch.long, device=dev)
        self.last_attacker_dz = torch.zeros((self.A, self.N), device=dev)

        self.dome_ttl.zero_()
        self.shrine_prog = torch.zeros((self.A, 3), device=dev)
        self.shrine_owner = torch.full((self.A, 3), -1, dtype=torch.long, device=dev)
        self.relic_x = torch.full((self.A,), 0.50 * self.W, device=dev)
        self.relic_y = torch.full((self.A,), 0.50 * self.H, device=dev)
        self.relic_z = torch.full((self.A,), 3.0, device=dev)
        self.relic_carrier = torch.full((self.A,), -1, dtype=torch.long, device=dev)

        self.team_score = torch.zeros((self.A, 2), device=dev)
        self.team_kills = torch.zeros((self.A, 2), device=dev)

        # Emergent 3D tactics + heterogeneous archetype telemetry counters [A, 2]
        self.stat_vert_hits = torch.zeros((self.A, 2), device=dev)
        self.stat_bridge_drops = torch.zeros((self.A, 2), device=dev)
        self.stat_cryo_seals = torch.zeros((self.A, 2), device=dev)
        self.stat_stasis_domes = torch.zeros((self.A, 2), device=dev)
        self.stat_pressure_blasts = torch.zeros((self.A, 2), device=dev)
        self.stat_updraft_soars = torch.zeros((self.A, 2), device=dev)
        self.stat_aquatic_ambushes = torch.zeros((self.A, 2), device=dev)
        self.stat_squad_combos = torch.zeros((self.A, 2), device=dev)
        self.oil_primed_ttl = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.oil_primed_by = torch.full((self.A, self.N), -1, dtype=torch.long, device=dev)

    def build_observations(self) -> dict[str, torch.Tensor]:
        """Fully vectorized (zero Python loop) observation builder for ActuatorPolicyNetV3."""
        A, N = self.A, self.N
        dev = self.device
        a_idx = self.a_idx

        ix = self.pos_x.round().long().clamp(0, self.W - 1)
        iy = self.pos_y.round().long().clamp(0, self.H - 1)
        iz = self.pos_z.round().long().clamp(0, self.L - 1)
        iz_above = (iz + 2).clamp(0, self.L - 1)
        iz_below = (iz - 1).clamp(0, self.L - 1)

        local_temp = self.w_state[a_idx, CH_TEMP, iz, iy, ix]
        local_water = torch.maximum(
            self.w_state[a_idx, SP_WATER, iz, iy, ix],
            self.w_state[a_idx, SP_WATER, iz_below, iy, ix],
        )
        local_ice = self.w_state[a_idx, SP_ICE, iz, iy, ix]
        local_smoke = self.w_state[a_idx, SP_SMOKE, iz, iy, ix]
        local_oil = self.w_state[a_idx, SP_OIL, iz, iy, ix]
        above_solid = (
            self.w_state[a_idx, SP_ROCK, iz_above, iy, ix]
            + self.w_state[a_idx, SP_SOIL, iz_above, iy, ix]
            + self.w_state[a_idx, SP_WOOD, iz_above, iy, ix]
        ).clamp(0.0, 1.0)

        team_sign = torch.where(self.team == 0, 1.0, -1.0)
        is_carrier = (self.relic_carrier.unsqueeze(1) == self.bot_idx_row).float()
        cls_oh = F.one_hot(self.cls, num_classes=4).float()

        dome_ready = (self.dome_ttl == 0).float()

        self_feat = torch.stack(
            (
                self.pos_x / self.W * 2.0 - 1.0,
                self.pos_y / self.H * 2.0 - 1.0,
                self.pos_z / max(self.L - 1, 1) * 2.0 - 1.0,
                self.vel_x * 0.25,
                self.vel_y * 0.25,
                self.hp,
                self.mana,
                self.alive.float(),
                team_sign,
                is_carrier,
                local_temp * 0.35,
                local_water,
                local_ice,
                local_smoke,
                local_oil,
                self.lane_id,
                above_solid,
                dome_ready,
            ),
            dim=-1,
        )  # [A, N, 18]
        self_feat = torch.cat((self_feat, cls_oh), dim=-1)  # [A, N, 22]

        # Vectorized 14 tokens: [0..5] 6 enemies, [6..9] 4 allies, [10..12] 3 shrines, [13] relic
        toks = torch.zeros((A, N, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM), device=dev)
        mask = torch.zeros((A, N, MAX_TOKENS_3D), dtype=torch.bool, device=dev)

        px = self.pos_x.unsqueeze(2)  # [A, N, 1]
        py = self.pos_y.unsqueeze(2)
        pz = self.pos_z.unsqueeze(2)

        # Enemies [A, N, 6]
        eg = self.enemy_gather  # [N, 6]
        ex = self.pos_x[:, eg]  # [A, N, 6]
        ey = self.pos_y[:, eg]
        ez = self.pos_z[:, eg]
        edx = ((ex - px) / 24.0).clamp(-2.0, 2.0)
        edy = ((ey - py) / 24.0).clamp(-2.0, 2.0)
        edz = ((ez - pz) / 8.0).clamp(-2.0, 2.0)
        edist = torch.sqrt(edx * edx + edy * edy + edz * edz + 1e-6).clamp(0.0, 3.0)
        toks[:, :, 0:6, 0] = edx
        toks[:, :, 0:6, 1] = edy
        toks[:, :, 0:6, 2] = edz
        toks[:, :, 0:6, 3] = edist
        toks[:, :, 0:6, 4] = self.hp[:, eg]
        toks[:, :, 0:6, 5] = 1.0
        toks[:, :, 0:6, 6] = (self.relic_carrier[:, None, None] == eg[None, :, :]).float()
        toks[:, :, 0:6, 7] = (self.dome_ttl[:, eg] > 0).float()
        toks[:, :, 0:6, 8] = (ez > pz + 1.5).float()
        toks[:, :, 0:6, 9] = (ez < pz - 1.5).float()
        mask[:, :, 0:6] = self.alive[:, eg] & (local_smoke.unsqueeze(2) < 0.75)

        # Allies [A, N, 4]
        ag = self.ally_gather  # [N, 4]
        ax = self.pos_x[:, ag]
        ay = self.pos_y[:, ag]
        az = self.pos_z[:, ag]
        adx = ((ax - px) / 24.0).clamp(-2.0, 2.0)
        ady = ((ay - py) / 24.0).clamp(-2.0, 2.0)
        adz = ((az - pz) / 8.0).clamp(-2.0, 2.0)
        adist = torch.sqrt(adx * adx + ady * ady + adz * adz + 1e-6).clamp(0.0, 3.0)
        toks[:, :, 6:10, 0] = adx
        toks[:, :, 6:10, 1] = ady
        toks[:, :, 6:10, 2] = adz
        toks[:, :, 6:10, 3] = adist
        toks[:, :, 6:10, 4] = self.hp[:, ag]
        toks[:, :, 6:10, 10] = 1.0
        mask[:, :, 6:10] = self.alive[:, ag]

        # 3 Shrines [A, N, 3]
        sdx = ((self.shrine_x[:, None, :] - px) / 24.0).clamp(-2.0, 2.0)
        sdy = ((self.shrine_y[:, None, :] - py) / 24.0).clamp(-2.0, 2.0)
        sdz = ((self.shrine_z[:, None, :] - pz) / 8.0).clamp(-2.0, 2.0)
        sdist = torch.sqrt(sdx * sdx + sdy * sdy + sdz * sdz + 1e-6).clamp(0.0, 3.0)
        my_t = self.team[:, :, None]  # [A, N, 1]
        s_own = self.shrine_owner[:, None, :]  # [A, 1, 3]
        own_rel = torch.where(s_own == my_t, 1.0, torch.where(s_own == (1 - my_t), -1.0, 0.0))
        toks[:, :, 10:13, 0] = sdx
        toks[:, :, 10:13, 1] = sdy
        toks[:, :, 10:13, 2] = sdz
        toks[:, :, 10:13, 3] = sdist
        toks[:, :, 10:13, 11] = 1.0
        toks[:, :, 10:13, 12] = own_rel
        mask[:, :, 10:13] = True

        # Relic [A, N, 1] (col 4 = relic free; col 7 = self Stasis Dome ready flag so bot knows cooldown!)
        rdx = ((self.relic_x[:, None] - self.pos_x) / 24.0).clamp(-2.0, 2.0)
        rdy = ((self.relic_y[:, None] - self.pos_y) / 24.0).clamp(-2.0, 2.0)
        rdz = ((self.relic_z[:, None] - self.pos_z) / 8.0).clamp(-2.0, 2.0)
        rdist = torch.sqrt(rdx * rdx + rdy * rdy + rdz * rdz + 1e-6).clamp(0.0, 3.0)
        toks[:, :, 13, 0] = rdx
        toks[:, :, 13, 1] = rdy
        toks[:, :, 13, 2] = rdz
        toks[:, :, 13, 3] = rdist
        toks[:, :, 13, 4] = (self.relic_carrier[:, None] < 0).float()
        toks[:, :, 13, 7] = (self.dome_ttl == 0).float()
        toks[:, :, 13, 13] = 1.0
        mask[:, :, 13] = True

        pos_xyz = torch.stack((self.pos_x, self.pos_y, self.pos_z), dim=-1).reshape(A * N, 3)
        return {
            "pos_xyz": pos_xyz,
            "self_feat": self_feat.reshape(A * N, SELF3D_FEAT_DIM),
            "tokens": toks.reshape(A * N, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM),
            "token_mask": mask.reshape(A * N, MAX_TOKENS_3D),
        }

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """
        actions: [A, N, 12] bounded continuous 3D actuators.
        Returns (rewards [A, N], done).
        """
        A, N = self.A, self.N
        dev = self.device
        a_idx = self.a_idx
        self.tick += 1
        rewards = torch.zeros((A, N), device=dev)
        prev_team_score = self.team_score.clone()

        # Decrement TimeDomes & respawn timers
        self.dome_ttl = (self.dome_ttl - 1).clamp(min=0)
        just_respawn = (~self.alive) & (self.respawn_cd <= 1)
        self.respawn_cd = (self.respawn_cd - 1).clamp(min=0)
        if just_respawn.any():
            self.alive = self.alive | just_respawn
            self.hp = torch.where(just_respawn, torch.ones_like(self.hp), self.hp)
            self.mana = torch.where(just_respawn, torch.ones_like(self.mana), self.mana)
            sx = torch.where(
                self.cls == 2,
                torch.where(self.team == 0, torch.full_like(self.pos_x, 0.39 * self.W), torch.full_like(self.pos_x, 0.61 * self.W)),
                torch.where(self.team == 0, torch.full_like(self.pos_x, 0.24 * self.W), torch.full_like(self.pos_x, 0.76 * self.W)),
            )
            self.pos_x = torch.where(just_respawn, sx, self.pos_x)
            ref_z = torch.where(self.cls == 2, torch.full_like(self.pos_z, 3.0), torch.full_like(self.pos_z, 8.5))
            tz, _ = self._ground_z_near(self.pos_x, self.pos_y, ref_z)
            tz = torch.where(self.cls == 1, (tz + 2.2).clamp(max=float(self.L - 2)), tz)
            self.pos_z = torch.where(just_respawn, tz, self.pos_z)

        alive_f = self.alive.float()

        # Check Stasis TimeDome dilation per bot (0.25x speed inside an ENEMY Stasis dome!)
        d2_domes = (
            (self.pos_x[:, :, None] - self.dome_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.dome_y[:, None, :]) ** 2
            + (self.pos_z[:, :, None] - self.dome_z[:, None, :]) ** 2
        )
        in_enemy_dome = ((self.dome_ttl[:, None, :] > 0) & self.enemy_mask & (d2_domes < 52.0)).any(dim=-1)
        time_scale = torch.where(in_enemy_dome, 0.25, 1.0)

        # 1. Heterogeneous 3D Archetype Locomotion
        ix = self.pos_x.round().long().clamp(0, self.W - 1)
        iy = self.pos_y.round().long().clamp(0, self.H - 1)
        iz = self.pos_z.round().long().clamp(0, self.L - 1)
        iz_m1 = (iz - 1).clamp(0, self.L - 1)
        iz_m2 = (iz - 2).clamp(0, self.L - 1)
        iz_m3 = (iz - 3).clamp(0, self.L - 1)

        is_skywing = self.cls == 1
        is_tidekin = self.cls == 2
        is_stalker = self.cls == 3

        water_here = torch.maximum(self.w_state[a_idx, SP_WATER, iz, iy, ix], self.w_state[a_idx, SP_WATER, iz_m1, iy, ix])
        in_water = (water_here > 0.35) & (self.w_state[a_idx, SP_ICE, iz, iy, ix] < 0.40)
        on_ice = self.w_state[a_idx, SP_ICE, iz_m1, iy, ix] > 0.40
        in_smoke = self.w_state[a_idx, SP_SMOKE, iz, iy, ix] > 0.18
        col_temp_below = torch.maximum(
            torch.maximum(self.w_state[a_idx, CH_TEMP, iz_m1, iy, ix], self.w_state[a_idx, CH_TEMP, iz_m2, iy, ix]),
            self.w_state[a_idx, CH_TEMP, iz_m3, iy, ix],
        )
        col_steam_here = torch.maximum(self.w_state[a_idx, SP_STEAM, iz, iy, ix], self.w_state[a_idx, SP_STEAM, iz_m2, iy, ix])

        base_spd = torch.where(
            is_tidekin & in_water,
            3.05,  # Passive Hydrodynamic Surge: 2.2x+ swim boost in rivers/flooded caverns!
            torch.where(
                is_skywing,
                2.50,
                torch.where(
                    is_stalker,
                    torch.where(in_smoke, 2.80, 2.35),
                    torch.where(in_water, 1.05, torch.where(on_ice, 2.55, 1.95)),
                ),
            ),
        )
        speed = base_spd * time_scale * alive_f

        mv_x = actions[:, :, 0]
        mv_y = actions[:, :, 1]
        mv_jump = actions[:, :, 2]
        mv_norm = torch.hypot(mv_x, mv_y).clamp(min=1.0)
        self.vel_x = (mv_x / mv_norm) * speed
        self.vel_y = (mv_y / mv_norm) * speed

        new_x = (self.pos_x + self.vel_x).clamp(2.0, self.W - 3.0)
        new_y = (self.pos_y + self.vel_y).clamp(2.0, self.H - 3.0)

        climb_cap = torch.where(is_stalker, 3.6, 1.8)
        climb_boost = torch.where((mv_jump > 0.30) & self.alive, torch.where(is_stalker, 2.4, 1.5), 0.0)
        gnd_z, gnd_fell = self._ground_z_near(new_x, new_y, self.pos_z + climb_boost)

        # Skywing true 3D flight + passive thermal updraft soaring over geothermal vents/fire/steam!
        updraft = ((col_temp_below > 1.00) | (col_steam_here > 0.22)) & is_skywing & self.alive
        fly_dz = mv_jump * 1.60 + torch.where(updraft, 1.65, -0.35)
        fly_z = torch.maximum(self.pos_z + fly_dz, gnd_z).clamp(max=float(self.L - 2))
        if updraft.any():
            for t in (0, 1):
                self.stat_updraft_soars[:, t] += (updraft & (self.team == t)).float().sum(dim=1) * 0.25
            rewards += updraft.float() * 0.02

        target_z = torch.where(is_skywing, fly_z, gnd_z)
        fell = torch.where(is_skywing, torch.zeros_like(gnd_fell), gnd_fell)

        blocked = (~is_skywing) & ((target_z - self.pos_z) > (climb_cap + climb_boost))
        self.pos_x = torch.where(blocked, self.pos_x, new_x)
        self.pos_y = torch.where(blocked, self.pos_y, new_y)
        self.pos_z = torch.where(blocked, self.pos_z, target_z)

        # Clean objective proximity shaping (unowned shrines, or home base if carrying relic!)
        d_shrines = torch.sqrt(
            (self.pos_x[:, :, None] - self.shrine_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.shrine_y[:, None, :]) ** 2
            + ((self.pos_z[:, :, None] - self.shrine_z[:, None, :]) * 1.3) ** 2
        )
        not_ours = self.shrine_owner[:, None, :] != self.team[:, :, None]
        d_target = torch.where(not_ours, d_shrines, d_shrines + 14.0).min(dim=-1).values
        is_carrier = self.relic_carrier.unsqueeze(1) == self.bot_idx_row
        home_x = torch.where(self.team == 0, torch.full_like(self.pos_x, 0.18 * self.W), torch.full_like(self.pos_x, 0.82 * self.W))
        d_home = torch.hypot(self.pos_x - home_x, self.pos_y - 0.50 * self.H)
        d_obj = torch.where(is_carrier, d_home, d_target)
        rewards += (1.0 - (d_obj / 34.0).clamp(0.0, 1.0)) * 0.04 * alive_f

        # Squad cohesion ONLY when actively contesting/engaged near objectives (never pays for camping spawn!)
        d_pair = torch.sqrt(
            (self.pos_x[:, :, None] - self.pos_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.pos_y[:, None, :]) ** 2
        )
        near_squadmate = (self.same_squad_ally & (d_pair < 12.0) & self.alive[:, None, :]).any(dim=-1)
        rewards += torch.where((~self.is_solo) & near_squadmate & (d_target < 18.0) & self.alive, 0.012, 0.0)

        # Update Tidekin submerged state at post-move position
        ix_post = self.pos_x.round().long().clamp(0, self.W - 1)
        iy_post = self.pos_y.round().long().clamp(0, self.H - 1)
        iz_post = self.pos_z.round().long().clamp(0, self.L - 1)
        water_post = torch.maximum(
            self.w_state[a_idx, SP_WATER, iz_post, iy_post, ix_post],
            self.w_state[a_idx, SP_WATER, (iz_post - 1).clamp(0, self.L - 1), iy_post, ix_post],
        )
        self._tidekin_in_water = is_tidekin & (water_post > 0.35) & self.alive
        self.oil_primed_ttl = (self.oil_primed_ttl - 1).clamp(min=0)

        # Physical kinetic fall damage (e.g., dropped from a burning collapsed bridge!)
        fall_dmg = ((fell - 2.4).clamp(min=0.0) * 0.16) * alive_f
        if (fall_dmg > 0.05).any():
            dropped_mask = fall_dmg > 0.05
            for t in (0, 1):
                opp_drops = (dropped_mask & (self.team == (1 - t))).float().sum(dim=1)
                self.stat_bridge_drops[:, t] += opp_drops
                rewards += (self.team == t).float() * opp_drops.unsqueeze(1) * 0.08

        # 2. Conservation-Bounded 3D Physical Actuators (smooth power curves — zero flat deadzone traps!)
        aim_dx = actions[:, :, 3]
        aim_dy = actions[:, :, 4]
        aim_dz = actions[:, :, 5]
        thermal = actions[:, :, 6]
        emit_water = actions[:, :, 7].clamp(0.0, 1.0) ** 1.5
        emit_oil = actions[:, :, 8].clamp(0.0, 1.0) ** 1.5
        emit_smoke = actions[:, :, 9].clamp(0.0, 1.0) ** 1.5
        focus = actions[:, :, 10]
        imp_dome = actions[:, :, 11]

        tx = (self.pos_x + aim_dx * AIM_RANGE_XY).clamp(1.0, self.W - 2.0)
        ty = (self.pos_y + aim_dy * AIM_RANGE_XY).clamp(1.0, self.H - 2.0)
        tz_tgt = (self.pos_z + aim_dz * AIM_RANGE_Z).clamp(1.0, self.L - 2.0)

        # Mana budget scales smoothly with active emissions/impulse/dome (no idle spam drain!)
        rad = 1.4 + focus * 2.2
        impulse_raw = (imp_dome - 0.10).clamp(min=0.0) / 0.90
        want_dome_pre = (imp_dome < -0.35) & (self.dome_ttl == 0) & self.alive
        energy_cost = (
            thermal.abs() * 0.10
            + emit_water * 0.04
            + emit_oil * 0.045
            + emit_smoke * 0.03
            + impulse_raw * 0.085
        ) * (0.72 + 0.28 * focus) + want_dome_pre.float() * 0.16
        can_pay = (self.mana >= energy_cost) & self.alive
        scale_pay = torch.where(can_pay, torch.ones_like(energy_cost), (self.mana / (energy_cost + 1e-5)).clamp(0.0, 1.0)) * alive_f
        self.mana = (self.mana - energy_cost * scale_pay + 0.10 * alive_f).clamp(0.0, 1.0)

        # Place Stasis Dome when impulse_or_dome < -0.35 and mana sufficient
        want_dome = want_dome_pre & (scale_pay > 0.45)
        if want_dome.any():
            self.dome_x = torch.where(want_dome, tx, self.dome_x)
            self.dome_y = torch.where(want_dome, ty, self.dome_y)
            self.dome_z = torch.where(want_dome, tz_tgt, self.dome_z)
            self.dome_ttl = torch.where(want_dome, torch.full_like(self.dome_ttl, 18), self.dome_ttl)
            for t in (0, 1):
                self.stat_stasis_domes[:, t] += (want_dome & (self.team == t)).float().sum(dim=1)

        # Deposit thermal & species into 3D voxel grid (focused beam = higher peak density!)
        tx_i = tx.round().long()
        ty_i = ty.round().long()
        tz_i = tz_tgt.round().long()

        inv_2s2 = 1.0 / (2.0 * (rad * 0.62) ** 2 + 1e-4)  # [A, N]
        w_kern = torch.exp(-self.st_r2.view(1, 1, -1) * inv_2s2.unsqueeze(-1))  # [A, N, K]
        peak_scale = 6.4 - 1.8 * focus.unsqueeze(-1)
        w_kern = w_kern / (w_kern.sum(dim=-1, keepdim=True) + 1e-6) * peak_scale

        gz_idx = (tz_i.unsqueeze(-1) + self.st_dz.view(1, 1, -1)).clamp(0, self.L - 1)
        gy_idx = (ty_i.unsqueeze(-1) + self.st_dy.view(1, 1, -1)).clamp(0, self.H - 1)
        gx_idx = (tx_i.unsqueeze(-1) + self.st_dx.view(1, 1, -1)).clamp(0, self.W - 1)
        flat_idx_a = ((gz_idx * self.H + gy_idx) * self.W + gx_idx).reshape(A, -1)

        d_temp = (thermal * scale_pay * 1.65).unsqueeze(-1) * w_kern
        d_water = (emit_water * scale_pay * 0.55).unsqueeze(-1) * w_kern
        d_oil = (emit_oil * scale_pay * 0.55).unsqueeze(-1) * w_kern
        d_smoke = (emit_smoke * scale_pay * 0.50).unsqueeze(-1) * w_kern

        prev_temp = self.w_state[:, CH_TEMP].clone()
        prev_steam = self.w_state[:, SP_STEAM].clone()

        def _scatter_add_ch(ch: int, vals: torch.Tensor) -> None:
            buf = self.w_state[:, ch].reshape(A, -1)
            buf.scatter_add_(1, flat_idx_a, vals.reshape(A, -1))

        _scatter_add_ch(CH_TEMP, d_temp)
        _scatter_add_ch(SP_WATER, d_water)
        _scatter_add_ch(SP_OIL, d_oil)
        _scatter_add_ch(SP_SMOKE, d_smoke)

        # 3. Emergent 3D Phase Changes, Oil Combustion, Timber Bridge Burning & Persistent Geothermal Plumes
        temp = self.w_state[:, CH_TEMP].clamp(-1.8, 5.5)
        water = self.w_state[:, SP_WATER].clamp(0.0, 2.0)
        ice = self.w_state[:, SP_ICE].clamp(0.0, 2.0)
        wood = self.w_state[:, SP_WOOD].clamp(0.0, 2.0)
        oil = self.w_state[:, SP_OIL].clamp(0.0, 2.0)
        smoke = self.w_state[:, SP_SMOKE].clamp(0.0, 2.0)
        steam = self.w_state[:, SP_STEAM].clamp(0.0, 2.0)

        # Freeze water -> ice (cryo-seals & ice bridges!)
        r_freeze = torch.minimum(water, 0.35 * (temp < -0.25).float())
        water = water - r_freeze
        ice = (ice + r_freeze).clamp(0.0, 2.0)

        # Melt ice -> water
        r_melt = torch.minimum(ice, 0.25 * (temp > 0.60).float())
        ice = ice - r_melt
        water = (water + r_melt).clamp(0.0, 2.0)

        # Flash-boil water -> steam (pressure wave!)
        r_boil = torch.minimum(water, 0.48 * (temp > 1.00).float())
        water = water - r_boil
        steam = (steam + 1.30 * r_boil).clamp(0.0, 2.0)

        # Oil combustion (massive exothermic pressure spike!)
        r_oil = torch.minimum(oil, 0.65 * (temp > 0.85).float())
        oil = oil - r_oil
        temp = (temp + 2.9 * r_oil).clamp(-1.8, 5.5)
        smoke = (smoke + 0.60 * r_oil).clamp(0.0, 2.0)

        # Timber bridge burning -> structural collapse when wood drops below 0.42!
        r_wood = torch.minimum(wood, 0.30 * (temp > 1.25).float())
        wood = wood - r_wood
        temp = (temp + 0.85 * r_wood).clamp(-1.8, 5.5)
        smoke = (smoke + 0.35 * r_wood).clamp(0.0, 2.0)

        # Thermal relaxation + buoyancy + persistent geothermal vent recharge
        temp = temp * 0.88 + 0.02
        temp = torch.where(self.vent_mask, torch.maximum(temp, torch.tensor(1.65, device=dev)), temp)
        smoke = F.pad(smoke[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0) * 0.72 + smoke * 0.18
        steam = F.pad(steam[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0) * 0.70 + steam * 0.16
        steam = torch.where(self.vent_mask & (self.gz.unsqueeze(0) >= 4.0), torch.maximum(steam, torch.tensor(0.35, device=dev)), steam)

        self.w_state[:, CH_TEMP] = temp
        self.w_state[:, SP_WATER] = water
        self.w_state[:, SP_ICE] = ice
        self.w_state[:, SP_WOOD] = wood
        self.w_state[:, SP_OIL] = oil
        self.w_state[:, SP_SMOKE] = smoke
        self.w_state[:, SP_STEAM] = steam
        self.w_state[:, CH_LOGMASS] = torch.log1p(self.w_state[:, :10].sum(dim=1))

        # Track cryo-seals created by bots
        cryo_emit = (thermal < -0.35) & (scale_pay > 0.4)
        for t in (0, 1):
            self.stat_cryo_seals[:, t] += (cryo_emit & (self.team == t)).float().sum(dim=1) * 0.25

        # 4. Physics-Derived Combat Damage (Symmetric Heat/Cryo + Pressure + Impulse + Oil Ignition + Elevation + Aquatic + Fall)
        ix = self.pos_x.round().long().clamp(0, self.W - 1)
        iy = self.pos_y.round().long().clamp(0, self.H - 1)
        iz = self.pos_z.round().long().clamp(0, self.L - 1)

        bot_temp = temp[a_idx, iz, iy, ix]
        bot_dtemp = (bot_temp - prev_temp[a_idx, iz, iy, ix]).clamp(min=0.0)
        bot_dsteam = (steam[a_idx, iz, iy, ix] - prev_steam[a_idx, iz, iy, ix]).clamp(min=0.0)

        scorch = (bot_temp - 0.95).clamp(min=0.0) ** 2 * 0.13
        hypo = (-0.50 - bot_temp).clamp(min=0.0) ** 2 * 0.12
        pressure = (bot_dtemp - 0.38).clamp(min=0.0) * 0.36 + (bot_dsteam - 0.12).clamp(min=0.0) * 0.40

        # Credit enemy actuator deposits within 3D focus sphere (true 3D vertical precision!)
        dist_tx = self.pos_x[:, :, None] - tx[:, None, :]  # [A, victim, attacker]
        dist_ty = self.pos_y[:, :, None] - ty[:, None, :]
        dist_tz = self.pos_z[:, :, None] - tz_tgt[:, None, :]
        dist_3d = torch.sqrt(dist_tx ** 2 + dist_ty ** 2 + (dist_tz * 1.35) ** 2 + 1e-6)
        hit_rad = rad[:, None, :] + 1.3
        hit_mask = self.enemy_mask & (dist_3d <= hit_rad) & (scale_pay[:, None, :] > 0.25)

        # Kinetic shockwave (imp_dome > 0.35) deals physical impulse + radial knockback (knocks enemies off bridges!)
        impulse_mag = impulse_raw * scale_pay
        imp_falloff = (1.0 - dist_3d / (hit_rad + 1e-4)).clamp(min=0.0)
        imp_dmg_mat = torch.where(hit_mask, impulse_mag[:, None, :] * 0.24 * imp_falloff, 0.0)
        imp_dmg = imp_dmg_mat.sum(dim=-1)

        # Physical radial knockback
        kb_mag = torch.where(hit_mask, impulse_mag[:, None, :] * 2.4 * imp_falloff, 0.0)
        inv_dxy = 1.0 / torch.hypot(dist_tx, dist_ty).clamp(min=0.5)
        kb_x = (kb_mag * dist_tx * inv_dxy).sum(dim=-1).clamp(-3.5, 3.5) * alive_f
        kb_y = (kb_mag * dist_ty * inv_dxy).sum(dim=-1).clamp(-3.5, 3.5) * alive_f
        if (kb_x.abs() + kb_y.abs() > 0.1).any():
            kx = (self.pos_x + kb_x).clamp(2.0, self.W - 3.0)
            ky = (self.pos_y + kb_y).clamp(2.0, self.H - 3.0)
            kz, k_fell = self._ground_z_near(kx, ky, self.pos_z)
            # Skywings recover mid-air from knockback; grounded units fall off ledges!
            kz = torch.where(is_skywing, self.pos_z, kz)
            k_fell = torch.where(is_skywing, torch.zeros_like(k_fell), k_fell)
            self.pos_x, self.pos_y, self.pos_z = kx, ky, kz
            k_fall_dmg = (k_fell - 2.4).clamp(min=0.0) * 0.20 * alive_f
            fall_dmg = fall_dmg + k_fall_dmg
            if (k_fall_dmg > 0.05).any():
                for t in (0, 1):
                    d_cnt = ((k_fall_dmg > 0.05) & (self.team == (1 - t))).float().sum(dim=1)
                    self.stat_bridge_drops[:, t] += d_cnt
                    rewards += (self.team == t).float() * d_cnt.unsqueeze(1) * 0.08

        # True 3D elevation advantage (firing downward from high ground/Skywing flight = +30% energy transfer!)
        # and Tidekin aquatic ambush bonus (+25% strike burst when striking from submerged river!)
        att_z_mat = self.pos_z[:, None, :]  # [A, 1, attacker]
        vic_z_mat = self.pos_z[:, :, None]  # [A, victim, 1]
        elev_mult = torch.where((att_z_mat - vic_z_mat) >= 2.0, 1.30, 1.0)
        aqua_mult = torch.where(self._tidekin_in_water[:, None, :], 1.25, 1.0)

        # Direct thermal beam hit matrix (heat deals strong direct thermal damage + ignites oil/bridges; cryo chills/freezes!)
        heat_dmg_part = thermal.clamp(min=0.0) * 0.19
        cryo_dmg_part = (-thermal).clamp(min=0.0) * 0.125
        act_dmg_mat = torch.where(
            hit_mask,
            ((heat_dmg_part + cryo_dmg_part)[:, None, :] + imp_dmg_mat) * imp_falloff * elev_mult * aqua_mult,
            0.0,
        )
        best_att_val, best_att_idx = act_dmg_mat.max(dim=-1)
        got_hit = (best_att_val > 0.02) & self.alive
        self.last_attacker = torch.where(got_hit, best_att_idx, self.last_attacker)

        # True 3D vertical hits (|z_attacker - z_victim| >= 2.0m!)
        att_z = self.pos_z.gather(1, best_att_idx.clamp(0, N - 1))
        vert_hit = got_hit & ((att_z - self.pos_z).abs() >= 2.0)
        if vert_hit.any():
            for t in (0, 1):
                vh_cnt = (vert_hit & (self.team == (1 - t))).float().sum(dim=1)
                self.stat_vert_hits[:, t] += vh_cnt
                rewards += (self.team == t).float() * vh_cnt.unsqueeze(1) * 0.03

        # Tidekin aquatic ambush (striking enemies from submerged water!)
        att_in_water = self._tidekin_in_water.gather(1, best_att_idx.clamp(0, N - 1))
        aq_hit = got_hit & att_in_water
        if aq_hit.any():
            for t in (0, 1):
                aq_cnt = (aq_hit & (self.team == (1 - t))).float().sum(dim=1)
                self.stat_aquatic_ambushes[:, t] += aq_cnt
                rewards += (self.team == t).float() * aq_cnt.unsqueeze(1) * 0.04

        # Physical Oil-Primer -> Thermal Flash-Ignition Combo (100% physics: oil-coated target + heat = flash burst!)
        oil_hit = hit_mask & (emit_oil[:, None, :] > 0.15)
        not_yet_oiled = self.oil_primed_ttl == 0
        # Direct per-bot physical credit for coating an unprimed enemy with combustible oil
        oil_prime_credit = (oil_hit & not_yet_oiled[:, :, None] & self.alive[:, :, None]).float().sum(dim=1)
        rewards += oil_prime_credit * 0.06 * alive_f

        any_oil_hit = oil_hit.any(dim=-1) & self.alive
        oil_src = oil_hit.float().argmax(dim=-1)
        self.oil_primed_by = torch.where(any_oil_hit, oil_src, self.oil_primed_by)
        self.oil_primed_ttl = torch.where(any_oil_hit, torch.full_like(self.oil_primed_ttl, 8), self.oil_primed_ttl)

        att_thermal = thermal.gather(1, best_att_idx.clamp(0, N - 1))
        oil_ignited = got_hit & (self.oil_primed_ttl > 0) & (att_thermal > 0.20) & (self.oil_primed_by >= 0)
        oil_combo_dmg = torch.zeros_like(scorch)
        if oil_ignited.any():
            oil_combo_dmg = torch.where(oil_ignited, torch.full_like(scorch, 0.24), torch.zeros_like(scorch))
            pressure = pressure + torch.where(oil_ignited, torch.full_like(pressure, 0.12), torch.zeros_like(pressure))
            # Credit both igniter and primer directly for the exothermic combo burst!
            ign_onehot = F.one_hot(best_att_idx.clamp(0, N - 1), num_classes=N).float() * oil_ignited.float().unsqueeze(-1)
            prm_onehot = F.one_hot(self.oil_primed_by.clamp(0, N - 1), num_classes=N).float() * oil_ignited.float().unsqueeze(-1)
            rewards += (ign_onehot.sum(dim=1) + prm_onehot.sum(dim=1)) * 0.14 * alive_f
            sq_combo = oil_ignited & (self.oil_primed_by != best_att_idx)
            if sq_combo.any():
                for t in (0, 1):
                    c_cnt = (sq_combo & (self.team == (1 - t))).float().sum(dim=1)
                    self.stat_squad_combos[:, t] += c_cnt
                    rewards += (self.team == t).float() * c_cnt.unsqueeze(1) * 0.08
            self.oil_primed_ttl = torch.where(oil_ignited, torch.zeros_like(self.oil_primed_ttl), self.oil_primed_ttl)

        if (pressure > 0.06).any():
            p_hit_att = (hit_mask & (pressure > 0.06)[:, :, None] & self.alive[:, :, None]).float().sum(dim=1)
            rewards += p_hit_att * 0.06 * alive_f
            for t in (0, 1):
                self.stat_pressure_blasts[:, t] += ((pressure > 0.06) & (self.team == (1 - t))).float().sum(dim=1)

        # Allied Stasis Dome shields allies by 45% + slows enemies inside by 75% (direct credit to caster!)
        in_ally_dome = ((self.dome_ttl[:, None, :] > 0) & (~self.enemy_mask) & (d2_domes < 52.0)).any(dim=-1)
        dome_shield = torch.where(in_ally_dome, 0.55, 1.0)
        smoke_shield = torch.where(is_stalker & in_smoke, 0.72, 1.0)
        dome_active_hits = ((self.dome_ttl[:, None, :] > 0) & self.enemy_mask & (d2_domes < 52.0) & self.alive[:, :, None]).float().sum(dim=1)
        dome_ally_shields = ((self.dome_ttl[:, None, :] > 0) & (~self.enemy_mask) & (d2_domes < 52.0) & self.alive[:, :, None]).float().sum(dim=1)
        rewards += (dome_active_hits * 0.06 + dome_ally_shields * 0.015) * alive_f

        direct_thermal_dmg = act_dmg_mat.sum(dim=-1) * 0.55
        total_dmg = (scorch + hypo + pressure + imp_dmg + direct_thermal_dmg + oil_combo_dmg + fall_dmg) * dome_shield * smoke_shield * alive_f
        self.hp = (self.hp - total_dmg).clamp(0.0, 1.0)

        # Reward attacker directly for physical damage dealt (strong per-bot combat credit!)
        dmg_dealt_per_att = act_dmg_mat.sum(dim=1)  # [A, N]
        rewards += dmg_dealt_per_att * 0.85

        # Eliminations
        died = self.alive & (self.hp <= 0.0)
        if died.any():
            self.alive = self.alive & (~died)
            self.respawn_cd = torch.where(died, torch.full_like(self.respawn_cd, 9), self.respawn_cd)
            rewards -= died.float() * 0.65
            carrier_died = (self.relic_carrier >= 0) & died.gather(1, self.relic_carrier.clamp(0, N - 1).unsqueeze(1)).squeeze(1)
            self.relic_carrier = torch.where(carrier_died, torch.full_like(self.relic_carrier, -1), self.relic_carrier)

            for t in (0, 1):
                opp_died = (died & (self.team == (1 - t))).float().sum(dim=1)
                self.team_kills[:, t] += opp_died
                self.team_score[:, t] += opp_died * 8.0
                rewards += (self.team == t).float() * opp_died.unsqueeze(1) * 0.28

        # 5. 3D Shrine Control (requires being within 7m XY AND 3.2m Z — Skywings hovering slightly above still contest!)
        for k in range(3):
            dxy = torch.hypot(self.pos_x - self.shrine_x[:, k : k + 1], self.pos_y - self.shrine_y[:, k : k + 1])
            dz = (self.pos_z - self.shrine_z[:, k : k + 1]).abs()
            z_tol = torch.where(is_skywing, 4.5, 3.0)
            on_shrine = self.alive & (dxy < 7.0) & (dz < z_tol)
            w_cnt = (on_shrine & (self.team == 0)).float().sum(dim=1)
            e_cnt = (on_shrine & (self.team == 1)).float().sum(dim=1)
            delta_p = (w_cnt - e_cnt).clamp(-2.0, 2.0) * 0.15
            self.shrine_prog[:, k] = (self.shrine_prog[:, k] + delta_p).clamp(-1.0, 1.0)
            self.shrine_owner[:, k] = torch.where(
                self.shrine_prog[:, k] >= 0.85, 0,
                torch.where(self.shrine_prog[:, k] <= -0.85, 1, self.shrine_owner[:, k]),
            )
            rewards += on_shrine.float() * 0.08
            self.team_score[:, 0] += (self.shrine_owner[:, k] == 0).float() * 0.50
            self.team_score[:, 1] += (self.shrine_owner[:, k] == 1).float() * 0.50
            rewards += (self.team == 0).float() * (self.shrine_owner[:, k] == 0).float().unsqueeze(1) * 0.045
            rewards += (self.team == 1).float() * (self.shrine_owner[:, k] == 1).float().unsqueeze(1) * 0.045

        # 6. 3D Relic Pickup & Delivery (center river relic -> home base!)
        unheld = self.relic_carrier < 0
        if unheld.any():
            dr = torch.sqrt(
                (self.pos_x - self.relic_x.unsqueeze(1)) ** 2
                + (self.pos_y - self.relic_y.unsqueeze(1)) ** 2
                + (self.pos_z - self.relic_z.unsqueeze(1)) ** 2
            )
            dr = torch.where(self.alive, dr, torch.full_like(dr, 999.0))
            min_d, min_i = dr.min(dim=1)
            picked = unheld & (min_d < 4.8)
            self.relic_carrier = torch.where(picked, min_i, self.relic_carrier)

        held = self.relic_carrier >= 0
        if held.any():
            c_idx = self.relic_carrier.clamp(0, N - 1).unsqueeze(1)
            cx = self.pos_x.gather(1, c_idx).squeeze(1)
            cy = self.pos_y.gather(1, c_idx).squeeze(1)
            cz = self.pos_z.gather(1, c_idx).squeeze(1)
            self.relic_x = torch.where(held, cx, self.relic_x)
            self.relic_y = torch.where(held, cy, self.relic_y)
            self.relic_z = torch.where(held, cz, self.relic_z)

            c_team = (self.relic_carrier >= 6).long()
            delivered_w = held & (c_team == 0) & (cx < 0.24 * self.W)
            delivered_e = held & (c_team == 1) & (cx > 0.76 * self.W)
            self.team_score[:, 0] += delivered_w.float() * 30.0
            self.team_score[:, 1] += delivered_e.float() * 30.0
            rewards += (self.team == 0).float() * delivered_w.float().unsqueeze(1) * 1.6
            rewards += (self.team == 1).float() * delivered_e.float().unsqueeze(1) * 1.6
            reset_r = delivered_w | delivered_e
            self.relic_carrier = torch.where(reset_r, torch.full_like(self.relic_carrier, -1), self.relic_carrier)
            self.relic_x = torch.where(reset_r, torch.full_like(self.relic_x, 0.50 * self.W), self.relic_x)
            self.relic_y = torch.where(reset_r, torch.full_like(self.relic_y, 0.50 * self.H), self.relic_y)
            self.relic_z = torch.where(reset_r, torch.full_like(self.relic_z, 3.0), self.relic_z)

        # Refresh frozen 3D world latents periodically
        if self.tick % self.refresh_every == 0:
            self.z_world = self.world_encoder.encode(self.w_state)

        # Dense zero-rule game-score advantage delta (pure win-objective gradient!)
        d_sc = self.team_score - prev_team_score
        rewards += (self.team == 0).float() * (d_sc[:, 0] - d_sc[:, 1]).unsqueeze(1) * 0.09
        rewards += (self.team == 1).float() * (d_sc[:, 1] - d_sc[:, 0]).unsqueeze(1) * 0.09

        done = self.tick >= self.max_ticks
        if done:
            win_w = (self.team_score[:, 0] > self.team_score[:, 1]).float()
            win_e = (self.team_score[:, 1] > self.team_score[:, 0]).float()
            rewards += (self.team == 0).float() * (win_w - win_e).unsqueeze(1) * 4.0
            rewards += (self.team == 1).float() * (win_e - win_w).unsqueeze(1) * 4.0
        return rewards, done


def pure_spatial_nav_aim_targets(arena: Physics3DArenaV3) -> torch.Tensor:
    """
    Phase 1 Pure-Spatial Geometry Teacher (ONLY channels 0..5: move_x, move_y, move_jump, aim_dx, aim_dy, aim_dz!).
    ZERO spell rules, ZERO thermal/elemental/dome supervision: physical channels 6..11 are NEVER touched here!
    """
    A, N = arena.A, arena.N
    dev = arena.device
    tgt = torch.zeros((A, N, 6), device=dev)

    # Hold assigned lane shrine (prevents leaving shrines/relic undefended!), rotate only if lane is owned AND enemies are far
    lane_s = arena.lane_shrine_idx.unsqueeze(0).expand(A, -1)  # [A, N]
    gx = arena.shrine_x.gather(1, lane_s)
    gy = arena.shrine_y.gather(1, lane_s)
    gz = arena.shrine_z.gather(1, lane_s)

    # If relic is unheld, Center-lane bots (lane_shrine == 1) prioritize grabbing it!
    unheld = (arena.relic_carrier < 0).unsqueeze(1)
    want_relic = unheld & (lane_s == 1)
    gx = torch.where(want_relic, arena.relic_x.unsqueeze(1), gx)
    gy = torch.where(want_relic, arena.relic_y.unsqueeze(1), gy)
    gz = torch.where(want_relic, arena.relic_z.unsqueeze(1), gz)

    # Relic carrier returns home
    is_carrier = arena.relic_carrier.unsqueeze(1) == arena.bot_idx_row
    home_x = torch.where(arena.team == 0, torch.full_like(gx, 0.18 * arena.W), torch.full_like(gx, 0.82 * arena.W))
    home_y = torch.full_like(gy, 0.50 * arena.H)
    gx = torch.where(is_carrier, home_x, gx)
    gy = torch.where(is_carrier, home_y, gy)

    dx = gx - arena.pos_x
    dy = gy - arena.pos_y
    dn = torch.hypot(dx, dy).clamp(min=1e-4)
    # Smooth arrival damping inside 2.5m so bots stand steadily on shrines
    arr_scale = (dn / 2.5).clamp(0.0, 1.0)
    tgt[:, :, 0] = (dx / dn * arr_scale).clamp(-0.95, 0.95)
    tgt[:, :, 1] = (dy / dn * arr_scale).clamp(-0.95, 0.95)

    # Pure spatial Z locomotion: Skywing climbs/soars high; grounded units jump/climb when goal is higher
    is_skywing = arena.cls == 1
    need_climb = gz > (arena.pos_z + 1.0)
    tgt[:, :, 2] = torch.where(is_skywing, 0.72, torch.where(need_climb, 0.65, 0.15))

    # Pure 3D geometric aim ray toward nearest living enemy
    ex = arena.pos_x[:, None, :] - arena.pos_x[:, :, None]  # [A, bot, target]
    ey = arena.pos_y[:, None, :] - arena.pos_y[:, :, None]
    ez = arena.pos_z[:, None, :] - arena.pos_z[:, :, None]
    d3 = torch.sqrt(ex ** 2 + ey ** 2 + ez ** 2 + 1e-6)
    valid = arena.enemy_mask & arena.alive[:, None, :]
    d3 = torch.where(valid, d3, torch.full_like(d3, 999.0))
    _, min_j = d3.min(dim=-1)

    t_dx = ex.gather(2, min_j.unsqueeze(-1)).squeeze(-1)
    t_dy = ey.gather(2, min_j.unsqueeze(-1)).squeeze(-1)
    t_dz = ez.gather(2, min_j.unsqueeze(-1)).squeeze(-1)

    tgt[:, :, 3] = (t_dx / AIM_RANGE_XY).clamp(-0.95, 0.95)
    tgt[:, :, 4] = (t_dy / AIM_RANGE_XY).clamp(-0.95, 0.95)
    tgt[:, :, 5] = (t_dz / AIM_RANGE_Z).clamp(-0.95, 0.95)
    return tgt


def scripted_3d_policy(arena: Physics3DArenaV3) -> torch.Tensor:
    """Strong realistic tactical 3D scripted baseline (shrine/relic pursuit + 3D vertical targeting + heat/cryo/impulse)."""
    A, N = arena.A, arena.N
    dev = arena.device
    act = torch.zeros((A, N, ACT3D_DIM), device=dev)

    lane_shrine = arena.lane_shrine_idx
    gx = arena.shrine_x[:, lane_shrine]
    gy = arena.shrine_y[:, lane_shrine]

    # Carrier runs home
    is_carrier = arena.relic_carrier.unsqueeze(1) == arena.bot_idx_row
    home_x = torch.where(arena.team == 0, torch.full_like(gx, 0.16 * arena.W), torch.full_like(gx, 0.84 * arena.W))
    gx = torch.where(is_carrier, home_x, gx)

    dx = gx - arena.pos_x
    dy = gy - arena.pos_y
    dn = torch.hypot(dx, dy).clamp(min=1e-4)
    act[:, :, 0] = (dx / dn).clamp(-1.0, 1.0)
    act[:, :, 1] = (dy / dn).clamp(-1.0, 1.0)
    act[:, :, 2] = 0.45

    # Aim at nearest living enemy in 3D
    ex = arena.pos_x[:, None, :] - arena.pos_x[:, :, None]
    ey = arena.pos_y[:, None, :] - arena.pos_y[:, :, None]
    ez = arena.pos_z[:, None, :] - arena.pos_z[:, :, None]
    d3 = torch.sqrt(ex ** 2 + ey ** 2 + ez ** 2 + 1e-6)
    valid = arena.enemy_mask & arena.alive[:, None, :]
    d3 = torch.where(valid, d3, torch.full_like(d3, 999.0))
    min_d, min_j = d3.min(dim=-1)

    t_dx = ex.gather(2, min_j.unsqueeze(-1)).squeeze(-1)
    t_dy = ey.gather(2, min_j.unsqueeze(-1)).squeeze(-1)
    t_dz = ez.gather(2, min_j.unsqueeze(-1)).squeeze(-1)

    act[:, :, 3] = (t_dx / AIM_RANGE_XY).clamp(-1.0, 1.0)
    act[:, :, 4] = (t_dy / AIM_RANGE_XY).clamp(-1.0, 1.0)
    act[:, :, 5] = (t_dz / AIM_RANGE_Z).clamp(-1.0, 1.0)

    in_range = min_d < 12.0
    act[:, :, 6] = torch.where(in_range, torch.where(arena.cls == 1, -0.85, 0.90), 0.0)
    act[:, :, 7] = torch.where(in_range & (arena.cls == 1), 0.45, 0.0)
    act[:, :, 8] = torch.where(in_range & (arena.cls == 2), 0.50, 0.0)
    act[:, :, 10] = 0.25
    act[:, :, 11] = torch.where(min_d < 8.5, 0.65, 0.0)
    return act


@torch.no_grad()
def evaluate_matchup(
    policy_w: ActuatorPolicyNetV3,
    opponent_kind: str,
    arena: Physics3DArenaV3,
    num_matches: int = 24,
    base_seed: int = 9000,
) -> dict[str, Any]:
    """
    True 50/50 side-balanced held-out 3D tournament (exact half West + half East) on unseen seeds.
    In every batch, arenas [0..half_b-1] assign RL to West (team 0) and arenas [half_b..take-1]
    assign RL to East (team 1), running 24 matches in a single GPU batch when num_matches <= arena.A!
    """
    policy_w.eval()
    wins = 0.0
    losses = 0.0
    draws = 0.0
    score_diff_sum = 0.0
    west_wins = 0
    west_total = 0
    east_wins = 0
    east_total = 0
    tactics = {
        "vert_hits": 0.0,
        "bridge_drops": 0.0,
        "updraft_soars": 0.0,
        "aquatic_ambushes": 0.0,
        "squad_combos": 0.0,
        "cryo_seals": 0.0,
        "stasis_domes": 0.0,
        "pressure_blasts": 0.0,
    }

    dev = arena.device
    target_w = num_matches // 2
    target_e = num_matches - target_w
    done_w = 0
    done_e = 0
    batch_idx = 0

    while (done_w + done_e) < num_matches:
        rem_w = target_w - done_w
        rem_e = target_e - done_e
        take = min(arena.A, rem_w + rem_e)
        take_w = min(rem_w, max(1, take // 2) if rem_e > 0 else take)
        take_e = min(rem_e, take - take_w)
        # Ensure we fill up to `take` if one side finished early
        if take_w + take_e < take:
            take_w = min(rem_w, take - take_e)

        # Arena side assignment: True -> RL plays East (team 1), False -> RL plays West (team 0)
        rl_is_east = torch.zeros(arena.A, dtype=torch.bool, device=dev)
        if take_e > 0:
            rl_is_east[take_w : take_w + take_e] = True

        arena.reset(seed=base_seed + batch_idx * 37)
        batch_idx += 1
        done = False
        while not done:
            obs = arena.build_observations()
            out_rl = policy_w(
                arena.z_world,
                obs["pos_xyz"],
                obs["self_feat"],
                obs["tokens"],
                token_mask=obs["token_mask"],
                arena_idx=arena.arena_idx_flat,
            )
            act_rl = out_rl["mean"].reshape(arena.A, arena.N, ACT3D_DIM)

            if opponent_kind == "scripted_3d":
                act_opp = scripted_3d_policy(arena)
            elif opponent_kind == "flat_2d_ablation":
                act_opp = act_rl.clone()
                act_opp[:, :, 5] = 0.0  # force flat aim_dz = 0
                act_opp[:, :, 11] = act_opp[:, :, 11].clamp(min=0.0)  # disable Stasis domes
            elif opponent_kind == "random_3d":
                act_opp = torch.rand((arena.A, arena.N, ACT3D_DIM), device=dev) * 2.0 - 1.0
                act_opp[:, :, 7:11] = act_opp[:, :, 7:11].abs()
            else:
                raise ValueError(opponent_kind)

            # Compose per-arena side-balanced actions
            act_west_rl = torch.cat((act_rl[:, :6], act_opp[:, 6:]), dim=1)
            act_east_rl = torch.cat((act_opp[:, :6], act_rl[:, 6:]), dim=1)
            act = torch.where(rl_is_east[:, None, None], act_east_rl, act_west_rl)
            _, done = arena.step(act)

        # Tally West arenas [0 : take_w] and East arenas [take_w : take_w + take_e]
        for sl, my_t, opp_t, is_e in [
            (slice(0, take_w), 0, 1, False),
            (slice(take_w, take_w + take_e), 1, 0, True),
        ]:
            cnt = sl.stop - sl.start
            if cnt <= 0:
                continue
            my_sc = arena.team_score[sl, my_t]
            op_sc = arena.team_score[sl, opp_t]
            w_cur = int((my_sc > op_sc).sum().item())
            l_cur = int((my_sc < op_sc).sum().item())
            d_cur = int((my_sc == op_sc).sum().item())
            wins += w_cur
            losses += l_cur
            draws += d_cur
            if is_e:
                east_wins += w_cur
                east_total += cnt
            else:
                west_wins += w_cur
                west_total += cnt
            score_diff_sum += float((my_sc - op_sc).sum().item())
            tactics["vert_hits"] += float(arena.stat_vert_hits[sl, my_t].sum().item())
            tactics["bridge_drops"] += float(arena.stat_bridge_drops[sl, my_t].sum().item())
            tactics["updraft_soars"] += float(arena.stat_updraft_soars[sl, my_t].sum().item())
            tactics["aquatic_ambushes"] += float(arena.stat_aquatic_ambushes[sl, my_t].sum().item())
            tactics["squad_combos"] += float(arena.stat_squad_combos[sl, my_t].sum().item())
            tactics["cryo_seals"] += float(arena.stat_cryo_seals[sl, my_t].sum().item())
            tactics["stasis_domes"] += float(arena.stat_stasis_domes[sl, my_t].sum().item())
            tactics["pressure_blasts"] += float(arena.stat_pressure_blasts[sl, my_t].sum().item())

        done_w += take_w
        done_e += take_e

    wr = (wins + 0.5 * draws) / max(num_matches, 1)
    elo = 400.0 * math.log10(max(wr, 0.01) / max(1.0 - wr, 0.01))
    return {
        "opponent": opponent_kind,
        "matches": num_matches,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "west_wins": west_wins,
        "west_matches": west_total,
        "east_wins": east_wins,
        "east_matches": east_total,
        "win_rate": round(wr, 4),
        "elo_delta": round(elo, 1),
        "mean_score_diff": round(score_diff_sum / max(num_matches, 1), 2),
        "tactics_per_match": {k: round(v / max(num_matches, 1), 2) for k, v in tactics.items()},
    }


def save_viz(history: list[dict[str, float]], tournament: dict[str, Any], out_png: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Viz] Skipping plot ({exc})", flush=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    # Panel 1: PPO Self-Play Return & Win-Rate Curve
    iters = [h["iter"] for h in history]
    rets = [h["mean_return"] for h in history]
    wrs = [h.get("eval_scripted_wr", float("nan")) * 100.0 for h in history]
    axes[0].plot(iters, rets, color="#58a6ff", lw=2.0, alpha=0.85, label="Episode Return")
    axes[0].set_title("Wave 14A: ActuatorPolicyNetV3 (Z=16) PPO League", color="#f0f6fc", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("PPO Iteration (after 12-step spatial warmstart)", color="#8b949e")
    axes[0].set_ylabel("Mean Return", color="#58a6ff")
    ax0_r = axes[0].twinx()
    valid_pts = [(i, w) for i, w in zip(iters, wrs) if not math.isnan(w)]
    if valid_pts:
        ax0_r.plot([p[0] for p in valid_pts], [p[1] for p in valid_pts], "o-", color="#3fb950", lw=2.4, ms=5, label="vs Scripted-3D WR%")
        ax0_r.axhline(80.0, color="#3fb950", ls="--", lw=1.1, alpha=0.65)
        ax0_r.set_ylabel("Win Rate vs Scripted-3D (%)", color="#3fb950")
        ax0_r.set_ylim(0, 108)
        ax0_r.tick_params(colors="#3fb950")

    # Panel 2: Held-Out 3D Tournament Win Rates (24 side-balanced matches)
    names = []
    vals = []
    elos = []
    colors = ["#3fb950", "#a371f7", "#f0883e"]
    for k, label in [("scripted_3d", "vs Scripted-3D"), ("flat_2d_ablation", "vs Flat-2D\nAblation"), ("random_3d", "vs Random-3D")]:
        if k in tournament:
            names.append(label)
            vals.append(tournament[k]["win_rate"] * 100.0)
            elos.append(tournament[k]["elo_delta"])
    bars = axes[1].bar(names, vals, color=colors[: len(names)], width=0.55)
    axes[1].axhline(50.0, color="#8b949e", ls="--", lw=1.1)
    axes[1].axhline(80.0, color="#3fb950", ls=":", lw=1.1, alpha=0.7)
    axes[1].set_ylim(0, 118)
    axes[1].set_title("Held-Out 3D Multi-Biome Tournament (24 Matches)", color="#f0f6fc", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Win Rate (%)", color="#c9d1d9")
    for b, v, elo in zip(bars, vals, elos):
        axes[1].text(
            b.get_x() + b.get_width() / 2.0,
            v + 2.5,
            f"{v:.1f}%\n({elo:+.0f} Elo)",
            ha="center",
            color="#f0f6fc",
            fontweight="bold",
            fontsize=9,
        )

    # Panel 3: Emergent 3D Tactical & Multi-Archetype Discoveries per Match
    tac = tournament.get("scripted_3d", {}).get("tactics_per_match", {})
    t_labels = ["3D Vert\nHits", "Skywing\nUpdrafts", "Tidekin\nAquatic", "Squad\nCombos", "Bridge\nDrops", "Cryo\nSeals", "Stasis\nDomes", "Pressure\nBlasts"]
    t_keys = ["vert_hits", "updraft_soars", "aquatic_ambushes", "squad_combos", "bridge_drops", "cryo_seals", "stasis_domes", "pressure_blasts"]
    t_vals = [float(tac.get(k, 0.0)) for k in t_keys]
    t_cols = ["#58a6ff", "#ffa657", "#39d353", "#f778ba", "#f85149", "#79c0ff", "#d2a8ff", "#e3b341"]
    b3 = axes[2].bar(t_labels, t_vals, color=t_cols, width=0.65)
    axes[2].set_title("Emergent 3D & Archetype Tactics / Match (Zero Spells!)", color="#f0f6fc", fontsize=10.5, fontweight="bold")
    axes[2].set_ylabel("Events / Match", color="#c9d1d9")
    axes[2].tick_params(axis="x", labelsize=8)
    top_y = max(t_vals + [1.0]) * 1.18
    axes[2].set_ylim(0, top_y)
    for b, v in zip(b3, t_vals):
        axes[2].text(b.get_x() + b.get_width() / 2.0, v + top_y * 0.025, f"{v:.1f}", ha="center", color="#f0f6fc", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Viz] Saved {out_png}", flush=True)


def run_phase1_spatial_warmstart(
    policy: ActuatorPolicyNetV3,
    arena: Physics3DArenaV3,
    warmstart_steps: int,
    rollout_steps: int,
    minibatch: int,
    device: torch.device,
    t0: float,
) -> list[dict[str, float]]:
    """
    Phase 1: Pure-Spatial Navigation & 3D Aiming Warmstart (ONLY channels 0..5!).
    Physical actuator channels 6..11 (`thermal_flux, emit_water, emit_oil, emit_smoke, focus_radius, impulse_or_dome`)
    receive ZERO supervision — 100% unsupervised for pure RL emergence!
    """
    if warmstart_steps <= 0:
        return []

    print(f"\n[Phase 1] Starting {warmstart_steps}-step Pure-Spatial Warmstart (channels 0..5 ONLY; 6..11 100% UNSUPERVISED)...", flush=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=1.4e-3, weight_decay=1e-4)
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    ws_logs: list[dict[str, float]] = []

    for step_idx in range(1, warmstart_steps + 1):
        arena.reset(seed=200 + step_idx * 29)
        buf_fine, buf_coarse, buf_sf, buf_tok, buf_mask, buf_tgt = [], [], [], [], [], []

        # DAgger schedule: mix spatial teacher & policy spatial predictions so policy visits realistic states on BOTH sides
        beta = max(0.25, 1.0 - (step_idx - 1) / max(warmstart_steps, 1))
        policy.eval()
        ws_ticks = min(rollout_steps, 40)
        for _ in range(ws_ticks):
            obs = arena.build_observations()
            spatial_tgt = pure_spatial_nav_aim_targets(arena)  # [A, N, 6]
            with torch.inference_mode():
                fine_raw, coarse_raw, _ = policy.sample_raw_crops(
                    arena.z_world, obs["pos_xyz"], arena_idx=arena.arena_idx_flat
                )
                out = policy.forward_from_crops(
                    fine_raw, coarse_raw, obs["self_feat"], obs["tokens"], token_mask=obs["token_mask"]
                )
                pol_mean = out["mean"].reshape(arena.A, arena.N, ACT3D_DIM)

            # Execute mixed spatial channels + neutral random exploration on physical channels (no spell imitation!)
            exec_act = torch.zeros((arena.A, arena.N, ACT3D_DIM), device=device)
            use_t = (torch.rand((arena.A, arena.N, 1), device=device) < beta).float()
            exec_act[:, :, :6] = (use_t * spatial_tgt + (1.0 - use_t) * pol_mean[:, :, :6] + torch.randn_like(spatial_tgt) * 0.08).clamp(-1.0, 1.0)
            # Random physical emissions during warmstart rollouts so 3D world state has diverse fire/water/smoke/bridges
            exec_act[:, :, 6] = torch.empty((arena.A, arena.N), device=device).uniform_(-0.5, 0.5)
            exec_act[:, :, 7:11] = torch.empty((arena.A, arena.N, 4), device=device).uniform_(0.0, 0.35)
            exec_act[:, :, 11] = torch.empty((arena.A, arena.N), device=device).uniform_(-0.3, 0.3)

            _, done = arena.step(exec_act)

            # Keep ALL 12 bots (West + East) for 100% side-balanced spatial learning!
            buf_fine.append(fine_raw.detach())
            buf_coarse.append(coarse_raw.detach())
            buf_sf.append(obs["self_feat"].detach())
            buf_tok.append(obs["tokens"].detach())
            buf_mask.append(obs["token_mask"].detach())
            buf_tgt.append(spatial_tgt.reshape(-1, 6).detach())
            if done:
                arena.reset(seed=3000 + step_idx * 53)

        f_fine = torch.cat(buf_fine, dim=0)
        f_coarse = torch.cat(buf_coarse, dim=0)
        f_sf = torch.cat(buf_sf, dim=0)
        f_tok = torch.cat(buf_tok, dim=0)
        f_mask = torch.cat(buf_mask, dim=0)
        f_tgt = torch.cat(buf_tgt, dim=0)

        policy.train()
        total_n = f_tgt.shape[0]
        last_mse = 0.0
        for _ in range(3):
            perm = torch.randperm(total_n, device=device)
            for s in range(0, total_n, minibatch):
                mb = perm[s : s + minibatch]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out_mb = policy.forward_from_crops(
                        f_fine[mb], f_coarse[mb], f_sf[mb], f_tok[mb], token_mask=f_mask[mb]
                    )
                pred_spatial = out_mb["mean"][:, :6].float()
                # ONLY spatial channels 0..5 supervised; channels 6..11 untouched!
                loss = F.mse_loss(pred_spatial, f_tgt[mb])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()
                last_mse = float(loss.item())

        ws_logs.append({"ws_step": float(step_idx), "spatial_mse": round(last_mse, 5)})
        if step_idx in (1, 4, 8, warmstart_steps) or step_idx % 4 == 0:
            print(f"[Phase 1 Warmstart {step_idx:2d}/{warmstart_steps}] spatial_mse={last_mse:.4f} | elapsed={time.time()-t0:.1f}s", flush=True)

    # Neutral low-emission bias on sigmoid channels 7..9 (sigmoid(-1.1)=0.25 -> 0.25^1.5=0.125, smooth gradient everywhere!)
    # and calibrated exploration stds: tighter on learned spatial (0..5), wider on pure-RL physical actuators (6..11).
    with torch.no_grad():
        policy.actor_mean.bias[7:10].fill_(-1.1)
        policy.actor_log_std[:6].fill_(-1.15)   # std ~ 0.32 on spatial nav/aim
        policy.actor_log_std[6:].fill_(-0.45)   # std ~ 0.64 on pure-RL physical actuators
    print(f"[Phase 1] Complete in {time.time()-t0:.1f}s! Channels 6..11 remain 100% unsupervised.\n", flush=True)
    return ws_logs


def _resolve_world_ckpt(path: Path) -> Path | None:
    if path.exists():
        return path
    alt = Path("/content") / path.name
    if alt.exists():
        return alt
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-ckpt", type=Path, default=Path("models/checkpoints/world_model_3d_v5.pt"))
    ap.add_argument("--out-ckpt", type=Path, default=Path("models/checkpoints/actuator_policy_v3_rl.pt"))
    ap.add_argument("--out-json", type=Path, default=Path("models/ACTUATOR_V3_TOURNAMENT.json"))
    ap.add_argument("--out-png", type=Path, default=Path("models/actuator_v3_viz.png"))
    ap.add_argument("--num-arenas", type=int, default=24)
    ap.add_argument("--grid-xy", type=int, default=64)
    ap.add_argument("--depth-z", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=64)
    ap.add_argument("--warmstart-steps", type=int, default=12)
    ap.add_argument("--iters", type=int, default=110)
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--eval-matches", type=int, default=24)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.num_arenas = 2
        args.grid_xy = 32
        args.rollout_steps = 8
        args.warmstart_steps = 2
        args.iters = 2
        args.ppo_epochs = 1
        args.minibatch = 32
        args.eval_matches = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    t0 = time.time()
    print(
        f"[ActuatorV3-RL Wave14A] Device={device} (bf16={use_bf16}) | arenas={args.num_arenas} | "
        f"grid={args.grid_xy}x{args.grid_xy}x{args.depth_z} | ws={args.warmstart_steps} | iters={args.iters} | steps={args.rollout_steps}",
        flush=True,
    )

    resolved_wm = _resolve_world_ckpt(args.world_ckpt)
    world_enc = FrozenWorldLatentExtractor(resolved_wm, device=device)
    arena = Physics3DArenaV3(
        num_arenas=args.num_arenas,
        world_encoder=world_enc,
        device=device,
        grid_xy=args.grid_xy,
        depth_z=args.depth_z,
        max_ticks=max(args.rollout_steps, 16) if args.smoke_test else args.rollout_steps,
    )

    policy = ActuatorPolicyNetV3().to(device)
    n_params = count_parameters(policy)
    print(f"[ActuatorV3-RL] Policy params={n_params:,} | WorldLatent mode={world_enc.mode}", flush=True)

    # Phase 1: 12-Step Pure-Spatial Navigation & 3D Aiming Warmstart (channels 0..5 ONLY; 6..11 100% unsupervised!)
    ws_logs = run_phase1_spatial_warmstart(
        policy=policy,
        arena=arena,
        warmstart_steps=args.warmstart_steps,
        rollout_steps=args.rollout_steps,
        minibatch=args.minibatch,
        device=device,
        t0=t0,
    )

    # Phase 2: Pure-Physics PPO League Self-Play (50% vs full-strength Scripted-3D + 50% League Pool!)
    # First: 2 Critic-Only Warmup Iters so untrained critic NEVER blasts huge random value gradients into Phase 1's spatial trunk!
    opp_pool: list[ActuatorPolicyNetV3] = [copy.deepcopy(policy).eval()]

    # Precompute 50/50 West + East learning-bot flat indices across arenas [0..half_A-1] (West) & [half_A..A-1] (East)
    half_A = max(arena.A // 2, 1)
    rl_is_east = torch.zeros(arena.A, dtype=torch.bool, device=device)
    if arena.A > 1:
        rl_is_east[half_A:] = True
    learn_idx_list = []
    for a_i in range(arena.A):
        base = a_i * arena.N
        if rl_is_east[a_i]:
            learn_idx_list.extend(range(base + 6, base + 12))
        else:
            learn_idx_list.extend(range(base, base + 6))
    learn_flat_idx = torch.tensor(learn_idx_list, dtype=torch.long, device=device)  # [A * 6]

    # Critic-only warmup (2 fast iters, 100% actor frozen)
    critic_opt = torch.optim.AdamW(policy.critic.parameters(), lr=3.0e-3)
    last_closs = 0.0
    for cw in range(1, (2 if args.smoke_test else 3)):
        arena.reset(seed=800 + cw * 19)
        c_fine, c_coarse, c_sf, c_tok, c_mask, c_rew, c_done = [], [], [], [], [], [], []
        policy.eval()
        for step in range(args.rollout_steps):
            obs = arena.build_observations()
            with torch.inference_mode():
                fine_raw, coarse_raw, _ = policy.sample_raw_crops(arena.z_world, obs["pos_xyz"], arena_idx=arena.arena_idx_flat)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = policy.forward_from_crops(fine_raw, coarse_raw, obs["self_feat"], obs["tokens"], token_mask=obs["token_mask"])
                act_all, _, _ = policy.sample_action(out)
                act_3d = act_all.reshape(arena.A, arena.N, ACT3D_DIM)
                opp_act = scripted_3d_policy(arena)
                exec_west = torch.cat((act_3d[:, :6], opp_act[:, 6:]), dim=1)
                exec_east = torch.cat((opp_act[:, :6], act_3d[:, 6:]), dim=1)
                exec_act = torch.where(rl_is_east[:, None, None], exec_east, exec_west)
            rew_all, done = arena.step(exec_act)
            c_fine.append(fine_raw[learn_flat_idx].detach())
            c_coarse.append(coarse_raw[learn_flat_idx].detach())
            c_sf.append(obs["self_feat"][learn_flat_idx].detach())
            c_tok.append(obs["tokens"][learn_flat_idx].detach())
            c_mask.append(obs["token_mask"][learn_flat_idx].detach())
            c_rew.append(rew_all.reshape(-1)[learn_flat_idx].detach().float() * 0.25)
            c_done.append(torch.full((arena.A * 6,), float(done), device=device))
            if done:
                arena.reset(seed=850 + cw * 31 + step)
        T_c = len(c_rew)
        rews_c = torch.stack(c_rew, dim=0)
        dones_c = torch.stack(c_done, dim=0)
        rets_c = torch.zeros_like(rews_c)
        running_r = torch.zeros(arena.A * 6, device=device)
        for t in reversed(range(T_c)):
            running_r = rews_c[t] + 0.98 * running_r * (1.0 - dones_c[t])
            rets_c[t] = running_r
        fc_fine, fc_coarse = torch.cat(c_fine, 0), torch.cat(c_coarse, 0)
        fc_sf, fc_tok, fc_mask = torch.cat(c_sf, 0), torch.cat(c_tok, 0), torch.cat(c_mask, 0)
        fc_ret = rets_c.reshape(-1)
        policy.train()
        for _ in range(3):
            perm = torch.randperm(fc_ret.shape[0], device=device)
            for s in range(0, fc_ret.shape[0], args.minibatch):
                mb = perm[s : s + args.minibatch]
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        fine_crop = policy.z_adapter(fc_fine[mb].to(policy.actor_mean.weight.dtype))
                        coarse_crop = policy.z_adapter(fc_coarse[mb].to(policy.actor_mean.weight.dtype))
                        h_f, h_c = policy.fine_enc(fine_crop), policy.coarse_enc(coarse_crop)
                        N_mb = mb.shape[0]
                        p_dt = policy.actor_mean.weight.dtype
                        gv = torch.zeros((N_mb, policy.goal_dim), device=device, dtype=p_dt)
                        ev_z = torch.zeros((N_mb, 8), device=device, dtype=p_dt)
                        h_s = policy.self_proj(torch.cat((fc_sf[mb].to(p_dt), gv, ev_z), dim=-1))
                        h_t = policy.tok_proj(fc_tok[mb].to(p_dt))
                        ao, _ = policy.cross_attn(h_s.unsqueeze(1), h_t, h_t, key_padding_mask=~fc_mask[mb], need_weights=False)
                        h_a = policy.attn_norm(h_s + ao.squeeze(1))
                        h_trunk = policy.trunk(torch.cat((h_f, h_c, h_s, h_a), dim=-1)).float()
                v_pred = policy.critic(h_trunk).squeeze(-1)
                closs = F.smooth_l1_loss(v_pred, fc_ret[mb])
                critic_opt.zero_grad(set_to_none=True)
                closs.backward()
                critic_opt.step()
                last_closs = float(closs.item())
    print(f"[Phase 2] Critic pre-warmed (smooth_l1={last_closs:.3f})! Starting {args.iters}-iter Pure-Physics PPO League...", flush=True)

    # Differential LR: actor_mean + actor_log_std + critic adapt fast (2.6x), shared spatial trunk adapts smoothly (0.65x)
    head_params = list(policy.actor_mean.parameters()) + [policy.actor_log_std] + list(policy.critic.parameters())
    head_ids = {id(p) for p in head_params}
    trunk_params = [p for p in policy.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": trunk_params, "lr": args.lr * 0.65},
            {"params": head_params, "lr": args.lr * 2.6},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.iters, 1), eta_min=args.lr * 0.20)

    history: list[dict[str, float]] = []
    best_key = (-1.0, -1e9)  # (vs Scripted-3D WR, mean_score_diff)
    best_sd = copy.deepcopy(policy.state_dict())

    for it in range(1, args.iters + 1):
        arena.reset(seed=1000 + it * 17)
        # 50% vs full-strength Scripted-3D + 50% vs League Self-Play Snapshot Pool!
        use_scripted_opp = (it % 2 == 1)
        opp_net = opp_pool[(it // 2) % len(opp_pool)]

        buf_fine, buf_coarse, buf_sf, buf_tok, buf_mask = [], [], [], [], []
        buf_u, buf_lp, buf_val, buf_rew, buf_done = [], [], [], [], []

        policy.eval()
        for step in range(args.rollout_steps):
            obs = arena.build_observations()
            with torch.inference_mode():
                fine_raw, coarse_raw, _ = policy.sample_raw_crops(
                    arena.z_world, obs["pos_xyz"], arena_idx=arena.arena_idx_flat
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = policy.forward_from_crops(
                        fine_raw, coarse_raw, obs["self_feat"], obs["tokens"], token_mask=obs["token_mask"]
                    )
                act_all, u_all, lp_all = policy.sample_action(out)
                val_all = out["value"]

                act_3d = act_all.reshape(arena.A, arena.N, ACT3D_DIM)
                if use_scripted_opp:
                    opp_act = scripted_3d_policy(arena)
                else:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        opp_out = opp_net.forward_from_crops(
                            fine_raw, coarse_raw, obs["self_feat"], obs["tokens"], token_mask=obs["token_mask"]
                        )
                    opp_act = opp_out["mean"].float().reshape(arena.A, arena.N, ACT3D_DIM)

                # Side-balanced composition: half arenas RL=West, half arenas RL=East!
                exec_west = torch.cat((act_3d[:, :6], opp_act[:, 6:]), dim=1)
                exec_east = torch.cat((opp_act[:, :6], act_3d[:, 6:]), dim=1)
                exec_act = torch.where(rl_is_east[:, None, None], exec_east, exec_west)

            rew_all, done = arena.step(exec_act)
            # Clean reward scaling (* 0.25) keeps critic targets O(1..5) for rock-solid PPO!
            rew_flat = rew_all.reshape(-1)[learn_flat_idx] * 0.25

            buf_fine.append(fine_raw[learn_flat_idx].detach())
            buf_coarse.append(coarse_raw[learn_flat_idx].detach())
            buf_sf.append(obs["self_feat"][learn_flat_idx].detach())
            buf_tok.append(obs["tokens"][learn_flat_idx].detach())
            buf_mask.append(obs["token_mask"][learn_flat_idx].detach())
            buf_u.append(u_all[learn_flat_idx].detach().float())
            buf_lp.append(lp_all[learn_flat_idx].detach().float())
            buf_val.append(val_all[learn_flat_idx].detach().float())
            buf_rew.append(rew_flat.detach().float())
            buf_done.append(torch.full((arena.A * 6,), float(done), device=device))

            if done:
                arena.reset(seed=50000 + it * 101 + step)

        # GAE-Lambda advantage computation in strict float32
        T = len(buf_rew)
        B_w = arena.A * 6
        rews = torch.stack(buf_rew, dim=0)    # [T, B_w]
        vals = torch.stack(buf_val, dim=0)    # [T, B_w]
        dones = torch.stack(buf_done, dim=0)  # [T, B_w]
        adv = torch.zeros_like(rews)
        last_gae = torch.zeros(B_w, device=device)
        gamma, lam = 0.98, 0.95
        for t in reversed(range(T)):
            next_v = vals[t + 1] if t + 1 < T else torch.zeros(B_w, device=device)
            non_term = 1.0 - dones[t]
            delta = rews[t] + gamma * next_v * non_term - vals[t]
            last_gae = delta + gamma * lam * non_term * last_gae
            adv[t] = last_gae
        returns = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        f_fine = torch.cat(buf_fine, dim=0)
        f_coarse = torch.cat(buf_coarse, dim=0)
        f_sf = torch.cat(buf_sf, dim=0)
        f_tok = torch.cat(buf_tok, dim=0)
        f_mask = torch.cat(buf_mask, dim=0)
        f_u = torch.cat(buf_u, dim=0)
        f_lp = torch.cat(buf_lp, dim=0)
        f_adv = adv.reshape(-1)
        f_ret = returns.reshape(-1)

        policy.train()
        total_n = f_u.shape[0]
        frac = it / max(args.iters, 1)
        # Anneal exploration std cap smoothly (-0.40 -> -1.25) so deterministic mean MUST become sharp & decisive!
        ent_coef = max(0.0, 0.0010 * (1.0 - frac))
        max_log_std_phys = -0.40 - 0.85 * frac
        max_log_std_spat = -0.95 - 0.55 * frac
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
                        f_fine[mb], f_coarse[mb], f_sf[mb], f_tok[mb], token_mask=f_mask[mb]
                    )
                new_lp, ent = policy.evaluate_actions(out_mb, f_u[mb])
                log_ratio = (new_lp - f_lp[mb]).clamp(-5.0, 5.0)
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                if approx_kl > 0.05:
                    kl_stop = True
                    break
                surr1 = ratio * f_adv[mb]
                surr2 = ratio.clamp(0.80, 1.20) * f_adv[mb]
                pi_loss = -torch.minimum(surr1, surr2).mean()
                v_loss = F.smooth_l1_loss(out_mb["value"].float(), f_ret[mb])
                loss = pi_loss + 0.30 * v_loss - ent_coef * ent.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.65)
                optimizer.step()
                with torch.no_grad():
                    policy.actor_log_std[:6].clamp_(-2.0, max_log_std_spat)
                    policy.actor_log_std[6:].clamp_(-2.0, max_log_std_phys)
                last_loss = float(loss.item())
        scheduler.step()

        row: dict[str, float] = {
            "iter": float(it),
            "mean_return": float(rews.sum(dim=0).mean().item()),
            "loss": round(last_loss, 4),
        }

        eval_every = 1 if args.smoke_test else 10
        if it in (1, 5) or it % eval_every == 0 or it == args.iters:
            ev_n = 2 if args.smoke_test else min(16, args.eval_matches)
            ev = evaluate_matchup(policy, "scripted_3d", arena, num_matches=ev_n, base_seed=7000)
            row["eval_scripted_wr"] = ev["win_rate"]
            row["eval_score_diff"] = ev["mean_score_diff"]
            cur_key = (ev["win_rate"], ev["mean_score_diff"])
            if cur_key >= best_key:
                best_key = cur_key
                best_sd = copy.deepcopy(policy.state_dict())
            if len(opp_pool) >= 4:
                opp_pool.pop(0)  # keep most recent strong League snapshots (never stale passive dummies!)
            opp_pool.append(copy.deepcopy(policy).eval())
            print(
                f"[Iter {it:3d}/{args.iters}] opp={'Scr3D' if use_scripted_opp else 'Pool '} | ret={row['mean_return']:+6.2f} "
                f"| vs Scripted-3D WR={ev['win_rate']*100:5.1f}% (W:{ev['west_wins']}/{ev['west_matches']} E:{ev['east_wins']}/{ev['east_matches']} diff={ev['mean_score_diff']:+6.1f} best={best_key[0]*100:5.1f}%) "
                f"| tactics={ev['tactics_per_match']} | t={time.time()-t0:.1f}s",
                flush=True,
            )
        history.append(row)

    # Restore best checkpoint & run full 24-match side-balanced held-out 3D tournament
    policy.load_state_dict(best_sd)
    policy.eval()

    print("\n=== HELD-OUT 3D MULTI-BIOME TOURNAMENT (24 SIDE-BALANCED MATCHES, UNSEEN SEEDS) ===", flush=True)
    tournament: dict[str, Any] = {}
    for opp in ("scripted_3d", "flat_2d_ablation", "random_3d"):
        res = evaluate_matchup(policy, opp, arena, num_matches=args.eval_matches, base_seed=9100)
        tournament[opp] = res
        print(
            f"[Held-Out 3D Tournament] vs {opp:18s}: WR={res['win_rate']*100:5.1f}% "
            f"(W:{res['west_wins']}/{res['west_matches']} E:{res['east_wins']}/{res['east_matches']}) "
            f"| Elo={res['elo_delta']:+6.1f} | diff={res['mean_score_diff']:+6.2f} | tactics={res['tactics_per_match']}",
            flush=True,
        )

    if not args.smoke_test:
        args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_sd,
                "config": {
                    "arch": "ActuatorPolicyNetV3",
                    "z_world_ch": 48,
                    "z_eco_ch": 32,
                    "act_dim": ACT3D_DIM,
                    "self_dim": SELF3D_FEAT_DIM,
                    "tok_dim": TOKEN3D_FEAT_DIM,
                    "world_latent_mode": world_enc.mode,
                    "params": n_params,
                },
                "tournament": tournament,
            },
            str(args.out_ckpt),
        )
        print(f"[Save] Saved checkpoint -> {args.out_ckpt} ({args.out_ckpt.stat().st_size / 1024:.1f} KB)", flush=True)

        payload = {
            "model": "ActuatorPolicyNetV3",
            "wave": "14A",
            "params": n_params,
            "world_latent_mode": world_enc.mode,
            "device": str(device),
            "train_seconds": round(time.time() - t0, 1),
            "grid": [args.grid_xy, args.grid_xy, args.depth_z],
            "warmstart_steps": args.warmstart_steps,
            "iters": args.iters,
            "rollout_steps": args.rollout_steps,
            "num_arenas": args.num_arenas,
            "warmstart_history": ws_logs,
            "tournament": tournament,
            "history": history,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[Save] Saved metrics -> {args.out_json}", flush=True)
        save_viz(history, tournament, args.out_png)
    else:
        print(f"[Smoke Test] PASS in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
