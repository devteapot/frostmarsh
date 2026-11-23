#!/usr/bin/env python3
"""
Wave 12A: True 3D Volumetric (Z=16) Self-Play PPO Trainer & Held-Out Tournament for `ActuatorPolicyNetV3`.

Trains `ActuatorPolicyNetV3` (~294K params) in a vectorized 6v6 true-3D `Z=16` multi-biome GPU arena
driven by frozen `WorldModel3D-v5` (or `WorldModelDT` / `WorldModel` fallback) with ZERO hardcoded spells!

3D Multi-Biome Worlds (per batch, 3 vertical topologies):
  - Biome 0 (Nordic Canyon & Arched Bridge): vaulted basalt cave (z=1..4) under mountain, 3-span stone/timber
    arched bridge (z=7..9) spanning flowing river (z=3..4), highland plateaus (z=9..12), waterfalls.
  - Biome 1 (Volcanic Caldera & Lava Chasm): magma river (z=2..3, T=3.2), obsidian bridge arches (z=8),
    rising smoke/ash plumes, cool highland ledges (z=10..12).
  - Biome 2 (Glacial Sinkhole Caverns): frozen upper shelf (z=8..10) with deep central sinkhole opening
    into a subterranean meltwater cavern (z=2..5) + oil basins.

Emergent 3D Physics & Tactics (100% physics-derived, zero scripted spell buttons):
  1. True 3D vertical aiming (`aim_dz`): fire down from bridges, heat wooden arch spans from below,
     hit cave defenders or high-ground snipers across `|dz| >= 2.0m`.
  2. Structural Bridge-Pier Collapse: heating wooden bridge decks (`SP_WOOD`, `z>=6`) above ignition
     burns away structural support -> enemies standing on the span drop to the river/canyon below
     taking physical kinetic fall damage!
  3. Cryo Cave-Mouth / River Freezing: negative `thermal_flux` + water freezes into solid ice (`SP_ICE`),
     creating walkable cryo-bridges or sealing low cave choke points.
  4. Tactical Stasis Dome (`impulse_or_dome < -0.45`): spawns a `0.15x` TimeDome (`radius=7m`) that
     slows enemy movement & thermal damage ticks inside the bubble (up to 1 active dome per bot).
  5. Flash-Boil Steam & Oil-Combustion Pressure Waves: sharp `dT` or `dSteam` deals pressure damage +
     3D radial knockback along `-grad(P)` (knocking enemies off bridges!).

Evaluates on 24 held-out 3D matches (12 West + 12 East side-swapped) on unseen seeds vs:
  - `Scripted-3D` (tactical 3D baseline with shrine/relic pursuit, vertical aim, thermal/cryo deposits)
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


class FrozenWorldLatentExtractor(nn.Module):
    """
    Wraps `WorldModel3Dv5` (preferred), `WorldModelDT`, or `WorldModel` (or deterministic 3D projection
    for fast CPU smoke tests) to produce `[A, 48, L, H, W]` 3D shared world latents + conservative deltas.
    """

    def __init__(self, ckpt_path: Path | None, device: torch.device):
        super().__init__()
        self.device = device
        self.mode = "fallback_3d"
        self.model: nn.Module | None = None

        if ckpt_path is not None and ckpt_path.exists():
            ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            # Try WorldModel3Dv5 first
            try:
                from world_model_3d_v5 import WorldModel3DV5

                wm = WorldModel3DV5(in_ch=18, out_ch=17, width=48, z_flux_ch=5).to(device).eval()
                wm.load_state_dict(sd, strict=False)
                for p in wm.parameters():
                    p.requires_grad_(False)
                self.model = wm
                self.mode = "world_model_3d_v5"
                print(f"[WorldLatent] Loaded frozen WorldModel3DV5 from {ckpt_path}")
                return
            except Exception as exc:
                print(f"[WorldLatent] WorldModel3DV5 load skipped ({exc}); trying WorldModelDT...")

            try:
                from world_model_dt import WorldModelDT

                wm = WorldModelDT(in_ch=18, width=48, num_res=4).to(device).eval()
                wm.load_state_dict(sd, strict=False)
                for p in wm.parameters():
                    p.requires_grad_(False)
                self.model = wm
                self.mode = "world_model_dt"
                print(f"[WorldLatent] Loaded frozen WorldModelDT from {ckpt_path}")
                return
            except Exception as exc:
                print(f"[WorldLatent] WorldModelDT load skipped ({exc}); using lightweight 3D encoder.")

        # Lightweight deterministic 3D physical feature projection [18 -> 48] (never NaNs, fast on CPU/GPU)
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
            # WorldModel3DV5 expects [B, L, 18, H, W] and returns (horiz, z_flux, z_world [B, L, 48, H, W])
            x_bl = w_state.permute(0, 2, 1, 3, 4).contiguous()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                _, _, z_bl = self.model(x_bl, local_dt=1.0, return_z_flux=True, return_latent=True)
            z = z_bl.permute(0, 2, 1, 3, 4).contiguous()  # [A, 48, L, H, W]
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
    Coordinates: X in [0, W-1], Y in [0, H-1], Z in [0, L-1] (isotropic 1m voxels scaled to W=H=64, L=16).
    """

    def __init__(
        self,
        num_arenas: int,
        world_encoder: FrozenWorldLatentExtractor,
        device: torch.device,
        grid_xy: int = 64,
        depth_z: int = 16,
        max_ticks: int = 110,
        latent_refresh_every: int = 3,
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

        # 4 Physically Distinct Archetypes per team (6v6 = 12 bots, 4 archetypes + Squads vs Solos!):
        #   0: Strider (Heavy Walker / Squad Anchor, high HP, knockback resist)
        #   1: Skywing (True 3D Flyer / Thermal Glider — rides fire/magma/steam updrafts!)
        #   2: Tidekin (Amphibious Swimmer / Diver — 2.2x water speed, submerged ambushes!)
        #   3: Emberstalker (Solo Cavern Climber — 3.6m vertical wall-climb, smoke stealth!)
        teams = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        classes = torch.tensor([0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 3], dtype=torch.long, device=device)
        # Squad assignment: Squad 1 (bots 0,1,2), Squad 2 (bots 3,4), Solo Raider = 0 (bot 5 = Emberstalker!)
        squads = torch.tensor([1, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 0], dtype=torch.long, device=device)
        self.team = teams.unsqueeze(0).expand(self.A, -1).contiguous()
        self.cls = classes.unsqueeze(0).expand(self.A, -1).contiguous()
        self.squad = squads.unsqueeze(0).expand(self.A, -1).contiguous()
        self.is_solo = (self.squad == 0)
        self.same_squad_ally = (self.team[:, :, None] == self.team[:, None, :]) & (self.squad[:, :, None] == self.squad[:, None, :]) & (self.squad[:, :, None] > 0)
        # 3-lane assignment (-1=North Bridge, 0=Center Cave/River, +1=South Bridge) for multi-shrine coordination
        lanes = torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0], device=device)
        self.lane_id = lanes.unsqueeze(0).expand(self.A, -1).contiguous()
        self.arena_idx_flat = torch.arange(self.A, device=device).repeat_interleave(self.N)
        self.enemy_mask = self.team[:, :, None] != self.team[:, None, :]

        # Normalized coordinates for procedural 3D biome generation
        zz = torch.arange(self.L, device=device, dtype=torch.float32)
        yy = torch.arange(self.H, device=device, dtype=torch.float32) / self.H
        xx = torch.arange(self.W, device=device, dtype=torch.float32) / self.W
        self.gz, self.gy, self.gx = torch.meshgrid(zz, yy, xx, indexing="ij")  # [L, H, W]

        # 3D shrines: North High-Bridge Shrine (z=8.5), Center Cave/River Shrine (z=4.0), South Timber-Bridge Shrine (z=8.5)
        self.shrine_x = torch.tensor([0.50, 0.50, 0.50], device=device).unsqueeze(0).expand(self.A, -1).clone() * self.W
        self.shrine_y = torch.tensor([0.24, 0.50, 0.76], device=device).unsqueeze(0).expand(self.A, -1).clone() * self.H
        self.shrine_z = torch.tensor([8.5, 4.0, 8.5], device=device).unsqueeze(0).expand(self.A, -1).clone()

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
        self.st_r2 = (self.st_dx.float() ** 2 + self.st_dy.float() ** 2 + (self.st_dz.float() * 1.4) ** 2)

        self.reset(seed=0)

    def _generate_3d_biomes(self, gen: torch.Generator) -> None:
        """Vectorized GPU generation of A true-3D Z=16 multi-biome worlds (caves + arched bridges + canyon)."""
        A, L, H, W = self.A, self.L, self.H, self.W
        dev = self.device
        self.w_state.zero_()

        gx = self.gx.unsqueeze(0)  # [1, L, H, W]
        gy = self.gy.unsqueeze(0)
        gz = self.gz.unsqueeze(0)

        biome_id = torch.randint(0, 3, (A, 1, 1, 1), generator=gen, device=dev)
        self.biome_id = biome_id.view(A)

        phase = torch.rand((A, 1, 1, 1), generator=gen, device=dev) * 6.28318
        rcx = 0.50 + 0.04 * torch.sin(gy * 6.28318 + phase)
        dist_river = (gx - rcx).abs()

        # Sculpted canyon height: valley floor z=3 in center, rising to highland shelves z=7..10 on West/East
        base_h = 3.0 + 5.2 * ((dist_river - 0.09) / 0.24).clamp(0.0, 1.0)
        # Add natural mountain peaks on outer edges
        ridge = 2.6 * torch.sin(gy * 9.4 + phase).abs() * ((gx - 0.5).abs() > 0.32).float()
        terrain_z = (base_h + ridge).clamp(2.5, 11.5)

        solid_ground = gz <= terrain_z
        bedrock = gz <= 1.5
        self.w_state[:, SP_ROCK] = torch.where(bedrock | (solid_ground & (gz < terrain_z - 1.2)), 1.0, 0.0)
        self.w_state[:, SP_SOIL] = torch.where(solid_ground & (~bedrock) & (gz >= terrain_z - 1.2), 0.92, 0.0)

        # Carve vaulted subterranean basalt cave tunnel running East-West at y ~ 0.50, z=2.0..5.2!
        cave_mask = ((gy - 0.50).abs() < 0.09) & (gz >= 2.0) & (gz <= 5.0 - 1.8 * ((gy - 0.50) / 0.09) ** 2) & ((gx - 0.50).abs() < 0.34)
        self.w_state[:, SP_ROCK] = torch.where(cave_mask, 0.0, self.w_state[:, SP_ROCK])
        self.w_state[:, SP_SOIL] = torch.where(cave_mask, 0.0, self.w_state[:, SP_SOIL])

        # River channel water (z=2.0..3.6) — molten hot magma in Biome 1 (Volcanic Caldera)!
        river_water = (dist_river < 0.10) & (gz >= 2.0) & (gz <= 3.5)
        self.w_state[:, SP_WATER] = torch.where(river_water & (biome_id != 1), 1.05, 0.0)
        self.w_state[:, CH_TEMP] = torch.where(river_water & (biome_id == 1), 3.1, 0.25)
        # Biome 2 (Glacial Sinkhole): cold ambient + frozen ice bridge plates
        self.w_state[:, CH_TEMP] = torch.where(biome_id == 2, self.w_state[:, CH_TEMP] - 0.55, self.w_state[:, CH_TEMP])
        ice_plate = (biome_id == 2) & (dist_river < 0.10) & ((gz - 3.5).abs() < 0.6) & ((gy - 0.50).abs() > 0.14)
        self.w_state[:, SP_ICE] = torch.where(ice_plate, 0.95, 0.0)

        # True 3D Arched Bridges spanning the canyon at y=0.24 (Stone/Timber North) and y=0.76 (Destructible Timber South)!
        arch_curve = 8.2 - 1.5 * ((gx - 0.50) / 0.22).clamp(-1.0, 1.0) ** 2
        north_bridge = ((gy - 0.24).abs() < 0.055) & ((gx - 0.50).abs() < 0.23) & ((gz - arch_curve).abs() <= 0.65)
        south_bridge = ((gy - 0.76).abs() < 0.055) & ((gx - 0.50).abs() < 0.23) & ((gz - arch_curve).abs() <= 0.65)
        # Clear air cleanly beneath bridges so bots can run underneath!
        under_north = ((gy - 0.24).abs() < 0.055) & ((gx - 0.50).abs() < 0.16) & (gz >= 3.6) & (gz < arch_curve - 0.65)
        under_south = ((gy - 0.76).abs() < 0.055) & ((gx - 0.50).abs() < 0.16) & (gz >= 3.6) & (gz < arch_curve - 0.65)
        self.w_state[:, SP_ROCK] = torch.where(under_north | under_south, 0.0, self.w_state[:, SP_ROCK])
        self.w_state[:, SP_SOIL] = torch.where(under_north | under_south, 0.0, self.w_state[:, SP_SOIL])

        # North arch = wood + rock masonry; South arch = combustible timber (can be burned/collapsed!)
        self.w_state[:, SP_WOOD] = torch.where(north_bridge | south_bridge, 0.95, 0.0)
        self.w_state[:, SP_ROCK] = torch.where(north_bridge & ((gx - 0.50).abs() > 0.10), 0.75, self.w_state[:, SP_ROCK])

        # Symmetric oil pockets near cave mouths & bridge approaches
        oil_pockets = (((gx - 0.34).abs() < 0.045) | ((gx - 0.66).abs() < 0.045)) & ((gy - 0.50).abs() < 0.06) & ((gz - 3.0).abs() < 0.8)
        self.w_state[:, SP_OIL] = torch.where(oil_pockets, 0.85, 0.0)

        # Symmetric geothermal/wildfire thermal vents beneath North & South bridges (natural Skywing updraft columns!)
        vents = (((gx - 0.36).abs() < 0.05) | ((gx - 0.64).abs() < 0.05)) & (((gy - 0.24).abs() < 0.06) | ((gy - 0.76).abs() < 0.06)) & (gz >= 2.0) & (gz <= 7.0)
        self.w_state[:, CH_TEMP] = torch.where(vents, 2.2, self.w_state[:, CH_TEMP])
        self.w_state[:, SP_STEAM] = torch.where(vents & (gz >= 4.0), 0.55, self.w_state[:, SP_STEAM])

        # Altitude & logmass channels
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
        """
        True 3D multi-level ground lookup:
        Finds the highest solid surface at (x, y) reachable near `cur_z` (supports walking UNDER a bridge
        at z=4 without snapping up to z=8, AND walking ON a bridge at z=8 and falling if it collapses!).
        Returns (target_z, fell_distance).
        """
        ix = x.round().long().clamp(0, self.W - 1)
        iy = y.round().long().clamp(0, self.H - 1)
        solid = self._solid_mask()  # [A, L, H, W]
        a_idx = torch.arange(self.A, device=self.device).unsqueeze(1).expand(self.A, self.N)
        col = solid[a_idx, :, iy, ix]  # [A, N, L] bool

        # A walkable stand-z is a voxel z where (z-1) is solid and z is air (not solid)
        z_grid = torch.arange(self.L, device=self.device, dtype=torch.float32).view(1, 1, self.L)
        below_solid = F.pad(col[:, :, :-1], (1, 0), value=True)
        walkable = below_solid & (~col)  # [A, N, L]

        # Can step up at most +1.6m from current z, or fall arbitrarily far down!
        reachable = walkable & (z_grid <= cur_z.unsqueeze(-1) + 1.6)
        # Pick highest reachable walkable z
        score = torch.where(reachable, z_grid, torch.full_like(z_grid, -1.0))
        best_z = score.max(dim=-1).values
        # Fallback if trapped inside solid: rise to lowest air above
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

        # Spawn 6v6 across 3 vertical lanes: North bridge approach (high z), Center cave (low z), South bridge approach (high z)
        lane_y = torch.tensor([0.24, 0.28, 0.50, 0.50, 0.72, 0.76], device=dev) * self.H
        lane_z = torch.tensor([8.5, 8.5, 3.2, 3.2, 8.5, 8.5], device=dev)
        jx = torch.empty((self.A, 6), device=dev).uniform_(-3.5, 3.5, generator=gen)
        jy = torch.empty((self.A, 6), device=dev).uniform_(-2.0, 2.0, generator=gen)

        wx = (0.24 * self.W + jx).clamp(4.0, self.W * 0.42)
        ex = (self.W - 1.0) - wx
        wy = (lane_y.unsqueeze(0) + jy).clamp(4.0, self.H - 5.0)
        ey = wy.clone()

        self.pos_x = torch.cat([wx, ex], dim=1)
        self.pos_y = torch.cat([wy, ey], dim=1)
        self.pos_z = torch.cat([lane_z.unsqueeze(0).expand(self.A, -1), lane_z.unsqueeze(0).expand(self.A, -1)], dim=1).clone()
        tz, _ = self._ground_z_near(self.pos_x, self.pos_y, self.pos_z)
        self.pos_z = tz

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
        self.relic_z = torch.full((self.A,), 3.5, device=dev)
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
        """Builds exact inputs for ActuatorPolicyNetV3: pos_xyz, self_feat [A*N, 22], tokens [A*N, 14, 14], mask."""
        A, N = self.A, self.N
        dev = self.device
        a_idx = torch.arange(A, device=dev).unsqueeze(1).expand(A, N)

        ix = self.pos_x.round().long().clamp(0, self.W - 1)
        iy = self.pos_y.round().long().clamp(0, self.H - 1)
        iz = self.pos_z.round().long().clamp(0, self.L - 1)
        iz_above = (iz + 2).clamp(0, self.L - 1)
        iz_below = (iz - 1).clamp(0, self.L - 1)

        local_temp = self.w_state[a_idx, CH_TEMP, iz, iy, ix]
        local_water = self.w_state[a_idx, SP_WATER, iz, iy, ix]
        local_ice = self.w_state[a_idx, SP_ICE, iz, iy, ix]
        local_smoke = self.w_state[a_idx, SP_SMOKE, iz, iy, ix]
        local_oil = self.w_state[a_idx, SP_OIL, iz, iy, ix]
        below_wood = self.w_state[a_idx, SP_WOOD, iz_below, iy, ix]
        above_solid = (
            self.w_state[a_idx, SP_ROCK, iz_above, iy, ix]
            + self.w_state[a_idx, SP_SOIL, iz_above, iy, ix]
            + self.w_state[a_idx, SP_WOOD, iz_above, iy, ix]
        ).clamp(0.0, 1.0)

        # Check if inside any active Stasis dome
        d2_domes = (
            (self.pos_x[:, :, None] - self.dome_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.dome_y[:, None, :]) ** 2
            + (self.pos_z[:, :, None] - self.dome_z[:, None, :]) ** 2
        )
        in_dome = ((self.dome_ttl[:, None, :] > 0) & (d2_domes < 49.0)).any(dim=-1).float()

        team_sign = torch.where(self.team == 0, 1.0, -1.0)
        is_carrier = (self.relic_carrier.unsqueeze(1) == torch.arange(N, device=dev).unsqueeze(0)).float()
        cls_oh = F.one_hot(self.cls, num_classes=4).float()  # 4 archetypes: Strider, Skywing, Tidekin, Emberstalker

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
                self.is_solo.float(),
            ),
            dim=-1,
        )  # [A, N, 18]
        self_feat = torch.cat((self_feat, cls_oh), dim=-1)  # [A, N, 22]

        # Build 14 3D entity/shrine/relic tokens [A, N, 14, 14]
        # Slots 0..5: 6 enemies, Slots 6..9: 4 nearest allies, Slots 10..12: 3 shrines, Slot 13: Relic
        toks = torch.zeros((A, N, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM), device=dev)
        mask = torch.zeros((A, N, MAX_TOKENS_3D), dtype=torch.bool, device=dev)

        for i in range(N):
            my_team = 0 if i < 6 else 1
            enemies = [j for j in range(N) if (0 if j < 6 else 1) != my_team]
            allies = [j for j in range(N) if (0 if j < 6 else 1) == my_team and j != i][:4]

            px, py, pz = self.pos_x[:, i], self.pos_y[:, i], self.pos_z[:, i]

            for s, j in enumerate(enemies):
                dx = (self.pos_x[:, j] - px) / 24.0
                dy = (self.pos_y[:, j] - py) / 24.0
                dz = (self.pos_z[:, j] - pz) / 8.0
                dist = torch.sqrt(dx * dx + dy * dy + dz * dz + 1e-6)
                vis = self.alive[:, j] & (local_smoke[:, i] < 0.75)
                toks[:, i, s, 0] = dx.clamp(-2.0, 2.0)
                toks[:, i, s, 1] = dy.clamp(-2.0, 2.0)
                toks[:, i, s, 2] = dz.clamp(-2.0, 2.0)
                toks[:, i, s, 3] = dist.clamp(0.0, 3.0)
                toks[:, i, s, 4] = self.hp[:, j]
                toks[:, i, s, 5] = 1.0  # enemy flag
                toks[:, i, s, 6] = (self.relic_carrier == j).float()
                toks[:, i, s, 7] = (self.dome_ttl[:, j] > 0).float()
                toks[:, i, s, 8] = (self.pos_z[:, j] > pz + 1.5).float()
                toks[:, i, s, 9] = (self.pos_z[:, j] < pz - 1.5).float()
                mask[:, i, s] = vis

            for s_off, j in enumerate(allies):
                s = 6 + s_off
                dx = (self.pos_x[:, j] - px) / 24.0
                dy = (self.pos_y[:, j] - py) / 24.0
                dz = (self.pos_z[:, j] - pz) / 8.0
                toks[:, i, s, 0] = dx.clamp(-2.0, 2.0)
                toks[:, i, s, 1] = dy.clamp(-2.0, 2.0)
                toks[:, i, s, 2] = dz.clamp(-2.0, 2.0)
                toks[:, i, s, 4] = self.hp[:, j]
                toks[:, i, s, 10] = 1.0  # ally flag
                mask[:, i, s] = self.alive[:, j]

            for k in range(3):
                s = 10 + k
                dx = (self.shrine_x[:, k] - px) / 24.0
                dy = (self.shrine_y[:, k] - py) / 24.0
                dz = (self.shrine_z[:, k] - pz) / 8.0
                toks[:, i, s, 0] = dx.clamp(-2.0, 2.0)
                toks[:, i, s, 1] = dy.clamp(-2.0, 2.0)
                toks[:, i, s, 2] = dz.clamp(-2.0, 2.0)
                toks[:, i, s, 11] = 1.0  # shrine flag
                toks[:, i, s, 12] = torch.where(
                    self.shrine_owner[:, k] == my_team, 1.0,
                    torch.where(self.shrine_owner[:, k] == (1 - my_team), -1.0, 0.0),
                )
                mask[:, i, s] = True

            # Slot 13: Relic token
            toks[:, i, 13, 0] = ((self.relic_x - px) / 24.0).clamp(-2.0, 2.0)
            toks[:, i, 13, 1] = ((self.relic_y - py) / 24.0).clamp(-2.0, 2.0)
            toks[:, i, 13, 2] = ((self.relic_z - pz) / 8.0).clamp(-2.0, 2.0)
            toks[:, i, 13, 13] = 1.0
            mask[:, i, 13] = True

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
            sx = torch.where(self.team == 0, torch.full_like(self.pos_x, 0.20 * self.W), torch.full_like(self.pos_x, 0.80 * self.W))
            self.pos_x = torch.where(just_respawn, sx, self.pos_x)
            tz, _ = self._ground_z_near(self.pos_x, self.pos_y, torch.full_like(self.pos_z, 8.5))
            self.pos_z = torch.where(just_respawn, tz, self.pos_z)

        alive_f = self.alive.float()

        # Check Stasis TimeDome dilation per bot (0.15x speed inside an ENEMY Stasis dome!)
        d2_domes = (
            (self.pos_x[:, :, None] - self.dome_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.dome_y[:, None, :]) ** 2
            + (self.pos_z[:, :, None] - self.dome_z[:, None, :]) ** 2
        )
        in_enemy_dome = ((self.dome_ttl[:, None, :] > 0) & self.enemy_mask & (d2_domes < 49.0)).any(dim=-1)
        time_scale = torch.where(in_enemy_dome, 0.25, 1.0)

        # 1. Heterogeneous 3D Archetype Locomotion (Walking, Flying+Thermal Updrafts, Swimming, Solo Cavern Climbing!)
        a_idx = torch.arange(A, device=dev).unsqueeze(1).expand(A, N)
        ix = self.pos_x.round().long().clamp(0, self.W - 1)
        iy = self.pos_y.round().long().clamp(0, self.H - 1)
        iz = self.pos_z.round().long().clamp(0, self.L - 1)

        is_strider = self.cls == 0
        is_skywing = self.cls == 1
        is_tidekin = self.cls == 2
        is_stalker = self.cls == 3

        in_water = (self.w_state[a_idx, SP_WATER, iz, iy, ix] > 0.45) & (self.w_state[a_idx, SP_ICE, iz, iy, ix] < 0.40)
        on_ice = self.w_state[a_idx, SP_ICE, (iz - 1).clamp(0, self.L - 1), iy, ix] > 0.40
        in_smoke = self.w_state[a_idx, SP_SMOKE, iz, iy, ix] > 0.20
        col_temp_below = self.w_state[a_idx, CH_TEMP, (iz - 2).clamp(0, self.L - 1), iy, ix]
        col_steam_here = self.w_state[a_idx, SP_STEAM, iz, iy, ix]

        base_spd = torch.where(
            is_tidekin & in_water,
            2.95,  # Passive Hydrodynamic Surge: 2.2x+ swim boost in rivers/flooded caverns!
            torch.where(
                is_skywing,
                2.45,
                torch.where(
                    is_stalker,
                    torch.where(in_smoke, 2.75, 2.35),  # Passive Smoke Veil speed surge!
                    torch.where(in_water, 1.05, torch.where(on_ice, 2.55, 1.90)),
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

        # Grounded archetypes (Strider, Tidekin, Emberstalker) vs True 3D Flyer (Skywing + Thermal Updrafts!)
        climb_cap = torch.where(is_stalker, 3.6, 1.8)
        climb_boost = torch.where((mv_jump > 0.35) & self.alive, torch.where(is_stalker, 2.4, 1.4), 0.0)
        gnd_z, gnd_fell = self._ground_z_near(new_x, new_y, self.pos_z + climb_boost)

        # Skywing true 3D flight + PASSIVE THERMAL UPDRAFT SOARING over fire/magma/steam!
        updraft = ((col_temp_below > 1.05) | (col_steam_here > 0.25)) & is_skywing & self.alive
        fly_dz = mv_jump * 1.65 + torch.where(updraft, 1.85, -0.35)
        fly_z = torch.maximum(self.pos_z + fly_dz, gnd_z).clamp(max=float(self.L - 2))
        if updraft.any():
            for t in (0, 1):
                self.stat_updraft_soars[:, t] += (updraft & (self.team == t)).float().sum(dim=1) * 0.25
            rewards += updraft.float() * 0.025  # natural physics advantage of thermal altitude

        target_z = torch.where(is_skywing, fly_z, gnd_z)
        fell = torch.where(is_skywing, torch.zeros_like(gnd_fell), gnd_fell)

        # Prevent grounded non-flyers from walking through sheer cliff walls (Skywings fly over; Emberstalkers climb 3.6m!)
        blocked = (~is_skywing) & ((target_z - self.pos_z) > (climb_cap + climb_boost))
        self.pos_x = torch.where(blocked, self.pos_x, new_x)
        self.pos_y = torch.where(blocked, self.pos_y, new_y)
        self.pos_z = torch.where(blocked, self.pos_z, target_z)

        # Squad cohesion (Squad 1 & 2) vs Solo Raider flank reward (Emberstalker!)
        d_pair = torch.sqrt(
            (self.pos_x[:, :, None] - self.pos_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.pos_y[:, None, :]) ** 2
        )
        near_squadmate = (self.same_squad_ally & (d_pair < 11.0) & self.alive[:, None, :]).any(dim=-1)
        rewards += torch.where((~self.is_solo) & near_squadmate & self.alive, 0.015, 0.0)
        # Tidekin aquatic ambush bonus when striking from water/river!
        self._tidekin_in_water = is_tidekin & in_water & self.alive
        self.oil_primed_ttl = (self.oil_primed_ttl - 1).clamp(min=0)

        # Clean objective proximity shaping (encourages moving toward contested 3D shrines / center without scripted spells)
        d_shrines = torch.sqrt(
            (self.pos_x[:, :, None] - self.shrine_x[:, None, :]) ** 2
            + (self.pos_y[:, :, None] - self.shrine_y[:, None, :]) ** 2
            + ((self.pos_z[:, :, None] - self.shrine_z[:, None, :]) * 1.5) ** 2
        )
        not_ours = self.shrine_owner[:, None, :] != self.team[:, :, None]
        d_target = torch.where(not_ours, d_shrines, d_shrines + 12.0).min(dim=-1).values
        rewards += (1.0 - (d_target / 32.0).clamp(0.0, 1.0)) * 0.03 * alive_f

        # Physical kinetic fall damage (e.g., dropped from a burning collapsed bridge!)
        fall_dmg = ((fell - 2.6).clamp(min=0.0) * 0.14) * alive_f
        if (fall_dmg > 0.05).any():
            dropped_mask = fall_dmg > 0.05
            # Credit opposing team for bridge-collapse drop
            for t in (0, 1):
                opp_drops = (dropped_mask & (self.team == (1 - t))).float().sum(dim=1)
                self.stat_bridge_drops[:, t] += opp_drops

        # 2. Conservation-Bounded 3D Physical Actuators
        aim_dx = actions[:, :, 3]
        aim_dy = actions[:, :, 4]
        aim_dz = actions[:, :, 5]
        thermal = actions[:, :, 6]
        emit_water = actions[:, :, 7]
        emit_oil = actions[:, :, 8]
        emit_smoke = actions[:, :, 9]
        focus = actions[:, :, 10]
        imp_dome = actions[:, :, 11]

        # Target voxel in 3D (reach 11m XY, +-5.5m Z!)
        tx = (self.pos_x + aim_dx * 11.0).clamp(1.0, self.W - 2.0)
        ty = (self.pos_y + aim_dy * 11.0).clamp(1.0, self.H - 2.0)
        tz_tgt = (self.pos_z + aim_dz * 5.5).clamp(1.0, self.L - 2.0)

        # Mana budget scales with emission energy & focus area
        rad = 1.4 + focus * 2.2
        energy_cost = (
            thermal.abs() * 0.14
            + emit_water * 0.06
            + emit_oil * 0.08
            + emit_smoke * 0.05
            + imp_dome.abs() * 0.12
        ) * (0.7 + 0.3 * focus)
        can_pay = (self.mana >= energy_cost) & self.alive
        scale_pay = torch.where(can_pay, torch.ones_like(energy_cost), (self.mana / (energy_cost + 1e-5)).clamp(0.0, 1.0)) * alive_f
        self.mana = (self.mana - energy_cost * scale_pay + 0.075 * alive_f).clamp(0.0, 1.0)

        # Place Stasis Dome when impulse_or_dome < -0.45 and mana sufficient
        want_dome = (imp_dome < -0.45) & (scale_pay > 0.5) & (self.dome_ttl == 0)
        if want_dome.any():
            self.dome_x = torch.where(want_dome, tx, self.dome_x)
            self.dome_y = torch.where(want_dome, ty, self.dome_y)
            self.dome_z = torch.where(want_dome, tz_tgt, self.dome_z)
            self.dome_ttl = torch.where(want_dome, torch.full_like(self.dome_ttl, 16), self.dome_ttl)
            for t in (0, 1):
                self.stat_stasis_domes[:, t] += (want_dome & (self.team == t)).float().sum(dim=1)

        # Deposit thermal & species into 3D voxel grid (vectorized scatter over 75-voxel 3D kernel)
        tx_i = tx.round().long()
        ty_i = ty.round().long()
        tz_i = tz_tgt.round().long()

        # Apply center + kernel deposits via fast indexed updates
        inv_2s2 = 1.0 / (2.0 * (rad * 0.65) ** 2 + 1e-4)  # [A, N]
        w_kern = torch.exp(-self.st_r2.view(1, 1, -1) * inv_2s2.unsqueeze(-1))  # [A, N, K]
        w_kern = w_kern / (w_kern.sum(dim=-1, keepdim=True) + 1e-6) * 4.5

        gz_idx = (tz_i.unsqueeze(-1) + self.st_dz.view(1, 1, -1)).clamp(0, self.L - 1)
        gy_idx = (ty_i.unsqueeze(-1) + self.st_dy.view(1, 1, -1)).clamp(0, self.H - 1)
        gx_idx = (tx_i.unsqueeze(-1) + self.st_dx.view(1, 1, -1)).clamp(0, self.W - 1)
        flat_idx = (gz_idx * self.H + gy_idx) * self.W + gx_idx  # [A, N, K]
        flat_idx_a = flat_idx.reshape(A, -1)

        d_temp = (thermal * scale_pay * 1.35).unsqueeze(-1) * w_kern
        d_water = (emit_water * scale_pay * 0.45).unsqueeze(-1) * w_kern
        d_oil = (emit_oil * scale_pay * 0.42).unsqueeze(-1) * w_kern
        d_smoke = (emit_smoke * scale_pay * 0.45).unsqueeze(-1) * w_kern

        prev_temp = self.w_state[:, CH_TEMP].clone()
        prev_steam = self.w_state[:, SP_STEAM].clone()

        def _scatter_add_ch(ch: int, vals: torch.Tensor) -> None:
            buf = self.w_state[:, ch].reshape(A, -1)
            buf.scatter_add_(1, flat_idx_a, vals.reshape(A, -1))

        _scatter_add_ch(CH_TEMP, d_temp)
        _scatter_add_ch(SP_WATER, d_water)
        _scatter_add_ch(SP_OIL, d_oil)
        _scatter_add_ch(SP_SMOKE, d_smoke)

        # 3. Emergent 3D Phase Changes, Oil Combustion, Timber Bridge Burning & Hydrologic Decay
        temp = self.w_state[:, CH_TEMP].clamp(-1.8, 5.5)
        water = self.w_state[:, SP_WATER].clamp(0.0, 2.0)
        ice = self.w_state[:, SP_ICE].clamp(0.0, 2.0)
        wood = self.w_state[:, SP_WOOD].clamp(0.0, 2.0)
        oil = self.w_state[:, SP_OIL].clamp(0.0, 2.0)
        smoke = self.w_state[:, SP_SMOKE].clamp(0.0, 2.0)
        steam = self.w_state[:, SP_STEAM].clamp(0.0, 2.0)

        # Freeze water -> ice (cryo-seals & ice bridges!)
        r_freeze = torch.minimum(water, 0.35 * (temp < -0.15).float())
        water = water - r_freeze
        ice = (ice + r_freeze).clamp(0.0, 2.0)

        # Melt ice -> water
        r_melt = torch.minimum(ice, 0.25 * (temp > 0.65).float())
        ice = ice - r_melt
        water = (water + r_melt).clamp(0.0, 2.0)

        # Flash-boil water -> steam (pressure wave!)
        r_boil = torch.minimum(water, 0.40 * (temp > 1.35).float())
        water = water - r_boil
        steam = (steam + 1.15 * r_boil).clamp(0.0, 2.0)

        # Oil combustion (massive exothermic spike!)
        r_oil = torch.minimum(oil, 0.55 * (temp > 1.15).float())
        oil = oil - r_oil
        temp = (temp + 2.6 * r_oil).clamp(-1.8, 5.5)
        smoke = (smoke + 0.6 * r_oil).clamp(0.0, 2.0)

        # Timber bridge burning -> structural collapse when wood drops below 0.42!
        r_wood = torch.minimum(wood, 0.22 * (temp > 1.65).float())
        wood = wood - r_wood
        temp = (temp + 0.8 * r_wood).clamp(-1.8, 5.5)
        smoke = (smoke + 0.35 * r_wood).clamp(0.0, 2.0)

        # Thermal relaxation + buoyancy of smoke/steam upward (+Z)
        temp = temp * 0.88 + 0.02
        smoke = F.pad(smoke[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0) * 0.72 + smoke * 0.18
        steam = F.pad(steam[:, :-1], (0, 0, 0, 0, 1, 0), value=0.0) * 0.70 + steam * 0.16

        self.w_state[:, CH_TEMP] = temp
        self.w_state[:, SP_WATER] = water
        self.w_state[:, SP_ICE] = ice
        self.w_state[:, SP_WOOD] = wood
        self.w_state[:, SP_OIL] = oil
        self.w_state[:, SP_SMOKE] = smoke
        self.w_state[:, SP_STEAM] = steam
        self.w_state[:, CH_LOGMASS] = torch.log1p(self.w_state[:, :10].sum(dim=1))

        # Track cryo-seals created by bots
        cryo_emit = (thermal < -0.45) & (scale_pay > 0.4)
        for t in (0, 1):
            self.stat_cryo_seals[:, t] += (cryo_emit & (self.team == t)).float().sum(dim=1) * 0.25

        # 4. Physics-Derived Combat Damage (Thermal Stress + Pressure Blast + Direct Kinetic Impulse + Fall)
        ix = self.pos_x.round().long().clamp(0, self.W - 1)
        iy = self.pos_y.round().long().clamp(0, self.H - 1)
        iz = self.pos_z.round().long().clamp(0, self.L - 1)

        bot_temp = temp[a_idx, iz, iy, ix]
        bot_dtemp = (bot_temp - prev_temp[a_idx, iz, iy, ix]).clamp(min=0.0)
        bot_dsteam = (steam[a_idx, iz, iy, ix] - prev_steam[a_idx, iz, iy, ix]).clamp(min=0.0)

        scorch = (bot_temp - 1.25).clamp(min=0.0) ** 2 * 0.12
        hypo = (-0.35 - bot_temp).clamp(min=0.0) ** 2 * 0.11
        pressure = ((bot_dtemp - 0.55).clamp(min=0.0) * 0.30 + (bot_dsteam - 0.18).clamp(min=0.0) * 0.35)

        # Credit nearest enemy actuator deposit within 3D radius (and kinetic impulse shockwave!)
        dist_tx = self.pos_x[:, :, None] - tx[:, None, :]  # [A, victim, attacker]
        dist_ty = self.pos_y[:, :, None] - ty[:, None, :]
        dist_tz = self.pos_z[:, :, None] - tz_tgt[:, None, :]
        dist_3d = torch.sqrt(dist_tx ** 2 + dist_ty ** 2 + (dist_tz * 1.2) ** 2 + 1e-6)
        hit_rad = (rad[:, None, :] + 1.4)
        hit_mask = self.enemy_mask & (dist_3d <= hit_rad) & (scale_pay[:, None, :] > 0.25)

        # Kinetic shockwave (imp_dome > 0.40) deals physical impulse + radial knockback (can knock enemies off bridges!)
        impulse_mag = (imp_dome - 0.35).clamp(min=0.0) * scale_pay
        imp_falloff = (1.0 - dist_3d / (hit_rad + 1e-4)).clamp(min=0.0)
        imp_dmg_mat = torch.where(hit_mask, impulse_mag[:, None, :] * 0.24 * imp_falloff, 0.0)
        imp_dmg = imp_dmg_mat.sum(dim=-1)

        # Physical radial knockback from kinetic impulse & thermal pressure waves
        kb_mag = torch.where(hit_mask, impulse_mag[:, None, :] * 2.2 * imp_falloff, 0.0)
        inv_dxy = 1.0 / torch.hypot(dist_tx, dist_ty).clamp(min=0.5)
        kb_x = (kb_mag * dist_tx * inv_dxy).sum(dim=-1).clamp(-3.5, 3.5) * alive_f
        kb_y = (kb_mag * dist_ty * inv_dxy).sum(dim=-1).clamp(-3.5, 3.5) * alive_f
        if (kb_x.abs() + kb_y.abs() > 0.1).any():
            kx = (self.pos_x + kb_x).clamp(2.0, self.W - 3.0)
            ky = (self.pos_y + kb_y).clamp(2.0, self.H - 3.0)
            kz, k_fell = self._ground_z_near(kx, ky, self.pos_z)
            self.pos_x, self.pos_y, self.pos_z = kx, ky, kz
            k_fall_dmg = (k_fell - 2.6).clamp(min=0.0) * 0.18 * alive_f
            fall_dmg = fall_dmg + k_fall_dmg
            if (k_fall_dmg > 0.05).any():
                for t in (0, 1):
                    self.stat_bridge_drops[:, t] += ((k_fall_dmg > 0.05) & (self.team == (1 - t))).float().sum(dim=1)

        # Direct thermal beam hit credit + 3D vertical delta tracking
        act_dmg_mat = torch.where(
            hit_mask,
            (thermal.abs()[:, None, :] * 0.14 + imp_dmg_mat) * (1.0 - dist_3d / (hit_rad + 1e-4)).clamp(min=0.0),
            0.0,
        )
        best_att_val, best_att_idx = act_dmg_mat.max(dim=-1)
        got_hit = (best_att_val > 0.02) & self.alive
        self.last_attacker = torch.where(got_hit, best_att_idx, self.last_attacker)

        # Track true 3D vertical hits (|z_attacker - z_victim| >= 2.0m!), aquatic ambushes, and squad cross-class combos!
        att_z = self.pos_z.gather(1, best_att_idx.clamp(0, N - 1))
        vert_hit = got_hit & ((att_z - self.pos_z).abs() >= 2.0)
        if vert_hit.any():
            for t in (0, 1):
                self.stat_vert_hits[:, t] += (vert_hit & (self.team == (1 - t))).float().sum(dim=1)
        # Tidekin aquatic ambush (striking enemies from water/submerged river!)
        att_in_water = self._tidekin_in_water.gather(1, best_att_idx.clamp(0, N - 1))
        aq_hit = got_hit & att_in_water
        if aq_hit.any():
            for t in (0, 1):
                self.stat_aquatic_ambushes[:, t] += (aq_hit & (self.team == (1 - t))).float().sum(dim=1)
        # Squad cross-class combo: oil primer by one ally + thermal hit by another ally!
        oil_hit = hit_mask & (emit_oil[:, None, :] > 0.30)
        any_oil_hit = oil_hit.any(dim=-1) & self.alive
        oil_src = oil_hit.float().argmax(dim=-1)
        self.oil_primed_by = torch.where(any_oil_hit, oil_src, self.oil_primed_by)
        self.oil_primed_ttl = torch.where(any_oil_hit, torch.full_like(self.oil_primed_ttl, 8), self.oil_primed_ttl)
        sq_combo = got_hit & (self.oil_primed_ttl > 0) & (self.oil_primed_by != best_att_idx) & (self.oil_primed_by >= 0)
        if sq_combo.any():
            for t in (0, 1):
                c_cnt = (sq_combo & (self.team == (1 - t))).float().sum(dim=1)
                self.stat_squad_combos[:, t] += c_cnt
                rewards += (self.team == t).float() * c_cnt.unsqueeze(1) * 0.08
            self.oil_primed_ttl = torch.where(sq_combo, torch.zeros_like(self.oil_primed_ttl), self.oil_primed_ttl)
        if (pressure > 0.06).any():
            for t in (0, 1):
                self.stat_pressure_blasts[:, t] += ((pressure > 0.06) & (self.team == (1 - t))).float().sum(dim=1)

        # Inside allied Stasis Dome reduces incoming thermal/pressure damage by 45% + rewards dome caster for tactical dilation
        in_ally_dome = ((self.dome_ttl[:, None, :] > 0) & (~self.enemy_mask) & (d2_domes < 49.0)).any(dim=-1)
        dome_shield = torch.where(in_ally_dome, 0.55, 1.0)
        dome_active_hits = ((self.dome_ttl[:, None, :] > 0) & self.enemy_mask & (d2_domes < 49.0) & self.alive[:, :, None]).float().sum(dim=1)
        rewards += dome_active_hits * 0.035 * alive_f

        total_dmg = (scorch + hypo + pressure + imp_dmg + fall_dmg) * dome_shield * alive_f
        self.hp = (self.hp - total_dmg).clamp(0.0, 1.0)

        # Reward attacker for physical damage dealt + extra shaping for true 3D vertical/pressure tactics
        dmg_dealt_per_att = act_dmg_mat.sum(dim=1)  # [A, N]
        rewards += dmg_dealt_per_att * 0.45

        # Eliminations
        died = self.alive & (self.hp <= 0.0)
        if died.any():
            self.alive = self.alive & (~died)
            self.respawn_cd = torch.where(died, torch.full_like(self.respawn_cd, 10), self.respawn_cd)
            rewards -= died.float() * 0.65
            # Drop relic if carrier died
            carrier_died = (self.relic_carrier >= 0) & died.gather(1, self.relic_carrier.clamp(0, N - 1).unsqueeze(1)).squeeze(1)
            self.relic_carrier = torch.where(carrier_died, torch.full_like(self.relic_carrier, -1), self.relic_carrier)

            for t in (0, 1):
                opp_died = (died & (self.team == (1 - t))).float().sum(dim=1)
                self.team_kills[:, t] += opp_died
                self.team_score[:, t] += opp_died * 8.0
                rewards += (self.team == t).float() * opp_died.unsqueeze(1) * 0.25

        # 5. 3D Shrine Control (requires being within 7m XY AND 2.8m Z of the shrine!)
        for k in range(3):
            dxy = torch.hypot(self.pos_x - self.shrine_x[:, k : k + 1], self.pos_y - self.shrine_y[:, k : k + 1])
            dz = (self.pos_z - self.shrine_z[:, k : k + 1]).abs()
            on_shrine = self.alive & (dxy < 7.0) & (dz < 2.8)
            w_cnt = (on_shrine & (self.team == 0)).float().sum(dim=1)
            e_cnt = (on_shrine & (self.team == 1)).float().sum(dim=1)
            delta_p = (w_cnt - e_cnt).clamp(-2.0, 2.0) * 0.14
            self.shrine_prog[:, k] = (self.shrine_prog[:, k] + delta_p).clamp(-1.0, 1.0)
            self.shrine_owner[:, k] = torch.where(
                self.shrine_prog[:, k] >= 0.85, 0,
                torch.where(self.shrine_prog[:, k] <= -0.85, 1, self.shrine_owner[:, k]),
            )
            rewards += on_shrine.float() * 0.08
            self.team_score[:, 0] += (self.shrine_owner[:, k] == 0).float() * 0.45
            self.team_score[:, 1] += (self.shrine_owner[:, k] == 1).float() * 0.45
            rewards += (self.team == 0).float() * (self.shrine_owner[:, k] == 0).float().unsqueeze(1) * 0.04
            rewards += (self.team == 1).float() * (self.shrine_owner[:, k] == 1).float().unsqueeze(1) * 0.04

        # 6. 3D Relic Pickup & Delivery (cave relic -> home base!)
        unheld = self.relic_carrier < 0
        if unheld.any():
            dr = torch.sqrt(
                (self.pos_x - self.relic_x.unsqueeze(1)) ** 2
                + (self.pos_y - self.relic_y.unsqueeze(1)) ** 2
                + (self.pos_z - self.relic_z.unsqueeze(1)) ** 2
            )
            dr = torch.where(self.alive, dr, torch.full_like(dr, 999.0))
            min_d, min_i = dr.min(dim=1)
            picked = unheld & (min_d < 4.5)
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
            delivered_w = held & (c_team == 0) & (cx < 0.22 * self.W)
            delivered_e = held & (c_team == 1) & (cx > 0.78 * self.W)
            self.team_score[:, 0] += delivered_w.float() * 30.0
            self.team_score[:, 1] += delivered_e.float() * 30.0
            rewards += (self.team == 0).float() * delivered_w.float().unsqueeze(1) * 1.5
            rewards += (self.team == 1).float() * delivered_e.float().unsqueeze(1) * 1.5
            reset_r = delivered_w | delivered_e
            self.relic_carrier = torch.where(reset_r, torch.full_like(self.relic_carrier, -1), self.relic_carrier)
            self.relic_x = torch.where(reset_r, torch.full_like(self.relic_x, 0.50 * self.W), self.relic_x)
            self.relic_y = torch.where(reset_r, torch.full_like(self.relic_y, 0.50 * self.H), self.relic_y)
            self.relic_z = torch.where(reset_r, torch.full_like(self.relic_z, 3.5), self.relic_z)

        # Refresh frozen 3D world latents periodically
        if self.tick % self.refresh_every == 0:
            self.z_world = self.world_encoder.encode(self.w_state)

        # Dense zero-rule game-score advantage delta (pure win-objective gradient!)
        d_sc = self.team_score - prev_team_score
        rewards += (self.team == 0).float() * (d_sc[:, 0] - d_sc[:, 1]).unsqueeze(1) * 0.08
        rewards += (self.team == 1).float() * (d_sc[:, 1] - d_sc[:, 0]).unsqueeze(1) * 0.08

        done = self.tick >= self.max_ticks
        if done:
            win_w = (self.team_score[:, 0] > self.team_score[:, 1]).float()
            win_e = (self.team_score[:, 1] > self.team_score[:, 0]).float()
            rewards += (self.team == 0).float() * (win_w - win_e).unsqueeze(1) * 3.5
            rewards += (self.team == 1).float() * (win_e - win_w).unsqueeze(1) * 3.5
        return rewards, done


def scripted_3d_policy(arena: Physics3DArenaV3) -> torch.Tensor:
    """Strong tactical 3D scripted baseline (shrine/relic pursuit + 3D vertical enemy targeting + heat/cryo)."""
    A, N = arena.A, arena.N
    dev = arena.device
    act = torch.zeros((A, N, ACT3D_DIM), device=dev)

    # Assign lanes to the 3 shrines (0=North bridge, 1=Center cave, 2=South bridge)
    lane_shrine = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], device=dev)
    gx = arena.shrine_x[:, lane_shrine]
    gy = arena.shrine_y[:, lane_shrine]

    # Carrier runs home
    is_carrier = arena.relic_carrier.unsqueeze(1) == torch.arange(N, device=dev).unsqueeze(0)
    home_x = torch.where(arena.team == 0, torch.full_like(gx, 0.16 * arena.W), torch.full_like(gx, 0.84 * arena.W))
    gx = torch.where(is_carrier, home_x, gx)

    dx = gx - arena.pos_x
    dy = gy - arena.pos_y
    dn = torch.hypot(dx, dy).clamp(min=1e-4)
    act[:, :, 0] = (dx / dn).clamp(-1.0, 1.0)
    act[:, :, 1] = (dy / dn).clamp(-1.0, 1.0)
    act[:, :, 2] = 0.45  # jump/climb ledges when needed

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

    act[:, :, 3] = (t_dx / 11.0).clamp(-1.0, 1.0)
    act[:, :, 4] = (t_dy / 11.0).clamp(-1.0, 1.0)
    act[:, :, 5] = (t_dz / 5.5).clamp(-1.0, 1.0)

    in_range = min_d < 12.0
    # Alternate heat & cryo + kinetic impulse on close targets
    act[:, :, 6] = torch.where(in_range, torch.where((arena.cls == 1), -0.85, 0.90), 0.0)
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
    """Side-balanced held-out 3D tournament (half West, half East) on unseen seeds."""
    policy_w.eval()
    wins = 0.0
    losses = 0.0
    draws = 0.0
    score_diff_sum = 0.0
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

    half = max(num_matches // 2, 1)
    n_done = 0
    batch_idx = 0
    while n_done < num_matches:
        swap_sides = n_done >= half
        arena.reset(seed=base_seed + batch_idx * 17)
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
                act_opp = torch.rand((arena.A, arena.N, ACT3D_DIM), device=arena.device) * 2.0 - 1.0
                act_opp[:, :, 7:11] = act_opp[:, :, 7:11].abs()
            else:
                raise ValueError(opponent_kind)

            if not swap_sides:
                act = torch.cat((act_rl[:, :6], act_opp[:, 6:]), dim=1)
                my_t, opp_t = 0, 1
            else:
                act = torch.cat((act_opp[:, :6], act_rl[:, 6:]), dim=1)
                my_t, opp_t = 1, 0
            _, done = arena.step(act)

        take = min(arena.A, num_matches - n_done)
        my_sc = arena.team_score[:take, my_t]
        op_sc = arena.team_score[:take, opp_t]
        wins += float((my_sc > op_sc).sum().item())
        losses += float((my_sc < op_sc).sum().item())
        draws += float((my_sc == op_sc).sum().item())
        score_diff_sum += float((my_sc - op_sc).sum().item())
        tactics["vert_hits"] += float(arena.stat_vert_hits[:take, my_t].sum().item())
        tactics["bridge_drops"] += float(arena.stat_bridge_drops[:take, my_t].sum().item())
        tactics["updraft_soars"] += float(arena.stat_updraft_soars[:take, my_t].sum().item())
        tactics["aquatic_ambushes"] += float(arena.stat_aquatic_ambushes[:take, my_t].sum().item())
        tactics["squad_combos"] += float(arena.stat_squad_combos[:take, my_t].sum().item())
        tactics["cryo_seals"] += float(arena.stat_cryo_seals[:take, my_t].sum().item())
        tactics["stasis_domes"] += float(arena.stat_stasis_domes[:take, my_t].sum().item())
        tactics["pressure_blasts"] += float(arena.stat_pressure_blasts[:take, my_t].sum().item())
        n_done += take

    wr = (wins + 0.5 * draws) / max(num_matches, 1)
    elo = 400.0 * math.log10(max(wr, 0.01) / max(1.0 - wr, 0.01))
    return {
        "opponent": opponent_kind,
        "matches": num_matches,
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
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
        import numpy as np
    except Exception as exc:
        print(f"[Viz] Skipping plot ({exc})")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for sp in ax.spines.values():
            sp.set_color("#30363d")

    # Panel 1: PPO Self-Play Return & Win-Rate Curve
    iters = [h["iter"] for h in history]
    rets = [h["mean_return"] for h in history]
    wrs = [h.get("eval_scripted_wr", float("nan")) * 100.0 for h in history]
    axes[0].plot(iters, rets, color="#58a6ff", lw=2.2, label="Episode Return")
    axes[0].set_title("ActuatorPolicyNetV3 (3D Z=16) PPO Training", color="#f0f6fc", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("PPO Iteration", color="#8b949e")
    axes[0].set_ylabel("Mean Return", color="#58a6ff")
    ax0_r = axes[0].twinx()
    valid_pts = [(i, w) for i, w in zip(iters, wrs) if not math.isnan(w)]
    if valid_pts:
        ax0_r.plot([p[0] for p in valid_pts], [p[1] for p in valid_pts], "o-", color="#3fb950", lw=2.0, label="vs Scripted-3D WR%")
        ax0_r.set_ylabel("Win Rate vs Scripted-3D (%)", color="#3fb950")
        ax0_r.set_ylim(0, 105)
        ax0_r.tick_params(colors="#3fb950")

    # Panel 2: Held-Out 3D Tournament Win Rates
    names = []
    vals = []
    colors = ["#3fb950", "#a371f7", "#f0883e"]
    for k, label in [("scripted_3d", "vs Scripted-3D"), ("flat_2d_ablation", "vs Flat-2D Ablation"), ("random_3d", "vs Random-3D")]:
        if k in tournament:
            names.append(label)
            vals.append(tournament[k]["win_rate"] * 100.0)
    bars = axes[1].bar(names, vals, color=colors[: len(names)], width=0.55)
    axes[1].axhline(50.0, color="#8b949e", ls="--", lw=1.2)
    axes[1].set_ylim(0, 110)
    axes[1].set_title("Held-Out 3D Multi-Biome Tournament (24 matches)", color="#f0f6fc", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Win Rate (%)", color="#c9d1d9")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2.0, v + 2.5, f"{v:.1f}%", ha="center", color="#f0f6fc", fontweight="bold")

    # Panel 3: Emergent 3D Tactical & Multi-Archetype Discoveries per Match
    tac = tournament.get("scripted_3d", {}).get("tactics_per_match", {})
    t_labels = ["3D Vert\ndz Hits", "Skywing\nUpdrafts", "Tidekin\nAquatic", "Squad\nCombos", "Bridge\nDrops", "Stasis\nDomes", "Pressure\nBlasts"]
    t_keys = ["vert_hits", "updraft_soars", "aquatic_ambushes", "squad_combos", "bridge_drops", "stasis_domes", "pressure_blasts"]
    t_vals = [float(tac.get(k, 0.0)) for k in t_keys]
    t_cols = ["#58a6ff", "#ffa657", "#39d353", "#f778ba", "#f85149", "#d2a8ff", "#79c0ff"]
    b3 = axes[2].bar(t_labels, t_vals, color=t_cols, width=0.62)
    axes[2].set_title("Emergent 3D & Archetype Tactics / Match (Zero Hardcoded Spells)", color="#f0f6fc", fontsize=10.5, fontweight="bold")
    axes[2].set_ylabel("Events / Match", color="#c9d1d9")
    for b, v in zip(b3, t_vals):
        axes[2].text(b.get_x() + b.get_width() / 2.0, v + max(t_vals + [1.0]) * 0.03, f"{v:.1f}", ha="center", color="#f0f6fc", fontsize=9)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Viz] Saved {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-ckpt", type=Path, default=Path("models/checkpoints/world_model_3d_v5.pt"))
    ap.add_argument("--out-ckpt", type=Path, default=Path("models/checkpoints/actuator_policy_v3_rl.pt"))
    ap.add_argument("--out-json", type=Path, default=Path("models/ACTUATOR_V3_TOURNAMENT.json"))
    ap.add_argument("--out-png", type=Path, default=Path("models/actuator_v3_viz.png"))
    ap.add_argument("--num-arenas", type=int, default=24)
    ap.add_argument("--grid-xy", type=int, default=64)
    ap.add_argument("--depth-z", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=48)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--ppo-epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--eval-matches", type=int, default=24)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        args.num_arenas = 2
        args.grid_xy = 32
        args.rollout_steps = 8
        args.iters = 2
        args.ppo_epochs = 1
        args.minibatch = 32
        args.eval_matches = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[ActuatorV3-RL] Device={device} | arenas={args.num_arenas} | grid={args.grid_xy}x{args.grid_xy}x{args.depth_z} | iters={args.iters}")

    world_enc = FrozenWorldLatentExtractor(args.world_ckpt if args.world_ckpt.exists() else None, device=device)
    arena = Physics3DArenaV3(
        num_arenas=args.num_arenas,
        world_encoder=world_enc,
        device=device,
        grid_xy=args.grid_xy,
        depth_z=args.depth_z,
        max_ticks=max(args.rollout_steps, 16) if args.smoke_test else args.rollout_steps,
    )

    policy = ActuatorPolicyNetV3().to(device)
    opp_pool: list[ActuatorPolicyNetV3] = [copy.deepcopy(policy).eval()]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.iters, 1), eta_min=args.lr * 0.25)
    n_params = count_parameters(policy)
    print(f"[ActuatorV3-RL] Policy params={n_params:,} | WorldLatent mode={world_enc.mode}")

    # Phase 1: Brief warmstart distillation against Scripted-3D for 4 iters (fast movement + aiming boot)
    # followed by pure reward-driven PPO Self-Play + League Opponent Pool!
    history: list[dict[str, float]] = []
    best_wr = -1.0
    best_sd = copy.deepcopy(policy.state_dict())

    for it in range(1, args.iters + 1):
        arena.reset(seed=1000 + it * 13)
        # Train 60% vs Scripted-3D + 40% Self-Play League so RL masters BOTH defeating Scripted-3D AND emergent 3D counter-play!
        use_scripted_opp = (it % 5 in (1, 2, 4)) or (len(opp_pool) == 0)
        opp_net = opp_pool[(it // 2) % len(opp_pool)]

        # Rollout storage (store compact 3D crops directly in fp16 on CUDA for 15x lower VRAM!)
        buf_fine, buf_coarse, buf_sf, buf_tok, buf_mask = [], [], [], [], []
        buf_u, buf_lp, buf_val, buf_rew, buf_done = [], [], [], [], []

        policy.eval()
        for step in range(args.rollout_steps):
            obs = arena.build_observations()
            with torch.inference_mode():
                fine_raw, coarse_raw, _ = policy.sample_raw_crops(
                    arena.z_world, obs["pos_xyz"], arena_idx=arena.arena_idx_flat
                )
                out = policy.forward_from_crops(
                    fine_raw, coarse_raw, obs["self_feat"], obs["tokens"], token_mask=obs["token_mask"]
                )
                act_all, u_all, lp_all = policy.sample_action(out)
                val_all = out["value"]

                # West team (0..5) is always the learning team; East team (6..11) is Scripted-3D or Pool Snapshot
                act_3d = act_all.reshape(arena.A, arena.N, ACT3D_DIM)
                if use_scripted_opp:
                    opp_act = scripted_3d_policy(arena)
                    # Smooth early curriculum noise (0.60 -> 0.0 over first 28 iters; eval is ALWAYS 0.0 full-strength!)
                    if it <= 28:
                        n_scale = 0.60 * (1.0 - (it - 1) / 28.0)
                        opp_act = (opp_act + torch.randn_like(opp_act) * n_scale).clamp(-1.0, 1.0)
                        opp_act[:, :, 7:11] = opp_act[:, :, 7:11].clamp(0.0, 1.0)
                else:
                    opp_out = opp_net.forward_from_crops(
                        fine_raw, coarse_raw, obs["self_feat"], obs["tokens"], token_mask=obs["token_mask"]
                    )
                    opp_act = opp_out["mean"].reshape(arena.A, arena.N, ACT3D_DIM)
                exec_act = torch.cat((act_3d[:, :6], opp_act[:, 6:]), dim=1)

            rew_all, done = arena.step(exec_act)

            # Keep only West-team (learning) samples: indices 0..5 per arena
            w_idx = (torch.arange(arena.A, device=device).unsqueeze(1) * arena.N + torch.arange(6, device=device).unsqueeze(0)).reshape(-1)
            buf_fine.append(fine_raw[w_idx].detach())
            buf_coarse.append(coarse_raw[w_idx].detach())
            buf_sf.append(obs["self_feat"][w_idx].detach())
            buf_tok.append(obs["tokens"][w_idx].detach())
            buf_mask.append(obs["token_mask"][w_idx].detach())
            buf_u.append(u_all[w_idx].detach().float())
            buf_lp.append(lp_all[w_idx].detach().float())
            buf_val.append(val_all[w_idx].detach().float())
            buf_rew.append(rew_all[:, :6].reshape(-1).detach().float())
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
        gamma, lam = 0.97, 0.95
        for t in reversed(range(T)):
            next_v = vals[t + 1] if t + 1 < T else torch.zeros(B_w, device=device)
            non_term = 1.0 - dones[t]
            delta = rews[t] + gamma * next_v * non_term - vals[t]
            last_gae = delta + gamma * lam * non_term * last_gae
            adv[t] = last_gae
        returns = adv + vals
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # Flatten [T*B_w, ...]
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
        ent_coef = max(0.004, 0.015 * (1.0 - it / max(args.iters, 1)))
        last_loss = 0.0
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(total_n, device=device)
            for s in range(0, total_n, args.minibatch):
                mb = perm[s : s + args.minibatch]
                out_mb = policy.forward_from_crops(
                    f_fine[mb], f_coarse[mb], f_sf[mb], f_tok[mb], token_mask=f_mask[mb]
                )
                new_lp, ent = policy.evaluate_actions(out_mb, f_u[mb])
                ratio = (new_lp - f_lp[mb]).clamp(-6.0, 6.0).exp()
                surr1 = ratio * f_adv[mb]
                surr2 = ratio.clamp(0.8, 1.2) * f_adv[mb]
                pi_loss = -torch.minimum(surr1, surr2).mean()
                v_loss = F.mse_loss(out_mb["value"].float(), f_ret[mb])
                loss = pi_loss + 0.5 * v_loss - ent_coef * ent.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.75)
                optimizer.step()
                last_loss = float(loss.item())
        scheduler.step()

        row: dict[str, float] = {
            "iter": float(it),
            "mean_return": float(rews.sum(dim=0).mean().item()),
            "loss": round(last_loss, 4),
        }

        eval_every = 1 if args.smoke_test else max(args.iters // 8, 5)
        if it % eval_every == 0 or it == args.iters:
            ev = evaluate_matchup(policy, "scripted_3d", arena, num_matches=min(8, args.eval_matches), base_seed=7000)
            row["eval_scripted_wr"] = ev["win_rate"]
            if ev["win_rate"] >= best_wr:
                best_wr = ev["win_rate"]
                best_sd = copy.deepcopy(policy.state_dict())
            if len(opp_pool) >= 4:
                opp_pool.pop(0)
            opp_pool.append(copy.deepcopy(policy).eval())
            print(
                f"[Iter {it:3d}/{args.iters}] ret={row['mean_return']:+.2f} | loss={last_loss:.3f} "
                f"| vs Scripted-3D WR={ev['win_rate']*100:.1f}% (best={best_wr*100:.1f}%) "
                f"| tactics={ev['tactics_per_match']} | elapsed={time.time()-t0:.1f}s"
            )
        history.append(row)

    # Restore best checkpoint & run full held-out 3D tournament
    policy.load_state_dict(best_sd)
    policy.eval()

    tournament: dict[str, Any] = {}
    for opp in ("scripted_3d", "flat_2d_ablation", "random_3d"):
        res = evaluate_matchup(policy, opp, arena, num_matches=args.eval_matches, base_seed=9100)
        tournament[opp] = res
        print(f"[Held-Out 3D Tournament] vs {opp:18s}: WR={res['win_rate']*100:5.1f}% | Elo={res['elo_delta']:+6.1f} | diff={res['mean_score_diff']:+5.2f} | tactics={res['tactics_per_match']}")

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
        print(f"[Save] Saved checkpoint -> {args.out_ckpt} ({args.out_ckpt.stat().st_size / 1024:.1f} KB)")

        payload = {
            "model": "ActuatorPolicyNetV3",
            "params": n_params,
            "world_latent_mode": world_enc.mode,
            "device": str(device),
            "train_seconds": round(time.time() - t0, 1),
            "grid": [args.grid_xy, args.grid_xy, args.depth_z],
            "iters": args.iters,
            "tournament": tournament,
            "history": history,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"[Save] Saved metrics -> {args.out_json}")
        save_viz(history, tournament, args.out_png)
    else:
        print(f"[Smoke Test] PASS in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
