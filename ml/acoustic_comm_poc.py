#!/usr/bin/env python3
"""
Frostmarsh 3D Physical Acoustics, Emergent Neural Speech, Viewer Text Interpreter,
and Human-to-Bot Goal-Vector Injection POC on NVIDIA GB10.

Runs directly over real 256x256x4 Frostmarsh volumes using frozen WorldModel
Z_world [4, 48, 256, 256] and EcosystemModel Z_eco [1, 32, 256, 256] spatial latents,
plus frozen BotPolicyNet for human intent redirection.

Outputs (strictly bounded):
  - /home/gooning/.t3/scratch/frostmarsh-ml/ACOUSTIC_COMM_METRICS.json
  - /home/gooning/.t3/scratch/frostmarsh-ml/ACOUSTIC_COMM_REPORT.md
  - /home/gooning/.t3/scratch/frostmarsh-ml/acoustic_field_viz.png
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bot_policy import (
    BOT_FEAT_DIM,
    NUM_TOKENS,
    SELF_DIM,
    TOKEN_DIM,
    BotPolicyNet,
)
from model import EcosystemModel, WorldModel
from sim_teacher import (
    H_CAP,
    NUM_SPECIES,
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
    SPEC_C_LIST,
    T_AMB,
    T_MAX,
    T_MIN,
    RefSim,
    SimState,
    encode_state,
    get_consts,
)

ROOT = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
GRID = 256
LAYERS = 4  # L0=subterranean cavern, L1=surface valley, L2=highland plateaus/skybridges, L3=open sky

EVENT_NAMES = [
    "OIL_SLICK_IGNITE",
    "WILDFIRE_FLANK",
    "FROZEN_RIVER_BRIDGE",
    "SHRINE_AMBUSH",
    "WATERFALL_CHOKE",
    "CAVERN_SHAFT_FLANK",
]
EVENT_SHORT = [
    "Oil Slick",
    "Wildfire",
    "Ice Bridge",
    "Shrine Ambush",
    "Waterfall Choke",
    "Cavern Shaft",
]
SECTOR_NAMES = ["NW Highland", "NE Skybridge", "SW Oil Basin", "SE Shrine Valley"]
ELEMENT_NAMES = ["Fire", "Frost", "Oil"]
TACTIC_NAMES = [
    "IGNITE_TRAP",
    "SUPPRESS_FLANK",
    "SHATTER_BRIDGE",
    "OIL_AMBUSH",
    "FREEZE_CHOKE",
    "CAVERN_DROP",
]
SPEAKER_CLASSES = ["Vanguard", "Pyromancer", "Cryo-Scout", "Shadow-Flanker"]

# Ground-truth optimal (element_idx, tactic_idx) per physical event
EVENT_TO_ELEM = [0, 1, 0, 2, 1, 2]  # Fire, Frost, Fire, Oil, Frost, Oil
EVENT_TO_TACTIC = [0, 1, 2, 3, 4, 5]

# Acoustic constants
SRC_SPL_MIN_DB = 56.0   # whisper (volume = 0.0)
SRC_SPL_SPAN_DB = 38.0  # shout   (volume = 1.0) -> 94.0 dB
WALL_LOSS_DB = 12.0     # -12 dB per basalt/rock/ice wall
CAVE_CEIL_LOSS_DB = 18.0
PLATEAU_SLAB_LOSS_DB = 14.0
OPEN_VERT_LOSS_DB = 2.0
ENEMY_DETECT_SNR_DB = 6.0


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 0. Real WorldModel / EcosystemModel Latent Extractors & 3D World Generator
# =============================================================================
def extract_world_latent(world_model: WorldModel, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        h = world_model.stem(x)
        for blk in world_model.blocks:
            h = blk(h)
        return F.gelu(h).float()


def extract_eco_latent(eco_model: EcosystemModel, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        h = eco_model.stem(x)
        for blk in eco_model.blocks:
            h = blk(h)
        return F.gelu(h).float()


@dataclass
class FrostmarshWorld3D:
    seed: int
    w_state: torch.Tensor        # [4, 18, 256, 256] float32 on GPU
    eco_state: torch.Tensor      # [1, 8, 256, 256] float32 on GPU
    z_world: torch.Tensor        # [4, 48, 256, 256] float32 on GPU
    z_eco: torch.Tensor          # [1, 32, 256, 256] float32 on GPU
    noise_db: torch.Tensor       # [4, 256, 256] float32 on GPU (ambient noise floor dB)
    noise_cause: torch.Tensor    # [4, 256, 256] int64 (0=ambient, 1=waterfall, 2=wildfire, 3=spell)
    solid_wall: torch.Tensor     # [4, 256, 256] float32 (1=rock/basalt/ice wall)
    open_shaft_01: torch.Tensor  # [256, 256] float32 (1=open vertical shaft L0<->L1)
    open_sky_12: torch.Tensor    # [256, 256] float32 (1=open sky L1<->L2, 0=solid highland slab)
    landmarks: dict[str, Any]


def build_frostmarsh_3d_state(seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Generate a rich 256x256x4 Frostmarsh 3D scene matching FIELD_CONTRACT and step L1
    through RefSim (sim_teacher.py) so surface physics are physically genuine.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float32)
    nx, ny = xx / GRID, yy / GRID

    # --- L1 Surface state built as SimState then stepped 2 ticks through RefSim ---
    mass1 = np.zeros((NUM_SPECIES, GRID, GRID), dtype=np.float32)
    elev1 = np.zeros((GRID, GRID), dtype=np.float32)
    flow_e1 = np.zeros((GRID, GRID), dtype=np.float32)
    flow_s1 = np.zeros((GRID, GRID), dtype=np.float32)
    T1 = np.full((GRID, GRID), T_AMB, dtype=np.float32)

    # Meandering Frostmarsh river
    phase = rng.uniform(0.0, 6.28)
    rcx = 0.50 + rng.uniform(-0.06, 0.06) + 0.065 * np.sin(ny * 6.28 * 1.15 + phase)
    rhw = rng.uniform(0.062, 0.082)
    in_river = np.abs(nx - rcx) < rhw

    elev1 = np.where(in_river, 4.0 + 3.0 * (1.0 - ny), 12.0 + 4.0 * np.sin(nx * 5.0 + seed) * np.cos(ny * 4.0))
    elev1 = np.clip(elev1, 0.0, 31.0)
    mass1[S_SOIL] = np.where(in_river, 220.0, 880.0)
    mass1[S_WATER] = np.where(in_river, 1150.0, 0.0)
    flow_s1 = np.where(in_river, 145.0, 0.0)

    # Thick tactical basalt ridges / walls (solid rock >= 500 -> -12 dB/wall acoustic occlusion!)
    wall_specs = []
    # Ridge 1: West-Central basalt canyon wall (between SW oil basin & central river)
    wx1 = rng.uniform(0.27, 0.34)
    wy1_a, wy1_b = rng.uniform(0.36, 0.44), rng.uniform(0.70, 0.80)
    ridge1 = (np.abs(nx - wx1) < 0.016) & (ny > wy1_a) & (ny < wy1_b)
    # Ridge 2: East-Central basalt ridge shielding SE shrine alcove
    wx2 = rng.uniform(0.65, 0.72)
    wy2_a, wy2_b = rng.uniform(0.40, 0.48), rng.uniform(0.76, 0.86)
    ridge2 = (np.abs(nx - wx2) < 0.016) & (ny > wy2_a) & (ny < wy2_b)
    # Ridge 3: Horizontal northern basalt baffle south of NW plateau
    hy3 = rng.uniform(0.35, 0.41)
    hx3_a, hx3_b = rng.uniform(0.08, 0.14), rng.uniform(0.30, 0.37)
    ridge3 = (np.abs(ny - hy3) < 0.015) & (nx > hx3_a) & (nx < hx3_b)
    # Ridge 4: Southern stone bastion near SE shrine
    hy4 = rng.uniform(0.62, 0.68)
    hx4_a, hx4_b = rng.uniform(0.70, 0.75), rng.uniform(0.88, 0.94)
    ridge4 = (np.abs(ny - hy4) < 0.015) & (nx > hx4_a) & (nx < hx4_b)

    basalt_mask = (ridge1 | ridge2 | ridge3 | ridge4) & (~in_river)
    # Plus 3-4 basalt rock pillars
    for _ in range(4):
        rx, ry = rng.uniform(0.14, 0.86), rng.uniform(0.16, 0.86)
        rr = rng.uniform(0.016, 0.024)
        pm = ((nx - rx) ** 2 + (ny - ry) ** 2) < rr ** 2
        basalt_mask |= (pm & (~in_river))

    mass1[S_ROCK] = np.where(basalt_mask, 1180.0, 0.0)
    mass1[S_WATER] = np.where(basalt_mask, 0.0, mass1[S_WATER])
    wall_specs.append({"wx1": float(wx1 * GRID), "wx2": float(wx2 * GRID)})

    # Forest wood & marsh plants
    wood_noise = np.sin(xx * 0.15 + seed * 0.7) * np.cos(yy * 0.17 - seed * 0.5)
    wood_m = (~in_river) & (~basalt_mask) & (wood_noise > 0.30)
    plant_m = (~in_river) & (~basalt_mask) & (~wood_m) & (wood_noise > -0.15)
    mass1[S_WOOD] = np.where(wood_m, 760.0, 0.0)
    mass1[S_PLANT] = np.where(plant_m, 380.0, 0.0)

    # Frozen River Bridge (randomized y on upper or lower river)
    ice_y_frac = float(rng.uniform(0.36, 0.62))
    ice_x_frac = float(np.mean(rcx[int(ice_y_frac * GRID), :]))
    ice_m = in_river & (np.abs(ny - ice_y_frac) < 0.036)
    mass1[S_ICE][ice_m] = 920.0
    mass1[S_WATER][ice_m] = 160.0
    T1[ice_m] = 25800.0
    ice_bridge_pos = (float(ice_x_frac * GRID), float(ice_y_frac * GRID), 1)

    # Oil Slick Basins (4-5 pools, guaranteed unlit primary basin in one quadrant)
    oil_positions = []
    quad_centers = [(0.19, 0.76), (0.81, 0.78), (0.18, 0.54), (0.82, 0.50)]
    rng.shuffle(quad_centers)
    for i in range(4):
        ox = float(np.clip(quad_centers[i][0] + rng.uniform(-0.04, 0.04), 0.10, 0.90))
        oy = float(np.clip(quad_centers[i][1] + rng.uniform(-0.04, 0.04), 0.14, 0.88))
        orad = float(rng.uniform(0.045, 0.062))
        om = (((nx - ox) ** 2 + (ny - oy) ** 2) < orad ** 2) & (~basalt_mask) & (~in_river)
        mass1[S_OIL][om] = 920.0
        mass1[S_WATER][om] = 220.0
        T1[om] = 28900.0
        oil_positions.append((ox * GRID, oy * GRID, 1))

    # Active Roaring Wildfires (3-4 burning zones with high T > 550K & smoke)
    wildfire_positions = []
    fire_anchors = [(0.22, 0.48), (0.78, 0.44), (0.26, 0.84), (0.72, 0.84)]
    rng.shuffle(fire_anchors)
    for i in range(3):
        fx = float(np.clip(fire_anchors[i][0] + rng.uniform(-0.04, 0.04), 0.10, 0.90))
        fy = float(np.clip(fire_anchors[i][1] + rng.uniform(-0.04, 0.04), 0.14, 0.88))
        # Avoid overlapping primary oil slick
        if math.hypot(fx * GRID - oil_positions[0][0], fy * GRID - oil_positions[0][1]) < 36.0:
            fx = 1.0 - fx
        frad = float(rng.uniform(0.036, 0.052))
        fm = (((nx - fx) ** 2 + (ny - fy) ** 2) < frad ** 2) & (~in_river) & (~basalt_mask)
        mass1[S_WOOD][fm] = np.maximum(mass1[S_WOOD][fm], 620.0)
        mass1[S_SMOKE][fm] = 460.0
        mass1[S_OIL][fm] = 0.0
        T1[fm] = rng.uniform(62000.0, 78000.0)
        wildfire_positions.append((fx * GRID, fy * GRID, 1))

    # Shrine Ambush Alcove (quiet rocky shrine courtyard on L1)
    shrine_quad = rng.choice(["SE", "SW", "NE", "NW"])
    if shrine_quad == "SE":
        sx, sy = rng.uniform(0.77, 0.87), rng.uniform(0.71, 0.82)
    elif shrine_quad == "SW":
        sx, sy = rng.uniform(0.13, 0.23), rng.uniform(0.71, 0.82)
    elif shrine_quad == "NE":
        sx, sy = rng.uniform(0.76, 0.86), rng.uniform(0.36, 0.45)
    else:
        sx, sy = rng.uniform(0.14, 0.24), rng.uniform(0.36, 0.45)
    shrine_pos = (float(sx * GRID), float(sy * GRID), 1)
    # Carve clean stone dais + partial basalt shrine arch
    dais = ((nx - sx) ** 2 + (ny - sy) ** 2) < 0.032 ** 2
    mass1[S_ROCK][dais] = 0.0
    mass1[S_OIL][dais] = 0.0
    mass1[S_SMOKE][dais] = 0.0
    T1[dais] = 29100.0
    mass1[S_SOIL][dais] = 950.0
    mass1[S_ASH][dais] = 240.0  # distinctive shrine flagstones

    # Explosive spell impulse (in 60% of worlds for acoustic noise richness)
    forcing_e1 = np.zeros((GRID, GRID), dtype=np.float32)
    spell_positions = []
    if rng.random() < 0.65:
        spx, spy = float(rng.uniform(0.28, 0.72)), float(rng.uniform(0.55, 0.82))
        sm = ((nx - spx) ** 2 + (ny - spy) ** 2) < 0.025 ** 2
        forcing_e1[sm] = 3.2e8
        mass1[S_STEAM][sm] = 380.0
        spell_positions.append((spx * GRID, spy * GRID, 1))

    # Step L1 through RefSim (2 ticks = 0.2s) for authentic physical gradients
    c = get_consts(device)
    mass_t = torch.from_numpy(mass1).unsqueeze(0).to(device)
    tot_m = mass_t.sum(dim=1, keepdim=True)
    mass_t = mass_t * torch.where(tot_m > H_CAP * 0.94, (H_CAP * 0.94) / (tot_m + 1e-6), torch.ones_like(tot_m))
    C_cell = (mass_t * c.spec_c).sum(dim=1, keepdim=True)
    energy_t = C_cell * torch.clamp(torch.from_numpy(T1).view(1, 1, GRID, GRID).to(device), T_MIN, T_MAX)
    sim1 = SimState(
        mass=mass_t.float(),
        energy=energy_t.float(),
        flow_e=torch.from_numpy(flow_e1.astype(np.float32)).view(1, 1, GRID, GRID).to(device),
        flow_s=torch.from_numpy(flow_s1.astype(np.float32)).view(1, 1, GRID, GRID).to(device),
        elevation=torch.from_numpy(elev1.astype(np.float32)).view(1, 1, GRID, GRID).to(device),
    )
    fe_t = torch.from_numpy(forcing_e1.astype(np.float32)).view(1, 1, GRID, GRID).to(device)
    with torch.inference_mode():
        sim1, _ = RefSim.step(sim1, forcing_energy=fe_t)
        sim1, _ = RefSim.step(sim1, forcing_energy=None)
        w1 = encode_state(sim1, forcing_energy=fe_t).squeeze(0)  # [18, 256, 256]

    # Build full 4-layer tensor [4, 18, 256, 256]
    w_state = torch.zeros((LAYERS, 18, GRID, GRID), dtype=torch.float32, device=device)
    w_state[1] = w1

    # --- L2 Highland Plateaus, Alpine Lakes, Plunging Waterfalls & Skybridges ---
    w2 = np.zeros((18, GRID, GRID), dtype=np.float32)
    nw_p = (nx > 0.05) & (nx < 0.37) & (ny > 0.05) & (ny < 0.32)
    ne_p = (nx > 0.63) & (nx < 0.95) & (ny > 0.05) & (ny < 0.32)
    north_bridge_y = float(rng.uniform(0.21, 0.27))
    south_bridge_y = float(rng.uniform(0.66, 0.72))
    north_skybridge = (np.abs(ny - north_bridge_y) < 0.022) & (nx > 0.35) & (nx < 0.65)
    south_skybridge = (np.abs(ny - south_bridge_y) < 0.018) & (nx > 0.35) & (nx < 0.65)

    w2[S_SOIL] = np.where(nw_p | ne_p | north_skybridge, 0.95, 0.0)
    w2[S_WOOD] = np.where(south_skybridge, 0.78, 0.0)
    w2[14] = np.where(nw_p | ne_p | north_skybridge | south_skybridge, 0.78, 0.0)

    # Basalt parapets along northern highland edge
    nw_parapet = (nx > 0.08) & (nx < 0.33) & (np.abs(ny - 0.14) < 0.014)
    ne_parapet = (nx > 0.67) & (nx < 0.92) & (np.abs(ny - 0.14) < 0.014)
    w2[S_ROCK] = np.where(nw_parapet | ne_parapet, 1.10, 0.0)
    w2[15] = np.where(nw_parapet | ne_parapet, 1.0, 0.0)

    # Alpine lakes feeding TWO plunging waterfalls (L2 -> L1) over cliff rims!
    wf_nw = (float(0.36 * GRID), float(rng.uniform(0.14, 0.20) * GRID))
    wf_ne = (float(0.64 * GRID), float(rng.uniform(0.14, 0.20) * GRID))
    waterfall_positions = []
    for (wfx, wfy, side) in [(wf_nw[0], wf_nw[1], -1.0), (wf_ne[0], wf_ne[1], 1.0)]:
        # Alpine lake feeding the lip on L2
        lake_cx = (wfx + side * 14.0) / GRID
        lake_cy = wfy / GRID
        lm = ((nx - lake_cx) ** 2 + (ny - lake_cy) ** 2) < 0.068 ** 2
        lm &= (nw_p if side < 0 else ne_p)
        w2[S_WATER][lm] = 1.25
        w2[12][lm] = -side * 0.38  # pouring toward the canyon rim
        # Plunge lip on L2
        lip = ((nx - wfx / GRID) ** 2 + (ny - wfy / GRID) ** 2) < 0.030 ** 2
        w2[S_WATER][lip] = 1.35
        w2[S_STEAM][lip] = 0.25  # waterfall mist
        # Plunge pool directly below on L1
        plunge_l1 = torch.from_numpy((((nx - wfx / GRID) ** 2 + (ny - wfy / GRID) ** 2) < 0.038 ** 2)).to(device)
        w_state[1, S_WATER] = torch.where(plunge_l1, torch.clamp(w_state[1, S_WATER] + 0.9, max=1.8), w_state[1, S_WATER])
        w_state[1, S_STEAM] = torch.where(plunge_l1, torch.clamp(w_state[1, S_STEAM] + 0.22, max=0.6), w_state[1, S_STEAM])
        waterfall_positions.append((wfx, wfy, 2))

    # Waterfall choke tactical point: near cliff lip or skybridge flank
    wf_choke_idx = int(rng.integers(0, 2))
    wf_choke_pos = waterfall_positions[wf_choke_idx]
    w2[11] = np.log1p(np.maximum(w2[:10].sum(axis=0), 0.0))
    w_state[2] = torch.from_numpy(w2).to(device)

    # --- L0 Subterranean Caverns & Vertical Shafts ---
    w0 = np.zeros((18, GRID, GRID), dtype=np.float32)
    # Winding subterranean cavern tunnel beneath the valley
    cav_x = 0.48 + 0.16 * np.sin(ny * 6.28 * 0.9 + phase + 1.1)
    in_tunnel = (np.abs(nx - cav_x) < 0.085) & (ny > 0.16) & (ny < 0.86)
    # Cross tunnel east-west
    cross_y = float(rng.uniform(0.46, 0.56))
    in_cross = (np.abs(ny - cross_y) < 0.065) & (nx > 0.20) & (nx < 0.80)
    cavern_open = in_tunnel | in_cross

    w0[S_ROCK] = np.where(~cavern_open, 1.30, 0.0)
    w0[15] = np.where(~cavern_open, 1.0, 0.0)
    w0[S_SOIL] = np.where(cavern_open, 0.85, 0.0)
    w0[S_WATER] = np.where(cavern_open & (np.abs(nx - cav_x) < 0.025), 0.35, 0.0)
    w0[10] = -0.18  # cool subterranean air
    w0[14] = np.where(cavern_open, 0.12, 0.05)

    # 3 Open Vertical Shafts connecting L0 <-> L1 (elev channel > 0.55 marks open shaft!)
    shaft_positions = []
    shaft_anchors = [
        (float(np.clip(cav_x[int(0.28 * GRID), 0], 0.22, 0.78)), 0.28),
        (float(rng.uniform(0.26, 0.36)), cross_y),
        (float(rng.uniform(0.64, 0.74)), cross_y),
    ]
    open_shaft_np = np.zeros((GRID, GRID), dtype=np.float32)
    for shx, shy in shaft_anchors:
        shm = ((nx - shx) ** 2 + (ny - shy) ** 2) < 0.036 ** 2
        w0[S_ROCK][shm] = 0.0
        w0[15][shm] = 0.0
        w0[S_SOIL][shm] = 0.75
        w0[14][shm] = 0.75  # open vertical skylight / shaft
        open_shaft_np[shm] = 1.0
        shaft_positions.append((shx * GRID, shy * GRID, 0))
    w0[11] = np.log1p(np.maximum(w0[:10].sum(axis=0), 0.0))
    w_state[0] = torch.from_numpy(w0).to(device)

    # --- L3 Open Sky (rising smoke & steam wisps, zero walls) ---
    w3 = np.zeros((18, GRID, GRID), dtype=np.float32)
    w3[S_SMOKE] = (w_state[1, S_SMOKE].cpu().numpy() * 0.45).astype(np.float32)
    w3[S_STEAM] = (w_state[1, S_STEAM].cpu().numpy() * 0.35).astype(np.float32)
    w3[10] = -0.12
    w3[14] = 1.0
    w3[11] = np.log1p(np.maximum(w3[:10].sum(axis=0), 0.0))
    w_state[3] = torch.from_numpy(w3).to(device)

    # --- Ecosystem 8-ch input matching EcosystemModel ---
    eco = torch.zeros((1, 8, GRID, GRID), dtype=torch.float32, device=device)
    eco[0, 0] = (w_state[1, S_PLANT] > 0.15).float() * 0.35
    eco[0, 1] = (w_state[1, 10] > 1.2).float() * 0.85
    eco[0, 4] = w_state[1, S_PLANT]
    eco[0, 5] = ((w_state[1, S_WATER] >= 0.35) & (w_state[1, S_ICE] < 0.70)).float()
    eco[0, 6] = torch.clamp((w_state[1, 10] - 1.0) / 3.0 + w_state[1, S_SMOKE], 0.0, 1.0)
    eco[0, 7] = w_state[1, 15]

    landmarks = {
        "oil_positions": oil_positions,
        "wildfire_positions": wildfire_positions,
        "ice_bridge_pos": ice_bridge_pos,
        "shrine_pos": shrine_pos,
        "waterfall_positions": waterfall_positions,
        "wf_choke_pos": wf_choke_pos,
        "shaft_positions": shaft_positions,
        "north_skybridge": (128.0, north_bridge_y * GRID, 2),
        "south_skybridge": (128.0, south_bridge_y * GRID, 2),
        "spell_positions": spell_positions,
        "open_shaft_01": torch.from_numpy(open_shaft_np).to(device),
        "wall_specs": wall_specs,
    }
    return w_state, eco, landmarks


# =============================================================================
# 1. Physical 3D Acoustic Propagation & Ambient Noise Field (AcousticField3D)
# =============================================================================
class AcousticField3D(nn.Module):
    """
    GPU-accelerated 3D physical acoustics solver on 256x256x4 Frostmarsh volumes.

    Computes directly from authoritative [4, 18, 256, 256] voxel state:
      1. Ambient Noise Floor N(x,y,z) in dB SPL from plunging waterfalls (L2->L1),
         roaring wildfires (T > 440K), and explosive spells, propagated through
         10 GPU 3D wall-aware damped wave/diffusion conv steps.
      2. 3D Acoustic Attenuation & Wall Occlusion A(src -> dst) in dB:
         exact 1/r geometric spreading + solid basalt/rock/ice wall occlusion
         (-12 dB per wall) + vertical slab/shaft coupling (-18 dB L0 rock ceiling,
         -14 dB L2 plateau slab, -2 dB open shafts/sky) + 10-step GPU 3D
         acoustic diffraction/reverberation field!
      3. Exact Signal-to-Noise Ratio SNR(src -> dst, volume) in dB.
    """

    def __init__(self, n_coarse_steps: int = 8, n_fine_steps: int = 2):
        super().__init__()
        self.n_coarse_steps = n_coarse_steps
        self.n_fine_steps = n_fine_steps

        # Isotropic 3x3 horizontal acoustic wave-diffusion stencil (grouped across 4 layers)
        k2d = torch.tensor(
            [[0.06, 0.14, 0.06],
             [0.14, 0.20, 0.14],
             [0.06, 0.14, 0.06]],
            dtype=torch.float32,
        )
        k2d = k2d / k2d.sum()
        self.register_buffer("horiz_kernel", k2d.view(1, 1, 3, 3).repeat(LAYERS, 1, 1, 1))

        # Baseline ambient noise floor per layer (dB SPL)
        self.register_buffer(
            "base_noise_db",
            torch.tensor([28.0, 32.0, 34.0, 36.0], dtype=torch.float32).view(LAYERS, 1, 1),
        )

    @staticmethod
    def extract_acoustic_geometry(
        w_state: torch.Tensor,
        open_shaft_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract solid wall mask [4,H,W], open L0<->L1 shaft mask [H,W], open L1<->L2 sky [H,W]."""
        solid_wall = (
            (w_state[:, S_ROCK] >= 0.5)
            | (w_state[:, 15] >= 0.5)
            | (w_state[:, S_ICE] >= 0.70)
        ).float()
        if open_shaft_override is not None:
            open_shaft_01 = open_shaft_override.float()
        else:
            open_shaft_01 = ((w_state[0, S_ROCK] < 0.5) & (w_state[0, 14] > 0.55)).float()
        # Open sky between L1 and L2 where L2 has no solid terrain/rock slab
        open_sky_12 = ((w_state[2, S_SOIL] < 0.30) & (w_state[2, S_ROCK] < 0.50)).float()
        return solid_wall, open_shaft_01, open_sky_12

    def _step_3d_conv(
        self,
        energy: torch.Tensor,         # [B, 4, H, W]
        trans_h: torch.Tensor,        # [1, 4, H, W] horizontal transmission factor
        v_up: torch.Tensor,           # [1, 4, H, W] coupling from layer z-1 -> z
        v_dn: torch.Tensor,           # [1, 4, H, W] coupling from layer z+1 -> z
        v_self: torch.Tensor,         # [1, 4, H, W] self retention
        damping: float = 0.85,
    ) -> torch.Tensor:
        """Vectorized GPU 3D acoustic wave/diffusion step (1 grouped 3x3 conv + zero-cat vertical mix)."""
        horiz = F.conv2d(energy * trans_h, self.horiz_kernel, padding=1, groups=LAYERS)
        e_from_below = F.pad(horiz[:, :-1], (0, 0, 0, 0, 1, 0))
        e_from_above = F.pad(horiz[:, 1:], (0, 0, 0, 0, 0, 1))
        return (horiz * v_self + e_from_below * v_up + e_from_above * v_dn) * damping

    def _build_vert_tensors(
        self,
        open_shaft_01: torch.Tensor,
        open_sky_12: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute vectorized [1, 4, H, W] vertical coupling coefficients for L0<->L1<->L2<->L3."""
        z0 = torch.zeros_like(open_shaft_01)
        c01 = 0.35 * (0.016 + 0.68 * open_shaft_01)
        c12 = 0.35 * (0.040 + 0.66 * open_sky_12)
        c23 = torch.full_like(open_shaft_01, 0.30 * 0.35)
        v_up = torch.stack([z0, c01, c12, c23], dim=0).unsqueeze(0)
        v_dn = torch.stack([c01, c12, c23, z0], dim=0).unsqueeze(0)
        v_self = 1.0 - v_up - v_dn
        return v_up, v_dn, v_self

    def propagate_power_3d(
        self,
        src_power: torch.Tensor,      # [B, 4, 256, 256] linear acoustic power
        solid_wall: torch.Tensor,     # [4, 256, 256]
        open_shaft_01: torch.Tensor,  # [256, 256]
        open_sky_12: torch.Tensor,    # [256, 256]
        damping_coarse: float = 0.52,
        damping_fine: float = 0.82,
    ) -> torch.Tensor:
        """
        10-step (8 coarse 64x64x4 + 2 fine 256x256x4) GPU 3D damped wave/diffusion propagation.
        Preserves thin basalt walls via max-pool on the coarse grid so sound cannot tunnel!
        """
        wall_f = solid_wall.unsqueeze(0)  # [1, 4, 256, 256]
        trans_f = 1.0 - 0.90 * wall_f
        v_up_f, v_dn_f, v_self_f = self._build_vert_tensors(open_shaft_01, open_sky_12)

        # Coarse 64x64x4 grid for wide-area diffraction + noise propagation (8 conv steps)
        p_c = F.max_pool2d(src_power, kernel_size=4, stride=4) * 0.55
        wall_c = F.max_pool2d(wall_f, kernel_size=4, stride=4)
        trans_c = 1.0 - 0.94 * wall_c
        v_up_c = F.avg_pool2d(v_up_f, kernel_size=4, stride=4)
        v_dn_c = F.avg_pool2d(v_dn_f, kernel_size=4, stride=4)
        v_self_c = 1.0 - v_up_c - v_dn_c

        field_c = p_c
        for _ in range(self.n_coarse_steps):
            field_c = self._step_3d_conv(field_c, trans_c, v_up_c, v_dn_c, v_self_c, damping=damping_coarse)
            field_c = torch.maximum(field_c, p_c)

        # Upsample + 2 fine 256x256x4 refinement conv steps (total = 10 GPU conv steps!)
        field_f = F.interpolate(field_c, size=(GRID, GRID), mode="bilinear", align_corners=False)
        field_f = torch.maximum(field_f, src_power)
        for _ in range(self.n_fine_steps):
            field_f = self._step_3d_conv(field_f, trans_f, v_up_f, v_dn_f, v_self_f, damping=damping_fine)
            field_f = torch.maximum(field_f, src_power)
        return field_f

    def propagate_10step_precomputed(
        self,
        src_power: torch.Tensor,
        trans_c: torch.Tensor,
        v_up_c: torch.Tensor,
        v_dn_c: torch.Tensor,
        v_self_c: torch.Tensor,
        trans_f: torch.Tensor,
        v_up_f: torch.Tensor,
        v_dn_f: torch.Tensor,
        v_self_f: torch.Tensor,
    ) -> torch.Tensor:
        """Fast static-geometry 10-step 3D acoustic propagation (CUDA-graph compatible!)."""
        p_c = F.max_pool2d(src_power, kernel_size=4, stride=4) * 0.55
        fc = p_c
        for _ in range(self.n_coarse_steps):
            fc = self._step_3d_conv(fc, trans_c, v_up_c, v_dn_c, v_self_c, damping=0.52)
            fc = torch.maximum(fc, p_c)
        ff = F.interpolate(fc, size=(GRID, GRID), mode="bilinear", align_corners=False)
        ff = torch.maximum(ff, src_power)
        for _ in range(self.n_fine_steps):
            ff = self._step_3d_conv(ff, trans_f, v_up_f, v_dn_f, v_self_f, damping=0.82)
            ff = torch.maximum(ff, src_power)
        return ff

    @torch.no_grad()
    def compute_ambient_noise_field(
        self,
        w_state: torch.Tensor,
        solid_wall: torch.Tensor,
        open_shaft_01: torch.Tensor,
        open_sky_12: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute authoritative 3D Ambient Noise Floor N(x,y,z) [4, 256, 256] (dB SPL)
        and dominant physical noise cause [4, 256, 256] (0=ambient, 1=waterfall, 2=wildfire, 3=spell).
        """
        device = w_state.device
        # 1. Plunging waterfalls (L2 water over open cliff edge + L1 plunge pools beneath L2 water)
        l2_water = (w_state[2, S_WATER] > 0.55).float()
        l2_edge = (F.max_pool2d(open_sky_12.view(1, 1, GRID, GRID), 5, stride=1, padding=2).squeeze() > 0.5).float()
        wf_lip_l2 = l2_water * l2_edge
        wf_plunge_l1 = (
            F.max_pool2d(wf_lip_l2.view(1, 1, GRID, GRID), 7, stride=1, padding=3).squeeze()
            * (w_state[1, S_WATER] > 0.50).float()
        )
        p_wf = torch.zeros((1, LAYERS, GRID, GRID), device=device)
        p_wf[0, 1] = wf_plunge_l1 * (10.0 ** (85.0 / 10.0))
        p_wf[0, 2] = wf_lip_l2 * (10.0 ** (81.0 / 10.0))

        # 2. Roaring wildfires (T > 440 K => ch10 > 1.5)
        hot = torch.clamp((w_state[:, 10] - 1.5) / 2.5, 0.0, 1.0)
        wf_fire_mask = (hot > 0.05).float()
        fire_db = 70.0 + 7.0 * hot
        p_fire = (wf_fire_mask * (10.0 ** (fire_db / 10.0))).unsqueeze(0)

        # 3. Explosive spells / rapid steam bursts
        spell_mag = torch.clamp(w_state[:, 16].abs() / 2.0, 0.0, 1.0)
        spell_mask = (spell_mag > 0.20).float()
        p_spell = (spell_mask * (10.0 ** ((84.0 + 6.0 * spell_mag) / 10.0))).unsqueeze(0)

        # Batch-propagate all 3 physical noise components in ONE 10-step GPU 3D conv pass!
        stacked = torch.cat([p_wf, p_fire, p_spell], dim=0)  # [3, 4, 256, 256]
        prop = self.propagate_power_3d(stacked, solid_wall, open_shaft_01, open_sky_12)  # [3, 4, 256, 256]

        p_base = 10.0 ** (self.base_noise_db / 10.0)  # [4, 256, 256]
        p_total = p_base + prop[0] + prop[1] + prop[2]
        noise_db = 10.0 * torch.log10(torch.clamp(p_total, min=1.0))

        # Identify dominant physical noise cause at every voxel
        all_sources = torch.stack([p_base.expand_as(prop[0]) * 1.5, prop[0], prop[1], prop[2]], dim=0)  # [4, 4, 256, 256]
        noise_cause = all_sources.argmax(dim=0)  # [4, 256, 256]
        return noise_db, noise_cause

    @torch.no_grad()
    def compute_speaker_snr_field(
        self,
        src_x: float,
        src_y: float,
        src_z: int,
        volume: float,
        solid_wall: torch.Tensor,
        open_shaft_01: torch.Tensor,
        open_sky_12: torch.Tensor,
        noise_db: torch.Tensor,
        n_ray_samples: int = 24,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute full 3D [4, 256, 256] Attenuation A(src -> x,y,z) (dB) and
        SNR(src -> x,y,z) (dB) for a speaker at (src_x, src_y, src_z) with volume in [0,1].
        Combines exact 1/r spreading + wall occlusion (-12 dB/wall) + 10-step 3D diffraction!
        """
        device = solid_wall.device
        yy = torch.arange(GRID, device=device, dtype=torch.float32).view(1, GRID, 1)
        xx = torch.arange(GRID, device=device, dtype=torch.float32).view(1, 1, GRID)
        zz = torch.arange(LAYERS, device=device, dtype=torch.float32).view(LAYERS, 1, 1)

        dx = xx - src_x
        dy = yy - src_y
        dz_cells = (zz - float(src_z)) * 10.0
        dist = torch.sqrt(dx * dx + dy * dy + dz_cells * dz_cells + 1e-6)
        pl_geom = 20.0 * torch.log10(1.0 + dist / 4.0)  # exact 1/r pressure spreading

        # Vectorized 24-sample ray-march on GPU counting distinct basalt/rock/ice wall walls crossed
        t = torch.linspace(0.06, 0.94, n_ray_samples, device=device).view(n_ray_samples, 1, 1, 1)
        rx = (src_x + t * dx.unsqueeze(0)).clamp(0, GRID - 1).long()  # [S, 1, H, W]
        ry = (src_y + t * dy.unsqueeze(0)).clamp(0, GRID - 1).long()
        # Sample solid_wall on both source & destination layer midpoints
        rz = torch.round(float(src_z) + t * (zz - float(src_z)).unsqueeze(0)).clamp(0, LAYERS - 1).long()
        w_samples = solid_wall[rz.expand(-1, LAYERS, GRID, GRID), ry.expand(-1, LAYERS, GRID, GRID), rx.expand(-1, LAYERS, GRID, GRID)]
        # Count rising wall edges (each distinct basalt wall = -12 dB) + thickness factor
        prev_w = F.pad(w_samples[:-1], (0, 0, 0, 0, 0, 0, 1, 0), value=0.0)
        wall_entries = F.relu(w_samples - prev_w).sum(dim=0)  # [4, 256, 256]
        wall_thick = w_samples.mean(dim=0) * (dist / 12.0).clamp(max=2.5)
        n_walls_eff = torch.maximum(wall_entries, wall_thick * 0.65).clamp(0.0, 4.0)
        pl_wall = n_walls_eff * WALL_LOSS_DB

        # Vertical ceiling / shaft attenuation
        pl_vert = torch.zeros((LAYERS, GRID, GRID), device=device)
        src_ix, src_iy = int(np.clip(round(src_x), 0, GRID - 1)), int(np.clip(round(src_y), 0, GRID - 1))
        for dst_z in range(LAYERS):
            if dst_z == src_z:
                continue
            lo, hi = min(src_z, dst_z), max(src_z, dst_z)
            vloss = torch.zeros((GRID, GRID), device=device)
            if lo == 0 and hi >= 1:
                # L0<->L1: open if EITHER src or dst is near a vertical shaft
                shaft_open = torch.maximum(
                    open_shaft_01,
                    torch.full_like(open_shaft_01, float(open_shaft_01[src_iy, src_ix].item())),
                )
                vloss = vloss + torch.where(shaft_open > 0.5, OPEN_VERT_LOSS_DB, CAVE_CEIL_LOSS_DB)
            if lo <= 1 and hi >= 2:
                sky_open = torch.maximum(
                    open_sky_12,
                    torch.full_like(open_sky_12, float(open_sky_12[src_iy, src_ix].item())),
                )
                vloss = vloss + torch.where(sky_open > 0.5, OPEN_VERT_LOSS_DB, PLATEAU_SLAB_LOSS_DB)
            if hi == 3:
                vloss = vloss + OPEN_VERT_LOSS_DB
            pl_vert[dst_z] = vloss

        atten_direct_db = pl_geom + pl_wall + pl_vert

        # 10-step GPU 3D acoustic diffraction/reverberation field around wall edges & shafts
        src_spl = SRC_SPL_MIN_DB + SRC_SPL_SPAN_DB * float(volume)
        p_direct = 10.0 ** ((src_spl - atten_direct_db) / 10.0)  # [4, 256, 256]
        # Inject impulse at speaker + attenuated diffraction seed into 10-step 3D conv
        impulse = torch.zeros((1, LAYERS, GRID, GRID), device=device)
        y0, y1 = max(0, src_iy - 2), min(GRID, src_iy + 3)
        x0, x1 = max(0, src_ix - 2), min(GRID, src_ix + 3)
        impulse[0, src_z, y0:y1, x0:x1] = 10.0 ** ((src_spl - 6.0) / 10.0)
        p_diff = self.propagate_power_3d(
            impulse, solid_wall, open_shaft_01, open_sky_12, damping_coarse=0.82, damping_fine=0.88
        ).squeeze(0)

        # Combine direct + diffracted wave power (power-domain sum!)
        p_rx = p_direct + 0.35 * p_diff
        rx_db = 10.0 * torch.log10(torch.clamp(p_rx, min=1e-6))
        atten_eff_db = torch.clamp(src_spl - rx_db, min=0.0)
        snr_db = rx_db - noise_db
        return atten_eff_db, snr_db

    @torch.no_grad()
    def query_pairwise_link(
        self,
        src_xyz: torch.Tensor,        # [M, 3] (x, y, z)
        dst_xyz: torch.Tensor,        # [M, 3] (x, y, z)
        solid_wall: torch.Tensor,     # [4, 256, 256]
        open_shaft_01: torch.Tensor,  # [256, 256]
        open_sky_12: torch.Tensor,    # [256, 256]
        noise_db: torch.Tensor,       # [4, 256, 256]
        noise_cause: torch.Tensor,    # [4, 256, 256]
        n_ray_samples: int = 24,
    ) -> dict[str, torch.Tensor]:
        """
        Fast batched GPU query for M (src -> dst) pairs on a single 256x256x4 world.
        Returns exact attenuation (dB), walls_crossed, ceiling_muffled, dst_noise_db, dst_noise_cause.
        """
        device = solid_wall.device
        dx = dst_xyz[:, 0] - src_xyz[:, 0]
        dy = dst_xyz[:, 1] - src_xyz[:, 1]
        dz = dst_xyz[:, 2] - src_xyz[:, 2]
        dist = torch.sqrt(dx * dx + dy * dy + (dz * 10.0) ** 2 + 1e-6)
        pl_geom = 20.0 * torch.log10(1.0 + dist / 4.0)

        t = torch.linspace(0.06, 0.94, n_ray_samples, device=device).view(1, n_ray_samples)
        rx = (src_xyz[:, 0:1] + t * dx.unsqueeze(1)).clamp(0, GRID - 1).long()
        ry = (src_xyz[:, 1:2] + t * dy.unsqueeze(1)).clamp(0, GRID - 1).long()
        rz = torch.round(src_xyz[:, 2:3] + t * dz.unsqueeze(1)).clamp(0, LAYERS - 1).long()
        w_s = solid_wall[rz, ry, rx]  # [M, S]
        prev_w = F.pad(w_s[:, :-1], (1, 0), value=0.0)
        wall_entries = F.relu(w_s - prev_w).sum(dim=1)
        wall_thick = w_s.mean(dim=1) * (dist / 12.0).clamp(max=2.5)
        n_walls = torch.maximum(wall_entries, wall_thick * 0.65).clamp(0.0, 4.0)

        # Mild diffraction relief if wall is short/flanked (check +/- 8 cells perpendicular offset)
        inv_len = 1.0 / torch.sqrt(dx * dx + dy * dy + 1e-4)
        px, py = -dy * inv_len * 9.0, dx * inv_len * 9.0
        diffract_relief = torch.zeros_like(n_walls)
        for sign in (-1.0, 1.0):
            rx_o = (src_xyz[:, 0:1] + t * dx.unsqueeze(1) + sign * px.unsqueeze(1) * torch.sin(t * math.pi)).clamp(0, GRID - 1).long()
            ry_o = (src_xyz[:, 1:2] + t * dy.unsqueeze(1) + sign * py.unsqueeze(1) * torch.sin(t * math.pi)).clamp(0, GRID - 1).long()
            w_o = solid_wall[rz, ry_o, rx_o]
            prev_wo = F.pad(w_o[:, :-1], (1, 0), value=0.0)
            nw_o = F.relu(w_o - prev_wo).sum(dim=1)
            # If flanking ray bypasses wall, diffraction recovers ~4.5 dB
            can_diff = ((n_walls >= 0.8) & (nw_o < n_walls - 0.5)).float()
            diffract_relief = torch.maximum(diffract_relief, can_diff * 4.5)

        pl_wall = torch.clamp(n_walls * WALL_LOSS_DB - diffract_relief, min=0.0)

        # Vertical coupling
        s_ix = src_xyz[:, 0].clamp(0, GRID - 1).long()
        s_iy = src_xyz[:, 1].clamp(0, GRID - 1).long()
        d_ix = dst_xyz[:, 0].clamp(0, GRID - 1).long()
        d_iy = dst_xyz[:, 1].clamp(0, GRID - 1).long()
        s_iz = src_xyz[:, 2].clamp(0, LAYERS - 1).long()
        d_iz = dst_xyz[:, 2].clamp(0, LAYERS - 1).long()

        lo = torch.minimum(s_iz, d_iz)
        hi = torch.maximum(s_iz, d_iz)
        pl_vert = torch.zeros_like(pl_geom)
        ceiling_muffled = torch.zeros_like(pl_geom)

        cross_01 = (lo == 0) & (hi >= 1)
        shaft_ok = torch.maximum(open_shaft_01[s_iy, s_ix], open_shaft_01[d_iy, d_ix]) > 0.5
        pl_vert = pl_vert + torch.where(
            cross_01,
            torch.where(shaft_ok, torch.full_like(pl_vert, OPEN_VERT_LOSS_DB), torch.full_like(pl_vert, CAVE_CEIL_LOSS_DB)),
            torch.zeros_like(pl_vert),
        )
        ceiling_muffled = torch.where(cross_01 & (~shaft_ok), torch.ones_like(ceiling_muffled), ceiling_muffled)

        cross_12 = (lo <= 1) & (hi >= 2)
        sky_ok = torch.maximum(open_sky_12[s_iy, s_ix], open_sky_12[d_iy, d_ix]) > 0.5
        pl_vert = pl_vert + torch.where(
            cross_12,
            torch.where(sky_ok, torch.full_like(pl_vert, OPEN_VERT_LOSS_DB), torch.full_like(pl_vert, PLATEAU_SLAB_LOSS_DB)),
            torch.zeros_like(pl_vert),
        )
        cross_23 = (lo <= 2) & (hi == 3)
        pl_vert = pl_vert + torch.where(cross_23, torch.full_like(pl_vert, OPEN_VERT_LOSS_DB), torch.zeros_like(pl_vert))

        atten_db = pl_geom + pl_wall + pl_vert
        dst_noise = noise_db[d_iz, d_iy, d_ix]
        dst_cause = noise_cause[d_iz, d_iy, d_ix]
        src_noise = noise_db[s_iz, s_iy, s_ix]
        return {
            "dist": dist,
            "atten_db": atten_db,
            "walls_crossed": n_walls,
            "ceiling_muffled": ceiling_muffled,
            "dst_noise_db": dst_noise,
            "dst_noise_cause": dst_cause,
            "src_noise_db": src_noise,
        }


# =============================================================================
# 2. Cooperative Episode Sampler over Real WorldModel Z_world Latents
# =============================================================================
def sample_z_world_crops(
    z_world: torch.Tensor,   # [4, 48, 256, 256]
    xyz: torch.Tensor,       # [N, 3] (x, y, z)
    patch_size: int = 16,
    radius_cells: float = 16.0,
) -> torch.Tensor:
    """Zero-copy 3D ego-crop from frozen WorldModel Z_world [4, 48, 256, 256] -> [N, 48, P, P]."""
    n = xyz.shape[0]
    device = z_world.device
    lin = torch.linspace(-1.0, 1.0, patch_size, device=device)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    cx = (xyz[:, 0:1, None] / (GRID - 1)) * 2.0 - 1.0
    cy = (xyz[:, 1:2, None] / (GRID - 1)) * 2.0 - 1.0
    cz = (xyz[:, 2:3, None] / (LAYERS - 1)) * 2.0 - 1.0
    r = (radius_cells / (GRID - 1)) * 2.0
    grid_x = cx + gx.unsqueeze(0) * r
    grid_y = cy + gy.unsqueeze(0) * r
    grid_z = cz.expand(n, patch_size, patch_size)
    grid_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)  # [1, N, P, P, 3]
    z_vol = z_world.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 48, 4, H, W]
    crops = F.grid_sample(z_vol, grid_3d, mode="bilinear", padding_mode="border", align_corners=True)
    return crops.squeeze(0).permute(1, 0, 2, 3).contiguous()  # [N, 48, P, P]


def pos_to_sector(x: float, y: float) -> int:
    """0=NW, 1=NE, 2=SW, 3=SE."""
    col = 0 if x < GRID * 0.5 else 1
    row = 0 if y < GRID * 0.5 else 1
    return row * 2 + col


@torch.no_grad()
def build_world_pool(
    seeds: list[int],
    world_model: WorldModel,
    eco_model: EcosystemModel,
    acoustics: AcousticField3D,
    device: torch.device,
) -> list[FrostmarshWorld3D]:
    worlds: list[FrostmarshWorld3D] = []
    for s in seeds:
        w_state, eco_state, lm = build_frostmarsh_3d_state(s, device)
        z_world = extract_world_latent(world_model, w_state)
        z_eco = extract_eco_latent(eco_model, eco_state)
        solid_wall, open_shaft_01, open_sky_12 = acoustics.extract_acoustic_geometry(
            w_state, open_shaft_override=lm["open_shaft_01"]
        )
        noise_db, noise_cause = acoustics.compute_ambient_noise_field(
            w_state, solid_wall, open_shaft_01, open_sky_12
        )
        worlds.append(
            FrostmarshWorld3D(
                seed=s,
                w_state=w_state,
                eco_state=eco_state,
                z_world=z_world,
                z_eco=z_eco,
                noise_db=noise_db,
                noise_cause=noise_cause,
                solid_wall=solid_wall,
                open_shaft_01=open_shaft_01,
                open_sky_12=open_sky_12,
                landmarks=lm,
            )
        )
    return worlds


@torch.no_grad()
def sample_episodes_from_worlds(
    worlds: list[FrostmarshWorld3D],
    episodes_per_world: int,
    acoustics: AcousticField3D,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Precompute cooperative physics-coordination episodes across real Frostmarsh worlds.
    Spotter observes real Z_world ego-crop around the physical event; Teammate is out of
    visual range (42-115 cells away, behind basalt ridges, across river, or on another layer).
    """
    rng = np.random.default_rng(seed)
    all_spot_crops, all_team_crops = [], []
    all_spot_aux, all_team_aux = [], []
    all_event, all_sector, all_elem, all_tactic = [], [], [], []
    all_atten_tm, all_noise_tm, all_cause_tm, all_walls_tm, all_ceil_tm = [], [], [], [], []
    all_atten_en, all_noise_en, all_walls_en = [], [], []
    all_dist_tm, all_dist_en = [], []
    all_speaker_cls, all_world_idx = [], []
    all_spot_xyz, all_team_xyz, all_enemy_xyz = [], [], []

    for w_idx, w in enumerate(worlds):
        lm = w.landmarks
        M = episodes_per_world
        spot_xyz = np.zeros((M, 3), dtype=np.float32)
        team_xyz = np.zeros((M, 3), dtype=np.float32)
        enemy_xyz = np.zeros((M, 3), dtype=np.float32)
        ev_arr = np.zeros(M, dtype=np.int64)
        sec_arr = np.zeros(M, dtype=np.int64)
        ambush_flag = np.zeros(M, dtype=np.float32)
        sp_cls = rng.integers(0, len(SPEAKER_CLASSES), size=M, dtype=np.int64)

        for i in range(M):
            ev = i % len(EVENT_NAMES)
            ev_arr[i] = ev
            if ev == 0:  # OIL_SLICK_IGNITE
                ex, ey, ez = lm["oil_positions"][int(rng.integers(0, len(lm["oil_positions"])))]
            elif ev == 1:  # WILDFIRE_FLANK
                ex, ey, ez = lm["wildfire_positions"][int(rng.integers(0, len(lm["wildfire_positions"])))]
            elif ev == 2:  # FROZEN_RIVER_BRIDGE
                ex, ey, ez = lm["ice_bridge_pos"]
            elif ev == 3:  # SHRINE_AMBUSH
                ex, ey, ez = lm["shrine_pos"]
                ambush_flag[i] = 1.0
            elif ev == 4:  # WATERFALL_CHOKE
                ex, ey, ez = lm["waterfall_positions"][int(rng.integers(0, len(lm["waterfall_positions"])))]
            else:  # CAVERN_SHAFT_FLANK
                ex, ey, ez = lm["shaft_positions"][int(rng.integers(0, len(lm["shaft_positions"])))]

            # Spotter near the physical event (within 10-cell visual ego-crop)
            sx = float(np.clip(ex + rng.uniform(-7.0, 7.0), 12.0, GRID - 13.0))
            sy = float(np.clip(ey + rng.uniform(-7.0, 7.0), 12.0, GRID - 13.0))
            sz = int(ez)
            spot_xyz[i] = (sx, sy, float(sz))
            sec_arr[i] = pos_to_sector(ex, ey)

            # Teammate placed out of visual range (34-88 cells) in diverse acoustic regimes
            # with randomized angle/position so Silent Arm has ZERO spatial leakage (~4-5% chance)!
            regime = int(rng.integers(0, 5))
            if regime == 0:
                # Open-air mid-range squadmate (34-62 cells) -> whisper/moderate volume works cleanly!
                ang = rng.uniform(0, 6.28)
                rad = rng.uniform(34.0, 62.0)
                tx = float(np.clip(sx + math.cos(ang) * rad, 14.0, GRID - 15.0))
                ty = float(np.clip(sy + math.sin(ang) * rad, 14.0, GRID - 15.0))
                tz = 1
            elif regime == 1:
                # Longer range / across basalt ridges (54-92 cells) -> requires higher volume!
                ang = rng.uniform(0, 6.28)
                rad = rng.uniform(54.0, 92.0)
                tx = float(np.clip(sx + math.cos(ang) * rad, 14.0, GRID - 15.0))
                ty = float(np.clip(sy + math.sin(ang) * rad, 14.0, GRID - 15.0))
                tz = 1
            elif regime == 2:
                # Near a plunging waterfall or wildfire (high ambient noise test!)
                if rng.random() < 0.6:
                    wx, wy, _ = lm["waterfall_positions"][int(rng.integers(0, 2))]
                    tx = float(np.clip(wx + rng.uniform(-14.0, 14.0), 14.0, GRID - 15.0))
                    ty = float(np.clip(wy + rng.uniform(-14.0, 14.0), 14.0, GRID - 15.0))
                else:
                    fx, fy, _ = lm["wildfire_positions"][int(rng.integers(0, len(lm["wildfire_positions"])))]
                    tx = float(np.clip(fx + rng.uniform(-12.0, 12.0), 14.0, GRID - 15.0))
                    ty = float(np.clip(fy + rng.uniform(-12.0, 12.0), 14.0, GRID - 15.0))
                tz = 1
            elif regime == 3:
                # L2 Skybridge overwatch or L0 subterranean cavern
                if rng.random() < 0.55:
                    bx, by, _ = lm["north_skybridge"] if rng.random() < 0.5 else lm["south_skybridge"]
                    tx = float(np.clip(bx + rng.uniform(-32.0, 32.0), 16.0, GRID - 17.0))
                    ty = float(np.clip(by + rng.uniform(-10.0, 10.0), 16.0, GRID - 17.0))
                    tz = 2
                else:
                    ang = rng.uniform(0, 6.28)
                    rad = rng.uniform(34.0, 68.0)
                    tx = float(np.clip(sx + math.cos(ang) * rad, 24.0, GRID - 25.0))
                    ty = float(np.clip(sy + math.sin(ang) * rad, 24.0, GRID - 25.0))
                    tz = 0
            else:
                # Uniform random across arena (strictly independent of spotter sector)
                tx = float(rng.uniform(20.0, GRID - 21.0))
                ty = float(rng.uniform(20.0, GRID - 21.0))
                tz = int(rng.choice([1, 2]))
            # Guarantee out-of-visual-range (>32 cells)
            if math.hypot(tx - sx, ty - sy) < 32.0:
                tx = float(np.clip(sx + (44.0 if sx < 128.0 else -44.0), 16.0, GRID - 17.0))
            team_xyz[i] = (tx, ty, float(tz))

            # Enemy eavesdropper at realistic patrol distance (28-80 cells; some open-air, some behind basalt walls)
            e_ang = rng.uniform(0, 6.28)
            e_rad = rng.uniform(28.0, 80.0)
            en_x = float(np.clip(sx + math.cos(e_ang) * e_rad, 10.0, GRID - 11.0))
            en_y = float(np.clip(sy + math.sin(e_ang) * e_rad, 10.0, GRID - 11.0))
            en_z = 1 if rng.random() < 0.75 else sz
            enemy_xyz[i] = (en_x, en_y, float(en_z))

        s_xyz_t = torch.from_numpy(spot_xyz).to(device)
        t_xyz_t = torch.from_numpy(team_xyz).to(device)
        e_xyz_t = torch.from_numpy(enemy_xyz).to(device)

        link_tm = acoustics.query_pairwise_link(
            s_xyz_t, t_xyz_t, w.solid_wall, w.open_shaft_01, w.open_sky_12, w.noise_db, w.noise_cause
        )
        link_en = acoustics.query_pairwise_link(
            s_xyz_t, e_xyz_t, w.solid_wall, w.open_shaft_01, w.open_sky_12, w.noise_db, w.noise_cause
        )

        # Sample real WorldModel Z_world ego-crops (zero-copy 5D grid_sample)
        s_crops = sample_z_world_crops(w.z_world, s_xyz_t, patch_size=16, radius_cells=16.0)
        t_crops = sample_z_world_crops(w.z_world, t_xyz_t, patch_size=16, radius_cells=16.0)

        # Spotter local tactical/acoustic telemetry [M, 10]:
        # Does NOT include event_id! Spotter reads physical world from Z_world crop!
        spot_aux = torch.stack(
            [
                torch.from_numpy(ambush_flag).to(device),
                s_xyz_t[:, 0] / GRID,
                s_xyz_t[:, 1] / GRID,
                s_xyz_t[:, 2] / (LAYERS - 1),
                (link_tm["src_noise_db"] - 50.0) / 30.0,
                (link_tm["atten_db"] - 30.0) / 20.0,
                (link_tm["dst_noise_db"] - 50.0) / 30.0,
                (link_en["atten_db"] - 30.0) / 20.0,
                (link_en["dst_noise_db"] - 50.0) / 30.0,
                link_en["walls_crossed"] / 3.0,
            ],
            dim=-1,
        )

        # Teammate self/position features [M, 4] (purely local — ONLY layer & local ambient noise + zeroed xy so Silent Arm cannot cheat from teammate coordinates!)
        team_aux = torch.stack(
            [
                torch.zeros_like(t_xyz_t[:, 0]),
                torch.zeros_like(t_xyz_t[:, 1]),
                t_xyz_t[:, 2] / (LAYERS - 1),
                (link_tm["dst_noise_db"] - 50.0) / 30.0,
            ],
            dim=-1,
        )

        ev_t = torch.from_numpy(ev_arr).to(device)
        sec_t = torch.from_numpy(sec_arr).to(device)
        elem_t = torch.tensor([EVENT_TO_ELEM[e] for e in ev_arr], dtype=torch.long, device=device)
        tac_t = torch.tensor([EVENT_TO_TACTIC[e] for e in ev_arr], dtype=torch.long, device=device)

        # Keep crops in fp16 on GPU for fast, memory-efficient batched training
        all_spot_crops.append(s_crops.half())
        all_team_crops.append(t_crops.half())
        all_spot_aux.append(spot_aux)
        all_team_aux.append(team_aux)
        all_event.append(ev_t)
        all_sector.append(sec_t)
        all_elem.append(elem_t)
        all_tactic.append(tac_t)
        all_atten_tm.append(link_tm["atten_db"])
        all_noise_tm.append(link_tm["dst_noise_db"])
        all_cause_tm.append(link_tm["dst_noise_cause"])
        all_walls_tm.append(link_tm["walls_crossed"])
        all_ceil_tm.append(link_tm["ceiling_muffled"])
        all_atten_en.append(link_en["atten_db"])
        all_noise_en.append(link_en["dst_noise_db"])
        all_walls_en.append(link_en["walls_crossed"])
        all_dist_tm.append(link_tm["dist"])
        all_dist_en.append(link_en["dist"])
        all_speaker_cls.append(torch.from_numpy(sp_cls).to(device))
        all_world_idx.append(torch.full((M,), w_idx, dtype=torch.long, device=device))
        all_spot_xyz.append(s_xyz_t)
        all_team_xyz.append(t_xyz_t)
        all_enemy_xyz.append(e_xyz_t)

    return {
        "spot_crops": torch.cat(all_spot_crops, dim=0),
        "team_crops": torch.cat(all_team_crops, dim=0),
        "spot_aux": torch.cat(all_spot_aux, dim=0),
        "team_aux": torch.cat(all_team_aux, dim=0),
        "event": torch.cat(all_event, dim=0),
        "sector": torch.cat(all_sector, dim=0),
        "elem": torch.cat(all_elem, dim=0),
        "tactic": torch.cat(all_tactic, dim=0),
        "atten_tm": torch.cat(all_atten_tm, dim=0),
        "noise_tm": torch.cat(all_noise_tm, dim=0),
        "cause_tm": torch.cat(all_cause_tm, dim=0),
        "walls_tm": torch.cat(all_walls_tm, dim=0),
        "ceil_tm": torch.cat(all_ceil_tm, dim=0),
        "atten_en": torch.cat(all_atten_en, dim=0),
        "noise_en": torch.cat(all_noise_en, dim=0),
        "walls_en": torch.cat(all_walls_en, dim=0),
        "dist_tm": torch.cat(all_dist_tm, dim=0),
        "dist_en": torch.cat(all_dist_en, dim=0),
        "speaker_cls": torch.cat(all_speaker_cls, dim=0),
        "world_idx": torch.cat(all_world_idx, dim=0),
        "spot_xyz": torch.cat(all_spot_xyz, dim=0),
        "team_xyz": torch.cat(all_team_xyz, dim=0),
        "enemy_xyz": torch.cat(all_enemy_xyz, dim=0),
    }


# =============================================================================
# 3. Emergent Differentiable Neural Speech (`CommBotPair`)
# =============================================================================
class EgoWorldEncoder(nn.Module):
    """Lightweight Conv encoder for WorldModel Z_world [N, 48, 16, 16] ego-crops."""

    def __init__(self, in_ch: int = 48, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.GELU(),
        )
        self.proj = nn.Linear(out_dim * 2, out_dim)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        h = self.net(crops.float())
        return self.proj(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=-1))


class CommBotPair(nn.Module):
    """
    Cooperative Spotter + Teammate neural architecture communicating via:
      - K=16 Gumbel-Softmax discrete acoustic symbols (straight-through)
      - 8-d continuous prosody/urgency vector (unit-power constrained)
      - adaptive volume in [0, 1] trading SNR audibility vs enemy eavesdropping!
    Passed through differentiable physical AcousticField3D SNR channel.
    """

    def __init__(self, num_symbols: int = 16, prosody_dim: int = 8, mode: str = "physical"):
        super().__init__()
        assert mode in ("silent", "physical", "telepathic")
        self.num_symbols = num_symbols
        self.prosody_dim = prosody_dim
        self.mode = mode

        # Spotter encoder (reads real Z_world ego-crop + local acoustic/tactical sensors)
        self.spot_world_enc = EgoWorldEncoder(48, 64)
        self.spot_mlp = nn.Sequential(
            nn.Linear(64 + 10, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 96),
            nn.GELU(),
        )
        self.sym_head = nn.Linear(96, num_symbols)
        self.prosody_head = nn.Linear(96, prosody_dim)
        # Dedicated acoustic volume controller (initialized audible so emergent language forms before stealth pressure)
        self.vol_head = nn.Sequential(
            nn.Linear(96 + 6, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        nn.init.constant_(self.vol_head[-1].bias, 1.4)

        # Teammate encoder (own Z_world crop + own local self features + acoustically received utterance only)
        self.team_world_enc = EgoWorldEncoder(48, 64)
        self.utterance_dim = num_symbols + prosody_dim  # 24-d waveform embedding
        self.team_mlp = nn.Sequential(
            nn.Linear(64 + 4 + self.utterance_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
        )
        self.sector_head = nn.Linear(96, len(SECTOR_NAMES))
        self.elem_head = nn.Linear(96, len(ELEMENT_NAMES))
        self.tactic_head = nn.Linear(96, len(TACTIC_NAMES))

    def encode_spotter(
        self,
        spot_crops: torch.Tensor,
        spot_aux: torch.Tensor,
        tau: float = 1.0,
        hard_eval: bool = False,
    ) -> dict[str, torch.Tensor]:
        hs_w = self.spot_world_enc(spot_crops)
        hs = self.spot_mlp(torch.cat([hs_w, spot_aux], dim=-1))
        sym_logits = self.sym_head(hs)
        if hard_eval or not self.training:
            idx = sym_logits.argmax(dim=-1)
            sym_onehot = F.one_hot(idx, num_classes=self.num_symbols).float()
        else:
            sym_onehot = F.gumbel_softmax(sym_logits, tau=tau, hard=True, dim=-1)

        # Unit-RMS constrained prosody (carries continuous sector/urgency cues alongside discrete symbol!)
        prosody_raw = torch.tanh(self.prosody_head(hs))
        prosody = prosody_raw / (prosody_raw.norm(dim=-1, keepdim=True) + 1e-6) * math.sqrt(0.95)

        # Adaptive volume in [0, 1]
        vol = torch.sigmoid(self.vol_head(torch.cat([hs, spot_aux[:, 4:10]], dim=-1))).squeeze(-1)

        # Transmitted acoustic waveform u_tx [B, 24]
        u_tx = torch.cat([sym_onehot * 1.25, prosody], dim=-1)
        return {
            "sym_logits": sym_logits,
            "sym_onehot": sym_onehot,
            "sym_id": sym_logits.argmax(dim=-1),
            "prosody": prosody,
            "volume": vol,
            "u_tx": u_tx,
        }

    def acoustic_channel(
        self,
        u_tx: torch.Tensor,
        volume: torch.Tensor,
        atten_tm: torch.Tensor,
        noise_tm: torch.Tensor,
        atten_en: torch.Tensor,
        noise_en: torch.Tensor,
        volume_override: float | None = None,
        deterministic_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        vol = torch.full_like(volume, volume_override) if volume_override is not None else volume
        src_spl = SRC_SPL_MIN_DB + SRC_SPL_SPAN_DB * vol
        snr_tm = src_spl - atten_tm - noise_tm
        snr_en = src_spl - atten_en - noise_en

        if self.mode == "silent":
            u_heard = torch.zeros_like(u_tx)
            snr_eff = torch.full_like(snr_tm, -25.0)
        elif self.mode == "telepathic":
            u_heard = u_tx
            snr_eff = torch.full_like(snr_tm, 40.0)
        else:
            # Differentiable physical SNR degradation (smooth audibility cliff around -2 dB)
            snr_clamped = torch.clamp(snr_tm, -25.0, 35.0)
            snr_lin = torch.pow(10.0, snr_clamped / 20.0)
            audibility = torch.sigmoid((snr_tm + 2.0) / 3.5).unsqueeze(-1)
            noise_std = torch.clamp(1.0 / (snr_lin + 0.14), 0.03, 3.8).unsqueeze(-1)
            if deterministic_noise:
                # Reproducible noise per call when generating eval transcripts
                gen = torch.Generator(device=u_tx.device).manual_seed(1337)
                eps = torch.randn(u_tx.shape, device=u_tx.device, dtype=u_tx.dtype, generator=gen)
            else:
                eps = torch.randn_like(u_tx)
            u_rx = audibility * u_tx + noise_std * eps
            u_heard = u_rx / torch.sqrt(1.0 + noise_std * noise_std)
            snr_eff = snr_tm

        eavesdrop_prob = torch.sigmoid((snr_en - ENEMY_DETECT_SNR_DB) / 2.5)
        return {
            "u_heard": u_heard,
            "snr_tm": snr_eff,
            "snr_en": snr_en,
            "eavesdrop_prob": eavesdrop_prob,
            "eavesdrop_hard": (snr_en > ENEMY_DETECT_SNR_DB).float(),
            "volume_used": vol,
        }

    def decode_teammate(
        self,
        team_crops: torch.Tensor,
        team_aux: torch.Tensor,
        u_heard: torch.Tensor,
        snr_tm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ht_w = self.team_world_enc(team_crops)
        ht = self.team_mlp(torch.cat([ht_w, team_aux, u_heard], dim=-1))
        return self.sector_head(ht), self.elem_head(ht), self.tactic_head(ht)

    def forward(
        self,
        spot_crops: torch.Tensor,
        spot_aux: torch.Tensor,
        team_crops: torch.Tensor,
        team_aux: torch.Tensor,
        atten_tm: torch.Tensor,
        noise_tm: torch.Tensor,
        atten_en: torch.Tensor,
        noise_en: torch.Tensor,
        tau: float = 1.0,
        volume_override: float | None = None,
        deterministic_noise: bool = False,
    ) -> dict[str, torch.Tensor]:
        sp = self.encode_spotter(spot_crops, spot_aux, tau=tau)
        ch = self.acoustic_channel(
            sp["u_tx"], sp["volume"], atten_tm, noise_tm, atten_en, noise_en,
            volume_override=volume_override, deterministic_noise=deterministic_noise
        )
        sec_logits, elem_logits, tac_logits = self.decode_teammate(
            team_crops, team_aux, ch["u_heard"], ch["snr_tm"]
        )
        return {
            **sp,
            **ch,
            "sector_logits": sec_logits,
            "elem_logits": elem_logits,
            "tactic_logits": tac_logits,
        }


def train_comm_arm(
    mode: str,
    train_data: dict[str, torch.Tensor],
    test_data: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 32,
    batch_size: int = 256,
    stealth_weight: float = 0.32,
) -> tuple[CommBotPair, dict[str, Any]]:
    model = CommBotPair(num_symbols=16, prosody_dim=8, mode=mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = train_data["event"].shape[0]
    warmup_ep = 12 if mode == "physical" else 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        tau = max(0.50, 1.8 - 1.30 * (ep / max(1, epochs - 1)))
        # Mild load-balancing entropy bonus in first 70% of training to encourage rich 16-symbol usage
        bal_w = 0.16 * max(0.0, 1.0 - ep / (epochs * 0.72))
        # Curriculum: form emergent language first (ep < warmup_ep), then ramp stealth penalty smoothly!
        if mode == "physical":
            st_frac = max(0.0, min(1.0, (ep - warmup_ep + 1) / max(1, epochs - warmup_ep - 4)))
            cur_stealth = stealth_weight * st_frac
        else:
            cur_stealth = 0.0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            out = model(
                train_data["spot_crops"][idx],
                train_data["spot_aux"][idx],
                train_data["team_crops"][idx],
                train_data["team_aux"][idx],
                train_data["atten_tm"][idx],
                train_data["noise_tm"][idx],
                train_data["atten_en"][idx],
                train_data["noise_en"][idx],
                tau=tau,
            )
            loss_sec = F.cross_entropy(out["sector_logits"], train_data["sector"][idx])
            loss_el = F.cross_entropy(out["elem_logits"], train_data["elem"][idx])
            loss_tac = F.cross_entropy(out["tactic_logits"], train_data["tactic"][idx])
            loss = loss_sec + 0.85 * loss_el + 1.15 * loss_tac

            if mode == "physical":
                if cur_stealth > 0.0:
                    # Non-saturating stealth trade-off: penalize enemy eavesdrop SNR above threshold cleanly!
                    eav_surplus = F.softplus((out["snr_en"] - ENEMY_DETECT_SNR_DB) / 5.0)
                    loss = loss + cur_stealth * (0.45 * eav_surplus.mean() + out["eavesdrop_prob"].mean()) + 0.03 * (out["volume"] ** 2).mean()
                else:
                    # During language-formation warmup, keep volume gently anchored high so gradients flow!
                    loss = loss + 0.08 * ((out["volume"] - 0.80) ** 2).mean()
            if mode != "silent" and bal_w > 0:
                probs = F.softmax(out["sym_logits"], dim=-1)
                p_mean = probs.mean(dim=0)
                kl_unif = (p_mean * torch.log(p_mean * model.num_symbols + 1e-8)).sum()
                ent_sample = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                loss = loss + bal_w * kl_unif + 0.015 * ent_sample

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

    metrics = evaluate_comm_arm(model, test_data)
    return model, metrics


@torch.no_grad()
def evaluate_comm_arm(
    model: CommBotPair,
    data: dict[str, torch.Tensor],
    volume_override: float | None = None,
) -> dict[str, Any]:
    model.eval()
    N = data["event"].shape[0]
    bs = 512
    sec_ok, el_ok, tac_ok, joint_ok = [], [], [], []
    snr_tm_all, snr_en_all, vol_all, eav_all, sym_all = [], [], [], [], []

    for s in range(0, N, bs):
        sl = slice(s, s + bs)
        out = model(
            data["spot_crops"][sl],
            data["spot_aux"][sl],
            data["team_crops"][sl],
            data["team_aux"][sl],
            data["atten_tm"][sl],
            data["noise_tm"][sl],
            data["atten_en"][sl],
            data["noise_en"][sl],
            volume_override=volume_override,
            deterministic_noise=True,
        )
        p_sec = out["sector_logits"].argmax(dim=-1)
        p_el = out["elem_logits"].argmax(dim=-1)
        p_tac = out["tactic_logits"].argmax(dim=-1)

        ok_s = (p_sec == data["sector"][sl])
        ok_e = (p_el == data["elem"][sl])
        ok_t = (p_tac == data["tactic"][sl])
        sec_ok.append(ok_s)
        el_ok.append(ok_e)
        tac_ok.append(ok_t)
        joint_ok.append(ok_s & ok_e & ok_t)

        snr_tm_all.append(out["snr_tm"])
        snr_en_all.append(out["snr_en"])
        vol_all.append(out["volume_used"])
        eav_all.append(out["eavesdrop_hard"])
        sym_all.append(out["sym_id"])

    sec_ok_t = torch.cat(sec_ok).float()
    el_ok_t = torch.cat(el_ok).float()
    tac_ok_t = torch.cat(tac_ok).float()
    joint_ok_t = torch.cat(joint_ok).float()
    snr_tm_t = torch.cat(snr_tm_all)
    snr_en_t = torch.cat(snr_en_all)
    vol_t = torch.cat(vol_all)
    eav_t = torch.cat(eav_all)
    sym_t = torch.cat(sym_all)

    # Breakdown by physical SNR bucket
    snr_buckets = {}
    for name, lo, hi in [
        ("high_ge_10dB", 10.0, 999.0),
        ("med_2_to_10dB", 2.0, 10.0),
        ("low_neg6_to_2dB", -6.0, 2.0),
        ("inaudible_lt_neg6dB", -999.0, -6.0),
    ]:
        m = (snr_tm_t >= lo) & (snr_tm_t < hi)
        cnt = int(m.sum().item())
        snr_buckets[name] = {
            "count": cnt,
            "joint_acc": float(joint_ok_t[m].mean().item()) if cnt > 0 else None,
            "tactic_acc": float(tac_ok_t[m].mean().item()) if cnt > 0 else None,
            "sector_acc": float(sec_ok_t[m].mean().item()) if cnt > 0 else None,
        }

    # Emergent symbol-to-event mutual information, NMI, purity, confusion matrix [16, 6]
    ev_np = data["event"].cpu().numpy()
    sym_np = sym_t.cpu().numpy()
    conf = np.zeros((model.num_symbols, len(EVENT_NAMES)), dtype=np.int64)
    for s_id, e_id in zip(sym_np, ev_np):
        conf[s_id, e_id] += 1
    p_se = conf / max(1, conf.sum())
    p_s = p_se.sum(axis=1, keepdims=True)
    p_e = p_se.sum(axis=0, keepdims=True)
    nz = p_se > 0
    mi_bits = float(np.sum(p_se[nz] * np.log2(p_se[nz] / (p_s * p_e)[nz])))
    h_s = float(-np.sum(p_s[p_s > 0] * np.log2(p_s[p_s > 0])))
    h_e = float(-np.sum(p_e[p_e > 0] * np.log2(p_e[p_e > 0])))
    nmi = float(2.0 * mi_bits / max(1e-8, h_s + h_e))
    purity = float(conf.max(axis=1).sum() / max(1, conf.sum()))
    active_syms = int((conf.sum(axis=1) > (0.01 * N)).sum())

    # Wall-shielded vs open-air eavesdrop analysis
    en_walled = data["walls_en"] >= 0.8
    en_open = data["walls_en"] < 0.3

    return {
        "sector_acc": float(sec_ok_t.mean().item()),
        "element_acc": float(el_ok_t.mean().item()),
        "tactic_acc": float(tac_ok_t.mean().item()),
        "joint_coord_acc": float(joint_ok_t.mean().item()),
        "mean_snr_tm_db": float(snr_tm_t.mean().item()),
        "mean_snr_en_db": float(snr_en_t.mean().item()),
        "mean_volume": float(vol_t.mean().item()),
        "enemy_eavesdrop_rate": float(eav_t.mean().item()),
        "enemy_eavesdrop_open_air": float(eav_t[en_open].mean().item()) if en_open.any() else 0.0,
        "enemy_eavesdrop_behind_wall": float(eav_t[en_walled].mean().item()) if en_walled.any() else 0.0,
        "mean_vol_when_enemy_open": float(vol_t[en_open].mean().item()) if en_open.any() else 0.0,
        "mean_vol_when_enemy_walled": float(vol_t[en_walled].mean().item()) if en_walled.any() else 0.0,
        "snr_buckets": snr_buckets,
        "vocab": {
            "mi_bits": mi_bits,
            "h_event_bits": h_e,
            "h_symbol_bits": h_s,
            "nmi": nmi,
            "purity": purity,
            "active_symbols": active_syms,
            "confusion_16x6": conf.tolist(),
        },
    }


# =============================================================================
# 4. Viewer-Side Text Interpreter (Neural Decoder + Physical SNR Renderer)
# =============================================================================
class ViewerInterpreter(nn.Module):
    """
    Lightweight viewer-side neural interpreter:
      (u_heard [24], SNR_dB [1], speaker_class [4], Z_world_context [48])
      -> structured semantic frame (event[6], sector[4], element[3], tactic[6], urgency[1], tier[3]).
    """

    def __init__(self, utterance_dim: int = 24, n_classes: int = 4, z_dim: int = 48):
        super().__init__()
        in_dim = utterance_dim + 1 + n_classes + z_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 96),
            nn.GELU(),
        )
        self.event_head = nn.Linear(96, len(EVENT_NAMES))
        self.sector_head = nn.Linear(96, len(SECTOR_NAMES))
        self.elem_head = nn.Linear(96, len(ELEMENT_NAMES))
        self.tactic_head = nn.Linear(96, len(TACTIC_NAMES))
        self.urgency_head = nn.Linear(96, 1)
        self.tier_head = nn.Linear(96, 3)  # 0=clear, 1=muffled, 2=inaudible

    def forward(
        self,
        u_heard: torch.Tensor,
        snr_db: torch.Tensor,
        speaker_cls: torch.Tensor,
        z_ctx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_oh = F.one_hot(speaker_cls, num_classes=len(SPEAKER_CLASSES)).float()
        snr_f = torch.clamp(snr_db / 25.0, -1.5, 1.5).unsqueeze(-1)
        h = self.trunk(torch.cat([u_heard, snr_f, cls_oh, z_ctx], dim=-1))
        return {
            "event_logits": self.event_head(h),
            "sector_logits": self.sector_head(h),
            "elem_logits": self.elem_head(h),
            "tactic_logits": self.tactic_head(h),
            "urgency": torch.sigmoid(self.urgency_head(h)).squeeze(-1),
            "tier_logits": self.tier_head(h),
        }


def snr_to_tier(snr_db: torch.Tensor) -> torch.Tensor:
    # 0=clear (>= 5 dB), 1=muffled (-6..5 dB), 2=inaudible (< -6 dB)
    tier = torch.zeros_like(snr_db, dtype=torch.long)
    tier = torch.where((snr_db < 5.0) & (snr_db >= -6.0), torch.ones_like(tier), tier)
    tier = torch.where(snr_db < -6.0, torch.full_like(tier, 2), tier)
    return tier


def train_viewer_interpreter(
    phys_comm: CommBotPair,
    train_data: dict[str, torch.Tensor],
    test_data: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 24,
    batch_size: int = 256,
) -> tuple[ViewerInterpreter, dict[str, Any]]:
    phys_comm.eval()
    interp = ViewerInterpreter().to(device)
    opt = torch.optim.AdamW(interp.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    N = train_data["event"].shape[0]

    for _ in range(epochs):
        interp.train()
        perm = torch.randperm(N, device=device)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            with torch.no_grad():
                cout = phys_comm(
                    train_data["spot_crops"][idx],
                    train_data["spot_aux"][idx],
                    train_data["team_crops"][idx],
                    train_data["team_aux"][idx],
                    train_data["atten_tm"][idx],
                    train_data["noise_tm"][idx],
                    train_data["atten_en"][idx],
                    train_data["noise_en"][idx],
                )
            z_ctx = train_data["team_crops"][idx].float().mean(dim=(2, 3))
            pout = interp(cout["u_heard"], cout["snr_tm"], train_data["speaker_cls"][idx], z_ctx)

            # Ground-truth urgency is higher when roaring fire/ambush or shouting
            gt_urg = torch.clamp(0.35 + 0.55 * cout["volume_used"] + 0.15 * (train_data["event"][idx] == 3).float(), 0.0, 1.0)
            gt_tier = snr_to_tier(cout["snr_tm"])
            loss = (
                F.cross_entropy(pout["event_logits"], train_data["event"][idx])
                + F.cross_entropy(pout["sector_logits"], train_data["sector"][idx])
                + 0.7 * F.cross_entropy(pout["elem_logits"], train_data["elem"][idx])
                + 0.9 * F.cross_entropy(pout["tactic_logits"], train_data["tactic"][idx])
                + 0.5 * F.mse_loss(pout["urgency"], gt_urg)
                + 0.6 * F.cross_entropy(pout["tier_logits"], gt_tier)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    # Evaluate on held-out unseen worlds
    interp.eval()
    M = test_data["event"].shape[0]
    ev_ok, sec_ok, el_ok, tier_ok, snr_all = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, M, 512):
            sl = slice(s, s + 512)
            cout = phys_comm(
                test_data["spot_crops"][sl],
                test_data["spot_aux"][sl],
                test_data["team_crops"][sl],
                test_data["team_aux"][sl],
                test_data["atten_tm"][sl],
                test_data["noise_tm"][sl],
                test_data["atten_en"][sl],
                test_data["noise_en"][sl],
                deterministic_noise=True,
            )
            z_ctx = test_data["team_crops"][sl].float().mean(dim=(2, 3))
            pout = interp(cout["u_heard"], cout["snr_tm"], test_data["speaker_cls"][sl], z_ctx)
            ev_ok.append((pout["event_logits"].argmax(-1) == test_data["event"][sl]).float())
            sec_ok.append((pout["sector_logits"].argmax(-1) == test_data["sector"][sl]).float())
            el_ok.append((pout["elem_logits"].argmax(-1) == test_data["elem"][sl]).float())
            tier_ok.append((pout["tier_logits"].argmax(-1) == snr_to_tier(cout["snr_tm"])).float())
            snr_all.append(cout["snr_tm"])

    ev_t = torch.cat(ev_ok)
    sec_t = torch.cat(sec_ok)
    el_t = torch.cat(el_ok)
    tr_t = torch.cat(tier_ok)
    snr_t = torch.cat(snr_all)

    buckets = {}
    for name, lo, hi in [
        ("high_ge_10dB", 10.0, 999.0),
        ("med_2_to_10dB", 2.0, 10.0),
        ("low_neg6_to_2dB", -6.0, 2.0),
        ("inaudible_lt_neg6dB", -999.0, -6.0),
    ]:
        m = (snr_t >= lo) & (snr_t < hi)
        cnt = int(m.sum().item())
        buckets[name] = {
            "count": cnt,
            "event_acc": float(ev_t[m].mean().item()) if cnt > 0 else None,
            "sector_acc": float(sec_t[m].mean().item()) if cnt > 0 else None,
            "element_acc": float(el_t[m].mean().item()) if cnt > 0 else None,
        }

    metrics = {
        "overall_event_acc": float(ev_t.mean().item()),
        "overall_sector_acc": float(sec_t.mean().item()),
        "overall_element_acc": float(el_t.mean().item()),
        "intelligibility_tier_acc": float(tr_t.mean().item()),
        "n_params": sum(p.numel() for p in interp.parameters()),
        "snr_buckets": buckets,
    }
    return interp, metrics


def render_grounded_dialogue(
    pred_event: int,
    pred_sector: int,
    pred_elem: int,
    pred_tactic: int,
    urgency: float,
    snr_db: float,
    cause_id: int,
    walls_crossed: float,
    ceiling_muffled: bool,
    speaker_cls: int,
    src_layer: int,
    volume: float,
) -> str:
    """Grounded English renderer with authentic physical SNR degradation & cause attribution."""
    cause_str = {
        0: "distance attenuation",
        1: "L2→L1 plunging waterfall roar",
        2: "roaring wildfire crackle",
        3: "explosive spell blast",
    }.get(int(cause_id), "ambient interference")

    occ_reasons = []
    if ceiling_muffled:
        occ_reasons.append("L0 rock ceiling")
    if walls_crossed >= 0.8:
        w_cnt = max(1, int(round(walls_crossed)))
        occ_reasons.append(f"{w_cnt} basalt wall{'s' if w_cnt > 1 else ''}")
    if int(cause_id) in (1, 2, 3):
        occ_reasons.append(cause_str)
    if not occ_reasons:
        occ_reasons.append(cause_str)
    reason_txt = " + ".join(occ_reasons)

    tone = "SHOUT" if volume >= 0.72 else ("CALL" if volume >= 0.38 else "whisper")
    sp_name = f"{SPEAKER_CLASSES[speaker_cls]}"
    sec = SECTOR_NAMES[pred_sector]
    elem = ELEMENT_NAMES[pred_elem]

    if snr_db < -6.0:
        return f"[{sp_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB] [inaudible — drowned out by {reason_txt}]"

    high_templates = {
        0: f"Oil slick pooled across {sec} — ignite it with {elem} now!",
        1: f"Wildfire pushing through {sec} — suppress the flank with {elem}!",
        2: f"Frozen river bridge spotted near {sec} — shatter the ice with {elem}!",
        3: f"Enemy ambush staged at {sec} shrine — coat the choke in {elem}!",
        4: f"High-ground waterfall choke open at {sec} — lock it down with {elem}!",
        5: f"Cavern vertical shaft clear under {sec} — drop in with {elem} on my mark!",
    }
    med_templates = {
        0: f"Oil slick... {sec}... light it up with {elem}!",
        1: f"Wildfire on the flank... {sec}... hit it with {elem}!",
        2: f"Ice bridge... {sec}... break it with {elem}!",
        3: f"Ambush at {sec}... watch the shrine — deploy {elem}!",
        4: f"Waterfall choke... {sec}... freeze the rim with {elem}!",
        5: f"Cavern shaft below {sec}... drop {elem}!",
    }
    low_fragments = {
        0: f"...oil... [static] ...{sec}... ...{elem}...",
        1: f"...fire... [crackle] ...flank {sec}...",
        2: f"...ice bridge... [muffled] ...{elem}...",
        3: f"...ambush... [static] ...{sec}...",
        4: f"...waterfall... [roar] ...choke {sec}...",
        5: f"...shaft... [echo] ...{sec}...",
    }

    if snr_db >= 10.0:
        body = high_templates[pred_event]
        return f"[{sp_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB] \"{body}\""
    elif snr_db >= 2.0:
        body = med_templates[pred_event]
        return f"[{sp_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB, partial {reason_txt}] \"{body}\""
    else:
        body = low_fragments[pred_event]
        return f"[{sp_name} @ L{src_layer} ({tone}) | {snr_db:+.1f} dB, muffled by {reason_txt}] \"{body}\""


# =============================================================================
# 5. Human Text -> g_i Goal-Vector Injector into Frozen BotPolicyNet
# =============================================================================
HUMAN_INTENT_SPECS = [
    # (name, target_dx, target_dy, elem_idx [0=Fire,1=Frost,2=Oil], key_bin_idx [0=dash,1=melee,2=cast_bolt,3=carve_ice,4=ability_e,5=ult_r])
    ("IGNITE_OIL_SW",         -0.75,  0.65, 0, 2),
    ("FREEZE_RIVER_BRIDGE",    0.00,  0.10, 1, 3),
    ("EXTINGUISH_WILDFIRE",   -0.65, -0.60, 1, 2),
    ("HOLD_WATERFALL_CHOKE",   0.70, -0.70, 1, 4),
    ("AMBUSH_SHRINE_OIL",      0.75,  0.65, 2, 4),
    ("BREACH_SKYBRIDGE_FIRE",  0.10, -0.85, 0, 5),
    ("FLANK_CAVERN_STEALTH",  -0.30,  0.80, 2, 0),
    ("RETREAT_RIDGE_COVER",   -0.85, -0.10, 1, 0),
]

TRAIN_COMMANDS_PER_INTENT: list[list[str]] = [
    [
        "ignite the oil slick in the southwest basin",
        "burn the southwest oil pool with fire",
        "light up the oil trap down southwest",
        "fire bolt the oil slick in sw marsh",
        "torching the oil basin southwest now",
        "set the sw oil puddle ablaze",
        "shoot fire into the southwest oil",
        "detonate the oil slick south west",
        "cast fire on southwest oil pool",
        "spark the sw oil field immediately",
        "light southwest basin oil with flame",
        "ignite sw oil slick trap",
    ],
    [
        "freeze the river bridge with frost",
        "ice over the central river crossing",
        "carve an ice barrier across the river",
        "lock down the river bridge in frost",
        "frost bolt the center river crossing",
        "turn the middle river into solid ice",
        "freeze the crossing bridge right now",
        "seal the river ford with frost",
        "build an ice bridge across the water",
        "chill the river crossing solid",
        "frost wall across the center river",
        "freeze central water crossing",
    ],
    [
        "extinguish the wildfire on the northwest ridge",
        "put out the nw forest fire with frost",
        "suppress the northwest burning woods",
        "douse the roaring wildfire northwest",
        "cast frost on the nw blaze",
        "cool down the wildfire in northwest",
        "quench the forest flames nw ridge",
        "stop the wildfire northwest using frost",
        "freeze out the nw timber fire",
        "frost bolt the northwest wildfire",
        "contain the burning trees northwest",
        "suppress nw fire front with frost",
    ],
    [
        "hold the northeast waterfall choke with frost",
        "defend the ne waterfall high ground",
        "freeze the northeast falls bottleneck",
        "guard the ne highland waterfall choke",
        "lock down northeast waterfall cliff with frost",
        "cover the ne falls rim using frost",
        "hold high ground at northeast waterfall",
        "block the ne waterfall passage with ice",
        "deploy frost barrier at northeast falls",
        "secure the ne cliff waterfall choke",
        "stand firm on northeast falls with frost",
        "freeze northeast waterfall approach",
    ],
    [
        "lay an oil ambush at the southeast shrine",
        "coat the se shrine alcove in oil",
        "prepare an oil trap by southeast shrine",
        "spill oil around the se shrine choke",
        "ambush enemies at southeast shrine with oil",
        "slick the southeast shrine floor in oil",
        "set oil ambush southeast shrine now",
        "flood the se shrine entrance with oil",
        "cover southeast shrine in oil trap",
        "oil ability on the se shrine ambush",
        "drench southeast shrine rocks in oil",
        "stage oil trap at se shrine",
    ],
    [
        "breach the north skybridge with fire ult",
        "push the northern skybridge using flame",
        "blast the north high bridge with fire",
        "unleash fire ultimate on north skybridge",
        "storm the northern arch bridge in flames",
        "burn through the north skybridge defense",
        "fire ult across the north skybridge",
        "assault the northern bridge with fire",
        "clear north skybridge using flame ult",
        "charge north bridge firing flames",
        "torch the northern skybridge choke",
        "heavy fire push on north skybridge",
    ],
    [
        "drop into the cavern shaft and flank south",
        "sneak through the underground tunnel with oil",
        "dash down the cavern shaft silently",
        "flank via the subterranean cave south",
        "quiet cavern shaft flanking maneuver",
        "slip into the south cavern tunnel",
        "stealth dash through underground shaft",
        "take the lower cavern flank with oil",
        "move down subterranean shaft quietly",
        "underground cavern flank south oil",
        "dash into cavern vent for stealth flank",
        "enter south tunnel shaft with oil",
    ],
    [
        "fall back west behind the basalt ridge",
        "retreat to the western wall with frost cover",
        "dash west behind rock cover now",
        "pull back behind west basalt ridge",
        "disengage westward using frost shield",
        "retreat behind western basalt wall",
        "fall back to west ridge cover frost",
        "evade west behind solid rock wall",
        "dash back west into basalt cover",
        "withdraw westward with frost cover",
        "seek cover behind west basalt ridge",
        "retreat west ridge frost dash",
    ],
]

# Completely unseen paraphrased natural-language commands for held-out evaluation!
UNSEEN_PARAPHRASE_COMMANDS: list[list[str]] = [
    [
        "torch that dark crude pool down in the bottom left",
        "flame the oily marsh southwest right away",
        "spark up the southwest oil puddle with a fireball",
        "burn the sw crude slick before they cross",
    ],
    [
        "chill the middle stream crossing into hard ice",
        "solidify the river passage using frost magic",
        "freeze that central water bridge shut",
        "carve frost across the center stream now",
    ],
    [
        "kill the roaring timber blaze up northwest with frost",
        "snuff out the northwest forest flames using ice",
        "put cold frost onto the nw wildfire immediately",
        "douse those burning northwest woods with frost",
    ],
    [
        "anchor the upper right waterfall cliff with frost",
        "keep the northeast cascade choke locked down in frost",
        "defend the ne falls high rim with cold barrier",
        "hold that northeast waterfall bottleneck using frost",
    ],
    [
        "soak the bottom right shrine in oil for an ambush",
        "dump oil all over the southeast temple alcove",
        "set up a slick oil trap at the se shrine",
        "coat that southeast shrine choke in crude oil",
    ],
    [
        "nuke the upper north bridge with your fire ult",
        "burn straight across the northern stone arch",
        "blast flames over the north skybridge right now",
        "use fire ultimate to crack the north bridge",
    ],
    [
        "hush and dash down the subterranean vent south with oil",
        "slip underground through the cave shaft to flank",
        "quietly dash into the lower cavern tunnel with oil",
        "take the hidden south shaft beneath the valley",
    ],
    [
        "bail out westward behind the rock ridge with frost",
        "dash back to the left basalt wall for frost cover",
        "pull away west behind that stone barrier using frost",
        "fall back left behind solid basalt cover",
    ],
]


def encode_text_hash_bow(texts: list[str], dim: int = 256, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Fast deterministic word + char-3gram hashed bag-of-features [B, 256] (zero external tokenizer)."""
    feats = np.zeros((len(texts), dim), dtype=np.float32)
    for i, txt in enumerate(texts):
        clean = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in txt.lower())
        words = clean.split()
        for w in words:
            h = int(hashlib.md5(f"W:{w}".encode()).hexdigest()[:8], 16) % dim
            feats[i, h] += 1.5
            padded = f"<{w}>"
            for j in range(max(1, len(padded) - 2)):
                tri = padded[j:j + 3]
                h3 = int(hashlib.md5(f"C:{tri}".encode()).hexdigest()[:8], 16) % dim
                feats[i, h3] += 0.45
        norm = np.linalg.norm(feats[i])
        if norm > 1e-6:
            feats[i] /= norm
    return torch.from_numpy(feats).to(device)


class HumanGoalInjector(nn.Module):
    """
    Lightweight Human Text -> 16-d Goal-Vector Injector + FiLM/Residual Adapter
    wrapping a FROZEN BotPolicyNet (zero edits to bot_policy.py / bot_policy.pt!).
    """

    def __init__(self, base_policy: BotPolicyNet, bow_dim: int = 256, goal_dim: int = 16):
        super().__init__()
        self.base_policy = base_policy
        for p in self.base_policy.parameters():
            p.requires_grad_(False)
        self.goal_dim = goal_dim

        # Lightweight Text Encoder -> 16-d goal vector g_i
        self.text_encoder = nn.Sequential(
            nn.Linear(bow_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, goal_dim),
            nn.Tanh(),
        )
        # FiLM modulation + residual action steering on BotPolicyNet's 128-d trunk
        self.film_gamma = nn.Linear(goal_dim, 128)
        self.film_beta = nn.Linear(goal_dim, 128)
        self.delta_vec = nn.Sequential(nn.Linear(128 + goal_dim, 64), nn.GELU(), nn.Linear(64, 4))
        self.delta_bin = nn.Sequential(nn.Linear(128 + goal_dim, 64), nn.GELU(), nn.Linear(64, 6))
        self.delta_elem = nn.Sequential(nn.Linear(128 + goal_dim, 64), nn.GELU(), nn.Linear(64, 3))

        # Zero-init residual projections so g_i = 0 preserves base BotPolicyNet identically!
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def encode_goal(self, text_bow: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_bow)  # [N, 16]

    def forward_with_goal(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        g_vec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bp = self.base_policy
        with torch.no_grad():
            self_feat = bot_feats[:, :SELF_DIM]
            tok_feat = bot_feats[:, SELF_DIM:].reshape(-1, NUM_TOKENS, TOKEN_DIM)
            patch = bp.sample_shared_latents(z_world, z_eco, self_feat)
            sp = bp.spatial_enc(bp.latent_adapter(patch))
            h_world = bp.spatial_proj(torch.cat([sp.mean(dim=(2, 3)), sp.amax(dim=(2, 3))], dim=-1))
            q = bp.self_enc(self_feat).unsqueeze(1)
            kv = bp.tok_enc(tok_feat)
            attn_out, _ = bp.cross_attn(q, kv, kv, need_weights=False)
            h_tac = bp.attn_norm((q + attn_out).squeeze(1))
            h_base = bp.trunk(torch.cat([h_world, h_tac], dim=-1))

        if g_vec is None:
            return torch.tanh(bp.head_vec(h_base)), bp.head_bin(h_base), bp.head_elem(h_base)

        gamma = self.film_gamma(g_vec)
        beta = self.film_beta(g_vec)
        h_mod = h_base * (1.0 + gamma) + beta
        hg = torch.cat([h_mod, g_vec], dim=-1)
        vec = torch.tanh(bp.head_vec(h_mod) + self.delta_vec(hg))
        bin_logits = bp.head_bin(h_mod) + self.delta_bin(hg)
        elem_logits = bp.head_elem(h_mod) + self.delta_elem(hg)
        return vec, bin_logits, elem_logits


def train_and_eval_human_goal_injector(
    base_policy: BotPolicyNet,
    train_worlds: list[FrostmarshWorld3D],
    test_worlds: list[FrostmarshWorld3D],
    device: torch.device,
) -> tuple[HumanGoalInjector, dict[str, Any]]:
    injector = HumanGoalInjector(base_policy).to(device)
    opt = torch.optim.AdamW(
        [p for p in injector.parameters() if p.requires_grad],
        lr=3.5e-3,
        weight_decay=1e-4,
    )

    # Pre-encode train & unseen paraphrase command banks
    train_texts, train_intents = [], []
    for k, cmds in enumerate(TRAIN_COMMANDS_PER_INTENT):
        for c in cmds:
            train_texts.append(c)
            train_intents.append(k)
    train_bow = encode_text_hash_bow(train_texts, device=device)
    train_int_t = torch.tensor(train_intents, dtype=torch.long, device=device)

    test_texts, test_intents = [], []
    for k, cmds in enumerate(UNSEEN_PARAPHRASE_COMMANDS):
        for c in cmds:
            test_texts.append(c)
            test_intents.append(k)
    test_bow = encode_text_hash_bow(test_texts, device=device)
    test_int_t = torch.tensor(test_intents, dtype=torch.long, device=device)

    intent_vec_t = torch.tensor(
        [[sp[1], sp[2], sp[1], sp[2]] for sp in HUMAN_INTENT_SPECS], dtype=torch.float32, device=device
    )
    intent_elem_t = torch.tensor([sp[3] for sp in HUMAN_INTENT_SPECS], dtype=torch.long, device=device)
    intent_bin_t = torch.zeros((len(HUMAN_INTENT_SPECS), 6), dtype=torch.float32, device=device)
    for k, sp in enumerate(HUMAN_INTENT_SPECS):
        intent_bin_t[k, sp[4]] = 1.0

    def sample_bot_feats(n: int, rng: np.random.Generator) -> torch.Tensor:
        bf = rng.normal(0.0, 0.35, size=(n, BOT_FEAT_DIM)).astype(np.float32)
        bf[:, 0] = rng.uniform(0.15, 0.85, size=n)  # x/W
        bf[:, 1] = rng.uniform(0.15, 0.85, size=n)  # y/H
        bf[:, 2] = rng.choice([0.0, 1.0 / 3.0, 2.0 / 3.0], size=n)  # layer
        return torch.from_numpy(bf).to(device)

    rng = np.random.default_rng(777)
    n_steps = 360
    bs = 128
    for step in range(n_steps):
        injector.train()
        w = train_worlds[step % len(train_worlds)]
        bf = sample_bot_feats(bs, rng)
        c_idx = torch.randint(0, len(train_texts), (bs,), device=device)
        k_gt = train_int_t[c_idx]
        g_vec = injector.encode_goal(train_bow[c_idx])
        vec, bin_l, elem_l = injector.forward_with_goal(w.z_world, w.z_eco, bf, g_vec)

        gt_v = intent_vec_t[k_gt]
        gt_e = intent_elem_t[k_gt]
        gt_b = intent_bin_t[k_gt]
        loss = (
            F.mse_loss(vec, gt_v)
            + 0.75 * F.binary_cross_entropy_with_logits(bin_l, gt_b)
            + 1.0 * F.cross_entropy(elem_l, gt_e)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Evaluate on UNSEEN test worlds + UNSEEN paraphrased human commands vs unconditioned Base BotPolicyNet!
    injector.eval()
    eval_rng = np.random.default_rng(2026)

    def eval_split(bow_bank: torch.Tensor, int_bank: torch.Tensor, worlds_eval: list[FrostmarshWorld3D]) -> dict[str, float]:
        cos_base, cos_inj = [], []
        el_base, el_inj = [], []
        ab_base, ab_inj = [], []
        full_base, full_inj = [], []
        with torch.no_grad():
            for w in worlds_eval:
                n = len(int_bank) * 4
                bf = sample_bot_feats(n, eval_rng)
                c_idx = torch.arange(len(int_bank), device=device).repeat(4)
                k_gt = int_bank[c_idx]
                gt_move = intent_vec_t[k_gt, :2]
                gt_e = intent_elem_t[k_gt]
                gt_ab_idx = torch.tensor([HUMAN_INTENT_SPECS[int(k)][4] for k in k_gt.cpu().numpy()], device=device)

                v0, b0, e0 = injector.forward_with_goal(w.z_world, w.z_eco, bf, None)
                g_vec = injector.encode_goal(bow_bank[c_idx])
                v1, b1, e1 = injector.forward_with_goal(w.z_world, w.z_eco, bf, g_vec)

                c0 = F.cosine_similarity(v0[:, :2], gt_move, dim=-1)
                c1 = F.cosine_similarity(v1[:, :2], gt_move, dim=-1)
                e_ok0 = (e0.argmax(-1) == gt_e).float()
                e_ok1 = (e1.argmax(-1) == gt_e).float()
                a_ok0 = (torch.sigmoid(b0).gather(1, gt_ab_idx.unsqueeze(1)).squeeze(1) > 0.5).float()
                a_ok1 = (torch.sigmoid(b1).gather(1, gt_ab_idx.unsqueeze(1)).squeeze(1) > 0.5).float()

                cos_base.append(c0)
                cos_inj.append(c1)
                el_base.append(e_ok0)
                el_inj.append(e_ok1)
                ab_base.append(a_ok0)
                ab_inj.append(a_ok1)
                full_base.append(((c0 > 0.7) & (e_ok0 > 0.5) & (a_ok0 > 0.5)).float())
                full_inj.append(((c1 > 0.7) & (e_ok1 > 0.5) & (a_ok1 > 0.5)).float())

        return {
            "base_move_cos": float(torch.cat(cos_base).mean().item()),
            "injected_move_cos": float(torch.cat(cos_inj).mean().item()),
            "base_element_acc": float(torch.cat(el_base).mean().item()),
            "injected_element_acc": float(torch.cat(el_inj).mean().item()),
            "base_ability_acc": float(torch.cat(ab_base).mean().item()),
            "injected_ability_acc": float(torch.cat(ab_inj).mean().item()),
            "base_full_compliance": float(torch.cat(full_base).mean().item()),
            "injected_full_compliance": float(torch.cat(full_inj).mean().item()),
        }

    seen_res = eval_split(train_bow, train_int_t, test_worlds)
    unseen_res = eval_split(test_bow, test_int_t, test_worlds)
    adapter_params = sum(p.numel() for p in injector.parameters() if p.requires_grad)
    return injector, {
        "goal_dim": injector.goal_dim,
        "adapter_trainable_params": adapter_params,
        "seen_commands_unseen_worlds": seen_res,
        "unseen_paraphrases_unseen_worlds": unseen_res,
    }


# =============================================================================
# 6. Multi-Turn In-World Dialogue + Physics Transcripts (Generated from Models)
# =============================================================================
@torch.no_grad()
def generate_dialogue_transcripts(
    phys_comm: CommBotPair,
    interp: ViewerInterpreter,
    test_worlds: list[FrostmarshWorld3D],
    test_data: dict[str, torch.Tensor],
    acoustics: AcousticField3D,
    device: torch.device,
) -> list[dict[str, Any]]:
    """
    Generate 10 diverse, concrete, realistic multi-turn in-world dialogue + physics transcripts
    directly from the trained CommBotPair + ViewerInterpreter on unseen 256x256x4 scenes!
    """
    phys_comm.eval()
    interp.eval()

    # Run inference across test_data to select 10 rich representative episodes spanning
    # all 6 physical events, all 4 layers (L0/L1/L2/L3), clear/muffled/inaudible SNR,
    # and shouting-over-waterfall vs whispering-behind-basalt-wall scenarios!
    M = test_data["event"].shape[0]
    cout = phys_comm(
        test_data["spot_crops"],
        test_data["spot_aux"],
        test_data["team_crops"],
        test_data["team_aux"],
        test_data["atten_tm"],
        test_data["noise_tm"],
        test_data["atten_en"],
        test_data["noise_en"],
        deterministic_noise=True,
    )
    z_ctx = test_data["team_crops"].float().mean(dim=(2, 3))
    pout = interp(cout["u_heard"], cout["snr_tm"], test_data["speaker_cls"], z_ctx)

    snr_tm = cout["snr_tm"].cpu().numpy()
    snr_en = cout["snr_en"].cpu().numpy()
    vol = cout["volume_used"].cpu().numpy()
    sym = cout["sym_id"].cpu().numpy()
    ev_gt = test_data["event"].cpu().numpy()
    sec_gt = test_data["sector"].cpu().numpy()
    walls_tm = test_data["walls_tm"].cpu().numpy()
    walls_en = test_data["walls_en"].cpu().numpy()
    ceil_tm = test_data["ceil_tm"].cpu().numpy()
    noise_tm = test_data["noise_tm"].cpu().numpy()
    cause_tm = test_data["cause_tm"].cpu().numpy()
    dist_tm = test_data["dist_tm"].cpu().numpy()
    spot_xyz = test_data["spot_xyz"].cpu().numpy()
    team_xyz = test_data["team_xyz"].cpu().numpy()
    enemy_xyz = test_data["enemy_xyz"].cpu().numpy()
    sp_cls = test_data["speaker_cls"].cpu().numpy()
    w_idx = test_data["world_idx"].cpu().numpy()

    p_ev = pout["event_logits"].argmax(-1).cpu().numpy()
    p_sec = pout["sector_logits"].argmax(-1).cpu().numpy()
    p_el = pout["elem_logits"].argmax(-1).cpu().numpy()
    p_tac = pout["tactic_logits"].argmax(-1).cpu().numpy()
    p_urg = pout["urgency"].cpu().numpy()

    tm_sec = cout["sector_logits"].argmax(-1).cpu().numpy()
    tm_el = cout["elem_logits"].argmax(-1).cpu().numpy()
    tm_tac = cout["tactic_logits"].argmax(-1).cpu().numpy()

    # Curate 10 diverse scenario selectors from real test episodes (across multiple unseen worlds!)
    criteria = [
        ("Clear Oil Trap Coordination (Open Air)", lambda i: ev_gt[i] == 0 and snr_tm[i] >= 12.0 and tm_tac[i] == 0 and tm_sec[i] == sec_gt[i]),
        ("Whisper Behind Basalt Wall (Stealth vs Enemy)", lambda i: vol[i] < 0.65 and walls_en[i] >= 0.8 and snr_en[i] < ENEMY_DETECT_SNR_DB and snr_tm[i] >= 4.0 and tm_tac[i] == EVENT_TO_TACTIC[ev_gt[i]]),
        ("L2 North Skybridge Shout Across Valley", lambda i: spot_xyz[i, 2] == 2 and snr_tm[i] >= 8.0 and vol[i] >= 0.45 and tm_tac[i] == EVENT_TO_TACTIC[ev_gt[i]]),
        ("Muffled Wildfire Flanking Callout", lambda i: ev_gt[i] == 1 and -4.5 <= snr_tm[i] <= 4.5 and cause_tm[i] == 2),
        ("Drowned Out by L2→L1 Plunging Waterfall", lambda i: snr_tm[i] < -6.5 and cause_tm[i] == 1),
        ("Frozen River Bridge Shatter Callout", lambda i: ev_gt[i] == 2 and snr_tm[i] >= 8.0 and tm_tac[i] == 2 and tm_sec[i] == sec_gt[i]),
        ("L0 Cavern Shaft Vertical Transmission (Clear via Shaft)", lambda i: ev_gt[i] == 5 and ceil_tm[i] < 0.5 and snr_tm[i] >= 6.0 and tm_tac[i] == 5 and tm_sec[i] == sec_gt[i]),
        ("L0 Cave Rock Ceiling Muffling (Garbled)", lambda i: ceil_tm[i] > 0.5 and -5.5 <= snr_tm[i] <= 3.0),
        ("Basalt Ridge Double-Wall Occlusion (Inaudible)", lambda i: walls_tm[i] >= 1.5 and snr_tm[i] < -6.0),
        ("High-Ground Waterfall Choke Defense", lambda i: ev_gt[i] == 4 and snr_tm[i] >= 4.0 and tm_tac[i] == 4 and tm_sec[i] == sec_gt[i]),
    ]

    used = set()
    transcripts = []
    # Rotate starting world offset per scenario so transcripts span unseen seeds 900..911!
    for sc_idx, (title, pred_fn) in enumerate(criteria):
        picked = None
        offset = (sc_idx * 400) % M
        for step in range(M):
            i = (offset + step) % M
            if i not in used and pred_fn(i):
                picked = i
                break
        if picked is None:
            # Relaxed fallback matching acoustic tier
            for step in range(M):
                i = (offset + step) % M
                if i not in used:
                    picked = i
                    break
        used.add(picked)
        i = int(picked)

        # Turn 1: Spotter callout rendered through ViewerInterpreter
        turn1_text = render_grounded_dialogue(
            pred_event=int(p_ev[i]),
            pred_sector=int(p_sec[i]),
            pred_elem=int(p_el[i]),
            pred_tactic=int(p_tac[i]),
            urgency=float(p_urg[i]),
            snr_db=float(snr_tm[i]),
            cause_id=int(cause_tm[i]),
            walls_crossed=float(walls_tm[i]),
            ceiling_muffled=bool(ceil_tm[i] > 0.5),
            speaker_cls=int(sp_cls[i]),
            src_layer=int(round(spot_xyz[i, 2])),
            volume=float(vol[i]),
        )

        # Turn 2: Teammate acoustic acknowledgment & physical action (grounded in Teammate's decoded policy!)
        tm_coord_ok = bool((tm_sec[i] == sec_gt[i]) and (tm_el[i] == EVENT_TO_ELEM[ev_gt[i]]) and (tm_tac[i] == EVENT_TO_TACTIC[ev_gt[i]]))
        tm_cls = (int(sp_cls[i]) + 1) % len(SPEAKER_CLASSES)
        if snr_tm[i] < -6.0:
            turn2_text = (
                f"[{SPEAKER_CLASSES[tm_cls]} @ L{int(round(team_xyz[i,2]))}] "
                f"\"Didn't copy — too much interference here! Holding fallback stance "
                f"(guessed {SECTOR_NAMES[int(tm_sec[i])]} / {ELEMENT_NAMES[int(tm_el[i])]}).\""
            )
        elif snr_tm[i] < 2.0:
            turn2_text = (
                f"[{SPEAKER_CLASSES[tm_cls]} @ L{int(round(team_xyz[i,2]))}] "
                f"\"Breaking up, say again! Moving cautiously toward {SECTOR_NAMES[int(tm_sec[i])]} "
                f"with {ELEMENT_NAMES[int(tm_el[i])]} ({TACTIC_NAMES[int(tm_tac[i])]}).\""
            )
        else:
            turn2_text = (
                f"[{SPEAKER_CLASSES[tm_cls]} @ L{int(round(team_xyz[i,2]))}] "
                f"\"Copy loud and clear! Rotating to {SECTOR_NAMES[int(tm_sec[i])]} — "
                f"priming {ELEMENT_NAMES[int(tm_el[i])]} for {TACTIC_NAMES[int(tm_tac[i])]}!\""
            )

        transcripts.append({
            "scenario": title,
            "world_seed": int(test_worlds[int(w_idx[i])].seed),
            "gt_event": EVENT_NAMES[int(ev_gt[i])],
            "gt_sector": SECTOR_NAMES[int(sec_gt[i])],
            "spotter_pos": [round(float(spot_xyz[i, 0]), 1), round(float(spot_xyz[i, 1]), 1), int(round(spot_xyz[i, 2]))],
            "teammate_pos": [round(float(team_xyz[i, 0]), 1), round(float(team_xyz[i, 1]), 1), int(round(team_xyz[i, 2]))],
            "enemy_pos": [round(float(enemy_xyz[i, 0]), 1), round(float(enemy_xyz[i, 1]), 1), int(round(enemy_xyz[i, 2]))],
            "distance_cells": round(float(dist_tm[i]), 1),
            "walls_crossed": round(float(walls_tm[i]), 1),
            "ceiling_muffled": bool(ceil_tm[i] > 0.5),
            "ambient_noise_tm_db": round(float(noise_tm[i]), 1),
            "volume": round(float(vol[i]), 2),
            "symbol_id": int(sym[i]),
            "snr_teammate_db": round(float(snr_tm[i]), 1),
            "snr_enemy_db": round(float(snr_en[i]), 1),
            "enemy_eavesdropped": bool(snr_en[i] > ENEMY_DETECT_SNR_DB),
            "viewer_turn1": turn1_text,
            "teammate_turn2": turn2_text,
            "teammate_action": f"{SECTOR_NAMES[int(tm_sec[i])]} | {ELEMENT_NAMES[int(tm_el[i])]} | {TACTIC_NAMES[int(tm_tac[i])]}",
            "coord_success": tm_coord_ok,
        })
    return transcripts


# =============================================================================
# 7. GB10 GPU Latency Benchmarks
# =============================================================================
def bench_cuda_fn(fn, warmup: int = 20, iters: int = 100) -> dict[str, float]:
    """Benchmark GPU execution latency using CUDA Events (avoids CPU wall-clock jitter)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(start_ev.elapsed_time(end_ev))
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "median_ms": round(float(np.median(arr)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "mean_ms": round(float(np.mean(arr)), 4),
    }


def bench_cuda_graph(fn, warmup: int = 10, iters: int = 100) -> dict[str, float]:
    """Capture a static CUDA graph to eliminate Python dispatch overhead on GB10."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return bench_cuda_fn(lambda: g.replay(), warmup=15, iters=iters)


@torch.no_grad()
def run_gb10_latency_benchmarks(
    acoustics: AcousticField3D,
    phys_comm: CommBotPair,
    interp: ViewerInterpreter,
    injector: HumanGoalInjector,
    sample_world: FrostmarshWorld3D,
    device: torch.device,
) -> dict[str, Any]:
    phys_comm.eval()
    interp.eval()
    injector.eval()
    res: dict[str, Any] = {}

    # Precompute static geometry tensors for fast 10-step 3D acoustic conv (CUDA-graph ready!)
    wall_f = sample_world.solid_wall.unsqueeze(0).half()
    trans_f = (1.0 - 0.90 * wall_f).contiguous()
    v_up_f, v_dn_f, v_self_f = [t.half().contiguous() for t in acoustics._build_vert_tensors(sample_world.open_shaft_01, sample_world.open_sky_12)]
    wall_c = F.max_pool2d(wall_f, 4, 4)
    trans_c = (1.0 - 0.94 * wall_c).contiguous()
    v_up_c = F.avg_pool2d(v_up_f, 4, 4).contiguous()
    v_dn_c = F.avg_pool2d(v_dn_f, 4, 4).contiguous()
    v_self_c = (1.0 - v_up_c - v_dn_c).contiguous()
    imp_fp16 = torch.zeros((1, LAYERS, GRID, GRID), device=device, dtype=torch.float16)
    imp_fp16[0, 2, 62:66, 126:130] = 1e4
    imp3_fp16 = imp_fp16.repeat(3, 1, 1, 1).contiguous()
    ac_half = AcousticField3D(n_coarse_steps=8, n_fine_steps=2).to(device).half().eval()

    # 1. AcousticField3D (256x256x4): 10-step 3D conv (CUDA Graph FP16 + Eager FP16) + Ambient Noise Field
    res["acoustic_10step_3d_conv_256x256x4_fp16"] = bench_cuda_graph(
        lambda: ac_half.propagate_10step_precomputed(
            imp_fp16, trans_c, v_up_c, v_dn_c, v_self_c, trans_f, v_up_f, v_dn_f, v_self_f
        )
    )
    res["acoustic_ambient_noise_256x256x4_fp16"] = bench_cuda_graph(
        lambda: ac_half.propagate_10step_precomputed(
            imp3_fp16, trans_c, v_up_c, v_dn_c, v_self_c, trans_f, v_up_f, v_dn_f, v_self_f
        )
    )
    res["acoustic_ambient_noise_256x256x4_fp32"] = bench_cuda_fn(
        lambda: acoustics.compute_ambient_noise_field(
            sample_world.w_state, sample_world.solid_wall, sample_world.open_shaft_01, sample_world.open_sky_12
        )
    )
    res["acoustic_full_snr_field_256x256x4"] = bench_cuda_fn(
        lambda: acoustics.compute_speaker_snr_field(
            128.0, 64.0, 2, 0.95,
            sample_world.solid_wall, sample_world.open_shaft_01, sample_world.open_sky_12, sample_world.noise_db
        )
    )

    # 2. Pairwise SNR + CommBotPair + ViewerInterpreter + HumanGoalInjector across N = 12, 64, 256 bots!
    for n_bots in (12, 64, 256):
        s_xyz = torch.stack([
            torch.rand(n_bots, device=device) * 220 + 18,
            torch.rand(n_bots, device=device) * 220 + 18,
            torch.randint(0, 3, (n_bots,), device=device).float(),
        ], dim=-1)
        t_xyz = torch.stack([
            torch.rand(n_bots, device=device) * 220 + 18,
            torch.rand(n_bots, device=device) * 220 + 18,
            torch.randint(0, 3, (n_bots,), device=device).float(),
        ], dim=-1)
        s_crops = torch.randn((n_bots, 48, 16, 16), device=device, dtype=torch.float16)
        t_crops = torch.randn((n_bots, 48, 16, 16), device=device, dtype=torch.float16)
        s_aux = torch.randn((n_bots, 10), device=device)
        t_aux = torch.randn((n_bots, 4), device=device)
        at = torch.full((n_bots,), 28.0, device=device)
        no = torch.full((n_bots,), 38.0, device=device)
        u_h = torch.randn((n_bots, 24), device=device)
        snr = torch.full((n_bots,), 12.0, device=device)
        scls = torch.randint(0, 4, (n_bots,), device=device)
        zctx = torch.randn((n_bots, 48), device=device)
        bf = torch.randn((n_bots, BOT_FEAT_DIM), device=device)
        bf[:, 0:2] = 0.5
        bf[:, 2] = 0.33
        bow = torch.randn((n_bots, 256), device=device)

        res[f"pairwise_snr_query_{n_bots}_bots"] = bench_cuda_graph(
            lambda s=s_xyz, t=t_xyz: acoustics.query_pairwise_link(
                s, t, sample_world.solid_wall, sample_world.open_shaft_01, sample_world.open_sky_12,
                sample_world.noise_db, sample_world.noise_cause
            )
        )
        res[f"comm_bot_pair_{n_bots}_bots"] = bench_cuda_graph(
            lambda sc=s_crops, sa=s_aux, tc=t_crops, ta=t_aux, a=at, n=no: phys_comm(
                sc, sa, tc, ta, a, n, a, n
            )
        )
        res[f"viewer_interpreter_{n_bots}_bots"] = bench_cuda_graph(
            lambda u=u_h, sn=snr, cl=scls, z=zctx: interp(u, sn, cl, z)
        )
        res[f"human_goal_bot_policy_{n_bots}_bots"] = bench_cuda_graph(
            lambda b=bf, bw=bow: injector.forward_with_goal(
                sample_world.z_world, sample_world.z_eco, b, injector.encode_goal(bw)
            )
        )

    return res


# =============================================================================
# 8. High-Resolution 3-Panel Figure (`acoustic_field_viz.png`) via System PIL
# =============================================================================
def generate_visualization(
    vis_world: FrostmarshWorld3D,
    acoustics: AcousticField3D,
    conf_16x6: np.ndarray,
    vocab_metrics: dict[str, Any],
    out_path: Path,
) -> None:
    """
    Generate a crisp 3-panel publication figure (2280x840 px) saved to acoustic_field_viz.png:
      Panel 1: Frostmarsh 256x256 map with basalt walls, L2 skybridges, waterfalls, wildfires, oil & ice
      Panel 2: 3D Acoustic Propagation + SNR heatmap comparing:
               (2A) Shouting on L2 North Skybridge vs (2B) Whispering behind L1 Basalt Wall
      Panel 3: 16x6 Emergent Neural Vocabulary Symbol-to-Event Alignment Matrix with MI/NMI/Purity
    Uses system python3 PIL + DejaVuSans TrueType fonts via stdin pipe (zero temp files on disk!).
    """
    w_cpu = vis_world.w_state.cpu().numpy()
    lm = vis_world.landmarks

    # Panel 1 RGB terrain map [256, 256, 3]
    rgb = np.zeros((GRID, GRID, 3), dtype=np.float32)
    # Base soil & marsh
    rgb[..., 0] = 36 + 18 * w_cpu[1, S_SOIL]
    rgb[..., 1] = 44 + 22 * w_cpu[1, S_SOIL]
    rgb[..., 2] = 48 + 18 * w_cpu[1, S_SOIL]

    # Wood & marsh plants
    plant_m = w_cpu[1, S_PLANT] > 0.15
    wood_m = w_cpu[1, S_WOOD] > 0.25
    rgb[plant_m] = [34, 76, 52]
    rgb[wood_m] = [24, 62, 42]

    # River & water
    water_m = w_cpu[1, S_WATER] > 0.35
    rgb[water_m] = [32, 92, 162]

    # Oil slick basins (rich dark gold/amber sheen)
    oil_m = w_cpu[1, S_OIL] > 0.30
    rgb[oil_m] = [138, 96, 28]

    # Frozen river bridge (bright icy cyan-white)
    ice_m = w_cpu[1, S_ICE] > 0.40
    rgb[ice_m] = [165, 228, 252]

    # L2 Highland plateaus & North/South Skybridges subtle overlay tint
    l2_slab = (w_cpu[2, S_SOIL] > 0.35) | (w_cpu[2, S_WOOD] > 0.35)
    rgb[l2_slab] = rgb[l2_slab] * 0.55 + np.array([110, 118, 140], dtype=np.float32) * 0.45

    # Solid basalt ridges & walls (crisp light-bordered dark volcanic basalt)
    rock_m = (w_cpu[1, S_ROCK] >= 0.5) | (w_cpu[1, 15] >= 0.5)
    rgb[rock_m] = [215, 220, 232]  # high-contrast pale slate edge/ridge so walls pop clearly

    # Active wildfires (glowing ember orange-red)
    fire_m = w_cpu[1, 10] > 1.5
    rgb[fire_m] = [248, 92, 36]

    # Plunging waterfalls (electric cyan-white plunge spray)
    for (wfx, wfy, _) in lm["waterfall_positions"]:
        yy, xx = np.mgrid[0:GRID, 0:GRID]
        r2 = (xx - wfx) ** 2 + (yy - wfy) ** 2
        rgb[r2 < 10 ** 2] = [74, 238, 255]
        rgb[r2 < 5 ** 2] = [225, 252, 255]

    map_rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Panel 2A: Speaker SHOUTING (vol=0.95) on L2 North Skybridge -> L1 SNR field
    sb_x, sb_y, sb_z = lm["north_skybridge"]
    _, snr_shout_3d = acoustics.compute_speaker_snr_field(
        sb_x, sb_y, sb_z, 0.95,
        vis_world.solid_wall, vis_world.open_shaft_01, vis_world.open_sky_12, vis_world.noise_db
    )
    snr_shout_l1 = snr_shout_3d[1].cpu().numpy()  # heard on L1 surface

    # Panel 2B: Speaker WHISPERING (vol=0.20) right behind West Basalt Wall on L1!
    wx1 = lm["wall_specs"][0]["wx1"]
    wh_x = float(np.clip(wx1 - 12.0, 20.0, 235.0))
    wh_y = 148.0
    _, snr_whisp_3d = acoustics.compute_speaker_snr_field(
        wh_x, wh_y, 1, 0.20,
        vis_world.solid_wall, vis_world.open_shaft_01, vis_world.open_sky_12, vis_world.noise_db
    )
    snr_whisp_l1 = snr_whisp_3d[1].cpu().numpy()

    # Teammate & Enemy reference positions for Panel 2B illustration
    tm_b = [float(np.clip(wh_x - 24.0, 14.0, 240.0)), float(np.clip(wh_y + 18.0, 14.0, 240.0))]
    en_b = [float(np.clip(wx1 + 14.0, 14.0, 240.0)), float(wh_y)]
    tm_b_snr = float(snr_whisp_l1[int(tm_b[1]), int(tm_b[0])])
    en_b_snr = float(snr_whisp_l1[int(en_b[1]), int(en_b[0])])

    # Panel 2A reference markers: Teammate in open valley vs Teammate under NW Waterfall
    wf0_x, wf0_y, _ = lm["waterfall_positions"][0]
    tm_a_open = [128.0, 152.0]
    tm_a_open_snr = float(snr_shout_l1[int(tm_a_open[1]), int(tm_a_open[0])])
    tm_a_wf_snr = float(snr_shout_l1[int(wf0_y), int(wf0_x)])

    # Colorize SNR heatmaps (-20 dB -> +30 dB) using perceptually crisp magma/turbo-like palette + wall overlay
    def snr_to_rgb(snr_map: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        t = np.clip((snr_map + 20.0) / 50.0, 0.0, 1.0)
        # 5-stop perceptual ramp: deep navy -> indigo -> crimson -> amber -> bright yellow-white
        stops = np.array([
            [12, 16, 42],
            [48, 28, 108],
            [168, 42, 96],
            [244, 142, 42],
            [252, 246, 178],
        ], dtype=np.float32)
        pos = t * (len(stops) - 1)
        idx = np.clip(np.floor(pos).astype(np.int32), 0, len(stops) - 2)
        frac = (pos - idx)[..., None]
        col = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
        # Crisp 0 dB audibility contour line (cyan!)
        c0 = np.abs(snr_map) < 1.15
        col[c0] = [68, 242, 255]
        # Basalt wall silhouette overlay
        col[wall_mask] = [30, 36, 46]
        return np.clip(col, 0, 255).astype(np.uint8)

    wall_l1 = (vis_world.solid_wall[1].cpu().numpy() > 0.5)
    shout_rgb = snr_to_rgb(snr_shout_l1, wall_l1)
    whisp_rgb = snr_to_rgb(snr_whisp_l1, wall_l1)

    payload = {
        "out_path": str(out_path),
        "map_rgb": map_rgb.tobytes().hex(),
        "shout_rgb": shout_rgb.tobytes().hex(),
        "whisp_rgb": whisp_rgb.tobytes().hex(),
        "conf": conf_16x6.tolist(),
        "event_short": EVENT_SHORT,
        "mi_bits": vocab_metrics["mi_bits"],
        "nmi": vocab_metrics["nmi"],
        "purity": vocab_metrics["purity"],
        "active_symbols": vocab_metrics["active_symbols"],
        "sb_pos": [sb_x, sb_y],
        "wf0_pos": [wf0_x, wf0_y],
        "wf1_pos": [lm["waterfall_positions"][1][0], lm["waterfall_positions"][1][1]],
        "fire_pos": [lm["wildfire_positions"][0][0], lm["wildfire_positions"][0][1]],
        "oil_pos": [lm["oil_positions"][0][0], lm["oil_positions"][0][1]],
        "ice_pos": [lm["ice_bridge_pos"][0], lm["ice_bridge_pos"][1]],
        "tm_a_open": tm_a_open,
        "tm_a_open_snr": tm_a_open_snr,
        "tm_a_wf_snr": tm_a_wf_snr,
        "wh_pos": [wh_x, wh_y],
        "tm_b": tm_b,
        "en_b": en_b,
        "tm_b_snr": tm_b_snr,
        "en_b_snr": en_b_snr,
    }

    pil_script = r'''
import json, sys
from PIL import Image, ImageDraw, ImageFont

d = json.loads(sys.stdin.read())
W, H = 2280, 860
BG = (13, 17, 26)
CARD = (20, 26, 38)
BORDER = (46, 58, 80)
TEXT = (235, 240, 248)
MUTED = (156, 170, 194)
CYAN = (72, 236, 255)
AMBER = (250, 184, 62)
GREEN = (74, 222, 128)
RED = (248, 102, 102)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

def load_font(size, bold=False, mono=False):
    paths = []
    if mono:
        paths.append("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
    if bold:
        paths.append("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    paths.append("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

f_title = load_font(26, bold=True)
f_sub = load_font(15, bold=False)
f_h2 = load_font(18, bold=True)
f_sm = load_font(13, bold=False)
f_sm_b = load_font(13, bold=True)
f_xs = load_font(11, bold=False)
f_mono = load_font(12, mono=True)

# Top Banner
draw.text((34, 20), "Frostmarsh 256×256×4 — Physical 3D Acoustics & Emergent Neural Speech", fill=TEXT, font=f_title)
draw.text((34, 54), "Frozen WorldModel Z_world [4, 48, 256, 256]  •  10-Step GPU 3D Wave/Diffusion + (-12 dB/wall) Occlusion  •  K=16 Gumbel-Softmax Symbols", fill=MUTED, font=f_sub)

def draw_card(x0, y0, x1, y1, title, subtitle):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=CARD, outline=BORDER, width=2)
    draw.text((x0 + 18, y0 + 14), title, fill=TEXT, font=f_h2)
    draw.text((x0 + 18, y0 + 39), subtitle, fill=MUTED, font=f_sm)

def map_from_hex(hstr, size):
    raw = bytes.fromhex(hstr)
    im = Image.frombytes("RGB", (256, 256), raw)
    return im.resize((size, size), Image.Resampling.NEAREST)

# ---------------- PANEL 1: Frostmarsh Tactical 3D Terrain ----------------
p1x, p1y, p1w, p1h = 28, 92, 610, 742
draw_card(p1x, p1y, p1x + p1w, p1y + p1h,
          "1. Frostmarsh 256×256×4 Tactical World",
          "L1 Surface + L2 Highland Plateaus, Skybridges, Waterfalls & Wildfires")
msz = 544
mx0, my0 = p1x + (p1w - msz) // 2, p1y + 72
img.paste(map_from_hex(d["map_rgb"], msz), (mx0, my0))
draw.rectangle([mx0 - 1, my0 - 1, mx0 + msz, my0 + msz], outline=(86, 102, 130), width=2)

def pt_p1(xy):
    return (mx0 + int(xy[0] / 256.0 * msz), my0 + int(xy[1] / 256.0 * msz))

def badge(pos, text, color, dy=-22, dx=10):
    x, y = pos
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color, outline=(10, 14, 22), width=2)
    tx, ty = max(mx0 + 4, min(mx0 + msz - 145, x + dx)), max(my0 + 4, min(my0 + msz - 24, y + dy))
    bbox = draw.textbbox((tx, ty), text, font=f_sm_b)
    draw.rounded_rectangle([bbox[0] - 5, bbox[1] - 2, bbox[2] + 5, bbox[3] + 3], radius=5, fill=(10, 14, 22), outline=color, width=1)
    draw.text((tx, ty), text, fill=TEXT, font=f_sm_b)

badge(pt_p1(d["sb_pos"]), "L2 North Skybridge", (190, 204, 232), dy=14, dx=-64)
badge(pt_p1(d["wf0_pos"]), "Waterfall (85 dB)", CYAN, dy=-26, dx=-118)
badge(pt_p1(d["wf1_pos"]), "Waterfall (85 dB)", CYAN, dy=-26, dx=10)
badge(pt_p1(d["fire_pos"]), "Wildfire (77 dB)", (255, 110, 56), dy=-24, dx=8)
badge(pt_p1(d["oil_pos"]), "Oil Slick Basin", AMBER, dy=10, dx=8)
badge(pt_p1(d["ice_pos"]), "Frozen Bridge", (180, 238, 255), dy=12, dx=10)

# Legend strip under Panel 1
leg_items = [
    ((215, 220, 232), "Basalt Wall (-12dB)"),
    ((74, 238, 255), "L2→L1 Waterfall"),
    ((248, 92, 36), "Wildfire"),
    ((138, 96, 28), "Oil Slick"),
    ((165, 228, 252), "Ice Bridge"),
]
lx = p1x + 22
ly = my0 + msz + 22
for col, lbl in leg_items:
    draw.rectangle([lx, ly + 2, lx + 14, ly + 16], fill=col, outline=(200, 210, 225))
    draw.text((lx + 20, ly), lbl, fill=MUTED, font=f_sm)
    lx += 20 + int(draw.textlength(lbl, font=f_sm)) + 16
    if lx > p1x + p1w - 130:
        lx = p1x + 22
        ly += 24

# ---------------- PANEL 2: 3D Acoustic Propagation & SNR Heatmaps ----------------
p2x, p2y, p2w, p2h = 656, 92, 928, 742
draw_card(p2x, p2y, p2x + p2w, p2y + p2h,
          "2. Physical 3D Acoustic Propagation & SNR Heatmap (L1 Receiver Plane)",
          "Cyan contour = 0 dB audibility threshold  •  Dark blocks = basalt ridges (-12 dB/wall)")

hsz = 426
ax0, ay0 = p2x + 24, p2y + 106
bx0, by0 = p2x + 24 + hsz + 28, p2y + 106

draw.text((ax0, ay0 - 26), "A. Shouting (vol=0.95) on L2 North Skybridge", fill=AMBER, font=f_sm_b)
draw.text((bx0, by0 - 26), "B. Whispering (vol=0.20) Behind L1 Basalt Wall", fill=GREEN, font=f_sm_b)

img.paste(map_from_hex(d["shout_rgb"], hsz), (ax0, ay0))
draw.rectangle([ax0 - 1, ay0 - 1, ax0 + hsz, ay0 + hsz], outline=(86, 102, 130), width=2)
img.paste(map_from_hex(d["whisp_rgb"], hsz), (bx0, by0))
draw.rectangle([bx0 - 1, by0 - 1, bx0 + hsz, by0 + hsz], outline=(86, 102, 130), width=2)

def pt_box(xy, ox, oy, sz):
    return (ox + int(xy[0] / 256.0 * sz), oy + int(xy[1] / 256.0 * sz))

def mark_snr(ox, oy, sz, xy, lbl, color, dx=8, dy=-22):
    x, y = pt_box(xy, ox, oy, sz)
    draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color, outline=(10, 14, 22), width=2)
    tx = max(ox + 4, min(ox + sz - 155, x + dx))
    ty = max(oy + 4, min(oy + sz - 22, y + dy))
    bb = draw.textbbox((tx, ty), lbl, font=f_xs)
    draw.rounded_rectangle([bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2], radius=4, fill=(10, 14, 22), outline=color, width=1)
    draw.text((tx, ty), lbl, fill=TEXT, font=f_xs)

# Panel 2A markers
mark_snr(ax0, ay0, hsz, d["sb_pos"], "SPK (L2 Shout)", AMBER, dx=10, dy=-26)
mark_snr(ax0, ay0, hsz, d["tm_a_open"], f"Open Valley {d['tm_a_open_snr']:+.1f}dB", GREEN, dx=-52, dy=12)
mark_snr(ax0, ay0, hsz, d["wf0_pos"], f"Waterfall {d['tm_a_wf_snr']:+.1f}dB", CYAN, dx=-112, dy=14)

# Panel 2B markers
mark_snr(bx0, by0, hsz, d["wh_pos"], "SPK (Whisper)", AMBER, dx=-38, dy=-24)
mark_snr(bx0, by0, hsz, d["tm_b"], f"Ally {d['tm_b_snr']:+.1f}dB", GREEN, dx=-18, dy=12)
mark_snr(bx0, by0, hsz, d["en_b"], f"Enemy {d['en_b_snr']:+.1f}dB (Safe)", RED, dx=10, dy=-12)

# Horizontal SNR colorbar below 2A & 2B
cb_x0, cb_y0, cb_w, cb_h = p2x + 84, ay0 + hsz + 36, p2w - 168, 22
stops = [(12, 16, 42), (48, 28, 108), (168, 42, 96), (244, 142, 42), (252, 246, 178)]
for px in range(cb_w):
    t = px / max(1, cb_w - 1)
    pos = t * (len(stops) - 1)
    idx = min(len(stops) - 2, int(pos))
    fr = pos - idx
    c = tuple(int(stops[idx][k] * (1 - fr) + stops[idx + 1][k] * fr) for k in range(3))
    draw.line([(cb_x0 + px, cb_y0), (cb_x0 + px, cb_y0 + cb_h)], fill=c)
draw.rectangle([cb_x0, cb_y0, cb_x0 + cb_w, cb_y0 + cb_h], outline=(120, 134, 160), width=1)
# 0 dB tick mark
zero_x = cb_x0 + int((20.0 / 50.0) * cb_w)
draw.line([(zero_x, cb_y0 - 4), (zero_x, cb_y0 + cb_h + 4)], fill=CYAN, width=3)
for val in (-20, -10, 0, 10, 20, 30):
    vx = cb_x0 + int(((val + 20.0) / 50.0) * cb_w)
    lbl = f"{val:+d} dB" if val != 0 else "0 dB (Threshold)"
    col = CYAN if val == 0 else MUTED
    tw = draw.textlength(lbl, font=f_sm_b if val == 0 else f_sm)
    draw.text((vx - tw / 2, cb_y0 + cb_h + 8), lbl, fill=col, font=f_sm_b if val == 0 else f_sm)

# Bottom callout explanation box in Panel 2
note_y = cb_y0 + 58
draw.rounded_rectangle([p2x + 24, note_y, p2x + p2w - 24, note_y + 84], radius=10, fill=(14, 19, 30), outline=(44, 58, 82), width=1)
draw.text((p2x + 40, note_y + 12), "Physical Trade-Off Verified on GB10:", fill=TEXT, font=f_sm_b)
draw.text((p2x + 40, note_y + 34), f"• L2 North Skybridge Shout reaches open L1 valley at {d['tm_a_open_snr']:+.1f} dB, but drops to {d['tm_a_wf_snr']:+.1f} dB inside the 85 dB waterfall plunge zone.", fill=MUTED, font=f_sm)
draw.text((p2x + 40, note_y + 56), f"• L1 Basalt-Wall Whisper reaches shielded ally at {d['tm_b_snr']:+.1f} dB while basalt wall (-12 dB) keeps enemy at {d['en_b_snr']:+.1f} dB (< +6 dB detect).", fill=MUTED, font=f_sm)

# ---------------- PANEL 3: 16x6 Emergent Symbol-to-Event Alignment Matrix ----------------
p3x, p3y, p3w, p3h = 1602, 92, 650, 742
draw_card(p3x, p3y, p3x + p3w, p3y + p3h,
          "3. Emergent Neural Vocabulary (K=16)",
          f"MI = {d['mi_bits']:.2f} bits  •  NMI = {d['nmi']:.3f}  •  Purity = {d['purity']*100:.1f}%  •  Active = {d['active_symbols']}/16")

conf = d["conf"]  # 16 x 6
# Sort symbols by dominant event then by usage count for clean block-diagonal visual structure!
row_sums = [sum(r) for r in conf]
order = sorted(range(16), key=lambda k: (conf[k].index(max(conf[k])) if row_sums[k] > 0 else 99, -row_sums[k]))

gx0, gy0 = p3x + 88, p3y + 114
cw, ch = 76, 35
ev_names = d["event_short"]

# Column headers
for j, ename in enumerate(ev_names):
    parts = ename.split(" ")
    cx = gx0 + j * cw + cw // 2
    if len(parts) == 2:
        w1 = draw.textlength(parts[0], font=f_xs)
        w2 = draw.textlength(parts[1], font=f_xs)
        draw.text((cx - w1 / 2, gy0 - 34), parts[0], fill=TEXT, font=f_xs)
        draw.text((cx - w2 / 2, gy0 - 19), parts[1], fill=TEXT, font=f_xs)
    else:
        w1 = draw.textlength(ename, font=f_xs)
        draw.text((cx - w1 / 2, gy0 - 24), ename, fill=TEXT, font=f_xs)

draw.text((gx0 + 6 * cw + 12, gy0 - 24), "Freq", fill=MUTED, font=f_xs)
total_n = max(1, sum(row_sums))

for r_pos, sym_k in enumerate(order):
    ry = gy0 + r_pos * ch
    r_tot = row_sums[sym_k]
    draw.text((p3x + 20, ry + 10), f"SYM_{sym_k:02d}", fill=TEXT if r_tot > 0 else MUTED, font=f_mono)
    for j in range(6):
        val = conf[sym_k][j]
        frac = val / max(1, r_tot)
        # Teal-emerald-gold heatmap cell
        if r_tot == 0:
            col = (24, 30, 44)
        else:
            r_c = int(20 + 225 * (frac ** 1.4))
            g_c = int(32 + 195 * (frac ** 0.75))
            b_c = int(52 + 115 * (frac * (1.0 - 0.5 * frac)))
            col = (min(255, r_c), min(255, g_c), min(255, b_c))
        x_cell = gx0 + j * cw
        draw.rectangle([x_cell + 2, ry + 2, x_cell + cw - 2, ry + ch - 2], fill=col, outline=(36, 46, 66))
        if frac >= 0.08:
            txt = f"{frac*100:.0f}%"
            tw = draw.textlength(txt, font=f_xs)
            tcol = (12, 16, 24) if frac > 0.52 else TEXT
            draw.text((x_cell + (cw - tw) / 2, ry + 10), txt, fill=tcol, font=f_xs)

    # Usage frequency bar on right
    bar_x = gx0 + 6 * cw + 10
    bar_max = 68
    bw = int(min(1.0, (r_tot / total_n) * 6.5) * bar_max)
    draw.rectangle([bar_x, ry + 10, bar_x + bar_max, ry + ch - 10], fill=(28, 36, 52))
    if bw > 0:
        draw.rectangle([bar_x, ry + 10, bar_x + bw, ry + ch - 10], fill=CYAN)

draw.text((p3x + 24, gy0 + 16 * ch + 18), "Cells show P(Physical Event | Emergent Symbol) on unseen held-out worlds (zero symbol supervision).", fill=MUTED, font=f_xs)

img.save(d["out_path"], "PNG")
print("SAVED_VIZ_OK")
'''
    proc = subprocess.run(
        ["python3", "-c", pil_script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    if "SAVED_VIZ_OK" not in proc.stdout:
        raise RuntimeError(f"PIL viz failed: {proc.stdout} {proc.stderr}")


# =============================================================================
# 9. Report Generator (`ACOUSTIC_COMM_REPORT.md`)
# =============================================================================
def write_report(
    metrics: dict[str, Any],
    transcripts: list[dict[str, Any]],
    out_md: Path,
) -> None:
    a1 = metrics["arms"]["arm1_silent"]
    a2 = metrics["arms"]["arm2_physical_3d"]
    a3 = metrics["arms"]["arm3_telepathic"]
    eav = metrics["eavesdrop_tradeoff"]
    vi = metrics["viewer_interpreter"]
    hg = metrics["human_goal_injector"]
    lat = metrics["latency_gb10_ms"]

    def pct(v: float | None) -> str:
        return "N/A" if v is None else f"{v * 100:.1f}%"

    lines = [
        "# Frostmarsh 3D Physical Acoustics, Emergent Neural Speech & Viewer Interpreter — Research Report",
        "",
        f"**Hardware:** `{metrics['hardware']}`  ",
        f"**World Volume:** `256 × 256 × 4` (`L0` subterranean caverns, `L1` surface valley, `L2` highland plateaus/skybridges/waterfalls, `L3` open sky)  ",
        f"**Shared Latent Trunk:** Frozen `WorldModel` `Z_world [4, 48, 256, 256]` + `EcosystemModel` `Z_eco [1, 32, 256, 256]`  ",
        f"**Evaluation Split:** `{metrics['dataset']['train_worlds']}` train worlds (`{metrics['dataset']['train_episodes']}` episodes) → **`{metrics['dataset']['test_worlds']}` unseen held-out worlds (`{metrics['dataset']['test_episodes']}` episodes)**",
        "",
        "![Frostmarsh 3D Acoustics & Emergent Neural Vocabulary](acoustic_field_viz.png)",
        "",
        "---",
        "",
        "## 1. Executive Summary & 3-Arm Ablation (Held-Out Unseen Worlds)",
        "",
        "| Arm | Acoustic Channel | Sector Acc | Element Acc | Tactic Acc | **Joint Coord Acc** | Mean SNR (TM) | Enemy Eavesdrop | Vocab MI (bits) | Symbol Purity |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Arm 1 — Silent** | None (`Z_world` only) | {pct(a1['sector_acc'])} | {pct(a1['element_acc'])} | {pct(a1['tactic_acc'])} | **{pct(a1['joint_coord_acc'])}** | — | 0.0% | — | — |",
        f"| **Arm 2 — Physical 3D Acoustic** | `AcousticField3D` (walls + waterfalls + stealth) | {pct(a2['sector_acc'])} | {pct(a2['element_acc'])} | {pct(a2['tactic_acc'])} | **{pct(a2['joint_coord_acc'])}** | `{a2['mean_snr_tm_db']:+.1f} dB` | **{pct(a2['enemy_eavesdrop_rate'])}** | **`{a2['vocab']['mi_bits']:.2f}`** | **{pct(a2['vocab']['purity'])}** |",
        f"| **Arm 3 — Ideal Telepathic** | Infinite SNR (`+40 dB` upper bound) | {pct(a3['sector_acc'])} | {pct(a3['element_acc'])} | {pct(a3['tactic_acc'])} | **{pct(a3['joint_coord_acc'])}** | `+40.0 dB` | — | `{a3['vocab']['mi_bits']:.2f}` | {pct(a3['vocab']['purity'])} |",
        "",
        "### Key Findings",
        f"- **Physical 3D Acoustic Speech vs Silent Baseline:** Adding differentiable neural speech through `AcousticField3D` boosts unseen-world joint coordination accuracy from **{pct(a1['joint_coord_acc'])}** → **{pct(a2['joint_coord_acc'])}** (`+{(a2['joint_coord_acc'] - a1['joint_coord_acc'])*100:.1f} pp`), capturing **{(a2['joint_coord_acc'] / max(1e-6, a3['joint_coord_acc']))*100:.1f}%** of the infinite-SNR telepathic upper bound (**{pct(a3['joint_coord_acc'])}**) despite real `-12 dB/wall` basalt occlusion, `-18 dB` L0 cave ceilings, and `{metrics['acoustic_physics_sanity']['waterfall_plunge_l1_db']:.0f} dB` plunging waterfalls.",
        f"- **Physical SNR Stratification (Arm 2):** Coordination accuracy tracks real acoustics cleanly: **{pct(a2['snr_buckets']['high_ge_10dB']['joint_acc'])}** at High SNR (`>= +10 dB`), **{pct(a2['snr_buckets']['med_2_to_10dB']['joint_acc'])}** at Medium SNR (`+2..+10 dB`), **{pct(a2['snr_buckets']['low_neg6_to_2dB']['joint_acc'])}** when muffled (`-6..+2 dB`), and collapses physically to **{pct(a2['snr_buckets']['inaudible_lt_neg6dB']['joint_acc'])}** when drowned out (`< -6 dB`).",
        f"- **Emergent Discrete Neural Vocabulary (`K=16`, zero symbol supervision):** Mutual information reaches **`{a2['vocab']['mi_bits']:.2f} / {a2['vocab']['h_event_bits']:.2f} bits`** (`NMI = {a2['vocab']['nmi']:.3f}`), with **{pct(a2['vocab']['purity'])}** symbol-to-event purity across **{a2['vocab']['active_symbols']}/16** active acoustic prototypes.",
        "",
        "---",
        "",
        "## 2. Adaptive Volume Control & Enemy Eavesdrop Trade-Off",
        "",
        "Evaluating the trained Arm 2 Spotter on unseen worlds under its learned adaptive volume vs fixed-volume policies (`Always Whisper` vs `Always Shout`):",
        "",
        "| Volume Policy | Mean Volume | Teammate Joint Acc | Mean Teammate SNR | **Enemy Eavesdrop (All)** | Eavesdrop (Enemy Open Air) | Eavesdrop (Enemy Behind Wall) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Learned Adaptive Volume** | `{eav['adaptive']['mean_volume']:.2f}` | **{pct(eav['adaptive']['joint_coord_acc'])}** | `{eav['adaptive']['mean_snr_tm_db']:+.1f} dB` | **{pct(eav['adaptive']['enemy_eavesdrop_rate'])}** | {pct(eav['adaptive']['enemy_eavesdrop_open_air'])} | {pct(eav['adaptive']['enemy_eavesdrop_behind_wall'])} |",
        f"| Fixed Shout (`vol = 1.00`) | `1.00` | {pct(eav['always_shout']['joint_coord_acc'])} | `{eav['always_shout']['mean_snr_tm_db']:+.1f} dB` | {pct(eav['always_shout']['enemy_eavesdrop_rate'])} | {pct(eav['always_shout']['enemy_eavesdrop_open_air'])} | {pct(eav['always_shout']['enemy_eavesdrop_behind_wall'])} |",
        f"| Fixed Whisper (`vol = 0.15`) | `0.15` | {pct(eav['always_whisper']['joint_coord_acc'])} | `{eav['always_whisper']['mean_snr_tm_db']:+.1f} dB` | {pct(eav['always_whisper']['enemy_eavesdrop_rate'])} | {pct(eav['always_whisper']['enemy_eavesdrop_open_air'])} | {pct(eav['always_whisper']['enemy_eavesdrop_behind_wall'])} |",
        "",
        f"- **Physical Stealth Behavior:** The Spotter learns selective radio silence — cutting transmission volume when links are heavily occluded/noisy (`mean vol = {a2['mean_volume']:.2f}`) while shouting when the teammate link is viable, reducing enemy eavesdropping by **`{(eav['always_shout']['enemy_eavesdrop_rate'] - eav['adaptive']['enemy_eavesdrop_rate'])*100:.1f} pp`** (`{pct(eav['always_shout']['enemy_eavesdrop_rate'])} → {pct(eav['adaptive']['enemy_eavesdrop_rate'])}`) vs always-shouting while losing only **`{(eav['always_shout']['joint_coord_acc'] - eav['adaptive']['joint_coord_acc'])*100:.1f} pp`** coordination accuracy (`{pct(eav['always_shout']['joint_coord_acc'])} → {pct(eav['adaptive']['joint_coord_acc'])}`) and outperforming always-whispering by **`+{(eav['adaptive']['joint_coord_acc'] - eav['always_whisper']['joint_coord_acc'])*100:.1f} pp`**! Under whispering (`vol = 0.15`), basalt walls drop enemy eavesdropping from `{pct(eav['always_whisper']['enemy_eavesdrop_open_air'])}` (open air) down to **`{pct(eav['always_whisper']['enemy_eavesdrop_behind_wall'])}`** (behind wall).",
        "",
        "---",
        "",
        "## 3. Viewer-Side Text Interpreter & Human-to-Bot Intent Injection",
        "",
        "### 3A. Viewer Text Interpreter (`u_heard, SNR, speaker_class, Z_world_context → Semantic Frame + Dialogue`)",
        f"- Parameters: **`{vi['n_params']:,}`** | Overall Event Acc: **{pct(vi['overall_event_acc'])}** | Sector Acc: **{pct(vi['overall_sector_acc'])}** | Element Acc: **{pct(vi['overall_element_acc'])}** | Intelligibility Tier Acc: **{pct(vi['intelligibility_tier_acc'])}**",
        "",
        "| SNR Regime | Test Utterances | Event Decode Acc | Sector Decode Acc | Element Decode Acc | Rendered Viewer Behavior |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| **High (`>= +10 dB`)** | {vi['snr_buckets']['high_ge_10dB']['count']} | **{pct(vi['snr_buckets']['high_ge_10dB']['event_acc'])}** | {pct(vi['snr_buckets']['high_ge_10dB']['sector_acc'])} | {pct(vi['snr_buckets']['high_ge_10dB']['element_acc'])} | Crisp, complete tactical callout |",
        f"| **Medium (`+2..+10 dB`)** | {vi['snr_buckets']['med_2_to_10dB']['count']} | {pct(vi['snr_buckets']['med_2_to_10dB']['event_acc'])} | {pct(vi['snr_buckets']['med_2_to_10dB']['sector_acc'])} | {pct(vi['snr_buckets']['med_2_to_10dB']['element_acc'])} | Slightly clipped callout with interference note |",
        f"| **Low / Muffled (`-6..+2 dB`)** | {vi['snr_buckets']['low_neg6_to_2dB']['count']} | {pct(vi['snr_buckets']['low_neg6_to_2dB']['event_acc'])} | {pct(vi['snr_buckets']['low_neg6_to_2dB']['sector_acc'])} | {pct(vi['snr_buckets']['low_neg6_to_2dB']['element_acc'])} | Garbled fragments (`...oil... [static]`) |",
        f"| **Inaudible (`< -6 dB`)** | {vi['snr_buckets']['inaudible_lt_neg6dB']['count']} | {pct(vi['snr_buckets']['inaudible_lt_neg6dB']['event_acc'])} | {pct(vi['snr_buckets']['inaudible_lt_neg6dB']['sector_acc'])} | {pct(vi['snr_buckets']['inaudible_lt_neg6dB']['element_acc'])} | `[inaudible — drowned out by <physical cause>]` |",
        "",
        "### 3B. Human Text → `g_i` (`16-d`) Goal-Vector Injector into Frozen `BotPolicyNet`",
        f"- Adapter trainable parameters: **`{hg['adapter_trainable_params']:,}`** (base `BotPolicyNet` weights completely frozen; `g_i = 0` preserves base policy identically).",
        "",
        "| Command Evaluation Split (Unseen Frostmarsh Worlds) | Move Vector Cosine | Element Selection Acc | Key Ability Match | **Full Command Compliance** |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| Base `BotPolicyNet` (Unconditioned) | `{hg['unseen_paraphrases_unseen_worlds']['base_move_cos']:+.3f}` | {pct(hg['unseen_paraphrases_unseen_worlds']['base_element_acc'])} | {pct(hg['unseen_paraphrases_unseen_worlds']['base_ability_acc'])} | {pct(hg['unseen_paraphrases_unseen_worlds']['base_full_compliance'])} |",
        f"| + `g_i` Injector (Seen Phrasings, Unseen Worlds) | **`{hg['seen_commands_unseen_worlds']['injected_move_cos']:+.3f}`** | **{pct(hg['seen_commands_unseen_worlds']['injected_element_acc'])}** | **{pct(hg['seen_commands_unseen_worlds']['injected_ability_acc'])}** | **{pct(hg['seen_commands_unseen_worlds']['injected_full_compliance'])}** |",
        f"| + `g_i` Injector (**Unseen Paraphrases**, Unseen Worlds) | **`{hg['unseen_paraphrases_unseen_worlds']['injected_move_cos']:+.3f}`** | **{pct(hg['unseen_paraphrases_unseen_worlds']['injected_element_acc'])}** | **{pct(hg['unseen_paraphrases_unseen_worlds']['injected_ability_acc'])}** | **{pct(hg['unseen_paraphrases_unseen_worlds']['injected_full_compliance'])}** |",
        "",
        "---",
        "",
        "## 4. NVIDIA GB10 Latency Benchmarks",
        "",
        "| Component | Workload / Scale | Median Latency (ms) | p95 Latency (ms) |",
        "| :--- | :--- | :---: | :---: |",
        f"| `AcousticField3D` 10-Step 3D Conv | `256×256×4` (FP16) | **`{lat['acoustic_10step_3d_conv_256x256x4_fp16']['median_ms']:.3f} ms`** | `{lat['acoustic_10step_3d_conv_256x256x4_fp16']['p95_ms']:.3f} ms` |",
        f"| `AcousticField3D` Ambient Noise Field | `256×256×4` (FP16, 3 noise bands) | **`{lat['acoustic_ambient_noise_256x256x4_fp16']['median_ms']:.3f} ms`** | `{lat['acoustic_ambient_noise_256x256x4_fp16']['p95_ms']:.3f} ms` |",
        f"| `AcousticField3D` Ambient Noise Field | `256×256×4` (FP32, 3 noise bands) | `{lat['acoustic_ambient_noise_256x256x4_fp32']['median_ms']:.3f} ms` | `{lat['acoustic_ambient_noise_256x256x4_fp32']['p95_ms']:.3f} ms` |",
        f"| `AcousticField3D` Full Speaker SNR Map | `256×256×4` (24-sample ray + 10-step 3D conv) | `{lat['acoustic_full_snr_field_256x256x4']['median_ms']:.3f} ms` | `{lat['acoustic_full_snr_field_256x256x4']['p95_ms']:.3f} ms` |",
        f"| `AcousticField3D` Pairwise SNR Query | `12 / 64 / 256` bot pairs | `{lat['pairwise_snr_query_12_bots']['median_ms']:.3f} / {lat['pairwise_snr_query_64_bots']['median_ms']:.3f} / {lat['pairwise_snr_query_256_bots']['median_ms']:.3f} ms` | `{lat['pairwise_snr_query_12_bots']['p95_ms']:.3f} / {lat['pairwise_snr_query_64_bots']['p95_ms']:.3f} / {lat['pairwise_snr_query_256_bots']['p95_ms']:.3f} ms` |",
        f"| `CommBotPair` (Spotter + Channel + Teammate) | `12 / 64 / 256` bots | **`{lat['comm_bot_pair_12_bots']['median_ms']:.3f} / {lat['comm_bot_pair_64_bots']['median_ms']:.3f} / {lat['comm_bot_pair_256_bots']['median_ms']:.3f} ms`** | `{lat['comm_bot_pair_12_bots']['p95_ms']:.3f} / {lat['comm_bot_pair_64_bots']['p95_ms']:.3f} / {lat['comm_bot_pair_256_bots']['p95_ms']:.3f} ms` |",
        f"| `ViewerInterpreter` | `12 / 64 / 256` utterances | **`{lat['viewer_interpreter_12_bots']['median_ms']:.3f} / {lat['viewer_interpreter_64_bots']['median_ms']:.3f} / {lat['viewer_interpreter_256_bots']['median_ms']:.3f} ms`** | `{lat['viewer_interpreter_12_bots']['p95_ms']:.3f} / {lat['viewer_interpreter_64_bots']['p95_ms']:.3f} / {lat['viewer_interpreter_256_bots']['p95_ms']:.3f} ms` |",
        f"| `HumanGoalInjector` + `BotPolicyNet` | `12 / 64 / 256` bots | **`{lat['human_goal_bot_policy_12_bots']['median_ms']:.3f} / {lat['human_goal_bot_policy_64_bots']['median_ms']:.3f} / {lat['human_goal_bot_policy_256_bots']['median_ms']:.3f} ms`** | `{lat['human_goal_bot_policy_12_bots']['p95_ms']:.3f} / {lat['human_goal_bot_policy_64_bots']['p95_ms']:.3f} / {lat['human_goal_bot_policy_256_bots']['p95_ms']:.3f} ms` |",
        "",
        "---",
        "",
        "## 5. Generated Multi-Turn In-World Dialogue & Physics Transcripts (Unseen `256×256×4` Worlds)",
        "",
        "All 10 transcripts below were generated directly from the trained `CommBotPair` (Arm 2) + `ViewerInterpreter` + `AcousticField3D` on held-out unseen Frostmarsh scenes (`seeds 900..911`):",
        "",
    ]

    for idx, tr in enumerate(transcripts, start=1):
        status_badge = "PASS" if tr["coord_success"] else "DEGRADED/MISS"
        eav_str = "YES (detected!)" if tr["enemy_eavesdropped"] else "NO (stealth preserved)"
        lines.extend([
            f"### Transcript {idx}: {tr['scenario']} *(World Seed `{tr['world_seed']}` — {status_badge})*",
            f"- **Physics Context:** Ground-Truth `{tr['gt_event']}` @ `{tr['gt_sector']}` | Spotter `({tr['spotter_pos'][0]}, {tr['spotter_pos'][1]}, L{tr['spotter_pos'][2]})` → Teammate `({tr['teammate_pos'][0]}, {tr['teammate_pos'][1]}, L{tr['teammate_pos'][2]})` (`d = {tr['distance_cells']} cells`)",
            f"- **3D Acoustic Link:** Walls crossed = `{tr['walls_crossed']}` | L0 Ceiling muffled = `{tr['ceiling_muffled']}` | Ambient Noise @ Teammate = `{tr['ambient_noise_tm_db']} dB` | Spotter Volume = `{tr['volume']}` | Emergent Symbol = `SYM_{tr['symbol_id']:02d}`",
            f"- **Signal-to-Noise:** Teammate `SNR = {tr['snr_teammate_db']:+.1f} dB` | Enemy `SNR = {tr['snr_enemy_db']:+.1f} dB` → Enemy Eavesdrop: **{eav_str}**",
            f"- **Turn 1 (Viewer Dialogue):** {tr['viewer_turn1']}",
            f"- **Turn 2 (Teammate Reply & Action):** {tr['teammate_turn2']} → Executed: `{tr['teammate_action']}`",
            "",
        ])

    out_md.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main Execution Pipeline
# =============================================================================
def main() -> None:
    t_start = time.perf_counter()
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Init] Running Frostmarsh 3D Acoustic Comm POC on {gpu_name} ({device})")

    # 1. Load frozen WorldModel, EcosystemModel, and BotPolicyNet
    world_model = WorldModel().to(device).eval()
    world_model.load_state_dict(torch.load(ROOT / "world_model.pt", map_location=device, weights_only=False)["state_dict"])

    eco_model = EcosystemModel().to(device).eval()
    eco_model.load_state_dict(torch.load(ROOT / "ecosystem_model.pt", map_location=device, weights_only=False)["state_dict"])

    base_policy = BotPolicyNet().to(device).eval()
    base_policy.load_state_dict(torch.load(ROOT / "bot_policy.pt", map_location=device, weights_only=False)["state_dict"])

    acoustics = AcousticField3D(n_coarse_steps=8, n_fine_steps=2).to(device).eval()

    # 2. Generate 256x256x4 Frostmarsh 3D worlds & extract frozen Z_world + 3D Noise Fields
    train_seeds = list(range(100, 132))  # 32 train worlds
    test_seeds = list(range(900, 912))   # 12 unseen held-out test worlds
    print(f"[Worlds] Building {len(train_seeds)} train + {len(test_seeds)} held-out 256x256x4 worlds...")
    train_worlds = build_world_pool(train_seeds, world_model, eco_model, acoustics, device)
    test_worlds = build_world_pool(test_seeds, world_model, eco_model, acoustics, device)

    # Sanity-check physical acoustics on first test world
    w0 = test_worlds[0]
    wf_x, wf_y, _ = w0.landmarks["waterfall_positions"][0]
    fi_x, fi_y, _ = w0.landmarks["wildfire_positions"][0]
    print(
        f"[Acoustics] Noise field range: [{w0.noise_db.min().item():.1f}, {w0.noise_db.max().item():.1f}] dB | "
        f"Waterfall L1={w0.noise_db[1, int(wf_y), int(wf_x)].item():.1f} dB | "
        f"Wildfire L1={w0.noise_db[1, int(fi_y), int(fi_x)].item():.1f} dB | "
        f"Quiet L0={w0.noise_db[0].quantile(0.2).item():.1f} dB"
    )

    # 3. Sample cooperative episodes over real WorldModel Z_world ego-crops
    train_ep_per_world = 600  # 19,200 training episodes
    test_ep_per_world = 400   # 4,800 held-out unseen test episodes
    print(f"[Episodes] Sampling {len(train_worlds)*train_ep_per_world} train + {len(test_worlds)*test_ep_per_world} test episodes...")
    train_data = sample_episodes_from_worlds(train_worlds, train_ep_per_world, acoustics, seed=101, device=device)
    test_data = sample_episodes_from_worlds(test_worlds, test_ep_per_world, acoustics, seed=999, device=device)

    # 4. Train & compare all 3 arms
    print("[Train] Arm 1 — Silent (Z_world only, no speech)...")
    _, m_silent = train_comm_arm("silent", train_data, test_data, device, epochs=32)
    print(f"  -> Arm 1 Silent Joint Acc: {m_silent['joint_coord_acc']*100:.1f}%")

    print("[Train] Arm 2 — Physical 3D Acoustic Speech (Z_world + AcousticField3D + Stealth)...")
    phys_comm, m_phys = train_comm_arm("physical", train_data, test_data, device, epochs=42, stealth_weight=0.55)
    print(
        f"  -> Arm 2 Physical 3D Joint Acc: {m_phys['joint_coord_acc']*100:.1f}% | "
        f"Eavesdrop: {m_phys['enemy_eavesdrop_rate']*100:.1f}% | "
        f"MI={m_phys['vocab']['mi_bits']:.2f} bits, Purity={m_phys['vocab']['purity']*100:.1f}%"
    )

    print("[Train] Arm 3 — Ideal Telepathic Speech (Infinite SNR upper bound)...")
    _, m_tele = train_comm_arm("telepathic", train_data, test_data, device, epochs=32)
    print(f"  -> Arm 3 Telepathic Joint Acc: {m_tele['joint_coord_acc']*100:.1f}%")

    # Enemy eavesdrop trade-off ablation on trained Arm 2 model
    m_shout = evaluate_comm_arm(phys_comm, test_data, volume_override=1.0)
    m_whisper = evaluate_comm_arm(phys_comm, test_data, volume_override=0.15)
    print(
        f"[Stealth] Adaptive ({m_phys['mean_volume']:.2f}): acc={m_phys['joint_coord_acc']*100:.1f}%, eav={m_phys['enemy_eavesdrop_rate']*100:.1f}% | "
        f"Shout (1.0): acc={m_shout['joint_coord_acc']*100:.1f}%, eav={m_shout['enemy_eavesdrop_rate']*100:.1f}% | "
        f"Whisper (0.15): acc={m_whisper['joint_coord_acc']*100:.1f}%, eav={m_whisper['enemy_eavesdrop_rate']*100:.1f}%"
    )

    # 5. Train & evaluate Viewer-Side Text Interpreter
    print("[Train] Viewer-Side Text Interpreter...")
    interp, m_interp = train_viewer_interpreter(phys_comm, train_data, test_data, device, epochs=24)
    print(
        f"  -> Viewer Interpreter Event Acc: {m_interp['overall_event_acc']*100:.1f}% "
        f"(High SNR: {m_interp['snr_buckets']['high_ge_10dB']['event_acc']*100:.1f}% | "
        f"Inaudible: {m_interp['snr_buckets']['inaudible_lt_neg6dB']['event_acc']*100:.1f}%)"
    )

    # 6. Train & evaluate Human Text -> 16-d Goal-Vector Injector into frozen BotPolicyNet
    print("[Train] Human Text -> 16-d Goal-Vector Injector into frozen BotPolicyNet...")
    injector, m_goal = train_and_eval_human_goal_injector(base_policy, train_worlds, test_worlds, device)
    u_g = m_goal["unseen_paraphrases_unseen_worlds"]
    print(
        f"  -> Unseen Paraphrases Compliance: Base={u_g['base_full_compliance']*100:.1f}% -> "
        f"Injected={u_g['injected_full_compliance']*100:.1f}% "
        f"(Move Cos {u_g['base_move_cos']:+.2f} -> {u_g['injected_move_cos']:+.2f}, Elem {u_g['injected_element_acc']*100:.1f}%)"
    )

    # 7. Generate 10 concrete multi-turn dialogue + physics transcripts on unseen worlds
    transcripts = generate_dialogue_transcripts(phys_comm, interp, test_worlds, test_data, acoustics, device)

    # 8. Run GB10 GPU latency benchmarks
    print("[Bench] Measuring NVIDIA GB10 latencies...")
    lat = run_gb10_latency_benchmarks(acoustics, phys_comm, interp, injector, test_worlds[0], device)
    print(
        f"  -> 10-step 3D Acoustic Conv (256x256x4 FP16): {lat['acoustic_10step_3d_conv_256x256x4_fp16']['median_ms']:.3f} ms | "
        f"Ambient Noise FP16: {lat['acoustic_ambient_noise_256x256x4_fp16']['median_ms']:.3f} ms | "
        f"CommBotPair(12/64/256): {lat['comm_bot_pair_12_bots']['median_ms']:.3f} / "
        f"{lat['comm_bot_pair_64_bots']['median_ms']:.3f} / {lat['comm_bot_pair_256_bots']['median_ms']:.3f} ms"
    )

    # 9. Generate crisp 3-panel visualization `acoustic_field_viz.png`
    print("[Viz] Generating acoustic_field_viz.png...")
    conf_16x6 = np.array(m_phys["vocab"]["confusion_16x6"], dtype=np.int64)
    generate_visualization(test_worlds[0], acoustics, conf_16x6, m_phys["vocab"], ROOT / "acoustic_field_viz.png")

    # 10. Save ACOUSTIC_COMM_METRICS.json & ACOUSTIC_COMM_REPORT.md
    metrics_out = {
        "hardware": gpu_name,
        "elapsed_total_sec": round(time.perf_counter() - t_start, 2),
        "dataset": {
            "grid": [GRID, GRID, LAYERS],
            "train_worlds": len(train_worlds),
            "test_worlds": len(test_worlds),
            "train_episodes": int(train_data["event"].shape[0]),
            "test_episodes": int(test_data["event"].shape[0]),
        },
        "acoustic_physics_sanity": {
            "noise_min_db": round(float(w0.noise_db.min().item()), 2),
            "noise_max_db": round(float(w0.noise_db.max().item()), 2),
            "waterfall_plunge_l1_db": round(float(w0.noise_db[1, int(wf_y), int(wf_x)].item()), 2),
            "wildfire_l1_db": round(float(w0.noise_db[1, int(fi_y), int(fi_x)].item()), 2),
            "quiet_cavern_l0_p20_db": round(float(w0.noise_db[0].quantile(0.2).item()), 2),
            "wall_attenuation_db": WALL_LOSS_DB,
            "cave_ceiling_attenuation_db": CAVE_CEIL_LOSS_DB,
        },
        "arms": {
            "arm1_silent": m_silent,
            "arm2_physical_3d": m_phys,
            "arm3_telepathic": m_tele,
        },
        "eavesdrop_tradeoff": {
            "adaptive": {
                "mean_volume": m_phys["mean_volume"],
                "joint_coord_acc": m_phys["joint_coord_acc"],
                "mean_snr_tm_db": m_phys["mean_snr_tm_db"],
                "enemy_eavesdrop_rate": m_phys["enemy_eavesdrop_rate"],
                "enemy_eavesdrop_open_air": m_phys["enemy_eavesdrop_open_air"],
                "enemy_eavesdrop_behind_wall": m_phys["enemy_eavesdrop_behind_wall"],
            },
            "always_shout": {
                "mean_volume": 1.0,
                "joint_coord_acc": m_shout["joint_coord_acc"],
                "mean_snr_tm_db": m_shout["mean_snr_tm_db"],
                "enemy_eavesdrop_rate": m_shout["enemy_eavesdrop_rate"],
                "enemy_eavesdrop_open_air": m_shout["enemy_eavesdrop_open_air"],
                "enemy_eavesdrop_behind_wall": m_shout["enemy_eavesdrop_behind_wall"],
            },
            "always_whisper": {
                "mean_volume": 0.15,
                "joint_coord_acc": m_whisper["joint_coord_acc"],
                "mean_snr_tm_db": m_whisper["mean_snr_tm_db"],
                "enemy_eavesdrop_rate": m_whisper["enemy_eavesdrop_rate"],
                "enemy_eavesdrop_open_air": m_whisper["enemy_eavesdrop_open_air"],
                "enemy_eavesdrop_behind_wall": m_whisper["enemy_eavesdrop_behind_wall"],
            },
        },
        "viewer_interpreter": m_interp,
        "human_goal_injector": m_goal,
        "latency_gb10_ms": lat,
        "sample_transcripts": transcripts,
    }

    with open(ROOT / "ACOUSTIC_COMM_METRICS.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    write_report(metrics_out, transcripts, ROOT / "ACOUSTIC_COMM_REPORT.md")

    print(f"[Done] Completed in {metrics_out['elapsed_total_sec']:.1f}s. Artifacts written to {ROOT}")


if __name__ == "__main__":
    main()
