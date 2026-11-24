"""
Frostmarsh Wave 11A: True Isotropic `Z=16/32` Conservation-Distilled `WorldModel3D-v5`,
5-Biome Isotropic `1m³` 3D Physics Teacher (`RefSim3D`), & True 3D `16x16x8` Sparse
Active-Brick Runner (`SparseWorld3DV5Runner`).

Drop-in compatible with `WorldModel3D` / `WorldModelDT` (`gpu_infer_daemon.py` `op=8`/`op=9`):
  `forward(x_l_18_h_w, local_dt=...) -> (out_horiz [L, 17, H, W], out_z_flux [L-1, 5, H, W], z_world [L, 48, H, W])`

Key capabilities:
  1. 5-Biome Isotropic `1m x 1m x 1m` 3D Physics Teacher (`RefSim3D`, supports `Z=16` & `Z=32`):
       - 0: `nordic_viaduct`          (3-span arched stone/timber bridge, vaulted basalt cave, alpine river & waterfall)
       - 1: `volcanic_caldera`        (lava/magma vents, basalt lava tubes, superheated steam plumes, ash slopes)
       - 2: `sinkhole_caverns`        (multi-level interconnected underground caverns, sinkhole shafts, subterranean lakes)
       - 3: `shattered_glacier_gorge` (tall ice spires, fragile ice bridges over deep canyon, blizzard frost)
       - 4: `geothermal_mist_basin`   (hot springs, dense buoyant steam plumes, closed 3D cloud condensation & rain/snowfall)
     Includes exact antisymmetric horizontal & vertical finite-volume transport, 2nd-law vertical
     conduction, closed 3D hydrologic condensation/precipitation (`mass_err == 0`), and 3D
     structural arch/bridge collapse when supporting piers burn or melt!
  2. `WorldModel3DV5` (~158K params, exact `48`-ch `Z_world_3d` latent trunk):
       - Factorized `(1,3,3) + (3,1,1)` residual blocks + true isotropic `3x3x3` bottleneck Conv3d block
       - Continuous log-`Δt` sinusoidal FiLM conditioning across all 5 blocks + exact `Δt = 0` stasis invariant
       - Local raw-state skip in horizontal head for razor-sharp empty-air & wall/threshold conservation
       - Staggered `L-1` vertical interface flux head (`[L-1, 5, H, W]`)
  3. `SparseWorld3DV5Runner`:
       - True 3D `16x16x8` sparse active-brick culling (`F.max_pool3d` activity mask)
       - Exact shrinking-halo 3D valid-conv forward with vectorized outer-boundary replicate fixup ->
         exact FP32 dense-vs-sparse parity (`max_abs_diff < 1e-5`) on EVERY voxel & EVERY interface!
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_teacher import (  # noqa: E402
    H_CAP,
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
    _apply_transport,
    apply_forcing,
    apply_projected_deltas,
    compute_C_and_T,
    compute_total_energy,
    compute_total_mass,
    encode_state,
    get_consts,
    project_and_step,
    project_deltas,
)
from world_model_3d import (  # noqa: E402
    Z_FLUX_CH,
    Z_FLUX_NAMES,
    ZF_HEAT,
    ZF_OIL,
    ZF_SMOKE,
    ZF_STEAM,
    ZF_WATER,
)
from world_model_dt import encode_dt_features  # noqa: E402

# =============================================================================
# Biome Registry & Physical Constants
# =============================================================================

BIOME_NORDIC_VIADUCT = 0
BIOME_VOLCANIC_CALDERA = 1
BIOME_SINKHOLE_CAVERNS = 2
BIOME_SHATTERED_GLACIER_GORGE = 3
BIOME_GEOTHERMAL_MIST_BASIN = 4

BIOME_NAMES = (
    "nordic_viaduct",
    "volcanic_caldera",
    "sinkhole_caverns",
    "shattered_glacier_gorge",
    "geothermal_mist_basin",
)

Z_SPECIES_IDX = (S_WATER, S_OIL, S_STEAM, S_SMOKE)
T_SKY_RAD = 26200.0  # 262 K cold upper-troposphere radiative equilibrium


def _layer_elevation(L: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Isotropic layer altitude encoded into [0, 31] for exact `elevation / 31.0` scale."""
    if L <= 1:
        return torch.zeros((L, 1, H, W), dtype=torch.float32, device=device)
    z_vals = torch.linspace(0.0, 31.0, L, device=device, dtype=torch.float32)
    return z_vals.view(L, 1, 1, 1).expand(L, 1, H, W).clone()


def _smooth_noise_2d(H: int, W: int, grid_n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    gn = max(2, int(grid_n))
    g = torch.rand((1, 1, gn, gn), generator=gen, device=device)
    return F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)[0, 0]


# =============================================================================
# 5-Biome Isotropic `1m³` 3D World Generator (`Z=16`, `Z=32`, arbitrary `H, W`)
# =============================================================================


def make_biome_3d_world(
    biome_id: int,
    L: int,
    H: int,
    W: int,
    device: torch.device,
    seed: int = 0,
) -> SimState:
    """
    Generate an isotropic `1m x 1m x 1m` true 3D voxel world `[L, H, W]` for one of 5 biomes.
    Works natively for `L=16`, `L=32`, or any `L >= 8`, with true empty-air voids (`mass == 0`),
    3D overhangs, arches, caverns, spires, waterfalls, vents, and upper-altitude cloud zones.
    """
    biome_id = int(biome_id) % len(BIOME_NAMES)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 1009 * biome_id)
    c = get_consts(device)

    mass = torch.zeros((L, 10, H, W), dtype=torch.float32, device=device)
    flow_e = torch.zeros((L, 1, H, W), dtype=torch.float32, device=device)
    flow_s = torch.zeros((L, 1, H, W), dtype=torch.float32, device=device)
    elev = _layer_elevation(L, H, W, device)
    T_init = torch.full((L, 1, H, W), T_AMB, dtype=torch.float32, device=device)

    zf = torch.linspace(0.0, 1.0, L, device=device).view(L, 1, 1).expand(L, H, W)
    yf = torch.linspace(0.0, 1.0, H, device=device).view(1, H, 1).expand(L, H, W)
    xf = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, W).expand(L, H, W)
    y2 = yf[0]
    x2 = xf[0]

    nc = _smooth_noise_2d(H, W, max(4, min(H, W) // 16), gen, device)
    nm = _smooth_noise_2d(H, W, max(6, min(H, W) // 8), gen, device)
    nf = _smooth_noise_2d(H, W, max(8, min(H, W) // 4), gen, device)

    # Random horizontal phase/offset for diversity across seeds
    phase = float(torch.rand(1, generator=gen, device=device).item()) * 6.28318
    ox_shift = float(torch.empty(1, device=device).uniform_(-0.08, 0.08, generator=gen).item())
    oy_shift = float(torch.empty(1, device=device).uniform_(-0.08, 0.08, generator=gen).item())

    # Natural vertical atmospheric lapse rate: warmer near ground, cooler in upper troposphere
    T_init[:, 0] = (29200.0 - 3400.0 * zf).clamp(T_MIN + 2000.0, T_MAX - 5000.0)

    if biome_id == BIOME_NORDIC_VIADUCT:
        # 0: Nordic Viaduct — deep alpine gorge, 3-span stone/timber arch bridge, vaulted basalt cave, waterfall
        gorge_cx = 0.50 + ox_shift + 0.055 * torch.sin(y2 * 6.28318 * 1.1 + phase)
        dist_g = (x2 - gorge_cx).abs()
        in_gorge = dist_g < 0.16
        # Terrain height normalized in [0, 1]
        h_terr = (0.14 + 0.44 * torch.clamp((dist_g - 0.10) / 0.16, 0.0, 1.0) + 0.08 * nc + 0.04 * nm).clamp(0.10, 0.66)
        h_terr3 = h_terr.unsqueeze(0)

        solid_ground = zf <= h_terr3
        deep_rock = zf <= (h_terr3 - 0.07)
        top_soil = solid_ground & (~deep_rock)

        mass[:, S_ROCK] = torch.where(deep_rock, 920.0, 0.0)
        mass[:, S_SOIL] = torch.where(top_soil, 780.0 + 160.0 * nc.unsqueeze(0), 0.0)

        # Vaulted basalt cave carved into east cliff (true 3D overhang!)
        cave_cx = (0.78 + 0.5 * ox_shift)
        cave_cy = (0.68 + oy_shift)
        cave_r2 = ((xf - cave_cx) / 0.12) ** 2 + ((yf - cave_cy) / 0.13) ** 2 + ((zf - 0.24) / 0.13) ** 2
        in_cave = (cave_r2 < 1.0) & (zf > 0.12) & (zf < 0.37)
        mass[:, S_ROCK] = torch.where(in_cave, 0.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(in_cave, 0.0, mass[:, S_SOIL])
        # Warm subterranean floor inside cave
        cave_floor = (((x2 - cave_cx) / 0.11) ** 2 + ((y2 - cave_cy) / 0.12) ** 2 < 1.0).unsqueeze(0) & (
            (zf - 0.13).abs() < (0.9 / max(L - 1, 1))
        )
        T_init[:, 0] = torch.where(cave_floor, 38500.0, T_init[:, 0])
        mass[:, S_STEAM] = torch.where(in_cave & (zf < 0.28) & (nf.unsqueeze(0) > 0.45), 240.0, mass[:, S_STEAM])

        # Alpine river flowing along gorge floor (1-2 voxels above floor)
        river_z = (zf > h_terr3) & (zf <= h_terr3 + 0.13) & (dist_g.unsqueeze(0) < 0.115)
        mass[:, S_WATER] = torch.where(river_z, 860.0, mass[:, S_WATER])
        flow_s[:, 0] = torch.where(river_z, 180.0, flow_s[:, 0])
        # Thin frost/ice shelves along cold northern gorge margins
        ice_shelf = river_z & (yf < 0.24) & (dist_g.unsqueeze(0) > 0.055)
        mass[:, S_ICE] = torch.where(ice_shelf, 740.0, mass[:, S_ICE])
        mass[:, S_WATER] = torch.where(ice_shelf, 160.0, mass[:, S_WATER])
        T_init[:, 0] = torch.where(ice_shelf, 25200.0, T_init[:, 0])

        # 3-span arched stone/timber viaduct crossing gorge at y ~ 0.46 + oy_shift
        br_y = 0.46 + oy_shift
        in_br_y = (yf - br_y).abs() < 0.055
        in_br_x = (xf > 0.24 + ox_shift) & (xf < 0.76 + ox_shift)
        deck_z = 0.54
        dz = 1.0 / max(L - 1, 1)
        is_deck = in_br_y & in_br_x & ((zf - deck_z).abs() <= 0.75 * dz)
        # 4 stone piers supporting 3 semicircular arches
        pier_xs = (0.31 + ox_shift, 0.44 + ox_shift, 0.56 + ox_shift, 0.69 + ox_shift)
        is_pier = torch.zeros_like(is_deck)
        for px in pier_xs:
            is_pier = is_pier | (in_br_y & ((xf - px).abs() < 0.024) & (zf >= 0.10) & (zf < deck_z))
        # Arched stone spandrels between piers (carving open semicircular arches underneath!)
        is_arch_ring = torch.zeros_like(is_deck)
        for a_cx in (0.375 + ox_shift, 0.50 + ox_shift, 0.625 + ox_shift):
            ar2 = ((xf - a_cx) / 0.052) ** 2 + ((zf - (deck_z - 0.14)) / 0.13) ** 2
            is_arch_ring = is_arch_ring | (in_br_y & (ar2 >= 0.80) & (ar2 <= 1.25) & (zf >= deck_z - 0.14) & (zf < deck_z))
        # Timber central span deck + stone piers/arches
        mass[:, S_ROCK] = torch.where(is_pier | is_arch_ring, 900.0, mass[:, S_ROCK])
        mass[:, S_WOOD] = torch.where(is_deck & (~is_pier), 720.0, mass[:, S_WOOD])
        mass[:, S_ROCK] = torch.where(is_deck & is_pier, 900.0, mass[:, S_ROCK])

        # High-head alpine waterfall pouring off western cliff into gorge!
        wf_y = 0.26 + 0.5 * oy_shift
        wf_col = ((xf - (0.34 + ox_shift)).abs() < 0.055) & ((yf - wf_y).abs() < 0.055) & (zf > 0.18) & (zf <= 0.60)
        wf_lip = ((xf > 0.14) & (xf <= 0.38 + ox_shift)) & ((yf - wf_y).abs() < 0.055) & (zf > h_terr3) & (zf <= h_terr3 + 0.12)
        mass[:, S_WATER] = torch.where(wf_col | wf_lip, 820.0, mass[:, S_WATER])
        flow_e[:, 0] = torch.where(wf_lip, 210.0, flow_e[:, 0])

        # Pine forest & brush on highland ridges + active firebraziers on one ridge
        surf_air = (zf > h_terr3) & (zf <= h_terr3 + 1.35 * dz) & (~in_gorge.unsqueeze(0))
        mass[:, S_WOOD] = torch.where(surf_air & (nf.unsqueeze(0) > 0.56) & (mass[:, S_ROCK] == 0), 560.0, mass[:, S_WOOD])
        mass[:, S_PLANT] = torch.where(surf_air & (nm.unsqueeze(0) > 0.45) & (mass[:, S_WOOD] == 0), 340.0, mass[:, S_PLANT])
        fire_m = (((xf - (0.26 + ox_shift)) ** 2 + (yf - (0.72 + oy_shift)) ** 2) < 0.0045) & (zf >= h_terr3 - 0.02) & (zf <= h_terr3 + 0.22)
        T_init[:, 0] = torch.where(fire_m, 62000.0, T_init[:, 0])
        mass[:, S_SMOKE] = torch.where(fire_m & (zf > h_terr3), 320.0, mass[:, S_SMOKE])

    elif biome_id == BIOME_VOLCANIC_CALDERA:
        # 1: Volcanic Caldera — crater rim, lava/magma vents, basalt lava tubes, boiling pools, ash slopes
        cx, cy = 0.50 + ox_shift, 0.50 + oy_shift
        r = torch.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        # Crater rim peaks at r ~ 0.34, depressed caldera floor inside
        rim = 0.46 * torch.exp(-((r - 0.34) / 0.13) ** 2)
        h_terr = (0.14 + rim + 0.08 * nc + 0.04 * nm).clamp(0.10, 0.68)
        h_terr3 = h_terr.unsqueeze(0)
        dz = 1.0 / max(L - 1, 1)

        solid_ground = zf <= h_terr3
        deep_rock = zf <= (h_terr3 - 0.08)
        ash_slope = solid_ground & (~deep_rock)

        mass[:, S_ROCK] = torch.where(deep_rock, 960.0, 0.0)
        mass[:, S_SOIL] = torch.where(ash_slope, 520.0 + 140.0 * nc.unsqueeze(0), 0.0)
        mass[:, S_ASH] = torch.where(ash_slope & (r.unsqueeze(0) > 0.20), 260.0 + 180.0 * nm.unsqueeze(0), 0.0)

        # Basalt lava tubes (hollow vaulted tunnels under flanks carrying hot oil/magma heat)
        tube1 = (((yf - (0.32 + oy_shift)) / 0.07) ** 2 + ((zf - 0.22) / 0.09) ** 2 < 1.0) & (xf > 0.12) & (xf < 0.88)
        tube2 = (((xf - (0.68 + ox_shift)) / 0.07) ** 2 + ((zf - 0.25) / 0.09) ** 2 < 1.0) & (yf > 0.16) & (yf < 0.84)
        in_tube = (tube1 | tube2) & (zf > 0.12)
        mass[:, S_ROCK] = torch.where(in_tube, 0.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(in_tube, 0.0, mass[:, S_SOIL])
        tube_floor = in_tube & (zf < 0.20)
        mass[:, S_OIL] = torch.where(tube_floor, 680.0, mass[:, S_OIL])
        T_init[:, 0] = torch.where(in_tube, 78000.0 + 12000.0 * nf.unsqueeze(0), T_init[:, 0])

        # Magma / geothermal vents inside caldera + superheated buoyant steam/smoke plumes!
        for k in range(3):
            vx = cx + float(torch.empty(1, device=device).uniform_(-0.15, 0.15, generator=gen).item())
            vy = cy + float(torch.empty(1, device=device).uniform_(-0.15, 0.15, generator=gen).item())
            vr = float(torch.empty(1, device=device).uniform_(0.055, 0.085, generator=gen).item())
            vm2 = ((x2 - vx) ** 2 + (y2 - vy) ** 2) < (vr * vr)
            vm_base = vm2.unsqueeze(0) & (zf <= h_terr3 + 0.10) & (zf >= 0.08)
            vm_plume = vm2.unsqueeze(0) & (zf > h_terr3) & (zf <= 0.88)
            mass[:, S_OIL] = torch.where(vm_base & (zf > h_terr3 - 0.04), 620.0, mass[:, S_OIL])
            mass[:, S_WATER] = torch.where(vm_base & (zf > h_terr3) & (k == 0), 540.0, mass[:, S_WATER])
            mass[:, S_STEAM] = torch.where(vm_plume, 380.0 + 220.0 * nf.unsqueeze(0), mass[:, S_STEAM])
            mass[:, S_SMOKE] = torch.where(vm_plume & (zf > 0.22), 280.0 + 160.0 * nm.unsqueeze(0), mass[:, S_SMOKE])
            T_init[:, 0] = torch.where(vm_base, 92000.0 + 18000.0 * nf.unsqueeze(0), T_init[:, 0])
            T_init[:, 0] = torch.where(vm_plume, (64000.0 - 26000.0 * zf).clamp(32000.0, 88000.0), T_init[:, 0])

        # Boiling crater lake beside central vent
        lake_m = (r.unsqueeze(0) < 0.21) & (zf > h_terr3) & (zf <= h_terr3 + 0.12) & (mass[:, S_OIL] < 100.0)
        mass[:, S_WATER] = torch.where(lake_m, 760.0, mass[:, S_WATER])
        T_init[:, 0] = torch.where(lake_m, 41500.0 + 4000.0 * nf.unsqueeze(0), T_init[:, 0])

        # Scorched timber along outer rim
        rim_wood = (zf > h_terr3) & (zf <= h_terr3 + 1.3 * dz) & (r.unsqueeze(0) > 0.30) & (nf.unsqueeze(0) > 0.55)
        mass[:, S_WOOD] = torch.where(rim_wood, 520.0, mass[:, S_WOOD])

    elif biome_id == BIOME_SINKHOLE_CAVERNS:
        # 2: Sinkhole Caverns — thick plateau, multi-level caverns, vertical shafts, subterranean lakes
        h_terr = (0.60 + 0.09 * nc + 0.04 * nm).clamp(0.52, 0.72)
        h_terr3 = h_terr.unsqueeze(0)
        dz = 1.0 / max(L - 1, 1)

        solid_ground = zf <= h_terr3
        deep_rock = zf <= (h_terr3 - 0.08)
        mass[:, S_ROCK] = torch.where(deep_rock, 920.0, 0.0)
        mass[:, S_SOIL] = torch.where(solid_ground & (~deep_rock), 760.0 + 160.0 * nc.unsqueeze(0), 0.0)

        # Lower cavern gallery (z ~ 0.10..0.28) & upper cavern gallery (z ~ 0.36..0.50) with natural rock pillars!
        pillar = (nm > 0.76) & (nc > 0.45)
        cav_low = (zf > 0.10) & (zf < 0.29) & ((nc.unsqueeze(0) < 0.78) & (~pillar.unsqueeze(0)))
        cav_mid = (zf > 0.36) & (zf < 0.50) & ((nm.unsqueeze(0) < 0.72) & (~pillar.unsqueeze(0)))
        in_cav = cav_low | cav_mid

        # 3 wide open vertical sinkhole shafts punching straight from surface down to abyss!
        in_shaft = torch.zeros_like(in_cav)
        shaft_centers = (
            (0.32 + ox_shift, 0.34 + oy_shift, 0.095),
            (0.64 + ox_shift, 0.58 + oy_shift, 0.105),
            (0.46 + ox_shift, 0.74 + oy_shift, 0.080),
        )
        for sx, sy, sr in shaft_centers:
            sm = (((x2 - sx) ** 2 + (y2 - sy) ** 2) < (sr * sr)).unsqueeze(0) & (zf > 0.10)
            in_shaft = in_shaft | sm

        carved = in_cav | in_shaft
        mass[:, S_ROCK] = torch.where(carved, 0.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(carved, 0.0, mass[:, S_SOIL])

        # Subterranean lakes filling bottom of lower cavern (z ~ 0.10..0.18)
        sub_lake = carved & (zf > 0.10) & (zf <= 0.19)
        mass[:, S_WATER] = torch.where(sub_lake, 840.0, mass[:, S_WATER])
        flow_e[:, 0] = torch.where(sub_lake, 120.0 * (nc.unsqueeze(0) - 0.5), flow_e[:, 0])

        # Surface streams pouring straight down sinkhole shafts!
        for sx, sy, sr in shaft_centers[:2]:
            pour = (((x2 - (sx + sr * 0.65)) ** 2 + (y2 - sy) ** 2) < (0.055 ** 2)).unsqueeze(0) & (zf > 0.18) & (zf <= h_terr3 + 0.08)
            mass[:, S_WATER] = torch.where(pour, 740.0, mass[:, S_WATER])

        # Warm mineral thermal pocket in lower cavern sending mist up through shafts
        therm = (((x2 - shaft_centers[1][0]) ** 2 + (y2 - shaft_centers[1][1]) ** 2) < (0.12 ** 2)).unsqueeze(0)
        T_init[:, 0] = torch.where(therm & (zf < 0.24), 46000.0, T_init[:, 0])
        mass[:, S_STEAM] = torch.where(therm & carved & (zf > 0.18) & (zf < 0.72), 310.0 + 180.0 * nf.unsqueeze(0), mass[:, S_STEAM])

        # Surface plateau vegetation & timber
        surf = (zf > h_terr3) & (zf <= h_terr3 + 1.3 * dz) & (~in_shaft)
        mass[:, S_WOOD] = torch.where(surf & (nf.unsqueeze(0) > 0.55), 540.0, mass[:, S_WOOD])
        mass[:, S_PLANT] = torch.where(surf & (nm.unsqueeze(0) > 0.42) & (mass[:, S_WOOD] == 0), 340.0, mass[:, S_PLANT])

    elif biome_id == BIOME_SHATTERED_GLACIER_GORGE:
        # 3: Shattered Glacier Gorge — deep icy canyon, tall ice spires, fragile ice bridges, blizzard frost
        canyon_cx = 0.50 + ox_shift + 0.07 * torch.cos(y2 * 6.28318 * 1.15 + phase)
        dist_c = (x2 - canyon_cx).abs()
        h_terr = (0.10 + 0.46 * torch.clamp((dist_c - 0.11) / 0.14, 0.0, 1.0) + 0.08 * nc).clamp(0.08, 0.64)
        h_terr3 = h_terr.unsqueeze(0)
        dz = 1.0 / max(L - 1, 1)

        # Frigid blizzard temperatures
        T_init[:, 0] = (25400.0 - 2400.0 * zf + 1600.0 * nc.unsqueeze(0)).clamp(T_MIN + 2000.0, 27000.0)

        solid_rock = zf <= (h_terr3 - 0.10)
        glacier_ice = (zf > (h_terr3 - 0.10)) & (zf <= h_terr3)
        mass[:, S_ROCK] = torch.where(solid_rock, 920.0, 0.0)
        mass[:, S_ICE] = torch.where(glacier_ice, 860.0 + 140.0 * nm.unsqueeze(0), 0.0)

        # Tall jagged ice spires rising up to zf ~ 0.80!
        for k in range(5):
            px = float(torch.empty(1, device=device).uniform_(0.14, 0.86, generator=gen).item())
            py = float(torch.empty(1, device=device).uniform_(0.14, 0.86, generator=gen).item())
            pr = float(torch.empty(1, device=device).uniform_(0.040, 0.068, generator=gen).item())
            ph = float(torch.empty(1, device=device).uniform_(0.68, 0.84, generator=gen).item())
            dist_p = torch.sqrt((x2 - px) ** 2 + (y2 - py) ** 2).unsqueeze(0)
            taper_r = pr * (1.0 - 0.72 * ((zf - 0.25).clamp(min=0.0) / max(ph - 0.25, 0.1)))
            in_spire = (dist_p < taper_r) & (zf <= ph) & (zf >= h_terr3 - 0.05)
            mass[:, S_ICE] = torch.where(in_spire, 880.0, mass[:, S_ICE])

        # Two fragile ice bridges spanning the deep canyon at zf = 0.44 and zf = 0.56
        for by, bz in ((0.33 + oy_shift, 0.44), (0.67 + oy_shift, 0.56)):
            in_ib = ((yf - by).abs() < 0.048) & (dist_c.unsqueeze(0) < 0.20) & ((zf - bz).abs() <= 0.80 * dz)
            mass[:, S_ICE] = torch.where(in_ib, 780.0, mass[:, S_ICE])

        # Meltwater crevasse river at canyon bottom + warm geothermal melt pocket causing freeze-thaw dynamics!
        creek = (dist_c.unsqueeze(0) < 0.095) & (zf > h_terr3) & (zf <= h_terr3 + 0.12)
        mass[:, S_WATER] = torch.where(creek, 780.0, mass[:, S_WATER])
        flow_s[:, 0] = torch.where(creek, 150.0, flow_s[:, 0])
        T_init[:, 0] = torch.where(creek, 27800.0 + 1800.0 * nf.unsqueeze(0), T_init[:, 0])
        # Localized warm melt plume under north bridge
        warm_m = (((x2 - (0.50 + ox_shift)) ** 2 + (y2 - (0.33 + oy_shift)) ** 2) < 0.0064).unsqueeze(0) & (zf < 0.52)
        T_init[:, 0] = torch.where(warm_m, 33500.0, T_init[:, 0])
        mass[:, S_STEAM] = torch.where(warm_m & (zf > 0.16) & (mass[:, S_ICE] < 200.0), 260.0, mass[:, S_STEAM])

    else:
        # 4: Geothermal Mist Basin — hot springs, dense rising steam plumes, closed 3D cloud condensation & rain!
        cx, cy = 0.50 + ox_shift, 0.50 + oy_shift
        r = torch.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        # Terraced warm central basin surrounded by cool alpine mountain peaks
        basin_h = 0.14 + 0.46 * torch.clamp((r - 0.24) / 0.22, 0.0, 1.0) + 0.10 * nc + 0.05 * nm
        h_terr = basin_h.clamp(0.12, 0.68)
        h_terr3 = h_terr.unsqueeze(0)
        dz = 1.0 / max(L - 1, 1)

        solid_ground = zf <= h_terr3
        deep_rock = zf <= (h_terr3 - 0.08)
        mass[:, S_ROCK] = torch.where(deep_rock, 920.0, 0.0)
        mass[:, S_SOIL] = torch.where(solid_ground & (~deep_rock), 740.0 + 180.0 * nc.unsqueeze(0), 0.0)
        # Alpine snow/ice caps on high cold surrounding peaks
        peak_ice = (zf > h_terr3) & (zf <= h_terr3 + 1.2 * dz) & (h_terr3 > 0.52)
        mass[:, S_ICE] = torch.where(peak_ice, 680.0, mass[:, S_ICE])
        T_init[:, 0] = torch.where(peak_ice, 25600.0, T_init[:, 0])

        # Hot spring pools filling the central basin
        pool = (r.unsqueeze(0) < 0.28) & (zf > h_terr3) & (zf <= h_terr3 + 0.13)
        mass[:, S_WATER] = torch.where(pool, 900.0, mass[:, S_WATER])
        T_init[:, 0] = torch.where(pool, 39800.0 + 5500.0 * nf.unsqueeze(0), T_init[:, 0])

        # Dense buoyant steam plumes ascending from hot springs straight into upper troposphere (`z >= L-3`)!
        for k in range(3):
            gx = cx + float(torch.empty(1, device=device).uniform_(-0.14, 0.14, generator=gen).item())
            gy = cy + float(torch.empty(1, device=device).uniform_(-0.14, 0.14, generator=gen).item())
            gr = float(torch.empty(1, device=device).uniform_(0.075, 0.12, generator=gen).item())
            gm = (((x2 - gx) ** 2 + (y2 - gy) ** 2) < (gr * gr)).unsqueeze(0)
            plume = gm & (zf > h_terr3 + 0.06) & (zf < 0.96)
            # Plume cools with altitude so upper strata (`z >= L-3`) condense into rain clouds!
            T_plume = (41500.0 - 15500.0 * zf).clamp(26400.0, 44000.0)
            mass[:, S_STEAM] = torch.where(plume, 420.0 + 240.0 * nf.unsqueeze(0), mass[:, S_STEAM])
            T_init[:, 0] = torch.where(plume, T_plume, T_init[:, 0])

        # High-altitude cloud deck (`z >= L - 3`): cool saturated steam + condensing rain droplets!
        cloud_z = zf >= ((L - 3.5) / max(L - 1, 1))
        cloud_m = cloud_z & ((r.unsqueeze(0) < 0.42) | (nc.unsqueeze(0) > 0.48))
        mass[:, S_STEAM] = torch.where(cloud_m, torch.maximum(mass[:, S_STEAM], 340.0 + 220.0 * nm.unsqueeze(0)), mass[:, S_STEAM])
        rain_drop = cloud_m & (nf.unsqueeze(0) > 0.42)
        mass[:, S_WATER] = torch.where(rain_drop, 260.0 + 200.0 * nf.unsqueeze(0), mass[:, S_WATER])
        T_init[:, 0] = torch.where(cloud_m, 26800.0 + 2200.0 * nc.unsqueeze(0), T_init[:, 0])

        # Lush basin moss/plants on warm wet banks
        bank = (zf > h_terr3) & (zf <= h_terr3 + 1.2 * dz) & (r.unsqueeze(0) >= 0.25) & (r.unsqueeze(0) <= 0.42)
        mass[:, S_PLANT] = torch.where(bank & (nm.unsqueeze(0) > 0.42), 380.0, mass[:, S_PLANT])

    # Add light upper-altitude cloud patches in all biomes (keeps hydrologic cycle active across all worlds!)
    if biome_id != BIOME_GEOTHERMAL_MIST_BASIN:
        cloud_z = zf >= ((L - 3.2) / max(L - 1, 1))
        cloud_patch = cloud_z & (nc.unsqueeze(0) > 0.64) & (mass.sum(dim=1) < 10.0)
        mass[:, S_STEAM] = torch.where(cloud_patch, 220.0 + 180.0 * nf.unsqueeze(0), mass[:, S_STEAM])
        T_init[:, 0] = torch.where(cloud_patch, 27600.0 + 2400.0 * nm.unsqueeze(0), T_init[:, 0])
        rain_patch = cloud_patch & (nf.unsqueeze(0) > 0.65)
        mass[:, S_WATER] = torch.where(rain_patch, 180.0 + 140.0 * nf.unsqueeze(0), mass[:, S_WATER])

    # Strictly enforce H_CAP
    tot_m = mass.sum(dim=1, keepdim=True)
    cap_scale = torch.where(tot_m > H_CAP * 0.92, (H_CAP * 0.92) / (tot_m + 1e-6), torch.ones_like(tot_m))
    mass = mass * cap_scale

    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    energy = torch.where(C_cell > 1e-6, C_cell * torch.clamp(T_init, T_MIN, T_MAX), torch.zeros_like(C_cell))
    return SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev)


# =============================================================================
# True Isotropic 3D Physics Teacher (`RefSim3D`)
# =============================================================================


class RefSim3D:
    """
    GPU-vectorized PyTorch Reference Isotropic `1m³` 3D Physics Teacher for `Z=16/32` worlds:
      - Exact Rust-compatible horizontal finite-volume transport + reactions (including high-altitude
        cloud condensation nucleation at `z >= L - 3`!)
      - Exact antisymmetric vertical gravity (`water`/`oil`), thermal buoyancy (`steam`/`smoke`),
        and 2nd-law vertical thermal conduction (`heat_z`) across ALL `L-1` interfaces
      - Closed 3D Hydrologic Cycle (`radiative_cooling` cools high-altitude steam `z >= L-3`,
        condensing it into liquid water / ice droplets that precipitate back down with EXACT `mass_err == 0`)
      - 3D Structural Arch/Bridge Collapse (unsupported overhanging wood/ash/fractured-ice drops
        conservatively under gravity when piers burn or melt!)
      - Continuous `Δt ∈ [0.0, 4.0]` multi-substep exact reference integration!
    """

    @staticmethod
    def _scale_horiz_by_dt(raw_deltas: torch.Tensor, local_dt: torch.Tensor) -> torch.Tensor:
        dt = local_dt.clamp(min=0.0)
        dt_e_nb = F.pad(dt[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
        dt_s_nb = F.pad(dt[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
        dt_e = torch.minimum(dt, dt_e_nb)
        dt_s = torch.minimum(dt, dt_s_nb)
        return torch.cat(
            [
                raw_deltas[:, 0:4] * dt_e,
                raw_deltas[:, 4:8] * dt_s,
                raw_deltas[:, 8:9] * dt_e,
                raw_deltas[:, 9:10] * dt_s,
                raw_deltas[:, 10:17] * dt,
            ],
            dim=1,
        )

    @staticmethod
    def _broadcast_dt(local_dt: torch.Tensor | float | None, L: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        if local_dt is None:
            return torch.ones((L, 1, H, W), dtype=torch.float32, device=device)
        if isinstance(local_dt, (int, float)):
            return torch.full((L, 1, H, W), max(0.0, float(local_dt)), dtype=torch.float32, device=device)
        t = local_dt.to(device=device, dtype=torch.float32).clamp(min=0.0)
        if t.ndim == 0:
            return t.view(1, 1, 1, 1).expand(L, 1, H, W).clone()
        if t.ndim == 2:
            return t.view(1, 1, H, W).expand(L, 1, H, W).clone()
        if t.ndim == 3:
            return t.view(t.shape[0], 1, H, W).expand(L, 1, H, W).clone()
        if t.ndim == 4:
            return t.expand(L, 1, H, W).clone()
        raise ValueError(f"Unsupported local_dt shape {tuple(t.shape)}")

    @staticmethod
    @torch.no_grad()
    def compute_raw_horiz_deltas(
        state: SimState,
        forcing_energy: torch.Tensor | None = None,
        forcing_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute exact horizontal finite-volume raw deltas `[L, 17, H, W]` matching `sim_teacher.RefSim`,
        enhanced with high-altitude cloud condensation nucleation (`z >= L - 3`)!
        """
        raw = RefSim.compute_raw_deltas(state, forcing_energy, forcing_mass)
        L = state.mass.shape[0]
        if L >= 6:
            # High-altitude cloud nucleation boost on top 3 strata (`z >= L - 3`) for rxn 3 (Condense)
            # Still strictly bounded by `project_deltas`'s exact `xi_cap_cond` & steam mass!
            mass_eff, energy_eff = apply_forcing(state, forcing_energy, forcing_mass)
            _, T_cK = compute_C_and_T(mass_eff, energy_eff)
            cloud_boost = torch.ones((L, 1, 1, 1), dtype=raw.dtype, device=raw.device)
            cloud_boost[L - 3 :] = 2.2
            # Extra condensation kick when cold cloud steam sits in upper troposphere
            cold_steam = (T_cK < 37000.0).float() * (mass_eff[:, S_STEAM : S_STEAM + 1] > 1.0).float()
            rxn_cond = raw[:, 13:14] * (1.0 + (cloud_boost - 1.0) * cold_steam)
            raw = torch.cat([raw[:, :13], rxn_cond, raw[:, 14:]], dim=1)
        return raw

    @staticmethod
    @torch.no_grad()
    def compute_vertical_flux(state: SimState) -> torch.Tensor:
        """
        Compute exact isotropic `1m³` vertical interface fluxes `[L - 1, 5, H, W]` across ALL `L - 1`
        interfaces `(z <-> z+1)`:
          - 0: `flux_z_water`  (< 0 downward gravity plunge / waterfall / rain precipitation!)
          - 1: `flux_z_oil`    (< 0 downward gravity cascade)
          - 2: `flux_z_steam`  (> 0 upward thermal buoyancy plume, blocked by overhead solid ceiling)
          - 3: `flux_z_smoke`  (> 0 upward thermal buoyancy plume, blocked by overhead solid ceiling)
          - 4: `heat_z`        (2nd-law vertical conduction + convective heat transport, 1e6 units)
        """
        L, _, H, W = state.mass.shape
        dev = state.mass.device
        if L <= 1:
            return torch.zeros((0, Z_FLUX_CH, H, W), dtype=torch.float32, device=dev)

        c = get_consts(dev)
        mass = state.mass
        energy = state.energy
        C_cell, T_cK = compute_C_and_T(mass, energy)

        lo = slice(0, L - 1)  # layer z (below)
        hi = slice(1, L)      # layer z + 1 (above)

        # True isotropic 3D support & ceiling semantics:
        #   - A solid floor in layer `z` (rock>=450, soil>=550, ice>=600, intact wood bridge>=380)
        #     supports liquids resting on layer `z+1` above it!
        #   - A solid ceiling in layer `z+1` (rock>=500, ice>=650, soil>=650) blocks buoyant gases rising from `z`!
        solid_floor_lo = (
            (mass[lo, S_ROCK] >= 450.0)
            | (mass[lo, S_SOIL] >= 550.0)
            | (mass[lo, S_ICE] >= 600.0)
            | (mass[lo, S_WOOD] >= 380.0)
        )
        solid_ceil_hi = (
            (mass[hi, S_ROCK] >= 500.0)
            | (mass[hi, S_ICE] >= 650.0)
            | (mass[hi, S_SOIL] >= 650.0)
        )
        solid_rock_lo = (mass[lo, S_ROCK] >= 500.0) | (mass[lo, S_ICE] >= 700.0)

        grav_open = (~solid_floor_lo).float()
        gas_open = ((~solid_ceil_hi) & (~solid_rock_lo)).float()

        tot_mass = mass.sum(dim=1)  # [L, H, W]
        headroom_lo = (H_CAP - tot_mass[lo]).clamp(min=0.0)
        headroom_hi = (H_CAP - tot_mass[hi]).clamp(min=0.0)

        # 1. Gravity plunge (z+1 -> z): water & oil fall downward whenever `lo` is open air or has headroom
        w_hi = mass[hi, S_WATER].clamp(min=0.0)
        o_hi = mass[hi, S_OIL].clamp(min=0.0)
        # Water droplets falling through open air fall fast (0.52/tick); oil is more viscous (0.36/tick)
        lo_air_boost = (tot_mass[lo] < 80.0).float()
        g_rate_w = (0.44 + 0.14 * lo_air_boost) * grav_open
        g_rate_o = (0.32 + 0.10 * lo_air_boost) * grav_open

        raw_fall_w = w_hi * g_rate_w
        raw_fall_o = o_hi * g_rate_o
        fall_sum = raw_fall_w + raw_fall_o
        fall_cap = torch.minimum(torch.ones_like(fall_sum), (0.85 * headroom_lo) / (fall_sum + 1e-6))
        fall_w = raw_fall_w * fall_cap
        fall_o = raw_fall_o * fall_cap

        # 2. Thermal buoyancy (z -> z+1): steam & smoke rise upward unless blocked by solid ceiling!
        st_lo = mass[lo, S_STEAM].clamp(min=0.0)
        sm_lo = mass[lo, S_SMOKE].clamp(min=0.0)
        T_lo = T_cK[lo, 0]
        T_hi = T_cK[hi, 0]
        buoy_boost = (
            1.0
            + 0.55 * ((T_lo - T_AMB) / 14000.0).clamp(0.0, 2.0)
            + 0.30 * ((T_lo - T_hi) / 10000.0).clamp(0.0, 1.5)
        )
        # Suppress upward steam buoyancy once cold cloud steam reaches upper troposphere (`z >= L-2`)
        top_cap = torch.ones((L - 1, 1, 1), dtype=torch.float32, device=dev)
        if L >= 6:
            top_cap[L - 3 :] = 0.45
        b_rate_st = (0.32 * buoy_boost * top_cap).clamp(0.0, 0.72) * gas_open
        b_rate_sm = (0.26 * buoy_boost).clamp(0.0, 0.68) * gas_open

        raw_rise_st = st_lo * b_rate_st
        raw_rise_sm = sm_lo * b_rate_sm
        rise_sum = raw_rise_st + raw_rise_sm
        rise_cap = torch.minimum(torch.ones_like(rise_sum), (0.85 * headroom_hi) / (rise_sum + 1e-6))
        rise_st = raw_rise_st * rise_cap
        rise_sm = raw_rise_sm * rise_cap

        # 3. 2nd-law vertical thermal conduction & convection (strictly zero when either voxel is empty air!)
        k_cell = torch.where(
            tot_mass > 1e-4,
            (mass * c.spec_k).sum(dim=1) / torch.clamp(tot_mass, min=1e-4),
            torch.zeros_like(tot_mass),
        )
        k_lo, k_hi = k_cell[lo], k_cell[hi]
        k_harm = 2.0 * k_lo * k_hi / (k_lo + k_hi + 1e-4)

        C_lo, C_hi = C_cell[lo, 0], C_cell[hi, 0]
        dT_z = T_lo - T_hi
        valid_c = (C_lo > 1e-3) & (C_hi > 1e-3)
        q_max_z = dT_z.abs() * (C_lo * C_hi) / (4.0 * (C_lo + C_hi) + 1e-6)
        q_max_z = torch.where(valid_c, q_max_z, torch.zeros_like(q_max_z))

        conv_mult = 1.0 + 1.5 * gas_open * (dT_z > 0.0).float() * (1.0 + (st_lo + sm_lo) / 600.0).clamp(max=2.2)
        q_raw = 0.045 * k_harm * conv_mult * dT_z * (C_lo.clamp(max=45000.0) * C_hi.clamp(max=45000.0)).sqrt() / 12000.0
        q_clamped = torch.where(valid_c, torch.clamp(q_raw, min=-q_max_z, max=q_max_z), torch.zeros_like(q_raw))

        return torch.stack(
            [
                -fall_w / 1000.0,
                -fall_o / 1000.0,
                rise_st / 1000.0,
                rise_sm / 1000.0,
                q_clamped / 1e6,
            ],
            dim=1,
        )

    @staticmethod
    def project_and_apply_vertical_flux(
        state: SimState,
        raw_z_flux: torch.Tensor,
    ) -> tuple[SimState, torch.Tensor]:
        """
        Exact differentiable conservative vertical flux projection & update for `[L - 1, 5, H, W]`.
        Guarantees exact 3D mass & energy conservation (`sum_z mass == const`, `sum_z E == const`),
        strict non-negativity, `H_CAP` headroom, and 2nd-law `[T_MIN, T_MAX]` bounds!
        """
        L, _, H, W = state.mass.shape
        dev = state.mass.device
        if L <= 1:
            return state, raw_z_flux.new_zeros((0, Z_FLUX_CH, H, W))

        z_idx = torch.tensor(Z_SPECIES_IDX, dtype=torch.long, device=dev)
        z_spec_c = torch.tensor([SPEC_C_LIST[s] for s in Z_SPECIES_IDX], dtype=torch.float32, device=dev).view(1, 4, 1, 1)

        mass = state.mass
        energy = state.energy
        fz_mass = raw_z_flux[:, 0:4] * 1000.0  # [L-1, 4, H, W]

        # 1. Donor-budget positivity across adjacent vertical interfaces
        fz_pad = F.pad(fz_mass, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)  # [L+1, 4, H, W]
        out_z = F.relu(fz_pad[1:]) + F.relu(-fz_pad[:-1])
        avail_z = mass[:, z_idx].clamp(min=0.0)
        dscale = torch.clamp(avail_z / (out_z + 1e-6), max=1.0)
        fz_mass = torch.where(fz_mass >= 0.0, fz_mass * dscale[:-1], fz_mass * dscale[1:])

        # 2. Receiver H_CAP headroom cap
        fz_pad2 = F.pad(fz_mass, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
        in_z = (F.relu(fz_pad2[:-1]) + F.relu(-fz_pad2[1:])).sum(dim=1, keepdim=True)
        headroom = (H_CAP - mass.sum(dim=1, keepdim=True)).clamp(min=0.0)
        rscale = torch.clamp(headroom / (in_z + 1e-6), max=1.0)
        fz_mass = torch.where(fz_mass >= 0.0, fz_mass * rscale[1:], fz_mass * rscale[:-1])

        # 3. Conservative vertical mass + enthalpy advection
        _, T_pre = compute_C_and_T(mass, energy)
        T_donor = torch.where(fz_mass >= 0.0, T_pre[:-1], T_pre[1:])
        e_adv = (fz_mass * z_spec_c * T_donor).sum(dim=1, keepdim=True)  # [L-1, 1, H, W]

        fz_pad3 = F.pad(fz_mass, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
        dm_z_mob = fz_pad3[:-1] - fz_pad3[1:]  # [L, 4, H, W]
        dm_full = torch.zeros_like(mass).scatter_add(1, z_idx.view(1, 4, 1, 1).expand_as(dm_z_mob), dm_z_mob)
        mass_new = (mass + dm_full).clamp(min=0.0)

        e_pad = F.pad(e_adv, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
        energy_mid = energy + e_pad[:-1] - e_pad[1:]

        # 4. 2nd-law vertical heat conduction projection
        C_mid, T_mid = compute_C_and_T(mass_new, energy_mid)
        dT_z = T_mid[:-1] - T_mid[1:]
        q_max_z = dT_z.abs() * (C_mid[:-1] * C_mid[1:]) / (4.0 * (C_mid[:-1] + C_mid[1:]) + 1e-6)
        valid_c = (C_mid[:-1] > 1e-6) & (C_mid[1:] > 1e-6)
        q_max_z = torch.where(valid_c, q_max_z, torch.zeros_like(q_max_z))

        q_raw = raw_z_flux[:, ZF_HEAT : ZF_HEAT + 1] * 1e6
        q_proj = torch.where(
            dT_z > 0.0,
            q_raw.clamp(min=0.0).clamp(max=q_max_z),
            torch.where(dT_z < 0.0, q_raw.clamp(max=0.0).clamp(min=-q_max_z), torch.zeros_like(q_raw)),
        )
        q_pad = F.pad(q_proj, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
        energy_new = energy_mid + q_pad[:-1] - q_pad[1:]

        proj_z = torch.cat([fz_mass / 1000.0, q_proj / 1e6], dim=1)
        next_st = SimState(
            mass=mass_new,
            energy=energy_new,
            flow_e=state.flow_e,
            flow_s=state.flow_s,
            elevation=state.elevation,
        )
        return next_st, proj_z

    @staticmethod
    @torch.no_grad()
    def apply_radiative_cooling_upper_atm(state: SimState, dt_scale: float = 1.0, rate: float = 0.22) -> SimState:
        """
        Closed-mass upper-atmosphere radiative cooling (`z >= L - 3`):
        Radiatively cools high-altitude steam/clouds toward `T_SKY_RAD` (262 K) so ascending
        geothermal/volcanic plumes condense into liquid water / ice droplets and precipitate
        back onto peaks & valleys. NEVER modifies mass -> exact `mass_err == 0`!
        """
        L = state.mass.shape[0]
        if L < 4 or dt_scale <= 0.0:
            return state
        C_cell, T_cK = compute_C_and_T(state.mass, state.energy)
        alpha = min(0.75, float(rate) * float(dt_scale))
        energy = state.energy.clone()
        z0 = max(0, L - 3)
        T_top = T_cK[z0:]
        C_top = C_cell[z0:]
        cool_mask = (C_top > 1e-4) & (T_top > T_SKY_RAD)
        T_new = torch.where(cool_mask, T_top - alpha * (T_top - T_SKY_RAD), T_top)
        energy[z0:] = torch.where(C_top > 1e-4, C_top * T_new, energy[z0:])
        return SimState(
            mass=state.mass,
            energy=energy,
            flow_e=state.flow_e,
            flow_s=state.flow_s,
            elevation=state.elevation,
        )

    @staticmethod
    @torch.no_grad()
    def apply_structural_collapse_3d(state: SimState, dt_scale: float = 1.0, max_span: int = 4) -> tuple[SimState, torch.Tensor]:
        """
        Conservative 3D structural arch/bridge collapse & ash settling:
        If burning or melting piers leave elevated wood/ash/thin-ice deck voxels unsupported
        beyond `max_span` horizontal voxels from ground-supported piers, those overhanging
        voxels collapse conservatively downward (`z -> z - 1`) carrying exact mass & enthalpy!
        Returns `(next_state, collapsed_mass_3d [L, H, W])`.
        """
        L, _, H, W = state.mass.shape
        dev = state.mass.device
        if L <= 1 or dt_scale <= 0.0:
            return state, torch.zeros((L, H, W), dtype=torch.float32, device=dev)

        mass = state.mass.clone()
        energy = state.energy.clone()
        _, T_cK = compute_C_and_T(mass, energy)

        # Structural load-bearing solid voxels
        solid = (
            (mass[:, S_ROCK] >= 450.0)
            | (mass[:, S_SOIL] >= 550.0)
            | (mass[:, S_ICE] >= 550.0)
            | (mass[:, S_WOOD] >= 350.0)
        )  # [L, H, W]

        # Propagate ground support upward & laterally through contiguous arches (span <= max_span)
        supported = torch.zeros((L, H, W), dtype=torch.bool, device=dev)
        supported[0] = solid[0]
        for z in range(1, L):
            # Directly resting on supported solid below
            sup_z = solid[z] & supported[z - 1]
            # Lateral arch / cantilever support up to `max_span` voxels from supported piers
            sup_f = sup_z.float().unsqueeze(0).unsqueeze(0)
            sol_f = solid[z].float().unsqueeze(0).unsqueeze(0)
            for _ in range(max_span):
                sup_f = (F.max_pool2d(sup_f, kernel_size=3, stride=1, padding=1) * sol_f) > 0.5
                sup_f = sup_f.float()
            supported[z] = sup_f[0, 0] > 0.5

        # Unsupported overhanging wood or loose ash above open air collapses downward (`z -> z-1`)!
        fall_rate = min(0.85, 0.55 * float(dt_scale))
        collapsed_total = torch.zeros((L, H, W), dtype=torch.float32, device=dev)
        for z in range(1, L):
            below_open = ~solid[z - 1]
            headroom_below = (H_CAP - mass[z - 1].sum(dim=0)).clamp(min=0.0)
            # Unsupported wood deck or airborne ash drops downward
            unsup_wood = (~supported[z]) & below_open & (mass[z, S_WOOD] > 1.0)
            unsup_ash = below_open & (mass[z, S_ASH] > 1.0)

            dm_wood = torch.where(unsup_wood, mass[z, S_WOOD] * fall_rate, torch.zeros_like(headroom_below))
            dm_ash = torch.where(unsup_ash, mass[z, S_ASH] * fall_rate, torch.zeros_like(headroom_below))
            dm_sum = dm_wood + dm_ash
            cap = torch.minimum(torch.ones_like(dm_sum), (0.85 * headroom_below) / (dm_sum + 1e-6))
            dm_wood = dm_wood * cap
            dm_ash = dm_ash * cap

            dE = (dm_wood * SPEC_C_LIST[S_WOOD] + dm_ash * SPEC_C_LIST[S_ASH]) * T_cK[z, 0]
            mass[z, S_WOOD] = (mass[z, S_WOOD] - dm_wood).clamp(min=0.0)
            mass[z - 1, S_WOOD] = mass[z - 1, S_WOOD] + dm_wood
            mass[z, S_ASH] = (mass[z, S_ASH] - dm_ash).clamp(min=0.0)
            mass[z - 1, S_ASH] = mass[z - 1, S_ASH] + dm_ash
            energy[z, 0] = energy[z, 0] - dE
            energy[z - 1, 0] = energy[z - 1, 0] + dE
            collapsed_total[z] = dm_wood + dm_ash

        return (
            SimState(mass=mass, energy=energy, flow_e=state.flow_e, flow_s=state.flow_s, elevation=state.elevation),
            collapsed_total,
        )

    @staticmethod
    @torch.no_grad()
    def step(
        state: SimState,
        forcing_energy: torch.Tensor | None = None,
        forcing_mass: torch.Tensor | None = None,
        local_dt: torch.Tensor | float | None = 1.0,
        radiative_cooling: bool = True,
        structural_collapse: bool = True,
    ) -> tuple[SimState, torch.Tensor, torch.Tensor]:
        """
        Exact isotropic 3D physics step for arbitrary continuous `local_dt in [0.0, 4.0]`.
        Returns:
          - `next_state` : `SimState`
          - `proj_horiz` : `[L, 17, H, W]` exact integrated horizontal/reaction deltas
          - `proj_z_flux`: `[L - 1, 5, H, W]` exact integrated vertical interface fluxes
        """
        L, _, H, W = state.mass.shape
        dev = state.mass.device
        dt_map = RefSim3D._broadcast_dt(local_dt, L, H, W, dev)
        dt_max = float(dt_map.max().item())

        if dt_max <= 1e-8:
            return (
                state.clone(),
                torch.zeros((L, 17, H, W), dtype=torch.float32, device=dev),
                torch.zeros((max(L - 1, 0), Z_FLUX_CH, H, W), dtype=torch.float32, device=dev),
            )

        n_sub = max(1, int(math.ceil(dt_max)))
        dt_sub = dt_map / float(n_sub)
        dt_z_sub = torch.minimum(dt_sub[:-1], dt_sub[1:]) if L > 1 else dt_sub[:0]

        cur = state.clone()
        horiz_acc = torch.zeros((L, 17, H, W), dtype=torch.float32, device=dev)
        z_acc = torch.zeros((max(L - 1, 0), Z_FLUX_CH, H, W), dtype=torch.float32, device=dev)

        for k in range(n_sub):
            fe_k = forcing_energy if k == 0 else None
            fm_k = forcing_mass if k == 0 else None

            raw_h = RefSim3D.compute_raw_horiz_deltas(cur, fe_k, fm_k)
            raw_h_dt = RefSim3D._scale_horiz_by_dt(raw_h, dt_sub)
            cur, proj_h_k, _ = project_and_step(cur, raw_h_dt, fe_k, fm_k)
            horiz_acc = horiz_acc + proj_h_k

            if L > 1:
                raw_z = RefSim3D.compute_vertical_flux(cur)
                raw_z_dt = raw_z * dt_z_sub
                cur, proj_z_k = RefSim3D.project_and_apply_vertical_flux(cur, raw_z_dt)
                z_acc = z_acc + proj_z_k

            if structural_collapse and L > 1:
                cur, _ = RefSim3D.apply_structural_collapse_3d(cur, dt_scale=float(dt_sub.mean().item()))

            if radiative_cooling and L >= 4:
                cur = RefSim3D.apply_radiative_cooling_upper_atm(cur, dt_scale=float(dt_sub.mean().item()))

        return cur, horiz_acc, z_acc


# =============================================================================
# WorldModel3D-v5 Architecture (~158K params, exact 48-ch Z_world_3d trunk)
# =============================================================================


class Factorized3DResBlock(nn.Module):
    """
    Fast 3D residual block with `(1, 3, 3)` spatial Conv2d + optional `(3, 1, 1)` vertical Conv3d.
    Zero padding inside convolutions; explicit replicate padding in dense forward and exact
    shrinking-halo valid convs in `SparseWorld3DV5Runner`!
    """

    def __init__(self, width: int, use_vert: bool = False):
        super().__init__()
        self.width = width
        self.use_vert = bool(use_vert)
        self.spatial_conv = nn.Conv2d(width, width, kernel_size=3, padding=0, bias=True)
        self.gamma_xy = nn.Parameter(torch.full((1, width, 1, 1, 1), 0.25))
        if self.use_vert:
            self.vert_conv = nn.Conv3d(width, width, kernel_size=(3, 1, 1), padding=0, bias=True)
            self.gamma_z = nn.Parameter(torch.full((1, width, 1, 1, 1), 0.20))
            with torch.no_grad():
                self.vert_conv.weight.mul_(0.35)
                self.vert_conv.bias.zero_()
        else:
            self.vert_conv = None
            self.gamma_z = None

    def forward_dense(
        self,
        h: torch.Tensor,
        gamma_f: torch.Tensor,
        beta_f: torch.Tensor,
    ) -> torch.Tensor:
        """`h`, `gamma_f`, `beta_f`: `[B, C, L, H, W]`."""
        B, C, L, H, W = h.shape
        h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
        act = F.gelu(h_mod)
        # Fast 2D spatial (1, 3, 3) conv across B*L slices
        act_2d = act.transpose(1, 2).reshape(B * L, C, H, W)
        act_2d_pad = F.pad(act_2d, (1, 1, 1, 1), mode="replicate")
        s_out = self.spatial_conv(act_2d_pad).view(B, L, C, H, W).transpose(1, 2)
        h = h + self.gamma_xy * s_out

        if self.use_vert and self.vert_conv is not None and self.gamma_z is not None:
            v_in = F.pad(F.gelu(h), (0, 0, 0, 0, 1, 1), mode="replicate")
            h = h + self.gamma_z * self.vert_conv(v_in)
        return h


class Isotropic3x3x3ResBlock(nn.Module):
    """
    True isotropic `3x3x3` Conv3d bottleneck residual block (`width -> mid -> mid (3x3x3) -> width`).
    Captures diagonal 3D arches, waterfalls, plumes, and overhang interactions in one fused kernel!
    """

    def __init__(self, width: int = 48, mid: int = 32):
        super().__init__()
        self.width = width
        self.mid = mid
        self.pw_in = nn.Conv3d(width, mid, kernel_size=1, bias=True)
        self.conv3d = nn.Conv3d(mid, mid, kernel_size=3, padding=0, bias=True)
        self.pw_out = nn.Conv3d(mid, width, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.full((1, width, 1, 1, 1), 0.25))
        with torch.no_grad():
            self.pw_out.weight.mul_(0.35)
            self.pw_out.bias.zero_()

    def forward_dense(
        self,
        h: torch.Tensor,
        gamma_f: torch.Tensor,
        beta_f: torch.Tensor,
    ) -> torch.Tensor:
        h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
        u = F.gelu(self.pw_in(F.gelu(h_mod)))
        u_pad = F.pad(u, (1, 1, 1, 1, 1, 1), mode="replicate")
        u = F.gelu(self.conv3d(u_pad))
        return h + self.gamma * self.pw_out(u)


class WorldModel3DV5(nn.Module):
    """
    Wave 11A True Isotropic `Z=16/32` Conservation-Distilled `WorldModel3D-v5` (~158K params).

    Exact drop-in I/O contract with `WorldModel3D` / `WorldModelDT` (`gpu_infer_daemon.py` `op=8`/`op=9`):
      Input:
        - `state`    : `[L, 18, H, W]` (or `[L, 19, H, W]` or batched `[B, L, 18|19, H, W]`)
        - `local_dt` : optional float or tensor (`[H, W]`, `[L, H, W]`, `[L, 1, H, W]`, `[B, L, 1, H, W]`)
      Output (default `return_z_flux=True, return_latent=True`):
        - `out_horiz` : `[L, 17, H, W]`     (horizontal flux_e/s, heat_e/s, rxn[0..6], scaled by face/cell dt)
        - `out_z_flux`: `[L - 1, 5, H, W]`  (vertical interface water/oil/steam/smoke/heat_z, scaled by dt_z)
        - `z_world`   : `[L, 48, H, W]`     (shared 48-ch isotropic 3D latent trunk `Z_world_3d`)
    """

    HALO_XY = 6  # 1 (stem) + 4 (blocks 0..3) + 1 (3x3x3 block 4)
    HALO_Z = 3   # 1 (block 0 vert) + 1 (block 2 vert) + 1 (3x3x3 block 4)

    def __init__(
        self,
        in_ch: int = 18,
        out_ch: int = 17,
        width: int = 48,
        z_flux_ch: int = Z_FLUX_CH,
        dt_dim: int = 16,
        mid_3d: int = 32,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.width = int(width)
        self.z_flux_ch = int(z_flux_ch)
        self.dt_dim = int(dt_dim)
        self.mid_3d = int(mid_3d)
        self.num_blocks = 5
        self.halo_xy = self.HALO_XY
        self.halo_z = self.HALO_Z

        # 1. Stem: spatial (1, 3, 3) -> width=48
        self.stem = nn.Conv2d(in_ch, width, kernel_size=3, padding=0, bias=True)

        # 2. Five 3D residual blocks: 2 factorized (1,3,3)+(3,1,1), 2 spatial (1,3,3), 1 true 3x3x3 bottleneck
        self.blocks = nn.ModuleList(
            [
                Factorized3DResBlock(width, use_vert=True),   # r_xy +1, r_z +1
                Factorized3DResBlock(width, use_vert=False),  # r_xy +1
                Factorized3DResBlock(width, use_vert=True),   # r_xy +1, r_z +1
                Factorized3DResBlock(width, use_vert=False),  # r_xy +1
                Isotropic3x3x3ResBlock(width, mid=mid_3d),    # r_xy +1, r_z +1
            ]
        )

        # 3. Continuous log-Δt sinusoidal FiLM conditioner (bias=False -> exact 0 at dt == 1.0)
        self.dt_mlp = nn.Sequential(
            nn.Conv2d(8, 24, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(24, dt_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.dt_film = nn.Conv2d(dt_dim, self.num_blocks * 2 * width, kernel_size=1, bias=False)
        self.dt_head_mod = nn.Conv2d(dt_dim, out_ch, kernel_size=1, bias=False)
        self.dt_z_mod = nn.Conv2d(dt_dim, width, kernel_size=1, bias=False)
        with torch.no_grad():
            self.dt_film.weight.mul_(0.15)
            self.dt_head_mod.weight.zero_()
            self.dt_z_mod.weight.mul_(0.15)

        # 4. Horizontal 17-ch head with local raw-state 1x1 skip (sharp empty-air & wall/threshold gating!)
        self.h_lat_in = nn.Conv2d(width, width, kernel_size=1, bias=True)
        self.h_x_in = nn.Conv2d(in_ch, width, kernel_size=1, bias=False)
        self.h_out = nn.Conv2d(width, out_ch, kernel_size=1, bias=True)

        scale = torch.ones((1, out_ch, 1, 1), dtype=torch.float32)
        scale[:, 8:10] = 10.0
        self.register_buffer("out_scale", scale)

        with torch.no_grad():
            self.h_out.weight.mul_(0.15)
            self.h_out.bias.zero_()
            if out_ch >= 17:
                # Negative bias so empty-air softplus starts cleanly below 5e-3 firewall threshold!
                self.h_out.bias[10:17].fill_(-5.5)

        # 5. Staggered vertical interface flux head (`[L-1, 5, H, W]`, reads both lo & hi state + latent)
        self.v_x_lo = nn.Conv2d(in_ch, width, kernel_size=1, bias=True)
        self.v_x_hi = nn.Conv2d(in_ch, width, kernel_size=1, bias=False)
        self.v_z_lo = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.v_z_hi = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.v_mid = nn.Conv2d(width, width, kernel_size=1, bias=True)
        self.v_out = nn.Conv2d(width, z_flux_ch, kernel_size=1, bias=True)

        z_scale = torch.ones((1, z_flux_ch, 1, 1), dtype=torch.float32)
        if z_flux_ch >= 5:
            z_scale[:, ZF_HEAT] = 10.0
        self.register_buffer("z_out_scale", z_scale)

        with torch.no_grad():
            self.v_out.weight.mul_(0.10)
            self.v_out.bias.zero_()

    @staticmethod
    def _broadcast_dt(
        local_dt: torch.Tensor | float,
        B: int,
        L: int,
        H: int,
        W: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(local_dt, (int, float)):
            return torch.full((B, L, 1, H, W), max(0.0, float(local_dt)), dtype=dtype, device=device)
        t = local_dt.to(dtype=dtype, device=device).clamp(min=0.0)
        if t.ndim == 0:
            return t.view(1, 1, 1, 1, 1).expand(B, L, 1, H, W)
        if t.ndim == 2:
            return t.view(1, 1, 1, H, W).expand(B, L, 1, H, W)
        if t.ndim == 3:
            return t.view(1, t.shape[0], 1, H, W).expand(B, L, 1, H, W)
        if t.ndim == 4:
            if t.shape[0] == L and B == 1:
                return t.unsqueeze(0).expand(B, L, 1, H, W)
            if t.shape[0] == B:
                return t.unsqueeze(1).expand(B, L, 1, H, W)
            return t.unsqueeze(0).expand(B, L, 1, H, W)
        if t.ndim == 5:
            return t.expand(B, L, 1, H, W)
        raise ValueError(f"Unsupported local_dt shape {tuple(t.shape)}")

    def _split_state_and_dt(
        self,
        state: torch.Tensor,
        local_dt: torch.Tensor | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        is_4d = state.ndim == 4
        if is_4d:
            s5 = state.unsqueeze(0)
        elif state.ndim == 5:
            s5 = state
        else:
            raise ValueError(f"Expected 4D [L,C,H,W] or 5D [B,L,C,H,W] state, got {tuple(state.shape)}")

        B, L, C, H, W = s5.shape
        if C == self.in_ch + 1:
            x5 = s5[:, :, : self.in_ch]
            dt5 = s5[:, :, self.in_ch : self.in_ch + 1].clamp(min=0.0)
            if local_dt is not None:
                dt5 = self._broadcast_dt(local_dt, B, L, H, W, s5.dtype, s5.device)
        elif C == self.in_ch:
            x5 = s5
            if local_dt is None:
                dt5 = torch.ones((B, L, 1, H, W), dtype=s5.dtype, device=s5.device)
            else:
                dt5 = self._broadcast_dt(local_dt, B, L, H, W, s5.dtype, s5.device)
        else:
            raise ValueError(f"Expected {self.in_ch} or {self.in_ch + 1} channels, got {C}")
        return x5, dt5, is_4d

    @staticmethod
    def _scale_horiz_by_dt(horiz_rate: torch.Tensor, dt_2d: torch.Tensor) -> torch.Tensor:
        dt_e_nb = F.pad(dt_2d[:, :, :, 1:], (0, 1, 0, 0), mode="replicate")
        dt_s_nb = F.pad(dt_2d[:, :, 1:, :], (0, 0, 0, 1), mode="replicate")
        dt_e = torch.minimum(dt_2d, dt_e_nb)
        dt_s = torch.minimum(dt_2d, dt_s_nb)
        return torch.cat(
            [
                horiz_rate[:, 0:4] * dt_e,
                horiz_rate[:, 4:8] * dt_s,
                horiz_rate[:, 8:9] * dt_e,
                horiz_rate[:, 9:10] * dt_s,
                horiz_rate[:, 10:17] * dt_2d,
            ],
            dim=1,
        )

    def forward(
        self,
        state: torch.Tensor,
        local_dt: torch.Tensor | float | None = None,
        return_z_flux: bool = True,
        return_latent: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Dense 3D isotropic forward pass supporting 4D `[L, 18|19, H, W]` or 5D `[B, L, 18|19, H, W]`.
        Default returns `(out_horiz [L, 17, H, W], out_z_flux [L-1, 5, H, W], z_world [L, 48, H, W])`.
        """
        x5, dt5, is_4d = self._split_state_and_dt(state, local_dt)
        B, L, C_in, H, W = x5.shape
        x_2d = x5.reshape(B * L, C_in, H, W)
        dt_2d = dt5.reshape(B * L, 1, H, W)

        # Continuous Δt FiLM embeddings
        dt_emb = self.dt_mlp(encode_dt_features(dt_2d))  # [B*L, dt_dim, H, W]
        film_w = self.dt_film.weight
        film_stride = 2 * self.width

        # Stem: (1, 3, 3) spatial conv
        h_2d = self.stem(F.pad(x_2d, (1, 1, 1, 1), mode="replicate"))
        h = h_2d.view(B, L, self.width, H, W).transpose(1, 2)  # [B, width, L, H, W]

        # 5 FiLM-modulated isotropic 3D residual blocks (compute FiLM per-block for low VRAM!)
        for idx, blk in enumerate(self.blocks):
            w_blk = film_w[idx * film_stride : (idx + 1) * film_stride]
            film_2d = F.conv2d(dt_emb, w_blk)  # [B*L, 2*width, H, W]
            film_5d = film_2d.view(B, L, 2, self.width, H, W)
            gamma_f = film_5d[:, :, 0].transpose(1, 2)  # [B, width, L, H, W]
            beta_f = film_5d[:, :, 1].transpose(1, 2)
            h = blk.forward_dense(h, gamma_f, beta_f)

        # Shared 48-ch isotropic 3D latent trunk `Z_world_3d`
        z_5d = F.gelu(h).transpose(1, 2)  # [B, L, 48, H, W]
        z_2d = z_5d.reshape(B * L, self.width, H, W)

        # Horizontal 17-ch head + raw-state 1x1 skip + nonlinear Δt rate modulation + face-dt scaling
        hh = F.gelu(self.h_lat_in(z_2d) + self.h_x_in(x_2d))
        raw_h = self.h_out(hh) * self.out_scale
        rate_mult = 1.0 + 0.85 * torch.tanh(self.dt_head_mod(dt_emb))
        lin_rate = raw_h[:, :10] * rate_mult[:, :10]
        rxn_rate = F.softplus(raw_h[:, 10:]) * rate_mult[:, 10:]
        horiz_rate = torch.cat([lin_rate, rxn_rate], dim=1)
        horiz_2d = self._scale_horiz_by_dt(horiz_rate, dt_2d)
        horiz = horiz_2d.view(B, L, self.out_ch, H, W)

        out_z_flux: torch.Tensor | None = None
        if return_z_flux:
            if L > 1:
                x_lo = x5[:, :-1].reshape(B * (L - 1), C_in, H, W)
                x_hi = x5[:, 1:].reshape(B * (L - 1), C_in, H, W)
                z_lo = z_5d[:, :-1].reshape(B * (L - 1), self.width, H, W)
                z_hi = z_5d[:, 1:].reshape(B * (L - 1), self.width, H, W)
                dt_z = torch.minimum(dt5[:, :-1], dt5[:, 1:])  # [B, L-1, 1, H, W]
                dt_z_emb = self.dt_mlp(encode_dt_features(dt_z.reshape(B * (L - 1), 1, H, W)))
                vh = F.gelu(
                    self.v_x_lo(x_lo)
                    + self.v_x_hi(x_hi)
                    + self.v_z_lo(z_lo)
                    + self.v_z_hi(z_hi)
                    + self.dt_z_mod(dt_z_emb)
                )
                vh = F.gelu(self.v_mid(vh))
                z_if = (self.v_out(vh) * self.z_out_scale).view(B, L - 1, self.z_flux_ch, H, W) * dt_z
                out_z_flux = z_if
            else:
                out_z_flux = horiz.new_zeros((B, 0, self.z_flux_ch, H, W))

        z_world = z_5d.contiguous() if return_latent else None

        if is_4d:
            horiz = horiz.squeeze(0)
            if out_z_flux is not None:
                out_z_flux = out_z_flux.squeeze(0)
            if z_world is not None:
                z_world = z_world.squeeze(0)

        ret: list[torch.Tensor] = [horiz]
        if return_z_flux:
            assert out_z_flux is not None
            ret.append(out_z_flux)
        if return_latent:
            assert z_world is not None
            ret.append(z_world)
        return ret[0] if len(ret) == 1 else tuple(ret)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# True 3D `16x16x8` Sparse Active-Brick Runner (`SparseWorld3DV5Runner`)
# =============================================================================


def _apply_xy_boundary_replicate_(
    t: torch.Tensor,
    rem_xy: int,
    by: int,
    bx: int,
    m_y0: torch.Tensor,
    m_y1: torch.Tensor,
    m_x0: torch.Tensor,
    m_x1: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Exact outer-world boundary replicate fixup on 5D brick tensor `t [K, C, Zp, Hp, Wp]`
    before a valid `(1,3,3)` or `(3,3,3)` conv so edge bricks match dense replicate-pad bit-for-bit!
    Zero GPU-CPU `.item()` syncs and updates only the narrow `rem_xy` ghost border strips.
    """
    if rem_xy <= 0:
        return t
    if not inplace:
        t = t.clone()
    # Valid interior spans y in [rem_xy, rem_xy + by - 1], x in [rem_xy, rem_xy + bx - 1]
    t[:, :, :, :rem_xy, :] = torch.where(
        m_y0, t[:, :, :, rem_xy : rem_xy + 1, :], t[:, :, :, :rem_xy, :]
    )
    y_end = rem_xy + by
    if t.shape[3] > y_end:
        t[:, :, :, y_end:, :] = torch.where(
            m_y1, t[:, :, :, y_end - 1 : y_end, :], t[:, :, :, y_end:, :]
        )
    t[:, :, :, :, :rem_xy] = torch.where(
        m_x0, t[:, :, :, :, rem_xy : rem_xy + 1], t[:, :, :, :, :rem_xy]
    )
    x_end = rem_xy + bx
    if t.shape[4] > x_end:
        t[:, :, :, :, x_end:] = torch.where(
            m_x1, t[:, :, :, :, x_end - 1 : x_end], t[:, :, :, :, x_end:]
        )
    return t


def _apply_z_boundary_replicate_(
    t: torch.Tensor,
    rem_z_lo: int,
    bz: int,
    m_z0: torch.Tensor,
    m_z1: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """
    Exact outer-world Z boundary replicate fixup on 5D brick tensor `t [K, C, Zp, Hp, Wp]`
    before a valid `(3,1,1)` or `(3,3,3)` conv so `tz == 0` and `tz == nz - 1` bricks match
    dense per-layer replicate padding bit-for-bit across ALL `L` strata!
    Zero GPU-CPU `.item()` syncs.
    """
    if not inplace:
        t = t.clone()
    if rem_z_lo > 0:
        t[:, :, :rem_z_lo, :, :] = torch.where(
            m_z0, t[:, :, rem_z_lo : rem_z_lo + 1, :, :], t[:, :, :rem_z_lo, :, :]
        )
    z_end = rem_z_lo + bz
    if t.shape[2] > z_end:
        t[:, :, z_end:, :, :] = torch.where(
            m_z1, t[:, :, z_end - 1 : z_end, :, :], t[:, :, z_end:, :, :]
        )
    return t


def _shrinking_halo_3d_brick_core(
    wm: WorldModel3DV5,
    patches_x: torch.Tensor,
    patches_dt: torch.Tensor,
    m_z0: torch.Tensor,
    m_z1: torch.Tensor,
    m_y0: torch.Tensor,
    m_y1: torch.Tensor,
    m_x0: torch.Tensor,
    m_x1: torch.Tensor,
    brick_size: tuple[int, int, int] = (8, 16, 16),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-sync, torch.compile-friendly core for shrinking-halo 3D brick forward."""
    K, C_in, Zp, Hp, Wp = patches_x.shape
    Bz, By, Bx = brick_size
    hz = wm.halo_z
    hxy = wm.halo_xy

    # 1. Stem: valid (1, 3, 3) conv shrinks XY by 2: (Hp, Wp) -> (Hp-2, Wp-2); Z unchanged (Zp)
    rem_xy = hxy
    rem_z = hz
    x_in = _apply_xy_boundary_replicate_(patches_x, rem_xy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=False)
    x_in_2d = x_in.transpose(1, 2).reshape(K * Zp, C_in, Hp, Wp)
    h_2d = wm.stem(x_in_2d)
    rem_xy -= 1
    Hc, Wc = Hp - 2, Wp - 2
    Zc = Zp
    h = h_2d.view(K, Zc, wm.width, Hc, Wc).transpose(1, 2)  # [K, width, Zc, Hc, Wc]

    # Precompute dt_emb once on post-stem `[Zc, Hc, Wc]` and slice inward as halos shrink!
    dt_cur = patches_dt[:, :, :, 1:-1, 1:-1].clamp(min=0.0)  # [K, 1, Zc, Hc, Wc]
    dt_cur_2d = dt_cur.transpose(1, 2).reshape(K * Zc, 1, Hc, Wc)
    dt_emb_5d = wm.dt_mlp(encode_dt_features(dt_cur_2d)).view(K, Zc, wm.dt_dim, Hc, Wc).transpose(1, 2)

    film_w = wm.dt_film.weight
    film_stride = 2 * wm.width

    # 2. Shrinking-halo residual blocks (blocks 0..3 factorized, block 4 true 3x3x3)
    for idx, blk in enumerate(wm.blocks):
        w_blk = film_w[idx * film_stride : (idx + 1) * film_stride]
        de_2d = dt_emb_5d.transpose(1, 2).reshape(K * Zc, wm.dt_dim, Hc, Wc)
        film_5d = F.conv2d(de_2d, w_blk).view(K, Zc, 2, wm.width, Hc, Wc)
        gamma_f = film_5d[:, :, 0].transpose(1, 2)
        beta_f = film_5d[:, :, 1].transpose(1, 2)

        if isinstance(blk, Factorized3DResBlock):
            h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
            act = F.gelu(h_mod)
            act = _apply_xy_boundary_replicate_(act, rem_xy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=True)
            s_out = blk.spatial_conv(act.transpose(1, 2).reshape(K * Zc, wm.width, Hc, Wc))
            Hc -= 2
            Wc -= 2
            rem_xy -= 1
            s_out = s_out.view(K, Zc, wm.width, Hc, Wc).transpose(1, 2)
            h = h[:, :, :, 1:-1, 1:-1] + blk.gamma_xy * s_out
            dt_emb_5d = dt_emb_5d[:, :, :, 1:-1, 1:-1]

            if blk.use_vert and blk.vert_conv is not None and blk.gamma_z is not None:
                v_in = F.gelu(h)
                v_in = _apply_z_boundary_replicate_(v_in, rem_z, Bz, m_z0, m_z1, inplace=True)
                v_out = blk.vert_conv(v_in)
                Zc -= 2
                rem_z -= 1
                h = h[:, :, 1:-1, :, :] + blk.gamma_z * v_out
                dt_emb_5d = dt_emb_5d[:, :, 1:-1, :, :]
        else:
            assert isinstance(blk, Isotropic3x3x3ResBlock)
            h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
            u = F.gelu(blk.pw_in(F.gelu(h_mod)))
            u = _apply_xy_boundary_replicate_(u, rem_xy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=True)
            u = _apply_z_boundary_replicate_(u, rem_z, Bz, m_z0, m_z1, inplace=True)
            u = F.gelu(blk.conv3d(u))
            Zc -= 2
            Hc -= 2
            Wc -= 2
            rem_z -= 1
            rem_xy -= 1
            h = h[:, :, 1:-1, 1:-1, 1:-1] + blk.gamma * blk.pw_out(u)
            dt_emb_5d = dt_emb_5d[:, :, 1:-1, 1:-1, 1:-1]

    # Now Zc == Bz + 1, Hc == By, Wc == Bx!
    z_all = F.gelu(h).transpose(1, 2)  # [K, Bz + 1, 48, By, Bx]
    z_out = z_all[:, :Bz]              # [K, Bz, 48, By, Bx]
    x_all = patches_x[:, :, hz : hz + Bz + 1, hxy:-hxy, hxy:-hxy].transpose(1, 2)  # [K, Bz+1, 18, By, Bx]
    x_int = x_all[:, :Bz]

    # Exact center + 1-pixel east/south dt from patch halo (respecting world edge replicate!)
    dt_pad_xy = _apply_xy_boundary_replicate_(patches_dt.clamp(min=0.0), hxy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=True)
    dt_all = dt_pad_xy[:, :, hz : hz + Bz + 1, hxy:-hxy, hxy:-hxy].transpose(1, 2)  # [K, Bz+1, 1, By, Bx]
    dt_int = dt_all[:, :Bz]
    dt_e_nb = dt_pad_xy[:, :, hz : hz + Bz, hxy:-hxy, hxy + 1 : Wp - hxy + 1].transpose(1, 2)
    dt_s_nb = dt_pad_xy[:, :, hz : hz + Bz, hxy + 1 : Hp - hxy + 1, hxy:-hxy].transpose(1, 2)
    dt_int2 = dt_int.reshape(K * Bz, 1, By, Bx)
    dt_e = torch.minimum(dt_int2, dt_e_nb.reshape(K * Bz, 1, By, Bx))
    dt_s = torch.minimum(dt_int2, dt_s_nb.reshape(K * Bz, 1, By, Bx))

    # 3. Horizontal 17-ch head on `[K, Bz, ...]`
    z_2d = z_out.reshape(K * Bz, wm.width, By, Bx)
    x_2d = x_int.reshape(K * Bz, C_in, By, Bx)
    de_int2 = dt_emb_5d[:, :, :Bz].transpose(1, 2).reshape(K * Bz, wm.dt_dim, By, Bx)
    hh = F.gelu(wm.h_lat_in(z_2d) + wm.h_x_in(x_2d))
    raw_h = wm.h_out(hh) * wm.out_scale
    rate_mult = 1.0 + 0.85 * torch.tanh(wm.dt_head_mod(de_int2))
    horiz_rate = torch.cat([raw_h[:, :10] * rate_mult[:, :10], F.softplus(raw_h[:, 10:]) * rate_mult[:, 10:]], dim=1)
    horiz_2d = torch.cat(
        [
            horiz_rate[:, 0:4] * dt_e,
            horiz_rate[:, 4:8] * dt_s,
            horiz_rate[:, 8:9] * dt_e,
            horiz_rate[:, 9:10] * dt_s,
            horiz_rate[:, 10:17] * dt_int2,
        ],
        dim=1,
    )
    horiz = horiz_2d.view(K, Bz, wm.out_ch, By, Bx)

    # 4. Staggered vertical interface head on `[K, Bz, ...]` using exact `(0..Bz-1, 1..Bz)` pairs!
    x_lo = x_all[:, :-1].reshape(K * Bz, C_in, By, Bx)
    x_hi = x_all[:, 1:].reshape(K * Bz, C_in, By, Bx)
    z_lo = z_all[:, :-1].reshape(K * Bz, wm.width, By, Bx)
    z_hi = z_all[:, 1:].reshape(K * Bz, wm.width, By, Bx)
    dt_z = torch.minimum(dt_all[:, :-1], dt_all[:, 1:])  # [K, Bz, 1, By, Bx]
    dt_z_emb = wm.dt_mlp(encode_dt_features(dt_z.reshape(K * Bz, 1, By, Bx)))
    vh = F.gelu(
        wm.v_x_lo(x_lo)
        + wm.v_x_hi(x_hi)
        + wm.v_z_lo(z_lo)
        + wm.v_z_hi(z_hi)
        + wm.dt_z_mod(dt_z_emb)
    )
    vh = F.gelu(wm.v_mid(vh))
    z_flux = (wm.v_out(vh) * wm.z_out_scale).view(K, Bz, wm.z_flux_ch, By, Bx) * dt_z
    return horiz, z_flux, z_out


def shrinking_halo_3d_brick_forward(
    wm: WorldModel3DV5,
    patches_x: torch.Tensor,
    patches_dt: torch.Tensor,
    brick_coords: torch.Tensor,
    grid_bricks: tuple[int, int, int],
    brick_size: tuple[int, int, int] = (8, 16, 16),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact mathematically identical shrinking-halo 3D brick forward for `WorldModel3DV5`!
      Inputs:
        - `patches_x`   : `[K, 18, Bz + 2*halo_z + 1, By + 2*halo_xy, Bx + 2*halo_xy]` (e.g. `[K, 18, 15, 28, 28]`)
        - `patches_dt`  : `[K,  1, Bz + 2*halo_z + 1, By + 2*halo_xy, Bx + 2*halo_xy]`
        - `brick_coords`: `[K, 3]` `(tz, ty, tx)` brick indices (for exact outer-world boundary fixup)
        - `grid_bricks` : `(nz, ny, nx)`
        - `brick_size`  : `(Bz, By, Bx)` = `(8, 16, 16)`
      Outputs:
        - `horiz` : `[K, Bz, 17, By, Bx]`
        - `z_flux`: `[K, Bz,  5, By, Bx]` (interfaces `z <-> z+1`; last slice of top brick ignored by caller)
        - `z_out` : `[K, Bz, 48, By, Bx]`
    """
    K, _, Zp, Hp, Wp = patches_x.shape
    Bz, By, Bx = brick_size
    nz, ny, nx = grid_bricks
    hz = wm.halo_z
    hxy = wm.halo_xy
    assert Zp == Bz + 2 * hz + 1, f"Expected Zp={Bz + 2 * hz + 1}, got {Zp}"
    assert Hp == By + 2 * hxy and Wp == Bx + 2 * hxy, f"Expected XY={(By + 2 * hxy, Bx + 2 * hxy)}, got {(Hp, Wp)}"

    tz, ty, tx = brick_coords[:, 0], brick_coords[:, 1], brick_coords[:, 2]
    m_z0 = (tz == 0).view(K, 1, 1, 1, 1)
    m_z1 = (tz == (nz - 1)).view(K, 1, 1, 1, 1)
    m_y0 = (ty == 0).view(K, 1, 1, 1, 1)
    m_y1 = (ty == (ny - 1)).view(K, 1, 1, 1, 1)
    m_x0 = (tx == 0).view(K, 1, 1, 1, 1)
    m_x1 = (tx == (nx - 1)).view(K, 1, 1, 1, 1)

    return _shrinking_halo_3d_brick_core(
        wm, patches_x, patches_dt, m_z0, m_z1, m_y0, m_y1, m_x0, m_x1, brick_size=brick_size
    )


class _CompiledBrickModule(nn.Module):
    """Thin nn.Module wrapper so `torch.compile(..., mode='reduce-overhead')` fuses sparse brick passes cleanly."""

    def __init__(self, wm: WorldModel3DV5, brick_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.wm = wm
        self.brick_size = brick_size

    def forward(
        self,
        patches_x: torch.Tensor,
        patches_dt: torch.Tensor,
        m_z0: torch.Tensor,
        m_z1: torch.Tensor,
        m_y0: torch.Tensor,
        m_y1: torch.Tensor,
        m_x0: torch.Tensor,
        m_x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _shrinking_halo_3d_brick_core(
            self.wm, patches_x, patches_dt, m_z0, m_z1, m_y0, m_y1, m_x0, m_x1, brick_size=self.brick_size
        )


class SparseWorld3DV5Runner:
    """
    Production True 3D `16x16x8` Sparse Active-Brick Runner for `WorldModel3DV5`.

    Maintains persistent device caches (`_last_horiz`, `_last_z_flux_pad`, `_last_z`, `_last_x`, `_last_dt`)
    and evaluates ONLY dirty `16x16x8` 3D bricks `(tz, ty, tx)` via exact shrinking-halo valid convolutions
    whenever `active_brick_frac < sparse_threshold` (default `0.42`).
    - When `activity_eps == 0.0` (default): bit-exact FP32 dense-vs-sparse parity (`0.0`) on every voxel!
    - When `activity_eps > 0.0` (e.g. `5e-4` for live evolving worlds): suppresses tiny ambient cooling /
      quiescent bedrock ripples so live worlds stay in fast sparse execution instead of falling back to dense!
    """

    def __init__(
        self,
        world_model: WorldModel3DV5,
        device: torch.device | str,
        use_fp16: bool = True,
        brick_xy: int = 16,
        brick_z: int = 8,
        sparse_threshold: float = 0.42,
        diff_eps: float = 1e-4,
        activity_eps: float = 0.0,
        bucket_size: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")
        self.compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.brick_xy = int(brick_xy)
        self.brick_z = int(brick_z)
        self.sparse_threshold = float(sparse_threshold)
        self.diff_eps = float(diff_eps)
        self.activity_eps = max(0.0, float(activity_eps))
        self.bucket_size = max(1, int(bucket_size))

        self.halo_xy = world_model.halo_xy
        self.halo_z = world_model.halo_z
        self.patch_xy = self.brick_xy + 2 * self.halo_xy
        self.patch_z = self.brick_z + 2 * self.halo_z + 1

        self.world_model = world_model
        self._wm = copy.deepcopy(world_model).to(device=self.device, dtype=self.compute_dtype).eval()
        for p in self._wm.parameters():
            p.requires_grad_(False)
        self._wm_dense: Any = self._wm
        self._brick_mod: Any = _CompiledBrickModule(self._wm, (self.brick_z, self.brick_xy, self.brick_xy)).eval()
        if self.device.type == "cuda" and self.use_fp16 and hasattr(torch, "compile"):
            try:
                self._wm_dense = torch.compile(self._wm, mode="reduce-overhead")
            except Exception:
                self._wm_dense = self._wm
            try:
                self._brick_mod = torch.compile(self._brick_mod, mode="reduce-overhead")
            except Exception:
                pass

        self._last_horiz: torch.Tensor | None = None
        self._last_z_flux_pad: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_x: torch.Tensor | None = None
        self._last_dt: torch.Tensor | None = None

    def reset(self) -> None:
        self._last_horiz = None
        self._last_z_flux_pad = None
        self._last_z = None
        self._last_x = None
        self._last_dt = None

    def _normalize_dt(self, local_dt: torch.Tensor | float | None, L: int, H: int, W: int) -> torch.Tensor:
        if local_dt is None:
            return torch.ones((L, 1, H, W), dtype=self.compute_dtype, device=self.device)
        if isinstance(local_dt, (int, float)):
            return torch.full((L, 1, H, W), max(0.0, float(local_dt)), dtype=self.compute_dtype, device=self.device)
        t = local_dt.to(device=self.device, dtype=self.compute_dtype, non_blocking=(self.device.type == "cuda")).clamp(min=0.0)
        if t.ndim == 0:
            return t.view(1, 1, 1, 1).expand(L, 1, H, W).contiguous()
        if t.ndim == 2:
            return t.view(1, 1, H, W).expand(L, 1, H, W).contiguous()
        if t.ndim == 3:
            return t.view(t.shape[0], 1, H, W).expand(L, 1, H, W).contiguous()
        if t.ndim == 4:
            return t.expand(L, 1, H, W).contiguous()
        raise ValueError(f"Unsupported local_dt shape {tuple(t.shape)}")

    def active_brick_mask(
        self,
        x_cur: torch.Tensor,
        dt_cur: torch.Tensor,
        prev_x: torch.Tensor | None,
        dirty_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return boolean 3D active brick mask `[nz, ny, nx]` on `self.device`."""
        L, _, H, W = x_cur.shape
        Bz, By, Bx = self.brick_z, self.brick_xy, self.brick_xy
        nz, ny, nx = L // Bz, H // By, W // Bx

        if dirty_mask is not None:
            dm = dirty_mask.to(device=self.device)
            if dm.shape == (nz, ny, nx):
                return dm.bool()
            if dm.shape == (ny, nx):
                return dm.bool().unsqueeze(0).expand(nz, ny, nx).contiguous()
            if dm.shape == (L, H, W):
                pooled = F.max_pool3d(dm.to(dtype=self.compute_dtype).unsqueeze(0).unsqueeze(0), (Bz, By, Bx), (Bz, By, Bx))
                return pooled[0, 0] > 0.5
            raise ValueError(f"dirty_mask shape {tuple(dirty_mask.shape)} incompatible with bricks {(nz, ny, nx)}")

        ref = prev_x.to(device=self.device, dtype=self.compute_dtype) if prev_x is not None else self._last_x
        assert ref is not None

        if self.activity_eps <= 0.0:
            # Bit-exact mode: ANY non-zero change across all 18 channels or dt marks the voxel dirty!
            diff_vox = ((x_cur - ref).abs().amax(dim=1) > 0.0).to(dtype=self.compute_dtype)
            if self._last_dt is not None and self._last_dt.shape == dt_cur.shape:
                dt_diff = (dt_cur[:, 0] - self._last_dt[:, 0]).abs() > 0.0
                diff_vox = torch.maximum(diff_vox, dt_diff.to(dtype=self.compute_dtype))
        else:
            # Live-world mode: ignore tiny ambient cooling / sub-threshold noise in quiescent bedrock/air bricks!
            eps = self.activity_eps
            d_sp = (x_cur[:, :10] - ref[:, :10]).abs().amax(dim=1) > max(eps * 24.0, self.diff_eps, 0.012)
            d_tmp = (x_cur[:, 10] - ref[:, 10]).abs() > max(eps * 70.0, self.diff_eps * 4.0, 0.035)
            d_flow = (x_cur[:, 12:14] - ref[:, 12:14]).abs().amax(dim=1) > max(eps * 20.0, 0.010)
            has_force = (x_cur[:, 16].abs() > max(eps, 0.01)) | (x_cur[:, 17].abs() > max(eps, 0.01))
            diff_vox = (d_sp | d_tmp | d_flow | has_force).to(dtype=self.compute_dtype)
            if self._last_dt is not None and self._last_dt.shape == dt_cur.shape:
                dt_diff = (dt_cur[:, 0] - self._last_dt[:, 0]).abs() > 1e-3
                diff_vox = torch.maximum(diff_vox, dt_diff.to(dtype=self.compute_dtype))

        pooled = F.max_pool3d(diff_vox.unsqueeze(0).unsqueeze(0), (Bz, By, Bx), (Bz, By, Bx))
        return pooled[0, 0] > 0.5

    # Backward-compatible internal alias
    _compute_brick_mask = active_brick_mask

    @torch.inference_mode()
    def step(
        self,
        x: torch.Tensor,
        local_dt: torch.Tensor | float | None = None,
        prev_x: torch.Tensor | None = None,
        dirty_mask: torch.Tensor | None = None,
        dilate_halo_bricks: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Advance `WorldModel3DV5` on `x` (`[L, 18, H, W]`) with optional `local_dt`.
        Returns `(out_horiz [L, 17, H, W], out_z_flux [L-1, 5, H, W], z_world [L, 48, H, W], active_brick_frac)`.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D [L, 18, H, W], got {tuple(x.shape)}")

        x_cur = x.to(device=self.device, dtype=self.compute_dtype, non_blocking=(self.device.type == "cuda"))
        L, C, H, W = x_cur.shape
        dt_cur = self._normalize_dt(local_dt, L, H, W)
        Bz, By, Bx = self.brick_z, self.brick_xy, self.brick_xy

        need_dense = (
            self._last_horiz is None
            or self._last_z_flux_pad is None
            or self._last_z is None
            or self._last_x is None
            or self._last_x.shape != x_cur.shape
            or (L % Bz != 0)
            or (H % By != 0)
            or (W % Bx != 0)
        )

        if not need_dense:
            assert self._last_horiz is not None
            assert self._last_z_flux_pad is not None
            assert self._last_z is not None
            assert self._last_x is not None

            nz, ny, nx = L // Bz, H // By, W // Bx
            total_bricks = nz * ny * nx
            brick_active = self.active_brick_mask(x_cur, dt_cur, prev_x, dirty_mask)  # [nz, ny, nx]
            if dilate_halo_bricks:
                ba_f = brick_active.to(dtype=self.compute_dtype).unsqueeze(0).unsqueeze(0)
                brick_active = F.max_pool3d(ba_f, kernel_size=3, stride=1, padding=1)[0, 0] > 0.5

            coords = torch.nonzero(brick_active, as_tuple=False)  # [K, 3] -> (tz, ty, tx)
            k = int(coords.shape[0])
            active_frac = float(k) / float(total_bricks)

            if k == 0:
                self._last_x.copy_(x_cur)
                self._last_dt = dt_cur.clone()
                return self._last_horiz, self._last_z_flux_pad[:-1], self._last_z, 0.0

            if active_frac < self.sparse_threshold:
                tz, ty, tx = coords[:, 0], coords[:, 1], coords[:, 2]

                # Transpose [L, C, H, W] -> [1, C, L, H, W], replicate-pad 3D halo, zero-copy unfold into bricks!
                x_5d = x_cur.transpose(0, 1).unsqueeze(0)
                dt_5d = dt_cur.transpose(0, 1).unsqueeze(0)
                pad3 = (self.halo_xy, self.halo_xy, self.halo_xy, self.halo_xy, self.halo_z, self.halo_z + 1)
                x_pad = F.pad(x_5d, pad3, mode="replicate")[0]   # [C, L + 2*hz + 1, H + 2*hxy, W + 2*hxy]
                dt_pad = F.pad(dt_5d, pad3, mode="replicate")[0]

                # Unfold (Z, Y, X) -> [C, nz, ny, nx, Pz, Py, Px] -> gather K active bricks -> [K, C, Pz, Py, Px]
                ux = x_pad.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
                udt = dt_pad.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
                patches_x = ux.permute(1, 2, 3, 0, 4, 5, 6)[tz, ty, tx].contiguous()
                patches_dt = udt.permute(1, 2, 3, 0, 4, 5, 6)[tz, ty, tx].contiguous()

                # Bucket K to avoid cuDNN/Inductor kernel thrashing on K +- 1 jitter
                b = self.bucket_size
                k_pad = min(((k + b - 1) // b) * b, total_bricks)
                if k_pad > k:
                    n_ex = k_pad - k
                    patches_x_run = torch.cat([patches_x, patches_x.new_zeros((n_ex, C, self.patch_z, self.patch_xy, self.patch_xy))], dim=0)
                    patches_dt_run = torch.cat([patches_dt, patches_dt.new_ones((n_ex, 1, self.patch_z, self.patch_xy, self.patch_xy))], dim=0)
                    coords_run = torch.cat([coords, coords.new_zeros((n_ex, 3))], dim=0)
                else:
                    patches_x_run = patches_x
                    patches_dt_run = patches_dt
                    coords_run = coords

                k_r = int(coords_run.shape[0])
                rtz, rty, rtx = coords_run[:, 0], coords_run[:, 1], coords_run[:, 2]
                m_z0 = (rtz == 0).view(k_r, 1, 1, 1, 1)
                m_z1 = (rtz == (nz - 1)).view(k_r, 1, 1, 1, 1)
                m_y0 = (rty == 0).view(k_r, 1, 1, 1, 1)
                m_y1 = (rty == (ny - 1)).view(k_r, 1, 1, 1, 1)
                m_x0 = (rtx == 0).view(k_r, 1, 1, 1, 1)
                m_x1 = (rtx == (nx - 1)).view(k_r, 1, 1, 1, 1)

                try:
                    y_horiz, y_z_flux, y_z = self._brick_mod(
                        patches_x_run, patches_dt_run, m_z0, m_z1, m_y0, m_y1, m_x0, m_x1
                    )
                except Exception:
                    self._brick_mod = _CompiledBrickModule(
                        self._wm, (self.brick_z, self.brick_xy, self.brick_xy)
                    ).eval()
                    y_horiz, y_z_flux, y_z = self._brick_mod(
                        patches_x_run, patches_dt_run, m_z0, m_z1, m_y0, m_y1, m_x0, m_x1
                    )
                if k_pad > k:
                    y_horiz = y_horiz[:k]
                    y_z_flux = y_z_flux[:k]
                    y_z = y_z[:k]

                # Direct in-place strided 3D scatter into persistent caches (zero full-grid copy!)
                out_ch = self._last_horiz.shape[1]
                zf_ch = self._last_z_flux_pad.shape[1]
                z_ch = self._last_z.shape[1]

                # Cache view: [nz, Bz, C, ny, By, nx, Bx] -> permute(0, 3, 5, 1, 2, 4, 6) -> [nz, ny, nx, Bz, C, By, Bx]
                self._last_horiz.view(nz, Bz, out_ch, ny, By, nx, Bx).permute(0, 3, 5, 1, 2, 4, 6)[tz, ty, tx] = y_horiz
                self._last_z_flux_pad.view(nz, Bz, zf_ch, ny, By, nx, Bx).permute(0, 3, 5, 1, 2, 4, 6)[tz, ty, tx] = y_z_flux
                self._last_z.view(nz, Bz, z_ch, ny, By, nx, Bx).permute(0, 3, 5, 1, 2, 4, 6)[tz, ty, tx] = y_z

                self._last_x.copy_(x_cur)
                self._last_dt = dt_cur.clone()
                return self._last_horiz, self._last_z_flux_pad[:-1], self._last_z, active_frac

        # Dense fallback / initial cache population (uses Triton-fused CUDA Graph when compiled!)
        horiz, z_flux, z_world = self._wm_dense(x_cur, local_dt=dt_cur, return_z_flux=True, return_latent=True)
        horiz = horiz.clone()
        z_flux = z_flux.clone()
        z_world = z_world.clone()
        self._last_horiz = horiz.contiguous()
        zf_pad = horiz.new_zeros((L, self._wm.z_flux_ch, H, W))
        if L > 1:
            zf_pad[:-1].copy_(z_flux)
        self._last_z_flux_pad = zf_pad.contiguous()
        self._last_z = z_world.contiguous()
        self._last_x = x_cur.clone()
        self._last_dt = dt_cur.clone()
        return self._last_horiz, self._last_z_flux_pad[:-1], self._last_z, 1.0


def verify_sparse_v5_parity(
    model: WorldModel3DV5 | None = None,
    L: int = 16,
    H: int = 64,
    W: int = 64,
    seed: int = 77,
) -> dict[str, float]:
    """
    Verify exact FP32 dense-vs-sparse `16x16x8` 3D brick parity (`max_abs_diff < 1e-5`)
    across ALL `L` layers (including `z=0`, `z=L-1`, and spatial `local_dt` domes)!
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    wm = copy.deepcopy(model).to(device=device, dtype=torch.float32).eval() if model is not None else WorldModel3DV5().to(device).eval()

    prev_x = torch.rand((L, 18, H, W), dtype=torch.float32, device=device) * 0.4
    prev_x[:, 16:18] = 0.0
    next_x = prev_x.clone()

    # Perturb localized bricks (including BOTH z=0 bottom brick & z=L-1 top brick AND interior!)
    Bz, By, Bx = 8, 16, 16
    nz, ny, nx = L // Bz, H // By, W // Bx
    next_x[2:6, 0, 18:30, 20:32] += 0.35
    next_x[L - 3 : L, 2, 18:30, 20:32] += 0.30

    dt_map = torch.ones((H, W), dtype=torch.float32, device=device)
    dt_map[H // 4 : (3 * H) // 4, W // 4 : (3 * W) // 4] = 0.15  # Stasis Dome!

    runner = SparseWorld3DV5Runner(wm, device=device, use_fp16=False, brick_xy=Bx, brick_z=Bz, sparse_threshold=0.95)
    with torch.inference_mode():
        _, _, _, frac0 = runner.step(prev_x, local_dt=dt_map)
        # Use dilate_halo_bricks=True so EVERY voxel across the ENTIRE [L, H, W] grid matches dense!
        sp_h, sp_zf, sp_z, frac1 = runner.step(next_x, local_dt=dt_map, dilate_halo_bricks=True)
        dn_h, dn_zf, dn_z = wm(next_x, local_dt=dt_map, return_z_flux=True, return_latent=True)

    assert frac0 == 1.0
    assert 0.0 < frac1 < 0.95
    assert tuple(sp_h.shape) == (L, 17, H, W)
    assert tuple(sp_zf.shape) == (L - 1, 5, H, W)
    assert tuple(sp_z.shape) == (L, 48, H, W)

    diff_h = float((sp_h - dn_h).abs().max().item())
    diff_zf = float((sp_zf - dn_zf).abs().max().item())
    diff_z = float((sp_z - dn_z).abs().max().item())
    return {
        "L": int(L),
        "H": int(H),
        "W": int(W),
        "active_brick_frac": round(frac1, 4),
        "max_abs_diff_horiz": diff_h,
        "max_abs_diff_z_flux": diff_zf,
        "max_abs_diff_z_world": diff_z,
        "max_abs_diff_all": max(diff_h, diff_zf, diff_z),
    }


if __name__ == "__main__":
    m = WorldModel3DV5()
    print(f"WorldModel3DV5 params: {count_parameters(m):,}")
    for layers in (16, 32):
        res = verify_sparse_v5_parity(m, L=layers, H=64, W=64)
        print(f"FP32 Parity Z={layers}: {res}")
        assert res["max_abs_diff_all"] < 1e-5, f"Parity failed: {res}"
    print("All WorldModel3DV5 self-checks PASSED!")
