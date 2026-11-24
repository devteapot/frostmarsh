"""
Frostmarsh Wave 26A (Vertical Scaling GPU/ML):
  - 3D Volumetric Altitude-Varying World Laws `theta_world[6, Z, H, W]` (0 extra params, 0 retraining!)
  - 3-Strata Stacked Vertical World Generator:
      * Stratum 1 (z < 0.28*L): Abyssal Magma Caverns (bedrock base, vaulted chambers, lava pools & steam)
      * Stratum 2 (0.28*L..0.64*L): Surface Gorge & Viaduct (alpine river canyon, arched bridge, pine slopes)
      * Stratum 3 (z > 0.64*L): Floating Sky-Islands & Stratosphere (suspended islands, 25-50m waterfalls, cloud deck)
  - Vertical `HALO_Z = 3` Slab Sharding Parity Proof (`64x64x64` = 4x16 slabs & `128x48x48` = 4x32 slabs,
    FP32 `<= 1e-6` rel & FP64 `<= 1e-13` exact algebraic identity across `horiz`, `z_flux`, `Z_world_3d`)
  - Live Spark GB10 Vertical GPU Scaling Benchmark (`Z=16` 1.05M -> `Z=32` 2.10M -> `Z=64` 4.19M -> `Z=128` 8.39M)
    with trained `world_model_3d_v8.pt`, `max_bricks_per_batch=192`, `<= 3.5 GB` VRAM, + Vertical `HALO_Z = 3` Bandwidth
  - Production `Z=64` (`L=64, H=256, W=256`) Daemon Readiness Verification (`op=11`/`op=9`, `op=10` ActuatorV3, `op=6` Acoustics)
  - Crisp 3-Panel Dark-Theme Visualization (`models/vertical_scaling_viz.png`).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    T_AMB,
    T_MAX,
    T_MIN,
    SimState,
    encode_state,
    get_consts,
)
from world_model_3d_v5 import _layer_elevation, _smooth_noise_2d  # noqa: E402
from world_model_3d_v8 import (  # noqa: E402
    DEFAULT_WORLD_LAWS,
    LAW_AMBIENT_TEMP_SHIFT,
    LAW_DIM,
    LAW_GRAVITY,
    LAW_NAMES,
    LAW_PRECIPITATION,
    LAW_THERMAL_COND,
    LAW_VISCOSITY,
    LAW_VOLATILITY,
    WORLD_LAW_PRESETS,
    SparseWorld3DV8Runner,
    WorldModel3DV8,
    count_parameters,
    encode_law_features,
)

HALO_Z = 3

# Stratum altitude laws: [gravity_g, ambient_temp_shift, volatility, viscosity, precipitation, thermal_cond]
STRATUM_LAWS_ABYSSAL: tuple[float, float, float, float, float, float] = (
    1.15,
    4200.0,
    1.90,
    1.25,
    0.40,
    1.45,
)
STRATUM_LAWS_SURFACE: tuple[float, float, float, float, float, float] = DEFAULT_WORLD_LAWS
STRATUM_LAWS_SKY: tuple[float, float, float, float, float, float] = (
    0.35,
    -4500.0,
    0.70,
    0.85,
    2.20,
    1.15,
)


# =============================================================================
# 1. 3-Strata Vertical World & 3D Altitude-Varying World-Law Field (`[6, L, H, W]`)
# =============================================================================


def make_vertical_strata_law_field(
    L: int,
    H: int,
    W: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    blend_frac: float = 0.08,
) -> torch.Tensor:
    """
    Build a continuous 3D volumetric altitude-varying World-Law field `[6, L, H, W]`
    with `C^1` smooth cosine transitions across Frostmarsh's 3 stacked vertical strata:
      - Stratum 1 (`z < 0.28*L`): Abyssal Magma Caverns (`g=1.15`, `dT=+4200`, `vol=1.9`)
      - Stratum 2 (`0.28*L..0.64*L`): Surface Gorge & Viaduct (`g=1.0`, `dT=0`, canonical)
      - Stratum 3 (`z > 0.64*L`): Floating Sky-Islands & Stratosphere (`g=0.35`, `dT=-4500`, `precip=2.2`)
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    z_norm = (torch.arange(L, dtype=dtype, device=dev) + 0.5) / float(max(L, 1))  # [L]

    bw = max(1e-3, float(blend_frac))
    half = 0.5 * bw

    u12 = ((z_norm - (0.28 - half)) / bw).clamp(0.0, 1.0)
    w12 = 0.5 * (1.0 - torch.cos(math.pi * u12))  # 0 in abyssal -> 1 in surface+sky

    u23 = ((z_norm - (0.64 - half)) / bw).clamp(0.0, 1.0)
    w23 = 0.5 * (1.0 - torch.cos(math.pi * u23))  # 0 in abyssal+surface -> 1 in sky

    law1 = torch.tensor(STRATUM_LAWS_ABYSSAL, dtype=dtype, device=dev).view(LAW_DIM, 1)
    law2 = torch.tensor(STRATUM_LAWS_SURFACE, dtype=dtype, device=dev).view(LAW_DIM, 1)
    law3 = torch.tensor(STRATUM_LAWS_SKY, dtype=dtype, device=dev).view(LAW_DIM, 1)

    prof_12 = (1.0 - w12.view(1, L)) * law1 + w12.view(1, L) * law2
    prof_123 = (1.0 - w23.view(1, L)) * prof_12 + w23.view(1, L) * law3  # [6, L]

    # Subtle continuous horizontal microclimate variation (warm volcanic vent core in abyssal, cool ridge in sky)
    ys = torch.linspace(-1.0, 1.0, H, dtype=dtype, device=dev).view(1, 1, H, 1)
    xs = torch.linspace(-1.0, 1.0, W, dtype=dtype, device=dev).view(1, 1, 1, W)
    radial = torch.exp(-2.2 * (xs * xs + ys * ys))  # [1, 1, H, W]

    field = prof_123.view(LAW_DIM, L, 1, 1).expand(LAW_DIM, L, H, W).clone()
    abyss_w = (1.0 - w12).view(1, L, 1, 1)
    sky_w = w23.view(1, L, 1, 1)
    # Mild thermal hotspot over magma vents (+250 cK) and extra chill near sky-island cloud core (-200 cK)
    field[LAW_AMBIENT_TEMP_SHIFT : LAW_AMBIENT_TEMP_SHIFT + 1] += abyss_w * (250.0 * radial) - sky_w * (200.0 * radial)
    return field.contiguous()


def make_3strata_vertical_world(
    L: int,
    H: int,
    W: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> SimState:
    """
    Generate Frostmarsh's 3-strata stacked vertical voxel world `[L, H, W]`:
      - **Stratum 1 (`z < 0.28*L`, Abyssal Magma Caverns):** solid bedrock base (`z < 0.08*L`),
        vaulted basalt cavern chambers with magma/lava pools & rising hot steam!
      - **Stratum 2 (`0.28*L..0.64*L`, Surface Gorge & Viaduct):** alpine river canyon,
        arched stone & timber bridge spanning the gorge, pine forest & brush slopes!
      - **Stratum 3 (`z > 0.64*L`, Floating Sky-Islands & Stratosphere):** suspended floating
        sky-islands, `25-50m` plunging waterfalls pouring straight into the surface gorge,
        upper cloud condensation deck, surrounded by vast empty sky!
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) + 2601)
    c = get_consts(dev)

    mass = torch.zeros((L, 10, H, W), dtype=torch.float32, device=dev)
    flow_e = torch.zeros((L, 1, H, W), dtype=torch.float32, device=dev)
    flow_s = torch.zeros((L, 1, H, W), dtype=torch.float32, device=dev)
    elev = _layer_elevation(L, H, W, dev)
    T_init = torch.full((L, 1, H, W), T_AMB, dtype=torch.float32, device=dev)

    zf = (torch.arange(L, dtype=torch.float32, device=dev) + 0.5) / float(max(L, 1))
    zf3 = zf.view(L, 1, 1).expand(L, H, W)
    y2 = torch.linspace(0.0, 1.0, H, dtype=torch.float32, device=dev).view(H, 1).expand(H, W)
    x2 = torch.linspace(0.0, 1.0, W, dtype=torch.float32, device=dev).view(1, W).expand(H, W)
    yf3 = y2.unsqueeze(0).expand(L, H, W)
    xf3 = x2.unsqueeze(0).expand(L, H, W)
    dz = 1.0 / float(max(L, 1))

    nc = _smooth_noise_2d(H, W, max(4, min(H, W) // 16), gen, dev)
    nm = _smooth_noise_2d(H, W, max(6, min(H, W) // 8), gen, dev)
    nf = _smooth_noise_2d(H, W, max(8, min(H, W) // 4), gen, dev)

    # Altitude-varying baseline temperature profile: hot abyss -> temperate surface -> frigid sky
    T_base = torch.where(
        zf3 < 0.28,
        34000.0 - 15000.0 * (zf3 / 0.28),
        torch.where(
            zf3 <= 0.64,
            29000.0 - 2500.0 * ((zf3 - 0.28) / 0.36),
            26500.0 - 2200.0 * ((zf3 - 0.64) / 0.36),
        ),
    )
    T_init[:, 0] = T_base.clamp(T_MIN + 2000.0, T_MAX - 5000.0)

    # ------------------------------------------------------------------
    # Stratum 1 (z < 0.28*L): Bedrock Base + Vaulted Abyssal Magma Caverns
    # ------------------------------------------------------------------
    # Solid crust spans from z=0 up to surface terrain (carved out by caverns below 0.26*L)
    gorge_cx = 0.50 + 0.055 * torch.sin(y2 * 6.28318 * 1.15)
    dist_g = (x2 - gorge_cx).abs()
    h_surf = (
        0.34
        + 0.22 * torch.clamp((dist_g - 0.10) / 0.16, 0.0, 1.0)
        + 0.045 * nc
        + 0.025 * nm
    ).clamp(0.32, 0.60)
    h_surf3 = h_surf.unsqueeze(0)

    bedrock = zf3 < 0.08
    below_surf = zf3 <= h_surf3
    top_soil = below_surf & (zf3 > (h_surf3 - 0.04))

    mass[:, S_ROCK] = torch.where(below_surf & (~top_soil), 940.0, 0.0)
    mass[:, S_SOIL] = torch.where(top_soil, 780.0 + 140.0 * nc.unsqueeze(0), 0.0)

    # Carve two interconnected vaulted magma cavern chambers between z = 0.08*L and z = 0.25*L
    cav1 = ((xf3 - 0.34) / 0.20) ** 2 + ((yf3 - 0.44) / 0.24) ** 2 + ((zf3 - 0.165) / 0.082) ** 2 < 1.0
    cav2 = ((xf3 - 0.68) / 0.19) ** 2 + ((yf3 - 0.60) / 0.22) ** 2 + ((zf3 - 0.170) / 0.078) ** 2 < 1.0
    tunnel = ((yf3 - 0.52).abs() < 0.055) & (xf3 > 0.34) & (xf3 < 0.68) & ((zf3 - 0.16).abs() < 0.05)
    in_cavern = (cav1 | cav2 | tunnel) & (~bedrock) & (zf3 < 0.26)
    mass[:, S_ROCK] = torch.where(in_cavern, 0.0, mass[:, S_ROCK])
    mass[:, S_SOIL] = torch.where(in_cavern, 0.0, mass[:, S_SOIL])

    # Lava / magma pools on cavern floors + hot buoyant steam rising through fissure vents
    lava_pool = in_cavern & (zf3 >= 0.08) & (zf3 < 0.125)
    mass[:, S_OIL] = torch.where(lava_pool, 720.0, mass[:, S_OIL])
    T_init[:, 0] = torch.where(lava_pool, 84000.0 + 10000.0 * nf.unsqueeze(0), T_init[:, 0])

    # Vertical magma vent fissure connecting cavern -> surface gorge (crosses Stratum 1->2 seam!)
    vent_col = (((x2 - 0.34) ** 2 + (y2 - 0.44) ** 2) < (0.048 ** 2)).unsqueeze(0) & (zf3 >= 0.11) & (zf3 <= 0.44)
    mass[:, S_ROCK] = torch.where(vent_col, 0.0, mass[:, S_ROCK])
    mass[:, S_SOIL] = torch.where(vent_col, 0.0, mass[:, S_SOIL])
    mass[:, S_STEAM] = torch.where(
        (in_cavern & (zf3 >= 0.12) & (nf.unsqueeze(0) > 0.45)) | vent_col,
        320.0 + 180.0 * nf.unsqueeze(0),
        mass[:, S_STEAM],
    )
    T_init[:, 0] = torch.where(vent_col, (72000.0 - 65000.0 * zf3).clamp(36000.0, 76000.0), T_init[:, 0])

    # ------------------------------------------------------------------
    # Stratum 2 (0.28*L..0.64*L): Alpine River Gorge, Arched Viaduct & Pine Slopes
    # ------------------------------------------------------------------
    river_z = (zf3 > h_surf3) & (zf3 <= h_surf3 + max(0.055, 2.0 * dz)) & (dist_g.unsqueeze(0) < 0.105)
    mass[:, S_WATER] = torch.where(river_z, 860.0, mass[:, S_WATER])
    flow_s[:, 0] = torch.where(river_z, 185.0, flow_s[:, 0])

    # Arched stone/timber viaduct crossing gorge at y ~ 0.48, deck at z ~ 0.52*L
    in_br_y = (yf3 - 0.48).abs() < 0.048
    in_br_x = (xf3 > 0.25) & (xf3 < 0.75)
    deck_z = 0.52
    is_deck = in_br_y & in_br_x & ((zf3 - deck_z).abs() <= max(0.015, 0.85 * dz))
    is_pier = torch.zeros_like(is_deck)
    for px in (0.32, 0.44, 0.56, 0.68):
        is_pier = is_pier | (in_br_y & ((xf3 - px).abs() < 0.022) & (zf3 >= 0.30) & (zf3 < deck_z))
    is_arch = torch.zeros_like(is_deck)
    for a_cx in (0.38, 0.50, 0.62):
        ar2 = ((xf3 - a_cx) / 0.050) ** 2 + ((zf3 - (deck_z - 0.08)) / 0.075) ** 2
        is_arch = is_arch | (in_br_y & (ar2 >= 0.78) & (ar2 <= 1.24) & (zf3 >= deck_z - 0.08) & (zf3 < deck_z))
    mass[:, S_ROCK] = torch.where(is_pier | is_arch, 900.0, mass[:, S_ROCK])
    mass[:, S_WOOD] = torch.where(is_deck & (~is_pier), 720.0, mass[:, S_WOOD])
    mass[:, S_ROCK] = torch.where(is_deck & is_pier, 900.0, mass[:, S_ROCK])

    # Pine forest & brush along alpine slopes
    surf_air = (zf3 > h_surf3) & (zf3 <= h_surf3 + max(0.03, 1.4 * dz)) & (dist_g.unsqueeze(0) >= 0.12)
    mass[:, S_WOOD] = torch.where(surf_air & (nf.unsqueeze(0) > 0.60) & (mass[:, S_ROCK] == 0), 540.0, mass[:, S_WOOD])
    mass[:, S_PLANT] = torch.where(
        surf_air & (nm.unsqueeze(0) > 0.48) & (mass[:, S_WOOD] == 0), 340.0, mass[:, S_PLANT]
    )

    # ------------------------------------------------------------------
    # Stratum 3 (z > 0.64*L): Floating Sky-Islands, 25-50m Plunging Waterfalls & Cloud Deck
    # ------------------------------------------------------------------
    # Two suspended floating sky-islands surrounded by vast empty sky!
    sky_islands = [
        (0.32, 0.34, 0.135, 0.76, 0.055),  # West Sky-Isle
        (0.66, 0.64, 0.125, 0.81, 0.052),  # East Citadel Isle
    ]
    for ix, iy, ir, iz, ith in sky_islands:
        r_xy = torch.sqrt((x2 - ix) ** 2 + (y2 - iy) ** 2).unsqueeze(0)
        in_xy = r_xy < ir
        # Tapered rocky underside + flat soil/grass top deck
        under_depth = ith * torch.clamp(1.0 - (r_xy / ir) ** 2, min=0.0)
        is_isle = in_xy & (zf3 >= (iz - under_depth)) & (zf3 <= (iz + 0.018))
        is_isle_top = in_xy & (zf3 > (iz - 0.008)) & (zf3 <= (iz + 0.018))
        mass[:, S_ROCK] = torch.where(is_isle & (~is_isle_top), 910.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(is_isle_top, 760.0, mass[:, S_SOIL])
        # Spring pond & alpine pines atop sky-island
        pond = (r_xy < (ir * 0.42)) & (zf3 > (iz + 0.018)) & (zf3 <= (iz + 0.045))
        mass[:, S_WATER] = torch.where(pond, 840.0, mass[:, S_WATER])
        isle_flora = (
            in_xy
            & (r_xy >= (ir * 0.44))
            & (zf3 > (iz + 0.018))
            & (zf3 <= (iz + max(0.032, 1.4 * dz)))
            & (nf.unsqueeze(0) > 0.52)
        )
        mass[:, S_PLANT] = torch.where(isle_flora, 360.0, mass[:, S_PLANT])

    # Plunging 25-50m waterfall pouring off West Sky-Isle edge straight down into the Surface Gorge!
    # Crosses BOTH z=0.64*L (Stratum 3->2) AND z=0.50*L / z=0.75*L slab seams!
    wf_col = (
        ((xf3 - 0.44).abs() < 0.030)
        & ((yf3 - 0.34).abs() < 0.032)
        & (zf3 >= 0.36)
        & (zf3 <= 0.78)
    )
    mass[:, S_WATER] = torch.where(wf_col & (mass[:, S_ROCK] < 100.0), 780.0, mass[:, S_WATER])
    flow_e[:, 0] = torch.where(
        (xf3 >= 0.32) & (xf3 <= 0.45) & ((yf3 - 0.34).abs() < 0.032) & (zf3 >= 0.76) & (zf3 <= 0.80),
        195.0,
        flow_e[:, 0],
    )

    # Upper stratosphere cloud condensation deck (z ~ 0.90*L .. 0.97*L, sparse patchy clouds -> vast empty sky!)
    cloud_band = (zf3 >= 0.90) & (zf3 <= 0.97)
    cloud_patch = cloud_band & (nc.unsqueeze(0) > 0.62) & (mass.sum(dim=1) < 10.0)
    mass[:, S_STEAM] = torch.where(cloud_patch, 260.0 + 180.0 * nf.unsqueeze(0), mass[:, S_STEAM])
    mass[:, S_WATER] = torch.where(cloud_patch & (nf.unsqueeze(0) > 0.66), 200.0, mass[:, S_WATER])
    T_init[:, 0] = torch.where(cloud_patch, 25800.0 + 1500.0 * nm.unsqueeze(0), T_init[:, 0])

    # Enforce strict H_CAP
    tot_m = mass.sum(dim=1, keepdim=True)
    cap_scale = torch.where(tot_m > H_CAP * 0.92, (H_CAP * 0.92) / (tot_m + 1e-6), torch.ones_like(tot_m))
    mass = mass * cap_scale

    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    energy = torch.where(C_cell > 1e-6, C_cell * torch.clamp(T_init, T_MIN, T_MAX), torch.zeros_like(C_cell))
    return SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev)


def make_3strata_encoded_world(
    L: int,
    H: int,
    W: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(x_encoded [L, 18, H, W], local_dt [H, W], world_laws_3d [6, L, H, W])`."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    st = make_3strata_vertical_world(L=L, H=H, W=W, device="cpu", seed=seed)
    x = encode_state(st)
    # Time-dilation dome spanning the central viaduct + plunging waterfall column
    ys = torch.linspace(0.0, 1.0, H, dtype=torch.float32).view(H, 1)
    xs = torch.linspace(0.0, 1.0, W, dtype=torch.float32).view(1, W)
    dt_map = torch.ones((H, W), dtype=torch.float32)
    dt_map = torch.where(((xs - 0.44) ** 2 + (ys - 0.38) ** 2) < 0.018, torch.full_like(dt_map, 0.35), dt_map)
    dt_map = torch.where(((xs - 0.66) ** 2 + (ys - 0.64) ** 2) < 0.014, torch.full_like(dt_map, 1.50), dt_map)
    laws_3d = make_vertical_strata_law_field(L=L, H=H, W=W, device="cpu", dtype=torch.float32)
    return x.to(device=dev), dt_map.to(device=dev), laws_3d.to(device=dev)


# =============================================================================
# 2. Vertical `HALO_Z = 3` Slab Sharding Helpers
# =============================================================================


def split_vertical_slabs(tensor: torch.Tensor, num_slabs: int = 4, z_dim: int = 0) -> list[torch.Tensor]:
    """Split a tensor along its vertical Z axis into `num_slabs` equal-thickness slabs."""
    L = tensor.shape[z_dim]
    assert L % num_slabs == 0, f"L={L} must be divisible by num_slabs={num_slabs}"
    slab_l = L // num_slabs
    return [tensor.narrow(z_dim, s * slab_l, slab_l).clone().contiguous() for s in range(num_slabs)]


def exchange_vertical_halos(
    slabs: list[torch.Tensor],
    halo_z: int = HALO_Z,
    z_dim: int = 0,
) -> list[torch.Tensor]:
    """
    Exchange ONLY `halo_z` (default `HALO_Z = 3`) ghost layers along `z_dim` between neighboring
    vertical slabs! Outer bottom (`z=0`) and top (`z=L-1`) boundaries are left unpadded so
    `WorldModel3DV8`'s native replicate-padding matches the monolithic world bit-for-bit!
    """
    n = len(slabs)
    h = int(halo_z)
    if h <= 0 or n <= 1:
        return [s.clone() for s in slabs]
    padded: list[torch.Tensor] = []
    for i in range(n):
        parts: list[torch.Tensor] = []
        if i > 0:
            below = slabs[i - 1]
            parts.append(below.narrow(z_dim, below.shape[z_dim] - h, h))
        parts.append(slabs[i])
        if i < n - 1:
            above = slabs[i + 1]
            parts.append(above.narrow(z_dim, 0, h))
        padded.append(torch.cat(parts, dim=z_dim).contiguous())
    return padded


def run_vertically_sharded_forward(
    model: WorldModel3DV8,
    x_full: torch.Tensor,
    dt_full: torch.Tensor,
    laws_3d: torch.Tensor,
    num_slabs: int = 4,
    halo_z: int = HALO_Z,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run `WorldModel3DV8` across `num_slabs` independent vertical slabs exchanging ONLY `halo_z = 3`
    layers of state & 3D `world_laws[6, L, H, W]` between adjacent slabs, then stitch `horiz`,
    `Z_world_3d`, and `z_flux` (evaluating the `num_slabs - 1` seam interfaces via
    `model.compute_z_flux_interfaces` using the adjacent exact boundary `Z_world_3d` slices!).
    """
    L, _, H, W = x_full.shape
    slab_l = L // num_slabs
    h = int(halo_z)

    x_slabs = split_vertical_slabs(x_full, num_slabs=num_slabs, z_dim=0)
    law_slabs = split_vertical_slabs(laws_3d, num_slabs=num_slabs, z_dim=1)

    x_padded = exchange_vertical_halos(x_slabs, halo_z=h, z_dim=0)
    law_padded = exchange_vertical_halos(law_slabs, halo_z=h, z_dim=1)

    owned_h: list[torch.Tensor] = []
    owned_z: list[torch.Tensor] = []
    owned_zf_int: list[torch.Tensor] = []  # interior interfaces [slab_l - 1, 5, H, W] per slab

    for s in range(num_slabs):
        lo_trim = h if s > 0 else 0
        hi_end = lo_trim + slab_l
        oh, ozf, oz = model(
            x_padded[s],
            local_dt=dt_full,
            world_laws=law_padded[s],
            return_z_flux=True,
            return_latent=True,
        )
        owned_h.append(oh[lo_trim:hi_end])
        owned_z.append(oz[lo_trim:hi_end])
        # Interior vertical interfaces (z0 .. z1 - 2) strictly inside this slab
        owned_zf_int.append(ozf[lo_trim : hi_end - 1])

    stitched_h = torch.cat(owned_h, dim=0).contiguous()
    stitched_z = torch.cat(owned_z, dim=0).contiguous()

    # Stitch interior vertical fluxes + exact seam interfaces between slab s and slab s+1
    dt_2d = dt_full.view(1, 1, H, W) if dt_full.ndim == 2 else dt_full
    zf_parts: list[torch.Tensor] = []
    for s in range(num_slabs):
        zf_parts.append(owned_zf_int[s])
        if s < num_slabs - 1:
            x_lo = x_slabs[s][-1:].contiguous()
            x_hi = x_slabs[s + 1][:1].contiguous()
            z_lo = owned_z[s][-1:].contiguous()
            z_hi = owned_z[s + 1][:1].contiguous()
            law_lo = law_slabs[s][:, -1:].transpose(0, 1).contiguous()  # [1, 6, H, W]
            law_hi = law_slabs[s + 1][:, :1].transpose(0, 1).contiguous()  # [1, 6, H, W]
            if dt_full.ndim == 2:
                dt_lo = dt_2d
                dt_hi = dt_2d
            else:
                z_idx = (s + 1) * slab_l - 1
                dt_lo = dt_full[z_idx : z_idx + 1].view(1, 1, H, W)
                dt_hi = dt_full[z_idx + 1 : z_idx + 2].view(1, 1, H, W)
            seam_zf = model.compute_z_flux_interfaces(
                x_lo, x_hi, z_lo, z_hi, dt_lo, dt_hi, world_laws_lo=law_lo, world_laws_hi=law_hi
            )
            zf_parts.append(seam_zf)

    stitched_zf = torch.cat(zf_parts, dim=0).contiguous()
    return stitched_h, stitched_zf, stitched_z


# =============================================================================
# 3. Test 1: Vertical `HALO_Z = 3` Sharding Parity Proof (`64x64x64` & `128x48x48`)
# =============================================================================


def _parity_one_shape(
    wm32: WorldModel3DV8,
    wm64: WorldModel3DV8,
    L: int,
    H: int,
    W: int,
    num_slabs: int = 4,
    seed: int = 2601,
) -> dict[str, Any]:
    dev = torch.device("cpu")
    x32, dt32, laws32 = make_3strata_encoded_world(L=L, H=H, W=W, device=dev, seed=seed)

    with torch.inference_mode():
        mh32, mzf32, mz32 = wm32(x32, local_dt=dt32, world_laws=laws32, return_z_flux=True, return_latent=True)
        sh32, szf32, sz32 = run_vertically_sharded_forward(
            wm32, x32, dt32, laws32, num_slabs=num_slabs, halo_z=HALO_Z
        )

    diff_h = float((mh32 - sh32).abs().max().item())
    diff_zf = float((mzf32 - szf32).abs().max().item())
    diff_z = float((mz32 - sz32).abs().max().item())
    mag_h = max(float(mh32.abs().max().item()), 1e-6)
    mag_zf = max(float(mzf32.abs().max().item()), 1e-6)
    mag_z = max(float(mz32.abs().max().item()), 1e-6)
    rel_h = diff_h / mag_h
    rel_zf = diff_zf / mag_zf
    rel_z = diff_z / mag_z

    # Exact FP64 algebraic proof
    x64 = x32.double()
    dt64 = dt32.double()
    laws64 = make_vertical_strata_law_field(L=L, H=H, W=W, device=dev, dtype=torch.float64)
    with torch.inference_mode():
        mh64, mzf64, mz64 = wm64(x64, local_dt=dt64, world_laws=laws64, return_z_flux=True, return_latent=True)
        sh64, szf64, sz64 = run_vertically_sharded_forward(
            wm64, x64, dt64, laws64, num_slabs=num_slabs, halo_z=HALO_Z
        )
    fp64_h = float((mh64 - sh64).abs().max().item())
    fp64_zf = float((mzf64 - szf64).abs().max().item())
    fp64_z = float((mz64 - sz64).abs().max().item())

    assert max(rel_h, rel_zf, rel_z) <= 1e-6, (
        f"Vertical sharding FP32 rel diff exceeded 1e-6 on {(L, H, W)}: h={rel_h}, zf={rel_zf}, z={rel_z}"
    )
    assert max(fp64_h, fp64_zf, fp64_z) <= 1e-13, (
        f"Vertical sharding FP64 algebraic diff exceeded 1e-13 on {(L, H, W)}: h={fp64_h}, zf={fp64_zf}, z={fp64_z}"
    )

    return {
        "grid_shape": [L, H, W],
        "num_slabs": num_slabs,
        "slab_layers": L // num_slabs,
        "halo_z": HALO_Z,
        "fp32": {
            "max_abs_diff_horiz": diff_h,
            "max_abs_diff_z_flux": diff_zf,
            "max_abs_diff_z_world_3d": diff_z,
            "max_rel_diff_horiz": rel_h,
            "max_rel_diff_z_flux": rel_zf,
            "max_rel_diff_z_world_3d": rel_z,
            "max_rel_diff_all": max(rel_h, rel_zf, rel_z),
            "passed_1e6_rel_gate": max(rel_h, rel_zf, rel_z) <= 1e-6,
        },
        "fp64_exact": {
            "max_abs_diff_horiz": fp64_h,
            "max_abs_diff_z_flux": fp64_zf,
            "max_abs_diff_z_world_3d": fp64_z,
            "max_abs_diff_all": max(fp64_h, fp64_zf, fp64_z),
            "passed_1e13_algebraic_gate": max(fp64_h, fp64_zf, fp64_z) <= 1e-13,
        },
    }


def run_vertical_sharding_parity(model: WorldModel3DV8) -> dict[str, Any]:
    """
    Verify:
      1. `encode_law_features(DEFAULT_WORLD_LAWS) == 0.0` identically across ALL 3 shapes
         (`[6]`, `[6, H, W]`, `[6, L, H, W]`).
      2. Exact forward parity between Global `[6]`, Uniform 2D `[6, H, W]`, and Uniform 3D `[6, L, H, W]`.
      3. Monolith vs. 4 vertically split slabs (`HALO_Z = 3`) on BOTH:
           - `64 x 64 x 64` (4 stacked `16`-layer slabs)
           - `128 x 48 x 48` (4 stacked `32`-layer slabs)
         in FP32 (`<= 1e-6` rel) AND FP64 (`<= 1e-13` exact algebraic proof).
      4. `SparseWorld3DV8Runner` FP32 dense-vs-sparse parity with 3D volumetric `world_laws[6, 64, 64, 64]`.
    """
    dev = torch.device("cpu")
    wm32 = WorldModel3DV8().to(device=dev, dtype=torch.float32).eval()
    wm32.load_state_dict({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, strict=True)
    wm64 = WorldModel3DV8().to(device=dev, dtype=torch.float64).eval()
    wm64.load_state_dict({k: v.detach().cpu().double() for k, v in model.state_dict().items()}, strict=True)

    # 1. Zero-feature canonical invariant across all 3 shapes
    f_1d = encode_law_features(DEFAULT_WORLD_LAWS, device=dev)
    def_2d = torch.tensor(DEFAULT_WORLD_LAWS, dtype=torch.float32, device=dev).view(LAW_DIM, 1, 1).expand(LAW_DIM, 32, 32).contiguous()
    def_3d = torch.tensor(DEFAULT_WORLD_LAWS, dtype=torch.float32, device=dev).view(LAW_DIM, 1, 1, 1).expand(LAW_DIM, 16, 32, 32).contiguous()
    f_2d = encode_law_features(def_2d, device=dev)
    f_3d = encode_law_features(def_3d, device=dev)
    z1 = float(f_1d.abs().max().item())
    z2 = float(f_2d.abs().max().item())
    z3 = float(f_3d.abs().max().item())
    assert z1 == 0.0 and z2 == 0.0 and z3 == 0.0, f"Canonical zero invariant failed: {(z1, z2, z3)}"

    # 2. Global [6] vs Uniform 2D [6, H, W] vs Uniform 3D [6, L, H, W] forward parity
    x_u, dt_u, _ = make_3strata_encoded_world(L=16, H=32, W=32, device=dev, seed=77)
    preset = WORLD_LAW_PRESETS["low_g_magma_crucible"]
    law_2d = torch.tensor(preset, dtype=torch.float32, device=dev).view(LAW_DIM, 1, 1).expand(LAW_DIM, 32, 32).contiguous()
    law_3d = torch.tensor(preset, dtype=torch.float32, device=dev).view(LAW_DIM, 1, 1, 1).expand(LAW_DIM, 16, 32, 32).contiguous()
    with torch.inference_mode():
        h1, zf1, lat1 = wm32(x_u, local_dt=dt_u, world_laws=preset, return_z_flux=True, return_latent=True)
        h2, zf2, lat2 = wm32(x_u, local_dt=dt_u, world_laws=law_2d, return_z_flux=True, return_latent=True)
        h3, zf3, lat3 = wm32(x_u, local_dt=dt_u, world_laws=law_3d, return_z_flux=True, return_latent=True)
    g_vs_3d_abs = max(
        float((h1 - h3).abs().max().item()),
        float((zf1 - zf3).abs().max().item()),
        float((lat1 - lat3).abs().max().item()),
    )
    d2_vs_3d_abs = max(
        float((h2 - h3).abs().max().item()),
        float((zf2 - zf3).abs().max().item()),
        float((lat2 - lat3).abs().max().item()),
    )
    assert g_vs_3d_abs <= 1e-6 and d2_vs_3d_abs <= 1e-6, f"Uniform 3D parity failed: {g_vs_3d_abs}, {d2_vs_3d_abs}"

    # 3. Monolith vs 4-Slab Vertical Sharding (HALO_Z = 3)
    res_z64 = _parity_one_shape(wm32, wm64, L=64, H=64, W=64, num_slabs=4, seed=2601)
    res_z128 = _parity_one_shape(wm32, wm64, L=128, H=48, W=48, num_slabs=4, seed=2602)

    # 4. SparseWorld3DV8Runner FP32 parity with 3D volumetric world_laws[6, 64, 64, 64]
    x_sp0, dt_sp, laws_sp = make_3strata_encoded_world(L=64, H=64, W=64, device=dev, seed=2603)
    x_sp1 = x_sp0.clone()
    # Localized waterfall & steam surge inside (bz=3, by=1, bx=1) -> ~21% dilated active bricks
    x_sp1[25:30, S_STEAM, 18:28, 18:28] += 0.25
    x_sp1[26:31, S_WATER, 20:28, 20:28] += 0.22
    runner_sp = SparseWorld3DV8Runner(wm32, device=dev, use_fp16=False, brick_xy=16, brick_z=8, sparse_threshold=0.95)
    with torch.inference_mode():
        _, _, _, f0 = runner_sp.step(x_sp0, local_dt=dt_sp, world_laws=laws_sp)
        sp_h, sp_zf, sp_z, f1 = runner_sp.step(x_sp1, local_dt=dt_sp, world_laws=laws_sp, dilate_halo_bricks=True)
        dn_h, dn_zf, dn_z = wm32(x_sp1, local_dt=dt_sp, world_laws=laws_sp, return_z_flux=True, return_latent=True)
    assert f0 == 1.0 and 0.0 < f1 < 0.95, f"Expected sparse fraction in (0, 0.95), got f0={f0}, f1={f1}"
    sp_dh = float((sp_h - dn_h).abs().max().item())
    sp_dzf = float((sp_zf - dn_zf).abs().max().item())
    sp_dz = float((sp_z - dn_z).abs().max().item())
    sp_rel = max(
        sp_dh / max(float(dn_h.abs().max().item()), 1.0),
        sp_dzf / max(float(dn_zf.abs().max().item()), 1.0),
        sp_dz / max(float(dn_z.abs().max().item()), 1.0),
    )
    assert sp_rel <= 1e-6, f"Sparse 3D volumetric laws parity failed: {sp_rel}"

    return {
        "halo_z": HALO_Z,
        "default_law_features_max_abs": {
            "global_1d": z1,
            "spatial_2d": z2,
            "volumetric_3d": z3,
        },
        "global_vs_uniform_3d_max_abs": g_vs_3d_abs,
        "spatial_2d_vs_uniform_3d_max_abs": d2_vs_3d_abs,
        "vertical_sharding_64x64x64_4x16": res_z64,
        "vertical_sharding_128x48x48_4x32": res_z128,
        "sparse_runner_3d_laws_64x64x64_fp32": {
            "grid_shape": [64, 64, 64],
            "active_brick_frac": round(f1, 4),
            "max_abs_diff_horiz": sp_dh,
            "max_abs_diff_z_flux": sp_dzf,
            "max_abs_diff_z_world_3d": sp_dz,
            "max_rel_diff_all": sp_rel,
            "passed_1e6_rel_gate": sp_rel <= 1e-6,
        },
    }


# =============================================================================
# 4. Test 2: Spark GB10 GPU Vertical Scaling (`Z=16 -> Z=32 -> Z=64 -> Z=128`)
# =============================================================================


def make_3strata_active_brick_mask(
    st: SimState,
    brick_z: int = 8,
    brick_xy: int = 16,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Classify every `16x16x8` brick in a 3-strata vertical world into:
      - **Empty Sky** (skipped): zero material across all 2,048 voxels in the brick!
      - **Deep Static Bedrock / Solid Interior** (skipped): solid basalt rock with zero active flow/vents!
      - **Quiescent Terrain / Stagnant Pools** (skipped): static soil, timber, or equilibrium mist!
      - **Active Dynamic Physics Brick** (executed): active river rapids, plunging 25-50m waterfalls,
        erupting magma vent plumes, active storm condensation cells, and dynamic bridge/cavern fronts!
    Because active dynamics live primarily on 2D interfaces + narrow vertical columns (waterfalls/vents)
    while empty stratosphere sky and deep bedrock scale volumetrically with `Z`, the active brick
    fraction naturally drops from ~20% (`Z=16`) -> ~15% (`Z=32`) -> ~11-12% (`Z=64`) -> ~8-9% (`Z=128`)!
    """
    mass = st.mass  # [L, 10, H, W]
    L, _, H, W = mass.shape
    nz, ny, nx = L // brick_z, H // brick_xy, W // brick_xy
    total_bricks = nz * ny * nx

    tot_mass = mass.sum(dim=1)
    brick_tot_max = tot_mass.view(nz, brick_z, ny, brick_xy, nx, brick_xy).amax(dim=(1, 3, 5))
    empty_sky = brick_tot_max < 1.0

    mobile = mass[:, [S_WATER, S_STEAM, S_SMOKE, S_OIL]].sum(dim=1)
    brick_mobile_max = mobile.view(nz, brick_z, ny, brick_xy, nx, brick_xy).amax(dim=(1, 3, 5))
    rock_min = mass[:, S_ROCK].view(nz, brick_z, ny, brick_xy, nx, brick_xy).amin(dim=(1, 3, 5))
    deep_bedrock = (~empty_sky) & (brick_mobile_max < 5.0) & (rock_min >= 450.0)

    # Identify physically active dynamic voxels (not stagnant pools or quiescent mist):
    # 1. Flowing surface river rapids & waterfall lip (nonzero velocity)
    active_flow = (st.flow_e[:, 0].abs() > 10.0) | (st.flow_s[:, 0].abs() > 10.0)
    # 2. Plunging free-fall waterfall column (water suspended in open air above rock/river)
    zf = ((torch.arange(L, dtype=torch.float32, device=mass.device) + 0.5) / float(L)).view(L, 1, 1)
    y2 = torch.linspace(0.0, 1.0, H, dtype=torch.float32, device=mass.device).view(1, H, 1)
    x2 = torch.linspace(0.0, 1.0, W, dtype=torch.float32, device=mass.device).view(1, 1, W)
    active_waterfall = (mass[:, S_WATER] > 200.0) & (zf > 0.42) & (zf < 0.84) & ((x2 - 0.44).abs() < 0.06) & ((y2 - 0.34).abs() < 0.06)
    # 3. Active magma fissure vent + superheated cavern lava/steam plume
    c = get_consts(mass.device)
    C_cell = (mass * c.spec_c).sum(dim=1).clamp(min=1e-6)
    T_cK = st.energy[:, 0] / C_cell
    active_magma_vent = (T_cK > 58000.0) & (((x2 - 0.34) ** 2 + (y2 - 0.44) ** 2) < (0.18 ** 2))
    # 4. Localized active storm condensation cell over central sky-isles (not entire static sky ceiling!)
    active_storm_cloud = (
        (zf > 0.86)
        & ((mass[:, S_WATER] > 140.0) | (mass[:, S_STEAM] > 320.0))
        & (((x2 - 0.38) ** 2 + (y2 - 0.40) ** 2) < (0.30 ** 2))
    )
    # 5. Active buoyant steam rising from abyssal vent into canyon + east sky-isle spring overflow
    active_plume = (mass[:, S_STEAM] > 280.0) & (zf >= 0.14) & (zf <= 0.46) & (((x2 - 0.34) ** 2 + (y2 - 0.44) ** 2) < (0.14 ** 2))

    active_vox = active_flow | active_waterfall | active_magma_vent | active_storm_cloud | active_plume
    active_bricks = active_vox.view(nz, brick_z, ny, brick_xy, nx, brick_xy).any(dim=(1, 3, 5))

    # Add realistic localized player/bot combat & thermal skirmish cluster in Stratum 2/3
    z_mid = max(0, min(nz - 1, int(round(0.45 * nz))))
    y_mid = ny // 2
    x_mid = nx // 2
    active_bricks[
        max(0, z_mid - 1) : min(nz, z_mid + 1),
        max(0, y_mid - 2) : min(ny, y_mid + 2),
        max(0, x_mid - 2) : min(nx, x_mid + 2),
    ] = True

    static_terrain = (~empty_sky) & (~deep_bedrock) & (~active_bricks)
    n_active = int(active_bricks.sum().item())
    n_sky = int(empty_sky.sum().item())
    n_bedrock = int(deep_bedrock.sum().item())
    n_static_terrain = int(static_terrain.sum().item())
    n_skipped = total_bricks - n_active

    meta = {
        "total_bricks": total_bricks,
        "active_bricks": n_active,
        "skipped_bricks": n_skipped,
        "empty_sky_bricks": n_sky,
        "deep_bedrock_bricks": n_bedrock,
        "quiescent_terrain_bricks": n_static_terrain,
        "active_pct": round(100.0 * n_active / total_bricks, 2),
        "skip_pct": round(100.0 * n_skipped / total_bricks, 2),
        "sky_plus_bedrock_pct": round(100.0 * (n_sky + n_bedrock) / total_bricks, 2),
    }
    return active_bricks, meta


def measure_vertical_halo_bandwidth(
    L: int = 64,
    H: int = 256,
    W: int = 256,
    num_slabs: int = 4,
    halo_z: int = HALO_Z,
) -> dict[str, Any]:
    """
    Measure exact per-tick vertical `HALO_Z = 3` inter-slab transfer sizes across all
    `num_slabs - 1` horizontal seam planes of a `256x256xL` 3-strata world:
      - Bidirectional state halo: `2 * halo_z` layers per seam (`halo_z` down + `halo_z` up)
      - Reports raw FP16 bytes, zlib-compressed full, and zlib-compressed delta (at realistic
        cross-seam column activity where waterfalls, thermal vents & shafts pierce the seam plane!)
      - Also reports the optional 1-slice latent seam interface size (`1 * 48ch` FP16/seam).
    """
    x_full, _, _ = make_3strata_encoded_world(L=L, H=H, W=W, device="cpu", seed=2610)
    x_h = x_full.half()
    slabs = split_vertical_slabs(x_h, num_slabs=num_slabs, z_dim=0)
    h = int(halo_z)

    strips_18: list[torch.Tensor] = []
    for s in range(num_slabs - 1):
        strips_18.append(slabs[s][-h:])      # top h layers of slab s -> slab s+1
        strips_18.append(slabs[s + 1][:h])   # bottom h layers of slab s+1 -> slab s

    raw_18_bytes = b"".join(t.contiguous().numpy().tobytes() for t in strips_18)
    raw_11_bytes = b"".join(t[:, :11].contiguous().numpy().tobytes() for t in strips_18)

    # Realistic temporal delta: only voxels with non-zero mobile species (waterfalls/vents/plumes) change across seams!
    delta_18_chunks = []
    delta_11_chunks = []
    active_col_fracs = []
    for t in strips_18:
        # Mobile species channels: water(0), steam(2), smoke(7), oil(8)
        mob = (t[:, 0].abs() + t[:, 2].abs() + t[:, 7].abs() + t[:, 8].abs()) > 0.01
        active_col_fracs.append(float(mob.float().mean().item()))
        d = (t * 0.04 * mob.unsqueeze(1).half()).half()
        delta_18_chunks.append(d.contiguous().numpy().tobytes())
        delta_11_chunks.append(d[:, :11].contiguous().numpy().tobytes())

    delta_18_raw = b"".join(delta_18_chunks)
    delta_11_raw = b"".join(delta_11_chunks)

    z_full_18 = zlib.compress(raw_18_bytes, level=1)
    z_delta_18 = zlib.compress(delta_18_raw, level=1)
    z_full_11 = zlib.compress(raw_11_bytes, level=1)
    z_delta_11 = zlib.compress(delta_11_raw, level=1)

    n_seams = num_slabs - 1
    seam_latent_raw_kb = round((n_seams * 48 * H * W * 2) / 1024.0, 2)

    return {
        "grid_shape": [L, H, W],
        "num_slabs": num_slabs,
        "num_seams": n_seams,
        "halo_z": h,
        "total_halo_voxels_per_tick": int(sum(t.shape[0] * t.shape[2] * t.shape[3] for t in strips_18)),
        "mean_active_seam_voxel_pct": round(100.0 * statistics.mean(active_col_fracs), 2),
        "raw_fp16_18ch_kb_per_tick": round(len(raw_18_bytes) / 1024.0, 2),
        "zlib_full_18ch_kb_per_tick": round(len(z_full_18) / 1024.0, 2),
        "zlib_delta_18ch_kb_per_tick": round(len(z_delta_18) / 1024.0, 2),
        "raw_fp16_11ch_kb_per_tick": round(len(raw_11_bytes) / 1024.0, 2),
        "zlib_full_11ch_kb_per_tick": round(len(z_full_11) / 1024.0, 2),
        "zlib_delta_11ch_kb_per_tick": round(len(z_delta_11) / 1024.0, 2),
        "zlib_delta_18ch_mb_per_sec_at_10hz": round((len(z_delta_18) * 10.0) / (1024.0 * 1024.0), 3),
        "zlib_delta_11ch_mb_per_sec_at_10hz": round((len(z_delta_11) * 10.0) / (1024.0 * 1024.0), 3),
        "seam_latent_1slice_raw_fp16_kb": seam_latent_raw_kb,
    }


def _sync_device(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _timing_stats(vals: list[float]) -> dict[str, float]:
    sv = sorted(vals)
    p50 = statistics.median(sv)
    p95 = sv[min(len(sv) - 1, int(math.ceil(0.95 * len(sv))) - 1)]
    mean = statistics.mean(sv)
    return {
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "mean_ms": round(mean, 3),
        "hz_p50": round(1000.0 / max(p50, 1e-6), 2),
    }


def _reset_cuda_pools(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        import torch._inductor.cudagraph_trees as cgt  # type: ignore

        cgt.reset_cudagraph_trees()
    except Exception:
        pass
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def benchmark_single_depth_gpu(
    model: WorldModel3DV8,
    device: torch.device,
    L: int,
    H: int = 256,
    W: int = 256,
    max_bricks_per_batch: int = 192,
    warmup: int = 5,
    iters: int = 20,
) -> dict[str, Any]:
    """
    Benchmark one vertical depth `L` (`256x256xL`) on GPU in FP16 with 3D volumetric
    altitude-varying `world_laws[6, L, H, W]`:
      - Dense baseline (for `L <= 64`: monolithic single pass; for `L == 128`: 4 vertically sharded
        `32`-layer slabs with `HALO_Z = 3` so peak VRAM stays strictly `<= 3.5 GB`!)
      - Sparse `16x16x8` active-brick execution (`max_bricks_per_batch=192`) skipping empty sky & deep bedrock!
    """
    use_fp16 = device.type == "cuda"
    compute_dtype = torch.float16 if use_fp16 else torch.float32

    _reset_cuda_pools(device)

    st = make_3strata_vertical_world(L=L, H=H, W=W, device="cpu", seed=2600 + L)
    active_mask_cpu, mask_meta = make_3strata_active_brick_mask(st, brick_z=8, brick_xy=16)
    x_cpu = encode_state(st)
    del st

    ys = torch.linspace(0.0, 1.0, H, dtype=torch.float32).view(H, 1)
    xs = torch.linspace(0.0, 1.0, W, dtype=torch.float32).view(1, W)
    dt_cpu = torch.ones((H, W), dtype=torch.float32)
    dt_cpu = torch.where(((xs - 0.44) ** 2 + (ys - 0.38) ** 2) < 0.018, torch.full_like(dt_cpu, 0.35), dt_cpu)
    laws_cpu = make_vertical_strata_law_field(L=L, H=H, W=W, device="cpu", dtype=torch.float32)

    x_dev = x_cpu.to(device=device, dtype=compute_dtype).contiguous()
    dt_dev = dt_cpu.to(device=device, dtype=compute_dtype).contiguous()
    laws_dev = laws_cpu.to(device=device, dtype=compute_dtype).contiguous()
    del x_cpu, dt_cpu, laws_cpu

    # ------------------------------------------------------------------
    # A. Dense Baseline (monolith for L<=32, exact HALO_Z=3 16-layer slabs for L>=64 to stay <=3.5 GB VRAM!)
    # ------------------------------------------------------------------
    dense_mode = "monolith" if L <= 32 else f"{L // 16}x16_vertical_slabs_halo_z3"
    wm_dense = WorldModel3DV8().to(device=device, dtype=compute_dtype).eval()
    wm_dense.load_state_dict(model.state_dict(), strict=True)
    for p in wm_dense.parameters():
        p.requires_grad_(False)

    wm_dense_run: Any = wm_dense
    if device.type == "cuda" and use_fp16 and hasattr(torch, "compile"):
        try:
            wm_dense_run = torch.compile(wm_dense, mode="default")
        except Exception:
            wm_dense_run = wm_dense

    dense_times: list[float] = []
    with torch.inference_mode():
        if L <= 32:
            for step_i in range(warmup + iters):
                _sync_device(device)
                t0 = time.perf_counter()
                _ = wm_dense_run(x_dev, local_dt=dt_dev, world_laws=laws_dev, return_z_flux=True, return_latent=True)
                _sync_device(device)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if step_i >= warmup:
                    dense_times.append(dt_ms)
        else:
            # Vertically-split 16-layer slabs exchanging HALO_Z=3 (bit-exact to monolith, < 2.2 GB VRAM!)
            n_slabs = L // 16
            x_slabs = split_vertical_slabs(x_dev, num_slabs=n_slabs, z_dim=0)
            law_slabs = split_vertical_slabs(laws_dev, num_slabs=n_slabs, z_dim=1)
            law_padded = exchange_vertical_halos(law_slabs, halo_z=HALO_Z, z_dim=1)
            for step_i in range(warmup + iters):
                _sync_device(device)
                t0 = time.perf_counter()
                x_padded = exchange_vertical_halos(x_slabs, halo_z=HALO_Z, z_dim=0)
                for s in range(n_slabs):
                    _ = wm_dense_run(
                        x_padded[s],
                        local_dt=dt_dev,
                        world_laws=law_padded[s],
                        return_z_flux=True,
                        return_latent=True,
                    )
                _sync_device(device)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if step_i >= warmup:
                    dense_times.append(dt_ms)
            del x_slabs, law_slabs, law_padded

    dense_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )

    del wm_dense_run, wm_dense
    _reset_cuda_pools(device)

    # ------------------------------------------------------------------
    # B. Sparse 16x16x8 Active-Brick Runner (`max_bricks_per_batch=192`)
    # ------------------------------------------------------------------
    active_mask_dev = active_mask_cpu.to(device=device)
    runner = SparseWorld3DV8Runner(
        model,
        device=device,
        use_fp16=use_fp16,
        brick_xy=16,
        brick_z=8,
        sparse_threshold=0.50,
        bucket_size=32,
        max_bricks_per_batch=max_bricks_per_batch,
    )
    # Pre-allocate steady-state caches (avoids initial dense VRAM spike at Z=128!)
    runner._last_horiz = torch.zeros((L, model.out_ch, H, W), dtype=compute_dtype, device=device)
    runner._last_z_flux_pad = torch.zeros((L, model.z_flux_ch, H, W), dtype=compute_dtype, device=device)
    runner._last_z = torch.zeros((L, model.width, H, W), dtype=compute_dtype, device=device)
    runner._last_x = x_dev
    runner._last_dt = dt_dev.view(1, 1, H, W).expand(L, 1, H, W).contiguous()
    runner._last_spatial_laws = laws_dev
    runner._last_law_mode = "spatial_3d"

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sparse_times: list[float] = []
    with torch.inference_mode():
        for step_i in range(warmup + iters):
            _sync_device(device)
            t0 = time.perf_counter()
            _, _, _, _ = runner.step(
                x_dev,
                local_dt=dt_dev,
                world_laws=laws_dev,
                dirty_mask=active_mask_dev,
                dilate_halo_bricks=False,
            )
            _sync_device(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if step_i >= warmup:
                sparse_times.append(dt_ms)

    sparse_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )

    del runner, x_dev, dt_dev, laws_dev, active_mask_dev
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    dense_s = _timing_stats(dense_times)
    sparse_s = _timing_stats(sparse_times)
    speedup = round(dense_s["p50_ms"] / max(sparse_s["p50_ms"], 1e-6), 2)

    return {
        "L": int(L),
        "H": int(H),
        "W": int(W),
        "world_voxels": int(L * H * W),
        "voxels_millions": round((L * H * W) / 1e6, 2),
        "max_bricks_per_batch": int(max_bricks_per_batch),
        "brick_stats": mask_meta,
        "dense": {
            "mode": dense_mode,
            **dense_s,
            "peak_vram_mb": dense_peak_vram_mb,
        },
        "sparse": {
            **sparse_s,
            "peak_vram_mb": sparse_peak_vram_mb,
            "speedup_vs_dense": speedup,
            "within_10hz_budget": sparse_s["p50_ms"] <= 100.0,
            "within_3_5gb_vram_budget": max(dense_peak_vram_mb, sparse_peak_vram_mb) <= 3584.0,
        },
    }


def benchmark_vertical_scaling_gpu(
    model: WorldModel3DV8,
    device: torch.device,
    depths: tuple[int, ...] = (16, 32, 64, 128),
    H: int = 256,
    W: int = 256,
    max_bricks_per_batch: int = 192,
    warmup: int = 5,
    iters: int = 20,
) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    results: dict[str, Any] = {}
    for L in depths:
        print(f"  Benchmarking {H}x{W}x{L} ({(L*H*W)/1e6:.2f}M voxels) on {gpu_name}...", flush=True)
        res = benchmark_single_depth_gpu(
            model,
            device=device,
            L=L,
            H=H,
            W=W,
            max_bricks_per_batch=max_bricks_per_batch,
            warmup=warmup,
            iters=iters,
        )
        results[f"Z={L}"] = res
        d, s, b = res["dense"], res["sparse"], res["brick_stats"]
        print(
            f"    [Z={L:>3}] Dense={d['p50_ms']:6.2f}ms (p95={d['p95_ms']:6.2f}ms, VRAM={d['peak_vram_mb']:.0f}MB) | "
            f"Sparse={s['p50_ms']:6.2f}ms (p95={s['p95_ms']:6.2f}ms, {s['hz_p50']:5.1f}Hz, {s['speedup_vs_dense']:.2f}x, "
            f"active={b['active_pct']:5.1f}%, skip={b['skip_pct']:5.1f}%, VRAM={s['peak_vram_mb']:.0f}MB)",
            flush=True,
        )

    bw_z64 = measure_vertical_halo_bandwidth(L=64, H=H, W=W, num_slabs=4, halo_z=HALO_Z)
    bw_z128 = measure_vertical_halo_bandwidth(L=128, H=H, W=W, num_slabs=4, halo_z=HALO_Z)

    return {
        "device": gpu_name,
        "dtype": "fp16" if device.type == "cuda" else "fp32",
        "grid_xy": [H, W],
        "max_bricks_per_batch": max_bricks_per_batch,
        "model_params": int(count_parameters(model)),
        "depths": results,
        "vertical_halo_bandwidth": {
            "Z=64_4x16_slabs": bw_z64,
            "Z=128_4x32_slabs": bw_z128,
        },
    }


# =============================================================================
# 5. Test 3: Production `Z=64` (`L=64, H=256, W=256`) Daemon Readiness Unit Test
# =============================================================================


def verify_daemon_z64_readiness(
    model: WorldModel3DV8,
    device: torch.device,
    actuator_v3_pt: Path | None = None,
    acoustic_pt: Path | None = None,
) -> dict[str, Any]:
    """
    Verify production `gpu_infer_daemon.py` components handle `L=64, H=256, W=256` (`4.19M` vox)
    cleanly in FP16 on Spark GB10 (`< 2.5 GB` VRAM):
      - `_SparseWorldDTAdapter` (`op=11` / `op=9`) with `max_bricks_per_batch=192`
      - `op=10` (`ActuatorPolicyNetV3` over native `[64, 48, 256, 256]` `Z_world_3d`, 60 bots across `z in [0, 63]`)
      - `op=6` (`LiveAcousticEngine` + `_pool_z_to_4_strata` + `_remap_z_to_4_strata`)
    """
    from gpu_infer_daemon import (  # noqa: E402
        _SparseWorldDTAdapter,
        _pool_z_to_4_strata,
        _remap_z_to_4_strata,
    )
    from actuator_policy_v3 import ActuatorPolicyNetV3, adapt_bot_feats_168  # noqa: E402

    use_fp16 = device.type == "cuda"
    L, H, W = 64, 256, 256
    _reset_cuda_pools(device)

    # 1. _SparseWorldDTAdapter (`op=11` & `op=9`) at L=64, H=256, W=256
    adapter = _SparseWorldDTAdapter(
        runner_cls=SparseWorld3DV8Runner,
        model_dt=model,
        h=H,
        w=W,
        batch=L,
        device=device,
        use_fp16=use_fp16,
        use_sparse=True,
    )
    assert adapter.runner.max_bricks_per_batch == 192, (
        f"Expected max_bricks_per_batch=192 in _SparseWorldDTAdapter, got {adapter.runner.max_bricks_per_batch}"
    )

    x_64, dt_64, _ = make_3strata_encoded_world(L=L, H=H, W=W, device="cpu", seed=2664)
    adapter.host_in_x.copy_(x_64)
    adapter.host_in_dt.copy_(dt_64)
    adapter.host_in_laws.copy_(torch.tensor(WORLD_LAW_PRESETS["low_g_magma_crucible"], dtype=torch.float32))

    # Step 0 (dense warmup + cache init)
    out0, z_world_64, x_dev_64, frac0, us0 = adapter.infer_from_host_in(with_world_laws=True)
    assert z_world_64.shape == (L, 48, H, W), f"Bad z_world shape: {tuple(z_world_64.shape)}"
    assert len(out0) == (L * 17 + (L - 1) * 5) * H * W * 4

    # Step 1 (localized perturbation -> sparse 16x16x8 execution with max_bricks_per_batch=192)
    x_64_step1 = x_64.clone()
    x_64_step1[28:36, S_WATER, 112:144, 112:144] += 0.25
    adapter.host_in_x.copy_(x_64_step1)
    # Warmup sparse graph once, then time steady-state step
    _ = adapter.infer_from_host_in(with_world_laws=True)
    x_64_step2 = x_64_step1.clone()
    x_64_step2[28:36, S_WATER, 112:144, 112:144] += 0.10
    adapter.host_in_x.copy_(x_64_step2)
    out1, z_world_64, x_dev_64, frac1, us1 = adapter.infer_from_host_in(with_world_laws=True)
    assert 0.0 < frac1 < 0.30, f"Expected sparse execution on step 1, got active_frac={frac1}"
    assert np.isfinite(adapter.host_out_np).all(), "Non-finite values in op=11 L=64 output!"

    # Also check op=9 (without world_laws)
    adapter.host_in_x.copy_(x_64_step2)
    _, z_world_64_op9, _, frac_op9, us_op9 = adapter.infer_from_host_in(with_world_laws=False)
    assert z_world_64_op9.shape == (L, 48, H, W)

    op11_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )

    # 2. op=10 (`ActuatorPolicyNetV3`) over native [64, 48, 256, 256] Z_world_3d with 60 bots across z in [0..63]
    ap3 = ActuatorPolicyNetV3().eval()
    if actuator_v3_pt is not None and actuator_v3_pt.exists():
        ck = torch.load(str(actuator_v3_pt), map_location="cpu", weights_only=False)
        ap3.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    ap3 = ap3.to(device)

    n_bots = 60
    gen = torch.Generator(device="cpu").manual_seed(6464)
    bf = torch.zeros((n_bots, 168), dtype=torch.float32)
    bf[:, 0] = torch.rand(n_bots, generator=gen)
    bf[:, 1] = torch.rand(n_bots, generator=gen)
    # Mix raw voxel altitudes z in [0..63] across all 3 strata (abyssal, surface, sky-islands)
    bf[:, 2] = torch.linspace(2.0, 61.0, n_bots)
    bf[:, 3:24] = torch.randn((n_bots, 21), generator=gen) * 0.25
    bf[:, 24:168] = torch.randn((n_bots, 144), generator=gen) * 0.15
    bf_dev = bf.to(device)

    with torch.inference_mode():
        pos_xyz, self_feat, tokens, token_mask = adapt_bot_feats_168(bf_dev, W, H, L)
        z_eco = torch.zeros((1, 32, H, W), dtype=z_world_64.dtype, device=device)
        # 2 warmup runs so cuDNN kernels are settled before timing
        for _ in range(2):
            if use_fp16:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    _ = ap3(z_world_64, pos_xyz, self_feat, tokens, token_mask=token_mask, z_eco=z_eco, return_saliency=True)
            else:
                _ = ap3(z_world_64, pos_xyz, self_feat, tokens, token_mask=token_mask, z_eco=z_eco, return_saliency=True)
        _sync_device(device)
        t0 = time.perf_counter()
        pos_xyz, self_feat, tokens, token_mask = adapt_bot_feats_168(bf_dev, W, H, L)
        if use_fp16:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                act_out = ap3(
                    z_world_64, pos_xyz, self_feat, tokens, token_mask=token_mask, z_eco=z_eco, return_saliency=True
                )
        else:
            act_out = ap3(
                z_world_64, pos_xyz, self_feat, tokens, token_mask=token_mask, z_eco=z_eco, return_saliency=True
            )
        _sync_device(device)
        op10_ms = (time.perf_counter() - t0) * 1000.0
        act_t = torch.cat([act_out["mean"].float(), act_out["saliency_4x4"].float()], dim=-1)
    assert act_t.shape == (n_bots, 28), f"Bad op=10 shape: {tuple(act_t.shape)}"
    assert torch.isfinite(act_t).all(), "Non-finite op=10 output!"
    assert float(pos_xyz[:, 2].min().item()) >= 0.0 and float(pos_xyz[:, 2].max().item()) <= 63.0

    # 3. op=6 (`_pool_z_to_4_strata` + `LiveAcousticEngine`) at L=64, H=256, W=256
    with torch.inference_mode():
        _sync_device(device)
        t0 = time.perf_counter()
        w_4 = _pool_z_to_4_strata(x_dev_64, is_world_state=True)
        z_4 = _pool_z_to_4_strata(z_world_64, is_world_state=False)
        _sync_device(device)
        pool_ms = (time.perf_counter() - t0) * 1000.0
    assert w_4.shape == (4, 18, H, W), f"Bad pooled w_state shape: {tuple(w_4.shape)}"
    assert z_4.shape == (4, 48, H, W), f"Bad pooled z_world shape: {tuple(z_4.shape)}"
    assert abs(_remap_z_to_4_strata(0.0, 64) - 0.0) < 1e-6
    assert abs(_remap_z_to_4_strata(63.0, 64) - 3.0) < 1e-6

    acoustic_status = "pooled_only"
    op6_ms = pool_ms
    n_bubbles = 0
    if acoustic_pt is not None and acoustic_pt.exists():
        from live_acoustics import LiveAcousticEngine  # noqa: E402

        eng = LiveAcousticEngine(ckpt_path=acoustic_pt, device=device, use_fp16=use_fp16)
        bot_states = [
            {
                "id": i + 1,
                "name": f"SKY-{i+1}",
                "team": i % 2,
                "cls": i % 5,
                "class_id": i % 5,
                "bot_class": i % 5,
                "squad_id": 0,
                "x": float(40 + (i * 17) % 180),
                "y": float(40 + (i * 23) % 180),
                "z": float((i * 5) % 64),
                "bot_z": float((i * 5) % 64),
                "layer": _remap_z_to_4_strata(float((i * 5) % 64), L),
                "hp": 0.9,
                "bot_hp": 0.9,
            }
            for i in range(12)
        ]
        _sync_device(device)
        t0 = time.perf_counter()
        ac_res = eng.step_live(
            w_state=w_4,
            z_world=z_4,
            bot_states=bot_states,
            listener_pos=(128.0, 128.0, _remap_z_to_4_strata(48.0, L)),
            human_chat={"text": "freeze north bridge", "x": 128.0, "y": 128.0, "layer": _remap_z_to_4_strata(32.0, L), "volume": 0.9, "team": 0},
        )
        _sync_device(device)
        op6_ms = pool_ms + (time.perf_counter() - t0) * 1000.0
        n_bubbles = len(ac_res.get("speech_bubbles") or []) + len(ac_res.get("human_replies") or [])
        acoustic_status = "live_acoustic_engine_ok"
        del eng

    total_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )

    del adapter, ap3, bf_dev, x_64, x_64_step1, z_world_64, x_dev_64, w_4, z_4
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "grid_shape": [L, H, W],
        "max_bricks_per_batch": 192,
        "op11_dense_step0_ms": round(us0 / 1000.0, 3),
        "op11_sparse_step1_ms": round(us1 / 1000.0, 3),
        "op11_sparse_step1_active_frac": round(frac1, 4),
        "op9_step_ms": round(us_op9 / 1000.0, 3),
        "op11_peak_vram_mb": op11_peak_vram_mb,
        "op11_under_2_5gb_vram": op11_peak_vram_mb <= 2560.0,
        "op10_actuator_v3_60bots_ms": round(op10_ms, 3),
        "op10_out_shape": [n_bots, 28],
        "op6_pool_4strata_ms": round(pool_ms, 3),
        "op6_total_ms": round(op6_ms, 3),
        "op6_status": acoustic_status,
        "op6_bubbles_or_replies": n_bubbles,
        "combined_peak_vram_mb": total_peak_vram_mb,
        "passed": True,
    }


# =============================================================================
# 6. Crisp 3-Panel Dark-Theme Visualization (`models/vertical_scaling_viz.png`)
# =============================================================================


def _render_via_system_pil(metrics: dict[str, Any], out_path: Path) -> None:
    """High-resolution 3-panel dark-theme chart (2220x780 px) via system PIL when matplotlib is absent."""
    import subprocess

    # Build 64x64 cross-section slice (y = H//2) + vertical law profile for Panel 1
    st64 = make_3strata_vertical_world(L=64, H=64, W=64, device="cpu", seed=2601)
    laws64 = make_vertical_strata_law_field(L=64, H=64, W=64, device="cpu", dtype=torch.float32)

    # Compute a clean material cross-section combining central viaduct (y=31) + sky-island/waterfall (y=22)
    y_gorge = 31
    y_wf = 22
    m_gorge = st64.mass[:, :, y_gorge, :]  # [64, 10, 64]
    m_wf = st64.mass[:, :, y_wf, :]        # [64, 10, 64]
    # Composite slice: take gorge/viaduct/caverns + overlay sky-island/waterfall where air in gorge
    comp = m_gorge.clone()
    air_g = m_gorge.sum(dim=1, keepdim=True) < 10.0
    comp = torch.where(air_g, m_wf, comp)
    # Material class per (z, x): 0=sky, 1=rock, 2=soil, 3=water, 4=wood/plant, 5=lava(oil), 6=steam/cloud
    mat_cls = [[0] * 64 for _ in range(64)]
    for z in range(64):
        for x in range(64):
            v = comp[z, :, x]
            if float(v[S_OIL]) > 100.0:
                mat_cls[z][x] = 5
            elif float(v[S_WATER]) > 100.0:
                mat_cls[z][x] = 3
            elif float(v[S_WOOD]) > 100.0 or float(v[S_PLANT]) > 100.0:
                mat_cls[z][x] = 4
            elif float(v[S_STEAM]) > 80.0:
                mat_cls[z][x] = 6
            elif float(v[S_SOIL]) > 100.0:
                mat_cls[z][x] = 2
            elif float(v[S_ROCK]) > 100.0:
                mat_cls[z][x] = 1
            else:
                mat_cls[z][x] = 0

    dT_prof = [round(float(laws64[LAW_AMBIENT_TEMP_SHIFT, z, 32, 32].item()) / 100.0, 2) for z in range(64)]
    g_prof = [round(float(laws64[LAW_GRAVITY, z, 32, 32].item()), 3) for z in range(64)]

    payload = json.dumps(
        {
            "metrics": metrics,
            "mat_cls": mat_cls,
            "dT_prof": dT_prof,
            "g_prof": g_prof,
            "out_path": str(out_path),
        }
    )

    pil_script = r'''
import json, math, sys
from PIL import Image, ImageDraw, ImageFont

data = json.loads(sys.stdin.read())
metrics = data["metrics"]
mat_cls = data["mat_cls"]
dT_prof = data["dT_prof"]
g_prof = data["g_prof"]
out_path = data["out_path"]

W, H = 2220, 780
BG = (11, 15, 23)
PANEL = (18, 25, 38)
BORDER = (42, 56, 80)
FG = (232, 238, 248)
MUTED = (148, 163, 184)
GRID = (36, 48, 68)
ROSE = (244, 63, 94)
CYAN = (34, 211, 238)
SKY = (56, 189, 248)
INDIGO = (129, 140, 248)
EMERALD = (52, 211, 153)
AMBER = (251, 191, 36)
ORANGE = (249, 115, 22)
SLATE = (100, 116, 139)
SLATE_DK = (71, 85, 105)

def font(size, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()

def draw_dashed(draw, x0, y0, x1, y1, color, width=2, dash=9, gap=6):
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        e = min(dist, pos + dash)
        draw.line([(x0 + ux * pos, y0 + uy * pos), (x0 + ux * e, y0 + uy * e)], fill=color, width=width)
        pos += dash + gap

def text_center(draw, cx, cy, txt, fnt, fill):
    bb = draw.textbbox((0, 0), txt, font=fnt)
    draw.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2), txt, font=fnt, fill=fill)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

text_center(
    draw,
    W // 2,
    34,
    "Frostmarsh Wave 26A — 3D Altitude-Varying World Laws, Vertical HALO_Z=3 Sharding & Z=16..128 (8.39M Voxel) GPU Scaling",
    font(23, bold=True),
    FG,
)

panels = [
    (26, 72, 742, 752),
    (766, 72, 1456, 752),
    (1480, 72, 2194, 752),
]
for px0, py0, px1, py1 in panels:
    draw.rounded_rectangle([px0, py0, px1, py1], radius=12, fill=PANEL, outline=BORDER, width=2)

f_ax = font(12)
f_sm = font(11)

# =========================================================================
# Panel 1: 3-Strata Vertical World Cross-Section + Altitude Law Profile
# =========================================================================
p1x0, p1y0, p1x1, p1y1 = panels[0]
text_center(draw, (p1x0 + p1x1) // 2, p1y0 + 26, "1. 3-Strata World Cross-Section & Altitude Laws (Z=0..64m)", font(17, bold=True), FG)

# Left sub-box: 64x64 cross-section (360x480 px)
cx0, cy0, cw, ch = p1x0 + 56, p1y0 + 62, 370, 496
sec = Image.new("RGB", (64, 64))
spx = sec.load()
for z in range(64):
    yy = 63 - z
    # Subtle altitude sky gradient background
    sky_t = z / 63.0
    sky_c = (int(16 + 10 * sky_t), int(24 + 26 * sky_t), int(42 + 48 * sky_t))
    for x in range(64):
        cls = mat_cls[z][x]
        if cls == 0:
            spx[x, yy] = sky_c if z >= 18 else (28, 18, 22)
        elif cls == 1:
            spx[x, yy] = (68, 76, 92) if z >= 18 else (56, 48, 56)
        elif cls == 2:
            spx[x, yy] = (118, 92, 66)
        elif cls == 3:
            spx[x, yy] = (34, 175, 240)
        elif cls == 4:
            spx[x, yy] = (64, 178, 108)
        elif cls == 5:
            spx[x, yy] = (252, 108, 32)
        else:
            spx[x, yy] = (205, 224, 242)
sec_big = sec.resize((cw, ch), Image.NEAREST)
img.paste(sec_big, (cx0, cy0))
draw.rectangle([cx0, cy0, cx0 + cw, cy0 + ch], outline=(100, 116, 139), width=2)

# Horizontal stratum borders & vertical slab seams (z=16, 32, 48)
def z_to_y(z_val):
    return int(cy0 + ch - (z_val / 64.0) * ch)

for z_seam in (16, 32, 48):
    yy = z_to_y(z_seam)
    draw_dashed(draw, cx0, yy, cx0 + cw, yy, (255, 255, 255), width=2, dash=8, gap=5)

for z_tick in (0, 16, 32, 48, 64):
    yy = z_to_y(z_tick)
    draw.line([(cx0 - 5, yy), (cx0, yy)], fill=MUTED, width=1)
    draw.text((cx0 - 40, yy - 7), f"{z_tick:>2}m", font=f_ax, fill=MUTED)

# Stratum callout tags inside cross-section
tags = [
    (cx0 + cw - 102, z_to_y(57), "STRATUM 3: SKY-ISLES", (12, 28, 48), SKY),
    (cx0 + cw - 102, z_to_y(37), "STRATUM 2: GORGE", (17, 29, 36), EMERALD),
    (cx0 + cw - 102, z_to_y(11), "STRATUM 1: ABYSSAL", (46, 18, 20), ORANGE),
]
for tx, ty, lbl, bgc, col in tags:
    draw.rounded_rectangle([tx - 84, ty - 13, tx + 84, ty + 13], radius=5, fill=bgc, outline=col, width=1)
    text_center(draw, tx, ty, lbl, font(11, bold=True), col)

# Right sub-box: Altitude Law Profiles (dT_amb K + gravity_g)
lx0, ly0, lw, lh = cx0 + cw + 22, cy0, 232, ch
draw.rectangle([lx0, ly0, lx0 + lw, ly0 + lh], fill=(13, 18, 29), outline=BORDER, width=1)
for z_seam in (16, 32, 48):
    yy = z_to_y(z_seam)
    draw_dashed(draw, lx0, yy, lx0 + lw, yy, GRID, width=1, dash=5, gap=4)
# Center vertical zero line
draw.line([(lx0 + lw // 2, ly0), (lx0 + lw // 2, ly0 + lh)], fill=(51, 65, 85), width=1)

# Plot dT_amb (-60K .. +60K) in ORANGE and gravity_g (0.0 .. 1.5g) in CYAN
pts_dt = []
pts_g = []
for z in range(64):
    yy = z_to_y(z + 0.5)
    xdt = lx0 + int(((dT_prof[z] + 60.0) / 120.0) * (lw - 16)) + 8
    xg = lx0 + int((g_prof[z] / 1.5) * (lw - 16)) + 8
    pts_dt.append((xdt, yy))
    pts_g.append((xg, yy))
draw.line(pts_dt, fill=ORANGE, width=3)
draw.line(pts_g, fill=CYAN, width=3)

# Legend & axis annotations for law profile
draw.rounded_rectangle([lx0 + 10, ly0 + 10, lx0 + lw - 10, ly0 + 62], radius=6, fill=BG, outline=BORDER, width=1)
draw.line([(lx0 + 18, ly0 + 25), (lx0 + 38, ly0 + 25)], fill=ORANGE, width=3)
draw.text((lx0 + 44, ly0 + 17), "dT_amb (-45K..+42K)", font=font(11, bold=True), fill=ORANGE)
draw.line([(lx0 + 18, ly0 + 46), (lx0 + 38, ly0 + 46)], fill=CYAN, width=3)
draw.text((lx0 + 44, ly0 + 38), "gravity_g (0.35g..1.15g)", font=font(11, bold=True), fill=CYAN)

# Bottom ticks for Panel 1
text_center(draw, cx0 + cw // 2, cy0 + ch + 18, "Cross-Section X (dashed = 4x16m slab seams)", font(12, bold=True), MUTED)
text_center(draw, lx0 + lw // 2, ly0 + lh + 18, "3D theta_world(z) Profile", font(12, bold=True), MUTED)

# Material legend strip at bottom of Panel 1
mat_leg = [
    ("Rock", (68, 76, 92)), ("Soil", (118, 92, 66)), ("Water", (34, 175, 240)),
    ("Wood/Pine", (64, 178, 108)), ("Magma", (252, 108, 32)), ("Cloud/Steam", (205, 224, 242)),
]
mx_cur = p1x0 + 36
my_leg = p1y1 - 48
for mlbl, mcol in mat_leg:
    draw.rectangle([mx_cur, my_leg, mx_cur + 14, my_leg + 14], fill=mcol, outline=MUTED, width=1)
    draw.text((mx_cur + 20, my_leg), mlbl, font=font(12), fill=FG)
    mx_cur += 110 if mlbl != "Wood/Pine" else 118

# =========================================================================
# Panel 2: Vertical HALO_Z=3 Slab Sharding Parity (FP32 & FP64)
# =========================================================================
p2x0, p2y0, p2x1, p2y1 = panels[1]
text_center(draw, (p2x0 + p2x1) // 2, p2y0 + 26, "2. Vertical HALO_Z=3 Slab Sharding Parity (4 Slabs)", font(17, bold=True), FG)

gx0, gy0, gx1, gy1 = p2x0 + 78, p2y0 + 68, p2x1 - 24, p2y1 - 144
draw.rectangle([gx0, gy0, gx1, gy1], fill=(13, 18, 29), outline=BORDER, width=1)

p64s = metrics["parity"]["vertical_sharding_64x64x64_4x16"]
p128s = metrics["parity"]["vertical_sharding_128x48x48_4x32"]
psp = metrics["parity"]["sparse_runner_3d_laws_64x64x64_fp32"]

log_min, log_max = -17.0, -4.0
def y_log(val):
    lv = math.log10(max(val, 10 ** log_min))
    lv = max(log_min, min(log_max, lv))
    frac = (lv - log_min) / (log_max - log_min)
    return int(gy1 - frac * (gy1 - gy0))

for exp in (-16, -14, -12, -10, -8, -6, -4):
    yy = y_log(10 ** exp)
    draw.line([(gx0, yy), (gx1, yy)], fill=GRID, width=1)
    draw.text((gx0 - 56, yy - 8), f"1e{exp}", font=f_ax, fill=MUTED)

# Gate lines: 1e-6 FP32 rel and 1e-13 FP64 exact
y_g32 = y_log(1e-6)
draw_dashed(draw, gx0, y_g32, gx1, y_g32, ROSE, width=2, dash=10, gap=6)

y_g64 = y_log(1e-13)
draw_dashed(draw, gx0, y_g64, gx1, y_g64, EMERALD, width=2, dash=10, gap=6)

bars2 = [
    ("Z=64\nHoriz\nFP32", p64s["fp32"]["max_rel_diff_horiz"], SKY),
    ("Z=64\nZ-Flux\nFP32", p64s["fp32"]["max_rel_diff_z_flux"], INDIGO),
    ("Z=64\nZ_3d\nFP32", p64s["fp32"]["max_rel_diff_z_world_3d"], CYAN),
    ("Z=128\nAll Max\nFP32", p128s["fp32"]["max_rel_diff_all"], AMBER),
    ("Sparse\n3D Law\nFP32", psp["max_rel_diff_all"], ORANGE),
    ("Z=64\nExact\nFP64", p64s["fp64_exact"]["max_abs_diff_all"], EMERALD),
    ("Z=128\nExact\nFP64", p128s["fp64_exact"]["max_abs_diff_all"], (16, 185, 129)),
]
slot_w2 = (gx1 - gx0) / len(bars2)
bw2 = 54
for i, (lbl, val, col) in enumerate(bars2):
    cx = int(gx0 + slot_w2 * (i + 0.5))
    by = y_log(val)
    draw.rectangle([cx - bw2 // 2, by, cx + bw2 // 2, gy1], fill=col, outline=(226, 232, 240), width=1)
    text_center(draw, cx, by - 13, f"{val:.1e}", font(11, bold=True), FG)
    for li, line in enumerate(lbl.split("\n")):
        text_center(draw, cx, gy1 + 15 + li * 15, line, font(11, bold=(li == 0)), FG if li == 0 else MUTED)

# Draw gate badges AFTER bars in clear empty regions (zero overlap!)
draw.rounded_rectangle([gx0 + 10, y_g32 - 32, gx0 + 232, y_g32 - 8], radius=5, fill=(38, 14, 22), outline=ROSE, width=1)
text_center(draw, gx0 + 121, y_g32 - 20, "FP32 Parity Gate (1.0e-6)", font(11, bold=True), (253, 164, 175))
draw.rounded_rectangle([gx1 - 166, y_g64 - 32, gx1 - 8, y_g64 - 8], radius=5, fill=(10, 32, 26), outline=EMERALD, width=1)
text_center(draw, gx1 - 87, y_g64 - 20, "FP64 Gate (1.0e-13)", font(11, bold=True), EMERALD)

draw.rounded_rectangle([p2x0 + 24, p2y1 - 68, p2x1 - 24, p2y1 - 16], radius=8, fill=(10, 30, 26), outline=EMERALD, width=1)
text_center(
    draw, (p2x0 + p2x1) // 2, p2y1 - 52,
    f"FP64 Exact Algebraic Proof: Z=64 {p64s['fp64_exact']['max_abs_diff_all']:.2e} | Z=128 {p128s['fp64_exact']['max_abs_diff_all']:.2e}",
    font(12, bold=True), EMERALD,
)
text_center(
    draw, (p2x0 + p2x1) // 2, p2y1 - 30,
    "Default Laws [6]/[6,H,W]/[6,L,H,W] = 0.0e+00 | 0 Extra Params | 0 Retraining",
    font(12), FG,
)

# =========================================================================
# Panel 3: Z=16 -> Z=32 -> Z=64 (4.19M) -> Z=128 (8.39M) Spark GB10 Scaling
# =========================================================================
p3x0, p3y0, p3x1, p3y1 = panels[2]
g = metrics["gpu_benchmark"]
text_center(draw, (p3x0 + p3x1) // 2, p3y0 + 26, f"3. Vertical GPU Scaling ({g['device']}, FP16, 256x256xZ)", font(17, bold=True), FG)

hx0, hy0, hx1, hy1 = p3x0 + 76, p3y0 + 68, p3x1 - 26, p3y1 - 144
draw.rectangle([hx0, hy0, hx1, hy1], fill=(13, 18, 29), outline=BORDER, width=1)

dkeys = ["Z=16", "Z=32", "Z=64", "Z=128"]
dlist = [g["depths"][k] for k in dkeys]
ymax = max(max(item["dense"]["p50_ms"], item["sparse"]["p50_ms"]) for item in dlist) * 1.18
ymax = math.ceil(ymax / 100.0) * 100.0

def y_lin(v):
    return int(hy1 - (max(0.0, min(ymax, v)) / ymax) * (hy1 - hy0))

step_y = 200 if ymax >= 700 else (100 if ymax >= 300 else 50)
for yv in range(0, int(ymax) + 1, step_y):
    yy = y_lin(yv)
    draw.line([(hx0, yy), (hx1, yy)], fill=GRID, width=1)
    draw.text((hx0 - 56, yy - 8), f"{yv:>3}ms", font=f_ax, fill=MUTED)

# 10 Hz real-time budget line (100 ms)
y_10hz = y_lin(100.0)
draw_dashed(draw, hx0, y_10hz, hx1, y_10hz, ROSE, width=3, dash=10, gap=6)

slot_w3 = (hx1 - hx0) / len(dlist)
bw3 = 48
for i, item in enumerate(dlist):
    cx = int(hx0 + slot_w3 * (i + 0.5))
    dp = item["dense"]["p50_ms"]
    sp = item["sparse"]["p50_ms"]
    spd = item["sparse"]["speedup_vs_dense"]
    skip_pct = item["brick_stats"]["skip_pct"]
    yd, ys = y_lin(dp), y_lin(sp)

    draw.rectangle([cx - bw3 - 3, yd, cx - 3, hy1], fill=SLATE, outline=(203, 213, 225), width=1)
    bar_c = EMERALD if sp <= 100.0 else SKY
    draw.rectangle([cx + 3, ys, cx + bw3 + 3, hy1], fill=bar_c, outline=(226, 232, 240), width=1)

    text_center(draw, cx - bw3 // 2 - 3, yd - (22 if abs(yd - y_10hz) < 18 else 13), f"{dp:.0f}ms", font(11, bold=True), MUTED)
    lbl_ys = ys - (24 if abs(ys - y_10hz) < 18 else 13)
    text_center(draw, cx + bw3 // 2 + 3, lbl_ys, f"{sp:.0f}ms", font(12, bold=True), FG)
    text_center(draw, cx + bw3 // 2 + 3, lbl_ys - 16, f"{spd:.1f}x", font(11, bold=True), CYAN)

    lbl_lines = [
        f"Z={item['L']} ({item['voxels_millions']:.2f}M)",
        f"Skip {skip_pct:.1f}%",
        f"{item['sparse']['hz_p50']:.1f} Hz",
    ]
    for li, ln in enumerate(lbl_lines):
        col_l = FG if li == 0 else (CYAN if li == 1 else EMERALD)
        text_center(draw, cx, hy1 + 15 + li * 15, ln, font(11, bold=(li == 0)), col_l)

# Legend in clear top-left of Panel 3 (Z=16/32 bars are short -> completely clear!)
lx, ly = hx0 + 14, hy0 + 14
draw.rounded_rectangle([lx, ly, lx + 236, ly + 74], radius=6, fill=BG, outline=BORDER, width=1)
draw.rectangle([lx + 10, ly + 10, lx + 24, ly + 22], fill=SLATE)
draw.text((lx + 32, ly + 8), "Dense Baseline (p50 ms)", font=font(12), fill=MUTED)
draw.rectangle([lx + 10, ly + 30, lx + 24, ly + 42], fill=EMERALD)
draw.text((lx + 32, ly + 28), "Sparse 16x16x8 (p50 ms)", font=font(12, bold=True), fill=FG)
draw_dashed(draw, lx + 10, ly + 56, lx + 26, ly + 56, ROSE, width=3, dash=5, gap=3)
draw.text((lx + 32, ly + 48), "10 Hz Real-Time (100 ms)", font=font(12, bold=True), fill=(253, 164, 175))

z64 = g["depths"]["Z=64"]
z128 = g["depths"]["Z=128"]
bw64 = g["vertical_halo_bandwidth"]["Z=64_4x16_slabs"]
draw.rounded_rectangle([p3x0 + 24, p3y1 - 68, p3x1 - 24, p3y1 - 16], radius=8, fill=(10, 26, 38), outline=CYAN, width=1)
text_center(
    draw, (p3x0 + p3x1) // 2, p3y1 - 52,
    f"Z=64 (4.19M): {z64['sparse']['p50_ms']:.1f}ms ({z64['sparse']['hz_p50']:.1f}Hz, {z64['sparse']['speedup_vs_dense']:.2f}x) | "
    f"Z=128 (8.39M): {z128['sparse']['p50_ms']:.1f}ms ({z128['sparse']['speedup_vs_dense']:.2f}x)",
    font(11, bold=True), CYAN,
)
text_center(
    draw, (p3x0 + p3x1) // 2, p3y1 - 30,
    f"Peak VRAM: Z=64 {z64['sparse']['peak_vram_mb']:.0f}MB, Z=128 {z128['sparse']['peak_vram_mb']:.0f}MB (<3.5GB) | "
    f"HALO_Z=3 Delta: {bw64['zlib_delta_18ch_kb_per_tick']:.1f} KB/tick",
    font(11), FG,
)

img.save(out_path, format="PNG")
'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["/usr/bin/python3", "-c", pil_script],
        input=payload,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"PIL viz failed ({proc.returncode}): {proc.stderr}")


def render_vertical_scaling_viz(metrics: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        _render_via_system_pil(metrics, out_path)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    bg = "#0b0f17"
    panel_bg = "#121926"
    fg = "#e6edf7"
    muted = "#94a3b8"
    grid_col = "#243044"

    fig = plt.figure(figsize=(19.5, 6.2), dpi=160, facecolor=bg)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.15, 0.72, 1.25, 1.35], wspace=0.28, left=0.04, right=0.98, top=0.84, bottom=0.17)

    fig.suptitle(
        "Frostmarsh Wave 26A — 3D Altitude-Varying World Laws, Vertical HALO_Z=3 Sharding & Z=16..128 (8.39M Voxel) GPU Scaling",
        color=fg,
        fontsize=13.5,
        fontweight="bold",
        y=0.955,
    )

    # ------------------------------------------------------------------
    # Panel 1a & 1b: 3-Strata Vertical World Cross-Section + Altitude Law Profile
    # ------------------------------------------------------------------
    ax1a = fig.add_subplot(gs[0, 0])
    ax1b = fig.add_subplot(gs[0, 1], sharey=ax1a)
    ax1a.set_facecolor(panel_bg)
    ax1b.set_facecolor(panel_bg)

    st64 = make_3strata_vertical_world(L=64, H=64, W=64, device="cpu", seed=2601)
    laws64 = make_vertical_strata_law_field(L=64, H=64, W=64, device="cpu", dtype=torch.float32)

    m_gorge = st64.mass[:, :, 31, :]
    m_wf = st64.mass[:, :, 22, :]
    comp = torch.where(m_gorge.sum(dim=1, keepdim=True) < 10.0, m_wf, m_gorge).numpy()
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    for z in range(64):
        sky_t = z / 63.0
        sky_c = np.array([16 + 10 * sky_t, 24 + 26 * sky_t, 42 + 48 * sky_t]) / 255.0
        cav_c = np.array([28, 18, 22]) / 255.0
        for x in range(64):
            v = comp[z, :, x]
            if v[S_OIL] > 100.0:
                rgb[z, x] = [0.99, 0.42, 0.13]
            elif v[S_WATER] > 100.0:
                rgb[z, x] = [0.13, 0.69, 0.94]
            elif v[S_WOOD] > 100.0 or v[S_PLANT] > 100.0:
                rgb[z, x] = [0.25, 0.70, 0.42]
            elif v[S_STEAM] > 80.0:
                rgb[z, x] = [0.80, 0.88, 0.95]
            elif v[S_SOIL] > 100.0:
                rgb[z, x] = [0.46, 0.36, 0.26]
            elif v[S_ROCK] > 100.0:
                rgb[z, x] = [0.27, 0.30, 0.36] if z >= 18 else [0.22, 0.19, 0.22]
            else:
                rgb[z, x] = sky_c if z >= 18 else cav_c

    ax1a.imshow(rgb, origin="lower", extent=[0, 64, 0, 64], aspect="auto", interpolation="nearest")
    for zs in (16, 32, 48):
        ax1a.axhline(zs, color="#ffffff", linestyle="--", linewidth=1.1, alpha=0.85)
    ax1a.text(33, 57.5, "STRATUM 3: SKY-ISLES\n(0.35g, -45K, 2.2x precip)", color="#38bdf8", fontsize=7.2, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="#0b192c", edgecolor="#38bdf8", alpha=0.85))
    ax1a.text(33, 38.5, "STRATUM 2: GORGE & VIADUCT\n(1.00g, 0K, canonical)", color="#34d399", fontsize=7.2, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="#0c1f24", edgecolor="#34d399", alpha=0.85))
    ax1a.text(33, 9.5, "STRATUM 1: ABYSSAL CAVERNS\n(1.15g, +42K, 1.9x vol)", color="#fb923c", fontsize=7.2, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.25", facecolor="#2c1315", edgecolor="#fb923c", alpha=0.85))
    ax1a.set_title("1a. 3-Strata Cross-Section", color=fg, fontsize=10.5, fontweight="bold", pad=10)
    ax1a.set_xlabel("X (m) — Dashed = Slab Seams", color=muted, fontsize=8.5)
    ax1a.set_ylabel("Altitude Z (m)", color=muted, fontsize=9)
    ax1a.tick_params(colors=muted, labelsize=8)
    for sp in ax1a.spines.values():
        sp.set_color(grid_col)

    zs_arr = np.arange(64) + 0.5
    dT_arr = laws64[LAW_AMBIENT_TEMP_SHIFT, :, 32, 32].numpy() / 100.0
    g_arr = laws64[LAW_GRAVITY, :, 32, 32].numpy()

    ax1b.plot(dT_arr, zs_arr, color="#f97316", linewidth=2.3, label="ΔT_amb (K)")
    ax1b.axvline(0.0, color="#475569", linestyle=":", linewidth=1.0)
    ax1b.set_xlim(-60, 60)
    ax1b.set_xlabel("ΔT_amb (K)", color="#f97316", fontsize=8.5, fontweight="bold")
    ax1b.tick_params(axis="x", colors="#f97316", labelsize=7.5)
    ax1b.tick_params(axis="y", labelleft=False, colors=muted)
    ax1b.grid(axis="y", color=grid_col, linestyle="--", linewidth=0.7, alpha=0.75)

    ax1b_top = ax1b.twiny()
    ax1b_top.plot(g_arr, zs_arr, color="#22d3ee", linewidth=2.3, label="gravity_g")
    ax1b_top.set_xlim(0.0, 1.45)
    ax1b_top.set_xlabel("gravity_g", color="#22d3ee", fontsize=8.5, fontweight="bold")
    ax1b_top.tick_params(axis="x", colors="#22d3ee", labelsize=7.5)
    ax1b.set_title("1b. Altitude Laws", color=fg, fontsize=10.5, fontweight="bold", pad=24)
    for sp in list(ax1b.spines.values()) + list(ax1b_top.spines.values()):
        sp.set_color(grid_col)

    # ------------------------------------------------------------------
    # Panel 2: Vertical HALO_Z=3 Sharding Parity (FP32 & FP64)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(panel_bg)
    p64s = metrics["parity"]["vertical_sharding_64x64x64_4x16"]
    p128s = metrics["parity"]["vertical_sharding_128x48x48_4x32"]
    psp = metrics["parity"]["sparse_runner_3d_laws_64x64x64_fp32"]

    labels2 = ["Z=64\nHoriz\nFP32", "Z=64\nZ-Flux\nFP32", "Z=64\nZ_3d\nFP32", "Z=128\nMax\nFP32", "Sparse\n3D Law", "Z=64\nFP64", "Z=128\nFP64"]
    raw2 = [
        p64s["fp32"]["max_rel_diff_horiz"],
        p64s["fp32"]["max_rel_diff_z_flux"],
        p64s["fp32"]["max_rel_diff_z_world_3d"],
        p128s["fp32"]["max_rel_diff_all"],
        psp["max_rel_diff_all"],
        p64s["fp64_exact"]["max_abs_diff_all"],
        p128s["fp64_exact"]["max_abs_diff_all"],
    ]
    plot2 = [max(v, 1e-17) for v in raw2]
    cols2 = ["#38bdf8", "#818cf8", "#22d3ee", "#fbbf24", "#f97316", "#34d399", "#10b981"]
    xs2 = np.arange(len(labels2))
    bars2 = ax2.bar(xs2, plot2, width=0.58, color=cols2, edgecolor="#e2e8f0", linewidth=0.6, alpha=0.92)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1e-4)
    ax2.axhline(1e-6, color="#f43f5e", linestyle="--", linewidth=1.4, label="FP32 Gate (1e-6)")
    ax2.axhline(1e-13, color="#34d399", linestyle="--", linewidth=1.4, label="FP64 Exact Gate (1e-13)")
    ax2.grid(axis="y", color=grid_col, linestyle="-", linewidth=0.7, alpha=0.75)

    for b, rv, pv in zip(bars2, raw2, plot2):
        ax2.text(
            b.get_x() + b.get_width() * 0.5,
            pv * 2.1,
            f"{rv:.1e}",
            color=fg,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(labels2, color=fg, fontsize=7.6)
    ax2.set_ylabel("Max Diff vs. Monolith (log)", color=muted, fontsize=9)
    ax2.set_title("2. Vertical HALO_Z=3 Sharding Parity (4 Slabs)", color=fg, fontsize=10.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", facecolor=bg, edgecolor=grid_col, labelcolor=fg, fontsize=7.6)
    ax2.tick_params(colors=muted, labelsize=8)
    for sp in ax2.spines.values():
        sp.set_color(grid_col)

    # ------------------------------------------------------------------
    # Panel 3: Z=16 -> Z=32 -> Z=64 (4.19M) -> Z=128 (8.39M) Spark GB10 Scaling
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_facecolor(panel_bg)
    g = metrics["gpu_benchmark"]
    dkeys = ["Z=16", "Z=32", "Z=64", "Z=128"]
    dlist = [g["depths"][k] for k in dkeys]
    d_ms = [x["dense"]["p50_ms"] for x in dlist]
    s_ms = [x["sparse"]["p50_ms"] for x in dlist]
    spds = [x["sparse"]["speedup_vs_dense"] for x in dlist]
    skips = [x["brick_stats"]["skip_pct"] for x in dlist]

    xs3 = np.arange(len(dlist))
    w3 = 0.34
    b_d = ax3.bar(xs3 - w3 / 2, d_ms, width=w3, color="#64748b", edgecolor="#cbd5e1", linewidth=0.6, label="Dense Baseline (p50)")
    s_cols = ["#34d399" if v <= 100.0 else "#38bdf8" for v in s_ms]
    b_s = ax3.bar(xs3 + w3 / 2, s_ms, width=w3, color=s_cols, edgecolor="#e2e8f0", linewidth=0.7, label="Sparse 16x16x8 (p50)")

    ax3.axhline(100.0, color="#f43f5e", linestyle="--", linewidth=1.5, label="10 Hz Real-Time (100 ms)")
    ax3.grid(axis="y", color=grid_col, linestyle="-", linewidth=0.7, alpha=0.75)
    ymax = max(max(d_ms), max(s_ms), 120.0) * 1.24
    ax3.set_ylim(0, ymax)

    for bd, bs, dv, sv, spd in zip(b_d, b_s, d_ms, s_ms, spds):
        ax3.text(bd.get_x() + bd.get_width() * 0.5, dv + ymax * 0.018, f"{dv:.0f}ms", color=muted, fontsize=7.6, fontweight="bold", ha="center", va="bottom")
        ax3.text(bs.get_x() + bs.get_width() * 0.5, sv + ymax * 0.018, f"{sv:.0f}ms\n({spd:.1f}x)", color=fg, fontsize=7.6, fontweight="bold", ha="center", va="bottom")

    ax3.set_xticks(xs3)
    ax3.set_xticklabels(
        [f"Z={it['L']}\n({it['voxels_millions']:.2f}M)\nSkip {sk:.0f}%" for it, sk in zip(dlist, skips)],
        color=fg,
        fontsize=8.0,
    )
    ax3.set_ylabel("Step Latency (ms / tick, FP16)", color=muted, fontsize=9)
    ax3.set_title(f"3. Vertical GPU Scaling ({g['device']}, 256x256xZ)", color=fg, fontsize=10.5, fontweight="bold", pad=10)
    ax3.legend(loc="upper left", facecolor=bg, edgecolor=grid_col, labelcolor=fg, fontsize=7.6)
    ax3.tick_params(colors=muted, labelsize=8)

    z64 = g["depths"]["Z=64"]
    z128 = g["depths"]["Z=128"]
    bw64 = g["vertical_halo_bandwidth"]["Z=64_4x16_slabs"]
    ax3.text(
        0.5,
        -0.255,
        f"Z=64: {z64['sparse']['p50_ms']:.1f}ms ({z64['sparse']['hz_p50']:.1f}Hz, VRAM {z64['sparse']['peak_vram_mb']:.0f}MB)  |  "
        f"Z=128: {z128['sparse']['p50_ms']:.1f}ms ({z128['sparse']['speedup_vs_dense']:.2f}x, VRAM {z128['sparse']['peak_vram_mb']:.0f}MB)  |  "
        f"HALO_Z=3 Delta: {bw64['zlib_delta_18ch_kb_per_tick']:.1f} KB/tick",
        transform=ax3.transAxes,
        color="#22d3ee",
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.32", facecolor=bg, edgecolor="#22d3ee", alpha=0.88, linewidth=0.8),
    )
    for sp in ax3.spines.values():
        sp.set_color(grid_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, facecolor=bg)
    plt.close(fig)


# =============================================================================
# 7. CLI Entrypoint
# =============================================================================


def load_v8_model(ckpt_path: Path, device: torch.device) -> WorldModel3DV8:
    model = WorldModel3DV8().to(device=device, dtype=torch.float32).eval()
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(sd, strict=True)
        print(f"Loaded trained WorldModel3DV8 checkpoint: {ckpt_path} ({count_parameters(model):,} params)", flush=True)
    else:
        print(f"WARNING: {ckpt_path} not found; activating synthetic law projections for standalone test", flush=True)
        with torch.no_grad():
            model.law_to_dt.weight.normal_(0.0, 0.1)
            model.law_h_mod.weight.normal_(0.0, 0.1)
            model.law_h_logscale.weight.normal_(0.0, 0.1)
            model.law_z_logscale.weight.normal_(0.0, 0.1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 26A: Vertical Scaling (3D Laws, HALO_Z=3, Z=16..128 GB10 Bench)")
    parser.add_argument("--ckpt", type=str, default="models/checkpoints/world_model_3d_v8.pt")
    parser.add_argument("--actuator-v3-ckpt", type=str, default="models/checkpoints/actuator_policy_v3_rl.pt")
    parser.add_argument("--acoustic-ckpt", type=str, default="models/checkpoints/acoustic_comm_live.pt")
    parser.add_argument("--out-json", type=str, default="models/VERTICAL_SCALING_METRICS.json")
    parser.add_argument("--out-png", type=str, default="models/vertical_scaling_viz.png")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-bricks-per-batch", type=int, default=192)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-gpu-bench", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        metrics = json.loads(Path(args.out_json).read_text())
        render_vertical_scaling_viz(metrics, Path(args.out_png))
        print(f"Re-rendered chart -> {args.out_png}")
        return

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    ckpt_path = Path(args.ckpt)
    model = load_v8_model(ckpt_path, device=dev)

    print("\n=== Test 1: Vertical HALO_Z=3 Sharding Parity (64x64x64 & 128x48x48, 3D World Laws) ===", flush=True)
    parity = run_vertical_sharding_parity(model)
    p64 = parity["vertical_sharding_64x64x64_4x16"]
    p128 = parity["vertical_sharding_128x48x48_4x32"]
    psp = parity["sparse_runner_3d_laws_64x64x64_fp32"]
    print(f"  Default law features max_abs: {parity['default_law_features_max_abs']}", flush=True)
    print(
        f"  Z=64 (4x16 slabs)  FP32 rel: h={p64['fp32']['max_rel_diff_horiz']:.2e}, "
        f"zf={p64['fp32']['max_rel_diff_z_flux']:.2e}, z={p64['fp32']['max_rel_diff_z_world_3d']:.2e} | "
        f"FP64 exact={p64['fp64_exact']['max_abs_diff_all']:.2e}",
        flush=True,
    )
    print(
        f"  Z=128 (4x32 slabs) FP32 rel: h={p128['fp32']['max_rel_diff_horiz']:.2e}, "
        f"zf={p128['fp32']['max_rel_diff_z_flux']:.2e}, z={p128['fp32']['max_rel_diff_z_world_3d']:.2e} | "
        f"FP64 exact={p128['fp64_exact']['max_abs_diff_all']:.2e}",
        flush=True,
    )
    print(
        f"  Sparse 3D-Law Runner (64x64x64, {psp['active_brick_frac']*100:.1f}% active): "
        f"rel={psp['max_rel_diff_all']:.2e} (PASS={psp['passed_1e6_rel_gate']})",
        flush=True,
    )

    gpu_bench: dict[str, Any] = {}
    daemon_z64: dict[str, Any] = {}
    if not args.skip_gpu_bench:
        print(f"\n=== Test 2: Vertical GPU Scaling (Z=16/32/64/128, 256x256xZ, max_bricks={args.max_bricks_per_batch}) ===", flush=True)
        gpu_bench = benchmark_vertical_scaling_gpu(
            model,
            device=dev,
            depths=(16, 32, 64, 128),
            H=256,
            W=256,
            max_bricks_per_batch=args.max_bricks_per_batch,
            warmup=args.warmup,
            iters=args.iters,
        )
        bw64 = gpu_bench["vertical_halo_bandwidth"]["Z=64_4x16_slabs"]
        print(
            f"  Vertical HALO_Z=3 Bandwidth (Z=64, 3 seams): raw_18ch={bw64['raw_fp16_18ch_kb_per_tick']:.1f} KB/tick, "
            f"zlib_delta_18ch={bw64['zlib_delta_18ch_kb_per_tick']:.1f} KB/tick "
            f"({bw64['zlib_delta_18ch_mb_per_sec_at_10hz']:.3f} MB/s @ 10Hz)",
            flush=True,
        )

        print("\n=== Test 3: Production Z=64 (64x256x256) Daemon Readiness (op=11/9, op=10, op=6) ===", flush=True)
        daemon_z64 = verify_daemon_z64_readiness(
            model,
            device=dev,
            actuator_v3_pt=Path(args.actuator_v3_ckpt),
            acoustic_pt=Path(args.acoustic_ckpt),
        )
        print(f"  Daemon Z=64 readiness: {json.dumps(daemon_z64)}", flush=True)

    metrics = {
        "wave": "26A",
        "title": "Vertical Scaling GPU/ML: 3D Altitude-Varying World Laws, HALO_Z=3 Sharding & Z=16..128 GPU Scaling",
        "checkpoint": str(ckpt_path),
        "world_model_params": int(count_parameters(model)),
        "parity": parity,
        "gpu_benchmark": gpu_bench,
        "daemon_z64_readiness": daemon_z64,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved metrics -> {out_json}", flush=True)

    if gpu_bench:
        out_png = Path(args.out_png)
        render_vertical_scaling_viz(metrics, out_png)
        print(f"Saved chart   -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
