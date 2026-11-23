#!/usr/bin/env python3
"""
Iteration 8A: True Isotropic 3D Z=16 (1m x 1m x 1m) Neural World Field &
Learned NeuralMaterialHead(Z_world, state).

Provides:
  1. True isotropic 3D Z=16 procedural multi-biome world generator (`generate_world_3d_z16`):
       - Canonical single-world tensor `[Z=16, 18, H, W]` (or batched `[B, Z=16, 18, H, W]`)
       - 10 physical species masses + normalized temperature + log total mass + 3D flow
         (flow_x, flow_y, flow_z) + vertical altitude + thermal/mass forcing.
       - Rich multi-biome landmarks built from smooth C2 SDFs (zero 1-cell spikes):
           * Vaulted basalt caves (z=0..5) with arched ceiling, support pillars,
             subterranean stream, mossy cave mouth, and glowing warm magma vents.
           * Sculpted alpine mountains & carved river canyon (z=3..13) with smooth
             rock/soil/snow transitions.
           * True 3-span arched stone & timber bridge (z=4..9) with masonry piers rising
             from the riverbed and open flowing water underneath the arches.
           * Plunging 3D waterfall pouring from an upper glacial lake (z=10) into the
             river canyon (z=4) with mist/steam spray.
           * Biomass-scaled Nordic pines (wood trunk + plant canopy reaching z=13..15).
           * Burning wildfire & buoyant rising smoke/steam plumes + iridescent oil slick.
  2. Context-aware ground-truth material teacher (`compute_material_targets_3d`) and
     auxiliary 3D one-step physics evolution teacher (`compute_physics_evolution_3d`).
  3. Compact 3D Neural World Encoder (`NeuralField3DEncoder`, `Z_world_3d [B, 32, Z, H, W]`)
     with exact mathematically equivalent sparse active-brick inference
     (`forward_sparse_bricks`, supporting `8x8x8` and `16x16x8` bricks with shrinking-halo
     valid 3D convolutions).
  4. Learned `NeuralMaterialHead` (~12.4K params, `1x1x1` + `3x3x3` conv over
     `[Z_world_3d, physical_state]`) predicting two direct-to-GPU 4-channel RGBA fields in `[0, 1]`:
       - `density_mat`    (4 ch): `(solid_density, fluid_density, gas_optical_density, roughness_wetness)`
       - `albedo_thermal` (4 ch): `(albedo_r, albedo_g, albedo_b, blackbody_temp_glow)`
  5. Auxiliary `PhysicsDelta3DHead` predicting one-step 3D `(delta_mass[10], delta_temp[1])`
     from `Z_world_3d` so the latent captures genuine 3D dynamics.
  6. Exact FM3D little-endian binary volume exporter & loader (`save_fm3d_volume`, `load_fm3d_volume`).
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Species & Channel Constants (aligned with FIELD_CONTRACT.md)
# ---------------------------------------------------------------------------

S_WATER = 0
S_ICE = 1
S_STEAM = 2
S_SOIL = 3
S_ROCK = 4
S_WOOD = 5
S_ASH = 6
S_SMOKE = 7
S_OIL = 8
S_PLANT = 9

NUM_SPECIES = 10
IN_CH = 18

CH_TEMP = 10
CH_LOG_MASS = 11
CH_FLOW_X = 12
CH_FLOW_Y = 13
CH_FLOW_Z = 14
CH_ELEV_Z = 15
CH_FORCE_E = 16
CH_FORCE_M = 17

Z_DEPTH_DEFAULT = 16

DENSITY_MAT_NAMES = (
    "solid_density",
    "fluid_density",
    "gas_optical_density",
    "roughness_wetness",
)
ALBEDO_THERMAL_NAMES = (
    "albedo_r",
    "albedo_g",
    "albedo_b",
    "blackbody_temp_glow",
)

FM3D_MAGIC = b"FM3D"


# ---------------------------------------------------------------------------
# Smooth SDF & Procedural Math Helpers (vectorized PyTorch)
# ---------------------------------------------------------------------------


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smin(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """Polynomial smooth minimum (C1 continuous) for smooth terrain/architecture blending."""
    h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _smax(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    return -_smin(-a, -b, k)


def _fbm_2d(
    u: torch.Tensor,
    v: torch.Tensor,
    rng: np.random.Generator,
    octaves: int = 4,
    base_freq: float = 2.0,
) -> torch.Tensor:
    """Smooth C-infinity multi-octave trigonometric fBm (zero 1-cell grid spikes)."""
    out = torch.zeros_like(u)
    amp = 0.55
    freq = base_freq
    norm = 0.0
    for _ in range(octaves):
        ang1 = float(rng.uniform(0.0, 2.0 * math.pi))
        ang2 = ang1 + float(rng.uniform(0.9, 2.2))
        ph1 = float(rng.uniform(0.0, 2.0 * math.pi))
        ph2 = float(rng.uniform(0.0, 2.0 * math.pi))
        c1 = math.cos(ang1) * u + math.sin(ang1) * v
        c2 = math.cos(ang2) * u + math.sin(ang2) * v
        wave = 0.5 * (torch.sin(2.0 * math.pi * freq * c1 + ph1) + torch.cos(2.0 * math.pi * freq * c2 + ph2))
        out = out + amp * wave
        norm += amp
        amp *= 0.5
        freq *= 1.85
    return out / max(norm, 1e-6)


# ---------------------------------------------------------------------------
# True Isotropic 3D Z=16 (1m x 1m x 1m) Procedural World Generator
# ---------------------------------------------------------------------------


def generate_world_3d_z16(
    height: int = 128,
    width: int = 128,
    depth: int = Z_DEPTH_DEFAULT,
    seed: int = 0,
    device: torch.device | str = "cpu",
    return_landmarks: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """
    Procedurally synthesize a rich isotropic 1m x 1m x 1m 3D multi-biome voxel world
    of shape `[Z, 18, H, W]` (default `Z=16`).

    Landmarks (all built from smooth SDFs so terrain, arches, and cave ceilings have
    continuous `nabla solid_density` with zero 1-cell spikes):
      - Vaulted basalt cave (z=0..5) inside the West alpine mountain with arched ceiling,
        basalt support pillars, subterranean stream, mossy mouth, and glowing magma vents.
      - Sculpted alpine mountains & carved river canyon (z=3..13) with smooth rock/soil/snow.
      - True 3-span arched stone & timber bridge (z=4..9) with masonry piers rising from
        the riverbed and open flowing river water underneath the arches.
      - Upper glacial lake (z=10) pouring a plunging 3D waterfall into the river canyon (z=4).
      - Biomass-scaled Nordic pines (wood trunks + plant canopies reaching z=13..15).
      - Active burning wildfire & buoyant rising smoke/steam plumes + floating oil slick.
    """
    dev = torch.device(device)
    rng = np.random.default_rng(seed)

    # World coordinates: isotropic 1m grid in Z, normalized (u, v) in [0, 1] horizontally
    # Physical scale factor for feature radii so 128, 256, 512 keep crisp 1m structures
    z_1d = torch.arange(depth, dtype=torch.float32, device=dev)
    y_1d = torch.linspace(0.0, 1.0, height, dtype=torch.float32, device=dev)
    x_1d = torch.linspace(0.0, 1.0, width, dtype=torch.float32, device=dev)

    # 2D grids [H, W]
    v2, u2 = torch.meshgrid(y_1d, x_1d, indexing="ij")
    # 3D grids [Z, H, W]
    z3 = z_1d[:, None, None].expand(depth, height, width)
    u3 = u2[None, :, :].expand(depth, height, width)
    v3 = v2[None, :, :].expand(depth, height, width)

    # Seeded mild geometry perturbations so held-out worlds are genuinely unseen
    river_phase = float(rng.uniform(-0.45, 0.45))
    river_amp = float(rng.uniform(0.035, 0.065))
    v_bridge = float(rng.uniform(0.47, 0.53))
    v_cave = v_bridge + float(rng.uniform(-0.025, 0.025))
    v_wf = float(rng.uniform(0.24, 0.32))
    v_fire = v_bridge + float(rng.uniform(-0.03, 0.05))
    u_fire = float(rng.uniform(0.76, 0.83))
    v_oil = float(rng.uniform(0.72, 0.80))

    # Smooth noise fields [H, W]
    n_macro = _fbm_2d(u2, v2, rng, octaves=3, base_freq=1.6)
    n_detail = _fbm_2d(u2, v2, rng, octaves=3, base_freq=4.5)
    n_ridge = _fbm_2d(u2, v2, rng, octaves=2, base_freq=2.6)

    # -----------------------------------------------------------------------
    # 1. Sculpted Alpine Mountains & Carved River Canyon (z = 2.8 .. 13.4)
    # -----------------------------------------------------------------------
    # River centerline u_river(v)
    u_river_2d = 0.50 + river_amp * torch.sin(2.0 * math.pi * (v2 - v_bridge) + river_phase)
    # Keep river straight & centered near the bridge span for crisp arch alignment
    bridge_align = torch.exp(-((v2 - v_bridge) / 0.14) ** 2)
    u_river_2d = u_river_2d * (1.0 - 0.65 * bridge_align) + 0.50 * (0.65 * bridge_align)
    d_river_2d = (u2 - u_river_2d).abs()

    # Base valley / bank heightfield
    # West alpine mountain ridge around u ~ 0.16, rising to z ~ 12.8..13.5
    west_peak = 5.6 * torch.exp(-(((u2 - 0.14) / 0.20) ** 2 + ((v2 - 0.48) / 0.42) ** 2))
    west_shoulder = 2.6 * torch.exp(-(((u2 - 0.26) / 0.14) ** 2 + ((v2 - 0.50) / 0.36) ** 2))
    # East alpine highland & glacial basin ridge around (0.80, v_wf)
    east_highland = 4.6 * torch.exp(-(((u2 - 0.84) / 0.22) ** 2 + ((v2 - v_wf) / 0.30) ** 2))
    east_south_hill = 2.4 * torch.exp(-(((u2 - 0.82) / 0.20) ** 2 + ((v2 - 0.78) / 0.26) ** 2))

    h_base = 6.3 + west_peak + west_shoulder + east_highland + east_south_hill + 1.15 * n_macro + 0.35 * n_detail

    # Smooth glacial lake basin carved into the upper East highland around (u_lake, v_wf)
    u_lake = 0.76
    r_lake = torch.sqrt(((u2 - u_lake) / 0.115) ** 2 + ((v2 - v_wf) / 0.095) ** 2 + 1e-8)
    lake_bowl = _smoothstep(1.25, 0.45, r_lake)
    z_lake_water = 10.15
    z_lake_bed = 8.45
    # Ensure highland rim surrounds the lake at z >= 10.8 except at the spillway lip!
    rim_boost = 1.6 * torch.exp(-((r_lake - 1.05) / 0.32) ** 2)
    h_base = h_base + rim_boost
    h_terrain = h_base * (1.0 - lake_bowl) + z_lake_bed * lake_bowl

    # Spillway notch pouring West from the glacial lake (u in 0.61..0.72, v near v_wf)
    spill_v = torch.exp(-((v2 - v_wf) / 0.032) ** 2)
    spill_u = _smoothstep(0.56, 0.63, u2) * _smoothstep(0.77, 0.69, u2)
    spill_mask = spill_v * spill_u
    h_terrain = h_terrain * (1.0 - 0.85 * spill_mask) + (9.25 * (0.85 * spill_mask))

    # Carve smooth river canyon along u_river_2d
    canyon_half_w = 0.125 + 0.015 * n_ridge
    canyon_t = _smoothstep(canyon_half_w, 0.042, d_river_2d)  # 1 at river center, 0 on plateau
    z_river_bed = 2.25 + 0.18 * torch.cos(2.0 * math.pi * v2)
    z_river_water = 3.65
    h_terrain = h_terrain * (1.0 - canyon_t) + z_river_bed * canyon_t

    # Flatten bridge approach pads smoothly at u = 0.355 and u = 0.645 near v_bridge
    z_bridge_deck = 8.15
    app_w = torch.exp(-(((u2 - 0.355) / 0.048) ** 2 + ((v2 - v_bridge) / 0.055) ** 2))
    app_e = torch.exp(-(((u2 - 0.645) / 0.048) ** 2 + ((v2 - v_bridge) / 0.055) ** 2))
    app_mask = (app_w + app_e).clamp(0.0, 1.0)
    h_terrain = h_terrain * (1.0 - 0.72 * app_mask) + (z_bridge_deck - 0.25) * (0.72 * app_mask)

    # Clamp smooth terrain height to [1.6, 13.4]
    h_terrain = h_terrain.clamp(1.6, 13.4)

    # Signed distance to terrain surface (negative = inside solid ground)
    sdf_terrain = z3 - h_terrain[None, :, :]

    # -----------------------------------------------------------------------
    # 2. Vaulted Basalt Cave (z = 0..5) inside the West Alpine Mountain
    # -----------------------------------------------------------------------
    # Tunnel runs from deep inside West mountain (u=0.09) out to canyon wall mouth (u=0.41)
    # with slight organic S-curve in v and a wide vaulted chamber around u=0.20..0.28.
    v_cave_axis_u = v_cave + 0.020 * torch.sin(math.pi * ((u3 - 0.10) / 0.30).clamp(0.0, 1.0))
    # Half-width of cave tunnel expands into a vaulted cavern around u=0.21
    chamber_bulge = 0.046 * torch.exp(-((u3 - 0.21) / 0.075) ** 2)
    cave_half_v = 0.045 + chamber_bulge  # ~11-23m wide at 256x256
    z_cave_floor = 1.25 + 1.05 * _smoothstep(0.27, 0.41, u3)  # rises smoothly to meet riverbed at canyon mouth
    cave_arch_h = 3.55 + 0.45 * torch.exp(-((u3 - 0.21) / 0.08) ** 2)  # ceiling apex z ~ 5.0..5.2

    # Elliptical arch cross-section (flat smooth floor at z_cave_floor, arched vault ceiling)
    dv_cave = (v3 - v_cave_axis_u) / cave_half_v
    dz_cave_up = (z3 - z_cave_floor).clamp(min=0.0) / cave_arch_h
    dz_cave_dn = (z_cave_floor - z3).clamp(min=0.0) / 0.50
    # Longitudinal envelope: closed back wall at u=0.09, open cave mouth emerging at u=0.42+
    du_back = ((0.095 - u3).clamp(min=0.0) / 0.032)
    du_front = ((u3 - 0.440).clamp(min=0.0) / 0.032)

    # Radial arch distance (< 1.0 inside cave void)
    cave_r = torch.sqrt(dv_cave ** 2 + dz_cave_up ** 2 + dz_cave_dn ** 2 + du_back ** 2 + du_front ** 2 + 1e-8)
    # Signed distance to cave void (positive = inside hollow cave, negative = solid rock)
    sdf_cave_void = (1.0 - cave_r) * 2.4

    # Basalt support pillars inside the vaulted cavern (smooth flared columns)
    pillar_specs = [
        (0.185, v_cave + 0.036, 0.013),
        (0.245, v_cave - 0.034, 0.013),
    ]
    pillar_solid = torch.zeros_like(z3)
    for pu, pv, pr in pillar_specs:
        # Slight capital/base flare near ceiling and floor
        flare = 1.0 + 0.26 * torch.exp(-((z3 - z_cave_floor) / 0.8) ** 2) + 0.26 * torch.exp(-((z3 - 4.8) / 0.8) ** 2)
        d_pil = torch.sqrt(((u3 - pu) / (pr * flare)) ** 2 + ((v3 - pv) / (pr * flare)) ** 2 + 1e-8)
        in_z = _smoothstep(0.6, 1.2, z3) * _smoothstep(5.9, 5.0, z3)
        pillar_solid = torch.maximum(pillar_solid, _smoothstep(1.22, 0.65, d_pil) * in_z)

    # Carve cave void out of terrain (solid rock stays everywhere outside cave void, plus pillars!)
    sdf_ground = _smax(sdf_terrain, sdf_cave_void, k=0.40)
    ground_occ = _smoothstep(0.62, -0.62, sdf_ground)
    ground_occ = torch.maximum(ground_occ, pillar_solid)

    # -----------------------------------------------------------------------
    # 3. True Multi-Span Arched Stone & Timber Bridge (z = 4 .. 9)
    # -----------------------------------------------------------------------
    u_br_w, u_br_e = 0.365, 0.635
    br_half_v = 0.034
    br_span_u = _smoothstep(u_br_w - 0.025, u_br_w + 0.012, u3) * _smoothstep(u_br_e + 0.025, u_br_e - 0.012, u3)
    br_width_mask = _smoothstep(br_half_v + 0.010, br_half_v - 0.004, (v3 - v_bridge).abs())
    br_envelope = br_span_u * br_width_mask

    # Gentle architectural crown along the bridge deck (highest at center u=0.50)
    u_rel_br = ((u3 - 0.50) / 0.135).clamp(-1.0, 1.0)
    z_deck_top = z_bridge_deck + 0.45 * (1.0 - u_rel_br ** 2)
    z_deck_bot = z_deck_top - 0.85

    # Roadway timber deck (wood) + structural masonry arch ring (rock)
    deck_wood_occ = (
        _smoothstep(z_deck_top + 0.35, z_deck_top - 0.25, z3)
        * _smoothstep(z_deck_top - 0.85, z_deck_top - 0.35, z3)
        * br_span_u
        * _smoothstep(br_half_v - 0.004, br_half_v - 0.012, (v3 - v_bridge).abs())
    )
    # Stone parapets along N & S edges of the bridge deck (z_deck_top .. z_deck_top + 0.75)
    parapet_band = (
        _smoothstep(br_half_v - 0.013, br_half_v - 0.007, (v3 - v_bridge).abs())
        * _smoothstep(br_half_v + 0.006, br_half_v - 0.002, (v3 - v_bridge).abs())
    )
    parapet_occ = (
        br_span_u
        * parapet_band
        * _smoothstep(z_deck_top + 0.85, z_deck_top + 0.35, z3)
        * _smoothstep(z_deck_bot - 0.25, z_deck_bot + 0.25, z3)
    )

    # Masonry bridge substructure: spans from riverbed z=1.8 up to z_deck_bot + 0.35,
    # with 3 true semicircular/parabolic arches carved cleanly out between piers!
    substruct_raw = (
        br_envelope
        * _smoothstep(z_deck_top - 0.15, z_deck_top - 0.65, z3)
        * _smoothstep(1.6, 2.3, z3)
    )

    # 3 arch spans centered at u = 0.422, 0.500, 0.578 -> masonry piers stand at u = 0.461 and u = 0.539!
    arch_centers = (0.422, 0.500, 0.578)
    arch_half_w = 0.030
    arch_spring_z = 4.10
    arch_crown_z = 7.35
    arch_cutout = torch.zeros_like(z3)
    for ac in arch_centers:
        du_a = (u3 - ac) / arch_half_w
        dz_a = (z3 - arch_spring_z).clamp(min=0.0) / (arch_crown_z - arch_spring_z)
        r_arch = torch.sqrt(du_a ** 2 + dz_a ** 2 + 1e-8)
        arch_void = _smoothstep(1.08, 0.78, r_arch) * _smoothstep(1.4, 2.1, z3)
        arch_cutout = torch.maximum(arch_cutout, arch_void)

    # Keep cave mouth unobstructed beneath the West bridge abutment
    cave_open_guard = 1.0 - _smoothstep(1.08, 0.76, cave_r)
    bridge_masonry_occ = (substruct_raw * (1.0 - arch_cutout) * cave_open_guard).clamp(0.0, 1.0)
    bridge_masonry_occ = torch.maximum(bridge_masonry_occ, parapet_occ)

    # -----------------------------------------------------------------------
    # 4. Species Mass Volumes [10, Z, H, W]
    # -----------------------------------------------------------------------
    mass = torch.zeros((NUM_SPECIES, depth, height, width), dtype=torch.float32, device=dev)
    temp_norm = torch.zeros((depth, height, width), dtype=torch.float32, device=dev)
    flow_x = torch.zeros((depth, height, width), dtype=torch.float32, device=dev)
    flow_y = torch.zeros((depth, height, width), dtype=torch.float32, device=dev)
    flow_z = torch.zeros((depth, height, width), dtype=torch.float32, device=dev)
    force_e = torch.zeros((depth, height, width), dtype=torch.float32, device=dev)
    force_m = torch.zeros((depth, height, width), dtype=torch.float32, device=dev)

    # 4a. Terrain stratigraphy: deep basalt rock, mid-slope soil crust, alpine snow/ice cap
    depth_below_surf = (h_terrain[None, :, :] - z3).clamp(min=0.0)
    # Cave zone is pure basalt rock; open mountain slopes have ~1.3m soil crust below z=9.2
    in_cave_vicinity = _smoothstep(1.8, 0.9, cave_r)
    soil_elev_mask = _smoothstep(10.2, 8.2, z3) * _smoothstep(3.6, 5.2, z3) * (1.0 - 0.85 * in_cave_vicinity)
    soil_crust = _smoothstep(1.9, 0.3, depth_below_surf) * soil_elev_mask
    # Alpine ice/snow cap on high ledges & glacial lake rim (z >= 9.0)
    snow_mask = _smoothstep(8.7, 10.6, z3) * _smoothstep(1.4, 0.15, depth_below_surf) * (1.0 - in_cave_vicinity)
    # Glacial lake frozen rim shelf
    glacial_rim = _smoothstep(1.25, 0.88, r_lake) * _smoothstep(0.55, 0.85, r_lake) * _smoothstep(10.8, 9.8, z3) * _smoothstep(8.7, 9.5, z3)

    rock_frac = (1.0 - 0.72 * soil_crust - 0.65 * snow_mask).clamp(0.18, 1.0)
    soil_frac = (0.72 * soil_crust * (1.0 - snow_mask)).clamp(0.0, 0.85)
    ice_frac = (0.68 * snow_mask).clamp(0.0, 0.85)

    mass[S_ROCK] = (ground_occ * rock_frac * 1.15 + bridge_masonry_occ * 1.25).clamp(0.0, 1.6)
    mass[S_SOIL] = (ground_occ * soil_frac * 1.05).clamp(0.0, 1.2)
    mass[S_ICE] = (ground_occ * ice_frac * 1.05 + glacial_rim[None, :, :] * 0.85 * (1.0 - ground_occ)).clamp(0.0, 1.25)
    mass[S_WOOD] = (deck_wood_occ * 1.10).clamp(0.0, 1.3)

    # Surface groundcover vegetation (alpine grass/moss on mid-elevation soil & cave mouth)
    moss_mouth = (
        torch.exp(-(((u3 - 0.38) / 0.045) ** 2 + ((v3 - v_cave) / 0.065) ** 2 + ((z3 - 3.6) / 2.2) ** 2))
        * _smoothstep(0.95, 0.15, (ground_occ - 0.5).abs() * 2.0)
    )
    meadow_grass = (
        _smoothstep(0.65, -0.1, (z3 - h_terrain[None, :, :]).abs() - 0.35)
        * soil_elev_mask
        * (1.0 - ground_occ * 0.65)
        * (1.0 - br_envelope)
        * _smoothstep(0.06, 0.13, d_river_2d[None, :, :])
    )
    mass[S_PLANT] = (0.35 * meadow_grass + 0.55 * moss_mouth).clamp(0.0, 0.9)

    # Combined solid occupancy so fluids/gases stay cleanly outside solid rock/wood/ice
    solid_pre = (mass[S_ROCK] + mass[S_SOIL] + mass[S_ICE] + mass[S_WOOD]).clamp(0.0, 1.0)
    open_air = (1.0 - solid_pre).clamp(0.0, 1.0)

    # -----------------------------------------------------------------------
    # 5. River Canyon Water, Subterranean Cave Stream, Glacial Lake & 3D Waterfall
    # -----------------------------------------------------------------------
    # 5a. Main river canyon water (flows South underneath the 3 bridge arches!)
    river_water = (
        _smoothstep(z_river_water + 0.45, z_river_water - 0.35, z3)
        * _smoothstep(canyon_half_w[None, :, :] * 0.92, canyon_half_w[None, :, :] * 0.45, d_river_2d[None, :, :])
        * open_air
    )
    # 5b. Subterranean basalt cave stream (flows East along cave floor z=0.8..1.75)
    cave_stream = (
        _smoothstep(1.85, 1.10, z3)
        * _smoothstep(0.65, 0.95, z3)
        * _smoothstep(0.55, 0.15, dv_cave.abs())
        * _smoothstep(0.11, 0.16, u3)
        * _smoothstep(0.46, 0.39, u3)
        * open_air
    )
    # 5c. Upper glacial lake (z = 8.4 .. 10.15)
    glacial_lake = (
        _smoothstep(z_lake_water + 0.40, z_lake_water - 0.35, z3)
        * _smoothstep(8.2, 8.8, z3)
        * _smoothstep(0.95, 0.55, r_lake[None, :, :])
        * open_air
    )
    # 5d. Plunging 3D waterfall sheet pouring from glacial lake spillway (u=0.65, z=10.1)
    #     into the river canyon splash pool (u=0.56, z=4.1)!
    # Parabolic free-fall trajectory u_wf_traj(z) from u=0.66 at z=10.1 to u=0.565 at z=4.1
    t_fall = ((z_lake_water - z3) / (z_lake_water - z_river_water)).clamp(0.0, 1.0)
    u_wf_traj = 0.665 - 0.100 * torch.sqrt(t_fall + 0.02)
    wf_col = (
        torch.exp(-(((u3 - u_wf_traj) / 0.020) ** 2 + ((v3 - v_wf) / 0.026) ** 2))
        * _smoothstep(z_river_water - 0.4, z_river_water + 0.4, z3)
        * _smoothstep(z_lake_water + 0.35, z_lake_water - 0.35, z3)
        * open_air
    )
    # Spillway channel feeding the waterfall lip
    wf_spill_water = (
        spill_mask[None, :, :]
        * _smoothstep(z_lake_water + 0.35, z_lake_water - 0.30, z3)
        * _smoothstep(8.8, 9.4, z3)
        * open_air
    )

    water_total = (river_water * 1.05 + cave_stream * 0.95 + glacial_lake * 1.05 + wf_col * 1.05 + wf_spill_water * 0.95).clamp(0.0, 1.25)
    mass[S_WATER] = water_total

    # 3D Flow field for water
    flow_y = flow_y + 0.55 * river_water
    flow_x = flow_x + 0.42 * cave_stream - 0.45 * (wf_col + wf_spill_water)
    flow_z = flow_z - 0.95 * wf_col
    force_m = force_m + 0.65 * wf_spill_water

    # Waterfall splash mist/steam halo around plunge pool (u ~ 0.565, v ~ v_wf, z = 4.0..7.2)
    wf_mist = (
        torch.exp(-(((u3 - 0.565) / 0.045) ** 2 + ((v3 - v_wf) / 0.050) ** 2 + ((z3 - 5.1) / 1.55) ** 2))
        * open_air
    )
    mass[S_STEAM] = (mass[S_STEAM] + 0.65 * wf_mist).clamp(0.0, 1.1)
    flow_z = flow_z + 0.35 * wf_mist

    # -----------------------------------------------------------------------
    # 6. Biomass-Scaled Nordic Pines (wood trunk + plant canopy up to z=13..15)
    # -----------------------------------------------------------------------
    # Candidate pine locations on valid slopes (well away from cave void, river, bridge, waterfall)
    pine_candidates = [
        (0.27, 0.22, 1.05),
        (0.20, 0.30, 1.10),
        (0.30, 0.36, 0.95),
        (0.29, 0.65, 1.05),
        (0.19, 0.74, 1.12),
        (0.32, 0.82, 0.98),
        (0.68, 0.14, 1.00),
        (0.86, 0.42, 1.05),
        (0.71, 0.62, 1.02),
        (0.84, 0.66, 1.08),
        (0.69, 0.78, 0.96),
        (0.78, 0.86, 1.05),
    ]
    pine_coords: list[dict[str, float]] = []
    for base_u, base_v, base_scale in pine_candidates:
        pu = float(np.clip(base_u + rng.uniform(-0.022, 0.022), 0.08, 0.92))
        pv = float(np.clip(base_v + rng.uniform(-0.022, 0.022), 0.08, 0.92))
        # Skip if too close to bridge, river, waterfall, or wildfire center
        if abs(pv - v_bridge) < 0.065 and 0.33 < pu < 0.67:
            continue
        if math.hypot(pu - u_fire, pv - v_fire) < 0.055:
            continue
        iy = int(round(pv * (height - 1)))
        ix = int(round(pu * (width - 1)))
        z_base = float(h_terrain[iy, ix].item())
        if z_base < 4.8 or z_base > 10.8:
            continue
        # Biomass-scaled tree height reaching z = 12.5 .. 15.0
        tree_h = min(float(depth - 1.1) - z_base, float(rng.uniform(4.8, 6.4)) * base_scale)
        if tree_h < 3.2:
            continue

        # Smooth tapered wood trunk
        r_dist = torch.sqrt((u3 - pu) ** 2 + (v3 - pv) ** 2 + 1e-10)
        z_rel = z3 - z_base
        trunk_r = 0.0085 * (1.0 - 0.45 * (z_rel / max(tree_h, 1.0)).clamp(0.0, 1.0))
        trunk_occ = (
            _smoothstep(trunk_r + 0.0045, trunk_r - 0.002, r_dist)
            * _smoothstep(-0.3, 0.4, z_rel)
            * _smoothstep(tree_h * 0.82, tree_h * 0.62, z_rel)
        )

        # Tiered conical Nordic pine needle canopy (plant)
        canopy_start = tree_h * 0.28
        canopy_t = ((z_rel - canopy_start) / max(tree_h - canopy_start, 1.0)).clamp(0.0, 1.2)
        # Subtle whorl undulations for authentic Nordic conifer silhouette
        whorl = 1.0 + 0.14 * torch.sin(canopy_t * 3.0 * math.pi)
        canopy_r = (0.028 * base_scale * (1.05 - 0.92 * canopy_t).clamp(min=0.003)) * whorl
        canopy_occ = (
            _smoothstep(canopy_r + 0.007, canopy_r - 0.004, r_dist)
            * _smoothstep(canopy_start - 0.45, canopy_start + 0.35, z_rel)
            * _smoothstep(tree_h + 0.35, tree_h - 0.45, z_rel)
        )
        mass[S_WOOD] = torch.maximum(mass[S_WOOD], trunk_occ * 0.95)
        mass[S_PLANT] = torch.maximum(mass[S_PLANT], canopy_occ * 0.88 * (1.0 - 0.7 * trunk_occ))
        pine_coords.append({"u": pu, "v": pv, "z_base": z_base, "z_top": z_base + tree_h})

    # -----------------------------------------------------------------------
    # 7. Glowing Cavern Magma Vents, Burning Wildfire + Rising Smoke, & Oil Slick
    # -----------------------------------------------------------------------
    # 7a. Glowing warm mineral/magma vents recessed along the basalt cave floor/wall
    vent_specs = [
        (0.165, v_cave - 0.036, 1.25),
        (0.265, v_cave + 0.036, 1.35),
    ]
    for vu, vv, vz in vent_specs:
        vent_core = torch.exp(-(((u3 - vu) / 0.018) ** 2 + ((v3 - vv) / 0.018) ** 2 + ((z3 - vz) / 0.95) ** 2))
        vent_plume = (
            torch.exp(-(((u3 - vu) / 0.025) ** 2 + ((v3 - vv) / 0.025) ** 2))
            * _smoothstep(vz - 0.3, vz + 0.5, z3)
            * _smoothstep(4.9, 3.6, z3)
            * open_air
        )
        temp_norm = temp_norm + 8.6 * vent_core + 2.2 * vent_plume
        force_e = force_e + 4.5 * vent_core
        mass[S_STEAM] = (mass[S_STEAM] + 0.42 * vent_plume).clamp(0.0, 1.1)
        flow_z = flow_z + 0.45 * vent_plume

    # 7b. Burning wildfire on the East wooded bench + buoyant rising 3D smoke plume
    iy_f = int(round(v_fire * (height - 1)))
    ix_f = int(round(u_fire * (width - 1)))
    z_fire_base = float(h_terrain[iy_f, ix_f].item())
    r_fire_2d = torch.sqrt((u3 - u_fire) ** 2 + (v3 - v_fire) ** 2 + 1e-10)

    # Burning log/tree pile + scorched ash ring
    fire_fuel = (
        _smoothstep(0.036, 0.010, r_fire_2d)
        * _smoothstep(z_fire_base - 0.3, z_fire_base + 0.3, z3)
        * _smoothstep(z_fire_base + 2.4, z_fire_base + 1.0, z3)
    )
    ash_ring = (
        _smoothstep(0.068, 0.022, r_fire_2d)
        * _smoothstep(1.2, 0.2, (z3 - z_fire_base).abs())
    )
    mass[S_WOOD] = torch.maximum(mass[S_WOOD], fire_fuel * 0.85)
    mass[S_PLANT] = torch.maximum(mass[S_PLANT], ash_ring * 0.45 * (1.0 - fire_fuel))
    mass[S_ASH] = (mass[S_ASH] + 0.72 * ash_ring).clamp(0.0, 1.0)

    # Intense flame core temperature (T ~ 1050-1250K -> temp_norm ~ 7.6-9.6)
    flame_core = torch.exp(-((r_fire_2d / 0.026) ** 2 + ((z3 - (z_fire_base + 0.95)) / 1.35) ** 2))
    flame_halo = torch.exp(-((r_fire_2d / 0.055) ** 2 + ((z3 - (z_fire_base + 1.4)) / 2.2) ** 2))
    temp_norm = temp_norm + 9.2 * flame_core + 2.6 * flame_halo
    force_e = force_e + 5.5 * flame_core

    # Buoyant rising 3D smoke plume drifting slightly East & North up to z=15
    dz_plume = (z3 - (z_fire_base + 0.4)).clamp(min=0.0)
    u_plume = u_fire + 0.0085 * dz_plume
    v_plume = v_fire - 0.0045 * dz_plume
    plume_rad = 0.024 + 0.0065 * dz_plume
    r_plume = torch.sqrt((u3 - u_plume) ** 2 + (v3 - v_plume) ** 2 + 1e-10)
    smoke_plume = (
        torch.exp(-((r_plume / plume_rad) ** 2))
        * _smoothstep(z_fire_base - 0.2, z_fire_base + 1.1, z3)
        * (1.0 - 0.28 * (dz_plume / 9.0).clamp(0.0, 1.0))
        * open_air
    )
    mass[S_SMOKE] = (mass[S_SMOKE] + 0.88 * smoke_plume).clamp(0.0, 1.15)
    temp_norm = temp_norm + 2.1 * smoke_plume * torch.exp(-dz_plume / 4.5)
    flow_z = flow_z + 0.78 * smoke_plume
    flow_x = flow_x + 0.22 * smoke_plume
    flow_y = flow_y - 0.12 * smoke_plume

    # 7c. Iridescent oil slick floating on calm river eddy around (u_river(v_oil), v_oil)
    iy_o = int(round(v_oil * (height - 1)))
    u_oil = float(u_river_2d[iy_o, width // 2].item())
    r_oil = torch.sqrt(((u3 - u_oil) / 0.042) ** 2 + ((v3 - v_oil) / 0.050) ** 2 + 1e-8)
    oil_slick = (
        _smoothstep(1.15, 0.35, r_oil)
        * _smoothstep(z_river_water + 0.45, z_river_water - 0.10, z3)
        * _smoothstep(z_river_water - 0.85, z_river_water - 0.20, z3)
        * open_air
    )
    mass[S_OIL] = (mass[S_OIL] + 0.75 * oil_slick).clamp(0.0, 1.0)
    force_m = force_m + 0.35 * oil_slick

    # -----------------------------------------------------------------------
    # 8. Ambient Temperature Profile (alpine frost cooler at high z, cozy cave warm)
    # -----------------------------------------------------------------------
    # Ambient lapse rate: cooler near snowy peaks (T ~ 266K -> temp_norm = -0.24),
    # cool near glacial lake, ambient ~288K in valley
    temp_ambient = -0.28 * _smoothstep(7.5, 13.0, z3) - 0.22 * _smoothstep(1.1, 0.4, r_lake[None, :, :])
    temp_norm = (temp_norm + temp_ambient).clamp(-1.2, 11.5)

    # -----------------------------------------------------------------------
    # 9. Pack [Z, 18, H, W] Canonical Tensor
    # -----------------------------------------------------------------------
    total_mass = mass.sum(dim=0)
    log_mass = torch.log1p(total_mass.clamp(min=0.0))
    elev_z = (z3 / max(float(depth - 1), 1.0)).clamp(0.0, 1.0)

    state = torch.zeros((depth, IN_CH, height, width), dtype=torch.float32, device=dev)
    state[:, :NUM_SPECIES] = mass.transpose(0, 1)  # [Z, 10, H, W]
    state[:, CH_TEMP] = temp_norm
    state[:, CH_LOG_MASS] = log_mass
    state[:, CH_FLOW_X] = flow_x.clamp(-2.0, 2.0)
    state[:, CH_FLOW_Y] = flow_y.clamp(-2.0, 2.0)
    state[:, CH_FLOW_Z] = flow_z.clamp(-2.0, 2.0)
    state[:, CH_ELEV_Z] = elev_z
    state[:, CH_FORCE_E] = force_e.clamp(-10.0, 10.0)
    state[:, CH_FORCE_M] = force_m.clamp(-4.0, 4.0)

    if not return_landmarks:
        return state

    landmarks: dict[str, Any] = {
        "seed": seed,
        "height": height,
        "width": width,
        "depth": depth,
        "v_bridge": v_bridge,
        "u_bridge_west": u_br_w,
        "u_bridge_east": u_br_e,
        "z_bridge_deck": z_bridge_deck,
        "arch_centers_u": list(arch_centers),
        "pier_u": [0.461, 0.539],
        "v_cave": v_cave,
        "u_cave_mouth": 0.39,
        "u_cave_deep": 0.13,
        "z_cave_floor": 1.05,
        "z_cave_ceiling": 4.85,
        "cave_pillars": [{"u": p[0], "v": p[1], "r": p[2]} for p in pillar_specs],
        "magma_vents": [{"u": v[0], "v": v[1], "z": v[2]} for v in vent_specs],
        "v_waterfall": v_wf,
        "u_lake": u_lake,
        "z_lake_water": z_lake_water,
        "z_river_water": z_river_water,
        "u_fire": u_fire,
        "v_fire": v_fire,
        "z_fire_base": z_fire_base,
        "u_oil": u_oil,
        "v_oil": v_oil,
        "num_pines": len(pine_coords),
        "pines": pine_coords,
    }
    return state, landmarks


def state_zchw_to_bczhw(state: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Convert `[Z, 18, H, W]` or `[B, Z, 18, H, W]` -> `[B, 18, Z, H, W]` + `was_4d` flag."""
    if state.ndim == 4:
        return state.transpose(0, 1).unsqueeze(0).contiguous(), True
    if state.ndim == 5:
        return state.permute(0, 2, 1, 3, 4).contiguous(), False
    raise ValueError(f"Expected 4D [Z,18,H,W] or 5D [B,Z,18,H,W], got {tuple(state.shape)}")


# ---------------------------------------------------------------------------
# Context-Aware 3D Ground-Truth Material & Optics Teacher
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_material_targets_3d(state_bczhw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute physically rich, true 3D context-aware target material fields from
    `state_bczhw` (`[B, 18, Z, H, W]`):
      - `density_mat`    `[B, 4, Z, H, W]` in `[0, 1]`:
          0: solid_density       (smooth C1 solid occupancy for collision & surface raymarching)
          1: fluid_density       (water + oil liquid density)
          2: gas_optical_density (smoke + steam + waterfall mist optical opacity)
          3: roughness_wetness   (0=wet/glossy/specular, 1=dry/matte/scorched)
      - `albedo_thermal` `[B, 4, Z, H, W]` in `[0, 1]`:
          0..2: albedo_rgb       (context-aware RGB: wet darkened rock, scorched charcoal,
                                  sky-exposed alpine snow frost, iridescent oil sheen,
                                  mossy cave mouths, 3D cavity AO, incandescent fire/magma)
          3: blackbody_temp_glow (blackbody thermal glow in [0, 1])

    Uses 3D neighborhood convolutions (3x3x3 and 5x5x5, strictly inside RF radius 3 = 7x7x7)
    so `NeuralField3DEncoder` + `NeuralMaterialHead` must learn genuine 3D spatial context!
    """
    m_water = state_bczhw[:, S_WATER : S_WATER + 1].clamp(min=0.0)
    m_ice = state_bczhw[:, S_ICE : S_ICE + 1].clamp(min=0.0)
    m_steam = state_bczhw[:, S_STEAM : S_STEAM + 1].clamp(min=0.0)
    m_soil = state_bczhw[:, S_SOIL : S_SOIL + 1].clamp(min=0.0)
    m_rock = state_bczhw[:, S_ROCK : S_ROCK + 1].clamp(min=0.0)
    m_wood = state_bczhw[:, S_WOOD : S_WOOD + 1].clamp(min=0.0)
    m_ash = state_bczhw[:, S_ASH : S_ASH + 1].clamp(min=0.0)
    m_smoke = state_bczhw[:, S_SMOKE : S_SMOKE + 1].clamp(min=0.0)
    m_oil = state_bczhw[:, S_OIL : S_OIL + 1].clamp(min=0.0)
    m_plant = state_bczhw[:, S_PLANT : S_PLANT + 1].clamp(min=0.0)

    temp_n = state_bczhw[:, CH_TEMP : CH_TEMP + 1]
    flow_z = state_bczhw[:, CH_FLOW_Z : CH_FLOW_Z + 1]
    elev_z = state_bczhw[:, CH_ELEV_Z : CH_ELEV_Z + 1]

    # 1. Smooth C1 solid, fluid, gas density fields
    solid_raw = (
        1.10 * m_rock
        + 1.05 * m_soil
        + 1.05 * m_ice
        + 1.00 * m_wood
        + 0.45 * m_ash
        + 0.22 * m_plant
    ).clamp(0.0, 1.35)
    # Mild 3x3x3 smoothing blended with raw keeps crisp arches/piers while guaranteeing smooth nabla!
    solid_blur = F.avg_pool3d(solid_raw, kernel_size=3, stride=1, padding=1)
    solid_density = (0.78 * solid_raw + 0.22 * solid_blur).clamp(0.0, 1.0)

    fluid_raw = (1.05 * m_water + 1.10 * m_oil) * (1.0 - 0.85 * solid_density)
    fluid_density = fluid_raw.clamp(0.0, 1.0)

    # Waterfall spray boosts gas optical density slightly around fast plunging water
    wf_spray = ((-flow_z - 0.25).clamp(0.0, 1.0) * (m_water > 0.15).float())
    wf_spray_halo = F.avg_pool3d(wf_spray, kernel_size=3, stride=1, padding=1)
    gas_optical_density = (0.95 * m_smoke + 0.80 * m_steam + 0.45 * wf_spray_halo).clamp(0.0, 1.0)

    # 2. True 3D spatial context fields (all within 5x5x5 = radius 2, well inside RF radius 3!)
    water_5 = F.avg_pool3d((m_water + 0.35 * m_steam).clamp(0.0, 1.0), kernel_size=5, stride=1, padding=2)
    water_3 = F.avg_pool3d(m_water.clamp(0.0, 1.0), kernel_size=3, stride=1, padding=1)
    wetness_ctx = (2.6 * water_3 + 1.8 * water_5 + 0.8 * wf_spray_halo).clamp(0.0, 1.0)

    hot_mask = ((temp_n - 1.8) / 3.5).clamp(0.0, 1.0)
    scorch_ctx = (2.2 * F.avg_pool3d(hot_mask, kernel_size=5, stride=1, padding=2) + 0.7 * m_ash).clamp(0.0, 1.0)

    oil_ctx = (2.5 * F.avg_pool3d(m_oil.clamp(0.0, 1.0), kernel_size=3, stride=1, padding=1)).clamp(0.0, 1.0)

    # Overhead sky exposure (check 1..2 voxels above in Z: cave interiors have solid above -> sky_open ~ 0!)
    # Shift solid_density downward by 1 and 2 voxels in Z (zero at top boundary = open sky)
    solid_up1 = F.pad(solid_density[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1), value=0.0)
    solid_up2 = F.pad(solid_density[:, :, 2:, :, :], (0, 0, 0, 0, 0, 2), value=0.0)
    overhead_solid = torch.maximum(solid_up1, solid_up2)
    sky_exposed = (1.0 - overhead_solid).clamp(0.0, 1.0)

    # Snow-frosted upper ledges: high z + sky exposed + cool temp
    frost_ctx = (
        sky_exposed
        * ((elev_z - 0.54) / 0.24).clamp(0.0, 1.0)
        * (1.0 - scorch_ctx)
        * (1.0 - 0.7 * wetness_ctx)
    ).clamp(0.0, 1.0)

    # Mossy cave mouths & damp sheltered ledges: solid surface near both moisture, plant, and partial overhang
    plant_5 = F.avg_pool3d(m_plant.clamp(0.0, 1.0), kernel_size=5, stride=1, padding=2)
    moss_ctx = (
        (m_rock + m_soil).clamp(0.0, 1.0)
        * (2.4 * plant_5).clamp(0.0, 1.0)
        * (0.4 + 0.6 * wetness_ctx)
        * (1.0 - frost_ctx)
        * (1.0 - scorch_ctx)
    ).clamp(0.0, 1.0)

    # 3D Ambient Occlusion / Cavity Darkening (dense solid neighborhood)
    ao_nbr = F.avg_pool3d(solid_density, kernel_size=3, stride=1, padding=1)
    cavity_ao = (1.0 - 0.30 * ao_nbr).clamp(0.68, 1.0)

    # 3. Roughness / Wetness channel (0 = glossy/wet/liquid, 1 = dry/rough/charcoal)
    w_tot = (
        m_rock + m_soil + m_wood + m_plant + m_ice + m_ash
        + m_water + m_oil + 0.5 * m_smoke + 0.5 * m_steam + 1e-4
    )
    base_rough = (
        0.78 * m_rock
        + 0.86 * m_soil
        + 0.74 * m_wood
        + 0.82 * m_plant
        + 0.18 * m_ice
        + 0.94 * m_ash
        + 0.08 * m_water
        + 0.05 * m_oil
        + 0.40 * m_smoke
        + 0.32 * m_steam
    ) / w_tot
    # Apply 3D context: wetness makes rock/soil/wood glossy; oil slick ultra-glossy; scorch matte!
    is_solid_surf = (m_rock + m_soil + m_wood + m_plant).clamp(0.0, 1.0)
    rough_wet = base_rough - 0.44 * wetness_ctx * is_solid_surf - 0.25 * oil_ctx * (1.0 - is_solid_surf * 0.5)
    rough_wet = rough_wet + 0.22 * scorch_ctx * is_solid_surf - 0.18 * frost_ctx * is_solid_surf
    # Pure empty air defaults to neutral mid-low 0.50
    non_empty = (solid_density + fluid_density + gas_optical_density).clamp(0.0, 1.0)
    roughness_wetness = (rough_wet * non_empty + 0.50 * (1.0 - non_empty)).clamp(0.0, 1.0)

    # 4. Context-aware Albedo RGB [B, 3, Z, H, W]
    # Base linear-ish sRGB material albedos per species
    # [rock, soil, wood, plant, ice, ash, water, oil, smoke, steam]
    def _rgb(r: float, g: float, b: float) -> torch.Tensor:
        return torch.tensor([r, g, b], dtype=state_bczhw.dtype, device=state_bczhw.device).view(1, 3, 1, 1, 1)

    # Water turns frothy white-cyan when plunging in waterfall
    wf_foam = ((-flow_z - 0.22) / 0.65).clamp(0.0, 1.0)
    c_water = _rgb(0.11, 0.42, 0.60) * (1.0 - 0.75 * wf_foam) + _rgb(0.76, 0.91, 0.97) * (0.75 * wf_foam)

    # Iridescent oil sheen when floating on/near water (spatial phase modulated by elev_z & water_3)
    irid_phase = (6.5 * water_3 + 12.0 * oil_ctx + 9.0 * elev_z)
    irid_r = 0.5 + 0.5 * torch.sin(irid_phase)
    irid_g = 0.5 + 0.5 * torch.sin(irid_phase + 2.094)
    irid_b = 0.5 + 0.5 * torch.sin(irid_phase + 4.188)
    irid_rgb = torch.cat([irid_r, irid_g, irid_b], dim=1)
    c_oil = _rgb(0.09, 0.10, 0.13) + 0.34 * oil_ctx * water_3.clamp(0.0, 1.0) * irid_rgb

    albedo_num = (
        m_rock * _rgb(0.36, 0.39, 0.44)
        + m_soil * _rgb(0.40, 0.29, 0.19)
        + m_wood * _rgb(0.54, 0.35, 0.20)
        + m_plant * _rgb(0.14, 0.36, 0.18)
        + m_ice * _rgb(0.78, 0.91, 0.98)
        + m_ash * _rgb(0.16, 0.15, 0.15)
        + m_water * c_water
        + m_oil * c_oil
        + 0.5 * m_smoke * _rgb(0.25, 0.23, 0.24)
        + 0.5 * m_steam * _rgb(0.85, 0.90, 0.95)
    )
    albedo = albedo_num / w_tot

    # 3D context modulations on albedo:
    # a) Wet darkened rock/soil/wood near rivers, waterfalls & cave stream
    albedo = albedo * (1.0 - 0.42 * wetness_ctx * is_solid_surf)
    # b) Mossy green tint around damp cave mouths & shaded stone
    albedo = albedo * (1.0 - 0.62 * moss_ctx) + _rgb(0.16, 0.42, 0.20) * (0.62 * moss_ctx)
    # c) Snow-frosted upper alpine ledges & high pine tips
    albedo = albedo * (1.0 - 0.72 * frost_ctx * is_solid_surf) + _rgb(0.90, 0.94, 0.99) * (0.72 * frost_ctx * is_solid_surf)
    # d) Scorched charcoal around active flames
    albedo = albedo * (1.0 - 0.78 * scorch_ctx * is_solid_surf) + _rgb(0.08, 0.07, 0.07) * (0.78 * scorch_ctx * is_solid_surf)
    # e) 3D cavity ambient occlusion darkening inside caves & under arches
    albedo = albedo * cavity_ao

    # 5. Blackbody Thermal Glow [B, 1, Z, H, W] in [0, 1]
    # temp_n = (T_cK - 29000) / 10000: 1.6 -> 450K, 5.5 -> 840K, 9.5 -> 1240K
    bb_raw = ((temp_n - 1.55) / 6.8).clamp(0.0, 1.0)
    blackbody_glow = (bb_raw ** 1.45).clamp(0.0, 1.0)
    # Incandescent color shift inside glowing fire cores & magma vents
    incand_rgb = _rgb(1.00, 0.46, 0.10) * (1.0 - 0.5 * blackbody_glow) + _rgb(1.00, 0.93, 0.72) * (0.5 * blackbody_glow)
    albedo = albedo * (1.0 - 0.85 * blackbody_glow) + incand_rgb * (0.85 * blackbody_glow)

    # Empty air has neutral dark-sky baseline albedo (weighted by non_empty so empty space stays clean)
    albedo = (albedo * non_empty + _rgb(0.08, 0.11, 0.16) * (1.0 - non_empty)).clamp(0.0, 1.0)

    density_mat = torch.cat(
        [solid_density, fluid_density, gas_optical_density, roughness_wetness],
        dim=1,
    )
    albedo_thermal = torch.cat([albedo, blackbody_glow], dim=1)
    return density_mat, albedo_thermal


# ---------------------------------------------------------------------------
# Auxiliary 3D One-Step Physics Evolution Teacher (delta_mass[10], delta_temp[1])
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_physics_evolution_3d(state_bczhw: torch.Tensor) -> torch.Tensor:
    """
    Fast physically faithful 3D one-step evolution teacher returning
    `delta_phys [B, 11, Z, H, W] = (delta_mass[10], delta_temp[1])`:
      - Gravity plunge of water & oil over open air (z -> z-1)
      - Buoyant rise of steam & smoke unobstructed by solid ceilings (z -> z+1)
      - Horizontal advection along (flow_x, flow_y)
      - 3D 6-neighbor heat conduction + combustion (wood/plant/oil -> ash + smoke + heat)
        + phase change (boil/freeze/melt)
    Used as an auxiliary training target so `Z_world_3d` captures genuine 3D dynamics!
    """
    mass = state_bczhw[:, :NUM_SPECIES].clamp(min=0.0)
    temp_n = state_bczhw[:, CH_TEMP : CH_TEMP + 1]
    fx = state_bczhw[:, CH_FLOW_X : CH_FLOW_X + 1]
    fy = state_bczhw[:, CH_FLOW_Y : CH_FLOW_Y + 1]
    force_e = state_bczhw[:, CH_FORCE_E : CH_FORCE_E + 1]

    solid = (
        mass[:, S_ROCK : S_ROCK + 1]
        + mass[:, S_SOIL : S_SOIL + 1]
        + mass[:, S_ICE : S_ICE + 1]
        + mass[:, S_WOOD : S_WOOD + 1]
    ).clamp(0.0, 1.0)
    open_space = (1.0 - solid).clamp(0.0, 1.0)

    delta_mass = torch.zeros_like(mass)

    # 1. Gravity plunge (water & oil fall z+1 -> z when lower cell is open)
    for sp, rate in ((S_WATER, 0.28), (S_OIL, 0.22)):
        m = mass[:, sp : sp + 1]
        open_dn = F.pad(open_space[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0), value=0.0)
        fall_out = rate * m * open_dn
        fall_out[:, :, 0] = 0.0  # floor boundary
        fall_in = F.pad(fall_out[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1), value=0.0)
        # Mild horizontal advection along (fx, fy)
        m_w = F.pad(m[:, :, :, :, :-1], (1, 0, 0, 0, 0, 0), value=0.0)
        m_e = F.pad(m[:, :, :, :, 1:], (0, 1, 0, 0, 0, 0), value=0.0)
        m_n = F.pad(m[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0), value=0.0)
        m_s = F.pad(m[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0), value=0.0)
        adv = 0.08 * (
            fx.clamp(min=0.0) * (m_w - m)
            + (-fx).clamp(min=0.0) * (m_e - m)
            + fy.clamp(min=0.0) * (m_n - m)
            + (-fy).clamp(min=0.0) * (m_s - m)
        ) * open_space
        delta_mass[:, sp : sp + 1] = fall_in - fall_out + adv

    # 2. Buoyant vertical rise (steam & smoke rise z -> z+1 when upper cell is open)
    buoy_boost = (1.0 + 0.35 * temp_n.clamp(0.0, 5.0)).clamp(1.0, 2.5)
    for sp, rate in ((S_STEAM, 0.24), (S_SMOKE, 0.20)):
        m = mass[:, sp : sp + 1]
        open_up = F.pad(open_space[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1), value=1.0)
        rise_out = (rate * buoy_boost * m * open_up).clamp(max=0.65 * m)
        rise_in = F.pad(rise_out[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0), value=0.0)
        delta_mass[:, sp : sp + 1] = rise_in - rise_out

    # 3. Phase change & exothermic combustion reactions
    # Boil: water -> steam when T > 373K (temp_n > 0.83)
    boil_rate = ((temp_n - 0.83) / 2.0).clamp(0.0, 0.35) * mass[:, S_WATER : S_WATER + 1]
    # Freeze / Melt around T = 273K (temp_n = -0.17)
    freeze_rate = ((-0.17 - temp_n) / 1.5).clamp(0.0, 0.12) * mass[:, S_WATER : S_WATER + 1]
    melt_rate = ((temp_n + 0.17) / 1.5).clamp(0.0, 0.15) * mass[:, S_ICE : S_ICE + 1]
    # Wood & Plant burn when T > 520K (temp_n > 2.3)
    burn_wood = ((temp_n - 2.3) / 4.0).clamp(0.0, 0.18) * mass[:, S_WOOD : S_WOOD + 1]
    burn_plant = ((temp_n - 1.9) / 3.5).clamp(0.0, 0.22) * mass[:, S_PLANT : S_PLANT + 1]
    burn_oil = ((temp_n - 1.6) / 3.5).clamp(0.0, 0.20) * mass[:, S_OIL : S_OIL + 1]

    delta_mass[:, S_WATER : S_WATER + 1] += melt_rate - freeze_rate - boil_rate
    delta_mass[:, S_ICE : S_ICE + 1] += freeze_rate - melt_rate
    delta_mass[:, S_STEAM : S_STEAM + 1] += boil_rate
    delta_mass[:, S_WOOD : S_WOOD + 1] -= burn_wood
    delta_mass[:, S_PLANT : S_PLANT + 1] -= burn_plant
    delta_mass[:, S_OIL : S_OIL + 1] -= burn_oil
    delta_mass[:, S_ASH : S_ASH + 1] += 0.5 * (burn_wood + burn_plant)
    delta_mass[:, S_SMOKE : S_SMOKE + 1] += 0.5 * (burn_wood + burn_plant) + burn_oil

    # 4. 3D 6-neighbor heat conduction + convective vertical plume lift + exothermic heat
    t_avg = F.avg_pool3d(temp_n, kernel_size=3, stride=1, padding=1)
    lap_t = t_avg - temp_n
    t_dn = F.pad(temp_n[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0), value=0.0)
    conv_up = 0.12 * open_space * (t_dn - temp_n).clamp(min=0.0) * (
        0.3 + mass[:, S_SMOKE : S_SMOKE + 1] + mass[:, S_STEAM : S_STEAM + 1]
    ).clamp(max=1.5)
    exo_heat = 1.45 * burn_wood + 1.10 * burn_plant + 1.30 * burn_oil - 0.65 * boil_rate
    delta_temp = 0.22 * lap_t + conv_up + exo_heat + 0.08 * force_e - 0.025 * temp_n

    return torch.cat([delta_mass, delta_temp], dim=1)


# ---------------------------------------------------------------------------
# Compact 3D Neural World Encoder + Learned NeuralMaterialHead + Aux Physics Head
# ---------------------------------------------------------------------------


class NeuralField3DEncoder(nn.Module):
    """
    Lightweight isotropic 3D convolutional trunk producing `Z_world_3d [B, 32, Z, H, W]`.

    Exact spatial receptive-field radius = 2 (`5x5x5` cube):
      - `stem`:   `Conv3d(18, 32, kernel_size=3, padding=1)`          (halo +1)
      - `block1`: `GELU -> Conv3d(32, 32, 3, padding=1) -> GELU -> Conv3d(32, 32, 1)` + res gamma1 (halo +1)
      - `block2`: `GELU -> Conv3d(32, 32, 1) -> GELU -> Conv3d(32, 32, 1)` + res gamma2 (halo +0)

    Combined with `NeuralMaterialHead` (which has a single `3x3x3` conv, halo +1),
    total exact halo = 3 (`7x7x7` 3D receptive field), enabling mathematically identical
    shrinking-halo sparse `8x8x8` / `16x16x8` brick inference (`max_abs_diff < 1e-5` FP32)!
    """

    def __init__(self, in_ch: int = IN_CH, latent_ch: int = 32):
        super().__init__()
        self.in_ch = in_ch
        self.latent_ch = latent_ch
        self.encoder_halo = 2

        self.stem = nn.Conv3d(in_ch, latent_ch, kernel_size=3, padding=1, bias=True)

        self.b1_conv3 = nn.Conv3d(latent_ch, latent_ch, kernel_size=3, padding=1, bias=True)
        self.b1_pw = nn.Conv3d(latent_ch, latent_ch, kernel_size=1, bias=True)
        self.gamma1 = nn.Parameter(torch.full((1, latent_ch, 1, 1, 1), 0.25, dtype=torch.float32))

        self.b2_pw1 = nn.Conv3d(latent_ch, latent_ch, kernel_size=1, bias=True)
        self.b2_pw2 = nn.Conv3d(latent_ch, latent_ch, kernel_size=1, bias=True)
        self.gamma2 = nn.Parameter(torch.full((1, latent_ch, 1, 1, 1), 0.25, dtype=torch.float32))

    def forward(self, state_bczhw: torch.Tensor) -> torch.Tensor:
        """Dense forward: `[B, 18, Z, H, W]` -> `Z_world_3d [B, 32, Z, H, W]`."""
        h = self.stem(state_bczhw)
        r1 = self.b1_pw(F.gelu(self.b1_conv3(F.gelu(h))))
        h = h + self.gamma1 * r1
        r2 = self.b2_pw2(F.gelu(self.b2_pw1(F.gelu(h))))
        z_world = F.gelu(h + self.gamma2 * r2)
        return z_world

    def forward_valid_shrinking(
        self,
        patches: torch.Tensor,
        mask_post_stem: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Exact shrinking-halo valid-convolution encoder pass on padded brick patches:
          Input:  `[K, 18, Pz, Py, Px]`
          Output: `[K, 32, Pz - 4, Py - 4, Px - 4]` (shrinks by `2 * encoder_halo = 4` per axis)
        If `mask_post_stem` (`[K, 1, Pz - 2, Py - 2, Px - 2]` in `{0, 1}`) is provided,
        out-of-volume boundary voxels are zeroed after `stem` so `b1_conv3` matches dense
        `Conv3d(padding=1)` to machine precision even on outer world-boundary bricks!
        """
        h = F.conv3d(patches, self.stem.weight, self.stem.bias, padding=0)
        h_act = F.gelu(h)
        if mask_post_stem is not None:
            h_act = h_act * mask_post_stem
        c1 = F.conv3d(h_act, self.b1_conv3.weight, self.b1_conv3.bias, padding=0)
        r1 = self.b1_pw(F.gelu(c1))
        h = h[:, :, 1:-1, 1:-1, 1:-1] + self.gamma1 * r1
        r2 = self.b2_pw2(F.gelu(self.b2_pw1(F.gelu(h))))
        return F.gelu(h + self.gamma2 * r2)


class NeuralMaterialHead(nn.Module):
    """
    Compact learned `NeuralMaterialHead(Z_world_3d, physical_state)` (~12.4K params):
      - `1x1x1` fusion over `[Z_world_3d (32 ch), physical_state (18 ch)]` -> `20` ch
      - `3x3x3` residual 3D spatial refinement (`20 -> 20`)
      - `1x1x1` channel mix (`20 -> 20`)
      - Two direct-to-GPU 4-channel RGBA heads (sigmoid -> exact `[0, 1]`):
          * `density_mat`    (`4` ch): `(solid_density, fluid_density, gas_optical_density, roughness_wetness)`
          * `albedo_thermal` (`4` ch): `(albedo_r, albedo_g, albedo_b, blackbody_temp_glow)`
    """

    def __init__(self, latent_ch: int = 32, state_ch: int = IN_CH, hidden_ch: int = 20):
        super().__init__()
        self.latent_ch = latent_ch
        self.state_ch = state_ch
        self.hidden_ch = hidden_ch
        self.head_halo = 1

        self.fuse_pw = nn.Conv3d(latent_ch + state_ch, hidden_ch, kernel_size=1, bias=True)
        self.spatial_conv3 = nn.Conv3d(hidden_ch, hidden_ch, kernel_size=3, padding=1, bias=True)
        self.refine_pw = nn.Conv3d(hidden_ch, hidden_ch, kernel_size=1, bias=True)
        self.out_density_mat = nn.Conv3d(hidden_ch, 4, kernel_size=1, bias=True)
        self.out_albedo_thermal = nn.Conv3d(hidden_ch, 4, kernel_size=1, bias=True)

    def forward(
        self,
        z_world_3d: torch.Tensor,
        state_bczhw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense forward returning `(density_mat [B,4,Z,H,W], albedo_thermal [B,4,Z,H,W])` in `[0, 1]`."""
        x = torch.cat([z_world_3d, state_bczhw], dim=1)
        h = F.gelu(self.fuse_pw(x))
        h = h + F.gelu(self.spatial_conv3(h))
        h = F.gelu(self.refine_pw(h))
        density_mat = torch.sigmoid(self.out_density_mat(h))
        albedo_thermal = torch.sigmoid(self.out_albedo_thermal(h))
        return density_mat, albedo_thermal

    def forward_valid_shrinking(
        self,
        z_world_patch: torch.Tensor,
        state_patch: torch.Tensor,
        mask_post_fuse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Exact shrinking-halo valid-convolution pass on padded brick patches:
          Input:  `z_world_patch [K, 32, Bz+2, By+2, Bx+2]`, `state_patch [K, 18, Bz+2, By+2, Bx+2]`
          Output: `(density_mat [K, 4, Bz, By, Bx], albedo_thermal [K, 4, Bz, By, Bx])`
        """
        x = torch.cat([z_world_patch, state_patch], dim=1)
        h = F.gelu(self.fuse_pw(x))
        h_in = h * mask_post_fuse if mask_post_fuse is not None else h
        s = F.gelu(F.conv3d(h_in, self.spatial_conv3.weight, self.spatial_conv3.bias, padding=0))
        h = h[:, :, 1:-1, 1:-1, 1:-1] + s
        h = F.gelu(self.refine_pw(h))
        density_mat = torch.sigmoid(self.out_density_mat(h))
        albedo_thermal = torch.sigmoid(self.out_albedo_thermal(h))
        return density_mat, albedo_thermal


class PhysicsDelta3DHead(nn.Module):
    """
    Auxiliary 3D one-step physics evolution head (`Z_world_3d -> (delta_mass[10], delta_temp[1])`)
    so the shared `Z_world_3d` latent encodes genuine 3D dynamics (gravity, buoyancy, heat, reactions).
    """

    def __init__(self, latent_ch: int = 32, hidden_ch: int = 32, out_ch: int = NUM_SPECIES + 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_ch, hidden_ch, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv3d(hidden_ch, out_ch, kernel_size=1, bias=True),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, z_world_3d: torch.Tensor) -> torch.Tensor:
        return self.net(z_world_3d)


class NeuralWorldField3D(nn.Module):
    """
    Unified Iteration 8A True Isotropic 3D Z=16 Neural World Field:
      - `encoder`:      `NeuralField3DEncoder` (`[B, 18, Z, H, W]` -> `Z_world_3d [B, 32, Z, H, W]`)
      - `material_head`: `NeuralMaterialHead`  (`(Z_world_3d, state)` -> `(density_mat, albedo_thermal)`)
      - `physics_head`:  `PhysicsDelta3DHead`  (`Z_world_3d` -> `delta_phys [B, 11, Z, H, W]`)

    Supports both dense full-volume inference and exact shrinking-halo sparse active-brick
    inference (`forward_sparse_bricks`) for `8x8x8` or `16x16x8` bricks!
    """

    def __init__(
        self,
        in_ch: int = IN_CH,
        latent_ch: int = 32,
        mat_hidden_ch: int = 20,
    ):
        super().__init__()
        self.encoder = NeuralField3DEncoder(in_ch=in_ch, latent_ch=latent_ch)
        self.material_head = NeuralMaterialHead(latent_ch=latent_ch, state_ch=in_ch, hidden_ch=mat_hidden_ch)
        self.physics_head = PhysicsDelta3DHead(latent_ch=latent_ch, hidden_ch=latent_ch, out_ch=NUM_SPECIES + 1)
        self.total_halo = self.encoder.encoder_halo + self.material_head.head_halo  # 3

    def forward(
        self,
        state: torch.Tensor,
        return_latent: bool = False,
        return_physics: bool = False,
        is_bczhw: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Dense forward pass. Accepts `[Z, 18, H, W]`, `[B, Z, 18, H, W]`, or
        `[B, 18, Z, H, W]` (when `is_bczhw=True`).
        Returns dict with `density_mat`, `albedo_thermal`, and optionally `z_world_3d`, `delta_phys`.
        """
        if is_bczhw:
            x = state
            was_4d = False
        else:
            x, was_4d = state_zchw_to_bczhw(state)

        z_world = self.encoder(x)
        density_mat, albedo_thermal = self.material_head(z_world, x)

        out: dict[str, torch.Tensor] = {
            "density_mat": density_mat.squeeze(0) if was_4d else density_mat,
            "albedo_thermal": albedo_thermal.squeeze(0) if was_4d else albedo_thermal,
        }
        if return_latent:
            out["z_world_3d"] = z_world.squeeze(0) if was_4d else z_world
        if return_physics:
            dp = self.physics_head(z_world)
            out["delta_phys"] = dp.squeeze(0) if was_4d else dp
        return out

    @torch.inference_mode()
    def forward_sparse_bricks(
        self,
        state_bczhw: torch.Tensor,
        active_brick_coords: torch.Tensor,
        brick_size: tuple[int, int, int] = (8, 16, 16),
        cache_density: torch.Tensor | None = None,
        cache_albedo: torch.Tensor | None = None,
        cache_latent: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Exact shrinking-halo sparse active-brick 3D inference!

        Args:
            state_bczhw:         `[1, 18, Z, H, W]` volume tensor (FP16 or FP32).
            active_brick_coords: `[K, 3]` integer tensor of `(bz, by, bx)` brick indices.
            brick_size:          `(Bz, By, Bx)` — e.g. `(8, 16, 16)` for `16x16x8` bricks
                                 or `(8, 8, 8)` for `8x8x8` bricks.
            cache_density:       Optional persistent `[1, 4, Z, H, W]` cache updated in-place.
            cache_albedo:        Optional persistent `[1, 4, Z, H, W]` cache updated in-place.
            cache_latent:        Optional persistent `[1, 32, Z, H, W]` cache updated in-place.
            return_latent:       Whether to scatter & return `Z_world_3d`.

        Returns:
            `(density_mat [1, 4, Z, H, W], albedo_thermal [1, 4, Z, H, W], z_world_3d or None)`
            Every active brick is mathematically identical to dense forward (`max_abs_diff < 1e-5` FP32)!
        """
        assert state_bczhw.ndim == 5 and state_bczhw.shape[0] == 1, (
            f"Expected [1, 18, Z, H, W], got {tuple(state_bczhw.shape)}"
        )
        _, C, Z, H, W = state_bczhw.shape
        Bz, By, Bx = brick_size
        assert Z % Bz == 0 and H % By == 0 and W % Bx == 0, (
            f"Volume {(Z, H, W)} must be divisible by brick_size {brick_size}"
        )

        if cache_density is None:
            cache_density = state_bczhw.new_zeros((1, 4, Z, H, W))
        if cache_albedo is None:
            cache_albedo = state_bczhw.new_zeros((1, 4, Z, H, W))
        if return_latent and cache_latent is None:
            cache_latent = state_bczhw.new_zeros((1, self.encoder.latent_ch, Z, H, W))

        K = int(active_brick_coords.shape[0])
        if K == 0:
            return cache_density, cache_albedo, (cache_latent if return_latent else None)

        halo = self.total_halo  # 3
        Pz, Py, Px = Bz + 2 * halo, By + 2 * halo, Bx + 2 * halo

        # Zero-pad full volume identically to Conv3d(padding=1) boundary semantics
        x_pad = F.pad(state_bczhw, (halo, halo, halo, halo, halo, halo), mode="constant", value=0.0)

        # Strided 3D sliding-brick view -> zero-copy index gather of [K, C, Pz, Py, Px]
        # unfold dims 2 (Z), 3 (H), 4 (W) with step (Bz, By, Bx)
        unfolded = (
            x_pad.squeeze(0)
            .unfold(1, Pz, Bz)
            .unfold(2, Py, By)
            .unfold(3, Px, Bx)
        )  # [C, nz, ny, nx, Pz, Py, Px]
        bz_i = active_brick_coords[:, 0]
        by_i = active_brick_coords[:, 1]
        bx_i = active_brick_coords[:, 2]
        # Gather active patches -> [C, K, Pz, Py, Px] -> [K, C, Pz, Py, Px]
        patches = unfolded[:, bz_i, by_i, bx_i].permute(1, 0, 2, 3, 4).contiguous()

        # Lightweight separable 1D in-bounds masks so boundary bricks match dense Conv3d(padding=1) exactly!
        dev = state_bczhw.device
        dt = state_bczhw.dtype
        oz1 = torch.arange(-2, Bz + 2, device=dev)
        oy1 = torch.arange(-2, By + 2, device=dev)
        ox1 = torch.arange(-2, Bx + 2, device=dev)
        gz1 = bz_i[:, None] * Bz + oz1[None, :]
        gy1 = by_i[:, None] * By + oy1[None, :]
        gx1 = bx_i[:, None] * Bx + ox1[None, :]
        mz1 = ((gz1 >= 0) & (gz1 < Z)).to(dt)[:, None, :, None, None]
        my1 = ((gy1 >= 0) & (gy1 < H)).to(dt)[:, None, None, :, None]
        mx1 = ((gx1 >= 0) & (gx1 < W)).to(dt)[:, None, None, None, :]
        mask_post_stem = mz1 * my1 * mx1  # [K, 1, Bz+4, By+4, Bx+4]
        mask_post_fuse = mask_post_stem[:, :, 1:-1, 1:-1, 1:-1]  # [K, 1, Bz+2, By+2, Bx+2]

        # Shrinking-halo exact valid 3D convolutions:
        # Encoder: (Pz, Py, Px) -> (Bz+2, By+2, Bx+2)
        z_patch = self.encoder.forward_valid_shrinking(patches, mask_post_stem=mask_post_stem)
        # Align raw state patch to (Bz+2, By+2, Bx+2) for material head
        enc_h = self.encoder.encoder_halo  # 2
        s_patch = patches[:, :, enc_h:-enc_h, enc_h:-enc_h, enc_h:-enc_h]
        # Material head: (Bz+2, By+2, Bx+2) -> (Bz, By, Bx)
        d_bricks, a_bricks = self.material_head.forward_valid_shrinking(
            z_patch, s_patch, mask_post_fuse=mask_post_fuse
        )

        # Vectorized scatter back into persistent brick-tiled views of cache tensors
        nz, ny, nx = Z // Bz, H // By, W // Bx
        d_view = cache_density.view(4, nz, Bz, ny, By, nx, Bx).permute(1, 3, 5, 0, 2, 4, 6)
        a_view = cache_albedo.view(4, nz, Bz, ny, By, nx, Bx).permute(1, 3, 5, 0, 2, 4, 6)
        d_view[bz_i, by_i, bx_i] = d_bricks
        a_view[bz_i, by_i, bx_i] = a_bricks

        if return_latent and cache_latent is not None:
            z_center = z_patch[:, :, 1:-1, 1:-1, 1:-1]
            z_view = cache_latent.view(self.encoder.latent_ch, nz, Bz, ny, By, nx, Bx).permute(1, 3, 5, 0, 2, 4, 6)
            z_view[bz_i, by_i, bx_i] = z_center

        return cache_density, cache_albedo, (cache_latent if return_latent else None)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------
# Exact FM3D Little-Endian Binary Exporter & Loader
# ---------------------------------------------------------------------------


def save_fm3d_volume(
    path: str | Path,
    density_mat: torch.Tensor | np.ndarray,
    albedo_thermal: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """
    Export neural-predicted 3D brick volume to little-endian binary FM3D format:
      - `magic`:          `[u8; 4] = b"FM3D"`
      - `width`:          `u32` (e.g. 256)
      - `height`:         `u32` (e.g. 256)
      - `depth`:          `u32` (e.g. 16)
      - `density_mat`:    `depth * height * width * 4` bytes (`u8` RGBA8, Z-major `[z][y][x][c]`)
      - `albedo_thermal`: `depth * height * width * 4` bytes (`u8` RGBA8, Z-major `[z][y][x][c]`)

    Accepts `[4, Z, H, W]` or `[1, 4, Z, H, W]` float tensors/arrays in `[0, 1]`.
    """
    if isinstance(density_mat, torch.Tensor):
        dm = density_mat.detach().float().cpu().numpy()
    else:
        dm = np.asarray(density_mat, dtype=np.float32)
    if isinstance(albedo_thermal, torch.Tensor):
        at = albedo_thermal.detach().float().cpu().numpy()
    else:
        at = np.asarray(albedo_thermal, dtype=np.float32)

    if dm.ndim == 5:
        assert dm.shape[0] == 1
        dm = dm[0]
    if at.ndim == 5:
        assert at.shape[0] == 1
        at = at[0]
    assert dm.ndim == 4 and dm.shape[0] == 4, f"Expected [4,Z,H,W], got {dm.shape}"
    assert at.shape == dm.shape, f"Shape mismatch: {dm.shape} vs {at.shape}"

    _, depth, height, width = dm.shape

    # Convert [4, Z, H, W] -> Z-major [Z, H, W, 4] u8 RGBA8
    dm_u8 = np.rint(np.clip(np.transpose(dm, (1, 2, 3, 0)), 0.0, 1.0) * 255.0).astype(np.uint8)
    at_u8 = np.rint(np.clip(np.transpose(at, (1, 2, 3, 0)), 0.0, 1.0) * 255.0).astype(np.uint8)
    dm_u8 = np.ascontiguousarray(dm_u8)
    at_u8 = np.ascontiguousarray(at_u8)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<4sIII", FM3D_MAGIC, int(width), int(height), int(depth))
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(dm_u8.tobytes())
        f.write(at_u8.tobytes())

    expected_bytes = 16 + 2 * depth * height * width * 4
    actual_bytes = out_path.stat().st_size
    assert actual_bytes == expected_bytes, f"Size mismatch: {actual_bytes} != {expected_bytes}"
    return {
        "path": str(out_path),
        "magic": FM3D_MAGIC.decode("ascii"),
        "width": int(width),
        "height": int(height),
        "depth": int(depth),
        "bytes": int(actual_bytes),
    }


def load_fm3d_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Load and validate an FM3D little-endian binary volume, returning `(density_u8, albedo_u8, meta)` in `[Z, H, W, 4]`."""
    raw = Path(path).read_bytes()
    if len(raw) < 16:
        raise ValueError(f"FM3D file too short: {len(raw)} bytes")
    magic, width, height, depth = struct.unpack("<4sIII", raw[:16])
    if magic != FM3D_MAGIC:
        raise ValueError(f"Bad FM3D magic: {magic!r}")
    payload_each = depth * height * width * 4
    expected = 16 + 2 * payload_each
    if len(raw) != expected:
        raise ValueError(f"FM3D byte size {len(raw)} != expected {expected}")
    dm_u8 = np.frombuffer(raw, dtype=np.uint8, count=payload_each, offset=16).reshape(depth, height, width, 4).copy()
    at_u8 = np.frombuffer(raw, dtype=np.uint8, count=payload_each, offset=16 + payload_each).reshape(depth, height, width, 4).copy()
    return dm_u8, at_u8, {"width": width, "height": height, "depth": depth, "bytes": len(raw)}
