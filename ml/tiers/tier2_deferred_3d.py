#!/usr/bin/env python3
"""
Tier 2 — Real 3D Free-Camera G-Buffer -> Neural Deferred Renderer (`tier2_deferred_3d.onnx`).

EXACT ONNX CONTRACT FOR RUST 3D CLIENT:
  Input:  `gbuf`   float32 [B, 18, H, W] (dynamic B, H, W; trained at 256x256, works at 512x512!)
    0..10: Compositional species fractions at ray-hit cell
           (water, ice, steam, soil, rock, wood, ash, smoke, oil, plant), 0 for sky
    10:    temp_norm (0..1)
    11:    burning (1.0 or 0.0)
    12:    critter_density (0..1)
    13:    depth_norm (0..1, 1.0 = sky)
    14:    normal_x (-1..1 world-space surface normal X)
    15:    normal_y (-1..1 world-space surface normal Y / up)
    16:    normal_z (-1..1 world-space surface normal Z)
    17:    world_height_norm (0..1 surface elevation + rock/tree height)
  Output: `rgb_3d` float32 [B, 3, H, W] in [0, 1] (Sigmoid).
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (
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
    compute_burning_mask,
    count_parameters,
    decode_aplus_wire,
    encode_aplus_wire,
    gaussian_blur,
    shift2d,
)

# Physical 3D world vertical scale (X, Z in [0, 1], Y in [0, HEIGHT_WORLD_SCALE])
HEIGHT_WORLD_SCALE = 0.36
DEPTH_MAX = 2.60


# =============================================================================
# 1. Realistic 3D Heightfield & Scene Field Builder from Frostmarsh A+ Terrain
# =============================================================================
@torch.no_grad()
def build_3d_terrain_fields(
    mass: torch.Tensor,
    t_ck: torch.Tensor,
    flow_e: torch.Tensor,
    flow_s: torch.Tensor,
    critter_norm: torch.Tensor,
    elev_norm: torch.Tensor,
    t_sec: torch.Tensor,
    grid_res: int = 256,
) -> dict[str, torch.Tensor]:
    """
    Build 256x256 3D heightfield, world-space surface normals, A+ compositional species,
    and volumetric firelight bounce map from 128x128 Frostmarsh A+ state.

    Realistic 3D heights (`world_height_norm` in [0, 1]):
      - river trough (water): 0.08
      - ice shelf:            0.14
      - banks (soil/ash/oil): 0.26 (+ gentle terrain elevation relief)
      - marsh plants:         0.33
      - pine trees (wood):    0.58 (+ conical conifer canopy crown relief)
      - rock cliffs:          0.78 (+ faceted craggy ridge relief)
    """
    b, _, h, w = mass.shape
    device = mass.device

    # Pass through exact A+ wire codec so G-buffer species match client A+ wire!
    wire = encode_aplus_wire(mass, t_ck, flow_e, flow_s, critter_norm)
    aplus = decode_aplus_wire(wire, t_sec)  # [B, 17, 128, 128]

    sp_128 = aplus[:, 0:10]
    temp_128 = aplus[:, 10:11]
    burn_128 = aplus[:, 11:12]
    crit_128 = aplus[:, 14:15]

    # Upsample to smooth 256x256 terrain fields for crisp 3D silhouettes
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=(grid_res, grid_res), mode="bilinear", align_corners=True)

    sp = _up(sp_128)
    sp_sum = sp.sum(dim=1, keepdim=True)
    sp = sp / torch.clamp(sp_sum, min=1e-4)
    temp = _up(temp_128)
    burn = (_up(burn_128) > 0.35).float()
    burn_soft = _up(burn_128)
    crit = _up(crit_128)
    elev = _up(elev_norm)

    f_water = sp[:, S_WATER:S_WATER + 1]
    f_ice = sp[:, S_ICE:S_ICE + 1]
    f_soil = sp[:, S_SOIL:S_SOIL + 1]
    f_rock = sp[:, S_ROCK:S_ROCK + 1]
    f_wood = sp[:, S_WOOD:S_WOOD + 1]
    f_ash = sp[:, S_ASH:S_ASH + 1]
    f_oil = sp[:, S_OIL:S_OIL + 1]
    f_plant = sp[:, S_PLANT:S_PLANT + 1]
    f_steam = sp[:, S_STEAM:S_STEAM + 1]
    f_smoke = sp[:, S_SMOKE:S_SMOKE + 1]

    # Deterministic procedural 3D micro-relief for faceted cliffs, conical pines, and gentle water waves
    yy = torch.linspace(0.0, 1.0, grid_res, device=device).view(1, 1, grid_res, 1)
    xx = torch.linspace(0.0, 1.0, grid_res, device=device).view(1, 1, 1, grid_res)
    crag = (
        0.5 * torch.sin(xx * 42.0 + yy * 29.0) * torch.cos(xx * 27.0 - yy * 47.0)
        + 0.3 * torch.sin(xx * 85.0 - yy * 73.0)
    )
    pine_crown = 0.5 + 0.5 * torch.sin(xx * 96.0 + 0.5) * torch.sin(yy * 96.0 + 0.3)
    water_wave = 0.5 * torch.sin(xx * 110.0 + yy * 65.0) * torch.cos(xx * 55.0 - yy * 95.0)

    # Exact target heights from spec:
    # river=0.08, ice=0.14, banks=0.26, plants=0.33, pines=0.58, rock cliffs=0.78
    solid_w = f_water + f_ice + f_soil + f_ash + f_oil + f_plant + f_wood + f_rock
    base_h = (
        0.08 * f_water
        + 0.14 * f_ice
        + 0.26 * (f_soil + f_ash + f_oil)
        + 0.33 * f_plant
        + (0.58 + 0.08 * (pine_crown - 0.5)) * f_wood
        + (0.78 + 0.11 * crag) * f_rock
    ) / torch.clamp(solid_w, min=0.15)
    # Subtle natural topographic slope from sim elevation on land (keeps river trough cleanly around 0.08)
    land_mask = torch.clamp(1.0 - f_water - 0.8 * f_ice, 0.0, 1.0)
    h_norm = base_h + 0.06 * (elev - 0.35) * land_mask + 0.008 * water_wave * f_water
    # Gentle smoothing on transitions so cliffs have steep sloped walls rather than 1-pixel spikes
    h_norm = torch.clamp(gaussian_blur(h_norm, 0.65), 0.04, 0.98)

    # Exact analytical world-space surface normals (X=east, Y=up, Z=south)
    # dx_world = 1.0 / (grid_res - 1), dy_world = dh_norm * HEIGHT_WORLD_SCALE
    dx_w = 1.0 / (grid_res - 1)
    dh_dx = (shift2d(h_norm, 0, -1) - shift2d(h_norm, 0, 1)) * (HEIGHT_WORLD_SCALE / (2.0 * dx_w))
    dh_dz = (shift2d(h_norm, -1, 0) - shift2d(h_norm, 1, 0)) * (HEIGHT_WORLD_SCALE / (2.0 * dx_w))
    nx = -dh_dx
    ny = torch.ones_like(h_norm)
    nz = -dh_dz
    n_len = torch.sqrt(nx * nx + ny * ny + nz * nz + 1e-8)
    normal = torch.cat([nx / n_len, ny / n_len, nz / n_len], dim=1)  # [B, 3, G, G]

    # Volumetric 3D firelight irradiance field + directional firelight gradient (for bounce onto cliffs/ice/water!)
    fire_emit = burn_soft * (0.85 + 0.55 * temp) + torch.clamp((temp - 0.42) / 0.48, 0.0, 1.0) * 0.35
    fire_tight = gaussian_blur(fire_emit, 3.2)
    fire_med = gaussian_blur(fire_emit, 9.0)
    fire_wide = gaussian_blur(fire_emit, 20.0)
    fire_irr = torch.clamp(fire_tight * 0.55 + fire_med * 0.50 + fire_wide * 0.45, 0.0, 1.6)
    # Direction FROM surface point TOWARD nearest fire peak (gradient of smooth fire field)
    fgx = shift2d(fire_med + fire_wide, 0, -2) - shift2d(fire_med + fire_wide, 0, 2)
    fgz = shift2d(fire_med + fire_wide, -2, 0) - shift2d(fire_med + fire_wide, 2, 0)
    fglen = torch.sqrt(fgx * fgx + fgz * fgz + 1e-6)
    fire_dir_x = fgx / (fglen + 0.02)
    fire_dir_z = fgz / (fglen + 0.02)

    # Frost glow field
    frost_emit = torch.clamp((0.095 - temp) / 0.095, 0.0, 1.0) * (0.45 + 0.65 * f_ice)
    frost_irr = gaussian_blur(frost_emit, 5.0) * 0.55 + gaussian_blur(frost_emit, 12.0) * 0.45

    return {
        "sp": sp,
        "temp": temp,
        "burn": burn,
        "burn_soft": burn_soft,
        "crit": crit,
        "h_norm": h_norm,
        "normal": normal,
        "fire_irr": fire_irr,
        "fire_dir_x": fire_dir_x,
        "fire_dir_z": fire_dir_z,
        "frost_irr": frost_irr,
        "f_steam": f_steam,
        "f_smoke": f_smoke,
    }


# =============================================================================
# 2. Fast GPU-Vectorized 3D Heightfield Ray-Marcher & Path-Shaded Target Renderer
# =============================================================================
def _sample_field_2d(field: torch.Tensor, px: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
    """Bilinear sample `field` [B, C, G, G] at world (px, pz) in [0, 1] -> [B, C, H, W]."""
    grid = torch.stack([px * 2.0 - 1.0, pz * 2.0 - 1.0], dim=-1)  # [B, H, W, 2]
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def make_orbit_camera_rays(
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    dist: torch.Tensor,
    target_xz: torch.Tensor,
    out_h: int,
    out_w: int,
    fov_deg: float = 44.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    Construct pinhole perspective camera rays for batch of orbit cameras:
      yaw: [B] radians in [0, 2*pi]
      pitch: [B] radians in [25 deg, 75 deg]
      dist: [B] orbit distance
      target_xz: [B, 2] look-at point (cx, cz) near (0.5, 0.5)
    Returns (cam_pos [B, 3], ray_dir [B, 3, H, W], cam_basis dict).
    """
    b = yaw.shape[0]
    device = yaw.device

    look_at = torch.stack([
        target_xz[:, 0],
        torch.full((b,), 0.11, device=device),
        target_xz[:, 1],
    ], dim=1)  # [B, 3]

    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    cam_offset = torch.stack([dist * cp * cy, dist * sp, dist * cp * sy], dim=1)
    cam_pos = look_at + cam_offset  # [B, 3]

    fwd = F.normalize(look_at - cam_pos, dim=1)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3).expand(b, 3)
    right = F.normalize(torch.cross(fwd, world_up, dim=1), dim=1)
    up = F.normalize(torch.cross(right, fwd, dim=1), dim=1)

    aspect = float(out_w) / float(out_h)
    tan_half = math.tan(math.radians(fov_deg * 0.5))

    # NDC: ndc_x in [-1, 1] left->right, ndc_y in [1, -1] top->bottom
    ndc_y = torch.linspace(1.0, -1.0, out_h, device=device).view(1, 1, out_h, 1)
    ndc_x = torch.linspace(-1.0, 1.0, out_w, device=device).view(1, 1, 1, out_w)

    rd = (
        fwd.view(b, 3, 1, 1)
        + right.view(b, 3, 1, 1) * (ndc_x * aspect * tan_half)
        + up.view(b, 3, 1, 1) * (ndc_y * tan_half)
    )
    rd = F.normalize(rd, dim=1)
    basis = {"cam_pos": cam_pos, "fwd": fwd, "right": right, "up": up, "tan_half": torch.tensor(tan_half, device=device), "aspect": torch.tensor(aspect, device=device)}
    return cam_pos, rd, basis


@torch.no_grad()
def raymarch_gbuf_and_target(
    fields: dict[str, torch.Tensor],
    yaw: torch.Tensor,
    pitch: torch.Tensor,
    dist: torch.Tensor,
    target_xz: torch.Tensor,
    out_h: int = 256,
    out_w: int = 256,
    n_steps: int = 64,
    n_refine: int = 6,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """
    GPU-vectorized 3D heightfield ray-marcher generating:
      1. Exact 18-channel screen-space G-buffer `gbuf` [B, 18, H, W]
      2. High-quality ray-traced / path-shaded 3D target RGB `rgb_3d` [B, 3, H, W]
      3. Geometry metadata (`hit_pos`, `hit_mask`, `cam_basis`) for 360° reprojection consistency!
    """
    h_norm = fields["h_norm"]
    b = h_norm.shape[0]
    device = h_norm.device

    cam_pos, rd, basis = make_orbit_camera_rays(yaw, pitch, dist, target_xz, out_h, out_w)
    ox = cam_pos[:, 0].view(b, 1, 1)
    oy = cam_pos[:, 1].view(b, 1, 1)
    oz = cam_pos[:, 2].view(b, 1, 1)
    dx = rd[:, 0]
    dy = rd[:, 1]
    dz = rd[:, 2]

    # Analytic ray-AABB intersection for world box [0, 1] x [0, HEIGHT_WORLD_SCALE] x [0, 1]
    eps = 1e-6
    inv_dx = 1.0 / torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
    inv_dy = 1.0 / torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)
    inv_dz = 1.0 / torch.where(dz.abs() < eps, torch.full_like(dz, eps), dz)

    tx0 = (0.0 - ox) * inv_dx
    tx1 = (1.0 - ox) * inv_dx
    ty0 = (0.0 - oy) * inv_dy
    ty1 = (HEIGHT_WORLD_SCALE - oy) * inv_dy
    tz0 = (0.0 - oz) * inv_dz
    tz1 = (1.0 - oz) * inv_dz

    t_min_x, t_max_x = torch.minimum(tx0, tx1), torch.maximum(tx0, tx1)
    t_min_y, t_max_y = torch.minimum(ty0, ty1), torch.maximum(ty0, ty1)
    t_min_z, t_max_z = torch.minimum(tz0, tz1), torch.maximum(tz0, tz1)

    t_enter = torch.maximum(torch.maximum(t_min_x, t_min_y), torch.maximum(t_min_z, torch.full_like(t_min_x, 0.02)))
    t_exit = torch.minimum(torch.minimum(t_max_x, t_max_y), t_max_z)
    valid_box = t_exit > t_enter

    # Coarse uniform march across [t_enter, t_exit] to find first heightfield intersection
    dt = torch.where(valid_box, (t_exit - t_enter) / float(n_steps), torch.zeros_like(t_enter))
    hit = torch.zeros((b, out_h, out_w), dtype=torch.bool, device=device)
    t_lo = t_enter.clone()
    t_hi = t_exit.clone()

    for s in range(n_steps + 1):
        t_cur = t_enter + dt * float(s)
        px = ox + dx * t_cur
        py = oy + dy * t_cur
        pz = oz + dz * t_cur
        in_xz = (px >= 0.0) & (px <= 1.0) & (pz >= 0.0) & (pz <= 1.0)
        h_s = _sample_field_2d(h_norm, px.clamp(0.0, 1.0), pz.clamp(0.0, 1.0))[:, 0] * HEIGHT_WORLD_SCALE
        below = valid_box & in_xz & (py <= h_s)
        newly_hit = below & (~hit)
        t_lo = torch.where(newly_hit, torch.clamp(t_cur - dt, min=t_enter), t_lo)
        t_hi = torch.where(newly_hit, t_cur, t_hi)
        hit = hit | newly_hit

    # 6-iteration binary search refinement for razor-sharp, zero-staircase 3D surface intersections
    for _ in range(n_refine):
        t_mid = 0.5 * (t_lo + t_hi)
        px_m = (ox + dx * t_mid).clamp(0.0, 1.0)
        py_m = oy + dy * t_mid
        pz_m = (oz + dz * t_mid).clamp(0.0, 1.0)
        h_m = _sample_field_2d(h_norm, px_m, pz_m)[:, 0] * HEIGHT_WORLD_SCALE
        below_m = py_m <= h_m
        t_hi = torch.where(hit & below_m, t_mid, t_hi)
        t_lo = torch.where(hit & (~below_m), t_mid, t_lo)

    t_hit = 0.5 * (t_lo + t_hi)
    px = (ox + dx * t_hit).clamp(0.0, 1.0)
    pz = (oz + dz * t_hit).clamp(0.0, 1.0)
    py = _sample_field_2d(h_norm, px, pz)[:, 0] * HEIGHT_WORLD_SCALE
    hit_f = hit.float().unsqueeze(1)  # [B, 1, H, W]

    # --- Assemble EXACT 18-channel G-Buffer ---
    sp_hit = _sample_field_2d(fields["sp"], px, pz) * hit_f
    temp_hit = _sample_field_2d(fields["temp"], px, pz) * hit_f
    burn_hit = (_sample_field_2d(fields["burn"], px, pz) > 0.45).float() * hit_f
    crit_hit = _sample_field_2d(fields["crit"], px, pz) * hit_f
    depth_norm = torch.where(hit.unsqueeze(1), torch.clamp(t_hit.unsqueeze(1) / DEPTH_MAX, 0.0, 0.995), torch.ones((b, 1, out_h, out_w), device=device))
    norm_hit = F.normalize(_sample_field_2d(fields["normal"], px, pz) + 1e-6, dim=1) * hit_f
    h_hit = _sample_field_2d(h_norm, px, pz) * hit_f

    gbuf = torch.cat([
        sp_hit,       # 0..10: 10 species fractions (0 for sky)
        temp_hit,     # 10: temp_norm
        burn_hit,     # 11: burning
        crit_hit,     # 12: critter_density
        depth_norm,   # 13: depth_norm (1.0 = sky)
        norm_hit,     # 14..17: world-space normal (nx, ny, nz)
        h_hit,        # 17: world_height_norm
    ], dim=1)
    assert gbuf.shape == (b, 18, out_h, out_w)

    # =========================================================================
    # High-Quality 3D Path-Shaded / Ray-Traced Target RGB
    # =========================================================================
    # Fixed dramatic golden-hour northern sun in world space
    sun_dir = F.normalize(torch.tensor([-0.54, 0.58, -0.61], device=device), dim=0)
    lx, ly, lz = float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])

    # 1. Real 3D Soft Shadow Rays (20 march steps along sun_dir from hit point!)
    shadow = torch.ones((b, out_h, out_w), device=device)
    y_start = py + 0.008
    for s in range(1, 21):
        ts = 0.012 * float(s) + 0.0025 * float(s * s)
        sx = px + lx * ts
        sy = y_start + ly * ts
        sz = pz + lz * ts
        in_b = (sx >= 0.0) & (sx <= 1.0) & (sz >= 0.0) & (sz <= 1.0)
        h_sh = _sample_field_2d(h_norm, sx.clamp(0.0, 1.0), sz.clamp(0.0, 1.0))[:, 0] * HEIGHT_WORLD_SCALE
        # IQ penumbra soft shadow estimate
        pen = torch.clamp(9.5 * (sy - h_sh) / (ts + 1e-4), 0.0, 1.0)
        shadow = torch.where(hit & in_b, torch.minimum(shadow, pen), shadow)
    shadow = shadow.unsqueeze(1)

    # 2. Heightfield Horizon Ambient Occlusion (SSAO / HBAO across 8 directions x 2 radii)
    ao_occ = torch.zeros((b, out_h, out_w), device=device)
    for rad in (0.014, 0.036):
        for ang_i in range(8):
            ang = ang_i * (math.pi / 4.0)
            qx = (px + math.cos(ang) * rad).clamp(0.0, 1.0)
            qz = (pz + math.sin(ang) * rad).clamp(0.0, 1.0)
            hq = _sample_field_2d(h_norm, qx, qz)[:, 0] * HEIGHT_WORLD_SCALE
            slope = torch.clamp((hq - py) / (rad * 1.15), 0.0, 1.0)
            ao_occ = ao_occ + slope
    ao = torch.clamp(1.0 - (ao_occ / 16.0) * 0.72, 0.28, 1.0).unsqueeze(1)

    # 3. Physically-Based Material Albedos & Micro-Detail from G-buffer species
    f_w = sp_hit[:, S_WATER:S_WATER + 1]
    f_i = sp_hit[:, S_ICE:S_ICE + 1]
    f_st = sp_hit[:, S_STEAM:S_STEAM + 1]
    f_so = sp_hit[:, S_SOIL:S_SOIL + 1]
    f_rk = sp_hit[:, S_ROCK:S_ROCK + 1]
    f_wd = sp_hit[:, S_WOOD:S_WOOD + 1]
    f_as = sp_hit[:, S_ASH:S_ASH + 1]
    f_sm = sp_hit[:, S_SMOKE:S_SMOKE + 1]
    f_oi = sp_hit[:, S_OIL:S_OIL + 1]
    f_pl = sp_hit[:, S_PLANT:S_PLANT + 1]

    nx_h, ny_h, nz_h = norm_hit[:, 0:1], norm_hit[:, 1:2], norm_hit[:, 2:3]
    vx, vy, vz = -dx.unsqueeze(1), -dy.unsqueeze(1), -dz.unsqueeze(1)

    # World-space micro-texture variation derived from height & world-space normal (100% 3D view-consistent & G-buffer learnable!)
    strata = 0.5 + 0.5 * torch.sin(h_hit * 58.0 + nx_h * 5.0 - nz_h * 4.0)
    foliage_var = 0.5 + 0.5 * torch.sin(h_hit * 85.0 + nx_h * 8.0) * torch.cos(h_hit * 65.0 + nz_h * 8.0)
    ripple_n = 0.06 * torch.sin(h_hit * 180.0 + nx_h * 12.0 - nz_h * 10.0) * f_w

    # Perturbed normal for shimmering water ripples
    n_eff = F.normalize(torch.cat([nx_h + ripple_n, ny_h, nz_h - ripple_n * 0.8], dim=1), dim=1)
    ndotl = torch.clamp(n_eff[:, 0:1] * lx + n_eff[:, 1:2] * ly + n_eff[:, 2:3] * lz, 0.0, 1.0)
    ndotv = torch.clamp(n_eff[:, 0:1] * vx + n_eff[:, 1:2] * vy + n_eff[:, 2:3] * vz, 0.0, 1.0)

    # Half-vector specular for sun glint on water, ice, oil
    hx = lx + vx
    hy = ly + vy
    hz = lz + vz
    hlen = torch.sqrt(hx * hx + hy * hy + hz * hz + 1e-6)
    ndoth = torch.clamp((n_eff[:, 0:1] * hx + n_eff[:, 1:2] * hy + n_eff[:, 2:3] * hz) / hlen, 0.0, 1.0)
    sun_spec = torch.pow(ndoth, 48.0) * shadow * (0.95 * f_w + 0.85 * f_i + 0.90 * f_oi + 0.12 * f_rk)

    def _c(r, g, b_):
        return torch.tensor([r, g, b_], device=device).view(1, 3, 1, 1)

    # Material albedos
    alb_water = _c(0.055, 0.21, 0.34) * (1.0 - 0.35 * torch.clamp(h_hit * 4.0, 0.0, 1.0)) + _c(0.09, 0.36, 0.45) * 0.35
    alb_ice = _c(0.54, 0.84, 0.97)
    wet_m = torch.clamp(f_w * 2.5, 0.0, 1.0)
    alb_soil = _c(0.35, 0.27, 0.18) * (1.0 - 0.60 * wet_m) + _c(0.14, 0.11, 0.09) * (0.60 * wet_m)
    # Stratified granite cliff albedo (lighter sunlit cap, darker steep slate walls)
    steep = torch.clamp(1.0 - ny_h, 0.0, 1.0)
    alb_rock = _c(0.36, 0.38, 0.43) * (0.82 + 0.28 * strata) * (1.0 - 0.30 * steep) + _c(0.24, 0.27, 0.32) * (0.30 * steep)
    alb_wood = _c(0.08, 0.24, 0.12) * (0.75 + 0.45 * foliage_var) + _c(0.07, 0.12, 0.04) * torch.clamp((h_hit - 0.42) * 2.5, 0.0, 1.0)
    alb_ash = _c(0.13, 0.12, 0.13)
    alb_plant = _c(0.16, 0.42, 0.17) * (0.82 + 0.35 * foliage_var)

    base_w = f_w + f_i + f_so + f_rk + f_wd + f_as + f_pl
    albedo = (
        f_w * alb_water + f_i * alb_ice + f_so * alb_soil
        + f_rk * alb_rock + f_wd * alb_wood + f_as * alb_ash + f_pl * alb_plant
    ) / torch.clamp(base_w, min=1e-4)

    # Oil iridescence film on albedo
    oil_a = torch.clamp(f_oi * 1.4, 0.0, 0.85)
    iri_p = 9.0 * f_oi + 4.5 * ndotv + 16.0 * h_hit
    iri = torch.cat([0.5 + 0.5 * torch.sin(iri_p), 0.5 + 0.5 * torch.sin(iri_p + 2.09), 0.5 + 0.5 * torch.sin(iri_p + 4.19)], dim=1)
    albedo = albedo * (1.0 - oil_a * 0.75) + (_c(0.06, 0.05, 0.10) + 0.35 * iri) * oil_a

    # 4. Direct Golden-Hour Sun + Sky Hemisphere Ambient + Shadowed Bounce
    sun_col = _c(1.38, 1.18, 0.92)
    sky_amb = _c(0.32, 0.46, 0.68) * (0.55 + 0.45 * ny_h) + _c(0.16, 0.14, 0.12) * (0.45 - 0.45 * ny_h)
    lit = albedo * (sky_amb * ao + sun_col * ndotl * shadow) + sun_col * sun_spec * 0.85

    # 5. Subsurface Glacier Ice Glow (wrap backlit transmission + cyan frost scatter)
    ice_sss = f_i * (0.38 + 0.45 * torch.clamp(0.45 + 0.55 * ndotl, 0.0, 1.0) * (0.4 + 0.6 * shadow))
    lit = lit + ice_sss * _c(0.16, 0.58, 0.82) * 0.48
    frost_irr_hit = _sample_field_2d(fields["frost_irr"], px, pz)
    lit = lit + frost_irr_hit * _c(0.18, 0.78, 1.00) * 0.32 * hit_f

    # 6. Water Fresnel Sky Reflection (Grazing angle reflects crisp northern twilight sky!)
    fresnel = (0.04 + 0.96 * torch.pow(1.0 - ndotv, 4.2)) * torch.clamp(f_w * 1.1 + f_oi * 0.6 + f_i * 0.35, 0.0, 1.0)
    # Reflected ray Y component for sky reflection color
    refl_y = torch.clamp(2.0 * ndotv * n_eff[:, 1:2] - vy, 0.0, 1.0)
    sky_refl_col = _c(0.70, 0.80, 0.92) * (1.0 - refl_y) + _c(0.16, 0.36, 0.66) * refl_y
    lit = lit * (1.0 - 0.75 * fresnel) + sky_refl_col * (0.85 * fresnel)

    # 7. VOLUMETRIC FIRELIGHT BOUNCING ONTO SURROUNDING CLIFFS, ICE, WATER & PINES!
    fire_irr_hit = _sample_field_2d(fields["fire_irr"], px, pz)
    fdir_x = _sample_field_2d(fields["fire_dir_x"], px, pz)
    fdir_z = _sample_field_2d(fields["fire_dir_z"], px, pz)
    # Normal alignment with incoming firelight (cliffs & pine faces turned toward fire catch warm bounce!)
    n_dot_fire = torch.clamp(nx_h * fdir_x + nz_h * fdir_z + 0.35 * ny_h, 0.12, 1.0)
    # Specular fire reflection on water/ice/oil!
    fire_spec = torch.clamp(f_w + f_i * 0.85 + f_oi * 0.85, 0.0, 1.0) * (0.45 + 0.65 * (1.0 - ndotv))
    fire_bounce = fire_irr_hit * (0.85 * n_dot_fire * (0.45 + 0.55 * albedo) + 0.75 * fire_spec)
    lit = lit + fire_bounce * _c(1.45, 0.52, 0.10) * 0.95

    # Active emissive fire core & smoldering ash embers
    burn_soft_hit = _sample_field_2d(fields["burn_soft"], px, pz)
    flame_a = torch.clamp(burn_soft_hit * (0.78 + 0.35 * temp_hit), 0.0, 0.92)
    flame_col = _c(1.35, 0.36, 0.05) * (1.0 - flame_a) + _c(1.65, 1.15, 0.28) * flame_a
    lit = lit * (1.0 - flame_a) + flame_col * flame_a
    ember = f_as * torch.clamp((temp_hit - 0.26) / 0.42, 0.0, 1.0)
    lit = lit + ember * _c(1.15, 0.32, 0.04) * 0.75

    # Translucent smoke & steam plumes with warm fire under-lighting
    sm_a = torch.clamp(f_sm * 1.1, 0.0, 0.76)
    sm_col = _c(0.16, 0.17, 0.20) * (0.6 + 0.5 * shadow) + fire_irr_hit * _c(0.95, 0.34, 0.07) * 0.70
    lit = lit * (1.0 - sm_a) + sm_col * sm_a

    st_a = torch.clamp(f_st * 1.15, 0.0, 0.72)
    st_col = _c(0.84, 0.91, 0.96) * (0.72 + 0.32 * shadow) + fire_irr_hit * _c(0.45, 0.22, 0.08) * 0.45
    lit = lit * (1.0 - st_a) + st_col * st_a

    # Glowing living wildlife herds on 3D terrain
    crit_a = torch.clamp((crit_hit - 0.10) / 0.50, 0.0, 1.0)
    lit = lit * (1.0 - crit_a) + _c(1.15, 0.88, 0.26) * crit_a

    # 8. Atmospheric Depth Fog + Silhouette-Anchored Northern Twilight Sky Gradient
    # Anchoring horizon mist to terrain silhouette proximity makes the sky 100% G-buffer inferable
    # across all camera pitches & resolutions with ZERO border halo and rich Nordic twilight mood!
    horizon_glow = torch.clamp(gaussian_blur(hit_f, 9.0) * 1.45, 0.0, 1.0)
    sky_zenith = _c(0.15, 0.28, 0.52)
    sky_horizon = _c(0.68, 0.72, 0.80) + _c(0.22, 0.14, 0.06)
    sky_col = sky_zenith * (1.0 - horizon_glow) + sky_horizon * horizon_glow
    fog_col = _c(0.56, 0.65, 0.78)
    # Distance-based exponential aerial perspective fog (height-attenuated so tall cliffs pierce mist)
    fog_factor = torch.clamp(1.0 - torch.exp(-1.15 * torch.pow(torch.clamp(depth_norm - 0.16, 0.0, 1.0), 1.45)), 0.0, 0.72)
    fog_factor = fog_factor * (1.0 - 0.35 * h_hit)
    surface_rgb = lit * (1.0 - fog_factor) + fog_col * fog_factor

    rgb_raw = torch.where(hit.unsqueeze(1), surface_rgb, sky_col)

    # ACES-like filmic soft-knee tone mapping -> [0.012, 0.988]
    rgb_raw = torch.clamp(rgb_raw, min=0.0)
    rgb_3d = (rgb_raw * (2.51 * rgb_raw + 0.03)) / (rgb_raw * (2.43 * rgb_raw + 0.59) + 0.14)
    rgb_3d = torch.clamp(rgb_3d, 0.012, 0.988)

    meta = {
        "hit_pos": torch.stack([px, py, pz], dim=1),  # [B, 3, H, W]
        "hit_mask": hit,  # [B, H, W]
        "t_hit": t_hit,  # [B, H, W]
        "cam_basis": basis,
    }
    return gbuf, rgb_3d, meta


# =============================================================================
# 3. ~160k-Param Dilated Multi-Scale Neural Deferred Shader
# =============================================================================
class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.gelu(self.conv1(x)))


class Tier2Deferred3DNet(nn.Module):
    """
    ~162k-param Dilated Multi-Scale Neural Deferred Shader:
      Input:  gbuf   float32 [B, 18, H, W] (dynamic B, H, W)
      Output: rgb_3d float32 [B, 3, H, W] in [0, 1] (Sigmoid)
    """

    def __init__(self, in_ch: int = 18, base_ch: int = 40, pyr_ch: int = 28):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Wide multi-scale pyramid for long-range shadows, volumetric firelight bounce & sky gradient
        self.pyr_in = nn.Conv2d(base_ch, pyr_ch, kernel_size=1)
        self.pyr_c1 = nn.Conv2d(pyr_ch, pyr_ch, kernel_size=3, padding=1)
        self.pyr_c2 = nn.Conv2d(pyr_ch, pyr_ch, kernel_size=3, padding=1)
        self.pyr_c3 = nn.Conv2d(pyr_ch, pyr_ch, kernel_size=3, padding=1)
        self.pyr_out = nn.Conv2d(pyr_ch, base_ch, kernel_size=1)

        # Multi-scale dilated residual blocks (dilations 1, 2, 4, 8 -> huge RF at full resolution!)
        self.res1 = DilatedResBlock(base_ch, dilation=1)
        self.res2 = DilatedResBlock(base_ch, dilation=2)
        self.res3 = DilatedResBlock(base_ch, dilation=4)
        self.res4 = DilatedResBlock(base_ch, dilation=8)

        # Crisp deferred shading head
        self.head1 = nn.Conv2d(base_ch, 32, kernel_size=3, padding=1)
        self.head2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.head3 = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, gbuf: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.stem(gbuf))

        p4 = F.avg_pool2d(self.pyr_in(h), kernel_size=4, stride=4)
        p4 = F.gelu(self.pyr_c1(p4))
        p8 = F.gelu(self.pyr_c2(F.avg_pool2d(p4, kernel_size=2, stride=2)))
        p8_up = F.interpolate(p8, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.gelu(self.pyr_c3(p4 + p8_up))
        p_up = self.pyr_out(F.interpolate(p4, size=h.shape[-2:], mode="bilinear", align_corners=False))

        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h) + p_up

        u = F.gelu(self.head1(h))
        u = F.gelu(self.head2(u))
        return torch.sigmoid(self.head3(u))


# =============================================================================
# 4. Training Loop (B=8, Random 3D Orbit Cameras)
# =============================================================================
class SobelLoss3D(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device=device) / 4.0
        ky = kx.t().contiguous()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        px = F.conv2d(pred, self.kx, padding=1, groups=3)
        py = F.conv2d(pred, self.ky, padding=1, groups=3)
        tx = F.conv2d(target, self.kx, padding=1, groups=3)
        ty = F.conv2d(target, self.ky, padding=1, groups=3)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


def _sample_random_cameras(b: int, device: torch.device):
    yaw = torch.rand((b,), device=device) * (2.0 * math.pi)
    pitch = torch.empty((b,), device=device).uniform_(math.radians(25.0), math.radians(75.0))
    dist = torch.empty((b,), device=device).uniform_(0.88, 1.55)
    target_xz = torch.empty((b, 2), device=device).uniform_(0.40, 0.60)
    return yaw, pitch, dist, target_xz


def train_tier2(
    train_split: dict[str, torch.Tensor],
    device: torch.device,
    steps: int = 1800,
    batch_size: int = 8,
    seed: int = 2002,
) -> tuple[Tier2Deferred3DNet, dict]:
    torch.manual_seed(seed)
    model = Tier2Deferred3DNet(in_ch=18, base_ch=40, pyr_ch=28).to(device)
    sobel = SobelLoss3D(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=8e-5)

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    n_data = train_split["mass"].shape[0]
    t0 = time.time()
    last_l1, last_sob = 0.0, 0.0

    cached_fields = None
    for step in range(1, steps + 1):
        if cached_fields is None or (step - 1) % 4 == 0:
            idx = torch.randint(0, n_data, (batch_size,))
            mass = train_split["mass"][idx].to(device, non_blocking=True)
            t_ck = train_split["t_ck"][idx].to(device, non_blocking=True)
            fe = train_split["flow_e"][idx].to(device, non_blocking=True)
            fs = train_split["flow_s"][idx].to(device, non_blocking=True)
            crit = train_split["critter"][idx].to(device, non_blocking=True)
            elev = train_split["elev"][idx].to(device, non_blocking=True)
            t_sec = torch.rand((batch_size,), device=device) * (2.0 * math.pi / 2.5)
            cached_fields = build_3d_terrain_fields(mass, t_ck, fe, fs, crit, elev, t_sec, grid_res=256)

        yaw, pitch, dist, target_xz = _sample_random_cameras(batch_size, device)
        if step % 5 == 0:
            sub_fields = {k: v[:4] for k, v in cached_fields.items()}
            with torch.no_grad():
                gbuf, target, _ = raymarch_gbuf_and_target(
                    sub_fields, yaw[:4], pitch[:4], dist[:4], target_xz[:4], out_h=384, out_w=384
                )
        else:
            with torch.no_grad():
                gbuf, target, _ = raymarch_gbuf_and_target(
                    cached_fields, yaw, pitch, dist, target_xz, out_h=256, out_w=256
                )

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
            pred = model(gbuf)
            l_l1 = F.l1_loss(pred, target)
            l_sob = sobel(pred, target)
            loss = l_l1 + 0.25 * l_sob

        assert torch.isfinite(loss), f"Non-finite Tier 2 loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

        last_l1 = float(l_l1.item())
        last_sob = float(l_sob.item())
        if step in (1, 450, 900, 1350, steps):
            print(
                f"  [Tier2] step {step:4d}/{steps} | L1={last_l1:.4f} Sobel={last_sob:.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )

    model.eval()
    return model, {"train_time_s": round(time.time() - t0, 1), "steps": steps, "batch_size": batch_size, "final_l1": last_l1, "final_sobel": last_sob}


# =============================================================================
# 5. Evaluation: Held-Out 256² & 512² PSNR + 360° Multi-View 3D Consistency
# =============================================================================
def _eval_psnr_mae(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    diff = pred.float() - target.float()
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    return psnr, mae


@torch.no_grad()
def evaluate_tier2(
    model: Tier2Deferred3DNet,
    val_split: dict[str, torch.Tensor],
    hero_scene: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict, dict]:
    model.eval()
    torch.manual_seed(9876)

    # 1. Held-out 256x256 evaluation across 32 random orbit cameras
    n_val = min(32, val_split["mass"].shape[0])
    psnrs_256, maes_256 = [], []
    for i in range(0, n_val, 8):
        sl = slice(i, min(i + 8, n_val))
        b = sl.stop - sl.start
        fields = build_3d_terrain_fields(
            val_split["mass"][sl].to(device),
            val_split["t_ck"][sl].to(device),
            val_split["flow_e"][sl].to(device),
            val_split["flow_s"][sl].to(device),
            val_split["critter"][sl].to(device),
            val_split["elev"][sl].to(device),
            torch.linspace(0.1, 2.2, b, device=device),
        )
        yaw, pitch, dist, txz = _sample_random_cameras(b, device)
        gbuf, gt, _ = raymarch_gbuf_and_target(fields, yaw, pitch, dist, txz, out_h=256, out_w=256)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(gbuf).float()
        p, m = _eval_psnr_mae(pred, gt)
        psnrs_256.append(p)
        maes_256.append(m)

    # 2. Zero-shot 512x512 dynamic resolution evaluation on hero + val scenes
    hero_fields = build_3d_terrain_fields(
        hero_scene["mass"].to(device),
        hero_scene["t_ck"].to(device),
        hero_scene["flow_e"].to(device),
        hero_scene["flow_s"].to(device),
        hero_scene["critter"].to(device),
        hero_scene["elev"].to(device),
        torch.tensor([0.9], device=device),
    )

    # 4 Dramatic 3D camera angles of the EXACT same live Frostmarsh battle (rendered at 512x512!)
    orbit_specs = [
        ("1. Isometric Tactical", 42.0, 42.0, 1.15, (0.50, 0.50)),
        ("2. Low Dramatic Battle", 128.0, 26.0, 0.90, (0.54, 0.46)),
        ("3. High Tactical Overview", 222.0, 68.0, 1.30, (0.50, 0.50)),
        ("4. Reverse Orbit Firelight", 312.0, 35.0, 1.04, (0.52, 0.48)),
    ]
    orbit_vis = []
    psnrs_512, maes_512 = [], []
    for title, yaw_d, pitch_d, d_val, (cx, cz) in orbit_specs:
        y_t = torch.tensor([math.radians(yaw_d)], device=device)
        p_t = torch.tensor([math.radians(pitch_d)], device=device)
        d_t = torch.tensor([d_val], device=device)
        txz = torch.tensor([[cx, cz]], device=device)
        gbuf_512, gt_512, _ = raymarch_gbuf_and_target(hero_fields, y_t, p_t, d_t, txz, out_h=512, out_w=512)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred_512 = model(gbuf_512).float()
        psnr_i, mae_i = _eval_psnr_mae(pred_512, gt_512)
        psnrs_512.append(psnr_i)
        maes_512.append(mae_i)
        # Visualize G-buffer normals for inspection strip
        norm_vis = torch.clamp(gbuf_512[:, 14:17] * 0.5 + 0.5, 0.0, 1.0)
        sky_m = (gbuf_512[:, 13:14] >= 0.999)
        norm_vis = torch.where(sky_m, torch.full_like(norm_vis, 0.08), norm_vis)
        orbit_vis.append({
            "title": title,
            "camera_str": f"yaw={yaw_d:.0f} deg, pitch={pitch_d:.0f} deg, dist={d_val:.2f}",
            "psnr_db": round(psnr_i, 2),
            "pred": pred_512[0].cpu().numpy(),
            "gt": gt_512[0].cpu().numpy(),
            "normal_vis": norm_vis[0].cpu().numpy(),
        })

    # 3. 360-degree Multi-View 3D Temporal Consistency (36 frames, 10 deg yaw steps)
    # Reproject 3D hit points of frame k into camera k+1 and measure photometric consistency on co-visible surfaces!
    n_orb = 36
    orb_preds, orb_gts, orb_metas = [], [], []
    for k in range(n_orb):
        yk = torch.tensor([math.radians(k * (360.0 / n_orb))], device=device)
        pk = torch.tensor([math.radians(38.0)], device=device)
        dk = torch.tensor([1.12], device=device)
        txz = torch.tensor([[0.50, 0.50]], device=device)
        gb, gt_k, meta_k = raymarch_gbuf_and_target(hero_fields, yk, pk, dk, txz, out_h=256, out_w=256)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pr_k = model(gb).float()
        orb_preds.append(pr_k)
        orb_gts.append(gt_k)
        orb_metas.append(meta_k)

    neural_reproj_rms, gt_reproj_rms, neural_reproj_psnr, covis_fracs = [], [], [], []
    for k in range(n_orb):
        kn = (k + 1) % n_orb
        pos_k = orb_metas[k]["hit_pos"]     # [1, 3, H, W]
        hm_k = orb_metas[k]["hit_mask"][0]  # [H, W]
        bas_n = orb_metas[kn]["cam_basis"]
        t_n = orb_metas[kn]["t_hit"]        # [1, H, W]

        # Project 3D world point pos_k into camera kn
        rel = pos_k - bas_n["cam_pos"].view(1, 3, 1, 1)
        z_cam = (rel * bas_n["fwd"].view(1, 3, 1, 1)).sum(dim=1)    # [1, H, W]
        x_cam = (rel * bas_n["right"].view(1, 3, 1, 1)).sum(dim=1)
        y_cam = (rel * bas_n["up"].view(1, 3, 1, 1)).sum(dim=1)
        dist_3d = torch.sqrt((rel * rel).sum(dim=1) + 1e-8)

        th = float(bas_n["tan_half"].item())
        asp = float(bas_n["aspect"].item())
        u_ndc = x_cam / (torch.clamp(z_cam, min=1e-4) * asp * th)   # [-1, 1] left->right
        v_ndc = -y_cam / (torch.clamp(z_cam, min=1e-4) * th)        # [-1, 1] top->bottom (grid_sample y=-1 is top!)

        in_view = hm_k & (z_cam[0] > 0.05) & (u_ndc[0].abs() < 0.96) & (v_ndc[0].abs() < 0.96)
        grid = torch.stack([u_ndc, v_ndc], dim=-1)  # [1, H, W, 2]
        t_sampled = F.grid_sample(t_n.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]
        covis = in_view & ((dist_3d[0] - t_sampled).abs() < 0.025)
        covis_fracs.append(float(covis.float().mean().item()))

        pr_next_warp = F.grid_sample(orb_preds[kn], grid, mode="bilinear", padding_mode="border", align_corners=True)
        gt_next_warp = F.grid_sample(orb_gts[kn], grid, mode="bilinear", padding_mode="border", align_corners=True)

        d_pr = (orb_preds[k][0, :, covis] - pr_next_warp[0, :, covis])
        d_gt = (orb_gts[k][0, :, covis] - gt_next_warp[0, :, covis])
        mse_pr = float((d_pr * d_pr).mean().item())
        mse_gt = float((d_gt * d_gt).mean().item())
        neural_reproj_rms.append(math.sqrt(mse_pr))
        gt_reproj_rms.append(math.sqrt(mse_gt))
        neural_reproj_psnr.append(10.0 * math.log10(1.0 / max(mse_pr, 1e-12)))

    metrics = {
        "n_params": count_parameters(model),
        "val_256_psnr_db": round(float(np.mean(psnrs_256)), 2),
        "val_256_mae": round(float(np.mean(maes_256)), 4),
        "zero_shot_512_psnr_db": round(float(np.mean(psnrs_512)), 2),
        "zero_shot_512_mae": round(float(np.mean(maes_512)), 4),
        "orbit_4angles_psnr_db": {o["title"]: o["psnr_db"] for o in orbit_vis},
        "multiview_360_consistency": {
            "n_views": n_orb,
            "step_deg": 360.0 / n_orb,
            "avg_covisible_fraction": round(float(np.mean(covis_fracs)), 4),
            "neural_reproj_rms": round(float(np.mean(neural_reproj_rms)), 5),
            "neural_reproj_psnr_db": round(float(np.mean(neural_reproj_psnr)), 2),
            "raytraced_ref_reproj_rms": round(float(np.mean(gt_reproj_rms)), 5),
            "excess_view_flicker_rms": round(max(0.0, float(np.mean(neural_reproj_rms)) - float(np.mean(gt_reproj_rms))), 5),
        },
    }
    return metrics, {"orbit_vis": orbit_vis}


# =============================================================================
# 6. ONNX Export (Opset 17, Exact Contract) + GPU FP16 (256² & 512²) & CPU ORT
# =============================================================================
def export_and_benchmark_tier2(model: Tier2Deferred3DNet, out_onnx: Path, device: torch.device) -> dict:
    model_cpu = model.cpu().eval()
    data_sidecar = Path(str(out_onnx) + ".data")
    for p in (data_sidecar, out_onnx):
        if p.exists():
            p.unlink()

    dummy = torch.randn(1, 18, 256, 256, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        (dummy,),
        str(out_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["gbuf"],
        output_names=["rgb_3d"],
        dynamic_axes={
            "gbuf": {0: "batch", 2: "height", 3: "width"},
            "rgb_3d": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,
    )

    proto = onnx.load(str(out_onnx), load_external_data=True)
    onnx.checker.check_model(proto)
    onnx.save_model(proto, str(out_onnx), save_as_external_data=False)
    if data_sidecar.exists():
        data_sidecar.unlink()
    assert not data_sidecar.exists(), "External .data sidecar must not exist!"

    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    in_meta = sess.get_inputs()[0]
    out_meta = sess.get_outputs()[0]
    assert in_meta.name == "gbuf", f"Expected input 'gbuf', got {in_meta.name}"
    assert out_meta.name == "rgb_3d", f"Expected output 'rgb_3d', got {out_meta.name}"

    parity = {}
    for shape in [(1, 18, 256, 256), (2, 18, 256, 256), (1, 18, 512, 512), (1, 18, 192, 320)]:
        x = torch.rand(*shape, dtype=torch.float32)
        with torch.inference_mode():
            y_pt = model_cpu(x).numpy()
        y_ort = sess.run(["rgb_3d"], {"gbuf": x.numpy()})[0]
        assert y_ort.shape == (shape[0], 3, shape[2], shape[3])
        assert float(y_ort.min()) >= 0.0 and float(y_ort.max()) <= 1.0
        max_diff = float(np.max(np.abs(y_pt - y_ort)))
        assert max_diff < 1e-4, f"Tier 2 ORT parity error {max_diff} >= 1e-4 at {shape}"
        parity[f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"] = max_diff
        print(f"  [tier2_deferred_3d.onnx OK] {shape} -> {y_ort.shape}, max_abs_diff={max_diff:.2e}", flush=True)

    # GPU FP16 Latency at BOTH 256x256 and 512x512
    gpu_bench = {}
    if device.type == "cuda":
        m_gpu = model.to(device).eval().half()
        for res in (256, 512):
            x_gpu = torch.rand(1, 18, res, res, device=device, dtype=torch.float16)
            with torch.inference_mode():
                for _ in range(25):
                    _ = m_gpu(x_gpu)
                torch.cuda.synchronize()
                times = []
                for _ in range(100):
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
                    _ = m_gpu(x_gpu)
                    ev1.record()
                    torch.cuda.synchronize()
                    times.append(ev0.elapsed_time(ev1))
            gpu_bench[f"{res}x{res}_fp16"] = {
                "mean_ms": round(float(np.mean(times)), 3),
                "median_ms": round(float(np.median(times)), 3),
                "p95_ms": round(float(np.percentile(times, 95)), 3),
                "fps": round(1000.0 / float(np.median(times)), 1),
            }
        model.float().to(device)

    # CPU ORT Latency (256x256 across 1/4/8 threads + 512x512 at 4/8 threads)
    cpu_bench = {}
    for res, thr_list in [(256, (1, 4, 8)), (512, (4, 8))]:
        x_bench = np.random.rand(1, 18, res, res).astype(np.float32)
        cpu_bench[f"{res}x{res}"] = {}
        for n_thr in thr_list:
            so = ort.SessionOptions()
            so.intra_op_num_threads = n_thr
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            s_t = ort.InferenceSession(str(out_onnx), so, providers=["CPUExecutionProvider"])
            for _ in range(3):
                s_t.run(["rgb_3d"], {"gbuf": x_bench})
            n_iters = 10 if (res == 512 or n_thr == 1) else 25
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                s_t.run(["rgb_3d"], {"gbuf": x_bench})
                times.append((time.perf_counter() - t0) * 1000.0)
            cpu_bench[f"{res}x{res}"][f"{n_thr}_threads"] = {
                "mean_ms": round(float(np.mean(times)), 2),
                "median_ms": round(float(np.median(times)), 2),
                "p95_ms": round(float(np.percentile(times, 95)), 2),
            }

    return {
        "path": str(out_onnx),
        "size_bytes": out_onnx.stat().st_size,
        "size_kb": round(out_onnx.stat().st_size / 1024.0, 2),
        "opset": 17,
        "n_params": count_parameters(model_cpu),
        "input_name": in_meta.name,
        "input_shape": in_meta.shape,
        "output_name": out_meta.name,
        "output_shape": out_meta.shape,
        "max_abs_diff": parity,
        "gpu_fp16_latency_ms": gpu_bench,
        "ort_cpu_latency_ms": cpu_bench,
    }


# =============================================================================
# 7. Dramatic 4-Angle 3D Orbit Showcase (`tier2_3d_orbit.png`)
# =============================================================================
def _to_pil(chw: np.ndarray) -> Image.Image:
    hwc = np.clip(np.transpose(chw, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, mode="RGB")


def save_tier2_orbit_png(vis_data: dict, metrics: dict, out_png: Path) -> None:
    """
    Save `tier2_3d_orbit.png`:
      4 dramatic 3D camera angles of the EXACT same live Frostmarsh battle:
        1. Isometric Tactical  |  2. Low Dramatic Battle  |  3. High Tactical Overview  |  4. Reverse Orbit Firelight
      Row 1 (512x512 large): Tier 2 Neural Deferred 3D (`tier2_deferred_3d.onnx`)
      Row 2 (512x512 large): Ray-Traced 3D Reference Target (Soft Shadow Rays + SSAO + Volumetric Firelight + Fresnel)
      Row 3 (compact 220px): Input 3D G-Buffer World-Space Normals (channel 14..16)
    """
    orbits = vis_data["orbit_vis"]
    tile = 384
    norm_h = 192
    pad = 12
    header_h = 48
    row_lbl_h = 24
    cols = 4

    cw = pad + cols * (tile + pad)
    ch = header_h + (row_lbl_h + tile + pad) * 2 + (row_lbl_h + norm_h + pad) + pad
    canvas = Image.new("RGB", (cw, ch), color=(12, 15, 20))
    draw = ImageDraw.Draw(canvas)

    mv = metrics["multiview_360_consistency"]
    banner = (
        f"Tier 2 Real 3D Neural Deferred Renderer (tier2_deferred_3d.onnx, {metrics['n_params']:,} params)  |  "
        f"Val 256²: {metrics['val_256_psnr_db']:.2f} dB  |  Zero-Shot 512²: {metrics['zero_shot_512_psnr_db']:.2f} dB  |  "
        f"360° Reproj Consistency: {mv['neural_reproj_psnr_db']:.2f} dB (RMS {mv['neural_reproj_rms']:.4f})"
    )
    draw.text((pad + 4, 8), banner, fill=(248, 224, 145))

    for ci, o in enumerate(orbits):
        x0 = pad + ci * (tile + pad)
        draw.text((x0 + 4, 28), f"{o['title']}  ({o['camera_str']})", fill=(165, 225, 255))

    # Row 1: Tier 2 Neural Deferred 3D
    y1_lbl = header_h
    y1_img = y1_lbl + row_lbl_h
    draw.text((pad + 4, y1_lbl + 4), "Row 1: Tier 2 Neural Deferred 3D Output (Single Feed-Forward Pass @ 512x512)", fill=(125, 250, 170))
    for ci, o in enumerate(orbits):
        x0 = pad + ci * (tile + pad)
        im = _to_pil(o["pred"]).resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(im, (x0, y1_img))
        draw.rectangle([x0, y1_img, x0 + tile - 1, y1_img + tile - 1], outline=(85, 235, 145), width=2)
        draw.rectangle([x0 + 6, y1_img + 6, x0 + 96, y1_img + 24], fill=(10, 12, 18))
        draw.text((x0 + 12, y1_img + 9), f"{o['psnr_db']:.2f} dB", fill=(125, 250, 170))

    # Row 2: Ray-Traced 3D Reference
    y2_lbl = y1_img + tile + pad
    y2_img = y2_lbl + row_lbl_h
    draw.text((pad + 4, y2_lbl + 4), "Row 2: Ray-Traced 3D Path-Shaded Target (Soft Shadow Rays + SSAO + Fire Bounce + Water Fresnel)", fill=(235, 238, 245))
    for ci, o in enumerate(orbits):
        x0 = pad + ci * (tile + pad)
        im = _to_pil(o["gt"]).resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(im, (x0, y2_img))

    # Row 3: G-Buffer World-Space Normals
    y3_lbl = y2_img + tile + pad
    y3_img = y3_lbl + row_lbl_h
    draw.text((pad + 4, y3_lbl + 4), "Row 3: Input 18-ch G-Buffer Geometry Preview (World-Space Surface Normals ch 14..16)", fill=(185, 200, 225))
    for ci, o in enumerate(orbits):
        x0 = pad + ci * (tile + pad)
        im = _to_pil(o["normal_vis"]).resize((tile, norm_h), Image.Resampling.BILINEAR)
        canvas.paste(im, (x0, y3_img))

    canvas.save(str(out_png), format="PNG", optimize=True)
