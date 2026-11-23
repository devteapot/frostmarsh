#!/usr/bin/env python3
"""
Tier 1 — High-Res 512x512 Painterly RPG Neural Renderer (`tier1_painterly.onnx`).

Takes 128x128 A+ state (17 ch) + 2 deterministic world-space texture coordinate channels
-> 19 channels [B, 19, H, W] -> 4x upsampling [B, 3, 4*H, 4*W] via a ~400k-param ConvNet
with checkerboard-free bilinear+conv upsampling and multi-harmonic world-coord injection,
trained with L1 + Multi-Scale Sobel + Multi-Scale Conditional PatchGAN (Hinge + Feature Matching).
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
    LATENT_DIR,
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
    rgb_const,
    shift2d,
)


# =============================================================================
# 1. Deterministic World-Space Texture Coordinates (Zero-Swimming Anchor)
# =============================================================================
def make_world_tex_coords(
    b: int,
    h: int,
    w: int,
    device: torch.device,
    ox: torch.Tensor | None = None,
    oy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Deterministic 2-channel world-space texture coordinate field [B, 2, H, W] in [0, 1].
    Smoothly varying, irrational-ratio multi-octave world phase locked to integer world
    cell coordinates (wy, wx) = (y + oy, x + ox) so micro-detail sticks to terrain with
    zero swimming under camera panning and zero grid/checkerboard aliasing!
    """
    if ox is None:
        ox = torch.zeros((b,), device=device, dtype=torch.float32)
    if oy is None:
        oy = torch.zeros((b,), device=device, dtype=torch.float32)
    yy = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1) + oy.view(b, 1, 1, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w) + ox.view(b, 1, 1, 1)

    # Smooth medium-wavelength world phases (period ~8..18 cells) -> bilinear upsampling is organic & smooth!
    u = 0.5 + 0.30 * torch.sin(0.37 * xx + 0.23 * yy) + 0.19 * torch.cos(0.79 * xx - 0.53 * yy)
    v = 0.5 + 0.30 * torch.cos(0.29 * xx - 0.41 * yy) + 0.19 * torch.sin(0.67 * xx + 0.83 * yy)
    return torch.cat([u, v], dim=1)


# =============================================================================
# 2. High-Detail Organic 512x512 Painterly RPG Target Renderer (World-Locked)
# =============================================================================
def _up4(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)


@torch.no_grad()
def render_painterly_target_512(
    mass: torch.Tensor,
    t_ck: torch.Tensor,
    flow_e: torch.Tensor,
    flow_s: torch.Tensor,
    critter_norm: torch.Tensor,
    t_sec: torch.Tensor,
    world_tex_128: torch.Tensor,
) -> torch.Tensor:
    """
    Organic, checkerboard-free 512x512 painterly RPG target renderer anchored strictly to
    world_tex_128 (zero swimming on camera pan, crisp painterly brushwork):
      - Crisp grass blades (directional organic strokes & sunlit emerald tips)
      - Pine canopy brushwork (layered conifer bough tufts & sunlit NW crowns)
      - Fractured ice veins (bold sapphire crack networks & frosted crystal rims)
      - Glowing ember sparks (sharp golden-orange sparks & curling flame tongues)
      - Directional rock cliffs (sculpted granite facets, bright NW ridges, dark SE crevices)
      - Water caustics (luminous organic caustic web lines, flow-aligned streaks, frothy foam)
    """
    b, _, h, w = mass.shape
    device = mass.device

    tot = mass.sum(dim=1, keepdim=True)
    f = mass / torch.clamp(tot, min=1e-4)
    f_water = f[:, S_WATER:S_WATER + 1]
    f_ice = f[:, S_ICE:S_ICE + 1]
    f_steam = f[:, S_STEAM:S_STEAM + 1]
    f_soil = f[:, S_SOIL:S_SOIL + 1]
    f_rock = f[:, S_ROCK:S_ROCK + 1]
    f_wood = f[:, S_WOOD:S_WOOD + 1]
    f_ash = f[:, S_ASH:S_ASH + 1]
    f_smoke = f[:, S_SMOKE:S_SMOKE + 1]
    f_oil = f[:, S_OIL:S_OIL + 1]
    f_plant = f[:, S_PLANT:S_PLANT + 1]

    temp_n = torch.clamp((t_ck - 20000.0) / 90000.0, 0.0, 1.0)
    burning = compute_burning_mask(mass, t_ck).float()
    crit = torch.clamp(critter_norm, 0.0, 1.0)

    # Quantized flow vector matching A+ wire schema
    mag = torch.sqrt(flow_e * flow_e + flow_s * flow_s)
    ang = torch.atan2(flow_s, flow_e)
    dir_q = torch.remainder(torch.round(ang / (math.pi / 4.0)), 8.0) * (math.pi / 4.0)
    spd = torch.where(mag < 15.0, torch.zeros_like(mag), torch.where(mag >= 100.0, torch.ones_like(mag), torch.full_like(mag, 0.5)))
    fx = torch.cos(dir_q) * spd
    fy = torch.sin(dir_q) * spd

    t_view = t_sec.view(b, 1, 1, 1).to(device=device, dtype=torch.float32)
    tsin = torch.sin(2.5 * t_view)
    tcos = torch.cos(2.5 * t_view)

    # 128x128 macro terrain geometry & cast shadows
    land = torch.clamp(f_soil + f_rock + f_wood + f_plant + f_ash, 0.0, 1.0)
    land_tight = gaussian_blur(land + f_ice * 0.62, 1.0)
    land_wide = gaussian_blur(land + f_ice * 0.52, 3.0)
    shallow = torch.clamp(land_wide * 1.65, 0.0, 1.0)
    foam_band = torch.clamp((land_tight - 0.08) * 3.4, 0.0, 1.0) * torch.clamp((0.60 - land_tight) * 3.4, 0.0, 1.0)

    elev = (
        0.04 * f_water + 0.16 * f_ice + 0.32 * f_soil + 0.28 * f_ash
        + 0.48 * f_plant + 0.86 * f_wood + 1.00 * f_rock
    )
    gx = shift2d(elev, 0, -1) - shift2d(elev, 0, 1)
    gy = shift2d(elev, -1, 0) - shift2d(elev, 1, 0)
    nw_bevel = torch.clamp((gx + gy) * 1.85, -1.0, 1.0)

    rock_shad = torch.clamp(
        0.62 * shift2d(f_rock, 1, 1) + 0.46 * shift2d(f_rock, 2, 2) + 0.26 * shift2d(f_rock, 3, 2),
        0.0, 1.0,
    ) * (1.0 - f_rock)
    wood_shad = torch.clamp(
        0.52 * shift2d(f_wood, 1, 1) + 0.30 * shift2d(f_wood, 2, 1),
        0.0, 1.0,
    ) * (1.0 - f_wood) * (1.0 - f_rock)
    cast_shad = gaussian_blur(torch.clamp(rock_shad * 0.80 + wood_shad * 0.58, 0.0, 0.85), 0.65)
    ao = torch.clamp(gaussian_blur(f_rock * 0.85 + f_wood * 0.55, 1.2) * (1.0 - f_rock * 0.72 - f_wood * 0.45), 0.0, 0.48)

    # Upsample to 512x512 (smooth organic contours + crisp thresholded rock/pine silhouettes)
    w_u = _up4(f_water)
    i_u = _up4(f_ice)
    st_u = gaussian_blur(_up4(f_steam), 2.4)
    so_u = _up4(f_soil)
    rk_u = torch.clamp((_up4(f_rock) - 0.12) / 0.74, 0.0, 1.0)
    wd_u = torch.clamp((_up4(f_wood) - 0.10) / 0.78, 0.0, 1.0)
    as_u = _up4(f_ash)
    sm_u = gaussian_blur(_up4(f_smoke), 2.6)
    oi_u = _up4(f_oil)
    pl_u = _up4(f_plant)

    temp_u = _up4(temp_n)
    burn_u = _up4(burning)
    fx_u = _up4(fx)
    fy_u = _up4(fy)
    spd_u = torch.sqrt(fx_u * fx_u + fy_u * fy_u + 1e-6)
    shallow_u = _up4(shallow)
    foam_u = _up4(foam_band)
    nw_u = _up4(nw_bevel)
    shad_u = _up4(cast_shad)
    ao_u = _up4(ao)

    # Smooth organic world-locked 512x512 procedural coordinates (ZERO 4x4 grid periodicity!)
    wt4 = _up4(world_tex_128)  # [B, 2, 512, 512]
    wu = (wt4[:, 0:1] - 0.5) * 2.0  # [-1, 1]
    wv = (wt4[:, 1:2] - 0.5) * 2.0  # [-1, 1]
    # Organic multi-scale domain-warped brush fields derived purely from (wu, wv) & local gradients!
    w_gx = (shift2d(wu, 0, -2) - shift2d(wu, 0, 2)) * 4.5
    w_gy = (shift2d(wv, -2, 0) - shift2d(wv, 2, 0)) * 4.5
    warp_u = wu + 0.28 * torch.sin(3.6 * wv + 1.4 * w_gx)
    warp_v = wv + 0.28 * torch.cos(3.4 * wu - 1.4 * w_gy)

    brush_coarse = 0.5 + 0.5 * torch.sin(4.2 * warp_u + 3.1 * warp_v) * torch.cos(3.5 * warp_v - 4.0 * warp_u)
    brush_fine = 0.5 + 0.5 * torch.sin(9.5 * warp_u - 6.8 * warp_v) * torch.cos(7.4 * warp_u + 8.6 * warp_v)
    brush_stroke = 0.55 * brush_coarse + 0.45 * brush_fine

    # Crisp 512x512 directional surface normals
    rk_nw = torch.clamp((shift2d(rk_u, 0, -2) - shift2d(rk_u, 0, 2) + shift2d(rk_u, -2, 0) - shift2d(rk_u, 2, 0)) * 1.9, -1.0, 1.0)
    ic_nw = torch.clamp((shift2d(i_u, 0, -2) - shift2d(i_u, 0, 2) + shift2d(i_u, -2, 0) - shift2d(i_u, 2, 0)) * 1.8, -1.0, 1.0)
    wd_nw = torch.clamp((shift2d(wd_u, 0, -2) - shift2d(wd_u, 0, 2) + shift2d(wd_u, -2, 0) - shift2d(wd_u, 2, 0)) * 1.7, -1.0, 1.0)

    # --- 0. WATER: Luminous Organic Caustic Web + Flow-Aligned Ripples + Frothy Foam ---
    c1 = torch.sin(6.8 * warp_u + 4.4 * warp_v) * tcos + torch.cos(6.8 * warp_u + 4.4 * warp_v) * tsin
    c2 = torch.sin(6.2 * warp_v - 5.1 * warp_u) * tcos - torch.cos(6.2 * warp_v - 5.1 * warp_u) * tsin
    caustic_web = torch.pow(torch.clamp(1.0 - torch.abs(c1 + c2) * 0.72, 0.0, 1.0), 3.2)
    caustic_web2 = torch.pow(torch.clamp(1.0 - torch.abs(c1 - c2) * 0.78, 0.0, 1.0), 4.0)
    caustics = torch.clamp(caustic_web * 0.75 + caustic_web2 * 0.55, 0.0, 1.0)

    flow_phase = 7.5 * (fx_u * warp_u + fy_u * warp_v) + 6.5 * shallow_u
    ripple = torch.clamp(0.5 + 0.5 * (torch.sin(flow_phase) * tcos + torch.cos(flow_phase) * tsin), 0.0, 1.0)
    ripple_crest = torch.pow(ripple, 3.0) * (0.45 + 0.90 * spd_u)
    foam_crisp = torch.clamp(foam_u * (0.62 + 0.65 * brush_fine + 0.25 * c1), 0.0, 1.0)
    col_water = (
        rgb_const(0.042, 0.145, 0.295, device) * (1.0 - shallow_u)
        + rgb_const(0.095, 0.400, 0.515, device) * shallow_u
        + rgb_const(0.36, 0.74, 0.80, device) * caustics * (0.30 + 0.70 * shallow_u) * 0.62
        + rgb_const(0.30, 0.52, 0.62, device) * ripple_crest * 0.78
        + rgb_const(0.86, 0.96, 1.00, device) * torch.pow(foam_crisp, 1.25) * 0.80
    )

    # --- 1. ICE: Bold Fractured Ice Veins + Glacier Crystal Facets + Subsurface Cyan ---
    vein1 = torch.abs(torch.sin(5.4 * warp_u + 4.8 * warp_v))
    vein2 = torch.abs(torch.sin(6.1 * warp_v - 5.3 * warp_u + 1.1))
    ice_crack = torch.pow(torch.clamp(1.0 - torch.minimum(vein1, vein2) * 1.85, 0.0, 1.0), 2.8)
    # Frosted specular ridge right next to each crack
    crack_rim = torch.clamp(shift2d(ice_crack, 1, 1) - ice_crack * 0.5, 0.0, 1.0)
    ice_facet = torch.clamp(0.5 + 0.5 * torch.sin(4.0 * wu - 3.6 * wv), 0.0, 1.0)
    ice_glint = torch.pow(torch.clamp(0.5 + 0.5 * (torch.sin(8.2 * warp_u) * tcos + torch.cos(8.2 * warp_v) * tsin), 0.0, 1.0), 4.5)
    col_ice = (
        rgb_const(0.32, 0.67, 0.90, device) * (1.0 - 0.50 * i_u)
        + rgb_const(0.76, 0.94, 0.99, device) * (0.50 * i_u)
        + rgb_const(0.26, 0.28, 0.30, device) * torch.clamp(ic_nw, 0.0, 1.0)
        - rgb_const(0.18, 0.16, 0.10, device) * torch.clamp(-ic_nw, 0.0, 1.0)
        - rgb_const(0.30, 0.20, 0.05, device) * ice_crack * 0.78  # deep sapphire fracture vein
        + rgb_const(0.30, 0.34, 0.36, device) * crack_rim * 0.85  # crisp frosted crystal rim
        + rgb_const(0.08, 0.13, 0.16, device) * (ice_facet - 0.5)
        + rgb_const(0.24, 0.28, 0.30, device) * ice_glint * 0.55
    )

    # --- 3. SOIL / WET MUD: Painterly Earth Strokes & Wet Sheen ---
    wetness = torch.clamp(w_u * 2.5, 0.0, 1.0)
    earth_dab = brush_stroke - 0.45
    col_dry_soil = (
        rgb_const(0.33, 0.25, 0.165, device)
        + rgb_const(0.10, 0.085, 0.058, device) * nw_u
        + rgb_const(0.095, 0.078, 0.052, device) * earth_dab
    )
    mud_sheen = torch.pow(torch.clamp(0.5 + 0.5 * torch.sin(6.5 * warp_u + 3.0 * nw_u) * tcos, 0.0, 1.0), 2.5) * wetness
    col_wet_mud = rgb_const(0.125, 0.102, 0.080, device) + rgb_const(0.18, 0.26, 0.34, device) * mud_sheen * 0.35
    col_soil = col_dry_soil * (1.0 - 0.74 * wetness) + col_wet_mud * (0.74 * wetness)

    # --- 4. ROCK CLIFFS: Directional Faceted Granite Strata, Crisp NW Ridges, Deep SE Crevices ---
    strata = torch.sin(7.2 * warp_u + 3.2 * warp_v)
    crevice = torch.pow(torch.clamp(1.0 - torch.abs(strata) * 1.45, 0.0, 1.0), 2.6)
    facet_dir = torch.clamp(rk_nw * 1.15 + 0.48 * strata + 0.22 * (brush_coarse - 0.5), -1.0, 1.0)
    col_rock = (
        rgb_const(0.30, 0.33, 0.39, device)
        + rgb_const(0.38, 0.39, 0.41, device) * torch.clamp(facet_dir, 0.0, 1.0)
        - rgb_const(0.24, 0.25, 0.27, device) * torch.clamp(-facet_dir, 0.0, 1.0)
        - rgb_const(0.18, 0.18, 0.20, device) * crevice * 0.75
    )

    # --- 5. PINE CANOPY BRUSHWORK: Layered Conifer Bough Tufts & Sunlit Needle Clusters ---
    bough_wave = torch.sin(8.4 * warp_u + 5.2 * warp_v) * torch.cos(6.6 * warp_v - 7.8 * warp_u)
    bough_tuft = torch.clamp(0.5 + 0.65 * bough_wave, 0.0, 1.0)
    bough_edge = torch.clamp(shift2d(bough_tuft, -1, -1) - shift2d(bough_tuft, 1, 1), -1.0, 1.0)
    canopy_lit = torch.clamp(0.44 + 0.78 * wd_nw + 0.48 * (bough_tuft - 0.5) + 0.40 * bough_edge, 0.0, 1.0)
    col_wood = (
        rgb_const(0.042, 0.145, 0.088, device) * (1.0 - canopy_lit)
        + rgb_const(0.180, 0.445, 0.195, device) * canopy_lit
        + rgb_const(0.15, 0.25, 0.07, device) * torch.pow(canopy_lit * bough_tuft, 2.2) * 0.75
        - rgb_const(0.05, 0.08, 0.05, device) * torch.pow(1.0 - bough_tuft, 2.0) * 0.60
    )

    # --- 6. ASH: Charred Silt + Crisp Smoldering Ember Veins ---
    ember_heat = torch.clamp((temp_u - 0.24) / 0.44, 0.0, 1.0)
    ember_vein = torch.pow(torch.clamp(1.0 - torch.abs(torch.sin(8.5 * warp_u + 6.0 * warp_v)) * 1.5, 0.0, 1.0), 2.8)
    ember_pulse = 0.65 + 0.35 * (torch.sin(6.5 * wv + 3.0 * as_u) * tcos + torch.cos(6.5 * wu) * tsin)
    col_ash = (
        rgb_const(0.115, 0.110, 0.122, device)
        + rgb_const(0.05, 0.05, 0.05, device) * (brush_stroke - 0.5)
        + rgb_const(1.00, 0.34, 0.04, device) * ember_heat * (0.35 + 0.90 * ember_vein) * ember_pulse * 0.88
    )

    # --- 9. PLANT: Crisp Organic Grass Blades & Sunlit Meadow Brushwork ---
    blade_sway = 0.20 * (torch.sin(4.0 * wu) * tcos + torch.cos(4.0 * wv) * tsin)
    # Anisotropic directional blade field (elongated strokes oriented NE-SW!)
    blade_coord = 11.5 * warp_u + 3.6 * warp_v + blade_sway * 3.5
    blade_cross = 5.2 * warp_v - 2.4 * warp_u
    blade_stroke = torch.sin(blade_coord) * (0.65 + 0.35 * torch.cos(blade_cross))
    blade_tip = torch.pow(torch.clamp(0.5 + 0.55 * blade_stroke, 0.0, 1.0), 2.4)
    blade_trough = torch.pow(torch.clamp(0.5 - 0.55 * blade_stroke, 0.0, 1.0), 2.2)
    col_plant = (
        rgb_const(0.110, 0.320, 0.140, device) * (1.0 - 0.45 * pl_u)
        + rgb_const(0.265, 0.560, 0.190, device) * (0.45 * pl_u)
        + rgb_const(0.26, 0.35, 0.06, device) * blade_tip * 0.82
        - rgb_const(0.07, 0.13, 0.05, device) * blade_trough * 0.60
    )

    # Composite base opaque substrate
    base_w = w_u + i_u + so_u + rk_u + wd_u + as_u + pl_u
    fallback_col = rgb_const(0.13, 0.17, 0.21, device)
    base_num = (
        w_u * col_water + i_u * col_ice + so_u * col_soil
        + rk_u * col_rock + wd_u * col_wood + as_u * col_ash + pl_u * col_plant
    )
    has_base = torch.clamp(base_w / 0.12, 0.0, 1.0)
    rgb = (base_num / torch.clamp(base_w, min=1e-4)) * has_base + fallback_col * (1.0 - has_base)

    # Apply directional cast shadows & AO
    shadow_factor = 1.0 - (shad_u * 0.55 + ao_u * 0.32) * (1.0 - burn_u * 0.85)
    rgb = rgb * torch.clamp(shadow_factor, 0.35, 1.0)

    # --- Oil Thin-Film Iridescent Slick ---
    oil_alpha = torch.clamp(oi_u * 1.45, 0.0, 0.88)
    iri_phase = 8.5 * oi_u + 5.2 * warp_u + 3.8 * warp_v
    iri_r = 0.5 + 0.5 * (torch.sin(iri_phase) * tcos + torch.cos(iri_phase) * tsin)
    iri_g = 0.5 + 0.5 * (torch.sin(iri_phase + 2.094) * tcos + torch.cos(iri_phase + 2.094) * tsin)
    iri_b = 0.5 + 0.5 * (torch.sin(iri_phase + 4.188) * tcos + torch.cos(iri_phase + 4.188) * tsin)
    iri_col = torch.cat([iri_r, iri_g, iri_b], dim=1)
    col_oil = rgb_const(0.05, 0.045, 0.09, device) + iri_col * (0.30 + 0.24 * oi_u)
    rgb = rgb * (1.0 - oil_alpha * 0.78) + col_oil * oil_alpha + rgb * iri_col * (0.24 * oil_alpha)

    # --- Active Burning Flames + Crisp Glowing Ember Sparks ---
    flicker = 0.78 + 0.22 * (torch.sin(6.0 * warp_u) * tcos + torch.cos(6.0 * warp_v) * tsin)
    flame_tongue = torch.clamp(0.50 + 0.60 * torch.sin(7.8 * warp_u - 4.2 * warp_v) * tcos, 0.0, 1.0)
    flame_str = torch.clamp(burn_u * (0.72 + 0.48 * temp_u) * flicker * (0.72 + 0.40 * flame_tongue), 0.0, 1.0)
    col_fire = (
        rgb_const(0.95, 0.18, 0.02, device) * (1.0 - flame_str)
        + rgb_const(1.00, 0.72, 0.14, device) * flame_str
        + rgb_const(0.45, 0.46, 0.55, device) * torch.pow(flame_str, 2.3)
    )
    flame_alpha = torch.clamp(flame_str * 0.86, 0.0, 0.90)
    rgb = rgb * (1.0 - flame_alpha) + col_fire * flame_alpha

    # Crisp glowing ember sparks scattered around active fires (organic world-locked spark peaks!)
    fire_halo = _up4(gaussian_blur(burning, 1.8))
    sp1 = torch.sin(10.5 * warp_u + 4.8 * warp_v) * tcos + torch.cos(10.5 * warp_u + 4.8 * warp_v) * tsin
    sp2 = torch.cos(9.8 * warp_v - 5.4 * warp_u) * tcos - torch.sin(9.8 * warp_v - 5.4 * warp_u) * tsin
    spark_pin = torch.pow(torch.clamp(sp1 * sp2, 0.0, 1.0), 4.0) * torch.clamp(fire_halo * 2.4, 0.0, 1.0)
    rgb = rgb + spark_pin * rgb_const(1.55, 0.88, 0.22, device)

    # --- Multi-Scale Dynamic Lighting (Fire Glow + Frost Glow + Critter Glow) ---
    warm_src = burning * (0.95 + 0.15 * tsin) + torch.clamp((temp_n - 0.38) / 0.55, 0.0, 1.0) * 0.42
    warm_tight = _up4(gaussian_blur(warm_src, 2.0))
    warm_wide = _up4(gaussian_blur(warm_src, 5.5))

    frost_src = torch.clamp((0.095 - temp_n) / 0.095, 0.0, 1.0) * (0.45 + 0.65 * f_ice)
    frost_tight = _up4(gaussian_blur(frost_src, 2.0))
    frost_wide = _up4(gaussian_blur(frost_src, 5.0))
    crit_glow = _up4(gaussian_blur(crit * 0.72, 1.8))

    spec_boost = 1.0 + 0.55 * torch.clamp(w_u + i_u * 0.95 + oi_u * 0.90, 0.0, 1.0)
    rgb = (
        rgb
        + (warm_tight * 0.58 + warm_wide * 0.42) * spec_boost * rgb_const(0.98, 0.44, 0.09, device) * 0.64
        + (frost_tight * 0.50 + frost_wide * 0.40) * spec_boost * rgb_const(0.20, 0.84, 1.00, device) * 0.48
        + crit_glow * rgb_const(1.00, 0.82, 0.28, device) * 0.36
    )

    # --- Painterly Smoke & Steam Plumes ---
    puff = 0.76 + 0.24 * torch.sin(5.6 * warp_u + 4.8 * warp_v)
    smoke_alpha = torch.clamp(sm_u * 1.15 * puff, 0.0, 0.80)
    smoke_col = (
        rgb_const(0.13, 0.14, 0.17, device)
        + rgb_const(0.08, 0.08, 0.09, device) * brush_coarse
        + torch.clamp(warm_wide * 1.15 + warm_tight * 0.65, 0.0, 1.25) * rgb_const(0.78, 0.28, 0.06, device)
    )
    rgb = rgb * (1.0 - smoke_alpha) + smoke_col * smoke_alpha

    steam_swirl = 0.82 + 0.18 * (torch.sin(6.4 * warp_u) * tcos + torch.cos(6.4 * warp_v) * tsin)
    steam_alpha = torch.clamp(st_u * 1.20 * steam_swirl, 0.0, 0.76)
    steam_col = (
        rgb_const(0.82, 0.92, 0.97, device)
        + rgb_const(0.10, 0.10, 0.08, device) * (brush_coarse - 0.5)
        + torch.clamp(warm_wide, 0.0, 0.8) * rgb_const(0.30, 0.16, 0.06, device)
    )
    rgb = rgb * (1.0 - steam_alpha) + steam_col * steam_alpha

    # --- Glowing Living Wildlife Herds ---
    crit_u = _up4(crit)
    crit_shad = shift2d(crit_u, 4, 4) * (1.0 - crit_u)
    rgb = rgb * (1.0 - 0.38 * crit_shad)
    crit_alpha = torch.clamp((crit_u - 0.10) / 0.52, 0.0, 1.0)
    crit_col = rgb_const(0.99, 0.78, 0.22, device) + rgb_const(0.16, 0.22, 0.40, device) * brush_coarse
    rgb = rgb * (1.0 - crit_alpha) + crit_col * crit_alpha

    # Filmic tone mapping in [0.012, 0.988]
    rgb = torch.clamp(rgb, min=0.0)
    rgb = rgb / (1.0 + 0.15 * rgb)
    return torch.clamp(rgb, 0.012, 0.988)


# =============================================================================
# 3. ~400k-Param Checkerboard-Free 4x Painterly Generator + Multi-Scale PatchGAN
# =============================================================================
class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.gelu(self.conv1(x)))


def _world_harmonic_features(wtex: torch.Tensor) -> torch.Tensor:
    """
    Expand 2-ch world texture coords [B, 2, H, W] into 10-ch multi-frequency harmonic basis
    (standard ONNX Sin/Cos ops!) so upsampled stages synthesize sharp world-locked brushwork.
    """
    u = (wtex[:, 0:1] - 0.5) * 2.0
    v = (wtex[:, 1:2] - 0.5) * 2.0
    return torch.cat([
        u, v,
        torch.sin(4.0 * u), torch.cos(4.0 * u),
        torch.sin(4.0 * v), torch.cos(4.0 * v),
        torch.sin(8.0 * u + 4.0 * v), torch.cos(8.0 * u - 4.0 * v),
        torch.sin(8.0 * v - 4.0 * u), torch.cos(8.0 * v + 4.0 * u),
    ], dim=1)


class Tier1PainterlyNet(nn.Module):
    """
    ~390k-param Checkerboard-Free 4x Upsampling Painterly RPG Renderer:
      Input:  aplus_world [B, 19, H, W] (17 A+ channels + 2 world-space texture coords)
      Output: rgb_512     [B, 3, 4*H, 4*W] in [0, 1] (Sigmoid)
    """

    def __init__(self, in_ch: int = 19, base_ch: int = 68, glow_ch: int = 32, mid_ch: int = 48, hi_ch: int = 24):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        # Multi-scale global lighting / shadow / glow pyramid at 1/4 and 1/8
        self.glow_in = nn.Conv2d(base_ch, glow_ch, kernel_size=1)
        self.glow_c1 = nn.Conv2d(glow_ch, glow_ch, kernel_size=3, padding=1)
        self.glow_c2 = nn.Conv2d(glow_ch, glow_ch, kernel_size=3, padding=1)
        self.glow_c3 = nn.Conv2d(glow_ch, glow_ch, kernel_size=3, padding=1)
        self.glow_out = nn.Conv2d(glow_ch, base_ch, kernel_size=1)

        # Dilated residual backbone at HxW (128x128)
        self.res1 = DilatedResBlock(base_ch, dilation=1)
        self.res2 = DilatedResBlock(base_ch, dilation=2)
        self.res3 = DilatedResBlock(base_ch, dilation=4)

        # Stage 1 upsample: HxW -> 2Hx2W (checkerboard-free bilinear + world-harmonic fusion)
        self.up1_fuse = nn.Conv2d(base_ch + 10, mid_ch, kernel_size=3, padding=1)
        self.mid_res = DilatedResBlock(mid_ch, dilation=1)

        # Stage 2 upsample: 2Hx2W -> 4Hx4W (fast 24-ch checkerboard-free micro-brushwork head)
        self.up2_fuse = nn.Conv2d(mid_ch + 10, hi_ch, kernel_size=3, padding=1)
        self.hi_conv1 = nn.Conv2d(hi_ch, hi_ch, kernel_size=3, padding=1)
        self.hi_out = nn.Conv2d(hi_ch, 3, kernel_size=3, padding=1)

    def forward(self, aplus_world: torch.Tensor) -> torch.Tensor:
        wtex = aplus_world[:, 17:19]  # [B, 2, H, W]
        h = F.gelu(self.stem(aplus_world))

        g4 = F.avg_pool2d(self.glow_in(h), kernel_size=4, stride=4)
        g4 = F.gelu(self.glow_c1(g4))
        g8 = F.gelu(self.glow_c2(F.avg_pool2d(g4, kernel_size=2, stride=2)))
        g8_up = F.interpolate(g8, size=g4.shape[-2:], mode="bilinear", align_corners=False)
        g4 = F.gelu(self.glow_c3(g4 + g8_up))
        g_up = self.glow_out(F.interpolate(g4, size=h.shape[-2:], mode="bilinear", align_corners=False))

        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h) + g_up

        # 2x stage (bilinear -> zero checkerboard!)
        u2 = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        wt2 = _world_harmonic_features(F.interpolate(wtex, size=u2.shape[-2:], mode="bilinear", align_corners=False))
        u2 = self.mid_res(F.gelu(self.up1_fuse(torch.cat([u2, wt2], dim=1))))

        # 4x stage (bilinear -> crisp organic painterly brushwork!)
        u4 = F.interpolate(u2, scale_factor=2, mode="bilinear", align_corners=False)
        wt4 = _world_harmonic_features(F.interpolate(wtex, size=u4.shape[-2:], mode="bilinear", align_corners=False))
        u4 = F.gelu(self.up2_fuse(torch.cat([u4, wt4], dim=1)))
        u4 = u4 + 0.5 * F.gelu(self.hi_conv1(u4))
        return torch.sigmoid(self.hi_out(u4))


class PatchDiscriminator(nn.Module):
    """Single-scale spectral-normalized PatchGAN discriminator returning logits + intermediate features."""

    def __init__(self, in_ch: int = 6, base: int = 32):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.c1 = sn(nn.Conv2d(in_ch, base, 4, stride=2, padding=1))
        self.c2 = sn(nn.Conv2d(base, base * 2, 4, stride=2, padding=1))
        self.c3 = sn(nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1))
        self.c4 = sn(nn.Conv2d(base * 4, base * 4, 4, stride=1, padding=1))
        self.out = sn(nn.Conv2d(base * 4, 1, 4, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        f1 = F.leaky_relu(self.c1(x), 0.2, inplace=True)
        f2 = F.leaky_relu(self.c2(f1), 0.2, inplace=True)
        f3 = F.leaky_relu(self.c3(f2), 0.2, inplace=True)
        f4 = F.leaky_relu(self.c4(f3), 0.2, inplace=True)
        logits = self.out(f4)
        return logits, [f1, f2, f3, f4]


class MultiScalePatchGAN(nn.Module):
    """2-Scale PatchGAN Discriminator operating at 512x512 (fine brushwork) and 256x256 (macro form)."""

    def __init__(self, in_ch: int = 6):
        super().__init__()
        self.d_hi = PatchDiscriminator(in_ch=in_ch, base=32)
        self.d_lo = PatchDiscriminator(in_ch=in_ch, base=32)

    def forward(self, rgb: torch.Tensor, cond_512: torch.Tensor):
        x_hi = torch.cat([rgb, cond_512], dim=1)
        x_lo = F.avg_pool2d(x_hi, kernel_size=2, stride=2)
        return self.d_hi(x_hi), self.d_lo(x_lo)


# =============================================================================
# 4. Losses & Quality / Sharpness / Stability Metrics
# =============================================================================
class MultiScaleSobelLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, device=device) / 4.0
        ky = kx.t().contiguous()
        self.register_buffer("kx", kx.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("ky", ky.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def grad_mag(self, img: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(img, self.kx, padding=1, groups=3)
        gy = F.conv2d(img, self.ky, padding=1, groups=3)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        p, t = pred, target
        for w_scale in (1.0, 0.5):
            px = F.conv2d(p, self.kx, padding=1, groups=3)
            py = F.conv2d(p, self.ky, padding=1, groups=3)
            tx = F.conv2d(t, self.kx, padding=1, groups=3)
            ty = F.conv2d(t, self.ky, padding=1, groups=3)
            loss = loss + w_scale * (F.l1_loss(px, tx) + F.l1_loss(py, ty))
            p = F.avg_pool2d(p, 2, 2)
            t = F.avg_pool2d(t, 2, 2)
        return loss


def eval_psnr_mae(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    diff = pred.float() - target.float()
    mse = float((diff * diff).mean().item())
    mae = float(diff.abs().mean().item())
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    return psnr, mae


@torch.no_grad()
def measure_edge_contrast(img: torch.Tensor, sobel: MultiScaleSobelLoss) -> dict[str, float]:
    gmag = sobel.grad_mag(img.float())
    lap_k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=img.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    lap = F.conv2d(img.float(), lap_k, padding=1, groups=3)
    return {
        "sobel_mean": round(float(gmag.mean().item()), 5),
        "sobel_p90": round(float(torch.quantile(gmag.flatten(), 0.90).item()), 5),
        "laplacian_rms": round(float(torch.sqrt((lap * lap).mean()).item()), 5),
    }


def _make_disc_cond_512(aplus_world: torch.Tensor) -> torch.Tensor:
    c0 = aplus_world[:, S_WATER:S_WATER + 1] + aplus_world[:, S_ICE:S_ICE + 1]
    c1 = aplus_world[:, S_WOOD:S_WOOD + 1] + aplus_world[:, S_ROCK:S_ROCK + 1] + aplus_world[:, S_PLANT:S_PLANT + 1]
    c2 = 0.6 * aplus_world[:, 11:12] + 0.4 * aplus_world[:, 10:11]
    cond = torch.cat([c0, c1, c2], dim=1)
    return F.interpolate(cond, scale_factor=4, mode="bilinear", align_corners=False)


# =============================================================================
# 5. Training Loop (L1 + Sobel + Multi-Scale PatchGAN, B=8)
# =============================================================================
def _sample_t1_batch(split: dict[str, torch.Tensor], batch_size: int, device: torch.device):
    n = split["mass"].shape[0]
    idx = torch.randint(0, n, (batch_size,))
    mass = split["mass"][idx].to(device, non_blocking=True)
    t_ck = split["t_ck"][idx].to(device, non_blocking=True)
    flow_e = split["flow_e"][idx].to(device, non_blocking=True)
    flow_s = split["flow_s"][idx].to(device, non_blocking=True)
    crit = split["critter"][idx].to(device, non_blocking=True)
    t_sec = torch.rand((batch_size,), device=device) * (2.0 * math.pi / 2.5)
    ox = torch.randint(-64, 65, (batch_size,), device=device).float()
    oy = torch.randint(-64, 65, (batch_size,), device=device).float()

    if torch.rand(1).item() < 0.5:
        mass = torch.flip(mass, dims=[3])
        t_ck = torch.flip(t_ck, dims=[3])
        flow_e = -torch.flip(flow_e, dims=[3])
        flow_s = torch.flip(flow_s, dims=[3])
        crit = torch.flip(crit, dims=[3])
    if torch.rand(1).item() < 0.5:
        mass = torch.flip(mass, dims=[2])
        t_ck = torch.flip(t_ck, dims=[2])
        flow_e = torch.flip(flow_e, dims=[2])
        flow_s = -torch.flip(flow_s, dims=[2])
        crit = torch.flip(crit, dims=[2])
    return mass, t_ck, flow_e, flow_s, crit, t_sec, ox, oy


def train_tier1(
    train_split: dict[str, torch.Tensor],
    device: torch.device,
    steps: int = 1400,
    warmup_steps: int = 300,
    batch_size: int = 8,
    seed: int = 1001,
) -> tuple[Tier1PainterlyNet, dict]:
    torch.manual_seed(seed)
    gen = Tier1PainterlyNet(in_ch=19, base_ch=68, glow_ch=32, mid_ch=48, hi_ch=24).to(device)
    disc = MultiScalePatchGAN(in_ch=6).to(device)
    sobel = MultiScaleSobelLoss(device)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=1.4e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=6.0e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=steps, eta_min=8e-5)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max(1, steps - warmup_steps), eta_min=4e-5)

    # Native Blackwell BF16: same dynamic range as FP32 -> zero NaN / overflow!
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    t0 = time.time()
    hist = {"g_l1": 0.0, "g_sobel": 0.0, "g_adv": 0.0, "g_fm": 0.0, "d_loss": 0.0}

    for step in range(1, steps + 1):
        mass, t_ck, fe, fs, crit, t_sec, ox, oy = _sample_t1_batch(train_split, batch_size, device)
        with torch.no_grad():
            wire = encode_aplus_wire(mass, t_ck, fe, fs, crit)
            aplus = decode_aplus_wire(wire, t_sec)
            wtex = make_world_tex_coords(batch_size, 128, 128, device, ox=ox, oy=oy)
            x_in = torch.cat([aplus, wtex], dim=1)
            target = render_painterly_target_512(mass, t_ck, fe, fs, crit, t_sec, wtex)
            cond_512 = _make_disc_cond_512(x_in)

        use_gan = step > warmup_steps

        # --- Discriminator step ---
        d_loss_val = 0.0
        if use_gan:
            opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                with torch.no_grad():
                    fake_det = gen(x_in)
                (r_hi, _), (r_lo, _) = disc(target, cond_512)
                (f_hi, _), (f_lo, _) = disc(fake_det, cond_512)
                d_loss = 0.5 * (
                    (F.relu(1.0 - r_hi).mean() + F.relu(1.0 + f_hi).mean())
                    + (F.relu(1.0 - r_lo).mean() + F.relu(1.0 + f_lo).mean())
                )
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 2.0)
            opt_d.step()
            sched_d.step()
            d_loss_val = float(d_loss.item())

        # --- Generator step ---
        opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
            pred = gen(x_in)
            l_l1 = F.l1_loss(pred, target)
            l_sobel = sobel(pred, target)
            loss_g = l_l1 + 0.35 * l_sobel
            l_adv_val = 0.0
            l_fm_val = 0.0
            if use_gan:
                (f_hi, f_feats_hi), (f_lo, f_feats_lo) = disc(pred, cond_512)
                with torch.no_grad():
                    (_, r_feats_hi), (_, r_feats_lo) = disc(target, cond_512)
                l_adv = -0.5 * (f_hi.mean() + f_lo.mean())
                l_fm = sum(F.l1_loss(ff, rf) for ff, rf in zip(f_feats_hi, r_feats_hi)) / len(f_feats_hi)
                l_fm = l_fm + sum(F.l1_loss(ff, rf) for ff, rf in zip(f_feats_lo, r_feats_lo)) / len(f_feats_lo)
                gan_ramp = min(1.0, (step - warmup_steps) / 200.0)
                loss_g = loss_g + (0.015 * gan_ramp) * l_adv + (0.22 * gan_ramp) * l_fm
                l_adv_val = float(l_adv.item())
                l_fm_val = float(l_fm.item())

        assert torch.isfinite(loss_g), f"Non-finite Tier 1 loss at step {step}: {loss_g}"
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.5)
        opt_g.step()
        sched_g.step()

        hist = {
            "g_l1": float(l_l1.item()),
            "g_sobel": float(l_sobel.item()),
            "g_adv": l_adv_val,
            "g_fm": l_fm_val,
            "d_loss": d_loss_val,
        }
        if step in (1, warmup_steps, 650, 1000, steps):
            print(
                f"  [Tier1] step {step:4d}/{steps} | L1={hist['g_l1']:.4f} Sobel={hist['g_sobel']:.4f} "
                f"Adv={hist['g_adv']:+.3f} FM={hist['g_fm']:.4f} D={hist['d_loss']:.3f} ({time.time() - t0:.1f}s)",
                flush=True,
            )

    gen.eval()
    return gen, {"train_time_s": round(time.time() - t0, 1), "steps": steps, "warmup_steps": warmup_steps, "batch_size": batch_size, "final_losses": hist}


# =============================================================================
# 6. Evaluation: PSNR, Edge Contrast, Temporal Stability (Static & Shifting), Showcase
# =============================================================================
@torch.no_grad()
def evaluate_tier1(
    gen: Tier1PainterlyNet,
    val_split: dict[str, torch.Tensor],
    hero_scene: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict, dict]:
    gen.eval()
    sobel = MultiScaleSobelLoss(device)

    v2_path = LATENT_DIR / "renderer_v2.onnx"
    v2_sess = ort.InferenceSession(str(v2_path), providers=["CPUExecutionProvider"])

    n_val = min(32, val_split["mass"].shape[0])
    psnrs_t1, maes_t1 = [], []
    psnrs_v2, maes_v2 = [], []
    ec_gt_list, ec_t1_list, ec_v2_list = [], [], []

    for i in range(0, n_val, 4):
        sl = slice(i, min(i + 4, n_val))
        b = sl.stop - sl.start
        mass = val_split["mass"][sl].to(device)
        t_ck = val_split["t_ck"][sl].to(device)
        fe = val_split["flow_e"][sl].to(device)
        fs = val_split["flow_s"][sl].to(device)
        crit = val_split["critter"][sl].to(device)
        t_sec = torch.linspace(0.2, 2.0, b, device=device)
        ox = torch.arange(i, i + b, device=device, dtype=torch.float32) * 7.0
        oy = torch.arange(i, i + b, device=device, dtype=torch.float32) * 5.0

        wire = encode_aplus_wire(mass, t_ck, fe, fs, crit)
        aplus = decode_aplus_wire(wire, t_sec)
        wtex = make_world_tex_coords(b, 128, 128, device, ox=ox, oy=oy)
        x_in = torch.cat([aplus, wtex], dim=1)
        gt = render_painterly_target_512(mass, t_ck, fe, fs, crit, t_sec, wtex)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred_t1 = gen(x_in).float()

        v2_256 = torch.from_numpy(v2_sess.run(["rgb_frame"], {"aplus_state": aplus.cpu().numpy()})[0]).to(device)
        v2_512 = F.interpolate(v2_256, size=(512, 512), mode="bilinear", align_corners=False)

        p1, m1 = eval_psnr_mae(pred_t1, gt)
        p2, m2 = eval_psnr_mae(v2_512, gt)
        psnrs_t1.append(p1)
        maes_t1.append(m1)
        psnrs_v2.append(p2)
        maes_v2.append(m2)

        ec_gt_list.append(measure_edge_contrast(gt, sobel))
        ec_t1_list.append(measure_edge_contrast(pred_t1, sobel))
        ec_v2_list.append(measure_edge_contrast(v2_512, sobel))

    def _avg_ec(lst):
        return {k: round(float(np.mean([d[k] for d in lst])), 5) for k in lst[0]}

    ec_gt = _avg_ec(ec_gt_list)
    ec_t1 = _avg_ec(ec_t1_list)
    ec_v2 = _avg_ec(ec_v2_list)

    # --- Temporal Stability Benchmark ---
    h_mass = hero_scene["mass"].to(device)
    h_tck = hero_scene["t_ck"].to(device)
    h_fe = hero_scene["flow_e"].to(device)
    h_fs = hero_scene["flow_s"].to(device)
    h_crit = hero_scene["critter"].to(device)
    h_wire = encode_aplus_wire(h_mass, h_tck, h_fe, h_fs, h_crit)
    wtex0 = make_world_tex_coords(1, 128, 128, device)

    tot_h = h_mass.sum(dim=1, keepdim=True)
    frac_h = h_mass / torch.clamp(tot_h, min=1e-4)
    solid_128 = ((frac_h[:, S_ROCK:S_ROCK + 1] + frac_h[:, S_SOIL:S_SOIL + 1] + frac_h[:, S_WOOD:S_WOOD + 1]) > 0.75) & (~compute_burning_mask(h_mass, h_tck))
    solid_512 = _up4(solid_128.float()) > 0.9

    static_preds, static_gts = [], []
    for k in range(12):
        tk = torch.tensor([k * 0.1], device=device)
        ap_k = decode_aplus_wire(h_wire, tk)
        x_k = torch.cat([ap_k, wtex0], dim=1)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            static_preds.append(gen(x_k).float())
        static_gts.append(render_painterly_target_512(h_mass, h_tck, h_fe, h_fs, h_crit, tk, wtex0))

    full_diffs, solid_diffs, gt_full_diffs, gt_solid_diffs = [], [], [], []
    for k in range(len(static_preds) - 1):
        d = static_preds[k + 1] - static_preds[k]
        dg = static_gts[k + 1] - static_gts[k]
        full_diffs.append(float(torch.sqrt((d * d).mean()).item()))
        gt_full_diffs.append(float(torch.sqrt((dg * dg).mean()).item()))
        solid_diffs.append(float(torch.sqrt((d[:, :, solid_512[0, 0]] ** 2).mean()).item()))
        gt_solid_diffs.append(float(torch.sqrt((dg[:, :, solid_512[0, 0]] ** 2).mean()).item()))

    # 2. Shifting scene flicker / anti-swimming test
    t_fix = torch.tensor([0.8], device=device)
    ap_full = decode_aplus_wire(h_wire, t_fix)
    wtex_full = make_world_tex_coords(1, 128, 128, device)
    x_ref = torch.cat([ap_full[:, :, 8:120, 8:120], wtex_full[:, :, 8:120, 8:120]], dim=1)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        y_ref = gen(x_ref).float()

    shift_rms_locked, shift_rms_unlocked = [], []
    for dx, dy in [(1, 0), (0, 1), (2, 1), (3, 2), (4, 3)]:
        ap_s = ap_full[:, :, 8 + dy:120 + dy, 8 + dx:120 + dx]
        wt_locked = wtex_full[:, :, 8 + dy:120 + dy, 8 + dx:120 + dx]
        wt_screen = wtex_full[:, :, 8:120, 8:120]
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            y_locked = gen(torch.cat([ap_s, wt_locked], dim=1)).float()
            y_unlocked = gen(torch.cat([ap_s, wt_screen], dim=1)).float()
        m = 16
        a = y_ref[:, :, 4 * dy + m:448 - m, 4 * dx + m:448 - m]
        b_l = y_locked[:, :, m:448 - 4 * dy - m, m:448 - 4 * dx - m]
        b_u = y_unlocked[:, :, m:448 - 4 * dy - m, m:448 - 4 * dx - m]
        shift_rms_locked.append(float(torch.sqrt(((a - b_l) ** 2).mean()).item()))
        shift_rms_unlocked.append(float(torch.sqrt(((a - b_u) ** 2).mean()).item()))

    vis_items = []
    for label, sc in [
        ("Hero Live Battle (Cliffs, Pines, River, Ice Bridge, Wildfire)", hero_scene),
        ("Validation Rollout (Marshland, Oil Slicks & Fire/Frost Bolts)", {k: v[2:3] for k, v in val_split.items()}),
    ]:
        m_i = sc["mass"].to(device)
        t_i = sc["t_ck"].to(device)
        fe_i = sc["flow_e"].to(device)
        fs_i = sc["flow_s"].to(device)
        cr_i = sc["critter"].to(device)
        ts_i = torch.tensor([0.9], device=device)
        wi_i = encode_aplus_wire(m_i, t_i, fe_i, fs_i, cr_i)
        ap_i = decode_aplus_wire(wi_i, ts_i)
        wt_i = make_world_tex_coords(1, 128, 128, device)
        gt_i = render_painterly_target_512(m_i, t_i, fe_i, fs_i, cr_i, ts_i, wt_i)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            t1_i = gen(torch.cat([ap_i, wt_i], dim=1)).float()
        v2_i = torch.from_numpy(v2_sess.run(["rgb_frame"], {"aplus_state": ap_i.cpu().numpy()})[0])
        raw_128 = render_raw_aplus_128(ap_i).cpu().numpy()[0]
        vis_items.append({
            "label": label,
            "raw_128": raw_128,
            "v2_256": v2_i.numpy()[0],
            "t1_512": t1_i.cpu().numpy()[0],
            "gt_512": gt_i.cpu().numpy()[0],
        })

    metrics = {
        "n_params": count_parameters(gen),
        "val_frames": n_val,
        "tier1_psnr_db": round(float(np.mean(psnrs_t1)), 2),
        "tier1_mae": round(float(np.mean(maes_t1)), 4),
        "renderer_v2_upscaled_psnr_db": round(float(np.mean(psnrs_v2)), 2),
        "renderer_v2_upscaled_mae": round(float(np.mean(maes_v2)), 4),
        "edge_contrast": {
            "target_512": ec_gt,
            "tier1_painterly_512": ec_t1,
            "renderer_v2_256_up512": ec_v2,
            "tier1_vs_v2_sobel_ratio": round(ec_t1["sobel_mean"] / max(ec_v2["sobel_mean"], 1e-6), 2),
            "tier1_vs_v2_laplacian_ratio": round(ec_t1["laplacian_rms"] / max(ec_v2["laplacian_rms"], 1e-6), 2),
        },
        "temporal_stability": {
            "static_scene_full_rms_flicker": round(float(np.mean(full_diffs)), 5),
            "static_scene_solid_terrain_rms_flicker": round(float(np.mean(solid_diffs)), 5),
            "target_static_full_rms": round(float(np.mean(gt_full_diffs)), 5),
            "target_static_solid_rms": round(float(np.mean(gt_solid_diffs)), 5),
            "shifting_scene_world_locked_rms": round(float(np.mean(shift_rms_locked)), 5),
            "shifting_scene_unlocked_ablation_rms": round(float(np.mean(shift_rms_unlocked)), 5),
            "anti_swimming_reduction_x": round(float(np.mean(shift_rms_unlocked)) / max(float(np.mean(shift_rms_locked)), 1e-6), 2),
        },
    }
    return metrics, {"items": vis_items}


@torch.no_grad()
def render_raw_aplus_128(aplus: torch.Tensor) -> torch.Tensor:
    dev = aplus.device
    palette = torch.tensor([
        [0.10, 0.34, 0.56],
        [0.64, 0.88, 0.98],
        [0.82, 0.90, 0.95],
        [0.34, 0.26, 0.17],
        [0.42, 0.45, 0.52],
        [0.14, 0.32, 0.16],
        [0.18, 0.18, 0.20],
        [0.24, 0.25, 0.28],
        [0.24, 0.10, 0.30],
        [0.22, 0.52, 0.20],
    ], dtype=torch.float32, device=dev).view(1, 10, 3, 1, 1)
    sp = aplus[:, 0:10].unsqueeze(2)
    rgb = (sp * palette).sum(dim=1)
    burn = aplus[:, 11:12]
    crit = aplus[:, 14:15]
    rgb = rgb * (1.0 - 0.85 * burn) + rgb_const(1.0, 0.52, 0.08, dev) * (0.85 * burn)
    rgb = rgb * (1.0 - 0.75 * crit) + rgb_const(1.0, 0.85, 0.22, dev) * (0.75 * crit)
    return torch.clamp(rgb, 0.0, 1.0)


# =============================================================================
# 7. ONNX Export (Opset 17, Single File) + GPU FP16 & CPU ORT Benchmarks
# =============================================================================
def export_and_benchmark_tier1(gen: Tier1PainterlyNet, out_onnx: Path, device: torch.device) -> dict:
    model_cpu = gen.cpu().eval()
    data_sidecar = Path(str(out_onnx) + ".data")
    for p in (data_sidecar, out_onnx):
        if p.exists():
            p.unlink()

    dummy = torch.randn(1, 19, 128, 128, dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        (dummy,),
        str(out_onnx),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["aplus_world"],
        output_names=["rgb_512"],
        dynamic_axes={
            "aplus_world": {0: "batch", 2: "height", 3: "width"},
            "rgb_512": {0: "batch", 2: "out_height", 3: "out_width"},
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
    assert in_meta.name == "aplus_world"
    assert out_meta.name == "rgb_512"

    parity = {}
    for shape in [(1, 19, 128, 128), (2, 19, 128, 128), (1, 19, 64, 64)]:
        x = torch.rand(*shape, dtype=torch.float32)
        with torch.inference_mode():
            y_pt = model_cpu(x).numpy()
        y_ort = sess.run(["rgb_512"], {"aplus_world": x.numpy()})[0]
        assert y_ort.shape == (shape[0], 3, shape[2] * 4, shape[3] * 4)
        assert float(y_ort.min()) >= 0.0 and float(y_ort.max()) <= 1.0
        max_diff = float(np.max(np.abs(y_pt - y_ort)))
        assert max_diff < 1e-4, f"Tier 1 ORT parity error {max_diff}"
        parity[f"{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}"] = max_diff
        print(f"  [tier1_painterly.onnx OK] {shape} -> {y_ort.shape}, max_abs_diff={max_diff:.2e}", flush=True)

    gpu_bench = {}
    if device.type == "cuda":
        gen_gpu = gen.to(device).eval().half()
        x_gpu = torch.rand(1, 19, 128, 128, device=device, dtype=torch.float16)
        with torch.inference_mode():
            for _ in range(25):
                _ = gen_gpu(x_gpu)
            torch.cuda.synchronize()
            times = []
            for _ in range(100):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                _ = gen_gpu(x_gpu)
                ev1.record()
                torch.cuda.synchronize()
                times.append(ev0.elapsed_time(ev1))
        gpu_bench["128_to_512_fp16"] = {
            "mean_ms": round(float(np.mean(times)), 3),
            "median_ms": round(float(np.median(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "fps": round(1000.0 / float(np.median(times)), 1),
        }
        gen.float().to(device)

    x_bench = np.random.rand(1, 19, 128, 128).astype(np.float32)
    cpu_bench = {}
    for n_thr in (1, 4, 8):
        so = ort.SessionOptions()
        so.intra_op_num_threads = n_thr
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s_t = ort.InferenceSession(str(out_onnx), so, providers=["CPUExecutionProvider"])
        for _ in range(3):
            s_t.run(["rgb_512"], {"aplus_world": x_bench})
        n_iters = 10 if n_thr == 1 else 20
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            s_t.run(["rgb_512"], {"aplus_world": x_bench})
            times.append((time.perf_counter() - t0) * 1000.0)
        cpu_bench[f"{n_thr}_threads"] = {
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
# 8. Visual Showcase PNG (`tier1_showcase.png`)
# =============================================================================
def _to_pil(chw: np.ndarray) -> Image.Image:
    hwc = np.clip(np.transpose(chw, (1, 2, 0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, mode="RGB")


def save_tier1_showcase_png(vis_data: dict, metrics: dict, out_png: Path) -> None:
    items = vis_data["items"]
    hero = items[0]
    val0 = items[1]

    tile = 384
    pad = 12
    header_h = 46
    row_lbl_h = 26
    cols = 4
    rows = 3

    cw = pad + cols * (tile + pad)
    ch = header_h + rows * (row_lbl_h + tile + pad) + pad
    canvas = Image.new("RGB", (cw, ch), color=(14, 17, 22))
    draw = ImageDraw.Draw(canvas)

    title = (
        f"Tier 1 High-Res 512x512 Painterly RPG Renderer  |  "
        f"PSNR: {metrics['tier1_psnr_db']:.2f} dB  |  "
        f"Sobel Sharpness: {metrics['edge_contrast']['tier1_vs_v2_sobel_ratio']:.2f}x vs renderer_v2  |  "
        f"Anti-Swim: {metrics['temporal_stability']['anti_swimming_reduction_x']:.1f}x"
    )
    draw.text((pad + 4, 8), title, fill=(245, 222, 145))

    col_headers = [
        "1. Raw 128x128 A+ State (Nearest)",
        "2. 256x256 L1 renderer_v2 (Smooth)",
        "3. 512x512 Tier 1 Painterly (GAN+WorldTex)",
        "4. 512x512 Painterly RPG Target (GT)",
    ]
    for ci, chdr in enumerate(col_headers):
        x0 = pad + ci * (tile + pad)
        draw.text((x0 + 4, 28), chdr, fill=(180, 225, 255) if ci != 2 else (130, 250, 175))

    def _prepare_row_images(item: dict, crop_frac: tuple[float, float, float, float] | None = None):
        imgs = [
            (_to_pil(item["raw_128"]), Image.Resampling.NEAREST),
            (_to_pil(item["v2_256"]), Image.Resampling.BILINEAR),
            (_to_pil(item["t1_512"]), Image.Resampling.LANCZOS),
            (_to_pil(item["gt_512"]), Image.Resampling.LANCZOS),
        ]
        out = []
        for im, resample in imgs:
            if crop_frac is not None:
                w0, h0 = im.size
                x1, y1, x2, y2 = crop_frac
                im = im.crop((int(w0 * x1), int(h0 * y1), int(w0 * x2), int(h0 * y2)))
            out.append(im.resize((tile, tile), resample))
        return out

    row_specs = [
        ("Row 1: Hero Live Frostmarsh Battle (Full 128x128 Arena -> 512x512)", _prepare_row_images(hero, None)),
        ("Row 2: 2.5x Micro-Detail Zoom (Cliffs, Pine Brushwork, Ice Veins, Water Caustics, Embers)", _prepare_row_images(hero, (0.28, 0.34, 0.68, 0.74))),
        ("Row 3: Held-Out Frostmarsh Rollout (Full 128x128 Arena -> 512x512)", _prepare_row_images(val0, None)),
    ]

    for ri, (rtitle, rimgs) in enumerate(row_specs):
        y_lbl = header_h + ri * (row_lbl_h + tile + pad)
        y_img = y_lbl + row_lbl_h
        draw.text((pad + 4, y_lbl + 4), rtitle, fill=(230, 236, 245))
        for ci, pim in enumerate(rimgs):
            x0 = pad + ci * (tile + pad)
            canvas.paste(pim, (x0, y_img))
            if ci == 2:
                draw.rectangle([x0, y_img, x0 + tile - 1, y_img + tile - 1], outline=(90, 235, 150), width=2)

    canvas.save(str(out_png), format="PNG", optimize=True)
