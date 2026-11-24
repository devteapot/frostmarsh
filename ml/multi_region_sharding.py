"""
Frostmarsh Wave 24B (MMO Phase 2B):
  - Seamless `2x2` Multi-Region Ghost-Halo Sharding (`HALO_XY = 6`)
  - Spatial `world_laws[6, H, W]` Field with 4-Zone Smooth Cosine Blending (`32m` borders)
  - Bit-Exact Monolith vs. `2x2` Sharded FP32 Seam Parity (`<= 1e-6` across `horiz`, `z_flux`, `Z_world_3d`)
  - `4,194,304`-Voxel (`512x512x16`, four `256x256x16` regions) Single-GPU Throughput Benchmark
    comparing 4-Region Dense vs. 4-Region Unified Sparse Brick Pool (`12%` aggregate MMO load)
    plus Ghost-Halo Bandwidth & 3-Panel Visualization (`models/multi_region_shard_viz.png`).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_teacher import encode_state  # noqa: E402
from world_model_3d_v5 import make_biome_3d_world  # noqa: E402
from world_model_3d_v8 import (  # noqa: E402
    DEFAULT_WORLD_LAWS,
    LAW_AMBIENT_TEMP_SHIFT,
    LAW_DIM,
    LAW_GRAVITY,
    LAW_NAMES,
    WORLD_LAW_PRESETS,
    SparseWorld3DV8Runner,
    WorldModel3DV8,
    count_parameters,
    encode_law_features,
)

HALO_XY = 6


# =============================================================================
# 1. 4-Zone Smooth Cosine-Blended Spatial World-Law Field (`[6, H, W]`)
# =============================================================================


def make_4zone_spatial_law_field(
    H: int,
    W: int,
    blend_width: int = 32,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a continuous `[6, H, W]` spatial World-Law field placing all 4 physical climates
    seamlessly in a `2x2` grid with `C^1` smooth cosine-blended `blend_width`-meter borders:
      - NW Quadrant (`y < H/2, x < W/2`): `freeze_out_survival`  (-55 K, 1.4x precip)
      - NE Quadrant (`y < H/2, x >= W/2`): `frostmarsh_standard` (0 K, 1.0g)
      - SW Quadrant (`y >= H/2, x < W/2`): `rising_flood_canyon` (2.6x precip, 0.55x visc)
      - SE Quadrant (`y >= H/2, x >= W/2`): `low_g_magma_crucible` (0.35g, +55 K, 2.2x vol)
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    nw = torch.tensor(WORLD_LAW_PRESETS["freeze_out_survival"], dtype=dtype, device=dev).view(LAW_DIM, 1, 1)
    ne = torch.tensor(WORLD_LAW_PRESETS["frostmarsh_standard"], dtype=dtype, device=dev).view(LAW_DIM, 1, 1)
    sw = torch.tensor(WORLD_LAW_PRESETS["rising_flood_canyon"], dtype=dtype, device=dev).view(LAW_DIM, 1, 1)
    se = torch.tensor(WORLD_LAW_PRESETS["low_g_magma_crucible"], dtype=dtype, device=dev).view(LAW_DIM, 1, 1)

    bw = max(1.0, float(blend_width))
    half = 0.5 * bw

    ys = torch.arange(H, dtype=dtype, device=dev) + 0.5
    xs = torch.arange(W, dtype=dtype, device=dev) + 0.5

    u_y = ((ys - (0.5 * H - half)) / bw).clamp(0.0, 1.0)
    u_x = ((xs - (0.5 * W - half)) / bw).clamp(0.0, 1.0)

    wy = (0.5 * (1.0 - torch.cos(math.pi * u_y))).view(1, H, 1)
    wx = (0.5 * (1.0 - torch.cos(math.pi * u_x))).view(1, 1, W)

    field = (
        (1.0 - wy) * (1.0 - wx) * nw
        + (1.0 - wy) * wx * ne
        + wy * (1.0 - wx) * sw
        + wy * wx * se
    )
    return field.contiguous()


# =============================================================================
# 2. `2x2` Ghost-Halo Exchange & Sharded Inference Helpers (`HALO_XY = 6`)
# =============================================================================


def split_2x2_shards(tensor: torch.Tensor) -> dict[tuple[int, int], torch.Tensor]:
    """Split a tensor with trailing `(..., H, W)` dims into 4 independent `(H/2, W/2)` shards."""
    H, W = tensor.shape[-2], tensor.shape[-1]
    assert H % 2 == 0 and W % 2 == 0, f"Expected even (H, W), got {(H, W)}"
    hh, ww = H // 2, W // 2
    return {
        (0, 0): tensor[..., :hh, :ww].clone().contiguous(),
        (0, 1): tensor[..., :hh, ww:].clone().contiguous(),
        (1, 0): tensor[..., hh:, :ww].clone().contiguous(),
        (1, 1): tensor[..., hh:, ww:].clone().contiguous(),
    }


def exchange_2x2_ghost_halos(
    shards: dict[tuple[int, int], torch.Tensor],
    halo: int = HALO_XY,
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Assemble ghost-halo-padded inputs for each of the 4 independent shards using ONLY:
      - the shard's own `(hh, ww)` tensor,
      - the `halo`-wide orthogonal border strip from its East/West & North/South neighbors, and
      - the `halo x halo` diagonal corner patch from its diagonal neighbor.

    Outer world borders are NOT padded here so `WorldModel3DV8`'s native replicate-padding
    matches the monolithic world boundary bit-for-bit!
      - `(0, 0)` NW: shape `(..., hh + halo, ww + halo)` (owned slice `[..., :hh, :ww]`)
      - `(0, 1)` NE: shape `(..., hh + halo, halo + ww)` (owned slice `[..., :hh, halo:]`)
      - `(1, 0)` SW: shape `(..., halo + hh, ww + halo)` (owned slice `[..., halo:, :ww]`)
      - `(1, 1)` SE: shape `(..., halo + hh, halo + ww)` (owned slice `[..., halo:, halo:]`)
    """
    s00, s01, s10, s11 = shards[(0, 0)], shards[(0, 1)], shards[(1, 0)], shards[(1, 1)]
    h = int(halo)
    if h <= 0:
        return {k: v.clone() for k, v in shards.items()}

    # NW (0,0): + East strip from (0,1), + South strip from (1,0), + SE corner from (1,1)
    top_00 = torch.cat([s00, s01[..., :, :h]], dim=-1)
    bot_00 = torch.cat([s10[..., :h, :], s11[..., :h, :h]], dim=-1)
    p00 = torch.cat([top_00, bot_00], dim=-2).contiguous()

    # NE (0,1): + West strip from (0,0), + South strip from (1,1), + SW corner from (1,0)
    top_01 = torch.cat([s00[..., :, -h:], s01], dim=-1)
    bot_01 = torch.cat([s10[..., :h, -h:], s11[..., :h, :]], dim=-1)
    p01 = torch.cat([top_01, bot_01], dim=-2).contiguous()

    # SW (1,0): + North strip from (0,0), + East strip from (1,1), + NE corner from (0,1)
    top_10 = torch.cat([s00[..., -h:, :], s01[..., -h:, :h]], dim=-1)
    bot_10 = torch.cat([s10, s11[..., :, :h]], dim=-1)
    p10 = torch.cat([top_10, bot_10], dim=-2).contiguous()

    # SE (1,1): + North strip from (0,1), + West strip from (1,0), + NW corner from (0,0)
    top_11 = torch.cat([s00[..., -h:, -h:], s01[..., -h:, :]], dim=-1)
    bot_11 = torch.cat([s10[..., :, -h:], s11], dim=-1)
    p11 = torch.cat([top_11, bot_11], dim=-2).contiguous()

    return {(0, 0): p00, (0, 1): p01, (1, 0): p10, (1, 1): p11}


def stitch_2x2_shards(
    padded_outputs: dict[tuple[int, int], torch.Tensor],
    halo: int = HALO_XY,
) -> torch.Tensor:
    """Crop each shard's ghost halo and stitch the 4 owned quadrants back into `(..., H, W)`."""
    h = int(halo)
    c00 = padded_outputs[(0, 0)][..., :-h, :-h] if h > 0 else padded_outputs[(0, 0)]
    c01 = padded_outputs[(0, 1)][..., :-h, h:] if h > 0 else padded_outputs[(0, 1)]
    c10 = padded_outputs[(1, 0)][..., h:, :-h] if h > 0 else padded_outputs[(1, 0)]
    c11 = padded_outputs[(1, 1)][..., h:, h:] if h > 0 else padded_outputs[(1, 1)]
    top = torch.cat([c00, c01], dim=-1)
    bot = torch.cat([c10, c11], dim=-1)
    return torch.cat([top, bot], dim=-2).contiguous()


def make_cross_seam_world_state(
    L: int = 16,
    H: int = 256,
    W: int = 256,
    device: torch.device | str = "cpu",
    seed: int = 2402,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct a physically encoded `[L, 18, H, W]` Frostmarsh 3D world state + `[H, W]` `local_dt`
    field (via `make_biome_3d_world` + `encode_state`) with flowing rivers, active fires, rising
    steam/smoke plumes, and time-dilation domes explicitly crossing BOTH `x = W/2` and `y = H/2` seams!
    """
    dev_cpu = torch.device("cpu")
    st = make_biome_3d_world(0, L=L, H=H, W=W, device=dev_cpu, seed=seed)
    hh, ww = H // 2, W // 2
    band = max(4, min(10, H // 24))
    span = max(12, min(44, W // 6))

    # East-West river crossing the x = W/2 seam
    st.mass[4:6, 0, hh - band : hh + band, ww - span : ww + span] = 950.0
    st.flow_e[4:6, 0, hh - band : hh + band, ww - span : ww + span] = 180.0
    # North-South river crossing the y = H/2 seam
    st.mass[4:6, 0, hh - span : hh + span, ww - band : ww + band] = 920.0
    st.flow_s[4:6, 0, hh - span : hh + span, ww - band : ww + band] = 175.0
    # Hot wildfire + oil + steam/smoke plume straddling the 4-way seam corner (hh, ww)
    r_f = max(4, min(14, H // 18))
    st.mass[4:7, 8, hh - r_f : hh + r_f, ww - r_f : ww + r_f] += 420.0
    st.energy[4:7, 0, hh - r_f : hh + r_f, ww - r_f : ww + r_f] *= 1.35
    st.mass[6:min(L, 12), 2, hh - r_f : hh + r_f, ww - r_f : ww + r_f] += 260.0
    st.mass[6:min(L, 13), 7, hh - r_f : hh + r_f, ww - r_f : ww + r_f] += 220.0

    x = encode_state(st)

    # Spatial local_dt domes straddling both seams
    ys = torch.arange(H, dtype=torch.float32).view(H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, W)
    d_c = torch.sqrt((ys - hh) ** 2 + (xs - ww) ** 2)
    dt_map = torch.ones((H, W), dtype=torch.float32)
    dt_map = torch.where(d_c < max(8.0, H / 8.0), torch.full_like(dt_map, 0.25), dt_map)
    dt_map = torch.where(
        ((ys - hh).abs() < max(4.0, H / 16.0)) & ((xs - (ww + W / 8.0)).abs() < max(8.0, W / 8.0)),
        torch.full_like(dt_map, 1.75),
        dt_map,
    )
    return x.to(device=device), dt_map.to(device=device)


# =============================================================================
# 3. Test 1: Bit-Exact Monolith vs. `2x2` Sharded Parity (`HALO_XY = 6`)
# =============================================================================


def run_monolith_vs_sharded_parity(
    model: WorldModel3DV8,
    L: int = 16,
    H: int = 256,
    W: int = 256,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Verify:
      1. `encode_law_features(DEFAULT_WORLD_LAWS)` is identically `0.0` for BOTH global `[6]`
         and uniform spatial `[6, H, W]`.
      2. Global `[6]` vs uniform spatial `[6, H, W]` forward pass parity (`<= 1e-6`).
      3. Monolithic `256x256x16` forward vs. four independent `128x128x16` shards exchanging
         ONLY `HALO_XY = 6` ghost halos (state + `local_dt` + 4-zone spatial `world_laws[6, 256, 256]`)
         achieves IEEE-754 FP32 machine-ULP parity (`rel_err <= 1e-6`, `z_flux`/`Z_world_3d` `abs <= 1e-6`)
         AND exact algebraic FP64 identity (`<= 1e-14` across all outputs)!
      4. `SparseWorld3DV8Runner` FP32 dense-vs-sparse parity with spatial `world_laws[6, 128, 128]`.
    """
    # Run exact IEEE-754 FP32/FP64 parity on CPU to avoid GPU TF32/Winograd kernel-selection jitter
    dev = torch.device("cpu")
    wm = WorldModel3DV8().to(device=dev, dtype=torch.float32).eval()
    wm.load_state_dict({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, strict=True)

    # 1. Exact zero-feature canonical invariant (global + spatial uniform)
    f_glob = encode_law_features(DEFAULT_WORLD_LAWS, device=dev)
    default_spatial = (
        torch.tensor(DEFAULT_WORLD_LAWS, dtype=torch.float32, device=dev)
        .view(LAW_DIM, 1, 1)
        .expand(LAW_DIM, H, W)
        .contiguous()
    )
    f_spat = encode_law_features(default_spatial, device=dev)
    zero_glob = float(f_glob.abs().max().item())
    zero_spat = float(f_spat.abs().max().item())
    assert zero_glob == 0.0, f"Global default law features != 0: {zero_glob}"
    assert zero_spat == 0.0, f"Spatial default law features != 0: {zero_spat}"

    # 2. Global [6] vs uniform spatial [6, 64, 64] parity on non-canonical preset
    x_sm, dt_sm = make_cross_seam_world_state(L=L, H=64, W=64, device=dev, seed=101)
    magma_vec = WORLD_LAW_PRESETS["low_g_magma_crucible"]
    magma_spatial = (
        torch.tensor(magma_vec, dtype=torch.float32, device=dev)
        .view(LAW_DIM, 1, 1)
        .expand(LAW_DIM, 64, 64)
        .contiguous()
    )
    with torch.inference_mode():
        hg, zfg, zg = wm(x_sm, local_dt=dt_sm, world_laws=magma_vec, return_z_flux=True, return_latent=True)
        hs, zfs, zs = wm(x_sm, local_dt=dt_sm, world_laws=magma_spatial, return_z_flux=True, return_latent=True)
    global_vs_uniform_spatial_abs = max(
        float((hg - hs).abs().max().item()),
        float((zfg - zfs).abs().max().item()),
        float((zg - zs).abs().max().item()),
    )
    global_vs_uniform_spatial_rel = max(
        float((hg - hs).abs().max().item()) / max(float(hg.abs().max().item()), 1.0),
        float((zfg - zfs).abs().max().item()) / max(float(zfg.abs().max().item()), 1.0),
        float((zg - zs).abs().max().item()) / max(float(zg.abs().max().item()), 1.0),
    )

    # 3. Monolith 256x256x16 vs 2x2 independent 128x128x16 shards with HALO_XY=6 ghost halos (FP32)
    x_full, dt_full = make_cross_seam_world_state(L=L, H=H, W=W, device=dev, seed=2402)
    laws_4zone = make_4zone_spatial_law_field(H, W, blend_width=32, device=dev, dtype=torch.float32)

    with torch.inference_mode():
        m_horiz, m_z_flux, m_z_world = wm(
            x_full, local_dt=dt_full, world_laws=laws_4zone, return_z_flux=True, return_latent=True
        )

        # Split into 4 independent shards (each owns ONLY its H/2 x W/2 interior)
        x_shards = split_2x2_shards(x_full)
        dt_shards = split_2x2_shards(dt_full)
        law_shards = split_2x2_shards(laws_4zone)

        # Exchange ONLY HALO_XY = 6 ghost halo strips (+ 6x6 corners)
        x_padded = exchange_2x2_ghost_halos(x_shards, halo=HALO_XY)
        dt_padded = exchange_2x2_ghost_halos(dt_shards, halo=HALO_XY)
        law_padded = exchange_2x2_ghost_halos(law_shards, halo=HALO_XY)

        sh_h: dict[tuple[int, int], torch.Tensor] = {}
        sh_zf: dict[tuple[int, int], torch.Tensor] = {}
        sh_z: dict[tuple[int, int], torch.Tensor] = {}
        for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
            oh, ozf, oz = wm(
                x_padded[key],
                local_dt=dt_padded[key],
                world_laws=law_padded[key],
                return_z_flux=True,
                return_latent=True,
            )
            sh_h[key] = oh
            sh_zf[key] = ozf
            sh_z[key] = oz

        s_horiz = stitch_2x2_shards(sh_h, halo=HALO_XY)
        s_z_flux = stitch_2x2_shards(sh_zf, halo=HALO_XY)
        s_z_world = stitch_2x2_shards(sh_z, halo=HALO_XY)

    diff_horiz = float((m_horiz - s_horiz).abs().max().item())
    diff_z_flux = float((m_z_flux - s_z_flux).abs().max().item())
    diff_z_world = float((m_z_world - s_z_world).abs().max().item())
    mag_horiz = max(float(m_horiz.abs().max().item()), 1e-6)
    mag_z_flux = max(float(m_z_flux.abs().max().item()), 1e-6)
    mag_z_world = max(float(m_z_world.abs().max().item()), 1e-6)
    rel_horiz = diff_horiz / mag_horiz
    rel_z_flux = diff_z_flux / mag_z_flux
    rel_z_world = diff_z_world / mag_z_world
    # Separate non-heat channels (0..7 mass fluxes, 10..16 reactions; channels 8..9 have out_scale=10.0)
    non_heat_idx = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16]
    diff_horiz_mass_rxn = float((m_horiz[:, non_heat_idx] - s_horiz[:, non_heat_idx]).abs().max().item())

    # Specifically inspect seam strip cells (within 2 voxels of x=W/2 or y=H/2)
    hh, ww = H // 2, W // 2
    seam_mask = torch.zeros((H, W), dtype=torch.bool, device=dev)
    seam_mask[hh - 2 : hh + 2, :] = True
    seam_mask[:, ww - 2 : ww + 2] = True
    diff_seam_horiz = float((m_horiz - s_horiz).abs()[..., seam_mask].max().item())
    diff_seam_z_flux = float((m_z_flux - s_z_flux).abs()[..., seam_mask].max().item())
    diff_seam_z_world = float((m_z_world - s_z_world).abs()[..., seam_mask].max().item())

    # 3b. Exact algebraic identity verification in IEEE-754 FP64 (64x64x16) proving ZERO halo error
    wm64 = WorldModel3DV8().to(device=dev, dtype=torch.float64).eval()
    wm64.load_state_dict({k: v.detach().cpu().double() for k, v in model.state_dict().items()}, strict=True)
    x64, dt64 = make_cross_seam_world_state(L=L, H=64, W=64, device=dev, seed=2402)
    x64, dt64 = x64.double(), dt64.double()
    laws64 = make_4zone_spatial_law_field(64, 64, blend_width=16, device=dev, dtype=torch.float64)
    with torch.inference_mode():
        mh64, mzf64, mz64 = wm64(x64, local_dt=dt64, world_laws=laws64, return_z_flux=True, return_latent=True)
        xp64 = exchange_2x2_ghost_halos(split_2x2_shards(x64), halo=HALO_XY)
        dtp64 = exchange_2x2_ghost_halos(split_2x2_shards(dt64), halo=HALO_XY)
        lp64 = exchange_2x2_ghost_halos(split_2x2_shards(laws64), halo=HALO_XY)
        sh_h64, sh_zf64, sh_z64 = {}, {}, {}
        for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
            sh_h64[key], sh_zf64[key], sh_z64[key] = wm64(
                xp64[key], local_dt=dtp64[key], world_laws=lp64[key], return_z_flux=True, return_latent=True
            )
        fp64_diff_h = float((mh64 - stitch_2x2_shards(sh_h64, halo=HALO_XY)).abs().max().item())
        fp64_diff_zf = float((mzf64 - stitch_2x2_shards(sh_zf64, halo=HALO_XY)).abs().max().item())
        fp64_diff_z = float((mz64 - stitch_2x2_shards(sh_z64, halo=HALO_XY)).abs().max().item())

    # 4. SparseWorld3DV8Runner FP32 parity with spatial 4-zone world_laws[6, 128, 128]
    x_sp0, dt_sp = make_cross_seam_world_state(L=L, H=128, W=128, device=dev, seed=303)
    x_sp1 = x_sp0.clone()
    x_sp1[4:7, 0, 52:76, 52:76] += 0.25  # localized river surge across center seam
    x_sp1[6:10, 2, 56:72, 56:72] += 0.20  # steam burst
    laws_sp = make_4zone_spatial_law_field(128, 128, blend_width=32, device=dev, dtype=torch.float32)

    runner_sp = SparseWorld3DV8Runner(wm, device=dev, use_fp16=False, brick_xy=16, brick_z=8, sparse_threshold=0.95)
    with torch.inference_mode():
        _, _, _, f0 = runner_sp.step(x_sp0, local_dt=dt_sp, world_laws=laws_sp)
        sp_h, sp_zf, sp_z, f1 = runner_sp.step(x_sp1, local_dt=dt_sp, world_laws=laws_sp, dilate_halo_bricks=True)
        dn_h, dn_zf, dn_z = wm(x_sp1, local_dt=dt_sp, world_laws=laws_sp, return_z_flux=True, return_latent=True)

    assert f0 == 1.0 and 0.0 < f1 < 0.95
    sparse_diff_horiz = float((sp_h - dn_h).abs().max().item())
    sparse_diff_z_flux = float((sp_zf - dn_zf).abs().max().item())
    sparse_diff_z_world = float((sp_z - dn_z).abs().max().item())
    sparse_rel_all = max(
        sparse_diff_horiz / max(float(dn_h.abs().max().item()), 1.0),
        sparse_diff_z_flux / max(float(dn_zf.abs().max().item()), 1.0),
        sparse_diff_z_world / max(float(dn_z.abs().max().item()), 1.0),
    )
    sparse_diff_all = max(sparse_diff_horiz, sparse_diff_z_flux, sparse_diff_z_world)

    assert max(rel_horiz, rel_z_flux, rel_z_world) <= 1e-6, (
        f"Monolith vs 2x2 Sharded FP32 rel diff exceeded 1e-6: h={rel_horiz}, zf={rel_z_flux}, z={rel_z_world}"
    )
    assert max(diff_z_flux, diff_z_world) <= 1e-6, (
        f"Monolith vs 2x2 Sharded FP32 abs diff exceeded 1e-6: zf={diff_z_flux}, z={diff_z_world}"
    )
    assert max(fp64_diff_h, fp64_diff_zf, fp64_diff_z) <= 1e-13, (
        f"Exact FP64 algebraic parity failed: h={fp64_diff_h}, zf={fp64_diff_zf}, z={fp64_diff_z}"
    )
    assert sparse_rel_all <= 1e-6, f"Sparse spatial world_laws FP32 rel parity exceeded 1e-6: {sparse_rel_all}"

    return {
        "grid_shape": [L, H, W],
        "shard_shape": [L, H // 2, W // 2],
        "halo_xy": HALO_XY,
        "blend_width_m": 32,
        "default_law_features_global_max_abs": zero_glob,
        "default_law_features_spatial_max_abs": zero_spat,
        "global_vs_uniform_spatial_max_abs_diff": global_vs_uniform_spatial_abs,
        "global_vs_uniform_spatial_max_rel_diff": global_vs_uniform_spatial_rel,
        "monolith_vs_2x2_sharded_fp32": {
            "max_abs_diff_horiz": diff_horiz,
            "max_abs_diff_horiz_mass_rxn": diff_horiz_mass_rxn,
            "max_abs_diff_z_flux": diff_z_flux,
            "max_abs_diff_z_world_3d": diff_z_world,
            "max_rel_diff_horiz": rel_horiz,
            "max_rel_diff_z_flux": rel_z_flux,
            "max_rel_diff_z_world_3d": rel_z_world,
            "max_rel_diff_all": max(rel_horiz, rel_z_flux, rel_z_world),
            "tensor_max_mag_horiz": mag_horiz,
            "tensor_max_mag_z_flux": mag_z_flux,
            "tensor_max_mag_z_world_3d": mag_z_world,
            "seam_strip_max_abs_diff_horiz": diff_seam_horiz,
            "seam_strip_max_abs_diff_z_flux": diff_seam_z_flux,
            "seam_strip_max_abs_diff_z_world_3d": diff_seam_z_world,
            "passed_1e6_rel_gate": max(rel_horiz, rel_z_flux, rel_z_world) <= 1e-6,
        },
        "monolith_vs_2x2_sharded_fp64_exact": {
            "grid_shape": [L, 64, 64],
            "max_abs_diff_horiz": fp64_diff_h,
            "max_abs_diff_z_flux": fp64_diff_zf,
            "max_abs_diff_z_world_3d": fp64_diff_z,
            "max_abs_diff_all": max(fp64_diff_h, fp64_diff_zf, fp64_diff_z),
            "passed_1e13_algebraic_gate": max(fp64_diff_h, fp64_diff_zf, fp64_diff_z) <= 1e-13,
        },
        "sparse_runner_spatial_laws_fp32": {
            "grid_shape": [L, 128, 128],
            "active_brick_frac": round(f1, 4),
            "max_abs_diff_horiz": sparse_diff_horiz,
            "max_abs_diff_z_flux": sparse_diff_z_flux,
            "max_abs_diff_z_world_3d": sparse_diff_z_world,
            "max_abs_diff_all": sparse_diff_all,
            "max_rel_diff_all": sparse_rel_all,
            "passed_1e6_rel_gate": sparse_rel_all <= 1e-6,
        },
    }


# =============================================================================
# 4. Test 2: `4.19M`-Voxel (`512x512x16`) Spark GB10 Live GPU Scaling Benchmark
# =============================================================================


def measure_halo_bandwidth_4regions(
    L: int = 16,
    region_h: int = 256,
    region_w: int = 256,
    halo: int = HALO_XY,
    active_seam_frac: float = 0.12,
) -> dict[str, Any]:
    """
    Measure exact per-tick inter-shard ghost-halo transfer sizes across all 4 internal
    seams of a `2x2` grid of `256x256x16` regions (`HALO_XY = 6`):
      - 4 bidirectional orthogonal seams (2 E/W + 2 N/S -> 8 directed strips of `L x 256 x 6`)
      - 4 diagonal corner patches (`L x 6 x 6` each)
    Reports raw FP16 bytes, zlib-compressed full state, and zlib-compressed delta (at `12%` activity)
    for both full 18-channel encoded state AND minimal 11-channel conserved state (10 species + temp).
    """
    x_full, _ = make_cross_seam_world_state(L=L, H=2 * region_h, W=2 * region_w, device="cpu", seed=909)
    shards = split_2x2_shards(x_full.half())
    h = int(halo)

    # Collect all directed halo payloads exchanged in 1 tick across the 2x2 seams
    strips_18 = [
        shards[(0, 0)][:, :, :, -h:],  # (0,0) -> (0,1)
        shards[(0, 1)][:, :, :, :h],   # (0,1) -> (0,0)
        shards[(1, 0)][:, :, :, -h:],  # (1,0) -> (1,1)
        shards[(1, 1)][:, :, :, :h],   # (1,1) -> (1,0)
        shards[(0, 0)][:, :, -h:, :],  # (0,0) -> (1,0)
        shards[(1, 0)][:, :, :h, :],   # (1,0) -> (0,0)
        shards[(0, 1)][:, :, -h:, :],  # (0,1) -> (1,1)
        shards[(1, 1)][:, :, :h, :],   # (1,1) -> (0,1)
        # 4 diagonal corners
        shards[(0, 0)][:, :, -h:, -h:],
        shards[(0, 1)][:, :, -h:, :h],
        shards[(1, 0)][:, :, :h, -h:],
        shards[(1, 1)][:, :, :h, :h],
    ]

    raw_18_bytes = b"".join(s.contiguous().numpy().tobytes() for s in strips_18)
    raw_11_bytes = b"".join(s[:, :11].contiguous().numpy().tobytes() for s in strips_18)

    # Temporal delta at realistic 12% MMO seam activity (XOR/delta against previous tick)
    gen = torch.Generator(device="cpu").manual_seed(910)
    delta_18_chunks = []
    delta_11_chunks = []
    for s in strips_18:
        mask = (torch.rand((1, 1, s.shape[2], s.shape[3]), generator=gen) < active_seam_frac).half()
        d = (s * 0.05 * mask).half()
        delta_18_chunks.append(d.contiguous().numpy().tobytes())
        delta_11_chunks.append(d[:, :11].contiguous().numpy().tobytes())
    delta_18_raw = b"".join(delta_18_chunks)
    delta_11_raw = b"".join(delta_11_chunks)

    z_full_18 = zlib.compress(raw_18_bytes, level=1)
    z_delta_18 = zlib.compress(delta_18_raw, level=1)
    z_full_11 = zlib.compress(raw_11_bytes, level=1)
    z_delta_11 = zlib.compress(delta_11_raw, level=1)

    out: dict[str, Any] = {
        "halo_xy": h,
        "total_halo_voxels_exchanged_per_tick": int(sum(s.shape[0] * s.shape[2] * s.shape[3] for s in strips_18)),
        "raw_fp16_18ch_kb_per_tick": round(len(raw_18_bytes) / 1024.0, 2),
        "zlib_full_18ch_kb_per_tick": round(len(z_full_18) / 1024.0, 2),
        "zlib_delta_18ch_kb_per_tick": round(len(z_delta_18) / 1024.0, 2),
        "raw_fp16_11ch_kb_per_tick": round(len(raw_11_bytes) / 1024.0, 2),
        "zlib_full_11ch_kb_per_tick": round(len(z_full_11) / 1024.0, 2),
        "zlib_delta_11ch_kb_per_tick": round(len(z_delta_11) / 1024.0, 2),
        "zlib_delta_18ch_mb_per_sec_at_10hz": round((len(z_delta_18) * 10.0) / (1024.0 * 1024.0), 3),
        "zlib_delta_11ch_mb_per_sec_at_10hz": round((len(z_delta_11) * 10.0) / (1024.0 * 1024.0), 3),
    }

    try:
        import lz4.frame as lz4f  # type: ignore

        lz4_delta_18 = lz4f.compress(delta_18_raw)
        lz4_delta_11 = lz4f.compress(delta_11_raw)
        out["lz4_delta_18ch_kb_per_tick"] = round(len(lz4_delta_18) / 1024.0, 2)
        out["lz4_delta_11ch_kb_per_tick"] = round(len(lz4_delta_11) / 1024.0, 2)
    except Exception:
        pass

    return out


def make_mmo_4region_active_brick_mask(
    L: int = 16,
    H: int = 512,
    W: int = 512,
    brick_z: int = 8,
    brick_xy: int = 16,
    region_fracs: tuple[float, float, float, float] = (0.25, 0.10, 0.10, 0.03),
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Build realistic 4-region MMO active brick mask `[nz, ny, nx]` for `512x512x16`:
      - Region 0 (NW, hot combat/town):  25% active
      - Region 1 (NE, medium frontier):  10% active
      - Region 2 (SW, medium flood):     10% active
      - Region 3 (SE, quiet wilderness):  3% active
      -> ~12% aggregate active across 4,194,304 voxels (2,048 bricks total)!
    """
    nz, ny, nx = L // brick_z, H // brick_xy, W // brick_xy
    ry, rx = ny // 2, nx // 2
    per_region_total = nz * ry * rx

    mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device="cpu")
    region_meta = {}
    quad_slices = [
        ("region_0_nw_hot", (0, ry, 0, rx), region_fracs[0]),
        ("region_1_ne_med", (0, ry, rx, nx), region_fracs[1]),
        ("region_2_sw_med", (ry, ny, 0, rx), region_fracs[2]),
        ("region_3_se_wild", (ry, ny, rx, nx), region_fracs[3]),
    ]

    for idx_q, (name, (y0, y1, x0, x1), frac) in enumerate(quad_slices):
        k_target = max(1, int(round(per_region_total * frac)))
        # Cluster active bricks around a realistic hot focal point + seam crossing
        cy = 0.5 * (y0 + y1) + (2.0 if idx_q in (0, 2) else -2.0)
        cx = 0.5 * (x0 + x1) + (3.0 if idx_q in (0, 1) else -3.0)
        coords = []
        for bz in range(nz):
            for by in range(y0, y1):
                for bx in range(x0, x1):
                    d = (by - cy) ** 2 + (bx - cx) ** 2 + 0.5 * bz
                    coords.append((d, bz, by, bx))
        coords.sort(key=lambda t: t[0])
        for _, bz, by, bx in coords[:k_target]:
            mask[bz, by, bx] = True
        region_meta[name] = {
            "active_bricks": k_target,
            "total_bricks": per_region_total,
            "active_pct": round(100.0 * k_target / per_region_total, 2),
        }

    total_active = int(mask.sum().item())
    total_bricks = int(mask.numel())
    meta = {
        "total_bricks": total_bricks,
        "total_active_bricks": total_active,
        "aggregate_active_pct": round(100.0 * total_active / total_bricks, 2),
        "per_region": region_meta,
    }
    return mask.to(device=device), meta


def _sync_device(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def benchmark_4_19m_voxels_gpu(
    model: WorldModel3DV8,
    device: torch.device,
    L: int = 16,
    H: int = 512,
    W: int = 512,
    warmup: int = 5,
    iters: int = 25,
) -> dict[str, Any]:
    """
    Benchmark `4,194,304`-voxel (`512x512x16`, four `256x256x16` regions) live FP16 GPU throughput:
      1. 4-Region Dense (`4 x 262x262x16` with `HALO_XY = 6` ghost halo exchange per tick)
      2. 4-Region Unified Sparse Brick Pool (`12%` aggregate MMO active bricks across all 4 regions,
         including `HALO_XY = 6` ghost halo exchange + spatial `world_laws[6, 512, 512]`)
    Stays cleanly under `<= 3 GB` VRAM!
    """
    use_fp16 = device.type == "cuda"
    compute_dtype = torch.float16 if use_fp16 else torch.float32
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Build 512x512x16 world state + 4-zone spatial world_laws[6, 512, 512]
    x_full_f32, dt_full_f32 = make_cross_seam_world_state(L=L, H=H, W=W, device=device, seed=4194)
    x_full = x_full_f32.to(dtype=compute_dtype).contiguous()
    dt_full = dt_full_f32.to(dtype=compute_dtype).contiguous()
    del x_full_f32, dt_full_f32
    laws_full = make_4zone_spatial_law_field(H, W, blend_width=32, device=device, dtype=torch.float32)

    # Split into 4 independent 256x256x16 regions for realistic halo exchange timing
    x_shards = split_2x2_shards(x_full)
    dt_shards = split_2x2_shards(dt_full)
    law_shards = split_2x2_shards(laws_full)
    law_padded_static = exchange_2x2_ghost_halos(law_shards, halo=HALO_XY)
    dt_padded_static = exchange_2x2_ghost_halos(dt_shards, halo=HALO_XY)

    # ------------------------------------------------------------------
    # Benchmark A: 4-Region Dense (4 x 262x262x16 with ghost halo exchange)
    # ------------------------------------------------------------------
    wm_dense = WorldModel3DV8().to(device=device, dtype=compute_dtype).eval()
    wm_dense.load_state_dict(model.state_dict(), strict=True)
    for p in wm_dense.parameters():
        p.requires_grad_(False)
    wm_dense_run: Any = wm_dense
    if device.type == "cuda" and use_fp16 and hasattr(torch, "compile"):
        try:
            wm_dense_run = torch.compile(wm_dense, mode="reduce-overhead")
        except Exception:
            wm_dense_run = wm_dense

    dense_times_ms: list[float] = []
    with torch.inference_mode():
        for step_i in range(warmup + iters):
            _sync_device(device)
            t0 = time.perf_counter()
            x_pad_4 = exchange_2x2_ghost_halos(x_shards, halo=HALO_XY)
            for key in ((0, 0), (0, 1), (1, 0), (1, 1)):
                if device.type == "cuda" and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                _ = wm_dense_run(
                    x_pad_4[key],
                    local_dt=dt_padded_static[key],
                    world_laws=law_padded_static[key],
                    return_z_flux=True,
                    return_latent=True,
                )
            _sync_device(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if step_i >= warmup:
                dense_times_ms.append(dt_ms)

    dense_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )

    # Free dense model buffers before sparse benchmark so peak VRAM stays cleanly < 3 GB
    del wm_dense_run, wm_dense, law_padded_static, dt_padded_static, x_shards, dt_shards, law_shards
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # ------------------------------------------------------------------
    # Benchmark B: 4-Region Unified Sparse Brick Pool (12% sync + 8.9% async multi-rate)
    # ------------------------------------------------------------------
    active_mask_sync, mask_meta_sync = make_mmo_4region_active_brick_mask(
        L=L, H=H, W=W, brick_z=8, brick_xy=16, region_fracs=(0.25, 0.10, 0.10, 0.03), device=device
    )
    # Async multi-rate schedule (Region 0 @ 10 Hz, Regions 1-2 @ 5 Hz, Region 3 @ 2 Hz):
    # effective active bricks per 10 Hz tick = (25%, 5%, 5%, 0.6%) -> ~8.9% aggregate!
    active_mask_async, mask_meta_async = make_mmo_4region_active_brick_mask(
        L=L, H=H, W=W, brick_z=8, brick_xy=16, region_fracs=(0.25, 0.05, 0.05, 0.006), device=device
    )

    # Use max_bricks_per_batch=64 so activation memory fits cleanly in L2 cache & < 1.2 GB VRAM!
    runner = SparseWorld3DV8Runner(
        model,
        device=device,
        use_fp16=use_fp16,
        brick_xy=16,
        brick_z=8,
        sparse_threshold=0.45,
        bucket_size=64,
        max_bricks_per_batch=64,
    )

    # Pre-allocate steady-state regional caches (avoiding a monolithic 512x512x16 dense VRAM spike!)
    runner._last_horiz = torch.zeros((L, model.out_ch, H, W), dtype=compute_dtype, device=device)
    runner._last_z_flux_pad = torch.zeros((L, model.z_flux_ch, H, W), dtype=compute_dtype, device=device)
    runner._last_z = torch.zeros((L, model.width, H, W), dtype=compute_dtype, device=device)
    runner._last_x = x_full.clone()
    runner._last_dt = dt_full.view(1, 1, H, W).expand(L, 1, H, W).contiguous()
    runner._last_spatial_laws = laws_full.clone()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def _bench_sparse(mask_t: torch.Tensor) -> tuple[list[float], list[float], float]:
        times_ms: list[float] = []
        halo_ms: list[float] = []
        hh, ww = H // 2, W // 2
        h_xy = HALO_XY
        with torch.inference_mode():
            for step_i in range(warmup + iters):
                _sync_device(device)
                t0 = time.perf_counter()
                # Explicit 4-region seam halo sync (6-voxel ghost strips across x=256 and y=256)
                _ = (
                    x_full[:, :, :, ww - h_xy : ww + h_xy].clone(),
                    x_full[:, :, hh - h_xy : hh + h_xy, :].clone(),
                )
                _sync_device(device)
                t_halo = time.perf_counter()
                _, _, _, _ = runner.step(
                    x_full,
                    local_dt=dt_full,
                    world_laws=laws_full,
                    dirty_mask=mask_t,
                    dilate_halo_bricks=False,
                )
                _sync_device(device)
                t1 = time.perf_counter()
                if step_i >= warmup:
                    halo_ms.append((t_halo - t0) * 1000.0)
                    times_ms.append((t1 - t0) * 1000.0)
        peak_mb = (
            round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
        )
        return times_ms, halo_ms, peak_mb

    sparse_times_ms, halo_sync_times_ms, sparse_peak_vram_mb = _bench_sparse(active_mask_sync)
    async_times_ms, _, async_peak_vram_mb = _bench_sparse(active_mask_async)

    def _stats(vals: list[float]) -> dict[str, float]:
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

    dense_s = _stats(dense_times_ms)
    sparse_s = _stats(sparse_times_ms)
    async_s = _stats(async_times_ms)
    halo_s = _stats(halo_sync_times_ms)
    speedup = round(dense_s["p50_ms"] / max(sparse_s["p50_ms"], 1e-6), 2)
    async_speedup = round(dense_s["p50_ms"] / max(async_s["p50_ms"], 1e-6), 2)

    halo_bw = measure_halo_bandwidth_4regions(
        L=L, region_h=H // 2, region_w=W // 2, halo=HALO_XY, active_seam_frac=0.12
    )

    return {
        "device": gpu_name,
        "dtype": "fp16" if use_fp16 else "fp32",
        "world_voxels": int(L * H * W),
        "world_shape": [L, H, W],
        "num_regions": 4,
        "region_shape": [L, H // 2, W // 2],
        "halo_xy": HALO_XY,
        "model_params": int(count_parameters(model)),
        "dense_4region": {
            **dense_s,
            "peak_vram_mb": dense_peak_vram_mb,
            "within_10hz_budget": dense_s["p50_ms"] <= 100.0,
        },
        "unified_sparse_brick_pool": {
            **sparse_s,
            "halo_sync_p50_ms": halo_s["p50_ms"],
            "peak_vram_mb": sparse_peak_vram_mb,
            "speedup_vs_dense": speedup,
            "within_10hz_budget": sparse_s["p50_ms"] <= 100.0,
            "within_3gb_vram_budget": max(dense_peak_vram_mb, sparse_peak_vram_mb) <= 3072.0,
            "mmo_load": mask_meta_sync,
        },
        "unified_sparse_multirate_async": {
            **async_s,
            "peak_vram_mb": async_peak_vram_mb,
            "speedup_vs_dense": async_speedup,
            "within_10hz_budget": async_s["p50_ms"] <= 100.0,
            "within_3gb_vram_budget": max(dense_peak_vram_mb, async_peak_vram_mb) <= 3072.0,
            "mmo_load": mask_meta_async,
        },
        "halo_bandwidth": halo_bw,
    }


# =============================================================================
# 5. Crisp 3-Panel Dark-Theme Visualization (`models/multi_region_shard_viz.png`)
# =============================================================================


def _render_via_system_pil(metrics: dict[str, Any], out_path: Path) -> None:
    """
    High-resolution 3-panel dark-theme figure (2160x760 px) rendered via system `/usr/bin/python3`
    PIL + DejaVuSans TrueType fonts when `matplotlib` is absent from the CUDA venv.
    """
    import subprocess

    laws_256 = make_4zone_spatial_law_field(256, 256, blend_width=16, device="cpu", dtype=torch.float32)
    dT_grid = (laws_256[LAW_AMBIENT_TEMP_SHIFT] / 100.0).round(decimals=2).tolist()
    g_grid = laws_256[LAW_GRAVITY].round(decimals=3).tolist()

    payload = json.dumps(
        {
            "metrics": metrics,
            "dT_grid": dT_grid,
            "g_grid": g_grid,
            "out_path": str(out_path),
        }
    )

    pil_script = r'''
import json, math, sys
from PIL import Image, ImageDraw, ImageFont

data = json.loads(sys.stdin.read())
metrics = data["metrics"]
dT_grid = data["dT_grid"]
g_grid = data["g_grid"]
out_path = data["out_path"]

W, H = 2160, 760
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
SLATE = (100, 116, 139)
SLATE_DK = (71, 85, 105)

def font(size, bold=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()

def coolwarm(val, vmin=-60.0, vmax=60.0):
    t = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
    # Smooth diverging blue -> slate-white -> crimson colormap
    if t < 0.5:
        u = t * 2.0
        r = int(42 + u * (222 - 42))
        g = int(112 + u * (228 - 112))
        b = int(218 + u * (238 - 218))
    else:
        u = (t - 0.5) * 2.0
        r = int(222 + u * (216 - 222))
        g = int(228 + u * (58 - 228))
        b = int(238 + u * (64 - 238))
    return (r, g, b)

def draw_dashed_line(draw, x0, y0, x1, y1, color, width=2, dash=10, gap=6):
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
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((cx - tw / 2, cy - th / 2), txt, font=fnt, fill=fill)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

# Main header
text_center(
    draw,
    W // 2,
    34,
    "Frostmarsh Wave 24B — Seamless 2x2 Ghost-Halo Sharding, Spatial World-Law Field & 4.19M-Voxel GPU Scaling",
    font(24, bold=True),
    FG,
)

# 3 panel boxes
panels = [
    (28, 72, 704, 732),
    (732, 72, 1418, 732),
    (1446, 72, 2132, 732),
]
for px0, py0, px1, py1 in panels:
    draw.rounded_rectangle([px0, py0, px1, py1], radius=12, fill=PANEL, outline=BORDER, width=2)

# =========================================================================
# Panel 1: 2x2 Spatial World-Law Field (ambient_T + gravity_g contours + seams)
# =========================================================================
p1x0, p1y0, p1x1, p1y1 = panels[0]
text_center(draw, (p1x0 + p1x1) // 2, p1y0 + 26, "1. 2x2 Spatial World-Law Field (32m Cosine Blend)", font(18, bold=True), FG)

map_sz = 512
mx0, my0 = p1x0 + 58, p1y0 + 62
# Build 256x256 heatmap and scale smoothly to 512x512
hm = Image.new("RGB", (256, 256))
hpx = hm.load()
for yy in range(256):
    row_t = dT_grid[yy]
    for xx in range(256):
        hpx[xx, yy] = coolwarm(row_t[xx])
# Overlay gravity_g contours directly on 256x256 grid
contour_levels = [0.45, 0.70, 0.95, 1.05, 1.12]
for yy in range(1, 255):
    for xx in range(1, 255):
        v = g_grid[yy][xx]
        vn = g_grid[yy - 1][xx]
        vw = g_grid[yy][xx - 1]
        for lv in contour_levels:
            if (v - lv) * (vn - lv) <= 0.0 or (v - lv) * (vw - lv) <= 0.0:
                hpx[xx, yy] = (248, 250, 252)
                break
hm_big = hm.resize((map_sz, map_sz), Image.BILINEAR)
img.paste(hm_big, (mx0, my0))
draw.rectangle([mx0, my0, mx0 + map_sz, my0 + map_sz], outline=(100, 116, 139), width=2)

# Shard seams & 32m cosine transition boundaries
draw_dashed_line(draw, mx0 + 256, my0, mx0 + 256, my0 + map_sz, (255, 255, 255), width=3, dash=12, gap=7)
draw_dashed_line(draw, mx0, my0 + 256, mx0 + map_sz, my0 + 256, (255, 255, 255), width=3, dash=12, gap=7)
for off in (240, 272):
    draw_dashed_line(draw, mx0 + off, my0, mx0 + off, my0 + map_sz, SKY, width=1, dash=4, gap=4)
    draw_dashed_line(draw, mx0, my0 + off, mx0 + map_sz, my0 + off, SKY, width=1, dash=4, gap=4)

# Quadrant callout cards
f_q_b = font(12, bold=True)
f_q_s = font(11, bold=False)
quads = [
    (mx0 + 128, my0 + 92, "Shard (0,0) NW — FREEZE-OUT", "-55 K | 1.00g | 1.4x precip", "25% hot active (10 Hz)", (12, 26, 46)),
    (mx0 + 384, my0 + 92, "Shard (0,1) NE — STANDARD", "0 K | 1.00g | canonical", "10% med active (5 Hz)", (17, 24, 39)),
    (mx0 + 128, my0 + 420, "Shard (1,0) SW — RISING FLOOD", "-6 K | 1.15g | 2.6x precip", "10% med active (5 Hz)", (12, 32, 46)),
    (mx0 + 384, my0 + 420, "Shard (1,1) SE — LOW-G MAGMA", "+55 K | 0.35g | 2.2x vol", "3% wild active (2 Hz)", (44, 18, 22)),
]
for qx, qy, l1, l2, l3, bgc in quads:
    draw.rounded_rectangle([qx - 118, qy - 36, qx + 118, qy + 36], radius=8, fill=bgc, outline=(148, 163, 184), width=1)
    text_center(draw, qx, qy - 19, l1, f_q_b, (248, 250, 252))
    text_center(draw, qx, qy + 1, l2, f_q_s, (203, 213, 225))
    text_center(draw, qx, qy + 20, l3, f_q_s, CYAN)

# Gravity contour legend chip at center-bottom of map
draw.rounded_rectangle([mx0 + 150, my0 + 236, mx0 + 362, my0 + 276], radius=8, fill=(11, 15, 23), outline=SKY, width=1)
text_center(draw, mx0 + 256, my0 + 256, "32m Blend | White = gravity_g", font(12, bold=True), FG)

# Colorbar on right of Panel 1
cb_x0, cb_y0, cb_w, cb_h = mx0 + map_sz + 18, my0, 20, map_sz
for yy in range(cb_h):
    v = 60.0 - 120.0 * (yy / float(cb_h - 1))
    draw.line([(cb_x0, cb_y0 + yy), (cb_x0 + cb_w, cb_y0 + yy)], fill=coolwarm(v))
draw.rectangle([cb_x0, cb_y0, cb_x0 + cb_w, cb_y0 + cb_h], outline=MUTED, width=1)
f_ax = font(12)
for val, lbl in [(55, "+55K"), (0, "0K"), (-55, "-55K")]:
    yy = cb_y0 + int((60.0 - val) / 120.0 * (cb_h - 1))
    draw.text((cb_x0 + cb_w + 5, yy - 7), lbl, font=f_ax, fill=MUTED)

# Axes labels for Panel 1
for tick in (0, 128, 256, 384, 512):
    tx = mx0 + tick
    ty = my0 + tick
    draw.line([(tx, my0 + map_sz), (tx, my0 + map_sz + 5)], fill=MUTED, width=1)
    text_center(draw, tx, my0 + map_sz + 18, str(tick), f_ax, MUTED)
    draw.line([(mx0 - 5, ty), (mx0, ty)], fill=MUTED, width=1)
    draw.text((mx0 - 36, ty - 7), f"{tick:>3}", font=f_ax, fill=MUTED)
text_center(draw, mx0 + 256, p1y1 - 24, "World X (m) — Seam @ x=256 (HALO_XY=6m)", font(14, bold=True), MUTED)

# =========================================================================
# Panel 2: Monolith vs 2x2 Sharded FP32 Seam Parity (Log Scale)
# =========================================================================
p2x0, p2y0, p2x1, p2y1 = panels[1]
text_center(draw, (p2x0 + p2x1) // 2, p2y0 + 26, "2. Monolith vs. 2x2 Sharded Parity (HALO_XY=6)", font(18, bold=True), FG)

gx0, gy0, gx1, gy1 = p2x0 + 82, p2y0 + 70, p2x1 - 28, p2y1 - 122
draw.rectangle([gx0, gy0, gx1, gy1], fill=(13, 18, 29), outline=BORDER, width=1)

p_fp32 = metrics["parity"]["monolith_vs_2x2_sharded_fp32"]
p_fp64 = metrics["parity"]["monolith_vs_2x2_sharded_fp64_exact"]
p_sp = metrics["parity"]["sparse_runner_spatial_laws_fp32"]

log_min, log_max = -9.0, -4.0
def y_log(val):
    lv = math.log10(max(val, 10 ** log_min))
    lv = max(log_min, min(log_max, lv))
    frac = (lv - log_min) / (log_max - log_min)
    return int(gy1 - frac * (gy1 - gy0))

for exp in range(-9, -3):
    yy = y_log(10 ** exp)
    draw.line([(gx0, yy), (gx1, yy)], fill=GRID, width=1)
    draw.text((gx0 - 54, yy - 8), f"1e{exp}", font=f_ax, fill=MUTED)

# 1e-6 gate line (place label cleanly on top-left clear area above bars 1-2!)
y_gate = y_log(1e-6)
draw_dashed_line(draw, gx0, y_gate, gx1, y_gate, ROSE, width=3, dash=10, gap=6)
draw.rounded_rectangle([gx0 + 12, y_gate - 36, gx0 + 270, y_gate - 10], radius=6, fill=(38, 14, 22), outline=ROSE, width=1)
text_center(draw, gx0 + 141, y_gate - 23, "Bit-Exact Parity Gate (1.0e-6)", font(12, bold=True), (253, 164, 175))

bars2 = [
    ("Horiz\n(FP32 Rel)", p_fp32["max_rel_diff_horiz"], SKY),
    ("Z-Flux\n(FP32 Abs)", p_fp32["max_abs_diff_z_flux"], INDIGO),
    ("Z_world_3d\n(FP32 Abs)", p_fp32["max_abs_diff_z_world_3d"], EMERALD),
    ("Sparse Spatial\n(FP32 Rel)", p_sp["max_rel_diff_all"], AMBER),
]
bw2 = 86
slot_w2 = (gx1 - gx0) / len(bars2)
for i, (lbl, val, col) in enumerate(bars2):
    cx = int(gx0 + slot_w2 * (i + 0.5))
    by = y_log(val)
    draw.rectangle([cx - bw2 // 2, by, cx + bw2 // 2, gy1], fill=col, outline=(226, 232, 240), width=1)
    text_center(draw, cx, by - 16, f"{val:.2e}", font(13, bold=True), FG)
    for li, line in enumerate(lbl.split("\n")):
        text_center(draw, cx, gy1 + 18 + li * 16, line, font(13, bold=(li == 0)), FG if li == 0 else MUTED)

# Summary badge at bottom of Panel 2
draw.rounded_rectangle([p2x0 + 30, p2y1 - 62, p2x1 - 30, p2y1 - 16], radius=8, fill=(10, 30, 26), outline=EMERALD, width=1)
text_center(
    draw,
    (p2x0 + p2x1) // 2,
    p2y1 - 39,
    f"FP64 Exact Algebraic Seam Diff = {p_fp64['max_abs_diff_all']:.2e}  |  Default Laws = 0.0e+00",
    font(13, bold=True),
    EMERALD,
)

# =========================================================================
# Panel 3: 4.19M-Voxel (512x512x16) Spark GB10 Single-GPU Scaling
# =========================================================================
p3x0, p3y0, p3x1, p3y1 = panels[2]
g = metrics["gpu_benchmark"]
text_center(draw, (p3x0 + p3x1) // 2, p3y0 + 26, f"3. 4.19M-Voxel Scaling ({g['device']}, FP16)", font(18, bold=True), FG)

hx0, hy0, hx1, hy1 = p3x0 + 78, p3y0 + 70, p3x1 - 28, p3y1 - 122
draw.rectangle([hx0, hy0, hx1, hy1], fill=(13, 18, 29), outline=BORDER, width=1)

d_b = g["dense_4region"]
s_b = g["unified_sparse_brick_pool"]
a_b = g["unified_sparse_multirate_async"]
bw_info = g["halo_bandwidth"]

ymax = max(d_b["p95_ms"], s_b["p95_ms"], a_b["p95_ms"], 120.0) * 1.18
ymax = math.ceil(ymax / 50.0) * 50.0

def y_lin(v):
    return int(hy1 - (max(0.0, min(ymax, v)) / ymax) * (hy1 - hy0))

step_y = 100 if ymax >= 350 else 50
for yv in range(0, int(ymax) + 1, step_y):
    yy = y_lin(yv)
    draw.line([(hx0, yy), (hx1, yy)], fill=GRID, width=1)
    draw.text((hx0 - 52, yy - 8), f"{yv:>3}ms", font=f_ax, fill=MUTED)

# 10 Hz real-time budget line (100 ms)
y_10hz = y_lin(100.0)
draw_dashed_line(draw, hx0, y_10hz, hx1, y_10hz, ROSE, width=3, dash=10, gap=6)

groups = [
    ("4-Region Dense\n(4x 262x262x16)", d_b["p50_ms"], d_b["p95_ms"], SLATE, SLATE_DK),
    (f"Sparse Sync\n({s_b['mmo_load']['aggregate_active_pct']:.1f}% Bricks)", s_b["p50_ms"], s_b["p95_ms"], SKY, (14, 116, 144)),
    (f"Sparse Async\n(10/5/2Hz, {a_b['mmo_load']['aggregate_active_pct']:.1f}%)", a_b["p50_ms"], a_b["p95_ms"], EMERALD, (16, 135, 96)),
]
slot_w3 = (hx1 - hx0) / len(groups)
bw3 = 54
for i, (lbl, p50, p95, c50, c95) in enumerate(groups):
    cx = int(hx0 + slot_w3 * (i + 0.5))
    y50, y95 = y_lin(p50), y_lin(p95)
    draw.rectangle([cx - bw3 - 3, y50, cx - 3, hy1], fill=c50, outline=(226, 232, 240), width=1)
    draw.rectangle([cx + 3, y95, cx + bw3 + 3, hy1], fill=c95, outline=(203, 213, 225), width=1)
    text_center(draw, cx - bw3 // 2 - 3, y50 - 14, f"{p50:.0f}", font(13, bold=True), FG)
    text_center(draw, cx + bw3 // 2 + 3, y95 - 14, f"{p95:.0f}", font(12, bold=False), MUTED)
    for li, line in enumerate(lbl.split("\n")):
        text_center(draw, cx, hy1 + 18 + li * 16, line, font(13, bold=(li == 0)), FG if li == 0 else MUTED)

# Legend for p50/p95 + 10 Hz budget in clear top-right sky of Panel 3 (zero overlap!)
lx, ly = hx1 - 228, hy0 + 14
draw.rounded_rectangle([lx, ly, hx1 - 10, ly + 72], radius=6, fill=BG, outline=BORDER, width=1)
draw.rectangle([lx + 10, ly + 10, lx + 24, ly + 22], fill=SKY)
draw.text((lx + 32, ly + 8), "p50 ms (left bar)", font=font(12), fill=FG)
draw.rectangle([lx + 10, ly + 30, lx + 24, ly + 42], fill=(14, 116, 144))
draw.text((lx + 32, ly + 28), "p95 ms (right bar)", font=font(12), fill=MUTED)
draw_dashed_line(draw, lx + 10, ly + 56, lx + 26, ly + 56, ROSE, width=3, dash=5, gap=3)
draw.text((lx + 32, ly + 48), "10 Hz Budget (100 ms)", font=font(12, bold=True), fill=(253, 164, 175))

# Summary callout badge at bottom of Panel 3
peak_vram = max(d_b["peak_vram_mb"], s_b["peak_vram_mb"], a_b["peak_vram_mb"])
draw.rounded_rectangle([p3x0 + 26, p3y1 - 62, p3x1 - 26, p3y1 - 16], radius=8, fill=(10, 26, 38), outline=CYAN, width=1)
text_center(
    draw,
    (p3x0 + p3x1) // 2,
    p3y1 - 48,
    f"Async Sparse: {a_b['p50_ms']:.1f} ms ({a_b['hz_p50']:.1f} Hz, {a_b['speedup_vs_dense']:.2f}x)  |  Sync: {s_b['speedup_vs_dense']:.2f}x",
    font(12, bold=True),
    CYAN,
)
text_center(
    draw,
    (p3x0 + p3x1) // 2,
    p3y1 - 28,
    f"Peak VRAM: {peak_vram:.0f} MB (<3 GB)  |  Halo Delta: {bw_info['zlib_delta_18ch_kb_per_tick']:.1f} KB/tick ({bw_info['zlib_delta_18ch_mb_per_sec_at_10hz']:.2f} MB/s)",
    font(12, bold=False),
    FG,
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


def render_multi_region_viz(metrics: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib
    except ImportError:
        _render_via_system_pil(metrics, out_path)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    bg = "#0b0f17"
    panel_bg = "#121926"
    fg = "#e6edf7"
    muted = "#94a3b8"
    grid_col = "#243044"

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8), dpi=160, facecolor=bg)
    plt.subplots_adjust(left=0.045, right=0.975, top=0.84, bottom=0.15, wspace=0.26)

    fig.suptitle(
        "Frostmarsh Wave 24B — Seamless 2x2 Ghost-Halo Sharding, Spatial World-Law Field & 4.19M-Voxel GPU Scaling",
        color=fg,
        fontsize=14,
        fontweight="bold",
        y=0.955,
    )

    # ------------------------------------------------------------------
    # Panel 1: 2x2 4-Zone Spatial Law Map (ambient_T + gravity_g contours + seams)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(panel_bg)
    laws_512 = make_4zone_spatial_law_field(512, 512, blend_width=32, device="cpu", dtype=torch.float32).numpy()
    dT_kelvin = laws_512[LAW_AMBIENT_TEMP_SHIFT] / 100.0  # cK -> K (-55 K .. +55 K)
    grav_g = laws_512[LAW_GRAVITY]

    im = ax1.imshow(dT_kelvin, cmap="coolwarm", vmin=-60.0, vmax=60.0, origin="upper", extent=[0, 512, 512, 0])
    cs = ax1.contour(
        np.linspace(0, 512, 512),
        np.linspace(0, 512, 512),
        grav_g,
        levels=[0.45, 0.70, 0.95, 1.05, 1.12],
        colors="#f8fafc",
        linewidths=1.05,
        alpha=0.82,
    )
    ax1.clabel(cs, fmt="%.2fg", fontsize=7.5, colors="#f8fafc", inline=True)

    # Draw 32m blend band and exact shard seams
    ax1.axvline(256, color="#ffffff", linestyle="--", linewidth=1.6, alpha=0.95)
    ax1.axhline(256, color="#ffffff", linestyle="--", linewidth=1.6, alpha=0.95)
    for b_off in (240, 272):
        ax1.axvline(b_off, color="#38bdf8", linestyle=":", linewidth=0.9, alpha=0.65)
        ax1.axhline(b_off, color="#38bdf8", linestyle=":", linewidth=0.9, alpha=0.65)

    quad_labels = [
        (128, 86, "Shard (0,0) NW\nFREEZE-OUT\n-55 K | 1.4x precip\n25% hot active", "#0b192c"),
        (384, 86, "Shard (0,1) NE\nSTANDARD\n0 K | 1.00g\n10% med active", "#111827"),
        (128, 424, "Shard (1,0) SW\nRISING FLOOD\n-6 K | 2.6x precip | 0.55x visc\n10% med active", "#0c1f2e"),
        (384, 424, "Shard (1,1) SE\nLOW-G MAGMA\n+55 K | 0.35g | 2.2x vol\n3% wild active", "#2a1215"),
    ]
    for qx, qy, txt, box_c in quad_labels:
        ax1.text(
            qx,
            qy,
            txt,
            color="#f8fafc",
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=box_c, edgecolor="#64748b", alpha=0.82, linewidth=0.8),
        )

    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cbar.set_label("Ambient Temp Shift (K)", color=muted, fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color=muted, labelcolor=muted, labelsize=8)
    ax1.set_title("1. 2x2 Spatial World-Law Field (32m Cosine Blend)", color=fg, fontsize=10.5, fontweight="bold", pad=10)
    ax1.set_xlabel("World X (m) — Seam @ x=256 (Halo=6m)", color=muted, fontsize=9)
    ax1.set_ylabel("World Y (m) — Seam @ y=256", color=muted, fontsize=9)
    ax1.tick_params(colors=muted, labelsize=8)
    for sp in ax1.spines.values():
        sp.set_color(grid_col)

    # ------------------------------------------------------------------
    # Panel 2: Monolith vs. 2x2 Sharded Seam Parity (FP32 Rel + FP64 Exact)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(panel_bg)
    p = metrics["parity"]["monolith_vs_2x2_sharded_fp32"]
    p64 = metrics["parity"]["monolith_vs_2x2_sharded_fp64_exact"]
    sp_p = metrics["parity"]["sparse_runner_spatial_laws_fp32"]

    labels = ["Horiz\n(FP32 Rel)", "Z-Flux\n(FP32 Abs)", "Z_world_3d\n(FP32 Abs)", "Sparse Spatial\n(FP32 Rel)"]
    raw_vals = [
        p["max_rel_diff_horiz"],
        p["max_abs_diff_z_flux"],
        p["max_abs_diff_z_world_3d"],
        sp_p["max_rel_diff_all"],
    ]
    plot_vals = [max(v, 1e-9) for v in raw_vals]
    bar_cols = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24"]

    xs_b = np.arange(len(labels))
    bars = ax2.bar(xs_b, plot_vals, width=0.56, color=bar_cols, edgecolor="#e2e8f0", linewidth=0.7, alpha=0.92)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-9, 1e-4)
    ax2.axhline(1e-6, color="#f43f5e", linestyle="--", linewidth=1.6, label="Bit-Exact Parity Gate (1.0e-6)")
    ax2.grid(axis="y", color=grid_col, linestyle="-", linewidth=0.7, alpha=0.75)

    for b, rv, pv in zip(bars, raw_vals, plot_vals):
        lbl = f"{rv:.2e}" if rv > 0 else "0.00e+00"
        ax2.text(
            b.get_x() + b.get_width() * 0.5,
            pv * 1.65,
            lbl,
            color=fg,
            fontsize=8.3,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    ax2.set_xticks(xs_b)
    ax2.set_xticklabels(labels, color=fg, fontsize=8.4)
    ax2.set_ylabel("Max Error vs. Monolith (FP32, log)", color=muted, fontsize=9)
    ax2.set_title("2. Monolith vs. 2x2 Sharded Parity (HALO_XY=6)", color=fg, fontsize=10.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", facecolor=bg, edgecolor=grid_col, labelcolor=fg, fontsize=8)
    ax2.tick_params(colors=muted, labelsize=8)
    ax2.text(
        0.5,
        0.06,
        f"FP64 Exact Algebraic Seam Diff = {p64['max_abs_diff_all']:.2e} | Default Laws = 0.0",
        transform=ax2.transAxes,
        color="#34d399",
        fontsize=8.1,
        fontweight="bold",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor="#34d399", alpha=0.85, linewidth=0.8),
    )
    for sp in ax2.spines.values():
        sp.set_color(grid_col)

    # ------------------------------------------------------------------
    # Panel 3: 4.19M-Voxel (512x512x16) Single-GPU Scaling on Spark GB10
    # ------------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(panel_bg)
    g = metrics["gpu_benchmark"]
    d_p50 = g["dense_4region"]["p50_ms"]
    d_p95 = g["dense_4region"]["p95_ms"]
    s_p50 = g["unified_sparse_brick_pool"]["p50_ms"]
    s_p95 = g["unified_sparse_brick_pool"]["p95_ms"]
    speedup = g["unified_sparse_brick_pool"]["speedup_vs_dense"]
    agg_pct = g["unified_sparse_brick_pool"]["mmo_load"]["aggregate_active_pct"]
    peak_vram = max(g["dense_4region"]["peak_vram_mb"], g["unified_sparse_brick_pool"]["peak_vram_mb"])
    bw_kb = g["halo_bandwidth"]["zlib_delta_18ch_kb_per_tick"]

    x3 = np.arange(2)
    w_bar = 0.30
    b_p50 = ax3.bar(x3 - w_bar / 2, [d_p50, s_p50], width=w_bar, color=["#64748b", "#22d3ee"], label="p50 ms", edgecolor="#e2e8f0", linewidth=0.7)
    b_p95 = ax3.bar(x3 + w_bar / 2, [d_p95, s_p95], width=w_bar, color=["#475569", "#0ea5e9"], label="p95 ms", edgecolor="#e2e8f0", linewidth=0.7, alpha=0.88)

    ax3.axhline(100.0, color="#f43f5e", linestyle="--", linewidth=1.6, label="10 Hz Real-Time Budget (100 ms)")
    ax3.grid(axis="y", color=grid_col, linestyle="-", linewidth=0.7, alpha=0.75)

    max_y = max(d_p95, s_p95, 110.0) * 1.24
    ax3.set_ylim(0, max_y)

    for grp in (b_p50, b_p95):
        for bar in grp:
            val = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() * 0.5,
                val + max_y * 0.02,
                f"{val:.1f} ms",
                color=fg,
                fontsize=8.3,
                fontweight="bold",
                ha="center",
                va="bottom",
            )

    ax3.set_xticks(x3)
    ax3.set_xticklabels(
        [
            "4-Region Dense\n(4x 262x262x16)",
            f"Unified Sparse Pool\n({agg_pct:.1f}% Active Bricks)",
        ],
        color=fg,
        fontsize=8.7,
    )
    ax3.set_ylabel("Step Latency (ms / tick, FP16)", color=muted, fontsize=9)
    ax3.set_title(f"3. 4.19M-Voxel Scaling ({g['device']})", color=fg, fontsize=10.5, fontweight="bold", pad=10)
    ax3.legend(loc="upper right", facecolor=bg, edgecolor=grid_col, labelcolor=fg, fontsize=8)
    ax3.tick_params(colors=muted, labelsize=8)

    callout = (
        f"Sparse: {s_p50:.1f} ms ({g['unified_sparse_brick_pool']['hz_p50']:.1f} Hz)  |  {speedup:.2f}x Speedup\n"
        f"Peak VRAM: {peak_vram:.0f} MB (<3 GB)  |  Halo Delta: {bw_kb:.1f} KB/tick"
    )
    ax3.text(
        0.5,
        0.74,
        callout,
        transform=ax3.transAxes,
        color="#22d3ee",
        fontsize=8.0,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.38", facecolor=bg, edgecolor="#22d3ee", alpha=0.90, linewidth=0.9),
    )
    for sp in ax3.spines.values():
        sp.set_color(grid_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, facecolor=bg)
    plt.close(fig)


# =============================================================================
# 6. CLI Entrypoint
# =============================================================================


def load_v8_model(ckpt_path: Path, device: torch.device) -> WorldModel3DV8:
    model = WorldModel3DV8().to(device=device, dtype=torch.float32).eval()
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(sd, strict=True)
        print(f"Loaded trained WorldModel3DV8 checkpoint: {ckpt_path} ({count_parameters(model):,} params)")
    else:
        print(f"WARNING: checkpoint {ckpt_path} not found; activating synthetic law weights for standalone test")
        with torch.no_grad():
            model.law_to_dt.weight.normal_(0.0, 0.1)
            model.law_h_mod.weight.normal_(0.0, 0.1)
            model.law_h_logscale.weight.normal_(0.0, 0.1)
            model.law_z_logscale.weight.normal_(0.0, 0.1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 24B: 2x2 Multi-Region Ghost-Halo Sharding & 4.19M-Voxel GPU Bench")
    parser.add_argument("--ckpt", type=str, default="models/world_model_3d_v8.pt", help="Path to world_model_3d_v8.pt")
    parser.add_argument("--out-json", type=str, default="models/MULTI_REGION_SHARD_METRICS.json")
    parser.add_argument("--out-png", type=str, default="models/multi_region_shard_viz.png")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--plot-only", action="store_true", help="Re-render chart directly from existing --out-json")
    args = parser.parse_args()

    if args.plot_only:
        metrics = json.loads(Path(args.out_json).read_text())
        render_multi_region_viz(metrics, Path(args.out_png))
        print(f"Re-rendered chart -> {args.out_png}")
        return

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    ckpt_path = Path(args.ckpt)
    model = load_v8_model(ckpt_path, device=dev)

    print("\n=== Test 1: Monolith vs. 2x2 Sharded FP32/FP64 Seam Parity (256x256x16, HALO_XY=6) ===")
    parity = run_monolith_vs_sharded_parity(model, L=16, H=256, W=256, device=dev)
    p2x2 = parity["monolith_vs_2x2_sharded_fp32"]
    p64 = parity["monolith_vs_2x2_sharded_fp64_exact"]
    psp = parity["sparse_runner_spatial_laws_fp32"]
    print(
        f"  default_law_features max_abs: global={parity['default_law_features_global_max_abs']:.1e}, "
        f"spatial={parity['default_law_features_spatial_max_abs']:.1e}"
    )
    print(
        f"  global vs uniform spatial: abs={parity['global_vs_uniform_spatial_max_abs_diff']:.2e}, "
        f"rel={parity['global_vs_uniform_spatial_max_rel_diff']:.2e}"
    )
    print(
        f"  Monolith vs 2x2 Sharded FP32: horiz_abs={p2x2['max_abs_diff_horiz']:.2e} (rel={p2x2['max_rel_diff_horiz']:.2e}, "
        f"mass/rxn={p2x2['max_abs_diff_horiz_mass_rxn']:.2e}), z_flux={p2x2['max_abs_diff_z_flux']:.2e}, "
        f"Z_world_3d={p2x2['max_abs_diff_z_world_3d']:.2e} (PASS={p2x2['passed_1e6_rel_gate']})"
    )
    print(
        f"  Monolith vs 2x2 Sharded FP64 (Exact Algebraic): horiz={p64['max_abs_diff_horiz']:.2e}, "
        f"z_flux={p64['max_abs_diff_z_flux']:.2e}, Z_world_3d={p64['max_abs_diff_z_world_3d']:.2e} "
        f"(PASS={p64['passed_1e13_algebraic_gate']})"
    )
    print(
        f"  Sparse Spatial Runner FP32 ({psp['active_brick_frac']*100:.1f}% active): "
        f"max_abs={psp['max_abs_diff_all']:.2e}, max_rel={psp['max_rel_diff_all']:.2e} "
        f"(PASS={psp['passed_1e6_rel_gate']})"
    )

    print(f"\n=== Test 2: 4,194,304-Voxel (512x512x16) Live GPU Scaling on {dev} ===")
    gpu_bench = benchmark_4_19m_voxels_gpu(
        model, device=dev, L=16, H=512, W=512, warmup=args.warmup, iters=args.iters
    )
    d_b = gpu_bench["dense_4region"]
    s_b = gpu_bench["unified_sparse_brick_pool"]
    a_b = gpu_bench["unified_sparse_multirate_async"]
    bw = gpu_bench["halo_bandwidth"]
    print(
        f"  4-Region Dense (4x 262x262x16): p50={d_b['p50_ms']:.2f} ms, p95={d_b['p95_ms']:.2f} ms, "
        f"{d_b['hz_p50']:.2f} Hz, peak_vram={d_b['peak_vram_mb']:.1f} MB"
    )
    print(
        f"  Unified Sparse Pool Sync ({s_b['mmo_load']['aggregate_active_pct']:.2f}% active, "
        f"{s_b['mmo_load']['total_active_bricks']}/{s_b['mmo_load']['total_bricks']} bricks): "
        f"p50={s_b['p50_ms']:.2f} ms, p95={s_b['p95_ms']:.2f} ms, {s_b['hz_p50']:.2f} Hz, "
        f"speedup={s_b['speedup_vs_dense']:.2f}x, peak_vram={s_b['peak_vram_mb']:.1f} MB"
    )
    print(
        f"  Unified Sparse Pool Async Multi-Rate 10/5/2Hz ({a_b['mmo_load']['aggregate_active_pct']:.2f}% active, "
        f"{a_b['mmo_load']['total_active_bricks']}/{a_b['mmo_load']['total_bricks']} bricks): "
        f"p50={a_b['p50_ms']:.2f} ms, p95={a_b['p95_ms']:.2f} ms, {a_b['hz_p50']:.2f} Hz, "
        f"speedup={a_b['speedup_vs_dense']:.2f}x, peak_vram={a_b['peak_vram_mb']:.1f} MB"
    )
    print(
        f"  Halo Transfer (HALO_XY=6, 4 seams): raw_18ch={bw['raw_fp16_18ch_kb_per_tick']:.1f} KB/tick, "
        f"zlib_delta_18ch={bw['zlib_delta_18ch_kb_per_tick']:.1f} KB/tick "
        f"({bw['zlib_delta_18ch_mb_per_sec_at_10hz']:.3f} MB/s @ 10Hz)"
    )

    metrics = {
        "wave": "24B",
        "title": "MMO Phase 2B: Seamless 2x2 Multi-Region Ghost-Halo Sharding & Spatial World-Law Field",
        "checkpoint": str(ckpt_path),
        "world_model_params": int(count_parameters(model)),
        "parity": parity,
        "gpu_benchmark": gpu_bench,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved metrics -> {out_json}")

    out_png = Path(args.out_png)
    render_multi_region_viz(metrics, out_png)
    print(f"Saved chart   -> {out_png}")


if __name__ == "__main__":
    main()
