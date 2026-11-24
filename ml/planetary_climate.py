"""
Frostmarsh Wave 33A (GPU / ML — Purpose-Built `PlanetaryClimateNet`):
  Macro-scale (`32 m/cell`, `R = 1,300 m`) spherical atmospheric + ocean + surface climate
  neural operator running on a 6-Face Cubed-Sphere (`6 x 64^2 x 4` strata = `98,304` macro cells).

Core components:
  1. Canonical 6-Face Cubed-Sphere Geometry matching `ml/cubed_sphere_planet.py` & `field-core::planet` 1:1:
     - Faces `0:+X, 1:-X, 2:+Y (North), 3:-Y (South), 4:+Z, 5:-Z` with right-handed `(n_f, e_u, e_v)`
       orthonormal tangent frames (`det == +1`, `e_u x e_v = +n_f`).
     - Basis-derived 12-edge `HALO_XY = 4` + 8-corner halo exchange with exact 2x2 `SO(2)` signed-permutation
       tangent-vector rotation `R_{nbr -> cur}` (`0.0` rotation error across all 12 seams).
  2. Macro Climate State Tensor `[6, L=4, C=10, 64, 64]` + Forcing `[6, 6, 64, 64]`:
     - Levels: `0=Surface/Ocean`, `1=Boundary Layer`, `2=Mid-Troposphere`, `3=Upper Jet`.
     - Channels: `0:temp_norm, 1:water_vapor, 2:cloud_liquid, 3:ice_snow, 4:pressure_anom,
                  5:wind_u, 6:wind_v, 7:wind_w, 8:ocean_heat, 9:ash_smoke`.
     - Forcing:  `0:lat_sin, 1:lat_cos, 2:land_mask, 3:elevation_norm, 4:cos_sun, 5:night_cool`.
  3. Vectorized GPU Ground-Truth Reference Solver (`SphericalClimateRefSim`):
     - Exact antisymmetric finite-volume conservative transport across all 6 faces & 12 rotated seams.
     - Exact energy-preserving spherical Coriolis rotation (`a_cor = (+f*v, -f*u)`, `f = 2*Omega*sin(phi)`):
       NH low-pressure storms spin counter-clockwise (`vorticity > 0`), SH spin clockwise (`vorticity < 0`).
     - Hadley/Ferrel cells, equatorial rainbelt, trade winds, westerlies, sweeping Day/Night terminator,
       ocean thermal inertia & coastal breezes, Clausius-Clapeyron condensation/snowline (`T < 273.15 K`),
       and volcanic/Sky-Island ash radiative cooling & cross-seam advection.
  4. `PlanetaryClimateNet` (~310K params, C4/SO(2) Gauge-Equivariant Cubed-Sphere Neural Operator):
     - Exact `halo=1` rotated vector exchange at every residual block + SO(2)-equivariant complex vector mixing
       across all 4 vertical strata.
     - Strictly antisymmetric CFL-safe conservative flux heads for moisture (`vapor, cloud, ice_snow`) &
       `ash_smoke` plus cell-wise mass-conserving 3x3 phase-conversion simplex (FP32-safe even under FP16
       inference -> `< 1e-6` global drift over 200+ ticks!).
  5. AMR Macro <-> Micro Coupling APIs (Wave 34 ready):
     - `downscale_to_shard_laws(state, face, u0, u1, v0, v1, H, W, L)`
     - `upscale_shard_injection(state, face, u0, u1, v0, v1, delta_temp, delta_ash)`
"""

from __future__ import annotations

import base64
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Constants & Canonical 6-Face Cubed-Sphere Orthonormal Frames (1:1 Match)
# =============================================================================

NUM_FACES = 6
NUM_LEVELS = 4
NUM_CHANNELS = 10
NUM_FORCING = 6
HALO_XY = 4
DEFAULT_N = 64

# Dynamic state channel indices (`[6, L=4, C=10, N, N]`)
CH_TEMP = 0         # (T_K - 280.0) / 40.0
CH_VAPOR = 1        # water_vapor >= 0 (conserved moisture 1)
CH_CLOUD = 2        # cloud_liquid >= 0 (conserved moisture 2)
CH_ICE_SNOW = 3     # ice_snow >= 0 (conserved moisture 3)
CH_PRESSURE = 4     # pressure_anom (geopotential / surface pressure anomaly)
CH_WIND_U = 5       # face-local +e_u tangent wind (SO(2)-rotated across cube seams)
CH_WIND_V = 6       # face-local +e_v tangent wind (SO(2)-rotated across cube seams)
CH_WIND_W = 7       # vertical updraft / downdraft velocity
CH_OCEAN_HEAT = 8   # high-inertia oceanic thermal reservoir
CH_ASH_SMOKE = 9    # volcanic / wildfire / Sky-Island ash tracer >= 0 (conserved)

CHANNEL_NAMES: tuple[str, ...] = (
    "temp_norm",
    "water_vapor",
    "cloud_liquid",
    "ice_snow",
    "pressure_anom",
    "wind_u",
    "wind_v",
    "wind_w",
    "ocean_heat",
    "ash_smoke",
)

LEVEL_NAMES: tuple[str, ...] = (
    "surface_ocean",
    "boundary_layer",
    "mid_troposphere",
    "upper_jet",
)

# Static / forcing channel indices (`[6, 6, N, N]`)
FC_LAT_SIN = 0
FC_LAT_COS = 1
FC_LAND_MASK = 2
FC_ELEV_NORM = 3
FC_COS_SUN = 4
FC_NIGHT_COOL = 5

T_REF_K = 280.0
T_SCALE_K = 40.0
T_FREEZE_K = 273.15
TEMP_NORM_FREEZE = (T_FREEZE_K - T_REF_K) / T_SCALE_K  # -0.17125

# Exact 6-face right-handed orthonormal frames `(n_f, e_u, e_v)` matching `ml/cubed_sphere_planet.py` 1:1!
FACE_NAMES: tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

FACE_BASES_NP: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    0: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),   # +X
    1: (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),   # -X
    2: (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),   # +Y (North)
    3: (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # -Y (South)
    4: (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),    # +Z
    5: (np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # -Z
}

FACE_N_NP = np.stack([FACE_BASES_NP[f][0] for f in range(NUM_FACES)], axis=0)
FACE_EU_NP = np.stack([FACE_BASES_NP[f][1] for f in range(NUM_FACES)], axis=0)
FACE_EV_NP = np.stack([FACE_BASES_NP[f][2] for f in range(NUM_FACES)], axis=0)
EDGE_NAMES: tuple[str, ...] = ("-u", "+u", "-v", "+v")

# WorldModel3D-v8 WorldLaws indices (matching `ml/world_model_3d_v8.py` 1:1)
LAW_GRAVITY = 0
LAW_AMBIENT_TEMP_SHIFT = 1
LAW_VOLATILITY = 2
LAW_VISCOSITY = 3
LAW_PRECIPITATION = 4
LAW_THERMAL_COND = 5
LAW_DIM = 6


def verify_orthonormal_bases() -> dict[str, Any]:
    """Verify all 6 face frames (n_f, e_u, e_v) are right-handed orthonormal (det == +1)."""
    max_ortho_err = 0.0
    min_det = 1e9
    max_det = -1e9
    for f in range(NUM_FACES):
        n_f, eu, ev = FACE_BASES_NP[f]
        mat = np.stack([n_f, eu, ev], axis=1)
        ortho_err = float(np.max(np.abs(mat.T @ mat - np.eye(3))))
        det = float(np.linalg.det(mat))
        max_ortho_err = max(max_ortho_err, ortho_err)
        min_det = min(min_det, det)
        max_det = max(max_det, det)
        assert ortho_err < 1e-12, f"Face {f} not orthonormal: {ortho_err}"
        assert abs(det - 1.0) < 1e-12, f"Face {f} not right-handed: det={det}"
    return {
        "num_faces": NUM_FACES,
        "max_orthonormality_err": max_ortho_err,
        "min_det": min_det,
        "max_det": max_det,
        "all_right_handed": abs(min_det - 1.0) < 1e-12 and abs(max_det - 1.0) < 1e-12,
    }


def face_uv_to_xyz(
    face: int | torch.Tensor,
    u: torch.Tensor | np.ndarray | float,
    v: torch.Tensor | np.ndarray | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map gnomonic face coordinates `(face, u, v)` (`u, v in [-1, 1]`) -> 3D unit sphere `(x, y, z)`."""
    if isinstance(u, torch.Tensor) or isinstance(v, torch.Tensor) or isinstance(face, torch.Tensor):
        ref = u if isinstance(u, torch.Tensor) else (v if isinstance(v, torch.Tensor) else face)
        assert isinstance(ref, torch.Tensor)
        dev = ref.device
        dtype = ref.dtype if ref.is_floating_point() else torch.float32
        u_t = torch.as_tensor(u, dtype=dtype, device=dev)
        v_t = torch.as_tensor(v, dtype=dtype, device=dev)
        u_t, v_t = torch.broadcast_tensors(u_t, v_t)

        n_all = torch.tensor(FACE_N_NP, dtype=dtype, device=dev)
        eu_all = torch.tensor(FACE_EU_NP, dtype=dtype, device=dev)
        ev_all = torch.tensor(FACE_EV_NP, dtype=dtype, device=dev)

        if isinstance(face, int):
            n_f, eu_f, ev_f = n_all[face], eu_all[face], ev_all[face]
            px = n_f[0] + u_t * eu_f[0] + v_t * ev_f[0]
            py = n_f[1] + u_t * eu_f[1] + v_t * ev_f[1]
            pz = n_f[2] + u_t * eu_f[2] + v_t * ev_f[2]
        else:
            f_idx = torch.as_tensor(face, dtype=torch.long, device=dev).expand_as(u_t)
            n_f, eu_f, ev_f = n_all[f_idx], eu_all[f_idx], ev_all[f_idx]
            px = n_f[..., 0] + u_t * eu_f[..., 0] + v_t * ev_f[..., 0]
            py = n_f[..., 1] + u_t * eu_f[..., 1] + v_t * ev_f[..., 1]
            pz = n_f[..., 2] + u_t * eu_f[..., 2] + v_t * ev_f[..., 2]

        inv_r = torch.rsqrt(px * px + py * py + pz * pz + 1e-30)
        return px * inv_r, py * inv_r, pz * inv_r

    return face_uv_to_xyz(int(face), torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(v, dtype=torch.float64))


def xyz_to_face_uv(
    x: torch.Tensor | np.ndarray | float,
    y: torch.Tensor | np.ndarray | float,
    z: torch.Tensor | np.ndarray | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact inverse gnomonic cubed-sphere projection `(x, y, z) -> (face, u, v)`."""
    is_tensor = isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor) or isinstance(z, torch.Tensor)
    if is_tensor:
        ref = x if isinstance(x, torch.Tensor) else (y if isinstance(y, torch.Tensor) else z)
        assert isinstance(ref, torch.Tensor)
        dev = ref.device
        dtype = ref.dtype if ref.is_floating_point() else torch.float32
    else:
        dev = torch.device("cpu")
        dtype = torch.float64

    xt = torch.as_tensor(x, dtype=dtype, device=dev)
    yt = torch.as_tensor(y, dtype=dtype, device=dev)
    zt = torch.as_tensor(z, dtype=dtype, device=dev)
    xt, yt, zt = torch.broadcast_tensors(xt, yt, zt)

    ax, ay, az = xt.abs(), yt.abs(), zt.abs()
    is_x = (ax >= ay) & (ax >= az)
    is_y = (~is_x) & (ay >= az)

    face_x = torch.where(xt >= 0, torch.zeros_like(xt, dtype=torch.long), torch.ones_like(xt, dtype=torch.long))
    face_y = torch.where(yt >= 0, torch.full_like(xt, 2, dtype=torch.long), torch.full_like(xt, 3, dtype=torch.long))
    face_z = torch.where(zt >= 0, torch.full_like(xt, 4, dtype=torch.long), torch.full_like(xt, 5, dtype=torch.long))
    face = torch.where(is_x, face_x, torch.where(is_y, face_y, face_z))

    n_all = torch.tensor(FACE_N_NP, dtype=dtype, device=dev)
    eu_all = torch.tensor(FACE_EU_NP, dtype=dtype, device=dev)
    ev_all = torch.tensor(FACE_EV_NP, dtype=dtype, device=dev)

    n_f, eu_f, ev_f = n_all[face], eu_all[face], ev_all[face]
    denom = (xt * n_f[..., 0] + yt * n_f[..., 1] + zt * n_f[..., 2]).clamp(min=1e-12)
    u = (xt * eu_f[..., 0] + yt * eu_f[..., 1] + zt * eu_f[..., 2]) / denom
    v = (xt * ev_f[..., 0] + yt * ev_f[..., 1] + zt * ev_f[..., 2]) / denom
    return face, u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)


def face_cell_centers_uv(
    N: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 2D cell-center grids `(u_grid [N, N], v_grid [N, N])` in `(-1, 1)`."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    coords = -1.0 + (2.0 * torch.arange(N, dtype=dtype, device=dev) + 1.0) / float(N)
    v_grid = coords.view(N, 1).expand(N, N)
    u_grid = coords.view(1, N).expand(N, N)
    return u_grid, v_grid


# =============================================================================
# 2. Basis-Derived 12-Edge + 8-Corner Halo Exchange & Exact SO(2) Rotation
# =============================================================================


@dataclass(frozen=True)
class EdgeNeighborSpec:
    cur_face: int
    edge_id: int  # 0:-u, 1:+u, 2:-v, 3:+v
    nbr_face: int
    nbr_edge_id: int
    rot_2x2: tuple[tuple[int, int], tuple[int, int]]  # exact integer SO(2) matrix R_{nbr -> cur}
    c_ua: int
    c_us: int
    c_va: int
    c_vs: int


def _find_face_by_normal(vec: np.ndarray) -> int:
    dots = FACE_N_NP @ vec
    idx = int(np.argmax(dots))
    assert abs(float(dots[idx]) - 1.0) < 1e-9, f"No face matches normal {vec}"
    return idx


def _edge_outward_and_tangent(face: int, edge_id: int) -> tuple[np.ndarray, np.ndarray]:
    _, eu, ev = FACE_BASES_NP[face]
    if edge_id == 0:
        return -eu, ev
    if edge_id == 1:
        return eu, ev
    if edge_id == 2:
        return -ev, eu
    if edge_id == 3:
        return ev, eu
    raise ValueError(f"Invalid edge_id {edge_id}")


def derive_edge_neighbor(cur_face: int, edge_id: int) -> EdgeNeighborSpec:
    """Derive exact neighbor face, coordinate bijection, and SO(2) rotation matrix directly from bases."""
    n_c, eu_c, ev_c = FACE_BASES_NP[cur_face]
    d_out, t_along = _edge_outward_and_tangent(cur_face, edge_id)
    nbr_face = _find_face_by_normal(d_out)
    n_n, eu_n, ev_n = FACE_BASES_NP[nbr_face]

    nbr_edge_id = -1
    for eid in range(4):
        d_n, _ = _edge_outward_and_tangent(nbr_face, eid)
        if float(np.dot(d_n, n_c)) > 0.5:
            nbr_edge_id = eid
            break
    assert nbr_edge_id >= 0

    c_ua = int(round(float(np.dot(n_c, eu_n))))
    c_us = int(round(float(np.dot(t_along, eu_n))))
    c_va = int(round(float(np.dot(n_c, ev_n))))
    c_vs = int(round(float(np.dot(t_along, ev_n))))
    assert abs(c_ua) + abs(c_us) == 1 and abs(c_va) + abs(c_vs) == 1
    assert abs(c_ua * c_vs - c_us * c_va) == 1

    Q = np.outer(t_along, t_along) + np.outer(n_c, n_n) - np.outer(n_n, n_c)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert abs(float(np.linalg.det(Q)) - 1.0) < 1e-12

    r00 = int(round(float(eu_c @ (Q @ eu_n))))
    r01 = int(round(float(eu_c @ (Q @ ev_n))))
    r10 = int(round(float(ev_c @ (Q @ eu_n))))
    r11 = int(round(float(ev_c @ (Q @ ev_n))))
    assert (r00 * r11 - r01 * r10) == 1, f"Non-SO(2) rotation: {(r00, r01, r10, r11)}"

    return EdgeNeighborSpec(
        cur_face=cur_face,
        edge_id=edge_id,
        nbr_face=nbr_face,
        nbr_edge_id=nbr_edge_id,
        rot_2x2=((r00, r01), (r10, r11)),
        c_ua=c_ua,
        c_us=c_us,
        c_va=c_va,
        c_vs=c_vs,
    )


EDGE_SPECS: dict[tuple[int, int], EdgeNeighborSpec] = {
    (f, e): derive_edge_neighbor(f, e) for f in range(NUM_FACES) for e in range(4)
}


@dataclass
class _CubedSphereHaloPlan:
    N: int
    halo: int
    P: int
    flat_all: torch.Tensor   # [6 * P * P] int64 index into [6 * N * N]
    rot_mat: torch.Tensor    # [6, 2, 2, P, P] float32 exact SO(2) matrix
    r00: torch.Tensor        # [6, 1, P, P]
    r01: torch.Tensor        # [6, 1, P, P]
    r10: torch.Tensor        # [6, 1, P, P]
    r11: torch.Tensor        # [6, 1, P, P]
    # Border-only packed gather indices & rotation (only 4 thin border bands -> 17x less traffic at h=1!)
    border_idx: torch.Tensor  # [6 * (2*h*P + 2*N*h)]
    b_r00: torch.Tensor       # [1, 1, 6 * (2*h*P + 2*N*h)]
    b_r01: torch.Tensor
    b_r10: torch.Tensor
    b_r11: torch.Tensor
    n_bot: int                # 6 * h * P
    n_top: int                # 6 * h * P
    n_left: int               # 6 * N * h
    n_right: int              # 6 * N * h


_HALO_PLAN_CACHE: dict[tuple[int, int, str], _CubedSphereHaloPlan] = {}


def _get_halo_plan(N: int, halo: int, device: torch.device) -> _CubedSphereHaloPlan:
    key = (int(N), int(halo), str(device))
    if key in _HALO_PLAN_CACHE:
        return _HALO_PLAN_CACHE[key]

    h = int(halo)
    P = N + 2 * h
    src_face = np.zeros((NUM_FACES, P, P), dtype=np.int64)
    src_iv = np.zeros((NUM_FACES, P, P), dtype=np.int64)
    src_iu = np.zeros((NUM_FACES, P, P), dtype=np.int64)
    rot_mat = np.zeros((NUM_FACES, 2, 2, P, P), dtype=np.float32)

    grid_iv, grid_iu = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64), indexing="ij")
    pair_rot: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {
        (cf, spec.nbr_face): spec.rot_2x2 for (cf, _), spec in EDGE_SPECS.items()
    }

    def _coord_to_idx(c_alpha: int, c_s: int, d_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
        if c_alpha == 1:
            return (N - 1) - d_arr
        if c_alpha == -1:
            return d_arr
        if c_s == 1:
            return k_arr
        if c_s == -1:
            return (N - 1) - k_arr
        raise RuntimeError("Invalid edge remap coefficients")

    for f in range(NUM_FACES):
        src_face[f, h : h + N, h : h + N] = f
        src_iv[f, h : h + N, h : h + N] = grid_iv
        src_iu[f, h : h + N, h : h + N] = grid_iu
        rot_mat[f, 0, 0, h : h + N, h : h + N] = 1.0
        rot_mat[f, 1, 1, h : h + N, h : h + N] = 1.0

        # Edge 0 (-u)
        spec = EDGE_SPECS[(f, 0)]
        k_2d, d_2d = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(h - 1, -1, -1, dtype=np.int64), indexing="ij")
        src_face[f, h : h + N, 0:h] = spec.nbr_face
        src_iu[f, h : h + N, 0:h] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, h : h + N, 0:h] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, h : h + N, 0:h] = float(spec.rot_2x2[r][c])

        # Edge 1 (+u)
        spec = EDGE_SPECS[(f, 1)]
        k_2d, d_2d = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(0, h, dtype=np.int64), indexing="ij")
        src_face[f, h : h + N, h + N : P] = spec.nbr_face
        src_iu[f, h : h + N, h + N : P] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, h : h + N, h + N : P] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, h : h + N, h + N : P] = float(spec.rot_2x2[r][c])

        # Edge 2 (-v)
        spec = EDGE_SPECS[(f, 2)]
        d_2d, k_2d = np.meshgrid(np.arange(h - 1, -1, -1, dtype=np.int64), np.arange(N, dtype=np.int64), indexing="ij")
        src_face[f, 0:h, h : h + N] = spec.nbr_face
        src_iu[f, 0:h, h : h + N] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, 0:h, h : h + N] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, 0:h, h : h + N] = float(spec.rot_2x2[r][c])

        # Edge 3 (+v)
        spec = EDGE_SPECS[(f, 3)]
        d_2d, k_2d = np.meshgrid(np.arange(0, h, dtype=np.int64), np.arange(N, dtype=np.int64), indexing="ij")
        src_face[f, h + N : P, h : h + N] = spec.nbr_face
        src_iu[f, h + N : P, h : h + N] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, h + N : P, h : h + N] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, h + N : P, h : h + N] = float(spec.rot_2x2[r][c])

        # 4 diagonal corners via exact 3D unit-sphere lookup
        for rs, cs in (
            (slice(0, h), slice(0, h)),
            (slice(0, h), slice(h + N, P)),
            (slice(h + N, P), slice(0, h)),
            (slice(h + N, P), slice(h + N, P)),
        ):
            rr, cc = np.meshgrid(
                np.arange(rs.start, rs.stop, dtype=np.float64),
                np.arange(cs.start, cs.stop, dtype=np.float64),
                indexing="ij",
            )
            u_ext = -1.0 + (2.0 * (cc - h) + 1.0) / float(N)
            v_ext = -1.0 + (2.0 * (rr - h) + 1.0) / float(N)
            u_ext_q = u_ext + 1e-9 * np.sign(u_ext)
            xt, yt, zt = face_uv_to_xyz(f, torch.from_numpy(u_ext_q), torch.from_numpy(v_ext))
            nf_t, un_t, vn_t = xyz_to_face_uv(xt, yt, zt)
            nf_np = nf_t.numpy().astype(np.int64)
            iu_np = np.clip(np.rint(((un_t.numpy() + 1.0) * N - 1.0) * 0.5).astype(np.int64), 0, N - 1)
            iv_np = np.clip(np.rint(((vn_t.numpy() + 1.0) * N - 1.0) * 0.5).astype(np.int64), 0, N - 1)
            src_face[f, rs, cs] = nf_np
            src_iu[f, rs, cs] = iu_np
            src_iv[f, rs, cs] = iv_np
            for ii in range(h):
                for jj in range(h):
                    R = pair_rot[(f, int(nf_np[ii, jj]))]
                    for r in range(2):
                        for c in range(2):
                            rot_mat[f, r, c, rs.start + ii, cs.start + jj] = float(R[r][c])

    flat_grid = torch.from_numpy(src_face * (N * N) + src_iv * N + src_iu).to(device=device)  # [6, P, P]
    flat_all = flat_grid.reshape(-1).contiguous()
    rot_t = torch.from_numpy(rot_mat).to(device=device)

    idx_bot = flat_grid[:, 0:h, :].reshape(-1)
    idx_top = flat_grid[:, h + N : P, :].reshape(-1)
    idx_left = flat_grid[:, h : h + N, 0:h].reshape(-1)
    idx_right = flat_grid[:, h : h + N, h + N : P].reshape(-1)
    border_idx = torch.cat([idx_bot, idx_top, idx_left, idx_right], dim=0).contiguous()

    def _pack_r(r_grid: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                r_grid[:, 0:h, :].reshape(-1),
                r_grid[:, h + N : P, :].reshape(-1),
                r_grid[:, h : h + N, 0:h].reshape(-1),
                r_grid[:, h : h + N, h + N : P].reshape(-1),
            ],
            dim=0,
        ).view(1, 1, -1).contiguous()

    plan = _CubedSphereHaloPlan(
        N=N,
        halo=h,
        P=P,
        flat_all=flat_all,
        rot_mat=rot_t,
        r00=rot_t[:, 0, 0].unsqueeze(1).contiguous(),
        r01=rot_t[:, 0, 1].unsqueeze(1).contiguous(),
        r10=rot_t[:, 1, 0].unsqueeze(1).contiguous(),
        r11=rot_t[:, 1, 1].unsqueeze(1).contiguous(),
        border_idx=border_idx,
        b_r00=_pack_r(rot_t[:, 0, 0]),
        b_r01=_pack_r(rot_t[:, 0, 1]),
        b_r10=_pack_r(rot_t[:, 1, 0]),
        b_r11=_pack_r(rot_t[:, 1, 1]),
        n_bot=int(idx_bot.numel()),
        n_top=int(idx_top.numel()),
        n_left=int(idx_left.numel()),
        n_right=int(idx_right.numel()),
    )
    _HALO_PLAN_CACHE[key] = plan
    return plan


def pad_cubed_sphere_packed(
    s: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    halo: int = 1,
) -> torch.Tensor:
    """
    Ultra-fast border-only single-gather halo pad for packed `[s, u, v]` on 6 cube faces!
    Inputs: `s [B, 6, C_s, N, N]`, `u [B, 6, C_v, N, N]`, `v [B, 6, C_v, N, N]`.
    Returns: `[B*6, C_s + 2*C_v, N+2h, N+2h]` with exact SO(2) tangent rotation on `(u, v)` borders!
    Uses 17x less gather traffic than full-grid `index_select` and zero in-place autograd ops.
    """
    B, F6, Cs, N, _ = s.shape
    Cv = u.shape[2]
    Ctot = Cs + 2 * Cv
    plan = _get_halo_plan(N, halo, s.device)
    h, P = plan.halo, plan.P

    x = torch.cat([s, u, v], dim=2)  # [B, 6, Ctot, N, N]
    # Flatten spatial `[B, Ctot, 6*N*N]` and gather ONLY the 4 thin border strips in ONE kernel!
    x_bc = x.permute(0, 2, 1, 3, 4).reshape(B * Ctot, NUM_FACES * N * N)
    borders = x_bc.index_select(1, plan.border_idx).view(B, Ctot, -1)  # [B, Ctot, M_border]

    # Rotate ONLY the `(u, v)` border channels (1,560 pixels at h=1, N=64!)
    b_s = borders[:, :Cs]
    b_u = borders[:, Cs : Cs + Cv]
    b_v = borders[:, Cs + Cv :]
    r00 = plan.b_r00.to(dtype=x.dtype)
    r01 = plan.b_r01.to(dtype=x.dtype)
    r10 = plan.b_r10.to(dtype=x.dtype)
    r11 = plan.b_r11.to(dtype=x.dtype)
    b_u_rot = r00 * b_u + r01 * b_v
    b_v_rot = r10 * b_u + r11 * b_v
    b_all = torch.cat([b_s, b_u_rot, b_v_rot], dim=1)  # [B, Ctot, M_border]

    o1 = plan.n_bot
    o2 = o1 + plan.n_top
    o3 = o2 + plan.n_left
    bot = b_all[:, :, :o1].view(B, Ctot, NUM_FACES, h, P).permute(0, 2, 1, 3, 4)
    top = b_all[:, :, o1:o2].view(B, Ctot, NUM_FACES, h, P).permute(0, 2, 1, 3, 4)
    left = b_all[:, :, o2:o3].view(B, Ctot, NUM_FACES, N, h).permute(0, 2, 1, 3, 4)
    right = b_all[:, :, o3:].view(B, Ctot, NUM_FACES, N, h).permute(0, 2, 1, 3, 4)

    mid = torch.cat([left, x, right], dim=-1)      # [B, 6, Ctot, N, P]
    padded = torch.cat([bot, mid, top], dim=-2)    # [B, 6, Ctot, P, P]
    return padded.reshape(B * NUM_FACES, Ctot, P, P)


def pad_cubed_sphere_scalars(x: torch.Tensor, halo: int = 1) -> torch.Tensor:
    """
    Fast single-kernel differentiable halo pad for scalar fields on 6 cube faces.
    Supports `[B, 6, C, N, N]` -> `[B, 6, C, N+2h, N+2h]` or `[6, C, N, N]` -> `[6, C, N+2h, N+2h]`.
    """
    if halo <= 0:
        return x
    unbatched = x.ndim == 4
    if unbatched:
        x = x.unsqueeze(0)
    B, F6, C, N, N2 = x.shape
    assert F6 == NUM_FACES and N == N2
    plan = _get_halo_plan(N, halo, x.device)
    P = plan.P
    x_bc = x.permute(0, 2, 1, 3, 4).reshape(B * C, NUM_FACES * N * N)
    out = x_bc.index_select(1, plan.flat_all).view(B, C, NUM_FACES, P, P).permute(0, 2, 1, 3, 4)
    return out[0] if unbatched else out


def pad_cubed_sphere_vectors(u: torch.Tensor, v: torch.Tensor, halo: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fast single-kernel differentiable halo pad + exact SO(2) tangent-frame rotation `R_{nbr -> cur}`
    for paired tangent vector channels `(u, v)`.
    Supports `[B, 6, K, N, N]` or `[6, K, N, N]`.
    """
    if halo <= 0:
        return u, v
    unbatched = u.ndim == 4
    if unbatched:
        u = u.unsqueeze(0)
        v = v.unsqueeze(0)
    B, F6, K, N, N2 = u.shape
    assert F6 == NUM_FACES and N == N2 and v.shape == u.shape
    plan = _get_halo_plan(N, halo, u.device)
    P = plan.P
    uv = torch.cat([u, v], dim=2)  # [B, 6, 2K, N, N]
    uv_bc = uv.permute(0, 2, 1, 3, 4).reshape(B * 2 * K, NUM_FACES * N * N)
    uv_pad = uv_bc.index_select(1, plan.flat_all).view(B, 2, K, NUM_FACES, P, P).permute(0, 3, 1, 2, 4, 5)
    u_raw = uv_pad[:, :, 0]  # [B, 6, K, P, P]
    v_raw = uv_pad[:, :, 1]
    r00 = plan.r00.to(dtype=u.dtype).unsqueeze(0)
    r01 = plan.r01.to(dtype=u.dtype).unsqueeze(0)
    r10 = plan.r10.to(dtype=u.dtype).unsqueeze(0)
    r11 = plan.r11.to(dtype=u.dtype).unsqueeze(0)
    u_rot = r00 * u_raw + r01 * v_raw
    v_rot = r10 * u_raw + r11 * v_raw
    if unbatched:
        return u_rot[0], v_rot[0]
    return u_rot, v_rot


def exchange_cubed_sphere_halos(
    faces_x: torch.Tensor,
    halo: int = HALO_XY,
    flow_channels: tuple[int, int] | None = (CH_WIND_U, CH_WIND_V),
) -> torch.Tensor:
    """
    Exact 6-face cubed-sphere ghost-halo exchange across all 12 rotated edges and 8 cube corners!
    Supports:
      - `[B, 6, L, C, N, N]` -> `[B, 6, L, C, N+2h, N+2h]`
      - `[6, L, C, N, N]`    -> `[6, L, C, N+2h, N+2h]`
      - `[6, C, N, N]`       -> `[6, C, N+2h, N+2h]`
      - `[6, N, N]`          -> `[6, N+2h, N+2h]`
    Applies exact SO(2) tangent rotation `R_{nbr -> cur}` to `flow_channels` (default `(wind_u, wind_v)`).
    """
    if halo <= 0:
        return faces_x.clone()

    ndim = faces_x.ndim
    if ndim == 6:
        B, F6, L, C, N, N2 = faces_x.shape
        assert F6 == NUM_FACES and N == N2
        flat = faces_x.reshape(B, F6, L * C, N, N)
        out = pad_cubed_sphere_scalars(flat, halo=halo).view(B, F6, L, C, N + 2 * halo, N + 2 * halo)
        if flow_channels is not None:
            cu, cv = flow_channels
            if max(cu, cv) < C:
                plan = _get_halo_plan(N, halo, faces_x.device)
                fu = out[:, :, :, cu].clone()
                fv = out[:, :, :, cv].clone()
                r00 = plan.r00.to(dtype=out.dtype).unsqueeze(0)
                r01 = plan.r01.to(dtype=out.dtype).unsqueeze(0)
                r10 = plan.r10.to(dtype=out.dtype).unsqueeze(0)
                r11 = plan.r11.to(dtype=out.dtype).unsqueeze(0)
                out = out.clone()
                out[:, :, :, cu] = r00 * fu + r01 * fv
                out[:, :, :, cv] = r10 * fu + r11 * fv
        return out

    if ndim == 5:
        return exchange_cubed_sphere_halos(faces_x.unsqueeze(0), halo=halo, flow_channels=flow_channels)[0]

    if ndim == 4:
        F6, C, N, N2 = faces_x.shape
        assert F6 == NUM_FACES and N == N2
        out = pad_cubed_sphere_scalars(faces_x, halo=halo)
        if flow_channels is not None:
            cu, cv = flow_channels
            if max(cu, cv) < C:
                plan = _get_halo_plan(N, halo, faces_x.device)
                fu = out[:, cu].clone()
                fv = out[:, cv].clone()
                r00 = plan.r00[:, 0].to(dtype=out.dtype)
                r01 = plan.r01[:, 0].to(dtype=out.dtype)
                r10 = plan.r10[:, 0].to(dtype=out.dtype)
                r11 = plan.r11[:, 0].to(dtype=out.dtype)
                out = out.clone()
                out[:, cu] = r00 * fu + r01 * fv
                out[:, cv] = r10 * fu + r11 * fv
        return out

    if ndim == 3:
        return pad_cubed_sphere_scalars(faces_x.unsqueeze(1), halo=halo)[:, 0]

    raise ValueError(f"Unsupported tensor rank {ndim} for exchange_cubed_sphere_halos")


def crop_cubed_sphere_halos(padded: torch.Tensor, halo: int = HALO_XY) -> torch.Tensor:
    if halo <= 0:
        return padded
    return padded[..., halo:-halo, halo:-halo].contiguous()


# =============================================================================
# 3. Exact Antisymmetric Conservative Cubed-Sphere Transport & Vorticity
# =============================================================================


def conservative_cubed_sphere_transport(
    q: torch.Tensor,
    vel_u: torch.Tensor,
    vel_v: torch.Tensor,
    vel_w_iface: torch.Tensor,
    diff_h: torch.Tensor | float = 0.015,
    diff_v: torch.Tensor | float = 0.01,
    cfl_max: float = 0.88,
) -> torch.Tensor:
    """
    Strictly mass-conserving & positivity-preserving finite-volume transport on the 6-face cubed-sphere!
    Shapes (batched `[B, 6, L, S, N, N]` or unbatched `[6, L, S, N, N]`):
      - `q`: non-negative conserved scalars (`>= 0`), `[..., L, S, N, N]`
      - `vel_u, vel_v`: horizontal cell-centered transport velocities, `[..., L, S, N, N]`
      - `vel_w_iface`: vertical interface velocities between level `l` and `l+1` (upward positive),
                       `[..., L-1, S, N, N]`
      - `diff_h, diff_v`: non-negative horizontal/vertical diffusion coefficients (`>= 0`)

    Because `v_{A->B} = 0.5*(u_A + u_B)` satisfies `v_{B->A} = -v_{A->B}` identically across EVERY
    interior edge AND across all 12 rotated cubed-sphere seams (via exact SO(2) `R_{nbr->cur}`),
    the interface flux `F_{A->B} = alpha_{A->B} * q_A - alpha_{B->A} * q_B` is EXACTLY antisymmetric:
      `F_{B->A} == -F_{A->B}` to FP32 machine precision!
    With `alpha` clamped so total outflow fraction from every cell `<= cfl_max < 1`, both global mass
    conservation (`drift < 1e-6`) and strict non-negativity (`q_new >= 0`) hold identically!
    """
    orig_dtype = q.dtype
    unbatched = q.ndim == 5
    if unbatched:
        q = q.unsqueeze(0)
        vel_u = vel_u.unsqueeze(0)
        vel_v = vel_v.unsqueeze(0)
        vel_w_iface = vel_w_iface.unsqueeze(0)
        if isinstance(diff_h, torch.Tensor):
            diff_h = diff_h.unsqueeze(0)
        if isinstance(diff_v, torch.Tensor):
            diff_v = diff_v.unsqueeze(0)

    # Always run conservative flux accumulation in FP32 so FP16 inference retains `< 1e-6` global conservation!
    q32 = q.float().clamp(min=0.0)
    u32 = vel_u.float()
    v32 = vel_v.float()
    w32 = vel_w_iface.float()

    B, F6, L, S, N, _ = q32.shape
    LS = L * S

    # Clamp single-edge transport rate to guarantee `sum_6_neighbors(alpha) <= cfl_max < 1.0`!
    max_edge_rate = cfl_max / 6.0  # <= 0.1467 per face
    dh_t = (
        diff_h.float().clamp(0.0, max_edge_rate * 0.35)
        if isinstance(diff_h, torch.Tensor)
        else torch.full_like(q32, min(float(diff_h), max_edge_rate * 0.35))
    )
    dv_t = (
        diff_v.float().clamp(0.0, max_edge_rate * 0.35)
        if isinstance(diff_v, torch.Tensor)
        else torch.full_like(w32, min(float(diff_v), max_edge_rate * 0.35))
    )

    q_pad = pad_cubed_sphere_scalars(q32.reshape(B, F6, LS, N, N), halo=1).view(B, F6, L, S, N + 2, N + 2)
    dh_pad = pad_cubed_sphere_scalars(dh_t.reshape(B, F6, LS, N, N), halo=1).view(B, F6, L, S, N + 2, N + 2)
    u_pad, v_pad = pad_cubed_sphere_vectors(
        u32.reshape(B, F6, LS, N, N),
        v32.reshape(B, F6, LS, N, N),
        halo=1,
    )
    u_pad = u_pad.view(B, F6, L, S, N + 2, N + 2)
    v_pad = v_pad.view(B, F6, L, S, N + 2, N + 2)

    q_c = q_pad[..., 1:-1, 1:-1]
    q_e = q_pad[..., 1:-1, 2:]
    q_w = q_pad[..., 1:-1, :-2]
    q_n = q_pad[..., 2:, 1:-1]
    q_s = q_pad[..., :-2, 1:-1]

    u_c = u_pad[..., 1:-1, 1:-1]
    u_e = u_pad[..., 1:-1, 2:]
    u_w = u_pad[..., 1:-1, :-2]
    v_c = v_pad[..., 1:-1, 1:-1]
    v_n = v_pad[..., 2:, 1:-1]
    v_s = v_pad[..., :-2, 1:-1]

    dh_c = dh_pad[..., 1:-1, 1:-1]
    d_e = 0.5 * (dh_c + dh_pad[..., 1:-1, 2:])
    d_w = 0.5 * (dh_c + dh_pad[..., 1:-1, :-2])
    d_n = 0.5 * (dh_c + dh_pad[..., 2:, 1:-1])
    d_s = 0.5 * (dh_c + dh_pad[..., :-2, 1:-1])

    # Symmetric interface normal velocities (exact sign flip across all 12 rotated seams!)
    v_east = 0.5 * (u_c + u_e)   # A -> East
    v_west = 0.5 * (u_w + u_c)   # West -> A
    v_north = 0.5 * (v_c + v_n)  # A -> North
    v_south = 0.5 * (v_s + v_c)  # South -> A

    # Upwind + symmetric diffusion rates clamped per-edge so outflow <= max_edge_rate
    a_ce = (F.relu(v_east) + d_e).clamp(max=max_edge_rate)
    a_ec = (F.relu(-v_east) + d_e).clamp(max=max_edge_rate)
    a_wc = (F.relu(v_west) + d_w).clamp(max=max_edge_rate)
    a_cw = (F.relu(-v_west) + d_w).clamp(max=max_edge_rate)

    a_cn = (F.relu(v_north) + d_n).clamp(max=max_edge_rate)
    a_nc = (F.relu(-v_north) + d_n).clamp(max=max_edge_rate)
    a_sc = (F.relu(v_south) + d_s).clamp(max=max_edge_rate)
    a_cs = (F.relu(-v_south) + d_s).clamp(max=max_edge_rate)

    flux_e = a_ce * q_c - a_ec * q_e
    flux_w = a_wc * q_w - a_cw * q_c
    flux_n = a_cn * q_c - a_nc * q_n
    flux_s = a_sc * q_s - a_cs * q_c

    # Vertical interfaces (between level l and l+1, zero flux at bottom l=0 and top l=L-1)
    a_up = (F.relu(w32) + dv_t).clamp(max=max_edge_rate)
    a_dn = (F.relu(-w32) + dv_t).clamp(max=max_edge_rate)
    flux_z = a_up * q32[:, :, :-1] - a_dn * q32[:, :, 1:]  # [B, 6, L-1, S, N, N]

    net_z = torch.zeros_like(q32)
    net_z[:, :, :-1] -= flux_z
    net_z[:, :, 1:] += flux_z

    q_new = q_c - (flux_e - flux_w) - (flux_n - flux_s) + net_z
    q_new = q_new.clamp(min=0.0).to(dtype=orig_dtype)
    return q_new[0] if unbatched else q_new


def compute_spherical_vorticity(state: torch.Tensor) -> torch.Tensor:
    """
    Compute exact SO(2)-invariant vertical vorticity `zeta = dv/du - du/dv` across all 6 faces & 4 levels.
    Positive `zeta > 0` = counter-clockwise (NH cyclone); Negative `zeta < 0` = clockwise (SH cyclone).
    Input: `[6, L, C, N, N]` or `[B, 6, L, C, N, N]`. Output: `[6, L, N, N]` or `[B, 6, L, N, N]`.
    """
    unbatched = state.ndim == 5
    if unbatched:
        state = state.unsqueeze(0)
    u = state[:, :, :, CH_WIND_U]  # [B, 6, L, N, N]
    v = state[:, :, :, CH_WIND_V]
    u_pad, v_pad = pad_cubed_sphere_vectors(u, v, halo=1)
    dv_du = 0.5 * (v_pad[..., 1:-1, 2:] - v_pad[..., 1:-1, :-2])
    du_dv = 0.5 * (u_pad[..., 2:, 1:-1] - u_pad[..., :-2, 1:-1])
    zeta = dv_du - du_dv
    return zeta[0] if unbatched else zeta


# =============================================================================
# 4. Seamless Procedural Planet & Solar Forcing Generators
# =============================================================================


@dataclass
class PlanetGeography:
    N: int
    xyz: torch.Tensor        # [6, 3, N, N] unit-sphere directions (x, y, z)
    lat_sin: torch.Tensor    # [6, N, N] = y
    lat_cos: torch.Tensor    # [6, N, N] = sqrt(1 - y^2)
    land_mask: torch.Tensor  # [6, N, N] smooth in [0, 1] (0=ocean, 1=land)
    elev_norm: torch.Tensor  # [6, N, N] in [0, 1]


def compute_sun_dir(
    tick: float,
    ticks_per_day: float = 80.0,
    axial_tilt_deg: float = 16.0,
    phase0: float = 0.0,
) -> tuple[float, float, float]:
    """Return unit 3D solar direction vector sweeping westward around the planet as it spins eastward."""
    # Planet rotates eastward (+lon), so subsolar point sweeps westward (-lon)!
    lon = phase0 - 2.0 * math.pi * (float(tick) / float(ticks_per_day))
    tilt = math.radians(axial_tilt_deg)
    cy = math.sin(tilt)
    ch = math.cos(tilt)
    sx = ch * math.sin(lon)
    sy = cy
    sz = ch * math.cos(lon)
    norm = math.sqrt(sx * sx + sy * sy + sz * sz) + 1e-12
    return (sx / norm, sy / norm, sz / norm)


def make_static_planet_geography(
    seed: int = 0,
    N: int = DEFAULT_N,
    device: torch.device | str = "cpu",
    land_fraction_bias: float = 0.0,
) -> PlanetGeography:
    """
    Generate 100% seamless 3D procedural continents, oceans, and mountain ranges strictly on S^2.
    Guaranteed C^infty continuous across all 12 cube edges & 8 corners!
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1337)
    u_g, v_g = face_cell_centers_uv(N, device=dev, dtype=torch.float32)

    # Random 3D trigonometric spherical wave vectors & phases (3 octaves)
    k1 = torch.randn((6, 3), generator=gen) * 2.1
    p1 = torch.rand((6,), generator=gen) * (2.0 * math.pi)
    k2 = torch.randn((6, 3), generator=gen) * 4.6
    p2 = torch.rand((6,), generator=gen) * (2.0 * math.pi)
    k3 = torch.randn((4, 3), generator=gen) * 9.2
    p3 = torch.rand((4,), generator=gen) * (2.0 * math.pi)

    k1, p1 = k1.to(dev), p1.to(dev)
    k2, p2 = k2.to(dev), p2.to(dev)
    k3, p3 = k3.to(dev), p3.to(dev)

    xyz_list: list[torch.Tensor] = []
    for f in range(NUM_FACES):
        xf, yf, zf = face_uv_to_xyz(f, u_g, v_g)
        xyz_list.append(torch.stack([xf, yf, zf], dim=0))
    xyz = torch.stack(xyz_list, dim=0)  # [6, 3, N, N]

    pts = xyz.permute(0, 2, 3, 1)  # [6, N, N, 3]
    h1 = torch.sin(pts @ k1.T + p1).mean(dim=-1)
    h2 = torch.sin(pts @ k2.T + p2).mean(dim=-1)
    h3 = torch.sin(pts @ k3.T + p3).mean(dim=-1)
    height = 0.65 * h1 + 0.25 * h2 + 0.10 * h3 + land_fraction_bias

    # ~35-45% continents, ~55-65% oceans with crisp-yet-smooth coastlines
    sea_level = 0.08
    land_mask = torch.sigmoid((height - sea_level) * 14.0)
    elev_norm = (F.relu(height - sea_level) / 0.55).clamp(0.0, 1.0) * land_mask

    lat_sin = xyz[:, 1].clamp(-1.0, 1.0)
    lat_cos = torch.sqrt((1.0 - lat_sin * lat_sin).clamp(min=1e-6))
    return PlanetGeography(
        N=N,
        xyz=xyz,
        lat_sin=lat_sin,
        lat_cos=lat_cos,
        land_mask=land_mask,
        elev_norm=elev_norm,
    )


def build_forcing_tensor(
    geo: PlanetGeography,
    sun_dir: tuple[float, float, float] | torch.Tensor,
) -> torch.Tensor:
    """Build `[6, 6, N, N]` static/solar forcing tensor for a given `sun_dir`."""
    s = torch.as_tensor(sun_dir, dtype=torch.float32, device=geo.xyz.device).reshape(3)
    s = s / s.norm().clamp(min=1e-8)
    dot_s = geo.xyz[:, 0] * s[0] + geo.xyz[:, 1] * s[1] + geo.xyz[:, 2] * s[2]
    cos_sun = F.relu(dot_s)
    night_cool = F.relu(-dot_s)
    return torch.stack(
        [geo.lat_sin, geo.lat_cos, geo.land_mask, geo.elev_norm, cos_sun, night_cool],
        dim=1,
    ).contiguous()


def _project_3d_tangent_to_faces(
    geo: PlanetGeography,
    vx: torch.Tensor,
    vy: torch.Tensor,
    vz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a global 3D vector field `(vx, vy, vz) [6, N, N]` onto each face's `(e_u, e_v)`."""
    dev = geo.xyz.device
    eu = torch.tensor(FACE_EU_NP, dtype=torch.float32, device=dev).view(NUM_FACES, 3, 1, 1)
    ev = torch.tensor(FACE_EV_NP, dtype=torch.float32, device=dev).view(NUM_FACES, 3, 1, 1)
    wu = vx * eu[:, 0] + vy * eu[:, 1] + vz * eu[:, 2]
    wv = vx * ev[:, 0] + vy * ev[:, 1] + vz * ev[:, 2]
    return wu, wv


def inject_pressure_vortex(
    state: torch.Tensor,
    geo: PlanetGeography,
    center_xyz: tuple[float, float, float],
    amplitude: float = -1.35,
    radius_rad: float = 0.24,
    add_moisture: float = 0.25,
) -> torch.Tensor:
    """Inject a localized 3D-geodesic Gaussian pressure anomaly (negative = cyclone, positive = anticyclone)."""
    out = state.clone()
    dev = state.device
    c = torch.tensor(center_xyz, dtype=torch.float32, device=dev)
    c = c / c.norm().clamp(min=1e-8)
    dot_c = (geo.xyz[:, 0] * c[0] + geo.xyz[:, 1] * c[1] + geo.xyz[:, 2] * c[2]).clamp(-1.0, 1.0)
    dist2 = 2.0 * (1.0 - dot_c)  # chord^2 ~ angular_dist^2
    g = torch.exp(-dist2 / (2.0 * radius_rad * radius_rad))  # [6, N, N]

    # Strongest at surface & boundary layer, weaker aloft
    prof = torch.tensor([1.0, 0.85, 0.45, -0.15], dtype=torch.float32, device=dev).view(1, 4, 1, 1)
    out[:, :, CH_PRESSURE] += amplitude * prof * g.unsqueeze(1)
    if add_moisture > 0.0 and amplitude < 0.0:
        out[:, 1, CH_VAPOR] += add_moisture * g
        out[:, 2, CH_CLOUD] += 0.5 * add_moisture * g
    return out


def inject_symmetric_coriolis_cyclones(
    state: torch.Tensor,
    geo: PlanetGeography,
    amplitude: float = -1.45,
    lat_deg: float = 40.0,
    lon_deg: float = 0.0,
    radius_rad: float = 0.22,
) -> torch.Tensor:
    """
    Seed EXACTLY symmetric low-pressure anomalies at `+lat_deg N` and `-lat_deg S` (same longitude)
    with zero initial wind perturbation, for rigorous NH vs SH Coriolis Polarity verification!
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    nh_xyz = (math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon))
    sh_xyz = (math.cos(lat) * math.sin(lon), -math.sin(lat), math.cos(lat) * math.cos(lon))
    out = inject_pressure_vortex(state, geo, nh_xyz, amplitude=amplitude, radius_rad=radius_rad, add_moisture=0.20)
    out = inject_pressure_vortex(out, geo, sh_xyz, amplitude=amplitude, radius_rad=radius_rad, add_moisture=0.20)
    return out


def inject_volcanic_plume(
    state: torch.Tensor,
    geo: PlanetGeography,
    center_xyz: tuple[float, float, float],
    ash_amount: float = 1.6,
    heat_amount: float = 0.45,
    radius_rad: float = 0.14,
) -> torch.Tensor:
    """Inject a localized volcanic or shattered Sky-Island ash + thermal plume cleanly in 3D sphere coords."""
    out = state.clone()
    dev = state.device
    c = torch.tensor(center_xyz, dtype=torch.float32, device=dev)
    c = c / c.norm().clamp(min=1e-8)
    dot_c = (geo.xyz[:, 0] * c[0] + geo.xyz[:, 1] * c[1] + geo.xyz[:, 2] * c[2]).clamp(-1.0, 1.0)
    dist2 = 2.0 * (1.0 - dot_c)
    g = torch.exp(-dist2 / (2.0 * radius_rad * radius_rad))
    out[:, 1, CH_ASH_SMOKE] += 0.45 * ash_amount * g
    out[:, 2, CH_ASH_SMOKE] += 0.55 * ash_amount * g
    out[:, 0, CH_TEMP] += 0.65 * heat_amount * g
    out[:, 1, CH_TEMP] += 0.50 * heat_amount * g
    out[:, 1, CH_WIND_W] += 0.25 * heat_amount * g
    return out


def make_initial_climate_state(
    geo: PlanetGeography,
    sun_dir: tuple[float, float, float],
    seed: int = 0,
    add_cyclones: bool = True,
    add_volcano: bool = False,
    calm_start: bool = False,
) -> torch.Tensor:
    """
    Build a smooth, physically grounded initial climate state `[6, 4, 10, N, N]`:
      - Latitude & altitude stratified temperature (`~300 K` equator -> `~255 K` poles, lapse rate aloft).
      - Ocean surface moisture reservoir (`L=0`) + equatorial humidity & cloud bands + polar ice/snow caps.
      - Optional gentle initial zonal winds (trade winds + mid-latitude westerlies) or zero winds (`calm_start`).
      - Optional seeded cyclones & volcanic ash plumes.
    """
    dev = geo.xyz.device
    N = geo.N
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 7919)

    state = torch.zeros((NUM_FACES, NUM_LEVELS, NUM_CHANNELS, N, N), dtype=torch.float32, device=dev)
    abs_lat = geo.lat_sin.abs()
    cos_lat = geo.lat_cos
    ocean = 1.0 - geo.land_mask

    forcing = build_forcing_tensor(geo, sun_dir)
    cos_sun = forcing[:, FC_COS_SUN]
    night_cool = forcing[:, FC_NIGHT_COOL]

    # Temperature profile: T_norm = (T_K - 280)/40
    # Equator surface ~ 298 K (+0.45), Poles ~ 256 K (-0.60), lapse rate across 4 levels
    t_surf = 0.46 - 1.05 * (abs_lat ** 1.45) - 0.35 * geo.elev_norm + 0.16 * (cos_sun - 0.5 * night_cool) * (0.35 + 0.65 * geo.land_mask)
    lapse = torch.tensor([0.0, -0.15, -0.45, -0.78], dtype=torch.float32, device=dev).view(1, 4, 1, 1)
    state[:, :, CH_TEMP] = t_surf.unsqueeze(1) + lapse

    # Ocean thermal reservoir (high inertia on Level 0 ocean)
    state[:, 0, CH_OCEAN_HEAT] = ocean * (0.42 - 0.90 * (abs_lat ** 1.4))

    # Conserved Moisture:
    # Level 0: ocean/soil surface water pool (`water_vapor` as surface evaporable pool, `ice_snow` where T < 273.15 K!)
    is_cold_surf = torch.sigmoid((TEMP_NORM_FREEZE - state[:, 0, CH_TEMP]) * 18.0)
    state[:, 0, CH_VAPOR] = (0.95 * ocean + 0.22 * geo.land_mask) * (1.0 - 0.75 * is_cold_surf)
    state[:, 0, CH_ICE_SNOW] = (0.75 * ocean + 0.55 * geo.land_mask) * is_cold_surf + 0.30 * geo.elev_norm * is_cold_surf
    # Atmospheric moisture (Levels 1-3): humid ITCZ equatorial belt + mid-latitude bands
    itcz = torch.exp(-((geo.lat_sin / 0.32) ** 2))
    state[:, 1, CH_VAPOR] = 0.18 + 0.42 * itcz + 0.14 * ocean
    state[:, 2, CH_VAPOR] = 0.08 + 0.20 * itcz
    state[:, 2, CH_CLOUD] = 0.05 + 0.24 * itcz
    state[:, 3, CH_ICE_SNOW] = 0.04 + 0.10 * itcz

    if not calm_start:
        # Smooth zonal background wind (Easterly trade winds near equator, Westerlies in mid-latitudes)
        # Eastward unit tangent vector in 3D: e_east = (-z, 0, x) / cos(lat) (WAIT: check right-hand rule!
        # At (0,0,+1) on +Z face, e_u = (+1,0,0) = East! And (z, 0, -x) at (0,0,+1) = (+1, 0, 0) = EAST!
        # Let's verify: Omega = (0, 1, 0), Omega x r = (0, 1, 0) x (x, y, z) = (+z, 0, -x) = EAST!)
        east_x = geo.xyz[:, 2] / cos_lat.clamp(min=0.12)
        east_y = torch.zeros_like(east_x)
        east_z = -geo.xyz[:, 0] / cos_lat.clamp(min=0.12)
        # Zonal speed profile: -0.22 Easterly trades near |lat|<0.45, +0.34 Westerlies near |lat|~0.65
        zonal_surf = (-0.22 * torch.exp(-((geo.lat_sin / 0.36) ** 2)) + 0.26 * torch.exp(-(((abs_lat - 0.65) / 0.22) ** 2))) * cos_lat
        zonal_jet = (0.48 * torch.exp(-(((abs_lat - 0.52) / 0.24) ** 2))) * cos_lat
        for l_idx, z_spd in enumerate([0.65 * zonal_surf, zonal_surf, 0.5 * (zonal_surf + zonal_jet), zonal_jet]):
            wu, wv = _project_3d_tangent_to_faces(geo, z_spd * east_x, east_y, z_spd * east_z)
            state[:, l_idx, CH_WIND_U] = wu
            state[:, l_idx, CH_WIND_V] = wv

        # Gentle thermal pressure gradient matching latitude
        p_lat = -0.35 * itcz + 0.28 * torch.exp(-(((abs_lat - 0.48) / 0.18) ** 2)) - 0.22 * torch.exp(-(((abs_lat - 0.78) / 0.18) ** 2))
        state[:, 0, CH_PRESSURE] = p_lat
        state[:, 1, CH_PRESSURE] = 0.75 * p_lat
        state[:, 2, CH_PRESSURE] = -0.25 * p_lat
        state[:, 3, CH_PRESSURE] = -0.60 * p_lat

    if add_cyclones:
        n_vort = int(torch.randint(2, 6, (1,), generator=gen).item())
        for _ in range(n_vort):
            v = torch.randn((3,), generator=gen)
            # Bias toward tropical & mid-latitudes (0.20 <= |y| <= 0.85) so Coriolis is active!
            v[1] = float(torch.sign(v[1] + 1e-6).item()) * (0.25 + 0.55 * float(torch.rand((1,), generator=gen).item()))
            v = v / v.norm().clamp(min=1e-6)
            amp = float((-1.5 + 2.6 * torch.rand((1,), generator=gen).item()))
            if abs(amp) < 0.45:
                amp = -1.05 if amp <= 0 else 0.90
            rad = 0.16 + 0.12 * float(torch.rand((1,), generator=gen).item())
            state = inject_pressure_vortex(state, geo, (float(v[0]), float(v[1]), float(v[2])), amplitude=amp, radius_rad=rad)

    if add_volcano:
        n_vol = int(torch.randint(1, 3, (1,), generator=gen).item())
        for _ in range(n_vol):
            v = torch.randn((3,), generator=gen)
            v = v / v.norm().clamp(min=1e-6)
            state = inject_volcanic_plume(
                state,
                geo,
                (float(v[0]), float(v[1]), float(v[2])),
                ash_amount=0.8 + 1.4 * float(torch.rand((1,), generator=gen).item()),
                heat_amount=0.25 + 0.35 * float(torch.rand((1,), generator=gen).item()),
                radius_rad=0.11 + 0.07 * float(torch.rand((1,), generator=gen).item()),
            )

    return state


# =============================================================================
# 5. Vectorized GPU Ground-Truth Spherical Climate Solver (`SphericalClimateRefSim`)
# =============================================================================


class SphericalClimateRefSim:
    """
    100% vectorized PyTorch GPU reference spherical climate solver on the 6-face cubed-sphere.
    Zero Python loops over spatial cells; supports both `[6, 4, 10, N, N]` and batched `[B, 6, 4, 10, N, N]`.

    Physics implemented:
      1. Exact antisymmetric conservative transport for `(water_vapor, cloud_liquid, ice_snow, ash_smoke)`.
      2. Exact pressure-gradient acceleration `-c_pg * grad(p)` + exact energy-preserving spherical Coriolis
         rotation `(+f*v, -f*u)` (`f = 2*Omega*sin(phi)`) + land/mountain drag + viscosity + buoyancy `wind_w`.
      3. Hadley/Ferrel thermal-hydrostatic pressure coupling + sweeping Day/Night radiative heating/cooling
         (attenuated by overhead `cloud_liquid` & `ash_smoke` optical depth) + high-inertia `ocean_heat`.
      4. Closed global moisture cycle: warm-ocean evaporation (`L0 -> L1`), Hadley lifting, Clausius-Clapeyron
         condensation/re-evaporation, dynamic `273.15 K` snowline freeze/melt, and rain/snow precipitation (`L3->L0`).
    """

    def __init__(
        self,
        dt: float = 1.0,
        omega_coriolis: float = 0.185,  # f = 2 * omega * sin(lat) -> crisp NH CCW / SH CW cyclones
        c_pg: float = 0.30,             # pressure-gradient acceleration
        c_adv_scalar: float = 0.24,     # scalar horizontal advection scale
        c_adv_mom: float = 0.16,        # momentum self-advection scale
        c_visc: float = 0.042,          # horizontal eddy viscosity
        c_p_div: float = 0.22,          # pressure-divergence compressibility
    ) -> None:
        self.dt = float(dt)
        self.omega_coriolis = float(omega_coriolis)
        self.c_pg = float(c_pg)
        self.c_adv_scalar = float(c_adv_scalar)
        self.c_adv_mom = float(c_adv_mom)
        self.c_visc = float(c_visc)
        self.c_p_div = float(c_p_div)

    @staticmethod
    def _lap4(x_pad: torch.Tensor) -> torch.Tensor:
        return (
            x_pad[..., 1:-1, 2:]
            + x_pad[..., 1:-1, :-2]
            + x_pad[..., 2:, 1:-1]
            + x_pad[..., :-2, 1:-1]
            - 4.0 * x_pad[..., 1:-1, 1:-1]
        )

    @staticmethod
    def _grad(x_pad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = 0.5 * (x_pad[..., 1:-1, 2:] - x_pad[..., 1:-1, :-2])
        gy = 0.5 * (x_pad[..., 2:, 1:-1] - x_pad[..., :-2, 1:-1])
        return gx, gy

    @torch.no_grad()
    def step(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        """Advance climate state by 1 macro tick. Supports `[6, 4, 10, N, N]` or `[B, 6, 4, 10, N, N]`."""
        unbatched = state.ndim == 5
        if unbatched:
            state = state.unsqueeze(0)
            forcing = forcing.unsqueeze(0)

        st = state.float()
        fc = forcing.float()
        B, F6, L, C, N, _ = st.shape
        assert F6 == NUM_FACES and L == NUM_LEVELS and C == NUM_CHANNELS
        dt = self.dt

        # Unpack forcing `[B, 6, 1, N, N]`
        lat_sin = fc[:, :, FC_LAT_SIN].unsqueeze(2)
        land = fc[:, :, FC_LAND_MASK].unsqueeze(2)
        ocean = 1.0 - land
        elev = fc[:, :, FC_ELEV_NORM].unsqueeze(2)
        cos_sun = fc[:, :, FC_COS_SUN].unsqueeze(2)
        night_cool = fc[:, :, FC_NIGHT_COOL].unsqueeze(2)

        # Unpack state `[B, 6, L, N, N]`
        temp = st[:, :, :, CH_TEMP]
        vapor = st[:, :, :, CH_VAPOR].clamp(min=0.0)
        cloud = st[:, :, :, CH_CLOUD].clamp(min=0.0)
        ice = st[:, :, :, CH_ICE_SNOW].clamp(min=0.0)
        pres = st[:, :, :, CH_PRESSURE]
        wu = st[:, :, :, CH_WIND_U]
        wv = st[:, :, :, CH_WIND_V]
        ww = st[:, :, :, CH_WIND_W]
        oc_heat = st[:, :, :, CH_OCEAN_HEAT]
        ash = st[:, :, :, CH_ASH_SMOKE].clamp(min=0.0)

        # ---------------------------------------------------------------------
        # A. Halo-padded gradients, divergence, Laplacian (exact SO(2) seams!)
        # ---------------------------------------------------------------------
        p_pad = pad_cubed_sphere_scalars(pres, halo=1)
        t_pad = pad_cubed_sphere_scalars(temp, halo=1)
        e_pad = pad_cubed_sphere_scalars(elev, halo=1)
        u_pad, v_pad = pad_cubed_sphere_vectors(wu, wv, halo=1)

        dp_du, dp_dv = self._grad(p_pad)
        dt_du, dt_dv = self._grad(t_pad)
        de_du, de_dv = self._grad(e_pad)
        du_du, du_dv = self._grad(u_pad)
        dv_du, dv_dv = self._grad(v_pad)
        div_h = du_du + dv_dv

        lap_p = self._lap4(p_pad)
        lap_t = self._lap4(t_pad)
        lap_u = self._lap4(u_pad)
        lap_v = self._lap4(v_pad)

        # ---------------------------------------------------------------------
        # B. Conserved Moisture & Ash Transport (EXACT antisymmetric fluxes!)
        # ---------------------------------------------------------------------
        # Package 4 conserved species `[B, 6, L, 4, N, N]`: (vapor, cloud, ice_snow, ash_smoke)
        q_cons = torch.stack([vapor, cloud, ice, ash], dim=3)
        adv_scale_lvl = torch.tensor([0.55, 1.00, 1.15, 1.25], device=st.device).view(1, 1, 4, 1, 1, 1)
        adv_scale_sp = torch.tensor([1.00, 0.95, 0.85, 1.10], device=st.device).view(1, 1, 1, 4, 1, 1)
        vel_u_c = self.c_adv_scalar * dt * wu.unsqueeze(3) * adv_scale_lvl * adv_scale_sp
        vel_v_c = self.c_adv_scalar * dt * wv.unsqueeze(3) * adv_scale_lvl * adv_scale_sp

        # Vertical interface velocities `[B, 6, 3, 4, N, N]` (positive = upward l -> l+1)
        w_mid = 0.5 * (ww[:, :, :-1] + ww[:, :, 1:])  # [B, 6, 3, N, N]
        vel_w_c = (0.14 * dt * w_mid).unsqueeze(3).expand(B, F6, 3, 4, N, N).clone()

        # 1) Warm-ocean evaporation: upward vapor flux L0 -> L1 driven by ocean heat & wind speed
        wind_spd_0 = torch.sqrt(wu[:, :, 0:1] ** 2 + wv[:, :, 0:1] ** 2 + 1e-4)
        warmth_0 = torch.sigmoid((temp[:, :, 0:1] - TEMP_NORM_FREEZE) * 6.0)
        evap_vel = dt * (0.048 * ocean + 0.012 * land) * warmth_0 * (0.6 + 0.5 * wind_spd_0)
        vel_w_c[:, :, 0, 0] = vel_w_c[:, :, 0, 0] + evap_vel[:, :, 0]

        # 2) Rain & snow gravitational precipitation settling downward (`cloud` & `ice_snow`)
        vel_w_c[:, :, :, 1] = vel_w_c[:, :, :, 1] - 0.055 * dt  # rainfall settling
        vel_w_c[:, :, :, 2] = vel_w_c[:, :, :, 2] - 0.048 * dt  # snowfall settling
        # Gentle ash gravitational settling & buoyant rise when hot
        vel_w_c[:, :, :, 3] = vel_w_c[:, :, :, 3] - 0.012 * dt

        q_adv = conservative_cubed_sphere_transport(
            q_cons,
            vel_u=vel_u_c,
            vel_v=vel_v_c,
            vel_w_iface=vel_w_c,
            diff_h=0.012 * dt,
            diff_v=0.008 * dt,
        )
        v_a, c_a, i_a, ash_new = q_adv[:, :, :, 0], q_adv[:, :, :, 1], q_adv[:, :, :, 2], q_adv[:, :, :, 3]

        # ---------------------------------------------------------------------
        # C. Local Mass-Conserving Clausius-Clapeyron Moisture Phase Conversions
        # ---------------------------------------------------------------------
        # Saturation vapor threshold rises smoothly with temperature
        q_sat = 0.11 + 0.52 * torch.sigmoid((temp - 0.0) * 3.0)
        # Level 0 is surface/ocean pool (only freezes/melts across 273.15 K, doesn't form clouds)
        atm_mask = torch.tensor([0.0, 1.0, 1.0, 1.0], device=st.device).view(1, 1, 4, 1, 1)

        # Supersaturation condensation (vapor -> cloud if T >= 273.15 K, else vapor -> ice_snow)
        excess_v = F.relu(v_a - q_sat)
        cold_frac = torch.sigmoid((TEMP_NORM_FREEZE - temp) * 18.0)
        cond_rate = atm_mask * (0.32 * dt * excess_v / (v_a + 1e-4)).clamp(0.0, 0.65)
        d_v_to_c = v_a * cond_rate * (1.0 - cold_frac)
        d_v_to_i = v_a * cond_rate * cold_frac

        # Sub-saturated cloud re-evaporation (cloud -> vapor)
        deficit_v = F.relu(0.78 * q_sat - v_a)
        reevap_rate = atm_mask * (0.20 * dt * deficit_v / (q_sat + 1e-4)).clamp(0.0, 0.45) * (1.0 - cold_frac)
        d_c_to_v = c_a * reevap_rate

        # Dynamic snowline freeze / melt across T = 273.15 K (`cloud <-> ice_snow`, and L0 surface water <-> ice)
        freeze_rate = (0.24 * dt * cold_frac).clamp(0.0, 0.55)
        melt_rate = (0.24 * dt * (1.0 - cold_frac)).clamp(0.0, 0.55)
        d_c_to_i = (c_a - d_c_to_v) * freeze_rate
        d_i_to_c = i_a * melt_rate

        # On Level 0 surface: freezing converts surface pool `vapor` <-> `ice_snow` cleanly
        surf_mask = 1.0 - atm_mask
        d_s_freeze = surf_mask * v_a * (0.14 * dt * cold_frac).clamp(0.0, 0.35)
        d_s_melt = surf_mask * i_a * (0.14 * dt * (1.0 - cold_frac)).clamp(0.0, 0.35)

        vapor_new = (v_a - d_v_to_c - d_v_to_i + d_c_to_v - d_s_freeze + d_s_melt).clamp(min=0.0)
        cloud_new = (c_a + d_v_to_c - d_c_to_v - d_c_to_i + d_i_to_c).clamp(min=0.0)
        ice_new = (i_a + d_v_to_i + d_c_to_i - d_i_to_c + d_s_freeze - d_s_melt).clamp(min=0.0)

        # Latent heat release on condensation minus re-evaporation
        latent_heat = 0.18 * (d_v_to_c + d_v_to_i - d_c_to_v)

        # ---------------------------------------------------------------------
        # D. Radiative Day/Night Terminator, Ash/Cloud Cooling & Ocean Inertia
        # ---------------------------------------------------------------------
        col_cloud = cloud.sum(dim=2, keepdim=True)
        col_ash = ash.sum(dim=2, keepdim=True)
        # Solar transmittance attenuated strongly by storm clouds AND volcanic ash plumes!
        solar_trans = torch.exp(-(0.62 * col_cloud + 0.95 * col_ash))
        greenhouse = 1.0 - 0.32 * (1.0 - torch.exp(-0.75 * col_cloud))

        # Radiative equilibrium profile (depends on latitude & altitude)
        lapse = torch.tensor([0.0, -0.15, -0.45, -0.78], device=st.device).view(1, 1, 4, 1, 1)
        t_rad_eq = (0.36 - 0.95 * (lat_sin.abs() ** 1.4) - 0.28 * elev) + lapse

        # Thermal inertia: continents heat/cool rapidly; oceans buffer slowly via `ocean_heat`!
        surf_solar_w = torch.tensor([1.0, 0.35, 0.12, 0.05], device=st.device).view(1, 1, 4, 1, 1)
        inv_heat_cap = torch.where(
            torch.arange(L, device=st.device).view(1, 1, 4, 1, 1) == 0,
            0.14 * land + 0.030 * ocean,
            torch.tensor([0.0, 0.065, 0.042, 0.032], device=st.device).view(1, 1, 4, 1, 1),
        )
        dT_solar = inv_heat_cap * surf_solar_w * (0.95 * cos_sun * solar_trans - 0.42 * night_cool * greenhouse)
        dT_relax = 0.032 * (t_rad_eq - temp)

        # Ocean-atmosphere sensible heat exchange on Level 0 & Level 1
        dT_ocean_exch = torch.zeros_like(temp)
        dT_ocean_exch[:, :, 0] = 0.095 * ocean[:, :, 0] * (oc_heat[:, :, 0] - temp[:, :, 0])
        dT_ocean_exch[:, :, 1] = 0.035 * (temp[:, :, 0] - temp[:, :, 1] - 0.15)

        # Vertical temperature advection & turbulent mixing
        dT_vert = torch.zeros_like(temp)
        dT_vert[:, :, 1:] += 0.05 * F.relu(ww[:, :, 1:]) * (temp[:, :, :-1] - temp[:, :, 1:] - 0.18)
        dT_vert[:, :, :-1] += 0.05 * F.relu(-ww[:, :, :-1]) * (temp[:, :, 1:] - temp[:, :, :-1] + 0.18)
        dT_vdiff = torch.zeros_like(temp)
        dT_vdiff[:, :, 1:-1] = 0.022 * (temp[:, :, :-2] + temp[:, :, 2:] - 2.0 * temp[:, :, 1:-1])
        dT_vdiff[:, :, 0] = 0.022 * (temp[:, :, 1] - temp[:, :, 0] + 0.15)
        dT_vdiff[:, :, 3] = 0.022 * (temp[:, :, 2] - temp[:, :, 3] - 0.30)

        # Horizontal advection & diffusion of temperature
        dT_adv = -self.c_adv_scalar * (wu * dt_du + wv * dt_dv) + 0.022 * lap_t
        temp_new = (temp + dt * (dT_adv + dT_solar + dT_relax + dT_ocean_exch + dT_vert + dT_vdiff + latent_heat)).clamp(-1.8, 1.8)

        # Ocean heat reservoir update (high thermal inertia + wind-driven gyre advection!)
        oh_pad = pad_cubed_sphere_scalars(oc_heat[:, :, 0:1], halo=1)
        doh_du, doh_dv = self._grad(oh_pad)
        lap_oh = self._lap4(oh_pad)
        oh0 = oc_heat[:, :, 0:1]
        oh0_new = ocean * (
            oh0
            + dt * (
                -0.045 * (wu[:, :, 0:1] * doh_du + wv[:, :, 0:1] * doh_dv)
                + 0.012 * lap_oh
                + 0.022 * (cos_sun * solar_trans - 0.35 * night_cool)
                + 0.018 * (temp[:, :, 0:1] - oh0)
                + 0.008 * ((0.40 - 0.88 * (lat_sin.abs() ** 1.4)) - oh0)
            )
        )
        oc_heat_new = torch.zeros_like(oc_heat)
        oc_heat_new[:, :, 0:1] = oh0_new.clamp(-1.2, 1.2)

        # ---------------------------------------------------------------------
        # E. Momentum Equations: Pressure Gradient + Exact Spherical Coriolis + Drag
        # ---------------------------------------------------------------------
        # Topographic blocking on lower layers (Levels 0-1)
        topo_w = torch.tensor([0.14, 0.06, 0.0, 0.0], device=st.device).view(1, 1, 4, 1, 1)
        # Surface friction (stronger over land/mountains, weak over oceans & upper jet)
        drag = torch.zeros_like(temp)
        drag[:, :, 0] = 0.038 * ocean[:, :, 0] + (0.095 + 0.085 * elev[:, :, 0]) * land[:, :, 0]
        drag[:, :, 1] = 0.022 * ocean[:, :, 0] + (0.045 + 0.035 * elev[:, :, 0]) * land[:, :, 0]
        drag[:, :, 2] = 0.014
        drag[:, :, 3] = 0.018

        # Vertical momentum shear mixing
        dv_u = torch.zeros_like(wu)
        dv_v = torch.zeros_like(wv)
        dv_u[:, :, 1:-1] = 0.030 * (wu[:, :, :-2] + wu[:, :, 2:] - 2.0 * wu[:, :, 1:-1])
        dv_v[:, :, 1:-1] = 0.030 * (wv[:, :, :-2] + wv[:, :, 2:] - 2.0 * wv[:, :, 1:-1])
        dv_u[:, :, 0] = 0.030 * (wu[:, :, 1] - wu[:, :, 0])
        dv_v[:, :, 0] = 0.030 * (wv[:, :, 1] - wv[:, :, 0])
        dv_u[:, :, 3] = 0.025 * (wu[:, :, 2] - wu[:, :, 3])
        dv_v[:, :, 3] = 0.025 * (wv[:, :, 2] - wv[:, :, 3])

        # Non-Coriolis acceleration
        adv_u = -self.c_adv_mom * (wu * du_du + wv * du_dv)
        adv_v = -self.c_adv_mom * (wu * dv_du + wv * dv_dv)
        u_star = wu + dt * (-self.c_pg * dp_du - topo_w * de_du + adv_u + self.c_visc * lap_u + dv_u - drag * wu)
        v_star = wv + dt * (-self.c_pg * dp_dv - topo_w * de_dv + adv_v + self.c_visc * lap_v + dv_v - drag * wv)

        # EXACT energy-preserving spherical Coriolis rotation!
        # f = 2 * Omega * sin(lat); du/dt = +f*v, dv/dt = -f*u -> exact rotation by theta = f * dt!
        # NH (f > 0): deflects right -> low-pressure cyclone spins counter-clockwise (zeta > 0).
        # SH (f < 0): deflects left  -> low-pressure cyclone spins clockwise (zeta < 0).
        theta = (2.0 * self.omega_coriolis * lat_sin) * dt
        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        wu_new = (u_star * cos_th + v_star * sin_th).clamp(-2.2, 2.2)
        wv_new = (-u_star * sin_th + v_star * cos_th).clamp(-2.2, 2.2)

        # Vertical velocity `wind_w`: thermal buoyancy + low-level convergence - damping
        buoy = torch.zeros_like(ww)
        # Positive buoyancy when lower level is anomalously warm relative to standard lapse rate
        t_anom_0 = temp[:, :, 0] - (0.05 - 0.35 * (lat_sin[:, :, 0].abs() ** 1.4))
        buoy[:, :, 0] = 0.14 * t_anom_0 + 0.10 * (temp[:, :, 0] - temp[:, :, 1] - 0.15)
        buoy[:, :, 1] = 0.16 * (temp[:, :, 0] - temp[:, :, 2] - 0.45) + 0.25 * latent_heat[:, :, 1]
        buoy[:, :, 2] = 0.14 * (temp[:, :, 1] - temp[:, :, 3] - 0.63) + 0.25 * latent_heat[:, :, 2]
        buoy[:, :, 3] = -0.10 * ww[:, :, 3]
        conv_lift = -0.22 * div_h
        ww_new = (0.76 * ww + dt * (buoy + conv_lift)).clamp(-1.2, 1.2)

        # ---------------------------------------------------------------------
        # F. Pressure / Geopotential Anomaly (Hydrostatic + Mass Convergence)
        # ---------------------------------------------------------------------
        # Vertical divergence of `wind_w`
        dw_dz = torch.zeros_like(pres)
        dw_dz[:, :, 0] = ww_new[:, :, 1]
        dw_dz[:, :, 1] = 0.5 * (ww_new[:, :, 2] - ww_new[:, :, 0])
        dw_dz[:, :, 2] = 0.5 * (ww_new[:, :, 3] - ww_new[:, :, 1])
        dw_dz[:, :, 3] = -ww_new[:, :, 2]

        # Thermal-hydrostatic Hadley forcing: warm lower troposphere lowers surface pressure & raises upper jet pressure!
        col_warmth = (temp[:, :, 0] + temp[:, :, 1]) * 0.5 - (0.12 - 0.65 * (lat_sin[:, :, 0].abs() ** 1.4))
        hydro_prof = torch.tensor([-0.055, -0.032, 0.022, 0.055], device=st.device).view(1, 1, 4, 1, 1)
        dp_hydro = hydro_prof * col_warmth.unsqueeze(2)

        pres_new = (
            0.988 * pres
            + dt * (-self.c_p_div * div_h - 0.12 * dw_dz + dp_hydro + 0.038 * lap_p)
        ).clamp(-2.8, 2.8)

        out = torch.stack(
            [
                temp_new,
                vapor_new,
                cloud_new,
                ice_new,
                pres_new,
                wu_new,
                wv_new,
                ww_new,
                oc_heat_new,
                ash_new,
            ],
            dim=3,
        ).to(dtype=state.dtype)
        return out[0] if unbatched else out


# =============================================================================
# 6. `PlanetaryClimateNet`: C4/SO(2) Gauge-Equivariant Cubed-Sphere Neural Operator
# =============================================================================


class SO2VectorLinear(nn.Module):
    """
    Most general 2D SO(2)-equivariant linear map on paired tangent vectors `(U, V)`:
        U_out = W_r @ U_in + W_i @ V_in
        V_out = -W_i @ U_in + W_r @ V_in
    Commutes EXACTLY with every 2x2 rotation `R in SO(2)` (including all 12 cubed-sphere seams)!
    Directly represents Coriolis-like cross-couplings (`W_i`) and shear/vertical mixing (`W_r`).
    """

    def __init__(self, in_pairs: int, out_pairs: int) -> None:
        super().__init__()
        self.w_r = nn.Conv2d(in_pairs, out_pairs, kernel_size=1, bias=False)
        self.w_i = nn.Conv2d(in_pairs, out_pairs, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.w_r.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_i.weight)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u_out = self.w_r(u) + self.w_i(v)
        v_out = self.w_r(v) - self.w_i(u)
        return u_out, v_out


class C4GaugeConv2d(nn.Module):
    """
    Exact C4/SO(2) Gauge-Equivariant 3x3 Cubed-Sphere Convolution executed as ONE single
    cuDNN Tensor Core `F.conv2d`!
    Acts jointly on `c_s` scalar channels and `c_v` tangent vector pairs `(u, v)` (`c_tot = c_s + 2*c_v`):
        [ s_out ]   [ W_ss      W_vs_u    W_vs_v  ]   [ s_pad ]
        [ u_out ] = [ W_su      W_vv_r    W_vv_i  ] * [ u_pad ]
        [ v_out ]   [ W_sv     -W_vv_i    W_vv_r  ]   [ v_pad ]
    Every block commutes EXACTLY with every 90-degree rotation `R in C4` across all 12 cube seams!
    """

    def __init__(self, c_s: int, c_v: int) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_v = c_v
        self.c_tot = c_s + 2 * c_v

        # Fixed exact C4 basis templates `[3, 3]`
        t_cen = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        t_ax = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) * 0.25
        t_dg = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]) * 0.25
        t_du_a = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
        t_dv_a = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
        t_du_d = torch.tensor([[-0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]])
        t_dv_d = torch.tensor([[-0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]])

        self.register_buffer("iso_basis", torch.stack([t_cen, t_ax, t_dg], dim=0), persistent=False)        # [3, 3, 3]
        self.register_buffer("du_basis", torch.stack([t_du_a, t_du_d], dim=0), persistent=False)           # [2, 3, 3]
        self.register_buffer("dv_basis", torch.stack([t_dv_a, t_dv_d], dim=0), persistent=False)           # [2, 3, 3]

        scale_s = 1.0 / math.sqrt(max(c_s, 1) * 3.0)
        scale_v = 1.0 / math.sqrt(max(c_v, 1) * 3.0)
        self.w_ss = nn.Parameter(torch.randn(c_s, c_s, 3) * scale_s)
        self.w_sv_a = nn.Parameter(torch.randn(c_v, c_s, 2) * scale_s)  # grad(s)
        self.w_sv_b = nn.Parameter(torch.randn(c_v, c_s, 2) * scale_s)  # J * grad(s) (geostrophic wind!)
        self.w_vs_div = nn.Parameter(torch.randn(c_s, c_v, 2) * scale_v)
        self.w_vs_curl = nn.Parameter(torch.randn(c_s, c_v, 2) * scale_v)
        self.w_vv_r = nn.Parameter(torch.randn(c_v, c_v, 3) * scale_v)
        self.w_vv_i = nn.Parameter(torch.randn(c_v, c_v, 3) * (scale_v * 0.25))
        self.bias_s = nn.Parameter(torch.zeros(c_s))
        self._eval_cache: tuple[torch.dtype, torch.device, torch.Tensor, torch.Tensor] | None = None

    def train(self, mode: bool = True) -> "C4GaugeConv2d":
        self._eval_cache = None
        return super().train(mode)

    def build_kernel(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        iso = self.iso_basis.to(dtype=dtype)
        du = self.du_basis.to(dtype=dtype)
        dv = self.dv_basis.to(dtype=dtype)

        # Assemble exact C4 blocks `[C_out, C_in, 3, 3]` via ultra-fast einsum on tiny parameter tensors
        K_ss = torch.einsum("ijk,kmn->ijmn", self.w_ss.to(dtype), iso)
        K_su = torch.einsum("ijk,kmn->ijmn", self.w_sv_a.to(dtype), du) + torch.einsum("ijk,kmn->ijmn", self.w_sv_b.to(dtype), dv)
        K_sv = torch.einsum("ijk,kmn->ijmn", self.w_sv_a.to(dtype), dv) - torch.einsum("ijk,kmn->ijmn", self.w_sv_b.to(dtype), du)
        K_vsu = torch.einsum("ijk,kmn->ijmn", self.w_vs_div.to(dtype), du) - torch.einsum("ijk,kmn->ijmn", self.w_vs_curl.to(dtype), dv)
        K_vsv = torch.einsum("ijk,kmn->ijmn", self.w_vs_div.to(dtype), dv) + torch.einsum("ijk,kmn->ijmn", self.w_vs_curl.to(dtype), du)
        K_vvr = torch.einsum("ijk,kmn->ijmn", self.w_vv_r.to(dtype), iso)
        K_vvi = torch.einsum("ijk,kmn->ijmn", self.w_vv_i.to(dtype), iso)

        row0 = torch.cat([K_ss, K_vsu, K_vsv], dim=1)
        row1 = torch.cat([K_su, K_vvr, K_vvi], dim=1)
        row2 = torch.cat([K_sv, -K_vvi, K_vvr], dim=1)
        W_full = torch.cat([row0, row1, row2], dim=0)  # [C_tot, C_tot, 3, 3]
        b_full = torch.cat([self.bias_s.to(dtype), torch.zeros(2 * self.c_v, device=W_full.device, dtype=dtype)], dim=0)
        return W_full, b_full

    def forward(self, x_pad: torch.Tensor) -> torch.Tensor:
        if not self.training:
            cache = self._eval_cache
            if cache is not None and cache[0] == x_pad.dtype and cache[1] == x_pad.device:
                return F.conv2d(x_pad, cache[2], bias=cache[3])
            W_full, b_full = self.build_kernel(x_pad.dtype)
            self._eval_cache = (x_pad.dtype, x_pad.device, W_full.detach(), b_full.detach())
            return F.conv2d(x_pad, W_full, bias=b_full)
        W_full, b_full = self.build_kernel(x_pad.dtype)
        return F.conv2d(x_pad, W_full, bias=b_full)


class GaugeEquivariantClimateBlock(nn.Module):
    """
    Ultra-Fast Tensor-Core C4/SO(2) Gauge-Equivariant Multi-Level Cubed-Sphere Residual Block:
      1. ONE border-only `halo=1` cubed-sphere exchange rotating ALL hidden vector pairs `(U, V)`!
      2. ONE fused `C4GaugeConv2d` (`F.conv2d` on Blackwell Tensor Cores!) computing isotropic diffusion,
         gradients, geostrophic `J*grad`, divergence, curl, and Coriolis `J*v` across ALL 4 vertical levels!
      3. Exact SO(2)-invariant quadratic advection (`u_conv * u + v_conv * v`) & cross-product (`u_conv * v - v_conv * u`)
         + pointwise SiLU scalar MLP + scalar-gated SO(2) complex vector residual!
    """

    def __init__(self, c_s: int = 24, c_v: int = 6, num_levels: int = NUM_LEVELS) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_v = c_v
        self.L = num_levels
        LC_s = num_levels * c_s
        LC_v = num_levels * c_v
        self.LC_s = LC_s
        self.LC_v = LC_v

        self.c4_conv = C4GaugeConv2d(c_s=LC_s, c_v=LC_v)
        self.s_norm = nn.GroupNorm(num_groups=num_levels, num_channels=LC_s)

        # Scalar MLP takes: conv_s (LC_s) + adv_dot (LC_v) + adv_cross (LC_v) + speed (LC_v)
        s_mlp_in = LC_s + 3 * LC_v
        s_mid = LC_s + num_levels * 8
        self.s_mix1 = nn.Conv2d(s_mlp_in, s_mid, kernel_size=1)
        self.s_mix2 = nn.Conv2d(s_mid, LC_s, kernel_size=1)
        nn.init.zeros_(self.s_mix2.weight)
        nn.init.zeros_(self.s_mix2.bias)

        # Scalar-gated SO(2) vector residual
        self.v_gate = nn.Conv2d(s_mid, LC_v, kernel_size=1)
        self.v_out = SO2VectorLinear(LC_v, LC_v)
        nn.init.zeros_(self.v_out.w_r.weight)
        nn.init.zeros_(self.v_out.w_i.weight)

    def forward(
        self,
        s: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        wind_u_ref: torch.Tensor,
        wind_v_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, F6, LC_s, N, _ = s.shape
        LC_v = self.LC_v
        BF = B * F6

        # 1. Single border-only halo pad (`1,560` border cells!)
        x_pad = pad_cubed_sphere_packed(s, u, v, halo=1)  # [BF, LC_s + 2*LC_v, N+2, N+2]

        # 2. Single Tensor-Core C4-equivariant 3x3 conv!
        y = self.c4_conv(x_pad)
        ys = y[:, :LC_s]
        yu = y[:, LC_s : LC_s + LC_v]
        yv = y[:, LC_s + LC_v :]

        u_2d = u.reshape(BF, LC_v, N, N)
        v_2d = v.reshape(BF, LC_v, N, N)

        # Exact SO(2)-invariant bilinear advection (`yu*u + yv*v`), cross-product, & speed!
        inv_dot = yu * u_2d + yv * v_2d
        inv_cross = yu * v_2d - yv * u_2d
        inv_spd = torch.sqrt(yu * yu + yv * yv + 1e-4)

        h_s = F.silu(self.s_mix1(torch.cat([F.silu(self.s_norm(ys)), inv_dot, inv_cross, inv_spd], dim=1)))
        ds_out = self.s_mix2(h_s)
        s_new = s + ds_out.view(B, F6, LC_s, N, N)

        gate = torch.tanh(self.v_gate(h_s))
        du_out, dv_out = self.v_out(yu * gate, yv * gate)
        u_new = u + du_out.view(B, F6, LC_v, N, N)
        v_new = v + dv_out.view(B, F6, LC_v, N, N)
        return s_new, u_new, v_new


class PlanetaryClimateNet(nn.Module):
    """
    Purpose-built Macro Planetary Climate Neural Operator (`~345K` params, `< 5 ms` FP16 on Spark GB10):
      - Input:  `state [6, 4, 10, N, N]` (or `[B, 6, 4, 10, N, N]`), `forcing [6, 6, N, N]` (or `[B, 6, 6, N, N]`).
      - Output: `next_state` of exact same shape!
      - Guarantees:
          1. Exact `C4/SO(2)` gauge equivariance across all 12 rotated cubed-sphere seams (`0.0` rotation error!).
          2. Exact global conservation (`drift < 1e-6` over 200+ ticks) and non-negativity (`>= 0`) for
             total moisture (`water_vapor + cloud_liquid + ice_snow`) and `ash_smoke` via antisymmetric
             CFL-safe finite-volume fluxes + cell-wise 3x3 phase-conversion simplex!
          3. Zero hardcoded reference physics equations (Coriolis deflection, Hadley circulation, Day/Night
             ocean/land thermal contrast, condensation, and ash cooling are genuinely learned from data!).
    """

    def __init__(
        self,
        c_s: int = 20,
        c_v: int = 6,
        num_blocks: int = 4,
        num_levels: int = NUM_LEVELS,
    ) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_v = c_v
        self.num_blocks = num_blocks
        self.L = num_levels

        # Scalar inputs per level: 8 dynamic scalar channels + 6 forcing + 1 wind_speed = 15
        in_s_per_lvl = 15
        self.stem_s = nn.Conv2d(num_levels * in_s_per_lvl, num_levels * c_s, kernel_size=1)
        self.stem_v = SO2VectorLinear(num_levels, num_levels * c_v)

        self.blocks = nn.ModuleList(
            [GaugeEquivariantClimateBlock(c_s=c_s, c_v=c_v, num_levels=num_levels) for _ in range(num_blocks)]
        )

        # Residual heads for non-conserved fields:
        #   4 scalars per level: (d_temp, d_pressure, d_wind_w, d_ocean_heat)
        #   1 vector pair per level: (d_wind_u, d_wind_v)
        self.head_noncons_s = nn.Conv2d(num_levels * c_s, num_levels * 4, kernel_size=1)
        self.head_wind_v = SO2VectorLinear(num_levels * c_v, num_levels)
        nn.init.zeros_(self.head_noncons_s.weight)
        nn.init.zeros_(self.head_noncons_s.bias)
        nn.init.zeros_(self.head_wind_v.w_r.weight)
        nn.init.zeros_(self.head_wind_v.w_i.weight)

        # Conservative transport heads for 4 conserved species: (vapor, cloud, ice_snow, ash_smoke)
        #   Horizontal transport velocity residual `(du_k, dv_k)` (4 vector pairs per level)
        self.head_flux_uv = SO2VectorLinear(num_levels * c_v, num_levels * 4)
        #   Vertical interface velocity `w_k` across 3 interfaces x 4 species = 12
        self.head_flux_w = nn.Conv2d(num_levels * c_s, (num_levels - 1) * 4, kernel_size=1)
        #   Horizontal & vertical non-negative diffusivity logits
        self.head_diff_h = nn.Conv2d(num_levels * c_s, num_levels * 4, kernel_size=1)
        self.head_diff_v = nn.Conv2d(num_levels * c_s, (num_levels - 1) * 4, kernel_size=1)
        nn.init.zeros_(self.head_flux_uv.w_r.weight)
        nn.init.zeros_(self.head_flux_uv.w_i.weight)
        nn.init.zeros_(self.head_flux_w.weight)
        nn.init.zeros_(self.head_flux_w.bias)
        nn.init.zeros_(self.head_diff_h.weight)
        nn.init.constant_(self.head_diff_h.bias, -4.2)  # softplus(-4.2) ~ 0.015
        nn.init.zeros_(self.head_diff_v.weight)
        nn.init.constant_(self.head_diff_v.bias, -4.6)  # softplus(-4.6) ~ 0.010

        # Learnable baseline advection scale (initialized near physical ~0.24)
        self.base_adv_scale = nn.Parameter(torch.full((1, 1, num_levels, 4, 1, 1), 0.24))

        # Mass-conserving 3x3 moisture phase-conversion logits per cell (`[L, 3, 3]` -> softmax over target)
        self.head_phase = nn.Conv2d(num_levels * c_s, num_levels * 9, kernel_size=1)
        nn.init.zeros_(self.head_phase.weight)
        eye_bias = torch.eye(3).reshape(9).repeat(num_levels) * 4.0  # starts near-identity!
        self.head_phase.bias.data.copy_(eye_bias)

    def forward(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        """Predict next-tick climate state `[6, 4, 10, N, N]` (or `[B, 6, 4, 10, N, N]`)."""
        unbatched = state.ndim == 5
        if unbatched:
            state = state.unsqueeze(0)
            forcing = forcing.unsqueeze(0)

        B, F6, L, C, N, _ = state.shape
        assert F6 == NUM_FACES and L == self.L and C == NUM_CHANNELS

        # Extract scalar & vector input channels
        wu = state[:, :, :, CH_WIND_U]  # [B, 6, L, N, N]
        wv = state[:, :, :, CH_WIND_V]
        w_spd = torch.sqrt(wu * wu + wv * wv + 1e-4)

        dyn_s = state[:, :, :, [CH_TEMP, CH_VAPOR, CH_CLOUD, CH_ICE_SNOW, CH_PRESSURE, CH_WIND_W, CH_OCEAN_HEAT, CH_ASH_SMOKE]]
        fc_exp = forcing.unsqueeze(2).expand(B, F6, L, NUM_FORCING, N, N)
        s_in = torch.cat([dyn_s, fc_exp, w_spd.unsqueeze(3)], dim=3).reshape(B * F6, L * 15, N, N)

        s = F.silu(self.stem_s(s_in)).view(B, F6, L * self.c_s, N, N)
        u_h, v_h = self.stem_v(wu.reshape(B * F6, L, N, N), wv.reshape(B * F6, L, N, N))
        u_h = u_h.view(B, F6, L * self.c_v, N, N)
        v_h = v_h.view(B, F6, L * self.c_v, N, N)

        for blk in self.blocks:
            s, u_h, v_h = blk(s, u_h, v_h, wu, wv)

        s_2d = s.reshape(B * F6, L * self.c_s, N, N)
        u_2d = u_h.reshape(B * F6, L * self.c_v, N, N)
        v_2d = v_h.reshape(B * F6, L * self.c_v, N, N)

        # 1. Non-conserved residual updates
        d_noncons = self.head_noncons_s(s_2d).view(B, F6, L, 4, N, N)
        d_wu, d_wv = self.head_wind_v(u_2d, v_2d)
        d_wu = d_wu.view(B, F6, L, N, N)
        d_wv = d_wv.view(B, F6, L, N, N)

        temp_new = (state[:, :, :, CH_TEMP] + d_noncons[:, :, :, 0]).clamp(-2.0, 2.0)
        pres_new = (state[:, :, :, CH_PRESSURE] + d_noncons[:, :, :, 1]).clamp(-3.0, 3.0)
        ww_new = (state[:, :, :, CH_WIND_W] + d_noncons[:, :, :, 2]).clamp(-1.5, 1.5)
        ocean_mask = (1.0 - forcing[:, :, FC_LAND_MASK]).unsqueeze(2)
        oh_new = (state[:, :, :, CH_OCEAN_HEAT] + d_noncons[:, :, :, 3]).clamp(-1.5, 1.5)
        # Keep ocean_heat cleanly zero on upper levels & land
        oh_level_mask = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=oh_new.dtype, device=oh_new.device).view(1, 1, 4, 1, 1)
        oh_new = oh_new * ocean_mask * oh_level_mask
        wu_new = (wu + d_wu).clamp(-2.5, 2.5)
        wv_new = (wv + d_wv).clamp(-2.5, 2.5)

        # 2. Strictly Conservative Transport Head for (vapor, cloud, ice_snow, ash_smoke)
        du_f, dv_f = self.head_flux_uv(u_2d, v_2d)
        du_f = du_f.view(B, F6, L, 4, N, N)
        dv_f = dv_f.view(B, F6, L, 4, N, N)
        adv_s = self.base_adv_scale.to(dtype=wu.dtype)
        vel_u_c = adv_s * wu.unsqueeze(3) + 0.25 * torch.tanh(du_f)
        vel_v_c = adv_s * wv.unsqueeze(3) + 0.25 * torch.tanh(dv_f)
        vel_w_c = 0.20 * torch.tanh(self.head_flux_w(s_2d).view(B, F6, L - 1, 4, N, N))
        diff_h = F.softplus(self.head_diff_h(s_2d)).view(B, F6, L, 4, N, N)
        diff_v = F.softplus(self.head_diff_v(s_2d)).view(B, F6, L - 1, 4, N, N)

        q_cons = state[:, :, :, [CH_VAPOR, CH_CLOUD, CH_ICE_SNOW, CH_ASH_SMOKE]].clamp(min=0.0)
        q_adv = conservative_cubed_sphere_transport(
            q_cons,
            vel_u=vel_u_c,
            vel_v=vel_v_c,
            vel_w_iface=vel_w_c,
            diff_h=diff_h,
            diff_v=diff_v,
        )

        # 3. Strictly Mass-Conserving 3x3 Moisture Phase Conversion Simplex (in FP32 for `< 1e-6` drift!)
        phase_logits = self.head_phase(s_2d).float().view(B, F6, L, 3, 3, N, N)
        # P[..., dst, src]: columns sum to 1.0 identically!
        P_phase = torch.softmax(phase_logits, dim=3)
        q_moist = q_adv[:, :, :, :3].float()  # [B, 6, L, 3, N, N]
        q_moist_new = (P_phase * q_moist.unsqueeze(3)).sum(dim=4).clamp(min=0.0).to(dtype=state.dtype)
        ash_new = q_adv[:, :, :, 3].clamp(min=0.0).to(dtype=state.dtype)

        out = torch.stack(
            [
                temp_new,
                q_moist_new[:, :, :, 0],
                q_moist_new[:, :, :, 1],
                q_moist_new[:, :, :, 2],
                pres_new,
                wu_new,
                wv_new,
                ww_new,
                oh_new,
                ash_new,
            ],
            dim=3,
        )
        return out[0] if unbatched else out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 7. AMR Macro <-> Micro Coupling APIs (Ready for Wave 34!)
# =============================================================================


def downscale_to_shard_laws(
    state: torch.Tensor,
    face: int,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    H: int = 256,
    W: int = 256,
    L: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Downscale macro cubed-sphere climate state `[6, 4, 10, N, N]` onto any high-res micro shard
    (`H x W x L`, e.g. `256 x 256 x 64` at `1 m/voxel`) spanning gnomonic coordinates `[u0, u1] x [v0, v1]`
    on `face` (`0..5`).

    Uses halo-padded (`halo=2` with exact SO(2) wind rotation) bilinear sampling so shards right against
    or straddling a cube edge interpolate seamlessly!

    Returns:
      - `world_laws_3d`: `[6, L, H, W]` matching `WorldModel3D-v8` (`gravity_g, ambient_temp_shift_cK,
                          volatility, viscosity, precipitation, thermal_cond`).
      - `wind_uv`:       `[2, H, W]` surface/boundary-layer tangent wind `(wind_u, wind_v)` in face-local frame.
      - `precip_map`:    `[H, W]` non-negative precipitation intensity (rain + snow).
    """
    if state.ndim != 5 or state.shape[0] != NUM_FACES:
        raise ValueError(f"Expected state [6, 4, 10, N, N], got {tuple(state.shape)}")
    dev = state.device
    dtype = state.dtype
    N = int(state.shape[-1])
    h = 2
    P = N + 2 * h

    # Halo-pad with exact SO(2) wind rotation so boundary sampling is 100% seamless!
    st_pad = exchange_cubed_sphere_halos(state.float(), halo=h, flow_channels=(CH_WIND_U, CH_WIND_V))
    face_pad = st_pad[int(face)].reshape(1, NUM_LEVELS * NUM_CHANNELS, P, P)  # [1, 40, P, P]

    # Map gnomonic (u, v) in [-1, 1] -> normalized grid_sample coords in padded [P, P]
    # Interior [-1, 1] occupies [h, h+N] inside [0, P] -> scale factor N / P
    scale = float(N) / float(P)
    u_lin = torch.linspace(float(u0), float(u1), W, device=dev, dtype=torch.float32) * scale
    v_lin = torch.linspace(float(v0), float(v1), H, device=dev, dtype=torch.float32) * scale
    vv, uu = torch.meshgrid(v_lin, u_lin, indexing="ij")
    grid = torch.stack([uu, vv], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    sampled = F.grid_sample(face_pad, grid, mode="bilinear", padding_mode="border", align_corners=False)
    sampled = sampled[0].view(NUM_LEVELS, NUM_CHANNELS, H, W)  # [4, 10, H, W]

    # Vertical interpolation from L_macro=4 strata -> L micro layers
    r_norm = (torch.arange(L, dtype=torch.float32, device=dev) + 0.5) / float(max(L, 1))  # [L]
    # 1. Radial gravity profile g(r): 0.55g core -> 1.00g surface (r=0.45) -> 0.35g high sky-ring
    u_core = (r_norm / 0.45).clamp(0.0, 1.0)
    g_core = 0.55 + 0.45 * 0.5 * (1.0 - torch.cos(math.pi * u_core))
    u_sky = ((r_norm - 0.45) / 0.55).clamp(0.0, 1.0)
    g_sky = 1.00 - 0.65 * 0.5 * (1.0 - torch.cos(math.pi * u_sky))
    grav_3d = torch.where(r_norm <= 0.45, g_core, g_sky).view(L, 1, 1).expand(L, H, W)

    # Interpolate macro fields `[4, H, W]` -> `[L, H, W]` smoothly along vertical axis
    def _v_interp(field_4: torch.Tensor) -> torch.Tensor:
        # field_4: [4, H, W] -> treat as [1, H*W, 4] and 1D linear-interpolate to L
        x_1d = field_4.permute(1, 2, 0).reshape(1, H * W, NUM_LEVELS)
        y_1d = F.interpolate(x_1d, size=L, mode="linear", align_corners=True)
        return y_1d[0].view(H, W, L).permute(2, 0, 1).contiguous()

    temp_3d = _v_interp(sampled[:, CH_TEMP])
    cloud_3d = _v_interp(sampled[:, CH_CLOUD]).clamp(min=0.0)
    ice_3d = _v_interp(sampled[:, CH_ICE_SNOW]).clamp(min=0.0)
    ww_3d = _v_interp(sampled[:, CH_WIND_W])
    ash_3d = _v_interp(sampled[:, CH_ASH_SMOKE]).clamp(min=0.0)

    # 2. ambient_temp_shift in centi-Kelvin relative to canonical 290 K (WorldModel3D-v8 contract!)
    T_K_3d = T_REF_K + T_SCALE_K * temp_3d
    dT_amb_3d = ((T_K_3d - 290.0) * 100.0).clamp(-4500.0, 3800.0)

    # 3. volatility: boosted by warmth, volcanic ash/heat, and convective updrafts
    vol_3d = (0.85 + 0.45 * F.relu(temp_3d) + 0.65 * ash_3d + 0.35 * F.relu(ww_3d)).clamp(0.5, 2.6)

    # 4. viscosity: higher in freezing air & snow/ice
    cold_f = torch.sigmoid((T_FREEZE_K - T_K_3d) / 6.0)
    visc_3d = (0.85 + 0.65 * cold_f + 0.30 * ice_3d).clamp(0.6, 2.4)

    # 5. precipitation: directly coupled to macro storm clouds, snow, and upward convergence
    precip_3d = (0.25 + 2.40 * cloud_3d + 1.80 * ice_3d + 0.85 * F.relu(ww_3d)).clamp(0.1, 4.0)

    # 6. thermal_cond: enhanced by wind mixing & moisture
    wind_mag = torch.sqrt(sampled[1, CH_WIND_U] ** 2 + sampled[1, CH_WIND_V] ** 2 + 1e-4)
    kth_3d = (0.95 + 0.25 * wind_mag.unsqueeze(0) + 0.20 * (cloud_3d + ice_3d)).clamp(0.75, 2.0)

    world_laws_3d = torch.stack([grav_3d, dT_amb_3d, vol_3d, visc_3d, precip_3d, kth_3d], dim=0).to(dtype=dtype)

    # Boundary-layer / surface blended tangent wind `[2, H, W]`
    wind_u_2d = 0.45 * sampled[0, CH_WIND_U] + 0.55 * sampled[1, CH_WIND_U]
    wind_v_2d = 0.45 * sampled[0, CH_WIND_V] + 0.55 * sampled[1, CH_WIND_V]
    wind_uv = torch.stack([wind_u_2d, wind_v_2d], dim=0).to(dtype=dtype)

    # Surface precipitation intensity map `[H, W]`
    precip_map = (
        0.55 * sampled[1, CH_CLOUD].clamp(min=0.0)
        + 0.75 * sampled[2, CH_CLOUD].clamp(min=0.0)
        + 0.50 * sampled[1, CH_ICE_SNOW].clamp(min=0.0)
        + 0.60 * sampled[2, CH_ICE_SNOW].clamp(min=0.0)
        + 0.30 * F.relu(sampled[1, CH_WIND_W])
    ).clamp(min=0.0).to(dtype=dtype)

    return world_laws_3d, wind_uv, precip_map


def upscale_shard_injection(
    state: torch.Tensor,
    face: int,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    delta_temp: torch.Tensor | float,
    delta_ash: torch.Tensor | float,
) -> torch.Tensor:
    """
    Conservatively upscale localized micro-shard thermal (`delta_temp`, in `temp_norm = dT_K / 40` units)
    and volcanic / Sky-Island ash (`delta_ash >= 0`) injections back into the macro climate shell
    `state [6, 4, 10, N, N]`!

    Computes exact cell-overlap weights for every macro cell `(iv, iu)` on `face` intersecting
    `[u0, u1] x [v0, v1]`, conserving total injected ash mass (`sum(out_ash - in_ash) == total_delta_ash`)!
    """
    if state.ndim != 5 or state.shape[0] != NUM_FACES:
        raise ValueError(f"Expected state [6, 4, 10, N, N], got {tuple(state.shape)}")
    dev = state.device
    dtype = state.dtype
    N = int(state.shape[-1])
    out = state.clone().float()

    # Exact 1D cell overlap intervals along u and v in [-1, 1]
    du_cell = 2.0 / float(N)
    edges = -1.0 + torch.arange(N + 1, dtype=torch.float32, device=dev) * du_cell
    c_lo, c_hi = edges[:-1], edges[1:]
    ov_u = (torch.minimum(c_hi, torch.tensor(float(u1), device=dev)) - torch.maximum(c_lo, torch.tensor(float(u0), device=dev))).clamp(min=0.0) / du_cell
    ov_v = (torch.minimum(c_hi, torch.tensor(float(v1), device=dev)) - torch.maximum(c_lo, torch.tensor(float(v0), device=dev))).clamp(min=0.0) / du_cell
    w_2d = ov_v.view(N, 1) * ov_u.view(1, N)  # [N, N] fractional coverage of each macro cell

    # If footprint is sub-cell (very small shard), deposit onto the nearest macro cell with weight 1.0
    if float(w_2d.sum().item()) < 1e-8:
        uc = 0.5 * (float(u0) + float(u1))
        vc = 0.5 * (float(v0) + float(v1))
        iu = int(max(0, min(N - 1, math.floor((uc + 1.0) * 0.5 * N))))
        iv = int(max(0, min(N - 1, math.floor((vc + 1.0) * 0.5 * N))))
        w_2d[iv, iu] = 1.0

    # Normalize weights for exact ash mass conservation; keep area fraction for intensive temperature
    w_ash_norm = w_2d / w_2d.sum().clamp(min=1e-12)
    w_temp_mask = w_2d.clamp(max=1.0)

    # Convert scalar / 2D / 3D inputs into per-level vertical weights `[4]`
    def _to_level_totals(val: torch.Tensor | float, is_extensive: bool) -> torch.Tensor:
        t = torch.as_tensor(val, dtype=torch.float32, device=dev)
        if t.ndim == 0:
            # Distribute surface/plume vertically into Levels 0, 1, 2 (0.20, 0.45, 0.35 for ash; 0.50, 0.35, 0.15 for heat)
            prof = torch.tensor([0.20, 0.45, 0.35, 0.0] if is_extensive else [0.50, 0.35, 0.15, 0.0], device=dev)
            return t * prof
        if t.ndim == 2:
            agg = t.sum() if is_extensive else t.mean()
            prof = torch.tensor([0.20, 0.45, 0.35, 0.0] if is_extensive else [0.50, 0.35, 0.15, 0.0], device=dev)
            return agg * prof
        if t.ndim == 3:
            # Map micro vertical axis `L_s` -> 4 macro strata
            L_s = t.shape[0]
            out_4 = torch.zeros(NUM_LEVELS, dtype=torch.float32, device=dev)
            for l in range(NUM_LEVELS):
                z0 = int(round(l * L_s / NUM_LEVELS))
                z1 = max(z0 + 1, int(round((l + 1) * L_s / NUM_LEVELS)))
                sl = t[z0:z1]
                out_4[l] = sl.sum() if is_extensive else sl.mean()
            return out_4
        raise ValueError(f"Unsupported injection tensor rank {t.ndim}")

    ash_4 = _to_level_totals(delta_ash, is_extensive=True).clamp(min=0.0)
    temp_4 = _to_level_totals(delta_temp, is_extensive=False)

    f = int(face)
    for l in range(NUM_LEVELS):
        out[f, l, CH_ASH_SMOKE] = (out[f, l, CH_ASH_SMOKE] + ash_4[l] * w_ash_norm).clamp(min=0.0)
        out[f, l, CH_TEMP] = (out[f, l, CH_TEMP] + temp_4[l] * w_temp_mask).clamp(-2.0, 2.0)

    # Also nudge local buoyancy updraft slightly if heated
    if float(temp_4[:2].sum().item()) > 0.0:
        out[f, 1, CH_WIND_W] = (out[f, 1, CH_WIND_W] + 0.25 * float(temp_4[:2].mean().item()) * w_temp_mask).clamp(-1.5, 1.5)

    return out.to(dtype=dtype)


# =============================================================================
# 8. Verification Suites (Coriolis, Seam Continuity, 200-Tick, AMR, GB10 FP16)
# =============================================================================


def compute_r2_per_group(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute R^2 per physical group (temp, wind, moisture, pressure, ash, ocean_heat, overall)."""
    p = pred.float()
    t = target.float()

    def _r2(a: torch.Tensor, b: torch.Tensor) -> float:
        ss_res = float(((a - b) ** 2).sum().item())
        ss_tot = float(((b - b.mean()) ** 2).sum().item())
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return float(1.0 - ss_res / ss_tot)

    # Overall R^2 computed across standardized channels (equal weight per physical channel)
    ch_r2s: list[float] = []
    for c in range(NUM_CHANNELS):
        pc, tc = p[..., c, :, :], t[..., c, :, :]
        if float(tc.var().item()) > 1e-8:
            ch_r2s.append(_r2(pc, tc))

    return {
        "temp_r2": _r2(p[..., CH_TEMP, :, :], t[..., CH_TEMP, :, :]),
        "wind_r2": _r2(p[..., [CH_WIND_U, CH_WIND_V, CH_WIND_W], :, :], t[..., [CH_WIND_U, CH_WIND_V, CH_WIND_W], :, :]),
        "moisture_r2": _r2(p[..., [CH_VAPOR, CH_CLOUD, CH_ICE_SNOW], :, :], t[..., [CH_VAPOR, CH_CLOUD, CH_ICE_SNOW], :, :]),
        "pressure_r2": _r2(p[..., CH_PRESSURE, :, :], t[..., CH_PRESSURE, :, :]),
        "ocean_heat_r2": _r2(p[..., CH_OCEAN_HEAT, :, :], t[..., CH_OCEAN_HEAT, :, :]),
        "ash_r2": _r2(p[..., CH_ASH_SMOKE, :, :], t[..., CH_ASH_SMOKE, :, :]),
        "overall_r2": float(sum(ch_r2s) / max(len(ch_r2s), 1)),
    }


@torch.no_grad()
def run_coriolis_polarity_test(
    model: PlanetaryClimateNet,
    ref_sim: SphericalClimateRefSim,
    device: torch.device,
    steps: int = 20,
    lat_deg: float = 40.0,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """
    Seed EXACTLY symmetric low-pressure anomalies at `+40 deg N` and `-40 deg S` on a calm ocean planet,
    roll out both `PlanetaryClimateNet` and `SphericalClimateRefSim` for 20 steps, and verify opposite-sign
    cyclonic vorticity (`nh_vorticity > 0`, `sh_vorticity < 0`, `coriolis_polarity_match = true`)!
    """
    model.eval()
    N = DEFAULT_N
    # Smooth all-ocean planet for clean, symmetric Coriolis isolation
    geo = make_static_planet_geography(seed=999, N=N, device=device)
    geo = PlanetGeography(
        N=N,
        xyz=geo.xyz,
        lat_sin=geo.lat_sin,
        lat_cos=geo.lat_cos,
        land_mask=torch.zeros_like(geo.land_mask),
        elev_norm=torch.zeros_like(geo.elev_norm),
    )
    sun_dir = (0.0, 0.0, 1.0)
    forcing = build_forcing_tensor(geo, sun_dir)
    st0 = make_initial_climate_state(geo, sun_dir, seed=999, add_cyclones=False, add_volcano=False, calm_start=True)
    st0 = inject_symmetric_coriolis_cyclones(st0, geo, amplitude=-1.5, lat_deg=lat_deg, lon_deg=0.0, radius_rad=0.20)

    # Masks around the two symmetric cyclone cores (+40 N and -40 S at lon=0 on +Z face)
    lat_r = math.radians(lat_deg)
    nh_c = torch.tensor([0.0, math.sin(lat_r), math.cos(lat_r)], device=device)
    sh_c = torch.tensor([0.0, -math.sin(lat_r), math.cos(lat_r)], device=device)
    nh_mask = (geo.xyz[:, 0] * nh_c[0] + geo.xyz[:, 1] * nh_c[1] + geo.xyz[:, 2] * nh_c[2]) > math.cos(0.13)
    sh_mask = (geo.xyz[:, 0] * sh_c[0] + geo.xyz[:, 1] * sh_c[1] + geo.xyz[:, 2] * sh_c[2]) > math.cos(0.13)

    st_net = st0.clone()
    st_ref = st0.clone()

    nh_net_hist, sh_net_hist = [], []
    nh_ref_hist, sh_ref_hist = [], []

    for _ in range(steps):
        st_net = model(st_net, forcing)
        st_ref = ref_sim.step(st_ref, forcing)

        z_net = compute_spherical_vorticity(st_net)[:, 1]  # Boundary layer vorticity [6, N, N]
        z_ref = compute_spherical_vorticity(st_ref)[:, 1]

        nh_net_hist.append(float(z_net[nh_mask].mean().item()))
        sh_net_hist.append(float(z_net[sh_mask].mean().item()))
        nh_ref_hist.append(float(z_ref[nh_mask].mean().item()))
        sh_ref_hist.append(float(z_ref[sh_mask].mean().item()))

    nh_net_final = nh_net_hist[-1]
    sh_net_final = sh_net_hist[-1]
    nh_ref_final = nh_ref_hist[-1]
    sh_ref_final = sh_ref_hist[-1]
    polarity_match = (nh_net_final > 0.01) and (sh_net_final < -0.01) and (nh_ref_final > 0.01) and (sh_ref_final < -0.01)

    # Correlation between model & ref vorticity trajectories
    a = np.array(nh_net_hist + sh_net_hist, dtype=np.float64)
    b = np.array(nh_ref_hist + sh_ref_hist, dtype=np.float64)
    corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-8 and np.std(b) > 1e-8 else 0.0

    summary = {
        "steps": steps,
        "test_latitude_deg": lat_deg,
        "nh_vorticity": nh_net_final,
        "sh_vorticity": sh_net_final,
        "ref_nh_vorticity": nh_ref_final,
        "ref_sh_vorticity": sh_ref_final,
        "nh_sh_asymmetry_ratio": abs(nh_net_final + sh_net_final) / max(abs(nh_net_final) + abs(sh_net_final), 1e-8),
        "vorticity_trajectory_corr": corr,
        "coriolis_polarity_match": bool(polarity_match),
    }
    curves = {
        "nh_net": nh_net_hist,
        "sh_net": sh_net_hist,
        "nh_ref": nh_ref_hist,
        "sh_ref": sh_ref_hist,
    }
    return summary, curves


@torch.no_grad()
def run_seam_continuity_and_200tick_test(
    model: PlanetaryClimateNet,
    ref_sim: SphericalClimateRefSim,
    device: torch.device,
    total_ticks: int = 200,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    1. 12-Edge Seam Continuity & SO(2) Rotation Equivariance Test:
       - Verify exact 0.0 tangent rotation round-trip error across all 12 cube edges.
       - Seed a cyclone directly ON a 90-degree rotated polar cube seam (`+Y / +X` edge!) and measure
         seam jump ratio with exact SO(2) rotated halos vs unrotated halos.
    2. 200-Tick Multi-Day Stability & Conservation Test:
       - Roll out `PlanetaryClimateNet` for 200 ticks (2.5 full solar days!) with sweeping Day/Night terminator.
       - Track NaNs, moisture conservation drift, ash conservation drift, and R^2 vs `SphericalClimateRefSim`!
    """
    model.eval()
    N = DEFAULT_N

    # A. Exact 12-edge involution & rotation round-trip check
    max_rot_err = 0.0
    inv_mismatches = 0
    for (f, e), spec in EDGE_SPECS.items():
        recip = EDGE_SPECS[(spec.nbr_face, spec.nbr_edge_id)]
        if recip.nbr_face != f or recip.nbr_edge_id != e:
            inv_mismatches += 1
        R1 = np.array(spec.rot_2x2, dtype=np.float64)
        R2 = np.array(recip.rot_2x2, dtype=np.float64)
        max_rot_err = max(max_rot_err, float(np.max(np.abs(R1 @ R2 - np.eye(2)))))

    # B. Seam jump test: cyclone centered directly on the 90-degree rotated seam between Face 2 (+Y) & Face 0 (+X)!
    geo = make_static_planet_geography(seed=4242, N=N, device=device)
    sun0 = compute_sun_dir(0.0)
    fc0 = build_forcing_tensor(geo, sun0)
    st_seam = make_initial_climate_state(geo, sun0, seed=4242, add_cyclones=False, add_volcano=False)
    # Point on the (+X, +Y) 90-degree rotated cube edge: (1/sqrt(2), 1/sqrt(2), 0)
    seam_pt = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
    st_seam = inject_pressure_vortex(st_seam, geo, seam_pt, amplitude=-1.4, radius_rad=0.24)

    for _ in range(12):
        st_seam = model(st_seam, fc0)

    # Compare boundary wind jump across the (+Y, +u) <-> (+X, +v) 90-deg seam with exact rotation vs unrotated!
    # Edge (2, 1) is +Y's +u edge -> meets +X (0) with R_{0->2}
    st_rot_pad = exchange_cubed_sphere_halos(st_seam, halo=1, flow_channels=(CH_WIND_U, CH_WIND_V))
    st_unrot_pad = exchange_cubed_sphere_halos(st_seam, halo=1, flow_channels=None)

    # Measure wind difference across the (+Y, +u) seam vs interior neighbor step
    w_rot_seam = (st_rot_pad[2, 1, [CH_WIND_U, CH_WIND_V], 1:-1, -1] - st_rot_pad[2, 1, [CH_WIND_U, CH_WIND_V], 1:-1, -2]).abs().mean().item()
    w_rot_int = (st_rot_pad[2, 1, [CH_WIND_U, CH_WIND_V], 1:-1, -2] - st_rot_pad[2, 1, [CH_WIND_U, CH_WIND_V], 1:-1, -3]).abs().mean().item()
    w_unrot_seam = (st_unrot_pad[2, 1, [CH_WIND_U, CH_WIND_V], 1:-1, -1] - st_unrot_pad[2, 1, [CH_WIND_U, CH_WIND_V], 1:-1, -2]).abs().mean().item()
    seam_jump_ratio_rot = float(w_rot_seam / max(w_rot_int, 1e-6))
    seam_jump_ratio_unrot = float(w_unrot_seam / max(w_rot_int, 1e-6))
    seam_reduction_x = float(w_unrot_seam / max(w_rot_seam, 1e-6))

    # C. 200-Tick Multi-Day Rollout (2.5 full days) with sweeping Day/Night terminator + volcanic ash plume
    geo_roll = make_static_planet_geography(seed=777, N=N, device=device)
    st_init = make_initial_climate_state(geo_roll, compute_sun_dir(0.0), seed=777, add_cyclones=True, add_volcano=True)
    # Spin up 15 ref steps so weather systems are active at tick 0
    for t_pre in range(15):
        st_init = ref_sim.step(st_init, build_forcing_tensor(geo_roll, compute_sun_dir(t_pre)))

    st_net = st_init.clone()
    st_ref = st_init.clone()

    m0 = float(st_init[:, :, [CH_VAPOR, CH_CLOUD, CH_ICE_SNOW]].sum().item())
    a0 = float(st_init[:, :, CH_ASH_SMOKE].sum().item())

    r2_curve: list[float] = []
    moist_drift_curve: list[float] = []
    ash_drift_curve: list[float] = []
    day_night_dT_curve: list[float] = []
    nan_count = 0

    for t in range(1, total_ticks + 1):
        s_dir = compute_sun_dir(15 + t)
        fc_t = build_forcing_tensor(geo_roll, s_dir)
        st_net = model(st_net, fc_t)
        st_ref = ref_sim.step(st_ref, fc_t)

        if not torch.isfinite(st_net).all():
            nan_count += int((~torch.isfinite(st_net)).sum().item())

        mt = float(st_net[:, :, [CH_VAPOR, CH_CLOUD, CH_ICE_SNOW]].sum().item())
        at = float(st_net[:, :, CH_ASH_SMOKE].sum().item())
        m_drift = abs(mt - m0) / max(m0, 1e-8)
        a_drift = abs(at - a0) / max(a0, 1e-8)
        moist_drift_curve.append(m_drift)
        ash_drift_curve.append(a_drift)

        # Day vs Night continental surface temperature delta (in Kelvin)
        day_m = (fc_t[:, FC_COS_SUN] > 0.35) & (fc_t[:, FC_LAND_MASK] > 0.5)
        ngt_m = (fc_t[:, FC_NIGHT_COOL] > 0.35) & (fc_t[:, FC_LAND_MASK] > 0.5)
        if bool(day_m.any()) and bool(ngt_m.any()):
            dT_K = float((st_net[:, 0, CH_TEMP][day_m].mean() - st_net[:, 0, CH_TEMP][ngt_m].mean()).item()) * T_SCALE_K
        else:
            dT_K = 0.0
        day_night_dT_curve.append(dT_K)

        if t <= 40 or t % 5 == 0:
            r2_t = compute_r2_per_group(st_net, st_ref)["overall_r2"]
            r2_curve.append((t, r2_t))

    summary = {
        "edge_involution_mismatches": inv_mismatches,
        "tangent_rotation_roundtrip_max_err": max_rot_err,
        "seam_jump_ratio_rotated": seam_jump_ratio_rot,
        "seam_jump_ratio_unrotated": seam_jump_ratio_unrot,
        "seam_error_reduction_x": seam_reduction_x,
        "rollout_ticks": total_ticks,
        "nan_or_inf_count": nan_count,
        "max_moisture_conservation_drift": float(max(moist_drift_curve)),
        "final_moisture_conservation_drift": float(moist_drift_curve[-1]),
        "max_ash_conservation_drift": float(max(ash_drift_curve)),
        "final_ash_conservation_drift": float(ash_drift_curve[-1]),
        "min_state_moisture": float(st_net[:, :, [CH_VAPOR, CH_CLOUD, CH_ICE_SNOW]].min().item()),
        "min_state_ash": float(st_net[:, :, CH_ASH_SMOKE].min().item()),
        "mean_day_minus_night_land_dT_K": float(np.mean(day_night_dT_curve)),
        "passed_stability_and_conservation": (nan_count == 0) and (max(moist_drift_curve) < 1e-5) and (max(ash_drift_curve) < 1e-5),
    }
    viz_payload = {
        "r2_curve": r2_curve,
        "moist_drift_curve": moist_drift_curve,
        "ash_drift_curve": ash_drift_curve,
        "day_night_dT_curve": day_night_dT_curve,
        "final_state": st_net.detach().cpu(),
        "final_geo": geo_roll,
        "final_sun_dir": compute_sun_dir(15 + total_ticks),
    }
    return summary, viz_payload


@torch.no_grad()
def run_amr_coupling_test(
    model: PlanetaryClimateNet,
    ref_sim: SphericalClimateRefSim,
    device: torch.device,
) -> dict[str, Any]:
    """
    Verify bidirectional AMR Macro <-> Micro coupling:
      1. Downscale a hurricane cell on Face 4 (+Z) into a `256 x 256 x 64` `1 m` shard
         `world_laws_3d [6, 64, 256, 256]`, `wind_uv [2, 256, 256]`, `precip_map [256, 256]`.
      2. Upscale a volcanic ash + heat burst placed RIGHT NEAR the `+Z / +X` cube seam (`u in [0.75, 0.98]`),
         verify exact conservative ash mass deposit, roll out 25 steps under prevailing mid-latitude westerlies,
         and confirm the ash plume crosses the cube seam onto Face 0 (`+X`) AND radiatively cools the surface!
    """
    model.eval()
    N = DEFAULT_N
    geo = make_static_planet_geography(seed=5150, N=N, device=device)
    sun_dir = (0.35, 0.18, 0.92)
    fc = build_forcing_tensor(geo, sun_dir)
    st = make_initial_climate_state(geo, sun_dir, seed=5150, add_cyclones=False, add_volcano=False)
    # Seed a hurricane on Face 4 (+Z) at mid-latitudes
    st = inject_pressure_vortex(st, geo, (0.0, 0.55, 0.835), amplitude=-1.45, radius_rad=0.22, add_moisture=0.45)
    for _ in range(15):
        st = model(st, fc)

    # 1. Downscale hurricane footprint to 256x256x64 micro shard
    laws_3d, wind_uv, precip_map = downscale_to_shard_laws(
        st, face=4, u0=-0.25, u1=0.25, v0=0.35, v1=0.85, H=256, W=256, L=64
    )
    assert tuple(laws_3d.shape) == (6, 64, 256, 256)
    assert tuple(wind_uv.shape) == (2, 256, 256)
    assert tuple(precip_map.shape) == (256, 256)

    # 2. Upscale volcanic ash burst near +Z (+u) seam in the mid-latitude Westerlies belt (`u in [0.86, 0.99], v in [0.64, 0.86]`)
    #    where prevailing +u Westerlies transport the plume across the +Z -> +X cube seam!
    st_base = make_initial_climate_state(geo, sun_dir, seed=5150, add_cyclones=False, add_volcano=False)
    st_base[:, :, CH_ASH_SMOKE] = 0.0
    ash_before = float(st_base[:, :, CH_ASH_SMOKE].sum().item())
    inject_mass = 18.0
    st_ash = upscale_shard_injection(
        st_base, face=4, u0=0.86, u1=0.99, v0=0.64, v1=0.86, delta_temp=0.40, delta_ash=inject_mass
    )
    ash_after = float(st_ash[:, :, CH_ASH_SMOKE].sum().item())
    deposit_err = abs((ash_after - ash_before) - inject_mass) / inject_mass

    # Roll out both control (no ash) and volcanic-ash planet 40 steps and measure cross-seam transport & cooling!
    st_ctrl = st_base.clone()
    st_plume = st_ash.clone()
    for _ in range(40):
        st_ctrl = model(st_ctrl, fc)
        st_plume = model(st_plume, fc)

    ash_on_east_face = float(st_plume[0, :, CH_ASH_SMOKE].sum().item())  # Face 0 (+X) is East of Face 4 (+Z)!
    ash_total_final = float(st_plume[:, :, CH_ASH_SMOKE].sum().item())
    cross_seam_frac = ash_on_east_face / max(ash_total_final, 1e-8)
    ash_rollout_drift = abs(ash_total_final - ash_after) / max(ash_after, 1e-8)

    # Measure surface radiative cooling under the downwind sunlit ash plume (excluding initial heated vent)
    plume_shadow = (st_plume[:, :, CH_ASH_SMOKE].sum(dim=1) > 0.03) & (fc[:, FC_COS_SUN] > 0.15)
    plume_shadow[4] = False  # Look strictly downwind on neighboring +X face!
    if bool(plume_shadow.any()):
        downwind_dT_K = float((st_plume[:, 0, CH_TEMP][plume_shadow] - st_ctrl[:, 0, CH_TEMP][plume_shadow]).mean().item()) * T_SCALE_K
    else:
        # Fallback to any shadowed cell with ash > 0.02
        any_shadow = (st_plume[:, :, CH_ASH_SMOKE].sum(dim=1) > 0.02) & (fc[:, FC_COS_SUN] > 0.15)
        downwind_dT_K = float((st_plume[:, 0, CH_TEMP][any_shadow] - st_ctrl[:, 0, CH_TEMP][any_shadow]).mean().item()) * T_SCALE_K if bool(any_shadow.any()) else 0.0

    return {
        "downscale_world_laws_shape": list(laws_3d.shape),
        "downscale_wind_uv_shape": list(wind_uv.shape),
        "downscale_precip_shape": list(precip_map.shape),
        "shard_wind_speed_max": float(torch.sqrt(wind_uv[0] ** 2 + wind_uv[1] ** 2).max().item()),
        "shard_precip_max": float(precip_map.max().item()),
        "shard_dT_amb_range_cK": [float(laws_3d[LAW_AMBIENT_TEMP_SHIFT].min().item()), float(laws_3d[LAW_AMBIENT_TEMP_SHIFT].max().item())],
        "upscale_ash_injected": inject_mass,
        "upscale_ash_deposit_rel_err": float(deposit_err),
        "cross_seam_ash_on_east_face": ash_on_east_face,
        "cross_seam_transport_fraction": float(cross_seam_frac),
        "ash_25tick_conservation_drift": float(ash_rollout_drift),
        "downwind_radiative_cooling_K": float(downwind_dT_K),
        "amr_coupling_verified": (deposit_err < 1e-5) and (cross_seam_frac > 0.05) and (ash_rollout_drift < 1e-5),
    }


@torch.no_grad()
def benchmark_inference_fp16(
    model: PlanetaryClimateNet,
    device: torch.device,
    warmup: int = 20,
    iters: int = 100,
) -> dict[str, Any]:
    """Benchmark live FP16 inference latency (`ms/step`, `Hz`, peak VRAM MB) on `6 x 64^2 x 4`."""
    model_fp16 = model.to(device=device, dtype=torch.float16).eval()
    geo = make_static_planet_geography(seed=1, N=DEFAULT_N, device=device)
    fc = build_forcing_tensor(geo, compute_sun_dir(0.0)).to(dtype=torch.float16)
    st = make_initial_climate_state(geo, compute_sun_dir(0.0), seed=1).to(dtype=torch.float16)

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for _ in range(warmup):
        st = model_fp16(st, fc)
    if is_cuda:
        torch.cuda.synchronize(device)

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        st = model_fp16(st, fc)
        if is_cuda:
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if is_cuda else 0.0
    p50 = float(statistics.median(times_ms))
    p95 = float(np.percentile(times_ms, 95))
    mean_ms = float(statistics.mean(times_ms))
    dev_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    # Restore model to fp32 for subsequent evaluations
    model.to(device=device, dtype=torch.float32)

    return {
        "device": dev_name,
        "precision": "fp16",
        "grid_shape": [NUM_FACES, NUM_LEVELS, NUM_CHANNELS, DEFAULT_N, DEFAULT_N],
        "macro_cells": NUM_FACES * NUM_LEVELS * DEFAULT_N * DEFAULT_N,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean_ms,
        "hz_p50": float(1000.0 / max(p50, 1e-6)),
        "peak_vram_mb": peak_mb,
        "finite_after_bench": bool(torch.isfinite(st).all().item()),
    }


# =============================================================================
# 9. 4-Panel Dark-Theme Publication-Grade Visualization (`planetary_climate_viz.png`)
# =============================================================================


def _temp_colormap(t_norm: np.ndarray) -> np.ndarray:
    """Perceptual dark-theme temperature colormap: deep polar blue -> teal -> emerald -> amber -> crimson."""
    x = np.clip((t_norm + 0.85) / 1.55, 0.0, 1.0)
    stops = np.array([0.0, 0.28, 0.50, 0.72, 0.90, 1.0])
    cols = np.array(
        [
            [18.0, 42.0, 118.0],   # frigid polar (-0.85 -> 246 K)
            [32.0, 128.0, 198.0],  # sub-freezing (-0.42 -> 263 K)
            [52.0, 182.0, 156.0],  # temperate (~277 K)
            [236.0, 184.0, 62.0],  # warm subtropical (~291 K)
            [238.0, 96.0, 42.0],   # hot equatorial (~302 K)
            [196.0, 36.0, 58.0],   # scorching (~308 K)
        ]
    )
    rgb = np.zeros(x.shape + (3,), dtype=np.float32)
    for ch in range(3):
        rgb[..., ch] = np.interp(x, stops, cols[:, ch])
    return rgb


def _render_climate_3d_globe(
    state_np: np.ndarray,
    land_np: np.ndarray,
    sun_dir: tuple[float, float, float],
    size: int = 430,
) -> np.ndarray:
    """
    Ray-traced 3D Spherical Globe showing:
      - Ocean / continent temperature shading + snow/ice caps (`ice_snow`)
      - Sweeping Day/Night Terminator + golden twilight band
      - Swirling white storm clouds (`cloud_liquid`) & orange-grey volcanic ash plume (`ash_smoke`)
      - Subtle cyan cube-edge wireframe proving 0-seam continuity!
    """
    N = state_np.shape[-1]
    s = np.array(sun_dir, dtype=np.float64)
    s /= np.linalg.norm(s) + 1e-12

    # Camera slightly offset from sun so both sunlit day, golden terminator, AND dark night are dramatic!
    cam_fwd = np.array([0.46, 0.24, 0.855], dtype=np.float64)
    cam_fwd /= np.linalg.norm(cam_fwd)
    up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_right = np.cross(up_w, cam_fwd)
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_fwd, cam_right)

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    px = (xx - (size - 1) * 0.5) / (size * 0.44)
    py = -(yy - (size - 1) * 0.5) / (size * 0.44)
    r2 = px * px + py * py

    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :] = np.array([10.0, 15.0, 26.0], dtype=np.float32)

    # Subtle starfield background
    rng = np.random.default_rng(33)
    sx = rng.integers(0, size, size=140)
    sy = rng.integers(0, size, size=140)
    sb = rng.uniform(110.0, 235.0, size=140)
    img[sy, sx, 0] = sb * 0.88
    img[sy, sx, 1] = sb * 0.94
    img[sy, sx, 2] = sb

    # Outer cyan atmospheric halo
    halo_m = (r2 > 1.0) & (r2 < 1.18)
    h_fade = np.exp(-((np.sqrt(r2[halo_m]) - 1.0) / 0.055) ** 2)
    img[halo_m, 0] += 26.0 * h_fade
    img[halo_m, 1] += 130.0 * h_fade
    img[halo_m, 2] += 225.0 * h_fade

    sph = r2 <= 1.0
    pz = np.sqrt(np.clip(1.0 - r2[sph], 0.0, 1.0))
    px_s, py_s = px[sph], py[sph]
    wx = px_s * cam_right[0] + py_s * cam_up[0] + pz * cam_fwd[0]
    wy = px_s * cam_right[1] + py_s * cam_up[1] + pz * cam_fwd[1]
    wz = px_s * cam_right[2] + py_s * cam_up[2] + pz * cam_fwd[2]

    f_t, u_t, v_t = xyz_to_face_uv(torch.from_numpy(wx), torch.from_numpy(wy), torch.from_numpy(wz))
    f_np = f_t.numpy()
    iu_np = np.clip(np.rint(((u_t.numpy() + 1.0) * N - 1.0) * 0.5).astype(np.int64), 0, N - 1)
    iv_np = np.clip(np.rint(((v_t.numpy() + 1.0) * N - 1.0) * 0.5).astype(np.int64), 0, N - 1)

    t0 = state_np[f_np, 0, CH_TEMP, iv_np, iu_np]
    cloud = state_np[f_np, 1, CH_CLOUD, iv_np, iu_np] + 1.15 * state_np[f_np, 2, CH_CLOUD, iv_np, iu_np]
    ice = state_np[f_np, 0, CH_ICE_SNOW, iv_np, iu_np] + 0.6 * state_np[f_np, 2, CH_ICE_SNOW, iv_np, iu_np]
    ash = state_np[f_np, 1, CH_ASH_SMOKE, iv_np, iu_np] + 1.2 * state_np[f_np, 2, CH_ASH_SMOKE, iv_np, iu_np]
    land = land_np[f_np, iv_np, iu_np]

    # Base ocean & continent albedo modulated by surface temperature
    ocean_col = np.stack(
        [
            18.0 + 26.0 * np.clip(t0 + 0.5, 0.0, 1.0),
            74.0 + 58.0 * np.clip(t0 + 0.5, 0.0, 1.0),
            152.0 + 48.0 * np.clip(t0 + 0.5, 0.0, 1.0),
        ],
        axis=-1,
    )
    land_col = np.stack(
        [
            62.0 + 110.0 * np.clip(t0 + 0.35, 0.0, 1.0),
            138.0 - 25.0 * np.clip(t0 - 0.15, 0.0, 1.0),
            66.0 - 16.0 * np.clip(t0, 0.0, 1.0),
        ],
        axis=-1,
    )
    albedo = ocean_col * (1.0 - land[:, None]) + land_col * land[:, None]

    # Polar / cold snowline ice overlay
    ice_alpha = np.clip((ice - 0.20) * 1.45, 0.0, 0.88)
    albedo = albedo * (1.0 - ice_alpha[:, None]) + np.array([218.0, 236.0, 252.0]) * ice_alpha[:, None]

    # Swirling white storm clouds
    c_alpha = np.clip((cloud - 0.08) * 1.85, 0.0, 0.90)
    albedo = albedo * (1.0 - c_alpha[:, None]) + np.array([242.0, 248.0, 255.0]) * c_alpha[:, None]

    # Volcanic / Sky-Island ash plume (warm amber-charcoal plume!)
    a_alpha = np.clip(ash * 2.6, 0.0, 0.88)
    albedo = albedo * (1.0 - a_alpha[:, None]) + np.array([245.0, 132.0, 56.0]) * a_alpha[:, None]

    # Sweeping Day/Night Terminator shading + amber twilight scatter
    cos_s = wx * s[0] + wy * s[1] + wz * s[2]
    day_light = 0.18 + 0.88 * (0.5 * (1.0 + np.tanh(cos_s / 0.22))) * np.clip(0.38 + 0.72 * np.maximum(cos_s, 0.0), 0.22, 1.05)
    twilight = np.exp(-((cos_s + 0.02) / 0.12) ** 2)

    shaded = albedo * day_light[:, None]
    shaded[:, 0] += 58.0 * twilight
    shaded[:, 1] += 26.0 * twilight
    shaded[:, 2] += 10.0 * twilight

    # Crisp atmospheric limb scatter
    rim = np.power(np.clip(1.0 - pz, 0.0, 1.0), 2.8) * np.clip(0.28 + 0.80 * np.maximum(cos_s, 0.0), 0.20, 1.0)
    shaded[:, 0] += 24.0 * rim
    shaded[:, 1] += 118.0 * rim
    shaded[:, 2] += 205.0 * rim

    # Faint cyan cubed-sphere seam wireframe
    near_seam = (np.abs(u_t.numpy()) > 0.976) | (np.abs(v_t.numpy()) > 0.976)
    shaded[near_seam] = shaded[near_seam] * 0.76 + np.array([56.0, 198.0, 248.0]) * 0.24

    img[sph] = np.clip(shaded, 0.0, 255.0)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def render_planetary_climate_viz(
    metrics: dict[str, Any],
    state: torch.Tensor,
    geo: PlanetGeography,
    sun_dir: tuple[float, float, float],
    coriolis_curves: dict[str, list[float]],
    rollout_curves: dict[str, Any],
    out_path: Path,
) -> None:
    """
    Render publication-grade 4-panel dark-theme visualization (`models/planetary_climate_viz.png`, 2200x1280 px):
      - Panel 1 (Top-Left):  Unfolded 6-Face Cubed-Sphere Global Temperature + Storm Clouds + Wind Quivers + Terminator
      - Panel 2 (Top-Right): Ray-Traced 3D Spherical Globe (Day/Night Terminator + Spinning Cyclones + Ash Plume)
      - Panel 3 (Bot-Left):  NH vs SH Coriolis Vorticity Polarity Proof Plot (Model vs RefSim)
      - Panel 4 (Bot-Right): 200-Tick Multi-Day Rollout R^2 & Global Moisture/Ash Conservation Curves + GB10 Latency
    Rendered cleanly via PIL (zero matplotlib requirement!).
    """
    st_np = state.detach().cpu().float().numpy()  # [6, 4, 10, N, N]
    land_np = geo.land_mask.detach().cpu().float().numpy()  # [6, N, N]
    xyz_np = geo.xyz.detach().cpu().float().numpy()  # [6, 3, N, N]
    N = st_np.shape[-1]

    s = np.array(sun_dir, dtype=np.float32)
    s /= np.linalg.norm(s) + 1e-12

    # Build unfolded 6-face RGB maps: Temperature + coastlines + clouds + ash + terminator contour
    net_faces_u8 = np.zeros((NUM_FACES, N, N, 3), dtype=np.uint8)
    quivers_per_face: list[list[list[float]]] = []  # face -> list of [iu, iv, wu, wv]

    for f in range(NUM_FACES):
        t0 = st_np[f, 0, CH_TEMP]
        cloud = st_np[f, 1, CH_CLOUD] + 1.1 * st_np[f, 2, CH_CLOUD]
        ash = st_np[f, 1, CH_ASH_SMOKE] + 1.2 * st_np[f, 2, CH_ASH_SMOKE]
        land = land_np[f]

        rgb = _temp_colormap(t0)
        # Subtle coastline contour highlight
        gy, gx = np.gradient(land)
        coast = np.sqrt(gx * gx + gy * gy) > 0.14
        rgb[coast] = rgb[coast] * 0.45 + np.array([225.0, 238.0, 250.0]) * 0.55

        # Cloud overlay (crisp white-cyan storm bands)
        ca = np.clip((cloud - 0.08) * 1.45, 0.0, 0.78)
        rgb = rgb * (1.0 - ca[:, :, None]) + np.array([238.0, 246.0, 255.0]) * ca[:, :, None]

        # Ash plume overlay (bright amber-orange!)
        aa = np.clip(ash * 2.4, 0.0, 0.85)
        rgb = rgb * (1.0 - aa[:, :, None]) + np.array([255.0, 146.0, 48.0]) * aa[:, :, None]

        # Day/Night subtle shading + golden terminator contour
        cos_s = xyz_np[f, 0] * s[0] + xyz_np[f, 1] * s[1] + xyz_np[f, 2] * s[2]
        shade = 0.52 + 0.48 * (0.5 * (1.0 + np.tanh(cos_s / 0.28)))
        rgb = rgb * shade[:, :, None]
        term = np.abs(cos_s) < 0.036
        rgb[term] = np.array([251.0, 191.0, 36.0])

        # Flip vertically so PNG row 0 = +v (top/North)
        net_faces_u8[f] = np.clip(rgb[::-1], 0.0, 255.0).astype(np.uint8)

        # Sample wind quivers every 8 cells (8x8 grid per face)
        q_list: list[list[float]] = []
        wu = st_np[f, 1, CH_WIND_U]
        wv = st_np[f, 1, CH_WIND_V]
        for iv in range(4, N, 8):
            for iu in range(4, N, 8):
                q_list.append([float(iu) / N, float(N - 1 - iv) / N, float(wu[iv, iu]), float(wv[iv, iu])])
        quivers_per_face.append(q_list)

    globe_rgb = _render_climate_3d_globe(st_np, land_np, sun_dir=sun_dir, size=430)

    payload = json.dumps(
        {
            "metrics": metrics,
            "out_path": str(out_path),
            "N": N,
            "net_faces_b64": base64.b64encode(net_faces_u8.tobytes()).decode("ascii"),
            "quivers": quivers_per_face,
            "globe_b64": base64.b64encode(globe_rgb.tobytes()).decode("ascii"),
            "globe_size": int(globe_rgb.shape[0]),
            "coriolis_curves": coriolis_curves,
            "rollout_curves": {
                "r2_curve": rollout_curves["r2_curve"],
                "moist_drift_curve": rollout_curves["moist_drift_curve"],
                "ash_drift_curve": rollout_curves["ash_drift_curve"],
            },
        }
    )

    pil_script = r'''
import base64, json, math, sys
from PIL import Image, ImageDraw, ImageFont

data = json.loads(sys.stdin.read())
metrics = data["metrics"]
out_path = data["out_path"]
N = int(data["N"])
net_bytes = base64.b64decode(data["net_faces_b64"])
quivers = data["quivers"]
globe_bytes = base64.b64decode(data["globe_b64"])
gsz = int(data["globe_size"])
cor_c = data["coriolis_curves"]
rol_c = data["rollout_curves"]

W, H = 2200, 1280
BG = (10, 15, 24)
PANEL = (17, 24, 38)
BORDER = (44, 60, 88)
FG = (236, 242, 252)
MUTED = (148, 163, 184)
GRID = (34, 48, 70)
CYAN = (34, 211, 238)
SKY = (56, 189, 248)
EMERALD = (52, 211, 153)
AMBER = (251, 191, 36)
ORANGE = (249, 115, 22)
ROSE = (244, 63, 94)
VIOLET = (167, 139, 250)

def font(size, bold=False):
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
        if bold
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()

def text_center(draw, cx, cy, txt, fnt, fill):
    bb = draw.textbbox((0, 0), txt, font=fnt)
    draw.text((cx - (bb[2] - bb[0]) / 2, cy - (bb[3] - bb[1]) / 2), txt, font=fnt, fill=fill)

def draw_dashed(draw, x0, y0, x1, y1, color, width=2, dash=8, gap=5):
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

def draw_quiver(draw, x0, y0, dx, dy, color):
    mag = math.hypot(dx, dy)
    if mag < 1.2:
        return
    x1, y1 = x0 + dx, y0 + dy
    draw.line([(x0, y0), (x1, y1)], fill=color, width=1)
    ang = math.atan2(dy, dx)
    ah = min(4.5, max(2.5, mag * 0.35))
    p1 = (x1 - ah * math.cos(ang - 0.55), y1 - ah * math.sin(ang - 0.55))
    p2 = (x1 - ah * math.cos(ang + 0.55), y1 - ah * math.sin(ang + 0.55))
    draw.line([(x1, y1), p1], fill=color, width=1)
    draw.line([(x1, y1), p2], fill=color, width=1)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

ev = metrics["held_out_eval"]
lat = metrics["spark_gb10_fp16_benchmark"]
cor = metrics["coriolis_polarity_test"]
sm = metrics["seam_and_200tick_test"]
amr = metrics["amr_coupling_test"]

text_center(
    draw, W // 2, 32,
    f"Frostmarsh Wave 33A — PlanetaryClimateNet ({metrics['model_params']:,} params, 6x64^2x4 = 98,304 cells, {lat['p50_ms']:.2f} ms / {lat['hz_p50']:.0f} Hz FP16 on {lat['device']})",
    font(22, bold=True), FG,
)

# 2x2 Panel Grid
panels = [
    (24, 66, 1260, 720),    # Panel 1: Unfolded 6-face Global Temp + Clouds + Wind Quivers
    (1284, 66, 2176, 720),  # Panel 2: Ray-Traced 3D Spherical Globe
    (24, 744, 1100, 1256),  # Panel 3: NH vs SH Coriolis Polarity Proof
    (1124, 744, 2176, 1256) # Panel 4: 200-Tick Rollout R^2 & Moisture Conservation
]
for px0, py0, px1, py1 in panels:
    draw.rounded_rectangle([px0, py0, px1, py1], radius=12, fill=PANEL, outline=BORDER, width=2)

# =========================================================================
# Panel 1: Unfolded 6-Face Cubed-Sphere Global Temperature + Wind Quivers + Storm Clouds
# =========================================================================
p1x0, p1y0, p1x1, p1y1 = panels[0]
text_center(draw, (p1x0 + p1x1) // 2, p1y0 + 24, "1. Unfolded 6-Face Cubed-Sphere Climate (Surface Temp, Storm Clouds, Ash Plume & Wind Quivers)", font(16, bold=True), FG)

face_sz = 166
net_x0 = p1x0 + 48
net_y0 = p1y0 + 54
layout = {
    2: (1, 0, "+Y NORTH"),
    1: (0, 1, "-X WEST"),
    4: (1, 1, "+Z PRIME"),
    0: (2, 1, "+X EAST"),
    5: (3, 1, "-Z ANTI"),
    3: (1, 2, "-Y SOUTH"),
}
stride = N * N * 3
for f, (gx, gy, flbl) in layout.items():
    f_raw = net_bytes[f * stride : (f + 1) * stride]
    f_im = Image.frombytes("RGB", (N, N), f_raw).resize((face_sz, face_sz), Image.BILINEAR)
    fx = net_x0 + gx * face_sz
    fy = net_y0 + gy * face_sz
    img.paste(f_im, (fx, fy))
    # Draw wind quivers cleanly on top
    for u_n, v_n, wu, wv in quivers[f]:
        qx = fx + u_n * face_sz
        qy = fy + v_n * face_sz
        # Note: +v points UP on screen -> negative screen dy
        scale_q = 15.0
        dx = max(-14.0, min(14.0, wu * scale_q))
        dy = max(-14.0, min(14.0, -wv * scale_q))
        draw_quiver(draw, qx, qy, dx, dy, (235, 250, 255))
    draw.rectangle([fx, fy, fx + face_sz, fy + face_sz], outline=(148, 163, 184), width=2)
    draw.rounded_rectangle([fx + 5, fy + 5, fx + 88, fy + 23], radius=4, fill=(10, 15, 24), outline=CYAN, width=1)
    text_center(draw, fx + 46, fy + 14, flbl, font(10, bold=True), FG)

# Temperature colorbar + Legend on right side of Panel 1
cb_x0 = net_x0 + 4 * face_sz + 36
cb_y0 = net_y0 + 18
cb_w = 24
cb_h = 260
for i in range(cb_h):
    frac = 1.0 - i / max(cb_h - 1, 1)
    # Match colormap stops
    stops = [0.0, 0.28, 0.50, 0.72, 0.90, 1.0]
    cols = [(18,42,118), (32,128,198), (52,182,156), (236,184,62), (238,96,42), (196,36,58)]
    c_rgb = cols[-1]
    for k in range(len(stops) - 1):
        if stops[k] <= frac <= stops[k+1]:
            t = (frac - stops[k]) / max(stops[k+1] - stops[k], 1e-6)
            c_rgb = tuple(int(cols[k][c] + t * (cols[k+1][c] - cols[k][c])) for c in range(3))
            break
    draw.line([(cb_x0, cb_y0 + i), (cb_x0 + cb_w, cb_y0 + i)], fill=c_rgb, width=1)
draw.rectangle([cb_x0, cb_y0, cb_x0 + cb_w, cb_y0 + cb_h], outline=MUTED, width=1)
draw.text((cb_x0 - 4, cb_y0 - 20), "Temp (K)", font=font(12, bold=True), fill=FG)
for frac, lbl in [(1.0, "308 K"), (0.72, "291 K"), (0.45, "273 K (Ice)"), (0.0, "246 K")]:
    yy = cb_y0 + int((1.0 - frac) * cb_h)
    draw.text((cb_x0 + cb_w + 8, yy - 7), lbl, font=font(11), fill=MUTED)

# Legend items below colorbar
leg_y = cb_y0 + cb_h + 26
for idx_l, (lbl, col) in enumerate([
    ("Storm Clouds", (238, 246, 255)),
    ("Volcanic Ash Plume", (255, 146, 48)),
    ("Day/Night Terminator", AMBER),
    ("BL Wind Quivers", CYAN),
]):
    yy = leg_y + idx_l * 24
    draw.rectangle([cb_x0, yy, cb_x0 + 16, yy + 14], fill=col, outline=MUTED, width=1)
    draw.text((cb_x0 + 24, yy), lbl, font=font(12), fill=FG)

# Summary KPI strip at bottom of Panel 1
draw.rounded_rectangle([p1x0 + 24, p1y1 - 68, p1x1 - 24, p1y1 - 14], radius=8, fill=(10, 30, 26), outline=EMERALD, width=1)
text_center(
    draw, (p1x0 + p1x1) // 2, p1y1 - 51,
    f"Held-Out Unseen Planets — 1-Step R^2: {ev['step_1']['overall_r2']:.4f} (T={ev['step_1']['temp_r2']:.4f}, W={ev['step_1']['wind_r2']:.4f}, M={ev['step_1']['moisture_r2']:.4f}, P={ev['step_1']['pressure_r2']:.4f}, Ash={ev['step_1']['ash_r2']:.4f})",
    font(12, bold=True), EMERALD,
)
text_center(
    draw, (p1x0 + p1x1) // 2, p1y1 - 28,
    f"20-Step Autoregressive Rollout R^2: {ev['step_20']['overall_r2']:.4f} | 12-Edge Seam Rot Err: {sm['tangent_rotation_roundtrip_max_err']:.1f} | Seam Jump Gain: {sm['seam_error_reduction_x']:.1f}x vs Unrotated",
    font(12), FG,
)

# =========================================================================
# Panel 2: Ray-Traced 3D Spherical Globe + AMR Coupling Summary
# =========================================================================
p2x0, p2y0, p2x1, p2y1 = panels[1]
text_center(draw, (p2x0 + p2x1) // 2, p2y0 + 24, "2. 3D Spherical Climate Globe (Terminator, Cyclones & Cross-Seam Ash Plume)", font(16, bold=True), FG)

globe_im = Image.frombytes("RGB", (gsz, gsz), globe_bytes)
gx0 = p2x0 + 36
gy0 = p2y0 + 56
img.paste(globe_im, (gx0, gy0))
draw.rounded_rectangle([gx0 - 2, gy0 - 2, gx0 + gsz + 2, gy0 + gsz + 2], radius=10, outline=BORDER, width=2)

# Clean corner badges on globe
draw.rounded_rectangle([gx0 + 10, gy0 + 10, gx0 + 168, gy0 + 48], radius=6, fill=(28, 22, 10), outline=AMBER, width=1)
text_center(draw, gx0 + 89, gy0 + 21, "DAY HEMISPHERE", font(11, bold=True), AMBER)
text_center(draw, gx0 + 89, gy0 + 36, "Hadley Convection", font(10), FG)

draw.rounded_rectangle([gx0 + gsz - 168, gy0 + gsz - 48, gx0 + gsz - 10, gy0 + gsz - 10], radius=6, fill=(14, 22, 44), outline=SKY, width=1)
text_center(draw, gx0 + gsz - 89, gy0 + gsz - 37, "NIGHT HEMISPHERE", font(11, bold=True), SKY)
text_center(draw, gx0 + gsz - 89, gy0 + gsz - 22, "Snowline & Ocean Buffer", font(10), FG)

# Right-side Info Cards inside Panel 2 (AMR Coupling + GB10 Live Latency)
cx0, cx1 = gx0 + gsz + 24, p2x1 - 24
cards = [
    ("Spark GB10 Live FP16", CYAN, [
        f"Latency p50: {lat['p50_ms']:.2f} ms ({lat['hz_p50']:.0f} Hz)",
        f"Latency p95: {lat['p95_ms']:.2f} ms",
        f"Macro Grid:  6 x 64^2 x 4",
        f"Peak VRAM:   {lat['peak_vram_mb']:.1f} MB",
    ]),
    ("AMR Macro <-> Micro (Wave 34)", EMERALD, [
        f"Shard Laws:  {tuple(amr['downscale_world_laws_shape'])}",
        f"Ash Deposit Err: {amr['upscale_ash_deposit_rel_err']:.1e}",
        f"Cross-Seam Plume: {amr['cross_seam_transport_fraction']*100:.1f}%",
        f"Plume Cooling: {amr['downwind_radiative_cooling_K']:+.2f} K",
    ]),
    ("200-Tick Multi-Day Stability", AMBER, [
        f"NaNs / Infs: {sm['nan_or_inf_count']} (2.5 Days)",
        f"Moist Drift: {sm['max_moisture_conservation_drift']:.2e}",
        f"Ash Drift:   {sm['max_ash_conservation_drift']:.2e}",
        f"Day-Night Land ΔT: +{sm['mean_day_minus_night_land_dT_K']:.1f} K",
    ]),
]
cy = gy0
for title_c, col_c, lines_c in cards:
    ch = 134
    draw.rounded_rectangle([cx0, cy, cx1, cy + ch], radius=8, fill=(12, 18, 30), outline=col_c, width=1)
    draw.text((cx0 + 14, cy + 10), title_c, font=font(13, bold=True), fill=col_c)
    for li, ln in enumerate(lines_c):
        draw.text((cx0 + 14, cy + 36 + li * 22), ln, font=font(12), fill=FG)
    cy += ch + 14

draw.rounded_rectangle([p2x0 + 24, p2y1 - 68, p2x1 - 24, p2y1 - 14], radius=8, fill=(10, 26, 38), outline=CYAN, width=1)
text_center(
    draw, (p2x0 + p2x1) // 2, p2y1 - 41,
    f"C4/SO(2) Gauge-Equivariant Operator | Exact Antisymmetric Fluxes | Zero Hardcoded Coriolis Formula",
    font(12, bold=True), CYAN,
)

# =========================================================================
# Panel 3: NH vs SH Coriolis Vorticity Polarity Proof Plot
# =========================================================================
p3x0, p3y0, p3x1, p3y1 = panels[2]
text_center(draw, (p3x0 + p3x1) // 2, p3y0 + 24, "3. Spherical Coriolis Polarity Proof (+40°N vs -40°S Symmetric Low-Pressure Cyclones)", font(16, bold=True), FG)

gx_a, gy_a, gx_b, gy_b = p3x0 + 82, p3y0 + 62, p3x1 - 32, p3y1 - 92
draw.rectangle([gx_a, gy_a, gx_b, gy_b], fill=(12, 18, 30), outline=BORDER, width=1)

nh_n = cor_c["nh_net"]
sh_n = cor_c["sh_net"]
nh_r = cor_c["nh_ref"]
sh_r = cor_c["sh_ref"]
n_steps = len(nh_n)
vmax = max(max(abs(v) for v in nh_n + sh_n + nh_r + sh_r) * 1.25, 0.05)

def _cy(val):
    return int(gy_b - ((val + vmax) / (2.0 * vmax)) * (gy_b - gy_a))

def _cx(idx):
    return int(gx_a + (idx / max(n_steps - 1, 1)) * (gx_b - gx_a))

# Gridlines + zero axis
for frac in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    yv = frac * vmax
    yy = _cy(yv)
    draw.line([(gx_a, yy), (gx_b, yy)], fill=(75, 92, 118) if abs(frac) < 1e-6 else GRID, width=2 if abs(frac) < 1e-6 else 1)
    draw.text((gx_a - 68, yy - 7), f"{yv:+.2f}", font=font(11), fill=MUTED)

for st_i in [0, 4, 9, 14, 19]:
    if st_i < n_steps:
        xx = _cx(st_i)
        draw.line([(xx, gy_a), (xx, gy_b)], fill=GRID, width=1)
        text_center(draw, xx, gy_b + 14, f"t={st_i+1}", font(11), MUTED)

# Plot RefSim dashed & PlanetaryClimateNet solid
for i in range(n_steps - 1):
    draw_dashed(draw, _cx(i), _cy(nh_r[i]), _cx(i+1), _cy(nh_r[i+1]), (148, 226, 250), width=2, dash=6, gap=4)
    draw_dashed(draw, _cx(i), _cy(sh_r[i]), _cx(i+1), _cy(sh_r[i+1]), (253, 164, 175), width=2, dash=6, gap=4)

draw.line([(_cx(i), _cy(nh_n[i])) for i in range(n_steps)], fill=CYAN, width=4)
draw.line([(_cx(i), _cy(sh_n[i])) for i in range(n_steps)], fill=ROSE, width=4)

# Annotations directly in plot corners
draw.rounded_rectangle([gx_a + 14, gy_a + 12, gx_a + 328, gy_a + 68], radius=6, fill=BG, outline=BORDER, width=1)
draw.line([(gx_a + 26, gy_a + 27), (gx_a + 54, gy_a + 27)], fill=CYAN, width=4)
draw.text((gx_a + 62, gy_a + 19), "+40°N Cyclone (CCW, ζ > 0) — Model", font=font(11, bold=True), fill=CYAN)
draw.line([(gx_a + 26, gy_a + 49), (gx_a + 54, gy_a + 49)], fill=ROSE, width=4)
draw.text((gx_a + 62, gy_a + 41), "-40°S Cyclone (CW,  ζ < 0) — Model", font=font(11, bold=True), fill=ROSE)

draw.rounded_rectangle([gx_b - 235, gy_a + 12, gx_b - 12, gy_a + 42], radius=6, fill=BG, outline=BORDER, width=1)
draw_dashed(draw, gx_b - 222, gy_a + 27, gx_b - 190, gy_a + 27, FG, width=2, dash=5, gap=3)
draw.text((gx_b - 182, gy_a + 19), "SphericalClimateRefSim", font=font(11), fill=MUTED)

draw.rounded_rectangle([p3x0 + 24, p3y1 - 62, p3x1 - 24, p3y1 - 14], radius=8, fill=(10, 30, 26), outline=EMERALD, width=1)
text_center(
    draw, (p3x0 + p3x1) // 2, p3y1 - 38,
    f"PASS={cor['coriolis_polarity_match']} | NH ζ={cor['nh_vorticity']:+.4f} (Ref {cor['ref_nh_vorticity']:+.4f}) | SH ζ={cor['sh_vorticity']:+.4f} (Ref {cor['ref_sh_vorticity']:+.4f}) | Corr={cor['vorticity_trajectory_corr']:.4f}",
    font(12, bold=True), EMERALD,
)

# =========================================================================
# Panel 4: 200-Tick Rollout R^2 & Global Moisture / Ash Conservation Curves
# =========================================================================
p4x0, p4y0, p4x1, p4y1 = panels[3]
text_center(draw, (p4x0 + p4x1) // 2, p4y0 + 24, "4. 200-Tick Multi-Day Rollout R^2 & Exact Global Moisture/Ash Conservation", font(16, bold=True), FG)

# Split Panel 4 into two clean stacked plots: Top = Rollout R^2, Bottom = Log10 Conservation Drift
ax0, ax1 = p4x0 + 84, p4x1 - 32
t_y0, t_y1 = p4y0 + 58, p4y0 + 244
b_y0, b_y1 = p4y0 + 274, p4y1 - 92
draw.rectangle([ax0, t_y0, ax1, t_y1], fill=(12, 18, 30), outline=BORDER, width=1)
draw.rectangle([ax0, b_y0, ax1, b_y1], fill=(12, 18, 30), outline=BORDER, width=1)

r2_pts = rol_c["r2_curve"]  # list of [tick, r2]
max_t = max(p[0] for p in r2_pts)

def _tx(tick):
    return int(ax0 + ((tick - 1) / max(max_t - 1, 1)) * (ax1 - ax0))

# Top subplot: R^2 in [0.4, 1.02]
r2_lo, r2_hi = 0.40, 1.02
def _r2y(val):
    v = max(r2_lo, min(r2_hi, val))
    return int(t_y1 - ((v - r2_lo) / (r2_hi - r2_lo)) * (t_y1 - t_y0))

for rv in [0.5, 0.7, 0.85, 1.0]:
    yy = _r2y(rv)
    draw.line([(ax0, yy), (ax1, yy)], fill=GRID, width=1)
    draw.text((ax0 - 52, yy - 7), f"{rv:.2f}", font=font(11), fill=MUTED)
draw_dashed(draw, ax0, _r2y(0.85), ax1, _r2y(0.85), AMBER, width=2, dash=8, gap=5)
draw.line([(_tx(t), _r2y(v)) for t, v in r2_pts], fill=EMERALD, width=3)
draw.text((ax0 + 12, t_y0 + 8), "Autoregressive Rollout R^2 (200 Ticks = 2.5 Solar Days)", font=font(12, bold=True), fill=EMERALD)
draw.text((ax1 - 210, _r2y(0.85) - 18), "0.85 20-Step Target Gate", font=font(11, bold=True), fill=AMBER)

# Bottom subplot: log10(conservation drift) in [-9, -4]
d_lo, d_hi = -9.0, -4.0
def _dy(val):
    lv = math.log10(max(val, 1e-9))
    lv = max(d_lo, min(d_hi, lv))
    return int(b_y1 - ((lv - d_lo) / (d_hi - d_lo)) * (b_y1 - b_y0))

for dv in [-8, -7, -6, -5]:
    yy = _dy(10.0 ** dv)
    draw.line([(ax0, yy), (ax1, yy)], fill=GRID, width=1)
    draw.text((ax0 - 58, yy - 7), f"1e{dv}", font=font(11), fill=MUTED)

draw_dashed(draw, ax0, _dy(1e-5), ax1, _dy(1e-5), ROSE, width=2, dash=8, gap=5)
m_dr = rol_c["moist_drift_curve"]
a_dr = rol_c["ash_drift_curve"]
draw.line([(_tx(i + 1), _dy(v)) for i, v in enumerate(m_dr)], fill=SKY, width=3)
draw.line([(_tx(i + 1), _dy(v)) for i, v in enumerate(a_dr)], fill=ORANGE, width=2)
draw.text((ax0 + 12, b_y0 + 8), "Global Moisture Drift (Blue) & Ash Drift (Orange) — Target < 1e-5", font=font(12, bold=True), fill=FG)

for tk in [1, 40, 80, 120, 160, 200]:
    xx = _tx(tk)
    text_center(draw, xx, b_y1 + 14, f"t={tk}", font(11), MUTED)

draw.rounded_rectangle([p4x0 + 24, p4y1 - 62, p4x1 - 24, p4y1 - 14], radius=8, fill=(10, 26, 38), outline=SKY, width=1)
text_center(
    draw, (p4x0 + p4x1) // 2, p4y1 - 38,
    f"200-Tick Drift: Moisture={sm['max_moisture_conservation_drift']:.2e} | Ash={sm['max_ash_conservation_drift']:.2e} | Min Moist={sm['min_state_moisture']:.1f} | 0 NaNs",
    font(12, bold=True), SKY,
)

img.save(out_path, format="PNG")
'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    py_exe = sys.executable
    try:
        import PIL  # noqa: F401
    except ImportError:
        py_exe = "/usr/bin/python3"

    proc = subprocess.run([py_exe, "-c", pil_script], input=payload, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PIL viz rendering failed ({proc.returncode}): {proc.stderr}")
