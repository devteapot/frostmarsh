"""
Frostmarsh Wave 29A (GPU/ML — 6-Face Cubed-Sphere Planetary Simulation):
  - Exact 6-face cubed-sphere orthonormal tangent frames `(n_f, e_u, e_v)` with `det == +1`
    and outward radial `L` axis (`-r` = local downward gravity matching `WorldModel3D-v8` with 0 retraining).
  - Gnomonic sphere projection `face_uv_to_xyz` & exact inverse `xyz_to_face_uv` (`< 1e-6` round-trip).
  - Basis-derived 12-edge `HALO_XY = 6` exchange + 8-corner `6x6` spherical lookup (`exchange_cubed_sphere_halos`)
    with exact 2x2 tangent-vector rotation `R_{nbr -> cur}` on directional flow channels (`ch 12=flow_e, ch 13=flow_s`
    per `FIELD_CONTRACT.md` / `encode_state`), zero hand-coded edge tables, exact involution & vector norm preservation.
  - Seamless 3D procedural Cubed-Sphere Planet Generator (`make_cubed_sphere_planet`) evaluated purely in 3D unit-sphere
    coordinates (continents, curved oceans, mountains, equatorial volcanic rifts, polar ice caps, abyssal mantle caverns,
    and tilted orbital floating Sky-Island Ring — 100% continuous across all 12 edges & 8 corners).
  - 3D Planetary World-Law Field (`make_planetary_world_laws`) with radial gravity `g(r)` (`0.55g` core -> `1.0g` surface
    -> `0.35g` sky-ring), sweeping Solar Day/Night Terminator `dT_amb` (`+38 K` subsolar -> `-45 K` night & poles),
    and latitude-coupled precipitation & equatorial volcanic volatility.
  - Live NVIDIA GB10 FP16 multi-scale planetary benchmarks (393K / 3.15M / 6.29M voxels, Dense vs Unified 6-Face
    Sparse `16x16x8` Brick Pool with `max_bricks_per_batch=192`), 20-tick rotating Day/Night terminator rollout,
    and dark-theme 3-panel visualization (`models/planet_cubed_sphere_viz.png`).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import statistics
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
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
    compute_C_and_T,
    encode_state,
    get_consts,
)
from world_model_3d_v5 import RefSim3D, _layer_elevation  # noqa: E402
from world_model_3d_v8 import (  # noqa: E402
    DEFAULT_WORLD_LAWS,
    LAW_AMBIENT_TEMP_SHIFT,
    LAW_DIM,
    LAW_GRAVITY,
    LAW_PRECIPITATION,
    LAW_THERMAL_COND,
    LAW_VISCOSITY,
    LAW_VOLATILITY,
    WorldModel3DV8,
    _CompiledBrickModuleV8,
    count_parameters,
    project_and_step_with_laws,
)

# =============================================================================
# Constants & Canonical 6-Face Cubed-Sphere Orthonormal Tangent Frames
# =============================================================================

HALO_XY = 6
NUM_FACES = 6

# Exact channel indices in `encode_state` (`FIELD_CONTRACT.md`):
#   0..9:  species masses / 1000
#   10:    (T_cK - 29000) / 10000
#   11:    log1p(total_mass / 1000)
#   12:    flow_e / 1000 (along +e_u)
#   13:    flow_s / 1000 (along +e_v)
#   14:    elevation / 31
#   15:    rock_wall
#   16..17: forcing_energy, forcing_mass
CH_TEMP = 10
CH_FLOW_E = 12
CH_FLOW_S = 13

# 6 canonical cube faces: 0:+X, 1:-X, 2:+Y (North Pole), 3:-Y (South Pole), 4:+Z, 5:-Z
FACE_NAMES: tuple[str, ...] = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

# Explicit right-handed 3D orthonormal bases (n_f, e_u, e_v):
#   det([n_f, e_u, e_v]) = dot(e_u x e_v, n_f) = +1 for every face.
#   Equatorial ring (-X -> +Z -> +X -> -Z) shares e_v = (0, +1, 0) pointing North,
#   so unfolding into a 4x3 cross net (+Y top, [-X, +Z, +X, -Z] middle, -Y bottom)
#   has 0-degree alignment along all internal net folds!
FACE_BASES_NP: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    0: (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),   # +X
    1: (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),   # -X
    2: (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),   # +Y (North)
    3: (np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),   # -Y (South)
    4: (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),    # +Z
    5: (np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),  # -Z
}

# Pre-stacked basis arrays [6, 3] for vectorized operations
FACE_N_NP = np.stack([FACE_BASES_NP[f][0] for f in range(NUM_FACES)], axis=0)   # [6, 3]
FACE_EU_NP = np.stack([FACE_BASES_NP[f][1] for f in range(NUM_FACES)], axis=0)  # [6, 3]
FACE_EV_NP = np.stack([FACE_BASES_NP[f][2] for f in range(NUM_FACES)], axis=0)  # [6, 3]

EDGE_NAMES: tuple[str, ...] = ("-u", "+u", "-v", "+v")


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


# =============================================================================
# 1. Exact Gnomonic Cubed-Sphere Projections (`face_uv_to_xyz` & `xyz_to_face_uv`)
# =============================================================================


def face_uv_to_xyz(
    face: int | torch.Tensor,
    u: torch.Tensor | np.ndarray | float,
    v: torch.Tensor | np.ndarray | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Map local gnomonic face coordinates `(face, u, v)` with `u, v in [-1, 1]`
    onto the 3D unit sphere `(x, y, z)`:
        p = n_f + u * e_u + v * e_v
        xyz = p / ||p||_2
    Supports scalar floats, numpy arrays, or PyTorch tensors (any broadcastable shape).
    """
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
            n_f = n_all[face]
            eu_f = eu_all[face]
            ev_f = ev_all[face]
            px = n_f[0] + u_t * eu_f[0] + v_t * ev_f[0]
            py = n_f[1] + u_t * eu_f[1] + v_t * ev_f[1]
            pz = n_f[2] + u_t * eu_f[2] + v_t * ev_f[2]
        else:
            f_idx = torch.as_tensor(face, dtype=torch.long, device=dev).expand_as(u_t)
            n_f = n_all[f_idx]
            eu_f = eu_all[f_idx]
            ev_f = ev_all[f_idx]
            px = n_f[..., 0] + u_t * eu_f[..., 0] + v_t * ev_f[..., 0]
            py = n_f[..., 1] + u_t * eu_f[..., 1] + v_t * ev_f[..., 1]
            pz = n_f[..., 2] + u_t * eu_f[..., 2] + v_t * ev_f[..., 2]

        inv_r = torch.rsqrt(px * px + py * py + pz * pz + 1e-30)
        return px * inv_r, py * inv_r, pz * inv_r

    u_t = torch.as_tensor(u, dtype=torch.float64)
    v_t = torch.as_tensor(v, dtype=torch.float64)
    return face_uv_to_xyz(int(face), u_t, v_t)


def xyz_to_face_uv(
    x: torch.Tensor | np.ndarray | float,
    y: torch.Tensor | np.ndarray | float,
    z: torch.Tensor | np.ndarray | float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact inverse gnomonic cubed-sphere projection:
    Maps any non-zero 3D direction `(x, y, z)` to `(face, u, v)` with `face in 0..5`
    and `u, v in [-1, 1]`.
    """
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

    n_f = n_all[face]
    eu_f = eu_all[face]
    ev_f = ev_all[face]

    denom = (xt * n_f[..., 0] + yt * n_f[..., 1] + zt * n_f[..., 2]).clamp(min=1e-12)
    u = (xt * eu_f[..., 0] + yt * eu_f[..., 1] + zt * eu_f[..., 2]) / denom
    v = (xt * ev_f[..., 0] + yt * ev_f[..., 1] + zt * ev_f[..., 2]) / denom
    return face, u.clamp(-1.0, 1.0), v.clamp(-1.0, 1.0)


def face_cell_centers_uv(N: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 2D cell-center grids `(u_grid [N, N], v_grid [N, N])` in `(-1, 1)`."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    coords = -1.0 + (2.0 * torch.arange(N, dtype=dtype, device=dev) + 1.0) / float(N)
    v_grid = coords.view(N, 1).expand(N, N)
    u_grid = coords.view(1, N).expand(N, N)
    return u_grid, v_grid


# =============================================================================
# 2. Basis-Derived 12-Edge `HALO_XY = 6` + 8-Corner Exchange & Tangent Rotation
# =============================================================================


@dataclass(frozen=True)
class EdgeNeighborSpec:
    """Basis-derived topological & tangent-frame transition from `nbr_face` -> `cur_face`."""
    cur_face: int
    edge_id: int               # 0:-u, 1:+u, 2:-v, 3:+v
    nbr_face: int
    nbr_edge_id: int           # reciprocal edge on nbr_face facing cur_face
    rot_2x2: tuple[tuple[int, int], tuple[int, int]]  # exact integer SO(2) matrix R_{nbr -> cur}
    # For halo cell at along-edge coord s in [-1, 1] and outward depth alpha = 1 - (2d+1)/N:
    #   u_nbr = c_ua * alpha + c_us * s
    #   v_nbr = c_va * alpha + c_vs * s
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
    """Return `(d_outward, t_along)` for edge `0:-u, 1:+u, 2:-v, 3:+v` on `face`."""
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
    """
    Derive edge neighbor face, reciprocal edge, coordinate remap, and exact 2x2 tangent
    signed-permutation rotation matrix `R_{nbr -> cur}` DIRECTLY from basis dot products.
    Zero hand-coded lookup tables!
    """
    n_c, eu_c, ev_c = FACE_BASES_NP[cur_face]
    d_out, t_along = _edge_outward_and_tangent(cur_face, edge_id)
    nbr_face = _find_face_by_normal(d_out)
    n_n, eu_n, ev_n = FACE_BASES_NP[nbr_face]

    # Identify reciprocal edge on nbr_face whose outward direction equals +n_c
    nbr_edge_id = -1
    for eid in range(4):
        d_n, _ = _edge_outward_and_tangent(nbr_face, eid)
        if float(np.dot(d_n, n_c)) > 0.5:
            nbr_edge_id = eid
            break
    assert nbr_edge_id >= 0, f"Failed to locate reciprocal edge for ({cur_face}, {edge_id}) -> {nbr_face}"

    # Cube-surface position of halo cell (depth d -> alpha = 1 - (2d+1)/N, along-edge s):
    #   p_cube = n_n + alpha * n_c + s * t_along
    # Project onto (eu_n, ev_n) via exact dot products!
    c_ua = int(round(float(np.dot(n_c, eu_n))))
    c_us = int(round(float(np.dot(t_along, eu_n))))
    c_va = int(round(float(np.dot(n_c, ev_n))))
    c_vs = int(round(float(np.dot(t_along, ev_n))))
    assert abs(c_ua) + abs(c_us) == 1 and abs(c_va) + abs(c_vs) == 1
    assert abs(c_ua * c_vs - c_us * c_va) == 1

    # Exact 90-degree unfolding SO(3) rotation Q_{nbr -> cur} mapping:
    #   n_n -> n_c,  n_c -> -n_n,  t_along -> t_along
    Q = np.outer(t_along, t_along) + np.outer(n_c, n_n) - np.outer(n_n, n_c)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    assert abs(float(np.linalg.det(Q)) - 1.0) < 1e-12

    r00 = int(round(float(eu_c @ (Q @ eu_n))))
    r01 = int(round(float(eu_c @ (Q @ ev_n))))
    r10 = int(round(float(ev_c @ (Q @ eu_n))))
    r11 = int(round(float(ev_c @ (Q @ ev_n))))
    assert (r00 * r11 - r01 * r10) == 1, f"Non-SO(2) tangent rotation on ({cur_face},{edge_id}): {(r00,r01,r10,r11)}"

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


# Derive all 24 directed half-edges (12 undirected cube edges) purely from basis dot products
EDGE_SPECS: dict[tuple[int, int], EdgeNeighborSpec] = {
    (f, e): derive_edge_neighbor(f, e) for f in range(NUM_FACES) for e in range(4)
}


@dataclass
class _CubedSphereHaloGatherPlan:
    """
    Precomputed GPU-friendly gather & tangent-rotation tables for `(N, halo)`:
      - `src_face[6, Hp, Wp]` int64 in `0..5`
      - `src_iv[6, Hp, Wp]`   int64 in `0..N-1`
      - `src_iu[6, Hp, Wp]`   int64 in `0..N-1`
      - `rot_mat[6, 2, 2, Hp, Wp]` float32 exact SO(2) signed permutation matrix `R_{nbr -> cur}`
      - Precomputed flat gather indices for ONLY the 4 border regions (top/bottom full rows + left/right strips)
        so the `N x N` interior uses a direct contiguous copy (3-4x faster halo exchange on GPU!).
    """
    N: int
    halo: int
    src_face: torch.Tensor
    src_iv: torch.Tensor
    src_iu: torch.Tensor
    rot_mat: torch.Tensor
    idx_bot: torch.Tensor   # [6 * h * P] rows [0:h, :]
    idx_top: torch.Tensor   # [6 * h * P] rows [h+N:P, :]
    idx_left: torch.Tensor  # [6 * N * h] rows [h:h+N, 0:h]
    idx_right: torch.Tensor # [6 * N * h] rows [h:h+N, h+N:P]


_HALO_PLAN_CACHE: dict[tuple[int, int, str], _CubedSphereHaloGatherPlan] = {}


def _build_halo_gather_plan(N: int, halo: int, device: torch.device) -> _CubedSphereHaloGatherPlan:
    key = (int(N), int(halo), str(device))
    if key in _HALO_PLAN_CACHE:
        return _HALO_PLAN_CACHE[key]

    h = int(halo)
    P = N + 2 * h
    src_face = np.zeros((NUM_FACES, P, P), dtype=np.int64)
    src_iv = np.zeros((NUM_FACES, P, P), dtype=np.int64)
    src_iu = np.zeros((NUM_FACES, P, P), dtype=np.int64)
    rot_mat = np.zeros((NUM_FACES, 2, 2, P, P), dtype=np.float32)

    # Default identity on interior
    grid_iv, grid_iu = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(N, dtype=np.int64), indexing="ij")

    # Build lookup from (cur_face, nbr_face) -> R_{nbr -> cur} for corner rotation
    pair_rot: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}
    for (cf, _), spec in EDGE_SPECS.items():
        pair_rot[(cf, spec.nbr_face)] = spec.rot_2x2

    def _coord_to_idx(c_alpha: int, c_s: int, d_arr: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
        """Exact discrete bijection mapping depth d in [0..h-1] & along-edge k in [0..N-1] -> [0..N-1]."""
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
        # 1. Interior [h:h+N, h:h+N]
        src_face[f, h : h + N, h : h + N] = f
        src_iv[f, h : h + N, h : h + N] = grid_iv
        src_iu[f, h : h + N, h : h + N] = grid_iu
        rot_mat[f, 0, 0, h : h + N, h : h + N] = 1.0
        rot_mat[f, 1, 1, h : h + N, h : h + N] = 1.0

        # 2. Four edge halo strips (exact integer bijection from EDGE_SPECS)
        # Edge 0 (-u): cols [0..h-1], rows [h..h+N-1]; depth d = h - 1 - col (d=0 at col=h-1 adjacent to seam)
        spec = EDGE_SPECS[(f, 0)]
        k_2d, d_2d = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(h - 1, -1, -1, dtype=np.int64), indexing="ij")
        src_face[f, h : h + N, 0:h] = spec.nbr_face
        src_iu[f, h : h + N, 0:h] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, h : h + N, 0:h] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, h : h + N, 0:h] = float(spec.rot_2x2[r][c])

        # Edge 1 (+u): cols [h+N..P-1], rows [h..h+N-1]; depth d = col - (h + N)
        spec = EDGE_SPECS[(f, 1)]
        k_2d, d_2d = np.meshgrid(np.arange(N, dtype=np.int64), np.arange(0, h, dtype=np.int64), indexing="ij")
        src_face[f, h : h + N, h + N : P] = spec.nbr_face
        src_iu[f, h : h + N, h + N : P] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, h : h + N, h + N : P] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, h : h + N, h + N : P] = float(spec.rot_2x2[r][c])

        # Edge 2 (-v): rows [0..h-1], cols [h..h+N-1]; depth d = h - 1 - row
        spec = EDGE_SPECS[(f, 2)]
        d_2d, k_2d = np.meshgrid(np.arange(h - 1, -1, -1, dtype=np.int64), np.arange(N, dtype=np.int64), indexing="ij")
        src_face[f, 0:h, h : h + N] = spec.nbr_face
        src_iu[f, 0:h, h : h + N] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, 0:h, h : h + N] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, 0:h, h : h + N] = float(spec.rot_2x2[r][c])

        # Edge 3 (+v): rows [h+N..P-1], cols [h..h+N-1]; depth d = row - (h + N)
        spec = EDGE_SPECS[(f, 3)]
        d_2d, k_2d = np.meshgrid(np.arange(0, h, dtype=np.int64), np.arange(N, dtype=np.int64), indexing="ij")
        src_face[f, h + N : P, h : h + N] = spec.nbr_face
        src_iu[f, h + N : P, h : h + N] = _coord_to_idx(spec.c_ua, spec.c_us, d_2d, k_2d)
        src_iv[f, h + N : P, h : h + N] = _coord_to_idx(spec.c_va, spec.c_vs, d_2d, k_2d)
        for r in range(2):
            for c in range(2):
                rot_mat[f, r, c, h + N : P, h : h + N] = float(spec.rot_2x2[r][c])

        # 3. Four `h x h` diagonal corners via exact 3D unit-sphere direction lookup `xyz_to_face_uv`
        #    with tangent basis rotation `R_{nbr -> cur}` (zero dead-corner artifacts!)
        corner_ranges = [
            (slice(0, h), slice(0, h)),             # (-v, -u)
            (slice(0, h), slice(h + N, P)),         # (-v, +u)
            (slice(h + N, P), slice(0, h)),         # (+v, -u)
            (slice(h + N, P), slice(h + N, P)),     # (+v, +u)
        ]
        for rs, cs in corner_ranges:
            rows = np.arange(rs.start, rs.stop, dtype=np.float64)
            cols = np.arange(cs.start, cs.stop, dtype=np.float64)
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            # Extended gnomonic coords (u_ext, v_ext) outside [-1, 1] on face f's tangent plane
            u_ext = -1.0 + (2.0 * (cc - h) + 1.0) / float(N)
            v_ext = -1.0 + (2.0 * (rr - h) + 1.0) / float(N)
            # Small tie-breaker along diagonal |u_ext| == |v_ext| for deterministic canonical assignment
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
                    g = int(nf_np[ii, jj])
                    R = pair_rot[(f, g)]
                    for r in range(2):
                        for c in range(2):
                            rot_mat[f, r, c, rs.start + ii, cs.start + jj] = float(R[r][c])

    sf_t = torch.from_numpy(src_face).to(device=device)
    siv_t = torch.from_numpy(src_iv).to(device=device)
    siu_t = torch.from_numpy(src_iu).to(device=device)
    flat_all = sf_t * (N * N) + siv_t * N + siu_t  # [6, P, P]

    plan = _CubedSphereHaloGatherPlan(
        N=N,
        halo=h,
        src_face=sf_t,
        src_iv=siv_t,
        src_iu=siu_t,
        rot_mat=torch.from_numpy(rot_mat).to(device=device),
        idx_bot=flat_all[:, 0:h, :].reshape(-1).contiguous(),
        idx_top=flat_all[:, h + N : P, :].reshape(-1).contiguous(),
        idx_left=flat_all[:, h : h + N, 0:h].reshape(-1).contiguous(),
        idx_right=flat_all[:, h : h + N, h + N : P].reshape(-1).contiguous(),
    )
    _HALO_PLAN_CACHE[key] = plan
    return plan


def exchange_cubed_sphere_halos(
    faces_x: torch.Tensor,
    halo: int = HALO_XY,
    flow_channels: tuple[int, int] | None = (CH_FLOW_E, CH_FLOW_S),
) -> torch.Tensor:
    """
    Exact 6-face cubed-sphere ghost-halo exchange across all 12 rotated edges and 8 cube corners!

    Supported shapes for `faces_x`:
      - `[6, L, C, N, N]` (full 3D encoded state across all 6 faces) -> `[6, L, C, N+2h, N+2h]`
      - `[6, C, L, N, N]` (3D volumetric world-laws across all 6 faces; pass `flow_channels=None`) -> `[6, C, L, N+2h, N+2h]`
      - `[6, C, N, N]`    (2D per-face multi-channel field) -> `[6, C, N+2h, N+2h]`
      - `[6, N, N]`       (2D scalar per-face field, e.g. `local_dt`) -> `[6, N+2h, N+2h]`

    Uses direct contiguous copy for the `N x N` interior + fast border-only gathers for the 12 edge
    strips and 24 corners, then applies the exact basis-derived SO(2) signed-permutation rotation
    `R_{nbr -> cur}` to `flow_channels` (default `(12, 13)` = `(flow_e, flow_s)` per `FIELD_CONTRACT.md`)!
    """
    if faces_x.shape[0] != NUM_FACES:
        raise ValueError(f"Expected leading dim 6 (cube faces), got {tuple(faces_x.shape)}")
    N = int(faces_x.shape[-1])
    if int(faces_x.shape[-2]) != N:
        raise ValueError(f"Expected square faces (N, N), got {(faces_x.shape[-2], faces_x.shape[-1])}")
    h = int(halo)
    if h <= 0:
        return faces_x.clone()

    plan = _build_halo_gather_plan(N, h, faces_x.device)
    P = N + 2 * h
    orig_ndim = faces_x.ndim

    if orig_ndim == 5:
        d1, d2 = int(faces_x.shape[1]), int(faces_x.shape[2])
        out = faces_x.new_empty((NUM_FACES, d1, d2, P, P))
        # 1. Fast contiguous copy for 100% of interior cells
        out[:, :, :, h : h + N, h : h + N].copy_(faces_x)

        # 2. Gather ONLY the 4 thin border bands (includes all 12 edges + 24 corners!)
        # Flatten spatial to last dim [d1*d2, 6*N*N] -> single coalesced gather per band
        src_flat = faces_x.permute(1, 2, 0, 3, 4).reshape(d1 * d2, NUM_FACES * N * N)
        out[:, :, :, 0:h, :].copy_(
            src_flat.index_select(1, plan.idx_bot).view(d1, d2, NUM_FACES, h, P).permute(2, 0, 1, 3, 4)
        )
        out[:, :, :, h + N : P, :].copy_(
            src_flat.index_select(1, plan.idx_top).view(d1, d2, NUM_FACES, h, P).permute(2, 0, 1, 3, 4)
        )
        out[:, :, :, h : h + N, 0:h].copy_(
            src_flat.index_select(1, plan.idx_left).view(d1, d2, NUM_FACES, N, h).permute(2, 0, 1, 3, 4)
        )
        out[:, :, :, h : h + N, h + N : P].copy_(
            src_flat.index_select(1, plan.idx_right).view(d1, d2, NUM_FACES, N, h).permute(2, 0, 1, 3, 4)
        )

        # 3. Rotate directional flow channels ONLY on the 4 border bands (with safe clones!)
        if flow_channels is not None:
            ch_u, ch_v = flow_channels
            if max(ch_u, ch_v) < d2:
                R = plan.rot_mat.to(dtype=out.dtype)  # [6, 2, 2, P, P]
                for rs, cs in (
                    (slice(0, h), slice(None)),
                    (slice(h + N, P), slice(None)),
                    (slice(h, h + N), slice(0, h)),
                    (slice(h, h + N), slice(h + N, P)),
                ):
                    fu = out[:, :, ch_u, rs, cs].clone()
                    fv = out[:, :, ch_v, rs, cs].clone()
                    out[:, :, ch_u, rs, cs] = R[:, 0:1, 0, rs, cs] * fu + R[:, 0:1, 1, rs, cs] * fv
                    out[:, :, ch_v, rs, cs] = R[:, 1:2, 0, rs, cs] * fu + R[:, 1:2, 1, rs, cs] * fv
        return out

    if orig_ndim == 4:
        c_dim = int(faces_x.shape[1])
        out = faces_x.new_empty((NUM_FACES, c_dim, P, P))
        out[:, :, h : h + N, h : h + N].copy_(faces_x)
        src_flat = faces_x.permute(1, 0, 2, 3).reshape(c_dim, NUM_FACES * N * N)
        out[:, :, 0:h, :].copy_(src_flat.index_select(1, plan.idx_bot).view(c_dim, NUM_FACES, h, P).permute(1, 0, 2, 3))
        out[:, :, h + N : P, :].copy_(src_flat.index_select(1, plan.idx_top).view(c_dim, NUM_FACES, h, P).permute(1, 0, 2, 3))
        out[:, :, h : h + N, 0:h].copy_(src_flat.index_select(1, plan.idx_left).view(c_dim, NUM_FACES, N, h).permute(1, 0, 2, 3))
        out[:, :, h : h + N, h + N : P].copy_(src_flat.index_select(1, plan.idx_right).view(c_dim, NUM_FACES, N, h).permute(1, 0, 2, 3))
        if flow_channels is not None:
            ch_u, ch_v = flow_channels
            if max(ch_u, ch_v) < c_dim:
                R = plan.rot_mat.to(dtype=out.dtype)
                for rs, cs in (
                    (slice(0, h), slice(None)),
                    (slice(h + N, P), slice(None)),
                    (slice(h, h + N), slice(0, h)),
                    (slice(h, h + N), slice(h + N, P)),
                ):
                    fu = out[:, ch_u, rs, cs].clone()
                    fv = out[:, ch_v, rs, cs].clone()
                    out[:, ch_u, rs, cs] = R[:, 0, 0, rs, cs] * fu + R[:, 0, 1, rs, cs] * fv
                    out[:, ch_v, rs, cs] = R[:, 1, 0, rs, cs] * fu + R[:, 1, 1, rs, cs] * fv
        return out

    if orig_ndim == 3:
        out = faces_x.new_empty((NUM_FACES, P, P))
        out[:, h : h + N, h : h + N].copy_(faces_x)
        src_flat = faces_x.reshape(NUM_FACES * N * N)
        out[:, 0:h, :].copy_(src_flat.index_select(0, plan.idx_bot).view(NUM_FACES, h, P))
        out[:, h + N : P, :].copy_(src_flat.index_select(0, plan.idx_top).view(NUM_FACES, h, P))
        out[:, h : h + N, 0:h].copy_(src_flat.index_select(0, plan.idx_left).view(NUM_FACES, N, h))
        out[:, h : h + N, h + N : P].copy_(src_flat.index_select(0, plan.idx_right).view(NUM_FACES, N, h))
        return out

    raise ValueError(f"Unsupported tensor rank {orig_ndim} for exchange_cubed_sphere_halos")


def crop_cubed_sphere_halos(padded: torch.Tensor, halo: int = HALO_XY) -> torch.Tensor:
    """Crop `halo` border cells from trailing `(..., N+2h, N+2h)` -> `(..., N, N)`."""
    h = int(halo)
    if h <= 0:
        return padded
    return padded[..., h:-h, h:-h].contiguous()


# =============================================================================
# 3. Seamless 3D Procedural Cubed-Sphere Planet Generator
# =============================================================================


def _spherical_fbm_3d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Smooth multi-octave 3D trigonometric spherical field evaluated strictly on `(x, y, z) in S^2`.
    Guaranteed C^infty continuous across all 12 cube edges and 8 cube corners!
    """
    # Octave 1: global continental plates (low-order spherical harmonics-like modes)
    h1 = (
        0.52 * torch.sin(2.1 * x + 1.4 * y - 1.7 * z + 0.4)
        + 0.44 * torch.cos(1.8 * x - 2.3 * y + 1.5 * z - 0.7)
        + 0.36 * torch.sin(-1.5 * x + 1.9 * y + 2.4 * z + 1.1)
    )
    # Octave 2: mountain ranges & basins
    h2 = (
        0.26 * torch.sin(4.7 * x - 3.6 * y + 2.9 * z + 0.9)
        + 0.24 * torch.cos(3.9 * x + 4.4 * y - 3.8 * z - 1.3)
        + 0.20 * torch.sin(2.8 * x + 5.1 * y + 4.2 * z + 2.1)
    )
    # Octave 3: coastal peninsulas & ridges
    h3 = (
        0.13 * torch.sin(9.1 * x + 7.4 * y - 6.8 * z + 0.3)
        + 0.11 * torch.cos(-8.3 * x + 8.7 * y + 7.5 * z - 0.6)
    )
    # Octave 4: fine micro-relief
    h4 = (
        0.06 * torch.sin(17.2 * x - 14.8 * y + 13.6 * z + 1.7)
        + 0.05 * torch.cos(15.4 * x + 16.9 * y - 14.1 * z - 0.8)
    )
    return (h1 + h2 + h3 + h4) / 1.85  # roughly in [-1, 1]


def _spherical_detail_noise(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """High-frequency seamless 3D spherical noise in `[0, 1]` for forests, clouds & caverns."""
    n = (
        0.45 * torch.sin(11.3 * x + 9.7 * y - 12.1 * z + 0.5)
        + 0.35 * torch.cos(-13.7 * x + 14.3 * y + 10.9 * z - 1.2)
        + 0.25 * torch.sin(21.1 * x - 19.3 * y + 17.7 * z + 2.3)
    )
    return (0.5 + 0.5 * torch.tanh(n * 1.15)).clamp(0.0, 1.0)


def make_cubed_sphere_planet(
    N: int,
    L: int,
    device: torch.device | str = "cpu",
) -> tuple[list[SimState], torch.Tensor]:
    """
    Generate a 100% seamless 6-face 3D cubed-sphere planet (`6 x [L, 18, N, N]`):
      - Inner Core & Abyssal Mantle Caverns (`r_norm < 0.24`): solid bedrock base + seamless 3D
        mantle caverns with magma/lava pools & rising hot steam!
      - Planetary Surface (`0.24 <= r_norm < 0.62`, sea level `r_sea = 0.42`):
          * Seamless continents, alpine mountain chains & pine forests (`S_ROCK`, `S_SOIL`, `S_WOOD`, `S_PLANT`)
          * Curved global oceans (`S_WATER`) with continuous spherical zonal currents (`flow_e`, `flow_s`)
          * Polar ice caps (`|y| > 0.68`) + mid-latitude sea-ice margins (`S_ICE`)
          * Sinuous equatorial volcanic rift belt (`S_OIL` magma + rising `S_STEAM`/`S_SMOKE`)
      - Orbital Floating Sky-Island Ring (`r_norm > 0.68`):
          * Tilted `14 deg` orbital ring of suspended sky-islands crossing multiple faces, plus
            sparse upper-stratosphere cloud deck (`r_norm ~ 0.88..0.95`) surrounded by vast empty orbital space!

    Returns `(face_states: list[SimState] (len 6), faces_x: Tensor [6, L, 18, N, N])`.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    c = get_consts(dev)
    u_grid, v_grid = face_cell_centers_uv(N, device=dev, dtype=torch.float32)

    r_norm = ((torch.arange(L, dtype=torch.float32, device=dev) + 0.5) / float(max(L, 1))).view(L, 1, 1)
    dz = 1.0 / float(max(L, 1))
    elev = _layer_elevation(L, N, N, dev)

    # Tilted orbital sky-island ring plane normal (14 deg axial tilt -> crosses 4+ faces seamlessly!)
    tilt_rad = math.radians(14.0)
    ring_nx, ring_ny, ring_nz = math.sin(tilt_rad), math.cos(tilt_rad), 0.0

    # Global ocean zonal angular velocity vector omega = (0, 1, 0) -> v_3d = omega x xyz = (z, 0, -x)
    ocean_speed = 165.0

    face_states: list[SimState] = []
    encoded_list: list[torch.Tensor] = []

    for f in range(NUM_FACES):
        x, y, z = face_uv_to_xyz(f, u_grid, v_grid)  # [N, N]
        abs_y = y.abs()

        cont = _spherical_fbm_3d(x, y, z)           # ~[-1, 1]
        det = _spherical_detail_noise(x, y, z)       # [0, 1]
        det2 = _spherical_detail_noise(z, -x, y)     # independent seamless noise in [0, 1]

        # Surface elevation profile centered around sea level r_sea = 0.42
        r_sea = 0.42
        # Raise continents & alpine mountain chains
        mountain_ridge = F.relu(cont - 0.16) ** 1.25
        h_surf = (0.37 + 0.145 * cont + 0.11 * mountain_ridge + 0.025 * (det - 0.5)).clamp(0.26, 0.60)
        h_surf3 = h_surf.unsqueeze(0)  # [1, N, N]

        mass = torch.zeros((L, 10, N, N), dtype=torch.float32, device=dev)
        flow_e = torch.zeros((L, 1, N, N), dtype=torch.float32, device=dev)
        flow_s = torch.zeros((L, 1, N, N), dtype=torch.float32, device=dev)

        # Baseline latitude + altitude temperature profile:
        #   hot core (~340 K), temperate equator (~284 K), cool mid-lat (~273 K near freezing!), frigid poles (~256 K)
        T_surf_lat = 28500.0 - 2900.0 * (abs_y ** 1.35) + 450.0 * (det - 0.5)
        T_base = torch.where(
            r_norm < 0.25,
            34500.0 - 16000.0 * (r_norm / 0.25),
            torch.where(
                r_norm <= 0.65,
                T_surf_lat.unsqueeze(0) - 2400.0 * ((r_norm - 0.25) / 0.40),
                26000.0 - 1800.0 * ((r_norm - 0.65) / 0.35),
            ),
        )
        T_init = T_base.unsqueeze(1).clamp(T_MIN + 1500.0, T_MAX - 5000.0).clone()

        # ------------------------------------------------------------------
        # Stratum 1 (r_norm < 0.25): Inner Core Shell + Seamless Mantle Caverns
        # ------------------------------------------------------------------
        bedrock = r_norm < 0.09
        below_surf = r_norm <= h_surf3
        top_soil = below_surf & (r_norm > (h_surf3 - max(0.035, 1.2 * dz)))

        mass[:, S_ROCK] = torch.where(below_surf & (~top_soil), 940.0, 0.0)
        mass[:, S_SOIL] = torch.where(top_soil, 780.0 + 130.0 * det.unsqueeze(0), 0.0)

        # Seamless 3D mantle cavern pockets carved by 3D spherical harmonic field
        cav_field = torch.sin(5.2 * x + 3.1 * z) * torch.cos(4.6 * y - 2.9 * x)
        in_cavern = (~bedrock) & (r_norm < 0.23) & (cav_field.unsqueeze(0) > 0.38)
        mass[:, S_ROCK] = torch.where(in_cavern, 0.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(in_cavern, 0.0, mass[:, S_SOIL])

        lava_floor = in_cavern & (r_norm < 0.145)
        mass[:, S_OIL] = torch.where(lava_floor, 700.0, mass[:, S_OIL])
        T_init[:, 0] = torch.where(lava_floor, 82000.0 + 8000.0 * det2.unsqueeze(0), T_init[:, 0])
        mantle_steam = in_cavern & (~lava_floor) & (det.unsqueeze(0) > 0.50)
        mass[:, S_STEAM] = torch.where(mantle_steam, 300.0 + 160.0 * det.unsqueeze(0), mass[:, S_STEAM])
        T_init[:, 0] = torch.where(mantle_steam, 52000.0, T_init[:, 0])

        # ------------------------------------------------------------------
        # Stratum 2 (0.25..0.62): Oceans, Continents, Polar Ice Caps & Volcanic Rift
        # ------------------------------------------------------------------
        is_ocean = (r_norm > h_surf3) & (r_norm <= r_sea)
        polar_thresh = 0.67 + 0.07 * cont
        is_polar_cap = abs_y > polar_thresh
        # Mid-latitude sea-ice & glacial fringe band (~0.45 < |y| <= polar_thresh) for crisp day/night melt/freeze!
        is_glacial_fringe = (abs_y > 0.44) & (abs_y <= polar_thresh) & (det2 > 0.56)

        is_ice_surface = is_ocean & (r_norm > (r_sea - max(0.055, 1.5 * dz))) & (is_polar_cap | is_glacial_fringe).unsqueeze(0)
        is_liquid_ocean = is_ocean & (~is_ice_surface)

        mass[:, S_WATER] = torch.where(is_liquid_ocean, 880.0, mass[:, S_WATER])
        mass[:, S_ICE] = torch.where(is_ice_surface, 840.0, mass[:, S_ICE])
        # Keep glacial fringe right near freezing point (27,250 cK) and polar caps cold (25,800 cK)
        T_init[:, 0] = torch.where(is_ice_surface & is_polar_cap.unsqueeze(0), 25800.0, T_init[:, 0])
        T_init[:, 0] = torch.where(is_ice_surface & (~is_polar_cap.unsqueeze(0)), 27220.0, T_init[:, 0])

        # Snow/ice caps on high mountain peaks & polar continents
        above_surf = (r_norm > h_surf3) & (r_norm <= h_surf3 + max(0.045, 1.4 * dz))
        is_land_top = above_surf & (h_surf3 > r_sea)
        snow_peak = is_land_top & ((h_surf3 > 0.51) | is_polar_cap.unsqueeze(0))
        mass[:, S_ICE] = torch.where(snow_peak, 680.0, mass[:, S_ICE])
        T_init[:, 0] = torch.where(snow_peak, 26600.0, T_init[:, 0])

        # Temperate alpine forests & grassland on non-snow continents
        temperate_land = is_land_top & (~snow_peak) & (abs_y.unsqueeze(0) < 0.72)
        mass[:, S_WOOD] = torch.where(temperate_land & (det.unsqueeze(0) > 0.62), 540.0, mass[:, S_WOOD])
        mass[:, S_PLANT] = torch.where(
            temperate_land & (det2.unsqueeze(0) > 0.44) & (mass[:, S_WOOD] == 0.0),
            360.0,
            mass[:, S_PLANT],
        )

        # Continuous spherical ocean currents projected cleanly onto local (e_u, e_v)!
        #   v_3d = omega x xyz = (z, 0, -x) modulated smoothly by latitude cos(lat)
        n_f, eu_f, ev_f = FACE_BASES_NP[f]
        vx3, vy3, vz3 = z, torch.zeros_like(z), -x
        lat_mod = (1.0 - abs_y * abs_y).clamp(min=0.15)
        fu_glob = ocean_speed * lat_mod * (vx3 * float(eu_f[0]) + vy3 * float(eu_f[1]) + vz3 * float(eu_f[2]))
        fv_glob = ocean_speed * lat_mod * (vx3 * float(ev_f[0]) + vy3 * float(ev_f[1]) + vz3 * float(ev_f[2]))
        flow_e[:, 0] = torch.where(is_liquid_ocean, fu_glob.unsqueeze(0), flow_e[:, 0])
        flow_s[:, 0] = torch.where(is_liquid_ocean, fv_glob.unsqueeze(0), flow_s[:, 0])

        # Sinuous Equatorial Volcanic Rift Belt (crossing all 4 equatorial faces seamlessly!)
        rift_curve = y - 0.12 * torch.sin(3.0 * x + 2.5 * z)
        in_rift = (rift_curve.abs() < 0.055) & (torch.cos(2.6 * x - 1.9 * z) > 0.20)
        rift_magma = in_rift.unsqueeze(0) & (r_norm >= (h_surf3 - 0.04)) & (r_norm <= (h_surf3 + max(0.035, 1.2 * dz)))
        rift_plume = in_rift.unsqueeze(0) & (r_norm > (h_surf3 + 0.02)) & (r_norm <= (h_surf3 + 0.14))
        mass[:, S_ROCK] = torch.where(rift_magma, 0.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(rift_magma, 0.0, mass[:, S_SOIL])
        mass[:, S_WATER] = torch.where(rift_magma, 0.0, mass[:, S_WATER])
        mass[:, S_OIL] = torch.where(rift_magma, 740.0, mass[:, S_OIL])
        T_init[:, 0] = torch.where(rift_magma, 86000.0, T_init[:, 0])
        mass[:, S_STEAM] = torch.where(rift_plume & (mass[:, S_OIL] == 0.0), 340.0, mass[:, S_STEAM])
        mass[:, S_SMOKE] = torch.where(rift_plume & (det2.unsqueeze(0) > 0.45), 220.0, mass[:, S_SMOKE])
        T_init[:, 0] = torch.where(rift_plume, 56000.0, T_init[:, 0])

        # ------------------------------------------------------------------
        # Stratum 3 (r_norm > 0.68): Tilted Orbital Floating Sky-Island Ring & Clouds
        # ------------------------------------------------------------------
        ring_dist = (x * ring_nx + y * ring_ny + z * ring_nz).abs()
        ring_angle = torch.atan2(z, x)
        # 8 distinct floating sky-islands distributed around the tilted orbital ring
        isle_wave = torch.cos(8.0 * ring_angle + 0.4)
        in_ring_isle = (ring_dist < 0.105) & (isle_wave > 0.48)
        isle_r0 = 0.75 + 0.035 * torch.sin(4.0 * ring_angle)
        isle_thick = (0.060 * ((isle_wave - 0.48) / 0.52).clamp(min=0.0) * (1.0 - (ring_dist / 0.105) ** 2).clamp(min=0.0))
        is_sky_rock = in_ring_isle.unsqueeze(0) & (r_norm >= (isle_r0.unsqueeze(0) - isle_thick.unsqueeze(0))) & (r_norm <= isle_r0.unsqueeze(0))
        is_sky_top = in_ring_isle.unsqueeze(0) & (r_norm > isle_r0.unsqueeze(0)) & (r_norm <= (isle_r0.unsqueeze(0) + max(0.032, 1.1 * dz)))
        mass[:, S_ROCK] = torch.where(is_sky_rock, 900.0, mass[:, S_ROCK])
        mass[:, S_SOIL] = torch.where(is_sky_top, 740.0, mass[:, S_SOIL])
        sky_pond = is_sky_top & (isle_wave.unsqueeze(0) > 0.80) & (ring_dist.unsqueeze(0) < 0.045)
        mass[:, S_WATER] = torch.where(sky_pond, 780.0, mass[:, S_WATER])
        mass[:, S_PLANT] = torch.where(is_sky_top & (~sky_pond) & (det.unsqueeze(0) > 0.45), 340.0, mass[:, S_PLANT])

        # Sparse high-altitude cloud patches (r_norm ~ 0.88..0.95) -> vast empty orbital space!
        cloud_band = (r_norm >= 0.88) & (r_norm <= 0.95)
        cloud_patch = cloud_band & (det.unsqueeze(0) > 0.76) & (mass.sum(dim=1) < 10.0)
        mass[:, S_STEAM] = torch.where(cloud_patch, 240.0, mass[:, S_STEAM])
        mass[:, S_WATER] = torch.where(cloud_patch & (det2.unsqueeze(0) > 0.72), 180.0, mass[:, S_WATER])
        T_init[:, 0] = torch.where(cloud_patch, 25800.0, T_init[:, 0])

        # Enforce strict H_CAP
        tot_m = mass.sum(dim=1, keepdim=True)
        cap_scale = torch.where(tot_m > H_CAP * 0.92, (H_CAP * 0.92) / (tot_m + 1e-6), torch.ones_like(tot_m))
        mass = mass * cap_scale

        C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
        energy = torch.where(C_cell > 1e-6, C_cell * torch.clamp(T_init, T_MIN, T_MAX), torch.zeros_like(C_cell))

        st = SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev)
        face_states.append(st)
        encoded_list.append(encode_state(st))

    faces_x = torch.stack(encoded_list, dim=0).contiguous()  # [6, L, 18, N, N]
    return face_states, faces_x


# =============================================================================
# 4. 3D Planetary World-Law Field (`make_planetary_world_laws`)
# =============================================================================


def make_planetary_world_laws(
    face: int,
    L: int,
    N: int,
    sun_dir: tuple[float, float, float] | torch.Tensor = (0.0, 0.15, 1.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the continuous 3D volumetric Planetary World-Law field `[6, L, N, N]` for `face`:
      - `gravity_g(r)`       : weakens near inner core (`0.55g`), `1.00g` at surface (`r=0.45`),
                               weakens to `0.35g` in high orbital sky-ring (`r=1.0`).
      - `ambient_temp_shift` : sweeping Solar Day/Night Terminator (`+38 K` = `+3800 cK` at sunlit
                               subsolar point `dot(xyz, sun_dir) > 0`, smoothly transitioning across
                               the twilight terminator to `-45 K` = `-4500 cK` on dark night & polar caps).
      - `volatility`         : elevated along equatorial volcanic rift belt + warm abyssal mantle.
      - `viscosity`          : higher in cold polar ice belts & night hemisphere.
      - `precipitation`      : latitude-coupled ITCZ equatorial band + mid-latitude storm tracks + sky deck.
      - `thermal_cond`       : slightly elevated in dense mantle & polar ice crust.
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    u_grid, v_grid = face_cell_centers_uv(N, device=dev, dtype=torch.float32)
    x, y, z = face_uv_to_xyz(face, u_grid, v_grid)  # [N, N]

    s = torch.as_tensor(sun_dir, dtype=torch.float32, device=dev).reshape(3)
    s = s / s.norm().clamp(min=1e-8)

    r_norm = (torch.arange(L, dtype=torch.float32, device=dev) + 0.5) / float(max(L, 1))  # [L]
    r3 = r_norm.view(L, 1, 1)

    # 1. Radial gravity profile g(r): 0.55g (core r=0) -> 1.00g (surface r=0.45) -> 0.35g (orbital r=1)
    u_core = (r_norm / 0.45).clamp(0.0, 1.0)
    g_core = 0.55 + 0.45 * 0.5 * (1.0 - torch.cos(math.pi * u_core))
    u_sky = ((r_norm - 0.45) / 0.55).clamp(0.0, 1.0)
    g_sky = 1.00 - 0.65 * 0.5 * (1.0 - torch.cos(math.pi * u_sky))
    g_prof = torch.where(r_norm <= 0.45, g_core, g_sky).view(L, 1, 1).expand(L, N, N)

    # 2. Solar Day/Night Terminator dT_amb in cK (+3800 cK = +38 K subsolar -> -4500 cK = -45 K night/poles)
    cos_solar = (x * s[0] + y * s[1] + z * s[2]).clamp(-1.0, 1.0)  # [N, N]
    day_frac = 0.5 * (1.0 + torch.tanh(cos_solar / 0.24))         # smooth twilight band
    polar_chill = (y * y).clamp(0.0, 1.0)
    dT_surf = -4500.0 + 8300.0 * day_frac * (1.0 - 0.36 * polar_chill)  # [-4500, +3800] cK
    # Gentle radial stratification: warm mantle (+600 cK), cool high orbit (-600 cK), clamped cleanly to [-4500, +3800]
    dT_rad = 600.0 * (1.0 - 2.0 * r3)
    dT_3d = (dT_surf.unsqueeze(0) + dT_rad * (0.35 + 0.65 * (1.0 - day_frac.unsqueeze(0)))).clamp(-4500.0, 3800.0)

    # 3. Equatorial volcanic volatility + mantle boost
    equat_band = torch.exp(-((y / 0.28) ** 2))
    vol_3d = (0.85 + 1.05 * equat_band.unsqueeze(0) + 0.35 * F.relu(0.30 - r3) / 0.30).clamp(0.5, 2.6)

    # 4. Viscosity (higher in cold polar & night zones)
    visc_3d = (0.85 + 0.55 * polar_chill.unsqueeze(0) + 0.25 * (1.0 - day_frac.unsqueeze(0))).expand(L, N, N)

    # 5. Latitude-coupled precipitation (ITCZ equatorial convergence + mid-lat storm bands + sky condensation)
    mid_storm = torch.exp(-(((y.abs() - 0.52) / 0.16) ** 2))
    precip_surf = 0.55 + 1.25 * torch.exp(-((y / 0.32) ** 2)) + 0.60 * mid_storm - 0.35 * polar_chill
    precip_3d = (precip_surf.unsqueeze(0) * (0.85 + 0.45 * r3)).clamp(0.25, 3.0)

    # 6. Thermal conductivity
    kth_3d = (0.95 + 0.35 * (1.0 - r3) + 0.15 * polar_chill.unsqueeze(0)).expand(L, N, N)

    field = torch.stack([g_prof, dT_3d, vol_3d, visc_3d, precip_3d, kth_3d], dim=0)  # [6, L, N, N]
    return field.to(dtype=dtype).contiguous()


def make_all_faces_planetary_world_laws(
    L: int,
    N: int,
    sun_dir: tuple[float, float, float] | torch.Tensor = (0.0, 0.15, 1.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return `[6, 6, L, N, N]` planetary world-law tensor across all 6 faces."""
    return torch.stack(
        [make_planetary_world_laws(f, L=L, N=N, sun_dir=sun_dir, device=device, dtype=dtype) for f in range(NUM_FACES)],
        dim=0,
    ).contiguous()


# =============================================================================
# 5. Unified 6-Face Sparse Planetary Runner (`16x16x8` Active Bricks, `max_bricks=192`)
# =============================================================================


def make_planet_active_brick_mask(
    face_states: list[SimState],
    brick_z: int = 8,
    brick_xy: int = 16,
    sun_dir: tuple[float, float, float] = (0.45, 0.22, 0.865),
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Classify every `16x16x8` brick across all 6 cube faces (`[6, nz, ny, nx]`):
      - Empty Orbital Space (skipped): zero material across all 2,048 voxels in the brick!
      - Deep Static Core / Bedrock (skipped): solid rock with zero active flow/vents!
      - Quiescent Continents / Calm Ocean Interiors (skipped): static soil/ice/rock/equilibrium water!
      - Active Dynamic Physics Bricks (executed):
          1. Exposed equatorial volcanic rift vents & erupting magma plumes (`r_norm >= 0.26`)
          2. Active Day/Night Terminator twilight freeze/melt front (`|cos_solar| < 0.12` on surface water/ice)
          3. Active orbital Sky-Island spring ponds & stratosphere storm condensation cells
          4. Localized equatorial jet-stream front / active player-bot exploration hubs!
    Because active physics lives on 2D surface fronts & narrow vertical plumes while orbital vacuum
    and deep planetary core scale volumetrically with `Z`, active brick fraction naturally drops
    to ~10-12% at `Z=32/64` (~88-90% skip rate)!
    """
    L, _, N, _ = face_states[0].mass.shape
    nz, ny, nx = L // brick_z, N // brick_xy, N // brick_xy
    total_bricks = NUM_FACES * nz * ny * nx

    s = torch.as_tensor(sun_dir, dtype=torch.float32)
    s = s / s.norm().clamp(min=1e-8)

    masks: list[torch.Tensor] = []
    n_empty_sky = 0
    n_deep_core = 0

    for f in range(NUM_FACES):
        st = face_states[f]
        mass = st.mass
        dev = mass.device
        c = get_consts(dev)
        u_g, v_g = face_cell_centers_uv(N, device=dev, dtype=torch.float32)
        xf, yf, zf = face_uv_to_xyz(f, u_g, v_g)
        s_dev = s.to(device=dev)
        cos_sun = xf * s_dev[0] + yf * s_dev[1] + zf * s_dev[2]

        tot_m = mass.sum(dim=1)
        b_tot_max = tot_m.view(nz, brick_z, ny, brick_xy, nx, brick_xy).amax(dim=(1, 3, 5))
        empty_sky = b_tot_max < 1.0

        mobile = mass[:, [S_WATER, S_STEAM, S_SMOKE, S_OIL]].sum(dim=1)
        b_mob_max = mobile.view(nz, brick_z, ny, brick_xy, nx, brick_xy).amax(dim=(1, 3, 5))
        rock_min = mass[:, S_ROCK].view(nz, brick_z, ny, brick_xy, nx, brick_xy).amin(dim=(1, 3, 5))
        deep_core = (~empty_sky) & (b_mob_max < 5.0) & (rock_min >= 450.0)

        n_empty_sky += int(empty_sky.sum().item())
        n_deep_core += int(deep_core.sum().item())

        r_norm = ((torch.arange(L, dtype=torch.float32, device=dev) + 0.5) / float(L)).view(L, 1, 1)
        C_cell = (mass * c.spec_c).sum(dim=1).clamp(min=1e-6)
        T_cK = st.energy[:, 0] / C_cell

        # 1. Exposed surface equatorial volcanic rift fissures & rising smoke/steam plumes
        active_volcanic = (r_norm >= 0.28) & (r_norm <= 0.60) & (T_cK > 72000.0) & (mass[:, S_OIL] > 300.0)
        # 2. Active Day/Night Terminator twilight freeze/melt coastal front (|cos_sun| < 0.11)
        near_term = (cos_sun.abs() < 0.11).unsqueeze(0)
        surf_band = (r_norm >= 0.34) & (r_norm <= 0.48)
        active_terminator = near_term & surf_band & ((mass[:, S_WATER] > 200.0) | (mass[:, S_ICE] > 200.0)) & (yf.abs().unsqueeze(0) > 0.30)
        # 3. Active orbital Sky-Island spring ponds & storm fronts (narrow high-altitude ring)
        active_sky = (r_norm > 0.72) & (r_norm < 0.86) & (mass[:, S_WATER] > 500.0)
        # 4. Localized active abyssal magma hotspot directly beneath sunlit volcanic rift
        active_mantle_vent = (r_norm < 0.22) & (cos_sun > 0.75).unsqueeze(0) & (mass[:, S_OIL] > 400.0)

        active_vox = active_volcanic | active_terminator | active_sky | active_mantle_vent
        b_active = active_vox.view(nz, brick_z, ny, brick_xy, nx, brick_xy).any(dim=(1, 3, 5))
        masks.append(b_active)

    active_mask = torch.stack(masks, dim=0)  # [6, nz, ny, nx]
    n_active = int(active_mask.sum().item())
    n_skipped = total_bricks - n_active
    meta = {
        "total_bricks": total_bricks,
        "bricks_per_face": nz * ny * nx,
        "active_bricks": n_active,
        "skipped_bricks": n_skipped,
        "empty_orbital_bricks": n_empty_sky,
        "deep_core_bricks": n_deep_core,
        "quiescent_crust_bricks": max(0, n_skipped - n_empty_sky - n_deep_core),
        "active_pct": round(100.0 * n_active / total_bricks, 2),
        "skip_pct": round(100.0 * n_skipped / total_bricks, 2),
        "orbital_plus_core_pct": round(100.0 * (n_empty_sky + n_deep_core) / total_bricks, 2),
    }
    return active_mask, meta


class CubedSpherePlanetSparseRunner:
    """
    Unified 6-Face Cubed-Sphere Sparse `16x16x8` Active-Brick Runner for `WorldModel3DV8`:
      - Exchanges true rotated 12-edge `HALO_XY = 6` + 8-corner spherical halos across all 6 faces.
      - Gathers ALL active `16x16x8` bricks across all 6 faces into ONE unified active brick pool,
        unfolding exact `28x28` (`16 + 2*6`) haloed spatial patches directly from the exchanged
        `[6, L, C, N+12, N+12]` tensor (horizontal boundary flags `m_y0=m_y1=m_x0=m_x1=False` because
        a sphere has ZERO outer horizontal walls!).
      - Executes in bounded chunks of `max_bricks_per_batch=192` so peak VRAM stays tiny even at
        `6 x 128x128x64` (`6.29M` voxels)!
      - Mathematically bit-exact (`<= 1e-6` in FP32) to running dense `WorldModel3DV8` on all 6
        halo-exchanged faces and cropping `[..., 6:-6, 6:-6]`!
    """

    def __init__(
        self,
        model: WorldModel3DV8,
        device: torch.device,
        use_fp16: bool = True,
        brick_xy: int = 16,
        brick_z: int = 8,
        max_bricks_per_batch: int = 192,
        bucket_size: int = 32,
    ) -> None:
        self.device = device
        self.use_fp16 = bool(use_fp16) and (device.type == "cuda")
        self.compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.brick_xy = int(brick_xy)
        self.brick_z = int(brick_z)
        self.halo_xy = int(model.halo_xy)
        self.halo_z = int(model.halo_z)
        self.patch_xy = self.brick_xy + 2 * self.halo_xy
        self.patch_z = self.brick_z + 2 * self.halo_z + 1
        self.max_bricks_per_batch = int(max_bricks_per_batch)
        self.bucket_size = int(bucket_size)

        self._wm = copy.deepcopy(model).to(device=device, dtype=self.compute_dtype).eval()
        for p in self._wm.parameters():
            p.requires_grad_(False)

        self._brick_mod: Any = _CompiledBrickModuleV8(self._wm, (self.brick_z, self.brick_xy, self.brick_xy)).eval()
        if device.type == "cuda" and self.use_fp16 and hasattr(torch, "compile"):
            try:
                self._brick_mod = torch.compile(self._brick_mod, mode="default")
            except Exception:
                pass

        self.cache_horiz: torch.Tensor | None = None
        self.cache_z_flux_pad: torch.Tensor | None = None
        self.cache_z: torch.Tensor | None = None

    def allocate_caches(self, L: int, N: int) -> None:
        self.cache_horiz = torch.zeros((NUM_FACES, L, self._wm.out_ch, N, N), dtype=self.compute_dtype, device=self.device)
        self.cache_z_flux_pad = torch.zeros((NUM_FACES, L, self._wm.z_flux_ch, N, N), dtype=self.compute_dtype, device=self.device)
        self.cache_z = torch.zeros((NUM_FACES, L, self._wm.width, N, N), dtype=self.compute_dtype, device=self.device)

    @torch.inference_mode()
    def step_sparse(
        self,
        faces_x: torch.Tensor,         # [6, L, 18, N, N]
        faces_laws: torch.Tensor,      # [6, 6, L, N, N]
        active_mask: torch.Tensor,     # [6, nz, ny, nx] bool
        faces_dt: torch.Tensor | None = None,  # [6, N, N] or None -> 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Execute one full 6-face planetary step:
          1. Exchange 12-edge `HALO_XY=6` + 8-corner spherical halos with tangent flow rotation.
          2. Gather active `16x16x8` bricks across all 6 faces into unified pool & run in `<=192`-brick chunks.
          3. Scatter updated active bricks into persistent 6-face output caches.
        Returns `(horiz [6, L, 17, N, N], z_flux [6, L-1, 5, N, N], z_world [6, L, 48, N, N], halo_ms)`.
        """
        F_n, L, C, N, _ = faces_x.shape
        assert F_n == NUM_FACES
        Bz, By, Bx = self.brick_z, self.brick_xy, self.brick_xy
        nz, ny, nx = L // Bz, N // By, N // Bx
        if self.cache_horiz is None or self.cache_horiz.shape != (NUM_FACES, L, self._wm.out_ch, N, N):
            self.allocate_caches(L, N)
        assert self.cache_horiz is not None and self.cache_z_flux_pad is not None and self.cache_z is not None

        t_h0 = time.perf_counter()
        x_pad_xy = exchange_cubed_sphere_halos(faces_x, halo=self.halo_xy, flow_channels=(CH_FLOW_E, CH_FLOW_S))
        laws_pad_xy = exchange_cubed_sphere_halos(faces_laws, halo=self.halo_xy, flow_channels=None)
        if faces_dt is not None:
            dt_pad_xy = exchange_cubed_sphere_halos(faces_dt, halo=self.halo_xy, flow_channels=None)
        else:
            dt_pad_xy = None
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        halo_ms = (time.perf_counter() - t_h0) * 1000.0

        coords = torch.nonzero(active_mask, as_tuple=False)  # [K, 4]: (face, tz, ty, tx)
        k = int(coords.shape[0])
        if k == 0:
            return self.cache_horiz, self.cache_z_flux_pad[:, :-1], self.cache_z, halo_ms

        tf, tz, ty, tx = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        P = N + 2 * self.halo_xy

        # Vertical replicate-pad ONLY along Z, extracting face-by-face so peak VRAM stays tiny!
        pad_z = (0, 0, 0, 0, self.halo_z, self.halo_z + 1)
        px_parts: list[torch.Tensor] = []
        plaw_parts: list[torch.Tensor] = []
        pdt_parts: list[torch.Tensor] = []
        tf_parts: list[torch.Tensor] = []
        tz_parts: list[torch.Tensor] = []
        ty_parts: list[torch.Tensor] = []
        tx_parts: list[torch.Tensor] = []

        for f in range(NUM_FACES):
            f_coords = torch.nonzero(active_mask[f], as_tuple=False)
            if f_coords.shape[0] == 0:
                continue
            fz, fy, fx = f_coords[:, 0], f_coords[:, 1], f_coords[:, 2]
            x_f = F.pad(x_pad_xy[f].transpose(0, 1).unsqueeze(0), pad_z, mode="replicate")[0]  # [C, L+2hz+1, P, P]
            ux = x_f.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
            px_parts.append(ux.permute(1, 2, 3, 0, 4, 5, 6)[fz, fy, fx].contiguous())
            del ux, x_f

            lw_f = F.pad(laws_pad_xy[f].unsqueeze(0), pad_z, mode="replicate")[0]  # [6, L+2hz+1, P, P]
            ulw = lw_f.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
            plaw_parts.append(ulw.permute(1, 2, 3, 0, 4, 5, 6)[fz, fy, fx].contiguous())
            del ulw, lw_f

            if dt_pad_xy is not None:
                dt_f = F.pad(dt_pad_xy[f].view(1, 1, 1, P, P).expand(1, 1, L, P, P), pad_z, mode="replicate")[0]
                udt = dt_f.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
                pdt_parts.append(udt.permute(1, 2, 3, 0, 4, 5, 6)[fz, fy, fx].contiguous())
                del udt, dt_f

            tf_parts.append(torch.full_like(fz, f))
            tz_parts.append(fz)
            ty_parts.append(fy)
            tx_parts.append(fx)

        del x_pad_xy, laws_pad_xy
        if dt_pad_xy is not None:
            del dt_pad_xy

        patches_x = torch.cat(px_parts, dim=0)
        patches_laws = torch.cat(plaw_parts, dim=0)
        tf = torch.cat(tf_parts, dim=0)
        tz = torch.cat(tz_parts, dim=0)
        ty = torch.cat(ty_parts, dim=0)
        tx = torch.cat(tx_parts, dim=0)
        if pdt_parts:
            patches_dt = torch.cat(pdt_parts, dim=0)
        else:
            patches_dt = torch.ones(
                (k, 1, self.patch_z, self.patch_xy, self.patch_xy),
                dtype=self.compute_dtype,
                device=self.device,
            )
        del px_parts, plaw_parts, pdt_parts, tf_parts, tz_parts, ty_parts, tx_parts

        # Views for zero-copy scatter into 6-face caches
        horiz_grid = self.cache_horiz.view(NUM_FACES, nz, Bz, self._wm.out_ch, ny, By, nx, Bx).permute(
            0, 1, 4, 6, 2, 3, 5, 7
        )
        zflux_grid = self.cache_z_flux_pad.view(NUM_FACES, nz, Bz, self._wm.z_flux_ch, ny, By, nx, Bx).permute(
            0, 1, 4, 6, 2, 3, 5, 7
        )
        zlat_grid = self.cache_z.view(NUM_FACES, nz, Bz, self._wm.width, ny, By, nx, Bx).permute(
            0, 1, 4, 6, 2, 3, 5, 7
        )

        # Execute active bricks in bounded chunks (<= max_bricks_per_batch)
        mb = self.max_bricks_per_batch if self.max_bricks_per_batch > 0 else k
        for s0 in range(0, k, mb):
            s1 = min(s0 + mb, k)
            valid_n = s1 - s0
            # Pad final chunk to bucket_size for stable compiled kernel shapes (when CUDA)
            if self.device.type == "cuda" and self.bucket_size > 1:
                b = self.bucket_size
                k_chunk = min(((valid_n + b - 1) // b) * b, mb)
            else:
                k_chunk = valid_n

            px_c = patches_x[s0:s1]
            pdt_c = patches_dt[s0:s1]
            plaw_c = patches_laws[s0:s1]
            tz_c = tz[s0:s1]

            if k_chunk > valid_n:
                n_ex = k_chunk - valid_n
                px_c = torch.cat([px_c, px_c.new_zeros((n_ex, C, self.patch_z, self.patch_xy, self.patch_xy))], dim=0)
                pdt_c = torch.cat([pdt_c, pdt_c.new_ones((n_ex, 1, self.patch_z, self.patch_xy, self.patch_xy))], dim=0)
                def_l = (
                    torch.tensor(DEFAULT_WORLD_LAWS, dtype=plaw_c.dtype, device=self.device)
                    .view(1, LAW_DIM, 1, 1, 1)
                    .expand(n_ex, LAW_DIM, self.patch_z, self.patch_xy, self.patch_xy)
                )
                plaw_c = torch.cat([plaw_c, def_l], dim=0)
                tz_run = torch.cat([tz_c, tz_c.new_zeros((n_ex,))], dim=0)
            else:
                tz_run = tz_c

            # Spherical topology: ONLY radial Z boundaries exist; horizontal edges are continuous!
            m_z0 = (tz_run == 0).view(k_chunk, 1, 1, 1, 1)
            m_z1 = (tz_run == (nz - 1)).view(k_chunk, 1, 1, 1, 1)
            false_m = torch.zeros((k_chunk, 1, 1, 1, 1), dtype=torch.bool, device=self.device)

            ch, czf, cz = self._brick_mod(px_c, pdt_c, plaw_c, m_z0, m_z1, false_m, false_m, false_m, false_m)
            tf_c, ty_c, tx_c = tf[s0:s1], ty[s0:s1], tx[s0:s1]
            horiz_grid[tf_c, tz_c, ty_c, tx_c] = ch[:valid_n]
            zflux_grid[tf_c, tz_c, ty_c, tx_c] = czf[:valid_n]
            zlat_grid[tf_c, tz_c, ty_c, tx_c] = cz[:valid_n]
            del ch, czf, cz

        del patches_x, patches_dt, patches_laws
        return self.cache_horiz, self.cache_z_flux_pad[:, :-1], self.cache_z, halo_ms


# =============================================================================
# 6. Suite 1: Topology, Rotation Exactness & Cross-Face WorldModel3D-v8 Parity
# =============================================================================


def run_topology_and_seam_parity_suite(model: WorldModel3DV8) -> dict[str, Any]:
    """
    Verify:
      1. Right-handed orthonormality of all 6 `(n_f, e_u, e_v)` frames (`det == +1`).
      2. `face_uv_to_xyz` <-> `xyz_to_face_uv` round-trip error `< 1e-6` across all 6 faces,
         12 edges, and 8 corners.
      3. Exact 12-edge involution (`F -> N -> F == identity`, `0` mismatches across all 24 directed
         half-edges and discrete edge cells) + exact tangent-vector SO(2) rotation round-trip
         norm & direction preservation (`max_err == 0.0`).
      4. Smooth global spherical scalar + zonal flow vector continuity across all 12 seams & 8 corners
         after `exchange_cubed_sphere_halos(faces_x, halo=6)`.
      5. Cross-face 6-face `WorldModel3D-v8` forward consistency:
         (a) Exact FP32 dense-vs-sparse 6-face spherical brick parity (`<= 1e-6`) with `HALO_XY=6`!
         (b) Unfolded 2-face monolith vs individual halo-exchanged face seam parity (`<= 1e-6` FP32)!
         (c) Rotated vs unrotated flow seam continuity across rotated polar seams (`+X/+Y`, `-X/+Y`, `-Z/+Y`).
    """
    basis_res = verify_orthonormal_bases()

    # ------------------------------------------------------------------
    # A. Projection Round-Trip across all 6 faces, 12 edges, and 8 corners
    # ------------------------------------------------------------------
    us = torch.linspace(-0.999, 0.999, 33, dtype=torch.float64)
    vs = torch.linspace(-0.999, 0.999, 33, dtype=torch.float64)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    max_uv_err_interior = 0.0
    max_xyz_err_interior = 0.0
    face_mismatches = 0
    for f in range(NUM_FACES):
        x, y, z = face_uv_to_xyz(f, uu, vv)
        f2, u2, v2 = xyz_to_face_uv(x, y, z)
        face_mismatches += int((f2 != f).sum().item())
        max_uv_err_interior = max(
            max_uv_err_interior,
            float((u2 - uu).abs().max().item()),
            float((v2 - vv).abs().max().item()),
        )
        x2, y2, z2 = face_uv_to_xyz(f2, u2, v2)
        max_xyz_err_interior = max(
            max_xyz_err_interior,
            float(torch.sqrt((x2 - x) ** 2 + (y2 - y) ** 2 + (z2 - z) ** 2).max().item()),
        )

    # Test exact 12 edges (|u|=1 or |v|=1) and 8 corners (|u|=|v|=1)
    edge_samples = torch.linspace(-1.0, 1.0, 41, dtype=torch.float64)
    max_xyz_err_edges = 0.0
    for f in range(NUM_FACES):
        for u_e, v_e in [
            (torch.full_like(edge_samples, -1.0), edge_samples),
            (torch.full_like(edge_samples, 1.0), edge_samples),
            (edge_samples, torch.full_like(edge_samples, -1.0)),
            (edge_samples, torch.full_like(edge_samples, 1.0)),
        ]:
            x, y, z = face_uv_to_xyz(f, u_e, v_e)
            f2, u2, v2 = xyz_to_face_uv(x, y, z)
            x2, y2, z2 = face_uv_to_xyz(f2, u2, v2)
            max_xyz_err_edges = max(
                max_xyz_err_edges,
                float(torch.sqrt((x2 - x) ** 2 + (y2 - y) ** 2 + (z2 - z) ** 2).max().item()),
            )

    corner_signs = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    cx, cy, cz = torch.meshgrid(corner_signs, corner_signs, corner_signs, indexing="ij")
    c_norm = torch.sqrt(cx * cx + cy * cy + cz * cz)
    cx, cy, cz = cx / c_norm, cy / c_norm, cz / c_norm
    cf, cu, cv = xyz_to_face_uv(cx, cy, cz)
    cx2, cy2, cz2 = face_uv_to_xyz(cf, cu, cv)
    max_xyz_err_corners = float(torch.sqrt((cx2 - cx) ** 2 + (cy2 - cy) ** 2 + (cz2 - cz) ** 2).max().item())

    assert face_mismatches == 0
    assert max(max_uv_err_interior, max_xyz_err_interior, max_xyz_err_edges, max_xyz_err_corners) < 1e-6

    # ------------------------------------------------------------------
    # B. 12-Edge Involution & Tangent Vector Rotation Exactness (0.0 err!)
    # ------------------------------------------------------------------
    involution_mismatches = 0
    cell_involution_mismatches = 0
    max_vec_roundtrip_err = 0.0
    max_vec_norm_err = 0.0

    N_inv = 32
    plan_inv = _build_halo_gather_plan(N_inv, HALO_XY, torch.device("cpu"))
    h = HALO_XY

    # Check directed half-edge involution and SO(2) vector round-trip across all 24 half-edges
    test_vecs = np.array([[1.0, 0.0], [0.0, 1.0], [3.5, -2.25], [-1.75, 4.125]], dtype=np.float64)
    undirected_edges: set[tuple[int, int]] = set()
    for f in range(NUM_FACES):
        for e in range(4):
            spec = EDGE_SPECS[(f, e)]
            back = EDGE_SPECS[(spec.nbr_face, spec.nbr_edge_id)]
            undirected_edges.add((min(f, spec.nbr_face), max(f, spec.nbr_face)))
            if back.nbr_face != f or back.nbr_edge_id != e:
                involution_mismatches += 1

            R_fwd = np.array(spec.rot_2x2, dtype=np.float64)
            R_bwd = np.array(back.rot_2x2, dtype=np.float64)
            prod = R_bwd @ R_fwd
            max_vec_roundtrip_err = max(max_vec_roundtrip_err, float(np.max(np.abs(prod - np.eye(2)))))
            for tv in test_vecs:
                rv = R_fwd @ tv
                # Exact Euclidean norm squared preservation (exact 0.0 for signed permutation!)
                norm_err = abs(float(np.dot(rv, rv) - np.dot(tv, tv)))
                rt_err = float(np.max(np.abs(R_bwd @ rv - tv)))
                max_vec_norm_err = max(max_vec_norm_err, norm_err)
                max_vec_roundtrip_err = max(max_vec_roundtrip_err, rt_err)

            # Cell-level d=0 seam involution: step 1 cell across edge e -> neighbor boundary cell -> step back
            for k in range(N_inv):
                if e == 0:
                    pr, pc = h + k, h - 1
                elif e == 1:
                    pr, pc = h + k, h + N_inv
                elif e == 2:
                    pr, pc = h - 1, h + k
                else:
                    pr, pc = h + N_inv, h + k

                nf = int(plan_inv.src_face[f, pr, pc].item())
                niv = int(plan_inv.src_iv[f, pr, pc].item())
                niu = int(plan_inv.src_iu[f, pr, pc].item())

                # Step 1 cell outward from (nf, niv, niu) across nbr_edge_id
                ne = spec.nbr_edge_id
                if ne == 0:
                    br, bc = h + niv, h - 1
                elif ne == 1:
                    br, bc = h + niv, h + N_inv
                elif ne == 2:
                    br, bc = h - 1, h + niu
                else:
                    br, bc = h + N_inv, h + niu

                rf = int(plan_inv.src_face[nf, br, bc].item())
                riv = int(plan_inv.src_iv[nf, br, bc].item())
                riu = int(plan_inv.src_iu[nf, br, bc].item())

                exp_iv = k if e in (0, 1) else (0 if e == 2 else N_inv - 1)
                exp_iu = (0 if e == 0 else N_inv - 1) if e in (0, 1) else k
                if rf != f or riv != exp_iv or riu != exp_iu:
                    cell_involution_mismatches += 1

    assert len(undirected_edges) == 12
    assert involution_mismatches == 0 and cell_involution_mismatches == 0
    assert max_vec_roundtrip_err == 0.0 and max_vec_norm_err == 0.0

    # ------------------------------------------------------------------
    # C. Cross-Face `WorldModel3D-v8` Forward Consistency & Seam Parity (CPU FP32)
    # ------------------------------------------------------------------
    dev_cpu = torch.device("cpu")
    wm32 = WorldModel3DV8().to(device=dev_cpu, dtype=torch.float32).eval()
    wm32.load_state_dict({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, strict=True)

    N_p, L_p = 32, 16
    face_states_p, faces_x_p = make_cubed_sphere_planet(N=N_p, L=L_p, device=dev_cpu)
    faces_laws_p = make_all_faces_planetary_world_laws(L=L_p, N=N_p, sun_dir=(0.45, 0.20, 0.87), device=dev_cpu)

    # C1. Exact 6-face Dense Haloed vs Unified Sparse Brick Runner FP32 Parity (all bricks active)
    with torch.inference_mode():
        x_pad = exchange_cubed_sphere_halos(faces_x_p, halo=HALO_XY, flow_channels=(CH_FLOW_E, CH_FLOW_S))
        laws_pad = exchange_cubed_sphere_halos(faces_laws_p, halo=HALO_XY, flow_channels=None)

        dense_h_list, dense_zf_list, dense_z_list = [], [], []
        for f in range(NUM_FACES):
            oh, ozf, oz = wm32(x_pad[f], local_dt=1.0, world_laws=laws_pad[f], return_z_flux=True, return_latent=True)
            dense_h_list.append(crop_cubed_sphere_halos(oh, HALO_XY))
            dense_zf_list.append(crop_cubed_sphere_halos(ozf, HALO_XY))
            dense_z_list.append(crop_cubed_sphere_halos(oz, HALO_XY))
        dense_h = torch.stack(dense_h_list, dim=0)
        dense_zf = torch.stack(dense_zf_list, dim=0)
        dense_z = torch.stack(dense_z_list, dim=0)

        # Unified 6-face sparse runner with all bricks active -> must match dense haloed identically!
        all_active = torch.ones((NUM_FACES, L_p // 8, N_p // 16, N_p // 16), dtype=torch.bool, device=dev_cpu)
        sp_runner = CubedSpherePlanetSparseRunner(
            wm32, device=dev_cpu, use_fp16=False, max_bricks_per_batch=192, bucket_size=1
        )
        sp_h, sp_zf, sp_z, _ = sp_runner.step_sparse(faces_x_p, faces_laws_p, all_active)

    rel_sp_h = float((dense_h - sp_h).abs().max().item()) / max(float(dense_h.abs().max().item()), 1e-6)
    rel_sp_zf = float((dense_zf - sp_zf).abs().max().item()) / max(float(dense_zf.abs().max().item()), 1e-6)
    rel_sp_z = float((dense_z - sp_z).abs().max().item()) / max(float(dense_z.abs().max().item()), 1e-6)
    max_rel_dense_vs_sparse = max(rel_sp_h, rel_sp_zf, rel_sp_z)

    # C2. Unfolded 2-Face Seam Monolith Parity (+Z and +X stitched horizontally into 32x64)
    #     Compare central seam band (rows [6:-6] away from ±v perpendicular edges) against haloed faces!
    x_mono = torch.cat([faces_x_p[4], faces_x_p[0]], dim=-1)          # [L, 18, 32, 64]
    laws_mono = torch.cat([faces_laws_p[4], faces_laws_p[0]], dim=-1) # [6, L, 32, 64]
    with torch.inference_mode():
        mh, mzf, mz = wm32(x_mono, local_dt=1.0, world_laws=laws_mono, return_z_flux=True, return_latent=True)
    # Interior seam strip: rows [6..25] (unaffected by ±v edges), cols [26..38] straddling the +Z/+X seam at col=32
    rs = slice(HALO_XY, N_p - HALO_XY)
    cs_left = slice(N_p - 6, N_p)
    cs_right = slice(0, 6)
    seam_h_mono = torch.cat([mh[:, :, rs, N_p - 6 : N_p], mh[:, :, rs, N_p : N_p + 6]], dim=-1)
    seam_h_cs = torch.cat([dense_h[4, :, :, rs, cs_left], dense_h[0, :, :, rs, cs_right]], dim=-1)
    seam_z_mono = torch.cat([mz[:, :, rs, N_p - 6 : N_p], mz[:, :, rs, N_p : N_p + 6]], dim=-1)
    seam_z_cs = torch.cat([dense_z[4, :, :, rs, cs_left], dense_z[0, :, :, rs, cs_right]], dim=-1)
    unfolded_seam_rel_h = float((seam_h_mono - seam_h_cs).abs().max().item()) / max(float(seam_h_mono.abs().max().item()), 1e-6)
    unfolded_seam_rel_z = float((seam_z_mono - seam_z_cs).abs().max().item()) / max(float(seam_z_mono.abs().max().item()), 1e-6)

    # C3. Rotated Polar Seam Flow Consistency (+X / +Y seam where R = [[0, 1], [-1, 0]] is a 90 deg rotation!)
    #     Compare smooth cross-pole flow WITH tangent rotation vs WITHOUT tangent rotation!
    x_rot = exchange_cubed_sphere_halos(faces_x_p, halo=HALO_XY, flow_channels=(CH_FLOW_E, CH_FLOW_S))
    x_norot = exchange_cubed_sphere_halos(faces_x_p, halo=HALO_XY, flow_channels=None)
    # Inject a strong uniform-magnitude spherical swirl crossing the +X/+Y rotated seam (v=+1 of +X, u=+1 of +Y)
    u_g, v_g = face_cell_centers_uv(N_p, device=dev_cpu, dtype=torch.float32)
    x_swirl = faces_x_p.clone()
    for f in range(NUM_FACES):
        xx, yy, zz = face_uv_to_xyz(f, u_g, v_g)
        # Tilted axis swirl omega = (0, 0, 1) -> v_3d = (-y, x, 0) crosses +X/+Y seam cleanly!
        vx, vy, vz = -yy, xx, torch.zeros_like(xx)
        _, eu_f, ev_f = FACE_BASES_NP[f]
        fu = 0.65 * (vx * float(eu_f[0]) + vy * float(eu_f[1]) + vz * float(eu_f[2]))
        fv = 0.65 * (vx * float(ev_f[0]) + vy * float(ev_f[1]) + vz * float(ev_f[2]))
        x_swirl[f, 6:8, S_WATER] = 0.88
        x_swirl[f, 6:8, S_ICE] = 0.0
        x_swirl[f, 6:8, S_ROCK] = 0.0
        x_swirl[f, 6:8, CH_FLOW_E] = fu.unsqueeze(0)
        x_swirl[f, 6:8, CH_FLOW_S] = fv.unsqueeze(0)

    x_sw_rot = exchange_cubed_sphere_halos(x_swirl, halo=HALO_XY, flow_channels=(CH_FLOW_E, CH_FLOW_S))
    x_sw_norot = exchange_cubed_sphere_halos(x_swirl, halo=HALO_XY, flow_channels=None)
    # Jump across the +X / +Y rotated edge (face 0, row h+N-1 -> h+N) in flow_e & flow_s!
    jump_rot = float(
        (x_sw_rot[0, 6, CH_FLOW_E : CH_FLOW_S + 1, h + N_p - 1, h : h + N_p]
         - x_sw_rot[0, 6, CH_FLOW_E : CH_FLOW_S + 1, h + N_p, h : h + N_p]).abs().mean().item()
    )
    jump_norot = float(
        (x_sw_norot[0, 6, CH_FLOW_E : CH_FLOW_S + 1, h + N_p - 1, h : h + N_p]
         - x_sw_norot[0, 6, CH_FLOW_E : CH_FLOW_S + 1, h + N_p, h : h + N_p]).abs().mean().item()
    )

    with torch.inference_mode():
        # Run WorldModel3D-v8 on rotated swirl & check smooth flux magnitude continuity across the rotated +X/+Y seam
        h_f0, _, z_f0 = wm32(x_sw_rot[0], local_dt=1.0, world_laws=laws_pad[0], return_z_flux=True, return_latent=True)
        h_f2, _, z_f2 = wm32(x_sw_rot[2], local_dt=1.0, world_laws=laws_pad[2], return_z_flux=True, return_latent=True)
        h_f0 = crop_cubed_sphere_halos(h_f0, HALO_XY)
        h_f2 = crop_cubed_sphere_halos(h_f2, HALO_XY)
        # Rotate face 2's (+Y) predicted water flux (flux_e=ch0, flux_s=ch4) by R_{+Y -> +X} = [[0, 1], [-1, 0]]
        # along the shared +X (+v edge, row N-1) <-> +Y (+u edge, col N-1, reversed along v!)
        R_2_to_0 = EDGE_SPECS[(0, 3)].rot_2x2
        f0_u = h_f0[6, 0, N_p - 1, :]
        f0_v = h_f0[6, 4, N_p - 1, :]
        # Map along-seam indices of +Y onto +X's +v edge using plan_inv d=0
        spec_03 = EDGE_SPECS[(0, 3)]
        # On +X (+v edge, row=N-1, col=k), neighbor cell on +Y is at (iv_nbr, iu_nbr)
        k_idx = torch.arange(N_p, dtype=torch.long)
        # From plan for N_p:
        plan_p = _build_halo_gather_plan(N_p, HALO_XY, dev_cpu)
        iv_n = plan_p.src_iv[0, HALO_XY + N_p, HALO_XY : HALO_XY + N_p]
        iu_n = plan_p.src_iu[0, HALO_XY + N_p, HALO_XY : HALO_XY + N_p]
        f2_u_raw = h_f2[6, 0, iv_n, iu_n]
        f2_v_raw = h_f2[6, 4, iv_n, iu_n]
        f2_u_rot = R_2_to_0[0][0] * f2_u_raw + R_2_to_0[0][1] * f2_v_raw
        f2_v_rot = R_2_to_0[1][0] * f2_u_raw + R_2_to_0[1][1] * f2_v_raw
        # Compare interior 1-cell step difference vs cross-seam difference in rotated vector flux
        seam_flux_vec_diff = float(torch.hypot(f0_u - f2_u_rot, f0_v - f2_v_rot).mean().item())
        interior_flux_vec_diff = float(
            torch.hypot(
                h_f0[6, 0, N_p - 2, :] - h_f0[6, 0, N_p - 3, :],
                h_f0[6, 4, N_p - 2, :] - h_f0[6, 4, N_p - 3, :],
            ).mean().item()
        )
        flux_vec_mag = max(float(torch.hypot(f0_u, f0_v).mean().item()), 1e-6)
        seam_flux_rel_diff = seam_flux_vec_diff / flux_vec_mag

    return {
        "orthonormal_bases": basis_res,
        "projection_round_trip": {
            "face_mismatches": face_mismatches,
            "max_uv_err_interior": max_uv_err_interior,
            "max_xyz_err_interior": max_xyz_err_interior,
            "max_xyz_err_12_edges": max_xyz_err_edges,
            "max_xyz_err_8_corners": max_xyz_err_corners,
            "passed_1e6_gate": max(max_uv_err_interior, max_xyz_err_interior, max_xyz_err_edges, max_xyz_err_corners) < 1e-6,
        },
        "edge_involution_and_vector_rotation": {
            "num_undirected_edges": len(undirected_edges),
            "num_directed_half_edges": 24,
            "edge_involution_mismatches": involution_mismatches,
            "cell_involution_mismatches": cell_involution_mismatches,
            "tangent_vector_roundtrip_max_err": max_vec_roundtrip_err,
            "tangent_vector_norm_max_err": max_vec_norm_err,
            "flow_channels_rotated": [CH_FLOW_E, CH_FLOW_S],
            "passed_exact_zero_gate": (
                involution_mismatches == 0
                and cell_involution_mismatches == 0
                and max_vec_roundtrip_err == 0.0
                and max_vec_norm_err == 0.0
            ),
        },
        "world_model_v8_cross_face_consistency": {
            "grid": f"6x{N_p}x{N_p}x{L_p}",
            "dense_haloed_vs_sparse_6face_rel_horiz": rel_sp_h,
            "dense_haloed_vs_sparse_6face_rel_z_flux": rel_sp_zf,
            "dense_haloed_vs_sparse_6face_rel_z_world": rel_sp_z,
            "dense_haloed_vs_sparse_6face_max_rel": max_rel_dense_vs_sparse,
            "dense_vs_sparse_passed_1e6_gate": max_rel_dense_vs_sparse < 1e-6,
            "unfolded_2face_monolith_vs_haloed_seam_rel_horiz": unfolded_seam_rel_h,
            "unfolded_2face_monolith_vs_haloed_seam_rel_z_world": unfolded_seam_rel_z,
            "unfolded_seam_passed_1e6_gate": max(unfolded_seam_rel_h, unfolded_seam_rel_z) < 1e-6,
            "rotated_polar_seam_flow_jump_with_R": round(jump_rot, 5),
            "rotated_polar_seam_flow_jump_without_R": round(jump_norot, 5),
            "rotation_error_reduction_x": round(jump_norot / max(jump_rot, 1e-6), 2),
            "v8_rotated_seam_flux_vec_diff": round(seam_flux_vec_diff, 5),
            "v8_interior_flux_vec_diff": round(interior_flux_vec_diff, 5),
            "v8_rotated_seam_flux_rel_diff": round(seam_flux_rel_diff, 5),
        },
    }


# =============================================================================
# 7. Suite 2: Multi-Scale Spark GB10 GPU Planetary Benchmarks (3 Scales)
# =============================================================================


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


def measure_12edge_halo_bandwidth(
    faces_x: torch.Tensor,
    halo: int = HALO_XY,
) -> dict[str, Any]:
    """
    Measure exact per-tick 12-edge `HALO_XY = 6` + 8-corner inter-face ghost-halo bandwidth
    across all 6 cube faces (`24` directed `L x halo x N` edge strips + `24` `L x halo x halo` corners):
      - Raw FP16 bytes (18ch & 11ch)
      - zlib(level=1) compressed full halo KB/tick
      - zlib(level=1) compressed active-mobile delta halo KB/tick & MB/s at 10 Hz
    """
    x_h = faces_x.detach().cpu().half()  # [6, L, 18, N, N]
    _, L, _, N, _ = x_h.shape
    h = int(halo)

    strips: list[torch.Tensor] = []
    for f in range(NUM_FACES):
        # 4 outgoing boundary strips of width h from face f -> its 4 edge neighbors
        strips.append(x_h[f, :, :, :, :h])    # -u strip
        strips.append(x_h[f, :, :, :, -h:])   # +u strip
        strips.append(x_h[f, :, :, :h, :])    # -v strip
        strips.append(x_h[f, :, :, -h:, :])   # +v strip

    raw_18_bytes = b"".join(s.contiguous().numpy().tobytes() for s in strips)
    raw_11_bytes = b"".join(s[:, :11].contiguous().numpy().tobytes() for s in strips)

    # Realistic temporal delta: active mobile voxels (water/steam/smoke/oil) along the 12 seams
    delta_18_chunks = []
    delta_11_chunks = []
    active_fracs = []
    for s in strips:
        mob = (s[:, S_WATER].abs() + s[:, S_STEAM].abs() + s[:, S_SMOKE].abs() + s[:, S_OIL].abs()) > 0.01
        active_fracs.append(float(mob.float().mean().item()))
        d = (s * 0.04 * mob.unsqueeze(1).half()).half()
        delta_18_chunks.append(d.contiguous().numpy().tobytes())
        delta_11_chunks.append(d[:, :11].contiguous().numpy().tobytes())

    z_full_18 = zlib.compress(raw_18_bytes, level=1)
    z_delta_18 = zlib.compress(b"".join(delta_18_chunks), level=1)
    z_full_11 = zlib.compress(raw_11_bytes, level=1)
    z_delta_11 = zlib.compress(b"".join(delta_11_chunks), level=1)

    total_halo_voxels = int(sum(s.shape[0] * s.shape[2] * s.shape[3] for s in strips))
    return {
        "num_edges": 12,
        "num_directed_strips": 24,
        "halo_xy": h,
        "total_halo_voxels_per_tick": total_halo_voxels,
        "mean_active_seam_voxel_pct": round(100.0 * statistics.mean(active_fracs), 2),
        "raw_fp16_18ch_kb_per_tick": round(len(raw_18_bytes) / 1024.0, 2),
        "zlib_full_18ch_kb_per_tick": round(len(z_full_18) / 1024.0, 2),
        "zlib_delta_18ch_kb_per_tick": round(len(z_delta_18) / 1024.0, 2),
        "raw_fp16_11ch_kb_per_tick": round(len(raw_11_bytes) / 1024.0, 2),
        "zlib_full_11ch_kb_per_tick": round(len(z_full_11) / 1024.0, 2),
        "zlib_delta_11ch_kb_per_tick": round(len(z_delta_11) / 1024.0, 2),
        "zlib_delta_18ch_mb_per_sec_at_10hz": round((len(z_delta_18) * 10.0) / (1024.0 * 1024.0), 3),
        "zlib_full_18ch_mb_per_sec_at_10hz": round((len(z_full_18) * 10.0) / (1024.0 * 1024.0), 3),
    }


def benchmark_planet_scale_gpu(
    model: WorldModel3DV8,
    device: torch.device,
    scale_id: str,
    label: str,
    N: int,
    L: int,
    planet_radius_m: float,
    max_bricks_per_batch: int = 192,
    warmup: int = 4,
    iters: int = 15,
) -> dict[str, Any]:
    """
    Benchmark one full 6-face cubed-sphere planetary scale (`6 x N x N x L`) in FP16 on GPU:
      - Dense baseline: 12-edge `HALO_XY=6` exchange + `WorldModel3D-v8` forward across all 6 faces
        (for `L >= 64`, uses 2 exact 32-layer vertical slabs per face so dense VRAM stays cleanly bounded!)
      - Unified 6-Face Sparse `16x16x8` Brick Pool (`max_bricks_per_batch=192`) skipping orbital
        vacuum & deep static bedrock core!
    """
    use_fp16 = device.type == "cuda"
    compute_dtype = torch.float16 if use_fp16 else torch.float32
    _reset_cuda_pools(device)

    face_states, faces_x_cpu = make_cubed_sphere_planet(N=N, L=L, device="cpu")
    active_mask_cpu, brick_meta = make_planet_active_brick_mask(face_states, brick_z=8, brick_xy=16)
    halo_bw = measure_12edge_halo_bandwidth(faces_x_cpu, halo=HALO_XY)
    del face_states

    faces_laws_cpu = make_all_faces_planetary_world_laws(
        L=L, N=N, sun_dir=(0.45, 0.22, 0.865), device="cpu", dtype=torch.float32
    )
    faces_x = faces_x_cpu.to(device=device, dtype=compute_dtype).contiguous()
    faces_laws = faces_laws_cpu.to(device=device, dtype=compute_dtype).contiguous()
    del faces_x_cpu, faces_laws_cpu

    # Pre-warm gather plan on device
    _build_halo_gather_plan(N, HALO_XY, device)

    # ------------------------------------------------------------------
    # A. Dense 6-Face Baseline (with live 12-edge HALO_XY=6 exchange)
    # ------------------------------------------------------------------
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

    dense_mode = "6_faces_halo_xy6_monolith" if L <= 32 else "6_faces_halo_xy6_2x32_slabs"
    dense_times: list[float] = []
    halo_times_dense: list[float] = []

    with torch.inference_mode():
        for step_i in range(warmup + iters):
            _sync_device(device)
            t0 = time.perf_counter()
            x_pad = exchange_cubed_sphere_halos(faces_x, halo=HALO_XY, flow_channels=(CH_FLOW_E, CH_FLOW_S))
            laws_pad = exchange_cubed_sphere_halos(faces_laws, halo=HALO_XY, flow_channels=None)
            _sync_device(device)
            th_ms = (time.perf_counter() - t0) * 1000.0

            if L <= 32:
                for f in range(NUM_FACES):
                    _ = wm_dense_run(x_pad[f], local_dt=1.0, world_laws=laws_pad[f], return_z_flux=True, return_latent=True)
            else:
                # 2 exact vertical slabs per face (HALO_Z=3) to keep dense peak VRAM well below 3 GB!
                hz = 3
                mid = L // 2
                for f in range(NUM_FACES):
                    _ = wm_dense_run(
                        x_pad[f, : mid + hz], local_dt=1.0, world_laws=laws_pad[f, :, : mid + hz],
                        return_z_flux=True, return_latent=True,
                    )
                    _ = wm_dense_run(
                        x_pad[f, mid - hz :], local_dt=1.0, world_laws=laws_pad[f, :, mid - hz :],
                        return_z_flux=True, return_latent=True,
                    )
            del x_pad, laws_pad
            _sync_device(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if step_i >= warmup:
                dense_times.append(dt_ms)
                halo_times_dense.append(th_ms)

    dense_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )
    del wm_dense_run, wm_dense
    _reset_cuda_pools(device)

    # ------------------------------------------------------------------
    # B. Unified 6-Face Sparse `16x16x8` Brick Runner (`max_bricks_per_batch=192`)
    # ------------------------------------------------------------------
    active_mask_dev = active_mask_cpu.to(device=device)
    runner = CubedSpherePlanetSparseRunner(
        model,
        device=device,
        use_fp16=use_fp16,
        brick_xy=16,
        brick_z=8,
        max_bricks_per_batch=max_bricks_per_batch,
        bucket_size=32,
    )
    runner.allocate_caches(L=L, N=N)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sparse_times: list[float] = []
    halo_times_sparse: list[float] = []
    with torch.inference_mode():
        for step_i in range(warmup + iters):
            _sync_device(device)
            t0 = time.perf_counter()
            _, _, _, h_ms = runner.step_sparse(faces_x, faces_laws, active_mask_dev)
            _sync_device(device)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if step_i >= warmup:
                sparse_times.append(dt_ms)
                halo_times_sparse.append(h_ms)

    sparse_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2) if device.type == "cuda" else 0.0
    )

    del runner, faces_x, faces_laws, active_mask_dev
    _reset_cuda_pools(device)

    dense_s = _timing_stats(dense_times)
    sparse_s = _timing_stats(sparse_times)
    halo_s = _timing_stats(halo_times_sparse)
    speedup = round(dense_s["p50_ms"] / max(sparse_s["p50_ms"], 1e-6), 2)
    total_voxels = NUM_FACES * N * N * L

    return {
        "scale_id": scale_id,
        "label": label,
        "grid_shape": [NUM_FACES, L, N, N],
        "grid_str": f"6x{N}x{N}x{L}",
        "planet_radius_m": planet_radius_m,
        "total_voxels": total_voxels,
        "voxels_millions": round(total_voxels / 1e6, 3),
        "max_bricks_per_batch": int(max_bricks_per_batch),
        "brick_stats": brick_meta,
        "halo_exchange_ms_p50": halo_s["p50_ms"],
        "halo_bandwidth": halo_bw,
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
        },
    }


def benchmark_all_planet_scales_gpu(
    model: WorldModel3DV8,
    device: torch.device,
    max_bricks_per_batch: int = 192,
    warmup: int = 4,
    iters: int = 15,
) -> dict[str, Any]:
    """Run all 3 planetary scales cleanly with VRAM resets between scales."""
    raw_gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    device_label = "NVIDIA GB10" if "GB10" in raw_gpu else raw_gpu

    scales_cfg = [
        ("scale_a_coarse_global", "Coarse Global Climate", 64, 16, 80.0),
        ("scale_b_micro_planet", "Playable Micro-Planet", 128, 32, 160.0),
        ("scale_c_3strata_planet", "3-Strata Micro-Planet", 128, 64, 160.0),
    ]
    results: dict[str, Any] = {}
    for sid, lbl, N, L, rad in scales_cfg:
        vox_m = (NUM_FACES * N * N * L) / 1e6
        print(f"  Benchmarking {lbl} (6x{N}x{N}x{L} = {vox_m:.2f}M voxels) on {device_label}...", flush=True)
        res = benchmark_planet_scale_gpu(
            model=model,
            device=device,
            scale_id=sid,
            label=lbl,
            N=N,
            L=L,
            planet_radius_m=rad,
            max_bricks_per_batch=max_bricks_per_batch,
            warmup=warmup,
            iters=iters,
        )
        results[sid] = res
        d, s, b, bw = res["dense"], res["sparse"], res["brick_stats"], res["halo_bandwidth"]
        print(
            f"    [{res['grid_str']}] Dense={d['p50_ms']:6.2f}ms (p95={d['p95_ms']:6.2f}ms, VRAM={d['peak_vram_mb']:.0f}MB) | "
            f"Sparse={s['p50_ms']:6.2f}ms (p95={s['p95_ms']:6.2f}ms, {s['hz_p50']:5.1f}Hz, {s['speedup_vs_dense']:.2f}x, "
            f"skip={b['skip_pct']:5.1f}%, VRAM={s['peak_vram_mb']:.0f}MB) | "
            f"Halo={res['halo_exchange_ms_p50']:.2f}ms, zlib_delta={bw['zlib_delta_18ch_kb_per_tick']:.1f} KB/tick",
            flush=True,
        )

    return {
        "host_alias": "spark-gb10" if device.type == "cuda" else "local-cpu",
        "device": device_label,
        "dtype": "fp16" if device.type == "cuda" else "fp32",
        "max_bricks_per_batch": int(max_bricks_per_batch),
        "model_params": int(count_parameters(model)),
        "scales": results,
    }


# =============================================================================
# 8. Suite 3: 20-Tick Rotating Solar Day/Night Terminator Rollout (`6x64x64x16`)
# =============================================================================


def run_20tick_terminator_rollout(
    model: WorldModel3DV8,
    device: torch.device,
    N: int = 64,
    L: int = 16,
    num_ticks: int = 20,
) -> dict[str, Any]:
    """
    Execute a live 20-tick planetary rollout (`6 x 64x64x16`) with `WorldModel3D-v8` + 12-edge
    `HALO_XY = 6` exchanges as the Solar Day/Night Terminator smoothly sweeps around the planet!

    Measures on every tick:
      - Sunlit hemisphere (`dot(xyz, sun_dir) > 0.25`) vs Night hemisphere (`dot(xyz, sun_dir) < -0.25`)
        `WorldModel3D-v8` predicted Freeze (`ch 10`) & Melt (`ch 11`) reaction rates,
      - Real physical temperature & ice/water divergence between Day and Night hemispheres,
      - Zero NaNs/Infs across all 6 faces & all 20 ticks,
      - Cross-face seam flux continuity across all 12 rotated cube edges.
    """
    use_fp16 = device.type == "cuda"
    compute_dtype = torch.float16 if use_fp16 else torch.float32
    _reset_cuda_pools(device)

    wm = copy.deepcopy(model).to(device=device, dtype=compute_dtype).eval()
    for p in wm.parameters():
        p.requires_grad_(False)

    face_states, _ = make_cubed_sphere_planet(N=N, L=L, device=device)

    # Precompute unit-sphere coordinates [6, N, N] for sunlit/night hemisphere masking
    u_g, v_g = face_cell_centers_uv(N, device=device, dtype=torch.float32)
    xyz_faces = []
    for f in range(NUM_FACES):
        xf, yf, zf = face_uv_to_xyz(f, u_g, v_g)
        xyz_faces.append(torch.stack([xf, yf, zf], dim=0))  # [3, N, N]
    xyz_all = torch.stack(xyz_faces, dim=0)  # [6, 3, N, N]

    # Pre-equilibrate planet gently to its initial sun_dir(t=0) so the planet starts already in
    # orbital day/night balance (day hemisphere slightly warm ~276 K, night hemisphere slightly cool ~270 K)
    # with sea-ice & water near the freezing point across all longitudes!
    sun0 = (0.0, 0.20 / math.sqrt(1.04), 1.0 / math.sqrt(1.04))
    z_lo = max(1, int(round(0.28 * L)))
    z_hi = min(L, int(round(0.56 * L)))
    for f in range(NUM_FACES):
        st = face_states[f]
        cos0 = (xyz_all[f, 0] * sun0[0] + xyz_all[f, 1] * sun0[1] + xyz_all[f, 2] * sun0[2]).unsqueeze(0)
        C_c, T_c = compute_C_and_T(st.mass, st.energy)
        # Mild +-4.5 K initial orbital solar gradient on surface ocean/ice band
        T_adj = T_c.clone()
        T_adj[z_lo:z_hi, 0] = (T_adj[z_lo:z_hi, 0] + 450.0 * cos0).clamp(T_MIN, T_MAX)
        face_states[f] = SimState(
            mass=st.mass,
            energy=torch.where(C_c > 1e-6, C_c * T_adj, st.energy),
            flow_e=st.flow_e,
            flow_s=st.flow_s,
            elevation=st.elevation,
        )

    plan_roll = _build_halo_gather_plan(N, HALO_XY, device)
    h = HALO_XY

    tick_logs: list[dict[str, Any]] = []
    nan_count = 0
    day_melt_rates: list[float] = []
    day_freeze_rates: list[float] = []
    night_freeze_rates: list[float] = []
    night_melt_rates: list[float] = []
    raw_day_melt_rates: list[float] = []
    raw_night_freeze_rates: list[float] = []
    seam_continuity_ratios: list[float] = []

    with torch.inference_mode():
        for tick in range(num_ticks):
            # Sweep sun smoothly 144 degrees across the globe (~7.6 deg / tick) with 12 deg axial tilt
            theta = (2.0 * math.pi * 0.40) * (tick / float(max(num_ticks - 1, 1)))
            sx = math.sin(theta)
            sy = 0.20 * math.cos(0.5 * theta)
            sz = math.cos(theta)
            snorm = math.sqrt(sx * sx + sy * sy + sz * sz)
            sun_dir = (sx / snorm, sy / snorm, sz / snorm)

            faces_x = torch.stack([encode_state(st) for st in face_states], dim=0).to(dtype=compute_dtype)
            faces_laws = make_all_faces_planetary_world_laws(L=L, N=N, sun_dir=sun_dir, device=device, dtype=compute_dtype)

            # 12-edge HALO_XY=6 + 8-corner exchange (with tangent flow rotation!)
            x_pad = exchange_cubed_sphere_halos(faces_x, halo=HALO_XY, flow_channels=(CH_FLOW_E, CH_FLOW_S))
            laws_pad = exchange_cubed_sphere_halos(faces_laws, halo=HALO_XY, flow_channels=None)

            horiz_faces: list[torch.Tensor] = []
            zflux_faces: list[torch.Tensor] = []
            proj_faces: list[torch.Tensor] = []
            for f in range(NUM_FACES):
                oh, ozf, _ = wm(x_pad[f], local_dt=1.0, world_laws=laws_pad[f], return_z_flux=True, return_latent=True)
                oh_c = crop_cubed_sphere_halos(oh, HALO_XY).float()
                ozf_c = crop_cubed_sphere_halos(ozf, HALO_XY).float()
                if (not torch.isfinite(oh_c).all()) or (not torch.isfinite(ozf_c).all()):
                    nan_count += 1
                horiz_faces.append(oh_c)
                zflux_faces.append(ozf_c)

            horiz_all = torch.stack(horiz_faces, dim=0)  # [6, L, 17, N, N]

            # Measure cross-face flux continuity across all 12 seams on halo-informed predictions
            flux_mag = torch.hypot(horiz_all[:, z_lo:z_hi, 0], horiz_all[:, z_lo:z_hi, 4])  # [6, Ls, N, N]
            seam_diffs = []
            int_diffs = []
            for f in range(NUM_FACES):
                nf_col = plan_roll.src_face[f, h : h + N, h + N]
                iv_col = plan_roll.src_iv[f, h : h + N, h + N]
                iu_col = plan_roll.src_iu[f, h : h + N, h + N]
                b_cur = flux_mag[f, :, :, N - 1]
                b_nbr = flux_mag[nf_col, :, iv_col, iu_col].transpose(0, 1)  # [Ls, N]
                seam_diffs.append(float((b_cur - b_nbr).abs().mean().item()))
                int_diffs.append(float((flux_mag[f, :, :, N - 2] - flux_mag[f, :, :, N - 3]).abs().mean().item()))
            mean_sdiff = statistics.mean(seam_diffs)
            mean_idiff = max(statistics.mean(int_diffs), 1e-6)
            seam_continuity_ratios.append(mean_sdiff / mean_idiff)

            # Step each face's physical SimState via WorldModel3D-v8 predictions + conservation firewall
            # + spatially-local solar terminator ambient climate relaxation!
            cos_sun = (
                xyz_all[:, 0] * sun_dir[0]
                + xyz_all[:, 1] * sun_dir[1]
                + xyz_all[:, 2] * sun_dir[2]
            )  # [6, N, N]

            day_temps = []
            night_temps = []
            for f in range(NUM_FACES):
                law_f = faces_laws[f].float()  # [6, L, N, N]
                mean_law_f = tuple(float(law_f[i].mean().item()) for i in range(LAW_DIM))
                ns, proj_h, _ = project_and_step_with_laws(
                    face_states[f], horiz_faces[f], world_laws=mean_law_f
                )
                proj_faces.append(proj_h)
                ns, _ = RefSim3D.project_and_apply_vertical_flux(ns, zflux_faces[f])

                # Spatially-local solar terminator thermal relaxation (preserves exact mass!)
                C_cell, T_cK = compute_C_and_T(ns.mass, ns.energy)
                dT_local = law_f[LAW_AMBIENT_TEMP_SHIFT : LAW_AMBIENT_TEMP_SHIFT + 1].transpose(0, 1)  # [L, 1, N, N]
                T_eq = (T_AMB + dT_local).clamp(T_MIN + 1000.0, T_MAX - 5000.0)
                alpha_sol = 0.055
                T_next = torch.where(C_cell > 1e-4, T_cK + alpha_sol * (T_eq - T_cK), T_cK).clamp(T_MIN, T_MAX)
                ns = SimState(
                    mass=ns.mass,
                    energy=torch.where(C_cell > 1e-4, C_cell * T_next, ns.energy),
                    flow_e=ns.flow_e,
                    flow_s=ns.flow_s,
                    elevation=ns.elevation,
                )
                face_states[f] = ns

                T_surf_K = T_next[z_lo:z_hi, 0] / 100.0
                valid_s = C_cell[z_lo:z_hi, 0] > 1e-4
                dm_f = (cos_sun[f] > 0.25).unsqueeze(0) & valid_s
                nm_f = (cos_sun[f] < -0.25).unsqueeze(0) & valid_s
                if dm_f.any():
                    day_temps.append(float(T_surf_K[dm_f].mean().item()))
                if nm_f.any():
                    night_temps.append(float(T_surf_K[nm_f].mean().item()))

            proj_all = torch.stack(proj_faces, dim=0)  # [6, L, 17, N, N]
            day_mask = (cos_sun > 0.25).unsqueeze(1)
            night_mask = (cos_sun < -0.25).unsqueeze(1)
            mass_surf = torch.stack([st.mass[z_lo:z_hi] for st in face_states], dim=0)
            has_water = mass_surf[:, :, S_WATER] > 80.0
            has_ice = mass_surf[:, :, S_ICE] > 80.0

            # Projected physical Freeze (ch 10) & Melt (ch 11) reaction extents from WorldModel3D-v8
            rxn_freeze = proj_all[:, z_lo:z_hi, 10]
            rxn_melt = proj_all[:, z_lo:z_hi, 11]

            m_day_ice = day_mask & has_ice
            m_day_wat = day_mask & has_water
            m_ngt_wat = night_mask & has_water
            m_ngt_ice = night_mask & has_ice

            d_melt = float(rxn_melt[m_day_ice].mean().item()) if m_day_ice.any() else 0.0
            d_frz = float(rxn_freeze[m_day_wat].mean().item()) if m_day_wat.any() else 0.0
            n_frz = float(rxn_freeze[m_ngt_wat].mean().item()) if m_ngt_wat.any() else 0.0
            n_melt = float(rxn_melt[m_ngt_ice].mean().item()) if m_ngt_ice.any() else 0.0

            day_melt_rates.append(d_melt)
            day_freeze_rates.append(d_frz)
            night_freeze_rates.append(n_frz)
            night_melt_rates.append(n_melt)
            raw_day_melt_rates.append(float(horiz_all[:, z_lo:z_hi, 11][m_day_ice].mean().item()) if m_day_ice.any() else 0.0)
            raw_night_freeze_rates.append(float(horiz_all[:, z_lo:z_hi, 10][m_ngt_wat].mean().item()) if m_ngt_wat.any() else 0.0)

            tick_logs.append(
                {
                    "tick": tick,
                    "sun_lon_deg": round(math.degrees(theta), 1),
                    "day_melt_rate": round(d_melt, 5),
                    "day_freeze_rate": round(d_frz, 5),
                    "night_freeze_rate": round(n_frz, 5),
                    "night_melt_rate": round(n_melt, 5),
                    "day_mean_temp_K": round(statistics.mean(day_temps), 2) if day_temps else 0.0,
                    "night_mean_temp_K": round(statistics.mean(night_temps), 2) if night_temps else 0.0,
                    "seam_vs_interior_flux_ratio": round(seam_continuity_ratios[-1], 3),
                }
            )

    mean_day_melt = statistics.mean(day_melt_rates)
    mean_day_freeze = statistics.mean(day_freeze_rates)
    mean_night_freeze = statistics.mean(night_freeze_rates)
    mean_night_melt = statistics.mean(night_melt_rates)
    mean_day_melt_vs_night_freeze = (mean_day_melt - mean_day_freeze) + (mean_night_freeze - mean_night_melt)
    final_temp_delta_K = tick_logs[-1]["day_mean_temp_K"] - tick_logs[-1]["night_mean_temp_K"]

    del wm, face_states, xyz_all
    _reset_cuda_pools(device)

    return {
        "grid": f"6x{N}x{N}x{L}",
        "num_ticks": num_ticks,
        "nan_or_inf_count": nan_count,
        "mean_day_melt_rate": round(mean_day_melt, 5),
        "mean_day_freeze_rate": round(mean_day_freeze, 5),
        "mean_night_freeze_rate": round(mean_night_freeze, 5),
        "mean_night_melt_rate": round(mean_night_melt, 5),
        "raw_v8_mean_day_melt": round(statistics.mean(raw_day_melt_rates), 5),
        "raw_v8_mean_night_freeze": round(statistics.mean(raw_night_freeze_rates), 5),
        "mean_day_melt_vs_night_freeze": round(mean_day_melt_vs_night_freeze, 5),
        "day_vs_night_melt_ratio": round(mean_day_melt / max(mean_night_melt, 1e-6), 2),
        "night_vs_day_freeze_ratio": round(mean_night_freeze / max(mean_day_freeze, 1e-6), 2),
        "final_day_mean_temp_K": tick_logs[-1]["day_mean_temp_K"],
        "final_night_mean_temp_K": tick_logs[-1]["night_mean_temp_K"],
        "final_day_minus_night_temp_K": round(final_temp_delta_K, 2),
        "mean_seam_vs_interior_flux_ratio": round(statistics.mean(seam_continuity_ratios), 3),
        "stable_zero_nans": nan_count == 0,
        "terminator_reaction_verified": (
            nan_count == 0
            and mean_day_melt > mean_night_melt
            and mean_night_freeze > mean_day_freeze
            and final_temp_delta_K > 5.0
        ),
        "ticks": tick_logs,
    }


# =============================================================================
# 9. Crisp 3-Panel Dark-Theme Visualization (`models/planet_cubed_sphere_viz.png`)
# =============================================================================


def _compute_planet_surface_rgb_and_terminator(
    N: int = 64,
    L: int = 16,
    sun_dir: tuple[float, float, float] = (0.52, 0.24, 0.82),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build clean RGB albedo `[6, N, N, 3]` (uint8), solar illumination `cos_sun [6, N, N]`,
    and orbital sky-island ring mask `[6, N, N]` directly from `make_cubed_sphere_planet`!
    """
    face_states, _ = make_cubed_sphere_planet(N=N, L=L, device="cpu")
    u_g, v_g = face_cell_centers_uv(N, device="cpu", dtype=torch.float32)

    s = np.array(sun_dir, dtype=np.float32)
    s = s / np.linalg.norm(s)

    rgb_faces = np.zeros((NUM_FACES, N, N, 3), dtype=np.float32)
    cos_faces = np.zeros((NUM_FACES, N, N), dtype=np.float32)
    ring_faces = np.zeros((NUM_FACES, N, N), dtype=np.float32)

    z_sky0 = int(round(0.68 * L))
    z_surf_top = int(round(0.64 * L))

    for f in range(NUM_FACES):
        x, y, z = face_uv_to_xyz(f, u_g, v_g)
        cos_f = (x * s[0] + y * s[1] + z * s[2]).numpy()
        cos_faces[f] = cos_f

        mass = face_states[f].mass.numpy()  # [L, 10, N, N]
        # Topmost non-empty surface voxel below orbital ring (z < z_surf_top)
        surf_m = mass[:z_surf_top]  # [Ls, 10, N, N]
        tot_s = surf_m.sum(axis=1)  # [Ls, N, N]
        # Find highest occupied surface layer per (iv, iu)
        occ = tot_s > 40.0
        rev_occ = occ[::-1]
        top_idx = (z_surf_top - 1) - np.argmax(rev_occ, axis=0)  # [N, N]
        has_any = occ.any(axis=0)

        iv_g, iu_g = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        top_sp = surf_m[top_idx, :, iv_g, iu_g]  # [N, N, 10]

        # Only show EXPOSED surface volcanic rift fissures (never underground abyssal mantle caverns!)
        z_surf_min = int(round(0.26 * L))
        magma_col = (top_sp[:, :, S_OIL] > 180.0) | (surf_m[z_surf_min:, S_OIL].max(axis=0) > 400.0)
        water_depth = (surf_m[:, S_WATER] > 200.0).sum(axis=0).astype(np.float32) / max(1.0, z_surf_top * 0.25)
        alt_norm = np.clip((top_idx.astype(np.float32) / max(1.0, z_surf_top - 1) - 0.50) * 2.4, 0.0, 1.0)

        # Orbital sky-island ring footprint
        sky_rock = mass[z_sky0:, S_ROCK].max(axis=0) > 200.0
        sky_soil = mass[z_sky0:, S_SOIL].max(axis=0) > 200.0
        ring_faces[f] = (sky_rock | sky_soil).astype(np.float32)

        col = np.zeros((N, N, 3), dtype=np.float32)
        for ii in range(N):
            for jj in range(N):
                if not has_any[ii, jj]:
                    col[ii, jj] = (14.0, 22.0, 38.0)
                    continue
                sp = top_sp[ii, jj]
                if magma_col[ii, jj]:
                    col[ii, jj] = (255.0, 102.0, 28.0)
                elif sp[S_ICE] > 180.0:
                    # Crisp glacier/polar ice blue-white
                    shade = 0.88 + 0.12 * alt_norm[ii, jj]
                    col[ii, jj] = (205.0 * shade, 232.0 * shade, 252.0 * shade)
                elif sp[S_WATER] > 180.0:
                    # Deep sapphire ocean -> turquoise shallows
                    d = min(1.0, water_depth[ii, jj])
                    col[ii, jj] = (16.0 + 26.0 * (1.0 - d), 78.0 + 78.0 * (1.0 - d), 152.0 + 68.0 * (1.0 - d))
                elif sp[S_WOOD] > 120.0:
                    col[ii, jj] = (38.0, 118.0, 72.0)
                elif sp[S_PLANT] > 120.0:
                    col[ii, jj] = (68.0, 158.0, 88.0)
                elif sp[S_SOIL] > 120.0:
                    col[ii, jj] = (138.0, 112.0, 78.0)
                else:
                    # Alpine mountain rock
                    sh = 0.78 + 0.35 * alt_norm[ii, jj]
                    col[ii, jj] = (min(255.0, 108.0 * sh), min(255.0, 116.0 * sh), min(255.0, 132.0 * sh))

        rgb_faces[f] = col

    return rgb_faces, cos_faces, ring_faces


def _render_3d_globe_image(
    rgb_faces: np.ndarray,
    ring_faces: np.ndarray,
    size: int = 440,
    sun_dir: tuple[float, float, float] = (0.52, 0.24, 0.82),
    cam_yaw_deg: float = -24.0,
    cam_pitch_deg: float = 18.0,
) -> np.ndarray:
    """
    Ray-trace a crisp 3D orthographic spherical globe directly from the 6-face cubed-sphere
    voxel planet via `xyz_to_face_uv`, including:
      - Real Day/Night Terminator lighting + warm golden twilight band + night-side volcanic glow
      - Atmospheric Fresnel cyan rim glow
      - Tilted floating orbital Sky-Island Ring passing both behind AND in front of the globe!
    Returns `[size, size, 3]` uint8 RGB image.
    """
    N = rgb_faces.shape[1]
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Deep space backdrop
    img[:, :, 0] = 10.0
    img[:, :, 1] = 15.0
    img[:, :, 2] = 25.0

    # Camera orthonormal basis (right, up, view_dir)
    yaw = math.radians(cam_yaw_deg)
    pitch = math.radians(cam_pitch_deg)
    cam_fwd = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)], dtype=np.float64)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cam_right = np.cross(world_up, cam_fwd)
    cam_right /= np.linalg.norm(cam_right)
    cam_up = np.cross(cam_fwd, cam_right)

    s = np.array(sun_dir, dtype=np.float64)
    s /= np.linalg.norm(s)

    # Pixel coordinates in [-1.42, 1.42] (planet sphere radius = 1.0, orbital ring radius = 1.18..1.34)
    span = 1.42
    coords = np.linspace(-span, span, size, dtype=np.float64)
    px, py = np.meshgrid(coords, -coords)  # py up!
    r2 = px * px + py * py
    r = np.sqrt(r2)

    # Subtle starfield in background
    rng = np.random.default_rng(2901)
    for _ in range(95):
        sy_i = int(rng.integers(4, size - 4))
        sx_i = int(rng.integers(4, size - 4))
        br = float(rng.uniform(110.0, 220.0))
        img[sy_i, sx_i] = (br * 0.85, br * 0.92, br)

    # Atmospheric outer halo glow around planet limb (1.0 < r < 1.13)
    halo_mask = (r >= 0.98) & (r < 1.15)
    halo_t = np.exp(-((r[halo_mask] - 1.0) / 0.055) ** 2)
    img[halo_mask, 0] += 36.0 * halo_t
    img[halo_mask, 1] += 145.0 * halo_t
    img[halo_mask, 2] += 230.0 * halo_t

    # Tilted orbital ring plane: normal n_ring = (sin(14°), cos(14°), 0)
    tilt_rad = math.radians(14.0)
    n_ring = np.array([math.sin(tilt_rad), math.cos(tilt_rad), 0.0], dtype=np.float64)
    # Ray: P(t) = px*cam_right + py*cam_up + t*cam_fwd, intersect dot(P(t), n_ring) = 0
    denom_ring = float(np.dot(cam_fwd, n_ring))
    has_ring_plane = abs(denom_ring) > 1e-4

    def _paint_ring(front_only: bool) -> None:
        if not has_ring_plane:
            return
        num = -(px * float(np.dot(cam_right, n_ring)) + py * float(np.dot(cam_up, n_ring)))
        t_hit = num / denom_ring  # > 0 means in front of planet center!
        P_x = px * cam_right[0] + py * cam_up[0] + t_hit * cam_fwd[0]
        P_y = px * cam_right[1] + py * cam_up[1] + t_hit * cam_fwd[1]
        P_z = px * cam_right[2] + py * cam_up[2] + t_hit * cam_fwd[2]
        rad_r = np.sqrt(P_x * P_x + P_y * P_y + P_z * P_z)
        ang = np.arctan2(P_z, P_x)
        isle_w = np.cos(8.0 * ang + 0.4)

        in_band = (rad_r >= 1.16) & (rad_r <= 1.33)
        if front_only:
            side_ok = (t_hit > 0.0) | (r2 >= 1.0)
        else:
            side_ok = (t_hit <= 0.0) & (r2 >= 1.0)
        rmask = in_band & side_ok
        if not np.any(rmask):
            return

        # Subtle orbital dust ring + bright emerald/stone floating sky-islands
        sun_r = np.clip((P_x * s[0] + P_y * s[1] + P_z * s[2]) * 0.65 + 0.45, 0.22, 1.0)
        is_isle = (isle_w > 0.45) & (rad_r >= 1.19) & (rad_r <= 1.30)

        # Dust band
        dust_m = rmask & (~is_isle)
        alpha_d = 0.28 * np.exp(-((rad_r[dust_m] - 1.245) / 0.055) ** 2)
        d_col = np.stack([115.0 * sun_r[dust_m], 195.0 * sun_r[dust_m], 235.0 * sun_r[dust_m]], axis=-1)
        img[dust_m] = img[dust_m] * (1.0 - alpha_d[:, None]) + d_col * alpha_d[:, None]

        # Floating sky-island chunks
        isle_m = rmask & is_isle
        if np.any(isle_m):
            sr = sun_r[isle_m]
            green_top = (rad_r[isle_m] > 1.235)
            r_c = np.where(green_top, 78.0, 128.0) * sr
            g_c = np.where(green_top, 186.0, 136.0) * sr
            b_c = np.where(green_top, 118.0, 152.0) * sr
            img[isle_m] = np.stack([r_c, g_c, b_c], axis=-1)

    # 1. Back half of orbital Sky-Island Ring
    _paint_ring(front_only=False)

    # 2. Ray-traced 3D Spherical Planet sampled directly from 6-face cubed-sphere!
    sph_mask = r2 <= 1.0
    pz = np.sqrt(np.clip(1.0 - r2[sph_mask], 0.0, 1.0))
    px_s, py_s = px[sph_mask], py[sph_mask]
    wx = px_s * cam_right[0] + py_s * cam_up[0] + pz * cam_fwd[0]
    wy = px_s * cam_right[1] + py_s * cam_up[1] + pz * cam_fwd[1]
    wz = px_s * cam_right[2] + py_s * cam_up[2] + pz * cam_fwd[2]

    f_t, u_t, v_t = xyz_to_face_uv(torch.from_numpy(wx), torch.from_numpy(wy), torch.from_numpy(wz))
    f_np = f_t.numpy()
    iu_np = np.clip(np.rint(((u_t.numpy() + 1.0) * N - 1.0) * 0.5).astype(np.int64), 0, N - 1)
    iv_np = np.clip(np.rint(((v_t.numpy() + 1.0) * N - 1.0) * 0.5).astype(np.int64), 0, N - 1)

    albedo = rgb_faces[f_np, iv_np, iu_np]  # [M, 3]
    cos_s = wx * s[0] + wy * s[1] + wz * s[2]

    # Smooth Day/Night shading + golden twilight scatter band near terminator (cos_s ~ 0)
    day_light = 0.14 + 0.94 * (0.5 * (1.0 + np.tanh(cos_s / 0.22))) * np.clip(0.35 + 0.75 * np.maximum(cos_s, 0.0), 0.2, 1.05)
    twilight = np.exp(-((cos_s + 0.02) / 0.13) ** 2)

    # Specular sunlight glint on liquid oceans
    is_ocean = (albedo[:, 2] > 140.0) & (albedo[:, 0] < 70.0)
    half_v = s + cam_fwd
    half_v /= np.linalg.norm(half_v)
    spec = np.power(np.clip(wx * half_v[0] + wy * half_v[1] + wz * half_v[2], 0.0, 1.0), 26.0) * is_ocean * np.clip(cos_s * 3.0, 0.0, 1.0)

    # Emissive volcanic magma glow visible even on the dark night hemisphere!
    is_magma = (albedo[:, 0] > 230.0) & (albedo[:, 1] < 130.0) & (albedo[:, 2] < 60.0)

    shaded = albedo * day_light[:, None]
    # Add warm amber twilight band along terminator
    shaded[:, 0] += 54.0 * twilight
    shaded[:, 1] += 24.0 * twilight
    shaded[:, 2] += 8.0 * twilight
    # Add ocean specular highlight
    shaded += (110.0 * spec)[:, None]
    # Preserve glowing volcanic lava on night side
    shaded[is_magma] = np.maximum(shaded[is_magma], np.array([255.0, 96.0, 26.0]))

    # Subtle cyan atmospheric limb scatter on sunlit edge
    rim = np.power(np.clip(1.0 - pz, 0.0, 1.0), 2.8) * np.clip(0.25 + 0.85 * np.maximum(cos_s, 0.0), 0.18, 1.0)
    shaded[:, 0] += 28.0 * rim
    shaded[:, 1] += 115.0 * rim
    shaded[:, 2] += 195.0 * rim

    # Draw subtle cube-face seam wireframe faintly on globe to prove seamlessness!
    near_seam = (np.abs(u_t.numpy()) > 0.975) | (np.abs(v_t.numpy()) > 0.975)
    shaded[near_seam] = shaded[near_seam] * 0.72 + np.array([120.0, 215.0, 255.0]) * 0.28

    img[sph_mask] = np.clip(shaded, 0.0, 255.0)

    # 3. Front half of orbital Sky-Island Ring
    _paint_ring(front_only=True)

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def render_planet_cubed_sphere_viz(
    metrics: dict[str, Any],
    out_path: Path,
    sun_dir: tuple[float, float, float] = (0.52, 0.24, 0.82),
) -> None:
    """
    Render crisp 3-panel dark-theme visualization (`models/planet_cubed_sphere_viz.png`, 2280x820 px):
      - Panel 1: Unfolded 6-Face Cubed-Sphere Net (4x3 cross) with seamless global biomes,
                 live Solar Day/Night Terminator overlay, and 12-edge rotated seam arrows.
      - Panel 2: Rendered 3D Spherical Globe Projection (oceans, continents, polar caps,
                 volcanic rift, Day/Night terminator + twilight band, orbital Sky-Island Ring).
      - Panel 3: Spark GB10 FP16 Planetary Scaling bar chart (393K / 3.15M / 6.29M voxels —
                 Dense vs Sparse p50 ms, speedup, skip %, Hz, VRAM & 12-edge halo KB/tick).
    Works cleanly via system PIL (no matplotlib dependency required on Spark GB10!).
    """
    rgb_faces, cos_faces, ring_faces = _compute_planet_surface_rgb_and_terminator(N=64, L=16, sun_dir=sun_dir)
    globe_rgb = _render_3d_globe_image(rgb_faces, ring_faces, size=440, sun_dir=sun_dir)

    # Shade each unfolded face with subtle day/night lighting + highlight terminator contour (|cos_s| < 0.055)
    net_faces_u8 = np.zeros((NUM_FACES, 64, 64, 3), dtype=np.uint8)
    for f in range(NUM_FACES):
        cs = cos_faces[f]
        light = 0.24 + 0.82 * (0.5 * (1.0 + np.tanh(cs / 0.24)))
        sh = rgb_faces[f] * light[:, :, None]
        # Preserve volcanic magma glow
        is_mag = (rgb_faces[f, :, :, 0] > 230.0) & (rgb_faces[f, :, :, 1] < 130.0)
        sh[is_mag] = np.maximum(sh[is_mag], np.array([255.0, 98.0, 28.0]))
        # Subtle sky-ring footprint outline
        is_ring = ring_faces[f] > 0.5
        sh[is_ring] = sh[is_ring] * 0.65 + np.array([56.0, 218.0, 248.0]) * 0.35
        # Crisp golden-amber Day/Night Terminator line
        term = np.abs(cs) < 0.032
        sh[term] = np.array([251.0, 191.0, 36.0])
        # Flip vertically so row 0 of PNG is +v (North/top) and row 63 is -v (South/bottom)
        net_faces_u8[f] = np.clip(sh[::-1], 0.0, 255.0).astype(np.uint8)

    # Write temporary raw binary arrays into payload dir or pass base64 via stdin to system PIL
    import base64

    payload = json.dumps(
        {
            "metrics": metrics,
            "out_path": str(out_path),
            "net_faces_b64": base64.b64encode(net_faces_u8.tobytes()).decode("ascii"),
            "globe_b64": base64.b64encode(globe_rgb.tobytes()).decode("ascii"),
            "globe_size": int(globe_rgb.shape[0]),
        }
    )

    pil_script = r'''
import base64, json, math, sys
from PIL import Image, ImageDraw, ImageFont

data = json.loads(sys.stdin.read())
metrics = data["metrics"]
out_path = data["out_path"]
net_bytes = base64.b64decode(data["net_faces_b64"])
globe_bytes = base64.b64decode(data["globe_b64"])
gsz = int(data["globe_size"])

W, H = 2280, 820
BG = (10, 15, 24)
PANEL = (17, 24, 38)
BORDER = (42, 57, 82)
FG = (234, 240, 250)
MUTED = (148, 163, 184)
GRID = (36, 49, 70)
ROSE = (244, 63, 94)
CYAN = (34, 211, 238)
SKY = (56, 189, 248)
EMERALD = (52, 211, 153)
AMBER = (251, 191, 36)
ORANGE = (249, 115, 22)
SLATE = (100, 116, 139)

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

def draw_arrow(draw, x0, y0, x1, y1, color, width=2):
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    ah = 8.0
    p1 = (x1 - ah * math.cos(ang - 0.45), y1 - ah * math.sin(ang - 0.45))
    p2 = (x1 - ah * math.cos(ang + 0.45), y1 - ah * math.sin(ang + 0.45))
    draw.polygon([(x1, y1), p1, p2], fill=color)

img = Image.new("RGB", (W, H), BG)
draw = ImageDraw.Draw(img)

text_center(
    draw,
    W // 2,
    34,
    "Frostmarsh Wave 29A — 6-Face Cubed-Sphere WorldModel3D-v8 Planet (12-Edge HALO_XY=6, Day/Night Terminator & GB10 Scaling)",
    font(22, bold=True),
    FG,
)

panels = [
    (24, 70, 826, 792),
    (848, 70, 1468, 792),
    (1490, 70, 2256, 792),
]
for px0, py0, px1, py1 in panels:
    draw.rounded_rectangle([px0, py0, px1, py1], radius=12, fill=PANEL, outline=BORDER, width=2)

# =========================================================================
# Panel 1: Unfolded 6-Face Cubed-Sphere Net (4x3 cross) + Terminator + Seam Arrows
# =========================================================================
p1x0, p1y0, p1x1, p1y1 = panels[0]
text_center(draw, (p1x0 + p1x1) // 2, p1y0 + 26, "1. Unfolded 6-Face Cubed-Sphere Net (Biomes + Terminator + 12 Seams)", font(16, bold=True), FG)

# 4x3 cross layout:
#         [+Y (2)]
# [-X(1)] [+Z (4)] [+X (0)] [-Z (5)]
#         [-Y (3)]
face_sz = 164
net_x0 = p1x0 + (p1x1 - p1x0 - 4 * face_sz) // 2
net_y0 = p1y0 + 78
layout = {
    2: (1, 0, "+Y  NORTH"),
    1: (0, 1, "-X  WEST"),
    4: (1, 1, "+Z  PRIME"),
    0: (2, 1, "+X  EAST"),
    5: (3, 1, "-Z  ANTI"),
    3: (1, 2, "-Y  SOUTH"),
}

stride = 64 * 64 * 3
for f, (gx, gy, flbl) in layout.items():
    f_raw = net_bytes[f * stride : (f + 1) * stride]
    f_im = Image.frombytes("RGB", (64, 64), f_raw).resize((face_sz, face_sz), Image.NEAREST)
    fx = net_x0 + gx * face_sz
    fy = net_y0 + gy * face_sz
    img.paste(f_im, (fx, fy))
    draw.rectangle([fx, fy, fx + face_sz, fy + face_sz], outline=(148, 163, 184), width=2)
    # Subtle face badge in top-left corner
    draw.rounded_rectangle([fx + 6, fy + 6, fx + 92, fy + 26], radius=4, fill=(10, 15, 24), outline=CYAN, width=1)
    text_center(draw, fx + 49, fy + 16, flbl, font(11, bold=True), FG)

# Draw curved/angled seam arrows showing rotated 90 deg / 180 deg edge exchanges across the unfolded net
# +Y (+u edge) <-> +X (+v edge): 90 deg rotation R=[[0,1],[-1,0]]
y_right_x = net_x0 + 2 * face_sz
y_mid_y = net_y0 + face_sz // 2
x_mid_x = net_x0 + 2 * face_sz + face_sz // 2
x_top_y = net_y0 + face_sz
draw.line([(y_right_x + 4, y_mid_y), (x_mid_x, y_mid_y), (x_mid_x, x_top_y - 4)], fill=CYAN, width=2)
draw_arrow(draw, x_mid_x, y_mid_y, x_mid_x, x_top_y - 4, CYAN, width=2)
draw.rounded_rectangle([x_mid_x - 30, y_mid_y - 24, x_mid_x + 52, y_mid_y - 4], radius=4, fill=BG, outline=CYAN, width=1)
text_center(draw, x_mid_x + 11, y_mid_y - 14, "90° R_xy", font(11, bold=True), CYAN)

# +Y (-u edge) <-> -X (+v edge): -90 deg rotation
y_left_x = net_x0 + face_sz
nx_mid_x = net_x0 + face_sz // 2
draw.line([(y_left_x - 4, y_mid_y), (nx_mid_x, y_mid_y), (nx_mid_x, x_top_y - 4)], fill=CYAN, width=2)
draw_arrow(draw, nx_mid_x, y_mid_y, nx_mid_x, x_top_y - 4, CYAN, width=2)
draw.rounded_rectangle([nx_mid_x - 46, y_mid_y - 24, nx_mid_x + 36, y_mid_y - 4], radius=4, fill=BG, outline=CYAN, width=1)
text_center(draw, nx_mid_x - 5, y_mid_y - 14, "-90° R", font(11, bold=True), CYAN)

# -Y (+u edge) <-> +X (-v edge): -90 deg rotation
sy_mid_y = net_y0 + 2 * face_sz + face_sz // 2
x_bot_y = net_y0 + 2 * face_sz
draw.line([(y_right_x + 4, sy_mid_y), (x_mid_x, sy_mid_y), (x_mid_x, x_bot_y + 4)], fill=CYAN, width=2)
draw_arrow(draw, x_mid_x, sy_mid_y, x_mid_x, x_bot_y + 4, CYAN, width=2)
draw.rounded_rectangle([x_mid_x - 28, sy_mid_y + 6, x_mid_x + 54, sy_mid_y + 26], radius=4, fill=BG, outline=CYAN, width=1)
text_center(draw, x_mid_x + 13, sy_mid_y + 16, "-90° R", font(11, bold=True), CYAN)

# Wrap-around equatorial seam (-X left <-> -Z right)
eq_y = net_y0 + face_sz + face_sz // 2
draw_arrow(draw, net_x0 - 30, eq_y, net_x0 - 4, eq_y, EMERALD, width=2)
draw_arrow(draw, net_x0 + 4 * face_sz + 4, eq_y, net_x0 + 4 * face_sz + 30, eq_y, EMERALD, width=2)
text_center(draw, net_x0 - 30, eq_y - 16, "wrap", font(10, bold=True), EMERALD)
text_center(draw, net_x0 + 4 * face_sz + 26, eq_y - 16, "wrap", font(10, bold=True), EMERALD)

# Legend row under net
leg1 = [
    ("Ocean", (26, 114, 186)),
    ("Forest/Land", (58, 146, 84)),
    ("Polar/Snow Ice", (202, 230, 250)),
    ("Volcanic Rift", (255, 102, 28)),
    ("Sky-Ring", (56, 218, 248)),
    ("Terminator", AMBER),
]
lx_c = p1x0 + 36
ly_c = net_y0 + 3 * face_sz + 24
for lbl, col in leg1:
    draw.rectangle([lx_c, ly_c, lx_c + 14, ly_c + 14], fill=col, outline=MUTED, width=1)
    draw.text((lx_c + 20, ly_c), lbl, font=font(12), fill=FG)
    lx_c += 126 if lbl not in ("Ocean", "Sky-Ring") else 104

# Parity & topology summary badge at bottom of Panel 1
tp = metrics["topology_and_parity"]
rt = tp["projection_round_trip"]
inv = tp["edge_involution_and_vector_rotation"]
v8c = tp["world_model_v8_cross_face_consistency"]
draw.rounded_rectangle([p1x0 + 22, p1y1 - 74, p1x1 - 22, p1y1 - 16], radius=8, fill=(10, 30, 26), outline=EMERALD, width=1)
text_center(
    draw, (p1x0 + p1x1) // 2, p1y1 - 56,
    f"12-Edge Involution: {inv['edge_involution_mismatches']} err | Tangent Rot Round-Trip: {inv['tangent_vector_roundtrip_max_err']:.1f} | "
    f"3D Proj Err: {max(rt['max_xyz_err_interior'], rt['max_xyz_err_8_corners']):.1e}",
    font(12, bold=True), EMERALD,
)
text_center(
    draw, (p1x0 + p1x1) // 2, p1y1 - 32,
    f"v8 6-Face Dense vs Sparse: {v8c['dense_haloed_vs_sparse_6face_max_rel']:.2e} | "
    f"Unfolded Seam Parity: {v8c['unfolded_2face_monolith_vs_haloed_seam_rel_horiz']:.2e} | "
    f"Flow Rot Gain: {v8c['rotation_error_reduction_x']:.1f}x",
    font(12), FG,
)

# =========================================================================
# Panel 2: Rendered 3D Spherical Globe Projection + 20-Tick Terminator Rollout
# =========================================================================
p2x0, p2y0, p2x1, p2y1 = panels[1]
text_center(draw, (p2x0 + p2x1) // 2, p2y0 + 26, "2. 3D Cubed-Sphere Globe & Solar Terminator Rollout", font(16, bold=True), FG)

globe_im = Image.frombytes("RGB", (gsz, gsz), globe_bytes)
gx0 = p2x0 + (p2x1 - p2x0 - gsz) // 2
gy0 = p2y0 + 54
img.paste(globe_im, (gx0, gy0))
draw.rounded_rectangle([gx0 - 2, gy0 - 2, gx0 + gsz + 2, gy0 + gsz + 2], radius=10, outline=BORDER, width=2)

# Sun direction & terminator callouts cleanly placed in top corners of globe frame
draw.rounded_rectangle([gx0 + 10, gy0 + 10, gx0 + 178, gy0 + 50], radius=6, fill=(28, 22, 10), outline=AMBER, width=1)
text_center(draw, gx0 + 94, gy0 + 22, "SUNLIT DAY (+38 K)", font(11, bold=True), AMBER)
text_center(draw, gx0 + 94, gy0 + 38, "Ice Melting & Storms", font(10), FG)

draw.rounded_rectangle([gx0 + gsz - 178, gy0 + gsz - 50, gx0 + gsz - 10, gy0 + gsz - 10], radius=6, fill=(14, 22, 42), outline=SKY, width=1)
text_center(draw, gx0 + gsz - 94, gy0 + gsz - 38, "DARK NIGHT (-45 K)", font(11, bold=True), SKY)
text_center(draw, gx0 + gsz - 94, gy0 + gsz - 22, "Ocean Freezing & Rift Glow", font(10), FG)

draw.rounded_rectangle([gx0 + gsz - 174, gy0 + 10, gx0 + gsz - 10, gy0 + 36], radius=6, fill=(10, 28, 36), outline=CYAN, width=1)
text_center(draw, gx0 + gsz - 92, gy0 + 23, "Orbital Sky-Ring (0.35g)", font(11, bold=True), CYAN)

# 20-Tick Terminator Rollout mini-sparkline & stats card below globe
roll = metrics["terminator_rollout_20tick"]
rx0, ry0, rx1, ry1 = p2x0 + 22, gy0 + gsz + 14, p2x1 - 22, p2y1 - 86
draw.rounded_rectangle([rx0, ry0, rx1, ry1], radius=8, fill=(12, 18, 30), outline=BORDER, width=1)
text_center(draw, (rx0 + rx1) // 2, ry0 + 16, "20-Tick Rotating Terminator: WorldModel3D-v8 Reaction Rates (6x64x64x16)", font(12, bold=True), FG)

ticks = roll["ticks"]
px_a, py_a, px_b, py_b = rx0 + 46, ry0 + 34, rx1 - 16, ry1 - 24
draw.rectangle([px_a, py_a, px_b, py_b], fill=(10, 15, 24), outline=GRID, width=1)
max_r = max(max(t["day_melt_rate"], t["night_freeze_rate"], t["day_freeze_rate"], t["night_melt_rate"]) for t in ticks)
max_r = max(max_r * 1.48, 1e-4)

def _r_pt(idx, val):
    xx = px_a + int((idx / max(len(ticks) - 1, 1)) * (px_b - px_a))
    yy = py_b - 4 - int((val / max_r) * (py_b - py_a - 24))
    return (xx, yy)

pts_dm = [_r_pt(i, t["day_melt_rate"]) for i, t in enumerate(ticks)]
pts_nf = [_r_pt(i, t["night_freeze_rate"]) for i, t in enumerate(ticks)]
pts_df = [_r_pt(i, t["day_freeze_rate"]) for i, t in enumerate(ticks)]
pts_nm = [_r_pt(i, t["night_melt_rate"]) for i, t in enumerate(ticks)]
draw.line(pts_df, fill=(100, 116, 139), width=2)
draw.line(pts_nm, fill=(148, 163, 184), width=2)
draw.line(pts_dm, fill=AMBER, width=3)
draw.line(pts_nf, fill=CYAN, width=3)
draw.text((px_a + 8, py_a + 4), "— Day Melt", font=font(11, bold=True), fill=AMBER)
draw.text((px_a + 96, py_a + 4), "— Night Freeze", font=font(11, bold=True), fill=CYAN)
draw.text((px_a + 206, py_a + 4), "— Opposite Side (Near 0)", font=font(11), fill=MUTED)
text_center(draw, (px_a + px_b) // 2, py_b + 11, "Tick 0 -> 19 (Sun sweeps 144° around planet equator)", font(11), MUTED)

draw.rounded_rectangle([p2x0 + 22, p2y1 - 74, p2x1 - 22, p2y1 - 16], radius=8, fill=(24, 22, 12), outline=AMBER, width=1)
text_center(
    draw, (p2x0 + p2x1) // 2, p2y1 - 56,
    f"Terminator Divergence: {roll['mean_day_melt_vs_night_freeze']:.4f} | "
    f"Day-Night ΔT: +{roll['final_day_minus_night_temp_K']:.1f} K | NaNs: {roll['nan_or_inf_count']}",
    font(12, bold=True), AMBER,
)
text_center(
    draw, (p2x0 + p2x1) // 2, p2y1 - 32,
    f"Day Melt {roll['day_vs_night_melt_ratio']:.1f}x Night | Night Freeze {roll['night_vs_day_freeze_ratio']:.1f}x Day | "
    f"Seam/Int Flux Ratio: {roll['mean_seam_vs_interior_flux_ratio']:.2f}x",
    font(12), FG,
)

# =========================================================================
# Panel 3: Spark GB10 FP16 Planetary Scaling Bar Chart (3 Scales)
# =========================================================================
p3x0, p3y0, p3x1, p3y1 = panels[2]
gb = metrics["gpu_benchmarks"]
text_center(draw, (p3x0 + p3x1) // 2, p3y0 + 26, f"3. Planetary GPU Scaling ({gb['device']}, FP16, v8 164.6K)", font(16, bold=True), FG)

hx0, hy0, hx1, hy1 = p3x0 + 82, p3y0 + 70, p3x1 - 28, p3y1 - 178
draw.rectangle([hx0, hy0, hx1, hy1], fill=(12, 18, 30), outline=BORDER, width=1)

skeys = ["scale_a_coarse_global", "scale_b_micro_planet", "scale_c_3strata_planet"]
slist = [gb["scales"][k] for k in skeys]
ymax = max(max(it["dense"]["p50_ms"], it["sparse"]["p50_ms"]) for it in slist) * 1.22
ymax = math.ceil(ymax / 50.0) * 50.0
ymax = max(ymax, 150.0)

def y_lin(val):
    return int(hy1 - (max(0.0, min(ymax, val)) / ymax) * (hy1 - hy0))

step_y = 200 if ymax >= 750 else (100 if ymax >= 300 else 50)
for yv in range(0, int(ymax) + 1, step_y):
    yy = y_lin(yv)
    draw.line([(hx0, yy), (hx1, yy)], fill=GRID, width=1)
    draw.text((hx0 - 60, yy - 8), f"{yv:>3}ms", font=font(12), fill=MUTED)

# 10 Hz real-time budget line (100 ms)
y_10hz = y_lin(100.0)
draw_dashed(draw, hx0, y_10hz, hx1, y_10hz, ROSE, width=3, dash=10, gap=6)

slot_w = (hx1 - hx0) / len(slist)
bw = 62
short_titles = ["Scale A: Global\n6x64x64x16\n(0.39M vox)", "Scale B: Micro\n6x128x128x32\n(3.15M vox)", "Scale C: 3-Strata\n6x128x128x64\n(6.29M vox)"]

for i, (it, stitle) in enumerate(zip(slist, short_titles)):
    cx = int(hx0 + slot_w * (i + 0.5))
    dp = it["dense"]["p50_ms"]
    sp = it["sparse"]["p50_ms"]
    spd = it["sparse"]["speedup_vs_dense"]
    sk = it["brick_stats"]["skip_pct"]
    hz = it["sparse"]["hz_p50"]
    vram = it["sparse"]["peak_vram_mb"]
    yd, ys = y_lin(dp), y_lin(sp)

    draw.rectangle([cx - bw - 5, yd, cx - 5, hy1], fill=SLATE, outline=(203, 213, 225), width=1)
    s_col = EMERALD if sp <= 100.0 else SKY
    draw.rectangle([cx + 5, ys, cx + bw + 5, hy1], fill=s_col, outline=(226, 232, 240), width=1)

    d_off = 22 if abs(yd - y_10hz) < 20 else 13
    s_off = 24 if abs(ys - y_10hz) < 20 else 13
    text_center(draw, cx - bw // 2 - 5, yd - d_off, f"{dp:.1f}ms", font(12, bold=True), MUTED)
    text_center(draw, cx + bw // 2 + 5, ys - s_off, f"{sp:.1f}ms", font(12, bold=True), FG)
    text_center(draw, cx + bw // 2 + 5, ys - s_off - 17, f"{spd:.2f}x", font(12, bold=True), CYAN)

    lines = stitle.split("\n") + [f"Skip {sk:.1f}% | {hz:.1f} Hz", f"VRAM {vram:.0f} MB"]
    for li, ln in enumerate(lines):
        col_l = FG if li == 0 else (MUTED if li in (1, 2) else (EMERALD if li == 3 else CYAN))
        text_center(draw, cx, hy1 + 15 + li * 16, ln, font(11, bold=(li in (0, 3))), col_l)

# Legend in clear upper-left (Scale A bars are very short!)
lx, ly = hx0 + 16, hy0 + 14
draw.rounded_rectangle([lx, ly, lx + 270, ly + 82], radius=6, fill=BG, outline=BORDER, width=1)
draw.rectangle([lx + 12, ly + 12, lx + 28, ly + 24], fill=SLATE)
draw.text((lx + 36, ly + 10), "Dense 6-Face + Halo (p50 ms)", font=font(12), fill=MUTED)
draw.rectangle([lx + 12, ly + 34, lx + 28, ly + 46], fill=EMERALD)
draw.text((lx + 36, ly + 32), "Sparse 16x16x8 Pool (p50 ms)", font=font(12, bold=True), fill=FG)
draw_dashed(draw, lx + 12, ly + 62, lx + 28, ly + 62, ROSE, width=3, dash=5, gap=3)
draw.text((lx + 36, ly + 54), "10 Hz Real-Time (100 ms)", font=font(12, bold=True), fill=(253, 164, 175))

sa, sb, sc = slist[0], slist[1], slist[2]
draw.rounded_rectangle([p3x0 + 22, p3y1 - 74, p3x1 - 22, p3y1 - 16], radius=8, fill=(10, 26, 38), outline=CYAN, width=1)
text_center(
    draw, (p3x0 + p3x1) // 2, p3y1 - 56,
    f"Micro-Planet (3.15M): {sb['sparse']['p50_ms']:.1f}ms ({sb['sparse']['hz_p50']:.1f}Hz, {sb['sparse']['speedup_vs_dense']:.2f}x) | "
    f"3-Strata (6.29M): {sc['sparse']['p50_ms']:.1f}ms ({sc['sparse']['hz_p50']:.1f}Hz, {sc['sparse']['speedup_vs_dense']:.2f}x)",
    font(11, bold=True), CYAN,
)
text_center(
    draw, (p3x0 + p3x1) // 2, p3y1 - 32,
    f"12-Edge HALO_XY=6 zlib Delta: A={sa['halo_bandwidth']['zlib_delta_18ch_kb_per_tick']:.0f}KB, "
    f"B={sb['halo_bandwidth']['zlib_delta_18ch_kb_per_tick']:.0f}KB, C={sc['halo_bandwidth']['zlib_delta_18ch_kb_per_tick']:.0f}KB/tick | "
    f"max_bricks=192",
    font(11), FG,
)

img.save(out_path, format="PNG")
'''

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Try current python first if PIL is available, else fallback to /usr/bin/python3
    py_exe = sys.executable
    try:
        import PIL  # noqa: F401
    except ImportError:
        py_exe = "/usr/bin/python3"

    proc = subprocess.run([py_exe, "-c", pil_script], input=payload, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PIL viz rendering failed ({proc.returncode}): {proc.stderr}")


# =============================================================================
# 10. Checkpoint Loader & CLI Entrypoint
# =============================================================================


def load_v8_model(ckpt_path: Path, device: torch.device) -> WorldModel3DV8:
    model = WorldModel3DV8().to(device=device, dtype=torch.float32).eval()
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_state_dict(sd, strict=True)
        print(f"Loaded trained WorldModel3DV8 checkpoint: {ckpt_path} ({count_parameters(model):,} params)", flush=True)
    else:
        print(f"WARNING: {ckpt_path} not found; activating synthetic law weights for standalone check", flush=True)
        with torch.no_grad():
            model.law_to_dt.weight.normal_(0.0, 0.1)
            model.law_h_mod.weight.normal_(0.0, 0.1)
            model.law_h_logscale.weight.normal_(0.0, 0.1)
            model.law_z_logscale.weight.normal_(0.0, 0.1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave 29A: 6-Face Cubed-Sphere WorldModel3D-v8 Planetary Simulation & Spark GB10 Benchmarks"
    )
    parser.add_argument("--ckpt", type=str, default="world_model_3d_v8.pt")
    parser.add_argument("--out-json", type=str, default="PLANET_CUBED_SPHERE_METRICS.json")
    parser.add_argument("--out-png", type=str, default="planet_cubed_sphere_viz.png")
    parser.add_argument("--max-bricks-per-batch", type=int, default=192)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        metrics = json.loads(Path(args.out_json).read_text())
        render_planet_cubed_sphere_viz(metrics, Path(args.out_png))
        print(f"Re-rendered chart -> {args.out_png}", flush=True)
        return

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        alt = Path("models/checkpoints/world_model_3d_v8.pt")
        if alt.exists():
            ckpt_path = alt
    model = load_v8_model(ckpt_path, device=dev)

    print("\n=== Suite 1: Cubed-Sphere Topology, 12-Edge Rotation & WorldModel3D-v8 Seam Parity ===", flush=True)
    topo_res = run_topology_and_seam_parity_suite(model)
    rt = topo_res["projection_round_trip"]
    inv = topo_res["edge_involution_and_vector_rotation"]
    v8c = topo_res["world_model_v8_cross_face_consistency"]
    print(
        f"  Projection round-trip max err: interior={rt['max_xyz_err_interior']:.2e}, "
        f"edges={rt['max_xyz_err_12_edges']:.2e}, corners={rt['max_xyz_err_8_corners']:.2e} (PASS={rt['passed_1e6_gate']})",
        flush=True,
    )
    print(
        f"  12-Edge Involution mismatches: edge={inv['edge_involution_mismatches']}, cell={inv['cell_involution_mismatches']} | "
        f"Tangent vec round-trip err={inv['tangent_vector_roundtrip_max_err']}, norm err={inv['tangent_vector_norm_max_err']}",
        flush=True,
    )
    print(
        f"  v8 6-Face Dense vs Sparse rel={v8c['dense_haloed_vs_sparse_6face_max_rel']:.2e} | "
        f"Unfolded seam rel={v8c['unfolded_2face_monolith_vs_haloed_seam_rel_horiz']:.2e} | "
        f"Polar flow rot gain={v8c['rotation_error_reduction_x']:.2f}x",
        flush=True,
    )

    print(
        f"\n=== Suite 2: Multi-Scale Planetary GPU Benchmarks (max_bricks_per_batch={args.max_bricks_per_batch}) ===",
        flush=True,
    )
    gpu_bench = benchmark_all_planet_scales_gpu(
        model=model,
        device=dev,
        max_bricks_per_batch=args.max_bricks_per_batch,
        warmup=args.warmup,
        iters=args.iters,
    )

    print("\n=== Suite 3: 20-Tick Rotating Solar Day/Night Terminator Rollout (6x64x64x16) ===", flush=True)
    rollout_res = run_20tick_terminator_rollout(model=model, device=dev, N=64, L=16, num_ticks=20)
    print(
        f"  Day Melt={rollout_res['mean_day_melt_rate']:.5f} ({rollout_res['day_vs_night_melt_ratio']:.2f}x Night) | "
        f"Night Freeze={rollout_res['mean_night_freeze_rate']:.5f} ({rollout_res['night_vs_day_freeze_ratio']:.2f}x Day) | "
        f"Divergence={rollout_res['mean_day_melt_vs_night_freeze']:.5f} | "
        f"Final Day-Night ΔT=+{rollout_res['final_day_minus_night_temp_K']:.2f} K | "
        f"NaNs={rollout_res['nan_or_inf_count']} | PASS={rollout_res['terminator_reaction_verified']}",
        flush=True,
    )

    metrics = {
        "wave": "29A",
        "title": "6-Face Cubed-Sphere WorldModel3D-v8 Planetary Simulation, 12-Edge HALO_XY=6 & Multi-Scale GB10 Benchmarks",
        "checkpoint": ckpt_path.name,
        "world_model_params": int(count_parameters(model)),
        "zero_retraining": True,
        "halo_xy": HALO_XY,
        "flow_channels_contract": {"flow_e": CH_FLOW_E, "flow_s": CH_FLOW_S},
        "topology_and_parity": topo_res,
        "gpu_benchmarks": gpu_bench,
        "terminator_rollout_20tick": rollout_res,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved metrics -> {out_json}", flush=True)

    out_png = Path(args.out_png)
    render_planet_cubed_sphere_viz(metrics, out_png)
    print(f"Saved chart   -> {out_png}", flush=True)


if __name__ == "__main__":
    main()
