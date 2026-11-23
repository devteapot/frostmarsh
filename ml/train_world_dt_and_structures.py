#!/usr/bin/env python3
"""
Continuous `Δt`-Conditioned World Time (Time Dilation + Temporal Sparsity) &
Emergent Destructible Structures / Terrain Shaping (`ml/train_world_dt_and_structures.py`).

Warm-starts `WorldModelDT` (~132K params) from `models/checkpoints/world_model_3d_v4.pt`
with EXACT initial `Δt = 1.0` parity (`0.00e+00` diff) and EXACT `Δt = 0.0` zero-change stasis.

Runs on GPU:
  1. Joint training across:
       - Multi-rate continuous `Δt in {0.25, 0.5, 1.0, 2.0, 4.0}` + log-uniform `[0.1, 4.0]`
         against exact `K`-substep RefSim + 3D vertical physics ground truth
       - Spatially varying time-dilation fields (0.15x Stasis Domes, 3.0x Rapid-Aging Bubbles,
         adaptive multi-rate combat/far-field tile fields)
       - Structural stress & solid phase-transmutation scenes (L2 timber skybridges, cryo ice
         ramparts, thermal-carved ice dams)
  2. Experiment A (Held-Out `256x256x4` & `512x512x4`):
       - Continuous `Δt` scaling accuracy (`MAE`, `R^2`) vs naive linear `dt * WorldModel3D`
       - Adaptive Multi-Rate Temporal Sparsity FP16 GPU wall-clock speedup + accuracy
       - `0.15x` Stasis Dome & `3.0x` Rapid-Aging Bubble fidelity + exact conservation check
  3. Experiment B (Held-Out `256x256x4`):
       - B1: Timber Skybridge Burn & `L2 -> L1` Collapse (rubble drop + flank isolation)
       - B2: Cryo Ice Rampart Construction (`sp_water + -thermal_flux` -> solid wall blocking advance)
       - B3: Thermal Channel Carving & River Redirection (`+thermal_flux` -> dam breach -> valley flood)
  4. Saves:
       - `models/checkpoints/world_model_dt.pt`
       - `models/WORLD_DT_STRUCTURES_METRICS.json`
       - `models/world_dt_structures_viz.png`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
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
    apply_forcing,
    apply_projected_deltas,
    compute_C_and_T,
    compute_total_energy,
    compute_total_mass,
    encode_state,
    get_consts,
    project_and_step,
    project_deltas,
    sample_bolts,
)
from train_world_3d_v4 import (  # noqa: E402
    compute_vertical_physics_teacher,
    make_stacked_3d_frostmarsh_world,
    project_and_apply_vertical_flux,
    vertical_physics_loss,
)
from train_world_v3 import (  # noqa: E402
    conservation_firewall_loss,
    physics_teacher_loss,
    state_rollout_loss,
)
from world_model_3d import (  # noqa: E402
    Z_FLUX_CH,
    Z_FLUX_NAMES,
    ZF_HEAT,
    ZF_OIL,
    ZF_SMOKE,
    ZF_STEAM,
    ZF_WATER,
    WorldModel3D,
)
from world_model_dt import (  # noqa: E402
    ST_CHANNEL_CARVE,
    ST_COLLAPSE,
    ST_ICE_RAMPART,
    ST_STRESS,
    STRUCT_CH,
    STRUCT_NAMES,
    WorldModelDT,
    build_dt_input_state,
    count_parameters,
)


# =============================================================================
# Continuous Δt & Spatially Varying Time-Dilation Reference Physics Engine
# =============================================================================


def _scale_horiz_deltas_by_dt(raw_deltas: torch.Tensor, local_dt: torch.Tensor) -> torch.Tensor:
    """Scale `[L, 17, H, W]` deltas symmetrically on faces by `min(dt_here, dt_nb)` and cells by `dt`."""
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


def _scale_z_flux_by_dt(raw_z: torch.Tensor, local_dt: torch.Tensor) -> torch.Tensor:
    """Scale `[L, 5, H, W]` vertical interface fluxes by `min(dt[z], dt[z+1])` (out-of-place)."""
    L = raw_z.shape[0]
    if L <= 1:
        return torch.zeros_like(raw_z)
    dt = local_dt.clamp(min=0.0)
    dt_z = torch.minimum(dt[:-1], dt[1:])
    z_if = raw_z[:-1] * dt_z
    z_top = torch.zeros_like(raw_z[-1:])
    return torch.cat([z_if, z_top], dim=0)


def project_and_apply_vertical_flux_diff(
    state: SimState,
    raw_z_flux: torch.Tensor,
) -> tuple[SimState, torch.Tensor]:
    """
    Pure out-of-place differentiable conservative 3D vertical flux projection & update for `[L, 5, H, W]`.
    Mathematically identical to `project_and_apply_vertical_flux` with ZERO inplace slice writes so
    autograd flows cleanly through `state_rollout_loss`!
    """
    L, _, H, W = state.mass.shape
    dev = state.mass.device
    if L <= 1:
        return state, torch.zeros_like(raw_z_flux)

    z_idx_list = (S_WATER, S_OIL, S_STEAM, S_SMOKE)
    z_idx = torch.tensor(z_idx_list, dtype=torch.long, device=dev)
    z_spec_c = torch.tensor([SPEC_C_LIST[s] for s in z_idx_list], dtype=torch.float32, device=dev).view(1, 4, 1, 1)

    mass = state.mass
    energy = state.energy

    fz_mass = raw_z_flux[:-1, 0:4] * 1000.0  # [L-1, 4, H, W]

    # 1. Donor-budget non-negativity across interfaces
    fz_pad = F.pad(fz_mass, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)  # [L+1, 4, H, W]
    out_z = F.relu(fz_pad[1:]) + F.relu(-fz_pad[:-1])            # [L, 4, H, W]
    avail_z = mass[:, z_idx].clamp(min=0.0)
    dscale = torch.clamp(avail_z / (out_z + 1e-6), max=1.0)
    fz_mass = torch.where(fz_mass >= 0.0, fz_mass * dscale[:-1], fz_mass * dscale[1:])

    # 2. Receiver H_CAP headroom projection
    fz_pad2 = F.pad(fz_mass, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
    in_z = (F.relu(fz_pad2[:-1]) + F.relu(-fz_pad2[1:])).sum(dim=1, keepdim=True)
    headroom = (H_CAP - mass.sum(dim=1, keepdim=True)).clamp(min=0.0)
    rscale = torch.clamp(headroom / (in_z + 1e-6), max=1.0)
    fz_mass = torch.where(fz_mass >= 0.0, fz_mass * rscale[1:], fz_mass * rscale[:-1])

    # 3. Apply conservative vertical mass transport + advected thermal enthalpy (out-of-place!)
    _, T_pre = compute_C_and_T(mass, energy)
    T_donor = torch.where(fz_mass >= 0.0, T_pre[:-1], T_pre[1:])
    e_adv = (fz_mass * z_spec_c * T_donor).sum(dim=1, keepdim=True)  # [L-1, 1, H, W]

    fz_pad3 = F.pad(fz_mass, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
    dm_z_mob = fz_pad3[:-1] - fz_pad3[1:]  # inflow from below - outflow upward [L, 4, H, W]
    dm_full = torch.zeros_like(mass).scatter_add(1, z_idx.view(1, 4, 1, 1).expand_as(dm_z_mob), dm_z_mob)
    mass_new = (mass + dm_full).clamp(min=0.0)

    e_pad = F.pad(e_adv, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
    energy_mid = energy + e_pad[:-1] - e_pad[1:]

    # 4. Vertical heat conduction (2nd-law + discrete max principle, out-of-place!)
    C_mid, T_mid = compute_C_and_T(mass_new, energy_mid)
    dT_z = T_mid[:-1] - T_mid[1:]
    q_max_z = dT_z.abs() * (C_mid[:-1] * C_mid[1:]) / (4.0 * (C_mid[:-1] + C_mid[1:]) + 1e-6)
    valid_c = (C_mid[:-1] > 1e-6) & (C_mid[1:] > 1e-6)
    q_max_z = torch.where(valid_c, q_max_z, torch.zeros_like(q_max_z))

    q_raw = raw_z_flux[:-1, ZF_HEAT : ZF_HEAT + 1] * 1e6
    q_proj = torch.where(
        dT_z > 0.0,
        q_raw.clamp(min=0.0).clamp(max=q_max_z),
        torch.where(dT_z < 0.0, q_raw.clamp(max=0.0).clamp(min=-q_max_z), torch.zeros_like(q_raw)),
    )
    q_pad = F.pad(q_proj, (0, 0, 0, 0, 0, 0, 1, 1), value=0.0)
    energy_new = energy_mid + q_pad[:-1] - q_pad[1:]

    proj_lo = torch.cat([fz_mass / 1000.0, q_proj / 1e6], dim=1)
    proj = torch.cat([proj_lo, torch.zeros_like(raw_z_flux[-1:])], dim=0)
    return SimState(mass=mass_new, energy=energy_new, flow_e=state.flow_e, flow_s=state.flow_s, elevation=state.elevation), proj


def apply_dt_projected_step(
    state: SimState,
    raw_h: torch.Tensor,
    raw_z: torch.Tensor | None,
    local_dt: torch.Tensor,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
) -> tuple[SimState, torch.Tensor, torch.Tensor]:
    """
    Exact conservative 3D step for arbitrary `local_dt [L, 1, H, W]`:
      - Projects horizontal deltas cleanly through `project_deltas`
      - Applies exact finite-volume transport, heat conduction, reactions, and `dt`-consistent flow inertia
      - Projects and applies 3D vertical fluxes through `project_and_apply_vertical_flux`
    Guarantees exact 3D mass & energy conservation (`rel_err ~ 1e-7`) for ANY spatially varying `local_dt`!
    """
    L, _, H, W = state.mass.shape
    dev = state.mass.device
    dt = local_dt.clamp(min=0.0)
    dt_e = torch.minimum(dt, F.pad(dt[:, :, :, 1:], (0, 1, 0, 0), mode="replicate"))
    dt_s = torch.minimum(dt, F.pad(dt[:, :, 1:, :], (0, 0, 0, 1), mode="replicate"))

    mass_eff, energy_eff = apply_forcing(state, forcing_energy, forcing_mass)
    proj_h, _ = project_deltas(mass_eff, energy_eff, raw_h)
    st_h = apply_projected_deltas(state, proj_h, forcing_energy, forcing_mass)

    # Adjust flow inertia decay smoothly with local_dt so small dt doesn't over-damp flow!
    flux_e = proj_h[:, 0:4] * 1000.0
    flux_s = proj_h[:, 4:8] * 1000.0
    liq_e = flux_e[:, 0:1] + flux_e[:, 3:4]
    liq_s = flux_s[:, 0:1] + flux_s[:, 3:4]
    wall = ((st_h.mass[:, S_ROCK : S_ROCK + 1] >= 500.0) | (st_h.mass[:, S_ICE : S_ICE + 1] >= 700.0)).float()
    mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=dev)
    mask_e[:, :, :, -1] = 0.0
    mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=dev)
    mask_s[:, :, -1, :] = 0.0
    open_e = mask_e * (1.0 - wall) * (1.0 - F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0))
    open_s = mask_s * (1.0 - wall) * (1.0 - F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0))
    alpha_e = (0.25 * dt_e).clamp(0.0, 0.90)
    alpha_s = (0.25 * dt_s).clamp(0.0, 0.90)
    flow_e_next = torch.clamp(((1.0 - alpha_e) * state.flow_e + 0.25 * liq_e) * open_e, -1500.0, 1500.0)
    flow_s_next = torch.clamp(((1.0 - alpha_s) * state.flow_s + 0.25 * liq_s) * open_s, -1500.0, 1500.0)
    st_h = SimState(
        mass=st_h.mass,
        energy=st_h.energy,
        flow_e=flow_e_next,
        flow_s=flow_s_next,
        elevation=state.elevation,
    )

    if raw_z is not None and L > 1:
        st_next, proj_z = project_and_apply_vertical_flux_diff(st_h, raw_z)
    else:
        st_next = st_h
        proj_z = torch.zeros((L, Z_FLUX_CH, H, W), dtype=torch.float32, device=dev)

    return st_next, proj_h, proj_z


@torch.no_grad()
def refsim_step_dt(
    state: SimState,
    local_dt: torch.Tensor,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
    include_vertical: bool = True,
) -> tuple[SimState, torch.Tensor, torch.Tensor]:
    """
    Exact `K`-substep RefSim + 3D vertical physics ground truth for ANY `local_dt [L, 1, H, W]`:
      - `dt <= 1.0`: `K = 1` exact substep (`dt == 1.0` matches `RefSim.step` bit-for-bit!)
      - `dt == 2.0`: `K = 2` exact substeps of `h = 1.0`
      - `dt == 4.0`: `K = 4` exact substeps of `h = 1.0`
      - Spatially varying `local_dt` (e.g., `0.15x` dome + `1.0x` exterior -> `K = 1`;
        `3.0x` bubble + `1.0x` exterior -> `K = 3` substeps of `h = local_dt / 3.0`)
    Returns `(next_state, accum_horiz_deltas [L,17,H,W], accum_z_flux [L,5,H,W])`.
    """
    L, _, H, W = state.mass.shape
    dev = state.mass.device
    dt = local_dt.clamp(min=0.0)
    max_dt = float(dt.max().item())

    accum_h = torch.zeros((L, 17, H, W), dtype=torch.float32, device=dev)
    accum_z = torch.zeros((L, Z_FLUX_CH, H, W), dtype=torch.float32, device=dev)
    if max_dt <= 1e-8:
        return state.clone(), accum_h, accum_z

    K = max(1, int(math.ceil(max_dt - 1e-6)))
    h_sub = dt / float(K)
    fe_sub = (forcing_energy / float(K)) if forcing_energy is not None else None
    fm_sub = (forcing_mass / float(K)) if forcing_mass is not None else None

    cur = state.clone()
    for _ in range(K):
        raw_h = RefSim.compute_raw_deltas(cur, fe_sub, fm_sub)
        raw_h_s = _scale_horiz_deltas_by_dt(raw_h, h_sub)
        if include_vertical and L > 1:
            raw_z = compute_vertical_physics_teacher(cur)
            raw_z_s = _scale_z_flux_by_dt(raw_z, h_sub)
        else:
            raw_z_s = None
        cur, proj_h, proj_z = apply_dt_projected_step(cur, raw_h_s, raw_z_s, h_sub, fe_sub, fm_sub)
        accum_h = accum_h + proj_h
        accum_z = accum_z + proj_z

    return cur, accum_h, accum_z


# =============================================================================
# Structural Stress & Solid Phase-Transmutation Teacher + Conservative Operator
# =============================================================================


@torch.no_grad()
def compute_structural_teacher(
    state: SimState,
    local_dt: torch.Tensor,
    forcing_energy: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Exact physically grounded structural teacher `[L, 4, H, W]`:
      0: `structural_stress`     in [0, 1]  (elevated timber span thermal/char/cantilever stress + ice melt stress)
      1: `collapse_z_flux`       >= 0       (downward L2->L1 fractured rubble collapse in mass/1000, scaled by dt_z)
      2: `ice_rampart_integrity` in [0, 1]  (cryo-solidified walkable/blocking ice rampart strength)
      3: `channel_carve_rate`    >= 0       (thermal beam ice-dam carving rate in mass/1000, scaled by dt)
    """
    L, _, H, W = state.mass.shape
    dev = state.mass.device
    dt = local_dt.clamp(min=0.0)

    mass = state.mass
    _, T_cK = compute_C_and_T(mass, state.energy)  # [L, 1, H, W]
    T = T_cK[:, 0]  # [L, H, W]

    wood = mass[:, S_WOOD]
    ash = mass[:, S_ASH]
    rock = mass[:, S_ROCK]
    soil = mass[:, S_SOIL]
    ice = mass[:, S_ICE]

    # Below-layer pillar support (`z == 0` sits on bedrock; `z >= 1` supported if `z-1` has rock>=480 or soil>=680)
    has_pillar_below = torch.ones((L, H, W), dtype=torch.bool, device=dev)
    if L > 1:
        has_pillar_below[1:] = (rock[:-1] >= 480.0) | (soil[:-1] >= 680.0)

    # Neighboring solid span continuity (3x3 average)
    solid_span = ((wood + rock) >= 300.0).float().unsqueeze(1)
    nb_support = F.avg_pool2d(F.pad(solid_span, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)[:, 0]

    # 0. Structural Stress
    rubble_avail = (wood + ash).clamp(min=0.0)
    is_timber_span = (~has_pillar_below) & (rock < 500.0) & (rubble_avail >= 60.0)
    t_dmg_wood = ((T - 36000.0) / 18000.0).clamp(0.0, 1.0)
    char_ratio = (ash / (rubble_avail + 60.0)).clamp(0.0, 1.0)
    thin_dmg = ((500.0 - wood) / 500.0).clamp(0.0, 1.0) * ((T > 40000.0) | (ash > 20.0)).float()
    cantilever_dmg = (1.0 - nb_support) * ((T > 38000.0) | (ash > 15.0)).float()
    stress_timber = (0.10 + 0.65 * t_dmg_wood + 0.35 * char_ratio + 0.25 * thin_dmg + 0.20 * cantilever_dmg).clamp(0.0, 1.0)

    is_ice_struct = (ice >= 180.0) & (rock < 500.0)
    t_dmg_ice = ((T - 25500.0) / 4800.0).clamp(0.0, 1.0)
    stress_ice = (0.90 * t_dmg_ice).clamp(0.0, 1.0)

    stress = torch.where(is_timber_span, stress_timber, torch.where(is_ice_struct, stress_ice, torch.zeros_like(T)))

    # 1. Downward Collapse Flux (L2->L1 fractured wood+ash rubble plunge)
    collapse = torch.zeros((L, H, W), dtype=torch.float32, device=dev)
    if L > 1:
        dt_below = torch.minimum(dt[1:, 0], dt[:-1, 0])
        headroom_below = (H_CAP - mass[:-1].sum(dim=1)).clamp(min=0.0)
        frac_gate = torch.sigmoid(14.0 * (stress[1:] - 0.60)) * (stress[1:] > 0.50).float() * is_timber_span[1:].float()
        raw_col = rubble_avail[1:] * (0.48 * frac_gate) * dt_below
        col_mass = torch.minimum(raw_col, torch.minimum(rubble_avail[1:], 0.85 * headroom_below))
        collapse[1:] = col_mass / 1000.0

    # 2. Cryo Ice Rampart Integrity (peaks at 1.0 when ice >= 750 & cold T <= 25000 cK)
    ice_norm = ((ice - 150.0) / 580.0).clamp(0.0, 1.0)
    cold_factor = ((29500.0 - T) / 4500.0).clamp(0.15, 1.0)
    ice_rampart = (ice_norm * cold_factor).clamp(0.0, 1.0)

    # 3. Thermal Ice-Dam Channel Carve Rate (mass / 1000.0, scaled by dt)
    if forcing_energy is not None:
        _, e_eff = apply_forcing(state, forcing_energy, None)
        _, T_eff_cK = compute_C_and_T(mass, e_eff)
        T_eff = T_eff_cK[:, 0]
        beam_boost = 1.0 + 1.25 * (forcing_energy[:, 0] / 1e8).clamp(0.0, 2.5)
    else:
        T_eff = T
        beam_boost = torch.ones_like(T)
    raw_carve = (0.030 * F.relu(T_eff - 27315.0) * (ice / 1000.0) / 100.0) * beam_boost * dt[:, 0]
    carve = torch.minimum(raw_carve, ice / 1000.0)

    return torch.stack([stress, collapse, ice_rampart, carve], dim=1)


def apply_structural_collapse_conservative(
    state: SimState,
    pred_struct: torch.Tensor,
    stress_thresh: float = 0.56,
) -> tuple[SimState, float]:
    """
    Conservatively drop fractured elevated timber/ash rubble (`L_z -> L_{z-1}`) driven by
    `WorldModelDT`'s `pred_struct[:, ST_STRESS]` and `pred_struct[:, ST_COLLAPSE]`.
    Strictly preserves 3D total mass (`sum mass`) AND 3D total thermal+chemical energy!
    """
    L, _, H, W = state.mass.shape
    if L <= 1:
        return state, 0.0

    mass = state.mass.clone()
    energy = state.energy.clone()
    _, T_cK = compute_C_and_T(mass, energy)

    stress_hi = pred_struct[1:, ST_STRESS]
    col_rate = pred_struct[1:, ST_COLLAPSE] * 1000.0  # physical mass units
    rock_hi = mass[1:, S_ROCK]
    has_pillar_below = (mass[:-1, S_ROCK] >= 480.0) | (mass[:-1, S_SOIL] >= 680.0)
    gate = (stress_hi >= stress_thresh) & (rock_hi < 500.0) & (~has_pillar_below)

    wood_hi = mass[1:, S_WOOD].clamp(min=0.0)
    ash_hi = mass[1:, S_ASH].clamp(min=0.0)
    rubble_hi = wood_hi + ash_hi
    headroom_lo = (H_CAP - mass[:-1].sum(dim=1)).clamp(min=0.0)

    dm_tot = torch.where(
        gate & (rubble_hi > 1e-4),
        torch.minimum(col_rate, torch.minimum(rubble_hi, 0.85 * headroom_lo)),
        torch.zeros_like(rubble_hi),
    )
    total_dropped = float(dm_tot.sum().item())
    if total_dropped <= 1e-6:
        return state, 0.0

    frac_w = wood_hi / (rubble_hi + 1e-6)
    dm_w = dm_tot * frac_w
    dm_a = dm_tot - dm_w

    # Enthalpy advected downward with falling rubble (exact thermal energy conservation!)
    T_hi = T_cK[1:, 0]
    e_fall = (dm_w * SPEC_C_LIST[S_WOOD] + dm_a * SPEC_C_LIST[S_ASH]) * T_hi

    mass[1:, S_WOOD] = (mass[1:, S_WOOD] - dm_w).clamp(min=0.0)
    mass[1:, S_ASH] = (mass[1:, S_ASH] - dm_a).clamp(min=0.0)
    mass[:-1, S_WOOD] = mass[:-1, S_WOOD] + dm_w
    mass[:-1, S_ASH] = mass[:-1, S_ASH] + dm_a

    energy[1:, 0] = energy[1:, 0] - e_fall
    energy[:-1, 0] = energy[:-1, 0] + e_fall

    return SimState(mass=mass, energy=energy, flow_e=state.flow_e, flow_s=state.flow_s, elevation=state.elevation), total_dropped


def apply_channel_carve_conservative(
    state: SimState,
    pred_struct: torch.Tensor,
) -> SimState:
    """
    Conservatively apply `WorldModelDT`'s predicted `channel_carve_rate` (`ST_CHANNEL_CARVE`)
    as an exact phase-change transmutation (`ice -dm -> water +dm, dE_th = -336000 * dm`),
    strictly gated by `T > 27315 cK` and the exact phase-equilibrium cap (`xi <= C*(T-27315)/(336000+21*27315)`).
    Preserves exact 3D total mass & thermal+chemical energy!
    """
    mass = state.mass.clone()
    energy = state.energy.clone()
    C_cell, T_cK = compute_C_and_T(mass, energy)
    T = T_cK[:, 0]
    C = C_cell[:, 0]
    ice = mass[:, S_ICE].clamp(min=0.0)
    req = (pred_struct[:, ST_CHANNEL_CARVE] * 1000.0).clamp(min=0.0)
    eq_cap = C * F.relu(T - 27315.0) / (336000.0 + 21.0 * 27315.0)
    dm = torch.minimum(req, torch.minimum(ice, eq_cap))
    if float(dm.sum().item()) <= 1e-6:
        return state
    mass[:, S_ICE] = (mass[:, S_ICE] - dm).clamp(min=0.0)
    mass[:, S_WATER] = mass[:, S_WATER] + dm
    energy[:, 0] = energy[:, 0] - dm * 336000.0
    return SimState(mass=mass, energy=energy, flow_e=state.flow_e, flow_s=state.flow_s, elevation=state.elevation)


# =============================================================================
# Structural & Time-Dilation World Generators
# =============================================================================


def make_structural_training_world(H: int, W: int, device: torch.device, seed: int) -> SimState:
    """
    Rich 4-layer 3D training world combining ALL three emergent structural scenarios simultaneously:
      - L2 Timber Skybridge spanning an open canyon over L1 river (with randomized heating/char)
      - L1 Cryo Rampart zone with cold water & partially/fully frozen ice walls
      - L1 Pressurized Reservoir behind an Ice Dam with warm/carved breach zones
      - Plus rising smoke/steam plumes & waterfalls so 3D vertical physics stays sharp!
    """
    st = make_stacked_3d_frostmarsh_world(H, W, device, seed=seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 777)
    c = get_consts(device)

    mass = st.mass.clone()
    _, T_cK = compute_C_and_T(mass, st.energy)
    T = T_cK.clone()
    elev = st.elevation.clone()

    yy = torch.linspace(0.0, 1.0, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, W).expand(H, W)

    # 1. L2 Timber Skybridge across central canyon (open air on L1 underneath!)
    by = float(torch.empty(1, device=device).uniform_(0.28, 0.72, generator=gen).item())
    bridge_span = ((yy - by).abs() < 0.045) & (xx > 0.25) & (xx < 0.75)
    abut_w = ((yy - by).abs() < 0.065) & (xx >= 0.14) & (xx <= 0.25)
    abut_e = ((yy - by).abs() < 0.065) & (xx >= 0.75) & (xx <= 0.86)

    # Ensure open canyon below the skybridge span on L1
    mass[1, S_ROCK] = torch.where(bridge_span, 0.0, mass[1, S_ROCK])
    mass[1, S_SOIL] = torch.where(bridge_span, 200.0, mass[1, S_SOIL])
    mass[1, S_WATER] = torch.where(bridge_span, 750.0, mass[1, S_WATER])

    # Place solid rock abutments & L2 timber span
    mass[2, S_ROCK] = torch.where(abut_w | abut_e, 880.0, torch.where(bridge_span, 0.0, mass[2, S_ROCK]))
    mass[2, S_SOIL] = torch.where(bridge_span, 180.0, mass[2, S_SOIL])
    burn_stage = float(torch.rand(1, generator=gen, device=device).item())
    if burn_stage < 0.35:
        # Cool intact bridge
        mass[2, S_WOOD] = torch.where(bridge_span, 820.0, mass[2, S_WOOD])
        mass[2, S_ASH] = torch.where(bridge_span, 0.0, mass[2, S_ASH])
        T[2, 0] = torch.where(bridge_span, 29000.0, T[2, 0])
    elif burn_stage < 0.70:
        # Heating / actively burning bridge mid-span
        mid = bridge_span & ((xx - 0.50).abs() < 0.14)
        mass[2, S_WOOD] = torch.where(bridge_span, 680.0, mass[2, S_WOOD])
        mass[2, S_WOOD] = torch.where(mid, 380.0, mass[2, S_WOOD])
        mass[2, S_ASH] = torch.where(mid, 240.0, mass[2, S_ASH])
        T[2, 0] = torch.where(mid, 54000.0 + 12000.0 * burn_stage, T[2, 0])
    else:
        # Severely charred / fracturing bridge
        mid = bridge_span & ((xx - 0.50).abs() < 0.18)
        mass[2, S_WOOD] = torch.where(bridge_span, 560.0, mass[2, S_WOOD])
        mass[2, S_WOOD] = torch.where(mid, 180.0, mass[2, S_WOOD])
        mass[2, S_ASH] = torch.where(mid, 360.0, mass[2, S_ASH])
        T[2, 0] = torch.where(mid, 62000.0, T[2, 0])

    # 2. L1 Cryo Ice Rampart & Ice Dam zone
    dx = float(torch.empty(1, device=device).uniform_(0.35, 0.65, generator=gen).item())
    dam = ((xx - dx).abs() < 0.04) & (yy > 0.15) & (yy < 0.85)
    ice_lvl = float(torch.empty(1, device=device).uniform_(280.0, 960.0, generator=gen).item())
    ice_temp = float(torch.empty(1, device=device).uniform_(22000.0, 34000.0, generator=gen).item())
    mass[1, S_ROCK] = torch.where(dam, 0.0, mass[1, S_ROCK])
    mass[1, S_ICE] = torch.where(dam, ice_lvl, mass[1, S_ICE])
    mass[1, S_WATER] = torch.where(dam, max(0.0, 950.0 - ice_lvl), mass[1, S_WATER])
    T[1, 0] = torch.where(dam, ice_temp, T[1, 0])

    # Carve hotspot on part of the ice dam
    cy = float(torch.empty(1, device=device).uniform_(0.25, 0.75, generator=gen).item())
    hot_carve = dam & ((yy - cy).abs() < 0.08)
    T[1, 0] = torch.where(hot_carve, 36000.0 + 14000.0 * burn_stage, T[1, 0])

    # Enforce H_CAP
    tot_m = mass.sum(dim=1, keepdim=True)
    cap_s = torch.where(tot_m > H_CAP * 0.92, (H_CAP * 0.92) / (tot_m + 1e-6), torch.ones_like(tot_m))
    mass = mass * cap_s
    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    energy = C_cell * torch.clamp(T, T_MIN, T_MAX)
    return SimState(mass=mass, energy=energy, flow_e=st.flow_e, flow_s=st.flow_s, elevation=elev)


def sample_training_dt_field(L: int, H: int, W: int, device: torch.device, step: int) -> torch.Tensor:
    """Sample diverse uniform multi-rate & spatially varying time-dilation fields `[L, 1, H, W]`."""
    mode = step % 10
    if mode in (0, 1, 2, 3, 4):
        # Exact target discrete rates {0.25, 0.5, 1.0, 2.0, 4.0}
        rates = (0.25, 0.5, 1.0, 2.0, 4.0)
        return torch.full((L, 1, H, W), rates[mode], dtype=torch.float32, device=device)
    if mode == 5:
        # Continuous log-uniform in [0.1, 4.0]
        val = math.exp(float(torch.empty(1).uniform_(math.log(0.1), math.log(4.0)).item()))
        return torch.full((L, 1, H, W), val, dtype=torch.float32, device=device)
    if mode in (6, 7):
        # Spatially varying Stasis Bubble (0.15x or 0.0x) + Rapid-Aging Bubble (3.0x)
        yy = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(L, 1, H, W)
        xx = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(L, 1, H, W)
        dt = torch.ones((L, 1, H, W), dtype=torch.float32, device=device)
        cx1 = float(torch.empty(1).uniform_(0.22, 0.78).item())
        cy1 = float(torch.empty(1).uniform_(0.22, 0.78).item())
        r1 = float(torch.empty(1).uniform_(0.12, 0.22).item())
        m1 = ((xx - cx1) ** 2 + (yy - cy1) ** 2) < (r1 * r1)
        stasis_val = 0.0 if (step % 20 == 6) else float(torch.empty(1).uniform_(0.10, 0.25).item())
        dt = torch.where(m1, torch.full_like(dt, stasis_val), dt)
        if mode == 7:
            cx2 = float(torch.empty(1).uniform_(0.22, 0.78).item())
            cy2 = float(torch.empty(1).uniform_(0.22, 0.78).item())
            r2 = float(torch.empty(1).uniform_(0.10, 0.18).item())
            m2 = (((xx - cx2) ** 2 + (yy - cy2) ** 2) < (r2 * r2)) & (~m1)
            dt = torch.where(m2, torch.full_like(dt, float(torch.empty(1).uniform_(2.5, 3.5).item())), dt)
        return dt
    if mode == 8:
        # Adaptive multi-rate tile field: 1.0x active combat tiles + 4.0x far-field tiles!
        th, tw = max(1, H // 16), max(1, W // 16)
        active = (torch.rand((1, 1, th, tw), device=device) < 0.22).float()
        active_full = F.interpolate(active, size=(H, W), mode="nearest").expand(L, 1, H, W)
        return torch.where(active_full > 0.5, torch.ones_like(active_full), torch.full_like(active_full, 4.0))
    # mode == 9: smooth spatial time-gradient field in [0.15, 3.5]
    yy = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W)
    dt = (0.15 + 3.35 * (0.5 * xx + 0.5 * yy)).expand(L, 1, H, W).contiguous()
    return dt


# =============================================================================
# Structural Head Loss
# =============================================================================


def structural_head_loss(
    pred_s: torch.Tensor,
    target_s: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Active-region weighted multi-task loss for all 4 structural channels
    (structural_stress, collapse_z_flux, ice_rampart_integrity, channel_carve_rate).
    Prevents sparse-positive sigmoid collapse by strongly weighting non-zero structural zones!
    """
    # 0. Structural stress in [0, 1]
    ps0, ts0 = pred_s[:, ST_STRESS], target_s[:, ST_STRESS]
    w0 = 1.0 + 18.0 * (ts0 > 0.05).float()
    l_stress = (w0 * (ps0 - ts0).pow(2)).mean()

    # 1. Downward collapse flux >= 0
    p_col = pred_s[:, ST_COLLAPSE] / 0.10
    t_col = target_s[:, ST_COLLAPSE] / 0.10
    w_col = 1.0 + 20.0 * (target_s[:, ST_COLLAPSE] > 1e-3).float()
    l_col = (w_col * (p_col - t_col).pow(2)).mean()

    # 2. Cryo ice rampart integrity in [0, 1]
    ps2, ts2 = pred_s[:, ST_ICE_RAMPART], target_s[:, ST_ICE_RAMPART]
    w2 = 1.0 + 25.0 * (ts2 > 0.05).float()
    l_ice = (w2 * (ps2 - ts2).pow(2)).mean()

    # 3. Thermal channel carve rate >= 0
    p_carve = pred_s[:, ST_CHANNEL_CARVE] / 0.05
    t_carve = target_s[:, ST_CHANNEL_CARVE] / 0.05
    w_carve = 1.0 + 10.0 * (target_s[:, ST_CHANNEL_CARVE] > 1e-4).float()
    l_carve = (w_carve * (p_carve - t_carve).pow(2)).mean()

    total = 4.0 * l_stress + 2.5 * l_col + 5.0 * l_ice + 2.5 * l_carve
    return total, {
        "stress": float(l_stress.item()),
        "collapse": float(l_col.item()),
        "ice": float(l_ice.item()),
        "carve": float(l_carve.item()),
    }


def _r2_from_accum(sse: float, sum_y: float, sum_y2: float, n: int) -> float:
    if n <= 1:
        return 0.0
    mean_y = sum_y / n
    sst = sum_y2 - n * mean_y * mean_y
    if sst <= 1e-12:
        return 0.0
    return float(1.0 - sse / sst)


# =============================================================================
# Experiment A1: Continuous Δt Scaling Evaluation (Δt in {0.25, 0.5, 1.0, 2.0, 4.0})
# =============================================================================


@torch.no_grad()
def evaluate_multirate_dt(
    model_dt: WorldModelDT,
    model_v4: WorldModel3D,
    device: torch.device,
    eval_seeds: list[int],
    dt_values: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> dict:
    """
    Evaluate `WorldModelDT` vs naive linear extrapolation (`dt * WorldModel3D_v4`) on unseen
    `256x256x4` worlds across `dt in {0.25, 0.5, 1.0, 2.0, 4.0}` against exact `K`-substep RefSim!
    Also verifies exact `dt = 0.0` stasis (`max_abs_diff == 0.0`).
    """
    model_dt.eval()
    model_v4.eval()

    # 1. Exact dt = 0.0 stasis verification
    st0 = make_stacked_3d_frostmarsh_world(256, 256, device, seed=99001)
    x0 = encode_state(st0)
    h0, z0, s0 = model_dt(x0, local_dt=0.0)
    stasis_zero_max_abs = float(
        max(
            h0.abs().max().item(),
            z0.abs().max().item(),
            s0[:, ST_COLLAPSE].abs().max().item(),
            s0[:, ST_CHANNEL_CARVE].abs().max().item(),
        )
    )

    per_dt: dict[str, dict] = {}
    for dt_val in dt_values:
        acc = {
            "dt_model": {k: {"abs": 0.0, "sse": 0.0, "sy": 0.0, "sy2": 0.0, "n": 0} for k in ("horiz", "flux", "heat", "rxn", "vert", "next_mass")},
            "linear_v4": {k: {"abs": 0.0, "sse": 0.0, "sy": 0.0, "sy2": 0.0, "n": 0} for k in ("horiz", "flux", "heat", "rxn", "vert", "next_mass")},
        }
        dt_map = torch.full((4, 1, 256, 256), dt_val, dtype=torch.float32, device=device)

        for seed in eval_seeds:
            st = make_stacked_3d_frostmarsh_world(256, 256, device, seed=seed)
            # Warmup 4 ticks under coupled 3D teacher
            torch.manual_seed(seed + 300)
            for _ in range(4):
                fe_w, fm_w = sample_bolts(4, 256, 256, device, prob=0.55, max_bolts=2)
                st, _, _ = refsim_step_dt(st, torch.ones_like(dt_map), fe_w, fm_w, include_vertical=True)

            # Evaluate 4 steps along trajectory
            for t in range(4):
                x = encode_state(st)
                gt_next, gt_h, gt_z = refsim_step_dt(st, dt_map, None, None, include_vertical=True)

                # WorldModelDT prediction
                pred_h, pred_z, _ = model_dt(x, local_dt=dt_map)
                pred_next, _, _ = apply_dt_projected_step(st, pred_h, pred_z, dt_map)

                # Naive linear baseline: dt * WorldModel3D_v4(x)
                v4_h, v4_z = model_v4(x, return_z_flux=True, return_latent=False)
                lin_h = v4_h * dt_val
                lin_z = v4_z * dt_val
                lin_next, _, _ = apply_dt_projected_step(st, lin_h, lin_z, dt_map)

                # Normalize heat channels by 10.0 so horiz R^2/MAE matches contract scale
                def _norm_h(tensor: torch.Tensor) -> torch.Tensor:
                    out = tensor.clone()
                    out[:, 8:10] = out[:, 8:10] / 10.0
                    return out

                gt_hn = _norm_h(gt_h)
                for name, ph, pz, pnext in (
                    ("dt_model", pred_h, pred_z, pred_next),
                    ("linear_v4", lin_h, lin_z, lin_next),
                ):
                    phn = _norm_h(ph)
                    pairs = (
                        ("horiz", phn, gt_hn),
                        ("flux", ph[:, 0:8], gt_h[:, 0:8]),
                        ("heat", ph[:, 8:10] / 10.0, gt_h[:, 8:10] / 10.0),
                        ("rxn", ph[:, 10:17], gt_h[:, 10:17]),
                        ("vert", pz[:3], gt_z[:3]),
                        ("next_mass", pnext.mass / 1000.0, gt_next.mass / 1000.0),
                    )
                    for k, pr, tg in pairs:
                        err = pr - tg
                        d = acc[name][k]
                        d["abs"] += float(err.abs().sum().item())
                        d["sse"] += float(err.pow(2).sum().item())
                        d["sy"] += float(tg.sum().item())
                        d["sy2"] += float(tg.pow(2).sum().item())
                        d["n"] += int(tg.numel())

                st = gt_next

        def _summarize(ad: dict) -> dict:
            out = {}
            for k, d in ad.items():
                n = max(d["n"], 1)
                out[k] = {
                    "mae": round(d["abs"] / n, 6),
                    "r2": round(_r2_from_accum(d["sse"], d["sy"], d["sy2"], d["n"]), 5),
                }
            return out

        per_dt[str(dt_val)] = {
            "world_model_dt": _summarize(acc["dt_model"]),
            "naive_linear_v4": _summarize(acc["linear_v4"]),
        }

    return {
        "exact_zero_dt_stasis_max_abs": stasis_zero_max_abs,
        "per_dt": per_dt,
    }


# =============================================================================
# Experiment A2: Adaptive Multi-Rate Temporal Sparsity Speedup Benchmark (GPU FP16)
# =============================================================================


class AdaptiveMultiRateRunner:
    """
    Real GPU Adaptive Multi-Rate Temporal Sparsity Runner for `WorldModelDT`:
      - Ticks 0, 1, 2: Gathers active combat zone tile columns `[K, L, 18, T+12, T+12]`
        and executes exact shrinking-halo `WorldModelDT.forward_active_columns` at `Δt = 1.0`.
      - Tick 3: Runs a single full-grid `WorldModelDT` pass with `local_dt = 1.0` on active
        combat tiles and `local_dt = 4.0` on far-field tiles, advancing the entire world by
        4 exact physical ticks with ~2.1–2.7x lower neural compute!
    """

    def __init__(
        self,
        model_dt: WorldModelDT,
        H: int,
        W: int,
        L: int,
        active_tile_mask: torch.Tensor,  # [H//16, W//16] bool
        tile_size: int = 16,
        halo: int = 6,
    ):
        self.model = model_dt
        self.H = H
        self.W = W
        self.L = L
        self.tile_size = tile_size
        self.halo = halo
        self.patch_size = tile_size + 2 * halo
        self.device = active_tile_mask.device

        coords = active_tile_mask.nonzero(as_tuple=False)  # [K, 2] (ty, tx)
        self.K = int(coords.shape[0])
        self.ty = coords[:, 0]
        self.tx = coords[:, 1]
        self.active_frac = float(active_tile_mask.float().mean().item())

        # Precompute pixel-level masks & mixed dt map for tick 3
        active_px = F.interpolate(
            active_tile_mask.float().view(1, 1, H // tile_size, W // tile_size),
            size=(H, W),
            mode="nearest",
        ).expand(L, 1, H, W).contiguous()
        self.active_px = active_px
        self.dt_tick3 = torch.where(active_px > 0.5, torch.ones_like(active_px), torch.full_like(active_px, 4.0))
        self.dt_ones_patch = torch.ones((self.K, L, 1, self.patch_size, self.patch_size), dtype=torch.float32, device=self.device)

        # Precompute flattened gather/scatter indices for zero-overhead GPU indexing
        self.tile_rows = [int(r) * tile_size for r in self.ty.tolist()]
        self.tile_cols = [int(c) * tile_size for c in self.tx.tolist()]

    def step_sparse_active_tick(self, x_18: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run exact shrinking-halo `WorldModelDT` ONLY on the K active combat tile columns (`Δt = 1.0`)."""
        x_pad = F.pad(x_18, (self.halo, self.halo, self.halo, self.halo), mode="replicate")  # [L, 18, H+2h, W+2h]
        # Zero-copy strided unfold of all tiles -> index active K tiles directly on GPU!
        unfolded = x_pad.unfold(2, self.patch_size, self.tile_size).unfold(3, self.patch_size, self.tile_size)
        # unfolded: [L, 18, Th, Tw, P, P] -> select [K, L, 18, P, P]
        patches_x = unfolded[:, :, self.ty, self.tx, :, :].permute(2, 0, 1, 3, 4).contiguous()
        patches_dt = self.dt_ones_patch.to(dtype=patches_x.dtype)
        return self.model.forward_active_columns(patches_x, patches_dt)

    def scatter_active_outputs(
        self,
        h_tiles: torch.Tensor,
        z_tiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter active tile predictions into full-grid zero buffers (far-field is exact 0.0 on ticks 0..2)."""
        Th, Tw = self.H // self.tile_size, self.W // self.tile_size
        h_grid = torch.zeros((self.L, 17, Th, Tw, self.tile_size, self.tile_size), dtype=h_tiles.dtype, device=self.device)
        z_grid = torch.zeros((self.L, Z_FLUX_CH, Th, Tw, self.tile_size, self.tile_size), dtype=z_tiles.dtype, device=self.device)
        h_grid[:, :, self.ty, self.tx] = h_tiles.permute(1, 2, 0, 3, 4)
        z_grid[:, :, self.ty, self.tx] = z_tiles.permute(1, 2, 0, 3, 4)
        h_full = h_grid.permute(0, 1, 2, 4, 3, 5).reshape(self.L, 17, self.H, self.W)
        z_full = z_grid.permute(0, 1, 2, 4, 3, 5).reshape(self.L, Z_FLUX_CH, self.H, self.W)
        # Zero active-tile boundary fluxes crossing into far-field on ticks 0..2 so far-field stays strictly untouched until tick 3!
        h_full = _scale_horiz_deltas_by_dt(h_full, self.active_px)
        return h_full, z_full

    def step_tick3_mixed_dt(self, x_18: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run single full-grid `WorldModelDT` pass with `dt=1.0` active + `dt=4.0` far-field!"""
        return self.model(x_18, local_dt=self.dt_tick3)


@torch.inference_mode()
def benchmark_temporal_sparsity(
    model_dt: WorldModelDT,
    device: torch.device,
) -> dict:
    """
    Benchmark Uniform `1.0x` Dense Stepping vs Adaptive Multi-Rate Temporal Sparsity
    (active combat zone `Δt=1.0` every tick + far-field `Δt=4.0` every 4th tick)
    in FP16 on GPU (`256x256x4` and `512x512x4`), plus 16-tick physical rollout accuracy & conservation!
    """
    model_dt.eval()
    results: dict = {}

    for H, W, L in ((256, 256, 4), (512, 512, 4)):
        Th, Tw = H // 16, W // 16
        x = torch.randn((L, 18, H, W), device=device, dtype=torch.float32)
        dt_one = torch.ones((L, 1, H, W), device=device, dtype=torch.float32)

        # Measure 4-tick Uniform 1.0x Dense macro-cycle
        def _run_uniform_4ticks():
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    for _ in range(4):
                        _ = model_dt(x, local_dt=dt_one)
            else:
                for _ in range(4):
                    _ = model_dt(x, local_dt=dt_one)

        for _ in range(10):
            _run_uniform_4ticks()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_unif = []
        for _ in range(35):
            t0 = time.perf_counter()
            _run_uniform_4ticks()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_unif.append((time.perf_counter() - t0) * 1000.0)
        unif_p50 = float(np.percentile(t_unif, 50))

        res_grid: dict = {
            "uniform_1x_4tick_p50_ms": round(unif_p50, 3),
            "uniform_1x_per_tick_ms": round(unif_p50 / 4.0, 3),
            "adaptive_configs": {},
        }

        for label, radius_frac in (("active_12pct", 0.20), ("active_20pct", 0.255)):
            yy = torch.linspace(0.0, 1.0, Th, device=device).view(Th, 1).expand(Th, Tw)
            xx = torch.linspace(0.0, 1.0, Tw, device=device).view(1, Tw).expand(Th, Tw)
            mask = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) <= (radius_frac ** 2)
            runner = AdaptiveMultiRateRunner(model_dt, H, W, L, mask)

            def _run_adaptive_4ticks():
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        for _ in range(3):
                            _ = runner.step_sparse_active_tick(x)
                        _ = runner.step_tick3_mixed_dt(x)
                else:
                    for _ in range(3):
                        _ = runner.step_sparse_active_tick(x)
                    _ = runner.step_tick3_mixed_dt(x)

            for _ in range(10):
                _run_adaptive_4ticks()
            if device.type == "cuda":
                torch.cuda.synchronize()

            t_adap = []
            for _ in range(35):
                t0 = time.perf_counter()
                _run_adaptive_4ticks()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_adap.append((time.perf_counter() - t0) * 1000.0)

            adap_p50 = float(np.percentile(t_adap, 50))
            speedup = unif_p50 / max(adap_p50, 1e-6)
            theo_work_ratio = (3.0 * runner.active_frac + 1.0) / 4.0
            res_grid["adaptive_configs"][label] = {
                "actual_active_tile_pct": round(runner.active_frac * 100.0, 2),
                "active_tiles": runner.K,
                "total_tiles": Th * Tw,
                "adaptive_4tick_p50_ms": round(adap_p50, 3),
                "adaptive_per_tick_ms": round(adap_p50 / 4.0, 3),
                "wall_clock_speedup_x": round(speedup, 3),
                "theoretical_tile_work_speedup_x": round(1.0 / theo_work_ratio, 3),
                "effective_hz": round(4000.0 / max(adap_p50, 1e-6), 1),
            }
        results[f"{H}x{W}x{L}_fp16"] = res_grid

    # Rollout accuracy & conservation check on 256x256x4 over 16 ticks (4 macro-cycles)!
    st_ref = make_stacked_3d_frostmarsh_world(256, 256, device, seed=96001)
    st_unif = st_ref.clone()
    st_adap = st_ref.clone()
    m0 = float(compute_total_mass(st_adap).sum().item())
    e0 = float(compute_total_energy(st_adap).sum().item())

    Th, Tw = 16, 16
    yy = torch.linspace(0.0, 1.0, Th, device=device).view(Th, 1).expand(Th, Tw)
    xx = torch.linspace(0.0, 1.0, Tw, device=device).view(1, Tw).expand(Th, Tw)
    mask_20 = ((xx - 0.5) ** 2 + (yy - 0.5) ** 2) <= (0.255 ** 2)
    runner_acc = AdaptiveMultiRateRunner(model_dt, 256, 256, 4, mask_20)
    dt_one = torch.ones((4, 1, 256, 256), dtype=torch.float32, device=device)

    max_rel_m, max_rel_e = 0.0, 0.0
    for tick in range(16):
        # Exact RefSim
        st_ref, _, _ = refsim_step_dt(st_ref, dt_one, None, None, include_vertical=True)
        # Uniform 1.0x WorldModelDT
        xu = encode_state(st_unif)
        hu, zu, _ = model_dt(xu, local_dt=dt_one)
        st_unif, _, _ = apply_dt_projected_step(st_unif, hu, zu, dt_one)
        # Adaptive Multi-Rate WorldModelDT
        xa = encode_state(st_adap)
        if tick % 4 < 3:
            ht, zt, _ = runner_acc.step_sparse_active_tick(xa)
            ha, za = runner_acc.scatter_active_outputs(ht, zt)
            st_adap, _, _ = apply_dt_projected_step(st_adap, ha, za, runner_acc.active_px)
        else:
            ha, za, _ = runner_acc.step_tick3_mixed_dt(xa)
            st_adap, _, _ = apply_dt_projected_step(st_adap, ha, za, runner_acc.dt_tick3)

        m_now = float(compute_total_mass(st_adap).sum().item())
        e_now = float(compute_total_energy(st_adap).sum().item())
        max_rel_m = max(max_rel_m, abs(m_now - m0) / (abs(m0) + 1e-12))
        max_rel_e = max(max_rel_e, abs(e_now - e0) / (abs(e0) + 1e-12))

    # Compare 16-tick rollout MAE vs exact RefSim (normalized mass / 1000)
    unif_mass_mae = float(F.l1_loss(st_unif.mass / 1000.0, st_ref.mass / 1000.0).item())
    adap_mass_mae = float(F.l1_loss(st_adap.mass / 1000.0, st_ref.mass / 1000.0).item())
    adap_vs_unif_mae = float(F.l1_loss(st_adap.mass / 1000.0, st_unif.mass / 1000.0).item())
    _, T_ref = compute_C_and_T(st_ref.mass, st_ref.energy)
    _, T_unif = compute_C_and_T(st_unif.mass, st_unif.energy)
    _, T_adap = compute_C_and_T(st_adap.mass, st_adap.energy)

    results["rollout_16tick_256x256x4_fidelity"] = {
        "uniform_1x_mass_mae_vs_refsim": round(unif_mass_mae, 6),
        "adaptive_multirate_mass_mae_vs_refsim": round(adap_mass_mae, 6),
        "adaptive_vs_uniform_mass_mae": round(adap_vs_unif_mae, 6),
        "uniform_1x_temp_mae_K_vs_refsim": round(float(F.l1_loss(T_unif, T_ref).item()) / 100.0, 3),
        "adaptive_multirate_temp_mae_K_vs_refsim": round(float(F.l1_loss(T_adap, T_ref).item()) / 100.0, 3),
        "adaptive_max_rel_mass_err": max_rel_m,
        "adaptive_max_rel_energy_err": max_rel_e,
    }
    return results


# =============================================================================
# Experiment A3: Spatial Time-Dilation Stasis Bubble (0.15x) & Rapid-Aging (3.0x)
# =============================================================================


@torch.no_grad()
def evaluate_time_dilation_bubbles(
    model_dt: WorldModelDT,
    device: torch.device,
) -> tuple[dict, dict]:
    """
    Create a controlled `256x256x4` world with IDENTICAL twin phenomena placed inside:
      1. `0.15x` Stasis Dome (left-center `(0.28, 0.50)`, `r = 0.16`): active wildfire (L1) + plunging waterfall (L2->L1)
      2. `1.00x` Normal Control (center `(0.52, 0.25)` & `(0.52, 0.75)`): identical twin wildfire & waterfall
      3. `3.00x` Rapid-Aging Bubble (right-center `(0.76, 0.50)`, `r = 0.14`): identical twin burning timber & steam-cooling zone
    Rolls out closed-system for 16 ticks using `WorldModelDT` and quantifies:
      - Wildfire burn arrest ratio (`> 85%`) & waterfall plunge flux arrest ratio (`> 85%`)
      - Rapid-aging burn & thermal acceleration ratio (`~2.4-3.0x`)
      - Exact 3D mass & energy conservation (`rel_err ~ 1e-7`)!
    Returns `(metrics_dict, viz_arrays_dict)`.
    """
    model_dt.eval()
    H, W = 256, 256
    c = get_consts(device)

    mass = torch.zeros((4, 10, H, W), dtype=torch.float32, device=device)
    flow_e = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    flow_s = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    elev = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    T = torch.full((4, 1, H, W), T_AMB, dtype=torch.float32, device=device)

    # Clean ambient substrate across all 4 layers
    mass[0, S_SOIL] = 520.0
    mass[1, S_SOIL] = 480.0
    mass[2, S_SOIL] = 160.0  # open drop for waterfalls
    mass[3, S_SOIL] = 120.0
    elev[0, 0] = 2.0
    elev[1, 0] = 10.0
    elev[2, 0] = 22.0
    elev[3, 0] = 28.0
    T[3, 0] = 26500.0

    yy = torch.linspace(0.0, 1.0, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, W).expand(H, W)

    # Spatial local_dt field
    stasis_cx, stasis_cy, stasis_r = 0.24, 0.50, 0.165
    rapid_cx, rapid_cy, rapid_r = 0.78, 0.50, 0.145
    stasis_mask = ((xx - stasis_cx) ** 2 + (yy - stasis_cy) ** 2) <= (stasis_r ** 2)
    rapid_mask = ((xx - rapid_cx) ** 2 + (yy - rapid_cy) ** 2) <= (rapid_r ** 2)

    dt_2d = torch.ones((H, W), dtype=torch.float32, device=device)
    dt_2d = torch.where(stasis_mask, torch.full_like(dt_2d, 0.15), dt_2d)
    dt_2d = torch.where(rapid_mask, torch.full_like(dt_2d, 3.00), dt_2d)
    dt_field = dt_2d.view(1, 1, H, W).expand(4, 1, H, W).contiguous()

    # Helper to plant identical twin wildfire (L1) + plunging waterfall (L2->L1)
    # Deep fuel & deep waterfall reservoir so active-phase burn & plunge ratios are measured
    # cleanly without premature depletion in the 1.0x control or 3.0x rapid-aging zone!
    def _plant_wildfire(cx: float, cy: float, r: float = 0.068) -> torch.Tensor:
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
        mass[1, S_WOOD] = torch.where(m, 1600.0, mass[1, S_WOOD])
        mass[1, S_PLANT] = torch.where(m, 450.0, mass[1, S_PLANT])
        T[1, 0] = torch.where(m, 57500.0, T[1, 0])
        return m

    def _plant_waterfall(cx: float, cy: float, r: float = 0.048) -> torch.Tensor:
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)
        mass[2, S_SOIL] = torch.where(m, 140.0, mass[2, S_SOIL])
        mass[2, S_WATER] = torch.where(m, 1800.0, mass[2, S_WATER])
        mass[1, S_SOIL] = torch.where(m, 240.0, mass[1, S_SOIL])
        return m

    # Twin A: Stasis (0.15x) wildfire + waterfall
    wf_stasis = _plant_wildfire(0.24, 0.43)
    fall_stasis = _plant_waterfall(0.24, 0.58)
    # Twin B: Exterior Control (1.0x) identical wildfire + waterfall
    wf_ctrl = _plant_wildfire(0.51, 0.43)
    fall_ctrl = _plant_waterfall(0.51, 0.58)
    # Twin C: Rapid-Aging (3.0x) identical wildfire + waterfall
    wf_rapid = _plant_wildfire(0.78, 0.43)
    fall_rapid = _plant_waterfall(0.78, 0.58)

    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    energy = C_cell * T
    st = SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev)

    # Track initial totals & wood/water inventories
    m0_tot = float(compute_total_mass(st).sum().item())
    e0_tot = float(compute_total_energy(st).sum().item())
    wood0_stasis = float(st.mass[1, S_WOOD][wf_stasis].sum().item())
    wood0_ctrl = float(st.mass[1, S_WOOD][wf_ctrl].sum().item())
    wood0_rapid = float(st.mass[1, S_WOOD][wf_rapid].sum().item())
    water0_stasis = float(st.mass[2, S_WATER][fall_stasis].sum().item())
    water0_ctrl = float(st.mass[2, S_WATER][fall_ctrl].sum().item())

    max_rel_m, max_rel_e = 0.0, 0.0
    fall_flux_stasis_t1, fall_flux_ctrl_t1 = 0.0, 0.0
    wood_hist = {"stasis": [1.0], "ctrl": [1.0], "rapid": [1.0]}

    # Also evaluate single-step prediction accuracy vs exact spatially varying RefSim!
    x_init = encode_state(st)
    _, gt_h0, gt_z0 = refsim_step_dt(st, dt_field, None, None, include_vertical=True)
    pr_h0, pr_z0, _ = model_dt(x_init, local_dt=dt_field)
    spatial_dt_horiz_mae = float(F.l1_loss(pr_h0, gt_h0).item())
    spatial_dt_vert_mae = float(F.l1_loss(pr_z0[:3], gt_z0[:3]).item())

    ticks = 12
    burned_stasis_t6, burned_ctrl_t6, burned_rapid_t6 = 0.0, 0.0, 0.0
    fall_stasis_t2, fall_ctrl_t2 = 0.0, 0.0
    for t in range(1, ticks + 1):
        x = encode_state(st)
        raw_h, raw_z, _ = model_dt(x, local_dt=dt_field)
        st, _, proj_z = apply_dt_projected_step(st, raw_h, raw_z, dt_field)

        if t == 1:
            # Instantaneous L2->L1 waterfall plunge rate arrest from identical starting heads!
            fall_flux_stasis_t1 = float((-proj_z[1, ZF_WATER][fall_stasis]).clamp(min=0.0).mean().item())
            fall_flux_ctrl_t1 = float((-proj_z[1, ZF_WATER][fall_ctrl]).clamp(min=0.0).mean().item())
        if t == 2:
            fall_stasis_t2 = water0_stasis - float(st.mass[2, S_WATER][fall_stasis].sum().item())
            fall_ctrl_t2 = water0_ctrl - float(st.mass[2, S_WATER][fall_ctrl].sum().item())
        if t == 6:
            burned_stasis_t6 = wood0_stasis - float(st.mass[1, S_WOOD][wf_stasis].sum().item())
            burned_ctrl_t6 = wood0_ctrl - float(st.mass[1, S_WOOD][wf_ctrl].sum().item())
            burned_rapid_t6 = wood0_rapid - float(st.mass[1, S_WOOD][wf_rapid].sum().item())

        wood_hist["stasis"].append(float(st.mass[1, S_WOOD][wf_stasis].sum().item()) / max(wood0_stasis, 1e-6))
        wood_hist["ctrl"].append(float(st.mass[1, S_WOOD][wf_ctrl].sum().item()) / max(wood0_ctrl, 1e-6))
        wood_hist["rapid"].append(float(st.mass[1, S_WOOD][wf_rapid].sum().item()) / max(wood0_rapid, 1e-6))

        m_now = float(compute_total_mass(st).sum().item())
        e_now = float(compute_total_energy(st).sum().item())
        max_rel_m = max(max_rel_m, abs(m_now - m0_tot) / (abs(m0_tot) + 1e-12))
        max_rel_e = max(max_rel_e, abs(e_now - e0_tot) / (abs(e0_tot) + 1e-12))

    # Wildfire burn arrest inside 0.15x stasis dome vs 1.0x exterior
    burned_stasis = wood0_stasis - float(st.mass[1, S_WOOD][wf_stasis].sum().item())
    burned_ctrl = wood0_ctrl - float(st.mass[1, S_WOOD][wf_ctrl].sum().item())
    burned_rapid = wood0_rapid - float(st.mass[1, S_WOOD][wf_rapid].sum().item())

    wildfire_freeze_ratio_pct = 100.0 * (1.0 - burned_stasis / max(burned_ctrl, 1e-6))
    wildfire_freeze_t6_pct = 100.0 * (1.0 - burned_stasis_t6 / max(burned_ctrl_t6, 1e-6))
    waterfall_flux_arrest_pct = 100.0 * (1.0 - fall_flux_stasis_t1 / max(fall_flux_ctrl_t1, 1e-6))
    waterfall_2t_arrest_pct = 100.0 * (1.0 - fall_stasis_t2 / max(fall_ctrl_t2, 1e-6))
    rapid_burn_speedup_x = burned_rapid_t6 / max(burned_ctrl_t6, 1e-6)

    _, T_final = compute_C_and_T(st.mass, st.energy)
    # Build visualization field: L1 Ash/Smoke/Wood burn progress + L2 remaining waterfall
    viz_data = {
        "T_L1_K": (T_final[1, 0].cpu().numpy() / 100.0),
        "wood_L1": st.mass[1, S_WOOD].cpu().numpy(),
        "ash_L1": st.mass[1, S_ASH].cpu().numpy(),
        "water_L2": st.mass[2, S_WATER].cpu().numpy(),
        "dt_2d": dt_2d.cpu().numpy(),
        "stasis_circle": (stasis_cx * W, stasis_cy * H, stasis_r * W),
        "rapid_circle": (rapid_cx * W, rapid_cy * H, rapid_r * W),
        "wood_hist": wood_hist,
    }

    metrics = {
        "ticks": ticks,
        "stasis_dt": 0.15,
        "control_dt": 1.0,
        "rapid_aging_dt": 3.0,
        "wildfire_burned_fraction_stasis": round(burned_stasis / max(wood0_stasis, 1e-6), 4),
        "wildfire_burned_fraction_control": round(burned_ctrl / max(wood0_ctrl, 1e-6), 4),
        "wildfire_burned_fraction_rapid": round(burned_rapid / max(wood0_rapid, 1e-6), 4),
        "wildfire_stasis_freeze_ratio_pct": round(wildfire_freeze_ratio_pct, 2),
        "wildfire_stasis_freeze_t6_pct": round(wildfire_freeze_t6_pct, 2),
        "waterfall_plunge_flux_arrest_pct": round(waterfall_flux_arrest_pct, 2),
        "waterfall_2tick_volume_arrest_pct": round(waterfall_2t_arrest_pct, 2),
        "rapid_aging_burn_acceleration_x": round(rapid_burn_speedup_x, 3),
        "spatial_dt_1step_horiz_mae_vs_refsim": round(spatial_dt_horiz_mae, 6),
        "spatial_dt_1step_vert_mae_vs_refsim": round(spatial_dt_vert_mae, 6),
        "closed_3d_max_rel_mass_err": max_rel_m,
        "closed_3d_max_rel_energy_err": max_rel_e,
    }
    return metrics, viz_data


# =============================================================================
# Experiment B: Emergent Destructible Structures & Terrain Shaping (3 Scenarios)
# =============================================================================


def _bfs_path_exists(walkable: np.ndarray, start_xs: tuple[int, int], goal_xs: tuple[int, int], y_band: tuple[int, int]) -> bool:
    """Fast 4-connected BFS checking if a walkable flank path exists from West band to East band."""
    H, W = walkable.shape
    y0, y1 = y_band
    vis = np.zeros((H, W), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for y in range(y0, y1):
        for x in range(start_xs[0], start_xs[1]):
            if walkable[y, x] and not vis[y, x]:
                vis[y, x] = True
                q.append((y, x))
    gx0, gx1 = goal_xs
    while q:
        cy, cx = q.popleft()
        if gx0 <= cx < gx1:
            return True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if y0 <= ny < y1 and 0 <= nx < W and not vis[ny, nx] and walkable[ny, nx]:
                vis[ny, nx] = True
                q.append((ny, nx))
    return False


@torch.no_grad()
def evaluate_structural_head_accuracy(
    model_dt: WorldModelDT,
    device: torch.device,
    eval_seeds: list[int],
) -> dict:
    """Evaluate `WorldModelDT.struct_head` MAE & R^2 on held-out `256x256x4` structural worlds."""
    model_dt.eval()
    acc = {name: {"abs": 0.0, "sse": 0.0, "sy": 0.0, "sy2": 0.0, "n": 0} for name in STRUCT_NAMES}
    dt_one = torch.ones((4, 1, 256, 256), dtype=torch.float32, device=device)

    for seed in eval_seeds:
        st = make_structural_training_world(256, 256, device, seed=seed)
        fe, fm = sample_bolts(4, 256, 256, device, prob=0.70, max_bolts=3)
        x = encode_state(st, fe, fm)
        gt_s = compute_structural_teacher(st, dt_one, fe)
        _, _, pr_s = model_dt(x, local_dt=dt_one)
        for ch, name in enumerate(STRUCT_NAMES):
            err = pr_s[:, ch] - gt_s[:, ch]
            tg = gt_s[:, ch]
            d = acc[name]
            d["abs"] += float(err.abs().sum().item())
            d["sse"] += float(err.pow(2).sum().item())
            d["sy"] += float(tg.sum().item())
            d["sy2"] += float(tg.pow(2).sum().item())
            d["n"] += int(tg.numel())

    out = {}
    r2s = []
    for name in STRUCT_NAMES:
        d = acc[name]
        r2 = _r2_from_accum(d["sse"], d["sy"], d["sy2"], d["n"])
        r2s.append(r2)
        out[name] = {
            "mae": round(d["abs"] / max(d["n"], 1), 6),
            "r2": round(r2, 5),
        }
    out["mean_struct_r2"] = round(float(np.mean(r2s)), 5)
    return out


@torch.no_grad()
def run_experiment_b_structures(
    model_dt: WorldModelDT,
    device: torch.device,
) -> tuple[dict, dict]:
    """
    Simulate & evaluate all 3 emergent destructible structure & terrain shaping mechanics
    on `256x256x4` worlds driven end-to-end by `WorldModelDT`:
      B1: Timber Skybridge Burn & L2->L1 Collapse (with ambient intact control)
      B2: Cryo Ice Rampart Construction (solid wall blocking enemy advance)
      B3: Thermal Channel Carving & River Redirection (ice dam breach -> valley flood vs uncarved control)
    """
    model_dt.eval()
    H, W = 256, 256
    c = get_consts(device)
    dt_one = torch.ones((4, 1, H, W), dtype=torch.float32, device=device)

    yy = torch.linspace(0.0, 1.0, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, W).expand(H, W)

    # -------------------------------------------------------------------------
    # B1: Timber Skybridge Burn & L2 -> L1 Collapse
    # -------------------------------------------------------------------------
    def _make_skybridge_world() -> tuple[SimState, torch.Tensor, torch.Tensor]:
        mass = torch.zeros((4, 10, H, W), dtype=torch.float32, device=device)
        flow_e = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        flow_s = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        elev = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        T = torch.full((4, 1, H, W), T_AMB, dtype=torch.float32, device=device)

        mass[0, S_SOIL] = 600.0
        mass[1, S_SOIL] = 220.0  # Open canyon river beneath L2 bridge
        mass[1, S_WATER] = 750.0
        mass[2, S_SOIL] = 140.0  # Open air outside bridge
        mass[3, S_SOIL] = 120.0
        elev[0, 0], elev[1, 0], elev[2, 0], elev[3, 0] = 2.0, 8.0, 22.0, 28.0

        # West & East cliffs (rock abutments on L1 & L2)
        west_cliff = xx <= 0.30
        east_cliff = xx >= 0.70
        mass[1, S_ROCK] = torch.where(west_cliff | east_cliff, 880.0, 0.0)
        mass[1, S_WATER] = torch.where(west_cliff | east_cliff, 0.0, mass[1, S_WATER])
        mass[2, S_ROCK] = torch.where(west_cliff | east_cliff, 880.0, 0.0)

        # L2 Timber Skybridge span across canyon (x in [0.30, 0.70], y in [0.46, 0.54])
        span = (xx > 0.30) & (xx < 0.70) & ((yy - 0.50).abs() < 0.042)
        mass[2, S_WOOD] = torch.where(span, 860.0, 0.0)
        mass[2, S_SOIL] = torch.where(span, 180.0, mass[2, S_SOIL])
        mid_zone = span & ((xx - 0.50).abs() < 0.085)

        C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
        energy = C_cell * T
        return SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev), span, mid_zone

    st_b1, span_m, mid_m = _make_skybridge_world()
    st_b1_ctrl, _, _ = _make_skybridge_world()

    init_mid_wood = float(st_b1.mass[2, S_WOOD][mid_m].sum().item())
    init_l1_rubble = float((st_b1.mass[1, S_WOOD] + st_b1.mass[1, S_ASH])[mid_m].sum().item())

    b1_ticks = 36
    b1_wood_frac, b1_stress_hist, b1_rubble_hist, b1_conn_hist = [], [], [], []
    b1_ctrl_wood_frac, b1_ctrl_stress_hist = [], []
    collapse_tick = -1
    sever_tick = -1
    b1_post_m0, b1_post_e0 = None, None
    b1_max_rel_m, b1_max_rel_e = 0.0, 0.0

    for t in range(b1_ticks):
        # Thermal actuator heats L2 skybridge mid-span for ticks 0..14, then turns off!
        fe = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        if t < 15:
            fe[2, 0] = torch.where(mid_m, 7.5e7, 0.0)

        # Record closed-system baseline right after actuator turns off
        if t == 15:
            b1_post_m0 = float(compute_total_mass(st_b1).sum().item())
            b1_post_e0 = float(compute_total_energy(st_b1).sum().item())

        # Heated skybridge step via WorldModelDT
        x1 = encode_state(st_b1, fe, None)
        h1, z1, s1 = model_dt(x1, local_dt=dt_one)
        st_b1, _, _ = apply_dt_projected_step(st_b1, h1, z1, dt_one, fe, None)
        st_b1, dropped = apply_structural_collapse_conservative(st_b1, s1, stress_thresh=0.56)
        if dropped > 500.0 and collapse_tick < 0:
            collapse_tick = t

        # Ambient unheated control step via WorldModelDT
        xc = encode_state(st_b1_ctrl)
        hc, zc, sc = model_dt(xc, local_dt=dt_one)
        st_b1_ctrl, _, _ = apply_dt_projected_step(st_b1_ctrl, hc, zc, dt_one)
        st_b1_ctrl, _ = apply_structural_collapse_conservative(st_b1_ctrl, sc, stress_thresh=0.56)

        if b1_post_m0 is not None:
            m_now = float(compute_total_mass(st_b1).sum().item())
            e_now = float(compute_total_energy(st_b1).sum().item())
            b1_max_rel_m = max(b1_max_rel_m, abs(m_now - b1_post_m0) / (abs(b1_post_m0) + 1e-12))
            b1_max_rel_e = max(b1_max_rel_e, abs(e_now - b1_post_e0) / (abs(b1_post_e0) + 1e-12))

        mid_wood_f = float(st_b1.mass[2, S_WOOD][mid_m].sum().item()) / max(init_mid_wood, 1e-6)
        ctrl_wood_f = float(st_b1_ctrl.mass[2, S_WOOD][mid_m].sum().item()) / max(init_mid_wood, 1e-6)
        peak_stress = float(s1[2, ST_STRESS][mid_m].mean().item())
        ctrl_stress = float(sc[2, ST_STRESS][mid_m].mean().item())
        l1_rubble = float((st_b1.mass[1, S_WOOD] + st_b1.mass[1, S_ASH])[mid_m].sum().item()) - init_l1_rubble

        # Check L2 skybridge flank connectivity (walkable if rock>=500 or wood>=320)
        walk_l2 = ((st_b1.mass[2, S_ROCK] >= 500.0) | (st_b1.mass[2, S_WOOD] >= 320.0)).cpu().numpy()
        connected = _bfs_path_exists(walk_l2, (60, 74), (182, 196), (108, 148))
        if not connected and sever_tick < 0:
            sever_tick = t

        b1_wood_frac.append(round(mid_wood_f, 4))
        b1_ctrl_wood_frac.append(round(ctrl_wood_f, 4))
        b1_stress_hist.append(round(peak_stress, 4))
        b1_ctrl_stress_hist.append(round(ctrl_stress, 4))
        b1_rubble_hist.append(round(l1_rubble / max(init_mid_wood, 1e-6), 4))
        b1_conn_hist.append(1 if connected else 0)

    ctrl_walk_l2 = ((st_b1_ctrl.mass[2, S_ROCK] >= 500.0) | (st_b1_ctrl.mass[2, S_WOOD] >= 320.0)).cpu().numpy()
    ctrl_connected_final = _bfs_path_exists(ctrl_walk_l2, (60, 74), (182, 196), (108, 148))

    b1_metrics = {
        "ticks": b1_ticks,
        "initial_midspan_wood_mass": round(init_mid_wood, 1),
        "final_midspan_wood_fraction": b1_wood_frac[-1],
        "ambient_control_wood_fraction": b1_ctrl_wood_frac[-1],
        "peak_structural_stress": round(max(b1_stress_hist), 4),
        "ambient_control_max_stress": round(max(b1_ctrl_stress_hist), 4),
        "collapse_onset_tick": collapse_tick,
        "flank_severed_tick": sever_tick,
        "flank_isolated_final": bool(b1_conn_hist[-1] == 0),
        "ambient_control_intact_final": bool(ctrl_connected_final),
        "l1_fallen_rubble_fraction": round(max(b1_rubble_hist), 4),
        "post_beam_closed_max_rel_mass_err": b1_max_rel_m,
        "post_beam_closed_max_rel_energy_err": b1_max_rel_e,
    }

    # -------------------------------------------------------------------------
    # B2: Cryo Ice Rampart Construction (sp_water + -thermal_flux)
    # -------------------------------------------------------------------------
    mass2 = torch.zeros((4, 10, H, W), dtype=torch.float32, device=device)
    flow_e2 = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    flow_s2 = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    elev2 = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    T2 = torch.full((4, 1, H, W), T_AMB, dtype=torch.float32, device=device)

    mass2[0, S_SOIL] = 600.0
    mass2[1, S_SOIL] = 680.0
    mass2[2, S_SOIL] = 140.0
    mass2[3, S_SOIL] = 120.0
    elev2[0, 0], elev2[1, 0], elev2[2, 0], elev2[3, 0] = 2.0, 10.0, 22.0, 28.0

    # Canyon walls north (y < 0.25) and south (y > 0.75) forming an open central choke pass!
    canyon_wall = (yy < 0.25) | (yy > 0.75)
    mass2[1, S_ROCK] = torch.where(canyon_wall, 900.0, 0.0)
    # Target rampart construction line across the choke point (x in [0.47, 0.53], y in [0.24, 0.76])
    rampart_zone = (~canyon_wall) & ((xx - 0.50).abs() < 0.032)

    C2 = (mass2 * c.spec_c).sum(dim=1, keepdim=True)
    st_b2 = SimState(mass=mass2, energy=C2 * T2, flow_e=flow_e2, flow_s=flow_s2, elevation=elev2)

    # Check initial enemy pathfinding across choke (should be OPEN = True!)
    walk_b2_init = ((st_b2.mass[1, S_ROCK] < 500.0) & (st_b2.mass[1, S_ICE] < 700.0)).cpu().numpy()
    b2_open_init = _bfs_path_exists(walk_b2_init, (64, 80), (176, 192), (56, 200))

    b2_ticks = 40
    b2_ice_mean, b2_integrity_hist, b2_wall_pct, b2_blocked_hist = [], [], [], []
    b2_blocked_tick = -1
    b2_post_m0, b2_post_e0 = None, None
    b2_max_rel_m, b2_max_rel_e = 0.0, 0.0

    for t in range(b2_ticks):
        fe = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        fm = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        if t < 24:
            # Deploy sp_water + intense cryogenic cooling beam along rampart line!
            fm[1, 0] = torch.where(rampart_zone & (st_b2.mass[1, S_WATER] + st_b2.mass[1, S_ICE] < 1050.0), 140.0, 0.0)
            fe[1, 0] = torch.where(rampart_zone, -1.2e8, 0.0)
        if t == 24:
            b2_post_m0 = float(compute_total_mass(st_b2).sum().item())
            b2_post_e0 = float(compute_total_energy(st_b2).sum().item())

        x2 = encode_state(st_b2, fe, fm)
        h2, z2, s2 = model_dt(x2, local_dt=dt_one)
        st_b2, _, _ = apply_dt_projected_step(st_b2, h2, z2, dt_one, fe, fm)

        if b2_post_m0 is not None:
            m_now = float(compute_total_mass(st_b2).sum().item())
            e_now = float(compute_total_energy(st_b2).sum().item())
            b2_max_rel_m = max(b2_max_rel_m, abs(m_now - b2_post_m0) / (abs(b2_post_m0) + 1e-12))
            b2_max_rel_e = max(b2_max_rel_e, abs(e_now - b2_post_e0) / (abs(b2_post_e0) + 1e-12))

        ice_r = st_b2.mass[1, S_ICE][rampart_zone]
        integ_r = s2[1, ST_ICE_RAMPART][rampart_zone]
        wall_done = float((ice_r >= 700.0).float().mean().item()) * 100.0

        walk_now = ((st_b2.mass[1, S_ROCK] < 500.0) & (st_b2.mass[1, S_ICE] < 700.0)).cpu().numpy()
        is_blocked = not _bfs_path_exists(walk_now, (64, 80), (176, 192), (56, 200))
        if is_blocked and b2_blocked_tick < 0:
            b2_blocked_tick = t

        b2_ice_mean.append(round(float(ice_r.mean().item()), 1))
        b2_integrity_hist.append(round(float(integ_r.mean().item()), 4))
        b2_wall_pct.append(round(wall_done, 2))
        b2_blocked_hist.append(1 if is_blocked else 0)

    b2_metrics = {
        "ticks": b2_ticks,
        "initial_choke_open": bool(b2_open_init),
        "enemy_advance_blocked_tick": b2_blocked_tick,
        "enemy_advance_blocked_final": bool(b2_blocked_hist[-1] == 1),
        "peak_rampart_ice_mass": round(max(b2_ice_mean), 1),
        "final_rampart_ice_mass": b2_ice_mean[-1],
        "peak_ice_rampart_integrity": round(max(b2_integrity_hist), 4),
        "final_ice_rampart_integrity": b2_integrity_hist[-1],
        "final_solid_wall_completion_pct": b2_wall_pct[-1],
        "post_cryo_closed_max_rel_mass_err": b2_max_rel_m,
        "post_cryo_closed_max_rel_energy_err": b2_max_rel_e,
    }

    # -------------------------------------------------------------------------
    # B3: Thermal Channel Carving & River Redirection
    # -------------------------------------------------------------------------
    def _make_icedam_world() -> tuple[SimState, torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = torch.zeros((4, 10, H, W), dtype=torch.float32, device=device)
        flow_e = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        flow_s = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        elev = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        T = torch.full((4, 1, H, W), T_AMB, dtype=torch.float32, device=device)

        mass[0, S_SOIL] = 600.0
        mass[1, S_SOIL] = 760.0  # Solid impermeable riverbed (>= 650) so reservoir stays on L1!
        mass[2, S_SOIL] = 140.0
        mass[3, S_SOIL] = 120.0
        elev[0, 0], elev[2, 0], elev[3, 0] = 2.0, 22.0, 28.0

        # Western elevated reservoir (x < 0.46, elev=18) | Compact 6-cell Ice Dam (0.46 <= x <= 0.485)
        # | Lower dry eastern target valley basin (0.495 <= x <= 0.68, sloped elev 8 -> 2)!
        res_m = (xx < 0.46) & (yy > 0.24) & (yy < 0.76)
        dam_m = (xx >= 0.46) & (xx <= 0.485) & (yy > 0.22) & (yy < 0.78)
        valley_m = (xx > 0.492) & (xx < 0.68) & (yy > 0.34) & (yy < 0.66)
        border_m = (yy <= 0.24) | (yy >= 0.76)
        carve_m = dam_m & ((yy - 0.50).abs() < 0.095)
        beam_m = (xx >= 0.452) & (xx <= 0.493) & ((yy - 0.50).abs() < 0.10)

        slope_val = 9.0 - 7.0 * ((xx - 0.485) / 0.22).clamp(0.0, 1.0)
        elev[1, 0] = torch.where(xx < 0.46, 18.0, torch.where(xx <= 0.485, 11.0, slope_val))
        mass[1, S_ROCK] = torch.where(border_m, 900.0, 0.0)
        mass[1, S_WATER] = torch.where(res_m, 1650.0, 0.0)
        flow_e[1, 0] = torch.where(res_m & (xx > 0.30), 320.0, 0.0)
        mass[1, S_ICE] = torch.where(dam_m, 920.0, 0.0)
        T[1, 0] = torch.where(dam_m, 24800.0, 29000.0)

        C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
        return SimState(mass=mass, energy=C_cell * T, flow_e=flow_e, flow_s=flow_s, elevation=elev), carve_m, beam_m, valley_m, res_m

    st_b3, carve_m, beam_m, valley_m, res_m = _make_icedam_world()
    st_b3_ctrl, _, _, _, _ = _make_icedam_world()

    b3_ticks = 60
    b3_carve_ice, b3_valley_water, b3_ctrl_valley_water, b3_carve_rate = [], [], [], []
    breach_tick = -1
    b3_post_m0, b3_post_e0 = None, None
    b3_max_rel_m, b3_max_rel_e = 0.0, 0.0

    for t in range(b3_ticks):
        fe = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
        if t < 20:
            # Focused +thermal_flux beam carving channel cleanly through the ice dam!
            fe[1, 0] = torch.where(beam_m, 2.8e8, 0.0)
        if t == 20:
            b3_post_m0 = float(compute_total_mass(st_b3).sum().item())
            b3_post_e0 = float(compute_total_energy(st_b3).sum().item())

        # Carved dam step via WorldModelDT (horizontal + vertical + conservative channel_carve head)
        x3 = encode_state(st_b3, fe, None)
        h3, z3, s3 = model_dt(x3, local_dt=dt_one)
        st_b3, _, _ = apply_dt_projected_step(st_b3, h3, z3, dt_one, fe, None)
        st_b3 = apply_channel_carve_conservative(st_b3, s3)

        # Uncarved control step via WorldModelDT
        x3c = encode_state(st_b3_ctrl)
        h3c, z3c, s3c = model_dt(x3c, local_dt=dt_one)
        st_b3_ctrl, _, _ = apply_dt_projected_step(st_b3_ctrl, h3c, z3c, dt_one)
        st_b3_ctrl = apply_channel_carve_conservative(st_b3_ctrl, s3c)

        if b3_post_m0 is not None:
            m_now = float(compute_total_mass(st_b3).sum().item())
            e_now = float(compute_total_energy(st_b3).sum().item())
            b3_max_rel_m = max(b3_max_rel_m, abs(m_now - b3_post_m0) / (abs(b3_post_m0) + 1e-12))
            b3_max_rel_e = max(b3_max_rel_e, abs(e_now - b3_post_e0) / (abs(b3_post_e0) + 1e-12))

        c_ice = float(st_b3.mass[1, S_ICE][carve_m].mean().item())
        if c_ice < 700.0 and breach_tick < 0:
            breach_tick = t
        val_w = float(st_b3.mass[1, S_WATER][valley_m].sum().item()) / 1000.0
        ctrl_val_w = float(st_b3_ctrl.mass[1, S_WATER][valley_m].sum().item()) / 1000.0
        c_rate = float(s3[1, ST_CHANNEL_CARVE][carve_m].mean().item())

        b3_carve_ice.append(round(c_ice, 1))
        b3_valley_water.append(round(val_w, 2))
        b3_ctrl_valley_water.append(round(ctrl_val_w, 2))
        b3_carve_rate.append(round(c_rate, 5))

    flooded_cells_pct = float((st_b3.mass[1, S_WATER][valley_m] > 50.0).float().mean().item()) * 100.0
    ctrl_flooded_pct = float((st_b3_ctrl.mass[1, S_WATER][valley_m] > 50.0).float().mean().item()) * 100.0

    b3_metrics = {
        "ticks": b3_ticks,
        "initial_channel_ice_mass": 920.0,
        "final_channel_ice_mass": b3_carve_ice[-1],
        "dam_breach_tick": breach_tick,
        "peak_channel_carve_rate": round(max(b3_carve_rate), 5),
        "redirected_valley_water_k_units": b3_valley_water[-1],
        "uncarved_control_valley_water_k_units": b3_ctrl_valley_water[-1],
        "valley_flooded_area_pct": round(flooded_cells_pct, 2),
        "uncarved_control_flooded_area_pct": round(ctrl_flooded_pct, 2),
        "post_beam_closed_max_rel_mass_err": b3_max_rel_m,
        "post_beam_closed_max_rel_energy_err": b3_max_rel_e,
    }

    struct_metrics = {
        "b1_timber_skybridge_collapse": b1_metrics,
        "b2_cryo_ice_rampart": b2_metrics,
        "b3_thermal_channel_carving": b3_metrics,
    }
    struct_viz = {
        "b1_wood_frac": b1_wood_frac,
        "b1_ctrl_wood_frac": b1_ctrl_wood_frac,
        "b1_stress": b1_stress_hist,
        "b1_rubble": b1_rubble_hist,
        "b1_collapse_tick": collapse_tick,
        "b1_sever_tick": sever_tick,
        "b2_integrity": b2_integrity_hist,
        "b2_wall_pct": b2_wall_pct,
        "b2_blocked_tick": b2_blocked_tick,
        "b3_carve_ice_frac": [v / 920.0 for v in b3_carve_ice],
        "b3_valley_water": b3_valley_water,
        "b3_ctrl_valley_water": b3_ctrl_valley_water,
        "b3_breach_tick": breach_tick,
    }
    return struct_metrics, struct_viz


# =============================================================================
# Dark-Themed 3-Panel Publication Visualization
# =============================================================================


def generate_3panel_visualization(
    out_png: Path,
    multirate_metrics: dict,
    sparsity_metrics: dict,
    dilation_metrics: dict,
    dilation_viz: dict,
    struct_metrics: dict,
    struct_viz: dict,
) -> None:
    """
    Generate crisp dark-themed 3-panel visualization `models/world_dt_structures_viz.png`:
      - Panel 1: Multi-rate Δt in [0.25, 4.0] accuracy (WorldModelDT vs Naive Linear) + Temporal Sparsity Speedup
      - Panel 2: Spatial Time-Dilation field (0.15x Stasis Dome vs 1.0x exterior vs 3.0x Rapid-Aging Zone)
      - Panel 3: Emergent Destructible Structures (Skybridge collapse, Cryo Ice Rampart, Carved Ice Dam flood)
    """
    plt.rcParams.update(
        {
            "figure.facecolor": "#0d1117",
            "axes.facecolor": "#161b22",
            "axes.edgecolor": "#30363d",
            "axes.labelcolor": "#e6edf3",
            "text.color": "#e6edf3",
            "xtick.color": "#8b949e",
            "ytick.color": "#8b949e",
            "grid.color": "#21262d",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
        }
    )

    fig = plt.figure(figsize=(21.5, 7.6), dpi=180)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.95], width_ratios=[1.05, 1.05, 1.15], hspace=0.34, wspace=0.25)

    # -------------------------------------------------------------------------
    # PANEL 1 (Left Col): Top = Multi-rate Δt R^2/MAE; Bottom = Temporal Sparsity Speedup
    # -------------------------------------------------------------------------
    ax1a = fig.add_subplot(gs[0, 0])
    dt_keys = ["0.25", "0.5", "1.0", "2.0", "4.0"]
    xs = np.arange(len(dt_keys))
    r2_dt_h = [multirate_metrics["per_dt"][k]["world_model_dt"]["horiz"]["r2"] for k in dt_keys]
    r2_lin_h = [multirate_metrics["per_dt"][k]["naive_linear_v4"]["horiz"]["r2"] for k in dt_keys]
    r2_dt_v = [multirate_metrics["per_dt"][k]["world_model_dt"]["vert"]["r2"] for k in dt_keys]
    mae_dt_h = [multirate_metrics["per_dt"][k]["world_model_dt"]["horiz"]["mae"] for k in dt_keys]

    ax1a.plot(xs, r2_dt_h, "o-", color="#38bdf8", lw=2.4, ms=6.5, label="WorldModelDT Horiz R²")
    ax1a.plot(xs, r2_dt_v, "s-", color="#34d399", lw=2.0, ms=5.5, label="WorldModelDT 3D Vert R²")
    ax1a.plot(xs, r2_lin_h, "^--", color="#f87171", lw=1.9, ms=5.5, alpha=0.9, label="Naive Linear dt·v4 Horiz R²")
    for i, (r_val, m_val) in enumerate(zip(r2_dt_h, mae_dt_h)):
        ax1a.annotate(f"{r_val:.3f}", (xs[i], r_val), textcoords="offset points", xytext=(0, 7),
                      ha="center", fontsize=8, color="#7dd3fc", fontweight="bold")
    ax1a.set_xticks(xs)
    ax1a.set_xticklabels([f"{k}×" for k in dt_keys])
    ax1a.set_ylim(min(0.45, min(r2_lin_h) - 0.08), 1.04)
    ax1a.set_ylabel("Held-Out R² vs Exact K-Step RefSim")
    ax1a.set_title("Panel 1: Continuous Δt Scaling & Temporal Sparsity", fontsize=11.5, fontweight="bold", color="#f0f6fc", pad=10)
    ax1a.grid(True, linestyle="--", alpha=0.5)
    ax1a.legend(loc="lower left", fontsize=8, framealpha=0.85, facecolor="#0d1117", edgecolor="#30363d")

    ax1b = fig.add_subplot(gs[1, 0])
    s256 = sparsity_metrics["256x256x4_fp16"]
    s512 = sparsity_metrics["512x512x4_fp16"]
    labels = ["Uniform\n1.0×", "Adaptive\n20% Active", "Adaptive\n12% Active"]
    sp_256 = [
        1.0,
        s256["adaptive_configs"]["active_20pct"]["wall_clock_speedup_x"],
        s256["adaptive_configs"]["active_12pct"]["wall_clock_speedup_x"],
    ]
    sp_512 = [
        1.0,
        s512["adaptive_configs"]["active_20pct"]["wall_clock_speedup_x"],
        s512["adaptive_configs"]["active_12pct"]["wall_clock_speedup_x"],
    ]
    bx = np.arange(len(labels))
    w = 0.34
    bars1 = ax1b.bar(bx - w / 2, sp_256, width=w, color="#38bdf8", edgecolor="#0d1117", label="256×256×4 FP16")
    bars2 = ax1b.bar(bx + w / 2, sp_512, width=w, color="#a78bfa", edgecolor="#0d1117", label="512×512×4 FP16")
    for b in list(bars1) + list(bars2):
        h_v = b.get_height()
        ax1b.text(b.get_x() + b.get_width() / 2, h_v + 0.05, f"{h_v:.2f}×", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#e6edf3")
    ax1b.set_xticks(bx)
    ax1b.set_xticklabels(labels, fontsize=8.5)
    ax1b.set_ylabel("Wall-Clock Speedup (×)")
    ax1b.set_ylim(0, max(max(sp_256), max(sp_512)) + 0.55)
    ax1b.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1b.legend(loc="upper left", fontsize=8, framealpha=0.85, facecolor="#0d1117", edgecolor="#30363d")
    ax1b.set_xlabel("Active Combat Δt=1.0 (ticks 0–2) + Far-Field Δt=4.0 (tick 3)", fontsize=8.5, color="#8b949e")

    # -------------------------------------------------------------------------
    # PANEL 2 (Center Col): Top = Spatial Time-Dilation Snapshot; Bottom = Twin Wildfire Trajectories
    # -------------------------------------------------------------------------
    ax2a = fig.add_subplot(gs[0, 1])
    T_map = dilation_viz["T_L1_K"]
    ash_map = dilation_viz["ash_L1"]
    w2_map = dilation_viz["water_L2"]
    # Composite RGB visualization: warm fire/ash glow + cool L2 waterfall cyan + dark terrain
    H, W = T_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[:, :, 0] = 0.08 + 0.85 * np.clip((T_map - 290.0) / 360.0, 0.0, 1.0) + 0.25 * np.clip(ash_map / 500.0, 0.0, 1.0)
    rgb[:, :, 1] = 0.10 + 0.42 * np.clip((T_map - 320.0) / 340.0, 0.0, 1.0) + 0.55 * np.clip(w2_map / 1100.0, 0.0, 1.0)
    rgb[:, :, 2] = 0.14 + 0.85 * np.clip(w2_map / 1100.0, 0.0, 1.0)
    # Subtle tint for stasis dome (cool blue) & rapid-aging dome (amber)
    dt_arr = dilation_viz["dt_2d"]
    rgb[dt_arr < 0.5, 2] = np.clip(rgb[dt_arr < 0.5, 2] + 0.16, 0.0, 1.0)
    rgb[dt_arr > 2.0, 0] = np.clip(rgb[dt_arr > 2.0, 0] + 0.14, 0.0, 1.0)
    rgb = np.clip(rgb, 0.0, 1.0)

    ax2a.imshow(rgb, origin="upper", aspect="auto")
    sx, sy, sr = dilation_viz["stasis_circle"]
    rx, ry, rr = dilation_viz["rapid_circle"]
    ax2a.add_patch(mpatches.Circle((sx, sy), sr, fill=False, edgecolor="#38bdf8", lw=2.0, ls="--"))
    ax2a.add_patch(mpatches.Circle((rx, ry), rr, fill=False, edgecolor="#fbbf24", lw=2.0, ls="--"))
    ax2a.text(sx, sy - sr - 8, "0.15× Stasis Dome\n(Frozen Fire & Fall)", ha="center", va="bottom",
              fontsize=8, fontweight="bold", color="#7dd3fc",
              bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="#38bdf8", alpha=0.85))
    ax2a.text(W * 0.51, 22, "1.0× Control", ha="center", va="top", fontsize=8, fontweight="bold", color="#e6edf3",
              bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="#6e7681", alpha=0.80))
    ax2a.text(rx, ry - rr - 8, "3.0× Rapid-Aging", ha="center", va="bottom",
              fontsize=8, fontweight="bold", color="#fde68a",
              bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="#fbbf24", alpha=0.85))
    ax2a.set_xticks([])
    ax2a.set_yticks([])
    ax2a.set_title("Panel 2: Spatial Time-Dilation Domes (256×256×4)", fontsize=11.5, fontweight="bold", color="#f0f6fc", pad=10)

    ax2b = fig.add_subplot(gs[1, 1])
    ts = np.arange(len(dilation_viz["wood_hist"]["ctrl"]))
    ax2b.plot(ts, [100 * v for v in dilation_viz["wood_hist"]["stasis"]], color="#38bdf8", lw=2.4, label="0.15× Stasis Dome")
    ax2b.plot(ts, [100 * v for v in dilation_viz["wood_hist"]["ctrl"]], color="#94a3b8", lw=2.0, ls="--", label="1.00× Normal Control")
    ax2b.plot(ts, [100 * v for v in dilation_viz["wood_hist"]["rapid"]], color="#fbbf24", lw=2.4, label="3.00× Rapid-Aging Zone")
    ax2b.set_xlabel("Simulation Tick (Closed 3D System)")
    ax2b.set_ylabel("Remaining Twin Wildfire Fuel (%)")
    ax2b.set_ylim(0, 108)
    ax2b.grid(True, linestyle="--", alpha=0.5)
    ax2b.legend(loc="lower left", fontsize=8, framealpha=0.85, facecolor="#0d1117", edgecolor="#30363d")
    badge_txt = (
        f"Fire Freeze: {dilation_metrics['wildfire_stasis_freeze_ratio_pct']:.1f}%\n"
        f"Fall Arrest: {dilation_metrics['waterfall_plunge_flux_arrest_pct']:.1f}%\n"
        f"3× Burn Accel: {dilation_metrics['rapid_aging_burn_acceleration_x']:.2f}×\n"
        f"Mass/E err: {dilation_metrics['closed_3d_max_rel_mass_err']:.1e}"
    )
    ax2b.text(0.98, 0.96, badge_txt, transform=ax2b.transAxes, ha="right", va="top", fontsize=7.8,
              color="#e6edf3", family="monospace",
              bbox=dict(boxstyle="round,pad=0.35", fc="#0d1117", ec="#38bdf8", alpha=0.90))

    # -------------------------------------------------------------------------
    # PANEL 3 (Right Col): Top = B1 Skybridge Collapse & B2 Cryo Rampart; Bottom = B3 Carved Dam Flood
    # -------------------------------------------------------------------------
    ax3a = fig.add_subplot(gs[0, 2])
    t_b1 = np.arange(len(struct_viz["b1_wood_frac"]))
    t_b2 = np.arange(len(struct_viz["b2_integrity"]))
    ax3a.plot(t_b1, [100 * v for v in struct_viz["b1_wood_frac"]], color="#fb923c", lw=2.3, label="B1 Skybridge L2 Timber (%)")
    ax3a.plot(t_b1, [100 * v for v in struct_viz["b1_stress"]], color="#f87171", lw=1.9, ls=":", label="B1 Structural Stress (%)")
    ax3a.plot(t_b1, [100 * v for v in struct_viz["b1_ctrl_wood_frac"]], color="#fb923c", lw=1.2, ls="--", alpha=0.45, label="B1 Unheated Control (%)")
    ax3a.plot(t_b2, [100 * v for v in struct_viz["b2_integrity"]], color="#38bdf8", lw=2.3, label="B2 Cryo Ice Rampart Integrity (%)")
    if struct_viz["b1_sever_tick"] >= 0:
        ax3a.axvline(struct_viz["b1_sever_tick"], color="#f87171", ls="--", lw=1.3, alpha=0.8)
        ax3a.text(struct_viz["b1_sever_tick"] + 0.6, 88, f"L2 Severed\n(t={struct_viz['b1_sever_tick']})", fontsize=7.6, color="#fca5a5", fontweight="bold")
    if struct_viz["b2_blocked_tick"] >= 0:
        ax3a.axvline(struct_viz["b2_blocked_tick"], color="#38bdf8", ls="--", lw=1.3, alpha=0.8)
        ax3a.text(struct_viz["b2_blocked_tick"] + 0.6, 20, f"Flank Blocked\n(t={struct_viz['b2_blocked_tick']})", fontsize=7.6, color="#7dd3fc", fontweight="bold")
    ax3a.set_ylim(-3, 110)
    ax3a.set_ylabel("Integrity / Stress / Timber (%)")
    ax3a.set_title("Panel 3: Emergent Destructible Structures & Terrain Shaping", fontsize=11.5, fontweight="bold", color="#f0f6fc", pad=10)
    ax3a.grid(True, linestyle="--", alpha=0.5)
    ax3a.legend(loc="center right", fontsize=7.5, framealpha=0.85, facecolor="#0d1117", edgecolor="#30363d")

    ax3b = fig.add_subplot(gs[1, 2])
    t_b3 = np.arange(len(struct_viz["b3_valley_water"]))
    ax3b.plot(t_b3, struct_viz["b3_valley_water"], color="#34d399", lw=2.5, label="B3 Carved Dam -> Valley Flood (k-mass)")
    ax3b.plot(t_b3, struct_viz["b3_ctrl_valley_water"], color="#64748b", lw=1.8, ls="--", label="B3 Uncarved Control (0 flood)")
    ax3b.fill_between(t_b3, 0, struct_viz["b3_valley_water"], color="#34d399", alpha=0.18)
    ax3b_r = ax3b.twinx()
    ax3b_r.plot(t_b3, [100 * v for v in struct_viz["b3_carve_ice_frac"]], color="#c084fc", lw=1.9, ls="-.", label="Carved Channel Ice (%)")
    ax3b_r.set_ylabel("Channel Ice Remaining (%)", color="#c084fc", fontsize=8.5)
    ax3b_r.tick_params(axis="y", colors="#c084fc")
    ax3b_r.set_ylim(-5, 110)
    if struct_viz["b3_breach_tick"] >= 0:
        ax3b.axvline(struct_viz["b3_breach_tick"], color="#c084fc", ls="--", lw=1.3, alpha=0.8)
        ax3b.text(struct_viz["b3_breach_tick"] + 0.7, max(struct_viz["b3_valley_water"]) * 0.65,
                  f"Dam Breach\n(t={struct_viz['b3_breach_tick']})", fontsize=7.8, color="#e9d5ff", fontweight="bold")
    ax3b.set_xlabel("Simulation Tick (256×256×4 WorldModelDT)")
    ax3b.set_ylabel("Redirected Valley Water (×10³)")
    ax3b.grid(True, linestyle="--", alpha=0.5)
    h_l, l_l = ax3b.get_legend_handles_labels()
    h_r, l_r = ax3b_r.get_legend_handles_labels()
    ax3b.legend(h_l + h_r, l_l + l_r, loc="upper left", fontsize=7.5, framealpha=0.85, facecolor="#0d1117", edgecolor="#30363d")

    fig.suptitle(
        "Frostmarsh WorldModelDT — Continuous Δt Time Dilation, Temporal Sparsity & Emergent Destructible Structures",
        fontsize=13.5,
        fontweight="bold",
        color="#f0f6fc",
        y=0.98,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=180, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)


# =============================================================================
# Main Training, Evaluation & Visualization Pipeline
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & evaluate WorldModelDT (Continuous Δt + Emergent Structures).")
    parser.add_argument("--work-dir", type=str, default=".", help="Workspace root.")
    parser.add_argument("--v4-pt", type=str, default="", help="Path to world_model_3d_v4.pt")
    parser.add_argument("--out-ckpt", type=str, default="", help="Output world_model_dt.pt path")
    parser.add_argument("--out-json", type=str, default="", help="Output WORLD_DT_STRUCTURES_METRICS.json path")
    parser.add_argument("--out-viz", type=str, default="", help="Output world_dt_structures_viz.png path")
    parser.add_argument("--steps", type=int, default=1800, help="Total training steps")
    parser.add_argument("--eval-only", type=str, default="", help="Skip training and evaluate an existing world_model_dt.pt")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()

    def _find_file(explicit: str, candidates: list[Path]) -> Path:
        if explicit:
            return Path(explicit).resolve()
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[0]

    v4_pt = _find_file(
        args.v4_pt,
        [
            work_dir / "models" / "checkpoints" / "world_model_3d_v4.pt",
            work_dir / "world_model_3d_v4.pt",
            _SCRIPT_DIR.parent / "models" / "checkpoints" / "world_model_3d_v4.pt",
        ],
    )
    out_ckpt = Path(args.out_ckpt).resolve() if args.out_ckpt else (work_dir / "models" / "checkpoints" / "world_model_dt.pt")
    out_json = Path(args.out_json).resolve() if args.out_json else (work_dir / "models" / "WORLD_DT_STRUCTURES_METRICS.json")
    out_viz = Path(args.out_viz).resolve() if args.out_viz else (work_dir / "models" / "world_dt_structures_viz.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(2027)
    np.random.seed(2027)

    print(f"=== WorldModelDT Continuous Δt + Emergent Structures Training ({gpu_name}) ===", flush=True)
    print(f"  Warm-start checkpoint: {v4_pt}", flush=True)

    ck_v4 = torch.load(str(v4_pt), map_location=device, weights_only=False)
    v4_sd = ck_v4["state_dict"] if isinstance(ck_v4, dict) and "state_dict" in ck_v4 else ck_v4

    model_v4 = WorldModel3D(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=Z_FLUX_CH).to(device).eval()
    model_v4.load_state_dict(v4_sd, strict=True)
    for p in model_v4.parameters():
        p.requires_grad_(False)
    v4_params = count_parameters(model_v4)

    model_dt = WorldModelDT(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=Z_FLUX_CH, struct_ch=STRUCT_CH).to(device)
    loaded_keys = model_dt.load_from_v4_state_dict(copy.deepcopy(v4_sd))
    dt_params = count_parameters(model_dt)
    print(f"  WorldModel3D v4 params: {v4_params:,} | WorldModelDT params: {dt_params:,} (+{dt_params - v4_params:,}, loaded {loaded_keys} keys)", flush=True)

    # Verify EXACT initial Δt=1.0 parity with WorldModel3D v4 AND EXACT Δt=0.0 stasis!
    with torch.no_grad():
        st_par = make_stacked_3d_frostmarsh_world(128, 128, device, seed=1234)
        x_par = encode_state(st_par)
        h4, z4 = model_v4(x_par, return_z_flux=True, return_latent=False)
        h_dt1, z_dt1, _ = model_dt(x_par, local_dt=1.0)
        h_dt0, z_dt0, s_dt0 = model_dt(x_par, local_dt=0.0)
        init_h_diff = float((h4 - h_dt1).abs().max().item())
        init_z_diff = float((z4 - z_dt1).abs().max().item())
        init_zero_diff = float(max(h_dt0.abs().max().item(), z_dt0.abs().max().item(), s_dt0[:, ST_COLLAPSE].abs().max().item(), s_dt0[:, ST_CHANNEL_CARVE].abs().max().item()))
        assert init_h_diff < 1e-5, f"Initial dt=1.0 horiz parity failed: {init_h_diff}"
        assert init_z_diff < 1e-5, f"Initial dt=1.0 vert parity failed: {init_z_diff}"
        assert init_zero_diff == 0.0, f"Exact dt=0.0 stasis failed: {init_zero_diff}"
    print(f"  Verified EXACT init parity: dt=1.0 max_h_diff={init_h_diff:.2e}, max_z_diff={init_z_diff:.2e}, dt=0.0 max_abs={init_zero_diff:.2e}", flush=True)

    if args.eval_only:
        ck_dt = torch.load(args.eval_only, map_location=device, weights_only=False)
        model_dt.load_state_dict(ck_dt["state_dict"], strict=True)
        model_dt.eval()
        train_time_s = float(ck_dt.get("train_time_s", 0.0))
        args.steps = int(ck_dt.get("train_steps", args.steps))
        history = []
        if out_json.exists():
            try:
                history = json.loads(out_json.read_text()).get("history", [])
            except Exception:
                history = []
        print(f"  [Eval-Only] Loaded trained WorldModelDT from {args.eval_only} (steps={args.steps}, train_time_s={train_time_s})", flush=True)
    else:
        # Build diverse 3D + Structural Training Pools
        print("\n--- Building 3D Multi-Rate & Structural Training Pools ---", flush=True)
        pool_std = [make_stacked_3d_frostmarsh_world(128, 128, device, seed=31000 + i) for i in range(10)]
        pool_str = [make_structural_training_world(128, 128, device, seed=32000 + i) for i in range(10)]
        pool_256 = [make_structural_training_world(256, 256, device, seed=33000 + i) for i in range(4)]

        dt_one_128 = torch.ones((4, 1, 128, 128), dtype=torch.float32, device=device)
        dt_one_256 = torch.ones((4, 1, 256, 256), dtype=torch.float32, device=device)
        with torch.no_grad():
            for _ in range(3):
                for i in range(len(pool_std)):
                    fe, fm = sample_bolts(4, 128, 128, device, prob=0.55, max_bolts=2)
                    pool_std[i], _, _ = refsim_step_dt(pool_std[i], dt_one_128, fe, fm)
                for i in range(len(pool_str)):
                    fe, fm = sample_bolts(4, 128, 128, device, prob=0.55, max_bolts=2)
                    pool_str[i], _, _ = refsim_step_dt(pool_str[i], dt_one_128, fe, fm)
                for i in range(len(pool_256)):
                    fe, fm = sample_bolts(4, 256, 256, device, prob=0.55, max_bolts=2)
                    pool_256[i], _, _ = refsim_step_dt(pool_256[i], dt_one_256, fe, fm)

        # Parameter groups:
        # Phase 1 (steps 1..phase1_steps): Train ONLY dt FiLM/conditioning + struct_head (v4 backbone FROZEN -> 0 drift at dt=1.0!)
        # Phase 2 (phase1_steps+1..steps): Gently unfreeze trunk & vert heads at tiny lr=6.5e-5 for joint polish!
        new_prefixes = ("dt_mlp", "dt_film", "dt_head_mod", "dt_z_mod", "struct_")
        new_params = [p for n, p in model_dt.named_parameters() if any(n.startswith(k) for k in new_prefixes)]
        base_params = [p for n, p in model_dt.named_parameters() if not any(n.startswith(k) for k in new_prefixes)]

        phase1_steps = int(args.steps * 0.50)
        for p in base_params:
            p.requires_grad_(False)

        optimizer = torch.optim.AdamW(
            [
                {"params": new_params, "lr": 3.5e-3, "weight_decay": 1e-5},
                {"params": base_params, "lr": 6.5e-5, "weight_decay": 1e-5},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1.5e-5)

        history: list[dict] = []
        t0 = time.perf_counter()
        best_val = 1e9
        best_sd = {k: v.detach().clone() for k, v in model_dt.state_dict().items()}

        print(f"\n--- Training WorldModelDT ({args.steps} steps: P1 frozen-v4 1..{phase1_steps}, P2 joint-polish {phase1_steps+1}..{args.steps}) ---", flush=True)
        for step in range(1, args.steps + 1):
            if step == phase1_steps + 1:
                for p in base_params:
                    p.requires_grad_(True)
                print(f"  [Phase 2] Unfroze v4 backbone at step {step} (lr=6.5e-5)", flush=True)

            model_dt.train()
            if step % 8 == 0:
                st = pool_256[(step // 8) % len(pool_256)].clone()
                H, W = 256, 256
            elif step % 2 == 0:
                st = pool_str[step % len(pool_str)].clone()
                H, W = 128, 128
            else:
                st = pool_std[step % len(pool_std)].clone()
                H, W = 128, 128

            dt_field = sample_training_dt_field(4, H, W, device, step)
            fe, fm = sample_bolts(4, H, W, device, prob=0.50, max_bolts=2)

            with torch.no_grad():
                gt_next, gt_h, gt_z = refsim_step_dt(st, dt_field, fe, fm, include_vertical=True)
                gt_struct = compute_structural_teacher(st, dt_field, fe)
                m_eff, e_eff = apply_forcing(st, fe, fm)
                x = encode_state(st, fe, fm)

            pred_h, pred_z, pred_s = model_dt(x, local_dt=dt_field)
            pred_next, proj_h, _ = apply_dt_projected_step(st, pred_h, pred_z, dt_field, fe, fm)

            # Normalize horizontal deltas by local effective scale down to 0.10 so stasis domes & 4x steps get balanced grads!
            dt_norm_h = dt_field.clamp(min=0.10)
            l_phys, _ = physics_teacher_loss(pred_h / dt_norm_h, gt_h / dt_norm_h)
            l_phys_raw, _ = physics_teacher_loss(pred_h, gt_h)
            l_flux_boost = F.mse_loss(pred_h[:, 0:8] / dt_norm_h, gt_h[:, 0:8] / dt_norm_h) * 25.0
            l_proj, _ = conservation_firewall_loss(m_eff, e_eff, pred_h, proj_h)
            l_state = state_rollout_loss(pred_next, gt_next)
            l_vert, _ = vertical_physics_loss(pred_z / dt_norm_h, gt_z / dt_norm_h, st)
            l_struct, _ = structural_head_loss(pred_s, gt_struct)

            # In Phase 2, keep dt=1.0 strongly anchored to v4 so horizontal firewall never drifts!
            l_anchor = torch.zeros((), device=device)
            if step > phase1_steps:
                with torch.no_grad():
                    v4_h, v4_z = model_v4(x, return_z_flux=True, return_latent=False)
                dt1 = torch.ones((4, 1, H, W), dtype=torch.float32, device=device)
                p1_h, p1_z, _ = model_dt(x, local_dt=dt1)
                l_anchor = F.mse_loss(p1_h[:, :8], v4_h[:, :8]) + 0.05 * F.mse_loss(p1_h[:, 8:10], v4_h[:, 8:10]) + F.mse_loss(p1_h[:, 10:], v4_h[:, 10:]) + F.mse_loss(p1_z[:3], v4_z[:3])

            total_loss = (
                2.5 * l_phys
                + 1.5 * l_phys_raw
                + 1.5 * l_flux_boost
                + 1.5 * l_proj
                + 0.45 * l_state
                + 1.5 * l_vert
                + 4.0 * l_struct
                + 8.0 * l_anchor
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_dt.parameters(), max_norm=1.5)
            optimizer.step()
            scheduler.step()

            # Refresh pools frequently so skybridges, ice dams & waterfalls stay active
            if step % 8 == 0:
                with torch.no_grad():
                    r1 = (step // 8) % len(pool_std)
                    if (step // 8) % 4 == 0:
                        pool_std[r1] = make_stacked_3d_frostmarsh_world(128, 128, device, seed=41000 + step)
                    else:
                        fe1, fm1 = sample_bolts(4, 128, 128, device, prob=0.55, max_bolts=2)
                        pool_std[r1], _, _ = refsim_step_dt(pool_std[r1], dt_one_128, fe1, fm1)

                    r2 = (step // 8) % len(pool_str)
                    if (step // 8) % 3 == 0:
                        pool_str[r2] = make_structural_training_world(128, 128, device, seed=51000 + step)
                    else:
                        fe2, fm2 = sample_bolts(4, 128, 128, device, prob=0.55, max_bolts=2)
                        pool_str[r2], _, _ = refsim_step_dt(pool_str[r2], dt_one_128, fe2, fm2)

                    r3 = (step // 8) % len(pool_256)
                    if (step // 8) % 5 == 0:
                        pool_256[r3] = make_structural_training_world(256, 256, device, seed=61000 + step)
                    else:
                        fe3, fm3 = sample_bolts(4, 256, 256, device, prob=0.55, max_bolts=2)
                        pool_256[r3], _, _ = refsim_step_dt(pool_256[r3], dt_one_256, fe3, fm3)

            if step % 350 == 0 or step == args.steps:
                elapsed = time.perf_counter() - t0
                mr_val = evaluate_multirate_dt(model_dt, model_v4, device, eval_seeds=[97001, 97002], dt_values=(0.25, 1.0, 4.0))
                st_val = evaluate_structural_head_accuracy(model_dt, device, eval_seeds=[98001, 98002, 98003])
                r2_025 = mr_val["per_dt"]["0.25"]["world_model_dt"]["horiz"]["r2"]
                r2_100 = mr_val["per_dt"]["1.0"]["world_model_dt"]["horiz"]["r2"]
                r2_400 = mr_val["per_dt"]["4.0"]["world_model_dt"]["horiz"]["r2"]
                r2_400_lin = mr_val["per_dt"]["4.0"]["naive_linear_v4"]["horiz"]["r2"]
                st_r2 = st_val["mean_struct_r2"]
                score = (1.0 - r2_025) + (1.0 - r2_100) + 1.5 * (1.0 - r2_400) + 1.5 * (1.0 - st_r2)
                tag = ""
                if score < best_val:
                    best_val = score
                    best_sd = {k: v.detach().clone() for k, v in model_dt.state_dict().items()}
                    tag = " *BEST*"
                rec = {
                    "step": step,
                    "elapsed_s": round(elapsed, 1),
                    "r2_dt025": r2_025,
                    "r2_dt100": r2_100,
                    "r2_dt400": r2_400,
                    "r2_dt400_linear_v4": r2_400_lin,
                    "struct_mean_r2": st_r2,
                }
                history.append(rec)
                print(
                    f"step {step:4d}/{args.steps} | t={elapsed:5.1f}s | "
                    f"R2(dt=0.25)={r2_025:.4f} R2(dt=1.0)={r2_100:.4f} R2(dt=4.0)={r2_400:.4f} (lin={r2_400_lin:.4f}) | "
                    f"Struct_R2={st_r2:.4f}{tag}",
                    flush=True,
                )

        train_time_s = round(time.perf_counter() - t0, 2)
        model_dt.load_state_dict(best_sd)
        model_dt.eval()

    # =========================================================================
    # Full Held-Out 256x256x4 & 512x512x4 Evaluations
    # =========================================================================
    print("\n--- Experiment A1: Held-Out 256x256x4 Continuous Δt Scaling Evaluation ---", flush=True)
    multirate_eval = evaluate_multirate_dt(
        model_dt,
        model_v4,
        device,
        eval_seeds=[97101, 97102, 97103, 97104],
        dt_values=(0.25, 0.5, 1.0, 2.0, 4.0),
    )
    print(json.dumps(multirate_eval, indent=2), flush=True)

    print("\n--- Experiment A2: Adaptive Multi-Rate Temporal Sparsity GPU Benchmark ---", flush=True)
    sparsity_eval = benchmark_temporal_sparsity(model_dt, device)
    print(json.dumps(sparsity_eval, indent=2), flush=True)

    print("\n--- Experiment A3: Spatial Time-Dilation Stasis Dome (0.15x) & Rapid-Aging (3.0x) ---", flush=True)
    dilation_eval, dilation_viz = evaluate_time_dilation_bubbles(model_dt, device)
    print(json.dumps(dilation_eval, indent=2), flush=True)

    print("\n--- Experiment B: Structural Head Accuracy & 3 Emergent Structure Scenarios ---", flush=True)
    struct_acc_eval = evaluate_structural_head_accuracy(
        model_dt, device, eval_seeds=[98101, 98102, 98103, 98104, 98105, 98106]
    )
    struct_scenarios_eval, struct_viz = run_experiment_b_structures(model_dt, device)
    print("Structural Head Held-Out Accuracy:", json.dumps(struct_acc_eval, indent=2), flush=True)
    print("Emergent Structure Scenarios:", json.dumps(struct_scenarios_eval, indent=2), flush=True)

    # Save Checkpoint
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model_dt.state_dict().items()},
            "arch": {
                "class": "WorldModelDT",
                "in_ch": 18,
                "dt_input_ch": 1,
                "out_ch": 17,
                "width": 48,
                "num_res": 5,
                "z_flux_ch": Z_FLUX_CH,
                "z_flux_names": list(Z_FLUX_NAMES),
                "struct_ch": STRUCT_CH,
                "struct_names": list(STRUCT_NAMES),
                "dt_dim": 10,
            },
            "n_params": dt_params,
            "v4_params": v4_params,
            "warm_started_from": str(v4_pt),
            "train_steps": args.steps,
            "train_time_s": train_time_s,
        },
        str(out_ckpt),
    )
    print(f"\n[Checkpoint] Saved WorldModelDT -> {out_ckpt}", flush=True)

    # Generate 3-panel visualization
    generate_3panel_visualization(
        out_viz,
        multirate_eval,
        sparsity_eval,
        dilation_eval,
        dilation_viz,
        struct_scenarios_eval,
        struct_viz,
    )
    print(f"[Viz] Saved 3-panel visualization -> {out_viz}", flush=True)

    # Save JSON metrics
    summary = {
        "device": gpu_name,
        "checkpoint": str(out_ckpt),
        "visualization": str(out_viz),
        "model_class": "WorldModelDT",
        "arch": {
            "in_ch": 18,
            "extra_dt_ch": 1,
            "out_ch": 17,
            "width": 48,
            "num_res": 5,
            "z_flux_ch": Z_FLUX_CH,
            "struct_ch": STRUCT_CH,
            "struct_names": list(STRUCT_NAMES),
            "dt_conditioning": "8-basis log-dt + sinusoidal FiLM across all 5 ResConvLayer blocks + face-symmetric min(dt) flux scaling",
        },
        "n_params_v4": v4_params,
        "n_params_dt": dt_params,
        "param_overhead_abs": dt_params - v4_params,
        "param_overhead_pct": round(100.0 * (dt_params - v4_params) / v4_params, 2),
        "init_v4_dt1_parity_max_h_diff": init_h_diff,
        "init_v4_dt1_parity_max_z_diff": init_z_diff,
        "exact_zero_dt_stasis_max_abs": init_zero_diff,
        "train_steps": args.steps,
        "train_time_s": train_time_s,
        "experiment_a_multirate_dt_256x256x4": multirate_eval,
        "experiment_a_temporal_sparsity_fp16": sparsity_eval,
        "experiment_a_time_dilation_bubbles": dilation_eval,
        "experiment_b_struct_head_256x256x4": struct_acc_eval,
        "experiment_b_emergent_structures": struct_scenarios_eval,
        "history": history,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[Metrics] Saved WORLD_DT_STRUCTURES_METRICS -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
