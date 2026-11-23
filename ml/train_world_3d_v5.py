#!/usr/bin/env python3
"""
Frostmarsh Wave 11A: Conservation-Distilled Training & Comprehensive Evaluation of
`WorldModel3D-v5` (`ml/train_world_3d_v5.py`).

Trains `WorldModel3DV5` (~158K params, 48-ch `Z_world_3d`) on Google Colab Cloud GPU across:
  - All 5 isotropic `1m³` biomes (`nordic_viaduct`, `volcanic_caldera`, `sinkhole_caverns`,
    `shattered_glacier_gorge`, `geothermal_mist_basin`)
  - Mixed `Z=16` (`64x64x16`) and `Z=32` (`48x48x32`) random crops + live multi-tick rollouts
  - Continuous `Δt in [0.15, 4.0]` (including uniform & spatially varying Stasis/Haste domes)
  - Strong differentiable Conservation-Firewall Distillation penalty (horizontal donor/wall/heat/rxn
    + vertical donor/sign/2nd-law) targeting held-out `proj_active_pct < 8%` vs `WorldModelDT` baseline!

Evaluates comprehensively on unseen held-out 5-biome worlds (`Z=16` and `Z=32`):
  1. Head-to-head `WorldModel3D-v5` vs `WorldModelDT (Z=4 trained)` baseline on `Z=16` unseen 5-biome worlds:
       - `proj_active_pct` (overall + per-biome), Horizontal MAE/R², Vertical 3D gravity/buoyancy/heat R²,
         Hydrologic Cloud-Condensation & Precipitation R², Continuous `Δt` (`0.15x` stasis & `4.0x` haste) R²
  2. Native `Z=32` unseen 5-biome evaluation of `WorldModel3D-v5`
  3. Exact invariants:
       - `Δt = 0.0` exact stasis (`0.00e+00`)
       - FP32 dense-vs-sparse `16x16x8` 3D brick parity (`max_abs_diff < 1e-5`) at `Z=16` & `Z=32`
       - 30-tick Closed 3D Hydrologic Cycle mass conservation (`rel_err < 1e-8`)
       - 3D Nordic Viaduct structural arch collapse mass & energy conservation (`rel_err < 1e-8`)
  4. FP16 GPU Latency Benchmark (Dense vs Sparse `16x16x8` 3D Bricks):
       - `256x256x16` (`1.05M` voxels)
       - `256x256x32` (`2.10M` voxels)
       - `512x512x16` (`4.19M` voxels)
  5. Outputs:
       - `models/checkpoints/world_model_3d_v5.pt`
       - `models/WORLD_3D_V5_METRICS.json`
       - `models/world_3d_v5_viz.png` (crisp 3-panel dark-themed showcase!)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
    T_AMB,
    T_MAX,
    T_MIN,
    SimState,
    _apply_transport,
    apply_forcing,
    compute_C_and_T,
    compute_total_energy,
    compute_total_mass,
    encode_state,
    get_consts,
    project_and_step,
    project_deltas,
    sample_bolts,
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
from world_model_dt import WorldModelDT  # noqa: E402
from world_model_3d_v5 import (  # noqa: E402
    BIOME_GEOTHERMAL_MIST_BASIN,
    BIOME_NAMES,
    BIOME_NORDIC_VIADUCT,
    BIOME_SHATTERED_GLACIER_GORGE,
    BIOME_SINKHOLE_CAVERNS,
    BIOME_VOLCANIC_CALDERA,
    Z_SPECIES_IDX,
    RefSim3D,
    SparseWorld3DV5Runner,
    WorldModel3DV5,
    count_parameters,
    make_biome_3d_world,
    verify_sparse_v5_parity,
)


# =============================================================================
# Conservation-Firewall & Distillation Losses (Isotropic 3D + Continuous Δt)
# =============================================================================


def conservation_firewall_loss_3d(
    mass_eff: torch.Tensor,
    energy_eff: torch.Tensor,
    raw_deltas: torch.Tensor,
    proj_deltas: torch.Tensor,
    local_dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Differentiable Conservation Projection-Firewall Penalty for isotropic 3D `Z=16/32` grids
    (including strict empty-air zeroing, donor budget, wall/ice blocking, 2nd-law heat,
    reaction gates/caps, and sharp exact `|Y_raw - Y_proj|` hinge penalty!).
    """
    c = get_consts(mass_eff.device)
    L, _, H, W = mass_eff.shape
    dev = mass_eff.device

    flux_e = raw_deltas[:, 0:4]
    flux_s = raw_deltas[:, 4:8]
    heat_e = raw_deltas[:, 8:9]
    heat_s = raw_deltas[:, 9:10]
    rxn = raw_deltas[:, 10:17]

    mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=dev)
    mask_e[:, :, :, -1] = 0.0
    mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=dev)
    mask_s[:, :, -1, :] = 0.0

    wall = ((mass_eff[:, S_ROCK : S_ROCK + 1] >= 500.0) | (mass_eff[:, S_ICE : S_ICE + 1] >= 700.0)).float()
    wall_e_nb = F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0)
    wall_s_nb = F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0)
    open_e = mask_e * (1.0 - wall) * (1.0 - wall_e_nb)
    open_s = mask_s * (1.0 - wall) * (1.0 - wall_s_nb)

    # 1. Donor outflow excess over available local cell mass
    mob_mass_norm = mass_eff[:, c.mobile_idx].clamp(min=0.0) / 1000.0
    fw_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    fn_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    total_out = F.relu(flux_e) + F.relu(-fw_in) + F.relu(flux_s) + F.relu(-fn_in)
    donor_excess = F.relu(total_out - mob_mass_norm)
    l_donor = donor_excess.mean() + 5.0 * F.relu(donor_excess - 3e-4).mean()

    # 2. Wall, boundary & H_CAP leakage
    l_wall = (flux_e.abs() * (1.0 - open_e)).mean() + (flux_s.abs() * (1.0 - open_s)).mean()
    total_in = (F.relu(fw_in) + F.relu(-flux_e) + F.relu(fn_in) + F.relu(-flux_s)).sum(dim=1, keepdim=True)
    headroom_norm = (H_CAP - mass_eff.sum(dim=1, keepdim=True)).clamp(min=0.0) / 1000.0
    l_hcap = F.relu(total_in - headroom_norm).mean()

    # 3. 2nd-law heat conduction & empty-air heat zeroing
    C_cell, T_cK = compute_C_and_T(mass_eff, energy_eff)
    C_e_nb = F.pad(C_cell[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    T_e_nb = F.pad(T_cK[:, :, :, 1:], (0, 1, 0, 0), value=T_AMB)
    dT_e = T_cK - T_e_nb
    q_max_e = torch.abs(dT_e) * (C_cell * C_e_nb) / (4.0 * (C_cell + C_e_nb) + 1e-6)
    q_max_e = torch.where((C_cell > 1e-6) & (C_e_nb > 1e-6), q_max_e, torch.zeros_like(q_max_e)) * mask_e / 1e6

    C_s_nb = F.pad(C_cell[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    T_s_nb = F.pad(T_cK[:, :, 1:, :], (0, 0, 0, 1), value=T_AMB)
    dT_s = T_cK - T_s_nb
    q_max_s = torch.abs(dT_s) * (C_cell * C_s_nb) / (4.0 * (C_cell + C_s_nb) + 1e-6)
    q_max_s = torch.where((C_cell > 1e-6) & (C_s_nb > 1e-6), q_max_s, torch.zeros_like(q_max_s)) * mask_s / 1e6

    hv_e = (
        torch.where(dT_e > 1e-2, F.relu(-heat_e), torch.where(dT_e < -1e-2, F.relu(heat_e), heat_e.abs()))
        + F.relu(heat_e.abs() - q_max_e)
    ) / 10.0
    hv_s = (
        torch.where(dT_s > 1e-2, F.relu(-heat_s), torch.where(dT_s < -1e-2, F.relu(heat_s), heat_s.abs()))
        + F.relu(heat_s.abs() - q_max_s)
    ) / 10.0
    l_heat_b = hv_e.mean() + hv_s.mean() + 4.0 * (F.relu(hv_e - 1e-3).mean() + F.relu(hv_s - 1e-3).mean())

    # 4. Reaction threshold gates & reactant availability (critical for empty-air voxels!)
    with torch.no_grad():
        mass_tr, energy_tr = _apply_transport(
            mass_eff,
            energy_eff,
            proj_deltas[:, 0:4] * 1000.0,
            proj_deltas[:, 4:8] * 1000.0,
            proj_deltas[:, 8:9] * 1e6,
            proj_deltas[:, 9:10] * 1e6,
        )
        C_tr, T_tr = compute_C_and_T(mass_tr, energy_tr)
        t_gate = (((T_tr - c.rxn_thresh) * c.rxn_dir) > 0.0).float() * (C_tr > 1e-6).float()
    l_rxn_gate = (rxn * (1.0 - t_gate)).mean() + 4.0 * F.relu(rxn * (1.0 - t_gate) - 5e-4).mean()
    consumed = torch.einsum("brhw,rs->bshw", rxn * 100.0, c.nu_neg)
    l_rxn_react = F.relu(consumed - mass_tr).mean() / 100.0

    # 5. Direct exact projection gap ||Y_raw - Y_proj.detach()|| (strictly weighted on dt <= 1.25 where 1-step projection is exact!)
    dt_weight = (local_dt <= 1.25).float()
    diff = (raw_deltas - proj_deltas.detach()).abs()
    diff_h = diff[:, 8:10] / 10.0
    p_flux = (diff[:, 0:8] * dt_weight).mean() + 6.0 * (F.relu(diff[:, 0:8] - 1e-3) * dt_weight).mean()
    p_heat = (diff_h * dt_weight).mean() + 6.0 * (F.relu(diff_h - 1e-3) * dt_weight).mean()
    p_rxn = (diff[:, 10:17] * dt_weight).mean() + 6.0 * (F.relu(diff[:, 10:17] - 1e-3) * dt_weight).mean()

    total = (
        18.0 * l_donor
        + 12.0 * l_wall
        + 6.0 * l_hcap
        + 18.0 * l_heat_b
        + 16.0 * (l_rxn_gate + l_rxn_react)
        + 14.0 * p_flux
        + 16.0 * p_heat
        + 12.0 * p_rxn
    )
    return total, {
        "donor": float(l_donor.item()),
        "heat_b": float(l_heat_b.item()),
        "rxn_b": float((l_rxn_gate + l_rxn_react).item()),
    }


def horiz_teacher_loss_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    local_dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Variance-balanced horizontal + reaction + cloud-condensation distillation loss."""
    dt_norm = local_dt.clamp(min=0.25)
    p_n = pred / dt_norm
    t_n = target / dt_norm

    l_flux = F.smooth_l1_loss(p_n[:, 0:8], t_n[:, 0:8], beta=0.005) + 1.0 * F.mse_loss(p_n[:, 0:8], t_n[:, 0:8])
    l_heat = F.smooth_l1_loss(p_n[:, 8:10] / 10.0, t_n[:, 8:10] / 10.0, beta=0.005) + 1.0 * F.mse_loss(
        p_n[:, 8:10] / 10.0, t_n[:, 8:10] / 10.0
    )
    l_rxn = F.smooth_l1_loss(p_n[:, 10:17], t_n[:, 10:17], beta=0.005) + 1.0 * F.mse_loss(p_n[:, 10:17], t_n[:, 10:17])

    # Active-voxel emphasis (critical in 3D where ~65% of voxels are empty air!)
    act_f = target[:, 0:8].abs() > 1e-3
    l_flux_act = F.smooth_l1_loss(p_n[:, 0:8][act_f], t_n[:, 0:8][act_f], beta=0.01) if act_f.any() else pred.new_tensor(0.0)
    act_h = target[:, 8:10].abs() > 1e-2
    l_heat_act = (
        F.smooth_l1_loss(p_n[:, 8:10][act_h] / 10.0, t_n[:, 8:10][act_h] / 10.0, beta=0.01)
        if act_h.any()
        else pred.new_tensor(0.0)
    )

    l_rxn_act = pred.new_tensor(0.0)
    n_seen = 0
    for r in range(7):
        mr = target[:, 10 + r] > 1e-3
        if mr.any():
            # Extra weight on rxn 3 (Condense: hydrologic cloud condensation!) and rxn 0/1 (Freeze/Melt)
            w_r = 2.0 if r == 3 else 1.0
            l_rxn_act = l_rxn_act + w_r * F.smooth_l1_loss(p_n[:, 10 + r][mr], t_n[:, 10 + r][mr], beta=0.015)
            n_seen += 1
    if n_seen > 0:
        l_rxn_act = l_rxn_act / n_seen

    total = (
        22.0 * l_flux + 10.0 * l_flux_act
        + 18.0 * l_heat + 10.0 * l_heat_act
        + 18.0 * l_rxn + 10.0 * l_rxn_act
    )
    return total, {
        "flux": float(l_flux.item()),
        "heat": float(l_heat.item()),
        "rxn": float(l_rxn.item()),
    }


def vertical_teacher_loss_3d(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    state: SimState,
    local_dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Supervised 3D vertical transport distillation + vertical conservation firewall penalty
    across ALL `L - 1` interfaces (gravity water/oil, buoyancy steam/smoke, vertical heat).
    """
    L = state.mass.shape[0]
    if L <= 1:
        z0 = pred_z.new_tensor(0.0)
        return z0, {"z_mse": 0.0, "z_heat": 0.0, "z_donor": 0.0}

    dt_z = torch.minimum(local_dt[:-1], local_dt[1:]).clamp(min=0.25)  # [L-1, 1, H, W]
    ch_scale = torch.tensor([0.040, 0.015, 0.035, 0.018, 0.008], dtype=torch.float32, device=pred_z.device).view(1, 5, 1, 1)

    p_norm = (pred_z / dt_z) / ch_scale
    t_norm = (target_z / dt_z) / ch_scale

    l_mse = F.mse_loss(p_norm, t_norm)
    l_huber = F.smooth_l1_loss(p_norm, t_norm, beta=0.08)

    # Active vertical interface emphasis (waterfalls, plumes, rain!)
    act_z = t_norm.abs() > 0.05
    l_act = F.smooth_l1_loss(p_norm[act_z], t_norm[act_z], beta=0.10) if act_z.any() else pred_z.new_tensor(0.0)

    # Physical sign & donor conservation firewall
    l_sign = F.relu(p_norm[:, 0:2]).pow(2).mean() + F.relu(-p_norm[:, 2:4]).pow(2).mean()
    avail_lo = state.mass[:-1][:, list(Z_SPECIES_IDX)].clamp(min=0.0) / 1000.0
    avail_hi = state.mass[1:][:, list(Z_SPECIES_IDX)].clamp(min=0.0) / 1000.0
    p_m = pred_z[:, 0:4]
    l_z_donor = F.relu(F.relu(p_m) - avail_lo).pow(2).mean() + F.relu(F.relu(-p_m) - avail_hi).pow(2).mean()

    C_cell, T_cK = compute_C_and_T(state.mass, state.energy)
    dT_z = T_cK[:-1] - T_cK[1:]
    valid_c = (C_cell[:-1] > 1e-3) & (C_cell[1:] > 1e-3)
    p_h = p_norm[:, ZF_HEAT : ZF_HEAT + 1]
    l_z_2nd = torch.where(
        valid_c & (dT_z > 1e-2),
        F.relu(-p_h).pow(2),
        torch.where(valid_c & (dT_z < -1e-2), F.relu(p_h).pow(2), p_h.pow(2)),
    ).mean()

    total = 3.0 * l_mse + 1.0 * l_huber + 2.5 * l_act + 0.6 * l_sign + 6.0 * l_z_donor + 0.8 * l_z_2nd
    return total, {
        "z_mse": float(l_mse.item()),
        "z_heat": float(F.mse_loss(p_norm[:, ZF_HEAT], t_norm[:, ZF_HEAT]).item()),
        "z_donor": float(l_z_donor.item()),
    }


# =============================================================================
# Training Buffer & Random `Δt` / Crop Sampler
# =============================================================================


def sample_3d_forcing(L: int, H: int, W: int, device: torch.device, prob: float = 0.40) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample localized 3D thermal/frost spells across active mid-to-surface strata."""
    fe = torch.zeros((L, 1, H, W), dtype=torch.float32, device=device)
    fm = torch.zeros((L, 1, H, W), dtype=torch.float32, device=device)
    if float(torch.rand(1, device=device).item()) > prob:
        return fe, fm

    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    n_bolts = int(torch.randint(1, 3, (1,), device=device).item())
    for _ in range(n_bolts):
        cy = float(torch.randint(4, max(5, H - 4), (1,), device=device).item())
        cx = float(torch.randint(4, max(5, W - 4), (1,), device=device).item())
        cz = int(torch.randint(1, max(2, L - 2), (1,), device=device).item())
        rad = float(torch.empty(1, device=device).uniform_(3.0, 5.5).item())
        k2 = torch.clamp(1.0 - ((yy - cy) ** 2 + (xx - cx) ** 2) / (rad * rad), min=0.0)
        is_fire = bool(torch.rand(1, device=device).item() < 0.55)
        mag = float(torch.empty(1, device=device).uniform_(1.4e8, 3.6e8).item()) * (1.0 if is_fire else -1.0)
        for dz in (-1, 0, 1):
            zz = cz + dz
            if 0 <= zz < L:
                fe[zz, 0] += k2 * mag * (1.0 if dz == 0 else 0.5)
    return fe, fm


def sample_training_dt(L: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Sample diverse continuous `Δt in [0.15, 4.0]` (55% uniform scalar, 30% spatial dome, 15% exact 1.0)."""
    u = float(torch.rand(1, device=device).item())
    if u < 0.20:
        return torch.ones((L, 1, H, W), dtype=torch.float32, device=device)
    if u < 0.72:
        # Log-uniform over [0.15, 4.0] plus anchor points {0.15, 0.5, 1.0, 2.0, 4.0}
        if float(torch.rand(1, device=device).item()) < 0.45:
            choices = (0.15, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
            dt_val = choices[int(torch.randint(0, len(choices), (1,), device=device).item())]
        else:
            log_dt = float(torch.empty(1, device=device).uniform_(math.log(0.15), math.log(4.0)).item())
            dt_val = math.exp(log_dt)
        return torch.full((L, 1, H, W), dt_val, dtype=torch.float32, device=device)
    # Spatially varying Time-Dilation Dome (0.15x stasis or 3.5x-4.0x haste bubble on 1.0x background!)
    yy = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W)
    cy = float(torch.empty(1, device=device).uniform_(0.22, 0.78).item())
    cx = float(torch.empty(1, device=device).uniform_(0.22, 0.78).item())
    rad = float(torch.empty(1, device=device).uniform_(0.18, 0.34).item())
    in_dome = ((yy - cy) ** 2 + (xx - cx) ** 2) < (rad * rad)
    dome_dt = 0.15 if float(torch.rand(1, device=device).item()) < 0.5 else float(torch.empty(1, device=device).uniform_(2.5, 4.0).item())
    dt_hw = torch.where(in_dome, torch.full_like(yy + xx, dome_dt), torch.ones_like(yy + xx))
    return dt_hw.expand(L, 1, H, W).contiguous()


def crop_sim_state(st: SimState, ch: int, cw: int) -> SimState:
    _, _, H, W = st.mass.shape
    if H <= ch and W <= cw:
        return st
    y0 = int(torch.randint(0, H - ch + 1, (1,), device=st.mass.device).item())
    x0 = int(torch.randint(0, W - cw + 1, (1,), device=st.mass.device).item())
    return SimState(
        mass=st.mass[:, :, y0 : y0 + ch, x0 : x0 + cw].contiguous(),
        energy=st.energy[:, :, y0 : y0 + ch, x0 : x0 + cw].contiguous(),
        flow_e=st.flow_e[:, :, y0 : y0 + ch, x0 : x0 + cw].contiguous(),
        flow_s=st.flow_s[:, :, y0 : y0 + ch, x0 : x0 + cw].contiguous(),
        elevation=st.elevation[:, :, y0 : y0 + ch, x0 : x0 + cw].contiguous(),
    )


# =============================================================================
# Metric Helpers & Comprehensive Held-Out Evaluation
# =============================================================================


class R2Accum:
    def __init__(self) -> None:
        self.sse = 0.0
        self.sae = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.n = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        if target.numel() == 0:
            return
        p = pred.detach().to(torch.float64)
        t = target.detach().to(torch.float64)
        err = p - t
        self.sse += float((err * err).sum().item())
        self.sae += float(err.abs().sum().item())
        self.sum_y += float(t.sum().item())
        self.sum_y2 += float((t * t).sum().item())
        self.n += int(t.numel())

    def result(self) -> dict[str, float]:
        if self.n <= 1:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}
        mean_y = self.sum_y / self.n
        sst = self.sum_y2 - self.n * mean_y * mean_y
        r2 = float(1.0 - self.sse / sst) if sst > 1e-12 else 0.0
        return {
            "mae": round(self.sae / self.n, 6),
            "rmse": round(math.sqrt(self.sse / self.n), 6),
            "r2": round(r2, 5),
        }


def _model_infer_compat(
    model: nn.Module,
    x: torch.Tensor,
    local_dt: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified forward returning `(out_horiz [L, 17, H, W], out_z_flux [L-1, 5, H, W])` for v5 or WorldModelDT."""
    if isinstance(model, WorldModel3DV5):
        h, zf, _ = model(x, local_dt=local_dt, return_z_flux=True, return_latent=True)
        return h, zf
    # WorldModelDT returns [L, 5, H, W] (last slice 0) -> slice [:-1]
    h, zf_full = model(x, local_dt=local_dt, return_z_flux=True, return_struct=False, return_latent=False)
    return h, zf_full[:-1]


@torch.no_grad()
def evaluate_model_on_5biomes(
    model: nn.Module,
    L: int,
    H: int,
    W: int,
    device: torch.device,
    base_seed: int = 95000,
    warmup_ticks: int = 4,
    eval_ticks: int = 6,
) -> dict[str, Any]:
    """
    Evaluate a model across ALL 5 unseen held-out biomes at depth `L` (`16` or `32`) on:
      - `proj_active_pct` (cell_active_5e3 matching v3/v4/DT + cell_active_2e3)
      - Horizontal MAE & R² (all, flux, heat, rxn)
      - 3D Vertical R² & MAE (gravity water/oil, buoyancy steam/smoke, vertical heat)
      - Hydrologic Cloud-Condensation (`rxn 3` at `z >= L-3`) & Precipitation (`flux_z_water` at `z >= L-5`) R²
      - Continuous `Δt` (`0.15x`, `0.5x`, `1.0x`, `2.0x`, `4.0x`, spatial dome) R²
    """
    model.eval()
    c = get_consts(device)

    per_biome: dict[str, dict[str, Any]] = {}
    proj_5e3_all, proj_2e3_all = [], []
    donor_all, heat_c_all, rxn_c_all = [], [], []

    acc_h_all, acc_h_flux, acc_h_heat, acc_h_rxn = R2Accum(), R2Accum(), R2Accum(), R2Accum()
    acc_z_w, acc_z_o, acc_z_st, acc_z_sm, acc_z_h = R2Accum(), R2Accum(), R2Accum(), R2Accum(), R2Accum()
    acc_hydro_cond, acc_hydro_precip = R2Accum(), R2Accum()

    for b_id, b_name in enumerate(BIOME_NAMES):
        st = make_biome_3d_world(b_id, L, H, W, device, seed=base_seed + b_id * 37)
        torch.manual_seed(base_seed + 1000 + b_id)
        for _ in range(warmup_ticks):
            fe_w, fm_w = sample_3d_forcing(L, H, W, device, prob=0.45)
            st, _, _ = RefSim3D.step(st, fe_w, fm_w, local_dt=1.0, radiative_cooling=True, structural_collapse=True)

        b_proj_5e3 = []
        b_h = R2Accum()
        b_zw, b_zst, b_zh = R2Accum(), R2Accum(), R2Accum()

        for t in range(eval_ticks):
            torch.manual_seed(base_seed + 2000 + b_id * 100 + t)
            fe, fm = sample_3d_forcing(L, H, W, device, prob=0.50)
            x = encode_state(st, fe, fm)

            pred_h, pred_z = _model_infer_compat(model, x, local_dt=1.0)
            st_next, gt_h, gt_z = RefSim3D.step(st, fe, fm, local_dt=1.0, radiative_cooling=True, structural_collapse=True)

            # Projection firewall activity on raw horizontal output
            m_eff, e_eff = apply_forcing(st, fe, fm)
            proj_h, _ = project_deltas(m_eff, e_eff, pred_h)
            diff = (pred_h - proj_h).abs()
            diff_scaled = diff.clone()
            diff_scaled[:, 8:10] = diff_scaled[:, 8:10] / 10.0

            p5 = float((diff_scaled > 5e-3).any(dim=1).float().mean().item() * 100.0)
            p2 = float((diff_scaled > 2e-3).any(dim=1).float().mean().item() * 100.0)
            b_proj_5e3.append(p5)
            proj_5e3_all.append(p5)
            proj_2e3_all.append(p2)

            mob_m = m_eff[:, c.mobile_idx].clamp(min=0.0) / 1000.0
            flux_e, flux_s = pred_h[:, 0:4], pred_h[:, 4:8]
            fw_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
            fn_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
            tot_out = F.relu(flux_e) + F.relu(-fw_in) + F.relu(flux_s) + F.relu(-fn_in)
            donor_all.append(float(((tot_out - mob_m) > 2e-3).any(dim=1).float().mean().item() * 100.0))
            heat_c_all.append(float((diff_scaled[:, 8:10] > 5e-3).any(dim=1).float().mean().item() * 100.0))
            rxn_c_all.append(float((diff[:, 10:17] > 2e-3).any(dim=1).float().mean().item() * 100.0))

            # Horizontal accuracy
            acc_h_all.update(pred_h, gt_h)
            acc_h_flux.update(pred_h[:, 0:8], gt_h[:, 0:8])
            acc_h_heat.update(pred_h[:, 8:10] / 10.0, gt_h[:, 8:10] / 10.0)
            acc_h_rxn.update(pred_h[:, 10:17], gt_h[:, 10:17])
            b_h.update(pred_h, gt_h)

            # 3D vertical accuracy across ALL L-1 interfaces
            acc_z_w.update(pred_z[:, ZF_WATER], gt_z[:, ZF_WATER])
            acc_z_o.update(pred_z[:, ZF_OIL], gt_z[:, ZF_OIL])
            acc_z_st.update(pred_z[:, ZF_STEAM], gt_z[:, ZF_STEAM])
            acc_z_sm.update(pred_z[:, ZF_SMOKE], gt_z[:, ZF_SMOKE])
            acc_z_h.update(pred_z[:, ZF_HEAT] / 10.0, gt_z[:, ZF_HEAT] / 10.0)
            b_zw.update(pred_z[:, ZF_WATER], gt_z[:, ZF_WATER])
            b_zst.update(pred_z[:, ZF_STEAM], gt_z[:, ZF_STEAM])
            b_zh.update(pred_z[:, ZF_HEAT] / 10.0, gt_z[:, ZF_HEAT] / 10.0)

            # Hydrologic cloud-condensation (rxn 3 at z >= L - 3) & precipitation (flux_z_water at z >= L - 5)
            z_cloud = max(0, L - 3)
            z_precip = max(0, L - 5)
            acc_hydro_cond.update(pred_h[z_cloud:, 13], gt_h[z_cloud:, 13])
            acc_hydro_precip.update(pred_z[z_precip:, ZF_WATER], gt_z[z_precip:, ZF_WATER])

            st = st_next

        per_biome[b_name] = {
            "proj_active_pct": round(float(np.mean(b_proj_5e3)), 3),
            "horiz_r2": b_h.result()["r2"],
            "horiz_mae": b_h.result()["mae"],
            "vert_water_r2": b_zw.result()["r2"],
            "vert_steam_r2": b_zst.result()["r2"],
            "vert_heat_r2": b_zh.result()["r2"],
        }

    # Continuous Δt evaluation across all 5 biomes for dt in {0.15, 0.5, 1.0, 2.0, 4.0} + spatial dome
    dt_eval: dict[str, dict[str, Any]] = {}
    for dt_val in (0.15, 0.5, 1.0, 2.0, 4.0):
        acc_dh, acc_dz = R2Accum(), R2Accum()
        for b_id in range(len(BIOME_NAMES)):
            st = make_biome_3d_world(b_id, L, H, W, device, seed=base_seed + 5000 + b_id * 19)
            for _ in range(3):
                st, _, _ = RefSim3D.step(st, local_dt=1.0, radiative_cooling=True, structural_collapse=True)
            x = encode_state(st)
            ph, pz = _model_infer_compat(model, x, local_dt=dt_val)
            _, gh, gz = RefSim3D.step(st, local_dt=dt_val, radiative_cooling=True, structural_collapse=True)
            acc_dh.update(ph, gh)
            acc_dz.update(pz, gz)
        r_h = acc_dh.result()
        r_z = acc_dz.result()
        dt_eval[f"{dt_val:.2f}"] = {
            "horiz_mae": r_h["mae"],
            "horiz_r2": r_h["r2"],
            "vert_mae": r_z["mae"],
            "vert_r2": r_z["r2"],
            "mean_r2": round(0.5 * (r_h["r2"] + r_z["r2"]), 5),
        }

    # Spatial Time-Dilation Dome (0.15x stasis center + 4.0x haste ring + 1.0x outer)
    yy = torch.linspace(0.0, 1.0, H, device=device).view(H, 1)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, W)
    r2_d = (yy - 0.5) ** 2 + (xx - 0.5) ** 2
    dt_dome = torch.where(r2_d < 0.04, torch.full((H, W), 0.15, device=device), torch.where(r2_d < 0.11, torch.full((H, W), 4.0, device=device), torch.ones((H, W), device=device)))
    acc_sdh, acc_sdz = R2Accum(), R2Accum()
    for b_id in (BIOME_NORDIC_VIADUCT, BIOME_VOLCANIC_CALDERA, BIOME_GEOTHERMAL_MIST_BASIN):
        st = make_biome_3d_world(b_id, L, H, W, device, seed=base_seed + 7000 + b_id)
        for _ in range(3):
            st, _, _ = RefSim3D.step(st, local_dt=1.0, radiative_cooling=True, structural_collapse=True)
        x = encode_state(st)
        ph, pz = _model_infer_compat(model, x, local_dt=dt_dome)
        _, gh, gz = RefSim3D.step(st, local_dt=dt_dome, radiative_cooling=True, structural_collapse=True)
        acc_sdh.update(ph, gh)
        acc_sdz.update(pz, gz)
    dt_eval["spatial_dome_015_400"] = {
        "horiz_mae": acc_sdh.result()["mae"],
        "horiz_r2": acc_sdh.result()["r2"],
        "vert_mae": acc_sdz.result()["mae"],
        "vert_r2": acc_sdz.result()["r2"],
        "mean_r2": round(0.5 * (acc_sdh.result()["r2"] + acc_sdz.result()["r2"]), 5),
    }

    r_zw = acc_z_w.result()
    r_zo = acc_z_o.result()
    r_zst = acc_z_st.result()
    r_zsm = acc_z_sm.result()
    r_zh = acc_z_h.result()
    mean_vert_r2 = float(np.mean([r_zw["r2"], r_zo["r2"], r_zst["r2"], r_zsm["r2"], r_zh["r2"]]))
    r_hc = acc_hydro_cond.result()
    r_hp = acc_hydro_precip.result()

    return {
        "grid": f"{H}x{W}x{L}",
        "proj_active_pct": round(float(np.mean(proj_5e3_all)), 3),
        "proj_active_2e3_pct": round(float(np.mean(proj_2e3_all)), 3),
        "donor_clamp_pct": round(float(np.mean(donor_all)), 3),
        "heat_clamp_pct": round(float(np.mean(heat_c_all)), 3),
        "rxn_clamp_pct": round(float(np.mean(rxn_c_all)), 3),
        "horizontal": {
            "all": acc_h_all.result(),
            "flux": acc_h_flux.result(),
            "heat": acc_h_heat.result(),
            "rxn": acc_h_rxn.result(),
        },
        "vertical_3d": {
            "gravity_water": r_zw,
            "gravity_oil": r_zo,
            "buoyancy_steam": r_zst,
            "buoyancy_smoke": r_zsm,
            "vertical_heat": r_zh,
            "mean_vertical_r2": round(mean_vert_r2, 5),
        },
        "hydrologic_cycle": {
            "cloud_condensation": r_hc,
            "cloud_precipitation": r_hp,
            "mean_hydro_r2": round(0.5 * (r_hc["r2"] + r_hp["r2"]), 5),
        },
        "continuous_dt": dt_eval,
        "per_biome": per_biome,
    }


@torch.no_grad()
def verify_closed_hydrologic_and_collapse(
    model: WorldModel3DV5,
    device: torch.device,
) -> dict[str, Any]:
    """
    Verify:
      1. Exact `Δt = 0.0` stasis (`max_abs == 0.0`)
      2. 30-tick Closed 3D Hydrologic Cycle (Geothermal Mist Basin `Z=16` & `Z=32`):
         steam rises -> condenses at `z >= L-3` -> precipitates back down, `rel_mass_err < 1e-8`!
      3. 3D Nordic Viaduct Structural Arch Collapse under pier fire:
         burning piers drop overhanging timber deck conservatively (`rel_mass_err < 1e-8`, `rel_energy_err < 1e-8`)!
    """
    model.eval()
    # 1. Exact Δt = 0 stasis
    x_probe = encode_state(make_biome_3d_world(BIOME_GEOTHERMAL_MIST_BASIN, 16, 48, 48, device, seed=888))
    h0, z0, _ = model(x_probe, local_dt=0.0, return_z_flux=True, return_latent=True)
    zero_dt_max = float(max(h0.abs().max().item(), z0.abs().max().item()))

    # 2. 30-tick Closed 3D Hydrologic Cycle on Geothermal Mist Basin (Z=16 & Z=32) using WorldModel3D-v5 + firewall!
    hydro_results: dict[str, Any] = {}
    for L in (16, 32):
        st = make_biome_3d_world(BIOME_GEOTHERMAL_MIST_BASIN, L, 64, 64, device, seed=9900 + L)
        m0 = float(compute_total_mass(st).sum().item())
        max_rel_m_err = 0.0
        total_condensed_cloud = 0.0
        total_precip_down = 0.0
        z_cloud = L - 3

        for _ in range(30):
            x = encode_state(st)
            pred_h, pred_z, _ = model(x, local_dt=1.0, return_z_flux=True, return_latent=True)
            st, proj_h, _ = project_and_step(st, pred_h)
            st, proj_z = RefSim3D.project_and_apply_vertical_flux(st, pred_z)
            st = RefSim3D.apply_radiative_cooling_upper_atm(st, dt_scale=1.0)

            total_condensed_cloud += float(proj_h[z_cloud:, 13].sum().item() * 100.0)
            total_precip_down += float(F.relu(-proj_z[z_cloud - 2 :, ZF_WATER]).sum().item() * 1000.0)

            mt = float(compute_total_mass(st).sum().item())
            max_rel_m_err = max(max_rel_m_err, abs(mt - m0) / max(m0, 1.0))

        hydro_results[f"Z{L}"] = {
            "ticks": 30,
            "total_cloud_condensed_mass": round(total_condensed_cloud, 1),
            "total_rain_precipitated_mass": round(total_precip_down, 1),
            "max_rel_mass_err": max_rel_m_err,
            "mass_conserved_lt_1e8": bool(max_rel_m_err < 1e-8),
        }

    # 3. 3D Nordic Viaduct Structural Arch Collapse under intense pier fire (closed mass & energy!)
    st_v = make_biome_3d_world(BIOME_NORDIC_VIADUCT, 16, 64, 64, device, seed=7701)
    # Remove central stone piers at y~0.46 to simulate demolished support under midspan timber deck!
    st_v.mass[1:9, S_ROCK, 24:36, 24:40] = 0.0
    m_v0 = float(compute_total_mass(st_v).sum().item())
    e_v0 = float(compute_total_energy(st_v).sum().item())
    collapsed_mass_sum = 0.0
    max_rel_m_v = 0.0
    max_rel_e_v = 0.0
    for _ in range(16):
        x = encode_state(st_v)
        pred_h, pred_z, _ = model(x, local_dt=1.0, return_z_flux=True, return_latent=True)
        st_v, _, _ = project_and_step(st_v, pred_h)
        st_v, _ = RefSim3D.project_and_apply_vertical_flux(st_v, pred_z)
        st_v, col_m = RefSim3D.apply_structural_collapse_3d(st_v, dt_scale=1.0, max_span=3)
        collapsed_mass_sum += float(col_m.sum().item())
        mt = float(compute_total_mass(st_v).sum().item())
        et = float(compute_total_energy(st_v).sum().item())
        max_rel_m_v = max(max_rel_m_v, abs(mt - m_v0) / max(m_v0, 1.0))
        max_rel_e_v = max(max_rel_e_v, abs(et - e_v0) / max(abs(e_v0), 1.0))

    return {
        "exact_zero_dt_stasis_max_abs": zero_dt_max,
        "closed_hydrologic_30t": hydro_results,
        "structural_collapse_16t": {
            "collapsed_deck_mass": round(collapsed_mass_sum, 1),
            "max_rel_mass_err": max_rel_m_v,
            "max_rel_energy_err": max_rel_e_v,
            "conserved_lt_1e8": bool(max_rel_m_v < 1e-8 and max_rel_e_v < 1e-8),
        },
    }


# =============================================================================
# FP16 Dense vs Sparse `16x16x8` 3D Brick Latency Benchmark (1.05M / 2.10M / 4.19M)
# =============================================================================


@torch.inference_mode()
def benchmark_fp16_brick_scaling(
    model_v5: WorldModel3DV5,
    baseline_dt: WorldModelDT | None,
    device: torch.device,
    iters: int = 25,
    warmup: int = 8,
) -> dict[str, Any]:
    """
    Benchmark FP16 GPU latency of `WorldModel3DV5` Dense vs `SparseWorld3DV5Runner` (`16x16x8` 3D bricks)
    at 1.05M (`256x256x16`), 2.10M (`256x256x32`), and 4.19M (`512x512x16`) voxels!
    """
    assert device.type == "cuda", "Latency benchmark requires CUDA GPU"
    wm_fp16 = copy.deepcopy(model_v5).to(device=device, dtype=torch.float16).eval()
    dt_fp16 = copy.deepcopy(baseline_dt).to(device=device, dtype=torch.float16).eval() if baseline_dt is not None else None

    configs = (
        ("256x256x16_1.05M", 16, 256, 256),
        ("256x256x32_2.10M", 32, 256, 256),
        ("512x512x16_4.19M", 16, 512, 512),
    )

    def _time_fn(fn) -> dict[str, float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        times = []
        for _ in range(iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize(device)
            times.append(float(e0.elapsed_time(e1)))
        arr = np.array(times, dtype=np.float64)
        p50 = float(np.percentile(arr, 50))
        return {
            "p50_ms": round(p50, 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "mean_ms": round(float(arr.mean()), 3),
            "hz": round(1000.0 / max(p50, 1e-3), 1),
        }

    results: dict[str, Any] = {}
    for label, L, H, W in configs:
        voxels = L * H * W
        nz, ny, nx = L // 8, H // 16, W // 16
        total_bricks = nz * ny * nx

        st = make_biome_3d_world(BIOME_GEOTHERMAL_MIST_BASIN, L, H, W, device, seed=1234 + L + H)
        x = encode_state(st).to(dtype=torch.float16)
        dt_map = torch.ones((H, W), dtype=torch.float16, device=device)

        # Dense WorldModel3DV5 FP16
        dense_stats = _time_fn(lambda: wm_fp16(x, local_dt=dt_map, return_z_flux=True, return_latent=True))

        # Optional Dense WorldModelDT baseline FP16
        dt_dense_stats = None
        if dt_fp16 is not None:
            try:
                dt_dense_stats = _time_fn(
                    lambda: dt_fp16(x, local_dt=dt_map, return_z_flux=True, return_struct=False, return_latent=True)
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                dt_dense_stats = {"error": "OOM"}

        # Sparse `16x16x8` 3D bricks at 10% and 20% active brick density!
        runner = SparseWorld3DV5Runner(
            model_v5, device=device, use_fp16=True, brick_xy=16, brick_z=8, sparse_threshold=0.55, bucket_size=32
        )
        # Prime persistent caches once
        runner.step(x, local_dt=dt_map)
        torch.cuda.synchronize(device)

        sparse_by_pct: dict[str, Any] = {}
        for pct in (10, 20):
            k_target = max(1, int(round(total_bricks * (pct / 100.0))))
            mask_flat = torch.zeros((total_bricks,), dtype=torch.bool, device=device)
            # Pick spatially coherent active cluster (surface + cloud plume bricks!)
            mask_flat[:k_target] = True
            dmask = mask_flat.view(nz, ny, nx)
            sp_stats = _time_fn(lambda dm=dmask: runner.step(x, local_dt=dt_map, dirty_mask=dm))
            sp_stats["active_bricks"] = k_target
            sp_stats["total_bricks"] = total_bricks
            sp_stats["actual_active_pct"] = round(100.0 * k_target / total_bricks, 2)
            sp_stats["speedup_vs_dense_x"] = round(dense_stats["p50_ms"] / max(sp_stats["p50_ms"], 1e-3), 2)
            sparse_by_pct[f"sparse_{pct}pct"] = sp_stats

        results[label] = {
            "L": L,
            "H": H,
            "W": W,
            "voxels": voxels,
            "total_bricks_16x16x8": total_bricks,
            "dense_v5_fp16": dense_stats,
            "dense_dt_baseline_fp16": dt_dense_stats,
            "sparse_16x16x8_fp16": sparse_by_pct,
        }
        del st, x, runner
        torch.cuda.empty_cache()

    return results


# =============================================================================
# Crisp 3-Panel Dark-Themed Visualization (`models/world_3d_v5_viz.png`)
# =============================================================================


@torch.no_grad()
def render_v5_showcase(
    model_v5: WorldModel3DV5,
    metrics: dict[str, Any],
    device: torch.device,
    out_png: Path,
) -> None:
    """
    Generate a crisp 3-panel dark-themed figure `models/world_3d_v5_viz.png`:
      Panel 1: 5-Biome `Z=16/32` vertical XZ cross-sections + closed 3D hydrologic cloud/rain cycle
      Panel 2: `WorldModel3D-v5` vs `WorldModelDT (Z=4)` `proj_active_pct` & 3D/Hydrologic/Δt R²
      Panel 3: `1.05M / 2.10M / 4.19M` FP16 Dense vs Sparse `16x16x8` 3D Brick Latency Scaling
    """
    bg = "#0d1117"
    card = "#161b22"
    fg = "#e6edf3"
    muted = "#8b949e"
    grid_c = "#30363d"
    c_cyan = "#39c5cf"
    c_blue = "#58a6ff"
    c_orange = "#f0883e"
    c_green = "#3fb950"
    c_purple = "#bc8cff"
    c_red = "#f85149"
    c_gold = "#e3b341"

    fig = plt.figure(figsize=(20, 7.4), dpi=150, facecolor=bg)
    outer = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 1.0], wspace=0.22, left=0.04, right=0.98, top=0.86, bottom=0.11)

    fig.suptitle(
        "Frostmarsh Wave 11A: True Isotropic Z=16/32 WorldModel3D-v5 — 5 Biomes, Closed 3D Hydrologic Weather & 4.19M-Voxel Sparse Bricks",
        color=fg,
        fontsize=14.5,
        fontweight="bold",
        y=0.955,
    )

    # --- Panel 1: 5-Biome XZ Cross-Sections (2x3 mini-grid: 5 biomes + hydrologic rain flux) ---
    p1 = outer[0].subgridspec(3, 2, hspace=0.34, wspace=0.10)
    titles_p1 = [
        ("0: Nordic Viaduct (Z=16)", BIOME_NORDIC_VIADUCT, 16),
        ("1: Volcanic Caldera (Z=16)", BIOME_VOLCANIC_CALDERA, 16),
        ("2: Sinkhole Caverns (Z=16)", BIOME_SINKHOLE_CAVERNS, 16),
        ("3: Shattered Glacier (Z=16)", BIOME_SHATTERED_GLACIER_GORGE, 16),
        ("4: Geothermal Mist (Z=32)", BIOME_GEOTHERMAL_MIST_BASIN, 32),
    ]

    def _colorize_xz(st: SimState, y_slice: int) -> np.ndarray:
        m = st.mass[:, :, y_slice, :].cpu().numpy()  # [L, 10, W]
        _, T = compute_C_and_T(st.mass, st.energy)
        T_s = T[:, 0, y_slice, :].cpu().numpy()
        L, _, W = m.shape
        rgb = np.zeros((L, W, 3), dtype=np.float32)
        # Sky gradient background
        z_grad = np.linspace(0.08, 0.17, L)[:, None]
        rgb[:, :, 0] = 0.05 + 0.03 * z_grad
        rgb[:, :, 1] = 0.08 + 0.07 * z_grad
        rgb[:, :, 2] = 0.14 + 0.14 * z_grad

        # Layer species compositing
        rock = np.clip(m[:, S_ROCK] / 900.0, 0, 1)
        soil = np.clip(m[:, S_SOIL] / 800.0, 0, 1)
        wood = np.clip(m[:, S_WOOD] / 700.0, 0, 1)
        plant = np.clip(m[:, S_PLANT] / 400.0, 0, 1)
        ice = np.clip(m[:, S_ICE] / 800.0, 0, 1)
        water = np.clip(m[:, S_WATER] / 800.0, 0, 1)
        oil = np.clip(m[:, S_OIL] / 650.0, 0, 1)
        steam = np.clip(m[:, S_STEAM] / 450.0, 0, 1)
        smoke = np.clip(m[:, S_SMOKE] / 350.0, 0, 1)
        ash = np.clip(m[:, S_ASH] / 350.0, 0, 1)
        hot = np.clip((T_s - 42000.0) / 45000.0, 0, 1)

        def _blend(a: np.ndarray, col: tuple[float, float, float]) -> None:
            for ci in range(3):
                rgb[:, :, ci] = rgb[:, :, ci] * (1.0 - a) + col[ci] * a

        _blend(rock, (0.34, 0.37, 0.43))
        _blend(soil, (0.48, 0.34, 0.22))
        _blend(ash, (0.42, 0.40, 0.41))
        _blend(wood, (0.66, 0.43, 0.21))
        _blend(plant, (0.22, 0.64, 0.34))
        _blend(ice, (0.58, 0.87, 0.99))
        _blend(water, (0.14, 0.56, 0.96))
        _blend(oil, (0.94, 0.36, 0.12))
        _blend(smoke * 0.72, (0.35, 0.36, 0.40))
        _blend(steam * 0.80, (0.85, 0.92, 0.99))
        _blend(hot * 0.65, (1.00, 0.45, 0.10))
        return np.clip(rgb, 0.0, 1.0)

    hydro_st_32 = None
    for idx, (t_str, b_id, L_b) in enumerate(titles_p1):
        ax = fig.add_subplot(p1[idx // 2, idx % 2], facecolor=card)
        st_b = make_biome_3d_world(b_id, L_b, 64, 64, device, seed=5550 + b_id)
        for _ in range(4):
            st_b, _, _ = RefSim3D.step(st_b, local_dt=1.0, radiative_cooling=True, structural_collapse=True)
        if b_id == BIOME_GEOTHERMAL_MIST_BASIN:
            hydro_st_32 = st_b
        y_sl = 30 if b_id == BIOME_NORDIC_VIADUCT else 32
        img = _colorize_xz(st_b, y_sl)
        ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(t_str, color=fg, fontsize=8.8, fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([0, L_b - 1])
        ax.set_yticklabels(["0m", f"{L_b}m"], color=muted, fontsize=7)
        for sp in ax.spines.values():
            sp.set_color(grid_c)

    # 6th mini-slot: WorldModel3D-v5 predicted 3D hydrologic vertical flux (Z=32 Mist Basin: +steam rise / -rain fall!)
    ax_hy = fig.add_subplot(p1[2, 1], facecolor=card)
    assert hydro_st_32 is not None
    x_hy = encode_state(hydro_st_32)
    _, pred_z_hy, _ = model_v5(x_hy, local_dt=1.0, return_z_flux=True, return_latent=True)
    # Net vertical H2O transport: steam rising (+) minus rain falling (-)
    net_h2o_z = (pred_z_hy[:, ZF_STEAM, 32, :] + pred_z_hy[:, ZF_WATER, 32, :]).cpu().numpy()
    vmax = max(0.05, float(np.percentile(np.abs(net_h2o_z), 99)))
    im = ax_hy.imshow(net_h2o_z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_hy.set_title("v5 Pred: +Steam Rise / -Rain Fall (Z=32)", color=c_cyan, fontsize=8.5, fontweight="bold", pad=3)
    ax_hy.set_xticks([])
    ax_hy.set_yticks([0, 30])
    ax_hy.set_yticklabels(["0m", "32m"], color=muted, fontsize=7)
    for sp in ax_hy.spines.values():
        sp.set_color(c_cyan)

    # --- Panel 2: WorldModel3D-v5 vs WorldModelDT (Z=4 Baseline) on Unseen 5-Biome Z=16 & Z=32 ---
    p2 = outer[1].subgridspec(2, 1, height_ratios=[0.78, 1.22], hspace=0.38)
    ax_proj = fig.add_subplot(p2[0], facecolor=card)
    ax_r2 = fig.add_subplot(p2[1], facecolor=card)

    e_v5_16 = metrics["eval_z16_5biomes"]["world_model_3d_v5"]
    e_dt_16 = metrics["eval_z16_5biomes"]["world_model_dt_z4_baseline"]
    e_v5_32 = metrics["eval_z32_5biomes"]["world_model_3d_v5"]

    labels_p = ["WorldModelDT\n(Z=4 Baseline)", "WorldModel3D-v5\n(Z=16 5-Biome)", "WorldModel3D-v5\n(Z=32 5-Biome)"]
    vals_p = [e_dt_16["proj_active_pct"], e_v5_16["proj_active_pct"], e_v5_32["proj_active_pct"]]
    cols_p = [c_red, c_green, c_cyan]
    bars = ax_proj.bar(labels_p, vals_p, color=cols_p, width=0.52, edgecolor=fg, linewidth=0.5)
    ax_proj.axhline(8.0, color=c_gold, linestyle="--", linewidth=1.3, label="Target < 8.0%")
    for b, v in zip(bars, vals_p):
        ax_proj.text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.2f}%", ha="center", va="bottom", color=fg, fontsize=9, fontweight="bold")
    ax_proj.set_ylabel("Firewall Clamped %", color=fg, fontsize=9)
    ax_proj.set_title("Conservation-Firewall Activity (Lower = Better)", color=fg, fontsize=10.5, fontweight="bold", pad=6)
    ax_proj.set_ylim(0, max(max(vals_p) * 1.24, 14.0))
    ax_proj.tick_params(colors=muted, labelsize=8.2)
    ax_proj.grid(axis="y", color=grid_c, alpha=0.5)
    ax_proj.legend(facecolor=card, edgecolor=grid_c, labelcolor=c_gold, fontsize=8, loc="upper right")
    for sp in ax_proj.spines.values():
        sp.set_color(grid_c)

    r2_cats = ["Horiz\nFlux/Rxn", "3D Vert\nMean", "Hydro\nCloud/Rain", "Δt=0.15×\nStasis", "Δt=4.0×\nHaste"]
    v5_16_r2 = [
        e_v5_16["horizontal"]["all"]["r2"],
        e_v5_16["vertical_3d"]["mean_vertical_r2"],
        e_v5_16["hydrologic_cycle"]["mean_hydro_r2"],
        e_v5_16["continuous_dt"]["0.15"]["mean_r2"],
        e_v5_16["continuous_dt"]["4.00"]["mean_r2"],
    ]
    v5_32_r2 = [
        e_v5_32["horizontal"]["all"]["r2"],
        e_v5_32["vertical_3d"]["mean_vertical_r2"],
        e_v5_32["hydrologic_cycle"]["mean_hydro_r2"],
        e_v5_32["continuous_dt"]["0.15"]["mean_r2"],
        e_v5_32["continuous_dt"]["4.00"]["mean_r2"],
    ]
    dt_16_r2 = [
        max(-0.25, e_dt_16["horizontal"]["all"]["r2"]),
        max(-0.25, e_dt_16["vertical_3d"]["mean_vertical_r2"]),
        max(-0.25, e_dt_16["hydrologic_cycle"]["mean_hydro_r2"]),
        max(-0.25, e_dt_16["continuous_dt"]["0.15"]["mean_r2"]),
        max(-0.25, e_dt_16["continuous_dt"]["4.00"]["mean_r2"]),
    ]

    x_idx = np.arange(len(r2_cats))
    bw = 0.26
    ax_r2.bar(x_idx - bw, dt_16_r2, width=bw, color=c_red, alpha=0.85, label="WorldModelDT (Z=4)")
    b2 = ax_r2.bar(x_idx, v5_16_r2, width=bw, color=c_green, label="WorldModel3D-v5 (Z=16)")
    b3 = ax_r2.bar(x_idx + bw, v5_32_r2, width=bw, color=c_cyan, label="WorldModel3D-v5 (Z=32)")
    for b, v in zip(b2, v5_16_r2):
        ax_r2.text(b.get_x() + b.get_width() / 2, max(0.03, v) + 0.025, f"{v:.2f}", ha="center", va="bottom", color=c_green, fontsize=7.5, fontweight="bold")
    for b, v in zip(b3, v5_32_r2):
        ax_r2.text(b.get_x() + b.get_width() / 2, max(0.03, v) + 0.025, f"{v:.2f}", ha="center", va="bottom", color=c_cyan, fontsize=7.5, fontweight="bold")
    ax_r2.axhline(0.0, color=muted, linewidth=0.8)
    ax_r2.set_xticks(x_idx)
    ax_r2.set_xticklabels(r2_cats, color=fg, fontsize=8.2)
    ax_r2.set_ylabel("Unseen 5-Biome R²", color=fg, fontsize=9)
    ax_r2.set_ylim(-0.28, 1.15)
    ax_r2.set_title("True 3D Physics, Hydrologic & Δt Accuracy (R²)", color=fg, fontsize=10.5, fontweight="bold", pad=6)
    ax_r2.tick_params(axis="y", colors=muted, labelsize=8)
    ax_r2.grid(axis="y", color=grid_c, alpha=0.5)
    ax_r2.legend(facecolor=card, edgecolor=grid_c, labelcolor=fg, fontsize=7.8, loc="lower right")
    for sp in ax_r2.spines.values():
        sp.set_color(grid_c)

    # --- Panel 3: 1.05M / 2.10M / 4.19M FP16 Dense vs Sparse 16x16x8 Brick Latency ---
    ax_lat = fig.add_subplot(outer[2], facecolor=card)
    lat = metrics["latency_fp16_brick_scaling"]
    keys_lat = ("256x256x16_1.05M", "256x256x32_2.10M", "512x512x16_4.19M")
    x_lbls = ["256×256×16\n(1.05M vox)", "256×256×32\n(2.10M vox)", "512×512×16\n(4.19M vox)"]
    dense_ms = [lat[k]["dense_v5_fp16"]["p50_ms"] for k in keys_lat]
    sp20_ms = [lat[k]["sparse_16x16x8_fp16"]["sparse_20pct"]["p50_ms"] for k in keys_lat]
    sp10_ms = [lat[k]["sparse_16x16x8_fp16"]["sparse_10pct"]["p50_ms"] for k in keys_lat]
    sp10_x = [lat[k]["sparse_16x16x8_fp16"]["sparse_10pct"]["speedup_vs_dense_x"] for k in keys_lat]

    xi = np.arange(len(keys_lat))
    w3 = 0.25
    bd = ax_lat.bar(xi - w3, dense_ms, width=w3, color=c_orange, label="Dense v5 FP16")
    bs2 = ax_lat.bar(xi, sp20_ms, width=w3, color=c_blue, label="Sparse 16×16×8 (20%)")
    bs1 = ax_lat.bar(xi + w3, sp10_ms, width=w3, color=c_green, label="Sparse 16×16×8 (10%)")

    for b, v in zip(bd, dense_ms):
        ax_lat.text(b.get_x() + b.get_width() / 2, v + max(dense_ms) * 0.015, f"{v:.1f}ms", ha="center", va="bottom", color=c_orange, fontsize=8, fontweight="bold")
    for b, v in zip(bs2, sp20_ms):
        ax_lat.text(b.get_x() + b.get_width() / 2, v + max(dense_ms) * 0.015, f"{v:.1f}ms", ha="center", va="bottom", color=c_blue, fontsize=8, fontweight="bold")
    for b, v, spx in zip(bs1, sp10_ms, sp10_x):
        ax_lat.text(b.get_x() + b.get_width() / 2, v + max(dense_ms) * 0.015, f"{v:.1f}ms\n({spx:.1f}×)", ha="center", va="bottom", color=c_green, fontsize=8, fontweight="bold")

    ax_lat.axhline(100.0, color=c_purple, linestyle=":", linewidth=1.2, label="10 Hz Budget (100 ms)")
    ax_lat.set_xticks(xi)
    ax_lat.set_xticklabels(x_lbls, color=fg, fontsize=8.8)
    ax_lat.set_ylabel("GPU FP16 Latency p50 (ms)", color=fg, fontsize=9.5)
    dev_name = metrics.get("device", "GPU").replace("NVIDIA ", "")
    ax_lat.set_title(f"3D Sparse 16×16×8 Brick Scaling ({dev_name})", color=fg, fontsize=10.5, fontweight="bold", pad=8)
    ax_lat.set_ylim(0, max(dense_ms) * 1.24)
    ax_lat.tick_params(axis="y", colors=muted, labelsize=8.5)
    ax_lat.grid(axis="y", color=grid_c, alpha=0.5)
    ax_lat.legend(facecolor=card, edgecolor=grid_c, labelcolor=fg, fontsize=8.2, loc="upper left")
    for sp in ax_lat.spines.values():
        sp.set_color(grid_c)

    # Bottom invariant strip
    par16 = metrics["invariants"]["fp32_sparse_vs_dense_parity_z16"]["max_abs_diff_all"]
    par32 = metrics["invariants"]["fp32_sparse_vs_dense_parity_z32"]["max_abs_diff_all"]
    hm16 = metrics["invariants"]["closed_hydrologic_30t"]["Z16"]["max_rel_mass_err"]
    hm32 = metrics["invariants"]["closed_hydrologic_30t"]["Z32"]["max_rel_mass_err"]
    n_p = metrics["n_params_v5"]
    footer = (
        f"WorldModel3D-v5: {n_p:,} params | 48-ch Z_world_3d | Δt=0 Stasis: {metrics['invariants']['exact_zero_dt_stasis_max_abs']:.1e} | "
        f"FP32 Sparse Parity: Z16={par16:.2e}, Z32={par32:.2e} (<1e-5) | Closed Hydrologic Mass Err: Z16={hm16:.2e}, Z32={hm32:.2e} (<1e-8)"
    )
    fig.text(0.5, 0.022, footer, ha="center", color=muted, fontsize=8.8, family="monospace")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=bg, dpi=150)
    plt.close(fig)


# =============================================================================
# Main Training + Comprehensive Benchmark Entrypoint
# =============================================================================


def train_and_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[Wave 11A] Training WorldModel3D-v5 on {dev_name} ({device})", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    model = WorldModel3DV5(in_ch=18, out_ch=17, width=48, z_flux_ch=Z_FLUX_CH, dt_dim=16, mid_3d=32).to(device)
    n_params = count_parameters(model)
    print(f"[Wave 11A] WorldModel3DV5 parameters: {n_params:,} (halo_xy={model.halo_xy}, halo_z={model.halo_z})", flush=True)
    assert 140_000 <= n_params <= 185_000, f"Parameter count {n_params} outside target ~150K-180K range!"

    # Load WorldModelDT (Z=4 trained) baseline for exact head-to-head comparison
    baseline_dt: WorldModelDT | None = None
    dt_ckpt = Path(args.baseline_dt_ckpt)
    if dt_ckpt.exists():
        baseline_dt = WorldModelDT(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=Z_FLUX_CH).to(device).eval()
        raw_sd = torch.load(dt_ckpt, map_location=device, weights_only=False)
        if isinstance(raw_sd, dict) and "state_dict" in raw_sd:
            sd = raw_sd["state_dict"]
        elif isinstance(raw_sd, dict) and "model_state_dict" in raw_sd:
            sd = raw_sd["model_state_dict"]
        else:
            sd = raw_sd
        baseline_dt.load_state_dict(sd)
        print(f"[Wave 11A] Loaded WorldModelDT (Z=4) baseline from {dt_ckpt} ({count_parameters(baseline_dt):,} params)", flush=True)
    else:
        print(f"[Wave 11A] WARNING: baseline checkpoint {dt_ckpt} not found; using fresh WorldModelDT", flush=True)
        baseline_dt = WorldModelDT(in_ch=18, out_ch=17, width=48, num_res=5, z_flux_ch=Z_FLUX_CH).to(device).eval()

    # Maintain live rolling pool of 5-biome Z=16 (96x96x16 -> crop 64x64x16) & Z=32 (64x64x32 -> crop 48x48x32) worlds!
    pool_z16: list[tuple[int, SimState, int]] = []
    pool_z32: list[tuple[int, SimState, int]] = []
    seed_counter = args.seed * 100

    for b_id in range(len(BIOME_NAMES)):
        seed_counter += 1
        pool_z16.append((b_id, make_biome_3d_world(b_id, 16, 96, 96, device, seed=seed_counter), 0))
        seed_counter += 1
        pool_z32.append((b_id, make_biome_3d_world(b_id, 32, 64, 64, device, seed=seed_counter), 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.04)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    t_start = time.time()
    history: list[dict[str, Any]] = []
    best_score = -1e9
    best_state_dict: dict[str, torch.Tensor] | None = None

    for step in range(1, args.steps + 1):
        model.train()
        # Alternate 65% Z=16 (64x64x16 crops) and 35% Z=32 (48x48x32 crops) across all 5 biomes!
        use_z32 = (step % 3 == 0)
        b_id = (step - 1) % len(BIOME_NAMES)

        if use_z32:
            _, full_st, age = pool_z32[b_id]
            L_cur, ch_c, cw_c = 32, 48, 48
        else:
            _, full_st, age = pool_z16[b_id]
            L_cur, ch_c, cw_c = 16, 64, 64

        # Advance full world state 1 tick with random forcing so training sees evolved waterfalls/plumes/rain/collapse!
        fe_full, fm_full = sample_3d_forcing(L_cur, full_st.mass.shape[2], full_st.mass.shape[3], device, prob=0.45)
        next_full, _, _ = RefSim3D.step(full_st, fe_full, fm_full, local_dt=1.0, radiative_cooling=True, structural_collapse=True)
        age += 1
        if age >= 24:
            seed_counter += 1
            H_f, W_f = full_st.mass.shape[2], full_st.mass.shape[3]
            next_full = make_biome_3d_world(b_id, L_cur, H_f, W_f, device, seed=seed_counter)
            age = 0
        if use_z32:
            pool_z32[b_id] = (b_id, next_full, age)
        else:
            pool_z16[b_id] = (b_id, next_full, age)

        # Random spatial crop + continuous Δt sampling + exact RefSim3D teacher target
        st_crop = crop_sim_state(full_st, ch_c, cw_c)
        fe_c, fm_c = sample_3d_forcing(L_cur, ch_c, cw_c, device, prob=0.45)
        dt_crop = sample_training_dt(L_cur, ch_c, cw_c, device)

        with torch.no_grad():
            x_in = encode_state(st_crop, fe_c, fm_c)
            _, gt_h, gt_z = RefSim3D.step(
                st_crop, fe_c, fm_c, local_dt=dt_crop, radiative_cooling=True, structural_collapse=True
            )
            m_eff, e_eff = apply_forcing(st_crop, fe_c, fm_c)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred_h, pred_z, _ = model(x_in, local_dt=dt_crop, return_z_flux=True, return_latent=True)

        # Compute physics & conservation firewall losses in full FP32 for exact gradients!
        pred_h_f = pred_h.float()
        pred_z_f = pred_z.float()
        l_horiz, h_info = horiz_teacher_loss_3d(pred_h_f, gt_h, dt_crop)
        l_vert, v_info = vertical_teacher_loss_3d(pred_z_f, gt_z, st_crop, dt_crop)

        # Conservation firewall distillation penalty (ramps up cleanly after step 150!)
        fw_weight = min(1.0, max(0.25, step / 400.0))
        with torch.no_grad():
            proj_h_pred, _ = project_deltas(m_eff, e_eff, pred_h_f.detach())
        l_fw, fw_info = conservation_firewall_loss_3d(m_eff, e_eff, pred_h_f, proj_h_pred, dt_crop)

        loss = l_horiz + 2.5 * l_vert + fw_weight * 0.45 * l_fw
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            elapsed = time.time() - t_start
            # Fast validation check on unseen 5-biome 64x64x16
            val = evaluate_model_on_5biomes(model, L=16, H=64, W=64, device=device, base_seed=88000, warmup_ticks=2, eval_ticks=3)
            proj_pct = val["proj_active_pct"]
            h_r2 = val["horizontal"]["all"]["r2"]
            v_r2 = val["vertical_3d"]["mean_vertical_r2"]
            hy_r2 = val["hydrologic_cycle"]["mean_hydro_r2"]
            dt015_r2 = val["continuous_dt"]["0.15"]["mean_r2"]
            dt400_r2 = val["continuous_dt"]["4.00"]["mean_r2"]
            score = h_r2 + v_r2 + hy_r2 + 0.5 * (dt015_r2 + dt400_r2) - 0.04 * max(0.0, proj_pct - 4.0)
            if score > best_score:
                best_score = score
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            row = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "loss": round(float(loss.item()), 4),
                "val_proj_active_pct": proj_pct,
                "val_horiz_r2": h_r2,
                "val_vert_mean_r2": v_r2,
                "val_hydro_mean_r2": hy_r2,
                "val_dt015_r2": dt015_r2,
                "val_dt400_r2": dt400_r2,
            }
            history.append(row)
            print(
                f"  step {step:4d}/{args.steps} ({elapsed:5.1f}s) | loss={loss.item():.3f} | "
                f"proj={proj_pct:5.2f}% | H_R2={h_r2:.4f} | V_R2={v_r2:.4f} | "
                f"Hydro_R2={hy_r2:.4f} | dt0.15={dt015_r2:.4f} | dt4.0={dt400_r2:.4f}",
                flush=True,
            )

    train_time_s = round(time.time() - t_start, 2)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    # -------------------------------------------------------------------------
    # Final Comprehensive Benchmarks on Unseen Held-Out 5-Biome Z=16 & Z=32!
    # -------------------------------------------------------------------------
    print("\n[Wave 11A] Running Final Held-Out 5-Biome Evaluation (Z=16 vs WorldModelDT, and Z=32)...", flush=True)
    eval_v5_z16 = evaluate_model_on_5biomes(model, L=16, H=128, W=128, device=device, base_seed=95000, warmup_ticks=4, eval_ticks=6)
    eval_dt_z16 = evaluate_model_on_5biomes(baseline_dt, L=16, H=128, W=128, device=device, base_seed=95000, warmup_ticks=4, eval_ticks=6)
    eval_v5_z32 = evaluate_model_on_5biomes(model, L=32, H=96, W=96, device=device, base_seed=97000, warmup_ticks=4, eval_ticks=5)

    print(
        f"  Z=16 WorldModelDT (Z=4): proj={eval_dt_z16['proj_active_pct']:.2f}% | "
        f"H_R2={eval_dt_z16['horizontal']['all']['r2']:.4f} | V_R2={eval_dt_z16['vertical_3d']['mean_vertical_r2']:.4f} | "
        f"Hydro_R2={eval_dt_z16['hydrologic_cycle']['mean_hydro_r2']:.4f}",
        flush=True,
    )
    print(
        f"  Z=16 WorldModel3D-v5:    proj={eval_v5_z16['proj_active_pct']:.2f}% | "
        f"H_R2={eval_v5_z16['horizontal']['all']['r2']:.4f} | V_R2={eval_v5_z16['vertical_3d']['mean_vertical_r2']:.4f} | "
        f"Hydro_R2={eval_v5_z16['hydrologic_cycle']['mean_hydro_r2']:.4f}",
        flush=True,
    )
    print(
        f"  Z=32 WorldModel3D-v5:    proj={eval_v5_z32['proj_active_pct']:.2f}% | "
        f"H_R2={eval_v5_z32['horizontal']['all']['r2']:.4f} | V_R2={eval_v5_z32['vertical_3d']['mean_vertical_r2']:.4f} | "
        f"Hydro_R2={eval_v5_z32['hydrologic_cycle']['mean_hydro_r2']:.4f}",
        flush=True,
    )

    print("\n[Wave 11A] Verifying Exact Invariants (Δt=0 stasis, FP32 sparse parity, closed hydrologic cycle, arch collapse)...", flush=True)
    invariants = verify_closed_hydrologic_and_collapse(model, device)
    parity_z16 = verify_sparse_v5_parity(model, L=16, H=64, W=64, seed=77)
    parity_z32 = verify_sparse_v5_parity(model, L=32, H=64, W=64, seed=78)
    invariants["fp32_sparse_vs_dense_parity_z16"] = parity_z16
    invariants["fp32_sparse_vs_dense_parity_z32"] = parity_z32
    print(f"  Δt=0 stasis max_abs:      {invariants['exact_zero_dt_stasis_max_abs']:.2e}", flush=True)
    print(f"  FP32 Sparse Parity Z=16:  {parity_z16['max_abs_diff_all']:.2e} (< 1e-5: {parity_z16['max_abs_diff_all'] < 1e-5})", flush=True)
    print(f"  FP32 Sparse Parity Z=32:  {parity_z32['max_abs_diff_all']:.2e} (< 1e-5: {parity_z32['max_abs_diff_all'] < 1e-5})", flush=True)
    print(f"  Closed Hydrologic Z=16:   {invariants['closed_hydrologic_30t']['Z16']}", flush=True)
    print(f"  Closed Hydrologic Z=32:   {invariants['closed_hydrologic_30t']['Z32']}", flush=True)
    print(f"  Structural Collapse Z=16: {invariants['structural_collapse_16t']}", flush=True)

    assert invariants["exact_zero_dt_stasis_max_abs"] == 0.0, "Exact Δt=0 stasis violated!"
    assert parity_z16["max_abs_diff_all"] < 1e-5, f"Z=16 FP32 sparse parity failed: {parity_z16}"
    assert parity_z32["max_abs_diff_all"] < 1e-5, f"Z=32 FP32 sparse parity failed: {parity_z32}"
    assert invariants["closed_hydrologic_30t"]["Z16"]["mass_conserved_lt_1e8"], "Z=16 hydrologic mass conservation failed!"
    assert invariants["closed_hydrologic_30t"]["Z32"]["mass_conserved_lt_1e8"], "Z=32 hydrologic mass conservation failed!"

    print("\n[Wave 11A] Benchmarking FP16 Dense vs Sparse 16x16x8 3D Brick Scaling (1.05M / 2.10M / 4.19M voxels)...", flush=True)
    lat_bench = benchmark_fp16_brick_scaling(model, baseline_dt, device)
    for k, v in lat_bench.items():
        d_ms = v["dense_v5_fp16"]["p50_ms"]
        s20 = v["sparse_16x16x8_fp16"]["sparse_20pct"]
        s10 = v["sparse_16x16x8_fp16"]["sparse_10pct"]
        print(
            f"  {k:18s}: Dense={d_ms:6.2f} ms | Sparse-20%={s20['p50_ms']:5.2f} ms ({s20['speedup_vs_dense_x']:.2f}x) | "
            f"Sparse-10%={s10['p50_ms']:5.2f} ms ({s10['speedup_vs_dense_x']:.2f}x, {s10['hz']:.1f} Hz)",
            flush=True,
        )

    # Save checkpoint, metrics JSON, and 3-panel visualization!
    out_ckpt = Path(args.out_ckpt)
    out_metrics = Path(args.out_metrics)
    out_viz = Path(args.out_viz)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_viz.parent.mkdir(parents=True, exist_ok=True)

    ckpt_payload = {
        "model_class": "WorldModel3DV5",
        "arch": {
            "in_ch": 18,
            "out_ch": 17,
            "width": 48,
            "z_flux_ch": Z_FLUX_CH,
            "dt_dim": 16,
            "mid_3d": 32,
            "halo_xy": model.halo_xy,
            "halo_z": model.halo_z,
            "brick_size": [8, 16, 16],
        },
        "n_params": n_params,
        "biomes": list(BIOME_NAMES),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(ckpt_payload, out_ckpt)

    metrics: dict[str, Any] = {
        "wave": "11A",
        "device": dev_name,
        "model_class": "WorldModel3DV5",
        "checkpoint": str(out_ckpt),
        "visualization": str(out_viz),
        "arch": ckpt_payload["arch"],
        "n_params_v5": n_params,
        "n_params_dt_baseline": count_parameters(baseline_dt) if baseline_dt is not None else 0,
        "biomes": list(BIOME_NAMES),
        "train_steps": args.steps,
        "train_time_s": train_time_s,
        "eval_z16_5biomes": {
            "world_model_3d_v5": eval_v5_z16,
            "world_model_dt_z4_baseline": eval_dt_z16,
            "proj_active_reduction_abs_pct": round(eval_dt_z16["proj_active_pct"] - eval_v5_z16["proj_active_pct"], 3),
            "proj_active_reduction_factor_x": round(
                eval_dt_z16["proj_active_pct"] / max(eval_v5_z16["proj_active_pct"], 1e-4), 2
            ),
        },
        "eval_z32_5biomes": {
            "world_model_3d_v5": eval_v5_z32,
        },
        "invariants": invariants,
        "latency_fp16_brick_scaling": lat_bench,
        "history": history,
    }

    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    render_v5_showcase(model, metrics, device, out_viz)
    print(f"\n[Wave 11A] Saved:\n  - {out_ckpt}\n  - {out_metrics}\n  - {out_viz}", flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wave 11A: Train & Benchmark WorldModel3D-v5")
    p.add_argument("--steps", type=int, default=2600, help="Training steps")
    p.add_argument("--eval-every", type=int, default=325, help="Validation cadence")
    p.add_argument("--lr", type=float, default=1.8e-3, help="Initial AdamW learning rate")
    p.add_argument("--seed", type=int, default=1101, help="Random seed")
    p.add_argument("--baseline-dt-ckpt", type=str, default="models/checkpoints/world_model_dt.pt")
    p.add_argument("--out-ckpt", type=str, default="models/checkpoints/world_model_3d_v5.pt")
    p.add_argument("--out-metrics", type=str, default="models/WORLD_3D_V5_METRICS.json")
    p.add_argument("--out-viz", type=str, default="models/world_3d_v5_viz.png")
    return p.parse_args()


if __name__ == "__main__":
    train_and_benchmark(parse_args())
