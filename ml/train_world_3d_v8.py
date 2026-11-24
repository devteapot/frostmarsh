#!/usr/bin/env python3
"""
Frostmarsh Wave 20A: Training & Head-to-Head Evaluation for `WorldModel3D-v8`
— Law-Conditioned Universal 3D Physics World Model (`ml/train_world_3d_v8.py`).

Warm-starts from `models/checkpoints/world_model_3d_v7.pt` (bit-for-bit Step-0 identity!)
and trains `2,000–2,400` steps (`bfloat16`, `GradScaler(enabled=False)`) across:
  - 25% canonical Frostmarsh `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0]`
  - 20% axis-disentangled 1-or-2-law continuous perturbations
  - 55% broad 6D continuous World-Law space (`gravity_g`, `ambient_temp_shift`, `volatility`,
    `viscosity`, `precipitation`, `thermal_cond`)
  - All 5 biomes (`Z=16` & `Z=32`) & continuous `local_dt` / time-dilation domes
  - Wave 17A smooth-deadzone conservation firewall + variance-normalized active-face R² losses

Outputs:
  - `models/checkpoints/world_model_3d_v8.pt`
  - `models/WORLD_3D_V8_METRICS.json`
  - `models/world_3d_v8_viz.png`
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_teacher import (  # noqa: E402
    H_CAP,
    S_ICE,
    S_ROCK,
    T_AMB,
    SimState,
    _apply_transport,
    apply_forcing,
    compute_C_and_T,
    encode_state,
    get_consts,
    sample_bolts,
)
from train_world_3d_v5 import sample_3d_forcing  # noqa: E402
from world_model_3d import ZF_HEAT, ZF_OIL, ZF_SMOKE, ZF_STEAM, ZF_WATER  # noqa: E402
from world_model_3d_v5 import (  # noqa: E402
    BIOME_NAMES,
    Z_SPECIES_IDX,
    WorldModel3DV5,
    count_parameters,
    make_biome_3d_world,
)
from world_model_3d_v8 import (  # noqa: E402
    DEFAULT_WORLD_LAWS,
    LAW_DIM,
    WORLD_LAW_PRESET_NAMES,
    WORLD_LAW_PRESETS,
    WorldLawsLike,
    WorldModel3DV8,
    compute_law_rxn_thresholds,
    project_deltas_with_laws,
    step_refsim_3d_with_laws,
    verify_canonical_teacher_identity,
    verify_sparse_v8_parity,
    verify_v7_warmstart_identity,
)


# =============================================================================
# Continuous 6-Float Physical World-Law Sampler
# =============================================================================


def _sample_log_uniform(lo: float, hi: float, gen: torch.Generator, device: torch.device) -> float:
    u = float(torch.rand(1, generator=gen, device=device).item())
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def _sample_uniform(lo: float, hi: float, gen: torch.Generator, device: torch.device) -> float:
    u = float(torch.rand(1, generator=gen, device=device).item())
    return lo + u * (hi - lo)


def sample_world_laws(
    gen: torch.Generator,
    device: torch.device,
    force_non_canonical: bool = False,
) -> tuple[float, float, float, float, float, float]:
    """
    Sample continuous 6-float physical world laws `theta_world`:
      - 25%: exact canonical Frostmarsh `(1.0, 0.0, 1.0, 1.0, 1.0, 1.0)`
      - 20%: disentangled 1-or-2-axis continuous perturbation
      - 55%: full 6D broad continuous physics law space covering extreme cold, floods,
             low-G, high volatility, high/low viscosity, and thermal conductivity!
    """
    u = float(torch.rand(1, generator=gen, device=device).item())
    if (not force_non_canonical) and (u < 0.25):
        return DEFAULT_WORLD_LAWS

    def _draw_axis(ax: int) -> float:
        if ax == 0:
            return _sample_log_uniform(0.25, 1.80, gen, device)
        if ax == 1:
            return _sample_uniform(-7000.0, 7000.0, gen, device)
        if ax == 2:
            return _sample_log_uniform(0.40, 2.80, gen, device)
        if ax == 3:
            return _sample_log_uniform(0.40, 2.20, gen, device)
        if ax == 4:
            return _sample_log_uniform(0.20, 3.20, gen, device)
        return _sample_log_uniform(0.45, 2.20, gen, device)

    if u < 0.45:
        vals = list(DEFAULT_WORLD_LAWS)
        n_ax = 1 if float(torch.rand(1, generator=gen, device=device).item()) < 0.5 else 2
        for _ in range(n_ax):
            ax = int(torch.randint(0, LAW_DIM, (1,), generator=gen, device=device).item())
            vals[ax] = _draw_axis(ax)
        return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])

    return (
        _draw_axis(0),
        _draw_axis(1),
        _draw_axis(2),
        _draw_axis(3),
        _draw_axis(4),
        _draw_axis(5),
    )


# =============================================================================
# Wave 20A Law-Aware 3D Losses (Extending Wave 17A Smooth Deadzone Firewall)
# =============================================================================


def v8_active_face_horiz_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    local_dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Law-balanced Wave 20A global + active-face variance-normalized horizontal loss across ALL 3
    groups (flux `0:8`, heat `8:10`, reactions `10:17`) with boosted per-reaction balance for
    Freeze (rxn 0), Condense (rxn 3), and volatile combustion (rxn 4..6) so `Freeze-Out Survival`
    and `Rising Flood Canyon` converge crisply alongside canonical Frostmarsh!
    """
    dt_n = local_dt.clamp(min=0.25)
    p = pred / dt_n
    t = target / dt_n

    def _group_loss(
        pg: torch.Tensor,
        tg: torch.Tensor,
        act_thresh: float,
        scale_ref: float,
        var_floor: float,
        inact_w: float,
    ) -> tuple[torch.Tensor, float]:
        var_g = tg.var(unbiased=False).clamp(min=var_floor)
        l_glob_r2 = F.mse_loss(pg, tg) / var_g
        l_glob_hub = F.smooth_l1_loss(pg / scale_ref, tg / scale_ref, beta=0.04)

        act = tg.abs() > act_thresh
        if act.any():
            pa = pg[act]
            ta = tg[act]
            var_a = ta.var(unbiased=False).clamp(min=var_floor)
            l_act_r2 = F.mse_loss(pa, ta) / var_a
            l_act_hub = F.smooth_l1_loss(pa / scale_ref, ta / scale_ref, beta=0.08)
            l_act = l_act_r2 + 0.5 * l_act_hub
        else:
            l_act = pg.new_tensor(0.0)

        inact = ~act
        if inact.any():
            pi = pg[inact] / scale_ref
            l_inact = (pi * pi).mean()
        else:
            l_inact = pg.new_tensor(0.0)

        tot = l_glob_r2 + 0.65 * l_act + 0.5 * l_glob_hub + inact_w * l_inact
        return tot, float(l_glob_r2.item())

    l_flux, r2l_f = _group_loss(p[:, 0:8], t[:, 0:8], act_thresh=5e-4, scale_ref=0.05, var_floor=1e-5, inact_w=2.0)
    l_heat, r2l_h = _group_loss(p[:, 8:10], t[:, 8:10], act_thresh=5e-3, scale_ref=1.50, var_floor=1e-3, inact_w=2.0)
    l_rxn, r2l_r = _group_loss(p[:, 10:17], t[:, 10:17], act_thresh=5e-4, scale_ref=0.10, var_floor=1e-5, inact_w=2.8)

    # Strong per-reaction balance (wakes up Freeze rxn[0] in deep-freeze & Melt/Burn/Condense!)
    rxn_weights = (2.4, 1.4, 1.2, 1.6, 1.5, 1.5, 1.3)
    l_rxn_per = pred.new_tensor(0.0)
    w_sum = 0.0
    for r in range(7):
        tr = t[:, 10 + r]
        pr = p[:, 10 + r]
        mr = tr > 5e-4
        if mr.any():
            vr_g = tr.var(unbiased=False).clamp(min=1e-5)
            vr_a = tr[mr].var(unbiased=False).clamp(min=1e-4)
            w_r = rxn_weights[r]
            l_r = (F.mse_loss(pr, tr) / vr_g).clamp(max=25.0) + 0.75 * (F.mse_loss(pr[mr], tr[mr]) / vr_a).clamp(
                max=25.0
            )
            l_rxn_per = l_rxn_per + w_r * l_r
            w_sum += w_r
    if w_sum > 0.0:
        l_rxn_per = l_rxn_per / w_sum

    total = 1.65 * l_flux + 3.40 * l_heat + 1.45 * l_rxn + 1.35 * l_rxn_per
    return total, {
        "flux_1mr2": r2l_f,
        "heat_1mr2": r2l_h,
        "rxn_1mr2": r2l_r,
    }


def v8_vertical_teacher_loss_3d(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    state: SimState,
    local_dt: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Well-conditioned law-invariant 3D vertical transport loss across ALL `L - 1` interfaces.
    Variance-normalized R² loss keeps low-G (0.35g) and high-G (1.5g) equally well-conditioned!
    """
    L = state.mass.shape[0]
    if L <= 1:
        z0 = pred_z.new_tensor(0.0)
        return z0, {"z_mse": 0.0, "z_donor": 0.0}

    dt_z = torch.minimum(local_dt[:-1], local_dt[1:]).clamp(min=0.25)
    ch_scale = torch.tensor([0.040, 0.020, 0.035, 0.018, 0.250], dtype=torch.float32, device=pred_z.device).view(
        1, 5, 1, 1
    )

    p_norm = (pred_z / dt_z) / ch_scale
    t_norm = (target_z / dt_z) / ch_scale

    l_mse = F.mse_loss(p_norm, t_norm)
    l_hub = F.smooth_l1_loss(p_norm, t_norm, beta=0.08)

    act_z = t_norm.abs() > 0.04
    l_act = F.smooth_l1_loss(p_norm[act_z], t_norm[act_z], beta=0.10) if act_z.any() else pred_z.new_tensor(0.0)

    # Scale-invariant variance-normalized R² across water, steam, smoke (matches Wave 17A well-conditioned design!)
    l_r2_z = pred_z.new_tensor(0.0)
    for ch, w_ch in ((ZF_WATER, 1.15), (ZF_STEAM, 1.15), (ZF_SMOKE, 1.0)):
        tc = target_z[:, ch] / dt_z[:, 0]
        pc = pred_z[:, ch] / dt_z[:, 0]
        vc = tc.var(unbiased=False)
        if float(vc.item()) > 1e-6:
            l_r2_z = l_r2_z + w_ch * (F.mse_loss(pc, tc) / vc.clamp(min=1e-5)).clamp(max=10.0)

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

    total = 2.4 * l_mse + 1.0 * l_hub + 2.0 * l_act + 0.75 * l_r2_z + 0.5 * l_sign + 5.0 * l_z_donor + 0.4 * l_z_2nd
    return total, {"z_mse": float(l_mse.item()), "z_donor": float(l_z_donor.item())}


def v8_smooth_firewall_loss_3d(
    mass_eff: torch.Tensor,
    energy_eff: torch.Tensor,
    raw_deltas: torch.Tensor,
    proj_deltas: torch.Tensor,
    local_dt: torch.Tensor,
    world_laws: WorldLawsLike = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Law-aware Wave 17A smooth deadzone-hinge Conservation Firewall Loss.
    Uses exact `compute_law_rxn_thresholds` for reaction gates so shifted freezing & ignition
    thresholds under arbitrary `world_laws` have ZERO conflicting gradients!
    """
    c = get_consts(mass_eff.device)
    L, _, H, W = mass_eff.shape
    dev = mass_eff.device
    rxn_thresh = compute_law_rxn_thresholds(L, world_laws, dev)

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

    def _hinge(excess: torch.Tensor, margin: float, beta: float) -> torch.Tensor:
        ex = F.relu(excess - margin)
        return F.smooth_l1_loss(ex, torch.zeros_like(ex), beta=beta) + 12.0 * (ex * ex).mean()

    mob_mass_norm = mass_eff[:, c.mobile_idx].clamp(min=0.0) / 1000.0
    fw_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    fn_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    total_out = F.relu(flux_e) + F.relu(-fw_in) + F.relu(flux_s) + F.relu(-fn_in)
    l_donor = _hinge(total_out - mob_mass_norm, margin=1.5e-3, beta=0.005)
    l_wall = _hinge(flux_e.abs() * (1.0 - open_e), margin=1.5e-3, beta=0.005) + _hinge(
        flux_s.abs() * (1.0 - open_s), margin=1.5e-3, beta=0.005
    )

    total_in = (F.relu(fw_in) + F.relu(-flux_e) + F.relu(fn_in) + F.relu(-flux_s)).sum(dim=1, keepdim=True)
    headroom_norm = (H_CAP - mass_eff.sum(dim=1, keepdim=True)).clamp(min=0.0) / 1000.0
    l_hcap = _hinge(total_in - headroom_norm, margin=2e-3, beta=0.01)

    C_cell, T_cK = compute_C_and_T(mass_eff, energy_eff)
    with torch.no_grad():
        pfe = proj_deltas[:, 0:4] * 1000.0
        pfs = proj_deltas[:, 4:8] * 1000.0
        pfw = F.pad(pfe[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
        pfn = F.pad(pfs[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
        C_adv = torch.clamp(C_cell + ((pfw - pfe + pfn - pfs) * c.mobile_c).sum(dim=1, keepdim=True), min=0.0)
        C_cond = torch.where(C_adv > 1e-2, torch.minimum(C_cell, C_adv), torch.zeros_like(C_cell))
    C_e_nb = F.pad(C_cond[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    T_e_nb = F.pad(T_cK[:, :, :, 1:], (0, 1, 0, 0), value=T_AMB)
    dT_e = T_cK - T_e_nb
    q_max_e = torch.abs(dT_e) * (C_cond * C_e_nb) / (4.0 * (C_cond + C_e_nb) + 1e-6)
    q_max_e = torch.where((C_cond > 1e-6) & (C_e_nb > 1e-6), q_max_e, torch.zeros_like(q_max_e)) * mask_e / 1e6

    C_s_nb = F.pad(C_cond[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    T_s_nb = F.pad(T_cK[:, :, 1:, :], (0, 0, 0, 1), value=T_AMB)
    dT_s = T_cK - T_s_nb
    q_max_s = torch.abs(dT_s) * (C_cond * C_s_nb) / (4.0 * (C_cond + C_s_nb) + 1e-6)
    q_max_s = torch.where((C_cond > 1e-6) & (C_s_nb > 1e-6), q_max_s, torch.zeros_like(q_max_s)) * mask_s / 1e6

    hv_e = (
        torch.where(dT_e > 1e-2, F.relu(-heat_e), torch.where(dT_e < -1e-2, F.relu(heat_e), heat_e.abs()))
        + F.relu(heat_e.abs() - q_max_e)
    ) / 10.0
    hv_s = (
        torch.where(dT_s > 1e-2, F.relu(-heat_s), torch.where(dT_s < -1e-2, F.relu(heat_s), heat_s.abs()))
        + F.relu(heat_s.abs() - q_max_s)
    ) / 10.0
    l_heat_b = _hinge(hv_e, margin=2.0e-3, beta=0.008) + _hinge(hv_s, margin=2.0e-3, beta=0.008)

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
        t_gate = (((T_tr - rxn_thresh) * c.rxn_dir) > 0.0).float() * (C_tr > 1e-6).float()
    l_rxn_gate = _hinge(rxn * (1.0 - t_gate), margin=1.5e-3, beta=0.005)
    consumed = torch.einsum("brhw,rs->bshw", rxn * 100.0, c.nu_neg)
    l_rxn_react = _hinge((consumed - mass_tr) / 100.0, margin=1.5e-3, beta=0.005)

    dt_w = (local_dt <= 1.25).float()
    diff = (raw_deltas - proj_deltas.detach()).abs()
    p_flux = _hinge(diff[:, 0:8] * dt_w, margin=1.5e-3, beta=0.005)
    p_heat = _hinge((diff[:, 8:10] / 10.0) * dt_w, margin=2.0e-3, beta=0.008)
    p_rxn = _hinge(diff[:, 10:17] * dt_w, margin=1.5e-3, beta=0.005)

    ex_rf = F.relu(diff[:, 0:8] * dt_w - 2.5e-3)
    ex_rh = F.relu(diff[:, 8:10] * dt_w - 0.025)
    ex_rr = F.relu(diff[:, 10:17] * dt_w - 2.5e-3)
    rate_flux = (ex_rf / (ex_rf.detach() + 0.015)).amax(dim=1).mean()
    rate_heat = (ex_rh / (ex_rh.detach() + 0.150)).amax(dim=1).mean()
    rate_rxn = (ex_rr / (ex_rr.detach() + 0.015)).amax(dim=1).mean()

    total = (
        18.0 * l_donor
        + 14.0 * l_wall
        + 6.0 * l_hcap
        + 10.0 * l_heat_b
        + 28.0 * (l_rxn_gate + l_rxn_react)
        + 18.0 * p_flux
        + 10.0 * p_heat
        + 30.0 * p_rxn
        + 10.0 * rate_flux
        + 14.0 * rate_heat
        + 18.0 * rate_rxn
    )
    return total, {
        "donor": float(l_donor.item()),
        "heat_b": float(l_heat_b.item()),
        "rxn_b": float((l_rxn_gate + l_rxn_react + p_rxn).item()),
    }


# =============================================================================
# Head-to-Head Evaluation Across ALL 4 World-Law Presets (v7 vs v8)
# =============================================================================


def _r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.float().reshape(-1)
    t = target.float().reshape(-1)
    if t.numel() == 0:
        return 0.0
    var = ((t - t.mean()) ** 2).sum().item()
    if var < 1e-12:
        return 0.0
    sse = ((p - t) ** 2).sum().item()
    return float(1.0 - sse / var)


@torch.no_grad()
def evaluate_models_world_laws(
    models: dict[str, nn.Module],
    device: torch.device,
    L: int = 16,
    H: int = 96,
    W: int = 96,
    presets: tuple[str, ...] = WORLD_LAW_PRESET_NAMES,
) -> dict[str, Any]:
    """
    Reproducible held-out 5-biome evaluation across ALL 4 World-Law Presets on unseen seeds.
    Preset 0 (`frostmarsh_standard`) uses exact seeds `8800 + bid * 31` matching Wave 17A!
    """
    out: dict[str, Any] = {}

    for preset_idx, pname in enumerate(presets):
        laws = WORLD_LAW_PRESETS[pname]
        eval_cases: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for bid, bname in enumerate(BIOME_NAMES):
            seed_w = 8800 + bid * 31 + preset_idx * 101
            seed_b = 9900 + bid * 13 + L + preset_idx * 97
            st = make_biome_3d_world(bid, L=L, H=H, W=W, device=device, seed=seed_w)
            torch.manual_seed(seed_b)
            for _ in range(3):
                fh, fw = sample_bolts(L, H, W, device)
                st, _, _ = step_refsim_3d_with_laws(st, fh, fw, local_dt=1.0, world_laws=laws)
            fh, fw = sample_bolts(L, H, W, device)
            x_in = encode_state(st, fh, fw)
            _, tgt_h, tgt_z = step_refsim_3d_with_laws(st, fh, fw, local_dt=1.0, world_laws=laws)
            mass_eff, energy_eff = apply_forcing(st, fh, fw)
            eval_cases.append((bname, x_in, tgt_h, tgt_z, mass_eff, energy_eff))

        preset_res: dict[str, Any] = {
            "world_laws": list(laws),
            "grid": f"{W}x{H}x{L}",
        }
        for m_name, m in models.items():
            m.eval()
            p_f, t_f, p_h, t_h, p_r, t_r = [], [], [], [], [], []
            pz_w, tz_w, pz_o, tz_o, pz_s, tz_s, pz_k, tz_k, pz_q, tz_q = [], [], [], [], [], [], [], [], [], []
            proj_tot, vox_tot = 0, 0
            proj_f_tot, proj_h_tot, proj_r_tot = 0, 0, 0

            for _, x_in, tgt_h, tgt_z, mass_eff, energy_eff in eval_cases:
                if isinstance(m, WorldModel3DV8):
                    pred_h, pred_z, _ = m(x_in, local_dt=1.0, world_laws=laws, return_z_flux=True, return_latent=True)
                else:
                    pred_h, pred_z, _ = m(x_in, local_dt=1.0, return_z_flux=True, return_latent=True)

                proj_h, _ = project_deltas_with_laws(mass_eff, energy_eff, pred_h, world_laws=laws)
                gap = (pred_h - proj_h).abs()
                act_f = gap[:, :8].amax(dim=1) > 1e-2
                act_h = gap[:, 8:10].amax(dim=1) > 0.1
                act_r = gap[:, 10:17].amax(dim=1) > 1e-2
                act_v = act_f | act_h | act_r
                proj_tot += int(act_v.sum().item())
                proj_f_tot += int(act_f.sum().item())
                proj_h_tot += int(act_h.sum().item())
                proj_r_tot += int(act_r.sum().item())
                vox_tot += int(act_v.numel())

                p_f.append(pred_h[:, 0:8].cpu()); t_f.append(tgt_h[:, 0:8].cpu())
                p_h.append(pred_h[:, 8:10].cpu()); t_h.append(tgt_h[:, 8:10].cpu())
                p_r.append(pred_h[:, 10:17].cpu()); t_r.append(tgt_h[:, 10:17].cpu())
                pz_w.append(pred_z[:, ZF_WATER].cpu()); tz_w.append(tgt_z[:, ZF_WATER].cpu())
                pz_o.append(pred_z[:, ZF_OIL].cpu()); tz_o.append(tgt_z[:, ZF_OIL].cpu())
                pz_s.append(pred_z[:, ZF_STEAM].cpu()); tz_s.append(tgt_z[:, ZF_STEAM].cpu())
                pz_k.append(pred_z[:, ZF_SMOKE].cpu()); tz_k.append(tgt_z[:, ZF_SMOKE].cpu())
                pz_q.append(pred_z[:, ZF_HEAT].cpu()); tz_q.append(tgt_z[:, ZF_HEAT].cpu())

            rf = _r2(torch.cat(p_f), torch.cat(t_f))
            rh = _r2(torch.cat(p_h), torch.cat(t_h))
            rr = _r2(torch.cat(p_r), torch.cat(t_r))
            rw = _r2(torch.cat(pz_w), torch.cat(tz_w))
            ro = _r2(torch.cat(pz_o), torch.cat(tz_o))
            rs = _r2(torch.cat(pz_s), torch.cat(tz_s))
            rk = _r2(torch.cat(pz_k), torch.cat(tz_k))
            rq = _r2(torch.cat(pz_q), torch.cat(tz_q))
            h_mean = (rf + rh + rr) / 3.0
            v_mean = (rw + rs + rk) / 3.0
            all_mean = (rf + rh + rr + rw + rs + rk) / 6.0

            preset_res[m_name] = {
                "proj_active_pct": round(100.0 * proj_tot / max(vox_tot, 1), 3),
                "proj_flux_pct": round(100.0 * proj_f_tot / max(vox_tot, 1), 3),
                "proj_heat_pct": round(100.0 * proj_h_tot / max(vox_tot, 1), 3),
                "proj_rxn_pct": round(100.0 * proj_r_tot / max(vox_tot, 1), 3),
                "horiz_flux_r2": round(rf, 4),
                "horiz_heat_r2": round(rh, 4),
                "horiz_rxn_r2": round(rr, 4),
                "horiz_mean_r2": round(h_mean, 4),
                "vert_water_r2": round(rw, 4),
                "vert_oil_r2": round(ro, 4),
                "vert_steam_r2": round(rs, 4),
                "vert_smoke_r2": round(rk, 4),
                "vert_heat_r2": round(rq, 4),
                "vert_mean_r2": round(v_mean, 4),
                "overall_mean_r2": round(all_mean, 4),
            }
        out[pname] = preset_res
    return out


# =============================================================================
# Crisp 3-Panel Dark-Theme Chart (`models/world_3d_v8_viz.png`)
# =============================================================================


def save_v8_viz(metrics: dict[str, Any], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Viz] Skipped ({exc})", flush=True)
        return

    e16 = metrics["eval_z16"]
    presets = WORLD_LAW_PRESET_NAMES
    short_labels = [
        "Frostmarsh\nStandard",
        "Freeze-Out\nSurvival\n(-55K)",
        "Rising Flood\nCanyon\n(2.6x Rain)",
        "Low-G Magma\nCrucible\n(0.35g, +55K)",
    ]

    v7_h = [e16[p]["world_model_3d_v7"]["horiz_mean_r2"] for p in presets]
    v8_h = [e16[p]["world_model_3d_v8"]["horiz_mean_r2"] for p in presets]

    v7_v = [e16[p]["world_model_3d_v7"]["vert_mean_r2"] for p in presets]
    v8_v = [e16[p]["world_model_3d_v8"]["vert_mean_r2"] for p in presets]

    v7_fw = [e16[p]["world_model_3d_v7"]["proj_active_pct"] for p in presets]
    v8_fw = [e16[p]["world_model_3d_v8"]["proj_active_pct"] for p in presets]

    bg = "#0d1117"
    card = "#161b22"
    fg = "#f0f6fc"
    grid_c = "#30363d"
    c_v7 = "#f85149"
    c_v8_h = "#3fb950"
    c_v8_v = "#58a6ff"
    c_v8_fw = "#39c5cf"

    fig, axes = plt.subplots(1, 3, figsize=(19.2, 6.1), dpi=150, facecolor=bg)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.80, bottom=0.18, wspace=0.25)
    fig.suptitle(
        "Frostmarsh Wave 20A: WorldModel3D-v8 (Law-Conditioned, 164.6K) vs Law-Blind v7 Across 4 Physical Regimes",
        color=fg,
        fontsize=14,
        fontweight="bold",
        y=0.95,
    )
    fig.text(
        0.5,
        0.885,
        "Held-out 5-biome 96×96×16 evaluation | θ_world = [gravity_g, ΔT_amb, volatility, viscosity, precipitation, thermal_cond]",
        ha="center",
        color="#8b949e",
        fontsize=10,
    )

    for ax in axes:
        ax.set_facecolor(card)
        ax.tick_params(colors="#c9d1d9", labelsize=8.8)
        ax.grid(axis="y", color=grid_c, alpha=0.45, linestyle="--", linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color(grid_c)

    x = np.arange(len(presets))
    w = 0.34

    # --- Panel 1: Horizontal Mean R² (Flux / Heat / Rxn) ---
    ax0 = axes[0]
    v7_h_plot = [max(0.0, val) for val in v7_h]
    b7_h = ax0.bar(x - w / 2, v7_h_plot, w, color=c_v7, alpha=0.9, label="v7 (Law-Blind)")
    b8_h = ax0.bar(x + w / 2, v8_h, w, color=c_v8_h, label="v8 (Law-Conditioned)")
    ax0.axhline(0.85, color="#d29922", linestyle="--", linewidth=1.2, label="Target ≥ 0.85")
    for b, raw_v in zip(b7_h, v7_h):
        y_pos = max(0.0, raw_v) + 0.022
        lbl = f"{raw_v:.2f}" if raw_v < 0.0 else f"{raw_v:.3f}"
        ax0.text(
            b.get_x() + b.get_width() / 2.0,
            min(y_pos, 1.07),
            lbl,
            ha="center",
            va="bottom",
            color="#ff7b72",
            fontsize=8.2,
            fontweight="bold",
        )
    for b, val in zip(b8_h, v8_h):
        ax0.text(
            b.get_x() + b.get_width() / 2.0,
            min(val + 0.022, 1.07),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color="#56d364",
            fontsize=8.4,
            fontweight="bold",
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(short_labels, color=fg, fontsize=8.4)
    ax0.set_ylabel("Horizontal Mean R² (Flux / Heat / Rxn)", color=fg, fontsize=9.5)
    ax0.set_ylim(0.0, 1.15)
    ax0.set_title("Horizontal Physics R² Across World Laws", color=fg, fontsize=11.0, fontweight="bold", pad=10)
    ax0.legend(facecolor=bg, edgecolor=grid_c, labelcolor=fg, fontsize=8.2, loc="lower left")

    # --- Panel 2: Vertical 3D Mean R² (Water / Steam / Smoke) ---
    ax1 = axes[1]
    v7_v_plot = [max(0.0, val) for val in v7_v]
    b7_v = ax1.bar(x - w / 2, v7_v_plot, w, color=c_v7, alpha=0.9, label="v7 (Law-Blind)")
    b8_v = ax1.bar(x + w / 2, v8_v, w, color=c_v8_v, label="v8 (Law-Conditioned)")
    ax1.axhline(0.90, color="#d29922", linestyle="--", linewidth=1.2, label="Target ≥ 0.90")
    for b, raw_v in zip(b7_v, v7_v):
        if raw_v < 0.0:
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                0.04,
                f"{raw_v:.2f}\n(collapse)",
                ha="center",
                va="bottom",
                color="#ff7b72",
                fontsize=7.8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#0d1117", edgecolor="#f85149", lw=0.9),
            )
        else:
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                min(raw_v + 0.025, 1.07),
                f"{raw_v:.3f}",
                ha="center",
                va="bottom",
                color="#ff7b72",
                fontsize=8.2,
                fontweight="bold",
            )
    for b, val in zip(b8_v, v8_v):
        ax1.text(
            b.get_x() + b.get_width() / 2.0,
            min(val + 0.025, 1.07),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color="#79c0ff",
            fontsize=8.4,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, color=fg, fontsize=8.4)
    ax1.set_ylabel("Vertical 3D Interface Mean R²", color=fg, fontsize=9.5)
    ax1.set_ylim(0.0, 1.15)
    ax1.set_title("Vertical 3D Gravity & Buoyancy R²", color=fg, fontsize=11.0, fontweight="bold", pad=10)
    ax1.legend(facecolor=bg, edgecolor=grid_c, labelcolor=fg, fontsize=8.2, loc="lower left")

    # --- Panel 3: Conservation Firewall Clamped % (Lower = Better) ---
    ax2 = axes[2]
    b7_fw = ax2.bar(x - w / 2, v7_fw, w, color=c_v7, alpha=0.9, label="v7 (Law-Blind)")
    b8_fw = ax2.bar(x + w / 2, v8_fw, w, color=c_v8_fw, label="v8 (Law-Conditioned)")
    ax2.axhline(3.5, color="#d29922", linestyle="--", linewidth=1.2, label="Regime Target < 3.5%")
    ymax_fw = max(max(v7_fw + v8_fw) * 1.24, 6.0)
    for b, val in zip(b7_fw, v7_fw):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + ymax_fw * 0.022,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            color="#ff7b72",
            fontsize=8.2,
            fontweight="bold",
        )
    for b, val in zip(b8_fw, v8_fw):
        ax2.text(
            b.get_x() + b.get_width() / 2.0,
            val + ymax_fw * 0.022,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            color="#56d4dd",
            fontsize=8.4,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, color=fg, fontsize=8.4)
    ax2.set_ylabel("Firewall Clamped Voxels (%)", color=fg, fontsize=9.5)
    ax2.set_ylim(0.0, ymax_fw)
    ax2.set_title("Conservation Firewall Clamped % (Lower = Better)", color=fg, fontsize=11.0, fontweight="bold", pad=10)
    ax2.legend(facecolor=bg, edgecolor=grid_c, labelcolor=fg, fontsize=8.2, loc="upper left")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=150, facecolor=bg)
    plt.close(fig)
    print(f"[Viz] Saved {out_png}", flush=True)


# =============================================================================
# Main Training & Evaluation Entrypoint
# =============================================================================


def _load_v7_ref(path: Path, device: torch.device) -> WorldModel3DV5 | None:
    if not path.exists():
        return None
    m = WorldModel3DV5(in_ch=18, out_ch=17, width=48, z_flux_ch=5).to(device)
    ck = torch.load(str(path), map_location=device, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    m.load_state_dict(sd, strict=True)
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Frostmarsh Wave 20A: WorldModel3D-v8 Training & Eval")
    ap.add_argument("--init-v7", type=Path, default=Path("models/checkpoints/world_model_3d_v7.pt"))
    ap.add_argument("--out-ckpt", type=Path, default=Path("models/checkpoints/world_model_3d_v8.pt"))
    ap.add_argument("--out-json", type=Path, default=Path("models/WORLD_3D_V8_METRICS.json"))
    ap.add_argument("--out-png", type=Path, default=Path("models/world_3d_v8_viz.png"))
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--lr", type=float, default=9.0e-4)
    ap.add_argument("--law-lr-mult", type=float, default=3.5)
    ap.add_argument("--fw-weight", type=float, default=1.05)
    ap.add_argument("--vert-weight", type=float, default=1.50)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=16)
    args = ap.parse_args()

    if args.smoke_test:
        args.steps = int(args.smoke_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(
        f"[WorldModel3D-v8] Device={device} | steps={args.steps} | lr={args.lr} (law x{args.law_lr_mult}) "
        f"| fw_w={args.fw_weight} | init_v7={args.init_v7}",
        flush=True,
    )

    # 1. Verify canonical teacher identity & Step-0 v7->v8 bit-for-bit warm-start identity!
    teacher_check = verify_canonical_teacher_identity(L=16, H=32, W=32)
    print(
        f"[Check] Canonical teacher identity max_diff="
        f"{max(teacher_check['canonical_diff_horiz'], teacher_check['canonical_diff_z_flux']):.2e} (PASS)",
        flush=True,
    )

    v7_ref = _load_v7_ref(args.init_v7, device)
    warmstart_check: dict[str, float] = {}
    if args.init_v7.exists():
        warmstart_check = verify_v7_warmstart_identity(args.init_v7, L=16, H=32, W=32)
        print(f"[Check] Step-0 v7->v8 warm-start identity: {warmstart_check} (PASS)", flush=True)
        model = WorldModel3DV8.load_from_v7(args.init_v7, map_location=device)
        print(
            f"[WorldModel3D-v8] Warm-started from {args.init_v7} ({count_parameters(model):,} params)",
            flush=True,
        )
    else:
        model = WorldModel3DV8().to(device)
        print(f"[WorldModel3D-v8] Fresh init ({count_parameters(model):,} params)", flush=True)

    # 2. Optimizer with differential LR for new World-Law conditioning modules
    opt = torch.optim.AdamW(
        [
            {"params": model.base_parameters(), "lr": args.lr, "weight_decay": 1e-4},
            {"params": model.law_parameters(), "lr": args.lr * args.law_lr_mult, "weight_decay": 1e-4},
        ]
    )
    base_warmup = min(160, max(1, args.steps // 12))
    law_warmup = min(50, max(1, args.steps // 30))

    def _base_lr_lambda(s: int) -> float:
        if s < base_warmup:
            return 0.05 + 0.95 * float(s + 1) / float(base_warmup)
        prog = float(s - base_warmup) / float(max(1, args.steps - base_warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    def _law_lr_lambda(s: int) -> float:
        if s < law_warmup:
            return 0.25 + 0.75 * float(s + 1) / float(law_warmup)
        prog = float(s - law_warmup) / float(max(1, args.steps - law_warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [_base_lr_lambda, _law_lr_lambda])
    # bfloat16 on A100/G4/L4; GradScaler strictly disabled
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    ema_model = copy.deepcopy(model).eval()
    ema_decay = 0.992

    crop_xy = 32 if args.smoke_test else 64
    pool: list[SimState] = []
    pool_laws: list[tuple[float, float, float, float, float, float]] = []
    gen = torch.Generator(device=device)
    gen.manual_seed(202620)

    def _refill_pool(seed_base: int) -> None:
        pool.clear()
        pool_laws.clear()
        for bid in range(len(BIOME_NAMES)):
            L_w = 16 if (bid % 2 == 0 or args.smoke_test) else 32
            st = make_biome_3d_world(bid, L=L_w, H=crop_xy, W=crop_xy, device=device, seed=seed_base + bid * 19)
            laws_slot = DEFAULT_WORLD_LAWS if bid == 0 else sample_world_laws(gen, device, force_non_canonical=True)
            pool.append(st)
            pool_laws.append(laws_slot)

    _refill_pool(2000)
    best_score = -1e9
    best_sd: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []

    model.train()
    for step in range(1, args.steps + 1):
        if step % 35 == 0:
            _refill_pool(2000 + step * 13)
        idx = (step - 1) % len(pool)
        st = pool[idx]
        L_cur, _, H_cur, W_cur = st.mass.shape

        # Mix persistent regime evolution (60%) and fresh continuous law sampling (40%)
        if float(torch.rand(1, generator=gen, device=device).item()) < 0.60:
            laws = pool_laws[idx]
        else:
            laws = sample_world_laws(gen, device)

        # Mix 65% layer-wise bolts, 25% 3D localized spells, 10% unforced evolution
        u_f = float(torch.rand(1, generator=gen, device=device).item())
        if u_f < 0.65:
            fh, fw = sample_bolts(L_cur, H_cur, W_cur, device)
        elif u_f < 0.90:
            fh, fw = sample_3d_forcing(L_cur, H_cur, W_cur, device, prob=0.75)
        else:
            fh = torch.zeros((L_cur, 1, H_cur, W_cur), dtype=torch.float32, device=device)
            fw = torch.zeros((L_cur, 1, H_cur, W_cur), dtype=torch.float32, device=device)

        # Continuous local_dt: 56% exact 1.0, 24% uniform [0.25, 3.2], 20% spatial stasis/haste dome
        r_dt = float(torch.rand(1, generator=gen, device=device).item())
        if args.smoke_test or r_dt < 0.56:
            dt_val: Any = 1.0
            dt_t = torch.ones((L_cur, 1, H_cur, W_cur), device=device)
        elif r_dt < 0.80:
            dt_val = float(torch.empty(1, device=device).uniform_(0.25, 3.2, generator=gen).item())
            dt_t = torch.full((L_cur, 1, H_cur, W_cur), dt_val, device=device)
        else:
            dt_2d = torch.ones((H_cur, W_cur), device=device)
            dt_2d[H_cur // 4 : (3 * H_cur) // 4, W_cur // 4 : (3 * W_cur) // 4] = 0.15 if (step % 2 == 0) else 2.5
            dt_t = dt_2d.view(1, 1, H_cur, W_cur).expand(L_cur, 1, H_cur, W_cur).clone()
            dt_val = dt_2d

        with torch.no_grad():
            x_in = encode_state(st, fh, fw)
            next_st, tgt_h, tgt_z = step_refsim_3d_with_laws(st, fh, fw, local_dt=dt_val, world_laws=laws)
            mass_eff, energy_eff = apply_forcing(st, fh, fw)
            pool[idx] = next_st

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            pred_h, pred_z, _ = model(x_in, local_dt=dt_t, world_laws=laws, return_z_flux=True, return_latent=True)

        pred_h_f = pred_h.float()
        pred_z_f = pred_z.float()
        proj_h, _ = project_deltas_with_laws(mass_eff, energy_eff, pred_h_f, world_laws=laws)

        l_horiz, h_stats = v8_active_face_horiz_loss(pred_h_f, tgt_h, dt_t)
        l_vert, _ = v8_vertical_teacher_loss_3d(pred_z_f, tgt_z, st, dt_t)
        l_fw, _ = v8_smooth_firewall_loss_3d(mass_eff, energy_eff, pred_h_f, proj_h, dt_t, world_laws=laws)

        prog_s = float(step) / float(max(1, args.steps))
        fw_mult = args.fw_weight * (0.75 + 0.70 * min(1.0, prog_s * 1.25))
        loss = l_horiz + args.vert_weight * l_vert + fw_mult * l_fw

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = float(nn.utils.clip_grad_norm_(model.parameters(), 2.5).item())
        scaler.step(opt)
        scaler.update()
        sched.step()

        with torch.no_grad():
            d_eff = min(ema_decay, float(step) / float(step + 15))
            for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                p_ema.mul_(d_eff).add_(p_cur, alpha=1.0 - d_eff)

        log_every = max(args.steps // 12, 1)
        if step % log_every == 0 or step == args.steps:
            print(
                f"[Step {step:4d}/{args.steps}] loss={loss.item():.3f} | horiz={l_horiz.item():.3f} "
                f"(1-r2 f={h_stats['flux_1mr2']:.3f} h={h_stats['heat_1mr2']:.3f} r={h_stats['rxn_1mr2']:.3f}) "
                f"| vert={l_vert.item():.3f} | fw={l_fw.item():.3f} | gn={gn:.2f} | elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

        # Periodic 4-regime validation on unseen 64x64x16 worlds to select best EMA checkpoint
        if (not args.smoke_test) and (step % 250 == 0 or step == args.steps):
            val_m = evaluate_models_world_laws({"ema": ema_model, "raw": model}, device=device, L=16, H=64, W=64)
            for tag in ("ema", "raw"):
                sc = 0.0
                for p_idx, pname in enumerate(WORLD_LAW_PRESET_NAMES):
                    vm = val_m[pname][tag]
                    w_p = 1.35 if p_idx == 0 else 1.0
                    cap_fw = 2.6 if p_idx == 0 else 3.2
                    fw_pen = max(0.0, vm["proj_active_pct"] - cap_fw) * 0.40
                    sc += w_p * (
                        vm["horiz_flux_r2"]
                        + vm["horiz_heat_r2"]
                        + vm["horiz_rxn_r2"]
                        + 0.7 * (vm["vert_water_r2"] + vm["vert_steam_r2"] + vm["vert_smoke_r2"])
                        - fw_pen
                    )
                if sc > best_score:
                    best_score = sc
                    best_sd = copy.deepcopy((ema_model if tag == "ema" else model).state_dict())

            row: dict[str, Any] = {"step": step, "elapsed_s": round(time.time() - t0, 1)}
            parts = []
            for pname in WORLD_LAW_PRESET_NAMES:
                ve = val_m[pname]["ema"]
                row[pname] = {
                    "horiz_mean_r2": ve["horiz_mean_r2"],
                    "vert_mean_r2": ve["vert_mean_r2"],
                    "overall_mean_r2": ve["overall_mean_r2"],
                    "proj_active_pct": ve["proj_active_pct"],
                }
                parts.append(
                    f"{pname[:6]}:H={ve['horiz_mean_r2']:.3f}/V={ve['vert_mean_r2']:.3f}/FW={ve['proj_active_pct']:.2f}%"
                )
            history.append(row)
            print(f"  [Val@step {step:4d} EMA] " + " | ".join(parts) + f" (best={best_score:.3f})", flush=True)
            model.train()

    if best_sd is not None:
        model.load_state_dict(best_sd, strict=True)
    model.eval()

    # Final Head-to-Head Held-Out Evaluation (v7 vs v8 across ALL 4 World-Law Presets!)
    eval_models: dict[str, nn.Module] = {"world_model_3d_v8": model}
    if v7_ref is not None:
        eval_models["world_model_3d_v7"] = v7_ref

    eval_z16 = evaluate_models_world_laws(
        eval_models,
        device=device,
        L=16,
        H=32 if args.smoke_test else 96,
        W=32 if args.smoke_test else 96,
    )
    eval_z32 = evaluate_models_world_laws(
        eval_models,
        device=device,
        L=32,
        H=32 if args.smoke_test else 64,
        W=32 if args.smoke_test else 64,
    )

    # FP32 16x16x8 Sparse-vs-Dense Parity Checks (< 1e-6 on x86/Colab; < 5e-6 on ARM64 NEON)
    parity_tol = 5e-6 if args.smoke_test else 1e-6
    parity_checks: dict[str, Any] = {}
    for pname in ("frostmarsh_standard", "rising_flood_canyon", "low_g_magma_crucible"):
        for layers in (16, 32):
            key = f"{pname}_z{layers}"
            res = verify_sparse_v8_parity(model, L=layers, H=64, W=64, world_laws=WORLD_LAW_PRESETS[pname])
            parity_checks[key] = res
            print(f"[Parity {key}] max_abs_diff_all={res['max_abs_diff_all']:.2e} (frac={res['active_brick_frac']})", flush=True)
            assert res["max_abs_diff_all"] < parity_tol, f"Sparse parity failed on {key}: {res}"

    print("\n=== Head-to-Head Held-Out Summary (Z=16) ===", flush=True)
    for pname in WORLD_LAW_PRESET_NAMES:
        for tag in ("world_model_3d_v7", "world_model_3d_v8"):
            if tag in eval_z16[pname]:
                s = eval_z16[pname][tag]
                print(
                    f"  [{pname:20s}] {tag}: hFlux={s['horiz_flux_r2']:.4f} hHeat={s['horiz_heat_r2']:.4f} "
                    f"hRxn={s['horiz_rxn_r2']:.4f} (hMean={s['horiz_mean_r2']:.4f}) | "
                    f"vW={s['vert_water_r2']:.4f} vSt={s['vert_steam_r2']:.4f} vSm={s['vert_smoke_r2']:.4f} "
                    f"(vMean={s['vert_mean_r2']:.4f}) | all={s['overall_mean_r2']:.4f} | fw={s['proj_active_pct']:.2f}%",
                    flush=True,
                )

    print("\n=== Head-to-Head Held-Out Summary (Z=32) ===", flush=True)
    for pname in WORLD_LAW_PRESET_NAMES:
        for tag in ("world_model_3d_v7", "world_model_3d_v8"):
            if tag in eval_z32[pname]:
                s = eval_z32[pname][tag]
                print(
                    f"  [{pname:20s}] {tag}: hMean={s['horiz_mean_r2']:.4f} vMean={s['vert_mean_r2']:.4f} "
                    f"all={s['overall_mean_r2']:.4f} | fw={s['proj_active_pct']:.2f}%",
                    flush=True,
                )

    if not args.smoke_test:
        args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "arch": "WorldModel3DV8",
                "version": "v8-law-conditioned-universal-3d",
                "params": count_parameters(model),
                "law_dim": LAW_DIM,
                "default_world_laws": list(DEFAULT_WORLD_LAWS),
                "world_law_presets": {k: list(v) for k, v in WORLD_LAW_PRESETS.items()},
            },
            str(args.out_ckpt),
        )
        metrics = {
            "wave": "20A",
            "model_class": "WorldModel3DV8 (Law-Conditioned Universal 3D Physics World Model)",
            "params": count_parameters(model),
            "base_v7_params": count_parameters(WorldModel3DV5()),
            "extra_law_params": count_parameters(model) - count_parameters(WorldModel3DV5()),
            "steps": args.steps,
            "train_seconds": round(time.time() - t0, 1),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "law_names": ["gravity_g", "ambient_temp_shift", "volatility", "viscosity", "precipitation", "thermal_cond"],
            "default_world_laws": list(DEFAULT_WORLD_LAWS),
            "world_law_presets": {k: list(v) for k, v in WORLD_LAW_PRESETS.items()},
            "canonical_teacher_identity": teacher_check,
            "v7_step0_warmstart_identity": warmstart_check,
            "eval_z16": eval_z16,
            "eval_z32": eval_z32,
            "fp32_sparse_parity": parity_checks,
            "history": history,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(metrics, indent=2))
        save_v8_viz(metrics, args.out_png)
        print(f"[Saved] {args.out_ckpt}, {args.out_json}, {args.out_png}", flush=True)
    else:
        print(f"[WorldModel3D-v8 Smoke Test] PASS in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
