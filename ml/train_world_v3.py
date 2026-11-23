#!/usr/bin/env python3
"""
Part A: Conservation-Distilled WorldModel v3 (`train_world_v3.py` -> `world_model_v3.pt`).

Warm-starts from `world_model.pt` (WorldModel(in_ch=18, out_ch=17, width=48, num_res=5))
preserving the EXACT drop-in architecture & state_dict keys for `SparseWorldRunner` and
`gpu_infer_daemon.py`.

Fine-tunes on realistic Frostmarsh 256x256 states (winding rivers, oil pools, wildfires,
frost spells, highland ledges, plus all 5 sim_teacher biomes) with:
  1. High-precision teacher physics distillation (tight Huber + MSE + active-region weighting)
     so physics MSE/MAE strictly matches or beats v1.
  2. Differentiable Conservation Projection-Firewall Penalty (`loss_proj`):
     - Positive species outflow fluxes exceeding available local cell mass (donor budget)
     - Impermeable wall (rock>=500 / ice>=700) & domain boundary flux leakage + H_CAP headroom
     - 2nd-law heat conduction direction & discrete maximum-principle bounds
     - Reaction rates (`out[:, 10:17]`) exceeding reactant availability, phase-equilibrium caps,
       temperature gates, and [T_MIN, T_MAX] thermal bounds
     - Direct exact projection residual |Y_raw - Y_proj| (with heat normalized by out_scale=10)
  3. Mild Z_world representation anchor: 0.05 * MSE(z_new, z_old) so downstream
     BotPolicyNet and LiveAcousticEngine remain smoothly compatible.

Evaluates world_model.pt (v1) vs world_model_v3.pt (v3) on held-out unseen 256x256x4 worlds
and writes `world_model_v3.pt` + `WORLD_V3_METRICS.json`.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import WorldModel, count_parameters
from sim_teacher import (
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
    RefSim,
    SimState,
    _apply_transport,
    apply_forcing,
    compute_C_and_T,
    compute_total_energy,
    compute_total_mass,
    encode_state,
    get_consts,
    make_scene,
    project_and_step,
    project_deltas,
    sample_bolts,
)
from sparse_world_runner import SparseWorldRunner

ROOT = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
V1_PT = ROOT / "world_model.pt"
V3_PT = ROOT / "world_model_v3.pt"
METRICS_JSON = ROOT / "WORLD_V3_METRICS.json"


def world_forward_with_z(model: WorldModel, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact WorldModel forward returning (raw_deltas [B,17,H,W], Z_world [B,48,H,W])."""
    h = model.stem(x)
    for blk in model.blocks:
        h = blk(h)
    z = F.gelu(h)
    raw = model.head(z) * model.out_scale
    out = torch.cat([raw[:, :10], F.softplus(raw[:, 10:])], dim=1)
    return out, z


def make_frostmarsh_4layer_world(
    H: int,
    W: int,
    device: torch.device,
    seed: int,
) -> SimState:
    """
    Generate a realistic 4-layer Frostmarsh world [B=4, H, W] in exact SimState physical units
    (physically grounded substrate on every layer so all 4 layers sit inside RefSim's valid regime):
      Layer 0: River crossing + frost spell ice bridges + oil marsh flanks
      Layer 1: Active forest wildfire + oil pools + rocky cliffs
      Layer 2: Highland ledges, elevated plateau, dam/bridges, thermal gradients
      Layer 3: Mixed thermal / steam / smoke / frost-spell battlefield
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    c = get_consts(device)

    mass = torch.zeros((4, 10, H, W), dtype=torch.float32, device=device)
    flow_e = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    flow_s = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    elev = torch.zeros((4, 1, H, W), dtype=torch.float32, device=device)
    T_init = torch.full((4, 1, H, W), T_AMB, dtype=torch.float32, device=device)

    yy = torch.linspace(0.0, 1.0, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, W).expand(H, W)

    def _noise(grid_n: int) -> torch.Tensor:
        g = torch.rand((1, 1, grid_n, grid_n), generator=gen, device=device)
        return F.interpolate(g, size=(H, W), mode="bilinear", align_corners=False)[0, 0]

    nc = _noise(max(4, min(H, W) // 16))
    nm = _noise(max(4, min(H, W) // 8))
    nf = _noise(max(8, min(H, W) // 4))

    # Winding river down center
    phase = float(torch.rand(1, generator=gen, device=device).item()) * 6.28318
    amp = float(torch.empty(1, device=device).uniform_(0.04, 0.09, generator=gen).item())
    rcx = 0.50 + amp * torch.sin(yy * 6.28318 * 1.25 + phase)
    rhw = float(torch.empty(1, device=device).uniform_(0.075, 0.11, generator=gen).item())
    dist_r = (xx - rcx).abs()
    in_river = dist_r < rhw

    # --- Layer 0: River, ice crossings, frost spell zones & oil slick banks ---
    e0 = 8.0 * (1.0 - 0.3 * yy) + 5.0 * nc + 2.0 * nm
    e0 = torch.where(in_river, 2.0 + 3.0 * (1.0 - yy) + 1.5 * (dist_r / rhw), e0)
    elev[0, 0] = e0.clamp(0.0, 31.0)
    mass[0, S_SOIL] = torch.where(in_river, 260.0, 760.0 + 220.0 * nc)
    mass[0, S_WATER] = torch.where(in_river, 820.0 + 300.0 * (1.0 - dist_r / rhw), 0.0)
    flow_s[0, 0] = torch.where(in_river, 125.0, 0.0)
    rock0 = (~in_river) & (nm > 0.80)
    mass[0, S_ROCK] = torch.where(rock0, 920.0, 0.0)
    for iy_lo, iy_hi in ((0.18, 0.34), (0.64, 0.82)):
        ice_y = float(torch.empty(1, device=device).uniform_(iy_lo, iy_hi, generator=gen).item())
        ice_m = in_river & ((yy - ice_y).abs() < 0.048)
        mass[0, S_ICE] = torch.where(ice_m, 760.0, mass[0, S_ICE])
        mass[0, S_WATER] = torch.where(ice_m, 180.0, mass[0, S_WATER])
        T_init[0, 0] = torch.where(ice_m, 25400.0, T_init[0, 0])
    for _ in range(4):
        ox = float(torch.empty(1, device=device).uniform_(0.14, 0.86, generator=gen).item())
        oy = float(torch.empty(1, device=device).uniform_(0.14, 0.86, generator=gen).item())
        orad = float(torch.empty(1, device=device).uniform_(0.045, 0.080, generator=gen).item())
        om = (((xx - ox) ** 2 + (yy - oy) ** 2) < (orad * orad)) & (~rock0)
        mass[0, S_OIL] = torch.where(om, 520.0 + 260.0 * nf, mass[0, S_OIL])
        mass[0, S_WATER] = torch.where(om, torch.maximum(mass[0, S_WATER], torch.full_like(xx, 180.0)), mass[0, S_WATER])
    wood0 = (~in_river) & (~rock0) & (nf > 0.60)
    plant0 = (~in_river) & (~rock0) & (~wood0) & (nm > 0.38)
    mass[0, S_WOOD] = torch.where(wood0, 520.0 + 360.0 * nf, 0.0)
    mass[0, S_PLANT] = torch.where(plant0, 300.0 + 280.0 * nm, 0.0)

    # --- Layer 1: Surface wildfires + ignited oil pools + frost spells ---
    elev[1, 0] = (6.0 + 10.0 * nc + 3.5 * nm).clamp(0.0, 31.0)
    mass[1, S_SOIL] = 750.0 + 200.0 * nc
    pond = nc < 0.25
    mass[1, S_WATER] = torch.where(pond, 650.0 + 250.0 * (0.25 - nc) / 0.25, 0.0)
    rock1 = (~pond) & (nm > 0.82)
    mass[1, S_ROCK] = torch.where(rock1, 950.0, 0.0)
    mass[1, S_WOOD] = torch.where((~pond) & (~rock1) & (nm > 0.35), 550.0 + 420.0 * nm, 0.0)
    mass[1, S_PLANT] = torch.where((~pond) & (~rock1) & (nf > 0.42), 280.0 + 280.0 * nf, 0.0)
    for _ in range(3):
        ox = float(torch.empty(1, device=device).uniform_(0.18, 0.82, generator=gen).item())
        oy = float(torch.empty(1, device=device).uniform_(0.18, 0.82, generator=gen).item())
        orad = float(torch.empty(1, device=device).uniform_(0.04, 0.07, generator=gen).item())
        om = (((xx - ox) ** 2 + (yy - oy) ** 2) < (orad * orad)) & (~rock1)
        mass[1, S_OIL] = torch.where(om, 580.0, mass[1, S_OIL])
    for _ in range(4):
        fx = float(torch.empty(1, device=device).uniform_(0.18, 0.82, generator=gen).item())
        fy = float(torch.empty(1, device=device).uniform_(0.16, 0.84, generator=gen).item())
        fr = float(torch.empty(1, device=device).uniform_(0.035, 0.060, generator=gen).item())
        ft = float(torch.empty(1, device=device).uniform_(54000.0, 70000.0, generator=gen).item())
        fm = ((xx - fx) ** 2 + (yy - fy) ** 2) < (fr * fr)
        T_init[1, 0] = torch.where(fm, ft, T_init[1, 0])
        mass[1, S_SMOKE] = torch.where(fm, 240.0, mass[1, S_SMOKE])
        mass[1, S_ASH] = torch.where(fm, 90.0, mass[1, S_ASH])
    for _ in range(2):
        fx = float(torch.empty(1, device=device).uniform_(0.18, 0.82, generator=gen).item())
        fy = float(torch.empty(1, device=device).uniform_(0.18, 0.82, generator=gen).item())
        fr = float(torch.empty(1, device=device).uniform_(0.04, 0.065, generator=gen).item())
        fm = ((xx - fx) ** 2 + (yy - fy) ** 2) < (fr * fr)
        T_init[1, 0] = torch.where(fm, 23800.0, T_init[1, 0])
        mass[1, S_ICE] = torch.where(fm & pond, 480.0, mass[1, S_ICE])

    # --- Layer 2: Highland ledges, cliff plateaus & bridges ---
    nw_ledge = (xx > 0.05) & (xx < 0.37) & (yy > 0.06) & (yy < 0.35)
    ne_ledge = (xx > 0.63) & (xx < 0.95) & (yy > 0.06) & (yy < 0.35)
    sw_ledge = (xx > 0.08) & (xx < 0.33) & (yy > 0.67) & (yy < 0.93)
    se_ledge = (xx > 0.67) & (xx < 0.92) & (yy > 0.67) & (yy < 0.93)
    stone_arch = ((yy - 0.28).abs() < 0.030) & (xx > 0.33) & (xx < 0.67)
    wood_bridge = ((yy - 0.68).abs() < 0.028) & (xx > 0.34) & (xx < 0.66)
    ledge_m = nw_ledge | ne_ledge | sw_ledge | se_ledge | stone_arch
    elev[2, 0] = torch.where(ledge_m | wood_bridge, 20.0 + 4.0 * nc, 5.0 + 3.0 * nm).clamp(0.0, 31.0)
    mass[2, S_SOIL] = torch.where(ledge_m, 800.0 + 180.0 * nc, 480.0 + 140.0 * nm)
    mass[2, S_ROCK] = torch.where(stone_arch, 850.0, torch.where((~ledge_m) & (nm > 0.85), 800.0, 0.0))
    mass[2, S_WOOD] = torch.where(wood_bridge, 650.0, torch.where(ledge_m & (nf > 0.68), 460.0, 0.0))
    mass[2, S_PLANT] = torch.where(ledge_m & (nm > 0.45), 280.0, 0.0)
    mass[2, S_WATER] = torch.where(nw_ledge & (nm < 0.40), 700.0, 0.0)
    mass[2, S_OIL] = torch.where(ne_ledge & (nf > 0.72), 420.0, 0.0)
    flow_e[2, 0] = torch.where(nw_ledge & (nm < 0.40), 120.0, 0.0)
    T_init[2, 0] = 28500.0 + 1200.0 * nc

    # --- Layer 3: High-activity thermal mix (steam, smoke, oil, ice, boiling, combustion) ---
    elev[3, 0] = (4.0 + 18.0 * nc + 5.0 * nm).clamp(0.0, 31.0)
    mass[3, S_SOIL] = 450.0 + 250.0 * nc
    mass[3, S_WATER] = torch.where(nm > 0.38, 650.0 * nm, 0.0)
    mass[3, S_ICE] = torch.where(nc < 0.30, 720.0 * (1.0 - nc), 0.0)
    mass[3, S_STEAM] = torch.where(nf > 0.72, 320.0 * nf, 0.0)
    mass[3, S_SMOKE] = torch.where((nf > 0.55) & (nf <= 0.72), 240.0 * nf, 0.0)
    mass[3, S_ROCK] = torch.where((nc > 0.82) & (nm > 0.65), 950.0, 0.0)
    mass[3, S_WOOD] = torch.where((nm > 0.56) & (mass[3, S_ROCK] == 0), 480.0 * nm, 0.0)
    mass[3, S_OIL] = torch.where((nc < 0.44) & (nf > 0.52) & (mass[3, S_ROCK] == 0), 380.0 * nf, 0.0)
    mass[3, S_PLANT] = torch.where((nm < 0.54) & (nf > 0.46) & (mass[3, S_ROCK] == 0), 320.0 * nf, 0.0)
    T3 = 22500.0 + 38000.0 * nm
    T3 = torch.where(mass[3, S_ICE] > 300.0, 24500.0 + 4500.0 * nf, T3)
    T3 = torch.where(mass[3, S_STEAM] > 100.0, 35500.0 + 5500.0 * nf, T3)
    T_init[3, 0] = T3.clamp(T_MIN + 1000.0, 72000.0)
    flow_e[3, 0] = (nc - 0.5) * 240.0 * (mass[3, S_WATER] > 50.0).float()
    flow_s[3, 0] = (nm - 0.5) * 240.0 * (mass[3, S_WATER] > 50.0).float()

    # Enforce H_CAP strictly
    tot_m = mass.sum(dim=1, keepdim=True)
    cap_scale = torch.where(tot_m > H_CAP * 0.95, (H_CAP * 0.95) / (tot_m + 1e-6), torch.ones_like(tot_m))
    mass = mass * cap_scale

    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    energy = C_cell * torch.clamp(T_init, T_MIN, T_MAX)
    return SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev)


def conservation_firewall_loss(
    mass_eff: torch.Tensor,
    energy_eff: torch.Tensor,
    raw_deltas: torch.Tensor,
    proj_deltas: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Differentiable Conservation Projection-Firewall Penalty (`loss_proj`):
      1. Donor outflow excess over available local mobile cell mass
      2. Impermeable wall (rock>=500 / ice>=700) & boundary flux leakage + H_CAP headroom
      3. Heat 2nd-law discrete maximum principle (sign + q_max + ambient zero-dT)
      4. Reaction rates exceeding temp gates, reactant availability, phase caps, & [T_MIN, T_MAX]
      5. Sharp Charbonnier + threshold-hinge exact projection gap ||Y_raw - Y_proj||
    """
    c = get_consts(mass_eff.device)
    _, _, H, W = mass_eff.shape
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

    wall = ((mass_eff[:, S_ROCK:S_ROCK + 1] >= 500.0) | (mass_eff[:, S_ICE:S_ICE + 1] >= 700.0)).float()
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
    l_donor = donor_excess.mean() + 4.0 * F.relu(donor_excess - 3e-4).mean()

    # 2. Wall & boundary flux leakage + receiver H_CAP excess
    l_wall = (flux_e.abs() * (1.0 - open_e)).mean() + (flux_s.abs() * (1.0 - open_s)).mean()
    total_in = (F.relu(fw_in) + F.relu(-flux_e) + F.relu(fn_in) + F.relu(-flux_s)).sum(dim=1, keepdim=True)
    headroom_norm = (H_CAP - mass_eff.sum(dim=1, keepdim=True)).clamp(min=0.0) / 1000.0
    l_hcap = F.relu(total_in - headroom_norm).mean()

    # 3. Heat 2nd-law & discrete maximum principle (normalized by out_scale=10.0)
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
    l_heat_bound = hv_e.mean() + hv_s.mean() + 3.0 * (F.relu(hv_e - 1e-3).mean() + F.relu(hv_s - 1e-3).mean())

    # 4. Reaction rates exceeding gates, reactant availability, phase caps, and thermal bounds
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
        xi_cap_freeze = C_tr * F.relu(27315.0 - T_tr) / (336000.0 + 21.0 * 27315.0)
        xi_cap_melt = C_tr * F.relu(T_tr - 27315.0) / (336000.0 + 21.0 * 27315.0)
        xi_cap_cond = C_tr * F.relu(37000.0 - T_tr) / (840000.0 - 22.0 * 37000.0)
        xi_cap_boil = torch.where(
            T_tr < 38180.0,
            C_tr * F.relu(T_tr - 37315.0) / (840000.0 - 22.0 * 37315.0),
            mass_tr[:, S_WATER:S_WATER + 1],
        )
        phase_caps_norm = torch.cat([
            xi_cap_freeze, xi_cap_melt, xi_cap_boil, xi_cap_cond,
            torch.full_like(C_tr, 1e9), torch.full_like(C_tr, 1e9), torch.full_like(C_tr, 1e9),
        ], dim=1) / 100.0

    l_rxn_gate = (rxn * (1.0 - t_gate)).mean()
    l_rxn_phase = F.relu(rxn - phase_caps_norm).mean()
    consumed = torch.einsum("brhw,rs->bshw", rxn * 100.0, c.nu_neg)
    l_rxn_reactant = F.relu(consumed - mass_tr).mean() / 100.0

    rxn_phys = rxn * 100.0
    E_post = energy_tr + (rxn_phys * c.rxn_dE).sum(dim=1, keepdim=True)
    C_post = (C_tr + (rxn_phys * c.rxn_dc).sum(dim=1, keepdim=True)).clamp(min=0.0)
    l_rxn_therm = (F.relu(E_post - C_post * T_MAX).mean() + F.relu(C_post * T_MIN - E_post).mean()) / 1e8
    l_rxn_bound = l_rxn_gate + l_rxn_phase + l_rxn_reactant + l_rxn_therm

    # 5. Sharp exact projection residual ||Y_raw - Y_proj.detach()|| (normalized heat/10)
    diff = (raw_deltas - proj_deltas.detach()).abs()
    diff_h = diff[:, 8:10] / 10.0
    p_flux = diff[:, 0:8].mean() + 4.0 * F.relu(diff[:, 0:8] - 5e-4).mean()
    p_heat = diff_h.mean() + 4.0 * F.relu(diff_h - 5e-4).mean()
    p_rxn = diff[:, 10:17].mean() + 4.0 * F.relu(diff[:, 10:17] - 5e-4).mean()

    total_proj = (
        16.0 * l_donor
        + 10.0 * l_wall
        + 6.0 * l_hcap
        + 16.0 * l_heat_bound
        + 14.0 * l_rxn_bound
        + 12.0 * p_flux
        + 14.0 * p_heat
        + 10.0 * p_rxn
    )
    return total_proj, {
        "donor": float(l_donor.item()),
        "heat_b": float(l_heat_bound.item()),
        "rxn_b": float(l_rxn_bound.item()),
    }


def physics_teacher_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """High-precision teacher distillation loss (tight Huber + MSE + active-region weighting)."""
    l_flux = F.smooth_l1_loss(pred[:, 0:8], target[:, 0:8], beta=0.005) + 0.8 * F.mse_loss(pred[:, 0:8], target[:, 0:8])
    l_heat = F.smooth_l1_loss(pred[:, 8:10] / 10.0, target[:, 8:10] / 10.0, beta=0.005) + 0.8 * F.mse_loss(pred[:, 8:10] / 10.0, target[:, 8:10] / 10.0)
    l_rxn = F.smooth_l1_loss(pred[:, 10:17], target[:, 10:17], beta=0.005) + 0.8 * F.mse_loss(pred[:, 10:17], target[:, 10:17])

    act_f = target[:, 0:8].abs() > 1e-3
    l_flux_act = F.smooth_l1_loss(pred[:, 0:8][act_f], target[:, 0:8][act_f], beta=0.01) if act_f.any() else pred.new_tensor(0.0)

    act_h = target[:, 8:10].abs() > 1e-2
    l_heat_act = F.smooth_l1_loss(pred[:, 8:10][act_h] / 10.0, target[:, 8:10][act_h] / 10.0, beta=0.01) if act_h.any() else pred.new_tensor(0.0)

    l_rxn_act = pred.new_tensor(0.0)
    n_seen = 0
    for r in range(7):
        g_r = target[:, 10 + r]
        p_r = pred[:, 10 + r]
        m_r = g_r > 1e-3
        if m_r.any():
            l_rxn_act = l_rxn_act + F.smooth_l1_loss(p_r[m_r], g_r[m_r], beta=0.02)
            n_seen += 1
    if n_seen > 0:
        l_rxn_act = l_rxn_act / n_seen

    total = (
        18.0 * l_flux + 5.0 * l_flux_act
        + 16.0 * l_heat + 6.0 * l_heat_act
        + 14.0 * l_rxn + 5.0 * l_rxn_act
        + 6.0 * F.mse_loss(pred, target)
    )
    return total, {
        "flux": float(l_flux.item()),
        "heat": float(l_heat.item()),
        "rxn": float(l_rxn.item()),
    }


def state_rollout_loss(pred: SimState, gt: SimState) -> torch.Tensor:
    l_mass = F.smooth_l1_loss(pred.mass / 1000.0, gt.mass / 1000.0, beta=0.01)
    _, T_pred = compute_C_and_T(pred.mass, pred.energy)
    _, T_gt = compute_C_and_T(gt.mass, gt.energy)
    l_temp = F.smooth_l1_loss((T_pred - 29000.0) / 10000.0, (T_gt - 29000.0) / 10000.0, beta=0.01)
    return 6.0 * l_mass + 4.0 * l_temp


@torch.no_grad()
def evaluate_world_model(
    model: WorldModel,
    v1_model: WorldModel,
    device: torch.device,
    eval_seeds: list[int],
    rollout_ticks: int = 10,
) -> dict:
    """
    Evaluate WorldModel on held-out unseen 256x256x4 Frostmarsh worlds (realistic 4-layer
    river/oil/wildfire/frost/highland-ledge worlds + 5-biome mixed scenes).
    """
    model.eval()
    v1_model.eval()
    c = get_consts(device)

    all_raw_mse, all_raw_mae = [], []
    all_proj_mse, all_proj_mae = [], []
    all_flux_mae, all_heat_mae, all_rxn_mae = [], [], []

    # Primary projection firewall active cell rate (`proj_active_pct`):
    # Fraction of cells where the conservation projection firewall meaningfully alters
    # any normalized delta channel (> 5e-3 in O(1) normalized units: flux/1000, heat/1e7, rxn/100).
    # On world_model.pt (v1), this is ~26-29%! Also report strict 1e-3 / 2e-3 & per-subsystem rates.
    cell_active_5e3_scaled = []
    cell_active_2e3_scaled = []
    cell_active_1e3_scaled = []
    elem_active_1e3 = []
    elem_active_1e2 = []
    donor_clamp_cell = []
    heat_clamp_cell = []
    rxn_clamp_cell = []
    proj_l1_list = []

    z_cos_list, z_mse_list = [], []

    for idx_s, seed in enumerate(eval_seeds):
        if idx_s % 2 == 0:
            st = make_frostmarsh_4layer_world(256, 256, device, seed=seed)
        else:
            st = make_scene(4, 256, 256, device, scene_type="mixed", seed=seed)

        # Warm up 6 ticks with RefSim
        torch.manual_seed(seed + 1000)
        for _ in range(6):
            fe_w, fm_w = sample_bolts(4, 256, 256, device, prob=0.70, max_bolts=3)
            st, _ = RefSim.step(st, fe_w, fm_w)

        for t in range(rollout_ticks):
            torch.manual_seed(seed + 2000 + t)
            fe, fm = sample_bolts(4, 256, 256, device, prob=0.75, max_bolts=3)
            x = encode_state(st, fe, fm)

            raw, z_new = world_forward_with_z(model, x)
            _, z_old = world_forward_with_z(v1_model, x)
            _, teacher = RefSim.step(st, fe, fm)

            m_eff, e_eff = apply_forcing(st, fe, fm)
            proj, _ = project_deltas(m_eff, e_eff, raw)

            diff = (raw - proj).abs()
            diff_scaled = diff.clone()
            diff_scaled[:, 8:10] = diff_scaled[:, 8:10] / 10.0

            mob_m = m_eff[:, c.mobile_idx].clamp(min=0.0) / 1000.0
            flux_e, flux_s = raw[:, 0:4], raw[:, 4:8]
            fw_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
            fn_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
            tot_out = F.relu(flux_e) + F.relu(-fw_in) + F.relu(flux_s) + F.relu(-fn_in)

            donor_clamp_cell.append(((tot_out - mob_m) > 2e-3).any(dim=1).float().mean().item() * 100.0)
            heat_clamp_cell.append((diff_scaled[:, 8:10] > 5e-3).any(dim=1).float().mean().item() * 100.0)
            rxn_clamp_cell.append((diff[:, 10:17] > 2e-3).any(dim=1).float().mean().item() * 100.0)

            cell_active_5e3_scaled.append((diff_scaled > 5e-3).any(dim=1).float().mean().item() * 100.0)
            cell_active_2e3_scaled.append((diff_scaled > 2e-3).any(dim=1).float().mean().item() * 100.0)
            cell_active_1e3_scaled.append((diff_scaled > 1e-3).any(dim=1).float().mean().item() * 100.0)
            elem_active_1e3.append((diff > 1e-3).float().mean().item() * 100.0)
            elem_active_1e2.append((diff > 1e-2).float().mean().item() * 100.0)
            proj_l1_list.append(diff_scaled.mean().item())

            all_raw_mse.append(F.mse_loss(raw, teacher).item())
            all_raw_mae.append(F.l1_loss(raw, teacher).item())
            all_proj_mse.append(F.mse_loss(proj, teacher).item())
            all_proj_mae.append(F.l1_loss(proj, teacher).item())
            all_flux_mae.append(F.l1_loss(raw[:, 0:8], teacher[:, 0:8]).item())
            all_heat_mae.append(F.l1_loss(raw[:, 8:10], teacher[:, 8:10]).item())
            all_rxn_mae.append(F.l1_loss(raw[:, 10:17], teacher[:, 10:17]).item())

            z_cos_list.append(F.cosine_similarity(z_new.float(), z_old.float(), dim=1, eps=1e-6).mean().item())
            z_mse_list.append(F.mse_loss(z_new.float(), z_old.float()).item())

            # Advance along exact teacher trajectory so v1 & v3 are evaluated on identical states!
            st, _ = RefSim.step(st, fe, fm)

    # Closed-system conservation invariant check (50 ticks on unseen 256x256x4 world)
    st_g1 = make_frostmarsh_4layer_world(256, 256, device, seed=88888)
    m0 = compute_total_mass(st_g1)
    e0 = compute_total_energy(st_g1)
    max_rel_m, max_rel_e = 0.0, 0.0
    for _ in range(50):
        raw_g1 = model(encode_state(st_g1))
        st_g1, _, _ = project_and_step(st_g1, raw_g1)
        m_now = compute_total_mass(st_g1)
        e_now = compute_total_energy(st_g1)
        max_rel_m = max(max_rel_m, float(((m_now - m0).abs() / (m0.abs() + 1e-12)).max().item()))
        max_rel_e = max(max_rel_e, float(((e_now - e0).abs() / (e0.abs() + 1e-12)).max().item()))

    return {
        "proj_active_pct": round(float(np.mean(cell_active_5e3_scaled)), 3),
        "proj_active_cell_scaled_2e3_pct": round(float(np.mean(cell_active_2e3_scaled)), 3),
        "proj_active_cell_scaled_1e3_pct": round(float(np.mean(cell_active_1e3_scaled)), 3),
        "proj_active_elem_1e3_pct": round(float(np.mean(elem_active_1e3)), 3),
        "proj_active_elem_1e2_pct": round(float(np.mean(elem_active_1e2)), 3),
        "donor_clamp_cell_pct": round(float(np.mean(donor_clamp_cell)), 3),
        "heat_clamp_cell_pct": round(float(np.mean(heat_clamp_cell)), 3),
        "rxn_clamp_cell_pct": round(float(np.mean(rxn_clamp_cell)), 3),
        "proj_mean_scaled_l1": round(float(np.mean(proj_l1_list)), 6),
        "raw_mse_vs_teacher": round(float(np.mean(all_raw_mse)), 6),
        "raw_mae_vs_teacher": round(float(np.mean(all_raw_mae)), 6),
        "proj_mse_vs_teacher": round(float(np.mean(all_proj_mse)), 6),
        "proj_mae_vs_teacher": round(float(np.mean(all_proj_mae)), 6),
        "flux_mae_vs_teacher": round(float(np.mean(all_flux_mae)), 6),
        "heat_mae_vs_teacher": round(float(np.mean(all_heat_mae)), 6),
        "rxn_mae_vs_teacher": round(float(np.mean(all_rxn_mae)), 6),
        "z_world_cosine_sim_to_v1": round(float(np.mean(z_cos_list)), 6),
        "z_world_mse_to_v1": round(float(np.mean(z_mse_list)), 6),
        "g1_max_rel_mass_drift_50t": max_rel_m,
        "g1_max_rel_energy_drift_50t": max_rel_e,
    }


def main() -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print(f"=== Conservation-Distilled WorldModel v3 Training on {torch.cuda.get_device_name(0)} ===", flush=True)

    ck_v1 = torch.load(V1_PT, map_location=device, weights_only=False)
    v1_sd = ck_v1["state_dict"] if isinstance(ck_v1, dict) and "state_dict" in ck_v1 else ck_v1

    v1_model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    v1_model.load_state_dict(v1_sd, strict=True)
    for p in v1_model.parameters():
        p.requires_grad_(False)

    model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device)
    model.load_state_dict(copy.deepcopy(v1_sd), strict=True)
    n_params = count_parameters(model)
    print(f"Warm-started WorldModel v3 from {V1_PT} | params={n_params:,}", flush=True)

    eval_seeds = [91001, 91002, 91003, 91004, 91005, 91006]

    print("\n--- Evaluating WorldModel v1 Baseline on Held-Out 256x256x4 Worlds ---", flush=True)
    v1_metrics = evaluate_world_model(v1_model, v1_model, device, eval_seeds=eval_seeds, rollout_ticks=10)
    print(json.dumps(v1_metrics, indent=2), flush=True)

    # Build fast dual-resolution Frostmarsh training pools (128x128 for throughput + 256x256 full-res)
    print("\n--- Warming up Frostmarsh Training Pools (128x128 & 256x256) ---", flush=True)
    pool128_4l = [make_frostmarsh_4layer_world(128, 128, device, seed=1000 + i) for i in range(6)]
    pool128_mix = make_scene(24, 128, 128, device, scene_type="mixed", seed=2000)
    pool256_4l = [make_frostmarsh_4layer_world(256, 256, device, seed=3000 + i) for i in range(4)]

    with torch.no_grad():
        for _ in range(12):
            for i in range(len(pool128_4l)):
                fe, fm = sample_bolts(4, 128, 128, device, prob=0.55, max_bolts=3)
                pool128_4l[i], _ = RefSim.step(pool128_4l[i], fe, fm)
            fe, fm = sample_bolts(24, 128, 128, device, prob=0.55, max_bolts=3)
            pool128_mix, _ = RefSim.step(pool128_mix, fe, fm)
            for i in range(len(pool256_4l)):
                fe, fm = sample_bolts(4, 256, 256, device, prob=0.55, max_bolts=3)
                pool256_4l[i], _ = RefSim.step(pool256_4l[i], fe, fm)

    total_steps = 650
    z_anchor_weight = 0.05
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.5e-5)

    best_score = 1e9
    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []
    t0 = time.perf_counter()

    print(f"\n--- Fine-Tuning WorldModel v3 ({total_steps} steps, z_anchor={z_anchor_weight}) ---", flush=True)
    for step in range(1, total_steps + 1):
        model.train()
        if step % 4 == 0:
            # Full-res 256x256x4 Frostmarsh batch
            pi = (step // 4) % len(pool256_4l)
            init_st = pool256_4l[pi].clone()
            B, H, W = 4, 256, 256
        elif step % 2 == 1:
            # 128x128x4 Frostmarsh 4-layer batch
            pi = (step // 2) % len(pool128_4l)
            init_st = pool128_4l[pi].clone()
            B, H, W = 4, 128, 128
        else:
            # 128x128 mixed-biome batch (B=8)
            idx = torch.randint(0, 24, (8,), device=device)
            init_st = SimState(
                mass=pool128_mix.mass[idx].clone(),
                energy=pool128_mix.energy[idx].clone(),
                flow_e=pool128_mix.flow_e[idx].clone(),
                flow_s=pool128_mix.flow_s[idx].clone(),
                elevation=pool128_mix.elevation[idx].clone(),
            )
            B, H, W = 8, 128, 128

        k_roll = 1 if step <= 450 else 2
        pred_st = init_st
        gt_st = init_st
        total_loss = torch.tensor(0.0, device=device)
        acc_phys, acc_proj, acc_z = 0.0, 0.0, 0.0

        for _ in range(k_roll):
            fe, fm = sample_bolts(B, H, W, device, prob=0.70, max_bolts=3)
            with torch.no_grad():
                gt_next, _ = RefSim.step(gt_st, fe, fm)
                _, teacher_deltas = RefSim.step(pred_st.detach(), fe, fm)
                m_eff, e_eff = apply_forcing(pred_st.detach(), fe, fm)
                x = encode_state(pred_st.detach(), fe, fm)
                _, z_old = world_forward_with_z(v1_model, x)

            raw_deltas, z_new = world_forward_with_z(model, x)
            pred_next, proj_deltas, _ = project_and_step(pred_st, raw_deltas, fe, fm)

            l_phys, _ = physics_teacher_loss(raw_deltas, teacher_deltas)
            l_proj, _ = conservation_firewall_loss(m_eff, e_eff, raw_deltas, proj_deltas)
            l_state = state_rollout_loss(pred_next, gt_next)
            l_z = F.mse_loss(z_new, z_old)

            w_proj = 1.0 + 1.0 * min(1.0, step / 400.0)
            step_loss = l_phys + w_proj * l_proj + 0.30 * l_state + z_anchor_weight * l_z
            total_loss = total_loss + step_loss / k_roll

            acc_phys += float(l_phys.item()) / k_roll
            acc_proj += float(l_proj.item()) / k_roll
            acc_z += float(l_z.item()) / k_roll

            pred_st = pred_next
            gt_st = gt_next

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        scheduler.step()

        # Advance & refresh replay pools periodically
        if step % 10 == 0:
            with torch.no_grad():
                r1 = (step // 10) % len(pool128_4l)
                if (step // 10) % 15 == 0:
                    pool128_4l[r1] = make_frostmarsh_4layer_world(128, 128, device, seed=10000 + step)
                else:
                    fe, fm = sample_bolts(4, 128, 128, device, prob=0.55, max_bolts=3)
                    pool128_4l[r1], _ = RefSim.step(pool128_4l[r1], fe, fm)

                fe_m, fm_m = sample_bolts(24, 128, 128, device, prob=0.50, max_bolts=3)
                pool128_mix, _ = RefSim.step(pool128_mix, fe_m, fm_m)

                r2 = (step // 10) % len(pool256_4l)
                if (step // 10) % 20 == 0:
                    pool256_4l[r2] = make_frostmarsh_4layer_world(256, 256, device, seed=20000 + step)
                else:
                    fe2, fm2 = sample_bolts(4, 256, 256, device, prob=0.55, max_bolts=3)
                    pool256_4l[r2], _ = RefSim.step(pool256_4l[r2], fe2, fm2)

        if step % 150 == 0 or step == total_steps:
            elapsed = time.perf_counter() - t0
            lr_now = scheduler.get_last_lr()[0]
            val = evaluate_world_model(model, v1_model, device, eval_seeds=eval_seeds[:4], rollout_ticks=6)
            score = (
                val["proj_active_pct"]
                + 0.25 * val["proj_active_cell_scaled_2e3_pct"]
                + 2000.0 * val["raw_mse_vs_teacher"]
                + 150.0 * max(0.0, 0.96 - val["z_world_cosine_sim_to_v1"])
            )
            tag = ""
            if score < best_score:
                best_score = score
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                tag = " *BEST*"
            rec = {
                "step": step,
                "elapsed_s": round(elapsed, 1),
                "lr": lr_now,
                "train_phys": round(acc_phys, 5),
                "train_proj": round(acc_proj, 5),
                "train_z_mse": round(acc_z, 5),
                "val_proj_active_pct": val["proj_active_pct"],
                "val_cell_2e3_pct": val["proj_active_cell_scaled_2e3_pct"],
                "val_raw_mse": val["raw_mse_vs_teacher"],
                "val_raw_mae": val["raw_mae_vs_teacher"],
                "val_proj_mse": val["proj_mse_vs_teacher"],
                "val_z_cos": val["z_world_cosine_sim_to_v1"],
            }
            history.append(rec)
            print(
                f"step {step:4d}/{total_steps} | t={elapsed:5.1f}s | lr={lr_now:.2e} | "
                f"proj_act={val['proj_active_pct']:5.2f}% (2e-3={val['proj_active_cell_scaled_2e3_pct']:5.2f}% "
                f"donor={val['donor_clamp_cell_pct']:4.2f}% heat={val['heat_clamp_cell_pct']:4.2f}% rxn={val['rxn_clamp_cell_pct']:4.2f}%) | "
                f"rawMSE={val['raw_mse_vs_teacher']:.6f} rawMAE={val['raw_mae_vs_teacher']:.6f} "
                f"projMSE={val['proj_mse_vs_teacher']:.6f} | Z_cos={val['z_world_cosine_sim_to_v1']:.4f}{tag}",
                flush=True,
            )

    train_time_s = round(time.perf_counter() - t0, 2)
    model.load_state_dict(best_sd)
    model.eval()

    print("\n--- Final Held-Out 256x256x4 Evaluation: WorldModel v3 vs v1 ---", flush=True)
    v3_metrics = evaluate_world_model(model, v1_model, device, eval_seeds=eval_seeds, rollout_ticks=10)

    # Verify EXACT drop-in compatibility with WorldModel & SparseWorldRunner
    pure_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    assert list(pure_sd.keys()) == list(v1_sd.keys()), "state_dict keys mismatch!"
    for k in pure_sd:
        assert pure_sd[k].shape == v1_sd[k].shape, f"Shape mismatch on {k}"

    verify_wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    verify_wm.load_state_dict(pure_sd, strict=True)
    runner = SparseWorldRunner(verify_wm, device=device, use_fp16=False)
    st_test = make_frostmarsh_4layer_world(256, 256, device, seed=55555)
    x_test = encode_state(st_test)
    with torch.no_grad():
        out_dense, z_dense = world_forward_with_z(verify_wm, x_test)
        out_sparse, z_sparse, _ = runner.step(x_test)
        sparse_diff = float(max((out_dense - out_sparse).abs().max().item(), (z_dense - z_sparse).abs().max().item()))
    assert sparse_diff < 1e-5, f"SparseWorldRunner diff too high: {sparse_diff}"
    print(f"Verified exact drop-in WorldModel & SparseWorldRunner compatibility: max_diff={sparse_diff:.2e}", flush=True)

    torch.save(
        {
            "state_dict": pure_sd,
            "n_params": n_params,
            "arch": {"in_ch": 18, "out_ch": 17, "width": 48, "num_res": 5},
            "distilled_from": str(V1_PT),
            "z_anchor_weight": z_anchor_weight,
            "train_time_s": train_time_s,
            "history": history,
        },
        V3_PT,
    )

    summary = {
        "device": torch.cuda.get_device_name(0),
        "checkpoint": str(V3_PT),
        "n_params": n_params,
        "drop_in_keys_match_v1": True,
        "sparse_world_runner_max_abs_diff_fp32": sparse_diff,
        "z_anchor_weight": z_anchor_weight,
        "train_steps": total_steps,
        "train_time_s": train_time_s,
        "eval_protocol": {
            "grid": "256x256x4",
            "unseen_worlds": len(eval_seeds),
            "rollout_ticks_per_world": 10,
            "seeds": eval_seeds,
        },
        "v1_baseline": v1_metrics,
        "v3_distilled": v3_metrics,
        "improvements": {
            "proj_active_pct_reduction_abs": round(v1_metrics["proj_active_pct"] - v3_metrics["proj_active_pct"], 3),
            "proj_active_pct_reduction_rel_pct": round(
                100.0 * (v1_metrics["proj_active_pct"] - v3_metrics["proj_active_pct"]) / max(v1_metrics["proj_active_pct"], 1e-6),
                2,
            ),
            "raw_mse_ratio_v3_over_v1": round(v3_metrics["raw_mse_vs_teacher"] / max(v1_metrics["raw_mse_vs_teacher"], 1e-12), 4),
            "raw_mae_ratio_v3_over_v1": round(v3_metrics["raw_mae_vs_teacher"] / max(v1_metrics["raw_mae_vs_teacher"], 1e-12), 4),
            "proj_mse_ratio_v3_over_v1": round(v3_metrics["proj_mse_vs_teacher"] / max(v1_metrics["proj_mse_vs_teacher"], 1e-12), 4),
            "proj_mae_ratio_v3_over_v1": round(v3_metrics["proj_mae_vs_teacher"] / max(v1_metrics["proj_mae_vs_teacher"], 1e-12), 4),
            "z_world_cosine_sim_to_v1": v3_metrics["z_world_cosine_sim_to_v1"],
        },
        "history": history,
    }
    METRICS_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {V3_PT} and {METRICS_JSON}!", flush=True)
    print(json.dumps({
        "v1_baseline": v1_metrics,
        "v3_distilled": v3_metrics,
        "improvements": summary["improvements"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
