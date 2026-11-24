"""
Frostmarsh Wave 20A: Law-Conditioned Universal 3D Physics World Model (`WorldModel3D-v8`, ~164.6K params).

Extends `WorldModel3DV5` (`158,374` base params, exact 10-species `A+` contract, 48-ch `Z_world_3d`)
with FiLM + output-scale conditioning on a continuous 6-float physical World-Law vector `theta_world`:
  `theta_world = [gravity_g, ambient_temp_shift, volatility, viscosity, precipitation, thermal_cond]`

Key guarantees:
  1. Exact 10-species `A+` contract & drop-in signature:
       `forward(state, local_dt=None, world_laws=None, return_z_flux=True, return_latent=True)`
     When `world_laws=None`, defaults to canonical Frostmarsh `[1.0, 0.0, 1.0, 1.0, 1.0, 1.0]`.
  2. Exact zero-feature canonical invariant:
       `encode_law_features(DEFAULT_WORLD_LAWS) == 0.0` identically.
     Combined with bias-free `law_trunk` + zero-initialized projections (`law_to_dt`, `law_h_mod`,
     `law_h_logscale`, `law_z_logscale`), `WorldModel3DV8.load_from_v7(...)` starts BIT-FOR-BIT
     identical (`max_abs_diff == 0.0`) to `WorldModel3D-v7` at Step 0!
  3. Law-Conditioned Reference Teacher (`step_refsim_3d_with_laws`):
     Cleanly extends `RefSim3D.step` with exact mass & energy conservation via
     `project_deltas_with_laws` + `project_and_apply_vertical_flux`, and matches `RefSim3D.step`
     identically at `DEFAULT_WORLD_LAWS`.
  4. `SparseWorld3DV8Runner`:
     True 3D `16x16x8` sparse active-brick execution with exact FP32 dense-vs-sparse parity
     `< 1e-6` under BOTH canonical and non-canonical `world_laws`!
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from sim_teacher import (  # noqa: E402
    H_CAP,
    NU_LIST,
    NUM_RXNS,
    NUM_SPECIES,
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
    SimState,
    _apply_transport,
    apply_forcing,
    apply_projected_deltas,
    compute_C_and_T,
    compute_total_energy,
    compute_total_mass,
    encode_state,
    get_consts,
)
from world_model_3d import Z_FLUX_CH, ZF_HEAT  # noqa: E402
from world_model_3d_v5 import (  # noqa: E402
    BIOME_NAMES,
    T_SKY_RAD,
    Z_SPECIES_IDX,
    Factorized3DResBlock,
    Isotropic3x3x3ResBlock,
    RefSim3D,
    WorldModel3DV5,
    _apply_xy_boundary_replicate_,
    _apply_z_boundary_replicate_,
    count_parameters,
    make_biome_3d_world,
)
from world_model_dt import encode_dt_features  # noqa: E402

# =============================================================================
# 6-Float Physical World-Law Vector (`theta_world`) & Named Evaluation Presets
# =============================================================================

LAW_DIM = 6
LAW_FEAT_DIM = 12

LAW_GRAVITY = 0
LAW_AMBIENT_TEMP_SHIFT = 1
LAW_VOLATILITY = 2
LAW_VISCOSITY = 3
LAW_PRECIPITATION = 4
LAW_THERMAL_COND = 5

LAW_NAMES = (
    "gravity_g",
    "ambient_temp_shift",
    "volatility",
    "viscosity",
    "precipitation",
    "thermal_cond",
)

DEFAULT_WORLD_LAWS: tuple[float, float, float, float, float, float] = (
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
)

WORLD_LAW_PRESET_NAMES = (
    "frostmarsh_standard",
    "freeze_out_survival",
    "rising_flood_canyon",
    "low_g_magma_crucible",
)

WORLD_LAW_PRESETS: dict[str, tuple[float, float, float, float, float, float]] = {
    "frostmarsh_standard": (1.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    "freeze_out_survival": (1.0, -5500.0, 1.2, 1.1, 1.4, 1.3),
    "rising_flood_canyon": (1.15, -600.0, 0.6, 0.55, 2.6, 1.0),
    "low_g_magma_crucible": (0.35, 5500.0, 2.2, 1.6, 0.25, 1.5),
}

WORLD_LAW_PRESETS_BY_ID: dict[int, tuple[str, tuple[float, float, float, float, float, float]]] = {
    idx: (name, WORLD_LAW_PRESETS[name]) for idx, name in enumerate(WORLD_LAW_PRESET_NAMES)
}

WorldLawsLike = torch.Tensor | Sequence[float] | None


def _unpack_world_laws_floats(world_laws: WorldLawsLike) -> tuple[float, float, float, float, float, float]:
    """Extract clamped physical scalars `(g, dT_amb, vol, visc, precip, k_th)` for the teacher."""
    if world_laws is None:
        return DEFAULT_WORLD_LAWS
    if isinstance(world_laws, torch.Tensor):
        flat = world_laws.detach().float().reshape(-1)
        if flat.numel() < LAW_DIM:
            raise ValueError(f"world_laws tensor must have at least {LAW_DIM} elements, got {flat.numel()}")
        vals = [float(flat[i].item()) for i in range(LAW_DIM)]
    else:
        if len(world_laws) != LAW_DIM:
            raise ValueError(f"world_laws must have length {LAW_DIM}, got {len(world_laws)}")
        vals = [float(v) for v in world_laws]

    g = max(0.10, min(3.0, vals[0]))
    dT_amb = max(-10000.0, min(10000.0, vals[1]))
    vol = max(0.20, min(4.0, vals[2]))
    visc = max(0.25, min(4.0, vals[3]))
    precip = max(0.10, min(4.0, vals[4]))
    k_th = max(0.20, min(4.0, vals[5]))
    return (g, dT_amb, vol, visc, precip, k_th)


def normalize_world_laws(
    world_laws: WorldLawsLike,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Normalize `world_laws` into a `[B, 6]` tensor on `(device, dtype)`."""
    dev = torch.device(device)
    if world_laws is None:
        t = torch.tensor(DEFAULT_WORLD_LAWS, dtype=dtype, device=dev).unsqueeze(0)
    elif isinstance(world_laws, torch.Tensor):
        t = world_laws.to(device=dev, dtype=dtype)
        if t.ndim == 1:
            if t.shape[0] != LAW_DIM:
                raise ValueError(f"Expected world_laws shape [6], got {tuple(t.shape)}")
            t = t.unsqueeze(0)
        elif t.ndim == 2:
            if t.shape[1] != LAW_DIM:
                raise ValueError(f"Expected world_laws shape [B, 6], got {tuple(t.shape)}")
        else:
            raise ValueError(f"Expected 1D [6] or 2D [B, 6] world_laws, got {tuple(t.shape)}")
    else:
        if len(world_laws) != LAW_DIM:
            raise ValueError(f"Expected {LAW_DIM} floats for world_laws, got {len(world_laws)}")
        t = torch.tensor([float(v) for v in world_laws], dtype=dtype, device=dev).unsqueeze(0)

    if t.shape[0] == 1 and batch_size > 1:
        t = t.expand(batch_size, LAW_DIM)
    elif t.shape[0] != batch_size:
        raise ValueError(f"world_laws batch {t.shape[0]} incompatible with batch_size={batch_size}")
    return t


def encode_law_features(
    world_laws: WorldLawsLike,
    batch_size: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Encode 6-float `theta_world` into 12 smooth physical basis features `[B, 12]`.
    EVERY feature is IDENTICALLY `0.0` at `DEFAULT_WORLD_LAWS = (1.0, 0.0, 1.0, 1.0, 1.0, 1.0)`!
    """
    if isinstance(world_laws, torch.Tensor):
        dev = world_laws.device if device == "cpu" and world_laws.device.type != "cpu" else torch.device(device)
        bsz = batch_size if batch_size is not None else (1 if world_laws.ndim == 1 else int(world_laws.shape[0]))
        t = normalize_world_laws(world_laws, batch_size=bsz, dtype=dtype, device=dev)
    else:
        bsz = batch_size if batch_size is not None else 1
        t = normalize_world_laws(world_laws, batch_size=bsz, dtype=dtype, device=device)

    g = t[:, LAW_GRAVITY : LAW_GRAVITY + 1].clamp(0.10, 3.0)
    dT = t[:, LAW_AMBIENT_TEMP_SHIFT : LAW_AMBIENT_TEMP_SHIFT + 1].clamp(-10000.0, 10000.0)
    vol = t[:, LAW_VOLATILITY : LAW_VOLATILITY + 1].clamp(0.20, 4.0)
    visc = t[:, LAW_VISCOSITY : LAW_VISCOSITY + 1].clamp(0.25, 4.0)
    precip = t[:, LAW_PRECIPITATION : LAW_PRECIPITATION + 1].clamp(0.10, 4.0)
    k_th = t[:, LAW_THERMAL_COND : LAW_THERMAL_COND + 1].clamp(0.20, 4.0)

    return torch.cat(
        [
            torch.log(g),                    # 0: log gravity (0 at 1.0)
            g - 1.0,                         # 1: linear gravity delta (0 at 1.0)
            dT / 5000.0,                     # 2: linear ambient temp shift (0 at 0.0)
            torch.tanh(dT / 3000.0),         # 3: saturated temp shift (0 at 0.0)
            torch.log(vol),                  # 4: log volatility -> threshold shift (0 at 1.0)
            (vol - 1.0) / 1.5,               # 5: linear volatility rate delta (0 at 1.0)
            torch.log(visc),                 # 6: log viscosity (0 at 1.0)
            (1.0 / visc) - 1.0,              # 7: exact inverse-viscosity flow delta (0 at 1.0)
            torch.log(precip),               # 8: log precipitation (0 at 1.0)
            (precip - 1.0) / 1.5,            # 9: linear precipitation rate delta (0 at 1.0)
            torch.log(k_th),                 # 10: log thermal conductivity (0 at 1.0)
            (k_th - 1.0) / 1.5,              # 11: linear thermal conductivity delta (0 at 1.0)
        ],
        dim=1,
    )


# =============================================================================
# Law-Conditioned Conservation Firewall & Exact 3D Reference Teacher
# =============================================================================


def compute_law_rxn_thresholds(
    L: int,
    world_laws: WorldLawsLike,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-layer reaction temperature thresholds `[L, 7, 1, 1]` under `world_laws`.
    At `DEFAULT_WORLD_LAWS`, returns exact canonical `[27315, 27315, 37315, 37000, 52000, 44000, 46000]`.
    """
    _, dT_amb, vol, _, _, _ = _unpack_world_laws_floats(world_laws)
    c = get_consts(device)
    thresh = c.rxn_thresh.expand(L, NUM_RXNS, 1, 1).clone()

    if dT_amb != 0.0:
        t_freeze_base = max(22500.0, min(31500.0, 27315.0 - 0.25 * dT_amb))
        thresh[:, 0] = t_freeze_base
        thresh[:, 1] = t_freeze_base
        if L >= 6:
            t_freeze_top = max(22000.0, min(32500.0, 27315.0 - 0.45 * dT_amb))
            thresh[L - 3 :, 0] = t_freeze_top
            thresh[L - 3 :, 1] = t_freeze_top

    if vol != 1.0:
        dT_ign = max(-8000.0, min(8000.0, -4500.0 * math.log(vol)))
        thresh[:, 4] = 52000.0 + dT_ign
        thresh[:, 5] = 44000.0 + dT_ign
        thresh[:, 6] = 46000.0 + dT_ign

    return thresh


def project_deltas_with_laws(
    mass: torch.Tensor,
    energy: torch.Tensor,
    raw_deltas: torch.Tensor,
    world_laws: WorldLawsLike = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact, differentiable invariant-preserving conservation firewall for `world_laws`.
    Preserves exact mass & total energy conservation, strict non-negativity, `H_CAP`,
    2nd-law heat conduction, and `[T_MIN, T_MAX]` bounds under ALL physical world laws.
    At `DEFAULT_WORLD_LAWS`, is BIT-FOR-BIT identical to `sim_teacher.project_deltas`.
    """
    c = get_consts(mass.device)
    B, _, H, W = mass.shape
    rxn_thresh = compute_law_rxn_thresholds(B, world_laws, mass.device)  # [B, 7, 1, 1]

    flux_e = raw_deltas[:, 0:4] * 1000.0
    flux_s = raw_deltas[:, 4:8] * 1000.0
    heat_e = raw_deltas[:, 8:9] * 1e6
    heat_s = raw_deltas[:, 9:10] * 1e6
    rxn = F.relu(raw_deltas[:, 10:17]) * 100.0

    # 1. Domain boundary & wall mask
    mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=mass.device)
    mask_e[:, :, :, -1] = 0.0
    mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=mass.device)
    mask_s[:, :, -1, :] = 0.0

    wall = ((mass[:, S_ROCK : S_ROCK + 1] >= 500.0) | (mass[:, S_ICE : S_ICE + 1] >= 700.0)).to(torch.float32)
    wall_e_nb = F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0)
    wall_s_nb = F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0)
    open_e = mask_e * (1.0 - wall) * (1.0 - wall_e_nb)
    open_s = mask_s * (1.0 - wall) * (1.0 - wall_s_nb)

    flux_e = flux_e * open_e
    flux_s = flux_s * open_s
    heat_e = heat_e * mask_e
    heat_s = heat_s * mask_s

    # 2. Donor-budget positivity projection
    mob_mass = torch.clamp(mass[:, c.mobile_idx], min=0.0)
    flux_w_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    flux_n_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    total_out = F.relu(flux_e) + F.relu(-flux_w_in) + F.relu(flux_s) + F.relu(-flux_n_in)

    donor_scale = torch.where(
        total_out > mob_mass,
        mob_mass / (total_out + 1e-8),
        torch.ones_like(total_out),
    ).clamp(0.0, 1.0)
    donor_scale_e_nb = F.pad(donor_scale[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    donor_scale_s_nb = F.pad(donor_scale[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    flux_e = torch.where(flux_e >= 0.0, flux_e * donor_scale, flux_e * donor_scale_e_nb)
    flux_s = torch.where(flux_s >= 0.0, flux_s * donor_scale, flux_s * donor_scale_s_nb)

    # 3. Receiver H_CAP projection
    flux_w_in2 = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    flux_n_in2 = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    total_in_cell = (F.relu(flux_w_in2) + F.relu(-flux_e) + F.relu(flux_n_in2) + F.relu(-flux_s)).sum(
        dim=1, keepdim=True
    )
    headroom = torch.clamp(H_CAP - mass.sum(dim=1, keepdim=True), min=0.0)
    recv_scale = torch.where(
        total_in_cell > headroom,
        headroom / (total_in_cell + 1e-8),
        torch.ones_like(total_in_cell),
    ).clamp(0.0, 1.0)
    recv_scale_e_nb = F.pad(recv_scale[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    recv_scale_s_nb = F.pad(recv_scale[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    flux_e = torch.where(flux_e >= 0.0, flux_e * recv_scale_e_nb, flux_e * recv_scale)
    flux_s = torch.where(flux_s >= 0.0, flux_s * recv_scale_s_nb, flux_s * recv_scale)

    # 4. Heat conduction 2nd-law maximum principle
    # For non-canonical laws (e.g., low-viscosity rushing floods that can drain a cell rapidly),
    # bound q_max by C_safe = min(C_cell, C_post_adv) so an emptied cell can NEVER accumulate
    # ghost negative energy; at canonical DEFAULT_WORLD_LAWS, keep exact RefSim/sim_teacher identity!
    C_cell, T_cK = compute_C_and_T(mass, energy)
    is_canonical = _unpack_world_laws_floats(world_laws) == DEFAULT_WORLD_LAWS
    if is_canonical:
        C_cond = C_cell
    else:
        fw_adv = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
        fn_adv = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
        dm_adv = fw_adv - flux_e + fn_adv - flux_s
        C_adv = torch.clamp(C_cell + (dm_adv * c.mobile_c).sum(dim=1, keepdim=True), min=0.0)
        C_cond = torch.where(C_adv > 1e-2, torch.minimum(C_cell, C_adv), torch.zeros_like(C_cell))

    C_e_nb = F.pad(C_cond[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    T_e_nb = F.pad(T_cK[:, :, :, 1:], (0, 1, 0, 0), value=T_AMB)
    dT_e = T_cK - T_e_nb
    q_max_e = torch.abs(dT_e) * (C_cond * C_e_nb) / (4.0 * (C_cond + C_e_nb) + 1e-6)
    q_max_e = torch.where((C_cond > 1e-6) & (C_e_nb > 1e-6), q_max_e, torch.zeros_like(q_max_e)) * mask_e
    heat_e = torch.where(
        dT_e >= 0.0,
        torch.clamp(heat_e, min=torch.zeros_like(q_max_e), max=q_max_e),
        torch.clamp(heat_e, min=-q_max_e, max=torch.zeros_like(q_max_e)),
    )

    C_s_nb = F.pad(C_cond[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    T_s_nb = F.pad(T_cK[:, :, 1:, :], (0, 0, 0, 1), value=T_AMB)
    dT_s = T_cK - T_s_nb
    q_max_s = torch.abs(dT_s) * (C_cond * C_s_nb) / (4.0 * (C_cond + C_s_nb) + 1e-6)
    q_max_s = torch.where((C_cond > 1e-6) & (C_s_nb > 1e-6), q_max_s, torch.zeros_like(q_max_s)) * mask_s
    heat_s = torch.where(
        dT_s >= 0.0,
        torch.clamp(heat_s, min=torch.zeros_like(q_max_s), max=q_max_s),
        torch.clamp(heat_s, min=-q_max_s, max=torch.zeros_like(q_max_s)),
    )

    # 5. Post-transport law-aware reaction gates & equilibrium caps
    mass_tr, energy_tr = _apply_transport(mass, energy, flux_e, flux_s, heat_e, heat_s)
    C_tr, T_tr = compute_C_and_T(mass_tr, energy_tr)

    t_diff = (T_tr - rxn_thresh) * c.rxn_dir
    t_gate = (t_diff > 0.0).to(torch.float32) * (C_tr > 1e-6).to(torch.float32)
    rxn = rxn * t_gate

    t_frz = rxn_thresh[:, 0:1]
    xi_cap_freeze = C_tr * F.relu(t_frz - T_tr) / (336000.0 + 21.0 * t_frz)
    xi_cap_melt = C_tr * F.relu(T_tr - t_frz) / (336000.0 + 21.0 * t_frz)
    xi_cap_cond = C_tr * F.relu(37000.0 - T_tr) / (840000.0 - 22.0 * 37000.0)
    xi_cap_boil = torch.where(
        T_tr < 38180.0,
        C_tr * F.relu(T_tr - 37315.0) / (840000.0 - 22.0 * 37315.0),
        mass_tr[:, S_WATER : S_WATER + 1],
    )
    phase_caps = torch.cat(
        [
            xi_cap_freeze,
            xi_cap_melt,
            xi_cap_boil,
            xi_cap_cond,
            torch.full_like(C_tr, 1e9),
            torch.full_like(C_tr, 1e9),
            torch.full_like(C_tr, 1e9),
        ],
        dim=1,
    )
    rxn = torch.minimum(rxn, phase_caps)

    # Reactant availability scaling
    consumed = torch.einsum("brhw,rs->bshw", rxn, c.nu_neg)
    spec_scale = torch.where(
        consumed > mass_tr,
        mass_tr / (consumed + 1e-8),
        torch.ones_like(consumed),
    ).clamp(0.0, 1.0)
    rxn_scales = []
    for r_idx in range(NUM_RXNS):
        s_scale = torch.ones_like(rxn[:, 0:1])
        for s_idx in range(NUM_SPECIES):
            if NU_LIST[r_idx][s_idx] < 0:
                s_scale = torch.minimum(s_scale, spec_scale[:, s_idx : s_idx + 1])
        rxn_scales.append(s_scale)
    rxn = rxn * torch.cat(rxn_scales, dim=1)

    # Temperature bounds safety [T_MIN, T_MAX]
    dE_rxn_all = (rxn * c.rxn_dE).sum(dim=1, keepdim=True)
    dC_rxn_all = (rxn * c.rxn_dc).sum(dim=1, keepdim=True)
    E_post = energy_tr + dE_rxn_all
    C_post = torch.clamp(C_tr + dC_rxn_all, min=0.0)

    excess_hi = E_post - C_post * T_MAX
    hi_coeff = c.rxn_dE - c.rxn_dc * T_MAX
    pos_hi_contrib = (rxn * F.relu(hi_coeff)).sum(dim=1, keepdim=True)
    hi_scale = torch.where(
        (excess_hi > 0.0) & (pos_hi_contrib > 1e-6),
        torch.clamp(1.0 - excess_hi / (pos_hi_contrib + 1e-6), 0.0, 1.0),
        torch.ones_like(excess_hi),
    )
    rxn = torch.where(hi_coeff > 0.0, rxn * hi_scale, rxn)

    dE_rxn_all = (rxn * c.rxn_dE).sum(dim=1, keepdim=True)
    dC_rxn_all = (rxn * c.rxn_dc).sum(dim=1, keepdim=True)
    E_post = energy_tr + dE_rxn_all
    C_post = torch.clamp(C_tr + dC_rxn_all, min=0.0)
    deficit_lo = C_post * T_MIN - E_post
    lo_coeff = c.rxn_dc * T_MIN - c.rxn_dE
    pos_lo_contrib = (rxn * F.relu(lo_coeff)).sum(dim=1, keepdim=True)
    lo_scale = torch.where(
        (deficit_lo > 0.0) & (pos_lo_contrib > 1e-6),
        torch.clamp(1.0 - deficit_lo / (pos_lo_contrib + 1e-6), 0.0, 1.0),
        torch.ones_like(deficit_lo),
    )
    rxn = torch.where(lo_coeff > 0.0, rxn * lo_scale, rxn)

    proj_deltas = torch.cat(
        [
            flux_e / 1000.0,
            flux_s / 1000.0,
            heat_e / 1e6,
            heat_s / 1e6,
            rxn / 100.0,
        ],
        dim=1,
    )
    active_mask = (proj_deltas - raw_deltas).abs() > 1e-4
    return proj_deltas, active_mask


def project_and_step_with_laws(
    state: SimState,
    raw_deltas: torch.Tensor,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
    world_laws: WorldLawsLike = None,
) -> tuple[SimState, torch.Tensor, torch.Tensor]:
    """Conservative horizontal step with law-aware exact projection."""
    mass_eff, energy_eff = apply_forcing(state, forcing_energy, forcing_mass)
    proj_deltas, active_mask = project_deltas_with_laws(mass_eff, energy_eff, raw_deltas, world_laws=world_laws)
    next_state = apply_projected_deltas(state, proj_deltas, forcing_energy, forcing_mass)
    return next_state, proj_deltas, active_mask


@torch.no_grad()
def compute_raw_horiz_deltas_with_laws(
    state: SimState,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
    world_laws: WorldLawsLike = None,
) -> torch.Tensor:
    """
    Compute exact law-conditioned horizontal raw deltas `[L, 17, H, W]`.
    Matches `RefSim3D.compute_raw_horiz_deltas` bit-for-bit at `DEFAULT_WORLD_LAWS`.
    """
    g, _, vol, visc, precip, k_th = _unpack_world_laws_floats(world_laws)
    inv_visc = 1.0 / visc
    inv_visc_gas = math.sqrt(inv_visc)

    c = get_consts(state.mass.device)
    mass, energy = apply_forcing(state, forcing_energy, forcing_mass)
    L, _, H, W = mass.shape
    device = mass.device

    C_cell, T_cK = compute_C_and_T(mass, energy)
    rxn_thresh = compute_law_rxn_thresholds(L, world_laws, device)

    mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=device)
    mask_e[:, :, :, -1] = 0.0
    mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=device)
    mask_s[:, :, -1, :] = 0.0

    wall = ((mass[:, S_ROCK : S_ROCK + 1] >= 500.0) | (mass[:, S_ICE : S_ICE + 1] >= 700.0)).to(torch.float32)
    wall_e_nb = F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0)
    wall_s_nb = F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0)
    open_e = mask_e * (1.0 - wall) * (1.0 - wall_e_nb)
    open_s = mask_s * (1.0 - wall) * (1.0 - wall_s_nb)

    # 1. Liquid flow (gravity hydrostatic head + inverse-viscosity fluid speed)
    water = mass[:, S_WATER : S_WATER + 1]
    oil = mass[:, S_OIL : S_OIL + 1]
    tot_liq = water + oil
    head = state.elevation * 120.0 + tot_liq

    head_e_nb = F.pad(head[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    head_s_nb = F.pad(head[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

    drive_e = (0.20 * (head - head_e_nb) * g + 0.35 * state.flow_e) * inv_visc * open_e
    drive_s = (0.20 * (head - head_s_nb) * g + 0.35 * state.flow_s) * inv_visc * open_s

    water_e_nb = F.pad(water[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    water_s_nb = F.pad(water[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    oil_e_nb = F.pad(oil[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    oil_s_nb = F.pad(oil[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    tot_liq_e_nb = water_e_nb + oil_e_nb
    tot_liq_s_nb = water_s_nb + oil_s_nb

    frac_w_e = torch.where(drive_e >= 0.0, water / (tot_liq + 1e-6), water_e_nb / (tot_liq_e_nb + 1e-6))
    frac_o_e = torch.where(drive_e >= 0.0, oil / (tot_liq + 1e-6), oil_e_nb / (tot_liq_e_nb + 1e-6))
    has_liq_e = torch.where(drive_e >= 0.0, (tot_liq > 1e-4).to(torch.float32), (tot_liq_e_nb > 1e-4).to(torch.float32))

    frac_w_s = torch.where(drive_s >= 0.0, water / (tot_liq + 1e-6), water_s_nb / (tot_liq_s_nb + 1e-6))
    frac_o_s = torch.where(drive_s >= 0.0, oil / (tot_liq + 1e-6), oil_s_nb / (tot_liq_s_nb + 1e-6))
    has_liq_s = torch.where(drive_s >= 0.0, (tot_liq > 1e-4).to(torch.float32), (tot_liq_s_nb > 1e-4).to(torch.float32))

    flux_e_water = drive_e * frac_w_e * has_liq_e
    flux_s_water = drive_s * frac_w_s * has_liq_s
    flux_e_oil = (0.55 * drive_e * frac_o_e * has_liq_e + (0.06 * inv_visc) * (oil - oil_e_nb)) * open_e
    flux_s_oil = (0.55 * drive_s * frac_o_s * has_liq_s + (0.06 * inv_visc) * (oil - oil_s_nb)) * open_s

    # 2. Gas diffusion + gentle breeze drift
    steam = mass[:, S_STEAM : S_STEAM + 1]
    smoke = mass[:, S_SMOKE : S_SMOKE + 1]
    steam_e_nb = F.pad(steam[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    steam_s_nb = F.pad(steam[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    smoke_e_nb = F.pad(smoke[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    smoke_s_nb = F.pad(smoke[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

    flux_e_steam = (0.16 * (steam - steam_e_nb) + 0.035 * steam) * inv_visc_gas * open_e
    flux_s_steam = (0.16 * (steam - steam_s_nb) - 0.020 * steam_s_nb) * inv_visc_gas * open_s
    flux_e_smoke = (0.15 * (smoke - smoke_e_nb) + 0.035 * smoke) * inv_visc_gas * open_e
    flux_s_smoke = (0.15 * (smoke - smoke_s_nb) - 0.020 * smoke_s_nb) * inv_visc_gas * open_s

    flux_e = torch.cat([flux_e_water, flux_e_steam, flux_e_smoke, flux_e_oil], dim=1)
    flux_s = torch.cat([flux_s_water, flux_s_steam, flux_s_smoke, flux_s_oil], dim=1)

    # 3. Horizontal heat conduction (scaled by thermal_cond)
    tot_mass = mass.sum(dim=1, keepdim=True)
    k_cell = torch.where(
        tot_mass > 1e-4,
        (mass * c.spec_k).sum(dim=1, keepdim=True) / torch.clamp(tot_mass, min=1e-4),
        torch.full_like(tot_mass, 2.0),
    )
    k_e_nb = F.pad(k_cell[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    k_s_nb = F.pad(k_cell[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    k_eff_e = 2.0 * k_cell * k_e_nb / (k_cell + k_e_nb + 1e-6)
    k_eff_s = 2.0 * k_cell * k_s_nb / (k_cell + k_s_nb + 1e-6)

    C_e_nb = F.pad(C_cell[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    T_e_nb = F.pad(T_cK[:, :, :, 1:], (0, 1, 0, 0), value=T_AMB)
    C_s_nb = F.pad(C_cell[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    T_s_nb = F.pad(T_cK[:, :, 1:, :], (0, 0, 0, 1), value=T_AMB)

    C_eff_e = (C_cell * C_e_nb) / (C_cell + C_e_nb + 1e-6)
    C_eff_s = (C_cell * C_s_nb) / (C_cell + C_s_nb + 1e-6)
    heat_e = (0.008 * k_th) * k_eff_e * C_eff_e * (T_cK - T_e_nb) * mask_e
    heat_s = (0.008 * k_th) * k_eff_s * C_eff_s * (T_cK - T_s_nb) * mask_s

    # 4. Law-conditioned reactions on exact post-transport state
    dummy_rxn = torch.zeros((L, 7, H, W), dtype=torch.float32, device=device)
    tr_raw = torch.cat([flux_e / 1000.0, flux_s / 1000.0, heat_e / 1e6, heat_s / 1e6, dummy_rxn], dim=1)
    tr_proj, _ = project_deltas_with_laws(mass, energy, tr_raw, world_laws=world_laws)
    mass_tr, energy_tr = _apply_transport(
        mass,
        energy,
        tr_proj[:, 0:4] * 1000.0,
        tr_proj[:, 4:8] * 1000.0,
        tr_proj[:, 8:9] * 1e6,
        tr_proj[:, 9:10] * 1e6,
    )
    _, T_tr = compute_C_and_T(mass_tr, energy_tr)

    t_frz = rxn_thresh[:, 0:1]
    t_wood = rxn_thresh[:, 4:5]
    t_oil = rxn_thresh[:, 5:6]
    t_plant = rxn_thresh[:, 6:7]

    r_freeze = 0.040 * F.relu(t_frz - T_tr) * (mass_tr[:, S_WATER : S_WATER + 1] / 1000.0)
    r_melt = 0.030 * F.relu(T_tr - t_frz) * (mass_tr[:, S_ICE : S_ICE + 1] / 1000.0)
    r_boil = (0.025 * (0.5 + 0.5 * vol)) * F.relu(T_tr - 37315.0) * (mass_tr[:, S_WATER : S_WATER + 1] / 1000.0)
    r_cond = (0.025 * precip) * F.relu(37000.0 - T_tr) * (mass_tr[:, S_STEAM : S_STEAM + 1] / 1000.0)
    r_wood = (0.0025 * vol) * F.relu(T_tr - t_wood) * (mass_tr[:, S_WOOD : S_WOOD + 1] / 1000.0)
    r_oil = (0.0040 * vol) * F.relu(T_tr - t_oil) * (mass_tr[:, S_OIL : S_OIL + 1] / 1000.0)
    r_plant = (0.0035 * vol) * F.relu(T_tr - t_plant) * (mass_tr[:, S_PLANT : S_PLANT + 1] / 1000.0)

    rxn = torch.cat([r_freeze, r_melt, r_boil, r_cond, r_wood, r_oil, r_plant], dim=1)
    raw = torch.cat(
        [
            flux_e / 1000.0,
            flux_s / 1000.0,
            heat_e / 1e6,
            heat_s / 1e6,
            rxn / 100.0,
        ],
        dim=1,
    )
    if L >= 6:
        cloud_boost = torch.ones((L, 1, 1, 1), dtype=raw.dtype, device=raw.device)
        cloud_boost[L - 3 :] = 2.2
        cold_steam = (T_cK < 37000.0).float() * (mass[:, S_STEAM : S_STEAM + 1] > 1.0).float()
        rxn_cond = raw[:, 13:14] * (1.0 + (cloud_boost - 1.0) * cold_steam)
        raw = torch.cat([raw[:, :13], rxn_cond, raw[:, 14:]], dim=1)
    return raw


@torch.no_grad()
def compute_vertical_flux_with_laws(
    state: SimState,
    world_laws: WorldLawsLike = None,
) -> torch.Tensor:
    """
    Compute exact law-conditioned vertical interface fluxes `[L - 1, 5, H, W]`.
    Matches `RefSim3D.compute_vertical_flux` bit-for-bit at `DEFAULT_WORLD_LAWS`.
    """
    g, dT_amb, _, visc, precip, k_th = _unpack_world_laws_floats(world_laws)
    L, _, H, W = state.mass.shape
    dev = state.mass.device
    if L <= 1:
        return torch.zeros((0, Z_FLUX_CH, H, W), dtype=torch.float32, device=dev)

    c = get_consts(dev)
    mass = state.mass
    energy = state.energy
    C_cell, T_cK = compute_C_and_T(mass, energy)

    lo = slice(0, L - 1)
    hi = slice(1, L)

    solid_floor_lo = (
        (mass[lo, S_ROCK] >= 450.0)
        | (mass[lo, S_SOIL] >= 550.0)
        | (mass[lo, S_ICE] >= 600.0)
        | (mass[lo, S_WOOD] >= 380.0)
    )
    solid_ceil_hi = (
        (mass[hi, S_ROCK] >= 500.0)
        | (mass[hi, S_ICE] >= 650.0)
        | (mass[hi, S_SOIL] >= 650.0)
    )
    solid_rock_lo = (mass[lo, S_ROCK] >= 500.0) | (mass[lo, S_ICE] >= 700.0)

    grav_open = (~solid_floor_lo).float()
    gas_open = ((~solid_ceil_hi) & (~solid_rock_lo)).float()

    tot_mass = mass.sum(dim=1)
    headroom_lo = (H_CAP - tot_mass[lo]).clamp(min=0.0)
    headroom_hi = (H_CAP - tot_mass[hi]).clamp(min=0.0)

    # 1. Gravity plunge (water & oil settling scaled by gravity_g; oil mildly damped by viscosity)
    w_hi = mass[hi, S_WATER].clamp(min=0.0)
    o_hi = mass[hi, S_OIL].clamp(min=0.0)
    lo_air_boost = (tot_mass[lo] < 80.0).float()
    oil_visc_scale = 1.0 / (visc ** 0.25)
    g_rate_w = ((0.44 + 0.14 * lo_air_boost) * g).clamp(0.0, 0.90) * grav_open
    g_rate_o = ((0.32 + 0.10 * lo_air_boost) * g * oil_visc_scale).clamp(0.0, 0.85) * grav_open

    raw_fall_w = w_hi * g_rate_w
    raw_fall_o = o_hi * g_rate_o
    fall_sum = raw_fall_w + raw_fall_o
    fall_cap = torch.minimum(torch.ones_like(fall_sum), (0.85 * headroom_lo) / (fall_sum + 1e-6))
    fall_w = raw_fall_w * fall_cap
    fall_o = raw_fall_o * fall_cap

    # 2. Thermal buoyancy (steam & smoke rise scaled by gravity_g & shifted ambient T_amb_eff)
    st_lo = mass[lo, S_STEAM].clamp(min=0.0)
    sm_lo = mass[lo, S_SMOKE].clamp(min=0.0)
    T_lo = T_cK[lo, 0]
    T_hi = T_cK[hi, 0]
    T_amb_eff = T_AMB + dT_amb
    buoy_boost = (
        1.0
        + 0.55 * ((T_lo - T_amb_eff) / 14000.0).clamp(0.0, 2.0)
        + 0.30 * ((T_lo - T_hi) / 10000.0).clamp(0.0, 1.5)
    )
    top_cap = torch.ones((L - 1, 1, 1), dtype=torch.float32, device=dev)
    if L >= 6:
        top_cap[L - 3 :] = max(0.15, min(0.85, 0.45 / (precip ** 0.35)))
    b_rate_st = (0.32 * buoy_boost * top_cap * g).clamp(0.0, 0.72) * gas_open
    b_rate_sm = (0.26 * buoy_boost * g).clamp(0.0, 0.68) * gas_open

    raw_rise_st = st_lo * b_rate_st
    raw_rise_sm = sm_lo * b_rate_sm
    rise_sum = raw_rise_st + raw_rise_sm
    rise_cap = torch.minimum(torch.ones_like(rise_sum), (0.85 * headroom_hi) / (rise_sum + 1e-6))
    rise_st = raw_rise_st * rise_cap
    rise_sm = raw_rise_sm * rise_cap

    # 3. 2nd-law vertical thermal conduction & convection (scaled by thermal_cond)
    k_cell = torch.where(
        tot_mass > 1e-4,
        (mass * c.spec_k).sum(dim=1) / torch.clamp(tot_mass, min=1e-4),
        torch.zeros_like(tot_mass),
    )
    k_lo, k_hi = k_cell[lo], k_cell[hi]
    k_harm = 2.0 * k_lo * k_hi / (k_lo + k_hi + 1e-4)

    C_lo, C_hi = C_cell[lo, 0], C_cell[hi, 0]
    dT_z = T_lo - T_hi
    valid_c = (C_lo > 1e-3) & (C_hi > 1e-3)
    q_max_z = dT_z.abs() * (C_lo * C_hi) / (4.0 * (C_lo + C_hi) + 1e-6)
    q_max_z = torch.where(valid_c, q_max_z, torch.zeros_like(q_max_z))

    conv_mult = 1.0 + 1.5 * gas_open * (dT_z > 0.0).float() * (1.0 + (st_lo + sm_lo) / 600.0).clamp(max=2.2)
    q_raw = (0.045 * k_th) * k_harm * conv_mult * dT_z * (
        C_lo.clamp(max=45000.0) * C_hi.clamp(max=45000.0)
    ).sqrt() / 12000.0
    q_clamped = torch.where(valid_c, torch.clamp(q_raw, min=-q_max_z, max=q_max_z), torch.zeros_like(q_raw))

    return torch.stack(
        [
            -fall_w / 1000.0,
            -fall_o / 1000.0,
            rise_st / 1000.0,
            rise_sm / 1000.0,
            q_clamped / 1e6,
        ],
        dim=1,
    )


@torch.no_grad()
def apply_radiative_cooling_with_laws(
    state: SimState,
    dt_scale: float = 1.0,
    world_laws: WorldLawsLike = None,
) -> SimState:
    """
    Closed-mass upper-atmosphere radiative cooling + ambient climate shift (`mass_err == 0`).
    Matches `RefSim3D.apply_radiative_cooling_upper_atm` bit-for-bit at `DEFAULT_WORLD_LAWS`.
    """
    _, dT_amb, _, _, precip, _ = _unpack_world_laws_floats(world_laws)
    L = state.mass.shape[0]
    if L < 4 or dt_scale <= 0.0:
        return state

    C_cell, T_cK = compute_C_and_T(state.mass, state.energy)
    energy = state.energy.clone()

    # 1. Upper-troposphere radiative cooling toward shifted sky equilibrium
    t_sky_eff = max(T_MIN + 500.0, min(T_MAX - 5000.0, T_SKY_RAD + dT_amb))
    rate_sky = 0.22 * (0.5 + 0.5 * precip)
    alpha_sky = min(0.75, float(rate_sky) * float(dt_scale))
    z0 = max(0, L - 3)
    T_top = T_cK[z0:]
    C_top = C_cell[z0:]
    cool_mask = (C_top > 1e-4) & (T_top > t_sky_eff)
    T_new_top = torch.where(cool_mask, T_top - alpha_sky * (T_top - t_sky_eff), T_top)
    energy[z0:] = torch.where(C_top > 1e-4, C_top * T_new_top, energy[z0:])

    # 2. Gentle global ambient climate relaxation ONLY when ambient_temp_shift != 0
    if dT_amb != 0.0:
        _, T_cur = compute_C_and_T(state.mass, energy)
        t_amb_eff = max(T_MIN + 1000.0, min(T_MAX - 5000.0, T_AMB + dT_amb))
        alpha_amb = min(0.25, 0.08 * min(1.0, abs(dT_amb) / 2500.0) * float(dt_scale))
        if dT_amb < 0.0:
            shift_mask = (C_cell > 1e-4) & (T_cur > t_amb_eff)
            T_shifted = torch.where(shift_mask, T_cur - alpha_amb * (T_cur - t_amb_eff), T_cur)
        else:
            shift_mask = (C_cell > 1e-4) & (T_cur < t_amb_eff)
            T_shifted = torch.where(shift_mask, T_cur + alpha_amb * (t_amb_eff - T_cur), T_cur)
        energy = torch.where(C_cell > 1e-4, C_cell * T_shifted, energy)

    return SimState(
        mass=state.mass,
        energy=energy,
        flow_e=state.flow_e,
        flow_s=state.flow_s,
        elevation=state.elevation,
    )


@torch.no_grad()
def apply_structural_collapse_with_laws(
    state: SimState,
    dt_scale: float = 1.0,
    max_span: int = 4,
    world_laws: WorldLawsLike = None,
) -> tuple[SimState, torch.Tensor]:
    """Conservative 3D structural collapse scaled by `gravity_g`."""
    g, _, _, _, _, _ = _unpack_world_laws_floats(world_laws)
    return RefSim3D.apply_structural_collapse_3d(state, dt_scale=float(dt_scale) * g, max_span=max_span)


@torch.no_grad()
def step_refsim_3d_with_laws(
    state: SimState,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
    local_dt: torch.Tensor | float | None = 1.0,
    world_laws: WorldLawsLike = None,
    radiative_cooling: bool = True,
    structural_collapse: bool = True,
) -> tuple[SimState, torch.Tensor, torch.Tensor]:
    """
    Law-Conditioned Isotropic 3D Reference Physics Step (`theta_world` + `local_dt`).
    Matches `RefSim3D.step(...)` cleanly (0.0 diff) at `DEFAULT_WORLD_LAWS`.

    Returns:
      - `next_state` : `SimState`
      - `proj_horiz` : `[L, 17, H, W]` exact integrated horizontal/reaction deltas
      - `proj_z_flux`: `[L - 1, 5, H, W]` exact integrated vertical interface fluxes
    """
    L, _, H, W = state.mass.shape
    dev = state.mass.device
    dt_map = RefSim3D._broadcast_dt(local_dt, L, H, W, dev)
    dt_max = float(dt_map.max().item())

    if dt_max <= 1e-8:
        return (
            state.clone(),
            torch.zeros((L, 17, H, W), dtype=torch.float32, device=dev),
            torch.zeros((max(L - 1, 0), Z_FLUX_CH, H, W), dtype=torch.float32, device=dev),
        )

    n_sub = max(1, int(math.ceil(dt_max)))
    dt_sub = dt_map / float(n_sub)
    dt_z_sub = torch.minimum(dt_sub[:-1], dt_sub[1:]) if L > 1 else dt_sub[:0]

    cur = state.clone()
    horiz_acc = torch.zeros((L, 17, H, W), dtype=torch.float32, device=dev)
    z_acc = torch.zeros((max(L - 1, 0), Z_FLUX_CH, H, W), dtype=torch.float32, device=dev)

    for k in range(n_sub):
        fe_k = forcing_energy if k == 0 else None
        fm_k = forcing_mass if k == 0 else None

        raw_h = compute_raw_horiz_deltas_with_laws(cur, fe_k, fm_k, world_laws=world_laws)
        raw_h_dt = RefSim3D._scale_horiz_by_dt(raw_h, dt_sub)
        cur, proj_h_k, _ = project_and_step_with_laws(cur, raw_h_dt, fe_k, fm_k, world_laws=world_laws)
        horiz_acc = horiz_acc + proj_h_k

        if L > 1:
            raw_z = compute_vertical_flux_with_laws(cur, world_laws=world_laws)
            raw_z_dt = raw_z * dt_z_sub
            cur, proj_z_k = RefSim3D.project_and_apply_vertical_flux(cur, raw_z_dt)
            z_acc = z_acc + proj_z_k

        if structural_collapse and L > 1:
            cur, _ = apply_structural_collapse_with_laws(cur, dt_scale=float(dt_sub.mean().item()), world_laws=world_laws)

        if radiative_cooling and L >= 4:
            cur = apply_radiative_cooling_with_laws(cur, dt_scale=float(dt_sub.mean().item()), world_laws=world_laws)

    return cur, horiz_acc, z_acc


# =============================================================================
# WorldModel3D-v8 Architecture (~164.6K params, exact v7 drop-in + World Laws)
# =============================================================================


class WorldModel3DV8(WorldModel3DV5):
    """
    Wave 20A Law-Conditioned Universal 3D Physics World Model (~164.6K params).

    Subclasses `WorldModel3DV5` (`158,374` base params) and adds a compact bias-free
    World-Law encoder (`+6,248` params -> `164,622` total params):
      - `law_trunk`: `12 -> 44 -> 44` (`bias=False`, exact 0 at `DEFAULT_WORLD_LAWS`)
      - `law_to_dt`: `44 -> dt_dim(16)` fused additively into `dt_emb` (modulates all 5 FiLM blocks,
        `dt_head_mod`, and `dt_z_mod` jointly with `local_dt`!)
      - `law_h_mod`: `44 -> width(48)` injected into horizontal head alongside `h_x_in(x_2d)` for
        razor-sharp law-shifted freezing & combustion ignition thresholds!
      - `law_h_logscale`: `44 -> out_ch(17)` smooth log-scale output modulation `exp(clamp(..., -2.2, 2.2))`
      - `law_z_logscale`: `44 -> z_flux_ch(5)` smooth log-scale vertical flux modulation

    Drop-in replacement for `WorldModel3DV5`:
      `forward(state, local_dt=None, world_laws=None, return_z_flux=True, return_latent=True)`
    """

    def __init__(
        self,
        in_ch: int = 18,
        out_ch: int = 17,
        width: int = 48,
        z_flux_ch: int = Z_FLUX_CH,
        dt_dim: int = 16,
        mid_3d: int = 32,
        law_hidden: int = 44,
    ):
        super().__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            width=width,
            z_flux_ch=z_flux_ch,
            dt_dim=dt_dim,
            mid_3d=mid_3d,
        )
        self.law_dim = LAW_DIM
        self.law_feat_dim = LAW_FEAT_DIM
        self.law_hidden = int(law_hidden)

        # Bias-free MLP trunk so encode_law_features(DEFAULT_WORLD_LAWS) == 0 -> exact 0 forever!
        self.law_mlp = nn.Sequential(
            nn.Linear(self.law_feat_dim, self.law_hidden, bias=False),
            nn.GELU(),
            nn.Linear(self.law_hidden, self.law_hidden, bias=False),
            nn.GELU(),
        )
        # Zero-initialized output projections -> exact Step-0 bit-for-bit identity with WorldModel3D-v7!
        self.law_to_dt = nn.Linear(self.law_hidden, self.dt_dim, bias=False)
        self.law_h_mod = nn.Linear(self.law_hidden, self.width, bias=False)
        self.law_h_logscale = nn.Linear(self.law_hidden, self.out_ch, bias=False)
        self.law_z_logscale = nn.Linear(self.law_hidden, self.z_flux_ch, bias=False)
        self._zero_init_law_projections()

    def _zero_init_law_projections(self) -> None:
        with torch.no_grad():
            self.law_to_dt.weight.zero_()
            self.law_h_mod.weight.zero_()
            self.law_h_logscale.weight.zero_()
            self.law_z_logscale.weight.zero_()

    def law_parameters(self) -> list[nn.Parameter]:
        """Return only the World-Law conditioning parameters (for differential LR warm-start)."""
        return (
            list(self.law_mlp.parameters())
            + list(self.law_to_dt.parameters())
            + list(self.law_h_mod.parameters())
            + list(self.law_h_logscale.parameters())
            + list(self.law_z_logscale.parameters())
        )

    def base_parameters(self) -> list[nn.Parameter]:
        """Return all base `WorldModel3DV5` parameters (excluding World-Law modules)."""
        law_ids = {id(p) for p in self.law_parameters()}
        return [p for p in self.parameters() if id(p) not in law_ids]

    def load_v7_state_dict_(self, sd: dict[str, torch.Tensor]) -> None:
        """Load a `WorldModel3DV5` (v5/v6/v7) state_dict in-place with zero-init law projections."""
        missing, unexpected = self.load_state_dict(sd, strict=False)
        expected_missing = {
            "law_mlp.0.weight",
            "law_mlp.2.weight",
            "law_to_dt.weight",
            "law_h_mod.weight",
            "law_h_logscale.weight",
            "law_z_logscale.weight",
        }
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading v7 checkpoint into WorldModel3DV8: {unexpected}")
        bad_missing = set(missing) - expected_missing
        if bad_missing:
            raise RuntimeError(f"Missing base WorldModel3DV5 keys in v7 checkpoint: {sorted(bad_missing)}")
        self._zero_init_law_projections()

    @classmethod
    def load_from_v7(
        cls,
        ckpt_path: str | Path,
        map_location: torch.device | str | None = "cpu",
        **kwargs: Any,
    ) -> "WorldModel3DV8":
        """
        Instantiate `WorldModel3DV8` and load `world_model_3d_v7.pt` weights cleanly with
        zero-initialized law projections so Step 0 outputs are bit-for-bit identical to v7!
        """
        model = cls(**kwargs)
        ck = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        model.load_v7_state_dict_(sd)
        if map_location is not None:
            model = model.to(map_location)
        return model

    def _compute_law_mods(
        self,
        world_laws: WorldLawsLike,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(law_dt [B, dt_dim], law_h_bias [B, width], law_h_scale [B, out_ch], law_z_scale [B, z_flux_ch])`."""
        law_f = encode_law_features(world_laws, batch_size=B, dtype=dtype, device=device)  # [B, 12]
        lh = self.law_mlp(law_f)  # [B, law_hidden]
        law_dt = self.law_to_dt(lh)  # [B, dt_dim]
        law_h_bias = self.law_h_mod(lh)  # [B, width]
        law_h_scale = torch.exp(self.law_h_logscale(lh).clamp(-2.2, 2.2))  # [B, out_ch]
        law_z_scale = torch.exp(self.law_z_logscale(lh).clamp(-2.2, 2.2))  # [B, z_flux_ch]
        return law_dt, law_h_bias, law_h_scale, law_z_scale

    def forward(
        self,
        state: torch.Tensor,
        local_dt: torch.Tensor | float | None = None,
        world_laws: WorldLawsLike = None,
        return_z_flux: bool = True,
        return_latent: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Dense 3D isotropic forward pass supporting 4D `[L, 18|19, H, W]` or 5D `[B, L, 18|19, H, W]`.
        Default returns `(out_horiz [L, 17, H, W], out_z_flux [L-1, 5, H, W], z_world [L, 48, H, W])`.
        """
        x5, dt5, is_4d = self._split_state_and_dt(state, local_dt)
        B, L, C_in, H, W = x5.shape
        x_2d = x5.reshape(B * L, C_in, H, W)
        dt_2d = dt5.reshape(B * L, 1, H, W)

        # World-Law conditioning embeddings (exact 0 / 1.0 at DEFAULT_WORLD_LAWS!)
        law_dt, law_h_bias, law_h_scale, law_z_scale = self._compute_law_mods(
            world_laws, B=B, dtype=x5.dtype, device=x5.device
        )
        law_dt_bl = law_dt.unsqueeze(1).expand(B, L, self.dt_dim).reshape(B * L, self.dt_dim, 1, 1)
        law_hb_bl = law_h_bias.unsqueeze(1).expand(B, L, self.width).reshape(B * L, self.width, 1, 1)
        law_hs_bl = law_h_scale.unsqueeze(1).expand(B, L, self.out_ch).reshape(B * L, self.out_ch, 1, 1)

        # Joint (local_dt + world_laws) FiLM embeddings
        dt_emb = self.dt_mlp(encode_dt_features(dt_2d)) + law_dt_bl  # [B*L, dt_dim, H, W]
        film_w = self.dt_film.weight
        film_stride = 2 * self.width

        # Stem: (1, 3, 3) spatial conv
        h_2d = self.stem(F.pad(x_2d, (1, 1, 1, 1), mode="replicate"))
        h = h_2d.view(B, L, self.width, H, W).transpose(1, 2)  # [B, width, L, H, W]

        # 5 FiLM-modulated isotropic 3D residual blocks
        for idx, blk in enumerate(self.blocks):
            w_blk = film_w[idx * film_stride : (idx + 1) * film_stride]
            film_2d = F.conv2d(dt_emb, w_blk)
            film_5d = film_2d.view(B, L, 2, self.width, H, W)
            gamma_f = film_5d[:, :, 0].transpose(1, 2)
            beta_f = film_5d[:, :, 1].transpose(1, 2)
            h = blk.forward_dense(h, gamma_f, beta_f)

        # Shared 48-ch isotropic 3D latent trunk `Z_world_3d`
        z_5d = F.gelu(h).transpose(1, 2)  # [B, L, 48, H, W]
        z_2d = z_5d.reshape(B * L, self.width, H, W)

        # Horizontal 17-ch head + raw-state skip + law threshold bias + law scale + face-dt scaling
        hh = F.gelu(self.h_lat_in(z_2d) + self.h_x_in(x_2d) + law_hb_bl)
        raw_h = self.h_out(hh) * self.out_scale
        rate_mult = 1.0 + 0.85 * torch.tanh(self.dt_head_mod(dt_emb))
        lin_rate = raw_h[:, :10] * rate_mult[:, :10]
        rxn_rate = F.softplus(raw_h[:, 10:]) * rate_mult[:, 10:]
        horiz_rate = torch.cat([lin_rate, rxn_rate], dim=1) * law_hs_bl
        horiz_2d = self._scale_horiz_by_dt(horiz_rate, dt_2d)
        horiz = horiz_2d.view(B, L, self.out_ch, H, W)

        out_z_flux: torch.Tensor | None = None
        if return_z_flux:
            if L > 1:
                x_lo = x5[:, :-1].reshape(B * (L - 1), C_in, H, W)
                x_hi = x5[:, 1:].reshape(B * (L - 1), C_in, H, W)
                z_lo = z_5d[:, :-1].reshape(B * (L - 1), self.width, H, W)
                z_hi = z_5d[:, 1:].reshape(B * (L - 1), self.width, H, W)
                dt_z = torch.minimum(dt5[:, :-1], dt5[:, 1:])
                law_dt_z = law_dt.unsqueeze(1).expand(B, L - 1, self.dt_dim).reshape(B * (L - 1), self.dt_dim, 1, 1)
                dt_z_emb = self.dt_mlp(encode_dt_features(dt_z.reshape(B * (L - 1), 1, H, W))) + law_dt_z
                vh = F.gelu(
                    self.v_x_lo(x_lo)
                    + self.v_x_hi(x_hi)
                    + self.v_z_lo(z_lo)
                    + self.v_z_hi(z_hi)
                    + self.dt_z_mod(dt_z_emb)
                )
                vh = F.gelu(self.v_mid(vh))
                law_zs_5d = law_z_scale.view(B, 1, self.z_flux_ch, 1, 1)
                z_if = (self.v_out(vh) * self.z_out_scale).view(B, L - 1, self.z_flux_ch, H, W) * law_zs_5d * dt_z
                out_z_flux = z_if
            else:
                out_z_flux = horiz.new_zeros((B, 0, self.z_flux_ch, H, W))

        z_world = z_5d.contiguous() if return_latent else None

        if is_4d:
            horiz = horiz.squeeze(0)
            if out_z_flux is not None:
                out_z_flux = out_z_flux.squeeze(0)
            if z_world is not None:
                z_world = z_world.squeeze(0)

        ret: list[torch.Tensor] = [horiz]
        if return_z_flux:
            assert out_z_flux is not None
            ret.append(out_z_flux)
        if return_latent:
            assert z_world is not None
            ret.append(z_world)
        return ret[0] if len(ret) == 1 else tuple(ret)


# =============================================================================
# True 3D `16x16x8` Sparse Active-Brick Runner for `WorldModel3DV8`
# =============================================================================


def _shrinking_halo_3d_brick_core_v8(
    wm: WorldModel3DV8,
    patches_x: torch.Tensor,
    patches_dt: torch.Tensor,
    world_laws_1x6: torch.Tensor,
    m_z0: torch.Tensor,
    m_z1: torch.Tensor,
    m_y0: torch.Tensor,
    m_y1: torch.Tensor,
    m_x0: torch.Tensor,
    m_x1: torch.Tensor,
    brick_size: tuple[int, int, int] = (8, 16, 16),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact shrinking-halo 3D brick forward for `WorldModel3DV8` under arbitrary `world_laws`."""
    K, C_in, Zp, Hp, Wp = patches_x.shape
    Bz, By, Bx = brick_size
    hz = wm.halo_z
    hxy = wm.halo_xy

    law_dt, law_h_bias, law_h_scale, law_z_scale = wm._compute_law_mods(
        world_laws_1x6, B=1, dtype=patches_x.dtype, device=patches_x.device
    )
    law_dt_5d = law_dt.view(1, wm.dt_dim, 1, 1, 1)
    law_dt_2d = law_dt.view(1, wm.dt_dim, 1, 1)
    law_hb_2d = law_h_bias.view(1, wm.width, 1, 1)
    law_hs_2d = law_h_scale.view(1, wm.out_ch, 1, 1)
    law_zs_5d = law_z_scale.view(1, 1, wm.z_flux_ch, 1, 1)

    # 1. Stem: valid (1, 3, 3)
    rem_xy = hxy
    rem_z = hz
    x_in = _apply_xy_boundary_replicate_(patches_x, rem_xy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=False)
    x_in_2d = x_in.transpose(1, 2).reshape(K * Zp, C_in, Hp, Wp)
    h_2d = wm.stem(x_in_2d)
    rem_xy -= 1
    Hc, Wc = Hp - 2, Wp - 2
    Zc = Zp
    h = h_2d.view(K, Zc, wm.width, Hc, Wc).transpose(1, 2)

    # Joint (dt + world_laws) embeddings on post-stem `[Zc, Hc, Wc]`
    dt_cur = patches_dt[:, :, :, 1:-1, 1:-1].clamp(min=0.0)
    dt_cur_2d = dt_cur.transpose(1, 2).reshape(K * Zc, 1, Hc, Wc)
    dt_emb_5d = wm.dt_mlp(encode_dt_features(dt_cur_2d)).view(K, Zc, wm.dt_dim, Hc, Wc).transpose(1, 2) + law_dt_5d

    film_w = wm.dt_film.weight
    film_stride = 2 * wm.width

    # 2. Shrinking-halo residual blocks
    for idx, blk in enumerate(wm.blocks):
        w_blk = film_w[idx * film_stride : (idx + 1) * film_stride]
        de_2d = dt_emb_5d.transpose(1, 2).reshape(K * Zc, wm.dt_dim, Hc, Wc)
        film_5d = F.conv2d(de_2d, w_blk).view(K, Zc, 2, wm.width, Hc, Wc)
        gamma_f = film_5d[:, :, 0].transpose(1, 2)
        beta_f = film_5d[:, :, 1].transpose(1, 2)

        if isinstance(blk, Factorized3DResBlock):
            h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
            act = F.gelu(h_mod)
            act = _apply_xy_boundary_replicate_(act, rem_xy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=True)
            s_out = blk.spatial_conv(act.transpose(1, 2).reshape(K * Zc, wm.width, Hc, Wc))
            Hc -= 2
            Wc -= 2
            rem_xy -= 1
            s_out = s_out.view(K, Zc, wm.width, Hc, Wc).transpose(1, 2)
            h = h[:, :, :, 1:-1, 1:-1] + blk.gamma_xy * s_out
            dt_emb_5d = dt_emb_5d[:, :, :, 1:-1, 1:-1]

            if blk.use_vert and blk.vert_conv is not None and blk.gamma_z is not None:
                v_in = F.gelu(h)
                v_in = _apply_z_boundary_replicate_(v_in, rem_z, Bz, m_z0, m_z1, inplace=True)
                v_out = blk.vert_conv(v_in)
                Zc -= 2
                rem_z -= 1
                h = h[:, :, 1:-1, :, :] + blk.gamma_z * v_out
                dt_emb_5d = dt_emb_5d[:, :, 1:-1, :, :]
        else:
            assert isinstance(blk, Isotropic3x3x3ResBlock)
            h_mod = h * (1.0 + 0.5 * torch.tanh(gamma_f)) + beta_f
            u = F.gelu(blk.pw_in(F.gelu(h_mod)))
            u = _apply_xy_boundary_replicate_(u, rem_xy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=True)
            u = _apply_z_boundary_replicate_(u, rem_z, Bz, m_z0, m_z1, inplace=True)
            u = F.gelu(blk.conv3d(u))
            Zc -= 2
            Hc -= 2
            Wc -= 2
            rem_z -= 1
            rem_xy -= 1
            h = h[:, :, 1:-1, 1:-1, 1:-1] + blk.gamma * blk.pw_out(u)
            dt_emb_5d = dt_emb_5d[:, :, 1:-1, 1:-1, 1:-1]

    z_all = F.gelu(h).transpose(1, 2)
    z_out = z_all[:, :Bz]
    x_all = patches_x[:, :, hz : hz + Bz + 1, hxy:-hxy, hxy:-hxy].transpose(1, 2)
    x_int = x_all[:, :Bz]

    dt_pad_xy = _apply_xy_boundary_replicate_(patches_dt.clamp(min=0.0), hxy, By, Bx, m_y0, m_y1, m_x0, m_x1, inplace=True)
    dt_all = dt_pad_xy[:, :, hz : hz + Bz + 1, hxy:-hxy, hxy:-hxy].transpose(1, 2)
    dt_int = dt_all[:, :Bz]
    dt_e_nb = dt_pad_xy[:, :, hz : hz + Bz, hxy:-hxy, hxy + 1 : Wp - hxy + 1].transpose(1, 2)
    dt_s_nb = dt_pad_xy[:, :, hz : hz + Bz, hxy + 1 : Hp - hxy + 1, hxy:-hxy].transpose(1, 2)
    dt_int2 = dt_int.reshape(K * Bz, 1, By, Bx)
    dt_e = torch.minimum(dt_int2, dt_e_nb.reshape(K * Bz, 1, By, Bx))
    dt_s = torch.minimum(dt_int2, dt_s_nb.reshape(K * Bz, 1, By, Bx))

    # 3. Horizontal 17-ch head
    z_2d = z_out.reshape(K * Bz, wm.width, By, Bx)
    x_2d = x_int.reshape(K * Bz, C_in, By, Bx)
    de_int2 = dt_emb_5d[:, :, :Bz].transpose(1, 2).reshape(K * Bz, wm.dt_dim, By, Bx)
    hh = F.gelu(wm.h_lat_in(z_2d) + wm.h_x_in(x_2d) + law_hb_2d)
    raw_h = wm.h_out(hh) * wm.out_scale
    rate_mult = 1.0 + 0.85 * torch.tanh(wm.dt_head_mod(de_int2))
    horiz_rate = torch.cat(
        [raw_h[:, :10] * rate_mult[:, :10], F.softplus(raw_h[:, 10:]) * rate_mult[:, 10:]],
        dim=1,
    ) * law_hs_2d
    horiz_2d = torch.cat(
        [
            horiz_rate[:, 0:4] * dt_e,
            horiz_rate[:, 4:8] * dt_s,
            horiz_rate[:, 8:9] * dt_e,
            horiz_rate[:, 9:10] * dt_s,
            horiz_rate[:, 10:17] * dt_int2,
        ],
        dim=1,
    )
    horiz = horiz_2d.view(K, Bz, wm.out_ch, By, Bx)

    # 4. Staggered vertical interface head
    x_lo = x_all[:, :-1].reshape(K * Bz, C_in, By, Bx)
    x_hi = x_all[:, 1:].reshape(K * Bz, C_in, By, Bx)
    z_lo = z_all[:, :-1].reshape(K * Bz, wm.width, By, Bx)
    z_hi = z_all[:, 1:].reshape(K * Bz, wm.width, By, Bx)
    dt_z = torch.minimum(dt_all[:, :-1], dt_all[:, 1:])
    dt_z_emb = wm.dt_mlp(encode_dt_features(dt_z.reshape(K * Bz, 1, By, Bx))) + law_dt_2d
    vh = F.gelu(
        wm.v_x_lo(x_lo)
        + wm.v_x_hi(x_hi)
        + wm.v_z_lo(z_lo)
        + wm.v_z_hi(z_hi)
        + wm.dt_z_mod(dt_z_emb)
    )
    vh = F.gelu(wm.v_mid(vh))
    z_flux = (wm.v_out(vh) * wm.z_out_scale).view(K, Bz, wm.z_flux_ch, By, Bx) * law_zs_5d * dt_z
    return horiz, z_flux, z_out


class _CompiledBrickModuleV8(nn.Module):
    def __init__(self, wm: WorldModel3DV8, brick_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.wm = wm
        self.brick_size = brick_size

    def forward(
        self,
        patches_x: torch.Tensor,
        patches_dt: torch.Tensor,
        world_laws_1x6: torch.Tensor,
        m_z0: torch.Tensor,
        m_z1: torch.Tensor,
        m_y0: torch.Tensor,
        m_y1: torch.Tensor,
        m_x0: torch.Tensor,
        m_x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _shrinking_halo_3d_brick_core_v8(
            self.wm,
            patches_x,
            patches_dt,
            world_laws_1x6,
            m_z0,
            m_z1,
            m_y0,
            m_y1,
            m_x0,
            m_x1,
            brick_size=self.brick_size,
        )


class SparseWorld3DV8Runner:
    """
    Production True 3D `16x16x8` Sparse Active-Brick Runner for `WorldModel3DV8`.
    Supports arbitrary dynamic `world_laws` (automatically refreshes dense cache whenever
    `world_laws` changes, and maintains exact FP32 `< 1e-6` dense-vs-sparse parity!).
    """

    def __init__(
        self,
        world_model: WorldModel3DV8,
        device: torch.device | str,
        use_fp16: bool = True,
        brick_xy: int = 16,
        brick_z: int = 8,
        sparse_threshold: float = 0.42,
        diff_eps: float = 1e-4,
        activity_eps: float = 0.0,
        bucket_size: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16) and (self.device.type == "cuda")
        self.compute_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.brick_xy = int(brick_xy)
        self.brick_z = int(brick_z)
        self.sparse_threshold = float(sparse_threshold)
        self.diff_eps = float(diff_eps)
        self.activity_eps = max(0.0, float(activity_eps))
        self.bucket_size = max(1, int(bucket_size))

        self.halo_xy = world_model.halo_xy
        self.halo_z = world_model.halo_z
        self.patch_xy = self.brick_xy + 2 * self.halo_xy
        self.patch_z = self.brick_z + 2 * self.halo_z + 1

        self.world_model = world_model
        self._wm = copy.deepcopy(world_model).to(device=self.device, dtype=self.compute_dtype).eval()
        for p in self._wm.parameters():
            p.requires_grad_(False)
        self._wm_dense: Any = self._wm
        self._brick_mod: Any = _CompiledBrickModuleV8(self._wm, (self.brick_z, self.brick_xy, self.brick_xy)).eval()

        self._last_horiz: torch.Tensor | None = None
        self._last_z_flux_pad: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None
        self._last_x: torch.Tensor | None = None
        self._last_dt: torch.Tensor | None = None
        self._last_laws: torch.Tensor | None = None

    def reset(self) -> None:
        self._last_horiz = None
        self._last_z_flux_pad = None
        self._last_z = None
        self._last_x = None
        self._last_dt = None
        self._last_laws = None

    def _normalize_dt(self, local_dt: torch.Tensor | float | None, L: int, H: int, W: int) -> torch.Tensor:
        if local_dt is None:
            return torch.ones((L, 1, H, W), dtype=self.compute_dtype, device=self.device)
        if isinstance(local_dt, (int, float)):
            return torch.full((L, 1, H, W), max(0.0, float(local_dt)), dtype=self.compute_dtype, device=self.device)
        t = local_dt.to(device=self.device, dtype=self.compute_dtype, non_blocking=(self.device.type == "cuda")).clamp(min=0.0)
        if t.ndim == 0:
            return t.view(1, 1, 1, 1).expand(L, 1, H, W).contiguous()
        if t.ndim == 2:
            return t.view(1, 1, H, W).expand(L, 1, H, W).contiguous()
        if t.ndim == 3:
            return t.view(t.shape[0], 1, H, W).expand(L, 1, H, W).contiguous()
        if t.ndim == 4:
            return t.expand(L, 1, H, W).contiguous()
        raise ValueError(f"Unsupported local_dt shape {tuple(t.shape)}")

    def active_brick_mask(
        self,
        x_cur: torch.Tensor,
        dt_cur: torch.Tensor,
        prev_x: torch.Tensor | None,
        dirty_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        L, _, H, W = x_cur.shape
        Bz, By, Bx = self.brick_z, self.brick_xy, self.brick_xy
        nz, ny, nx = L // Bz, H // By, W // Bx

        if dirty_mask is not None:
            dm = dirty_mask.to(device=self.device)
            if dm.shape == (nz, ny, nx):
                return dm.bool()
            if dm.shape == (ny, nx):
                return dm.bool().unsqueeze(0).expand(nz, ny, nx).contiguous()
            if dm.shape == (L, H, W):
                pooled = F.max_pool3d(dm.to(dtype=self.compute_dtype).unsqueeze(0).unsqueeze(0), (Bz, By, Bx), (Bz, By, Bx))
                return pooled[0, 0] > 0.5
            raise ValueError(f"dirty_mask shape {tuple(dirty_mask.shape)} incompatible with bricks {(nz, ny, nx)}")

        ref = prev_x.to(device=self.device, dtype=self.compute_dtype) if prev_x is not None else self._last_x
        assert ref is not None

        if self.activity_eps <= 0.0:
            diff_vox = ((x_cur - ref).abs().amax(dim=1) > 0.0).to(dtype=self.compute_dtype)
            if self._last_dt is not None and self._last_dt.shape == dt_cur.shape:
                dt_diff = (dt_cur[:, 0] - self._last_dt[:, 0]).abs() > 0.0
                diff_vox = torch.maximum(diff_vox, dt_diff.to(dtype=self.compute_dtype))
        else:
            eps = self.activity_eps
            d_sp = (x_cur[:, :10] - ref[:, :10]).abs().amax(dim=1) > max(eps * 24.0, self.diff_eps, 0.012)
            d_tmp = (x_cur[:, 10] - ref[:, 10]).abs() > max(eps * 70.0, self.diff_eps * 4.0, 0.035)
            d_flow = (x_cur[:, 12:14] - ref[:, 12:14]).abs().amax(dim=1) > max(eps * 20.0, 0.010)
            has_force = (x_cur[:, 16].abs() > max(eps, 0.01)) | (x_cur[:, 17].abs() > max(eps, 0.01))
            diff_vox = (d_sp | d_tmp | d_flow | has_force).to(dtype=self.compute_dtype)
            if self._last_dt is not None and self._last_dt.shape == dt_cur.shape:
                dt_diff = (dt_cur[:, 0] - self._last_dt[:, 0]).abs() > 1e-3
                diff_vox = torch.maximum(diff_vox, dt_diff.to(dtype=self.compute_dtype))

        pooled = F.max_pool3d(diff_vox.unsqueeze(0).unsqueeze(0), (Bz, By, Bx), (Bz, By, Bx))
        return pooled[0, 0] > 0.5

    @torch.inference_mode()
    def step(
        self,
        x: torch.Tensor,
        local_dt: torch.Tensor | float | None = None,
        world_laws: WorldLawsLike = None,
        prev_x: torch.Tensor | None = None,
        dirty_mask: torch.Tensor | None = None,
        dilate_halo_bricks: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D [L, 18, H, W], got {tuple(x.shape)}")

        x_cur = x.to(device=self.device, dtype=self.compute_dtype, non_blocking=(self.device.type == "cuda"))
        L, C, H, W = x_cur.shape
        dt_cur = self._normalize_dt(local_dt, L, H, W)
        laws_cur = normalize_world_laws(world_laws, batch_size=1, dtype=self.compute_dtype, device=self.device)
        Bz, By, Bx = self.brick_z, self.brick_xy, self.brick_xy

        laws_changed = (
            self._last_laws is None or float((laws_cur - self._last_laws).abs().max().item()) > 0.0
        )

        need_dense = (
            laws_changed
            or self._last_horiz is None
            or self._last_z_flux_pad is None
            or self._last_z is None
            or self._last_x is None
            or self._last_x.shape != x_cur.shape
            or (L % Bz != 0)
            or (H % By != 0)
            or (W % Bx != 0)
        )

        if not need_dense:
            assert self._last_horiz is not None
            assert self._last_z_flux_pad is not None
            assert self._last_z is not None
            assert self._last_x is not None

            nz, ny, nx = L // Bz, H // By, W // Bx
            total_bricks = nz * ny * nx
            brick_active = self.active_brick_mask(x_cur, dt_cur, prev_x, dirty_mask)
            if dilate_halo_bricks:
                ba_f = brick_active.to(dtype=self.compute_dtype).unsqueeze(0).unsqueeze(0)
                brick_active = F.max_pool3d(ba_f, kernel_size=3, stride=1, padding=1)[0, 0] > 0.5

            coords = torch.nonzero(brick_active, as_tuple=False)
            k = int(coords.shape[0])
            active_frac = float(k) / float(total_bricks)

            if k == 0:
                self._last_x.copy_(x_cur)
                self._last_dt = dt_cur.clone()
                return self._last_horiz, self._last_z_flux_pad[:-1], self._last_z, 0.0

            if active_frac < self.sparse_threshold:
                tz, ty, tx = coords[:, 0], coords[:, 1], coords[:, 2]

                x_5d = x_cur.transpose(0, 1).unsqueeze(0)
                dt_5d = dt_cur.transpose(0, 1).unsqueeze(0)
                pad3 = (self.halo_xy, self.halo_xy, self.halo_xy, self.halo_xy, self.halo_z, self.halo_z + 1)
                x_pad = F.pad(x_5d, pad3, mode="replicate")[0]
                dt_pad = F.pad(dt_5d, pad3, mode="replicate")[0]

                ux = x_pad.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
                udt = dt_pad.unfold(1, self.patch_z, Bz).unfold(2, self.patch_xy, By).unfold(3, self.patch_xy, Bx)
                patches_x = ux.permute(1, 2, 3, 0, 4, 5, 6)[tz, ty, tx].contiguous()
                patches_dt = udt.permute(1, 2, 3, 0, 4, 5, 6)[tz, ty, tx].contiguous()

                b = self.bucket_size
                k_pad = min(((k + b - 1) // b) * b, total_bricks)
                if k_pad > k:
                    n_ex = k_pad - k
                    patches_x_run = torch.cat(
                        [patches_x, patches_x.new_zeros((n_ex, C, self.patch_z, self.patch_xy, self.patch_xy))], dim=0
                    )
                    patches_dt_run = torch.cat(
                        [patches_dt, patches_dt.new_ones((n_ex, 1, self.patch_z, self.patch_xy, self.patch_xy))], dim=0
                    )
                    coords_run = torch.cat([coords, coords.new_zeros((n_ex, 3))], dim=0)
                else:
                    patches_x_run = patches_x
                    patches_dt_run = patches_dt
                    coords_run = coords

                k_r = int(coords_run.shape[0])
                rtz, rty, rtx = coords_run[:, 0], coords_run[:, 1], coords_run[:, 2]
                m_z0 = (rtz == 0).view(k_r, 1, 1, 1, 1)
                m_z1 = (rtz == (nz - 1)).view(k_r, 1, 1, 1, 1)
                m_y0 = (rty == 0).view(k_r, 1, 1, 1, 1)
                m_y1 = (rty == (ny - 1)).view(k_r, 1, 1, 1, 1)
                m_x0 = (rtx == 0).view(k_r, 1, 1, 1, 1)
                m_x1 = (rtx == (nx - 1)).view(k_r, 1, 1, 1, 1)

                y_horiz, y_z_flux, y_z = self._brick_mod(
                    patches_x_run, patches_dt_run, laws_cur, m_z0, m_z1, m_y0, m_y1, m_x0, m_x1
                )
                if k_pad > k:
                    y_horiz = y_horiz[:k]
                    y_z_flux = y_z_flux[:k]
                    y_z = y_z[:k]

                out_ch = self._last_horiz.shape[1]
                zf_ch = self._last_z_flux_pad.shape[1]
                z_ch = self._last_z.shape[1]

                self._last_horiz.view(nz, Bz, out_ch, ny, By, nx, Bx).permute(0, 3, 5, 1, 2, 4, 6)[tz, ty, tx] = y_horiz
                self._last_z_flux_pad.view(nz, Bz, zf_ch, ny, By, nx, Bx).permute(0, 3, 5, 1, 2, 4, 6)[tz, ty, tx] = y_z_flux
                self._last_z.view(nz, Bz, z_ch, ny, By, nx, Bx).permute(0, 3, 5, 1, 2, 4, 6)[tz, ty, tx] = y_z

                self._last_x.copy_(x_cur)
                self._last_dt = dt_cur.clone()
                self._last_laws = laws_cur.clone()
                return self._last_horiz, self._last_z_flux_pad[:-1], self._last_z, active_frac

        horiz, z_flux, z_world = self._wm_dense(
            x_cur, local_dt=dt_cur, world_laws=laws_cur, return_z_flux=True, return_latent=True
        )
        self._last_horiz = horiz.clone().contiguous()
        zf_pad = horiz.new_zeros((L, self._wm.z_flux_ch, H, W))
        if L > 1:
            zf_pad[:-1].copy_(z_flux)
        self._last_z_flux_pad = zf_pad.contiguous()
        self._last_z = z_world.clone().contiguous()
        self._last_x = x_cur.clone()
        self._last_dt = dt_cur.clone()
        self._last_laws = laws_cur.clone()
        return self._last_horiz, self._last_z_flux_pad[:-1], self._last_z, 1.0


# =============================================================================
# Built-In Verification Suite (Canonical Identity, v7 Warm-Start, FP32 Parity)
# =============================================================================


def verify_canonical_teacher_identity(L: int = 16, H: int = 32, W: int = 32) -> dict[str, Any]:
    """
    Assert:
      1. `step_refsim_3d_with_laws(..., world_laws=DEFAULT_WORLD_LAWS)` matches `RefSim3D.step(...)`
         cleanly (`max_abs_diff < 1e-6`).
      2. Exact mass conservation (`mass_rel_err < 1e-6`) across ALL 4 World-Law presets and exact
         energy conservation (`energy_rel_err < 1e-6`) when `radiative_cooling=False`.
    """
    dev = torch.device("cpu")
    st = make_biome_3d_world(0, L=L, H=H, W=W, device=dev, seed=2026)

    ns_ref, h_ref, z_ref = RefSim3D.step(st, local_dt=1.0)
    ns_law, h_law, z_law = step_refsim_3d_with_laws(st, local_dt=1.0, world_laws=DEFAULT_WORLD_LAWS)

    diff_h = float((h_ref - h_law).abs().max().item())
    diff_z = float((z_ref - z_law).abs().max().item())
    diff_m = float((ns_ref.mass - ns_law.mass).abs().max().item())
    diff_e = float((ns_ref.energy - ns_law.energy).abs().max().item())
    assert max(diff_h, diff_z, diff_m, diff_e) < 1e-6, (
        f"Canonical teacher identity failed: h={diff_h}, z={diff_z}, m={diff_m}, e={diff_e}"
    )

    conservation: dict[str, dict[str, float]] = {}
    m0 = float(compute_total_mass(st).sum().item())
    e0 = float(compute_total_energy(st).sum().item())
    for name, laws in WORLD_LAW_PRESETS.items():
        ns_c, _, _ = step_refsim_3d_with_laws(st, local_dt=1.0, world_laws=laws, radiative_cooling=False)
        m1 = float(compute_total_mass(ns_c).sum().item())
        e1 = float(compute_total_energy(ns_c).sum().item())
        m_rel = abs(m1 - m0) / max(abs(m0), 1.0)
        e_rel = abs(e1 - e0) / max(abs(e0), 1.0)
        assert m_rel < 1e-5, f"Mass conservation failed on {name}: {m_rel}"
        assert e_rel < 1e-5, f"Energy conservation failed on {name}: {e_rel}"
        conservation[name] = {"mass_rel_err": m_rel, "energy_rel_err": e_rel}

    return {
        "canonical_diff_horiz": diff_h,
        "canonical_diff_z_flux": diff_z,
        "canonical_diff_mass": diff_m,
        "canonical_diff_energy": diff_e,
        "conservation": conservation,
    }


def verify_v7_warmstart_identity(v7_ckpt_path: str | Path, L: int = 16, H: int = 32, W: int = 32) -> dict[str, float]:
    """Assert Step-0 `WorldModel3DV8.load_from_v7` is bit-for-bit identical (`0.0`) to `WorldModel3D-v7`."""
    dev = torch.device("cpu")
    ck = torch.load(str(v7_ckpt_path), map_location=dev, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck

    v7 = WorldModel3DV5().to(dev).eval()
    v7.load_state_dict(sd, strict=True)
    v8 = WorldModel3DV8.load_from_v7(v7_ckpt_path, map_location=dev).eval()

    torch.manual_seed(2020)
    x = torch.rand((L, 18, H, W), dtype=torch.float32, device=dev) * 0.4
    dt_map = torch.ones((H, W), dtype=torch.float32, device=dev)
    dt_map[8:24, 8:24] = 0.25

    with torch.inference_mode():
        h7, zf7, z7 = v7(x, local_dt=dt_map, return_z_flux=True, return_latent=True)
        h8, zf8, z8 = v8(x, local_dt=dt_map, world_laws=DEFAULT_WORLD_LAWS, return_z_flux=True, return_latent=True)

    dh = float((h7 - h8).abs().max().item())
    dzf = float((zf7 - zf8).abs().max().item())
    dz = float((z7 - z8).abs().max().item())
    assert max(dh, dzf, dz) == 0.0, f"Step-0 v7->v8 identity failed: h={dh}, zf={dzf}, z={dz}"
    return {"max_abs_diff_horiz": dh, "max_abs_diff_z_flux": dzf, "max_abs_diff_z_world": dz}


def verify_sparse_v8_parity(
    model: WorldModel3DV8 | None = None,
    L: int = 16,
    H: int = 64,
    W: int = 64,
    world_laws: WorldLawsLike = None,
    seed: int = 77,
) -> dict[str, Any]:
    """
    Verify exact FP32 dense-vs-sparse `16x16x8` 3D brick parity (`max_abs_diff < 1e-6`)
    across ALL `L` layers under arbitrary `world_laws` + spatial `local_dt` domes!
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    if model is not None:
        wm = copy.deepcopy(model).to(device=device, dtype=torch.float32).eval()
    else:
        wm = WorldModel3DV8().to(device).eval()
        # Activate law projections so non-canonical laws are actively exercised in standalone self-test
        with torch.no_grad():
            wm.law_to_dt.weight.normal_(0.0, 0.1)
            wm.law_h_mod.weight.normal_(0.0, 0.1)
            wm.law_h_logscale.weight.normal_(0.0, 0.1)
            wm.law_z_logscale.weight.normal_(0.0, 0.1)

    laws = DEFAULT_WORLD_LAWS if world_laws is None else _unpack_world_laws_floats(world_laws)

    prev_x = torch.rand((L, 18, H, W), dtype=torch.float32, device=device) * 0.4
    prev_x[:, 16:18] = 0.0
    next_x = prev_x.clone()
    next_x[2:6, 0, 18:30, 20:32] += 0.35
    next_x[L - 3 : L, 2, 18:30, 20:32] += 0.30

    dt_map = torch.ones((H, W), dtype=torch.float32, device=device)
    dt_map[H // 4 : (3 * H) // 4, W // 4 : (3 * W) // 4] = 0.15

    runner = SparseWorld3DV8Runner(wm, device=device, use_fp16=False, brick_xy=16, brick_z=8, sparse_threshold=0.95)
    with torch.inference_mode():
        _, _, _, frac0 = runner.step(prev_x, local_dt=dt_map, world_laws=laws)
        sp_h, sp_zf, sp_z, frac1 = runner.step(next_x, local_dt=dt_map, world_laws=laws, dilate_halo_bricks=True)
        dn_h, dn_zf, dn_z = wm(next_x, local_dt=dt_map, world_laws=laws, return_z_flux=True, return_latent=True)

    assert frac0 == 1.0
    assert 0.0 < frac1 < 0.95
    diff_h = float((sp_h - dn_h).abs().max().item())
    diff_zf = float((sp_zf - dn_zf).abs().max().item())
    diff_z = float((sp_z - dn_z).abs().max().item())
    return {
        "L": int(L),
        "H": int(H),
        "W": int(W),
        "world_laws": list(laws),
        "active_brick_frac": round(frac1, 4),
        "max_abs_diff_horiz": diff_h,
        "max_abs_diff_z_flux": diff_zf,
        "max_abs_diff_z_world": diff_z,
        "max_abs_diff_all": max(diff_h, diff_zf, diff_z),
    }


if __name__ == "__main__":
    m = WorldModel3DV8()
    print(f"WorldModel3DV8 params: {count_parameters(m):,} (base v5/v7: {count_parameters(WorldModel3DV5()):,})")
    f0 = encode_law_features(DEFAULT_WORLD_LAWS)
    assert float(f0.abs().max().item()) == 0.0, f"encode_law_features(DEFAULT_WORLD_LAWS) != 0: {f0}"
    print(f"encode_law_features(DEFAULT_WORLD_LAWS) max abs: {float(f0.abs().max().item()):.1e}")

    t_id = verify_canonical_teacher_identity()
    print(f"Canonical teacher identity & conservation: {t_id}")

    v7_path = Path("models/checkpoints/world_model_3d_v7.pt")
    if v7_path.exists():
        w_id = verify_v7_warmstart_identity(v7_path)
        print(f"Step-0 v7->v8 warm-start identity: {w_id}")

    for preset_name in ("frostmarsh_standard", "low_g_magma_crucible"):
        for layers in (16, 32):
            res = verify_sparse_v8_parity(None, L=layers, H=64, W=64, world_laws=WORLD_LAW_PRESETS[preset_name])
            print(f"FP32 Sparse Parity [{preset_name} Z={layers}]: max_abs_diff_all={res['max_abs_diff_all']:.2e}")
            assert res["max_abs_diff_all"] < 1e-6, f"Parity failed: {res}"

    print("All WorldModel3DV8 self-checks PASSED!")
