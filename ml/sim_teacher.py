"""
Frostmarsh Neural World Model POC — Reference Physics Teacher & Exact Projection.

Matches the shared Rust `field-core` contract bit-for-bit in schema:
- 10 species: water, ice, steam, soil, rock, wood, ash, smoke, oil, plant
- 4 mobile species: [0: water, 2: steam, 7: smoke, 8: oil]
- 7 reactions: Freeze, Melt, Boil, Condense, WoodBurn, OilBurn, PlantBurn
- Input state: [B, 18, H, W] float32
- Output deltas: [B, 17, H, W] float32
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

# =============================================================================
# Shared Physical & Schema Constants
# =============================================================================
DT = 0.1  # 10 Hz
C_NOM = 1000.0
H_CAP = 4000.0

T_AMB = 29000.0  # 290 K in cK
T_MIN = 15000.0  # 150 K in cK
T_MAX = 150000.0  # 1500 K in cK

NUM_SPECIES = 10
NUM_MOBILE = 4
NUM_RXNS = 7

# Species indices
S_WATER = 0
S_ICE = 1
S_STEAM = 2
S_SOIL = 3
S_ROCK = 4
S_WOOD = 5
S_ASH = 6
S_SMOKE = 7
S_OIL = 8
S_PLANT = 9

SPECIES_NAMES = (
    "water", "ice", "steam", "soil", "rock",
    "wood", "ash", "smoke", "oil", "plant",
)

MOBILE_IDX = (S_WATER, S_STEAM, S_SMOKE, S_OIL)

# Specific heat c, thermal conductivity k, chemical energy u0
SPEC_C_LIST = [42.0, 21.0, 20.0, 15.0, 10.0, 17.0, 10.0, 12.0, 20.0, 25.0]
SPEC_K_LIST = [6.0, 22.0, 2.0, 10.0, 20.0, 3.0, 2.0, 2.0, 3.0, 4.0]
SPEC_U0_LIST = [
    0.0, -336000.0, 840000.0, 0.0, 0.0,
    450000.0, 0.0, 0.0, 750000.0, 300000.0,
]

# Reactions:
# 0: Freeze    (water -1, ice +1,             T < 27315, dE_th = +336000)
# 1: Melt      (ice -1, water +1,             T > 27315, dE_th = -336000)
# 2: Boil      (water -1, steam +1,           T > 37315, dE_th = -840000)
# 3: Condense  (steam -1, water +1,           T < 37000, dE_th = +840000)
# 4: WoodBurn  (wood -2, ash +1, smoke +1,    T > 52000, dE_th = +900000)
# 5: OilBurn   (oil -1, smoke +1,             T > 44000, dE_th = +750000)
# 6: PlantBurn (plant -2, ash +1, smoke +1,   T > 46000, dE_th = +600000)
RXN_NAMES = (
    "Freeze", "Melt", "Boil", "Condense", "WoodBurn", "OilBurn", "PlantBurn"
)
RXN_DE_LIST = [
    336000.0, -336000.0, -840000.0, 840000.0, 900000.0, 750000.0, 600000.0
]
RXN_T_THRESH_LIST = [
    27315.0, 27315.0, 37315.0, 37000.0, 52000.0, 44000.0, 46000.0
]
# Direction: -1 means T < thresh, +1 means T > thresh
RXN_T_DIR_LIST = [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]

# Stoichiometry matrix NU [7, 10]
NU_LIST = [
    [-1, +1,  0,  0,  0,  0,  0,  0,  0,  0],  # 0: Freeze
    [+1, -1,  0,  0,  0,  0,  0,  0,  0,  0],  # 1: Melt
    [-1,  0, +1,  0,  0,  0,  0,  0,  0,  0],  # 2: Boil
    [+1,  0, -1,  0,  0,  0,  0,  0,  0,  0],  # 3: Condense
    [ 0,  0,  0,  0,  0, -2, +1, +1,  0,  0],  # 4: WoodBurn
    [ 0,  0,  0,  0,  0,  0,  0, +1, -1,  0],  # 5: OilBurn
    [ 0,  0,  0,  0,  0,  0, +1, +1,  0, -2],  # 6: PlantBurn
]


class PhysicsConsts:
    """Cached device tensors for fast vectorized operations."""

    def __init__(self, device: torch.device):
        self.device = device
        self.spec_c = torch.tensor(SPEC_C_LIST, dtype=torch.float32, device=device).view(1, 10, 1, 1)
        self.spec_k = torch.tensor(SPEC_K_LIST, dtype=torch.float32, device=device).view(1, 10, 1, 1)
        self.spec_u0 = torch.tensor(SPEC_U0_LIST, dtype=torch.float32, device=device).view(1, 10, 1, 1)
        self.mobile_idx = torch.tensor(MOBILE_IDX, dtype=torch.long, device=device)
        self.mobile_c = torch.tensor([SPEC_C_LIST[i] for i in MOBILE_IDX], dtype=torch.float32, device=device).view(1, 4, 1, 1)
        self.nu = torch.tensor(NU_LIST, dtype=torch.float32, device=device)  # [7, 10]
        self.nu_neg = torch.clamp(-self.nu, min=0.0)  # [7, 10] reactant requirements
        self.rxn_dE = torch.tensor(RXN_DE_LIST, dtype=torch.float32, device=device).view(1, 7, 1, 1)
        self.rxn_dc = (self.nu @ self.spec_c.view(10)).view(1, 7, 1, 1)  # [1, 7, 1, 1]
        self.rxn_thresh = torch.tensor(RXN_T_THRESH_LIST, dtype=torch.float32, device=device).view(1, 7, 1, 1)
        self.rxn_dir = torch.tensor(RXN_T_DIR_LIST, dtype=torch.float32, device=device).view(1, 7, 1, 1)


_CONSTS_CACHE: dict[torch.device, PhysicsConsts] = {}


def get_consts(device: torch.device) -> PhysicsConsts:
    if device not in _CONSTS_CACHE:
        _CONSTS_CACHE[device] = PhysicsConsts(device)
    return _CONSTS_CACHE[device]


@dataclass
class SimState:
    """Complete physical grid state."""
    mass: torch.Tensor       # [B, 10, H, W] float32
    energy: torch.Tensor     # [B, 1, H, W]  float32 (thermal energy in cK * c)
    flow_e: torch.Tensor     # [B, 1, H, W]  float32 (liquid momentum east)
    flow_s: torch.Tensor     # [B, 1, H, W]  float32 (liquid momentum south)
    elevation: torch.Tensor  # [B, 1, H, W]  float32 in [0, 31]

    def clone(self) -> "SimState":
        return SimState(
            mass=self.mass.clone(),
            energy=self.energy.clone(),
            flow_e=self.flow_e.clone(),
            flow_s=self.flow_s.clone(),
            elevation=self.elevation.clone(),
        )

    def detach(self) -> "SimState":
        return SimState(
            mass=self.mass.detach(),
            energy=self.energy.detach(),
            flow_e=self.flow_e.detach(),
            flow_s=self.flow_s.detach(),
            elevation=self.elevation.detach(),
        )


def compute_C_and_T(mass: torch.Tensor, energy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cell heat capacity C_cell [B,1,H,W] and temperature T_cK [B,1,H,W]."""
    c = get_consts(mass.device)
    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    T_cK = torch.where(C_cell > 1e-6, energy / torch.clamp(C_cell, min=1e-6), torch.full_like(C_cell, T_AMB))
    return C_cell, T_cK


def compute_total_energy(state: SimState) -> torch.Tensor:
    """Total energy (thermal + chemical) per batch item in float64 [B]."""
    c = get_consts(state.mass.device)
    e_th = state.energy.to(torch.float64).sum(dim=(1, 2, 3))
    e_chem = (state.mass.to(torch.float64) * c.spec_u0.to(torch.float64)).sum(dim=(1, 2, 3))
    return e_th + e_chem


def compute_total_mass(state: SimState) -> torch.Tensor:
    """Total mass per batch item in float64 [B]."""
    return state.mass.to(torch.float64).sum(dim=(1, 2, 3))


def encode_state(
    state: SimState,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Encode SimState + forcing into the exact ONNX input tensor [B, 18, H, W] float32.
      0..10:  mass[0..10] / 1000.0
      10:     (T_cK - 29000.0) / 10000.0
      11:     log1p(total_mass / 1000.0)
      12:     flow_e / 1000.0
      13:     flow_s / 1000.0
      14:     elevation / 31.0
      15:     rock_wall (1.0 if mass[rock] >= 500 else 0.0)
      16:     forcing_energy / 1e8 (clamped [-10, 10])
      17:     forcing_mass / 1000.0 (clamped [-4, 4])
    """
    B, _, H, W = state.mass.shape
    device = state.mass.device
    if forcing_energy is None:
        forcing_energy = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    if forcing_mass is None:
        forcing_mass = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    _, T_cK = compute_C_and_T(state.mass, state.energy)
    total_mass = state.mass.sum(dim=1, keepdim=True)
    rock_wall = (state.mass[:, S_ROCK:S_ROCK + 1] >= 500.0).to(torch.float32)

    x = torch.cat([
        state.mass / 1000.0,                                      # 0..10
        (T_cK - 29000.0) / 10000.0,                               # 10
        torch.log1p(torch.clamp(total_mass, min=0.0) / 1000.0),   # 11
        state.flow_e / 1000.0,                                    # 12
        state.flow_s / 1000.0,                                    # 13
        state.elevation / 31.0,                                   # 14
        rock_wall,                                                # 15
        torch.clamp(forcing_energy / 1e8, -10.0, 10.0),           # 16
        torch.clamp(forcing_mass / 1000.0, -4.0, 4.0),            # 17
    ], dim=1)
    return x


def apply_forcing(
    state: SimState,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply external player/environment forcing to get pre-step (mass_eff, energy_eff)."""
    mass = state.mass
    energy = state.energy
    c = get_consts(mass.device)

    if forcing_mass is not None and (forcing_mass != 0).any():
        # External mass forcing injects/removes water while respecting [0, H_CAP]
        total_m = mass.sum(dim=1, keepdim=True)
        headroom = torch.clamp(H_CAP - total_m, min=0.0)
        dm_pos = torch.minimum(F.relu(forcing_mass), headroom)
        dm_neg = -torch.minimum(F.relu(-forcing_mass), mass[:, S_WATER:S_WATER + 1])
        dm = dm_pos + dm_neg
        _, T_cK = compute_C_and_T(mass, energy)
        mass = mass.clone()
        mass[:, S_WATER:S_WATER + 1] = mass[:, S_WATER:S_WATER + 1] + dm
        # Added water arrives at T_AMB; removed water removes its thermal energy
        energy = energy + dm_pos * SPEC_C_LIST[S_WATER] * T_AMB + dm_neg * SPEC_C_LIST[S_WATER] * T_cK

    if forcing_energy is not None and (forcing_energy != 0).any():
        C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
        e_min = C_cell * T_MIN
        e_max = C_cell * T_MAX
        energy = torch.where(
            C_cell > 1e-6,
            torch.clamp(energy + forcing_energy, min=e_min, max=e_max),
            torch.zeros_like(energy),
        )

    return mass, energy


def project_deltas(
    mass: torch.Tensor,
    energy: torch.Tensor,
    raw_deltas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact, differentiable invariant-preserving projection on raw network/teacher deltas.

    raw_deltas: [B, 17, H, W]
      0..4:   flux_e[water, steam, smoke, oil] in mass / 1000.0
      4..8:   flux_s[water, steam, smoke, oil] in mass / 1000.0
      8..10:  heat_e, heat_s in 1e6 energy units
      10..17: rxn_extent[0..7] in 100.0 units

    Returns:
      proj_deltas: [B, 17, H, W]
      active_mask: [B, 17, H, W] bool tensor where projection meaningfully altered raw_deltas
    """
    c = get_consts(mass.device)
    B, _, H, W = mass.shape

    # Unpack to physical units
    flux_e = raw_deltas[:, 0:4] * 1000.0
    flux_s = raw_deltas[:, 4:8] * 1000.0
    heat_e = raw_deltas[:, 8:9] * 1e6
    heat_s = raw_deltas[:, 9:10] * 1e6
    rxn = F.relu(raw_deltas[:, 10:17]) * 100.0

    # 1. Domain boundary zeroing (last col E, last row S)
    mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=mass.device)
    mask_e[:, :, :, -1] = 0.0
    mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=mass.device)
    mask_s[:, :, -1, :] = 0.0

    # Wall mask: rock >= 500 or ice >= 700 blocks mobile flux
    wall = ((mass[:, S_ROCK:S_ROCK + 1] >= 500.0) | (mass[:, S_ICE:S_ICE + 1] >= 700.0)).to(torch.float32)
    wall_e_nb = F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0)
    wall_s_nb = F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0)
    open_e = mask_e * (1.0 - wall) * (1.0 - wall_e_nb)
    open_s = mask_s * (1.0 - wall) * (1.0 - wall_s_nb)

    flux_e = flux_e * open_e
    flux_s = flux_s * open_s
    heat_e = heat_e * mask_e
    heat_s = heat_s * mask_s

    # 2. Donor-budget positivity projection for mobile species
    mob_mass = torch.clamp(mass[:, c.mobile_idx], min=0.0)  # [B, 4, H, W]
    # Outgoing flux from cell (y, x):
    # East: relu(flux_e[y, x])
    # West: relu(-flux_e[y, x-1])
    # South: relu(flux_s[y, x])
    # North: relu(-flux_s[y-1, x])
    flux_w_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    flux_n_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)

    out_e = F.relu(flux_e)
    out_w = F.relu(-flux_w_in)
    out_s = F.relu(flux_s)
    out_n = F.relu(-flux_n_in)
    total_out = out_e + out_w + out_s + out_n

    # Scale outgoing fluxes so donor never goes negative
    donor_scale = torch.where(
        total_out > mob_mass,
        mob_mass / (total_out + 1e-8),
        torch.ones_like(total_out),
    )
    donor_scale = torch.clamp(donor_scale, 0.0, 1.0)

    # For E edge (y, x) -> (y, x+1): positive uses donor (y,x), negative uses donor (y,x+1)
    donor_scale_e_nb = F.pad(donor_scale[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    flux_e = torch.where(flux_e >= 0.0, flux_e * donor_scale, flux_e * donor_scale_e_nb)

    donor_scale_s_nb = F.pad(donor_scale[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    flux_s = torch.where(flux_s >= 0.0, flux_s * donor_scale, flux_s * donor_scale_s_nb)

    # 3. Receiver H_CAP projection (prevents total mass > H_CAP)
    flux_w_in2 = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    flux_n_in2 = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    in_from_w = F.relu(flux_w_in2)
    in_from_e = F.relu(-flux_e)
    in_from_n = F.relu(flux_n_in2)
    in_from_s = F.relu(-flux_s)
    total_in_cell = (in_from_w + in_from_e + in_from_n + in_from_s).sum(dim=1, keepdim=True)

    total_mass_cell = mass.sum(dim=1, keepdim=True)
    # Conservative headroom guarantees total_mass_next <= H_CAP strictly even if neighbors also clamp
    headroom = torch.clamp(H_CAP - total_mass_cell, min=0.0)

    recv_scale = torch.where(
        total_in_cell > headroom,
        headroom / (total_in_cell + 1e-8),
        torch.ones_like(total_in_cell),
    )
    recv_scale = torch.clamp(recv_scale, 0.0, 1.0)
    recv_scale_e_nb = F.pad(recv_scale[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    recv_scale_s_nb = F.pad(recv_scale[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

    # Positive flux_e enters (y, x+1); negative flux_e enters (y, x)
    flux_e = torch.where(flux_e >= 0.0, flux_e * recv_scale_e_nb, flux_e * recv_scale)
    flux_s = torch.where(flux_s >= 0.0, flux_s * recv_scale_s_nb, flux_s * recv_scale)

    # 4. Heat conduction 2nd-law discrete maximum principle projection
    C_cell, T_cK = compute_C_and_T(mass, energy)
    C_e_nb = F.pad(C_cell[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
    T_e_nb = F.pad(T_cK[:, :, :, 1:], (0, 1, 0, 0), value=T_AMB)
    dT_e = T_cK - T_e_nb
    q_max_e = torch.abs(dT_e) * (C_cell * C_e_nb) / (4.0 * (C_cell + C_e_nb) + 1e-6)
    q_max_e = torch.where((C_cell > 1e-6) & (C_e_nb > 1e-6), q_max_e, torch.zeros_like(q_max_e)) * mask_e
    heat_e = torch.where(
        dT_e >= 0.0,
        torch.clamp(heat_e, min=torch.zeros_like(q_max_e), max=q_max_e),
        torch.clamp(heat_e, min=-q_max_e, max=torch.zeros_like(q_max_e)),
    )

    C_s_nb = F.pad(C_cell[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
    T_s_nb = F.pad(T_cK[:, :, 1:, :], (0, 0, 0, 1), value=T_AMB)
    dT_s = T_cK - T_s_nb
    q_max_s = torch.abs(dT_s) * (C_cell * C_s_nb) / (4.0 * (C_cell + C_s_nb) + 1e-6)
    q_max_s = torch.where((C_cell > 1e-6) & (C_s_nb > 1e-6), q_max_s, torch.zeros_like(q_max_s)) * mask_s
    heat_s = torch.where(
        dT_s >= 0.0,
        torch.clamp(heat_s, min=torch.zeros_like(q_max_s), max=q_max_s),
        torch.clamp(heat_s, min=-q_max_s, max=torch.zeros_like(q_max_s)),
    )

    # 5. Compute post-transport mass & energy for reaction projection
    mass_tr, energy_tr = _apply_transport(mass, energy, flux_e, flux_s, heat_e, heat_s)
    C_tr, T_tr = compute_C_and_T(mass_tr, energy_tr)

    # Temperature threshold gates for 7 reactions
    # dir < 0 => active if T < thresh; dir > 0 => active if T > thresh
    t_diff = (T_tr - c.rxn_thresh) * c.rxn_dir  # [B, 7, H, W], > 0 when active
    t_gate = (t_diff > 0.0).to(torch.float32) * (C_tr > 1e-6).to(torch.float32)
    rxn = rxn * t_gate

    # Phase-change equilibrium caps (prevent overshoot past transition temperature)
    # 0 Freeze:   xi * (336000 + 21*27315) <= C * (27315 - T)
    # 1 Melt:     xi * (336000 + 21*27315) <= C * (T - 27315)
    # 2 Boil:     for T < 38181, xi * 19070 <= C * (T - 37315); also cap <= 0.4 * water
    # 3 Condense: xi * (840000 - 22*37000) <= C * (37000 - T)
    xi_cap_freeze = C_tr * F.relu(27315.0 - T_tr) / (336000.0 + 21.0 * 27315.0)
    xi_cap_melt = C_tr * F.relu(T_tr - 27315.0) / (336000.0 + 21.0 * 27315.0)
    xi_cap_cond = C_tr * F.relu(37000.0 - T_tr) / (840000.0 - 22.0 * 37000.0)
    xi_cap_boil = torch.where(
        T_tr < 38180.0,
        C_tr * F.relu(T_tr - 37315.0) / (840000.0 - 22.0 * 37315.0),
        mass_tr[:, S_WATER:S_WATER + 1],
    )
    phase_caps = torch.cat([
        xi_cap_freeze,
        xi_cap_melt,
        xi_cap_boil,
        xi_cap_cond,
        torch.full_like(C_tr, 1e9),
        torch.full_like(C_tr, 1e9),
        torch.full_like(C_tr, 1e9),
    ], dim=1)
    rxn = torch.minimum(rxn, phase_caps)

    # Reactant availability scaling: ensure consumed[s] <= mass_tr[s]
    # consumed: [B, 10, H, W] = einsum('brhw,rs->bshw', rxn, nu_neg)
    consumed = torch.einsum("brhw,rs->bshw", rxn, c.nu_neg)
    spec_scale = torch.where(
        consumed > mass_tr,
        mass_tr / (consumed + 1e-8),
        torch.ones_like(consumed),
    )
    spec_scale = torch.clamp(spec_scale, 0.0, 1.0)
    # For each reaction r, take min spec_scale over its consumed species (pure out-of-place for autograd)
    rxn_scales = []
    for r_idx in range(NUM_RXNS):
        s_scale = torch.ones_like(rxn[:, 0:1])
        for s_idx in range(NUM_SPECIES):
            if NU_LIST[r_idx][s_idx] < 0:
                s_scale = torch.minimum(s_scale, spec_scale[:, s_idx:s_idx + 1])
        rxn_scales.append(s_scale)
    rxn_scale = torch.cat(rxn_scales, dim=1)
    rxn = rxn * rxn_scale

    # Temperature bounds safety [T_MIN, T_MAX] after reactions (exact energy conservation!)
    dE_rxn_all = (rxn * c.rxn_dE).sum(dim=1, keepdim=True)
    dC_rxn_all = (rxn * c.rxn_dc).sum(dim=1, keepdim=True)
    E_post = energy_tr + dE_rxn_all
    C_post = torch.clamp(C_tr + dC_rxn_all, min=0.0)

    # Check upper bound E_post <= C_post * T_MAX
    # For any reaction r, its contribution to (E - C*T_MAX) is rxn_r * (dE_r - dC_r * T_MAX)
    excess_hi = E_post - C_post * T_MAX
    hi_coeff = c.rxn_dE - c.rxn_dc * T_MAX  # [1, 7, 1, 1]
    pos_hi_contrib = (rxn * F.relu(hi_coeff)).sum(dim=1, keepdim=True)
    hi_scale = torch.where(
        (excess_hi > 0.0) & (pos_hi_contrib > 1e-6),
        torch.clamp(1.0 - excess_hi / (pos_hi_contrib + 1e-6), 0.0, 1.0),
        torch.ones_like(excess_hi),
    )
    rxn = torch.where(hi_coeff > 0.0, rxn * hi_scale, rxn)

    # Check lower bound E_post >= C_post * T_MIN
    dE_rxn_all = (rxn * c.rxn_dE).sum(dim=1, keepdim=True)
    dC_rxn_all = (rxn * c.rxn_dc).sum(dim=1, keepdim=True)
    E_post = energy_tr + dE_rxn_all
    C_post = torch.clamp(C_tr + dC_rxn_all, min=0.0)
    deficit_lo = C_post * T_MIN - E_post
    lo_coeff = c.rxn_dc * T_MIN - c.rxn_dE  # positive means cools towards/past T_MIN
    pos_lo_contrib = (rxn * F.relu(lo_coeff)).sum(dim=1, keepdim=True)
    lo_scale = torch.where(
        (deficit_lo > 0.0) & (pos_lo_contrib > 1e-6),
        torch.clamp(1.0 - deficit_lo / (pos_lo_contrib + 1e-6), 0.0, 1.0),
        torch.ones_like(deficit_lo),
    )
    rxn = torch.where(lo_coeff > 0.0, rxn * lo_scale, rxn)

    # Repack to normalized ONNX contract units
    proj_deltas = torch.cat([
        flux_e / 1000.0,
        flux_s / 1000.0,
        heat_e / 1e6,
        heat_s / 1e6,
        rxn / 100.0,
    ], dim=1)

    active_mask = (proj_deltas - raw_deltas).abs() > 1e-4
    return proj_deltas, active_mask


def _apply_transport(
    mass: torch.Tensor,
    energy: torch.Tensor,
    flux_e: torch.Tensor,
    flux_s: torch.Tensor,
    heat_e: torch.Tensor,
    heat_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conservative finite-volume transport of mobile mass, advected enthalpy, and conduction heat."""
    c = get_consts(mass.device)
    _, T_cK = compute_C_and_T(mass, energy)

    # Mass divergence for 4 mobile species
    flux_w_in = F.pad(flux_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    flux_n_in = F.pad(flux_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    dm_mob = flux_w_in - flux_e + flux_n_in - flux_s  # [B, 4, H, W]

    # Scatter mobile mass deltas cleanly without breaking autograd
    dm_full = torch.zeros_like(mass)
    dm_full = dm_full.scatter_add(1, c.mobile_idx.view(1, 4, 1, 1).expand_as(dm_mob), dm_mob)
    mass_new = torch.clamp(mass + dm_full, min=0.0)

    # Advected enthalpy on E and S edges (upwind T_donor * c_m * flux)
    T_e_nb = F.pad(T_cK[:, :, :, 1:], (0, 1, 0, 0), value=T_AMB)
    T_s_nb = F.pad(T_cK[:, :, 1:, :], (0, 0, 0, 1), value=T_AMB)

    # Positive flux_e leaves (y,x) at T_cK; negative flux_e enters (y,x) at T_e_nb
    adv_e = ((F.relu(flux_e) * T_cK - F.relu(-flux_e) * T_e_nb) * c.mobile_c).sum(dim=1, keepdim=True)
    adv_s = ((F.relu(flux_s) * T_cK - F.relu(-flux_s) * T_s_nb) * c.mobile_c).sum(dim=1, keepdim=True)

    q_e = adv_e + heat_e
    q_s = adv_s + heat_s
    q_w_in = F.pad(q_e[:, :, :, :-1], (1, 0, 0, 0), value=0.0)
    q_n_in = F.pad(q_s[:, :, :-1, :], (0, 0, 1, 0), value=0.0)
    dE_transport = q_w_in - q_e + q_n_in - q_s

    energy_new = energy + dE_transport
    return mass_new, energy_new


def apply_projected_deltas(
    state: SimState,
    proj_deltas: torch.Tensor,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
) -> SimState:
    """Apply already-projected deltas to state (after external forcing) -> next SimState."""
    c = get_consts(state.mass.device)
    mass_eff, energy_eff = apply_forcing(state, forcing_energy, forcing_mass)

    flux_e = proj_deltas[:, 0:4] * 1000.0
    flux_s = proj_deltas[:, 4:8] * 1000.0
    heat_e = proj_deltas[:, 8:9] * 1e6
    heat_s = proj_deltas[:, 9:10] * 1e6
    rxn = proj_deltas[:, 10:17] * 100.0

    # 1. Conservative transport + heat conduction
    mass_tr, energy_tr = _apply_transport(mass_eff, energy_eff, flux_e, flux_s, heat_e, heat_s)

    # 2. Conservative reactions
    dm_rxn = torch.einsum("brhw,rs->bshw", rxn, c.nu)
    dE_rxn = (rxn * c.rxn_dE).sum(dim=1, keepdim=True)

    mass_next = torch.clamp(mass_tr + dm_rxn, min=0.0)
    energy_next = energy_tr + dE_rxn

    # 3. Update river liquid flow inertia (water idx 0, oil idx 3 in mobile list)
    liq_flux_e = flux_e[:, 0:1] + flux_e[:, 3:4]
    liq_flux_s = flux_s[:, 0:1] + flux_s[:, 3:4]
    wall = ((mass_next[:, S_ROCK:S_ROCK + 1] >= 500.0) | (mass_next[:, S_ICE:S_ICE + 1] >= 700.0)).to(torch.float32)
    wall_e_nb = F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0)
    wall_s_nb = F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0)
    H, W = mass_next.shape[2], mass_next.shape[3]
    mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=mass_next.device)
    mask_e[:, :, :, -1] = 0.0
    mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=mass_next.device)
    mask_s[:, :, -1, :] = 0.0
    open_e = mask_e * (1.0 - wall) * (1.0 - wall_e_nb)
    open_s = mask_s * (1.0 - wall) * (1.0 - wall_s_nb)

    flow_e_next = torch.clamp((0.75 * state.flow_e + 0.25 * liq_flux_e) * open_e, -1500.0, 1500.0)
    flow_s_next = torch.clamp((0.75 * state.flow_s + 0.25 * liq_flux_s) * open_s, -1500.0, 1500.0)

    return SimState(
        mass=mass_next,
        energy=energy_next,
        flow_e=flow_e_next,
        flow_s=flow_s_next,
        elevation=state.elevation,
    )


def project_and_step(
    state: SimState,
    raw_deltas: torch.Tensor,
    forcing_energy: torch.Tensor | None = None,
    forcing_mass: torch.Tensor | None = None,
) -> tuple[SimState, torch.Tensor, torch.Tensor]:
    """Full step with exact projection: returns (next_state, proj_deltas, active_mask)."""
    mass_eff, energy_eff = apply_forcing(state, forcing_energy, forcing_mass)
    proj_deltas, active_mask = project_deltas(mass_eff, energy_eff, raw_deltas)
    next_state = apply_projected_deltas(state, proj_deltas, forcing_energy, forcing_mass)
    return next_state, proj_deltas, active_mask


# =============================================================================
# Reference Physics Simulator (RefSim)
# =============================================================================
class RefSim:
    """
    GPU-vectorized PyTorch Reference Physics Teacher for Frostmarsh.
    Produces exact, physically grounded (raw_deltas, proj_deltas, next_state).
    """

    @staticmethod
    @torch.no_grad()
    def compute_raw_deltas(
        state: SimState,
        forcing_energy: torch.Tensor | None = None,
        forcing_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = get_consts(state.mass.device)
        mass, energy = apply_forcing(state, forcing_energy, forcing_mass)
        B, _, H, W = mass.shape
        device = mass.device

        C_cell, T_cK = compute_C_and_T(mass, energy)

        # Wall & boundary masks
        mask_e = torch.ones((1, 1, H, W), dtype=torch.float32, device=device)
        mask_e[:, :, :, -1] = 0.0
        mask_s = torch.ones((1, 1, H, W), dtype=torch.float32, device=device)
        mask_s[:, :, -1, :] = 0.0

        wall = ((mass[:, S_ROCK:S_ROCK + 1] >= 500.0) | (mass[:, S_ICE:S_ICE + 1] >= 700.0)).to(torch.float32)
        wall_e_nb = F.pad(wall[:, :, :, 1:], (0, 1, 0, 0), value=1.0)
        wall_s_nb = F.pad(wall[:, :, 1:, :], (0, 0, 0, 1), value=1.0)
        open_e = mask_e * (1.0 - wall) * (1.0 - wall_e_nb)
        open_s = mask_s * (1.0 - wall) * (1.0 - wall_s_nb)

        # --- 1. Liquid flow (water + oil) ---
        water = mass[:, S_WATER:S_WATER + 1]
        oil = mass[:, S_OIL:S_OIL + 1]
        tot_liq = water + oil
        head = state.elevation * 120.0 + tot_liq

        head_e_nb = F.pad(head[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        head_s_nb = F.pad(head[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

        drive_e = (0.20 * (head - head_e_nb) + 0.35 * state.flow_e) * open_e
        drive_s = (0.20 * (head - head_s_nb) + 0.35 * state.flow_s) * open_s

        water_e_nb = F.pad(water[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        water_s_nb = F.pad(water[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
        oil_e_nb = F.pad(oil[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        oil_s_nb = F.pad(oil[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
        tot_liq_e_nb = water_e_nb + oil_e_nb
        tot_liq_s_nb = water_s_nb + oil_s_nb

        # Donor fractions & availability
        frac_w_e = torch.where(
            drive_e >= 0.0,
            water / (tot_liq + 1e-6),
            water_e_nb / (tot_liq_e_nb + 1e-6),
        )
        frac_o_e = torch.where(
            drive_e >= 0.0,
            oil / (tot_liq + 1e-6),
            oil_e_nb / (tot_liq_e_nb + 1e-6),
        )
        has_liq_e = torch.where(drive_e >= 0.0, (tot_liq > 1e-4).to(torch.float32), (tot_liq_e_nb > 1e-4).to(torch.float32))

        frac_w_s = torch.where(
            drive_s >= 0.0,
            water / (tot_liq + 1e-6),
            water_s_nb / (tot_liq_s_nb + 1e-6),
        )
        frac_o_s = torch.where(
            drive_s >= 0.0,
            oil / (tot_liq + 1e-6),
            oil_s_nb / (tot_liq_s_nb + 1e-6),
        )
        has_liq_s = torch.where(drive_s >= 0.0, (tot_liq > 1e-4).to(torch.float32), (tot_liq_s_nb > 1e-4).to(torch.float32))

        # Oil viscosity ~0.55x water + slight oil slick surface spreading
        flux_e_water = drive_e * frac_w_e * has_liq_e
        flux_s_water = drive_s * frac_w_s * has_liq_s
        flux_e_oil = (0.55 * drive_e * frac_o_e * has_liq_e + 0.06 * (oil - oil_e_nb)) * open_e
        flux_s_oil = (0.55 * drive_s * frac_o_s * has_liq_s + 0.06 * (oil - oil_s_nb)) * open_s

        # --- 2. Gas diffusion + gentle breeze drift (steam, smoke) ---
        steam = mass[:, S_STEAM:S_STEAM + 1]
        smoke = mass[:, S_SMOKE:S_SMOKE + 1]
        steam_e_nb = F.pad(steam[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        steam_s_nb = F.pad(steam[:, :, 1:, :], (0, 0, 0, 1), value=0.0)
        smoke_e_nb = F.pad(smoke[:, :, :, 1:], (0, 1, 0, 0), value=0.0)
        smoke_s_nb = F.pad(smoke[:, :, 1:, :], (0, 0, 0, 1), value=0.0)

        # Gentle breeze: +0.035 east, -0.02 south (i.e. north-east drift)
        flux_e_steam = (0.16 * (steam - steam_e_nb) + 0.035 * steam) * open_e
        flux_s_steam = (0.16 * (steam - steam_s_nb) - 0.020 * steam_s_nb) * open_s
        flux_e_smoke = (0.15 * (smoke - smoke_e_nb) + 0.035 * smoke) * open_e
        flux_s_smoke = (0.15 * (smoke - smoke_s_nb) - 0.020 * smoke_s_nb) * open_s

        flux_e = torch.cat([flux_e_water, flux_e_steam, flux_e_smoke, flux_e_oil], dim=1)
        flux_s = torch.cat([flux_s_water, flux_s_steam, flux_s_smoke, flux_s_oil], dim=1)

        # --- 3. Heat conduction ---
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
        heat_e = 0.008 * k_eff_e * C_eff_e * (T_cK - T_e_nb) * mask_e
        heat_s = 0.008 * k_eff_s * C_eff_s * (T_cK - T_s_nb) * mask_s

        # --- 4. Reactions (computed on post-transport T & mass for tight consistency with projection) ---
        # First project transport alone to match exact post-transport state
        dummy_rxn = torch.zeros((B, 7, H, W), dtype=torch.float32, device=device)
        tr_raw = torch.cat([flux_e / 1000.0, flux_s / 1000.0, heat_e / 1e6, heat_s / 1e6, dummy_rxn], dim=1)
        tr_proj, _ = project_deltas(mass, energy, tr_raw)
        mass_tr, energy_tr = _apply_transport(
            mass, energy,
            tr_proj[:, 0:4] * 1000.0,
            tr_proj[:, 4:8] * 1000.0,
            tr_proj[:, 8:9] * 1e6,
            tr_proj[:, 9:10] * 1e6,
        )
        _, T_tr = compute_C_and_T(mass_tr, energy_tr)

        r_freeze = 0.040 * F.relu(27315.0 - T_tr) * (mass_tr[:, S_WATER:S_WATER + 1] / 1000.0)
        r_melt = 0.030 * F.relu(T_tr - 27315.0) * (mass_tr[:, S_ICE:S_ICE + 1] / 1000.0)
        r_boil = 0.025 * F.relu(T_tr - 37315.0) * (mass_tr[:, S_WATER:S_WATER + 1] / 1000.0)
        r_cond = 0.025 * F.relu(37000.0 - T_tr) * (mass_tr[:, S_STEAM:S_STEAM + 1] / 1000.0)
        r_wood = 0.0025 * F.relu(T_tr - 52000.0) * (mass_tr[:, S_WOOD:S_WOOD + 1] / 1000.0)
        r_oil = 0.0040 * F.relu(T_tr - 44000.0) * (mass_tr[:, S_OIL:S_OIL + 1] / 1000.0)
        r_plant = 0.0035 * F.relu(T_tr - 46000.0) * (mass_tr[:, S_PLANT:S_PLANT + 1] / 1000.0)

        rxn = torch.cat([r_freeze, r_melt, r_boil, r_cond, r_wood, r_oil, r_plant], dim=1)
        raw_deltas = torch.cat([
            flux_e / 1000.0,
            flux_s / 1000.0,
            heat_e / 1e6,
            heat_s / 1e6,
            rxn / 100.0,
        ], dim=1)
        return raw_deltas

    @staticmethod
    @torch.no_grad()
    def step(
        state: SimState,
        forcing_energy: torch.Tensor | None = None,
        forcing_mass: torch.Tensor | None = None,
    ) -> tuple[SimState, torch.Tensor]:
        """Advance RefSim one 0.1s tick; returns (next_state, proj_deltas)."""
        raw_deltas = RefSim.compute_raw_deltas(state, forcing_energy, forcing_mass)
        next_state, proj_deltas, _ = project_and_step(state, raw_deltas, forcing_energy, forcing_mass)
        return next_state, proj_deltas


# =============================================================================
# Procedural Frostmarsh Scene & Player Forcing Generators
# =============================================================================
def _smooth_noise(B: int, H: int, W: int, device: torch.device, scale: int = 8) -> torch.Tensor:
    """Smooth 2D value noise via bilinear upsampling."""
    gh = max(2, H // scale)
    gw = max(2, W // scale)
    grid = torch.rand((B, 1, gh, gw), device=device)
    return F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)


def sample_bolts(
    B: int,
    H: int,
    W: int,
    device: torch.device,
    prob: float = 0.35,
    max_bolts: int = 2,
    force_kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random player Fire Bolts (+1.5e8..+4e8 energy, radius 3..5)
    and Frost Bolts (-1.5e8..-3.5e8 energy, radius 3..5).
    """
    forcing_e = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    forcing_m = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)

    yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)

    for b in range(B):
        if torch.rand(1, device=device).item() > prob:
            continue
        n_bolts = int(torch.randint(1, max_bolts + 1, (1,), device=device).item())
        for _ in range(n_bolts):
            cy = float(torch.randint(4, max(5, H - 4), (1,), device=device).item())
            cx = float(torch.randint(4, max(5, W - 4), (1,), device=device).item())
            rad = float(torch.empty(1, device=device).uniform_(3.0, 5.2).item())
            dist2 = (yy[0] - cy) ** 2 + (xx[0] - cx) ** 2
            kernel = torch.clamp(1.0 - dist2 / (rad * rad), min=0.0)

            if force_kind == "fire":
                is_fire = True
            elif force_kind == "frost":
                is_fire = False
            else:
                is_fire = bool(torch.rand(1, device=device).item() < 0.52)

            if is_fire:
                mag = float(torch.empty(1, device=device).uniform_(1.5e8, 4.0e8).item())
                forcing_e[b, 0] += kernel * mag
            else:
                mag = float(torch.empty(1, device=device).uniform_(1.5e8, 3.5e8).item())
                forcing_e[b, 0] -= kernel * mag

    return forcing_e, forcing_m


def make_scene(
    B: int,
    H: int,
    W: int,
    device: torch.device,
    scene_type: str = "mixed",
    seed: int | None = None,
) -> SimState:
    """
    Generate diverse Frostmarsh procedural scenes.
    scene_type in {'mixed', 'river_crossing', 'forest_wildfire', 'oil_slick_marsh', 'dam_break', 'thermal_mix'}
    """
    if seed is not None:
        torch.manual_seed(seed)

    c = get_consts(device)
    mass = torch.zeros((B, NUM_SPECIES, H, W), dtype=torch.float32, device=device)
    flow_e = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    flow_s = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    elev = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    T_init = torch.full((B, 1, H, W), T_AMB, dtype=torch.float32, device=device)

    yy = torch.linspace(0.0, 1.0, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.linspace(0.0, 1.0, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

    n_coarse = _smooth_noise(B, H, W, device, scale=max(4, min(H, W) // 6))
    n_med = _smooth_noise(B, H, W, device, scale=max(2, min(H, W) // 14))
    n_fine = _smooth_noise(B, H, W, device, scale=max(2, min(H, W) // 24))

    types = ("river_crossing", "forest_wildfire", "oil_slick_marsh", "dam_break", "thermal_mix")

    for b in range(B):
        st = scene_type if scene_type != "mixed" else types[b % len(types)]
        nc = n_coarse[b, 0]
        nm = n_med[b, 0]
        nf = n_fine[b, 0]
        y = yy[b, 0]
        x = xx[b, 0]

        if st == "river_crossing":
            # Winding river cutting vertically or diagonally with sloped bed
            phase = float(torch.empty(1, device=device).uniform_(0, 6.28).item())
            amp = float(torch.empty(1, device=device).uniform_(0.08, 0.16).item())
            river_x = 0.5 + amp * torch.sin(y * 5.0 + phase)
            dist = torch.abs(x - river_x)
            river_w = float(torch.empty(1, device=device).uniform_(0.11, 0.17).item())
            in_river = dist < river_w

            e_b = 10.0 * (1.0 - 0.35 * y) + 6.0 * nc + 2.0 * nm
            e_b = torch.where(in_river, 2.0 + 3.5 * (1.0 - y) + 1.5 * (dist / river_w), e_b)
            elev[b, 0] = torch.clamp(e_b, 0.0, 31.0)

            mass[b, S_SOIL] = torch.where(in_river, 250.0, 750.0 + 250.0 * nc)
            mass[b, S_WATER] = torch.where(in_river, 750.0 + 350.0 * (1.0 - dist / river_w), 0.0)
            flow_s[b, 0] = torch.where(in_river, 120.0, 0.0)

            # Rocky banks / boulders
            rock_mask = (~in_river) & (nm > 0.78)
            mass[b, S_ROCK] = torch.where(rock_mask, 900.0, 0.0)

            # Forest & marsh plants on banks
            wood_mask = (~in_river) & (~rock_mask) & (nf > 0.60)
            plant_mask = (~in_river) & (~rock_mask) & (~wood_mask) & (nm > 0.38)
            mass[b, S_WOOD] = torch.where(wood_mask, 500.0 + 400.0 * nf, 0.0)
            mass[b, S_PLANT] = torch.where(plant_mask, 300.0 + 350.0 * nm, 0.0)

            # Occasional icy patches near river edges
            ice_patch = in_river & (nf > 0.86)
            mass[b, S_ICE] = torch.where(ice_patch, 450.0, 0.0)
            T_init[b, 0] = torch.where(ice_patch, 26500.0, 28600.0 + 800.0 * nc)

        elif st == "forest_wildfire":
            elev[b, 0] = torch.clamp(6.0 + 12.0 * nc + 4.0 * nm, 0.0, 31.0)
            mass[b, S_SOIL] = 800.0 + 200.0 * nc
            pond = nc < 0.22
            mass[b, S_WATER] = torch.where(pond, 600.0 + 300.0 * (0.22 - nc) / 0.22, 0.0)
            rock_mask = (~pond) & (nm > 0.82)
            mass[b, S_ROCK] = torch.where(rock_mask, 950.0, 0.0)
            wood_mask = (~pond) & (~rock_mask) & (nm > 0.32)
            plant_mask = (~pond) & (~rock_mask) & (nf > 0.42)
            mass[b, S_WOOD] = torch.where(wood_mask, 600.0 + 500.0 * nm, 0.0)
            mass[b, S_PLANT] = torch.where(plant_mask, 250.0 + 300.0 * nf, 0.0)

            # Active ignition hotspot so fire spreads immediately
            fy = float(torch.empty(1, device=device).uniform_(0.25, 0.75).item())
            fx = float(torch.empty(1, device=device).uniform_(0.25, 0.75).item())
            hot = ((y - fy) ** 2 + (x - fx) ** 2) < 0.008
            T_init[b, 0] = torch.where(hot, 68000.0, 29200.0 + 600.0 * nm)

        elif st == "oil_slick_marsh":
            elev[b, 0] = torch.clamp(3.0 + 6.0 * nc + 2.0 * nm, 0.0, 31.0)
            wet = nc < 0.62
            mass[b, S_SOIL] = torch.where(wet, 350.0, 750.0)
            mass[b, S_WATER] = torch.where(wet, 550.0 + 350.0 * nm, 80.0 * nm)
            oil_patch = wet & (nf > 0.55)
            mass[b, S_OIL] = torch.where(oil_patch, 400.0 + 450.0 * nf, 0.0)
            rock_mask = (~wet) & (nm > 0.75)
            mass[b, S_ROCK] = torch.where(rock_mask, 850.0, 0.0)
            plant_mask = (~rock_mask) & (nm > 0.40) & (~oil_patch)
            mass[b, S_PLANT] = torch.where(plant_mask, 350.0 + 300.0 * nm, 0.0)
            wood_mask = (~wet) & (~rock_mask) & (nf > 0.62)
            mass[b, S_WOOD] = torch.where(wood_mask, 500.0, 0.0)

            # Occasional hot spark on oil
            if torch.rand(1, device=device).item() < 0.65:
                fy = float(torch.empty(1, device=device).uniform_(0.25, 0.75).item())
                fx = float(torch.empty(1, device=device).uniform_(0.25, 0.75).item())
                hot = ((y - fy) ** 2 + (x - fx) ** 2) < 0.006
                T_init[b, 0] = torch.where(hot, 56000.0, 29000.0)

        elif st == "dam_break":
            # High reservoir on left (x < 0.38), rock wall at x in [0.38, 0.44] with ice plug breach
            res = x < 0.38
            wall_band = (x >= 0.38) & (x < 0.44)
            gap = torch.abs(y - 0.5) < 0.12
            e_b = torch.where(res, 18.0 + 3.0 * nc, 16.0 * (1.0 - x) + 3.0 * nc)
            elev[b, 0] = torch.clamp(e_b, 0.0, 31.0)
            mass[b, S_SOIL] = 500.0
            mass[b, S_WATER] = torch.where(res, 1200.0 + 600.0 * nm, 0.0)
            mass[b, S_ROCK] = torch.where(wall_band & (~gap), 1200.0, 0.0)
            # Half the time ice plug, half open breach
            if b % 2 == 0:
                mass[b, S_ICE] = torch.where(wall_band & gap, 850.0, 0.0)
                T_init[b, 0] = torch.where(wall_band & gap, 32000.0, 29000.0)
            else:
                flow_e[b, 0] = torch.where(res, 150.0, 0.0)
            # Downstream vegetation & oil
            ds = x > 0.46
            mass[b, S_PLANT] = torch.where(ds & (nm > 0.45), 350.0, 0.0)
            mass[b, S_WOOD] = torch.where(ds & (nf > 0.68), 500.0, 0.0)
            mass[b, S_OIL] = torch.where(ds & (nc < 0.30), 300.0, 0.0)

        else:  # thermal_mix — broad state coverage for all 10 species & 7 reactions
            elev[b, 0] = torch.clamp(22.0 * nc + 6.0 * nm, 0.0, 31.0)
            mass[b, S_SOIL] = 400.0 + 300.0 * nc
            mass[b, S_WATER] = torch.where(nm > 0.35, 700.0 * nm, 0.0)
            mass[b, S_ICE] = torch.where(nc < 0.32, 750.0 * (1.0 - nc), 0.0)
            mass[b, S_STEAM] = torch.where(nf > 0.72, 350.0 * nf, 0.0)
            mass[b, S_SMOKE] = torch.where((nf > 0.55) & (nf <= 0.72), 250.0 * nf, 0.0)
            mass[b, S_ROCK] = torch.where((nc > 0.82) & (nm > 0.65), 1000.0, 0.0)
            mass[b, S_WOOD] = torch.where((nm > 0.55) & (mass[b, S_ROCK] == 0), 500.0 * nm, 0.0)
            mass[b, S_OIL] = torch.where((nc < 0.45) & (nf > 0.50) & (mass[b, S_ROCK] == 0), 400.0 * nf, 0.0)
            mass[b, S_PLANT] = torch.where((nm < 0.55) & (nf > 0.45) & (mass[b, S_ROCK] == 0), 350.0 * nf, 0.0)
            mass[b, S_ASH] = torch.where(nf < 0.12, 150.0, 0.0)

            # Wide temperature range: sub-freezing, ambient, boiling, burning
            T_b = 21000.0 + 42000.0 * nm
            T_b = torch.where(mass[b, S_ICE] > 300.0, 24000.0 + 5000.0 * nf, T_b)
            T_b = torch.where(mass[b, S_STEAM] > 100.0, 35000.0 + 6000.0 * nf, T_b)
            T_init[b, 0] = torch.clamp(T_b, T_MIN + 500.0, 75000.0)
            flow_e[b, 0] = (nc - 0.5) * 300.0 * (mass[b, S_WATER] > 50.0).to(torch.float32)
            flow_s[b, 0] = (nm - 0.5) * 300.0 * (mass[b, S_WATER] > 50.0).to(torch.float32)

    # Enforce H_CAP strictly on initial state
    tot_m = mass.sum(dim=1, keepdim=True)
    cap_scale = torch.where(tot_m > H_CAP * 0.95, (H_CAP * 0.95) / (tot_m + 1e-6), torch.ones_like(tot_m))
    mass = mass * cap_scale

    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    energy = C_cell * torch.clamp(T_init, T_MIN, T_MAX)

    return SimState(mass=mass, energy=energy, flow_e=flow_e, flow_s=flow_s, elevation=elev)
