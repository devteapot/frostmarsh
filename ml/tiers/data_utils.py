#!/usr/bin/env python3
"""
Shared Frostmarsh A+ Data & Rollout Utilities for Tiers 1 & 2.

Loads real Frostmarsh physics (`sim_teacher.py`, `world_model.pt`, `ecosystem_model.pt`)
and exact 4 B/cell A+ wire encoding/decoding from `LATENT_EXPERIMENT`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ML_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
LATENT_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-latent-exp")
PYDEPS_DIR = LATENT_DIR / ".pydeps"
OUT_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-tiers-1-2")

for _p in (str(PYDEPS_DIR), str(ML_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EcosystemModel, WorldModel  # noqa: E402
from sim_teacher import (  # noqa: E402
    H_CAP,
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
    SimState,
    T_AMB,
    T_MAX,
    T_MIN,
    compute_C_and_T,
    encode_state,
    get_consts,
    make_scene,
    project_and_step,
    sample_bolts,
)

# =============================================================================
# Gaussian Blur, Shift & Misc Helpers
# =============================================================================
_GAUSS_CACHE: dict[tuple[torch.device, float, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_gauss_kernels(device: torch.device, sigma: float, radius: int | None = None):
    if radius is None:
        radius = max(1, int(math.ceil(2.5 * sigma)))
    key = (device, round(float(sigma), 3), radius)
    if key not in _GAUSS_CACHE:
        coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        k = torch.exp(-0.5 * (coords / max(sigma, 1e-4)) ** 2)
        k = k / k.sum()
        _GAUSS_CACHE[key] = (k.view(1, 1, 1, -1), k.view(1, 1, -1, 1))
    return _GAUSS_CACHE[key]


def gaussian_blur(x: torch.Tensor, sigma: float, radius: int | None = None) -> torch.Tensor:
    if sigma <= 0.05:
        return x
    kx, ky = _get_gauss_kernels(x.device, sigma, radius)
    r = kx.shape[-1] // 2
    c = x.shape[1]
    if c > 1:
        kx = kx.expand(c, 1, 1, -1)
        ky = ky.expand(c, 1, -1, 1)
    y = F.pad(x, (r, r, 0, 0), mode="replicate")
    y = F.conv2d(y, kx, groups=c)
    y = F.pad(y, (0, 0, r, r), mode="replicate")
    y = F.conv2d(y, ky, groups=c)
    return y


def shift2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    pad_l, pad_r = max(dx, 0), max(-dx, 0)
    pad_t, pad_b = max(dy, 0), max(-dy, 0)
    p = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="replicate")
    h, w = x.shape[-2], x.shape[-1]
    return p[:, :, max(-dy, 0):max(-dy, 0) + h, max(-dx, 0):max(-dx, 0) + w]


def smooth_noise(b: int, h: int, w: int, device: torch.device, scale: int) -> torch.Tensor:
    gh = max(2, h // scale)
    gw = max(2, w // scale)
    return F.interpolate(torch.rand((b, 1, gh, gw), device=device), size=(h, w), mode="bilinear", align_corners=False)


def rgb_const(r: float, g: float, b: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([r, g, b], dtype=torch.float32, device=device).view(1, 3, 1, 1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Exact A+ Wire Codec (Matching LATENT_EXPERIMENT / renderer_v2.onnx)
# =============================================================================
def compute_burning_mask(mass: torch.Tensor, t_ck: torch.Tensor) -> torch.Tensor:
    oil_burn = (t_ck > 44000.0) & (mass[:, S_OIL:S_OIL + 1] > 15.0)
    plant_burn = (t_ck > 46000.0) & (mass[:, S_PLANT:S_PLANT + 1] > 15.0)
    wood_burn = (t_ck > 52000.0) & (mass[:, S_WOOD:S_WOOD + 1] > 15.0)
    hot_fire = t_ck > 58000.0
    return oil_burn | plant_burn | wood_burn | hot_fire


def compute_hazard_critter_byte(mass: torch.Tensor, t_ck: torch.Tensor, critter_norm: torch.Tensor) -> torch.Tensor:
    wall = mass[:, S_ROCK:S_ROCK + 1] >= 500.0
    deep_water = (mass[:, S_WATER:S_WATER + 1] >= 350.0) & (mass[:, S_ICE:S_ICE + 1] < 700.0)
    walkable = (~wall) & (~deep_water)
    burning = compute_burning_mask(mass, t_ck)
    freezing = t_ck < 27315.0
    water_slow = (mass[:, S_WATER:S_WATER + 1] >= 150.0) & (mass[:, S_ICE:S_ICE + 1] < 700.0)
    crit_3b = torch.clamp(torch.round(torch.clamp(critter_norm, 0.0, 1.0) * 7.0), 0, 7).to(torch.uint8)
    return (
        walkable.to(torch.uint8)
        | (wall.to(torch.uint8) << 1)
        | (burning.to(torch.uint8) << 2)
        | (freezing.to(torch.uint8) << 3)
        | (water_slow.to(torch.uint8) << 4)
        | (crit_3b << 5)
    )


def encode_temp_8b(t_ck: torch.Tensor) -> torch.Tensor:
    t_q = torch.round(((t_ck - 20000.0) / 90000.0) * 255.0)
    return torch.clamp(t_q, 0.0, 255.0).to(torch.uint8)


@torch.no_grad()
def encode_aplus_wire(
    mass: torch.Tensor,
    t_ck: torch.Tensor,
    flow_e: torch.Tensor,
    flow_s: torch.Tensor,
    critter_norm: torch.Tensor,
) -> torch.Tensor:
    top_vals, top_idx = torch.topk(mass, k=2, dim=1)
    m1 = top_vals[:, 0:1]
    m2 = top_vals[:, 1:2]
    s1 = top_idx[:, 0:1].to(torch.uint8)
    s2 = top_idx[:, 1:2].to(torch.uint8)

    empty1 = m1 < 1.0
    empty2 = (m2 < 1.0) | empty1
    s1 = torch.where(empty1, torch.full_like(s1, 15), s1)
    s2 = torch.where(empty2, torch.full_like(s2, 15), s2)

    denom = torch.clamp(m1 + m2, min=1e-6)
    s2_frac_4b = torch.round(torch.clamp(m2 / denom, 0.0, 1.0) * 15.0).to(torch.uint8)
    s2_frac_4b = torch.where(empty2, torch.zeros_like(s2_frac_4b), s2_frac_4b)

    mag = torch.sqrt(flow_e * flow_e + flow_s * flow_s)
    angle = torch.atan2(flow_s, flow_e)
    dir_idx = torch.remainder(torch.round(angle / (math.pi / 4.0)).long(), 8).to(torch.uint8)
    still = mag < 15.0
    fast = mag >= 100.0
    flow_slow = dir_idx + 1
    flow_fast = torch.clamp(dir_idx + 9, max=15)
    flow_dir_4b = torch.where(still, torch.zeros_like(dir_idx), torch.where(fast, flow_fast, flow_slow))

    byte0 = (s1 << 4) | s2
    byte1 = (s2_frac_4b << 4) | flow_dir_4b
    byte2 = encode_temp_8b(t_ck)
    byte3 = compute_hazard_critter_byte(mass, t_ck, critter_norm)
    return torch.cat([byte0, byte1, byte2, byte3], dim=1)


@torch.no_grad()
def decode_aplus_wire(wire_u8: torch.Tensor, t_sec: torch.Tensor) -> torch.Tensor:
    """Unpack exact 4-byte A+ wire [B, 4, H, W] -> `aplus_state` float32 [B, 17, H, W]."""
    b, _, h, w = wire_u8.shape
    device = wire_u8.device
    byte0 = wire_u8[:, 0]
    byte1 = wire_u8[:, 1]
    byte2 = wire_u8[:, 2]
    byte3 = wire_u8[:, 3]

    s1 = (byte0 >> 4).long()
    s2 = (byte0 & 0x0F).long()
    frac = (byte1 >> 4).float() / 15.0
    flow_dir = (byte1 & 0x0F).long()

    species = torch.zeros((b, 10, h, w), dtype=torch.float32, device=device)
    valid1 = s1 < 10
    valid2 = s2 < 10
    s1_safe = torch.where(valid1, s1, torch.zeros_like(s1)).unsqueeze(1)
    s2_safe = torch.where(valid2, s2, torch.zeros_like(s2)).unsqueeze(1)
    w1 = torch.where(valid1, 1.0 - frac, torch.zeros_like(frac)).unsqueeze(1)
    w2 = torch.where(valid2, frac, torch.zeros_like(frac)).unsqueeze(1)
    species.scatter_add_(1, s1_safe, w1)
    species.scatter_add_(1, s2_safe, w2)

    temp_norm = (byte2.float() / 255.0).unsqueeze(1)
    burning = (((byte3 >> 2) & 1).float()).unsqueeze(1)

    still = flow_dir == 0
    fast = flow_dir >= 9
    dir_idx = torch.where(still, torch.zeros_like(flow_dir), (flow_dir - 1) & 7)
    angle = dir_idx.float() * (math.pi / 4.0)
    speed = torch.where(still, torch.zeros_like(angle), torch.where(fast, torch.ones_like(angle), torch.full_like(angle, 0.5)))
    flow_x = (torch.cos(angle) * speed).unsqueeze(1)
    flow_y = (torch.sin(angle) * speed).unsqueeze(1)

    critter = (((byte3 >> 5) & 7).float() / 7.0).unsqueeze(1)
    t_view = t_sec.view(b, 1, 1, 1).to(device=device, dtype=torch.float32)
    time_sin = (torch.sin(2.5 * t_view) * 0.5 + 0.5).expand(b, 1, h, w)
    time_cos = (torch.cos(2.5 * t_view) * 0.5 + 0.5).expand(b, 1, h, w)
    return torch.cat([species, temp_norm, burning, flow_x, flow_y, critter, time_sin, time_cos], dim=1)


# =============================================================================
# Pretrained World & Ecosystem Rollout Collector
# =============================================================================
def load_pretrained_sim_models(device: torch.device) -> tuple[WorldModel, EcosystemModel]:
    w_model = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    w_ckpt = torch.load(str(ML_DIR / "world_model.pt"), map_location=device, weights_only=False)
    w_model.load_state_dict(w_ckpt["state_dict"] if isinstance(w_ckpt, dict) and "state_dict" in w_ckpt else w_ckpt)

    e_model = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).to(device).eval()
    e_ckpt = torch.load(str(ML_DIR / "ecosystem_model.pt"), map_location=device, weights_only=False)
    e_model.load_state_dict(e_ckpt["state_dict"] if isinstance(e_ckpt, dict) and "state_dict" in e_ckpt else e_ckpt)
    return w_model, e_model


def _init_eco_state(sim: SimState) -> torch.Tensor:
    b, _, h, w = sim.mass.shape
    device = sim.mass.device
    _, t_ck = compute_C_and_T(sim.mass, sim.energy)
    rock = (sim.mass[:, S_ROCK:S_ROCK + 1] >= 500.0).float()
    water_bar = ((sim.mass[:, S_WATER:S_WATER + 1] >= 350.0) & (sim.mass[:, S_ICE:S_ICE + 1] < 700.0)).float()
    open_m = torch.clamp(1.0 - rock - water_bar, 0.0, 1.0)

    yy = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
    rho = torch.zeros((b, 1, h, w), device=device)
    for bi in range(b):
        for _ in range(5):
            cy = float(torch.randint(12, h - 12, (1,), device=device).item())
            cx = float(torch.randint(12, w - 12, (1,), device=device).item())
            rad = float(torch.empty(1, device=device).uniform_(3.0, 6.0).item())
            blob = torch.exp(-((yy[0, 0] - cy) ** 2 + (xx[0, 0] - cx) ** 2) / (2.0 * rad * rad))
            rho[bi, 0] += blob * 0.55
    rho = rho * open_m
    plant_n = sim.mass[:, S_PLANT:S_PLANT + 1] / 1000.0
    haz = torch.clamp(F.relu(t_ck - 38000.0) / 18000.0 + sim.mass[:, S_SMOKE:S_SMOKE + 1] / 150.0, 0.0, 2.0)
    zeros = torch.zeros_like(rho)
    return torch.cat([rho, zeros, zeros, rho * 0.4, plant_n, water_bar, haz, rock], dim=1)


def _step_eco(e_model: EcosystemModel, eco_state: torch.Tensor, sim: SimState) -> torch.Tensor:
    _, t_ck = compute_C_and_T(sim.mass, sim.energy)
    with torch.amp.autocast(device_type=sim.mass.device.type, dtype=torch.float16, enabled=sim.mass.device.type == "cuda"):
        eco_next = e_model(eco_state).float()
    plant_n = sim.mass[:, S_PLANT:S_PLANT + 1] / 1000.0
    rock = (sim.mass[:, S_ROCK:S_ROCK + 1] >= 500.0).float()
    water_bar = ((sim.mass[:, S_WATER:S_WATER + 1] >= 350.0) & (sim.mass[:, S_ICE:S_ICE + 1] < 700.0)).float()
    haz = torch.clamp(F.relu(t_ck - 38000.0) / 18000.0 + sim.mass[:, S_SMOKE:S_SMOKE + 1] / 150.0, 0.0, 2.0)
    return torch.cat([eco_next, plant_n, water_bar, haz, rock], dim=1)


def _enrich_frostmarsh_scene(
    mass: torch.Tensor,
    t_ck: torch.Tensor,
    flow_e: torch.Tensor,
    flow_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add rich realistic Frostmarsh mixtures (oil slicks, ice shelves, active fires, steam plumes)."""
    b, _, h, w = mass.shape
    device = mass.device
    mass = mass.clone()
    t_ck = t_ck.clone()
    flow_e = flow_e.clone()
    flow_s = flow_s.clone()

    n1 = smooth_noise(b, h, w, device, scale=12)
    n2 = smooth_noise(b, h, w, device, scale=6)

    # Oil slicks & occasional burning slicks on water
    open_water = (mass[:, S_WATER:S_WATER + 1] > 200.0) & (mass[:, S_ICE:S_ICE + 1] < 50.0)
    slick = open_water & (n1 > 0.64) & (n2 > 0.48)
    mass[:, S_OIL:S_OIL + 1] = torch.where(slick, mass[:, S_OIL:S_OIL + 1] + 240.0 + 360.0 * n2, mass[:, S_OIL:S_OIL + 1])
    burn_slick = slick & (n1 > 0.77)
    t_ck = torch.where(burn_slick, torch.maximum(t_ck, torch.full_like(t_ck, 56000.0)), t_ck)
    mass[:, S_SMOKE:S_SMOKE + 1] = torch.where(burn_slick, mass[:, S_SMOKE:S_SMOKE + 1] + 180.0 * n2, mass[:, S_SMOKE:S_SMOKE + 1])

    # Crisp frozen ice shelves along parts of rivers/ponds
    ice_zone = open_water & (n1 < 0.28) & (n2 > 0.45)
    mass[:, S_ICE:S_ICE + 1] = torch.where(ice_zone, mass[:, S_ICE:S_ICE + 1] + 520.0 + 280.0 * n2, mass[:, S_ICE:S_ICE + 1])
    t_ck = torch.where(ice_zone, torch.minimum(t_ck, torch.full_like(t_ck, 25800.0)), t_ck)

    # Steam plumes near hot water
    hot_w = open_water & (~ice_zone) & (n2 < 0.25) & (n1 > 0.45)
    mass[:, S_STEAM:S_STEAM + 1] = torch.where(hot_w, mass[:, S_STEAM:S_STEAM + 1] + 280.0 * (1.0 - n2), mass[:, S_STEAM:S_STEAM + 1])
    t_ck = torch.where(hot_w, torch.maximum(t_ck, torch.full_like(t_ck, 38200.0)), t_ck)

    # Ensure active glowing fire pockets for strong dynamic firelight training signal
    for bi in range(b):
        if bi % 2 == 0:
            ang = (bi % 8) * (math.pi / 4.0)
            mag_b = 130.0 if (bi % 3 == 0) else 65.0
            has_w = mass[bi, S_WATER] > 200.0
            flow_e[bi, 0] = torch.where(has_w, flow_e[bi, 0] + math.cos(ang) * mag_b, flow_e[bi, 0])
            flow_s[bi, 0] = torch.where(has_w, flow_s[bi, 0] + math.sin(ang) * mag_b, flow_s[bi, 0])

    return mass, t_ck, flow_e, flow_s


@torch.no_grad()
def collect_frostmarsh_dataset(
    w_model: WorldModel,
    e_model: EcosystemModel,
    n_worlds: int,
    n_steps: int,
    sample_every: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Run real Frostmarsh 128x128 rollouts with `world_model.pt` + `ecosystem_model.pt`."""
    sim = make_scene(n_worlds, 128, 128, device, scene_type="mixed", seed=seed)
    eco = _init_eco_state(sim)

    masses, temps, fes, fss, crits, elevs = [], [], [], [], [], []
    for step in range(n_steps):
        fe, fm = sample_bolts(n_worlds, 128, 128, device, prob=0.45, max_bolts=2)
        x_w = encode_state(sim, fe, fm)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw_deltas = w_model(x_w).float()
        sim, _, _ = project_and_step(sim, raw_deltas, fe, fm)
        eco = _step_eco(e_model, eco, sim)

        if (step + 1) % sample_every == 0:
            _, t_ck = compute_C_and_T(sim.mass, sim.energy)
            m_s, t_s, fl_e, fl_s = _enrich_frostmarsh_scene(sim.mass, t_ck, sim.flow_e, sim.flow_s)
            crit_n = torch.clamp(eco[:, 0:1] / 0.55, 0.0, 1.0)
            masses.append(m_s.cpu())
            temps.append(t_s.cpu())
            fes.append(fl_e.cpu())
            fss.append(fl_s.cpu())
            crits.append(crit_n.cpu())
            elevs.append((sim.elevation / 31.0).cpu())

    return {
        "mass": torch.cat(masses, dim=0),
        "t_ck": torch.cat(temps, dim=0),
        "flow_e": torch.cat(fes, dim=0),
        "flow_s": torch.cat(fss, dim=0),
        "critter": torch.cat(crits, dim=0),
        "elev": torch.cat(elevs, dim=0),
    }


@torch.no_grad()
def build_hero_battle_scene(
    w_model: WorldModel,
    e_model: EcosystemModel,
    device: torch.device,
    seed: int = 2026,
) -> dict[str, torch.Tensor]:
    """
    Build a dramatic, physically simulated 128x128 Frostmarsh live battle frame featuring:
    - Winding turquoise river with glacier ice bridge/shelf
    - Towering craggy granite cliffs on one flank & lush pine forest on the other
    - Active Fire Bolt ignition blazing across pine woods & oil slick (volumetric firelight!)
    - Frost Bolt chill rim & steam clouds rising from river edge
    - Golden wildlife herd fleeing the blaze
    """
    torch.manual_seed(seed)
    sim = make_scene(1, 128, 128, device, scene_type="river_crossing", seed=seed)
    eco = _init_eco_state(sim)

    yy = torch.linspace(0.0, 1.0, 128, device=device).view(1, 1, 128, 1)
    xx = torch.linspace(0.0, 1.0, 128, device=device).view(1, 1, 1, 128)
    n1 = smooth_noise(1, 128, 128, device, scale=10)
    n2 = smooth_noise(1, 128, 128, device, scale=5)

    # Sculpt dramatic cliff ridges & thick pine grove + glacier shelf + oil slick for visual impact
    mass = sim.mass.clone()
    in_river = mass[:, S_WATER:S_WATER + 1] > 250.0
    # Craggy cliffs on NW/W & SE flanks
    cliff1 = (((yy - 0.30) / 0.16) ** 2 + ((xx - 0.22) / 0.13) ** 2) < (0.85 + 0.35 * n2)
    cliff2 = (((yy - 0.72) / 0.15) ** 2 + ((xx - 0.80) / 0.12) ** 2) < (0.80 + 0.35 * n1)
    cliff = (cliff1 | cliff2) & (~in_river)
    mass[:, S_ROCK:S_ROCK + 1] = torch.where(cliff, 1150.0, mass[:, S_ROCK:S_ROCK + 1])
    mass[:, S_WOOD:S_WOOD + 1] = torch.where(cliff, 0.0, mass[:, S_WOOD:S_WOOD + 1])

    # Dense pine forest on NE & SW banks
    pine1 = (((yy - 0.28) / 0.20) ** 2 + ((xx - 0.75) / 0.17) ** 2) < (0.9 + 0.4 * n2)
    pine2 = (((yy - 0.74) / 0.18) ** 2 + ((xx - 0.25) / 0.16) ** 2) < (0.9 + 0.4 * n2)
    pine = (pine1 | pine2) & (~in_river) & (~cliff) & (n2 > 0.32)
    mass[:, S_WOOD:S_WOOD + 1] = torch.where(pine, 780.0 + 300.0 * n2, mass[:, S_WOOD:S_WOOD + 1])

    # Glacier ice shelf across southern river bend
    ice_shelf = in_river & (yy > 0.56) & (yy < 0.78) & (n2 > 0.28)
    mass[:, S_ICE:S_ICE + 1] = torch.where(ice_shelf, 820.0 + 200.0 * n2, mass[:, S_ICE:S_ICE + 1])
    mass[:, S_WATER:S_WATER + 1] = torch.where(ice_shelf, 120.0, mass[:, S_WATER:S_WATER + 1])

    # Oil slick floating on northern river bend
    oil_slick = in_river & (yy > 0.22) & (yy < 0.44) & (n1 > 0.38)
    mass[:, S_OIL:S_OIL + 1] = torch.where(oil_slick, 520.0 + 280.0 * n2, mass[:, S_OIL:S_OIL + 1])

    # Rebuild consistent thermal energy
    c = get_consts(device)
    _, t_ck = compute_C_and_T(sim.mass, sim.energy)
    t_ck = torch.where(ice_shelf, torch.full_like(t_ck, 25000.0), t_ck)
    C_cell = (mass * c.spec_c).sum(dim=1, keepdim=True)
    sim = SimState(mass=mass, energy=C_cell * t_ck, flow_e=sim.flow_e, flow_s=sim.flow_s, elevation=sim.elevation)

    # Step 14 real ticks with targeted Fire Bolts (near cliffs/pines/river) & Frost Bolt (on ice shelf)
    y_px = torch.arange(128, device=device, dtype=torch.float32).view(1, 1, 128, 1)
    x_px = torch.arange(128, device=device, dtype=torch.float32).view(1, 1, 1, 128)
    for step in range(14):
        fe = torch.zeros((1, 1, 128, 128), device=device)
        fm = torch.zeros_like(fe)
        if step in (1, 3, 6, 9):
            # Fire bolt hitting eastern pine bank & northern oil slick (casts dramatic bounce light onto cliffs & water!)
            for (cy, cx, rad, mag) in [(44.0, 82.0, 9.5, 4.2e8), (42.0, 62.0, 7.5, 3.6e8), (82.0, 44.0, 7.0, 3.5e8)]:
                k = torch.clamp(1.0 - ((y_px - cy) ** 2 + (x_px - cx) ** 2) / (rad * rad), min=0.0)
                fe = fe + k * mag
        if step in (2, 7):
            # Frost bolt chilling southern glacier bridge
            k = torch.clamp(1.0 - ((y_px - 86.0) ** 2 + (x_px - 66.0) ** 2) / (10.0 * 10.0), min=0.0)
            fe = fe - k * 3.0e8
        x_w = encode_state(sim, fe, fm)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            raw_deltas = w_model(x_w).float()
        sim, _, _ = project_and_step(sim, raw_deltas, fe, fm)
        eco = _step_eco(e_model, eco, sim)

    _, t_ck = compute_C_and_T(sim.mass, sim.energy)
    crit_n = torch.clamp(eco[:, 0:1] / 0.55, 0.0, 1.0)
    return {
        "mass": sim.mass.cpu(),
        "t_ck": t_ck.cpu(),
        "flow_e": sim.flow_e.cpu(),
        "flow_s": sim.flow_s.cpu(),
        "critter": crit_n.cpu(),
        "elev": (sim.elevation / 31.0).cpu(),
    }
