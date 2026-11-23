"""
Phase 3 Bonus (G5-B): Living Ecosystem Neural Field (`ecosystem_model.onnx`).

1. Fast GPU-vectorized 500-agent Frostmarsh ecosystem simulator:
   - 4 territorial herds (500 critters total) with Boids flocking, cohesion, alignment, separation
   - Plant foraging & grazing depletion
   - Wildfire/smoke panic fleeing + social fear contagion across herd members
   - Rock wall & deep water avoidance; walkable ice bridges (ice >= 700) unlock river crossings!
2. Distills the 500-agent non-equational dynamics into a 10 Hz neural field:
   `EcosystemModel`: [B, 8, H, W] -> [B, 4, H, W]
   (critter density, fear, forage depletion, herd momentum)
3. Trains an optimized classical Reaction-Diffusion + Chemotaxis PDE baseline on the same data
   and compares both on held-out Frostmarsh scenarios (including ice-bridge crossing & wildfire panic).
4. Exports single self-contained `ecosystem_model.onnx` (opset 17, dynamic B/H/W) and verifies in ORT.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EcosystemModel, count_parameters
from sim_teacher import (
    RefSim,
    S_ICE,
    S_PLANT,
    S_ROCK,
    S_SMOKE,
    S_WATER,
    SimState,
    compute_C_and_T,
    make_scene,
    sample_bolts,
)

OUT_DIR = Path("/home/gooning/.t3/scratch/frostmarsh-ml")


def _gaussian_kernel2d(sigma: float, radius: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.sum()
    return g2d.view(1, 1, 2 * radius + 1, 2 * radius + 1)


def _blur2d(x: torch.Tensor, sigma: float = 1.0, radius: int = 2) -> torch.Tensor:
    k = _gaussian_kernel2d(sigma, radius, x.device)
    c = x.shape[1]
    k = k.expand(c, 1, -1, -1)
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(x_pad, k, groups=c)


def _grad2d(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central difference spatial gradient (gx, gy) of [B, 1, H, W]."""
    f_pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (f_pad[:, :, 1:-1, 2:] - f_pad[:, :, 1:-1, :-2])
    gy = 0.5 * (f_pad[:, :, 2:, 1:-1] - f_pad[:, :, :-2, 1:-1])
    return gx, gy


def _sample_field(field: torch.Tensor, pos_xy: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B, C, H, W] at continuous pixel coords pos_xy [B, N, 2] -> [B, N, C]."""
    B, C, H, W = field.shape
    gx = 2.0 * (pos_xy[..., 0] / max(W - 1, 1)) - 1.0
    gy = 2.0 * (pos_xy[..., 1] / max(H - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B, N, 1, 2]
    vals = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return vals.squeeze(-1).permute(0, 2, 1)  # [B, N, C]


def _splat_agents(pos_xy: torch.Tensor, weights: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Bilinearly splat agent quantities weights [B, N, C] onto grid [B, C, H, W].
    """
    B, N, C = weights.shape
    device = pos_xy.device
    x = torch.clamp(pos_xy[..., 0], 0.0, W - 1.001)
    y = torch.clamp(pos_xy[..., 1], 0.0, H - 1.001)

    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = torch.clamp(x0 + 1, max=W - 1)
    y1 = torch.clamp(y0 + 1, max=H - 1)

    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)

    out = torch.zeros((B, H * W, C), dtype=torch.float32, device=device)

    for (ix, iy, w) in [
        (x0, y0, (1.0 - fx) * (1.0 - fy)),
        (x1, y0, fx * (1.0 - fy)),
        (x0, y1, (1.0 - fx) * fy),
        (x1, y1, fx * fy),
    ]:
        idx = (iy * W + ix).unsqueeze(-1).expand(B, N, C)
        out.scatter_add_(1, idx, weights * w)

    return out.permute(0, 2, 1).view(B, C, H, W)


def extract_env_channels(sim: SimState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (plant_norm, water_barrier, hazard, rock_wall) each [B, 1, H, W]."""
    _, T_cK = compute_C_and_T(sim.mass, sim.energy)
    plant_norm = sim.mass[:, S_PLANT:S_PLANT + 1] / 1000.0
    rock_wall = (sim.mass[:, S_ROCK:S_ROCK + 1] >= 500.0).to(torch.float32)
    water_barrier = (
        (sim.mass[:, S_WATER:S_WATER + 1] >= 350.0)
        & (sim.mass[:, S_ICE:S_ICE + 1] < 700.0)
    ).to(torch.float32)
    fire_haz = F.relu(T_cK - 38000.0) / 18000.0
    smoke_haz = sim.mass[:, S_SMOKE:S_SMOKE + 1] / 150.0
    hazard = torch.clamp(fire_haz + smoke_haz, 0.0, 2.0)
    return plant_norm, water_barrier, hazard, rock_wall


class AgentEcosystemSim:
    """
    GPU-vectorized 500-agent Frostmarsh ecosystem simulator.
    Non-equational discrete dynamics:
    - 4 territorial herds with Boids alignment, cohesion, personal separation, rival-herd repulsion
    - Foraging toward plants + grazing depletion
    - Fire/smoke panic fleeing + social panic contagion across nearby herd members
    - Avoidance of rock walls & deep unfrozen water; crosses frozen rivers over ice bridges!
    """

    def __init__(self, sim_state: SimState, n_agents: int = 500, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)
        self.sim = sim_state
        self.B, _, self.H, self.W = sim_state.mass.shape
        self.N = n_agents
        self.device = sim_state.mass.device

        # Assign 4 territorial herds
        self.herd_id = (torch.arange(n_agents, device=self.device) % 4).view(1, n_agents).expand(self.B, n_agents)
        self.same_herd = (self.herd_id.unsqueeze(2) == self.herd_id.unsqueeze(1)).to(torch.float32)  # [B, N, N]
        eye = torch.eye(n_agents, device=self.device).unsqueeze(0)
        self.same_herd = self.same_herd * (1.0 - eye)
        self.diff_herd = (1.0 - self.same_herd) * (1.0 - eye)

        # Spawn critters in 4 herd clusters on valid (non-blocked) cells
        _, water_bar, _, rock_wall = extract_env_channels(sim_state)
        blocked = (water_bar > 0.5) | (rock_wall > 0.5)  # [B, 1, H, W]

        pos = torch.zeros((self.B, n_agents, 2), dtype=torch.float32, device=self.device)
        for b in range(self.B):
            valid = (~blocked[b, 0]).nonzero(as_tuple=False)  # [K, 2] (y, x)
            if valid.shape[0] < 4:
                valid = torch.tensor([[self.H // 2, self.W // 2]], device=self.device)
            # Pick 4 herd anchor centers on valid land
            anchor_idx = torch.randint(0, valid.shape[0], (4,), device=self.device)
            anchors = valid[anchor_idx].float()  # [4, 2] (y, x)
            for h_id in range(4):
                m = (self.herd_id[b] == h_id)
                cnt = int(m.sum().item())
                cy, cx = anchors[h_id, 0], anchors[h_id, 1]
                jitter = torch.randn((cnt, 2), device=self.device) * 3.5
                px = torch.clamp(cx + jitter[:, 0], 1.5, self.W - 2.5)
                py = torch.clamp(cy + jitter[:, 1], 1.5, self.H - 2.5)
                # Snap any agent spawned in water/rock onto a valid cell near anchor
                ix = px.long().clamp(0, self.W - 1)
                iy = py.long().clamp(0, self.H - 1)
                bad = blocked[b, 0, iy, ix]
                if bad.any():
                    repl = valid[torch.randint(0, valid.shape[0], (int(bad.sum().item()),), device=self.device)]
                    px[bad] = repl[:, 1].float()
                    py[bad] = repl[:, 0].float()
                pos[b, m, 0] = px
                pos[b, m, 1] = py

        self.pos = pos
        self.vel = torch.randn((self.B, n_agents, 2), device=self.device) * 0.4
        self.fear = torch.zeros((self.B, n_agents), device=self.device)
        self.prev_forage = torch.zeros((self.B, 1, self.H, self.W), device=self.device)

    @torch.no_grad()
    def get_fields(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (density, fear_field, herd_momentum) each [B, 1, H, W]."""
        speed = torch.linalg.norm(self.vel, dim=-1, keepdim=True)  # [B, N, 1]
        ones = torch.ones_like(speed)
        w = torch.cat([ones, self.fear.unsqueeze(-1), speed], dim=-1)  # [B, N, 3]
        splat = _splat_agents(self.pos, w, self.H, self.W)
        # Smooth with sigma=0.85 so grid representation is well-conditioned for 10Hz CNN
        splat = _blur2d(splat, sigma=0.85, radius=2)
        density = splat[:, 0:1]
        fear_field = torch.clamp(splat[:, 1:2] / (density + 0.08), 0.0, 1.0)
        momentum = splat[:, 2:3]
        return density, fear_field, momentum

    @torch.no_grad()
    def encode_eco_state(self) -> torch.Tensor:
        """Build exact [B, 8, H, W] input tensor for EcosystemModel."""
        density, fear_field, momentum = self.get_fields()
        plant_norm, water_bar, hazard, rock_wall = extract_env_channels(self.sim)
        return torch.cat([
            density,
            fear_field,
            self.prev_forage,
            momentum,
            plant_norm,
            water_bar,
            hazard,
            rock_wall,
        ], dim=1)

    @torch.no_grad()
    def step(
        self,
        forcing_energy: torch.Tensor | None = None,
        forcing_mass: torch.Tensor | None = None,
        step_physics: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Advance ecosystem + physics by 1 tick (dt = 0.1s).
        Returns: (eco_input [B, 8, H, W], eco_target [B, 4, H, W])
        """
        eco_in = self.encode_eco_state()

        if step_physics:
            self.sim, _ = RefSim.step(self.sim, forcing_energy, forcing_mass)

        plant_norm, water_bar, hazard, rock_wall = extract_env_channels(self.sim)
        blocked = torch.clamp(water_bar + rock_wall, 0.0, 1.0)

        # Precompute blurred guidance fields & gradients for agent perception
        plant_sense = _blur2d(plant_norm, sigma=2.2, radius=4)
        haz_sense = _blur2d(hazard, sigma=2.4, radius=5)
        blk_sense = _blur2d(blocked, sigma=1.4, radius=3)

        px_g, py_g = _grad2d(plant_sense)
        hx_g, hy_g = _grad2d(haz_sense)
        bx_g, by_g = _grad2d(blk_sense)
        env_pack = torch.cat([px_g, py_g, hx_g, hy_g, bx_g, by_g, haz_sense, plant_norm], dim=1)

        # 2 substeps per 0.1s tick for smooth agent steering
        for _ in range(2):
            sampled = _sample_field(env_pack, self.pos)  # [B, N, 8]
            grad_plant = sampled[..., 0:2]
            grad_haz = sampled[..., 2:4]
            grad_blk = sampled[..., 4:6]
            local_haz = sampled[..., 6]  # [B, N]

            # Pairwise agent vectors for Boids + social panic
            # diff[b, i, j] = pos[j] - pos[i]
            diff = self.pos.unsqueeze(1) - self.pos.unsqueeze(2)  # [B, N, N, 2]
            dist2 = (diff ** 2).sum(dim=-1) + 1e-4  # [B, N, N]
            dist = torch.sqrt(dist2)

            # 1. Personal separation (d < 1.8)
            sep_mask = ((dist < 1.8) & (dist > 1e-2)).to(torch.float32)
            sep_w = sep_mask / (dist2 + 0.15)
            f_sep = -(diff * sep_w.unsqueeze(-1)).sum(dim=2) / (sep_mask.sum(dim=2, keepdim=True) + 1.0)

            # 2. Same-herd cohesion & velocity alignment (d < 7.0)
            coh_mask = (dist < 7.5).to(torch.float32) * self.same_herd
            coh_cnt = coh_mask.sum(dim=2, keepdim=True) + 1e-4
            centroid_dir = (diff * coh_mask.unsqueeze(-1)).sum(dim=2) / coh_cnt
            f_coh = centroid_dir * (coh_mask.sum(dim=2, keepdim=True) > 0.5).to(torch.float32)

            ali_mask = (dist < 5.0).to(torch.float32) * self.same_herd
            ali_cnt = ali_mask.sum(dim=2, keepdim=True) + 1e-4
            avg_vel = torch.bmm(ali_mask, self.vel) / ali_cnt
            f_ali = (avg_vel - self.vel) * (ali_mask.sum(dim=2, keepdim=True) > 0.5).to(torch.float32)

            # 3. Rival-herd territorial repulsion (d < 4.5)
            riv_mask = (dist < 4.5).to(torch.float32) * self.diff_herd
            f_riv = -(diff * (riv_mask / (dist + 0.5)).unsqueeze(-1)).sum(dim=2) / (riv_mask.sum(dim=2, keepdim=True) + 1.0)

            # 4. Fear & social panic contagion
            self.fear = torch.clamp(self.fear * 0.94 + 1.8 * F.relu(local_haz - 0.04), 0.0, 1.0)
            panic_mask = (dist < 5.5).to(torch.float32) * self.same_herd
            nb_fear = (panic_mask * self.fear.unsqueeze(1)).max(dim=2).values
            self.fear = torch.maximum(self.fear, 0.88 * nb_fear)

            # 5. Combine steering forces
            calm = (1.0 - 0.8 * self.fear).unsqueeze(-1)
            panic = self.fear.unsqueeze(-1)

            accel = (
                1.6 * f_sep
                + 0.35 * f_coh
                + 0.55 * f_ali
                + 0.65 * f_riv
                + 4.5 * calm * grad_plant
                - 14.0 * (0.4 + panic) * grad_haz
                - 6.5 * grad_blk
                + 0.12 * torch.randn_like(self.vel)
            )

            self.vel = 0.82 * self.vel + 0.35 * accel
            max_spd = 0.85 + 1.35 * self.fear  # panic sprint!
            spd = torch.linalg.norm(self.vel, dim=-1, keepdim=True) + 1e-6
            self.vel = self.vel * torch.minimum(torch.ones_like(spd), max_spd.unsqueeze(-1) / spd)

            new_pos = self.pos + 0.45 * self.vel
            new_pos[..., 0] = torch.clamp(new_pos[..., 0], 1.0, self.W - 2.0)
            new_pos[..., 1] = torch.clamp(new_pos[..., 1], 1.0, self.H - 2.0)

            # Strict barrier collision check: reject move into blocked (rock or unfrozen deep water)
            blk_new = _sample_field(blocked, new_pos).squeeze(-1) > 0.35
            self.vel = torch.where(blk_new.unsqueeze(-1), -0.6 * self.vel, self.vel)
            self.pos = torch.where(blk_new.unsqueeze(-1), self.pos, new_pos)

        # Grazing / forage depletion on plant mass
        graze_req = (8.0 * (1.0 - 0.85 * self.fear)).unsqueeze(-1)  # [B, N, 1] mass units
        graze_grid = _splat_agents(self.pos, graze_req, self.H, self.W)  # [B, 1, H, W]
        avail_plant = self.sim.mass[:, S_PLANT:S_PLANT + 1]
        actual_graze = torch.minimum(graze_grid, avail_plant)
        # Update plant mass & its thermal energy cleanly
        _, T_cK = compute_C_and_T(self.sim.mass, self.sim.energy)
        self.sim.mass[:, S_PLANT:S_PLANT + 1] = avail_plant - actual_graze
        self.sim.energy = self.sim.energy - actual_graze * 25.0 * T_cK

        forage_out = _blur2d(actual_graze / 100.0, sigma=0.6, radius=1)
        self.prev_forage = forage_out

        next_density, next_fear, next_momentum = self.get_fields()
        eco_target = torch.cat([next_density, next_fear, forage_out, next_momentum], dim=1)
        return eco_in, eco_target


class ReactionDiffusionBaseline(nn.Module):
    """
    Classical tuned Reaction-Diffusion + Chemotaxis PDE baseline (14 learnable physical constants).
    Represents the best classical analytical PDE alternative to the neural ecosystem field.
    """

    def __init__(self):
        super().__init__()
        # Diffusion, chemotaxis, hazard repulsion, decay/grazing/momentum scalars
        self.D_rho = nn.Parameter(torch.tensor(0.08))
        self.chi_plant = nn.Parameter(torch.tensor(0.25))
        self.chi_haz = nn.Parameter(torch.tensor(0.60))
        self.f_bias = nn.Parameter(torch.tensor(-2.0))
        self.f_self = nn.Parameter(torch.tensor(3.5))
        self.f_lap = nn.Parameter(torch.tensor(0.5))
        self.f_haz = nn.Parameter(torch.tensor(4.0))
        self.f_rho = nn.Parameter(torch.tensor(0.1))
        self.k_graze = nn.Parameter(torch.tensor(0.08))
        self.b_fear = nn.Parameter(torch.tensor(0.5))
        self.m_self = nn.Parameter(torch.tensor(0.6))
        self.m_rho = nn.Parameter(torch.tensor(0.3))
        self.m_fear = nn.Parameter(torch.tensor(0.5))
        self.m_grad = nn.Parameter(torch.tensor(0.2))

        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        self.register_buffer("lap_k", lap.view(1, 1, 3, 3))

    def _lap(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), self.lap_k)

    def forward(self, eco_state: torch.Tensor) -> torch.Tensor:
        rho = eco_state[:, 0:1]
        fear = eco_state[:, 1:2]
        mom = eco_state[:, 3:4]
        plant = eco_state[:, 4:5]
        water_bar = eco_state[:, 5:6]
        haz = eco_state[:, 6:7]
        rock = eco_state[:, 7:8]
        open_mask = torch.clamp(1.0 - water_bar - rock, 0.0, 1.0)

        # Reaction-diffusion + chemotaxis flux divergence
        px, py = _grad2d(plant)
        hx, hy = _grad2d(haz)
        jx = self.chi_plant * rho * px - self.chi_haz * rho * hx
        jy = self.chi_plant * rho * py - self.chi_haz * rho * hy
        jx_x, _ = _grad2d(jx)
        _, jy_y = _grad2d(jy)
        div_j = jx_x + jy_y

        D = F.softplus(self.D_rho) * 0.25
        next_rho = F.relu(rho + D * self._lap(rho) - div_j) * open_mask

        next_fear = torch.sigmoid(
            self.f_bias + self.f_self * fear + self.f_lap * self._lap(fear) + self.f_haz * haz + self.f_rho * rho
        )

        k_g = F.softplus(self.k_graze)
        forage = F.relu(k_g * rho * plant * torch.clamp(1.0 - torch.sigmoid(self.b_fear) * fear, 0.0, 1.0))

        grad_mag = torch.sqrt(px ** 2 + py ** 2 + hx ** 2 + hy ** 2 + 1e-6)
        next_mom = F.relu(
            F.softplus(self.m_self) * mom
            + F.softplus(self.m_rho) * rho
            + F.softplus(self.m_fear) * fear * rho
            + F.softplus(self.m_grad) * grad_mag * rho
        ) * open_mask

        return torch.cat([next_rho, next_fear, forage, next_mom], dim=1)


def eco_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    l_rho = F.smooth_l1_loss(pred[:, 0:1], target[:, 0:1], beta=0.05)
    l_fear = F.smooth_l1_loss(pred[:, 1:2], target[:, 1:2], beta=0.05)
    l_forage = F.smooth_l1_loss(pred[:, 2:3], target[:, 2:3], beta=0.02)
    l_mom = F.smooth_l1_loss(pred[:, 3:4], target[:, 3:4], beta=0.05)
    tot = 4.0 * l_rho + 3.0 * l_fear + 6.0 * l_forage + 4.0 * l_mom
    return tot, {
        "rho": float(l_rho.item()),
        "fear": float(l_fear.item()),
        "forage": float(l_forage.item()),
        "mom": float(l_mom.item()),
    }


def collect_trajectories(
    n_worlds: int,
    n_steps: int,
    H: int,
    W: int,
    device: torch.device,
    seed: int,
    ice_bridge_event: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect [n_steps, n_worlds, 8, H, W] inputs and [n_steps, n_worlds, 4, H, W] targets
    from the 500-agent Frostmarsh ecosystem simulator.
    """
    sim = make_scene(n_worlds, H, W, device, scene_type="mixed", seed=seed)
    eco = AgentEcosystemSim(sim, n_agents=500, seed=seed + 1000)

    # Warm up 10 steps so herds settle onto terrain
    for _ in range(10):
        eco.step(step_physics=True)

    xs, ys = [], []
    for t in range(n_steps):
        fe, fm = sample_bolts(n_worlds, H, W, device, prob=0.18, max_bolts=1)
        if ice_bridge_event and (15 <= t <= 22):
            # Cast strong Frost Bolt straight across river center to create a walkable ice bridge!
            yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
            xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
            cy, cx = H * 0.5, W * 0.5
            k = torch.clamp(1.0 - ((yy - cy) ** 2 + (xx - cx) ** 2) / (6.0 ** 2), min=0.0)
            fe[:, 0] -= k * 3.5e8
        if ice_bridge_event and (t == 45):
            # Cast Fire Bolt on one bank to trigger herd stampede!
            yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
            xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
            cy, cx = H * 0.35, W * 0.3
            k = torch.clamp(1.0 - ((yy - cy) ** 2 + (xx - cx) ** 2) / (5.0 ** 2), min=0.0)
            fe[:, 0] += k * 3.8e8

        x, y = eco.step(fe, fm, step_physics=True)
        xs.append(x)
        ys.append(y)

    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def eval_closed_loop(
    model: nn.Module,
    xs_val: torch.Tensor,
    ys_val: torch.Tensor,
    horizons: tuple[int, ...] = (1, 10, 50),
) -> dict[str, float]:
    """
    Evaluate 1-step and multi-step closed-loop autoregressive rollout against 500-agent ground truth.
    xs_val: [T, B, 8, H, W], ys_val: [T, B, 4, H, W]
    """
    model.eval()
    T, B, _, H, W = xs_val.shape
    metrics = {}
    with torch.no_grad():
        # 1-step across all steps
        x_flat = xs_val.reshape(T * B, 8, H, W)
        y_flat = ys_val.reshape(T * B, 4, H, W)
        p_flat = model(x_flat)
        metrics["step1_mae_all"] = float((p_flat - y_flat).abs().mean().item())
        metrics["step1_rmse_all"] = float(torch.sqrt(((p_flat - y_flat) ** 2).mean()).item())
        metrics["step1_mae_density"] = float((p_flat[:, 0] - y_flat[:, 0]).abs().mean().item())
        metrics["step1_mae_fear"] = float((p_flat[:, 1] - y_flat[:, 1]).abs().mean().item())
        metrics["step1_mae_forage"] = float((p_flat[:, 2] - y_flat[:, 2]).abs().mean().item())
        metrics["step1_mae_momentum"] = float((p_flat[:, 3] - y_flat[:, 3]).abs().mean().item())

        # Closed-loop rollout from t=0 using ground-truth physics channels (4..8) + predicted eco channels (0..4)
        cur = xs_val[0].clone()  # [B, 8, H, W]
        for t in range(max(horizons)):
            pred = model(cur)  # [B, 4, H, W]
            step_num = t + 1
            if step_num in horizons:
                gt = ys_val[t]
                metrics[f"roll{step_num}_mae_all"] = float((pred - gt).abs().mean().item())
                metrics[f"roll{step_num}_mae_density"] = float((pred[:, 0] - gt[:, 0]).abs().mean().item())
                metrics[f"roll{step_num}_mae_fear"] = float((pred[:, 1] - gt[:, 1]).abs().mean().item())
                metrics[f"roll{step_num}_mae_forage"] = float((pred[:, 2] - gt[:, 2]).abs().mean().item())
                metrics[f"roll{step_num}_mae_momentum"] = float((pred[:, 3] - gt[:, 3]).abs().mean().item())
                # Spatial correlation on critter density
                p0 = pred[:, 0].reshape(B, -1)
                g0 = gt[:, 0].reshape(B, -1)
                p0c = p0 - p0.mean(dim=1, keepdim=True)
                g0c = g0 - g0.mean(dim=1, keepdim=True)
                corr = (p0c * g0c).sum(dim=1) / (
                    torch.sqrt((p0c ** 2).sum(dim=1) * (g0c ** 2).sum(dim=1)) + 1e-8
                )
                metrics[f"roll{step_num}_density_corr"] = float(corr.mean().item())

            if t + 1 < T:
                # Feed predicted [density, fear, forage, momentum] + next tick's physics environment channels
                cur = torch.cat([pred, xs_val[t + 1, :, 4:8]], dim=1)
    return metrics


def export_and_verify_eco_onnx(model: EcosystemModel, onnx_path: Path) -> dict:
    model_cpu = EcosystemModel().eval().cpu()
    model_cpu.load_state_dict(model.state_dict())
    dummy = torch.randn(1, 8, 64, 64, dtype=torch.float32)

    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    if onnx_path.exists():
        onnx_path.unlink()

    torch.onnx.export(
        model_cpu,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["eco_state"],
        output_names=["eco_next"],
        dynamic_axes={
            "eco_state": {0: "batch", 2: "height", 3: "width"},
            "eco_next": {0: "batch", 2: "height", 3: "width"},
        },
        dynamo=False,
    )

    proto = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(proto, str(onnx_path), save_as_external_data=False)
    if data_file.exists():
        data_file.unlink()

    onnx.checker.check_model(str(onnx_path))
    assert not data_file.exists(), "External .data file must not exist!"

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    stats = {
        "path": str(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "opset": 17,
        "input_name": sess.get_inputs()[0].name,
        "output_name": sess.get_outputs()[0].name,
    }
    for (B, H, W) in [(1, 128, 128), (1, 256, 256)]:
        x = torch.rand(B, 8, H, W, dtype=torch.float32)
        with torch.no_grad():
            y_pt = model_cpu(x).numpy()
        y_ort = sess.run(["eco_next"], {"eco_state": x.numpy()})[0]
        diff = float(np.max(np.abs(y_pt - y_ort)))
        assert y_ort.shape == (B, 4, H, W)
        assert diff < 1e-4, f"ORT mismatch {diff}"
        stats[f"ort_max_diff_{H}x{W}"] = diff
        print(f"  [Eco ONNX OK] {B}x8x{H}x{W} -> {y_ort.shape}, max_abs_diff={diff:.2e}")
    return stats


def main():
    torch.manual_seed(77)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Phase 3 Ecosystem Training on {device}")

    t0 = time.time()
    print("Simulating 500-agent Frostmarsh ecosystem trajectories on GPU...")
    # Training set: 24 worlds x 80 ticks at 64x64 (1,920 full grid transitions!)
    xs_tr_a, ys_tr_a = collect_trajectories(12, 80, 64, 64, device, seed=10, ice_bridge_event=True)
    xs_tr_b, ys_tr_b = collect_trajectories(12, 80, 64, 64, device, seed=20, ice_bridge_event=False)
    xs_tr = torch.cat([xs_tr_a, xs_tr_b], dim=1)  # [80, 24, 8, 64, 64]
    ys_tr = torch.cat([ys_tr_a, ys_tr_b], dim=1)  # [80, 24, 4, 64, 64]

    # Held-out validation set at 128x128 (generalization to 4x larger grid + unseen seeds + ice bridge + fire stampede!)
    xs_val, ys_val = collect_trajectories(6, 60, 128, 128, device, seed=999, ice_bridge_event=True)
    print(f"Collected train={tuple(xs_tr.shape)}, val={tuple(xs_val.shape)} in {time.time() - t0:.1f}s")

    # 1. Train Neural Ecosystem Field (with multi-step rollout k=1 -> 2 -> 4 -> 8)
    eco_model = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).to(device)
    n_params = count_parameters(eco_model)
    print(f"\nTraining EcosystemModel ({n_params:,} params)...")
    opt = torch.optim.AdamW(eco_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1400, eta_min=1e-4)

    T_tr, B_tr, _, H_tr, W_tr = xs_tr.shape
    for step in range(1, 1401):
        k_roll = 1 if step <= 400 else (2 if step <= 750 else (4 if step <= 1100 else 8))
        b_idx = torch.randint(0, B_tr, (12,), device=device)
        t_start = int(torch.randint(0, T_tr - k_roll, (1,)).item())

        cur = xs_tr[t_start, b_idx].clone()
        loss_tot = torch.tensor(0.0, device=device)
        for r in range(k_roll):
            noise = torch.zeros_like(cur)
            noise[:, :4] = torch.randn_like(cur[:, :4]) * 0.003
            pred = eco_model(cur + noise)
            l, comp = eco_loss(pred, ys_tr[t_start + r, b_idx])
            loss_tot = loss_tot + l / k_roll
            if r + 1 < k_roll:
                cur = torch.cat([pred, xs_tr[t_start + r + 1, b_idx, 4:8]], dim=1)

        opt.zero_grad(set_to_none=True)
        loss_tot.backward()
        torch.nn.utils.clip_grad_norm_(eco_model.parameters(), 2.0)
        opt.step()
        sched.step()

        if step % 200 == 0:
            print(
                f"  eco step {step:4d}/1400 (k={k_roll}) | loss={loss_tot.item():.5f} "
                f"[rho={comp['rho']:.5f} fear={comp['fear']:.5f} forage={comp['forage']:.5f} mom={comp['mom']:.5f}]"
            )

    # 2. Train Classical Reaction-Diffusion + Chemotaxis PDE Baseline on identical data
    rd_base = ReactionDiffusionBaseline().to(device)
    print(f"\nTraining Reaction-Diffusion PDE Baseline ({count_parameters(rd_base)} params)...")
    opt_rd = torch.optim.Adam(rd_base.parameters(), lr=2e-2)
    for step in range(1, 601):
        b_idx = torch.randint(0, B_tr, (16,), device=device)
        t_idx = int(torch.randint(0, T_tr, (1,)).item())
        pred = rd_base(xs_tr[t_idx, b_idx])
        l, _ = eco_loss(pred, ys_tr[t_idx, b_idx])
        opt_rd.zero_grad(set_to_none=True)
        l.backward()
        opt_rd.step()
        if step % 200 == 0:
            print(f"  rd_base step {step:4d}/600 | loss={l.item():.5f}")

    # 3. Evaluate both on held-out 128x128 Frostmarsh ecosystem scenes
    neural_metrics = eval_closed_loop(eco_model, xs_val, ys_val, horizons=(1, 10, 50))
    rd_metrics = eval_closed_loop(rd_base, xs_val, ys_val, horizons=(1, 10, 50))

    print("\n=== G5-B Held-out 128x128 Comparison (Neural Field vs Reaction-Diffusion Baseline) ===")
    for k in [
        "step1_mae_all", "step1_mae_density", "step1_mae_fear", "step1_mae_forage", "step1_mae_momentum",
        "roll10_mae_all", "roll10_density_corr",
        "roll50_mae_all", "roll50_mae_density", "roll50_mae_fear", "roll50_mae_momentum", "roll50_density_corr",
    ]:
        nv, rv = neural_metrics[k], rd_metrics[k]
        print(f"  {k:22s} | Neural={nv:.5f} | RD_Baseline={rv:.5f}")

    # 4. Save checkpoint & export self-contained ONNX
    torch.save({"state_dict": eco_model.state_dict(), "n_params": n_params}, OUT_DIR / "ecosystem_model.pt")
    onnx_stats = export_and_verify_eco_onnx(eco_model, OUT_DIR / "ecosystem_model.onnx")

    result = {
        "n_params_neural": n_params,
        "n_params_rd_baseline": count_parameters(rd_base),
        "train_time_s": time.time() - t0,
        "neural_metrics": neural_metrics,
        "rd_baseline_metrics": rd_metrics,
        "onnx": onnx_stats,
    }
    (OUT_DIR / "eco_eval_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved ecosystem_model.onnx ({onnx_stats['size_bytes']/1024:.1f} KiB) & eco_eval_summary.json")


if __name__ == "__main__":
    main()
