#!/usr/bin/env python3
"""
Part A: Zero-Copy GPU Self-Play RL (`BotPolicyNet-RL`) on NVIDIA GB10.

Runs 32 parallel Frostmarsh 6v6 combat+physics arenas x 12 bots (384 concurrent bots)
on GPU using REAL frozen WorldModel (Z_world [A, L, 48, H, W]) and EcosystemModel
(Z_eco [A, 1, 32, H, W]) internal spatial latents zero-copy!

Stage 1: Fast vectorized PPO self-play on 32 parallel 128x128x2 arenas (384 bots)
Stage 2: High-resolution 256x256x4 polish on 8 parallel arenas (96 bots) for exact
         4-layer / 256x256 alignment with production gpu_infer_daemon.py!

Warm-starts BotPolicyNet from bot_policy.pt, trains with a separate RL-only critic
head + gentle KL anchor to BC against a self-play mix (current policy, lagged RL
snapshot, BC snapshot, and Scripted Teacher), and saves a 100% pure BotPolicyNet
state_dict() to /home/gooning/.t3/scratch/frostmarsh-ml/bot_policy_rl.pt.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bot_policy import (
    ACTION_OUT_DIM,
    BOT_FEAT_DIM,
    NUM_TOKENS,
    SELF_DIM,
    TOKEN_DIM,
    BotPolicyNet,
    count_parameters,
)
from model import EcosystemModel, WorldModel

ROOT = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
WORLD_PT = ROOT / "world_model.pt"
ECO_PT = ROOT / "ecosystem_model.pt"
BC_PT = ROOT / "bot_policy.pt"
RL_PT = ROOT / "bot_policy_rl.pt"

# Exact FIELD_CONTRACT species indices
SP_WATER, SP_ICE, SP_STEAM, SP_SOIL, SP_ROCK, SP_WOOD, SP_ASH, SP_SMOKE, SP_OIL, SP_PLANT = range(10)
CH_TEMP = 10
CH_LOGMASS = 11
CH_FLOW = 13
CH_ALT = 14
CH_OBST = 15

CANON_SIZE = 256.0
NUM_BOTS = 12


def load_ckpt_state_dict(path: Path, device: torch.device) -> dict:
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


def load_frozen_world_models(device: torch.device) -> tuple[WorldModel, EcosystemModel]:
    wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    wm.load_state_dict(load_ckpt_state_dict(WORLD_PT, device))
    for p in wm.parameters():
        p.requires_grad_(False)

    em = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).to(device).eval()
    em.load_state_dict(load_ckpt_state_dict(ECO_PT, device))
    for p in em.parameters():
        p.requires_grad_(False)
    return wm, em


@torch.inference_mode()
def extract_world_latent_and_delta(wm: WorldModel, w_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    w_state: [A, L, 18, H, W] float32
    Returns:
      z_world: [A, L, 48, H, W] float16
      deltas:  [A, L, 17, H, W] float16
    """
    a, l, c, h, w = w_state.shape
    x = w_state.reshape(a * l, c, h, w)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        st = wm.stem(x)
        h_feat = st
        for blk in wm.blocks:
            h_feat = blk(h_feat)
        z = F.gelu(h_feat)
        raw = wm.head(z) * wm.out_scale
        out = torch.cat([raw[:, :10], F.softplus(raw[:, 10:])], dim=1)
    return z.half().reshape(a, l, 48, h, w), out.half().reshape(a, l, 17, h, w)


@torch.inference_mode()
def extract_eco_latent(em: EcosystemModel, e_state: torch.Tensor) -> torch.Tensor:
    """
    e_state: [A, 1, 8, H, W] float32
    Returns:
      z_eco: [A, 1, 32, H, W] float16
    """
    a, _, c, h, w = e_state.shape
    x = e_state.reshape(a, c, h, w)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h_feat = em.stem(x)
        for blk in em.blocks:
            h_feat = blk(h_feat)
        z = F.gelu(h_feat)
    return z.half().reshape(a, 1, 32, h, w)


def batched_policy_forward(
    policy: BotPolicyNet,
    z_world: torch.Tensor,   # [A, L, 48, H, W]
    z_eco: torch.Tensor,     # [A, 1, 32, H, W]
    bot_feats: torch.Tensor, # [A, N, 168]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact multi-arena vectorized forward pass through unmodified BotPolicyNet!
    Mathematically identical to calling policy.forward_logits(z_world[a], z_eco[a], bot_feats[a])
    for each arena a, executed in a single batched GPU pass over all A*N bots.
    Returns: (vec [A,N,4], bin_logits [A,N,6], elem_logits [A,N,3], trunk_h [A,N,128])
    """
    a, n, _ = bot_feats.shape
    p = policy.patch_size
    bf_flat = bot_feats.reshape(a * n, BOT_FEAT_DIM)
    self_feat = bf_flat[:, :SELF_DIM]
    tok_feat = bf_flat[:, SELF_DIM:].reshape(a * n, NUM_TOKENS, TOKEN_DIM)

    if policy.blind_world:
        patch = torch.zeros(
            (a * n, policy.z_world_ch + policy.z_eco_ch, p, p),
            device=bot_feats.device,
            dtype=torch.float32,
        )
    else:
        _, num_l, _, h, w = z_world.shape
        sf_a = bot_feats[:, :, :SELF_DIM]  # [A, N, 24]
        cx = sf_a[:, :, 0:1, None] * 2.0 - 1.0  # [A, N, 1, 1]
        cy = sf_a[:, :, 1:2, None] * 2.0 - 1.0  # [A, N, 1, 1]
        rx = (policy.patch_radius_cells / max(w - 1, 1)) * 2.0
        ry = (policy.patch_radius_cells / max(h - 1, 1)) * 2.0

        # base_gx/base_gy are [1, P, P] -> broadcast to [A, N, P, P]
        bgx = policy.base_gx.unsqueeze(0)
        bgy = policy.base_gy.unsqueeze(0)
        grid_x = cx + bgx * rx
        grid_y = cy + bgy * ry

        # Single 4D grid_sample for Z_eco across all A arenas & N bots
        grid_2d = torch.stack([grid_x, grid_y], dim=-1).reshape(a, n * p, p, 2)
        eco_in = z_eco[:, 0].float()  # [A, 32, H, W]
        eco_patch = F.grid_sample(
            eco_in, grid_2d.float(), mode="bilinear", padding_mode="border", align_corners=True
        ).reshape(a, 32, n, p, p).permute(0, 2, 1, 3, 4).reshape(a * n, 32, p, p)

        # Single 5D grid_sample for Z_world across all A arenas & N bots
        z_vol = z_world.permute(0, 2, 1, 3, 4).float()  # [A, 48, L, H, W]
        if num_l > 1:
            cz = (sf_a[:, :, 2:3, None] * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            cz = torch.zeros_like(cx)
        grid_3d = torch.stack(
            [grid_x, grid_y, cz.expand(-1, -1, p, p)], dim=-1
        )  # [A, N, P, P, 3]
        w_patch = F.grid_sample(
            z_vol, grid_3d.float(), mode="bilinear", padding_mode="border", align_corners=True
        ).permute(0, 2, 1, 3, 4).reshape(a * n, 48, p, p)

        patch = torch.cat([w_patch, eco_patch], dim=1)

    sp = policy.spatial_enc(policy.latent_adapter(patch))
    h_world = policy.spatial_proj(torch.cat([sp.mean(dim=(2, 3)), sp.amax(dim=(2, 3))], dim=-1))

    q = policy.self_enc(self_feat).unsqueeze(1)
    kv = policy.tok_enc(tok_feat)
    attn_out, _ = policy.cross_attn(q, kv, kv, need_weights=False)
    h_tac = policy.attn_norm((q + attn_out).squeeze(1))

    h_trunk = policy.trunk(torch.cat([h_world, h_tac], dim=-1))
    vec = torch.tanh(policy.head_vec(h_trunk)).reshape(a, n, 4)
    bin_logits = policy.head_bin(h_trunk).reshape(a, n, 6)
    elem_logits = policy.head_elem(h_trunk).reshape(a, n, 3)
    return vec, bin_logits, elem_logits, h_trunk.reshape(a, n, 128)


class RLBotActorCritic(nn.Module):
    """Wraps pure BotPolicyNet with RL-only critic head & continuous exploration log_std."""

    def __init__(self, policy: BotPolicyNet):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Learnable continuous exploration std for (move_x, move_y, aim_x, aim_y)
        self.log_std = nn.Parameter(torch.tensor([-1.55, -1.55, -1.95, -1.95], dtype=torch.float32))

    def forward_batched(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vec, bin_logits, elem_logits, h_trunk = batched_policy_forward(self.policy, z_world, z_eco, bot_feats)
        values = self.value_head(h_trunk).squeeze(-1)  # [A, N]
        return vec, bin_logits, elem_logits, values

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-2.6, -0.7))

    def sample_actions(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (act_vec [A,N,4], act_bin [A,N,6], act_elem [A,N], log_prob [A,N], values [A,N])."""
        vec, bin_logits, elem_logits, values = self.forward_batched(z_world, z_eco, bot_feats)
        std = self.get_std().view(1, 1, 4)

        if deterministic:
            a_vec = vec
            a_bin = (torch.sigmoid(bin_logits) > 0.5).float()
            a_elem = elem_logits.argmax(dim=-1)
            lp = torch.zeros_like(values)
            return a_vec, a_bin, a_elem, lp, values

        dist_vec = torch.distributions.Normal(vec, std)
        a_vec = dist_vec.sample().clamp(-1.0, 1.0)
        lp_vec = dist_vec.log_prob(a_vec).sum(dim=-1)

        dist_bin = torch.distributions.Bernoulli(logits=bin_logits)
        a_bin = dist_bin.sample()
        lp_bin = dist_bin.log_prob(a_bin).sum(dim=-1)

        dist_elem = torch.distributions.Categorical(logits=elem_logits)
        a_elem = dist_elem.sample()
        lp_elem = dist_elem.log_prob(a_elem)

        return a_vec, a_bin, a_elem, (lp_vec + lp_bin + lp_elem), values

    def evaluate_actions(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        a_vec: torch.Tensor,
        a_bin: torch.Tensor,
        a_elem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vec, bin_logits, elem_logits, values = self.forward_batched(z_world, z_eco, bot_feats)
        std = self.get_std().view(1, 1, 4)

        dist_vec = torch.distributions.Normal(vec, std)
        lp_vec = dist_vec.log_prob(a_vec).sum(dim=-1)
        ent_vec = dist_vec.entropy().sum(dim=-1)

        dist_bin = torch.distributions.Bernoulli(logits=bin_logits)
        lp_bin = dist_bin.log_prob(a_bin).sum(dim=-1)
        ent_bin = dist_bin.entropy().sum(dim=-1)

        dist_elem = torch.distributions.Categorical(logits=elem_logits)
        lp_elem = dist_elem.log_prob(a_elem)
        ent_elem = dist_elem.entropy()

        return (
            lp_vec + lp_bin + lp_elem,
            ent_vec + ent_bin + ent_elem,
            values,
            vec,
            bin_logits,
            elem_logits,
        )


def compute_bc_kl_loss(
    vec: torch.Tensor,
    bin_logits: torch.Tensor,
    elem_logits: torch.Tensor,
    bc_vec: torch.Tensor,
    bc_bin_logits: torch.Tensor,
    bc_elem_logits: torch.Tensor,
) -> torch.Tensor:
    """Gentle KL anchor to BC in strict float32 (no bf16 log(0) NaNs!)."""
    vec, bc_vec = vec.float(), bc_vec.float()
    bin_logits, bc_bin_logits = bin_logits.float(), bc_bin_logits.float()
    elem_logits, bc_elem_logits = elem_logits.float(), bc_elem_logits.float()

    kl_aim = ((vec[..., 2:] - bc_vec[..., 2:]) ** 2).sum(dim=-1)
    kl_move = ((vec[..., :2] - bc_vec[..., :2]) ** 2).sum(dim=-1)

    # Numerically exact Bernoulli KL via log_sigmoid (never produces NaN!)
    p = torch.sigmoid(bin_logits)
    kl_bin = (
        p * (F.logsigmoid(bin_logits) - F.logsigmoid(bc_bin_logits))
        + (1.0 - p) * (F.logsigmoid(-bin_logits) - F.logsigmoid(-bc_bin_logits))
    ).sum(dim=-1)

    log_pe = F.log_softmax(elem_logits, dim=-1)
    log_qe = F.log_softmax(bc_elem_logits, dim=-1)
    pe = log_pe.exp()
    kl_elem = (pe * (log_pe - log_qe)).sum(dim=-1)

    return 1.2 * kl_aim + 0.15 * kl_move + 0.08 * kl_bin + 0.06 * kl_elem


class FrostmarshGPUArena:
    """
    Pure PyTorch GPU vectorized Frostmarsh 6v6 Combat + Physics Arena.
    Supports both 128x128x2 (fast 32-arena RL rollouts) and 256x256x4 (full tournament eval).
    Uses REAL WorldModel & EcosystemModel spatial latents zero-copy in VRAM!
    """

    def __init__(
        self,
        num_arenas: int,
        world_model: WorldModel,
        eco_model: EcosystemModel,
        device: torch.device,
        grid_size: int = 128,
        num_layers: int = 2,
        max_ticks: int = 120,
        latent_refresh_every: int = 4,
    ):
        self.A = num_arenas
        self.N = NUM_BOTS
        self.H = grid_size
        self.W = grid_size
        self.L = num_layers
        self.max_ticks = max_ticks
        self.refresh_every = latent_refresh_every
        self.device = device
        self.wm = world_model
        self.em = eco_model

        self.surf_l = 0 if num_layers == 2 else 1
        self.high_l = 1 if num_layers == 2 else 2

        # Static team & class tables: 0..5 = West (0), 6..11 = East (1)
        teams = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long, device=device)
        classes = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long, device=device)
        self.team = teams.unsqueeze(0).expand(self.A, -1).contiguous()       # [A, 12]
        self.cls = classes.unsqueeze(0).expand(self.A, -1).contiguous()      # [A, 12]
        self.arena_idx = torch.arange(self.A, device=device).unsqueeze(1).expand(self.A, self.N)
        self.bot_idx = torch.arange(self.N, device=device).unsqueeze(0).expand(self.A, self.N)
        self.enemy_pair_mask = self.team[:, :, None] != self.team[:, None, :]  # [A, 12, 12]

        # Normalized spatial grid for fast world generation
        yy = torch.arange(self.H, device=device, dtype=torch.float32) / self.H
        xx = torch.arange(self.W, device=device, dtype=torch.float32) / self.W
        self.ny, self.nx = torch.meshgrid(yy, xx, indexing="ij")  # [H, W]

        # Fixed 3x3 and 5x5 splash stencils
        r3 = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)
        dy3, dx3 = torch.meshgrid(r3, r3, indexing="ij")
        self.st3_dy = dy3.reshape(-1)
        self.st3_dx = dx3.reshape(-1)

        r5 = torch.tensor([-2, -1, 0, 1, 2], device=device, dtype=torch.long)
        dy5, dx5 = torch.meshgrid(r5, r5, indexing="ij")
        d5 = (dy5.float() ** 2 + dx5.float() ** 2).sqrt()
        self.st5_dy = dy5.reshape(-1)
        self.st5_dx = dx5.reshape(-1)
        self.st5_w = (1.0 - (d5.reshape(-1) / 3.2)).clamp(min=0.25)

        # Shrine coordinates in canonical 256-scale (matching train_bot_policy.py)
        self.shrine_x = torch.tensor([128.0, 128.0, 128.0], device=device).unsqueeze(0).expand(self.A, -1).clone()
        self.shrine_y = torch.tensor([0.18 * 256.0, 0.50 * 256.0, 0.82 * 256.0], device=device).unsqueeze(0).expand(self.A, -1).clone()

        # Allocate dynamic state buffers
        self.w_state = torch.zeros((self.A, self.L, 18, self.H, self.W), dtype=torch.float32, device=device)
        self.e_state = torch.zeros((self.A, 1, 8, self.H, self.W), dtype=torch.float32, device=device)
        self.z_world = torch.zeros((self.A, self.L, 48, self.H, self.W), dtype=torch.float16, device=device)
        self.z_eco = torch.zeros((self.A, 1, 32, self.H, self.W), dtype=torch.float16, device=device)

        self.reset(seed=0)

    def _generate_worlds(self, gen: torch.Generator) -> None:
        """Vectorized GPU generation of A diverse Frostmarsh 3D worlds matching FIELD_CONTRACT."""
        A, L, H, W = self.A, self.L, self.H, self.W
        dev = self.device
        self.w_state.zero_()
        self.e_state.zero_()

        nx = self.nx.unsqueeze(0)  # [1, H, W]
        ny = self.ny.unsqueeze(0)  # [1, H, W]

        # Symmetric river with mild random meander so West vs East matches are fair!
        phase = torch.rand((A, 1, 1), generator=gen, device=dev) * 6.28318
        rcx = 0.50 + 0.045 * torch.sin(ny * 6.28318 * 1.3 + phase)
        rhw = torch.empty((A, 1, 1), device=dev).uniform_(0.06, 0.085, generator=gen)
        in_river = (nx - rcx).abs() < rhw  # [A, H, W]

        sl = self.surf_l
        hl = self.high_l

        self.w_state[:, sl, SP_SOIL] = torch.where(~in_river, 0.92, 0.0)
        self.w_state[:, sl, SP_WATER] = torch.where(in_river, 1.10, 0.0)
        self.w_state[:, sl, CH_FLOW] = torch.where(in_river, 0.13, 0.0)
        self.w_state[:, sl, CH_ALT] = torch.where(in_river, 0.15, 0.45)

        # Two ice crossings across the river
        for iy_lo, iy_hi in ((0.16, 0.34), (0.66, 0.84)):
            ice_y = torch.empty((A, 1, 1), device=dev).uniform_(iy_lo, iy_hi, generator=gen)
            ice_m = in_river & ((ny - ice_y).abs() < 0.042)
            self.w_state[:, sl, SP_ICE] = torch.where(ice_m, 0.95, self.w_state[:, sl, SP_ICE])
            self.w_state[:, sl, SP_WATER] = torch.where(ice_m, 0.15, self.w_state[:, sl, SP_WATER])
            self.w_state[:, sl, CH_TEMP] = torch.where(ice_m, -0.25, self.w_state[:, sl, CH_TEMP])

        # Symmetric pairs of oil basins (4 pairs = 8 pools per map, including near shrines!)
        for k in range(4):
            ox = torch.empty((A, 1, 1), device=dev).uniform_(0.18, 0.48, generator=gen)
            oy = torch.empty((A, 1, 1), device=dev).uniform_(0.15, 0.85, generator=gen)
            orad = torch.empty((A, 1, 1), device=dev).uniform_(0.042, 0.072, generator=gen)
            for cx in (ox, 1.0 - ox):
                om = ((nx - cx) ** 2 + (ny - oy) ** 2) < (orad ** 2)
                self.w_state[:, sl, SP_OIL] = torch.where(om, 0.85, self.w_state[:, sl, SP_OIL])
                self.w_state[:, sl, SP_WATER] = torch.where(om, 0.18, self.w_state[:, sl, SP_WATER])

        # Wood forests & plants
        s_off = torch.rand((A, 1, 1), generator=gen, device=dev) * 10.0
        wood_noise = torch.sin(nx * 43.0 + s_off) * torch.cos(ny * 48.0 - s_off)
        self.w_state[:, sl, SP_WOOD] = torch.where((~in_river) & (wood_noise > 0.35), 0.90, 0.0)
        self.w_state[:, sl, SP_PLANT] = torch.where((~in_river) & (wood_noise > -0.10), 0.45, 0.0)

        # Symmetric active wildfires (4 pairs = 8 wildfires, positioned along lanes/shrines so fire evasion matters!)
        for k in range(4):
            fx = torch.empty((A, 1, 1), device=dev).uniform_(0.22, 0.48, generator=gen)
            fy = torch.empty((A, 1, 1), device=dev).uniform_(0.14, 0.86, generator=gen)
            frad = torch.empty((A, 1, 1), device=dev).uniform_(0.036, 0.060, generator=gen)
            ftemp = torch.empty((A, 1, 1), device=dev).uniform_(2.6, 4.4, generator=gen)
            for cx in (fx, 1.0 - fx):
                fm = ((nx - cx) ** 2 + (ny - fy) ** 2) < (frad ** 2)
                self.w_state[:, sl, CH_TEMP] = torch.where(fm, ftemp, self.w_state[:, sl, CH_TEMP])
                self.w_state[:, sl, SP_SMOKE] = torch.where(fm, 0.35, self.w_state[:, sl, SP_SMOKE])
                self.w_state[:, sl, SP_WOOD] = torch.where(fm, self.w_state[:, sl, SP_WOOD].clamp(min=0.40), self.w_state[:, sl, SP_WOOD])

        # Highland layer (plateaus + bridges)
        nw_p = (nx > 0.06) & (nx < 0.36) & (ny > 0.06) & (ny < 0.33)
        ne_p = (nx > 0.64) & (nx < 0.94) & (ny > 0.06) & (ny < 0.33)
        arch = ((ny - 0.29).abs() < 0.025) & (nx > 0.34) & (nx < 0.66)
        tbridge = ((ny - 0.67).abs() < 0.024) & (nx > 0.35) & (nx < 0.65)
        self.w_state[:, hl, SP_SOIL] = torch.where(nw_p | ne_p | arch, 0.95, 0.0)
        self.w_state[:, hl, SP_WOOD] = torch.where(tbridge, 0.55, 0.0)
        self.w_state[:, hl, CH_ALT] = 0.65

        if L == 4:
            self.w_state[:, 0, SP_WATER] = torch.where(in_river, 1.15, 0.0)
            self.w_state[:, 0, SP_ROCK] = torch.where(~in_river, 1.20, 0.0)
            self.w_state[:, 3, SP_STEAM] = torch.where(self.w_state[:, sl, CH_TEMP] > 2.0, 0.25, 0.0)
            self.w_state[:, 3, CH_ALT] = 0.95

        # Log total mass channel 11
        self.w_state[:, :, CH_LOGMASS] = torch.log1p(self.w_state[:, :, :10].sum(dim=2))

        # Ecosystem state [A, 1, 8, H, W]
        crit = torch.zeros((A, H, W), device=dev)
        for _ in range(2):
            cx = torch.empty((A, 1, 1), device=dev).uniform_(0.18, 0.45, generator=gen)
            cy = torch.empty((A, 1, 1), device=dev).uniform_(0.18, 0.82, generator=gen)
            for c_sym in (cx, 1.0 - cx):
                d2 = (nx - c_sym) ** 2 + (ny - cy) ** 2
                crit = torch.maximum(crit, torch.exp(-d2 * 196.0))
        self.e_state[:, 0, 0] = crit
        self.e_state[:, 0, 2] = 0.8
        self.e_state[:, 0, 4] = ((self.w_state[:, sl, SP_ROCK] < 0.5) & (self.w_state[:, sl, SP_WATER] < 0.45)).float()
        self.e_state[:, 0, 5] = (self.w_state[:, sl, CH_TEMP] > 1.5).float()
        self.e_state[:, 0, 6] = self.w_state[:, sl, SP_PLANT]
        self.e_state[:, 0, 7] = self.w_state[:, sl, SP_ICE]

    def refresh_latents(self, refresh_eco: bool = False) -> None:
        self.z_world, _ = extract_world_latent_and_delta(self.wm, self.w_state)
        if refresh_eco:
            self.e_state[:, 0, 5] = (self.w_state[:, self.surf_l, CH_TEMP] > 1.5).float()
            self.e_state[:, 0, 7] = self.w_state[:, self.surf_l, SP_ICE]
            self.z_eco = extract_eco_latent(self.em, self.e_state)

    def reset(self, seed: int = 0) -> torch.Tensor:
        dev = self.device
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        self.gen = gen
        self.tick = 0

        self._generate_worlds(gen)
        self.refresh_latents(refresh_eco=True)

        # Spawn West (0..5) and East (6..11) symmetrically across 3 lanes!
        lane_y = torch.tensor([46.0, 90.0, 128.0, 166.0, 210.0, 128.0], device=dev)
        jitter_x = torch.empty((self.A, 6), device=dev).uniform_(-10.0, 10.0, generator=gen)
        jitter_y = torch.empty((self.A, 6), device=dev).uniform_(-10.0, 10.0, generator=gen)

        west_x = (48.0 + jitter_x).clamp(18.0, 90.0)
        east_x = (256.0 - west_x)
        west_y = (lane_y.unsqueeze(0) + jitter_y).clamp(20.0, 236.0)
        east_y = west_y.clone()

        self.pos_x = torch.cat([west_x, east_x], dim=1)  # [A, 12]
        self.pos_y = torch.cat([west_y, east_y], dim=1)  # [A, 12]
        self.vel_x = torch.zeros((self.A, self.N), device=dev)
        self.vel_y = torch.zeros((self.A, self.N), device=dev)
        self.bot_layer = torch.full((self.A, self.N), self.surf_l, dtype=torch.long, device=dev)

        self.hp = torch.ones((self.A, self.N), device=dev)
        self.mana = torch.ones((self.A, self.N), device=dev)
        self.stam = torch.ones((self.A, self.N), device=dev)
        self.alive = torch.ones((self.A, self.N), dtype=torch.bool, device=dev)
        self.respawn_cd = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)

        self.dash_cd = torch.zeros((self.A, self.N), device=dev)
        self.e_cd = torch.zeros((self.A, self.N), device=dev)
        self.r_cd = torch.zeros((self.A, self.N), device=dev)

        self.oiled_t = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.wet_t = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.chilled_t = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.frozen_t = torch.zeros((self.A, self.N), dtype=torch.int32, device=dev)
        self.last_elem = torch.zeros((self.A, self.N), dtype=torch.long, device=dev)
        self.last_attacker = torch.full((self.A, self.N), -1, dtype=torch.long, device=dev)

        self.carrying = torch.zeros((self.A, self.N), dtype=torch.bool, device=dev)
        self.relic_x = torch.full((self.A,), 128.0, device=dev)
        self.relic_y = (128.0 + torch.empty((self.A,), device=dev).uniform_(-24.0, 24.0, generator=gen))
        self.relic_carrier = torch.full((self.A,), -1, dtype=torch.long, device=dev)

        self.shrine_prog = torch.zeros((self.A, 3), device=dev)
        self.shrine_owner = torch.full((self.A, 3), -1, dtype=torch.long, device=dev)

        # Match statistics [A, 2]
        self.team_score = torch.zeros((self.A, 2), device=dev)
        self.team_kills = torch.zeros((self.A, 2), device=dev)
        self.team_deaths = torch.zeros((self.A, 2), device=dev)
        self.team_shrine_caps = torch.zeros((self.A, 2), device=dev)
        self.team_combos = torch.zeros((self.A, 2), device=dev)
        self.team_burn_taken = torch.zeros((self.A, 2), device=dev)
        self.team_relic_deliveries = torch.zeros((self.A, 2), device=dev)

        return self.build_bot_features()

    def _grid_xy(self, px: torch.Tensor, py: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = (px * (self.W / CANON_SIZE)).long().clamp(0, self.W - 1)
        gy = (py * (self.H / CANON_SIZE)).long().clamp(0, self.H - 1)
        return gx, gy

    def _splash_world(
        self,
        il: torch.Tensor,      # [A, 12]
        igx: torch.Tensor,     # [A, 12]
        igy: torch.Tensor,     # [A, 12]
        ch: int,
        amount: torch.Tensor,  # [A, 12] (0 where inactive)
        radius5: bool = False,
        mode: str = "add",     # "add" or "set_min" (for cooling)
    ) -> None:
        """Vectorized GPU ground splash on w_state channel ch with zero Python loops over bots."""
        if not (amount.abs() > 1e-6).any():
            return
        dy = self.st5_dy if radius5 else self.st3_dy
        dx = self.st5_dx if radius5 else self.st3_dx
        w_st = self.st5_w if radius5 else torch.ones_like(dy, dtype=torch.float32)

        # Broadcast [A, 12, S]
        sy = (igy.unsqueeze(-1) + dy.view(1, 1, -1)).clamp(0, self.H - 1)
        sx = (igx.unsqueeze(-1) + dx.view(1, 1, -1)).clamp(0, self.W - 1)
        a_id = self.arena_idx.unsqueeze(-1).expand_as(sy)
        l_id = il.unsqueeze(-1).expand_as(sy)
        flat_idx = (((a_id * self.L + l_id) * self.H + sy) * self.W + sx).reshape(-1)
        vals = (amount.unsqueeze(-1) * w_st.view(1, 1, -1)).reshape(-1)

        plane = self.w_state[:, :, ch].reshape(-1)
        if mode == "add":
            plane.scatter_add_(0, flat_idx, vals)
        else:
            # Cooling: subtract towards negative temp
            plane.scatter_add_(0, flat_idx, -vals)

    def build_bot_features(self) -> torch.Tensor:
        """Build exact [A, 12, 168] float32 feature tensor matching train_bot_policy.py."""
        A, N = self.A, self.N
        dev = self.device
        feats = torch.zeros((A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)

        # Self features [A, 12, 24]
        feats[:, :, 0] = self.pos_x / CANON_SIZE
        feats[:, :, 1] = self.pos_y / CANON_SIZE
        feats[:, :, 2] = self.bot_layer.float() / max(self.L - 1, 1)
        feats[:, :, 3] = (self.vel_x / 10.0).clamp(-0.5, 0.5)
        feats[:, :, 4] = (self.vel_y / 10.0).clamp(-0.5, 0.5)
        feats[:, :, 5] = self.hp
        feats[:, :, 6] = self.mana
        feats[:, :, 7] = self.stam
        feats[:, :, 8] = (self.cls == 0).float()
        feats[:, :, 9] = (self.cls == 1).float()
        feats[:, :, 10] = (self.cls == 2).float()
        feats[:, :, 11] = self.team.float()
        feats[:, :, 12] = self.carrying.float()
        feats[:, :, 13] = max(0.0, 1.0 - self.tick / float(self.max_ticks))
        feats[:, :, 20] = self.dash_cd
        feats[:, :, 21] = self.e_cd
        feats[:, :, 22] = self.r_cd
        feats[:, :, 23] = self.last_elem.float() / 3.0

        # Pairwise entity tokens (nearest 8 other bots) -> tokens 0..7
        dx = self.pos_x[:, None, :] - self.pos_x[:, :, None]  # [A, 12, 12]
        dy = self.pos_y[:, None, :] - self.pos_y[:, :, None]
        dz = (self.bot_layer[:, None, :] - self.bot_layer[:, :, None]).float() / 2.0
        dist = torch.hypot(dx, dy).clamp(min=1e-3)

        # Mask out self and dead bots by setting large sort distance
        sort_dist = dist.clone()
        eye = torch.eye(N, dtype=torch.bool, device=dev).unsqueeze(0)
        sort_dist = sort_dist.masked_fill(eye, 1e6)
        dead_j = (~self.alive)[:, None, :].expand(A, N, N)
        sort_dist = sort_dist.masked_fill(dead_j, 5e5)

        nn_idx = torch.argsort(sort_dist, dim=-1)[:, :, :8]  # [A, 12, 8]

        nn_dx = torch.gather(dx, 2, nn_idx)
        nn_dy = torch.gather(dy, 2, nn_idx)
        nn_dz = torch.gather(dz, 2, nn_idx)
        nn_d = torch.gather(dist, 2, nn_idx)
        nn_en = torch.gather(self.enemy_pair_mask.float(), 2, nn_idx)

        hp_exp = self.hp[:, None, :].expand(A, N, N)
        car_exp = self.carrying.float()[:, None, :].expand(A, N, N)
        cls_exp = (self.cls.float() / 2.0)[:, None, :].expand(A, N, N)
        alive_exp = self.alive.float()[:, None, :].expand(A, N, N)

        nn_hp = torch.gather(hp_exp, 2, nn_idx)
        nn_car = torch.gather(car_exp, 2, nn_idx)
        nn_cls = torch.gather(cls_exp, 2, nn_idx)
        nn_valid = torch.gather(alive_exp, 2, nn_idx)

        toks = feats[:, :, SELF_DIM:].view(A, N, NUM_TOKENS, TOKEN_DIM)
        toks[:, :, :8, 0] = (nn_dx / 80.0).clamp(-1.0, 1.0) * nn_valid
        toks[:, :, :8, 1] = (nn_dy / 80.0).clamp(-1.0, 1.0) * nn_valid
        toks[:, :, :8, 2] = nn_dz * nn_valid
        toks[:, :, :8, 3] = nn_en * nn_valid
        toks[:, :, :8, 4] = nn_hp * nn_valid
        toks[:, :, :8, 5] = nn_car * nn_valid
        toks[:, :, :8, 6] = nn_cls * nn_valid
        toks[:, :, :8, 7] = (nn_d / 100.0).clamp(max=1.0) * nn_valid
        toks[:, :, :8, 8] = (nn_dx / nn_d) * nn_valid
        toks[:, :, :8, 9] = (nn_dy / nn_d) * nn_valid
        toks[:, :, :8, 11] = nn_valid

        # 3 Shrine tokens (8..10)
        sdx = self.shrine_x[:, None, :] - self.pos_x[:, :, None]  # [A, 12, 3]
        sdy = self.shrine_y[:, None, :] - self.pos_y[:, :, None]
        sprog = self.shrine_prog[:, None, :].expand(A, N, 3)
        is_west = (self.team == 0).unsqueeze(-1)
        s_need = torch.where(is_west, sprog > -0.8, sprog < 0.8).float()

        toks[:, :, 8:11, 0] = (sdx / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 8:11, 1] = (sdy / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 8:11, 3] = s_need
        toks[:, :, 8:11, 4] = sprog
        toks[:, :, 8:11, 11] = 1.0

        # Golden Relic token (11)
        rdx = self.relic_x[:, None] - self.pos_x
        rdy = self.relic_y[:, None] - self.pos_y
        toks[:, :, 11, 0] = (rdx / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 11, 1] = (rdy / 100.0).clamp(-1.0, 1.0)
        toks[:, :, 11, 3] = self.carrying.float()
        toks[:, :, 11, 11] = 1.0

        return feats

    def step(
        self,
        act_vec: torch.Tensor,   # [A, 12, 4] (move_x, move_y, aim_x, aim_y)
        act_bin: torch.Tensor,   # [A, 12, 6] (dash, melee, cast_bolt, carve_ice, ability_e, ult_r)
        act_elem: torch.Tensor,  # [A, 12] int64 (0=Fire, 1=Frost, 2=Oil)
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Advance all A arenas by 1 tick. Returns (next_feats, rewards [A, 12], done)."""
        A, N = self.A, self.N
        dev = self.device
        self.tick += 1

        prev_score = self.team_score.clone()
        rewards = torch.zeros((A, N), dtype=torch.float32, device=dev)

        # 1. Handle Respawns
        dead = ~self.alive
        self.respawn_cd = torch.where(dead, (self.respawn_cd - 1).clamp(min=0), self.respawn_cd)
        do_respawn = dead & (self.respawn_cd == 0)
        if do_respawn.any():
            base_x = torch.where(self.team == 0, 38.0, 218.0)
            rjx = torch.empty((A, N), device=dev).uniform_(-12.0, 12.0, generator=self.gen)
            rjy = torch.empty((A, N), device=dev).uniform_(36.0, 220.0, generator=self.gen)
            self.pos_x = torch.where(do_respawn, (base_x + rjx).clamp(16.0, 240.0), self.pos_x)
            self.pos_y = torch.where(do_respawn, rjy, self.pos_y)
            self.bot_layer = torch.where(do_respawn, torch.full_like(self.bot_layer, self.surf_l), self.bot_layer)
            self.hp = torch.where(do_respawn, torch.ones_like(self.hp), self.hp)
            self.mana = torch.where(do_respawn, torch.ones_like(self.mana), self.mana)
            self.stam = torch.where(do_respawn, torch.ones_like(self.stam), self.stam)
            self.alive = self.alive | do_respawn
            self.oiled_t = torch.where(do_respawn, torch.zeros_like(self.oiled_t), self.oiled_t)
            self.wet_t = torch.where(do_respawn, torch.zeros_like(self.wet_t), self.wet_t)
            self.chilled_t = torch.where(do_respawn, torch.zeros_like(self.chilled_t), self.chilled_t)
            self.frozen_t = torch.where(do_respawn, torch.zeros_like(self.frozen_t), self.frozen_t)
            self.last_attacker = torch.where(do_respawn, torch.full_like(self.last_attacker, -1), self.last_attacker)

        alive_f = self.alive.float()

        # 2. Regen & Cooldown Decay
        self.mana = (self.mana + 0.075 * alive_f).clamp(0.0, 1.0)
        self.stam = (self.stam + 0.08 * alive_f).clamp(0.0, 1.0)
        self.dash_cd = (self.dash_cd - 0.16).clamp(min=0.0)
        self.e_cd = (self.e_cd - 0.09).clamp(min=0.0)
        self.r_cd = (self.r_cd - 0.045).clamp(min=0.0)

        self.oiled_t = (self.oiled_t - 1).clamp(min=0)
        self.wet_t = (self.wet_t - 1).clamp(min=0)
        self.chilled_t = (self.chilled_t - 1).clamp(min=0)
        self.frozen_t = (self.frozen_t - 1).clamp(min=0)

        # 3. Movement & Terrain Physics
        mvx = act_vec[:, :, 0]
        mvy = act_vec[:, :, 1]
        mnorm = torch.hypot(mvx, mvy).clamp(min=1.0)
        mvx = mvx / mnorm
        mvy = mvy / mnorm

        gx, gy = self._grid_xy(self.pos_x, self.pos_y)
        cur_water = self.w_state[self.arena_idx, self.bot_layer, SP_WATER, gy, gx]
        cur_ice = self.w_state[self.arena_idx, self.bot_layer, SP_ICE, gy, gx]
        cur_oil = self.w_state[self.arena_idx, self.bot_layer, SP_OIL, gy, gx]

        in_deep_water = (cur_water > 0.45) & (cur_ice < 0.35)
        on_ice = cur_ice > 0.35
        on_oil = cur_oil > 0.25

        self.wet_t = torch.where(in_deep_water & self.alive, torch.clamp(self.wet_t, min=5), self.wet_t)
        self.oiled_t = torch.where(on_oil & self.alive, torch.clamp(self.oiled_t, min=6), self.oiled_t)

        # Speed modifiers
        base_spd = torch.where(self.cls == 2, 4.7, torch.where(self.cls == 0, 4.0, 4.3))
        spd = base_spd
        spd = torch.where(in_deep_water, spd * 0.55, spd)
        spd = torch.where(on_ice, spd * 1.08, spd)
        spd = torch.where(self.frozen_t > 0, spd * 0.30, spd)
        spd = torch.where(self.chilled_t > 0, spd * 0.78, spd)
        spd = torch.where(self.carrying, spd * 0.90, spd)

        # Dash
        do_dash = (act_bin[:, :, 0] > 0.5) & self.alive & (self.dash_cd <= 0.0) & (self.stam >= 0.25)
        spd = torch.where(do_dash, spd + 3.2, spd)
        self.dash_cd = torch.where(do_dash, torch.ones_like(self.dash_cd), self.dash_cd)
        self.stam = torch.where(do_dash, (self.stam - 0.28).clamp(min=0.0), self.stam)

        # Carve ice (minor defensive/speed utility when on ice)
        do_carve = (act_bin[:, :, 3] > 0.5) & self.alive & on_ice
        self.hp = torch.where(do_carve, (self.hp + 0.025).clamp(max=1.0), self.hp)

        prev_x, prev_y = self.pos_x.clone(), self.pos_y.clone()
        new_x = (self.pos_x + mvx * spd * alive_f).clamp(10.0, 246.0)
        new_y = (self.pos_y + mvy * spd * alive_f).clamp(10.0, 246.0)
        self.vel_x = (new_x - prev_x)
        self.vel_y = (new_y - prev_y)
        self.pos_x = new_x
        self.pos_y = new_y

        # Wildlife Sanctuary heal if standing near dense critters at low HP (uses Z_eco!)
        gx, gy = self._grid_xy(self.pos_x, self.pos_y)
        cur_crit = self.e_state[self.arena_idx, 0, 0, gy, gx]
        sanc_heal = (cur_crit > 0.30) & (self.hp < 0.70) & self.alive
        self.hp = torch.where(sanc_heal, (self.hp + 0.03).clamp(max=1.0), self.hp)
        rewards += 0.06 * sanc_heal.float()

        # 4. Combat, Elemental Casts, Ground Modification & Physics Combos!
        aim_x = act_vec[:, :, 2]
        aim_y = act_vec[:, :, 3]
        anorm = torch.hypot(aim_x, aim_y).clamp(min=1e-4)
        ux = aim_x / anorm
        uy = aim_y / anorm

        dx_ij = self.pos_x[:, None, :] - self.pos_x[:, :, None]  # [A, 12, 12]
        dy_ij = self.pos_y[:, None, :] - self.pos_y[:, :, None]
        dist_ij = torch.hypot(dx_ij, dy_ij).clamp(min=1e-3)
        cos_ij = (dx_ij * ux[:, :, None] + dy_ij * uy[:, :, None]) / dist_ij
        valid_en = self.enemy_pair_mask & self.alive[:, :, None] & self.alive[:, None, :]

        # Victim ground / status flags for combo resolution [A, 12]
        v_oil = (self.w_state[self.arena_idx, self.bot_layer, SP_OIL, gy, gx] > 0.20) | (self.oiled_t > 0)
        v_wet_chill = (self.w_state[self.arena_idx, self.bot_layer, SP_WATER, gy, gx] > 0.35) | (self.wet_t > 0) | (self.chilled_t > 0)
        v_frozen_ice = (self.frozen_t > 0) | (self.w_state[self.arena_idx, self.bot_layer, SP_ICE, gy, gx] > 0.35)

        dmg_matrix = torch.zeros((A, N, N), device=dev)  # [A, attacker i, victim j]
        combos_by_i = torch.zeros((A, N), device=dev)
        primed_by_i = torch.zeros((A, N), device=dev)

        # --- 4a. Melee ---
        do_melee = (act_bin[:, :, 1] > 0.5) & self.alive
        melee_cand = valid_en & (dist_ij < 8.5) & ((cos_ij > 0.25) | (dist_ij < 5.0))
        melee_d = dist_ij.masked_fill(~melee_cand, 1e6)
        m_min_d, m_tgt = melee_d.min(dim=-1)  # [A, 12]
        m_hit = do_melee & (m_min_d < 8.5)
        m_dmg = torch.where(self.cls == 0, 0.18, 0.14) * m_hit.float()
        dmg_matrix.scatter_add_(2, m_tgt.unsqueeze(-1), m_dmg.unsqueeze(-1))

        # --- 4b. Elemental Cast Bolt ---
        do_bolt = (act_bin[:, :, 2] > 0.5) & self.alive & (self.mana >= 0.14)
        self.mana = torch.where(do_bolt, (self.mana - 0.14).clamp(min=0.0), self.mana)
        self.last_elem = torch.where(do_bolt, act_elem, self.last_elem)

        bolt_range = torch.where(self.cls == 2, 54.0, 48.0).unsqueeze(-1)
        bolt_cand = valid_en & (dist_ij <= bolt_range) & (cos_ij >= 0.72)
        bolt_d = dist_ij.masked_fill(~bolt_cand, 1e6)
        b_min_d, b_tgt = bolt_d.min(dim=-1)
        b_hit = do_bolt & (b_min_d <= bolt_range.squeeze(-1))

        # Impact coords for ground splash
        hit_x = torch.gather(self.pos_x, 1, b_tgt)
        hit_y = torch.gather(self.pos_y, 1, b_tgt)
        imp_x = torch.where(b_hit, hit_x, (self.pos_x + ux * 24.0).clamp(8.0, 248.0))
        imp_y = torch.where(b_hit, hit_y, (self.pos_y + uy * 24.0).clamp(8.0, 248.0))
        igx, igy = self._grid_xy(imp_x, imp_y)

        is_fire = act_elem == 0
        is_frost = act_elem == 1
        is_oil = act_elem == 2

        tgt_on_oil = torch.gather(v_oil, 1, b_tgt)
        tgt_wet_chill = torch.gather(v_wet_chill, 1, b_tgt)
        tgt_frozen_ice = torch.gather(v_frozen_ice, 1, b_tgt)

        b_oil_inferno = b_hit & is_fire & tgt_on_oil
        b_shatter = b_hit & is_frost & tgt_wet_chill
        b_steam = b_hit & is_fire & (~tgt_on_oil) & tgt_frozen_ice
        b_any_combo = b_oil_inferno | b_shatter | b_steam

        b_base = torch.where(self.cls == 2, 0.125, torch.where(self.cls == 1, 0.12, 0.105))
        b_base = torch.where(is_oil, b_base * 0.65, b_base)
        b_mult = torch.where(b_oil_inferno, 2.5, torch.where(b_shatter | b_steam, 2.0, 1.0))
        b_dmg = b_base * b_mult * b_hit.float()
        dmg_matrix.scatter_add_(2, b_tgt.unsqueeze(-1), b_dmg.unsqueeze(-1))

        # Oil Inferno AoE splash damage to enemies near detonated target!
        if b_oil_inferno.any():
            det_dx = self.pos_x[:, None, :] - hit_x[:, :, None]
            det_dy = self.pos_y[:, None, :] - hit_y[:, :, None]
            det_dist = torch.hypot(det_dx, det_dy)
            aoe_hit = valid_en & b_oil_inferno[:, :, None] & (det_dist < 14.0)
            dmg_matrix += aoe_hit.float() * 0.09

        combos_by_i += b_any_combo.float()

        # Apply status priming from bolt hits
        prime_oil_hit = b_hit & is_oil & (~tgt_on_oil)
        prime_frost_hit = b_hit & is_frost & (~tgt_wet_chill)
        primed_by_i += (prime_oil_hit | prime_frost_hit).float()

        # Scatter status timers onto hit targets
        one_hot_bt = F.one_hot(b_tgt, num_classes=N).float()  # [A, 12, 12]
        got_oiled = (one_hot_bt * (b_hit & is_oil).float().unsqueeze(-1)).sum(dim=1) > 0.5
        got_chilled = (one_hot_bt * (b_hit & is_frost).float().unsqueeze(-1)).sum(dim=1) > 0.5
        got_frozen = (one_hot_bt * b_shatter.float().unsqueeze(-1)).sum(dim=1) > 0.5
        cleared_oil = (one_hot_bt * b_oil_inferno.float().unsqueeze(-1)).sum(dim=1) > 0.5
        cleared_frz = (one_hot_bt * b_steam.float().unsqueeze(-1)).sum(dim=1) > 0.5

        self.oiled_t = torch.where(got_oiled, torch.clamp(self.oiled_t, min=8), self.oiled_t)
        self.oiled_t = torch.where(cleared_oil, torch.zeros_like(self.oiled_t), self.oiled_t)
        self.chilled_t = torch.where(got_chilled, torch.clamp(self.chilled_t, min=6), self.chilled_t)
        self.frozen_t = torch.where(got_frozen, torch.clamp(self.frozen_t, min=4), self.frozen_t)
        self.frozen_t = torch.where(cleared_frz, torch.zeros_like(self.frozen_t), self.frozen_t)

        # Ground modifications from bolt casts (modifies real w_state -> refreshed into Z_world!)
        imp_ground_oil = self.w_state[self.arena_idx, self.bot_layer, SP_OIL, igy, igx]
        fire_cast_m = do_bolt & is_fire
        frost_cast_m = do_bolt & is_frost
        oil_cast_m = do_bolt & is_oil

        # Fire raises temp + smoke; bigger detonation if ground has oil!
        fire_boost = torch.where((imp_ground_oil > 0.20) | b_oil_inferno, 3.8, 2.2) * fire_cast_m.float()
        self._splash_world(self.bot_layer, igx, igy, CH_TEMP, fire_boost, radius5=True, mode="add")
        self._splash_world(self.bot_layer, igx, igy, SP_SMOKE, 0.30 * fire_cast_m.float(), radius5=False, mode="add")
        if b_steam.any():
            self._splash_world(self.bot_layer, igx, igy, SP_STEAM, 0.65 * b_steam.float(), radius5=True, mode="add")
            self._splash_world(self.bot_layer, igx, igy, SP_ICE, 0.60 * b_steam.float(), radius5=False, mode="sub")

        # Frost cools fire & freezes water into ice
        self._splash_world(self.bot_layer, igx, igy, CH_TEMP, 2.6 * frost_cast_m.float(), radius5=True, mode="sub")
        self._splash_world(self.bot_layer, igx, igy, SP_ICE, 0.70 * frost_cast_m.float(), radius5=False, mode="add")

        # Oil coats ground
        self._splash_world(self.bot_layer, igx, igy, SP_OIL, 0.85 * oil_cast_m.float(), radius5=True, mode="add")

        # --- 4c. Ability E (AoE Elemental Skill) & Ult R (Elemental Nova) ---
        for ab_idx, cost, cd_attr, max_r, aoe_r, dmg_scale in (
            (4, 0.28, "e_cd", 30.0, 13.0, 1.45),
            (5, 0.50, "r_cd", 28.0, 16.5, 2.10),
        ):
            cd_t = getattr(self, cd_attr)
            do_ab = (act_bin[:, :, ab_idx] > 0.5) & self.alive & (cd_t <= 0.0) & (self.mana >= cost)
            if not do_ab.any():
                continue
            self.mana = torch.where(do_ab, (self.mana - cost).clamp(min=0.0), self.mana)
            setattr(self, cd_attr, torch.where(do_ab, torch.ones_like(cd_t), cd_t))
            self.last_elem = torch.where(do_ab, act_elem, self.last_elem)

            # Warden E with Frost grants shield HP
            if ab_idx == 4:
                w_shield = do_ab & (self.cls == 0) & is_frost
                self.hp = torch.where(w_shield, (self.hp + 0.08).clamp(max=1.0), self.hp)

            ab_cand = valid_en & (dist_ij <= max_r) & (cos_ij >= 0.55)
            ab_d = dist_ij.masked_fill(~ab_cand, 1e6)
            ab_min_d, ab_tgt = ab_d.min(dim=-1)
            ab_locked = do_ab & (ab_min_d <= max_r)

            ax = torch.where(ab_locked, torch.gather(self.pos_x, 1, ab_tgt), (self.pos_x + ux * (max_r * 0.65)).clamp(8.0, 248.0))
            ay = torch.where(ab_locked, torch.gather(self.pos_y, 1, ab_tgt), (self.pos_y + uy * (max_r * 0.65)).clamp(8.0, 248.0))
            agx, agy = self._grid_xy(ax, ay)

            # All enemies within aoe_r of (ax, ay)
            ed_aoe = torch.hypot(self.pos_x[:, None, :] - ax[:, :, None], self.pos_y[:, None, :] - ay[:, :, None])
            in_aoe = valid_en & do_ab[:, :, None] & (ed_aoe <= aoe_r)

            if in_aoe.any():
                aoe_oil = in_aoe & is_fire[:, :, None] & v_oil[:, None, :]
                aoe_shat = in_aoe & is_frost[:, :, None] & v_wet_chill[:, None, :]
                aoe_stm = in_aoe & is_fire[:, :, None] & (~v_oil[:, None, :]) & v_frozen_ice[:, None, :]
                aoe_combo = aoe_oil | aoe_shat | aoe_stm

                mult = torch.where(aoe_oil, 2.5, torch.where(aoe_shat | aoe_stm, 2.0, 1.0))
                base_a = torch.where(is_oil[:, :, None], 0.085, 0.12) * dmg_scale
                dmg_matrix += in_aoe.float() * base_a * mult
                combos_by_i += aoe_combo.float().sum(dim=-1)

                # Apply status from AoE
                self.oiled_t = torch.where((in_aoe & is_oil[:, :, None]).any(dim=1), torch.clamp(self.oiled_t, min=8), self.oiled_t)
                self.chilled_t = torch.where((in_aoe & is_frost[:, :, None]).any(dim=1), torch.clamp(self.chilled_t, min=6), self.chilled_t)
                self.frozen_t = torch.where(aoe_shat.any(dim=1), torch.clamp(self.frozen_t, min=4), self.frozen_t)

            # Ground splash
            self._splash_world(self.bot_layer, agx, agy, CH_TEMP, 3.2 * (do_ab & is_fire).float(), radius5=True, mode="add")
            self._splash_world(self.bot_layer, agx, agy, CH_TEMP, 3.0 * (do_ab & is_frost).float(), radius5=True, mode="sub")
            self._splash_world(self.bot_layer, agx, agy, SP_ICE, 0.85 * (do_ab & is_frost).float(), radius5=True, mode="add")
            self._splash_world(self.bot_layer, agx, agy, SP_OIL, 0.90 * (do_ab & is_oil).float(), radius5=True, mode="add")

        # Clamp modified world state channels to valid physical ranges
        self.w_state[:, :, CH_TEMP].clamp_(-0.5, 5.5)
        self.w_state[:, :, SP_OIL].clamp_(0.0, 1.5)
        self.w_state[:, :, SP_ICE].clamp_(0.0, 1.5)
        self.w_state[:, :, SP_SMOKE].clamp_(0.0, 1.5)
        self.w_state[:, :, SP_STEAM].clamp_(0.0, 1.5)
        self.w_state[:, :, CH_LOGMASS] = torch.log1p(self.w_state[:, :, :10].sum(dim=2))

        # 5. Apply Combat Damage + Wildfire Burn Damage!
        # Warden takes 0.82x combat damage
        armor = torch.where(self.cls == 0, 0.82, 1.0)
        dmg_dealt_by_i = dmg_matrix.sum(dim=2)  # [A, 12]
        dmg_taken_by_j = dmg_matrix.sum(dim=1) * armor  # [A, 12]

        # Record strongest attacker for kill credit
        max_atk_dmg, best_atk = dmg_matrix.max(dim=1)  # [A, 12]
        self.last_attacker = torch.where(max_atk_dmg > 1e-4, best_atk, self.last_attacker)

        # Wildfire burn damage from standing on burning cells (temp > 1.5)
        cur_temp = self.w_state[self.arena_idx, self.bot_layer, CH_TEMP, gy, gx]
        in_fire = (cur_temp > 1.5) & self.alive
        burn_dmg = torch.where(in_fire, (0.055 * (cur_temp / 2.5)).clamp(0.045, 0.125), torch.zeros_like(cur_temp))

        self.hp = (self.hp - dmg_taken_by_j - burn_dmg).clamp(min=0.0)

        # Record team burn taken & physics combos
        w_mask = (self.team == 0).float()
        e_mask = (self.team == 1).float()
        self.team_burn_taken[:, 0] += (burn_dmg * w_mask).sum(dim=1)
        self.team_burn_taken[:, 1] += (burn_dmg * e_mask).sum(dim=1)
        self.team_combos[:, 0] += (combos_by_i * w_mask).sum(dim=1)
        self.team_combos[:, 1] += (combos_by_i * e_mask).sum(dim=1)
        # Combo score bonus (+2.5 team score per combo!)
        self.team_score[:, 0] += 2.5 * (combos_by_i * w_mask).sum(dim=1)
        self.team_score[:, 1] += 2.5 * (combos_by_i * e_mask).sum(dim=1)

        # 6. Deaths & Kill Credit
        just_died = self.alive & (self.hp <= 0.0)
        if just_died.any():
            self.alive = self.alive & (~just_died)
            self.respawn_cd = torch.where(just_died, torch.full_like(self.respawn_cd, 8), self.respawn_cd)

            # Drop relic if carrier died
            dropped = just_died & self.carrying
            if dropped.any():
                self.carrying = self.carrying & (~dropped)
                drop_arena = dropped.any(dim=1)
                self.relic_carrier = torch.where(drop_arena, torch.full_like(self.relic_carrier, -1), self.relic_carrier)

            self.team_deaths[:, 0] += (just_died.float() * w_mask).sum(dim=1)
            self.team_deaths[:, 1] += (just_died.float() * e_mask).sum(dim=1)

            # Credit kill to opposing team (+8.0 team score) and individual killer (+2.5 reward)
            west_died = (just_died.float() * w_mask).sum(dim=1)
            east_died = (just_died.float() * e_mask).sum(dim=1)
            self.team_kills[:, 1] += west_died
            self.team_kills[:, 0] += east_died
            self.team_score[:, 1] += 8.0 * west_died
            self.team_score[:, 0] += 8.0 * east_died

            # Individual killer reward
            valid_kill = just_died & (self.last_attacker >= 0)
            killer_idx = self.last_attacker.clamp(min=0)
            rewards.scatter_add_(1, killer_idx, 2.5 * valid_kill.float())
            rewards -= 2.0 * just_died.float()

        # 7. Shrine Capture & Scoring
        sd_bs = torch.hypot(
            self.pos_x[:, :, None] - self.shrine_x[:, None, :],
            self.pos_y[:, :, None] - self.shrine_y[:, None, :],
        )  # [A, 12, 3]
        in_shrine = (sd_bs < 18.0) & self.alive[:, :, None] & (~in_fire[:, :, None])
        west_in = (in_shrine & (self.team[:, :, None] == 0)).float().sum(dim=1)  # [A, 3]
        east_in = (in_shrine & (self.team[:, :, None] == 1)).float().sum(dim=1)  # [A, 3]

        self.shrine_prog = (self.shrine_prog + 0.12 * (east_in - west_in)).clamp(-1.0, 1.0)
        new_owner = torch.where(
            self.shrine_prog <= -0.8,
            torch.zeros_like(self.shrine_owner),
            torch.where(self.shrine_prog >= 0.8, torch.ones_like(self.shrine_owner), self.shrine_owner),
        )
        flipped_w = (new_owner == 0) & (self.shrine_owner != 0)
        flipped_e = (new_owner == 1) & (self.shrine_owner != 1)
        self.shrine_owner = new_owner

        self.team_shrine_caps[:, 0] += flipped_w.float().sum(dim=1)
        self.team_shrine_caps[:, 1] += flipped_e.float().sum(dim=1)
        self.team_score[:, 0] += 10.0 * flipped_w.float().sum(dim=1) + (self.shrine_owner == 0).float().sum(dim=1) * 1.0
        self.team_score[:, 1] += 10.0 * flipped_e.float().sum(dim=1) + (self.shrine_owner == 1).float().sum(dim=1) * 1.0

        # Individual shrine pressure & potential-based approach shaping
        need_shrine = torch.where(
            self.team[:, :, None] == 0,
            self.shrine_prog[:, None, :] > -0.8,
            self.shrine_prog[:, None, :] < 0.8,
        )
        prev_sd_bs = torch.hypot(
            prev_x[:, :, None] - self.shrine_x[:, None, :],
            prev_y[:, :, None] - self.shrine_y[:, None, :],
        )
        prev_s_min = prev_sd_bs.masked_fill(~need_shrine, 1e4).min(dim=-1).values
        new_s_min = sd_bs.masked_fill(~need_shrine, 1e4).min(dim=-1).values
        has_need_shrine = need_shrine.any(dim=-1) & (~self.carrying) & self.alive
        shrine_approach = torch.where(has_need_shrine, (prev_s_min - new_s_min).clamp(-6.0, 6.0), torch.zeros_like(rewards))
        shrine_contrib = (in_shrine & need_shrine).any(dim=-1).float()
        rewards += 1.20 * shrine_contrib + 0.14 * shrine_approach

        # 8. Golden Relic Pickup, Carry & Delivery
        no_carrier = self.relic_carrier < 0
        if no_carrier.any():
            rdist = torch.hypot(self.pos_x - self.relic_x[:, None], self.pos_y - self.relic_y[:, None])
            rdist = rdist.masked_fill(~self.alive, 1e6)
            r_min, r_who = rdist.min(dim=1)
            picked = no_carrier & (r_min < 12.0)
            self.relic_carrier = torch.where(picked, r_who, self.relic_carrier)
            if picked.any():
                oh = F.one_hot(r_who, num_classes=N).bool() & picked.unsqueeze(-1)
                self.carrying = self.carrying | oh
                rewards += 0.6 * oh.float()

        has_carrier = self.relic_carrier >= 0
        if has_carrier.any():
            cid = self.relic_carrier.clamp(min=0)
            cx = torch.gather(self.pos_x, 1, cid.unsqueeze(1)).squeeze(1)
            cy = torch.gather(self.pos_y, 1, cid.unsqueeze(1)).squeeze(1)
            self.relic_x = torch.where(has_carrier, cx, self.relic_x)
            self.relic_y = torch.where(has_carrier, cy, self.relic_y)

            c_team = torch.gather(self.team, 1, cid.unsqueeze(1)).squeeze(1)
            self.team_score[:, 0] += (has_carrier & (c_team == 0)).float() * 0.25
            self.team_score[:, 1] += (has_carrier & (c_team == 1)).float() * 0.25

            # Progress toward home base while carrying
            home_x = torch.where(self.team == 0, 28.0, 228.0)
            prev_hd = (prev_x - home_x).abs()
            new_hd = (self.pos_x - home_x).abs()
            rewards += torch.where(self.carrying, 0.22 * (prev_hd - new_hd), torch.zeros_like(rewards))

            deliv_w = has_carrier & (c_team == 0) & (cx < 38.0)
            deliv_e = has_carrier & (c_team == 1) & (cx > 218.0)
            deliv = deliv_w | deliv_e
            if deliv.any():
                self.team_score[:, 0] += 25.0 * deliv_w.float()
                self.team_score[:, 1] += 25.0 * deliv_e.float()
                self.team_relic_deliveries[:, 0] += deliv_w.float()
                self.team_relic_deliveries[:, 1] += deliv_e.float()
                rewards += 6.0 * (self.carrying & deliv.unsqueeze(-1)).float()
                self.carrying = self.carrying & (~deliv.unsqueeze(-1))
                self.relic_carrier = torch.where(deliv, torch.full_like(self.relic_carrier, -1), self.relic_carrier)
                self.relic_x = torch.where(deliv, torch.full_like(self.relic_x, 128.0), self.relic_x)
                self.relic_y = torch.where(deliv, torch.full_like(self.relic_y, 128.0), self.relic_y)

        # 9. Individual Combat + Combo + Fire Evasion + Team-Aligned Reward
        score_delta = self.team_score - prev_score  # [A, 2]
        my_team_delta = torch.where(self.team == 0, score_delta[:, 0:1], score_delta[:, 1:2])
        en_team_delta = torch.where(self.team == 0, score_delta[:, 1:2], score_delta[:, 0:1])

        rewards += (
            2.2 * dmg_dealt_by_i
            + 1.6 * combos_by_i
            + 0.35 * primed_by_i
            - 8.5 * burn_dmg
            - 0.60 * dmg_taken_by_j
            + 0.35 * my_team_delta
            - 0.15 * en_team_delta
            + 0.03 * (alive_f * (~in_fire).float())
        )

        # 10. Periodic Real WorldModel & EcosystemModel Latent Refresh + Gentle Thermal Decay
        if (self.tick % self.refresh_every) == 0:
            # Gentle thermal decay & fire spread across oil pools
            temp = self.w_state[:, :, CH_TEMP]
            oil = self.w_state[:, :, SP_OIL]
            hot_nb = F.max_pool2d(temp.reshape(A * self.L, 1, self.H, self.W), 3, stride=1, padding=1).reshape(A, self.L, self.H, self.W)
            ignite_oil = (hot_nb > 2.2) & (oil > 0.25)
            temp = torch.where(ignite_oil, torch.maximum(temp, torch.full_like(temp, 3.2)), temp)
            temp = torch.where(temp > 0.1, temp * 0.96, temp * 0.94)
            self.w_state[:, :, CH_TEMP] = temp
            self.w_state[:, :, SP_OIL] = torch.where(ignite_oil, (oil - 0.12).clamp(min=0.0), oil)
            self.w_state[:, :, SP_SMOKE].mul_(0.90)
            self.w_state[:, :, SP_STEAM].mul_(0.88)
            self.w_state[:, :, CH_LOGMASS] = torch.log1p(self.w_state[:, :, :10].sum(dim=2))

            self.refresh_latents(refresh_eco=(self.tick % (self.refresh_every * 4) == 0))

        done = self.tick >= self.max_ticks
        return self.build_bot_features(), rewards, done


@torch.inference_mode()
def scripted_teacher_act(arena: FrostmarshGPUArena) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch GPU vectorized implementation of the exact physics-coupled Scripted Teacher
    from train_bot_policy.py (lines 264-364).
    Returns: (act_vec [A, 12, 4], act_bin [A, 12, 6], act_elem [A, 12])
    """
    A, N, H, W = arena.A, arena.N, arena.H, arena.W
    dev = arena.device
    scale_x = W / CANON_SIZE
    scale_y = H / CANON_SIZE

    # Nearest unowned shrine per bot
    sd = torch.hypot(
        arena.shrine_x[:, None, :] - arena.pos_x[:, :, None],
        arena.shrine_y[:, None, :] - arena.pos_y[:, :, None],
    )  # [A, 12, 3]
    need = torch.where(
        arena.team[:, :, None] == 0,
        arena.shrine_prog[:, None, :] > -0.8,
        arena.shrine_prog[:, None, :] < 0.8,
    )
    sd_need = sd.masked_fill(~need, 1e6)
    _, best_si = sd_need.min(dim=-1)  # [A, 12]
    best_sx = torch.gather(arena.shrine_x, 1, best_si)
    best_sy = torch.gather(arena.shrine_y, 1, best_si)

    # Role goal: carrying -> home, i%3==0 -> relic, else -> best_shrine
    home_x = torch.where(arena.team == 0, 0.12 * CANON_SIZE, 0.88 * CANON_SIZE)
    home_y = torch.full_like(home_x, 0.50 * CANON_SIZE)
    is_relic_runner = (arena.bot_idx % 3) == 0

    gx_goal = torch.where(arena.carrying, home_x, torch.where(is_relic_runner, arena.relic_x[:, None], best_sx))
    gy_goal = torch.where(arena.carrying, home_y, torch.where(is_relic_runner, arena.relic_y[:, None], best_sy))

    mdx = gx_goal - arena.pos_x
    mdy = gy_goal - arena.pos_y
    ml = torch.hypot(mdx, mdy).clamp(min=1e-3)
    mvx, mvy = mdx / ml, mdy / ml

    # Sample local physics patches around each bot (matching radius ~10 canonical cells)
    pr = max(3, int(round(10.0 * scale_x)))
    offs = torch.arange(-pr, pr + 1, device=dev, dtype=torch.long)
    oy, ox = torch.meshgrid(offs, offs, indexing="ij")
    oy_f, ox_f = oy.reshape(-1), ox.reshape(-1)  # [K]

    ix, iy = arena._grid_xy(arena.pos_x, arena.pos_y)
    px_g = (ix.unsqueeze(-1) + ox_f.view(1, 1, -1)).clamp(0, W - 1)
    py_g = (iy.unsqueeze(-1) + oy_f.view(1, 1, -1)).clamp(0, H - 1)
    a_g = arena.arena_idx.unsqueeze(-1).expand_as(px_g)
    l_g = arena.bot_layer.unsqueeze(-1).expand_as(px_g)

    local_temp = arena.w_state[a_g, l_g, CH_TEMP, py_g, px_g]   # [A, 12, K]
    local_oil = arena.w_state[a_g, l_g, SP_OIL, py_g, px_g]
    local_crit = arena.e_state[a_g, 0, 0, py_g, px_g]

    # Probe 5 canonical cells ahead
    ax_p, ay_p = arena._grid_xy(arena.pos_x + mvx * 5.0, arena.pos_y + mvy * 5.0)
    fire_here = arena.w_state[arena.arena_idx, arena.bot_layer, CH_TEMP, iy, ix] > 1.5
    fire_ahead = arena.w_state[arena.arena_idx, arena.bot_layer, CH_TEMP, ay_p, ax_p] > 1.5
    water_ahead = (
        (arena.w_state[arena.arena_idx, arena.bot_layer, SP_WATER, ay_p, ax_p] > 0.45)
        & (arena.w_state[arena.arena_idx, arena.bot_layer, SP_ICE, ay_p, ax_p] < 0.35)
    )
    ice_here = arena.w_state[arena.arena_idx, arena.bot_layer, SP_ICE, iy, ix] > 0.40
    oil_nearby = local_oil.max(dim=-1).values > 0.35

    # 1. Wildfire evasion
    dash = torch.zeros((A, N), device=dev)
    evade_fire = fire_here | fire_ahead
    f_idx = local_temp.argmax(dim=-1)
    f_dx = ox_f[f_idx].float() / scale_x
    f_dy = oy_f[f_idx].float() / scale_y
    fl = torch.hypot(f_dx, f_dy).clamp(min=1e-3)
    # If hottest cell is exact bot center (0,0), steer opposite current movement
    zero_f = (f_dx.abs() + f_dy.abs()) < 1e-3
    ev_x = torch.where(zero_f, -mvx, -f_dx / fl)
    ev_y = torch.where(zero_f, -mvy, -f_dy / fl)

    mvx = torch.where(evade_fire, ev_x, mvx)
    mvy = torch.where(evade_fire, ev_y, mvy)
    dash = torch.where(fire_here & (arena.dash_cd <= 0.0) & (arena.stam > 0.25), torch.ones_like(dash), dash)

    # 2. Low-HP Wildlife Sanctuary retreat
    sanc = (~evade_fire) & (arena.hp < 0.36) & (local_crit.max(dim=-1).values > 0.25)
    c_idx = local_crit.argmax(dim=-1)
    c_dx = ox_f[c_idx].float() / scale_x
    c_dy = oy_f[c_idx].float() / scale_y
    cl = torch.hypot(c_dx, c_dy).clamp(min=1e-3)
    mvx = torch.where(sanc, c_dx / cl, mvx)
    mvy = torch.where(sanc, c_dy / cl, mvy)

    # 3. Nearest alive enemy within 60 canonical units
    dx_ij = arena.pos_x[:, None, :] - arena.pos_x[:, :, None]
    dy_ij = arena.pos_y[:, None, :] - arena.pos_y[:, :, None]
    dist_ij = torch.hypot(dx_ij, dy_ij).clamp(min=1e-3)
    en_mask = arena.enemy_pair_mask & arena.alive[:, None, :]
    en_d = dist_ij.masked_fill(~en_mask, 1e6)
    ed, ej = en_d.min(dim=-1)
    has_en = ed < 60.0

    edx = torch.gather(dx_ij, 2, ej.unsqueeze(-1)).squeeze(-1)
    edy = torch.gather(dy_ij, 2, ej.unsqueeze(-1)).squeeze(-1)
    aimx = torch.where(has_en, edx / ed.clamp(min=1e-3), mvx)
    aimy = torch.where(has_en, edy / ed.clamp(min=1e-3), mvy)

    ex_i, ey_i = arena._grid_xy(torch.gather(arena.pos_x, 1, ej), torch.gather(arena.pos_y, 1, ej))
    el_i = torch.gather(arena.bot_layer, 1, ej)
    enemy_on_oil = has_en & (arena.w_state[arena.arena_idx, el_i, SP_OIL, ey_i, ex_i] > 0.25)
    enemy_in_water = has_en & (arena.w_state[arena.arena_idx, el_i, SP_WATER, ey_i, ex_i] > 0.40)

    melee = (ed < 6.2).float()
    carve_ice = (ice_here & arena.carrying).float()

    oil_cond = oil_nearby | enemy_on_oil
    water_cond = (~oil_cond) & (water_ahead | enemy_in_water)
    dry_cond = (~oil_cond) & (~water_cond)

    dry_elem = torch.where(
        (arena.cls == 1) & (ed < 38.0),
        torch.full((A, N), 2, dtype=torch.long, device=dev),
        torch.where(arena.cls == 2, torch.zeros((A, N), dtype=torch.long, device=dev), torch.ones((A, N), dtype=torch.long, device=dev)),
    )
    elem = torch.where(oil_cond, torch.zeros_like(dry_elem), torch.where(water_cond, torch.ones_like(dry_elem), dry_elem))

    cast_bolt = torch.where(
        oil_cond | water_cond,
        (arena.mana > 0.12).float(),
        ((ed < 48.0) & (arena.mana > 0.15)).float(),
    )
    ab_e = torch.where(
        water_cond,
        (water_ahead & (arena.e_cd <= 0.0) & (arena.mana > 0.30) & (arena.cls == 0)).float(),
        torch.where(dry_cond, ((ed < 24.0) & (arena.e_cd <= 0.0) & (arena.mana > 0.30)).float(), torch.zeros((A, N), device=dev)),
    )
    ult_r = torch.where(
        oil_cond,
        ((ed < 22.0) & (arena.r_cd <= 0.0) & (arena.mana > 0.55) & (arena.cls == 1)).float(),
        torch.where(dry_cond, ((ed < 20.0) & (arena.r_cd <= 0.0) & (arena.mana > 0.55)).float(), torch.zeros((A, N), device=dev)),
    )

    act_vec = torch.stack([mvx, mvy, aimx, aimy], dim=-1)
    act_bin = torch.stack([dash, melee, cast_bolt, carve_ice, ab_e, ult_r], dim=-1)
    return act_vec, act_bin, elem


@torch.inference_mode()
def policy_deterministic_act(
    policy: BotPolicyNet,
    z_world: torch.Tensor,
    z_eco: torch.Tensor,
    bot_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec, bin_logits, elem_logits, _ = batched_policy_forward(policy, z_world, z_eco, bot_feats)
    return vec, (torch.sigmoid(bin_logits) > 0.5).float(), elem_logits.argmax(dim=-1)


def run_ppo_stage(
    stage_name: str,
    ac: RLBotActorCritic,
    bc_policy: BotPolicyNet,
    arena: FrostmarshGPUArena,
    optimizer: torch.optim.Optimizer,
    num_updates: int,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
    kl_coef: float,
    ent_coef: float,
    seed_offset: int,
) -> dict:
    """Run vectorized PPO self-play stage on GPU arenas."""
    A, N = arena.A, arena.N
    dev = arena.device
    lagged_policy = copy.deepcopy(ac.policy).eval()
    for p in lagged_policy.parameters():
        p.requires_grad_(False)

    # Split arenas across opponent mix (with West/East symmetry!)
    # 0..n_self-1: Self-play (both sides RL or RL vs lagged RL)
    # n_self..n_self+n_bc-1: RL vs BC (half West, half East)
    # rest: RL vs Scripted Teacher (half West, half East)
    n_self = max(2, A // 3)
    n_bc = max(2, (A - n_self) // 2)
    n_teach = A - n_self - n_bc

    # Boolean mask [A, 12] indicating which bots belong to the learning RL policy
    rl_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    # Self-play arenas: first half pure mirror (both sides train), second half RL vs lagged snapshot
    n_mirror = max(1, n_self // 2)
    rl_mask[:n_mirror, :] = True
    rl_mask[n_mirror:n_self, :6] = True  # West = RL, East = lagged
    # BC arenas: half RL=West, half RL=East
    bc_start = n_self
    bc_mid = bc_start + n_bc // 2
    rl_mask[bc_start:bc_mid, :6] = True
    rl_mask[bc_mid : bc_start + n_bc, 6:] = True
    # Teacher arenas: half RL=West, half RL=East
    t_start = bc_start + n_bc
    t_mid = t_start + n_teach // 2
    rl_mask[t_start:t_mid, :6] = True
    rl_mask[t_mid:, 6:] = True

    opp_lag_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_lag_mask[n_mirror:n_self, 6:] = True
    opp_bc_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_bc_mask[bc_start:bc_mid, 6:] = True
    opp_bc_mask[bc_mid : bc_start + n_bc, :6] = True
    opp_teach_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_teach_mask[t_start:t_mid, 6:] = True
    opp_teach_mask[t_mid:, :6] = True

    t0 = time.perf_counter()
    feats = arena.reset(seed=seed_offset)
    total_bot_steps = 0
    recent_vs_teach_wr = []
    recent_vs_bc_wr = []
    best_metric = -1e9
    best_policy_sd = copy.deepcopy(ac.policy.state_dict())

    for upd in range(num_updates):
        # Update lagged self-play snapshot every 10 updates
        if upd > 0 and upd % 10 == 0:
            lagged_policy.load_state_dict(ac.policy.state_dict())

        # Rollout storage (store ego-extracted latent snapshots or per-refresh Z references!)
        # Memory-efficient trick: Z_world only refreshes every refresh_every ticks, so store unique Z snapshots!
        z_world_snaps: list[torch.Tensor] = []
        z_eco_snaps: list[torch.Tensor] = []
        step_z_idx = torch.zeros((rollout_steps,), dtype=torch.long, device=dev)

        buf_feats = torch.empty((rollout_steps, A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)
        buf_avec = torch.empty((rollout_steps, A, N, 4), dtype=torch.float32, device=dev)
        buf_abin = torch.empty((rollout_steps, A, N, 6), dtype=torch.float32, device=dev)
        buf_aelem = torch.empty((rollout_steps, A, N), dtype=torch.long, device=dev)
        buf_lp = torch.empty((rollout_steps, A, N), dtype=torch.float32, device=dev)
        buf_val = torch.empty((rollout_steps, A, N), dtype=torch.float32, device=dev)
        buf_rew = torch.empty((rollout_steps, A, N), dtype=torch.float32, device=dev)
        buf_done = torch.empty((rollout_steps,), dtype=torch.float32, device=dev)

        ac.eval()
        last_zw_ptr = -1
        with torch.inference_mode():
            for t in range(rollout_steps):
                # Snapshot Z_world/Z_eco only when refreshed (saves 4x VRAM!)
                cur_ptr = arena.z_world.data_ptr()
                if cur_ptr != last_zw_ptr or t == 0:
                    z_world_snaps.append(arena.z_world.clone())
                    z_eco_snaps.append(arena.z_eco.clone())
                    last_zw_ptr = cur_ptr
                step_z_idx[t] = len(z_world_snaps) - 1

                buf_feats[t] = feats

                # Sample RL actions for all bots
                rl_vec, rl_bin, rl_elem, rl_lp, rl_val = ac.sample_actions(arena.z_world, arena.z_eco, feats, deterministic=False)

                # Compute opponent actions
                act_vec = rl_vec.clone()
                act_bin = rl_bin.clone()
                act_elem = rl_elem.clone()

                if opp_lag_mask.any():
                    lv, lb, le = policy_deterministic_act(lagged_policy, arena.z_world, arena.z_eco, feats)
                    act_vec = torch.where(opp_lag_mask.unsqueeze(-1), lv, act_vec)
                    act_bin = torch.where(opp_lag_mask.unsqueeze(-1), lb, act_bin)
                    act_elem = torch.where(opp_lag_mask, le, act_elem)

                if opp_bc_mask.any():
                    bv, bb, be = policy_deterministic_act(bc_policy, arena.z_world, arena.z_eco, feats)
                    act_vec = torch.where(opp_bc_mask.unsqueeze(-1), bv, act_vec)
                    act_bin = torch.where(opp_bc_mask.unsqueeze(-1), bb, act_bin)
                    act_elem = torch.where(opp_bc_mask, be, act_elem)

                if opp_teach_mask.any():
                    tv, tb, te = scripted_teacher_act(arena)
                    act_vec = torch.where(opp_teach_mask.unsqueeze(-1), tv, act_vec)
                    act_bin = torch.where(opp_teach_mask.unsqueeze(-1), tb, act_bin)
                    act_elem = torch.where(opp_teach_mask, te, act_elem)

                buf_avec[t] = rl_vec
                buf_abin[t] = rl_bin
                buf_aelem[t] = rl_elem
                buf_lp[t] = rl_lp
                buf_val[t] = rl_val

                next_feats, rews, done = arena.step(act_vec, act_bin, act_elem)
                buf_rew[t] = rews
                buf_done[t] = 1.0 if done else 0.0
                total_bot_steps += A * N

                if done:
                    # Log match win rates vs BC and vs Teacher before resetting
                    s_w = arena.team_score[:, 0]
                    s_e = arena.team_score[:, 1]
                    bc_rl_w = (s_w[bc_start:bc_mid] > s_e[bc_start:bc_mid]).float()
                    bc_rl_e = (s_e[bc_mid : bc_start + n_bc] > s_w[bc_mid : bc_start + n_bc]).float()
                    recent_vs_bc_wr.append(float(torch.cat([bc_rl_w, bc_rl_e]).mean().item()))

                    t_rl_w = (s_w[t_start:t_mid] > s_e[t_start:t_mid]).float()
                    t_rl_e = (s_e[t_mid:] > s_w[t_mid:]).float()
                    recent_vs_teach_wr.append(float(torch.cat([t_rl_w, t_rl_e]).mean().item()))

                    feats = arena.reset(seed=seed_offset + (upd + 1) * 100 + t)
                else:
                    feats = next_feats

            # Bootstrap value for GAE
            _, _, _, last_val = ac.forward_batched(arena.z_world, arena.z_eco, feats)

        # Compute GAE advantages & returns [T, A, N]
        gamma = 0.98
        lam = 0.95
        adv = torch.zeros_like(buf_rew)
        gae = torch.zeros((A, N), device=dev)
        for t in reversed(range(rollout_steps)):
            nonterm = 1.0 - buf_done[t]
            next_v = last_val if t == rollout_steps - 1 else buf_val[t + 1]
            delta = buf_rew[t] + gamma * next_v * nonterm - buf_val[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_val

        # Precompute ego-crops or train directly by grouping steps sharing the same Z snapshot!
        # Even faster & cleaner: iterate over unique Z snapshots inside PPO epochs (each snapshot has
        # refresh_every * A * N samples = e.g. 4 * 32 * 12 = 1536 bots, perfectly sized GPU minibatch!)
        rl_mask_t = rl_mask.unsqueeze(0).expand(rollout_steps, A, N)
        adv_rl = adv[rl_mask_t]
        adv_norm = (adv - adv_rl.mean()) / (adv_rl.std() + 1e-6)

        # Precompute frozen BC logits for KL anchor on every rollout step
        bc_vecs = torch.empty_like(buf_avec)
        bc_bins = torch.empty_like(buf_abin)
        bc_elems = torch.empty((rollout_steps, A, N, 3), dtype=torch.float32, device=dev)
        with torch.inference_mode():
            for t in range(rollout_steps):
                zi = int(step_z_idx[t].item())
                bv, bb, be, _ = batched_policy_forward(bc_policy, z_world_snaps[zi], z_eco_snaps[zi], buf_feats[t])
                bc_vecs[t] = bv
                bc_bins[t] = bb
                bc_elems[t] = be

        ac.train()
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_kl_loss = 0.0
        n_opt_steps = 0

        # Group time steps by Z snapshot index so each forward pass reuses Z_world/Z_eco zero-copy!
        snap_groups: dict[int, list[int]] = {}
        for t in range(rollout_steps):
            zi = int(step_z_idx[t].item())
            snap_groups.setdefault(zi, []).append(t)
        group_keys = list(snap_groups.keys())

        for _ in range(ppo_epochs):
            perm = torch.randperm(len(group_keys)).tolist()
            for gi in perm:
                zi = group_keys[gi]
                ts = snap_groups[zi]
                zw = z_world_snaps[zi]
                ze = z_eco_snaps[zi]

                # Stack steps sharing this Z snapshot along bot dim: [A, len(ts)*N, ...]
                t_idx = torch.tensor(ts, dtype=torch.long, device=dev)
                k = len(ts)
                f_b = buf_feats[t_idx].permute(1, 0, 2, 3).reshape(A, k * N, BOT_FEAT_DIM)
                av_b = buf_avec[t_idx].permute(1, 0, 2, 3).reshape(A, k * N, 4)
                ab_b = buf_abin[t_idx].permute(1, 0, 2, 3).reshape(A, k * N, 6)
                ae_b = buf_aelem[t_idx].permute(1, 0, 2).reshape(A, k * N)
                old_lp_b = buf_lp[t_idx].permute(1, 0, 2).reshape(A, k * N)
                adv_b = adv_norm[t_idx].permute(1, 0, 2).reshape(A, k * N)
                ret_b = ret[t_idx].permute(1, 0, 2).reshape(A, k * N)
                mask_b = rl_mask.unsqueeze(1).expand(A, k, N).reshape(A, k * N).float()
                bcv_b = bc_vecs[t_idx].permute(1, 0, 2, 3).reshape(A, k * N, 4)
                bcb_b = bc_bins[t_idx].permute(1, 0, 2, 3).reshape(A, k * N, 6)
                bce_b = bc_elems[t_idx].permute(1, 0, 2, 3).reshape(A, k * N, 3)

                m_sum = mask_b.sum().clamp(min=1.0)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    new_lp, ent, vals, vec_p, bin_p, elem_p = ac.evaluate_actions(zw, ze, f_b, av_b, ab_b, ae_b)
                # Compute PPO + KL losses in strict float32 for numerical stability
                new_lp, ent, vals = new_lp.float(), ent.float(), vals.float()
                ratio = torch.exp((new_lp - old_lp_b.float()).clamp(-10.0, 10.0))
                surr1 = ratio * adv_b.float()
                surr2 = ratio.clamp(0.82, 1.18) * adv_b.float()
                pg_loss = -(torch.minimum(surr1, surr2) * mask_b).sum() / m_sum
                v_loss = 0.5 * (((vals - ret_b.float()) ** 2) * mask_b).sum() / m_sum
                ent_loss = -(ent * mask_b).sum() / m_sum
                kl_val = compute_bc_kl_loss(vec_p, bin_p, elem_p, bcv_b, bcb_b, bce_b)
                kl_loss = (kl_val * mask_b).sum() / m_sum
                loss = pg_loss + 0.5 * v_loss + ent_coef * ent_loss + kl_coef * kl_loss

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                    optimizer.step()

                total_pg_loss += float(pg_loss.item())
                total_v_loss += float(v_loss.item())
                total_kl_loss += float(kl_loss.item())
                n_opt_steps += 1
                if float(kl_loss.item()) > 0.35:
                    break
            if n_opt_steps > 0 and (total_kl_loss / n_opt_steps) > 0.35:
                break

        mean_rew = float(buf_rew[rl_mask_t].mean().item())
        cur_wr_b = float(np.mean(recent_vs_bc_wr[-4:])) if recent_vs_bc_wr else 0.0
        cur_wr_t = float(np.mean(recent_vs_teach_wr[-4:])) if recent_vs_teach_wr else 0.0
        cur_metric = mean_rew + 1.5 * cur_wr_b + 3.0 * cur_wr_t
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_policy_sd = copy.deepcopy(ac.policy.state_dict())

        if (upd + 1) % max(1, num_updates // 5) == 0 or upd == num_updates - 1:
            elapsed = time.perf_counter() - t0
            wr_t = cur_wr_t * 100.0
            wr_b = cur_wr_b * 100.0
            print(
                f"[{stage_name}] upd {upd+1:3d}/{num_updates} | t={elapsed:5.1f}s | "
                f"steps={total_bot_steps:,} | rew={mean_rew:+.3f} | "
                f"vsTeach={wr_t:5.1f}% | vsBC={wr_b:5.1f}% | "
                f"kl={total_kl_loss/max(n_opt_steps,1):.3f} | std={ac.get_std().mean().item():.3f}",
                flush=True,
            )

    ac.policy.load_state_dict(best_policy_sd)
    elapsed = time.perf_counter() - t0
    return {
        "stage": stage_name,
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "parallel_arenas": A,
        "concurrent_bots": A * N,
        "updates": num_updates,
        "total_bot_steps": total_bot_steps,
        "elapsed_s": round(elapsed, 2),
        "bot_steps_per_sec": round(total_bot_steps / max(elapsed, 1e-3), 1),
        "final_train_wr_vs_teacher_pct": round(float(np.mean(recent_vs_teach_wr[-4:])) * 100.0, 1) if recent_vs_teach_wr else 0.0,
        "final_train_wr_vs_bc_pct": round(float(np.mean(recent_vs_bc_wr[-4:])) * 100.0, 1) if recent_vs_bc_wr else 0.0,
    }


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(2026)

    t_global = time.perf_counter()
    print("=== Zero-Copy GPU Self-Play RL for Frostmarsh BotPolicyNet ===")
    wm, em = load_frozen_world_models(device)

    # Load BC checkpoint as both warm-start and frozen KL anchor
    bc_sd = load_ckpt_state_dict(BC_PT, device)
    bc_policy = BotPolicyNet(blind_world=False).to(device).eval()
    bc_policy.load_state_dict(bc_sd)
    for p in bc_policy.parameters():
        p.requires_grad_(False)

    rl_policy = BotPolicyNet(blind_world=False).to(device)
    rl_policy.load_state_dict(bc_sd)
    ac = RLBotActorCritic(rl_policy).to(device)

    # Verify batched_policy_forward matches BotPolicyNet.forward_logits bit-for-bit!
    with torch.inference_mode():
        tz_w = torch.randn((1, 4, 48, 64, 64), device=device)
        tz_e = torch.randn((1, 1, 32, 64, 64), device=device)
        tf = torch.rand((1, 12, BOT_FEAT_DIM), device=device)
        v1, b1, e1 = rl_policy.forward_logits(tz_w[0], tz_e[0], tf[0])
        v2, b2, e2, _ = batched_policy_forward(rl_policy, tz_w, tz_e, tf)
        diff = max((v1 - v2[0]).abs().max().item(), (b1 - b2[0]).abs().max().item(), (e1 - e2[0]).abs().max().item())
        assert diff < 1e-4, f"Batched forward mismatch: {diff}"
        print(f"Verified batched zero-copy BotPolicyNet equivalence: max_diff={diff:.2e} | params={count_parameters(rl_policy):,}")

    # Stage 1: 32 parallel 128x128x2 arenas x 12 bots = 384 concurrent bots
    arena_128 = FrostmarshGPUArena(
        num_arenas=32,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=128,
        num_layers=2,
        max_ticks=120,
        latent_refresh_every=6,
    )
    # Stage 0: Fast Online DAgger calibration inside real GPU arenas (aligns Shrine/Relic macro to 3D Z_world)
    print("Running fast Stage 0 DAgger macro calibration on 32 parallel arenas...", flush=True)
    opt_dag = torch.optim.AdamW(ac.policy.parameters(), lr=6e-4, weight_decay=1e-4)
    feats_d = arena_128.reset(seed=777)
    ac.policy.train()
    for d_step in range(90):
        with torch.no_grad():
            tv, tb, te = scripted_teacher_act(arena_128)
            tv_t, tb_t, te_t = tv.clone(), tb.clone(), te.clone()
            pv, pb, pe = policy_deterministic_act(ac.policy, arena_128.z_world, arena_128.z_eco, feats_d)
            use_p = (d_step >= 20)
            mix_v = pv if use_p else tv
            mix_b = pb if use_p else tb
            mix_e = pe if use_p else te
            zw_d, ze_d, fd_d = arena_128.z_world.clone(), arena_128.z_eco.clone(), feats_d.clone()
        opt_dag.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vec_o, bin_o, elem_o, _ = batched_policy_forward(ac.policy, zw_d, ze_d, fd_d)
            cos_mv = 1.0 - F.cosine_similarity(vec_o[..., :2], tv_t[..., :2], dim=-1).mean()
            cos_am = 1.0 - F.cosine_similarity(vec_o[..., 2:], tv_t[..., 2:], dim=-1).mean()
            l_vec = F.smooth_l1_loss(vec_o, tv_t) + 0.6 * cos_mv + 0.6 * cos_am
            l_bin = F.binary_cross_entropy_with_logits(bin_o, tb_t)
            l_elm = F.cross_entropy(elem_o.reshape(-1, 3), te_t.reshape(-1))
            loss_d = 1.5 * l_vec + 1.0 * l_bin + 1.0 * l_elm
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(ac.policy.parameters(), 1.0)
        opt_dag.step()
        with torch.inference_mode():
            feats_d, _, done_d = arena_128.step(mix_v, mix_b, mix_e)
            if done_d:
                feats_d = arena_128.reset(seed=777 + d_step)
    # Refresh KL anchor to calibrated policy so PPO improves strictly BEYOND macro-calibrated BC!
    bc_policy.load_state_dict(ac.policy.state_dict())

    opt1 = torch.optim.AdamW(
        [
            {"params": ac.policy.parameters(), "lr": 8.0e-5, "weight_decay": 1e-4},
            {"params": ac.value_head.parameters(), "lr": 1.0e-3, "weight_decay": 1e-4},
            {"params": [ac.log_std], "lr": 1.0e-4},
        ]
    )
    s1 = run_ppo_stage(
        stage_name="Stage1-128x128x2 (32x12=384 bots)",
        ac=ac,
        bc_policy=bc_policy,
        arena=arena_128,
        optimizer=opt1,
        num_updates=14,
        rollout_steps=40,
        ppo_epochs=2,
        minibatch_size=2048,
        kl_coef=0.08,
        ent_coef=0.002,
        seed_offset=1000,
    )
    del arena_128
    torch.cuda.empty_cache()

    # Stage 2: High-res 256x256x4 polish (8 parallel arenas x 12 bots = 96 concurrent bots)
    arena_256 = FrostmarshGPUArena(
        num_arenas=8,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=256,
        num_layers=4,
        max_ticks=120,
        latent_refresh_every=6,
    )
    opt2 = torch.optim.AdamW(
        [
            {"params": ac.policy.parameters(), "lr": 5.0e-5, "weight_decay": 1e-4},
            {"params": ac.value_head.parameters(), "lr": 6.0e-4, "weight_decay": 1e-4},
            {"params": [ac.log_std], "lr": 5.0e-5},
        ]
    )
    s2 = run_ppo_stage(
        stage_name="Stage2-256x256x4 Polish (8x12=96 bots)",
        ac=ac,
        bc_policy=bc_policy,
        arena=arena_256,
        optimizer=opt2,
        num_updates=6,
        rollout_steps=40,
        ppo_epochs=2,
        minibatch_size=1024,
        kl_coef=0.08,
        ent_coef=0.001,
        seed_offset=50000,
    )

    total_time = round(time.perf_counter() - t_global, 2)

    # Save PURE BotPolicyNet state_dict() ONLY (100% drop-in compatible with gpu_infer_daemon.py!)
    pure_sd = {k: v.detach().cpu() for k, v in ac.policy.state_dict().items()}
    # Verify strict reload into a clean BotPolicyNet
    verify_net = BotPolicyNet()
    verify_net.load_state_dict(pure_sd, strict=True)

    torch.save(
        {
            "state_dict": pure_sd,
            "rl_training": {
                "total_time_s": total_time,
                "stage1": s1,
                "stage2": s2,
            },
        },
        RL_PT,
    )
    print(f"\nSaved drop-in pure BotPolicyNet checkpoint to {RL_PT} in {total_time:.1f}s!")
    print(json.dumps({"stage1": s1, "stage2": s2, "total_time_s": total_time}, indent=2))


if __name__ == "__main__":
    main()
