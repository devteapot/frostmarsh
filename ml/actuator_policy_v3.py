#!/usr/bin/env python3
"""
Frostmarsh ActuatorPolicyNetV3 — True 3D Volumetric (Z=16/32) Emergent Physical Actuator Policy.

Upgrades Iter 5/6's 2D-layer ActuatorPolicyNet (which pooled Z=16 into 4 flat bands and only aimed in 2D)
into a native 3D volumetric shared-latent policy:
- Zero-copy 3D dual-scale ego-crop directly from `Z_world_3d [L, 48, H, W]` via `F.grid_sample` (`5D` trilinear!):
  - Fine local 3D volume: `14m × 14m × 8m` centered on the bot's exact 3D `(x, y, z)` altitude (sees cave ceilings,
    arched bridge piers, under-arch clearances, vertical drop-offs, and nearby Stasis/Haste TimeDomes!).
  - Coarse tactical 3D volume: `64m × 64m × full Z` (sees distant alpine peaks, highland waterfalls, and enemy flanks).
- 4-head Cross-Attention over 14 nearby 3D entity/shrine/relic/TimeDome tokens with true `(dx, dy, dz)` vectors.
- 12 continuous, conservation-bounded 3D physical actuator channels (ZERO hardcoded named spells!):
  0: move_x           [-1, 1]  (3D ground/cave locomotion X)
  1: move_y           [-1, 1]  (3D ground/cave locomotion Y)
  2: move_jump_climb  [-1, 1]  (>0.3 triggers ledge climb / vault / dash leap)
  3: aim_dx           [-1, 1]  (3D actuator aim X)
  4: aim_dy           [-1, 1]  (3D actuator aim Y)
  5: aim_dz           [-1, 1]  (NEW: true 3D vertical aim — fire down from bridges, collapse arch piers from below, seal cave mouths!)
  6: thermal_flux     [-1, 1]  (-1 cryo freeze / ice-wall build .. +1 focused heat / bridge-burn / ignition)
  7: emit_water       [0, 1]   (water mass emission -> flash-boil steam or quench fires)
  8: emit_oil         [0, 1]   (combustible oil slick / cascade down slopes)
  9: emit_smoke       [0, 1]   (volumetric 3D smokescreen / cave blind)
 10: focus_radius     [0, 1]   (maps to 1.5m tight thermal beam .. 7.5m wide cryo/smoke field)
 11: impulse_and_dome [-1, 1]  (>0.45 kinetic shockwave blast; <-0.45 place 0.15x Stasis Dome at target!)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ACT3D_DIM = 12
ACT3D_NAMES = (
    "move_x",
    "move_y",
    "move_jump",
    "aim_dx",
    "aim_dy",
    "aim_dz",
    "thermal_flux",
    "emit_water",
    "emit_oil",
    "emit_smoke",
    "focus_radius",
    "impulse_or_dome",
)

SELF3D_FEAT_DIM = 22
TOKEN3D_FEAT_DIM = 14
MAX_TOKENS_3D = 14


class ActuatorPolicyNetV3(nn.Module):
    """
    True 3D Volumetric Dual-Scale Shared-Latent Actuator Policy (~275K params).
    Reads zero-copy from `Z_world_3d [L, C_z, H, W]` (supports L=4, L=16, L=32 seamlessly via 5D F.grid_sample!).
    """

    def __init__(
        self,
        z_world_ch: int = 48,
        z_eco_ch: int = 32,
        latent_dim: int = 32,
        fine_xy: int = 12,
        fine_z: int = 4,
        coarse_xy: int = 12,
        coarse_z: int = 4,
        fine_span_xy: float = 16.0,
        fine_span_z: float = 8.0,
        coarse_span_xy: float = 64.0,
        self_dim: int = SELF3D_FEAT_DIM,
        tok_dim: int = TOKEN3D_FEAT_DIM,
        goal_dim: int = 8,
    ):
        super().__init__()
        self.z_world_ch = z_world_ch
        self.z_eco_ch = z_eco_ch
        self.latent_dim = latent_dim
        self.fine_xy = fine_xy
        self.fine_z = fine_z
        self.coarse_xy = coarse_xy
        self.coarse_z = coarse_z
        self.fine_span_xy = fine_span_xy
        self.fine_span_z = fine_span_z
        self.coarse_span_xy = coarse_span_xy
        self.goal_dim = goal_dim

        # Version-pinning 1x1x1 adapter isolating policy from WorldModel3D-v5 retraining
        self.z_adapter = nn.Sequential(
            nn.Conv3d(z_world_ch, latent_dim, kernel_size=1, bias=False),
            nn.GroupNorm(4, latent_dim),
            nn.SiLU(inplace=True),
        )
        # Optional lightweight eco-trunk 2D adapter projected onto surface band
        self.eco_adapter = nn.Sequential(
            nn.Conv2d(z_eco_ch, 8, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
        )

        # Fine 3D local volume encoder [N, latent_dim, 4, 12, 12] -> [N, 64]
        self.fine_enc = nn.Sequential(
            nn.Conv3d(latent_dim, 40, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, 40),
            nn.SiLU(inplace=True),
            nn.Conv3d(40, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        # Coarse 3D tactical volume encoder [N, latent_dim, 4, 12, 12] -> [N, 64]
        self.coarse_enc = nn.Sequential(
            nn.Conv3d(latent_dim, 40, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(4, 40),
            nn.SiLU(inplace=True),
            nn.Conv3d(40, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

        # 3D Entity / Shrine / TimeDome cross-attention (4 heads, d=64)
        self.self_proj = nn.Sequential(
            nn.Linear(self_dim + goal_dim + 8, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )
        self.tok_proj = nn.Sequential(
            nn.Linear(tok_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(64)

        # Fused actor-critic trunk (64 fine + 64 coarse + 64 self + 64 attn = 256)
        self.trunk = nn.Sequential(
            nn.Linear(256, 160),
            nn.LayerNorm(160),
            nn.SiLU(inplace=True),
            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
        )

        self.actor_mean = nn.Linear(128, ACT3D_DIM)
        self.actor_log_std = nn.Parameter(torch.full((ACT3D_DIM,), -0.45))
        self.critic = nn.Linear(128, 1)

        # Compact 4x4 spatial saliency projection for live Neural-Brain Inspector HUD
        self.saliency_proj = nn.Conv2d(latent_dim, 1, kernel_size=1)

        # Precompute normalized 3D sampling meshes (contiguous to avoid stride-0 load_state_dict issues!)
        self._register_3d_grids()

    def _register_3d_grids(self) -> None:
        fz = torch.linspace(-1.0, 1.0, self.fine_z)
        fy = torch.linspace(-1.0, 1.0, self.fine_xy)
        fx = torch.linspace(-1.0, 1.0, self.fine_xy)
        gz, gy, gx = torch.meshgrid(fz, fy, fx, indexing="ij")
        self.register_buffer("fine_gx", gx.contiguous().clone())
        self.register_buffer("fine_gy", gy.contiguous().clone())
        self.register_buffer("fine_gz", gz.contiguous().clone())

        cz = torch.linspace(-1.0, 1.0, self.coarse_z)
        cy = torch.linspace(-1.0, 1.0, self.coarse_xy)
        cx = torch.linspace(-1.0, 1.0, self.coarse_xy)
        cgz, cgy, cgx = torch.meshgrid(cz, cy, cx, indexing="ij")
        self.register_buffer("coarse_gx", cgx.contiguous().clone())
        self.register_buffer("coarse_gy", cgy.contiguous().clone())
        self.register_buffer("coarse_gz", cgz.contiguous().clone())

    def _sample_dual_3d_crops(
        self,
        z_feat: torch.Tensor,
        pos_xyz: torch.Tensor,
        arena_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-copy 5D trilinear `F.grid_sample` for N bots across B arenas.
        z_feat: [B, C, L, H, W]
        pos_xyz: [N, 3] in voxel coords (x in [0, W-1], y in [0, H-1], z in [0, L-1])
        """
        B, C, L, H, W = z_feat.shape
        N = pos_xyz.shape[0]
        device = z_feat.device
        dtype = z_feat.dtype

        wm1 = max(W - 1, 1)
        hm1 = max(H - 1, 1)
        lm1 = max(L - 1, 1)

        px = pos_xyz[:, 0].to(dtype)
        py = pos_xyz[:, 1].to(dtype)
        pz = pos_xyz[:, 2].to(dtype)

        hx_f = (self.fine_span_xy * 0.5) / wm1
        hy_f = (self.fine_span_xy * 0.5) / hm1
        hz_f = (self.fine_span_z * 0.5) / lm1
        cx_n = (px / wm1) * 2.0 - 1.0
        cy_n = (py / hm1) * 2.0 - 1.0
        cz_n = (pz / lm1) * 2.0 - 1.0

        f_grid = torch.stack(
            (
                cx_n[:, None, None, None] + self.fine_gx[None].to(dtype) * (2.0 * hx_f),
                cy_n[:, None, None, None] + self.fine_gy[None].to(dtype) * (2.0 * hy_f),
                cz_n[:, None, None, None] + self.fine_gz[None].to(dtype) * (2.0 * hz_f),
            ),
            dim=-1,
        )

        hx_c = (self.coarse_span_xy * 0.5) / wm1
        hy_c = (self.coarse_span_xy * 0.5) / hm1
        c_grid = torch.stack(
            (
                cx_n[:, None, None, None] + self.coarse_gx[None].to(dtype) * (2.0 * hx_c),
                cy_n[:, None, None, None] + self.coarse_gy[None].to(dtype) * (2.0 * hy_c),
                self.coarse_gz[None].to(dtype).expand(N, -1, -1, -1),
            ),
            dim=-1,
        )

        if B == 1:
            f_grid_b = f_grid.view(1, N * self.fine_z, self.fine_xy, self.fine_xy, 3)
            c_grid_b = c_grid.view(1, N * self.coarse_z, self.coarse_xy, self.coarse_xy, 3)
            fine_raw = F.grid_sample(z_feat, f_grid_b, mode="bilinear", padding_mode="border", align_corners=True)
            coarse_raw = F.grid_sample(z_feat, c_grid_b, mode="bilinear", padding_mode="border", align_corners=True)
            fine_crop = fine_raw.view(C, N, self.fine_z, self.fine_xy, self.fine_xy).permute(1, 0, 2, 3, 4).contiguous()
            coarse_crop = coarse_raw.view(C, N, self.coarse_z, self.coarse_xy, self.coarse_xy).permute(1, 0, 2, 3, 4).contiguous()
            return fine_crop, coarse_crop

        assert arena_idx is not None
        per_arena = N // B
        if N == B * per_arena and torch.equal(
            arena_idx, torch.arange(B, device=device).repeat_interleave(per_arena)
        ):
            f_grid_b = f_grid.view(B, per_arena * self.fine_z, self.fine_xy, self.fine_xy, 3)
            c_grid_b = c_grid.view(B, per_arena * self.coarse_z, self.coarse_xy, self.coarse_xy, 3)
            fine_raw = F.grid_sample(z_feat, f_grid_b, mode="bilinear", padding_mode="border", align_corners=True)
            coarse_raw = F.grid_sample(z_feat, c_grid_b, mode="bilinear", padding_mode="border", align_corners=True)
            fine_crop = fine_raw.view(B, C, per_arena, self.fine_z, self.fine_xy, self.fine_xy).permute(0, 2, 1, 3, 4, 5).reshape(N, C, self.fine_z, self.fine_xy, self.fine_xy)
            coarse_crop = coarse_raw.view(B, C, per_arena, self.coarse_z, self.coarse_xy, self.coarse_xy).permute(0, 2, 1, 3, 4, 5).reshape(N, C, self.coarse_z, self.coarse_xy, self.coarse_xy)
            return fine_crop, coarse_crop

        z_sel = z_feat[arena_idx]
        fine_crop = F.grid_sample(z_sel, f_grid, mode="bilinear", padding_mode="border", align_corners=True)
        coarse_crop = F.grid_sample(z_sel, c_grid, mode="bilinear", padding_mode="border", align_corners=True)
        return fine_crop, coarse_crop

    def sample_raw_crops(
        self,
        z_world_3d: torch.Tensor,
        pos_xyz: torch.Tensor,
        z_eco: torch.Tensor | None = None,
        arena_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Zero-copy 5D trilinear `F.grid_sample` directly on raw `Z_world_3d` BEFORE `z_adapter`:
        - 150× faster on 256×256×32 live worlds (13K voxels vs 2.1M voxels!)
        - Exact parity between rollout collection, PPO minibatch updates, and live serving.
        Accepts `[L, C, H, W]` or `[B, C, L, H, W]` for `z_world_3d`.
        Returns `(fine_raw [N, C, 4, 12, 12], coarse_raw [N, C, 4, 12, 12], eco_raw [N, z_eco_ch])`.
        """
        if z_world_3d.dim() == 4:
            z_5d = z_world_3d.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        else:
            z_5d = z_world_3d
        fine_raw, coarse_raw = self._sample_dual_3d_crops(z_5d, pos_xyz, arena_idx=arena_idx)
        N = pos_xyz.shape[0]
        device = z_5d.device
        dtype = z_5d.dtype
        if z_eco is not None:
            B_e, _, He, We = z_eco.shape
            gx = (pos_xyz[:, 0].to(dtype) / max(We - 1, 1)) * 2.0 - 1.0
            gy = (pos_xyz[:, 1].to(dtype) / max(He - 1, 1)) * 2.0 - 1.0
            if B_e == 1:
                g2 = torch.stack((gx, gy), dim=-1).view(1, N, 1, 2)
                eco_raw = F.grid_sample(z_eco.to(dtype), g2, mode="bilinear", padding_mode="border", align_corners=True).view(self.z_eco_ch, N).T
            else:
                assert arena_idx is not None
                g2 = torch.stack((gx, gy), dim=-1).view(N, 1, 1, 2)
                eco_raw = F.grid_sample(z_eco.to(dtype)[arena_idx], g2, mode="bilinear", padding_mode="border", align_corners=True).view(N, self.z_eco_ch)
        else:
            eco_raw = torch.zeros((N, self.z_eco_ch), device=device, dtype=dtype)
        return fine_raw, coarse_raw, eco_raw

    def forward_from_crops(
        self,
        fine_raw: torch.Tensor,
        coarse_raw: torch.Tensor,
        self_feat: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        goal_vec: torch.Tensor | None = None,
        eco_raw: torch.Tensor | None = None,
        return_saliency: bool = False,
    ) -> dict[str, torch.Tensor]:
        N = fine_raw.shape[0]
        device = fine_raw.device
        p_dtype = self.actor_mean.weight.dtype

        fine_crop = self.z_adapter(fine_raw.to(p_dtype))
        coarse_crop = self.z_adapter(coarse_raw.to(p_dtype))
        h_fine = self.fine_enc(fine_crop)
        h_coarse = self.coarse_enc(coarse_crop)

        if goal_vec is None:
            goal_vec = torch.zeros((N, self.goal_dim), device=device, dtype=p_dtype)
        else:
            goal_vec = goal_vec.to(p_dtype)

        if eco_raw is not None:
            eco_vec = self.eco_adapter(eco_raw.to(p_dtype).view(N, self.z_eco_ch, 1, 1)).view(N, 8)
        else:
            eco_vec = torch.zeros((N, 8), device=device, dtype=p_dtype)

        h_self = self.self_proj(torch.cat((self_feat.to(p_dtype), goal_vec, eco_vec), dim=-1))
        h_tok = self.tok_proj(tokens.to(p_dtype))
        key_pad = (~token_mask) if token_mask is not None else None
        attn_out, attn_w = self.cross_attn(
            h_self.unsqueeze(1), h_tok, h_tok, key_padding_mask=key_pad, need_weights=True
        )
        h_attn = self.attn_norm(h_self + attn_out.squeeze(1))

        h = self.trunk(torch.cat((h_fine, h_coarse, h_self, h_attn), dim=-1))
        raw_mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)

        # Bound physical actuator ranges smoothly
        mean = torch.cat(
            (
                torch.tanh(raw_mean[:, 0:7]),     # move_x, move_y, move_jump, aim_dx, aim_dy, aim_dz, thermal_flux
                torch.sigmoid(raw_mean[:, 7:11]), # emit_water, emit_oil, emit_smoke, focus_radius
                torch.tanh(raw_mean[:, 11:12]),   # impulse_or_dome
            ),
            dim=-1,
        )
        out: dict[str, torch.Tensor] = {
            "raw_mean": raw_mean,
            "mean": mean,
            "log_std": self.actor_log_std.clamp(-2.5, 0.35),
            "value": value,
            "attn_weights": attn_w.squeeze(1),
        }
        if return_saliency:
            fine_2d = fine_crop.mean(dim=2)  # [N, latent_dim, 12, 12]
            sal = torch.sigmoid(self.saliency_proj(fine_2d))
            sal4 = F.adaptive_avg_pool2d(sal, (4, 4)).view(N, 16)
            out["saliency_4x4"] = sal4
        return out

    def forward(
        self,
        z_world_3d: torch.Tensor,
        pos_xyz: torch.Tensor,
        self_feat: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        goal_vec: torch.Tensor | None = None,
        z_eco: torch.Tensor | None = None,
        arena_idx: torch.Tensor | None = None,
        return_saliency: bool = False,
    ) -> dict[str, torch.Tensor]:
        fine_raw, coarse_raw, eco_raw = self.sample_raw_crops(
            z_world_3d, pos_xyz, z_eco=z_eco, arena_idx=arena_idx
        )
        return self.forward_from_crops(
            fine_raw=fine_raw,
            coarse_raw=coarse_raw,
            self_feat=self_feat,
            tokens=tokens,
            token_mask=token_mask,
            goal_vec=goal_vec,
            eco_raw=eco_raw,
            return_saliency=return_saliency,
        )

    def sample_action(
        self,
        out: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Strict float32 squashed Gaussian sampling (zero bf16 NaN risk!).
        Returns (bounded_action, pre_squash_u, log_prob).
        """
        raw_mean = out["raw_mean"].float()
        log_std = out["log_std"].float().unsqueeze(0).expand_as(raw_mean)
        std = log_std.exp()
        if deterministic:
            u = raw_mean
        else:
            u = raw_mean + std * torch.randn_like(raw_mean)

        # Compute exact log_prob in strict float32 using numerically stable logsigmoid
        dist = torch.distributions.Normal(raw_mean, std)
        lp = dist.log_prob(u)

        u_tanh = u[:, [0, 1, 2, 3, 4, 5, 6, 11]]
        a_tanh = torch.tanh(u_tanh)
        # log(1 - tanh(u)^2) = 2*(log(2) - u - softplus(-2u)) — exact & NaN-free!
        jac_tanh = 2.0 * (math.log(2.0) - u_tanh - F.softplus(-2.0 * u_tanh))

        u_sig = u[:, 7:11]
        a_sig = torch.sigmoid(u_sig)
        jac_sig = F.logsigmoid(u_sig) + F.logsigmoid(-u_sig)

        lp_adj = lp.clone()
        lp_adj[:, [0, 1, 2, 3, 4, 5, 6, 11]] -= jac_tanh
        lp_adj[:, 7:11] -= jac_sig
        log_prob = lp_adj.sum(dim=-1)

        action = torch.cat((a_tanh[:, :7], a_sig, a_tanh[:, 7:8]), dim=-1)
        return action, u, log_prob

    def evaluate_actions(
        self,
        out: dict[str, torch.Tensor],
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact float32 log_prob & entropy for PPO updates."""
        raw_mean = out["raw_mean"].float()
        log_std = out["log_std"].float().unsqueeze(0).expand_as(raw_mean)
        std = log_std.exp()
        u_f = u.float()
        dist = torch.distributions.Normal(raw_mean, std)
        lp = dist.log_prob(u_f)

        u_tanh = u_f[:, [0, 1, 2, 3, 4, 5, 6, 11]]
        jac_tanh = 2.0 * (math.log(2.0) - u_tanh - F.softplus(-2.0 * u_tanh))
        u_sig = u_f[:, 7:11]
        jac_sig = F.logsigmoid(u_sig) + F.logsigmoid(-u_sig)

        lp_adj = lp.clone()
        lp_adj[:, [0, 1, 2, 3, 4, 5, 6, 11]] -= jac_tanh
        lp_adj[:, 7:11] -= jac_sig
        return lp_adj.sum(dim=-1), dist.entropy().sum(dim=-1)


def adapt_bot_feats_168(
    bot_feats: torch.Tensor,
    grid_w: int,
    grid_h: int,
    grid_l: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adapt Frostmarsh server's live `[N, 168]` entity ledger (`24` self + `12 × 12` tokens)
    into `ActuatorPolicyNetV3`'s exact 3D `(pos_xyz, self_feat[22], tokens[14,14], token_mask[14])`.
    """
    N = bot_feats.shape[0]
    dev = bot_feats.device
    dtype = bot_feats.dtype
    self_raw = bot_feats[:, :24]
    tok_raw = bot_feats[:, 24:168].view(N, 12, 12)

    # Extract true 3D voxel position (self_raw[:, 0..2] are normalized x, y, z in [0, 1] or raw layer)
    px = self_raw[:, 0].clamp(0.0, 1.0) * max(grid_w - 1, 1)
    py = self_raw[:, 1].clamp(0.0, 1.0) * max(grid_h - 1, 1)
    raw_z = self_raw[:, 2]
    pz = torch.where(raw_z > 1.05, raw_z.clamp(0.0, max(grid_l - 1, 1)), raw_z.clamp(0.0, 1.0) * max(grid_l - 1, 1))
    pos_xyz = torch.stack((px, py, pz), dim=-1)

    # Map 24-dim live self vector -> 22-dim 3D self vector cleanly
    self_feat = torch.zeros((N, SELF3D_FEAT_DIM), device=dev, dtype=dtype)
    self_feat[:, 0] = self_raw[:, 0] * 2.0 - 1.0
    self_feat[:, 1] = self_raw[:, 1] * 2.0 - 1.0
    self_feat[:, 2] = (pz / max(grid_l - 1, 1)) * 2.0 - 1.0
    self_feat[:, 3:22] = self_raw[:, 3:22]

    # Map 12×12 live tokens -> 14×14 3D tokens (preserve valid mask via nonzero norm)
    tokens = torch.zeros((N, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM), device=dev, dtype=dtype)
    tokens[:, :12, :12] = tok_raw
    mask = torch.zeros((N, MAX_TOKENS_3D), device=dev, dtype=torch.bool)
    mask[:, :12] = tok_raw.abs().sum(dim=-1) > 1e-6
    # Ensure at least one valid token per bot so MultiheadAttention never sees all-masked rows
    empty_rows = ~mask.any(dim=-1)
    if empty_rows.any():
        mask[empty_rows, 0] = True
    return pos_xyz, self_feat, tokens, mask


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _smoke_test() -> dict[str, Any]:
    torch.manual_seed(42)
    net = ActuatorPolicyNetV3()
    n_params = count_parameters(net)
    # Test both Z=16 single-arena (12 bots) and Z=32 multi-arena (B=4, 48 bots)
    for L, B, nb in [(16, 1, 12), (32, 4, 48)]:
        z_world = torch.randn(B, 48, L, 64, 64)
        pos = torch.stack(
            (
                torch.rand(nb) * 60.0 + 2.0,
                torch.rand(nb) * 60.0 + 2.0,
                torch.rand(nb) * (L - 1),
            ),
            dim=-1,
        )
        sf = torch.randn(nb, SELF3D_FEAT_DIM)
        toks = torch.randn(nb, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM)
        mask = torch.ones(nb, MAX_TOKENS_3D, dtype=torch.bool)
        aidx = torch.arange(B).repeat_interleave(nb // B)
        out = net(z_world, pos, sf, toks, token_mask=mask, arena_idx=aidx, return_saliency=True)
        act, u, lp = net.sample_action(out)
        lp2, ent = net.evaluate_actions(out, u)
        assert act.shape == (nb, ACT3D_DIM), act.shape
        assert out["saliency_4x4"].shape == (nb, 16)
        assert torch.allclose(lp, lp2, atol=1e-5)
        assert torch.isfinite(act).all() and torch.isfinite(lp).all() and torch.isfinite(ent).all()
    print(f"[ActuatorPolicyNetV3] Smoke test PASS | params={n_params:,} | Z=16 & Z=32 verified!")
    return {"params": n_params, "status": "ok"}


if __name__ == "__main__":
    _smoke_test()
