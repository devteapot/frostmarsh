"""
Frostmarsh Shared-Latent Neural Agent Policy (`BotPolicyNet`).

Interacts with the authoritative neural world by reading ZERO-COPY directly from
WorldModel's internal [L, 48, H, W] spatial latent trunk (Z_world) and
EcosystemModel's [1, 32, H, W] trunk (Z_eco) already resident in GB10 VRAM.

2-Stream Tailored Architecture (~115k params):
  - Stream A (Shared World-Latent Ego-Crop):
      F.grid_sample(Z_world[layer], 16x16 ego patch) + F.grid_sample(Z_eco, 16x16 patch)
      -> 1x1 Version-Pinning Latent Adapter (80 -> 32)
      -> 3-layer strided Conv encoder -> 64-d physics-intuition embedding
  - Stream B (Entity & Objective Cross-Attention):
      Self ledger vector (24-d) queries 12 entity/shrine/relic tokens (12-d each)
      via 4-head Multi-Head Cross-Attention -> 64-d tactical embedding
  - Joint Action Head:
      Outputs continuous (move_x, move_y, aim_x, aim_y), 6 binary ability logits
      (dash, melee, cast_bolt, carve_ice, ability_e, ult_r), and 3-class element logits
      (Fire, Frost, Oil) for ALL N bots in a single batched GPU pass (<0.5 ms on GB10).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SELF_DIM = 24
NUM_TOKENS = 12  # 8 nearest players + 3 shrines + 1 golden relic
TOKEN_DIM = 12
BOT_FEAT_DIM = SELF_DIM + NUM_TOKENS * TOKEN_DIM  # 168 floats per bot
ACTION_OUT_DIM = 13  # move_x, move_y, aim_x, aim_y, 6 ability probs, 3 element probs
SALIENCY_DIM = 16  # 4x4 spatial saliency heatmap from spatial_enc layer 2
INSPECT_OUT_DIM = ACTION_OUT_DIM + SALIENCY_DIM  # 29 floats per bot


class BotPolicyNet(nn.Module):
    def __init__(
        self,
        z_world_ch: int = 48,
        z_eco_ch: int = 32,
        patch_size: int = 16,
        patch_radius_cells: float = 14.0,
        d_model: int = 64,
        blind_world: bool = False,
    ):
        super().__init__()
        self.z_world_ch = z_world_ch
        self.z_eco_ch = z_eco_ch
        self.patch_size = patch_size
        self.patch_radius_cells = patch_radius_cells
        self.blind_world = blind_world

        lin = torch.linspace(-1.0, 1.0, patch_size)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("base_gx", gx.unsqueeze(0).contiguous().clone())  # [1, P, P]
        self.register_buffer("base_gy", gy.unsqueeze(0).contiguous().clone())  # [1, P, P]

        # Stream A: 1x1 Version-Pinning Latent Adapter (re-fittable in seconds if WorldModel upgrades)
        self.latent_adapter = nn.Sequential(
            nn.Conv2d(z_world_ch + z_eco_ch, 32, kernel_size=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),      # 16 -> 8
            nn.GELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),      # 8 -> 4
            nn.GELU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1), # 4 -> 2
            nn.GELU(),
        )
        self.spatial_proj = nn.Linear(d_model * 2, d_model)

        # Stream B: Self State + Entity / Shrine / Relic Cross-Attention
        self.self_enc = nn.Sequential(
            nn.Linear(SELF_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.tok_enc = nn.Sequential(
            nn.Linear(TOKEN_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # Joint Policy Trunk & Action Heads
        self.trunk = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.head_vec = nn.Linear(128, 4)   # move_x, move_y, aim_x, aim_y (tanh)
        self.head_bin = nn.Linear(128, 6)   # dash, melee, cast_bolt, carve_ice, ability_e, ult_r
        self.head_elem = nn.Linear(128, 3)  # Fire, Frost, Oil logits

    def sample_shared_latents(
        self,
        z_world: torch.Tensor,    # [L, 48, H, W]
        z_eco: torch.Tensor,      # [1, 32, H, W]
        self_feat: torch.Tensor,  # [N, SELF_DIM]: [0]=x/W, [1]=y/H, [2]=layer_idx/(L-1)
    ) -> torch.Tensor:
        n = self_feat.shape[0]
        if self.blind_world:
            return torch.zeros(
                (n, self.z_world_ch + self.z_eco_ch, self.patch_size, self.patch_size),
                device=self_feat.device,
                dtype=torch.float32,
            )

        num_l, _, h, w = z_world.shape
        cx = self_feat[:, 0:1, None] * 2.0 - 1.0  # [N, 1, 1]
        cy = self_feat[:, 1:2, None] * 2.0 - 1.0
        rx = (self.patch_radius_cells / max(w - 1, 1)) * 2.0
        ry = (self.patch_radius_cells / max(h - 1, 1)) * 2.0

        grid_x = cx + self.base_gx * rx
        grid_y = cy + self.base_gy * ry
        grid_2d = torch.stack([grid_x, grid_y], dim=-1)  # [N, P, P, 2]

        eco_rep = z_eco.expand(n, -1, -1, -1)
        eco_patch = F.grid_sample(
            eco_rep.float(), grid_2d.float(), mode="bilinear", padding_mode="border", align_corners=True
        )

        # Sample Z_world at each bot's exact 3D vertical layer via a single 5D grid_sample
        z_vol = z_world.permute(1, 0, 2, 3).unsqueeze(0).float()  # [1, 48, L, H, W]
        if num_l > 1:
            cz = (self_feat[:, 2:3, None] * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            cz = torch.zeros_like(cx)
        grid_3d = torch.stack(
            [grid_x, grid_y, cz.expand(-1, self.patch_size, self.patch_size)], dim=-1
        ).unsqueeze(0)  # [1, N, P, P, 3]
        w_patch = F.grid_sample(
            z_vol, grid_3d.float(), mode="bilinear", padding_mode="border", align_corners=True
        ).squeeze(0).permute(1, 0, 2, 3)  # [N, 48, P, P]

        return torch.cat([w_patch, eco_patch], dim=1)

    def _forward_core(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        return_saliency: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self_feat = bot_feats[:, :SELF_DIM]
        tok_feat = bot_feats[:, SELF_DIM:].reshape(-1, NUM_TOKENS, TOKEN_DIM)

        patch = self.sample_shared_latents(z_world, z_eco, self_feat)
        adapted = self.latent_adapter(patch)
        # First 2 strided conv blocks (Conv1->GELU->Conv2->GELU) yield the [N, 64, 4, 4] feature map
        feat_4x4 = self.spatial_enc[:4](adapted)
        sp = self.spatial_enc[4:](feat_4x4)
        h_world = self.spatial_proj(torch.cat([sp.mean(dim=(2, 3)), sp.amax(dim=(2, 3))], dim=-1))

        saliency: torch.Tensor | None = None
        if return_saliency:
            # Zero-cost 4x4 spatial saliency: L2 norm across channels, per-bot min-max normalized to [0, 1]
            sal_map = torch.linalg.vector_norm(feat_4x4.float(), ord=2, dim=1)  # [N, 4, 4]
            if sal_map.shape[-2:] != (4, 4):
                sal_map = F.adaptive_avg_pool2d(sal_map.unsqueeze(1), (4, 4)).squeeze(1)
            sal_flat = sal_map.reshape(sal_map.shape[0], SALIENCY_DIM)
            sal_min = sal_flat.amin(dim=-1, keepdim=True)
            sal_max = sal_flat.amax(dim=-1, keepdim=True)
            saliency = ((sal_flat - sal_min) / (sal_max - sal_min + 1e-6)).clamp(0.0, 1.0)

        q = self.self_enc(self_feat).unsqueeze(1)
        kv = self.tok_enc(tok_feat)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h_tac = self.attn_norm((q + attn_out).squeeze(1))

        h = self.trunk(torch.cat([h_world, h_tac], dim=-1))
        vec = torch.tanh(self.head_vec(h))
        bin_logits = self.head_bin(h)
        elem_logits = self.head_elem(h)
        return vec, bin_logits, elem_logits, saliency

    def forward_logits(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vec, bin_logits, elem_logits, _ = self._forward_core(
            z_world, z_eco, bot_feats, return_saliency=False
        )
        return vec, bin_logits, elem_logits

    def forward(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        return_saliency: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        vec, bin_logits, elem_logits, saliency = self._forward_core(
            z_world, z_eco, bot_feats, return_saliency=return_saliency
        )
        actions = torch.cat([vec, torch.sigmoid(bin_logits), F.softmax(elem_logits, dim=-1)], dim=-1)
        if return_saliency:
            assert saliency is not None
            return actions, saliency
        return actions


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
