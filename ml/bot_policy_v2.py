"""
Frostmarsh Dual-Scale Shared-Latent Neural Agent Policy (`BotPolicyNetV2`).

Drop-in upgrade over `BotPolicyNet` (identical 168-float input, 13-float action output,
and optional `return_saliency=True` 4x4 saliency heatmap API).

Dual-Scale Z_world + Z_eco Architecture (~210K params, <1.0 ms for 12 bots on GB10):
  - Stream A1 (Fine Z_world + Z_eco Ego-Crop, radius = 14 cells):
      16x16 bilinear ego-patch around the bot's exact (x, y, layer) in Z_world [L, 48, H, W]
      concatenated with 16x16 Z_eco [1, 32, H, W] -> 1x1 Latent Adapter (80 -> 32)
      -> 3-layer strided Conv encoder (16 -> 8 -> 4 -> 2) -> 64-d fine hazard/terrain embedding.
      Handles immediate wildfire evasion, ice bridges, local oil pools, rock walls, critter sanctuaries.
  - Stream A2 (Coarse Z_world Tactical-Sight Crop, radius = 60 cells):
      16x16 bilinear patch at radius = 60 canonical cells matching Scripted Teacher's 60-cell
      tactical sight -> 1x1 Latent Adapter (48 -> 24)
      -> 3-layer strided Conv encoder (16 -> 8 -> 4 -> 2) -> 64-d coarse tactical terrain embedding.
      Sees distant oil slicks under enemies, frozen/wet river crossings, enemy flanks, shrine hazards!
  - Stream B (Entity / Shrine / Relic Cross-Attention):
      Self ledger (24-d) queries 12 tokens (8 nearest players + 3 shrines + 1 golden relic, 12-d each)
      via 4-head Multi-Head Cross-Attention -> 64-d tactical ledger embedding.
  - Joint Policy Trunk (192 -> 128 -> 128) & Action Heads:
      Outputs continuous (move_x, move_y, aim_x, aim_y), 6 binary ability probs
      (dash, melee, cast_bolt, carve_ice, ability_e, ult_r), and 3-class element probs
      (Fire, Frost, Oil).
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
SALIENCY_DIM = 16  # 4x4 spatial saliency heatmap
INSPECT_OUT_DIM = ACTION_OUT_DIM + SALIENCY_DIM  # 29 floats per bot
CANONICAL_WORLD_SPAN = 255.0  # Matches 256x256 grid span so 128x128 & 256x256 share identical physical radius


class BotPolicyNetV2(nn.Module):
    def __init__(
        self,
        z_world_ch: int = 48,
        z_eco_ch: int = 32,
        patch_size: int = 16,
        fine_radius_cells: float = 14.0,
        coarse_radius_cells: float = 60.0,
        d_model: int = 64,
        blind_world: bool = False,
    ):
        super().__init__()
        self.z_world_ch = z_world_ch
        self.z_eco_ch = z_eco_ch
        self.patch_size = patch_size
        self.fine_radius_cells = fine_radius_cells
        self.coarse_radius_cells = coarse_radius_cells
        self.patch_radius_cells = fine_radius_cells  # backward-compat alias
        self.blind_world = blind_world

        lin = torch.linspace(-1.0, 1.0, patch_size)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("base_gx", gx.unsqueeze(0).contiguous().clone())  # [1, P, P]
        self.register_buffer("base_gy", gy.unsqueeze(0).contiguous().clone())  # [1, P, P]

        # Stream A1: Fine Z_world (48) + Z_eco (32) at radius=14 cells
        self.fine_adapter = nn.Sequential(
            nn.Conv2d(z_world_ch + z_eco_ch, 32, kernel_size=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
        )
        self.fine_enc = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),      # 16 -> 8
            nn.GELU(),
            nn.Conv2d(48, 60, kernel_size=3, stride=2, padding=1),      # 8 -> 4
            nn.GELU(),
            nn.Conv2d(60, d_model, kernel_size=3, stride=2, padding=1), # 4 -> 2
            nn.GELU(),
        )
        self.fine_proj = nn.Linear(d_model * 2, d_model)

        # Stream A2: Coarse Z_world (48) at radius=60 cells (matching Teacher's 60-cell tactical sight!)
        self.coarse_adapter = nn.Sequential(
            nn.Conv2d(z_world_ch, 24, kernel_size=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.coarse_enc = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),      # 16 -> 8
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),      # 8 -> 4
            nn.GELU(),
            nn.Conv2d(48, d_model, kernel_size=3, stride=2, padding=1), # 4 -> 2
            nn.GELU(),
        )
        self.coarse_proj = nn.Linear(d_model * 2, d_model)

        # Stream B: Self State + 4-head Entity / Shrine / Relic Cross-Attention
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
        # Learned slot/type embedding so cross-attention cleanly separates Players (0..7), Shrines (8..10), Relic (11)
        self.tok_type_emb = nn.Parameter(torch.randn(1, NUM_TOKENS, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # Joint Policy Trunk (Fine 64 + Coarse 64 + Cross-Attn 64 = 192 -> 128 -> 128)
        self.trunk = nn.Sequential(
            nn.Linear(d_model * 3, 128),
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
        self_feat: torch.Tensor,  # [N, SELF_DIM]: [0]=x/256, [1]=y/256, [2]=layer_idx/(L-1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-copy dual-scale sampling from resident VRAM Z_world & Z_eco.
        Returns:
          fine_patch:   [N, 80, P, P] (Fine Z_world @ r=14 + Z_eco @ r=14)
          coarse_patch: [N, 48, P, P] (Coarse Z_world @ r=60)
        """
        n = self_feat.shape[0]
        p = self.patch_size
        if self.blind_world:
            z_f = torch.zeros((n, self.z_world_ch + self.z_eco_ch, p, p), device=self_feat.device, dtype=torch.float32)
            z_c = torch.zeros((n, self.z_world_ch, p, p), device=self_feat.device, dtype=torch.float32)
            return z_f, z_c

        num_l = z_world.shape[0]
        cx = self_feat[:, 0:1, None] * 2.0 - 1.0  # [N, 1, 1]
        cy = self_feat[:, 1:2, None] * 2.0 - 1.0

        # Canonical-scale radii so 128x128 training & 256x256 inference cover exact same physical distance
        r_fine = (self.fine_radius_cells / CANONICAL_WORLD_SPAN) * 2.0
        r_coarse = (self.coarse_radius_cells / CANONICAL_WORLD_SPAN) * 2.0

        gx_f = cx + self.base_gx * r_fine
        gy_f = cy + self.base_gy * r_fine
        gx_c = cx + self.base_gx * r_coarse
        gy_c = cy + self.base_gy * r_coarse

        # Sample Z_eco on fine 14-cell grid directly in z_eco.dtype (zero full-grid fp32 copy!)
        grid_eco = torch.stack([gx_f, gy_f], dim=-1).reshape(1, n * p, p, 2).to(z_eco.dtype)
        eco_patch = (
            F.grid_sample(z_eco, grid_eco, mode="bilinear", padding_mode="border", align_corners=True)
            .reshape(1, self.z_eco_ch, n, p, p)
            .squeeze(0)
            .permute(1, 0, 2, 3)
            .float()
        )  # [N, 32, P, P]

        # Single fused 5D grid_sample directly in z_world.dtype for BOTH fine & coarse crops (zero 50MB copy!)
        z_vol = z_world.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 48, L, H, W]
        if num_l > 1:
            cz = (self_feat[:, 2:3, None] * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            cz = torch.zeros_like(cx)
        cz_p = cz.expand(-1, p, p)

        gx_both = torch.cat([gx_f, gx_c], dim=1)  # [N, 2*P, P]
        gy_both = torch.cat([gy_f, gy_c], dim=1)  # [N, 2*P, P]
        cz_both = torch.cat([cz_p, cz_p], dim=1)  # [N, 2*P, P]
        grid_3d = torch.stack([gx_both, gy_both, cz_both], dim=-1).unsqueeze(0).to(z_vol.dtype)  # [1, N, 2*P, P, 3]

        w_both = (
            F.grid_sample(z_vol, grid_3d, mode="bilinear", padding_mode="border", align_corners=True)
            .squeeze(0)
            .permute(1, 0, 2, 3)
            .float()
        )  # [N, 48, 2*P, P]
        w_fine = w_both[:, :, :p, :]
        w_coarse = w_both[:, :, p:, :]

        fine_patch = torch.cat([w_fine, eco_patch], dim=1)
        return fine_patch, w_coarse

    def forward_from_patches(
        self,
        fine_patch: torch.Tensor,    # [M, 80, P, P]
        coarse_patch: torch.Tensor,  # [M, 48, P, P]
        bot_feats: torch.Tensor,     # [M, 168]
        return_saliency: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Core forward pass from pre-sampled patches; returns (vec, bin_logits, elem_logits, h_trunk, saliency)."""
        self_feat = bot_feats[:, :SELF_DIM]
        tok_feat = bot_feats[:, SELF_DIM:].reshape(-1, NUM_TOKENS, TOKEN_DIM)

        # Fine stream
        f_adapt = self.fine_adapter(fine_patch.float())
        f_4x4 = self.fine_enc[:4](f_adapt)
        f_sp = self.fine_enc[4:](f_4x4)
        h_fine = self.fine_proj(torch.cat([f_sp.mean(dim=(2, 3)), f_sp.amax(dim=(2, 3))], dim=-1))

        # Coarse stream
        c_adapt = self.coarse_adapter(coarse_patch.float())
        c_4x4 = self.coarse_enc[:4](c_adapt)
        c_sp = self.coarse_enc[4:](c_4x4)
        h_coarse = self.coarse_proj(torch.cat([c_sp.mean(dim=(2, 3)), c_sp.amax(dim=(2, 3))], dim=-1))

        saliency: torch.Tensor | None = None
        if return_saliency:
            # Dual-scale 4x4 spatial saliency combining fine local hazard & coarse tactical energy
            sal_f = torch.linalg.vector_norm(f_4x4.float(), ord=2, dim=1)  # [M, 4, 4]
            sal_c = torch.linalg.vector_norm(c_4x4.float(), ord=2, dim=1)  # [M, 4, 4]
            sal_map = sal_f + 0.35 * sal_c
            if sal_map.shape[-2:] != (4, 4):
                sal_map = F.adaptive_avg_pool2d(sal_map.unsqueeze(1), (4, 4)).squeeze(1)
            sal_flat = sal_map.reshape(sal_map.shape[0], SALIENCY_DIM)
            sal_min = sal_flat.amin(dim=-1, keepdim=True)
            sal_max = sal_flat.amax(dim=-1, keepdim=True)
            saliency = ((sal_flat - sal_min) / (sal_max - sal_min + 1e-6)).clamp(0.0, 1.0)

        # Entity / Shrine / Relic 4-head Cross-Attention with slot-type embeddings
        q = self.self_enc(self_feat).unsqueeze(1)
        kv = self.tok_enc(tok_feat) + self.tok_type_emb
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h_tac = self.attn_norm((q + attn_out).squeeze(1))

        h_trunk = self.trunk(torch.cat([h_fine, h_coarse, h_tac], dim=-1))
        vec = torch.tanh(self.head_vec(h_trunk))
        bin_logits = self.head_bin(h_trunk)
        elem_logits = self.head_elem(h_trunk)
        return vec, bin_logits, elem_logits, h_trunk, saliency

    def forward_logits(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine_p, coarse_p = self.sample_shared_latents(z_world, z_eco, bot_feats[:, :SELF_DIM])
        vec, bin_logits, elem_logits, _, _ = self.forward_from_patches(
            fine_p, coarse_p, bot_feats, return_saliency=False
        )
        return vec, bin_logits, elem_logits

    def forward(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        return_saliency: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        fine_p, coarse_p = self.sample_shared_latents(z_world, z_eco, bot_feats[:, :SELF_DIM])
        vec, bin_logits, elem_logits, _, saliency = self.forward_from_patches(
            fine_p, coarse_p, bot_feats, return_saliency=return_saliency
        )
        actions = torch.cat([vec, torch.sigmoid(bin_logits), F.softmax(elem_logits, dim=-1)], dim=-1)
        if return_saliency:
            assert saliency is not None
            return actions, saliency
        return actions


def sample_batched_dual_patches(
    policy: BotPolicyNetV2,
    z_world: torch.Tensor,    # [A, L, 48, H, W]
    z_eco: torch.Tensor,      # [A, 1, 32, H, W]
    bot_feats: torch.Tensor,  # [A, N, 168]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-arena zero-copy dual-scale sampling matching BotPolicyNetV2.sample_shared_latents exactly.
    Returns:
      fine_patches:   [A, N, 80, P, P]
      coarse_patches: [A, N, 48, P, P]
    """
    a, n, _ = bot_feats.shape
    p = policy.patch_size
    if policy.blind_world:
        z_f = torch.zeros((a, n, policy.z_world_ch + policy.z_eco_ch, p, p), device=bot_feats.device, dtype=torch.float32)
        z_c = torch.zeros((a, n, policy.z_world_ch, p, p), device=bot_feats.device, dtype=torch.float32)
        return z_f, z_c

    num_l = z_world.shape[1]
    sf = bot_feats[:, :, :SELF_DIM]
    cx = sf[:, :, 0:1, None] * 2.0 - 1.0  # [A, N, 1, 1]
    cy = sf[:, :, 1:2, None] * 2.0 - 1.0

    r_fine = (policy.fine_radius_cells / CANONICAL_WORLD_SPAN) * 2.0
    r_coarse = (policy.coarse_radius_cells / CANONICAL_WORLD_SPAN) * 2.0

    bgx = policy.base_gx.unsqueeze(0)  # [1, 1, P, P]
    bgy = policy.base_gy.unsqueeze(0)
    gx_f = cx + bgx * r_fine
    gy_f = cy + bgy * r_fine
    gx_c = cx + bgx * r_coarse
    gy_c = cy + bgy * r_coarse

    grid_eco = torch.stack([gx_f, gy_f], dim=-1).reshape(a, n * p, p, 2).to(z_eco.dtype)
    eco_patch = (
        F.grid_sample(z_eco[:, 0], grid_eco, mode="bilinear", padding_mode="border", align_corners=True)
        .reshape(a, policy.z_eco_ch, n, p, p)
        .permute(0, 2, 1, 3, 4)
        .float()
    )

    z_vol = z_world.permute(0, 2, 1, 3, 4)  # [A, 48, L, H, W]
    if num_l > 1:
        cz = (sf[:, :, 2:3, None] * 2.0 - 1.0).clamp(-1.0, 1.0)
    else:
        cz = torch.zeros_like(cx)
    cz_p = cz.expand(-1, -1, p, p)

    gx_both = torch.cat([gx_f, gx_c], dim=2)  # [A, N, 2*P, P]
    gy_both = torch.cat([gy_f, gy_c], dim=2)
    cz_both = torch.cat([cz_p, cz_p], dim=2)
    grid_3d = torch.stack([gx_both, gy_both, cz_both], dim=-1).to(z_vol.dtype)  # [A, N, 2*P, P, 3]

    w_both = (
        F.grid_sample(z_vol, grid_3d, mode="bilinear", padding_mode="border", align_corners=True)
        .permute(0, 2, 1, 3, 4)
        .float()
    )  # [A, N, 48, 2*P, P]
    w_fine = w_both[:, :, :, :p, :]
    w_coarse = w_both[:, :, :, p:, :]

    fine_patch = torch.cat([w_fine, eco_patch], dim=2)
    return fine_patch, w_coarse


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
