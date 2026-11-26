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
BRIEF_DIM = 16  # 15 conditioning floats + 1 daemon/arena bot_slot_key (masked from net)
BRIEF_MODEL_DIM = 15
BOT_FEAT_DIM_V3 = BOT_FEAT_DIM + BRIEF_DIM  # 184 floats per bot
GRU_HIDDEN_DIM = 64
NUM_ROLES = 5  # 0=Pyromancer, 1=Frostweaver, 2=Alchemist, 3=Emberstalker, 4=Burrower
ACTION_OUT_DIM = 13  # move_x, move_y, aim_x, aim_y, 6 ability probs, 3 element probs
SALIENCY_DIM = 16  # 4x4 spatial saliency heatmap
INSPECT_OUT_DIM = ACTION_OUT_DIM + SALIENCY_DIM  # 29 floats per bot
CANONICAL_WORLD_SPAN = 255.0  # Matches 256x256 grid span so 128x128 & 256x256 share identical physical radius

OBJ_FREE_ROAM = 0
OBJ_CAPTURE_SHRINE = 1
OBJ_ESCORT_RELIC = 2
OBJ_HUNT_ENEMY = 3
OBJ_DEFEND_HOLD = 4

STANCE_COMMIT = 0
STANCE_HOLD = 1
STANCE_FLEE_KITE = 2


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
        use_gru: bool = True,
        gru_hidden_dim: int = GRU_HIDDEN_DIM,
        use_brief: bool = True,
        num_roles: int = NUM_ROLES,
    ):
        super().__init__()
        self.z_world_ch = z_world_ch
        self.z_eco_ch = z_eco_ch
        self.patch_size = patch_size
        self.fine_radius_cells = fine_radius_cells
        self.coarse_radius_cells = coarse_radius_cells
        self.patch_radius_cells = fine_radius_cells  # backward-compat alias
        self.blind_world = blind_world
        self.use_gru = bool(use_gru)
        self.gru_hidden_dim = int(gru_hidden_dim)
        self.use_brief = bool(use_brief)
        self.num_roles = int(num_roles)

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

        # NMMO 2.0-style Task Brief + 5-Species Role Conditioning (zero-init residual projs for clean v2 warm-start)
        if self.use_brief:
            self.role_emb = nn.Embedding(self.num_roles, 16)
            self.role_proj = nn.Linear(16, 128)
            nn.init.zeros_(self.role_proj.weight)
            nn.init.zeros_(self.role_proj.bias)
            self.brief_enc = nn.Sequential(
                nn.Linear(BRIEF_MODEL_DIM + 16, 48),
                nn.LayerNorm(48),
                nn.GELU(),
                nn.Linear(48, 32),
                nn.GELU(),
            )
            self.brief_proj = nn.Linear(32, 128)
            nn.init.zeros_(self.brief_proj.weight)
            nn.init.zeros_(self.brief_proj.bias)
        else:
            self.role_emb = None
            self.role_proj = None
            self.brief_enc = None
            self.brief_proj = None

        # Lightweight 64-unit GRU residual recurrence for target/hazard object permanence
        if self.use_gru:
            self.gru_cell = nn.GRUCell(128, self.gru_hidden_dim)
            self.gru_out_proj = nn.Linear(self.gru_hidden_dim, 128)
            nn.init.normal_(self.gru_out_proj.weight, std=0.02)
            nn.init.zeros_(self.gru_out_proj.bias)
        else:
            self.gru_cell = None
            self.gru_out_proj = None

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

    def _extract_brief_and_role(
        self, bot_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract 15-float conditioning brief (masking slot 15 bot_slot_key) and integer role_idx in 0..4."""
        self_feat = bot_feats[:, :SELF_DIM]
        if bot_feats.shape[-1] >= BOT_FEAT_DIM_V3:
            brief_15 = bot_feats[:, BOT_FEAT_DIM : BOT_FEAT_DIM + BRIEF_MODEL_DIM].float()
            role_idx = torch.round(brief_15[:, 12] * float(self.num_roles - 1)).long().clamp(0, self.num_roles - 1)
        else:
            brief_15 = torch.zeros(
                (bot_feats.shape[0], BRIEF_MODEL_DIM), device=bot_feats.device, dtype=torch.float32
            )
            base_cls = torch.argmax(self_feat[:, 8:11], dim=-1)
            role_idx = torch.where(
                self_feat[:, 16] > 0.5,
                torch.full_like(base_cls, 4),
                torch.where(self_feat[:, 14] > 0.5, torch.full_like(base_cls, 3), base_cls),
            ).clamp(0, self.num_roles - 1)
            brief_15[:, 12] = role_idx.float() / float(self.num_roles - 1)
        return brief_15, role_idx

    def forward_from_patches(
        self,
        fine_patch: torch.Tensor,    # [M, 80, P, P]
        coarse_patch: torch.Tensor,  # [M, 48, P, P]
        bot_feats: torch.Tensor,     # [M, 168] or [M, 184]
        return_saliency: bool = False,
        h_prev: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        """Core forward pass from pre-sampled patches; returns 5-tuple (or 6-tuple with h_next when return_hidden=True)."""
        self_feat = bot_feats[:, :SELF_DIM]
        tok_feat = bot_feats[:, SELF_DIM:BOT_FEAT_DIM].reshape(-1, NUM_TOKENS, TOKEN_DIM)

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

        h_base = self.trunk(torch.cat([h_fine, h_coarse, h_tac], dim=-1))

        if self.use_brief and self.brief_enc is not None and self.role_emb is not None and self.brief_proj is not None:
            brief_15, role_idx = self._extract_brief_and_role(bot_feats)
            r_emb = self.role_emb(role_idx).to(h_base.dtype)
            if self.role_proj is not None:
                h_base = h_base + self.role_proj(r_emb)
            # Explicitly gate brief residual by brief_active (slot 13) so free-play combat has zero brief interference!
            brief_gate = brief_15[:, 13:14].clamp(0.0, 1.0).to(h_base.dtype)
            h_brief = self.brief_enc(torch.cat([brief_15.to(h_base.dtype), r_emb], dim=-1))
            h_base = h_base + brief_gate * self.brief_proj(h_brief)

        h_next: torch.Tensor | None = None
        if self.use_gru and self.gru_cell is not None and self.gru_out_proj is not None:
            if h_prev is None:
                h_cur = torch.zeros(
                    (h_base.shape[0], self.gru_hidden_dim), device=h_base.device, dtype=torch.float32
                )
            else:
                h_cur = h_prev.float()
            if reset_mask is not None:
                keep = (1.0 - reset_mask.float()).unsqueeze(-1)
                h_cur = h_cur * keep
            h_next = self.gru_cell(h_base.float(), h_cur)
            h_trunk = h_base + self.gru_out_proj(h_next.to(h_base.dtype))
        else:
            h_next = h_prev
            h_trunk = h_base

        vec = torch.tanh(self.head_vec(h_trunk))
        bin_logits = self.head_bin(h_trunk)
        elem_logits = self.head_elem(h_trunk)
        if return_hidden:
            return vec, bin_logits, elem_logits, h_trunk, saliency, h_next
        return vec, bin_logits, elem_logits, h_trunk, saliency

    def forward_sequence_from_patches(
        self,
        fine_seq: torch.Tensor,    # [T, B, 80, P, P]
        coarse_seq: torch.Tensor,  # [T, B, 48, P, P]
        feat_seq: torch.Tensor,    # [T, B, 168 or 184]
        done_seq: torch.Tensor,    # [T, B] float/bool (1 where prev step ended -> reset before step t)
        h0: torch.Tensor | None = None,  # [B, gru_hidden_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Truncated-BPTT sequence forward pass for recurrent PPO over [T, B, ...]."""
        t_steps, batch_sz = feat_seq.shape[0], feat_seq.shape[1]
        h_cur = (
            h0.float()
            if h0 is not None
            else torch.zeros((batch_sz, self.gru_hidden_dim), device=feat_seq.device, dtype=torch.float32)
        )
        vecs, bins, elems, trunks = [], [], [], []
        for t in range(t_steps):
            reset_t = done_seq[t] if t > 0 else None
            v, bl, el, ht, _, h_cur = self.forward_from_patches(
                fine_seq[t],
                coarse_seq[t],
                feat_seq[t],
                return_saliency=False,
                h_prev=h_cur,
                reset_mask=reset_t,
                return_hidden=True,
            )
            vecs.append(v)
            bins.append(bl)
            elems.append(el)
            trunks.append(ht)
        return (
            torch.stack(vecs, dim=0),
            torch.stack(bins, dim=0),
            torch.stack(elems, dim=0),
            torch.stack(trunks, dim=0),
            h_cur,
        )

    def forward_logits(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fine_p, coarse_p = self.sample_shared_latents(z_world, z_eco, bot_feats[:, :SELF_DIM])
        vec, bin_logits, elem_logits, _, _ = self.forward_from_patches(
            fine_p, coarse_p, bot_feats, return_saliency=False, h_prev=h_prev, reset_mask=reset_mask
        )
        return vec, bin_logits, elem_logits

    def forward(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        return_saliency: bool = False,
        h_prev: torch.Tensor | None = None,
        reset_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        fine_p, coarse_p = self.sample_shared_latents(z_world, z_eco, bot_feats[:, :SELF_DIM])
        vec, bin_logits, elem_logits, _, saliency, h_next = self.forward_from_patches(
            fine_p,
            coarse_p,
            bot_feats,
            return_saliency=return_saliency,
            h_prev=h_prev,
            reset_mask=reset_mask,
            return_hidden=True,
        )
        actions = torch.cat([vec, torch.sigmoid(bin_logits), F.softmax(elem_logits, dim=-1)], dim=-1)
        if return_saliency and return_hidden:
            assert saliency is not None
            return actions, saliency, h_next
        if return_saliency:
            assert saliency is not None
            return actions, saliency
        if return_hidden:
            return actions, h_next
        return actions


def load_bot_policy_v2_or_v3(
    ckpt_or_sd,
    device: torch.device | str = "cpu",
    upgrade_to_v3: bool = False,
) -> BotPolicyNetV2:
    """Load either a legacy feedforward BotPolicyNetV2 checkpoint or a recurrent+brief V3 checkpoint."""
    if isinstance(ckpt_or_sd, dict):
        sd = ckpt_or_sd["state_dict"] if "state_dict" in ckpt_or_sd else ckpt_or_sd
    else:
        raw = torch.load(ckpt_or_sd, map_location=device, weights_only=False)
        sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    has_gru = any(k.startswith("gru_cell.") for k in sd)
    has_brief = any(k.startswith("brief_enc.") for k in sd)
    use_gru = has_gru or upgrade_to_v3
    use_brief = has_brief or upgrade_to_v3
    model = BotPolicyNetV2(use_gru=use_gru, use_brief=use_brief)
    model.load_state_dict(sd, strict=not (upgrade_to_v3 and (not has_gru or not has_brief)))
    return model.to(device)


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
