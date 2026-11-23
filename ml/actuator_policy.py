"""
Frostmarsh Zero-Hardcoded-Rules Emergent Physical Actuator Policy (`ActuatorPolicyNet`).

Drop-in compatible with `gpu_infer_daemon.py` (`op=7`):
  - Input:
      z_world:   [L, 48, H, W] resident VRAM WorldModel v3 spatial latent
      z_eco:     [1, 32, H, W] resident VRAM EcosystemModel spatial latent
      bot_feats: [N, 168]      exact Frostmarsh entity / shrine / relic ledger
  - Output:
      act:       [N, 10]       10 continuous physical actuator channels
      (optional) saliency: [N, 16] 4x4 dual-scale spatial saliency heatmap

Actuator Channel Layout (ZERO fixed spell buttons or element enums):
  [0] move_x        in [-1, 1]  locomotion force X
  [1] move_y        in [-1, 1]  locomotion force Y
  [2] emit_dx       in [-1, 1]  physical emitter offset X (up to 36 canonical cells)
  [3] emit_dy       in [-1, 1]  physical emitter offset Y (up to 36 canonical cells)
  [4] thermal_flux  in [-1, 1]  continuous heat injection (+dE) or cryo extraction (-dE)
  [5] sp_water      in [ 0, 1]  water matter emission rate (mana-budgeted)
  [6] sp_oil        in [ 0, 1]  oil matter emission rate (mana-budgeted)
  [7] sp_smoke      in [ 0, 1]  smoke matter emission rate (mana-budgeted)
  [8] focus_radius  in [ 0, 1]  maps to 1.5..8.5 cells (narrow high-intensity beam vs wide field)
  [9] impulse_mag   in [ 0, 1]  kinetic wave / dash / knockback gust along emit vector

Architecture (~215K params):
  Reuses Frostmarsh's proven Dual-Scale Z_world (Fine r=14 + Coarse r=60) + Z_eco
  + 4-head Dual-Query (Navigation + Physical-Targeting) Entity/Shrine/Relic Cross-Attention backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

SELF_DIM = 24
NUM_TOKENS = 12  # 8 nearest players + 3 shrines + 1 golden relic
TOKEN_DIM = 12
BOT_FEAT_DIM = SELF_DIM + NUM_TOKENS * TOKEN_DIM  # 168 floats per bot

ACTUATOR_DIM = 10
BIPOLAR_DIM = 5    # move_x, move_y, emit_dx, emit_dy, thermal_flux in [-1, 1]
UNIPOLAR_DIM = 5   # sp_water, sp_oil, sp_smoke, focus_radius, impulse_mag in [0, 1]
SALIENCY_DIM = 16  # 4x4 spatial saliency heatmap
ACTUATOR_INSPECT_DIM = ACTUATOR_DIM + SALIENCY_DIM  # 26 floats per bot

CANONICAL_WORLD_SPAN = 255.0
MAX_EMIT_RANGE_CELLS = 36.0
MIN_FOCUS_RADIUS = 1.5
MAX_FOCUS_RADIUS = 8.5

ACTUATOR_NAMES = (
    "move_x",
    "move_y",
    "emit_dx",
    "emit_dy",
    "thermal_flux",
    "sp_water",
    "sp_oil",
    "sp_smoke",
    "focus_radius",
    "impulse_mag",
)

# Channel indices
ACT_MOVE_X = 0
ACT_MOVE_Y = 1
ACT_EMIT_DX = 2
ACT_EMIT_DY = 3
ACT_THERMAL = 4
ACT_WATER = 5
ACT_OIL = 6
ACT_SMOKE = 7
ACT_FOCUS = 8
ACT_IMPULSE = 9


def focus_radius_cells(focus_01: torch.Tensor) -> torch.Tensor:
    """Map normalized focus_radius in [0, 1] to physical radius in [1.5, 8.5] canonical cells."""
    return MIN_FOCUS_RADIUS + (MAX_FOCUS_RADIUS - MIN_FOCUS_RADIUS) * focus_01.clamp(0.0, 1.0)


def actuator_area_factor(focus_01: torch.Tensor) -> torch.Tensor:
    """Mana area multiplier for a physical field deposit of normalized focus_radius."""
    return 0.45 + 1.10 * focus_01.clamp(0.0, 1.0).pow(1.30)


def actuator_mana_cost(act: torch.Tensor, base_rate: float = 0.20) -> torch.Tensor:
    """Compute mana demand for actuator vector act [..., 10]."""
    flux_sum = (
        act[..., ACT_THERMAL].abs()
        + act[..., ACT_WATER]
        + act[..., ACT_OIL]
        + act[..., ACT_SMOKE]
        + act[..., ACT_IMPULSE]
    )
    return flux_sum * actuator_area_factor(act[..., ACT_FOCUS]) * base_rate


def raw_to_actuators(raw: torch.Tensor) -> torch.Tensor:
    """Transform unconstrained logits [..., 10] -> bounded physical actuators [..., 10]."""
    bipolar = torch.tanh(raw[..., :BIPOLAR_DIM])
    unipolar = torch.sigmoid(raw[..., BIPOLAR_DIM:])
    return torch.cat([bipolar, unipolar], dim=-1)


class ActuatorPolicyNet(nn.Module):
    """
    Dual-Scale Z_world (14-cell + 60-cell) + Z_eco + 4-head Dual-Query Cross-Attention policy
    outputting 10 continuous physical actuator channels (~215K parameters).
    """

    def __init__(
        self,
        z_world_ch: int = 48,
        z_eco_ch: int = 32,
        patch_size: int = 16,
        fine_radius_cells: float = 14.0,
        coarse_radius_cells: float = 60.0,
        d_model: int = 64,
        blind_world: bool = False,
        thermal_only: bool = False,
    ):
        super().__init__()
        self.z_world_ch = z_world_ch
        self.z_eco_ch = z_eco_ch
        self.patch_size = patch_size
        self.fine_radius_cells = fine_radius_cells
        self.coarse_radius_cells = coarse_radius_cells
        self.patch_radius_cells = fine_radius_cells
        self.blind_world = blind_world
        self.thermal_only = thermal_only

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
            nn.Conv2d(32, 44, kernel_size=3, stride=2, padding=1),       # 16 -> 8
            nn.GELU(),
            nn.Conv2d(44, 52, kernel_size=3, stride=2, padding=1),       # 8 -> 4
            nn.GELU(),
            nn.Conv2d(52, d_model, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.GELU(),
        )
        self.fine_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Stream A2: Coarse Z_world (48) at radius=60 cells
        self.coarse_adapter = nn.Sequential(
            nn.Conv2d(z_world_ch, 24, kernel_size=1),
            nn.GroupNorm(4, 24),
            nn.GELU(),
        )
        self.coarse_enc = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),       # 16 -> 8
            nn.GELU(),
            nn.Conv2d(32, 44, kernel_size=3, stride=2, padding=1),       # 8 -> 4
            nn.GELU(),
            nn.Conv2d(44, d_model, kernel_size=3, stride=2, padding=1),  # 4 -> 2
            nn.GELU(),
        )
        self.coarse_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Stream B: Self State + 4-head Dual-Query (Nav + Target) Entity / Shrine / Relic Cross-Attention
        self.self_enc = nn.Sequential(
            nn.Linear(SELF_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.query_emb = nn.Parameter(torch.randn(1, 2, d_model) * 0.05)  # [nav_query, target_query]
        self.tok_enc = nn.Sequential(
            nn.Linear(TOKEN_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.tok_type_emb = nn.Parameter(torch.randn(1, NUM_TOKENS, d_model) * 0.04)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)

        # Joint Policy Trunk (Fine 64 + Coarse 64 + NavAttn 64 + TargetAttn 64 = 256 -> 128 -> 128)
        self.trunk = nn.Sequential(
            nn.Linear(d_model * 4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # Continuous Physical Actuator Heads (10 channels + learned spatial pointers: 5 nav slots [3 shrines+relic+free], 13 emit slots)
        self.head_bipolar = nn.Linear(128 + d_model * 2, BIPOLAR_DIM)    # res_mv(2), res_em(2), thermal_flux(1)
        self.head_unipolar = nn.Linear(128 + d_model * 2, UNIPOLAR_DIM)  # sp_water, sp_oil, sp_smoke, focus_radius, impulse_mag
        self.head_ptr_mv = nn.Linear(128 + d_model * 2, 5)               # 4 objective tokens (8..11) + free residual
        self.head_ptr_em = nn.Linear(128 + d_model * 2, NUM_TOKENS + 1)  # all 12 tokens + free residual

        # Orthogonal initialization (standard PPO best practice for fast from-scratch RL convergence)
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(mod.weight, gain=2.0 ** 0.5)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

        # Small-gain output heads with balanced initial biases (matter/impulse start low to conserve mana)
        with torch.no_grad():
            nn.init.orthogonal_(self.head_bipolar.weight, gain=0.15)
            self.head_bipolar.bias.zero_()
            nn.init.orthogonal_(self.head_unipolar.weight, gain=0.15)
            # [water, oil, smoke, focus_radius, impulse_mag]
            self.head_unipolar.bias.copy_(torch.tensor([-1.8, -1.1, -1.5, -0.3, -1.4]))
            nn.init.orthogonal_(self.head_ptr_mv.weight, gain=0.15)
            self.head_ptr_mv.bias.zero_()
            nn.init.orthogonal_(self.head_ptr_em.weight, gain=0.15)
            self.head_ptr_em.bias.zero_()

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
            z_f = torch.zeros(
                (n, self.z_world_ch + self.z_eco_ch, p, p),
                device=self_feat.device,
                dtype=torch.float32,
            )
            z_c = torch.zeros(
                (n, self.z_world_ch, p, p),
                device=self_feat.device,
                dtype=torch.float32,
            )
            return z_f, z_c

        num_l = z_world.shape[0]
        cx = self_feat[:, 0:1, None] * 2.0 - 1.0
        cy = self_feat[:, 1:2, None] * 2.0 - 1.0

        r_fine = (self.fine_radius_cells / CANONICAL_WORLD_SPAN) * 2.0
        r_coarse = (self.coarse_radius_cells / CANONICAL_WORLD_SPAN) * 2.0

        gx_f = cx + self.base_gx * r_fine
        gy_f = cy + self.base_gy * r_fine
        gx_c = cx + self.base_gx * r_coarse
        gy_c = cy + self.base_gy * r_coarse

        grid_eco = torch.stack([gx_f, gy_f], dim=-1).reshape(1, n * p, p, 2).to(z_eco.dtype)
        eco_patch = (
            F.grid_sample(z_eco, grid_eco, mode="bilinear", padding_mode="border", align_corners=True)
            .reshape(1, self.z_eco_ch, n, p, p)
            .squeeze(0)
            .permute(1, 0, 2, 3)
            .float()
        )

        z_vol = z_world.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 48, L, H, W]
        if num_l > 1:
            cz = (self_feat[:, 2:3, None] * 2.0 - 1.0).clamp(-1.0, 1.0)
        else:
            cz = torch.zeros_like(cx)
        cz_p = cz.expand(-1, p, p)

        gx_both = torch.cat([gx_f, gx_c], dim=1)
        gy_both = torch.cat([gy_f, gy_c], dim=1)
        cz_both = torch.cat([cz_p, cz_p], dim=1)
        grid_3d = torch.stack([gx_both, gy_both, cz_both], dim=-1).unsqueeze(0).to(z_vol.dtype)

        w_both = (
            F.grid_sample(z_vol, grid_3d, mode="bilinear", padding_mode="border", align_corners=True)
            .squeeze(0)
            .permute(1, 0, 2, 3)
            .float()
        )
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Core forward pass from pre-sampled dual-scale patches.
        Returns:
          raw_act:  [M, 10] unconstrained pre-activation actuator outputs
          h_trunk:  [M, 128] shared trunk representation (for RL critic)
          saliency: [M, 16] or None
        """
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
            sal_f = torch.linalg.vector_norm(f_4x4.float(), ord=2, dim=1)
            sal_c = torch.linalg.vector_norm(c_4x4.float(), ord=2, dim=1)
            sal_map = sal_f + 0.35 * sal_c
            if sal_map.shape[-2:] != (4, 4):
                sal_map = F.adaptive_avg_pool2d(sal_map.unsqueeze(1), (4, 4)).squeeze(1)
            sal_flat = sal_map.reshape(sal_map.shape[0], SALIENCY_DIM)
            sal_min = sal_flat.amin(dim=-1, keepdim=True)
            sal_max = sal_flat.amax(dim=-1, keepdim=True)
            saliency = ((sal_flat - sal_min) / (sal_max - sal_min + 1e-6)).clamp(0.0, 1.0)

        # Dual-Query 4-head Cross-Attention (Nav Query + Physical Target Query)
        h_self = self.self_enc(self_feat)                            # [M, 64]
        q = h_self.unsqueeze(1) + self.query_emb                     # [M, 2, 64]
        kv = self.tok_enc(tok_feat) + self.tok_type_emb              # [M, 12, 64]
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        h_tac = self.attn_norm(q + attn_out).reshape(-1, h_self.shape[-1] * 2)  # [M, 128]

        h_trunk = self.trunk(torch.cat([h_fine, h_coarse, h_tac], dim=-1))
        h_head = torch.cat([h_trunk, h_tac], dim=-1)
        raw_bi = self.head_bipolar(h_head)
        raw_uni = self.head_unipolar(h_head)

        # Learned spatial pointers: navigation over objective tokens (8..11)+free; emission over all 12 tokens+free
        w_mv = F.softmax(self.head_ptr_mv(h_head), dim=-1)  # [M, 5]
        w_em = F.softmax(self.head_ptr_em(h_head), dim=-1)  # [M, 13]
        tok_xy = tok_feat[:, :, 0:2]                        # [M, 12, 2]
        obj_xy = tok_xy[:, 8:12, :]                         # [M, 4, 2] (3 shrines + relic/home)
        res_mv = raw_bi[:, 0:2]
        res_em = raw_bi[:, 2:4]
        mv_raw = (w_mv[:, :4].unsqueeze(-1) * (obj_xy * 4.5)).sum(dim=1) + (w_mv[:, 4:5].clamp(min=0.18)) * res_mv
        em_raw = (w_em[:, :NUM_TOKENS].unsqueeze(-1) * (tok_xy * 2.22)).sum(dim=1) + (w_em[:, NUM_TOKENS:].clamp(min=0.18)) * res_em
        raw_act = torch.cat([mv_raw, em_raw, raw_bi[:, 4:5], raw_uni], dim=-1)
        return raw_act, h_head, saliency

    def forward(
        self,
        z_world: torch.Tensor,
        z_eco: torch.Tensor,
        bot_feats: torch.Tensor,
        return_saliency: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Drop-in op=7 inference API:
          z_world:   [L, 48, H, W]
          z_eco:     [1, 32, H, W]
          bot_feats: [N, 168]
        Returns:
          act [N, 10] (or (act [N, 10], saliency [N, 16]) if return_saliency=True).
        """
        fine_p, coarse_p = self.sample_shared_latents(z_world, z_eco, bot_feats[:, :SELF_DIM])
        raw_act, _, saliency = self.forward_from_patches(
            fine_p, coarse_p, bot_feats, return_saliency=return_saliency
        )
        act = raw_to_actuators(raw_act)
        if self.thermal_only:
            act = act.clone()
            act[:, ACT_WATER : ACT_SMOKE + 1] = 0.0
        if return_saliency:
            assert saliency is not None
            return act, saliency
        return act


def sample_batched_dual_patches(
    policy: ActuatorPolicyNet,
    z_world: torch.Tensor,    # [A, L, 48, H, W]
    z_eco: torch.Tensor,      # [A, 1, 32, H, W]
    bot_feats: torch.Tensor,  # [A, N, 168]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-arena zero-copy dual-scale sampling matching ActuatorPolicyNet.sample_shared_latents exactly.
    Returns:
      fine_patches:   [A, N, 80, P, P]
      coarse_patches: [A, N, 48, P, P]
    """
    a, n, _ = bot_feats.shape
    p = policy.patch_size
    if policy.blind_world:
        z_f = torch.zeros(
            (a, n, policy.z_world_ch + policy.z_eco_ch, p, p),
            device=bot_feats.device,
            dtype=torch.float32,
        )
        z_c = torch.zeros(
            (a, n, policy.z_world_ch, p, p),
            device=bot_feats.device,
            dtype=torch.float32,
        )
        return z_f, z_c

    num_l = z_world.shape[1]
    sf = bot_feats[:, :, :SELF_DIM]
    cx = sf[:, :, 0:1, None] * 2.0 - 1.0
    cy = sf[:, :, 1:2, None] * 2.0 - 1.0

    r_fine = (policy.fine_radius_cells / CANONICAL_WORLD_SPAN) * 2.0
    r_coarse = (policy.coarse_radius_cells / CANONICAL_WORLD_SPAN) * 2.0

    bgx = policy.base_gx.unsqueeze(0)
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

    z_vol = z_world.permute(0, 2, 1, 3, 4)
    if num_l > 1:
        cz = (sf[:, :, 2:3, None] * 2.0 - 1.0).clamp(-1.0, 1.0)
    else:
        cz = torch.zeros_like(cx)
    cz_p = cz.expand(-1, -1, p, p)

    gx_both = torch.cat([gx_f, gx_c], dim=2)
    gy_both = torch.cat([gy_f, gy_c], dim=2)
    cz_both = torch.cat([cz_p, cz_p], dim=2)
    grid_3d = torch.stack([gx_both, gy_both, cz_both], dim=-1).to(z_vol.dtype)

    w_both = (
        F.grid_sample(z_vol, grid_3d, mode="bilinear", padding_mode="border", align_corners=True)
        .permute(0, 2, 1, 3, 4)
        .float()
    )
    w_fine = w_both[:, :, :, :p, :]
    w_coarse = w_both[:, :, :, p:, :]

    fine_patch = torch.cat([w_fine, eco_patch], dim=2)
    return fine_patch, w_coarse


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
