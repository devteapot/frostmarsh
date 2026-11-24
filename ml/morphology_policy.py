#!/usr/bin/env python3
"""
Frostmarsh Wave 20B: `MorphologyActuatorNet` (~310K params) — Continuous Morphology-Conditioned,
Zero-Hardcoded-Rules 3D Universal Physical Actuator Policy.

Extends `ActuatorPolicyNetV3` (zero-copy 5D trilinear `F.grid_sample` over frozen `WorldModel3D-v7`
48-ch `Z_world_3d` crops + 4-head Cross-Attention over 3D physical tokens) with FiLM conditioning on:
  - `phi_body`    (6D continuous physical morphology in [0, 1]):
      [mass, wing_lift, buoyancy, climb_grip, bore_torque, thermal_insul]
  - `theta_world` (6D continuous environment physical laws):
      [gravity_g, ambient_temp_offset, precip_rate, thermal_dissipation, friction_coeff, vent_flux]
  - `homeostasis` (4D continuous physiological state):
      [core_temp_norm, integrity_hp, energy_mana, submergence_stress]

ZERO class enums. ZERO hardcoded spell buttons. ZERO game-mode IDs.
Outputs the exact same 12 continuous physical actuators (`ACT3D_DIM = 12`).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from actuator_policy_v3 import (  # noqa: E402
    ACT3D_DIM,
    ACT3D_NAMES,
    MAX_TOKENS_3D,
    SELF3D_FEAT_DIM,
    TOKEN3D_FEAT_DIM,
    ActuatorPolicyNetV3,
    count_parameters,
)

MORPH_DIM = 6
MORPH_NAMES = (
    "mass",
    "wing_lift",
    "buoyancy",
    "climb_grip",
    "bore_torque",
    "thermal_insul",
)

WORLD_LAW_DIM = 6
WORLD_LAW_NAMES = (
    "gravity_g",
    "ambient_temp_offset",
    "precip_rate",
    "thermal_dissipation",
    "friction_coeff",
    "vent_flux",
)

HOMEO_DIM = 4
HOMEO_NAMES = (
    "core_temp_norm",
    "integrity_hp",
    "energy_mana",
    "submergence_stress",
)

COND_DIM = MORPH_DIM + WORLD_LAW_DIM + HOMEO_DIM  # 16

# Strictly held-out continuous hybrid morphologies for Zero-Shot Morphology Generalization evaluation
HELD_OUT_MORPHOLOGIES: dict[str, tuple[float, ...]] = {
    "amphibious_glider": (0.35, 0.85, 0.90, 0.20, 0.10, 0.40),
    "insulated_heavy_excavator": (0.90, 0.05, 0.20, 0.50, 0.95, 0.90),
}

# Representative specialist anchors used (with continuous jitter) to cover the 6D morphology cube richly
_TRAINING_PROTOTYPES: tuple[tuple[float, ...], ...] = (
    (0.22, 0.92, 0.20, 0.30, 0.10, 0.25),  # Light thermal flyer (low buoyancy)
    (0.45, 0.10, 0.92, 0.25, 0.15, 0.35),  # Aquatic swimmer (flightless)
    (0.38, 0.15, 0.25, 0.92, 0.20, 0.30),  # Alpine rock climber (flightless, low buoyancy)
    (0.82, 0.08, 0.18, 0.20, 0.90, 0.28),  # Uninsulated rock borer
    (0.75, 0.12, 0.35, 0.35, 0.25, 0.92),  # Arctic insulated walker
    (0.50, 0.50, 0.50, 0.50, 0.50, 0.50),  # Balanced generalist
    (0.25, 0.15, 0.18, 0.20, 0.15, 0.15),  # Fragile cold/water-sensitive scout (must rely on hearths/cryo!)
    (0.88, 0.10, 0.85, 0.20, 0.30, 0.75),  # Heavy buoyant diver
)


def get_held_out_morphology_tensor(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return `[2, 6]` tensor of the two strictly held-out zero-shot hybrid morphologies."""
    return torch.tensor(
        [HELD_OUT_MORPHOLOGIES["amphibious_glider"], HELD_OUT_MORPHOLOGIES["insulated_heavy_excavator"]],
        device=device,
        dtype=dtype,
    )


def sample_training_morphologies(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator | None = None,
    min_holdout_l2: float = 0.35,
) -> tuple[torch.Tensor, float]:
    """
    Continuously sample `phi_body` in `[0.05, 0.98]^6` while strictly excluding a ball of radius
    `min_holdout_l2` around both held-out zero-shot hybrid morphologies!

    Returns `(phi_body [*shape, 6], actual_min_l2_to_holdout)`.
    """
    total = 1
    for s in shape:
        total *= int(s)

    holdouts = get_held_out_morphology_tensor(device)  # [2, 6]
    protos = torch.tensor(_TRAINING_PROTOTYPES, device=device, dtype=torch.float32)  # [P, 6]

    def _draw_raw(n: int) -> torch.Tensor:
        # 50% continuous uniform across [0.05, 0.98]^6, 50% continuously jittered specialist prototypes
        u_uni = torch.empty((n, MORPH_DIM), device=device).uniform_(0.05, 0.98, generator=generator)
        idx = torch.randint(0, protos.shape[0], (n,), device=device, generator=generator)
        noise = torch.randn((n, MORPH_DIM), device=device, generator=generator) * 0.16
        u_proto = (protos[idx] + noise).clamp(0.05, 0.98)
        pick_uni = (torch.rand((n, 1), device=device, generator=generator) < 0.50)
        return torch.where(pick_uni, u_uni, u_proto)

    phi = _draw_raw(total)
    # Rejection resample any sample too close to held-out zero-shot hybrids
    for _ in range(12):
        d = torch.cdist(phi, holdouts, p=2.0).min(dim=-1).values  # [total]
        bad = d < min_holdout_l2
        if not bad.any():
            break
        n_bad = int(bad.sum().item())
        phi[bad] = _draw_raw(n_bad)

    min_dist = float(torch.cdist(phi, holdouts, p=2.0).min().item())
    return phi.view(*shape, MORPH_DIM), min_dist


def _apply_film(x: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
    """Bounded smooth FiLM modulation: `x * (1 + 0.65 * tanh(gamma)) + beta`."""
    gamma, beta = gb.chunk(2, dim=-1)
    return x * (1.0 + 0.65 * torch.tanh(gamma)) + beta


class MorphologyActuatorNet(ActuatorPolicyNetV3):
    """
    Universal Morphology-Conditioned 3D Physical Actuator Policy (~310K params).

    Extends `ActuatorPolicyNetV3` with FiLM conditioning on continuous
    `[phi_body (6), theta_world (6), homeostasis_state (4)]` across:
      - Fine 3D volumetric representation `h_fine` (64-dim)
      - Coarse 3D tactical representation `h_coarse` (64-dim)
      - Proprioceptive self-query `h_self` (64-dim) BEFORE 4-head Cross-Attention
      - Fused actor-critic trunk output `h_trunk` (128-dim)

    Also supports `blind_morphology=True` (zeroes `phi_body` cleanly at evaluation)
    for airtight Morphology-Blind Ablation verification.
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
        coarse_span_xy: float = 48.0,
        self_dim: int = SELF3D_FEAT_DIM,
        tok_dim: int = TOKEN3D_FEAT_DIM,
        goal_dim: int = 8,
        cond_hidden: int = 28,
    ):
        super().__init__(
            z_world_ch=z_world_ch,
            z_eco_ch=z_eco_ch,
            latent_dim=latent_dim,
            fine_xy=fine_xy,
            fine_z=fine_z,
            coarse_xy=coarse_xy,
            coarse_z=coarse_z,
            fine_span_xy=fine_span_xy,
            fine_span_z=fine_span_z,
            coarse_span_xy=coarse_span_xy,
            self_dim=self_dim,
            tok_dim=tok_dim,
            goal_dim=goal_dim,
        )
        self.cond_dim = COND_DIM
        self.cond_hidden = cond_hidden

        # Continuous [phi_body(6), theta_world(6), homeostasis(4)] -> nonlinear interaction FiLM code
        # with explicit residual skip for phi_body so morphology × environment-law interactions stay crisp
        self.cond_in_proj = nn.Sequential(
            nn.Linear(COND_DIM, 40),
            nn.LayerNorm(40),
            nn.SiLU(inplace=True),
        )
        self.cond_out_proj = nn.Sequential(
            nn.Linear(40 + MORPH_DIM, cond_hidden),
            nn.LayerNorm(cond_hidden),
            nn.SiLU(inplace=True),
        )

        # FiLM affine modulation generators (gamma, beta) across spatial, attention, and policy trunk stages
        self.film_fine = nn.Linear(cond_hidden, 64 * 2)
        self.film_coarse = nn.Linear(cond_hidden, 64 * 2)
        self.film_self = nn.Linear(cond_hidden, 64 * 2)
        self.film_trunk = nn.Linear(cond_hidden, 128 * 2)
        # Direct conditioning residual into actuator logits (eliminates cross-game sign tug-of-war on thermal/jump!)
        self.cond_actor_bias = nn.Linear(cond_hidden, ACT3D_DIM)

        # Crisp initialization giving immediate strong gradient flow into phi_body, theta_world, and homeostasis
        for head in (self.film_fine, self.film_coarse, self.film_self, self.film_trunk):
            nn.init.normal_(head.weight, mean=0.0, std=0.08)
            nn.init.zeros_(head.bias)
        nn.init.normal_(self.cond_actor_bias.weight, mean=0.0, std=0.06)
        nn.init.zeros_(self.cond_actor_bias.bias)

    def _build_cond_vec(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        phi_body: torch.Tensor | None = None,
        theta_world: torch.Tensor | None = None,
        homeostasis: torch.Tensor | None = None,
        blind_morphology: bool = False,
    ) -> torch.Tensor:
        if phi_body is None or blind_morphology:
            phi = torch.zeros((batch_size, MORPH_DIM), device=device, dtype=dtype)
        else:
            phi = phi_body.to(device=device, dtype=dtype)

        if theta_world is None:
            theta = torch.tensor(
                [1.0, 0.0, 0.0, 0.05, 1.0, 0.5], device=device, dtype=dtype
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            theta = theta_world.to(device=device, dtype=dtype)

        if homeostasis is None:
            homeo = torch.tensor(
                [0.5, 1.0, 1.0, 0.0], device=device, dtype=dtype
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            homeo = homeostasis.to(device=device, dtype=dtype)

        return torch.cat((phi, theta, homeo), dim=-1)

    def forward_from_crops(
        self,
        fine_raw: torch.Tensor,
        coarse_raw: torch.Tensor,
        self_feat: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        goal_vec: torch.Tensor | None = None,
        eco_raw: torch.Tensor | None = None,
        phi_body: torch.Tensor | None = None,
        theta_world: torch.Tensor | None = None,
        homeostasis: torch.Tensor | None = None,
        blind_morphology: bool = False,
        return_saliency: bool = False,
    ) -> dict[str, torch.Tensor]:
        N = fine_raw.shape[0]
        device = fine_raw.device
        p_dtype = self.actor_mean.weight.dtype

        cond_in = self._build_cond_vec(
            N,
            device=device,
            dtype=p_dtype,
            phi_body=phi_body,
            theta_world=theta_world,
            homeostasis=homeostasis,
            blind_morphology=blind_morphology,
        )
        phi_part = cond_in[:, :MORPH_DIM]
        h_cond = self.cond_out_proj(torch.cat((self.cond_in_proj(cond_in), phi_part), dim=-1))

        fine_crop = self.z_adapter(fine_raw.to(p_dtype))
        coarse_crop = self.z_adapter(coarse_raw.to(p_dtype))
        h_fine = _apply_film(self.fine_enc(fine_crop), self.film_fine(h_cond))
        h_coarse = _apply_film(self.coarse_enc(coarse_crop), self.film_coarse(h_cond))

        if goal_vec is None:
            goal_vec = torch.zeros((N, self.goal_dim), device=device, dtype=p_dtype)
        else:
            goal_vec = goal_vec.to(p_dtype)

        if eco_raw is not None:
            eco_vec = self.eco_adapter(eco_raw.to(p_dtype).view(N, self.z_eco_ch, 1, 1)).view(N, 8)
        else:
            eco_vec = torch.zeros((N, 8), device=device, dtype=p_dtype)

        h_self_base = self.self_proj(torch.cat((self_feat.to(p_dtype), goal_vec, eco_vec), dim=-1))
        h_self = _apply_film(h_self_base, self.film_self(h_cond))

        h_tok = self.tok_proj(tokens.to(p_dtype))
        key_pad = (~token_mask) if token_mask is not None else None
        attn_out, attn_w = self.cross_attn(
            h_self.unsqueeze(1), h_tok, h_tok, key_padding_mask=key_pad, need_weights=True
        )
        h_attn = self.attn_norm(h_self + attn_out.squeeze(1))

        h = self.trunk(torch.cat((h_fine, h_coarse, h_self, h_attn), dim=-1))
        h = _apply_film(h, self.film_trunk(h_cond))

        raw_mean = self.actor_mean(h) + self.cond_actor_bias(h_cond)
        value = self.critic(h).squeeze(-1)

        # Exact same 12 bounded physical actuator ranges as ActuatorPolicyNetV3
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
            fine_2d = fine_crop.mean(dim=2)
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
        phi_body: torch.Tensor | None = None,
        theta_world: torch.Tensor | None = None,
        homeostasis: torch.Tensor | None = None,
        blind_morphology: bool = False,
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
            phi_body=phi_body,
            theta_world=theta_world,
            homeostasis=homeostasis,
            blind_morphology=blind_morphology,
            return_saliency=return_saliency,
        )


def _smoke_test() -> dict[str, Any]:
    torch.manual_seed(42)
    net = MorphologyActuatorNet()
    n_params = count_parameters(net)
    base_params = count_parameters(ActuatorPolicyNetV3())
    phi_tr, min_d = sample_training_morphologies((64,), device=torch.device("cpu"))
    assert min_d >= 0.35 - 1e-6, f"Holdout leak: min_d={min_d}"

    for L, B, nb in [(16, 1, 6), (32, 3, 18)]:
        z_world = torch.randn(B, 48, L, 48, 48)
        pos = torch.stack(
            (
                torch.rand(nb) * 42.0 + 3.0,
                torch.rand(nb) * 42.0 + 3.0,
                torch.rand(nb) * (L - 1),
            ),
            dim=-1,
        )
        sf = torch.randn(nb, SELF3D_FEAT_DIM)
        toks = torch.randn(nb, MAX_TOKENS_3D, TOKEN3D_FEAT_DIM)
        mask = torch.ones(nb, MAX_TOKENS_3D, dtype=torch.bool)
        aidx = torch.arange(B).repeat_interleave(nb // B)
        phi, _ = sample_training_morphologies((nb,), device=torch.device("cpu"))
        theta = torch.randn(nb, WORLD_LAW_DIM)
        homeo = torch.rand(nb, HOMEO_DIM)

        out = net(
            z_world, pos, sf, toks, token_mask=mask, arena_idx=aidx,
            phi_body=phi, theta_world=theta, homeostasis=homeo, return_saliency=True,
        )
        out_blind = net(
            z_world, pos, sf, toks, token_mask=mask, arena_idx=aidx,
            phi_body=phi, theta_world=theta, homeostasis=homeo, blind_morphology=True,
        )
        act, u, lp = net.sample_action(out)
        lp2, ent = net.evaluate_actions(out, u)
        assert act.shape == (nb, ACT3D_DIM)
        assert out["saliency_4x4"].shape == (nb, 16)
        assert torch.allclose(lp, lp2, atol=1e-5)
        assert torch.isfinite(act).all() and torch.isfinite(lp).all() and torch.isfinite(ent).all()
        diff_blind = (out["mean"] - out_blind["mean"]).abs().max().item()
        assert diff_blind > 1e-5, f"FiLM had no effect on phi_body! diff={diff_blind}"

    print(
        f"[MorphologyActuatorNet] Smoke test PASS | params={n_params:,} (base V3={base_params:,}, FiLM=+{n_params-base_params:,}) "
        f"| min_holdout_L2={min_d:.3f} | Z=16 & Z=32 verified!"
    )
    return {"params": n_params, "base_params": base_params, "min_holdout_l2": min_d, "status": "ok"}


if __name__ == "__main__":
    _smoke_test()
