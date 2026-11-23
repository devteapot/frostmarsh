#!/usr/bin/env python3
"""
Part B: Dual-Scale Z_world `BotPolicyNetV2` Online DAgger + Self-Play RL (`train_bot_rl_v2.py`).

Pipeline:
  - Stage 0: Online DAgger Calibration (~150 steps on 32 parallel 128x128x2 arenas = 384 bots)
             against Scripted Teacher using dual-scale Z_world (Fine r=14 + Coarse r=60) + Z_eco.
  - Stage 1: Trust-Region Float32-Safe Dual-Horizon PPO Self-Play on 32 parallel 128x128x2 arenas
             (lr=8e-5, KL early-stop=0.35, best-checkpoint selection, optimizing Shrine + Relic +
              Combo + Kill + Fire-Evasion against Teacher + Lagged-Self + Mirror-Self).
  - Stage 2: High-Res 256x256x4 PPO Self-Play Polish (8 parallel arenas = 96 bots, lr=5e-5,
             KL early-stop=0.35, best-checkpoint selection on 256x256x4 validation arenas).
  - Evaluation: Exact 24-match side-balanced held-out 256x256x4 tournament (matching eval_bot_arena.py)
             vs Scripted Teacher, BotPolicyNet-RL v1, BotPolicyNet-BC, and Blind Ledger-Only,
             plus GB10 batched latency benchmark (12 / 64 / 256 bots).

Saves:
  - /home/gooning/.t3/scratch/frostmarsh-ml/bot_policy_v2_rl.pt
  - /home/gooning/.t3/scratch/frostmarsh-ml/BOT_RL_V2_TOURNAMENT.json
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bot_policy import BotPolicyNet
from bot_policy_v2 import (
    ACTION_OUT_DIM,
    BOT_FEAT_DIM,
    NUM_TOKENS,
    SALIENCY_DIM,
    SELF_DIM,
    TOKEN_DIM,
    BotPolicyNetV2,
    count_parameters,
    sample_batched_dual_patches,
)
from train_bot_rl import (
    BC_PT,
    NUM_BOTS,
    RL_PT,
    ROOT,
    FrostmarshGPUArena,
    compute_observable_aim_aux_loss,
    load_ckpt_state_dict,
    load_frozen_world_models,
    policy_deterministic_act as v1_policy_deterministic_act,
    scripted_teacher_act,
)

V2_RL_PT = ROOT / "bot_policy_v2_rl.pt"
OUT_JSON = ROOT / "BOT_RL_V2_TOURNAMENT.json"


def batched_v2_forward(
    policy: BotPolicyNetV2,
    z_world: torch.Tensor,
    z_eco: torch.Tensor,
    bot_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, n, _ = bot_feats.shape
    p = policy.patch_size
    fp, cp = sample_batched_dual_patches(policy, z_world, z_eco, bot_feats)
    vec, bin_logits, elem_logits, h_trunk, _ = policy.forward_from_patches(
        fp.reshape(a * n, -1, p, p),
        cp.reshape(a * n, -1, p, p),
        bot_feats.reshape(a * n, BOT_FEAT_DIM),
        return_saliency=False,
    )
    return (
        vec.reshape(a, n, 4),
        bin_logits.reshape(a, n, 6),
        elem_logits.reshape(a, n, 3),
        h_trunk.reshape(a, n, 128),
    )


@torch.inference_mode()
def v2_deterministic_act(
    policy: BotPolicyNetV2,
    z_world: torch.Tensor,
    z_eco: torch.Tensor,
    bot_feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vec, bin_logits, elem_logits, _ = batched_v2_forward(policy, z_world, z_eco, bot_feats)
    return vec, (torch.sigmoid(bin_logits) > 0.5).float(), elem_logits.argmax(dim=-1)


class RLBotActorCriticV2(nn.Module):
    """Wraps pure BotPolicyNetV2 with dual-horizon FP32 RL value critics (tactical + combat)."""

    def __init__(self, policy: BotPolicyNetV2):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.90, -1.90, -2.20, -2.20], dtype=torch.float32))

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-2.8, -1.35))

    def sample_actions_from_patches(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        deterministic: bool = False,
    ):
        a, n, _ = bot_feats.shape
        p = self.policy.patch_size
        vec, bin_logits, elem_logits, h_trunk, _ = self.policy.forward_from_patches(
            fine_patches.reshape(a * n, -1, p, p).float(),
            coarse_patches.reshape(a * n, -1, p, p).float(),
            bot_feats.reshape(a * n, BOT_FEAT_DIM).float(),
            return_saliency=False,
        )
        values = self.value_head(h_trunk).reshape(a, n, 2)
        vec = vec.reshape(a, n, 4)
        bin_logits = bin_logits.reshape(a, n, 6)
        elem_logits = elem_logits.reshape(a, n, 3)

        if deterministic:
            z1 = torch.zeros((a, n), device=values.device)
            z6 = torch.zeros((a, n, 6), device=values.device)
            return vec, (torch.sigmoid(bin_logits) > 0.5).float(), elem_logits.argmax(dim=-1), z1, z1, z6, z1, values

        std = self.get_std().view(1, 1, 4)
        dist_mv = torch.distributions.Normal(vec[..., :2], std[..., :2])
        a_mv_raw = dist_mv.sample()
        lp_mv = dist_mv.log_prob(a_mv_raw).sum(dim=-1)

        dist_aim = torch.distributions.Normal(vec[..., 2:], std[..., 2:])
        a_aim_raw = dist_aim.sample()
        lp_aim = dist_aim.log_prob(a_aim_raw).sum(dim=-1)

        a_vec_raw = torch.cat([a_mv_raw, a_aim_raw], dim=-1)

        dist_bin = torch.distributions.Bernoulli(logits=bin_logits.clamp(-5.0, 5.0))
        a_bin = dist_bin.sample()
        lp_bin = dist_bin.log_prob(a_bin)

        dist_elem = torch.distributions.Categorical(logits=elem_logits.clamp(-5.0, 5.0))
        a_elem = dist_elem.sample()
        lp_elem = dist_elem.log_prob(a_elem)

        return a_vec_raw, a_bin, a_elem, lp_mv, lp_aim, lp_bin, lp_elem, values

    def evaluate_flat_minibatch(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        a_vec_raw: torch.Tensor,
        a_bin: torch.Tensor,
        a_elem: torch.Tensor,
    ):
        # Strictly FP32 for trust-region PPO numerical safety
        vec, bin_logits, elem_logits, h_trunk, _ = self.policy.forward_from_patches(
            fine_patches.float(), coarse_patches.float(), bot_feats.float(), return_saliency=False
        )
        values = self.value_head(h_trunk)
        std = self.get_std().view(1, 4)

        dist_mv = torch.distributions.Normal(vec[:, :2], std[:, :2])
        lp_mv = dist_mv.log_prob(a_vec_raw[:, :2]).sum(dim=-1)
        ent_mv = dist_mv.entropy().sum(dim=-1)

        dist_aim = torch.distributions.Normal(vec[:, 2:], std[:, 2:])
        lp_aim = dist_aim.log_prob(a_vec_raw[:, 2:]).sum(dim=-1)
        ent_aim = dist_aim.entropy().sum(dim=-1)

        dist_bin = torch.distributions.Bernoulli(logits=bin_logits.clamp(-5.0, 5.0))
        lp_bin = dist_bin.log_prob(a_bin)
        ent_bin = dist_bin.entropy().mean(dim=-1)

        dist_elem = torch.distributions.Categorical(logits=elem_logits.clamp(-5.0, 5.0))
        lp_elem = dist_elem.log_prob(a_elem)
        ent_elem = dist_elem.entropy()

        return (
            lp_mv, lp_aim, lp_bin, lp_elem,
            ent_mv, ent_aim, ent_bin, ent_elem,
            values, vec, bin_logits, elem_logits,
        )


def compute_policy_kl_fp32(
    vec: torch.Tensor,
    bin_logits: torch.Tensor,
    elem_logits: torch.Tensor,
    ref_vec: torch.Tensor,
    ref_bin_logits: torch.Tensor,
    ref_elem_logits: torch.Tensor,
) -> torch.Tensor:
    vec = vec.float()
    ref_vec = ref_vec.float()
    bl = bin_logits.float().clamp(-7.0, 7.0)
    rbl = ref_bin_logits.float().clamp(-7.0, 7.0)
    el = elem_logits.float().clamp(-7.0, 7.0)
    rel = ref_elem_logits.float().clamp(-7.0, 7.0)

    kl_v = ((vec - ref_vec) ** 2).mean(dim=-1)
    p = torch.sigmoid(bl)
    kl_b = (p * (F.logsigmoid(bl) - F.logsigmoid(rbl)) + (1.0 - p) * (F.logsigmoid(-bl) - F.logsigmoid(-rbl))).mean(dim=-1)
    lpe = F.log_softmax(el, dim=-1)
    lqe = F.log_softmax(rel, dim=-1)
    kl_e = (lpe.exp() * (lpe - lqe)).sum(dim=-1)
    return 0.25 * kl_v + 0.15 * kl_b + 0.15 * kl_e


@torch.inference_mode()
def quick_det_eval_v2(
    policy: BotPolicyNetV2,
    v1_rl_policy: BotPolicyNet,
    val_arena: FrostmarshGPUArena,
    seed: int = 77777,
) -> dict:
    """Fast side-balanced deterministic evaluation vs Scripted Teacher (half) & RL-v1 (half)."""
    policy.eval()
    A = val_arena.A
    half = A // 2
    f = val_arena.reset(seed=seed)

    v2_m = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    v2_m[: half // 2, :6] = True
    v2_m[half // 2 : half, 6:] = True
    v2_m[half : half + half // 2, :6] = True
    v2_m[half + half // 2 :, 6:] = True

    t_m = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    t_m[: half // 2, 6:] = True
    t_m[half // 2 : half, :6] = True

    for _ in range(val_arena.max_ticks):
        rv, rb, re = v2_deterministic_act(policy, val_arena.z_world, val_arena.z_eco, f)
        v1v, v1b, v1e = v1_policy_deterministic_act(v1_rl_policy, val_arena.z_world, val_arena.z_eco, f)
        tv, tb, te = scripted_teacher_act(val_arena)

        av = torch.where(v2_m.unsqueeze(-1), rv, torch.where(t_m.unsqueeze(-1), tv, v1v))
        ab = torch.where(v2_m.unsqueeze(-1), rb, torch.where(t_m.unsqueeze(-1), tb, v1b))
        ae = torch.where(v2_m, re, torch.where(t_m, te, v1e))
        f, _, _ = val_arena.step(av, ab, ae)

    sw, se = val_arena.team_score[:, 0], val_arena.team_score[:, 1]
    t_w = (sw[: half // 2] > se[: half // 2]).float() + 0.5 * (sw[: half // 2] == se[: half // 2]).float()
    t_e = (se[half // 2 : half] > sw[half // 2 : half]).float() + 0.5 * (se[half // 2 : half] == sw[half // 2 : half]).float()
    v1_w = (sw[half : half + half // 2] > se[half : half + half // 2]).float() + 0.5 * (sw[half : half + half // 2] == se[half : half + half // 2]).float()
    v1_e = (se[half + half // 2 :] > sw[half + half // 2 :]).float() + 0.5 * (se[half + half // 2 :] == sw[half + half // 2 :]).float()

    sc_v2 = float(torch.cat([sw[: half // 2], se[half // 2 : half]]).mean().item())
    sc_t = float(torch.cat([se[: half // 2], sw[half // 2 : half]]).mean().item())
    burn_v2 = float(torch.cat([val_arena.team_burn_taken[: half // 2, 0], val_arena.team_burn_taken[half // 2 : half, 1]]).mean().item())
    combos_v2 = float(torch.cat([val_arena.team_combos[: half // 2, 0], val_arena.team_combos[half // 2 : half, 1]]).mean().item())
    relics_v2 = float(torch.cat([val_arena.team_relic_deliveries[: half // 2, 0], val_arena.team_relic_deliveries[half // 2 : half, 1]]).mean().item())
    kills_v2 = float(torch.cat([val_arena.team_kills[: half // 2, 0], val_arena.team_kills[half // 2 : half, 1]]).mean().item())
    shrines_v2 = float(torch.cat([val_arena.team_shrine_caps[: half // 2, 0], val_arena.team_shrine_caps[half // 2 : half, 1]]).mean().item())

    return {
        "wr_teacher": float(torch.cat([t_w, t_e]).mean().item()) * 100.0,
        "wr_v1": float(torch.cat([v1_w, v1_e]).mean().item()) * 100.0,
        "score_v2": sc_v2,
        "score_t": sc_t,
        "burn": burn_v2,
        "combos": combos_v2,
        "relics": relics_v2,
        "kills": kills_v2,
        "shrines": shrines_v2,
    }


def run_stage0_online_dagger(
    policy: BotPolicyNetV2,
    arena: FrostmarshGPUArena,
    val_arena: FrostmarshGPUArena,
    v1_rl_policy: BotPolicyNet,
    num_steps: int = 150,
) -> dict:
    """
    Stage 0: Fast Online DAgger calibration on 32 parallel arenas (150 steps x 384 bots).
    Calibrates dual-scale Z_world (Fine r=14 + Coarse r=60) + Z_eco + Cross-Attention
    on immediate fire evasion, 60-cell oil/water combo selection, shrine play, and relic runs.
    """
    print(f"\n=== Stage 0: Online DAgger Calibration ({num_steps} steps x {arena.A}x{arena.N}={arena.A*arena.N} bots) ===", flush=True)
    dev = arena.device
    A, N = arena.A, arena.N
    P = policy.patch_size
    grad_steps_per_tick = 10
    total_grad_steps = num_steps * grad_steps_per_tick

    optimizer = torch.optim.AdamW(policy.parameters(), lr=2.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_grad_steps, eta_min=1.5e-4)

    t0 = time.perf_counter()
    feats = arena.reset(seed=100)
    total_bot_steps = 0

    # Pre-allocated circular GPU replay buffer (last 32 env steps = 12,288 diverse transitions)
    buf_cap = 32
    cap_n = buf_cap * A * N
    buf_fp = torch.empty((cap_n, 80, P, P), dtype=torch.float16, device=dev)
    buf_cp = torch.empty((cap_n, 48, P, P), dtype=torch.float16, device=dev)
    buf_f = torch.empty((cap_n, BOT_FEAT_DIM), dtype=torch.float32, device=dev)
    buf_tv = torch.empty((cap_n, 4), dtype=torch.float32, device=dev)
    buf_tb = torch.empty((cap_n, 6), dtype=torch.float32, device=dev)
    buf_te = torch.empty((cap_n,), dtype=torch.long, device=dev)
    buf_size = 0
    buf_ptr = 0

    best_score = -1e9
    best_sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    for step in range(1, num_steps + 1):
        # Anneal teacher mixing beta: 1.0 -> 0.25 (pure teacher warm-up first 30 steps, then online DAgger)
        if step <= 30:
            beta = 1.0
        else:
            beta = max(0.25, 1.0 - 0.75 * ((step - 30) / float(max(1, num_steps - 30))))

        with torch.inference_mode():
            fp, cp = sample_batched_dual_patches(policy, arena.z_world, arena.z_eco, feats)
            tv, tb, te = scripted_teacher_act(arena)

            if beta >= 0.999:
                act_v, act_b, act_e = tv, tb, te
            else:
                policy.eval()
                sv, sb, se = v2_deterministic_act(policy, arena.z_world, arena.z_eco, feats)
                use_t = torch.rand((A, N), device=dev) < beta
                act_v = torch.where(use_t.unsqueeze(-1), tv, sv)
                act_b = torch.where(use_t.unsqueeze(-1), tb, sb)
                act_e = torch.where(use_t, te, se)

            chunk = A * N
            end = buf_ptr + chunk
            buf_fp[buf_ptr:end] = fp.reshape(chunk, 80, P, P).half()
            buf_cp[buf_ptr:end] = cp.reshape(chunk, 48, P, P).half()
            buf_f[buf_ptr:end] = feats.reshape(chunk, BOT_FEAT_DIM)
            buf_tv[buf_ptr:end] = tv.reshape(chunk, 4)
            buf_tb[buf_ptr:end] = tb.reshape(chunk, 6)
            buf_te[buf_ptr:end] = te.reshape(chunk)
            buf_ptr = end % cap_n
            buf_size = min(cap_n, buf_size + chunk)

            next_feats, _, done = arena.step(act_v, act_b, act_e)
            total_bot_steps += chunk
            feats = arena.reset(seed=100 + step * 37) if done else next_feats

        # Run 10 fast AdamW updates per DAgger tick on decorrelated replay buffer (exact calibrated BCE!)
        policy.train()
        last_loss = 0.0
        mb = min(512, buf_size)
        for _ in range(grad_steps_per_tick):
            idx = torch.randint(0, buf_size, (mb,), device=dev)
            optimizer.zero_grad(set_to_none=True)
            pv, pb, pe, _, _ = policy.forward_from_patches(
                buf_fp[idx].float(), buf_cp[idx].float(), buf_f[idx].float(), return_saliency=False
            )
            l_mv = F.smooth_l1_loss(pv[:, :2], buf_tv[idx, :2], beta=0.1) + 0.40 * (
                1.0 - F.cosine_similarity(pv[:, :2], buf_tv[idx, :2], dim=-1, eps=1e-5)
            ).mean()
            l_aim = F.smooth_l1_loss(pv[:, 2:], buf_tv[idx, 2:], beta=0.1) + 0.50 * (
                1.0 - F.cosine_similarity(pv[:, 2:], buf_tv[idx, 2:], dim=-1, eps=1e-5)
            ).mean()
            l_bin = F.binary_cross_entropy_with_logits(pb, buf_tb[idx])
            l_elem = F.cross_entropy(pe, buf_te[idx])
            loss = 2.5 * l_mv + 2.5 * l_aim + 2.0 * l_bin + 1.5 * l_elem
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.5)
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.item())

        if step % 30 == 0 or step == num_steps:
            ev = quick_det_eval_v2(policy, v1_rl_policy, val_arena, seed=71000 + step)
            sc_m = 3.0 * ev["wr_teacher"] + 1.5 * ev["wr_v1"] + 0.25 * (ev["score_v2"] - ev["score_t"]) + 10.0 * ev["relics"] - 18.0 * ev["burn"]
            if sc_m > best_score:
                best_score = sc_m
                best_sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            print(
                f"  [DAgger] step {step:3d}/{num_steps} | beta={beta:.2f} | loss={last_loss:.3f} | "
                f"vsTeach={ev['wr_teacher']:5.1f}% ({ev['score_v2']:.0f}v{ev['score_t']:.0f}) | "
                f"vsRLv1={ev['wr_v1']:5.1f}% | shrines={ev['shrines']:.1f} relics={ev['relics']:.2f} "
                f"combos={ev['combos']:.1f} burn={ev['burn']:.2f} | t={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    policy.load_state_dict(best_sd)
    ev_final = quick_det_eval_v2(policy, v1_rl_policy, val_arena, seed=72000)
    elapsed = round(time.perf_counter() - t0, 2)
    print(
        f"  [DAgger BEST] vsTeach={ev_final['wr_teacher']:5.1f}% ({ev_final['score_v2']:.0f}v{ev_final['score_t']:.0f}) | "
        f"vsRLv1={ev_final['wr_v1']:5.1f}% | shrines={ev_final['shrines']:.1f} relics={ev_final['relics']:.2f} "
        f"combos={ev_final['combos']:.1f} burn={ev_final['burn']:.2f} | {elapsed:.1f}s",
        flush=True,
    )
    return {
        "steps": num_steps,
        "grad_steps": total_grad_steps,
        "total_bot_steps": total_bot_steps,
        "elapsed_s": elapsed,
        "val_after_dagger": {k: round(v, 2) for k, v in ev_final.items()},
    }


def run_ppo_v2_stage(
    stage_name: str,
    ac: RLBotActorCriticV2,
    anchor_policy: BotPolicyNetV2,
    v1_rl_policy: BotPolicyNet,
    arena: FrostmarshGPUArena,
    val_arena: FrostmarshGPUArena,
    optimizer: torch.optim.Optimizer,
    num_updates: int,
    rollout_steps: int,
    ppo_epochs: int,
    minibatch_size: int,
    kl_anchor_coef: float,
    kl_early_stop: float,
    aim_aux_coef: float,
    seed_offset: int,
    critic_warmup_updates: int = 0,
) -> tuple[dict, dict]:
    """
    Trust-Region Float32-Safe Dual-Horizon PPO Self-Play:
      - KL early-stop (target 0.35)
      - Best-checkpoint selection on held-out validation arenas
      - Optimizes Shrine + Relic + Combo + Kill + Fire-Evasion + Team Score Differential
      - Opponents: 50% Scripted Teacher (side-balanced), 25% Lagged-Self, 25% Mirror-Self
    """
    A, N = arena.A, arena.N
    P = ac.policy.patch_size
    dev = arena.device

    lagged_policy = copy.deepcopy(ac.policy).eval()
    for p in lagged_policy.parameters():
        p.requires_grad_(False)

    # Opponent arena splits (side-balanced!)
    n_teach = max(2, (A // 2) & ~1)
    n_rem = A - n_teach
    n_lag = max(2, (n_rem // 2) & ~1) if n_rem >= 4 else max(0, n_rem // 2)
    n_mirror = A - n_teach - n_lag

    t_mid = n_teach // 2
    l_start = n_teach
    l_mid = l_start + n_lag // 2
    m_start = l_start + n_lag

    rl_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    rl_mask[:t_mid, :6] = True
    rl_mask[t_mid:n_teach, 6:] = True
    if n_lag > 0:
        rl_mask[l_start:l_mid, :6] = True
        rl_mask[l_mid:m_start, 6:] = True
    if n_mirror > 0:
        rl_mask[m_start:, :] = True

    opp_teach_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_teach_mask[:t_mid, 6:] = True
    opp_teach_mask[t_mid:n_teach, :6] = True

    opp_lag_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    if n_lag > 0:
        opp_lag_mask[l_start:l_mid, 6:] = True
        opp_lag_mask[l_mid:m_start, :6] = True

    rl_mask_t = rl_mask.unsqueeze(0).expand(rollout_steps, A, N)

    t0 = time.perf_counter()
    feats = arena.reset(seed=seed_offset)
    total_bot_steps = 0
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    # Evaluate starting checkpoint as initial best
    ev0 = quick_det_eval_v2(ac.policy, v1_rl_policy, val_arena, seed=seed_offset + 50000)
    best_metric = (
        4.0 * ev0["wr_teacher"]
        + 1.5 * ev0["wr_v1"]
        + 0.30 * (ev0["score_v2"] - ev0["score_t"])
        + 12.0 * ev0["relics"]
        - 22.0 * ev0["burn"]
    )
    best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
    best_ev = ev0
    print(
        f"[{stage_name}] init | vsTeach={ev0['wr_teacher']:5.1f}% ({ev0['score_v2']:.0f}v{ev0['score_t']:.0f}) | "
        f"vsRLv1={ev0['wr_v1']:5.1f}% | shrines={ev0['shrines']:.1f} relics={ev0['relics']:.2f} "
        f"combos={ev0['combos']:.1f} kills={ev0['kills']:.1f} burn={ev0['burn']:.2f}",
        flush=True,
    )

    buf_fp = torch.empty((rollout_steps, A, N, 80, P, P), dtype=torch.float16, device=dev)
    buf_cp = torch.empty((rollout_steps, A, N, 48, P, P), dtype=torch.float16, device=dev)
    buf_feats = torch.empty((rollout_steps, A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)
    buf_avec = torch.empty((rollout_steps, A, N, 4), dtype=torch.float32, device=dev)
    buf_abin = torch.empty((rollout_steps, A, N, 6), dtype=torch.float32, device=dev)
    buf_aelem = torch.empty((rollout_steps, A, N), dtype=torch.long, device=dev)
    buf_lp_mv = torch.empty((rollout_steps, A, N), dtype=torch.float32, device=dev)
    buf_lp_aim = torch.empty((rollout_steps, A, N), dtype=torch.float32, device=dev)
    buf_lp_bin = torch.empty((rollout_steps, A, N, 6), dtype=torch.float32, device=dev)
    buf_lp_elem = torch.empty((rollout_steps, A, N), dtype=torch.float32, device=dev)
    buf_val = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_rew = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_done = torch.empty((rollout_steps,), dtype=torch.float32, device=dev)
    buf_ref_v = torch.empty((rollout_steps, A, N, 4), dtype=torch.float32, device=dev)
    buf_ref_b = torch.empty((rollout_steps, A, N, 6), dtype=torch.float32, device=dev)
    buf_ref_e = torch.empty((rollout_steps, A, N, 3), dtype=torch.float32, device=dev)

    kl_stops = 0

    for upd in range(num_updates):
        if upd > 0 and upd % 6 == 0:
            lagged_policy.load_state_dict(ac.policy.state_dict())

        ac.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                fp, cp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
                buf_fp[t] = fp.half()
                buf_cp[t] = cp.half()
                buf_feats[t] = feats

                rl_v, rl_b, rl_e, lp_mv, lp_aim, lp_b, lp_e, rl_val = ac.sample_actions_from_patches(
                    fp, cp, feats, deterministic=False
                )

                rv, rb, re_, _, _ = anchor_policy.forward_from_patches(
                    fp.reshape(A * N, 80, P, P),
                    cp.reshape(A * N, 48, P, P),
                    feats.reshape(A * N, BOT_FEAT_DIM),
                    return_saliency=False,
                )
                buf_ref_v[t] = rv.reshape(A, N, 4)
                buf_ref_b[t] = rb.reshape(A, N, 6)
                buf_ref_e[t] = re_.reshape(A, N, 3)

                act_vec = rl_v.clamp(-1.0, 1.0)
                act_bin = rl_b.clone()
                act_elem = rl_e.clone()

                if opp_lag_mask.any():
                    lv, lb, le, _, _ = lagged_policy.forward_from_patches(
                        fp.reshape(A * N, 80, P, P),
                        cp.reshape(A * N, 48, P, P),
                        feats.reshape(A * N, BOT_FEAT_DIM),
                        return_saliency=False,
                    )
                    act_vec = torch.where(opp_lag_mask.unsqueeze(-1), lv.reshape(A, N, 4), act_vec)
                    act_bin = torch.where(opp_lag_mask.unsqueeze(-1), (torch.sigmoid(lb.reshape(A, N, 6)) > 0.5).float(), act_bin)
                    act_elem = torch.where(opp_lag_mask, le.reshape(A, N, 3).argmax(dim=-1), act_elem)

                if opp_teach_mask.any():
                    tv, tb, te = scripted_teacher_act(arena)
                    act_vec = torch.where(opp_teach_mask.unsqueeze(-1), tv, act_vec)
                    act_bin = torch.where(opp_teach_mask.unsqueeze(-1), tb, act_bin)
                    act_elem = torch.where(opp_teach_mask, te, act_elem)

                buf_avec[t] = rl_v
                buf_abin[t] = rl_b
                buf_aelem[t] = rl_e
                buf_lp_mv[t] = lp_mv
                buf_lp_aim[t] = lp_aim
                buf_lp_bin[t] = lp_b
                buf_lp_elem[t] = lp_e
                buf_val[t] = rl_val

                # Pre-step team score, carry position & relic distance for training-only match-winning shaping
                prev_score = arena.team_score.clone()
                prev_px = arena.pos_x.clone()
                car_pre = arena.carrying.clone()
                no_car_pre = (arena.relic_carrier < 0).unsqueeze(1) & (arena.cls == 0) & arena.alive
                rdist_pre = torch.hypot(arena.pos_x - arena.relic_x[:, None], arena.pos_y - arena.relic_y[:, None])

                next_feats, rews_pair, done = arena.step(act_vec, act_bin, act_elem)

                # Training-only match-score differential & objective tempo shaping (arena scoring untouched!)
                d_score = arena.team_score - prev_score
                d_my = d_score[:, 0:1] * arena.w_mask + d_score[:, 1:2] * arena.e_mask
                d_op = d_score[:, 1:2] * arena.w_mask + d_score[:, 0:1] * arena.e_mask
                rdist_post = torch.hypot(arena.pos_x - arena.relic_x[:, None], arena.pos_y - arena.relic_y[:, None])
                relic_approach = ((rdist_pre - rdist_post).clamp(-6.0, 6.0)) * no_car_pre.float()
                home_x = torch.where(arena.team == 0, 28.0, 228.0)
                carry_prog = ((prev_px - home_x).abs() - (arena.pos_x - home_x).abs()).clamp(-6.0, 6.0) * car_pre.float()

                rews_pair = rews_pair.clone()
                rews_pair[..., 0] = (
                    rews_pair[..., 0]
                    + 0.22 * (d_my - d_op)
                    + 0.10 * relic_approach
                    + 0.14 * carry_prog
                )

                buf_rew[t] = rews_pair
                buf_done[t] = 1.0 if done else 0.0
                total_bot_steps += A * N

                feats = arena.reset(seed=seed_offset + (upd + 1) * 131 + t) if done else next_feats

            lfp, lcp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
            _, _, _, _, _, _, _, last_val = ac.sample_actions_from_patches(lfp, lcp, feats, deterministic=True)

        # Dual-Horizon GAE (FP32)
        adv = torch.zeros_like(buf_rew)
        for h_i, (gam, lam) in enumerate(((0.97, 0.95), (0.72, 0.85))):
            gae = torch.zeros((A, N), device=dev)
            for t in reversed(range(rollout_steps)):
                nonterm = 1.0 - buf_done[t]
                next_v = last_val[..., h_i] if t == rollout_steps - 1 else buf_val[t + 1, ..., h_i]
                delta = buf_rew[t, ..., h_i] + gam * next_v * nonterm - buf_val[t, ..., h_i]
                gae = delta + gam * lam * nonterm * gae
                adv[t, ..., h_i] = gae
        ret = adv + buf_val

        flat_fp = buf_fp[rl_mask_t]
        flat_cp = buf_cp[rl_mask_t]
        flat_feats = buf_feats[rl_mask_t]
        flat_avec = buf_avec[rl_mask_t]
        flat_abin = buf_abin[rl_mask_t]
        flat_aelem = buf_aelem[rl_mask_t]
        flat_old_lpmv = buf_lp_mv[rl_mask_t]
        flat_old_lpaim = buf_lp_aim[rl_mask_t]
        flat_old_lpb = buf_lp_bin[rl_mask_t]
        flat_old_lpe = buf_lp_elem[rl_mask_t]

        flat_adv = adv[rl_mask_t]
        adv_tac = (flat_adv[:, 0] - flat_adv[:, 0].mean()) / (flat_adv[:, 0].std() + 1e-6)
        adv_cmb = (flat_adv[:, 1] - flat_adv[:, 1].mean()) / (flat_adv[:, 1].std() + 1e-6)
        flat_ret = ret[rl_mask_t]

        flat_ref_v = buf_ref_v[rl_mask_t]
        flat_ref_b = buf_ref_b[rl_mask_t]
        flat_ref_e = buf_ref_e[rl_mask_t]

        M = flat_feats.shape[0]
        critic_only = upd < critic_warmup_updates

        ac.train()
        early_stopped = False
        max_kl_seen = 0.0
        for ep in range(ppo_epochs):
            perm = torch.randperm(M, device=dev)
            ep_kls = []
            for start in range(0, M, minibatch_size):
                idx = perm[start : start + minibatch_size]
                optimizer.zero_grad(set_to_none=True)

                lpmv, lpaim, lpb, lpe, ent_mv, ent_aim, ent_b, ent_e, vals, vec_p, bin_p, elem_p = ac.evaluate_flat_minibatch(
                    flat_fp[idx], flat_cp[idx], flat_feats[idx], flat_avec[idx], flat_abin[idx], flat_aelem[idx]
                )
                v_loss = F.smooth_l1_loss(vals, flat_ret[idx], beta=5.0)
                aim_aux = compute_observable_aim_aux_loss(vec_p, flat_feats[idx])

                if critic_only:
                    loss = v_loss
                else:
                    # Trust-region approx KL across joint action heads
                    log_r_mv = (lpmv - flat_old_lpmv[idx]).clamp(-4.0, 4.0)
                    log_r_aim = (lpaim - flat_old_lpaim[idx]).clamp(-4.0, 4.0)
                    log_r_b = (lpb - flat_old_lpb[idx]).clamp(-4.0, 4.0)
                    log_r_e = (lpe - flat_old_lpe[idx]).clamp(-4.0, 4.0)

                    log_r_joint = log_r_mv + 0.25 * log_r_aim + log_r_b.mean(dim=-1) + log_r_e
                    approx_kl = float((0.5 * (log_r_joint ** 2)).mean().item())
                    ep_kls.append(approx_kl)
                    if approx_kl > max_kl_seen:
                        max_kl_seen = approx_kl
                    if approx_kl > kl_early_stop:
                        early_stopped = True
                        kl_stops += 1
                        break

                    a_t = adv_tac[idx]
                    a_c = adv_cmb[idx]

                    r_mv = torch.exp(log_r_mv)
                    pg_mv = -torch.minimum(r_mv * a_t, r_mv.clamp(0.84, 1.16) * a_t).mean()

                    r_aim = torch.exp(log_r_aim)
                    pg_aim = -torch.minimum(r_aim * a_c, r_aim.clamp(0.84, 1.16) * a_c).mean()

                    a_bin = torch.stack([a_t, a_c, a_c, a_t, a_c, a_c], dim=-1)
                    r_b = torch.exp(log_r_b)
                    pg_b = -torch.minimum(r_b * a_bin, r_b.clamp(0.84, 1.16) * a_bin).mean()

                    r_e = torch.exp(log_r_e)
                    pg_e = -torch.minimum(r_e * a_c, r_e.clamp(0.84, 1.16) * a_c).mean()

                    pg_loss = pg_mv + 0.15 * pg_aim + pg_b + pg_e
                    ent_loss = -(0.0015 * (ent_mv.mean() + ent_aim.mean()) + 0.010 * ent_b.mean() + 0.015 * ent_e.mean())
                    logit_reg = 8e-4 * ((bin_p ** 2).mean() + (elem_p ** 2).mean())
                    kl_anc = compute_policy_kl_fp32(
                        vec_p, bin_p, elem_p, flat_ref_v[idx], flat_ref_b[idx], flat_ref_e[idx]
                    ).mean()

                    loss = pg_loss + 0.5 * v_loss + aim_aux_coef * aim_aux + ent_loss + logit_reg + kl_anchor_coef * kl_anc

                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                optimizer.step()

            if early_stopped or (ep_kls and float(np.mean(ep_kls)) > kl_early_stop):
                if not early_stopped:
                    kl_stops += 1
                break

        # Cosine LR decay after critic warmup
        if upd >= critic_warmup_updates:
            prog = (upd - critic_warmup_updates + 1) / max(1, num_updates - critic_warmup_updates)
            lr_mult = 0.25 + 0.75 * 0.5 * (1.0 + np.cos(np.pi * prog))
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * lr_mult

        eval_every = max(1, min(6, num_updates // 4))
        if (upd + 1) % eval_every == 0 or upd == num_updates - 1:
            elapsed = time.perf_counter() - t0
            ev = quick_det_eval_v2(ac.policy, v1_rl_policy, val_arena, seed=seed_offset + 50000)
            metric = (
                4.0 * ev["wr_teacher"]
                + 1.5 * ev["wr_v1"]
                + 0.30 * (ev["score_v2"] - ev["score_t"])
                + 12.0 * ev["relics"]
                - 22.0 * ev["burn"]
            )
            tag = ""
            if metric > best_metric and not critic_only:
                best_metric = metric
                best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
                best_ev = ev
                tag = " *BEST*"
            print(
                f"[{stage_name}] upd {upd+1:3d}/{num_updates} | t={elapsed:5.1f}s | kl={max_kl_seen:.3f} | "
                f"vsTeach={ev['wr_teacher']:5.1f}% ({ev['score_v2']:.0f}v{ev['score_t']:.0f}) | "
                f"vsRLv1={ev['wr_v1']:5.1f}% | shrines={ev['shrines']:.1f} relics={ev['relics']:.2f} "
                f"combos={ev['combos']:.1f} kills={ev['kills']:.1f} burn={ev['burn']:.2f}{tag}",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    stats = {
        "stage": stage_name,
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "parallel_arenas": A,
        "concurrent_bots": A * N,
        "updates": num_updates,
        "total_bot_steps": total_bot_steps,
        "elapsed_s": elapsed,
        "bot_steps_per_sec": round(total_bot_steps / max(elapsed, 1e-3), 1),
        "kl_early_stop_threshold": kl_early_stop,
        "kl_early_stops_triggered": kl_stops,
        "best_val": {k: round(v, 2) for k, v in best_ev.items()},
    }
    return stats, best_sd


def _agent_actions_any(
    agent_kind: str,
    policy: nn.Module | None,
    arena: FrostmarshGPUArena,
    feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if agent_kind == "teacher":
        return scripted_teacher_act(arena)
    if agent_kind == "v2":
        assert isinstance(policy, BotPolicyNetV2)
        return v2_deterministic_act(policy, arena.z_world, arena.z_eco, feats)
    assert isinstance(policy, BotPolicyNet)
    return v1_policy_deterministic_act(policy, arena.z_world, arena.z_eco, feats)


@torch.inference_mode()
def run_tournament_matchup(
    name_a: str,
    kind_a: str,
    policy_a: nn.Module | None,
    name_b: str,
    kind_b: str,
    policy_b: nn.Module | None,
    arena: FrostmarshGPUArena,
    num_matches: int = 24,
    ticks: int = 120,
    base_seed: int = 91000,
) -> dict:
    """Exact same 24-match side-balanced 256x256x4 evaluation protocol as eval_bot_arena.py."""
    assert num_matches % 2 == 0
    half = num_matches // 2
    batch = arena.A
    assert half % batch == 0

    a_scores, b_scores = [], []
    a_shrines, b_shrines = [], []
    a_combos, b_combos = [], []
    a_burns, b_burns = [], []
    a_kills, b_kills = [], []
    a_deaths, b_deaths = [], []
    a_relics, b_relics = [], []

    t0 = time.perf_counter()
    for side in (0, 1):
        for bi in range(half // batch):
            seed = base_seed + bi * 1000
            feats = arena.reset(seed=seed)
            a_mask = arena.team == side

            for _ in range(ticks):
                av_a, ab_a, ae_a = _agent_actions_any(kind_a, policy_a, arena, feats)
                av_b, ab_b, ae_b = _agent_actions_any(kind_b, policy_b, arena, feats)

                act_vec = torch.where(a_mask.unsqueeze(-1), av_a, av_b)
                act_bin = torch.where(a_mask.unsqueeze(-1), ab_a, ab_b)
                act_elem = torch.where(a_mask, ae_a, ae_b)

                feats, _, _ = arena.step(act_vec, act_bin, act_elem)

            ia, ib = side, 1 - side
            a_scores.append(arena.team_score[:, ia].cpu().numpy())
            b_scores.append(arena.team_score[:, ib].cpu().numpy())
            a_shrines.append(arena.team_shrine_caps[:, ia].cpu().numpy())
            b_shrines.append(arena.team_shrine_caps[:, ib].cpu().numpy())
            a_combos.append(arena.team_combos[:, ia].cpu().numpy())
            b_combos.append(arena.team_combos[:, ib].cpu().numpy())
            a_burns.append(arena.team_burn_taken[:, ia].cpu().numpy())
            b_burns.append(arena.team_burn_taken[:, ib].cpu().numpy())
            a_kills.append(arena.team_kills[:, ia].cpu().numpy())
            b_kills.append(arena.team_kills[:, ib].cpu().numpy())
            a_deaths.append(arena.team_deaths[:, ia].cpu().numpy())
            b_deaths.append(arena.team_deaths[:, ib].cpu().numpy())
            a_relics.append(arena.team_relic_deliveries[:, ia].cpu().numpy())
            b_relics.append(arena.team_relic_deliveries[:, ib].cpu().numpy())

    elapsed = time.perf_counter() - t0
    sa = np.concatenate(a_scores)
    sb = np.concatenate(b_scores)
    wins_a = int(np.sum(sa > sb))
    wins_b = int(np.sum(sb > sa))
    draws = int(np.sum(sa == sb))

    res = {
        "matchup": f"{name_a} vs {name_b}",
        "team_a": name_a,
        "team_b": name_b,
        "matches": num_matches,
        "ticks_per_match": ticks,
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a_pct": round(100.0 * (wins_a + 0.5 * draws) / num_matches, 2),
        "win_rate_b_pct": round(100.0 * (wins_b + 0.5 * draws) / num_matches, 2),
        "mean_score_a": round(float(sa.mean()), 2),
        "mean_score_b": round(float(sb.mean()), 2),
        "shrine_captures_per_match_a": round(float(np.concatenate(a_shrines).mean()), 2),
        "shrine_captures_per_match_b": round(float(np.concatenate(b_shrines).mean()), 2),
        "physics_combos_per_match_a": round(float(np.concatenate(a_combos).mean()), 2),
        "physics_combos_per_match_b": round(float(np.concatenate(b_combos).mean()), 2),
        "wildfire_burn_taken_per_match_a": round(float(np.concatenate(a_burns).mean()), 3),
        "wildfire_burn_taken_per_match_b": round(float(np.concatenate(b_burns).mean()), 3),
        "kills_per_match_a": round(float(np.concatenate(a_kills).mean()), 2),
        "kills_per_match_b": round(float(np.concatenate(b_kills).mean()), 2),
        "deaths_per_match_a": round(float(np.concatenate(a_deaths).mean()), 2),
        "deaths_per_match_b": round(float(np.concatenate(b_deaths).mean()), 2),
        "relic_deliveries_per_match_a": round(float(np.concatenate(a_relics).mean()), 2),
        "relic_deliveries_per_match_b": round(float(np.concatenate(b_relics).mean()), 2),
        "eval_time_s": round(elapsed, 2),
    }

    print(
        f"  [{res['matchup']:42s}] WR={res['win_rate_a_pct']:5.1f}% ({wins_a}W-{wins_b}L-{draws}D) | "
        f"Score {res['mean_score_a']:6.1f} vs {res['mean_score_b']:6.1f} | "
        f"Shrines {res['shrine_captures_per_match_a']:4.1f} vs {res['shrine_captures_per_match_b']:4.1f} | "
        f"Relics {res['relic_deliveries_per_match_a']:4.2f} vs {res['relic_deliveries_per_match_b']:4.2f} | "
        f"Combos {res['physics_combos_per_match_a']:5.1f} vs {res['physics_combos_per_match_b']:5.1f} | "
        f"Burn {res['wildfire_burn_taken_per_match_a']:5.2f} vs {res['wildfire_burn_taken_per_match_b']:5.2f} | "
        f"K/D {res['kills_per_match_a']:4.1f}/{res['deaths_per_match_a']:4.1f}",
        flush=True,
    )
    return res


@torch.inference_mode()
def benchmark_v2_latency(policy: BotPolicyNetV2, device: torch.device) -> dict:
    """Benchmark GB10 batched inference latency for BotPolicyNetV2 at N=12, 64, 256 bots."""
    policy.eval()
    z_world = torch.randn((4, 48, 256, 256), dtype=torch.float16, device=device)
    z_eco = torch.randn((1, 32, 256, 256), dtype=torch.float16, device=device)
    results = {}

    for n_bots in (12, 64, 256):
        bf = torch.rand((n_bots, BOT_FEAT_DIM), dtype=torch.float32, device=device)
        # Warmup
        for _ in range(30):
            _ = policy(z_world, z_eco, bf, return_saliency=False)
        torch.cuda.synchronize()

        times_std = []
        for _ in range(150):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = policy(z_world, z_eco, bf, return_saliency=False)
            e.record()
            torch.cuda.synchronize()
            times_std.append(s.elapsed_time(e))

        times_sal = []
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out_s, sal = policy(z_world, z_eco, bf, return_saliency=True)
            e.record()
            torch.cuda.synchronize()
            times_sal.append(s.elapsed_time(e))

        assert out.shape == (n_bots, ACTION_OUT_DIM)
        assert out_s.shape == (n_bots, ACTION_OUT_DIM) and sal.shape == (n_bots, SALIENCY_DIM)

        arr = np.array(times_std)
        arr_s = np.array(times_sal)
        results[f"bots_{n_bots}"] = {
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "mean_ms": round(float(arr.mean()), 3),
            "with_saliency_p50_ms": round(float(np.percentile(arr_s, 50)), 3),
            "with_saliency_p95_ms": round(float(np.percentile(arr_s, 95)), 3),
        }
        print(
            f"  [Latency N={n_bots:3d}] p50={results[f'bots_{n_bots}']['p50_ms']:.3f}ms "
            f"p95={results[f'bots_{n_bots}']['p95_ms']:.3f}ms "
            f"(+saliency p50={results[f'bots_{n_bots}']['with_saliency_p50_ms']:.3f}ms)",
            flush=True,
        )
    return results


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(2026)

    t_global = time.perf_counter()
    print("=== Dual-Scale Z_world BotPolicyNetV2 + Self-Play RL on GB10 ===", flush=True)

    wm, em = load_frozen_world_models(device)

    # Load v1 baselines for validation & held-out tournament
    v1_rl_net = BotPolicyNet(blind_world=False).to(device).eval()
    v1_rl_net.load_state_dict(load_ckpt_state_dict(RL_PT, device), strict=True)
    for p in v1_rl_net.parameters():
        p.requires_grad_(False)

    bc_net = BotPolicyNet(blind_world=False).to(device).eval()
    bc_net.load_state_dict(load_ckpt_state_dict(BC_PT, device), strict=True)
    for p in bc_net.parameters():
        p.requires_grad_(False)

    blind_net = BotPolicyNet(blind_world=True).to(device).eval()
    blind_net.load_state_dict(load_ckpt_state_dict(BC_PT, device), strict=True)
    for p in blind_net.parameters():
        p.requires_grad_(False)

    # Instantiate BotPolicyNetV2 & verify drop-in API + batched equivalence
    policy_v2 = BotPolicyNetV2(
        z_world_ch=48,
        z_eco_ch=32,
        patch_size=16,
        fine_radius_cells=14.0,
        coarse_radius_cells=60.0,
        d_model=64,
        blind_world=False,
    ).to(device)
    n_params_v2 = count_parameters(policy_v2)

    with torch.inference_mode():
        tz_w = torch.randn((1, 4, 48, 64, 64), device=device)
        tz_e = torch.randn((1, 1, 32, 64, 64), device=device)
        tf = torch.rand((1, 12, BOT_FEAT_DIM), device=device)
        a_single, sal_single = policy_v2(tz_w[0], tz_e[0], tf[0], return_saliency=True)
        assert a_single.shape == (12, ACTION_OUT_DIM) and sal_single.shape == (12, SALIENCY_DIM)
        v1, b1, e1 = policy_v2.forward_logits(tz_w[0], tz_e[0], tf[0])
        v2, b2, e2, _ = batched_v2_forward(policy_v2, tz_w, tz_e, tf)
        diff = max((v1 - v2[0]).abs().max().item(), (b1 - b2[0]).abs().max().item(), (e1 - e2[0]).abs().max().item())
        assert diff < 1e-4, f"Batched V2 forward mismatch: {diff}"
    print(f"Verified BotPolicyNetV2 drop-in API & batched equivalence: max_diff={diff:.2e} | params={n_params_v2:,}", flush=True)

    # Create 32-arena 128x128x2 training pool + 16-arena validation pool
    arena_128 = FrostmarshGPUArena(32, wm, em, device, grid_size=128, num_layers=2, max_ticks=120, latent_refresh_every=4)
    val_arena_128 = FrostmarshGPUArena(16, wm, em, device, grid_size=128, num_layers=2, max_ticks=120, latent_refresh_every=4)

    # Stage 0: Fast Online DAgger Calibration (150 steps on 32 parallel arenas)
    stage0_stats = run_stage0_online_dagger(policy_v2, arena_128, val_arena_128, v1_rl_net, num_steps=150)

    # Freeze Stage 0 anchor policy for optional gentle regularization
    anchor_policy = copy.deepcopy(policy_v2).eval()
    for p in anchor_policy.parameters():
        p.requires_grad_(False)

    # Stage 1: Trust-Region Float32-Safe PPO Self-Play (lr=8e-5, KL early-stop=0.35)
    ac = RLBotActorCriticV2(policy_v2).to(device)
    opt1 = torch.optim.AdamW(
        [
            {"params": ac.policy.parameters(), "lr": 8.0e-5, "weight_decay": 1e-4},
            {"params": ac.value_head.parameters(), "lr": 1.5e-3, "weight_decay": 1e-4},
            {"params": [ac.log_std], "lr": 1.5e-4},
        ]
    )
    stage1_stats, best_sd1 = run_ppo_v2_stage(
        stage_name="Stage1-128x128x2-PPO (32x12=384 bots)",
        ac=ac,
        anchor_policy=anchor_policy,
        v1_rl_policy=v1_rl_net,
        arena=arena_128,
        val_arena=val_arena_128,
        optimizer=opt1,
        num_updates=60,
        rollout_steps=30,
        ppo_epochs=4,
        minibatch_size=512,
        kl_anchor_coef=0.0,
        kl_early_stop=0.35,
        aim_aux_coef=1.5,
        seed_offset=2000,
        critic_warmup_updates=3,
    )
    ac.policy.load_state_dict(best_sd1)
    del arena_128, val_arena_128
    torch.cuda.empty_cache()

    # Stage 2: High-Res 256x256x4 PPO Self-Play Polish (8 parallel arenas = 96 bots)
    arena_256 = FrostmarshGPUArena(8, wm, em, device, grid_size=256, num_layers=4, max_ticks=120, latent_refresh_every=4)
    val_arena_256 = FrostmarshGPUArena(8, wm, em, device, grid_size=256, num_layers=4, max_ticks=120, latent_refresh_every=4)
    opt2 = torch.optim.AdamW(
        [
            {"params": ac.policy.parameters(), "lr": 5.0e-5, "weight_decay": 1e-4},
            {"params": ac.value_head.parameters(), "lr": 8.0e-4, "weight_decay": 1e-4},
            {"params": [ac.log_std], "lr": 8.0e-5},
        ]
    )
    stage2_stats, best_sd2 = run_ppo_v2_stage(
        stage_name="Stage2-256x256x4-Polish (8x12=96 bots)",
        ac=ac,
        anchor_policy=anchor_policy,
        v1_rl_policy=v1_rl_net,
        arena=arena_256,
        val_arena=val_arena_256,
        optimizer=opt2,
        num_updates=12,
        rollout_steps=30,
        ppo_epochs=4,
        minibatch_size=512,
        kl_anchor_coef=0.0,
        kl_early_stop=0.35,
        aim_aux_coef=1.2,
        seed_offset=60000,
        critic_warmup_updates=0,
    )
    ac.policy.load_state_dict(best_sd2)
    ac.policy.eval()
    del arena_256, val_arena_256
    torch.cuda.empty_cache()

    train_time_s = round(time.perf_counter() - t_global, 2)

    # Save pure BotPolicyNetV2 state_dict()
    pure_sd = {k: v.detach().cpu() for k, v in ac.policy.state_dict().items()}
    verify_v2 = BotPolicyNetV2()
    verify_v2.load_state_dict(pure_sd, strict=True)

    torch.save(
        {
            "state_dict": pure_sd,
            "arch": {
                "class": "BotPolicyNetV2",
                "z_world_ch": 48,
                "z_eco_ch": 32,
                "patch_size": 16,
                "fine_radius_cells": 14.0,
                "coarse_radius_cells": 60.0,
                "d_model": 64,
                "n_params": n_params_v2,
            },
            "rl_training": {
                "total_time_s": train_time_s,
                "stage0_dagger": stage0_stats,
                "stage1_ppo_128": stage1_stats,
                "stage2_ppo_256": stage2_stats,
            },
        },
        V2_RL_PT,
    )
    print(f"\nSaved drop-in BotPolicyNetV2 checkpoint to {V2_RL_PT} (train_time={train_time_s:.1f}s)", flush=True)

    # Benchmark GB10 batched latency (12 / 64 / 256 bots)
    print("\n=== GB10 Batched Latency Benchmark (BotPolicyNetV2) ===", flush=True)
    latency_stats = benchmark_v2_latency(ac.policy, device)

    # Full 24-match held-out 256x256x4 Tournament (exact same protocol as eval_bot_arena.py!)
    print("\n=== Held-Out 256x256x4 6v6 Tournament (24 Side-Balanced Matches per Matchup) ===", flush=True)
    t_tourney = time.perf_counter()
    tourney_arena = FrostmarshGPUArena(
        num_arenas=12,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=256,
        num_layers=4,
        max_ticks=120,
        latent_refresh_every=2,
    )
    m_teacher = run_tournament_matchup(
        "BotPolicyNetV2-RL", "v2", ac.policy, "Scripted Teacher", "teacher", None, tourney_arena, num_matches=24, ticks=120, base_seed=91000
    )
    m_rl_v1 = run_tournament_matchup(
        "BotPolicyNetV2-RL", "v2", ac.policy, "BotPolicyNet-RL v1", "v1", v1_rl_net, tourney_arena, num_matches=24, ticks=120, base_seed=92000
    )
    m_bc = run_tournament_matchup(
        "BotPolicyNetV2-RL", "v2", ac.policy, "BotPolicyNet-BC", "v1", bc_net, tourney_arena, num_matches=24, ticks=120, base_seed=93000
    )
    m_blind = run_tournament_matchup(
        "BotPolicyNetV2-RL", "v2", ac.policy, "Blind Ledger-Only", "v1", blind_net, tourney_arena, num_matches=24, ticks=120, base_seed=94000
    )
    tourney_time_s = round(time.perf_counter() - t_tourney, 2)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "grid": "256x256x4",
        "matches_per_matchup": 24,
        "ticks_per_match": 120,
        "side_balanced": True,
        "policy_class": "BotPolicyNetV2",
        "policy_params": n_params_v2,
        "fine_radius_cells": 14.0,
        "coarse_radius_cells": 60.0,
        "drop_in_api_compatible": True,
        "checkpoint": str(V2_RL_PT),
        "latency_gb10": latency_stats,
        "training": {
            "total_time_s": train_time_s,
            "stage0_dagger": stage0_stats,
            "stage1_ppo_128": stage1_stats,
            "stage2_ppo_256": stage2_stats,
        },
        "tournament_time_s": tourney_time_s,
        "matchups": {
            "v2_rl_vs_teacher": m_teacher,
            "v2_rl_vs_v1_rl": m_rl_v1,
            "v2_rl_vs_bc": m_bc,
            "v2_rl_vs_blind": m_blind,
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_JSON} OK!", flush=True)


if __name__ == "__main__":
    main()
