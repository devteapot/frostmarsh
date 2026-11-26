#!/usr/bin/env python3
"""
Wave 41D: PufferLib-Inspired Recurrent + Task-Brief 60-Bot RL Trainer (`train_bot_rl_v3.py`).

Trains `BotPolicyNetV2` (64-unit GRU + 16-d NMMO 2.0 TaskBrief + 5-Role Embedding) directly on
Frostmarsh's native zero-alloc multi-threaded double-buffered Rust RL Arena (`RustVecArena` via
`libfield_core.so`), encoding real `field-core::Field` states through frozen `WorldModel` +
`EcosystemModel` on NVIDIA GB10 so `Z_world` / `Z_eco` latents match live `gpu_infer_daemon.py`
with zero distribution shift!

Pipeline:
  1. Benchmark pure-Rust `RustVecArena` throughput at 12 / 32 / 60 bots/arena.
  2. Stage 0 — Recurrent DAgger Calibration (warm-started from `bot_policy_v2_rl.pt` with zero-init
     brief/GRU residual bridges so baseline quality is preserved from step 0) on 16 parallel 60-bot
     arenas (960 concurrent bots!) with randomized NMMO 2.0 TaskBriefs.
  3. Stage 1 — Multi-Density (`12 / 32 / 60` bots/arena) Truncated-BPTT Recurrent PPO Self-Play
     against Scripted Teacher + Historical Checkpoint Pool, optimizing Shrine + Relic + Combat +
     NMMO 2.0 dense 0..1 Brief Predicates (`close_target`, `stance_ok`, `risk_bound`).
  4. Stage 2 — Shipped 60-Bot Dense Server Polish (16 arenas x 60 bots = 960 live concurrent bots).
  5. Evaluation Suite (saved to `BOT_RL_V3_TOURNAMENT.json`):
     - Frozen-Weights `System2` Brief-Obedience Test (hold weights fixed, flip brief between
       COMMIT/CAPTURE_SHRINE, HOLD/DEFEND_HOLD, FLEE_KITE, HUNT_ENEMY, ESCORT_RELIC).
     - 60-Bot Server-Merge Tournament: `BotPolicyNetV3-Recurrent-60` vs Scripted Teacher AND vs
       Legacy 12-Bot Feedforward `BotPolicyNetV2` (`bot_policy_v2_rl.pt`) at 60-bot live density!
     - GB10 Batched Latency Benchmark (`12 / 60 / 256` bots with 64-unit GRU state updates).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bot_policy_v2 import (
    ACTION_OUT_DIM,
    BOT_FEAT_DIM,
    BOT_FEAT_DIM_V3,
    BRIEF_DIM,
    GRU_HIDDEN_DIM,
    OBJ_CAPTURE_SHRINE,
    OBJ_DEFEND_HOLD,
    OBJ_ESCORT_RELIC,
    OBJ_FREE_ROAM,
    OBJ_HUNT_ENEMY,
    STANCE_COMMIT,
    STANCE_FLEE_KITE,
    STANCE_HOLD,
    BotPolicyNetV2,
    count_parameters,
    load_bot_policy_v2_or_v3,
    sample_batched_dual_patches,
)
from rust_vec_arena import (
    RL_ACTION_DIM,
    RL_BRIEF_IN_DIM,
    RL_OBS_DIM,
    RL_PRED_DIM,
    RustVecArena,
    find_field_core_lib,
)
from train_bot_rl import (
    ROOT,
    extract_eco_latent,
    extract_world_latent_and_delta,
    load_frozen_world_models,
)

V2_RL_PT = ROOT / "bot_policy_v2_rl.pt"
V3_RL_PT = ROOT / "bot_policy_v3_rl.pt"
OUT_JSON = ROOT / "BOT_RL_V3_TOURNAMENT.json"


def field_to_eco_input(w_tensor: torch.Tensor) -> torch.Tensor:
    """Derive 8-channel EcosystemModel input directly from [A, 18, H, W] Field tensor."""
    # Field channels: 0=water, 1=ice, 2=steam, 3=soil, 4=rock, 5=wood, 6=ash, 7=smoke, 8=oil, 9=plant, 10=temp
    return torch.stack(
        [
            w_tensor[:, 9],   # plant
            w_tensor[:, 0],   # water
            w_tensor[:, 8],   # oil
            w_tensor[:, 7],   # smoke
            w_tensor[:, 10],  # temp
            w_tensor[:, 3],   # soil
            w_tensor[:, 5],   # wood
            w_tensor[:, 4],   # rock
        ],
        dim=1,
    ).unsqueeze(1)  # [A, 1, 8, H, W]


@torch.inference_mode()
def encode_arena_latents(
    arena: RustVecArena,
    wm: nn.Module,
    em: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Export real Rust Field tensors -> frozen WorldModel & EcosystemModel latents on GPU."""
    w_np = arena.export_field_tensors()  # [A, 18, H, W]
    w_t = torch.from_numpy(w_np).to(device, non_blocking=True)
    w_5d = w_t.unsqueeze(1)  # [A, 1, 18, H, W]
    z_world, _ = extract_world_latent_and_delta(wm, w_5d)
    z_eco = extract_eco_latent(em, field_to_eco_input(w_t))
    return z_world, z_eco


class RecurrentActorCriticV3(nn.Module):
    """Wraps recurrent + brief-conditioned BotPolicyNetV2 with a float32 RL value head."""

    def __init__(self, policy: BotPolicyNetV2):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.85, -1.85, -2.10, -2.10], dtype=torch.float32))

    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-2.8, -1.2))

    def act_step(
        self,
        z_world: torch.Tensor,  # [A, 1, 48, H, W]
        z_eco: torch.Tensor,    # [A, 1, 32, H, W]
        obs: torch.Tensor,      # [A, N, 184]
        h_prev: torch.Tensor,   # [A*N, 64]
        reset_mask: torch.Tensor | None = None,  # [A*N]
        deterministic: bool = False,
    ):
        a, n, _ = obs.shape
        p = self.policy.patch_size
        fp, cp = sample_batched_dual_patches(self.policy, z_world, z_eco, obs)
        fp_flat = fp.reshape(a * n, -1, p, p)
        cp_flat = cp.reshape(a * n, -1, p, p)
        obs_flat = obs.reshape(a * n, RL_OBS_DIM)

        vec, bin_logits, elem_logits, h_trunk, _, h_next = self.policy.forward_from_patches(
            fp_flat,
            cp_flat,
            obs_flat,
            return_saliency=False,
            h_prev=h_prev,
            reset_mask=reset_mask,
            return_hidden=True,
        )
        val = self.value_head(h_trunk.float()).squeeze(-1)

        if deterministic:
            vec_act = vec
            bin_act = (torch.sigmoid(bin_logits) > 0.5).float()
            elem_idx = elem_logits.argmax(dim=-1)
            logp = torch.zeros_like(val)
        else:
            std = self.std().unsqueeze(0)
            dist_v = torch.distributions.Normal(vec.float(), std)
            vec_act = dist_v.sample().clamp(-1.0, 1.0)
            dist_b = torch.distributions.Bernoulli(logits=bin_logits.float())
            bin_act = dist_b.sample()
            dist_e = torch.distributions.Categorical(logits=elem_logits.float())
            elem_idx = dist_e.sample()
            logp = (
                dist_v.log_prob(vec_act).sum(dim=-1)
                + dist_b.log_prob(bin_act).sum(dim=-1)
                + dist_e.log_prob(elem_idx)
            )

        elem_oh = F.one_hot(elem_idx, num_classes=3).float()
        actions = torch.cat([vec_act.float(), bin_act.float(), elem_oh], dim=-1).reshape(a, n, RL_ACTION_DIM)
        return actions, logp.reshape(a, n), val.reshape(a, n), h_next, fp_flat, cp_flat

    def evaluate_sequence(
        self,
        fp_seq: torch.Tensor,     # [T, B, 80, P, P]
        cp_seq: torch.Tensor,     # [T, B, 48, P, P]
        obs_seq: torch.Tensor,    # [T, B, 184]
        done_seq: torch.Tensor,   # [T, B]
        h0: torch.Tensor,         # [B, 64]
        act_seq: torch.Tensor,    # [T, B, 13]
    ):
        vec, bin_logits, elem_logits, h_trunk, _ = self.policy.forward_sequence_from_patches(
            fp_seq, cp_seq, obs_seq, done_seq, h0
        )
        val = self.value_head(h_trunk.float()).squeeze(-1)
        std = self.std().view(1, 1, 4)

        vec_a = act_seq[..., :4].float()
        bin_a = act_seq[..., 4:10].float()
        elem_idx = act_seq[..., 10:13].argmax(dim=-1)

        dist_v = torch.distributions.Normal(vec.float(), std)
        dist_b = torch.distributions.Bernoulli(logits=bin_logits.float())
        dist_e = torch.distributions.Categorical(logits=elem_logits.float())

        logp = (
            dist_v.log_prob(vec_a).sum(dim=-1)
            + dist_b.log_prob(bin_a).sum(dim=-1)
            + dist_e.log_prob(elem_idx)
        )
        ent = (
            dist_v.entropy().sum(dim=-1)
            + dist_b.entropy().sum(dim=-1)
            + dist_e.entropy()
        )
        return logp, ent, val


def compute_teacher_targets_from_obs(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast GPU brief-aware teacher targets mirroring Rust scripted_teacher_action for Stage-0 DAgger
    calibration so the brief_enc + 64-unit GRU immediately learn crisp stance/objective adherence!
    """
    # obs: [M, 184]
    brief = obs[:, BOT_FEAT_DIM : BOT_FEAT_DIM + 15]
    brief_active = brief[:, 13:14] > 0.5
    stance_flee = brief[:, 5 + STANCE_FLEE_KITE : 5 + STANCE_FLEE_KITE + 1] > 0.5
    stance_hold = brief[:, 5 + STANCE_HOLD : 5 + STANCE_HOLD + 1] > 0.5
    obj_hunt = brief[:, OBJ_HUNT_ENEMY : OBJ_HUNT_ENEMY + 1] > 0.5

    # Nearest enemy from tokens 0..8
    toks = obs[:, 24 : 24 + 8 * 12].reshape(-1, 8, 12)
    is_enemy = (toks[:, :, 3] > 0.5) & (toks[:, :, 11] > 0.5)
    dists = torch.where(is_enemy, toks[:, :, 7], torch.full_like(toks[:, :, 7], 9.0))
    best_idx = dists.argmin(dim=-1)
    idx_exp = best_idx.view(-1, 1, 1).expand(-1, 1, 12)
    nearest_e = toks.gather(1, idx_exp).squeeze(1)
    e_dist = nearest_e[:, 7:8] * 100.0
    e_dir = nearest_e[:, 8:10]
    has_enemy = (dists.amin(dim=-1, keepdim=True) < 1.0)

    # Target vector from brief (dx/100, dy/100)
    tdx = brief[:, 8:10]
    t_norm = tdx.norm(dim=-1, keepdim=True).clamp(min=1e-4)
    t_dir = tdx / t_norm
    t_dist = brief[:, 10:11] * 100.0

    # Relic carrier runs home; otherwise contest nearest uncaptured shrine (or grab nearby relic)
    carrying_relic = obs[:, 12:13] > 0.5
    team_f = obs[:, 11:12]  # 0=West (home x=0.14), 1=East (home x=0.86)
    home_dx = torch.where(team_f < 0.5, 0.14 - obs[:, 0:1], 0.86 - obs[:, 0:1])
    home_dy = 0.50 - obs[:, 1:2]
    home_vec = torch.cat([home_dx, home_dy], dim=-1)
    home_dir = home_vec / home_vec.norm(dim=-1, keepdim=True).clamp(min=1e-4)

    shrines = obs[:, 24 + 8 * 12 : 24 + 11 * 12].reshape(-1, 3, 12)
    s_need = shrines[:, :, 3]
    s_d = shrines[:, :, :2].norm(dim=-1) * torch.where(s_need > 0.5, 1.0, 2.5)
    s_idx = s_d.argmin(dim=-1).view(-1, 1, 1).expand(-1, 1, 12)
    best_s = shrines.gather(1, s_idx).squeeze(1)
    s_vec = best_s[:, :2]
    s_dir = s_vec / s_vec.norm(dim=-1, keepdim=True).clamp(min=1e-4)

    relic_vec = obs[:, 24 + 11 * 12 : 24 + 11 * 12 + 2]
    relic_d = relic_vec.norm(dim=-1, keepdim=True)
    relic_dir = relic_vec / relic_d.clamp(min=1e-4)
    free_dir = torch.where(carrying_relic, home_dir, torch.where(relic_d < 0.18, relic_dir, s_dir))

    # Compute desired movement vector
    flee_vec = -e_dir
    hold_vec = torch.where(
        t_dist < 6.0,
        torch.where(e_dist < 10.0, -e_dir * 0.6, torch.zeros_like(t_dir)),
        t_dir,
    )
    hunt_vec = torch.where(has_enemy & (e_dist < 80.0), e_dir, t_dir)
    active_move = torch.where(
        stance_flee & has_enemy & (e_dist < 28.0),
        flee_vec,
        torch.where(stance_hold, hold_vec, torch.where(obj_hunt, hunt_vec, t_dir)),
    )
    move_tgt = torch.where(brief_active, active_move, free_dir)
    aim_tgt = torch.where(has_enemy & (e_dist < 60.0), e_dir, move_tgt)
    vec_tgt = torch.cat([move_tgt, aim_tgt], dim=-1).clamp(-1.0, 1.0)

    flee_on = brief_active & stance_flee
    dash_t = ((flee_on & (e_dist < 16.0)) | (~flee_on & (e_dist > 14.0) & (e_dist < 26.0))).float()
    melee_t = (~flee_on & has_enemy & (e_dist < 5.0)).float()
    bolt_t = (~flee_on & has_enemy & (e_dist < 34.0)).float()
    ice_t = torch.zeros_like(bolt_t)
    e_t = (~flee_on & has_enemy & (e_dist < 18.0)).float()
    r_t = (~flee_on & has_enemy & (e_dist < 14.0) & (obs[:, 5:6] < 0.7)).float()
    bin_tgt = torch.cat([dash_t, melee_t, bolt_t, ice_t, e_t, r_t], dim=-1)

    # Element target from brief thermal_bias or class_id
    therm = brief[:, 14]
    cls_id = torch.argmax(obs[:, 8:11], dim=-1)
    elem_tgt = torch.where(
        brief_active.squeeze(-1) & (therm > 0.35),
        torch.zeros_like(cls_id),
        torch.where(
            brief_active.squeeze(-1) & (therm < -0.35),
            torch.ones_like(cls_id),
            cls_id,
        ),
    )
    return vec_tgt, bin_tgt, elem_tgt


def run_stage0_dagger(
    ac: RecurrentActorCriticV3,
    v2_ref: BotPolicyNetV2 | None,
    wm: nn.Module,
    em: nn.Module,
    device: torch.device,
    num_arenas: int = 16,
    bots_per_arena: int = 60,
    iters: int = 100,
    seq_len: int = 16,
) -> dict[str, float]:
    print(f"\n[Stage 0] Brief+GRU Adapter Distillation | {num_arenas}x{bots_per_arena} bots | iters={iters} x T={seq_len}", flush=True)
    arena = RustVecArena(num_arenas=num_arenas, bots_per_arena=bots_per_arena, grid_w=64, grid_h=64, seed=101, field_step_interval=4)
    arena.sample_random_briefs(0.60)

    # Phase A (Brief-Adapter Isolation): keep V2's pretrained combat backbone 100% untouched (lr=0)
    # and train ONLY the gated brief branch (`brief_enc`, `brief_proj`, `role_emb`) on brief_active=1 steps!
    brief_prefixes = ("brief_enc.", "brief_proj.", "role_emb.")
    brief_params = [p for n, p in ac.policy.named_parameters() if n.startswith(brief_prefixes)]
    for n, p in ac.policy.named_parameters():
        p.requires_grad_(n.startswith(brief_prefixes))
    opt = torch.optim.AdamW(brief_params, lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, iters), eta_min=1e-4)
    total_bots = num_arenas * bots_per_arena
    h_cur = torch.zeros((total_bots, ac.policy.gru_hidden_dim), device=device, dtype=torch.float32)

    losses = []
    t0 = time.perf_counter()
    arena.sample_random_briefs(1.0)
    for it in range(iters):
        if it % 6 == 0:
            arena.sample_random_briefs(1.0)
        z_world, z_eco = encode_arena_latents(arena, wm, em, device)

        fp_list, cp_list, obs_list, done_list = [], [], [], []
        h0 = h_cur.detach().clone()
        obs_np = arena.obs().copy()
        done_prev = np.zeros((num_arenas, bots_per_arena), dtype=np.float32)

        for t in range(seq_len):
            if t > 0 and t % 4 == 0:
                z_world, z_eco = encode_arena_latents(arena, wm, em, device)
            obs_t = torch.from_numpy(obs_np).to(device)
            reset_t = torch.from_numpy(done_prev.reshape(-1)).to(device)
            with torch.inference_mode():
                actions, _, _, h_next, fp_f, cp_f = ac.act_step(z_world, z_eco, obs_t, h_cur, reset_t, deterministic=True)
                act_np = actions.cpu().numpy()
            h_cur = h_next.detach()
            fp_list.append(fp_f)
            cp_list.append(cp_f)
            obs_list.append(obs_t.reshape(total_bots, RL_OBS_DIM))
            done_list.append(reset_t)

            mask = np.zeros((num_arenas, bots_per_arena), dtype=np.uint8)
            mask[:, 1::2] = 1
            arena.set_scripted_mask(mask)
            obs_np_next, _, done_np, _ = arena.step(act_np)
            obs_np = obs_np_next.copy()
            done_prev = done_np.astype(np.float32)

        fp_seq = torch.stack(fp_list, dim=0)
        cp_seq = torch.stack(cp_list, dim=0)
        obs_seq = torch.stack(obs_list, dim=0)
        done_seq = torch.stack(done_list, dim=0)

        vec_p, bin_p, elem_p, _, _ = ac.policy.forward_sequence_from_patches(fp_seq, cp_seq, obs_seq, done_seq, h0)
        obs_flat = obs_seq.reshape(-1, RL_OBS_DIM)
        vec_t, bin_t, elem_t = compute_teacher_targets_from_obs(obs_flat)
        loss = (
            F.mse_loss(vec_p.reshape(-1, 4), vec_t) * 3.5
            + F.binary_cross_entropy_with_logits(bin_p.reshape(-1, 6), bin_t) * 1.4
            + F.cross_entropy(elem_p.reshape(-1, 3), elem_t) * 0.9
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(brief_params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if (it + 1) % 25 == 0 or it == 0 or it + 1 == iters:
            print(f"  BriefAdapter iter {it+1:3d}/{iters} | loss={loss.item():.4f} | elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    # Unfreeze all parameters for Stage 1/2 (except brief_enc/brief_proj which stay crisp when brief_prob=0!)
    for p in ac.policy.parameters():
        p.requires_grad_(True)
    arena.close()
    return {"final_dagger_loss": losses[-1], "dagger_secs": round(time.perf_counter() - t0, 2)}


def run_recurrent_ppo_stage(
    stage_name: str,
    ac: RecurrentActorCriticV3,
    wm: nn.Module,
    em: nn.Module,
    device: torch.device,
    num_arenas: int,
    bots_per_arena: int,
    updates: int = 50,
    rollout_len: int = 24,
    lr: float = 1.2e-4,
    seed: int = 202,
    brief_prob: float = 0.80,
) -> dict[str, float]:
    print(
        f"\n[{stage_name}] Recurrent PPO | {num_arenas}x{bots_per_arena} bots ({num_arenas*bots_per_arena} total) "
        f"| updates={updates} x T={rollout_len}",
        flush=True,
    )
    arena = RustVecArena(
        num_arenas=num_arenas,
        bots_per_arena=bots_per_arena,
        grid_w=64,
        grid_h=64,
        seed=seed,
        field_step_interval=4,
    )
    arena.sample_random_briefs(brief_prob)
    # Separate LRs: fast value critic (5e-4) + gentle policy (lr), with critic-only warmup to prevent upd-1 shock!
    opt = torch.optim.AdamW(
        [
            {"params": ac.policy.parameters(), "lr": lr},
            {"params": list(ac.value_head.parameters()) + [ac.log_std], "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    critic_warmup_updates = min(10, updates // 4)

    # Historical opponent checkpoint pool (Robocode / PufferLib self-play discipline: scripts + past copies!)
    ckpt_pool: list[BotPolicyNetV2] = [copy.deepcopy(ac.policy).eval()]
    total_bots = num_arenas * bots_per_arena
    h_cur = torch.zeros((total_bots, ac.policy.gru_hidden_dim), device=device, dtype=torch.float32)
    h_opp = torch.zeros((total_bots, ac.policy.gru_hidden_dim), device=device, dtype=torch.float32)

    mean_rews = []
    mean_preds = []
    t0 = time.perf_counter()
    obs_np = arena.obs().copy()
    done_prev = np.zeros((num_arenas, bots_per_arena), dtype=np.float32)

    for upd in range(updates):
        # Refresh opponent mix & briefs periodically
        if upd % 6 == 0:
            arena.sample_random_briefs(brief_prob)
        if upd > 0 and upd % 12 == 0:
            ckpt_pool.append(copy.deepcopy(ac.policy).eval())
            if len(ckpt_pool) > 5:
                ckpt_pool.pop(0)

        # Team 0 (even indices) always controlled by learner; Team 1 (odd) alternates:
        # 40% Scripted Teacher, 40% Historical Checkpoint Pool, 20% Mirror Self-Play
        mode = upd % 5
        mask = np.zeros((num_arenas, bots_per_arena), dtype=np.uint8)
        use_pool_opp = False
        opp_model = ckpt_pool[upd % len(ckpt_pool)]
        if mode in (0, 1):
            mask[:, 1::2] = 1  # Scripted Teacher on Team 1
        elif mode in (2, 3):
            use_pool_opp = True
        arena.set_scripted_mask(mask)

        z_world, z_eco = encode_arena_latents(arena, wm, em, device)
        h0 = h_cur.detach().clone()

        fp_buf, cp_buf, obs_buf, done_buf, done_after_buf, act_buf, logp_buf, val_buf, rew_buf, pred_buf = (
            [], [], [], [], [], [], [], [], [], []
        )

        # PufferLib-style double-buffered rollout overlap: CPU steps next while GPU builds tensors
        for t in range(rollout_len):
            if t > 0 and t % 4 == 0:
                z_world, z_eco = encode_arena_latents(arena, wm, em, device)
            obs_t = torch.from_numpy(obs_np).to(device)
            reset_t = torch.from_numpy(done_prev.reshape(-1)).to(device)

            with torch.inference_mode():
                actions, logp, val, h_next, fp_f, cp_f = ac.act_step(
                    z_world, z_eco, obs_t, h_cur, reset_t, deterministic=False
                )
                if use_pool_opp:
                    vec_o, bin_o, elem_o, _, _, h_opp_next = opp_model.forward_from_patches(
                        fp_f, cp_f, obs_t.reshape(total_bots, RL_OBS_DIM), h_prev=h_opp, reset_mask=reset_t, return_hidden=True
                    )
                    opp_act = torch.cat(
                        [vec_o.float(), (torch.sigmoid(bin_o) > 0.5).float(), F.one_hot(elem_o.argmax(-1), 3).float()],
                        dim=-1,
                    ).reshape(num_arenas, bots_per_arena, RL_ACTION_DIM)
                    actions[:, 1::2] = opp_act[:, 1::2]
                    h_opp = h_opp_next.detach()
                act_np = actions.cpu().numpy()

            # Launch Rust worker pool asynchronously while we append tensors
            arena.step_async_begin(act_np)
            h_cur = h_next.detach()
            fp_buf.append(fp_f)
            cp_buf.append(cp_f)
            obs_buf.append(obs_t.reshape(total_bots, RL_OBS_DIM))
            done_buf.append(reset_t)
            act_buf.append(actions.reshape(total_bots, RL_ACTION_DIM))
            logp_buf.append(logp.reshape(total_bots))
            val_buf.append(val.reshape(total_bots))

            obs_next, rew_np, done_np, pred_np = arena.step_async_wait()
            rew_buf.append(torch.from_numpy(rew_np.copy().reshape(total_bots)).to(device))
            done_after_buf.append(torch.from_numpy(done_np.astype(np.float32).reshape(total_bots)).to(device))
            pred_buf.append(pred_np.mean(axis=(0, 1)).copy())
            obs_np = obs_next.copy()
            done_prev = done_np.astype(np.float32)

        # Bootstrap value at T
        with torch.inference_mode():
            obs_last = torch.from_numpy(obs_np).to(device)
            reset_last = torch.from_numpy(done_prev.reshape(-1)).to(device)
            _, _, last_val, _, _, _ = ac.act_step(z_world, z_eco, obs_last, h_cur, reset_last, deterministic=True)
            last_val = last_val.reshape(total_bots)

        rew_seq = torch.stack(rew_buf, dim=0)            # [T, B]
        val_seq = torch.stack(val_buf, dim=0)            # [T, B]
        done_seq = torch.stack(done_buf, dim=0)          # [T, B] (reset before step t)
        done_after_seq = torch.stack(done_after_buf, 0)  # [T, B] (exact termination after step t!)
        # Exact GAE (gamma=0.97, lam=0.95)
        adv = torch.zeros_like(rew_seq)
        last_gae = torch.zeros(total_bots, device=device)
        gamma, lam = 0.97, 0.95
        for t in reversed(range(rollout_len)):
            next_v = last_val if t == rollout_len - 1 else val_seq[t + 1]
            next_nonterm = 1.0 - done_after_seq[t]
            delta = rew_seq[t] + gamma * next_v * next_nonterm - val_seq[t]
            last_gae = delta + gamma * lam * next_nonterm * last_gae
            adv[t] = last_gae
        ret_seq = adv + val_seq

        # Only compute PPO gradient on learner-controlled bots (even indices, or all on mirror)
        learn_mask = torch.ones((num_arenas, bots_per_arena), device=device, dtype=torch.bool)
        if mode != 4:
            learn_mask[:, 1::2] = False
        lm_flat = learn_mask.reshape(total_bots)
        learn_idx = torch.nonzero(lm_flat, as_tuple=False).squeeze(-1)

        fp_seq = torch.stack(fp_buf, dim=0)[:, learn_idx]
        cp_seq = torch.stack(cp_buf, dim=0)[:, learn_idx]
        obs_seq = torch.stack(obs_buf, dim=0)[:, learn_idx]
        done_sub = done_seq[:, learn_idx]
        act_seq = torch.stack(act_buf, dim=0)[:, learn_idx]
        old_logp = torch.stack(logp_buf, dim=0)[:, learn_idx].detach()
        adv_sub = adv[:, learn_idx].detach()
        ret_sub = ret_seq[:, learn_idx].detach()
        h0_sub = h0[learn_idx]

        adv_sub = (adv_sub - adv_sub.mean()) / (adv_sub.std() + 1e-6)

        # PufferLib low-replay-ratio discipline + observable aim/brief anchor + best-checkpoint tracking
        n_learn = learn_idx.numel()
        mb_size = min(256, n_learn)
        for _epoch in range(2):
            perm = torch.randperm(n_learn, device=device)
            early_stop = False
            for s in range(0, n_learn, mb_size):
                mb = perm[s : s + mb_size]
                vec_p, bin_p, elem_p, h_trunk_p, _ = ac.policy.forward_sequence_from_patches(
                    fp_seq[:, mb], cp_seq[:, mb], obs_seq[:, mb], done_sub[:, mb], h0_sub[mb]
                )
                new_val = ac.value_head(h_trunk_p.float()).squeeze(-1)
                std = ac.std().view(1, 1, 4)
                act_mb = act_seq[:, mb]
                dist_v = torch.distributions.Normal(vec_p.float(), std)
                dist_b = torch.distributions.Bernoulli(logits=bin_p.float())
                dist_e = torch.distributions.Categorical(logits=elem_p.float())
                new_logp = (
                    dist_v.log_prob(act_mb[..., :4].float()).sum(dim=-1)
                    + dist_b.log_prob(act_mb[..., 4:10].float()).sum(dim=-1)
                    + dist_e.log_prob(act_mb[..., 10:13].argmax(dim=-1))
                )
                ent = dist_v.entropy().sum(dim=-1) + dist_b.entropy().sum(dim=-1) + dist_e.entropy()

                log_ratio = (new_logp - old_logp[:, mb]).clamp(-10.0, 10.0)
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                if upd < critic_warmup_updates:
                    # Warm up value critic first so random critic advantages never wreck the DAgger-calibrated policy!
                    loss = 0.5 * F.mse_loss(new_val, ret_sub[:, mb])
                else:
                    if approx_kl > 0.15:
                        early_stop = True
                        break
                    pg1 = -adv_sub[:, mb] * ratio
                    pg2 = -adv_sub[:, mb] * ratio.clamp(0.88, 1.12)
                    pg_loss = torch.max(pg1, pg2).mean()
                    v_loss = 0.5 * F.mse_loss(new_val, ret_sub[:, mb])
                    # Gentle observable aim + brief-stance anchor (only on brief-active steps for element/flee)
                    vec_t, bin_t, elem_t = compute_teacher_targets_from_obs(obs_seq[:, mb].reshape(-1, RL_OBS_DIM))
                    aim_aux = F.mse_loss(vec_p[..., 2:4].reshape(-1, 2), vec_t[:, 2:4])
                    brief_on = obs_seq[:, mb, BOT_FEAT_DIM + 13] > 0.5
                    brief_elem_aux = (
                        F.cross_entropy(elem_p[brief_on], elem_t.reshape(obs_seq.shape[0], -1)[brief_on])
                        if brief_on.any()
                        else torch.tensor(0.0, device=device)
                    )
                    flee_mask = brief_on & (obs_seq[:, mb, BOT_FEAT_DIM + 5 + STANCE_FLEE_KITE] > 0.5)
                    flee_aux = (
                        F.binary_cross_entropy_with_logits(bin_p[flee_mask], bin_t.reshape_as(bin_p)[flee_mask])
                        if flee_mask.any()
                        else torch.tensor(0.0, device=device)
                    )
                    loss = pg_loss + 0.5 * v_loss - 0.004 * ent.mean() + 0.25 * aim_aux + 0.15 * brief_elem_aux + 0.20 * flee_aux
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
            if early_stop:
                break

        mr = float(rew_seq[:, learn_idx].mean().item())
        mp = np.mean(pred_buf, axis=0)
        mean_rews.append(mr)
        mean_preds.append(mp.tolist())
        if (upd + 1) % 10 == 0 or upd == 0:
            print(
                f"  [{stage_name}] upd {upd+1:2d}/{updates} | rew={mr:+.4f} | "
                f"pred(close={mp[0]:.3f}, stance={mp[1]:.3f}, risk={mp[2]:.3f}, shrine={mp[3]:.3f}) | "
                f"elapsed={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    arena.close()
    return {
        "mean_reward_first5": round(float(np.mean(mean_rews[:5])), 5),
        "mean_reward_last5": round(float(np.mean(mean_rews[-5:])), 5),
        "final_predicates": [round(float(x), 4) for x in mean_preds[-1]],
        "elapsed_secs": round(time.perf_counter() - t0, 2),
    }


@torch.inference_mode()
def run_frozen_brief_obedience_audit(
    policy: BotPolicyNetV2,
    wm: nn.Module,
    em: nn.Module,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    Hold weights 100% FIXED and flip only the System-2 TaskBrief vector across 5 contrasting
    briefs on identical 60-bot worlds; verify the same policy commits, holds, flees, hunts, or
    switches thermal element on command!
    """
    print("\n[Audit 1] Frozen-Weights System-2 Brief-Obedience Test (60 bots/arena)", flush=True)
    cases = [
        ("COMMIT_CAPTURE_NORTH_FIRE", OBJ_CAPTURE_SHRINE, STANCE_COMMIT, 32.0, 14.0, 0.85, 0.90),
        ("HOLD_DEFEND_CENTER_FROST", OBJ_DEFEND_HOLD, STANCE_HOLD, 32.0, 32.0, 0.45, -0.90),
        ("FLEE_KITE_LOW_RISK", OBJ_FREE_ROAM, STANCE_FLEE_KITE, 10.0, 32.0, 0.15, -0.50),
        ("HUNT_ENEMY_COMMIT_FIRE", OBJ_HUNT_ENEMY, STANCE_COMMIT, 50.0, 32.0, 0.95, 0.85),
        ("ESCORT_RELIC_HOLD", OBJ_ESCORT_RELIC, STANCE_HOLD, 32.0, 32.0, 0.60, 0.00),
    ]
    results: dict[str, dict[str, float]] = {}
    n_arenas, n_bots, steps = 4, 60, 40

    for label, obj, stance, tx, ty, risk, therm in cases:
        arena = RustVecArena(num_arenas=n_arenas, bots_per_arena=n_bots, grid_w=64, grid_h=64, seed=777, field_step_interval=4)
        # Team 1 scripted opponents, Team 0 tested with fixed weights & specified brief
        mask = np.zeros((n_arenas, n_bots), dtype=np.uint8)
        mask[:, 1::2] = 1
        arena.set_scripted_mask(mask)

        briefs = np.zeros((n_arenas, n_bots, RL_BRIEF_IN_DIM), dtype=np.float32)
        briefs[:, :, 0] = obj
        briefs[:, :, 1] = stance
        briefs[:, :, 2] = tx
        briefs[:, :, 3] = ty
        briefs[:, :, 4] = risk
        briefs[:, :, 5] = 1.0  # active
        briefs[:, :, 6] = therm
        arena.set_briefs(briefs)

        h = torch.zeros((n_arenas * n_bots, policy.gru_hidden_dim), device=device)
        z_world, z_eco = encode_arena_latents(arena, wm, em, device)
        obs_np = arena.obs().copy()
        done_prev = np.zeros((n_arenas, n_bots), dtype=np.float32)

        preds_t0 = []
        fire_frac, frost_frac, aggro_cast, mean_enemy_d = [], [], [], []
        for t in range(steps):
            if t > 0 and t % 4 == 0:
                z_world, z_eco = encode_arena_latents(arena, wm, em, device)
            obs_t = torch.from_numpy(obs_np).to(device)
            reset_t = torch.from_numpy(done_prev.reshape(-1)).to(device)
            fp, cp = sample_batched_dual_patches(policy, z_world, z_eco, obs_t)
            vec, bin_l, elem_l, _, _, h = policy.forward_from_patches(
                fp.reshape(n_arenas * n_bots, -1, policy.patch_size, policy.patch_size),
                cp.reshape(n_arenas * n_bots, -1, policy.patch_size, policy.patch_size),
                obs_t.reshape(n_arenas * n_bots, RL_OBS_DIM),
                h_prev=h,
                reset_mask=reset_t,
                return_hidden=True,
            )
            elem_idx = elem_l.argmax(dim=-1)
            act = torch.cat(
                [vec.float(), (torch.sigmoid(bin_l) > 0.5).float(), F.one_hot(elem_idx, 3).float()],
                dim=-1,
            ).reshape(n_arenas, n_bots, RL_ACTION_DIM)
            # Measure Team 0 behavioral telemetry
            t0_act = act[:, 0::2]
            fire_frac.append(float((t0_act[..., 10] > 0.5).float().mean().item()))
            frost_frac.append(float((t0_act[..., 11] > 0.5).float().mean().item()))
            aggro_cast.append(float(t0_act[..., 6].mean().item()))

            obs_next, _, done_np, pred_np = arena.step(act.cpu().numpy())
            preds_t0.append(pred_np[:, 0::2].mean(axis=(0, 1)))
            # Nearest enemy distance from obs token 0 (now guaranteed nearest enemy!)
            tok0 = obs_next[:, 0::2, 24 : 24 + 12]
            mean_enemy_d.append(float(tok0[..., 7].mean() * 100.0))
            obs_np = obs_next.copy()
            done_prev = done_np.astype(np.float32)

        mp = np.mean(preds_t0, axis=0)
        results[label] = {
            "pred_close_target": round(float(mp[0]), 4),
            "pred_stance_ok": round(float(mp[1]), 4),
            "pred_risk_bound": round(float(mp[2]), 4),
            "fire_elem_pct": round(100.0 * float(np.mean(fire_frac)), 1),
            "frost_elem_pct": round(100.0 * float(np.mean(frost_frac)), 1),
            "bolt_cast_pct": round(100.0 * float(np.mean(aggro_cast)), 1),
            "mean_nearest_bot_dist": round(float(np.mean(mean_enemy_d)), 2),
        }
        print(f"  {label:26s} | {results[label]}", flush=True)
        arena.close()
    return results


@torch.inference_mode()
def run_60bot_tournament(
    policy_v3: BotPolicyNetV2,
    policy_v2_legacy: BotPolicyNetV2 | None,
    wm: nn.Module,
    em: nn.Module,
    device: torch.device,
    matches: int = 8,
    steps: int = 160,
) -> dict[str, dict[str, float]]:
    """Side-balanced 60-bot server-merge tournament (30v30 live density!)."""
    print(f"\n[Audit 2] 60-Bot Live-Density Server-Merge Tournament ({matches} arenas x 60 bots x {steps} ticks)", flush=True)
    summary: dict[str, dict[str, float]] = {}

    def eval_pair(name: str, opp_kind: str) -> dict[str, float]:
        arena = RustVecArena(num_arenas=matches, bots_per_arena=60, grid_w=64, grid_h=64, seed=909, field_step_interval=4)
        # Side-balanced: first half arenas V3 is Team 0, second half V3 is Team 1
        v3_is_t0 = np.array([a < matches // 2 for a in range(matches)], dtype=bool)
        mask = np.zeros((matches, 60), dtype=np.uint8)
        if opp_kind == "scripted":
            for a in range(matches):
                if v3_is_t0[a]:
                    mask[a, 1::2] = 1
                else:
                    mask[a, 0::2] = 1
        arena.set_scripted_mask(mask)
        # Apples-to-apples free-play 60-bot combat/shrine/relic tournament (no artificial FLEE_KITE handicap;
        # System-2 brief obedience is audited separately in Audit 1 with frozen weights!)
        arena.sample_random_briefs(0.0)

        h_v3 = torch.zeros((matches * 60, policy_v3.gru_hidden_dim), device=device)
        z_world, z_eco = encode_arena_latents(arena, wm, em, device)
        obs_np = arena.obs().copy()
        done_prev = np.zeros((matches, 60), dtype=np.float32)

        for t in range(steps):
            if t > 0 and t % 4 == 0:
                z_world, z_eco = encode_arena_latents(arena, wm, em, device)
            obs_t = torch.from_numpy(obs_np).to(device)
            reset_t = torch.from_numpy(done_prev.reshape(-1)).to(device)
            fp, cp = sample_batched_dual_patches(policy_v3, z_world, z_eco, obs_t)
            fp_f = fp.reshape(matches * 60, -1, policy_v3.patch_size, policy_v3.patch_size)
            cp_f = cp.reshape(matches * 60, -1, policy_v3.patch_size, policy_v3.patch_size)
            obs_f = obs_t.reshape(matches * 60, RL_OBS_DIM)

            vec, bin_l, elem_l, _, _, h_v3 = policy_v3.forward_from_patches(
                fp_f, cp_f, obs_f, h_prev=h_v3, reset_mask=reset_t, return_hidden=True
            )
            act_v3 = torch.cat(
                [vec.float(), (torch.sigmoid(bin_l) > 0.5).float(), F.one_hot(elem_l.argmax(-1), 3).float()],
                dim=-1,
            ).reshape(matches, 60, RL_ACTION_DIM)

            if opp_kind == "v2_legacy" and policy_v2_legacy is not None:
                vec2, bin2, elem2, _, _ = policy_v2_legacy.forward_from_patches(
                    fp_f, cp_f, obs_f[:, :BOT_FEAT_DIM]
                )
                act_v2 = torch.cat(
                    [vec2.float(), (torch.sigmoid(bin2) > 0.5).float(), F.one_hot(elem2.argmax(-1), 3).float()],
                    dim=-1,
                ).reshape(matches, 60, RL_ACTION_DIM)
                final_act = act_v3.clone()
                for a in range(matches):
                    if v3_is_t0[a]:
                        final_act[a, 1::2] = act_v2[a, 1::2]
                    else:
                        final_act[a, 0::2] = act_v2[a, 0::2]
            else:
                final_act = act_v3

            obs_next, _, done_np, _ = arena.step(final_act.cpu().numpy())
            obs_np = obs_next.copy()
            done_prev = done_np.astype(np.float32)

        scores = arena.get_team_scores()  # [matches, 4]: s0, s1, k0, k1
        v3_wins = 0
        v3_s_sum, opp_s_sum, v3_k_sum, opp_k_sum = 0.0, 0.0, 0.0, 0.0
        for a in range(matches):
            s_v3, s_op = (scores[a, 0], scores[a, 1]) if v3_is_t0[a] else (scores[a, 1], scores[a, 0])
            k_v3, k_op = (scores[a, 2], scores[a, 3]) if v3_is_t0[a] else (scores[a, 3], scores[a, 2])
            if s_v3 > s_op:
                v3_wins += 1
            v3_s_sum += float(s_v3)
            opp_s_sum += float(s_op)
            v3_k_sum += float(k_v3)
            opp_k_sum += float(k_op)
        arena.close()
        res = {
            "v3_win_rate_pct": round(100.0 * v3_wins / matches, 1),
            "v3_mean_score": round(v3_s_sum / matches, 2),
            "opp_mean_score": round(opp_s_sum / matches, 2),
            "v3_mean_kills": round(v3_k_sum / matches, 2),
            "opp_mean_kills": round(opp_k_sum / matches, 2),
        }
        print(f"  {name:34s} | {res}", flush=True)
        return res

    summary["v3_60bot_vs_scripted_teacher"] = eval_pair("V3-Recurrent-60 vs ScriptedTeacher", "scripted")
    if policy_v2_legacy is not None:
        summary["v3_60bot_vs_v2_legacy_12bot"] = eval_pair("V3-Recurrent-60 vs Legacy-V2-12bot", "v2_legacy")
    return summary


@torch.inference_mode()
def benchmark_gb10_latency(policy: BotPolicyNetV2, device: torch.device) -> dict[str, float]:
    z_world = torch.randn((4, 48, 256, 256), device=device, dtype=torch.float16)
    z_eco = torch.randn((1, 32, 256, 256), device=device, dtype=torch.float16)
    out: dict[str, float] = {}
    for n in (12, 60, 256):
        bf = torch.rand((n, RL_OBS_DIM), device=device, dtype=torch.float32)
        h = torch.zeros((n, policy.gru_hidden_dim), device=device, dtype=torch.float32)
        for _ in range(10):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, _, h = policy(z_world, z_eco, bf, return_saliency=True, h_prev=h, return_hidden=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, _, h = policy(z_world, z_eco, bf, return_saliency=True, h_prev=h, return_hidden=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000.0
        out[f"gb10_fp16_{n}bots_ms"] = round(ms, 3)
    print(f"\n[Audit 3] GB10 FP16 Recurrent+Brief Latency: {out}", flush=True)
    return out


def benchmark_rust_arena(num_arenas: int = 20, steps: int = 120) -> dict[str, float]:
    print(f"\n[Bench] Pure-Rust RustVecArena Throughput ({find_field_core_lib()})", flush=True)
    out: dict[str, float] = {}
    for n_bots in (12, 32, 60):
        arena = RustVecArena(num_arenas=num_arenas, bots_per_arena=n_bots, grid_w=64, grid_h=64, seed=42, field_step_interval=4)
        arena.set_scripted_mask(np.ones((num_arenas, n_bots), dtype=np.uint8))
        arena.sample_random_briefs(0.8)
        act = np.zeros((num_arenas, n_bots, RL_ACTION_DIM), dtype=np.float32)
        for _ in range(5):
            arena.step(act)
        t0 = time.perf_counter()
        for _ in range(steps):
            arena.step(act)
        dt = time.perf_counter() - t0
        sps = (num_arenas * n_bots * steps) / max(dt, 1e-9)
        out[f"rust_{num_arenas}x{n_bots}_agent_steps_per_sec"] = round(sps, 1)
        print(f"  {num_arenas}x{n_bots:2d} bots | {sps:,.0f} agent-steps/sec ({dt/steps*1000:.2f} ms/step)", flush=True)
        arena.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger-iters", type=int, default=45)
    parser.add_argument("--stage1-updates", type=int, default=30)
    parser.add_argument("--stage2-updates", type=int, default=45)
    parser.add_argument("--arenas", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_bot_rl_v3] device={device} | lib={find_field_core_lib()}", flush=True)

    rust_bench = benchmark_rust_arena(num_arenas=20, steps=120)
    wm, em = load_frozen_world_models(device)

    v2_legacy: BotPolicyNetV2 | None = None
    if V2_RL_PT.exists():
        v2_legacy = load_bot_policy_v2_or_v3(V2_RL_PT, device=device, upgrade_to_v3=False).eval()
        policy = load_bot_policy_v2_or_v3(V2_RL_PT, device=device, upgrade_to_v3=True)
        print(f"Warm-started V3 recurrent+brief policy from {V2_RL_PT} | params={count_parameters(policy):,}", flush=True)
    else:
        policy = BotPolicyNetV2(use_gru=True, use_brief=True).to(device)
        print(f"Initialized fresh V3 recurrent+brief policy | params={count_parameters(policy):,}", flush=True)

    ac = RecurrentActorCriticV3(policy).to(device)

    s0 = run_stage0_dagger(ac, v2_legacy, wm, em, device, num_arenas=args.arenas, bots_per_arena=60, iters=args.dagger_iters, seq_len=16)
    best_sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    t0_eval = run_60bot_tournament(policy.eval(), v2_legacy, wm, em, device, matches=4, steps=120)
    best_margin = (
        (t0_eval["v3_60bot_vs_scripted_teacher"]["v3_mean_score"] - t0_eval["v3_60bot_vs_scripted_teacher"]["opp_mean_score"])
        + (t0_eval.get("v3_60bot_vs_v2_legacy_12bot", {}).get("v3_mean_score", 0.0) - t0_eval.get("v3_60bot_vs_v2_legacy_12bot", {}).get("opp_mean_score", 0.0))
    )
    print(f"  [BestCkpt] Post-Stage0 combined margin = {best_margin:+.2f}", flush=True)
    policy.train()

    s1_32 = run_recurrent_ppo_stage(
        "Stage 1 (32-Bot Density)", ac, wm, em, device, num_arenas=args.arenas, bots_per_arena=32, updates=args.stage1_updates, rollout_len=24, lr=3.5e-5, seed=202, brief_prob=0.0
    )
    t1_eval = run_60bot_tournament(policy.eval(), v2_legacy, wm, em, device, matches=4, steps=120)
    m1 = (
        (t1_eval["v3_60bot_vs_scripted_teacher"]["v3_mean_score"] - t1_eval["v3_60bot_vs_scripted_teacher"]["opp_mean_score"])
        + (t1_eval.get("v3_60bot_vs_v2_legacy_12bot", {}).get("v3_mean_score", 0.0) - t1_eval.get("v3_60bot_vs_v2_legacy_12bot", {}).get("opp_mean_score", 0.0))
    )
    print(f"  [BestCkpt] Post-Stage1 combined margin = {m1:+.2f} (best={best_margin:+.2f})", flush=True)
    if m1 > best_margin:
        best_margin = m1
        best_sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    policy.train()

    s2_60 = run_recurrent_ppo_stage(
        "Stage 2 (60-Bot Shipped Density)", ac, wm, em, device, num_arenas=args.arenas, bots_per_arena=60, updates=args.stage2_updates, rollout_len=24, lr=2.5e-5, seed=303, brief_prob=0.0
    )
    t2_eval = run_60bot_tournament(policy.eval(), v2_legacy, wm, em, device, matches=4, steps=120)
    m2 = (
        (t2_eval["v3_60bot_vs_scripted_teacher"]["v3_mean_score"] - t2_eval["v3_60bot_vs_scripted_teacher"]["opp_mean_score"])
        + (t2_eval.get("v3_60bot_vs_v2_legacy_12bot", {}).get("v3_mean_score", 0.0) - t2_eval.get("v3_60bot_vs_v2_legacy_12bot", {}).get("opp_mean_score", 0.0))
    )
    print(f"  [BestCkpt] Post-Stage2 combined margin = {m2:+.2f} (best={best_margin:+.2f})", flush=True)
    if m2 > best_margin:
        best_margin = m2
        best_sd = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    # Restore best checkpoint (Suárez NMMO 3 rule: ship peak checkpoint, never degraded tail!)
    policy.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    policy.eval()
    brief_audit = run_frozen_brief_obedience_audit(policy, wm, em, device)
    tourney = run_60bot_tournament(policy, v2_legacy, wm, em, device, matches=8, steps=160)
    latency = benchmark_gb10_latency(policy, device) if device.type == "cuda" else {}

    ckpt = {
        "arch": "BotPolicyNetV2-Recurrent-Brief-V3",
        "params": count_parameters(policy),
        "use_gru": True,
        "gru_hidden_dim": GRU_HIDDEN_DIM,
        "use_brief": True,
        "brief_dim": BRIEF_DIM,
        "obs_dim": BOT_FEAT_DIM_V3,
        "state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
    }
    torch.save(ckpt, V3_RL_PT)

    report = {
        "checkpoint": str(V3_RL_PT),
        "params": count_parameters(policy),
        "rust_vec_arena_bench": rust_bench,
        "stage0_dagger": s0,
        "stage1_ppo_32bot": s1_32,
        "stage2_ppo_60bot": s2_60,
        "frozen_brief_obedience": brief_audit,
        "tournament_60bot": tourney,
        "gb10_latency": latency,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"\n[DONE] Saved {V3_RL_PT} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
