#!/usr/bin/env python3
"""
Held-Out 256x256x4 Frostmarsh 6v6 Tournament Evaluation (`eval_bot_arena.py`).

Evaluates 4 side-balanced matchups (24 unseen matches x 120 ticks each = 12 West + 12 East):
  1. BotPolicyNet-RL vs Scripted Teacher
  2. BotPolicyNet-RL vs BotPolicyNet-BC
  3. BotPolicyNet-RL vs Blind Ledger-Only
  4. BotPolicyNet-BC vs Scripted Teacher

Uses REAL frozen WorldModel [4, 48, 256, 256] and EcosystemModel [1, 32, 256, 256]
latents on GPU, verifies pure BotPolicyNet drop-in checkpoint loading, and writes:
  /home/gooning/.t3/scratch/frostmarsh-ml/BOT_RL_TOURNAMENT.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from bot_policy import BotPolicyNet, count_parameters
from train_bot_rl import (
    BC_PT,
    RL_PT,
    ROOT,
    FrostmarshGPUArena,
    load_ckpt_state_dict,
    load_frozen_world_models,
    policy_deterministic_act,
    scripted_teacher_act,
)

OUT_JSON = ROOT / "BOT_RL_TOURNAMENT.json"


def _agent_actions(
    agent_kind: str,
    policy: BotPolicyNet | None,
    arena: FrostmarshGPUArena,
    feats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if agent_kind == "teacher":
        return scripted_teacher_act(arena)
    assert policy is not None
    return policy_deterministic_act(policy, arena.z_world, arena.z_eco, feats)


@torch.inference_mode()
def run_matchup(
    name_a: str,
    kind_a: str,
    policy_a: BotPolicyNet | None,
    name_b: str,
    kind_b: str,
    policy_b: BotPolicyNet | None,
    arena: FrostmarshGPUArena,
    num_matches: int = 24,
    ticks: int = 120,
    base_seed: int = 90000,
) -> dict:
    """
    Play num_matches held-out 256x256x4 matches (half A=West/B=East, half A=East/B=West)
    for zero side bias. Returns complete per-matchup statistics.
    """
    assert num_matches % 2 == 0
    half = num_matches // 2
    batch = arena.A
    assert half % batch == 0, f"half ({half}) must be divisible by arena batch ({batch})"

    a_scores, b_scores = [], []
    a_shrines, b_shrines = [], []
    a_combos, b_combos = [], []
    a_burns, b_burns = [], []
    a_kills, b_kills = [], []
    a_deaths, b_deaths = [], []
    a_relics, b_relics = [], []

    t0 = time.perf_counter()
    for side in (0, 1):  # side=0: A=West(0), B=East(1); side=1: A=East(1), B=West(0)
        for bi in range(half // batch):
            seed = base_seed + bi * 1000
            feats = arena.reset(seed=seed)

            a_mask = (arena.team == side)
            for _ in range(ticks):
                av_a, ab_a, ae_a = _agent_actions(kind_a, policy_a, arena, feats)
                av_b, ab_b, ae_b = _agent_actions(kind_b, policy_b, arena, feats)

                act_vec = torch.where(a_mask.unsqueeze(-1), av_a, av_b)
                act_bin = torch.where(a_mask.unsqueeze(-1), ab_a, ab_b)
                act_elem = torch.where(a_mask, ae_a, ae_b)

                feats, _, _ = arena.step(act_vec, act_bin, act_elem)

            ia = side
            ib = 1 - side
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
        f"  [{res['matchup']:38s}] WR={res['win_rate_a_pct']:5.1f}% ({wins_a}W-{wins_b}L-{draws}D) | "
        f"Score {res['mean_score_a']:6.1f} vs {res['mean_score_b']:6.1f} | "
        f"Shrines {res['shrine_captures_per_match_a']:4.1f} vs {res['shrine_captures_per_match_b']:4.1f} | "
        f"Combos {res['physics_combos_per_match_a']:5.1f} vs {res['physics_combos_per_match_b']:5.1f} | "
        f"Burn {res['wildfire_burn_taken_per_match_a']:5.2f} vs {res['wildfire_burn_taken_per_match_b']:5.2f} | "
        f"K/D {res['kills_per_match_a']:4.1f}/{res['deaths_per_match_a']:4.1f} vs {res['kills_per_match_b']:4.1f}/{res['deaths_per_match_b']:4.1f}",
        flush=True,
    )
    return res


def main() -> None:
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    print("=== Frostmarsh 256x256x4 Held-Out 6v6 Neural Bot Tournament ===")
    wm, em = load_frozen_world_models(device)

    # Load pure BotPolicyNet checkpoints (verifying 100% strict compatibility!)
    rl_net = BotPolicyNet(blind_world=False).to(device).eval()
    rl_net.load_state_dict(load_ckpt_state_dict(RL_PT, device), strict=True)

    bc_net = BotPolicyNet(blind_world=False).to(device).eval()
    bc_net.load_state_dict(load_ckpt_state_dict(BC_PT, device), strict=True)

    blind_net = BotPolicyNet(blind_world=True).to(device).eval()
    blind_net.load_state_dict(load_ckpt_state_dict(BC_PT, device), strict=True)

    # Verify single-arena BotPolicyNet.forward() works identically (exact gpu_infer_daemon.py call!)
    with torch.inference_mode():
        dummy_zw = torch.zeros((4, 48, 256, 256), dtype=torch.float16, device=device)
        dummy_ze = torch.zeros((1, 32, 256, 256), dtype=torch.float16, device=device)
        dummy_bf = torch.zeros((12, 168), dtype=torch.float32, device=device)
        out = rl_net(dummy_zw, dummy_ze, dummy_bf)
        assert out.shape == (12, 13), f"Unexpected action shape: {out.shape}"

    # 12 parallel 256x256x4 arenas -> 2 batches per matchup = 24 matches (12 West + 12 East)
    arena = FrostmarshGPUArena(
        num_arenas=12,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=256,
        num_layers=4,
        max_ticks=120,
        latent_refresh_every=4,
    )

    t0 = time.perf_counter()
    matchups = [
        run_matchup("BotPolicyNet-RL", "policy", rl_net, "Scripted Teacher", "teacher", None, arena, num_matches=24, ticks=120, base_seed=91000),
        run_matchup("BotPolicyNet-RL", "policy", rl_net, "BotPolicyNet-BC", "policy", bc_net, arena, num_matches=24, ticks=120, base_seed=92000),
        run_matchup("BotPolicyNet-RL", "policy", rl_net, "Blind Ledger-Only", "policy", blind_net, arena, num_matches=24, ticks=120, base_seed=93000),
        run_matchup("BotPolicyNet-BC", "policy", bc_net, "Scripted Teacher", "teacher", None, arena, num_matches=24, ticks=120, base_seed=94000),
    ]

    rl_ckpt = torch.load(RL_PT, map_location="cpu", weights_only=False)
    rl_train_meta = rl_ckpt.get("rl_training", {}) if isinstance(rl_ckpt, dict) else {}

    summary = {
        "device": torch.cuda.get_device_name(0),
        "grid": "256x256x4",
        "matches_per_matchup": 24,
        "ticks_per_match": 120,
        "side_balanced": True,
        "policy_params": count_parameters(rl_net),
        "drop_in_compatible_with_gpu_infer_daemon": True,
        "rl_checkpoint": str(RL_PT),
        "rl_training_summary": rl_train_meta,
        "tournament_time_s": round(time.perf_counter() - t0, 2),
        "matchups": {
            "rl_vs_teacher": matchups[0],
            "rl_vs_bc": matchups[1],
            "rl_vs_blind": matchups[2],
            "bc_vs_teacher": matchups[3],
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_JSON} OK!")


if __name__ == "__main__":
    main()
