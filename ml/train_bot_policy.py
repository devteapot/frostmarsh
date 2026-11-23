#!/usr/bin/env python3
"""
Train & Ablate `BotPolicyNet` on NVIDIA GB10 using REAL frozen `WorldModel` [4, 48, 256, 256]
and `EcosystemModel` [1, 32, 256, 256] internal spatial latents!

Compares two matched ~115k-param neural agents on held-out unseen 3D Shattered Reach worlds:
  1. Shared-Latent Neural Bot (`+Z_world + Z_eco`): reads WorldModel's VRAM trunk zero-copy
  2. Blind Ledger-Only Ablation (`No Z_world`): same Entity Cross-Attention, world trunk zeroed

Saves:
  - /home/gooning/.t3/scratch/frostmarsh-ml/bot_policy.pt
  - /home/gooning/.t3/scratch/frostmarsh-ml/BOT_POLICY_METRICS.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bot_policy import (
    BOT_FEAT_DIM,
    NUM_TOKENS,
    SELF_DIM,
    TOKEN_DIM,
    BotPolicyNet,
    count_parameters,
)
from model import EcosystemModel, WorldModel

GRID = 256
LAYERS = 4
SP_WATER, SP_ICE, SP_STEAM, SP_SOIL, SP_ROCK, SP_WOOD, SP_ASH, SP_SMOKE, SP_OIL, SP_PLANT = range(10)


def extract_world_latent(world_model: WorldModel, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        h = world_model.stem(x)
        for blk in world_model.blocks:
            h = blk(h)
        return F.gelu(h).float()


def extract_eco_latent(eco_model: EcosystemModel, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        h = eco_model.stem(x)
        for blk in eco_model.blocks:
            h = blk(h)
        return F.gelu(h).float()


def make_synthetic_3d_world_state(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate diverse 256x256x4 Frostmarsh 3D tensors matching exact FIELD_CONTRACT channels."""
    rng = np.random.default_rng(seed)
    w_state = np.zeros((LAYERS, 18, GRID, GRID), dtype=np.float32)
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(np.float32)
    nx, ny = xx / GRID, yy / GRID

    # River center + random meanders, wildfires, oil pools, ice patches, timber bridges!
    phase = rng.uniform(0, 6.28)
    rcx = 0.50 + rng.uniform(-0.08, 0.08) + 0.055 * np.sin(ny * 6.28 * 1.3 + phase)
    rhw = rng.uniform(0.065, 0.095)
    in_river = np.abs(nx - rcx) < rhw

    # L1 (surface)
    w_state[1, SP_SOIL] = np.where(~in_river, 0.92, 0.0)
    w_state[1, SP_WATER] = np.where(in_river, 1.10, 0.0)
    w_state[1, 13] = np.where(in_river, 0.13, 0.0)  # southward flow
    w_state[1, 14] = np.where(in_river, 0.15, 0.45)

    # Random ice crossings
    ice_y = rng.uniform(0.14, 0.35)
    ice_m = in_river & (np.abs(ny - ice_y) < 0.04)
    w_state[1, SP_ICE][ice_m] = 0.95
    w_state[1, SP_WATER][ice_m] = 0.15
    w_state[1, 10][ice_m] = -0.25  # cold temp

    # Random oil basins (6-10 pockets)
    for _ in range(rng.integers(6, 11)):
        ox, oy = rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)
        orad = rng.uniform(0.035, 0.075)
        om = ((nx - ox) ** 2 + (ny - oy) ** 2) < orad ** 2
        w_state[1, SP_OIL][om] = 0.82
        w_state[1, SP_WATER][om] = 0.18

    # Random wood forests & plants
    wood_noise = np.sin(xx * 0.17 + seed) * np.cos(yy * 0.19 - seed)
    w_state[1, SP_WOOD] = np.where((~in_river) & (wood_noise > 0.35), 0.90, 0.0)
    w_state[1, SP_PLANT] = np.where((~in_river) & (wood_noise > -0.1), 0.45, 0.0)

    # Random rock outcrops
    for _ in range(rng.integers(5, 9)):
        rx, ry = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        rm = ((nx - rx) ** 2 + (ny - ry) ** 2) < rng.uniform(0.018, 0.032) ** 2
        w_state[1, SP_ROCK][rm] = 1.15
        w_state[1, 15][rm] = 1.0

    # Random active wildfires (hot temp > 440K => channel 10 > 1.5!)
    for _ in range(rng.integers(5, 11)):
        fx, fy = rng.uniform(0.12, 0.88), rng.uniform(0.12, 0.88)
        frad = rng.uniform(0.025, 0.055)
        fm = ((nx - fx) ** 2 + (ny - fy) ** 2) < frad ** 2
        w_state[1, 10][fm] = rng.uniform(2.5, 4.8)
        w_state[1, SP_SMOKE][fm] = 0.35
        w_state[1, SP_WOOD][fm] = np.maximum(w_state[1, SP_WOOD][fm], 0.40)

    # L2 (Highland plateaus + overhanging Skybridges + alpine lakes)
    nw_p = (nx > 0.06) & (nx < 0.36) & (ny > 0.06) & (ny < 0.33)
    ne_p = (nx > 0.64) & (nx < 0.94) & (ny > 0.06) & (ny < 0.33)
    arch = (np.abs(ny - 0.29) < 0.022) & (nx > 0.34) & (nx < 0.66)
    tbridge = (np.abs(ny - 0.67) < 0.020) & (nx > 0.35) & (nx < 0.65)
    w_state[2, SP_SOIL] = np.where(nw_p | ne_p | arch, 0.95, 0.0)
    w_state[2, SP_WOOD] = np.where(tbridge, 0.52, 0.0)
    # Random fire on L2 skybridge or cliff
    if rng.random() < 0.6:
        bfx = rng.uniform(0.38, 0.62)
        bfm = ((nx - bfx) ** 2 + (ny - 0.67) ** 2) < 0.028 ** 2
        w_state[2, 10][bfm] = 3.6
        w_state[2, SP_SMOKE][bfm] = 0.3
    w_state[2, 14] = 0.65

    # L0 & L3
    w_state[0, SP_WATER] = np.where(in_river, 1.15, 0.0)
    w_state[0, SP_ROCK] = np.where(~in_river, 1.2, 0.0)
    w_state[3, SP_STEAM] = np.where(w_state[1, 10] > 2.0, 0.25, 0.0)
    w_state[3, 14] = 0.95

    # Log total mass channel 11
    for l in range(LAYERS):
        w_state[l, 11] = np.log1p(w_state[l, :10].sum(axis=0))

    # Ecosystem state [1, 8, 256, 256]
    e_state = np.zeros((1, 8, GRID, GRID), dtype=np.float32)
    for _ in range(4):
        cx, cy = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        d = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
        e_state[0, 0] = np.maximum(e_state[0, 0], np.exp(-((d * 14.0) ** 2)))
    e_state[0, 2] = 0.8
    e_state[0, 4] = np.where((w_state[1, SP_ROCK] < 0.5) & (w_state[1, SP_WATER] < 0.45), 1.0, 0.0)
    e_state[0, 5] = (w_state[1, 10] > 1.5).astype(np.float32)
    e_state[0, 6] = w_state[1, SP_PLANT]
    e_state[0, 7] = w_state[1, SP_ICE]

    return w_state, e_state


def sample_expert_batch(
    w_state: np.ndarray,
    e_state: np.ndarray,
    n_bots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate physics-coupled expert demos where optimal element, movement, & abilities
    genuinely depend on the underlying 3D world physics (fire, oil, water, ice, wildlife)!
    Returns: (bot_feats [N,168], tgt_vec [N,4], tgt_bin [N,6], tgt_elem [N], phys_flag [N])
    """
    feats = np.zeros((n_bots, BOT_FEAT_DIM), dtype=np.float32)
    tgt_vec = np.zeros((n_bots, 4), dtype=np.float32)
    tgt_bin = np.zeros((n_bots, 6), dtype=np.float32)
    tgt_elem = np.zeros((n_bots,), dtype=np.int64)
    phys_flag = np.zeros((n_bots,), dtype=np.int64)  # 1=fire_dodge, 2=oil_combo, 3=river_frost, 4=sanctuary

    # Shrines & Relic
    shrines = np.array([
        [0.50 * GRID, 0.18 * GRID, rng.uniform(-1, 1)],
        [0.50 * GRID, 0.50 * GRID, rng.uniform(-1, 1)],
        [0.50 * GRID, 0.82 * GRID, rng.uniform(-1, 1)],
    ], dtype=np.float32)
    relic_x = rng.uniform(0.25, 0.75) * GRID
    relic_y = rng.uniform(0.25, 0.75) * GRID

    # Spawn synthetic pool of agents
    xs = rng.uniform(16, GRID - 16, size=n_bots).astype(np.float32)
    ys = rng.uniform(16, GRID - 16, size=n_bots).astype(np.float32)
    zs = np.where(rng.random(n_bots) < 0.25, 2.0, 1.0).astype(np.float32)
    teams = (np.arange(n_bots) % 2).astype(np.int32)
    classes = (np.arange(n_bots) % 3).astype(np.int32)
    hps = rng.uniform(0.20, 1.0, size=n_bots).astype(np.float32)

    for i in range(n_bots):
        bx, by, bz = float(xs[i]), float(ys[i]), float(zs[i])
        bl = int(round(bz))
        ix, iy = int(np.clip(bx, 0, GRID - 1)), int(np.clip(by, 0, GRID - 1))
        team, cls, hp = int(teams[i]), int(classes[i]), float(hps[i])
        mana = float(rng.uniform(0.15, 1.0))
        stam = float(rng.uniform(0.20, 1.0))
        carrying = 1.0 if (i == 0 and rng.random() < 0.35) else 0.0
        dash_cd = 0.0 if rng.random() < 0.65 else float(rng.uniform(0.1, 1.0))
        e_cd = 0.0 if rng.random() < 0.45 else float(rng.uniform(0.1, 1.0))
        r_cd = 0.0 if rng.random() < 0.25 else float(rng.uniform(0.1, 1.0))

        # Populate self_feat [24]
        sf = feats[i, :SELF_DIM]
        sf[0] = bx / GRID
        sf[1] = by / GRID
        sf[2] = bz / (LAYERS - 1)
        sf[3] = rng.uniform(-0.3, 0.3)
        sf[4] = rng.uniform(-0.3, 0.3)
        sf[5] = hp
        sf[6] = mana
        sf[7] = stam
        sf[8 + cls] = 1.0
        sf[11] = float(team)
        sf[12] = carrying
        sf[13] = rng.uniform(0.2, 1.0)
        sf[20] = dash_cd
        sf[21] = e_cd
        sf[22] = r_cd
        sf[23] = float(rng.integers(0, 4)) / 3.0

        # Nearest 8 other agents -> tokens 0..8
        dists = np.hypot(xs - bx, ys - by)
        dists[i] = 1e9
        nn = np.argsort(dists)[:8]
        nearest_enemy = None
        for ti, j in enumerate(nn):
            dx, dy = float(xs[j] - bx), float(ys[j] - by)
            d = max(math.hypot(dx, dy), 1e-3)
            is_en = 1.0 if teams[j] != team else 0.0
            tk = feats[i, SELF_DIM + ti * TOKEN_DIM : SELF_DIM + (ti + 1) * TOKEN_DIM]
            tk[0] = np.clip(dx / 80.0, -1.0, 1.0)
            tk[1] = np.clip(dy / 80.0, -1.0, 1.0)
            tk[2] = (zs[j] - bz) / 2.0
            tk[3] = is_en
            tk[4] = hps[j]
            tk[5] = 0.0
            tk[6] = classes[j] / 2.0
            tk[7] = min(d / 100.0, 1.0)
            tk[8] = dx / d
            tk[9] = dy / d
            tk[11] = 1.0
            if is_en > 0.5 and nearest_enemy is None:
                nearest_enemy = (j, dx, dy, d)

        # 3 Shrine tokens (8..11) + 1 Relic token (11)
        best_shrine = (shrines[1, 0], shrines[1, 1])
        best_sd = 1e9
        for si in range(3):
            sx, sy, sprog = float(shrines[si, 0]), float(shrines[si, 1]), float(shrines[si, 2])
            need = (sprog > -0.8) if team == 0 else (sprog < 0.8)
            tk = feats[i, SELF_DIM + (8 + si) * TOKEN_DIM : SELF_DIM + (9 + si) * TOKEN_DIM]
            tk[0] = np.clip((sx - bx) / 100.0, -1.0, 1.0)
            tk[1] = np.clip((sy - by) / 100.0, -1.0, 1.0)
            tk[3] = 1.0 if need else 0.0
            tk[4] = sprog
            tk[11] = 1.0
            sd = math.hypot(sx - bx, sy - by)
            if need and sd < best_sd:
                best_sd = sd
                best_shrine = (sx, sy)

        tk_r = feats[i, SELF_DIM + 11 * TOKEN_DIM : SELF_DIM + 12 * TOKEN_DIM]
        tk_r[0] = np.clip((relic_x - bx) / 100.0, -1.0, 1.0)
        tk_r[1] = np.clip((relic_y - by) / 100.0, -1.0, 1.0)
        tk_r[3] = carrying
        tk_r[11] = 1.0

        # Base tactical goal (home if carrying, else nearest unowned shrine / relic)
        if carrying > 0.5:
            gx, gy = (0.12 * GRID if team == 0 else 0.88 * GRID), 0.50 * GRID
        elif i % 3 == 0:
            gx, gy = relic_x, relic_y
        else:
            gx, gy = best_shrine

        mdx, mdy = gx - bx, gy - by
        ml = max(math.hypot(mdx, mdy), 1e-3)
        mvx, mvy = mdx / ml, mdy / ml

        # --- PHYSICS-COUPLED EXPERT DECISIONS (REQUIRES READING Z_WORLD & Z_ECO!) ---
        # Inspect local 14-cell physics patch around bot (ONLY observable via Z_world/Z_eco!)
        x0, x1 = max(0, ix - 10), min(GRID, ix + 11)
        y0, y1 = max(0, iy - 10), min(GRID, iy + 11)
        local_temp = w_state[bl, 10, y0:y1, x0:x1]
        local_water = w_state[bl, SP_WATER, y0:y1, x0:x1]
        local_ice = w_state[bl, SP_ICE, y0:y1, x0:x1]
        local_oil = w_state[bl, SP_OIL, y0:y1, x0:x1]
        local_crit = e_state[0, 0, y0:y1, x0:x1]

        # Probe 5 cells ahead along movement vector
        ax_p = int(np.clip(bx + mvx * 5.0, 0, GRID - 1))
        ay_p = int(np.clip(by + mvy * 5.0, 0, GRID - 1))
        fire_here = w_state[bl, 10, iy, ix] > 1.5
        fire_ahead = w_state[bl, 10, ay_p, ax_p] > 1.5
        water_ahead = (w_state[bl, SP_WATER, ay_p, ax_p] > 0.45) and (w_state[bl, SP_ICE, ay_p, ax_p] < 0.35)
        ice_here = w_state[bl, SP_ICE, iy, ix] > 0.40
        oil_nearby = float(local_oil.max()) > 0.35

        # 1. Wildfire Evasion: steer perpendicular/away from active fire & dash out if burning!
        dash = 0.0
        if fire_here or fire_ahead:
            fy_idx, fx_idx = np.unravel_index(np.argmax(local_temp), local_temp.shape)
            f_dx = (x0 + fx_idx) - bx
            f_dy = (y0 + fy_idx) - by
            fl = max(math.hypot(f_dx, f_dy), 1e-3)
            # Evade away from flame center!
            mvx, mvy = -f_dx / fl, -f_dy / fl
            if fire_here and dash_cd == 0.0 and stam > 0.25:
                dash = 1.0
            phys_flag[i] = 1
        elif hp < 0.36 and float(local_crit.max()) > 0.25:
            # 4. Low-HP Wildlife Sanctuary Retreat (uses Z_eco critter density!)
            cy_idx, cx_idx = np.unravel_index(np.argmax(local_crit), local_crit.shape)
            cdx = (x0 + cx_idx) - bx
            cdy = (y0 + cy_idx) - by
            cl = max(math.hypot(cdx, cdy), 1e-3)
            mvx, mvy = cdx / cl, cdy / cl
            phys_flag[i] = 4

        # Aim + Elemental Combo Logic (STRICTLY driven by local Z_world species!)
        if nearest_enemy is not None and nearest_enemy[3] < 60.0:
            ej, edx, edy, ed = nearest_enemy
            aimx, aimy = edx / ed, edy / ed
            ex_i = int(np.clip(xs[ej], 0, GRID - 1))
            ey_i = int(np.clip(ys[ej], 0, GRID - 1))
            el_i = int(round(zs[ej]))
            enemy_on_oil = w_state[el_i, SP_OIL, ey_i, ex_i] > 0.25
            enemy_in_water = w_state[el_i, SP_WATER, ey_i, ex_i] > 0.40
        else:
            aimx, aimy, ed = mvx, mvy, 999.0
            enemy_on_oil, enemy_in_water = False, False

        melee = 1.0 if ed < 6.2 else 0.0
        cast_bolt = 0.0
        carve_ice = 0.0
        ab_e = 0.0
        ult_r = 0.0

        # Physics-optimal element choice:
        if oil_nearby or enemy_on_oil:
            # Ignite Oil -> OIL INFERNO!
            elem = 0  # Fire
            cast_bolt = 1.0 if mana > 0.12 else 0.0
            if ed < 22.0 and r_cd == 0.0 and mana > 0.55 and cls == 1:
                ult_r = 1.0
            phys_flag[i] = 2
        elif water_ahead or enemy_in_water:
            # Freeze River / Deep Freeze wet enemy!
            elem = 1  # Frost
            cast_bolt = 1.0 if mana > 0.12 else 0.0
            if water_ahead and e_cd == 0.0 and mana > 0.30 and cls == 0:
                ab_e = 1.0  # Ice Bridge!
            phys_flag[i] = 3
        else:
            # Dry land combat: Alchemist primes with Oil (2), Warden Frost (1), Ranger Fire (0)
            elem = 2 if (cls == 1 and ed < 38.0) else (0 if cls == 2 else 1)
            cast_bolt = 1.0 if (ed < 48.0 and mana > 0.15) else 0.0
            if ed < 24.0 and e_cd == 0.0 and mana > 0.30:
                ab_e = 1.0
            if ed < 20.0 and r_cd == 0.0 and mana > 0.55:
                ult_r = 1.0

        if ice_here and carrying > 0.5:
            carve_ice = 1.0

        tgt_vec[i] = [mvx, mvy, aimx, aimy]
        tgt_bin[i] = [dash, melee, cast_bolt, carve_ice, ab_e, ult_r]
        tgt_elem[i] = elem

    return feats, tgt_vec, tgt_bin, tgt_elem, phys_flag


def train_and_eval(
    name: str,
    blind_world: bool,
    train_worlds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    val_worlds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]],
    device: torch.device,
    steps: int = 2000,
) -> tuple[BotPolicyNet, dict]:
    net = BotPolicyNet(blind_world=blind_world).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1.5e-4)

    t0 = time.perf_counter()
    net.train()
    n_tw = len(train_worlds)
    for step in range(steps):
        zw, ze, feats, t_vec, t_bin, t_elem = train_worlds[step % n_tw]
        idx = torch.randint(0, feats.shape[0], (256,), device=device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            p_vec, p_bin, p_elem = net.forward_logits(zw, ze, feats[idx])
            tv = t_vec[idx]
            l_mv = F.smooth_l1_loss(p_vec[:, :2], tv[:, :2]) + 0.5 * (1.0 - F.cosine_similarity(p_vec[:, :2], tv[:, :2], dim=-1)).mean()
            l_aim = F.smooth_l1_loss(p_vec[:, 2:], tv[:, 2:]) + 0.8 * (1.0 - F.cosine_similarity(p_vec[:, 2:], tv[:, 2:], dim=-1)).mean()
            l_bin = F.binary_cross_entropy_with_logits(p_bin, t_bin[idx])
            l_elem = F.cross_entropy(p_elem, t_elem[idx])
            loss = 2.0 * l_mv + 2.2 * l_aim + 1.0 * l_bin + 1.4 * l_elem
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    train_s = time.perf_counter() - t0

    # Held-out Evaluation on unseen 3D worlds
    net.eval()
    mv_cos_all, aim_cos_all, elem_acc_all, phys_elem_acc, fire_dodge_cos, sanct_cos, bin_acc_all = [], [], [], [], [], [], []
    with torch.inference_mode():
        for zw, ze, feats, t_vec, t_bin, t_elem, pflag in val_worlds:
            p_vec, p_bin, p_elem = net.forward_logits(zw, ze, feats)
            mv_cos = F.cosine_similarity(p_vec[:, :2], t_vec[:, :2], dim=-1).cpu().numpy()
            aim_cos = F.cosine_similarity(p_vec[:, 2:], t_vec[:, 2:], dim=-1).cpu().numpy()
            e_ok = (p_elem.argmax(dim=-1) == t_elem).float().cpu().numpy()
            b_ok = ((torch.sigmoid(p_bin) > 0.5) == (t_bin > 0.5)).float().cpu().numpy()

            mv_cos_all.append(mv_cos)
            aim_cos_all.append(aim_cos)
            elem_acc_all.append(e_ok)
            bin_acc_all.append(b_ok)

            m_phys = (pflag == 2) | (pflag == 3)
            if m_phys.any():
                phys_elem_acc.append(e_ok[m_phys])
            m_fire = pflag == 1
            if m_fire.any():
                fire_dodge_cos.append(mv_cos[m_fire])
            m_sanc = pflag == 4
            if m_sanc.any():
                sanct_cos.append(mv_cos[m_sanc])

    metrics = {
        "name": name,
        "params": count_parameters(net),
        "train_time_s": round(train_s, 2),
        "val_move_cosine": round(float(np.concatenate(mv_cos_all).mean()), 4),
        "val_aim_cosine": round(float(np.concatenate(aim_cos_all).mean()), 4),
        "val_ability_acc_pct": round(float(np.concatenate(bin_acc_all).mean()) * 100.0, 2),
        "val_element_acc_pct": round(float(np.concatenate(elem_acc_all).mean()) * 100.0, 2),
        "val_physics_combo_elem_acc_pct": round(float(np.concatenate(phys_elem_acc).mean()) * 100.0, 2),
        "val_wildfire_evasion_cosine": round(float(np.concatenate(fire_dodge_cos).mean()), 4),
        "val_sanctuary_retreat_cosine": round(float(np.concatenate(sanct_cos).mean()), 4),
    }
    print(f"[{name}] {json.dumps(metrics)}")
    return net, metrics


def main() -> None:
    root = Path("/home/gooning/.t3/scratch/frostmarsh-ml")
    device = torch.device("cuda")
    torch.manual_seed(42)

    wm = WorldModel(in_ch=18, out_ch=17, width=48, num_res=5).to(device).eval()
    ck_w = torch.load(root / "world_model.pt", map_location=device, weights_only=False)
    wm.load_state_dict(ck_w["state_dict"] if isinstance(ck_w, dict) and "state_dict" in ck_w else ck_w)

    em = EcosystemModel(in_ch=8, out_ch=4, width=32, num_res=5).to(device).eval()
    ck_e = torch.load(root / "ecosystem_model.pt", map_location=device, weights_only=False)
    em.load_state_dict(ck_e["state_dict"] if isinstance(ck_e, dict) and "state_dict" in ck_e else ck_e)

    print("Building 16 train + 4 unseen held-out 256x256x4 worlds & extracting real Z_world/Z_eco...")
    train_worlds = []
    for s in range(16):
        ws, es = make_synthetic_3d_world_state(100 + s)
        zw = extract_world_latent(wm, torch.from_numpy(ws).to(device))
        ze = extract_eco_latent(em, torch.from_numpy(es).to(device))
        f, tv, tb, te, _ = sample_expert_batch(ws, es, 1200, np.random.default_rng(1000 + s))
        train_worlds.append((
            zw, ze,
            torch.from_numpy(f).to(device),
            torch.from_numpy(tv).to(device),
            torch.from_numpy(tb).to(device),
            torch.from_numpy(te).to(device),
        ))

    val_worlds = []
    for s in range(4):
        ws, es = make_synthetic_3d_world_state(900 + s)
        zw = extract_world_latent(wm, torch.from_numpy(ws).to(device))
        ze = extract_eco_latent(em, torch.from_numpy(es).to(device))
        f, tv, tb, te, pf = sample_expert_batch(ws, es, 1000, np.random.default_rng(9000 + s))
        val_worlds.append((
            zw, ze,
            torch.from_numpy(f).to(device),
            torch.from_numpy(tv).to(device),
            torch.from_numpy(tb).to(device),
            torch.from_numpy(te).to(device),
            pf,
        ))

    net_shared, m_shared = train_and_eval("SharedLatent (+Z_world +Z_eco)", False, train_worlds, val_worlds, device, steps=2000)
    _, m_blind = train_and_eval("Blind Ablation (Ledger-Only)", True, train_worlds, val_worlds, device, steps=2000)

    # Save trained Shared-Latent BotPolicyNet
    out_pt = root / "bot_policy.pt"
    torch.save({"state_dict": net_shared.state_dict(), "metrics": m_shared}, out_pt)

    # Benchmark batched zero-copy GB10 GPU latency for N = 12, 64, 256 simultaneous neural bots!
    zw0, ze0 = val_worlds[0][0], val_worlds[0][1]
    lat_bench = {}
    net_shared.eval()
    for n_b in (12, 64, 256):
        bf = val_worlds[0][2][:n_b].contiguous()
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
            for _ in range(10):
                _ = net_shared(zw0, ze0, bf)
        torch.cuda.synchronize()
        ts = []
        for _ in range(80):
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
                _ = net_shared(zw0, ze0, bf)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        lat_bench[f"bots_{n_b}"] = {
            "p50_ms": round(float(np.median(ts)), 3),
            "p95_ms": round(float(np.percentile(ts, 95)), 3),
            "per_bot_us": round(float(np.median(ts)) * 1000.0 / n_b, 2),
        }
        print(f"  [GB10 Latency] N={n_b:3d} bots: p50={lat_bench[f'bots_{n_b}']['p50_ms']:.3f} ms ({lat_bench[f'bots_{n_b}']['per_bot_us']:.2f} us/bot)")

    summary = {
        "device": torch.cuda.get_device_name(0),
        "world_latent_shape": list(zw0.shape),
        "eco_latent_shape": list(ze0.shape),
        "shared_latent_bot": m_shared,
        "blind_ablation_bot": m_blind,
        "batched_gpu_latency": lat_bench,
    }
    (root / "BOT_POLICY_METRICS.json").write_text(json.dumps(summary, indent=2))
    print("Saved bot_policy.pt & BOT_POLICY_METRICS.json OK!")


if __name__ == "__main__":
    main()
