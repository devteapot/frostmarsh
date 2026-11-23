#!/usr/bin/env python3
"""
ActuatorPolicyNet v2 League Self-Play RL (`ml/train_actuator_rl_v2.py`).

Warm-starts the EXACT drop-in `ActuatorPolicyNet` architecture (`ml/actuator_policy.py`,
214,624 params, 10 continuous physical actuators, `op=7` compatible!) from Iter 5A
(`models/checkpoints/actuator_policy_rl.pt`) and trains via **League Self-Play** on
Colab A100 (`64` parallel arenas x `12` bots = `768` concurrent bots):

  - League Opponent Mix across 64 arenas (side-balanced West/East):
      * 24 arenas: Pure Mirror (12) + Lagged Self-Play Pool (12)
      * 24 arenas: Directly facing `Legacy Named-Spell Scripted` (12 W + 12 E)
      * 16 arenas: 8 facing `Thermal-Only` ablation + 8 facing frozen `Iter 5A` snapshot
  - Dense Macro + Emergent Physics Shaping (ZERO hardcoded spell rules, ZERO imitation teacher!):
      * Strong Shrine approach, active capture & flip pressure
      * Strong Relic pickup, home-delivery progress & carrier intercept pressure
      * Explicit physical pressure-wave detonation credit (emergent Oil->Ignition & Flash-Boil!)
      * Wildfire / scorch avoidance & mana-budget overload discipline
  - Best-checkpoint tracking on held-out score/win-rate vs Legacy + Iter5A + Thermal-Only
    so no late PPO update can regress!

Evaluates in the EXACT same 24-match side-balanced held-out `256x256x4` tournament
(100% unmodified physics & scoring rules!) against:
  1. Legacy Named-Spell Scripted (TARGET: > 70% WR!)
  2. ActuatorPolicyNet v1 (Iter 5A)
  3. Thermal-Only Actuators
  4. Random Continuous Actuators

Writes:
  - `models/checkpoints/actuator_policy_v2_rl.pt` (pure ActuatorPolicyNet state_dict, op=7 drop-in)
  - `models/ACTUATOR_V2_TOURNAMENT.json`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from actuator_policy import (  # noqa: E402
    ACT_EMIT_DX,
    ACT_EMIT_DY,
    ACT_FOCUS,
    ACT_IMPULSE,
    ACT_MOVE_X,
    ACT_MOVE_Y,
    ACT_OIL,
    ACT_SMOKE,
    ACT_THERMAL,
    ACT_WATER,
    ACTUATOR_DIM,
    BOT_FEAT_DIM,
    MAX_EMIT_RANGE_CELLS,
    ActuatorPolicyNet,
    actuator_area_factor,
    count_parameters,
    focus_radius_cells,
    raw_to_actuators,
    sample_batched_dual_patches,
)
from train_actuator_rl import (  # noqa: E402
    CH_LOGMASS,
    CH_TEMP,
    NUM_BOTS,
    SP_ASH,
    SP_ICE,
    SP_OIL,
    SP_SMOKE,
    SP_STEAM,
    SP_WATER,
    SP_WOOD,
    PhysicsActuatorArena,
    _pearson_from_sums,
    benchmark_inference_latency,
    extract_team_emergence,
    legacy_scripted_actuator_baseline,
    load_ckpt_state_dict,
    load_frozen_models,
    policy_deterministic_act,
    random_continuous_act,
)


# ---------------------------------------------------------------------------
# League Arena: Exact Identical Physics + V2 Dense Macro & Emergent Shaping
# ---------------------------------------------------------------------------


class LeaguePhysicsActuatorArena(PhysicsActuatorArena):
    """
    Inherits 100% identical world generation, neural WorldModel v3 + EcosystemModel stepping,
    observation encoding, combat physics, shrine/relic scoring, and emergence telemetry from
    `PhysicsActuatorArena`.

    Adds `step_league()` with V2 Dense Macro + Emergent Physics Shaping for training ONLY
    (ZERO hardcoded spell rules, ZERO imitation teacher!). Evaluation uses the unmodified
    `super().step()`.
    """

    def step_league(self, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Exact identical physics tick as `PhysicsActuatorArena.step()` with V2 macro + physics shaping."""
        A, N, L, H, W = self.A, self.N, self.L, self.H, self.W
        dev = self.device
        self.tick += 1

        prev_score = self.team_score.clone()
        rewards = torch.zeros((A, N), dtype=torch.float32, device=dev)

        # 1. Respawns (identical)
        dead = ~self.alive
        self.respawn_cd = torch.where(dead, (self.respawn_cd - 1).clamp(min=0), self.respawn_cd)
        do_respawn = dead & (self.respawn_cd == 0)
        if do_respawn.any():
            base_x = torch.where(self.team == 0, 38.0, 218.0)
            rjx = torch.empty((A, N), device=dev).uniform_(-12.0, 12.0, generator=self.gen)
            rjy = torch.empty((A, N), device=dev).uniform_(36.0, 220.0, generator=self.gen)
            self.pos_x = torch.where(do_respawn, (base_x + rjx).clamp(16.0, 240.0), self.pos_x)
            self.pos_y = torch.where(do_respawn, rjy, self.pos_y)
            self.vel_x = torch.where(do_respawn, torch.zeros_like(self.vel_x), self.vel_x)
            self.vel_y = torch.where(do_respawn, torch.zeros_like(self.vel_y), self.vel_y)
            self.bot_layer = torch.where(do_respawn, torch.full_like(self.bot_layer, self.surf_l), self.bot_layer)
            self.hp = torch.where(do_respawn, torch.ones_like(self.hp), self.hp)
            self.mana = torch.where(do_respawn, torch.ones_like(self.mana), self.mana)
            self.stam = torch.where(do_respawn, torch.ones_like(self.stam), self.stam)
            self.alive = self.alive | do_respawn
            self.last_attacker = torch.where(do_respawn, torch.full_like(self.last_attacker, -1), self.last_attacker)

        alive_f = self.alive.float()

        # 2. Regen & TTL decay (identical)
        self.mana = (self.mana + 0.20 * alive_f).clamp(0.0, 1.0)
        self.stam = (self.stam + 0.08 * alive_f).clamp(0.0, 1.0)
        self.cell_src_ttl = (self.cell_src_ttl - 1).clamp(min=0)
        self.cell_src_bot = torch.where(self.cell_src_ttl > 0, self.cell_src_bot, torch.full_like(self.cell_src_bot, -1))
        self.cell_oil_ttl = (self.cell_oil_ttl - 1).clamp(min=0)
        self.cell_oil_bot = torch.where(self.cell_oil_ttl > 0, self.cell_oil_bot, torch.full_like(self.cell_oil_bot, -1))

        # Decode 10 continuous physical actuators (identical)
        mvx = act[:, :, ACT_MOVE_X].clamp(-1.0, 1.0)
        mvy = act[:, :, ACT_MOVE_Y].clamp(-1.0, 1.0)
        mnorm = torch.hypot(mvx, mvy).clamp(min=1.0)
        mvx = mvx / mnorm
        mvy = mvy / mnorm

        emit_dx = act[:, :, ACT_EMIT_DX].clamp(-1.0, 1.0)
        emit_dy = act[:, :, ACT_EMIT_DY].clamp(-1.0, 1.0)
        thermal_in = act[:, :, ACT_THERMAL].clamp(-1.0, 1.0)
        sp_water_in = act[:, :, ACT_WATER].clamp(0.0, 1.0)
        sp_oil_in = act[:, :, ACT_OIL].clamp(0.0, 1.0)
        sp_smoke_in = act[:, :, ACT_SMOKE].clamp(0.0, 1.0)
        focus_01 = act[:, :, ACT_FOCUS].clamp(0.0, 1.0)
        impulse_in = act[:, :, ACT_IMPULSE].clamp(0.0, 1.0)

        thermal_raw = torch.where(thermal_in.abs() > 0.16, thermal_in, torch.zeros_like(thermal_in))
        water_raw = F.relu(sp_water_in - 0.30) / 0.70
        oil_raw = F.relu(sp_oil_in - 0.28) / 0.72
        smoke_raw = F.relu(sp_smoke_in - 0.30) / 0.70
        impulse_raw = F.relu(impulse_in - 0.28) / 0.72

        # 3. Conservation + Continuous Mana Budget (identical)
        area_f = actuator_area_factor(focus_01)
        raw_flux_sum = thermal_raw.abs() + water_raw + oil_raw + smoke_raw + impulse_raw
        raw_cost = raw_flux_sum * area_f * 0.22
        has_req = (raw_cost > 1e-6) & self.alive
        can_afford = has_req & (raw_cost <= self.mana + 1e-6)
        overloaded = has_req & (~can_afford)
        budget_scale = can_afford.float()
        mana_spent = torch.where(can_afford, raw_cost, torch.where(overloaded, 0.20 * self.mana, torch.zeros_like(raw_cost)))
        self.mana = (self.mana - mana_spent).clamp(0.0, 1.0)

        eff_thermal = thermal_raw * budget_scale
        eff_water = water_raw * budget_scale
        eff_oil = oil_raw * budget_scale
        eff_smoke = smoke_raw * budget_scale
        eff_impulse = impulse_raw * budget_scale
        self.last_emit_thermal = eff_thermal

        rad_cells = focus_radius_cells(focus_01)
        conc = 1.55 / (0.55 + 0.80 * focus_01)

        # 4. Locomotion, Surface Friction & Buoyancy (identical)
        gx0, gy0 = self._grid_xy(self.pos_x, self.pos_y)
        cur_water = self.w_state[self.arena_idx, self.bot_layer, SP_WATER, gy0, gx0]
        cur_ice = self.w_state[self.arena_idx, self.bot_layer, SP_ICE, gy0, gx0]
        cur_oil = self.w_state[self.arena_idx, self.bot_layer, SP_OIL, gy0, gx0]
        cur_temp0 = self.w_state[self.arena_idx, self.bot_layer, CH_TEMP, gy0, gx0]

        in_deep_water = (cur_water > 0.42) & (cur_ice < 0.35)
        on_ice = cur_ice > 0.35
        on_oil = cur_oil > 0.22
        cryo_drag = (1.0 - 0.45 * F.relu(-0.15 - cur_temp0)).clamp(0.45, 1.0)

        base_spd = torch.where(self.cls == 2, 4.6, torch.where(self.cls == 0, 4.1, 4.35))
        spd = base_spd * cryo_drag
        spd = torch.where(in_deep_water, spd * 0.30, spd)
        spd = torch.where(on_ice, spd * 1.32, spd)
        spd = torch.where(self.carrying, spd * 0.90, spd)

        emit_dist = torch.hypot(emit_dx, emit_dy) * MAX_EMIT_RANGE_CELLS
        e_norm = torch.hypot(emit_dx, emit_dy).clamp(min=1e-4)
        eux, euy = emit_dx / e_norm, emit_dy / e_norm
        dash_align = ((eux * mvx + euy * mvy) < 0.15) | (emit_dist < 10.0)
        recoil_boost = torch.where(dash_align, 2.8 * eff_impulse, 0.8 * eff_impulse)
        spd = spd + recoil_boost

        des_vx = mvx * spd * alive_f
        des_vy = mvy * spd * alive_f
        slip = torch.where(on_oil, 0.38, torch.where(on_ice, 0.22, 0.08))
        self.vel_x = slip * self.vel_x + (1.0 - slip) * des_vx
        self.vel_y = slip * self.vel_y + (1.0 - slip) * des_vy

        prev_x, prev_y = self.pos_x.clone(), self.pos_y.clone()
        self.pos_x = (self.pos_x + self.vel_x * alive_f).clamp(10.0, 246.0)
        self.pos_y = (self.pos_y + self.vel_y * alive_f).clamp(10.0, 246.0)

        gx, gy = self._grid_xy(self.pos_x, self.pos_y)
        cur_crit = self.e_state[self.arena_idx, 0, 0, gy, gx]
        sanc_heal = (cur_crit > 0.30) & (self.hp < 0.70) & self.alive
        self.hp = torch.where(sanc_heal, (self.hp + 0.025).clamp(max=1.0), self.hp)

        # 5. Physical Voxel Deposits (identical)
        raw_ex = (self.pos_x + emit_dx * MAX_EMIT_RANGE_CELLS).clamp(8.0, 248.0)
        raw_ey = (self.pos_y + emit_dy * MAX_EMIT_RANGE_CELLS).clamp(8.0, 248.0)
        egx0, egy0 = self._grid_xy(raw_ex, raw_ey)
        mgx0, mgy0 = self._grid_xy(0.5 * (self.pos_x + raw_ex), 0.5 * (self.pos_y + raw_ey))

        opt_tgt = (
            self.w_state[self.arena_idx, self.bot_layer, SP_SMOKE, egy0, egx0]
            + 0.85 * self.w_state[self.arena_idx, self.bot_layer, SP_STEAM, egy0, egx0]
        )
        opt_mid = (
            self.w_state[self.arena_idx, self.bot_layer, SP_SMOKE, mgy0, mgx0]
            + 0.85 * self.w_state[self.arena_idx, self.bot_layer, SP_STEAM, mgy0, mgx0]
        )
        opt_path = torch.maximum(opt_tgt, 0.75 * opt_mid)
        scatter_std = F.relu(opt_path - 0.25) * 6.5 * (emit_dist > 7.0).float()
        ex = (raw_ex + torch.randn((A, N), generator=self.gen, device=dev) * scatter_std).clamp(8.0, 248.0)
        ey = (raw_ey + torch.randn((A, N), generator=self.gen, device=dev) * scatter_std).clamp(8.0, 248.0)
        egx, egy = self._grid_xy(ex, ey)

        prev_temp = self.w_state[:, :, CH_TEMP].clone()
        prev_steam = self.w_state[:, :, SP_STEAM].clone()

        sigma = (0.55 * rad_cells).unsqueeze(-1)
        d_s = self.st_dist.view(1, 1, -1)
        g_w = torch.exp(-0.5 * (d_s / sigma).pow(2)) * (d_s <= (rad_cells.unsqueeze(-1) * 1.25)).float()

        sy = (egy.unsqueeze(-1) + self.st_dy.view(1, 1, -1)).clamp(0, H - 1)
        sx = (egx.unsqueeze(-1) + self.st_dx.view(1, 1, -1)).clamp(0, W - 1)
        a_id = self.arena_idx.unsqueeze(-1).expand_as(sy)
        l_id = self.bot_layer.unsqueeze(-1).expand_as(sy)
        al_id = a_id * L + l_id
        yx_id = sy * W + sx
        flat_idx = (al_id * (H * W) + yx_id).reshape(-1)

        w_flat = self.w_state.view(-1)
        hw = H * W

        def _scatter_ch(ch: int, vals_3d: torch.Tensor) -> None:
            ch_idx = ((al_id * 18 + ch) * hw + yx_id).reshape(-1)
            w_flat.scatter_add_(0, ch_idx, vals_3d.reshape(-1))

        d_temp_amp = eff_thermal * conc * 0.54
        if (d_temp_amp.abs() > 1e-4).any():
            _scatter_ch(CH_TEMP, d_temp_amp.unsqueeze(-1) * g_w)

        mass_scale = 0.55 + 0.25 * conc
        if (eff_water > 1e-4).any():
            _scatter_ch(SP_WATER, (eff_water * mass_scale * 0.95).unsqueeze(-1) * g_w)
        if (eff_oil > 1e-4).any():
            _scatter_ch(SP_OIL, (eff_oil * mass_scale * 1.10).unsqueeze(-1) * g_w)
        if (eff_smoke > 1e-4).any():
            _scatter_ch(SP_SMOKE, (eff_smoke * mass_scale * 1.25).unsqueeze(-1) * g_w)

        core_m = g_w > 0.35
        therm_active = (eff_thermal.abs() > 0.14).unsqueeze(-1) & core_m
        if therm_active.any():
            b_flat = self.bot_idx.unsqueeze(-1).expand_as(sy).reshape(-1)
            m_flat = therm_active.reshape(-1)
            self.cell_src_bot.view(-1)[flat_idx[m_flat]] = b_flat[m_flat]
            self.cell_src_ttl.view(-1)[flat_idx[m_flat]] = 6

        oil_active = (eff_oil > 0.16).unsqueeze(-1) & core_m
        if oil_active.any():
            b_flat = self.bot_idx.unsqueeze(-1).expand_as(sy).reshape(-1)
            m_flat = oil_active.reshape(-1)
            self.cell_oil_bot.view(-1)[flat_idx[m_flat]] = b_flat[m_flat]
            self.cell_oil_ttl.view(-1)[flat_idx[m_flat]] = 10

        # 6. Voxel Thermodynamics & WorldModel v3 Step (identical)
        temp = self.w_state[:, :, CH_TEMP]
        water = self.w_state[:, :, SP_WATER]
        ice = self.w_state[:, :, SP_ICE]
        oil = self.w_state[:, :, SP_OIL]
        wood = self.w_state[:, :, SP_WOOD]
        smoke = self.w_state[:, :, SP_SMOKE]
        steam = self.w_state[:, :, SP_STEAM]
        ash = self.w_state[:, :, SP_ASH]

        oil_ign_rate = 0.62 * torch.sigmoid((temp - 0.52) * 5.0) * (temp > 0.42).float()
        burn_oil = torch.minimum(oil, oil * oil_ign_rate)
        oil = (oil - burn_oil).clamp(min=0.0)
        temp = temp + 4.20 * burn_oil
        smoke = smoke + 0.65 * burn_oil
        ash = ash + 0.15 * burn_oil

        wood_ign = 0.14 * torch.sigmoid((temp - 1.65) * 3.0) * (temp > 1.45).float()
        burn_wood = torch.minimum(wood, wood * wood_ign)
        wood = (wood - burn_wood).clamp(min=0.0)
        temp = temp + 0.95 * burn_wood
        smoke = smoke + 0.35 * burn_wood

        boil_w_rate = 0.64 * ((temp - 0.45) / 0.95).clamp(0.0, 1.0)
        boil_w = torch.minimum(water, water * boil_w_rate)
        boil_i_rate = 0.55 * ((temp - 0.38) / 0.95).clamp(0.0, 1.0)
        boil_i = torch.minimum(ice, ice * boil_i_rate)
        water = (water - boil_w).clamp(min=0.0)
        ice = (ice - boil_i).clamp(min=0.0)
        steam = steam + 1.40 * (boil_w + boil_i)
        temp = temp - 0.14 * (boil_w + boil_i)

        frz_rate = 0.80 * ((-0.08 - temp) / 0.45).clamp(0.0, 1.0)
        freeze_w = torch.minimum(water, water * frz_rate)
        water = (water - freeze_w).clamp(min=0.0)
        ice = ice + freeze_w

        melt_rate = 0.22 * ((temp - 0.25) / 1.0).clamp(0.0, 1.0)
        melt_i = torch.minimum(ice, ice * melt_rate)
        ice = (ice - melt_i).clamp(min=0.0)
        water = water + melt_i

        temp = torch.where(temp > 0.05, temp * 0.92, temp * 0.88).clamp(-1.4, 5.5)
        smoke = (smoke * 0.86).clamp(0.0, 2.0)
        steam = (steam * 0.82).clamp(0.0, 2.0)
        oil = oil.clamp(0.0, 2.0)
        water = water.clamp(0.0, 2.0)
        ice = ice.clamp(0.0, 2.0)

        self.w_state[:, :, CH_TEMP] = temp
        self.w_state[:, :, SP_WATER] = water
        self.w_state[:, :, SP_ICE] = ice
        self.w_state[:, :, SP_OIL] = oil
        self.w_state[:, :, SP_WOOD] = wood
        self.w_state[:, :, SP_SMOKE] = smoke
        self.w_state[:, :, SP_STEAM] = steam
        self.w_state[:, :, SP_ASH] = ash

        if (self.tick % self.refresh_every) == 0:
            self.refresh_latents_and_world_step(
                apply_world_deltas=True,
                refresh_eco=(self.tick % (self.refresh_every * 4) == 0),
            )

        self.w_state[:, :, CH_LOGMASS] = torch.log1p(self.w_state[:, :, :10].sum(dim=2))

        dT = self.w_state[:, :, CH_TEMP] - prev_temp
        dSteam = self.w_state[:, :, SP_STEAM] - prev_steam

        # 7. Entity Effects: Thermal Stress + Pressure Waves + Kinetic Wave (identical)
        cur_T = self.w_state[self.arena_idx, self.bot_layer, CH_TEMP, gy, gx]
        scorch_dmg = (0.048 * F.relu(cur_T - 1.55).pow(2)).clamp(max=0.26) * alive_f
        hypo_dmg = (0.14 * F.relu(-0.15 - cur_T).pow(2)).clamp(max=0.20) * alive_f
        thermal_dmg = scorch_dmg + hypo_dmg

        p_comb = F.relu(dT - 0.65)
        p_boil = F.relu(dSteam - 0.25)
        p_raw = (0.34 * p_comb + 0.42 * p_boil).clamp(max=0.52)
        p_diff = F.avg_pool2d(p_raw.reshape(A * L, 1, H, W), kernel_size=5, stride=1, padding=2).reshape(A, L, H, W)
        p_field = torch.maximum(p_raw, 1.45 * p_diff).clamp(max=0.52)

        p_pad = F.pad(p_field, (1, 1, 1, 1), mode="replicate")
        grad_px = 0.5 * (p_pad[:, :, 1:-1, 2:] - p_pad[:, :, 1:-1, :-2])
        grad_py = 0.5 * (p_pad[:, :, 2:, 1:-1] - p_pad[:, :, :-2, 1:-1])

        p_loc = p_field[self.arena_idx, self.bot_layer, gy, gx] * alive_f
        kb_px = -grad_px[self.arena_idx, self.bot_layer, gy, gx] * 16.0 * alive_f
        kb_py = -grad_py[self.arena_idx, self.bot_layer, gy, gx] * 16.0 * alive_f

        k_dx = self.pos_x[:, None, :] - ex[:, :, None]
        k_dy = self.pos_y[:, None, :] - ey[:, :, None]
        k_dist = torch.hypot(k_dx, k_dy)
        k_rad = (rad_cells * 1.45).unsqueeze(-1)
        valid_en = self.enemy_pair_mask & self.alive[:, :, None] & self.alive[:, None, :]
        in_kin = valid_en & (k_dist < k_rad) & (eff_impulse.unsqueeze(-1) > 0.08)
        kin_falloff = (1.0 - (k_dist / (k_rad + 1e-4))).clamp(0.0, 1.0) * in_kin.float()
        kin_dmg_mat = kin_falloff * (0.13 * eff_impulse / (0.60 + 0.85 * focus_01)).unsqueeze(-1)
        kin_dmg_taken = kin_dmg_mat.sum(dim=1)
        kin_dmg_dealt = kin_dmg_mat.sum(dim=2)

        kin_kb_x = (kin_falloff * (eff_impulse * eux * 5.2).unsqueeze(-1)).sum(dim=1)
        kin_kb_y = (kin_falloff * (eff_impulse * euy * 5.2).unsqueeze(-1)).sum(dim=1)

        self.vel_x = (self.vel_x + kb_px + kin_kb_x).clamp(-10.0, 10.0)
        self.vel_y = (self.vel_y + kb_py + kin_kb_y).clamp(-10.0, 10.0)
        self.pos_x = (self.pos_x + (kb_px + kin_kb_x) * 0.45 * alive_f).clamp(10.0, 246.0)
        self.pos_y = (self.pos_y + (kb_py + kin_kb_y) * 0.45 * alive_f).clamp(10.0, 246.0)

        src_bot = self.cell_src_bot[self.arena_idx, self.bot_layer, gy, gx]
        has_src = src_bot >= 0
        src_safe = src_bot.clamp(min=0)
        src_team = torch.gather(self.team, 1, src_safe)
        is_enemy_src = has_src & (src_team != self.team)

        voxel_dmg = thermal_dmg + p_loc
        armor = torch.where(self.cls == 0, 0.84, 1.0)
        total_dmg_taken = (voxel_dmg + kin_dmg_taken) * armor
        self.hp = (self.hp - total_dmg_taken).clamp(min=0.0)

        dmg_dealt_by_i = kin_dmg_dealt.clone()
        en_voxel_dmg = voxel_dmg * is_enemy_src.float()
        dmg_dealt_by_i.scatter_add_(1, src_safe, en_voxel_dmg)

        # Extra intercept credit if damaging an enemy relic carrier
        carrier_dmg_bonus = torch.zeros((A, N), device=dev)
        carrier_dmg_bonus.scatter_add_(1, src_safe, en_voxel_dmg * self.carrying.float())
        carrier_dmg_bonus += (kin_dmg_mat * self.carrying[:, None, :].float()).sum(dim=2)

        oil_src_bot = self.cell_oil_bot[self.arena_idx, self.bot_layer, gy, gx]
        oil_src_safe = oil_src_bot.clamp(min=0)
        is_enemy_oil_src = (oil_src_bot >= 0) & (torch.gather(self.team, 1, oil_src_safe) != self.team)
        comb_loc = (0.24 * p_comb[self.arena_idx, self.bot_layer, gy, gx]).clamp(max=0.35) * alive_f
        oil_comb_credit = torch.zeros((A, N), device=dev)
        oil_comb_credit.scatter_add_(1, oil_src_safe, 0.55 * comb_loc * is_enemy_oil_src.float())
        dmg_dealt_by_i = dmg_dealt_by_i + oil_comb_credit

        en_press_dmg = p_loc * is_enemy_src.float()
        press_dealt_by_i = torch.zeros((A, N), device=dev)
        press_dealt_by_i.scatter_add_(1, src_safe, en_press_dmg)
        press_dealt_by_i = press_dealt_by_i + oil_comb_credit

        max_kin, best_kin = kin_dmg_mat.max(dim=1)
        self.last_attacker = torch.where(
            (en_voxel_dmg > 1e-4) & (en_voxel_dmg >= max_kin),
            src_safe,
            torch.where(max_kin > 1e-4, best_kin, self.last_attacker),
        )

        # 8. Eliminations & Kill Credit (identical scoring, V2 reward weights)
        just_died = self.alive & (self.hp <= 0.0)
        if just_died.any():
            self.alive = self.alive & (~just_died)
            self.respawn_cd = torch.where(just_died, torch.full_like(self.respawn_cd, 8), self.respawn_cd)

            dropped = just_died & self.carrying
            if dropped.any():
                self.carrying = self.carrying & (~dropped)
                drop_arena = dropped.any(dim=1)
                self.relic_carrier = torch.where(drop_arena, torch.full_like(self.relic_carrier, -1), self.relic_carrier)

            west_died = (just_died.float() * self.w_mask).sum(dim=1)
            east_died = (just_died.float() * self.e_mask).sum(dim=1)
            self.team_kills[:, 1] += west_died
            self.team_kills[:, 0] += east_died
            self.team_score[:, 1] += 8.0 * west_died
            self.team_score[:, 0] += 8.0 * east_died

            valid_kill = just_died & (self.last_attacker >= 0)
            killer_idx = self.last_attacker.clamp(min=0)
            rewards.scatter_add_(1, killer_idx, 5.5 * valid_kill.float())
            rewards -= 3.0 * just_died.float()

        # 9. Shrine Control & V2 Dense Macro Shaping (identical scoring!)
        prev_T = prev_temp[self.arena_idx, self.bot_layer, gy0, gx0]
        prev_in_fire = (prev_T > 1.55) & self.alive
        in_fire = (cur_T > 1.55) & self.alive
        escaped_fire = prev_in_fire & (~in_fire) & self.alive
        sd_bs = torch.hypot(
            self.pos_x[:, :, None] - self.shrine_x[:, None, :],
            self.pos_y[:, :, None] - self.shrine_y[:, None, :],
        )
        in_shrine = (sd_bs < 18.0) & self.alive[:, :, None] & (~in_fire[:, :, None])
        west_in = (in_shrine & (self.team[:, :, None] == 0)).float().sum(dim=1)
        east_in = (in_shrine & (self.team[:, :, None] == 1)).float().sum(dim=1)

        self.shrine_prog = (self.shrine_prog + 0.12 * (east_in - west_in)).clamp(-1.0, 1.0)
        new_owner = torch.where(
            self.shrine_prog <= -0.8,
            torch.zeros_like(self.shrine_owner),
            torch.where(self.shrine_prog >= 0.8, torch.ones_like(self.shrine_owner), self.shrine_owner),
        )
        flipped_w = (new_owner == 0) & (self.shrine_owner != 0)
        flipped_e = (new_owner == 1) & (self.shrine_owner != 1)
        self.shrine_owner = new_owner

        self.team_shrine_caps[:, 0] += flipped_w.float().sum(dim=1)
        self.team_shrine_caps[:, 1] += flipped_e.float().sum(dim=1)
        self.team_score[:, 0] += 10.0 * flipped_w.float().sum(dim=1) + (self.shrine_owner == 0).float().sum(dim=1) * 1.0
        self.team_score[:, 1] += 10.0 * flipped_e.float().sum(dim=1) + (self.shrine_owner == 1).float().sum(dim=1) * 1.0

        # Dense 3-lane shrine approach + capture + flip reward (breaks Shrine-1 clumping!
        # GATED OFF while `in_fire` so bots are NEVER punished for sidestepping out of fire!)
        need_shrine = torch.where(
            self.team[:, :, None] == 0,
            self.shrine_prog[:, None, :] > -0.8,
            self.shrine_prog[:, None, :] < 0.8,
        )
        prev_sd_bs = torch.hypot(
            prev_x[:, :, None] - self.shrine_x[:, None, :],
            prev_y[:, :, None] - self.shrine_y[:, None, :],
        )
        pref_s0 = (self.pos_y < 112.0) & need_shrine[:, :, 0]
        pref_s2 = (self.pos_y > 144.0) & need_shrine[:, :, 2]
        pref_s1 = (~pref_s0) & (~pref_s2) & need_shrine[:, :, 1]
        tie_bias = torch.zeros_like(sd_bs)
        tie_bias[:, :, 0] = torch.where(self.pos_y <= 128.0, -1.0, 1.0)
        tie_bias[:, :, 2] = torch.where(self.pos_y > 128.0, -1.0, 1.0)
        nearest_need_idx = (sd_bs + tie_bias).masked_fill(~need_shrine, 1e4).argmin(dim=-1)
        tgt_s_idx = torch.where(
            pref_s0,
            torch.zeros_like(nearest_need_idx),
            torch.where(
                pref_s2,
                torch.full_like(nearest_need_idx, 2),
                torch.where(pref_s1, torch.ones_like(nearest_need_idx), nearest_need_idx),
            ),
        )
        prev_s_tgt = torch.gather(prev_sd_bs, 2, tgt_s_idx.unsqueeze(-1)).squeeze(-1)
        new_s_tgt = torch.gather(sd_bs, 2, tgt_s_idx.unsqueeze(-1)).squeeze(-1)

        no_carrier = self.relic_carrier < 0
        has_carrier_pre = self.relic_carrier >= 0
        c_team_pre = torch.where(
            has_carrier_pre,
            torch.gather(self.team, 1, self.relic_carrier.clamp(min=0).unsqueeze(1)).squeeze(1),
            torch.full_like(self.relic_carrier, -1),
        )
        my_team_has_relic = has_carrier_pre.unsqueeze(1) & (c_team_pre.unsqueeze(1) == self.team)
        en_team_has_relic = has_carrier_pre.unsqueeze(1) & (c_team_pre.unsqueeze(1) != self.team)

        # Runners (cls==0) contest free relic AND intercept enemy relic carriers (token 11 tracks both!)
        runner_on_relic_duty = (self.cls == 0) & (~my_team_has_relic)
        shrine_w = torch.where(runner_on_relic_duty, 0.08, 0.78)
        has_need_shrine = need_shrine.any(dim=-1) & (~self.carrying) & (~in_fire) & (~prev_in_fire) & self.alive
        shrine_approach = torch.where(has_need_shrine, (prev_s_tgt - new_s_tgt).clamp(-6.0, 6.0), torch.zeros_like(rewards))
        shrine_contrib = (in_shrine & need_shrine).any(dim=-1).float()
        shrine_contrib_w = torch.where(runner_on_relic_duty, 0.50, 3.20)
        rewards += shrine_contrib_w * shrine_contrib + shrine_w * shrine_approach

        flip_for_team = torch.where(self.team[:, :, None] == 0, flipped_w[:, None, :], flipped_e[:, None, :])
        in_flipped = (in_shrine & flip_for_team).any(dim=-1).float()
        rewards += 7.0 * in_flipped
        owned_shrines_count = torch.where(
            self.team == 0,
            (self.shrine_owner == 0).float().sum(dim=1, keepdim=True),
            (self.shrine_owner == 1).float().sum(dim=1, keepdim=True),
        )
        rewards += 0.48 * owned_shrines_count * alive_f

        # 10. Golden Relic Pickup, Enemy-Carrier Intercept, Carry & Delivery (identical scoring!)
        rdist_prev = torch.hypot(prev_x - self.relic_x[:, None], prev_y - self.relic_y[:, None])
        rdist = torch.hypot(self.pos_x - self.relic_x[:, None], self.pos_y - self.relic_y[:, None])
        rel_delta = (rdist_prev - rdist).clamp(-6.0, 6.0)
        if no_carrier.any():
            is_runner = (self.cls == 0) & self.alive & (~in_fire) & no_carrier.unsqueeze(1)
            near_relic_helper = (self.cls != 0) & self.alive & (~in_fire) & no_carrier.unsqueeze(1) & (rdist < 28.0)
            rewards += torch.where(is_runner, 0.78 * rel_delta, torch.where(near_relic_helper, 0.25 * rel_delta, torch.zeros_like(rewards)))

            rdist_m = rdist.masked_fill(~self.alive, 1e6)
            r_min, r_who = rdist_m.min(dim=1)
            picked = no_carrier & (r_min < 12.0)
            self.relic_carrier = torch.where(picked, r_who, self.relic_carrier)
            if picked.any():
                oh = F.one_hot(r_who, num_classes=N).bool() & picked.unsqueeze(-1)
                self.carrying = self.carrying | oh
                rewards += 6.5 * oh.float()

        # Intercept enemy relic carrier (token 11 points right at them!)
        if en_team_has_relic.any():
            intercept_m = en_team_has_relic & self.alive & (~in_fire) & ((self.cls == 0) | (rdist < 42.0))
            rewards += torch.where(intercept_m, 0.65 * rel_delta, torch.zeros_like(rewards))

        has_carrier = self.relic_carrier >= 0
        if has_carrier.any():
            cid = self.relic_carrier.clamp(min=0)
            cx = torch.gather(self.pos_x, 1, cid.unsqueeze(1)).squeeze(1)
            cy = torch.gather(self.pos_y, 1, cid.unsqueeze(1)).squeeze(1)
            self.relic_x = torch.where(has_carrier, cx, self.relic_x)
            self.relic_y = torch.where(has_carrier, cy, self.relic_y)

            c_team = torch.gather(self.team, 1, cid.unsqueeze(1)).squeeze(1)
            self.team_score[:, 0] += (has_carrier & (c_team == 0)).float() * 0.25
            self.team_score[:, 1] += (has_carrier & (c_team == 1)).float() * 0.25

            home_x = torch.where(self.team == 0, 28.0, 228.0)
            prev_hd = (prev_x - home_x).abs()
            new_hd = (self.pos_x - home_x).abs()
            rewards += torch.where(
                self.carrying & self.alive,
                1.50 * (prev_hd - new_hd).clamp(-6.0, 6.0) + 0.35,
                torch.zeros_like(rewards),
            )

            deliv_w = has_carrier & (c_team == 0) & (cx < 38.0)
            deliv_e = has_carrier & (c_team == 1) & (cx > 218.0)
            deliv = deliv_w | deliv_e
            if deliv.any():
                self.team_score[:, 0] += 25.0 * deliv_w.float()
                self.team_score[:, 1] += 25.0 * deliv_e.float()
                self.team_relic_deliveries[:, 0] += deliv_w.float()
                self.team_relic_deliveries[:, 1] += deliv_e.float()
                rewards += 22.0 * (self.carrying & deliv.unsqueeze(-1)).float()
                team_deliv = torch.where(self.team == 0, deliv_w[:, None], deliv_e[:, None]).float()
                rewards += 3.5 * team_deliv
                self.carrying = self.carrying & (~deliv.unsqueeze(-1))
                self.relic_carrier = torch.where(deliv, torch.full_like(self.relic_carrier, -1), self.relic_carrier)
                self.relic_x = torch.where(deliv, torch.full_like(self.relic_x, 128.0), self.relic_x)
                self.relic_y = torch.where(deliv, torch.full_like(self.relic_y, 128.0), self.relic_y)

        # 11. V2 Dual-Horizon Rewards (Macro + Emergent Physics + Wildfire/Water Drag + Mana Discipline)
        score_delta = self.team_score - prev_score
        my_team_delta = torch.where(self.team == 0, score_delta[:, 0:1], score_delta[:, 1:2])
        en_team_delta = torch.where(self.team == 0, score_delta[:, 1:2], score_delta[:, 0:1])

        # Pure physical terrain/resource/contact feedback (ZERO hardcoded spell rules!):
        #   - wildfire_pen: avoid standing in burning cells
        #   - water_bog_pen: deep unfrozen water drags speed to 0.30x (naturally rewards cryo-bridging!)
        #   - overload_pen: unaffordable multi-channel overload fizzles & wastes mana (teaches crisp pulses!)
        #   - vuln_dmg: extra survival urgency when low HP or carrying relic (naturally rewards cover/evasion!)
        #   - emit_contact: physical emission landing near an in-range enemy (credits oil/prep deposits before detonation!)
        en_d_bot = torch.hypot(self.pos_x[:, None, :] - self.pos_x[:, :, None], self.pos_y[:, None, :] - self.pos_y[:, :, None])
        en_d_bot = en_d_bot.masked_fill(~valid_en, 1e6)
        has_en_in_range = (en_d_bot.min(dim=-1).values < 38.0) & self.alive
        en_d_emit = torch.hypot(self.pos_x[:, None, :] - ex[:, :, None], self.pos_y[:, None, :] - ey[:, :, None])
        en_d_emit = en_d_emit.masked_fill(~valid_en, 1e6)
        min_en_emit_d = en_d_emit.min(dim=-1).values
        off_emit = ((eff_thermal.abs() + eff_oil + eff_water + eff_impulse) > 0.18) & has_en_in_range
        emit_contact = torch.where(off_emit, torch.exp(-min_en_emit_d / 6.5) - 0.18, torch.zeros_like(rewards))

        wildfire_pen = 3.20 * scorch_dmg + 1.10 * in_fire.float()
        fire_escape_bonus = 1.60 * escaped_fire.float()
        water_bog_pen = 0.45 * (in_deep_water & self.alive).float()
        overload_pen = 0.80 * overloaded.float()
        vuln_mask = ((self.hp < 0.65) | self.carrying).float()
        dmg_taken_pen = (1.10 + 1.35 * vuln_mask) * total_dmg_taken

        rew_tac = (
            rewards
            + fire_escape_bonus
            + 3.5 * dmg_dealt_by_i
            + 6.5 * press_dealt_by_i
            + 6.0 * carrier_dmg_bonus
            + 0.40 * emit_contact
            - dmg_taken_pen
            - wildfire_pen
            - water_bog_pen
            + 1.30 * my_team_delta
            - 0.55 * en_team_delta
            + 0.05 * (alive_f * (~in_fire).float())
        )
        rew_act = (
            12.0 * dmg_dealt_by_i
            + 18.0 * press_dealt_by_i
            + 7.5 * carrier_dmg_bonus
            + 1.60 * emit_contact
            - dmg_taken_pen
            - 0.40 * wildfire_pen
            - water_bog_pen
            - overload_pen
            + 0.55 * my_team_delta
            - 0.25 * en_team_delta
            + 0.55 * shrine_contrib
        )
        rews_pair = torch.stack([rew_tac, rew_act], dim=-1)

        done = self.tick >= self.max_ticks
        return self.build_bot_features(), rews_pair, done


# ---------------------------------------------------------------------------
# PPO Actor-Critic Wrapper (Structured Pointer-Categorical + Continuous Raw-Space PPO!)
# ---------------------------------------------------------------------------


class RLActuatorActorCriticV2(nn.Module):
    """
    Wraps the EXACT unmodified `ActuatorPolicyNet` (214,624 params, op=7 drop-in) with:
      - Dual-Horizon FP32 Value Critic
      - Direct Categorical PPO exploration over `policy.head_ptr_mv` (5 slots: Shrine0/1/2/Relic-Home/Free)
        and `policy.head_ptr_em` (13 slots: 12 entity/objective tokens + Free)!
      - Pre-Activation Continuous Gaussian PPO over the 10 raw bipolar/unipolar outputs!

    Why this is a breakthrough (100% pure RL, ZERO imitation, ZERO hardcoded spell rules):
      - Sampling `ptr_mv ~ Categorical(head_ptr_mv)` breaks symmetric Shrine-0 vs Shrine-2 cancellation
        immediately: bots explore walking cleanly to Shrine 0, Shrine 1, Shrine 2, Relic/Home, or Free,
        and sharpen `head_ptr_mv` directly via exact Categorical policy gradients!
      - At inference (`ActuatorPolicyNet.forward()`), sharpened softmax pointers produce the exact same
        crisp navigation & targeting behavior with zero wrapper overhead!
    """

    def __init__(self, policy: ActuatorPolicyNet):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        # Pre-activation logit-space stds for [res_mv(2), res_em(2), thermal(1), water, oil, smoke, focus, impulse]
        init_log_std = torch.tensor(
            [-0.45, -0.45, -0.45, -0.45, -0.22, -0.28, -0.22, -0.24, -0.32, -0.26],
            dtype=torch.float32,
        )
        self.log_std = nn.Parameter(init_log_std)
        # Stage 2 soft-pointer logit-space exploration stds [ptr_mv_std, ptr_em_std]
        self.ptr_log_std = nn.Parameter(torch.tensor([-0.35, -0.15], dtype=torch.float32))

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-1.45, -0.05))

    def get_ptr_std(self) -> torch.Tensor:
        return torch.exp(self.ptr_log_std.clamp(-1.35, 0.10))

    def _heads_from_patches(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        use_bf16: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run exact ActuatorPolicyNet trunk and extract (ptr_mv_logits[M,5], ptr_em_logits[M,13], raw_cont[M,10], h_head[M,256])."""
        if use_bf16 and fine_patches.device.type == "cuda" and torch.cuda.is_bf16_supported():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, h_head, _ = self.policy.forward_from_patches(
                    fine_patches, coarse_patches, bot_feats, return_saliency=False
                )
            h_head = h_head.float()
        else:
            _, h_head, _ = self.policy.forward_from_patches(
                fine_patches.float(), coarse_patches.float(), bot_feats.float(), return_saliency=False
            )
        ptr_mv_logits = self.policy.head_ptr_mv(h_head).clamp(-8.0, 8.0)   # [M, 5]
        ptr_em_logits = self.policy.head_ptr_em(h_head).clamp(-8.0, 8.0)   # [M, 13]
        raw_bi = self.policy.head_bipolar(h_head)                          # [M, 5]: res_mv(2), res_em(2), thermal(1)
        raw_uni = self.policy.head_unipolar(h_head)                        # [M, 5]: w, o, sm, foc, imp
        raw_cont = torch.cat([raw_bi, raw_uni], dim=-1)                    # [M, 10]
        return ptr_mv_logits, ptr_em_logits, raw_cont, h_head

    def sample_from_patches(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        deterministic: bool = False,
        soft_mix: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          ptr_mv_idx [A, N] long
          ptr_em_idx [A, N] long
          z_cont     [A, N, 10] float32 (sampled raw bipolar + unipolar)
          act_exec   [A, N, 10] float32 (bounded physical actuators executed in arena)
          lp         [A, N, 5] float32 ([lp_ptr_mv, lp_ptr_em, lp_res_mv, lp_res_em, lp_phys])
          values     [A, N, 2] float32
        """
        a, n, _ = bot_feats.shape
        p = self.policy.patch_size
        M = a * n
        bf_flat = bot_feats.reshape(M, BOT_FEAT_DIM).float()
        ptr_mv_l, ptr_em_l, raw_cont, h_head = self._heads_from_patches(
            fine_patches.reshape(M, -1, p, p),
            coarse_patches.reshape(M, -1, p, p),
            bf_flat,
            use_bf16=False,
        )
        values = self.value_head(h_head).reshape(a, n, 2)

        tok_xy = bf_flat[:, 24:].reshape(M, 12, 12)[:, :, 0:2]  # [M, 12, 2]
        obj_xy = tok_xy[:, 8:12, :]                              # [M, 4, 2]

        if deterministic:
            w_mv = F.softmax(ptr_mv_l, dim=-1)
            w_em = F.softmax(ptr_em_l, dim=-1)
            mv_raw = (w_mv[:, :4].unsqueeze(-1) * (obj_xy * 4.5)).sum(dim=1) + w_mv[:, 4:5].clamp(min=0.18) * raw_cont[:, 0:2]
            em_raw = (w_em[:, :12].unsqueeze(-1) * (tok_xy * 2.22)).sum(dim=1) + w_em[:, 12:].clamp(min=0.18) * raw_cont[:, 2:4]
            raw_act = torch.cat([mv_raw, em_raw, raw_cont[:, 4:]], dim=-1)
            return (
                ptr_mv_l.argmax(dim=-1).reshape(a, n),
                ptr_em_l.argmax(dim=-1).reshape(a, n),
                raw_cont.reshape(a, n, ACTUATOR_DIM),
                raw_to_actuators(raw_act).reshape(a, n, ACTUATOR_DIM),
                torch.zeros((a, n, 5), device=raw_cont.device, dtype=torch.float32),
                values,
            )

        dist_pmv = torch.distributions.Categorical(logits=ptr_mv_l)
        dist_pem = torch.distributions.Categorical(logits=ptr_em_l)
        pmv_idx = dist_pmv.sample()  # [M]
        pem_idx = dist_pem.sample()  # [M]

        std = self.get_std().view(1, ACTUATOR_DIM)
        dist_c = torch.distributions.Normal(raw_cont, std)
        z_cont = dist_c.sample()  # [M, 10]

        lp_c = dist_c.log_prob(z_cont)
        lp = torch.stack(
            [
                dist_pmv.log_prob(pmv_idx),
                dist_pem.log_prob(pem_idx),
                lp_c[:, 0:2].sum(dim=-1),
                lp_c[:, 2:4].sum(dim=-1),
                lp_c[:, 4:10].mean(dim=-1),
            ],
            dim=-1,
        ).reshape(a, n, 5)

        # Compose executed physical vector from sampled pointer (optionally blended with soft pointer late in training)
        oh_mv = F.one_hot(pmv_idx, num_classes=5).float()
        oh_em = F.one_hot(pem_idx, num_classes=13).float()
        if soft_mix > 0.0:
            w_mv = (1.0 - soft_mix) * oh_mv + soft_mix * F.softmax(ptr_mv_l, dim=-1)
            w_em = (1.0 - soft_mix) * oh_em + soft_mix * F.softmax(ptr_em_l, dim=-1)
        else:
            w_mv, w_em = oh_mv, oh_em

        mv_raw = (w_mv[:, :4].unsqueeze(-1) * (obj_xy * 4.5)).sum(dim=1) + w_mv[:, 4:5].clamp(min=0.18) * z_cont[:, 0:2]
        em_raw = (w_em[:, :12].unsqueeze(-1) * (tok_xy * 2.22)).sum(dim=1) + w_em[:, 12:].clamp(min=0.18) * z_cont[:, 2:4]
        raw_act = torch.cat([mv_raw, em_raw, z_cont[:, 4:]], dim=-1)
        act_exec = raw_to_actuators(raw_act).reshape(a, n, ACTUATOR_DIM)
        return pmv_idx.reshape(a, n), pem_idx.reshape(a, n), z_cont.reshape(a, n, ACTUATOR_DIM), act_exec, lp, values

    def evaluate_flat_minibatch(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        pmv_idx: torch.Tensor,
        pem_idx: torch.Tensor,
        z_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr_mv_l, ptr_em_l, raw_cont, h_head = self._heads_from_patches(
            fine_patches, coarse_patches, bot_feats, use_bf16=True
        )
        values = self.value_head(h_head)

        dist_pmv = torch.distributions.Categorical(logits=ptr_mv_l)
        dist_pem = torch.distributions.Categorical(logits=ptr_em_l)
        std = self.get_std().view(1, ACTUATOR_DIM)
        dist_c = torch.distributions.Normal(raw_cont, std)
        lp_c = dist_c.log_prob(z_cont)

        lp = torch.stack(
            [
                dist_pmv.log_prob(pmv_idx),
                dist_pem.log_prob(pem_idx),
                lp_c[:, 0:2].sum(dim=-1),
                lp_c[:, 2:4].sum(dim=-1),
                lp_c[:, 4:10].mean(dim=-1),
            ],
            dim=-1,
        )
        # Balanced entropy across movement pointer, emission pointer, and continuous actuators
        ent = 0.35 * dist_pmv.entropy() + 0.25 * dist_pem.entropy() + 0.40 * dist_c.entropy().mean(dim=-1)
        return lp, ent, values, raw_cont, ptr_em_l


def compute_observable_emit_aux_loss(
    ptr_em_logits: torch.Tensor,  # [B, 13]
    bot_feats: torch.Tensor,      # [B, 168] (strictly the bot's own smoke-obscured observation!)
) -> torch.Tensor:
    """
    Self-supervised perceptual target-attention auxiliary loss (ZERO spell rules, ZERO imitation teacher!).
    Strictly uses the bot's own smoke-blurred `[168]` observation tokens `0..7`: when an alive enemy
    is observable within emitter range (< 38 cells) and the bot is neither wading in deep unfrozen
    water (free to cryo-bridge ahead) nor low-HP/carrying (free to smokescreen self), encourages
    `head_ptr_em` to attend sharply to the nearest observable enemy token slot rather than averaging
    across teammates & empty air!
    """
    toks = bot_feats[:, 24:].reshape(-1, 12, 12)
    nn_en = toks[:, :8, 3] > 0.5
    nn_valid = toks[:, :8, 11] > 0.5
    nn_d = toks[:, :8, 7]  # observed distance / 100.0 (already subject to physical smoke+steam blur!)
    en_in_range = nn_en & nn_valid & (nn_d < 0.38)
    has_en = en_in_range.any(dim=-1)

    in_water_obs = (bot_feats[:, 15] > 0.42) & (bot_feats[:, 16] < 0.35)
    vuln_obs = (bot_feats[:, 5] < 0.52) | (bot_feats[:, 12] > 0.5)
    mask = has_en & (~in_water_obs) & (~vuln_obs)
    if not mask.any():
        return ptr_em_logits.new_tensor(0.0)

    k_star = nn_d.masked_fill(~en_in_range, 1e6).argmin(dim=-1)
    return F.cross_entropy(ptr_em_logits[mask], k_star[mask])

    def sample_soft_from_patches(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Stage 2 Exact Soft-Pointer Logit-Gaussian PPO sampling:
        Explores directly in `head_ptr_mv` [5] and `head_ptr_em` [13] logit space and executes
        EXACT `softmax(z_ptr)` matching `ActuatorPolicyNet.forward()` with ZERO train/eval gap!
        """
        a, n, _ = bot_feats.shape
        p = self.policy.patch_size
        M = a * n
        bf_flat = bot_feats.reshape(M, BOT_FEAT_DIM).float()
        ptr_mv_l, ptr_em_l, raw_cont, h_head = self._heads_from_patches(
            fine_patches.reshape(M, -1, p, p),
            coarse_patches.reshape(M, -1, p, p),
            bf_flat,
            use_bf16=False,
        )
        values = self.value_head(h_head).reshape(a, n, 2)
        tok_xy = bf_flat[:, 24:].reshape(M, 12, 12)[:, :, 0:2]
        obj_xy = tok_xy[:, 8:12, :]

        if deterministic:
            w_mv = F.softmax(ptr_mv_l, dim=-1)
            w_em = F.softmax(ptr_em_l, dim=-1)
            mv_raw = (w_mv[:, :4].unsqueeze(-1) * (obj_xy * 4.5)).sum(dim=1) + w_mv[:, 4:5].clamp(min=0.18) * raw_cont[:, 0:2]
            em_raw = (w_em[:, :12].unsqueeze(-1) * (tok_xy * 2.22)).sum(dim=1) + w_em[:, 12:].clamp(min=0.18) * raw_cont[:, 2:4]
            raw_act = torch.cat([mv_raw, em_raw, raw_cont[:, 4:]], dim=-1)
            return (
                ptr_mv_l.reshape(a, n, 5),
                ptr_em_l.reshape(a, n, 13),
                raw_cont.reshape(a, n, ACTUATOR_DIM),
                raw_to_actuators(raw_act).reshape(a, n, ACTUATOR_DIM),
                torch.zeros((a, n, 5), device=raw_cont.device, dtype=torch.float32),
                values,
            )

        p_std = self.get_ptr_std()
        dist_pmv = torch.distributions.Normal(ptr_mv_l, p_std[0])
        dist_pem = torch.distributions.Normal(ptr_em_l, p_std[1])
        z_pmv = dist_pmv.sample()  # [M, 5]
        z_pem = dist_pem.sample()  # [M, 13]

        std = self.get_std().view(1, ACTUATOR_DIM)
        dist_c = torch.distributions.Normal(raw_cont, std)
        z_cont = dist_c.sample()  # [M, 10]

        lp_c = dist_c.log_prob(z_cont)
        lp = torch.stack(
            [
                dist_pmv.log_prob(z_pmv).mean(dim=-1),
                dist_pem.log_prob(z_pem).mean(dim=-1),
                lp_c[:, 0:2].sum(dim=-1),
                lp_c[:, 2:4].sum(dim=-1),
                lp_c[:, 4:10].mean(dim=-1),
            ],
            dim=-1,
        ).reshape(a, n, 5)

        w_mv = F.softmax(z_pmv, dim=-1)
        w_em = F.softmax(z_pem, dim=-1)
        mv_raw = (w_mv[:, :4].unsqueeze(-1) * (obj_xy * 4.5)).sum(dim=1) + w_mv[:, 4:5].clamp(min=0.18) * z_cont[:, 0:2]
        em_raw = (w_em[:, :12].unsqueeze(-1) * (tok_xy * 2.22)).sum(dim=1) + w_em[:, 12:].clamp(min=0.18) * z_cont[:, 2:4]
        raw_act = torch.cat([mv_raw, em_raw, z_cont[:, 4:]], dim=-1)
        act_exec = raw_to_actuators(raw_act).reshape(a, n, ACTUATOR_DIM)
        return (
            z_pmv.reshape(a, n, 5),
            z_pem.reshape(a, n, 13),
            z_cont.reshape(a, n, ACTUATOR_DIM),
            act_exec,
            lp,
            values,
        )

    def evaluate_flat_minibatch_soft(
        self,
        fine_patches: torch.Tensor,
        coarse_patches: torch.Tensor,
        bot_feats: torch.Tensor,
        z_pmv: torch.Tensor,
        z_pem: torch.Tensor,
        z_cont: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ptr_mv_l, ptr_em_l, raw_cont, h_head = self._heads_from_patches(
            fine_patches, coarse_patches, bot_feats, use_bf16=True
        )
        values = self.value_head(h_head)
        p_std = self.get_ptr_std()
        dist_pmv = torch.distributions.Normal(ptr_mv_l, p_std[0])
        dist_pem = torch.distributions.Normal(ptr_em_l, p_std[1])
        std = self.get_std().view(1, ACTUATOR_DIM)
        dist_c = torch.distributions.Normal(raw_cont, std)
        lp_c = dist_c.log_prob(z_cont)

        lp = torch.stack(
            [
                dist_pmv.log_prob(z_pmv).mean(dim=-1),
                dist_pem.log_prob(z_pem).mean(dim=-1),
                lp_c[:, 0:2].sum(dim=-1),
                lp_c[:, 2:4].sum(dim=-1),
                lp_c[:, 4:10].mean(dim=-1),
            ],
            dim=-1,
        )
        ent = 0.20 * dist_pmv.entropy().mean(dim=-1) + 0.20 * dist_pem.entropy().mean(dim=-1) + 0.60 * dist_c.entropy().mean(dim=-1)
        return lp, ent, values, raw_cont, ptr_mv_l, ptr_em_l


# ---------------------------------------------------------------------------
# Fast Side-Balanced League Validation (vs Legacy, vs Iter5A, vs Thermal-Only)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def quick_eval_league(
    policy: ActuatorPolicyNet,
    v1_policy: ActuatorPolicyNet,
    val_arena: PhysicsActuatorArena,
    seed: int = 82000,
) -> dict[str, Any]:
    """
    32-arena side-balanced deterministic validation on unmodified PhysicsActuatorArena:
      - Arenas  0..15 (16 matches): v2 vs Legacy Named-Spell Scripted (8 W + 8 E)
      - Arenas 16..23 ( 8 matches): v2 vs Iter 5A v1 snapshot (4 W + 4 E)
      - Arenas 24..31 ( 8 matches): v2 vs Thermal-Only Ablation (4 W + 4 E)
    """
    policy.eval()
    v1_policy.eval()
    A = val_arena.A
    assert A == 32, f"Expected 32 val arenas, got {A}"
    f = val_arena.reset(seed=seed)

    rl_mask = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    rl_mask[0:8, :6] = True
    rl_mask[8:16, 6:] = True
    rl_mask[16:20, :6] = True
    rl_mask[20:24, 6:] = True
    rl_mask[24:28, :6] = True
    rl_mask[28:32, 6:] = True

    leg_mask = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    leg_mask[0:8, 6:] = True
    leg_mask[8:16, :6] = True

    v1_mask = torch.zeros((A, NUM_BOTS), dtype=torch.bool, device=val_arena.device)
    v1_mask[16:20, 6:] = True
    v1_mask[20:24, :6] = True

    for _ in range(val_arena.max_ticks):
        act_rl = policy_deterministic_act(policy, val_arena.z_world, val_arena.z_eco, f, thermal_only=False)
        act_v1 = policy_deterministic_act(v1_policy, val_arena.z_world, val_arena.z_eco, f, thermal_only=False)
        act_th = act_rl.clone()
        act_th[:, :, ACT_WATER : ACT_SMOKE + 1] = 0.0
        act_leg = legacy_scripted_actuator_baseline(val_arena)

        act = torch.where(
            rl_mask.unsqueeze(-1),
            act_rl,
            torch.where(leg_mask.unsqueeze(-1), act_leg, torch.where(v1_mask.unsqueeze(-1), act_v1, act_th)),
        )
        f, _, _, _ = val_arena.step(act)

    sw, se = val_arena.team_score[:, 0], val_arena.team_score[:, 1]

    def _wr(i0: int, i1: int, i2: int) -> tuple[float, float, float]:
        w0 = (sw[i0:i1] > se[i0:i1]).float() + 0.5 * (sw[i0:i1] == se[i0:i1]).float()
        w1 = (se[i1:i2] > sw[i1:i2]).float() + 0.5 * (se[i1:i2] == sw[i1:i2]).float()
        wr = float(torch.cat([w0, w1]).mean().item()) * 100.0
        sa = float(torch.cat([sw[i0:i1], se[i1:i2]]).mean().item())
        sb = float(torch.cat([se[i0:i1], sw[i1:i2]]).mean().item())
        return wr, sa, sb

    wr_leg, sc_rl_leg, sc_leg = _wr(0, 8, 16)
    wr_v1, sc_rl_v1, sc_v1 = _wr(16, 20, 24)
    wr_th, sc_rl_th, sc_th = _wr(24, 28, 32)

    side_rl = torch.zeros((A,), dtype=torch.long, device=val_arena.device)
    side_rl[8:16] = 1
    side_rl[20:24] = 1
    side_rl[28:32] = 1
    em = extract_team_emergence(val_arena, side_rl)

    kills_rl = float(torch.cat([val_arena.team_kills[0:8, 0], val_arena.team_kills[8:16, 1]]).mean().item())
    shrines_rl = float(torch.cat([val_arena.team_shrine_caps[0:8, 0], val_arena.team_shrine_caps[8:16, 1]]).mean().item())
    relics_rl = float(torch.cat([val_arena.team_relic_deliveries[0:8, 0], val_arena.team_relic_deliveries[8:16, 1]]).mean().item())
    press_rl = float(torch.cat([val_arena.team_pressure_dmg_dealt[0:8, 0], val_arena.team_pressure_dmg_dealt[8:16, 1]]).mean().item())

    return {
        "wr_legacy_pct": round(wr_leg, 2),
        "wr_v1_pct": round(wr_v1, 2),
        "wr_thermal_pct": round(wr_th, 2),
        "score_rl_vs_legacy": round(sc_rl_leg, 2),
        "score_legacy": round(sc_leg, 2),
        "score_rl_vs_v1": round(sc_rl_v1, 2),
        "score_v1": round(sc_v1, 2),
        "score_rl_vs_thermal": round(sc_rl_th, 2),
        "score_thermal": round(sc_th, 2),
        "kills_vs_legacy": round(kills_rl, 2),
        "shrines_vs_legacy": round(shrines_rl, 2),
        "relics_vs_legacy": round(relics_rl, 2),
        "press_dmg_vs_legacy": round(press_rl, 2),
        **{k: round(v, 3) for k, v in em.items()},
    }


# ---------------------------------------------------------------------------
# League Self-Play PPO Training Loop (64 Arenas = 24 Self/Lag + 24 Legacy + 16 Therm/Iter5A)
# ---------------------------------------------------------------------------


def train_league_self_play_ppo(
    ac: RLActuatorActorCriticV2,
    v1_policy: ActuatorPolicyNet,
    arena: LeaguePhysicsActuatorArena,
    val_arena: PhysicsActuatorArena,
    num_updates: int = 240,
    rollout_steps: int = 30,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    lr: float = 4.5e-4,
    kl_early_stop: float = 0.25,
    eval_every: int = 12,
    seed_offset: int = 2000,
) -> tuple[dict, dict, list[dict]]:
    A, N = arena.A, arena.N
    assert A == 64, f"League opponent mix is calibrated for 64 arenas, got {A}"
    P = ac.policy.patch_size
    dev = arena.device

    optimizer = torch.optim.AdamW(ac.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)

    # Lagged self-play pool (5 rolling checkpoints)
    pool_size = 5
    lag_pool = [copy.deepcopy(ac.policy).eval() for _ in range(pool_size)]
    for m in lag_pool:
        for p in m.parameters():
            p.requires_grad_(False)
    pool_ptr = 0

    # Exact 64-Arena League Opponent Mix (Side-Balanced West/East):
    #   [ 0..11] (12): Pure Mirror Self-Play (all 12 bots RL)
    #   [12..23] (12): Lagged Self-Play Pool (6 RL=W, 6 RL=E)       -> 24 Self/Lag arenas
    #   [24..47] (24): Legacy Named-Spell Scripted (12 RL=W, 12 E)  -> 24 Legacy arenas
    #   [48..55] ( 8): Thermal-Only Ablation (4 RL=W, 4 RL=E)       ->  8 Thermal-Only
    #   [56..63] ( 8): Frozen Iter 5A Snapshot (4 RL=W, 4 RL=E)     ->  8 Iter 5A
    rl_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    rl_mask[0:12, :] = True
    rl_mask[12:18, :6] = True
    rl_mask[18:24, 6:] = True
    rl_mask[24:36, :6] = True
    rl_mask[36:48, 6:] = True
    rl_mask[48:52, :6] = True
    rl_mask[52:56, 6:] = True
    rl_mask[56:60, :6] = True
    rl_mask[60:64, 6:] = True

    opp_lag_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_lag_mask[12:18, 6:] = True
    opp_lag_mask[18:24, :6] = True

    opp_leg_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_leg_mask[24:36, 6:] = True
    opp_leg_mask[36:48, :6] = True

    opp_therm_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_therm_mask[48:52, 6:] = True
    opp_therm_mask[52:56, :6] = True

    opp_v1_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_v1_mask[56:60, 6:] = True
    opp_v1_mask[60:64, :6] = True

    rl_mask_t = rl_mask.unsqueeze(0).expand(rollout_steps, A, N)

    # Pre-allocated GPU rollout buffers
    p_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    buf_fp = torch.empty((rollout_steps, A, N, 80, P, P), dtype=p_dtype, device=dev)
    buf_cp = torch.empty((rollout_steps, A, N, 48, P, P), dtype=p_dtype, device=dev)
    buf_feats = torch.empty((rollout_steps, A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)
    buf_pmv = torch.empty((rollout_steps, A, N), dtype=torch.long, device=dev)
    buf_pem = torch.empty((rollout_steps, A, N), dtype=torch.long, device=dev)
    buf_z = torch.empty((rollout_steps, A, N, ACTUATOR_DIM), dtype=torch.float32, device=dev)
    buf_lp = torch.empty((rollout_steps, A, N, 5), dtype=torch.float32, device=dev)
    buf_val = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_rew = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_done = torch.empty((rollout_steps,), dtype=torch.float32, device=dev)

    t0 = time.perf_counter()
    feats = arena.reset(seed=seed_offset, close_spawn_frac=0.20)
    total_bot_steps = 0
    kl_stops = 0

    # Initial warm-start (upd 0) validation
    ev0 = quick_eval_league(ac.policy, v1_policy, val_arena, seed=82000)
    ev0["update"] = 0
    ev0["elapsed_s"] = 0.0
    ev0["mean_std"] = round(float(ac.get_std().mean().item()), 3)
    history: list[dict] = [ev0]

    def _ckpt_metric(ev: dict) -> float:
        # Heavily prioritize beating Legacy (>70% WR target!) + score margin + beating Iter5A/Thermal + emergence
        em_active = (
            min(ev["oil_ignition_detonations"], 20.0)
            + min(ev["flash_boil_steam_bursts"], 20.0)
            + min(ev["cryo_crossings"], 15.0)
            + min(ev["smokescreen_covers"], 10.0)
        )
        return (
            20.0 * ev["wr_legacy_pct"]
            + 3.0 * ev["wr_v1_pct"]
            + 2.5 * ev["wr_thermal_pct"]
            + 1.35 * (ev["score_rl_vs_legacy"] - ev["score_legacy"])
            + 10.0 * ev["kills_vs_legacy"]
            + 18.0 * ev["relics_vs_legacy"]
            + 8.0 * ev["shrines_vs_legacy"]
            + 1.5 * em_active
        )

    best_metric = _ckpt_metric(ev0)
    best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
    best_ev = ev0

    print(
        f"[League PPO v2] init (upd 0/{num_updates}) | 64x12={A*N} bots (24 Self/Lag + 24 Legacy + 16 Therm/v1) | "
        f"vsLegacy={ev0['wr_legacy_pct']:5.1f}% ({ev0['score_rl_vs_legacy']:.0f}v{ev0['score_legacy']:.0f}) | "
        f"vsV1={ev0['wr_v1_pct']:5.1f}% | vsTherm={ev0['wr_thermal_pct']:5.1f}% | "
        f"shr={ev0['shrines_vs_legacy']:.1f} rel={ev0['relics_vs_legacy']:.2f}",
        flush=True,
    )

    for upd in range(1, num_updates + 1):
        # Update lagged pool every 8 updates
        if upd > 1 and (upd % 8 == 0):
            lag_pool[pool_ptr].load_state_dict(ac.policy.state_dict())
            pool_ptr = (pool_ptr + 1) % pool_size
        opp_lag_pol = lag_pool[(upd // 3) % pool_size]

        ac.eval()
        # Gently anneal soft_mix from 0.0 -> 0.30 in the second half (keeps Categorical PPO strictly on-policy!)
        prog = upd / float(num_updates)
        soft_mix = 0.0 if prog < 0.45 else 0.30 * ((prog - 0.45) / 0.55)
        with torch.inference_mode():
            for t in range(rollout_steps):
                fp, cp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
                buf_fp[t] = fp.to(p_dtype)
                buf_cp[t] = cp.to(p_dtype)
                buf_feats[t] = feats

                pmv_rl, pem_rl, z_rl, act_rl, lp_rl, val_rl = ac.sample_from_patches(
                    fp, cp, feats, deterministic=False, soft_mix=soft_mix
                )
                act = act_rl

                # Lagged Self-Play opponent (arenas 12..23)
                opp_raw, _, _ = opp_lag_pol.forward_from_patches(
                    fp[12:24].reshape(12 * N, 80, P, P).float(),
                    cp[12:24].reshape(12 * N, 48, P, P).float(),
                    feats[12:24].reshape(12 * N, BOT_FEAT_DIM).float(),
                    return_saliency=False,
                )
                act_lag = raw_to_actuators(opp_raw).reshape(12, N, ACTUATOR_DIM)
                act[12:24] = torch.where(opp_lag_mask[12:24].unsqueeze(-1), act_lag, act[12:24])

                # Legacy Named-Spell Scripted opponent (arenas 24..47)
                act_leg = legacy_scripted_actuator_baseline(arena)
                act[24:48] = torch.where(opp_leg_mask[24:48].unsqueeze(-1), act_leg[24:48], act[24:48])

                # Thermal-Only opponent (arenas 48..55, from lagged/current deterministic policy)
                th_raw, _, _ = opp_lag_pol.forward_from_patches(
                    fp[48:56].reshape(8 * N, 80, P, P).float(),
                    cp[48:56].reshape(8 * N, 48, P, P).float(),
                    feats[48:56].reshape(8 * N, BOT_FEAT_DIM).float(),
                    return_saliency=False,
                )
                act_th = raw_to_actuators(th_raw).reshape(8, N, ACTUATOR_DIM)
                act_th[:, :, ACT_WATER : ACT_SMOKE + 1] = 0.0
                act[48:56] = torch.where(opp_therm_mask[48:56].unsqueeze(-1), act_th, act[48:56])

                # Frozen Iter 5A snapshot opponent (arenas 56..63)
                v1_raw, _, _ = v1_policy.forward_from_patches(
                    fp[56:64].reshape(8 * N, 80, P, P).float(),
                    cp[56:64].reshape(8 * N, 48, P, P).float(),
                    feats[56:64].reshape(8 * N, BOT_FEAT_DIM).float(),
                    return_saliency=False,
                )
                act_v1 = raw_to_actuators(v1_raw).reshape(8, N, ACTUATOR_DIM)
                act[56:64] = torch.where(opp_v1_mask[56:64].unsqueeze(-1), act_v1, act[56:64])

                buf_pmv[t] = pmv_rl
                buf_pem[t] = pem_rl
                buf_z[t] = z_rl
                buf_lp[t] = lp_rl
                buf_val[t] = val_rl

                next_feats, rews, done = arena.step_league(act)
                buf_rew[t] = rews
                buf_done[t] = 1.0 if done else 0.0
                total_bot_steps += A * N
                if done:
                    # Keep 80% full-map spawns (matching eval!) and 20% closer river spawns
                    feats = arena.reset(seed=seed_offset + upd * 251 + t, close_spawn_frac=0.20)
                else:
                    feats = next_feats

            lfp, lcp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
            _, _, _, _, _, last_val = ac.sample_from_patches(lfp, lcp, feats, deterministic=True)

        # Dual-Horizon Float32 GAE (Macro Tac: gamma=0.98, lam=0.95; Immediate Act: gamma=0.82, lam=0.88)
        adv = torch.zeros_like(buf_rew)
        for h_i, (gam, lam) in enumerate(((0.98, 0.95), (0.82, 0.88))):
            gae = torch.zeros((A, N), device=dev, dtype=torch.float32)
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
        flat_pmv = buf_pmv[rl_mask_t]
        flat_pem = buf_pem[rl_mask_t]
        flat_z = buf_z[rl_mask_t]
        flat_old_lp = buf_lp[rl_mask_t]
        flat_adv_pair = adv[rl_mask_t]
        adv_tac = (flat_adv_pair[:, 0] - flat_adv_pair[:, 0].mean()) / (flat_adv_pair[:, 0].std() + 1e-6)
        adv_act = (flat_adv_pair[:, 1] - flat_adv_pair[:, 1].mean()) / (flat_adv_pair[:, 1].std() + 1e-6)
        adv_phys = 0.62 * adv_act + 0.48 * adv_tac
        flat_ret = ret[rl_mask_t]

        M = flat_feats.shape[0]
        ent_coef = 0.015 * (1.0 - 0.85 * prog)

        # Gentle warmup over first 8 updates (value head warms up before policy moves aggressively)
        if upd <= 8:
            lr_now = lr * (0.25 + 0.75 * (upd / 8.0))
        else:
            cprog = (upd - 8) / float(max(num_updates - 8, 1))
            lr_now = lr * (0.14 + 0.86 * 0.5 * (1.0 + math.cos(math.pi * cprog)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        ac.train()
        early_stopped = False
        max_kl = 0.0
        for _ in range(ppo_epochs):
            perm = torch.randperm(M, device=dev)
            ep_kls = []
            for start in range(0, M, minibatch_size):
                idx = perm[start : start + minibatch_size]
                optimizer.zero_grad(set_to_none=True)

                new_lp, ent, vals, raw_cont, ptr_em_l = ac.evaluate_flat_minibatch(
                    flat_fp[idx], flat_cp[idx], flat_feats[idx], flat_pmv[idx], flat_pem[idx], flat_z[idx]
                )
                log_r = (new_lp - flat_old_lp[idx]).clamp(-4.0, 4.0)  # [B, 5]
                log_r_joint = log_r.sum(dim=-1)
                approx_kl = float((0.5 * log_r_joint.pow(2)).mean().item())
                ep_kls.append(approx_kl)
                if approx_kl > max_kl:
                    max_kl = approx_kl
                if approx_kl > kl_early_stop:
                    early_stopped = True
                    kl_stops += 1
                    break

                r_pmv = torch.exp(log_r[:, 0])
                r_pem = torch.exp(log_r[:, 1])
                r_rmv = torch.exp(log_r[:, 2])
                r_rem = torch.exp(log_r[:, 3])
                r_ph = torch.exp(log_r[:, 4])

                a_t = adv_tac[idx]
                a_p = adv_phys[idx]

                pg_pmv = -torch.minimum(r_pmv * a_t, r_pmv.clamp(0.82, 1.18) * a_t).mean()
                pg_pem = -torch.minimum(r_pem * a_p, r_pem.clamp(0.82, 1.18) * a_p).mean()
                pg_rmv = -torch.minimum(r_rmv * a_t, r_rmv.clamp(0.82, 1.18) * a_t).mean()
                pg_rem = -torch.minimum(r_rem * a_p, r_rem.clamp(0.82, 1.18) * a_p).mean()
                pg_ph = -torch.minimum(r_ph * a_p, r_ph.clamp(0.82, 1.18) * a_p).mean()

                pg_scale = 0.0 if upd <= 3 else 1.0
                pg_loss = pg_scale * (1.50 * pg_pmv + 2.00 * pg_pem + 0.65 * pg_rmv + 0.65 * pg_rem + 2.60 * pg_ph)

                v_loss = F.smooth_l1_loss(vals, flat_ret[idx], beta=5.0)
                ent_loss = -pg_scale * ent_coef * ent.mean()
                logit_reg = 1.0e-2 * F.relu(raw_cont[:, 4:].abs() - 2.4).pow(2).mean()
                emit_aux = compute_observable_emit_aux_loss(ptr_em_l, flat_feats[idx])

                loss = pg_loss + 0.5 * v_loss + ent_loss + logit_reg + 0.40 * emit_aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                optimizer.step()

            if early_stopped or (ep_kls and float(np.mean(ep_kls)) > kl_early_stop):
                if not early_stopped:
                    kl_stops += 1
                break

        if (upd % eval_every == 0) or (upd == num_updates):
            elapsed = time.perf_counter() - t0
            ev = quick_eval_league(ac.policy, v1_policy, val_arena, seed=82000)
            ev["update"] = upd
            ev["elapsed_s"] = round(elapsed, 2)
            ev["mean_std"] = round(float(ac.get_std().mean().item()), 3)
            history.append(ev)

            metric = _ckpt_metric(ev)
            tag = ""
            if metric > best_metric:
                best_metric = metric
                best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
                best_ev = ev
                tag = " *BEST*"

            print(
                f"[League PPO v2] upd {upd:3d}/{num_updates} | t={elapsed:5.1f}s | kl={max_kl:.3f} | "
                f"vsLegacy={ev['wr_legacy_pct']:5.1f}% ({ev['score_rl_vs_legacy']:.0f}v{ev['score_legacy']:.0f}) | "
                f"vsV1={ev['wr_v1_pct']:5.1f}% | vsTherm={ev['wr_thermal_pct']:5.1f}% | "
                f"k={ev['kills_vs_legacy']:.1f} shr={ev['shrines_vs_legacy']:.1f} rel={ev['relics_vs_legacy']:.2f} "
                f"pDmg={ev['press_dmg_vs_legacy']:.2f} | "
                f"oil={ev['oil_ignition_detonations']:.1f} boil={ev['flash_boil_steam_bursts']:.1f} "
                f"cryo={ev['cryo_crossings']:.1f} smk={ev['smokescreen_covers']:.1f}{tag}",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    train_summary = {
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "parallel_arenas": A,
        "concurrent_bots": A * N,
        "league_mix": {
            "pure_mirror_self_play_arenas": 12,
            "lagged_self_play_arenas": 12,
            "legacy_scripted_arenas": 24,
            "thermal_only_arenas": 8,
            "iter5a_snapshot_arenas": 8,
        },
        "updates": num_updates,
        "rollout_steps": rollout_steps,
        "total_bot_steps": total_bot_steps,
        "elapsed_s": elapsed,
        "bot_steps_per_sec": round(total_bot_steps / max(elapsed, 1e-3), 1),
        "kl_early_stop_threshold": kl_early_stop,
        "kl_early_stops_triggered": kl_stops,
        "zero_imitation_teacher": True,
        "zero_hardcoded_spell_rules": True,
        "best_val": best_ev,
    }
    return train_summary, best_sd, history


def train_stage2_soft_pointer_ppo(
    ac: RLActuatorActorCriticV2,
    v1_policy: ActuatorPolicyNet,
    arena: LeaguePhysicsActuatorArena,
    val_arena: PhysicsActuatorArena,
    num_updates: int = 300,
    rollout_steps: int = 30,
    ppo_epochs: int = 4,
    minibatch_size: int = 2048,
    lr: float = 2.6e-4,
    kl_early_stop: float = 0.18,
    eval_every: int = 15,
    critic_warmup: int = 18,
    seed_offset: int = 9000,
    stage_label: str = "Stage2-128",
) -> tuple[dict, dict, list[dict]]:
    """
    Stage 2 / Stage 3 League PPO Continuation & High-Res 256x256x4 Polish:
    Supports arbitrary arena count A (64 on 128x128x2, or 24 on 256x256x4!) with
    full critic warmup (`critic_warmup=18`) so warm-started policy never dips,
    keeps `compute_observable_emit_aux_loss` active, and selects the best checkpoint
    strictly on held-out win rate + score margin vs Legacy, Iter5A, and Thermal-Only!
    """
    A, N = arena.A, arena.N
    P = ac.policy.patch_size
    dev = arena.device

    with torch.no_grad():
        ac.log_std.copy_(torch.tensor([-0.68, -0.68, -0.68, -0.68, -0.45, -0.55, -0.45, -0.45, -0.50, -0.45], device=dev))

    optimizer = torch.optim.AdamW(ac.parameters(), lr=lr, eps=1e-5, weight_decay=1e-4)

    pool_size = 5
    lag_pool = [copy.deepcopy(ac.policy).eval() for _ in range(pool_size)]
    for m in lag_pool:
        for p in m.parameters():
            p.requires_grad_(False)
    pool_ptr = 0

    # Flexible side-balanced League Mix for A=64 (12+12+24+8+8) or A=24 (4+4+10+3+3 -> 4+4+10+2+4)
    if A == 64:
        n_mir, n_lag, n_leg, n_th, n_v1 = 12, 12, 24, 8, 8
    else:
        n_mir = max(2, (A // 6) & ~1)
        n_lag = max(2, (A // 6) & ~1)
        n_leg = max(4, (A * 5 // 12) & ~1)
        rem = A - n_mir - n_lag - n_leg
        n_th = max(2, (rem // 2) & ~1)
        n_v1 = A - n_mir - n_lag - n_leg - n_th

    i_lag0 = n_mir
    i_lag1 = i_lag0 + n_lag
    i_leg0 = i_lag1
    i_leg1 = i_leg0 + n_leg
    i_th0 = i_leg1
    i_th1 = i_th0 + n_th
    i_v10 = i_th1
    i_v11 = A

    rl_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_lag_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_leg_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_therm_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)
    opp_v1_mask = torch.zeros((A, N), dtype=torch.bool, device=dev)

    rl_mask[0:n_mir, :] = True

    def _split_half(mask_rl: torch.Tensor, mask_opp: torch.Tensor, s0: int, s1: int) -> None:
        mid = s0 + (s1 - s0) // 2
        mask_rl[s0:mid, :6] = True
        mask_rl[mid:s1, 6:] = True
        mask_opp[s0:mid, 6:] = True
        mask_opp[mid:s1, :6] = True

    _split_half(rl_mask, opp_lag_mask, i_lag0, i_lag1)
    _split_half(rl_mask, opp_leg_mask, i_leg0, i_leg1)
    _split_half(rl_mask, opp_therm_mask, i_th0, i_th1)
    if i_v11 > i_v10:
        _split_half(rl_mask, opp_v1_mask, i_v10, i_v11)

    rl_mask_t = rl_mask.unsqueeze(0).expand(rollout_steps, A, N)

    p_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    buf_fp = torch.empty((rollout_steps, A, N, 80, P, P), dtype=p_dtype, device=dev)
    buf_cp = torch.empty((rollout_steps, A, N, 48, P, P), dtype=p_dtype, device=dev)
    buf_feats = torch.empty((rollout_steps, A, N, BOT_FEAT_DIM), dtype=torch.float32, device=dev)
    buf_pmv = torch.empty((rollout_steps, A, N), dtype=torch.long, device=dev)
    buf_pem = torch.empty((rollout_steps, A, N), dtype=torch.long, device=dev)
    buf_z = torch.empty((rollout_steps, A, N, ACTUATOR_DIM), dtype=torch.float32, device=dev)
    buf_lp = torch.empty((rollout_steps, A, N, 5), dtype=torch.float32, device=dev)
    buf_val = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_rew = torch.empty((rollout_steps, A, N, 2), dtype=torch.float32, device=dev)
    buf_done = torch.empty((rollout_steps,), dtype=torch.float32, device=dev)

    t0 = time.perf_counter()
    feats = arena.reset(seed=seed_offset, close_spawn_frac=0.10)
    total_bot_steps = 0
    kl_stops = 0

    @torch.inference_mode()
    def _run_val() -> dict[str, Any]:
        if val_arena.A == 32:
            return quick_eval_league(ac.policy, v1_policy, val_arena, seed=82000)
        # 256x256x4 12-arena side-balanced val: 8 vs Legacy + 2 vs v1 + 2 vs Thermal
        ac.policy.eval()
        v1_policy.eval()
        Av = val_arena.A
        f_v = val_arena.reset(seed=83000)
        n_l = (Av * 2 // 3) & ~1
        n_v = ((Av - n_l) // 2) & ~1
        rm = torch.zeros((Av, NUM_BOTS), dtype=torch.bool, device=dev)
        lm = torch.zeros((Av, NUM_BOTS), dtype=torch.bool, device=dev)
        vm = torch.zeros((Av, NUM_BOTS), dtype=torch.bool, device=dev)
        rm[0 : n_l // 2, :6] = True
        rm[n_l // 2 : n_l, 6:] = True
        lm[0 : n_l // 2, 6:] = True
        lm[n_l // 2 : n_l, :6] = True
        rm[n_l : n_l + n_v // 2, :6] = True
        rm[n_l + n_v // 2 : n_l + n_v, 6:] = True
        vm[n_l : n_l + n_v // 2, 6:] = True
        vm[n_l + n_v // 2 : n_l + n_v, :6] = True
        rem_mid = n_l + n_v + (Av - n_l - n_v) // 2
        rm[n_l + n_v : rem_mid, :6] = True
        rm[rem_mid:Av, 6:] = True
        for _ in range(val_arena.max_ticks):
            a_rl = policy_deterministic_act(ac.policy, val_arena.z_world, val_arena.z_eco, f_v, False)
            a_v1 = policy_deterministic_act(v1_policy, val_arena.z_world, val_arena.z_eco, f_v, False)
            a_th = a_rl.clone()
            a_th[:, :, ACT_WATER : ACT_SMOKE + 1] = 0.0
            a_lg = legacy_scripted_actuator_baseline(val_arena)
            act_v = torch.where(rm.unsqueeze(-1), a_rl, torch.where(lm.unsqueeze(-1), a_lg, torch.where(vm.unsqueeze(-1), a_v1, a_th)))
            f_v, _, _, _ = val_arena.step(act_v)
        sw, se = val_arena.team_score[:, 0], val_arena.team_score[:, 1]
        def _w(s0: int, s1: int, s2: int) -> tuple[float, float, float]:
            w0 = (sw[s0:s1] > se[s0:s1]).float() + 0.5 * (sw[s0:s1] == se[s0:s1]).float()
            w1 = (se[s1:s2] > sw[s1:s2]).float() + 0.5 * (se[s1:s2] == sw[s1:s2]).float()
            return float(torch.cat([w0, w1]).mean().item()) * 100.0, float(torch.cat([sw[s0:s1], se[s1:s2]]).mean().item()), float(torch.cat([se[s0:s1], sw[s1:s2]]).mean().item())
        wr_lg, sc_rl_lg, sc_lg = _w(0, n_l // 2, n_l)
        wr_v1, sc_rl_v1, sc_v1 = _w(n_l, n_l + n_v // 2, n_l + n_v)
        wr_th, sc_rl_th, sc_th = _w(n_l + n_v, rem_mid, Av)
        side_rl = torch.zeros((Av,), dtype=torch.long, device=dev)
        side_rl[n_l // 2 : n_l] = 1
        side_rl[n_l + n_v // 2 : n_l + n_v] = 1
        side_rl[rem_mid:Av] = 1
        em = extract_team_emergence(val_arena, side_rl)
        return {
            "wr_legacy_pct": round(wr_lg, 2),
            "wr_v1_pct": round(wr_v1, 2),
            "wr_thermal_pct": round(wr_th, 2),
            "score_rl_vs_legacy": round(sc_rl_lg, 2),
            "score_legacy": round(sc_lg, 2),
            "score_rl_vs_v1": round(sc_rl_v1, 2),
            "score_v1": round(sc_v1, 2),
            "score_rl_vs_thermal": round(sc_rl_th, 2),
            "score_thermal": round(sc_th, 2),
            "kills_vs_legacy": round(float(torch.cat([val_arena.team_kills[0 : n_l // 2, 0], val_arena.team_kills[n_l // 2 : n_l, 1]]).mean().item()), 2),
            "shrines_vs_legacy": round(float(torch.cat([val_arena.team_shrine_caps[0 : n_l // 2, 0], val_arena.team_shrine_caps[n_l // 2 : n_l, 1]]).mean().item()), 2),
            "relics_vs_legacy": round(float(torch.cat([val_arena.team_relic_deliveries[0 : n_l // 2, 0], val_arena.team_relic_deliveries[n_l // 2 : n_l, 1]]).mean().item()), 2),
            "press_dmg_vs_legacy": round(float(torch.cat([val_arena.team_pressure_dmg_dealt[0 : n_l // 2, 0], val_arena.team_pressure_dmg_dealt[n_l // 2 : n_l, 1]]).mean().item()), 2),
            **{k: round(v, 3) for k, v in em.items()},
        }

    ev0 = _run_val()
    ev0["update"] = 0
    ev0["stage"] = stage_label
    ev0["elapsed_s"] = 0.0
    history: list[dict] = [ev0]

    def _s2_metric(ev: dict) -> float:
        em_active = (
            min(ev["oil_ignition_detonations"], 25.0)
            + min(ev["flash_boil_steam_bursts"], 25.0)
            + min(ev["cryo_crossings"], 12.0)
            + min(ev["smokescreen_covers"], 6.0)
        )
        return (
            25.0 * ev["wr_legacy_pct"]
            + 3.0 * ev["wr_v1_pct"]
            + 3.0 * ev["wr_thermal_pct"]
            + 1.60 * (ev["score_rl_vs_legacy"] - ev["score_legacy"])
            + 10.0 * ev["kills_vs_legacy"]
            + 16.0 * ev["relics_vs_legacy"]
            + 10.0 * ev["shrines_vs_legacy"]
            + 1.2 * em_active
        )

    best_metric = _s2_metric(ev0)
    best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
    best_ev = ev0

    print(
        f"[{stage_label}] init (upd 0/{num_updates}, {A}x12={A*N}b, {arena.H}x{arena.W}x{arena.L}) | "
        f"vsLegacy={ev0['wr_legacy_pct']:5.1f}% ({ev0['score_rl_vs_legacy']:.0f}v{ev0['score_legacy']:.0f}) | "
        f"vsV1={ev0['wr_v1_pct']:5.1f}% | vsTherm={ev0['wr_thermal_pct']:5.1f}% | "
        f"k={ev0['kills_vs_legacy']:.1f} shr={ev0['shrines_vs_legacy']:.1f} rel={ev0['relics_vs_legacy']:.2f}",
        flush=True,
    )

    for upd in range(1, num_updates + 1):
        if upd > critic_warmup and (upd % 10 == 0):
            lag_pool[pool_ptr].load_state_dict(ac.policy.state_dict())
            pool_ptr = (pool_ptr + 1) % pool_size
        opp_lag_pol = lag_pool[(upd // 3) % pool_size]

        prog = max(0.0, (upd - critic_warmup) / float(max(num_updates - critic_warmup, 1)))
        soft_mix = 0.20 + 0.15 * prog

        ac.eval()
        with torch.inference_mode():
            for t in range(rollout_steps):
                fp, cp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
                buf_fp[t] = fp.to(p_dtype)
                buf_cp[t] = cp.to(p_dtype)
                buf_feats[t] = feats

                pmv_rl, pem_rl, z_rl, act_rl, lp_rl, val_rl = ac.sample_from_patches(
                    fp, cp, feats, deterministic=False, soft_mix=soft_mix
                )
                act = act_rl

                if i_lag1 > i_lag0:
                    nl = i_lag1 - i_lag0
                    opp_raw, _, _ = opp_lag_pol.forward_from_patches(
                        fp[i_lag0:i_lag1].reshape(nl * N, 80, P, P).float(),
                        cp[i_lag0:i_lag1].reshape(nl * N, 48, P, P).float(),
                        feats[i_lag0:i_lag1].reshape(nl * N, BOT_FEAT_DIM).float(),
                        return_saliency=False,
                    )
                    act_lag = raw_to_actuators(opp_raw).reshape(nl, N, ACTUATOR_DIM)
                    act[i_lag0:i_lag1] = torch.where(opp_lag_mask[i_lag0:i_lag1].unsqueeze(-1), act_lag, act[i_lag0:i_lag1])

                act_leg = legacy_scripted_actuator_baseline(arena)
                act[i_leg0:i_leg1] = torch.where(opp_leg_mask[i_leg0:i_leg1].unsqueeze(-1), act_leg[i_leg0:i_leg1], act[i_leg0:i_leg1])

                if i_th1 > i_th0:
                    nt = i_th1 - i_th0
                    th_raw, _, _ = opp_lag_pol.forward_from_patches(
                        fp[i_th0:i_th1].reshape(nt * N, 80, P, P).float(),
                        cp[i_th0:i_th1].reshape(nt * N, 48, P, P).float(),
                        feats[i_th0:i_th1].reshape(nt * N, BOT_FEAT_DIM).float(),
                        return_saliency=False,
                    )
                    act_th = raw_to_actuators(th_raw).reshape(nt, N, ACTUATOR_DIM)
                    act_th[:, :, ACT_WATER : ACT_SMOKE + 1] = 0.0
                    act[i_th0:i_th1] = torch.where(opp_therm_mask[i_th0:i_th1].unsqueeze(-1), act_th, act[i_th0:i_th1])

                if i_v11 > i_v10:
                    nv = i_v11 - i_v10
                    v1_raw, _, _ = v1_policy.forward_from_patches(
                        fp[i_v10:i_v11].reshape(nv * N, 80, P, P).float(),
                        cp[i_v10:i_v11].reshape(nv * N, 48, P, P).float(),
                        feats[i_v10:i_v11].reshape(nv * N, BOT_FEAT_DIM).float(),
                        return_saliency=False,
                    )
                    act_v1 = raw_to_actuators(v1_raw).reshape(nv, N, ACTUATOR_DIM)
                    act[i_v10:i_v11] = torch.where(opp_v1_mask[i_v10:i_v11].unsqueeze(-1), act_v1, act[i_v10:i_v11])

                buf_pmv[t] = pmv_rl
                buf_pem[t] = pem_rl
                buf_z[t] = z_rl
                buf_lp[t] = lp_rl
                buf_val[t] = val_rl

                next_feats, rews, done = arena.step_league(act)
                buf_rew[t] = rews
                buf_done[t] = 1.0 if done else 0.0
                total_bot_steps += A * N
                feats = arena.reset(seed=seed_offset + upd * 263 + t, close_spawn_frac=0.10) if done else next_feats

            lfp, lcp = sample_batched_dual_patches(ac.policy, arena.z_world, arena.z_eco, feats)
            _, _, _, _, _, last_val = ac.sample_from_patches(lfp, lcp, feats, deterministic=True)

        adv = torch.zeros_like(buf_rew)
        for h_i, (gam, lam) in enumerate(((0.98, 0.95), (0.82, 0.88))):
            gae = torch.zeros((A, N), device=dev, dtype=torch.float32)
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
        flat_pmv = buf_pmv[rl_mask_t]
        flat_pem = buf_pem[rl_mask_t]
        flat_z = buf_z[rl_mask_t]
        flat_old_lp = buf_lp[rl_mask_t]
        flat_adv_pair = adv[rl_mask_t]
        adv_tac = (flat_adv_pair[:, 0] - flat_adv_pair[:, 0].mean()) / (flat_adv_pair[:, 0].std() + 1e-6)
        adv_act = (flat_adv_pair[:, 1] - flat_adv_pair[:, 1].mean()) / (flat_adv_pair[:, 1].std() + 1e-6)
        adv_phys = 0.65 * adv_act + 0.45 * adv_tac
        flat_ret = ret[rl_mask_t]

        M = flat_feats.shape[0]
        ent_coef = 0.006 * (1.0 - 0.75 * prog)

        if upd <= critic_warmup:
            lr_now = lr * 2.0
        else:
            lr_now = lr * (0.15 + 0.85 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        ac.train()
        early_stopped = False
        max_kl = 0.0
        for _ in range(ppo_epochs):
            perm = torch.randperm(M, device=dev)
            ep_kls = []
            for start in range(0, M, minibatch_size):
                idx = perm[start : start + minibatch_size]
                optimizer.zero_grad(set_to_none=True)

                new_lp, ent, vals, raw_cont, ptr_em_l = ac.evaluate_flat_minibatch(
                    flat_fp[idx], flat_cp[idx], flat_feats[idx], flat_pmv[idx], flat_pem[idx], flat_z[idx]
                )
                v_loss = F.smooth_l1_loss(vals, flat_ret[idx], beta=5.0)

                if upd <= critic_warmup:
                    # Strictly train value_head ONLY during critic warmup -> ZERO policy shock!
                    loss = v_loss
                    # Zero policy grads explicitly
                    loss.backward()
                    for p in ac.policy.parameters():
                        p.grad = None
                    ac.log_std.grad = None
                    torch.nn.utils.clip_grad_norm_(ac.value_head.parameters(), 1.0)
                    optimizer.step()
                    continue

                log_r = (new_lp - flat_old_lp[idx]).clamp(-4.0, 4.0)
                log_r_joint = log_r.sum(dim=-1)
                approx_kl = float((0.5 * log_r_joint.pow(2)).mean().item())
                ep_kls.append(approx_kl)
                if approx_kl > max_kl:
                    max_kl = approx_kl
                if approx_kl > kl_early_stop:
                    early_stopped = True
                    kl_stops += 1
                    break

                r_pmv = torch.exp(log_r[:, 0])
                r_pem = torch.exp(log_r[:, 1])
                r_rmv = torch.exp(log_r[:, 2])
                r_rem = torch.exp(log_r[:, 3])
                r_ph = torch.exp(log_r[:, 4])

                a_t = adv_tac[idx]
                a_p = adv_phys[idx]

                pg_pmv = -torch.minimum(r_pmv * a_t, r_pmv.clamp(0.85, 1.15) * a_t).mean()
                pg_pem = -torch.minimum(r_pem * a_p, r_pem.clamp(0.85, 1.15) * a_p).mean()
                pg_rmv = -torch.minimum(r_rmv * a_t, r_rmv.clamp(0.85, 1.15) * a_t).mean()
                pg_rem = -torch.minimum(r_rem * a_p, r_rem.clamp(0.85, 1.15) * a_p).mean()
                pg_ph = -torch.minimum(r_ph * a_p, r_ph.clamp(0.85, 1.15) * a_p).mean()

                pg_loss = 1.60 * pg_pmv + 2.00 * pg_pem + 0.65 * pg_rmv + 0.65 * pg_rem + 2.60 * pg_ph
                ent_loss = -ent_coef * ent.mean()
                logit_reg = 1.0e-2 * F.relu(raw_cont[:, 4:].abs() - 2.4).pow(2).mean()
                emit_aux = compute_observable_emit_aux_loss(ptr_em_l, flat_feats[idx])

                loss = pg_loss + 0.5 * v_loss + ent_loss + logit_reg + 0.40 * emit_aux
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                optimizer.step()

            if early_stopped or (ep_kls and float(np.mean(ep_kls)) > kl_early_stop):
                if not early_stopped:
                    kl_stops += 1
                break

        if (upd > critic_warmup and (upd % eval_every == 0)) or (upd == num_updates):
            elapsed = time.perf_counter() - t0
            ev = _run_val()
            ev["update"] = upd
            ev["stage"] = stage_label
            ev["elapsed_s"] = round(elapsed, 2)
            history.append(ev)

            metric = _s2_metric(ev)
            tag = ""
            if metric > best_metric:
                best_metric = metric
                best_sd = {k: v.detach().clone() for k, v in ac.policy.state_dict().items()}
                best_ev = ev
                tag = " *BEST*"

            print(
                f"[{stage_label}] upd {upd:3d}/{num_updates} | t={elapsed:5.1f}s | kl={max_kl:.3f} | "
                f"vsLegacy={ev['wr_legacy_pct']:5.1f}% ({ev['score_rl_vs_legacy']:.0f}v{ev['score_legacy']:.0f}) | "
                f"vsV1={ev['wr_v1_pct']:5.1f}% | vsTherm={ev['wr_thermal_pct']:5.1f}% | "
                f"k={ev['kills_vs_legacy']:.1f} shr={ev['shrines_vs_legacy']:.1f} rel={ev['relics_vs_legacy']:.2f} "
                f"pDmg={ev['press_dmg_vs_legacy']:.2f} | "
                f"oil={ev['oil_ignition_detonations']:.1f} boil={ev['flash_boil_steam_bursts']:.1f} "
                f"cryo={ev['cryo_crossings']:.1f} smk={ev['smokescreen_covers']:.1f}{tag}",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    s2_summary = {
        "stage": stage_label,
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "parallel_arenas": A,
        "updates": num_updates,
        "rollout_steps": rollout_steps,
        "total_bot_steps": total_bot_steps,
        "elapsed_s": elapsed,
        "bot_steps_per_sec": round(total_bot_steps / max(elapsed, 1e-3), 1),
        "kl_early_stops_triggered": kl_stops,
        "best_val": best_ev,
    }
    return s2_summary, best_sd, history


# ---------------------------------------------------------------------------
# Held-Out 256x256x4 Tournament (24 Side-Balanced Matches / Pair, 4 Opponents)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _run_agent_v2(
    kind: str,
    policy_v2: ActuatorPolicyNet,
    policy_v1: ActuatorPolicyNet,
    arena: PhysicsActuatorArena,
    feats: torch.Tensor,
) -> torch.Tensor:
    if kind == "rl_v2":
        return policy_deterministic_act(policy_v2, arena.z_world, arena.z_eco, feats, thermal_only=False)
    if kind == "rl_v1":
        return policy_deterministic_act(policy_v1, arena.z_world, arena.z_eco, feats, thermal_only=False)
    if kind == "thermal_only":
        return policy_deterministic_act(policy_v2, arena.z_world, arena.z_eco, feats, thermal_only=True)
    if kind == "legacy_scripted":
        return legacy_scripted_actuator_baseline(arena)
    if kind == "random":
        return random_continuous_act(arena)
    raise ValueError(f"Unknown agent kind: {kind}")


@torch.inference_mode()
def run_heldout_matchup_v2(
    name_a: str,
    kind_a: str,
    name_b: str,
    kind_b: str,
    policy_v2: ActuatorPolicyNet,
    policy_v1: ActuatorPolicyNet,
    arena: PhysicsActuatorArena,
    num_matches: int = 24,
    base_seed: int = 95000,
) -> dict[str, Any]:
    """Exact side-balanced 24-match evaluation on unseen held-out 256x256x4 worlds."""
    assert num_matches % 2 == 0
    half = num_matches // 2
    batch = arena.A
    assert half % batch == 0, f"half ({half}) must be divisible by arena.A ({batch})"
    rounds = half // batch

    t0 = time.perf_counter()
    wins_a = wins_b = draws = 0
    scores_a, scores_b = [], []
    kills_a, kills_b = [], []
    deaths_a, deaths_b = [], []
    shrines_a, shrines_b = [], []
    relics_a, relics_b = [], []
    press_a, press_b = [], []

    em_keys = ("oil_ignition_detonations", "flash_boil_steam_bursts", "cryo_crossings", "smokescreen_covers", "kinetic_pushes")
    em_a_acc = {k: [] for k in em_keys}
    em_b_acc = {k: [] for k in em_keys}
    foc_a = [0.0] * 13

    for side in (0, 1):
        mask_a = torch.zeros((batch, NUM_BOTS), dtype=torch.bool, device=arena.device)
        if side == 0:
            mask_a[:, :6] = True
        else:
            mask_a[:, 6:] = True

        for r in range(rounds):
            seed = base_seed + r * 31
            f = arena.reset(seed=seed)
            for _ in range(arena.max_ticks):
                act_a = _run_agent_v2(kind_a, policy_v2, policy_v1, arena, f)
                act_b = _run_agent_v2(kind_b, policy_v2, policy_v1, arena, f)
                act = torch.where(mask_a.unsqueeze(-1), act_a, act_b)
                f, _, _, _ = arena.step(act, collect_behavior_samples=False)

            sa = arena.team_score[:, side]
            sb = arena.team_score[:, 1 - side]
            for i in range(batch):
                va, vb = float(sa[i].item()), float(sb[i].item())
                if va > vb:
                    wins_a += 1
                elif vb > va:
                    wins_b += 1
                else:
                    draws += 1
                scores_a.append(va)
                scores_b.append(vb)
                kills_a.append(float(arena.team_kills[i, side].item()))
                kills_b.append(float(arena.team_kills[i, 1 - side].item()))
                deaths_a.append(float(arena.team_deaths[i, side].item()))
                deaths_b.append(float(arena.team_deaths[i, 1 - side].item()))
                shrines_a.append(float(arena.team_shrine_caps[i, side].item()))
                shrines_b.append(float(arena.team_shrine_caps[i, 1 - side].item()))
                relics_a.append(float(arena.team_relic_deliveries[i, side].item()))
                relics_b.append(float(arena.team_relic_deliveries[i, 1 - side].item()))
                press_a.append(float(arena.team_pressure_dmg_dealt[i, side].item()))
                press_b.append(float(arena.team_pressure_dmg_dealt[i, 1 - side].item()))

                em_a_acc["oil_ignition_detonations"].append(float(arena.emerge_oil_ignite[i, side].item()))
                em_b_acc["oil_ignition_detonations"].append(float(arena.emerge_oil_ignite[i, 1 - side].item()))
                em_a_acc["flash_boil_steam_bursts"].append(float(arena.emerge_flash_boil[i, side].item()))
                em_b_acc["flash_boil_steam_bursts"].append(float(arena.emerge_flash_boil[i, 1 - side].item()))
                em_a_acc["cryo_crossings"].append(float(arena.emerge_cryo_cross[i, side].item()))
                em_b_acc["cryo_crossings"].append(float(arena.emerge_cryo_cross[i, 1 - side].item()))
                em_a_acc["smokescreen_covers"].append(float(arena.emerge_smokescreen[i, side].item()))
                em_b_acc["smokescreen_covers"].append(float(arena.emerge_smokescreen[i, 1 - side].item()))
                em_a_acc["kinetic_pushes"].append(float(arena.emerge_kinetic_push[i, side].item()))
                em_b_acc["kinetic_pushes"].append(float(arena.emerge_kinetic_push[i, 1 - side].item()))

            foc_a[0] += float(arena.focus_count[:, side].sum().item())
            foc_a[1] += float(arena.focus_sum_f[:, side].sum().item())
            foc_a[2] += float(arena.focus_sum_f2[:, side].sum().item())
            foc_a[3] += float(arena.focus_sum_c[:, side].sum().item())
            foc_a[4] += float(arena.focus_sum_c2[:, side].sum().item())
            foc_a[5] += float(arena.focus_sum_fc[:, side].sum().item())
            foc_a[6] += float(arena.focus_sum_d[:, side].sum().item())
            foc_a[7] += float(arena.focus_sum_d2[:, side].sum().item())
            foc_a[8] += float(arena.focus_sum_fd[:, side].sum().item())
            foc_a[9] += float(arena.focus_single_sum[:, side].sum().item())
            foc_a[10] += float(arena.focus_single_n[:, side].sum().item())
            foc_a[11] += float(arena.focus_multi_sum[:, side].sum().item())
            foc_a[12] += float(arena.focus_multi_n[:, side].sum().item())

    elapsed = round(time.perf_counter() - t0, 2)
    wr_a = round((wins_a + 0.5 * draws) / num_matches * 100.0, 2)
    wr_b = round((wins_b + 0.5 * draws) / num_matches * 100.0, 2)
    mean_f = foc_a[1] / max(foc_a[0], 1.0)
    std_f = math.sqrt(max(0.0, foc_a[2] / max(foc_a[0], 1.0) - mean_f * mean_f))

    res = {
        "matchup": f"{name_a} vs {name_b}",
        "team_a": name_a,
        "team_b": name_b,
        "matches": num_matches,
        "ticks_per_match": arena.max_ticks,
        "grid": f"{arena.H}x{arena.W}x{arena.L}",
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a_pct": wr_a,
        "win_rate_b_pct": wr_b,
        "mean_score_a": round(float(np.mean(scores_a)), 2),
        "mean_score_b": round(float(np.mean(scores_b)), 2),
        "kills_per_match_a": round(float(np.mean(kills_a)), 2),
        "kills_per_match_b": round(float(np.mean(kills_b)), 2),
        "deaths_per_match_a": round(float(np.mean(deaths_a)), 2),
        "deaths_per_match_b": round(float(np.mean(deaths_b)), 2),
        "shrine_caps_per_match_a": round(float(np.mean(shrines_a)), 2),
        "shrine_caps_per_match_b": round(float(np.mean(shrines_b)), 2),
        "relic_deliveries_per_match_a": round(float(np.mean(relics_a)), 2),
        "relic_deliveries_per_match_b": round(float(np.mean(relics_b)), 2),
        "pressure_wave_dmg_per_match_a": round(float(np.mean(press_a)), 2),
        "pressure_wave_dmg_per_match_b": round(float(np.mean(press_b)), 2),
        "emergent_tactics_a_per_match": {k: round(float(np.mean(em_a_acc[k])), 2) for k in em_keys},
        "emergent_tactics_b_per_match": {k: round(float(np.mean(em_b_acc[k])), 2) for k in em_keys},
        "adaptive_focus_a": {
            "mean_radius_cells": round(mean_f, 3),
            "std_radius_cells": round(std_f, 3),
            "corr_vs_enemy_cluster": round(_pearson_from_sums(foc_a[0], foc_a[1], foc_a[2], foc_a[3], foc_a[4], foc_a[5]), 4),
            "corr_vs_emit_distance": round(_pearson_from_sums(foc_a[0], foc_a[1], foc_a[2], foc_a[6], foc_a[7], foc_a[8]), 4),
            "single_enemy_mean_cells": round(foc_a[9] / max(foc_a[10], 1.0), 3),
            "multi_enemy_mean_cells": round(foc_a[11] / max(foc_a[12], 1.0), 3),
        },
        "eval_time_s": elapsed,
    }
    print(
        f"  [256x256x4 24m] {name_a} vs {name_b}: WR={wr_a:5.1f}% ({wins_a}W-{wins_b}L-{draws}D) | "
        f"Score {res['mean_score_a']:.1f} vs {res['mean_score_b']:.1f} | "
        f"K={res['kills_per_match_a']:.1f} Shr={res['shrine_caps_per_match_a']:.2f} Rel={res['relic_deliveries_per_match_a']:.2f} | "
        f"OilIgn={res['emergent_tactics_a_per_match']['oil_ignition_detonations']:.1f} "
        f"Boil={res['emergent_tactics_a_per_match']['flash_boil_steam_bursts']:.1f} "
        f"Cryo={res['emergent_tactics_a_per_match']['cryo_crossings']:.1f} "
        f"Smoke={res['emergent_tactics_a_per_match']['smokescreen_covers']:.1f} | {elapsed:.1f}s",
        flush=True,
    )
    return res


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ActuatorPolicyNet v2 via League Self-Play RL.")
    parser.add_argument("--work-dir", type=str, default=".", help="Workspace root.")
    parser.add_argument("--world-pt", type=str, default="", help="Path to world_model_v3.pt")
    parser.add_argument("--eco-pt", type=str, default="", help="Path to ecosystem_model.pt")
    parser.add_argument("--warm-start-pt", type=str, default="", help="Path to actuator_policy_rl.pt (Iter 5A)")
    parser.add_argument("--resume-pt", type=str, default="", help="Optional Stage-1 checkpoint to resume Stage-2 polish from")
    parser.add_argument("--out-ckpt", type=str, default="", help="Output actuator_policy_v2_rl.pt path")
    parser.add_argument("--out-json", type=str, default="", help="Output ACTUATOR_V2_TOURNAMENT.json path")
    parser.add_argument("--num-arenas", type=int, default=64, help="Parallel training arenas (default 64 -> 768 bots)")
    parser.add_argument("--updates", type=int, default=520, help="Stage 1 Pointer-Categorical League PPO updates (128x128x2)")
    parser.add_argument("--stage2-updates", type=int, default=0, help="Stage 2 continuation updates (128x128x2)")
    parser.add_argument("--stage3-updates", type=int, default=48, help="Stage 3 high-res 256x256x4 polish updates (24 arenas)")
    parser.add_argument("--rollout-steps", type=int, default=30, help="Ticks per PPO rollout")
    parser.add_argument("--eval-matches", type=int, default=24, help="Held-out 256x256x4 matches per matchup")
    parser.add_argument("--eval-batch", type=int, default=6, help="Parallel 256x256x4 eval arenas")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()

    def _find_file(explicit: str, candidates: list[Path]) -> Path:
        if explicit:
            return Path(explicit).resolve()
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    world_pt = _find_file(
        args.world_pt,
        [
            work_dir / "models" / "checkpoints" / "world_model_v3.pt",
            work_dir / "world_model_v3.pt",
            _SCRIPT_DIR.parent / "models" / "checkpoints" / "world_model_v3.pt",
        ],
    )
    eco_pt = _find_file(
        args.eco_pt,
        [
            work_dir / "models" / "checkpoints" / "ecosystem_model.pt",
            work_dir / "ecosystem_model.pt",
            _SCRIPT_DIR.parent / "models" / "checkpoints" / "ecosystem_model.pt",
        ],
    )
    warm_pt = _find_file(
        args.warm_start_pt,
        [
            work_dir / "models" / "checkpoints" / "actuator_policy_rl.pt",
            work_dir / "actuator_policy_rl.pt",
            _SCRIPT_DIR.parent / "models" / "checkpoints" / "actuator_policy_rl.pt",
        ],
    )
    out_ckpt = Path(args.out_ckpt).resolve() if args.out_ckpt else (work_dir / "models" / "checkpoints" / "actuator_policy_v2_rl.pt")
    out_json = Path(args.out_json).resolve() if args.out_json else (work_dir / "models" / "ACTUATOR_V2_TOURNAMENT.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(2027)
    np.random.seed(2027)

    print(f"=== Frostmarsh ActuatorPolicyNet v2 League Self-Play RL ({gpu_name}) ===", flush=True)
    print(f"  WorldModel v3:  {world_pt}", flush=True)
    print(f"  EcosystemModel: {eco_pt}", flush=True)
    print(f"  Warm-Start v1:  {warm_pt}", flush=True)

    wm, em = load_frozen_models(world_pt, eco_pt, device)

    # Load frozen Iter 5A snapshot + warm-start exact drop-in ActuatorPolicyNet (214,624 params!)
    v1_sd = load_ckpt_state_dict(warm_pt, device)
    v1_policy = ActuatorPolicyNet().to(device).eval()
    v1_policy.load_state_dict(v1_sd, strict=True)
    for p in v1_policy.parameters():
        p.requires_grad_(False)

    policy = ActuatorPolicyNet().to(device)
    if args.resume_pt:
        res_sd = load_ckpt_state_dict(Path(args.resume_pt).resolve(), device)
        policy.load_state_dict(copy.deepcopy(res_sd), strict=True)
        print(f"  Resumed Stage-1 checkpoint for Stage-2 polish: {args.resume_pt}", flush=True)
    else:
        policy.load_state_dict(copy.deepcopy(v1_sd), strict=True)
        # Softly de-saturate the 4 final action heads while preserving Iter 5A's learned Z_world/Z_eco/attention trunk!
        with torch.no_grad():
            policy.head_ptr_mv.weight.mul_(0.25)
            policy.head_ptr_mv.bias.zero_()
            policy.head_ptr_em.weight.mul_(0.40)
            policy.head_ptr_em.bias.zero_()
            policy.head_bipolar.weight.mul_(0.35)
            policy.head_bipolar.bias.zero_()
            policy.head_unipolar.weight.mul_(0.35)
            policy.head_unipolar.bias.copy_(torch.tensor([-2.00, -1.00, -1.35, -1.00, -1.30], device=device))
    n_params = count_parameters(policy)
    assert n_params == 214624, f"Expected exact 214,624 params, got {n_params}"
    print(f"  ActuatorPolicyNet params: {n_params:,} (EXACT op=7 drop-in compatible!)", flush=True)

    # Verify op=7 API contract upfront
    with torch.inference_mode():
        tz_w = torch.zeros((4, 48, 64, 64), device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        tz_e = torch.zeros((1, 32, 64, 64), device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        tbf = torch.zeros((12, BOT_FEAT_DIM), device=device, dtype=torch.float32)
        t_act, t_sal = policy(tz_w, tz_e, tbf, return_saliency=True)
        assert tuple(t_act.shape) == (12, 10), f"Bad act shape: {t_act.shape}"
        assert tuple(t_sal.shape) == (12, 16), f"Bad saliency shape: {t_sal.shape}"

    ac = RLActuatorActorCriticV2(policy).to(device)

    train_arena = LeaguePhysicsActuatorArena(
        num_arenas=args.num_arenas,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=128,
        num_layers=2,
        max_ticks=120,
        latent_refresh_every=4,
    )
    val_arena = PhysicsActuatorArena(
        num_arenas=32,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=128,
        num_layers=2,
        max_ticks=120,
        latent_refresh_every=4,
    )

    history: list[dict] = []
    best_sd = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    train_summary: dict[str, Any] = {
        "grid": "128x128x2",
        "parallel_arenas": args.num_arenas,
        "concurrent_bots": args.num_arenas * NUM_BOTS,
        "league_mix": {
            "pure_mirror_self_play_arenas": 12,
            "lagged_self_play_arenas": 12,
            "legacy_scripted_arenas": 24,
            "thermal_only_arenas": 8,
            "iter5a_snapshot_arenas": 8,
        },
        "zero_imitation_teacher": True,
        "zero_hardcoded_spell_rules": True,
    }

    if args.updates > 0:
        s1_summary, best_sd, s1_hist = train_league_self_play_ppo(
            ac=ac,
            v1_policy=v1_policy,
            arena=train_arena,
            val_arena=val_arena,
            num_updates=args.updates,
            rollout_steps=args.rollout_steps,
            ppo_epochs=4,
            minibatch_size=2048,
            lr=7.0e-4,
            kl_early_stop=0.25,
            eval_every=max(1, args.updates // 16),
            seed_offset=2000,
        )
        policy.load_state_dict(best_sd, strict=True)
        train_summary["stage1_pointer_categorical"] = s1_summary
        train_summary["best_val"] = s1_summary["best_val"]
        history.extend(s1_hist)

    if args.stage2_updates > 0:
        s2_summary, best_sd, s2_hist = train_stage2_soft_pointer_ppo(
            ac=ac,
            v1_policy=v1_policy,
            arena=train_arena,
            val_arena=val_arena,
            num_updates=args.stage2_updates,
            rollout_steps=args.rollout_steps,
            ppo_epochs=4,
            minibatch_size=2048,
            lr=2.8e-4,
            kl_early_stop=0.18,
            eval_every=max(1, args.stage2_updates // 16),
            critic_warmup=18,
            seed_offset=9000,
            stage_label="Stage2-128",
        )
        policy.load_state_dict(best_sd, strict=True)
        train_summary["stage2_128_continuation"] = s2_summary
        train_summary["best_val"] = s2_summary["best_val"]
        history.extend(s2_hist)

    # Free 128x128 training buffers before optional 256x256x4 high-res polish & tournament
    del train_arena, val_arena
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.stage3_updates > 0:
        hr_train = LeaguePhysicsActuatorArena(
            num_arenas=24,
            world_model=wm,
            eco_model=em,
            device=device,
            grid_size=256,
            num_layers=4,
            max_ticks=120,
            latent_refresh_every=4,
        )
        hr_val = PhysicsActuatorArena(
            num_arenas=12,
            world_model=wm,
            eco_model=em,
            device=device,
            grid_size=256,
            num_layers=4,
            max_ticks=120,
            latent_refresh_every=4,
        )
        s3_summary, best_sd, s3_hist = train_stage2_soft_pointer_ppo(
            ac=ac,
            v1_policy=v1_policy,
            arena=hr_train,
            val_arena=hr_val,
            num_updates=args.stage3_updates,
            rollout_steps=30,
            ppo_epochs=4,
            minibatch_size=1024,
            lr=1.5e-4,
            kl_early_stop=0.16,
            eval_every=max(1, args.stage3_updates // 8),
            critic_warmup=8,
            seed_offset=15000,
            stage_label="Stage3-256x4",
        )
        policy.load_state_dict(best_sd, strict=True)
        train_summary["stage3_256x4_polish"] = s3_summary
        train_summary["best_val_256x4"] = s3_summary["best_val"]
        history.extend(s3_hist)
        del hr_train, hr_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Load best checkpoint & verify exact op=7 state_dict parity
    policy.load_state_dict(best_sd, strict=True)
    policy.eval()
    pure_sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    assert list(pure_sd.keys()) == list(v1_sd.keys()), "state_dict keys diverged from v1!"

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": pure_sd,
            "arch": "ActuatorPolicyNet",
            "version": "v2_league_rl",
            "params": n_params,
            "actuator_dim": ACTUATOR_DIM,
            "bot_feat_dim": BOT_FEAT_DIM,
            "fine_radius_cells": 14.0,
            "coarse_radius_cells": 60.0,
            "warm_started_from": str(warm_pt),
            "zero_hardcoded_rules": True,
            "zero_imitation_teacher": True,
        },
        str(out_ckpt),
    )
    print(f"[Checkpoint] Saved ActuatorPolicyNet v2 -> {out_ckpt}", flush=True)

    print(f"\n=== Held-Out Unseen 256x256x4 Tournament ({args.eval_matches} matches/pair) ===", flush=True)
    eval_arena = PhysicsActuatorArena(
        num_arenas=args.eval_batch,
        world_model=wm,
        eco_model=em,
        device=device,
        grid_size=256,
        num_layers=4,
        max_ticks=120,
        latent_refresh_every=4,
    )

    t_tour0 = time.perf_counter()
    matchups: dict[str, dict] = {}
    matchups["v2_vs_legacy_scripted"] = run_heldout_matchup_v2(
        "ActuatorPolicyNet-v2",
        "rl_v2",
        "Legacy Named-Spell Scripted",
        "legacy_scripted",
        policy,
        v1_policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=95000,
    )
    matchups["v2_vs_v1_iter5a"] = run_heldout_matchup_v2(
        "ActuatorPolicyNet-v2",
        "rl_v2",
        "ActuatorPolicyNet-v1 (Iter 5A)",
        "rl_v1",
        policy,
        v1_policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=98000,
    )
    matchups["v2_vs_thermal_only"] = run_heldout_matchup_v2(
        "ActuatorPolicyNet-v2",
        "rl_v2",
        "Thermal-Only Actuators",
        "thermal_only",
        policy,
        v1_policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=96000,
    )
    matchups["v2_vs_random"] = run_heldout_matchup_v2(
        "ActuatorPolicyNet-v2",
        "rl_v2",
        "Random Continuous Actuators",
        "random",
        policy,
        v1_policy,
        eval_arena,
        num_matches=args.eval_matches,
        base_seed=97000,
    )
    tour_time = round(time.perf_counter() - t_tour0, 2)

    del eval_arena
    if device.type == "cuda":
        torch.cuda.empty_cache()

    latency = benchmark_inference_latency(policy, device)
    print(
        f"[Latency {gpu_name}] 12b={latency['bots_12']['p50_ms']}ms | "
        f"64b={latency['bots_64']['p50_ms']}ms | 256b={latency['bots_256']['p50_ms']}ms",
        flush=True,
    )

    # Aggregate 5 emergent tactics summary across competitive matchups (vs Legacy, vs v1, vs Thermal)
    comp_keys = ("v2_vs_legacy_scripted", "v2_vs_v1_iter5a", "v2_vs_thermal_only")
    emergent_summary = {
        "oil_ignition_detonations_per_match": round(
            float(np.mean([matchups[k]["emergent_tactics_a_per_match"]["oil_ignition_detonations"] for k in comp_keys])), 2
        ),
        "flash_boil_steam_bursts_per_match": round(
            float(np.mean([matchups[k]["emergent_tactics_a_per_match"]["flash_boil_steam_bursts"] for k in comp_keys])), 2
        ),
        "cryo_crossings_per_match": round(
            float(np.mean([matchups[k]["emergent_tactics_a_per_match"]["cryo_crossings"] for k in comp_keys])), 2
        ),
        "smokescreen_covers_per_match": round(
            float(np.mean([matchups[k]["emergent_tactics_a_per_match"]["smokescreen_covers"] for k in comp_keys])), 2
        ),
        "kinetic_pushes_per_match": round(
            float(np.mean([matchups[k]["emergent_tactics_a_per_match"]["kinetic_pushes"] for k in comp_keys])), 2
        ),
        "adaptive_focus_std_cells_vs_legacy": matchups["v2_vs_legacy_scripted"]["adaptive_focus_a"]["std_radius_cells"],
        "adaptive_focus_mean_cells_vs_legacy": matchups["v2_vs_legacy_scripted"]["adaptive_focus_a"]["mean_radius_cells"],
        "adaptive_focus_corr_cluster_vs_legacy": matchups["v2_vs_legacy_scripted"]["adaptive_focus_a"]["corr_vs_enemy_cluster"],
    }

    metrics = {
        "device": gpu_name,
        "policy_class": "ActuatorPolicyNet",
        "policy_version": "v2_league_self_play",
        "policy_params": n_params,
        "actuator_channels": 10,
        "op7_drop_in_compatible": True,
        "zero_hardcoded_spell_rules": True,
        "zero_imitation_teacher": True,
        "warm_started_from": "actuator_policy_rl.pt",
        "world_model_version": "world_model_v3.pt",
        "ecosystem_model": "ecosystem_model.pt",
        "eval_grid": "256x256x4",
        "matches_per_matchup": args.eval_matches,
        "ticks_per_match": 120,
        "side_balanced": True,
        "training": train_summary,
        "learning_curve": history,
        "tournament_time_s": tour_time,
        "matchups": matchups,
        "emergent_tactics_summary": emergent_summary,
        "latency_gpu": latency,
        "artifacts": {
            "checkpoint": str(out_ckpt),
            "metrics_json": str(out_json),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"[Metrics] Wrote complete tournament JSON -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
