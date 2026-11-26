//! PufferLib-inspired zero-allocation, multi-threaded, double-buffered headless Rust RL arena
//! reusing `field-core::Field` exact conservation physics, 5 hero classes, shrines, golden relic,
//! 184-float observations (168 base + 16 NMMO 2.0 TaskBrief/GRU-slot tail), 6 dense 0..1 progress
//! predicates, and native scripted teacher mask.
//!
//! Designed to be loaded zero-copy from Python via `ctypes` (`libfield_core.{dylib,so}`) so
//! `train_bot_rl_v3.py` steps 12/32/60-bot arenas across all CPU cores while the GPU runs PPO.

use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread::{self, JoinHandle};

use crate::{
    Field, INPUT_CHANNELS, SP_ICE, SP_OIL, SP_PLANT, SP_ROCK, SP_SMOKE, SP_WATER, SP_WOOD,
    T_AMB,
};

pub const RL_SELF_DIM: usize = 24;
pub const RL_NUM_TOKENS: usize = 12;
pub const RL_TOKEN_DIM: usize = 12;
pub const RL_BASE_FEAT_DIM: usize = RL_SELF_DIM + RL_NUM_TOKENS * RL_TOKEN_DIM; // 168
pub const RL_BRIEF_DIM: usize = 16;
pub const RL_OBS_DIM: usize = RL_BASE_FEAT_DIM + RL_BRIEF_DIM; // 184
pub const RL_ACTION_DIM: usize = 13;
pub const RL_PRED_DIM: usize = 6;
pub const RL_BRIEF_IN_DIM: usize = 10;
pub const RL_CROP_CH: usize = 8;
pub const RL_CROP_SIZE: usize = 16;
pub const RL_CROP_STRIDE: usize = RL_CROP_CH * RL_CROP_SIZE * RL_CROP_SIZE;

// Objectives (NMMO 2.0-style conditioning)
pub const OBJ_FREE_ROAM: u8 = 0;
pub const OBJ_CAPTURE_SHRINE: u8 = 1;
pub const OBJ_ESCORT_RELIC: u8 = 2;
pub const OBJ_HUNT_ENEMY: u8 = 3;
pub const OBJ_DEFEND_HOLD: u8 = 4;

// Stances
pub const STANCE_COMMIT: u8 = 0;
pub const STANCE_HOLD: u8 = 1;
pub const STANCE_FLEE_KITE: u8 = 2;

#[derive(Clone, Copy, Debug)]
struct XorShift64(u64);

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / ((1u64 << 24) as f32))
    }
}

#[derive(Clone, Copy, Debug)]
struct BotBrief {
    objective: u8,
    stance: u8,
    target_x: f32,
    target_y: f32,
    risk_budget: f32,
    active: bool,
    thermal_bias: f32,
}

impl Default for BotBrief {
    fn default() -> Self {
        Self {
            objective: OBJ_FREE_ROAM,
            stance: STANCE_COMMIT,
            target_x: 64.0,
            target_y: 64.0,
            risk_budget: 0.65,
            active: false,
            thermal_bias: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BotState {
    id: u32,
    team: u8,
    class_id: u8,
    squad_id: u8,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    hp: f32,
    max_hp: f32,
    mana: f32,
    max_mana: f32,
    stamina: f32,
    max_stamina: f32,
    level: u8,
    carrying_relic: bool,
    oiled_until: u32,
    chilled_until: u32,
    dash_cd: u32,
    melee_cd: u32,
    bolt_cd: u32,
    ability_e_cd: u32,
    ult_r_cd: u32,
    prev_target_dist: f32,
    just_respawned: bool,
    brief: BotBrief,
}

#[derive(Clone, Copy, Debug)]
struct ShrineState {
    x: f32,
    y: f32,
    progress: f32, // -1.0 (team 0) .. +1.0 (team 1)
}

#[derive(Clone, Copy, Debug)]
struct Projectile {
    owner_idx: u16,
    team: u8,
    elem: u8,
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    ttl: u8,
}

struct ArenaInstance {
    arena_id: usize,
    grid_w: usize,
    grid_h: usize,
    field: Field,
    bots: Vec<BotState>,
    shrines: [ShrineState; 3],
    relic_x: f32,
    relic_y: f32,
    relic_carrier: Option<usize>,
    projectiles: Vec<Projectile>,
    tick: u32,
    episode_len: u32,
    field_step_interval: u32,
    weather_kind: u8,
    team_score: [f32; 2],
    team_kills: [f32; 2],
    rng: XorShift64,
    // Pre-allocated scratch for zero-alloc per-step nearest-neighbor sorting
    dist_scratch: Vec<(f32, usize)>,
}

impl ArenaInstance {
    fn new(
        arena_id: usize,
        bots_per_arena: usize,
        grid_w: usize,
        grid_h: usize,
        seed: u64,
        field_step_interval: u32,
    ) -> Self {
        let mut inst = Self {
            arena_id,
            grid_w,
            grid_h,
            field: Field::new_frostmarsh(grid_w, grid_h),
            bots: Vec::with_capacity(bots_per_arena),
            shrines: [
                ShrineState {
                    x: grid_w as f32 * 0.50,
                    y: grid_h as f32 * 0.24,
                    progress: 0.0,
                },
                ShrineState {
                    x: grid_w as f32 * 0.50,
                    y: grid_h as f32 * 0.50,
                    progress: 0.0,
                },
                ShrineState {
                    x: grid_w as f32 * 0.50,
                    y: grid_h as f32 * 0.76,
                    progress: 0.0,
                },
            ],
            relic_x: grid_w as f32 * 0.50,
            relic_y: grid_h as f32 * 0.50,
            relic_carrier: None,
            projectiles: Vec::with_capacity(bots_per_arena * 2),
            tick: 0,
            episode_len: 320,
            field_step_interval: field_step_interval.max(1),
            weather_kind: 0,
            team_score: [0.0; 2],
            team_kills: [0.0; 2],
            rng: XorShift64::new(seed ^ ((arena_id as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03))),
            dist_scratch: Vec::with_capacity(bots_per_arena),
        };
        inst.reset_inner(bots_per_arena, seed);
        inst
    }

    #[inline]
    fn class_stats(class_id: u8) -> (f32, f32, f32, f32) {
        // (max_hp, max_mana, max_stamina, move_speed)
        match class_id % 5 {
            0 => (125.0, 140.0, 110.0, 18.0), // Pyromancer
            1 => (130.0, 145.0, 115.0, 17.5), // Frostweaver
            2 => (135.0, 130.0, 125.0, 18.2), // Alchemist
            3 => (122.0, 125.0, 145.0, 19.6), // Emberstalker
            _ => (160.0, 115.0, 130.0, 16.4), // Burrower
        }
    }

    fn spawn_xy(&mut self, team: u8) -> (f32, f32) {
        let wf = self.grid_w as f32;
        let hf = self.grid_h as f32;
        let x_base = if team == 0 { wf * 0.16 } else { wf * 0.84 };
        let x = (x_base + (self.rng.next_f32() - 0.5) * wf * 0.16).clamp(3.0, wf - 4.0);
        let y = (hf * 0.12 + self.rng.next_f32() * hf * 0.76).clamp(3.0, hf - 4.0);
        (x, y)
    }

    fn reset_inner(&mut self, bots_per_arena: usize, seed: u64) {
        self.rng = XorShift64::new(
            seed ^ ((self.arena_id as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        self.field = Field::new_frostmarsh(self.grid_w, self.grid_h);
        self.tick = 0;
        self.weather_kind = (self.rng.next_u64() % 4) as u8;
        self.team_score = [0.0; 2];
        self.team_kills = [0.0; 2];
        self.projectiles.clear();
        self.relic_carrier = None;
        self.relic_x = self.grid_w as f32 * 0.50;
        self.relic_y = self.grid_h as f32 * 0.50;
        self.shrines = [
            ShrineState {
                x: self.grid_w as f32 * 0.50,
                y: self.grid_h as f32 * 0.24,
                progress: 0.0,
            },
            ShrineState {
                x: self.grid_w as f32 * 0.50,
                y: self.grid_h as f32 * 0.50,
                progress: 0.0,
            },
            ShrineState {
                x: self.grid_w as f32 * 0.50,
                y: self.grid_h as f32 * 0.76,
                progress: 0.0,
            },
        ];

        self.bots.clear();
        for b in 0..bots_per_arena {
            let team = (b % 2) as u8;
            let class_id = ((b / 2) % 5) as u8;
            let squad_id = ((b / 2) % 4) as u8;
            let (max_hp, max_mana, max_stamina, _) = Self::class_stats(class_id);
            let (x, y) = self.spawn_xy(team);
            let shrine_idx = (squad_id as usize) % 3;
            let tx = self.shrines[shrine_idx].x;
            let ty = self.shrines[shrine_idx].y;
            let init_dist = (tx - x).hypot(ty - y);
            self.bots.push(BotState {
                id: (self.arena_id * 256 + b + 1) as u32,
                team,
                class_id,
                squad_id,
                x,
                y,
                z: 1.0,
                vx: 0.0,
                vy: 0.0,
                hp: max_hp,
                max_hp,
                mana: max_mana,
                max_mana,
                stamina: max_stamina,
                max_stamina,
                level: 1,
                carrying_relic: false,
                oiled_until: 0,
                chilled_until: 0,
                dash_cd: 0,
                melee_cd: 0,
                bolt_cd: 0,
                ability_e_cd: 0,
                ult_r_cd: 0,
                prev_target_dist: init_dist,
                just_respawned: true,
                brief: BotBrief {
                    objective: OBJ_FREE_ROAM,
                    stance: STANCE_COMMIT,
                    target_x: tx,
                    target_y: ty,
                    risk_budget: 0.65,
                    active: false,
                    thermal_bias: 0.0,
                },
            });
        }
    }

    fn sample_random_briefs(&mut self, active_prob: f32) {
        let wf = self.grid_w as f32;
        let hf = self.grid_h as f32;
        for bi in 0..self.bots.len() {
            let active = self.rng.next_f32() < active_prob;
            let objective = if active {
                1 + (self.rng.next_u64() % 4) as u8
            } else {
                OBJ_FREE_ROAM
            };
            let stance = (self.rng.next_u64() % 3) as u8;
            let (tx, ty) = match objective {
                OBJ_CAPTURE_SHRINE | OBJ_DEFEND_HOLD => {
                    let s = &self.shrines[(self.rng.next_u64() % 3) as usize];
                    (s.x, s.y)
                }
                OBJ_ESCORT_RELIC => (self.relic_x, self.relic_y),
                OBJ_HUNT_ENEMY => {
                    let team = self.bots[bi].team;
                    let ex = if team == 0 { wf * 0.70 } else { wf * 0.30 };
                    let ey = hf * (0.2 + 0.6 * self.rng.next_f32());
                    (ex, ey)
                }
                _ => (wf * 0.5, hf * 0.5),
            };
            let risk_budget = match stance {
                STANCE_FLEE_KITE => 0.15 + 0.25 * self.rng.next_f32(),
                STANCE_HOLD => 0.35 + 0.30 * self.rng.next_f32(),
                _ => 0.60 + 0.40 * self.rng.next_f32(),
            };
            let thermal_bias = (self.rng.next_f32() * 2.0 - 1.0).clamp(-1.0, 1.0);
            let b = &mut self.bots[bi];
            b.brief = BotBrief {
                objective,
                stance,
                target_x: tx,
                target_y: ty,
                risk_budget,
                active,
                thermal_bias,
            };
            b.prev_target_dist = (tx - b.x).hypot(ty - b.y);
        }
    }

    fn scripted_teacher_action(&self, bi: usize) -> [f32; RL_ACTION_DIM] {
        let b = &self.bots[bi];
        let wf = self.grid_w as f32;
        let hf = self.grid_h as f32;

        // Find nearest enemy
        let mut best_enemy_d = f32::MAX;
        let mut enemy_dx = 0.0f32;
        let mut enemy_dy = 0.0f32;
        let mut enemy_oiled = false;
        for (oj, o) in self.bots.iter().enumerate() {
            if oj == bi || o.team == b.team {
                continue;
            }
            let dx = o.x - b.x;
            let dy = o.y - b.y;
            let d = dx.hypot(dy);
            if d < best_enemy_d {
                best_enemy_d = d;
                enemy_dx = dx;
                enemy_dy = dy;
                enemy_oiled = self.tick < o.oiled_until;
            }
        }

        // Choose movement objective (brief-aware if active!)
        let (mut goal_x, mut goal_y) = if b.brief.active {
            if b.brief.stance == STANCE_FLEE_KITE && best_enemy_d < 28.0 {
                // Retreat away from enemy toward home bank / brief target
                let inv = 1.0 / best_enemy_d.max(1e-3);
                (b.x - enemy_dx * inv * 24.0, b.y - enemy_dy * inv * 24.0)
            } else if b.brief.objective == OBJ_HUNT_ENEMY && best_enemy_d < 80.0 {
                (b.x + enemy_dx, b.y + enemy_dy)
            } else {
                (b.brief.target_x, b.brief.target_y)
            }
        } else if b.carrying_relic {
            let home_x = if b.team == 0 { wf * 0.14 } else { wf * 0.86 };
            (home_x, hf * 0.5)
        } else {
            // Contest nearest uncaptured shrine
            let mut best_s = (wf * 0.5, hf * 0.5);
            let mut best_sd = f32::MAX;
            for s in &self.shrines {
                let need = if b.team == 0 { s.progress > -0.7 } else { s.progress < 0.7 };
                let d = (s.x - b.x).hypot(s.y - b.y) * if need { 1.0 } else { 2.5 };
                if d < best_sd {
                    best_sd = d;
                    best_s = (s.x, s.y);
                }
            }
            best_s
        };

        // Hold stance: stop moving once near goal and maintain spacing
        let gdx = goal_x - b.x;
        let gdy = goal_y - b.y;
        let gdist = gdx.hypot(gdy);
        let (mut mx, mut my) = if b.brief.active && b.brief.stance == STANCE_HOLD && gdist < 6.0 {
            if best_enemy_d < 10.0 {
                let inv = 1.0 / best_enemy_d.max(1e-3);
                (-enemy_dx * inv * 0.6, -enemy_dy * inv * 0.6)
            } else {
                (0.0, 0.0)
            }
        } else if gdist > 1.0 {
            (gdx / gdist, gdy / gdist)
        } else {
            (0.0, 0.0)
        };

        // Local fire/hazard avoidance from real Field temperature
        let cx = (b.x as isize).clamp(1, (self.grid_w - 2) as isize) as usize;
        let cy = (b.y as isize).clamp(1, (self.grid_h - 2) as isize) as usize;
        let cur_temp = self.field.cell_temp_ck(self.field.idx(cx, cy));
        if cur_temp > 38_000 {
            // Step down temperature gradient
            let tl = self.field.cell_temp_ck(self.field.idx(cx - 1, cy));
            let tr = self.field.cell_temp_ck(self.field.idx(cx + 1, cy));
            let tu = self.field.cell_temp_ck(self.field.idx(cx, cy - 1));
            let td = self.field.cell_temp_ck(self.field.idx(cx, cy + 1));
            let gx = (tl - tr) as f32;
            let gy = (tu - td) as f32;
            let gm = gx.hypot(gy).max(1.0);
            mx = (mx * 0.2 + (gx / gm) * 0.9).clamp(-1.0, 1.0);
            my = (my * 0.2 + (gy / gm) * 0.9).clamp(-1.0, 1.0);
            goal_x = b.x + mx * 8.0;
            goal_y = b.y + my * 8.0;
            let _ = (goal_x, goal_y);
        }

        let (ax, ay) = if best_enemy_d < 60.0 {
            let inv = 1.0 / best_enemy_d.max(1e-3);
            (enemy_dx * inv, enemy_dy * inv)
        } else {
            (mx, my)
        };

        let flee_mode = b.brief.active && b.brief.stance == STANCE_FLEE_KITE;
        let dash = if (flee_mode && best_enemy_d < 16.0) || cur_temp > 42_000 || (!flee_mode && best_enemy_d > 14.0 && best_enemy_d < 26.0) {
            0.9
        } else {
            0.05
        };
        let melee = if !flee_mode && best_enemy_d < 4.5 { 0.9 } else { 0.05 };
        let cast_bolt = if !flee_mode && best_enemy_d < 34.0 { 0.85 } else { 0.05 };
        let carve_ice = if self.field.mass[SP_WATER][self.field.idx(cx, cy)] > 400 { 0.8 } else { 0.05 };
        let ability_e = if !flee_mode && best_enemy_d < 18.0 { 0.8 } else { 0.05 };
        let ult_r = if !flee_mode && best_enemy_d < 14.0 && b.hp / b.max_hp < 0.7 { 0.8 } else { 0.05 };

        // Choose element based on thermal bias / oil combo / class
        let (e_fire, e_frost, e_oil) = if b.brief.active && b.brief.thermal_bias.abs() > 0.35 {
            if b.brief.thermal_bias > 0.0 {
                (0.85, 0.10, 0.05)
            } else {
                (0.10, 0.85, 0.05)
            }
        } else if enemy_oiled {
            (0.90, 0.05, 0.05)
        } else {
            match b.class_id % 3 {
                0 => (0.80, 0.10, 0.10),
                1 => (0.10, 0.80, 0.10),
                _ => (0.15, 0.15, 0.70),
            }
        };

        [mx, my, ax, ay, dash, melee, cast_bolt, carve_ice, ability_e, ult_r, e_fire, e_frost, e_oil]
    }

    fn step_one(
        &mut self,
        actions: &[f32],
        scripted_mask: &[u8],
        out_obs: &mut [f32],
        out_rew: &mut [f32],
        out_done: &mut [u8],
        out_pred: &mut [f32],
        out_crop: Option<&mut [f32]>,
    ) {
        let n = self.bots.len();
        let wf = self.grid_w as f32;
        let hf = self.grid_h as f32;
        let dt = 0.12f32; // ~8-10 Hz macro decision cadence per step
        self.tick += 1;

        for r in out_rew.iter_mut() {
            *r = 0.0;
        }
        for d in out_done.iter_mut() {
            *d = 0;
        }

        // Clear transient respawn flags from previous tick before stepping
        for b in &mut self.bots {
            b.just_respawned = false;
        }

        // 1. Apply bot movement & abilities
        for bi in 0..n {
            let act: [f32; RL_ACTION_DIM] = if scripted_mask[bi] != 0 {
                self.scripted_teacher_action(bi)
            } else {
                let s = bi * RL_ACTION_DIM;
                let mut a = [0.0f32; RL_ACTION_DIM];
                a.copy_from_slice(&actions[s..s + RL_ACTION_DIM]);
                a
            };

            let mx = act[0].clamp(-1.0, 1.0);
            let my = act[1].clamp(-1.0, 1.0);
            let m_norm = mx.hypot(my).max(1.0);
            let (mx, my) = (mx / m_norm, my / m_norm);

            let ax_raw = act[2].clamp(-1.0, 1.0);
            let ay_raw = act[3].clamp(-1.0, 1.0);
            let a_norm = ax_raw.hypot(ay_raw);
            let (ax, ay) = if a_norm > 0.1 {
                (ax_raw / a_norm, ay_raw / a_norm)
            } else if m_norm > 0.05 {
                (mx, my)
            } else {
                (1.0, 0.0)
            };

            let elem = if act[10] >= act[11] && act[10] >= act[12] {
                0u8
            } else if act[11] >= act[12] {
                1u8
            } else {
                2u8
            };

            let (_, _, _, base_spd) = Self::class_stats(self.bots[bi].class_id);
            let chilled = self.tick < self.bots[bi].chilled_until;
            let mut spd = base_spd * if chilled { 0.65 } else { 1.0 };

            // Dash
            if act[4] > 0.55 && self.tick >= self.bots[bi].dash_cd && self.bots[bi].stamina >= 25.0 {
                self.bots[bi].stamina -= 25.0;
                self.bots[bi].dash_cd = self.tick + 16;
                spd *= 2.4;
            }

            let cur_x = self.bots[bi].x;
            let cur_y = self.bots[bi].y;
            let nx = (cur_x + mx * spd * dt).clamp(2.0, wf - 3.0);
            let ny = (cur_y + my * spd * dt).clamp(2.0, hf - 3.0);
            let nidx = self.field.idx(nx as usize, ny as usize);
            let blocked = self.field.mass[SP_ROCK][nidx] > 2600;
            if !blocked {
                self.bots[bi].vx = (nx - cur_x) / dt;
                self.bots[bi].vy = (ny - cur_y) / dt;
                self.bots[bi].x = nx;
                self.bots[bi].y = ny;
            } else {
                self.bots[bi].vx = 0.0;
                self.bots[bi].vy = 0.0;
            }

            // Passive regen
            let b = &mut self.bots[bi];
            b.mana = (b.mana + 2.8).min(b.max_mana);
            b.stamina = (b.stamina + 4.5).min(b.max_stamina);
            if b.hp < b.max_hp {
                b.hp = (b.hp + 0.15).min(b.max_hp);
            }

            let bx = b.x;
            let by = b.y;
            let bteam = b.team;

            // Melee attack
            if act[5] > 0.50 && self.tick >= self.bots[bi].melee_cd {
                self.bots[bi].melee_cd = self.tick + 6;
                for oj in 0..n {
                    if oj == bi || self.bots[oj].team == bteam {
                        continue;
                    }
                    let d = (self.bots[oj].x - bx).hypot(self.bots[oj].y - by);
                    if d < 5.0 {
                        let was_alive = self.bots[oj].hp > 0.0;
                        let dmg = 20.0;
                        self.bots[oj].hp -= dmg;
                        out_rew[bi] += 0.22;
                        if was_alive && self.bots[oj].hp <= 0.0 {
                            out_rew[bi] += 0.90;
                        }
                        out_rew[oj] -= 0.08;
                    }
                }
            }

            // Bolt cast (spawns projectile + interacts with conserved Field on impact)
            if act[6] > 0.48 && self.tick >= self.bots[bi].bolt_cd && self.bots[bi].mana >= 12.0 {
                self.bots[bi].mana -= 12.0;
                self.bots[bi].bolt_cd = self.tick + 5;
                if self.bots[bi].brief.active && self.bots[bi].brief.stance == STANCE_FLEE_KITE {
                    out_rew[bi] -= 0.03; // Reward clean disengage rather than cast-spamming during FLEE_KITE
                }
                if self.projectiles.len() < n * 3 {
                    self.projectiles.push(Projectile {
                        owner_idx: bi as u16,
                        team: bteam,
                        elem,
                        x: bx + ax * 1.5,
                        y: by + ay * 1.5,
                        vx: ax * 42.0,
                        vy: ay * 42.0,
                        ttl: 12,
                    });
                }
            }

            // Carve ice / freeze crossing
            if act[7] > 0.55 && self.bots[bi].mana >= 14.0 {
                self.bots[bi].mana -= 14.0;
                self.field.apply_frost_bolt((bx + ax * 3.0) as i32, (by + ay * 3.0) as i32, 2, 0.8);
            }

            // Ability E (elemental AoE into conserved Field!)
            if act[8] > 0.55 && self.tick >= self.bots[bi].ability_e_cd && self.bots[bi].mana >= 28.0 {
                self.bots[bi].mana -= 28.0;
                self.bots[bi].ability_e_cd = self.tick + 40;
                let tx = (bx + ax * 6.0) as i32;
                let ty = (by + ay * 6.0) as i32;
                match elem {
                    0 => self.field.apply_fire_bolt(tx, ty, 3, 1.0),
                    1 => self.field.apply_glacial_nova(tx, ty, 3),
                    _ => self.field.apply_oil_splash(tx, ty, 3, 1.0),
                }
            }

            // Ultimate R
            if act[9] > 0.55 && self.tick >= self.bots[bi].ult_r_cd && self.bots[bi].mana >= 45.0 {
                self.bots[bi].mana -= 45.0;
                self.bots[bi].ult_r_cd = self.tick + 90;
                let tx = (bx + ax * 8.0) as i32;
                let ty = (by + ay * 8.0) as i32;
                if elem == 0 {
                    self.field.apply_meteor_impact(tx, ty, 4);
                } else {
                    let _ = self.field.apply_steam_burst(tx, ty, 4);
                }
            }
        }

        // 2. Step projectiles + elemental combo hits
        let mut i = 0;
        while i < self.projectiles.len() {
            let mut p = self.projectiles[i];
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            let mut remove = p.ttl == 0
                || p.x < 1.0
                || p.x >= wf - 1.0
                || p.y < 1.0
                || p.y >= hf - 1.0;

            if !remove {
                p.ttl -= 1;
                let owner = p.owner_idx as usize;
                for bj in 0..n {
                    if self.bots[bj].team == p.team {
                        continue;
                    }
                    let d = (self.bots[bj].x - p.x).hypot(self.bots[bj].y - p.y);
                    if d < 2.8 {
                        let mut dmg = 16.0f32;
                        let mut combo_bonus = 0.0f32;
                        if p.elem == 0 && self.tick < self.bots[bj].oiled_until {
                            // Oil + Fire ignition combo!
                            dmg += 18.0;
                            combo_bonus = 0.35;
                            self.bots[bj].oiled_until = 0;
                        } else if p.elem == 1 {
                            self.bots[bj].chilled_until = self.tick + 20;
                        } else if p.elem == 2 {
                            self.bots[bj].oiled_until = self.tick + 30;
                        }
                        let was_alive = self.bots[bj].hp > 0.0;
                        self.bots[bj].hp -= dmg;
                        if owner < n {
                            out_rew[owner] += 0.18 + combo_bonus;
                            if was_alive && self.bots[bj].hp <= 0.0 {
                                out_rew[owner] += 0.90;
                            }
                        }
                        out_rew[bj] -= 0.06;
                        match p.elem {
                            0 => self.field.apply_fire_bolt(p.x as i32, p.y as i32, 2, 0.7),
                            1 => self.field.apply_frost_bolt(p.x as i32, p.y as i32, 2, 0.7),
                            _ => self.field.apply_oil_splash(p.x as i32, p.y as i32, 2, 0.7),
                        }
                        remove = true;
                        break;
                    }
                }
            }

            if remove {
                self.projectiles.swap_remove(i);
            } else {
                self.projectiles[i] = p;
                i += 1;
            }
        }

        // 3. Step conserved `field-core::Field` at configured cadence (zero-drift exact rules!)
        if self.tick % self.field_step_interval == 0 {
            self.field.apply_weather_tick(self.weather_kind, self.tick as u64);
            let raw = self.field.refsim_raw_outputs();
            let _ = self.field.project_and_apply(&raw);
        }

        // 4. Environmental fire/hazard damage + Shrines + Golden Relic + Deaths/Respawns
        for s in &mut self.shrines {
            let mut c0 = 0i32;
            let mut c1 = 0i32;
            for (bi, b) in self.bots.iter().enumerate() {
                if (b.x - s.x).hypot(b.y - s.y) < 10.0 {
                    if b.team == 0 {
                        c0 += 1;
                        if s.progress > -0.95 {
                            out_rew[bi] += 0.04;
                        }
                    } else {
                        c1 += 1;
                        if s.progress < 0.95 {
                            out_rew[bi] += 0.04;
                        }
                    }
                }
            }
            let diff = (c1 - c0) as f32;
            s.progress = (s.progress + diff * 0.03).clamp(-1.0, 1.0);
            if s.progress <= -0.8 {
                self.team_score[0] += 0.15;
            } else if s.progress >= 0.8 {
                self.team_score[1] += 0.15;
            }
        }

        // Relic pickup / delivery
        if let Some(ci) = self.relic_carrier {
            self.relic_x = self.bots[ci].x;
            self.relic_y = self.bots[ci].y;
            let home_x = if self.bots[ci].team == 0 { wf * 0.14 } else { wf * 0.86 };
            if (self.relic_x - home_x).abs() < 9.0 {
                let team = self.bots[ci].team as usize;
                self.team_score[team] += 12.0;
                out_rew[ci] += 1.5;
                self.bots[ci].carrying_relic = false;
                self.relic_carrier = None;
                self.relic_x = wf * 0.50;
                self.relic_y = hf * 0.50;
            }
        } else {
            for bi in 0..n {
                if (self.bots[bi].x - self.relic_x).hypot(self.bots[bi].y - self.relic_y) < 4.5 {
                    self.bots[bi].carrying_relic = true;
                    self.relic_carrier = Some(bi);
                    out_rew[bi] += 0.40;
                    break;
                }
            }
        }

        // Fire damage + death/respawn
        for bi in 0..n {
            let cx = (self.bots[bi].x as usize).min(self.grid_w - 1);
            let cy = (self.bots[bi].y as usize).min(self.grid_h - 1);
            let temp = self.field.cell_temp_ck(self.field.idx(cx, cy));
            if temp > 45_000 {
                self.bots[bi].hp -= 4.5;
                out_rew[bi] -= 0.08;
            }
            if self.bots[bi].hp <= 0.0 {
                let killer_team = 1usize - (self.bots[bi].team as usize);
                self.team_kills[killer_team] += 1.0;
                self.team_score[killer_team] += 2.0;
                if self.bots[bi].carrying_relic {
                    self.bots[bi].carrying_relic = false;
                    self.relic_carrier = None;
                }
                let team = self.bots[bi].team;
                let (sx, sy) = self.spawn_xy(team);
                let b = &mut self.bots[bi];
                b.x = sx;
                b.y = sy;
                b.vx = 0.0;
                b.vy = 0.0;
                b.hp = b.max_hp;
                b.mana = b.max_mana;
                b.stamina = b.max_stamina;
                b.oiled_until = 0;
                b.chilled_until = 0;
                b.just_respawned = true;
                b.prev_target_dist = (b.brief.target_x - sx).hypot(b.brief.target_y - sy);
                out_rew[bi] -= 0.60;
                out_done[bi] = 1;
            }
        }

        // Episode boundary soft reset
        let ep_done = self.tick >= self.episode_len;
        if ep_done {
            for d in out_done.iter_mut() {
                *d = 1;
            }
            let next_seed = self.rng.next_u64();
            self.reset_inner(n, next_seed);
        }

        // 5. Compute NMMO 2.0 dense 0..1 progress predicates + shape reward + pack 184-float obs
        self.write_obs_and_preds(out_obs, out_rew, out_pred, out_crop);
    }

    fn write_obs_and_preds(
        &mut self,
        out_obs: &mut [f32],
        out_rew: &mut [f32],
        out_pred: &mut [f32],
        mut out_crop: Option<&mut [f32]>,
    ) {
        let n = self.bots.len();
        let wf = self.grid_w as f32;
        let hf = self.grid_h as f32;

        for bi in 0..n {
            let b = self.bots[bi];
            let base = bi * RL_OBS_DIM;
            for v in &mut out_obs[base..base + RL_OBS_DIM] {
                *v = 0.0;
            }

            // Self features [0..24]
            out_obs[base] = (b.x / wf).clamp(0.0, 1.0);
            out_obs[base + 1] = (b.y / hf).clamp(0.0, 1.0);
            out_obs[base + 2] = (b.z / 3.0).clamp(0.0, 1.0);
            out_obs[base + 3] = (b.vx / 40.0).clamp(-1.0, 1.0);
            out_obs[base + 4] = (b.vy / 40.0).clamp(-1.0, 1.0);
            let hp_frac = (b.hp / b.max_hp.max(1.0)).clamp(0.0, 1.0);
            out_obs[base + 5] = hp_frac;
            out_obs[base + 6] = (b.mana / b.max_mana.max(1.0)).clamp(0.0, 1.0);
            out_obs[base + 7] = (b.stamina / b.max_stamina.max(1.0)).clamp(0.0, 1.0);
            out_obs[base + 8 + (b.class_id as usize % 3)] = 1.0;
            out_obs[base + 11] = b.team as f32;
            out_obs[base + 12] = if b.carrying_relic { 1.0 } else { 0.0 };
            out_obs[base + 13] = b.level as f32 / 5.0;
            out_obs[base + 14] = if b.class_id == 3 { 1.0 } else { 0.0 };
            out_obs[base + 15] = (b.squad_id as f32 / 8.0).clamp(0.0, 1.0);
            out_obs[base + 16] = if b.class_id == 4 { 1.0 } else { 0.0 };
            out_obs[base + 20] = (b.dash_cd.saturating_sub(self.tick) as f32 / 32.0).clamp(0.0, 1.0);
            out_obs[base + 21] = (b.ability_e_cd.saturating_sub(self.tick) as f32 / 120.0).clamp(0.0, 1.0);
            out_obs[base + 22] = (b.ult_r_cd.saturating_sub(self.tick) as f32 / 360.0).clamp(0.0, 1.0);
            out_obs[base + 23] = self.weather_kind as f32 / 3.0;

            // Pack nearest 6 ENEMIES + nearest 2 ALLIES into tokens 0..8 so 60-bot crowds never blind
            // cross-attention to enemies behind nearby teammates (Suárez 60-bot density fix!).
            self.dist_scratch.clear();
            let mut nearest_enemy_dist = 120.0f32;
            let mut top_enemies: [(f32, usize); 6] = [(f32::MAX, usize::MAX); 6];
            let mut top_allies: [(f32, usize); 2] = [(f32::MAX, usize::MAX); 2];
            for (oj, o) in self.bots.iter().enumerate() {
                if oj == bi {
                    continue;
                }
                let d = (o.x - b.x).hypot(o.y - b.y);
                if o.team != b.team {
                    if d < nearest_enemy_dist {
                        nearest_enemy_dist = d;
                    }
                    if d < top_enemies[5].0 {
                        top_enemies[5] = (d, oj);
                        let mut k = 5;
                        while k > 0 && top_enemies[k].0 < top_enemies[k - 1].0 {
                            top_enemies.swap(k, k - 1);
                            k -= 1;
                        }
                    }
                } else if d < top_allies[1].0 {
                    top_allies[1] = (d, oj);
                    if top_allies[1].0 < top_allies[0].0 {
                        top_allies.swap(0, 1);
                    }
                }
            }
            for &(d, oj) in &top_enemies {
                if oj != usize::MAX {
                    self.dist_scratch.push((d, oj));
                }
            }
            for &(d, oj) in &top_allies {
                if oj != usize::MAX && self.dist_scratch.len() < 8 {
                    self.dist_scratch.push((d, oj));
                }
            }

            for (ti, &(d, oj)) in self.dist_scratch.iter().enumerate() {
                let o = &self.bots[oj];
                let tb = base + RL_SELF_DIM + ti * RL_TOKEN_DIM;
                let d_safe = d.max(1e-3);
                out_obs[tb] = ((o.x - b.x) / 80.0).clamp(-1.0, 1.0);
                out_obs[tb + 1] = ((o.y - b.y) / 80.0).clamp(-1.0, 1.0);
                out_obs[tb + 2] = 0.0;
                out_obs[tb + 3] = if o.team != b.team { 1.0 } else { 0.0 };
                out_obs[tb + 4] = (o.hp / o.max_hp.max(1.0)).clamp(0.0, 1.0);
                out_obs[tb + 5] = if o.carrying_relic { 1.0 } else { 0.0 };
                out_obs[tb + 6] = (o.class_id as f32) / 2.0;
                out_obs[tb + 7] = (d / 100.0).min(1.0);
                out_obs[tb + 8] = (o.x - b.x) / d_safe;
                out_obs[tb + 9] = (o.y - b.y) / d_safe;
                out_obs[tb + 11] = 1.0;
            }

            // 3 Shrines -> tokens 8..11
            let mut best_shrine_pred = 0.0f32;
            for (si, s) in self.shrines.iter().enumerate() {
                let tb = base + RL_SELF_DIM + (8 + si) * RL_TOKEN_DIM;
                let need = if b.team == 0 { s.progress > -0.8 } else { s.progress < 0.8 };
                out_obs[tb] = ((s.x - b.x) / 100.0).clamp(-1.0, 1.0);
                out_obs[tb + 1] = ((s.y - b.y) / 100.0).clamp(-1.0, 1.0);
                out_obs[tb + 3] = if need { 1.0 } else { 0.0 };
                out_obs[tb + 4] = s.progress;
                out_obs[tb + 11] = 1.0;
                let sd = (s.x - b.x).hypot(s.y - b.y);
                let team_ctrl = if b.team == 0 { (-s.progress + 1.0) * 0.5 } else { (s.progress + 1.0) * 0.5 };
                let p_shrine = (1.0 - (sd / 40.0).min(1.0)) * 0.6 + team_ctrl * 0.4;
                if p_shrine > best_shrine_pred {
                    best_shrine_pred = p_shrine;
                }
            }

            // Golden Relic -> token 11
            let tr = base + RL_SELF_DIM + 11 * RL_TOKEN_DIM;
            out_obs[tr] = ((self.relic_x - b.x) / 100.0).clamp(-1.0, 1.0);
            out_obs[tr + 1] = ((self.relic_y - b.y) / 100.0).clamp(-1.0, 1.0);
            out_obs[tr + 3] = if b.carrying_relic { 1.0 } else { 0.0 };
            out_obs[tr + 11] = 1.0;

            // Brief conditioning tail [168..184]
            let bb = base + RL_BASE_FEAT_DIM;
            let obj = b.brief.objective.min(4) as usize;
            let st = b.brief.stance.min(2) as usize;
            out_obs[bb + obj] = 1.0;
            out_obs[bb + 5 + st] = 1.0;
            let tdx = b.brief.target_x - b.x;
            let tdy = b.brief.target_y - b.y;
            let tdist = tdx.hypot(tdy);
            out_obs[bb + 8] = (tdx / 100.0).clamp(-1.0, 1.0);
            out_obs[bb + 9] = (tdy / 100.0).clamp(-1.0, 1.0);
            out_obs[bb + 10] = (tdist / 100.0).min(1.0);
            out_obs[bb + 11] = b.brief.risk_budget.clamp(0.0, 1.0);
            out_obs[bb + 12] = (b.class_id.min(4) as f32) / 4.0;
            out_obs[bb + 13] = if b.brief.active { 1.0 } else { 0.0 };
            out_obs[bb + 14] = b.brief.thermal_bias.clamp(-1.0, 1.0);
            let slot_key = (b.id as f32) + 1.0;
            out_obs[bb + 15] = if b.just_respawned { -slot_key } else { slot_key };

            // NMMO 2.0 Dense 0..1 Progress Predicates [6]
            let cx = (b.x as usize).min(self.grid_w - 1);
            let cy = (b.y as usize).min(self.grid_h - 1);
            let cur_temp = self.field.cell_temp_ck(self.field.idx(cx, cy));
            let hazard_pen = if cur_temp > 38_000 { 1.0 } else { 0.0 };

            let close_delta = (b.prev_target_dist - tdist).clamp(-4.0, 4.0) / 4.0;
            self.bots[bi].prev_target_dist = tdist;

            let pred_close = ((1.0 - (tdist / 64.0).min(1.0)) * 0.7 + (close_delta * 0.5 + 0.5) * 0.3).clamp(0.0, 1.0);
            let pred_stance = match b.brief.stance {
                STANCE_FLEE_KITE => ((nearest_enemy_dist - 8.0) / 22.0).clamp(0.0, 1.0),
                STANCE_HOLD => {
                    let at_post = 1.0 - (tdist / 20.0).min(1.0);
                    let safe_band = if (8.0..=30.0).contains(&nearest_enemy_dist) { 1.0 } else { 0.5 };
                    (at_post * 0.65 + safe_band * 0.35).clamp(0.0, 1.0)
                }
                _ => {
                    // COMMIT: engage target or enemies aggressively
                    let engage = 1.0 - (nearest_enemy_dist / 36.0).min(1.0);
                    (engage * 0.5 + pred_close * 0.5).clamp(0.0, 1.0)
                }
            };
            let min_safe_hp = (1.0 - b.brief.risk_budget) * 0.65;
            let pred_risk = (((hp_frac - min_safe_hp) / (1.0 - min_safe_hp).max(0.2)).clamp(0.0, 1.0)
                * (1.0 - 0.6 * hazard_pen))
                .clamp(0.0, 1.0);
            let relic_d = (self.relic_x - b.x).hypot(self.relic_y - b.y);
            let pred_relic = if b.carrying_relic {
                1.0
            } else {
                (1.0 - (relic_d / 64.0).min(1.0)).clamp(0.0, 1.0)
            };
            let pred_combat = (out_rew[bi].max(0.0) * 2.0).clamp(0.0, 1.0);

            let pb = bi * RL_PRED_DIM;
            out_pred[pb] = pred_close;
            out_pred[pb + 1] = pred_stance;
            out_pred[pb + 2] = pred_risk;
            out_pred[pb + 3] = best_shrine_pred.clamp(0.0, 1.0);
            out_pred[pb + 4] = pred_relic;
            out_pred[pb + 5] = pred_combat;

            // Add dense brief-obedience + tactical shaping to scalar reward
            if b.brief.active {
                let brief_bonus = match b.brief.stance {
                    STANCE_FLEE_KITE => 0.08 * pred_stance + 0.06 * pred_risk,
                    STANCE_HOLD => 0.07 * pred_stance + 0.04 * pred_close + 0.03 * pred_risk,
                    _ => 0.06 * pred_close + 0.05 * pred_stance + 0.03 * close_delta,
                };
                out_rew[bi] += brief_bonus;
            } else {
                out_rew[bi] += 0.025 * best_shrine_pred + 0.015 * pred_relic + 0.01 * pred_risk;
            }

            // Optional local 8-channel 16x16 physical crop for pure-Rust fast mode
            if let Some(ref mut crops) = out_crop {
                let cb = bi * RL_CROP_STRIDE;
                let half = (RL_CROP_SIZE as isize) / 2;
                let bx_i = b.x as isize;
                let by_i = b.y as isize;
                for py in 0..RL_CROP_SIZE {
                    let wy = (by_i + (py as isize - half) * 2).clamp(0, (self.grid_h - 1) as isize) as usize;
                    for px in 0..RL_CROP_SIZE {
                        let wx = (bx_i + (px as isize - half) * 2).clamp(0, (self.grid_w - 1) as isize) as usize;
                        let idx = self.field.idx(wx, wy);
                        let p_off = py * RL_CROP_SIZE + px;
                        crops[cb + p_off] = (self.field.mass[SP_WATER][idx] as f32 / 2000.0).min(1.0);
                        crops[cb + 256 + p_off] = (self.field.mass[SP_ICE][idx] as f32 / 2000.0).min(1.0);
                        crops[cb + 512 + p_off] = ((self.field.cell_temp_ck(idx) - T_AMB) as f32 / 30_000.0).clamp(-1.0, 2.0);
                        crops[cb + 768 + p_off] = (self.field.mass[SP_OIL][idx] as f32 / 2000.0).min(1.0);
                        crops[cb + 1024 + p_off] = (self.field.mass[SP_ROCK][idx] as f32 / 3000.0).min(1.0);
                        crops[cb + 1280 + p_off] = ((self.field.mass[SP_WOOD][idx] as f32 + self.field.mass[SP_PLANT][idx] as f32) / 2000.0).min(1.0);
                        crops[cb + 1536 + p_off] = (self.field.mass[SP_SMOKE][idx] as f32 / 1500.0).min(1.0);
                        crops[cb + 1792 + p_off] = self.field.elevation[idx] as f32 / 255.0;
                    }
                }
            }
        }
    }
}

struct OutputSlot {
    obs: Vec<f32>,
    rew: Vec<f32>,
    done: Vec<u8>,
    pred: Vec<f32>,
    crop: Vec<f32>,
}

impl OutputSlot {
    fn new(total_bots: usize, enable_crops: bool) -> Self {
        Self {
            obs: vec![0.0; total_bots * RL_OBS_DIM],
            rew: vec![0.0; total_bots],
            done: vec![0; total_bots],
            pred: vec![0.0; total_bots * RL_PRED_DIM],
            crop: if enable_crops {
                vec![0.0; total_bots * RL_CROP_STRIDE]
            } else {
                Vec::new()
            },
        }
    }
}

struct SharedPoolState {
    generation: u64,
    completed_workers: usize,
    shutdown: bool,
    // Raw pointers to the active job buffers (valid only while generation is running)
    actions_ptr: usize,
    mask_ptr: usize,
    obs_ptr: usize,
    rew_ptr: usize,
    done_ptr: usize,
    pred_ptr: usize,
    crop_ptr: usize,
    enable_crops: bool,
}

struct WorkerSlice {
    arenas: Mutex<Vec<ArenaInstance>>,
}

/// Multi-threaded, zero-alloc-per-step, double-buffered Rust Vectorized RL Arena.
pub struct RustVecArena {
    pub num_arenas: usize,
    pub bots_per_arena: usize,
    pub grid_w: usize,
    pub grid_h: usize,
    pub enable_crops: bool,
    num_workers: usize,
    slices: Arc<Vec<WorkerSlice>>,
    arena_ranges: Vec<(usize, usize)>,
    pool_state: Arc<(Mutex<SharedPoolState>, Condvar, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    slots: [OutputSlot; 2],
    front_slot: usize,
    actions_staged: Vec<f32>,
    scripted_mask: Vec<u8>,
    async_in_flight: AtomicBool,
    total_steps: AtomicU64,
}

impl RustVecArena {
    pub fn new(
        num_arenas: usize,
        bots_per_arena: usize,
        grid_w: usize,
        grid_h: usize,
        seed: u64,
        num_threads: usize,
        field_step_interval: u32,
        enable_crops: bool,
    ) -> Self {
        let num_arenas = num_arenas.clamp(1, 64);
        let bots_per_arena = bots_per_arena.clamp(2, 128);
        let grid_w = grid_w.clamp(32, 256);
        let grid_h = grid_h.clamp(32, 256);
        let auto_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let num_workers = if num_threads == 0 {
            auto_threads.min(num_arenas)
        } else {
            num_threads.clamp(1, num_arenas)
        };

        let mut arena_ranges = Vec::with_capacity(num_workers);
        let mut slice_vecs = Vec::with_capacity(num_workers);
        let base = num_arenas / num_workers;
        let rem = num_arenas % num_workers;
        let mut cur = 0usize;
        for w in 0..num_workers {
            let count = base + if w < rem { 1 } else { 0 };
            let start = cur;
            let end = cur + count;
            arena_ranges.push((start, end));
            let mut local = Vec::with_capacity(count);
            for a_id in start..end {
                local.push(ArenaInstance::new(
                    a_id,
                    bots_per_arena,
                    grid_w,
                    grid_h,
                    seed,
                    field_step_interval,
                ));
            }
            slice_vecs.push(WorkerSlice {
                arenas: Mutex::new(local),
            });
            cur = end;
        }

        let slices = Arc::new(slice_vecs);
        let pool_state = Arc::new((
            Mutex::new(SharedPoolState {
                generation: 0,
                completed_workers: 0,
                shutdown: false,
                actions_ptr: 0,
                mask_ptr: 0,
                obs_ptr: 0,
                rew_ptr: 0,
                done_ptr: 0,
                pred_ptr: 0,
                crop_ptr: 0,
                enable_crops,
            }),
            Condvar::new(),
            Condvar::new(),
        ));

        let mut workers = Vec::with_capacity(num_workers);
        for w_idx in 0..num_workers {
            let slices_c = Arc::clone(&slices);
            let pool_c = Arc::clone(&pool_state);
            let (a_start, a_end) = arena_ranges[w_idx];
            let handle = thread::Builder::new()
                .name(format!("fm-rl-worker-{w_idx}"))
                .spawn(move || {
                    let mut seen_gen = 0u64;
                    loop {
                        let (
                            act_ptr,
                            mask_ptr,
                            obs_ptr,
                            rew_ptr,
                            done_ptr,
                            pred_ptr,
                            crop_ptr,
                            crops_on,
                        ) = {
                            let (lock, cv_start, _) = &*pool_c;
                            let mut st = lock.lock().unwrap();
                            while !st.shutdown && st.generation == seen_gen {
                                st = cv_start.wait(st).unwrap();
                            }
                            if st.shutdown {
                                break;
                            }
                            seen_gen = st.generation;
                            (
                                st.actions_ptr,
                                st.mask_ptr,
                                st.obs_ptr,
                                st.rew_ptr,
                                st.done_ptr,
                                st.pred_ptr,
                                st.crop_ptr,
                                st.enable_crops,
                            )
                        };

                        let mut arenas = slices_c[w_idx].arenas.lock().unwrap();
                        for (local_i, arena) in arenas.iter_mut().enumerate() {
                            let a_id = a_start + local_i;
                            debug_assert!(a_id < a_end);
                            let bot_off = a_id * bots_per_arena;
                            unsafe {
                                let act_s = std::slice::from_raw_parts(
                                    (act_ptr as *const f32).add(bot_off * RL_ACTION_DIM),
                                    bots_per_arena * RL_ACTION_DIM,
                                );
                                let mask_s = std::slice::from_raw_parts(
                                    (mask_ptr as *const u8).add(bot_off),
                                    bots_per_arena,
                                );
                                let obs_s = std::slice::from_raw_parts_mut(
                                    (obs_ptr as *mut f32).add(bot_off * RL_OBS_DIM),
                                    bots_per_arena * RL_OBS_DIM,
                                );
                                let rew_s = std::slice::from_raw_parts_mut(
                                    (rew_ptr as *mut f32).add(bot_off),
                                    bots_per_arena,
                                );
                                let done_s = std::slice::from_raw_parts_mut(
                                    (done_ptr as *mut u8).add(bot_off),
                                    bots_per_arena,
                                );
                                let pred_s = std::slice::from_raw_parts_mut(
                                    (pred_ptr as *mut f32).add(bot_off * RL_PRED_DIM),
                                    bots_per_arena * RL_PRED_DIM,
                                );
                                let crop_opt = if crops_on && crop_ptr != 0 {
                                    Some(std::slice::from_raw_parts_mut(
                                        (crop_ptr as *mut f32).add(bot_off * RL_CROP_STRIDE),
                                        bots_per_arena * RL_CROP_STRIDE,
                                    ))
                                } else {
                                    None
                                };
                                arena.step_one(
                                    act_s, mask_s, obs_s, rew_s, done_s, pred_s, crop_opt,
                                );
                            }
                        }
                        drop(arenas);

                        let (lock, _, cv_done) = &*pool_c;
                        let mut st = lock.lock().unwrap();
                        st.completed_workers += 1;
                        if st.completed_workers == num_workers {
                            cv_done.notify_all();
                        }
                    }
                })
                .expect("spawn fm-rl-worker");
            workers.push(handle);
        }

        let total_bots = num_arenas * bots_per_arena;
        let mut out = Self {
            num_arenas,
            bots_per_arena,
            grid_w,
            grid_h,
            enable_crops,
            num_workers,
            slices,
            arena_ranges,
            pool_state,
            workers,
            slots: [
                OutputSlot::new(total_bots, enable_crops),
                OutputSlot::new(total_bots, enable_crops),
            ],
            front_slot: 0,
            actions_staged: vec![0.0; total_bots * RL_ACTION_DIM],
            scripted_mask: vec![0u8; total_bots],
            async_in_flight: AtomicBool::new(false),
            total_steps: AtomicU64::new(0),
        };
        out.populate_initial_obs();
        out
    }

    fn populate_initial_obs(&mut self) {
        let slot = &mut self.slots[self.front_slot];
        for (w_idx, &(a_start, _)) in self.arena_ranges.iter().enumerate() {
            let mut arenas = self.slices[w_idx].arenas.lock().unwrap();
            for (local_i, arena) in arenas.iter_mut().enumerate() {
                let bot_off = (a_start + local_i) * self.bots_per_arena;
                let obs_s = &mut slot.obs[bot_off * RL_OBS_DIM..(bot_off + self.bots_per_arena) * RL_OBS_DIM];
                let rew_s = &mut slot.rew[bot_off..bot_off + self.bots_per_arena];
                let pred_s = &mut slot.pred[bot_off * RL_PRED_DIM..(bot_off + self.bots_per_arena) * RL_PRED_DIM];
                let crop_opt = if self.enable_crops {
                    Some(&mut slot.crop[bot_off * RL_CROP_STRIDE..(bot_off + self.bots_per_arena) * RL_CROP_STRIDE])
                } else {
                    None
                };
                arena.write_obs_and_preds(obs_s, rew_s, pred_s, crop_opt);
            }
        }
    }

    pub fn wait_if_in_flight(&mut self) {
        if self.async_in_flight.swap(false, Ordering::AcqRel) {
            let (lock, _, cv_done) = &*self.pool_state;
            let mut st = lock.lock().unwrap();
            while st.completed_workers < self.num_workers {
                st = cv_done.wait(st).unwrap();
            }
            self.front_slot ^= 1;
            self.total_steps.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn reset(&mut self, seed: u64) {
        self.wait_if_in_flight();
        for w_idx in 0..self.num_workers {
            let mut arenas = self.slices[w_idx].arenas.lock().unwrap();
            for arena in arenas.iter_mut() {
                arena.reset_inner(self.bots_per_arena, seed);
            }
        }
        self.populate_initial_obs();
    }

    pub fn set_scripted_mask(&mut self, mask: &[u8]) {
        self.wait_if_in_flight();
        let n = mask.len().min(self.scripted_mask.len());
        self.scripted_mask[..n].copy_from_slice(&mask[..n]);
    }

    pub fn set_briefs(&mut self, briefs: &[f32]) {
        self.wait_if_in_flight();
        let total = self.num_arenas * self.bots_per_arena;
        if briefs.len() < total * RL_BRIEF_IN_DIM {
            return;
        }
        for (w_idx, &(a_start, _)) in self.arena_ranges.iter().enumerate() {
            let mut arenas = self.slices[w_idx].arenas.lock().unwrap();
            for (local_i, arena) in arenas.iter_mut().enumerate() {
                let bot_off = (a_start + local_i) * self.bots_per_arena;
                for (bi, b) in arena.bots.iter_mut().enumerate() {
                    let s = (bot_off + bi) * RL_BRIEF_IN_DIM;
                    b.brief = BotBrief {
                        objective: (briefs[s].round().clamp(0.0, 4.0)) as u8,
                        stance: (briefs[s + 1].round().clamp(0.0, 2.0)) as u8,
                        target_x: briefs[s + 2].clamp(1.0, (self.grid_w - 2) as f32),
                        target_y: briefs[s + 3].clamp(1.0, (self.grid_h - 2) as f32),
                        risk_budget: briefs[s + 4].clamp(0.0, 1.0),
                        active: briefs[s + 5] > 0.5,
                        thermal_bias: briefs[s + 6].clamp(-1.0, 1.0),
                    };
                    b.prev_target_dist = (b.brief.target_x - b.x).hypot(b.brief.target_y - b.y);
                }
            }
        }
        self.populate_initial_obs();
    }

    pub fn sample_random_briefs(&mut self, active_prob: f32) {
        self.wait_if_in_flight();
        for w_idx in 0..self.num_workers {
            let mut arenas = self.slices[w_idx].arenas.lock().unwrap();
            for arena in arenas.iter_mut() {
                arena.sample_random_briefs(active_prob);
            }
        }
        self.populate_initial_obs();
    }

    pub fn step_async_begin(&mut self, actions: &[f32]) {
        self.wait_if_in_flight();
        let n = actions.len().min(self.actions_staged.len());
        self.actions_staged[..n].copy_from_slice(&actions[..n]);
        let back_slot = self.front_slot ^ 1;
        let slot = &mut self.slots[back_slot];

        let (lock, cv_start, _) = &*self.pool_state;
        let mut st = lock.lock().unwrap();
        st.actions_ptr = self.actions_staged.as_ptr() as usize;
        st.mask_ptr = self.scripted_mask.as_ptr() as usize;
        st.obs_ptr = slot.obs.as_mut_ptr() as usize;
        st.rew_ptr = slot.rew.as_mut_ptr() as usize;
        st.done_ptr = slot.done.as_mut_ptr() as usize;
        st.pred_ptr = slot.pred.as_mut_ptr() as usize;
        st.crop_ptr = if self.enable_crops {
            slot.crop.as_mut_ptr() as usize
        } else {
            0
        };
        st.enable_crops = self.enable_crops;
        st.completed_workers = 0;
        st.generation += 1;
        self.async_in_flight.store(true, Ordering::Release);
        cv_start.notify_all();
    }

    pub fn step_async_wait(&mut self) {
        self.wait_if_in_flight();
    }

    pub fn step_sync(&mut self, actions: &[f32]) {
        self.step_async_begin(actions);
        self.step_async_wait();
    }

    #[inline]
    pub fn obs(&self) -> &[f32] {
        &self.slots[self.front_slot].obs
    }

    #[inline]
    pub fn rew(&self) -> &[f32] {
        &self.slots[self.front_slot].rew
    }

    #[inline]
    pub fn done(&self) -> &[u8] {
        &self.slots[self.front_slot].done
    }

    #[inline]
    pub fn pred(&self) -> &[f32] {
        &self.slots[self.front_slot].pred
    }

    #[inline]
    pub fn crop(&self) -> &[f32] {
        &self.slots[self.front_slot].crop
    }

    pub fn export_field_tensors(&mut self, out: &mut [f32]) -> bool {
        self.wait_if_in_flight();
        let stride = INPUT_CHANNELS * self.grid_w * self.grid_h;
        if out.len() < self.num_arenas * stride {
            return false;
        }
        for (w_idx, &(a_start, _)) in self.arena_ranges.iter().enumerate() {
            let arenas = self.slices[w_idx].arenas.lock().unwrap();
            for (local_i, arena) in arenas.iter().enumerate() {
                let off = (a_start + local_i) * stride;
                let enc = arena.field.encode_input_tensor();
                out[off..off + stride].copy_from_slice(&enc);
            }
        }
        true
    }

    pub fn get_team_scores(&mut self, out: &mut [f32]) {
        self.wait_if_in_flight();
        if out.len() < self.num_arenas * 4 {
            return;
        }
        for (w_idx, &(a_start, _)) in self.arena_ranges.iter().enumerate() {
            let arenas = self.slices[w_idx].arenas.lock().unwrap();
            for (local_i, arena) in arenas.iter().enumerate() {
                let off = (a_start + local_i) * 4;
                out[off] = arena.team_score[0];
                out[off + 1] = arena.team_score[1];
                out[off + 2] = arena.team_kills[0];
                out[off + 3] = arena.team_kills[1];
            }
        }
    }
}

impl Drop for RustVecArena {
    fn drop(&mut self) {
        self.wait_if_in_flight();
        {
            let (lock, cv_start, _) = &*self.pool_state;
            let mut st = lock.lock().unwrap();
            st.shutdown = true;
            cv_start.notify_all();
        }
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

// ============================================================================
// C ABI Exports (Rust 2024 `#[unsafe(no_mangle)]`) for zero-copy Python ctypes
// ============================================================================

#[unsafe(no_mangle)]
pub extern "C" fn fm_arena_create(
    num_arenas: u32,
    bots_per_arena: u32,
    grid_w: u32,
    grid_h: u32,
    seed: u64,
    num_threads: u32,
    field_step_interval: u32,
    enable_crops: u8,
) -> *mut RustVecArena {
    let arena = RustVecArena::new(
        num_arenas as usize,
        bots_per_arena as usize,
        grid_w as usize,
        grid_h as usize,
        seed,
        num_threads as usize,
        field_step_interval,
        enable_crops != 0,
    );
    Box::into_raw(Box::new(arena))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_destroy(ptr: *mut RustVecArena) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)) };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_reset(ptr: *mut RustVecArena, seed: u64) {
    if let Some(a) = unsafe { ptr.as_mut() } {
        a.reset(seed);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_set_scripted_mask(
    ptr: *mut RustVecArena,
    mask_ptr: *const u8,
    len: usize,
) {
    if let (Some(a), false) = (unsafe { ptr.as_mut() }, mask_ptr.is_null()) {
        let s = unsafe { std::slice::from_raw_parts(mask_ptr, len) };
        a.set_scripted_mask(s);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_set_briefs(
    ptr: *mut RustVecArena,
    briefs_ptr: *const f32,
    len: usize,
) {
    if let (Some(a), false) = (unsafe { ptr.as_mut() }, briefs_ptr.is_null()) {
        let s = unsafe { std::slice::from_raw_parts(briefs_ptr, len) };
        a.set_briefs(s);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_sample_random_briefs(ptr: *mut RustVecArena, active_prob: f32) {
    if let Some(a) = unsafe { ptr.as_mut() } {
        a.sample_random_briefs(active_prob);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_step(
    ptr: *mut RustVecArena,
    actions_ptr: *const f32,
    actions_len: usize,
) {
    if let (Some(a), false) = (unsafe { ptr.as_mut() }, actions_ptr.is_null()) {
        let s = unsafe { std::slice::from_raw_parts(actions_ptr, actions_len) };
        a.step_sync(s);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_step_async_begin(
    ptr: *mut RustVecArena,
    actions_ptr: *const f32,
    actions_len: usize,
) {
    if let (Some(a), false) = (unsafe { ptr.as_mut() }, actions_ptr.is_null()) {
        let s = unsafe { std::slice::from_raw_parts(actions_ptr, actions_len) };
        a.step_async_begin(s);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_step_async_wait(ptr: *mut RustVecArena) {
    if let Some(a) = unsafe { ptr.as_mut() } {
        a.step_async_wait();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_obs_ptr(ptr: *const RustVecArena) -> *const f32 {
    unsafe { ptr.as_ref() }.map_or(std::ptr::null(), |a| a.obs().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_rew_ptr(ptr: *const RustVecArena) -> *const f32 {
    unsafe { ptr.as_ref() }.map_or(std::ptr::null(), |a| a.rew().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_done_ptr(ptr: *const RustVecArena) -> *const u8 {
    unsafe { ptr.as_ref() }.map_or(std::ptr::null(), |a| a.done().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_pred_ptr(ptr: *const RustVecArena) -> *const f32 {
    unsafe { ptr.as_ref() }.map_or(std::ptr::null(), |a| a.pred().as_ptr())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_crop_ptr(ptr: *const RustVecArena) -> *const f32 {
    unsafe { ptr.as_ref() }.map_or(std::ptr::null(), |a| {
        if a.enable_crops {
            a.crop().as_ptr()
        } else {
            std::ptr::null()
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_export_field_tensors(
    ptr: *mut RustVecArena,
    out_ptr: *mut f32,
    out_len: usize,
) -> i32 {
    if let (Some(a), false) = (unsafe { ptr.as_mut() }, out_ptr.is_null()) {
        let s = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
        if a.export_field_tensors(s) { 1 } else { 0 }
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn fm_arena_get_team_scores(
    ptr: *mut RustVecArena,
    out_ptr: *mut f32,
    out_len: usize,
) {
    if let (Some(a), false) = (unsafe { ptr.as_mut() }, out_ptr.is_null()) {
        let s = unsafe { std::slice::from_raw_parts_mut(out_ptr, out_len) };
        a.get_team_scores(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_60_bot_arena_obs_and_zero_nans() {
        let mut arena = RustVecArena::new(2, 60, 64, 64, 42, 2, 4, false);
        assert_eq!(arena.obs().len(), 2 * 60 * RL_OBS_DIM);
        assert_eq!(arena.rew().len(), 2 * 60);
        assert_eq!(arena.pred().len(), 2 * 60 * RL_PRED_DIM);
        // Initial spawn sets negative bot_slot_key to signal GRU reset
        assert!(arena.obs()[RL_BASE_FEAT_DIM + 15] < 0.0);

        let mask = vec![1u8; 2 * 60];
        arena.set_scripted_mask(&mask);
        let actions = vec![0.0f32; 2 * 60 * RL_ACTION_DIM];
        for _ in 0..8 {
            arena.step_sync(&actions);
        }
        assert!(arena.obs().iter().all(|v| v.is_finite()));
        assert!(arena.rew().iter().all(|v| v.is_finite()));
        assert!(arena.pred().iter().all(|v| (0.0..=1.0).contains(v)));
        // After stepping without death, bot_slot_key flips positive
        let positive_keys = (0..120)
            .filter(|&i| arena.obs()[i * RL_OBS_DIM + RL_BASE_FEAT_DIM + 15] > 0.0)
            .count();
        assert!(positive_keys > 100);
    }

    #[test]
    fn test_async_double_buffer_matches_sync() {
        let mut sync_a = RustVecArena::new(4, 12, 64, 64, 1337, 2, 4, false);
        let mut async_a = RustVecArena::new(4, 12, 64, 64, 1337, 2, 4, false);
        let mask = vec![1u8; 4 * 12];
        sync_a.set_scripted_mask(&mask);
        async_a.set_scripted_mask(&mask);
        let act = vec![0.0f32; 4 * 12 * RL_ACTION_DIM];

        for _ in 0..5 {
            sync_a.step_sync(&act);
            async_a.step_async_begin(&act);
            async_a.step_async_wait();
        }
        assert_eq!(sync_a.obs(), async_a.obs());
        assert_eq!(sync_a.rew(), async_a.rew());
        assert_eq!(sync_a.pred(), async_a.pred());
    }

    #[test]
    fn test_brief_conditioning_and_predicates() {
        let mut arena = RustVecArena::new(1, 12, 64, 64, 99, 1, 4, false);
        let mut briefs = vec![0.0f32; 12 * RL_BRIEF_IN_DIM];
        for i in 0..12 {
            let s = i * RL_BRIEF_IN_DIM;
            briefs[s] = OBJ_CAPTURE_SHRINE as f32;
            briefs[s + 1] = STANCE_FLEE_KITE as f32;
            briefs[s + 2] = 32.0;
            briefs[s + 3] = 16.0;
            briefs[s + 4] = 0.25;
            briefs[s + 5] = 1.0; // active
            briefs[s + 6] = -0.8; // frost bias
        }
        arena.set_briefs(&briefs);
        let obs = arena.obs();
        let bb = RL_BASE_FEAT_DIM;
        assert_eq!(obs[bb + OBJ_CAPTURE_SHRINE as usize], 1.0);
        assert_eq!(obs[bb + 5 + STANCE_FLEE_KITE as usize], 1.0);
        assert!((obs[bb + 11] - 0.25).abs() < 1e-5);
        assert_eq!(obs[bb + 13], 1.0);
        assert!((obs[bb + 14] - (-0.8)).abs() < 1e-5);
    }

    #[test]
    fn test_field_tensor_export() {
        let mut arena = RustVecArena::new(2, 12, 64, 64, 7, 2, 2, true);
        let mut buf = vec![0.0f32; 2 * INPUT_CHANNELS * 64 * 64];
        assert!(arena.export_field_tensors(&mut buf));
        assert!(buf.iter().all(|v| v.is_finite()));
        assert!(buf.iter().any(|&v| v > 0.0));
        assert_eq!(arena.crop().len(), 2 * 12 * RL_CROP_STRIDE);
    }
}
