use spacetimedb::{Identity, ReducerContext, ScheduleAt, Table};
use std::time::Duration;

pub const GRID_W: u16 = 128;
pub const GRID_H: u16 = 128;
pub const CHUNK_SIZE: u16 = 16;
pub const CHUNKS_X: u16 = GRID_W / CHUNK_SIZE;

// Action kinds sent to the 10 Hz Neural World Worker
pub const ACT_FIRE_BOLT: u8 = 0;
pub const ACT_FROST_BOLT: u8 = 1;
pub const ACT_OIL_SPLASH: u8 = 2;
pub const ACT_CARVE_ICE: u8 = 3;

#[spacetimedb::table(accessor = world_config, public)]
#[derive(Clone, Debug)]
pub struct WorldConfig {
    #[primary_key]
    pub id: u8,
    pub width: u16,
    pub height: u16,
    pub chunk_size: u16,
    /// "neural" | "refsim"
    pub sim_mode: String,
    pub ecosystem_enabled: bool,
    pub reset_nonce: u32,
    pub sim_tick: u64,
    pub score_west: u32,
    pub score_east: u32,
}

#[spacetimedb::table(accessor = player, public)]
#[derive(Clone, Debug)]
pub struct Player {
    #[primary_key]
    pub identity: Identity,
    pub name: String,
    pub team: u8, // 0 = West, 1 = East
    pub is_bot: bool,
    pub x: f32, // in cell coordinates [0..GRID_W]
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub aim_x: f32,
    pub aim_y: f32,
    pub hp: i32,
    pub max_hp: i32,
    pub dash_until_tick: u64,
    pub dash_cd_tick: u64,
    pub melee_until_tick: u64,
    pub melee_cd_tick: u64,
    pub bolt_cd_tick: u64,
    pub bolt_element: u8, // 0 = Fire, 1 = Frost, 2 = Oil
    pub carrying_relic: bool,
    pub frozen_until_tick: u64,
    pub kills: u32,
    pub last_input_seq: u32,
    pub online: bool,
}

#[spacetimedb::table(accessor = relic, public)]
#[derive(Clone, Debug)]
pub struct Relic {
    #[primary_key]
    pub id: u8,
    pub x: f32,
    pub y: f32,
    pub carrier: Option<Identity>,
}

#[spacetimedb::table(accessor = projectile, public)]
#[derive(Clone, Debug)]
pub struct Projectile {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub owner: Identity,
    pub team: u8,
    pub element: u8, // 0 = Fire, 1 = Frost, 2 = Oil
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub ttl: u16,
}

#[spacetimedb::table(accessor = cell_body, public)]
#[derive(Clone, Debug)]
pub struct CellBody {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub kind: u8, // 0 = IceRaft
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub radius: f32,
    pub mass: u32,
    pub ttl_ticks: u32,
}

#[spacetimedb::table(accessor = field_action, public)]
#[derive(Clone, Debug)]
pub struct FieldAction {
    #[primary_key]
    #[auto_inc]
    pub id: u64,
    pub kind: u8,
    pub cell_x: u16,
    pub cell_y: u16,
    pub radius: u8,
    pub strength: f32,
    pub processed: bool,
}

#[spacetimedb::table(accessor = field_chunk, public)]
#[derive(Clone, Debug)]
pub struct FieldChunk {
    #[primary_key]
    pub chunk_id: u32,
    pub cx: u16,
    pub cy: u16,
    pub tick: u64,
    /// 2 bytes/cell: [display_class, packed_temp_fill]
    pub display_bytes: Vec<u8>,
    /// 1 byte/cell: bit0=walkable, bit1=wall, bit2=burning, bit3=freezing, bit4=water_slow, bits5..7=critter(0..7)
    pub hazard_bytes: Vec<u8>,
}

#[spacetimedb::table(accessor = worker_telemetry, public)]
#[derive(Clone, Debug)]
pub struct WorkerTelemetry {
    #[primary_key]
    pub id: u8,
    pub mode: String,
    pub model_loaded: bool,
    pub ecosystem_loaded: bool,
    pub field_tick: u64,
    pub infer_ms_last: f32,
    pub infer_ms_p99: f32,
    pub proj_active_pct: f32,
    pub mass_error: i64,
    pub energy_error: i64,
    pub dirty_chunks_last: u32,
}

#[spacetimedb::table(accessor = sim_timer, scheduled(tick_sim))]
pub struct SimTimer {
    #[primary_key]
    #[auto_inc]
    pub scheduled_id: u64,
    pub scheduled_at: ScheduleAt,
}

fn spawn_point(team: u8, salt: u32) -> (f32, f32) {
    let offset = ((salt % 9) as f32 - 4.0) * 2.5;
    if team == 0 {
        (18.0, (GRID_H as f32) * 0.5 + offset)
    } else {
        ((GRID_W as f32) - 18.0, (GRID_H as f32) * 0.5 + offset)
    }
}

fn sample_hazard(ctx: &ReducerContext, x: f32, y: f32) -> u8 {
    let xi = (x.floor() as i32).clamp(0, (GRID_W - 1) as i32) as u16;
    let yi = (y.floor() as i32).clamp(0, (GRID_H - 1) as i32) as u16;
    let cx = xi / CHUNK_SIZE;
    let cy = yi / CHUNK_SIZE;
    let chunk_id = (cy as u32) * (CHUNKS_X as u32) + (cx as u32);
    if let Some(chunk) = ctx.db.field_chunk().chunk_id().find(chunk_id) {
        let lx = (xi % CHUNK_SIZE) as usize;
        let ly = (yi % CHUNK_SIZE) as usize;
        let idx = ly * (CHUNK_SIZE as usize) + lx;
        if idx < chunk.hazard_bytes.len() {
            return chunk.hazard_bytes[idx];
        }
    }
    // Default walkable before first worker commit
    1
}

fn on_any_cell_body(bodies: &[CellBody], x: f32, y: f32) -> Option<(f32, f32)> {
    for b in bodies {
        let dx = x - b.x;
        let dy = y - b.y;
        if dx * dx + dy * dy <= (b.radius + 0.8) * (b.radius + 0.8) {
            return Some((b.vx, b.vy));
        }
    }
    None
}

#[spacetimedb::reducer(init)]
pub fn init(ctx: &ReducerContext) {
    if ctx.db.world_config().id().find(0).is_none() {
        ctx.db.world_config().insert(WorldConfig {
            id: 0,
            width: GRID_W,
            height: GRID_H,
            chunk_size: CHUNK_SIZE,
            sim_mode: "neural".to_string(),
            ecosystem_enabled: true,
            reset_nonce: 0,
            sim_tick: 0,
            score_west: 0,
            score_east: 0,
        });
    }
    if ctx.db.relic().id().find(0).is_none() {
        ctx.db.relic().insert(Relic {
            id: 0,
            x: (GRID_W as f32) * 0.78,
            y: (GRID_H as f32) * 0.50,
            carrier: None,
        });
    }
    if ctx.db.worker_telemetry().id().find(0).is_none() {
        ctx.db.worker_telemetry().insert(WorkerTelemetry {
            id: 0,
            mode: "waiting_for_worker".to_string(),
            model_loaded: false,
            ecosystem_loaded: false,
            field_tick: 0,
            infer_ms_last: 0.0,
            infer_ms_p99: 0.0,
            proj_active_pct: 0.0,
            mass_error: 0,
            energy_error: 0,
            dirty_chunks_last: 0,
        });
    }
    if ctx.db.sim_timer().count() == 0 {
        ctx.db.sim_timer().insert(SimTimer {
            scheduled_id: 0,
            scheduled_at: ScheduleAt::Interval(Duration::from_millis(33).into()),
        });
    }
    log::info!("Frostmarsh SpacetimeDB module initialized ({}x{})", GRID_W, GRID_H);
}

#[spacetimedb::reducer(client_connected)]
pub fn on_connect(_ctx: &ReducerContext) {}

#[spacetimedb::reducer(client_disconnected)]
pub fn on_disconnect(ctx: &ReducerContext) {
    if let Some(p) = ctx.db.player().identity().find(ctx.sender()) {
        if p.carrying_relic {
            if let Some(r) = ctx.db.relic().id().find(0) {
                ctx.db.relic().id().update(Relic {
                    x: p.x,
                    y: p.y,
                    carrier: None,
                    ..r
                });
            }
        }
        ctx.db.player().identity().update(Player {
            online: false,
            carrying_relic: false,
            vx: 0.0,
            vy: 0.0,
            ..p
        });
    }
}

#[spacetimedb::reducer]
pub fn join_player(ctx: &ReducerContext, name: String, team: u8) {
    let team = team % 2;
    let count = ctx.db.player().count() as u32;
    let (sx, sy) = spawn_point(team, count);
    if let Some(existing) = ctx.db.player().identity().find(ctx.sender()) {
        ctx.db.player().identity().update(Player {
            name: if name.is_empty() { existing.name } else { name },
            team,
            online: true,
            hp: existing.max_hp,
            x: sx,
            y: sy,
            ..existing
        });
    } else {
        ctx.db.player().insert(Player {
            identity: ctx.sender(),
            name: if name.is_empty() {
                format!("Hero-{}", count + 1)
            } else {
                name
            },
            team,
            is_bot: false,
            x: sx,
            y: sy,
            vx: 0.0,
            vy: 0.0,
            aim_x: if team == 0 { 1.0 } else { -1.0 },
            aim_y: 0.0,
            hp: 100,
            max_hp: 100,
            dash_until_tick: 0,
            dash_cd_tick: 0,
            melee_until_tick: 0,
            melee_cd_tick: 0,
            bolt_cd_tick: 0,
            bolt_element: 1, // Start with Frost bolt to freeze the river!
            carrying_relic: false,
            frozen_until_tick: 0,
            kills: 0,
            last_input_seq: 0,
            online: true,
        });
    }
}

#[spacetimedb::reducer]
#[allow(clippy::too_many_arguments)]
pub fn player_input(
    ctx: &ReducerContext,
    seq: u32,
    move_x: f32,
    move_y: f32,
    aim_x: f32,
    aim_y: f32,
    dash: bool,
    melee: bool,
    cast_bolt: bool,
    bolt_element: u8,
    carve_ice: bool,
) {
    let Some(mut p) = ctx.db.player().identity().find(ctx.sender()) else {
        return;
    };
    let tick = ctx.db.world_config().id().find(0).map(|c| c.sim_tick).unwrap_or(0);
    apply_player_controls(
        ctx,
        &mut p,
        tick,
        seq,
        move_x,
        move_y,
        aim_x,
        aim_y,
        dash,
        melee,
        cast_bolt,
        bolt_element,
        carve_ice,
    );
    ctx.db.player().identity().update(p);
}

#[spacetimedb::reducer]
#[allow(clippy::too_many_arguments)]
pub fn bot_control(
    ctx: &ReducerContext,
    bot_id: u64,
    name: String,
    team: u8,
    move_x: f32,
    move_y: f32,
    aim_x: f32,
    aim_y: f32,
    dash: bool,
    melee: bool,
    cast_bolt: bool,
    bolt_element: u8,
) {
    let mut bytes = [0u8; 32];
    bytes[0..4].copy_from_slice(b"BOT!");
    bytes[4..12].copy_from_slice(&bot_id.to_le_bytes());
    let ident = Identity::from_byte_array(bytes);
    let tick = ctx.db.world_config().id().find(0).map(|c| c.sim_tick).unwrap_or(0);

    let mut p = if let Some(existing) = ctx.db.player().identity().find(ident) {
        existing
    } else {
        let (sx, sy) = spawn_point(team % 2, bot_id as u32);
        let new_p = Player {
            identity: ident,
            name,
            team: team % 2,
            is_bot: true,
            x: sx,
            y: sy,
            vx: 0.0,
            vy: 0.0,
            aim_x: if team % 2 == 0 { 1.0 } else { -1.0 },
            aim_y: 0.0,
            hp: 100,
            max_hp: 100,
            dash_until_tick: 0,
            dash_cd_tick: 0,
            melee_until_tick: 0,
            melee_cd_tick: 0,
            bolt_cd_tick: 0,
            bolt_element: bolt_element % 3,
            carrying_relic: false,
            frozen_until_tick: 0,
            kills: 0,
            last_input_seq: 0,
            online: true,
        };
        ctx.db.player().insert(new_p.clone());
        new_p
    };

    let next_seq = p.last_input_seq.wrapping_add(1);
    apply_player_controls(
        ctx,
        &mut p,
        tick,
        next_seq,
        move_x,
        move_y,
        aim_x,
        aim_y,
        dash,
        melee,
        cast_bolt,
        bolt_element,
        false,
    );
    ctx.db.player().identity().update(p);
}

#[allow(clippy::too_many_arguments)]
fn apply_player_controls(
    ctx: &ReducerContext,
    p: &mut Player,
    tick: u64,
    seq: u32,
    move_x: f32,
    move_y: f32,
    aim_x: f32,
    aim_y: f32,
    dash: bool,
    melee: bool,
    cast_bolt: bool,
    bolt_element: u8,
    carve_ice: bool,
) {
    p.last_input_seq = seq;
    p.bolt_element = bolt_element % 3;

    let aim_len = (aim_x * aim_x + aim_y * aim_y).sqrt();
    if aim_len > 0.01 {
        p.aim_x = aim_x / aim_len;
        p.aim_y = aim_y / aim_len;
    }

    let mv_len = (move_x * move_x + move_y * move_y).sqrt();
    let (dx, dy) = if mv_len > 0.01 {
        (move_x / mv_len, move_y / mv_len)
    } else {
        (0.0, 0.0)
    };

    // Dash: 150ms (5 ticks) i-frames + burst speed, 1s (30 ticks) cooldown
    if dash && tick >= p.dash_cd_tick {
        p.dash_until_tick = tick + 5;
        p.dash_cd_tick = tick + 30;
        let (ddx, ddy) = if mv_len > 0.01 { (dx, dy) } else { (p.aim_x, p.aim_y) };
        p.vx = ddx * 34.0;
        p.vy = ddy * 34.0;
    } else if tick >= p.dash_until_tick {
        let base_speed = if p.carrying_relic { 11.5 } else { 15.0 };
        p.vx = dx * base_speed;
        p.vy = dy * base_speed;
    }

    // Melee arc (90 deg, 4.5 cell reach, 15 ticks CD)
    if melee && tick >= p.melee_cd_tick {
        p.melee_until_tick = tick + 4;
        p.melee_cd_tick = tick + 15;
        let all_players: Vec<Player> = ctx.db.player().iter().collect();
        for mut target in all_players {
            if target.identity == p.identity || !target.online || target.team == p.team {
                continue;
            }
            if tick < target.dash_until_tick {
                continue; // i-frames!
            }
            let rx = target.x - p.x;
            let ry = target.y - p.y;
            let dist = (rx * rx + ry * ry).sqrt();
            if dist <= 5.0 && dist > 0.001 {
                let dot = (rx / dist) * p.aim_x + (ry / dist) * p.aim_y;
                if dot >= 0.45 {
                    target.hp -= 28;
                    if target.hp <= 0 {
                        p.kills += 1;
                        respawn_player(ctx, &mut target, tick);
                    }
                    ctx.db.player().identity().update(target);
                }
            }
        }
    }

    // Cast elemental bolt (writes to neural field on impact!)
    if cast_bolt && tick >= p.bolt_cd_tick {
        p.bolt_cd_tick = tick + 10;
        ctx.db.projectile().insert(Projectile {
            id: 0,
            owner: p.identity,
            team: p.team,
            element: p.bolt_element,
            x: p.x + p.aim_x * 1.4,
            y: p.y + p.aim_y * 1.4,
            vx: p.aim_x * 46.0,
            vy: p.aim_y * 46.0,
            ttl: 28,
        });
    }

    // Carve an ice raft from frozen river cells right in front of the player!
    if carve_ice {
        let tx = (p.x + p.aim_x * 2.5).clamp(2.0, (GRID_W - 3) as f32) as u16;
        let ty = (p.y + p.aim_y * 2.5).clamp(2.0, (GRID_H - 3) as f32) as u16;
        ctx.db.field_action().insert(FieldAction {
            id: 0,
            kind: ACT_CARVE_ICE,
            cell_x: tx,
            cell_y: ty,
            radius: 3,
            strength: 1.0,
            processed: false,
        });
    }
}

fn respawn_player(ctx: &ReducerContext, p: &mut Player, tick: u64) {
    if p.carrying_relic {
        p.carrying_relic = false;
        if let Some(r) = ctx.db.relic().id().find(0) {
            ctx.db.relic().id().update(Relic {
                x: p.x,
                y: p.y,
                carrier: None,
                ..r
            });
        }
    }
    let (sx, sy) = spawn_point(p.team, tick as u32);
    p.x = sx;
    p.y = sy;
    p.vx = 0.0;
    p.vy = 0.0;
    p.hp = p.max_hp;
}

#[spacetimedb::reducer]
pub fn toggle_sim_mode(ctx: &ReducerContext) {
    if let Some(cfg) = ctx.db.world_config().id().find(0) {
        let next = if cfg.sim_mode == "neural" {
            "refsim".to_string()
        } else {
            "neural".to_string()
        };
        ctx.db.world_config().id().update(WorldConfig {
            sim_mode: next,
            ..cfg
        });
    }
}

#[spacetimedb::reducer]
pub fn toggle_ecosystem(ctx: &ReducerContext) {
    if let Some(cfg) = ctx.db.world_config().id().find(0) {
        ctx.db.world_config().id().update(WorldConfig {
            ecosystem_enabled: !cfg.ecosystem_enabled,
            ..cfg
        });
    }
}

#[spacetimedb::reducer]
pub fn reset_world(ctx: &ReducerContext) {
    if let Some(cfg) = ctx.db.world_config().id().find(0) {
        ctx.db.world_config().id().update(WorldConfig {
            reset_nonce: cfg.reset_nonce.wrapping_add(1),
            ..cfg
        });
    }
    for b in ctx.db.cell_body().iter().collect::<Vec<_>>() {
        ctx.db.cell_body().id().delete(b.id);
    }
}

/// Called by the co-located 10 Hz Neural World Worker to push dirty chunks & telemetry.
#[spacetimedb::reducer]
#[allow(clippy::too_many_arguments)]
pub fn commit_field_step(
    ctx: &ReducerContext,
    mode: String,
    model_loaded: bool,
    ecosystem_loaded: bool,
    field_tick: u64,
    infer_ms_last: f32,
    infer_ms_p99: f32,
    proj_active_pct: f32,
    mass_error: i64,
    energy_error: i64,
    processed_action_ids: Vec<u64>,
    chunk_ids: Vec<u32>,
    chunk_cxs: Vec<u16>,
    chunk_cys: Vec<u16>,
    chunk_displays: Vec<Vec<u8>>,
    chunk_hazards: Vec<Vec<u8>>,
    new_raft_xs: Vec<f32>,
    new_raft_ys: Vec<f32>,
    new_raft_radii: Vec<f32>,
    new_raft_masses: Vec<u32>,
) {
    for aid in processed_action_ids {
        ctx.db.field_action().id().delete(aid);
    }

    let n = chunk_ids.len();
    for i in 0..n {
        let row = FieldChunk {
            chunk_id: chunk_ids[i],
            cx: chunk_cxs[i],
            cy: chunk_cys[i],
            tick: field_tick,
            display_bytes: chunk_displays[i].clone(),
            hazard_bytes: chunk_hazards[i].clone(),
        };
        if ctx.db.field_chunk().chunk_id().find(row.chunk_id).is_some() {
            ctx.db.field_chunk().chunk_id().update(row);
        } else {
            ctx.db.field_chunk().insert(row);
        }
    }

    for i in 0..new_raft_xs.len() {
        ctx.db.cell_body().insert(CellBody {
            id: 0,
            kind: 0,
            x: new_raft_xs[i],
            y: new_raft_ys[i],
            vx: 0.0,
            vy: 3.0, // drifts southward with river current
            radius: new_raft_radii[i],
            mass: new_raft_masses[i],
            ttl_ticks: 600, // melts after ~20s
        });
    }

    let tele = WorkerTelemetry {
        id: 0,
        mode,
        model_loaded,
        ecosystem_loaded,
        field_tick,
        infer_ms_last,
        infer_ms_p99,
        proj_active_pct,
        mass_error,
        energy_error,
        dirty_chunks_last: n as u32,
    };
    if ctx.db.worker_telemetry().id().find(0).is_some() {
        ctx.db.worker_telemetry().id().update(tele);
    } else {
        ctx.db.worker_telemetry().insert(tele);
    }
}

/// 30 Hz authoritative combat, movement, projectile, and raft tick.
#[spacetimedb::reducer]
pub fn tick_sim(ctx: &ReducerContext, _timer: SimTimer) {
    let Some(mut cfg) = ctx.db.world_config().id().find(0) else {
        return;
    };
    cfg.sim_tick += 1;
    let tick = cfg.sim_tick;
    let dt = 1.0 / 30.0f32;

    // 1. Advance floating CellBodies (ice rafts drift southward on water)
    let mut bodies: Vec<CellBody> = ctx.db.cell_body().iter().collect();
    let mut alive_bodies = Vec::with_capacity(bodies.len());
    for mut b in bodies.drain(..) {
        if b.ttl_ticks == 0 {
            ctx.db.cell_body().id().delete(b.id);
            continue;
        }
        b.ttl_ticks -= 1;
        let nx = (b.x + b.vx * dt).clamp(3.0, (GRID_W - 4) as f32);
        let ny = (b.y + b.vy * dt).clamp(3.0, (GRID_H - 4) as f32);
        let hz = sample_hazard(ctx, nx, ny);
        if (hz & (1 << 1)) == 0 {
            b.x = nx;
            b.y = ny;
        }
        // Slow down if grounded on land
        if (hz & (1 << 4)) == 0 {
            b.vx *= 0.85;
            b.vy *= 0.85;
        } else {
            b.vy = b.vy * 0.9 + 3.2 * 0.1;
        }
        ctx.db.cell_body().id().update(b.clone());
        alive_bodies.push(b);
    }

    // 2. Move players against neural field walkability + hazards + rafts
    let mut players: Vec<Player> = ctx.db.player().iter().collect();
    for p in &mut players {
        if !p.online {
            continue;
        }
        let hz_here = sample_hazard(ctx, p.x, p.y);
        let on_raft = on_any_cell_body(&alive_bodies, p.x, p.y);

        // Speed modifiers from neural field (mud/water slow, freezing slow)
        let mut speed_mult = 1.0f32;
        if on_raft.is_none() && (hz_here & (1 << 4)) != 0 {
            speed_mult *= 0.60;
        }
        if (hz_here & (1 << 3)) != 0 && tick >= p.dash_until_tick {
            speed_mult *= 0.72;
        }

        let (carry_vx, carry_vy) = on_raft.unwrap_or((0.0, 0.0));
        let nx = (p.x + (p.vx * speed_mult + carry_vx) * dt).clamp(2.5, (GRID_W as f32) - 2.5);
        let ny = (p.y + (p.vy * speed_mult + carry_vy) * dt).clamp(2.5, (GRID_H as f32) - 2.5);

        let hz_next = sample_hazard(ctx, nx, ny);
        let next_on_raft = on_any_cell_body(&alive_bodies, nx, ny).is_some();
        let can_step = (hz_next & (1 << 0)) != 0 || next_on_raft || tick < p.dash_until_tick;
        if can_step && (hz_next & (1 << 1)) == 0 {
            p.x = nx;
            p.y = ny;
        } else {
            // Slide along X or Y independently
            let hz_x = sample_hazard(ctx, nx, p.y);
            if ((hz_x & (1 << 0)) != 0 || on_any_cell_body(&alive_bodies, nx, p.y).is_some())
                && (hz_x & (1 << 1)) == 0
            {
                p.x = nx;
            }
            let hz_y = sample_hazard(ctx, p.x, ny);
            if ((hz_y & (1 << 0)) != 0 || on_any_cell_body(&alive_bodies, p.x, ny).is_some())
                && (hz_y & (1 << 1)) == 0
            {
                p.y = ny;
            }
        }

        // Fire hazard DoT from live neural field (every 6 ticks = 5 Hz)
        if tick % 6 == 0 && (hz_here & (1 << 2)) != 0 && tick >= p.dash_until_tick {
            p.hp -= 4;
            if p.hp <= 0 {
                respawn_player(ctx, p, tick);
            }
        } else if tick % 30 == 0 && p.hp < p.max_hp && (hz_here & (1 << 2)) == 0 {
            p.hp = (p.hp + 2).min(p.max_hp);
        }
    }

    // 3. Advance projectiles & resolve impacts (both player hits AND neural field injections!)
    let projectiles: Vec<Projectile> = ctx.db.projectile().iter().collect();
    for mut pr in projectiles {
        let mut hit = false;
        let nx = pr.x + pr.vx * dt;
        let ny = pr.y + pr.vy * dt;

        // Check bounds or solid wall hit
        if nx < 2.0 || nx >= (GRID_W as f32) - 2.0 || ny < 2.0 || ny >= (GRID_H as f32) - 2.0 {
            hit = true;
        } else {
            let hz = sample_hazard(ctx, nx, ny);
            if (hz & (1 << 1)) != 0 {
                hit = true;
            }
        }

        // Check player hits
        if !hit {
            for p in &mut players {
                if !p.online || p.identity == pr.owner || p.team == pr.team {
                    continue;
                }
                if tick < p.dash_until_tick {
                    continue; // dash i-frames!
                }
                let dx = p.x - nx;
                let dy = p.y - ny;
                if dx * dx + dy * dy <= 2.2 * 2.2 {
                    hit = true;
                    let dmg = match pr.element {
                        0 => 18, // fire
                        1 => 14, // frost
                        _ => 6,  // oil
                    };
                    p.hp -= dmg;
                    if p.hp <= 0 {
                        respawn_player(ctx, p, tick);
                    }
                    break;
                }
            }
        }

        if hit || pr.ttl <= 1 {
            // Inject bolt energy/material directly into the 10 Hz Neural World Model!
            let cx = nx.clamp(2.0, (GRID_W - 3) as f32) as u16;
            let cy = ny.clamp(2.0, (GRID_H - 3) as f32) as u16;
            let (kind, radius, strength) = match pr.element {
                0 => (ACT_FIRE_BOLT, 4u8, 1.5f32),
                1 => (ACT_FROST_BOLT, 5u8, 1.8f32),
                _ => (ACT_OIL_SPLASH, 4u8, 1.4f32),
            };
            ctx.db.field_action().insert(FieldAction {
                id: 0,
                kind,
                cell_x: cx,
                cell_y: cy,
                radius,
                strength,
                processed: false,
            });
            ctx.db.projectile().id().delete(pr.id);
        } else {
            pr.x = nx;
            pr.y = ny;
            pr.ttl -= 1;
            ctx.db.projectile().id().update(pr);
        }
    }

    // 4. Relic pickup & capture logic
    if let Some(mut relic) = ctx.db.relic().id().find(0) {
        if let Some(carrier_id) = relic.carrier {
            if let Some(cp) = players.iter_mut().find(|p| p.identity == carrier_id && p.online) {
                relic.x = cp.x;
                relic.y = cp.y;
                // Score if West carries relic back to West bank (< 24) or East to East bank (> GRID_W - 24)
                let scored_west = cp.team == 0 && cp.x < 24.0;
                let scored_east = cp.team == 1 && cp.x > (GRID_W as f32) - 24.0;
                if scored_west || scored_east {
                    if scored_west {
                        cfg.score_west += 1;
                    } else {
                        cfg.score_east += 1;
                    }
                    cp.carrying_relic = false;
                    relic.carrier = None;
                    relic.x = if scored_west {
                        (GRID_W as f32) * 0.78
                    } else {
                        (GRID_W as f32) * 0.22
                    };
                    relic.y = (GRID_H as f32) * 0.50;
                }
            } else {
                relic.carrier = None;
            }
        } else {
            for p in &mut players {
                if !p.online {
                    continue;
                }
                let dx = p.x - relic.x;
                let dy = p.y - relic.y;
                if dx * dx + dy * dy <= 3.0 * 3.0 {
                    p.carrying_relic = true;
                    relic.carrier = Some(p.identity);
                    break;
                }
            }
        }
        ctx.db.relic().id().update(relic);
    }

    for p in players {
        ctx.db.player().identity().update(p);
    }
    ctx.db.world_config().id().update(cfg);
}
