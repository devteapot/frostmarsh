use clap::Parser;
use field_core::{
    CHUNK_SIZE, Field, INPUT_CHANNELS, OUTPUT_CHANNELS, RawStepOutput, SP_ICE, SP_PLANT, SP_ROCK,
    SP_WATER,
};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use spacetimedb_sdk::{DbContext, Table};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use stdb_bindings::*;

#[derive(Parser, Debug)]
#[command(name = "world-worker", about = "10 Hz Authoritative Neural World Server for Frostmarsh")]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    uri: String,
    #[arg(long, default_value = "frostmarsh")]
    db: String,
    #[arg(long, default_value = "models/world_model.onnx")]
    model: PathBuf,
    #[arg(long, default_value = "models/ecosystem_model.onnx")]
    ecosystem_model: PathBuf,
    #[arg(long, default_value_t = 6)]
    bots: usize,
    #[arg(long, default_value_t = 10.0)]
    hz: f64,
}

struct NeuralPhysicsModel {
    session: Session,
    mtime: Option<SystemTime>,
}

impl NeuralPhysicsModel {
    fn load(env: &Environment, path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        let session = Session::builder(env)?
            .with_intra_threads(4)?
            .commit_from_file(path)?;
        Ok(Self { session, mtime })
    }

    fn predict(&mut self, field: &Field) -> Result<RawStepOutput, Box<dyn std::error::Error>> {
        let w = field.width;
        let h = field.height;
        let input_data = field.encode_input_tensor();
        let input_tensor = Tensor::from_array(([1usize, INPUT_CHANNELS, h, w], input_data))?;
        let outputs = self.session.run(ort::inputs!["state" => input_tensor])?;
        let (_shape, slice) = outputs["deltas"].try_extract_tensor::<f32>()?;
        let expected = OUTPUT_CHANNELS * h * w;
        if slice.len() != expected {
            return Err(format!(
                "ONNX output size mismatch: got {}, expected {}",
                slice.len(),
                expected
            )
            .into());
        }
        Ok(RawStepOutput {
            width: w,
            height: h,
            data: slice.to_vec(),
        })
    }
}

struct NeuralEcosystemModel {
    session: Session,
    mtime: Option<SystemTime>,
    /// Persistent coarse ecosystem channels [4, H, W]: critter_density, fear, forage, momentum
    state: Vec<f32>,
    w: usize,
    h: usize,
}

impl NeuralEcosystemModel {
    fn load(
        env: &Environment,
        path: &Path,
        w: usize,
        h: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        let session = Session::builder(env)?
            .with_intra_threads(2)?
            .commit_from_file(path)?;
        let mut eco = Self {
            session,
            mtime,
            state: vec![0.0; 4 * w * h],
            w,
            h,
        };
        eco.reset_initial();
        Ok(eco)
    }

    fn reset_initial(&mut self) {
        self.state.fill(0.0);
        let len = self.w * self.h;
        for y in 0..self.h {
            for x in 0..self.w {
                let nx = x as f32 / self.w as f32;
                let ny = y as f32 / self.h as f32;
                let idx = y * self.w + x;
                // Seed two wildlife herds on West & East grassy meadows
                let d1 = ((nx - 0.25).powi(2) + (ny - 0.30).powi(2)).sqrt();
                let d2 = ((nx - 0.76).powi(2) + (ny - 0.72).powi(2)).sqrt();
                let dens = ((-(d1 * 12.0).powi(2)).exp() + (-(d2 * 12.0).powi(2)).exp()).min(1.0);
                self.state[idx] = dens; // critter density
                self.state[len + idx] = 0.0; // fear
                self.state[2 * len + idx] = 0.8; // forage
                self.state[3 * len + idx] = 0.0; // momentum
            }
        }
    }

    fn step(&mut self, field: &mut Field) -> Result<(), Box<dyn std::error::Error>> {
        let len = self.w * self.h;
        // Build [1, 8, H, W] input: 4 eco channels + 4 terrain/hazard channels
        // [critter, fear, forage, momentum, walkable, burning, plant, ice_bridge]
        let mut inp = vec![0.0f32; 8 * len];
        for idx in 0..len {
            inp[idx] = self.state[idx];
            inp[len + idx] = self.state[len + idx];
            inp[2 * len + idx] = self.state[2 * len + idx];
            inp[3 * len + idx] = self.state[3 * len + idx];

            let t = field.cell_temp_ck(idx);
            let is_wall = field.mass[SP_ROCK][idx] >= 500;
            let is_deep_water = field.mass[SP_WATER][idx] >= 450 && field.mass[SP_ICE][idx] < 350;
            let walkable = if !is_wall && !is_deep_water { 1.0 } else { 0.0 };
            let burning = if t > 44_000 { 1.0 } else { 0.0 };
            let plant = (field.mass[SP_PLANT][idx] as f32 / 1000.0).clamp(0.0, 1.0);
            let ice = (field.mass[SP_ICE][idx] as f32 / 1000.0).clamp(0.0, 1.0);

            inp[4 * len + idx] = walkable;
            inp[5 * len + idx] = burning;
            inp[6 * len + idx] = plant;
            inp[7 * len + idx] = ice;
        }

        let input_tensor = Tensor::from_array(([1usize, 8, self.h, self.w], inp))?;
        let outputs = self.session.run(ort::inputs!["eco_state" => input_tensor])?;
        let (_shape, slice) = outputs["eco_next"].try_extract_tensor::<f32>()?;
        if slice.len() == 4 * len {
            for i in 0..(4 * len) {
                let v = slice[i];
                self.state[i] = if v.is_finite() { v.clamp(0.0, 1.5) } else { 0.0 };
            }
            for idx in 0..len {
                let d = (self.state[idx] * 7.0).round().clamp(0.0, 7.0) as u8;
                field.critter_density[idx] = d;
            }
        }
        Ok(())
    }
}

fn try_load_physics(env: &Environment, path: &Path) -> Option<NeuralPhysicsModel> {
    if !path.exists() {
        return None;
    }
    match NeuralPhysicsModel::load(env, path) {
        Ok(m) => {
            log::info!("Loaded ONNX Neural World Model from {}", path.display());
            Some(m)
        }
        Err(e) => {
            log::warn!("Failed to load ONNX model {}: {}", path.display(), e);
            None
        }
    }
}

fn try_load_ecosystem(
    env: &Environment,
    path: &Path,
    w: usize,
    h: usize,
) -> Option<NeuralEcosystemModel> {
    if !path.exists() {
        return None;
    }
    match NeuralEcosystemModel::load(env, path, w, h) {
        Ok(m) => {
            log::info!("Loaded ONNX Neural Ecosystem Model from {}", path.display());
            Some(m)
        }
        Err(e) => {
            log::warn!("Failed to load ecosystem model {}: {}", path.display(), e);
            None
        }
    }
}

fn step_bots(conn: &DbConnection, bot_count: usize, tick: u64, w: f32, h: f32) {
    if bot_count == 0 {
        return;
    }
    let relic = conn.db.relic().iter().next();
    let players: Vec<Player> = conn.db.player().iter().collect();

    for i in 0..bot_count {
        let bot_id = (i + 1) as u64;
        let team = (i % 2) as u8;
        let name = if team == 0 {
            format!("FrostBot-{}", i / 2 + 1)
        } else {
            format!("PyroBot-{}", i / 2 + 1)
        };

        // Find current bot state if spawned
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(b"BOT!");
        bytes[4..12].copy_from_slice(&bot_id.to_le_bytes());
        let ident = spacetimedb_sdk::Identity::from_byte_array(bytes);
        let me = players.iter().find(|p| p.identity == ident);

        let (mx, my) = me.map(|p| (p.x, p.y)).unwrap_or((
            if team == 0 { 18.0 } else { w - 18.0 },
            h * 0.5,
        ));

        // Target: carry relic home, or move toward relic / river crossing
        let (tx, ty) = if me.map(|p| p.carrying_relic).unwrap_or(false) {
            if team == 0 {
                (16.0, h * 0.5)
            } else {
                (w - 16.0, h * 0.5)
            }
        } else if let Some(ref r) = relic {
            // Aim toward northern ice bridge or river crossing near relic
            let crossing_y = if (mx - w * 0.5).abs() > 14.0 && ((team == 0) == (mx < w * 0.5)) {
                h * (0.20 + 0.14 * ((i as f32 * 1.7 + tick as f32 * 0.02).sin()))
            } else {
                r.y + ((i as f32 * 2.1).sin() * 6.0)
            };
            (r.x, crossing_y)
        } else {
            (w * 0.5, h * 0.5)
        };

        let dx = tx - mx;
        let dy = ty - my;

        // Find nearest enemy for aiming / casting
        let enemy = players
            .iter()
            .filter(|p| p.online && p.team != team)
            .min_by(|a, b| {
                let da = (a.x - mx).powi(2) + (a.y - my).powi(2);
                let db = (b.x - mx).powi(2) + (b.y - my).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        let (aim_x, aim_y, enemy_dist) = if let Some(e) = enemy {
            let ex = e.x - mx;
            let ey = e.y - my;
            (ex, ey, (ex * ex + ey * ey).sqrt())
        } else {
            (dx, dy, 999.0)
        };

        // West bots favor Frost bolts (freeze river!), East bots favor Fire bolts (melt bridges & burn woods!)
        let near_river = (mx - w * 0.5).abs() < 22.0;
        let elem = if team == 0 {
            if near_river && tick % 3 != 0 { 1 } else { 0 }
        } else if tick % 5 == 0 {
            2 // occasional oil splash
        } else {
            0
        };

        let cast_bolt = (tick + i as u64 * 3) % 7 == 0 && (enemy_dist < 42.0 || near_river);
        let melee = enemy_dist < 5.2;
        let dash = (tick + i as u64 * 11) % 23 == 0;

        let _ = conn.reducers.bot_control(
            bot_id,
            name,
            team,
            dx,
            dy,
            aim_x,
            aim_y,
            dash,
            melee,
            cast_bolt,
            elem,
        );
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    log::info!(
        "Connecting world-worker to SpacetimeDB at {} (db: {})",
        args.uri,
        args.db
    );

    let conn = DbConnection::builder()
        .with_uri(&args.uri)
        .with_database_name(&args.db)
        .on_connect(|_conn, identity, _token| {
            log::info!("Connected to SpacetimeDB as {:?}", identity);
        })
        .on_connect_error(|_ctx, err| {
            log::error!("SpacetimeDB connection error: {:?}", err);
        })
        .on_disconnect(|_ctx, err| {
            log::warn!("SpacetimeDB disconnected: {:?}", err);
        })
        .build()
        .expect("Failed to connect to SpacetimeDB");

    conn.subscription_builder()
        .on_applied(|_ctx| {
            log::info!("Subscribed to Frostmarsh tables.");
        })
        .subscribe([
            "SELECT * FROM world_config",
            "SELECT * FROM field_action",
            "SELECT * FROM player",
            "SELECT * FROM relic",
        ]);

    let ort_env = ort::init()
        .with_name("frostmarsh-world-worker")
        .build()
        .expect("Failed to initialize ONNX Runtime environment");

    let w = 128usize;
    let h = 128usize;
    let mut field = Field::new_frostmarsh(w, h);
    let mut physics_model = try_load_physics(&ort_env, &args.model);
    let mut eco_model = try_load_ecosystem(&ort_env, &args.ecosystem_model, w, h);

    let mut last_model_check = Instant::now();
    let mut last_reset_nonce: u32 = 0;
    let mut infer_window: VecDeque<f32> = VecDeque::with_capacity(120);
    let tick_period = Duration::from_secs_f64(1.0 / args.hz.max(1.0));

    // Wait briefly for initial subscription hydration
    for _ in 0..20 {
        let _ = conn.frame_tick();
        std::thread::sleep(Duration::from_millis(25));
    }

    log::info!(
        "Starting {} Hz world loop (neural physics: {}, neural ecosystem: {}, bots: {})",
        args.hz,
        physics_model.is_some(),
        eco_model.is_some(),
        args.bots
    );

    loop {
        let loop_start = Instant::now();
        if let Err(e) = conn.frame_tick() {
            log::error!("frame_tick error: {:?}", e);
            std::thread::sleep(Duration::from_millis(200));
            continue;
        }

        // mtime-based hot-reload: loads or refreshes ONNX weights whenever updated on disk
        if last_model_check.elapsed() > Duration::from_secs(3) {
            last_model_check = Instant::now();
            if args.model.exists() {
                let dm = std::fs::metadata(&args.model).ok().and_then(|m| m.modified().ok());
                if physics_model.as_ref().map(|m| m.mtime) != Some(dm) {
                    if let Some(m) = try_load_physics(&ort_env, &args.model) {
                        physics_model = Some(m);
                    }
                }
            }
            if args.ecosystem_model.exists() {
                let dm = std::fs::metadata(&args.ecosystem_model).ok().and_then(|m| m.modified().ok());
                if eco_model.as_ref().map(|m| m.mtime) != Some(dm) {
                    if let Some(m) = try_load_ecosystem(&ort_env, &args.ecosystem_model, w, h) {
                        eco_model = Some(m);
                    }
                }
            }
        }

        let cfg = conn.db.world_config().iter().next();
        let desired_mode = cfg
            .as_ref()
            .map(|c| c.sim_mode.as_str())
            .unwrap_or("neural");
        let eco_enabled = cfg.as_ref().map(|c| c.ecosystem_enabled).unwrap_or(true);
        let reset_nonce = cfg.as_ref().map(|c| c.reset_nonce).unwrap_or(0);

        if reset_nonce != last_reset_nonce {
            log::info!("Resetting Frostmarsh world (nonce={})", reset_nonce);
            last_reset_nonce = reset_nonce;
            field = Field::new_frostmarsh(w, h);
            field.mark_all_chunks_dirty();
            if let Some(ref mut eco) = eco_model {
                eco.reset_initial();
            }
        }

        // Drain player actions from SpacetimeDB into the live field
        let mut processed_ids = Vec::new();
        let mut raft_xs = Vec::new();
        let mut raft_ys = Vec::new();
        let mut raft_radii = Vec::new();
        let mut raft_masses = Vec::new();

        for act in conn.db.field_action().iter() {
            if act.processed {
                continue;
            }
            processed_ids.push(act.id);
            let cx = act.cell_x as i32;
            let cy = act.cell_y as i32;
            let r = act.radius as i32;
            match act.kind {
                0 => field.apply_fire_bolt(cx, cy, r, act.strength),
                1 => field.apply_frost_bolt(cx, cy, r, act.strength),
                2 => field.apply_oil_splash(cx, cy, r, act.strength),
                3 => {
                    if let Some(raft) = field.carve_ice_raft(cx, cy, r) {
                        raft_xs.push(raft.center_x);
                        raft_ys.push(raft.center_y);
                        raft_radii.push(raft.radius);
                        raft_masses.push(raft.ice_mass);
                    }
                }
                _ => {}
            }
        }

        // Run either the trained Neural World Model or exact RefSim baseline
        let t0 = Instant::now();
        let (active_mode, raw) = if desired_mode == "neural" {
            if let Some(ref mut model) = physics_model {
                match model.predict(&field) {
                    Ok(out) => ("neural".to_string(), out),
                    Err(e) => {
                        log::warn!("Neural inference error, falling back to RefSim: {}", e);
                        ("refsim_fallback".to_string(), field.refsim_raw_outputs())
                    }
                }
            } else {
                ("refsim_no_model".to_string(), field.refsim_raw_outputs())
            }
        } else {
            ("refsim".to_string(), field.refsim_raw_outputs())
        };
        let infer_ms = t0.elapsed().as_secs_f32() * 1000.0;

        // Pass raw predictions through the 10-step exact integer Projection Firewall
        let stats = field.project_and_apply(&raw);

        // Optional Phase 3 Neural Ecosystem step
        if eco_enabled {
            if let Some(ref mut eco) = eco_model {
                if let Err(e) = eco.step(&mut field) {
                    log::warn!("Ecosystem inference error: {}", e);
                }
            }
        } else {
            field.critter_density.fill(0);
        }

        if infer_window.len() >= 100 {
            infer_window.pop_front();
        }
        infer_window.push_back(infer_ms);
        let mut sorted: Vec<f32> = infer_window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99 = sorted
            .get(((sorted.len() as f32) * 0.99) as usize)
            .or_else(|| sorted.last())
            .copied()
            .unwrap_or(infer_ms);

        // Encode only dirty 16x16 chunks
        let mut chunk_ids = Vec::with_capacity(stats.dirty_chunks.len());
        let mut chunk_cxs = Vec::with_capacity(stats.dirty_chunks.len());
        let mut chunk_cys = Vec::with_capacity(stats.dirty_chunks.len());
        let mut chunk_displays = Vec::with_capacity(stats.dirty_chunks.len());
        let mut chunk_hazards = Vec::with_capacity(stats.dirty_chunks.len());

        // On first tick, push all chunks so clients get the complete Frostmarsh map
        let dirty_list: Vec<(u16, u16)> = if stats.tick == 1 {
            let cx_n = w.div_ceil(CHUNK_SIZE) as u16;
            let cy_n = h.div_ceil(CHUNK_SIZE) as u16;
            (0..cy_n)
                .flat_map(|cy| (0..cx_n).map(move |cx| (cx, cy)))
                .collect()
        } else {
            stats.dirty_chunks.clone()
        };

        for (cx, cy) in dirty_list {
            let payload = field.encode_chunk(cx, cy);
            chunk_ids.push(payload.chunk_id);
            chunk_cxs.push(payload.cx);
            chunk_cys.push(payload.cy);
            chunk_displays.push(payload.display_bytes);
            chunk_hazards.push(payload.hazard_bytes);
        }

        if let Err(e) = conn.reducers.commit_field_step(
            active_mode.clone(),
            physics_model.is_some(),
            eco_model.is_some(),
            stats.tick,
            infer_ms,
            p99,
            stats.proj_active_pct,
            stats.mass_error,
            stats.energy_error,
            processed_ids,
            chunk_ids,
            chunk_cxs,
            chunk_cys,
            chunk_displays,
            chunk_hazards,
            raft_xs,
            raft_ys,
            raft_radii,
            raft_masses,
        ) {
            log::warn!("commit_field_step failed: {:?}", e);
        }

        // Drive multiplayer AI bots at 5 Hz
        if stats.tick % 2 == 0 {
            step_bots(&conn, args.bots, stats.tick, w as f32, h as f32);
        }

        if stats.tick % 50 == 0 {
            log::info!(
                "tick={} mode={} infer={:.2}ms (p99={:.2}ms) proj={:.2}% mass_err={} energy_err={}",
                stats.tick,
                active_mode,
                infer_ms,
                p99,
                stats.proj_active_pct,
                stats.mass_error,
                stats.energy_error
            );
        }

        let elapsed = loop_start.elapsed();
        if elapsed < tick_period {
            std::thread::sleep(tick_period - elapsed);
        }
    }
}
