//! Wave 34A: Dual-Scale Locally-Flat Macro Planet Server Runtime (`--planet-macro`).
//!
//! Bridges two synchronized planetary scales on a live authoritative QUIC server:
//! - **Macro Scale (`6 × 64² × 4` = `98,304` cells):** Driven by `op=13` (`OP_PLANET_CLIMATE_STEP`)
//!   `PlanetaryClimateNet` on NVIDIA GB10 (`~5.9 ms` / `169 Hz` FP16) with automatic clean CPU
//!   `GlobalClimateShell` fallback, streaming `PlanetMacroClimateFace` overlays (`~20 KB/s`) and
//!   providing live Coriolis `(wind_u, wind_v)`, surface temperature, and cloud cover.
//! - **Micro Scale (`6 × 2048² × 64` = `1,610,612,736` (`1.61B`) virtual `1 m` voxels across `384` shards):**
//!   Lazily hydrates up to `4` `256×256×64` `1 m` `VolumeField` shards around human observers
//!   (`< 200 MB` RAM), exchanges fluxes bit-exactly across adjacent hydrated shard borders
//!   (`global_mass_error == 0 && global_energy_error == 0`), couples macro climate downscale +
//!   volcanic/Sky-Island upscale injections, and streams non-skippable `16×16×16` cubic sub-bricks
//!   via `PlanetShardBrick3D` / `PlanetShardBrick3DDelta`!

use bytes::Bytes;
use field_core::planet::{EDGE_COUNT, FACE_COUNT, FACE_POS_Y};
use field_core::planet_macro::{
    DualScalePlanet, MacroBiome, MacroPlanetConfig, MacroSkyIsland, PlanetShardKey, dot3_f64,
    face_uv_to_sphere_dir_f64, face_virtual_uv_f64, normalize3_f64, sample_macro_planet_column,
    shard_neighbor, sphere_dir_to_face_uv_f64,
};
use field_core::structural::{StructuralStepStats, fracture_buoyant_core};
use field_core::{
    BIOME_GLACIER_GORGE, C_NOM, H_CAP, MOBILE_SPECIES, NUM_MOBILE, NUM_REACTIONS, NUM_SPECIES,
    OUTPUT_CHANNELS, REACTIONS, RawStepOutput, SPECIES, SP_ICE, SP_OIL, SP_PLANT, SP_ROCK,
    SP_SMOKE, SP_SOIL, SP_STEAM, SP_WATER, SP_WOOD, T_AMB, T_MAX, T_MIN, VolumeField,
};
use frostmarsh_proto::{
    CHUNK_SIZE, CLASS_BURROWER, CLASS_SKYWING, CLASS_TIDEKIN, DEFAULT_BRICK3D_DELTA_DEADZONE,
    ServerDatagram, encode_brick3d_delta, encode_server_datagram, encode_subbrick3d_lz4,
    is_brick3d_delta_full_signal, is_subbrick_empty_sky, is_subbrick_skippable,
};
use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;
use std::sync::Arc;
use std::time::Instant;

pub const OP_PLANET_CLIMATE_STEP: u8 = 13;
pub const CLIMATE_GRID_N: usize = 64;
pub const CLIMATE_FACE_PIXELS: usize = CLIMATE_GRID_N * CLIMATE_GRID_N;
pub const CLIMATE_FACE_RGBA_BYTES: usize = CLIMATE_FACE_PIXELS * 4;
pub const CLIMATE_FACE_WIND_FLOATS: usize = CLIMATE_FACE_PIXELS * 2;
pub const CLIMATE_RGBA_BYTES: usize = FACE_COUNT * CLIMATE_FACE_RGBA_BYTES; // 98,304 bytes
pub const CLIMATE_WIND_BYTES: usize = FACE_COUNT * CLIMATE_FACE_WIND_FLOATS * 4; // 196,608 bytes
pub const CLIMATE_RESP_BYTES: usize = 8 + CLIMATE_RGBA_BYTES + CLIMATE_WIND_BYTES; // 294,920 bytes
pub const CLIMATE_REQ_PREFIX_BYTES: usize = 28;
pub const CLIMATE_REQ_INJ_STRIDE: usize = 20;

/// Wave 38B: Per-tick sub-brick re-encode budget (`<= 48` total: up to 32 observer/collapse/actuator
/// priority sub-bricks + 16 round-robin background sweep sub-bricks so the entire shard stays fresh
/// without burning CPU encoding 500–1024 sub-bricks every 10 Hz tick!).
pub const DEFAULT_SUBBRICK_REFRESH_BUDGET: usize = 48;
pub const DEFAULT_SUBBRICK_SWEEP_QUOTA: usize = 16;
/// Wave 38B: Active surface vertical slab height for `op=11` (`WorldModel3D-v8`) on tall 64-layer shards.
pub const OP11_ACTIVE_SLAB_LAYERS: usize = 16;
/// Wave 39B: Active horizontal ROI crop (`128×128`, 16-voxel brick aligned) centered on the primary
/// human observer for `op=11` (`WorldModel3D-v8`) — 4x smaller IPC and 4x fewer GPU 3D blocks!
pub const OP11_HERO_ROI_SIZE: usize = 128;
/// Wave 38B: Natural dynamic planetary world event cadence (~45 s at 10 Hz: `tick % 450 == 120`).
pub const MACRO_WORLD_EVENT_PERIOD_TICKS: u64 = 450;
pub const MACRO_WORLD_EVENT_PHASE_TICK: u64 = 120;
pub const MACRO_EVENT_VOLCANIC_SURGE: u8 = 0;
pub const MACRO_EVENT_COASTAL_SQUALL: u8 = 1;

pub type RawBrickArc = Arc<(Vec<u8>, Vec<u8>)>;
pub type ShardSubbrickKey = (PlanetShardKey, u16, u16, u16);

// =============================================================================
// 1. op=13 (`OP_PLANET_CLIMATE_STEP`) Binary Wire Codec
// =============================================================================

/// Localized AMR macro-upscale injection (volcanic eruption, wildfire heat burst, or shattered Sky-Island).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClimateInjection {
    pub face: u8,
    pub u: f32,
    pub v: f32,
    pub delta_temp: f32,
    pub delta_ash: f32,
}

/// Decoded output from one `op=13` (`OP_PLANET_CLIMATE_STEP`) `PlanetaryClimateNet` GPU RPC call.
#[derive(Clone, Debug)]
pub struct PlanetClimateOp13Response {
    pub infer_ms: f32,
    pub mean_temp_k: f32,
    /// `6` faces of `64×64×4` RGBA8 (`R=temp 240..320K, G=cloud_storm, B=ash_plume, A=ice_snow`).
    pub rgba_faces: [Vec<u8>; FACE_COUNT],
    /// `6` faces of `64×64×2` `f32` interleaved `(wind_u, wind_v)` tangent wind vectors.
    pub wind_uv_faces: [Vec<f32>; FACE_COUNT],
}

/// Pack an `op=13` (`OP_PLANET_CLIMATE_STEP`) request payload:
/// - 28-byte `<QfffII` prefix: `(seed: u64, sun_x: f32, sun_y: f32, sun_z: f32, steps: u32, n_inject: u32)`
/// - `n_inject` × 20-byte `<Bxxxffff` records: `(face: u8, pad[3], u: f32, v: f32, delta_temp: f32, delta_ash: f32)`
pub fn encode_planet_op13_request(
    seed: u64,
    sun_dir: [f32; 3],
    steps: u32,
    injections: &[ClimateInjection],
) -> Vec<u8> {
    let n_inj = injections.len().min(64);
    let mut out = Vec::with_capacity(CLIMATE_REQ_PREFIX_BYTES + n_inj * CLIMATE_REQ_INJ_STRIDE);
    out.extend_from_slice(&seed.to_le_bytes());
    out.extend_from_slice(&sun_dir[0].to_le_bytes());
    out.extend_from_slice(&sun_dir[1].to_le_bytes());
    out.extend_from_slice(&sun_dir[2].to_le_bytes());
    out.extend_from_slice(&steps.clamp(1, 4).to_le_bytes());
    out.extend_from_slice(&(n_inj as u32).to_le_bytes());
    for inj in injections.iter().take(n_inj) {
        out.push(inj.face % (FACE_COUNT as u8));
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(&inj.u.clamp(-1.0, 1.0).to_le_bytes());
        out.extend_from_slice(&inj.v.clamp(-1.0, 1.0).to_le_bytes());
        out.extend_from_slice(&inj.delta_temp.to_le_bytes());
        out.extend_from_slice(&inj.delta_ash.max(0.0).to_le_bytes());
    }
    out
}

/// Unpack an `op=13` (`OP_PLANET_CLIMATE_STEP`) response payload (`294,920` bytes):
/// - 8-byte `<ff`: `(infer_ms: f32, mean_temp_k: f32)`
/// - `98,304` bytes: `6 × 64 × 64 × 4` RGBA8 climate pack
/// - `196,608` bytes: `6 × 64 × 64 × 2` f32 surface `(wind_u, wind_v)` vectors
pub fn decode_planet_op13_response(buf: &[u8]) -> Result<PlanetClimateOp13Response, String> {
    if buf.len() != CLIMATE_RESP_BYTES {
        return Err(format!(
            "op=13 response len mismatch: got {} bytes, expected {}",
            buf.len(),
            CLIMATE_RESP_BYTES
        ));
    }
    let infer_ms = f32::from_le_bytes(buf[0..4].try_into().unwrap());
    let mean_temp_k = f32::from_le_bytes(buf[4..8].try_into().unwrap());

    let rgba_start = 8usize;
    let rgba_faces: [Vec<u8>; FACE_COUNT] = std::array::from_fn(|f| {
        let s = rgba_start + f * CLIMATE_FACE_RGBA_BYTES;
        buf[s..s + CLIMATE_FACE_RGBA_BYTES].to_vec()
    });

    let wind_start = rgba_start + CLIMATE_RGBA_BYTES;
    let wind_face_bytes = CLIMATE_FACE_WIND_FLOATS * 4;
    let wind_uv_faces: [Vec<f32>; FACE_COUNT] = std::array::from_fn(|f| {
        let s = wind_start + f * wind_face_bytes;
        let mut out = Vec::with_capacity(CLIMATE_FACE_WIND_FLOATS);
        for chunk in buf[s..s + wind_face_bytes].chunks_exact(4) {
            let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            out.push(if v.is_finite() { v } else { 0.0 });
        }
        out
    });

    Ok(PlanetClimateOp13Response {
        infer_ms: if infer_ms.is_finite() { infer_ms } else { 0.0 },
        mean_temp_k: if mean_temp_k.is_finite() {
            mean_temp_k
        } else {
            285.0
        },
        rgba_faces,
        wind_uv_faces,
    })
}

// =============================================================================
// 2. Scenic Coastal + Sky-Island Hero Spawn Site Solver (Matches planet_horizon_demo)
// =============================================================================

#[inline]
fn add_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn scale_f64(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn cross_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn tangent_basis_f64(dir: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let ref_up = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t1 = normalize3_f64(cross_f64(ref_up, dir));
    let t2 = normalize3_f64(cross_f64(dir, t1));
    (t1, t2)
}

/// Deterministic sunlit coastal beach + sapphire bay + Floating Sky-Island Hero Spawn site.
#[derive(Clone, Copy, Debug)]
pub struct MacroHeroSite {
    pub shard_key: PlanetShardKey,
    #[allow(dead_code)]
    pub beach_dir: [f64; 3],
    pub beach_gx: f32,
    pub beach_gy: f32,
    #[allow(dead_code)]
    pub beach_surf_z: f32,
    pub aim_x: f32,
    pub aim_y: f32,
    pub island_dir: [f64; 3],
}

/// Deterministically locate a sunlit coastal shard (preferring `+Z` / `+X` golden beach +
/// emerald meadow + nearby ocean bay) and position the Hero Floating Sky-Island right over the bay,
/// matching `planet_horizon_demo.rs`!
pub fn find_scenic_coastal_hero_site(cfg: MacroPlanetConfig, sun_dir: [f64; 3]) -> MacroHeroSite {
    let m = cfg.shards_per_face_axis.max(1);
    let n_virt = cfg.face_size_virtual();
    let s = cfg.shard_size.max(8);
    let r_surf = cfg.surface_radius_m();

    let mut best_shard = PlanetShardKey::new(4, (m / 2).min(m - 1), (m / 2).min(m - 1));
    let mut best_shard_score = f64::NEG_INFINITY;

    for f in 0..FACE_COUNT as u8 {
        for sy in 0..m {
            for sx in 0..m {
                let key = PlanetShardKey::new(f, sx, sy);
                let cdir = key.center_sphere_dir(m);
                let sdot = dot3_f64(cdir, sun_dir);
                if sdot < 0.56 || cdir[1].abs() > 0.42 {
                    continue;
                }
                let mut water_cnt = 0usize;
                let mut land_cnt = 0usize;
                let mut forest_cnt = 0usize;
                let mut max_z = 0.0f64;
                let mut has_rift = false;
                for iy in 0..8 {
                    let gy = (sy as usize * s) as f64 + (iy as f64 + 0.5) * (s as f64 / 8.0);
                    for ix in 0..8 {
                        let gx = (sx as usize * s) as f64 + (ix as f64 + 0.5) * (s as f64 / 8.0);
                        let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
                        let dir = face_uv_to_sphere_dir_f64(f as usize, u, v);
                        let col = sample_macro_planet_column(dir, cfg.seed);
                        if col.is_rift_core || col.is_rift_rim || col.is_polar {
                            has_rift = true;
                        }
                        if col.water_depth > 0.3 {
                            water_cnt += 1;
                        } else {
                            land_cnt += 1;
                        }
                        if col.biome == MacroBiome::PineForest
                            || col.biome == MacroBiome::EmeraldMeadow
                        {
                            forest_cnt += 1;
                        }
                        if col.surf_z > max_z {
                            max_z = col.surf_z;
                        }
                    }
                }
                if has_rift || water_cnt < 14 || land_cnt < 22 {
                    continue;
                }
                let frac_w = water_cnt as f64 / 64.0;
                let coast_balance = 1.0 - (frac_w - 0.42).abs() * 3.0;
                let relief = ((max_z - 25.0) / 14.0).clamp(0.0, 1.4);
                let face_bonus = if f == 4 || f == 0 { 0.35 } else { 0.0 };
                let score = coast_balance * 4.0
                    + relief * 2.5
                    + (forest_cnt as f64 / 64.0) * 2.0
                    + sdot * 2.5
                    + face_bonus;
                if score > best_shard_score {
                    best_shard_score = score;
                    best_shard = key;
                }
            }
        }
    }

    let gx0 = best_shard.sx as usize * s;
    let gy0 = best_shard.sy as usize * s;
    let margin = (s / 8).clamp(2, 28);
    let step = (s / 20).max(1).min(12);

    let mut best_cam_score = f64::NEG_INFINITY;
    let mut best_beach_dir = best_shard.center_sphere_dir(m);
    let mut best_beach_gx = (gx0 + s / 2) as f32 + 0.5;
    let mut best_beach_gy = (gy0 + s / 2) as f32 + 0.5;
    let mut best_beach_z = 26.0f64;
    let mut best_fwd = [1.0, 0.0, 0.0];

    for ly in (margin..s.saturating_sub(margin).max(margin + 1)).step_by(step) {
        for lx in (margin..s.saturating_sub(margin).max(margin + 1)).step_by(step) {
            let gx_f = (gx0 + lx) as f64 + 0.5;
            let gy_f = (gy0 + ly) as f64 + 0.5;
            let (u, v) = face_virtual_uv_f64(gx_f, gy_f, n_virt);
            let dir = face_uv_to_sphere_dir_f64(best_shard.face as usize, u, v);
            let col = sample_macro_planet_column(dir, cfg.seed);
            if col.surf_z < 24.9 || col.surf_z > 28.5 {
                continue;
            }
            let (t1, t2) = tangent_basis_f64(dir);
            for dir_i in 0..12 {
                let ang = (dir_i as f64 / 12.0) * std::f64::consts::TAU;
                let fwd =
                    normalize3_f64(add_f64(scale_f64(t1, ang.cos()), scale_f64(t2, ang.sin())));

                let mut bay_water = 0.0f64;
                let mut blocked_fg = false;
                for &d_m in &[12.0f64, 24.0, 42.0, 62.0, 82.0, 102.0, 130.0] {
                    let pd = normalize3_f64(add_f64(dir, scale_f64(fwd, d_m / r_surf)));
                    let pc = sample_macro_planet_column(pd, cfg.seed);
                    if d_m <= 28.0 && pc.surf_z > col.surf_z + 1.8 {
                        blocked_fg = true;
                        break;
                    }
                    if (35.0..=115.0).contains(&d_m) && pc.water_depth > 0.3 {
                        bay_water += 1.0;
                    }
                }
                if blocked_fg {
                    continue;
                }
                let right = normalize3_f64(cross_f64(fwd, dir));
                let mut flank_land = 0.0f64;
                for &side in &[-28.0f64, 28.0, -48.0, 48.0] {
                    let pd = normalize3_f64(add_f64(
                        dir,
                        add_f64(scale_f64(fwd, 40.0 / r_surf), scale_f64(right, side / r_surf)),
                    ));
                    let pc = sample_macro_planet_column(pd, cfg.seed);
                    if pc.surf_z > 25.5 && pc.surf_z < 36.0 {
                        flank_land += 1.0;
                    }
                }
                let sun_align = dot3_f64(fwd, sun_dir);
                let light_score = 1.0 - (sun_align - 0.15).abs();
                let sc = bay_water * 2.2 + flank_land * 1.1 + light_score * 1.8;
                if sc > best_cam_score {
                    best_cam_score = sc;
                    best_beach_dir = dir;
                    best_beach_gx = gx_f as f32;
                    best_beach_gy = gy_f as f32;
                    best_beach_z = col.surf_z;
                    best_fwd = fwd;
                }
            }
        }
    }

    // Place the Hero Floating Sky-Island `94 m` ahead and `14 m` to the right over the bay,
    // clamped cleanly inside `best_shard` so its BuoyantCore always hydrates inside the Hero Shard!
    let right = normalize3_f64(cross_f64(best_fwd, best_beach_dir));
    let raw_island_dir = normalize3_f64(add_f64(
        best_beach_dir,
        add_f64(
            scale_f64(best_fwd, 94.0 / r_surf),
            scale_f64(right, 14.0 / r_surf),
        ),
    ));
    let (u0, u1, v0, v1) = best_shard.uv_bounds(m);
    let pad_u = (u1 - u0) * 0.16;
    let pad_v = (v1 - v0) * 0.16;
    let (if_id, iu, iv) = sphere_dir_to_face_uv_f64(raw_island_dir);
    let island_dir = if if_id == best_shard.face as usize {
        face_uv_to_sphere_dir_f64(
            best_shard.face as usize,
            iu.clamp(u0 + pad_u, u1 - pad_u),
            iv.clamp(v0 + pad_v, v1 - pad_v),
        )
    } else {
        best_shard.center_sphere_dir(m)
    };

    // Project `best_fwd` onto the cube face's `(e_u, e_v)` for initial player aim vector
    let fr = field_core::planet::FACE_FRAMES[(best_shard.face as usize) % FACE_COUNT];
    let au = (best_fwd[0] * fr.e_u[0] as f64
        + best_fwd[1] * fr.e_u[1] as f64
        + best_fwd[2] * fr.e_u[2] as f64) as f32;
    let av = (best_fwd[0] * fr.e_v[0] as f64
        + best_fwd[1] * fr.e_v[1] as f64
        + best_fwd[2] * fr.e_v[2] as f64) as f32;
    let alen = au.hypot(av).max(1e-4);

    let z_scale = cfg.depth as f64 / 64.0;
    MacroHeroSite {
        shard_key: best_shard,
        beach_dir: best_beach_dir,
        beach_gx: best_beach_gx,
        beach_gy: best_beach_gy,
        beach_surf_z: ((best_beach_z * z_scale) + 1.2).clamp(2.0, cfg.depth as f64 - 2.0) as f32,
        aim_x: au / alen,
        aim_y: av / alen,
        island_dir,
    }
}

// =============================================================================
// 3. MacroPlanetRuntime: Live 1.61B Virtual-Voxel Planet + op=13 Climate Server
// =============================================================================

#[inline]
pub fn macro_sun_dir_from_angle(angle: f32) -> [f32; 3] {
    let sx = angle.cos();
    let sy = 0.34f32;
    let sz = angle.sin();
    let inv = 1.0 / (sx * sx + sy * sy + sz * sz).sqrt().max(1e-8);
    [sx * inv, sy * inv, sz * inv]
}

/// Gentle periodic natural planetary event triggered on `--planet-macro` (~45 s cadence).
#[derive(Clone, Debug, PartialEq)]
pub struct MacroWorldEvent {
    pub kind: u8,
    pub face: u8,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub title: String,
}

/// Wave 39B: Borrowed `op=11` (`WorldModel3D-v8`) Hero ROI slab (`128×128×16` or full shard crop)
/// spliced conservatively into `[x0..x0+roi_w, y0..y0+roi_h, z0..z0+horiz.len())`.
pub struct NeuralRoiSlab<'a> {
    pub x0: usize,
    pub y0: usize,
    pub roi_w: usize,
    pub roi_h: usize,
    pub z0: usize,
    pub horiz: &'a [RawStepOutput],
    #[allow(dead_code)]
    pub z_flux: Option<&'a [f32]>,
    pub op11_ms: f32,
    pub active_frac: f32,
}

/// Summary returned by one 10 Hz `MacroPlanetRuntime::step_macro_tick` call.
pub struct MacroPlanetTickOutcome {
    pub tick: u64,
    pub infer_ms: f32,
    pub op11_ms: f32,
    pub cpu_step_ms: f32,
    pub climate_infer_ms: f32,
    pub active_frac: f32,
    pub backend: String,
    pub neural_3d_vertical: bool,
    pub global_mass_error: i64,
    pub global_energy_error: i64,
    pub seam_mass_crossed: i64,
    pub fallen_solid_mass: i64,
    pub splashed_fluid_mass: i64,
    pub active_hydrated_shards: u16,
    pub dirty_subbricks: Vec<ShardSubbrickKey>,
    pub pending_dirty_subbricks: usize,
    pub newly_fracturing_cores: Vec<(u8, u8, String, f32, f32, f32)>,
    pub newly_impacted_cores: Vec<(u8, u8, String, f32, f32, f32)>,
    pub world_event: Option<MacroWorldEvent>,
}

/// Authoritative server runtime for Wave 34 `--planet-macro` mode (`R = 1,304 m`, `1.61B` virtual voxels).
pub struct MacroPlanetRuntime {
    pub planet: DualScalePlanet,
    pub hero_site: MacroHeroSite,
    pub max_hydrated_shards: usize,
    pub shard_last_touched: HashMap<PlanetShardKey, u64>,
    pub sun_angle: f32,
    pub sun_speed_rad_per_tick: f32,

    // Live 6×64² climate state (updated by op=13 PlanetaryClimateNet or CPU GlobalClimateShell fallback)
    pub climate_rgba_faces: [Vec<u8>; FACE_COUNT],
    pub climate_wind_uv_faces: [Vec<f32>; FACE_COUNT],
    pub climate_infer_ms: f32,
    pub climate_mean_temp_k: f32,
    pub climate_neural_active: bool,
    pub climate_face_stream_cursor: u8,
    pub op13_backoff: u32,
    pub pending_climate_injections: Vec<ClimateInjection>,

    // Per-shard 16×16×16 cubic sub-brick raw cache + Wave 38B priority/round-robin refresh queue
    pub subbrick_raw_cache: HashMap<ShardSubbrickKey, (u8, RawBrickArc)>,
    pub pending_dirty_subbricks: HashSet<ShardSubbrickKey>,
    pub priority_dirty_subbricks: HashSet<ShardSubbrickKey>,
    pub priority_focus_points: Vec<(u8, f32, f32)>,
    pub full_sync_cursor: usize,
    pub subbrick_sweep_cursor: usize,
    pub subbrick_refresh_budget: usize,
    pub subbrick_sweep_quota: usize,

    // Live backend & conservation telemetry
    pub last_infer_ms: f32,
    pub last_op11_ms: f32,
    pub last_cpu_step_ms: f32,
    pub last_active_frac: f32,
    pub last_backend: String,
    pub neural_3d_vertical: bool,
    pub neural_3d_recent_ticks: u8,
    pub seam_crossings_total: u64,
    pub climate_injections_total: u64,
    pub world_events_triggered: u64,

    // Wave 39B: Persistent zero-reallocation per-worker horizontal scratch pool (eliminates 48 MB/tick mmap churn!)
    layer_scratch_pool: Vec<LayerScratch>,
}

impl MacroPlanetRuntime {
    pub fn new(config: MacroPlanetConfig) -> Self {
        // Match `planet_horizon_demo.rs` sunlit hero sun angle (`[-0.72, 0.36, 0.59]`) at startup!
        let init_sun_angle = 0.59f32.atan2(-0.72f32).rem_euclid(TAU);
        let init_sun_f32 = macro_sun_dir_from_angle(init_sun_angle);
        let init_sun_f64 = normalize3_f64([
            init_sun_f32[0] as f64,
            init_sun_f32[1] as f64,
            init_sun_f32[2] as f64,
        ]);

        let mut planet = DualScalePlanet::new(config);
        planet.sun_dir = init_sun_f64;

        let hero_site = find_scenic_coastal_hero_site(config, init_sun_f64);

        // Place the iconic Hero Floating Sky-Island right over the sunlit coastal bay BEFORE hydrating!
        let cell_m = (std::f64::consts::FRAC_PI_2 * config.surface_radius_m())
            / (config.face_size_virtual().max(1) as f64);
        let isl_radius = 28.0f64.max(cell_m * 3.0);
        let core_radius = 9.2f64.max(cell_m * 1.45);
        if !planet.sky_islands.is_empty() {
            planet.sky_islands[0] = MacroSkyIsland {
                id: 0,
                name: "AETHER-CITADEL-PRIME".to_string(),
                center_dir: hero_site.island_dir,
                radius_m: isl_radius,
                z_bot: 46.0,
                z_top: 57.5,
                core_radius_m: core_radius,
                phase: 0.85,
                has_waterfall: true,
            };
        }

        // Hydrate the Hero Spawn Shard upfront (`~93 ms` for 256×256×64!)
        planet.hydrate_shard(hero_site.shard_key);
        let mut shard_last_touched = HashMap::new();
        shard_last_touched.insert(hero_site.shard_key, 0u64);

        let climate_rgba_faces = pack_climate_rgba_from_shell(&planet.climate);
        let climate_wind_uv_faces = pack_climate_wind_from_shell(&planet.climate);

        Self {
            planet,
            hero_site,
            max_hydrated_shards: 4,
            shard_last_touched,
            sun_angle: init_sun_angle,
            sun_speed_rad_per_tick: 0.006, // Smooth sunlit progression (~105s full planetary day at 10 Hz)
            climate_rgba_faces,
            climate_wind_uv_faces,
            climate_infer_ms: 0.0,
            climate_mean_temp_k: 285.0,
            climate_neural_active: false,
            climate_face_stream_cursor: 0,
            op13_backoff: 0,
            pending_climate_injections: Vec::new(),
            subbrick_raw_cache: HashMap::new(),
            pending_dirty_subbricks: HashSet::new(),
            priority_dirty_subbricks: HashSet::new(),
            priority_focus_points: Vec::new(),
            full_sync_cursor: 0,
            subbrick_sweep_cursor: 0,
            subbrick_refresh_budget: DEFAULT_SUBBRICK_REFRESH_BUDGET,
            subbrick_sweep_quota: DEFAULT_SUBBRICK_SWEEP_QUOTA,
            last_infer_ms: 0.0,
            last_op11_ms: 0.0,
            last_cpu_step_ms: 0.0,
            last_active_frac: 1.0,
            last_backend: "refsim-macro-planet".to_string(),
            neural_3d_vertical: false,
            neural_3d_recent_ticks: 0,
            seam_crossings_total: 0,
            climate_injections_total: 0,
            world_events_triggered: 0,
            layer_scratch_pool: Vec::new(),
        }
    }

    #[inline]
    pub fn config(&self) -> MacroPlanetConfig {
        self.planet.config
    }

    #[inline]
    pub fn face_size_virtual(&self) -> usize {
        self.planet.config.face_size_virtual()
    }

    #[inline]
    pub fn shard_size(&self) -> usize {
        self.planet.config.shard_size
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.planet.config.depth
    }

    #[inline]
    pub fn shard_chunks_xy(&self) -> u16 {
        self.planet.config.shard_size.div_ceil(CHUNK_SIZE).max(1) as u16
    }

    #[inline]
    pub fn subbricks_z(&self) -> u16 {
        self.planet.config.depth.div_ceil(CHUNK_SIZE).max(1) as u16
    }

    #[inline]
    pub fn sun_dir_f32(&self) -> [f32; 3] {
        [
            self.planet.sun_dir[0] as f32,
            self.planet.sun_dir[1] as f32,
            self.planet.sun_dir[2] as f32,
        ]
    }

    /// Map virtual face coordinates `(face, x, y)` (`x, y ∈ [0, face_size_virtual)`) to
    /// `(PlanetShardKey, local_x, local_y)`.
    #[inline]
    pub fn virtual_to_shard(&self, face: u8, x: f32, y: f32) -> (PlanetShardKey, f32, f32) {
        let m = self.planet.config.shards_per_face_axis.max(1) as usize;
        let s = self.planet.config.shard_size.max(1);
        let sf = s as f32;
        let max_v = (m * s) as f32 - 1e-3;
        let cx = x.clamp(0.0, max_v);
        let cy = y.clamp(0.0, max_v);
        let sx = ((cx / sf).floor() as usize).min(m - 1);
        let sy = ((cy / sf).floor() as usize).min(m - 1);
        let lx = (cx - (sx * s) as f32).clamp(0.0, sf - 1e-3);
        let ly = (cy - (sy * s) as f32).clamp(0.0, sf - 1e-3);
        (
            PlanetShardKey::new(face % (FACE_COUNT as u8), sx as u8, sy as u8),
            lx,
            ly,
        )
    }

    /// Convert local shard coordinates `(key, lx, ly)` back to virtual face coordinates `(face, gx, gy)`.
    #[inline]
    pub fn shard_local_to_virtual(&self, key: PlanetShardKey, lx: f32, ly: f32) -> (u8, f32, f32) {
        let s = self.planet.config.shard_size as f32;
        (
            key.face,
            (key.sx as f32) * s + lx,
            (key.sy as f32) * s + ly,
        )
    }

    #[inline]
    pub fn shard_at_virtual(&self, face: u8, x: f32, y: f32) -> Option<(&VolumeField, f32, f32)> {
        let (key, lx, ly) = self.virtual_to_shard(face, x, y);
        self.planet.shards.get(&key).map(|vf| (vf, lx, ly))
    }

    #[inline]
    pub fn shard_at_virtual_mut(
        &mut self,
        face: u8,
        x: f32,
        y: f32,
    ) -> Option<(&mut VolumeField, f32, f32)> {
        let (key, lx, ly) = self.virtual_to_shard(face, x, y);
        self.planet.shards.get_mut(&key).map(|vf| (vf, lx, ly))
    }

    /// Count intact levitating Sky-Island `BuoyantCore`s across all currently hydrated shards.
    pub fn active_sky_cores(&self) -> u8 {
        self.planet
            .shards
            .values()
            .flat_map(|vf| vf.structural.buoyant_cores.iter())
            .filter(|c| !c.collapsed && c.integrity > 0.05)
            .count()
            .min(255) as u8
    }

    /// Advance the planetary Day/Night terminator sun direction.
    pub fn advance_sun_dir(&mut self) -> [f32; 3] {
        self.sun_angle = (self.sun_angle + self.sun_speed_rad_per_tick).rem_euclid(TAU);
        let dir = macro_sun_dir_from_angle(self.sun_angle);
        self.planet.sun_dir = normalize3_f64([dir[0] as f64, dir[1] as f64, dir[2] as f64]);
        dir
    }

    /// Lazily hydrate each human observer's current shard plus any adjacent shard when an observer
    /// approaches within `border_margin_m` (default `64 m`) of a shard border, keeping at most
    /// `self.max_hydrated_shards` (`4`) shards resident in RAM and cleanly evicting distant
    /// uninhabited shards while maintaining bit-exact `0 / 0` conservation!
    pub fn ensure_hydrated_around_observers(
        &mut self,
        observers: &[(u8, f32, f32)],
    ) -> Vec<PlanetShardKey> {
        let tick = self.planet.tick;
        let m = self.planet.config.shards_per_face_axis;
        let s = self.planet.config.shard_size as f32;
        let margin = (s * 0.25).clamp(4.0, 64.0);

        let mut pinned: HashSet<PlanetShardKey> = HashSet::new();
        let mut desired: Vec<PlanetShardKey> = Vec::new();

        if observers.is_empty() {
            pinned.insert(self.hero_site.shard_key);
            desired.push(self.hero_site.shard_key);
        }

        for &(face, x, y) in observers {
            let (cur, lx, ly) = self.virtual_to_shard(face, x, y);
            pinned.insert(cur);
            desired.push(cur);
            if lx < margin {
                desired.push(shard_neighbor(cur, field_core::planet::EDGE_NEG_U, m).neighbor);
            }
            if lx > s - margin {
                desired.push(shard_neighbor(cur, field_core::planet::EDGE_POS_U, m).neighbor);
            }
            if ly < margin {
                desired.push(shard_neighbor(cur, field_core::planet::EDGE_NEG_V, m).neighbor);
            }
            if ly > s - margin {
                desired.push(shard_neighbor(cur, field_core::planet::EDGE_POS_V, m).neighbor);
            }
        }

        // Deduplicate while preserving priority (observer shards first, then border neighbors)
        let mut seen = HashSet::new();
        desired.retain(|k| seen.insert(*k));

        let mut newly_hydrated = Vec::new();
        for &k in &desired {
            self.shard_last_touched.insert(k, tick);
            if !self.planet.shards.contains_key(&k) {
                // Evict least-recently-used unpinned shard before hydrating if at capacity
                while self.planet.shards.len() >= self.max_hydrated_shards {
                    let evict_cand = self
                        .planet
                        .shards
                        .keys()
                        .copied()
                        .filter(|ek| !pinned.contains(ek) && !seen.contains(ek))
                        .min_by_key(|ek| {
                            let hero_bias = if *ek == self.hero_site.shard_key { 1u64 } else { 0u64 };
                            (hero_bias, self.shard_last_touched.get(ek).copied().unwrap_or(0))
                        })
                        .or_else(|| {
                            self.planet
                                .shards
                                .keys()
                                .copied()
                                .filter(|ek| !pinned.contains(ek))
                                .min_by_key(|ek| self.shard_last_touched.get(ek).copied().unwrap_or(0))
                        });
                    let Some(evict_key) = evict_cand else {
                        break;
                    };
                    let _ = self.planet.evict_shard(evict_key);
                    self.subbrick_raw_cache.retain(|&(sk, _, _, _), _| sk != evict_key);
                    self.pending_dirty_subbricks.retain(|&(sk, _, _, _)| sk != evict_key);
                    self.priority_dirty_subbricks.retain(|&(sk, _, _, _)| sk != evict_key);
                }
                if self.planet.shards.len() < self.max_hydrated_shards || pinned.contains(&k) {
                    if self.planet.hydrate_shard(k) {
                        newly_hydrated.push(k);
                    }
                }
            }
        }
        newly_hydrated
    }

    /// Enqueue a localized AMR macro-upscale injection `(face, u, v, delta_temp, delta_ash)`
    /// into the next `op=13` `PlanetaryClimateNet` call (and CPU fallback).
    pub fn enqueue_climate_injection_virtual(
        &mut self,
        face: u8,
        x: f32,
        y: f32,
        delta_temp: f32,
        delta_ash: f32,
    ) {
        let n_virt = self.face_size_virtual().max(1) as f32;
        let u = ((x / n_virt) * 2.0 - 1.0).clamp(-1.0, 1.0);
        let v = ((y / n_virt) * 2.0 - 1.0).clamp(-1.0, 1.0);
        if self.pending_climate_injections.len() < 32 {
            self.pending_climate_injections.push(ClimateInjection {
                face: face % (FACE_COUNT as u8),
                u,
                v,
                delta_temp: delta_temp.clamp(-2.0, 3.0),
                delta_ash: delta_ash.clamp(0.0, 4.0),
            });
            self.climate_injections_total = self.climate_injections_total.saturating_add(1);
        }
    }

    /// Returns `true` if `op=13` climate stepping should execute on this field tick (~3 Hz or when
    /// fresh AMR volcanic/wildfire injections are queued).
    #[inline]
    pub fn should_step_climate_this_tick(&self) -> bool {
        self.planet.tick == 0
            || self.planet.tick % 3 == 0
            || !self.pending_climate_injections.is_empty()
    }

    /// Apply either an `op=13` `PlanetaryClimateNet` response OR clean CPU `GlobalClimateShell`
    /// fallback (with Gaussian injection deposit so `/erupt` & heat pulses still create drifting
    /// ash/heat clouds even without the GPU daemon!).
    pub fn apply_climate_step(
        &mut self,
        op13_res: Option<Result<PlanetClimateOp13Response, String>>,
        applied_injections: &[ClimateInjection],
    ) {
        let mut used_neural_climate = false;
        if let Some(res) = op13_res {
            match res {
                Ok(resp) => {
                    self.climate_infer_ms = resp.infer_ms;
                    self.climate_mean_temp_k = resp.mean_temp_k;
                    self.climate_rgba_faces = resp.rgba_faces;
                    self.climate_wind_uv_faces = resp.wind_uv_faces;
                    self.climate_neural_active = true;
                    self.op13_backoff = 0;
                    sync_shell_from_climate_pack(
                        &mut self.planet.climate,
                        &self.climate_rgba_faces,
                        &self.climate_wind_uv_faces,
                    );
                    used_neural_climate = true;
                }
                Err(err) => {
                    log::debug!(
                        "CUDA op=13 (PlanetaryClimateNet) unavailable ({}); using CPU GlobalClimateShell fallback",
                        err
                    );
                    self.op13_backoff = 3;
                    self.climate_neural_active = false;
                }
            }
        } else if self.op13_backoff > 0 {
            self.op13_backoff -= 1;
        }

        if !used_neural_climate {
            let t0 = Instant::now();
            // Deposit any pending AMR upscale injections directly into CPU GlobalClimateShell
            let res = self.planet.climate.res;
            for inj in applied_injections {
                let f = (inj.face as usize) % FACE_COUNT;
                let cf = &mut self.planet.climate.faces[f];
                let cx = ((inj.u + 1.0) * 0.5 * res as f32).clamp(0.0, (res - 1) as f32);
                let cy = ((inj.v + 1.0) * 0.5 * res as f32).clamp(0.0, (res - 1) as f32);
                let r2 = 2.6f32 * 2.6f32;
                let ix0 = (cx.floor() as isize - 4).max(0) as usize;
                let ix1 = (cx.ceil() as isize + 4).min(res as isize - 1) as usize;
                let iy0 = (cy.floor() as isize - 4).max(0) as usize;
                let iy1 = (cy.ceil() as isize + 4).min(res as isize - 1) as usize;
                for y in iy0..=iy1 {
                    for x in ix0..=ix1 {
                        let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                        let g = (-d2 / (2.0 * r2)).exp();
                        let idx = y * res + x;
                        cf.temp_k[idx] =
                            (cf.temp_k[idx] + inj.delta_temp * 40.0 * g).clamp(240.0, 420.0);
                        cf.ash[idx] = (cf.ash[idx] + inj.delta_ash * 0.55 * g).clamp(0.0, 1.0);
                    }
                }
            }
            self.planet.climate.step_climate(self.planet.sun_dir, 0.3);
            self.climate_rgba_faces = pack_climate_rgba_from_shell(&self.planet.climate);
            self.climate_wind_uv_faces = pack_climate_wind_from_shell(&self.planet.climate);
            self.climate_infer_ms = t0.elapsed().as_secs_f32() * 1000.0;
            let mut sum_t = 0.0f64;
            let mut cnt = 0usize;
            for cf in &self.planet.climate.faces {
                for &t in &cf.temp_k {
                    sum_t += t as f64;
                    cnt += 1;
                }
            }
            self.climate_mean_temp_k = (sum_t / cnt.max(1) as f64) as f32;
        }
    }

    /// Register live human observer + recent actuator positions so sub-brick refresh prioritizes
    /// visible terrain/combat around active explorers first!
    pub fn set_priority_focus(&mut self, focus: &[(u8, f32, f32)]) {
        self.priority_focus_points.clear();
        self.priority_focus_points.extend_from_slice(focus);
    }

    /// Mark cubic sub-bricks surrounding a virtual world point `(face, x, y, z)` as urgent priority
    /// so actuators, eruptions, and Sky-Island collapses refresh on the very next tick.
    pub fn mark_priority_subbricks_around(
        &mut self,
        face: u8,
        x: f32,
        y: f32,
        z: f32,
        radius_m: f32,
    ) {
        let (sk, lx, ly) = self.virtual_to_shard(face, x, y);
        if !self.planet.shards.contains_key(&sk) {
            return;
        }
        let chunks_xy = self.shard_chunks_xy().max(1) as i32;
        let bz_dim = self.subbricks_z().max(1) as i32;
        let r_sb = ((radius_m.max(1.0) / CHUNK_SIZE as f32).ceil() as i32).clamp(0, 2);
        let cbx = ((lx as usize) / CHUNK_SIZE).min(chunks_xy as usize - 1) as i32;
        let cby = ((ly as usize) / CHUNK_SIZE).min(chunks_xy as usize - 1) as i32;
        let cbz = ((z.max(0.0) as usize) / CHUNK_SIZE).min(bz_dim as usize - 1) as i32;
        for bz in (cbz - 1).max(0)..=(cbz + 1).min(bz_dim - 1) {
            for by in (cby - r_sb).max(0)..=(cby + r_sb).min(chunks_xy - 1) {
                for bx in (cbx - r_sb).max(0)..=(cbx + r_sb).min(chunks_xy - 1) {
                    let key = (sk, bx as u16, by as u16, bz as u16);
                    self.priority_dirty_subbricks.insert(key);
                    self.pending_dirty_subbricks.insert(key);
                }
            }
        }
    }

    /// Active vertical layer range `[z0, z1)` for `op=11` (`WorldModel3D-v8`) on the Hero Shard.
    /// For tall `64`-layer macro shards, targets the `16`-layer ocean/beach/forest/combat surface
    /// band around `z_sea` (`4x` smaller I/O & `4x–8x` faster GPU sparse pass) while preserving exact
    /// 64-layer vertical gravity, buoyancy & Sky-Island waterfalls!
    #[inline]
    pub fn active_op11_slab_range(&self, depth: usize) -> (usize, usize) {
        if depth <= OP11_ACTIVE_SLAB_LAYERS {
            (0, depth)
        } else {
            let z_sea = self.planet.config.z_sea.min(depth.saturating_sub(1));
            let z0 = z_sea
                .saturating_sub(6)
                .min(depth.saturating_sub(OP11_ACTIVE_SLAB_LAYERS));
            (z0, z0 + OP11_ACTIVE_SLAB_LAYERS)
        }
    }

    /// Wave 39B: Compute the `16`-voxel aligned active Hero ROI `(x0, y0, roi_w, roi_h, z0, z1)`
    /// inside the Hero Shard centered on the primary human observer (falling back to the Hero beach
    /// spawn). On `256×256×64` shards this extracts a `128×128×16` crop (`4x` smaller IPC and `4x`
    /// fewer GPU `16×16×8` 3D bricks) aligned 1:1 to `SparseWorld3DV8Runner`'s brick grid!
    pub fn active_op11_hero_roi(
        &self,
        width: usize,
        height: usize,
        depth: usize,
        observers: &[(u8, f32, f32)],
    ) -> (usize, usize, usize, usize, usize, usize) {
        let (z0, z1) = self.active_op11_slab_range(depth);
        let roi_w = if width >= OP11_HERO_ROI_SIZE {
            OP11_HERO_ROI_SIZE
        } else {
            width
        };
        let roi_h = if height >= OP11_HERO_ROI_SIZE {
            OP11_HERO_ROI_SIZE
        } else {
            height
        };
        let hk = self.hero_site.shard_key;
        let (lx, ly) = observers
            .iter()
            .find_map(|&(f, x, y)| {
                let (sk, sx, sy) = self.virtual_to_shard(f, x, y);
                if sk == hk { Some((sx, sy)) } else { None }
            })
            .unwrap_or_else(|| {
                let (_, bx, by) = self.virtual_to_shard(
                    hk.face,
                    self.hero_site.beach_gx,
                    self.hero_site.beach_gy,
                );
                (bx, by)
            });

        let max_x0 = width.saturating_sub(roi_w);
        let max_y0 = height.saturating_sub(roi_h);
        let raw_x0 =
            (lx.round() as isize - (roi_w as isize / 2)).clamp(0, max_x0 as isize) as usize;
        let raw_y0 =
            (ly.round() as isize - (roi_h as isize / 2)).clamp(0, max_y0 as isize) as usize;
        let x0 = if max_x0 >= CHUNK_SIZE {
            (((raw_x0 + CHUNK_SIZE / 2) / CHUNK_SIZE) * CHUNK_SIZE).min(max_x0)
        } else {
            raw_x0
        };
        let y0 = if max_y0 >= CHUNK_SIZE {
            (((raw_y0 + CHUNK_SIZE / 2) / CHUNK_SIZE) * CHUNK_SIZE).min(max_y0)
        } else {
            raw_y0
        };
        (x0, y0, roi_w, roi_h, z0, z1)
    }

    /// Wave 38B: Trigger a gentle periodic natural planetary event every ~45 s (`tick % 450 == 120`),
    /// alternating between a **Volcanic Rift Geothermal Surge** and a **Coastal Storm Squall**,
    /// preserving exact `global_mass_error == 0 && global_energy_error == 0`.
    pub fn maybe_trigger_natural_world_event(&mut self) -> Option<MacroWorldEvent> {
        if self.planet.tick == 0
            || (self.planet.tick % MACRO_WORLD_EVENT_PERIOD_TICKS) != MACRO_WORLD_EVENT_PHASE_TICK
        {
            return None;
        }
        let event_idx = self.world_events_triggered;
        self.world_events_triggered = self.world_events_triggered.saturating_add(1);

        let hk = self.hero_site.shard_key;
        let face = hk.face;
        let n_virt = self.face_size_virtual() as f32;
        let s = self.shard_size() as f32;
        let shard_x0 = (hk.sx as f32) * s + 6.0;
        let shard_x1 = ((hk.sx as f32) + 1.0) * s - 6.0;
        let shard_y0 = (hk.sy as f32) * s + 6.0;
        let shard_y1 = ((hk.sy as f32) + 1.0) * s - 6.0;

        if event_idx % 2 == 0 {
            // 1. Volcanic Rift Geothermal Surge: warm ash plume into PlanetaryClimateNet + gentle steam vent
            let vx = (self.hero_site.beach_gx + 28.0).clamp(shard_x0, shard_x1);
            let vy = (self.hero_site.beach_gy - 22.0).clamp(shard_y0, shard_y1);
            let vz = self.find_surface_z(face, vx, vy);
            let pl = (vz.round() as usize).clamp(1, self.depth().saturating_sub(2));
            if let Some((vf, lx, ly)) = self.shard_at_virtual_mut(face, vx, vy) {
                vf.apply_physical_actuator(lx, ly, pl, 0.72, 0.0, 0.0, 0.62, 4.5);
            }
            self.mark_priority_subbricks_around(face, vx, vy, vz, 20.0);
            self.enqueue_climate_injection_virtual(face, vx, vy, 0.90, 1.65);

            // Immediately reflect the rising warm ash plume on the live climate overlay
            let f_idx = (face as usize) % FACE_COUNT;
            let cx = (vx / n_virt) * CLIMATE_GRID_N as f32;
            let cy = (vy / n_virt) * CLIMATE_GRID_N as f32;
            let rgba = &mut self.climate_rgba_faces[f_idx];
            let ix0 = (cx.floor() as isize - 4).max(0) as usize;
            let ix1 = (cx.ceil() as isize + 4).min(CLIMATE_GRID_N as isize - 1) as usize;
            let iy0 = (cy.floor() as isize - 4).max(0) as usize;
            let iy1 = (cy.ceil() as isize + 4).min(CLIMATE_GRID_N as isize - 1) as usize;
            for gy in iy0..=iy1 {
                for gx in ix0..=ix1 {
                    let d2 = (gx as f32 - cx).powi(2) + (gy as f32 - cy).powi(2);
                    let g = (-d2 / 10.0).exp();
                    let o4 = (gy * CLIMATE_GRID_N + gx) * 4;
                    rgba[o4] = ((rgba[o4] as f32) + 22.0 * g).clamp(0.0, 255.0).round() as u8;
                    rgba[o4 + 2] = ((rgba[o4 + 2] as f32) + 115.0 * g).clamp(0.0, 255.0).round() as u8;
                }
            }
            sync_shell_from_climate_pack(
                &mut self.planet.climate,
                &self.climate_rgba_faces,
                &self.climate_wind_uv_faces,
            );

            Some(MacroWorldEvent {
                kind: MACRO_EVENT_VOLCANIC_SURGE,
                face,
                x: vx,
                y: vy,
                z: vz,
                title: "VOLCANIC RIFT GEOTHERMAL SURGE — Warm Ash Plume Rising".to_string(),
            })
        } else {
            // 2. Coastal Storm Squall: intensify Coriolis clouds & winds over the coastal bay
            let sx = (self.hero_site.beach_gx + self.hero_site.aim_x * 46.0).clamp(8.0, n_virt - 8.0);
            let sy = (self.hero_site.beach_gy + self.hero_site.aim_y * 46.0).clamp(8.0, n_virt - 8.0);
            let sz = self.find_surface_z(face, sx, sy);
            self.enqueue_climate_injection_virtual(face, sx, sy, -0.45, 0.05);

            let f_idx = (face as usize) % FACE_COUNT;
            let cx = (sx / n_virt) * CLIMATE_GRID_N as f32;
            let cy = (sy / n_virt) * CLIMATE_GRID_N as f32;
            let rgba = &mut self.climate_rgba_faces[f_idx];
            let wuv = &mut self.climate_wind_uv_faces[f_idx];
            let ix0 = (cx.floor() as isize - 6).max(0) as usize;
            let ix1 = (cx.ceil() as isize + 6).min(CLIMATE_GRID_N as isize - 1) as usize;
            let iy0 = (cy.floor() as isize - 6).max(0) as usize;
            let iy1 = (cy.ceil() as isize + 6).min(CLIMATE_GRID_N as isize - 1) as usize;
            for gy in iy0..=iy1 {
                for gx in ix0..=ix1 {
                    let dx = gx as f32 - cx;
                    let dy = gy as f32 - cy;
                    let d2 = dx * dx + dy * dy;
                    let g = (-d2 / 18.0).exp();
                    let pix = gy * CLIMATE_GRID_N + gx;
                    let o4 = pix * 4;
                    let o2 = pix * 2;
                    // Cool slightly & intensify storm clouds
                    rgba[o4] = ((rgba[o4] as f32) - 10.0 * g).clamp(0.0, 255.0).round() as u8;
                    rgba[o4 + 1] = ((rgba[o4 + 1] as f32) + 110.0 * g).clamp(0.0, 255.0).round() as u8;
                    // Coriolis cyclone swirl + coastal gust boost
                    let inv_r = 1.0 / (d2.sqrt() + 1.5);
                    wuv[o2] = (wuv[o2] + (2.4 - dy * 1.4 * inv_r) * g).clamp(-14.0, 14.0);
                    wuv[o2 + 1] = (wuv[o2 + 1] + (1.5 + dx * 1.4 * inv_r) * g).clamp(-14.0, 14.0);
                }
            }
            sync_shell_from_climate_pack(
                &mut self.planet.climate,
                &self.climate_rgba_faces,
                &self.climate_wind_uv_faces,
            );

            Some(MacroWorldEvent {
                kind: MACRO_EVENT_COASTAL_SQUALL,
                face,
                x: sx,
                y: sy,
                z: sz,
                title: "COASTAL STORM SQUALL — Coriolis Winds & Cloud Front Building".to_string(),
            })
        }
    }

    /// Run one full 10 Hz macro-planet physics tick (backward-compatible wrapper around
    /// `step_macro_tick_slab` taking full-depth `Vec<RawStepOutput>`).
    #[allow(dead_code, clippy::type_complexity)]
    pub fn step_macro_tick(
        &mut self,
        op13_res: Option<Result<PlanetClimateOp13Response, String>>,
        applied_injections: &[ClimateInjection],
        hero_neural_op11: Option<(Vec<RawStepOutput>, Option<Vec<f32>>, f32, f32)>,
    ) -> MacroPlanetTickOutcome {
        match hero_neural_op11 {
            Some((ref horiz, ref zf, ms, afrac)) => self.step_macro_tick_slab(
                op13_res,
                applied_injections,
                Some((0, horiz.as_slice(), zf.as_deref(), ms, afrac)),
            ),
            None => self.step_macro_tick_slab(op13_res, applied_injections, None),
        }
    }

    /// Run one full 10 Hz macro-planet physics tick using zero-copy borrowed `op=11` full-width slab slices.
    #[allow(clippy::type_complexity)]
    pub fn step_macro_tick_slab(
        &mut self,
        op13_res: Option<Result<PlanetClimateOp13Response, String>>,
        applied_injections: &[ClimateInjection],
        hero_neural_slab: Option<(usize, &[RawStepOutput], Option<&[f32]>, f32, f32)>,
    ) -> MacroPlanetTickOutcome {
        let roi_opt = hero_neural_slab.and_then(|(z0, horiz, zf, ms, afrac)| {
            let first = horiz.first()?;
            Some(NeuralRoiSlab {
                x0: 0,
                y0: 0,
                roi_w: first.width,
                roi_h: first.height,
                z0,
                horiz,
                z_flux: zf,
                op11_ms: ms,
                active_frac: afrac,
            })
        });
        self.step_macro_tick_roi_slab(op13_res, applied_injections, roi_opt)
    }

    /// Wave 39B: Run one full 10 Hz macro-planet physics tick using an optional `128×128×16` Hero ROI
    /// `op=11` neural slab (zero perimeter flux leak), parallel active-chunk horizontal + red-black
    /// 3D vertical physics, budgeted priority + round-robin sub-brick refresh (`<= 48`/tick),
    /// natural planetary world events, and guaranteed **`global_mass_error == 0 && global_energy_error == 0`**!
    pub fn step_macro_tick_roi_slab(
        &mut self,
        op13_res: Option<Result<PlanetClimateOp13Response, String>>,
        applied_injections: &[ClimateInjection],
        hero_neural_roi: Option<NeuralRoiSlab<'_>>,
    ) -> MacroPlanetTickOutcome {
        let t0 = Instant::now();
        let _sun_dir = self.advance_sun_dir();

        if op13_res.is_some()
            || !applied_injections.is_empty()
            || self.should_step_climate_this_tick()
        {
            self.apply_climate_step(op13_res, applied_injections);
        }

        let mut used_neural_3d = false;
        let mut op11_ms = 0.0f32;
        let mut active_frac = self.last_active_frac;
        let z_sea = self.planet.config.z_sea;

        let mut keys: Vec<PlanetShardKey> = self.planet.shards.keys().copied().collect();
        keys.sort();

        // 1. Step hydrated shards in parallel across vertical layers (zero-copy 128×128×16 op=11 ROI
        //    slab on hero shard + active-chunk RefSim + parallel red-black 3D vertical gravity/buoyancy!)
        let hero_key = self.hero_site.shard_key;
        if let Some(roi) = hero_neural_roi {
            if let Some(vf) = self.planet.shards.get_mut(&hero_key) {
                let roi_valid = !roi.horiz.is_empty()
                    && roi.roi_w > 0
                    && roi.roi_h > 0
                    && roi.x0 + roi.roi_w <= vf.width
                    && roi.y0 + roi.roi_h <= vf.height
                    && roi.z0 + roi.horiz.len() <= vf.depth
                    && roi
                        .horiz
                        .first()
                        .map(|h| {
                            h.width == roi.roi_w
                                && h.height == roi.roi_h
                                && h.data.len() == OUTPUT_CHANNELS * roi.roi_w * roi.roi_h
                        })
                        .unwrap_or(false);
                if roi_valid {
                    let dirty_chunks = parallel_step_shard_volume(
                        vf,
                        z_sea,
                        Some(&roi),
                        &mut self.layer_scratch_pool,
                    );
                    for (layer, cx, cy) in dirty_chunks {
                        let bz = (layer as usize / CHUNK_SIZE) as u16;
                        self.pending_dirty_subbricks.insert((hero_key, cx, cy, bz));
                    }
                    used_neural_3d = true;
                    op11_ms = roi.op11_ms;
                    active_frac = roi.active_frac.clamp(0.0, 1.0);
                    self.neural_3d_recent_ticks = 6;
                }
            }
        }
        if !used_neural_3d && self.neural_3d_recent_ticks > 0 {
            self.neural_3d_recent_ticks -= 1;
        }

        let remaining_keys: Vec<PlanetShardKey> = keys
            .iter()
            .copied()
            .filter(|&k| !(used_neural_3d && k == hero_key))
            .collect();

        if remaining_keys.len() == 1 {
            let k = remaining_keys[0];
            if let Some(vf) = self.planet.shards.get_mut(&k) {
                let dirty_chunks =
                    parallel_step_shard_volume(vf, z_sea, None, &mut self.layer_scratch_pool);
                for (layer, cx, cy) in dirty_chunks {
                    let bz = (layer as usize / CHUNK_SIZE) as u16;
                    self.pending_dirty_subbricks.insert((k, cx, cy, bz));
                }
            }
        } else if remaining_keys.len() > 1 {
            let mut vals: Vec<(PlanetShardKey, VolumeField)> = remaining_keys
                .iter()
                .filter_map(|k| self.planet.shards.remove(k).map(|v| (*k, v)))
                .collect();
            let mut shard_dirty: Vec<(PlanetShardKey, Vec<(u8, u16, u16)>)> =
                Vec::with_capacity(vals.len());
            std::thread::scope(|sc| {
                let mut handles = Vec::with_capacity(vals.len());
                for (k, vf) in &mut vals {
                    let kk = *k;
                    handles.push(sc.spawn(move || {
                        let mut local_pool = Vec::new();
                        let dirty = parallel_step_shard_volume(vf, z_sea, None, &mut local_pool);
                        (kk, dirty)
                    }));
                }
                for h in handles {
                    shard_dirty.push(h.join().expect("macro shard step panicked"));
                }
            });
            for (k, v) in vals {
                self.planet.shards.insert(k, v);
            }
            for (k, chunks) in shard_dirty {
                for (layer, cx, cy) in chunks {
                    let bz = (layer as usize / CHUNK_SIZE) as u16;
                    self.pending_dirty_subbricks.insert((k, cx, cy, bz));
                }
            }
        }

        // 2. Exact antisymmetric cross-shard border conservation exchange
        let edge_stats = self.planet.exchange_hydrated_shard_borders();

        // 3. Macro -> Micro AMR climate downscale onto all hydrated shards (fast energy-skip exact equivalent)
        for &k in &keys {
            apply_climate_to_shard_fast(&mut self.planet, k);
        }

        // 4. Parallel structural mechanics (`RadialInnerShell` + Sky-Island `BuoyantCore` collapses)
        let mut struct_results: Vec<(PlanetShardKey, StructuralStepStats)> =
            Vec::with_capacity(keys.len());
        if keys.len() <= 1 {
            for &k in &keys {
                if let Some(vf) = self.planet.shards.get_mut(&k) {
                    struct_results.push((k, vf.step_structural_mechanics()));
                }
            }
        } else {
            let mut vals: Vec<(PlanetShardKey, VolumeField)> = keys
                .iter()
                .filter_map(|k| self.planet.shards.remove(k).map(|v| (*k, v)))
                .collect();
            std::thread::scope(|sc| {
                let mut handles = Vec::with_capacity(vals.len());
                for (k, vf) in &mut vals {
                    let kk = *k;
                    handles.push(sc.spawn(move || (kk, vf.step_structural_mechanics())));
                }
                for h in handles {
                    struct_results.push(h.join().expect("macro structural worker panicked"));
                }
            });
            for (k, v) in vals {
                self.planet.shards.insert(k, v);
            }
        }

        let mut fallen_solid_mass = 0i64;
        let mut splashed_fluid_mass = 0i64;
        let mut newly_fracturing_cores = Vec::new();
        let mut newly_impacted_cores = Vec::new();
        let chunks_xy = self.shard_chunks_xy();
        let bz_dim = self.subbricks_z();

        for (sk, sr) in struct_results {
            fallen_solid_mass += sr.fallen_solid_mass;
            splashed_fluid_mass += sr.splashed_fluid_mass;
            if sr.fallen_solid_mass > 0 || sr.splashed_fluid_mass > 0 {
                if let Some(vf) = self.planet.shards.get(&sk) {
                    for core in &vf.structural.buoyant_cores {
                        if core.collapsed || core.integrity < 0.99 {
                            let bx =
                                ((core.cx as usize) / CHUNK_SIZE).min(chunks_xy as usize - 1) as u16;
                            let by =
                                ((core.cy as usize) / CHUNK_SIZE).min(chunks_xy as usize - 1) as u16;
                            for bz in 0..bz_dim {
                                let center = (sk, bx, by, bz);
                                self.priority_dirty_subbricks.insert(center);
                                self.pending_dirty_subbricks.insert(center);
                                if bx > 0 {
                                    let k = (sk, bx - 1, by, bz);
                                    self.priority_dirty_subbricks.insert(k);
                                    self.pending_dirty_subbricks.insert(k);
                                }
                                if bx + 1 < chunks_xy {
                                    let k = (sk, bx + 1, by, bz);
                                    self.priority_dirty_subbricks.insert(k);
                                    self.pending_dirty_subbricks.insert(k);
                                }
                                if by > 0 {
                                    let k = (sk, bx, by - 1, bz);
                                    self.priority_dirty_subbricks.insert(k);
                                    self.pending_dirty_subbricks.insert(k);
                                }
                                if by + 1 < chunks_xy {
                                    let k = (sk, bx, by + 1, bz);
                                    self.priority_dirty_subbricks.insert(k);
                                    self.pending_dirty_subbricks.insert(k);
                                }
                            }
                        }
                    }
                }
            }
            for (cid, cname, lx, ly, cz) in sr.newly_fracturing_cores {
                let (f, gx, gy) = self.shard_local_to_virtual(sk, lx, ly);
                newly_fracturing_cores.push((f, cid, cname, gx, gy, cz));
            }
            for (cid, cname, lx, ly, cz) in sr.newly_impacted_cores {
                let (f, gx, gy) = self.shard_local_to_virtual(sk, lx, ly);
                self.enqueue_climate_injection_virtual(f, gx, gy, 0.55, 1.40);
                newly_impacted_cores.push((f, cid, cname, gx, gy, cz));
            }
        }

        // Advance tick & trigger periodic natural planetary world events (~45 s cadence)
        self.planet.tick += 1;
        let world_event = self.maybe_trigger_natural_world_event();

        // 5. Wave 38B: Budgeted Priority + Round-Robin Sub-Brick Refresh (<= 48/tick)
        //    Prune evicted shards first, then select:
        //    - Tier 1: Urgent structural collapses / actuators / eruptions (`priority_dirty_subbricks`)
        //    - Tier 2: Observer-proximate dirty sub-bricks up to `budget - sweep_quota` (32)
        //    - Tier 3: Background round-robin sweep (`sweep_quota` = 16 + any unused priority slots)
        let shards_ref = &self.planet.shards;
        self.pending_dirty_subbricks
            .retain(|&(sk, _, _, _)| shards_ref.contains_key(&sk));
        self.priority_dirty_subbricks
            .retain(|&(sk, _, _, _)| shards_ref.contains_key(&sk));

        let budget = self.subbrick_refresh_budget.max(1);
        let sweep_quota = self.subbrick_sweep_quota.min(budget);
        let prio_cap = budget.saturating_sub(sweep_quota).max(1);

        let shard_sz = self.shard_size() as f32;
        let half_chunk = (CHUNK_SIZE as f32) * 0.5;
        let fallback_focus = [(
            self.hero_site.shard_key.face,
            self.hero_site.beach_gx,
            self.hero_site.beach_gy,
        )];
        let focus_pts: &[(u8, f32, f32)] = if self.priority_focus_points.is_empty() {
            &fallback_focus
        } else {
            &self.priority_focus_points
        };
        let score_dist2 = |&(sk, bx, by, bz): &ShardSubbrickKey| -> u64 {
            let gx = (sk.sx as f32) * shard_sz + (bx as f32) * (CHUNK_SIZE as f32) + half_chunk;
            let gy = (sk.sy as f32) * shard_sz + (by as f32) * (CHUNK_SIZE as f32) + half_chunk;
            let mut best = f32::INFINITY;
            for &(ff, fx, fy) in focus_pts {
                let d2 = if ff == sk.face {
                    let dx = gx - fx;
                    let dy = gy - fy;
                    dx * dx + dy * dy
                } else {
                    1.0e9
                };
                if d2 < best {
                    best = d2;
                }
            }
            // Prefer active surface/sky bands slightly on ties
            let bz_bias = if bz == 1 || bz == 2 { 0u64 } else { 8u64 };
            ((best.min(1.0e12) as u64) << 4) | bz_bias
        };

        let mut selected: Vec<ShardSubbrickKey> = Vec::with_capacity(budget);
        let mut selected_set: HashSet<ShardSubbrickKey> = HashSet::with_capacity(budget * 2);

        // Tier 1: Urgent structural/actuator/eruption sub-bricks
        if !self.priority_dirty_subbricks.is_empty() {
            let mut prio_list: Vec<(u64, ShardSubbrickKey)> = self
                .priority_dirty_subbricks
                .iter()
                .map(|k| (score_dist2(k), *k))
                .collect();
            prio_list.sort_unstable_by_key(|&(d, k)| (d, k));
            for (_, k) in prio_list.into_iter().take(prio_cap) {
                self.priority_dirty_subbricks.remove(&k);
                self.pending_dirty_subbricks.remove(&k);
                if selected_set.insert(k) {
                    selected.push(k);
                }
            }
        }

        // Tier 2: Observer-proximate dirty sub-bricks up to `prio_cap`
        if selected.len() < prio_cap && !self.pending_dirty_subbricks.is_empty() {
            let need = prio_cap - selected.len();
            let mut near_list: Vec<(u64, ShardSubbrickKey)> = self
                .pending_dirty_subbricks
                .iter()
                .map(|k| (score_dist2(k), *k))
                .collect();
            if near_list.len() > need {
                near_list.select_nth_unstable_by_key(need, |&(d, k)| (d, k));
                near_list.truncate(need);
            }
            near_list.sort_unstable_by_key(|&(d, k)| (d, k));
            for (_, k) in near_list {
                self.pending_dirty_subbricks.remove(&k);
                if selected_set.insert(k) {
                    selected.push(k);
                }
            }
        }

        // Tier 3: Background round-robin sweep across remaining dirty sub-bricks + surface ring
        let remaining_slots = budget.saturating_sub(selected.len());
        if remaining_slots > 0 {
            if !self.pending_dirty_subbricks.is_empty() {
                let mut bg_list: Vec<ShardSubbrickKey> =
                    self.pending_dirty_subbricks.iter().copied().collect();
                bg_list.sort_unstable();
                let n = bg_list.len();
                let start = self.subbrick_sweep_cursor % n;
                let take_n = remaining_slots.min(n);
                for i in 0..take_n {
                    let k = bg_list[(start + i) % n];
                    self.pending_dirty_subbricks.remove(&k);
                    if selected_set.insert(k) {
                        selected.push(k);
                    }
                }
                self.subbrick_sweep_cursor = self.subbrick_sweep_cursor.wrapping_add(take_n);
            }

            // Guarantee continuous round-robin climate freshness across hydrated surface sub-bricks
            let total_2d = (chunks_xy as usize) * (chunks_xy as usize);
            let surf_bz = (self.planet.config.z_sea / CHUNK_SIZE).min(bz_dim as usize - 1) as u16;
            if total_2d > 0 && !keys.is_empty() {
                let extra_sweep = budget.saturating_sub(selected.len()).min(sweep_quota.min(4));
                for step in 0..extra_sweep {
                    let sk = keys[(self.full_sync_cursor + step) % keys.len()];
                    let ci = (self.full_sync_cursor + step) % total_2d;
                    let bx = (ci % chunks_xy as usize) as u16;
                    let by = (ci / chunks_xy as usize) as u16;
                    let k = (sk, bx, by, surf_bz);
                    if selected_set.insert(k) {
                        selected.push(k);
                    }
                }
                self.full_sync_cursor = (self.full_sync_cursor + extra_sweep.max(1)) % total_2d;
            }
        }

        // Parallel encode the selected <= 48 sub-bricks
        self.refresh_shard_subbricks_batch(&selected);
        let dirty_subbricks = selected;
        let pending_dirty_count = self.pending_dirty_subbricks.len();

        let (global_mass_error, global_energy_error) =
            parallel_global_conservation_errors(&self.planet);

        let cpu_step_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let neural_active = used_neural_3d || self.neural_3d_recent_ticks > 0;
        let infer_ms = op11_ms + cpu_step_ms;
        let backend = match (neural_active, self.climate_neural_active) {
            (true, true) => "cuda-fp16-macro-climate+v8".to_string(),
            (false, true) => "cuda-fp16-climate+refsim-1m".to_string(),
            (true, false) => "cuda-fp16-v8+cpu-climate".to_string(),
            (false, false) => "refsim-macro-planet".to_string(),
        };
        self.last_infer_ms = infer_ms;
        self.last_op11_ms = op11_ms;
        self.last_cpu_step_ms = cpu_step_ms;
        self.last_active_frac = active_frac;
        self.last_backend = backend.clone();
        self.neural_3d_vertical = neural_active;

        MacroPlanetTickOutcome {
            tick: self.planet.tick,
            infer_ms,
            op11_ms,
            cpu_step_ms,
            climate_infer_ms: self.climate_infer_ms,
            active_frac,
            backend,
            neural_3d_vertical: neural_active,
            global_mass_error,
            global_energy_error,
            seam_mass_crossed: edge_stats.seam_mass_crossed,
            fallen_solid_mass,
            splashed_fluid_mass,
            active_hydrated_shards: self.planet.shards.len().min(u16::MAX as usize) as u16,
            dirty_subbricks,
            pending_dirty_subbricks: pending_dirty_count,
            newly_fracturing_cores,
            newly_impacted_cores,
            world_event,
        }
    }

    /// Encode a batch of `<= 48` sub-bricks in parallel across CPU cores when `keys.len() >= 8`.
    pub fn refresh_shard_subbricks_batch(&mut self, keys: &[ShardSubbrickKey]) {
        if keys.is_empty() {
            return;
        }
        if keys.len() < 8 {
            for &(sk, bx, by, bz) in keys {
                let _ = self.refresh_shard_subbrick(sk, bx, by, bz);
            }
            return;
        }
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .clamp(2, 8)
            .min(keys.len());
        let chunk_sz = keys.len().div_ceil(n_threads);
        let shards = &self.planet.shards;
        let mut encoded: Vec<(ShardSubbrickKey, u8, RawBrickArc)> =
            Vec::with_capacity(keys.len());
        std::thread::scope(|sc| {
            let mut handles = Vec::with_capacity(n_threads);
            for chunk in keys.chunks(chunk_sz) {
                handles.push(sc.spawn(move || {
                    let mut local = Vec::with_capacity(chunk.len());
                    for &(sk, bx, by, bz) in chunk {
                        if let Some(vf) = shards.get(&sk) {
                            let (dm, at, sd) =
                                vf.encode_3d_subbrick_16(bx as usize, by as usize, bz as usize);
                            local.push(((sk, bx, by, bz), sd, Arc::new((dm, at))));
                        }
                    }
                    local
                }));
            }
            for h in handles {
                encoded.extend(h.join().expect("subbrick batch encode panicked"));
            }
        });
        for (key, sd, raw) in encoded {
            self.subbrick_raw_cache.insert(key, (sd, raw));
        }
    }

    // =========================================================================
    // 4. Local Climate Sampling, Surface Queries, Shrines, Spawns & Commands
    // =========================================================================

    /// Sample live `(wind_u, wind_v, temp_c, cloud_cover)` at virtual coordinate `(face, x, y)`
    /// from the `6 × 64²` `PlanetaryClimateNet` / `GlobalClimateShell` buffers using smooth bilinear
    /// interpolation aligned 1:1 with `face_virtual_uv_f64` plus a subtle 3×3 overhead sky-dome
    /// cloud-cover blend!
    pub fn sample_local_climate(&self, face: u8, x: f32, y: f32) -> (f32, f32, f32, f32) {
        let f = (face as usize) % FACE_COUNT;
        let n_virt = self.face_size_virtual().max(1) as f64;
        let (u, v) = face_virtual_uv_f64(x as f64, y as f64, n_virt as usize);
        let max_idx = (CLIMATE_GRID_N - 1) as f32;
        // Exact inverse of GlobalClimateShell cell centers `u = ((ix + 0.5) / N) * 2.0 - 1.0`
        let fx = (((u + 1.0) * 0.5) as f32 * CLIMATE_GRID_N as f32 - 0.5).clamp(0.0, max_idx);
        let fy = (((v + 1.0) * 0.5) as f32 * CLIMATE_GRID_N as f32 - 0.5).clamp(0.0, max_idx);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(CLIMATE_GRID_N - 1);
        let y1 = (y0 + 1).min(CLIMATE_GRID_N - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;

        let rgba = &self.climate_rgba_faces[f];
        let wuv = &self.climate_wind_uv_faces[f];

        let sample_pix = |ix: usize, iy: usize| -> (f32, f32, f32, f32) {
            let pix = iy * CLIMATE_GRID_N + ix;
            let o4 = pix * 4;
            let o2 = pix * 2;
            let temp_k = 240.0 + (rgba[o4] as f32 / 255.0) * 80.0;
            let cloud = rgba[o4 + 1] as f32 / 255.0;
            (wuv[o2], wuv[o2 + 1], temp_k - 273.15, cloud)
        };

        let p00 = sample_pix(x0, y0);
        let p10 = sample_pix(x1, y0);
        let p01 = sample_pix(x0, y1);
        let p11 = sample_pix(x1, y1);

        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;

        let wind_u = p00.0 * w00 + p10.0 * w10 + p01.0 * w01 + p11.0 * w11;
        let wind_v = p00.1 * w00 + p10.1 * w10 + p01.1 * w01 + p11.1 * w11;
        let temp_c = p00.2 * w00 + p10.2 * w10 + p01.2 * w01 + p11.2 * w11;
        let cloud_pt = p00.3 * w00 + p10.3 * w10 + p01.3 * w01 + p11.3 * w11;

        // Subtle 3×3 overhead sky-dome cloud average (~96m visible sky canopy)
        let cx = fx.round() as isize;
        let cy = fy.round() as isize;
        let mut cloud_sum = 0.0f32;
        let mut cloud_cnt = 0.0f32;
        for dy in -1..=1 {
            let iy = (cy + dy).clamp(0, CLIMATE_GRID_N as isize - 1) as usize;
            for dx in -1..=1 {
                let ix = (cx + dx).clamp(0, CLIMATE_GRID_N as isize - 1) as usize;
                let o4 = (iy * CLIMATE_GRID_N + ix) * 4;
                cloud_sum += rgba[o4 + 1] as f32 / 255.0;
                cloud_cnt += 1.0;
            }
        }
        let cloud_dome = cloud_sum / cloud_cnt.max(1.0);
        let cloud_cover = (0.65 * cloud_pt + 0.35 * cloud_dome).clamp(0.0, 1.0);

        (wind_u, wind_v, temp_c, cloud_cover)
    }

    /// Find highest solid planetary surface altitude `z` below the Sky-Island band at virtual `(face, x, y)`.
    /// Uses the hydrated `1 m` `VolumeField` when resident, else evaluates `sample_macro_planet_column` in O(1)!
    pub fn find_surface_z(&self, face: u8, x: f32, y: f32) -> f32 {
        let d = self.depth();
        let z_scale = d as f32 / 64.0;
        let sky_floor = ((44.0 * z_scale).round() as usize).clamp(4, d.saturating_sub(2));

        if let Some((vf, lx, ly)) = self.shard_at_virtual(face, x, y) {
            let xi = (lx.floor() as i32).clamp(0, (vf.width as i32) - 1) as usize;
            let yi = (ly.floor() as i32).clamp(0, (vf.height as i32) - 1) as usize;
            let idx = yi * vf.width + xi;
            for z in (1..sky_floor).rev() {
                let l = &vf.layers[z];
                let solid = l.mass[SP_ROCK][idx]
                    + l.mass[SP_SOIL][idx]
                    + l.mass[SP_ICE][idx]
                    + l.mass[SP_WOOD][idx]
                    + l.mass[SP_PLANT][idx];
                if solid >= 200 {
                    return (z as f32 + 1.15).min((d as f32) - 2.0);
                }
            }
            let z_sea = ((self.planet.config.z_sea as f32) * z_scale).clamp(2.0, d as f32 - 3.0);
            return z_sea + 0.6;
        }

        // Fast analytical fallback for unhydrated distant shards (for exploring bots!)
        let (u, v) = face_virtual_uv_f64(x as f64, y as f64, self.face_size_virtual());
        let dir = face_uv_to_sphere_dir_f64((face as usize) % FACE_COUNT, u, v);
        let col = sample_macro_planet_column(dir, self.planet.config.seed);
        let surf = col.surf_z.max(col.z_sea) as f32 * z_scale + 1.15;
        surf.clamp(2.0, (d as f32) - 2.0)
    }

    /// Locate a dry land spot inside a specific shard near `(pref_lx, pref_ly)`, returning virtual `(gx, gy, z)`.
    pub fn find_land_spot_in_shard(
        &self,
        key: PlanetShardKey,
        pref_lx: f32,
        pref_ly: f32,
    ) -> (f32, f32, f32) {
        let s = self.shard_size();
        let n_virt = self.face_size_virtual();
        let d = self.depth();
        let gx0 = key.sx as usize * s;
        let gy0 = key.sy as usize * s;

        let cx = (pref_lx.round() as i32).clamp(4, (s as i32) - 5);
        let cy = (pref_ly.round() as i32).clamp(4, (s as i32) - 5);

        let check = |lx: usize, ly: usize| -> Option<(f32, f32, f32)> {
            let gx = (gx0 + lx) as f64 + 0.5;
            let gy = (gy0 + ly) as f64 + 0.5;
            let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
            let dir = face_uv_to_sphere_dir_f64(key.face as usize, u, v);
            let col = sample_macro_planet_column(dir, self.planet.config.seed);
            if col.water_depth <= 0.1 && !col.is_rift_core {
                let z = self.find_surface_z(key.face, gx as f32, gy as f32);
                return Some((gx as f32, gy as f32, z));
            }
            None
        };

        if let Some(hit) = check(cx as usize, cy as usize) {
            return hit;
        }
        let max_r = (s / 2).saturating_sub(4) as i32;
        let step = (s / 32).max(2) as usize;
        for rad in (step..=(max_r.max(step as i32) as usize)).step_by(step) {
            let r = rad as i32;
            for dy in (-r..=r).step_by(step) {
                for dx in (-r..=r).step_by(step) {
                    if dx.abs() != r && dy.abs() != r {
                        continue;
                    }
                    let lx = (cx + dx).clamp(3, (s as i32) - 4) as usize;
                    let ly = (cy + dy).clamp(3, (s as i32) - 4) as usize;
                    if let Some(hit) = check(lx, ly) {
                        return hit;
                    }
                }
            }
        }

        let gx = (gx0 + cx as usize) as f32 + 0.5;
        let gy = (gy0 + cy as usize) as f32 + 0.5;
        let z = self.find_surface_z(key.face, gx, gy);
        (gx, gy, z.clamp(2.0, d as f32 - 2.0))
    }

    /// Return the human player's scenic golden-beach spawn `(face, x, y, z, aim_x, aim_y)` facing
    /// the sapphire bay & towering Floating Sky-Island!
    pub fn human_spawn_point(&self) -> (u8, f32, f32, f32, f32, f32) {
        let f = self.hero_site.shard_key.face;
        let gx = self.hero_site.beach_gx;
        let gy = self.hero_site.beach_gy;
        let z = self.find_surface_z(f, gx, gy);
        (f, gx, gy, z, self.hero_site.aim_x, self.hero_site.aim_y)
    }

    /// Build 6 macro-planet shrines:
    /// - 3 local shrines right inside the Hero Spawn Shard (for immediate coastal/Sky-Island battles!)
    /// - 3 distant global shrines across the globe (`+Y` North Polar Citadel, Volcanic Rift Forge, Opposite Atoll)
    pub fn build_macro_shrines(&self) -> Vec<(u8, String, u8, f32, f32, f32)> {
        let hk = self.hero_site.shard_key;
        let s = self.shard_size() as f32;
        let n_virt = self.face_size_virtual() as f32;

        let local_specs: [(&str, f32, f32); 3] = [
            ("GOLDEN BAY BEACON", s * 0.34, s * 0.36),
            ("EMERALD RIDGE SPIRE", s * 0.66, s * 0.38),
            ("AETHER WATCHTOWER", s * 0.50, s * 0.68),
        ];

        let mut out = Vec::with_capacity(6);
        for (i, &(name, lx, ly)) in local_specs.iter().enumerate() {
            let (gx, gy, z) = self.find_land_spot_in_shard(hk, lx, ly);
            out.push((i as u8, name.to_string(), hk.face, gx, gy, z));
        }

        let opp_face = match hk.face {
            0 => 1,
            1 => 0,
            2 => 3,
            3 => 2,
            4 => 5,
            _ => 4,
        };
        let side_face = if hk.face == 0 || hk.face == 1 { 4u8 } else { 0u8 };
        let global_specs: [(u8, &str, f32, f32); 3] = [
            (FACE_POS_Y as u8, "+Y NORTH POLAR CITADEL", n_virt * 0.50, n_virt * 0.50),
            (side_face, "EQUATORIAL RIFT FORGE", n_virt * 0.50, n_virt * 0.50),
            (opp_face, "ANTIPODEAN ATOLL", n_virt * 0.50, n_virt * 0.50),
        ];
        for (i, &(face, name, gx, gy)) in global_specs.iter().enumerate() {
            let z = self.find_surface_z(face, gx, gy);
            out.push(((i + 3) as u8, name.to_string(), face, gx, gy, z));
        }
        out
    }

    /// Spawn point for bot `bot_idx` out of `total_bots`:
    /// - First `~2/3` (`40 / 60`) spawn around the Hero Shard's 3 local shrines & Floating Sky-Island
    ///   so the human player drops immediately into a vibrant living coastal battle!
    /// - Remaining `~1/3` (`20 / 60`) spawn as far-ranging explorers/patrols distributed across all
    ///   6 faces & shards (including near shard borders & cube seams)!
    pub fn spawn_point_for_bot(
        &self,
        bot_idx: usize,
        total_bots: usize,
        class_id: u8,
        salt: u32,
    ) -> (u8, f32, f32, f32) {
        let local_quota = (total_bots * 2).div_ceil(3); // 40 of 60
        let d = self.depth() as f32;
        let z_scale = d / 64.0;
        let s = self.shard_size() as f32;
        let n_virt = self.face_size_virtual() as f32;
        let angle = (salt as f32) * 2.399_963_1;

        if bot_idx < local_quota {
            let hk = self.hero_site.shard_key;
            let rad = s * (0.12 + 0.28 * (((salt % 13) as f32) / 12.0));
            let pref_lx = (s * 0.5 + rad * angle.cos()).clamp(4.0, s - 4.0);
            let pref_ly = (s * 0.5 + rad * angle.sin()).clamp(4.0, s - 4.0);
            let (gx, gy, surf_z) = match class_id % 5 {
                CLASS_SKYWING => {
                    let (f, vx, vy) = self.shard_local_to_virtual(hk, pref_lx, pref_ly);
                    let sky_z = ((48.0 + (salt % 8) as f32) * z_scale).clamp(4.0, d - 1.5);
                    return (f, vx, vy, sky_z);
                }
                CLASS_TIDEKIN => {
                    let (f, vx, vy) = self.shard_local_to_virtual(hk, pref_lx, pref_ly);
                    let z_sea = (self.planet.config.z_sea as f32) * z_scale;
                    let sz = self.find_surface_z(f, vx, vy);
                    return (f, vx, vy, sz.max(z_sea + 0.5).min(d - 2.0));
                }
                CLASS_BURROWER => {
                    let (gx, gy, sz) = self.find_land_spot_in_shard(hk, pref_lx, pref_ly);
                    (gx, gy, (sz - 1.2).max(4.0))
                }
                _ => self.find_land_spot_in_shard(hk, pref_lx, pref_ly),
            };
            return (hk.face, gx, gy, surf_z.clamp(2.0, d - 2.0));
        }

        // Far-ranging global Skywing / Explorer bots patrolling across all 6 faces & shards!
        let f = (bot_idx % FACE_COUNT) as u8;
        let frac_r = 0.22 + 0.24 * (((salt % 11) as f32) / 10.0);
        let gx = (n_virt * (0.5 + frac_r * angle.cos())).clamp(8.0, n_virt - 8.0);
        let gy = (n_virt * (0.5 + frac_r * angle.sin())).clamp(8.0, n_virt - 8.0);
        let z = if class_id % 5 == CLASS_SKYWING {
            ((50.0 + (salt % 6) as f32) * z_scale).clamp(4.0, d - 1.5)
        } else {
            self.find_surface_z(f, gx, gy)
        };
        (f, gx, gy, z)
    }

    /// `/collapse_isle [id]`: Fracture the nearest intact Sky-Island `BuoyantCore` in the player's
    /// current hydrated shard (falling back to the Hero Spawn Shard), latch structural demagnetization,
    /// and enqueue a dramatic ash + thermal upscale pulse into `PlanetaryClimateNet`!
    pub fn collapse_nearest_isle(
        &mut self,
        player_face: u8,
        player_x: f32,
        player_y: f32,
        preferred_id: Option<u8>,
    ) -> Option<(u8, String, f32, f32, f32)> {
        let (cur_key, plx, ply) = self.virtual_to_shard(player_face, player_x, player_y);
        let mut search_order = vec![cur_key];
        if cur_key != self.hero_site.shard_key {
            search_order.push(self.hero_site.shard_key);
        }
        for &k in self.planet.shards.keys() {
            if !search_order.contains(&k) {
                search_order.push(k);
            }
        }

        for sk in search_order {
            let target_cid = {
                let Some(vf) = self.planet.shards.get(&sk) else {
                    continue;
                };
                if let Some(pid) = preferred_id {
                    vf.structural
                        .buoyant_cores
                        .iter()
                        .find(|c| c.id == pid && !c.collapsed)
                        .map(|c| c.id)
                } else {
                    None
                }
                .or_else(|| {
                    vf.structural
                        .buoyant_cores
                        .iter()
                        .filter(|c| !c.collapsed)
                        .min_by(|a, b| {
                            let da = (a.cx - plx).hypot(a.cy - ply);
                            let db = (b.cx - plx).hypot(b.cy - ply);
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|c| c.id)
                })
            };

            if let Some(cid) = target_cid {
                if let Some(vf) = self.planet.shards.get_mut(&sk) {
                    if let Some((cname, cx, cy, cz)) = fracture_buoyant_core(vf, cid) {
                        let _ = vf.step_structural_mechanics();
                        let (f, gx, gy) = self.shard_local_to_virtual(sk, cx, cy);
                        self.mark_priority_subbricks_around(f, gx, gy, cz, 24.0);
                        self.enqueue_climate_injection_virtual(f, gx, gy, 0.75, 1.85);
                        return Some((f, cname, gx, gy, cz));
                    }
                }
            }
        }
        None
    }

    /// `/erupt`: Trigger a conservation-exact volcanic heat + smoke burst in the player's current
    /// hydrated shard (if resident) AND inject a massive volcanic ash & heat plume directly into
    /// `PlanetaryClimateNet` (`op=13`) so real ash clouds drift downwind across the globe!
    pub fn trigger_eruption(
        &mut self,
        face: u8,
        x: f32,
        y: f32,
        z: f32,
    ) -> (u8, f32, f32, f32) {
        let f = face % (FACE_COUNT as u8);
        let d = self.depth();
        let pl = (z.round() as usize).clamp(1, d.saturating_sub(2));
        if let Some((vf, lx, ly)) = self.shard_at_virtual_mut(f, x, y) {
            vf.apply_physical_actuator(lx, ly, pl, 1.0, 0.0, 0.85, 1.0, 7.5);
            let pl_up = (pl + 1).min(d.saturating_sub(1));
            vf.apply_physical_actuator(lx, ly, pl_up, 0.85, 0.0, 0.35, 1.0, 6.5);
        }
        self.mark_priority_subbricks_around(f, x, y, z, 20.0);
        self.enqueue_climate_injection_virtual(f, x, y, 1.35, 2.40);
        (f, x, y, z)
    }

    // =========================================================================
    // 5. Cubic 16×16×16 Sub-Brick Streaming (`PlanetShardBrick3D` / `Delta`)
    //    & Global `PlanetMacroClimateFace` Overlay Streaming
    // =========================================================================

    pub fn refresh_shard_subbrick(
        &mut self,
        key: PlanetShardKey,
        bx: u16,
        by: u16,
        bz: u16,
    ) -> Option<(u8, RawBrickArc)> {
        let vf = self.planet.shards.get(&key)?;
        let (dm, at, sd) = vf.encode_3d_subbrick_16(bx as usize, by as usize, bz as usize);
        let raw = Arc::new((dm, at));
        self.subbrick_raw_cache
            .insert((key, bx, by, bz), (sd, raw.clone()));
        Some((sd, raw))
    }

    pub fn get_or_encode_shard_subbrick(
        &mut self,
        key: PlanetShardKey,
        bx: u16,
        by: u16,
        bz: u16,
    ) -> Option<(u8, RawBrickArc)> {
        if let Some((sd, arc)) = self.subbrick_raw_cache.get(&(key, bx, by, bz)) {
            return Some((*sd, arc.clone()));
        }
        self.refresh_shard_subbrick(key, bx, by, bz)
    }

    pub fn encode_shard_subbrick_full(
        &self,
        key: PlanetShardKey,
        bx: u16,
        by: u16,
        bz: u16,
        sub_depth: u8,
        dm: &[u8],
        at: &[u8],
    ) -> Bytes {
        let (lz4_density_mat, lz4_albedo_thermal) = encode_subbrick3d_lz4(dm, at);
        let pkt = ServerDatagram::PlanetShardBrick3D {
            face: key.face,
            sx: key.sx,
            sy: key.sy,
            bx,
            by,
            bz,
            total_depth: self.depth() as u16,
            sub_depth,
            tick: self.planet.tick,
            lz4_density_mat,
            lz4_albedo_thermal,
        };
        Bytes::from(encode_server_datagram(&pkt))
    }

    /// Encode an incremental `PlanetShardBrick3DDelta` (or full `PlanetShardBrick3D` if `>= 35%`
    /// changed or no shadow exists) for one sub-brick `(key, bx, by, bz)`.
    pub fn encode_incremental_shard_subbrick(
        &mut self,
        key: PlanetShardKey,
        bx: u16,
        by: u16,
        bz: u16,
        prev_shadow: Option<&RawBrickArc>,
        deadzone: u8,
    ) -> Option<(Bytes, bool, RawBrickArc)> {
        let (sub_depth, cur_arc) = self.get_or_encode_shard_subbrick(key, bx, by, bz)?;
        match prev_shadow {
            Some(prev_arc) if Arc::ptr_eq(prev_arc, &cur_arc) => None,
            Some(prev_arc) => {
                match encode_brick3d_delta(
                    &prev_arc.0,
                    &prev_arc.1,
                    &cur_arc.0,
                    &cur_arc.1,
                    deadzone.max(DEFAULT_BRICK3D_DELTA_DEADZONE),
                ) {
                    None => None,
                    Some(lz4_deltas) if !is_brick3d_delta_full_signal(&lz4_deltas) => {
                        let pkt = ServerDatagram::PlanetShardBrick3DDelta {
                            face: key.face,
                            sx: key.sx,
                            sy: key.sy,
                            bx,
                            by,
                            bz,
                            total_depth: self.depth() as u16,
                            sub_depth,
                            tick: self.planet.tick,
                            lz4_deltas,
                        };
                        Some((Bytes::from(encode_server_datagram(&pkt)), true, cur_arc))
                    }
                    Some(_) => {
                        let raw_pkt = self.encode_shard_subbrick_full(
                            key,
                            bx,
                            by,
                            bz,
                            sub_depth,
                            &cur_arc.0,
                            &cur_arc.1,
                        );
                        Some((raw_pkt, false, cur_arc))
                    }
                }
            }
            None => {
                if is_subbrick_skippable(&cur_arc.0, &cur_arc.1) {
                    return None;
                }
                let raw_pkt = self.encode_shard_subbrick_full(
                    key,
                    bx,
                    by,
                    bz,
                    sub_depth,
                    &cur_arc.0,
                    &cur_arc.1,
                );
                Some((raw_pkt, false, cur_arc))
            }
        }
    }

    /// Encode all non-skippable `16×16×sub_depth` cubic sub-bricks within `max_dist_m` (e.g. `192 m`)
    /// of `(player_face, player_x, player_y)` across the player's current shard and any adjacent
    /// hydrated shards, sorted closest-first so immediate coastal terrain & Sky-Island render first!
    /// Returns `(full_pkts, empty_sky_shadows, total_checked, skipped_count)`.
    #[allow(clippy::type_complexity)]
    pub fn encode_nearby_hydrated_subbricks(
        &mut self,
        player_face: u8,
        player_x: f32,
        player_y: f32,
        max_dist_m: f32,
    ) -> (
        Vec<(ShardSubbrickKey, RawBrickArc, Bytes)>,
        Vec<(ShardSubbrickKey, RawBrickArc)>,
        usize,
        usize,
    ) {
        let m = self.planet.config.shards_per_face_axis;
        let s = self.shard_size();
        let n_virt = self.face_size_virtual();
        let r_surf = self.planet.config.surface_radius_m();
        let (pu, pv) = face_virtual_uv_f64(player_x as f64, player_y as f64, n_virt);
        let p_dir = face_uv_to_sphere_dir_f64((player_face as usize) % FACE_COUNT, pu, pv);

        let (cur_shard, _, _) = self.virtual_to_shard(player_face, player_x, player_y);
        let mut target_shards: Vec<PlanetShardKey> = Vec::with_capacity(5);
        if self.planet.shards.contains_key(&cur_shard) {
            target_shards.push(cur_shard);
        }
        for e in 0..EDGE_COUNT {
            let nb = shard_neighbor(cur_shard, e, m).neighbor;
            if self.planet.shards.contains_key(&nb) && !target_shards.contains(&nb) {
                target_shards.push(nb);
            }
        }
        if target_shards.is_empty() && self.planet.shards.contains_key(&self.hero_site.shard_key) {
            target_shards.push(self.hero_site.shard_key);
        }

        let cx_dim = self.shard_chunks_xy();
        let bz_dim = self.subbricks_z();
        let mut candidates: Vec<(f32, ShardSubbrickKey)> = Vec::new();

        for sk in target_shards {
            let gx0 = sk.sx as usize * s;
            let gy0 = sk.sy as usize * s;
            for by in 0..cx_dim {
                let gy = (gy0 + (by as usize) * CHUNK_SIZE + CHUNK_SIZE / 2) as f64;
                for bx in 0..cx_dim {
                    let gx = (gx0 + (bx as usize) * CHUNK_SIZE + CHUNK_SIZE / 2) as f64;
                    let (bu, bv) = face_virtual_uv_f64(gx, gy, n_virt);
                    let b_dir = face_uv_to_sphere_dir_f64(sk.face as usize, bu, bv);
                    let cos_a = dot3_f64(p_dir, b_dir).clamp(-1.0, 1.0);
                    let dist_m = (cos_a.acos() * r_surf) as f32;
                    if sk == cur_shard || dist_m <= max_dist_m {
                        for bz in 0..bz_dim {
                            candidates.push((dist_m, (sk, bx, by, bz)));
                        }
                    }
                }
            }
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total = candidates.len();
        let mut pkts = Vec::with_capacity(total / 2);
        let mut empty_shadows = Vec::new();
        let mut skipped = 0usize;

        for (_, key) in candidates {
            let (sk, bx, by, bz) = key;
            let Some((sd, raw)) = self.get_or_encode_shard_subbrick(sk, bx, by, bz) else {
                continue;
            };
            if is_subbrick_skippable(&raw.0, &raw.1) {
                skipped += 1;
                if is_subbrick_empty_sky(&raw.0, &raw.1) {
                    empty_shadows.push((key, raw));
                }
                continue;
            }
            let pkt = self.encode_shard_subbrick_full(sk, bx, by, bz, sd, &raw.0, &raw.1);
            pkts.push((key, raw, pkt));
        }

        (pkts, empty_shadows, total, skipped)
    }

    /// Encode one `PlanetMacroClimateFace` LZ4-compressed `64×64` RGBA8 climate overlay packet.
    pub fn encode_climate_face_packet(&self, face: u8) -> Bytes {
        let f = (face as usize) % FACE_COUNT;
        let lz4_climate_rgba = lz4_flex::compress_prepend_size(&self.climate_rgba_faces[f]);
        let pkt = ServerDatagram::PlanetMacroClimateFace {
            face: f as u8,
            grid_n: CLIMATE_GRID_N as u16,
            tick: self.planet.tick,
            lz4_climate_rgba,
        };
        Bytes::from(encode_server_datagram(&pkt))
    }

    /// Stream `count` (1–2) `PlanetMacroClimateFace` packets round-robin across faces `0..6`
    /// (`~2 Hz` full-globe refresh = `~20 KB/s`).
    pub fn encode_next_climate_face_packets(&mut self, count: usize) -> Vec<Bytes> {
        let n = count.clamp(1, FACE_COUNT);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let f = self.climate_face_stream_cursor % (FACE_COUNT as u8);
            self.climate_face_stream_cursor = (f + 1) % (FACE_COUNT as u8);
            out.push(self.encode_climate_face_packet(f));
        }
        out
    }

    /// Encode all 6 `PlanetMacroClimateFace` packets for immediate full-globe initial sync.
    pub fn encode_all_climate_face_packets(&self) -> Vec<Bytes> {
        (0..FACE_COUNT as u8)
            .map(|f| self.encode_climate_face_packet(f))
            .collect()
    }

    /// Map a 3D unit sphere direction into virtual face coordinates `(face, gx, gy)`.
    #[allow(dead_code)]
    pub fn sphere_dir_to_virtual(&self, dir: [f64; 3]) -> (u8, f32, f32) {
        let (f, u, v) = sphere_dir_to_face_uv_f64(dir);
        let n = self.face_size_virtual() as f64;
        let gx = ((u + 1.0) * 0.5 * n).clamp(0.0, n - 1e-3) as f32;
        let gy = ((v + 1.0) * 0.5 * n).clamp(0.0, n - 1e-3) as f32;
        (f as u8, gx, gy)
    }
}

// =============================================================================
// 6. Wave 39B Ultra-Fast Parallel Shard Stepper & Climate Pack Helpers
// =============================================================================

#[inline]
fn sanitize_f32_fast(v: f32) -> f32 {
    if v.is_finite() {
        v.clamp(-1.0e6, 1.0e6)
    } else {
        0.0
    }
}

#[inline]
fn round_ties_even_i32_fast(v: f32) -> i32 {
    v.round_ties_even().clamp(-100_000.0, 100_000.0) as i32
}

#[inline]
fn field_solid_density_fast(f: &field_core::Field, idx: usize) -> f32 {
    (f.mass[SP_ROCK][idx] as f32
        + f.mass[SP_SOIL][idx] as f32
        + f.mass[SP_WOOD][idx] as f32
        + f.mass[SP_ICE][idx] as f32)
        / (C_NOM as f32)
}

#[derive(Clone, Copy, Default)]
struct LayerFlags {
    non_empty: bool,
    has_liquid: bool,
    has_gas: bool,
    has_wood_ice: bool,
    has_forcing: bool,
    has_hot_temp: bool,
}

#[derive(Clone, Copy)]
struct LawScaleParams {
    scale_raw: bool,
    liq_mul: f32,
    cond_mul: f32,
    precip_mul: f32,
    vol_mul: f32,
}

/// Thread-local reusable horizontal scratch buffers (allocated ONCE per worker thread per tick,
/// eliminating >500 MB/tick of per-layer `Vec` alloc/zero/free churn!).
struct LayerScratch {
    caps: Vec<i64>,
    vols: Vec<u32>,
    temps: Vec<i64>,
    rock_wall: Vec<bool>,
    blocked: Vec<bool>,
    heads: Vec<f32>,
    k_eff: Vec<f32>,
    adv_out_e: Vec<i64>,
    adv_in_e: Vec<i64>,
    qe: Vec<[i32; NUM_MOBILE]>,
    qs: Vec<[i32; NUM_MOBILE]>,
    ge: Vec<i64>,
    gs: Vec<i64>,
    chunk_active: Vec<bool>,
    chunk_needed: Vec<bool>,
    chunk_dirty: Vec<bool>,
}

impl LayerScratch {
    fn new(area: usize, num_chunks: usize) -> Self {
        Self {
            caps: vec![0; area],
            vols: vec![0; area],
            temps: vec![T_AMB; area],
            rock_wall: vec![false; area],
            blocked: vec![false; area],
            heads: vec![0.0; area],
            k_eff: vec![4.0; area],
            adv_out_e: vec![0; area],
            adv_in_e: vec![0; area],
            qe: vec![[0; NUM_MOBILE]; area],
            qs: vec![[0; NUM_MOBILE]; area],
            ge: vec![0; area],
            gs: vec![0; area],
            chunk_active: vec![false; num_chunks],
            chunk_needed: vec![false; num_chunks],
            chunk_dirty: vec![false; num_chunks],
        }
    }

    fn ensure_size(&mut self, area: usize, num_chunks: usize) {
        if self.caps.len() != area {
            self.caps.resize(area, 0);
            self.vols.resize(area, 0);
            self.temps.resize(area, T_AMB);
            self.rock_wall.resize(area, false);
            self.blocked.resize(area, false);
            self.heads.resize(area, 0.0);
            self.k_eff.resize(area, 4.0);
            self.adv_out_e.resize(area, 0);
            self.adv_in_e.resize(area, 0);
            self.qe.resize(area, [0; NUM_MOBILE]);
            self.qs.resize(area, [0; NUM_MOBILE]);
            self.ge.resize(area, 0);
            self.gs.resize(area, 0);
        }
        if self.chunk_active.len() != num_chunks {
            self.chunk_active.resize(num_chunks, false);
            self.chunk_needed.resize(num_chunks, false);
            self.chunk_dirty.resize(num_chunks, false);
        }
    }
}

/// Fast single-pass layer scanner: updates `prev_temp` & `prev_steam`, detects `LayerFlags`,
/// and marks active `16×16` chunks while skipping empty air cells (`energy == 0`) in 1 branch!
/// Also populates `temps` / `caps` / `vols` directly so we avoid redundant divisions!
fn scan_layer_and_snapshot_sensors(
    layer: &field_core::Field,
    prev_temp_slice: &mut [i32],
    prev_steam_slice: &mut [i32],
    chunk_active: &mut [bool],
    chunks_x: usize,
) -> LayerFlags {
    let w = layer.width;
    let h = layer.height;
    let mut flags = LayerFlags::default();
    chunk_active.fill(false);

    for cy in 0..h.div_ceil(CHUNK_SIZE) {
        let y0 = cy * CHUNK_SIZE;
        let y1 = (y0 + CHUNK_SIZE).min(h);
        for cx in 0..chunks_x {
            let x0 = cx * CHUNK_SIZE;
            let x1 = (x0 + CHUNK_SIZE).min(w);
            let ci = cy * chunks_x + cx;
            let mut c_active = false;
            let mut c_non_empty = false;
            let mut c_min_t = i64::MAX;
            let mut c_max_t = i64::MIN;

            for y in y0..y1 {
                let row = y * w;
                for x in x0..x1 {
                    let idx = row + x;
                    let e = layer.energy[idx];
                    let fe = layer.forcing_energy[idx];
                    let fm = layer.forcing_mass[idx];
                    if e == 0 && fe == 0 && fm == 0 {
                        prev_temp_slice[idx] = T_AMB as i32;
                        prev_steam_slice[idx] = 0;
                        continue;
                    }
                    c_non_empty = true;
                    let water = layer.mass[SP_WATER][idx];
                    let oil = layer.mass[SP_OIL][idx];
                    let steam = layer.mass[SP_STEAM][idx];
                    let smoke = layer.mass[SP_SMOKE][idx];
                    let wood = layer.mass[SP_WOOD][idx];
                    let ice = layer.mass[SP_ICE][idx];
                    if water > 8 || oil > 8 {
                        flags.has_liquid = true;
                    }
                    if steam > 8 || smoke > 12 {
                        flags.has_gas = true;
                    }
                    if wood >= 15 || ice >= 15 {
                        flags.has_wood_ice = true;
                    }
                    if water > 0 || oil > 0 || steam > 0 || smoke > 0 {
                        c_active = true;
                    }
                    if fe != 0 || fm != 0 {
                        flags.has_forcing = true;
                        c_active = true;
                    }

                    // Fast-path heat capacity: 90% of solid terrain is pure SP_ROCK or pure SP_WATER
                    let rock = layer.mass[SP_ROCK][idx];
                    let c = if rock > 0
                        && water == 0
                        && ice == 0
                        && steam == 0
                        && wood == 0
                        && smoke == 0
                        && oil == 0
                        && layer.mass[SP_SOIL][idx] == 0
                        && layer.mass[SP_PLANT][idx] == 0
                        && layer.mass[field_core::SP_ASH][idx] == 0
                    {
                        (rock as i64) * SPECIES[SP_ROCK].c
                    } else if water > 0
                        && rock == 0
                        && ice == 0
                        && steam == 0
                        && wood == 0
                        && smoke == 0
                        && oil == 0
                        && layer.mass[SP_SOIL][idx] == 0
                        && layer.mass[SP_PLANT][idx] == 0
                        && layer.mass[field_core::SP_ASH][idx] == 0
                    {
                        (water as i64) * SPECIES[SP_WATER].c
                    } else {
                        layer.cell_heat_capacity(idx)
                    };
                    let t = if c > 0 {
                        (e / c).clamp(T_MIN, T_MAX)
                    } else {
                        T_AMB
                    };
                    prev_temp_slice[idx] = t as i32;
                    prev_steam_slice[idx] = steam as i32;

                    if t < c_min_t {
                        c_min_t = t;
                    }
                    if t > c_max_t {
                        c_max_t = t;
                    }
                    if t > 37_315 {
                        flags.has_hot_temp = true;
                        c_active = true;
                    } else if t < 27_315 && water > 0 {
                        c_active = true;
                    } else if t > 27_315 && ice > 0 {
                        c_active = true;
                    }
                }
            }
            if c_non_empty {
                flags.non_empty = true;
                if c_max_t - c_min_t > 200 {
                    c_active = true;
                }
            }
            chunk_active[ci] = c_active;
        }
    }
    flags
}

/// Wave 39B: Exact integer conservative Projection Firewall applied strictly inside a `roi_w × roi_h`
/// Hero ROI `[x0..x0+roi_w, y0..y0+roi_h)` with ZERO perimeter flux leak (`mass_error == 0 &&
/// energy_error == 0` bit-exactly!).
#[allow(clippy::too_many_arguments)]
fn project_and_apply_roi_fast(
    layer: &mut field_core::Field,
    x0: usize,
    y0: usize,
    roi_w: usize,
    roi_h: usize,
    raw: &RawStepOutput,
    laws: LawScaleParams,
    scratch: &mut LayerScratch,
    chunks_x: usize,
) {
    let w = layer.width;
    let rlen = roi_w * roi_h;
    if rlen == 0 {
        return;
    }

    // Snapshot tick-start ROI state into scratch[0..rlen]
    for ly in 0..roi_h {
        let g_row = (y0 + ly) * w + x0;
        let l_row = ly * roi_w;
        for lx in 0..roi_w {
            let gidx = g_row + lx;
            let lidx = l_row + lx;
            let c = layer.cell_heat_capacity(gidx);
            let v = layer.cell_volume(gidx);
            scratch.caps[lidx] = c;
            scratch.vols[lidx] = v;
            scratch.temps[lidx] = if c > 0 {
                (layer.energy[gidx] / c).clamp(T_MIN, T_MAX)
            } else {
                T_AMB
            };
            scratch.rock_wall[lidx] = layer.mass[SP_ROCK][gidx] >= 500;
            scratch.adv_out_e[lidx] = 0;
            scratch.adv_in_e[lidx] = 0;
            scratch.qe[lidx] = [0; NUM_MOBILE];
            scratch.qs[lidx] = [0; NUM_MOBILE];
            scratch.ge[lidx] = 0;
            scratch.gs[lidx] = 0;
        }
    }

    // Steps 1-6: Interior ROI edges ONLY (perimeter East/South edges at lx==roi_w-1 / ly==roi_h-1 stay 0!)
    for ly in 0..roi_h {
        let g_row = (y0 + ly) * w + x0;
        let l_row = ly * roi_w;
        for lx in 0..roi_w {
            let a_loc = l_row + lx;
            let a = g_row + lx;
            for dir in 0..2 {
                let (b_loc, b, is_east) = if dir == 0 {
                    if lx + 1 >= roi_w {
                        continue;
                    }
                    (a_loc + 1, a + 1, true)
                } else {
                    if ly + 1 >= roi_h {
                        continue;
                    }
                    (a_loc + roi_w, a + w, false)
                };

                let ca = scratch.caps[a_loc];
                let cb = scratch.caps[b_loc];
                if ca == 0 && cb == 0 {
                    continue;
                }

                let base_c = if is_east { 0 } else { 4 };
                let heat_c = if is_east { 8 } else { 9 };
                let wall_edge = scratch.rock_wall[a_loc] || scratch.rock_wall[b_loc];

                let mut q = [0i32; NUM_MOBILE];
                if !wall_edge {
                    for m in 0..NUM_MOBILE {
                        let s = MOBILE_SPECIES[m];
                        let mut v = sanitize_f32_fast(raw.get(base_c + m, a_loc));
                        if laws.scale_raw && (m == 0 || m == 3) {
                            v *= laws.liq_mul;
                        }
                        let q_raw = round_ties_even_i32_fast(v * (C_NOM as f32));
                        q[m] = if q_raw > 0 {
                            q_raw.min((layer.mass[s][a] / 4) as i32)
                        } else if q_raw < 0 {
                            q_raw.max(-((layer.mass[s][b] / 4) as i32))
                        } else {
                            0
                        };
                    }

                    let pos_tot: i64 = q.iter().filter(|&&v| v > 0).map(|&v| v as i64).sum();
                    if pos_tot > 0 {
                        let cap_b = (H_CAP.saturating_sub(scratch.vols[b_loc]) / 4) as i64;
                        if pos_tot > cap_b {
                            for v in &mut q {
                                if *v > 0 {
                                    *v = ((*v as i64) * cap_b / pos_tot) as i32;
                                }
                            }
                        }
                    }

                    let neg_tot: i64 = q.iter().filter(|&&v| v < 0).map(|&v| (-v) as i64).sum();
                    if neg_tot > 0 {
                        let cap_a = (H_CAP.saturating_sub(scratch.vols[a_loc]) / 4) as i64;
                        if neg_tot > cap_a {
                            for v in &mut q {
                                if *v < 0 {
                                    *v = -(((-*v) as i64) * cap_a / neg_tot) as i32;
                                }
                            }
                        }
                    }
                }

                // 2nd-law discrete heat conduction
                let dt = scratch.temps[a_loc] - scratch.temps[b_loc];
                let g_clamped = if ca > 0 && cb > 0 && dt != 0 {
                    let mut g_raw_f = sanitize_f32_fast(raw.get(heat_c, a_loc));
                    if laws.scale_raw {
                        g_raw_f *= laws.cond_mul;
                    }
                    let g_raw = (g_raw_f as f64 * 1e6).round() as i64;
                    let max_g = ((dt.unsigned_abs() as u128)
                        * (ca as u128)
                        * (cb as u128)
                        / (4 * ((ca + cb) as u128))) as i64;
                    if dt > 0 {
                        g_raw.clamp(0, max_g)
                    } else {
                        g_raw.clamp(-max_g, 0)
                    }
                } else {
                    0
                };

                // Exact integer advected thermal energy (paired on local ROI indices)
                for m in 0..NUM_MOBILE {
                    let qv = q[m] as i64;
                    if qv == 0 {
                        continue;
                    }
                    let s = MOBILE_SPECIES[m];
                    if qv > 0 && ca > 0 {
                        let e_adv = ((qv as i128) * (SPECIES[s].c as i128)
                            * (layer.energy[a] as i128)
                            / (ca as i128)) as i64;
                        scratch.adv_out_e[a_loc] += e_adv;
                        scratch.adv_in_e[b_loc] += e_adv;
                    } else if qv < 0 && cb > 0 {
                        let q_abs = -qv;
                        let e_adv = ((q_abs as i128) * (SPECIES[s].c as i128)
                            * (layer.energy[b] as i128)
                            / (cb as i128)) as i64;
                        scratch.adv_out_e[b_loc] += e_adv;
                        scratch.adv_in_e[a_loc] += e_adv;
                    }
                }

                if is_east {
                    scratch.qe[a_loc] = q;
                    scratch.ge[a_loc] = g_clamped;
                } else {
                    scratch.qs[a_loc] = q;
                    scratch.gs[a_loc] = g_clamped;
                }
            }
        }
    }

    // Steps 7-8: Apply strictly antisymmetric mass & heat fluxes inside ROI
    for ly in 0..roi_h {
        let gy = y0 + ly;
        let cy = gy / CHUNK_SIZE;
        let g_row = gy * w + x0;
        let l_row = ly * roi_w;
        for lx in 0..roi_w {
            let gx = x0 + lx;
            let cx = gx / CHUNK_SIZE;
            let ci = cy * chunks_x + cx;
            let a_loc = l_row + lx;
            let a = g_row + lx;

            if lx + 1 < roi_w {
                let b_loc = a_loc + 1;
                let b = a + 1;
                let mut moved = false;
                for m in 0..NUM_MOBILE {
                    let qv = scratch.qe[a_loc][m];
                    if qv != 0 {
                        let s = MOBILE_SPECIES[m];
                        layer.mass[s][a] = (layer.mass[s][a] as i32 - qv) as u16;
                        layer.mass[s][b] = (layer.mass[s][b] as i32 + qv) as u16;
                        moved = true;
                    }
                }
                let mut g = scratch.ge[a_loc];
                if g > 0 {
                    let avail = (layer.energy[a] - scratch.adv_out_e[a_loc]).max(0) / 4;
                    g = g.min(avail);
                } else if g < 0 {
                    let avail = (layer.energy[b] - scratch.adv_out_e[b_loc]).max(0) / 4;
                    g = g.max(-avail);
                }
                if g != 0 {
                    layer.energy[a] -= g;
                    layer.energy[b] += g;
                }
                layer.flow_e[a] = (scratch.qe[a_loc][0] + scratch.qe[a_loc][3])
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                if moved {
                    scratch.chunk_dirty[ci] = true;
                    scratch.chunk_dirty[cy * chunks_x + ((gx + 1) / CHUNK_SIZE)] = true;
                }
            } else {
                layer.flow_e[a] = 0;
            }

            if ly + 1 < roi_h {
                let b_loc = a_loc + roi_w;
                let b = a + w;
                let mut moved = false;
                for m in 0..NUM_MOBILE {
                    let qv = scratch.qs[a_loc][m];
                    if qv != 0 {
                        let s = MOBILE_SPECIES[m];
                        layer.mass[s][a] = (layer.mass[s][a] as i32 - qv) as u16;
                        layer.mass[s][b] = (layer.mass[s][b] as i32 + qv) as u16;
                        moved = true;
                    }
                }
                let mut g = scratch.gs[a_loc];
                if g > 0 {
                    let avail = (layer.energy[a] - scratch.adv_out_e[a_loc]).max(0) / 4;
                    g = g.min(avail);
                } else if g < 0 {
                    let avail = (layer.energy[b] - scratch.adv_out_e[b_loc]).max(0) / 4;
                    g = g.max(-avail);
                }
                if g != 0 {
                    layer.energy[a] -= g;
                    layer.energy[b] += g;
                }
                layer.flow_s[a] = (scratch.qs[a_loc][0] + scratch.qs[a_loc][3])
                    .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                if moved {
                    scratch.chunk_dirty[ci] = true;
                    scratch.chunk_dirty[((gy + 1) / CHUNK_SIZE) * chunks_x + cx] = true;
                }
            } else {
                layer.flow_s[a] = 0;
            }
        }
    }

    // Apply advected energy + Step 9 reactions + Step 10 ambient relaxation & smoke dissipation inside ROI
    for ly in 0..roi_h {
        let gy = y0 + ly;
        let cy = gy / CHUNK_SIZE;
        let g_row = gy * w + x0;
        let l_row = ly * roi_w;
        for lx in 0..roi_w {
            let gx = x0 + lx;
            let ci = cy * chunks_x + (gx / CHUNK_SIZE);
            let lidx = l_row + lx;
            let gidx = g_row + lx;

            let de_adv = scratch.adv_in_e[lidx] - scratch.adv_out_e[lidx];
            if de_adv != 0 {
                layer.energy[gidx] += de_adv;
            }

            if layer.energy[gidx] > 0 {
                let t_now = layer.cell_temp_ck(gidx);
                if t_now < 27_315 || t_now > 27_315 {
                    let mut xi = [0i32; NUM_REACTIONS];
                    let mut any_rxn = false;
                    for r in 0..NUM_REACTIONS {
                        let spec = &REACTIONS[r];
                        if t_now < spec.t_lo || t_now > spec.t_hi {
                            continue;
                        }
                        let mut raw_xi = sanitize_f32_fast(raw.get(10 + r, lidx)).max(0.0);
                        if laws.scale_raw {
                            if r == 3 {
                                raw_xi *= laws.precip_mul;
                            } else if r >= 4 {
                                raw_xi *= laws.vol_mul;
                            }
                        }
                        let mut x_int = (raw_xi * 100.0).floor().min(2000.0) as i32;
                        if x_int <= 0 {
                            continue;
                        }
                        let de = spec.delta_e_th();
                        if de < 0 {
                            let c_now = layer.cell_heat_capacity(gidx);
                            let avail = (layer.energy[gidx] - c_now * T_MIN).max(0);
                            x_int = x_int.min((avail / (-de)) as i32);
                        } else if de > 0 {
                            let c_now = layer.cell_heat_capacity(gidx).max(10_000);
                            let room = (c_now * T_MAX - layer.energy[gidx]).max(0);
                            x_int = x_int.min((room / de) as i32);
                        }
                        if x_int > 0 {
                            xi[r] = x_int;
                            any_rxn = true;
                        }
                    }

                    if any_rxn {
                        for s in 0..NUM_SPECIES {
                            let demand: i64 = (0..NUM_REACTIONS)
                                .filter(|&r| REACTIONS[r].nu[s] < 0)
                                .map(|r| (-REACTIONS[r].nu[s] as i64) * (xi[r] as i64))
                                .sum();
                            let avail = layer.mass[s][gidx] as i64;
                            if demand > avail && demand > 0 {
                                for r in 0..NUM_REACTIONS {
                                    if REACTIONS[r].nu[s] < 0 {
                                        xi[r] = ((xi[r] as i64) * avail / demand) as i32;
                                    }
                                }
                            }
                        }
                        for r in 0..NUM_REACTIONS {
                            let x_int = xi[r];
                            if x_int <= 0 {
                                continue;
                            }
                            let spec = &REACTIONS[r];
                            for s in 0..NUM_SPECIES {
                                let dm = (spec.nu[s] as i32) * x_int;
                                if dm != 0 {
                                    layer.mass[s][gidx] = (layer.mass[s][gidx] as i32 + dm) as u16;
                                }
                            }
                            layer.energy[gidx] += spec.delta_e_th() * (x_int as i64);
                            scratch.chunk_dirty[ci] = true;
                        }
                    }
                }

                let c_now = layer.cell_heat_capacity(gidx);
                if c_now > 0 {
                    let target_e = c_now * T_AMB;
                    let diff = layer.energy[gidx] - target_e;
                    let cool = diff / 40;
                    if cool != 0 {
                        layer.energy[gidx] -= cool;
                        layer.atmosphere_energy += cool;
                        if (cool / c_now).abs() >= 150 {
                            scratch.chunk_dirty[ci] = true;
                        }
                    }
                }
                let sm = layer.mass[SP_SMOKE][gidx];
                if sm > 0 {
                    let diss = ((sm / 10).max(1)).min(sm);
                    let c_cell = layer.cell_heat_capacity(gidx).max(1);
                    let e_diss =
                        (diss as i64) * SPECIES[SP_SMOKE].c * (layer.energy[gidx] / c_cell);
                    layer.mass[SP_SMOKE][gidx] -= diss;
                    layer.energy[gidx] -= e_diss;
                    layer.atmosphere_mass[SP_SMOKE] += diss as i64;
                    layer.atmosphere_energy += e_diss + (diss as i64) * SPECIES[SP_SMOKE].u0;
                    scratch.chunk_dirty[ci] = true;
                }
            }
            if layer.forcing_energy[gidx] != 0 || layer.forcing_mass[gidx] != 0 {
                layer.forcing_energy[gidx] = 0;
                layer.forcing_mass[gidx] = 0;
                scratch.chunk_dirty[ci] = true;
            }
        }
    }
    layer.tick += 1;
}

/// Wave 39B: Zero-allocation active-chunk conservative horizontal RefSim + Projection Firewall.
/// Skips quiescent/empty `16×16` chunks completely while keeping exact antisymmetric mass & thermal
/// conservation across every active chunk and its 1-chunk halo (`mass_error == 0 && energy_error == 0`)!
fn step_layer_refsim_fast(
    layer: &mut field_core::Field,
    laws: LawScaleParams,
    scratch: &mut LayerScratch,
    chunks_x: usize,
    chunks_y: usize,
    exclude_roi: Option<(usize, usize, usize, usize)>,
) {
    let w = layer.width;
    let h = layer.height;
    let num_chunks = chunks_x * chunks_y;

    // Build `chunk_needed` = `chunk_active` dilated by 1 chunk so every edge leaving an active chunk
    // has valid snapshots on both endpoints. If `exclude_roi` is set (already stepped by neural op=11),
    // skip chunks wholly inside the ROI!
    scratch.chunk_needed.fill(false);
    let mut any_needed = false;
    for cy in 0..chunks_y {
        for cx in 0..chunks_x {
            let ci = cy * chunks_x + cx;
            if !scratch.chunk_active[ci] {
                continue;
            }
            if let Some((rx0, ry0, rw, rh)) = exclude_roi {
                let x0 = cx * CHUNK_SIZE;
                let y0 = cy * CHUNK_SIZE;
                if x0 >= rx0 && x0 + CHUNK_SIZE <= rx0 + rw && y0 >= ry0 && y0 + CHUNK_SIZE <= ry0 + rh {
                    continue;
                }
            }
            any_needed = true;
            for dy in -1isize..=1 {
                let ny = cy as isize + dy;
                if ny < 0 || ny >= chunks_y as isize {
                    continue;
                }
                for dx in -1isize..=1 {
                    let nx = cx as isize + dx;
                    if nx < 0 || nx >= chunks_x as isize {
                        continue;
                    }
                    scratch.chunk_needed[ny as usize * chunks_x + nx as usize] = true;
                }
            }
        }
    }
    if !any_needed {
        layer.tick += 1;
        return;
    }

    // Snapshot ONLY needed chunks into scratch
    for ci in 0..num_chunks {
        if !scratch.chunk_needed[ci] {
            continue;
        }
        let cx = ci % chunks_x;
        let cy = ci / chunks_x;
        let x0 = cx * CHUNK_SIZE;
        let x1 = (x0 + CHUNK_SIZE).min(w);
        let y0 = cy * CHUNK_SIZE;
        let y1 = (y0 + CHUNK_SIZE).min(h);
        for y in y0..y1 {
            let row = y * w;
            for x in x0..x1 {
                let idx = row + x;
                let e = layer.energy[idx];
                scratch.adv_out_e[idx] = 0;
                scratch.adv_in_e[idx] = 0;
                scratch.qe[idx] = [0; NUM_MOBILE];
                scratch.qs[idx] = [0; NUM_MOBILE];
                scratch.ge[idx] = 0;
                scratch.gs[idx] = 0;
                if e == 0 {
                    scratch.caps[idx] = 0;
                    scratch.vols[idx] = 0;
                    scratch.temps[idx] = T_AMB;
                    scratch.rock_wall[idx] = false;
                    scratch.blocked[idx] = false;
                    scratch.heads[idx] = (layer.elevation[idx] as f32) * 120.0;
                    scratch.k_eff[idx] = 4.0;
                    continue;
                }
                let mut vol = 0u32;
                let mut cap = 0i64;
                let mut k_sum = 0.0f32;
                for s in 0..NUM_SPECIES {
                    let m = layer.mass[s][idx];
                    if m != 0 {
                        vol += m as u32;
                        cap += (m as i64) * SPECIES[s].c;
                        k_sum += (m as f32) * (SPECIES[s].k as f32);
                    }
                }
                scratch.caps[idx] = cap;
                scratch.vols[idx] = vol;
                scratch.temps[idx] = if cap > 0 {
                    (e / cap).clamp(T_MIN, T_MAX)
                } else {
                    T_AMB
                };
                let rock = layer.mass[SP_ROCK][idx];
                let ice = layer.mass[SP_ICE][idx];
                scratch.rock_wall[idx] = rock >= 500;
                scratch.blocked[idx] = rock >= 500 || ice >= 700;
                let liq = (layer.mass[SP_WATER][idx] as f32) + (layer.mass[SP_OIL][idx] as f32);
                scratch.heads[idx] = (layer.elevation[idx] as f32) * 120.0 + liq;
                scratch.k_eff[idx] = if vol > 0 { k_sum / (vol as f32) } else { 4.0 };
            }
        }
    }

    let in_excluded_roi = |x: usize, y: usize| -> bool {
        if let Some((rx0, ry0, rw, rh)) = exclude_roi {
            x >= rx0 && x < rx0 + rw && y >= ry0 && y < ry0 + rh
        } else {
            false
        }
    };

    // Fused RefSim + integer Projection on active/needed edges ONLY
    for ci in 0..num_chunks {
        if !scratch.chunk_needed[ci] {
            continue;
        }
        let cx = ci % chunks_x;
        let cy = ci / chunks_x;
        let x0 = cx * CHUNK_SIZE;
        let x1 = (x0 + CHUNK_SIZE).min(w);
        let y0 = cy * CHUNK_SIZE;
        let y1 = (y0 + CHUNK_SIZE).min(h);

        for y in y0..y1 {
            let row = y * w;
            for x in x0..x1 {
                if in_excluded_roi(x, y) {
                    continue;
                }
                let a = row + x;
                for dir in 0..2 {
                    let (bx, by, b, is_east) = if dir == 0 {
                        if x + 1 >= w {
                            continue;
                        }
                        (x + 1, y, a + 1, true)
                    } else {
                        if y + 1 >= h {
                            continue;
                        }
                        (x, y + 1, a + w, false)
                    };
                    if in_excluded_roi(bx, by) {
                        continue;
                    }
                    let b_ci = (by / CHUNK_SIZE) * chunks_x + (bx / CHUNK_SIZE);
                    if !scratch.chunk_needed[b_ci] {
                        continue;
                    }
                    if !scratch.chunk_active[ci] && !scratch.chunk_active[b_ci] {
                        continue;
                    }

                    let ca = scratch.caps[a];
                    let cb = scratch.caps[b];
                    if ca == 0 && cb == 0 {
                        continue;
                    }

                    let mut q = [0i32; NUM_MOBILE];
                    if !scratch.blocked[a] && !scratch.blocked[b] {
                        let dh = scratch.heads[a] - scratch.heads[b];
                        let prev_flow = if is_east {
                            layer.flow_e[a] as f32
                        } else {
                            layer.flow_s[a] as f32
                        };
                        let w_a = layer.mass[SP_WATER][a] as f32;
                        let w_b = layer.mass[SP_WATER][b] as f32;
                        if w_a > 0.0 || w_b > 0.0 {
                            let w_flux = 0.16 * dh + 0.25 * prev_flow;
                            let mut w_clamped = if w_flux > 0.0 {
                                w_flux.min(w_a * 0.22)
                            } else {
                                w_flux.max(-(w_b * 0.22))
                            };
                            if laws.scale_raw {
                                w_clamped *= laws.liq_mul;
                            }
                            let q_raw = round_ties_even_i32_fast(w_clamped);
                            q[0] = if q_raw > 0 {
                                q_raw.min((layer.mass[SP_WATER][a] / 4) as i32)
                            } else if q_raw < 0 {
                                q_raw.max(-((layer.mass[SP_WATER][b] / 4) as i32))
                            } else {
                                0
                            };
                        }

                        let st_a = layer.mass[SP_STEAM][a] as f32;
                        let st_b = layer.mass[SP_STEAM][b] as f32;
                        if st_a > 0.0 || st_b > 0.0 {
                            let q_raw = round_ties_even_i32_fast(0.18 * (st_a - st_b));
                            q[1] = if q_raw > 0 {
                                q_raw.min((layer.mass[SP_STEAM][a] / 4) as i32)
                            } else if q_raw < 0 {
                                q_raw.max(-((layer.mass[SP_STEAM][b] / 4) as i32))
                            } else {
                                0
                            };
                        }

                        let sm_a = layer.mass[SP_SMOKE][a] as f32;
                        let sm_b = layer.mass[SP_SMOKE][b] as f32;
                        if sm_a > 0.0 || sm_b > 0.0 {
                            let drift = if is_east { 4.0 } else { -3.0 };
                            let sm_flux = 0.18 * (sm_a - sm_b)
                                + if sm_a + sm_b > 20.0 { drift } else { 0.0 };
                            let q_raw = round_ties_even_i32_fast(sm_flux);
                            q[2] = if q_raw > 0 {
                                q_raw.min((layer.mass[SP_SMOKE][a] / 4) as i32)
                            } else if q_raw < 0 {
                                q_raw.max(-((layer.mass[SP_SMOKE][b] / 4) as i32))
                            } else {
                                0
                            };
                        }

                        let o_a = layer.mass[SP_OIL][a] as f32;
                        let o_b = layer.mass[SP_OIL][b] as f32;
                        if o_a > 0.0 || o_b > 0.0 {
                            let o_flux = 0.09 * dh + 0.10 * (o_a - o_b);
                            let mut o_clamped = if o_flux > 0.0 {
                                o_flux.min(o_a * 0.18)
                            } else {
                                o_flux.max(-(o_b * 0.18))
                            };
                            if laws.scale_raw {
                                o_clamped *= laws.liq_mul;
                            }
                            let q_raw = round_ties_even_i32_fast(o_clamped);
                            q[3] = if q_raw > 0 {
                                q_raw.min((layer.mass[SP_OIL][a] / 4) as i32)
                            } else if q_raw < 0 {
                                q_raw.max(-((layer.mass[SP_OIL][b] / 4) as i32))
                            } else {
                                0
                            };
                        }

                        let pos_tot: i64 = q.iter().filter(|&&v| v > 0).map(|&v| v as i64).sum();
                        if pos_tot > 0 {
                            let cap_b = (H_CAP.saturating_sub(scratch.vols[b]) / 4) as i64;
                            if pos_tot > cap_b {
                                for v in &mut q {
                                    if *v > 0 {
                                        *v = ((*v as i64) * cap_b / pos_tot) as i32;
                                    }
                                }
                            }
                        }
                        let neg_tot: i64 = q.iter().filter(|&&v| v < 0).map(|&v| (-v) as i64).sum();
                        if neg_tot > 0 {
                            let cap_a = (H_CAP.saturating_sub(scratch.vols[a]) / 4) as i64;
                            if neg_tot > cap_a {
                                for v in &mut q {
                                    if *v < 0 {
                                        *v = -(((-*v) as i64) * cap_a / neg_tot) as i32;
                                    }
                                }
                            }
                        }
                    }

                    // Discrete 2nd-law heat conduction
                    let dt = scratch.temps[a] - scratch.temps[b];
                    let g_clamped = if ca > 0 && cb > 0 && dt.abs() > 2 {
                        let k_edge = 0.5 * (scratch.k_eff[a] + scratch.k_eff[b]);
                        let c_harm = (ca as f32 * cb as f32) / ((ca + cb) as f32);
                        let alpha = (0.012 * k_edge).min(0.22);
                        let mut g_f = alpha * (dt as f32) * c_harm;
                        if laws.scale_raw {
                            g_f *= laws.cond_mul;
                        }
                        let g_raw = g_f.round() as i64;
                        let max_g = ((dt.unsigned_abs() as u128)
                            * (ca as u128)
                            * (cb as u128)
                            / (4 * ((ca + cb) as u128))) as i64;
                        if dt > 0 {
                            g_raw.clamp(0, max_g)
                        } else {
                            g_raw.clamp(-max_g, 0)
                        }
                    } else {
                        0
                    };

                    // Advected thermal energy
                    for m in 0..NUM_MOBILE {
                        let qv = q[m] as i64;
                        if qv == 0 {
                            continue;
                        }
                        let s = MOBILE_SPECIES[m];
                        if qv > 0 && ca > 0 {
                            let e_adv = ((qv as i128) * (SPECIES[s].c as i128)
                                * (layer.energy[a] as i128)
                                / (ca as i128)) as i64;
                            scratch.adv_out_e[a] += e_adv;
                            scratch.adv_in_e[b] += e_adv;
                        } else if qv < 0 && cb > 0 {
                            let q_abs = -qv;
                            let e_adv = ((q_abs as i128) * (SPECIES[s].c as i128)
                                * (layer.energy[b] as i128)
                                / (cb as i128)) as i64;
                            scratch.adv_out_e[b] += e_adv;
                            scratch.adv_in_e[a] += e_adv;
                        }
                    }

                    if is_east {
                        scratch.qe[a] = q;
                        scratch.ge[a] = g_clamped;
                    } else {
                        scratch.qs[a] = q;
                        scratch.gs[a] = g_clamped;
                    }
                }
            }
        }
    }

    // Apply antisymmetric mass & heat fluxes on needed chunks
    for ci in 0..num_chunks {
        if !scratch.chunk_needed[ci] {
            continue;
        }
        let cx = ci % chunks_x;
        let cy = ci / chunks_x;
        let x0 = cx * CHUNK_SIZE;
        let x1 = (x0 + CHUNK_SIZE).min(w);
        let y0 = cy * CHUNK_SIZE;
        let y1 = (y0 + CHUNK_SIZE).min(h);

        for y in y0..y1 {
            let row = y * w;
            for x in x0..x1 {
                if in_excluded_roi(x, y) {
                    continue;
                }
                let a = row + x;
                if x + 1 < w && !in_excluded_roi(x + 1, y) {
                    let b = a + 1;
                    let mut moved = false;
                    for m in 0..NUM_MOBILE {
                        let qv = scratch.qe[a][m];
                        if qv != 0 {
                            let s = MOBILE_SPECIES[m];
                            layer.mass[s][a] = (layer.mass[s][a] as i32 - qv) as u16;
                            layer.mass[s][b] = (layer.mass[s][b] as i32 + qv) as u16;
                            moved = true;
                        }
                    }
                    let mut g = scratch.ge[a];
                    if g > 0 {
                        let avail = (layer.energy[a] - scratch.adv_out_e[a]).max(0) / 4;
                        g = g.min(avail);
                    } else if g < 0 {
                        let avail = (layer.energy[b] - scratch.adv_out_e[b]).max(0) / 4;
                        g = g.max(-avail);
                    }
                    if g != 0 {
                        layer.energy[a] -= g;
                        layer.energy[b] += g;
                    }
                    layer.flow_e[a] = (scratch.qe[a][0] + scratch.qe[a][3])
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    if moved {
                        scratch.chunk_dirty[ci] = true;
                        scratch.chunk_dirty[cy * chunks_x + ((x + 1) / CHUNK_SIZE)] = true;
                    }
                }
                if y + 1 < h && !in_excluded_roi(x, y + 1) {
                    let b = a + w;
                    let mut moved = false;
                    for m in 0..NUM_MOBILE {
                        let qv = scratch.qs[a][m];
                        if qv != 0 {
                            let s = MOBILE_SPECIES[m];
                            layer.mass[s][a] = (layer.mass[s][a] as i32 - qv) as u16;
                            layer.mass[s][b] = (layer.mass[s][b] as i32 + qv) as u16;
                            moved = true;
                        }
                    }
                    let mut g = scratch.gs[a];
                    if g > 0 {
                        let avail = (layer.energy[a] - scratch.adv_out_e[a]).max(0) / 4;
                        g = g.min(avail);
                    } else if g < 0 {
                        let avail = (layer.energy[b] - scratch.adv_out_e[b]).max(0) / 4;
                        g = g.max(-avail);
                    }
                    if g != 0 {
                        layer.energy[a] -= g;
                        layer.energy[b] += g;
                    }
                    layer.flow_s[a] = (scratch.qs[a][0] + scratch.qs[a][3])
                        .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                    if moved {
                        scratch.chunk_dirty[ci] = true;
                        scratch.chunk_dirty[((y + 1) / CHUNK_SIZE) * chunks_x + cx] = true;
                    }
                }
            }
        }
    }

    // Apply advected energy + stoichiometric RefSim reactions + ambient/smoke relaxation on needed chunks
    for ci in 0..num_chunks {
        if !scratch.chunk_needed[ci] {
            continue;
        }
        let cx = ci % chunks_x;
        let cy = ci / chunks_x;
        let x0 = cx * CHUNK_SIZE;
        let x1 = (x0 + CHUNK_SIZE).min(w);
        let y0 = cy * CHUNK_SIZE;
        let y1 = (y0 + CHUNK_SIZE).min(h);
        for y in y0..y1 {
            let row = y * w;
            for x in x0..x1 {
                if in_excluded_roi(x, y) {
                    continue;
                }
                let idx = row + x;
                let de_adv = scratch.adv_in_e[idx] - scratch.adv_out_e[idx];
                if de_adv != 0 {
                    layer.energy[idx] += de_adv;
                }
                if layer.energy[idx] > 0 {
                    let t_now = layer.cell_temp_ck(idx);
                    let mut xi = [0i32; NUM_REACTIONS];
                    let mut any_rxn = false;

                    // 0: Freeze
                    if t_now < REACTIONS[0].t_hi && layer.mass[SP_WATER][idx] > 0 {
                        let rate = ((27_315 - t_now) as f32 * 0.045)
                            .min(layer.mass[SP_WATER][idx] as f32 * 0.35);
                        xi[0] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[0] > 0;
                    }
                    // 1: Melt
                    if t_now > REACTIONS[1].t_lo && layer.mass[SP_ICE][idx] > 0 {
                        let rate = ((t_now - 27_315) as f32 * 0.030)
                            .min(layer.mass[SP_ICE][idx] as f32 * 0.30);
                        xi[1] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[1] > 0;
                    }
                    // 2: Boil
                    if t_now > REACTIONS[2].t_lo && layer.mass[SP_WATER][idx] > 0 {
                        let rate = ((t_now - 37_315) as f32 * 0.015)
                            .min(layer.mass[SP_WATER][idx] as f32 * 0.20);
                        xi[2] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[2] > 0;
                    }
                    // 3: Condense
                    if t_now < REACTIONS[3].t_hi && layer.mass[SP_STEAM][idx] > 0 {
                        let mut rate = ((37_000 - t_now) as f32 * 0.012)
                            .min(layer.mass[SP_STEAM][idx] as f32 * 0.25);
                        if laws.scale_raw {
                            rate *= laws.precip_mul;
                        }
                        xi[3] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[3] > 0;
                    }
                    // 4: WoodBurn
                    if t_now > REACTIONS[4].t_lo && layer.mass[SP_WOOD][idx] >= 2 {
                        let mut rate = ((t_now - 52_000) as f32 * 0.004).clamp(4.0, 45.0);
                        if laws.scale_raw {
                            rate *= laws.vol_mul;
                        }
                        xi[4] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[4] > 0;
                    }
                    // 5: OilBurn
                    if t_now > REACTIONS[5].t_lo && layer.mass[SP_OIL][idx] >= 1 {
                        let mut rate = ((t_now - 44_000) as f32 * 0.008).clamp(8.0, 85.0);
                        if laws.scale_raw {
                            rate *= laws.vol_mul;
                        }
                        xi[5] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[5] > 0;
                    }
                    // 6: PlantBurn
                    if t_now > REACTIONS[6].t_lo && layer.mass[SP_PLANT][idx] >= 2 {
                        let mut rate = ((t_now - 46_000) as f32 * 0.006).clamp(6.0, 60.0);
                        if laws.scale_raw {
                            rate *= laws.vol_mul;
                        }
                        xi[6] = rate.floor().min(2000.0) as i32;
                        any_rxn |= xi[6] > 0;
                    }

                    if any_rxn {
                        for r in 0..NUM_REACTIONS {
                            if xi[r] <= 0 {
                                continue;
                            }
                            let de = REACTIONS[r].delta_e_th();
                            if de < 0 {
                                let c_now = layer.cell_heat_capacity(idx);
                                let avail = (layer.energy[idx] - c_now * T_MIN).max(0);
                                xi[r] = xi[r].min((avail / (-de)) as i32).max(0);
                            } else if de > 0 {
                                let c_now = layer.cell_heat_capacity(idx).max(10_000);
                                let room = (c_now * T_MAX - layer.energy[idx]).max(0);
                                xi[r] = xi[r].min((room / de) as i32).max(0);
                            }
                        }
                        for s in 0..NUM_SPECIES {
                            let demand: i64 = (0..NUM_REACTIONS)
                                .filter(|&r| REACTIONS[r].nu[s] < 0)
                                .map(|r| (-REACTIONS[r].nu[s] as i64) * (xi[r] as i64))
                                .sum();
                            let avail = layer.mass[s][idx] as i64;
                            if demand > avail && demand > 0 {
                                for r in 0..NUM_REACTIONS {
                                    if REACTIONS[r].nu[s] < 0 {
                                        xi[r] = ((xi[r] as i64) * avail / demand) as i32;
                                    }
                                }
                            }
                        }
                        for r in 0..NUM_REACTIONS {
                            let x_int = xi[r];
                            if x_int <= 0 {
                                continue;
                            }
                            let spec = &REACTIONS[r];
                            for s in 0..NUM_SPECIES {
                                let dm = (spec.nu[s] as i32) * x_int;
                                if dm != 0 {
                                    layer.mass[s][idx] = (layer.mass[s][idx] as i32 + dm) as u16;
                                }
                            }
                            layer.energy[idx] += spec.delta_e_th() * (x_int as i64);
                            scratch.chunk_dirty[ci] = true;
                        }
                    }

                    let c_now = layer.cell_heat_capacity(idx);
                    if c_now > 0 {
                        let target_e = c_now * T_AMB;
                        let diff = layer.energy[idx] - target_e;
                        let cool = diff / 40;
                        if cool != 0 {
                            layer.energy[idx] -= cool;
                            layer.atmosphere_energy += cool;
                            if (cool / c_now).abs() >= 150 {
                                scratch.chunk_dirty[ci] = true;
                            }
                        }
                    }
                    let sm = layer.mass[SP_SMOKE][idx];
                    if sm > 0 {
                        let diss = ((sm / 10).max(1)).min(sm);
                        let c_cell = layer.cell_heat_capacity(idx).max(1);
                        let e_diss =
                            (diss as i64) * SPECIES[SP_SMOKE].c * (layer.energy[idx] / c_cell);
                        layer.mass[SP_SMOKE][idx] -= diss;
                        layer.energy[idx] -= e_diss;
                        layer.atmosphere_mass[SP_SMOKE] += diss as i64;
                        layer.atmosphere_energy += e_diss + (diss as i64) * SPECIES[SP_SMOKE].u0;
                        scratch.chunk_dirty[ci] = true;
                    }
                }
                if layer.forcing_energy[idx] != 0 || layer.forcing_mass[idx] != 0 {
                    layer.forcing_energy[idx] = 0;
                    layer.forcing_mass[idx] = 0;
                    scratch.chunk_dirty[ci] = true;
                }
            }
        }
    }
    layer.tick += 1;
}

/// Conservative vertical gravity, buoyancy & thermal conduction for one adjacent `(lo=z, hi=z+1)` pair.
#[allow(clippy::too_many_arguments)]
fn step_vertical_pair_conservative(
    z: usize,
    depth: usize,
    z_sea: usize,
    tick: u64,
    lo: &mut field_core::Field,
    hi: &mut field_core::Field,
    prev_t_lo: &[i32],
    prev_t_hi: &[i32],
    lo_flags: LayerFlags,
    hi_flags: LayerFlags,
    grav_min_mass: u16,
    standard_laws: bool,
    grav_fall_frac: f32,
    buoy_rise_frac: f32,
    cond_scale: f32,
    chunks_x: usize,
    dirty_lo: &mut [bool],
    dirty_hi: &mut [bool],
) {
    if !lo_flags.non_empty && !hi_flags.non_empty {
        return;
    }
    let w = lo.width;
    let len = w * lo.height;
    let deep_quiescent =
        z + 8 < z_sea && !hi_flags.has_liquid && !lo_flags.has_gas && !lo_flags.has_forcing && !hi_flags.has_forcing;
    if deep_quiescent && ((z as u64).wrapping_add(tick) % 8) != 0 {
        return;
    }

    let high_alt_buoyant = depth >= 48 && (z as f32) >= 0.65 * (depth as f32);
    let z_grav_fall_frac = if high_alt_buoyant {
        (grav_fall_frac * 0.45).clamp(0.05, 0.85)
    } else {
        grav_fall_frac
    };
    let z_buoy_rise_frac = if high_alt_buoyant {
        (buoy_rise_frac * 1.25).clamp(0.10, 0.85)
    } else {
        buoy_rise_frac
    };
    let use_std_grav = standard_laws && !high_alt_buoyant;
    let check_grav = hi_flags.has_liquid;
    let check_buoy = lo_flags.has_gas;
    let check_cond = lo_flags.non_empty && hi_flags.non_empty;

    for idx in 0..len {
        let e_lo = lo.energy[idx];
        let e_hi = hi.energy[idx];
        if e_lo == 0 && e_hi == 0 {
            continue;
        }
        let mut changed = false;

        // 1. GRAVITY: Water & Oil fall from `hi` (z+1) -> `lo` (z)
        if check_grav && e_hi > 0 {
            let hi_w = hi.mass[SP_WATER][idx];
            let hi_o = hi.mass[SP_OIL][idx];
            if hi_w > grav_min_mass || hi_o > grav_min_mass {
                let lo_blocked = lo.mass[SP_ROCK][idx] >= 500
                    || lo.mass[SP_SOIL][idx] >= 500
                    || lo.mass[SP_ICE][idx] >= 650;
                let hi_has_floor = hi.mass[SP_SOIL][idx] >= 350
                    || hi.mass[SP_ROCK][idx] >= 350
                    || hi.mass[SP_ICE][idx] >= 450;
                if !lo_blocked && !hi_has_floor {
                    for &sp in &[SP_WATER, SP_OIL] {
                        let avail = hi.mass[sp][idx];
                        if avail > grav_min_mass {
                            let room = H_CAP.saturating_sub(lo.cell_volume(idx)) as u16;
                            let q = if use_std_grav {
                                ((avail / 3).max(1)).min(room)
                            } else {
                                ((avail as f32 * z_grav_fall_frac).round() as u16)
                                    .max(1)
                                    .min(avail)
                                    .min(room)
                            };
                            if q > 0 {
                                let c_hi = hi.cell_heat_capacity(idx).max(1);
                                let t_hi = (hi.energy[idx] / c_hi).clamp(T_MIN, T_MAX);
                                let e_xfer = (q as i64) * SPECIES[sp].c * t_hi;
                                let e_clamped = e_xfer.min(hi.energy[idx].max(0));
                                hi.mass[sp][idx] -= q;
                                lo.mass[sp][idx] += q;
                                hi.energy[idx] -= e_clamped;
                                lo.energy[idx] += e_clamped;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // 2. BUOYANCY: Steam & Smoke rise from `lo` (z) -> `hi` (z+1)
        if check_buoy && e_lo > 0 {
            let lo_st = lo.mass[SP_STEAM][idx];
            let lo_sm = lo.mass[SP_SMOKE][idx];
            if lo_st > 12 || lo_sm > 12 {
                let hi_blocked = hi.mass[SP_ROCK][idx] >= 500
                    || hi.mass[SP_SOIL][idx] >= 500
                    || hi.mass[SP_ICE][idx] >= 650;
                if !hi_blocked {
                    for &sp in &[SP_STEAM, SP_SMOKE] {
                        let avail = lo.mass[sp][idx];
                        if avail > 12 {
                            let room = H_CAP.saturating_sub(hi.cell_volume(idx)) as u16;
                            let q = if use_std_grav {
                                ((avail / 3).max(1)).min(room)
                            } else {
                                ((avail as f32 * z_buoy_rise_frac).round() as u16)
                                    .max(1)
                                    .min(avail)
                                    .min(room)
                            };
                            if q > 0 {
                                let c_lo = lo.cell_heat_capacity(idx).max(1);
                                let t_lo = (lo.energy[idx] / c_lo).clamp(T_MIN, T_MAX);
                                let e_xfer = (q as i64) * SPECIES[sp].c * t_lo;
                                let e_clamped = e_xfer.min(lo.energy[idx].max(0));
                                lo.mass[sp][idx] -= q;
                                hi.mass[sp][idx] += q;
                                lo.energy[idx] -= e_clamped;
                                hi.energy[idx] += e_clamped;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // 3. VERTICAL THERMAL CONDUCTION (fast O(1) prev_temp pre-filter skips 1.6M redundant divisions on isothermal pairs!)
        if check_cond
            && lo.energy[idx] > 0
            && hi.energy[idx] > 0
            && (changed || (prev_t_lo[idx] - prev_t_hi[idx]).abs() > 1100)
        {
            let c_lo = lo.cell_heat_capacity(idx);
            let c_hi = hi.cell_heat_capacity(idx);
            if c_lo > 0 && c_hi > 0 {
                let t_lo = lo.energy[idx] / c_lo;
                let t_hi = hi.energy[idx] / c_hi;
                let dt = t_lo - t_hi;
                if dt.abs() > 1500 {
                    let base_rate = if dt > 0 { 22.0 } else { 64.0 };
                    let rate = if standard_laws {
                        if dt > 0 { 22 } else { 64 }
                    } else {
                        ((base_rate / cond_scale).round() as i64).max(6)
                    };
                    let g_raw = (dt * c_lo.min(c_hi)) / rate;
                    let g = if g_raw > 0 {
                        let avail = (lo.energy[idx] - c_lo * T_MIN).max(0) / 4;
                        let room = (c_hi * T_MAX - hi.energy[idx]).max(0) / 4;
                        g_raw.min(avail).min(room)
                    } else {
                        let avail = (hi.energy[idx] - c_hi * T_MIN).max(0) / 4;
                        let room = (c_lo * T_MAX - lo.energy[idx]).max(0) / 4;
                        g_raw.max(-avail).max(-room)
                    };
                    if g != 0 {
                        lo.energy[idx] -= g;
                        hi.energy[idx] += g;
                        if (g / c_lo).abs() >= 200 || (g / c_hi).abs() >= 200 {
                            changed = true;
                        }
                    }
                }
            }
        }

        if changed {
            let y = idx / w;
            let x = idx - y * w;
            let ci = (y / CHUNK_SIZE) * chunks_x + (x / CHUNK_SIZE);
            dirty_lo[ci] = true;
            dirty_hi[ci] = true;
        }
    }
}

/// Wave 39B: Ultra-fast multi-threaded conservative 64-layer shard stepper:
/// 1. Persistent zero-alloc scratch pool + quiescent bedrock/subsurface skip + `128×128×16` Hero ROI
///    projection / active-chunk RefSim (zero per-layer heap allocations!).
/// 2. Parallel strided conservation-exact ambient & altitude relaxation (`faucet_energy` tracked).
/// 3. Active-layer hydrologic condensation + arch collapse + parallel red-black `chunks_mut(2)` 3D
///    vertical gravity, buoyancy, and heat conduction (`mass_error == 0 && energy_error == 0` bit-exact!).
/// 4. Direct O(1) dirty-chunk bitset collection (eliminates redundant single-threaded 4.19M-voxel hashing!).
fn parallel_step_shard_volume(
    vf: &mut VolumeField,
    z_sea: usize,
    neural_roi: Option<&NeuralRoiSlab<'_>>,
    scratch_pool: &mut Vec<LayerScratch>,
) -> Vec<(u8, u16, u16)> {
    let depth = vf.layers.len();
    if depth == 0 || vf.width == 0 || vf.height == 0 {
        return Vec::new();
    }
    let w = vf.width;
    let h = vf.height;
    let area = w * h;
    let total = depth * area;
    let chunks_x = w.div_ceil(CHUNK_SIZE);
    let chunks_y = h.div_ceil(CHUNK_SIZE);
    let num_chunks = chunks_x * chunks_y;

    if vf.prev_temp.len() != total {
        vf.prev_temp.resize(total, T_AMB as i32);
    }
    if vf.prev_steam.len() != total {
        vf.prev_steam.resize(total, 0);
    }

    let laws = vf.world_laws;
    let law_scale = LawScaleParams {
        scale_raw: (laws.viscosity - 1.0).abs() > 1e-3
            || (laws.volatility - 1.0).abs() > 1e-3
            || (laws.thermal_cond - 1.0).abs() > 1e-3
            || (laws.precipitation - 1.0).abs() > 1e-3,
        liq_mul: (1.0 / laws.viscosity.clamp(0.25, 4.0)).clamp(0.25, 4.0),
        cond_mul: laws.thermal_cond.clamp(0.25, 4.0),
        precip_mul: laws.precipitation.clamp(0.1, 4.0),
        vol_mul: laws.volatility.clamp(0.2, 4.0),
    };
    let tick_now = vf.tick;
    let amb_shift = laws.ambient_temp_shift;
    let do_amb = amb_shift.is_finite() && amb_shift.abs() > 1.0;
    let t_amb_target = if do_amb {
        (T_AMB + amb_shift.round() as i64).clamp(T_MIN, T_MAX)
    } else {
        T_AMB
    };
    let z_abyss_hi = ((depth as f32) * 0.25).floor() as usize;
    let z_strato_lo = ((depth as f32) * 0.72).ceil() as usize;
    let relax_phase = (tick_now as usize) % 4;
    let z_bedrock_cut = z_sea.saturating_sub(8);
    let z_subsurface_cut = z_sea.saturating_sub(2);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
        .min(depth.max(1));

    // Ensure persistent per-worker scratch pool is warm (0 allocations after tick 0!)
    while scratch_pool.len() < n_threads {
        scratch_pool.push(LayerScratch::new(area, num_chunks));
    }
    for sc_buf in scratch_pool.iter_mut().take(n_threads) {
        sc_buf.ensure_size(area, num_chunks);
    }

    let mut layer_flags = vec![LayerFlags::default(); depth];
    let mut layer_dirty = vec![false; depth * num_chunks];

    // Phase 1 + Phase 2: Balanced multi-band parallel execution!
    // Splitting `[0..depth]` naively into `chunks_mut(8)` dumps ALL 16 active surface/ROI layers
    // (`z = 18..33`) onto only 2 of the 8 cores while the other 6 cores sit idle on empty sky / bedrock!
    // Instead, partition into 3 contiguous bands (`[0..hot_lo)`, `[hot_lo..hot_hi)`, `[hot_hi..depth)`)
    // and fan out EACH band across all `n_threads` cores so ALL 8 cores share the hot surface/ROI band!
    let hot_lo = neural_roi
        .map(|r| r.z0.min(z_subsurface_cut))
        .unwrap_or(z_subsurface_cut)
        .min(depth);
    let hot_hi = neural_roi
        .map(|r| (r.z0 + r.horiz.len()).max(z_sea + 12))
        .unwrap_or(z_sea + 12)
        .clamp(hot_lo, depth);

    let bands: [(usize, usize); 3] = [(0, hot_lo), (hot_lo, hot_hi), (hot_hi, depth)];
    for &(b_lo, b_hi) in &bands {
        let b_len = b_hi.saturating_sub(b_lo);
        if b_len == 0 {
            continue;
        }
        let b_threads = n_threads.min(b_len);
        let b_chunk_sz = b_len.div_ceil(b_threads);
        std::thread::scope(|sc| {
            let mut layer_chunks = vf.layers[b_lo..b_hi].chunks_mut(b_chunk_sz);
            let mut pt_chunks = vf.prev_temp[b_lo * area..b_hi * area].chunks_mut(b_chunk_sz * area);
            let mut ps_chunks = vf.prev_steam[b_lo * area..b_hi * area].chunks_mut(b_chunk_sz * area);
            let mut flag_chunks = layer_flags[b_lo..b_hi].chunks_mut(b_chunk_sz);
            let mut dirty_chunks =
                layer_dirty[b_lo * num_chunks..b_hi * num_chunks].chunks_mut(b_chunk_sz * num_chunks);
            let mut scratch_iter = scratch_pool.iter_mut();

            let mut base_z = b_lo;
            while let (
                Some(l_ch),
                Some(pt_ch),
                Some(ps_ch),
                Some(f_ch),
                Some(d_ch),
                Some(scratch),
            ) = (
                layer_chunks.next(),
                pt_chunks.next(),
                ps_chunks.next(),
                flag_chunks.next(),
                dirty_chunks.next(),
                scratch_iter.next(),
            ) {
                let start_z = base_z;
                base_z += l_ch.len();
                sc.spawn(move || {
                    for (li, layer) in l_ch.iter_mut().enumerate() {
                        let z = start_z + li;
                        let pt = &mut pt_ch[li * area..(li + 1) * area];
                        let ps = &mut ps_ch[li * area..(li + 1) * area];
                        scratch.chunk_dirty.fill(false);

                        let in_roi_z = neural_roi
                            .map(|r| z >= r.z0 && z < r.z0 + r.horiz.len())
                            .unwrap_or(false);

                        // Ultra-fast deep bedrock quiescent bypass (z < z_sea - 8, 7/8 ticks unless forced!)
                        if !in_roi_z
                            && z < z_bedrock_cut
                            && ((z as u64).wrapping_add(tick_now) % 8 != 0)
                            && !layer.forcing_energy.iter().any(|&e| e != 0)
                            && !layer.forcing_mass.iter().any(|&m| m != 0)
                        {
                            layer.tick += 1;
                            f_ch[li] = LayerFlags {
                                non_empty: true,
                                has_liquid: false,
                                has_gas: false,
                                has_wood_ice: false,
                                has_forcing: false,
                                has_hot_temp: false,
                            };
                            d_ch[li * num_chunks..(li + 1) * num_chunks].fill(false);
                            continue;
                        }

                        // Fast empty-sky bypass (top layers where all energies & forcings are 0)
                        if !in_roi_z
                            && z > z_sea + 12
                            && !layer.energy.iter().any(|&e| e != 0)
                            && !layer.forcing_energy.iter().any(|&e| e != 0)
                            && !layer.forcing_mass.iter().any(|&m| m != 0)
                        {
                            layer.tick += 1;
                            f_ch[li] = LayerFlags::default();
                            d_ch[li * num_chunks..(li + 1) * num_chunks].fill(false);
                            continue;
                        }

                        let mut fl = scan_layer_and_snapshot_sensors(
                            layer,
                            pt,
                            ps,
                            &mut scratch.chunk_active,
                            chunks_x,
                        );

                        if in_roi_z {
                            let roi = neural_roi.unwrap();
                            project_and_apply_roi_fast(
                                layer,
                                roi.x0,
                                roi.y0,
                                roi.roi_w,
                                roi.roi_h,
                                &roi.horiz[z - roi.z0],
                                law_scale,
                                scratch,
                                chunks_x,
                            );
                            // Outside the 128×128 Hero ROI: step immediately when forced/hot, or round-robin 1/4!
                            if (roi.roi_w < w || roi.roi_h < h)
                                && (fl.has_forcing
                                    || fl.has_hot_temp
                                    || ((z as u64).wrapping_add(tick_now) % 4 == 0))
                            {
                                step_layer_refsim_fast(
                                    layer,
                                    law_scale,
                                    scratch,
                                    chunks_x,
                                    chunks_y,
                                    Some((roi.x0, roi.y0, roi.roi_w, roi.roi_h)),
                                );
                            }
                            fl.non_empty = true;
                            fl.has_liquid = true;
                            fl.has_gas = true;
                        } else if fl.non_empty || fl.has_forcing {
                            let subsurface_quiescent = z < z_subsurface_cut
                                && !fl.has_forcing
                                && !fl.has_gas
                                && !fl.has_hot_temp;
                            if !subsurface_quiescent || ((z as u64).wrapping_add(tick_now) % 4 == 0) {
                                step_layer_refsim_fast(
                                    layer,
                                    law_scale,
                                    scratch,
                                    chunks_x,
                                    chunks_y,
                                    None,
                                );
                            } else {
                                layer.tick += 1;
                            }
                        } else {
                            layer.tick += 1;
                        }

                        // Phase 2: Strided conservation-exact ambient & altitude relaxation (zero energy_error!)
                        if fl.non_empty {
                            if do_amb
                                && (z >= z_subsurface_cut
                                    || ((z as u64).wrapping_add(tick_now) % 4 == 0))
                            {
                                let mut idx = relax_phase;
                                while idx < area {
                                    if layer.energy[idx] > 0 {
                                        let c = layer.cell_heat_capacity(idx);
                                        if c > 0 {
                                            let t = (layer.energy[idx] / c).clamp(T_MIN, T_MAX);
                                            let step_ck = (t_amb_target - t) / 8;
                                            if step_ck != 0 {
                                                let e_cur = layer.energy[idx];
                                                let e_next = (e_cur + step_ck * c)
                                                    .clamp(c * T_MIN, c * T_MAX);
                                                let delta = e_next - e_cur;
                                                if delta != 0 {
                                                    layer.energy[idx] = e_next;
                                                    layer.faucet_energy += delta;
                                                }
                                            }
                                        }
                                    }
                                    idx += 4;
                                }
                            }

                            if depth >= 48 {
                                if z >= 1 && z < z_abyss_hi {
                                    let mut idx = relax_phase;
                                    while idx < area {
                                        if layer.energy[idx] > 0 {
                                            let c = layer.cell_heat_capacity(idx);
                                            if c > 0 {
                                                let t = (layer.energy[idx] / c).clamp(T_MIN, T_MAX);
                                                if t < 33_000 {
                                                    let step_ck = ((33_000 - t) / 16)
                                                        .max(60)
                                                        .min(33_000 - t);
                                                    let e_cur = layer.energy[idx];
                                                    let e_next = (e_cur + step_ck * c)
                                                        .clamp(c * T_MIN, c * T_MAX);
                                                    let delta = e_next - e_cur;
                                                    if delta != 0 {
                                                        layer.energy[idx] = e_next;
                                                        layer.faucet_energy += delta;
                                                    }
                                                }
                                            }
                                        }
                                        idx += 4;
                                    }
                                } else if z >= z_strato_lo {
                                    let mut idx = relax_phase;
                                    while idx < area {
                                        if layer.energy[idx] > 0 {
                                            let c = layer.cell_heat_capacity(idx);
                                            if c > 0 {
                                                let t = (layer.energy[idx] / c).clamp(T_MIN, T_MAX);
                                                if t > 26_600 {
                                                    let step_ck = ((t - 26_600) / 16)
                                                        .max(60)
                                                        .min(t - 26_600);
                                                    let e_cur = layer.energy[idx];
                                                    let e_next = (e_cur - step_ck * c)
                                                        .clamp(c * T_MIN, c * T_MAX);
                                                    let delta = e_next - e_cur;
                                                    if delta != 0 {
                                                        layer.energy[idx] = e_next;
                                                        layer.faucet_energy += delta;
                                                    }
                                                }
                                            }
                                        }
                                        idx += 4;
                                    }
                                }
                            }
                        }

                        f_ch[li] = fl;
                        d_ch[li * num_chunks..(li + 1) * num_chunks]
                            .copy_from_slice(&scratch.chunk_dirty);
                    }
                });
            }
        });
    }

    // Phase 3a: Hydrologic Cloud-Top Condensation (`depth >= 12`, ONLY on cloud layers with steam!)
    if depth >= 12 {
        let precip = laws.precipitation.clamp(0.1, 4.0);
        let amb_shift_sky = (laws.ambient_temp_shift * 0.5).round() as i64;
        let cloud_layers = if precip > 1.5 { 5usize } else { 3usize };
        let z_cloud_lo = depth.saturating_sub(cloud_layers);
        let is_glacial = vf.biome_id == BIOME_GLACIER_GORGE;

        for z in z_cloud_lo..depth {
            if !layer_flags[z].has_gas {
                continue;
            }
            let z_from_top = (depth - 1) - z;
            let base_sky: i64 = if is_glacial {
                match z_from_top {
                    0 => 25_600,
                    1 => 26_400,
                    _ => 27_000,
                }
            } else {
                match z_from_top {
                    0 => 26_800,
                    1 => 29_800,
                    _ => 32_800,
                }
            };
            let t_sky = if precip > 1.5 && laws.ambient_temp_shift > -2000.0 {
                (base_sky + amb_shift_sky).max(28_200).clamp(T_MIN, T_MAX)
            } else {
                (base_sky + amb_shift_sky).clamp(T_MIN, T_MAX)
            };
            let f = &mut vf.layers[z];
            let d_row = &mut layer_dirty[z * num_chunks..(z + 1) * num_chunks];
            for idx in 0..area {
                let steam = f.mass[SP_STEAM][idx];
                if steam < 8 {
                    continue;
                }
                let c = f.cell_heat_capacity(idx);
                if c > 0 {
                    let t = f.cell_temp_ck(idx);
                    if t > t_sky {
                        let cool_ck = ((t - t_sky) / 2).max(1_200).min(t - t_sky);
                        let min_e = c * T_MIN;
                        let delta_cool = (cool_ck * c).min((f.energy[idx] - min_e).max(0));
                        if delta_cool > 0 {
                            f.energy[idx] -= delta_cool;
                            f.faucet_energy -= delta_cool;
                        }
                    }
                }
                let t_now = f.cell_temp_ck(idx);
                if t_now < 37_000 {
                    let target_sp = if t_now < 27_315 { SP_ICE } else { SP_WATER };
                    let dm = if (precip - 1.0).abs() <= 1e-4 {
                        ((steam / 2).max(1)).min(steam)
                    } else {
                        let frac = (0.50 * precip).clamp(0.05, 0.95);
                        ((steam as f32 * frac).round() as u16).max(1).min(steam)
                    };
                    if dm > 0 {
                        let e_before = f.energy[idx];
                        f.mass[SP_STEAM][idx] -= dm;
                        f.mass[target_sp][idx] += dm;
                        let c_new = f.cell_heat_capacity(idx);
                        let e_new = c_new * t_now.clamp(T_MIN, T_MAX);
                        let delta_e_th = e_new - e_before;
                        let delta_e_chem =
                            (dm as i64) * (SPECIES[target_sp].u0 - SPECIES[SP_STEAM].u0);
                        f.energy[idx] = e_new;
                        f.faucet_energy += delta_e_th + delta_e_chem;
                        layer_flags[z].has_liquid = true;
                        let y = idx / w;
                        let x = idx - y * w;
                        d_row[(y / CHUNK_SIZE) * chunks_x + (x / CHUNK_SIZE)] = true;
                    }
                }
            }
        }

        // Phase 3b: Destructible Bridge/Arch Collapse (ONLY on layers with wood/ice!)
        for z in 2..depth {
            if !layer_flags[z].has_wood_ice {
                continue;
            }
            let prev_base = z * area;
            let prev_t_slice = vf.prev_temp.get(prev_base..prev_base + area);
            let (lo_slice, hi_slice) = vf.layers.split_at_mut(z);
            let lo = &mut lo_slice[z - 1];
            let hi = &mut hi_slice[0];
            for y in 0..h {
                let row = y * w;
                for x in 0..w {
                    let idx = row + x;
                    let wood = hi.mass[SP_WOOD][idx];
                    let ice = hi.mass[SP_ICE][idx];
                    if wood < 15 && ice < 15 {
                        continue;
                    }
                    if field_solid_density_fast(lo, idx) >= 0.25 {
                        continue;
                    }
                    let t_ck = hi.cell_temp_ck(idx);
                    let prev_t_ck = prev_t_slice.map(|s| s[idx] as i64).unwrap_or(t_ck);
                    let t_peak = t_ck.max(prev_t_ck);
                    let wood_weak = wood >= 15 && t_peak > 55_000;
                    let ice_weak = ice >= 15 && t_peak > 27_800;
                    let snow_fall = z >= z_cloud_lo && ice >= 15 && ice < 400 && t_ck < 27_315;
                    if !wood_weak && !ice_weak && !snow_fall {
                        continue;
                    }
                    let mut solid_nb = 0u8;
                    if x > 0 && field_solid_density_fast(hi, idx - 1) >= 0.25 {
                        solid_nb += 1;
                    }
                    if x + 1 < w && field_solid_density_fast(hi, idx + 1) >= 0.25 {
                        solid_nb += 1;
                    }
                    if y > 0 && field_solid_density_fast(hi, idx - w) >= 0.25 {
                        solid_nb += 1;
                    }
                    if y + 1 < h && field_solid_density_fast(hi, idx + w) >= 0.25 {
                        solid_nb += 1;
                    }
                    if solid_nb >= 2 {
                        continue;
                    }
                    for &(sp, active) in &[(SP_WOOD, wood_weak), (SP_ICE, ice_weak || snow_fall)] {
                        if !active {
                            continue;
                        }
                        let avail = hi.mass[sp][idx];
                        if avail == 0 {
                            continue;
                        }
                        let room = H_CAP.saturating_sub(lo.cell_volume(idx)) as u16;
                        let q = avail.min(room);
                        if q > 0 {
                            let c_hi = hi.cell_heat_capacity(idx);
                            let e_xfer = if q == avail && hi.cell_volume(idx) == (q as u32) {
                                hi.energy[idx]
                            } else if c_hi > 0 {
                                let e_prop = ((q as i128)
                                    * (SPECIES[sp].c as i128)
                                    * (hi.energy[idx].max(0) as i128)
                                    / (c_hi as i128)) as i64;
                                e_prop.min(hi.energy[idx].max(0))
                            } else {
                                0
                            };
                            hi.mass[sp][idx] -= q;
                            lo.mass[sp][idx] += q;
                            hi.energy[idx] -= e_xfer;
                            lo.energy[idx] += e_xfer;
                            layer_flags[z - 1].non_empty = true;
                            let ci = (y / CHUNK_SIZE) * chunks_x + (x / CHUNK_SIZE);
                            layer_dirty[(z - 1) * num_chunks + ci] = true;
                            layer_dirty[z * num_chunks + ci] = true;
                        }
                    }
                }
            }
        }
    }

    // Phase 3c: Parallel Red-Black 3D Vertical Gravity, Buoyancy & Conduction (`chunks_mut(2)`, 100% safe!)
    if depth >= 2 {
        let grav_min_mass: u16 = if depth >= 12 { 8 } else { 20 };
        let standard_laws = laws.is_standard();
        let grav_fall_frac =
            (0.3333 * (laws.gravity_g.clamp(0.15, 2.5) / laws.viscosity.clamp(0.25, 4.0)))
                .clamp(0.08, 0.85);
        let buoy_rise_frac = (0.3333
            * (1.0 / laws.gravity_g.clamp(0.25, 2.5)).sqrt()
            * (1.0 + (laws.ambient_temp_shift.max(0.0) / 11_000.0)))
        .clamp(0.10, 0.85);
        let cond_scale = laws.thermal_cond.clamp(0.25, 4.0);
        let prev_temp_ref = &vf.prev_temp;

        for parity in 0..2usize {
            if parity == 1 && depth < 3 {
                break;
            }
            let pairs_total = (depth - parity) / 2;
            if pairs_total == 0 {
                continue;
            }
            let pairs_per_thread = pairs_total.div_ceil(n_threads);
            let slice_layers = &mut vf.layers[parity..parity + pairs_total * 2];
            let slice_dirty =
                &mut layer_dirty[parity * num_chunks..(parity + pairs_total * 2) * num_chunks];
            let flags_ref = &layer_flags;

            std::thread::scope(|sc| {
                let mut l_groups = slice_layers.chunks_mut(pairs_per_thread * 2);
                let mut d_groups = slice_dirty.chunks_mut(pairs_per_thread * 2 * num_chunks);
                let mut pair_idx = 0usize;
                while let (Some(lg), Some(dg)) = (l_groups.next(), d_groups.next()) {
                    let start_pair = pair_idx;
                    let group_pairs = lg.len() / 2;
                    pair_idx += group_pairs;
                    sc.spawn(move || {
                        for p in 0..group_pairs {
                            let z = parity + (start_pair + p) * 2;
                            let pair_l = &mut lg[p * 2..(p + 1) * 2];
                            let (lo_s, hi_s) = pair_l.split_at_mut(1);
                            let pair_d = &mut dg[p * 2 * num_chunks..(p + 1) * 2 * num_chunks];
                            let (d_lo, d_hi) = pair_d.split_at_mut(num_chunks);
                            let pt_lo = &prev_temp_ref[z * area..(z + 1) * area];
                            let pt_hi = &prev_temp_ref[(z + 1) * area..(z + 2) * area];
                            step_vertical_pair_conservative(
                                z,
                                depth,
                                z_sea,
                                tick_now,
                                &mut lo_s[0],
                                &mut hi_s[0],
                                pt_lo,
                                pt_hi,
                                flags_ref[z],
                                flags_ref[z + 1],
                                grav_min_mass,
                                standard_laws,
                                grav_fall_frac,
                                buoy_rise_frac,
                                cond_scale,
                                chunks_x,
                                d_lo,
                                d_hi,
                            );
                        }
                    });
                }
            });

            // Propagate flags after Pass 0 so Pass 1 never misses newly fallen water or risen steam
            if parity == 0 {
                for p in 0..pairs_total {
                    let z = p * 2;
                    if layer_flags[z + 1].has_liquid {
                        layer_flags[z].has_liquid = true;
                        layer_flags[z].non_empty = true;
                    }
                    if layer_flags[z].has_gas {
                        layer_flags[z + 1].has_gas = true;
                        layer_flags[z + 1].non_empty = true;
                    }
                }
            }
        }
    }

    vf.tick += 1;
    vf.world_time_secs += 0.1;

    // Collect dirty `(layer, cx, cy)` directly from our dirty bitset (O(depth * 256) = < 0.02 ms!)
    let mut dirty_out = Vec::new();
    for z in 0..depth {
        let base = z * num_chunks;
        for ci in 0..num_chunks {
            if layer_dirty[base + ci] {
                let cx = (ci % chunks_x) as u16;
                let cy = (ci / chunks_x) as u16;
                dirty_out.push((z as u8, cx, cy));
            }
        }
    }
    dirty_out
}

/// Wave 39B: Fast conservation-exact Macro -> Micro climate downscale matching
/// `DualScalePlanet::apply_climate_to_shard`, skipping empty sky layers in O(1) via `energy == 0`!
fn apply_climate_to_shard_fast(planet: &mut DualScalePlanet, key: PlanetShardKey) {
    let (avg_temp_k, avg_precip, _wu, _wv, _ash) = planet
        .climate
        .sample_shard_average(key, planet.config.shards_per_face_axis);
    let Some(vf) = planet.shards.get_mut(&key) else {
        return;
    };

    vf.world_laws.ambient_temp_shift = ((avg_temp_k - 290.0) * 100.0).clamp(-5500.0, 5500.0);
    vf.world_laws.precipitation = (0.45 + avg_precip * 1.65).clamp(0.25, 2.8);

    let w = vf.width;
    let h = vf.height;
    let d = vf.depth;
    let target_t = ((avg_temp_k * 100.0) as i64).clamp(T_MIN + 2000, T_MAX - 2000);
    let stride = 4usize;
    let phase = (planet.tick as usize) % stride;
    let cap_z = d.saturating_sub(2);

    // Pre-identify non-empty sky ceiling so we never scan empty upper troposphere layers!
    let mut top_active_z = 0usize;
    for z in (0..cap_z).rev() {
        if vf.layers[z].energy.iter().any(|&e| e != 0) {
            top_active_z = z + 1;
            break;
        }
    }
    if top_active_z == 0 {
        return;
    }

    for y in 0..h {
        let row = y * w;
        let x_start = (stride + phase - (y % stride)) % stride;
        let mut x = x_start;
        while x < w {
            let idx = row + x;
            for z in (0..top_active_z).rev() {
                let layer = &mut vf.layers[z];
                if layer.energy[idx] == 0 {
                    continue;
                }
                if layer.cell_volume(idx) >= 80 {
                    let c = layer.cell_heat_capacity(idx);
                    if c > 0 {
                        let cur_t = (layer.energy[idx] / c).clamp(T_MIN, T_MAX);
                        if cur_t > 70_000 && layer.mass[SP_ROCK][idx] >= 400 {
                            break;
                        }
                        let diff_t = target_t - cur_t;
                        let nudge_t = (diff_t / 12).clamp(-300, 300);
                        if nudge_t != 0 {
                            let desired_e = layer.energy[idx] + nudge_t * c;
                            let clamped_e = desired_e.clamp(c * T_MIN, c * T_MAX);
                            let delta = clamped_e - layer.energy[idx];
                            if delta != 0 {
                                layer.energy[idx] += delta;
                                layer.faucet_energy += delta;
                            }
                        }
                    }
                    break;
                }
            }
            x += stride;
        }
    }
}

/// Wave 39B: Single-pass parallel exact integer conservation audit matching
/// `(planet.global_mass_error(), planet.global_energy_error())` identically to the bit!
fn parallel_global_conservation_errors(planet: &DualScalePlanet) -> (i64, i64) {
    let mut total_mass = 0i64;
    let mut total_energy = 0i64;

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8);

    for vf in planet.shards.values() {
        let depth = vf.layers.len();
        if depth == 0 {
            continue;
        }
        let chunk_sz = depth.div_ceil(n_threads.min(depth));
        let (shard_m, shard_e) = std::thread::scope(|sc| {
            let mut handles = Vec::new();
            for l_chunk in vf.layers.chunks(chunk_sz) {
                handles.push(sc.spawn(move || {
                    let mut m_acc = 0i64;
                    let mut e_acc = 0i64;
                    for f in l_chunk {
                        let e_grid: i64 = f.energy.iter().sum();
                        e_acc += e_grid + f.atmosphere_energy - f.faucet_energy;
                        for s in 0..NUM_SPECIES {
                            let grid_s: i64 = f.mass[s].iter().map(|&v| v as i64).sum();
                            m_acc += grid_s + f.atmosphere_mass[s] - f.faucet_mass[s];
                            if SPECIES[s].u0 != 0 {
                                e_acc += grid_s * SPECIES[s].u0;
                            }
                        }
                    }
                    (m_acc, e_acc)
                }));
            }
            let mut sm = 0i64;
            let mut se = 0i64;
            for h in handles {
                let (lm, le) = h.join().expect("conservation audit worker panicked");
                sm += lm;
                se += le;
            }
            (sm, se)
        });
        total_mass += shard_m;
        total_energy += shard_e;
    }

    (
        total_mass - planet.initial_hydrated_mass,
        total_energy - planet.initial_hydrated_energy,
    )
}

fn pack_climate_rgba_from_shell(
    shell: &field_core::planet_macro::GlobalClimateShell,
) -> [Vec<u8>; FACE_COUNT] {
    let src_res = shell.res.max(1);
    std::array::from_fn(|f| {
        let cf = &shell.faces[f];
        let mut out = vec![0u8; CLIMATE_FACE_RGBA_BYTES];
        for y in 0..CLIMATE_GRID_N {
            let sy = (y * src_res) / CLIMATE_GRID_N;
            for x in 0..CLIMATE_GRID_N {
                let sx = (x * src_res) / CLIMATE_GRID_N;
                let s_idx = sy * src_res + sx;
                let o = (y * CLIMATE_GRID_N + x) * 4;
                let temp_k = cf.temp_k[s_idx];
                let r = (((temp_k - 240.0) / 80.0).clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (cf.precip[s_idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b = (cf.ash[s_idx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let snow = ((273.15 - temp_k) / 22.0).clamp(0.0, 1.0);
                let a = (snow * 255.0).round() as u8;
                out[o] = r;
                out[o + 1] = g;
                out[o + 2] = b;
                out[o + 3] = a;
            }
        }
        out
    })
}

fn pack_climate_wind_from_shell(
    shell: &field_core::planet_macro::GlobalClimateShell,
) -> [Vec<f32>; FACE_COUNT] {
    let src_res = shell.res.max(1);
    std::array::from_fn(|f| {
        let cf = &shell.faces[f];
        let mut out = vec![0.0f32; CLIMATE_FACE_WIND_FLOATS];
        for y in 0..CLIMATE_GRID_N {
            let sy = (y * src_res) / CLIMATE_GRID_N;
            for x in 0..CLIMATE_GRID_N {
                let sx = (x * src_res) / CLIMATE_GRID_N;
                let s_idx = sy * src_res + sx;
                let o = (y * CLIMATE_GRID_N + x) * 2;
                out[o] = cf.wind_u[s_idx];
                out[o + 1] = cf.wind_v[s_idx];
            }
        }
        out
    })
}

fn sync_shell_from_climate_pack(
    shell: &mut field_core::planet_macro::GlobalClimateShell,
    rgba_faces: &[Vec<u8>; FACE_COUNT],
    wind_faces: &[Vec<f32>; FACE_COUNT],
) {
    let dst_res = shell.res.max(1);
    for f in 0..FACE_COUNT {
        let cf = &mut shell.faces[f];
        let rgba = &rgba_faces[f];
        let wuv = &wind_faces[f];
        for y in 0..dst_res {
            let cy = (y * CLIMATE_GRID_N) / dst_res;
            for x in 0..dst_res {
                let cx = (x * CLIMATE_GRID_N) / dst_res;
                let pix = cy * CLIMATE_GRID_N + cx;
                let d_idx = y * dst_res + x;
                let o4 = pix * 4;
                let o2 = pix * 2;
                let temp_k = 240.0 + (rgba[o4] as f32 / 255.0) * 80.0;
                // Keep incandescent volcanic caldera magma hotspots intact while updating ambient air
                if cf.temp_k[d_idx] < 600.0 {
                    cf.temp_k[d_idx] = temp_k;
                }
                cf.precip[d_idx] = rgba[o4 + 1] as f32 / 255.0;
                cf.ash[d_idx] = rgba[o4 + 2] as f32 / 255.0;
                cf.wind_u[d_idx] = wuv[o2];
                cf.wind_v[d_idx] = wuv[o2 + 1];
            }
        }
    }
    shell.tick += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op13_climate_request_response_framing() {
        let inj = [
            ClimateInjection {
                face: 4,
                u: 0.25,
                v: -0.5,
                delta_temp: 1.2,
                delta_ash: 2.0,
            },
            ClimateInjection {
                face: 0,
                u: -0.75,
                v: 0.1,
                delta_temp: 0.5,
                delta_ash: 0.9,
            },
        ];
        let req = encode_planet_op13_request(33_033, [0.6, 0.3, 0.74], 2, &inj);
        assert_eq!(req.len(), 28 + 2 * 20);
        assert_eq!(u64::from_le_bytes(req[0..8].try_into().unwrap()), 33_033);
        assert_eq!(u32::from_le_bytes(req[20..24].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(req[24..28].try_into().unwrap()), 2);
        assert_eq!(req[28], 4);
        assert_eq!(req[48], 0);

        let mut resp = vec![0u8; CLIMATE_RESP_BYTES];
        resp[0..4].copy_from_slice(&5.9f32.to_le_bytes());
        resp[4..8].copy_from_slice(&287.5f32.to_le_bytes());
        resp[8] = 180;
        resp[9] = 120;
        resp[10] = 64;
        resp[11] = 20;
        let wind_off = 8 + CLIMATE_RGBA_BYTES;
        resp[wind_off..wind_off + 4].copy_from_slice(&1.25f32.to_le_bytes());
        resp[wind_off + 4..wind_off + 8].copy_from_slice(&(-0.75f32).to_le_bytes());

        let dec = decode_planet_op13_response(&resp).expect("decode op=13");
        assert!((dec.infer_ms - 5.9).abs() < 1e-4);
        assert!((dec.mean_temp_k - 287.5).abs() < 1e-4);
        assert_eq!(&dec.rgba_faces[0][0..4], &[180, 120, 64, 20]);
        assert!((dec.wind_uv_faces[0][0] - 1.25).abs() < 1e-5);
        assert!((dec.wind_uv_faces[0][1] + 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_sample_local_climate_dynamic_op13_and_uv_mapping() {
        let mut cfg = MacroPlanetConfig::default();
        cfg.shard_size = 32;
        cfg.depth = 32;
        cfg.shards_per_face_axis = 4;
        let mut mr = MacroPlanetRuntime::new(cfg);

        // Build a synthetic live op=13 climate response with distinct warm cloudy region vs cold clear region
        let face = mr.hero_site.shard_key.face as usize;
        let n_virt = mr.face_size_virtual() as f64;
        let (u_hot, v_hot) = face_virtual_uv_f64(
            mr.hero_site.beach_gx as f64,
            mr.hero_site.beach_gy as f64,
            n_virt as usize,
        );
        let ix_hot = ((((u_hot + 1.0) * 0.5) * CLIMATE_GRID_N as f64 - 0.5).round() as isize)
            .clamp(0, CLIMATE_GRID_N as isize - 1) as usize;
        let iy_hot = ((((v_hot + 1.0) * 0.5) * CLIMATE_GRID_N as f64 - 0.5).round() as isize)
            .clamp(0, CLIMATE_GRID_N as isize - 1) as usize;

        let mut rgba_faces: [Vec<u8>; FACE_COUNT] =
            std::array::from_fn(|_| vec![0u8; CLIMATE_FACE_RGBA_BYTES]);
        let mut wind_uv_faces: [Vec<f32>; FACE_COUNT] =
            std::array::from_fn(|_| vec![0.0f32; CLIMATE_FACE_WIND_FLOATS]);

        for iy in 0..CLIMATE_GRID_N {
            for ix in 0..CLIMATE_GRID_N {
                let pix = iy * CLIMATE_GRID_N + ix;
                let o4 = pix * 4;
                let o2 = pix * 2;
                let d2 = ((ix as f32 - ix_hot as f32).powi(2)
                    + (iy as f32 - iy_hot as f32).powi(2))
                .sqrt();
                if d2 <= 5.0 {
                    // Warm tropical storm cell (~305 K = +31.85 C, ~78% cloud cover, strong Coriolis wind)
                    rgba_faces[face][o4] = 208;
                    rgba_faces[face][o4 + 1] = 200;
                    wind_uv_faces[face][o2] = 4.25;
                    wind_uv_faces[face][o2 + 1] = -2.10;
                } else {
                    // Crisp cold clear zone (~262 K = -11.15 C, 5% clouds)
                    rgba_faces[face][o4] = 70;
                    rgba_faces[face][o4 + 1] = 13;
                    wind_uv_faces[face][o2] = -1.10;
                    wind_uv_faces[face][o2 + 1] = 0.65;
                }
            }
        }

        let op13 = PlanetClimateOp13Response {
            infer_ms: 5.4,
            mean_temp_k: 286.0,
            rgba_faces,
            wind_uv_faces,
        };
        mr.apply_climate_step(Some(Ok(op13)), &[]);
        assert!(mr.climate_neural_active);

        let (wu_hot, wv_hot, tc_hot, cc_hot) =
            mr.sample_local_climate(face as u8, mr.hero_site.beach_gx, mr.hero_site.beach_gy);
        assert!(
            cc_hot > 0.55,
            "expected live op=13 cloud cover > 0.55 at hero beach, got {}",
            cc_hot
        );
        assert!(
            tc_hot > 24.0,
            "expected warm op=13 local_temp_c > 24C at hero beach, got {}",
            tc_hot
        );
        assert!(
            wu_hot.hypot(wv_hot) > 3.0,
            "expected strong live Coriolis wind at hero beach, got ({}, {})",
            wu_hot,
            wv_hot
        );

        // Far corner on the same face must reflect the distinct cold/clear climate cell!
        let far_x = if mr.hero_site.beach_gx > (n_virt as f32) * 0.5 {
            2.0
        } else {
            (n_virt as f32) - 2.0
        };
        let far_y = if mr.hero_site.beach_gy > (n_virt as f32) * 0.5 {
            2.0
        } else {
            (n_virt as f32) - 2.0
        };
        let (_wu_cold, _wv_cold, tc_cold, cc_cold) =
            mr.sample_local_climate(face as u8, far_x, far_y);
        assert!(
            tc_cold < 0.0 && (tc_hot - tc_cold) > 25.0,
            "expected dynamic temperature contrast across op=13 face: hot={}C cold={}C",
            tc_hot,
            tc_cold
        );
        assert!(
            cc_cold < cc_hot * 0.25,
            "expected cloud cover contrast across op=13 face: hot={} cold={}",
            cc_hot,
            cc_cold
        );
    }

    #[test]
    fn test_subbrick_refresh_budget_priority_and_world_events() {
        let mut cfg = MacroPlanetConfig::default();
        cfg.shard_size = 64; // 4x4x2 = 32 sub-bricks per shard (test tighter budget = 10)
        cfg.depth = 32;
        cfg.shards_per_face_axis = 4;
        let mut mr = MacroPlanetRuntime::new(cfg);
        mr.subbrick_refresh_budget = 10;
        mr.subbrick_sweep_quota = 4;

        let hk = mr.hero_site.shard_key;
        let (hf, hx, hy, hz, _, _) = mr.human_spawn_point();
        mr.set_priority_focus(&[(hf, hx, hy)]);

        // Artificially dirty all 32 sub-bricks in the shard to simulate a heavy tick
        for bz in 0..mr.subbricks_z() {
            for by in 0..mr.shard_chunks_xy() {
                for bx in 0..mr.shard_chunks_xy() {
                    mr.pending_dirty_subbricks.insert((hk, bx, by, bz));
                }
            }
        }
        // Mark one specific urgent sub-brick via priority helper
        mr.mark_priority_subbricks_around(hf, hx, hy, hz, 4.0);
        let (_, lx, ly) = mr.virtual_to_shard(hf, hx, hy);
        let player_sb = (
            hk,
            (lx as usize / CHUNK_SIZE) as u16,
            (ly as usize / CHUNK_SIZE) as u16,
            (hz as usize / CHUNK_SIZE).min(mr.subbricks_z() as usize - 1) as u16,
        );

        let out1 = mr.step_macro_tick(None, &[], None);
        assert!(
            out1.dirty_subbricks.len() <= 10,
            "must enforce per-tick sub-brick refresh budget <= 10, got {}",
            out1.dirty_subbricks.len()
        );
        assert!(
            out1.dirty_subbricks.contains(&player_sb),
            "observer/priority sub-brick {:?} must refresh immediately on tick 1",
            player_sb
        );
        assert_eq!(out1.global_mass_error, 0);
        assert_eq!(out1.global_energy_error, 0);
        assert!(out1.cpu_step_ms >= 0.0);

        // Verify periodic natural world events alternate cleanly at tick % 450 == 120 with 0/0 error
        mr.planet.tick = MACRO_WORLD_EVENT_PHASE_TICK - 1;
        let ev_tick_1 = mr.step_macro_tick(None, &[], None);
        let ev1 = ev_tick_1
            .world_event
            .expect("tick 120 must trigger Volcanic Rift Geothermal Surge");
        assert_eq!(ev1.kind, MACRO_EVENT_VOLCANIC_SURGE);
        assert_eq!(ev_tick_1.global_mass_error, 0);
        assert_eq!(ev_tick_1.global_energy_error, 0);

        mr.planet.tick = MACRO_WORLD_EVENT_PERIOD_TICKS + MACRO_WORLD_EVENT_PHASE_TICK - 1;
        let ev_tick_2 = mr.step_macro_tick(None, &[], None);
        let ev2 = ev_tick_2
            .world_event
            .expect("tick 570 must trigger Coastal Storm Squall");
        assert_eq!(ev2.kind, MACRO_EVENT_COASTAL_SQUALL);
        assert_eq!(ev_tick_2.global_mass_error, 0);
        assert_eq!(ev_tick_2.global_energy_error, 0);
    }

    #[test]
    fn test_wave39b_sub25ms_full_256x256x64_and_roi_conservation() {
        let cfg = MacroPlanetConfig::default();
        let mut mr = MacroPlanetRuntime::new(cfg);
        let hk = mr.hero_site.shard_key;
        let (hf, hx, hy, _, _, _) = mr.human_spawn_point();
        mr.set_priority_focus(&[(hf, hx, hy)]);
        mr.planet.apply_climate_to_shard(hk);

        // 1. Verify 128×128×16 Hero ROI geometry & 16-voxel brick alignment
        let (rx0, ry0, roi_w, roi_h, rz0, rz1) =
            mr.active_op11_hero_roi(256, 256, 64, &[(hf, hx, hy)]);
        assert_eq!(roi_w, OP11_HERO_ROI_SIZE);
        assert_eq!(roi_h, OP11_HERO_ROI_SIZE);
        assert_eq!(rz1 - rz0, OP11_ACTIVE_SLAB_LAYERS);
        assert_eq!(rx0 % CHUNK_SIZE, 0, "rx0={} must be 16-voxel aligned", rx0);
        assert_eq!(ry0 % CHUNK_SIZE, 0, "ry0={} must be 16-voxel aligned", ry0);
        assert!(rx0 + roi_w <= 256 && ry0 + roi_h <= 256);

        // 2. Test adversarial/large neural 128×128×16 ROI fluxes (including nonzero perimeter fluxes!)
        //    and prove ZERO boundary leak -> global_mass_error == 0 && global_energy_error == 0!
        let mut adv_roi: Vec<RawStepOutput> = (0..(rz1 - rz0))
            .map(|_| {
                let mut r = RawStepOutput::zeros(roi_w, roi_h);
                for c in 0..OUTPUT_CHANNELS {
                    let base = c * roi_w * roi_h;
                    for i in 0..(roi_w * roi_h) {
                        r.data[base + i] = if (i + c) % 2 == 0 { 0.35 } else { -0.35 };
                    }
                }
                r
            })
            .collect();
        // Inject extreme fluxes directly along the ROI perimeter edges to verify exact zero-leak clamping
        for r in &mut adv_roi {
            for ly in 0..roi_h {
                r.set(0, ly * roi_w + (roi_w - 1), 999.0);
                r.set(8, ly * roi_w + (roi_w - 1), 999.0);
            }
            for lx in 0..roi_w {
                r.set(4, (roi_h - 1) * roi_w + lx, 999.0);
                r.set(9, (roi_h - 1) * roi_w + lx, 999.0);
            }
        }

        // Warmup tick + timed ticks (both RefSim & 128×128×16 neural ROI ticks)
        let mut refsim_ms = Vec::new();
        let mut roi_ms = Vec::new();
        for i in 0..6 {
            let outcome = if i % 2 == 1 {
                let slab = NeuralRoiSlab {
                    x0: rx0,
                    y0: ry0,
                    roi_w,
                    roi_h,
                    z0: rz0,
                    horiz: &adv_roi,
                    z_flux: None,
                    op11_ms: 14.0,
                    active_frac: 0.0625,
                };
                mr.step_macro_tick_roi_slab(None, &[], Some(slab))
            } else {
                mr.step_macro_tick_roi_slab(None, &[], None)
            };
            assert_eq!(outcome.global_mass_error, 0, "tick {} mass_err must be 0", i);
            assert_eq!(outcome.global_energy_error, 0, "tick {} energy_err must be 0", i);
            // Cross-check against field-core's exact reference global_mass_error / global_energy_error!
            assert_eq!(mr.planet.global_mass_error(), 0);
            assert_eq!(mr.planet.global_energy_error(), 0);
            eprintln!(
                "WAVE39B_TICK[{}] ({}): cpu_step={:.2}ms clim={:.2}ms dirty={} mass_err=0 energy_err=0",
                i,
                if i % 2 == 1 { "roi128" } else { "refsim" },
                outcome.cpu_step_ms,
                outcome.climate_infer_ms,
                outcome.dirty_subbricks.len()
            );
            if i >= 2 {
                if i % 2 == 1 {
                    roi_ms.push(outcome.cpu_step_ms);
                } else {
                    refsim_ms.push(outcome.cpu_step_ms);
                }
            }
        }
        let avg_refsim: f32 = refsim_ms.iter().sum::<f32>() / refsim_ms.len().max(1) as f32;
        let avg_roi: f32 = roi_ms.iter().sum::<f32>() / roi_ms.len().max(1) as f32;
        eprintln!(
            "WAVE39B_SUMMARY: avg_refsim_cpu_step={:.2}ms avg_roi128_cpu_step={:.2}ms",
            avg_refsim, avg_roi
        );
        #[cfg(not(debug_assertions))]
        {
            assert!(
                avg_refsim < 25.0,
                "release refsim cpu_step must be < 25ms, got {:.2}ms",
                avg_refsim
            );
            assert!(
                avg_roi < 25.0,
                "release roi128 cpu_step must be < 25ms, got {:.2}ms",
                avg_roi
            );
        }
    }
}
