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
    OUTPUT_CHANNELS, RawStepOutput, SP_ICE, SP_PLANT, SP_ROCK, SP_SOIL, SP_WOOD, VolumeField,
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

/// Summary returned by one 10 Hz `MacroPlanetRuntime::step_macro_tick` call.
pub struct MacroPlanetTickOutcome {
    pub tick: u64,
    pub infer_ms: f32,
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
    pub newly_fracturing_cores: Vec<(u8, u8, String, f32, f32, f32)>,
    pub newly_impacted_cores: Vec<(u8, u8, String, f32, f32, f32)>,
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

    // Per-shard 16×16×16 cubic sub-brick raw cache for temporal delta encoding
    pub subbrick_raw_cache: HashMap<ShardSubbrickKey, (u8, RawBrickArc)>,
    pub full_sync_cursor: usize,

    // Live backend & conservation telemetry
    pub last_infer_ms: f32,
    pub last_active_frac: f32,
    pub last_backend: String,
    pub neural_3d_vertical: bool,
    pub seam_crossings_total: u64,
    pub climate_injections_total: u64,
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
            full_sync_cursor: 0,
            last_infer_ms: 0.0,
            last_active_frac: 1.0,
            last_backend: "refsim-macro-planet".to_string(),
            neural_3d_vertical: false,
            seam_crossings_total: 0,
            climate_injections_total: 0,
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

    /// Run one full 10 Hz macro-planet physics tick:
    /// 1. Advance sweeping solar terminator `sun_dir`.
    /// 2. Apply `op=13` `PlanetaryClimateNet` (or CPU `GlobalClimateShell` fallback) when due.
    /// 3. Step all active hydrated `256×256×64` shards (with optional `op=11` neural output on hero shard,
    ///    otherwise parallel RefSim + exact Projection Firewall).
    /// 4. `exchange_hydrated_shard_borders()` across adjacent hydrated shards (exact `0 / 0` conservation!).
    /// 5. `apply_climate_to_shard()` AMR downscale onto all hydrated shards.
    /// 6. Parallel `step_structural_mechanics()` across all hydrated shards (Sky-Island collapses!).
    /// 7. Refresh dirty `16×16×16` cubic sub-bricks and guarantee **`global_mass_error == 0 && global_energy_error == 0`**!
    #[allow(clippy::type_complexity)]
    pub fn step_macro_tick(
        &mut self,
        op13_res: Option<Result<PlanetClimateOp13Response, String>>,
        applied_injections: &[ClimateInjection],
        hero_neural_op11: Option<(Vec<RawStepOutput>, Option<Vec<f32>>, f32, f32)>,
    ) -> MacroPlanetTickOutcome {
        let t0 = Instant::now();
        let _sun_dir = self.advance_sun_dir();

        if op13_res.is_some() || !applied_injections.is_empty() || self.should_step_climate_this_tick() {
            self.apply_climate_step(op13_res, applied_injections);
        }

        let mut dirty_set: HashSet<ShardSubbrickKey> = HashSet::new();
        let mut used_neural_3d = false;
        let mut reported_infer_ms = 0.0f32;
        let mut active_frac = 1.0f32;

        let mut keys: Vec<PlanetShardKey> = self.planet.shards.keys().copied().collect();
        keys.sort();

        // 1. Step hydrated shards (optional op=11 on hero shard + parallel RefSim on any others)
        let hero_key = self.hero_site.shard_key;
        let mut hero_op11 = hero_neural_op11;
        if let Some((ref horiz, ref zf, ms, afrac)) = hero_op11 {
            if let Some(vf) = self.planet.shards.get_mut(&hero_key) {
                if horiz.len() == vf.depth
                    && horiz
                        .first()
                        .map(|h| h.width == vf.width && h.height == vf.height && h.data.len() == OUTPUT_CHANNELS * vf.width * vf.height)
                        .unwrap_or(false)
                {
                    let st = vf.step_3d_with_neural_z(horiz, zf.as_deref());
                    for (layer, cx, cy) in st.dirty_chunks {
                        let bz = (layer as usize / CHUNK_SIZE) as u16;
                        dirty_set.insert((hero_key, cx, cy, bz));
                    }
                    used_neural_3d = true;
                    reported_infer_ms = ms;
                    active_frac = afrac.clamp(0.0, 1.0);
                }
            }
            hero_op11 = None;
        }
        let _ = hero_op11;

        let remaining_keys: Vec<PlanetShardKey> = keys
            .iter()
            .copied()
            .filter(|&k| !(used_neural_3d && k == hero_key))
            .collect();

        if remaining_keys.len() == 1 {
            let k = remaining_keys[0];
            if let Some(vf) = self.planet.shards.get_mut(&k) {
                let raw = vf.refsim_raw_outputs_all();
                let st = vf.step_3d(&raw);
                for (layer, cx, cy) in st.dirty_chunks {
                    let bz = (layer as usize / CHUNK_SIZE) as u16;
                    dirty_set.insert((k, cx, cy, bz));
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
                        let raw = vf.refsim_raw_outputs_all();
                        let st = vf.step_3d(&raw);
                        (kk, st.dirty_chunks)
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
                    dirty_set.insert((k, cx, cy, bz));
                }
            }
        }

        // 2. Exact antisymmetric cross-shard border conservation exchange
        let edge_stats = self.planet.exchange_hydrated_shard_borders();

        // 3. Macro -> Micro AMR climate downscale onto all hydrated shards
        for &k in &keys {
            self.planet.apply_climate_to_shard(k);
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
                                dirty_set.insert((sk, bx, by, bz));
                                if bx > 0 {
                                    dirty_set.insert((sk, bx - 1, by, bz));
                                }
                                if bx + 1 < chunks_xy {
                                    dirty_set.insert((sk, bx + 1, by, bz));
                                }
                                if by > 0 {
                                    dirty_set.insert((sk, bx, by - 1, bz));
                                }
                                if by + 1 < chunks_xy {
                                    dirty_set.insert((sk, bx, by + 1, bz));
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

        // Rotate a small ring of surface sub-bricks per hydrated shard so climate shifts stream smoothly
        let total_2d = (chunks_xy as usize) * (chunks_xy as usize);
        let surf_bz = (self.planet.config.z_sea / CHUNK_SIZE).min(bz_dim as usize - 1) as u16;
        if total_2d > 0 {
            for (i, &sk) in keys.iter().enumerate() {
                let ci = (self.full_sync_cursor + i) % total_2d;
                let bx = (ci % chunks_xy as usize) as u16;
                let by = (ci / chunks_xy as usize) as u16;
                dirty_set.insert((sk, bx, by, surf_bz));
            }
            self.full_sync_cursor = (self.full_sync_cursor + 1) % total_2d;
        }

        self.planet.tick += 1;

        let dirty_subbricks: Vec<ShardSubbrickKey> = dirty_set.into_iter().collect();
        for &(sk, bx, by, bz) in &dirty_subbricks {
            let _ = self.refresh_shard_subbrick(sk, bx, by, bz);
        }

        let total_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let infer_ms = if used_neural_3d {
            reported_infer_ms
        } else {
            total_ms
        };
        let backend = match (used_neural_3d, self.climate_neural_active) {
            (true, true) => "cuda-fp16-macro-climate+v8".to_string(),
            (false, true) => "cuda-fp16-climate+refsim-1m".to_string(),
            (true, false) => "cuda-fp16-v8+cpu-climate".to_string(),
            (false, false) => "refsim-macro-planet".to_string(),
        };
        self.last_infer_ms = infer_ms;
        self.last_active_frac = active_frac;
        self.last_backend = backend.clone();
        self.neural_3d_vertical = used_neural_3d;

        MacroPlanetTickOutcome {
            tick: self.planet.tick,
            infer_ms,
            climate_infer_ms: self.climate_infer_ms,
            active_frac,
            backend,
            neural_3d_vertical: used_neural_3d,
            global_mass_error: self.planet.global_mass_error(),
            global_energy_error: self.planet.global_energy_error(),
            seam_mass_crossed: edge_stats.seam_mass_crossed,
            fallen_solid_mass,
            splashed_fluid_mass,
            active_hydrated_shards: self.planet.shards.len().min(u16::MAX as usize) as u16,
            dirty_subbricks,
            newly_fracturing_cores,
            newly_impacted_cores,
        }
    }

    // =========================================================================
    // 4. Local Climate Sampling, Surface Queries, Shrines, Spawns & Commands
    // =========================================================================

    /// Sample live `(wind_u, wind_v, temp_c, cloud_cover)` at virtual coordinate `(face, x, y)`
    /// from the `6 × 64²` `PlanetaryClimateNet` / `GlobalClimateShell` buffers.
    pub fn sample_local_climate(&self, face: u8, x: f32, y: f32) -> (f32, f32, f32, f32) {
        let f = (face as usize) % FACE_COUNT;
        let n_virt = self.face_size_virtual().max(1) as f32;
        let iu = (((x / n_virt) * CLIMATE_GRID_N as f32).floor() as isize)
            .clamp(0, CLIMATE_GRID_N as isize - 1) as usize;
        let iv = (((y / n_virt) * CLIMATE_GRID_N as f32).floor() as isize)
            .clamp(0, CLIMATE_GRID_N as isize - 1) as usize;
        let pix = iv * CLIMATE_GRID_N + iu;

        let rgba = &self.climate_rgba_faces[f];
        let o4 = pix * 4;
        let temp_k = 240.0 + (rgba[o4] as f32 / 255.0) * 80.0;
        let temp_c = temp_k - 273.15;
        let cloud_cover = rgba[o4 + 1] as f32 / 255.0;

        let wuv = &self.climate_wind_uv_faces[f];
        let o2 = pix * 2;
        let wind_u = wuv[o2];
        let wind_v = wuv[o2 + 1];

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
// 6. Climate Pack <-> GlobalClimateShell Helpers
// =============================================================================

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
}
