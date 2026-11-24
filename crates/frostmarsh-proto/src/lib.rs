//! Compact binary QUIC protocol for Frostmarsh Unified v2 (zero WAL, zero DB overhead).
//!
//! - Dynamic world size (`128×128`, `256×256`, `512×512`) negotiated in `Welcome` & `TelemetrySnap`.
//! - Unreliable QUIC datagrams carry LZ4-compressed dirty 16×16 `A+` chunks (AOI-filtered +
//!   staggered macro minimap sync) and 30 Hz rich entity/shrine/combat snapshots.
//! - Reliable QUIC streams carry low-frequency control messages (`Join`, `SelectClass`,
//!   `ToggleSimMode`, `ToggleEcosystem`, `CycleWeather`, `ResetWorld`, `RequestFullSync`).

use serde::{Deserialize, Serialize};

pub const GRID_W: usize = 256;
pub const GRID_H: usize = 256;
pub const GRID_D: usize = 4; // 4 physically coupled 3D vertical strata (L0..L3 = 262,144 voxels!)
pub const CHUNK_SIZE: usize = 16;
pub const CHUNKS_X: usize = GRID_W / CHUNK_SIZE;
pub const CHUNKS_Y: usize = GRID_H / CHUNK_SIZE;

pub const CLASS_WARDEN: u8 = 0; // Frost Warden / Strider (Heavy Walker, Squad Anchor)
pub const CLASS_ALCHEMIST: u8 = 1; // Pyro Alchemist / Skywing (3D Flyer & Thermal Glider)
pub const CLASS_RANGER: u8 = 2; // Marsh Ranger / Tidekin (Amphibious Swimmer & Diver)
pub const CLASS_STRIDER: u8 = 0;
pub const CLASS_SKYWING: u8 = 1;
pub const CLASS_TIDEKIN: u8 = 2;
pub const CLASS_EMBERSTALKER: u8 = 3; // Solo Cavern Climber & Raider
pub const CLASS_BURROWER: u8 = 4; // Terramole: Conservative Subterranean Tunnel-Carver

pub const STATUS_BURNING: u8 = 1 << 0;
pub const STATUS_CHILLED: u8 = 1 << 1;
pub const STATUS_WET: u8 = 1 << 2;
pub const STATUS_OILED: u8 = 1 << 3;
pub const STATUS_SHIELDED: u8 = 1 << 4;
pub const STATUS_HASTE: u8 = 1 << 5;

pub const WEATHER_CLEAR: u8 = 0;
pub const WEATHER_BLIZZARD: u8 = 1;
pub const WEATHER_SQUALL: u8 = 2;
pub const WEATHER_EMBER: u8 = 3;

pub const MATCH_PHASE_ACTIVE: u8 = 0;
pub const MATCH_PHASE_WEST_VICTORY: u8 = 1;
pub const MATCH_PHASE_EAST_VICTORY: u8 = 2;
pub const MATCH_PHASE_DRAW: u8 = 3;

pub const fn world_law_preset_name(preset_id: u8) -> &'static str {
    match preset_id % 4 {
        1 => "FREEZE-OUT",
        2 => "RISING FLOOD",
        3 => "LOW-G MAGMA",
        _ => "STANDARD",
    }
}

pub type ClientControl = ClientReliable;

#[inline]
pub fn chunks_for_dim(dim: usize) -> usize {
    dim.div_ceil(CHUNK_SIZE)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClientReliable {
    Join { name: String, team: u8 },
    SelectClass { class_id: u8 },
    ToggleSimMode,
    ToggleEcosystem,
    ToggleNeuralBots,
    CycleWeather,
    ResetWorld,
    RequestFullSync,
    ChatCommand { text: String, volume: f32 },
    TogglePhysicsNative,
    CycleTimeScale,
    PhysicalActuator {
        dx: f32,
        dy: f32,
        thermal_flux: f32,
        sp_water: f32,
        sp_oil: f32,
        sp_smoke: f32,
        focus_radius: f32,
        impulse: f32,
    },
    /// Place or clear a spatial `WorldModelDT` time-dilation dome (`mode`: 0=clear, 1=0.15x Stasis, 2=3.0x Haste).
    PlaceTimeDome { gx: f32, gy: f32, mode: u8 },
    /// Cycle the authoritative 3D world through the 5 true-3D biomes (`0..=4`) and trigger full brick re-sync.
    CycleBiome,
    /// Cycle the evolving 3D world speed (`1x -> 4x -> 12x -> 1x`).
    CycleEvolutionSpeed,
    /// Cycle the live 6-float World-Law preset (`0=STANDARD -> 1=FREEZE-OUT -> 2=RISING FLOOD -> 3=LOW-G MAGMA -> 0`).
    CycleWorldLaw,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientInputDatagram {
    pub seq: u32,
    pub move_x: f32,
    pub move_y: f32,
    pub aim_x: f32,
    pub aim_y: f32,
    pub dash: bool,
    pub melee: bool,
    pub cast_bolt: bool,
    pub bolt_element: u8,
    pub carve_ice: bool,
    pub cast_ability_e: bool,
    pub cast_ultimate_r: bool,
    pub client_time_ms: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerSnap {
    pub id: u32,
    pub name: String,
    pub team: u8,
    pub class_id: u8,
    pub is_bot: bool,
    pub x: f32,
    pub y: f32,
    pub z: f32, // 3D altitude in layer units (0.0=riverbed, 1.0=ground, 2.0=highland/skybridge)
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub aim_x: f32,
    pub aim_y: f32,
    pub hp: i16,
    pub max_hp: i16,
    pub mana: u16,
    pub max_mana: u16,
    pub stamina: u16,
    pub max_stamina: u16,
    pub level: u8,
    pub xp: u16,
    pub kills: u16,
    pub deaths: u16,
    pub status_flags: u8,
    pub dashing: bool,
    pub melee_active: bool,
    pub ult_active: bool,
    pub carrying_relic: bool,
    pub dash_cd_frac: f32,
    pub ability_e_cd_frac: f32,
    pub ult_r_cd_frac: f32,
    pub ack_input_seq: u32,
    #[serde(default)]
    pub score: u32,
    /// Cubed-sphere face index (`0..6`) when `telemetry.planet_mode == true` (`0` on flat worlds).
    #[serde(default)]
    pub face: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectileSnap {
    pub id: u32,
    pub element: u8, // 0=Fire, 1=Frost, 2=Oil, 3=Meteor/Nova
    pub x: f32,
    pub y: f32,
    pub z: f32, // 3D altitude in layer units (0.0=riverbed, 1.0=ground, 2.0=highland/skybridge)
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub radius: f32,
    #[serde(default)]
    pub face: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellBodySnap {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub z: f32, // 3D altitude in layer units (0.0=riverbed, 1.0=ground, 2.0=highland/skybridge)
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub radius: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShrineSnap {
    pub id: u8,
    pub name: String,
    pub x: f32,
    pub y: f32,
    /// -1.0 = Full West (Frost), 0.0 = Neutral, +1.0 = Full East (Pyro)
    pub progress: f32,
    /// -1 = West, 0 = Neutral, 1 = East
    pub owner: i8,
    #[serde(default)]
    pub face: u8,
}

/// Recent combat / world event for floating damage popups, elemental combo callouts, & kill feed.
/// `kind`: 0=damage, 1=kill, 2=combo, 3=shrine, 4=relic, 5=weather
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CombatEvent {
    pub id: u32,
    pub kind: u8,
    pub x: f32,
    pub y: f32,
    pub value: i16,
    pub team: u8,
    pub label: String,
}

/// Live neural bot introspection snapshot (`op=5` zero-cost 4×4 `Z_world` saliency + policy intents).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BotBrainSnap {
    pub bot_id: u32,
    pub class_id: u8,
    pub team: u8,
    pub target_id: u32,
    pub elem_probs: [u8; 3],
    pub ability_probs: [u8; 6],
    pub z_heatmap_4x4: [u8; 16],
}

/// Recent kill feed entry (`elem`: 0=Fire, 1=Frost, 2=Oil, 3=Melee/Physical).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KillFeedEntry {
    pub killer_name: String,
    pub victim_name: String,
    pub killer_team: u8,
    pub elem: u8,
    pub tick: u32,
}

/// Live 3D acoustic speech bubble (emergent bot callouts + human chat commands).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SpeechBubbleSnap {
    pub speaker_id: u32,
    pub speaker_name: String,
    pub team: u8,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub snr_db: f32,
    pub symbol_id: u8,
    pub text: String,
    pub tick: u32,
    #[serde(default)]
    pub face: u8,
}

/// Short-lived 3D volumetric physical actuator pulse (heat beam, cryo field, matter stream, or kinetic shockwave).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ActuatorPulseSnap {
    pub x0: f32,
    pub y0: f32,
    pub z0: f32,
    pub x1: f32,
    pub y1: f32,
    pub z1: f32,
    pub thermal_flux: f32, // +heat, -cryo
    pub sp_oil: f32,
    pub sp_water: f32,
    pub sp_smoke: f32,
    pub focus_radius: f32,
    pub impulse: f32,
    pub team: u8,
    #[serde(default)]
    pub face: u8,
}

/// Active spatial `WorldModelDT` time-dilation dome (`0.15x` Stasis Dome or `3.0x` Haste Zone).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeDomeSnap {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub dt_scale: f32,
    pub ttl_s: f32,
    pub owner_team: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TelemetrySnap {
    pub mode: String,
    pub backend: String,
    pub ecosystem_on: bool,
    pub grid_w: u16,
    pub grid_h: u16,
    pub grid_d: u8, // Number of 3D vertical strata (e.g. 4 => 256×256×4 = 262,144 voxels)
    pub field_tick: u64,
    pub infer_ms_last: f32,
    pub infer_ms_p99: f32,
    pub eco_infer_ms: f32,
    pub proj_active_pct: f32,
    pub mass_error: i64,
    pub energy_error: i64,
    pub wire_kbps: f32,
    pub aoi_chunks_sent: u16,
    pub total_chunks: u16,
    pub active_bots: u16,
    pub neural_bots_on: bool,
    pub bot_infer_ms: f32,
    pub weather_kind: u8,
    pub day_time: f32,
    pub wind_x: f32,
    pub wind_y: f32,
    pub score_west: u16,
    pub score_east: u16,
    #[serde(default)]
    pub acoustic_ms: f32,
    #[serde(default)]
    pub world_sparse_pct: f32,
    #[serde(default)]
    pub physics_native: bool,
    #[serde(default)]
    pub time_scale: f32,
    #[serde(default)]
    pub world_day: u32,
    #[serde(default)]
    pub neural_3d_vertical: bool,
    #[serde(default)]
    pub active_time_domes: u8,
    #[serde(default)]
    pub system2_slm_active: bool,
    /// Wave 31: Live 6-Face Cubed-Sphere Planetary Mode (`true` when server launched with `--planet`).
    #[serde(default)]
    pub planet_mode: bool,
    /// Inner core radius `r_core` in voxel units when `planet_mode == true` (e.g. `48.0`).
    #[serde(default)]
    pub planet_r_core: f32,
    /// Normalized 3D world-space direction toward the sweeping planetary Sun (Day/Night Terminator).
    #[serde(default)]
    pub sun_dir: [f32; 3],
    /// Cumulative exact conserved integer mass exchanged across all 12 cubed-sphere seams.
    #[serde(default)]
    pub seam_mass_crossed: u64,
    /// Number of intact levitating Orbital Sky-Ring `BuoyantCore`s (out of 8).
    #[serde(default)]
    pub active_sky_cores: u8,
    /// Wave 34: Dual-Scale Locally-Flat Macro Planet Mode (`R ≈ 1,304 m`, `6 × 2048² × 64` = `1.61B` virtual voxels).
    #[serde(default)]
    pub macro_planet: bool,
    /// Shards per cube-face edge in macro planet mode (e.g. `8` => `8×8 = 64` shards/face = `384` total shards).
    #[serde(default)]
    pub planet_shards_per_axis: u8,
    /// Procedural world seed shared by server & client for deterministic macro globe terrain alignment.
    #[serde(default)]
    pub planet_seed: u64,
    /// Live `op=13` `PlanetaryClimateNet` GPU inference latency in ms on NVIDIA GB10.
    #[serde(default)]
    pub climate_infer_ms: f32,
    /// Number of currently hydrated `256×256×64` `1 m` shards on the server.
    #[serde(default)]
    pub active_hydrated_shards: u16,
    /// Live Coriolis macro wind vector `(u, v)` at the local player's position.
    #[serde(default)]
    pub local_wind_u: f32,
    #[serde(default)]
    pub local_wind_v: f32,
    /// Live `PlanetaryClimateNet` surface temperature (°C) & cloud cover (`0..1`) at the local player.
    #[serde(default)]
    pub local_temp_c: f32,
    #[serde(default)]
    pub local_cloud_cover: f32,
}

/// Additive `16×16×D` 3D volumetric column brick payload (`density_mat` + `albedo_thermal` RGBA8, LZ4-compressed).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Brick3DPayload {
    pub brick_id: u32,
    pub bx: u16,
    pub by: u16,
    pub depth: u8,
    pub density_mat_lz4: Vec<u8>,
    pub albedo_thermal_lz4: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ServerDatagram {
    /// 30 Hz authoritative combat/entity/shrine/event snapshot (unreliable — always take latest!)
    EntitySnapshot {
        sim_tick: u64,
        match_time_secs: u32,
        your_player_id: u32,
        echo_client_time_ms: u32,
        relic_x: f32,
        relic_y: f32,
        players: Vec<PlayerSnap>,
        projectiles: Vec<ProjectileSnap>,
        bodies: Vec<CellBodySnap>,
        shrines: Vec<ShrineSnap>,
        events: Vec<CombatEvent>,
        telemetry: TelemetrySnap,
        #[serde(default)]
        bot_brains: Vec<BotBrainSnap>,
        #[serde(default)]
        kill_feed: Vec<KillFeedEntry>,
        #[serde(default)]
        speech_bubbles: Vec<SpeechBubbleSnap>,
        #[serde(default)]
        actuator_pulses: Vec<ActuatorPulseSnap>,
        #[serde(default)]
        match_phase: u8,
        #[serde(default)]
        match_time_left_s: u16,
        #[serde(default)]
        mvp_name: String,
        #[serde(default)]
        mvp_score: u32,
        #[serde(default)]
        time_domes: Vec<TimeDomeSnap>,
        #[serde(default)]
        world_law_preset: u8,
    },
    /// 10 Hz single dirty 16×16×1 `A+` volumetric chunk at vertical layer `layer` (`0..grid_d`).
    APlusChunk {
        field_tick: u64,
        layer: u8,
        chunk_id: u16,
        cx: u8,
        cy: u8,
        lz4_cells: Vec<u8>,
    },
    /// 10 Hz LZ4-compressed `16×16×D` 3D volumetric brick (`density_mat` + `albedo_thermal` RGBA8).
    VolumetricBrick3D(Brick3DPayload),
    /// 10 Hz sparse temporal delta for a `16×16×D` 3D volumetric brick (`(u16 idx, dm[4], at[4])` × N, LZ4).
    VolumetricBrick3DDelta {
        bx: u16,
        by: u16,
        depth: u16,
        tick: u64,
        lz4_deltas: Vec<u8>,
    },
    /// 10 Hz LZ4-compressed `16×16×sub_depth` cubic 3D sub-brick (`bz = 0..total_depth.div_ceil(16)`),
    /// streamed when `total_depth > 32` (e.g. `Z=64`) while skipping pure empty sky & deep bedrock cubes!
    VolumetricSubBrick3D {
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_density_mat: Vec<u8>,
        lz4_albedo_thermal: Vec<u8>,
    },
    /// 10 Hz sparse temporal delta for a `16×16×sub_depth` cubic 3D sub-brick (`(u16 idx, dm[4], at[4])` × N, LZ4).
    VolumetricSubBrick3DDelta {
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_deltas: Vec<u8>,
    },
    /// Wave 31: 10 Hz LZ4-compressed `16×16×sub_depth` cubic 3D sub-brick for one of the 6 Cubed-Sphere planet faces (`face ∈ 0..6`).
    PlanetFaceBrick3D {
        face: u8,
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_density_mat: Vec<u8>,
        lz4_albedo_thermal: Vec<u8>,
    },
    /// Wave 31: 10 Hz sparse temporal delta for one `16×16×sub_depth` cubic 3D sub-brick on planet `face ∈ 0..6`.
    PlanetFaceBrick3DDelta {
        face: u8,
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_deltas: Vec<u8>,
    },
    /// Wave 34: Live `PlanetaryClimateNet` (`op=13`) `64×64` global weather/cloud/ash/snow overlay for one cube face.
    /// RGBA8 per cell: `R = temp_norm (240..320K)`, `G = cloud_storm`, `B = ash_plume`, `A = ice_snow`.
    PlanetMacroClimateFace {
        face: u8,
        grid_n: u16,
        tick: u64,
        lz4_climate_rgba: Vec<u8>,
    },
    /// Wave 34: `16×16×sub_depth` cubic `1 m` sub-brick for a hydrated `256×256×64` macro-planet shard `(face, sx, sy)`.
    PlanetShardBrick3D {
        face: u8,
        sx: u8,
        sy: u8,
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_density_mat: Vec<u8>,
        lz4_albedo_thermal: Vec<u8>,
    },
    /// Wave 34: Sparse temporal delta for a hydrated `256×256×64` macro-planet shard `(face, sx, sy)` sub-brick.
    PlanetShardBrick3DDelta {
        face: u8,
        sx: u8,
        sy: u8,
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_deltas: Vec<u8>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ServerReliable {
    Welcome {
        your_player_id: u32,
        width: u16,
        height: u16,
        depth: u8,
    },
}

pub fn compress_aplus_cells(raw_1024: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(raw_1024)
}

pub fn decompress_aplus_cells(compressed: &[u8]) -> Option<Vec<u8>> {
    lz4_flex::decompress_size_prepended(compressed).ok()
}

/// Compress raw Z-major `[depth][16][16][4]` `density_mat` and `albedo_thermal` brick buffers with LZ4.
pub fn encode_brick3d_lz4(
    brick_id: u32,
    bx: u16,
    by: u16,
    depth: u8,
    density_mat_raw: &[u8],
    albedo_thermal_raw: &[u8],
) -> Brick3DPayload {
    Brick3DPayload {
        brick_id,
        bx,
        by,
        depth,
        density_mat_lz4: lz4_flex::compress_prepend_size(density_mat_raw),
        albedo_thermal_lz4: lz4_flex::compress_prepend_size(albedo_thermal_raw),
    }
}

/// Decompress a `Brick3DPayload` back into raw `(density_mat, albedo_thermal)` RGBA8 byte buffers.
pub fn decode_brick3d_lz4(payload: &Brick3DPayload) -> Option<(Vec<u8>, Vec<u8>)> {
    let density_mat = lz4_flex::decompress_size_prepended(&payload.density_mat_lz4).ok()?;
    let albedo_thermal = lz4_flex::decompress_size_prepended(&payload.albedo_thermal_lz4).ok()?;
    Some((density_mat, albedo_thermal))
}

/// Compress raw Z-major `[sub_depth][16][16][4]` cubic sub-brick buffers with LZ4.
pub fn encode_subbrick3d_lz4(density_mat_raw: &[u8], albedo_thermal_raw: &[u8]) -> (Vec<u8>, Vec<u8>) {
    (
        lz4_flex::compress_prepend_size(density_mat_raw),
        lz4_flex::compress_prepend_size(albedo_thermal_raw),
    )
}

/// Decompress `VolumetricSubBrick3D` LZ4 buffers back into raw `(density_mat, albedo_thermal)` RGBA8 bytes.
pub fn decode_subbrick3d_lz4(
    lz4_density_mat: &[u8],
    lz4_albedo_thermal: &[u8],
) -> Option<(Vec<u8>, Vec<u8>)> {
    let density_mat = lz4_flex::decompress_size_prepended(lz4_density_mat).ok()?;
    let albedo_thermal = lz4_flex::decompress_size_prepended(lz4_albedo_thermal).ok()?;
    Some((density_mat, albedo_thermal))
}

/// Returns `true` if a `16×16×sub_depth` cubic sub-brick is **pure empty sky**:
/// every voxel has `solid_density == 0`, `fluid_density == 0`, `gas_density == 0`, and `thermal_glow == 0`.
pub fn is_subbrick_empty_sky(density_rgba: &[u8], albedo_rgba: &[u8]) -> bool {
    if density_rgba.is_empty() || density_rgba.len() != albedo_rgba.len() || density_rgba.len() % 4 != 0 {
        return false;
    }
    for (dm, at) in density_rgba.chunks_exact(4).zip(albedo_rgba.chunks_exact(4)) {
        if dm[0] != 0 || dm[1] != 0 || dm[2] != 0 || at[3] != 0 {
            return false;
        }
    }
    true
}

/// Returns `true` if a `16×16×sub_depth` cubic sub-brick is **pure uniform opaque solid interior**:
/// every voxel has `solid_density >= 245`, `fluid_density == 0`, `gas_density == 0`, and
/// uniform material & thermal spread (`<= 4` across roughness, RGB albedo, and thermal glow).
pub fn is_subbrick_uniform_solid(density_rgba: &[u8], albedo_rgba: &[u8]) -> bool {
    if density_rgba.is_empty() || density_rgba.len() != albedo_rgba.len() || density_rgba.len() % 4 != 0 {
        return false;
    }
    let mut min_rough = u8::MAX;
    let mut max_rough = 0u8;
    let mut min_r = u8::MAX;
    let mut max_r = 0u8;
    let mut min_g = u8::MAX;
    let mut max_g = 0u8;
    let mut min_b = u8::MAX;
    let mut max_b = 0u8;
    let mut min_t = u8::MAX;
    let mut max_t = 0u8;

    for (dm, at) in density_rgba.chunks_exact(4).zip(albedo_rgba.chunks_exact(4)) {
        if dm[0] < 245 || dm[1] != 0 || dm[2] != 0 {
            return false;
        }
        min_rough = min_rough.min(dm[3]);
        max_rough = max_rough.max(dm[3]);
        min_r = min_r.min(at[0]);
        max_r = max_r.max(at[0]);
        min_g = min_g.min(at[1]);
        max_g = max_g.max(at[1]);
        min_b = min_b.min(at[2]);
        max_b = max_b.max(at[2]);
        min_t = min_t.min(at[3]);
        max_t = max_t.max(at[3]);
        if max_rough - min_rough > 4
            || max_r - min_r > 4
            || max_g - min_g > 4
            || max_b - min_b > 4
            || max_t - min_t > 4
        {
            return false;
        }
    }
    true
}

/// Returns `true` if a `16×16×sub_depth` cubic sub-brick is either **pure empty sky** OR
/// **pure uniform opaque solid interior**, allowing server & client to completely skip sky & deep bedrock cubes.
pub fn is_subbrick_skippable(density_rgba: &[u8], albedo_rgba: &[u8]) -> bool {
    is_subbrick_empty_sky(density_rgba, albedo_rgba)
        || is_subbrick_uniform_solid(density_rgba, albedo_rgba)
}

/// Default per-channel visual deadzone for temporal 3D brick delta encoding (suppresses sub-4K jitter).
pub const DEFAULT_BRICK3D_DELTA_DEADZONE: u8 = 4;

/// Returns `true` when `encode_brick3d_delta` returned the full-brick fallback signal (`Some(vec![])`),
/// indicating `>= 35%` of voxels changed (or buffer geometry changed) and the caller should send
/// a full `ServerDatagram::VolumetricBrick3D`.
#[inline]
pub fn is_brick3d_delta_full_signal(encoded: &[u8]) -> bool {
    encoded.is_empty()
}

/// Compute a sparse LZ4-compressed temporal delta between two Z-major `16×16×D` RGBA8 bricks.
///
/// - Scans each voxel `idx` in `0..(cur_density.len() / 4)`.
/// - Skips any voxel whose maximum per-channel absolute difference across all 8 bytes
///   (`density[4]` + `albedo[4]`) is `<= deadzone`.
/// - Returns `None` when `0` voxels exceeded `deadzone` (caller sends **0 bytes**!).
/// - Returns `Some(lz4_deltas)` when `0 < changed < 35%` of total voxels, packing each changed
///   voxel as 10 bytes `(idx as u16 LE, density[4], albedo[4])` and LZ4-compressing.
/// - Returns `Some(Vec::new())` (full-brick signal) when `changed >= 35%` of total voxels (or
///   lengths mismatch / exceed `u16::MAX` voxels), signaling the caller to send `VolumetricBrick3D`.
pub fn encode_brick3d_delta(
    prev_density: &[u8],
    prev_albedo: &[u8],
    cur_density: &[u8],
    cur_albedo: &[u8],
    deadzone: u8,
) -> Option<Vec<u8>> {
    let byte_len = cur_density.len();
    if byte_len == 0
        || byte_len % 4 != 0
        || prev_density.len() != byte_len
        || prev_albedo.len() != byte_len
        || cur_albedo.len() != byte_len
    {
        return Some(Vec::new());
    }
    let total_voxels = byte_len / 4;
    if total_voxels > u16::MAX as usize {
        return Some(Vec::new());
    }
    // 35% threshold: if changed * 100 >= total_voxels * 35 -> full brick fallback
    let max_delta_voxels = (total_voxels * 35).saturating_sub(1) / 100;
    let mut packed: Vec<u8> = Vec::new();
    let mut changed = 0usize;

    for idx in 0..total_voxels {
        let o = idx * 4;
        let max_diff = cur_density[o]
            .abs_diff(prev_density[o])
            .max(cur_density[o + 1].abs_diff(prev_density[o + 1]))
            .max(cur_density[o + 2].abs_diff(prev_density[o + 2]))
            .max(cur_density[o + 3].abs_diff(prev_density[o + 3]))
            .max(cur_albedo[o].abs_diff(prev_albedo[o]))
            .max(cur_albedo[o + 1].abs_diff(prev_albedo[o + 1]))
            .max(cur_albedo[o + 2].abs_diff(prev_albedo[o + 2]))
            .max(cur_albedo[o + 3].abs_diff(prev_albedo[o + 3]));
        if max_diff <= deadzone {
            continue;
        }
        changed += 1;
        if changed > max_delta_voxels {
            return Some(Vec::new());
        }
        if packed.is_empty() {
            packed.reserve(64 * 10);
        }
        let idx_le = (idx as u16).to_le_bytes();
        packed.extend_from_slice(&idx_le);
        packed.extend_from_slice(&cur_density[o..o + 4]);
        packed.extend_from_slice(&cur_albedo[o..o + 4]);
    }

    if changed == 0 {
        return None;
    }

    Some(lz4_flex::compress_prepend_size(&packed))
}

/// Decompress `VolumetricBrick3DDelta::lz4_deltas` into raw `(idx, density[4], albedo[4])` voxel patches.
pub fn decode_brick3d_delta_voxels(
    lz4_deltas: &[u8],
) -> Result<Vec<(u16, [u8; 4], [u8; 4])>, String> {
    if lz4_deltas.is_empty() {
        return Err("empty lz4_deltas payload (full-brick signal)".to_string());
    }
    let raw = lz4_flex::decompress_size_prepended(lz4_deltas)
        .map_err(|e| format!("brick3d delta lz4 decompress failed: {}", e))?;
    if raw.len() % 10 != 0 {
        return Err(format!(
            "brick3d delta payload length {} is not a multiple of 10 bytes",
            raw.len()
        ));
    }
    let count = raw.len() / 10;
    let mut out = Vec::with_capacity(count);
    for rec in raw.chunks_exact(10) {
        let idx = u16::from_le_bytes([rec[0], rec[1]]);
        let dm = [rec[2], rec[3], rec[4], rec[5]];
        let at = [rec[6], rec[7], rec[8], rec[9]];
        out.push((idx, dm, at));
    }
    Ok(out)
}

/// Apply an LZ4-compressed `VolumetricBrick3DDelta` payload in-place onto Z-major `density` & `albedo`
/// brick buffers (`16×16×D×4` bytes each). Returns the exact number of patched voxels.
pub fn apply_brick3d_delta(
    density: &mut [u8],
    albedo: &mut [u8],
    lz4_deltas: &[u8],
) -> Result<usize, String> {
    if density.len() != albedo.len() || density.len() % 4 != 0 {
        return Err(format!(
            "brick3d buffer length mismatch: density={} albedo={}",
            density.len(),
            albedo.len()
        ));
    }
    let total_voxels = density.len() / 4;
    let patches = decode_brick3d_delta_voxels(lz4_deltas)?;
    for &(idx, dm, at) in &patches {
        let v = idx as usize;
        if v >= total_voxels {
            return Err(format!(
                "brick3d delta voxel index {} out of range (total {})",
                v, total_voxels
            ));
        }
        let o = v * 4;
        density[o..o + 4].copy_from_slice(&dm);
        albedo[o..o + 4].copy_from_slice(&at);
    }
    Ok(patches.len())
}

pub fn encode_msg<T: Serialize>(msg: &T) -> Vec<u8> {
    bincode::serialize(msg).expect("bincode serialize")
}

pub fn decode_msg<'a, T: Deserialize<'a>>(bytes: &'a [u8]) -> Option<T> {
    bincode::deserialize(bytes).ok()
}

/// Compress a `ServerDatagram` with LZ4 so rich 12-bot + shrine + event snapshots
/// always fit inside a single ~1,200 B QUIC MTU datagram.
pub fn encode_server_datagram(msg: &ServerDatagram) -> Vec<u8> {
    let raw = bincode::serialize(msg).expect("bincode serialize datagram");
    lz4_flex::compress_prepend_size(&raw)
}

pub fn decode_server_datagram(bytes: &[u8]) -> Option<ServerDatagram> {
    let raw = lz4_flex::decompress_size_prepended(bytes).ok()?;
    bincode::deserialize(&raw).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brick3d_lz4_and_datagram_roundtrip() {
        for &depth in &[16u8, 32u8] {
            let len = 16 * 16 * (depth as usize) * 4;
            let mut dm = vec![0u8; len];
            let mut at = vec![0u8; len];
            dm[0..4].copy_from_slice(&[255, 0, 0, 200]);
            dm[len - 4..len].copy_from_slice(&[0, 128, 64, 30]);
            at[0..4].copy_from_slice(&[90, 95, 105, 0]);
            at[len - 4..len].copy_from_slice(&[255, 140, 20, 240]);

            let payload = encode_brick3d_lz4(42, 3, 5, depth, &dm, &at);
            assert!(payload.density_mat_lz4.len() < len);
            assert!(payload.albedo_thermal_lz4.len() < len);

            let wire = encode_server_datagram(&ServerDatagram::VolumetricBrick3D(payload));
            let decoded = decode_server_datagram(&wire).expect("decode ServerDatagram::VolumetricBrick3D");
            match decoded {
                ServerDatagram::VolumetricBrick3D(b) => {
                    assert_eq!((b.brick_id, b.bx, b.by, b.depth), (42, 3, 5, depth));
                    let (dm2, at2) = decode_brick3d_lz4(&b).expect("decompress brick3d");
                    assert_eq!(dm2, dm);
                    assert_eq!(at2, at);
                }
                other => panic!("unexpected variant: {:?}", other),
            }
        }
        // Verify CycleBiome & CycleEvolutionSpeed reliable control messages roundtrip
        let cb_wire = encode_msg(&ClientReliable::CycleBiome);
        assert!(matches!(decode_msg::<ClientReliable>(&cb_wire), Some(ClientReliable::CycleBiome)));
        let ce_wire = encode_msg(&ClientReliable::CycleEvolutionSpeed);
        assert!(matches!(
            decode_msg::<ClientReliable>(&ce_wire),
            Some(ClientReliable::CycleEvolutionSpeed)
        ));
        let wl_wire = encode_msg(&ClientControl::CycleWorldLaw);
        assert!(matches!(
            decode_msg::<ClientControl>(&wl_wire),
            Some(ClientControl::CycleWorldLaw)
        ));
        assert_eq!(world_law_preset_name(0), "STANDARD");
        assert_eq!(world_law_preset_name(1), "FREEZE-OUT");
        assert_eq!(world_law_preset_name(2), "RISING FLOOD");
        assert_eq!(world_law_preset_name(3), "LOW-G MAGMA");

        // Verify EntitySnapshot with world_law_preset round-trips cleanly over LZ4 datagram
        let snap = ServerDatagram::EntitySnapshot {
            sim_tick: 77,
            match_time_secs: 12,
            your_player_id: 1,
            echo_client_time_ms: 42,
            relic_x: 128.0,
            relic_y: 128.0,
            players: Vec::new(),
            projectiles: Vec::new(),
            bodies: Vec::new(),
            shrines: Vec::new(),
            events: Vec::new(),
            telemetry: TelemetrySnap {
                mode: "refsim".to_string(),
                backend: "cpu".to_string(),
                ecosystem_on: true,
                grid_w: 256,
                grid_h: 256,
                grid_d: 16,
                field_tick: 77,
                infer_ms_last: 1.0,
                infer_ms_p99: 1.2,
                eco_infer_ms: 0.2,
                proj_active_pct: 10.0,
                mass_error: 0,
                energy_error: 0,
                wire_kbps: 50.0,
                aoi_chunks_sent: 4,
                total_chunks: 256,
                active_bots: 0,
                neural_bots_on: false,
                bot_infer_ms: 0.0,
                weather_kind: 0,
                day_time: 0.5,
                wind_x: 0.0,
                wind_y: 0.0,
                score_west: 0,
                score_east: 0,
                acoustic_ms: 0.0,
                world_sparse_pct: 100.0,
                physics_native: true,
                time_scale: 1.0,
                world_day: 1,
                neural_3d_vertical: false,
                active_time_domes: 0,
                system2_slm_active: false,
                planet_mode: false,
                planet_r_core: 0.0,
                sun_dir: [0.0, 0.0, 0.0],
                seam_mass_crossed: 0,
                active_sky_cores: 0,
                macro_planet: false,
                planet_shards_per_axis: 0,
                planet_seed: 0,
                climate_infer_ms: 0.0,
                active_hydrated_shards: 0,
                local_wind_u: 0.0,
                local_wind_v: 0.0,
                local_temp_c: 0.0,
                local_cloud_cover: 0.0,
            },
            bot_brains: Vec::new(),
            kill_feed: Vec::new(),
            speech_bubbles: Vec::new(),
            actuator_pulses: Vec::new(),
            match_phase: 0,
            match_time_left_s: 180,
            mvp_name: String::new(),
            mvp_score: 0,
            time_domes: Vec::new(),
            world_law_preset: 3,
        };
        let snap_wire = encode_server_datagram(&snap);
        match decode_server_datagram(&snap_wire).expect("decode EntitySnapshot") {
            ServerDatagram::EntitySnapshot { world_law_preset, sim_tick, .. } => {
                assert_eq!(sim_tick, 77);
                assert_eq!(world_law_preset, 3);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
        assert_eq!(CLASS_WARDEN, CLASS_STRIDER);
        assert_eq!(CLASS_ALCHEMIST, CLASS_SKYWING);
        assert_eq!(CLASS_RANGER, CLASS_TIDEKIN);
        assert_eq!(CLASS_EMBERSTALKER, 3);
        assert_eq!(CLASS_BURROWER, 4);
    }

    #[test]
    fn test_brick3d_delta_roundtrip_and_compression_ratio() {
        let depth = 16usize;
        let total_voxels = CHUNK_SIZE * CHUNK_SIZE * depth; // 4096 voxels
        let byte_len = total_voxels * 4; // 16384 bytes per channel buffer

        // Populate realistic pseudo-terrain brick so full LZ4 brick isn't trivially all-zeros
        let mut prev_dm = vec![0u8; byte_len];
        let mut prev_at = vec![0u8; byte_len];
        for z in 0..depth {
            for dy in 0..CHUNK_SIZE {
                for dx in 0..CHUNK_SIZE {
                    let idx = (z * CHUNK_SIZE + dy) * CHUNK_SIZE + dx;
                    let o = idx * 4;
                    let h_surf = 5 + ((dx * 7 + dy * 13) % 4);
                    if z <= h_surf {
                        prev_dm[o] = (200 + ((dx * 11 + dy * 5 + z * 3) % 55)) as u8;
                        prev_dm[o + 1] = ((dx * 3 + dy * 7) % 40) as u8;
                        prev_dm[o + 2] = 0;
                        prev_dm[o + 3] = (170 + ((dx + dy * 3) % 60)) as u8;
                        prev_at[o] = (80 + ((dx * 9 + z * 7) % 90)) as u8;
                        prev_at[o + 1] = (95 + ((dy * 11 + z * 5) % 85)) as u8;
                        prev_at[o + 2] = (70 + ((dx * 5 + dy * 9) % 80)) as u8;
                        prev_at[o + 3] = ((dx + dy + z) % 20) as u8;
                    }
                }
            }
        }

        // 1. Sub-deadzone micro-jitter (<= 4) -> MUST return None (0 bytes on wire!)
        let mut jitter_dm = prev_dm.clone();
        let mut jitter_at = prev_at.clone();
        for i in (0..byte_len).step_by(4) {
            jitter_at[i + 3] = jitter_at[i + 3].saturating_add(3);
            jitter_dm[i] = jitter_dm[i].saturating_sub(4);
        }
        assert!(
            encode_brick3d_delta(&prev_dm, &prev_at, &jitter_dm, &jitter_at, 4).is_none(),
            "sub-deadzone jitter must suppress wire traffic completely (None)"
        );

        // 2. Partial brick edit (24 changed voxels = ~0.6% of 4096 voxels, e.g. fire/tunnel/waterfall splash)
        let mut cur_dm = prev_dm.clone();
        let mut cur_at = prev_at.clone();
        let edited_indices: Vec<usize> = (0..24).map(|k| 1024 + k * 7).collect();
        for &idx in &edited_indices {
            let o = idx * 4;
            cur_dm[o] = 10;
            cur_dm[o + 1] = 210;
            cur_dm[o + 2] = 160;
            cur_dm[o + 3] = 45;
            cur_at[o] = 250;
            cur_at[o + 1] = 120;
            cur_at[o + 2] = 30;
            cur_at[o + 3] = 235;
        }

        let full_pkt = encode_server_datagram(&ServerDatagram::VolumetricBrick3D(
            encode_brick3d_lz4(7, 3, 4, depth as u8, &cur_dm, &cur_at),
        ));
        let lz4_deltas = encode_brick3d_delta(&prev_dm, &prev_at, &cur_dm, &cur_at, 4)
            .expect("partial edit must emit delta");
        assert!(!is_brick3d_delta_full_signal(&lz4_deltas));

        let delta_wire = encode_server_datagram(&ServerDatagram::VolumetricBrick3DDelta {
            bx: 3,
            by: 4,
            depth: depth as u16,
            tick: 99,
            lz4_deltas: lz4_deltas.clone(),
        });
        let ratio = full_pkt.len() as f64 / delta_wire.len() as f64;
        assert!(
            ratio > 10.0,
            "partial brick delta must achieve >10x compression over full brick (full={} B, delta={} B, ratio={:.2}x)",
            full_pkt.len(),
            delta_wire.len(),
            ratio
        );

        // Decode ServerDatagram::VolumetricBrick3DDelta and verify exact roundtrip
        let mut recon_dm = prev_dm.clone();
        let mut recon_at = prev_at.clone();
        match decode_server_datagram(&delta_wire).expect("decode VolumetricBrick3DDelta") {
            ServerDatagram::VolumetricBrick3DDelta { bx, by, depth: d, tick, lz4_deltas: d_lz4 } => {
                assert_eq!((bx, by, d, tick), (3, 4, 16, 99));
                let patched = apply_brick3d_delta(&mut recon_dm, &mut recon_at, &d_lz4)
                    .expect("apply_brick3d_delta");
                assert_eq!(patched, edited_indices.len());
            }
            other => panic!("unexpected datagram variant: {:?}", other),
        }
        assert_eq!(recon_dm, cur_dm);
        assert_eq!(recon_at, cur_at);

        // 3. Massive edit (>= 35% of voxels) -> returns Some(vec![]) full-brick signal
        let mut huge_dm = prev_dm.clone();
        for idx in 0..(total_voxels / 2) {
            huge_dm[idx * 4] = huge_dm[idx * 4].wrapping_add(120);
        }
        let huge_res = encode_brick3d_delta(&prev_dm, &prev_at, &huge_dm, &prev_at, 4)
            .expect("massive edit must return Some full-brick signal");
        assert!(
            is_brick3d_delta_full_signal(&huge_res),
            ">=35% change must signal full-brick fallback"
        );
    }

    #[test]
    fn test_subbrick3d_cubic_roundtrip_and_sky_bedrock_skip() {
        let sub_depth = 16u8;
        let total_depth = 64u16;
        let voxels = CHUNK_SIZE * CHUNK_SIZE * (sub_depth as usize);
        let byte_len = voxels * 4;

        // 1. Pure empty sky cube -> skippable!
        let mut sky_dm = vec![0u8; byte_len];
        let mut sky_at = vec![0u8; byte_len];
        for i in 0..voxels {
            sky_dm[i * 4 + 3] = 217; // default air roughness from VolumeField
            sky_at[i * 4] = 92;
            sky_at[i * 4 + 1] = 102;
            sky_at[i * 4 + 2] = 112;
        }
        assert!(is_subbrick_empty_sky(&sky_dm, &sky_at));
        assert!(is_subbrick_skippable(&sky_dm, &sky_at));

        // Adding a cloud (gas) or waterfall (fluid) voxel makes sky non-skippable
        let mut cloud_dm = sky_dm.clone();
        cloud_dm[100 * 4 + 2] = 64;
        assert!(!is_subbrick_skippable(&cloud_dm, &sky_at));
        let mut wf_dm = sky_dm.clone();
        wf_dm[100 * 4 + 1] = 180;
        assert!(!is_subbrick_skippable(&wf_dm, &sky_at));

        // 2. Pure uniform solid bedrock cube -> skippable!
        let mut bed_dm = vec![0u8; byte_len];
        let mut bed_at = vec![0u8; byte_len];
        for i in 0..voxels {
            bed_dm[i * 4] = 250 + (i % 6) as u8; // >= 245
            bed_dm[i * 4 + 3] = 204 + (i % 3) as u8; // spread <= 4
            bed_at[i * 4] = 90 + (i % 4) as u8;
            bed_at[i * 4 + 1] = 94 + (i % 4) as u8;
            bed_at[i * 4 + 2] = 104 + (i % 4) as u8;
            bed_at[i * 4 + 3] = (i % 3) as u8; // thermal spread <= 4
        }
        assert!(is_subbrick_uniform_solid(&bed_dm, &bed_at));
        assert!(is_subbrick_skippable(&bed_dm, &bed_at));

        // Thermal hotspot (> 4 spread) or carved cavern void makes solid cube non-skippable
        let mut hot_bed_at = bed_at.clone();
        hot_bed_at[50 * 4 + 3] = 120;
        assert!(!is_subbrick_skippable(&bed_dm, &hot_bed_at));
        let mut carved_dm = bed_dm.clone();
        carved_dm[50 * 4] = 0;
        assert!(!is_subbrick_skippable(&carved_dm, &bed_at));

        // 3. Full VolumetricSubBrick3D LZ4 + datagram roundtrip (e.g. Floating Sky-Island bz=3)
        let mut island_dm = sky_dm.clone();
        let mut island_at = sky_at.clone();
        for z in 2..8 {
            for dy in 3..13 {
                for dx in 3..13 {
                    let o = ((z * CHUNK_SIZE + dy) * CHUNK_SIZE + dx) * 4;
                    island_dm[o..o + 4].copy_from_slice(&[255, 0, 0, 200]);
                    island_at[o..o + 4].copy_from_slice(&[72, 118, 66, 0]);
                }
            }
        }
        assert!(!is_subbrick_skippable(&island_dm, &island_at));
        let (lz4_dm, lz4_at) = encode_subbrick3d_lz4(&island_dm, &island_at);
        let pkt = ServerDatagram::VolumetricSubBrick3D {
            bx: 7,
            by: 9,
            bz: 3,
            total_depth,
            sub_depth,
            tick: 123,
            lz4_density_mat: lz4_dm,
            lz4_albedo_thermal: lz4_at,
        };
        let wire = encode_server_datagram(&pkt);
        match decode_server_datagram(&wire).expect("decode VolumetricSubBrick3D") {
            ServerDatagram::VolumetricSubBrick3D {
                bx,
                by,
                bz,
                total_depth: td,
                sub_depth: sd,
                tick,
                lz4_density_mat,
                lz4_albedo_thermal,
            } => {
                assert_eq!((bx, by, bz, td, sd, tick), (7, 9, 3, 64, 16, 123));
                let (dm2, at2) =
                    decode_subbrick3d_lz4(&lz4_density_mat, &lz4_albedo_thermal).expect("decode lz4");
                assert_eq!(dm2, island_dm);
                assert_eq!(at2, island_at);
            }
            other => panic!("unexpected variant: {:?}", other),
        }

        // 4. Incremental VolumetricSubBrick3DDelta roundtrip on Z=64 cubic sub-brick
        let mut next_dm = island_dm.clone();
        let mut next_at = island_at.clone();
        for k in 0..16usize {
            let o = (100 + k * 13) * 4;
            next_dm[o..o + 4].copy_from_slice(&[0, 220, 40, 24]);
            next_at[o..o + 4].copy_from_slice(&[40, 150, 220, 0]);
        }
        let lz4_deltas = encode_brick3d_delta(&island_dm, &island_at, &next_dm, &next_at, 4)
            .expect("subbrick delta");
        assert!(!is_brick3d_delta_full_signal(&lz4_deltas));
        let delta_wire = encode_server_datagram(&ServerDatagram::VolumetricSubBrick3DDelta {
            bx: 7,
            by: 9,
            bz: 3,
            total_depth,
            sub_depth,
            tick: 124,
            lz4_deltas,
        });
        let mut recon_dm = island_dm.clone();
        let mut recon_at = island_at.clone();
        match decode_server_datagram(&delta_wire).expect("decode VolumetricSubBrick3DDelta") {
            ServerDatagram::VolumetricSubBrick3DDelta {
                bx,
                by,
                bz,
                total_depth: td,
                sub_depth: sd,
                tick,
                lz4_deltas,
            } => {
                assert_eq!((bx, by, bz, td, sd, tick), (7, 9, 3, 64, 16, 124));
                let patched =
                    apply_brick3d_delta(&mut recon_dm, &mut recon_at, &lz4_deltas).expect("apply");
                assert_eq!(patched, 16);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
        assert_eq!(recon_dm, next_dm);
        assert_eq!(recon_at, next_at);
    }

    #[test]
    fn test_planet_face_brick3d_and_telemetry_roundtrip() {
        let sub_depth = 16u8;
        let total_depth = 32u16;
        let voxels = CHUNK_SIZE * CHUNK_SIZE * (sub_depth as usize);
        let mut dm = vec![0u8; voxels * 4];
        let mut at = vec![0u8; voxels * 4];
        dm[0..4].copy_from_slice(&[255, 0, 0, 210]);
        at[0..4].copy_from_slice(&[72, 130, 82, 0]);
        let (lz4_dm, lz4_at) = encode_subbrick3d_lz4(&dm, &at);

        let full = ServerDatagram::PlanetFaceBrick3D {
            face: 4,
            bx: 2,
            by: 1,
            bz: 1,
            total_depth,
            sub_depth,
            tick: 55,
            lz4_density_mat: lz4_dm,
            lz4_albedo_thermal: lz4_at,
        };
        let wire = encode_server_datagram(&full);
        match decode_server_datagram(&wire).expect("decode PlanetFaceBrick3D") {
            ServerDatagram::PlanetFaceBrick3D {
                face,
                bx,
                by,
                bz,
                total_depth: td,
                sub_depth: sd,
                tick,
                lz4_density_mat,
                lz4_albedo_thermal,
            } => {
                assert_eq!((face, bx, by, bz, td, sd, tick), (4, 2, 1, 1, 32, 16, 55));
                let (dm2, at2) = decode_subbrick3d_lz4(&lz4_density_mat, &lz4_albedo_thermal)
                    .expect("decode planet brick lz4");
                assert_eq!(dm2, dm);
                assert_eq!(at2, at);
            }
            other => panic!("unexpected variant: {:?}", other),
        }

        let mut next_dm = dm.clone();
        let mut next_at = at.clone();
        next_dm[40..44].copy_from_slice(&[0, 220, 0, 20]);
        next_at[40..44].copy_from_slice(&[20, 95, 195, 0]);
        let lz4_deltas = encode_brick3d_delta(&dm, &at, &next_dm, &next_at, 4).expect("delta");
        let delta = ServerDatagram::PlanetFaceBrick3DDelta {
            face: 4,
            bx: 2,
            by: 1,
            bz: 1,
            total_depth,
            sub_depth,
            tick: 56,
            lz4_deltas,
        };
        let wire_d = encode_server_datagram(&delta);
        match decode_server_datagram(&wire_d).expect("decode PlanetFaceBrick3DDelta") {
            ServerDatagram::PlanetFaceBrick3DDelta { face, tick, lz4_deltas, .. } => {
                assert_eq!((face, tick), (4, 56));
                let mut r_dm = dm.clone();
                let mut r_at = at.clone();
                assert_eq!(apply_brick3d_delta(&mut r_dm, &mut r_at, &lz4_deltas).expect("apply"), 1);
                assert_eq!(r_dm, next_dm);
                assert_eq!(r_at, next_at);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }

    #[test]
    fn test_wave34_macro_planet_shard_and_climate_roundtrip() {
        let clim_raw = vec![128u8; 64 * 64 * 4];
        let clim_pkt = ServerDatagram::PlanetMacroClimateFace {
            face: 2,
            grid_n: 64,
            tick: 42,
            lz4_climate_rgba: compress_aplus_cells(&clim_raw),
        };
        let wire = encode_server_datagram(&clim_pkt);
        match decode_server_datagram(&wire).expect("decode PlanetMacroClimateFace") {
            ServerDatagram::PlanetMacroClimateFace { face, grid_n, tick, lz4_climate_rgba } => {
                assert_eq!((face, grid_n, tick), (2, 64, 42));
                assert_eq!(decompress_aplus_cells(&lz4_climate_rgba).unwrap().len(), 64 * 64 * 4);
            }
            other => panic!("unexpected variant: {:?}", other),
        }

        let voxels = CHUNK_SIZE * CHUNK_SIZE * 16;
        let dm = vec![200u8; voxels * 4];
        let at = vec![90u8; voxels * 4];
        let (ldm, lat) = encode_subbrick3d_lz4(&dm, &at);
        let shard_pkt = ServerDatagram::PlanetShardBrick3D {
            face: 4,
            sx: 3,
            sy: 5,
            bx: 1,
            by: 2,
            bz: 1,
            total_depth: 64,
            sub_depth: 16,
            tick: 77,
            lz4_density_mat: ldm,
            lz4_albedo_thermal: lat,
        };
        let wire2 = encode_server_datagram(&shard_pkt);
        match decode_server_datagram(&wire2).expect("decode PlanetShardBrick3D") {
            ServerDatagram::PlanetShardBrick3D {
                face,
                sx,
                sy,
                bx,
                by,
                bz,
                total_depth,
                sub_depth,
                tick,
                lz4_density_mat,
                lz4_albedo_thermal,
            } => {
                assert_eq!(
                    (face, sx, sy, bx, by, bz, total_depth, sub_depth, tick),
                    (4, 3, 5, 1, 2, 1, 64, 16, 77)
                );
                let (d2, a2) = decode_subbrick3d_lz4(&lz4_density_mat, &lz4_albedo_thermal).unwrap();
                assert_eq!(d2, dm);
                assert_eq!(a2, at);
            }
            other => panic!("unexpected variant: {:?}", other),
        }
    }
}
