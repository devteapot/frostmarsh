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

pub const CLASS_WARDEN: u8 = 0; // Frost Warden (Knight)
pub const CLASS_ALCHEMIST: u8 = 1; // Pyro Alchemist (Mage)
pub const CLASS_RANGER: u8 = 2; // Marsh Ranger (Scout)

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
        match_phase: u8,
        #[serde(default)]
        match_time_left_s: u16,
        #[serde(default)]
        mvp_name: String,
        #[serde(default)]
        mvp_score: u32,
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
