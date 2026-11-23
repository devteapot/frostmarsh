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
        assert_eq!(CLASS_WARDEN, CLASS_STRIDER);
        assert_eq!(CLASS_ALCHEMIST, CLASS_SKYWING);
        assert_eq!(CLASS_RANGER, CLASS_TIDEKIN);
        assert_eq!(CLASS_EMBERSTALKER, 3);
    }
}
