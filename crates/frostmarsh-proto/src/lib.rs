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
}
