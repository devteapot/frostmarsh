//! Wave 22B: Atomic Lossless World & Player Persistence (`"FMW1"` + LZ4-compressed bincode).
//!
//! Writes atomically via `<path>.tmp` -> `sync_all()` -> `rename()` so server crashes or SIGTERM
//! never corrupt the persistent MMO world state, and guarantees exact conservation
//! (`mass_error() == 0 && energy_error() == 0`) immediately upon reload.

use field_core::VolumeFieldSnapshot;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Versioned magic header for Frostmarsh Persistent World files (`"FMW1"`).
pub const PERSISTENCE_MAGIC: &[u8; 4] = b"FMW1";

/// Persistent per-player record keyed by player name.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PersistentPlayerRecord {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub class_id: u8,
    pub team: u8,
    pub xp: u16,
    pub kills: u16,
    pub deaths: u16,
    #[serde(default)]
    pub level: u8,
    #[serde(default)]
    pub score: u32,
}

/// Complete persistent MMO world + player snapshot payload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldPersistencePayload {
    pub volume: VolumeFieldSnapshot,
    pub players: HashMap<String, PersistentPlayerRecord>,
    pub world_epoch: u64,
    pub match_elapsed_secs: u64,
    #[serde(default)]
    pub world_day: u32,
    #[serde(default)]
    pub weather_kind: u8,
    #[serde(default)]
    pub day_time: f32,
    #[serde(default)]
    pub score_west: u16,
    #[serde(default)]
    pub score_east: u16,
    #[serde(default)]
    pub tunnels_carved: u64,
    #[serde(default)]
    pub tunnel_mass_displaced: u64,
}

/// Encode `WorldPersistencePayload` into `"FMW1"` + LZ4-compressed bincode bytes.
pub fn encode_persistence_payload(payload: &WorldPersistencePayload) -> Result<Vec<u8>, String> {
    let raw = bincode::serialize(payload).map_err(|e| format!("bincode serialize failed: {}", e))?;
    let compressed = lz4_flex::compress_prepend_size(&raw);
    let mut out = Vec::with_capacity(PERSISTENCE_MAGIC.len() + compressed.len());
    out.extend_from_slice(PERSISTENCE_MAGIC);
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Decode `"FMW1"` + LZ4-compressed bincode bytes back into `WorldPersistencePayload`.
pub fn decode_persistence_payload(bytes: &[u8]) -> Result<WorldPersistencePayload, String> {
    if bytes.len() < PERSISTENCE_MAGIC.len() + 4 {
        return Err(format!(
            "persistence file too short ({} bytes)",
            bytes.len()
        ));
    }
    if &bytes[0..4] != PERSISTENCE_MAGIC {
        return Err(format!(
            "unsupported persistence magic {:?} (expected {:?})",
            &bytes[0..4],
            PERSISTENCE_MAGIC
        ));
    }
    let raw = lz4_flex::decompress_size_prepended(&bytes[4..])
        .map_err(|e| format!("persistence lz4 decompress failed: {}", e))?;
    bincode::deserialize(&raw).map_err(|e| format!("persistence bincode deserialize failed: {}", e))
}

/// Atomically persist `payload` to `path` via `<path>.tmp` -> `sync_all()` -> `rename()`.
/// Returns total compressed bytes written on disk.
pub fn save_world_atomic(path: &Path, payload: &WorldPersistencePayload) -> Result<usize, String> {
    let encoded = encode_persistence_payload(payload)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create persistence dir {:?} failed: {}", parent, e))?;
        }
    }
    let mut tmp_os = path.as_os_str().to_os_string();
    tmp_os.push(".tmp");
    let tmp_path = PathBuf::from(tmp_os);

    {
        let mut file = File::create(&tmp_path)
            .map_err(|e| format!("create tmp persistence file {:?} failed: {}", tmp_path, e))?;
        file.write_all(&encoded)
            .map_err(|e| format!("write tmp persistence file {:?} failed: {}", tmp_path, e))?;
        file.sync_all()
            .map_err(|e| format!("sync_all tmp persistence file {:?} failed: {}", tmp_path, e))?;
    }

    fs::rename(&tmp_path, path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!("atomic rename {:?} -> {:?} failed: {}", tmp_path, path, e)
    })?;

    Ok(encoded.len())
}

/// Load a persisted world state file from disk.
pub fn load_world(path: &Path) -> Result<WorldPersistencePayload, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("read persistence file {:?} failed: {}", path, e))?;
    decode_persistence_payload(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use field_core::{VolumeField, BIOME_SINKHOLE_CAVERNS, WORLD_LAW_FREEZE_OUT};

    #[test]
    fn test_atomic_persistence_save_reload_exact_conservation() {
        let mut vf = VolumeField::new_biome_3d(32, 32, 16, BIOME_SINKHOLE_CAVERNS, 7);
        vf.set_world_law_preset(WORLD_LAW_FREEZE_OUT);
        vf.layers[4].apply_fire_bolt(14, 14, 3, 1.4);
        vf.layers[4].apply_frost_bolt(18, 18, 3, 1.8);
        vf.carve_tunnel_3d(16.0, 16.0, 5.0, 1.0, 0.0, 2.0);
        for _ in 0..3 {
            vf.step_world_evolution_3d(4.0);
            let raws = vf.refsim_raw_outputs_all();
            let st = vf.step_3d(&raws);
            assert_eq!(st.mass_error, 0);
            assert_eq!(st.energy_error, 0);
        }

        let mut players = HashMap::new();
        players.insert(
            "Valkyrie".to_string(),
            PersistentPlayerRecord {
                x: 19.5,
                y: 21.25,
                z: 6.5,
                class_id: 3,
                team: 1,
                xp: 85,
                kills: 7,
                deaths: 2,
                level: 3,
                score: 420,
            },
        );

        let payload = WorldPersistencePayload {
            volume: vf.to_snapshot(),
            players,
            world_epoch: vf.evolution_stats.world_epoch,
            match_elapsed_secs: 612,
            world_day: 4,
            weather_kind: 1,
            day_time: 0.65,
            score_west: 120,
            score_east: 145,
            tunnels_carved: 9,
            tunnel_mass_displaced: 4200,
        };

        let tmp_dir = std::env::temp_dir().join(format!(
            "frostmarsh_persist_test_{}",
            std::process::id()
        ));
        let state_path = tmp_dir.join("world.fmw1");
        let bytes_written = save_world_atomic(&state_path, &payload).expect("atomic save");
        assert!(bytes_written > 64);
        assert!(state_path.exists());
        let mut tmp_check = state_path.as_os_str().to_os_string();
        tmp_check.push(".tmp");
        assert!(!PathBuf::from(tmp_check).exists(), ".tmp must be atomically renamed away");

        let loaded = load_world(&state_path).expect("load_world");
        let _ = fs::remove_dir_all(&tmp_dir);

        assert_eq!(loaded.match_elapsed_secs, 612);
        assert_eq!(loaded.world_day, 4);
        assert_eq!(loaded.tunnels_carved, 9);
        assert_eq!(
            loaded.players.get("Valkyrie"),
            payload.players.get("Valkyrie")
        );

        let mut restored = VolumeField::from_snapshot(loaded.volume).expect("from_snapshot");
        assert_eq!(restored.mass_error(), 0);
        assert_eq!(restored.energy_error(), 0);
        assert_eq!(restored.biome_id, BIOME_SINKHOLE_CAVERNS);
        assert_eq!(restored.world_law_preset, WORLD_LAW_FREEZE_OUT);
        assert_eq!(restored.tick, vf.tick);

        let raws = restored.refsim_raw_outputs_all();
        let st = restored.step_3d(&raws);
        assert_eq!(st.mass_error, 0);
        assert_eq!(st.energy_error, 0);
    }
}
