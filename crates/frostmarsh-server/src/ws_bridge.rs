//! Wave 12B / Wave 24A: Pure-Rust Zero-Dependency RFC 6455 WebSocket Multiplayer Bridge.
//!
//! Streams exact Wave 12C / Wave 24A WebGPU/WASM Browser Client (`web/index.html`) binary frames
//! directly from the live authoritative `frostmarsh-server`:
//! - Frame `0x01` (Full Volumetric 3D Brick — initial sync & `>= 30%` massive mutations):
//!   `[u8=1, u16_le bx, u16_le by, u8 depth, u32_le den_len, density_rgba..., albedo_rgba...]`
//!   where `density_rgba` and `albedo_rgba` are raw Z-major `[z][dy][dx][rgba]` `16×16×depth`
//!   bricks packed by `VolumeField::encode_3d_brick_16`.
//! - Frame `0x02` (Live Entities & Telemetry JSON at ~15 Hz):
//!   `[u8=2, utf8_json...]` matching `crates/frostmarsh-web-client/src/live_entities.rs`.
//! - Frame `0x03` (Wave 24A Delta Volumetric 3D Brick — `< 30%` changed voxels with deadzone=5):
//!   `[u8=3, u16_le bx, u16_le by, u16_le depth, u16_le count, u8 flags=0, records...]`
//!   where each 10-byte record is `[u16_le voxel_idx, density_rgba[4], albedo_rgba[4]]`.
//! - Inbound Client Text Frames (`0x01`, masked):
//!   e.g. `{"type":"cycle_biome"}`, `{"type":"reset_world"}`, `{"type":"request_full_sync"}`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

const WS_MAGIC_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
const MAX_HANDSHAKE_BYTES: usize = 16 * 1024;
const MAX_CLIENT_FRAME_BYTES: usize = 1024 * 1024;
const PEER_QUEUE_CAPACITY: usize = 256;

/// Default maximum number of initial full-sync `0x01` bricks pushed per WS client per 10 Hz tick
/// (center-outward: 64 viewport bricks in 0.8s, full 256-brick world in 3.2s without 50-client spikes).
pub const WS_INITIAL_BRICKS_PER_TICK: usize = 16;

/// Default per-channel visual deadzone for WS `0x03` delta-brick encoding (`5/255`).
pub const WS_DELTA_DEADZONE: u8 = 5;

/// Bricks with `< 30%` changed voxels use `0x03` delta frames; `>= 30%` fall back to `0x01` full frames.
pub const WS_DELTA_MAX_CHANGE_PCT: usize = 30;

/// Per-client steady-state WS incremental brick wire budget per 10 Hz tick (`20 KB/tick` = `200 KB/s`).
pub const WS_MAX_DELTA_BYTES_PER_TICK: usize = 20 * 1024;

/// `0x03` delta flags byte: `0` = raw packed 10-byte records `[idx u16 LE, density[4], albedo[4]]`.
pub const WS_DELTA_FLAG_RAW: u8 = 0;

/// Header size of `0x03` delta-brick binary payload (matches `0x01`'s 10-byte header).
pub const WS_DELTA_HEADER_BYTES: usize = 10;

/// Per-voxel record size in `flags = 0` `0x03` delta-brick binary payload.
pub const WS_DELTA_RECORD_BYTES: usize = 10;

/// Shared `(density_rgba, albedo_rgba)` raw `16×16×D` brick buffer matching server `brick_raw_cache`.
pub type RawBrick = Arc<(Vec<u8>, Vec<u8>)>;

// ---------------- Tiny Zero-Dependency SHA-1 (FIPS 180-4) & Base64 ----------------

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xEFCD_AB89;
    let mut h2: u32 = 0x98BA_DCFE;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xC3D2_E1F0;

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let (mut a, mut b, mut c, mut d, mut e) = (h0, h1, h2, h3, h4);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDCu32),
                _ => (b ^ c ^ d, 0xCA62_C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

fn base64_encode(input: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// Compute RFC 6455 `Sec-WebSocket-Accept` response header value from a client's `Sec-WebSocket-Key`.
pub fn compute_ws_accept(sec_websocket_key: &str) -> String {
    let mut combined = Vec::with_capacity(sec_websocket_key.len() + WS_MAGIC_GUID.len());
    combined.extend_from_slice(sec_websocket_key.trim().as_bytes());
    combined.extend_from_slice(WS_MAGIC_GUID.as_bytes());
    base64_encode(&sha1(&combined))
}

// ---------------- RFC 6455 Framing & Frostmarsh Payloads ----------------

/// Encode an unmasked server-to-client RFC 6455 WebSocket frame (`FIN = 1`).
pub fn encode_ws_frame(opcode: u8, payload: &[u8]) -> Vec<u8> {
    let len = payload.len();
    let hdr_len = if len < 126 {
        2
    } else if len <= u16::MAX as usize {
        4
    } else {
        10
    };
    let mut out = Vec::with_capacity(hdr_len + len);
    out.push(0x80 | (opcode & 0x0F));
    if len < 126 {
        out.push(len as u8);
    } else if len <= u16::MAX as usize {
        out.push(126);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(127);
        out.extend_from_slice(&(len as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
    out
}

/// Encode an unmasked server-to-client RFC 6455 binary WebSocket frame (`opcode = 0x02`).
pub fn encode_ws_binary_frame(payload: &[u8]) -> Vec<u8> {
    encode_ws_frame(0x02, payload)
}

/// Build Frostmarsh Wave 12C Frame `0x01` payload:
/// `[u8=1, u16_le bx, u16_le by, u8 depth, u32_le den_len, density_rgba..., albedo_rgba...]`.
pub fn encode_brick_payload(
    bx: u16,
    by: u16,
    depth: u8,
    density_rgba: &[u8],
    albedo_rgba: &[u8],
) -> Vec<u8> {
    debug_assert_eq!(density_rgba.len(), albedo_rgba.len());
    let den_len = density_rgba.len().min(albedo_rgba.len());
    let mut out = Vec::with_capacity(10 + den_len * 2);
    out.push(0x01);
    out.extend_from_slice(&bx.to_le_bytes());
    out.extend_from_slice(&by.to_le_bytes());
    out.push(depth);
    out.extend_from_slice(&(den_len as u32).to_le_bytes());
    out.extend_from_slice(&density_rgba[..den_len]);
    out.extend_from_slice(&albedo_rgba[..den_len]);
    out
}

/// Build Frostmarsh Wave 12C Frame `0x02` payload: `[u8=2, utf8_json_bytes...]`.
pub fn encode_entities_payload(json: &str) -> Vec<u8> {
    let raw = json.as_bytes();
    let mut out = Vec::with_capacity(1 + raw.len());
    out.push(0x02);
    out.extend_from_slice(raw);
    out
}

/// Result of comparing a brick against a client's previously transmitted shadow brick.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BrickDeltaOutcome {
    /// Zero voxels exceeded the deadzone — send 0 bytes!
    Unchanged,
    /// `< 30%` of voxels changed — ready-to-frame `0x03` delta payload (`10 + 10 * count` bytes).
    Delta {
        payload: Vec<u8>,
        changed_voxels: usize,
    },
    /// `>= 30%` of voxels changed (or buffer geometry changed) — caller should send `0x01` full brick.
    FullRequired,
}

/// Build Frostmarsh Wave 24A Frame `0x03` delta-brick payload:
/// `[u8=3, u16_le bx, u16_le by, u16_le depth, u16_le count, u8 flags=0, records...]`
/// where each 10-byte record is `[u16_le idx, density_rgba[4], albedo_rgba[4]]`.
///
/// - Suppresses any voxel whose max per-channel absolute diff across all 8 bytes is `<= deadzone`.
/// - Returns `BrickDeltaOutcome::Unchanged` when 0 voxels changed (send 0 bytes).
/// - Returns `BrickDeltaOutcome::FullRequired` when `>= 30%` of voxels changed or dimensions mismatch.
pub fn encode_brick_delta_payload(
    bx: u16,
    by: u16,
    depth: u16,
    prev_density: &[u8],
    prev_albedo: &[u8],
    cur_density: &[u8],
    cur_albedo: &[u8],
    deadzone: u8,
) -> BrickDeltaOutcome {
    let byte_len = cur_density.len();
    if byte_len == 0
        || byte_len % 4 != 0
        || prev_density.len() != byte_len
        || prev_albedo.len() != byte_len
        || cur_albedo.len() != byte_len
    {
        return BrickDeltaOutcome::FullRequired;
    }
    let total_voxels = byte_len / 4;
    if total_voxels > u16::MAX as usize {
        return BrickDeltaOutcome::FullRequired;
    }

    // Strictly < 30% changed uses 0x03 delta; >= 30% falls back to 0x01 full frame.
    let max_delta_voxels = (total_voxels * WS_DELTA_MAX_CHANGE_PCT).saturating_sub(1) / 100;
    let mut out: Vec<u8> = Vec::new();
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
            return BrickDeltaOutcome::FullRequired;
        }
        if out.is_empty() {
            out.reserve(WS_DELTA_HEADER_BYTES + 64 * WS_DELTA_RECORD_BYTES);
            out.resize(WS_DELTA_HEADER_BYTES, 0);
        }
        out.extend_from_slice(&(idx as u16).to_le_bytes());
        out.extend_from_slice(&cur_density[o..o + 4]);
        out.extend_from_slice(&cur_albedo[o..o + 4]);
    }

    if changed == 0 {
        return BrickDeltaOutcome::Unchanged;
    }

    out[0] = 0x03;
    out[1..3].copy_from_slice(&bx.to_le_bytes());
    out[3..5].copy_from_slice(&by.to_le_bytes());
    out[5..7].copy_from_slice(&depth.to_le_bytes());
    out[7..9].copy_from_slice(&(changed as u16).to_le_bytes());
    out[9] = WS_DELTA_FLAG_RAW;

    BrickDeltaOutcome::Delta {
        payload: out,
        changed_voxels: changed,
    }
}

/// Decoded header & voxel patch list from a Wave 24A `0x03` delta-brick binary payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodedBrickDelta {
    pub bx: u16,
    pub by: u16,
    pub depth: u16,
    pub flags: u8,
    pub voxels: Vec<(u16, [u8; 4], [u8; 4])>,
}

/// Decode a Wave 24A `0x03` delta-brick binary payload back into its header and changed voxels.
#[allow(dead_code)]
pub fn decode_brick_delta_payload(payload: &[u8]) -> Result<DecodedBrickDelta, String> {
    if payload.len() < WS_DELTA_HEADER_BYTES {
        return Err(format!(
            "0x03 delta payload too short: {} bytes",
            payload.len()
        ));
    }
    if payload[0] != 0x03 {
        return Err(format!("Expected 0x03 delta tag, got 0x{:02X}", payload[0]));
    }
    let bx = u16::from_le_bytes([payload[1], payload[2]]);
    let by = u16::from_le_bytes([payload[3], payload[4]]);
    let depth = u16::from_le_bytes([payload[5], payload[6]]);
    let count = u16::from_le_bytes([payload[7], payload[8]]) as usize;
    let flags = payload[9];
    if flags != WS_DELTA_FLAG_RAW {
        return Err(format!("Unsupported 0x03 delta flags: {}", flags));
    }
    let expected = WS_DELTA_HEADER_BYTES + count * WS_DELTA_RECORD_BYTES;
    if payload.len() != expected {
        return Err(format!(
            "0x03 delta length mismatch: got {} bytes, expected {}",
            payload.len(),
            expected
        ));
    }
    let mut voxels = Vec::with_capacity(count);
    for rec in payload[WS_DELTA_HEADER_BYTES..].chunks_exact(WS_DELTA_RECORD_BYTES) {
        let idx = u16::from_le_bytes([rec[0], rec[1]]);
        let den = [rec[2], rec[3], rec[4], rec[5]];
        let alb = [rec[6], rec[7], rec[8], rec[9]];
        voxels.push((idx, den, alb));
    }
    Ok(DecodedBrickDelta {
        bx,
        by,
        depth,
        flags,
        voxels,
    })
}

/// Wave 17C: Additive tactical/synergy event entry streamed in the `0x02` live JSON `"events"` feed.
#[derive(Clone, Debug, PartialEq)]
pub struct WsTacticalEvent {
    pub label: String,
    pub team: u8,
    pub x: f32,
    pub y: f32,
    pub tick: u64,
}

fn escape_ws_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {}
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Append the additive `"events":[...]` JSON fragment (capped to the latest `max_events`,
/// default 5) onto an open JSON object buffer. 100% backwards-compatible with existing clients.
pub fn append_tactical_events_json(
    json: &mut String,
    events: &[WsTacticalEvent],
    max_events: usize,
) {
    use std::fmt::Write as _;
    json.push_str(",\"events\":[");
    let start = events.len().saturating_sub(max_events);
    for (i, ev) in events[start..].iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        let _ = write!(
            json,
            "{{\"label\":{},\"team\":{},\"x\":{:.2},\"y\":{:.2},\"tick\":{}}}",
            escape_ws_json_str(&ev.label),
            ev.team,
            ev.x,
            ev.y,
            ev.tick
        );
    }
    json.push(']');
}

/// Append the additive `"world_law":{...}` JSON fragment onto an open `0x02` JSON object buffer.
/// 100% backwards-compatible with existing clients that ignore unknown top-level keys.
pub fn append_world_law_json(
    json: &mut String,
    id: u8,
    name: &str,
    gravity_g: f32,
    ambient_shift: f32,
    precipitation: f32,
) {
    use std::fmt::Write as _;
    let _ = write!(
        json,
        ",\"world_law\":{{\"id\":{},\"name\":{},\"gravity_g\":{:.2},\"ambient_shift\":{:.1},\"precipitation\":{:.2}}}",
        id,
        escape_ws_json_str(name),
        gravity_g,
        ambient_shift,
        precipitation
    );
}

// ---------------- Client Commands & Live Peer Registry ----------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WsCommand {
    CycleBiome,
    CycleEvo,
    CycleLaws,
    ResetWorld,
    RequestFullSync { peer_id: u64 },
}

/// Parse a small JSON text frame command from a browser client (e.g. `{"type":"cycle_biome"}`).
pub fn parse_client_text_command(peer_id: u64, text: &str) -> Option<WsCommand> {
    let ty = extract_json_string_field(text, "type")?;
    match ty.as_str() {
        "cycle_biome" => Some(WsCommand::CycleBiome),
        "cycle_evo" => Some(WsCommand::CycleEvo),
        "cycle_laws" => Some(WsCommand::CycleLaws),
        "reset_world" => Some(WsCommand::ResetWorld),
        "request_full_sync" => Some(WsCommand::RequestFullSync { peer_id }),
        _ => None,
    }
}

fn extract_json_string_field(src: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let mut rest = src;
    while let Some(pos) = rest.find(&needle) {
        let after_key = &rest[pos + needle.len()..];
        let trimmed = after_key.trim_start();
        if let Some(after_colon) = trimmed.strip_prefix(':') {
            let val = after_colon.trim_start();
            if let Some(mut chars) = val.strip_prefix('"').map(str::chars) {
                let mut out = String::new();
                while let Some(ch) = chars.next() {
                    match ch {
                        '"' => return Some(out),
                        '\\' => {
                            if let Some(esc) = chars.next() {
                                out.push(esc);
                            } else {
                                return None;
                            }
                        }
                        c => out.push(c),
                    }
                }
                return None;
            }
        }
        rest = &rest[pos + needle.len()..];
    }
    None
}

fn ordered_brick_coords(chunks_x: usize, chunks_y: usize) -> VecDeque<(u16, u16)> {
    let mut coords = Vec::with_capacity(chunks_x * chunks_y);
    let cx_mid = (chunks_x as f32) * 0.5 - 0.5;
    let cy_mid = (chunks_y as f32) * 0.5 - 0.5;
    for by in 0..chunks_y {
        for bx in 0..chunks_x {
            let dx = bx as f32 - cx_mid;
            let dy = by as f32 - cy_mid;
            coords.push((dx * dx + dy * dy, bx as u16, by as u16));
        }
    }
    coords.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.1.cmp(&b.1))
    });
    coords.into_iter().map(|(_, bx, by)| (bx, by)).collect()
}

struct WsPeerState {
    tx: mpsc::Sender<Arc<[u8]>>,
    pending_initial_bricks: VecDeque<(u16, u16)>,
    pending_dirty_bricks: HashSet<(u16, u16)>,
    brick_shadow: HashMap<(u16, u16), RawBrick>,
}

#[derive(Clone)]
enum CachedDelta {
    Unchanged,
    Delta {
        frame: Arc<[u8]>,
        changed_voxels: usize,
    },
    FullRequired,
}

/// Shared handle to the live RFC 6455 WebSocket bridge server.
pub struct WsBridge {
    chunks_x: usize,
    chunks_y: usize,
    next_peer_id: AtomicU64,
    peers: Mutex<HashMap<u64, WsPeerState>>,
    commands: Mutex<VecDeque<WsCommand>>,
}

impl WsBridge {
    /// Bind a `tokio::net::TcpListener` on `bind_addr` and spawn the async RFC 6455 accept loop.
    pub async fn start(bind_addr: &str, chunks_x: usize, chunks_y: usize) -> io::Result<Arc<Self>> {
        let listener = TcpListener::bind(bind_addr).await?;
        let bridge = Arc::new(Self {
            chunks_x,
            chunks_y,
            next_peer_id: AtomicU64::new(1),
            peers: Mutex::new(HashMap::new()),
            commands: Mutex::new(VecDeque::new()),
        });
        let accept_bridge = bridge.clone();
        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, remote)) => {
                        let b = accept_bridge.clone();
                        tokio::spawn(async move {
                            handle_ws_connection(b, stream, remote).await;
                        });
                    }
                    Err(e) => {
                        log::warn!("WS bridge accept error: {}", e);
                        tokio::time::sleep(Duration::from_millis(25)).await;
                    }
                }
            }
        });
        Ok(bridge)
    }

    pub fn has_clients(&self) -> bool {
        !self.peers.lock().unwrap().is_empty()
    }

    pub fn drain_commands(&self) -> Vec<WsCommand> {
        let mut q = self.commands.lock().unwrap();
        q.drain(..).collect()
    }

    pub fn queue_full_sync_all(&self) {
        let coords = ordered_brick_coords(self.chunks_x, self.chunks_y);
        let mut peers = self.peers.lock().unwrap();
        for peer in peers.values_mut() {
            peer.pending_initial_bricks = coords.clone();
            peer.pending_dirty_bricks.clear();
            peer.brick_shadow.clear();
        }
    }

    pub fn queue_full_sync_peer(&self, peer_id: u64) {
        let coords = ordered_brick_coords(self.chunks_x, self.chunks_y);
        let mut peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get_mut(&peer_id) {
            peer.pending_initial_bricks = coords;
            peer.pending_dirty_bricks.clear();
            peer.brick_shadow.clear();
        }
    }

    /// Broadcast an already-encoded RFC 6455 WebSocket frame to all connected peers without blocking.
    pub fn broadcast_raw_frame(&self, ws_frame: Arc<[u8]>) {
        let mut peers = self.peers.lock().unwrap();
        if peers.is_empty() {
            return;
        }
        let mut dead = Vec::new();
        for (&pid, peer) in peers.iter() {
            match peer.tx.try_send(ws_frame.clone()) {
                Ok(()) => {}
                Err(mpsc::error::TrySendError::Full(_)) => {
                    // Drop transient frame for backpressured client so sim loop never stalls
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    dead.push(pid);
                }
            }
        }
        for pid in dead {
            peers.remove(&pid);
        }
    }

    /// Wave 24A: Smoothly pump paced initial full-sync `0x01` bricks and budget-capped `0x03`
    /// delta-bricks (with `0x01` fallback for `>= 30%` changes) using per-client shadow caches.
    ///
    /// - Encodes each distinct `(bx, by)` raw brick and each `(bx, by, prev_shadow_ptr)` delta
    ///   at most once per tick across all connected WS clients.
    /// - Prioritizes closest-to-center and most-changed bricks within `max_delta_bytes_per_peer`.
    /// - Deferred dirty bricks remain in each peer's `pending_dirty_bricks` and coalesce automatically.
    pub fn pump_bricks_delta<F>(
        &self,
        dirty_coords: &[(u16, u16)],
        depth: u16,
        max_initial_per_peer: usize,
        max_delta_bytes_per_peer: usize,
        deadzone: u8,
        mut get_raw_brick: F,
    ) where
        F: FnMut(u16, u16) -> RawBrick,
    {
        let mut peers = self.peers.lock().unwrap();
        if peers.is_empty() {
            return;
        }

        let mut raw_cache: HashMap<(u16, u16), RawBrick> = HashMap::new();
        let mut full_frame_cache: HashMap<(u16, u16), Arc<[u8]>> = HashMap::new();
        let mut delta_cache: HashMap<(u16, u16, usize), CachedDelta> = HashMap::new();
        let mut dead = Vec::new();

        let cx_mid = (self.chunks_x as f32) * 0.5 - 0.5;
        let cy_mid = (self.chunks_y as f32) * 0.5 - 0.5;
        let depth_u8 = depth.min(u8::MAX as u16) as u8;
        // Scale initial full-brick pacing smoothly when depth > 32 (e.g. Z=64 -> 128 KB/column)
        // so browser clients never experience sudden initial bandwidth spikes.
        let effective_initial_per_peer = if depth > 32 {
            (max_initial_per_peer / ((depth as usize) / 16).max(1)).max(2)
        } else {
            max_initial_per_peer
        };

        for (&pid, peer) in peers.iter_mut() {
            let mut is_dead = false;

            // 1. Stream paced initial full-sync 0x01 bricks (center-outward)
            let mut sent_init = 0usize;
            while sent_init < effective_initial_per_peer {
                let Some((bx, by)) = peer.pending_initial_bricks.pop_front() else {
                    break;
                };
                let raw = raw_cache
                    .entry((bx, by))
                    .or_insert_with(|| get_raw_brick(bx, by))
                    .clone();
                let frame = full_frame_cache
                    .entry((bx, by))
                    .or_insert_with(|| {
                        let payload = encode_brick_payload(bx, by, depth_u8, &raw.0, &raw.1);
                        Arc::from(encode_ws_binary_frame(&payload))
                    })
                    .clone();
                match peer.tx.try_send(frame) {
                    Ok(()) => {
                        peer.brick_shadow.insert((bx, by), raw);
                        peer.pending_dirty_bricks.remove(&(bx, by));
                        sent_init += 1;
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        peer.pending_initial_bricks.push_front((bx, by));
                        break;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        is_dead = true;
                        break;
                    }
                }
            }

            if is_dead {
                dead.push(pid);
                continue;
            }

            // 2. Queue dirty bricks that have already been synced into this peer's pending set
            let has_initial = !peer.pending_initial_bricks.is_empty();
            for &(bx, by) in dirty_coords {
                if has_initial && !peer.brick_shadow.contains_key(&(bx, by)) {
                    continue;
                }
                peer.pending_dirty_bricks.insert((bx, by));
            }

            // While initial sync is actively streaming full bricks this tick, hold incremental
            // deltas in `pending_dirty_bricks` so 50 concurrent joins never spike bandwidth.
            if sent_init > 0 || peer.pending_dirty_bricks.is_empty() {
                continue;
            }

            // 3. Evaluate diffs against this peer's shadow cache & prioritize closest/most-changed
            struct Candidate {
                bx: u16,
                by: u16,
                ring_band: u32,
                dist2_q: u32,
                is_full: bool,
                changed_voxels: usize,
                frame: Arc<[u8]>,
                cur_arc: RawBrick,
            }

            let pending_coords: Vec<(u16, u16)> = peer.pending_dirty_bricks.drain().collect();
            let mut candidates: Vec<Candidate> = Vec::with_capacity(pending_coords.len());

            for (bx, by) in pending_coords {
                let cur_arc = raw_cache
                    .entry((bx, by))
                    .or_insert_with(|| get_raw_brick(bx, by))
                    .clone();

                let dx = bx as f32 - cx_mid;
                let dy = by as f32 - cy_mid;
                let dist2_q = ((dx * dx + dy * dy) * 16.0) as u32;
                let ring_band = dist2_q / (16 * 16); // concentric 4-brick rings

                match peer.brick_shadow.get(&(bx, by)) {
                    Some(prev_arc) if Arc::ptr_eq(prev_arc, &cur_arc) => {
                        // Identical Arc -> 0 bytes!
                    }
                    Some(prev_arc) => {
                        let cache_key = (bx, by, Arc::as_ptr(prev_arc) as usize);
                        let cached = delta_cache
                            .entry(cache_key)
                            .or_insert_with(|| {
                                match encode_brick_delta_payload(
                                    bx,
                                    by,
                                    depth,
                                    &prev_arc.0,
                                    &prev_arc.1,
                                    &cur_arc.0,
                                    &cur_arc.1,
                                    deadzone,
                                ) {
                                    BrickDeltaOutcome::Unchanged => CachedDelta::Unchanged,
                                    BrickDeltaOutcome::Delta {
                                        payload,
                                        changed_voxels,
                                    } => CachedDelta::Delta {
                                        frame: Arc::from(encode_ws_binary_frame(&payload)),
                                        changed_voxels,
                                    },
                                    BrickDeltaOutcome::FullRequired => CachedDelta::FullRequired,
                                }
                            })
                            .clone();

                        match cached {
                            CachedDelta::Unchanged => {
                                // 0 voxels exceeded deadzone -> 0 bytes! Keep shadow so slow drift accumulates.
                            }
                            CachedDelta::Delta {
                                frame,
                                changed_voxels,
                            } => {
                                candidates.push(Candidate {
                                    bx,
                                    by,
                                    ring_band,
                                    dist2_q,
                                    is_full: false,
                                    changed_voxels,
                                    frame,
                                    cur_arc,
                                });
                            }
                            CachedDelta::FullRequired => {
                                let frame = full_frame_cache
                                    .entry((bx, by))
                                    .or_insert_with(|| {
                                        let payload = encode_brick_payload(
                                            bx,
                                            by,
                                            depth_u8,
                                            &cur_arc.0,
                                            &cur_arc.1,
                                        );
                                        Arc::from(encode_ws_binary_frame(&payload))
                                    })
                                    .clone();
                                candidates.push(Candidate {
                                    bx,
                                    by,
                                    ring_band,
                                    dist2_q,
                                    is_full: true,
                                    changed_voxels: cur_arc.0.len() / 4,
                                    frame,
                                    cur_arc,
                                });
                            }
                        }
                    }
                    None => {
                        let frame = full_frame_cache
                            .entry((bx, by))
                            .or_insert_with(|| {
                                let payload = encode_brick_payload(
                                    bx,
                                    by,
                                    depth_u8,
                                    &cur_arc.0,
                                    &cur_arc.1,
                                );
                                Arc::from(encode_ws_binary_frame(&payload))
                            })
                            .clone();
                        candidates.push(Candidate {
                            bx,
                            by,
                            ring_band,
                            dist2_q,
                            is_full: true,
                            changed_voxels: cur_arc.0.len() / 4,
                            frame,
                            cur_arc,
                        });
                    }
                }
            }

            // Prioritize compact 0x03 delta bricks before heavy 32KB full-brick fallbacks so one
            // massive brick never starves dozens of delta updates under the 20 KB/tick cap!
            candidates.sort_by(|a, b| {
                a.is_full
                    .cmp(&b.is_full)
                    .then_with(|| a.ring_band.cmp(&b.ring_band))
                    .then_with(|| b.changed_voxels.cmp(&a.changed_voxels))
                    .then_with(|| a.dist2_q.cmp(&b.dist2_q))
                    .then_with(|| a.by.cmp(&b.by))
                    .then_with(|| a.bx.cmp(&b.bx))
            });

            let mut bytes_sent = 0usize;
            let mut tx_full = false;
            for cand in candidates {
                if tx_full {
                    peer.pending_dirty_bricks.insert((cand.bx, cand.by));
                    continue;
                }
                let frame_len = cand.frame.len();
                if bytes_sent > 0 && bytes_sent.saturating_add(frame_len) > max_delta_bytes_per_peer
                {
                    peer.pending_dirty_bricks.insert((cand.bx, cand.by));
                    continue;
                }
                match peer.tx.try_send(cand.frame) {
                    Ok(()) => {
                        bytes_sent = bytes_sent.saturating_add(frame_len);
                        peer.brick_shadow.insert((cand.bx, cand.by), cand.cur_arc);
                    }
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        peer.pending_dirty_bricks.insert((cand.bx, cand.by));
                        tx_full = true;
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        dead.push(pid);
                        break;
                    }
                }
            }
        }

        for pid in dead {
            peers.remove(&pid);
        }
    }

    fn register_peer(&self, tx: mpsc::Sender<Arc<[u8]>>) -> u64 {
        let peer_id = self.next_peer_id.fetch_add(1, Ordering::Relaxed);
        let pending_initial_bricks = ordered_brick_coords(self.chunks_x, self.chunks_y);
        self.peers.lock().unwrap().insert(
            peer_id,
            WsPeerState {
                tx,
                pending_initial_bricks,
                pending_dirty_bricks: HashSet::new(),
                brick_shadow: HashMap::new(),
            },
        );
        peer_id
    }

    fn unregister_peer(&self, peer_id: u64) {
        self.peers.lock().unwrap().remove(&peer_id);
    }

    fn push_command(&self, cmd: WsCommand) {
        let mut q = self.commands.lock().unwrap();
        if q.len() < 64 {
            q.push_back(cmd);
        }
    }
}

// ---------------- Per-Connection RFC 6455 Handshake & Frame Loop ----------------

async fn ws_handshake(stream: &mut TcpStream) -> io::Result<bool> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 512];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);

    loop {
        let now = tokio::time::Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        let n = match tokio::time::timeout(deadline - now, stream.read(&mut tmp)).await {
            Ok(Ok(0)) => return Ok(false),
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Ok(false),
        };
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > MAX_HANDSHAKE_BYTES {
            let _ = stream
                .write_all(b"HTTP/1.1 431 Request Header Fields Too Large\r\nContent-Length: 0\r\n\r\n")
                .await;
            return Ok(false);
        }
    }

    let req_str = String::from_utf8_lossy(&buf);
    let mut ws_key: Option<String> = None;
    for line in req_str.split("\r\n").skip(1) {
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("sec-websocket-key") {
                let val = v.trim();
                if !val.is_empty() {
                    ws_key = Some(val.to_string());
                }
                break;
            }
        }
    }

    let Some(key) = ws_key else {
        let _ = stream
            .write_all(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
            .await;
        return Ok(false);
    };

    let accept = compute_ws_accept(&key);
    let resp = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
        accept
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.flush().await?;
    Ok(true)
}

async fn read_ws_frame(
    reader: &mut tokio::net::tcp::OwnedReadHalf,
) -> io::Result<Option<(u8, Vec<u8>)>> {
    let mut hdr = [0u8; 2];
    match reader.read_exact(&mut hdr).await {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }

    let opcode = hdr[0] & 0x0F;
    let masked = (hdr[1] & 0x80) != 0;
    let len_code = hdr[1] & 0x7F;
    let payload_len: usize = match len_code {
        0..=125 => len_code as usize,
        126 => {
            let mut ext = [0u8; 2];
            reader.read_exact(&mut ext).await?;
            u16::from_be_bytes(ext) as usize
        }
        _ => {
            let mut ext = [0u8; 8];
            reader.read_exact(&mut ext).await?;
            let len64 = u64::from_be_bytes(ext);
            if len64 > MAX_CLIENT_FRAME_BYTES as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "WS client frame exceeds limit",
                ));
            }
            len64 as usize
        }
    };

    if payload_len > MAX_CLIENT_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "WS client frame exceeds limit",
        ));
    }

    let mut mask = [0u8; 4];
    if masked {
        reader.read_exact(&mut mask).await?;
    }

    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        reader.read_exact(&mut payload).await?;
        if masked {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i & 3];
            }
        }
    }

    Ok(Some((opcode, payload)))
}

async fn handle_ws_connection(
    bridge: Arc<WsBridge>,
    mut stream: TcpStream,
    remote: std::net::SocketAddr,
) {
    let _ = stream.set_nodelay(true);
    match ws_handshake(&mut stream).await {
        Ok(true) => {}
        Ok(false) => return,
        Err(e) => {
            log::debug!("WS handshake failed for {}: {}", remote, e);
            return;
        }
    }

    let (mut reader, mut writer) = stream.into_split();
    let (tx, mut rx) = mpsc::channel::<Arc<[u8]>>(PEER_QUEUE_CAPACITY);
    let peer_id = bridge.register_peer(tx.clone());
    log::info!("WS browser client #{} connected from {}", peer_id, remote);

    let writer_task = tokio::spawn(async move {
        while let Some(frame) = rx.recv().await {
            if writer.write_all(&frame).await.is_err() {
                break;
            }
        }
        let _ = writer.shutdown().await;
    });

    loop {
        match read_ws_frame(&mut reader).await {
            Ok(Some((0x09, payload))) => {
                // Ping -> Pong (0x0A) with identical payload
                let pong: Arc<[u8]> = Arc::from(encode_ws_frame(0x0A, &payload));
                let _ = tx.try_send(pong);
            }
            Ok(Some((0x08, _))) => {
                // Close -> echo Close (0x08) and terminate cleanly
                let close: Arc<[u8]> = Arc::from(encode_ws_frame(0x08, &[]));
                let _ = tx.try_send(close);
                break;
            }
            Ok(Some((0x01, payload))) => {
                // Text frame (masked from browser)
                if let Ok(text) = std::str::from_utf8(&payload) {
                    if let Some(cmd) = parse_client_text_command(peer_id, text) {
                        log::info!("WS client #{} command: {:?}", peer_id, cmd);
                        bridge.push_command(cmd);
                    }
                }
            }
            Ok(Some(_)) => {
                // Ignore binary/pong/continuation frames from viewer clients
            }
            Ok(None) | Err(_) => break,
        }
    }

    bridge.unregister_peer(peer_id);
    drop(tx);
    let _ = writer_task.await;
    log::info!("WS browser client #{} disconnected ({})", peer_id, remote);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rfc6455_accept_vector() {
        // Official RFC 6455 Section 1.3 handshake test vector
        assert_eq!(
            compute_ws_accept("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn test_ws_binary_frame_encoding() {
        // Short payload (< 126 bytes)
        let short_payload = [0x02u8, b'{', b'}'];
        let f_short = encode_ws_binary_frame(&short_payload);
        assert_eq!(f_short[0], 0x82); // FIN=1, opcode=0x02 (binary)
        assert_eq!(f_short[1], 3); // unmasked, len=3
        assert_eq!(&f_short[2..], &short_payload);

        // Medium payload (126..=65535 bytes, e.g. Z=16 brick = 32778 bytes)
        let med_payload = vec![0xABu8; 32_778];
        let f_med = encode_ws_binary_frame(&med_payload);
        assert_eq!(f_med[0], 0x82);
        assert_eq!(f_med[1], 126);
        assert_eq!(u16::from_be_bytes([f_med[2], f_med[3]]), 32_778);
        assert_eq!(f_med.len(), 4 + 32_778);
        assert_eq!(&f_med[4..], &med_payload[..]);

        // Large payload (>= 65536 bytes, e.g. Z=32 brick = 65546 bytes)
        let large_payload = vec![0xCDu8; 65_546];
        let f_large = encode_ws_binary_frame(&large_payload);
        assert_eq!(f_large[0], 0x82);
        assert_eq!(f_large[1], 127);
        assert_eq!(
            u64::from_be_bytes(f_large[2..10].try_into().unwrap()),
            65_546
        );
        assert_eq!(f_large.len(), 10 + 65_546);
        assert_eq!(&f_large[10..], &large_payload[..]);
    }

    #[test]
    fn test_brick_and_entities_payload_framing() {
        let den = vec![0x11u8; 16 * 16 * 16 * 4];
        let alb = vec![0x22u8; 16 * 16 * 16 * 4];
        let pkt = encode_brick_payload(3, 5, 16, &den, &alb);
        assert_eq!(pkt[0], 0x01);
        assert_eq!(u16::from_le_bytes([pkt[1], pkt[2]]), 3);
        assert_eq!(u16::from_le_bytes([pkt[3], pkt[4]]), 5);
        assert_eq!(pkt[5], 16);
        assert_eq!(
            u32::from_le_bytes([pkt[6], pkt[7], pkt[8], pkt[9]]) as usize,
            den.len()
        );
        assert_eq!(&pkt[10..10 + den.len()], &den[..]);
        assert_eq!(&pkt[10 + den.len()..], &alb[..]);

        let ent = encode_entities_payload(r#"{"tick":42}"#);
        assert_eq!(ent[0], 0x02);
        assert_eq!(&ent[1..], br#"{"tick":42}"#);
    }

    #[test]
    fn test_parse_client_text_command() {
        assert_eq!(
            parse_client_text_command(7, r#"{"type":"cycle_biome"}"#),
            Some(WsCommand::CycleBiome)
        );
        assert_eq!(
            parse_client_text_command(7, r#"{"type":"cycle_evo"}"#),
            Some(WsCommand::CycleEvo)
        );
        assert_eq!(
            parse_client_text_command(7, r#"{"type":"cycle_laws"}"#),
            Some(WsCommand::CycleLaws)
        );
        assert_eq!(
            parse_client_text_command(7, r#"{ "type" : "reset_world", "v": 1 }"#),
            Some(WsCommand::ResetWorld)
        );
        assert_eq!(
            parse_client_text_command(7, r#"{"type":"request_full_sync"}"#),
            Some(WsCommand::RequestFullSync { peer_id: 7 })
        );
        assert_eq!(parse_client_text_command(7, r#"{"type":"noop"}"#), None);
    }

    #[test]
    fn test_wave17c_tactical_events_feed_0x02() {
        let mut evs = Vec::new();
        for i in 0..7u64 {
            evs.push(WsTacticalEvent {
                label: match i {
                    0 => "OLD EVENT 0".to_string(),
                    1 => "OLD EVENT 1".to_string(),
                    2 => "TIDEKIN QUENCH RESCUE!".to_string(),
                    3 => "SHIELD WALL!".to_string(),
                    4 => "SUBTERRANEAN BREACH!".to_string(),
                    5 => "CRYO-BRIDGE FORGED!".to_string(),
                    _ => "BASALT BULWARK!".to_string(),
                },
                team: (i % 2) as u8,
                x: 100.0 + i as f32,
                y: 80.0 + i as f32,
                tick: 100 + i,
            });
        }
        let mut json = String::from(r#"{"tick":106,"players":[]"#);
        append_tactical_events_json(&mut json, &evs, 5);
        json.push('}');

        let pkt = encode_entities_payload(&json);
        assert_eq!(pkt[0], 0x02);
        let body = std::str::from_utf8(&pkt[1..]).expect("utf8 json");
        // Keeps only the latest 5 tactical events (drops OLD EVENT 0 and OLD EVENT 1)
        assert!(!body.contains("OLD EVENT 0"));
        assert!(!body.contains("OLD EVENT 1"));
        assert!(body.contains("TIDEKIN QUENCH RESCUE!"));
        assert!(body.contains("SHIELD WALL!"));
        assert!(body.contains("SUBTERRANEAN BREACH!"));
        assert!(body.contains("CRYO-BRIDGE FORGED!"));
        assert!(body.contains("BASALT BULWARK!"));
        assert!(body.contains("\"events\":["));
    }

    #[test]
    fn test_wave20c_world_law_json_0x02() {
        let mut json = String::from(r#"{"tick":42,"players":[]"#);
        append_world_law_json(&mut json, 3, "LOW-G MAGMA", 0.35, 5500.0, 0.25);
        json.push('}');
        let pkt = encode_entities_payload(&json);
        assert_eq!(pkt[0], 0x02);
        let body = std::str::from_utf8(&pkt[1..]).expect("utf8 json");
        assert!(body.contains(r#""world_law":{"id":3,"name":"LOW-G MAGMA","gravity_g":0.35,"ambient_shift":5500.0,"precipitation":0.25}"#));
    }

    #[test]
    fn test_wave24a_ws_0x03_delta_encoding_deadzone_and_compression() {
        let depth = 16u16;
        let voxels = 16 * 16 * (depth as usize);
        let byte_len = voxels * 4;
        let prev_den = vec![100u8; byte_len];
        let prev_alb = vec![120u8; byte_len];

        // 1. Sub-deadzone jitter (<= 5 across all 8 channels) must emit 0 bytes (Unchanged)
        let mut jitter_den = prev_den.clone();
        let mut jitter_alb = prev_alb.clone();
        for idx in (0..voxels).step_by(7) {
            let o = idx * 4;
            jitter_den[o] = prev_den[o] + 5;
            jitter_alb[o + 3] = prev_alb[o + 3] - 4;
        }
        assert_eq!(
            encode_brick_delta_payload(
                4,
                9,
                depth,
                &prev_den,
                &prev_alb,
                &jitter_den,
                &jitter_alb,
                WS_DELTA_DEADZONE,
            ),
            BrickDeltaOutcome::Unchanged
        );

        // 2. Realistic 48-voxel mutation -> exact 0x03 roundtrip and > 15x compression vs 0x01
        let mut cur_den = prev_den.clone();
        let mut cur_alb = prev_alb.clone();
        let mut expected_indices = Vec::new();
        for k in 0..48usize {
            let idx = k * 73;
            let o = idx * 4;
            cur_den[o] = 250;
            cur_den[o + 1] = (k as u8).wrapping_mul(5);
            cur_den[o + 2] = 42;
            cur_den[o + 3] = 180;
            cur_alb[o] = 255;
            cur_alb[o + 1] = 140;
            cur_alb[o + 2] = 32;
            cur_alb[o + 3] = 220;
            expected_indices.push(idx as u16);
        }

        let full_pkt = encode_brick_payload(4, 9, depth as u8, &cur_den, &cur_alb);
        let outcome = encode_brick_delta_payload(
            4,
            9,
            depth,
            &prev_den,
            &prev_alb,
            &cur_den,
            &cur_alb,
            WS_DELTA_DEADZONE,
        );
        let (delta_pkt, changed_voxels) = match outcome {
            BrickDeltaOutcome::Delta {
                payload,
                changed_voxels,
            } => (payload, changed_voxels),
            other => panic!("Expected Delta outcome, got {:?}", other),
        };
        assert_eq!(changed_voxels, 48);
        assert_eq!(delta_pkt.len(), WS_DELTA_HEADER_BYTES + 48 * WS_DELTA_RECORD_BYTES);
        assert_eq!(delta_pkt[0], 0x03);
        // 32,778 bytes vs 490 bytes = 66.9x smaller (far above 15x requirement!)
        assert!(
            full_pkt.len() > delta_pkt.len() * 15,
            "Expected > 15x compression: full={} vs delta={}",
            full_pkt.len(),
            delta_pkt.len()
        );

        // Apply decoded delta onto prev copy and verify exact match with cur
        let decoded = decode_brick_delta_payload(&delta_pkt).expect("decode 0x03 delta");
        assert_eq!((decoded.bx, decoded.by, decoded.depth, decoded.flags), (4, 9, depth, 0));
        assert_eq!(decoded.voxels.len(), 48);

        let mut recon_den = prev_den.clone();
        let mut recon_alb = prev_alb.clone();
        for (i, &(idx, d4, a4)) in decoded.voxels.iter().enumerate() {
            assert_eq!(idx, expected_indices[i]);
            let o = (idx as usize) * 4;
            recon_den[o..o + 4].copy_from_slice(&d4);
            recon_alb[o..o + 4].copy_from_slice(&a4);
        }
        assert_eq!(recon_den, cur_den);
        assert_eq!(recon_alb, cur_alb);

        // 3. >= 30% mutation triggers FullRequired fallback to 0x01
        let mut huge_den = prev_den.clone();
        let cutoff = (voxels * WS_DELTA_MAX_CHANGE_PCT).div_ceil(100);
        for idx in 0..cutoff {
            huge_den[idx * 4] = 255;
        }
        assert_eq!(
            encode_brick_delta_payload(
                4,
                9,
                depth,
                &prev_den,
                &prev_alb,
                &huge_den,
                &prev_alb,
                WS_DELTA_DEADZONE,
            ),
            BrickDeltaOutcome::FullRequired
        );
    }

    #[test]
    fn test_wave24a_ws_bridge_shadow_cache_and_budget_pacing() {
        let bridge = WsBridge {
            chunks_x: 4,
            chunks_y: 4,
            next_peer_id: AtomicU64::new(1),
            peers: Mutex::new(HashMap::new()),
            commands: Mutex::new(VecDeque::new()),
        };
        let (tx, mut rx) = mpsc::channel::<Arc<[u8]>>(256);
        let _pid = bridge.register_peer(tx);

        let depth = 16u16;
        let byte_len = 16 * 16 * (depth as usize) * 4;
        let base_brick: RawBrick = Arc::new((vec![50u8; byte_len], vec![80u8; byte_len]));

        // Drain all 16 initial bricks across 2 ticks (8 bricks/tick) -> all 0x01
        bridge.pump_bricks_delta(&[], depth, 8, WS_MAX_DELTA_BYTES_PER_TICK, WS_DELTA_DEADZONE, |_, _| {
            base_brick.clone()
        });
        bridge.pump_bricks_delta(&[], depth, 8, WS_MAX_DELTA_BYTES_PER_TICK, WS_DELTA_DEADZONE, |_, _| {
            base_brick.clone()
        });

        let mut init_full_count = 0usize;
        while let Ok(frame) = rx.try_recv() {
            // 32,778-byte payload uses 4-byte WS header (0x82, 126, len_be_u16)
            assert_eq!(frame[0], 0x82);
            let payload_tag = if frame[1] == 126 { frame[4] } else { frame[2] };
            assert_eq!(payload_tag, 0x01);
            init_full_count += 1;
        }
        assert_eq!(init_full_count, 16);

        // Next tick with sub-deadzone jitter -> 0 frames sent!
        let mut jitter_dm = vec![50u8; byte_len];
        jitter_dm[0] = 54;
        let jitter_brick: RawBrick = Arc::new((jitter_dm, vec![80u8; byte_len]));
        bridge.pump_bricks_delta(
            &[(1, 1), (2, 2)],
            depth,
            8,
            WS_MAX_DELTA_BYTES_PER_TICK,
            WS_DELTA_DEADZONE,
            |_, _| jitter_brick.clone(),
        );
        assert!(rx.try_recv().is_err(), "sub-deadzone jitter must send 0 bytes");

        // Next tick with real 20-voxel edit on 3 bricks -> 3 x 0x03 delta frames!
        let mut mod_dm = vec![50u8; byte_len];
        let mut mod_at = vec![80u8; byte_len];
        for k in 0..20usize {
            let o = (k * 100) * 4;
            mod_dm[o] = 240;
            mod_at[o + 3] = 210;
        }
        let mod_brick: RawBrick = Arc::new((mod_dm, mod_at));
        bridge.pump_bricks_delta(
            &[(1, 1), (2, 1), (0, 0)],
            depth,
            8,
            WS_MAX_DELTA_BYTES_PER_TICK,
            WS_DELTA_DEADZONE,
            |_, _| mod_brick.clone(),
        );

        let mut delta_frames = 0usize;
        while let Ok(frame) = rx.try_recv() {
            assert_eq!(frame[0], 0x82);
            let payload = if frame[1] < 126 {
                &frame[2..]
            } else if frame[1] == 126 {
                &frame[4..]
            } else {
                &frame[10..]
            };
            assert_eq!(payload[0], 0x03);
            let dec = decode_brick_delta_payload(payload).expect("valid 0x03");
            assert_eq!(dec.voxels.len(), 20);
            delta_frames += 1;
        }
        assert_eq!(delta_frames, 3);
    }
}
