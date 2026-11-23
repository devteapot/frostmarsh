//! Wave 12B: Pure-Rust Zero-Dependency RFC 6455 WebSocket Multiplayer Bridge.
//!
//! Streams exact Wave 12C WebGPU/WASM Browser Client (`web/index.html`) binary frames directly
//! from the live authoritative `frostmarsh-server`:
//! - Frame `0x01` (Volumetric 3D Brick):
//!   `[u8=1, u16_le bx, u16_le by, u8 depth, u32_le den_len, density_rgba..., albedo_rgba...]`
//!   where `density_rgba` and `albedo_rgba` are raw Z-major `[z][dy][dx][rgba]` `16×16×depth`
//!   bricks packed by `VolumeField::encode_3d_brick_16`.
//! - Frame `0x02` (Live Entities & Telemetry JSON at ~15 Hz):
//!   `[u8=2, utf8_json...]` matching `crates/frostmarsh-web-client/src/live_entities.rs`.
//! - Inbound Client Text Frames (`0x01`, masked):
//!   e.g. `{"type":"cycle_biome"}`, `{"type":"reset_world"}`, `{"type":"request_full_sync"}`.

use std::collections::{HashMap, VecDeque};
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

/// Default maximum number of initial full-sync bricks pushed per WS client per 10 Hz tick.
pub const WS_INITIAL_BRICKS_PER_TICK: usize = 16;

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

// ---------------- Client Commands & Live Peer Registry ----------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WsCommand {
    CycleBiome,
    ResetWorld,
    RequestFullSync { peer_id: u64 },
}

/// Parse a small JSON text frame command from a browser client (e.g. `{"type":"cycle_biome"}`).
pub fn parse_client_text_command(peer_id: u64, text: &str) -> Option<WsCommand> {
    let ty = extract_json_string_field(text, "type")?;
    match ty.as_str() {
        "cycle_biome" => Some(WsCommand::CycleBiome),
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
        }
    }

    pub fn queue_full_sync_peer(&self, peer_id: u64) {
        let coords = ordered_brick_coords(self.chunks_x, self.chunks_y);
        let mut peers = self.peers.lock().unwrap();
        if let Some(peer) = peers.get_mut(&peer_id) {
            peer.pending_initial_bricks = coords;
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

    /// Smoothly pump throttled initial full-sync bricks and live dirty `0x01` bricks to connected WS peers.
    /// Encodes each distinct `(bx, by)` at most once per tick via `encode_brick`.
    pub fn pump_bricks<F>(
        &self,
        dirty_coords: &[(u16, u16)],
        max_initial_per_peer: usize,
        mut encode_brick: F,
    ) where
        F: FnMut(u16, u16) -> Arc<[u8]>,
    {
        let mut peers = self.peers.lock().unwrap();
        if peers.is_empty() {
            return;
        }
        let mut frame_cache: HashMap<(u16, u16), Arc<[u8]>> = HashMap::new();
        let mut dead = Vec::new();

        for (&pid, peer) in peers.iter_mut() {
            let mut is_dead = false;

            // 1. Stream throttled initial full-sync bricks (center-outward)
            let mut sent_init = 0usize;
            while sent_init < max_initial_per_peer {
                let Some((bx, by)) = peer.pending_initial_bricks.pop_front() else {
                    break;
                };
                let frame = frame_cache
                    .entry((bx, by))
                    .or_insert_with(|| encode_brick(bx, by))
                    .clone();
                match peer.tx.try_send(frame) {
                    Ok(()) => {
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

            // 2. Broadcast live dirty bricks (skip any still queued for initial sync to avoid redundant sends)
            for &(bx, by) in dirty_coords {
                if !peer.pending_initial_bricks.is_empty()
                    && peer.pending_initial_bricks.contains(&(bx, by))
                {
                    continue;
                }
                let frame = frame_cache
                    .entry((bx, by))
                    .or_insert_with(|| encode_brick(bx, by))
                    .clone();
                match peer.tx.try_send(frame) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => break,
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
            parse_client_text_command(7, r#"{ "type" : "reset_world", "v": 1 }"#),
            Some(WsCommand::ResetWorld)
        );
        assert_eq!(
            parse_client_text_command(7, r#"{"type":"request_full_sync"}"#),
            Some(WsCommand::RequestFullSync { peer_id: 7 })
        );
        assert_eq!(parse_client_text_command(7, r#"{"type":"noop"}"#), None);
    }
}
