//! Live 3D Volumetric QUIC (`quinn`) Client for Frostmarsh Unified v3 (`256×256×4` = 262,144 voxels!).
//! Receives sparse layered `APlusChunk { layer, cx, cy, lz4_cells }` across all 4 vertical strata
//! plus 30 Hz 3D `EntitySnapshot`s (with automatic WAN/Tailscale MTU uni-stream fallback).

use frostmarsh_proto::*;
use quinn::{ClientConfig, Connection, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static CURRENT_WORLD_LAW_PRESET: AtomicU8 = AtomicU8::new(0);

#[inline]
pub fn current_world_law_preset() -> u8 {
    CURRENT_WORLD_LAW_PRESET.load(Ordering::Relaxed)
}

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::RSA_PKCS1_SHA256,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512,
            SignatureScheme::ED25519,
        ]
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct LiveEntityState {
    pub sim_tick: u64,
    pub match_time_secs: u32,
    pub your_player_id: u32,
    pub echo_client_time_ms: u32,
    pub relic_x: f32,
    pub relic_y: f32,
    pub players: Vec<PlayerSnap>,
    pub projectiles: Vec<ProjectileSnap>,
    pub bodies: Vec<CellBodySnap>,
    pub shrines: Vec<ShrineSnap>,
    pub events: Vec<CombatEvent>,
    pub telemetry: TelemetrySnap,
    pub bot_brains: Vec<BotBrainSnap>,
    pub kill_feed: Vec<KillFeedEntry>,
    pub speech_bubbles: Vec<SpeechBubbleSnap>,
    pub actuator_pulses: Vec<ActuatorPulseSnap>,
    pub match_phase: u8,
    pub match_time_left_s: u16,
    pub mvp_name: String,
    pub mvp_score: u32,
    pub time_domes: Vec<TimeDomeSnap>,
}

pub struct LiveWorldState {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub chunks_x: usize,
    pub chunks_y: usize,
    pub your_player_id: u32,
    /// Layered 4-byte `A+` cells: `aplus_layers[layer][y * width + x]` (defaults to empty air `[0xff, 0, 26, 1]`).
    pub aplus_layers: Vec<Vec<[u8; 4]>>,
    /// Received chunk flags for surface layer (`surf_layer`) to track initial sync progress.
    pub received_surf_chunks: Vec<bool>,
    /// Dirty `(layer, chunk_idx)` flags across all `depth` layers.
    pub dirty_chunks: Vec<Vec<bool>>,
    /// Flat client-side true 3D `Z=16` staging arrays in GPU `(X=width, Y=depth, Z=height)` order:
    /// index = `((iz * depth + iy) * width + ix) * 4`. Lazily allocated when `depth >= 16`.
    pub z16_density_mat: Vec<u8>,
    pub z16_albedo_thermal: Vec<u8>,
    pub received_z16_bricks: Vec<bool>,
    pub dirty_z16_mask: Vec<bool>,
    pub dirty_z16_bricks: Vec<(usize, usize)>,
    pub dirty_z16_subbrick_mask: Vec<bool>,
    pub dirty_z16_subbricks: Vec<(usize, usize, usize)>,
    pub has_z16_bricks: bool,
    pub bricks_3d_rx: u64,
    pub bricks_3d_delta_rx: u64,
    pub latest_field_tick: u64,
    pub entities: Option<LiveEntityState>,
    pub chunks_rx: u64,
    pub snapshots_rx: u64,
    pub bytes_rx: u64,
    pub start_instant: Instant,
    pub rtt_ms: f32,
    pub world_law_preset: u8,
}

#[allow(dead_code)]
pub type SharedNetState = LiveWorldState;

impl LiveWorldState {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let d = depth.max(1);
        let cx = chunks_for_dim(width);
        let cy = chunks_for_dim(height);
        let cz = chunks_for_dim(d);
        let z16_bytes = if d >= 16 { width * height * d * 4 } else { 0 };
        Self {
            width,
            height,
            depth: d,
            chunks_x: cx,
            chunks_y: cy,
            your_player_id: 0,
            aplus_layers: vec![vec![[0xff, 0, 26, 1]; width * height]; d],
            received_surf_chunks: vec![false; cx * cy],
            dirty_chunks: vec![vec![false; cx * cy]; d],
            z16_density_mat: vec![0u8; z16_bytes],
            z16_albedo_thermal: vec![0u8; z16_bytes],
            received_z16_bricks: vec![false; cx * cy],
            dirty_z16_mask: vec![false; cx * cy],
            dirty_z16_bricks: Vec::new(),
            dirty_z16_subbrick_mask: vec![false; cx * cy * cz],
            dirty_z16_subbricks: Vec::new(),
            has_z16_bricks: d >= 16,
            bricks_3d_rx: 0,
            bricks_3d_delta_rx: 0,
            latest_field_tick: 0,
            entities: None,
            chunks_rx: 0,
            snapshots_rx: 0,
            bytes_rx: 0,
            start_instant: Instant::now(),
            rtt_ms: 0.0,
            world_law_preset: 0,
        }
    }

    pub fn surf_layer(&self) -> usize {
        if self.depth >= 2 { 1 } else { 0 }
    }

    pub fn resize_if_needed(&mut self, w: usize, h: usize, d: usize) {
        let d = d.max(1);
        if w == 0 || h == 0 || (self.width == w && self.height == h && self.depth == d) {
            return;
        }
        self.width = w;
        self.height = h;
        self.depth = d;
        self.chunks_x = chunks_for_dim(w);
        self.chunks_y = chunks_for_dim(h);
        self.aplus_layers = vec![vec![[0xff, 0, 26, 1]; w * h]; d];
        self.received_surf_chunks = vec![false; self.chunks_x * self.chunks_y];
        self.dirty_chunks = vec![vec![false; self.chunks_x * self.chunks_y]; d];
        let z16_bytes = if d >= 16 { w * h * d * 4 } else { 0 };
        self.z16_density_mat = vec![0u8; z16_bytes];
        self.z16_albedo_thermal = vec![0u8; z16_bytes];
        self.received_z16_bricks = vec![false; self.chunks_x * self.chunks_y];
        self.dirty_z16_mask = vec![false; self.chunks_x * self.chunks_y];
        self.dirty_z16_bricks.clear();
        let cz = chunks_for_dim(d);
        self.dirty_z16_subbrick_mask = vec![false; self.chunks_x * self.chunks_y * cz];
        self.dirty_z16_subbricks.clear();
        self.has_z16_bricks = d >= 16;
    }

    pub fn total_surf_chunks(&self) -> usize {
        self.chunks_x * self.chunks_y
    }

    pub fn received_surf_count(&self) -> usize {
        self.received_surf_chunks.iter().filter(|&&b| b).count()
    }

    pub fn received_z16_count(&self) -> usize {
        self.received_z16_bricks.iter().filter(|&&b| b).count()
    }

    pub fn take_dirty_z16_bricks(&mut self) -> Vec<(usize, usize)> {
        let out = std::mem::take(&mut self.dirty_z16_bricks);
        for &(bx, by) in &out {
            let bi = by * self.chunks_x + bx;
            if bi < self.dirty_z16_mask.len() {
                self.dirty_z16_mask[bi] = false;
            }
        }
        out
    }

    pub fn take_dirty_z16_subbricks(&mut self) -> Vec<(usize, usize, usize)> {
        let out = std::mem::take(&mut self.dirty_z16_subbricks);
        let cz = chunks_for_dim(self.depth);
        for &(bx, by, bz) in &out {
            let sbi = (by * self.chunks_x + bx) * cz + bz;
            if sbi < self.dirty_z16_subbrick_mask.len() {
                self.dirty_z16_subbrick_mask[sbi] = false;
            }
        }
        out
    }

    pub fn apply_aplus_chunk(
        &mut self,
        raw_len: u64,
        field_tick: u64,
        layer: u8,
        cx: u8,
        cy: u8,
        lz4_cells: &[u8],
    ) {
        let Some(raw) = decompress_aplus_cells(lz4_cells) else {
            return;
        };
        if raw.len() != CHUNK_SIZE * CHUNK_SIZE * 4 {
            return;
        }
        let l_u = layer as usize;
        let cx_u = cx as usize;
        let cy_u = cy as usize;
        if l_u >= self.depth || cx_u >= self.chunks_x || cy_u >= self.chunks_y {
            return;
        }
        self.bytes_rx += raw_len;
        self.chunks_rx += 1;
        self.latest_field_tick = self.latest_field_tick.max(field_tick);

        let x0 = cx_u * CHUNK_SIZE;
        let y0 = cy_u * CHUNK_SIZE;
        let w = self.width;
        let h = self.height;
        let grid = &mut self.aplus_layers[l_u];
        for dy in 0..CHUNK_SIZE {
            let gy = (y0 + dy).min(h - 1);
            for dx in 0..CHUNK_SIZE {
                let gx = (x0 + dx).min(w - 1);
                let src = (dy * CHUNK_SIZE + dx) * 4;
                grid[gy * w + gx] = [raw[src], raw[src + 1], raw[src + 2], raw[src + 3]];
            }
        }
        let ci = cy_u * self.chunks_x + cx_u;
        if l_u == self.surf_layer() {
            self.received_surf_chunks[ci] = true;
        }
        self.dirty_chunks[l_u][ci] = true;
    }

    pub fn apply_brick3d(&mut self, raw_len: u64, brick: &Brick3DPayload) {
        let d = brick.depth as usize;
        let bx = brick.bx as usize;
        let by = brick.by as usize;
        if d == 0 || bx >= self.chunks_x || by >= self.chunks_y {
            return;
        }
        if d != self.depth {
            if (16..=64).contains(&d) {
                self.resize_if_needed(self.width, self.height, d);
            } else {
                return;
            }
        }
        let expected_brick_bytes = CHUNK_SIZE * CHUNK_SIZE * d * 4;
        let Some((dm_raw, at_raw)) = decode_brick3d_lz4(brick) else {
            return;
        };
        if dm_raw.len() != expected_brick_bytes || at_raw.len() != expected_brick_bytes {
            return;
        }

        let w = self.width;
        let h = self.height;
        let total_bytes = w * h * d * 4;
        if self.z16_density_mat.len() != total_bytes {
            self.z16_density_mat = vec![0u8; total_bytes];
            self.z16_albedo_thermal = vec![0u8; total_bytes];
        }

        let x0 = bx * CHUNK_SIZE;
        let y0 = by * CHUNK_SIZE;
        let copy_w = CHUNK_SIZE.min(w.saturating_sub(x0));
        let copy_h = CHUNK_SIZE.min(h.saturating_sub(y0));
        let row_bytes = copy_w * 4;

        // Transpose Z-major [z][dy][dx][4] -> GPU (X=width, Y=depth, Z=height): [iz=gy][iy=z][ix=gx][4]
        for z in 0..d {
            for dy in 0..copy_h {
                let gy = y0 + dy;
                let src = ((z * CHUNK_SIZE + dy) * CHUNK_SIZE) * 4;
                let dst = ((gy * d + z) * w + x0) * 4;
                self.z16_density_mat[dst..dst + row_bytes]
                    .copy_from_slice(&dm_raw[src..src + row_bytes]);
                self.z16_albedo_thermal[dst..dst + row_bytes]
                    .copy_from_slice(&at_raw[src..src + row_bytes]);
            }
        }

        self.bytes_rx += raw_len;
        self.bricks_3d_rx += 1;
        self.has_z16_bricks = true;

        let bi = by * self.chunks_x + bx;
        if bi < self.received_z16_bricks.len() {
            self.received_z16_bricks[bi] = true;
        }
        if bi < self.dirty_z16_mask.len() && !self.dirty_z16_mask[bi] {
            self.dirty_z16_mask[bi] = true;
            self.dirty_z16_bricks.push((bx, by));
        }
    }

    pub fn apply_brick3d_delta(
        &mut self,
        raw_len: u64,
        bx: u16,
        by: u16,
        depth: u16,
        tick: u64,
        lz4_deltas: &[u8],
    ) {
        let d = depth as usize;
        let bx_u = bx as usize;
        let by_u = by as usize;
        if d == 0 || bx_u >= self.chunks_x || by_u >= self.chunks_y {
            return;
        }
        if d != self.depth {
            if (16..=64).contains(&d) {
                self.resize_if_needed(self.width, self.height, d);
            } else {
                return;
            }
        }
        let Ok(patches) = decode_brick3d_delta_voxels(lz4_deltas) else {
            return;
        };

        let w = self.width;
        let h = self.height;
        let total_bytes = w * h * d * 4;
        if self.z16_density_mat.len() != total_bytes {
            self.z16_density_mat = vec![0u8; total_bytes];
            self.z16_albedo_thermal = vec![0u8; total_bytes];
        }

        let x0 = bx_u * CHUNK_SIZE;
        let y0 = by_u * CHUNK_SIZE;
        let total_brick_voxels = CHUNK_SIZE * CHUNK_SIZE * d;
        for (idx, dm, at) in patches {
            let v = idx as usize;
            if v >= total_brick_voxels {
                continue;
            }
            let z = v / (CHUNK_SIZE * CHUNK_SIZE);
            let rem = v % (CHUNK_SIZE * CHUNK_SIZE);
            let dy = rem / CHUNK_SIZE;
            let dx = rem % CHUNK_SIZE;
            let gx = x0 + dx;
            let gy = y0 + dy;
            if gx >= w || gy >= h || z >= d {
                continue;
            }
            // GPU (X=width, Y=depth, Z=height): [iz=gy][iy=z][ix=gx][4]
            let dst = ((gy * d + z) * w + gx) * 4;
            self.z16_density_mat[dst..dst + 4].copy_from_slice(&dm);
            self.z16_albedo_thermal[dst..dst + 4].copy_from_slice(&at);
        }

        self.bytes_rx += raw_len;
        self.bricks_3d_rx += 1;
        self.bricks_3d_delta_rx += 1;
        self.latest_field_tick = self.latest_field_tick.max(tick);
        self.has_z16_bricks = true;

        let bi = by_u * self.chunks_x + bx_u;
        if bi < self.received_z16_bricks.len() {
            self.received_z16_bricks[bi] = true;
        }
        if bi < self.dirty_z16_mask.len() && !self.dirty_z16_mask[bi] {
            self.dirty_z16_mask[bi] = true;
            self.dirty_z16_bricks.push((bx_u, by_u));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_subbrick3d(
        &mut self,
        raw_len: u64,
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_density_mat: &[u8],
        lz4_albedo_thermal: &[u8],
    ) {
        let td = total_depth as usize;
        let sd = sub_depth as usize;
        let bx_u = bx as usize;
        let by_u = by as usize;
        let bz_u = bz as usize;
        if td == 0 || sd == 0 || bx_u >= self.chunks_x || by_u >= self.chunks_y {
            return;
        }
        if td != self.depth {
            if (16..=64).contains(&td) {
                self.resize_if_needed(self.width, self.height, td);
            } else {
                return;
            }
        }
        let z0 = bz_u * CHUNK_SIZE;
        if z0 >= td {
            return;
        }
        let expected_bytes = CHUNK_SIZE * CHUNK_SIZE * sd * 4;
        let Some((dm_raw, at_raw)) = decode_subbrick3d_lz4(lz4_density_mat, lz4_albedo_thermal) else {
            return;
        };
        if dm_raw.len() != expected_bytes || at_raw.len() != expected_bytes {
            return;
        }

        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let total_bytes = w * h * d * 4;
        if self.z16_density_mat.len() != total_bytes {
            self.z16_density_mat = vec![0u8; total_bytes];
            self.z16_albedo_thermal = vec![0u8; total_bytes];
        }

        let x0 = bx_u * CHUNK_SIZE;
        let y0 = by_u * CHUNK_SIZE;
        let copy_w = CHUNK_SIZE.min(w.saturating_sub(x0));
        let copy_h = CHUNK_SIZE.min(h.saturating_sub(y0));
        let copy_d = sd.min(d.saturating_sub(z0));
        let row_bytes = copy_w * 4;

        for dz in 0..copy_d {
            let gz = z0 + dz;
            for dy in 0..copy_h {
                let gy = y0 + dy;
                let src = ((dz * CHUNK_SIZE + dy) * CHUNK_SIZE) * 4;
                let dst = ((gy * d + gz) * w + x0) * 4;
                self.z16_density_mat[dst..dst + row_bytes]
                    .copy_from_slice(&dm_raw[src..src + row_bytes]);
                self.z16_albedo_thermal[dst..dst + row_bytes]
                    .copy_from_slice(&at_raw[src..src + row_bytes]);
            }
        }

        self.bytes_rx += raw_len;
        self.bricks_3d_rx += 1;
        self.latest_field_tick = self.latest_field_tick.max(tick);
        self.has_z16_bricks = true;

        let bi = by_u * self.chunks_x + bx_u;
        if bi < self.received_z16_bricks.len() {
            self.received_z16_bricks[bi] = true;
        }
        let cz = chunks_for_dim(d);
        let sbi = bi * cz + bz_u;
        if sbi < self.dirty_z16_subbrick_mask.len() && !self.dirty_z16_subbrick_mask[sbi] {
            self.dirty_z16_subbrick_mask[sbi] = true;
            self.dirty_z16_subbricks.push((bx_u, by_u, bz_u));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_subbrick3d_delta(
        &mut self,
        raw_len: u64,
        bx: u16,
        by: u16,
        bz: u16,
        total_depth: u16,
        sub_depth: u8,
        tick: u64,
        lz4_deltas: &[u8],
    ) {
        let td = total_depth as usize;
        let sd = sub_depth as usize;
        let bx_u = bx as usize;
        let by_u = by as usize;
        let bz_u = bz as usize;
        if td == 0 || sd == 0 || bx_u >= self.chunks_x || by_u >= self.chunks_y {
            return;
        }
        if td != self.depth {
            if (16..=64).contains(&td) {
                self.resize_if_needed(self.width, self.height, td);
            } else {
                return;
            }
        }
        let z0 = bz_u * CHUNK_SIZE;
        if z0 >= td {
            return;
        }
        let Ok(patches) = decode_brick3d_delta_voxels(lz4_deltas) else {
            return;
        };

        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let total_bytes = w * h * d * 4;
        if self.z16_density_mat.len() != total_bytes {
            self.z16_density_mat = vec![0u8; total_bytes];
            self.z16_albedo_thermal = vec![0u8; total_bytes];
        }

        let x0 = bx_u * CHUNK_SIZE;
        let y0 = by_u * CHUNK_SIZE;
        let total_sub_voxels = CHUNK_SIZE * CHUNK_SIZE * sd;
        for (idx, dm, at) in patches {
            let v = idx as usize;
            if v >= total_sub_voxels {
                continue;
            }
            let dz = v / (CHUNK_SIZE * CHUNK_SIZE);
            let rem = v % (CHUNK_SIZE * CHUNK_SIZE);
            let dy = rem / CHUNK_SIZE;
            let dx = rem % CHUNK_SIZE;
            let gx = x0 + dx;
            let gy = y0 + dy;
            let gz = z0 + dz;
            if gx >= w || gy >= h || gz >= d {
                continue;
            }
            let dst = ((gy * d + gz) * w + gx) * 4;
            self.z16_density_mat[dst..dst + 4].copy_from_slice(&dm);
            self.z16_albedo_thermal[dst..dst + 4].copy_from_slice(&at);
        }

        self.bytes_rx += raw_len;
        self.bricks_3d_rx += 1;
        self.bricks_3d_delta_rx += 1;
        self.latest_field_tick = self.latest_field_tick.max(tick);
        self.has_z16_bricks = true;

        let bi = by_u * self.chunks_x + bx_u;
        if bi < self.received_z16_bricks.len() {
            self.received_z16_bricks[bi] = true;
        }
        let cz = chunks_for_dim(d);
        let sbi = bi * cz + bz_u;
        if sbi < self.dirty_z16_subbrick_mask.len() && !self.dirty_z16_subbrick_mask[sbi] {
            self.dirty_z16_subbrick_mask[sbi] = true;
            self.dirty_z16_subbricks.push((bx_u, by_u, bz_u));
        }
    }

    pub fn take_dirty_list(&mut self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        for (l, row) in self.dirty_chunks.iter_mut().enumerate() {
            for (ci, d) in row.iter_mut().enumerate() {
                if *d {
                    *d = false;
                    out.push((l, ci));
                }
            }
        }
        out
    }

    pub fn take_dirty_budget(&mut self, max_chunks: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(max_chunks.min(64));
        for (l, row) in self.dirty_chunks.iter_mut().enumerate() {
            for (ci, d) in row.iter_mut().enumerate() {
                if *d {
                    *d = false;
                    out.push((l, ci));
                    if out.len() >= max_chunks {
                        return out;
                    }
                }
            }
        }
        out
    }

    pub fn client_time_ms(&self) -> u32 {
        self.start_instant.elapsed().as_millis() as u32
    }
}

pub struct QuicClientHandle {
    pub state: Arc<Mutex<LiveWorldState>>,
    pub conn: Connection,
    rt_handle: tokio::runtime::Handle,
    _endpoint: Endpoint,
}

impl QuicClientHandle {
    pub async fn connect(
        addr: SocketAddr,
        name: &str,
        team: u8,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = rustls::crypto::ring::default_provider().install_default();

        let crypto = rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
            .with_no_client_auth();

        let quic_crypto = quinn::crypto::rustls::QuicClientConfig::try_from(crypto)?;
        let mut client_cfg = ClientConfig::new(Arc::new(quic_crypto));
        let mut transport = quinn::TransportConfig::default();
        transport.max_idle_timeout(Some(Duration::from_secs(30).try_into()?));
        transport.keep_alive_interval(Some(Duration::from_secs(5)));
        transport.datagram_receive_buffer_size(Some(12 * 1024 * 1024));
        transport.datagram_send_buffer_size(4 * 1024 * 1024);
        client_cfg.transport_config(Arc::new(transport));

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse()?)?;
        endpoint.set_default_client_config(client_cfg);

        let conn = endpoint.connect(addr, "localhost")?.await?;
        let state = Arc::new(Mutex::new(LiveWorldState::new(GRID_W, GRID_H, GRID_D)));

        Self::send_reliable_on(&conn, &ClientReliable::Join { name: name.to_string(), team }).await?;
        Self::send_reliable_on(&conn, &ClientReliable::RequestFullSync).await?;

        let apply_snapshot = |st: &mut LiveWorldState,
                              raw_len: u64,
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
                              bot_brains: Vec<BotBrainSnap>,
                              kill_feed: Vec<KillFeedEntry>,
                              speech_bubbles: Vec<SpeechBubbleSnap>,
                              actuator_pulses: Vec<ActuatorPulseSnap>,
                              match_phase: u8,
                              match_time_left_s: u16,
                              mvp_name: String,
                              mvp_score: u32,
                              time_domes: Vec<TimeDomeSnap>,
                              world_law_preset: u8| {
            st.world_law_preset = world_law_preset;
            CURRENT_WORLD_LAW_PRESET.store(world_law_preset, Ordering::Relaxed);
            st.resize_if_needed(
                telemetry.grid_w as usize,
                telemetry.grid_h as usize,
                telemetry.grid_d as usize,
            );
            st.bytes_rx += raw_len;
            st.snapshots_rx += 1;
            st.your_player_id = your_player_id;
            if echo_client_time_ms > 0 {
                let now_ms = st.client_time_ms();
                let rtt = now_ms.saturating_sub(echo_client_time_ms) as f32;
                if rtt < 2000.0 {
                    st.rtt_ms = if st.rtt_ms <= 0.1 { rtt } else { st.rtt_ms * 0.85 + rtt * 0.15 };
                }
            }
            st.entities = Some(LiveEntityState {
                sim_tick,
                match_time_secs,
                your_player_id,
                echo_client_time_ms,
                relic_x,
                relic_y,
                players,
                projectiles,
                bodies,
                shrines,
                events,
                telemetry,
                bot_brains,
                kill_feed,
                speech_bubbles,
                actuator_pulses,
                match_phase,
                match_time_left_s,
                mvp_name,
                mvp_score,
                time_domes,
            });
        };

        // Reliable uni-stream listener (Welcome + WAN MTU-fallback EntitySnapshots, APlusChunks, & VolumetricBrick3D!)
        let st_rel = state.clone();
        let conn_rel = conn.clone();
        tokio::spawn(async move {
            while let Ok(mut recv) = conn_rel.accept_uni().await {
                if let Ok(bytes) = recv.read_to_end(524_288).await {
                    if bytes.first() == Some(&0xFE) {
                        if let Some(pkt) = decode_server_datagram(&bytes[1..]) {
                            let raw_len = bytes.len() as u64;
                            match pkt {
                                ServerDatagram::EntitySnapshot {
                                    sim_tick,
                                    match_time_secs,
                                    your_player_id,
                                    echo_client_time_ms,
                                    relic_x,
                                    relic_y,
                                    players,
                                    projectiles,
                                    bodies,
                                    shrines,
                                    events,
                                    telemetry,
                                    bot_brains,
                                    kill_feed,
                                    speech_bubbles,
                                    actuator_pulses,
                                    match_phase,
                                    match_time_left_s,
                                    mvp_name,
                                    mvp_score,
                                    time_domes,
                                    world_law_preset,
                                } => {
                                    let mut st = st_rel.lock().unwrap();
                                    apply_snapshot(
                                        &mut st,
                                        raw_len,
                                        sim_tick,
                                        match_time_secs,
                                        your_player_id,
                                        echo_client_time_ms,
                                        relic_x,
                                        relic_y,
                                        players,
                                        projectiles,
                                        bodies,
                                        shrines,
                                        events,
                                        telemetry,
                                        bot_brains,
                                        kill_feed,
                                        speech_bubbles,
                                        actuator_pulses,
                                        match_phase,
                                        match_time_left_s,
                                        mvp_name,
                                        mvp_score,
                                        time_domes,
                                        world_law_preset,
                                    );
                                }
                                ServerDatagram::APlusChunk {
                                    field_tick,
                                    layer,
                                    chunk_id: _,
                                    cx,
                                    cy,
                                    lz4_cells,
                                } => {
                                    let mut st = st_rel.lock().unwrap();
                                    st.apply_aplus_chunk(raw_len, field_tick, layer, cx, cy, &lz4_cells);
                                }
                                ServerDatagram::VolumetricBrick3D(brick) => {
                                    let mut st = st_rel.lock().unwrap();
                                    st.apply_brick3d(raw_len, &brick);
                                }
                                ServerDatagram::VolumetricBrick3DDelta {
                                    bx,
                                    by,
                                    depth,
                                    tick,
                                    lz4_deltas,
                                } => {
                                    let mut st = st_rel.lock().unwrap();
                                    st.apply_brick3d_delta(raw_len, bx, by, depth, tick, &lz4_deltas);
                                }
                                ServerDatagram::VolumetricSubBrick3D {
                                    bx,
                                    by,
                                    bz,
                                    total_depth,
                                    sub_depth,
                                    tick,
                                    lz4_density_mat,
                                    lz4_albedo_thermal,
                                } => {
                                    let mut st = st_rel.lock().unwrap();
                                    st.apply_subbrick3d(
                                        raw_len,
                                        bx,
                                        by,
                                        bz,
                                        total_depth,
                                        sub_depth,
                                        tick,
                                        &lz4_density_mat,
                                        &lz4_albedo_thermal,
                                    );
                                }
                                ServerDatagram::VolumetricSubBrick3DDelta {
                                    bx,
                                    by,
                                    bz,
                                    total_depth,
                                    sub_depth,
                                    tick,
                                    lz4_deltas,
                                } => {
                                    let mut st = st_rel.lock().unwrap();
                                    st.apply_subbrick3d_delta(
                                        raw_len,
                                        bx,
                                        by,
                                        bz,
                                        total_depth,
                                        sub_depth,
                                        tick,
                                        &lz4_deltas,
                                    );
                                }
                            }
                        }
                    } else if let Some(ServerReliable::Welcome {
                        your_player_id,
                        width,
                        height,
                        depth,
                    }) = decode_msg::<ServerReliable>(&bytes)
                    {
                        let mut st = st_rel.lock().unwrap();
                        st.your_player_id = your_player_id;
                        st.resize_if_needed(width as usize, height as usize, depth as usize);
                    }
                }
            }
        });

        // Unreliable datagram reader (3D layered APlusChunk + VolumetricBrick3D + EntitySnapshot)
        let st_dg = state.clone();
        let conn_dg = conn.clone();
        tokio::spawn(async move {
            while let Ok(bytes) = conn_dg.read_datagram().await {
                let raw_len = bytes.len() as u64;
                let Some(pkt) = decode_server_datagram(&bytes) else {
                    continue;
                };
                match pkt {
                    ServerDatagram::APlusChunk {
                        field_tick,
                        layer,
                        chunk_id: _,
                        cx,
                        cy,
                        lz4_cells,
                    } => {
                        let mut st = st_dg.lock().unwrap();
                        st.apply_aplus_chunk(raw_len, field_tick, layer, cx, cy, &lz4_cells);
                    }
                    ServerDatagram::EntitySnapshot {
                        sim_tick,
                        match_time_secs,
                        your_player_id,
                        echo_client_time_ms,
                        relic_x,
                        relic_y,
                        players,
                        projectiles,
                        bodies,
                        shrines,
                        events,
                        telemetry,
                        bot_brains,
                        kill_feed,
                        speech_bubbles,
                        actuator_pulses,
                        match_phase,
                        match_time_left_s,
                        mvp_name,
                        mvp_score,
                        time_domes,
                        world_law_preset,
                    } => {
                        let mut st = st_dg.lock().unwrap();
                        apply_snapshot(
                            &mut st,
                            raw_len,
                            sim_tick,
                            match_time_secs,
                            your_player_id,
                            echo_client_time_ms,
                            relic_x,
                            relic_y,
                            players,
                            projectiles,
                            bodies,
                            shrines,
                            events,
                            telemetry,
                            bot_brains,
                            kill_feed,
                            speech_bubbles,
                            actuator_pulses,
                            match_phase,
                            match_time_left_s,
                            mvp_name,
                            mvp_score,
                            time_domes,
                            world_law_preset,
                        );
                    }
                    ServerDatagram::VolumetricBrick3D(brick) => {
                        let mut st = st_dg.lock().unwrap();
                        st.apply_brick3d(raw_len, &brick);
                    }
                    ServerDatagram::VolumetricBrick3DDelta {
                        bx,
                        by,
                        depth,
                        tick,
                        lz4_deltas,
                    } => {
                        let mut st = st_dg.lock().unwrap();
                        st.apply_brick3d_delta(raw_len, bx, by, depth, tick, &lz4_deltas);
                    }
                    ServerDatagram::VolumetricSubBrick3D {
                        bx,
                        by,
                        bz,
                        total_depth,
                        sub_depth,
                        tick,
                        lz4_density_mat,
                        lz4_albedo_thermal,
                    } => {
                        let mut st = st_dg.lock().unwrap();
                        st.apply_subbrick3d(
                            raw_len,
                            bx,
                            by,
                            bz,
                            total_depth,
                            sub_depth,
                            tick,
                            &lz4_density_mat,
                            &lz4_albedo_thermal,
                        );
                    }
                    ServerDatagram::VolumetricSubBrick3DDelta {
                        bx,
                        by,
                        bz,
                        total_depth,
                        sub_depth,
                        tick,
                        lz4_deltas,
                    } => {
                        let mut st = st_dg.lock().unwrap();
                        st.apply_subbrick3d_delta(
                            raw_len,
                            bx,
                            by,
                            bz,
                            total_depth,
                            sub_depth,
                            tick,
                            &lz4_deltas,
                        );
                    }
                }
            }
        });

        Ok(Self {
            state,
            conn,
            rt_handle: tokio::runtime::Handle::current(),
            _endpoint: endpoint,
        })
    }

    pub async fn send_reliable_on(
        conn: &Connection,
        msg: &ClientReliable,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut send = conn.open_uni().await?;
        let raw = encode_msg(msg);
        send.write_all(&raw).await?;
        send.finish()?;
        Ok(())
    }

    pub fn send_reliable_detached(&self, msg: ClientReliable) {
        let conn = self.conn.clone();
        self.rt_handle.spawn(async move {
            let _ = Self::send_reliable_on(&conn, &msg).await;
        });
    }

    pub fn send_input(&self, inp: &ClientInputDatagram) {
        let raw = encode_msg(inp);
        let _ = self.conn.send_datagram(bytes::Bytes::from(raw));
    }

    pub async fn wait_for_sync(&self, min_frac: f32, max_wait: Duration) -> (usize, usize, bool) {
        let start = Instant::now();
        let mut last_req = Instant::now();
        loop {
            let (rc, tot, has_ent) = {
                let st = self.state.lock().unwrap();
                let rc = if st.depth >= 16 {
                    st.received_z16_count()
                } else {
                    st.received_surf_count()
                };
                (rc, st.total_surf_chunks(), st.entities.is_some())
            };
            let target = ((tot as f32) * min_frac).ceil() as usize;
            if rc >= target && has_ent {
                return (rc, tot, true);
            }
            if start.elapsed() >= max_wait {
                return (rc, tot, has_ent);
            }
            if last_req.elapsed() >= Duration::from_millis(900) && rc < target {
                let _ = Self::send_reliable_on(&self.conn, &ClientReliable::RequestFullSync).await;
                last_req = Instant::now();
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    }
}
