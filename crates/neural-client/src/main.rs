//! Thin QUIC Input & Display Harness + Client-Side Neural Frame Renderer (A+ Compositional).
//!
//! - Connects over **QUIC** (`quinn`) to `frostmarsh-server` (local `127.0.0.1:4433` or
//!   `spark-2b4e` GPU `100.116.218.113:4433`) — zero SpacetimeDB, zero TCP head-of-line blocking!
//! - **30 Hz Client-Side Movement Prediction & `ack_input_seq` Reconciliation** against local `A+`
//!   collision/hazard bits for snappy action-MMO feel even over Tailscale.
//! - **10 Hz Cached Neural World Renderer:**
//!   - Prefers **`renderer_v2.onnx`** (`aplus_state [1, 17, 128, 128] -> rgb_frame [1, 3, 256, 256]`)
//!     preserving full top-2 species mixtures, mass fractions, 8-bit temperature, flow vectors, and `(sin, cos)` shimmer.
//!   - Automatically falls back to `renderer_model.onnx` (v1) if `renderer_v2.onnx` is not present yet.
//! - **60 Hz Crisp Combat Overlay** (toggleable via `O`) composited pixel-exact on top.

mod cam3d;

use bytes::Bytes;
use cam3d::{
    GBUF_CHANNELS, H3D, OrbitCamera, W3D, apply_9step_tone_ramp, cell_height_norm,
    render_gbuffer_3d, shade_gbuffer_fallback,
};
use clap::Parser;
use frostmarsh_proto::*;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Scale, Window, WindowOptions};
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use quinn::{ClientConfig, Endpoint};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::mpsc;

const OUT_W: usize = 256;
const OUT_H: usize = 256;
const APLUS_CHANNELS: usize = 17;
const V1_CHANNELS: usize = 20;

#[derive(Parser, Debug)]
#[command(name = "neural-client", about = "Frostmarsh QUIC Thin Harness + Client Neural Renderer")]
struct Args {
    /// QUIC server address (local `127.0.0.1:4433` or Spark GPU `100.116.218.113:4433`)
    #[arg(long, default_value = "127.0.0.1:4433")]
    server: SocketAddr,
    #[arg(long, default_value = "models/renderer_v2.onnx")]
    renderer_v2: PathBuf,
    #[arg(long, default_value = "models/renderer_model.onnx")]
    renderer_v1: PathBuf,
    #[arg(long, default_value = "models/tier2_deferred_3d.onnx")]
    renderer_3d: PathBuf,
    /// Start directly in Tier 2 3D Free-Camera mode (also toggleable live with `V`)
    #[arg(long, default_value_t = false)]
    mode_3d: bool,
    /// Enable Approach 4's 9-step Cool-Indigo / Warm-Cream tone ramp post-pass (toggleable with `T`)
    #[arg(long, default_value_t = false)]
    tone_ramp: bool,
    #[arg(long, default_value = "Hero")]
    name: String,
    #[arg(long, default_value_t = 0)]
    team: u8,
    /// Optional headless mode: wait for live QUIC chunks, render one frame via ONNX, save PPM, exit.
    #[arg(long)]
    headless_ppm: Option<PathBuf>,
}

/// Dev POC verifier accepting the server's ephemeral self-signed QUIC certificate.
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
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }
    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Ok(HandshakeSignatureValid::assertion())
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

fn make_quic_client_config() -> ClientConfig {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    let mut cfg = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto).unwrap(),
    ));
    let mut transport = quinn::TransportConfig::default();
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    transport.datagram_receive_buffer_size(Some(2 * 1024 * 1024));
    cfg.transport_config(Arc::new(transport));
    cfg
}

// ---------------- Shared Client Net Mirror ----------------

struct NetMirror {
    connected: bool,
    my_player_id: u32,
    sim_tick: u64,
    field_tick: u64,
    chunks_received: u32,
    /// Full 128×128 A+ cell grid (4 bytes/cell)
    aplus_grid: Vec<u8>,
    chunk_seen: Vec<bool>,
    relic_x: f32,
    relic_y: f32,
    players: Vec<PlayerSnap>,
    projectiles: Vec<ProjectileSnap>,
    bodies: Vec<CellBodySnap>,
    telemetry: Option<TelemetrySnap>,
    rtt_ms_ema: f32,
}

impl NetMirror {
    fn new() -> Self {
        Self {
            connected: false,
            my_player_id: 0,
            sim_tick: 0,
            field_tick: 0,
            chunks_received: 0,
            aplus_grid: vec![0u8; GRID_W * GRID_H * 4],
            chunk_seen: vec![false; CHUNKS_X * CHUNKS_Y],
            relic_x: GRID_W as f32 * 0.78,
            relic_y: GRID_H as f32 * 0.5,
            players: Vec::new(),
            projectiles: Vec::new(),
            bodies: Vec::new(),
            telemetry: None,
            rtt_ms_ema: 0.0,
        }
    }

    fn apply_aplus_chunk(&mut self, cx: u8, cy: u8, field_tick: u64, raw_1024: &[u8]) {
        if raw_1024.len() != CHUNK_SIZE * CHUNK_SIZE * 4 {
            return;
        }
        let cid = (cy as usize) * CHUNKS_X + (cx as usize);
        if cid < self.chunk_seen.len() && !self.chunk_seen[cid] {
            self.chunk_seen[cid] = true;
            self.chunks_received += 1;
        }
        self.field_tick = self.field_tick.max(field_tick);
        let x0 = (cx as usize) * CHUNK_SIZE;
        let y0 = (cy as usize) * CHUNK_SIZE;
        for dy in 0..CHUNK_SIZE {
            let gy = y0 + dy;
            if gy >= GRID_H {
                continue;
            }
            for dx in 0..CHUNK_SIZE {
                let gx = x0 + dx;
                if gx >= GRID_W {
                    continue;
                }
                let src = (dy * CHUNK_SIZE + dx) * 4;
                let dst = (gy * GRID_W + gx) * 4;
                self.aplus_grid[dst..dst + 4].copy_from_slice(&raw_1024[src..src + 4]);
            }
        }
    }

    #[inline]
    fn sample_hazard(&self, x: f32, y: f32) -> u8 {
        let xi = (x.floor() as i32).clamp(0, (GRID_W - 1) as i32) as usize;
        let yi = (y.floor() as i32).clamp(0, (GRID_H - 1) as i32) as usize;
        self.aplus_grid[(yi * GRID_W + xi) * 4 + 3]
    }
}

// ---------------- Client-Side Movement Predictor ----------------

struct LocalPredictor {
    x: f32,
    y: f32,
    initialized: bool,
    pending: VecDeque<ClientInputDatagram>,
}

impl LocalPredictor {
    fn new() -> Self {
        Self {
            x: 18.0,
            y: 64.0,
            initialized: false,
            pending: VecDeque::with_capacity(64),
        }
    }

    fn push_input(&mut self, inp: ClientInputDatagram, mirror: &NetMirror) {
        self.step_one(&inp, mirror);
        if self.pending.len() >= 64 {
            self.pending.pop_front();
        }
        self.pending.push_back(inp);
    }

    fn reconcile(&mut self, me: &PlayerSnap, mirror: &NetMirror) {
        if !self.initialized {
            self.x = me.x;
            self.y = me.y;
            self.initialized = true;
            return;
        }
        while let Some(front) = self.pending.front() {
            if front.seq <= me.ack_input_seq {
                self.pending.pop_front();
            } else {
                break;
            }
        }
        let mut rx = me.x;
        let mut ry = me.y;
        let mut tmp = LocalPredictor {
            x: rx,
            y: ry,
            initialized: true,
            pending: VecDeque::new(),
        };
        for inp in &self.pending {
            tmp.step_one(inp, mirror);
        }
        rx = tmp.x;
        ry = tmp.y;
        let err = ((rx - self.x).powi(2) + (ry - self.y).powi(2)).sqrt();
        if err > 4.0 {
            self.x = rx;
            self.y = ry;
        } else {
            // Smooth micro-reconciliation
            self.x += (rx - self.x) * 0.35;
            self.y += (ry - self.y) * 0.35;
        }
    }

    fn step_one(&mut self, inp: &ClientInputDatagram, mirror: &NetMirror) {
        let dt = 1.0 / 30.0f32;
        let mlen = (inp.move_x * inp.move_x + inp.move_y * inp.move_y).sqrt();
        let (dx, dy) = if mlen > 0.01 {
            (inp.move_x / mlen, inp.move_y / mlen)
        } else {
            (0.0, 0.0)
        };
        let hz = mirror.sample_hazard(self.x, self.y);
        let mut speed = 15.0f32;
        if (hz & (1 << 4)) != 0 {
            speed *= 0.60;
        }
        if (hz & (1 << 3)) != 0 {
            speed *= 0.72;
        }
        let nx = (self.x + dx * speed * dt).clamp(2.5, (GRID_W as f32) - 2.5);
        let ny = (self.y + dy * speed * dt).clamp(2.5, (GRID_H as f32) - 2.5);
        let hzn = mirror.sample_hazard(nx, ny);
        if (hzn & (1 << 0)) != 0 && (hzn & (1 << 1)) == 0 {
            self.x = nx;
            self.y = ny;
        }
    }
}

// ---------------- Neural Renderers (v2 A+ Compositional & v1 Fallback) ----------------

enum RendererKind {
    V2APlus,
    V1Legacy,
}

struct NeuralRenderer {
    kind: RendererKind,
    session: Session,
    path: PathBuf,
    mtime: Option<SystemTime>,
}

impl NeuralRenderer {
    fn try_reload(env: &Environment, v2_path: &Path, v1_path: &Path, cur: &mut Option<Self>) {
        let (target_path, kind) = if v2_path.exists() {
            (v2_path, RendererKind::V2APlus)
        } else if v1_path.exists() {
            (v1_path, RendererKind::V1Legacy)
        } else {
            return;
        };
        let disk_mtime = std::fs::metadata(target_path)
            .ok()
            .and_then(|m| m.modified().ok());
        let needs = match cur {
            None => true,
            Some(r) => r.path != target_path || (disk_mtime.is_some() && disk_mtime != r.mtime),
        };
        if needs {
            let res: ort::Result<Session> = (|| {
                Ok(Session::builder(env)?
                    .with_intra_threads(4)?
                    .commit_from_file(target_path)?)
            })();
            match res {
                Ok(session) => {
                    log::info!("Loaded Client Neural Renderer ({:?}) from {}", match kind {
                        RendererKind::V2APlus => "v2-A+",
                        RendererKind::V1Legacy => "v1-legacy",
                    }, target_path.display());
                    *cur = Some(Self {
                        kind,
                        session,
                        path: target_path.to_path_buf(),
                        mtime: disk_mtime,
                    });
                }
                Err(e) => log::warn!("Renderer load failed {}: {}", target_path.display(), e),
            }
        }
    }

    fn render(
        &mut self,
        mirror: &NetMirror,
        time_sec: f32,
        fb: &mut [u32],
    ) -> Result<&'static str, Box<dyn std::error::Error>> {
        let plane = OUT_W * OUT_H;
        let (rgb_vec, tag) = match self.kind {
            RendererKind::V2APlus => {
                let tensor = pack_aplus_tensor(&mirror.aplus_grid, time_sec);
                let inp = Tensor::from_array(([1usize, APLUS_CHANNELS, GRID_H, GRID_W], tensor))?;
                let out = self.session.run(ort::inputs!["aplus_state" => inp])?;
                let (_s, slice) = out["rgb_frame"].try_extract_tensor::<f32>()?;
                (slice.to_vec(), "NN-V2-A+")
            }
            RendererKind::V1Legacy => {
                let tensor = pack_v1_from_aplus(&mirror.aplus_grid, mirror, time_sec);
                let inp = Tensor::from_array(([1usize, V1_CHANNELS, GRID_H, GRID_W], tensor))?;
                let out = self.session.run(ort::inputs!["frame_state" => inp])?;
                let (_s, slice) = out["rgb_frame"].try_extract_tensor::<f32>()?;
                (slice.to_vec(), "NN-V1")
            }
        };
        if rgb_vec.len() != 3 * plane {
            return Err(format!("Bad rgb len {}", rgb_vec.len()).into());
        }
        for i in 0..plane {
            let r = (rgb_vec[i].clamp(0.0, 1.0) * 255.0) as u32;
            let g = (rgb_vec[plane + i].clamp(0.0, 1.0) * 255.0) as u32;
            let b = (rgb_vec[2 * plane + i].clamp(0.0, 1.0) * 255.0) as u32;
            fb[i] = (r << 16) | (g << 8) | b;
        }
        Ok(tag)
    }
}

/// Unpack 4 B/cell `A+` wire grid into `[1, 17, 128, 128]` compositional neural renderer input.
fn pack_aplus_tensor(aplus: &[u8], time_sec: f32) -> Vec<f32> {
    let plane = GRID_W * GRID_H;
    let mut t = vec![0.0f32; APLUS_CHANNELS * plane];
    let t_sin = (time_sec * 2.5).sin() * 0.5 + 0.5;
    let t_cos = (time_sec * 2.5).cos() * 0.5 + 0.5;

    for idx in 0..plane {
        let b0 = aplus[idx * 4];
        let b1 = aplus[idx * 4 + 1];
        let b2 = aplus[idx * 4 + 2];
        let b3 = aplus[idx * 4 + 3];

        let s1 = ((b0 >> 4) & 0x0f) as usize;
        let s2 = (b0 & 0x0f) as usize;
        let frac = ((b1 >> 4) & 0x0f) as f32 / 15.0;
        let flow_dir = b1 & 0x0f;

        if s1 < 10 {
            t[s1 * plane + idx] += 1.0 - frac;
        }
        if s2 < 10 {
            t[s2 * plane + idx] += frac;
        }

        t[10 * plane + idx] = b2 as f32 / 255.0;
        t[11 * plane + idx] = if (b3 & (1 << 2)) != 0 { 1.0 } else { 0.0 };

        if flow_dir > 0 {
            let fast = flow_dir >= 9;
            let oct = if fast {
                (flow_dir - 9).min(7)
            } else {
                (flow_dir - 1).min(7)
            };
            let ang = (oct as f32) * (std::f32::consts::TAU / 8.0);
            let spd = if fast { 1.0 } else { 0.5 };
            t[12 * plane + idx] = ang.cos() * spd;
            t[13 * plane + idx] = ang.sin() * spd;
        }

        t[14 * plane + idx] = ((b3 >> 5) & 0x07) as f32 / 7.0;
        t[15 * plane + idx] = t_sin;
        t[16 * plane + idx] = t_cos;
    }
    t
}

/// Bridge `A+` grid to legacy `v1` 20-channel renderer until `renderer_v2.onnx` is present.
fn pack_v1_from_aplus(aplus: &[u8], mirror: &NetMirror, time_sec: f32) -> Vec<f32> {
    let plane = GRID_W * GRID_H;
    let mut t = vec![0.0f32; V1_CHANNELS * plane];
    let phase = (time_sec * 3.0).sin() * 0.5 + 0.5;
    for idx in 0..plane {
        let b0 = aplus[idx * 4];
        let b2 = aplus[idx * 4 + 2];
        let b3 = aplus[idx * 4 + 3];
        let s1 = ((b0 >> 4) & 0x0f) as usize;
        let s2 = (b0 & 0x0f) as usize;
        let burning = (b3 & (1 << 2)) != 0;

        let cls = if burning {
            12
        } else if (s1 == 0 && s2 == 3) || (s1 == 3 && s2 == 0) {
            5 // mud
        } else {
            match s1 {
                0 => 1,  // water
                1 => 2,  // ice
                2 => 3,  // steam
                3 => 4,  // soil
                4 => 6,  // rock
                5 => 7,  // wood
                6 => 8,  // ash
                7 => 9,  // smoke
                8 => 10, // oil
                9 => 11, // plant
                _ => 0,
            }
        };
        if (1..=12).contains(&cls) {
            t[(cls - 1) * plane + idx] = 1.0;
        }
        t[12 * plane + idx] = b2 as f32 / 255.0;
        t[13 * plane + idx] = 0.7;
        t[14 * plane + idx] = ((b3 >> 5) & 0x07) as f32 / 7.0;
        t[19 * plane + idx] = phase;
    }
    // Add subtle projectile & relic light splats
    for pr in &mirror.projectiles {
        let val = match pr.element {
            0 => 1.0,
            1 => -1.0,
            _ => 0.45,
        };
        splat(&mut t, 17, pr.x, pr.y, 1.5, val);
    }
    splat(&mut t, 18, mirror.relic_x, mirror.relic_y, 2.0, 1.0);
    t
}

fn splat(t: &mut [f32], ch: usize, cx: f32, cy: f32, r: f32, val: f32) {
    let plane = GRID_W * GRID_H;
    let base = ch * plane;
    let ri = r.ceil() as i32;
    let x0 = cx.floor() as i32;
    let y0 = cy.floor() as i32;
    let r2 = (r * r).max(0.25);
    for dy in -ri..=ri {
        let y = y0 + dy;
        if y < 0 || y >= GRID_H as i32 {
            continue;
        }
        for dx in -ri..=ri {
            let x = x0 + dx;
            if x < 0 || x >= GRID_W as i32 {
                continue;
            }
            let d2 = ((x as f32 + 0.5) - cx).powi(2) + ((y as f32 + 0.5) - cy).powi(2);
            if d2 <= r2 {
                let idx = base + (y as usize) * GRID_W + (x as usize);
                let v = val * (1.0 - (d2 / r2) * 0.5);
                if val >= 0.0 {
                    t[idx] = t[idx].max(v);
                } else {
                    t[idx] = t[idx].min(v);
                }
            }
        }
    }
}

fn render_fallback(aplus: &[u8], fb: &mut [u32]) {
    let sp_rgb: [[f32; 3]; 10] = [
        [28.0, 110.0, 200.0],  // water
        [185.0, 238.0, 255.0], // ice
        [210.0, 224.0, 235.0], // steam
        [96.0, 76.0, 54.0],    // soil
        [88.0, 92.0, 100.0],   // rock
        [62.0, 98.0, 46.0],    // wood
        [64.0, 64.0, 68.0],    // ash
        [44.0, 46.0, 52.0],    // smoke
        [42.0, 24.0, 58.0],    // oil
        [66.0, 142.0, 58.0],   // plant
    ];
    for oy in 0..OUT_H {
        let gy = oy / 2;
        for ox in 0..OUT_W {
            let gx = ox / 2;
            let idx = gy * GRID_W + gx;
            let b0 = aplus[idx * 4];
            let b1 = aplus[idx * 4 + 1];
            let b3 = aplus[idx * 4 + 3];
            let s1 = ((b0 >> 4) & 0x0f) as usize;
            let s2 = (b0 & 0x0f) as usize;
            let frac = ((b1 >> 4) & 0x0f) as f32 / 15.0;
            let mut rgb = if s1 < 10 { sp_rgb[s1] } else { [24.0, 28.0, 34.0] };
            if s2 < 10 && frac > 0.0 {
                for c in 0..3 {
                    rgb[c] = rgb[c] * (1.0 - frac) + sp_rgb[s2][c] * frac;
                }
            }
            if (b3 & (1 << 2)) != 0 {
                rgb = [255.0, 130.0, 30.0];
            }
            fb[oy * OUT_W + ox] =
                ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | (rgb[2] as u32);
        }
    }
}

// ---------------- Crisp 60 Hz Combat Overlay ----------------

#[inline]
fn put_px(fb: &mut [u32], x: i32, y: i32, col: u32) {
    if x >= 0 && y >= 0 && (x as usize) < OUT_W && (y as usize) < OUT_H {
        fb[(y as usize) * OUT_W + (x as usize)] = col;
    }
}

fn draw_disk(fb: &mut [u32], cx: f32, cy: f32, r: f32, col: u32) {
    let ri = r.ceil() as i32;
    let x0 = cx.round() as i32;
    let y0 = cy.round() as i32;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if (dx * dx + dy * dy) as f32 <= r * r {
                put_px(fb, x0 + dx, y0 + dy, col);
            }
        }
    }
}

fn draw_ring(fb: &mut [u32], cx: f32, cy: f32, r: f32, col: u32) {
    let ri = r.ceil() as i32 + 1;
    let x0 = cx.round() as i32;
    let y0 = cy.round() as i32;
    let rin = (r - 1.1).max(0.0);
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d >= rin && d <= r + 0.5 {
                put_px(fb, x0 + dx, y0 + dy, col);
            }
        }
    }
}

fn draw_line(fb: &mut [u32], x0: f32, y0: f32, x1: f32, y1: f32, col: u32) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs()) * 2.0).ceil().max(1.0) as i32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        put_px(
            fb,
            (x0 + (x1 - x0) * t).round() as i32,
            (y0 + (y1 - y0) * t).round() as i32,
            col,
        );
    }
}

fn composite_combat(mirror: &NetMirror, pred: &LocalPredictor, fb: &mut [u32]) {
    let sc = (OUT_W as f32) / (GRID_W as f32);

    for b in &mirror.bodies {
        draw_disk(fb, b.x * sc, b.y * sc, b.radius * sc, 0xA8F0FF);
        draw_ring(fb, b.x * sc, b.y * sc, b.radius * sc, 0xFFFFFF);
    }
    draw_disk(fb, mirror.relic_x * sc, mirror.relic_y * sc, 3.2, 0xFFD830);
    draw_ring(fb, mirror.relic_x * sc, mirror.relic_y * sc, 5.2, 0xFFF2A0);

    for pr in &mirror.projectiles {
        let col = match pr.element {
            0 => 0xFF7020,
            1 => 0x45EEFF,
            _ => 0xAA40E0,
        };
        draw_disk(fb, pr.x * sc, pr.y * sc, 2.2, col);
    }

    for p in &mirror.players {
        let is_me = p.id == mirror.my_player_id && pred.initialized;
        let px = (if is_me { pred.x } else { p.x }) * sc;
        let py = (if is_me { pred.y } else { p.y }) * sc;
        let col = if p.team == 0 { 0x38C4FF } else { 0xFF4444 };
        draw_disk(fb, px, py, 3.2, col);
        if is_me {
            draw_ring(fb, px, py, 4.8, 0xFFFFFF);
        }
        if p.dashing {
            draw_ring(fb, px, py, 6.5, 0xFFFF88);
        }
        if p.melee_active {
            draw_ring(fb, px + p.aim_x * 6.0, py + p.aim_y * 6.0, 5.5, 0xFFF5B0);
        }
        draw_line(fb, px, py, px + p.aim_x * 7.0, py + p.aim_y * 7.0, col);
        let hp_w = ((p.hp.max(0) as f32 / p.max_hp.max(1) as f32) * 8.0).round() as i32;
        let bx = px.round() as i32 - 4;
        let by = py.round() as i32 - 7;
        for dx in 0..8 {
            put_px(fb, bx + dx, by, if dx < hp_w { 0x35EE55 } else { 0x252528 });
        }
    }
}

struct NeuralDeferred3D {
    session: Session,
    mtime: Option<SystemTime>,
}

impl NeuralDeferred3D {
    fn try_reload(env: &Environment, path: &Path, cur: &mut Option<Self>) {
        if !path.exists() {
            return;
        }
        let disk_mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
        let needs = match cur {
            None => true,
            Some(r) => disk_mtime.is_some() && disk_mtime != r.mtime,
        };
        if needs {
            let res: ort::Result<Session> = (|| {
                Ok(Session::builder(env)?
                    .with_intra_threads(4)?
                    .commit_from_file(path)?)
            })();
            match res {
                Ok(session) => {
                    log::info!("Loaded Tier 2 3D Neural Deferred Renderer from {}", path.display());
                    *cur = Some(Self {
                        session,
                        mtime: disk_mtime,
                    });
                }
                Err(e) => log::warn!("Tier 2 3D renderer load failed {}: {}", path.display(), e),
            }
        }
    }

    fn shade(&mut self, gbuf: &[f32], fb: &mut [u32]) -> Result<(), Box<dyn std::error::Error>> {
        let inp = Tensor::from_array(([1usize, GBUF_CHANNELS, H3D, W3D], gbuf.to_vec()))?;
        let out = self.session.run(ort::inputs!["gbuf" => inp])?;
        let (_s, rgb) = out["rgb_3d"].try_extract_tensor::<f32>()?;
        let plane = W3D * H3D;
        if rgb.len() != 3 * plane {
            return Err(format!("Bad 3D rgb len {}", rgb.len()).into());
        }
        for i in 0..plane {
            let r = (rgb[i].clamp(0.0, 1.0) * 255.0) as u32;
            let g = (rgb[plane + i].clamp(0.0, 1.0) * 255.0) as u32;
            let b = (rgb[2 * plane + i].clamp(0.0, 1.0) * 255.0) as u32;
            fb[i] = (r << 16) | (g << 8) | b;
        }
        Ok(())
    }
}

/// Depth-tested 3D entity & combat projection matching the exact `OrbitCamera` used for the G-buffer!
fn composite_combat_3d(
    mirror: &NetMirror,
    pred: &LocalPredictor,
    cam: &OrbitCamera,
    depth_buf: &[f32],
    fb: &mut [u32],
) {
    let h_scale = 14.0f32;
    let is_visible = |sx: f32, sy: f32, d: f32| -> bool {
        let xi = sx.round() as i32;
        let yi = sy.round() as i32;
        if xi < 0 || yi < 0 || xi >= W3D as i32 || yi >= H3D as i32 {
            return false;
        }
        d <= depth_buf[(yi as usize) * W3D + (xi as usize)] + 0.035
    };

    // Floating ice rafts
    for b in &mirror.bodies {
        let gy = (b.y as usize).min(GRID_H - 1);
        let gx = (b.x as usize).min(GRID_W - 1);
        let wy = cell_height_norm(&mirror.aplus_grid, gx, gy) * h_scale + 0.6;
        if let Some((sx, sy, d)) = cam.project_point(b.x, wy, b.y) {
            if is_visible(sx, sy, d) {
                let rad = (b.radius * 2.4 / (d * 2.0 + 0.35)).clamp(2.5, 10.0);
                draw_disk(fb, sx, sy, rad, 0xA8F0FF);
                draw_ring(fb, sx, sy, rad, 0xFFFFFF);
            }
        }
    }

    // Golden Relic
    {
        let gx = (mirror.relic_x as usize).min(GRID_W - 1);
        let gy = (mirror.relic_y as usize).min(GRID_H - 1);
        let wy = cell_height_norm(&mirror.aplus_grid, gx, gy) * h_scale + 2.2;
        if let Some((sx, sy, d)) = cam.project_point(mirror.relic_x, wy, mirror.relic_y) {
            if is_visible(sx, sy, d) {
                draw_disk(fb, sx, sy, 3.6, 0xFFD830);
                draw_ring(fb, sx, sy, 5.8, 0xFFF2A0);
            }
        }
    }

    // Projectiles
    for pr in &mirror.projectiles {
        let gx = (pr.x as usize).min(GRID_W - 1);
        let gy = (pr.y as usize).min(GRID_H - 1);
        let wy = cell_height_norm(&mirror.aplus_grid, gx, gy) * h_scale + 2.4;
        if let Some((sx, sy, d)) = cam.project_point(pr.x, wy, pr.y) {
            if is_visible(sx, sy, d) {
                let col = match pr.element {
                    0 => 0xFF7020,
                    1 => 0x45EEFF,
                    _ => 0xAA40E0,
                };
                draw_disk(fb, sx, sy, 2.6, col);
            }
        }
    }

    // Players (with vertical body pillar + HP bar + dash i-frame halo)
    for p in &mirror.players {
        let is_me = p.id == mirror.my_player_id && pred.initialized;
        let px = if is_me { pred.x } else { p.x };
        let pz = if is_me { pred.y } else { p.y };
        let gx = (px as usize).min(GRID_W - 1);
        let gz = (pz as usize).min(GRID_H - 1);
        let ground_y = cell_height_norm(&mirror.aplus_grid, gx, gz) * h_scale;
        let head_y = ground_y + 3.2;
        if let (Some((fx, fy, _)), Some((hx, hy, d))) = (
            cam.project_point(px, ground_y + 0.4, pz),
            cam.project_point(px, head_y, pz),
        ) {
            if is_visible(hx, hy, d) {
                let col = if p.team == 0 { 0x38C4FF } else { 0xFF4444 };
                draw_line(fb, fx, fy, hx, hy, col);
                draw_disk(fb, hx, hy, 3.0, col);
                if is_me {
                    draw_ring(fb, hx, hy, 4.6, 0xFFFFFF);
                }
                if p.dashing {
                    draw_ring(fb, hx, hy, 6.2, 0xFFFF88);
                }
                let hp_w = ((p.hp.max(0) as f32 / p.max_hp.max(1) as f32) * 8.0).round() as i32;
                let bx = hx.round() as i32 - 4;
                let by = hy.round() as i32 - 6;
                for dx in 0..8 {
                    put_px(fb, bx + dx, by, if dx < hp_w { 0x35EE55 } else { 0x252528 });
                }
            }
        }
    }
}

enum OutboundMsg {
    Input(ClientInputDatagram),
    Reliable(ClientReliable),
}

fn main() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,ort=warn"),
    )
    .init();
    let args = Args::parse();

    let ort_env = ort::init()
        .with_name("frostmarsh-neural-client")
        .build()
        .expect("ort init");
    let mut renderer: Option<NeuralRenderer> = None;
    NeuralRenderer::try_reload(&ort_env, &args.renderer_v2, &args.renderer_v1, &mut renderer);
    let mut renderer_3d: Option<NeuralDeferred3D> = None;
    NeuralDeferred3D::try_reload(&ort_env, &args.renderer_3d, &mut renderer_3d);

    let mirror = Arc::new(Mutex::new(NetMirror::new()));
    let (tx, mut rx) = mpsc::unbounded_channel::<OutboundMsg>();
    let start_time = Instant::now();

    // Background Tokio QUIC thread
    let net_mirror = mirror.clone();
    let server_addr = args.server;
    let player_name = args.name.clone();
    let player_team = args.team;
    let net_start = start_time;
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("tokio rt");
        rt.block_on(async move {
            let mut ep = Endpoint::client("0.0.0.0:0".parse().unwrap()).expect("client ep");
            ep.set_default_client_config(make_quic_client_config());
            let conn = match ep.connect(server_addr, "localhost").unwrap().await {
                Ok(c) => c,
                Err(e) => {
                    log::error!("QUIC connect to {} failed: {}", server_addr, e);
                    return;
                }
            };
            log::info!("Connected via QUIC to {}", server_addr);
            {
                let mut m = net_mirror.lock().unwrap();
                m.connected = true;
            }

            // Send Join on reliable stream
            if let Ok(mut send) = conn.open_uni().await {
                let _ = send
                    .write_all(&encode_msg(&ClientReliable::Join {
                        name: player_name,
                        team: player_team,
                    }))
                    .await;
                let _ = send.finish();
            }

            // Spawn outbound pump
            let conn_out = conn.clone();
            tokio::spawn(async move {
                while let Some(msg) = rx.recv().await {
                    match msg {
                        OutboundMsg::Input(inp) => {
                            let _ = conn_out.send_datagram(Bytes::from(encode_msg(&inp)));
                        }
                        OutboundMsg::Reliable(rel) => {
                            if let Ok(mut s) = conn_out.open_uni().await {
                                let _ = s.write_all(&encode_msg(&rel)).await;
                                let _ = s.finish();
                            }
                        }
                    }
                }
            });

            // Spawn reliable reader (Welcome)
            let m_rel = net_mirror.clone();
            let conn_rel = conn.clone();
            tokio::spawn(async move {
                while let Ok(mut recv) = conn_rel.accept_uni().await {
                    if let Ok(bytes) = recv.read_to_end(4096).await {
                        if let Some(ServerReliable::Welcome { your_player_id, .. }) =
                            decode_msg::<ServerReliable>(&bytes)
                        {
                            let mut m = m_rel.lock().unwrap();
                            m.my_player_id = your_player_id;
                        }
                    }
                }
            });

            // Read unreliable datagrams (10 Hz LZ4 A+ chunks + 30 Hz EntitySnapshots)
            while let Ok(bytes) = conn.read_datagram().await {
                if let Some(pkt) = decode_server_datagram(&bytes) {
                    let now_ms = net_start.elapsed().as_millis() as u32;
                    let mut m = net_mirror.lock().unwrap();
                    match pkt {
                        ServerDatagram::APlusChunk {
                            field_tick,
                            cx,
                            cy,
                            lz4_cells,
                            ..
                        } => {
                            if let Some(raw) = decompress_aplus_cells(&lz4_cells) {
                                m.apply_aplus_chunk(cx, cy, field_tick, &raw);
                            }
                        }
                        ServerDatagram::EntitySnapshot {
                            sim_tick,
                            your_player_id,
                            echo_client_time_ms,
                            relic_x,
                            relic_y,
                            players,
                            projectiles,
                            bodies,
                            telemetry,
                            ..
                        } => {
                            if sim_tick >= m.sim_tick {
                                m.sim_tick = sim_tick;
                                m.my_player_id = your_player_id;
                                m.relic_x = relic_x;
                                m.relic_y = relic_y;
                                m.players = players;
                                m.projectiles = projectiles;
                                m.bodies = bodies;
                                m.telemetry = Some(telemetry);
                                if echo_client_time_ms > 0 && now_ms >= echo_client_time_ms {
                                    let rtt = (now_ms - echo_client_time_ms) as f32;
                                    m.rtt_ms_ema = if m.rtt_ms_ema == 0.0 {
                                        rtt
                                    } else {
                                        m.rtt_ms_ema * 0.85 + rtt * 0.15
                                    };
                                }
                            }
                        }
                        ServerDatagram::VolumetricBrick3D(_) => {}
                    }
                }
            }
        });
    });

    // Headless verification mode
    if let Some(ref ppm_path) = args.headless_ppm {
        let deadline = Instant::now() + Duration::from_secs(12);
        let mut seq = 1u32;
        while Instant::now() < deadline {
            let _ = tx.send(OutboundMsg::Input(ClientInputDatagram {
                seq,
                move_x: 0.0,
                move_y: 0.0,
                aim_x: 1.0,
                aim_y: 0.0,
                dash: false,
                melee: false,
                cast_bolt: false,
                bolt_element: 1,
                carve_ice: false,
                cast_ability_e: false,
                cast_ultimate_r: false,
                client_time_ms: start_time.elapsed().as_millis() as u32,
            }));
            seq += 1;
            {
                let m = mirror.lock().unwrap();
                if m.chunks_received >= 60 && m.telemetry.is_some() && m.rtt_ms_ema > 0.0 {
                    break;
                }
            }
            std::thread::sleep(Duration::from_millis(60));
        }

        let m = mirror.lock().unwrap();
        let mut fb = vec![0u32; OUT_W * OUT_H];
        let pred = LocalPredictor::new();
        let t0 = Instant::now();
        let tag = if args.mode_3d {
            let cam = OrbitCamera::default();
            let mut gbuf = vec![0.0f32; GBUF_CHANNELS * W3D * H3D];
            let mut depth_buf = vec![1.0f32; W3D * H3D];
            render_gbuffer_3d(&m.aplus_grid, &cam, &mut gbuf, &mut depth_buf);
            let t = if let Some(ref mut r3) = renderer_3d {
                if r3.shade(&gbuf, &mut fb).is_ok() {
                    if args.tone_ramp {
                        apply_9step_tone_ramp(&mut fb);
                        "TIER2-3D-NN+9STEP"
                    } else {
                        "TIER2-3D-NN"
                    }
                } else {
                    shade_gbuffer_fallback(&gbuf, &mut fb);
                    "TIER2-3D-FALLBACK"
                }
            } else {
                shade_gbuffer_fallback(&gbuf, &mut fb);
                "TIER2-3D-FALLBACK"
            };
            composite_combat_3d(&m, &pred, &cam, &depth_buf, &mut fb);
            t
        } else {
            let t = if let Some(ref mut r) = renderer {
                r.render(&m, 1.0, &mut fb).unwrap_or("FALLBACK")
            } else {
                render_fallback(&m.aplus_grid, &mut fb);
                "FALLBACK"
            };
            composite_combat(&m, &pred, &mut fb);
            t
        };
        let render_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let mut bytes = format!("P6\n{} {}\n255\n", OUT_W, OUT_H).into_bytes();
        bytes.reserve(OUT_W * OUT_H * 3);
        for &px in &fb {
            bytes.push(((px >> 16) & 0xff) as u8);
            bytes.push(((px >> 8) & 0xff) as u8);
            bytes.push((px & 0xff) as u8);
        }
        std::fs::write(ppm_path, &bytes).expect("write ppm");
        let tele = m.telemetry.as_ref();
        println!(
            "QUIC_HEADLESS_OK server={} renderer={} render_ms={:.2} quic_rtt_ms={:.2} chunks={} world_mode={} backend={} world_infer_ms={:.2} wire_kbps={:.1} mass_err={} energy_err={}",
            args.server,
            tag,
            render_ms,
            m.rtt_ms_ema,
            m.chunks_received,
            tele.map(|t| t.mode.as_str()).unwrap_or("?"),
            tele.map(|t| t.backend.as_str()).unwrap_or("?"),
            tele.map(|t| t.infer_ms_last).unwrap_or(-1.0),
            tele.map(|t| t.wire_kbps).unwrap_or(-1.0),
            tele.map(|t| t.mass_error).unwrap_or(-1),
            tele.map(|t| t.energy_error).unwrap_or(-1),
        );
        return;
    }

    // Interactive Window Loop
    let mut window = Window::new(
        "Frostmarsh — QUIC Neural World + Neural Renderer",
        OUT_W,
        OUT_H,
        WindowOptions {
            scale: Scale::X4,
            resize: false,
            ..WindowOptions::default()
        },
    )
    .expect("window");
    window.set_target_fps(60);

    let mut world_fb = vec![0u32; OUT_W * OUT_H];
    let mut final_fb = vec![0u32; OUT_W * OUT_H];
    let mut gbuf_3d = vec![0.0f32; GBUF_CHANNELS * W3D * H3D];
    let mut depth_3d = vec![1.0f32; W3D * H3D];
    let mut cam = OrbitCamera::default();
    let mut mode_3d = args.mode_3d;
    let mut tone_ramp = args.tone_ramp;
    let mut cam_dirty = true;
    let mut pred = LocalPredictor::new();
    let mut input_seq: u32 = 0;
    let mut selected_element: u8 = 1;
    let mut crisp_overlay = true;
    let mut last_input_send = Instant::now();
    let mut last_world_render = Instant::now() - Duration::from_secs(1);
    let mut last_field_tick: u64 = u64::MAX;
    let mut last_model_check = Instant::now();
    let mut last_title = Instant::now();
    let mut render_ms_ema: f32 = 0.0;
    let mut renderer_tag: &'static str = "FALLBACK";

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if last_model_check.elapsed() > Duration::from_secs(2) {
            last_model_check = Instant::now();
            NeuralRenderer::try_reload(&ort_env, &args.renderer_v2, &args.renderer_v1, &mut renderer);
            NeuralDeferred3D::try_reload(&ort_env, &args.renderer_3d, &mut renderer_3d);
        }

        if window.is_key_pressed(Key::V, KeyRepeat::No) {
            mode_3d = !mode_3d;
            cam_dirty = true;
        }
        if window.is_key_pressed(Key::T, KeyRepeat::No) {
            tone_ramp = !tone_ramp;
            cam_dirty = true;
        }
        if mode_3d {
            if window.is_key_down(Key::Q) {
                cam.yaw -= 0.035;
                cam_dirty = true;
            }
            if window.is_key_down(Key::Z) {
                cam.yaw += 0.035;
                cam_dirty = true;
            }
            if window.is_key_down(Key::U) {
                cam.pitch = (cam.pitch + 0.025).clamp(0.35, 1.38);
                cam_dirty = true;
            }
            if window.is_key_down(Key::J) {
                cam.pitch = (cam.pitch - 0.025).clamp(0.35, 1.38);
                cam_dirty = true;
            }
            if window.is_key_down(Key::Equal) {
                cam.dist = (cam.dist - 1.5).clamp(36.0, 145.0);
                cam_dirty = true;
            }
            if window.is_key_down(Key::Minus) {
                cam.dist = (cam.dist + 1.5).clamp(36.0, 145.0);
                cam_dirty = true;
            }
        }

        if window.is_key_pressed(Key::Key1, KeyRepeat::No) {
            selected_element = 0;
        }
        if window.is_key_pressed(Key::Key2, KeyRepeat::No) {
            selected_element = 1;
        }
        if window.is_key_pressed(Key::Key3, KeyRepeat::No) {
            selected_element = 2;
        }
        if window.is_key_pressed(Key::M, KeyRepeat::No)
            || window.is_key_pressed(Key::Tab, KeyRepeat::No)
        {
            let _ = tx.send(OutboundMsg::Reliable(ClientReliable::ToggleSimMode));
        }
        if window.is_key_pressed(Key::G, KeyRepeat::No) {
            let _ = tx.send(OutboundMsg::Reliable(ClientReliable::ToggleEcosystem));
        }
        if window.is_key_pressed(Key::R, KeyRepeat::No) {
            let _ = tx.send(OutboundMsg::Reliable(ClientReliable::ResetWorld));
        }
        if window.is_key_pressed(Key::O, KeyRepeat::No) {
            crisp_overlay = !crisp_overlay;
        }

        let dash = window.is_key_pressed(Key::Space, KeyRepeat::No);
        let melee = window.get_mouse_down(MouseButton::Right)
            || window.is_key_pressed(Key::F, KeyRepeat::No);
        let cast_bolt = window.get_mouse_down(MouseButton::Left);
        let carve_ice = window.is_key_pressed(Key::E, KeyRepeat::No);

        let m = mirror.lock().unwrap();
        if let Some(me) = m.players.iter().find(|p| p.id == m.my_player_id) {
            pred.reconcile(me, &m);
        }

        if last_input_send.elapsed() >= Duration::from_millis(33) || dash || melee || carve_ice {
            last_input_send = Instant::now();
            let mut mx = 0.0f32;
            let mut my = 0.0f32;
            if window.is_key_down(Key::W) || window.is_key_down(Key::Up) {
                my -= 1.0;
            }
            if window.is_key_down(Key::S) || window.is_key_down(Key::Down) {
                my += 1.0;
            }
            if window.is_key_down(Key::A) || window.is_key_down(Key::Left) {
                mx -= 1.0;
            }
            if window.is_key_down(Key::D) || window.is_key_down(Key::Right) {
                mx += 1.0;
            }

            let mut aim_x = 1.0f32;
            let mut aim_y = 0.0f32;
            if let Some((cx, cy)) = window.get_mouse_pos(MouseMode::Clamp) {
                let gx = (cx / OUT_W as f32) * GRID_W as f32;
                let gy = (cy / OUT_H as f32) * GRID_H as f32;
                aim_x = gx - pred.x;
                aim_y = gy - pred.y;
            }

            input_seq = input_seq.wrapping_add(1);
            let inp = ClientInputDatagram {
                seq: input_seq,
                move_x: mx,
                move_y: my,
                aim_x,
                aim_y,
                dash,
                melee,
                cast_bolt,
                bolt_element: selected_element,
                carve_ice,
                cast_ability_e: false,
                cast_ultimate_r: false,
                client_time_ms: start_time.elapsed().as_millis() as u32,
            };
            pred.push_input(inp.clone(), &m);
            let _ = tx.send(OutboundMsg::Input(inp));
        }

        // Layer 1 (10 Hz cached, or on 3D camera orbit): Neural Frame Renderer
        if cam_dirty
            || m.field_tick != last_field_tick
            || last_world_render.elapsed() >= Duration::from_millis(100)
        {
            cam_dirty = false;
            last_field_tick = m.field_tick;
            last_world_render = Instant::now();
            let t0 = Instant::now();
            renderer_tag = if mode_3d {
                render_gbuffer_3d(&m.aplus_grid, &cam, &mut gbuf_3d, &mut depth_3d);
                if let Some(ref mut r3) = renderer_3d {
                    if r3.shade(&gbuf_3d, &mut world_fb).is_ok() {
                        if tone_ramp {
                            apply_9step_tone_ramp(&mut world_fb);
                            "TIER2-3D-NN+9STEP"
                        } else {
                            "TIER2-3D-NN"
                        }
                    } else {
                        shade_gbuffer_fallback(&gbuf_3d, &mut world_fb);
                        "3D-FALLBACK"
                    }
                } else {
                    shade_gbuffer_fallback(&gbuf_3d, &mut world_fb);
                    "3D-FALLBACK"
                }
            } else if let Some(ref mut r) = renderer {
                match r.render(&m, start_time.elapsed().as_secs_f32(), &mut world_fb) {
                    Ok(tag) => tag,
                    Err(e) => {
                        log::warn!("Render err: {}", e);
                        render_fallback(&m.aplus_grid, &mut world_fb);
                        "FALLBACK"
                    }
                }
            } else {
                render_fallback(&m.aplus_grid, &mut world_fb);
                "FALLBACK"
            };
            let rms = t0.elapsed().as_secs_f32() * 1000.0;
            render_ms_ema = if render_ms_ema == 0.0 {
                rms
            } else {
                render_ms_ema * 0.85 + rms * 0.15
            };
        }

        // Layer 2 (60 Hz): Crisp predicted combat overlay (2D or depth-tested 3D!)
        final_fb.copy_from_slice(&world_fb);
        if crisp_overlay {
            if mode_3d {
                composite_combat_3d(&m, &pred, &cam, &depth_3d, &mut final_fb);
            } else {
                composite_combat(&m, &pred, &mut final_fb);
            }
        }

        if last_title.elapsed() > Duration::from_millis(250) {
            last_title = Instant::now();
            let elem = match selected_element {
                0 => "FIRE",
                1 => "FROST",
                _ => "OIL",
            };
            let title = if let Some(ref t) = m.telemetry {
                format!(
                    "Frostmarsh QUIC ({:.1}ms RTT) | World:{}/{} ({:.2}ms err=0) | Render:{} ({:.1}ms) | Spell:{} | [V]3D/2D [Q/Z/U/J]Orbit [M]Mode",
                    m.rtt_ms_ema,
                    t.mode.to_uppercase(),
                    t.backend,
                    t.infer_ms_last,
                    renderer_tag,
                    render_ms_ema,
                    elem
                )
            } else {
                format!("Frostmarsh QUIC — connecting to {} ...", args.server)
            };
            window.set_title(&title);
        }
        drop(m);

        window
            .update_with_buffer(&final_fb, OUT_W, OUT_H)
            .expect("present");
    }
}
