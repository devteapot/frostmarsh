//! Unified Scaled 3D Volumetric Zero-WAL QUIC Authoritative Neural World Server
//! (Frostmarsh v3: Shattered Reach 3D — `256 × 256 × 4 = 262,144` physically coupled 3D voxels).
//!
//! - **3D Volumetric Physics (`VolumeField`):**
//!   - Layer 0: Subterranean / Deep Fjord Riverbed
//!   - Layer 1: Surface / Marshland / Shrines / Lower Forests
//!   - Layer 2: Highland Basalt Plateaus / Alpine Glacial Lakes (Waterfall Sources!) /
//!              True 3D Overhanging Skybridges & Stone Arches directly over the Fjord!
//!   - Layer 3: Upper Atmosphere / Buoyant Rising 3D Steam & Wildfire Smoke Plumes!
//! - **Batched 3D Neural Inference (`op=1, batch=L`):** Runs all `L` vertical layers in a single
//!   FP16 CUDA Graph call on NVIDIA GB10 (`~18.7 ms` GPU for 262k voxels), followed by exact
//!   integer 3D vertical gravity/buoyancy/conduction (`mass_error == 0`, `energy_error == 0` in 3D!).
//! - **3D Combat & Verticality:** Players/bots/projectiles move in 3D `(x, y, z)` with climbing ramps,
//!   under-bridge rivers, cliff-drop gravity, destructible Layer 2 timber skybridges, and +25% High-Ground damage!

mod ws_bridge;

use bytes::Bytes;
use clap::Parser;
use field_core::{
    biome_name, EntityVoxelPhysics, Field, INPUT_CHANNELS, NUM_BIOMES, NUM_WORLD_LAW_PRESETS,
    OUTPUT_CHANNELS, RawStepOutput, SP_ICE, SP_OIL, SP_PLANT, SP_ROCK, SP_STEAM, SP_WATER,
    SP_WOOD, VolumeField, WorldLaws,
};
use frostmarsh_proto::*;
#[cfg(feature = "onnx")]
use ort::environment::Environment;
#[cfg(feature = "onnx")]
use ort::session::Session;
#[cfg(feature = "onnx")]
use ort::value::Tensor;
use quinn::{Connection, Endpoint, ServerConfig};
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
#[cfg(unix)]
use std::os::unix::net::UnixStream;
#[cfg(feature = "onnx")]
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
#[command(name = "frostmarsh-server", about = "Scaled 3D Volumetric QUIC Neural World Server")]
struct Args {
    #[arg(long, default_value = "0.0.0.0:4433")]
    bind: SocketAddr,
    /// Backend: "cuda-rpc", "onnx", or "refsim"
    #[arg(long, default_value = "onnx")]
    backend: String,
    #[arg(long, default_value = "127.0.0.1:14440")]
    cuda_addr: String,
    #[arg(long, default_value = "models/world_model.onnx")]
    model: PathBuf,
    #[arg(long, default_value = "models/ecosystem_model.onnx")]
    ecosystem_model: PathBuf,
    /// Horizontal square grid dimension (128, 256, or 512)
    #[arg(long, default_value_t = 256)]
    size: usize,
    /// Number of physically coupled 3D vertical strata (1..=32; 16..=32 enables isotropic 3D volumetric biomes!)
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// 3D Biome preset (0=Nordic Viaduct, 1=Volcanic Caldera, 2=Sinkhole Caverns, 3=Glacier Gorge, 4=Geothermal Mist Basin)
    #[arg(long, default_value_t = 0)]
    biome: u8,
    /// Deterministic 3D world generation seed
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Number of role-aware AI bots (up to 64; 12 default, supports 32-64 live scaling)
    #[arg(long, default_value_t = 12)]
    bots: usize,
    /// Per-player Area-of-Interest chunk radius (0 = full map)
    #[arg(long, default_value_t = 10)]
    aoi_radius: usize,
    /// Optional zero-dependency RFC 6455 WebSocket bridge bind address for Wave 12C WebGPU browser clients (e.g. 0.0.0.0:4434)
    #[arg(long)]
    ws_bind: Option<String>,
}

// ---------------- Neural Backends ----------------

enum RpcStream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(UnixStream),
}

impl Read for RpcStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            RpcStream::Tcp(s) => s.read(buf),
            #[cfg(unix)]
            RpcStream::Unix(s) => s.read(buf),
        }
    }
}

impl Write for RpcStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            RpcStream::Tcp(s) => s.write(buf),
            #[cfg(unix)]
            RpcStream::Unix(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            RpcStream::Tcp(s) => s.flush(),
            #[cfg(unix)]
            RpcStream::Unix(s) => s.flush(),
        }
    }
}

struct CudaRpcClient {
    addr: String,
    stream: Option<RpcStream>,
    seq: u32,
    actuator_v3_backoff: u32,
}

impl CudaRpcClient {
    fn new(addr: String) -> Self {
        let mut c = Self {
            addr,
            stream: None,
            seq: 0,
            actuator_v3_backoff: 0,
        };
        let _ = c.ensure_connected();
        c
    }

    fn ensure_connected(&mut self) -> Result<&mut RpcStream, Box<dyn std::error::Error>> {
        if self.stream.is_none() {
            #[cfg(unix)]
            if self.addr.starts_with('/') {
                let s = UnixStream::connect(&self.addr)?;
                s.set_read_timeout(Some(Duration::from_millis(8000)))?;
                s.set_write_timeout(Some(Duration::from_millis(8000)))?;
                log::info!("Connected to PyTorch CUDA FP16 3D daemon via Unix socket {}", self.addr);
                self.stream = Some(RpcStream::Unix(s));
                return Ok(self.stream.as_mut().unwrap());
            }
            let s = TcpStream::connect_timeout(&self.addr.parse()?, Duration::from_millis(800))?;
            s.set_nodelay(true)?;
            s.set_read_timeout(Some(Duration::from_millis(8000)))?;
            s.set_write_timeout(Some(Duration::from_millis(8000)))?;
            log::info!("Connected to PyTorch CUDA FP16 3D daemon at {}", self.addr);
            self.stream = Some(RpcStream::Tcp(s));
        }
        Ok(self.stream.as_mut().unwrap())
    }

    fn call_batched(
        &mut self,
        op: u8,
        batch: u8,
        in_ch: u16,
        h: u16,
        w: u16,
        data: &[f32],
    ) -> Result<(Vec<f32>, f32), Box<dyn std::error::Error>> {
        self.seq = self.seq.wrapping_add(1);
        let seq = self.seq;
        let payload_bytes = (data.len() * 4) as u32;
        let mut hdr = [0u8; 16];
        hdr[0] = op;
        hdr[1] = batch.max(1);
        hdr[2..4].copy_from_slice(&in_ch.to_le_bytes());
        hdr[4..6].copy_from_slice(&h.to_le_bytes());
        hdr[6..8].copy_from_slice(&w.to_le_bytes());
        hdr[8..12].copy_from_slice(&payload_bytes.to_le_bytes());
        hdr[12..16].copy_from_slice(&seq.to_le_bytes());

        let mut payload = Vec::with_capacity(data.len() * 4);
        for &v in data {
            payload.extend_from_slice(&v.to_le_bytes());
        }

        let res = (|| -> Result<(Vec<f32>, f32), Box<dyn std::error::Error>> {
            let s = self.ensure_connected()?;
            s.write_all(&hdr)?;
            s.write_all(&payload)?;

            let mut rhdr = [0u8; 16];
            s.read_exact(&mut rhdr)?;
            let status = rhdr[0];
            if status != 0 {
                return Err(format!("CUDA daemon returned error status {}", status).into());
            }
            let rlen = u32::from_le_bytes(rhdr[8..12].try_into().unwrap()) as usize;
            let infer_us = u32::from_le_bytes(rhdr[12..16].try_into().unwrap());
            let mut rbuf = vec![0u8; rlen];
            s.read_exact(&mut rbuf)?;
            let mut out = Vec::with_capacity(rlen / 4);
            for chunk in rbuf.chunks_exact(4) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            Ok((out, infer_us as f32 / 1000.0))
        })();

        if res.is_err() {
            self.stream = None;
        }
        res
    }

    fn call_json(
        &mut self,
        op: u8,
        json_req: &str,
    ) -> Result<(String, f32), Box<dyn std::error::Error>> {
        self.seq = self.seq.wrapping_add(1);
        let seq = self.seq;
        let payload = json_req.as_bytes();
        let payload_bytes = payload.len() as u32;
        let mut hdr = [0u8; 16];
        hdr[0] = op;
        hdr[1] = 1;
        hdr[2..4].copy_from_slice(&1u16.to_le_bytes());
        hdr[4..6].copy_from_slice(&0u16.to_le_bytes());
        hdr[6..8].copy_from_slice(&0u16.to_le_bytes());
        hdr[8..12].copy_from_slice(&payload_bytes.to_le_bytes());
        hdr[12..16].copy_from_slice(&seq.to_le_bytes());

        let res = (|| -> Result<(String, f32), Box<dyn std::error::Error>> {
            let s = self.ensure_connected()?;
            s.write_all(&hdr)?;
            s.write_all(payload)?;

            let mut rhdr = [0u8; 16];
            s.read_exact(&mut rhdr)?;
            let status = rhdr[0];
            if status != 0 {
                return Err(format!("CUDA daemon op={} returned error status {}", op, status).into());
            }
            let rlen = u32::from_le_bytes(rhdr[8..12].try_into().unwrap()) as usize;
            let infer_us = u32::from_le_bytes(rhdr[12..16].try_into().unwrap());
            let mut rbuf = vec![0u8; rlen];
            s.read_exact(&mut rbuf)?;
            let text = String::from_utf8(rbuf)?;
            Ok((text, infer_us as f32 / 1000.0))
        })();

        if res.is_err() {
            self.stream = None;
        }
        res
    }

    /// True 12-channel 3D neural actuator policy (`op=10`, returns `N × 28` f32 =
    /// `[move_x, move_y, move_jump, aim_dx, aim_dy, aim_dz, thermal_flux, emit_water, emit_oil,
    ///   emit_smoke, focus_radius, impulse_or_dome]` + `16` saliency floats).
    /// Automatically falls back to `op=7` (`N × 26` f32) when `actuator_v3` is not loaded on the daemon.
    /// Returns `(output_floats, infer_ms, is_v3)`.
    fn infer_actuators_v3(
        &mut self,
        n_bots: usize,
        feat_dim: usize,
        feats: &[f32],
    ) -> Result<(Vec<f32>, f32, bool), Box<dyn std::error::Error>> {
        const ACTUATOR_V3_INSPECT_DIM: usize = 28; // 12 continuous 3D channels + 16 saliency
        const ACTUATOR_V2_INSPECT_DIM: usize = 26; // 10 continuous 2D channels + 16 saliency

        if self.actuator_v3_backoff == 0 {
            match self.call_batched(10, 1, 1, n_bots as u16, feat_dim as u16, feats) {
                Ok((out, gpu_ms)) if out.len() == n_bots * ACTUATOR_V3_INSPECT_DIM => {
                    return Ok((out, gpu_ms, true));
                }
                Ok((out, _)) => {
                    log::debug!(
                        "CUDA op=10 returned unexpected len {} (expected {}); falling back to op=7",
                        out.len(),
                        n_bots * ACTUATOR_V3_INSPECT_DIM
                    );
                    self.actuator_v3_backoff = 60;
                }
                Err(e) => {
                    log::debug!("CUDA op=10 actuator_v3 unavailable ({}); falling back to op=7", e);
                    self.actuator_v3_backoff = 60;
                }
            }
        } else {
            self.actuator_v3_backoff -= 1;
        }

        let (out, gpu_ms) = self.call_batched(7, 1, 1, n_bots as u16, feat_dim as u16, feats)?;
        if out.len() != n_bots * ACTUATOR_V2_INSPECT_DIM {
            return Err(format!(
                "CUDA op=7 returned unexpected len {} (expected {})",
                out.len(),
                n_bots * ACTUATOR_V2_INSPECT_DIM
            )
            .into());
        }
        Ok((out, gpu_ms, false))
    }
}

// ---------------- Minimal Zero-Dependency JSON Codec for op=6 Acoustics ----------------

fn json_escape_str(s: &str) -> String {
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

#[derive(Debug)]
#[allow(dead_code)]
enum JsonVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(HashMap<String, JsonVal>),
}

impl JsonVal {
    fn as_f32(&self) -> Option<f32> {
        match self {
            JsonVal::Num(n) => Some(*n as f32),
            _ => None,
        }
    }
    fn as_i64(&self) -> Option<i64> {
        match self {
            JsonVal::Num(n) => Some(n.round() as i64),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            JsonVal::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[JsonVal]> {
        match self {
            JsonVal::Arr(v) => Some(v.as_slice()),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self {
            JsonVal::Bool(b) => Some(*b),
            _ => None,
        }
    }
    fn get(&self, key: &str) -> Option<&JsonVal> {
        match self {
            JsonVal::Obj(m) => m.get(key),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn parse(input: &'a str) -> Option<JsonVal> {
        let mut p = Self { bytes: input.as_bytes(), pos: 0 };
        p.parse_val()
    }
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }
    fn parse_val(&mut self) -> Option<JsonVal> {
        self.skip_ws();
        match self.peek()? {
            b'"' => self.parse_str().map(JsonVal::Str),
            b'{' => self.parse_obj(),
            b'[' => self.parse_arr(),
            b't' => {
                if self.bytes[self.pos..].starts_with(b"true") {
                    self.pos += 4;
                    Some(JsonVal::Bool(true))
                } else {
                    None
                }
            }
            b'f' => {
                if self.bytes[self.pos..].starts_with(b"false") {
                    self.pos += 5;
                    Some(JsonVal::Bool(false))
                } else {
                    None
                }
            }
            b'n' => {
                if self.bytes[self.pos..].starts_with(b"null") {
                    self.pos += 4;
                    Some(JsonVal::Null)
                } else {
                    None
                }
            }
            b'-' | b'0'..=b'9' => self.parse_num().map(JsonVal::Num),
            _ => None,
        }
    }
    fn parse_str(&mut self) -> Option<String> {
        if self.peek()? != b'"' {
            return None;
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Some(out);
            }
            if b == b'\\' {
                if self.pos >= self.bytes.len() {
                    return None;
                }
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        if self.pos + 4 <= self.bytes.len() {
                            if let Ok(hex) = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4]) {
                                if let Ok(cp) = u32::from_str_radix(hex, 16) {
                                    if let Some(ch) = char::from_u32(cp) {
                                        out.push(ch);
                                    }
                                }
                            }
                            self.pos += 4;
                        }
                    }
                    other => out.push(other as char),
                }
            } else if b < 0x80 {
                out.push(b as char);
            } else {
                // Multi-byte UTF-8 sequence
                let start = self.pos - 1;
                while self.pos < self.bytes.len() && (self.bytes[self.pos] & 0xC0) == 0x80 {
                    self.pos += 1;
                }
                if let Ok(s) = std::str::from_utf8(&self.bytes[start..self.pos]) {
                    out.push_str(s);
                }
            }
        }
        None
    }
    fn parse_num(&mut self) -> Option<f64> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        std::str::from_utf8(&self.bytes[start..self.pos]).ok()?.parse::<f64>().ok()
    }
    fn parse_arr(&mut self) -> Option<JsonVal> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Some(JsonVal::Arr(items));
            }
            items.push(self.parse_val()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Some(JsonVal::Arr(items));
                }
                _ => return None,
            }
        }
    }
    fn parse_obj(&mut self) -> Option<JsonVal> {
        self.pos += 1; // '{'
        let mut map = HashMap::new();
        loop {
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Some(JsonVal::Obj(map));
            }
            let k = self.parse_str()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return None;
            }
            self.pos += 1;
            let v = self.parse_val()?;
            map.insert(k, v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Some(JsonVal::Obj(map));
                }
                _ => return None,
            }
        }
    }
}

#[cfg(feature = "onnx")]
struct OnnxBackend {
    physics: Option<Session>,
    eco: Option<Session>,
}

#[cfg(feature = "onnx")]
impl OnnxBackend {
    fn load_session(env: &Environment, path: &Path, threads: usize) -> Option<Session> {
        if !path.exists() {
            return None;
        }
        let res: ort::Result<Session> = (|| {
            Ok(Session::builder(env)?
                .with_intra_threads(threads)?
                .commit_from_file(path)?)
        })();
        res.map_err(|e| log::warn!("ONNX load failed {}: {}", path.display(), e))
            .ok()
    }

    fn load(env: &Environment, world_path: &Path, eco_path: &Path) -> Self {
        let physics = Self::load_session(env, world_path, 4);
        let eco = Self::load_session(env, eco_path, 2);
        Self { physics, eco }
    }

    fn step_volume(&mut self, vol: &VolumeField) -> Result<Vec<RawStepOutput>, Box<dyn std::error::Error>> {
        let sess = self.physics.as_mut().ok_or("No ONNX world model")?;
        let mut outs = Vec::with_capacity(vol.depth);
        for l in &vol.layers {
            let inp = Tensor::from_array((
                [1usize, INPUT_CHANNELS, l.height, l.width],
                l.encode_input_tensor(),
            ))?;
            let out = sess.run(ort::inputs!["state" => inp])?;
            let (_s, slice) = out["deltas"].try_extract_tensor::<f32>()?;
            outs.push(RawStepOutput {
                width: l.width,
                height: l.height,
                data: slice.to_vec(),
            });
        }
        Ok(outs)
    }

    fn step_eco(
        &mut self,
        inp_data: Vec<f32>,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let sess = self.eco.as_mut().ok_or("No ONNX eco model")?;
        let inp = Tensor::from_array(([1usize, 8, h, w], inp_data))?;
        let out = sess.run(ort::inputs!["eco_state" => inp])?;
        let (_s, slice) = out["eco_next"].try_extract_tensor::<f32>()?;
        Ok(slice.to_vec())
    }
}

struct EcosystemState {
    channels: Vec<f32>, // [4, H, W]
    w: usize,
    h: usize,
}

impl EcosystemState {
    fn new(w: usize, h: usize) -> Self {
        let mut s = Self {
            channels: vec![0.0; 4 * w * h],
            w,
            h,
        };
        s.reset();
        s
    }

    fn reset(&mut self) {
        self.channels.fill(0.0);
        let len = self.w * self.h;
        let centers = [(0.22f32, 0.28f32), (0.26, 0.68), (0.76, 0.24), (0.78, 0.78)];
        for y in 0..self.h {
            for x in 0..self.w {
                let nx = x as f32 / self.w as f32;
                let ny = y as f32 / self.h as f32;
                let idx = y * self.w + x;
                let mut dens = 0.0f32;
                for &(cx, cy) in &centers {
                    let d = ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt();
                    dens += (-(d * 14.0).powi(2)).exp();
                }
                self.channels[idx] = dens.min(1.0);
                self.channels[2 * len + idx] = 0.85;
            }
        }
    }

    fn inject_fear_pulse(&mut self, wx: f32, wy: f32, radius: f32, strength: f32) {
        let len = self.w * self.h;
        let r = radius.ceil() as i32;
        let cx = wx.round() as i32;
        let cy = wy.round() as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let x = cx + dx;
                let y = cy + dy;
                if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 <= radius * radius {
                    let idx = y as usize * self.w + x as usize;
                    let fall = 1.0 - (d2.sqrt() / radius.max(1.0));
                    self.channels[len + idx] = (self.channels[len + idx] + strength * fall).min(1.5);
                }
            }
        }
    }

    fn build_input(&self, field: &Field) -> Vec<f32> {
        let len = self.w * self.h;
        let mut inp = vec![0.0f32; 8 * len];
        for idx in 0..len {
            inp[idx] = self.channels[idx];
            inp[len + idx] = self.channels[len + idx];
            inp[2 * len + idx] = self.channels[2 * len + idx];
            inp[3 * len + idx] = self.channels[3 * len + idx];
            let t = field.cell_temp_ck(idx);
            let is_wall = field.mass[SP_ROCK][idx] >= 500;
            let is_deep_water = field.mass[SP_WATER][idx] >= 450 && field.mass[SP_ICE][idx] < 350;
            inp[4 * len + idx] = if !is_wall && !is_deep_water { 1.0 } else { 0.0 };
            inp[5 * len + idx] = if t > 44_000 { 1.0 } else { 0.0 };
            inp[6 * len + idx] = (field.mass[SP_PLANT][idx] as f32 / 1000.0).clamp(0.0, 1.0);
            inp[7 * len + idx] = (field.mass[SP_ICE][idx] as f32 / 1000.0).clamp(0.0, 1.0);
        }
        inp
    }

    fn apply_output(&mut self, out: &[f32], field: &mut Field) {
        let len = self.w * self.h;
        if out.len() != 4 * len {
            return;
        }
        for i in 0..(4 * len) {
            let v = out[i];
            self.channels[i] = if v.is_finite() { v.clamp(0.0, 1.5) } else { 0.0 };
        }
        for idx in 0..len {
            field.critter_density[idx] = (self.channels[idx] * 7.0).round().clamp(0.0, 7.0) as u8;
        }
    }
}

// ---------------- 3D Game State ----------------

/// Physical body morphology parameters derived from each archetype to couple locomotion,
/// flight/glide, aquatic buoyancy, climbing, boring, knockback resistance, and thermal insulation
/// directly to the live `WorldLaws` & volumetric voxel field.
#[derive(Clone, Copy, Debug)]
struct BodyMorphology {
    mass: f32,
    wing_lift: f32,
    buoyancy: f32,
    climb_grip: f32,
    bore_torque: f32,
    thermal_insul: f32,
}

impl BodyMorphology {
    fn from_archetype(class_id: u8) -> Self {
        match class_id % 5 {
            CLASS_STRIDER => Self {
                mass: 2.4,
                wing_lift: 0.0,
                buoyancy: 0.15,
                climb_grip: 0.15,
                bore_torque: 0.0,
                thermal_insul: 0.35,
            },
            CLASS_SKYWING => Self {
                mass: 0.85,
                wing_lift: 0.85,
                buoyancy: 0.10,
                climb_grip: 0.10,
                bore_torque: 0.0,
                thermal_insul: 0.25,
            },
            CLASS_TIDEKIN => Self {
                mass: 1.05,
                wing_lift: 0.0,
                buoyancy: 1.0,
                climb_grip: 0.25,
                bore_torque: 0.0,
                thermal_insul: 0.72,
            },
            CLASS_EMBERSTALKER => Self {
                mass: 0.95,
                wing_lift: 0.12,
                buoyancy: 0.20,
                climb_grip: 0.95,
                bore_torque: 0.0,
                thermal_insul: 0.48,
            },
            _ => Self {
                // CLASS_BURROWER (Terramole)
                mass: 2.2,
                wing_lift: 0.0,
                buoyancy: 0.12,
                climb_grip: 0.30,
                bore_torque: 1.0,
                thermal_insul: 0.55,
            },
        }
    }
}

#[derive(Clone)]
struct PlayerEntity {
    id: u32,
    name: String,
    team: u8,
    class_id: u8,
    squad_id: u8, // 0 = Solo Raider (Emberstalker) / Human, 1..=8 = Squad Alpha/Bravo/Charlie/...
    is_bot: bool,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    desired_vz: f32,
    aim_x: f32,
    aim_y: f32,
    hp: i16,
    max_hp: i16,
    mana: f32,
    max_mana: f32,
    stamina: f32,
    max_stamina: f32,
    level: u8,
    xp: u16,
    kills: u16,
    deaths: u16,
    score: u32,
    dash_until: u64,
    dash_cd: u64,
    melee_until: u64,
    melee_cd: u64,
    bolt_cd: u64,
    ability_e_cd: u64,
    ult_r_cd: u64,
    ult_until: u64,
    shield_until: u64,
    haste_until: u64,
    oiled_until: u64,
    chilled_until: u64,
    carrying_relic: bool,
    ack_input_seq: u32,
    echo_client_time_ms: u32,
    kb_vx: f32,
    kb_vy: f32,
    last_phys_blast_field_tick: u64,
    burrow_ambush_until: u64,
    status_burn_until: u64,
    shield_wall_until: u64,
    quench_cd_until: u64,
    smoke_veil_active: bool,
    in_tunnel: bool,
    in_water: bool,
    airborne: bool,
}

impl PlayerEntity {
    fn archetype_name(class_id: u8) -> &'static str {
        match class_id % 5 {
            CLASS_STRIDER => "STRIDER",
            CLASS_SKYWING => "SKYWING",
            CLASS_TIDEKIN => "TIDEKIN",
            CLASS_EMBERSTALKER => "EMBERSTALKER",
            _ => "BURROWER",
        }
    }

    fn class_stats(class_id: u8, level: u8) -> (i16, f32, f32, f32) {
        let lvl_bonus = (level.saturating_sub(1)) as i16;
        match class_id % 5 {
            CLASS_STRIDER => (175 + lvl_bonus * 20, 105.0, 110.0, 15.2),
            CLASS_SKYWING => (115 + lvl_bonus * 12, 145.0, 115.0, 17.2),
            CLASS_TIDEKIN => (128 + lvl_bonus * 14, 120.0, 135.0, 18.2),
            CLASS_EMBERSTALKER => (122 + lvl_bonus * 13, 125.0, 145.0, 19.6),
            _ => (160 + lvl_bonus * 18, 115.0, 130.0, 16.4), // CLASS_BURROWER (Terramole)
        }
    }

    fn grant_xp(&mut self, amount: u16) -> bool {
        self.xp = self.xp.saturating_add(amount);
        let threshold = (self.level as u16) * 100;
        if self.level < 5 && self.xp >= threshold {
            self.xp -= threshold;
            self.level += 1;
            let (mhp, mmp, mst, _) = Self::class_stats(self.class_id, self.level);
            self.max_hp = mhp;
            self.hp = (self.hp + 30).min(mhp);
            self.max_mana = mmp;
            self.max_stamina = mst;
            true
        } else {
            false
        }
    }
}

struct ProjectileEntity {
    id: u32,
    owner: u32,
    team: u8,
    element: u8,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    radius: f32,
    power: f32,
    ttl: u16,
}

struct RaftEntity {
    id: u32,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
    radius: f32,
    ttl: u32,
}

struct ShrineEntity {
    id: u8,
    name: String,
    x: f32,
    y: f32,
    progress: f32,
    owner: i8,
}

#[derive(Clone, Debug)]
struct TimeDomeEntity {
    id: u32,
    x: f32,
    y: f32,
    radius: f32,
    dt_scale: f32, // 0.15 = Stasis Dome, 3.0 = Haste Zone
    expires_tick: u64,
    owner_team: u8,
}

struct ClientPeer {
    conn: Connection,
    needs_full_sync: bool,
    macro_cursor: usize,
}

#[derive(Clone, Debug)]
struct PendingChat {
    text: String,
    x: f32,
    y: f32,
    layer: f32,
    volume: f32,
    team: u8,
}

#[derive(Clone, Copy, Debug)]
struct BotCommandOverride {
    move_x: f32,
    move_y: f32,
    elem: Option<u8>,
    ability: Option<u8>,
    #[allow(dead_code)]
    snr_db: f32,
    expires_tick: u64,
}

const MATCH_DURATION_SECS: u16 = 300;
const MATCH_SCORE_LIMIT: u16 = 500;
const CELEBRATION_SECS: u16 = 10;

struct GameState {
    grid_w: usize,
    grid_h: usize,
    grid_d: usize,
    biome_id: u8,
    world_seed: u64,
    chunks_x: usize,
    chunks_y: usize,
    aoi_radius: usize,
    volume: VolumeField,
    eco: EcosystemState,
    sim_mode: String,
    backend_name: String,
    ecosystem_on: bool,
    neural_bots_on: bool,
    physics_native: bool,
    bot_infer_ms: f32,
    acoustic_ms: f32,
    world_sparse_pct: f32,
    bot_rpc: Option<CudaRpcClient>,
    sim_tick: u64,
    match_start_instant: Instant,
    celebration_started: Option<Instant>,
    match_phase: u8,
    match_time_left_s: u16,
    mvp_name: String,
    mvp_score: u32,
    next_id: u32,
    next_event_id: u32,
    score_west: u16,
    score_east: u16,
    relic_x: f32,
    relic_y: f32,
    relic_carrier: Option<u32>,
    shrines: Vec<ShrineEntity>,
    players: HashMap<u32, PlayerEntity>,
    projectiles: Vec<ProjectileEntity>,
    rafts: Vec<RaftEntity>,
    events: VecDeque<(u64, CombatEvent)>,
    bot_brains: Vec<BotBrainSnap>,
    kill_feed: VecDeque<KillFeedEntry>,
    speech_bubbles: VecDeque<(u64, SpeechBubbleSnap)>,
    pending_chat: Option<PendingChat>,
    bot_overrides: HashMap<u32, BotCommandOverride>,
    recent_actuators: VecDeque<(u32, u8, f32, f32, f32, u64)>,
    actuator_pulses: Vec<(u64, f32, ActuatorPulseSnap)>,
    time_domes: Vec<TimeDomeEntity>,
    system2_slm_active: bool,
    peers: HashMap<u32, ClientPeer>,
    hazard_layers: Vec<Vec<u8>>,
    weather_kind: u8,
    day_time: f32,
    time_scale: f32,
    evo_speed: f32,
    world_day: u32,
    last_epoch_name: &'static str,
    tunnels_carved: u64,
    tunnel_mass_displaced: u64,
    last_combo_tick: [u64; 10],
    last_squad_dome_tick: [u64; 2],
    tactical_events: VecDeque<ws_bridge::WsTacticalEvent>,
    pending_cryo_bridges: VecDeque<(u32, u8, f32, f32, usize, u16, u64)>,
    pending_basalt_quenches: VecDeque<(u32, u8, f32, f32, usize, u16, u64, u64)>,
    neural_3d_vertical: bool,
    wind_x: f32,
    wind_y: f32,
    infer_ms_last: f32,
    eco_infer_ms: f32,
    infer_window: VecDeque<f32>,
    proj_active_pct: f32,
    mass_error: i64,
    energy_error: i64,
    aoi_chunks_sent_last: u16,
    bytes_sent_window: u64,
    bytes_window_start: Instant,
    wire_kbps: f32,
    rt_handle: tokio::runtime::Handle,
    ws_bridge: Option<Arc<ws_bridge::WsBridge>>,
    tps_window_start: Instant,
    tps_window_ticks: u32,
    measured_tps: f32,
}

impl GameState {
    #[inline]
    fn is_z16(&self) -> bool {
        self.grid_d >= 16
    }

    fn surf_layer(&self) -> usize {
        if self.grid_d >= 16 {
            ((4.0 * (self.grid_d as f32 / 16.0)).round() as usize).min(self.grid_d - 1)
        } else if self.grid_d >= 2 {
            1
        } else {
            0
        }
    }

    fn find_z16_surface_z(&self, x: f32, y: f32, start_z: f32) -> f32 {
        let max_z = (self.grid_d as f32) - 2.0;
        let mut z = start_z.clamp(1.0, max_z.max(1.0));
        // If start_z begins inside solid rock/terrain, step upward into open air first
        while z < max_z && self.volume.sample_solid_density_3d(x, y, z) >= 0.5 {
            z += 0.35;
        }
        // Scan downward to find the highest solid floor with open-air headroom above
        let mut fallback_z: Option<f32> = None;
        while z > 0.6 {
            if self.volume.sample_solid_density_3d(x, y, z) >= 0.5 {
                let foot_z = (z + 0.25).min(max_z.max(1.0));
                if fallback_z.is_none() {
                    fallback_z = Some(foot_z);
                }
                let clear_mid = self.volume.sample_solid_density_3d(x, y, (foot_z + 0.85).min(max_z + 1.0)) < 0.5;
                let clear_top = self.volume.sample_solid_density_3d(x, y, (foot_z + 1.65).min(max_z + 1.5)) < 0.5;
                if clear_mid && clear_top {
                    return foot_z;
                }
            }
            z -= 0.2;
        }
        fallback_z.unwrap_or(1.5)
    }

    fn build_shrines(w: usize, h: usize) -> Vec<ShrineEntity> {
        let wf = w as f32;
        let hf = h as f32;
        vec![
            ShrineEntity { id: 0, name: "NORTH GLACIER".into(), x: wf * 0.50, y: hf * 0.18, progress: -0.35, owner: 0 },
            ShrineEntity { id: 1, name: "CENTRAL SPIRE".into(), x: wf * 0.50, y: hf * 0.50, progress: 0.0, owner: 0 },
            ShrineEntity { id: 2, name: "SOUTH CALDERA".into(), x: wf * 0.50, y: hf * 0.82, progress: 0.35, owner: 0 },
        ]
    }

    fn new(
        backend_name: String,
        cuda_addr: String,
        size: usize,
        layers: usize,
        biome: u8,
        seed: u64,
        bot_count: usize,
        aoi_radius: usize,
    ) -> Self {
        let grid_w = (size / CHUNK_SIZE).max(4) * CHUNK_SIZE;
        let grid_h = grid_w;
        let grid_d = layers.clamp(1, 32);
        let biome_id = biome % NUM_BIOMES;
        let chunks_x = chunks_for_dim(grid_w);
        let chunks_y = chunks_for_dim(grid_h);
        let volume = VolumeField::new_biome_3d(grid_w, grid_h, grid_d, biome_id, seed);
        let init_epoch = volume.evolution_epoch_name();
        let eco = EcosystemState::new(grid_w, grid_h);
        let shrines = Self::build_shrines(grid_w, grid_h);
        let use_cuda = backend_name == "cuda-rpc";

        let mut gs = Self {
            grid_w,
            grid_h,
            grid_d,
            biome_id,
            world_seed: seed,
            chunks_x,
            chunks_y,
            aoi_radius,
            hazard_layers: vec![vec![1u8; grid_w * grid_h]; grid_d],
            volume,
            eco,
            sim_mode: "neural".to_string(),
            backend_name,
            ecosystem_on: true,
            neural_bots_on: use_cuda,
            physics_native: true,
            bot_infer_ms: 0.0,
            acoustic_ms: 0.0,
            world_sparse_pct: 100.0,
            bot_rpc: if use_cuda { Some(CudaRpcClient::new(cuda_addr)) } else { None },
            sim_tick: 0,
            match_start_instant: Instant::now(),
            celebration_started: None,
            match_phase: MATCH_PHASE_ACTIVE,
            match_time_left_s: MATCH_DURATION_SECS,
            mvp_name: String::new(),
            mvp_score: 0,
            next_id: 1,
            next_event_id: 1,
            score_west: 0,
            score_east: 0,
            relic_x: (grid_w as f32) * 0.50,
            relic_y: (grid_h as f32) * 0.50,
            relic_carrier: None,
            shrines,
            players: HashMap::new(),
            projectiles: Vec::new(),
            rafts: Vec::new(),
            events: VecDeque::with_capacity(48),
            bot_brains: Vec::new(),
            kill_feed: VecDeque::with_capacity(8),
            speech_bubbles: VecDeque::with_capacity(6),
            pending_chat: None,
            bot_overrides: HashMap::new(),
            recent_actuators: VecDeque::with_capacity(32),
            actuator_pulses: Vec::with_capacity(32),
            time_domes: Vec::with_capacity(4),
            system2_slm_active: false,
            peers: HashMap::new(),
            weather_kind: WEATHER_CLEAR,
            day_time: 0.25,
            time_scale: 1.0,
            evo_speed: 1.0,
            world_day: 1,
            last_epoch_name: init_epoch,
            tunnels_carved: 0,
            tunnel_mass_displaced: 0,
            last_combo_tick: [0; 10],
            last_squad_dome_tick: [0, 0],
            tactical_events: VecDeque::with_capacity(16),
            pending_cryo_bridges: VecDeque::with_capacity(16),
            pending_basalt_quenches: VecDeque::with_capacity(16),
            neural_3d_vertical: false,
            wind_x: 0.65,
            wind_y: 0.35,
            infer_ms_last: 0.0,
            eco_infer_ms: 0.0,
            infer_window: VecDeque::with_capacity(100),
            proj_active_pct: 0.0,
            mass_error: 0,
            energy_error: 0,
            aoi_chunks_sent_last: 0,
            bytes_sent_window: 0,
            bytes_window_start: Instant::now(),
            wire_kbps: 0.0,
            rt_handle: tokio::runtime::Handle::current(),
            ws_bridge: None,
            tps_window_start: Instant::now(),
            tps_window_ticks: 0,
            measured_tps: 30.0,
        };
        gs.refresh_hazard_grids();
        let capped_bots = bot_count.min(64);
        let squad_size = if capped_bots >= 24 { 4usize } else { 3usize };
        let mut team_squad_counts = [0usize; 2];
        for i in 0..capped_bots {
            let team = (i % 2) as u8;
            let class_id = ((i / 2) % 5) as u8;
            let squad_id = if class_id == CLASS_EMBERSTALKER {
                0u8 // Solo Raider (~20% of bots)
            } else {
                let idx = team_squad_counts[team as usize];
                team_squad_counts[team as usize] += 1;
                ((idx / squad_size) + 1).min(8) as u8
            };
            let role_tag = match class_id {
                CLASS_STRIDER => "Strider",
                CLASS_SKYWING => "Skywing",
                CLASS_TIDEKIN => "Tidekin",
                CLASS_EMBERSTALKER => "Stalker",
                _ => "Terramole",
            };
            let prefix = if team == 0 { "Frost" } else { "Ember" };
            let name = if squad_id == 0 {
                format!("{}-{}-Solo{}", prefix, role_tag, i / 2 + 1)
            } else {
                let sq_code = match squad_id {
                    1 => "A",
                    2 => "B",
                    3 => "C",
                    4 => "D",
                    5 => "E",
                    6 => "F",
                    _ => "G",
                };
                format!("{}-{}-{}{}", prefix, role_tag, sq_code, i / 2 + 1)
            };
            let pid = gs.spawn_player(name, team, class_id, true);
            if let Some(p) = gs.players.get_mut(&pid) {
                p.squad_id = squad_id;
            }
        }
        gs
    }

    fn push_event(&mut self, kind: u8, x: f32, y: f32, value: i16, team: u8, label: String) {
        let id = self.next_event_id;
        self.next_event_id = self.next_event_id.wrapping_add(1);
        if kind >= 2 && !label.is_empty() {
            let dup_recent = self
                .tactical_events
                .back()
                .map(|last| last.label == label && self.sim_tick <= last.tick.saturating_add(15))
                .unwrap_or(false);
            if !dup_recent {
                self.tactical_events.push_back(ws_bridge::WsTacticalEvent {
                    label: label.clone(),
                    team,
                    x,
                    y,
                    tick: self.sim_tick,
                });
                while self.tactical_events.len() > 16 {
                    self.tactical_events.pop_front();
                }
            }
        }
        self.events.push_back((
            self.sim_tick + 90,
            CombatEvent { id, kind, x, y, value, team, label },
        ));
        while self.events.len() > 28 {
            self.events.pop_front();
        }
    }

    fn push_speech_bubble(&mut self, bubble: SpeechBubbleSnap) {
        // Keep one active bubble per speaker so no single bot floods the 5-bubble cap
        self.speech_bubbles
            .retain(|(_, b)| b.speaker_id != bubble.speaker_id);
        self.speech_bubbles.push_back((self.sim_tick + 120, bubble));
        while self.speech_bubbles.len() > 5 {
            self.speech_bubbles.pop_front();
        }
    }

    fn build_acoustic_request_json(&self, pending: Option<&PendingChat>) -> String {
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let surf_z = self.surf_layer() as f32;
        let tick = self.sim_tick;
        let listener = self
            .players
            .values()
            .find(|p| !p.is_bot)
            .map(|p| (p.x, p.y, p.z))
            .unwrap_or((gw * 0.5, gh * 0.5, surf_z));

        let mut bots: Vec<&PlayerEntity> = self.players.values().filter(|p| p.is_bot).collect();
        bots.sort_by_key(|p| p.id);

        let mut req = String::with_capacity(1536 + bots.len() * 192);
        req.push_str("{\"bots\":[");
        for (i, p) in bots.iter().enumerate() {
            if i > 0 {
                req.push(',');
            }
            let hp_frac = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
            let shield_wall_active = tick < p.shield_wall_until;
            let burning = tick < p.status_burn_until;
            req.push_str(&format!(
                "{{\"id\":{},\"name\":{},\"team\":{},\"class_id\":{},\"bot_class\":{},\"squad_id\":{},\"x\":{:.2},\"y\":{:.2},\"z\":{:.2},\"bot_z\":{:.2},\"hp_frac\":{:.2},\"bot_hp\":{},\"carrying_relic\":{},\"in_tunnel\":{},\"in_water\":{},\"airborne\":{},\"shield_wall_active\":{},\"burning\":{}}}",
                p.id,
                json_escape_str(&p.name),
                p.team,
                p.class_id,
                json_escape_str(PlayerEntity::archetype_name(p.class_id)),
                p.squad_id,
                p.x,
                p.y,
                p.z,
                p.z,
                hp_frac,
                p.hp.max(0),
                if p.carrying_relic { "true" } else { "false" },
                if p.in_tunnel { "true" } else { "false" },
                if p.in_water { "true" } else { "false" },
                if p.airborne { "true" } else { "false" },
                if shield_wall_active { "true" } else { "false" },
                if burning { "true" } else { "false" }
            ));
        }
        let epoch_name = self.volume.evolution_epoch_name();
        let bname = biome_name(self.biome_id);
        req.push_str(&format!(
            "],\"listener\":[{:.2},{:.2},{:.2}],\"grid_w\":{},\"grid_h\":{},\"grid_d\":{},\"epoch\":{},\"biome\":{},\"recent_events\":[",
            listener.0,
            listener.1,
            listener.2,
            self.grid_w,
            self.grid_h,
            self.grid_d,
            json_escape_str(epoch_name),
            json_escape_str(bname)
        ));
        let ev_start = self.tactical_events.len().saturating_sub(5);
        for (i, ev) in self.tactical_events.iter().skip(ev_start).enumerate() {
            if i > 0 {
                req.push(',');
            }
            req.push_str(&format!(
                "{{\"label\":{},\"x\":{:.2},\"y\":{:.2},\"team\":{}}}",
                json_escape_str(&ev.label),
                ev.x,
                ev.y,
                ev.team
            ));
        }
        req.push_str("],\"chat\":");
        if let Some(c) = pending {
            req.push_str(&format!(
                "{{\"text\":{},\"x\":{:.2},\"y\":{:.2},\"layer\":{:.2},\"volume\":{:.2},\"team\":{}}}",
                json_escape_str(&c.text),
                c.x,
                c.y,
                c.layer,
                c.volume,
                c.team
            ));
        } else {
            req.push_str("null");
        }
        req.push('}');
        req
    }

    /// Calls `op=6` (`OP_ACOUSTIC_STEP`) on `CudaRpcClient` (~10 Hz or immediately on `ChatCommand`),
    /// ingesting emergent bot speech bubbles, per-bot `g_i` command overrides, `acoustic_ms`, and `sparse_pct`.
    fn step_acoustics_rpc(&mut self) {
        let pending = self.pending_chat.take();
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let surf_z = self.surf_layer() as f32;

        let listener = self
            .players
            .values()
            .find(|p| !p.is_bot)
            .map(|p| (p.x, p.y, p.z))
            .unwrap_or((gw * 0.5, gh * 0.5, surf_z));

        let bots: Vec<(u32, String, u8, u8, f32, f32, f32, f32, bool)> = self
            .players
            .values()
            .filter(|p| p.is_bot)
            .map(|p| {
                (
                    p.id,
                    p.name.clone(),
                    p.team,
                    p.class_id,
                    p.x,
                    p.y,
                    p.z,
                    (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0),
                    p.carrying_relic,
                )
            })
            .collect();

        if self.bot_rpc.is_none() {
            // Lightweight offline/refsim fallback when a human sends a chat command without CUDA RPC
            if let Some(chat) = pending {
                self.apply_local_chat_fallback(&chat, &bots, listener);
            }
            return;
        }

        // Build UTF-8 JSON request payload with live tactical events, epoch, biome, and 5-archetype telemetry
        let req = self.build_acoustic_request_json(pending.as_ref());

        let rpc_res = if let Some(ref mut rpc) = self.bot_rpc {
            rpc.call_json(6, &req)
        } else {
            return;
        };

        let (resp_str, rpc_ms) = match rpc_res {
            Ok(v) => v,
            Err(_) => {
                if let Some(chat) = pending {
                    self.apply_local_chat_fallback(&chat, &bots, listener);
                }
                return;
            }
        };

        let Some(root) = JsonParser::parse(&resp_str) else {
            return;
        };
        self.acoustic_ms = root
            .get("acoustic_ms")
            .and_then(|v| v.as_f32())
            .unwrap_or(rpc_ms);
        if let Some(sp) = root.get("sparse_pct").and_then(|v| v.as_f32()) {
            if sp.is_finite() && sp > 0.0 {
                self.world_sparse_pct = sp.clamp(0.0, 100.0);
            }
        }
        if let Some(slm) = root.get("system2_slm_active").and_then(|v| v.as_bool()) {
            self.system2_slm_active = slm;
        }

        let tick_now = self.sim_tick;
        if let Some(bubbles) = root.get("bubbles").and_then(|v| v.as_arr()) {
            for b in bubbles {
                let text = b.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
                if text.is_empty() {
                    continue;
                }
                let speaker_id = b.get("speaker_id").and_then(|v| v.as_i64()).unwrap_or(0) as u32;
                let speaker_name = b
                    .get("speaker_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("BOT")
                    .to_string();
                let team = b.get("team").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
                let x = b.get("x").and_then(|v| v.as_f32()).unwrap_or(listener.0);
                let y = b.get("y").and_then(|v| v.as_f32()).unwrap_or(listener.1);
                let z = b.get("z").and_then(|v| v.as_f32()).unwrap_or(surf_z);
                let snr_db = b.get("snr_db").and_then(|v| v.as_f32()).unwrap_or(10.0);
                let symbol_id = b.get("symbol_id").and_then(|v| v.as_i64()).unwrap_or(0) as u8;
                self.push_speech_bubble(SpeechBubbleSnap {
                    speaker_id,
                    speaker_name,
                    team,
                    x,
                    y,
                    z,
                    snr_db,
                    symbol_id,
                    text: text.chars().take(64).collect(),
                    tick: tick_now as u32,
                });
            }
        }

        if let Some(overrides) = root.get("overrides").and_then(|v| v.as_arr()) {
            for ov in overrides {
                let bot_id = ov.get("bot_id").and_then(|v| v.as_i64()).unwrap_or(0) as u32;
                if bot_id == 0 {
                    continue;
                }
                let move_x = ov.get("move_x").and_then(|v| v.as_f32()).unwrap_or(0.0);
                let move_y = ov.get("move_y").and_then(|v| v.as_f32()).unwrap_or(0.0);
                let elem = ov
                    .get("elem")
                    .and_then(|v| v.as_i64())
                    .and_then(|e| if (0..=2).contains(&e) { Some(e as u8) } else { None });
                let ability = ov
                    .get("ability")
                    .and_then(|v| v.as_i64())
                    .and_then(|a| if (0..=5).contains(&a) { Some(a as u8) } else { None });
                let snr_db = ov.get("snr_db").and_then(|v| v.as_f32()).unwrap_or(10.0);
                let ttl = ov.get("ttl_ticks").and_then(|v| v.as_i64()).unwrap_or(60).clamp(15, 180) as u64;
                self.bot_overrides.insert(
                    bot_id,
                    BotCommandOverride {
                        move_x,
                        move_y,
                        elem,
                        ability,
                        snr_db,
                        expires_tick: tick_now + ttl,
                    },
                );
            }
        }
    }

    fn apply_local_chat_fallback(
        &mut self,
        chat: &PendingChat,
        bots: &[(u32, String, u8, u8, f32, f32, f32, f32, bool)],
        listener: (f32, f32, f32),
    ) {
        let txt = chat.text.to_ascii_lowercase();
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let elem = if txt.contains("freeze") || txt.contains("frost") || txt.contains("ice") {
            Some(1u8)
        } else if txt.contains("oil") || txt.contains("slick") {
            Some(2u8)
        } else if txt.contains("fire") || txt.contains("burn") || txt.contains("ignite") {
            Some(0u8)
        } else {
            None
        };
        let (tx, ty) = if txt.contains("north") {
            (gw * 0.50, gh * 0.28)
        } else if txt.contains("south") {
            (gw * 0.50, gh * 0.72)
        } else if txt.contains("east") {
            (gw * 0.75, gh * 0.50)
        } else if txt.contains("west") {
            (gw * 0.25, gh * 0.50)
        } else {
            (gw * 0.50, gh * 0.50)
        };
        let src_spl = 56.0 + 38.0 * chat.volume.clamp(0.1, 1.0);
        let z_scale = if self.is_z16() { 1.2 } else { 6.0 };
        let tick_now = self.sim_tick;
        let mut replies = 0usize;
        for &(bid, ref bname, bteam, _, bx, by, bz, _, _) in bots {
            if bteam != chat.team {
                continue;
            }
            let dist = (bx - chat.x).hypot(by - chat.y).hypot((bz - chat.layer) * z_scale);
            let snr = src_spl - 20.0 * (dist * 0.45 + 1.0).max(1.0).log10() - 42.0;
            if snr >= 0.0 {
                let dx = tx - bx;
                let dy = ty - by;
                let mag = dx.hypot(dy).max(1e-3);
                self.bot_overrides.insert(
                    bid,
                    BotCommandOverride {
                        move_x: dx / mag,
                        move_y: dy / mag,
                        elem,
                        ability: Some(2),
                        snr_db: snr,
                        expires_tick: tick_now + 60,
                    },
                );
                if replies < 2 {
                    let ldist = (bx - listener.0).hypot(by - listener.1).hypot((bz - listener.2) * z_scale);
                    let lsnr = 88.0 - 20.0 * (ldist * 0.45 + 1.0).max(1.0).log10() - 42.0;
                    self.push_speech_bubble(SpeechBubbleSnap {
                        speaker_id: bid,
                        speaker_name: bname.clone(),
                        team: bteam,
                        x: bx,
                        y: by,
                        z: bz,
                        snr_db: (lsnr * 10.0).round() / 10.0,
                        symbol_id: elem.unwrap_or(0) + 1,
                        text: format!("COPY! MOVING ON ORDER"),
                        tick: tick_now as u32,
                    });
                    replies += 1;
                }
            }
        }
    }

    fn spawn_point(&self, team: u8, salt: u32) -> (f32, f32, f32) {
        let wf = self.grid_w as f32;
        let hf = self.grid_h as f32;
        if self.is_z16() {
            let z_scale = (self.grid_d as f32 / 16.0).clamp(1.0, 2.0);
            let top_scan = (self.grid_d as f32 - 2.2).max(12.5);
            let off = ((salt % 13) as f32 - 6.0) * (hf * 0.016);
            let (sx, sy, scan_z) = if self.biome_id == field_core::BIOME_NORDIC_VIADUCT && salt % 4 == 3 {
                if team == 0 {
                    // Near the East mouth of the vaulted basalt cave
                    (wf * 0.375, hf * 0.700, 4.5 * z_scale)
                } else {
                    // East bridge approach overlooking the 3-span stone & timber bridge
                    (wf * 0.715, hf * 0.500, top_scan)
                }
            } else if team == 0 {
                (wf * (0.20 + (salt % 3) as f32 * 0.025), hf * 0.50 + off, top_scan)
            } else {
                (wf * (0.80 - (salt % 3) as f32 * 0.025), hf * 0.50 + off, top_scan)
            };
            let sz = self.find_z16_surface_z(sx, sy, scan_z);
            return (sx, sy, sz);
        }
        let surf_z = self.surf_layer() as f32;
        // Put every 3rd hero onto the Layer 2 Highland Plateau (when depth >= 3) for immediate 3D verticality!
        if self.grid_d >= 3 && salt % 3 == 1 {
            if team == 0 {
                return (wf * 0.28, hf * 0.28, 2.0);
            } else {
                return (wf * 0.72, hf * 0.28, 2.0);
            }
        }
        let off = ((salt % 13) as f32 - 6.0) * (hf * 0.024);
        if team == 0 {
            (wf * 0.12, hf * 0.50 + off, surf_z)
        } else {
            (wf * 0.88, hf * 0.50 + off, surf_z)
        }
    }

    fn push_combo_callout(&mut self, slot: usize, x: f32, y: f32, team: u8, label: &'static str) {
        let tick = self.sim_tick;
        let idx = slot.min(9);
        if tick < self.last_combo_tick[idx].saturating_add(45) {
            return;
        }
        self.last_combo_tick[idx] = tick;
        self.push_event(2, x, y, 0, team, label.to_string());
    }

    fn spawn_player(&mut self, name: String, team: u8, class_id: u8, is_bot: bool) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let cls = class_id % 5;
        let (x, y, mut z) = self.spawn_point(team % 2, id);
        if cls == CLASS_SKYWING {
            // Skywings launch slightly aloft for immediate 3D flight!
            z = (z + if self.is_z16() { 2.5 } else { 0.8 }).min((self.grid_d as f32 - 2.0).max(1.5));
        }
        let (mhp, mmp, mst, _) = PlayerEntity::class_stats(cls, 1);
        self.players.insert(
            id,
            PlayerEntity {
                id,
                name,
                team: team % 2,
                class_id: cls,
                squad_id: 0,
                is_bot,
                x,
                y,
                z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                desired_vz: 0.0,
                aim_x: if team % 2 == 0 { 1.0 } else { -1.0 },
                aim_y: 0.0,
                hp: mhp,
                max_hp: mhp,
                mana: mmp,
                max_mana: mmp,
                stamina: mst,
                max_stamina: mst,
                level: 1,
                xp: 0,
                kills: 0,
                deaths: 0,
                score: 0,
                dash_until: 0,
                dash_cd: 0,
                melee_until: 0,
                melee_cd: 0,
                bolt_cd: 0,
                ability_e_cd: 0,
                ult_r_cd: 0,
                ult_until: 0,
                shield_until: 0,
                haste_until: 0,
                oiled_until: 0,
                chilled_until: 0,
                carrying_relic: false,
                ack_input_seq: 0,
                echo_client_time_ms: 0,
                kb_vx: 0.0,
                kb_vy: 0.0,
                last_phys_blast_field_tick: 0,
                burrow_ambush_until: 0,
                status_burn_until: 0,
                shield_wall_until: 0,
                quench_cd_until: 0,
                smoke_veil_active: false,
                in_tunnel: false,
                in_water: false,
                airborne: cls == CLASS_SKYWING,
            },
        );
        id
    }

    fn refresh_hazard_grids(&mut self) {
        let len = self.grid_w * self.grid_h;
        for z in 0..self.grid_d {
            let f = &self.volume.layers[z];
            let hz_g = &mut self.hazard_layers[z];
            for idx in 0..len {
                let (_, hz) = f.encode_cell_display_and_hazard(idx);
                hz_g[idx] = hz;
            }
        }
    }

    #[inline]
    fn sample_hazard(&self, x: f32, y: f32, layer: usize) -> u8 {
        let l = layer.min(self.grid_d - 1);
        let xi = (x.floor() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
        let yi = (y.floor() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
        self.hazard_layers[l][yi * self.grid_w + xi]
    }

    /// True if Layer 2 has solid walkable deck/plateau matter (`soil`, `wood` deck, `ice`) at `(x, y)`.
    #[inline]
    fn layer2_has_walkable_deck(&self, x: f32, y: f32) -> bool {
        if self.grid_d < 3 {
            return false;
        }
        let xi = (x.floor() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
        let yi = (y.floor() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
        let f = &self.volume.layers[2];
        let idx = f.idx(xi, yi);
        let hz = self.hazard_layers[2][idx];
        (hz & (1 << 0)) != 0 && (hz & (1 << 1)) == 0 && f.cell_volume(idx) >= 220
    }

    /// Discrete per-cell solid density matching client `world_bridge.rs` 3×3 spaced tree trunks 1:1.
    #[inline]
    fn cell_solid_density(&self, xi: i32, yi: i32, layer: usize, allow_water: bool) -> f32 {
        let x = xi.clamp(0, (self.grid_w - 1) as i32) as usize;
        let y = yi.clamp(0, (self.grid_h - 1) as i32) as usize;
        let surf_l = self.surf_layer();
        if layer >= 2 && self.grid_d >= 3 {
            let f2 = &self.volume.layers[2];
            let idx = y * self.grid_w + x;
            if f2.mass[SP_ROCK][idx] >= 500 {
                return 1.0;
            }
            if f2.mass[SP_WOOD][idx] >= 650 {
                return if x % 3 == 0 && y % 3 == 0 { 1.0 } else { 0.0 };
            }
            let hz2 = self.hazard_layers[2][idx];
            if (hz2 & (1 << 0)) != 0 && (hz2 & (1 << 1)) == 0 && f2.cell_volume(idx) >= 220 {
                return 0.0;
            }
            // Open air over surface: block only solid surface rock cliffs
            let f1 = &self.volume.layers[surf_l];
            return if f1.mass[SP_ROCK][idx] >= 500 { 1.0 } else { 0.0 };
        }
        let f = &self.volume.layers[surf_l];
        let idx = y * self.grid_w + x;
        if f.mass[SP_ROCK][idx] >= 500 {
            return 1.0;
        }
        if f.mass[SP_WOOD][idx] >= 650 {
            return if x % 3 == 0 && y % 3 == 0 { 1.0 } else { 0.0 };
        }
        if !allow_water {
            let hz = self.hazard_layers[surf_l][idx];
            let walkable = (hz & (1 << 0)) != 0;
            let wall = (hz & (1 << 1)) != 0;
            let water_slow = (hz & (1 << 4)) != 0;
            if water_slow && !walkable && !wall {
                return 0.85;
            }
        }
        0.0
    }

    /// Continuous bilinear solid-density field at `(x, y, layer)`.
    #[inline]
    fn sample_solid_density(&self, x: f32, y: f32, layer: usize, allow_water: bool) -> f32 {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor() as i32;
        let y0 = fy.floor() as i32;
        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let d00 = self.cell_solid_density(x0, y0, layer, allow_water);
        let d10 = self.cell_solid_density(x0 + 1, y0, layer, allow_water);
        let d01 = self.cell_solid_density(x0, y0 + 1, layer, allow_water);
        let d11 = self.cell_solid_density(x0 + 1, y0 + 1, layer, allow_water);
        let d0 = d00 + (d10 - d00) * tx;
        let d1 = d01 + (d11 - d01) * tx;
        d0 + (d1 - d0) * ty
    }

    /// Hero capsule solid density (`R = 0.65`, center + 4 perimeter probes at `0.55`).
    #[inline]
    fn sample_capsule_density(&self, x: f32, y: f32, layer: usize, allow_water: bool) -> f32 {
        const PROBE_R: f32 = 0.55;
        let c0 = self.sample_solid_density(x, y, layer, allow_water);
        let pe = self.sample_solid_density(x + PROBE_R, y, layer, allow_water);
        let pw = self.sample_solid_density(x - PROBE_R, y, layer, allow_water);
        let ps = self.sample_solid_density(x, y + PROBE_R, layer, allow_water);
        let pn = self.sample_solid_density(x, y - PROBE_R, layer, allow_water);
        0.40 * c0 + 0.15 * (pe + pw + ps + pn)
    }

    /// Sample active Time-Dilation Dome `dt_scale` at `(x, y)` (`1.0` ambient, `0.15` Stasis, `3.0` Haste).
    #[inline]
    fn sample_time_dome_scale(&self, x: f32, y: f32) -> f32 {
        for dome in self.time_domes.iter().rev() {
            let d2 = (x - dome.x).powi(2) + (y - dome.y).powi(2);
            if d2 <= dome.radius * dome.radius {
                return dome.dt_scale;
            }
        }
        1.0
    }

    /// Build continuous `[H, W]` `local_dt` field for `WorldModelDT` (`op=9`) and fallback flux scaling.
    fn build_local_dt_map(&self) -> Vec<f32> {
        let len = self.grid_w * self.grid_h;
        let mut dt_map = vec![1.0f32; len];
        for dome in &self.time_domes {
            let r = dome.radius;
            let r_inner = (r - 1.5).max(1.0);
            let x0 = ((dome.x - r).floor() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
            let x1 = ((dome.x + r).ceil() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
            let y0 = ((dome.y - r).floor() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
            let y1 = ((dome.y + r).ceil() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
            for y in y0..=y1 {
                let dy = (y as f32 + 0.5) - dome.y;
                let row = y * self.grid_w;
                for x in x0..=x1 {
                    let dx = (x as f32 + 0.5) - dome.x;
                    let dist = dx.hypot(dy);
                    if dist <= r_inner {
                        dt_map[row + x] = dome.dt_scale;
                    } else if dist < r {
                        let t = ((dist - r_inner) / (r - r_inner)).clamp(0.0, 1.0);
                        let blend = 0.5 - 0.5 * (t * std::f32::consts::PI).cos();
                        dt_map[row + x] = dome.dt_scale + (1.0 - dome.dt_scale) * blend;
                    }
                }
            }
        }
        dt_map
    }

    fn place_time_dome(&mut self, pid: u32, gx: f32, gy: f32, mode: u8) {
        let owner_team = self.players.get(&pid).map(|p| p.team).unwrap_or(0);
        if mode == 0 {
            self.time_domes.clear();
            let (mx, my) = (self.grid_w as f32 * 0.5, self.grid_h as f32 * 0.5);
            self.push_event(5, mx, my, 0, owner_team, "TIME DOMES CLEARED!".into());
            return;
        }
        let cx = if gx.is_finite() {
            gx.clamp(8.0, (self.grid_w.saturating_sub(8)) as f32)
        } else {
            self.grid_w as f32 * 0.5
        };
        let cy = if gy.is_finite() {
            gy.clamp(8.0, (self.grid_h.saturating_sub(8)) as f32)
        } else {
            self.grid_h as f32 * 0.5
        };
        let dt_scale = if mode == 2 { 3.0f32 } else { 0.15f32 };
        if self.time_domes.len() >= 4 {
            self.time_domes.remove(0);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.time_domes.push(TimeDomeEntity {
            id,
            x: cx,
            y: cy,
            radius: 16.0,
            dt_scale,
            expires_tick: self.sim_tick + 540, // 18s at 30 Hz
            owner_team,
        });
        let lbl = if mode == 2 {
            "HASTE ZONE 3.0X!"
        } else {
            "STASIS DOME 0.15X!"
        };
        self.push_event(2, cx, cy, 0, owner_team, lbl.into());
    }

    fn apply_damage(&mut self, attacker: Option<u32>, victim_id: u32, mut dmg: i16, tag: &str, elem: u8) {
        let tick = self.sim_tick;
        let mut killed_info = None;
        let hit_pos;
        let v_team;
        let mut solo_assassination_evt: Option<(u32, f32, f32, u8)> = None;
        let mut eff_tag = tag;

        // Wave 17C: Solo Emberstalker (+35% ambush damage from Smoke Veil or from behind!)
        if let Some(aid) = attacker {
            if let (Some(a), Some(v)) = (self.players.get(&aid), self.players.get(&victim_id)) {
                if a.id != v.id
                    && a.team != v.team
                    && a.class_id == CLASS_EMBERSTALKER
                    && a.squad_id == 0
                {
                    let rx = v.x - a.x;
                    let ry = v.y - a.y;
                    let dist = rx.hypot(ry).max(1e-3);
                    let from_behind = ((rx / dist) * v.aim_x + (ry / dist) * v.aim_y) >= 0.35;
                    if a.smoke_veil_active || from_behind {
                        dmg = ((dmg as f32) * 1.35).round().max(1.0) as i16;
                        if eff_tag.is_empty() {
                            eff_tag = "AMBUSH!";
                        }
                        let v_isolated = !self.players.values().any(|o| {
                            o.id != v.id
                                && o.team == v.team
                                && (o.x - v.x).hypot(o.y - v.y) <= 14.0
                        });
                        let v_wounded = (v.hp as f32) < (v.max_hp.max(1) as f32) * 0.55;
                        if v_isolated || v_wounded || dmg >= v.hp {
                            solo_assassination_evt = Some((aid, v.x, v.y, a.team));
                        }
                    }
                }
            }
        }

        if let Some(v) = self.players.get_mut(&victim_id) {
            if tick < v.dash_until {
                return;
            }
            if tick < v.shield_until {
                dmg = (dmg / 2).max(1);
            }
            // Wave 17C: Strider Squad Shield-Wall cuts incoming damage to nearby squadmates by 35%
            if tick < v.shield_wall_until {
                dmg = ((dmg as f32) * 0.65).round().max(1.0) as i16;
            }
            if elem == 0 && dmg > 0 {
                v.status_burn_until = v.status_burn_until.max(tick + 90);
            }
            v.hp -= dmg;
            hit_pos = (v.x, v.y);
            v_team = v.team;
            if v.hp <= 0 {
                v.deaths += 1;
                killed_info = Some((v.name.clone(), v.x, v.y, v.team));
            }
        } else {
            return;
        }

        if let Some((aid, sx, sy, ateam)) = solo_assassination_evt {
            if let Some(a) = self.players.get_mut(&aid) {
                a.score = a.score.saturating_add(20);
                a.grant_xp(25);
            }
            self.push_combo_callout(6, sx, sy, ateam, "SOLO ASSASSINATION!");
        }

        if dmg > 0 {
            let lbl = if eff_tag.is_empty() { format!("-{}", dmg) } else { format!("-{} {}", dmg, eff_tag) };
            self.push_event(0, hit_pos.0, hit_pos.1, dmg, v_team, lbl);
        }

        if let Some((v_name, kx, ky, kt)) = killed_info {
            let mut k_name = "WILDFIRE".to_string();
            let mut k_team = 1 - kt;
            if let Some(aid) = attacker {
                if let Some(a) = self.players.get_mut(&aid) {
                    a.kills += 1;
                    a.score = a.score.saturating_add(10);
                    k_name = a.name.clone();
                    k_team = a.team;
                    let leveled = a.grant_xp(55);
                    if a.team == 0 {
                        self.score_west = self.score_west.saturating_add(10);
                    } else {
                        self.score_east = self.score_east.saturating_add(10);
                    }
                    if leveled {
                        let (ax, ay, lvl) = (a.x, a.y, a.level);
                        self.push_event(2, ax, ay, 0, k_team, format!("LEVEL {}!", lvl));
                    }
                }
            }
            self.kill_feed.push_back(KillFeedEntry {
                killer_name: k_name.clone(),
                victim_name: v_name.clone(),
                killer_team: k_team,
                elem,
                tick: tick as u32,
            });
            while self.kill_feed.len() > 5 {
                self.kill_feed.pop_front();
            }
            self.push_event(1, kx, ky, 0, k_team, format!("{} > {}", k_name, v_name));
            self.eco.inject_fear_pulse(kx, ky, 18.0, 0.8);
            self.respawn_player(victim_id);
        }
    }

    fn apply_input(&mut self, pid: u32, inp: &ClientInputDatagram) {
        if self.match_phase != MATCH_PHASE_ACTIVE {
            if let Some(p) = self.players.get_mut(&pid) {
                if inp.seq > p.ack_input_seq || p.is_bot {
                    p.ack_input_seq = inp.seq;
                    p.echo_client_time_ms = inp.client_time_ms;
                }
                p.vx = 0.0;
                p.vy = 0.0;
            }
            return;
        }
        let tick = self.sim_tick;
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let mut melee_hit_check = None;
        let mut spawn_projs: Vec<(u8, u8, f32, f32, f32, f32, f32, f32, f32, f32, u16)> = Vec::new();
        let mut carve_pos = None;
        let mut ability_e_fx = None;
        let mut ult_r_fx = None;

        if let Some(p) = self.players.get_mut(&pid) {
            if inp.seq > p.ack_input_seq || p.is_bot {
                p.ack_input_seq = inp.seq;
                p.echo_client_time_ms = inp.client_time_ms;
            }
            let alen = inp.aim_x.hypot(inp.aim_y);
            if alen > 0.01 {
                p.aim_x = inp.aim_x / alen;
                p.aim_y = inp.aim_y / alen;
            }
            let mlen = inp.move_x.hypot(inp.move_y);
            let (dx, dy) = if mlen > 0.01 { (inp.move_x / mlen, inp.move_y / mlen) } else { (0.0, 0.0) };

            let (_, _, _, base_spd) = PlayerEntity::class_stats(p.class_id, p.level);
            let dash_cd_ticks = match p.class_id {
                CLASS_EMBERSTALKER => 22,
                CLASS_TIDEKIN => 24,
                CLASS_BURROWER => 26,
                CLASS_SKYWING => 28,
                _ => 32,
            };
            if inp.dash && tick >= p.dash_cd && p.stamina >= 25.0 {
                p.stamina -= 25.0;
                p.dash_until = tick + 6;
                p.dash_cd = tick + dash_cd_ticks;
                let (ddx, ddy) = if mlen > 0.01 { (dx, dy) } else { (p.aim_x, p.aim_y) };
                let morph = BodyMorphology::from_archetype(p.class_id);
                let grav_g = self.volume.world_laws.gravity_g.clamp(0.20, 2.5);
                let low_g_boost = (1.0 / grav_g).powf(0.35).clamp(0.75, 1.65);
                p.vx = ddx * 42.0;
                p.vy = ddy * 42.0;
                p.vz = if self.grid_d >= 16 {
                    let base_leap = 5.4 / morph.mass.sqrt()
                        + morph.wing_lift * 2.9
                        + morph.buoyancy * 3.4
                        + morph.climb_grip * 1.9
                        + morph.bore_torque * 1.4;
                    (base_leap * low_g_boost).clamp(4.8, 14.5)
                } else {
                    let base_leap = 1.6 + morph.wing_lift * 1.3 + morph.buoyancy * 0.4 + morph.climb_grip * 0.3;
                    (base_leap * low_g_boost).clamp(1.4, 4.6)
                };
            } else if tick >= p.dash_until {
                let mut speed = if p.carrying_relic { base_spd * 0.78 } else { base_spd };
                if tick < p.haste_until {
                    speed *= 1.32;
                }
                p.vx = dx * speed;
                p.vy = dy * speed;
            }

            if inp.melee && tick >= p.melee_cd {
                p.melee_until = tick + 5;
                p.melee_cd = tick + 15;
                let dmg = match p.class_id {
                    CLASS_STRIDER => 36,
                    CLASS_BURROWER => 34,
                    CLASS_EMBERSTALKER => 30,
                    _ => 24,
                };
                melee_hit_check = Some((p.team, p.class_id, p.x, p.y, p.z, p.aim_x, p.aim_y, dmg + (p.level as i16) * 3, tick <= p.burrow_ambush_until));
            }

            let bolt_cd_ticks = match p.class_id {
                CLASS_EMBERSTALKER => 7,
                CLASS_SKYWING => 8,
                CLASS_TIDEKIN => 9,
                CLASS_BURROWER => 10,
                _ => 11,
            };
            if inp.cast_bolt && tick >= p.bolt_cd && p.mana >= 8.0 {
                p.mana -= 8.0;
                p.bolt_cd = tick + bolt_cd_ticks;
                let pwr = 1.0 + (p.level as f32 - 1.0) * 0.12;
                let vz = if self.grid_d < 16 && p.z >= 1.6 { -0.65 } else { 0.0 }; // High-ground bolts arc gently downward on L=4!
                spawn_projs.push((
                    p.team,
                    inp.bolt_element % 3,
                    p.x + p.aim_x * 1.5,
                    p.y + p.aim_y * 1.5,
                    p.z,
                    p.aim_x * 54.0,
                    p.aim_y * 54.0,
                    vz,
                    1.0,
                    pwr,
                    34,
                ));
            }

            if inp.carve_ice {
                carve_pos = Some((
                    (p.x + p.aim_x * 3.0).clamp(3.0, gw - 4.0) as i32,
                    (p.y + p.aim_y * 3.0).clamp(3.0, gh - 4.0) as i32,
                    p.team,
                ));
            }

            if inp.cast_ability_e && tick >= p.ability_e_cd && p.mana >= 28.0 {
                p.mana -= 28.0;
                p.ability_e_cd = tick + 120;
                let cls = p.class_id;
                match cls {
                    CLASS_STRIDER => {
                        p.shield_until = tick + 90;
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                    CLASS_SKYWING => {
                        for ang_off in [-0.24f32, 0.0, 0.24] {
                            let (ca, sa) = (ang_off.cos(), ang_off.sin());
                            let vx = (p.aim_x * ca - p.aim_y * sa) * 48.0;
                            let vy = (p.aim_x * sa + p.aim_y * ca) * 48.0;
                            spawn_projs.push((p.team, 2, p.x + p.aim_x * 1.5, p.y + p.aim_y * 1.5, p.z, vx, vy, -0.4, 1.3, 1.2, 26));
                        }
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                    CLASS_TIDEKIN => {
                        p.haste_until = tick + 90;
                        carve_pos = Some((
                            (p.x + p.aim_x * 3.2).clamp(3.0, gw - 4.0) as i32,
                            (p.y + p.aim_y * 3.2).clamp(3.0, gh - 4.0) as i32,
                            p.team,
                        ));
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                    CLASS_EMBERSTALKER => {
                        // CLASS_EMBERSTALKER: Smoke Veil + fast Oil/Fire Ambush!
                        p.haste_until = tick + 90;
                        spawn_projs.push((p.team, 2, p.x + p.aim_x * 1.4, p.y + p.aim_y * 1.4, p.z, p.aim_x * 52.0, p.aim_y * 52.0, 0.0, 1.2, 1.1, 22));
                        spawn_projs.push((p.team, 0, p.x + p.aim_x * 1.0, p.y + p.aim_y * 1.0, p.z, p.aim_x * 58.0, p.aim_y * 58.0, 0.0, 1.1, 1.2, 24));
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                    _ => {
                        // CLASS_BURROWER (Terramole): Subterranean Tunnel Bore + Haste!
                        p.haste_until = tick + 75;
                        p.burrow_ambush_until = tick + 120;
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                }
            }

            if inp.cast_ultimate_r && tick >= p.ult_r_cd && p.mana >= 55.0 {
                p.mana -= 55.0;
                p.ult_r_cd = tick + 360;
                p.ult_until = tick + 45;
                if p.class_id == CLASS_STRIDER || p.class_id == CLASS_BURROWER {
                    p.shield_until = tick + 120;
                } else if p.class_id == CLASS_TIDEKIN || p.class_id == CLASS_EMBERSTALKER {
                    p.haste_until = tick + 150;
                }
                if p.class_id == CLASS_BURROWER {
                    p.burrow_ambush_until = tick + 150;
                }
                ult_r_fx = Some((p.class_id, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
            }
        }

        if let Some((team, cls, px, py, pz, ax, ay, dmg, burrow_ambush)) = melee_hit_check {
            let z_tol = if self.is_z16() { 2.2 } else { 0.75 };
            let targets: Vec<u32> = self
                .players
                .values()
                .filter(|t| {
                    if t.id == pid || t.team == team || tick < t.dash_until || (t.z - pz).abs() > z_tol {
                        return false;
                    }
                    let rx = t.x - px;
                    let ry = t.y - py;
                    let dist = rx.hypot(ry);
                    dist <= 6.2 && dist > 0.001 && ((rx / dist) * ax + (ry / dist) * ay) >= 0.42
                })
                .map(|t| t.id)
                .collect();
            let hit_any = !targets.is_empty();
            for tid in targets {
                self.apply_damage(Some(pid), tid, dmg, if burrow_ambush { "AMBUSH" } else { "" }, 3);
            }
            if hit_any && cls == CLASS_BURROWER && burrow_ambush {
                self.push_combo_callout(2, px, py, team, "TUNNEL AMBUSH!");
            }
        }

        for (team, elem, px, py, pz, vx, vy, vz, rad, pwr, ttl) in spawn_projs {
            let id = self.next_id;
            self.next_id += 1;
            self.projectiles.push(ProjectileEntity {
                id,
                owner: pid,
                team,
                element: elem,
                x: px,
                y: py,
                z: pz,
                vx,
                vy,
                vz,
                radius: rad,
                power: pwr,
                ttl,
            });
        }

        if let Some((cls, team, px, py, pz, ax, ay)) = ability_e_fx {
            let pl = (pz.round() as usize).clamp(0, self.grid_d - 1);
            match cls {
                CLASS_STRIDER => {
                    for step in 1..=3 {
                        let tx = (px + ax * (step as f32 * 4.5)).clamp(3.0, gw - 4.0) as i32;
                        let ty = (py + ay * (step as f32 * 4.5)).clamp(3.0, gh - 4.0) as i32;
                        self.record_terrain_engineering_candidates(pid, team, tx as f32, ty as f32, pl, -0.95, 0.0);
                        self.volume.apply_frost_3d(tx, ty, pl, 5, 2.2);
                    }
                    self.push_event(2, px, py, 0, team, "ICE BRIDGE!".into());
                }
                CLASS_SKYWING => self.push_event(2, px, py, 0, team, "OIL BARRAGE!".into()),
                CLASS_TIDEKIN => self.push_event(2, px, py, 0, team, "HYDRO SURGE!".into()),
                CLASS_EMBERSTALKER => {
                    self.volume.apply_physical_actuator(px, py, pl, 0.25, 0.0, 0.35, 0.85, 4.5);
                    self.push_event(2, px, py, 0, team, "SMOKE AMBUSH!".into());
                }
                _ => {
                    // CLASS_BURROWER (Terramole): Conservative Subterranean Bore
                    let moved = self.volume.carve_tunnel_3d(
                        (px + ax * 1.6).clamp(3.0, gw - 4.0),
                        (py + ay * 1.6).clamp(3.0, gh - 4.0),
                        (pz + 0.7).clamp(1.0, (self.grid_d as f32 - 1.5).max(1.0)),
                        ax,
                        ay,
                        2.6,
                    );
                    if moved > 0 {
                        self.tunnels_carved = self.tunnels_carved.saturating_add(1);
                        self.tunnel_mass_displaced = self.tunnel_mass_displaced.saturating_add(moved as u64);
                    }
                    self.push_event(2, px, py, 0, team, "SUBTERRANEAN BORE!".into());
                }
            }
        }

        if let Some((cls, team, px, py, pz, ax, ay)) = ult_r_fx {
            let pl = (pz.round() as usize).clamp(0, self.grid_d - 1);
            match cls {
                CLASS_STRIDER => {
                    self.volume.layers[pl].apply_glacial_nova(px.clamp(4.0, gw - 5.0) as i32, py.clamp(4.0, gh - 5.0) as i32, 11);
                    let targets: Vec<u32> = self
                        .players
                        .values()
                        .filter(|t| t.team != team && (t.x - px).hypot(t.y - py) <= 13.0)
                        .map(|t| t.id)
                        .collect();
                    for tid in targets {
                        if let Some(t) = self.players.get_mut(&tid) {
                            t.chilled_until = tick + 90;
                        }
                        self.apply_damage(Some(pid), tid, 32, "NOVA", 1);
                    }
                    self.push_event(2, px, py, 0, team, "GLACIAL NOVA!".into());
                }
                CLASS_SKYWING => {
                    let tx = (px + ax * 16.0).clamp(6.0, gw - 7.0);
                    let ty = (py + ay * 16.0).clamp(6.0, gh - 7.0);
                    self.volume.layers[pl].apply_meteor_impact(tx as i32, ty as i32, 8);
                    let targets: Vec<u32> = self
                        .players
                        .values()
                        .filter(|t| t.team != team && (t.x - tx).hypot(t.y - ty) <= 11.5)
                        .map(|t| t.id)
                        .collect();
                    for tid in targets {
                        self.apply_damage(Some(pid), tid, 42, "METEOR", 0);
                    }
                    self.eco.inject_fear_pulse(tx, ty, 24.0, 1.2);
                    if pz >= self.surf_layer() as f32 + 1.5 {
                        self.push_combo_callout(0, tx, ty, team, "AERIAL OIL-IGNITION!");
                    }
                    self.push_event(2, tx, ty, 0, team, "METEOR STORM!".into());
                }
                CLASS_TIDEKIN => {
                    let cx = (px + ax * 10.0).clamp(5.0, gw - 6.0) as i32;
                    let cy = (py + ay * 10.0).clamp(5.0, gh - 6.0) as i32;
                    let boiled = self.volume.apply_steam_burst_3d(cx, cy, pl, 8);
                    for p in self.players.values_mut() {
                        if p.team == team && (p.x - px).hypot(p.y - py) <= 24.0 {
                            p.haste_until = tick + 150;
                            p.hp = (p.hp + 20).min(p.max_hp);
                        }
                    }
                    let phys_p = self.volume.sample_entity_physics(px, py, pl);
                    if phys_p.water_frac > 0.10 || boiled >= 4 {
                        self.push_combo_callout(1, px, py, team, "SUBMERGED GEYSER!");
                    }
                    self.push_event(2, px, py, 0, team, "TIDAL SURGE!".into());
                }
                CLASS_EMBERSTALKER => {
                    // CLASS_EMBERSTALKER: Inferno Raid (oil + fire burst + smoke veil)
                    let tx = (px + ax * 10.0).clamp(5.0, gw - 6.0);
                    let ty = (py + ay * 10.0).clamp(5.0, gh - 6.0);
                    self.volume.layers[pl].apply_oil_splash(tx as i32, ty as i32, 6, 1.3);
                    self.volume.layers[pl].apply_fire_bolt(tx as i32, ty as i32, 7, 2.2);
                    let targets: Vec<u32> = self
                        .players
                        .values()
                        .filter(|t| t.team != team && (t.x - tx).hypot(t.y - ty) <= 10.0)
                        .map(|t| t.id)
                        .collect();
                    for tid in targets {
                        self.apply_damage(Some(pid), tid, 38, "AMBUSH!", 0);
                    }
                    self.push_event(2, tx, ty, 0, team, "INFERNO RAID!".into());
                }
                _ => {
                    // CLASS_BURROWER (Terramole): Seismic Breach (3-step tunnel bore + shockwave!)
                    let mut total_moved = 0u32;
                    for step in 1..=3 {
                        let bx = (px + ax * (step as f32 * 2.2)).clamp(3.0, gw - 4.0);
                        let by = (py + ay * (step as f32 * 2.2)).clamp(3.0, gh - 4.0);
                        total_moved = total_moved.saturating_add(self.volume.carve_tunnel_3d(
                            bx,
                            by,
                            (pz + 0.7).clamp(1.0, (self.grid_d as f32 - 1.5).max(1.0)),
                            ax,
                            ay,
                            2.8,
                        ));
                    }
                    if total_moved > 0 {
                        self.tunnels_carved = self.tunnels_carved.saturating_add(1);
                        self.tunnel_mass_displaced =
                            self.tunnel_mass_displaced.saturating_add(total_moved as u64);
                    }
                    let tx = (px + ax * 6.0).clamp(4.0, gw - 5.0);
                    let ty = (py + ay * 6.0).clamp(4.0, gh - 5.0);
                    let targets: Vec<u32> = self
                        .players
                        .values()
                        .filter(|t| t.team != team && (t.x - tx).hypot(t.y - ty) <= 11.0)
                        .map(|t| t.id)
                        .collect();
                    for tid in targets {
                        self.apply_damage(Some(pid), tid, 36, "SEISMIC!", 3);
                    }
                    self.push_combo_callout(2, tx, ty, team, "TUNNEL AMBUSH!");
                    self.push_event(2, tx, ty, 0, team, "SEISMIC BREACH!".into());
                }
            }
        }

        if let Some((cx, cy, team)) = carve_pos {
            let sl = self.surf_layer();
            if let Some(raft) = self.volume.layers[sl].carve_ice_raft(cx, cy, 4) {
                let id = self.next_id;
                self.next_id += 1;
                self.rafts.push(RaftEntity {
                    id,
                    x: raft.center_x,
                    y: raft.center_y,
                    z: sl as f32,
                    vx: 0.0,
                    vy: 3.4,
                    vz: 0.0,
                    radius: raft.radius,
                    ttl: 720,
                });
                self.push_event(2, raft.center_x, raft.center_y, 0, team, "RAFT CARVED!".into());
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_player_physical_actuator(
        &mut self,
        pid: u32,
        dx: f32,
        dy: f32,
        thermal_flux: f32,
        sp_water: f32,
        sp_oil: f32,
        sp_smoke: f32,
        focus_radius: f32,
        impulse: f32,
    ) {
        self.apply_player_physical_actuator_3d(
            pid,
            dx,
            dy,
            None,
            thermal_flux,
            sp_water,
            sp_oil,
            sp_smoke,
            focus_radius,
            impulse,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_player_physical_actuator_3d(
        &mut self,
        pid: u32,
        dx: f32,
        dy: f32,
        target_z_override: Option<f32>,
        thermal_flux: f32,
        sp_water: f32,
        sp_oil: f32,
        sp_smoke: f32,
        focus_radius: f32,
        impulse: f32,
    ) {
        if self.match_phase != MATCH_PHASE_ACTIVE {
            return;
        }
        let dx = if dx.is_finite() { dx.clamp(-36.0, 36.0) } else { 0.0 };
        let dy = if dy.is_finite() { dy.clamp(-36.0, 36.0) } else { 0.0 };
        let tf = if thermal_flux.is_finite() { thermal_flux.clamp(-1.0, 1.0) } else { 0.0 };
        let sw = if sp_water.is_finite() { sp_water.clamp(0.0, 1.0) } else { 0.0 };
        let so = if sp_oil.is_finite() { sp_oil.clamp(0.0, 1.0) } else { 0.0 };
        let ss = if sp_smoke.is_finite() { sp_smoke.clamp(0.0, 1.0) } else { 0.0 };
        let r = if focus_radius.is_finite() { focus_radius.clamp(1.5, 8.5) } else { 4.0 };
        let imp = if impulse.is_finite() { impulse.clamp(0.0, 1.0) } else { 0.0 };

        let activity = tf.abs() + sw + so + ss + imp;
        if activity < 0.04 {
            return;
        }

        let mana_cost = (tf.abs() * 4.5 + (sw + so + ss) * 4.0 + imp * 7.5) * (0.65 + r * 0.08);
        let (px, py, pz, team, class_id, burrow_ambush_until, scale) = {
            let Some(p) = self.players.get_mut(&pid) else {
                return;
            };
            if p.mana < 1.5 {
                return;
            }
            let sc = (p.mana / mana_cost.max(0.1)).clamp(0.0, 1.0);
            p.mana = (p.mana - mana_cost * sc).max(0.0);
            let dlen = dx.hypot(dy);
            if dlen > 0.1 {
                p.aim_x = dx / dlen;
                p.aim_y = dy / dlen;
            }
            (p.x, p.y, p.z, p.team, p.class_id, p.burrow_ambush_until, sc)
        };
        if scale < 0.05 {
            return;
        }

        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let tx = (px + dx).clamp(3.0, gw - 4.0);
        let ty = (py + dy).clamp(3.0, gh - 4.0);
        let max_z1 = (self.grid_d as f32 - 2.0).max(1.0);
        let (layer, z1_beam) = if let Some(tz_req) = target_z_override {
            let z1 = if tz_req.is_finite() {
                tz_req.clamp(1.0, max_z1)
            } else {
                pz.clamp(1.0, max_z1)
            };
            ((z1.round() as usize).clamp(0, self.grid_d - 1), z1)
        } else {
            let mut l = (pz.round() as usize).clamp(0, self.grid_d - 1);
            // Project airborne Skywing actuators downward onto terrain/water surface below when targeting open air
            if self.is_z16() && l > 1 {
                let txi = (tx.round() as usize).min(self.grid_w - 1);
                let tyi = (ty.round() as usize).min(self.grid_h - 1);
                let tidx = tyi * self.grid_w + txi;
                if self.volume.layers[l].cell_volume(tidx) < 25 {
                    for z_scan in (1..l).rev() {
                        if self.volume.layers[z_scan].cell_volume(tidx) >= 40 {
                            l = z_scan;
                            break;
                        }
                    }
                }
            }
            (l, l as f32)
        };

        let tf_s = tf * scale;
        let sw_s = sw * scale;
        let so_s = so * scale;
        let ss_s = ss * scale;
        let imp_s = imp * scale;

        // Pre-sample target & caster voxel physics for emergent cross-archetype combo & terrain-engineering detection
        let txi_c = (tx.round() as usize).min(self.grid_w - 1);
        let tyi_c = (ty.round() as usize).min(self.grid_h - 1);
        let tidx_c = tyi_c * self.grid_w + txi_c;
        let target_has_oil = self.volume.layers[layer].mass[SP_OIL][tidx_c] >= 40;
        let target_has_water = self.volume.layers[layer].mass[SP_WATER][tidx_c] >= 80;
        let pl_caster = (pz.round() as usize).clamp(0, self.grid_d - 1);
        let caster_in_water = self.volume.sample_entity_physics(px, py, pl_caster).water_frac > 0.10;

        self.record_terrain_engineering_candidates(pid, team, tx, ty, layer, tf_s, sw_s);

        self.volume
            .apply_physical_actuator(tx, ty, layer, tf_s, sw_s, so_s, ss_s, r);

        let tick = self.sim_tick;
        let surf_z = self.surf_layer() as f32;
        if class_id == CLASS_SKYWING && pz >= surf_z + 1.5 && tf_s > 0.40 && target_has_oil {
            self.push_combo_callout(0, tx, ty, team, "AERIAL OIL-IGNITION!");
        } else if class_id == CLASS_TIDEKIN
            && caster_in_water
            && ((tf_s > 0.35 && target_has_water) || (sw_s > 0.45 && ss_s > 0.25))
        {
            self.push_combo_callout(1, tx, ty, team, "SUBMERGED GEYSER!");
        }
        self.recent_actuators
            .push_back((pid, team, tx, ty, r + 4.0, tick + 90));
        while self.recent_actuators.len() > 36 {
            self.recent_actuators.pop_front();
        }

        // Record short-lived 3D volumetric actuator pulse VFX (TTL ~8 ticks = 0.25s)
        if tf_s.abs() > 0.15 || so_s > 0.15 || sw_s > 0.15 || ss_s > 0.15 || imp_s > 0.15 {
            let energy = tf_s.abs() * 1.2 + so_s + sw_s + ss_s + imp_s * 1.4;
            self.actuator_pulses.push((
                tick + 8,
                energy,
                ActuatorPulseSnap {
                    x0: px,
                    y0: py,
                    z0: pz,
                    x1: tx,
                    y1: ty,
                    z1: z1_beam,
                    thermal_flux: tf_s,
                    sp_oil: so_s,
                    sp_water: sw_s,
                    sp_smoke: ss_s,
                    focus_radius: r,
                    impulse: imp_s,
                    team,
                },
            ));
            if self.actuator_pulses.len() > 48 {
                self.actuator_pulses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                self.actuator_pulses.truncate(32);
            }
        }

        if tf_s.abs() > 0.45 || imp_s > 0.35 {
            self.eco.inject_fear_pulse(tx, ty, (r * 3.0).clamp(10.0, 24.0), 0.75 * scale);
        }

        // Radial momentum impulse on nearby entities & floating ice rafts
        if imp_s > 0.1 {
            let eff_r = r + 4.5;
            let shock_z_tol = if self.is_z16() { 2.6 } else { 1.1 };
            let mut shock_hits: Vec<(u32, i16)> = Vec::new();
            for t in self.players.values_mut() {
                if t.id == pid || tick < t.dash_until || (t.z - z1_beam).abs() > shock_z_tol {
                    continue;
                }
                let rx = t.x - tx;
                let ry = t.y - ty;
                let dist = rx.hypot(ry);
                if dist <= eff_r {
                    let fall = (1.0 - dist / (eff_r + 0.5)).clamp(0.15, 1.0);
                    let (nx, ny) = if dist > 0.1 {
                        (rx / dist, ry / dist)
                    } else {
                        let dl = dx.hypot(dy).max(1e-3);
                        (dx / dl, dy / dl)
                    };
                    let mut kb_res = if t.class_id == CLASS_STRIDER || t.class_id == CLASS_BURROWER { 0.40 } else { 1.0 };
                    if tick < t.shield_wall_until {
                        kb_res *= 0.30;
                    }
                    let push = 38.0 * imp_s * fall * kb_res;
                    t.kb_vx = (t.kb_vx + nx * push).clamp(-48.0, 48.0);
                    t.kb_vy = (t.kb_vy + ny * push).clamp(-48.0, 48.0);
                    if t.team != team && imp_s > 0.30 && dist <= r + 2.5 {
                        let dmg = (14.0 * imp_s * fall).round().clamp(3.0, 18.0) as i16;
                        shock_hits.push((t.id, dmg));
                    }
                }
            }
            for raft in &mut self.rafts {
                let rx = raft.x - tx;
                let ry = raft.y - ty;
                let dist = rx.hypot(ry);
                if dist <= eff_r + raft.radius {
                    let fall = (1.0 - dist / (eff_r + raft.radius + 1.0)).clamp(0.15, 1.0);
                    let (nx, ny) = if dist > 0.1 { (rx / dist, ry / dist) } else { (0.0, 1.0) };
                    raft.vx = (raft.vx + nx * 20.0 * imp_s * fall).clamp(-26.0, 26.0);
                    raft.vy = (raft.vy + ny * 20.0 * imp_s * fall).clamp(-26.0, 26.0);
                }
            }
            let hit_enemy = !shock_hits.is_empty();
            for (tid, dmg) in shock_hits {
                self.apply_damage(Some(pid), tid, dmg, "SHOCK", 3);
            }
            if hit_enemy && class_id == CLASS_BURROWER && tick <= burrow_ambush_until {
                self.push_combo_callout(2, tx, ty, team, "TUNNEL AMBUSH!");
            }
        }
    }

    fn find_recent_actuator_owner(&self, x: f32, y: f32, victim_team: u8) -> Option<u32> {
        let tick = self.sim_tick;
        for &(pid, team, ax, ay, rad, exp) in self.recent_actuators.iter().rev() {
            if tick <= exp && team != victim_team && (x - ax).hypot(y - ay) <= rad + 4.0 {
                return Some(pid);
            }
        }
        None
    }

    /// Wave 17C: Record conservation-exact Cryo-Bridge (`SP_WATER -> SP_ICE`) and
    /// Basalt Bulwark (magma/oil quench -> `SP_ROCK`) candidates whenever cryo/water actuators fire.
    #[allow(clippy::too_many_arguments)]
    fn record_terrain_engineering_candidates(
        &mut self,
        pid: u32,
        team: u8,
        tx: f32,
        ty: f32,
        layer: usize,
        thermal_flux: f32,
        sp_water: f32,
    ) {
        let l = layer.min(self.grid_d.saturating_sub(1));
        let z_lo = l.saturating_sub(2);
        let z_hi = (l + 1).min(self.grid_d.saturating_sub(1));
        let cx = (tx.round() as i32).clamp(2, (self.grid_w as i32) - 3);
        let cy = (ty.round() as i32).clamp(2, (self.grid_h as i32) - 3);

        let mut water_cells = 0u16;
        let mut l_water_cells = 0u16;
        let mut ice_cells = 0u16;
        let mut rock_cells = 0u16;
        let mut hot_magma_or_oil = false;
        let mut best_water_z = l;
        let mut best_water_cnt = 0u16;
        let mut best_hot_z = l;

        for zs in z_lo..=z_hi {
            let f = &self.volume.layers[zs];
            let mut z_water = 0u16;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let x = (cx + dx).clamp(1, (self.grid_w as i32) - 2) as usize;
                    let y = (cy + dy).clamp(1, (self.grid_h as i32) - 2) as usize;
                    let idx = f.idx(x, y);
                    if f.mass[SP_WATER][idx] >= 120 {
                        water_cells += 1;
                        z_water += 1;
                        if zs == l {
                            l_water_cells += 1;
                        }
                    }
                    if zs == l && f.mass[SP_ICE][idx] >= 180 {
                        ice_cells += 1;
                    }
                    if zs == l && f.mass[SP_ROCK][idx] >= 180 {
                        rock_cells += 1;
                    }
                    let t_ck = f.cell_temp_ck(idx);
                    if t_ck > 50_000 && (f.mass[SP_OIL][idx] >= 12 || f.mass[SP_ROCK][idx] >= 150) {
                        hot_magma_or_oil = true;
                        if zs == l {
                            best_hot_z = l;
                        } else if best_hot_z != l {
                            best_hot_z = zs;
                        }
                    }
                }
            }
            if z_water > best_water_cnt {
                best_water_cnt = z_water;
                best_water_z = zs;
            }
        }
        // Prefer the exact layer being acted upon whenever it has open water
        if l_water_cells >= 3 {
            best_water_z = l;
        }
        // Baseline ice_cells on the chosen water layer
        if best_water_z != l {
            let fw = &self.volume.layers[best_water_z];
            ice_cells = 0;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let x = (cx + dx).clamp(1, (self.grid_w as i32) - 2) as usize;
                    let y = (cy + dy).clamp(1, (self.grid_h as i32) - 2) as usize;
                    if fw.mass[SP_ICE][fw.idx(x, y)] >= 180 {
                        ice_cells += 1;
                    }
                }
            }
        }
        if best_hot_z != l {
            let fh = &self.volume.layers[best_hot_z];
            rock_cells = 0;
            for dy in -3..=3 {
                for dx in -3..=3 {
                    let x = (cx + dx).clamp(1, (self.grid_w as i32) - 2) as usize;
                    let y = (cy + dy).clamp(1, (self.grid_h as i32) - 2) as usize;
                    if fh.mass[SP_ROCK][fh.idx(x, y)] >= 180 {
                        rock_cells += 1;
                    }
                }
            }
        }

        let tick = self.sim_tick;
        if thermal_flux < -0.25 && water_cells >= 3 {
            self.pending_cryo_bridges
                .push_back((pid, team, tx, ty, best_water_z, ice_cells, tick + 120));
            while self.pending_cryo_bridges.len() > 16 {
                self.pending_cryo_bridges.pop_front();
            }
        }
        if (sp_water > 0.25 || thermal_flux < -0.25) && hot_magma_or_oil {
            let near_combat = self
                .players
                .values()
                .any(|o| o.team != team && (o.x - tx).hypot(o.y - ty) <= 32.0)
                || self
                    .shrines
                    .iter()
                    .any(|s| (s.x - tx).hypot(s.y - ty) <= 32.0);
            if near_combat {
                let prev_bf = self.volume.evolution_stats.basalt_forged;
                self.pending_basalt_quenches
                    .push_back((pid, team, tx, ty, best_hot_z, rock_cells, prev_bf, tick + 120));
                while self.pending_basalt_quenches.len() > 16 {
                    self.pending_basalt_quenches.pop_front();
                }
            }
        }
    }

    /// Wave 17C: Inspect conservation-exact `VolumeField` outcomes for newly forged Cryo-Bridges
    /// (`SP_WATER -> SP_ICE` contiguous span) and Basalt Bulwarks (magma/oil quenched into `SP_ROCK`).
    fn check_terrain_engineering_milestones(&mut self) {
        let tick = self.sim_tick;
        let w = self.grid_w;
        let h = self.grid_h;

        // 1. Cryo-Bridge Forging (`CRYO-BRIDGE FORGED!`)
        let old_cryo = std::mem::take(&mut self.pending_cryo_bridges);
        let mut keep_cryo = VecDeque::with_capacity(old_cryo.len());
        let mut cryo_awards: Vec<(u32, u8, f32, f32)> = Vec::new();
        for item in old_cryo {
            let (pid, team, tx, ty, layer, prev_ice, exp) = item;
            if tick > exp {
                continue;
            }
            let l = layer.min(self.grid_d.saturating_sub(1));
            let f = &self.volume.layers[l];
            let cx = (tx.round() as i32).clamp(4, (w as i32) - 5);
            let cy = (ty.round() as i32).clamp(4, (h as i32) - 5);
            let mut ice_grid = [[false; 9]; 9];
            let mut ice_cnt = 0u16;
            for dy in -4i32..=4 {
                for dx in -4i32..=4 {
                    let x = (cx + dx) as usize;
                    let y = (cy + dy) as usize;
                    let idx = f.idx(x, y);
                    if f.mass[SP_ICE][idx] >= 180 {
                        ice_grid[(dy + 4) as usize][(dx + 4) as usize] = true;
                        ice_cnt += 1;
                    }
                }
            }
            // Measure largest 4-connected traversable ice span in the 9x9 window
            let mut visited = [[false; 9]; 9];
            let mut max_span = 0u16;
            let mut stack: Vec<(usize, usize)> = Vec::with_capacity(32);
            for gy in 0..9 {
                for gx in 0..9 {
                    if !ice_grid[gy][gx] || visited[gy][gx] {
                        continue;
                    }
                    visited[gy][gx] = true;
                    stack.push((gx, gy));
                    let mut span = 0u16;
                    while let Some((sx, sy)) = stack.pop() {
                        span += 1;
                        let nbs = [
                            (sx.wrapping_sub(1), sy),
                            (sx + 1, sy),
                            (sx, sy.wrapping_sub(1)),
                            (sx, sy + 1),
                        ];
                        for (nx, ny) in nbs {
                            if nx < 9 && ny < 9 && ice_grid[ny][nx] && !visited[ny][nx] {
                                visited[ny][nx] = true;
                                stack.push((nx, ny));
                            }
                        }
                    }
                    if span > max_span {
                        max_span = span;
                    }
                }
            }
            if ice_cnt >= prev_ice.saturating_add(2) && max_span >= 4 {
                cryo_awards.push((pid, team, tx, ty));
            } else {
                keep_cryo.push_back(item);
            }
        }
        self.pending_cryo_bridges = keep_cryo;
        for (pid, team, tx, ty) in cryo_awards {
            if let Some(p) = self.players.get_mut(&pid) {
                p.score = p.score.saturating_add(35);
                p.grant_xp(40);
            }
            if team == 0 {
                self.score_west = self.score_west.saturating_add(10);
            } else {
                self.score_east = self.score_east.saturating_add(10);
            }
            self.push_combo_callout(7, tx, ty, team, "CRYO-BRIDGE FORGED!");
        }

        // 2. Basalt Bulwark (`BASALT BULWARK!`)
        let cur_bf = self.volume.evolution_stats.basalt_forged;
        let old_basalt = std::mem::take(&mut self.pending_basalt_quenches);
        let mut keep_basalt = VecDeque::with_capacity(old_basalt.len());
        let mut basalt_awards: Vec<(u32, u8, f32, f32)> = Vec::new();
        for item in old_basalt {
            let (pid, team, tx, ty, layer, prev_rock, prev_bf, exp) = item;
            if tick > exp {
                continue;
            }
            let l = layer.min(self.grid_d.saturating_sub(1));
            let f = &self.volume.layers[l];
            let cx = (tx.round() as i32).clamp(4, (w as i32) - 5);
            let cy = (ty.round() as i32).clamp(4, (h as i32) - 5);
            let mut rock_cnt = 0u16;
            for dy in -4i32..=4 {
                for dx in -4i32..=4 {
                    let x = (cx + dx) as usize;
                    let y = (cy + dy) as usize;
                    let idx = f.idx(x, y);
                    if f.mass[SP_ROCK][idx] >= 180 {
                        rock_cnt += 1;
                    }
                }
            }
            if rock_cnt > prev_rock || cur_bf > prev_bf {
                basalt_awards.push((pid, team, tx, ty));
            } else {
                keep_basalt.push_back(item);
            }
        }
        self.pending_basalt_quenches = keep_basalt;
        for (pid, team, tx, ty) in basalt_awards {
            if let Some(p) = self.players.get_mut(&pid) {
                p.score = p.score.saturating_add(35);
                p.grant_xp(40);
            }
            if team == 0 {
                self.score_west = self.score_west.saturating_add(10);
            } else {
                self.score_east = self.score_east.saturating_add(10);
            }
            self.push_combo_callout(8, tx, ty, team, "BASALT BULWARK!");
        }
    }

    /// Wave 17C: Emergent Player-to-Player Synergy Pass (runs for both neural & scripted bots):
    /// - Strider Squad Shield-Wall (`SHIELD WALL!`: 35% dmg reduction + KB resist + visual pulse)
    /// - Tidekin Ally Quench Rescue (`TIDEKIN QUENCH RESCUE!`: physical water surge, clears burn, +22 HP, +25 score)
    /// - Burrower Subterranean Shrine Breach (`SUBTERRANEAN BREACH!`: conservative tunnel exit + shockwave)
    fn run_emergent_synergies(&mut self) {
        let tick = self.sim_tick;
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let is_z16 = self.is_z16();
        let surf_z = self.surf_layer() as f32;

        struct SynSnap {
            id: u32,
            team: u8,
            class_id: u8,
            squad_id: u8,
            x: f32,
            y: f32,
            z: f32,
            burning: bool,
            quench_ready: bool,
            in_tunnel: bool,
            burrow_ambush_until: u64,
        }

        let snaps: Vec<SynSnap> = self
            .players
            .values()
            .map(|p| {
                let pl = (p.z.round() as usize).clamp(0, self.grid_d - 1);
                let xi = (p.x.floor() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
                let yi = (p.y.floor() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
                let local_hot = self.volume.layers[pl].cell_temp_ck(yi * self.grid_w + xi) > 39_000;
                SynSnap {
                    id: p.id,
                    team: p.team,
                    class_id: p.class_id,
                    squad_id: p.squad_id,
                    x: p.x,
                    y: p.y,
                    z: p.z,
                    burning: tick < p.status_burn_until || local_hot,
                    quench_ready: tick >= p.quench_cd_until,
                    in_tunnel: p.in_tunnel,
                    burrow_ambush_until: p.burrow_ambush_until,
                }
            })
            .collect();

        // 1. Strider Squad Shield-Wall (`SHIELD WALL!`)
        let mut shield_wall_members: Vec<u32> = Vec::new();
        let mut shield_wall_pulses: Vec<(u8, f32, f32, f32, bool)> = Vec::new();
        for s in snaps.iter().filter(|p| p.class_id == CLASS_STRIDER) {
            let mut nearby_allies: Vec<u32> = Vec::new();
            for o in &snaps {
                if o.id == s.id || o.team != s.team {
                    continue;
                }
                let same_squad = (s.squad_id > 0 && o.squad_id == s.squad_id) || s.squad_id == 0;
                if same_squad && (o.x - s.x).hypot(o.y - s.y) <= 9.0 {
                    nearby_allies.push(o.id);
                }
            }
            if !nearby_allies.is_empty() {
                shield_wall_members.push(s.id);
                shield_wall_members.extend(nearby_allies);
                let enemy_near = snaps
                    .iter()
                    .any(|e| e.team != s.team && (e.x - s.x).hypot(e.y - s.y) <= 24.0);
                shield_wall_pulses.push((s.team, s.x, s.y, s.z, enemy_near));
            }
        }
        for mid in shield_wall_members {
            if let Some(p) = self.players.get_mut(&mid) {
                p.shield_wall_until = tick + 18;
            }
        }
        for (team, sx, sy, sz, enemy_near) in shield_wall_pulses {
            if enemy_near || tick % 18 == 0 {
                // Visual cyan-blue protective shield ring pulse
                self.actuator_pulses.push((
                    tick + 8,
                    0.95,
                    ActuatorPulseSnap {
                        x0: sx,
                        y0: sy,
                        z0: sz + 0.4,
                        x1: sx + 0.01,
                        y1: sy + 0.01,
                        z1: sz + 0.4,
                        thermal_flux: -0.45,
                        sp_oil: 0.0,
                        sp_water: 0.40,
                        sp_smoke: 0.0,
                        focus_radius: 7.5,
                        impulse: 0.0,
                        team,
                    },
                ));
                if enemy_near {
                    self.push_combo_callout(5, sx, sy, team, "SHIELD WALL!");
                }
            }
        }

        // 2. Tidekin Ally Quench Rescue (`TIDEKIN QUENCH RESCUE!`)
        let mut quench_actions: Vec<(u32, u32, u8, f32, f32, f32, f32, f32, f32)> = Vec::new();
        let mut already_rescued: Vec<u32> = Vec::new();
        for tk in snaps
            .iter()
            .filter(|p| p.class_id == CLASS_TIDEKIN && p.quench_ready)
        {
            let best_ally = snaps
                .iter()
                .filter(|o| {
                    o.id != tk.id
                        && o.team == tk.team
                        && o.burning
                        && !already_rescued.contains(&o.id)
                        && (o.x - tk.x).hypot(o.y - tk.y) <= 16.0
                })
                .min_by(|a, b| {
                    let da = (a.x - tk.x).hypot(a.y - tk.y);
                    let db = (b.x - tk.x).hypot(b.y - tk.y);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(ally) = best_ally {
                already_rescued.push(ally.id);
                quench_actions.push((
                    tk.id, tk.id, tk.team, tk.x, tk.y, ally.x, ally.y, ally.z, ally.id as f32,
                ));
            }
        }
        for (tk_id, _, team, tx, ty, ax, ay, az, ally_id_f) in quench_actions {
            let ally_id = ally_id_f as u32;
            // Physical water surge toward burning teammate (conservation-exact!)
            self.apply_player_physical_actuator_3d(
                tk_id,
                ax - tx,
                ay - ty,
                Some(az),
                -0.35,
                0.90,
                0.0,
                0.25,
                4.0,
                0.0,
            );
            if let Some(ally) = self.players.get_mut(&ally_id) {
                ally.status_burn_until = 0;
                ally.hp = (ally.hp + 22).min(ally.max_hp);
            }
            if let Some(tk) = self.players.get_mut(&tk_id) {
                tk.score = tk.score.saturating_add(25);
                tk.grant_xp(25);
                tk.quench_cd_until = tick + 45;
            }
            if team == 0 {
                self.score_west = self.score_west.saturating_add(5);
            } else {
                self.score_east = self.score_east.saturating_add(5);
            }
            self.push_combo_callout(4, ax, ay, team, "TIDEKIN QUENCH RESCUE!");
        }

        // 3. Burrower Subterranean Shrine Undermine (`SUBTERRANEAN BREACH!`)
        let shrine_list: Vec<(f32, f32, i8)> =
            self.shrines.iter().map(|s| (s.x, s.y, s.owner)).collect();
        let mut breaches: Vec<(u32, u8, f32, f32, f32)> = Vec::new();
        for br in snaps.iter().filter(|p| p.class_id == CLASS_BURROWER) {
            let underground =
                br.in_tunnel || (is_z16 && br.z <= 5.2) || tick <= br.burrow_ambush_until;
            if !underground {
                continue;
            }
            let desired_owner = if br.team == 0 { -1 } else { 1 };
            let under_enemy_shrine = shrine_list
                .iter()
                .any(|&(sx, sy, owner)| owner != desired_owner && (sx - br.x).hypot(sy - br.y) <= 10.5);
            if !under_enemy_shrine {
                continue;
            }
            let defender_near = snaps
                .iter()
                .any(|e| e.team != br.team && (e.x - br.x).hypot(e.y - br.y) <= 14.0);
            if defender_near {
                breaches.push((br.id, br.team, br.x, br.y, br.z));
            }
        }
        for (bid, team, bx, by, bz) in breaches {
            let max_z = (self.grid_d as f32 - 2.0).max(1.5);
            let moved = self.volume.carve_tunnel_3d(
                bx.clamp(3.0, gw - 4.0),
                by.clamp(3.0, gh - 4.0),
                (bz + 1.4).clamp(1.2, max_z),
                0.0,
                0.0,
                2.4,
            );
            if moved > 0 {
                self.tunnels_carved = self.tunnels_carved.saturating_add(1);
                self.tunnel_mass_displaced =
                    self.tunnel_mass_displaced.saturating_add(moved as u64);
            }
            if let Some(p) = self.players.get_mut(&bid) {
                p.vz = 9.5;
                p.desired_vz = 6.0;
                p.burrow_ambush_until = tick + 120;
                p.score = p.score.saturating_add(20);
                p.grant_xp(25);
            }
            self.apply_player_physical_actuator_3d(
                bid,
                0.0,
                0.0,
                Some(surf_z.max(bz)),
                0.20,
                0.0,
                0.0,
                0.65,
                5.5,
                0.95,
            );
            self.push_combo_callout(9, bx, by, team, "SUBTERRANEAN BREACH!");
        }
    }

    /// Compute 2D spatial gradient away from a local temperature or steam peak for physical blast knockback.
    fn voxel_blast_dir(&self, x: f32, y: f32, layer: usize, use_steam: bool) -> (f32, f32) {
        let l = layer.min(self.grid_d - 1);
        let f = &self.volume.layers[l];
        let xi = (x.floor() as i32).clamp(1, (self.grid_w as i32) - 2) as usize;
        let yi = (y.floor() as i32).clamp(1, (self.grid_h as i32) - 2) as usize;
        let sample = |sx: usize, sy: usize| -> f32 {
            let idx = f.idx(sx, sy);
            if use_steam {
                f.mass[SP_STEAM][idx] as f32
            } else {
                f.cell_temp_ck(idx) as f32
            }
        };
        let gx = sample(xi + 1, yi) - sample(xi - 1, yi);
        let gy = sample(xi, yi + 1) - sample(xi, yi - 1);
        // Push DOWN the gradient (away from peak hotspot / steam core)
        let mag = gx.hypot(gy);
        if mag > 1.0 {
            (-gx / mag, -gy / mag)
        } else {
            let ang = ((xi.wrapping_mul(73) ^ yi.wrapping_mul(151)) % 360) as f32 * 0.01745;
            (ang.cos(), ang.sin())
        }
    }

    fn respawn_player(&mut self, pid: u32) {
        let Some((team, carrying, rx, ry)) = self
            .players
            .get(&pid)
            .map(|p| (p.team, p.carrying_relic, p.x, p.y))
        else {
            return;
        };
        if carrying {
            self.relic_carrier = None;
            self.relic_x = rx;
            self.relic_y = ry;
        }
        let (sx, sy, sz) = self.spawn_point(team, self.sim_tick as u32 + pid);
        if let Some(p) = self.players.get_mut(&pid) {
            p.carrying_relic = false;
            p.x = sx;
            p.y = sy;
            p.z = sz;
            p.vx = 0.0;
            p.vy = 0.0;
            p.vz = 0.0;
            p.hp = p.max_hp;
            p.mana = p.max_mana;
            p.stamina = p.max_stamina;
            p.oiled_until = 0;
            p.chilled_until = 0;
            p.kb_vx = 0.0;
            p.kb_vy = 0.0;
            p.last_phys_blast_field_tick = 0;
            p.burrow_ambush_until = 0;
            p.status_burn_until = 0;
            p.shield_wall_until = 0;
            p.quench_cd_until = 0;
            p.smoke_veil_active = false;
            p.in_tunnel = false;
            p.in_water = false;
            p.airborne = p.class_id == CLASS_SKYWING;
        }
    }

    fn reset_match(&mut self) {
        let (w, h, d) = (self.grid_w, self.grid_h, self.grid_d);
        self.volume = VolumeField::new_biome_3d(w, h, d, self.biome_id, self.world_seed);
        self.last_epoch_name = self.volume.evolution_epoch_name();
        self.tunnels_carved = 0;
        self.tunnel_mass_displaced = 0;
        self.last_combo_tick = [0; 10];
        self.tactical_events.clear();
        self.pending_cryo_bridges.clear();
        self.pending_basalt_quenches.clear();
        self.eco.reset();
        self.refresh_hazard_grids();
        self.shrines = GameState::build_shrines(w, h);
        self.score_west = 0;
        self.score_east = 0;
        self.match_phase = MATCH_PHASE_ACTIVE;
        self.match_time_left_s = MATCH_DURATION_SECS;
        self.match_start_instant = Instant::now();
        self.celebration_started = None;
        self.mvp_name.clear();
        self.mvp_score = 0;
        self.relic_carrier = None;
        self.relic_x = (w as f32) * 0.50;
        self.relic_y = (h as f32) * 0.50;
        self.projectiles.clear();
        self.rafts.clear();
        self.events.clear();
        self.kill_feed.clear();
        self.speech_bubbles.clear();
        self.bot_overrides.clear();
        self.recent_actuators.clear();
        self.actuator_pulses.clear();
        self.time_domes.clear();
        self.pending_chat = None;
        self.weather_kind = WEATHER_CLEAR;

        let pids: Vec<u32> = self.players.keys().copied().collect();
        for pid in pids {
            let team = self.players[&pid].team;
            let (sx, sy, sz) = self.spawn_point(team, pid);
            if let Some(p) = self.players.get_mut(&pid) {
                let (mhp, mmp, mst, _) = PlayerEntity::class_stats(p.class_id, 1);
                p.x = sx;
                p.y = sy;
                p.z = sz;
                p.vx = 0.0;
                p.vy = 0.0;
                p.vz = 0.0;
                p.aim_x = if p.team == 0 { 1.0 } else { -1.0 };
                p.aim_y = 0.0;
                p.level = 1;
                p.xp = 0;
                p.kills = 0;
                p.deaths = 0;
                p.score = 0;
                p.max_hp = mhp;
                p.hp = mhp;
                p.max_mana = mmp;
                p.mana = mmp;
                p.max_stamina = mst;
                p.stamina = mst;
                p.dash_until = 0;
                p.dash_cd = 0;
                p.melee_until = 0;
                p.melee_cd = 0;
                p.bolt_cd = 0;
                p.ability_e_cd = 0;
                p.ult_r_cd = 0;
                p.ult_until = 0;
                p.shield_until = 0;
                p.haste_until = 0;
                p.oiled_until = 0;
                p.chilled_until = 0;
                p.carrying_relic = false;
                p.kb_vx = 0.0;
                p.kb_vy = 0.0;
                p.last_phys_blast_field_tick = 0;
                p.burrow_ambush_until = 0;
                p.status_burn_until = 0;
                p.shield_wall_until = 0;
                p.quench_cd_until = 0;
                p.smoke_veil_active = false;
                p.in_tunnel = false;
                p.in_water = false;
                p.airborne = p.class_id == CLASS_SKYWING;
            }
        }
        for peer in self.peers.values_mut() {
            peer.needs_full_sync = true;
        }
        if let Some(ref ws) = self.ws_bridge {
            ws.queue_full_sync_all();
        }
        self.push_event(5, (w as f32) * 0.5, (h as f32) * 0.5, 0, 2, "NEW MATCH STARTED!".into());
        log::info!(
            "Match cleanly reset ({}x{}x{}, biome={} [{}], seed={}, {} players)",
            w,
            h,
            d,
            self.biome_id,
            biome_name(self.biome_id),
            self.world_seed,
            self.players.len()
        );
    }

    fn cycle_biome(&mut self) {
        self.biome_id = (self.biome_id + 1) % NUM_BIOMES;
        let bname = biome_name(self.biome_id);
        if self.grid_d >= 16 {
            self.volume.init_biome_3d(self.biome_id, self.world_seed);
            self.last_epoch_name = self.volume.evolution_epoch_name();
            self.tunnels_carved = 0;
            self.tunnel_mass_displaced = 0;
            self.eco.reset();
            self.refresh_hazard_grids();
            self.projectiles.clear();
            self.rafts.clear();
            self.time_domes.clear();
            self.mass_error = 0;
            self.energy_error = 0;
            // Respawn all players safely on valid 3D terrain via surface scan
            let pids: Vec<u32> = self.players.keys().copied().collect();
            for pid in pids {
                let team = self.players[&pid].team;
                let (sx, sy, sz) = self.spawn_point(team, self.sim_tick as u32 + pid);
                if let Some(p) = self.players.get_mut(&pid) {
                    p.x = sx;
                    p.y = sy;
                    p.z = sz;
                    p.vx = 0.0;
                    p.vy = 0.0;
                    p.vz = 0.0;
                    p.kb_vx = 0.0;
                    p.kb_vy = 0.0;
                }
            }
        }
        let banner = format!("BIOME: {}!", bname.to_ascii_uppercase());
        self.push_event(5, (self.grid_w as f32) * 0.5, (self.grid_h as f32) * 0.5, 0, 2, banner);
        for peer in self.peers.values_mut() {
            peer.needs_full_sync = true;
        }
        if let Some(ref ws) = self.ws_bridge {
            ws.queue_full_sync_all();
        }
        log::info!(
            "Cycled 3D biome -> {} ({}) seed={} ({}x{}x{}, full brick re-sync queued for {} peers)",
            self.biome_id,
            bname,
            self.world_seed,
            self.grid_w,
            self.grid_h,
            self.grid_d,
            self.peers.len()
        );
    }

    fn cycle_evolution_speed(&mut self) {
        self.evo_speed = if (self.evo_speed - 1.0).abs() < 0.05 {
            4.0
        } else if (self.evo_speed - 4.0).abs() < 0.05 {
            12.0
        } else {
            1.0
        };
        let ev = self.volume.evolution_stats;
        let epoch = self.volume.evolution_epoch_name();
        let lbl = format!(
            "WORLD EVOLUTION: {:.0}X | {} (BASALT:{} TREES:{})",
            self.evo_speed, epoch, ev.basalt_forged, ev.saplings_grown
        );
        let (mx, my) = (self.grid_w as f32 * 0.5, self.grid_h as f32 * 0.5);
        self.push_event(5, mx, my, 0, 2, lbl);
    }

    fn cycle_world_law(&mut self) {
        let next = (self.volume.world_law_preset + 1) % NUM_WORLD_LAW_PRESETS;
        self.volume.set_world_law_preset(next);
        let laws = self.volume.world_laws;
        let name = WorldLaws::preset_name(next);
        let (mx, my) = (self.grid_w as f32 * 0.5, self.grid_h as f32 * 0.5);
        let banner = format!("WORLD LAW: {}!", name);
        self.push_event(5, mx, my, 0, 2, banner.clone());
        self.tactical_events.push_back(ws_bridge::WsTacticalEvent {
            label: banner,
            team: 2,
            x: mx,
            y: my,
            tick: self.sim_tick,
        });
        while self.tactical_events.len() > 16 {
            self.tactical_events.pop_front();
        }
        log::info!(
            "World Law Preset -> {} ({}) [g={:.2} dT={:+.0} vol={:.2} visc={:.2} precip={:.2} k={:.2}]",
            next,
            name,
            laws.gravity_g,
            laws.ambient_temp_shift,
            laws.volatility,
            laws.viscosity,
            laws.precipitation,
            laws.thermal_cond
        );
    }

    fn step_bots(&mut self) {
        if self.neural_bots_on && self.step_bots_neural_z() {
            return;
        }
        self.bot_brains.clear();
        self.step_bots_scripted();
    }

    /// Batched Shared-Latent Neural Bots (`op=5` inspect with `4x4` Z_world saliency, graceful fallback to `op=4`):
    /// Packs compact ledger/entity tokens (`168` f32/bot) and sends to GB10 `gpu_infer_daemon.py`,
    /// which samples `WorldModel`'s cached `[4, 48, 256, 256]` VRAM trunk `Z_world` + `Z_eco` ZERO-COPY!
    fn step_bots_neural_z(&mut self) -> bool {
        const SELF_DIM: usize = 24;
        const NUM_TOKENS: usize = 12;
        const TOKEN_DIM: usize = 12;
        const BOT_FEAT_DIM: usize = SELF_DIM + NUM_TOKENS * TOKEN_DIM; // 168
        const ACTION_OUT_DIM: usize = 13;
        const SALIENCY_DIM: usize = 16;
        const INSPECT_OUT_DIM: usize = ACTION_OUT_DIM + SALIENCY_DIM; // 29

        let bot_ids: Vec<u32> = self.players.values().filter(|p| p.is_bot).map(|p| p.id).collect();
        let n = bot_ids.len();
        if n == 0 || self.bot_rpc.is_none() {
            return false;
        }

        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let gd_m1 = (self.grid_d.saturating_sub(1)).max(1) as f32;
        let tick = self.sim_tick;
        let all_p: Vec<PlayerEntity> = self.players.values().cloned().collect();

        let mut feats = vec![0.0f32; n * BOT_FEAT_DIM];
        let mut nearest_enemy_ids = vec![0u32; n];
        for (bi, &bid) in bot_ids.iter().enumerate() {
            let Some(p) = self.players.get(&bid) else { continue; };
            let base = bi * BOT_FEAT_DIM;
            // Self features [24]
            feats[base] = p.x / gw;
            feats[base + 1] = p.y / gh;
            feats[base + 2] = (p.z / gd_m1).clamp(0.0, 1.0);
            feats[base + 3] = (p.vx / 40.0).clamp(-1.0, 1.0);
            feats[base + 4] = (p.vy / 40.0).clamp(-1.0, 1.0);
            feats[base + 5] = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
            feats[base + 6] = (p.mana / p.max_mana.max(1.0)).clamp(0.0, 1.0);
            feats[base + 7] = (p.stamina / p.max_stamina.max(1.0)).clamp(0.0, 1.0);
            feats[base + 8 + (p.class_id as usize % 3)] = 1.0;
            feats[base + 11] = p.team as f32;
            feats[base + 12] = if p.carrying_relic { 1.0 } else { 0.0 };
            feats[base + 13] = p.level as f32 / 5.0;
            feats[base + 14] = if p.class_id == CLASS_EMBERSTALKER { 1.0 } else { 0.0 };
            feats[base + 15] = (p.squad_id as f32 / 8.0).clamp(0.0, 1.0);
            feats[base + 16] = if p.class_id == CLASS_BURROWER { 1.0 } else { 0.0 };
            feats[base + 20] = (p.dash_cd.saturating_sub(tick) as f32 / 32.0).clamp(0.0, 1.0);
            feats[base + 21] = (p.ability_e_cd.saturating_sub(tick) as f32 / 120.0).clamp(0.0, 1.0);
            feats[base + 22] = (p.ult_r_cd.saturating_sub(tick) as f32 / 360.0).clamp(0.0, 1.0);
            feats[base + 23] = self.weather_kind as f32 / 3.0;

            // Nearest 8 other players -> tokens 0..8
            let mut others: Vec<(f32, &PlayerEntity)> = all_p
                .iter()
                .filter(|o| o.id != p.id)
                .map(|o| ((o.x - p.x).hypot(o.y - p.y), o))
                .collect();
            others.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(&(_, enemy)) = others.iter().find(|(_, o)| o.team != p.team) {
                nearest_enemy_ids[bi] = enemy.id;
            }
            let z_band_scale = 3.0 / gd_m1;
            for (ti, &(d, o)) in others.iter().take(8).enumerate() {
                let tb = base + SELF_DIM + ti * TOKEN_DIM;
                feats[tb] = ((o.x - p.x) / 80.0).clamp(-1.0, 1.0);
                feats[tb + 1] = ((o.y - p.y) / 80.0).clamp(-1.0, 1.0);
                feats[tb + 2] = (((o.z - p.z) * z_band_scale) / 2.0).clamp(-1.0, 1.0);
                feats[tb + 3] = if o.team != p.team { 1.0 } else { 0.0 };
                feats[tb + 4] = (o.hp as f32 / o.max_hp.max(1) as f32).clamp(0.0, 1.0);
                feats[tb + 5] = if o.carrying_relic { 1.0 } else { 0.0 };
                let d_safe = d.max(1e-3);
                feats[tb + 6] = o.class_id as f32 / 2.0;
                feats[tb + 7] = (d / 100.0).min(1.0);
                feats[tb + 8] = (o.x - p.x) / d_safe;
                feats[tb + 9] = (o.y - p.y) / d_safe;
                feats[tb + 11] = 1.0;
            }
            // 3 Shrines -> tokens 8..11
            for (si, s) in self.shrines.iter().take(3).enumerate() {
                let tb = base + SELF_DIM + (8 + si) * TOKEN_DIM;
                let need = if p.team == 0 { s.progress > -0.8 } else { s.progress < 0.8 };
                feats[tb] = ((s.x - p.x) / 100.0).clamp(-1.0, 1.0);
                feats[tb + 1] = ((s.y - p.y) / 100.0).clamp(-1.0, 1.0);
                feats[tb + 3] = if need { 1.0 } else { 0.0 };
                feats[tb + 4] = s.progress;
                feats[tb + 11] = 1.0;
            }
            // Golden Relic -> token 11
            let tr = base + SELF_DIM + 11 * TOKEN_DIM;
            feats[tr] = ((self.relic_x - p.x) / 100.0).clamp(-1.0, 1.0);
            feats[tr + 1] = ((self.relic_y - p.y) / 100.0).clamp(-1.0, 1.0);
            feats[tr + 3] = if p.carrying_relic { 1.0 } else { 0.0 };
            feats[tr + 11] = 1.0;
        }

        const ACTUATOR_OUT_DIM: usize = 10;
        const ACTUATOR_INSPECT_DIM: usize = ACTUATOR_OUT_DIM + SALIENCY_DIM; // 26
        const ACTUATOR_V3_OUT_DIM: usize = 12;
        const ACTUATOR_V3_INSPECT_DIM: usize = ACTUATOR_V3_OUT_DIM + SALIENCY_DIM; // 28

        if self.physics_native {
            let act_res = if let Some(ref mut rpc) = self.bot_rpc {
                rpc.infer_actuators_v3(n, BOT_FEAT_DIM, &feats)
            } else {
                return false;
            };
            if let Ok((out_buf, gpu_ms, is_v3)) = act_res {
                let expected_stride = if is_v3 { ACTUATOR_V3_INSPECT_DIM } else { ACTUATOR_INSPECT_DIM };
                if out_buf.len() == n * expected_stride {
                    self.bot_infer_ms = gpu_ms;
                    let is_z16 = self.is_z16();
                    let max_z1 = (self.grid_d as f32 - 2.0).max(1.0);
                    let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };
                    let mut brains = Vec::with_capacity(n);

                    for (bi, &bid) in bot_ids.iter().enumerate() {
                        let row = &out_buf[bi * expected_stride..(bi + 1) * expected_stride];
                        let (
                            mut move_x,
                            mut move_y,
                            move_jump,
                            mut aim_dx,
                            mut aim_dy,
                            aim_dz,
                            mut thermal_flux,
                            mut sp_water,
                            mut sp_oil,
                            mut sp_smoke,
                            focus_raw,
                            impulse_or_dome,
                            sal_off,
                        ) = if is_v3 {
                            (
                                row[0],
                                row[1],
                                row[2].clamp(-1.0, 1.0),
                                row[3],
                                row[4],
                                row[5].clamp(-1.0, 1.0),
                                row[6].clamp(-1.0, 1.0),
                                row[7].clamp(0.0, 1.0),
                                row[8].clamp(0.0, 1.0),
                                row[9].clamp(0.0, 1.0),
                                row[10],
                                row[11].clamp(-1.0, 1.0),
                                ACTUATOR_V3_OUT_DIM,
                            )
                        } else {
                            (
                                row[0],
                                row[1],
                                0.0f32,
                                row[2],
                                row[3],
                                0.0f32,
                                row[4].clamp(-1.0, 1.0),
                                row[5].clamp(0.0, 1.0),
                                row[6].clamp(0.0, 1.0),
                                row[7].clamp(0.0, 1.0),
                                row[8],
                                row[9].clamp(0.0, 1.0),
                                ACTUATOR_OUT_DIM,
                            )
                        };
                        let focus_radius = if focus_raw <= 1.0 {
                            1.5 + focus_raw.clamp(0.0, 1.0) * 7.0
                        } else {
                            focus_raw.clamp(1.5, 8.5)
                        };
                        let mut impulse = impulse_or_dome.clamp(0.0, 1.0);

                        if let Some(&ov) = self.bot_overrides.get(&bid) {
                            if tick <= ov.expires_tick {
                                if ov.move_x.hypot(ov.move_y) > 0.05 {
                                    move_x = ov.move_x;
                                    move_y = ov.move_y;
                                    aim_dx = ov.move_x;
                                    aim_dy = ov.move_y;
                                }
                                if let Some(oe) = ov.elem {
                                    match oe % 3 {
                                        0 => { thermal_flux = 0.95; }
                                        1 => { thermal_flux = -0.95; }
                                        _ => { sp_oil = 0.90; }
                                    }
                                }
                                if let Some(ab) = ov.ability {
                                    match ab {
                                        0 => { sp_smoke = sp_smoke.max(0.65); impulse = impulse.max(0.75); }
                                        1 | 5 => { impulse = impulse.max(0.90); }
                                        3 => { sp_water = sp_water.max(0.75); }
                                        _ => {}
                                    }
                                }
                            }
                        }

                        // Scale normalized emit vector [-1, 1] to physical target offset (up to 36 canonical cells)
                        let aim_mag = aim_dx.hypot(aim_dy);
                        let (tdx, tdy) = if aim_mag <= 1.5 {
                            (aim_dx * 28.0, aim_dy * 28.0)
                        } else {
                            (aim_dx.clamp(-36.0, 36.0), aim_dy.clamp(-36.0, 36.0))
                        };

                        // True 12-channel 3D neural verticality (`op=10` on Z >= 16):
                        // - `move_jump`: Skywing vertical flight lift, Emberstalker cliff climbing,
                        //   Tidekin river breaches, and Burrower conservative tunnel carving!
                        // - `aim_dz`: True diagonal 3D actuator beams (`z0 -> z1`)!
                        // - `impulse_or_dome < -0.50`: Autonomous bot `0.15x` Stasis TimeDome deployment!
                        let mut target_z_override: Option<f32> = None;
                        let mut neural_dome_req: Option<(u8, f32, f32)> = None;
                        let mut neural_burrow_req: Option<(f32, f32, f32, f32, f32)> = None;
                        if is_v3 && is_z16 {
                            if let Some(p) = self.players.get_mut(&bid) {
                                let z1 = (p.z + aim_dz * 6.0).clamp(1.0, max_z1);
                                target_z_override = Some(z1);
                                match p.class_id {
                                    CLASS_SKYWING => {
                                        p.desired_vz = (move_jump * 7.5).clamp(-3.8, 8.5);
                                    }
                                    CLASS_EMBERSTALKER => {
                                        if move_jump > 0.20 {
                                            p.desired_vz = (move_jump * 8.5).clamp(2.0, 9.5);
                                            p.vz = p.vz.max(6.4 + move_jump * 2.8);
                                        } else {
                                            p.desired_vz = 0.0;
                                        }
                                    }
                                    CLASS_TIDEKIN => {
                                        p.desired_vz = (move_jump * 6.5).clamp(-4.5, 9.0);
                                        if move_jump > 0.50 {
                                            p.vz = p.vz.max(8.2);
                                        }
                                    }
                                    CLASS_BURROWER => {
                                        p.desired_vz = 0.0;
                                        if move_jump.abs() > 0.20 {
                                            let mlen = move_x.hypot(move_y);
                                            let (bx_d, by_d) = if mlen > 0.05 {
                                                (move_x / mlen, move_y / mlen)
                                            } else {
                                                (p.aim_x, p.aim_y)
                                            };
                                            let z_bias = (p.z + 0.75 + move_jump * 0.9).clamp(1.0, max_z1);
                                            neural_burrow_req = Some((
                                                (p.x + bx_d * 1.15).clamp(3.0, gw - 4.0),
                                                (p.y + by_d * 1.15).clamp(3.0, gh - 4.0),
                                                z_bias,
                                                bx_d,
                                                by_d,
                                            ));
                                        }
                                    }
                                    _ => {
                                        p.desired_vz = 0.0;
                                        if move_jump > 0.65 {
                                            p.vz = p.vz.max(6.0);
                                        }
                                    }
                                }
                                if impulse_or_dome < -0.50
                                    && p.mana >= 32.0
                                    && tick >= p.ability_e_cd
                                    && self.time_domes.len() < 4
                                    && tick >= self.last_squad_dome_tick[p.team as usize].saturating_add(360)
                                {
                                    p.mana -= 32.0;
                                    p.ability_e_cd = tick + 240;
                                    let dtx = (p.x + tdx).clamp(8.0, gw - 8.0);
                                    let dty = (p.y + tdy).clamp(8.0, gh - 8.0);
                                    neural_dome_req = Some((p.team, dtx, dty));
                                }
                            }
                        }

                        if let Some((bx, by, bz, dx_b, dy_b)) = neural_burrow_req {
                            let moved = self.volume.carve_tunnel_3d(bx, by, bz, dx_b, dy_b, 1.9);
                            if moved > 0 {
                                self.tunnels_carved = self.tunnels_carved.saturating_add(1);
                                self.tunnel_mass_displaced =
                                    self.tunnel_mass_displaced.saturating_add(moved as u64);
                                if let Some(p) = self.players.get_mut(&bid) {
                                    p.burrow_ambush_until = tick + 90;
                                }
                            }
                        }
                        if let Some((dteam, dtx, dty)) = neural_dome_req {
                            self.last_squad_dome_tick[dteam as usize] = tick;
                            self.place_time_dome(bid, dtx, dty, 1);
                            self.push_combo_callout(3, dtx, dty, dteam, "STASIS RESCUE!");
                        }

                        let move_inp = ClientInputDatagram {
                            seq: tick as u32,
                            move_x,
                            move_y,
                            aim_x: aim_dx,
                            aim_y: aim_dy,
                            dash: impulse > 0.78 || (is_v3 && move_jump > 0.72),
                            melee: false,
                            cast_bolt: false,
                            bolt_element: 0,
                            carve_ice: false,
                            cast_ability_e: false,
                            cast_ultimate_r: false,
                            client_time_ms: 0,
                        };
                        self.apply_input(bid, &move_inp);
                        self.apply_player_physical_actuator_3d(
                            bid,
                            tdx,
                            tdy,
                            target_z_override,
                            thermal_flux,
                            sp_water,
                            sp_oil,
                            sp_smoke,
                            focus_radius,
                            impulse,
                        );

                        if let Some(p) = self.players.get(&bid) {
                            let mut z_heatmap_4x4 = [0u8; 16];
                            for k in 0..SALIENCY_DIM {
                                z_heatmap_4x4[k] = to_u8(row[sal_off + k]);
                            }
                            let elem_probs = [
                                to_u8(thermal_flux.max(0.0)),
                                to_u8((-thermal_flux).max(0.0)),
                                to_u8(sp_oil.max(sp_water).max(sp_smoke)),
                            ];
                            let ability_probs = [
                                to_u8(sp_smoke),
                                to_u8(impulse),
                                to_u8(thermal_flux.abs()),
                                to_u8(sp_water),
                                to_u8(sp_oil),
                                to_u8(((focus_radius - 2.0) / 6.0).clamp(0.0, 1.0)),
                            ];
                            brains.push(BotBrainSnap {
                                bot_id: bid,
                                class_id: p.class_id,
                                team: p.team,
                                target_id: nearest_enemy_ids[bi],
                                elem_probs,
                                ability_probs,
                                z_heatmap_4x4,
                            });
                        }
                    }
                    self.bot_brains = brains;
                    return true;
                }
            }
        }

        // Try op=5 (OP_BOT_POLICY_INSPECT: 13 actions + 16 saliency floats), fallback gracefully to op=4
        let (out_buf, gpu_ms, stride, has_saliency) = if let Some(ref mut rpc) = self.bot_rpc {
            match rpc.call_batched(5, 1, 1, n as u16, BOT_FEAT_DIM as u16, &feats) {
                Ok((v, ms)) if v.len() == n * INSPECT_OUT_DIM => (v, ms, INSPECT_OUT_DIM, true),
                _ => match rpc.call_batched(4, 1, 1, n as u16, BOT_FEAT_DIM as u16, &feats) {
                    Ok((v, ms)) if v.len() == n * ACTION_OUT_DIM => (v, ms, ACTION_OUT_DIM, false),
                    _ => return false,
                },
            }
        } else {
            return false;
        };
        self.bot_infer_ms = gpu_ms;

        let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };
        let mut brains = Vec::with_capacity(n);

        for (bi, &bid) in bot_ids.iter().enumerate() {
            let row = &out_buf[bi * stride..(bi + 1) * stride];
            let a = &row[..ACTION_OUT_DIM];
            let elem = if a[10] >= a[11] && a[10] >= a[12] {
                0
            } else if a[11] >= a[12] {
                1
            } else {
                2
            };
            let mut inp = ClientInputDatagram {
                seq: tick as u32,
                move_x: a[0],
                move_y: a[1],
                aim_x: a[2],
                aim_y: a[3],
                dash: a[4] > 0.55,
                melee: a[5] > 0.50,
                cast_bolt: a[6] > 0.48,
                carve_ice: a[7] > 0.55,
                cast_ability_e: a[8] > 0.55,
                cast_ultimate_r: a[9] > 0.55,
                bolt_element: elem,
                client_time_ms: 0,
            };
            let mut elem_probs = [to_u8(a[10]), to_u8(a[11]), to_u8(a[12])];
            let mut ability_probs = [
                to_u8(a[4]),
                to_u8(a[5]),
                to_u8(a[6]),
                to_u8(a[7]),
                to_u8(a[8]),
                to_u8(a[9]),
            ];
            // Apply live acoustic human command override (g_i) if this bot physically heard the order!
            if let Some(&ov) = self.bot_overrides.get(&bid) {
                if tick <= ov.expires_tick {
                    if ov.move_x.hypot(ov.move_y) > 0.05 {
                        inp.move_x = ov.move_x;
                        inp.move_y = ov.move_y;
                        inp.aim_x = ov.move_x;
                        inp.aim_y = ov.move_y;
                    }
                    if let Some(oe) = ov.elem {
                        inp.bolt_element = oe % 3;
                        elem_probs = [35, 35, 35];
                        elem_probs[(oe % 3) as usize] = 245;
                        if tick % 9 == 0 {
                            inp.cast_bolt = true;
                            ability_probs[2] = ability_probs[2].max(220);
                        }
                    }
                    if let Some(ab) = ov.ability {
                        match ab {
                            0 => { inp.dash = true; ability_probs[0] = 230; }
                            1 => { inp.melee = true; ability_probs[1] = 230; }
                            2 => { inp.cast_bolt = true; ability_probs[2] = 235; }
                            3 => { inp.carve_ice = true; ability_probs[3] = 230; }
                            4 => { inp.cast_ability_e = true; ability_probs[4] = 240; }
                            5 => { inp.cast_ultimate_r = true; ability_probs[5] = 245; }
                            _ => {}
                        }
                    }
                }
            }
            if let Some(p) = self.players.get(&bid) {
                let mut z_heatmap_4x4 = [0u8; 16];
                if has_saliency {
                    for k in 0..SALIENCY_DIM {
                        z_heatmap_4x4[k] = to_u8(row[ACTION_OUT_DIM + k]);
                    }
                }
                brains.push(BotBrainSnap {
                    bot_id: bid,
                    class_id: p.class_id,
                    team: p.team,
                    target_id: nearest_enemy_ids[bi],
                    elem_probs,
                    ability_probs,
                    z_heatmap_4x4,
                });
            }
            self.apply_input(bid, &inp);
        }
        self.bot_brains = brains;
        true
    }

    fn step_bots_scripted(&mut self) {
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let is_z16 = self.is_z16();
        let max_fly_z = (self.grid_d as f32 - 2.0).max(2.6);
        let tick = self.sim_tick;

        struct BotSnap {
            id: u32,
            team: u8,
            class_id: u8,
            squad_id: u8,
            x: f32,
            y: f32,
            z: f32,
            hp_frac: f32,
            carrying_relic: bool,
            veiled: bool,
            burning: bool,
        }

        let bot_ids: Vec<u32> = self.players.values().filter(|p| p.is_bot).map(|p| p.id).collect();
        let snap: Vec<BotSnap> = self
            .players
            .values()
            .map(|p| {
                let pl = (p.z.round() as usize).clamp(0, self.grid_d - 1);
                let phys = self.volume.sample_entity_physics(p.x, p.y, pl);
                let veiled = p.class_id == CLASS_EMBERSTALKER
                    && (phys.smoke_opacity > 0.15
                        || (is_z16 && self.volume.measure_vertical_clearance_3d((p.x, p.y, p.z)) < 6.0));
                let burning = tick < p.status_burn_until || phys.temp_c > 116.85;
                BotSnap {
                    id: p.id,
                    team: p.team,
                    class_id: p.class_id,
                    squad_id: p.squad_id,
                    x: p.x,
                    y: p.y,
                    z: p.z,
                    hp_frac: (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0),
                    carrying_relic: p.carrying_relic,
                    veiled,
                    burning,
                }
            })
            .collect();
        let shrine_goals: Vec<(f32, f32, i8)> = self.shrines.iter().map(|s| (s.x, s.y, s.owner)).collect();

        // Precompute per-(team, squad_id) centroids, lowest-HP ally, and shared dynamic squad objectives!
        let mut squad_sums: HashMap<(u8, u8), (f32, f32, u32)> = HashMap::new();
        let mut squad_wounded: HashMap<(u8, u8), (f32, f32, f32)> = HashMap::new();
        let mut team_relic_carrier: [Option<(f32, f32)>; 2] = [None, None];
        for s in &snap {
            if s.carrying_relic {
                team_relic_carrier[s.team as usize] = Some((s.x, s.y));
            }
            if s.squad_id > 0 {
                let entry = squad_sums.entry((s.team, s.squad_id)).or_insert((0.0, 0.0, 0));
                entry.0 += s.x;
                entry.1 += s.y;
                entry.2 += 1;
                if s.hp_frac < 0.45 {
                    let w = squad_wounded.entry((s.team, s.squad_id)).or_insert((s.x, s.y, s.hp_frac));
                    if s.hp_frac < w.2 {
                        *w = (s.x, s.y, s.hp_frac);
                    }
                }
            }
        }

        let mut pending_rescue_domes: Vec<(u32, u8, f32, f32)> = Vec::new();
        let mut pending_desired_vz: Vec<(u32, f32)> = Vec::new();

        for bid in bot_ids {
            let Some(me) = snap.iter().find(|s| s.id == bid) else {
                continue;
            };
            let (team, class_id, squad_id, mx, my, mz, carrying) =
                (me.team, me.class_id, me.squad_id, me.x, me.y, me.z, me.carrying_relic);
            let desired_owner = if team == 0 { -1 } else { 1 };

            // --- 1. Target Selection (Squads vs Solo Emberstalker Raiders + Smoke Veil Aggro Reduction) ---
            let mut best_enemy: Option<&BotSnap> = None;
            let mut best_enemy_score = f32::MAX;
            let mut nearby_enemy_cnt = 0usize;
            let mut enemy_cluster_cx = 0.0f32;
            let mut enemy_cluster_cy = 0.0f32;

            for cand in &snap {
                if cand.team == team {
                    continue;
                }
                let d = (cand.x - mx).hypot(cand.y - my);
                if d < 18.0 {
                    nearby_enemy_cnt += 1;
                    enemy_cluster_cx += cand.x;
                    enemy_cluster_cy += cand.y;
                }
                // Smoke Veil passive: Emberstalkers in smoke/caves drop enemy aggro beyond 14m!
                if cand.veiled && d > 14.0 {
                    continue;
                }
                let mut score = d;
                if class_id == CLASS_EMBERSTALKER {
                    // Wave 17C: Solo Emberstalkers hunt isolated (>14m from nearest ally) or wounded (HP < 55%) enemies!
                    let cand_allies = snap
                        .iter()
                        .filter(|o| o.id != cand.id && o.team == cand.team && (o.x - cand.x).hypot(o.y - cand.y) <= 14.0)
                        .count();
                    score += (cand_allies as f32) * 20.0;
                    if cand_allies == 0 {
                        score -= 12.0;
                    }
                    if cand.hp_frac < 0.55 {
                        score -= 14.0;
                    }
                } else if class_id == CLASS_SKYWING {
                    // Wave 17C: Airborne Skywings prioritize enemy clusters currently engaged by an allied Strider (<10m)!
                    let engaged_by_strider = snap.iter().any(|ally| {
                        ally.team == team
                            && ally.class_id == CLASS_STRIDER
                            && (ally.x - cand.x).hypot(ally.y - cand.y) < 10.0
                    });
                    if engaged_by_strider {
                        score -= 22.0;
                    }
                } else if cand.carrying_relic {
                    score -= 16.0;
                }
                if score < best_enemy_score {
                    best_enemy_score = score;
                    best_enemy = Some(cand);
                }
            }

            let (ax, ay, edist, enemy_pos_z) = if let Some(e) = best_enemy {
                (e.x - mx, e.y - my, (e.x - mx).hypot(e.y - my), Some((e.x, e.y, e.z)))
            } else {
                (gw * 0.5 - mx, gh * 0.5 - my, 999.0, None)
            };

            // --- 2. Dynamic Objective Steering: Squads (Alpha/Bravo/Charlie) vs Solo Emberstalkers ---
            let (mut tx, mut ty) = if carrying {
                if team == 0 { (gw * 0.12, gh * 0.50) } else { (gw * 0.88, gh * 0.50) }
            } else if squad_id == 0 {
                // SOLO EMBERSTALKER RAIDER AI:
                // Avoid big enemy clusters (>=3 within 18m), take subterranean cave/flank paths, ambush stragglers, or cap unguarded shrines!
                if nearby_enemy_cnt >= 3 {
                    let ecx = enemy_cluster_cx / nearby_enemy_cnt as f32;
                    let ecy = enemy_cluster_cy / nearby_enemy_cnt as f32;
                    let away_x = mx - ecx;
                    let away_y = my - ecy;
                    let amag = away_x.hypot(away_y).max(1e-3);
                    // Sidestep tangentially into caves/flanks rather than straight back
                    let side = if bid % 2 == 0 { 1.0 } else { -1.0 };
                    (
                        (mx + (away_x / amag * 0.7 - away_y / amag * 0.7 * side) * 20.0).clamp(6.0, gw - 6.0),
                        (my + (away_y / amag * 0.7 + away_x / amag * 0.7 * side) * 20.0).clamp(6.0, gh - 6.0),
                    )
                } else if let Some(e) = best_enemy {
                    let e_allies = snap
                        .iter()
                        .filter(|o| o.id != e.id && o.team == e.team && (o.x - e.x).hypot(o.y - e.y) < 16.0)
                        .count();
                    if e_allies <= 1 && edist < 48.0 {
                        // Ambush isolated straggler!
                        (e.x, e.y)
                    } else {
                        // Take cave/flank lane toward least-guarded enemy/neutral shrine or relic
                        let unguarded_shrine = shrine_goals
                            .iter()
                            .filter(|s| s.2 != desired_owner)
                            .min_by_key(|s| {
                                snap.iter()
                                    .filter(|o| o.team != team && (o.x - s.0).hypot(o.y - s.1) < 22.0)
                                    .count()
                            })
                            .map(|s| (s.0, s.1));
                        if is_z16 && self.biome_id == field_core::BIOME_NORDIC_VIADUCT && (tick / 180 + bid as u64) % 3 == 0 {
                            // Subterranean SW basalt cave flank corridor!
                            let phase = ((tick as f32 * 0.012 + bid as f32).sin() + 1.0) * 0.5;
                            (gw * (0.20 + 0.22 * phase), gh * (0.73 - 0.03 * phase))
                        } else {
                            unguarded_shrine.unwrap_or((self.relic_x, self.relic_y))
                        }
                    }
                } else {
                    (self.relic_x, self.relic_y)
                }
            } else {
                // COORDINATED SQUAD AI (Squad Alpha=1, Bravo=2, Charlie=3, Delta=4...):
                // Shared dynamic squad objective: Escort Relic Carrier, Shrine Push, or Viaduct/Bridge Assault!
                let (obj_x, obj_y) = if let Some((rx, ry)) = team_relic_carrier[team as usize] {
                    // Escort friendly relic carrier!
                    let esc_ang = (bid as f32) * 1.4;
                    (rx + esc_ang.cos() * 6.5, ry + esc_ang.sin() * 6.5)
                } else {
                    // Rotate squad role cleanly across shrines, bridge, and relic
                    let phase_idx = ((squad_id as usize) + (tick / 900) as usize) % 4;
                    match phase_idx {
                        1 => {
                            // Bridge / Central Viaduct Assault
                            let patrol_x = gw * (0.50 + 0.14 * ((tick as f32 * 0.008 + squad_id as f32).sin()));
                            (patrol_x, gh * 0.50)
                        }
                        2 => {
                            // North Shrine / Glacier Push (or first uncaptured shrine)
                            shrine_goals
                                .iter()
                                .find(|s| s.2 != desired_owner)
                                .map(|s| (s.0, s.1))
                                .unwrap_or((gw * 0.50, gh * 0.18))
                        }
                        3 => {
                            // South Shrine / Caldera Push (or last uncaptured shrine)
                            shrine_goals
                                .iter()
                                .rev()
                                .find(|s| s.2 != desired_owner)
                                .map(|s| (s.0, s.1))
                                .unwrap_or((gw * 0.50, gh * 0.82))
                        }
                        _ => {
                            // Central Relic Contestation
                            (self.relic_x, self.relic_y)
                        }
                    }
                };

                // Blend with Squad Cohesion (pull toward squad centroid if drifted >14m) + class-specific routing
                let mut gx = obj_x + (bid as f32 * 1.7 + tick as f32 * 0.006).sin() * 4.5;
                let mut gy = obj_y + (bid as f32 * 2.1 - tick as f32 * 0.006).cos() * 4.5;
                if let Some(&(sx, sy, cnt)) = squad_sums.get(&(team, squad_id)) {
                    if cnt >= 2 {
                        let cx = sx / cnt as f32;
                        let cy = sy / cnt as f32;
                        let dist_sq = (cx - mx).hypot(cy - my);
                        if dist_sq > 14.0 {
                            gx = gx * 0.52 + cx * 0.48;
                            gy = gy * 0.52 + cy * 0.48;
                        }
                    }
                }
                // Tidekins actively rescue burning teammates within 16m, else flank along the central fjord!
                if class_id == CLASS_TIDEKIN {
                    if let Some(burn_ally) = snap
                        .iter()
                        .filter(|o| o.id != bid && o.team == team && o.burning && (o.x - mx).hypot(o.y - my) <= 16.0)
                        .min_by(|a, b| {
                            (a.x - mx)
                                .hypot(a.y - my)
                                .partial_cmp(&(b.x - mx).hypot(b.y - my))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        gx = burn_ally.x;
                        gy = burn_ally.y;
                    } else if (mx - gw * 0.50).abs() > gw * 0.09 && edist > 20.0 {
                        gx = gx * 0.65 + (gw * 0.50) * 0.35;
                    }
                } else if class_id == CLASS_BURROWER {
                    // Wave 17C: Burrowers path underground toward enemy-held shrines to undermine defenders!
                    if let Some(&(sx, sy, _)) = shrine_goals
                        .iter()
                        .filter(|s| s.2 != desired_owner)
                        .min_by(|a, b| {
                            (a.0 - mx)
                                .hypot(a.1 - my)
                                .partial_cmp(&(b.0 - mx).hypot(b.1 - my))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        gx = gx * 0.30 + sx * 0.70;
                        gy = gy * 0.30 + sy * 0.70;
                    }
                } else if class_id == CLASS_SKYWING && mz > self.surf_layer() as f32 + 1.5 {
                    // Wave 17C: Airborne Skywings fly bombing passes over enemies engaged by an allied Strider (<10m)
                    if let Some(e) = best_enemy {
                        let engaged_by_strider = snap.iter().any(|ally| {
                            ally.team == team
                                && ally.class_id == CLASS_STRIDER
                                && (ally.x - e.x).hypot(ally.y - e.y) < 10.0
                        });
                        if engaged_by_strider {
                            gx = e.x;
                            gy = e.y;
                        }
                    }
                }
                (gx.clamp(5.0, gw - 5.0), gy.clamp(5.0, gh - 5.0))
            };

            // Light local ally separation so 32-64 bots form readable formations instead of stacking
            let mut sep_x = 0.0f32;
            let mut sep_y = 0.0f32;
            for other in &snap {
                if other.id == bid || other.team != team {
                    continue;
                }
                let dx = mx - other.x;
                let dy = my - other.y;
                let d2 = dx * dx + dy * dy;
                if d2 > 0.01 && d2 < 10.24 {
                    let d = d2.sqrt();
                    let push = (3.2 - d) / 3.2;
                    sep_x += (dx / d) * push * 4.5;
                    sep_y += (dy / d) * push * 4.5;
                }
            }
            tx = (tx + sep_x).clamp(4.0, gw - 4.0);
            ty = (ty + sep_y).clamp(4.0, gh - 4.0);

            // --- 3. 3D Vertical Flight / Diving Intent per Archetype ---
            let desired_vz = match class_id {
                CLASS_SKYWING => {
                    // Soar smoothly above chasms/bridges (cruise ~5.5-8.5m above floor on Z16, ~2.5 on L4)
                    let target_alt = if is_z16 {
                        let base_cruise = (self.grid_d as f32 * 0.58)
                            + ((tick as f32 * 0.04 + bid as f32).sin()) * 2.2;
                        base_cruise.clamp(5.5, max_fly_z)
                    } else {
                        2.55
                    };
                    ((target_alt - mz) * 2.2).clamp(-3.0, 6.5)
                }
                CLASS_TIDEKIN => {
                    // Dive smoothly in deep rivers or surge up toward banks
                    if is_z16 {
                        let pl = (mz.round() as usize).clamp(0, self.grid_d - 1);
                        let p_here = self.volume.sample_entity_physics(mx, my, pl);
                        if p_here.water_frac > 0.12 {
                            let target_z = 2.0 + ((tick as f32 * 0.06 + bid as f32).sin() * 0.8);
                            ((target_z - mz) * 2.5).clamp(-3.5, 4.5)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                }
                CLASS_BURROWER => {
                    // Wave 17C: Burrowers actively path underground (z = 2..4) toward enemy-held shrines
                    if is_z16 {
                        let target_sub_z = 3.0f32;
                        if mz > 4.2 {
                            ((target_sub_z - mz) * 2.2).clamp(-4.0, 0.0)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            pending_desired_vz.push((bid, desired_vz));

            // --- 4. Cross-Class Elemental Combos & Low-HP Ally Rescue ---
            let near_river = (mx - gw * 0.50).abs() < gw * 0.14;
            let (enemy_oiled, enemy_wet) = if let Some((ex, ey, ez)) = enemy_pos_z {
                let el = (ez.round() as usize).clamp(0, self.grid_d - 1);
                let ep = self.volume.sample_entity_physics(ex, ey, el);
                (ep.oil_frac > 0.08, ep.water_frac > 0.14)
            } else {
                (false, false)
            };

            let wounded_ally = if squad_id > 0 {
                squad_wounded
                    .get(&(team, squad_id))
                    .copied()
                    .filter(|&(wx, wy, _)| (wx - mx).hypot(wy - my) <= 18.0)
            } else {
                None
            };

            // Nearby Striders/Skywings drop a protective Stasis Dome over a low-HP squadmate!
            if let Some((wx, wy, _)) = wounded_ally {
                if (class_id == CLASS_STRIDER || class_id == CLASS_SKYWING)
                    && tick >= self.last_squad_dome_tick[team as usize].saturating_add(480)
                    && self.time_domes.len() < 4
                {
                    pending_rescue_domes.push((bid, team, wx, wy));
                }
            }

            let elem = if enemy_oiled {
                0 // Prioritize thermal ignition for OIL INFERNO combo!
            } else if enemy_wet && (class_id == CLASS_STRIDER || class_id == CLASS_TIDEKIN) {
                1 // Prioritize DEEP FREEZE combo!
            } else if enemy_wet && class_id == CLASS_SKYWING {
                0 // FLASH-BOIL STEAM BURST!
            } else {
                match class_id {
                    CLASS_STRIDER => if near_river || tick % 3 != 0 { 1 } else { 0 },
                    CLASS_SKYWING => if tick % 5 == 0 { 2 } else { 0 },
                    CLASS_EMBERSTALKER => if tick % 3 == 0 { 2 } else { 0 },
                    _ => ((tick / 15 + bid as u64) % 3) as u8,
                }
            };

            if self.physics_native {
                let mut move_inp = ClientInputDatagram {
                    seq: tick as u32,
                    move_x: tx - mx,
                    move_y: ty - my,
                    aim_x: ax,
                    aim_y: ay,
                    dash: (tick + bid as u64 * 17) % (if class_id == CLASS_EMBERSTALKER { 48 } else { 72 }) == 0,
                    melee: false,
                    cast_bolt: false,
                    bolt_element: 0,
                    carve_ice: false,
                    cast_ability_e: false,
                    cast_ultimate_r: false,
                    client_time_ms: 0,
                };
                let (mut tf, mut sw, mut so, mut ss, mut rad, mut imp) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 3.8f32, 0.0f32);
                let fire_cadence = if class_id == CLASS_EMBERSTALKER { 9 } else { 12 };
                if (tick + bid as u64 * 9) % fire_cadence == 0 && (edist < 38.0 || near_river || wounded_ally.is_some()) {
                    if wounded_ally.is_some() && (class_id == CLASS_STRIDER || class_id == CLASS_SKYWING) {
                        // Drop protective Cryo-Wall / Smoke barrier for low-HP squadmate!
                        tf = -0.95;
                        ss = 0.65;
                        rad = 5.0;
                    } else if enemy_oiled && (class_id == CLASS_SKYWING || class_id == CLASS_EMBERSTALKER) {
                        // Cross-class OIL -> IGNITION detonation!
                        tf = 1.0;
                        rad = 4.4;
                        if edist < 10.0 { imp = 0.70; }
                    } else if enemy_wet && (class_id == CLASS_STRIDER || class_id == CLASS_TIDEKIN) {
                        // Cross-class WATER -> CRYO FREEZE combo!
                        tf = -0.95;
                        rad = 4.6;
                    } else {
                        match class_id {
                            CLASS_STRIDER => {
                                tf = -0.90;
                                if edist < 9.0 { imp = 0.80; }
                                rad = 4.5;
                            }
                            CLASS_SKYWING => {
                                if (tick / 12 + bid as u64) % 3 == 0 {
                                    so = 0.88;
                                    rad = 4.4;
                                } else {
                                    tf = 0.98;
                                    rad = 4.0;
                                }
                            }
                            CLASS_TIDEKIN => {
                                if (tick / 12 + bid as u64) % 3 == 0 {
                                    sw = 0.85;
                                    ss = 0.40;
                                    rad = 4.2;
                                } else {
                                    tf = if near_river { -0.88 } else { 0.85 };
                                }
                                if edist < 9.0 { imp = 0.65; }
                            }
                            CLASS_EMBERSTALKER => {
                                // CLASS_EMBERSTALKER: rapid oil/smoke/ignition ambush burst!
                                match (tick / 9 + bid as u64) % 3 {
                                    0 => { so = 0.90; ss = 0.60; rad = 4.0; }
                                    1 => { tf = 1.0; rad = 4.2; if edist < 10.0 { imp = 0.75; } }
                                    _ => { ss = 0.80; so = 0.50; rad = 4.4; }
                                }
                            }
                            _ => {
                                // CLASS_BURROWER (Terramole): subterranean seismic shock + thermal smoke breach!
                                match (tick / 12 + bid as u64) % 3 {
                                    0 => { imp = 0.88; ss = 0.55; rad = 4.6; }
                                    1 => { tf = 0.90; rad = 4.2; if edist < 11.0 { imp = 0.80; } }
                                    _ => { tf = -0.85; ss = 0.45; rad = 4.4; }
                                }
                            }
                        }
                    }
                }
                if let Some(&ov) = self.bot_overrides.get(&bid) {
                    if tick <= ov.expires_tick {
                        if ov.move_x.hypot(ov.move_y) > 0.05 {
                            move_inp.move_x = ov.move_x;
                            move_inp.move_y = ov.move_y;
                            move_inp.aim_x = ov.move_x;
                            move_inp.aim_y = ov.move_y;
                        }
                        if let Some(oe) = ov.elem {
                            match oe % 3 {
                                0 => tf = 0.95,
                                1 => tf = -0.95,
                                _ => so = 0.90,
                            }
                        }
                    }
                }
                self.apply_input(bid, &move_inp);
                let dlen = move_inp.aim_x.hypot(move_inp.aim_y).max(1e-3);
                let dist_off = edist.min(16.0).max(4.0);
                let target_z_opt = if is_z16 {
                    enemy_pos_z.map(|(_, _, ez)| ez)
                } else {
                    None
                };
                self.apply_player_physical_actuator_3d(
                    bid,
                    (move_inp.aim_x / dlen) * dist_off,
                    (move_inp.aim_y / dlen) * dist_off,
                    target_z_opt,
                    tf,
                    sw,
                    so,
                    ss,
                    rad,
                    imp,
                );
                continue;
            }

            let mut inp = ClientInputDatagram {
                seq: tick as u32,
                move_x: tx - mx,
                move_y: ty - my,
                aim_x: ax,
                aim_y: ay,
                dash: (tick + bid as u64 * 17) % (if class_id == CLASS_EMBERSTALKER { 48 } else { 75 }) == 0,
                melee: edist < 6.0,
                cast_bolt: (tick + bid as u64 * 9) % 16 == 0 && (edist < 60.0 || near_river),
                bolt_element: elem,
                carve_ice: near_river && mz < 1.5 && (tick + bid as u64 * 31) % 240 == 0,
                cast_ability_e: (tick + bid as u64 * 23) % 150 == 0 && (edist < 34.0 || near_river),
                cast_ultimate_r: (tick + bid as u64 * 41) % 390 == 0 && edist < 26.0,
                client_time_ms: 0,
            };
            if let Some(&ov) = self.bot_overrides.get(&bid) {
                if tick <= ov.expires_tick {
                    if ov.move_x.hypot(ov.move_y) > 0.05 {
                        inp.move_x = ov.move_x;
                        inp.move_y = ov.move_y;
                        inp.aim_x = ov.move_x;
                        inp.aim_y = ov.move_y;
                    }
                    if let Some(oe) = ov.elem {
                        inp.bolt_element = oe % 3;
                        if tick % 9 == 0 {
                            inp.cast_bolt = true;
                        }
                    }
                    if let Some(ab) = ov.ability {
                        match ab {
                            0 => inp.dash = true,
                            1 => inp.melee = true,
                            2 => inp.cast_bolt = true,
                            3 => inp.carve_ice = true,
                            4 => inp.cast_ability_e = true,
                            5 => inp.cast_ultimate_r = true,
                            _ => {}
                        }
                    }
                }
            }
            self.apply_input(bid, &inp);
        }

        for (bid, dvz) in pending_desired_vz {
            if let Some(p) = self.players.get_mut(&bid) {
                p.desired_vz = dvz;
            }
        }
        for (bid, team, wx, wy) in pending_rescue_domes {
            if tick >= self.last_squad_dome_tick[team as usize].saturating_add(480) {
                self.last_squad_dome_tick[team as usize] = tick;
                self.place_time_dome(bid, wx, wy, 1);
                self.push_combo_callout(3, wx, wy, team, "STASIS RESCUE!");
            }
        }
    }

    fn tick_combat_30hz(&mut self) {
        self.sim_tick += 1;
        let tick = self.sim_tick;
        let dt = 1.0 / 30.0f32;
        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let surf_l = self.surf_layer();
        let surf_z = surf_l as f32;

        self.day_time += (dt / 240.0) * self.time_scale;
        while self.day_time >= 1.0 {
            self.day_time -= 1.0;
            self.world_day = self.world_day.saturating_add(1);
        }
        let wind_ang = (tick as f32) * 0.0012;
        let wind_mag = match self.weather_kind {
            WEATHER_BLIZZARD => 1.65,
            WEATHER_SQUALL => 1.35,
            WEATHER_EMBER => 0.95,
            _ => 0.65,
        };
        self.wind_x = wind_ang.cos() * wind_mag;
        self.wind_y = wind_ang.sin() * wind_mag;

        if tick % 1800 == 0 {
            self.weather_kind = (self.weather_kind + 1) % 4;
            let wname = match self.weather_kind {
                WEATHER_BLIZZARD => "BLIZZARD FRONT!",
                WEATHER_SQUALL => "THAW SQUALL!",
                WEATHER_EMBER => "EMBER DUSK!",
                _ => "CLEAR FROST!",
            };
            self.push_event(5, gw * 0.5, gh * 0.5, 0, 2, wname.to_string());
        }

        while let Some(&(exp, _)) = self.events.front() {
            if tick > exp { self.events.pop_front(); } else { break; }
        }
        self.speech_bubbles.retain(|(exp, _)| tick <= *exp);
        self.bot_overrides.retain(|_, ov| tick <= ov.expires_tick);
        self.recent_actuators.retain(|&(_, _, _, _, _, exp)| tick <= exp);
        self.actuator_pulses.retain(|(exp, _, _)| tick <= *exp);
        self.time_domes.retain(|d| tick <= d.expires_tick);

        // Hold victory celebration state for 10 seconds, then cleanly reset for the next match
        if self.match_phase != MATCH_PHASE_ACTIVE {
            let celeb_elapsed = self
                .celebration_started
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(CELEBRATION_SECS as u64);
            self.match_time_left_s = CELEBRATION_SECS.saturating_sub(celeb_elapsed as u16);
            if celeb_elapsed >= CELEBRATION_SECS as u64 {
                self.reset_match();
            }
            return;
        }
        let elapsed_s = self.match_start_instant.elapsed().as_secs() as u16;
        self.match_time_left_s = MATCH_DURATION_SECS.saturating_sub(elapsed_s);

        if self.pending_chat.is_some() || (tick % 3 == 0 && self.neural_bots_on) {
            self.step_acoustics_rpc();
        }

        if tick % 3 == 0 {
            self.step_bots();
            self.run_emergent_synergies();
        }
        if tick % 6 == 0 {
            self.check_terrain_engineering_milestones();
        }

        // Advance floating ice rafts on surface layer
        let old_rafts = std::mem::take(&mut self.rafts);
        let mut new_rafts = Vec::with_capacity(old_rafts.len());
        for mut b in old_rafts {
            if b.ttl == 0 { continue; }
            b.ttl -= 1;
            let nx = (b.x + b.vx * dt).clamp(4.0, gw - 5.0);
            let ny = (b.y + b.vy * dt).clamp(4.0, gh - 5.0);
            let hz = self.sample_hazard(nx, ny, surf_l);
            if (hz & (1 << 1)) == 0 { b.x = nx; b.y = ny; }
            if (hz & (1 << 4)) == 0 { b.vx *= 0.85; b.vy *= 0.85; } else { b.vy = b.vy * 0.9 + 3.6 * 0.1; }
            new_rafts.push(b);
        }
        self.rafts = new_rafts;

        // Move players in 3D `(x, y, z)` with 5 Physically Distinct Archetype Locomotion Modes!
        let pids: Vec<u32> = self.players.keys().copied().collect();
        let mut phys_dmg_events: Vec<(Option<u32>, u32, i16, &'static str, u8)> = Vec::new();
        let mut combo_callouts: Vec<(f32, f32, u8, String)> = Vec::new();
        let mut named_combo_callouts: Vec<(usize, f32, f32, u8, &'static str)> = Vec::new();
        let field_tick = self.volume.tick;
        let physics_native = self.physics_native;
        for pid in pids {
            let (px, py, pz, vx, vy, mut vz, desired_vz, mut kb_vx, mut kb_vy, dash_until, chilled_until, team, class_id, last_blast_ft, mut burrow_ambush_until) = {
                let p = &self.players[&pid];
                (
                    p.x, p.y, p.z, p.vx, p.vy, p.vz, p.desired_vz, p.kb_vx, p.kb_vy, p.dash_until,
                    p.chilled_until, p.team, p.class_id, p.last_phys_blast_field_tick, p.burrow_ambush_until,
                )
            };
            let is_z16 = self.is_z16();
            let cur_layer = if is_z16 {
                (pz.round() as usize).clamp(0, self.grid_d - 1)
            } else {
                (pz.round() as usize).clamp(surf_l, self.grid_d - 1)
            };
            let hz_here = self.sample_hazard(px, py, cur_layer);
            let phys = if physics_native {
                self.volume.sample_entity_physics(px, py, cur_layer)
            } else {
                EntityVoxelPhysics::default()
            };
            let on_raft = if !is_z16 && pz < 1.35 {
                self.rafts.iter().find_map(|b| {
                    if (px - b.x).powi(2) + (py - b.y).powi(2) <= (b.radius + 1.0).powi(2) {
                        Some((b.vx, b.vy))
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            let xi_c = (px.floor() as i32).clamp(0, (self.grid_w - 1) as i32) as usize;
            let yi_c = (py.floor() as i32).clamp(0, (self.grid_h - 1) as i32) as usize;
            let idx_c = yi_c * self.grid_w + xi_c;

            // Detect water contact for Tidekin Hydrodynamic Surge (current layer or layer directly below feet)
            let in_water = phys.water_frac > 0.12
                || (hz_here & (1 << 4)) != 0
                || self.volume.layers[cur_layer].mass[SP_WATER][idx_c] >= 120
                || (cur_layer > 0 && self.volume.layers[cur_layer - 1].mass[SP_WATER][idx_c] >= 200);

            // Detect Smoke Veil / subterranean cave stealth for Emberstalker
            let in_smoke_veil = class_id == CLASS_EMBERSTALKER
                && (phys.smoke_opacity > 0.15
                    || (is_z16 && self.volume.measure_vertical_clearance_3d((px, py, pz)) < 6.0));

            let laws = self.volume.world_laws;
            let morph = BodyMorphology::from_archetype(class_id);
            let grav_g = laws.gravity_g.clamp(0.15, 2.5);
            let visc = laws.viscosity.clamp(0.25, 4.0);

            // Detect Thermal Updraft column (`T > 340 K` = `34_000 cK`) beneath winged bodies
            let mut thermal_updraft_ck = 0i64;
            if morph.wing_lift > 0.30 {
                let z_lo = cur_layer.saturating_sub(if is_z16 { 6 } else { 2 });
                for zs in z_lo..=cur_layer {
                    let t_s = self.volume.layers[zs].cell_temp_ck(idx_c);
                    if t_s > thermal_updraft_ck {
                        thermal_updraft_ck = t_s;
                    }
                }
            }

            let mut mult = 1.0f32;
            if morph.buoyancy >= 0.70 && in_water {
                // High-buoyancy hydrodynamic propulsion coupled to fluid viscosity!
                let visc_bonus = (1.0 / visc).clamp(0.70, 1.35);
                mult *= (1.0 + morph.buoyancy * visc_bonus).clamp(1.4, 2.35);
            } else if physics_native {
                // Locomotion emerges directly from BodyMorphology × voxel fluid/ice/oil/temp × world_laws
                if on_raft.is_none() && morph.wing_lift < 0.50 && phys.water_frac > 0.05 && phys.ice_frac < 0.30 {
                    let drag = (0.46 * (1.0 - morph.buoyancy * 0.65) * visc.sqrt()).clamp(0.12, 0.72);
                    mult *= (1.0 - drag * phys.water_frac).clamp(0.38, 1.0);
                }
                if phys.ice_frac > 0.10 {
                    mult *= 1.0 + 0.24 * phys.ice_frac;
                }
                if phys.oil_frac > 0.10 {
                    mult *= 1.0 - (0.12 * visc.sqrt()).min(0.26) * phys.oil_frac;
                }
                if phys.temp_c < -15.0 && tick >= dash_until && morph.thermal_insul < 0.70 {
                    let cold_slow = (((-15.0 - phys.temp_c) / 80.0) * (1.0 - morph.thermal_insul)).clamp(0.0, 0.32);
                    mult *= 1.0 - cold_slow;
                }
                // Low-friction glide on ice / oil preserves momentum via kb_vx/kb_vy slip term
                let slip = (phys.ice_frac * 0.72).max(phys.oil_frac * 0.55).clamp(0.0, 0.78);
                if slip > 0.12 && morph.wing_lift < 0.50 {
                    kb_vx = kb_vx * (0.78 + 0.16 * slip) + vx * mult * (0.14 * slip);
                    kb_vy = kb_vy * (0.78 + 0.16 * slip) + vy * mult * (0.14 * slip);
                }
            } else {
                if on_raft.is_none() && morph.wing_lift < 0.50 && (hz_here & (1 << 4)) != 0 {
                    mult *= (0.60 + 0.25 * morph.buoyancy).min(0.95);
                }
                if ((hz_here & (1 << 3)) != 0 || tick < chilled_until) && tick >= dash_until && morph.thermal_insul < 0.70 {
                    mult *= 0.68 + 0.22 * morph.thermal_insul;
                }
            }
            if in_smoke_veil {
                // Emberstalker Smoke Veil: +28% agility in smoke & caves!
                mult *= 1.28;
            }
            let (cvx, cvy) = on_raft.unwrap_or((0.0, 0.0));
            let mut vx_s = vx * mult + cvx + kb_vx;
            let mut vy_s = vy * mult + cvy + kb_vy;
            kb_vx *= 0.82;
            kb_vy *= 0.82;
            if kb_vx.abs() < 0.05 { kb_vx = 0.0; }
            if kb_vy.abs() < 0.05 { kb_vy = 0.0; }

            let (fx, fy, fz);
            if is_z16 {
                let max_alt = (self.grid_d as f32 - 2.0).max(2.0);
                let eff_grav = 18.0 * grav_g;
                let low_g_lift = (1.0 / grav_g).sqrt().clamp(0.65, 1.85);
                // 1. Winged flight / low-G aerodynamic glide coupled to BodyMorphology.wing_lift × world_laws.gravity_g
                if morph.wing_lift > 0.05 {
                    let lift_frac = (morph.wing_lift * 0.92).clamp(0.05, 0.84);
                    vz += eff_grav * lift_frac * dt;
                    if morph.wing_lift >= 0.50 {
                        if thermal_updraft_ck > 34_000 {
                            let updraft_gain = ((thermal_updraft_ck - 34_000) as f32 / 12_000.0).clamp(0.45, 2.2);
                            vz = (vz + 15.0 * morph.wing_lift * low_g_lift * updraft_gain * dt).min(9.5);
                        }
                        if desired_vz.abs() > 0.05 {
                            vz = (vz + (desired_vz - vz) * 4.5 * dt).clamp(-3.6, 8.5 * low_g_lift.min(1.3));
                        } else if vx_s.hypot(vy_s) > 2.0 && pz < max_alt {
                            vz = vz.max(-2.6 * grav_g.sqrt());
                        }
                        vz = vz.clamp(-3.8, 9.5);
                    }
                }
                // 2. Aquatic buoyancy coupled to BodyMorphology.buoyancy × world_laws.gravity_g
                if morph.buoyancy >= 0.65 && in_water {
                    vz += eff_grav * morph.buoyancy * dt;
                    vz = (vz + (desired_vz - vz) * 5.0 * dt).clamp(-5.5, 9.5 * low_g_lift.min(1.3));
                    let spd_h = vx_s.hypot(vy_s);
                    if spd_h > 2.5 {
                        let nx_d = vx_s / spd_h;
                        let ny_d = vy_s / spd_h;
                        if self.volume.sample_solid_density_3d(px + nx_d * 0.9, py + ny_d * 0.9, pz + 0.5) >= 0.45 {
                            vz = vz.max(8.5 * low_g_lift.min(1.25));
                        }
                    }
                }
                // 3. Wall/ledge climbing coupled to BodyMorphology.climb_grip × world_laws.gravity_g
                if morph.climb_grip >= 0.65 {
                    let spd_h = vx_s.hypot(vy_s);
                    if spd_h > 2.0 {
                        let nx_d = vx_s / spd_h;
                        let ny_d = vy_s / spd_h;
                        let max_ledge = (1.6 + morph.climb_grip * 2.0) * low_g_lift.min(1.35);
                        let wall_ahead = self
                            .volume
                            .sample_solid_density_3d(px + nx_d * 0.95, py + ny_d * 0.95, pz + 0.55)
                            >= 0.45;
                        let can_top_out = self
                            .volume
                            .sample_solid_density_3d(px + nx_d * 0.95, py + ny_d * 0.95, (pz + max_ledge).min(max_alt))
                            < 0.50;
                        let headroom_ok = self.volume.sample_solid_density_3d(px, py, (pz + 2.0).min(max_alt)) < 0.50;
                        if wall_ahead && can_top_out && headroom_ok && pz < max_alt {
                            let climb_v = (26.0 * morph.climb_grip * low_g_lift.min(1.3)).clamp(16.0, 34.0);
                            vz = (vz + climb_v * dt).clamp(6.8, 10.8);
                        }
                    }
                }
                // 4. Conservative subterranean tunnel boring coupled to BodyMorphology.bore_torque
                if morph.bore_torque >= 0.50 {
                    let spd_h = vx_s.hypot(vy_s);
                    if spd_h > 1.5 {
                        let nx_d = vx_s / spd_h;
                        let ny_d = vy_s / spd_h;
                        let probe_x = (px + nx_d * 1.10).clamp(3.0, gw - 4.0);
                        let probe_y = (py + ny_d * 1.10).clamp(3.0, gh - 4.0);
                        let d_low = self.volume.sample_solid_density_3d(probe_x, probe_y, (pz + 0.35).clamp(1.0, max_alt));
                        let d_mid = self.volume.sample_solid_density_3d(probe_x, probe_y, (pz + 0.95).clamp(1.0, max_alt));
                        let d_hi = self.volume.sample_solid_density_3d(probe_x, probe_y, (pz + 1.55).clamp(1.0, max_alt));
                        if d_low >= 0.35 || d_mid >= 0.35 || d_hi >= 0.35 {
                            let bore_r = (1.20 + 0.75 * morph.bore_torque).clamp(1.4, 2.2);
                            let moved = self.volume.carve_tunnel_3d(
                                probe_x,
                                probe_y,
                                (pz + 0.85).clamp(1.2, max_alt),
                                nx_d,
                                ny_d,
                                bore_r,
                            );
                            if moved > 0 {
                                self.tunnels_carved = self.tunnels_carved.saturating_add(1);
                                self.tunnel_mass_displaced =
                                    self.tunnel_mass_displaced.saturating_add(moved as u64);
                                burrow_ambush_until = tick + 90;
                            }
                        } else if tick <= burrow_ambush_until
                            && self.volume.measure_vertical_clearance_3d((px, py, pz)) >= 4.5
                        {
                            let enemy_near = self.players.values().any(|o| {
                                o.team != team && (o.x - px).hypot(o.y - py) < 14.0
                            });
                            if enemy_near {
                                named_combo_callouts.push((2, px, py, team, "TUNNEL AMBUSH!"));
                                burrow_ambush_until = 0;
                            }
                        }
                    }
                }
                let ((nx, ny, nz), (_nvx, _nvy, nvz), _grounded) =
                    self.volume.step_capsule_3d((px, py, pz), (vx_s, vy_s, vz), 0.55, 1.80, dt);
                fx = nx;
                fy = ny;
                fz = if morph.wing_lift >= 0.50 {
                    nz.clamp(0.5, max_alt)
                } else {
                    nz.clamp(0.0, (self.grid_d as f32) - 0.5)
                };
                vz = nvz;
            } else {
                // Continuous bilinear solid-density gradient capsule collider (L=4 strata mode)
                let coll_layer = if pz >= 1.55 { 2.min(self.grid_d - 1) } else { surf_l };
                let dashing_now = tick < dash_until;
                let mut mx = (px + vx_s * dt).clamp(3.5, gw - 3.5);
                let mut my = (py + vy_s * dt).clamp(3.5, gh - 3.5);
                if morph.bore_torque >= 0.50
                    && vx_s.hypot(vy_s) > 1.5
                    && self.sample_capsule_density(mx, my, coll_layer, false) > 0.35
                {
                    let spd_h = vx_s.hypot(vy_s).max(1e-3);
                    let moved = self.volume.carve_tunnel_3d(
                        mx,
                        my,
                        (pz.round()).max(1.0),
                        vx_s / spd_h,
                        vy_s / spd_h,
                        (1.15 + 0.70 * morph.bore_torque).clamp(1.4, 2.1),
                    );
                    if moved > 0 {
                        self.tunnels_carved = self.tunnels_carved.saturating_add(1);
                        self.tunnel_mass_displaced =
                            self.tunnel_mass_displaced.saturating_add(moved as u64);
                        burrow_ambush_until = tick + 90;
                    }
                }
                let capsule_at = |tx: f32, ty: f32| -> f32 {
                    let allow_water = dashing_now
                        || morph.wing_lift >= 0.50
                        || morph.buoyancy >= 0.65
                        || on_raft.is_some()
                        || (pz < 1.35
                            && self.rafts.iter().any(|b| {
                                (tx - b.x).powi(2) + (ty - b.y).powi(2) <= (b.radius + 1.0).powi(2)
                            }));
                    self.sample_capsule_density(tx, ty, coll_layer, allow_water)
                };

                for _ in 0..2 {
                    let dens = capsule_at(mx, my);
                    if dens <= 0.35 {
                        break;
                    }
                    let eps = 0.35f32;
                    let gx = (capsule_at(mx + eps, my) - capsule_at(mx - eps, my)) / (2.0 * eps);
                    let gy = (capsule_at(mx, my + eps) - capsule_at(mx, my - eps)) / (2.0 * eps);
                    let gmag = gx.hypot(gy);
                    if gmag > 1e-4 {
                        let nx = gx / gmag;
                        let ny = gy / gmag;
                        let vn = vx_s * nx + vy_s * ny;
                        if vn > 0.0 {
                            vx_s -= vn * nx;
                            vy_s -= vn * ny;
                        }
                        let push = ((dens - 0.35) * 0.65).min(0.45);
                        mx = (px + vx_s * dt - nx * push).clamp(3.5, gw - 3.5);
                        my = (py + vy_s * dt - ny * push).clamp(3.5, gh - 3.5);
                    } else {
                        mx = px;
                        my = py;
                        break;
                    }
                }
                let d_final = capsule_at(mx, my);
                if d_final > 0.62 && capsule_at(px, py) < d_final {
                    mx = px;
                    my = py;
                }
                fx = mx;
                fy = my;

                // 3D Vertical Altitude Update on L=4:
                let l2_deck = if self.grid_d >= 3 {
                    if physics_native {
                        let p2 = self.volume.sample_entity_physics(fx, fy, 2);
                        p2.support_solid && (self.sample_hazard(fx, fy, 2) & (1 << 1)) == 0
                    } else {
                        self.layer2_has_walkable_deck(fx, fy)
                    }
                } else {
                    false
                };
                let nx_norm = fx / gw;
                let ny_norm = fy / gh;
                let on_ramp = self.grid_d >= 3
                    && (((ny_norm - 0.33).abs() < 0.025 && (nx_norm < 0.36 || nx_norm > 0.64))
                        || (((nx_norm - 0.76).powi(2) + (ny_norm - 0.66).powi(2)).sqrt() > 0.08
                            && ((nx_norm - 0.76).powi(2) + (ny_norm - 0.66).powi(2)).sqrt() < 0.11));

                let mut mz = pz;
                let eff_grav_l4 = 8.5 * grav_g;
                if morph.wing_lift >= 0.50 {
                    // Winged 3D flight & thermal glide on L=4 coupled to world_laws.gravity_g!
                    let max_l4_z = (self.grid_d as f32 - 0.6).max(2.6);
                    vz -= eff_grav_l4 * (1.0 - morph.wing_lift * 0.92).clamp(0.14, 0.85) * dt;
                    if thermal_updraft_ck > 34_000 {
                        vz = (vz + 4.5 * morph.wing_lift * dt).min(3.2);
                    }
                    if desired_vz.abs() > 0.05 {
                        vz = (vz + (desired_vz.clamp(-1.5, 2.2) - vz) * 4.0 * dt).clamp(-1.5, 2.8);
                    }
                    mz = (mz + vz * dt).clamp(surf_z, max_l4_z);
                    if mz <= surf_z {
                        vz = 0.0;
                    }
                } else if l2_deck && (pz >= 1.45 || on_ramp || morph.climb_grip >= 0.65) {
                    mz = (pz + (3.8 * (0.6 + 0.5 * morph.climb_grip)) * dt).min(2.0);
                    vz = 0.0;
                } else if morph.buoyancy >= 0.65 && in_water {
                    mz = (mz + (surf_z - mz) * 4.0 * dt).max(surf_z);
                    vz = 0.0;
                } else {
                    vz -= eff_grav_l4 * dt;
                    mz += vz * dt;
                    if mz <= surf_z {
                        mz = surf_z;
                        vz = 0.0;
                    }
                }
                fz = mz;
            }

            let critter_here = (self.sample_hazard(fx, fy, surf_l) >> 5) & 0x07;
            let near_friendly_shrine = {
                let want = if team == 0 { -1 } else { 1 };
                self.shrines.iter().any(|s| s.owner == want && (s.x - fx).hypot(s.y - fy) < 18.0)
            };
            let regen_mul = 1.0
                + if critter_here >= 2 { 0.45 } else { 0.0 }
                + if near_friendly_shrine { 0.55 } else { 0.0 };

            let mut new_last_blast_ft = last_blast_ft;
            let shield_wall_on = tick < self.players[&pid].shield_wall_until;
            let mut kb_res = (1.0 / morph.mass.max(0.5)).clamp(0.35, 1.25);
            if shield_wall_on {
                kb_res *= 0.30;
            }
            let local_temp_hot = self.volume.layers[cur_layer].cell_temp_ck(idx_c) > 39_000 || phys.temp_c > 116.85;
            let in_tunnel_now = is_z16
                && self.volume.measure_vertical_clearance_3d((fx, fy, fz)) < 5.0
                && fz <= 5.5;
            let airborne_now = if is_z16 {
                fz > surf_z + 2.5
                    && self
                        .volume
                        .sample_solid_density_3d(fx, fy, (fz - 1.2).max(0.5))
                        < 0.45
            } else {
                fz > surf_z + 0.85
            };
            if physics_native && tick >= dash_until {
                let owner = self.find_recent_actuator_owner(fx, fy, team);
                // 1. Emergent Blast Damage & Knockback from combustion (delta_temp_spike) & flash-boil (delta_steam_burst)
                if field_tick != last_blast_ft {
                    if phys.delta_temp_spike > 18.0 {
                        let dmg = ((phys.delta_temp_spike - 15.0) * 0.42 * (1.0 - 0.35 * morph.thermal_insul))
                            .round()
                            .clamp(4.0, 38.0) as i16;
                        let (bx, by) = self.voxel_blast_dir(fx, fy, cur_layer, false);
                        let push = (phys.delta_temp_spike.min(120.0) * 0.34).clamp(8.0, 42.0) * kb_res;
                        kb_vx = (kb_vx + bx * push).clamp(-48.0, 48.0);
                        kb_vy = (kb_vy + by * push).clamp(-48.0, 48.0);
                        phys_dmg_events.push((owner, pid, dmg, "DETONATION!", 0));
                        if let Some(oid) = owner {
                            if let Some(op) = self.players.get(&oid) {
                                if op.class_id == CLASS_SKYWING && op.z >= surf_z + 1.5 {
                                    named_combo_callouts.push((0, fx, fy, op.team, "AERIAL OIL-IGNITION!"));
                                }
                            }
                        }
                        if dmg >= 14 {
                            combo_callouts.push((fx, fy, 1 - team, "COMBUSTION BLAST!".into()));
                        }
                        new_last_blast_ft = field_tick;
                    } else if phys.delta_steam_burst > 0.06 {
                        let dmg = (phys.delta_steam_burst * 95.0 * (1.0 - 0.30 * morph.thermal_insul))
                            .round()
                            .clamp(4.0, 32.0) as i16;
                        let (bx, by) = self.voxel_blast_dir(fx, fy, cur_layer, true);
                        let push = (phys.delta_steam_burst * 72.0).clamp(8.0, 38.0) * kb_res;
                        kb_vx = (kb_vx + bx * push).clamp(-48.0, 48.0);
                        kb_vy = (kb_vy + by * push).clamp(-48.0, 48.0);
                        phys_dmg_events.push((owner, pid, dmg, "FLASH-BOIL!", 0));
                        if let Some(oid) = owner {
                            if let Some(op) = self.players.get(&oid) {
                                if op.class_id == CLASS_TIDEKIN {
                                    let ol = (op.z.round() as usize).clamp(0, self.grid_d - 1);
                                    if self.volume.sample_entity_physics(op.x, op.y, ol).water_frac > 0.10 {
                                        named_combo_callouts.push((1, fx, fy, op.team, "SUBMERGED GEYSER!"));
                                    }
                                }
                            }
                        }
                        if dmg >= 12 {
                            combo_callouts.push((fx, fy, 1 - team, "FLASH-BOIL BURST!".into()));
                        }
                        new_last_blast_ft = field_tick;
                    }
                }
                // 2. Scorch / Hypothermia damage from temp_c extremes (>120°C or <-25°C), attenuated by BodyMorphology.thermal_insul!
                let dome_dt_p = self.sample_time_dome_scale(fx, fy);
                let scorch_cadence: u64 = if dome_dt_p < 0.5 { 36 } else if dome_dt_p > 1.5 { 2 } else { 6 };
                let insul_mul = (1.0 - morph.thermal_insul).clamp(0.15, 1.0);
                if tick % scorch_cadence == 0 {
                    if phys.temp_c > 120.0 {
                        let dmg = (((phys.temp_c - 120.0) / 45.0).ceil() * insul_mul)
                            .round()
                            .clamp(1.0, 16.0) as i16;
                        phys_dmg_events.push((owner, pid, dmg, "SCORCH", 0));
                    } else if phys.temp_c < -25.0 && morph.thermal_insul < 0.70 {
                        let dmg = (((-25.0 - phys.temp_c) / 22.0).ceil() * insul_mul)
                            .round()
                            .clamp(1.0, 10.0) as i16;
                        phys_dmg_events.push((owner, pid, dmg, "HYPO", 1));
                    }
                }
            }

            let dome_dt_classic = self.sample_time_dome_scale(fx, fy);
            let burn_cadence_classic: u64 = if dome_dt_classic < 0.5 { 48 } else if dome_dt_classic > 1.5 { 3 } else { 8 };
            let p = self.players.get_mut(&pid).unwrap();
            p.x = fx;
            p.y = fy;
            p.z = fz;
            p.vz = vz;
            p.kb_vx = kb_vx;
            p.kb_vy = kb_vy;
            p.last_phys_blast_field_tick = new_last_blast_ft;
            p.burrow_ambush_until = burrow_ambush_until;
            p.smoke_veil_active = in_smoke_veil;
            p.in_tunnel = in_tunnel_now;
            p.in_water = in_water;
            p.airborne = airborne_now;
            if local_temp_hot {
                p.status_burn_until = p.status_burn_until.max(tick + 60);
            } else if in_water && phys.temp_c < 85.0 && tick < p.status_burn_until {
                // Cool voxel water physically quenches burning agents alongside Tidekin Quench Rescue!
                p.status_burn_until = 0;
            }
            p.mana = (p.mana + 9.5 * regen_mul * dt).min(p.max_mana);
            p.stamina = (p.stamina + 16.0 * regen_mul * dt).min(p.max_stamina);

            if physics_native {
                if tick % 25 == 0 && p.hp < p.max_hp && phys.temp_c <= 120.0 && phys.temp_c >= -25.0 {
                    let heal = if near_friendly_shrine { 4 } else { 2 };
                    p.hp = (p.hp + heal).min(p.max_hp);
                }
            } else if tick % burn_cadence_classic == 0 && (hz_here & (1 << 2)) != 0 && tick >= dash_until {
                let burn_dmg = if tick < p.oiled_until { 8 } else { 4 };
                phys_dmg_events.push((None, pid, burn_dmg, "BURN", 0));
            } else if tick % 25 == 0 && p.hp < p.max_hp && (hz_here & (1 << 2)) == 0 {
                let heal = if near_friendly_shrine { 4 } else { 2 };
                p.hp = (p.hp + heal).min(p.max_hp);
            }
        }
        for (owner, vid, dmg, tag, elem) in phys_dmg_events {
            self.apply_damage(owner, vid, dmg, tag, elem);
        }
        for (cx, cy, t, lbl) in combo_callouts {
            self.push_event(2, cx, cy, 0, t, lbl);
        }
        for (slot, cx, cy, t, lbl) in named_combo_callouts {
            self.push_combo_callout(slot, cx, cy, t, lbl);
        }

        // Advance 3D projectiles + Time-Dilation Dome Bullet-Time/Haste + High-Ground Damage Bonus!
        let old_proj = std::mem::take(&mut self.projectiles);
        let mut next_proj = Vec::with_capacity(old_proj.len());
        let mut proj_hits: Vec<(u32, u32, u8, f32, f32, f32, bool)> = Vec::new();
        let proj_min_z = if self.is_z16() { 0.5 } else { surf_z };
        let proj_z_tol = if self.is_z16() { 2.2 } else { 1.25 };
        for mut pr in old_proj {
            let dome_dt = self.sample_time_dome_scale(pr.x, pr.y);
            let pr_dt = dt * dome_dt;
            let nx = pr.x + pr.vx * pr_dt;
            let ny = pr.y + pr.vy * pr_dt;
            let nz = (pr.z + pr.vz * pr_dt).clamp(proj_min_z, (self.grid_d - 1) as f32);
            let pl = (nz.round() as usize).clamp(0, self.grid_d - 1);

            let mut hit = nx < 3.0
                || nx >= gw - 3.0
                || ny < 3.0
                || ny >= gh - 3.0
                || if self.is_z16() {
                    self.volume.sample_solid_density_3d(nx, ny, nz) >= 0.55
                } else {
                    (self.sample_hazard(nx, ny, pl) & (1 << 1)) != 0
                };

            let mut hit_pid = None;
            let mut high_ground = false;
            if !hit {
                for p in self.players.values() {
                    if p.id == pr.owner || p.team == pr.team || tick < p.dash_until {
                        continue;
                    }
                    if (p.x - nx).powi(2) + (p.y - ny).powi(2) <= (2.1 + pr.radius).powi(2)
                        && (p.z - nz).abs() <= proj_z_tol
                    {
                        hit = true;
                        hit_pid = Some(p.id);
                        high_ground = pr.z - p.z >= 0.45;
                        break;
                    }
                }
            }

            if hit || pr.ttl <= 1 {
                let cx = nx.clamp(3.0, gw - 4.0) as i32;
                let cy = ny.clamp(3.0, gh - 4.0) as i32;
                let idx = self.volume.layers[pl].idx(cx as usize, cy as usize);
                let has_oil = self.volume.layers[pl].mass[SP_OIL][idx] >= 120;
                let has_water_or_ice = self.volume.layers[pl].mass[SP_WATER][idx] >= 250
                    || self.volume.layers[pl].mass[SP_ICE][idx] >= 250;

                match pr.element {
                    0 => {
                        self.volume.apply_fire_3d(cx, cy, pl, 4, 1.55 * pr.power);
                        if has_oil {
                            self.volume.apply_fire_3d(cx, cy, pl, 6, 1.9);
                            let aerial_skywing = self
                                .players
                                .get(&pr.owner)
                                .map(|op| op.class_id == CLASS_SKYWING && op.z >= surf_z + 1.5)
                                .unwrap_or(false);
                            if aerial_skywing {
                                self.push_combo_callout(0, nx, ny, pr.team, "AERIAL OIL-IGNITION!");
                            } else {
                                self.push_event(2, nx, ny, 0, pr.team, "OIL INFERNO!".into());
                            }
                            self.eco.inject_fear_pulse(nx, ny, 20.0, 1.0);
                        } else if has_water_or_ice {
                            let boiled = self.volume.apply_steam_burst_3d(cx, cy, pl, 4);
                            if boiled >= 6 {
                                let submerged_tidekin = self
                                    .players
                                    .get(&pr.owner)
                                    .map(|op| {
                                        if op.class_id != CLASS_TIDEKIN {
                                            return false;
                                        }
                                        let ol = (op.z.round() as usize).clamp(0, self.grid_d - 1);
                                        self.volume.sample_entity_physics(op.x, op.y, ol).water_frac > 0.10
                                    })
                                    .unwrap_or(false);
                                if submerged_tidekin {
                                    self.push_combo_callout(1, nx, ny, pr.team, "SUBMERGED GEYSER!");
                                } else {
                                    self.push_event(2, nx, ny, 0, pr.team, "STEAM BURST!".into());
                                }
                            }
                        }
                    }
                    1 => {
                        self.record_terrain_engineering_candidates(pr.owner, pr.team, cx as f32, cy as f32, pl, -0.95, 0.0);
                        self.volume.apply_frost_3d(cx, cy, pl, 5, 1.9 * pr.power);
                    }
                    _ => self.volume.apply_oil_3d(cx, cy, pl, 4, 1.45 * pr.power),
                }

                if let Some(vid) = hit_pid {
                    proj_hits.push((pr.owner, vid, pr.element, pr.power, nx, ny, high_ground));
                }
            } else {
                pr.x = nx;
                pr.y = ny;
                pr.z = nz;
                if dome_dt < 0.5 {
                    // Bullet-time stasis: slow TTL expiry so slowed bolts linger visibly inside dome!
                    if (tick + pr.id as u64) % 6 == 0 {
                        pr.ttl = pr.ttl.saturating_sub(1);
                    }
                } else if dome_dt > 1.5 {
                    pr.ttl = pr.ttl.saturating_sub(2);
                } else {
                    pr.ttl -= 1;
                }
                next_proj.push(pr);
            }
        }
        self.projectiles = next_proj;

        for (owner, vid, elem, pwr, hx, hy, hg) in proj_hits {
            let hz = self.sample_hazard(hx, hy, surf_l);
            let is_wet = (hz & (1 << 4)) != 0;
            let (was_oiled, owner_team) = (
                self.players.get(&vid).map(|p| tick < p.oiled_until).unwrap_or(false),
                self.players.get(&owner).map(|p| p.team).unwrap_or(0),
            );
            let (mut dmg, mut tag) = match elem {
                0 => if was_oiled { (34, "INFERNO!") } else if is_wet { (24, "STEAM!") } else { (19, "") },
                1 => {
                    if let Some(v) = self.players.get_mut(&vid) {
                        v.chilled_until = tick + if is_wet { 90 } else { 45 };
                    }
                    if is_wet {
                        self.push_event(2, hx, hy, 0, owner_team, "DEEP FREEZE!".into());
                        (25, "FREEZE!")
                    } else {
                        (15, "")
                    }
                }
                _ => {
                    if let Some(v) = self.players.get_mut(&vid) {
                        v.oiled_until = tick + 150;
                    }
                    (7, "OILED")
                }
            };
            let hg_mul = if hg {
                if tag.is_empty() { tag = "HIGHGROUND!"; }
                1.25
            } else {
                1.0
            };
            dmg = ((dmg as f32) * pwr * hg_mul).round() as i16;
            self.apply_damage(Some(owner), vid, dmg, tag, elem);
        }

        // 3 Capturable Shrines
        let player_positions: Vec<(u32, u8, f32, f32)> =
            self.players.values().map(|p| (p.id, p.team, p.x, p.y)).collect();
        let mut shrine_events = Vec::new();
        for s in &mut self.shrines {
            let (mut w_cnt, mut e_cnt) = (0i32, 0i32);
            for &(_, team, px, py) in &player_positions {
                if (px - s.x).hypot(py - s.y) <= 13.0 {
                    if team == 0 { w_cnt += 1; } else { e_cnt += 1; }
                }
            }
            let net = (e_cnt - w_cnt) as f32;
            if net != 0.0 {
                s.progress = (s.progress + net * 0.0085).clamp(-1.0, 1.0);
            }
            let prev = s.owner;
            if s.progress <= -0.95 && s.owner != -1 {
                s.owner = -1;
                shrine_events.push((s.x, s.y, 0u8, format!("WEST CAP: {}", s.name)));
            } else if s.progress >= 0.95 && s.owner != 1 {
                s.owner = 1;
                shrine_events.push((s.x, s.y, 1u8, format!("EAST CAP: {}", s.name)));
            } else if prev != 0 && s.progress.abs() < 0.2 {
                s.owner = 0;
            }
        }
        for (sx, sy, team, lbl) in shrine_events {
            if team == 0 { self.score_west = self.score_west.saturating_add(25); }
            else { self.score_east = self.score_east.saturating_add(25); }
            for p in self.players.values_mut() {
                if p.team == team && (p.x - sx).hypot(p.y - sy) <= 14.0 {
                    p.score = p.score.saturating_add(25);
                }
            }
            self.push_event(3, sx, sy, 25, team, lbl);
        }
        if tick % 60 == 0 {
            let owned: Vec<(f32, f32, u8)> = self
                .shrines
                .iter()
                .filter_map(|s| {
                    if s.owner == -1 { Some((s.x, s.y, 0u8)) }
                    else if s.owner == 1 { Some((s.x, s.y, 1u8)) }
                    else { None }
                })
                .collect();
            for (sx, sy, team) in owned {
                if team == 0 { self.score_west = self.score_west.saturating_add(2); }
                else { self.score_east = self.score_east.saturating_add(2); }
                for p in self.players.values_mut() {
                    if p.team == team && (p.x - sx).hypot(p.y - sy) <= 18.0 {
                        p.score = p.score.saturating_add(2);
                    }
                }
            }
        }

        // Golden Relic
        if let Some(cid) = self.relic_carrier {
            if let Some((team, x, y, name)) = self.players.get(&cid).map(|p| (p.team, p.x, p.y, p.name.clone())) {
                self.relic_x = x;
                self.relic_y = y;
                let scored_w = team == 0 && x < gw * 0.16;
                let scored_e = team == 1 && x > gw * 0.84;
                if scored_w || scored_e {
                    if scored_w { self.score_west = self.score_west.saturating_add(100); }
                    else { self.score_east = self.score_east.saturating_add(100); }
                    if let Some(p) = self.players.get_mut(&cid) {
                        p.carrying_relic = false;
                        p.score = p.score.saturating_add(100);
                        p.grant_xp(90);
                    }
                    self.relic_carrier = None;
                    self.relic_x = gw * 0.50;
                    self.relic_y = gh * 0.50;
                    self.push_event(4, x, y, 100, team, format!("{} SCORED RELIC!", name));
                }
            } else {
                self.relic_carrier = None;
            }
        } else if let Some((pid, team, name, px, py)) = self
            .players
            .values()
            .find(|p| (p.x - self.relic_x).hypot(p.y - self.relic_y) <= 3.6 && (self.is_z16() || p.z < 1.55))
            .map(|p| (p.id, p.team, p.name.clone(), p.x, p.y))
        {
            if let Some(p) = self.players.get_mut(&pid) {
                p.carrying_relic = true;
                p.score = p.score.saturating_add(15);
            }
            self.relic_carrier = Some(pid);
            self.push_event(4, px, py, 0, team, format!("{} TOOK RELIC!", name));
        }

        // Check 5-minute match timeout or player-scaled early victory (500 pts @ 12P, 1600 pts @ 40P, 2560 pts @ 64P)
        let score_limit = MATCH_SCORE_LIMIT.max((self.players.len() as u16).saturating_mul(40));
        if self.score_west >= score_limit
            || self.score_east >= score_limit
            || self.match_time_left_s == 0
        {
            self.match_phase = if self.score_west > self.score_east {
                MATCH_PHASE_WEST_VICTORY
            } else if self.score_east > self.score_west {
                MATCH_PHASE_EAST_VICTORY
            } else {
                MATCH_PHASE_DRAW
            };
            let mut best_name = "NONE".to_string();
            let mut best_metric = 0u32;
            let mut best_kills = 0u16;
            let mut best_is_human = false;
            let mut best_id = u32::MAX;
            for p in self.players.values() {
                let metric = (p.kills as u32).saturating_mul(25).saturating_add(p.score);
                let is_human = !p.is_bot;
                let better = metric > best_metric
                    || (metric == best_metric && p.kills > best_kills)
                    || (metric == best_metric && p.kills == best_kills && is_human && !best_is_human)
                    || (metric == best_metric && p.kills == best_kills && is_human == best_is_human && p.id < best_id);
                if better {
                    best_metric = metric;
                    best_kills = p.kills;
                    best_is_human = is_human;
                    best_id = p.id;
                    best_name = p.name.clone();
                }
            }
            self.mvp_name = best_name;
            self.mvp_score = best_metric;
            self.celebration_started = Some(Instant::now());
            self.match_time_left_s = CELEBRATION_SECS;
            self.projectiles.clear();
            let banner = match self.match_phase {
                MATCH_PHASE_WEST_VICTORY => "WEST VICTORY!",
                MATCH_PHASE_EAST_VICTORY => "EAST VICTORY!",
                _ => "MATCH DRAW!",
            };
            self.push_event(5, gw * 0.5, gh * 0.5, 0, 2, banner.to_string());
            log::info!(
                "Match ended: {} (West {} - {} East) | MVP: {} ({} pts)",
                banner, self.score_west, self.score_east, self.mvp_name, self.mvp_score
            );
        }
    }

    fn process_ws_commands(&mut self) {
        let Some(ws) = self.ws_bridge.clone() else {
            return;
        };
        let mut world_changed = false;
        for cmd in ws.drain_commands() {
            match cmd {
                ws_bridge::WsCommand::CycleBiome => {
                    self.cycle_biome();
                    world_changed = true;
                }
                ws_bridge::WsCommand::CycleEvo => {
                    self.cycle_evolution_speed();
                    world_changed = true;
                }
                ws_bridge::WsCommand::CycleLaws => {
                    self.cycle_world_law();
                    world_changed = true;
                }
                ws_bridge::WsCommand::ResetWorld => {
                    self.reset_match();
                    world_changed = true;
                }
                ws_bridge::WsCommand::RequestFullSync { peer_id } => {
                    ws.queue_full_sync_peer(peer_id);
                }
            }
        }
        if world_changed {
            self.broadcast_ws_entities(true);
        }
    }

    fn encode_ws_brick_frame(&self, bx: u16, by: u16) -> Arc<[u8]> {
        let (dm, at) = self.volume.encode_3d_brick_16(bx as usize, by as usize);
        let payload = ws_bridge::encode_brick_payload(bx, by, self.grid_d as u8, &dm, &at);
        Arc::from(ws_bridge::encode_ws_binary_frame(&payload))
    }

    fn build_ws_entities_json(&self) -> String {
        let mut sorted_players: Vec<&PlayerEntity> = self.players.values().collect();
        sorted_players.sort_by_key(|p| p.id);

        let tick = self.sim_tick;
        let mut active_beams: Vec<(f32, &ActuatorPulseSnap)> = self
            .actuator_pulses
            .iter()
            .filter(|(exp, _, _)| tick <= *exp)
            .map(|(_, energy, snap)| (*energy, snap))
            .collect();
        active_beams.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        active_beams.truncate(16);

        let mut json = String::with_capacity(
            768 + sorted_players.len() * 140 + self.projectiles.len() * 96 + active_beams.len() * 128,
        );
        use std::fmt::Write as _;
        let _ = write!(
            json,
            "{{\"tick\":{},\"world_day\":{},\"players\":[",
            self.sim_tick, self.world_day
        );
        for (i, p) in sorted_players.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let (r, g, b) = if p.team == 0 {
                (0.16f32, 0.54f32, 0.98f32) // West team (blue)
            } else {
                (0.94f32, 0.24f32, 0.24f32) // East team (red)
            };
            let yaw = p.aim_y.atan2(p.aim_x);
            // WebGPU client axis convention: x = east (p.x), y = altitude (p.z), z = north (p.y)
            let _ = write!(
                json,
                "{{\"id\":{},\"name\":{},\"team\":{},\"class_id\":{},\"squad\":{},\"x\":{:.3},\"y\":{:.3},\"z\":{:.3},\"yaw\":{:.3},\"color\":[{:.2},{:.2},{:.2}],\"scale\":1.0}}",
                p.id,
                json_escape_str(&p.name),
                p.team,
                p.class_id,
                p.squad_id,
                p.x,
                p.z,
                p.y,
                yaw,
                r,
                g,
                b
            );
        }
        json.push_str("],\"projectiles\":[");
        for (i, pr) in self.projectiles.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let (r, g, b) = match pr.element {
                0 => (1.00f32, 0.46f32, 0.12f32), // Fire orb
                1 => (0.22f32, 0.86f32, 1.00f32), // Frost orb
                _ => (0.76f32, 0.38f32, 0.98f32), // Oil / Arcane orb
            };
            let rad = pr.radius.clamp(0.25, 1.2);
            let _ = write!(
                json,
                "{{\"id\":{},\"x\":{:.3},\"y\":{:.3},\"z\":{:.3},\"color\":[{:.2},{:.2},{:.2}],\"radius\":{:.2}}}",
                pr.id, pr.x, pr.z, pr.y, r, g, b, rad
            );
        }
        json.push_str("],\"beams\":[");
        for (i, &(_, pulse)) in active_beams.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            let (r, g, b) = if pulse.thermal_flux > 0.25 {
                (1.00f32, 0.44f32, 0.12f32) // Thermal heat beam
            } else if pulse.thermal_flux < -0.25 {
                (0.22f32, 0.86f32, 1.00f32) // Cryo beam
            } else if pulse.sp_oil > 0.20 {
                (0.74f32, 0.34f32, 0.96f32) // Volatile oil stream
            } else if pulse.sp_water > 0.20 {
                (0.18f32, 0.66f32, 1.00f32) // Hydro stream
            } else {
                (0.84f32, 0.88f32, 0.94f32) // Kinetic / Smoke pulse
            };
            // Match WebGPU axis convention: x = east, y = altitude (z), z = north (y)
            let _ = write!(
                json,
                "{{\"x0\":{:.3},\"y0\":{:.3},\"z0\":{:.3},\"x1\":{:.3},\"y1\":{:.3},\"z1\":{:.3},\"color\":[{:.2},{:.2},{:.2}]}}",
                pulse.x0, pulse.z0, pulse.y0, pulse.x1, pulse.z1, pulse.y1, r, g, b
            );
        }
        let ev = self.volume.evolution_stats;
        let epoch_name = self.volume.evolution_epoch_name();
        let status = format!("{} Z={}", biome_name(self.biome_id), self.grid_d);
        let _ = write!(
            json,
            "],\"evolution\":{{\"epoch\":{},\"basalt\":{},\"trees\":{},\"glaciers\":{},\"erosion\":{}}},\"telemetry\":{{\"server_tps\":{:.1},\"online_players\":{},\"status\":{}}}",
            json_escape_str(epoch_name),
            ev.basalt_forged,
            ev.saplings_grown,
            ev.glaciers_formed,
            ev.erosion_events,
            self.measured_tps,
            sorted_players.len(),
            json_escape_str(&status)
        );
        let tactical_slice: Vec<ws_bridge::WsTacticalEvent> =
            self.tactical_events.iter().cloned().collect();
        ws_bridge::append_tactical_events_json(&mut json, &tactical_slice, 5);
        let laws = self.volume.world_laws;
        ws_bridge::append_world_law_json(
            &mut json,
            self.volume.world_law_preset,
            WorldLaws::preset_name(self.volume.world_law_preset),
            laws.gravity_g,
            laws.ambient_temp_shift,
            laws.precipitation,
        );
        json.push('}');
        json
    }

    fn broadcast_ws_entities(&self, force: bool) {
        let Some(ref ws) = self.ws_bridge else {
            return;
        };
        if !ws.has_clients() || (!force && self.sim_tick % 2 != 0) {
            return;
        }

        let json = self.build_ws_entities_json();
        let payload = ws_bridge::encode_entities_payload(&json);
        let frame: Arc<[u8]> = Arc::from(ws_bridge::encode_ws_binary_frame(&payload));
        ws.broadcast_raw_frame(frame);
    }

    fn broadcast_entities_30hz(&mut self) {
        self.tps_window_ticks = self.tps_window_ticks.saturating_add(1);
        let tps_elapsed = self.tps_window_start.elapsed().as_secs_f32();
        if tps_elapsed >= 1.0 {
            self.measured_tps = (self.tps_window_ticks as f32 / tps_elapsed).clamp(1.0, 120.0);
            self.tps_window_ticks = 0;
            self.tps_window_start = Instant::now();
        }
        self.broadcast_ws_entities(false);

        if self.peers.is_empty() {
            return;
        }
        let tick = self.sim_tick;
        let surf_l = self.surf_layer();
        let is_z16 = self.is_z16();
        let players: Vec<PlayerSnap> = self
            .players
            .values()
            .map(|p| {
                let pl = (p.z.round() as usize).clamp(if is_z16 { 0 } else { surf_l }, self.grid_d - 1);
                let mut status = 0u8;
                if self.physics_native {
                    let phys = self.volume.sample_entity_physics(p.x, p.y, pl);
                    if phys.temp_c > 120.0 { status |= STATUS_BURNING; }
                    if phys.temp_c < -15.0 { status |= STATUS_CHILLED; }
                    if phys.water_frac > 0.18 && phys.ice_frac < 0.30 { status |= STATUS_WET; }
                    if phys.oil_frac > 0.12 { status |= STATUS_OILED; }
                } else {
                    let hz = self.sample_hazard(p.x, p.y, pl);
                    if (hz & (1 << 2)) != 0 { status |= STATUS_BURNING; }
                    if (hz & (1 << 3)) != 0 || tick < p.chilled_until { status |= STATUS_CHILLED; }
                    if (hz & (1 << 4)) != 0 { status |= STATUS_WET; }
                    if tick < p.oiled_until { status |= STATUS_OILED; }
                }
                if tick < p.shield_until { status |= STATUS_SHIELDED; }
                if tick < p.haste_until { status |= STATUS_HASTE; }
                PlayerSnap {
                    id: p.id,
                    name: p.name.clone(),
                    team: p.team,
                    class_id: p.class_id,
                    is_bot: p.is_bot,
                    x: p.x,
                    y: p.y,
                    z: p.z,
                    vx: p.vx,
                    vy: p.vy,
                    vz: p.vz,
                    aim_x: p.aim_x,
                    aim_y: p.aim_y,
                    hp: p.hp,
                    max_hp: p.max_hp,
                    mana: p.mana.round() as u16,
                    max_mana: p.max_mana.round() as u16,
                    stamina: p.stamina.round() as u16,
                    max_stamina: p.max_stamina.round() as u16,
                    level: p.level,
                    xp: p.xp,
                    kills: p.kills,
                    deaths: p.deaths,
                    status_flags: status,
                    dashing: tick < p.dash_until,
                    melee_active: tick < p.melee_until,
                    ult_active: tick < p.ult_until,
                    carrying_relic: p.carrying_relic,
                    dash_cd_frac: (p.dash_cd.saturating_sub(tick) as f32 / 32.0).clamp(0.0, 1.0),
                    ability_e_cd_frac: (p.ability_e_cd.saturating_sub(tick) as f32 / 120.0).clamp(0.0, 1.0),
                    ult_r_cd_frac: (p.ult_r_cd.saturating_sub(tick) as f32 / 360.0).clamp(0.0, 1.0),
                    ack_input_seq: p.ack_input_seq,
                    score: p.score,
                }
            })
            .collect();
        let projectiles: Vec<ProjectileSnap> = self
            .projectiles
            .iter()
            .map(|pr| {
                let dscale = self.sample_time_dome_scale(pr.x, pr.y);
                ProjectileSnap {
                    id: pr.id,
                    element: pr.element,
                    x: pr.x,
                    y: pr.y,
                    z: pr.z,
                    vx: pr.vx * dscale,
                    vy: pr.vy * dscale,
                    vz: pr.vz * dscale,
                    radius: pr.radius,
                }
            })
            .collect();
        let time_domes: Vec<TimeDomeSnap> = self
            .time_domes
            .iter()
            .map(|d| TimeDomeSnap {
                id: d.id,
                x: d.x,
                y: d.y,
                radius: d.radius,
                dt_scale: d.dt_scale,
                ttl_s: (d.expires_tick.saturating_sub(tick) as f32) / 30.0,
                owner_team: d.owner_team,
            })
            .collect();
        let bodies: Vec<CellBodySnap> = self
            .rafts
            .iter()
            .map(|b| CellBodySnap { id: b.id, x: b.x, y: b.y, z: b.z, vx: b.vx, vy: b.vy, vz: b.vz, radius: b.radius })
            .collect();
        let shrines: Vec<ShrineSnap> = self
            .shrines
            .iter()
            .map(|s| ShrineSnap { id: s.id, name: s.name.clone(), x: s.x, y: s.y, progress: s.progress, owner: s.owner })
            .collect();
        let events: Vec<CombatEvent> = self.events.iter().rev().take(10).map(|(_, e)| e.clone()).collect();

        let mut sorted: Vec<f32> = self.infer_window.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99 = sorted
            .get(((sorted.len() as f32) * 0.99) as usize)
            .or_else(|| sorted.last())
            .copied()
            .unwrap_or(self.infer_ms_last);

        let bot_cnt = self.players.values().filter(|p| p.is_bot).count() as u16;
        let telemetry = TelemetrySnap {
            mode: self.sim_mode.clone(),
            backend: self.backend_name.clone(),
            ecosystem_on: self.ecosystem_on,
            grid_w: self.grid_w as u16,
            grid_h: self.grid_h as u16,
            grid_d: self.grid_d as u8,
            field_tick: self.volume.tick,
            infer_ms_last: self.infer_ms_last,
            infer_ms_p99: p99,
            eco_infer_ms: self.eco_infer_ms,
            proj_active_pct: self.proj_active_pct,
            mass_error: self.mass_error,
            energy_error: self.energy_error,
            wire_kbps: self.wire_kbps,
            aoi_chunks_sent: self.aoi_chunks_sent_last,
            total_chunks: (self.chunks_x * self.chunks_y * self.grid_d) as u16,
            active_bots: bot_cnt,
            neural_bots_on: self.neural_bots_on,
            bot_infer_ms: self.bot_infer_ms,
            weather_kind: self.weather_kind,
            day_time: self.day_time,
            wind_x: self.wind_x,
            wind_y: self.wind_y,
            score_west: self.score_west,
            score_east: self.score_east,
            acoustic_ms: self.acoustic_ms,
            world_sparse_pct: self.world_sparse_pct,
            physics_native: self.physics_native,
            time_scale: self.time_scale,
            world_day: self.world_day,
            neural_3d_vertical: self.neural_3d_vertical,
            active_time_domes: time_domes.len().min(255) as u8,
            system2_slm_active: self.system2_slm_active,
        };
        let match_secs = self.match_start_instant.elapsed().as_secs() as u32;
        let kill_feed: Vec<KillFeedEntry> = self.kill_feed.iter().cloned().collect();
        let speech_bubbles: Vec<SpeechBubbleSnap> =
            self.speech_bubbles.iter().map(|(_, b)| b.clone()).collect();
        let mut active_pulses: Vec<(f32, ActuatorPulseSnap)> = self
            .actuator_pulses
            .iter()
            .filter(|(exp, _, _)| tick <= *exp)
            .map(|(_, energy, snap)| (*energy, snap.clone()))
            .collect();
        active_pulses.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let actuator_pulses: Vec<ActuatorPulseSnap> =
            active_pulses.into_iter().take(16).map(|(_, s)| s).collect();

        let mut dead = Vec::new();
        let mut total_bytes = 0u64;
        for (&pid, peer) in &self.peers {
            let echo = self.players.get(&pid).map(|p| p.echo_client_time_ms).unwrap_or(0);
            let msg = ServerDatagram::EntitySnapshot {
                sim_tick: tick,
                match_time_secs: match_secs,
                your_player_id: pid,
                echo_client_time_ms: echo,
                relic_x: self.relic_x,
                relic_y: self.relic_y,
                players: players.clone(),
                projectiles: projectiles.clone(),
                bodies: bodies.clone(),
                shrines: shrines.clone(),
                events: events.clone(),
                telemetry: telemetry.clone(),
                bot_brains: self.bot_brains.clone(),
                kill_feed: kill_feed.clone(),
                speech_bubbles: speech_bubbles.clone(),
                actuator_pulses: actuator_pulses.clone(),
                match_phase: self.match_phase,
                match_time_left_s: self.match_time_left_s,
                mvp_name: self.mvp_name.clone(),
                mvp_score: self.mvp_score,
                time_domes: time_domes.clone(),
                world_law_preset: self.volume.world_law_preset,
            };
            let raw = encode_server_datagram(&msg);
            total_bytes += raw.len() as u64;
            match peer.conn.send_datagram(Bytes::from(raw.clone())) {
                Ok(()) => {}
                Err(quinn::SendDatagramError::ConnectionLost(_)) => dead.push(pid),
                Err(quinn::SendDatagramError::TooLarge) => {
                    let conn_c = peer.conn.clone();
                    let rt_h = self.rt_handle.clone();
                    rt_h.spawn(async move {
                        if let Ok(mut send) = conn_c.open_uni().await {
                            let mut framed = Vec::with_capacity(1 + raw.len());
                            framed.push(0xFE);
                            framed.extend_from_slice(&raw);
                            let _ = send.write_all(&framed).await;
                            let _ = send.finish();
                        }
                    });
                }
                Err(_) => {}
            }
        }
        self.record_wire_bytes(total_bytes / (self.peers.len().max(1) as u64));
        for pid in dead {
            self.peers.remove(&pid);
            self.players.remove(&pid);
        }
    }

    fn encode_chunk_pkt(&self, layer: u8, cx: u16, cy: u16) -> Bytes {
        let payload = self.volume.layers[layer as usize].encode_aplus_chunk(cx, cy);
        let pkt = ServerDatagram::APlusChunk {
            field_tick: self.volume.tick,
            layer,
            chunk_id: payload.chunk_id as u16,
            cx: payload.cx as u8,
            cy: payload.cy as u8,
            lz4_cells: compress_aplus_cells(&payload.cells),
        };
        Bytes::from(encode_server_datagram(&pkt))
    }

    fn encode_brick3d_pkt(&self, bx: u16, by: u16) -> Bytes {
        let brick_id = (by as u32) * (self.chunks_x as u32) + (bx as u32);
        let (dm, at) = self.volume.encode_3d_brick_16(bx as usize, by as usize);
        let brick = encode_brick3d_lz4(brick_id, bx, by, self.grid_d as u8, &dm, &at);
        let pkt = ServerDatagram::VolumetricBrick3D(brick);
        Bytes::from(encode_server_datagram(&pkt))
    }

    fn send_packet_datagram_or_uni(rt_handle: &tokio::runtime::Handle, conn: &Connection, raw: Bytes) {
        if let Err(quinn::SendDatagramError::TooLarge) = conn.send_datagram(raw.clone()) {
            let conn_c = conn.clone();
            rt_handle.spawn(async move {
                if let Ok(mut send) = conn_c.open_uni().await {
                    let mut framed = Vec::with_capacity(1 + raw.len());
                    framed.push(0xFE);
                    framed.extend_from_slice(&raw);
                    let _ = send.write_all(&framed).await;
                    let _ = send.finish();
                }
            });
        }
    }

    /// Sends a batch of packets (e.g. `VolumetricBrick3D` full sync or 10 Hz dirty bricks) on a single
    /// Tokio task via `rt_handle`, trying unreliable QUIC datagram first and falling back cleanly to
    /// `0xFE`-framed uni-streams without ever panicking from the 10 Hz `std::thread` or exceeding
    /// QUIC's concurrent uni-stream window!
    fn send_packets_batched(rt_handle: &tokio::runtime::Handle, conn: Connection, pkts: Vec<Bytes>) {
        if pkts.is_empty() {
            return;
        }
        rt_handle.spawn(async move {
            for raw in pkts {
                match conn.send_datagram(raw.clone()) {
                    Ok(()) => {}
                    Err(quinn::SendDatagramError::TooLarge) => {
                        if let Ok(mut send) = conn.open_uni().await {
                            let mut framed = Vec::with_capacity(1 + raw.len());
                            framed.push(0xFE);
                            framed.extend_from_slice(&raw);
                            let _ = send.write_all(&framed).await;
                            let _ = send.finish();
                        }
                    }
                    Err(quinn::SendDatagramError::ConnectionLost(_)) => break,
                    Err(_) => {}
                }
            }
        });
    }

    fn broadcast_chunks_10hz(&mut self, dirty: &[(u8, u16, u16)]) {
        if let Some(ws) = self.ws_bridge.clone() {
            if ws.has_clients() {
                let mut dirty_xy: Vec<(u16, u16)> = Vec::new();
                for &(_layer, cx, cy) in dirty {
                    if !dirty_xy.contains(&(cx, cy)) {
                        dirty_xy.push((cx, cy));
                    }
                }
                ws.pump_bricks(
                    &dirty_xy,
                    ws_bridge::WS_INITIAL_BRICKS_PER_TICK,
                    |bx, by| self.encode_ws_brick_frame(bx, by),
                );
            }
        }

        if self.peers.is_empty() {
            return;
        }
        let is_z16 = self.is_z16();
        let surf_l = self.surf_layer() as u8;
        let total_2d = self.chunks_x * self.chunks_y;
        let aoi_r = self.aoi_radius as i32;
        let rt_handle = self.rt_handle.clone();

        // Pre-encode dirty 3D chunks once (on Z=16 only surface APlusChunks + full 16-layer VolumetricBrick3D!)
        let mut dirty_map: HashMap<(u8, u16, u16), Bytes> = HashMap::with_capacity(dirty.len());
        let mut dirty_bricks: HashMap<(u16, u16), Bytes> = HashMap::new();
        for &(layer, cx, cy) in dirty {
            if !is_z16 || layer == surf_l {
                dirty_map.insert((layer, cx, cy), self.encode_chunk_pkt(layer, cx, cy));
            }
            if is_z16 && !dirty_bricks.contains_key(&(cx, cy)) {
                dirty_bricks.insert((cx, cy), self.encode_brick3d_pkt(cx, cy));
            }
        }

        let any_full = self.peers.values().any(|p| p.needs_full_sync);
        let (full_pkts, full_bricks): (Vec<Bytes>, Vec<Bytes>) = if any_full {
            if is_z16 {
                // On --layers 16: stream full surface APlusChunks (minimap/2D fallback) + ALL VolumetricBrick3D bricks!
                let mut surf_v = Vec::with_capacity(total_2d);
                let mut brick_v = Vec::with_capacity(total_2d);
                for by in 0..(self.chunks_y as u16) {
                    for bx in 0..(self.chunks_x as u16) {
                        if let Some(b) = dirty_map.get(&(surf_l, bx, by)) {
                            surf_v.push(b.clone());
                        } else {
                            surf_v.push(self.encode_chunk_pkt(surf_l, bx, by));
                        }
                        if let Some(b) = dirty_bricks.get(&(bx, by)) {
                            brick_v.push(b.clone());
                        } else {
                            brick_v.push(self.encode_brick3d_pkt(bx, by));
                        }
                    }
                }
                (surf_v, brick_v)
            } else {
                let mut v = Vec::with_capacity(total_2d * 2);
                // Always send full surface layer (L1) + sparse non-empty chunks on L2 & L3!
                for layer in 0..(self.grid_d as u8) {
                    if layer == 0 && self.grid_d >= 2 {
                        // L0 is deep riverbed beneath L1; only upper layers need client visuals!
                        continue;
                    }
                    for cy in 0..(self.chunks_y as u16) {
                        for cx in 0..(self.chunks_x as u16) {
                            if layer != surf_l && !self.volume.is_chunk_non_empty(layer as usize, cx, cy) {
                                continue;
                            }
                            if let Some(b) = dirty_map.get(&(layer, cx, cy)) {
                                v.push(b.clone());
                            } else {
                                v.push(self.encode_chunk_pkt(layer, cx, cy));
                            }
                        }
                    }
                }
                (v, Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        };

        let player_chunks: HashMap<u32, (i32, i32)> = self
            .players
            .iter()
            .map(|(&pid, p)| (pid, ((p.x as i32) / CHUNK_SIZE as i32, (p.y as i32) / CHUNK_SIZE as i32)))
            .collect();

        let cx_dim = self.chunks_x;
        let mut macro_cache: HashMap<(u16, u16), Bytes> = HashMap::new();
        let mut macro_brick_cache: HashMap<(u16, u16), Bytes> = HashMap::new();
        if aoi_r > 0 {
            let cursors: Vec<usize> = self
                .peers
                .values()
                .filter(|p| !p.needs_full_sync)
                .map(|p| p.macro_cursor)
                .collect();
            for cur in cursors {
                for step in 0..12 {
                    let ci = (cur + step) % total_2d;
                    let cx = (ci % cx_dim) as u16;
                    let cy = (ci / cx_dim) as u16;
                    if !dirty_map.contains_key(&(surf_l, cx, cy)) && !macro_cache.contains_key(&(cx, cy)) {
                        macro_cache.insert((cx, cy), self.encode_chunk_pkt(surf_l, cx, cy));
                    }
                    if is_z16
                        && step < 2
                        && !dirty_bricks.contains_key(&(cx, cy))
                        && !macro_brick_cache.contains_key(&(cx, cy))
                    {
                        macro_brick_cache.insert((cx, cy), self.encode_brick3d_pkt(cx, cy));
                    }
                }
            }
        }
        let mut bytes_one_peer = 0u64;
        let mut chunks_sent_one = 0u16;
        for (&pid, peer) in self.peers.iter_mut() {
            let mut peer_b = 0u64;
            let mut sent_cnt = 0u16;
            if peer.needs_full_sync {
                peer.needs_full_sync = false;
                if is_z16 {
                    let mut batch_pkts = Vec::with_capacity(full_bricks.len() + full_pkts.len());
                    // Prioritize full 3D VolumetricBrick3D world first so client renders all 16 layers immediately!
                    for b in &full_bricks {
                        peer_b += b.len() as u64;
                        sent_cnt = sent_cnt.saturating_add(1);
                        batch_pkts.push(b.clone());
                    }
                    for b in &full_pkts {
                        peer_b += b.len() as u64;
                        sent_cnt = sent_cnt.saturating_add(1);
                        batch_pkts.push(b.clone());
                    }
                    log::info!(
                        "Full Z={} sync queued for player {}: {} VolumetricBrick3D bricks + {} surf chunks ({:.1} KB)",
                        self.grid_d,
                        pid,
                        full_bricks.len(),
                        full_pkts.len(),
                        peer_b as f32 / 1024.0
                    );
                    Self::send_packets_batched(&rt_handle, peer.conn.clone(), batch_pkts);
                } else {
                    for b in &full_pkts {
                        peer_b += b.len() as u64;
                        sent_cnt = sent_cnt.saturating_add(1);
                        Self::send_packet_datagram_or_uni(&rt_handle, &peer.conn, b.clone());
                    }
                }
            } else {
                let (pcx, pcy) = player_chunks
                    .get(&pid)
                    .copied()
                    .unwrap_or(((cx_dim / 2) as i32, (self.chunks_y / 2) as i32));
                for (&(layer, cx, cy), b) in &dirty_map {
                    if layer == 0 && self.grid_d >= 2 {
                        continue;
                    }
                    let in_aoi = aoi_r == 0
                        || ((cx as i32 - pcx).abs() <= aoi_r && (cy as i32 - pcy).abs() <= aoi_r);
                    if in_aoi {
                        peer_b += b.len() as u64;
                        sent_cnt = sent_cnt.saturating_add(1);
                        let _ = peer.conn.send_datagram(b.clone());
                    }
                }
                let mut z16_bricks_out: Vec<Bytes> = Vec::new();
                if is_z16 {
                    for (&(bx, by), b) in &dirty_bricks {
                        let in_aoi = aoi_r == 0
                            || ((bx as i32 - pcx).abs() <= aoi_r && (by as i32 - pcy).abs() <= aoi_r);
                        if in_aoi {
                            peer_b += b.len() as u64;
                            sent_cnt = sent_cnt.saturating_add(1);
                            z16_bricks_out.push(b.clone());
                        }
                    }
                }
                if aoi_r > 0 {
                    for step in 0..12 {
                        let ci = (peer.macro_cursor + step) % total_2d;
                        let cx = (ci % cx_dim) as u16;
                        let cy = (ci / cx_dim) as u16;
                        if let Some(b) = macro_cache.get(&(cx, cy)) {
                            peer_b += b.len() as u64;
                            sent_cnt = sent_cnt.saturating_add(1);
                            let _ = peer.conn.send_datagram(b.clone());
                        }
                        if is_z16 && step < 2 {
                            if let Some(b) = macro_brick_cache.get(&(cx, cy)) {
                                peer_b += b.len() as u64;
                                sent_cnt = sent_cnt.saturating_add(1);
                                z16_bricks_out.push(b.clone());
                            }
                        }
                    }
                    peer.macro_cursor = (peer.macro_cursor + 12) % total_2d;
                }
                if !z16_bricks_out.is_empty() {
                    Self::send_packets_batched(&rt_handle, peer.conn.clone(), z16_bricks_out);
                }
            }
            bytes_one_peer = peer_b;
            chunks_sent_one = sent_cnt;
        }
        self.aoi_chunks_sent_last = chunks_sent_one;
        self.record_wire_bytes(bytes_one_peer);
    }

    fn record_wire_bytes(&mut self, bytes_per_client: u64) {
        self.bytes_sent_window += bytes_per_client;
        let secs = self.bytes_window_start.elapsed().as_secs_f32();
        if secs >= 1.0 {
            self.wire_kbps = (self.bytes_sent_window as f32 / 1024.0) / secs;
            self.bytes_sent_window = 0;
            self.bytes_window_start = Instant::now();
        }
    }
}

fn make_quic_server_config() -> Result<ServerConfig, Box<dyn std::error::Error>> {
    let cert = rcgen::generate_simple_self_signed(vec![
        "localhost".into(),
        "127.0.0.1".into(),
        "100.116.218.113".into(),
        "spark-2b4e".into(),
    ])?;
    let cert_der = rustls::pki_types::CertificateDer::from(cert.cert);
    let key_der = rustls::pki_types::PrivateKeyDer::Pkcs8(
        rustls::pki_types::PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der()),
    );
    let mut cfg = ServerConfig::with_single_cert(vec![cert_der], key_der)?;
    let mut transport = quinn::TransportConfig::default();
    transport.max_idle_timeout(Some(Duration::from_secs(30).try_into()?));
    transport.keep_alive_interval(Some(Duration::from_secs(5)));
    transport.datagram_receive_buffer_size(Some(8 * 1024 * 1024));
    transport.datagram_send_buffer_size(8 * 1024 * 1024);
    transport.max_concurrent_uni_streams(512u32.into());
    cfg.transport_config(Arc::new(transport));
    Ok(cfg)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,ort=warn")).init();
    let args = Args::parse();

    let server_cfg = make_quic_server_config()?;
    let endpoint = Endpoint::server(server_cfg, args.bind)?;
    let state = Arc::new(Mutex::new(GameState::new(
        args.backend.clone(),
        args.cuda_addr.clone(),
        args.size,
        args.layers,
        args.biome,
        args.seed,
        args.bots,
        args.aoi_radius,
    )));
    let (grid_w, grid_h, grid_d, biome_id, world_seed, chunks_x, chunks_y, bot_counts) = {
        let gs = state.lock().unwrap();
        let mut cls_cnt = [0usize; 5];
        for p in gs.players.values().filter(|p| p.is_bot) {
            cls_cnt[(p.class_id as usize) % 5] += 1;
        }
        (gs.grid_w, gs.grid_h, gs.grid_d, gs.biome_id, gs.world_seed, gs.chunks_x, gs.chunks_y, cls_cnt)
    };
    if let Some(ref ws_addr) = args.ws_bind {
        let bridge = ws_bridge::WsBridge::start(ws_addr, chunks_x, chunks_y).await?;
        state.lock().unwrap().ws_bridge = Some(bridge);
        log::info!(
            "Wave 12B Live RFC 6455 WebSocket bridge listening on ws://{} ({}x{} bricks, Z={})",
            ws_addr,
            chunks_x,
            chunks_y,
            grid_d
        );
    }
    log::info!(
        "Frostmarsh v3 3D VOLUMETRIC QUIC Server listening on {} ({}x{}x{} = {} voxels, biome={} [{}], seed={}, backend={}, bots={} [Strider={} Skywing={} Tidekin={} Emberstalker={} Burrower={}], aoi_r={})",
        args.bind,
        grid_w,
        grid_h,
        grid_d,
        grid_w * grid_h * grid_d,
        biome_id,
        biome_name(biome_id),
        world_seed,
        args.backend,
        bot_counts.iter().sum::<usize>(),
        bot_counts[0],
        bot_counts[1],
        bot_counts[2],
        bot_counts[3],
        bot_counts[4],
        args.aoi_radius
    );

    // 10 Hz Batched 3D Neural World + Vertical Gravity/Buoyancy + Ecosystem Thread
    let field_state = state.clone();
    let backend_kind = args.backend.clone();
    let cuda_addr = args.cuda_addr.clone();
    let model_path = args.model.clone();
    let eco_path = args.ecosystem_model.clone();

    std::thread::spawn(move || {
        #[cfg(feature = "onnx")]
        let ort_env = ort::init().with_name("frostmarsh-quic-server").build().expect("ort init");
        let mut cuda_rpc = if backend_kind == "cuda-rpc" { Some(CudaRpcClient::new(cuda_addr)) } else { None };
        #[cfg(feature = "onnx")]
        let mut onnx = if backend_kind == "onnx" { Some(OnnxBackend::load(&ort_env, &model_path, &eco_path)) } else { None };
        #[cfg(not(feature = "onnx"))]
        let _ = (&model_path, &eco_path);

        let period = Duration::from_millis(100);
        let mut world_dt_backoff: u32 = 0;
        let mut world_3d_backoff: u32 = 0;
        loop {
            let t_loop = Instant::now();
            let (mode, eco_on, gw, gh, gd, weather_kind, day_time_now, batched_in, eco_in, fallback_raws, local_dt_map, has_active_domes) = {
                let gs = field_state.lock().unwrap();
                let sl = gs.surf_layer();
                let m = gs.sim_mode.clone();
                let e = gs.ecosystem_on;
                let bin = if m == "neural" { Some(gs.volume.encode_batched_input_tensor()) } else { None };
                let ein = if e { Some(gs.eco.build_input(&gs.volume.layers[sl])) } else { None };
                let raws = if m != "neural" { Some(gs.volume.refsim_raw_outputs_all()) } else { None };
                let has_d = !gs.time_domes.is_empty();
                let dt_map = gs.build_local_dt_map();
                (m, e, gs.grid_w, gs.grid_h, gs.grid_d, gs.weather_kind, gs.day_time, bin, ein, raws, dt_map, has_d)
            };

            let t0 = Instant::now();
            let (mut raw_layers, mut z_flux_opt, reported_ms, active_backend, used_neural_3d_z, used_world_dt) = if mode == "neural" {
                if let Some(ref mut rpc) = cuda_rpc {
                    let per = OUTPUT_CHANNELS * gw * gh;
                    let horiz_len = gd * per;
                    let z_len = gd.saturating_sub(1) * 5 * gw * gh;
                    let expected_3d_len = horiz_len + z_len;

                    // 1. Try op=9 (`OP_WORLD_DT_STEP` — WorldModelDT continuous local_dt Time-Dilation Domes!)
                    let mut op9_result: Option<(Vec<RawStepOutput>, Vec<f32>, f32)> = None;
                    if world_dt_backoff == 0 && gd >= 2 {
                        let bin = batched_in.as_ref().unwrap();
                        let mut dt_payload = Vec::with_capacity(bin.len() + local_dt_map.len());
                        dt_payload.extend_from_slice(bin);
                        dt_payload.extend_from_slice(&local_dt_map);
                        match rpc.call_batched(9, gd as u8, INPUT_CHANNELS as u16, gh as u16, gw as u16, &dt_payload) {
                            Ok((flat, gpu_ms)) if flat.len() == expected_3d_len => {
                                let mut layers_out = Vec::with_capacity(gd);
                                for z in 0..gd {
                                    let s = z * per;
                                    layers_out.push(RawStepOutput {
                                        width: gw,
                                        height: gh,
                                        data: flat[s..s + per].to_vec(),
                                    });
                                }
                                let z_flux = flat[horiz_len..expected_3d_len].to_vec();
                                op9_result = Some((layers_out, z_flux, gpu_ms));
                            }
                            Ok((flat, _)) => {
                                log::warn!(
                                    "CUDA op=9 returned unexpected payload len {} (expected {}), falling back to op=8",
                                    flat.len(),
                                    expected_3d_len
                                );
                                world_dt_backoff = 50;
                            }
                            Err(e) => {
                                // Daemon does not have world_model_dt.pt loaded; retry periodically
                                log::debug!("CUDA op=9 unavailable for {}x{}x{} ({}); falling back to op=8", gw, gh, gd, e);
                                world_dt_backoff = 50;
                            }
                        }
                    } else if world_dt_backoff > 0 {
                        world_dt_backoff -= 1;
                    }

                    if let Some((layers_out, z_flux, gpu_ms)) = op9_result {
                        (layers_out, Some(z_flux), gpu_ms, "cuda-fp16-dt".to_string(), true, true)
                    } else {
                        // 2. Fallback to op=8 (`WorldModel3D` v4)
                        let mut op8_result: Option<(Vec<RawStepOutput>, Vec<f32>, f32)> = None;
                        if world_3d_backoff == 0 && gd >= 2 {
                            match rpc.call_batched(8, gd as u8, INPUT_CHANNELS as u16, gh as u16, gw as u16, batched_in.as_ref().unwrap()) {
                                Ok((flat, gpu_ms)) if flat.len() == expected_3d_len => {
                                    let mut layers_out = Vec::with_capacity(gd);
                                    for z in 0..gd {
                                        let s = z * per;
                                        layers_out.push(RawStepOutput {
                                            width: gw,
                                            height: gh,
                                            data: flat[s..s + per].to_vec(),
                                        });
                                    }
                                    let z_flux = flat[horiz_len..expected_3d_len].to_vec();
                                    op8_result = Some((layers_out, z_flux, gpu_ms));
                                }
                                Ok((flat, _)) => {
                                    log::warn!(
                                        "CUDA op=8 returned unexpected payload len {} (expected {}), falling back to op=1",
                                        flat.len(),
                                        expected_3d_len
                                    );
                                    world_3d_backoff = 50;
                                }
                                Err(e) => {
                                    log::debug!("CUDA op=8 unavailable for {}x{}x{} ({}); falling back to op=1", gw, gh, gd, e);
                                    world_3d_backoff = 50;
                                }
                            }
                        } else if world_3d_backoff > 0 {
                            world_3d_backoff -= 1;
                        }

                        if let Some((layers_out, z_flux, gpu_ms)) = op8_result {
                            (layers_out, Some(z_flux), gpu_ms, "cuda-fp16-3d-v4".to_string(), true, false)
                        } else {
                            match rpc.call_batched(1, gd as u8, INPUT_CHANNELS as u16, gh as u16, gw as u16, batched_in.as_ref().unwrap()) {
                                Ok((flat, gpu_ms)) => {
                                    let mut layers_out = Vec::with_capacity(gd);
                                    for z in 0..gd {
                                        let s = z * per;
                                        layers_out.push(RawStepOutput {
                                            width: gw,
                                            height: gh,
                                            data: flat[s..s + per].to_vec(),
                                        });
                                    }
                                    (layers_out, None, gpu_ms, "cuda-fp16-3d".to_string(), false, false)
                                }
                                Err(e) => {
                                    log::warn!("CUDA 3D RPC failed ({}), fallback RefSim", e);
                                    let gs = field_state.lock().unwrap();
                                    (gs.volume.refsim_raw_outputs_all(), None, t0.elapsed().as_secs_f32() * 1000.0, "refsim-fallback".into(), false, false)
                                }
                            }
                        }
                    }
                } else {
                    #[cfg(feature = "onnx")]
                    if let Some(ref mut ob) = onnx {
                        let vclone = { field_state.lock().unwrap().volume.clone() };
                        match ob.step_volume(&vclone) {
                            Ok(v) => (v, None, t0.elapsed().as_secs_f32() * 1000.0, "onnx-cpu-3d".into(), false, false),
                            Err(_) => (vclone.refsim_raw_outputs_all(), None, t0.elapsed().as_secs_f32() * 1000.0, "refsim-fallback".into(), false, false),
                        }
                    } else {
                        let gs = field_state.lock().unwrap();
                        (gs.volume.refsim_raw_outputs_all(), None, t0.elapsed().as_secs_f32() * 1000.0, "refsim-3d".into(), false, false)
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        let gs = field_state.lock().unwrap();
                        (gs.volume.refsim_raw_outputs_all(), None, t0.elapsed().as_secs_f32() * 1000.0, "refsim-3d".into(), false, false)
                    }
                }
            } else {
                (fallback_raws.unwrap(), None, t0.elapsed().as_secs_f32() * 1000.0, "refsim-3d".into(), false, false)
            };

            // Graceful spatial time-dilation flux scaling when domes are active and op=9 WorldModelDT is unavailable
            if has_active_domes && !used_world_dt && local_dt_map.len() == gw * gh {
                let len = gw * gh;
                for raw in &mut raw_layers {
                    if raw.data.len() == OUTPUT_CHANNELS * len {
                        for y in 0..gh {
                            let row = y * gw;
                            let row_s = (y + 1).min(gh - 1) * gw;
                            for x in 0..gw {
                                let idx = row + x;
                                let dt_c = local_dt_map[idx];
                                let dt_e = dt_c.min(local_dt_map[row + (x + 1).min(gw - 1)]);
                                let dt_s = dt_c.min(local_dt_map[row_s + x]);
                                if (dt_c - 1.0).abs() > 1e-3 || (dt_e - 1.0).abs() > 1e-3 || (dt_s - 1.0).abs() > 1e-3 {
                                    for c in 0..4 { raw.data[c * len + idx] *= dt_e; }
                                    for c in 4..8 { raw.data[c * len + idx] *= dt_s; }
                                    raw.data[8 * len + idx] *= dt_e;
                                    raw.data[9 * len + idx] *= dt_s;
                                    for c in 10..OUTPUT_CHANNELS { raw.data[c * len + idx] *= dt_c; }
                                }
                            }
                        }
                    }
                }
                if let Some(ref mut zf) = z_flux_opt {
                    let n_if = gd.saturating_sub(1);
                    let stride = 5 * len;
                    if zf.len() >= n_if * stride {
                        for z in 0..n_if {
                            let base = z * stride;
                            for idx in 0..len {
                                let dt_c = local_dt_map[idx];
                                if (dt_c - 1.0).abs() > 1e-3 {
                                    for c in 0..5 {
                                        zf[base + c * len + idx] *= dt_c;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let (eco_out, eco_ms) = if eco_on {
                if let Some(einp) = eco_in {
                    let te = Instant::now();
                    if let Some(ref mut rpc) = cuda_rpc {
                        match rpc.call_batched(2, 1, 8, gh as u16, gw as u16, &einp) {
                            Ok((d, ms)) => (Some(d), ms),
                            Err(_) => (None, 0.0),
                        }
                    } else {
                        #[cfg(feature = "onnx")]
                        {
                            if let Some(ref mut ob) = onnx {
                                match ob.step_eco(einp, gh, gw) {
                                    Ok(d) => (Some(d), te.elapsed().as_secs_f32() * 1000.0),
                                    Err(_) => (None, 0.0),
                                }
                            } else { (None, 0.0) }
                        }
                        #[cfg(not(feature = "onnx"))]
                        { let _ = (einp, te); (None, 0.0) }
                    }
                } else { (None, 0.0) }
            } else { (None, 0.0) };

            {
                let mut gs = field_state.lock().unwrap();
                let sl = gs.surf_layer();
                let tick_now = gs.volume.tick;
                let evo_speed = gs.evo_speed;
                gs.volume.apply_weather_and_springs_3d(weather_kind, tick_now);
                gs.volume.apply_solar_cycle_3d(day_time_now, tick_now);
                if gs.volume.depth >= 12 && (evo_speed >= 4.0 || tick_now % 2 == 0) {
                    gs.volume.step_world_evolution_3d(evo_speed);
                    let new_epoch = gs.volume.evolution_epoch_name();
                    if new_epoch != gs.last_epoch_name {
                        gs.last_epoch_name = new_epoch;
                        let (mx, my) = (gs.grid_w as f32 * 0.5, gs.grid_h as f32 * 0.5);
                        gs.push_event(5, mx, my, 0, 2, format!("EPOCH: {}", new_epoch));
                        log::info!("Living World Epoch advanced -> {}", new_epoch);
                    }
                }
                let stats = gs.volume.step_3d_with_neural_z(&raw_layers, z_flux_opt.as_deref());
                gs.check_terrain_engineering_milestones();
                gs.neural_3d_vertical = used_neural_3d_z;
                if eco_on {
                    if let Some(ref eo) = eco_out {
                        let mut eco_tmp = std::mem::replace(&mut gs.eco, EcosystemState::new(1, 1));
                        eco_tmp.apply_output(eo, &mut gs.volume.layers[sl]);
                        gs.eco = eco_tmp;
                    }
                } else {
                    gs.volume.layers[sl].critter_density.fill(0);
                }
                gs.refresh_hazard_grids();
                gs.backend_name = active_backend;
                gs.infer_ms_last = reported_ms;
                gs.eco_infer_ms = eco_ms;
                if gs.infer_window.len() >= 100 { gs.infer_window.pop_front(); }
                gs.infer_window.push_back(reported_ms);
                gs.proj_active_pct = stats.proj_active_pct;
                gs.mass_error = stats.mass_error;
                gs.energy_error = stats.energy_error;
                gs.process_ws_commands();
                gs.broadcast_chunks_10hz(&stats.dirty_chunks);

                if stats.tick % 10 == 0 {
                    let bot_cnt = gs.players.values().filter(|p| p.is_bot).count();
                    let ev = gs.volume.evolution_stats;
                    let epoch_name = gs.volume.evolution_epoch_name();
                    log::info!(
                        "tick={} {}x{}x{} ({} vox) biome={} mode={} backend={} world={:.2}ms eco={:.2}ms grav_z={} buoy_z={} mass_err={} energy_err={} bots={} epoch=\"{}\" tunnels={} tunnel_mass={} evo={:.0}x basalt={} trees={} glaciers={} erosion={} wire={:.1}KB/s dirty={} peers={}",
                        stats.tick, gw, gh, gd, gw * gh * gd, biome_name(gs.biome_id), gs.sim_mode, gs.backend_name,
                        reported_ms, eco_ms, stats.vertical_gravity_mass, stats.vertical_buoyancy_mass,
                        stats.mass_error, stats.energy_error, bot_cnt, epoch_name, gs.tunnels_carved,
                        gs.tunnel_mass_displaced, gs.evo_speed, ev.basalt_forged, ev.saplings_grown,
                        ev.glaciers_formed, ev.erosion_events, gs.wire_kbps, stats.dirty_chunks.len(), gs.peers.len()
                    );
                }
            }

            let elapsed = t_loop.elapsed();
            if elapsed < period {
                std::thread::sleep(period - elapsed);
            } else {
                std::thread::yield_now();
            }
        }
    });

    // 30 Hz 3D Combat & Entity Snapshot broadcast
    let combat_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(33));
        loop {
            interval.tick().await;
            tokio::task::block_in_place(|| {
                let mut gs = combat_state.lock().unwrap();
                gs.process_ws_commands();
                gs.tick_combat_30hz();
                gs.broadcast_entities_30hz();
            });
        }
    });

    while let Some(incoming) = endpoint.accept().await {
        let st = state.clone();
        tokio::spawn(async move {
            let Ok(conn) = incoming.await else { return; };
            let remote = conn.remote_address();
            let (pid, gw, gh, gd) = {
                let mut gs = st.lock().unwrap();
                let pid = gs.spawn_player(format!("Hero-{}", remote.port() % 1000), 0, CLASS_WARDEN, false);
                gs.peers.insert(pid, ClientPeer { conn: conn.clone(), needs_full_sync: true, macro_cursor: 0 });
                (pid, gs.grid_w as u16, gs.grid_h as u16, gs.grid_d as u8)
            };
            log::info!("QUIC client connected from {} -> player_id={} ({}x{}x{})", remote, pid, gw, gh, gd);

            if let Ok(mut send) = conn.open_uni().await {
                let welcome = encode_msg(&ServerReliable::Welcome { your_player_id: pid, width: gw, height: gh, depth: gd });
                let _ = send.write_all(&welcome).await;
                let _ = send.finish();
            }

            let st_dg = st.clone();
            let conn_dg = conn.clone();
            let dg_task = tokio::spawn(async move {
                while let Ok(bytes) = conn_dg.read_datagram().await {
                    if let Some(inp) = decode_msg::<ClientInputDatagram>(&bytes) {
                        let mut gs = st_dg.lock().unwrap();
                        gs.apply_input(pid, &inp);
                    }
                }
            });

            while let Ok(mut recv) = conn.accept_uni().await {
                if let Ok(bytes) = recv.read_to_end(4096).await {
                    if let Some(cmd) = decode_msg::<ClientReliable>(&bytes) {
                        let mut gs = st.lock().unwrap();
                        match cmd {
                            ClientReliable::Join { name, team } => {
                                if let Some(p) = gs.players.get_mut(&pid) {
                                    p.name = name;
                                    p.team = team % 2;
                                }
                            }
                            ClientReliable::SelectClass { class_id } => {
                                if let Some(p) = gs.players.get_mut(&pid) {
                                    p.class_id = class_id % 5;
                                    let (mhp, mmp, mst, _) = PlayerEntity::class_stats(p.class_id, p.level);
                                    p.max_hp = mhp;
                                    p.hp = p.hp.min(mhp);
                                    p.max_mana = mmp;
                                    p.mana = p.mana.min(mmp);
                                    p.max_stamina = mst;
                                    p.stamina = p.stamina.min(mst);
                                }
                            }
                            ClientReliable::ToggleSimMode => {
                                gs.sim_mode = if gs.sim_mode == "neural" { "refsim".into() } else { "neural".into() };
                            }
                            ClientReliable::ToggleEcosystem => gs.ecosystem_on = !gs.ecosystem_on,
                            ClientReliable::ToggleNeuralBots => {
                                gs.neural_bots_on = !gs.neural_bots_on;
                                let lbl = if gs.neural_bots_on {
                                    "AI: NEURAL-Z LATENT BOTS!"
                                } else {
                                    "AI: SCRIPTED BOTS"
                                };
                                let (mx, my) = (gs.grid_w as f32 * 0.5, gs.grid_h as f32 * 0.5);
                                gs.push_event(5, mx, my, 0, 2, lbl.to_string());
                            }
                            ClientReliable::CycleWeather => {
                                gs.weather_kind = (gs.weather_kind + 1) % 4;
                                let wname = match gs.weather_kind {
                                    WEATHER_BLIZZARD => "BLIZZARD FRONT!",
                                    WEATHER_SQUALL => "THAW SQUALL!",
                                    WEATHER_EMBER => "EMBER DUSK!",
                                    _ => "CLEAR FROST!",
                                };
                                let (mx, my) = (gs.grid_w as f32 * 0.5, gs.grid_h as f32 * 0.5);
                                gs.push_event(5, mx, my, 0, 2, wname.to_string());
                            }
                            ClientReliable::ResetWorld => {
                                gs.reset_match();
                            }
                            ClientReliable::RequestFullSync => {
                                if let Some(peer) = gs.peers.get_mut(&pid) { peer.needs_full_sync = true; }
                            }
                            ClientReliable::ChatCommand { text, volume } => {
                                let clean: String = text
                                    .trim()
                                    .chars()
                                    .filter(|c| !c.is_control())
                                    .take(72)
                                    .collect();
                                if !clean.is_empty() {
                                    let vol = volume.clamp(0.1, 1.0);
                                    if let Some((px, py, pz, team)) = gs
                                        .players
                                        .get(&pid)
                                        .map(|p| (p.x, p.y, p.z, p.team))
                                    {
                                        let tick_u32 = gs.sim_tick as u32;
                                        gs.push_speech_bubble(SpeechBubbleSnap {
                                            speaker_id: pid,
                                            speaker_name: "YOU".to_string(),
                                            team,
                                            x: px,
                                            y: py,
                                            z: pz,
                                            snr_db: if vol >= 0.6 { 24.0 } else { 12.0 },
                                            symbol_id: if vol >= 0.6 { 1 } else { 0 },
                                            text: format!("YOU: {}", clean),
                                            tick: tick_u32,
                                        });
                                        gs.pending_chat = Some(PendingChat {
                                            text: clean,
                                            x: px,
                                            y: py,
                                            layer: pz,
                                            volume: vol,
                                            team,
                                        });
                                    }
                                }
                            }
                            ClientReliable::TogglePhysicsNative => {
                                gs.physics_native = !gs.physics_native;
                                let lbl = if gs.physics_native {
                                    "MODE: PHYSICS-NATIVE ACTUATORS!"
                                } else {
                                    "MODE: CLASSIC SPELLS"
                                };
                                let (mx, my) = (gs.grid_w as f32 * 0.5, gs.grid_h as f32 * 0.5);
                                gs.push_event(5, mx, my, 0, 2, lbl.to_string());
                            }
                            ClientReliable::CycleTimeScale => {
                                gs.time_scale = if (gs.time_scale - 1.0).abs() < 0.05 {
                                    4.0
                                } else if (gs.time_scale - 4.0).abs() < 0.05 {
                                    16.0
                                } else if (gs.time_scale - 16.0).abs() < 0.05 {
                                    0.25
                                } else {
                                    1.0
                                };
                                let lbl = if gs.time_scale < 0.5 {
                                    "TIME WARP: 0.25X SLOW-MO!".to_string()
                                } else {
                                    format!("TIME WARP: {:.0}X", gs.time_scale)
                                };
                                let (mx, my) = (gs.grid_w as f32 * 0.5, gs.grid_h as f32 * 0.5);
                                gs.push_event(5, mx, my, 0, 2, lbl);
                            }
                            ClientReliable::PhysicalActuator {
                                dx,
                                dy,
                                thermal_flux,
                                sp_water,
                                sp_oil,
                                sp_smoke,
                                focus_radius,
                                impulse,
                            } => {
                                gs.apply_player_physical_actuator(
                                    pid,
                                    dx,
                                    dy,
                                    thermal_flux,
                                    sp_water,
                                    sp_oil,
                                    sp_smoke,
                                    focus_radius,
                                    impulse,
                                );
                            }
                            ClientReliable::PlaceTimeDome { gx, gy, mode } => {
                                gs.place_time_dome(pid, gx, gy, mode);
                            }
                            ClientReliable::CycleBiome => {
                                gs.cycle_biome();
                            }
                            ClientReliable::CycleEvolutionSpeed => {
                                gs.cycle_evolution_speed();
                            }
                            ClientReliable::CycleWorldLaw => {
                                gs.cycle_world_law();
                            }
                        }
                    }
                }
            }

            dg_task.abort();
            let mut gs = st.lock().unwrap();
            gs.peers.remove(&pid);
            gs.players.remove(&pid);
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wave14b_five_archetypes_ws_json_and_evo_cycle() {
        let mut gs = GameState::new(
            "refsim".to_string(),
            "127.0.0.1:14440".to_string(),
            64,
            16,
            0,
            1,
            60,
            8,
        );
        // Verify 60 bots spawn evenly across all 5 archetypes (12 each)
        let mut cls_counts = [0usize; 5];
        for p in gs.players.values().filter(|p| p.is_bot) {
            cls_counts[(p.class_id as usize) % 5] += 1;
        }
        assert_eq!(cls_counts, [12, 12, 12, 12, 12]);
        assert_eq!(PlayerEntity::class_stats(CLASS_BURROWER, 1).0, 160);

        // Verify cycle_evolution_speed cycles 1x -> 4x -> 12x -> 1x
        assert!((gs.evo_speed - 1.0).abs() < 1e-3);
        gs.cycle_evolution_speed();
        assert!((gs.evo_speed - 4.0).abs() < 1e-3);
        gs.cycle_evolution_speed();
        assert!((gs.evo_speed - 12.0).abs() < 1e-3);
        gs.cycle_evolution_speed();
        assert!((gs.evo_speed - 1.0).abs() < 1e-3);

        // Emit a true 3D diagonal actuator beam (`z0 != z1`) and verify `0x02` live JSON
        let sample_pid = *gs.players.keys().next().unwrap();
        let pz0 = gs.players[&sample_pid].z;
        gs.apply_player_physical_actuator_3d(
            sample_pid,
            6.0,
            0.0,
            Some((pz0 + 3.5).clamp(1.0, 14.0)),
            0.9,
            0.0,
            0.0,
            0.0,
            3.5,
            0.2,
        );
        let json = gs.build_ws_entities_json();
        let parsed = JsonParser::parse(&json).expect("valid 0x02 WS JSON");
        let players = parsed.get("players").and_then(|v| v.as_arr()).expect("players array");
        assert_eq!(players.len(), 60);
        assert!(players[0].get("class_id").is_some());
        let beams = parsed.get("beams").and_then(|v| v.as_arr()).expect("beams array");
        assert!(!beams.is_empty(), "expected at least one 3D actuator beam in 0x02 JSON");
        let b0 = &beams[0];
        let y0 = b0.get("y0").and_then(|v| v.as_f32()).unwrap();
        let y1 = b0.get("y1").and_then(|v| v.as_f32()).unwrap();
        assert!((y1 - y0).abs() > 0.5, "expected diagonal 3D beam (y0={} != y1={})", y0, y1);
        let evo = parsed.get("evolution").expect("evolution object");
        assert_eq!(
            evo.get("epoch").and_then(|v| v.as_str()),
            Some("EPOCH I: PRISTINE")
        );
        assert!(evo.get("basalt").is_some());
        assert!(evo.get("trees").is_some());
        assert!(evo.get("glaciers").is_some());
        assert!(evo.get("erosion").is_some());
    }

    #[tokio::test]
    async fn test_wave17c_synergies_terrain_engineering_acoustics_and_conservation() {
        let mut gs = GameState::new(
            "refsim".to_string(),
            "127.0.0.1:14440".to_string(),
            64,
            16,
            0,
            1,
            0,
            8,
        );
        gs.sim_tick = 100;

        // 1. Strider Squad Shield-Wall (`SHIELD WALL!`) -> 35% incoming damage reduction
        let strider_id = gs.spawn_player("Frost-Strider-A1".into(), 0, CLASS_STRIDER, true);
        let ally_id = gs.spawn_player("Frost-Tidekin-A2".into(), 0, CLASS_TIDEKIN, true);
        let enemy_id = gs.spawn_player("Ember-Skywing-A3".into(), 1, CLASS_SKYWING, true);
        for (pid, sq, x, y, z) in [
            (strider_id, 1u8, 24.0f32, 32.0f32, 5.0f32),
            (ally_id, 1u8, 28.0f32, 32.0f32, 5.0f32),
            (enemy_id, 1u8, 36.0f32, 32.0f32, 5.0f32),
        ] {
            if let Some(p) = gs.players.get_mut(&pid) {
                p.squad_id = sq;
                p.x = x;
                p.y = y;
                p.z = z;
            }
        }
        gs.run_emergent_synergies();
        assert!(
            gs.players[&ally_id].shield_wall_until > gs.sim_tick,
            "ally within 9m of Strider should receive Shield-Wall aura"
        );
        let hp_before = gs.players[&ally_id].hp;
        gs.apply_damage(Some(enemy_id), ally_id, 40, "", 3);
        let dmg_taken = hp_before - gs.players[&ally_id].hp;
        assert_eq!(dmg_taken, 26, "Shield-Wall must cut 40 dmg by 35% -> 26");
        assert!(
            gs.tactical_events.iter().any(|e| e.label == "SHIELD WALL!"),
            "expected SHIELD WALL! tactical event"
        );

        // 2. Tidekin Ally Quench Rescue (`TIDEKIN QUENCH RESCUE!`): clears burn, +22 HP, +25 assist score
        if let Some(s) = gs.players.get_mut(&strider_id) {
            s.hp = s.max_hp - 40;
            s.status_burn_until = gs.sim_tick + 90;
        }
        let s_hp_pre = gs.players[&strider_id].hp;
        let tk_score_pre = gs.players[&ally_id].score;
        gs.run_emergent_synergies();
        assert_eq!(gs.players[&strider_id].status_burn_until, 0, "burn must be extinguished");
        assert_eq!(gs.players[&strider_id].hp, s_hp_pre + 22, "must restore +22 HP");
        assert_eq!(gs.players[&ally_id].score, tk_score_pre + 25, "must grant +25 assist score");
        assert!(
            gs.tactical_events.iter().any(|e| e.label == "TIDEKIN QUENCH RESCUE!"),
            "expected TIDEKIN QUENCH RESCUE! tactical event"
        );

        // 3. Solo Emberstalker Opportunistic Ambush (`SOLO ASSASSINATION!`): +35% damage from behind/smoke
        let stalker_id = gs.spawn_player("Frost-Stalker-Solo".into(), 0, CLASS_EMBERSTALKER, true);
        let lone_victim = gs.spawn_player("Ember-Tidekin-Iso".into(), 1, CLASS_TIDEKIN, true);
        if let Some(st) = gs.players.get_mut(&stalker_id) {
            st.squad_id = 0;
            st.x = 50.0;
            st.y = 50.0;
            st.z = 5.0;
            st.smoke_veil_active = true;
        }
        if let Some(v) = gs.players.get_mut(&lone_victim) {
            v.squad_id = 2;
            v.x = 53.0;
            v.y = 50.0;
            v.z = 5.0;
            v.aim_x = 1.0; // facing away from stalker (ambush from behind!)
            v.aim_y = 0.0;
        }
        let v_hp_pre = gs.players[&lone_victim].hp;
        gs.apply_damage(Some(stalker_id), lone_victim, 20, "", 3);
        assert_eq!(v_hp_pre - gs.players[&lone_victim].hp, 27, "+35% ambush bonus on 20 -> 27");
        assert!(
            gs.tactical_events.iter().any(|e| e.label == "SOLO ASSASSINATION!"),
            "expected SOLO ASSASSINATION! tactical event"
        );

        // 4. Burrower Subterranean Shrine Undermine (`SUBTERRANEAN BREACH!`)
        let burrower_id = gs.spawn_player("Frost-Terramole-A4".into(), 0, CLASS_BURROWER, true);
        let shrine_xy = (gs.shrines[1].x, gs.shrines[1].y);
        gs.shrines[1].owner = 1; // East-held shrine
        if let Some(br) = gs.players.get_mut(&burrower_id) {
            br.squad_id = 1;
            br.x = shrine_xy.0;
            br.y = shrine_xy.1;
            br.z = 3.0;
            br.in_tunnel = true;
        }
        if let Some(en) = gs.players.get_mut(&enemy_id) {
            en.x = shrine_xy.0 + 3.0;
            en.y = shrine_xy.1;
            en.z = 4.0;
        }
        gs.run_emergent_synergies();
        assert!(
            gs.tactical_events.iter().any(|e| e.label == "SUBTERRANEAN BREACH!"),
            "expected SUBTERRANEAN BREACH! tactical event"
        );

        // 5. Conservation-Exact Terrain Engineering: Cryo-Bridge Forging (`CRYO-BRIDGE FORGED!`) & Basalt Bulwark (`BASALT BULWARK!`)
        // Locate an actual open river water cell on the biome map
        let mut bridge_target = (32.0f32, 20.0f32, 2usize);
        'find_water: for z in 1..(gs.grid_d - 2) {
            let f = &gs.volume.layers[z];
            for y in 8..(gs.grid_h - 8) {
                for x in 8..(gs.grid_w - 8) {
                    let idx = f.idx(x, y);
                    if f.mass[SP_WATER][idx] >= 400 && f.mass[SP_ICE][idx] < 50 {
                        bridge_target = (x as f32, y as f32, z);
                        break 'find_water;
                    }
                }
            }
        }
        let (bridge_x, bridge_y, bridge_l) = bridge_target;
        gs.sim_tick += 50;
        if let Some(s) = gs.players.get_mut(&strider_id) {
            s.x = bridge_x;
            s.y = bridge_y;
            s.z = bridge_l as f32;
            s.mana = s.max_mana;
        }
        for _ in 0..3 {
            gs.apply_player_physical_actuator_3d(strider_id, 0.0, 0.0, Some(bridge_l as f32), -1.0, 0.0, 0.0, 0.0, 4.5, 0.0);
            if let Some(s) = gs.players.get_mut(&strider_id) {
                s.mana = s.max_mana;
            }
            let raws = gs.volume.refsim_raw_outputs_all();
            let step_stats = gs.volume.step_3d_with_neural_z(&raws, None);
            assert_eq!(step_stats.mass_error, 0, "mass conservation must remain exact");
            assert_eq!(step_stats.energy_error, 0, "energy conservation must remain exact");
            gs.check_terrain_engineering_milestones();
        }
        assert!(
            gs.tactical_events.iter().any(|e| e.label == "CRYO-BRIDGE FORGED!"),
            "expected CRYO-BRIDGE FORGED! tactical event"
        );

        // Basalt Bulwark: create hot oil + water quench conservatively near shrine, run evolution step
        let qx = shrine_xy.0 - 4.0;
        let qy = shrine_xy.1 - 4.0;
        let ql = 4usize;
        if let Some(tk) = gs.players.get_mut(&ally_id) {
            tk.x = qx;
            tk.y = qy;
            tk.z = ql as f32;
            tk.mana = tk.max_mana;
        }
        // Inject oil + intense heat conservatively, then quench with water
        for _ in 0..4 {
            gs.apply_player_physical_actuator_3d(ally_id, 0.0, 0.0, Some(ql as f32), 1.0, 0.0, 1.0, 0.0, 4.0, 0.0);
            if let Some(tk) = gs.players.get_mut(&ally_id) {
                tk.mana = tk.max_mana;
            }
        }
        gs.apply_player_physical_actuator_3d(ally_id, 0.0, 0.0, Some(ql as f32), -0.4, 1.0, 0.0, 0.0, 4.0, 0.0);
        gs.volume.step_world_evolution_3d(12.0);
        let raws2 = gs.volume.refsim_raw_outputs_all();
        let step_stats2 = gs.volume.step_3d_with_neural_z(&raws2, None);
        assert_eq!(step_stats2.mass_error, 0, "mass conservation must remain exact after basalt quench");
        assert_eq!(step_stats2.energy_error, 0, "energy conservation must remain exact after basalt quench");
        gs.check_terrain_engineering_milestones();
        assert!(
            gs.tactical_events.iter().any(|e| e.label == "BASALT BULWARK!"),
            "expected BASALT BULWARK! tactical event"
        );

        // 6. Verify op=6 acoustic JSON request includes recent_events, epoch, biome, and 5-archetype telemetry
        let ac_json = gs.build_acoustic_request_json(None);
        let ac_parsed = JsonParser::parse(&ac_json).expect("valid op=6 acoustic JSON");
        assert!(ac_parsed.get("epoch").and_then(|v| v.as_str()).is_some());
        assert!(ac_parsed.get("biome").and_then(|v| v.as_str()).is_some());
        let ac_events = ac_parsed.get("recent_events").and_then(|v| v.as_arr()).expect("recent_events");
        assert!(!ac_events.is_empty());
        let ac_bots = ac_parsed.get("bots").and_then(|v| v.as_arr()).expect("bots");
        assert!(ac_bots.iter().any(|b| b.get("bot_class").and_then(|v| v.as_str()) == Some("BURROWER")));
        assert!(ac_bots.iter().any(|b| b.get("in_tunnel").and_then(|v| v.as_bool()) == Some(true)));

        // 7. Verify 0x02 WebSocket JSON contains additive `"events"` array (capped to last 5)
        let ws_json = gs.build_ws_entities_json();
        let ws_parsed = JsonParser::parse(&ws_json).expect("valid 0x02 WS JSON");
        let ws_events = ws_parsed.get("events").and_then(|v| v.as_arr()).expect("0x02 events array");
        assert!(!ws_events.is_empty() && ws_events.len() <= 5);
    }

    #[tokio::test]
    async fn test_wave20c_world_laws_and_body_morphology() {
        let mut gs = GameState::new(
            "refsim".to_string(),
            "127.0.0.1:14449".to_string(),
            64,
            16,
            0,
            1,
            10,
            4,
        );
        // 1. Live World-Law cycling 0 -> 1 -> 2 -> 3 -> 0 + additive 0x02 JSON `"world_law"` block
        assert_eq!(gs.volume.world_law_preset, 0);
        gs.cycle_world_law();
        assert_eq!(gs.volume.world_law_preset, 1);
        gs.cycle_world_law();
        assert_eq!(gs.volume.world_law_preset, 2);
        gs.cycle_world_law();
        assert_eq!(gs.volume.world_law_preset, 3);
        let ws_json = gs.build_ws_entities_json();
        let ws_parsed = JsonParser::parse(&ws_json).expect("valid 0x02 WS JSON");
        let wl = ws_parsed.get("world_law").expect("world_law JSON field");
        assert_eq!(wl.get("id").and_then(|v| v.as_i64()), Some(3));
        assert_eq!(wl.get("name").and_then(|v| v.as_str()), Some("LOW-G MAGMA"));
        assert!((wl.get("gravity_g").and_then(|v| v.as_f32()).unwrap_or(0.0) - 0.35).abs() < 1e-2);
        assert!((wl.get("ambient_shift").and_then(|v| v.as_f32()).unwrap_or(0.0) - 5500.0).abs() < 1e-1);
        assert!((wl.get("precipitation").and_then(|v| v.as_f32()).unwrap_or(0.0) - 0.25).abs() < 1e-2);
        gs.cycle_world_law();
        assert_eq!(gs.volume.world_law_preset, 0);

        // 2. BodyMorphology physical differentiation & Low-G leap coupling
        let strider = BodyMorphology::from_archetype(CLASS_STRIDER);
        let skywing = BodyMorphology::from_archetype(CLASS_SKYWING);
        let tidekin = BodyMorphology::from_archetype(CLASS_TIDEKIN);
        let stalker = BodyMorphology::from_archetype(CLASS_EMBERSTALKER);
        let burrower = BodyMorphology::from_archetype(CLASS_BURROWER);
        assert!(strider.mass > skywing.mass * 2.0);
        assert!(skywing.wing_lift > 0.8);
        assert!(tidekin.buoyancy >= 0.99 && tidekin.thermal_insul > 0.70);
        assert!(stalker.climb_grip > 0.9);
        assert!(burrower.bore_torque > 0.9);

        // Compare dash launch vertical velocity under STANDARD (g=1.0) vs LOW-G MAGMA (g=0.35)
        let pid = *gs.players.keys().next().unwrap();
        let inp = ClientInputDatagram {
            seq: 1,
            move_x: 1.0,
            move_y: 0.0,
            aim_x: 1.0,
            aim_y: 0.0,
            dash: true,
            melee: false,
            cast_bolt: false,
            bolt_element: 0,
            carve_ice: false,
            cast_ability_e: false,
            cast_ultimate_r: false,
            client_time_ms: 100,
        };
        gs.volume.set_world_law_preset(field_core::WORLD_LAW_STANDARD);
        if let Some(p) = gs.players.get_mut(&pid) {
            p.dash_cd = 0;
            p.stamina = p.max_stamina;
        }
        gs.apply_input(pid, &inp);
        let vz_std = gs.players[&pid].vz;

        gs.volume.set_world_law_preset(field_core::WORLD_LAW_LOW_G_MAGMA);
        if let Some(p) = gs.players.get_mut(&pid) {
            p.dash_cd = 0;
            p.stamina = p.max_stamina;
        }
        gs.apply_input(pid, &inp);
        let vz_low_g = gs.players[&pid].vz;
        assert!(
            vz_low_g > vz_std * 1.25,
            "LOW-G MAGMA should boost leap velocity ({} vs {})",
            vz_low_g,
            vz_std
        );
    }
}
