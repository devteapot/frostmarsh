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

use bytes::Bytes;
use clap::Parser;
use field_core::{
    EntityVoxelPhysics, Field, INPUT_CHANNELS, OUTPUT_CHANNELS, RawStepOutput, SP_ICE, SP_OIL,
    SP_PLANT, SP_ROCK, SP_STEAM, SP_WATER, SP_WOOD, VolumeField,
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
    /// Number of physically coupled 3D vertical strata (1, 4, 8, or 16; default 4 => 262,144 voxels, 16 => isotropic 3D!)
    #[arg(long, default_value_t = 4)]
    layers: usize,
    /// Number of role-aware AI bots (6v6 = 12)
    #[arg(long, default_value_t = 12)]
    bots: usize,
    /// Per-player Area-of-Interest chunk radius (0 = full map)
    #[arg(long, default_value_t = 10)]
    aoi_radius: usize,
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
}

impl CudaRpcClient {
    fn new(addr: String) -> Self {
        let mut c = Self {
            addr,
            stream: None,
            seq: 0,
        };
        let _ = c.ensure_connected();
        c
    }

    fn ensure_connected(&mut self) -> Result<&mut RpcStream, Box<dyn std::error::Error>> {
        if self.stream.is_none() {
            #[cfg(unix)]
            if self.addr.starts_with('/') {
                let s = UnixStream::connect(&self.addr)?;
                s.set_read_timeout(Some(Duration::from_millis(3000)))?;
                s.set_write_timeout(Some(Duration::from_millis(3000)))?;
                log::info!("Connected to PyTorch CUDA FP16 3D daemon via Unix socket {}", self.addr);
                self.stream = Some(RpcStream::Unix(s));
                return Ok(self.stream.as_mut().unwrap());
            }
            let s = TcpStream::connect_timeout(&self.addr.parse()?, Duration::from_millis(800))?;
            s.set_nodelay(true)?;
            s.set_read_timeout(Some(Duration::from_millis(3000)))?;
            s.set_write_timeout(Some(Duration::from_millis(3000)))?;
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

#[derive(Clone)]
struct PlayerEntity {
    id: u32,
    name: String,
    team: u8,
    class_id: u8,
    is_bot: bool,
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
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
}

impl PlayerEntity {
    fn class_stats(class_id: u8, level: u8) -> (i16, f32, f32, f32) {
        let lvl_bonus = (level.saturating_sub(1)) as i16;
        match class_id {
            CLASS_WARDEN => (145 + lvl_bonus * 18, 100.0, 110.0, 15.5),
            CLASS_ALCHEMIST => (105 + lvl_bonus * 12, 140.0, 95.0, 16.0),
            _ => (118 + lvl_bonus * 14, 115.0, 130.0, 18.8),
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
    world_day: u32,
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
}

impl GameState {
    #[inline]
    fn is_z16(&self) -> bool {
        self.grid_d >= 16
    }

    fn surf_layer(&self) -> usize {
        if self.grid_d >= 16 {
            4
        } else if self.grid_d >= 2 {
            1
        } else {
            0
        }
    }

    fn find_z16_surface_z(&self, x: f32, y: f32, start_z: f32) -> f32 {
        let mut z = start_z.clamp(1.0, (self.grid_d as f32) - 1.5);
        while z > 0.6 {
            if self.volume.sample_solid_density_3d(x, y, z) >= 0.5 {
                return (z + 0.25).min((self.grid_d as f32) - 2.0);
            }
            z -= 0.2;
        }
        1.5
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
        bot_count: usize,
        aoi_radius: usize,
    ) -> Self {
        let grid_w = (size / CHUNK_SIZE).max(4) * CHUNK_SIZE;
        let grid_h = grid_w;
        let grid_d = layers.clamp(1, 16);
        let chunks_x = chunks_for_dim(grid_w);
        let chunks_y = chunks_for_dim(grid_h);
        let volume = VolumeField::new_shattered_reach_3d(grid_w, grid_h, grid_d);
        let eco = EcosystemState::new(grid_w, grid_h);
        let shrines = Self::build_shrines(grid_w, grid_h);
        let use_cuda = backend_name == "cuda-rpc";

        let mut gs = Self {
            grid_w,
            grid_h,
            grid_d,
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
            world_day: 1,
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
        };
        gs.refresh_hazard_grids();
        for i in 0..bot_count {
            let team = (i % 2) as u8;
            let class_id = ((i / 2) % 3) as u8;
            let role_tag = match class_id {
                CLASS_WARDEN => "Warden",
                CLASS_ALCHEMIST => "Pyro",
                _ => "Ranger",
            };
            let prefix = if team == 0 { "Frost" } else { "Ember" };
            let name = format!("{}-{}-{}", prefix, role_tag, i / 2 + 1);
            gs.spawn_player(name, team, class_id, true);
        }
        gs
    }

    fn push_event(&mut self, kind: u8, x: f32, y: f32, value: i16, team: u8, label: String) {
        let id = self.next_event_id;
        self.next_event_id = self.next_event_id.wrapping_add(1);
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

        // Build UTF-8 JSON request payload
        let mut req = String::with_capacity(1024);
        req.push_str("{\"bots\":[");
        for (i, (id, name, team, cls, x, y, z, hp_frac, relic)) in bots.iter().enumerate() {
            if i > 0 {
                req.push(',');
            }
            req.push_str(&format!(
                "{{\"id\":{},\"name\":{},\"team\":{},\"class_id\":{},\"x\":{:.2},\"y\":{:.2},\"z\":{:.2},\"hp_frac\":{:.2},\"carrying_relic\":{}}}",
                id,
                json_escape_str(name),
                team,
                cls,
                x,
                y,
                z,
                hp_frac,
                if *relic { "true" } else { "false" }
            ));
        }
        req.push_str(&format!(
            "],\"listener\":[{:.2},{:.2},{:.2}],\"grid_w\":{},\"grid_h\":{},\"grid_d\":{},\"chat\":",
            listener.0, listener.1, listener.2, self.grid_w, self.grid_h, self.grid_d
        ));
        if let Some(ref c) = pending {
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
        let tick_now = self.sim_tick;
        let mut replies = 0usize;
        for &(bid, ref bname, bteam, _, bx, by, bz, _, _) in bots {
            if bteam != chat.team {
                continue;
            }
            let dist = (bx - chat.x).hypot(by - chat.y).hypot((bz - chat.layer) * 6.0);
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
                    let ldist = (bx - listener.0).hypot(by - listener.1).hypot((bz - listener.2) * 6.0);
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
            let off = ((salt % 7) as f32 - 3.0) * (hf * 0.014);
            let (sx, sy, scan_z) = if salt % 3 == 1 {
                if team == 0 {
                    // Near the East mouth of the vaulted basalt cave (z ≈ 1.6..2.0)
                    (wf * 0.375, hf * 0.700, 4.5)
                } else {
                    // East bridge approach overlooking the 3-span stone & timber bridge (z ≈ 8.1)
                    (wf * 0.715, hf * 0.500, 12.5)
                }
            } else if team == 0 {
                (wf * 0.22, hf * 0.50 + off, 12.5)
            } else {
                (wf * 0.80, hf * 0.50 + off, 12.5)
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
        let off = ((salt % 11) as f32 - 5.0) * (hf * 0.025);
        if team == 0 {
            (wf * 0.12, hf * 0.50 + off, surf_z)
        } else {
            (wf * 0.88, hf * 0.50 + off, surf_z)
        }
    }

    fn spawn_player(&mut self, name: String, team: u8, class_id: u8, is_bot: bool) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        let (x, y, z) = self.spawn_point(team % 2, id);
        let (mhp, mmp, mst, _) = PlayerEntity::class_stats(class_id % 3, 1);
        self.players.insert(
            id,
            PlayerEntity {
                id,
                name,
                team: team % 2,
                class_id: class_id % 3,
                is_bot,
                x,
                y,
                z,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
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

        if let Some(v) = self.players.get_mut(&victim_id) {
            if tick < v.dash_until {
                return;
            }
            if tick < v.shield_until {
                dmg = (dmg / 2).max(1);
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

        if dmg > 0 {
            let lbl = if tag.is_empty() { format!("-{}", dmg) } else { format!("-{} {}", dmg, tag) };
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
            let dash_cd_ticks = if p.class_id == CLASS_RANGER { 24 } else { 32 };
            if inp.dash && tick >= p.dash_cd && p.stamina >= 25.0 {
                p.stamina -= 25.0;
                p.dash_until = tick + 6;
                p.dash_cd = tick + dash_cd_ticks;
                let (ddx, ddy) = if mlen > 0.01 { (dx, dy) } else { (p.aim_x, p.aim_y) };
                p.vx = ddx * 42.0;
                p.vy = ddy * 42.0;
                p.vz = if self.grid_d >= 16 { 6.2 } else { 1.8 }; // vertical leap when dashing!
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
                let dmg = if p.class_id == CLASS_WARDEN { 34 } else { 24 };
                melee_hit_check = Some((p.team, p.x, p.y, p.z, p.aim_x, p.aim_y, dmg + (p.level as i16) * 3));
            }

            let bolt_cd_ticks = match p.class_id {
                CLASS_ALCHEMIST => 8,
                CLASS_RANGER => 9,
                _ => 11,
            };
            if inp.cast_bolt && tick >= p.bolt_cd && p.mana >= 8.0 {
                p.mana -= 8.0;
                p.bolt_cd = tick + bolt_cd_ticks;
                let pwr = 1.0 + (p.level as f32 - 1.0) * 0.12;
                let vz = if p.z >= 1.6 { -0.65 } else { 0.0 }; // High-ground bolts arc gently downward!
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
                    CLASS_WARDEN => {
                        p.shield_until = tick + 90;
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                    CLASS_ALCHEMIST => {
                        for ang_off in [-0.24f32, 0.0, 0.24] {
                            let (ca, sa) = (ang_off.cos(), ang_off.sin());
                            let vx = (p.aim_x * ca - p.aim_y * sa) * 48.0;
                            let vy = (p.aim_x * sa + p.aim_y * ca) * 48.0;
                            spawn_projs.push((p.team, 2, p.x + p.aim_x * 1.5, p.y + p.aim_y * 1.5, p.z, vx, vy, -0.4, 1.3, 1.2, 26));
                        }
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                    _ => {
                        p.haste_until = tick + 90;
                        carve_pos = Some((
                            (p.x + p.aim_x * 3.2).clamp(3.0, gw - 4.0) as i32,
                            (p.y + p.aim_y * 3.2).clamp(3.0, gh - 4.0) as i32,
                            p.team,
                        ));
                        ability_e_fx = Some((cls, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
                    }
                }
            }

            if inp.cast_ultimate_r && tick >= p.ult_r_cd && p.mana >= 55.0 {
                p.mana -= 55.0;
                p.ult_r_cd = tick + 360;
                p.ult_until = tick + 45;
                if p.class_id == CLASS_WARDEN {
                    p.shield_until = tick + 120;
                } else if p.class_id == CLASS_RANGER {
                    p.haste_until = tick + 150;
                }
                ult_r_fx = Some((p.class_id, p.team, p.x, p.y, p.z, p.aim_x, p.aim_y));
            }
        }

        if let Some((team, px, py, pz, ax, ay, dmg)) = melee_hit_check {
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
            for tid in targets {
                self.apply_damage(Some(pid), tid, dmg, "", 3);
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
                CLASS_WARDEN => {
                    for step in 1..=3 {
                        let tx = (px + ax * (step as f32 * 4.5)).clamp(3.0, gw - 4.0) as i32;
                        let ty = (py + ay * (step as f32 * 4.5)).clamp(3.0, gh - 4.0) as i32;
                        self.volume.apply_frost_3d(tx, ty, pl, 5, 2.2);
                    }
                    self.push_event(2, px, py, 0, team, "ICE BRIDGE!".into());
                }
                CLASS_ALCHEMIST => self.push_event(2, px, py, 0, team, "OIL BARRAGE!".into()),
                _ => self.push_event(2, px, py, 0, team, "WIND SPRINT!".into()),
            }
        }

        if let Some((cls, team, px, py, pz, ax, ay)) = ult_r_fx {
            let pl = (pz.round() as usize).clamp(0, self.grid_d - 1);
            match cls {
                CLASS_WARDEN => {
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
                CLASS_ALCHEMIST => {
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
                    self.push_event(2, tx, ty, 0, team, "METEOR STORM!".into());
                }
                _ => {
                    let cx = (px + ax * 10.0).clamp(5.0, gw - 6.0) as i32;
                    let cy = (py + ay * 10.0).clamp(5.0, gh - 6.0) as i32;
                    self.volume.apply_steam_burst_3d(cx, cy, pl, 8);
                    for p in self.players.values_mut() {
                        if p.team == team && (p.x - px).hypot(p.y - py) <= 24.0 {
                            p.haste_until = tick + 150;
                            p.hp = (p.hp + 20).min(p.max_hp);
                        }
                    }
                    self.push_event(2, px, py, 0, team, "SPIRIT SURGE!".into());
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
        let (px, py, pz, team, scale) = {
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
            (p.x, p.y, p.z, p.team, sc)
        };
        if scale < 0.05 {
            return;
        }

        let gw = self.grid_w as f32;
        let gh = self.grid_h as f32;
        let tx = (px + dx).clamp(3.0, gw - 4.0);
        let ty = (py + dy).clamp(3.0, gh - 4.0);
        let layer = (pz.round() as usize).clamp(0, self.grid_d - 1);

        let tf_s = tf * scale;
        let sw_s = sw * scale;
        let so_s = so * scale;
        let ss_s = ss * scale;
        let imp_s = imp * scale;

        self.volume
            .apply_physical_actuator(tx, ty, layer, tf_s, sw_s, so_s, ss_s, r);

        let tick = self.sim_tick;
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
                    z1: layer as f32,
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
            let mut shock_hits: Vec<(u32, i16)> = Vec::new();
            for t in self.players.values_mut() {
                if t.id == pid || tick < t.dash_until || (t.z - pz).abs() > 1.1 {
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
                    let push = 38.0 * imp_s * fall;
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
            for (tid, dmg) in shock_hits {
                self.apply_damage(Some(pid), tid, dmg, "SHOCK", 3);
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
        }
    }

    fn reset_match(&mut self) {
        let (w, h, d) = (self.grid_w, self.grid_h, self.grid_d);
        self.volume = VolumeField::new_shattered_reach_3d(w, h, d);
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
            }
        }
        for peer in self.peers.values_mut() {
            peer.needs_full_sync = true;
        }
        self.push_event(5, (w as f32) * 0.5, (h as f32) * 0.5, 0, 2, "NEW MATCH STARTED!".into());
        log::info!("Match cleanly reset ({}x{}x{}, {} players)", w, h, d, self.players.len());
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
            for (ti, &(d, o)) in others.iter().take(8).enumerate() {
                let tb = base + SELF_DIM + ti * TOKEN_DIM;
                feats[tb] = ((o.x - p.x) / 80.0).clamp(-1.0, 1.0);
                feats[tb + 1] = ((o.y - p.y) / 80.0).clamp(-1.0, 1.0);
                feats[tb + 2] = ((o.z - p.z) / 2.0).clamp(-1.0, 1.0);
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

        if self.physics_native {
            let act_res = if let Some(ref mut rpc) = self.bot_rpc {
                rpc.call_batched(7, 1, 1, n as u16, BOT_FEAT_DIM as u16, &feats)
            } else {
                return false;
            };
            if let Ok((out_buf, gpu_ms)) = act_res {
                if out_buf.len() == n * ACTUATOR_INSPECT_DIM {
                    self.bot_infer_ms = gpu_ms;
                    let to_u8 = |v: f32| -> u8 { (v.clamp(0.0, 1.0) * 255.0).round() as u8 };
                    let mut brains = Vec::with_capacity(n);

                    for (bi, &bid) in bot_ids.iter().enumerate() {
                        let row = &out_buf[bi * ACTUATOR_INSPECT_DIM..(bi + 1) * ACTUATOR_INSPECT_DIM];
                        let mut move_x = row[0];
                        let mut move_y = row[1];
                        let mut aim_dx = row[2];
                        let mut aim_dy = row[3];
                        let mut thermal_flux = row[4].clamp(-1.0, 1.0);
                        let mut sp_water = row[5].clamp(0.0, 1.0);
                        let mut sp_oil = row[6].clamp(0.0, 1.0);
                        let mut sp_smoke = row[7].clamp(0.0, 1.0);
                        let focus_radius = if row[8] <= 1.0 {
                            1.5 + row[8].clamp(0.0, 1.0) * 7.0
                        } else {
                            row[8].clamp(1.5, 8.5)
                        };
                        let mut impulse = row[9].clamp(0.0, 1.0);

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

                        let move_inp = ClientInputDatagram {
                            seq: tick as u32,
                            move_x,
                            move_y,
                            aim_x: aim_dx,
                            aim_y: aim_dy,
                            dash: impulse > 0.78,
                            melee: false,
                            cast_bolt: false,
                            bolt_element: 0,
                            carve_ice: false,
                            cast_ability_e: false,
                            cast_ultimate_r: false,
                            client_time_ms: 0,
                        };
                        self.apply_input(bid, &move_inp);
                        self.apply_player_physical_actuator(
                            bid,
                            tdx,
                            tdy,
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
                                z_heatmap_4x4[k] = to_u8(row[ACTUATOR_OUT_DIM + k]);
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
        let bot_ids: Vec<u32> = self.players.values().filter(|p| p.is_bot).map(|p| p.id).collect();
        let snap: Vec<(u32, u8, u8, f32, f32, f32, bool)> = self
            .players
            .values()
            .map(|p| (p.id, p.team, p.class_id, p.x, p.y, p.z, p.carrying_relic))
            .collect();
        let shrine_goals: Vec<(f32, f32, i8)> = self.shrines.iter().map(|s| (s.x, s.y, s.owner)).collect();
        let tick = self.sim_tick;

        for bid in bot_ids {
            let Some(&(_, team, class_id, mx, my, mz, carrying)) = snap.iter().find(|x| x.0 == bid) else {
                continue;
            };
            let desired_owner = if team == 0 { -1 } else { 1 };
            let (tx, ty) = if carrying {
                if team == 0 { (gw * 0.12, gh * 0.50) } else { (gw * 0.88, gh * 0.50) }
            } else if self.is_z16() && bid % 3 == 1 {
                if team == 0 {
                    // Vaulted basalt cave & under-bridge patrol
                    let phase = ((tick as f32 * 0.008 + bid as f32).sin() + 1.0) * 0.5;
                    (gw * (0.235 + 0.145 * phase), gh * (0.720 - 0.025 * phase))
                } else {
                    // 3-span arched bridge patrol across the canyon
                    let patrol_x = gw * (0.50 + 0.18 * ((tick as f32 * 0.01 + bid as f32).sin()));
                    (patrol_x, gh * 0.50)
                }
            } else if !self.is_z16() && mz >= 1.6 && bid % 3 == 1 {
                // Highland Skybridge squad — actively duels across the Layer 2 Stone Sky-Arch!
                let patrol_x = gw * (0.50 + 0.14 * ((tick as f32 * 0.01 + bid as f32).sin()));
                (patrol_x, gh * 0.29)
            } else if bid % 3 == 0 {
                let on_home_side = (team == 0) == (mx < gw * 0.50);
                if (mx - gw * 0.50).abs() > gw * 0.14 && on_home_side {
                    let bridge_y = if self.is_z16() {
                        gh * 0.50
                    } else if bid % 2 == 0 {
                        gh * 0.31
                    } else {
                        gh * 0.71
                    };
                    (gw * 0.50, bridge_y)
                } else {
                    (self.relic_x + (bid as f32 * 1.9).sin() * 5.0, self.relic_y + (bid as f32 * 2.3).cos() * 5.0)
                }
            } else {
                let target_shrine = shrine_goals
                    .iter()
                    .filter(|s| s.2 != desired_owner)
                    .min_by(|a, b| (a.0 - mx).hypot(a.1 - my).partial_cmp(&(b.0 - mx).hypot(b.1 - my)).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|s| (s.0, s.1))
                    .unwrap_or((self.relic_x, self.relic_y));
                let jitter = (bid as f32 * 1.3 + tick as f32 * 0.008).sin() * 6.0;
                (target_shrine.0 + jitter, target_shrine.1 - jitter * 0.6)
            };

            let enemy = snap
                .iter()
                .filter(|x| x.1 != team)
                .min_by(|a, b| (a.3 - mx).hypot(a.4 - my).partial_cmp(&(b.3 - mx).hypot(b.4 - my)).unwrap_or(std::cmp::Ordering::Equal));
            let (ax, ay, edist) = if let Some(e) = enemy {
                (e.3 - mx, e.4 - my, (e.3 - mx).hypot(e.4 - my))
            } else {
                (tx - mx, ty - my, 999.0)
            };

            let near_river = (mx - gw * 0.50).abs() < gw * 0.14;
            let elem = match class_id {
                CLASS_WARDEN => if near_river || tick % 3 != 0 { 1 } else { 0 },
                CLASS_ALCHEMIST => if tick % 7 == 0 { 2 } else { 0 },
                _ => ((tick / 15 + bid as u64) % 3) as u8,
            };

            if self.physics_native {
                let mut move_inp = ClientInputDatagram {
                    seq: tick as u32,
                    move_x: tx - mx,
                    move_y: ty - my,
                    aim_x: ax,
                    aim_y: ay,
                    dash: (tick + bid as u64 * 17) % 75 == 0,
                    melee: false,
                    cast_bolt: false,
                    bolt_element: 0,
                    carve_ice: false,
                    cast_ability_e: false,
                    cast_ultimate_r: false,
                    client_time_ms: 0,
                };
                let (mut tf, mut sw, mut so, mut ss, mut rad, mut imp) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 3.8f32, 0.0f32);
                if (tick + bid as u64 * 9) % 12 == 0 && (edist < 34.0 || near_river) {
                    match class_id {
                        CLASS_WARDEN => {
                            tf = -0.90;
                            if edist < 8.0 { imp = 0.75; }
                            rad = 4.5;
                        }
                        CLASS_ALCHEMIST => {
                            if (tick / 12 + bid as u64) % 3 == 0 {
                                so = 0.85;
                                rad = 4.2;
                            } else {
                                tf = 0.95;
                                rad = 3.8;
                            }
                        }
                        _ => {
                            if (tick / 12 + bid as u64) % 3 == 0 {
                                ss = 0.75;
                                sw = 0.35;
                            } else {
                                tf = if near_river { -0.80 } else { 0.85 };
                            }
                            if edist < 9.0 { imp = 0.65; }
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
                let dist_off = edist.min(14.0).max(4.0);
                self.apply_player_physical_actuator(
                    bid,
                    (move_inp.aim_x / dlen) * dist_off,
                    (move_inp.aim_y / dlen) * dist_off,
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
                dash: (tick + bid as u64 * 17) % 75 == 0,
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

        // Move players in 3D `(x, y, z)` with Skybridge/Plateau altitude & cliff-drop gravity!
        let pids: Vec<u32> = self.players.keys().copied().collect();
        let mut phys_dmg_events: Vec<(Option<u32>, u32, i16, &'static str, u8)> = Vec::new();
        let mut combo_callouts: Vec<(f32, f32, u8, String)> = Vec::new();
        let field_tick = self.volume.tick;
        let physics_native = self.physics_native;
        for pid in pids {
            let (px, py, pz, vx, vy, mut vz, mut kb_vx, mut kb_vy, dash_until, chilled_until, team, last_blast_ft) = {
                let p = &self.players[&pid];
                (
                    p.x, p.y, p.z, p.vx, p.vy, p.vz, p.kb_vx, p.kb_vy, p.dash_until,
                    p.chilled_until, p.team, p.last_phys_blast_field_tick,
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

            let mut mult = 1.0f32;
            if physics_native {
                // Locomotion emerges directly from voxel ice_frac (glide), oil_frac (slip), water_frac (drag)
                if on_raft.is_none() && phys.water_frac > 0.05 && phys.ice_frac < 0.30 {
                    mult *= (1.0 - 0.42 * phys.water_frac).clamp(0.45, 1.0);
                }
                if phys.ice_frac > 0.10 {
                    mult *= 1.0 + 0.24 * phys.ice_frac;
                }
                if phys.oil_frac > 0.10 {
                    mult *= 1.0 - 0.12 * phys.oil_frac;
                }
                if phys.temp_c < -15.0 && tick >= dash_until {
                    let cold_slow = ((-15.0 - phys.temp_c) / 80.0).clamp(0.0, 0.32);
                    mult *= 1.0 - cold_slow;
                }
                // Low-friction glide on ice / oil preserves momentum via kb_vx/kb_vy slip term
                let slip = (phys.ice_frac * 0.72).max(phys.oil_frac * 0.55).clamp(0.0, 0.78);
                if slip > 0.12 {
                    kb_vx = kb_vx * (0.78 + 0.16 * slip) + vx * mult * (0.14 * slip);
                    kb_vy = kb_vy * (0.78 + 0.16 * slip) + vy * mult * (0.14 * slip);
                }
            } else {
                if on_raft.is_none() && (hz_here & (1 << 4)) != 0 { mult *= 0.60; }
                if ((hz_here & (1 << 3)) != 0 || tick < chilled_until) && tick >= dash_until { mult *= 0.68; }
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
                // True isotropic 3D continuous ∇solid_density capsule locomotion (caves, slopes, 3-span arched bridge!)
                let ((nx, ny, nz), (_nvx, _nvy, nvz), _grounded) =
                    self.volume.step_capsule_3d((px, py, pz), (vx_s, vy_s, vz), 0.55, 1.80, dt);
                fx = nx;
                fy = ny;
                fz = nz.clamp(0.0, (self.grid_d as f32) - 0.5);
                vz = nvz;
            } else {
                // Continuous bilinear solid-density gradient capsule collider (R = 0.65, probes at 0.55)
                // Exact 1:1 parity with client `predict_hero_step`: weaves between 3×3 pine trunks and glides smoothly along walls!
                let coll_layer = if pz >= 1.55 { 2.min(self.grid_d - 1) } else { surf_l };
                let dashing_now = tick < dash_until;
                let capsule_at = |tx: f32, ty: f32| -> f32 {
                    let allow_water = dashing_now
                        || on_raft.is_some()
                        || (pz < 1.35
                            && self.rafts.iter().any(|b| {
                                (tx - b.x).powi(2) + (ty - b.y).powi(2) <= (b.radius + 1.0).powi(2)
                            }));
                    self.sample_capsule_density(tx, ty, coll_layer, allow_water)
                };

                let mut mx = (px + vx_s * dt).clamp(3.5, gw - 3.5);
                let mut my = (py + vy_s * dt).clamp(3.5, gh - 3.5);
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

                // 3D Vertical Altitude Update:
                // - In physics_native: structural support comes directly from `sample_entity_physics(..., 2).support_solid`!
                //   If a timber skybridge burns or ice bridge melts beneath an L2 entity, gravity drops them cleanly to L1!
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
                if l2_deck && (pz >= 1.45 || on_ramp) {
                    mz = (pz + 3.2 * dt).min(2.0);
                    vz = 0.0;
                } else {
                    // Apply gravity toward surface layer
                    vz -= 8.5 * dt;
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
            if physics_native && tick >= dash_until {
                let owner = self.find_recent_actuator_owner(fx, fy, team);
                // 1. Emergent Blast Damage & Knockback from combustion (delta_temp_spike) & flash-boil (delta_steam_burst)
                if field_tick != last_blast_ft {
                    if phys.delta_temp_spike > 18.0 {
                        let dmg = ((phys.delta_temp_spike - 15.0) * 0.42).round().clamp(6.0, 38.0) as i16;
                        let (bx, by) = self.voxel_blast_dir(fx, fy, cur_layer, false);
                        let push = (phys.delta_temp_spike.min(120.0) * 0.34).clamp(8.0, 42.0);
                        kb_vx = (kb_vx + bx * push).clamp(-48.0, 48.0);
                        kb_vy = (kb_vy + by * push).clamp(-48.0, 48.0);
                        phys_dmg_events.push((owner, pid, dmg, "DETONATION!", 0));
                        if dmg >= 14 {
                            combo_callouts.push((fx, fy, 1 - team, "COMBUSTION BLAST!".into()));
                        }
                        new_last_blast_ft = field_tick;
                    } else if phys.delta_steam_burst > 0.06 {
                        let dmg = (phys.delta_steam_burst * 95.0).round().clamp(5.0, 32.0) as i16;
                        let (bx, by) = self.voxel_blast_dir(fx, fy, cur_layer, true);
                        let push = (phys.delta_steam_burst * 72.0).clamp(8.0, 38.0);
                        kb_vx = (kb_vx + bx * push).clamp(-48.0, 48.0);
                        kb_vy = (kb_vy + by * push).clamp(-48.0, 48.0);
                        phys_dmg_events.push((owner, pid, dmg, "FLASH-BOIL!", 0));
                        if dmg >= 12 {
                            combo_callouts.push((fx, fy, 1 - team, "FLASH-BOIL BURST!".into()));
                        }
                        new_last_blast_ft = field_tick;
                    }
                }
                // 2. Scorch / Hypothermia damage from temp_c extremes (>120°C or <-25°C), dilated inside TimeDomes!
                let dome_dt_p = self.sample_time_dome_scale(fx, fy);
                let scorch_cadence: u64 = if dome_dt_p < 0.5 { 36 } else if dome_dt_p > 1.5 { 2 } else { 6 };
                if tick % scorch_cadence == 0 {
                    if phys.temp_c > 120.0 {
                        let dmg = (((phys.temp_c - 120.0) / 45.0).ceil() as i16).clamp(2, 16);
                        phys_dmg_events.push((owner, pid, dmg, "SCORCH", 0));
                    } else if phys.temp_c < -25.0 {
                        let dmg = (((-25.0 - phys.temp_c) / 22.0).ceil() as i16).clamp(1, 10);
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
                            self.push_event(2, nx, ny, 0, pr.team, "OIL INFERNO!".into());
                            self.eco.inject_fear_pulse(nx, ny, 20.0, 1.0);
                        } else if has_water_or_ice {
                            let boiled = self.volume.apply_steam_burst_3d(cx, cy, pl, 4);
                            if boiled >= 6 {
                                self.push_event(2, nx, ny, 0, pr.team, "STEAM BURST!".into());
                            }
                        }
                    }
                    1 => self.volume.apply_frost_3d(cx, cy, pl, 5, 1.9 * pr.power),
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

        // Check 5-minute match timeout or 500-point early victory
        if self.score_west >= MATCH_SCORE_LIMIT
            || self.score_east >= MATCH_SCORE_LIMIT
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

    fn broadcast_entities_30hz(&mut self) {
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
            };
            let raw = encode_server_datagram(&msg);
            total_bytes += raw.len() as u64;
            match peer.conn.send_datagram(Bytes::from(raw.clone())) {
                Ok(()) => {}
                Err(quinn::SendDatagramError::ConnectionLost(_)) => dead.push(pid),
                Err(quinn::SendDatagramError::TooLarge) => {
                    let conn_c = peer.conn.clone();
                    tokio::spawn(async move {
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

    fn send_packet_datagram_or_uni(conn: &Connection, raw: Bytes) {
        if let Err(quinn::SendDatagramError::TooLarge) = conn.send_datagram(raw.clone()) {
            let conn_c = conn.clone();
            tokio::spawn(async move {
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

    fn broadcast_chunks_10hz(&mut self, dirty: &[(u8, u16, u16)]) {
        if self.peers.is_empty() {
            return;
        }
        let is_z16 = self.is_z16();
        let surf_l = self.surf_layer() as u8;
        let total_2d = self.chunks_x * self.chunks_y;
        let aoi_r = self.aoi_radius as i32;

        // Pre-encode dirty 3D chunks once (skipping completely empty air chunks on upper layers except when clearing)
        let mut dirty_map: HashMap<(u8, u16, u16), Bytes> = HashMap::with_capacity(dirty.len());
        let mut dirty_bricks: HashMap<(u16, u16), Bytes> = HashMap::new();
        for &(layer, cx, cy) in dirty {
            dirty_map.insert((layer, cx, cy), self.encode_chunk_pkt(layer, cx, cy));
            if is_z16 && !dirty_bricks.contains_key(&(cx, cy)) {
                dirty_bricks.insert((cx, cy), self.encode_brick3d_pkt(cx, cy));
            }
        }

        let any_full = self.peers.values().any(|p| p.needs_full_sync);
        let full_pkts: Vec<Bytes> = if any_full {
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
            if is_z16 {
                for by in 0..(self.chunks_y as u16) {
                    for bx in 0..(self.chunks_x as u16) {
                        if let Some(b) = dirty_bricks.get(&(bx, by)) {
                            v.push(b.clone());
                        } else {
                            v.push(self.encode_brick3d_pkt(bx, by));
                        }
                    }
                }
            }
            v
        } else {
            Vec::new()
        };

        let player_chunks: HashMap<u32, (i32, i32)> = self
            .players
            .iter()
            .map(|(&pid, p)| (pid, ((p.x as i32) / CHUNK_SIZE as i32, (p.y as i32) / CHUNK_SIZE as i32)))
            .collect();

        let cx_dim = self.chunks_x;
        let mut macro_cache: HashMap<(u16, u16), Bytes> = HashMap::new();
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
                for b in &full_pkts {
                    peer_b += b.len() as u64;
                    sent_cnt += 1;
                    Self::send_packet_datagram_or_uni(&peer.conn, b.clone());
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
                        sent_cnt += 1;
                        let _ = peer.conn.send_datagram(b.clone());
                    }
                }
                if is_z16 {
                    for (&(bx, by), b) in &dirty_bricks {
                        let in_aoi = aoi_r == 0
                            || ((bx as i32 - pcx).abs() <= aoi_r && (by as i32 - pcy).abs() <= aoi_r);
                        if in_aoi {
                            peer_b += b.len() as u64;
                            sent_cnt += 1;
                            Self::send_packet_datagram_or_uni(&peer.conn, b.clone());
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
                            sent_cnt += 1;
                            let _ = peer.conn.send_datagram(b.clone());
                        }
                    }
                    peer.macro_cursor = (peer.macro_cursor + 12) % total_2d;
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
        args.bots,
        args.aoi_radius,
    )));
    let (grid_w, grid_h, grid_d) = {
        let gs = state.lock().unwrap();
        (gs.grid_w, gs.grid_h, gs.grid_d)
    };
    log::info!(
        "Frostmarsh v3 3D VOLUMETRIC QUIC Server listening on {} ({}x{}x{} = {} voxels, backend={}, bots={}, aoi_r={})",
        args.bind,
        grid_w,
        grid_h,
        grid_d,
        grid_w * grid_h * grid_d,
        args.backend,
        args.bots,
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
                            Err(_) => {
                                // Daemon does not have world_model_dt.pt loaded; retry periodically
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
                                Err(_) => {
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
                gs.volume.apply_weather_and_springs_3d(weather_kind, tick_now);
                gs.volume.apply_solar_cycle_3d(day_time_now, tick_now);
                let stats = gs.volume.step_3d_with_neural_z(&raw_layers, z_flux_opt.as_deref());
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
                gs.broadcast_chunks_10hz(&stats.dirty_chunks);

                if stats.tick % 30 == 0 {
                    log::info!(
                        "tick={} {}x{}x{} ({} vox) mode={} backend={} world={:.2}ms eco={:.2}ms grav_z={} buoy_z={} mass_err={} energy_err={} wire={:.1}KB/s dirty={} peers={}",
                        stats.tick, gw, gh, gd, gw * gh * gd, gs.sim_mode, gs.backend_name,
                        reported_ms, eco_ms, stats.vertical_gravity_mass, stats.vertical_buoyancy_mass,
                        stats.mass_error, stats.energy_error, gs.wire_kbps, stats.dirty_chunks.len(), gs.peers.len()
                    );
                }
            }

            let elapsed = t_loop.elapsed();
            if elapsed < period {
                std::thread::sleep(period - elapsed);
            }
        }
    });

    // 30 Hz 3D Combat & Entity Snapshot broadcast
    let combat_state = state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(33));
        loop {
            interval.tick().await;
            let mut gs = combat_state.lock().unwrap();
            gs.tick_combat_30hz();
            gs.broadcast_entities_30hz();
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
                                    p.class_id = class_id % 3;
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
