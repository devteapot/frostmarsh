//! Zero-dependency JSON parser for live WebSocket multiplayer entity, actuator beam, & world evolution frames (`0x02`).
//!
//! Accepts payloads of the form (all Wave 14D fields are optional & 100% backward compatible):
//! ```json
//! {
//!   "tick": 120,
//!   "world_day": 1,
//!   "players": [
//!     { "id": 1, "class_id": 1, "x": 126.5, "y": 13.4, "z": 128.8, "yaw": 1.15, "color": [0.96, 0.78, 0.24], "scale": 1.0 }
//!   ],
//!   "projectiles": [
//!     { "id": 10, "x": 128.0, "y": 10.2, "z": 129.0, "color": [1.0, 0.45, 0.12], "radius": 0.35 }
//!   ],
//!   "beams": [
//!     { "x0": 120.5, "y0": 14.0, "z0": 124.2, "x1": 128.0, "y1": 9.1, "z1": 128.5, "color": [1.0, 0.56, 0.14] }
//!   ],
//!   "evolution": {
//!     "epoch": 42,
//!     "basalt": 128,
//!     "trees": 64,
//!     "glaciers": 19,
//!     "erosion": 310
//!   },
//!   "telemetry": {
//!     "server_tps": 30.0,
//!     "online_players": 5,
//!     "status": "Live WS"
//!   }
//! }
//! ```

use crate::hero_mesh::{archetype_default_color, HeroInstance, NUM_ARCHETYPES};
use glam::Vec3;
use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LiveBeam {
    pub start: [f32; 3],
    pub end: [f32; 3],
    pub color: [f32; 3],
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvolutionStats {
    pub epoch: u64,
    pub basalt: u64,
    pub trees: u64,
    pub glaciers: u64,
    pub erosion: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TacticalEvent {
    pub label: String,
    pub team: Option<u8>,
    pub tick: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct LiveEntitiesUpdate {
    pub tick: u64,
    pub world_day: u32,
    pub heroes: Vec<HeroInstance>,
    pub hero_classes: Vec<u8>,
    pub archetype_counts: [usize; NUM_ARCHETYPES],
    pub projectiles: Vec<HeroInstance>,
    pub beams: Vec<LiveBeam>,
    pub evolution: Option<EvolutionStats>,
    pub events: Vec<TacticalEvent>,
    pub has_events: bool,
    pub server_tps: f32,
    pub online_players: usize,
    pub status: Option<String>,
}

#[derive(Clone, Debug)]
enum JsonVal {
    Null,
    Bool,
    Num(f64),
    Str(String),
    Arr(Vec<JsonVal>),
    Obj(Vec<(String, JsonVal)>),
}

impl JsonVal {
    fn as_f32(&self) -> Option<f32> {
        match self {
            JsonVal::Num(n) => Some(*n as f32),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            JsonVal::Num(n) if *n >= 0.0 => Some(*n as u64),
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
            JsonVal::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    fn is_obj(&self) -> bool {
        matches!(self, JsonVal::Obj(_))
    }

    fn get(&self, key: &str) -> Option<&JsonVal> {
        match self {
            JsonVal::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            match self.bytes[self.pos] {
                b' ' | b'\t' | b'\r' | b'\n' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self) -> Result<JsonVal, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => self.parse_string().map(JsonVal::Str),
            Some(b't') => self.parse_literal(b"true", JsonVal::Bool),
            Some(b'f') => self.parse_literal(b"false", JsonVal::Bool),
            Some(b'n') => self.parse_literal(b"null", JsonVal::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number().map(JsonVal::Num),
            Some(c) => Err(format!("Unexpected token '{}' at byte {}", c as char, self.pos)),
            None => Err("Unexpected end of JSON input".to_string()),
        }
    }

    fn parse_literal(&mut self, lit: &[u8], val: JsonVal) -> Result<JsonVal, String> {
        if self.bytes.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(format!("Invalid JSON literal at byte {}", self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("Expected '\"' at byte {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        return Err("Unterminated string escape".to_string());
                    }
                    let esc = self.bytes[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            if self.pos + 4 > self.bytes.len() {
                                return Err("Truncated \\u escape".to_string());
                            }
                            let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                                .map_err(|_| "Invalid UTF-8 in \\u escape".to_string())?;
                            let code = u32::from_str_radix(hex, 16)
                                .map_err(|_| format!("Invalid \\u hex '{hex}'"))?;
                            self.pos += 4;
                            if let Some(ch) = char::from_u32(code) {
                                out.push(ch);
                            }
                        }
                        other => out.push(other as char),
                    }
                }
                c if c < 0x80 => out.push(c as char),
                _ => {
                    let start = self.pos - 1;
                    let rest = std::str::from_utf8(&self.bytes[start..])
                        .map_err(|_| "Invalid UTF-8 in JSON string".to_string())?;
                    if let Some(ch) = rest.chars().next() {
                        out.push(ch);
                        self.pos = start + ch.len_utf8();
                    }
                }
            }
        }
        Err("Unterminated JSON string".to_string())
    }

    fn parse_number(&mut self) -> Result<f64, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "Invalid number bytes".to_string())?;
        slice
            .parse::<f64>()
            .map_err(|e| format!("Invalid JSON number '{slice}': {e}"))
    }

    fn parse_array(&mut self) -> Result<JsonVal, String> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonVal::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonVal::Arr(items));
                }
                _ => return Err(format!("Expected ',' or ']' at byte {}", self.pos)),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonVal, String> {
        self.pos += 1; // '{'
        let mut entries = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonVal::Obj(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("Expected ':' at byte {}", self.pos));
            }
            self.pos += 1;
            let val = self.parse_value()?;
            entries.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonVal::Obj(entries));
                }
                _ => return Err(format!("Expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }
}

fn parse_rgb(obj: &JsonVal, default: [f32; 3]) -> [f32; 3] {
    if let Some(arr) = obj.get("color").and_then(JsonVal::as_arr)
        && arr.len() >= 3
        && let (Some(r), Some(g), Some(b)) = (arr[0].as_f32(), arr[1].as_f32(), arr[2].as_f32())
    {
        return [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
    }
    if let Some(fid) = obj.get("faction").or_else(|| obj.get("team")).and_then(JsonVal::as_u64) {
        return match fid % 4 {
            0 => [0.16, 0.54, 0.98],
            1 => [0.94, 0.24, 0.24],
            2 => [0.18, 0.84, 0.50],
            _ => [0.98, 0.68, 0.16],
        };
    }
    default
}

fn parse_pos_xyz(obj: &JsonVal) -> Option<[f32; 3]> {
    if let (Some(x), Some(y), Some(z)) = (
        obj.get("x").and_then(JsonVal::as_f32),
        obj.get("y").and_then(JsonVal::as_f32),
        obj.get("z").and_then(JsonVal::as_f32),
    ) {
        return Some([x, y, z]);
    }
    // Also support server grid coords (gx=east, gy=north->z, gz=alt->y)
    if let (Some(gx), Some(gy), Some(gz)) = (
        obj.get("gx").and_then(JsonVal::as_f32),
        obj.get("gy").and_then(JsonVal::as_f32),
        obj.get("gz").and_then(JsonVal::as_f32),
    ) {
        return Some([gx, gz, gy]);
    }
    None
}

fn parse_beam_endpoints(obj: &JsonVal) -> Option<([f32; 3], [f32; 3])> {
    if let (Some(x0), Some(y0), Some(z0), Some(x1), Some(y1), Some(z1)) = (
        obj.get("x0").and_then(JsonVal::as_f32),
        obj.get("y0").and_then(JsonVal::as_f32),
        obj.get("z0").and_then(JsonVal::as_f32),
        obj.get("x1").and_then(JsonVal::as_f32),
        obj.get("y1").and_then(JsonVal::as_f32),
        obj.get("z1").and_then(JsonVal::as_f32),
    ) {
        return Some(([x0, y0, z0], [x1, y1, z1]));
    }
    if let (Some(gx0), Some(gy0), Some(gz0), Some(gx1), Some(gy1), Some(gz1)) = (
        obj.get("gx0").and_then(JsonVal::as_f32),
        obj.get("gy0").and_then(JsonVal::as_f32),
        obj.get("gz0").and_then(JsonVal::as_f32),
        obj.get("gx1").and_then(JsonVal::as_f32),
        obj.get("gy1").and_then(JsonVal::as_f32),
        obj.get("gz1").and_then(JsonVal::as_f32),
    ) {
        return Some(([gx0, gz0, gy0], [gx1, gz1, gy1]));
    }
    None
}

/// Expand live diagonal 3D actuator beams into glowing volumetric orb-streams arcing from `(x0,y0,z0)` to `(x1,y1,z1)`.
pub fn expand_beams_to_orbs(beams: &[LiveBeam], sim_time: f32, out: &mut Vec<HeroInstance>) {
    for (b_idx, beam) in beams.iter().enumerate() {
        let p0 = Vec3::from_array(beam.start);
        let p1 = Vec3::from_array(beam.end);
        let delta = p1 - p0;
        let len = delta.length();
        if len < 0.05 {
            continue;
        }
        let dir = delta / len;
        let ref_up = if dir.y.abs() < 0.92 { Vec3::Y } else { Vec3::X };
        let side = dir.cross(ref_up).normalize_or(Vec3::X);
        let perp_up = side.cross(dir).normalize_or(Vec3::Y);

        let steps = ((len * 2.2).ceil() as usize).clamp(8, 24);
        let arc_lift = (len * 0.08).clamp(0.25, 1.85);
        let phase_seed = b_idx as f32 * 1.73;

        for i in 0..=steps {
            let u = i as f32 / steps as f32;
            let arc = 4.0 * u * (1.0 - u) * arc_lift;
            let env = (u * PI).sin();
            let helix_ang = u * 12.56 - sim_time * 8.5 + phase_seed;
            let helix_rad = beam.radius * 0.45 * env;
            let pos = p0.lerp(p1, u)
                + Vec3::Y * arc
                + (side * helix_ang.cos() + perp_up * helix_ang.sin()) * helix_rad;

            let wave = 0.72 + 0.42 * (u * 15.0 - sim_time * 10.0 + phase_seed).sin().abs();
            let rad = (beam.radius * wave * (0.72 + 0.34 * env)).clamp(0.08, 1.25);
            let boost = 0.25 * wave;
            let col = [
                (beam.color[0] + boost).clamp(0.0, 1.0),
                (beam.color[1] + boost).clamp(0.0, 1.0),
                (beam.color[2] + boost).clamp(0.0, 1.0),
            ];
            out.push(HeroInstance {
                pos_yaw: [pos.x, pos.y, pos.z, sim_time * 3.5 + u * 6.28],
                faction_scale: [col[0], col[1], col[2], rad],
            });
        }
    }
}

/// Parse a live WebSocket multiplayer entity + beams + evolution + telemetry JSON payload into GPU-ready instances.
pub fn parse_live_entities_json(json_str: &str) -> Result<LiveEntitiesUpdate, String> {
    let mut parser = Parser::new(json_str);
    let root = parser.parse_value()?;
    let mut out = LiveEntitiesUpdate::default();

    if let Some(t) = root.get("tick").and_then(JsonVal::as_u64) {
        out.tick = t;
    }
    if let Some(d) = root.get("world_day").and_then(JsonVal::as_u64) {
        out.world_day = d as u32;
    }

    let palette: [[f32; 3]; 4] = [
        [0.16, 0.54, 0.98],
        [0.94, 0.24, 0.24],
        [0.18, 0.84, 0.50],
        [0.98, 0.68, 0.16],
    ];

    let players_arr = root
        .get("players")
        .or_else(|| root.get("heroes"))
        .and_then(JsonVal::as_arr);

    if let Some(arr) = players_arr {
        for (idx, item) in arr.iter().enumerate() {
            if let Some(pos) = parse_pos_xyz(item) {
                let raw_class = item
                    .get("class_id")
                    .or_else(|| item.get("archetype"))
                    .and_then(JsonVal::as_u64);
                let class_id = raw_class
                    .map(|c| (c as usize % NUM_ARCHETYPES) as u8)
                    .unwrap_or(0);
                let yaw = item
                    .get("yaw")
                    .or_else(|| item.get("facing"))
                    .and_then(JsonVal::as_f32)
                    .unwrap_or(0.0);
                let scale = item
                    .get("scale")
                    .and_then(JsonVal::as_f32)
                    .unwrap_or(1.0)
                    .clamp(0.25, 4.0);
                let default_col = if raw_class.is_some() {
                    archetype_default_color(class_id)
                } else {
                    palette[idx % palette.len()]
                };
                let col = parse_rgb(item, default_col);
                out.heroes.push(HeroInstance {
                    pos_yaw: [pos[0], pos[1], pos[2], yaw],
                    faction_scale: [col[0], col[1], col[2], scale],
                });
                out.hero_classes.push(class_id);
                out.archetype_counts[class_id as usize] += 1;
            }
        }
    }

    if let Some(arr) = root.get("projectiles").and_then(JsonVal::as_arr) {
        for item in arr {
            if let Some(pos) = parse_pos_xyz(item) {
                let radius = item
                    .get("radius")
                    .or_else(|| item.get("scale"))
                    .and_then(JsonVal::as_f32)
                    .unwrap_or(0.36)
                    .clamp(0.10, 2.5);
                let col = parse_rgb(item, [1.0, 0.52, 0.14]);
                out.projectiles.push(HeroInstance {
                    pos_yaw: [pos[0], pos[1], pos[2], 0.0],
                    faction_scale: [col[0], col[1], col[2], radius],
                });
            }
        }
    }

    if let Some(arr) = root.get("beams").and_then(JsonVal::as_arr) {
        for item in arr {
            if let Some((start, end)) = parse_beam_endpoints(item) {
                let radius = item
                    .get("radius")
                    .or_else(|| item.get("width"))
                    .and_then(JsonVal::as_f32)
                    .unwrap_or(0.24)
                    .clamp(0.08, 1.2);
                let col = parse_rgb(item, [0.34, 0.88, 1.0]);
                out.beams.push(LiveBeam {
                    start,
                    end,
                    color: col,
                    radius,
                });
            }
        }
    }

    if let Some(evo) = root.get("evolution")
        && evo.is_obj()
    {
        out.evolution = Some(EvolutionStats {
            epoch: evo.get("epoch").and_then(JsonVal::as_u64).unwrap_or(0),
            basalt: evo
                .get("basalt")
                .or_else(|| evo.get("basalt_forged"))
                .and_then(JsonVal::as_u64)
                .unwrap_or(0),
            trees: evo
                .get("trees")
                .or_else(|| evo.get("trees_grown"))
                .and_then(JsonVal::as_u64)
                .unwrap_or(0),
            glaciers: evo
                .get("glaciers")
                .or_else(|| evo.get("glaciers_formed"))
                .and_then(JsonVal::as_u64)
                .unwrap_or(0),
            erosion: evo
                .get("erosion")
                .or_else(|| evo.get("erosion_events"))
                .and_then(JsonVal::as_u64)
                .unwrap_or(0),
        });
    }

    if let Some(arr) = root.get("events").and_then(JsonVal::as_arr) {
        out.has_events = true;
        for item in arr {
            if let Some(s) = item.as_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    out.events.push(TacticalEvent {
                        label: trimmed.to_string(),
                        team: None,
                        tick: None,
                    });
                }
            } else if item.is_obj()
                && let Some(lbl) = item
                    .get("label")
                    .or_else(|| item.get("text"))
                    .or_else(|| item.get("message"))
                    .and_then(JsonVal::as_str)
            {
                let trimmed = lbl.trim();
                if !trimmed.is_empty() {
                    out.events.push(TacticalEvent {
                        label: trimmed.to_string(),
                        team: item.get("team").and_then(JsonVal::as_u64).map(|t| t as u8),
                        tick: item.get("tick").and_then(JsonVal::as_u64),
                    });
                }
            }
        }
        if out.events.len() > 8 {
            let drain_n = out.events.len() - 8;
            out.events.drain(0..drain_n);
        }
    }

    out.online_players = out.heroes.len();
    if let Some(tel) = root.get("telemetry") {
        if let Some(tps) = tel.get("server_tps").and_then(JsonVal::as_f32) {
            out.server_tps = tps;
        }
        if let Some(n) = tel.get("online_players").and_then(JsonVal::as_u64) {
            out.online_players = n as usize;
        }
        if let Some(st) = tel.get("status").and_then(JsonVal::as_str) {
            out.status = Some(st.to_string());
        }
    } else if let Some(tps) = root.get("server_tps").and_then(JsonVal::as_f32) {
        out.server_tps = tps;
    }

    Ok(out)
}
