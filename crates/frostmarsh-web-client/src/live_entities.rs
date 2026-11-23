//! Zero-dependency JSON parser for live WebSocket multiplayer entity & telemetry frames (`0x02`).
//!
//! Accepts payloads of the form:
//! ```json
//! {
//!   "tick": 120,
//!   "world_day": 1,
//!   "players": [
//!     { "id": 1, "x": 126.5, "y": 9.04, "z": 128.8, "yaw": 1.15, "color": [0.16, 0.52, 0.98], "scale": 1.0 }
//!   ],
//!   "projectiles": [
//!     { "id": 10, "x": 128.0, "y": 10.2, "z": 129.0, "color": [1.0, 0.45, 0.12], "radius": 0.35 }
//!   ],
//!   "telemetry": {
//!     "server_tps": 30.0,
//!     "online_players": 4,
//!     "status": "Live WS"
//!   }
//! }
//! ```

use crate::hero_mesh::HeroInstance;

#[derive(Clone, Debug, Default)]
pub struct LiveEntitiesUpdate {
    pub tick: u64,
    pub world_day: u32,
    pub heroes: Vec<HeroInstance>,
    pub projectiles: Vec<HeroInstance>,
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
                    // Multi-byte UTF-8 sequence starting at self.pos - 1
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
    if let Some(fid) = obj.get("faction").and_then(JsonVal::as_u64) {
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

/// Parse a live WebSocket multiplayer entity + telemetry JSON payload into GPU-ready instances.
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
                let col = parse_rgb(item, palette[idx % palette.len()]);
                out.heroes.push(HeroInstance {
                    pos_yaw: [pos[0], pos[1], pos[2], yaw],
                    faction_scale: [col[0], col[1], col[2], scale],
                });
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
