//! Wave 31A: Live 6-Face Cubed-Sphere Planetary Server Runtime (`--planet`).
//!
//! Encapsulates:
//! - Live `CubedSpherePlanet` (`6 × N×N×D`) with sweeping Solar Day/Night Terminator (`sun_angle`, `sun_dir`).
//! - CUDA `op=12` (`OP_PLANET_V8_STEP`) binary framing with zero-dependency IEEE-754 `f16` / `f32`
//!   wire encoding & automatic fallback to `CubedSpherePlanet` RefSim with bit-exact conservation
//!   (`global_mass_error() == 0 && global_energy_error() == 0`).
//! - Seamless 12-edge cubed-sphere entity/projectile geodesic locomotion (`step_spherical_entity`
//!   + `EdgeLink::rotate_vec_f32` for aim/velocity rotation across seams).
//! - 6 Planetary Shrines (1 per face) & archetype-aware 60-bot spawns across all 6 faces.
//! - `/collapse_isle [0..7]` live Orbital Sky-Ring fracture trigger.
//! - Cubic `16×16×16` sub-brick temporal compression & streaming (`PlanetFaceBrick3D` / `PlanetFaceBrick3DDelta`)
//!   with empty orbital space & uniform inner-core skipping via `frostmarsh_proto::is_subbrick_skippable`.

use bytes::Bytes;
use field_core::planet::{
    CubedSpherePlanet, EDGE_COUNT, FACE_COUNT, FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X,
    FACE_POS_Y, FACE_POS_Z, SphericalEntity, edge_neighbor, step_spherical_entity,
};
use field_core::structural::StructuralStepStats;
use field_core::{
    INPUT_CHANNELS, OUTPUT_CHANNELS, RawStepOutput, SP_ICE, SP_OIL, SP_PLANT, SP_ROCK, SP_SOIL,
    SP_WATER, SP_WOOD,
};
use frostmarsh_proto::{
    CHUNK_SIZE, CLASS_BURROWER, CLASS_EMBERSTALKER, CLASS_SKYWING, CLASS_TIDEKIN,
    DEFAULT_BRICK3D_DELTA_DEADZONE, ServerDatagram, encode_brick3d_delta, encode_server_datagram,
    encode_subbrick3d_lz4, is_brick3d_delta_full_signal, is_subbrick_empty_sky,
    is_subbrick_skippable,
};
use std::collections::{HashMap, HashSet};
use std::f32::consts::TAU;
use std::sync::Arc;
use std::time::Instant;

pub const OP_PLANET_V8_STEP: u8 = 12;
pub const PLANET_FLAG_USE_SPARSE: u8 = 1 << 0;
pub const PLANET_FLAG_FP16_IO: u8 = 1 << 1;

pub type RawBrickArc = Arc<(Vec<u8>, Vec<u8>)>;

/// Zero-dependency IEEE-754 binary32 (`f32`) -> binary16 (`u16` bit pattern) conversion.
#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // Inf or NaN
        return if mant == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00 | ((mant >> 13) as u16)
        };
    }

    let exp16 = exp - 127 + 15;
    if exp16 >= 31 {
        // Overflow -> Inf
        return sign | 0x7C00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        // Subnormal f16
        let m = (mant | 0x0080_0000) >> (14 - exp16);
        let round_bit = 1u32 << (13 - exp16);
        let rounded = if (mant | 0x0080_0000) & round_bit != 0 {
            m + 1
        } else {
            m
        };
        return sign | (rounded as u16);
    }

    // Normal f16 with round-to-nearest-even
    let half_mant = mant >> 13;
    let rem = mant & 0x1FFF;
    let mut out = sign | ((exp16 as u16) << 10) | (half_mant as u16);
    if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) != 0) {
        out = out.wrapping_add(1);
    }
    out
}

/// Zero-dependency IEEE-754 binary16 (`u16` bit pattern) -> binary32 (`f32`) conversion.
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    let out_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Normalize subnormal
            let mut m = mant;
            let mut e = 127 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf or NaN
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out_bits)
}

/// Decoded output from one `op=12` (`OP_PLANET_V8_STEP`) GPU daemon call.
pub struct PlanetOp12Response {
    pub infer_ms: f32,
    pub active_frac: f32,
    pub horiz_faces: [Vec<RawStepOutput>; FACE_COUNT],
    pub zflux_faces: [Vec<f32>; FACE_COUNT],
}

/// Pack a `CubedSpherePlanet` state into the `op=12` binary request payload:
/// `[u16 N][u16 L][f32 sun_x][f32 sun_y][f32 sun_z][u8 flags][6 * L * 18 * N * N f16/f32]`.
pub fn encode_planet_op12_request(
    planet: &CubedSpherePlanet,
    sun_dir: [f32; 3],
    flags: u8,
) -> Vec<u8> {
    let n = planet.face_size;
    let l = planet.depth;
    let fp16_io = (flags & PLANET_FLAG_FP16_IO) != 0;
    let elem_bytes = if fp16_io { 2usize } else { 4usize };
    let per_face_elems = l * INPUT_CHANNELS * n * n;
    let total_bytes = 17 + FACE_COUNT * per_face_elems * elem_bytes;

    let mut out = Vec::with_capacity(total_bytes);
    out.extend_from_slice(&(n as u16).to_le_bytes());
    out.extend_from_slice(&(l as u16).to_le_bytes());
    out.extend_from_slice(&sun_dir[0].to_le_bytes());
    out.extend_from_slice(&sun_dir[1].to_le_bytes());
    out.extend_from_slice(&sun_dir[2].to_le_bytes());
    out.push(flags);

    for f in 0..FACE_COUNT {
        let enc = planet.faces[f].encode_batched_input_tensor();
        debug_assert_eq!(enc.len(), per_face_elems);
        if fp16_io {
            for &v in &enc {
                out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
        } else {
            for &v in &enc {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

/// Unpack an `op=12` binary response payload:
/// `[f32 infer_ms][f32 active_frac][6 * L * 17 * N * N f16/f32 horiz][6 * (L-1) * 5 * N * N f16/f32 zflux]`.
pub fn decode_planet_op12_response(
    buf: &[u8],
    n: usize,
    l: usize,
    fp16_io: bool,
) -> Result<PlanetOp12Response, String> {
    let elem_bytes = if fp16_io { 2usize } else { 4usize };
    let per_layer_h = OUTPUT_CHANNELS * n * n;
    let per_face_h = l * per_layer_h;
    let per_face_z = l.saturating_sub(1) * 5 * n * n;
    let horiz_elems = FACE_COUNT * per_face_h;
    let zflux_elems = FACE_COUNT * per_face_z;
    let expected = 8 + (horiz_elems + zflux_elems) * elem_bytes;
    if buf.len() != expected {
        return Err(format!(
            "op=12 response len mismatch: got {} bytes, expected {}",
            buf.len(),
            expected
        ));
    }
    let infer_ms = f32::from_le_bytes(buf[0..4].try_into().unwrap());
    let active_frac = f32::from_le_bytes(buf[4..8].try_into().unwrap());

    let mut pos = 8usize;
    let read_f32s = |src: &[u8], offset: &mut usize, count: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(count);
        if fp16_io {
            let end = *offset + count * 2;
            for chunk in src[*offset..end].chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = f16_to_f32(bits);
                v.push(if val.is_finite() { val } else { 0.0 });
            }
            *offset = end;
        } else {
            let end = *offset + count * 4;
            for chunk in src[*offset..end].chunks_exact(4) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                v.push(if val.is_finite() { val } else { 0.0 });
            }
            *offset = end;
        }
        v
    };

    let horiz_faces: [Vec<RawStepOutput>; FACE_COUNT] = std::array::from_fn(|_| {
        let mut layers = Vec::with_capacity(l);
        for _ in 0..l {
            let data = read_f32s(buf, &mut pos, per_layer_h);
            layers.push(RawStepOutput {
                width: n,
                height: n,
                data,
            });
        }
        layers
    });
    let zflux_faces: [Vec<f32>; FACE_COUNT] =
        std::array::from_fn(|_| read_f32s(buf, &mut pos, per_face_z));

    Ok(PlanetOp12Response {
        infer_ms,
        active_frac,
        horiz_faces,
        zflux_faces,
    })
}

/// Summary returned by one 10 Hz `PlanetRuntime::step_planet_neural_or_refsim` tick.
pub struct PlanetTickOutcome {
    pub tick: u64,
    pub infer_ms: f32,
    pub active_frac: f32,
    pub backend: String,
    pub neural_3d_vertical: bool,
    pub global_mass_error: i64,
    pub global_energy_error: i64,
    pub seam_mass_crossed: i64,
    pub fallen_solid_mass: i64,
    pub splashed_fluid_mass: i64,
    pub dirty_subbricks: Vec<(u8, u16, u16, u16)>,
    pub newly_fracturing_cores: Vec<(u8, u8, String, f32, f32, f32)>,
    pub newly_impacted_cores: Vec<(u8, u8, String, f32, f32, f32)>,
}

/// Live 6-Face Cubed-Sphere Planetary Server State.
pub struct PlanetRuntime {
    pub planet: CubedSpherePlanet,
    pub sun_angle: f32,
    pub sun_speed_rad_per_tick: f32,
    pub subbrick_raw_cache: HashMap<(u8, u16, u16, u16), (u8, RawBrickArc)>,
    pub full_sync_cursor: usize,
    pub last_infer_ms: f32,
    pub last_active_frac: f32,
    pub last_backend: String,
    pub neural_3d_vertical: bool,
    pub seam_crossings_total: u64,
    pub op12_backoff: u32,
}

impl PlanetRuntime {
    pub fn new(face_size: usize, depth: usize, r_core: f32, seed: u64) -> Self {
        let mut planet = CubedSpherePlanet::new(face_size, depth, r_core, seed);
        let sun_angle = 0.35f32;
        planet.sun_dir = sun_dir_from_angle(sun_angle);
        Self {
            planet,
            sun_angle,
            sun_speed_rad_per_tick: 0.018, // ~1 full solar day per ~35s at 10 Hz
            subbrick_raw_cache: HashMap::new(),
            full_sync_cursor: 0,
            last_infer_ms: 0.0,
            last_active_frac: 1.0,
            last_backend: "refsim-planet".to_string(),
            neural_3d_vertical: false,
            seam_crossings_total: 0,
            op12_backoff: 0,
        }
    }

    #[inline]
    pub fn face_size(&self) -> usize {
        self.planet.face_size
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.planet.depth
    }

    #[inline]
    pub fn chunks_xy(&self) -> usize {
        self.planet.face_size.div_ceil(CHUNK_SIZE).max(1)
    }

    #[inline]
    pub fn subbricks_z(&self) -> u16 {
        self.planet.depth.div_ceil(CHUNK_SIZE).max(1) as u16
    }

    /// Advance the sweeping equatorial Day/Night Terminator sun direction.
    pub fn advance_sun_dir(&mut self) -> [f32; 3] {
        self.sun_angle = (self.sun_angle + self.sun_speed_rad_per_tick).rem_euclid(TAU);
        let dir = sun_dir_from_angle(self.sun_angle);
        self.planet.sun_dir = dir;
        dir
    }

    /// Count intact levitating Orbital Sky-Ring `BuoyantCore`s across all 6 faces (0..=8).
    pub fn active_sky_cores(&self) -> u8 {
        self.planet
            .faces
            .iter()
            .flat_map(|f| f.structural.buoyant_cores.iter())
            .filter(|c| !c.collapsed && c.integrity > 0.05)
            .count()
            .min(255) as u8
    }

    /// Trigger conservation-exact fracture/demagnetization of Orbital Sky-Ring Island `isle_index` (`0..7`).
    /// Returns `Some((face_u8, core_name, cx, cy, cz))` if an uncollapsed core was fractured.
    pub fn collapse_orbital_isle(&mut self, isle_index: u8) -> Option<(u8, String, f32, f32, f32)> {
        // First try exact global core.id match across all 6 faces
        for f in 0..FACE_COUNT {
            if let Some(cid) = self.planet.faces[f]
                .structural
                .buoyant_cores
                .iter()
                .find(|c| c.id == isle_index)
                .map(|c| c.id)
            {
                if let Some((name, cx, cy, cz)) = self.planet.fracture_orbital_island(f, cid) {
                    // Latch structural demagnetization immediately while core is at 620 K
                    // (before next 3D thermodynamics tick melts core ice via latent heat).
                    let _ = self.planet.faces[f].step_structural_mechanics();
                    return Some((f as u8, name, cx, cy, cz));
                }
            }
        }
        // Fallback: map `0..7` across the 4 equatorial faces (2 cores per face)
        let eq_faces = [FACE_POS_X, FACE_NEG_X, FACE_POS_Z, FACE_NEG_Z];
        let target_face = eq_faces[((isle_index as usize) / 2) % eq_faces.len()];
        let local_idx = (isle_index as usize) % 2;
        if let Some(cid) = self.planet.faces[target_face]
            .structural
            .buoyant_cores
            .get(local_idx)
            .or_else(|| self.planet.faces[target_face].structural.buoyant_cores.first())
            .map(|c| c.id)
        {
            if let Some((name, cx, cy, cz)) = self.planet.fracture_orbital_island(target_face, cid)
            {
                let _ = self.planet.faces[target_face].step_structural_mechanics();
                return Some((target_face as u8, name, cx, cy, cz));
            }
        }
        None
    }

    /// Run one full 10 Hz planetary physics tick:
    /// - If `use_neural && self.op12_backoff == 0`, calls CUDA `op=12` (`OP_PLANET_V8_STEP`),
    ///   unpacks `(horiz, zflux)` per face, and applies `VolumeField::step_3d_with_neural_z`
    ///   in parallel across all 6 faces (`std::thread::scope`).
    /// - Otherwise (or on `op=12` error), falls back cleanly to parallel 6-face RefSim.
    /// - Runs `exchange_12_edges_conservative()`, `apply_planetary_solar_terminator(sun_dir, tick)`,
    ///   and parallel `step_structural_mechanics()` across all 6 faces.
    /// - Refreshes cached `16×16×16` cubic sub-bricks for all dirty regions and guarantees
    ///   **`global_mass_error == 0 && global_energy_error == 0`**!
    pub fn step_planet_neural_or_refsim<F>(
        &mut self,
        use_neural: bool,
        mut call_op12: F,
    ) -> PlanetTickOutcome
    where
        F: FnMut(&CubedSpherePlanet, [f32; 3], u8) -> Result<PlanetOp12Response, String>,
    {
        let t0 = Instant::now();
        let sun_dir = self.advance_sun_dir();
        let tick_now = self.planet.tick;

        let mut dirty_set: HashSet<(u8, u16, u16, u16)> = HashSet::new();
        let mut used_neural = false;
        let mut reported_infer_ms = 0.0f32;
        let mut active_frac = 1.0f32;

        if use_neural && self.op12_backoff == 0 {
            let flags = PLANET_FLAG_USE_SPARSE | PLANET_FLAG_FP16_IO;
            match call_op12(&self.planet, sun_dir, flags) {
                Ok(resp) => {
                    reported_infer_ms = resp.infer_ms;
                    active_frac = resp.active_frac.clamp(0.0, 1.0);
                    let PlanetOp12Response {
                        horiz_faces,
                        zflux_faces,
                        ..
                    } = resp;
                    let mut face_dirty: [Vec<(u8, u16, u16)>; FACE_COUNT] =
                        std::array::from_fn(|_| Vec::new());
                    std::thread::scope(|s| {
                        let mut handles = Vec::with_capacity(FACE_COUNT);
                        for ((face, horiz), zf) in self
                            .planet
                            .faces
                            .iter_mut()
                            .zip(horiz_faces.iter())
                            .zip(zflux_faces.iter())
                        {
                            handles.push(s.spawn(move || {
                                let st = face.step_3d_with_neural_z(horiz, Some(zf.as_slice()));
                                st.dirty_chunks
                            }));
                        }
                        for (fi, h) in handles.into_iter().enumerate() {
                            face_dirty[fi] = h.join().expect("planet neural face step panicked");
                        }
                    });
                    for (fi, chunks) in face_dirty.into_iter().enumerate() {
                        for (layer, cx, cy) in chunks {
                            let bz = (layer as usize / CHUNK_SIZE) as u16;
                            dirty_set.insert((fi as u8, cx, cy, bz));
                        }
                    }
                    used_neural = true;
                }
                Err(err) => {
                    log::debug!(
                        "CUDA op=12 (OP_PLANET_V8_STEP) unavailable ({}); falling back to 6-face RefSim",
                        err
                    );
                    self.op12_backoff = 50;
                }
            }
        } else if self.op12_backoff > 0 {
            self.op12_backoff -= 1;
        }

        if !used_neural {
            let mut face_dirty: [Vec<(u8, u16, u16)>; FACE_COUNT] =
                std::array::from_fn(|_| Vec::new());
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(FACE_COUNT);
                for face in &mut self.planet.faces {
                    handles.push(s.spawn(move || {
                        let raw = face.refsim_raw_outputs_all();
                        let st = face.step_3d(&raw);
                        st.dirty_chunks
                    }));
                }
                for (fi, h) in handles.into_iter().enumerate() {
                    face_dirty[fi] = h.join().expect("planet refsim face step panicked");
                }
            });
            for (fi, chunks) in face_dirty.into_iter().enumerate() {
                for (layer, cx, cy) in chunks {
                    let bz = (layer as usize / CHUNK_SIZE) as u16;
                    dirty_set.insert((fi as u8, cx, cy, bz));
                }
            }
            active_frac = 1.0;
        }

        // 2. Exact 12-edge antisymmetric cross-face conservation exchange
        let edge_stats = self.planet.exchange_12_edges_conservative();

        // 3. Sweeping planetary Solar Day/Night Terminator
        self.planet.apply_planetary_solar_terminator(sun_dir, tick_now);

        // 4. Parallel `RadialInnerShell` structural support & Orbital Sky-Ring collapse
        let mut struct_results: [StructuralStepStats; FACE_COUNT] =
            std::array::from_fn(|_| StructuralStepStats::default());
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(FACE_COUNT);
            for face in &mut self.planet.faces {
                handles.push(s.spawn(move || face.step_structural_mechanics()));
            }
            for (fi, h) in handles.into_iter().enumerate() {
                struct_results[fi] = h.join().expect("planet structural worker panicked");
            }
        });

        let mut fallen_solid_mass = 0i64;
        let mut splashed_fluid_mass = 0i64;
        let mut newly_fracturing_cores = Vec::new();
        let mut newly_impacted_cores = Vec::new();
        let chunks_xy = self.chunks_xy() as u16;
        let bz_dim = self.subbricks_z();

        for (fi, sr) in struct_results.into_iter().enumerate() {
            fallen_solid_mass += sr.fallen_solid_mass;
            splashed_fluid_mass += sr.splashed_fluid_mass;
            if sr.fallen_solid_mass > 0 || sr.splashed_fluid_mass > 0 {
                // Mark sub-bricks near any active/collapsed core dirty so the collapse streams immediately
                for core in &self.planet.faces[fi].structural.buoyant_cores {
                    if core.collapsed || core.integrity < 0.99 {
                        let bx = ((core.cx as usize) / CHUNK_SIZE).min(chunks_xy as usize - 1) as u16;
                        let by = ((core.cy as usize) / CHUNK_SIZE).min(chunks_xy as usize - 1) as u16;
                        for bz in 0..bz_dim {
                            dirty_set.insert((fi as u8, bx, by, bz));
                            if bx > 0 {
                                dirty_set.insert((fi as u8, bx - 1, by, bz));
                            }
                            if bx + 1 < chunks_xy {
                                dirty_set.insert((fi as u8, bx + 1, by, bz));
                            }
                            if by > 0 {
                                dirty_set.insert((fi as u8, bx, by - 1, bz));
                            }
                            if by + 1 < chunks_xy {
                                dirty_set.insert((fi as u8, bx, by + 1, bz));
                            }
                        }
                    }
                }
            }
            for (cid, cname, cx, cy, cz) in sr.newly_fracturing_cores {
                newly_fracturing_cores.push((fi as u8, cid, cname, cx, cy, cz));
            }
            for (cid, cname, cx, cy, cz) in sr.newly_impacted_cores {
                newly_impacted_cores.push((fi as u8, cid, cname, cx, cy, cz));
            }
        }

        // Rotate a small ring of surface sub-bricks so sweeping solar terminator thermal shifts stream smoothly
        let total_2d = (chunks_xy as usize) * (chunks_xy as usize);
        let surf_bz = ((self.planet.depth / 2) / CHUNK_SIZE) as u16;
        if total_2d > 0 {
            for f in 0..FACE_COUNT {
                let ci = (self.full_sync_cursor + f) % total_2d;
                let bx = (ci % chunks_xy as usize) as u16;
                let by = (ci / chunks_xy as usize) as u16;
                dirty_set.insert((f as u8, bx, by, surf_bz.min(bz_dim - 1)));
            }
            self.full_sync_cursor = (self.full_sync_cursor + 1) % total_2d;
        }

        self.planet.tick += 1;

        let dirty_subbricks: Vec<(u8, u16, u16, u16)> = dirty_set.into_iter().collect();
        for &(face, bx, by, bz) in &dirty_subbricks {
            self.refresh_subbrick(face, bx, by, bz);
        }

        let total_ms = t0.elapsed().as_secs_f32() * 1000.0;
        let infer_ms = if used_neural { reported_infer_ms } else { total_ms };
        let backend = if used_neural {
            "cuda-fp16-planet-v8".to_string()
        } else {
            "refsim-planet".to_string()
        };
        self.last_infer_ms = infer_ms;
        self.last_active_frac = active_frac;
        self.last_backend = backend.clone();
        self.neural_3d_vertical = used_neural;

        PlanetTickOutcome {
            tick: self.planet.tick,
            infer_ms,
            active_frac,
            backend,
            neural_3d_vertical: used_neural,
            global_mass_error: self.planet.global_mass_error(),
            global_energy_error: self.planet.global_energy_error(),
            seam_mass_crossed: edge_stats.seam_mass_crossed,
            fallen_solid_mass,
            splashed_fluid_mass,
            dirty_subbricks,
            newly_fracturing_cores,
            newly_impacted_cores,
        }
    }

    // ---------------- Planetary Surface, Shrines & 60-Bot Spawns ----------------

    /// Find highest solid planetary surface altitude `z` below the orbital Sky-Ring on `face`.
    pub fn find_surface_z(&self, face: u8, x: f32, y: f32) -> f32 {
        let vf = &self.planet.faces[(face as usize) % FACE_COUNT];
        let d = vf.depth;
        let ring_floor = d.saturating_sub(11).max(4);
        let xi = (x.floor() as i32).clamp(0, (vf.width as i32) - 1) as usize;
        let yi = (y.floor() as i32).clamp(0, (vf.height as i32) - 1) as usize;
        let idx = yi * vf.width + xi;

        for z in (1..ring_floor).rev() {
            let l = &vf.layers[z];
            let solid = l.mass[SP_ROCK][idx]
                + l.mass[SP_SOIL][idx]
                + l.mass[SP_ICE][idx]
                + l.mass[SP_WOOD][idx]
                + l.mass[SP_PLANT][idx];
            if solid >= 200 {
                return (z as f32 + 1.15).min((d as f32) - 2.0);
            }
        }
        // Fallback to sea level if column is open ocean
        let z_sea = ((15.0 * (d as f32 / 32.0)).round() as usize).clamp(2, d - 3);
        (z_sea as f32) + 0.6
    }

    /// Locate a dry land cell near `(pref_x, pref_y)` on `face` (avoiding deep ocean & molten lava).
    pub fn find_land_spot(&self, face: u8, pref_x: f32, pref_y: f32) -> (f32, f32, f32) {
        let vf = &self.planet.faces[(face as usize) % FACE_COUNT];
        let n = vf.width;
        let d = vf.depth;
        let ring_floor = d.saturating_sub(11).max(4);
        let z_sea = ((15.0 * (d as f32 / 32.0)).round() as usize).clamp(2, d - 3);

        let cx = (pref_x.round() as i32).clamp(4, (n as i32) - 5);
        let cy = (pref_y.round() as i32).clamp(4, (n as i32) - 5);

        let is_dry_land = |gx: usize, gy: usize| -> Option<f32> {
            let idx = gy * n + gx;
            for z in (z_sea..ring_floor).rev() {
                let l = &vf.layers[z];
                let solid = l.mass[SP_ROCK][idx]
                    + l.mass[SP_SOIL][idx]
                    + l.mass[SP_ICE][idx]
                    + l.mass[SP_WOOD][idx]
                    + l.mass[SP_PLANT][idx];
                let water = l.mass[SP_WATER][idx];
                let oil = l.mass[SP_OIL][idx];
                let temp = l.cell_temp_ck(idx);
                if solid >= 300 && water < 250 && oil < 150 && temp < 60_000 {
                    return Some((z as f32 + 1.15).min((d as f32) - 2.0));
                }
            }
            None
        };

        if let Some(z) = is_dry_land(cx as usize, cy as usize) {
            return (cx as f32 + 0.5, cy as f32 + 0.5, z);
        }

        let max_r = (n / 2).saturating_sub(4) as i32;
        for rad in (1..=max_r).step_by(2) {
            for dy in (-rad..=rad).step_by(2) {
                for dx in (-rad..=rad).step_by(2) {
                    if dx.abs() != rad && dy.abs() != rad {
                        continue;
                    }
                    let gx = (cx + dx).clamp(3, (n as i32) - 4) as usize;
                    let gy = (cy + dy).clamp(3, (n as i32) - 4) as usize;
                    if let Some(z) = is_dry_land(gx, gy) {
                        return (gx as f32 + 0.5, gy as f32 + 0.5, z);
                    }
                }
            }
        }
        let z = self.find_surface_z(face, pref_x, pref_y);
        (pref_x.clamp(4.0, n as f32 - 4.0), pref_y.clamp(4.0, n as f32 - 4.0), z)
    }

    /// Build the 6 canonical planetary shrines (1 per cube face) snapped onto dry land cells.
    pub fn build_planetary_shrines(&self) -> Vec<(u8, String, u8, f32, f32, f32)> {
        let nf = self.planet.face_size as f32;
        let specs: [(usize, &str, f32, f32); FACE_COUNT] = [
            (FACE_POS_X, "+X EQUATORIAL SPIRE", 0.50, 0.50),
            (FACE_NEG_X, "-X RIFT FORGE", 0.50, 0.50),
            (FACE_POS_Y, "+Y NORTH POLAR CITADEL", 0.50, 0.50),
            (FACE_NEG_Y, "-Y SOUTH GLACIAL CORE", 0.50, 0.50),
            (FACE_POS_Z, "+Z EMERALD BASIN", 0.50, 0.50),
            (FACE_NEG_Z, "-Z SAPPHIRE ATOLL", 0.50, 0.50),
        ];
        specs
            .into_iter()
            .enumerate()
            .map(|(id, (face, name, fu, fv))| {
                let (x, y, z) = self.find_land_spot(face as u8, fu * nf, fv * nf);
                (id as u8, name.to_string(), face as u8, x, y, z)
            })
            .collect()
    }

    /// Compute an archetype-appropriate planetary spawn `(face, x, y, z)`.
    pub fn spawn_point_for_entity(&self, face: u8, class_id: u8, salt: u32) -> (u8, f32, f32, f32) {
        let f = face % (FACE_COUNT as u8);
        let nf = self.planet.face_size as f32;
        let d = self.planet.depth as f32;

        // Deterministic spread across the face (including cells near seams so bots traverse edges quickly!)
        let angle = (salt as f32) * 2.399_963_1;
        let frac_r = 0.14 + 0.30 * (((salt % 11) as f32) / 10.0);
        let pref_x = (nf * (0.5 + frac_r * angle.cos())).clamp(4.0, nf - 4.0);
        let pref_y = (nf * (0.5 + frac_r * angle.sin())).clamp(4.0, nf - 4.0);

        match class_id % 5 {
            CLASS_SKYWING => {
                // Upper Orbital Sky-Ring altitude (`depth - 8 .. depth - 4`)
                let sky_lo = (d - 8.0).max(8.0);
                let sky_hi = (d - 4.0).max(sky_lo + 0.5);
                let sky_z = sky_lo + ((salt % 5) as f32 / 4.0) * (sky_hi - sky_lo);
                (f, pref_x, pref_y, sky_z.clamp(2.0, d - 1.5))
            }
            CLASS_TIDEKIN => {
                // Ocean surface level (`~15/32 * depth`)
                let z_sea = (15.0 * (d / 32.0)).clamp(3.0, d - 4.0);
                let z_surf = self.find_surface_z(f, pref_x, pref_y);
                (f, pref_x, pref_y, z_surf.max(z_sea + 0.4).min(d - 2.0))
            }
            CLASS_BURROWER => {
                // Mantle/surface border
                let (lx, ly, lz) = self.find_land_spot(f, pref_x, pref_y);
                let z_burrow = (lz - 1.5).max(5.0).min(d - 3.0);
                (f, lx, ly, z_burrow)
            }
            CLASS_EMBERSTALKER => {
                let (lx, ly, lz) = self.find_land_spot(f, pref_x, pref_y);
                (f, lx, ly, lz)
            }
            _ => {
                // CLASS_STRIDER
                let (lx, ly, lz) = self.find_land_spot(f, pref_x, pref_y);
                (f, lx, ly, lz)
            }
        }
    }

    // ---------------- Seamless 12-Edge Spherical Locomotion ----------------

    /// Advance a 3D entity or projectile along `(vx, vy)` on the cubed sphere using
    /// `field_core::planet::step_spherical_entity` and rotate its `(aim_x, aim_y)` vector
    /// cleanly across any crossed cube seam via `EdgeLink::rotate_vec_f32`!
    #[allow(clippy::too_many_arguments)]
    pub fn step_entity_on_sphere(
        face: &mut u8,
        x: &mut f32,
        y: &mut f32,
        z: f32,
        vx: &mut f32,
        vy: &mut f32,
        aim_x: &mut f32,
        aim_y: &mut f32,
        face_size: usize,
        dt: f32,
    ) -> u32 {
        let old_face = (*face as usize) % FACE_COUNT;
        let old_vx = *vx;
        let old_vy = *vy;

        let mut ent = SphericalEntity {
            face: old_face,
            u_cell: *x,
            v_cell: *y,
            z,
            vu: old_vx,
            vv: old_vy,
        };
        let crossings = step_spherical_entity(&mut ent, face_size, dt);
        if crossings > 0 && ent.face != old_face {
            // Locate exact shared edge link when a single edge was crossed, and rotate (aim_x, aim_y)
            let mut rotated = false;
            if crossings == 1 {
                for e in 0..EDGE_COUNT {
                    let link = edge_neighbor(old_face, e);
                    if link.face == ent.face {
                        let (nax, nay) = link.rotate_vec_f32(*aim_x, *aim_y);
                        *aim_x = nax;
                        *aim_y = nay;
                        rotated = true;
                        break;
                    }
                }
            }
            if !rotated {
                // Rare corner double-crossing: align aim with rotated velocity direction
                let spd = ent.vu.hypot(ent.vv);
                if spd > 1e-4 {
                    *aim_x = ent.vu / spd;
                    *aim_y = ent.vv / spd;
                }
            }
        }

        *face = (ent.face % FACE_COUNT) as u8;
        *x = ent.u_cell;
        *y = ent.v_cell;
        *vx = ent.vu;
        *vy = ent.vv;
        crossings
    }

    // ---------------- Cubic 16×16×16 Sub-Brick Streaming ----------------

    pub fn get_or_encode_subbrick(&mut self, face: u8, bx: u16, by: u16, bz: u16) -> (u8, RawBrickArc) {
        let f = face % (FACE_COUNT as u8);
        if let Some((sd, arc)) = self.subbrick_raw_cache.get(&(f, bx, by, bz)) {
            return (*sd, arc.clone());
        }
        self.refresh_subbrick(f, bx, by, bz)
    }

    pub fn refresh_subbrick(&mut self, face: u8, bx: u16, by: u16, bz: u16) -> (u8, RawBrickArc) {
        let f = face % (FACE_COUNT as u8);
        let (dm, at, sd) =
            self.planet.faces[f as usize].encode_3d_subbrick_16(bx as usize, by as usize, bz as usize);
        let raw = Arc::new((dm, at));
        self.subbrick_raw_cache.insert((f, bx, by, bz), (sd, raw.clone()));
        (sd, raw)
    }

    pub fn encode_planet_subbrick_full(
        &self,
        face: u8,
        bx: u16,
        by: u16,
        bz: u16,
        sub_depth: u8,
        dm: &[u8],
        at: &[u8],
    ) -> Bytes {
        let (lz4_density_mat, lz4_albedo_thermal) = encode_subbrick3d_lz4(dm, at);
        let pkt = ServerDatagram::PlanetFaceBrick3D {
            face,
            bx,
            by,
            bz,
            total_depth: self.planet.depth as u16,
            sub_depth,
            tick: self.planet.tick,
            lz4_density_mat,
            lz4_albedo_thermal,
        };
        Bytes::from(encode_server_datagram(&pkt))
    }

    /// Encode all non-skippable `16×16×sub_depth` cubic sub-bricks across all 6 faces for initial sync.
    /// Skips pure empty orbital space and pure uniform inner core sub-bricks via `is_subbrick_skippable`.
    /// Returns `(full_pkts, empty_sky_shadows, total_subbricks, skipped_subbricks)`.
    #[allow(clippy::type_complexity)]
    pub fn encode_full_planet_subbricks(
        &mut self,
    ) -> (
        Vec<((u8, u16, u16, u16), RawBrickArc, Bytes)>,
        Vec<((u8, u16, u16, u16), RawBrickArc)>,
        usize,
        usize,
    ) {
        let cx_dim = self.chunks_xy() as u16;
        let cy_dim = cx_dim;
        let bz_dim = self.subbricks_z();
        let total = FACE_COUNT * (cx_dim as usize) * (cy_dim as usize) * (bz_dim as usize);
        let mut pkts = Vec::with_capacity(total / 2);
        let mut empty_shadows = Vec::new();
        let mut skipped = 0usize;

        for f in 0..(FACE_COUNT as u8) {
            for bz in 0..bz_dim {
                for by in 0..cy_dim {
                    for bx in 0..cx_dim {
                        let (sd, raw) = self.get_or_encode_subbrick(f, bx, by, bz);
                        if is_subbrick_skippable(&raw.0, &raw.1) {
                            skipped += 1;
                            if is_subbrick_empty_sky(&raw.0, &raw.1) {
                                empty_shadows.push(((f, bx, by, bz), raw));
                            }
                            continue;
                        }
                        let pkt = self.encode_planet_subbrick_full(f, bx, by, bz, sd, &raw.0, &raw.1);
                        pkts.push(((f, bx, by, bz), raw, pkt));
                    }
                }
            }
        }
        (pkts, empty_shadows, total, skipped)
    }

    /// Encode an incremental `PlanetFaceBrick3DDelta` (or full `PlanetFaceBrick3D` if `>= 35%` changed
    /// or no shadow exists) for one sub-brick `(face, bx, by, bz)`.
    /// Returns `Ok(Some((bytes, is_delta, cur_arc)))` when a packet should be sent,
    /// or `Ok(None)` when unchanged / skippable.
    pub fn encode_incremental_subbrick(
        &mut self,
        face: u8,
        bx: u16,
        by: u16,
        bz: u16,
        prev_shadow: Option<&RawBrickArc>,
        deadzone: u8,
    ) -> Option<(Bytes, bool, RawBrickArc)> {
        let (sub_depth, cur_arc) = self.get_or_encode_subbrick(face, bx, by, bz);
        match prev_shadow {
            Some(prev_arc) if Arc::ptr_eq(prev_arc, &cur_arc) => None,
            Some(prev_arc) => {
                match encode_brick3d_delta(
                    &prev_arc.0,
                    &prev_arc.1,
                    &cur_arc.0,
                    &cur_arc.1,
                    deadzone.max(DEFAULT_BRICK3D_DELTA_DEADZONE),
                ) {
                    None => None,
                    Some(lz4_deltas) if !is_brick3d_delta_full_signal(&lz4_deltas) => {
                        let pkt = ServerDatagram::PlanetFaceBrick3DDelta {
                            face,
                            bx,
                            by,
                            bz,
                            total_depth: self.planet.depth as u16,
                            sub_depth,
                            tick: self.planet.tick,
                            lz4_deltas,
                        };
                        Some((Bytes::from(encode_server_datagram(&pkt)), true, cur_arc))
                    }
                    Some(_) => {
                        let raw_pkt = self.encode_planet_subbrick_full(
                            face, bx, by, bz, sub_depth, &cur_arc.0, &cur_arc.1,
                        );
                        Some((raw_pkt, false, cur_arc))
                    }
                }
            }
            None => {
                if is_subbrick_skippable(&cur_arc.0, &cur_arc.1) {
                    return None;
                }
                let raw_pkt = self.encode_planet_subbrick_full(
                    face, bx, by, bz, sub_depth, &cur_arc.0, &cur_arc.1,
                );
                Some((raw_pkt, false, cur_arc))
            }
        }
    }
}

#[inline]
pub fn sun_dir_from_angle(angle: f32) -> [f32; 3] {
    let sx = angle.cos();
    let sy = 0.24f32;
    let sz = angle.sin();
    let inv = 1.0 / (sx * sx + sy * sy + sz * sz).sqrt().max(1e-8);
    [sx * inv, sy * inv, sz * inv]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip_accuracy() {
        let samples = [0.0f32, -0.0, 1.0, -1.0, 0.5, -2.75, 0.0009765625, 123.5, 65504.0];
        for &v in &samples {
            let h = f32_to_f16(v);
            let rt = f16_to_f32(h);
            assert!(
                (rt - v).abs() <= v.abs() * 1e-3 + 1e-4,
                "f16 roundtrip mismatch for {}: got {}",
                v,
                rt
            );
        }
        assert!(f16_to_f32(f32_to_f16(f32::INFINITY)).is_infinite());
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn test_op12_request_response_framing_fp16_and_fp32() {
        let planet = CubedSpherePlanet::new_empty(16, 8, 24.0, 7);
        let sun = sun_dir_from_angle(0.5);

        for &flags in &[PLANET_FLAG_USE_SPARSE, PLANET_FLAG_USE_SPARSE | PLANET_FLAG_FP16_IO] {
            let fp16 = (flags & PLANET_FLAG_FP16_IO) != 0;
            let req = encode_planet_op12_request(&planet, sun, flags);
            let elem_b = if fp16 { 2 } else { 4 };
            assert_eq!(req.len(), 17 + 6 * 8 * 18 * 16 * 16 * elem_b);
            assert_eq!(u16::from_le_bytes([req[0], req[1]]), 16);
            assert_eq!(u16::from_le_bytes([req[2], req[3]]), 8);
            assert_eq!(req[16], flags);

            // Synthesize a valid response payload and verify decode
            let h_elems = 6 * 8 * 17 * 16 * 16;
            let z_elems = 6 * 7 * 5 * 16 * 16;
            let mut resp = Vec::with_capacity(8 + (h_elems + z_elems) * elem_b);
            resp.extend_from_slice(&12.5f32.to_le_bytes());
            resp.extend_from_slice(&0.18f32.to_le_bytes());
            for i in 0..(h_elems + z_elems) {
                let val = if i == 0 { 0.25f32 } else { 0.0f32 };
                if fp16 {
                    resp.extend_from_slice(&f32_to_f16(val).to_le_bytes());
                } else {
                    resp.extend_from_slice(&val.to_le_bytes());
                }
            }
            let dec = decode_planet_op12_response(&resp, 16, 8, fp16).expect("decode op=12 resp");
            assert!((dec.infer_ms - 12.5).abs() < 1e-4);
            assert!((dec.active_frac - 0.18).abs() < 1e-4);
            assert_eq!(dec.horiz_faces[0].len(), 8);
            assert!((dec.horiz_faces[0][0].data[0] - 0.25).abs() < 1e-3);
            assert_eq!(dec.zflux_faces[5].len(), 7 * 5 * 16 * 16);
        }
    }
}
