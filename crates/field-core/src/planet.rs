//! Wave 29B: `field-core::planet` — Exact 6-Face Cubed-Sphere Planetary Physics Engine.
//!
//! Couples 6 `VolumeField` shells (`+X, -X, +Y, -Y, +Z, -Z`) into a closed, seamless,
//! curved spherical planet with:
//! - Right-handed orthonormal tangent frames `(n_f, e_u, e_v)` and exact `face_uv <-> sphere_dir`
//!   bijection.
//! - Basis-derived **12-edge boundary neighbor mapping** with exact index remapping and exact
//!   proper 2×2 signed permutation matrices (`det == +1`) rotating tangent flow vectors `(flow_e, flow_s)`.
//! - **Bit-exact global integer conservation** (`global_mass_error() == 0 && global_energy_error() == 0`)
//!   across all 6 faces, all 12 rotated cube seams, sweeping solar terminator forcing, and
//!   `BaseAnchorMode::RadialInnerShell` orbital Sky-Ring collapse!
//! - Seamless 3D spherical procedural planet generator (molten inner core, mantle caverns,
//!   continents, sapphire oceans, polar glacier ice caps, equatorial volcanic rift caldera,
//!   and an Orbital Floating Sky-Island Ring anchored by destructible `BuoyantCore`s).
//! - Continuous 6-DOF geodesic `SphericalEntity` locomotion across all 12 cube seams & both poles.

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::structural::{
    BaseAnchorMode, BuoyantCore, StructuralState, StructuralStepStats, fracture_buoyant_core,
};
use crate::{
    Field, H_CAP, MOBILE_SPECIES, NUM_MOBILE, NUM_SPECIES, SPECIES, SP_ICE, SP_OIL, SP_PLANT,
    SP_ROCK, SP_SMOKE, SP_SOIL, SP_STEAM, SP_WATER, SP_WOOD, T_AMB, T_MAX, T_MIN, VolumeField,
    WORLD_LAW_STANDARD, WorldEvolutionStats, WorldLaws,
};

pub type FaceId = usize;
pub type EdgeId = usize;

pub const FACE_POS_X: FaceId = 0;
pub const FACE_NEG_X: FaceId = 1;
pub const FACE_POS_Y: FaceId = 2;
pub const FACE_NEG_Y: FaceId = 3;
pub const FACE_POS_Z: FaceId = 4;
pub const FACE_NEG_Z: FaceId = 5;
pub const FACE_COUNT: usize = 6;

/// `+u` edge (`x = n - 1`, East on face local grid).
pub const EDGE_POS_U: EdgeId = 0;
/// `-u` edge (`x = 0`, West on face local grid).
pub const EDGE_NEG_U: EdgeId = 1;
/// `+v` edge (`y = n - 1`, South on face local grid).
pub const EDGE_POS_V: EdgeId = 2;
/// `-v` edge (`y = 0`, North on face local grid).
pub const EDGE_NEG_V: EdgeId = 3;
pub const EDGE_COUNT: usize = 4;

/// Right-handed orthonormal 3D tangent frame `(n, e_u, e_v)` for one cube face:
/// `cross(e_u, e_v) == n` and `det([n, e_u, e_v]) == +1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceFrame {
    pub n: [i8; 3],
    pub e_u: [i8; 3],
    pub e_v: [i8; 3],
}

pub const FACE_FRAMES: [FaceFrame; FACE_COUNT] = [
    // 0: +X (equatorial)
    FaceFrame { n: [1, 0, 0], e_u: [0, 0, -1], e_v: [0, 1, 0] },
    // 1: -X (equatorial)
    FaceFrame { n: [-1, 0, 0], e_u: [0, 0, 1], e_v: [0, 1, 0] },
    // 2: +Y (North Pole)
    FaceFrame { n: [0, 1, 0], e_u: [1, 0, 0], e_v: [0, 0, -1] },
    // 3: -Y (South Pole)
    FaceFrame { n: [0, -1, 0], e_u: [1, 0, 0], e_v: [0, 0, 1] },
    // 4: +Z (equatorial)
    FaceFrame { n: [0, 0, 1], e_u: [1, 0, 0], e_v: [0, 1, 0] },
    // 5: -Z (equatorial)
    FaceFrame { n: [0, 0, -1], e_u: [-1, 0, 0], e_v: [0, 1, 0] },
];

pub const fn face_name(face: FaceId) -> &'static str {
    match face {
        FACE_POS_X => "+X",
        FACE_NEG_X => "-X",
        FACE_POS_Y => "+Y",
        FACE_NEG_Y => "-Y",
        FACE_POS_Z => "+Z",
        FACE_NEG_Z => "-Z",
        _ => "?",
    }
}

#[inline]
const fn dot_i8(a: [i8; 3], b: [i8; 3]) -> i8 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
const fn neg_i8(a: [i8; 3]) -> [i8; 3] {
    [-a[0], -a[1], -a[2]]
}

#[inline]
pub const fn edge_outward_uv(edge: EdgeId) -> [i8; 2] {
    match edge {
        EDGE_POS_U => [1, 0],
        EDGE_NEG_U => [-1, 0],
        EDGE_POS_V => [0, 1],
        EDGE_NEG_V => [0, -1],
        _ => [0, 0],
    }
}

#[inline]
pub const fn edge_along_uv(edge: EdgeId) -> [i8; 2] {
    match edge {
        EDGE_POS_U | EDGE_NEG_U => [0, 1],
        EDGE_POS_V | EDGE_NEG_V => [1, 0],
        _ => [0, 0],
    }
}

#[inline]
const fn edge_outward_3d(frame: FaceFrame, edge: EdgeId) -> [i8; 3] {
    match edge {
        EDGE_POS_U => frame.e_u,
        EDGE_NEG_U => neg_i8(frame.e_u),
        EDGE_POS_V => frame.e_v,
        EDGE_NEG_V => neg_i8(frame.e_v),
        _ => [0, 0, 0],
    }
}

#[inline]
const fn edge_along_3d(frame: FaceFrame, edge: EdgeId) -> [i8; 3] {
    match edge {
        EDGE_POS_U | EDGE_NEG_U => frame.e_v,
        EDGE_POS_V | EDGE_NEG_V => frame.e_u,
        _ => [0, 0, 0],
    }
}

/// Exact topological and tangent-frame link across one cube edge `(face_a, edge_a) -> (face, edge)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeLink {
    /// Neighbor cube face (`0..6`).
    pub face: FaceId,
    /// Matching edge on the neighbor face (`0..4`).
    pub edge: EdgeId,
    /// True if the along-edge coordinate `k ∈ 0..n` reverses (`k' = n - 1 - k`) across the seam.
    pub flip_along: bool,
    /// Exact 2×2 signed permutation matrix (`det == +1`) rotating a tangent vector `(v_u, v_v)`
    /// from source face `(e_u, e_v)` across the shared edge into neighbor face `(e_u', e_v')`:
    /// `v_u' = rot[0][0] * v_u + rot[0][1] * v_v`,
    /// `v_v' = rot[1][0] * v_u + rot[1][1] * v_v`.
    pub rot: [[i8; 2]; 2],
}

impl EdgeLink {
    /// Rotate a floating-point tangent vector `(vu, vv)` into the neighbor face's `(e_u, e_v)` basis.
    #[inline]
    pub fn rotate_vec_f32(&self, vu: f32, vv: f32) -> (f32, f32) {
        (
            self.rot[0][0] as f32 * vu + self.rot[0][1] as f32 * vv,
            self.rot[1][0] as f32 * vu + self.rot[1][1] as f32 * vv,
        )
    }

    /// Rotate an integer tangent flow `(flow_e, flow_s)` into the neighbor face's `(e_u, e_v)` basis.
    #[inline]
    pub fn rotate_flow_i16(&self, flow_e: i16, flow_s: i16) -> (i16, i16) {
        let fe = flow_e as i32;
        let fs = flow_s as i32;
        let nfe = (self.rot[0][0] as i32 * fe + self.rot[0][1] as i32 * fs)
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        let nfs = (self.rot[1][0] as i32 * fe + self.rot[1][1] as i32 * fs)
            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        (nfe, nfs)
    }

    /// Map a boundary `(u, v)` coordinate on `src_edge` (`u, v ∈ [-1, 1]`) to the exact
    /// coinciding `(u', v')` on `(self.face, self.edge)`.
    #[inline]
    pub fn map_uv(&self, src_edge: EdgeId, u: f32, v: f32) -> (f32, f32) {
        let t_a = match src_edge {
            EDGE_POS_U | EDGE_NEG_U => v,
            EDGE_POS_V | EDGE_NEG_V => u,
            _ => 0.0,
        };
        let t_b = if self.flip_along { -t_a } else { t_a };
        match self.edge {
            EDGE_POS_U => (1.0, t_b),
            EDGE_NEG_U => (-1.0, t_b),
            EDGE_POS_V => (t_b, 1.0),
            EDGE_NEG_V => (t_b, -1.0),
            _ => (0.0, 0.0),
        }
    }

    /// Map along-edge cell index `k ∈ 0..n` on source edge to `(x_b, y_b)` on neighbor face.
    #[inline]
    pub fn map_edge_cell(&self, k: usize, n: usize) -> (usize, usize) {
        let kb = if self.flip_along { n - 1 - k } else { k };
        edge_cell_coords(self.edge, kb, n)
    }

    /// Determinant of `self.rot` (always `+1` for right-handed cubed-sphere faces).
    #[inline]
    pub const fn det(&self) -> i8 {
        self.rot[0][0] * self.rot[1][1] - self.rot[0][1] * self.rot[1][0]
    }
}

/// Cell `(x, y)` at index `k ∈ 0..n` along `edge` of an `n × n` face.
#[inline]
pub const fn edge_cell_coords(edge: EdgeId, k: usize, n: usize) -> (usize, usize) {
    match edge {
        EDGE_POS_U => (n - 1, k),
        EDGE_NEG_U => (0, k),
        EDGE_POS_V => (k, n - 1),
        EDGE_NEG_V => (k, 0),
        _ => (0, 0),
    }
}

/// Compute the exact neighbor face, matching edge, along-edge flip flag, and 2×2 signed
/// permutation matrix directly from the 3D right-handed face frames `(n_f, e_u, e_v)`.
pub const fn edge_neighbor(face: FaceId, edge: EdgeId) -> EdgeLink {
    let fa = FACE_FRAMES[face];
    let d_out_a = edge_outward_3d(fa, edge);
    let t_a = edge_along_3d(fa, edge);

    // Neighbor face B is the unique face whose normal equals d_out_a.
    let mut face_b = 0usize;
    let mut i = 0usize;
    while i < FACE_COUNT {
        if dot_i8(FACE_FRAMES[i].n, d_out_a) == 1 {
            face_b = i;
            break;
        }
        i += 1;
    }
    let fb = FACE_FRAMES[face_b];

    // On face B, the shared edge lies along outward direction d_out_b == fa.n.
    let edge_b = if dot_i8(fa.n, fb.e_u) == 1 {
        EDGE_POS_U
    } else if dot_i8(fa.n, fb.e_u) == -1 {
        EDGE_NEG_U
    } else if dot_i8(fa.n, fb.e_v) == 1 {
        EDGE_POS_V
    } else {
        EDGE_NEG_V
    };

    let t_b = edge_along_3d(fb, edge_b);
    let s_along = dot_i8(t_a, t_b); // +1 or -1
    let flip_along = s_along < 0;

    // Parallel transport across the 90° fold:
    //   d_out_a  -> -d_out_b   (moving off A enters inward on B)
    //   t_a      -> s_along * t_b
    let out_uv_a = edge_outward_uv(edge);
    let along_uv_a = edge_along_uv(edge);
    let out_uv_b = edge_outward_uv(edge_b);
    let along_uv_b = edge_along_uv(edge_b);
    let in_uv_b = [-out_uv_b[0], -out_uv_b[1]];

    // Image of basis vector (1, 0) on face A:
    let c_across_u = out_uv_a[0];
    let c_along_u = along_uv_a[0];
    let r00 = c_across_u * in_uv_b[0] + s_along * c_along_u * along_uv_b[0];
    let r10 = c_across_u * in_uv_b[1] + s_along * c_along_u * along_uv_b[1];

    // Image of basis vector (0, 1) on face A:
    let c_across_v = out_uv_a[1];
    let c_along_v = along_uv_a[1];
    let r01 = c_across_v * in_uv_b[0] + s_along * c_along_v * along_uv_b[0];
    let r11 = c_across_v * in_uv_b[1] + s_along * c_along_v * along_uv_b[1];

    EdgeLink {
        face: face_b,
        edge: edge_b,
        flip_along,
        rot: [[r00, r01], [r10, r11]],
    }
}

/// Enumerate all 12 canonical undirected cube edges `(face_a, edge_a, link_to_b)` with `face_a < link_to_b.face`.
pub const fn canonical_12_edges() -> [(FaceId, EdgeId, EdgeLink); 12] {
    let dummy = EdgeLink {
        face: 0,
        edge: 0,
        flip_along: false,
        rot: [[1, 0], [0, 1]],
    };
    let mut out = [(0usize, 0usize, dummy); 12];
    let mut count = 0usize;
    let mut f = 0usize;
    while f < FACE_COUNT {
        let mut e = 0usize;
        while e < EDGE_COUNT {
            let link = edge_neighbor(f, e);
            if f < link.face {
                out[count] = (f, e, link);
                count += 1;
            }
            e += 1;
        }
        f += 1;
    }
    out
}

/// Project cubed-sphere face coordinates `(face, u, v)` (`u, v ∈ [-1, 1]`) onto the unit sphere.
#[inline]
pub fn face_uv_to_sphere_dir(face: FaceId, u: f32, v: f32) -> [f32; 3] {
    let fr = FACE_FRAMES[face % FACE_COUNT];
    let px = fr.n[0] as f32 + u * fr.e_u[0] as f32 + v * fr.e_v[0] as f32;
    let py = fr.n[1] as f32 + u * fr.e_u[1] as f32 + v * fr.e_v[1] as f32;
    let pz = fr.n[2] as f32 + u * fr.e_u[2] as f32 + v * fr.e_v[2] as f32;
    let inv_len = 1.0 / (px * px + py * py + pz * pz).sqrt().max(1e-12);
    [px * inv_len, py * inv_len, pz * inv_len]
}

/// Exact inverse projection from any non-zero 3D direction `dir` to `(face, u, v)` (`u, v ∈ [-1, 1]`).
#[inline]
pub fn sphere_dir_to_face_uv(dir: [f32; 3]) -> (FaceId, f32, f32) {
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();

    let (face, denom) = if ax >= ay && ax >= az {
        if dir[0] >= 0.0 {
            (FACE_POS_X, dir[0])
        } else {
            (FACE_NEG_X, -dir[0])
        }
    } else if ay >= ax && ay >= az {
        if dir[1] >= 0.0 {
            (FACE_POS_Y, dir[1])
        } else {
            (FACE_NEG_Y, -dir[1])
        }
    } else if dir[2] >= 0.0 {
        (FACE_POS_Z, dir[2])
    } else {
        (FACE_NEG_Z, -dir[2])
    };

    let inv_d = 1.0 / denom.max(1e-12);
    let fr = FACE_FRAMES[face];
    let u = ((dir[0] * fr.e_u[0] as f32 + dir[1] * fr.e_u[1] as f32 + dir[2] * fr.e_u[2] as f32)
        * inv_d)
        .clamp(-1.0, 1.0);
    let v = ((dir[0] * fr.e_v[0] as f32 + dir[1] * fr.e_v[1] as f32 + dir[2] * fr.e_v[2] as f32)
        * inv_d)
        .clamp(-1.0, 1.0);
    (face, u, v)
}

/// Continuous world position `(r_core + z_f) * face_uv_to_sphere_dir(face, u, v)`.
#[inline]
pub fn face_uv_z_to_world_pos(face: FaceId, u: f32, v: f32, z_f: f32, r_core: f32) -> [f32; 3] {
    let dir = face_uv_to_sphere_dir(face, u, v);
    let r = r_core + z_f;
    [r * dir[0], r * dir[1], r * dir[2]]
}

/// Project integer voxel cell center `(face, x, y, z)` into 3D spherical world coordinates:
/// `(r_core + z + 0.5) * face_uv_to_sphere_dir(face, u, v)`.
#[inline]
pub fn face_cell_to_world_pos(
    face: FaceId,
    x: usize,
    y: usize,
    z: usize,
    n: usize,
    r_core: f32,
) -> [f32; 3] {
    let nf = n.max(1) as f32;
    let u = ((x as f32 + 0.5) / nf) * 2.0 - 1.0;
    let v = ((y as f32 + 0.5) / nf) * 2.0 - 1.0;
    face_uv_z_to_world_pos(face, u, v, z as f32 + 0.5, r_core)
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1e-8 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// =============================================================================
// Seamless 3D Spherical Procedural Noise (evaluated purely in 3D sphere space!)
// =============================================================================

#[inline]
fn hash_u64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[inline]
fn hash_grid_3d(ix: i32, iy: i32, iz: i32, seed: u64) -> f32 {
    let h = hash_u64(
        (ix as i64 as u64).wrapping_mul(73856093)
            ^ (iy as i64 as u64).wrapping_mul(19349663)
            ^ (iz as i64 as u64).wrapping_mul(83492791)
            ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    ((h >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise_3d(p: [f32; 3], seed: u64) -> f32 {
    let ix = p[0].floor() as i32;
    let iy = p[1].floor() as i32;
    let iz = p[2].floor() as i32;
    let fx = smoothstep(p[0] - ix as f32);
    let fy = smoothstep(p[1] - iy as f32);
    let fz = smoothstep(p[2] - iz as f32);

    let c000 = hash_grid_3d(ix, iy, iz, seed);
    let c100 = hash_grid_3d(ix + 1, iy, iz, seed);
    let c010 = hash_grid_3d(ix, iy + 1, iz, seed);
    let c110 = hash_grid_3d(ix + 1, iy + 1, iz, seed);
    let c001 = hash_grid_3d(ix, iy, iz + 1, seed);
    let c101 = hash_grid_3d(ix + 1, iy, iz + 1, seed);
    let c011 = hash_grid_3d(ix, iy + 1, iz + 1, seed);
    let c111 = hash_grid_3d(ix + 1, iy + 1, iz + 1, seed);

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;
    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;
    y0 + (y1 - y0) * fz
}

fn fbm_3d(p: [f32; 3], seed: u64, octaves: u32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f32;
    let mut norm = 0.0f32;
    for oct in 0..octaves {
        let q = [p[0] * freq + 17.3 * oct as f32, p[1] * freq - 11.7 * oct as f32, p[2] * freq + 23.1 * oct as f32];
        sum += amp * value_noise_3d(q, seed.wrapping_add(oct as u64 * 1013));
        norm += amp;
        amp *= 0.5;
        freq *= 2.04;
    }
    if norm > 0.0 { sum / norm } else { 0.5 }
}

/// Construct an empty `VolumeField` configured with `BaseAnchorMode::RadialInnerShell`.
fn new_empty_radial_volume(face_size: usize, depth: usize, seed: u64) -> VolumeField {
    let d = depth.max(4);
    let layers: Vec<Field> = (0..d).map(|_| Field::new_empty(face_size, face_size)).collect();
    let vlen = d * face_size * face_size;
    let mut structural = StructuralState::default();
    structural.anchor_mode = BaseAnchorMode::RadialInnerShell;
    let mut vf = VolumeField {
        width: face_size,
        height: face_size,
        depth: d,
        biome_id: 0,
        seed,
        layers,
        prev_temp: vec![T_AMB as i32; vlen],
        prev_steam: vec![0i32; vlen],
        initial_total_mass: 0,
        initial_total_energy: 0,
        tick: 0,
        world_time_secs: 0.0,
        evolution_stats: WorldEvolutionStats::default(),
        world_laws: WorldLaws::default(),
        world_law_preset: WORLD_LAW_STANDARD,
        structural,
    };
    vf.snapshot_conservation_baseline();
    vf
}

/// Telemetry returned by one full `CubedSpherePlanet::step_planet` tick.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlanetStepStats {
    pub tick: u64,
    pub global_mass_error: i64,
    pub global_energy_error: i64,
    pub seam_mass_crossed: i64,
    pub seam_energy_crossed: i64,
    pub fallen_solid_mass: i64,
    pub splashed_fluid_mass: i64,
    pub step_ms: f64,
}

/// Summary of 12-edge cross-face conservative flux exchange.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeExchangeStats {
    pub seam_mass_crossed: i64,
    pub seam_energy_crossed: i64,
    pub active_seam_cells: usize,
}

/// Continuous 6-DOF spherical surface entity traversing all 6 cubed-sphere faces seamlessly.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphericalEntity {
    pub face: FaceId,
    pub u_cell: f32,
    pub v_cell: f32,
    pub z: f32,
    pub vu: f32,
    pub vv: f32,
}

impl SphericalEntity {
    /// Current 3D world position on the curved planet shell.
    pub fn world_pos(&self, face_size: usize, r_core: f32) -> [f32; 3] {
        let nf = face_size.max(1) as f32;
        let u = (self.u_cell / nf) * 2.0 - 1.0;
        let v = (self.v_cell / nf) * 2.0 - 1.0;
        face_uv_z_to_world_pos(self.face, u, v, self.z, r_core)
    }

    /// Current unit sphere direction.
    pub fn sphere_dir(&self, face_size: usize) -> [f32; 3] {
        let nf = face_size.max(1) as f32;
        let u = (self.u_cell / nf) * 2.0 - 1.0;
        let v = (self.v_cell / nf) * 2.0 - 1.0;
        face_uv_to_sphere_dir(self.face, u, v)
    }
}

/// Step a `SphericalEntity` along its tangent velocity `(vu, vv)`, smoothly crossing any of
/// the 12 cube edges with exact index remapping and 2×2 tangent velocity rotation!
/// Returns the number of cube edges crossed during this step.
pub fn step_spherical_entity(ent: &mut SphericalEntity, face_size: usize, dt: f32) -> u32 {
    let n = face_size.max(1) as f32;
    ent.u_cell += ent.vu * dt;
    ent.v_cell += ent.vv * dt;

    let mut crossings = 0u32;
    for _ in 0..4 {
        let (edge, overshoot, along) = if ent.u_cell >= n {
            (EDGE_POS_U, ent.u_cell - n, ent.v_cell.clamp(0.0, n - 1e-4))
        } else if ent.u_cell < 0.0 {
            (EDGE_NEG_U, -ent.u_cell, ent.v_cell.clamp(0.0, n - 1e-4))
        } else if ent.v_cell >= n {
            (EDGE_POS_V, ent.v_cell - n, ent.u_cell.clamp(0.0, n - 1e-4))
        } else if ent.v_cell < 0.0 {
            (EDGE_NEG_V, -ent.v_cell, ent.u_cell.clamp(0.0, n - 1e-4))
        } else {
            break;
        };

        let link = edge_neighbor(ent.face, edge);
        let new_along = if link.flip_along {
            (n - along).clamp(0.0, n - 1e-4)
        } else {
            along.clamp(0.0, n - 1e-4)
        };
        let inward = overshoot.clamp(1e-5, n - 1e-4);

        match link.edge {
            EDGE_POS_U => {
                ent.u_cell = n - inward;
                ent.v_cell = new_along;
            }
            EDGE_NEG_U => {
                ent.u_cell = inward;
                ent.v_cell = new_along;
            }
            EDGE_POS_V => {
                ent.u_cell = new_along;
                ent.v_cell = n - inward;
            }
            EDGE_NEG_V => {
                ent.u_cell = new_along;
                ent.v_cell = inward;
            }
            _ => {}
        }

        let (nvu, nvv) = link.rotate_vec_f32(ent.vu, ent.vv);
        ent.vu = nvu;
        ent.vv = nvv;
        ent.face = link.face;
        crossings += 1;
    }

    ent.u_cell = ent.u_cell.clamp(0.0, n - 1e-4);
    ent.v_cell = ent.v_cell.clamp(0.0, n - 1e-4);
    crossings
}

/// Live 6-Face Cubed-Sphere Planet coupling 6 `VolumeField` shells with exact global
/// integer conservation (`global_mass_error() == 0 && global_energy_error() == 0`).
#[derive(Clone, Debug)]
pub struct CubedSpherePlanet {
    pub faces: [VolumeField; FACE_COUNT],
    pub face_size: usize,
    pub depth: usize,
    pub r_core: f32,
    pub sun_dir: [f32; 3],
    pub tick: u64,
    pub initial_global_mass: i64,
    pub initial_global_energy: i64,
    pub total_seam_mass_crossed: i64,
    pub total_seam_energy_crossed: i64,
}

impl CubedSpherePlanet {
    /// Construct an empty 6-face planet (all faces initialized with `BaseAnchorMode::RadialInnerShell`).
    pub fn new_empty(face_size: usize, depth: usize, r_core: f32, seed: u64) -> Self {
        let faces: [VolumeField; FACE_COUNT] = std::array::from_fn(|f| {
            new_empty_radial_volume(face_size, depth, seed.wrapping_add(f as u64 * 7919))
        });
        let mut planet = Self {
            faces,
            face_size,
            depth: depth.max(4),
            r_core,
            sun_dir: normalize3([0.78, 0.24, 0.58]),
            tick: 0,
            initial_global_mass: 0,
            initial_global_energy: 0,
            total_seam_mass_crossed: 0,
            total_seam_energy_crossed: 0,
        };
        planet.snapshot_conservation_baseline();
        planet
    }

    /// Generate a complete, seamless 3D spherical planet across all 6 cubed-sphere faces:
    /// - Inner pressurized molten core crust (`z = 0..=3`, glowing hot rock & oil veins)
    /// - Subterranean mantle caverns (`z = 4..10`, rock pillars + cavern voids)
    /// - Seamless emerald/alpine continents (`z = 10..20`) & deep sapphire curved oceans (`z = 11..15`)
    /// - Polar glacier ice caps (`|dir.y| > 0.74`)
    /// - Equatorial volcanic rift caldera (`|dir.y| < 0.20` along one longitude belt, glowing lava/oil + rising steam)
    /// - Orbital Floating Sky-Island Ring (`z = depth - 10 .. depth - 5` near equator, anchored by registered `BuoyantCore`s)
    pub fn new(face_size: usize, depth: usize, r_core: f32, seed: u64) -> Self {
        let d = depth.max(16);
        let mut planet = Self::new_empty(face_size, d, r_core, seed);
        let nf = face_size as f32;
        let z_scale = d as f32 / 32.0;

        let z_core_hi = ((3.0 * z_scale).round() as usize).clamp(1, d - 6);
        let z_cav_lo = (z_core_hi + 1).min(d - 5);
        let z_cav_hi = ((10.0 * z_scale).round() as usize).clamp(z_cav_lo, d - 5);
        let z_sea = ((15.0 * z_scale).round() as usize).clamp(z_cav_hi + 1, d - 4);
        let z_peak_max = ((20.0 * z_scale).round() as usize).clamp(z_sea + 2, d - 4);

        // 1. Populate seamless 3D planetary body across all 6 faces purely from 3D `dir`
        for f in 0..FACE_COUNT {
            let vf = &mut planet.faces[f];
            vf.structural.anchor_mode = BaseAnchorMode::RadialInnerShell;

            for y in 0..face_size {
                let v = ((y as f32 + 0.5) / nf) * 2.0 - 1.0;
                for x in 0..face_size {
                    let u = ((x as f32 + 0.5) / nf) * 2.0 - 1.0;
                    let idx = y * face_size + x;
                    let dir = face_uv_to_sphere_dir(f, u, v);

                    // Seamless 3D spherical coordinates for terrain, biome & rift evaluation
                    let lon = dir[2].atan2(dir[0]); // [-pi, pi]
                    let lat_abs = dir[1].abs();

                    // Large-scale continents + tectonic mountain ridges (strictly 3D spherical!)
                    let n_cont = fbm_3d([dir[0] * 2.15, dir[1] * 2.15, dir[2] * 2.15], seed, 4);
                    let n_ridge_raw = fbm_3d([dir[0] * 5.2, dir[1] * 5.2, dir[2] * 5.2], seed ^ 0xA55A, 3);
                    let n_ridge = 1.0 - (n_ridge_raw * 2.0 - 1.0).abs();
                    let n_detail = fbm_3d([dir[0] * 11.0, dir[1] * 11.0, dir[2] * 11.0], seed ^ 0x5AA5, 2);

                    // Equatorial volcanic rift belt (`|dir.y| < 0.20` along longitude belt `lon ≈ -0.65..+0.55`,
                    // centered at `lon ≈ -0.15` so it catches the dusk terminator and blazes on the night side!)
                    let rift_curve = 0.065 * (lon * 3.8 + 0.4).sin() + 0.025 * (n_detail - 0.5);
                    let d_lat_rift = (dir[1] - rift_curve).abs();
                    // Wrapped angular distance from longitude center `lon = -0.25`
                    let mut d_lon = lon - (-0.25);
                    while d_lon > std::f32::consts::PI {
                        d_lon -= std::f32::consts::TAU;
                    }
                    while d_lon < -std::f32::consts::PI {
                        d_lon += std::f32::consts::TAU;
                    }
                    let in_rift_belt = d_lon.abs() < 0.85 && d_lat_rift < 0.19;
                    let rift_core = in_rift_belt && d_lat_rift < 0.068 && d_lon.abs() < 0.72;
                    let rift_rim = in_rift_belt && !rift_core && d_lat_rift < 0.145 && d_lon.abs() < 0.82;

                    // Polar glacier ice caps (`|dir.y| > 0.74`, slightly organic coastline)
                    let polar_thresh = 0.74 + 0.045 * (n_detail - 0.5);
                    let is_polar = lat_abs > polar_thresh;

                    // Compute smooth surface height in `z = 10..20` (~52% sapphire oceans, ~48% continents)
                    let mut h_norm = (n_cont - 0.515) * 2.55
                        + (n_ridge * n_ridge - 0.40) * 0.72
                        + (n_detail - 0.5) * 0.24;
                    if is_polar {
                        h_norm = h_norm.max(0.22) + (lat_abs - polar_thresh) * 1.45;
                    }
                    if rift_rim {
                        h_norm = h_norm.max(0.62) + 0.34;
                    } else if rift_core {
                        h_norm = 0.28; // Glowing volcanic rift channel cleanly above sea level!
                    }

                    let surf_z_f = (14.8 + h_norm * 5.6) * z_scale;
                    let surf_z = (surf_z_f.round() as usize).clamp(z_cav_hi + 1, z_peak_max);
                    let is_land = surf_z > z_sea || is_polar || rift_core;

                    // Fill vertical column `z = 0..=z_peak_max`
                    for z in 0..=z_peak_max {
                        let layer = &mut vf.layers[z];
                        layer.elevation[idx] = (z.min(31)) as u8;

                        if z <= z_core_hi {
                            // Inner pressurized glowing molten core crust (`z = 0..=3`)
                            layer.mass[SP_ROCK][idx] = 1050;
                            if z >= z_core_hi.saturating_sub(1) && (in_rift_belt || n_detail > 0.68) {
                                layer.mass[SP_OIL][idx] = 280;
                            }
                            let core_t = if z <= 1 {
                                78_000
                            } else if in_rift_belt {
                                68_000
                            } else {
                                38_000
                            };
                            layer.set_cell_temp(idx, core_t);
                        } else if z <= z_cav_hi {
                            // Subterranean mantle & caverns (`z = 4..10`)
                            // Keep solid vertical structural pillars so continents are 100% supported from RadialInnerShell!
                            let cav_p = [
                                dir[0] * 7.5 + z as f32 * 0.28,
                                dir[1] * 7.5 - z as f32 * 0.22,
                                dir[2] * 7.5 + z as f32 * 0.19,
                            ];
                            let n_cav = value_noise_3d(cav_p, seed ^ 0xCAFE);
                            let in_cavern = z > z_cav_lo && z + 1 < z_cav_hi && n_cav > 0.66;
                            if !in_cavern {
                                layer.mass[SP_ROCK][idx] = 1000;
                                let t_mantle = if rift_core { 58_000 } else { 31_000 };
                                layer.set_cell_temp(idx, t_mantle);
                            } else if z <= z_cav_lo + 2 && !in_rift_belt {
                                // Subterranean aquifer pockets
                                layer.mass[SP_WATER][idx] = 650;
                                layer.set_cell_temp(idx, 29_500);
                            }
                        } else if rift_core {
                            // Equatorial volcanic rift caldera: incandescent lava/oil river cleanly above sea level + rising steam!
                            let z_lava = (z_sea + 1).min(z_peak_max);
                            if z < z_lava {
                                layer.mass[SP_ROCK][idx] = 980;
                                layer.set_cell_temp(idx, 96_000);
                            } else if z == z_lava {
                                // Molten caldera floor with incandescent rock + volatile oil
                                layer.mass[SP_ROCK][idx] = 680;
                                layer.mass[SP_OIL][idx] = 520;
                                layer.set_cell_temp(idx, 122_000);
                            } else if z <= z_lava + 3 {
                                // Rising volcanic steam & ember smoke plume
                                let fade = (z - z_lava) as u16;
                                layer.mass[SP_STEAM][idx] = 320u16.saturating_sub(fade * 70);
                                layer.mass[SP_SMOKE][idx] = 220u16.saturating_sub(fade * 45);
                                layer.set_cell_temp(idx, 42_000);
                            }
                        } else if is_polar {
                            // Polar glacier ice caps (`|dir.y| > 0.74`)
                            if z + 2 < surf_z {
                                layer.mass[SP_ROCK][idx] = 980;
                                layer.set_cell_temp(idx, 26_600);
                            } else if z <= surf_z {
                                layer.mass[SP_ICE][idx] = 980;
                                if z == surf_z {
                                    layer.mass[SP_ROCK][idx] = 120;
                                }
                                layer.set_cell_temp(idx, 25_800);
                            }
                        } else if is_land {
                            // Emerald / Alpine continents (`z = 10..20`) & dark basalt rift rims
                            if z < surf_z {
                                layer.mass[SP_ROCK][idx] = 980;
                                let t_sub = if rift_rim { 46_000 } else { 28_900 };
                                layer.set_cell_temp(idx, t_sub);
                            } else if z == surf_z {
                                if rift_rim {
                                    layer.mass[SP_ROCK][idx] = 960;
                                    layer.set_cell_temp(idx, 54_000);
                                } else if surf_z >= z_peak_max.saturating_sub(1) {
                                    // High alpine snow/ice ridge
                                    layer.mass[SP_ROCK][idx] = 680;
                                    layer.mass[SP_ICE][idx] = 440;
                                    layer.set_cell_temp(idx, 26_700);
                                } else {
                                    // Lush emerald continent surface
                                    layer.mass[SP_SOIL][idx] = 860;
                                    layer.mass[SP_PLANT][idx] = 420;
                                    if n_detail > 0.56 && surf_z <= z_sea + 3 {
                                        layer.mass[SP_WOOD][idx] = 520;
                                    }
                                    layer.set_cell_temp(idx, 29_100);
                                }
                            }
                        } else {
                            // Deep sapphire curved ocean (`z = 11..15`)
                            if z <= surf_z {
                                layer.mass[SP_ROCK][idx] = 950;
                                layer.mass[SP_SOIL][idx] = 180;
                                layer.set_cell_temp(idx, 28_600);
                            } else if z <= z_sea {
                                layer.mass[SP_WATER][idx] = 1020;
                                // Gentle zonal equatorial current
                                layer.flow_e[idx] = 35;
                                layer.set_cell_temp(idx, 28_700);
                            }
                        }
                    }
                }
            }
        }

        // 2. Build the Orbital Floating Sky-Island Ring (`z = depth - 10 .. depth - 5` near equator)
        //    on the 4 equatorial faces (`+X, -X, +Z, -Z`), anchored by registered `BuoyantCore`s!
        let z_ring_lo = d.saturating_sub(10).max(z_peak_max + 2);
        let z_ring_hi = d.saturating_sub(5).max(z_ring_lo + 2).min(d - 2);
        // Embed Buoyant Crystal Core inside the lower tapered keel (`z_ring_lo + 1`) so emerald top plateau stays natural!
        let z_core_keel = (z_ring_lo as f32 + 1.1).min(z_ring_hi as f32 - 0.8);

        let eq_faces = [FACE_POS_X, FACE_NEG_X, FACE_POS_Z, FACE_NEG_Z];
        let mut next_core_id = 0u8;
        let island_rad = (nf * 0.082).clamp(2.8, 5.2);
        let core_rad = (island_rad * 0.46).clamp(1.7, 2.5);

        for &f in &eq_faces {
            // Two sculpted orbital islands per equatorial face -> 8 Sky-Ring islands circling the equator!
            let placements: [(f32, f32, &str); 2] = [
                (0.27, 0.45, "SKY-RING ALPHA"),
                (0.73, 0.55, "SKY-RING OMEGA"),
            ];
            let vf = &mut planet.faces[f];
            for (pi, (pu, pv, tag)) in placements.into_iter().enumerate() {
                let cx = pu * nf;
                let cy = pv * nf;
                if cx < island_rad + 3.0 || cx > nf - island_rad - 3.0 || cy < island_rad + 3.0 || cy > nf - island_rad - 3.0 {
                    continue;
                }
                let r_i = (island_rad * 1.25).ceil() as i32;
                let x0 = ((cx.floor() as i32) - r_i).max(2) as usize;
                let x1 = ((cx.floor() as i32) + r_i).min(face_size as i32 - 3) as usize;
                let y0 = ((cy.floor() as i32) - r_i).max(2) as usize;
                let y1 = ((cy.floor() as i32) + r_i).min(face_size as i32 - 3) as usize;
                let phase = (f as f32) * 1.7 + (pi as f32) * 2.4;

                let mut core_mass = 0i64;
                for z in z_ring_lo..=z_ring_hi {
                    let t_z = if z_ring_hi > z_ring_lo {
                        (z - z_ring_lo) as f32 / (z_ring_hi - z_ring_lo) as f32
                    } else {
                        0.5
                    };
                    // Sculpted tapered floating island silhouette: crystal spire keel -> wide organic plateau
                    let base_layer_r = island_rad * (0.28 + 0.78 * t_z.powf(0.58));
                    let dz = (z as f32 + 0.5) - z_core_keel;
                    let layer = &mut vf.layers[z];
                    for y in y0..=y1 {
                        let dy = (y as f32 + 0.5) - cy;
                        for x in x0..=x1 {
                            let dx = (x as f32 + 0.5) - cx;
                            let dist_xy = (dx * dx + dy * dy).sqrt();
                            let ang = dy.atan2(dx);
                            let organic = 1.0
                                + 0.16 * (ang * 3.0 + phase).sin()
                                + 0.10 * (ang * 5.0 - phase * 1.3).cos();
                            let layer_r = base_layer_r * organic;
                            let idx = y * face_size + x;
                            if dist_xy <= layer_r {
                                let in_core = z < z_ring_hi
                                    && (dx * dx + dy * dy + dz * dz <= core_rad * core_rad);
                                if in_core {
                                    // Cool levitating Buoyant Crystal Core (rock + crystal ice at 268 K)
                                    layer.mass[SP_ROCK][idx] = 740;
                                    layer.mass[SP_ICE][idx] = 360;
                                    layer.set_cell_temp(idx, 26_800);
                                    core_mass += 740 + 360;
                                } else if z == z_ring_hi {
                                    // Lush sky-meadow plateau top
                                    layer.mass[SP_ROCK][idx] = 520;
                                    layer.mass[SP_SOIL][idx] = 440;
                                    layer.mass[SP_PLANT][idx] = 400;
                                    if dist_xy < layer_r * 0.55 {
                                        layer.mass[SP_WOOD][idx] = 320;
                                    }
                                    layer.set_cell_temp(idx, 28_800);
                                } else {
                                    // Basalt-quartz floating hull
                                    layer.mass[SP_ROCK][idx] = 920;
                                    layer.set_cell_temp(idx, 27_800);
                                }
                            }
                        }
                    }
                }

                if core_mass > 0 {
                    let core = BuoyantCore {
                        id: next_core_id,
                        name: format!("{} {}", face_name(f), tag),
                        cx,
                        cy,
                        cz: z_core_keel,
                        radius: core_rad,
                        island_radius: island_rad * 1.2,
                        init_core_mass: core_mass,
                        current_core_mass: core_mass,
                        integrity: 1.0,
                        anchor_strength: 255,
                        collapsed: false,
                        collapse_ticks: 0,
                        announced_fracture: false,
                        announced_impact: false,
                    };
                    vf.structural.buoyant_cores.push(core);
                    next_core_id = next_core_id.wrapping_add(1);
                }
            }
            // Start quiescent; structural solver runs when a core is fractured or marked dirty
            vf.structural.dirty = false;
        }

        planet.snapshot_conservation_baseline();
        planet
    }

    /// Snapshot exact global mass & energy baseline across all 6 faces.
    pub fn snapshot_conservation_baseline(&mut self) {
        for f in &mut self.faces {
            f.snapshot_conservation_baseline();
            f.snapshot_prev_sensors();
        }
        self.initial_global_mass = self.current_global_mass();
        self.initial_global_energy = self.current_global_energy();
    }

    /// Total accounted integer mass across all 6 cubed-sphere faces.
    pub fn current_global_mass(&self) -> i64 {
        self.faces.iter().map(|f| f.current_accounted_mass()).sum()
    }

    /// Total accounted integer thermal + chemical energy across all 6 cubed-sphere faces.
    pub fn current_global_energy(&self) -> i64 {
        self.faces.iter().map(|f| f.current_accounted_energy()).sum()
    }

    /// Exact global mass conservation error across all 6 faces (MUST equal `0`).
    pub fn global_mass_error(&self) -> i64 {
        self.current_global_mass() - self.initial_global_mass
    }

    /// Exact global energy conservation error across all 6 faces (MUST equal `0`).
    pub fn global_energy_error(&self) -> i64 {
        self.current_global_energy() - self.initial_global_energy
    }

    /// Total number of active 3D voxels across all 6 faces (`6 * face_size * face_size * depth`).
    pub fn total_voxels(&self) -> usize {
        FACE_COUNT * self.face_size * self.face_size * self.depth
    }

    /// Total registered Buoyant Crystal Cores across all 6 faces.
    pub fn buoyant_core_count(&self) -> usize {
        self.faces.iter().map(|f| f.structural.buoyant_cores.len()).sum()
    }

    /// Conservatively fracture/demagnetize an orbital Sky-Island's `BuoyantCore` on `face`,
    /// causing the island to fracture rim-inward and crash radially inward (`z -> z - 1`) onto the
    /// curved planet surface with **exact `global_mass_error() == 0 && global_energy_error() == 0`**!
    pub fn fracture_orbital_island(
        &mut self,
        face: FaceId,
        core_id: u8,
    ) -> Option<(String, f32, f32, f32)> {
        if face >= FACE_COUNT {
            return None;
        }
        fracture_buoyant_core(&mut self.faces[face], core_id)
    }

    /// Exact antisymmetric conservative flux exchange across ALL 12 undirected cube edges
    /// and ALL radial layers `z = 0..depth`.
    ///
    /// - Conservatively exchanges mobile fluids (`SP_WATER`, `SP_OIL`) and gases (`SP_STEAM`, `SP_SMOKE`)
    ///   plus exact proportional advected thermal energy and 2nd-law heat conduction (`a -= q; b += q`).
    /// - Rotates tangent flow vectors `(flow_e, flow_s)` cleanly across each rotated cube edge using
    ///   its exact 2×2 signed permutation matrix `link.rot`.
    /// - Guarantees **`global_mass_error() == 0 && global_energy_error() == 0`** bit-exactly!
    pub fn exchange_12_edges_conservative(&mut self) -> EdgeExchangeStats {
        let edges = canonical_12_edges();
        let n = self.face_size;
        let d = self.depth;
        let mut stats = EdgeExchangeStats::default();

        for (face_a, edge_a, link_ab) in edges {
            let face_b = link_ab.face;
            let edge_b = link_ab.edge;
            let link_ba = edge_neighbor(face_b, edge_b);
            let out_a = edge_outward_uv(edge_a);
            let out_b = edge_outward_uv(edge_b);

            // face_a < face_b is guaranteed by canonical_12_edges()
            let (left, right) = self.faces.split_at_mut(face_b);
            let vf_a = &mut left[face_a];
            let vf_b = &mut right[0];

            for z in 0..d {
                let fa = &mut vf_a.layers[z];
                let fb = &mut vf_b.layers[z];

                for k in 0..n {
                    let (xa, ya) = edge_cell_coords(edge_a, k, n);
                    let (xb, yb) = link_ab.map_edge_cell(k, n);
                    let idx_a = ya * n + xa;
                    let idx_b = yb * n + xb;

                    let vol_a = fa.cell_volume(idx_a);
                    let vol_b = fb.cell_volume(idx_b);
                    if vol_a == 0 && vol_b == 0 {
                        continue;
                    }

                    let cap_a = fa.cell_heat_capacity(idx_a);
                    let cap_b = fb.cell_heat_capacity(idx_b);
                    let temp_a = fa.cell_temp_ck(idx_a);
                    let temp_b = fb.cell_temp_ck(idx_b);

                    let wall_a = fa.mass[SP_ROCK][idx_a] >= 500;
                    let wall_b = fb.mass[SP_ROCK][idx_b] >= 500;

                    // 1. Mobile fluid & gas transfer across seam (A -> B positive)
                    let mut q = [0i32; NUM_MOBILE];
                    if !wall_a && !wall_b {
                        let wa = fa.mass[SP_WATER][idx_a] as f32;
                        let wb = fb.mass[SP_WATER][idx_b] as f32;
                        let oa = fa.mass[SP_OIL][idx_a] as f32;
                        let ob = fb.mass[SP_OIL][idx_b] as f32;
                        let liq_a = wa + oa;
                        let liq_b = wb + ob;

                        // Outward flow on A and inward (A -> B) flow on B
                        let flow_out_a = (fa.flow_e[idx_a] as f32) * (out_a[0] as f32)
                            + (fa.flow_s[idx_a] as f32) * (out_a[1] as f32);
                        let flow_in_b = (fb.flow_e[idx_b] as f32) * (-out_b[0] as f32)
                            + (fb.flow_s[idx_b] as f32) * (-out_b[1] as f32);
                        let mom_ab = 0.5 * (flow_out_a + flow_in_b);

                        let drive = 0.24 * (liq_a - liq_b) + 0.35 * mom_ab;
                        let (q_w, q_o) = if drive >= 0.0 && liq_a > 1e-4 {
                            (
                                drive * (wa / (liq_a + 1e-6)),
                                0.60 * drive * (oa / (liq_a + 1e-6)) + 0.08 * (oa - ob),
                            )
                        } else if drive < 0.0 && liq_b > 1e-4 {
                            (
                                drive * (wb / (liq_b + 1e-6)),
                                0.60 * drive * (ob / (liq_b + 1e-6)) + 0.08 * (oa - ob),
                            )
                        } else {
                            (0.0, 0.08 * (oa - ob))
                        };

                        let sa = fa.mass[SP_STEAM][idx_a] as f32;
                        let sb = fb.mass[SP_STEAM][idx_b] as f32;
                        let ka = fa.mass[SP_SMOKE][idx_a] as f32;
                        let kb = fb.mass[SP_SMOKE][idx_b] as f32;
                        let q_st = 0.20 * (sa - sb);
                        let q_sm = 0.18 * (ka - kb);

                        let raw_q = [q_w, q_st, q_sm, q_o];
                        for m in 0..NUM_MOBILE {
                            let sp = MOBILE_SPECIES[m];
                            let q_int = raw_q[m].round_ties_even() as i32;
                            q[m] = if q_int > 0 {
                                q_int.min((fa.mass[sp][idx_a] / 4) as i32)
                            } else if q_int < 0 {
                                q_int.max(-((fb.mass[sp][idx_b] / 4) as i32))
                            } else {
                                0
                            };
                        }

                        // Volume ceiling on receiver B (positive flows)
                        let pos_tot: i64 = q.iter().filter(|&&v| v > 0).map(|&v| v as i64).sum();
                        let cap_room_b = (H_CAP.saturating_sub(vol_b) / 4) as i64;
                        if pos_tot > cap_room_b && pos_tot > 0 {
                            for v in &mut q {
                                if *v > 0 {
                                    *v = ((*v as i64) * cap_room_b / pos_tot) as i32;
                                }
                            }
                        }

                        // Volume ceiling on receiver A (negative flows)
                        let neg_tot: i64 = q.iter().filter(|&&v| v < 0).map(|&v| (-v) as i64).sum();
                        let cap_room_a = (H_CAP.saturating_sub(vol_a) / 4) as i64;
                        if neg_tot > cap_room_a && neg_tot > 0 {
                            for v in &mut q {
                                if *v < 0 {
                                    *v = -(((-*v) as i64) * cap_room_a / neg_tot) as i32;
                                }
                            }
                        }
                    }

                    // 2. Exact proportional advected thermal energy (`A -> B` positive)
                    let mut advected_e = 0i64;
                    let mut delta_c_a = 0i64;
                    let mut delta_c_b = 0i64;
                    for m in 0..NUM_MOBILE {
                        let qv = q[m] as i64;
                        if qv == 0 {
                            continue;
                        }
                        let sp = MOBILE_SPECIES[m];
                        let cs = SPECIES[sp].c;
                        delta_c_a -= qv * cs;
                        delta_c_b += qv * cs;
                        if qv > 0 && cap_a > 0 {
                            advected_e += ((qv as i128) * (cs as i128) * (fa.energy[idx_a] as i128)
                                / (cap_a as i128)) as i64;
                        } else if qv < 0 && cap_b > 0 {
                            let q_abs = -qv;
                            advected_e -= ((q_abs as i128)
                                * (cs as i128)
                                * (fb.energy[idx_b] as i128)
                                / (cap_b as i128)) as i64;
                        }
                    }

                    // 3. Exact discrete 2nd-law heat conduction across the seam (even through rock!)
                    let mut conducted_e = 0i64;
                    let dt = temp_a - temp_b;
                    if cap_a > 0 && cap_b > 0 && dt != 0 {
                        let k_a = weighted_conductivity(fa, idx_a);
                        let k_b = weighted_conductivity(fb, idx_b);
                        let k_eff = 2.0 * k_a * k_b / (k_a + k_b + 1e-6);
                        let c_eff = (cap_a as f64 * cap_b as f64) / ((cap_a + cap_b) as f64);
                        let g_raw = (0.012 * k_eff * c_eff * (dt as f64)).round() as i64;
                        let max_g = ((dt.unsigned_abs() as u128)
                            * (cap_a as u128)
                            * (cap_b as u128)
                            / (4 * ((cap_a + cap_b) as u128))) as i64;
                        conducted_e = if dt > 0 {
                            g_raw.clamp(0, max_g)
                        } else {
                            g_raw.clamp(-max_g, 0)
                        };

                        let c_a_post = (cap_a + delta_c_a).max(0);
                        let c_b_post = (cap_b + delta_c_b).max(0);
                        let e_a_adv = fa.energy[idx_a] - advected_e;
                        let e_b_adv = fb.energy[idx_b] + advected_e;

                        if conducted_e > 0 {
                            let max_give_a = (e_a_adv - c_a_post * T_MIN).max(0);
                            let max_take_b = (c_b_post * T_MAX - e_b_adv).max(0);
                            conducted_e = conducted_e.min(max_give_a).min(max_take_b);
                        } else if conducted_e < 0 {
                            let max_give_b = (e_b_adv - c_b_post * T_MIN).max(0);
                            let max_take_a = (c_a_post * T_MAX - e_a_adv).max(0);
                            conducted_e = conducted_e.max(-max_give_b).max(-max_take_a);
                        }
                    }

                    // 4. Apply strictly antisymmetric mass & thermal energy updates (`a -= x; b += x`)
                    let mut mass_moved = 0i64;
                    for m in 0..NUM_MOBILE {
                        let qv = q[m];
                        if qv != 0 {
                            let sp = MOBILE_SPECIES[m];
                            fa.mass[sp][idx_a] = (fa.mass[sp][idx_a] as i32 - qv) as u16;
                            fb.mass[sp][idx_b] = (fb.mass[sp][idx_b] as i32 + qv) as u16;
                            mass_moved += (qv as i64).abs();
                        }
                    }
                    let de = advected_e + conducted_e;
                    if de != 0 {
                        fa.energy[idx_a] -= de;
                        fb.energy[idx_b] += de;
                    }

                    // 5. Rotate & couple tangent flow vectors `(flow_e, flow_s)` cleanly across the rotated edge!
                    let liq_q = (q[0] + q[3]).clamp(i16::MIN as i32, i16::MAX as i32);
                    if liq_q != 0 {
                        let (rot_b_in_a_e, rot_b_in_a_s) =
                            link_ba.rotate_flow_i16(fb.flow_e[idx_b], fb.flow_s[idx_b]);
                        let along_a = edge_along_uv(edge_a);
                        let tang_a = ((fa.flow_e[idx_a] as i32 * along_a[0] as i32
                            + fa.flow_s[idx_a] as i32 * along_a[1] as i32)
                            + (rot_b_in_a_e as i32 * along_a[0] as i32
                                + rot_b_in_a_s as i32 * along_a[1] as i32))
                            / 2;
                        let new_fe_a = (liq_q * out_a[0] as i32 + tang_a * along_a[0] as i32)
                            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        let new_fs_a = (liq_q * out_a[1] as i32 + tang_a * along_a[1] as i32)
                            .clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        fa.flow_e[idx_a] = new_fe_a;
                        fa.flow_s[idx_a] = new_fs_a;

                        let (new_fe_b, new_fs_b) = link_ab.rotate_flow_i16(new_fe_a, new_fs_a);
                        fb.flow_e[idx_b] = new_fe_b;
                        fb.flow_s[idx_b] = new_fs_b;
                    }

                    if mass_moved > 0 || de != 0 {
                        stats.seam_mass_crossed += mass_moved;
                        stats.seam_energy_crossed += de.abs();
                        stats.active_seam_cells += 1;
                    }
                }
            }
        }

        self.total_seam_mass_crossed += stats.seam_mass_crossed;
        self.total_seam_energy_crossed += stats.seam_energy_crossed;
        stats
    }

    /// Real sweeping solar Day/Night Terminator across all 6 curved faces!
    /// - Computes `cos_theta = dot(dir, sun_dir)` per column.
    /// - Warms sunlit hemisphere (`cos_theta > 0.15`), radiatively cools dark night hemisphere
    ///   (`cos_theta < -0.10`, freezing exposed night rivers/oceans below `273.15 K`).
    /// - Strictly accounts for every joule in each face's `faucet_energy` so **`global_energy_error() == 0`**!
    pub fn apply_planetary_solar_terminator(&mut self, sun_dir: [f32; 3], tick: u64) {
        let sun = normalize3(sun_dir);
        self.sun_dir = sun;
        let n = self.face_size;
        let d = self.depth;
        let nf = n.max(1) as f32;
        let stride = 2usize;
        let phase = (tick as usize) % stride;
        let ring_z_floor = d.saturating_sub(10);

        for f in 0..FACE_COUNT {
            let vf = &mut self.faces[f];
            for y in 0..n {
                let v = ((y as f32 + 0.5) / nf) * 2.0 - 1.0;
                for x in 0..n {
                    if ((x + y) % stride) != phase {
                        continue;
                    }
                    let u = ((x as f32 + 0.5) / nf) * 2.0 - 1.0;
                    let dir = face_uv_to_sphere_dir(f, u, v);
                    let cos_theta = dot3(dir, sun);
                    if (-0.10..=0.15).contains(&cos_theta) {
                        continue;
                    }

                    let idx = y * n + x;
                    // Find topmost exposed planetary surface voxel (`z < ring_z_floor`)
                    let mut surf_z = None;
                    for z in (0..ring_z_floor.min(d)).rev() {
                        if vf.layers[z].cell_volume(idx) >= 80 {
                            surf_z = Some(z);
                            break;
                        }
                    }

                    if let Some(z) = surf_z {
                        let layer = &mut vf.layers[z];
                        let c = layer.cell_heat_capacity(idx);
                        if c > 0 {
                            let t = layer.cell_temp_ck(idx);
                            if cos_theta > 0.15 {
                                // Sunlit hemisphere warming
                                let sun_strength = ((cos_theta - 0.15) / 0.85).clamp(0.0, 1.0);
                                let target_t = if dir[1].abs() > 0.74 {
                                    // Keep polar glaciers sub-freezing even in oblique polar sun
                                    26_800
                                } else {
                                    30_600 + (2_600.0 * sun_strength) as i64
                                };
                                if t < target_t {
                                    let heat_ck = ((520.0 * sun_strength).max(80.0)) as i64;
                                    let cap_e = c * target_t;
                                    let delta = (heat_ck * c).min((cap_e - layer.energy[idx]).max(0));
                                    if delta > 0 {
                                        layer.energy[idx] += delta;
                                        layer.faucet_energy += delta;
                                    }
                                }
                            } else {
                                // Dark night hemisphere radiative cooling (freezes night rivers/oceans!)
                                let night_strength = ((-cos_theta - 0.10) / 0.90).clamp(0.0, 1.0);
                                // Preserve incandescent volcanic magma rifts on the night side
                                let is_active_magma = t > 70_000 && layer.mass[SP_ROCK][idx] >= 400;
                                if !is_active_magma {
                                    let target_t = 26_100 - (1_100.0 * night_strength) as i64;
                                    if t > target_t {
                                        let cool_ck = ((560.0 * night_strength).max(90.0)) as i64;
                                        let min_e = c * target_t;
                                        let delta =
                                            (cool_ck * c).min((layer.energy[idx] - min_e).max(0));
                                        if delta > 0 {
                                            layer.energy[idx] -= delta;
                                            layer.faucet_energy -= delta;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run one complete conservation-exact planetary physics tick:
    /// 1. Parallel per-face 3D RefSim + Projection Firewall + vertical gravity/buoyancy (`std::thread::scope`)
    /// 2. Deterministic 12-edge antisymmetric cross-face fluid/gas/heat exchange
    /// 3. Sweeping solar Day/Night Terminator radiative forcing
    /// 4. Parallel per-face `RadialInnerShell` structural support & orbital Sky-Ring collapse
    pub fn step_planet(&mut self) -> PlanetStepStats {
        let t0 = Instant::now();

        // 1. Parallel per-face 3D RefSim + vertical conservation step (100% independent per face!)
        std::thread::scope(|s| {
            for face in &mut self.faces {
                s.spawn(move || {
                    let raw = face.refsim_raw_outputs_all();
                    let _ = face.step_3d(&raw);
                });
            }
        });

        // 2. Exact 12-edge rotated cross-face flux conservation
        let edge_stats = self.exchange_12_edges_conservative();

        // 3. Sweeping solar Day/Night Terminator
        let sun = self.sun_dir;
        self.apply_planetary_solar_terminator(sun, self.tick);

        // 4. Parallel `RadialInnerShell` structural support & orbital Sky-Island collapse
        let mut fallen_solid_mass = 0i64;
        let mut splashed_fluid_mass = 0i64;
        let mut struct_results: [StructuralStepStats; FACE_COUNT] =
            std::array::from_fn(|_| StructuralStepStats::default());
        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(FACE_COUNT);
            for face in &mut self.faces {
                handles.push(s.spawn(move || face.step_structural_mechanics()));
            }
            for (i, h) in handles.into_iter().enumerate() {
                struct_results[i] = h.join().expect("structural worker panicked");
            }
        });
        for sr in &struct_results {
            fallen_solid_mass += sr.fallen_solid_mass;
            splashed_fluid_mass += sr.splashed_fluid_mass;
        }

        self.tick += 1;
        PlanetStepStats {
            tick: self.tick,
            global_mass_error: self.global_mass_error(),
            global_energy_error: self.global_energy_error(),
            seam_mass_crossed: edge_stats.seam_mass_crossed,
            seam_energy_crossed: edge_stats.seam_energy_crossed,
            fallen_solid_mass,
            splashed_fluid_mass,
            step_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
    }

    /// Sample exact voxel `(face, x, y, z)` for a given 3D unit sphere direction `dir` and layer `z`.
    #[inline]
    pub fn sample_dir_cell(&self, dir: [f32; 3]) -> (FaceId, usize, usize) {
        let (face, u, v) = sphere_dir_to_face_uv(dir);
        let nf = self.face_size as f32;
        let x = (((u + 1.0) * 0.5 * nf).floor() as isize).clamp(0, self.face_size as isize - 1)
            as usize;
        let y = (((v + 1.0) * 0.5 * nf).floor() as isize).clamp(0, self.face_size as isize - 1)
            as usize;
        (face, x, y)
    }
}

fn weighted_conductivity(field: &Field, idx: usize) -> f64 {
    let mut tot = 0.0f64;
    let mut sum_k = 0.0f64;
    for s in 0..NUM_SPECIES {
        let m = field.mass[s][idx] as f64;
        tot += m;
        sum_k += m * (SPECIES[s].k as f64);
    }
    if tot > 1e-4 { sum_k / tot } else { 2.0 }
}

// =============================================================================
// Unit Tests (5 Comprehensive Verification Suites)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubed_sphere_12_edge_involution_and_3d_projection_roundtrip() {
        // 1. Verify all 6 face frames are right-handed orthonormal (`e_u × e_v == n`).
        for f in 0..FACE_COUNT {
            let fr = FACE_FRAMES[f];
            let cx = fr.e_u[1] * fr.e_v[2] - fr.e_u[2] * fr.e_v[1];
            let cy = fr.e_u[2] * fr.e_v[0] - fr.e_u[0] * fr.e_v[2];
            let cz = fr.e_u[0] * fr.e_v[1] - fr.e_u[1] * fr.e_v[0];
            assert_eq!([cx, cy, cz], fr.n, "face {f} must be right-handed");
            assert_eq!(dot_i8(fr.n, fr.n), 1);
            assert_eq!(dot_i8(fr.e_u, fr.e_u), 1);
            assert_eq!(dot_i8(fr.e_v, fr.e_v), 1);
            assert_eq!(dot_i8(fr.n, fr.e_u), 0);
            assert_eq!(dot_i8(fr.n, fr.e_v), 0);
            assert_eq!(dot_i8(fr.e_u, fr.e_v), 0);
        }

        // 2. Verify all 24 directed half-edges satisfy strict involution, proper rotation (`det == +1`),
        //    `R_BA * R_AB == I`, and exact 3D point coincidence along every edge!
        let n = 16usize;
        for f in 0..FACE_COUNT {
            for e in 0..EDGE_COUNT {
                let ab = edge_neighbor(f, e);
                let ba = edge_neighbor(ab.face, ab.edge);
                assert_eq!(ba.face, f, "involution face mismatch on ({f},{e})");
                assert_eq!(ba.edge, e, "involution edge mismatch on ({f},{e})");
                assert_eq!(ab.flip_along, ba.flip_along, "flip_along must be symmetric");
                assert_eq!(ab.det(), 1, "edge rotation must be proper (det == +1)");

                // Verify R_BA * R_AB == I_2
                for r in 0..2 {
                    for c in 0..2 {
                        let prod = ba.rot[r][0] * ab.rot[0][c] + ba.rot[r][1] * ab.rot[1][c];
                        let want = if r == c { 1 } else { 0 };
                        assert_eq!(prod, want, "R_BA * R_AB != I on ({f},{e})");
                    }
                }

                // Verify exact 3D sphere direction continuity for every point along the seam
                for k in 0..n {
                    let t = ((k as f32 + 0.5) / n as f32) * 2.0 - 1.0;
                    let (ua, va) = match e {
                        EDGE_POS_U => (1.0, t),
                        EDGE_NEG_U => (-1.0, t),
                        EDGE_POS_V => (t, 1.0),
                        EDGE_NEG_V => (t, -1.0),
                        _ => unreachable!(),
                    };
                    let (ub, vb) = ab.map_uv(e, ua, va);
                    let da = face_uv_to_sphere_dir(f, ua, va);
                    let db = face_uv_to_sphere_dir(ab.face, ub, vb);
                    let err = ((da[0] - db[0]).powi(2)
                        + (da[1] - db[1]).powi(2)
                        + (da[2] - db[2]).powi(2))
                    .sqrt();
                    assert!(err < 1e-5, "3D seam discontinuity on ({f},{e}) k={k}: err={err}");
                }
            }
        }

        // Verify canonical_12_edges produces 12 distinct undirected edges
        let canon = canonical_12_edges();
        assert_eq!(canon.len(), 12);
        for (fa, _ea, link) in canon {
            assert!(fa < link.face);
        }

        // 3. Verify `face_uv_to_sphere_dir <-> sphere_dir_to_face_uv` roundtrip across all 6 faces
        for f in 0..FACE_COUNT {
            for iy in 0..7 {
                for ix in 0..7 {
                    let u = -0.9 + (ix as f32) * 0.3;
                    let v = -0.9 + (iy as f32) * 0.3;
                    let dir = face_uv_to_sphere_dir(f, u, v);
                    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
                    assert!((len - 1.0).abs() < 1e-5);
                    let (f2, u2, v2) = sphere_dir_to_face_uv(dir);
                    assert_eq!(f2, f);
                    assert!((u2 - u).abs() < 1e-5 && (v2 - v).abs() < 1e-5);
                }
            }
        }

        // Verify `face_cell_to_world_pos` radial shell distance
        let pos = face_cell_to_world_pos(FACE_POS_Z, 8, 8, 12, 16, 48.0);
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!((r - (48.0 + 12.5)).abs() < 1e-4);
    }

    #[test]
    fn test_planet_cross_face_river_and_heat_exact_conservation() {
        // Test water & heat flowing across the 90°-rotated seam between `+X` (`EDGE_POS_V`)
        // and `+Y` (`EDGE_POS_U`) — where `rot = [[0, -1], [1, 0]]` rotates +v flow into -u flow!
        let n = 16usize;
        let d = 8usize;
        let mut planet = CubedSpherePlanet::new_empty(n, d, 32.0, 42);

        let link = edge_neighbor(FACE_POS_X, EDGE_POS_V);
        assert_eq!(link.face, FACE_POS_Y);
        assert_eq!(link.edge, EDGE_POS_U);
        assert!(link.rot[0][0] == 0 && link.rot[1][1] == 0, "must be a 90-deg rotated seam");

        // Place hot river water along `+X`'s `EDGE_POS_V` (`y = n - 1`) pushing northward (`flow_s > 0`),
        // and cooler shallow water on `+Y`'s matching `EDGE_POS_U` (`x = n - 1`).
        let z = 3usize;
        for k in 4..12 {
            let (xa, ya) = edge_cell_coords(EDGE_POS_V, k, n);
            let idx_a = ya * n + xa;
            planet.faces[FACE_POS_X].layers[z].mass[SP_WATER][idx_a] = 1400;
            planet.faces[FACE_POS_X].layers[z].flow_s[idx_a] = 180;
            planet.faces[FACE_POS_X].layers[z].set_cell_temp(idx_a, 34_000);

            let (xb, yb) = link.map_edge_cell(k, n);
            let idx_b = yb * n + xb;
            planet.faces[FACE_POS_Y].layers[z].mass[SP_WATER][idx_b] = 200;
            planet.faces[FACE_POS_Y].layers[z].set_cell_temp(idx_b, 28_000);
        }
        planet.snapshot_conservation_baseline();

        let water_y_before: i64 = planet.faces[FACE_POS_Y].layers[z].mass[SP_WATER]
            .iter()
            .map(|&v| v as i64)
            .sum();
        let energy_y_before: i64 = planet.faces[FACE_POS_Y].layers[z].energy.iter().sum();

        let mut observed_neg_flow_e_y = false;
        let (xb_mid, yb_mid) = link.map_edge_cell(8, n);
        for step in 0..8 {
            let ex = planet.exchange_12_edges_conservative();
            assert!(ex.seam_mass_crossed > 0);
            assert_eq!(planet.global_mass_error(), 0, "exact mass conservation across 90-deg seam");
            assert_eq!(planet.global_energy_error(), 0, "exact energy conservation across 90-deg seam");
            if step == 0 {
                let flow_e_y = planet.faces[FACE_POS_Y].layers[z].flow_e[yb_mid * n + xb_mid];
                if flow_e_y < 0 {
                    observed_neg_flow_e_y = true;
                }
            }
        }

        let water_y_after: i64 = planet.faces[FACE_POS_Y].layers[z].mass[SP_WATER]
            .iter()
            .map(|&v| v as i64)
            .sum();
        let energy_y_after: i64 = planet.faces[FACE_POS_Y].layers[z].energy.iter().sum();
        assert!(
            water_y_after > water_y_before + 800,
            "water must flow across rotated +X -> +Y seam: before={water_y_before} after={water_y_after}"
        );
        assert!(
            energy_y_after > energy_y_before,
            "heat must flow across rotated +X -> +Y seam"
        );
        assert!(
            observed_neg_flow_e_y,
            "flow entering +Y from EDGE_POS_U must rotate to negative flow_e"
        );
    }

    #[test]
    fn test_planet_sweeping_terminator_and_radial_collapse_exact_conservation() {
        let mut planet = CubedSpherePlanet::new(24, 24, 32.0, 99);
        assert_eq!(planet.global_mass_error(), 0);
        assert_eq!(planet.global_energy_error(), 0);
        assert!(planet.buoyant_core_count() >= 4, "must have registered orbital Sky-Ring BuoyantCores");

        // Ensure all faces use RadialInnerShell
        for f in &planet.faces {
            assert_eq!(f.structural.anchor_mode, BaseAnchorMode::RadialInnerShell);
        }

        // 1. Test sweeping solar terminator: point sun at `+X`, verify `+X` warms while `-X` cools!
        let mean_surf_temp = |p: &CubedSpherePlanet, face: FaceId| -> f64 {
            let vf = &p.faces[face];
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            let ring_z_floor = p.depth.saturating_sub(10);
            for y in 4..(p.face_size - 4) {
                for x in 4..(p.face_size - 4) {
                    let idx = y * p.face_size + x;
                    for z in (0..ring_z_floor).rev() {
                        if vf.layers[z].cell_volume(idx) >= 80 {
                            let t = vf.layers[z].cell_temp_ck(idx);
                            if t < 60_000 {
                                sum += t as f64;
                                cnt += 1;
                            }
                            break;
                        }
                    }
                }
            }
            sum / cnt.max(1) as f64
        };

        let t_pos_x_0 = mean_surf_temp(&planet, FACE_POS_X);
        let t_neg_x_0 = mean_surf_temp(&planet, FACE_NEG_X);

        for step in 0..8u64 {
            planet.apply_planetary_solar_terminator([1.0, 0.0, 0.0], step);
            assert_eq!(planet.global_mass_error(), 0);
            assert_eq!(planet.global_energy_error(), 0);
        }

        let t_pos_x_1 = mean_surf_temp(&planet, FACE_POS_X);
        let t_neg_x_1 = mean_surf_temp(&planet, FACE_NEG_X);
        assert!(
            t_pos_x_1 > t_pos_x_0 + 150.0,
            "sunlit +X hemisphere must warm: {t_pos_x_0:.1} -> {t_pos_x_1:.1}"
        );
        assert!(
            t_neg_x_1 < t_neg_x_0 - 150.0,
            "dark -X night hemisphere must cool: {t_neg_x_0:.1} -> {t_neg_x_1:.1}"
        );

        // 2. Fracture an orbital Sky-Ring BuoyantCore on `+Z` and verify radial inward crash
        //    with exact `global_mass_error == 0 && global_energy_error == 0`!
        let core_id = planet.faces[FACE_POS_Z].structural.buoyant_cores[0].id;
        let fractured = planet.fracture_orbital_island(FACE_POS_Z, core_id);
        assert!(fractured.is_some());
        assert_eq!(planet.global_mass_error(), 0);
        assert_eq!(planet.global_energy_error(), 0);

        let mut total_fallen = 0i64;
        for _ in 0..18 {
            let sr = planet.faces[FACE_POS_Z].step_structural_mechanics();
            total_fallen += sr.fallen_solid_mass;
            assert_eq!(planet.global_mass_error(), 0);
            assert_eq!(planet.global_energy_error(), 0);
        }
        assert!(
            total_fallen > 5_000,
            "fractured orbital Sky-Ring island must collapse radially inward, fell {total_fallen}"
        );
    }

    #[test]
    fn test_spherical_entity_circumnavigation_equator_and_poles() {
        let n = 32usize;
        let r_core = 48.0f32;

        // 1. 360° Equatorial circumnavigation: `+Z -> +X -> -Z -> -X -> +Z`
        let mut eq = SphericalEntity {
            face: FACE_POS_Z,
            u_cell: 16.0,
            v_cell: 16.0,
            z: 18.0,
            vu: 1.0,
            vv: 0.0,
        };
        let mut eq_visited = [false; FACE_COUNT];
        let mut eq_crossings = 0u32;
        for _ in 0..(4 * n) {
            eq_visited[eq.face] = true;
            let p0 = eq.world_pos(n, r_core);
            eq_crossings += step_spherical_entity(&mut eq, n, 1.0);
            let p1 = eq.world_pos(n, r_core);
            let step_dist = ((p1[0] - p0[0]).powi(2)
                + (p1[1] - p0[1]).powi(2)
                + (p1[2] - p0[2]).powi(2))
            .sqrt();
            assert!(step_dist < 5.0, "no seam teleport glitch on equator: dist={step_dist}");
        }
        assert_eq!(eq_crossings, 4);
        assert!(eq_visited[FACE_POS_Z] && eq_visited[FACE_POS_X] && eq_visited[FACE_NEG_Z] && eq_visited[FACE_NEG_X]);
        assert_eq!(eq.face, FACE_POS_Z);
        assert!((eq.u_cell - 16.0).abs() < 1e-2 && (eq.v_cell - 16.0).abs() < 1e-2);
        assert!((eq.vu - 1.0).abs() < 1e-4 && eq.vv.abs() < 1e-4);

        // 2. 360° Polar circumnavigation over BOTH poles (`+X -> +Y -> -X -> -Y -> +X`, exercising 90°/270° rotated seams!)
        let mut pol = SphericalEntity {
            face: FACE_POS_X,
            u_cell: 11.5,
            v_cell: 16.0,
            z: 18.0,
            vu: 0.0,
            vv: 1.0,
        };
        let mut pol_visited = [false; FACE_COUNT];
        let mut pol_crossings = 0u32;
        for _ in 0..(4 * n) {
            pol_visited[pol.face] = true;
            let p0 = pol.world_pos(n, r_core);
            pol_crossings += step_spherical_entity(&mut pol, n, 1.0);
            let p1 = pol.world_pos(n, r_core);
            let step_dist = ((p1[0] - p0[0]).powi(2)
                + (p1[1] - p0[1]).powi(2)
                + (p1[2] - p0[2]).powi(2))
            .sqrt();
            assert!(step_dist < 5.0, "no seam teleport glitch over poles: dist={step_dist}");
        }
        assert_eq!(pol_crossings, 4);
        assert!(pol_visited[FACE_POS_X] && pol_visited[FACE_POS_Y] && pol_visited[FACE_NEG_X] && pol_visited[FACE_NEG_Y]);
        assert_eq!(pol.face, FACE_POS_X);
        assert!((pol.u_cell - 11.5).abs() < 1e-2 && (pol.v_cell - 16.0).abs() < 1e-2);
        assert!(pol.vu.abs() < 1e-4 && (pol.vv - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_planet_full_step_exact_conservation_and_timing() {
        let mut planet = CubedSpherePlanet::new(24, 24, 32.0, 29029);
        assert_eq!(planet.total_voxels(), 6 * 24 * 24 * 24);
        assert_eq!(planet.global_mass_error(), 0);
        assert_eq!(planet.global_energy_error(), 0);

        let mut total_seam_mass = 0i64;
        for step in 0..5 {
            // Rotate sun smoothly across ticks to test live sweeping terminator during full step
            let ang = step as f32 * 0.25;
            planet.sun_dir = normalize3([ang.cos(), 0.22, ang.sin()]);
            let st = planet.step_planet();
            assert_eq!(st.global_mass_error, 0, "global mass error must be 0 at tick {}", st.tick);
            assert_eq!(st.global_energy_error, 0, "global energy error must be 0 at tick {}", st.tick);
            total_seam_mass += st.seam_mass_crossed;
            assert!(st.step_ms < 5_000.0, "planet step timing must be bounded, got {} ms", st.step_ms);
        }
        assert!(total_seam_mass > 0, "oceans/gas must actively exchange across 12 planet edges");
    }
}
