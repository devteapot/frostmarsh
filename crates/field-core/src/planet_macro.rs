//! Wave 33B: `field-core::planet_macro` — Dual-Scale Locally-Flat Cubed-Sphere Planetary Engine.
//!
//! Bridges two synchronized planetary scales so ground play feels **100% locally flat and vast**
//! (`R = 1,300 m`, `horizon_drop_50m ≈ 0.96 m < 1.0 m`, `1 m` detailed voxels, caves, forests,
//! golden coasts, curved sapphire oceans, and towering Floating Sky-Islands) while seamlessly
//! scaling out to a complete closed `6 × 2048² × 64` (`1.61 billion` virtual voxels, `8.17 km`
//! circumference) cubed-sphere planet with:
//! - Exact `f64` floating-origin spherical geometry (`< 1e-6 m` vertex precision at both
//!   `R = 1,300 m` and Earth scale `R = 6,371,000 m`).
//! - Hierarchical `6 × M × M` (`384` shards at `M = 8`, `1,536` directed shard edges) cubed-sphere
//!   shard topology with exact involution and proper `det == +1` 2×2 tangent-flow rotation across
//!   all cube-face seams.
//! - Shared seamless 3D spherical procedural field (`sample_macro_planet_column`) guaranteeing
//!   consistent continents, oceans, polar caps, equatorial volcanic rift, and Sky-Islands across
//!   both the always-on `6 × 64²` `GlobalClimateShell` and every hydrated `256×256×64` `1 m` shard.
//! - On-demand lazy shard hydration/eviction with **bit-exact cross-shard mass & energy
//!   conservation** (`global_mass_error() == 0 && global_energy_error() == 0`) and bidirectional
//!   macro ↔ micro AMR climate coupling.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use crate::planet::{
    EDGE_COUNT, EDGE_NEG_U, EDGE_NEG_V, EDGE_POS_U, EDGE_POS_V, EdgeId, FACE_COUNT, FACE_FRAMES,
    FACE_NEG_X, FACE_NEG_Y, FACE_NEG_Z, FACE_POS_X, FACE_POS_Y, FACE_POS_Z, FaceId,
    edge_along_uv, edge_cell_coords, edge_neighbor, edge_outward_uv, face_name,
};
use crate::structural::{BaseAnchorMode, BuoyantCore, StructuralState};
use crate::{
    Field, H_CAP, MOBILE_SPECIES, NUM_MOBILE, NUM_SPECIES, SPECIES, SP_ASH, SP_ICE, SP_OIL,
    SP_PLANT, SP_ROCK, SP_SMOKE, SP_SOIL, SP_STEAM, SP_WATER, SP_WOOD, T_AMB, T_MAX, T_MIN,
    VolumeField, WORLD_LAW_STANDARD, WorldEvolutionStats, WorldLaws,
};

// =============================================================================
// 1. MacroPlanetConfig & Exact f64 Floating-Origin Spherical Geometry
// =============================================================================

/// Configuration for the Dual-Scale Locally-Flat Cubed-Sphere Planet.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MacroPlanetConfig {
    /// Number of shards along each axis of every cube face (default `8` -> `8×8 = 64` per face,
    /// `384` total shards across 6 faces).
    pub shards_per_face_axis: u8,
    /// Local horizontal resolution (`1 m` voxels) per hydrated shard (default `256`, yielding
    /// `8 × 256 = 2,048` virtual `1 m` cells per cube-face edge).
    pub shard_size: usize,
    /// Radial vertical voxel depth (`z ∈ 0..depth`, default `64` -> `1.61B` virtual voxels).
    pub depth: usize,
    /// Inner radial core radius in meters (`z = 0` shell radius; default `1280.0 m` so sea level
    /// `z_sea = 24` sits at `R_surf ≈ 1,304 m` and mean continent surface at `R ≈ 1,300..1,310 m`).
    pub r_core: f64,
    /// Sea-level voxel layer index (default `24` for `depth = 64`).
    pub z_sea: usize,
    /// Macro climate shell resolution per face axis (default `64` -> `6 × 64² = 24,576` columns).
    pub climate_res: usize,
    /// Deterministic planetary procedural seed.
    pub seed: u64,
}

impl Default for MacroPlanetConfig {
    fn default() -> Self {
        Self {
            shards_per_face_axis: 8,
            shard_size: 256,
            depth: 64,
            r_core: 1280.0,
            z_sea: 24,
            climate_res: 64,
            seed: 33_033,
        }
    }
}

impl MacroPlanetConfig {
    /// Virtual `1 m` cells along one cube-face edge (`shards_per_face_axis * shard_size`).
    #[inline]
    pub const fn face_size_virtual(&self) -> usize {
        (self.shards_per_face_axis as usize) * self.shard_size
    }

    /// Total shards across all 6 cube faces (`6 * M^2`).
    #[inline]
    pub const fn total_shards(&self) -> usize {
        let m = self.shards_per_face_axis as usize;
        FACE_COUNT * m * m
    }

    /// Total directed shard edges (`total_shards * 4`).
    #[inline]
    pub const fn directed_shard_edges(&self) -> usize {
        self.total_shards() * EDGE_COUNT
    }

    /// Total virtual `1 m` 3D voxels across the closed cubed-sphere (`6 * N_virt^2 * depth`).
    #[inline]
    pub const fn virtual_voxels(&self) -> u64 {
        let n = self.face_size_virtual() as u64;
        (FACE_COUNT as u64) * n * n * (self.depth as u64)
    }

    /// Nominal sea-level surface radius in meters (`r_core + z_sea`).
    #[inline]
    pub fn surface_radius_m(&self) -> f64 {
        self.r_core + self.z_sea as f64
    }

    /// Equatorial planetary circumference in meters (`2π R_surf`).
    #[inline]
    pub fn circumference_m(&self) -> f64 {
        std::f64::consts::TAU * self.surface_radius_m()
    }
}

/// Map virtual face coordinates `(gx, gy) ∈ [0, face_size_virtual]` to cube-face `(u, v) ∈ [-1, 1]`
/// in full `f64` precision.
#[inline]
pub fn face_virtual_uv_f64(gx: f64, gy: f64, face_size_virtual: usize) -> (f64, f64) {
    let n = face_size_virtual.max(1) as f64;
    let u = (gx / n) * 2.0 - 1.0;
    let v = (gy / n) * 2.0 - 1.0;
    (u, v)
}

/// Project cubed-sphere face coordinates `(face, u, v)` (`u, v ∈ [-1, 1]`) onto the unit sphere in `f64`.
#[inline]
pub fn face_uv_to_sphere_dir_f64(face: FaceId, u: f64, v: f64) -> [f64; 3] {
    let fr = FACE_FRAMES[face % FACE_COUNT];
    let px = fr.n[0] as f64 + u * fr.e_u[0] as f64 + v * fr.e_v[0] as f64;
    let py = fr.n[1] as f64 + u * fr.e_u[1] as f64 + v * fr.e_v[1] as f64;
    let pz = fr.n[2] as f64 + u * fr.e_u[2] as f64 + v * fr.e_v[2] as f64;
    let inv_len = 1.0 / (px * px + py * py + pz * pz).sqrt().max(1e-15);
    [px * inv_len, py * inv_len, pz * inv_len]
}

/// Exact inverse projection from any non-zero 3D direction `dir` to `(face, u, v)` in `f64`.
#[inline]
pub fn sphere_dir_to_face_uv_f64(dir: [f64; 3]) -> (FaceId, f64, f64) {
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

    let inv_d = 1.0 / denom.max(1e-15);
    let fr = FACE_FRAMES[face];
    let u = ((dir[0] * fr.e_u[0] as f64 + dir[1] * fr.e_u[1] as f64 + dir[2] * fr.e_u[2] as f64)
        * inv_d)
        .clamp(-1.0, 1.0);
    let v = ((dir[0] * fr.e_v[0] as f64 + dir[1] * fr.e_v[1] as f64 + dir[2] * fr.e_v[2] as f64)
        * inv_d)
        .clamp(-1.0, 1.0);
    (face, u, v)
}

/// Continuous `f64` 3D world position `(r_core + z) * face_uv_to_sphere_dir_f64(face, u, v)`.
#[inline]
pub fn face_uv_z_to_world_f64(face: FaceId, u: f64, v: f64, z: f64, r_core: f64) -> [f64; 3] {
    let dir = face_uv_to_sphere_dir_f64(face, u, v);
    let r = r_core + z;
    [r * dir[0], r * dir[1], r * dir[2]]
}

/// Convert an exact `f64` world position into camera-relative `f32` coordinates by subtracting
/// `cam_origin_f64` in `f64` BEFORE narrowing to `f32`.
///
/// Guarantees sub-millimeter local vertex precision even at `R = 1,300 m` and Earth scale
/// `R = 6,371,000 m` (zero floating-point jitter!).
#[inline]
pub fn world_f64_to_camera_rel_f32(world_pos: [f64; 3], cam_origin_f64: [f64; 3]) -> [f32; 3] {
    [
        (world_pos[0] - cam_origin_f64[0]) as f32,
        (world_pos[1] - cam_origin_f64[1]) as f32,
        (world_pos[2] - cam_origin_f64[2]) as f32,
    ]
}

/// Exact spherical horizon sagitta drop in meters at distance `dist_m` on a sphere of radius `radius_m`:
/// `R - sqrt(R^2 - d^2) = d^2 / (R + sqrt(R^2 - d^2))` (cancellation-free in `f64` even at Earth scale!).
#[inline]
pub fn horizon_drop_meters(dist_m: f64, radius_m: f64) -> f64 {
    let d = dist_m.abs();
    if radius_m <= 0.0 {
        return 0.0;
    }
    if d >= radius_m {
        return radius_m;
    }
    let rad = (radius_m * radius_m - d * d).sqrt();
    (d * d) / (radius_m + rad)
}

#[inline]
pub fn normalize3_f64(v: [f64; 3]) -> [f64; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= 1e-15 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
pub fn dot3_f64(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// =============================================================================
// 2. Hierarchical Cubed-Sphere Shard Topology (`6 × M × M` Shards)
// =============================================================================

/// Unique identifier for one shard on the `6 × M × M` hierarchical cubed-sphere planet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PlanetShardKey {
    pub face: u8,
    pub sx: u8,
    pub sy: u8,
}

impl PlanetShardKey {
    #[inline]
    pub const fn new(face: u8, sx: u8, sy: u8) -> Self {
        Self { face, sx, sy }
    }

    /// Virtual `[u0, u1] × [v0, v1]` bounds of this shard on its cube face (`u, v ∈ [-1, 1]`).
    #[inline]
    pub fn uv_bounds(self, shards_per_axis: u8) -> (f64, f64, f64, f64) {
        let m = (shards_per_axis.max(1)) as f64;
        let u0 = (self.sx as f64 / m) * 2.0 - 1.0;
        let u1 = ((self.sx as f64 + 1.0) / m) * 2.0 - 1.0;
        let v0 = (self.sy as f64 / m) * 2.0 - 1.0;
        let v1 = ((self.sy as f64 + 1.0) / m) * 2.0 - 1.0;
        (u0, u1, v0, v1)
    }

    /// 3D unit sphere direction at the center of this shard.
    #[inline]
    pub fn center_sphere_dir(self, shards_per_axis: u8) -> [f64; 3] {
        let (u0, u1, v0, v1) = self.uv_bounds(shards_per_axis);
        face_uv_to_sphere_dir_f64(self.face as usize, 0.5 * (u0 + u1), 0.5 * (v0 + v1))
    }

    /// Convert local shard cell `(lx, ly)` to virtual face coordinates `(gx, gy)`.
    #[inline]
    pub const fn local_to_virtual(self, lx: usize, ly: usize, shard_size: usize) -> (usize, usize) {
        (
            self.sx as usize * shard_size + lx,
            self.sy as usize * shard_size + ly,
        )
    }
}

/// Exact topological and tangent-frame link across one shard boundary edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardEdgeLink {
    pub neighbor: PlanetShardKey,
    pub edge: EdgeId,
    /// True if the along-edge coordinate reverses (`k' = shard_size - 1 - k`) across the seam.
    pub flip_along: bool,
    /// Exact 2×2 signed permutation matrix (`det == +1`) rotating `(v_u, v_v)` from source shard
    /// into neighbor shard's `(e_u', e_v')` basis.
    pub rot: [[i8; 2]; 2],
    /// True when this shard edge lies on one of the 12 cube-face boundaries.
    pub crosses_cube_face: bool,
}

impl ShardEdgeLink {
    /// Determinant of `self.rot` (always `+1` for right-handed cubed-sphere faces).
    #[inline]
    pub const fn det(&self) -> i8 {
        self.rot[0][0] * self.rot[1][1] - self.rot[0][1] * self.rot[1][0]
    }

    /// Map along-edge local cell index `k ∈ 0..shard_size` to `(xb, yb)` on `self.neighbor`.
    #[inline]
    pub const fn map_edge_cell(&self, k: usize, shard_size: usize) -> (usize, usize) {
        let kb = if self.flip_along {
            shard_size - 1 - k
        } else {
            k
        };
        edge_cell_coords(self.edge, kb, shard_size)
    }

    /// Rotate an integer tangent flow `(flow_e, flow_s)` into the neighbor shard's local basis.
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
}

/// Compute the exact neighbor shard, matching edge, along-edge flip flag, and 2×2 tangent
/// rotation matrix for ANY shard `(face, sx, sy)` and edge (`0..4`) on a `6 × M × M` cubed sphere.
pub const fn shard_neighbor(
    key: PlanetShardKey,
    edge: EdgeId,
    shards_per_axis: u8,
) -> ShardEdgeLink {
    let m = if shards_per_axis == 0 {
        1
    } else {
        shards_per_axis
    };
    let identity = [[1i8, 0], [0, 1]];

    match edge {
        EDGE_POS_U if key.sx + 1 < m => ShardEdgeLink {
            neighbor: PlanetShardKey {
                face: key.face,
                sx: key.sx + 1,
                sy: key.sy,
            },
            edge: EDGE_NEG_U,
            flip_along: false,
            rot: identity,
            crosses_cube_face: false,
        },
        EDGE_NEG_U if key.sx > 0 => ShardEdgeLink {
            neighbor: PlanetShardKey {
                face: key.face,
                sx: key.sx - 1,
                sy: key.sy,
            },
            edge: EDGE_POS_U,
            flip_along: false,
            rot: identity,
            crosses_cube_face: false,
        },
        EDGE_POS_V if key.sy + 1 < m => ShardEdgeLink {
            neighbor: PlanetShardKey {
                face: key.face,
                sx: key.sx,
                sy: key.sy + 1,
            },
            edge: EDGE_NEG_V,
            flip_along: false,
            rot: identity,
            crosses_cube_face: false,
        },
        EDGE_NEG_V if key.sy > 0 => ShardEdgeLink {
            neighbor: PlanetShardKey {
                face: key.face,
                sx: key.sx,
                sy: key.sy - 1,
            },
            edge: EDGE_POS_V,
            flip_along: false,
            rot: identity,
            crosses_cube_face: false,
        },
        _ => {
            // Cube-face boundary edge: delegate to exact `planet::edge_neighbor`!
            let fl = edge_neighbor(key.face as usize, edge);
            let k_shard = match edge {
                EDGE_POS_U | EDGE_NEG_U => key.sy,
                EDGE_POS_V | EDGE_NEG_V => key.sx,
                _ => 0,
            };
            let kb_shard = if fl.flip_along {
                m - 1 - k_shard
            } else {
                k_shard
            };
            let (nb_sx, nb_sy) = match fl.edge {
                EDGE_POS_U => (m - 1, kb_shard),
                EDGE_NEG_U => (0, kb_shard),
                EDGE_POS_V => (kb_shard, m - 1),
                EDGE_NEG_V => (kb_shard, 0),
                _ => (0, 0),
            };
            ShardEdgeLink {
                neighbor: PlanetShardKey {
                    face: fl.face as u8,
                    sx: nb_sx,
                    sy: nb_sy,
                },
                edge: fl.edge,
                flip_along: fl.flip_along,
                rot: fl.rot,
                crosses_cube_face: true,
            }
        }
    }
}

/// Locate the `PlanetShardKey` containing a given 3D unit sphere direction.
#[inline]
pub fn sphere_dir_to_shard_key(dir: [f64; 3], shards_per_axis: u8) -> PlanetShardKey {
    let m = shards_per_axis.max(1) as usize;
    let (face, u, v) = sphere_dir_to_face_uv_f64(dir);
    let sx = (((u + 1.0) * 0.5 * m as f64).floor() as isize).clamp(0, m as isize - 1) as u8;
    let sy = (((v + 1.0) * 0.5 * m as f64).floor() as isize).clamp(0, m as isize - 1) as u8;
    PlanetShardKey {
        face: face as u8,
        sx,
        sy,
    }
}

// =============================================================================
// 3. Seamless 3D Spherical Procedural Field (Shared by Macro Shell & 1m Shards!)
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
fn hash_grid_3d_f64(ix: i64, iy: i64, iz: i64, seed: u64) -> f64 {
    let h = hash_u64(
        (ix as u64).wrapping_mul(73_856_093)
            ^ (iy as u64).wrapping_mul(19_349_663)
            ^ (iz as u64).wrapping_mul(83_492_791)
            ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    ((h >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

#[inline]
fn smoothstep_f64(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn value_noise_3d_f64(p: [f64; 3], seed: u64) -> f64 {
    let ix = p[0].floor() as i64;
    let iy = p[1].floor() as i64;
    let iz = p[2].floor() as i64;
    let fx = smoothstep_f64(p[0] - ix as f64);
    let fy = smoothstep_f64(p[1] - iy as f64);
    let fz = smoothstep_f64(p[2] - iz as f64);

    let c000 = hash_grid_3d_f64(ix, iy, iz, seed);
    let c100 = hash_grid_3d_f64(ix + 1, iy, iz, seed);
    let c010 = hash_grid_3d_f64(ix, iy + 1, iz, seed);
    let c110 = hash_grid_3d_f64(ix + 1, iy + 1, iz, seed);
    let c001 = hash_grid_3d_f64(ix, iy, iz + 1, seed);
    let c101 = hash_grid_3d_f64(ix + 1, iy, iz + 1, seed);
    let c011 = hash_grid_3d_f64(ix, iy + 1, iz + 1, seed);
    let c111 = hash_grid_3d_f64(ix + 1, iy + 1, iz + 1, seed);

    let x00 = c000 + (c100 - c000) * fx;
    let x10 = c010 + (c110 - c010) * fx;
    let x01 = c001 + (c101 - c001) * fx;
    let x11 = c011 + (c111 - c011) * fx;
    let y0 = x00 + (x10 - x00) * fy;
    let y1 = x01 + (x11 - x01) * fy;
    y0 + (y1 - y0) * fz
}

fn fbm_3d_f64(p: [f64; 3], seed: u64, octaves: u32) -> f64 {
    let mut sum = 0.0f64;
    let mut amp = 0.5f64;
    let mut freq = 1.0f64;
    let mut norm = 0.0f64;
    for oct in 0..octaves {
        let q = [
            p[0] * freq + 17.3 * oct as f64,
            p[1] * freq - 11.7 * oct as f64,
            p[2] * freq + 23.1 * oct as f64,
        ];
        sum += amp * value_noise_3d_f64(q, seed.wrapping_add(oct as u64 * 1013));
        norm += amp;
        amp *= 0.5;
        freq *= 2.04;
    }
    if norm > 0.0 { sum / norm } else { 0.5 }
}

/// Continental macro-biome classification IDs (5 unmistakable planetary landscapes).
pub const CONTINENT_NORDIC_EMERALD: u8 = 0;
pub const CONTINENT_AUTUMN_HIGHLANDS: u8 = 1;
pub const CONTINENT_RED_OCHRE_MESA: u8 = 2;
pub const CONTINENT_CRYSTAL_TUNDRA: u8 = 3;
pub const CONTINENT_VOLCANIC_BADLANDS: u8 = 4;

/// Surface biome classification shared identically by the `6 × 64²` macro shell and `1 m` shards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MacroBiome {
    DeepOcean,
    CoastalBeach,
    EmeraldMeadow,
    PineForest,
    AlpineRock,
    AlpineSnow,
    PolarGlacier,
    VolcanicRim,
    VolcanicCaldera,
}

/// Continuous 3D spherical planetary column evaluated from unit sphere direction `dir`.
///
/// Evaluated identically by BOTH the `6 × 64²` `GlobalClimateShell` AND every hydrated
/// `256×256×64` `1 m` `VolumeField` shard so foreground `1 m` terrain stitches cleanly into
/// the distant `R = 1,300 m` macro globe!
#[derive(Clone, Copy, Debug)]
pub struct MacroColumnSample {
    /// Continuous terrain surface height in `z` units (`16.0..42.0`, sea level `z_sea = 24.0`).
    pub surf_z: f64,
    /// Sea-level height in `z` units (`24.0`).
    pub z_sea: f64,
    /// Ocean water depth (`(z_sea - surf_z).max(0.0)`).
    pub water_depth: f64,
    /// Carved highland/coastal river water depth above the riverbed (`0.0..1.6 m`).
    pub river_depth: f64,
    /// Natural glowing cavern sinkhole / grotto entrance depression (`0.0..10.5 m`).
    pub cavern_depth: f64,
    /// Surface equilibrium temperature in Kelvin.
    pub temp_k: f64,
    /// Tree/forest canopy density in `[0, 1]`.
    pub forest_density: f64,
    /// Moisture / precipitation potential in `[0, 1]`.
    pub moisture: f64,
    /// True if inside the polar glacier ice cap (`|dir.y| > 0.72`).
    pub is_polar: bool,
    /// True if inside the incandescent equatorial volcanic caldera core.
    pub is_rift_core: bool,
    /// True if on the scorched obsidian volcanic caldera rim.
    pub is_rift_rim: bool,
    /// Continental landscape ID (`CONTINENT_*`: 0=Nordic, 1=Autumn, 2=RedMesa, 3=Tundra, 4=Volcanic).
    pub continent_biome: u8,
    /// Surface biome tag.
    pub biome: MacroBiome,
    /// Physically plausible linear RGB surface albedo.
    pub albedo: [f32; 3],
    /// Linear RGB self-emissive radiance (incandescent volcanic rift).
    pub emissive: [f32; 3],
    /// Surface roughness in `[0, 1]`.
    pub roughness: f32,
}

/// Evaluate the seamless 3D spherical planetary terrain, biome, coastline, glacier, and volcanic
/// rift column at unit sphere direction `dir` (in `f64`).
pub fn sample_macro_planet_column(dir_in: [f64; 3], seed: u64) -> MacroColumnSample {
    let dir = normalize3_f64(dir_in);
    let lon = dir[2].atan2(dir[0]); // [-pi, pi]
    let lat_abs = dir[1].abs();
    let z_sea = 24.0f64;

    // Multi-scale seamless 3D spherical elevation octaves
    let n_cont = fbm_3d_f64([dir[0] * 2.1, dir[1] * 2.1, dir[2] * 2.1], seed, 4);
    let n_coast = fbm_3d_f64([dir[0] * 6.5, dir[1] * 6.5, dir[2] * 6.5], seed ^ 0x1357, 3);
    let n_ridge_raw = fbm_3d_f64(
        [dir[0] * 5.0, dir[1] * 5.0, dir[2] * 5.0],
        seed ^ 0xA55A,
        3,
    );
    let n_ridge = 1.0 - (n_ridge_raw * 2.0 - 1.0).abs();
    let n_detail = fbm_3d_f64(
        [dir[0] * 14.0, dir[1] * 14.0, dir[2] * 14.0],
        seed ^ 0x5AA5,
        2,
    );
    let n_moist = fbm_3d_f64(
        [dir[0] * 3.8 + 4.2, dir[1] * 3.8 - 1.7, dir[2] * 3.8 + 2.9],
        seed ^ 0x7777,
        3,
    );
    // Low-frequency 3D spherical continental biome field driving 5 distinct planetary landscapes
    let n_continent_biome = fbm_3d_f64(
        [dir[0] * 2.1, dir[1] * 2.1, dir[2] * 2.1],
        seed ^ 0xB10E,
        3,
    );

    // Equatorial volcanic caldera rift belt (`|dir.y| < 0.17` along longitude `lon ≈ -0.18`)
    // Taper smoothly along longitude with organic jaggedness so there are ZERO boxy cutoffs!
    let rift_curve = 0.068 * (lon * 3.8 + 0.35).sin()
        + 0.032 * (n_coast - 0.5)
        + 0.022 * (n_detail - 0.5);
    let d_lat_rift = (dir[1] - rift_curve).abs();
    let mut d_lon = lon - (-0.18);
    while d_lon > std::f64::consts::PI {
        d_lon -= std::f64::consts::TAU;
    }
    while d_lon < -std::f64::consts::PI {
        d_lon += std::f64::consts::TAU;
    }
    let lon_span = 0.72 + 0.12 * (n_coast - 0.5);
    let lon_taper = (1.0 - (d_lon / lon_span).powi(2)).max(0.0).powf(0.75);
    let core_half_w = (0.028 + 0.018 * (n_detail - 0.5).abs()) * lon_taper;
    let rim_half_w = (0.088 + 0.028 * n_detail) * lon_taper;
    let badlands_half_w = rim_half_w * 1.72;
    let in_rift_belt = lon_taper > 0.05 && d_lat_rift < rim_half_w;
    let is_rift_core = lon_taper > 0.14 && d_lat_rift < core_half_w;
    let is_rift_rim = in_rift_belt && !is_rift_core;
    let badlands_w = if lon_taper > 0.05 && !in_rift_belt && d_lat_rift < badlands_half_w {
        let t = (1.0 - (d_lat_rift - rim_half_w) / (badlands_half_w - rim_half_w).max(0.01))
            .clamp(0.0, 1.0);
        smoothstep_f64(t) * lon_taper
    } else {
        0.0
    };

    // Polar glacier ice caps (`|dir.y| > 0.74` with broad organic continental coastline modulation)
    let polar_thresh = 0.74 + 0.085 * (n_cont - 0.5) + 0.055 * (n_coast - 0.5);
    let is_polar = lat_abs > polar_thresh;

    // Smooth continental macro-biome blend weights across the 6 faces:
    // 1. Nordic Emerald Fjordlands (default temperate coastal fjords & hero spawn)
    // 2. Golden Autumn Birch & Amber Highlands (`n_continent_biome > 0.64 && lat_abs < 0.58`)
    // 3. Red-Ochre Terraced Canyon & Mesa Highlands (`n_continent_biome < 0.34 && n_moist < 0.46 && lat_abs < 0.50`)
    // 4. Crystal Glacier Tundra & Frozen Fjords (polar & sub-polar `lat_abs > 0.58`)
    // 5. Obsidian Volcanic Badlands & Magma Rift (equatorial caldera + scorched basalt highlands)
    let autumn_w = if !is_polar && !in_rift_belt && badlands_w < 0.35 && lat_abs < 0.62 {
        let b_t = smoothstep_f64(((n_continent_biome - 0.59) / 0.06).clamp(0.0, 1.0));
        let l_t = smoothstep_f64(((0.62 - lat_abs) / 0.06).clamp(0.0, 1.0));
        b_t * l_t
    } else {
        0.0
    };
    let mesa_w = if !is_polar && !in_rift_belt && badlands_w < 0.35 && lat_abs < 0.54 {
        let b_t = smoothstep_f64(((0.38 - n_continent_biome) / 0.06).clamp(0.0, 1.0));
        let m_t = smoothstep_f64(((0.50 - n_moist) / 0.06).clamp(0.0, 1.0));
        let l_t = smoothstep_f64(((0.54 - lat_abs) / 0.06).clamp(0.0, 1.0));
        b_t * m_t * l_t
    } else {
        0.0
    };
    let tundra_w = if !in_rift_belt && !is_polar && lat_abs > 0.56 {
        smoothstep_f64(((lat_abs - 0.56) / 0.16).clamp(0.0, 1.0))
    } else {
        0.0
    };

    // Smooth elevation in `z ∈ [16.0, 42.0]` (~52% curved oceans, ~48% continents & peaks)
    let mut h_norm = (n_cont - 0.515) * 2.55
        + (n_coast - 0.5) * 0.85
        + (n_ridge * n_ridge - 0.36) * 0.98
        + (n_detail - 0.5) * 0.28;

    if is_polar {
        h_norm = h_norm.max(0.20) + (lat_abs - polar_thresh) * 1.65;
    }
    if is_rift_rim {
        let rim_t = 1.0
            - ((d_lat_rift - core_half_w) / (rim_half_w - core_half_w).max(0.01)).clamp(0.0, 1.0);
        h_norm = h_norm.max(0.32) + 0.58 * rim_t * lon_taper;
    } else if is_rift_core {
        h_norm = 0.20; // Glowing lava fissure channel cleanly above sea level
    } else if badlands_w > 0.01 {
        h_norm += 0.18 * badlands_w * n_ridge;
    }

    let mut raw_surf_z = (23.5 + h_norm * 11.8).clamp(16.0, 42.0);
    // Sculpt stepped sandstone mesas in Red-Ochre Canyon & Mesa Highlands!
    if mesa_w > 0.01 && raw_surf_z > z_sea + 1.4 {
        let step_h = 2.65;
        let q = (raw_surf_z - z_sea) / step_h;
        let frac = q.fract();
        let riser = smoothstep_f64(((frac - 0.18) / 0.64).clamp(0.0, 1.0));
        let terraced = z_sea + (q.floor() + riser) * step_h;
        raw_surf_z = (raw_surf_z * (1.0 - 0.68 * mesa_w) + terraced * (0.68 * mesa_w)).clamp(16.0, 42.0);
    }

    // Wave 37: Seamless 3D spherical carved highland-to-coast river valleys!
    let n_riv_a = fbm_3d_f64(
        [dir[0] * 7.4 + 2.3, dir[1] * 7.4 - 4.1, dir[2] * 7.4 + 1.8],
        seed ^ 0x517E,
        3,
    );
    let riv_dist = (n_riv_a - 0.50).abs();
    let can_river = raw_surf_z > z_sea + 0.15 && raw_surf_z < 37.5 && !is_polar && !in_rift_belt;
    let (river_carve, river_depth) = if can_river && riv_dist < 0.030 {
        let bank_t = (1.0 - riv_dist / 0.030).clamp(0.0, 1.0);
        let smooth_bank = bank_t * bank_t * (3.0 - 2.0 * bank_t);
        let chan_t = (1.0 - riv_dist / 0.017).clamp(0.0, 1.0);
        let smooth_chan = chan_t * chan_t * (3.0 - 2.0 * chan_t);
        // Dramatic deeper carved canyon gorges in Red-Ochre Mesa Highlands!
        let canyon_boost = 1.0 + 0.72 * mesa_w;
        let carve = smooth_bank * 2.55 * canyon_boost;
        let r_depth = smooth_chan * 1.25;
        (carve, r_depth)
    } else {
        (0.0, 0.0)
    };

    // Wave 37: Deterministic natural glowing underground cavern grotto entrances on inland hillsides!
    let n_cav_grotto = value_noise_3d_f64(
        [dir[0] * 28.0 - 1.9, dir[1] * 28.0 + 3.4, dir[2] * 28.0 - 2.7],
        seed ^ 0xC47E,
    );
    let can_cavern = raw_surf_z > z_sea + 3.4
        && raw_surf_z < 35.5
        && !is_polar
        && !in_rift_belt
        && river_depth < 0.05;
    let cavern_depth = if can_cavern && n_cav_grotto > 0.905 {
        let ct = ((n_cav_grotto - 0.905) / 0.075).clamp(0.0, 1.0);
        let cs = ct * ct * (3.0 - 2.0 * ct);
        cs * 9.8
    } else {
        0.0
    };

    let surf_z = (raw_surf_z - river_carve - cavern_depth).clamp(16.0, 42.0);
    let water_depth = if cavern_depth > 1.2 {
        0.0
    } else {
        (z_sea - surf_z).max(0.0)
    };

    // Latitudinal + altitudinal equilibrium temperature (K)
    let alt_above_sea = (surf_z - z_sea).max(0.0);
    let base_temp = 298.0 - lat_abs * 30.0 - alt_above_sea * 1.15 + mesa_w * 7.5 - tundra_w * 9.0;
    let temp_k = if is_rift_core {
        1180.0
    } else if is_rift_rim {
        520.0
    } else if is_polar {
        (256.0 - (lat_abs - polar_thresh) * 24.0).clamp(242.0, 265.0)
    } else {
        (base_temp + badlands_w * 42.0).clamp(254.0, 340.0)
    };

    let continent_biome = if is_rift_core || is_rift_rim || badlands_w > 0.35 {
        CONTINENT_VOLCANIC_BADLANDS
    } else if is_polar || tundra_w > 0.45 {
        CONTINENT_CRYSTAL_TUNDRA
    } else if n_continent_biome < 0.34 && n_moist < 0.46 && lat_abs < 0.50 {
        CONTINENT_RED_OCHRE_MESA
    } else if n_continent_biome > 0.64 && lat_abs < 0.58 {
        CONTINENT_AUTUMN_HIGHLANDS
    } else {
        CONTINENT_NORDIC_EMERALD
    };

    let base_forest = if surf_z > z_sea + 1.3
        && surf_z < 35.5
        && !is_polar
        && !in_rift_belt
        && river_depth < 0.05
        && cavern_depth < 0.35
    {
        ((n_moist - 0.34) * 1.95).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let forest_density = (base_forest
        * (1.0 - 0.62 * mesa_w)
        * (1.0 - 0.85 * badlands_w)
        * (1.0 - 0.35 * tundra_w)
        + 0.18 * autumn_w * if surf_z > z_sea + 1.6 && surf_z < 34.5 { 1.0 } else { 0.0 })
        .clamp(0.0, 1.0);

    let mix3 = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
        let w = t.clamp(0.0, 1.0);
        [
            a[0] * (1.0 - w) + b[0] * w,
            a[1] * (1.0 - w) + b[1] * w,
            a[2] * (1.0 - w) + b[2] * w,
        ]
    };

    let (biome, albedo, emissive, roughness) = if is_rift_core {
        let core_heat = (1.0 - (d_lat_rift / core_half_w.max(0.005)).clamp(0.0, 1.0)) as f32;
        (
            MacroBiome::VolcanicCaldera,
            [0.14, 0.04, 0.02],
            [
                1.8 + 3.4 * core_heat,
                0.32 + 1.25 * core_heat * core_heat,
                0.04 + 0.22 * core_heat * core_heat,
            ],
            0.52,
        )
    } else if is_rift_rim {
        let rim_heat = (1.0
            - ((d_lat_rift - core_half_w) / (rim_half_w - core_half_w).max(0.01)).clamp(0.0, 1.0))
            as f32;
        (
            MacroBiome::VolcanicRim,
            [0.09, 0.085, 0.10],
            [
                0.45 * rim_heat * rim_heat,
                0.09 * rim_heat * rim_heat,
                0.01 * rim_heat,
            ],
            0.66,
        )
    } else if cavern_depth > 1.2 {
        // Bioluminescent basalt cavern grotto shaft & glowing mineral floor!
        let glow = ((cavern_depth - 1.2) / 8.0).clamp(0.0, 1.0) as f32;
        (
            MacroBiome::AlpineRock,
            [0.12 + 0.06 * glow, 0.17 + 0.18 * glow, 0.22 + 0.24 * glow],
            [0.04 * glow, 0.38 * glow, 0.56 * glow],
            0.58 - 0.25 * glow,
        )
    } else if is_polar {
        // Crystal Glacier Tundra & Frozen Fjords: glistening pale ice plateaus & frost-blue ridges
        let glint = (n_ridge * 0.55 + n_detail * 0.45) as f32;
        (
            MacroBiome::PolarGlacier,
            [0.72 + 0.16 * glint, 0.86 + 0.09 * glint, 0.97],
            [0.0; 3],
            0.20 + 0.08 * (1.0 - glint),
        )
    } else if surf_z <= z_sea {
        (
            MacroBiome::DeepOcean,
            [0.20, 0.26, 0.25],
            [0.0; 3],
            0.86,
        )
    } else if river_depth > 0.08 {
        // Wet river pebble bed (sandstone canyon pebbles in mesas, mossy slate in Nordic fjords)
        let bed = mix3([0.25, 0.27, 0.24], [0.44, 0.25, 0.15], mesa_w as f32);
        (MacroBiome::CoastalBeach, bed, [0.0; 3], 0.64)
    } else if badlands_w > 0.35 {
        // Obsidian Volcanic Badlands scorched basalt highlands apron
        let bw = badlands_w as f32;
        let vein = ((n_detail - 0.58).max(0.0) * 2.5).clamp(0.0, 1.0) as f32 * bw;
        (
            MacroBiome::VolcanicRim,
            [0.12 + 0.04 * (1.0 - bw), 0.11 + 0.04 * (1.0 - bw), 0.13 + 0.05 * (1.0 - bw)],
            [0.22 * vein, 0.04 * vein, 0.005 * vein],
            0.62,
        )
    } else if surf_z < z_sea + 1.25 {
        // Coastal beach tailored to continent (golden sand, warm amber, terracotta sandstone, or frosted pebble)
        let mut beach = [0.54, 0.44, 0.28];
        beach = mix3(beach, [0.58, 0.42, 0.22], autumn_w as f32);
        beach = mix3(beach, [0.64, 0.39, 0.21], mesa_w as f32);
        beach = mix3(beach, [0.62, 0.70, 0.78], tundra_w as f32);
        (
            MacroBiome::CoastalBeach,
            beach,
            [0.0; 3],
            0.88 - 0.24 * tundra_w as f32,
        )
    } else if surf_z >= (36.2 - tundra_w * 4.2) || (lat_abs > 0.65 && surf_z > 28.0) {
        // High alpine snow-capped summit / pale frost-blue crystal glacier ridge
        let ridge_col = mix3([0.84, 0.90, 0.95], [0.74, 0.88, 0.98], tundra_w as f32);
        (
            MacroBiome::AlpineSnow,
            ridge_col,
            [0.0; 3],
            0.30 - 0.08 * tundra_w as f32,
        )
    } else if surf_z >= 33.0 {
        // Exposed alpine crags (Nordic granite, amber crags, or stratified terracotta mesa rim)
        let band = ((surf_z * 1.15 + n_detail * 2.0).sin() * 0.5 + 0.5) as f32;
        let mesa_crag = [0.58 + 0.14 * band, 0.23 + 0.10 * band, 0.11 + 0.04 * band];
        let mut rock = [0.32, 0.34, 0.38];
        rock = mix3(rock, [0.42, 0.32, 0.22], autumn_w as f32 * 0.65);
        rock = mix3(rock, mesa_crag, mesa_w as f32);
        rock = mix3(rock, [0.54, 0.66, 0.76], tundra_w as f32 * 0.75);
        (
            MacroBiome::AlpineRock,
            rock,
            [0.0; 3],
            0.74 - 0.22 * tundra_w as f32,
        )
    } else if forest_density > 0.40 {
        // Continental forests: Nordic emerald spruce, Golden Autumn birch/amber, Red-Ochre scrub, or frost taiga
        let var = n_detail as f32;
        let mut col = [0.042, 0.175, 0.068];
        let autumn_forest = [0.46 + 0.14 * var, 0.24 + 0.11 * var, 0.055 + 0.025 * var];
        let mesa_scrub = [0.44 + 0.10 * var, 0.25 + 0.07 * var, 0.12 + 0.03 * var];
        let tundra_taiga = [0.16, 0.29, 0.31];
        col = mix3(col, autumn_forest, autumn_w as f32);
        col = mix3(col, mesa_scrub, mesa_w as f32);
        col = mix3(col, tundra_taiga, tundra_w as f32);
        (
            MacroBiome::PineForest,
            col,
            [0.0; 3],
            0.86 - 0.12 * tundra_w as f32,
        )
    } else {
        // Rolling continental highlands: Nordic emerald meadow, Golden Autumn amber meadow,
        // Red-Ochre terraced sandstone canyon plateau, or Crystal frost-lichen tundra
        let alt_t = ((surf_z - z_sea) / 10.0).clamp(0.0, 1.0) as f32;
        let band = ((surf_z * 1.15 + n_detail * 1.8).sin() * 0.5 + 0.5) as f32;
        let mut col = [0.085 + 0.11 * alt_t, 0.28 - 0.05 * alt_t, 0.082 + 0.025 * alt_t];
        let autumn_meadow = [0.44 + 0.16 * alt_t, 0.31 - 0.04 * alt_t, 0.085 + 0.02 * alt_t];
        let mesa_plateau = [0.56 + 0.14 * band, 0.24 + 0.10 * band, 0.12 + 0.04 * band];
        let tundra_lichen = [0.42 + 0.18 * alt_t, 0.56 + 0.14 * alt_t, 0.64 + 0.14 * alt_t];
        col = mix3(col, autumn_meadow, autumn_w as f32);
        col = mix3(col, mesa_plateau, mesa_w as f32);
        col = mix3(col, tundra_lichen, tundra_w as f32);
        (
            MacroBiome::EmeraldMeadow,
            col,
            [0.0; 3],
            0.82 - 0.04 * mesa_w as f32 - 0.30 * tundra_w as f32,
        )
    };

    MacroColumnSample {
        surf_z,
        z_sea,
        water_depth,
        river_depth,
        cavern_depth,
        temp_k,
        forest_density,
        moisture: n_moist,
        is_polar,
        is_rift_core,
        is_rift_rim,
        continent_biome,
        biome,
        albedo,
        emissive,
        roughness,
    }
}

/// High-frequency `1 m` 3D spherical micro-detail displacement (in meters, `±0.95 m`) added seamlessly
/// on hydrated `1 m` shards purely as a function of 3D `dir` so adjacent shards remain continuous.
#[inline]
pub fn sample_micro_relief_m(dir: [f64; 3], seed: u64) -> f64 {
    let n1 = value_noise_3d_f64([dir[0] * 68.0, dir[1] * 68.0, dir[2] * 68.0], seed ^ 0xBEEF);
    let n2 = value_noise_3d_f64(
        [dir[0] * 160.0, dir[1] * 160.0, dir[2] * 160.0],
        seed ^ 0xF00D,
    );
    (n1 - 0.5) * 1.45 + (n2 - 0.5) * 0.55
}

// =============================================================================
// 4. Orbital Floating Sky-Islands (`z = 46..58`) Shared Across Scales
// =============================================================================

/// Descriptor for a Floating Sky-Island anchored above the planet by a cool cyan Buoyant Crystal Core.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MacroSkyIsland {
    pub id: u8,
    pub name: String,
    /// Unit sphere direction of island center.
    pub center_dir: [f64; 3],
    /// Horizontal plateau radius in meters (`~24..46 m`).
    pub radius_m: f64,
    /// Lower tapered crystal keel altitude (`z_bot ≈ 46`).
    pub z_bot: f64,
    /// Upper emerald plateau altitude (`z_top ≈ 58`).
    pub z_top: f64,
    /// Buoyant Crystal Core radius in meters.
    pub core_radius_m: f64,
    /// Deterministic silhouette phase.
    pub phase: f64,
    /// True if a waterfall spills over the plateau rim into the ocean/valley below.
    pub has_waterfall: bool,
}

impl MacroSkyIsland {
    /// Generate the default planetary archipelago of 12 Orbital Floating Sky-Islands circling
    /// the equatorial & temperate latitudes (`z = 46..58`).
    pub fn default_archipelago(seed: u64) -> Vec<Self> {
        let mut islands = Vec::with_capacity(12);
        for i in 0..12usize {
            let h = hash_u64(seed.wrapping_add((i as u64 + 1) * 0x517cc1b7));
            let lon = (i as f64 / 12.0) * std::f64::consts::TAU + 0.18;
            let lat = (((h & 0xFF) as f64 / 255.0) - 0.5) * 0.62;
            let cos_lat = lat.cos();
            let dir = normalize3_f64([cos_lat * lon.cos(), lat.sin(), cos_lat * lon.sin()]);
            let radius_m = 26.0 + (((h >> 8) & 0xFF) as f64 / 255.0) * 18.0;
            islands.push(Self {
                id: i as u8,
                name: format!("AETHER-ISLE-{:02}", i + 1),
                center_dir: dir,
                radius_m,
                z_bot: 46.0 + (((h >> 16) & 0xF) as f64 / 15.0) * 2.0,
                z_top: 56.0 + (((h >> 20) & 0xF) as f64 / 15.0) * 2.0,
                core_radius_m: (radius_m * 0.34).clamp(7.0, 14.0),
                phase: (i as f64) * 1.9 + 0.5,
                has_waterfall: i % 2 == 0,
            });
        }
        islands
    }

    /// Evaluate the organic silhouette radius in meters at azimuth `ang` around the island center.
    #[inline]
    pub fn organic_radius_m(&self, ang: f64) -> f64 {
        self.radius_m
            * (1.0
                + 0.16 * (ang * 3.0 + self.phase).sin()
                + 0.10 * (ang * 5.0 - self.phase * 1.3).cos()
                + 0.05 * (ang * 7.0 + self.phase * 0.7).sin())
    }
}

// =============================================================================
// 5. GlobalClimateShell (`6 × 64²` Always-On Macro Planet Layer)
// =============================================================================

/// Per-face macro climate & surface telemetry (`res × res`, default `64 × 64`).
#[derive(Clone, Debug)]
pub struct ClimateFace {
    pub surf_z: Vec<f32>,
    pub water_depth: Vec<f32>,
    pub temp_k: Vec<f32>,
    pub wind_u: Vec<f32>,
    pub wind_v: Vec<f32>,
    pub precip: Vec<f32>,
    pub ash: Vec<f32>,
    pub albedo: Vec<[f32; 3]>,
}

impl ClimateFace {
    fn new(res: usize) -> Self {
        let len = res * res;
        Self {
            surf_z: vec![24.0; len],
            water_depth: vec![0.0; len],
            temp_k: vec![290.0; len],
            wind_u: vec![0.0; len],
            wind_v: vec![0.0; len],
            precip: vec![0.25; len],
            ash: vec![0.0; len],
            albedo: vec![[0.2, 0.3, 0.2]; len],
        }
    }
}

/// Always-on `6 × 64²` (`24,576`-column) macro surface & planetary climate shell.
#[derive(Clone, Debug)]
pub struct GlobalClimateShell {
    pub res: usize,
    pub seed: u64,
    pub faces: [ClimateFace; FACE_COUNT],
    pub tick: u64,
}

impl GlobalClimateShell {
    /// Populate the entire `6 × res²` macro shell from the shared seamless 3D `sample_macro_planet_column`.
    pub fn new(res: usize, seed: u64) -> Self {
        let r = res.max(8);
        let rf = r as f64;
        let faces = std::array::from_fn(|f| {
            let mut cf = ClimateFace::new(r);
            for y in 0..r {
                let v = ((y as f64 + 0.5) / rf) * 2.0 - 1.0;
                for x in 0..r {
                    let u = ((x as f64 + 0.5) / rf) * 2.0 - 1.0;
                    let idx = y * r + x;
                    let dir = face_uv_to_sphere_dir_f64(f, u, v);
                    let col = sample_macro_planet_column(dir, seed);
                    cf.surf_z[idx] = col.surf_z as f32;
                    cf.water_depth[idx] = col.water_depth as f32;
                    cf.temp_k[idx] = col.temp_k as f32;
                    // Hadley / trade-wind zonal circulation driven by latitude
                    let lat = dir[1];
                    let zonal = -4.5 * (lat * 3.0).cos();
                    let merid = 1.8 * (lat * 2.0).sin();
                    cf.wind_u[idx] = zonal as f32;
                    cf.wind_v[idx] = merid as f32;
                    cf.precip[idx] = (col.moisture * (1.0 - 0.45 * lat.abs())).clamp(0.0, 1.0) as f32;
                    cf.ash[idx] = if col.is_rift_core { 0.85 } else { 0.0 };
                    cf.albedo[idx] = col.albedo;
                }
            }
            cf
        });
        Self {
            res: r,
            seed,
            faces,
            tick: 0,
        }
    }

    /// Total macro columns across all 6 faces (`6 * res * res`).
    #[inline]
    pub const fn total_columns(&self) -> usize {
        FACE_COUNT * self.res * self.res
    }

    /// Advance macro solar heating, radiative cooling, wind drift, and ash settling across all 6 faces.
    pub fn step_climate(&mut self, sun_dir: [f64; 3], dt: f32) {
        let sun = normalize3_f64(sun_dir);
        let r = self.res;
        let rf = r as f64;
        let alpha = (dt * 0.08).clamp(0.005, 0.25);

        for f in 0..FACE_COUNT {
            let cf = &mut self.faces[f];
            for y in 0..r {
                let v = ((y as f64 + 0.5) / rf) * 2.0 - 1.0;
                for x in 0..r {
                    let u = ((x as f64 + 0.5) / rf) * 2.0 - 1.0;
                    let idx = y * r + x;
                    let dir = face_uv_to_sphere_dir_f64(f, u, v);
                    let cos_s = dot3_f64(dir, sun) as f32;
                    let lat_abs = dir[1].abs() as f32;

                    // Preserve incandescent volcanic rift hotspots while evolving temperate/polar climate
                    if cf.temp_k[idx] < 600.0 {
                        let target_t =
                            284.0 + 18.0 * cos_s - 34.0 * lat_abs - (cf.surf_z[idx] - 24.0).max(0.0) * 1.1;
                        cf.temp_k[idx] += (target_t.clamp(242.0, 320.0) - cf.temp_k[idx]) * alpha;
                    }
                    // Gentle ash dispersion/settling
                    cf.ash[idx] = (cf.ash[idx] * (1.0 - alpha * 0.25)).clamp(0.0, 1.0);
                    // Precip responds modestly to humidity & warmth
                    let target_p = if cf.water_depth[idx] > 0.2 {
                        0.68
                    } else {
                        (0.45 - 0.15 * lat_abs).clamp(0.1, 0.85)
                    };
                    cf.precip[idx] += (target_p - cf.precip[idx]) * (alpha * 0.2);
                }
            }
        }
        self.tick += 1;
    }

    /// Average macro climate state overlapping a shard's UV footprint on `key.face`.
    pub fn sample_shard_average(
        &self,
        key: PlanetShardKey,
        shards_per_axis: u8,
    ) -> (f32, f32, f32, f32, f32) {
        let m = shards_per_axis.max(1) as usize;
        let r = self.res;
        let x0 = (key.sx as usize * r) / m;
        let x1 = (((key.sx as usize + 1) * r) / m).max(x0 + 1).min(r);
        let y0 = (key.sy as usize * r) / m;
        let y1 = (((key.sy as usize + 1) * r) / m).max(y0 + 1).min(r);
        let cf = &self.faces[key.face as usize % FACE_COUNT];

        let mut sum_t = 0.0f32;
        let mut sum_p = 0.0f32;
        let mut sum_wu = 0.0f32;
        let mut sum_wv = 0.0f32;
        let mut sum_ash = 0.0f32;
        let mut cnt = 0.0f32;
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = y * r + x;
                sum_t += cf.temp_k[idx];
                sum_p += cf.precip[idx];
                sum_wu += cf.wind_u[idx];
                sum_wv += cf.wind_v[idx];
                sum_ash += cf.ash[idx];
                cnt += 1.0;
            }
        }
        let inv = 1.0 / cnt.max(1.0);
        (
            sum_t * inv,
            sum_p * inv,
            sum_wu * inv,
            sum_wv * inv,
            sum_ash * inv,
        )
    }
}

// =============================================================================
// 6. DualScalePlanet: On-Demand Shard Manager + Exact Cross-Shard Conservation
// =============================================================================

/// Telemetry returned by `DualScalePlanet::exchange_hydrated_shard_borders`.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ShardBorderExchangeStats {
    pub seam_pairs_exchanged: usize,
    pub cross_face_seams_exchanged: usize,
    pub seam_mass_crossed: i64,
    pub seam_energy_crossed: i64,
    pub active_seam_cells: usize,
}

/// Telemetry returned by `DualScalePlanet::step_hydrated_shards`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MacroStepStats {
    pub tick: u64,
    pub hydrated_shards: usize,
    pub global_mass_error: i64,
    pub global_energy_error: i64,
    pub seam_mass_crossed: i64,
    pub seam_energy_crossed: i64,
    pub step_ms: f64,
}

/// Construct an empty `VolumeField` configured with `BaseAnchorMode::RadialInnerShell`.
fn new_empty_radial_shard(shard_size: usize, depth: usize, seed: u64) -> VolumeField {
    let d = depth.max(8);
    let layers: Vec<Field> = (0..d)
        .map(|_| Field::new_empty(shard_size, shard_size))
        .collect();
    let vlen = d * shard_size * shard_size;
    let mut structural = StructuralState::default();
    structural.anchor_mode = BaseAnchorMode::RadialInnerShell;
    let mut vf = VolumeField {
        width: shard_size,
        height: shard_size,
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

/// Dual-Scale Locally-Flat Cubed-Sphere Planet combining an always-on `6 × 64²` `GlobalClimateShell`
/// with lazily hydrated `256×256×64` `1 m` `VolumeField` shards and bit-exact conservation across
/// all same-face and rotated cube-face shard seams.
#[derive(Clone, Debug)]
pub struct DualScalePlanet {
    pub config: MacroPlanetConfig,
    pub climate: GlobalClimateShell,
    pub shards: HashMap<PlanetShardKey, VolumeField>,
    pub sky_islands: Vec<MacroSkyIsland>,
    pub sun_dir: [f64; 3],
    pub tick: u64,
    pub initial_hydrated_mass: i64,
    pub initial_hydrated_energy: i64,
    pub total_seam_mass_crossed: i64,
    pub total_seam_energy_crossed: i64,
}

impl DualScalePlanet {
    /// Construct a new Dual-Scale Planet (only allocates the lightweight `6 × 64²` macro climate
    /// shell upfront; `1 m` shards are lazily hydrated on demand!).
    pub fn new(config: MacroPlanetConfig) -> Self {
        let climate = GlobalClimateShell::new(config.climate_res, config.seed);
        let sky_islands = MacroSkyIsland::default_archipelago(config.seed);
        Self {
            config,
            climate,
            shards: HashMap::new(),
            sky_islands,
            sun_dir: normalize3_f64([-0.74, 0.34, 0.58]),
            tick: 0,
            initial_hydrated_mass: 0,
            initial_hydrated_energy: 0,
            total_seam_mass_crossed: 0,
            total_seam_energy_crossed: 0,
        }
    }

    /// Construct a Dual-Scale Planet with default `R = 1,300 m`, `6 × 2048² × 64` (`1.61B` virtual
    /// voxels, `384` shards) configuration.
    pub fn new_default_1300m() -> Self {
        Self::new(MacroPlanetConfig::default())
    }

    /// Re-snapshot exact global conservation baseline across all currently hydrated shards.
    pub fn snapshot_conservation_baseline(&mut self) {
        for vf in self.shards.values_mut() {
            vf.snapshot_conservation_baseline();
            vf.snapshot_prev_sensors();
        }
        self.initial_hydrated_mass = self.current_hydrated_mass();
        self.initial_hydrated_energy = self.current_hydrated_energy();
    }

    /// Sum of accounted integer mass across all currently hydrated shards.
    pub fn current_hydrated_mass(&self) -> i64 {
        self.shards.values().map(|vf| vf.current_accounted_mass()).sum()
    }

    /// Sum of accounted integer thermal + chemical energy across all currently hydrated shards.
    pub fn current_hydrated_energy(&self) -> i64 {
        self.shards
            .values()
            .map(|vf| vf.current_accounted_energy())
            .sum()
    }

    /// Exact global mass conservation error across all hydrated shards (MUST equal `0`).
    pub fn global_mass_error(&self) -> i64 {
        self.current_hydrated_mass() - self.initial_hydrated_mass
    }

    /// Exact global energy conservation error across all hydrated shards (MUST equal `0`).
    pub fn global_energy_error(&self) -> i64 {
        self.current_hydrated_energy() - self.initial_hydrated_energy
    }

    /// Hydrate one `shard_size × shard_size × depth` `VolumeField` shard from the shared seamless
    /// 3D spherical procedural field (`sample_macro_planet_column` + `1 m` micro-detail + any
    /// overlapping Orbital Floating Sky-Islands with `BuoyantCore`s).
    ///
    /// Returns `true` if the shard was newly hydrated, `false` if already resident.
    pub fn hydrate_shard(&mut self, key: PlanetShardKey) -> bool {
        if self.shards.contains_key(&key) {
            return false;
        }
        let vf = self.generate_shard_volume(key);
        self.initial_hydrated_mass += vf.current_accounted_mass();
        self.initial_hydrated_energy += vf.current_accounted_energy();
        self.shards.insert(key, vf);
        true
    }

    /// Evict a hydrated shard cleanly while keeping global conservation baseline exact for all
    /// remaining hydrated shards.
    pub fn evict_shard(&mut self, key: PlanetShardKey) -> Option<VolumeField> {
        let removed = self.shards.remove(&key)?;
        self.initial_hydrated_mass -= removed.current_accounted_mass();
        self.initial_hydrated_energy -= removed.current_accounted_energy();
        Some(removed)
    }

    /// Lazily hydrate all shards within `radius_shards` topological hops (including across cube-face
    /// seams!) of virtual coordinate `(face, gx, gy)`, and evict shards strictly outside that ring.
    pub fn hydrate_around(
        &mut self,
        face: FaceId,
        gx: usize,
        gy: usize,
        radius_shards: u8,
    ) -> Vec<PlanetShardKey> {
        let m = self.config.shards_per_face_axis.max(1);
        let s = self.config.shard_size.max(1);
        let center = PlanetShardKey {
            face: (face % FACE_COUNT) as u8,
            sx: (gx / s).min(m as usize - 1) as u8,
            sy: (gy / s).min(m as usize - 1) as u8,
        };

        let mut keep: HashSet<PlanetShardKey> = HashSet::new();
        let mut queue: VecDeque<(PlanetShardKey, u8)> = VecDeque::new();
        keep.insert(center);
        queue.push_back((center, 0));

        while let Some((cur, dist)) = queue.pop_front() {
            if dist >= radius_shards {
                continue;
            }
            for edge in 0..EDGE_COUNT {
                let nb = shard_neighbor(cur, edge, m).neighbor;
                if keep.insert(nb) {
                    queue.push_back((nb, dist + 1));
                }
            }
        }

        // Evict shards outside the active observer neighborhood
        let to_evict: Vec<PlanetShardKey> = self
            .shards
            .keys()
            .copied()
            .filter(|k| !keep.contains(k))
            .collect();
        for k in to_evict {
            let _ = self.evict_shard(k);
        }

        // Hydrate all desired shards in deterministic key order
        let mut desired: Vec<PlanetShardKey> = keep.into_iter().collect();
        desired.sort();
        for &k in &desired {
            self.hydrate_shard(k);
        }
        desired
    }

    /// Generate one `shard_size × shard_size × depth` `1 m` radial `VolumeField` shard from the
    /// shared 3D spherical procedural field.
    fn generate_shard_volume(&self, key: PlanetShardKey) -> VolumeField {
        let s = self.config.shard_size;
        let d = self.config.depth.max(16);
        let n_virt = self.config.face_size_virtual();
        let seed = self.config.seed;
        let z_scale = d as f64 / 64.0;
        let z_core_hi = ((4.0 * z_scale).round() as usize).clamp(1, d - 6);
        let z_cav_lo = (z_core_hi + 1).min(d - 5);
        let z_cav_hi = ((15.0 * z_scale).round() as usize).clamp(z_cav_lo, d - 5);
        let z_sea = ((self.config.z_sea as f64) * z_scale)
            .round()
            .clamp((z_cav_hi + 1) as f64, (d - 4) as f64) as usize;
        let z_peak_max = ((42.0 * z_scale).round() as usize).clamp(z_sea + 2, d - 4);

        let shard_seed = seed
            ^ ((key.face as u64) << 40)
            ^ ((key.sx as u64) << 20)
            ^ (key.sy as u64);
        let mut vf = new_empty_radial_shard(s, d, shard_seed);
        vf.structural.anchor_mode = BaseAnchorMode::RadialInnerShell;

        for ly in 0..s {
            let gy = (key.sy as usize * s + ly) as f64 + 0.5;
            for lx in 0..s {
                let gx = (key.sx as usize * s + lx) as f64 + 0.5;
                let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
                let dir = face_uv_to_sphere_dir_f64(key.face as usize, u, v);
                let col = sample_macro_planet_column(dir, seed);
                let micro = sample_micro_relief_m(dir, seed);
                let idx = ly * s + lx;

                let surf_z_f = (col.surf_z + micro * if col.is_rift_core { 0.2 } else { 0.85 })
                    * z_scale;
                let surf_z = (surf_z_f.round() as usize).clamp(z_cav_hi + 1, z_peak_max);
                let is_land = surf_z > z_sea || col.is_polar || col.is_rift_core;

                // Deterministic tree placement hash in 3D sphere space
                let tree_n = value_noise_3d_f64(
                    [dir[0] * 320.0, dir[1] * 320.0, dir[2] * 320.0],
                    seed ^ 0x72EE,
                );

                for z in 0..=z_peak_max {
                    let layer = &mut vf.layers[z];
                    layer.elevation[idx] = z.min(63) as u8;

                    if z <= z_core_hi {
                        layer.mass[SP_ROCK][idx] = 1050;
                        let core_t = if col.is_rift_core { 76_000 } else { 36_000 };
                        layer.set_cell_temp(idx, core_t);
                    } else if z <= z_cav_hi {
                        let cav_p = [
                            dir[0] * 42.0 + z as f64 * 0.25,
                            dir[1] * 42.0 - z as f64 * 0.21,
                            dir[2] * 42.0 + z as f64 * 0.19,
                        ];
                        let n_cav = value_noise_3d_f64(cav_p, seed ^ 0xCAFE);
                        let grotto_chamber = col.cavern_depth > 0.8 && z > z_cav_lo + 1 && z < z_cav_hi;
                        let in_cavern = grotto_chamber || (z > z_cav_lo + 1 && z + 1 < z_cav_hi && n_cav > 0.68);
                        if !in_cavern {
                            layer.mass[SP_ROCK][idx] = 1000;
                            layer.set_cell_temp(idx, if col.is_rift_core { 58_000 } else { 30_500 });
                        } else if z <= z_cav_lo + 3 && !col.is_rift_core {
                            layer.mass[SP_WATER][idx] = 620;
                            layer.set_cell_temp(idx, 29_200);
                        }
                    } else if col.is_rift_core {
                        let z_lava = (z_sea + 1).min(z_peak_max);
                        if z < z_lava {
                            layer.mass[SP_ROCK][idx] = 980;
                            layer.set_cell_temp(idx, 94_000);
                        } else if z == z_lava {
                            layer.mass[SP_ROCK][idx] = 680;
                            layer.mass[SP_OIL][idx] = 500;
                            layer.set_cell_temp(idx, 118_000);
                        } else if z <= z_lava + 3 {
                            let fade = (z - z_lava) as u16;
                            layer.mass[SP_STEAM][idx] = 300u16.saturating_sub(fade * 65);
                            layer.mass[SP_SMOKE][idx] = 200u16.saturating_sub(fade * 45);
                            layer.set_cell_temp(idx, 42_000);
                        }
                    } else if col.is_polar {
                        if z + 2 < surf_z {
                            layer.mass[SP_ROCK][idx] = 980;
                            layer.set_cell_temp(idx, 26_600);
                        } else if z <= surf_z {
                            layer.mass[SP_ICE][idx] = 980;
                            layer.set_cell_temp(idx, 25_800);
                        }
                    } else if is_land {
                        if z < surf_z {
                            layer.mass[SP_ROCK][idx] = 980;
                            layer.set_cell_temp(idx, if col.is_rift_rim { 46_000 } else { 28_900 });
                        } else if z == surf_z {
                            if col.is_rift_rim {
                                layer.mass[SP_ROCK][idx] = 960;
                                layer.set_cell_temp(idx, 52_000);
                            } else if col.biome == MacroBiome::AlpineSnow {
                                layer.mass[SP_ROCK][idx] = 620;
                                layer.mass[SP_ICE][idx] = 480;
                                layer.set_cell_temp(idx, 26_600);
                            } else if col.biome == MacroBiome::AlpineRock {
                                layer.mass[SP_ROCK][idx] = 960;
                                layer.set_cell_temp(idx, 28_200);
                            } else if col.biome == MacroBiome::CoastalBeach {
                                layer.mass[SP_SOIL][idx] = 920;
                                layer.set_cell_temp(idx, 29_300);
                            } else {
                                layer.mass[SP_SOIL][idx] = 860;
                                layer.mass[SP_PLANT][idx] = 420;
                                if col.forest_density > 0.35 && tree_n > 0.72 {
                                    layer.mass[SP_WOOD][idx] = 620;
                                }
                                layer.set_cell_temp(idx, (col.temp_k * 100.0) as i64);
                            }
                        } else if col.river_depth > 0.15 && z == surf_z + 1 && z < d - 2 {
                            // Flowing alpine & coastal river water!
                            layer.mass[SP_WATER][idx] = 820;
                            layer.flow_e[idx] = 36;
                            layer.flow_s[idx] = 18;
                            layer.set_cell_temp(idx, (col.temp_k * 100.0) as i64);
                        } else if z <= surf_z + 3
                            && col.forest_density > 0.42
                            && tree_n > 0.80
                            && z < d - 2
                        {
                            // Multi-voxel `1 m` pine tree trunk & emerald canopy!
                            if z == surf_z + 1 {
                                layer.mass[SP_WOOD][idx] = 740;
                            } else {
                                layer.mass[SP_PLANT][idx] = 560;
                                layer.mass[SP_WOOD][idx] = 220;
                            }
                            layer.set_cell_temp(idx, (col.temp_k * 100.0) as i64);
                        }
                    } else if z <= surf_z {
                        layer.mass[SP_ROCK][idx] = 940;
                        layer.mass[SP_SOIL][idx] = 200;
                        layer.set_cell_temp(idx, 28_600);
                    } else if z <= z_sea {
                        layer.mass[SP_WATER][idx] = 1020;
                        layer.flow_e[idx] = 24;
                        layer.set_cell_temp(idx, 28_700);
                    }
                }
            }
        }

        // Stamp any overlapping Orbital Floating Sky-Islands (`z = 46..58`) + register `BuoyantCore`s
        if d >= 24 {
            let r_surf = self.config.surface_radius_m();
            let shard_center = key.center_sphere_dir(self.config.shards_per_face_axis);
            let shard_half_ang =
                (std::f64::consts::FRAC_PI_2 / self.config.shards_per_face_axis.max(1) as f64)
                    * 1.15;

            for isl in &self.sky_islands {
                let cos_d = dot3_f64(shard_center, isl.center_dir).clamp(-1.0, 1.0);
                if cos_d.acos() > shard_half_ang + (isl.radius_m * 1.4 / r_surf) {
                    continue;
                }

                // Local tangent orthonormal basis around the island center
                let ref_up = if isl.center_dir[1].abs() < 0.9 {
                    [0.0, 1.0, 0.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let t1 = normalize3_f64([
                    ref_up[1] * isl.center_dir[2] - ref_up[2] * isl.center_dir[1],
                    ref_up[2] * isl.center_dir[0] - ref_up[0] * isl.center_dir[2],
                    ref_up[0] * isl.center_dir[1] - ref_up[1] * isl.center_dir[0],
                ]);
                let t2 = normalize3_f64([
                    isl.center_dir[1] * t1[2] - isl.center_dir[2] * t1[1],
                    isl.center_dir[2] * t1[0] - isl.center_dir[0] * t1[2],
                    isl.center_dir[0] * t1[1] - isl.center_dir[1] * t1[0],
                ]);

                let z_lo = ((isl.z_bot * z_scale).round() as usize).clamp(z_peak_max + 2, d - 3);
                let z_hi = ((isl.z_top * z_scale).round() as usize).clamp(z_lo + 2, d - 2);
                let z_core = z_lo as f64 + (z_hi - z_lo) as f64 * 0.28;

                let mut core_mass = 0i64;
                let mut sum_lx = 0.0f64;
                let mut sum_ly = 0.0f64;
                let mut core_cnt = 0.0f64;

                for ly in 0..s {
                    let gy = (key.sy as usize * s + ly) as f64 + 0.5;
                    for lx in 0..s {
                        let gx = (key.sx as usize * s + lx) as f64 + 0.5;
                        let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
                        let dir = face_uv_to_sphere_dir_f64(key.face as usize, u, v);
                        let dx_m = dot3_f64(dir, t1) * r_surf;
                        let dy_m = dot3_f64(dir, t2) * r_surf;
                        let dist_m = (dx_m * dx_m + dy_m * dy_m).sqrt();
                        if dist_m > isl.radius_m * 1.35 {
                            continue;
                        }
                        let ang = dy_m.atan2(dx_m);
                        let rim_r = isl.organic_radius_m(ang);
                        let idx = ly * s + lx;

                        for z in z_lo..=z_hi {
                            let tz = (z - z_lo) as f64 / (z_hi - z_lo).max(1) as f64;
                            let layer_r = rim_r * (0.24 + 0.78 * tz.powf(0.55));
                            if dist_m <= layer_r {
                                let dz = (z as f64 + 0.5) - z_core;
                                let in_core = z < z_hi
                                    && (dx_m * dx_m + dy_m * dy_m + (dz * 2.0) * (dz * 2.0)
                                        <= isl.core_radius_m * isl.core_radius_m);
                                let layer = &mut vf.layers[z];
                                if in_core {
                                    layer.mass[SP_ROCK][idx] = 740;
                                    layer.mass[SP_ICE][idx] = 360;
                                    layer.set_cell_temp(idx, 26_600);
                                    core_mass += 1100;
                                    sum_lx += lx as f64 + 0.5;
                                    sum_ly += ly as f64 + 0.5;
                                    core_cnt += 1.0;
                                } else if z == z_hi {
                                    layer.mass[SP_ROCK][idx] = 520;
                                    layer.mass[SP_SOIL][idx] = 440;
                                    layer.mass[SP_PLANT][idx] = 420;
                                    if dist_m < layer_r * 0.58 {
                                        layer.mass[SP_WOOD][idx] = 340;
                                    }
                                    layer.set_cell_temp(idx, 28_800);
                                } else {
                                    layer.mass[SP_ROCK][idx] = 920;
                                    layer.set_cell_temp(idx, 27_800);
                                }
                            }
                        }

                        // Plunging Sky-Island waterfall cascade from plateau rim down into bay!
                        if isl.has_waterfall && (ang - 0.6).abs() < 0.14 && (dist_m - rim_r * 0.94).abs() < 2.2 {
                            for z in (z_sea + 1)..=z_hi {
                                let layer = &mut vf.layers[z];
                                if layer.mass[SP_ROCK][idx] < 500 {
                                    layer.mass[SP_WATER][idx] = 720;
                                    layer.set_cell_temp(idx, 28_500);
                                }
                            }
                        }
                    }
                }

                if core_mass > 0 && core_cnt > 0.0 {
                    vf.structural.buoyant_cores.push(BuoyantCore {
                        id: isl.id,
                        name: format!("{} {}", face_name(key.face as usize), isl.name),
                        cx: (sum_lx / core_cnt) as f32,
                        cy: (sum_ly / core_cnt) as f32,
                        cz: z_core as f32,
                        radius: isl.core_radius_m as f32,
                        island_radius: (isl.radius_m * 1.15) as f32,
                        init_core_mass: core_mass,
                        current_core_mass: core_mass,
                        integrity: 1.0,
                        anchor_strength: 255,
                        collapsed: false,
                        collapse_ticks: 0,
                        announced_fracture: false,
                        announced_impact: false,
                    });
                }
            }
        }

        vf.structural.dirty = false;
        vf.snapshot_conservation_baseline();
        vf.snapshot_prev_sensors();
        vf
    }

    /// Exact antisymmetric conservative flux exchange across ALL adjacent hydrated shard pairs —
    /// BOTH same-face borders AND rotated cube-face borders!
    ///
    /// Exchanges `SP_WATER`, `SP_OIL`, `SP_STEAM`, `SP_SMOKE`, exact proportional advected enthalpy,
    /// and discrete 2nd-law heat conduction (even through rock) with **`global_mass_error() == 0 &&
    /// global_energy_error() == 0`** bit-exactly!
    pub fn exchange_hydrated_shard_borders(&mut self) -> ShardBorderExchangeStats {
        let m = self.config.shards_per_face_axis.max(1);
        let s = self.config.shard_size;
        let d = self.config.depth;
        let mut stats = ShardBorderExchangeStats::default();

        // Deterministic canonical list of undirected adjacent hydrated shard pairs (`key_a < key_b`)
        let mut keys: Vec<PlanetShardKey> = self.shards.keys().copied().collect();
        keys.sort();

        let mut pairs: Vec<(PlanetShardKey, EdgeId, ShardEdgeLink)> = Vec::new();
        for &key_a in &keys {
            for edge_a in 0..EDGE_COUNT {
                let link_ab = shard_neighbor(key_a, edge_a, m);
                if key_a < link_ab.neighbor && self.shards.contains_key(&link_ab.neighbor) {
                    pairs.push((key_a, edge_a, link_ab));
                }
            }
        }

        for (key_a, edge_a, link_ab) in pairs {
            let key_b = link_ab.neighbor;
            let edge_b = link_ab.edge;
            let link_ba = shard_neighbor(key_b, edge_b, m);
            let out_a = edge_outward_uv(edge_a);
            let out_b = edge_outward_uv(edge_b);

            let Some(mut vf_b) = self.shards.remove(&key_b) else {
                continue;
            };
            {
                let vf_a = self
                    .shards
                    .get_mut(&key_a)
                    .expect("key_a must exist during border exchange");

                for z in 0..d.min(vf_a.depth).min(vf_b.depth) {
                    let fa = &mut vf_a.layers[z];
                    let fb = &mut vf_b.layers[z];

                    for k in 0..s {
                        let (xa, ya) = edge_cell_coords(edge_a, k, s);
                        let (xb, yb) = link_ab.map_edge_cell(k, s);
                        let idx_a = ya * s + xa;
                        let idx_b = yb * s + xb;

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

                        // 1. Mobile fluid & gas transfer across shard border (A -> B positive)
                        let mut q = [0i32; NUM_MOBILE];
                        if !wall_a && !wall_b {
                            let wa = fa.mass[SP_WATER][idx_a] as f32;
                            let wb = fb.mass[SP_WATER][idx_b] as f32;
                            let oa = fa.mass[SP_OIL][idx_a] as f32;
                            let ob = fb.mass[SP_OIL][idx_b] as f32;
                            let liq_a = wa + oa;
                            let liq_b = wb + ob;

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
                            for mi in 0..NUM_MOBILE {
                                let sp = MOBILE_SPECIES[mi];
                                let q_int = raw_q[mi].round_ties_even() as i32;
                                q[mi] = if q_int > 0 {
                                    q_int.min((fa.mass[sp][idx_a] / 4) as i32)
                                } else if q_int < 0 {
                                    q_int.max(-((fb.mass[sp][idx_b] / 4) as i32))
                                } else {
                                    0
                                };
                            }

                            // Volume ceiling on receiver B (positive flows)
                            let pos_tot: i64 =
                                q.iter().filter(|&&v| v > 0).map(|&v| v as i64).sum();
                            let cap_room_b = (H_CAP.saturating_sub(vol_b) / 4) as i64;
                            if pos_tot > cap_room_b && pos_tot > 0 {
                                for v in &mut q {
                                    if *v > 0 {
                                        *v = ((*v as i64) * cap_room_b / pos_tot) as i32;
                                    }
                                }
                            }

                            // Volume ceiling on receiver A (negative flows)
                            let neg_tot: i64 =
                                q.iter().filter(|&&v| v < 0).map(|&v| (-v) as i64).sum();
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
                        for mi in 0..NUM_MOBILE {
                            let qv = q[mi] as i64;
                            if qv == 0 {
                                continue;
                            }
                            let sp = MOBILE_SPECIES[mi];
                            let cs = SPECIES[sp].c;
                            delta_c_a -= qv * cs;
                            delta_c_b += qv * cs;
                            if qv > 0 && cap_a > 0 {
                                advected_e += ((qv as i128)
                                    * (cs as i128)
                                    * (fa.energy[idx_a] as i128)
                                    / (cap_a as i128))
                                    as i64;
                            } else if qv < 0 && cap_b > 0 {
                                let q_abs = -qv;
                                advected_e -= ((q_abs as i128)
                                    * (cs as i128)
                                    * (fb.energy[idx_b] as i128)
                                    / (cap_b as i128))
                                    as i64;
                            }
                        }

                        // 3. Exact discrete 2nd-law heat conduction across shard border
                        let mut conducted_e = 0i64;
                        let dt = temp_a - temp_b;
                        if cap_a > 0 && cap_b > 0 && dt != 0 {
                            let k_a = weighted_conductivity(fa, idx_a);
                            let k_b = weighted_conductivity(fb, idx_b);
                            let k_eff = 2.0 * k_a * k_b / (k_a + k_b + 1e-6);
                            let c_eff =
                                (cap_a as f64 * cap_b as f64) / ((cap_a + cap_b) as f64);
                            let g_raw = (0.012 * k_eff * c_eff * (dt as f64)).round() as i64;
                            let max_g = ((dt.unsigned_abs() as u128)
                                * (cap_a as u128)
                                * (cap_b as u128)
                                / (4 * ((cap_a + cap_b) as u128)))
                                as i64;
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

                        // 4. Strictly antisymmetric mass & thermal energy updates (`a -= x; b += x`)
                        let mut mass_moved = 0i64;
                        for mi in 0..NUM_MOBILE {
                            let qv = q[mi];
                            if qv != 0 {
                                let sp = MOBILE_SPECIES[mi];
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

                        // 5. Rotate & couple tangent flow vectors across the shard border
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
                            let new_fe_a = (liq_q * out_a[0] as i32
                                + tang_a * along_a[0] as i32)
                                .clamp(i16::MIN as i32, i16::MAX as i32)
                                as i16;
                            let new_fs_a = (liq_q * out_a[1] as i32
                                + tang_a * along_a[1] as i32)
                                .clamp(i16::MIN as i32, i16::MAX as i32)
                                as i16;
                            fa.flow_e[idx_a] = new_fe_a;
                            fa.flow_s[idx_a] = new_fs_a;

                            let (new_fe_b, new_fs_b) =
                                link_ab.rotate_flow_i16(new_fe_a, new_fs_a);
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
            self.shards.insert(key_b, vf_b);
            stats.seam_pairs_exchanged += 1;
            if link_ab.crosses_cube_face {
                stats.cross_face_seams_exchanged += 1;
            }
        }

        self.total_seam_mass_crossed += stats.seam_mass_crossed;
        self.total_seam_energy_crossed += stats.seam_energy_crossed;
        stats
    }

    /// Macro -> Micro AMR coupling: modulate shard `WorldLaws` and apply conservative atmospheric
    /// thermal forcing from overlapping `GlobalClimateShell` cells (every joule tracked in
    /// `faucet_energy` so **`global_energy_error() == 0`** bit-exactly!).
    pub fn apply_climate_to_shard(&mut self, key: PlanetShardKey) {
        let (avg_temp_k, avg_precip, _wu, _wv, _ash) = self
            .climate
            .sample_shard_average(key, self.config.shards_per_face_axis);
        let Some(vf) = self.shards.get_mut(&key) else {
            return;
        };

        vf.world_laws.ambient_temp_shift = ((avg_temp_k - 290.0) * 100.0).clamp(-5500.0, 5500.0);
        vf.world_laws.precipitation = (0.45 + avg_precip * 1.65).clamp(0.25, 2.8);

        // Conservative strided atmospheric nudging on topmost exposed surface voxels
        let w = vf.width;
        let h = vf.height;
        let d = vf.depth;
        let target_t = ((avg_temp_k * 100.0) as i64).clamp(T_MIN + 2000, T_MAX - 2000);
        let stride = 4usize;
        let phase = (self.tick as usize) % stride;
        let cap_z = d.saturating_sub(2);

        for y in 0..h {
            for x in 0..w {
                if ((x + y) % stride) != phase {
                    continue;
                }
                let idx = y * w + x;
                for z in (0..cap_z).rev() {
                    let layer = &mut vf.layers[z];
                    if layer.cell_volume(idx) >= 80 {
                        let c = layer.cell_heat_capacity(idx);
                        if c > 0 {
                            let cur_t = layer.cell_temp_ck(idx);
                            // Don't overwrite active volcanic magma
                            if cur_t > 70_000 && layer.mass[SP_ROCK][idx] >= 400 {
                                break;
                            }
                            let diff_t = target_t - cur_t;
                            let nudge_t = (diff_t / 12).clamp(-300, 300);
                            if nudge_t != 0 {
                                let desired_e = layer.energy[idx] + nudge_t * c;
                                let clamped_e = desired_e.clamp(c * T_MIN, c * T_MAX);
                                let delta = clamped_e - layer.energy[idx];
                                if delta != 0 {
                                    layer.energy[idx] += delta;
                                    layer.faucet_energy += delta;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    /// Micro -> Macro AMR upscaling: aggregate hydrated shard surface temperature and airborne
    /// smoke/ash back into overlapping `GlobalClimateShell` cells!
    pub fn upscale_shard_to_climate(&mut self, key: PlanetShardKey) {
        let Some(vf) = self.shards.get(&key) else {
            return;
        };
        let m = self.config.shards_per_face_axis.max(1) as usize;
        let r = self.climate.res;
        let w = vf.width;
        let h = vf.height;
        let d = vf.depth;
        let x0 = (key.sx as usize * r) / m;
        let x1 = (((key.sx as usize + 1) * r) / m).max(x0 + 1).min(r);
        let y0 = (key.sy as usize * r) / m;
        let y1 = (((key.sy as usize + 1) * r) / m).max(y0 + 1).min(r);

        let nx_c = (x1 - x0).max(1);
        let ny_c = (y1 - y0).max(1);
        let cf = &mut self.climate.faces[key.face as usize % FACE_COUNT];

        for cy in y0..y1 {
            let ly0 = ((cy - y0) * h) / ny_c;
            let ly1 = (((cy - y0 + 1) * h) / ny_c).max(ly0 + 1).min(h);
            for cx in x0..x1 {
                let lx0 = ((cx - x0) * w) / nx_c;
                let lx1 = (((cx - x0 + 1) * w) / nx_c).max(lx0 + 1).min(w);

                let mut sum_t = 0.0f64;
                let mut sum_ash = 0.0f64;
                let mut cnt = 0.0f64;
                for ly in ly0..ly1 {
                    for lx in lx0..lx1 {
                        let idx = ly * w + lx;
                        // Topmost exposed solid/liquid temperature + upper smoke/ash plume
                        let mut found = false;
                        for z in (0..d).rev() {
                            let l = &vf.layers[z];
                            sum_ash +=
                                (l.mass[SP_SMOKE][idx] as f64 + l.mass[SP_ASH][idx] as f64)
                                    / 2000.0;
                            if !found && l.cell_volume(idx) >= 80 {
                                sum_t += l.cell_temp_ck(idx) as f64 * 0.01;
                                found = true;
                                break;
                            }
                        }
                        if found {
                            cnt += 1.0;
                        }
                    }
                }
                if cnt > 0.0 {
                    let c_idx = cy * r + cx;
                    let micro_t = (sum_t / cnt) as f32;
                    let micro_ash = (sum_ash / cnt).clamp(0.0, 1.0) as f32;
                    cf.temp_k[c_idx] = cf.temp_k[c_idx] * 0.70 + micro_t * 0.30;
                    cf.ash[c_idx] = (cf.ash[c_idx] * 0.75 + micro_ash * 0.25).clamp(0.0, 1.0);
                }
            }
        }
    }

    /// Run one complete conservation-exact dual-scale physics tick across all hydrated shards +
    /// macro climate shell:
    /// 1. 3D RefSim + Projection Firewall interior step on every hydrated shard
    /// 2. Antisymmetric cross-shard border exchange (same-face + rotated cube-face seams)
    /// 3. Bidirectional Macro ↔ Micro climate coupling (`apply_climate_to_shard`,
    ///    `upscale_shard_to_climate`, `climate.step_climate`)
    pub fn step_hydrated_shards(&mut self) -> MacroStepStats {
        let t0 = Instant::now();

        // 1. Step all hydrated shards independently (parallel when > 1 shard)
        let mut keys: Vec<PlanetShardKey> = self.shards.keys().copied().collect();
        keys.sort();
        if self.shards.len() <= 1 {
            for vf in self.shards.values_mut() {
                let raw = vf.refsim_raw_outputs_all();
                let _ = vf.step_3d(&raw);
            }
        } else {
            let mut vals: Vec<(PlanetShardKey, VolumeField)> = keys
                .iter()
                .filter_map(|k| self.shards.remove(k).map(|v| (*k, v)))
                .collect();
            std::thread::scope(|sc| {
                for (_, vf) in &mut vals {
                    sc.spawn(move || {
                        let raw = vf.refsim_raw_outputs_all();
                        let _ = vf.step_3d(&raw);
                    });
                }
            });
            for (k, v) in vals {
                self.shards.insert(k, v);
            }
        }

        // 2. Exact antisymmetric cross-shard border conservation
        let ex = self.exchange_hydrated_shard_borders();

        // 3. Bidirectional Macro ↔ Micro AMR coupling
        for &k in &keys {
            self.apply_climate_to_shard(k);
            self.upscale_shard_to_climate(k);
        }
        self.climate.step_climate(self.sun_dir, 0.1);

        self.tick += 1;
        MacroStepStats {
            tick: self.tick,
            hydrated_shards: self.shards.len(),
            global_mass_error: self.global_mass_error(),
            global_energy_error: self.global_energy_error(),
            seam_mass_crossed: ex.seam_mass_crossed,
            seam_energy_crossed: ex.seam_energy_crossed,
            step_ms: t0.elapsed().as_secs_f64() * 1000.0,
        }
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
// Unit Tests (Wave 33B Comprehensive Verification)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_384_shard_cubed_sphere_topology_exact_involution() {
        let cfg = MacroPlanetConfig::default();
        let m = cfg.shards_per_face_axis;
        assert_eq!(m, 8);
        assert_eq!(cfg.total_shards(), 384);
        assert_eq!(cfg.directed_shard_edges(), 1536);
        assert_eq!(cfg.face_size_virtual(), 2048);
        assert_eq!(cfg.virtual_voxels(), 1_610_612_736);

        let s = 16usize; // sample 16 cells along every shard edge for 3D coincidence check
        let n_virt = (m as usize) * s;
        let mut total_directed = 0usize;
        let mut interior_directed = 0usize;
        let mut cross_face_directed = 0usize;

        for f in 0..FACE_COUNT as u8 {
            for sy in 0..m {
                for sx in 0..m {
                    let key = PlanetShardKey::new(f, sx, sy);
                    for edge in 0..EDGE_COUNT {
                        let ab = shard_neighbor(key, edge, m);
                        let ba = shard_neighbor(ab.neighbor, ab.edge, m);

                        // 1. Exact involution
                        assert_eq!(ba.neighbor, key, "involution key mismatch on ({key:?}, {edge})");
                        assert_eq!(ba.edge, edge, "involution edge mismatch on ({key:?}, {edge})");
                        assert_eq!(
                            ab.flip_along, ba.flip_along,
                            "flip_along must be symmetric on ({key:?}, {edge})"
                        );
                        assert_eq!(ab.crosses_cube_face, ba.crosses_cube_face);

                        // 2. Proper right-handed rotation (`det == +1`) & exact inverse (`R_BA * R_AB == I`)
                        assert_eq!(ab.det(), 1, "rot det must be +1 on ({key:?}, {edge})");
                        for r in 0..2 {
                            for c in 0..2 {
                                let prod =
                                    ba.rot[r][0] * ab.rot[0][c] + ba.rot[r][1] * ab.rot[1][c];
                                let want = if r == c { 1 } else { 0 };
                                assert_eq!(prod, want, "R_BA * R_AB != I on ({key:?}, {edge})");
                            }
                        }

                        // 3. Exact 3D sphere-direction coincidence along the shared boundary edge!
                        for k in [0usize, s / 2, s - 1] {
                            let (xa, ya) = edge_cell_coords(edge, k, s);
                            let (xb, yb) = ab.map_edge_cell(k, s);

                            let (gxa, gya) = match edge {
                                EDGE_POS_U => ((sx as usize * s + xa + 1) as f64, (sy as usize * s + ya) as f64 + 0.5),
                                EDGE_NEG_U => ((sx as usize * s + xa) as f64, (sy as usize * s + ya) as f64 + 0.5),
                                EDGE_POS_V => ((sx as usize * s + xa) as f64 + 0.5, (sy as usize * s + ya + 1) as f64),
                                EDGE_NEG_V => ((sx as usize * s + xa) as f64 + 0.5, (sy as usize * s + ya) as f64),
                                _ => unreachable!(),
                            };
                            let (gxb, gyb) = match ab.edge {
                                EDGE_POS_U => ((ab.neighbor.sx as usize * s + xb + 1) as f64, (ab.neighbor.sy as usize * s + yb) as f64 + 0.5),
                                EDGE_NEG_U => ((ab.neighbor.sx as usize * s + xb) as f64, (ab.neighbor.sy as usize * s + yb) as f64 + 0.5),
                                EDGE_POS_V => ((ab.neighbor.sx as usize * s + xb) as f64 + 0.5, (ab.neighbor.sy as usize * s + yb + 1) as f64),
                                EDGE_NEG_V => ((ab.neighbor.sx as usize * s + xb) as f64 + 0.5, (ab.neighbor.sy as usize * s + yb) as f64),
                                _ => unreachable!(),
                            };

                            let (ua, va) = face_virtual_uv_f64(gxa, gya, n_virt);
                            let (ub, vb) = face_virtual_uv_f64(gxb, gyb, n_virt);
                            let da = face_uv_to_sphere_dir_f64(f as usize, ua, va);
                            let db = face_uv_to_sphere_dir_f64(ab.neighbor.face as usize, ub, vb);
                            let err = ((da[0] - db[0]).powi(2)
                                + (da[1] - db[1]).powi(2)
                                + (da[2] - db[2]).powi(2))
                            .sqrt();
                            assert!(
                                err < 1e-12,
                                "3D seam mismatch on ({key:?}, {edge}) -> ({:?}, {}) k={k}: err={err}",
                                ab.neighbor,
                                ab.edge
                            );
                        }

                        total_directed += 1;
                        if ab.crosses_cube_face {
                            cross_face_directed += 1;
                        } else {
                            interior_directed += 1;
                        }
                    }
                }
            }
        }

        assert_eq!(total_directed, 1536);
        assert_eq!(cross_face_directed, 12 * 8 * 2); // 192 directed cross-face shard edges
        assert_eq!(interior_directed, 1536 - 192);   // 1344 directed interior shard edges
    }

    #[test]
    fn test_f64_floating_origin_precision_at_1300m_and_earth_scale() {
        // 1. Verify locally-flat horizon drop at R = 1,300 m is < 1.0 m over 50 m (~0.9619 m)
        let drop_50m = horizon_drop_meters(50.0, 1300.0);
        let approx = (50.0f64 * 50.0) / (2.0 * 1300.0);
        assert!(drop_50m < 1.0, "50m horizon drop must be < 1.0m, got {drop_50m}");
        assert!(
            (drop_50m - approx).abs() < 1e-3,
            "horizon drop {drop_50m} should closely match d^2/2R {approx}"
        );

        // 2. Verify Earth-scale horizon drop over 1,000 m (~0.07848 m) without cancellation!
        let drop_earth = horizon_drop_meters(1000.0, 6_371_000.0);
        let approx_earth = (1000.0f64 * 1000.0) / (2.0 * 6_371_000.0);
        assert!((drop_earth - approx_earth).abs() < 1e-6);

        // 3. Verify `world_f64_to_camera_rel_f32` achieves < 1e-5 m vertex precision at BOTH
        //    R = 1,300 m and Earth-scale R = 6,371,000 m!
        for &r_planet in &[1_300.0f64, 6_371_000.0f64] {
            let n_virt = 2048usize;
            let (uc, vc) = face_virtual_uv_f64(1024.5, 1024.5, n_virt);
            let cam_origin = face_uv_z_to_world_f64(FACE_POS_Z, uc, vc, 2.5, r_planet);

            // Test vertices offset by subtle millimeter-scale & meter-scale steps near the camera
            for &delta_m in &[0.001f64, 0.25, 12.3456, 128.0] {
                let target_f64 = [
                    cam_origin[0] + delta_m * 0.6,
                    cam_origin[1] - delta_m * 0.5,
                    cam_origin[2] + delta_m * 0.6245,
                ];
                let rel_f32 = world_f64_to_camera_rel_f32(target_f64, cam_origin);
                let exact_rel = [
                    target_f64[0] - cam_origin[0],
                    target_f64[1] - cam_origin[1],
                    target_f64[2] - cam_origin[2],
                ];
                let err = ((rel_f32[0] as f64 - exact_rel[0]).powi(2)
                    + (rel_f32[1] as f64 - exact_rel[1]).powi(2)
                    + (rel_f32[2] as f64 - exact_rel[2]).powi(2))
                .sqrt();
                assert!(
                    err < 1e-5,
                    "floating-origin error at R={r_planet}m, d={delta_m}m exceeded 1e-5m: {err}"
                );
            }
        }
    }

    #[test]
    fn test_dual_scale_hydrate_cross_face_seam_exact_conservation() {
        // Fast test config: 8x8 shards per face, 16x16x16 per shard, 16x16 macro climate
        let cfg = MacroPlanetConfig {
            shards_per_face_axis: 8,
            shard_size: 16,
            depth: 16,
            r_core: 1280.0,
            z_sea: 24,
            climate_res: 16,
            seed: 33_033,
        };
        let mut planet = DualScalePlanet::new(cfg);

        // Pick 2 shards straddling the 90°-rotated cube-face seam between `+X` (EDGE_POS_V) and `+Y` (EDGE_POS_U)
        let key_a = PlanetShardKey::new(FACE_POS_X as u8, 3, 7);
        let link = shard_neighbor(key_a, EDGE_POS_V, cfg.shards_per_face_axis);
        assert!(link.crosses_cube_face);
        assert_eq!(link.neighbor.face, FACE_POS_Y as u8);
        assert_eq!(link.edge, EDGE_POS_U);
        assert!(link.rot[0][0] == 0 && link.rot[1][1] == 0, "must be a 90-deg rotated seam");
        let key_b = link.neighbor;

        // Also hydrate a `+X` <-> `+Z` seam shard pair to test equatorial cube-face transitions simultaneously!
        let key_x_eq = PlanetShardKey::new(FACE_POS_X as u8, 0, 3);
        let link_xz = shard_neighbor(key_x_eq, EDGE_NEG_U, cfg.shards_per_face_axis);
        assert!(link_xz.crosses_cube_face);
        assert_eq!(link_xz.neighbor.face, FACE_POS_Z as u8);
        let key_z_eq = link_xz.neighbor;

        assert!(planet.hydrate_shard(key_a));
        assert!(planet.hydrate_shard(key_b));
        assert!(planet.hydrate_shard(key_x_eq));
        assert!(planet.hydrate_shard(key_z_eq));
        assert_eq!(planet.shards.len(), 4);
        assert_eq!(planet.global_mass_error(), 0);
        assert_eq!(planet.global_energy_error(), 0);

        // Seed a hot high-head water channel right along the rotated `+X -> +Y` seam (`z = 9`)
        // and clear rock on the matching boundary cells so fluid & heat flow vigorously across!
        let s = cfg.shard_size;
        let z_chan = 9usize;
        for k in 3..13 {
            let (xa, ya) = edge_cell_coords(EDGE_POS_V, k, s);
            let idx_a = ya * s + xa;
            {
                let la = &mut planet.shards.get_mut(&key_a).unwrap().layers[z_chan];
                for sp in 0..NUM_SPECIES {
                    la.mass[sp][idx_a] = 0;
                }
                la.mass[SP_WATER][idx_a] = 1500;
                la.flow_s[idx_a] = 180;
                la.set_cell_temp(idx_a, 34_500);
            }

            let (xb, yb) = link.map_edge_cell(k, s);
            let idx_b = yb * s + xb;
            {
                let lb = &mut planet.shards.get_mut(&key_b).unwrap().layers[z_chan];
                for sp in 0..NUM_SPECIES {
                    lb.mass[sp][idx_b] = 0;
                }
                lb.mass[SP_WATER][idx_b] = 150;
                lb.set_cell_temp(idx_b, 28_000);
            }
        }
        planet.snapshot_conservation_baseline();

        let water_b_before: i64 = planet.shards[&key_b].layers[z_chan].mass[SP_WATER]
            .iter()
            .map(|&v| v as i64)
            .sum();

        // Run 6 full dual-scale ticks (3D RefSim + cross-shard rotated exchange + macro<->micro coupling!)
        let mut total_crossed = 0i64;
        for _ in 0..6 {
            let st = planet.step_hydrated_shards();
            assert_eq!(st.global_mass_error, 0, "global_mass_error must be 0 at tick {}", st.tick);
            assert_eq!(st.global_energy_error, 0, "global_energy_error must be 0 at tick {}", st.tick);
            total_crossed += st.seam_mass_crossed;
        }

        let water_b_after: i64 = planet.shards[&key_b].layers[z_chan].mass[SP_WATER]
            .iter()
            .map(|&v| v as i64)
            .sum();

        assert!(
            total_crossed > 0,
            "positive mass must cross across the rotated cube-face shard seam"
        );
        assert!(
            water_b_after > water_b_before,
            "neighbor shard +Y must receive water across rotated +X -> +Y seam: before={water_b_before} after={water_b_after}"
        );

        // Verify lazy `hydrate_around` + eviction across cube-face corners maintains exact conservation!
        let active = planet.hydrate_around(FACE_POS_X, 0, 7 * s + 8, 1);
        assert!(active.len() >= 4);
        assert_eq!(planet.global_mass_error(), 0);
        assert_eq!(planet.global_energy_error(), 0);
    }
}
