//! Wave 34B: Live Locally-Flat `R = 1,304 m` Dual-Scale Spherical Clipmap (`MacroPlanetWorld`).
//!
//! Synchronizes three exact `f64` floating-origin planetary layers on Apple M4 Pro Metal:
//! - **Layer A (`200 m – ∞`, Full 6-Face `R = 1,304 m` Macro Globe):**
//!   Deterministically constructed from `field_core::planet_macro::sample_macro_planet_column` using
//!   `telemetry.planet_seed` (matching the server 1:1!), inset-clipped strictly inside hydrated
//!   `1 m` shard footprints (zero Z-fighting, zero cracks!), plus a live `6`-layer `64×64` RGBA8
//!   `PlanetaryClimateNet` weather/storm/ash/snow array texture!
//! - **Layer B (`0 – 256 m`, Live Hydrated `1 m` Shards, `256×256×64`):**
//!   Incrementally updated from `ServerDatagram::PlanetShardBrick3D` / `PlanetShardBrick3DDelta`,
//!   extracting crisp `1 m` curved terrain with `-4.5 m` edge skirts, golden beaches, Fresnel
//!   sapphire ocean bays, sculpted 3D Nordic pine clusters, coastal boulders, Floating Sky-Islands
//!   with radiant cyan-amethyst Buoyant Crystal Core keels & plunging waterfalls, and translucent
//!   steam/volcanic plumes — all projected around `origin_f64` via `world_f64_to_camera_rel_f32`!
//! - **Layer C (Live 60 3D Heroes, Shrines, Beams & Projectiles in `f64` Floating-Origin Space):**
//!   Projects all 5 archetypes (Strider, animated winged Skywing, Tidekin, Emberstalker, Burrower),
//!   planetary shrines with tall orbital light pillars, curved geodesic elemental actuator beams,
//!   and projectiles at exact `1 m` human scale, plus a live `256×128` equirectangular minimap
//!   compositing continents, oceans, and live `PlanetaryClimateNet` Coriolis storms & ash plumes!

use crate::net::{LiveEntityState, MacroClimateUpdate, MacroShardBrickUpdate};
use bytemuck::{Pod, Zeroable};
use field_core::planet::{FACE_COUNT, FACE_FRAMES};
use field_core::planet_macro::{
    dot3_f64, face_uv_to_sphere_dir_f64, face_uv_z_to_world_f64, face_virtual_uv_f64,
    normalize3_f64, sample_macro_planet_column, sample_micro_relief_m, sphere_dir_to_face_uv_f64,
    world_f64_to_camera_rel_f32, MacroPlanetConfig, MacroSkyIsland, PlanetShardKey,
};
use frostmarsh_proto::{
    CHUNK_SIZE, CLASS_BURROWER, CLASS_EMBERSTALKER, CLASS_SKYWING, CLASS_STRIDER, CLASS_TIDEKIN,
};
use std::collections::{HashMap, HashSet};

pub const MACRO_CLIMATE_RES: usize = 64;
pub const MACRO_MINIMAP_W: usize = 256;
pub const MACRO_MINIMAP_H: usize = 128;

// Material kind IDs matching `macro_planet_live.wgsl`
pub const MAT_MACRO_TERRAIN: f32 = 0.0;
pub const MAT_MACRO_PROP: f32 = 1.0;
pub const MAT_MACRO_OCEAN: f32 = 2.0;
pub const MAT_MACRO_CRYSTAL: f32 = 3.0;
pub const MAT_MACRO_ENTITY: f32 = 4.0;
pub const MAT_MACRO_BEAM: f32 = 5.0;
pub const MAT_MACRO_GAS: f32 = 6.0;

// =============================================================================
// Floating-Origin Dual-Scale Vertex Layout (72 bytes)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MacroPlanetVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// `rgb = base albedo / tint`, `a = roughness` (solid) or `alpha` (translucent)
    pub color: [f32; 4],
    /// `rgb = linear RGB emissive radiance`, `w = mat_kind`
    pub emissive_mat: [f32; 4],
    /// `x = water_depth`, `y = ice_frac`, `z = is_macro`, `w = reserved`
    pub water_ice: [f32; 4],
}

impl MacroPlanetVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MacroPlanetVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: 56,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// =============================================================================
// Exact f64 Spherical Vector Helpers
// =============================================================================

#[inline]
pub fn cross_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn sub_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn add_f64(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn scale_f64(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dist_f64(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[inline]
pub fn tangent_basis_f64(dir: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let ref_up = if dir[1].abs() < 0.9 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t1 = normalize3_f64(cross_f64(ref_up, dir));
    let t2 = normalize3_f64(cross_f64(dir, t1));
    (t1, t2)
}

/// Exact `f64` orthonormal tangent basis `(up, tang_u, tang_v)` on cube face `face`.
#[inline]
pub fn face_tangent_basis_f64(face: usize, u: f64, v: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let up = face_uv_to_sphere_dir_f64(face % FACE_COUNT, u, v);
    let fr = FACE_FRAMES[face % FACE_COUNT];
    let eu_raw = [fr.e_u[0] as f64, fr.e_u[1] as f64, fr.e_u[2] as f64];
    let ev_raw = [fr.e_v[0] as f64, fr.e_v[1] as f64, fr.e_v[2] as f64];
    let du = dot3_f64(eu_raw, up);
    let tang_u = normalize3_f64(sub_f64(eu_raw, scale_f64(up, du)));
    let dv = dot3_f64(ev_raw, up);
    let tv_raw = sub_f64(ev_raw, scale_f64(up, dv));
    let tang_v = if dot3_f64(tv_raw, tv_raw) > 0.25 {
        normalize3_f64(tv_raw)
    } else {
        normalize3_f64(cross_f64(up, tang_u))
    };
    (up, tang_u, tang_v)
}

/// Map virtual face coordinates `(face, gx, gy, gz)` (`gx, gy ∈ [0, 2048)`, `gz ∈ [0, 64)`) to `f64` 3D world position.
#[inline]
pub fn virtual_grid_to_world_f64(
    face: u8,
    gx: f64,
    gy: f64,
    gz: f64,
    face_size_virtual: usize,
    r_core: f64,
) -> [f64; 3] {
    let (u, v) = face_virtual_uv_f64(gx, gy, face_size_virtual);
    face_uv_z_to_world_f64(face as usize, u, v, gz, r_core)
}

/// Map an `f64` 3D spherical position to equirectangular minimap UV `(0..1, 0..1)`.
#[inline]
pub fn sphere_pos_f64_to_minimap_uv(pos: [f64; 3]) -> (f32, f32) {
    let d = normalize3_f64(pos);
    let lon = d[2].atan2(d[0]);
    let lat = d[1].clamp(-1.0, 1.0).asin();
    let u = (lon + std::f64::consts::PI) / std::f64::consts::TAU;
    let v = 0.5 - lat / std::f64::consts::PI;
    (u.clamp(0.0, 1.0) as f32, v.clamp(0.0, 1.0) as f32)
}

// =============================================================================
// Per-Shard Hydrated 256×256×64 1m Data & Incremental Column Cache
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct ShardColumn {
    surf_z: f64,
    water_depth: f32,
    ice_frac: f32,
    albedo: [f32; 3],
    emissive: [f32; 3],
    roughness: f32,
    has_wood: bool,
    has_plant: bool,
}

pub struct MacroShardData {
    pub face: u8,
    pub sx: u8,
    pub sy: u8,
    pub shard_size: usize,
    pub depth: usize,
    /// Z-major `[z][y][x][4]`: `[solid_density, fluid_density, gas_density, roughness]`
    pub density_mat: Vec<u8>,
    /// Z-major `[z][y][x][4]`: `[albedo_r, albedo_g, albedo_b, thermal_glow]`
    pub albedo_thermal: Vec<u8>,
    /// Bitmask of columns whose live sub-brick voxels were received.
    column_live_mask: Vec<bool>,
    column_cache: Vec<ShardColumn>,
    pub received_subbricks: HashSet<(u16, u16, u16)>,
    pub dirty: bool,
    pub terrain_verts: Vec<MacroPlanetVertex>,
    pub terrain_indices: Vec<u32>,
    pub ocean_verts: Vec<MacroPlanetVertex>,
    pub ocean_indices: Vec<u32>,
    pub gas_verts: Vec<MacroPlanetVertex>,
    pub gas_indices: Vec<u32>,
}

impl MacroShardData {
    fn new(
        face: u8,
        sx: u8,
        sy: u8,
        shard_size: usize,
        depth: usize,
        face_size_virtual: usize,
        seed: u64,
        z_sea_surf: f64,
    ) -> Self {
        let s = shard_size.max(16);
        let d = depth.max(16);
        let bytes = s * s * d * 4;
        let gx_base = sx as usize * s;
        let gy_base = sy as usize * s;

        // Pre-initialize column cache deterministically from `sample_macro_planet_column` so any
        // unstreamed / skipped bedrock/sky bricks seamlessly match the macro globe with ZERO holes!
        let mut column_cache = Vec::with_capacity(s * s);
        for ly in 0..s {
            let gy = (gy_base + ly) as f64 + 0.5;
            for lx in 0..s {
                let gx = (gx_base + lx) as f64 + 0.5;
                let (u, v) = face_virtual_uv_f64(gx, gy, face_size_virtual);
                let dir = face_uv_to_sphere_dir_f64(face as usize, u, v);
                let col = sample_macro_planet_column(dir, seed);
                let edge_d = lx.min(s - 1 - lx).min(ly).min(s - 1 - ly) as f64;
                let edge_blend = (edge_d / 10.0).clamp(0.0, 1.0);
                let surf_z = (col.surf_z + sample_micro_relief_m(dir, seed) * edge_blend * 0.85)
                    .clamp(16.0, 42.5);
                let water_depth = (z_sea_surf - surf_z).max(0.0) as f32;
                let ice_frac = if water_depth > 0.05 && col.temp_k < 271.5 {
                    ((271.5 - col.temp_k) / 8.0).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };
                column_cache.push(ShardColumn {
                    surf_z,
                    water_depth,
                    ice_frac,
                    albedo: col.albedo,
                    emissive: col.emissive,
                    roughness: col.roughness,
                    has_wood: col.forest_density > 0.40,
                    has_plant: col.forest_density > 0.15,
                });
            }
        }

        Self {
            face,
            sx,
            sy,
            shard_size: s,
            depth: d,
            density_mat: vec![0u8; bytes],
            albedo_thermal: vec![0u8; bytes],
            column_live_mask: vec![false; s * s],
            column_cache,
            received_subbricks: HashSet::new(),
            dirty: true,
            terrain_verts: Vec::new(),
            terrain_indices: Vec::new(),
            ocean_verts: Vec::new(),
            ocean_indices: Vec::new(),
            gas_verts: Vec::new(),
            gas_indices: Vec::new(),
        }
    }

    #[inline]
    fn voxel_offset(&self, x: usize, y: usize, z: usize) -> usize {
        ((z * self.shard_size + y) * self.shard_size + x) * 4
    }

    #[inline]
    fn solid_at(&self, x: isize, y: isize, z: isize) -> u8 {
        if x < 0
            || y < 0
            || z < 0
            || x >= self.shard_size as isize
            || y >= self.shard_size as isize
            || z >= self.depth as isize
        {
            return 0;
        }
        self.density_mat[self.voxel_offset(x as usize, y as usize, z as usize)]
    }

    /// Recompute only the `16×16` column cache region touched by brick `(bx, by)`.
    fn recompute_brick_columns(
        &mut self,
        bx: usize,
        by: usize,
        z_sea: usize,
        sky_z_floor: usize,
        face_size_virtual: usize,
        seed: u64,
    ) {
        let s = self.shard_size;
        let x0 = bx * CHUNK_SIZE;
        let y0 = by * CHUNK_SIZE;
        let x1 = (x0 + CHUNK_SIZE).min(s);
        let y1 = (y0 + CHUNK_SIZE).min(s);
        let z_sea_f = z_sea as f64 + 0.55;
        let max_z = sky_z_floor.min(self.depth);
        let gx_base = self.sx as usize * s;
        let gy_base = self.sy as usize * s;

        for ly in y0..y1 {
            let gy = (gy_base + ly) as f64 + 0.5;
            for lx in x0..x1 {
                let gx = (gx_base + lx) as f64 + 0.5;
                let ci = ly * s + lx;
                let (u, v) = face_virtual_uv_f64(gx, gy, face_size_virtual);
                let dir = face_uv_to_sphere_dir_f64(self.face as usize, u, v);
                let proc_col = sample_macro_planet_column(dir, seed);
                let edge_d = lx.min(s - 1 - lx).min(ly).min(s - 1 - ly) as f64;
                let edge_blend = (edge_d / 10.0).clamp(0.0, 1.0);
                let proc_surf_z = (proc_col.surf_z
                    + sample_micro_relief_m(dir, seed) * edge_blend * 0.85)
                    .clamp(16.0, 42.5);

                let mut found_solid = false;
                let mut solid_z = 0usize;
                let mut solid_fill = 0.0f64;
                let mut best_col = proc_col.albedo;
                let mut best_rough = proc_col.roughness;
                let mut max_thermal = 0.0f32;
                let mut fluid_cols = 0.0f32;
                let mut fluid_top_z = 0.0f64;
                let mut sea_ice_hits = 0.0f32;
                let mut sea_water_hits = 0.0f32;

                for z in (0..max_z).rev() {
                    let off = self.voxel_offset(lx, ly, z);
                    let solid = self.density_mat[off];
                    let fluid = self.density_mat[off + 1];
                    let gas = self.density_mat[off + 2];
                    let rough = self.density_mat[off + 3] as f32 / 255.0;
                    let ar = self.albedo_thermal[off] as f32 / 255.0;
                    let ag = self.albedo_thermal[off + 1] as f32 / 255.0;
                    let ab = self.albedo_thermal[off + 2] as f32 / 255.0;
                    let therm = self.albedo_thermal[off + 3] as f32 / 255.0;

                    if therm > max_thermal && (solid >= 40 || fluid >= 40 || gas >= 40) {
                        max_thermal = therm;
                    }
                    if fluid > 60 && therm < 0.35 {
                        fluid_cols += fluid as f32 / 255.0;
                        if fluid_top_z <= 0.0 {
                            fluid_top_z = z as f64 + (fluid as f64 / 255.0).clamp(0.25, 1.0);
                        }
                        if z == z_sea || z + 1 == z_sea {
                            sea_water_hits += fluid as f32 / 255.0;
                        }
                    }
                    if (z == z_sea || z + 1 == z_sea)
                        && solid >= 100
                        && ab > 0.72
                        && ar > 0.56
                        && rough < 0.38
                    {
                        sea_ice_hits += solid as f32 / 255.0;
                    }
                    if solid >= 96 && !found_solid {
                        found_solid = true;
                        solid_z = z;
                        solid_fill = (solid as f64 / 255.0).clamp(0.18, 1.0);
                        best_col = [ar, ag, ab];
                        best_rough = rough;
                        break;
                    }
                }

                if !found_solid {
                    // Only sky / upper sub-bricks received so far -> keep seamless procedural column
                    continue;
                }

                if solid_z + 1 < max_z {
                    let off_u = self.voxel_offset(lx, ly, solid_z + 1);
                    let therm_u = self.albedo_thermal[off_u + 3] as f32 / 255.0;
                    if therm_u > max_thermal {
                        max_thermal = therm_u;
                    }
                }

                // Blend live voxel height smoothly at outer 10m edge so shard stitches cleanly to macro globe!
                let live_surf_z = (solid_z as f64 + solid_fill).clamp(16.0, 42.5);
                let surf_z = proc_surf_z * (1.0 - edge_blend) + live_surf_z * edge_blend;

                let water_depth = if fluid_cols > 0.15 {
                    ((fluid_top_z.max(z_sea_f) - surf_z).max(fluid_cols as f64 * 0.65)).max(0.0) as f32
                } else {
                    (z_sea_f - surf_z).max(0.0) as f32
                };

                let ice_frac = if water_depth > 0.05 && (sea_ice_hits + sea_water_hits) > 0.01 {
                    (sea_ice_hits / (sea_ice_hits + sea_water_hits)).clamp(0.0, 1.0)
                } else if water_depth > 0.05 && proc_col.temp_k < 271.5 {
                    ((271.5 - proc_col.temp_k) / 8.0).clamp(0.0, 1.0) as f32
                } else {
                    0.0
                };

                let mut albedo = best_col;
                let mut emissive = proc_col.emissive;
                let mut roughness = best_rough.clamp(0.14, 0.94);
                let has_wood = best_col[1] > best_col[0] * 1.15 && best_col[1] < 0.34 && roughness > 0.72;
                let has_plant = best_col[1] > best_col[0] * 1.08 && best_col[1] > best_col[2] * 1.08;

                if max_thermal > 0.45 {
                    let hot = ((max_thermal - 0.35) / 0.65).clamp(0.20, 1.0);
                    albedo = [0.16, 0.05, 0.02];
                    emissive = [1.8 + 3.4 * hot, 0.36 + 1.25 * hot * hot, 0.05 + 0.20 * hot];
                    roughness = 0.50;
                } else if max_thermal > 0.16 {
                    let ember = ((max_thermal - 0.16) / 0.30).clamp(0.0, 1.0);
                    albedo = [
                        albedo[0] * 0.45 + 0.08,
                        albedo[1] * 0.40 + 0.04,
                        albedo[2] * 0.45 + 0.05,
                    ];
                    emissive = [0.55 * ember, 0.12 * ember, 0.02 * ember];
                } else if best_col[2] > 0.72 && best_col[0] > 0.56 && roughness < 0.36 {
                    albedo = [0.85, 0.91, 0.97];
                    roughness = 0.25;
                } else if surf_z > z_sea_f && surf_z < z_sea_f + 1.25 && !proc_col.is_polar {
                    albedo = [0.54, 0.44, 0.27]; // Golden sandy coastal beach
                    roughness = 0.88;
                } else if has_wood {
                    albedo = [0.042, 0.17, 0.068];
                    roughness = 0.86;
                } else if has_plant && surf_z > z_sea_f + 1.1 {
                    let alt_t = ((surf_z - z_sea_f) / 11.0).clamp(0.0, 1.0) as f32;
                    albedo = [0.085 + 0.10 * alt_t, 0.28 - 0.05 * alt_t, 0.08 + 0.025 * alt_t];
                    roughness = 0.82;
                }

                self.column_live_mask[ci] = true;
                self.column_cache[ci] = ShardColumn {
                    surf_z,
                    water_depth,
                    ice_frac,
                    albedo,
                    emissive,
                    roughness,
                    has_wood: has_wood || proc_col.forest_density > 0.42,
                    has_plant: has_plant || proc_col.forest_density > 0.18,
                };
            }
        }
    }
}

// =============================================================================
// Cached Base Macro Globe Face Geometry (Fast O(N²) Re-Clipping on Shard Hydration)
// =============================================================================

struct MacroBaseFace {
    sub: usize,
    dirs: Vec<[f64; 3]>,
    surf_z: Vec<f64>,
    normals: Vec<[f32; 3]>,
    albedos: Vec<[f32; 3]>,
    emissives: Vec<[f32; 3]>,
    roughness: Vec<f32>,
    water_depth: Vec<f32>,
    ice_frac: Vec<f32>,
}

// =============================================================================
// Main Dual-Scale Macro Planet World (`R = 1,304 m`, `6 × 2048² × 64`)
// =============================================================================

pub struct MacroPlanetWorld {
    pub r_core: f64,
    pub z_sea: f64,
    pub shards_per_axis: u8,
    pub shard_size: usize,
    pub depth: usize,
    pub seed: u64,

    /// Exact `f64` floating origin for near `1 m` shards & dynamic 3D entities (snapped when cam moves > 32 m).
    pub origin_f64: [f64; 3],
    origin_initialized: bool,

    /// Layer A: Deterministic 6-face macro globe cache & GPU-ready mesh buffers (origin = `[0,0,0]`)
    macro_sub: usize,
    macro_base_faces: Vec<MacroBaseFace>,
    macro_base_built: bool,
    macro_clip_dirty: bool,
    pub macro_terrain_verts: Vec<MacroPlanetVertex>,
    pub macro_terrain_indices: Vec<u32>,
    pub macro_ocean_verts: Vec<MacroPlanetVertex>,
    pub macro_ocean_indices: Vec<u32>,
    pub macro_sky_verts: Vec<MacroPlanetVertex>,
    pub macro_sky_indices: Vec<u32>,
    pub macro_gpu_dirty: bool,

    /// Live `6`-layer `64×64` RGBA8 `PlanetaryClimateNet` weather/storm/ash/snow overlay
    pub climate_res: usize,
    pub climate_faces: [Vec<u8>; FACE_COUNT],
    pub climate_dirty_faces: [bool; FACE_COUNT],
    pub received_climate_faces: HashSet<u8>,
    pub total_climate_updates_applied: u64,

    /// 12 Orbital Floating Sky-Islands matching `field_core::planet_macro`
    pub sky_islands: Vec<MacroSkyIsland>,

    /// Layer B: Live hydrated `256×256×64` `1 m` shards
    pub shards: HashMap<(u8, u8, u8), MacroShardData>,
    pub received_shard_bricks: HashSet<(u8, u8, u8, u16, u16, u16)>,
    pub total_shard_updates_applied: u64,
    pub near_terrain_verts: Vec<MacroPlanetVertex>,
    pub near_terrain_indices: Vec<u32>,
    pub near_ocean_verts: Vec<MacroPlanetVertex>,
    pub near_ocean_indices: Vec<u32>,
    pub near_gas_verts: Vec<MacroPlanetVertex>,
    pub near_gas_indices: Vec<u32>,
    pub near_gpu_dirty: bool,

    /// Layer C: Live per-frame 3D Heroes, Shrines, Beams & Projectiles (floating-origin relative to `origin_f64`)
    pub dynamic_solid_verts: Vec<MacroPlanetVertex>,
    pub dynamic_solid_indices: Vec<u32>,
    pub dynamic_fx_verts: Vec<MacroPlanetVertex>,
    pub dynamic_fx_indices: Vec<u32>,

    /// Live `256×128` equirectangular minimap (continents + oceans + live `PlanetaryClimateNet` storms & ash!)
    pub minimap_rgba: Vec<u8>,
    pub minimap_dirty: bool,
    minimap_needs_rebuild: bool,
}

impl MacroPlanetWorld {
    pub fn new(r_core: f64, shards_per_axis: u8, depth: usize, seed: u64) -> Self {
        let def = MacroPlanetConfig::default();
        let rc = if r_core > 100.0 { r_core } else { def.r_core };
        let spa = if shards_per_axis > 0 { shards_per_axis } else { def.shards_per_face_axis };
        let d = if depth >= 16 { depth } else { def.depth };
        let sd = if seed != 0 { seed } else { def.seed };
        let z_sea = ((24.0 * (d as f64 / 64.0)).round()).clamp(8.0, (d.saturating_sub(8)) as f64);

        let climate_faces = Self::init_procedural_climate_faces(MACRO_CLIMATE_RES, sd);
        let sky_islands = MacroSkyIsland::default_archipelago(sd);

        // 112 quads/face in release (150k tris, >320 FPS on M4 Pro Metal), 64 in debug tests for fast test execution
        let macro_sub = if cfg!(debug_assertions) { 64 } else { 112 };

        Self {
            r_core: rc,
            z_sea,
            shards_per_axis: spa,
            shard_size: def.shard_size,
            depth: d,
            seed: sd,
            origin_f64: [0.0, 0.0, rc + z_sea],
            origin_initialized: false,
            macro_sub,
            macro_base_faces: Vec::new(),
            macro_base_built: false,
            macro_clip_dirty: true,
            macro_terrain_verts: Vec::new(),
            macro_terrain_indices: Vec::new(),
            macro_ocean_verts: Vec::new(),
            macro_ocean_indices: Vec::new(),
            macro_sky_verts: Vec::new(),
            macro_sky_indices: Vec::new(),
            macro_gpu_dirty: true,
            climate_res: MACRO_CLIMATE_RES,
            climate_faces,
            climate_dirty_faces: [true; FACE_COUNT],
            received_climate_faces: HashSet::new(),
            total_climate_updates_applied: 0,
            sky_islands,
            shards: HashMap::new(),
            received_shard_bricks: HashSet::new(),
            total_shard_updates_applied: 0,
            near_terrain_verts: Vec::new(),
            near_terrain_indices: Vec::new(),
            near_ocean_verts: Vec::new(),
            near_ocean_indices: Vec::new(),
            near_gas_verts: Vec::new(),
            near_gas_indices: Vec::new(),
            near_gpu_dirty: false,
            dynamic_solid_verts: Vec::new(),
            dynamic_solid_indices: Vec::new(),
            dynamic_fx_verts: Vec::new(),
            dynamic_fx_indices: Vec::new(),
            minimap_rgba: vec![18u8; MACRO_MINIMAP_W * MACRO_MINIMAP_H * 4],
            minimap_dirty: true,
            minimap_needs_rebuild: true,
        }
    }

    pub fn configure_if_needed(&mut self, r_core: f64, shards_per_axis: u8, depth: usize, seed: u64) {
        let rc = if r_core > 100.0 { r_core } else { self.r_core };
        let spa = if shards_per_axis > 0 { shards_per_axis } else { self.shards_per_axis };
        let d = if depth >= 16 { depth } else { self.depth };
        let sd = if seed != 0 { seed } else { self.seed };
        if (self.r_core - rc).abs() > 0.5
            || self.shards_per_axis != spa
            || self.depth != d
            || self.seed != sd
        {
            *self = Self::new(rc, spa, d, sd);
        }
    }

    #[inline]
    pub fn face_size_virtual(&self) -> usize {
        (self.shards_per_axis.max(1) as usize) * self.shard_size
    }

    #[inline]
    pub fn r_surf(&self) -> f64 {
        self.r_core + self.z_sea
    }

    #[inline]
    pub fn r_ocean(&self) -> f64 {
        self.r_core + self.z_sea + 0.55
    }

    #[inline]
    pub fn sky_z_floor(&self) -> usize {
        ((44.0 * (self.depth as f64 / 64.0)).round() as usize).clamp(self.z_sea as usize + 8, self.depth.saturating_sub(4))
    }

    #[inline]
    pub fn hydrated_shard_count(&self) -> usize {
        self.shards.len()
    }

    #[inline]
    pub fn received_brick_count(&self) -> usize {
        self.received_shard_bricks.len()
    }

    #[inline]
    pub fn received_climate_face_count(&self) -> usize {
        self.received_climate_faces.len()
    }

    #[inline]
    pub fn has_live_climate(&self) -> bool {
        !self.received_climate_faces.is_empty()
    }

    #[inline]
    pub fn total_triangles(&self) -> usize {
        (self.macro_terrain_indices.len()
            + self.macro_ocean_indices.len()
            + self.macro_sky_indices.len()
            + self.near_terrain_indices.len()
            + self.near_ocean_indices.len()
            + self.near_gas_indices.len()
            + self.dynamic_solid_indices.len()
            + self.dynamic_fx_indices.len())
            / 3
    }

    /// Update floating origin `origin_f64` whenever the camera moves `> 32 m` away from it!
    /// Returns `true` if `origin_f64` shifted and near shard meshes were marked for re-centering.
    pub fn ensure_floating_origin(&mut self, cam_pos_f64: [f64; 3]) -> bool {
        let len = (cam_pos_f64[0] * cam_pos_f64[0]
            + cam_pos_f64[1] * cam_pos_f64[1]
            + cam_pos_f64[2] * cam_pos_f64[2])
        .sqrt();
        if len < 10.0 {
            return false;
        }
        // Anchor floating origin near the surface point below the camera when in ground/flight/strato
        let anchor = if len < self.r_surf() + 800.0 {
            let dir = normalize3_f64(cam_pos_f64);
            scale_f64(dir, self.r_surf())
        } else {
            cam_pos_f64
        };
        if !self.origin_initialized || dist_f64(anchor, self.origin_f64) > 32.0 {
            self.origin_f64 = anchor;
            self.origin_initialized = true;
            for sh in self.shards.values_mut() {
                sh.dirty = true;
            }
            return true;
        }
        false
    }

    /// Initialize deterministic fallback Coriolis cloud & snowline climate faces before live packets arrive.
    fn init_procedural_climate_faces(res: usize, seed: u64) -> [Vec<u8>; FACE_COUNT] {
        std::array::from_fn(|f| {
            let mut rgba = vec![0u8; res * res * 4];
            let rf = res as f64;
            for y in 0..res {
                let v = ((y as f64 + 0.5) / rf) * 2.0 - 1.0;
                for x in 0..res {
                    let u = ((x as f64 + 0.5) / rf) * 2.0 - 1.0;
                    let dir = face_uv_to_sphere_dir_f64(f, u, v);
                    let col = sample_macro_planet_column(dir, seed);
                    let lon = dir[2].atan2(dir[0]);
                    let lat = dir[1];
                    // Swirling Coriolis trade-wind & mid-latitude storm bands
                    let band = (lat * 7.5 + 0.8 * (lon * 3.0).sin()).sin() * 0.5 + 0.5;
                    let vortex = (lon * 4.0 + lat * 6.0).cos() * 0.5 + 0.5;
                    let cloud = (0.18 + 0.42 * band * vortex * col.moisture).clamp(0.0, 0.92);
                    let ash = if col.is_rift_core {
                        0.82
                    } else if col.is_rift_rim {
                        0.32
                    } else {
                        0.0
                    };
                    let snow = if col.is_polar {
                        0.95
                    } else if col.surf_z >= 36.0 {
                        0.78
                    } else {
                        ((272.0 - col.temp_k) / 18.0).clamp(0.0, 1.0)
                    };
                    let temp_n = ((col.temp_k - 240.0) / 80.0).clamp(0.0, 1.0);
                    let off = (y * res + x) * 4;
                    rgba[off] = (temp_n * 255.0) as u8;
                    rgba[off + 1] = (cloud * 255.0) as u8;
                    rgba[off + 2] = (ash * 255.0) as u8;
                    rgba[off + 3] = (snow * 255.0) as u8;
                }
            }
            rgba
        })
    }

    // =========================================================================
    // Live Network Ingestion (Climate Faces + Hydrated 1m Shard Bricks/Deltas)
    // =========================================================================

    pub fn apply_climate_updates(&mut self, updates: Vec<MacroClimateUpdate>) -> usize {
        let mut applied = 0usize;
        for u in updates {
            let f = u.face as usize;
            let gn = u.grid_n as usize;
            if f >= FACE_COUNT || gn == 0 || u.rgba.len() != gn * gn * 4 {
                continue;
            }
            // Resample cleanly into `self.climate_res` (`64×64`) if server sent a different size
            if gn == self.climate_res {
                self.climate_faces[f] = u.rgba;
            } else {
                let dst_n = self.climate_res;
                let mut out = vec![0u8; dst_n * dst_n * 4];
                for y in 0..dst_n {
                    let sy = (y * gn / dst_n).min(gn - 1);
                    for x in 0..dst_n {
                        let sx = (x * gn / dst_n).min(gn - 1);
                        let src_o = (sy * gn + sx) * 4;
                        let dst_o = (y * dst_n + x) * 4;
                        out[dst_o..dst_o + 4].copy_from_slice(&u.rgba[src_o..src_o + 4]);
                    }
                }
                self.climate_faces[f] = out;
            }
            self.climate_dirty_faces[f] = true;
            self.received_climate_faces.insert(u.face);
            self.total_climate_updates_applied += 1;
            self.minimap_needs_rebuild = true;
            applied += 1;
        }
        applied
    }

    pub fn apply_shard_updates(&mut self, updates: Vec<MacroShardBrickUpdate>) -> usize {
        let mut applied = 0usize;
        let z_sea = self.z_sea as usize;
        let sky_z = self.sky_z_floor();
        let n_virt = self.face_size_virtual();
        let seed = self.seed;
        let z_sea_surf = self.z_sea + 0.55;
        let shard_sz = self.shard_size;

        for u in updates {
            match u {
                MacroShardBrickUpdate::FullSubBrick {
                    face,
                    sx,
                    sy,
                    bx,
                    by,
                    bz,
                    total_depth,
                    sub_depth,
                    tick: _,
                    density_mat,
                    albedo_thermal,
                } => {
                    let td = total_depth as usize;
                    let sd = sub_depth as usize;
                    if face as usize >= FACE_COUNT || td == 0 || sd == 0 {
                        continue;
                    }
                    let expected = CHUNK_SIZE * CHUNK_SIZE * sd * 4;
                    if density_mat.len() != expected || albedo_thermal.len() != expected {
                        continue;
                    }
                    let key = (face, sx, sy);
                    let is_new = !self.shards.contains_key(&key);
                    let sh = self.shards.entry(key).or_insert_with(|| {
                        MacroShardData::new(
                            face,
                            sx,
                            sy,
                            shard_sz,
                            td.max(self.depth),
                            n_virt,
                            seed,
                            z_sea_surf,
                        )
                    });
                    if is_new {
                        self.macro_clip_dirty = true;
                    }

                    let x0 = (bx as usize) * CHUNK_SIZE;
                    let y0 = (by as usize) * CHUNK_SIZE;
                    let z0 = (bz as usize) * CHUNK_SIZE;
                    let s = sh.shard_size;
                    let d = sh.depth;
                    if x0 >= s || y0 >= s || z0 >= d {
                        continue;
                    }
                    let copy_w = CHUNK_SIZE.min(s - x0);
                    let copy_h = CHUNK_SIZE.min(s - y0);
                    let copy_d = sd.min(d - z0);
                    let row_bytes = copy_w * 4;

                    for dz in 0..copy_d {
                        let gz = z0 + dz;
                        for dy in 0..copy_h {
                            let gy = y0 + dy;
                            let src = ((dz * CHUNK_SIZE + dy) * CHUNK_SIZE) * 4;
                            let dst = ((gz * s + gy) * s + x0) * 4;
                            sh.density_mat[dst..dst + row_bytes]
                                .copy_from_slice(&density_mat[src..src + row_bytes]);
                            sh.albedo_thermal[dst..dst + row_bytes]
                                .copy_from_slice(&albedo_thermal[src..src + row_bytes]);
                        }
                    }
                    sh.received_subbricks.insert((bx, by, bz));
                    sh.recompute_brick_columns(bx as usize, by as usize, z_sea, sky_z, n_virt, seed);
                    sh.dirty = true;
                    self.received_shard_bricks.insert((face, sx, sy, bx, by, bz));
                    self.total_shard_updates_applied += 1;
                    applied += 1;
                }
                MacroShardBrickUpdate::DeltaSubBrick {
                    face,
                    sx,
                    sy,
                    bx,
                    by,
                    bz,
                    total_depth,
                    sub_depth,
                    tick: _,
                    patches,
                } => {
                    let td = total_depth as usize;
                    let sd = sub_depth as usize;
                    if face as usize >= FACE_COUNT || td == 0 || sd == 0 {
                        continue;
                    }
                    let key = (face, sx, sy);
                    let is_new = !self.shards.contains_key(&key);
                    let sh = self.shards.entry(key).or_insert_with(|| {
                        MacroShardData::new(
                            face,
                            sx,
                            sy,
                            shard_sz,
                            td.max(self.depth),
                            n_virt,
                            seed,
                            z_sea_surf,
                        )
                    });
                    if is_new {
                        self.macro_clip_dirty = true;
                    }
                    let x0 = (bx as usize) * CHUNK_SIZE;
                    let y0 = (by as usize) * CHUNK_SIZE;
                    let z0 = (bz as usize) * CHUNK_SIZE;
                    let s = sh.shard_size;
                    let d = sh.depth;
                    let total_sub = CHUNK_SIZE * CHUNK_SIZE * sd;
                    for (idx, dm, at) in patches {
                        let v = idx as usize;
                        if v >= total_sub {
                            continue;
                        }
                        let dz = v / (CHUNK_SIZE * CHUNK_SIZE);
                        let rem = v % (CHUNK_SIZE * CHUNK_SIZE);
                        let dy = rem / CHUNK_SIZE;
                        let dx = rem % CHUNK_SIZE;
                        let gx = x0 + dx;
                        let gy = y0 + dy;
                        let gz = z0 + dz;
                        if gx >= s || gy >= s || gz >= d {
                            continue;
                        }
                        let dst = ((gz * s + gy) * s + gx) * 4;
                        sh.density_mat[dst..dst + 4].copy_from_slice(&dm);
                        sh.albedo_thermal[dst..dst + 4].copy_from_slice(&at);
                    }
                    sh.received_subbricks.insert((bx, by, bz));
                    sh.recompute_brick_columns(bx as usize, by as usize, z_sea, sky_z, n_virt, seed);
                    sh.dirty = true;
                    self.received_shard_bricks.insert((face, sx, sy, bx, by, bz));
                    self.total_shard_updates_applied += 1;
                    applied += 1;
                }
            }
        }

        // Evict farthest hydrated shards if we exceed 12 resident shards so memory stays lean
        if self.shards.len() > 12 {
            let origin_dir = normalize3_f64(self.origin_f64);
            let spa = self.shards_per_axis;
            let mut scored: Vec<((u8, u8, u8), f64)> = self
                .shards
                .keys()
                .map(|&k| {
                    let cdir = PlanetShardKey::new(k.0, k.1, k.2).center_sphere_dir(spa);
                    (k, dot3_f64(cdir, origin_dir))
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(k, _) in scored.iter().skip(12) {
                self.shards.remove(&k);
                self.macro_clip_dirty = true;
            }
        }

        applied
    }

    // =========================================================================
    // Layer A & Layer B Mesh Construction (Zero Z-Fighting Inset-Clipped Clipmap)
    // =========================================================================

    pub fn rebuild_dirty(&mut self, max_shards: usize) -> usize {
        if !self.macro_base_built {
            self.build_macro_base_faces();
            self.build_macro_sky_islands();
            self.macro_base_built = true;
            self.macro_clip_dirty = true;
        }

        if self.macro_clip_dirty {
            self.rebuild_clipped_macro_globe();
            self.macro_clip_dirty = false;
            self.macro_gpu_dirty = true;
        }

        let mut dirty_keys: Vec<(u8, u8, u8)> = self
            .shards
            .iter()
            .filter_map(|(&k, s)| if s.dirty { Some(k) } else { None })
            .collect();
        // Rebuild shards closest to `origin_f64` first!
        let origin_dir = normalize3_f64(self.origin_f64);
        let spa = self.shards_per_axis;
        dirty_keys.sort_by(|a, b| {
            let da = dot3_f64(PlanetShardKey::new(a.0, a.1, a.2).center_sphere_dir(spa), origin_dir);
            let db = dot3_f64(PlanetShardKey::new(b.0, b.1, b.2).center_sphere_dir(spa), origin_dir);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut rebuilt = 0usize;
        for key in dirty_keys.into_iter().take(max_shards.max(1)) {
            self.rebuild_single_shard_mesh(key);
            rebuilt += 1;
        }

        if rebuilt > 0 {
            self.merge_near_shard_meshes();
            self.near_gpu_dirty = true;
        }

        if self.minimap_needs_rebuild {
            self.rebuild_equirectangular_minimap();
            self.minimap_needs_rebuild = false;
        }

        rebuilt
    }

    /// Build deterministic 6-face base vertices using padded central differences (5× faster, seam-free!).
    fn build_macro_base_faces(&mut self) {
        let sub = self.macro_sub;
        let stride = sub + 1;
        let pad_stride = sub + 3;
        let r_core = self.r_core;
        let z_sea_surf = self.z_sea + 0.55;
        let seed = self.seed;

        let mut faces = Vec::with_capacity(FACE_COUNT);
        for f in 0..FACE_COUNT {
            let mut pad_pos = vec![[0.0f64; 3]; pad_stride * pad_stride];
            let mut dirs = vec![[0.0f64; 3]; stride * stride];
            let mut surf_z = vec![24.0f64; stride * stride];
            let mut albedos = vec![[0.0f32; 3]; stride * stride];
            let mut emissives = vec![[0.0f32; 3]; stride * stride];
            let mut roughness = vec![0.8f32; stride * stride];
            let mut water_depth = vec![0.0f32; stride * stride];
            let mut ice_frac = vec![0.0f32; stride * stride];

            for py in 0..pad_stride {
                let iy = py as isize - 1;
                let v = (iy as f64 / sub as f64) * 2.0 - 1.0;
                for px in 0..pad_stride {
                    let ix = px as isize - 1;
                    let u = (ix as f64 / sub as f64) * 2.0 - 1.0;
                    let dir = face_uv_to_sphere_dir_f64(f, u, v);
                    let col = sample_macro_planet_column(dir, seed);
                    let r = r_core + col.surf_z;
                    pad_pos[py * pad_stride + px] = scale_f64(dir, r);

                    if iy >= 0 && iy <= sub as isize && ix >= 0 && ix <= sub as isize {
                        let vi = (iy as usize) * stride + (ix as usize);
                        let wd = (z_sea_surf - col.surf_z).max(0.0) as f32;
                        let ice = if wd > 0.05 && col.temp_k < 271.5 {
                            ((271.5 - col.temp_k) / 8.0).clamp(0.0, 1.0) as f32
                        } else {
                            0.0
                        };
                        dirs[vi] = dir;
                        surf_z[vi] = col.surf_z;
                        albedos[vi] = col.albedo;
                        emissives[vi] = col.emissive;
                        roughness[vi] = col.roughness;
                        water_depth[vi] = wd;
                        ice_frac[vi] = ice;
                    }
                }
            }

            let mut normals = vec![[0.0f32; 3]; stride * stride];
            for iy in 0..=sub {
                let py = iy + 1;
                for ix in 0..=sub {
                    let px = ix + 1;
                    let vi = iy * stride + ix;
                    let dx = sub_f64(
                        pad_pos[py * pad_stride + (px + 1)],
                        pad_pos[py * pad_stride + (px - 1)],
                    );
                    let dy = sub_f64(
                        pad_pos[(py + 1) * pad_stride + px],
                        pad_pos[(py - 1) * pad_stride + px],
                    );
                    let mut n = normalize3_f64(cross_f64(dx, dy));
                    if dot3_f64(n, dirs[vi]) < 0.0 {
                        n = scale_f64(n, -1.0);
                    }
                    normals[vi] = [n[0] as f32, n[1] as f32, n[2] as f32];
                }
            }

            faces.push(MacroBaseFace {
                sub,
                dirs,
                surf_z,
                normals,
                albedos,
                emissives,
                roughness,
                water_depth,
                ice_frac,
            });
        }
        self.macro_base_faces = faces;
    }

    /// Assemble Layer A 6-face macro globe vertices & indices, sinking the overlap border and
    /// clipping out quads strictly inside any hydrated `1 m` shard footprint (zero Z-fighting!).
    fn rebuild_clipped_macro_globe(&mut self) {
        let sub = self.macro_sub;
        let stride = sub + 1;
        let r_core = self.r_core;
        let z_sea_surf = self.z_sea + 0.55;
        let spa = self.shards_per_axis;
        let inset_uv = 0.008f64;

        // Collect hydrated shard UV footprints per face
        let mut hydrated_bounds: [Vec<(f64, f64, f64, f64)>; FACE_COUNT] = Default::default();
        for &(f, sx, sy) in self.shards.keys() {
            if (f as usize) < FACE_COUNT {
                let bounds = PlanetShardKey::new(f, sx, sy).uv_bounds(spa);
                hydrated_bounds[f as usize].push(bounds);
            }
        }

        self.macro_terrain_verts.clear();
        self.macro_terrain_indices.clear();
        self.macro_ocean_verts.clear();
        self.macro_ocean_indices.clear();
        self.macro_terrain_verts.reserve(FACE_COUNT * stride * stride);
        self.macro_terrain_indices.reserve(FACE_COUNT * sub * sub * 6);
        self.macro_ocean_verts.reserve(FACE_COUNT * stride * stride);
        self.macro_ocean_indices.reserve(FACE_COUNT * sub * sub * 6);

        for (f, bf) in self.macro_base_faces.iter().enumerate() {
            let base_t = self.macro_terrain_verts.len() as u32;
            let base_o = self.macro_ocean_verts.len() as u32;
            let f_bounds = &hydrated_bounds[f];

            for iy in 0..=bf.sub {
                let v = (iy as f64 / bf.sub as f64) * 2.0 - 1.0;
                for ix in 0..=bf.sub {
                    let u = (ix as f64 / bf.sub as f64) * 2.0 - 1.0;
                    let vi = iy * stride + ix;
                    let dir = bf.dirs[vi];

                    // Smoothly sink macro overlap ring beneath hydrated 1m shard so Layer B always wins cleanly!
                    let mut r_offset = 0.0f64;
                    for &(u0, u1, v0, v1) in f_bounds {
                        if u > u0 && u < u1 && v > v0 && v < v1 {
                            let du = (u - u0).min(u1 - u);
                            let dv = (v - v0).min(v1 - v);
                            let d_in = du.min(dv);
                            let sink = -(d_in / inset_uv).clamp(0.0, 1.0) * 1.4;
                            if sink < r_offset {
                                r_offset = sink;
                            }
                        }
                    }

                    let pos = scale_f64(dir, r_core + bf.surf_z[vi] + r_offset);
                    self.macro_terrain_verts.push(MacroPlanetVertex {
                        pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32],
                        normal: bf.normals[vi],
                        color: [
                            bf.albedos[vi][0],
                            bf.albedos[vi][1],
                            bf.albedos[vi][2],
                            bf.roughness[vi],
                        ],
                        emissive_mat: [
                            bf.emissives[vi][0],
                            bf.emissives[vi][1],
                            bf.emissives[vi][2],
                            MAT_MACRO_TERRAIN,
                        ],
                        water_ice: [bf.water_depth[vi], bf.ice_frac[vi], 1.0, 0.0],
                    });

                    let o_pos = scale_f64(dir, r_core + z_sea_surf + r_offset * 0.5);
                    self.macro_ocean_verts.push(MacroPlanetVertex {
                        pos: [o_pos[0] as f32, o_pos[1] as f32, o_pos[2] as f32],
                        normal: [dir[0] as f32, dir[1] as f32, dir[2] as f32],
                        color: [0.04, 0.24, 0.54, 0.08],
                        emissive_mat: [0.0, 0.0, 0.0, MAT_MACRO_OCEAN],
                        water_ice: [bf.water_depth[vi], bf.ice_frac[vi], 1.0, 0.0],
                    });
                }
            }

            for iy in 0..bf.sub {
                let v_mid = ((iy as f64 + 0.5) / bf.sub as f64) * 2.0 - 1.0;
                for ix in 0..bf.sub {
                    let u_mid = ((ix as f64 + 0.5) / bf.sub as f64) * 2.0 - 1.0;
                    let clipped = f_bounds.iter().any(|&(u0, u1, v0, v1)| {
                        u_mid > u0 + inset_uv
                            && u_mid < u1 - inset_uv
                            && v_mid > v0 + inset_uv
                            && v_mid < v1 - inset_uv
                    });
                    if clipped {
                        continue;
                    }
                    let i00 = base_t + (iy * stride + ix) as u32;
                    let i10 = i00 + 1;
                    let i01 = base_t + ((iy + 1) * stride + ix) as u32;
                    let i11 = i01 + 1;
                    self.macro_terrain_indices
                        .extend_from_slice(&[i00, i10, i11, i00, i11, i01]);

                    let o00 = base_o + (iy * stride + ix) as u32;
                    let o10 = o00 + 1;
                    let o01 = base_o + ((iy + 1) * stride + ix) as u32;
                    let o11 = o01 + 1;
                    let vi00 = iy * stride + ix;
                    if bf.water_depth[vi00] > 0.01
                        || bf.water_depth[vi00 + 1] > 0.01
                        || bf.water_depth[vi00 + stride] > 0.01
                        || bf.water_depth[vi00 + stride + 1] > 0.01
                    {
                        self.macro_ocean_indices
                            .extend_from_slice(&[o00, o10, o11, o00, o11, o01]);
                    }
                }
            }
        }
    }

    /// Build all 12 Orbital Floating Sky-Islands with glowing cyan-amethyst Buoyant Crystal Core keels,
    /// sculpted emerald plateaus, sky-pines, and plunging waterfalls!
    fn build_macro_sky_islands(&mut self) {
        self.macro_sky_verts.clear();
        self.macro_sky_indices.clear();
        let r_core = self.r_core;
        let z_sea_surf = self.z_sea + 0.55;
        let origin = [0.0f64; 3];
        for isl in &self.sky_islands {
            push_sky_island(
                &mut self.macro_sky_verts,
                &mut self.macro_sky_indices,
                isl,
                r_core,
                z_sea_surf,
                origin,
                if isl.has_waterfall { Some(0.65) } else { None },
                true,
            );
        }
    }

    /// Extract smooth `1 m` locally-flat curved terrain, golden beaches, Fresnel ocean bay,
    /// `-4.5 m` edge skirts, 3D Nordic pine clusters, coastal boulders, live Sky-Island voxels
    /// (`z >= 44`), and translucent steam/volcanic plumes for one hydrated `256×256×64` shard!
    fn rebuild_single_shard_mesh(&mut self, key: (u8, u8, u8)) {
        let r_core = self.r_core;
        let z_sea_surf = self.z_sea + 0.55;
        let origin_f64 = self.origin_f64;
        let n_virt = self.face_size_virtual();
        let seed = self.seed;
        let sky_z_floor = self.sky_z_floor();

        let Some(sh) = self.shards.get_mut(&key) else {
            return;
        };

        let s = sh.shard_size;
        let d = sh.depth;
        let face = sh.face as usize;
        let gx_base = sh.sx as usize * s;
        let gy_base = sh.sy as usize * s;

        // Use full 1m grid on closest shard (`step = 1` for <=256) or 2m step (`step = 2`) for distant shards
        let shard_center = PlanetShardKey::new(sh.face, sh.sx, sh.sy).center_sphere_dir(self.shards_per_axis);
        let shard_world = scale_f64(shard_center, r_core + z_sea_surf);
        let step = if dist_f64(shard_world, origin_f64) < 280.0 { 1usize } else { 2usize };
        let cells = s / step;
        let stride = cells + 1;

        let mut terrain_verts = Vec::with_capacity(stride * stride + 16_384);
        let mut terrain_indices = Vec::with_capacity(cells * cells * 6 + 32_768);
        let mut ocean_verts = Vec::with_capacity(stride * stride);
        let mut ocean_indices = Vec::with_capacity(cells * cells * 6);
        let mut gas_verts = Vec::with_capacity(512);
        let mut gas_indices = Vec::with_capacity(768);

        let mut grid_pos = Vec::with_capacity(stride * stride);
        let mut grid_dir = Vec::with_capacity(stride * stride);
        let mut grid_sz = Vec::with_capacity(stride * stride);
        let mut grid_col = Vec::with_capacity(stride * stride);
        let mut grid_em = Vec::with_capacity(stride * stride);
        let mut grid_rough = Vec::with_capacity(stride * stride);
        let mut grid_wd = Vec::with_capacity(stride * stride);
        let mut grid_ice = Vec::with_capacity(stride * stride);

        for vy in 0..=cells {
            let ly_f = (vy * step).min(s);
            let ly = ly_f.min(s - 1);
            let gy = (gy_base + ly_f) as f64;
            for vx in 0..=cells {
                let lx_f = (vx * step).min(s);
                let lx = lx_f.min(s - 1);
                let gx = (gx_base + lx_f) as f64;
                let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
                let dir = face_uv_to_sphere_dir_f64(face, u, v);
                let cc = sh.column_cache[ly * s + lx];

                let pos = scale_f64(dir, r_core + cc.surf_z);
                grid_pos.push(pos);
                grid_dir.push(dir);
                grid_sz.push(cc.surf_z);
                grid_col.push(cc.albedo);
                grid_em.push(cc.emissive);
                grid_rough.push(cc.roughness);
                grid_wd.push(cc.water_depth);
                grid_ice.push(cc.ice_frac);
            }
        }

        for vy in 0..=cells {
            for vx in 0..=cells {
                let vi = vy * stride + vx;
                let xm = vx.saturating_sub(1);
                let xp = (vx + 1).min(cells);
                let ym = vy.saturating_sub(1);
                let yp = (vy + 1).min(cells);
                let dx = sub_f64(grid_pos[vy * stride + xp], grid_pos[vy * stride + xm]);
                let dy = sub_f64(grid_pos[yp * stride + vx], grid_pos[ym * stride + vx]);
                let mut n = normalize3_f64(cross_f64(dx, dy));
                if dot3_f64(n, grid_dir[vi]) < 0.0 {
                    n = scale_f64(n, -1.0);
                }

                terrain_verts.push(MacroPlanetVertex {
                    pos: world_f64_to_camera_rel_f32(grid_pos[vi], origin_f64),
                    normal: [n[0] as f32, n[1] as f32, n[2] as f32],
                    color: [grid_col[vi][0], grid_col[vi][1], grid_col[vi][2], grid_rough[vi]],
                    emissive_mat: [grid_em[vi][0], grid_em[vi][1], grid_em[vi][2], MAT_MACRO_TERRAIN],
                    water_ice: [grid_wd[vi], grid_ice[vi], 0.0, 0.0],
                });

                let ocean_pos = scale_f64(grid_dir[vi], r_core + z_sea_surf);
                ocean_verts.push(MacroPlanetVertex {
                    pos: world_f64_to_camera_rel_f32(ocean_pos, origin_f64),
                    normal: [grid_dir[vi][0] as f32, grid_dir[vi][1] as f32, grid_dir[vi][2] as f32],
                    color: [0.04, 0.24, 0.54, 0.08],
                    emissive_mat: [0.0, 0.0, 0.0, MAT_MACRO_OCEAN],
                    water_ice: [grid_wd[vi], grid_ice[vi], 0.0, 0.0],
                });
            }
        }

        for vy in 0..cells {
            for vx in 0..cells {
                let i00 = (vy * stride + vx) as u32;
                let i10 = i00 + 1;
                let i01 = ((vy + 1) * stride + vx) as u32;
                let i11 = i01 + 1;
                terrain_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                if grid_wd[i00 as usize] > 0.01
                    || grid_wd[i10 as usize] > 0.01
                    || grid_wd[i01 as usize] > 0.01
                    || grid_wd[i11 as usize] > 0.01
                {
                    ocean_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                }
            }
        }

        // `-4.5 m` radial skirt around all 4 outer shard boundaries (zero grazing seam cracks!)
        let mut emit_skirt = |coords: Vec<(usize, usize)>, reverse: bool| {
            let base = terrain_verts.len() as u32;
            for &(vx, vy) in &coords {
                let vi = vy * stride + vx;
                let top_v = terrain_verts[vi];
                let bot_pos = scale_f64(grid_dir[vi], r_core + grid_sz[vi] - 4.5);
                let mut bot_v = top_v;
                bot_v.pos = world_f64_to_camera_rel_f32(bot_pos, origin_f64);
                terrain_verts.push(top_v);
                terrain_verts.push(bot_v);
            }
            for i in 0..coords.len().saturating_sub(1) {
                let a = base + (i * 2) as u32;
                let b = a + 1;
                let c = a + 2;
                let d_idx = a + 3;
                if reverse {
                    terrain_indices.extend_from_slice(&[a, c, d_idx, a, d_idx, b]);
                } else {
                    terrain_indices.extend_from_slice(&[a, d_idx, c, a, b, d_idx]);
                }
            }
        };
        emit_skirt((0..=cells).map(|vx| (vx, 0)).collect(), false);
        emit_skirt((0..=cells).map(|vx| (vx, cells)).collect(), true);
        emit_skirt((0..=cells).map(|vy| (0, vy)).collect(), true);
        emit_skirt((0..=cells).map(|vy| (cells, vy)).collect(), false);

        // Sculpted 3D Nordic pine clusters & coastal granite boulders on forested/coastal voxels!
        let prop_step = if step == 1 { 5usize } else { 8usize };
        for ly in (3..s - 3).step_by(prop_step) {
            for lx in (3..s - 3).step_by(prop_step) {
                let h = ((lx.wrapping_mul(73_856_093))
                    ^ (ly.wrapping_mul(19_349_663))
                    ^ (seed as usize))
                    % 1000;
                let jx = (lx + (h % 3)).min(s - 2);
                let jy = (ly + ((h / 7) % 3)).min(s - 2);
                let cc = sh.column_cache[jy * s + jx];
                let sz = cc.surf_z;
                if sz <= z_sea_surf + 0.15 {
                    continue;
                }
                let (u, v) = face_virtual_uv_f64((gx_base + jx) as f64 + 0.5, (gy_base + jy) as f64 + 0.5, n_virt);
                let dir = face_uv_to_sphere_dir_f64(face, u, v);
                let pos = scale_f64(dir, r_core + sz);

                if sz > z_sea_surf + 2.45 && sz < 35.5 && (cc.has_wood || cc.has_plant) && (h % 100) < 24 {
                    let tree_h = 5.4 + ((h % 37) as f64 / 37.0) * 4.4;
                    let tree_r = 1.55 + ((h % 19) as f64 / 19.0) * 0.95;
                    let tint = ((h % 11) as f32) / 11.0;
                    push_pine_tree(
                        &mut terrain_verts,
                        &mut terrain_indices,
                        pos,
                        dir,
                        origin_f64,
                        tree_h,
                        tree_r,
                        tint,
                    );
                } else if sz < z_sea_surf + 3.4 && (h % 100) < 9 {
                    let r_b = 0.85 + ((h % 13) as f64 / 13.0) * 1.15;
                    push_boulder(
                        &mut terrain_verts,
                        &mut terrain_indices,
                        pos,
                        dir,
                        origin_f64,
                        r_b * 1.2,
                        r_b * 0.9,
                        r_b * 0.65,
                    );
                }
            }
        }

        // Live Floating Sky-Island Voxels (`z >= sky_z_floor`) received in hydrated shard sub-bricks!
        // Scan only sub-bricks that have actually been received at `z >= sky_z_floor`.
        let mut sky_bricks: Vec<(u16, u16, u16)> = sh
            .received_subbricks
            .iter()
            .copied()
            .filter(|&(_, _, bz)| ((bz as usize) + 1) * CHUNK_SIZE > sky_z_floor)
            .collect();
        sky_bricks.sort_unstable();
        for (bx, by, bz) in sky_bricks {
            let x0 = (bx as usize) * CHUNK_SIZE;
            let y0 = (by as usize) * CHUNK_SIZE;
            let z0 = ((bz as usize) * CHUNK_SIZE).max(sky_z_floor);
            let x1 = (x0 + CHUNK_SIZE).min(s);
            let y1 = (y0 + CHUNK_SIZE).min(s);
            let z1 = (((bz as usize) + 1) * CHUNK_SIZE).min(d);

            for z in (z0..z1).step_by(2) {
                for y in (y0..y1).step_by(2) {
                    for x in (x0..x1).step_by(2) {
                        let off = sh.voxel_offset(x, y, z);
                        let solid = sh.density_mat[off];
                        if solid < 100 {
                            continue;
                        }
                        let air_above = sh.solid_at(x as isize, y as isize, z as isize + 2) < 100;
                        let air_below = sh.solid_at(x as isize, y as isize, z as isize - 2) < 100;
                        let air_side = sh.solid_at(x as isize + 2, y as isize, z as isize) < 100
                            || sh.solid_at(x as isize - 2, y as isize, z as isize) < 100
                            || sh.solid_at(x as isize, y as isize + 2, z as isize) < 100
                            || sh.solid_at(x as isize, y as isize - 2, z as isize) < 100;
                        if !air_above && !air_below && !air_side {
                            continue;
                        }
                        let ar = sh.albedo_thermal[off] as f32 / 255.0;
                        let ag = sh.albedo_thermal[off + 1] as f32 / 255.0;
                        let ab = sh.albedo_thermal[off + 2] as f32 / 255.0;
                        let therm = sh.albedo_thermal[off + 3] as f32 / 255.0;
                        let rough = (sh.density_mat[off + 3] as f32 / 255.0).clamp(0.15, 0.92);
                        let is_crystal = z <= sky_z_floor + 5 && ab > ar + 0.05 && ab > 0.42 && therm < 0.25;

                        let (u, v) = face_virtual_uv_f64(
                            (gx_base + x) as f64 + 1.0,
                            (gy_base + y) as f64 + 1.0,
                            n_virt,
                        );
                        let (up, tu, tv) = face_tangent_basis_f64(face, u, v);
                        let center = scale_f64(up, r_core + z as f64 + 1.0);
                        let (col, em_mat) = if is_crystal {
                            ([0.16, 0.80, 1.0, 0.16], [0.28, 1.65, 2.85, MAT_MACRO_CRYSTAL])
                        } else if air_above {
                            ([ar.max(0.08), ag.max(0.28), ab.max(0.10), rough], [0.0, 0.0, 0.0, MAT_MACRO_PROP])
                        } else {
                            ([0.18 + ar * 0.14, 0.19 + ag * 0.14, 0.23 + ab * 0.14, rough], [0.0, 0.0, 0.0, MAT_MACRO_PROP])
                        };
                        push_oriented_box_f64(
                            &mut terrain_verts,
                            &mut terrain_indices,
                            center,
                            tu,
                            up,
                            tv,
                            [2.05, 2.05, 2.05],
                            origin_f64,
                            col,
                            em_mat,
                        );
                    }
                }
            }
        }

        // Translucent Volumetric Steam / Volcanic Ash Plumes (`gas_density >= 48`) from live sub-bricks!
        for &(bx, by, bz) in &sh.received_subbricks {
            let x0 = (bx as usize) * CHUNK_SIZE;
            let y0 = (by as usize) * CHUNK_SIZE;
            let z0 = (bz as usize) * CHUNK_SIZE;
            let x1 = (x0 + CHUNK_SIZE).min(s);
            let y1 = (y0 + CHUNK_SIZE).min(s);
            let z1 = (z0 + CHUNK_SIZE).min(d);

            for z in (z0..z1).step_by(4) {
                for y in (y0..y1).step_by(4) {
                    for x in (x0..x1).step_by(4) {
                        let off = sh.voxel_offset(x, y, z);
                        let gas = sh.density_mat[off + 2];
                        if gas < 48 {
                            continue;
                        }
                        let g_frac = (gas as f32 / 255.0).clamp(0.18, 0.72);
                        let ar = sh.albedo_thermal[off] as f32 / 255.0;
                        let ag = sh.albedo_thermal[off + 1] as f32 / 255.0;
                        let ab = sh.albedo_thermal[off + 2] as f32 / 255.0;
                        let therm = sh.albedo_thermal[off + 3] as f32 / 255.0;

                        let (u, v) = face_virtual_uv_f64(
                            (gx_base + x) as f64 + 0.5,
                            (gy_base + y) as f64 + 0.5,
                            n_virt,
                        );
                        let (up, tu, tv) = face_tangent_basis_f64(face, u, v);
                        let center = scale_f64(up, r_core + z as f64 + 1.0);
                        let half_w = 2.2 + g_frac as f64 * 1.6;
                        let half_h = 2.6 + g_frac as f64 * 1.8;
                        let col = [
                            ar.max(0.28),
                            ag.max(0.28),
                            ab.max(0.32),
                            (g_frac * 0.42).clamp(0.12, 0.36),
                        ];
                        let em_mat = if therm > 0.35 {
                            [therm * 1.5, therm * 0.55, therm * 0.12, MAT_MACRO_GAS]
                        } else {
                            [0.0, 0.0, 0.0, MAT_MACRO_GAS]
                        };
                        let n_f = [up[0] as f32, up[1] as f32, up[2] as f32];
                        for axis in [tu, tv] {
                            let base = gas_verts.len() as u32;
                            let p0 = sub_f64(sub_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h * 0.4));
                            let p1 = sub_f64(add_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h * 0.4));
                            let p2 = add_f64(add_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h));
                            let p3 = add_f64(sub_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h));
                            for p in [p0, p1, p2, p3] {
                                gas_verts.push(MacroPlanetVertex {
                                    pos: world_f64_to_camera_rel_f32(p, origin_f64),
                                    normal: n_f,
                                    color: col,
                                    emissive_mat: em_mat,
                                    water_ice: [0.0; 4],
                                });
                            }
                            gas_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                        }
                    }
                }
            }
        }

        sh.terrain_verts = terrain_verts;
        sh.terrain_indices = terrain_indices;
        sh.ocean_verts = ocean_verts;
        sh.ocean_indices = ocean_indices;
        sh.gas_verts = gas_verts;
        sh.gas_indices = gas_indices;
        sh.dirty = false;
    }

    fn merge_near_shard_meshes(&mut self) {
        self.near_terrain_verts.clear();
        self.near_terrain_indices.clear();
        self.near_ocean_verts.clear();
        self.near_ocean_indices.clear();
        self.near_gas_verts.clear();
        self.near_gas_indices.clear();

        for sh in self.shards.values() {
            let tv_base = self.near_terrain_verts.len() as u32;
            self.near_terrain_verts.extend_from_slice(&sh.terrain_verts);
            self.near_terrain_indices
                .extend(sh.terrain_indices.iter().map(|&i| i + tv_base));

            let ov_base = self.near_ocean_verts.len() as u32;
            self.near_ocean_verts.extend_from_slice(&sh.ocean_verts);
            self.near_ocean_indices
                .extend(sh.ocean_indices.iter().map(|&i| i + ov_base));

            let gv_base = self.near_gas_verts.len() as u32;
            self.near_gas_verts.extend_from_slice(&sh.gas_verts);
            self.near_gas_indices
                .extend(sh.gas_indices.iter().map(|&i| i + gv_base));
        }
    }

    // =========================================================================
    // Equirectangular Minimap (Continents + Oceans + Live PlanetaryClimateNet Storms/Ash!)
    // =========================================================================

    fn sample_climate_rgba_cpu(&self, dir: [f64; 3]) -> [f32; 4] {
        let (face, u, v) = sphere_dir_to_face_uv_f64(dir);
        let r = self.climate_res;
        let x = (((u + 1.0) * 0.5 * r as f64).floor() as isize).clamp(0, r as isize - 1) as usize;
        let y = (((v + 1.0) * 0.5 * r as f64).floor() as isize).clamp(0, r as isize - 1) as usize;
        let off = (y * r + x) * 4;
        let buf = &self.climate_faces[face % FACE_COUNT];
        [
            buf[off] as f32 / 255.0,
            buf[off + 1] as f32 / 255.0,
            buf[off + 2] as f32 / 255.0,
            buf[off + 3] as f32 / 255.0,
        ]
    }

    pub fn rebuild_equirectangular_minimap(&mut self) {
        let w = MACRO_MINIMAP_W;
        let h = MACRO_MINIMAP_H;
        if self.minimap_rgba.len() != w * h * 4 {
            self.minimap_rgba.resize(w * h * 4, 255);
        }
        let seed = self.seed;
        for py in 0..h {
            let lat = std::f64::consts::FRAC_PI_2
                - ((py as f64 + 0.5) / h as f64) * std::f64::consts::PI;
            let clat = lat.cos();
            let slat = lat.sin();
            for px in 0..w {
                let lon = ((px as f64 + 0.5) / w as f64) * std::f64::consts::TAU
                    - std::f64::consts::PI;
                let dir = [clat * lon.cos(), slat, clat * lon.sin()];
                let col = sample_macro_planet_column(dir, seed);
                let clim = self.sample_climate_rgba_cpu(dir);

                let mut rgb = if col.is_rift_core {
                    [1.0f32, 0.46, 0.14]
                } else if col.water_depth > 0.1 {
                    if clim[3] > 0.72 {
                        [0.72, 0.88, 0.98]
                    } else {
                        [0.05, 0.22, 0.52]
                    }
                } else {
                    let mut c = [
                        (col.albedo[0] * 1.45).clamp(0.0, 1.0),
                        (col.albedo[1] * 1.45).clamp(0.0, 1.0),
                        (col.albedo[2] * 1.45).clamp(0.0, 1.0),
                    ];
                    if clim[3] > 0.52 && !col.is_rift_rim {
                        let s_w = ((clim[3] - 0.52) / 0.48).clamp(0.0, 0.8);
                        c[0] = c[0] * (1.0 - s_w) + 0.88 * s_w;
                        c[1] = c[1] * (1.0 - s_w) + 0.93 * s_w;
                        c[2] = c[2] * (1.0 - s_w) + 0.98 * s_w;
                    }
                    c
                };

                // Overlay live PlanetaryClimateNet storm clouds (G) & volcanic ash plumes (B)!
                let cloud_a = ((clim[1] - 0.25) / 0.75).clamp(0.0, 1.0) * 0.58;
                rgb[0] = rgb[0] * (1.0 - cloud_a) + 0.94 * cloud_a;
                rgb[1] = rgb[1] * (1.0 - cloud_a) + 0.97 * cloud_a;
                rgb[2] = rgb[2] * (1.0 - cloud_a) + 1.00 * cloud_a;

                let ash_a = ((clim[2] - 0.20) / 0.80).clamp(0.0, 1.0) * 0.55;
                rgb[0] = rgb[0] * (1.0 - ash_a) + 0.36 * ash_a;
                rgb[1] = rgb[1] * (1.0 - ash_a) + 0.28 * ash_a;
                rgb[2] = rgb[2] * (1.0 - ash_a) + 0.26 * ash_a;

                let off = (py * w + px) * 4;
                self.minimap_rgba[off] = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
                self.minimap_rgba[off + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
                self.minimap_rgba[off + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
                self.minimap_rgba[off + 3] = 255;
            }
        }
        self.minimap_dirty = true;
    }

    // =========================================================================
    // Layer C: Live 60 3D Heroes, Shrines, Beams & Projectiles in `f64` Floating Origin
    // =========================================================================

    pub fn sample_surface_z_f64(&self, face: u8, gx: f64, gy: f64) -> f64 {
        let n_virt = self.face_size_virtual();
        let s = self.shard_size;
        let sx = ((gx.max(0.0) as usize) / s).min(self.shards_per_axis.saturating_sub(1) as usize) as u8;
        let sy = ((gy.max(0.0) as usize) / s).min(self.shards_per_axis.saturating_sub(1) as usize) as u8;
        if let Some(sh) = self.shards.get(&(face, sx, sy)) {
            let lx = (gx.max(0.0) as usize % s).min(s - 1);
            let ly = (gy.max(0.0) as usize % s).min(s - 1);
            return sh.column_cache[ly * s + lx].surf_z;
        }
        let (u, v) = face_virtual_uv_f64(gx, gy, n_virt);
        let dir = face_uv_to_sphere_dir_f64(face as usize, u, v);
        sample_macro_planet_column(dir, self.seed).surf_z
    }

    pub fn entity_world_frame_f64(
        &self,
        face: u8,
        gx: f32,
        gy: f32,
        gz: f32,
        aim_x: f32,
        aim_y: f32,
    ) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let n_virt = self.face_size_virtual();
        let (u, v) = face_virtual_uv_f64(gx as f64, gy as f64, n_virt);
        let (up, tang_u, tang_v) = face_tangent_basis_f64(face as usize, u, v);
        let surf_z = self.sample_surface_z_f64(face, gx as f64, gy as f64);
        let z_eff = if gz > 0.5 {
            (gz as f64).max(surf_z)
        } else {
            surf_z.max(self.z_sea + 0.55)
        };
        let pos = scale_f64(up, self.r_core + z_eff);
        let fwd_raw = add_f64(scale_f64(tang_u, aim_x as f64), scale_f64(tang_v, aim_y as f64));
        let fwd = if dot3_f64(fwd_raw, fwd_raw) > 1e-6 {
            normalize3_f64(fwd_raw)
        } else {
            tang_u
        };
        let right = normalize3_f64(cross_f64(fwd, up));
        let fwd_ortho = normalize3_f64(cross_f64(up, right));
        (pos, up, right, fwd_ortho)
    }

    pub fn rebuild_dynamic_entities(&mut self, entities: Option<&LiveEntityState>, sim_time: f32) {
        self.dynamic_solid_verts.clear();
        self.dynamic_solid_indices.clear();
        self.dynamic_fx_verts.clear();
        self.dynamic_fx_indices.clear();

        let Some(ent) = entities else {
            return;
        };
        let origin = self.origin_f64;

        // 1. All 60 Live Neural Bots & Human Hero at exact 1m human scale (~2m tall!)
        for p in &ent.players {
            if p.hp <= 0 {
                continue;
            }
            let (base_pos, up, right, fwd) =
                self.entity_world_frame_f64(p.face, p.x, p.y, p.z, p.aim_x, p.aim_y);
            let is_self = p.id == ent.your_player_id;

            let team_col = if is_self {
                [1.00, 0.88, 0.28]
            } else if p.team == 0 {
                [0.20, 0.78, 1.00]
            } else {
                [1.00, 0.36, 0.18]
            };
            let accent_col = if p.team == 0 {
                [0.65, 0.95, 1.00]
            } else {
                [1.00, 0.82, 0.32]
            };

            let sc = if is_self { 1.15f64 } else { 1.0f64 };
            let em_s = if is_self { 0.55f32 } else if p.ult_active { 1.25 } else { 0.32 };
            let body_em = [team_col[0] * em_s, team_col[1] * em_s, team_col[2] * em_s, MAT_MACRO_ENTITY];
            let acc_em = [accent_col[0] * (em_s + 0.45), accent_col[1] * (em_s + 0.45), accent_col[2] * (em_s + 0.45), MAT_MACRO_ENTITY];

            match p.class_id {
                CLASS_SKYWING => {
                    let bob = ((sim_time * 4.5 + p.id as f32).sin() * 0.25) as f64;
                    let center = add_f64(base_pos, scale_f64(up, 1.45 * sc + bob));
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        center,
                        right,
                        up,
                        fwd,
                        [0.55 * sc, 0.70 * sc, 1.15 * sc],
                        origin,
                        [team_col[0], team_col[1], team_col[2], 0.30],
                        body_em,
                    );
                    let flap = ((sim_time * 7.2 + p.id as f32 * 1.3).sin() * 0.35) as f64;
                    for side in [-1.0f64, 1.0] {
                        let w_dir = normalize3_f64(add_f64(scale_f64(right, side), scale_f64(up, 0.22 + flap)));
                        let w_up = scale_f64(normalize3_f64(cross_f64(fwd, w_dir)), side);
                        let w_center = sub_f64(add_f64(center, scale_f64(w_dir, 1.25 * sc)), scale_f64(fwd, 0.15 * sc));
                        push_oriented_box_f64(
                            &mut self.dynamic_solid_verts,
                            &mut self.dynamic_solid_indices,
                            w_center,
                            w_dir,
                            w_up,
                            fwd,
                            [1.35 * sc, 0.14 * sc, 0.65 * sc],
                            origin,
                            [accent_col[0], accent_col[1], accent_col[2], 0.22],
                            acc_em,
                        );
                    }
                }
                CLASS_STRIDER => {
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 1.0 * sc)),
                        right,
                        up,
                        fwd,
                        [0.95 * sc, 1.25 * sc, 0.72 * sc],
                        origin,
                        [team_col[0], team_col[1], team_col[2], 0.36],
                        body_em,
                    );
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 1.92 * sc)),
                        right,
                        up,
                        fwd,
                        [0.56 * sc, 0.52 * sc, 0.56 * sc],
                        origin,
                        [accent_col[0], accent_col[1], accent_col[2], 0.24],
                        acc_em,
                    );
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(add_f64(base_pos, scale_f64(up, 1.05 * sc)), scale_f64(fwd, 0.62 * sc)),
                        right,
                        up,
                        fwd,
                        [0.88 * sc, 1.18 * sc, 0.18 * sc],
                        origin,
                        [accent_col[0], accent_col[1], accent_col[2], 0.18],
                        acc_em,
                    );
                }
                CLASS_TIDEKIN => {
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 0.92 * sc)),
                        right,
                        up,
                        fwd,
                        [0.72 * sc, 1.15 * sc, 0.68 * sc],
                        origin,
                        [team_col[0], team_col[1], team_col[2], 0.24],
                        body_em,
                    );
                    push_oriented_pyramid_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 1.48 * sc)),
                        right,
                        up,
                        fwd,
                        0.42 * sc,
                        0.95 * sc,
                        origin,
                        [accent_col[0], accent_col[1], accent_col[2], 0.20],
                        acc_em,
                    );
                }
                CLASS_EMBERSTALKER => {
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 0.88 * sc)),
                        right,
                        up,
                        fwd,
                        [0.68 * sc, 1.06 * sc, 0.86 * sc],
                        origin,
                        [1.0, 0.44, 0.18, 0.30],
                        [0.65, 0.26, 0.08, MAT_MACRO_ENTITY],
                    );
                    for side in [-0.58f64, 0.58] {
                        push_oriented_box_f64(
                            &mut self.dynamic_solid_verts,
                            &mut self.dynamic_solid_indices,
                            add_f64(
                                add_f64(add_f64(base_pos, scale_f64(up, 1.0 * sc)), scale_f64(right, side * sc)),
                                scale_f64(fwd, 0.52 * sc),
                            ),
                            right,
                            up,
                            fwd,
                            [0.14 * sc, 0.20 * sc, 1.18 * sc],
                            origin,
                            [1.0, 0.78, 0.26, 0.16],
                            [1.45, 1.05, 0.28, MAT_MACRO_ENTITY],
                        );
                    }
                }
                CLASS_BURROWER | _ => {
                    push_oriented_box_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 0.82 * sc)),
                        right,
                        up,
                        fwd,
                        [1.12 * sc, 0.90 * sc, 1.20 * sc],
                        origin,
                        [team_col[0], team_col[1], team_col[2], 0.34],
                        body_em,
                    );
                    push_oriented_pyramid_f64(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        add_f64(base_pos, scale_f64(up, 1.28 * sc)),
                        right,
                        up,
                        fwd,
                        0.62 * sc,
                        0.88 * sc,
                        origin,
                        [0.44, 0.94, 0.88, 0.18],
                        [0.25, 0.92, 0.85, MAT_MACRO_ENTITY],
                    );
                }
            }

            // Crisp team ground aura ring beneath every hero
            push_spherical_ring_f64(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                add_f64(base_pos, scale_f64(up, 0.16)),
                up,
                right,
                fwd,
                if is_self { 1.25 } else { 0.95 },
                if is_self { 1.75 } else { 1.35 },
                12,
                origin,
                [team_col[0], team_col[1], team_col[2], if is_self { 0.84 } else { 0.62 }],
                [team_col[0] * 1.4, team_col[1] * 1.4, team_col[2] * 1.4, MAT_MACRO_BEAM],
            );
        }

        // 2. Planetary Shrines with tall vertical light pillars visible from Skywing altitude & Orbit!
        for s in &ent.shrines {
            let (base_pos, up, right, fwd) =
                self.entity_world_frame_f64(s.face, s.x, s.y, 0.0, 1.0, 0.0);
            let shrine_col = if s.owner < 0 {
                [0.22, 0.80, 1.00]
            } else if s.owner > 0 {
                [1.00, 0.38, 0.18]
            } else {
                [1.00, 0.86, 0.34]
            };

            push_oriented_box_f64(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                add_f64(base_pos, scale_f64(up, 0.8)),
                right,
                up,
                fwd,
                [3.4, 1.6, 3.4],
                origin,
                [0.34, 0.38, 0.48, 0.45],
                [0.08, 0.10, 0.14, MAT_MACRO_ENTITY],
            );
            push_oriented_pyramid_f64(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                add_f64(base_pos, scale_f64(up, 1.6)),
                right,
                up,
                fwd,
                1.45,
                6.8,
                origin,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.16],
                [shrine_col[0] * 1.8, shrine_col[1] * 1.8, shrine_col[2] * 1.8, MAT_MACRO_ENTITY],
            );
            let spin = (sim_time * 1.6 + s.id as f32) as f64;
            let r_rot = add_f64(scale_f64(right, spin.cos()), scale_f64(fwd, spin.sin()));
            let f_rot = add_f64(scale_f64(right, -spin.sin()), scale_f64(fwd, spin.cos()));
            push_oriented_box_f64(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                add_f64(base_pos, scale_f64(up, 9.2 + ((sim_time * 2.5 + s.id as f32).sin() * 0.45) as f64)),
                r_rot,
                up,
                f_rot,
                [1.3, 1.8, 1.3],
                origin,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.12],
                [shrine_col[0] * 2.4, shrine_col[1] * 2.4, shrine_col[2] * 2.4, MAT_MACRO_ENTITY],
            );
            push_spherical_ring_f64(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                add_f64(base_pos, scale_f64(up, 0.32)),
                up,
                right,
                fwd,
                3.6,
                4.8,
                20,
                origin,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.78],
                [shrine_col[0] * 1.8, shrine_col[1] * 1.8, shrine_col[2] * 1.8, MAT_MACRO_BEAM],
            );
            // Tall glowing orbital light pillar (68 m tall -> visible from Ground, Flight, Strato & Orbit!)
            push_radial_pillar_f64(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                add_f64(base_pos, scale_f64(up, 2.0)),
                up,
                right,
                fwd,
                1.35,
                68.0,
                origin,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.44],
                [shrine_col[0] * 2.6, shrine_col[1] * 2.6, shrine_col[2] * 2.6, MAT_MACRO_BEAM],
            );
        }

        // 3. Curved Geodesic Elemental Actuator Beams (`+Heat`, `-Cryo`, Oil, Water, Smoke)
        for pulse in &ent.actuator_pulses {
            let (p0_base, up0, _, _) =
                self.entity_world_frame_f64(pulse.face, pulse.x0, pulse.y0, pulse.z0, 1.0, 0.0);
            let (p1_base, up1, r1, f1) =
                self.entity_world_frame_f64(pulse.face, pulse.x1, pulse.y1, pulse.z1, 1.0, 0.0);
            let p0 = add_f64(p0_base, scale_f64(up0, 1.35));
            let p1 = add_f64(p1_base, scale_f64(up1, 1.15));

            let (beam_col, em_val) = if pulse.thermal_flux > 0.2 {
                ([1.00, 0.56, 0.14], 3.2f32)
            } else if pulse.thermal_flux < -0.2 {
                ([0.32, 0.94, 1.00], 3.0)
            } else if pulse.sp_oil > 0.2 {
                ([0.78, 0.28, 1.00], 2.4)
            } else if pulse.sp_water > 0.2 {
                ([0.24, 0.78, 1.00], 2.2)
            } else {
                ([0.72, 0.78, 0.90], 1.2)
            };

            let segs = 10usize;
            let r0 = (dot3_f64(p0, p0)).sqrt();
            let r1_len = (dot3_f64(p1, p1)).sqrt();
            let d0 = scale_f64(p0, 1.0 / r0.max(1e-6));
            let d1 = scale_f64(p1, 1.0 / r1_len.max(1e-6));
            let arc_lift = (dist_f64(p0, p1) * 0.08).clamp(0.4, 3.0);
            let beam_w = (0.16 + pulse.focus_radius as f64 * 0.03).clamp(0.16, 0.36);

            let mut pts = Vec::with_capacity(segs + 1);
            for si in 0..=segs {
                let t = si as f64 / segs as f64;
                let dir = normalize3_f64([
                    d0[0] + (d1[0] - d0[0]) * t,
                    d0[1] + (d1[1] - d0[1]) * t,
                    d0[2] + (d1[2] - d0[2]) * t,
                ]);
                let r = r0 + (r1_len - r0) * t + (t * std::f64::consts::PI).sin() * arc_lift;
                pts.push(scale_f64(dir, r));
            }
            for si in 0..segs {
                let a = pts[si];
                let b = pts[si + 1];
                let seg_dir = normalize3_f64(sub_f64(b, a));
                let rad_mid = normalize3_f64(add_f64(a, b));
                let side = normalize3_f64(cross_f64(seg_dir, rad_mid));
                let n_f = [rad_mid[0] as f32, rad_mid[1] as f32, rad_mid[2] as f32];
                for perp in [side, rad_mid] {
                    let base = self.dynamic_fx_verts.len() as u32;
                    let c = [beam_col[0], beam_col[1], beam_col[2], 0.88];
                    let em = [beam_col[0] * em_val, beam_col[1] * em_val, beam_col[2] * em_val, MAT_MACRO_BEAM];
                    for pos in [
                        sub_f64(a, scale_f64(perp, beam_w)),
                        add_f64(a, scale_f64(perp, beam_w)),
                        add_f64(b, scale_f64(perp, beam_w)),
                        sub_f64(b, scale_f64(perp, beam_w)),
                    ] {
                        self.dynamic_fx_verts.push(MacroPlanetVertex {
                            pos: world_f64_to_camera_rel_f32(pos, origin),
                            normal: n_f,
                            color: c,
                            emissive_mat: em,
                            water_ice: [0.0; 4],
                        });
                    }
                    self.dynamic_fx_indices
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            let flare_sz = (0.42 + pulse.focus_radius as f64 * 0.08).clamp(0.42, 0.95);
            push_oriented_box_f64(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                p1,
                r1,
                up1,
                f1,
                [flare_sz, flare_sz, flare_sz],
                origin,
                [beam_col[0], beam_col[1], beam_col[2], 0.15],
                [beam_col[0] * em_val * 1.15, beam_col[1] * em_val * 1.15, beam_col[2] * em_val * 1.15, MAT_MACRO_BEAM],
            );
        }

        // 4. Live 3D Elemental Projectiles
        for pr in &ent.projectiles {
            let (base_pos, up, right, fwd) =
                self.entity_world_frame_f64(pr.face, pr.x, pr.y, pr.z, pr.vx, pr.vy);
            let p_pos = add_f64(base_pos, scale_f64(up, 1.25));
            let (pcol, em_v) = match pr.element {
                0 => ([1.00, 0.54, 0.14], 3.4f32),
                1 => ([0.36, 0.94, 1.00], 3.2),
                2 => ([0.82, 0.34, 1.00], 2.8),
                _ => ([1.00, 0.86, 0.32], 3.8),
            };
            let sz = (pr.radius as f64 * 0.65).clamp(0.38, 1.25);
            push_oriented_box_f64(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                p_pos,
                right,
                up,
                fwd,
                [sz, sz, sz * 2.0],
                origin,
                [pcol[0], pcol[1], pcol[2], 0.12],
                [pcol[0] * em_v, pcol[1] * em_v, pcol[2] * em_v, MAT_MACRO_BEAM],
            );
        }
    }

    /// Find the local hero (or sunlit coastal combat cluster) and nearest Floating Sky-Island direction
    /// for headless capture shots!
    pub fn find_hero_or_best_cluster_f64(
        &self,
        entities: Option<&LiveEntityState>,
        sun_dir: [f64; 3],
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let mut focus_pos = scale_f64(normalize3_f64(sun_dir), self.r_surf() + 1.5);
        let mut focus_fwd = tangent_basis_f64(normalize3_f64(sun_dir)).0;
        let mut found = false;

        if let Some(ent) = entities {
            if let Some(me) = ent
                .players
                .iter()
                .find(|p| p.id == ent.your_player_id && p.hp > 0)
                .or_else(|| ent.players.iter().find(|p| p.hp > 0))
            {
                let (wp, _, _, fwd) =
                    self.entity_world_frame_f64(me.face, me.x, me.y, me.z, me.aim_x, me.aim_y);
                focus_pos = wp;
                focus_fwd = fwd;
                found = true;
            }
        }
        if !found {
            if let Some(&(f, sx, sy)) = self.shards.keys().next() {
                let cdir = PlanetShardKey::new(f, sx, sy).center_sphere_dir(self.shards_per_axis);
                let col = sample_macro_planet_column(cdir, self.seed);
                focus_pos = scale_f64(cdir, self.r_core + col.surf_z.max(self.z_sea + 1.0));
                focus_fwd = tangent_basis_f64(cdir).0;
            }
        }

        // Find the Sky-Island closest to `focus_pos`
        let f_dir = normalize3_f64(focus_pos);
        let nearest_isl = self
            .sky_islands
            .iter()
            .max_by(|a, b| {
                dot3_f64(a.center_dir, f_dir)
                    .partial_cmp(&dot3_f64(b.center_dir, f_dir))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|isl| isl.center_dir)
            .unwrap_or_else(|| normalize3_f64(add_f64(f_dir, scale_f64(focus_fwd, 90.0 / self.r_surf()))));

        (focus_pos, focus_fwd, nearest_isl)
    }
}

// =============================================================================
// Sculpted 3D Spherical Prop, Sky-Island & Entity Geometry Helpers (`f64` Origin)
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn push_pine_tree(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up_dir: [f64; 3],
    origin_f64: [f64; 3],
    height_m: f64,
    radius_m: f64,
    tint_var: f32,
) {
    let (t1, t2) = tangent_basis_f64(up_dir);
    let mut emit_cone = |z0: f64,
                         z1: f64,
                         r0: f64,
                         r1: f64,
                         segs: usize,
                         col_bot: [f32; 3],
                         col_top: [f32; 3],
                         rough: f32| {
        let base_idx = verts.len() as u32;
        for s in 0..=segs {
            let ang = (s as f64 / segs as f64) * std::f64::consts::TAU;
            let rad_dir = add_f64(scale_f64(t1, ang.cos()), scale_f64(t2, ang.sin()));
            let n_dir = normalize3_f64(add_f64(scale_f64(rad_dir, 0.85), scale_f64(up_dir, 0.52)));
            let n_f32 = [n_dir[0] as f32, n_dir[1] as f32, n_dir[2] as f32];
            let p0 = add_f64(add_f64(base_world, scale_f64(up_dir, z0)), scale_f64(rad_dir, r0));
            let p1 = add_f64(add_f64(base_world, scale_f64(up_dir, z1)), scale_f64(rad_dir, r1));
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p0, origin_f64),
                normal: n_f32,
                color: [col_bot[0], col_bot[1], col_bot[2], rough],
                emissive_mat: [0.0, 0.0, 0.0, MAT_MACRO_PROP],
                water_ice: [0.0; 4],
            });
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p1, origin_f64),
                normal: n_f32,
                color: [col_top[0], col_top[1], col_top[2], rough],
                emissive_mat: [0.0, 0.0, 0.0, MAT_MACRO_PROP],
                water_ice: [0.0; 4],
            });
        }
        for s in 0..segs {
            let i0 = base_idx + (s * 2) as u32;
            indices.extend_from_slice(&[i0, i0 + 2, i0 + 3, i0, i0 + 3, i0 + 1]);
        }
    };

    emit_cone(-0.4, height_m * 0.34, radius_m * 0.24, radius_m * 0.14, 4, [0.18, 0.11, 0.06], [0.24, 0.15, 0.09], 0.90);
    let g0 = [0.038 + 0.018 * tint_var, 0.16 + 0.05 * tint_var, 0.065];
    let g1 = [0.065 + 0.025 * tint_var, 0.27 + 0.07 * tint_var, 0.095];
    let g2 = [0.095 + 0.030 * tint_var, 0.34 + 0.06 * tint_var, 0.115];
    emit_cone(height_m * 0.22, height_m * 0.60, radius_m * 1.05, radius_m * 0.20, 5, g0, g1, 0.88);
    emit_cone(height_m * 0.45, height_m * 0.82, radius_m * 0.82, radius_m * 0.14, 5, g0, g1, 0.88);
    emit_cone(height_m * 0.68, height_m * 1.04, radius_m * 0.54, 0.04, 5, g1, g2, 0.86);
}

#[allow(clippy::too_many_arguments)]
fn push_boulder(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up_dir: [f64; 3],
    origin_f64: [f64; 3],
    rx: f64,
    ry: f64,
    rz: f64,
) {
    let (t1, t2) = tangent_basis_f64(up_dir);
    let segs = 6usize;
    let rings = 3usize;
    let base_idx = verts.len() as u32;
    let stride = segs + 1;
    for ri in 0..=rings {
        let phi = (ri as f64 / rings as f64) * std::f64::consts::PI - std::f64::consts::FRAC_PI_2;
        let cp = phi.cos();
        let sp = phi.sin();
        for si in 0..=segs {
            let th = (si as f64 / segs as f64) * std::f64::consts::TAU;
            let p = add_f64(
                add_f64(base_world, scale_f64(up_dir, (sp * 0.65 + 0.35) * rz)),
                add_f64(scale_f64(t1, cp * th.cos() * rx), scale_f64(t2, cp * th.sin() * ry)),
            );
            let n = normalize3_f64(add_f64(
                scale_f64(up_dir, sp),
                add_f64(scale_f64(t1, cp * th.cos()), scale_f64(t2, cp * th.sin())),
            ));
            let shade = 0.28 + 0.14 * ((sp + 1.0) * 0.5) as f32;
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p, origin_f64),
                normal: [n[0] as f32, n[1] as f32, n[2] as f32],
                color: [shade, shade * 1.02, shade * 1.07, 0.78],
                emissive_mat: [0.0, 0.0, 0.0, MAT_MACRO_PROP],
                water_ice: [0.0; 4],
            });
        }
    }
    for ri in 0..rings {
        for si in 0..segs {
            let i00 = base_idx + (ri * stride + si) as u32;
            let i01 = base_idx + ((ri + 1) * stride + si) as u32;
            indices.extend_from_slice(&[i00, i00 + 1, i01 + 1, i00, i01 + 1, i01]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_sky_island(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    isl: &MacroSkyIsland,
    r_core: f64,
    z_sea: f64,
    origin_f64: [f64; 3],
    waterfall_az: Option<f64>,
    add_sky_pines: bool,
) {
    let cdir = normalize3_f64(isl.center_dir);
    let (t1, t2) = tangent_basis_f64(cdir);
    let r_surf = r_core + z_sea;
    let rings = 9usize;
    let segs = 22usize;

    let dir_at_offset = |dx_m: f64, dy_m: f64| -> [f64; 3] {
        normalize3_f64(add_f64(
            cdir,
            add_f64(scale_f64(t1, dx_m / r_surf), scale_f64(t2, dy_m / r_surf)),
        ))
    };

    let eval_top = |t: f64, ang: f64| -> ([f64; 3], [f32; 3], [f32; 3], f32, f32) {
        let rad = isl.organic_radius_m(ang) * t;
        let d = dir_at_offset(rad * ang.cos(), rad * ang.sin());
        let hill = (1.0 - t * t).powf(1.15) * 3.8 + 0.75 * (ang * 4.0 + isl.phase).sin() * t * (1.0 - t);
        let pos = scale_f64(d, r_core + isl.z_top + hill);
        let col = if t > 0.86 {
            [0.34, 0.35, 0.39]
        } else if t < 0.45 {
            [0.065, 0.24, 0.10]
        } else {
            [0.13, 0.39, 0.16]
        };
        (pos, col, [0.0; 3], 0.80, MAT_MACRO_PROP)
    };

    let eval_bot = |t: f64, ang: f64| -> ([f64; 3], [f32; 3], [f32; 3], f32, f32) {
        let rad = isl.organic_radius_m(ang) * t;
        let d = dir_at_offset(rad * ang.cos(), rad * ang.sin());
        let crag = 1.65 * (ang * 6.0 - isl.phase).sin() * t * (1.0 - t)
            + 0.85 * (ang * 11.0 + isl.phase * 1.7).cos() * t * (1.0 - t);
        let z_f = isl.z_bot + (isl.z_top - isl.z_bot) * t.powf(0.58) + crag;
        let pos = scale_f64(d, r_core + z_f);
        let core_glow = (1.0 - (t / 0.44).clamp(0.0, 1.0)).powf(1.65) as f32;
        let vein = ((ang * 5.0 + isl.phase).sin().abs().powf(8.0) as f32)
            * ((1.0 - t as f32).clamp(0.0, 1.0))
            * 0.65;
        let glow = (core_glow + vein * 0.42).clamp(0.0, 1.0);
        let rock_dark = 0.11 + 0.08 * (t as f32);
        let col = [
            rock_dark + 0.20 * glow,
            rock_dark * 1.06 + 0.66 * glow,
            rock_dark * 1.22 + 0.82 * glow,
        ];
        let em = [0.32 * glow, 1.85 * glow, 3.35 * glow];
        let mk = if glow > 0.35 { MAT_MACRO_CRYSTAL } else { MAT_MACRO_PROP };
        (pos, col, em, 0.76 - 0.50 * glow, mk)
    };

    for is_top in [true, false] {
        let base_idx = verts.len() as u32;
        let stride_s = segs + 1;
        for ri in 0..=rings {
            let t = ri as f64 / rings as f64;
            for si in 0..=segs {
                let ang = (si as f64 / segs as f64) * std::f64::consts::TAU;
                let (pos, col, em, rough, mk) = if is_top { eval_top(t, ang) } else { eval_bot(t, ang) };
                let dt = 0.03;
                let da = 0.06;
                let p_ta = if is_top { eval_top((t + dt).min(1.0), ang).0 } else { eval_bot((t + dt).min(1.0), ang).0 };
                let p_tb = if is_top { eval_top((t - dt).max(0.0), ang).0 } else { eval_bot((t - dt).max(0.0), ang).0 };
                let p_aa = if is_top { eval_top(t.max(0.04), ang + da).0 } else { eval_bot(t.max(0.04), ang + da).0 };
                let p_ab = if is_top { eval_top(t.max(0.04), ang - da).0 } else { eval_bot(t.max(0.04), ang - da).0 };
                let rad_dir = normalize3_f64(pos);
                let mut norm = normalize3_f64(cross_f64(sub_f64(p_ta, p_tb), sub_f64(p_aa, p_ab)));
                let want_sign = if is_top { 1.0 } else { -1.0 };
                if dot3_f64(norm, rad_dir) * want_sign < 0.0 {
                    norm = scale_f64(norm, -1.0);
                }
                verts.push(MacroPlanetVertex {
                    pos: world_f64_to_camera_rel_f32(pos, origin_f64),
                    normal: [norm[0] as f32, norm[1] as f32, norm[2] as f32],
                    color: [col[0], col[1], col[2], rough],
                    emissive_mat: [em[0], em[1], em[2], mk],
                    water_ice: [0.0; 4],
                });
            }
        }
        for ri in 0..rings {
            for si in 0..segs {
                let i00 = base_idx + (ri * stride_s + si) as u32;
                let i10 = base_idx + ((ri + 1) * stride_s + si) as u32;
                let i01 = i00 + 1;
                let i11 = i10 + 1;
                if is_top {
                    indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                } else {
                    indices.extend_from_slice(&[i00, i11, i10, i00, i01, i11]);
                }
            }
        }
    }

    if add_sky_pines {
        for pi in 0..10usize {
            let f_i = pi as f64;
            let ang = f_i * 2.39996 + isl.phase;
            let t_r = ((f_i + 0.5) / 10.0).sqrt() * 0.70;
            let (p_top, _, _, _, _) = eval_top(t_r, ang);
            let up_d = normalize3_f64(p_top);
            push_pine_tree(
                verts,
                indices,
                p_top,
                up_d,
                origin_f64,
                5.2 + (pi % 4) as f64 * 1.2,
                1.7 + (pi % 3) as f64 * 0.4,
                (pi % 5) as f32 * 0.2,
            );
        }
    }

    if let Some(wf_ang) = waterfall_az {
        let rim_r = isl.organic_radius_m(wf_ang) * 0.96;
        let out_dir = normalize3_f64(add_f64(scale_f64(t1, wf_ang.cos()), scale_f64(t2, wf_ang.sin())));
        let side_dir = normalize3_f64(cross_f64(cdir, out_dir));
        let z_start = isl.z_top + 0.2;
        let z_end = z_sea + 0.25;
        let wf_segs = 14usize;
        let wf_cols = 3usize;
        let base_idx = verts.len() as u32;
        let stride = wf_cols + 1;

        for ri in 0..=wf_segs {
            let s = ri as f64 / wf_segs as f64;
            let z_cur = z_start + (z_end - z_start) * (s * s * 0.65 + s * 0.35);
            let arc_out = rim_r + 3.8 * s.sqrt();
            let half_w = 1.15 + 1.15 * s;
            for ci in 0..=wf_cols {
                let u_w = (ci as f64 / wf_cols as f64) * 2.0 - 1.0;
                let bow = (1.0 - u_w * u_w) * 0.65;
                let d_cur = normalize3_f64([
                    cdir[0] + (out_dir[0] * (arc_out + bow) + side_dir[0] * (u_w * half_w)) / r_surf,
                    cdir[1] + (out_dir[1] * (arc_out + bow) + side_dir[1] * (u_w * half_w)) / r_surf,
                    cdir[2] + (out_dir[2] * (arc_out + bow) + side_dir[2] * (u_w * half_w)) / r_surf,
                ]);
                let pos = scale_f64(d_cur, r_core + z_cur);
                let streak = 0.5 + 0.5 * ((u_w * 7.0 + s * 14.0).sin() as f32);
                let foam = (0.35 + 0.35 * s as f32 + 0.25 * streak).clamp(0.0, 1.0);
                verts.push(MacroPlanetVertex {
                    pos: world_f64_to_camera_rel_f32(pos, origin_f64),
                    normal: [out_dir[0] as f32, out_dir[1] as f32, out_dir[2] as f32],
                    color: [0.22 + 0.55 * foam, 0.62 + 0.32 * foam, 0.86 + 0.14 * foam, 0.14],
                    emissive_mat: [0.06 * foam, 0.20 * foam, 0.32 * foam, MAT_MACRO_PROP],
                    water_ice: [0.0; 4],
                });
            }
        }
        for ri in 0..wf_segs {
            for ci in 0..wf_cols {
                let i00 = base_idx + (ri * stride + ci) as u32;
                let i10 = i00 + 1;
                let i01 = base_idx + ((ri + 1) * stride + ci) as u32;
                let i11 = i01 + 1;
                indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01, i00, i11, i10, i00, i01, i11]);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_oriented_box_f64(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    center: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    fwd: [f64; 3],
    size: [f64; 3],
    origin_f64: [f64; 3],
    color: [f32; 4],
    emissive_mat: [f32; 4],
) {
    let hx = scale_f64(right, size[0] * 0.5);
    let hy = scale_f64(up, size[1] * 0.5);
    let hz = scale_f64(fwd, size[2] * 0.5);
    let c = |sx: f64, sy: f64, sz: f64| {
        add_f64(
            add_f64(add_f64(center, scale_f64(hx, sx)), scale_f64(hy, sy)),
            scale_f64(hz, sz),
        )
    };
    let faces: [([f64; 3], [[f64; 3]; 4]); 6] = [
        (up, [c(-1.0, 1.0, -1.0), c(-1.0, 1.0, 1.0), c(1.0, 1.0, 1.0), c(1.0, 1.0, -1.0)]),
        (scale_f64(up, -1.0), [c(-1.0, -1.0, 1.0), c(-1.0, -1.0, -1.0), c(1.0, -1.0, -1.0), c(1.0, -1.0, 1.0)]),
        (right, [c(1.0, -1.0, 1.0), c(1.0, -1.0, -1.0), c(1.0, 1.0, -1.0), c(1.0, 1.0, 1.0)]),
        (scale_f64(right, -1.0), [c(-1.0, -1.0, -1.0), c(-1.0, -1.0, 1.0), c(-1.0, 1.0, 1.0), c(-1.0, 1.0, -1.0)]),
        (fwd, [c(-1.0, -1.0, 1.0), c(1.0, -1.0, 1.0), c(1.0, 1.0, 1.0), c(-1.0, 1.0, 1.0)]),
        (scale_f64(fwd, -1.0), [c(1.0, -1.0, -1.0), c(-1.0, -1.0, -1.0), c(-1.0, 1.0, -1.0), c(1.0, 1.0, -1.0)]),
    ];
    for (n, corners) in faces {
        let base = verts.len() as u32;
        let mut norm = normalize3_f64(cross_f64(sub_f64(corners[1], corners[0]), sub_f64(corners[2], corners[0])));
        let order = if dot3_f64(norm, n) < 0.0 {
            norm = scale_f64(norm, -1.0);
            [corners[0], corners[3], corners[2], corners[1]]
        } else {
            corners
        };
        let n_f = [norm[0] as f32, norm[1] as f32, norm[2] as f32];
        for p in order {
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p, origin_f64),
                normal: n_f,
                color,
                emissive_mat,
                water_ice: [0.0; 4],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[allow(clippy::too_many_arguments)]
fn push_oriented_pyramid_f64(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    base_center: [f64; 3],
    right: [f64; 3],
    up: [f64; 3],
    fwd: [f64; 3],
    half_w: f64,
    height: f64,
    origin_f64: [f64; 3],
    color: [f32; 4],
    emissive_mat: [f32; 4],
) {
    let apex = add_f64(base_center, scale_f64(up, height));
    let c0 = sub_f64(sub_f64(base_center, scale_f64(right, half_w)), scale_f64(fwd, half_w));
    let c1 = sub_f64(add_f64(base_center, scale_f64(right, half_w)), scale_f64(fwd, half_w));
    let c2 = add_f64(add_f64(base_center, scale_f64(right, half_w)), scale_f64(fwd, half_w));
    let c3 = add_f64(sub_f64(base_center, scale_f64(right, half_w)), scale_f64(fwd, half_w));
    for (a, b, c) in [(c0, c1, apex), (c1, c2, apex), (c2, c3, apex), (c3, c0, apex)] {
        let mid = scale_f64(add_f64(add_f64(a, b), c), 1.0 / 3.0);
        let out_hint = normalize3_f64(sub_f64(mid, base_center));
        let mut n = normalize3_f64(cross_f64(sub_f64(b, a), sub_f64(c, a)));
        let base = verts.len() as u32;
        let pts = if dot3_f64(n, out_hint) < 0.0 {
            n = scale_f64(n, -1.0);
            [a, c, b]
        } else {
            [a, b, c]
        };
        let n_f = [n[0] as f32, n[1] as f32, n[2] as f32];
        for p in pts {
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p, origin_f64),
                normal: n_f,
                color,
                emissive_mat,
                water_ice: [0.0; 4],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

#[allow(clippy::too_many_arguments)]
fn push_spherical_ring_f64(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    center: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    fwd: [f64; 3],
    r_inner: f64,
    r_outer: f64,
    segs: usize,
    origin_f64: [f64; 3],
    color: [f32; 4],
    emissive_mat: [f32; 4],
) {
    let base = verts.len() as u32;
    let r_rad = dot3_f64(center, center).sqrt();
    let n_f = [up[0] as f32, up[1] as f32, up[2] as f32];
    for i in 0..=segs {
        let ang = (i as f64 / segs as f64) * std::f64::consts::TAU;
        let dir_t = add_f64(scale_f64(right, ang.cos()), scale_f64(fwd, ang.sin()));
        let p_in = scale_f64(normalize3_f64(add_f64(center, scale_f64(dir_t, r_inner))), r_rad);
        let p_out = scale_f64(normalize3_f64(add_f64(center, scale_f64(dir_t, r_outer))), r_rad);
        verts.push(MacroPlanetVertex {
            pos: world_f64_to_camera_rel_f32(p_in, origin_f64),
            normal: n_f,
            color,
            emissive_mat,
            water_ice: [0.0; 4],
        });
        verts.push(MacroPlanetVertex {
            pos: world_f64_to_camera_rel_f32(p_out, origin_f64),
            normal: n_f,
            color,
            emissive_mat,
            water_ice: [0.0; 4],
        });
    }
    for i in 0..segs {
        let a = base + (i * 2) as u32;
        indices.extend_from_slice(&[a, a + 1, a + 3, a, a + 3, a + 2]);
    }
}

#[allow(clippy::too_many_arguments)]
fn push_radial_pillar_f64(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    base_pos: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    fwd: [f64; 3],
    radius: f64,
    height: f64,
    origin_f64: [f64; 3],
    color: [f32; 4],
    emissive_mat: [f32; 4],
) {
    let top_pos = add_f64(base_pos, scale_f64(up, height));
    let top_col = [color[0], color[1], color[2], color[3] * 0.08];
    for axis in [right, fwd] {
        let base = verts.len() as u32;
        let n = normalize3_f64(cross_f64(up, axis));
        let n_f = [n[0] as f32, n[1] as f32, n[2] as f32];
        let pts = [
            (sub_f64(base_pos, scale_f64(axis, radius)), color),
            (add_f64(base_pos, scale_f64(axis, radius)), color),
            (add_f64(top_pos, scale_f64(axis, radius * 0.35)), top_col),
            (sub_f64(top_pos, scale_f64(axis, radius * 0.35)), top_col),
        ];
        for (p, c) in pts {
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p, origin_f64),
                normal: n_f,
                color: c,
                emissive_mat,
                water_ice: [0.0; 4],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

// =============================================================================
// Wave 34B Unit & Headless Metal Verification Test
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::MacroPlanetCamera;
    use crate::net::LiveWorldState;
    use frostmarsh_proto::{compress_aplus_cells, encode_subbrick3d_lz4};
    use glam::DVec3;

    #[test]
    fn test_wave34b_macro_planet_clipmap_and_floating_origin() {
        let mut net = LiveWorldState::new(2048, 2048, 64);

        // 1. Ingest all 6 `PlanetMacroClimateFace` (64×64 RGBA8) payloads via `LiveWorldState`
        for face in 0..FACE_COUNT as u8 {
            let mut clim = vec![0u8; MACRO_CLIMATE_RES * MACRO_CLIMATE_RES * 4];
            for i in 0..(MACRO_CLIMATE_RES * MACRO_CLIMATE_RES) {
                clim[i * 4] = 160; // temp_norm
                clim[i * 4 + 1] = ((i * 17 + face as usize * 31) % 240) as u8; // Coriolis storm cloud
                clim[i * 4 + 2] = if face == 0 && i % 13 == 0 { 190 } else { 10 }; // ash plume
                clim[i * 4 + 3] = if face == 2 || face == 3 { 230 } else { 20 }; // snowline
            }
            let lz4 = compress_aplus_cells(&clim);
            net.apply_macro_climate_face(lz4.len() as u64, face, MACRO_CLIMATE_RES as u16, 10, &lz4);
        }
        assert!(net.macro_planet_mode);
        assert_eq!(net.received_macro_climate_face_count(), 6);

        // 2. Ingest `PlanetShardBrick3D` sub-bricks for hydrated shard (face=4, sx=3, sy=3)
        let sub_d = 16usize;
        let sub_bytes = CHUNK_SIZE * CHUNK_SIZE * sub_d * 4;
        for by in 0..2u16 {
            for bx in 0..2u16 {
                for bz in [1u16, 3u16] {
                    let mut dm = vec![0u8; sub_bytes];
                    let mut at = vec![0u8; sub_bytes];
                    for dz in 0..sub_d {
                        let gz = (bz as usize) * 16 + dz;
                        for dy in 0..CHUNK_SIZE {
                            for dx in 0..CHUNK_SIZE {
                                let o = ((dz * CHUNK_SIZE + dy) * CHUNK_SIZE + dx) * 4;
                                if gz <= 26 {
                                    dm[o] = 240;
                                    dm[o + 3] = 210;
                                    at[o] = 36;
                                    at[o + 1] = 115;
                                    at[o + 2] = 48;
                                } else if gz == 27 && bx == 1 {
                                    dm[o + 1] = 220;
                                    dm[o + 3] = 20;
                                    at[o] = 24;
                                    at[o + 1] = 110;
                                    at[o + 2] = 175;
                                } else if gz == 50 && dx >= 5 && dx <= 10 && dy >= 5 && dy <= 10 {
                                    // Live Buoyant Crystal Core Sky-Island voxel
                                    dm[o] = 235;
                                    dm[o + 3] = 45;
                                    at[o] = 75;
                                    at[o + 1] = 225;
                                    at[o + 2] = 255;
                                }
                            }
                        }
                    }
                    let (ldm, lat) = encode_subbrick3d_lz4(&dm, &at);
                    net.apply_macro_shard_brick3d(
                        (ldm.len() + lat.len()) as u64,
                        4,
                        3,
                        3,
                        bx,
                        by,
                        bz,
                        64,
                        sub_d as u8,
                        11,
                        &ldm,
                        &lat,
                    );
                }
            }
        }
        assert_eq!(net.received_macro_shard_brick_count(), 8);

        // 3. Apply into `MacroPlanetWorld` and verify floating-origin `f64 -> f32` precision (< 1e-4 m)
        let mut world = MacroPlanetWorld::new(1280.0, 8, 64, 33_033);
        let clim_applied = world.apply_climate_updates(net.take_macro_climate_updates());
        let shard_applied = world.apply_shard_updates(net.take_macro_shard_brick_updates());
        assert_eq!(clim_applied, 6);
        assert_eq!(shard_applied, 8);
        assert!(world.has_live_climate());
        assert_eq!(world.hydrated_shard_count(), 1);

        let cam_world = virtual_grid_to_world_f64(4, 3.5 * 256.0, 3.5 * 256.0, 28.0, 2048, 1280.0);
        assert!(world.ensure_floating_origin(cam_world));

        // Floating-origin precision check at R = 1,304 m: must be < 1e-4 m!
        let test_offset = [14.3752, -9.6214, 5.1843];
        let pt_f64 = add_f64(world.origin_f64, test_offset);
        let rel_f32 = world_f64_to_camera_rel_f32(pt_f64, world.origin_f64);
        let err_m = ((rel_f32[0] as f64 - test_offset[0]).powi(2)
            + (rel_f32[1] as f64 - test_offset[1]).powi(2)
            + (rel_f32[2] as f64 - test_offset[2]).powi(2))
        .sqrt();
        assert!(
            err_m < 1e-4,
            "floating-origin f64->f32 precision must be < 1e-4 m, got {} m",
            err_m
        );

        // 4. Rebuild dual-scale curved meshes and verify non-empty finite geometry on BOTH scales
        let rebuilt = world.rebuild_dirty(4);
        assert_eq!(rebuilt, 1);
        assert!(!world.macro_terrain_verts.is_empty());
        assert!(!world.macro_terrain_indices.is_empty());
        assert!(!world.macro_ocean_verts.is_empty());
        assert!(!world.macro_ocean_indices.is_empty());
        assert!(!world.macro_sky_verts.is_empty());
        assert!(!world.near_terrain_verts.is_empty());
        assert!(!world.near_terrain_indices.is_empty());
        assert!(!world.minimap_rgba.is_empty());

        for v in world.near_terrain_verts.iter().take(2048) {
            assert!(v.pos.iter().all(|c| c.is_finite()));
            assert!(v.normal.iter().all(|c| c.is_finite()));
        }
        for v in world.macro_terrain_verts.iter().take(2048) {
            assert!(v.pos.iter().all(|c| c.is_finite()));
            assert!(v.normal.iter().all(|c| c.is_finite()));
        }

        // 5. Headless Metal rendering verification of `macro_planet_live.wgsl` (Ground + Orbital!)
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let meshes = crate::mesh_gen::StaticMeshBank::build(1);
        let surfels = crate::surfel_gen::SurfelProtoBank::build();
        if let Ok(mut renderer) =
            rt.block_on(crate::renderer::Approach4Renderer::new_headless(&meshes, &surfels))
        {
            renderer.upload_macro_planet_world(&mut world);
            let sun = DVec3::new(-0.72, 0.36, 0.59).normalize();

            // Orbital Globe render check (verifies macro globe + live Coriolis climate array texture + atmosphere!)
            let orb_cam = MacroPlanetCamera::orbital_globe_shot(sun, world.r_surf());
            let orb_mats = orb_cam.compute_matrices(320.0, 180.0);
            let (rgba_orb, _) = renderer.render_macro_planet_to_rgba(
                320,
                180,
                &orb_mats,
                [sun.x as f32, sun.y as f32, sun.z as f32],
                orb_cam.exposure,
                1.2,
                [0.0, 0.0, 0.5],
                &world,
                None,
            );
            assert_eq!(rgba_orb.len(), 320 * 180 * 4);
            let orb_lit = rgba_orb
                .chunks_exact(4)
                .filter(|px| px[0] > 20 || px[1] > 20 || px[2] > 20)
                .count();
            assert!(
                orb_lit > 1000,
                "macro planet orbital render must have visible illuminated pixels, got {}",
                orb_lit
            );

            // Ground 3rd-Person locally-flat render check
            let (fpos, ffwd, isl_dir) = world.find_hero_or_best_cluster_f64(None, sun.to_array());
            let g_cam = MacroPlanetCamera::ground_3rd_person_shot(
                DVec3::from_array(fpos),
                DVec3::from_array(ffwd),
                DVec3::from_array(isl_dir),
            );
            let g_mats = g_cam.compute_matrices(320.0, 180.0);
            let (rgba_g, _) = renderer.render_macro_planet_to_rgba(
                320,
                180,
                &g_mats,
                [sun.x as f32, sun.y as f32, sun.z as f32],
                g_cam.exposure,
                1.2,
                [1.0, -0.5, 0.35],
                &world,
                None,
            );
            assert_eq!(rgba_g.len(), 320 * 180 * 4);
            let g_lit = rgba_g
                .chunks_exact(4)
                .filter(|px| px[0] > 20 || px[1] > 20 || px[2] > 20)
                .count();
            assert!(
                g_lit > 2000,
                "macro planet ground render must have visible sky & 1m terrain pixels, got {}",
                g_lit
            );
        }
    }
}
