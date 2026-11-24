//! Wave 31B: Live Curved 6-Face Cubed-Sphere Planetary World & Dynamic 3D Spherical Entity Mesher.
//!
//! Stores all 6 cubed-sphere faces (`+X, -X, +Y, -Y, +Z, -Z`) of live RGBA8 `density_mat` +
//! `albedo_thermal` voxels streamed over QUIC (`PlanetFaceBrick3D` / `PlanetFaceBrick3DDelta`),
//! incrementally rebuilds seamless curved 3D spherical meshes (terrain shell, Fresnel sapphire
//! ocean, Orbital Floating Sky-Ring Islands with glowing Buoyant Crystal Cores, and translucent
//! volumetric steam/smoke plumes), maintains a live equirectangular planetary minimap, and
//! builds every-frame live 3D curved spherical meshes for all 60 neural bots/heroes, 6 planetary
//! shrine spires + light pillars, and glowing elemental actuator beams & projectiles!

use crate::net::{LiveEntityState, PlanetBrickUpdate};
use bytemuck::{Pod, Zeroable};
use frostmarsh_proto::{
    CHUNK_SIZE, CLASS_BURROWER, CLASS_EMBERSTALKER, CLASS_SKYWING, CLASS_STRIDER, CLASS_TIDEKIN,
};
use glam::Vec3;
use std::collections::HashSet;

pub const FACE_COUNT: usize = 6;
pub const PLANET_MINIMAP_W: usize = 256;
pub const PLANET_MINIMAP_H: usize = 128;

// =============================================================================
// Exact 6-Face Cubed-Sphere Math (1:1 parity with `field_core::planet`)
// =============================================================================

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

/// Project cubed-sphere face coordinates `(face, u, v)` (`u, v ∈ [-1, 1]`) onto the unit sphere.
#[inline]
pub fn face_uv_to_sphere_dir(face: usize, u: f32, v: f32) -> [f32; 3] {
    let fr = FACE_FRAMES[face % FACE_COUNT];
    let px = fr.n[0] as f32 + u * fr.e_u[0] as f32 + v * fr.e_v[0] as f32;
    let py = fr.n[1] as f32 + u * fr.e_u[1] as f32 + v * fr.e_v[1] as f32;
    let pz = fr.n[2] as f32 + u * fr.e_u[2] as f32 + v * fr.e_v[2] as f32;
    let inv_len = 1.0 / (px * px + py * py + pz * pz).sqrt().max(1e-12);
    [px * inv_len, py * inv_len, pz * inv_len]
}

/// Exact inverse projection from any non-zero 3D direction `dir` to `(face, u, v)` (`u, v ∈ [-1, 1]`).
#[inline]
pub fn sphere_dir_to_face_uv(dir: [f32; 3]) -> (usize, f32, f32) {
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();

    let (face, denom) = if ax >= ay && ax >= az {
        if dir[0] >= 0.0 { (0, dir[0]) } else { (1, -dir[0]) }
    } else if ay >= ax && ay >= az {
        if dir[1] >= 0.0 { (2, dir[1]) } else { (3, -dir[1]) }
    } else if dir[2] >= 0.0 {
        (4, dir[2])
    } else {
        (5, -dir[2])
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

/// Continuous curved spherical world position `(r_core + z_f) * face_uv_to_sphere_dir(face, u, v)`.
#[inline]
pub fn face_uv_z_to_world_pos(face: usize, u: f32, v: f32, z_f: f32, r_core: f32) -> [f32; 3] {
    let dir = face_uv_to_sphere_dir(face, u, v);
    let r = r_core + z_f;
    [r * dir[0], r * dir[1], r * dir[2]]
}

/// Map face-local continuous grid coordinates `(face, gx, gy, gz)` to 3D curved spherical position.
#[inline]
pub fn face_grid_to_world_pos(
    face: u8,
    gx: f32,
    gy: f32,
    gz: f32,
    face_size: usize,
    r_core: f32,
) -> Vec3 {
    let nf = face_size.max(1) as f32;
    let u = (gx / nf) * 2.0 - 1.0;
    let v = (gy / nf) * 2.0 - 1.0;
    Vec3::from_array(face_uv_z_to_world_pos(face as usize, u, v, gz, r_core))
}

/// Orthonormal local spherical tangent frame `(up, tang_u, tang_v)` at `(face, u, v)`
/// such that `up = normalize(sphere_pos)`, `tang_u` points along increasing `u`, and
/// `tang_v = up.cross(tang_u)` (right-handed).
#[inline]
pub fn face_tangent_basis(face: usize, u: f32, v: f32) -> (Vec3, Vec3, Vec3) {
    let up = Vec3::from_array(face_uv_to_sphere_dir(face, u, v)).normalize_or(Vec3::Y);
    let fr = FACE_FRAMES[face % FACE_COUNT];
    let eu_raw = Vec3::new(fr.e_u[0] as f32, fr.e_u[1] as f32, fr.e_u[2] as f32);
    let ev_raw = Vec3::new(fr.e_v[0] as f32, fr.e_v[1] as f32, fr.e_v[2] as f32);
    let tang_u = (eu_raw - up * eu_raw.dot(up)).normalize_or(Vec3::X);
    let mut tang_v = (ev_raw - up * ev_raw.dot(up)).normalize_or_zero();
    if tang_v.length_squared() < 0.25 {
        tang_v = up.cross(tang_u).normalize_or(Vec3::Z);
    }
    (up, tang_u, tang_v)
}

// =============================================================================
// Planet Vertex Layout (64 bytes, tightly packed for Apple M4 Pro Metal)
// =============================================================================

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PlanetVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// `rgb = base albedo / tint`, `a = roughness` (solid) or `alpha` (translucent)
    pub color: [f32; 4],
    /// `x = water_depth`, `y = emissive_strength`, `z = mat_kind`, `w = ice_frac`
    pub flags: [f32; 4],
}

impl PlanetVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PlanetVertex>() as wgpu::BufferAddress,
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
            ],
        }
    }
}

// Material kind IDs matching `planet_live.wgsl`
pub const MAT_TERRAIN: f32 = 0.0;
pub const MAT_OCEAN: f32 = 1.0;
pub const MAT_CRYSTAL_CORE: f32 = 2.0;
pub const MAT_ENTITY: f32 = 3.0;
pub const MAT_BEAM: f32 = 4.0;
pub const MAT_GAS: f32 = 5.0;

// =============================================================================
// Per-Face Live Voxel Volume & Cached Curved Spherical Mesh
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct ColumnSurface {
    surf_z: f32,
    water_depth: f32,
    ice_frac: f32,
    color: [f32; 3],
    emissive: f32,
    roughness: f32,
}

impl Default for ColumnSurface {
    fn default() -> Self {
        Self {
            surf_z: 14.0,
            water_depth: 0.0,
            ice_frac: 0.0,
            color: [0.18, 0.28, 0.22],
            emissive: 0.0,
            roughness: 0.80,
        }
    }
}

pub struct PlanetFaceData {
    #[allow(dead_code)]
    pub face: usize,
    pub face_size: usize,
    pub depth: usize,
    /// Z-major `[z][y][x][4]`: `[solid_density, fluid_density, gas_density, roughness]`
    pub density_mat: Vec<u8>,
    /// Z-major `[z][y][x][4]`: `[albedo_r, albedo_g, albedo_b, thermal_glow]`
    pub albedo_thermal: Vec<u8>,
    column_cache: Vec<ColumnSurface>,
    pub dirty: bool,
    pub column_dirty: bool,
    pub has_data: bool,

    pub terrain_verts: Vec<PlanetVertex>,
    pub terrain_indices: Vec<u32>,
    pub ocean_verts: Vec<PlanetVertex>,
    pub ocean_indices: Vec<u32>,
    pub gas_verts: Vec<PlanetVertex>,
    pub gas_indices: Vec<u32>,
}

impl PlanetFaceData {
    pub fn new(face: usize, face_size: usize, depth: usize) -> Self {
        let n = face_size.max(1);
        let d = depth.max(1);
        let bytes = n * n * d * 4;
        Self {
            face,
            face_size: n,
            depth: d,
            density_mat: vec![0u8; bytes],
            albedo_thermal: vec![0u8; bytes],
            column_cache: vec![ColumnSurface::default(); n * n],
            dirty: false,
            column_dirty: false,
            has_data: false,
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
        ((z * self.face_size + y) * self.face_size + x) * 4
    }

    #[inline]
    fn solid_at(&self, x: isize, y: isize, z: isize) -> u8 {
        if x < 0
            || y < 0
            || z < 0
            || x >= self.face_size as isize
            || y >= self.face_size as isize
            || z >= self.depth as isize
        {
            return 0;
        }
        let off = self.voxel_offset(x as usize, y as usize, z as usize);
        self.density_mat[off]
    }

    fn recompute_columns(&mut self, z_sea: usize, ring_z_floor: usize) {
        let n = self.face_size;
        let z_sea_f = z_sea as f32 + 0.55;
        let max_z = ring_z_floor.min(self.depth);

        for y in 0..n {
            for x in 0..n {
                let mut found_solid = false;
                let mut solid_z = 0usize;
                let mut solid_fill = 0.0f32;
                let mut best_col = [0.20f32, 0.28, 0.24];
                let mut best_rough = 0.80f32;
                let mut max_thermal = 0.0f32;
                let mut fluid_cols = 0.0f32;
                let mut fluid_top_z = 0.0f32;
                let mut sea_ice_hits = 0.0f32;
                let mut sea_water_hits = 0.0f32;

                for z in (0..max_z).rev() {
                    let off = self.voxel_offset(x, y, z);
                    let solid = self.density_mat[off];
                    let fluid = self.density_mat[off + 1];
                    let rough = self.density_mat[off + 3] as f32 / 255.0;
                    let ar = self.albedo_thermal[off] as f32 / 255.0;
                    let ag = self.albedo_thermal[off + 1] as f32 / 255.0;
                    let ab = self.albedo_thermal[off + 2] as f32 / 255.0;
                    let therm = self.albedo_thermal[off + 3] as f32 / 255.0;

                    if therm > max_thermal && (solid >= 40 || fluid >= 40 || self.density_mat[off + 2] >= 40) {
                        max_thermal = therm;
                    }

                    // Water detection: fluid_density > 60 && thermal < 0.35
                    if fluid > 60 && therm < 0.35 {
                        fluid_cols += fluid as f32 / 255.0;
                        if fluid_top_z <= 0.0 {
                            fluid_top_z = z as f32 + (fluid as f32 / 255.0).clamp(0.25, 1.0);
                        }
                        if z == z_sea || z + 1 == z_sea {
                            sea_water_hits += fluid as f32 / 255.0;
                        }
                    }

                    // Sea-ice detection near sea level (cool high-brightness low-roughness solid)
                    if (z == z_sea || z + 1 == z_sea) && solid >= 100 && ab > 0.72 && ar > 0.56 && rough < 0.38 {
                        sea_ice_hits += solid as f32 / 255.0;
                    }

                    if solid >= 96 && !found_solid {
                        found_solid = true;
                        solid_z = z;
                        solid_fill = (solid as f32 / 255.0).clamp(0.18, 1.0);
                        best_col = [ar, ag, ab];
                        best_rough = rough;
                        break;
                    }
                }

                // Also peek 1 layer above solid for hot glowing oil/lava pool on top of rock
                if found_solid && solid_z + 1 < max_z {
                    let off_u = self.voxel_offset(x, y, solid_z + 1);
                    let therm_u = self.albedo_thermal[off_u + 3] as f32 / 255.0;
                    if therm_u > max_thermal {
                        max_thermal = therm_u;
                    }
                }

                let surf_z = if found_solid {
                    solid_z as f32 + solid_fill
                } else {
                    (z_sea.saturating_sub(4)) as f32
                };

                let water_depth = if fluid_cols > 0.15 {
                    (fluid_top_z.max(z_sea_f) - surf_z).max(fluid_cols * 0.65).max(0.0)
                } else {
                    (z_sea_f - surf_z).max(0.0) * 0.0
                };

                let ice_frac = if water_depth > 0.05 && (sea_ice_hits + sea_water_hits) > 0.01 {
                    (sea_ice_hits / (sea_ice_hits + sea_water_hits)).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                // Vivid planetary albedo & blazing night-side volcanic lava emissive
                let mut color = best_col;
                let mut emissive = 0.0f32;
                let mut roughness = best_rough.clamp(0.14, 0.94);

                if max_thermal > 0.45 {
                    // Blazing incandescent volcanic rift magma / active fire!
                    let hot = ((max_thermal - 0.35) / 0.65).clamp(0.20, 1.0);
                    color = [0.22, 0.06, 0.03];
                    emissive = 1.4 + 3.2 * hot;
                    roughness = 0.50;
                } else if max_thermal > 0.16 {
                    // Scorched caldera rim with warm ember veins
                    let ember = ((max_thermal - 0.16) / 0.30).clamp(0.0, 1.0);
                    color = [
                        color[0] * 0.45 + 0.08,
                        color[1] * 0.40 + 0.04,
                        color[2] * 0.45 + 0.05,
                    ];
                    emissive = 0.55 * ember;
                } else if color[2] > 0.72 && color[0] > 0.56 && roughness < 0.36 {
                    // Polar glacier ice cap / alpine snow peak
                    color = [
                        (color[0] * 1.04).min(0.96),
                        (color[1] * 1.04).min(0.98),
                        (color[2] * 1.03).min(1.0),
                    ];
                    roughness = 0.24;
                } else if color[1] > color[0] * 1.10 && color[1] > color[2] * 1.08 {
                    // Lush emerald temperate continents & alpine forest canopy
                    let alt_t = ((surf_z - z_sea_f) / 4.5).clamp(0.0, 1.0);
                    if surf_z < z_sea_f + 0.50 && surf_z >= z_sea_f - 0.35 {
                        color = [0.54, 0.47, 0.33]; // Golden sandy coast
                    } else {
                        color = [
                            (color[0] * 0.72 + 0.08 * alt_t).clamp(0.04, 0.45),
                            (color[1] * 1.08).clamp(0.14, 0.52),
                            (color[2] * 0.78).clamp(0.06, 0.32),
                        ];
                    }
                } else if surf_z <= z_sea_f && water_depth > 0.05 {
                    // Submerged sandy seafloor
                    color = [0.16, 0.25, 0.28];
                    roughness = 0.86;
                }

                self.column_cache[y * n + x] = ColumnSurface {
                    surf_z,
                    water_depth,
                    ice_frac,
                    color,
                    emissive,
                    roughness,
                };
            }
        }
        self.column_dirty = false;
    }
}

// =============================================================================
// Live 6-Face Planet World Manager
// =============================================================================

pub struct PlanetWorld {
    pub face_size: usize,
    pub depth: usize,
    pub r_core: f32,
    pub faces: [PlanetFaceData; FACE_COUNT],
    received_keys: HashSet<(u8, u16, u16, u16)>,
    pub total_updates_applied: u64,

    // Combined static buffers uploaded to Metal when `gpu_dirty == true`
    pub combined_terrain_verts: Vec<PlanetVertex>,
    pub combined_terrain_indices: Vec<u32>,
    pub combined_ocean_verts: Vec<PlanetVertex>,
    pub combined_ocean_indices: Vec<u32>,
    pub combined_gas_verts: Vec<PlanetVertex>,
    pub combined_gas_indices: Vec<u32>,
    pub gpu_dirty: bool,

    // Dynamic per-frame 3D curved entity / shrine / combat buffers
    pub dynamic_solid_verts: Vec<PlanetVertex>,
    pub dynamic_solid_indices: Vec<u32>,
    pub dynamic_fx_verts: Vec<PlanetVertex>,
    pub dynamic_fx_indices: Vec<u32>,

    // Live equirectangular planet minimap (256x128 RGBA8)
    pub minimap_rgba: Vec<u8>,
    pub minimap_dirty: bool,
}

impl PlanetWorld {
    pub fn new(face_size: usize, depth: usize, r_core: f32) -> Self {
        let n = face_size.max(16);
        let d = depth.max(16);
        let r = if r_core > 1.0 { r_core } else { 48.0 };
        let faces: [PlanetFaceData; FACE_COUNT] =
            std::array::from_fn(|f| PlanetFaceData::new(f, n, d));
        Self {
            face_size: n,
            depth: d,
            r_core: r,
            faces,
            received_keys: HashSet::new(),
            total_updates_applied: 0,
            combined_terrain_verts: Vec::new(),
            combined_terrain_indices: Vec::new(),
            combined_ocean_verts: Vec::new(),
            combined_ocean_indices: Vec::new(),
            combined_gas_verts: Vec::new(),
            combined_gas_indices: Vec::new(),
            gpu_dirty: false,
            dynamic_solid_verts: Vec::new(),
            dynamic_solid_indices: Vec::new(),
            dynamic_fx_verts: Vec::new(),
            dynamic_fx_indices: Vec::new(),
            minimap_rgba: vec![18u8; PLANET_MINIMAP_W * PLANET_MINIMAP_H * 4],
            minimap_dirty: true,
        }
    }

    pub fn resize_if_needed(&mut self, face_size: usize, depth: usize, r_core: f32) {
        let n = face_size.max(16);
        let d = depth.max(16);
        let r = if r_core > 1.0 { r_core } else { self.r_core };
        if self.face_size != n || self.depth != d {
            *self = Self::new(n, d, r);
        } else if (self.r_core - r).abs() > 0.01 {
            self.r_core = r;
            for f in &mut self.faces {
                if f.has_data {
                    f.dirty = true;
                }
            }
        }
    }

    #[inline]
    pub fn z_sea(&self) -> usize {
        ((15.0 * (self.depth as f32 / 32.0)).round() as usize).clamp(4, self.depth.saturating_sub(4).max(4))
    }

    #[inline]
    pub fn ring_z_floor(&self) -> usize {
        self.depth.saturating_sub(11).max(self.z_sea() + 2)
    }

    #[inline]
    pub fn r_ocean(&self) -> f32 {
        self.r_core + self.z_sea() as f32 + 0.55
    }

    #[inline]
    pub fn r_outer(&self) -> f32 {
        self.r_core + self.depth as f32
    }

    #[inline]
    pub fn received_brick_count(&self) -> usize {
        self.received_keys.len()
    }

    #[inline]
    pub fn populated_face_count(&self) -> usize {
        self.faces.iter().filter(|f| f.has_data).count()
    }

    #[inline]
    pub fn total_triangles(&self) -> usize {
        (self.combined_terrain_indices.len()
            + self.combined_ocean_indices.len()
            + self.combined_gas_indices.len()
            + self.dynamic_solid_indices.len()
            + self.dynamic_fx_indices.len())
            / 3
    }

    /// Apply one decoded `PlanetBrickUpdate` in O(1) directly into the target face's RGBA8 volume.
    pub fn apply_update(&mut self, upd: PlanetBrickUpdate) {
        match upd {
            PlanetBrickUpdate::FullSubBrick {
                face,
                bx,
                by,
                bz,
                total_depth,
                sub_depth,
                tick: _,
                density_mat,
                albedo_thermal,
            } => {
                let f = face as usize;
                let td = total_depth as usize;
                let sd = sub_depth as usize;
                if f >= FACE_COUNT || td == 0 || sd == 0 {
                    return;
                }
                if td != self.depth && (16..=64).contains(&td) {
                    self.resize_if_needed(self.face_size, td, self.r_core);
                }
                let expected = CHUNK_SIZE * CHUNK_SIZE * sd * 4;
                if density_mat.len() != expected || albedo_thermal.len() != expected {
                    return;
                }
                let x0 = (bx as usize) * CHUNK_SIZE;
                let y0 = (by as usize) * CHUNK_SIZE;
                let z0 = (bz as usize) * CHUNK_SIZE;
                let n = self.face_size;
                let d = self.depth;
                if x0 >= n || y0 >= n || z0 >= d {
                    return;
                }
                let copy_w = CHUNK_SIZE.min(n - x0);
                let copy_h = CHUNK_SIZE.min(n - y0);
                let copy_d = sd.min(d - z0);
                let row_bytes = copy_w * 4;

                let fd = &mut self.faces[f];
                for dz in 0..copy_d {
                    let gz = z0 + dz;
                    for dy in 0..copy_h {
                        let gy = y0 + dy;
                        let src = ((dz * CHUNK_SIZE + dy) * CHUNK_SIZE) * 4;
                        let dst = ((gz * n + gy) * n + x0) * 4;
                        fd.density_mat[dst..dst + row_bytes]
                            .copy_from_slice(&density_mat[src..src + row_bytes]);
                        fd.albedo_thermal[dst..dst + row_bytes]
                            .copy_from_slice(&albedo_thermal[src..src + row_bytes]);
                    }
                }
                fd.has_data = true;
                fd.dirty = true;
                fd.column_dirty = true;
                self.received_keys.insert((face, bx, by, bz));
                self.total_updates_applied += 1;
            }
            PlanetBrickUpdate::DeltaSubBrick {
                face,
                bx,
                by,
                bz,
                total_depth,
                sub_depth,
                tick: _,
                patches,
            } => {
                let f = face as usize;
                let td = total_depth as usize;
                let sd = sub_depth as usize;
                if f >= FACE_COUNT || td == 0 || sd == 0 {
                    return;
                }
                if td != self.depth && (16..=64).contains(&td) {
                    self.resize_if_needed(self.face_size, td, self.r_core);
                }
                let x0 = (bx as usize) * CHUNK_SIZE;
                let y0 = (by as usize) * CHUNK_SIZE;
                let z0 = (bz as usize) * CHUNK_SIZE;
                let n = self.face_size;
                let d = self.depth;
                let total_sub = CHUNK_SIZE * CHUNK_SIZE * sd;
                let fd = &mut self.faces[f];
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
                    if gx >= n || gy >= n || gz >= d {
                        continue;
                    }
                    let dst = ((gz * n + gy) * n + gx) * 4;
                    fd.density_mat[dst..dst + 4].copy_from_slice(&dm);
                    fd.albedo_thermal[dst..dst + 4].copy_from_slice(&at);
                }
                fd.has_data = true;
                fd.dirty = true;
                fd.column_dirty = true;
                self.received_keys.insert((face, bx, by, bz));
                self.total_updates_applied += 1;
            }
        }
    }

    pub fn apply_updates(&mut self, updates: Vec<PlanetBrickUpdate>) -> usize {
        let count = updates.len();
        for u in updates {
            self.apply_update(u);
        }
        count
    }

    /// Bilinearly sample the planetary surface purely as a function of 3D unit sphere direction `dir`
    /// (seam-free across all 12 cube edges!).
    fn sample_surface_smooth(&self, dir: [f32; 3]) -> ColumnSurface {
        let n = self.face_size;
        let nf = n as f32;
        let (face, u, v) = sphere_dir_to_face_uv(dir);
        let gx = ((u + 1.0) * 0.5 * nf) - 0.5;
        let gy = ((v + 1.0) * 0.5 * nf) - 0.5;
        let ix0 = gx.floor() as i32;
        let iy0 = gy.floor() as i32;
        let fx = (gx - ix0 as f32).clamp(0.0, 1.0);
        let fy = (gy - iy0 as f32).clamp(0.0, 1.0);

        let fetch = |cx: i32, cy: i32| -> ColumnSurface {
            if cx >= 0 && cx < n as i32 && cy >= 0 && cy < n as i32 {
                self.faces[face].column_cache[(cy as usize) * n + (cx as usize)]
            } else {
                let tu = ((cx as f32 + 0.5) / nf) * 2.0 - 1.0;
                let tv = ((cy as f32 + 0.5) / nf) * 2.0 - 1.0;
                let d_nb = face_uv_to_sphere_dir(face, tu, tv);
                let (f2, u2, v2) = sphere_dir_to_face_uv(d_nb);
                let x2 = (((u2 + 1.0) * 0.5 * nf).floor() as isize).clamp(0, (n - 1) as isize) as usize;
                let y2 = (((v2 + 1.0) * 0.5 * nf).floor() as isize).clamp(0, (n - 1) as isize) as usize;
                self.faces[f2].column_cache[y2 * n + x2]
            }
        };

        let s00 = fetch(ix0, iy0);
        let s10 = fetch(ix0 + 1, iy0);
        let s01 = fetch(ix0, iy0 + 1);
        let s11 = fetch(ix0 + 1, iy0 + 1);

        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        let blend = |a: f32, b: f32, c: f32, d: f32| a * w00 + b * w10 + c * w01 + d * w11;

        ColumnSurface {
            surf_z: blend(s00.surf_z, s10.surf_z, s01.surf_z, s11.surf_z),
            water_depth: blend(s00.water_depth, s10.water_depth, s01.water_depth, s11.water_depth),
            ice_frac: blend(s00.ice_frac, s10.ice_frac, s01.ice_frac, s11.ice_frac),
            color: [
                blend(s00.color[0], s10.color[0], s01.color[0], s11.color[0]),
                blend(s00.color[1], s10.color[1], s01.color[1], s11.color[1]),
                blend(s00.color[2], s10.color[2], s01.color[2], s11.color[2]),
            ],
            emissive: blend(s00.emissive, s10.emissive, s01.emissive, s11.emissive),
            roughness: blend(s00.roughness, s10.roughness, s01.roughness, s11.roughness),
        }
    }

    /// Sample smooth surface radius `(r_core + surf_z)` along any 3D direction `dir`.
    pub fn sample_surface_radius(&self, dir: Vec3) -> f32 {
        let d = dir.normalize_or(Vec3::Y);
        let s = self.sample_surface_smooth(d.to_array());
        self.r_core + s.surf_z.max(self.z_sea() as f32 + 0.55)
    }

    /// Incrementally rebuild up to `max_faces` dirty faces per frame (keeping FPS >100!),
    /// then refresh combined GPU staging buffers and equirectangular minimap if any face rebuilt.
    pub fn rebuild_dirty_faces(&mut self, max_faces: usize) -> usize {
        let z_sea = self.z_sea();
        let ring_z_floor = self.ring_z_floor();

        // First refresh all dirty column caches (super fast O(N^2 * D) scan) so cross-face seam lookups stay fresh
        for f in 0..FACE_COUNT {
            if self.faces[f].column_dirty {
                self.faces[f].recompute_columns(z_sea, ring_z_floor);
            }
        }

        let mut dirty_list = Vec::with_capacity(FACE_COUNT);
        for f in 0..FACE_COUNT {
            if self.faces[f].dirty && self.faces[f].has_data {
                dirty_list.push(f);
            }
        }
        if dirty_list.is_empty() {
            return 0;
        }

        let budget = max_faces.clamp(1, FACE_COUNT);
        let mut rebuilt = 0usize;
        for &f in dirty_list.iter().take(budget) {
            let (tv, ti, ov, oi, gv, gi) = self.build_single_face_meshes(f);
            let fd = &mut self.faces[f];
            fd.terrain_verts = tv;
            fd.terrain_indices = ti;
            fd.ocean_verts = ov;
            fd.ocean_indices = oi;
            fd.gas_verts = gv;
            fd.gas_indices = gi;
            fd.dirty = false;
            rebuilt += 1;
        }

        if rebuilt > 0 {
            self.merge_combined_meshes();
            self.rebuild_equirectangular_minimap();
            self.gpu_dirty = true;
        }
        rebuilt
    }

    fn build_single_face_meshes(
        &self,
        face: usize,
    ) -> (
        Vec<PlanetVertex>,
        Vec<u32>,
        Vec<PlanetVertex>,
        Vec<u32>,
        Vec<PlanetVertex>,
        Vec<u32>,
    ) {
        let n = self.face_size;
        let nf = n as f32;
        let d = self.depth;
        let r_core = self.r_core;
        let r_ocean = self.r_ocean();
        let ring_z_floor = self.ring_z_floor();

        // 1. Seamless Curved Terrain Shell (`sub × sub` smooth quads warped onto the sphere)
        let sub = (n * 3 / 2).clamp(32, 96);
        let stride = sub + 1;
        let mut terrain_verts = Vec::with_capacity(stride * stride + 4096);
        let mut terrain_indices = Vec::with_capacity(sub * sub * 6 + 6144);

        for iy in 0..=sub {
            let v = (iy as f32 / sub as f32) * 2.0 - 1.0;
            for ix in 0..=sub {
                let u = (ix as f32 / sub as f32) * 2.0 - 1.0;
                let dir = face_uv_to_sphere_dir(face, u, v);
                let dir_v = Vec3::from_array(dir);
                let samp = self.sample_surface_smooth(dir);
                let r = r_core + samp.surf_z;
                let pos = dir_v * r;

                // Seamless 3D analytical spherical tangent gradient normal (identical across cube seams!)
                let ref_up = if dir_v.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
                let t1 = ref_up.cross(dir_v).normalize();
                let t2 = dir_v.cross(t1).normalize();
                let eps = 0.015f32;
                let d_p1 = (dir_v + t1 * eps).normalize();
                let d_m1 = (dir_v - t1 * eps).normalize();
                let d_p2 = (dir_v + t2 * eps).normalize();
                let d_m2 = (dir_v - t2 * eps).normalize();

                let r_p1 = r_core + self.sample_surface_smooth(d_p1.to_array()).surf_z;
                let r_m1 = r_core + self.sample_surface_smooth(d_m1.to_array()).surf_z;
                let r_p2 = r_core + self.sample_surface_smooth(d_p2.to_array()).surf_z;
                let r_m2 = r_core + self.sample_surface_smooth(d_m2.to_array()).surf_z;

                let dp1 = d_p1 * r_p1 - d_m1 * r_m1;
                let dp2 = d_p2 * r_p2 - d_m2 * r_m2;
                let mut normal = dp1.cross(dp2).normalize_or_zero();
                if normal.dot(dir_v) < 0.0 {
                    normal = -normal;
                }
                if normal.length_squared() < 0.5 {
                    normal = dir_v;
                }

                terrain_verts.push(PlanetVertex {
                    pos: pos.to_array(),
                    normal: normal.to_array(),
                    color: [samp.color[0], samp.color[1], samp.color[2], samp.roughness],
                    flags: [samp.water_depth, samp.emissive, MAT_TERRAIN, samp.ice_frac],
                });
            }
        }
        for iy in 0..sub {
            for ix in 0..sub {
                let i00 = (iy * stride + ix) as u32;
                let i10 = i00 + 1;
                let i01 = ((iy + 1) * stride + ix) as u32;
                let i11 = i01 + 1;
                terrain_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
            }
        }

        // 2. Curved Specular/Fresnel Sapphire Ocean Shell (`o_sub × o_sub`)
        let o_sub = n.min(64);
        let o_stride = o_sub + 1;
        let mut ocean_verts = Vec::with_capacity(o_stride * o_stride);
        let mut ocean_indices = Vec::with_capacity(o_sub * o_sub * 6);
        let mut ocean_depths = vec![0.0f32; o_stride * o_stride];

        for iy in 0..=o_sub {
            let v = (iy as f32 / o_sub as f32) * 2.0 - 1.0;
            for ix in 0..=o_sub {
                let u = (ix as f32 / o_sub as f32) * 2.0 - 1.0;
                let dir = face_uv_to_sphere_dir(face, u, v);
                let samp = self.sample_surface_smooth(dir);
                ocean_depths[iy * o_stride + ix] = samp.water_depth;
                ocean_verts.push(PlanetVertex {
                    pos: [dir[0] * r_ocean, dir[1] * r_ocean, dir[2] * r_ocean],
                    normal: dir,
                    color: [0.04, 0.24, 0.54, 0.08],
                    flags: [samp.water_depth, 0.0, MAT_OCEAN, samp.ice_frac],
                });
            }
        }
        for iy in 0..o_sub {
            for ix in 0..o_sub {
                let idx00 = iy * o_stride + ix;
                let idx10 = idx00 + 1;
                let idx01 = (iy + 1) * o_stride + ix;
                let idx11 = idx01 + 1;
                if ocean_depths[idx00] > 0.04
                    || ocean_depths[idx10] > 0.04
                    || ocean_depths[idx01] > 0.04
                    || ocean_depths[idx11] > 0.04
                {
                    ocean_indices.extend_from_slice(&[
                        idx00 as u32,
                        idx10 as u32,
                        idx11 as u32,
                        idx00 as u32,
                        idx11 as u32,
                        idx01 as u32,
                    ]);
                }
            }
        }

        // 3. Live Orbital Floating Sky-Ring Islands & Buoyant Crystal Cores (`z >= ring_z_floor`)
        //    Every exposed solid voxel face is warped cleanly onto the 3D sphere via
        //    `face_uv_z_to_world_pos`, with smooth radial-blended normals and glowing Buoyant Cores!
        let fd = &self.faces[face];
        let mut push_curved_quad = |p0: Vec3,
                                    p1: Vec3,
                                    p2: Vec3,
                                    p3: Vec3,
                                    n_hint: Vec3,
                                    col: [f32; 4],
                                    flags: [f32; 4]| {
            let mut n_face = (p1 - p0).cross(p2 - p0).normalize_or_zero();
            if n_face.length_squared() < 0.25 {
                n_face = n_hint;
            } else if n_face.dot(n_hint) < 0.0 {
                // Flip winding to guarantee outward CCW orientation
                let base = terrain_verts.len() as u32;
                let n_s = (-n_face * 0.65 + n_hint * 0.35).normalize_or(n_hint);
                for p in [p0, p3, p2, p1] {
                    terrain_verts.push(PlanetVertex {
                        pos: p.to_array(),
                        normal: n_s.to_array(),
                        color: col,
                        flags,
                    });
                }
                terrain_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                return;
            }
            let n_s = (n_face * 0.65 + n_hint * 0.35).normalize_or(n_hint);
            let base = terrain_verts.len() as u32;
            for p in [p0, p1, p2, p3] {
                terrain_verts.push(PlanetVertex {
                    pos: p.to_array(),
                    normal: n_s.to_array(),
                    color: col,
                    flags,
                });
            }
            terrain_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        };

        for z in ring_z_floor..d {
            for y in 0..n {
                for x in 0..n {
                    let off = fd.voxel_offset(x, y, z);
                    let solid = fd.density_mat[off];
                    if solid < 100 {
                        continue;
                    }
                    let rough = (fd.density_mat[off + 3] as f32 / 255.0).clamp(0.15, 0.92);
                    let ar = fd.albedo_thermal[off] as f32 / 255.0;
                    let ag = fd.albedo_thermal[off + 1] as f32 / 255.0;
                    let ab = fd.albedo_thermal[off + 2] as f32 / 255.0;
                    let therm = fd.albedo_thermal[off + 3] as f32 / 255.0;

                    let air_below = fd.solid_at(x as isize, y as isize, z as isize - 1) < 100;
                    let air_above = fd.solid_at(x as isize, y as isize, z as isize + 1) < 100;
                    let is_crystal_core = z >= d.saturating_sub(11)
                        && z <= d.saturating_sub(8)
                        && solid >= 120
                        && therm < 0.25
                        && (ab > ar + 0.03 && ab > 0.36);

                    let (vcol, vflags) = if is_crystal_core {
                        (
                            [0.14, 0.72, 1.0, 0.18],
                            [0.0, 0.95, MAT_CRYSTAL_CORE, 0.0],
                        )
                    } else if therm > 0.45 {
                        (
                            [0.25, 0.08, 0.04, 0.50],
                            [0.0, 1.8 + therm * 2.4, MAT_TERRAIN, 0.0],
                        )
                    } else if air_above {
                        // Lush emerald top plateau of floating Sky-Ring island
                        let g_boost = if ag >= ar * 0.95 { (ag * 1.15).min(0.50) } else { ag };
                        ([ar * 0.78, g_boost, ab * 0.76, rough], [0.0, 0.0, MAT_TERRAIN, 0.0])
                    } else {
                        // Deep craggy dark-basalt hull & tapered keel of floating Sky-Ring island
                        ([0.22 + ar * 0.18, 0.24 + ag * 0.16, 0.30 + ab * 0.18, rough], [0.0, 0.0, MAT_TERRAIN, 0.0])
                    };

                    let u0 = (x as f32 / nf) * 2.0 - 1.0;
                    let u1 = ((x + 1) as f32 / nf) * 2.0 - 1.0;
                    let v0 = (y as f32 / nf) * 2.0 - 1.0;
                    let v1 = ((y + 1) as f32 / nf) * 2.0 - 1.0;
                    let z0 = z as f32;
                    let z1 = z as f32 + 1.0;

                    let wp = |u: f32, v: f32, zf: f32| {
                        Vec3::from_array(face_uv_z_to_world_pos(face, u, v, zf, r_core))
                    };
                    let (rad_up, tang_u, tang_v) =
                        face_tangent_basis(face, 0.5 * (u0 + u1), 0.5 * (v0 + v1));

                    // +Z (radial outward top)
                    if air_above {
                        push_curved_quad(
                            wp(u0, v0, z1),
                            wp(u1, v0, z1),
                            wp(u1, v1, z1),
                            wp(u0, v1, z1),
                            rad_up,
                            vcol,
                            vflags,
                        );
                    }
                    // -Z (radial inward underside)
                    if air_below {
                        push_curved_quad(
                            wp(u0, v0, z0),
                            wp(u0, v1, z0),
                            wp(u1, v1, z0),
                            wp(u1, v0, z0),
                            -rad_up,
                            vcol,
                            vflags,
                        );
                    }
                    // +U (east wall)
                    if fd.solid_at(x as isize + 1, y as isize, z as isize) < 100 {
                        push_curved_quad(
                            wp(u1, v0, z0),
                            wp(u1, v1, z0),
                            wp(u1, v1, z1),
                            wp(u1, v0, z1),
                            tang_u,
                            vcol,
                            vflags,
                        );
                    }
                    // -U (west wall)
                    if fd.solid_at(x as isize - 1, y as isize, z as isize) < 100 {
                        push_curved_quad(
                            wp(u0, v1, z0),
                            wp(u0, v0, z0),
                            wp(u0, v0, z1),
                            wp(u0, v1, z1),
                            -tang_u,
                            vcol,
                            vflags,
                        );
                    }
                    // +V (south wall)
                    if fd.solid_at(x as isize, y as isize + 1, z as isize) < 100 {
                        push_curved_quad(
                            wp(u1, v1, z0),
                            wp(u0, v1, z0),
                            wp(u0, v1, z1),
                            wp(u1, v1, z1),
                            tang_v,
                            vcol,
                            vflags,
                        );
                    }
                    // -V (north wall)
                    if fd.solid_at(x as isize, y as isize - 1, z as isize) < 100 {
                        push_curved_quad(
                            wp(u0, v0, z0),
                            wp(u1, v0, z0),
                            wp(u1, v0, z1),
                            wp(u0, v0, z1),
                            -tang_v,
                            vcol,
                            vflags,
                        );
                    }
                }
            }
        }

        // 4. Translucent Volumetric Steam / Smoke Plumes (`gas_density >= 48`)
        let mut gas_verts = Vec::with_capacity(512);
        let mut gas_indices = Vec::with_capacity(768);
        let z_min_gas = self.z_sea().saturating_sub(1);
        for z in (z_min_gas..d).step_by(2) {
            for y in (0..n).step_by(2) {
                for x in (0..n).step_by(2) {
                    let off = fd.voxel_offset(x, y, z);
                    let gas = fd.density_mat[off + 2];
                    if gas < 48 {
                        continue;
                    }
                    let g_frac = (gas as f32 / 255.0).clamp(0.18, 0.72);
                    let ar = fd.albedo_thermal[off] as f32 / 255.0;
                    let ag = fd.albedo_thermal[off + 1] as f32 / 255.0;
                    let ab = fd.albedo_thermal[off + 2] as f32 / 255.0;
                    let therm = fd.albedo_thermal[off + 3] as f32 / 255.0;

                    let u = ((x as f32 + 0.5) / nf) * 2.0 - 1.0;
                    let v = ((y as f32 + 0.5) / nf) * 2.0 - 1.0;
                    let center = Vec3::from_array(face_uv_z_to_world_pos(
                        face,
                        u,
                        v,
                        z as f32 + 0.7,
                        r_core,
                    ));
                    let (up, tu, tv) = face_tangent_basis(face, u, v);
                    let half_w = 1.45 + g_frac * 1.1;
                    let half_h = 1.65 + g_frac * 1.25;
                    let col = [
                        ar.max(0.28),
                        ag.max(0.28),
                        ab.max(0.32),
                        (g_frac * 0.42).clamp(0.12, 0.38),
                    ];
                    let flags = [0.0, if therm > 0.35 { therm * 1.4 } else { 0.0 }, MAT_GAS, 0.0];

                    // Two crossed vertical radial billboards per plume puff so visible from every orbit angle
                    for axis in [tu, tv] {
                        let base = gas_verts.len() as u32;
                        let p0 = center - axis * half_w - up * (half_h * 0.4);
                        let p1 = center + axis * half_w - up * (half_h * 0.4);
                        let p2 = center + axis * half_w + up * half_h;
                        let p3 = center - axis * half_w + up * half_h;
                        for p in [p0, p1, p2, p3] {
                            gas_verts.push(PlanetVertex {
                                pos: p.to_array(),
                                normal: up.to_array(),
                                color: col,
                                flags,
                            });
                        }
                        gas_indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                    }
                }
            }
        }

        (terrain_verts, terrain_indices, ocean_verts, ocean_indices, gas_verts, gas_indices)
    }

    fn merge_combined_meshes(&mut self) {
        self.combined_terrain_verts.clear();
        self.combined_terrain_indices.clear();
        self.combined_ocean_verts.clear();
        self.combined_ocean_indices.clear();
        self.combined_gas_verts.clear();
        self.combined_gas_indices.clear();

        for f in &self.faces {
            let tv_base = self.combined_terrain_verts.len() as u32;
            self.combined_terrain_verts.extend_from_slice(&f.terrain_verts);
            self.combined_terrain_indices
                .extend(f.terrain_indices.iter().map(|&i| i + tv_base));

            let ov_base = self.combined_ocean_verts.len() as u32;
            self.combined_ocean_verts.extend_from_slice(&f.ocean_verts);
            self.combined_ocean_indices
                .extend(f.ocean_indices.iter().map(|&i| i + ov_base));

            let gv_base = self.combined_gas_verts.len() as u32;
            self.combined_gas_verts.extend_from_slice(&f.gas_verts);
            self.combined_gas_indices
                .extend(f.gas_indices.iter().map(|&i| i + gv_base));
        }
    }

    /// Build a live 256×128 equirectangular satellite minimap across all 6 faces!
    fn rebuild_equirectangular_minimap(&mut self) {
        let w = PLANET_MINIMAP_W;
        let h = PLANET_MINIMAP_H;
        if self.minimap_rgba.len() != w * h * 4 {
            self.minimap_rgba.resize(w * h * 4, 255);
        }
        for py in 0..h {
            // lat: +pi/2 (north pole, top) -> -pi/2 (south pole, bottom)
            let lat = std::f32::consts::FRAC_PI_2
                - ((py as f32 + 0.5) / h as f32) * std::f32::consts::PI;
            let clat = lat.cos();
            let slat = lat.sin();
            for px in 0..w {
                let lon = ((px as f32 + 0.5) / w as f32) * std::f32::consts::TAU - std::f32::consts::PI;
                let dir = [clat * lon.cos(), slat, clat * lon.sin()];
                let s = self.sample_surface_smooth(dir);
                let rgb = if s.emissive > 0.35 {
                    [1.0, 0.44, 0.12]
                } else if s.water_depth > 0.12 {
                    if s.ice_frac > 0.45 {
                        [0.72, 0.88, 0.98]
                    } else {
                        [0.06, 0.24, 0.52]
                    }
                } else {
                    [
                        (s.color[0] * 1.15).clamp(0.0, 1.0),
                        (s.color[1] * 1.15).clamp(0.0, 1.0),
                        (s.color[2] * 1.15).clamp(0.0, 1.0),
                    ]
                };
                let off = (py * w + px) * 4;
                self.minimap_rgba[off] = (rgb[0] * 255.0) as u8;
                self.minimap_rgba[off + 1] = (rgb[1] * 255.0) as u8;
                self.minimap_rgba[off + 2] = (rgb[2] * 255.0) as u8;
                self.minimap_rgba[off + 3] = 255;
            }
        }
        self.minimap_dirty = true;
    }

    /// Map a 3D spherical position to equirectangular minimap UV `(0..1, 0..1)`.
    #[inline]
    pub fn sphere_pos_to_minimap_uv(pos: Vec3) -> (f32, f32) {
        let d = pos.normalize_or(Vec3::Z);
        let lon = d.z.atan2(d.x); // [-pi, pi]
        let lat = d.y.clamp(-1.0, 1.0).asin(); // [-pi/2, pi/2]
        let u = (lon + std::f32::consts::PI) / std::f32::consts::TAU;
        let v = 0.5 - lat / std::f32::consts::PI;
        (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
    }

    // =========================================================================
    // Dynamic Every-Frame 3D Spherical Entity, Shrine, Beam & Projectile Mesher
    // =========================================================================

    pub fn entity_world_frame(
        &self,
        face: u8,
        gx: f32,
        gy: f32,
        gz: f32,
        aim_x: f32,
        aim_y: f32,
    ) -> (Vec3, Vec3, Vec3, Vec3) {
        let nf = self.face_size.max(1) as f32;
        let u = (gx / nf) * 2.0 - 1.0;
        let v = (gy / nf) * 2.0 - 1.0;
        let (up, tang_u, tang_v) = face_tangent_basis(face as usize, u, v);
        let surf_z = self.sample_surface_smooth(up.to_array()).surf_z;
        let z_eff = if gz > 0.5 { gz.max(surf_z) } else { surf_z.max(self.z_sea() as f32 + 0.55) };
        let pos = up * (self.r_core + z_eff);
        let fwd = (tang_u * aim_x + tang_v * aim_y).normalize_or(tang_u);
        let right = fwd.cross(up).normalize_or(tang_v);
        let fwd_ortho = up.cross(right).normalize_or(fwd);
        (pos, up, right, fwd_ortho)
    }

    /// Rebuild the live per-frame curved 3D entity, shrine, actuator-beam, and projectile meshes!
    pub fn rebuild_dynamic_entities(&mut self, entities: Option<&LiveEntityState>, sim_time: f32) {
        self.dynamic_solid_verts.clear();
        self.dynamic_solid_indices.clear();
        self.dynamic_fx_verts.clear();
        self.dynamic_fx_indices.clear();

        let Some(ent) = entities else {
            return;
        };

        // 1. All 60 Live Neural Bots & Human Hero
        for p in &ent.players {
            if p.hp <= 0 {
                continue;
            }
            let (base_pos, up, right, fwd) =
                self.entity_world_frame(p.face, p.x, p.y, p.z, p.aim_x, p.aim_y);
            let is_self = p.id == ent.your_player_id;

            let team_col = if is_self {
                [1.00, 0.88, 0.28]
            } else if p.team == 0 {
                [0.20, 0.78, 1.00] // West Frost Sapphire/Cyan
            } else {
                [1.00, 0.36, 0.18] // East Pyro Crimson/Gold
            };
            let accent_col = if p.team == 0 {
                [0.65, 0.95, 1.00]
            } else {
                [1.00, 0.82, 0.32]
            };

            let scale = if is_self { 1.35 } else { 1.15 };
            let em_body = if is_self { 0.85 } else if p.ult_active { 1.35 } else { 0.45 };

            match p.class_id {
                CLASS_SKYWING => {
                    // Hovering Skywing with animated swept glowing wings!
                    let bob = (sim_time * 4.5 + p.id as f32).sin() * 0.25;
                    let center = base_pos + up * (1.45 * scale + bob);
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        center,
                        right,
                        up,
                        fwd,
                        Vec3::new(0.55, 0.70, 1.15) * scale,
                        [team_col[0], team_col[1], team_col[2], 0.30],
                        [0.0, em_body + 0.35, MAT_ENTITY, 0.0],
                    );
                    let flap = (sim_time * 7.0 + p.id as f32 * 1.3).sin() * 0.35;
                    for side in [-1.0f32, 1.0] {
                        let w_dir = (right * side + up * (0.22 + flap)).normalize();
                        let w_up = fwd.cross(w_dir).normalize_or(up) * side;
                        let w_center = center + w_dir * (1.25 * scale) - fwd * (0.15 * scale);
                        push_oriented_box(
                            &mut self.dynamic_solid_verts,
                            &mut self.dynamic_solid_indices,
                            w_center,
                            w_dir,
                            w_up,
                            fwd,
                            Vec3::new(1.35, 0.14, 0.65) * scale,
                            [accent_col[0], accent_col[1], accent_col[2], 0.22],
                            [0.0, em_body + 0.75, MAT_ENTITY, 0.0],
                        );
                    }
                }
                CLASS_STRIDER => {
                    // Heavy armored Strider: broad torso, pauldrons, helmet & tower shield
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (1.05 * scale),
                        right,
                        up,
                        fwd,
                        Vec3::new(1.05, 1.30, 0.78) * scale,
                        [team_col[0], team_col[1], team_col[2], 0.36],
                        [0.0, em_body, MAT_ENTITY, 0.0],
                    );
                    // Crown / Helmet
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (2.05 * scale),
                        right,
                        up,
                        fwd,
                        Vec3::new(0.62, 0.56, 0.62) * scale,
                        [accent_col[0], accent_col[1], accent_col[2], 0.24],
                        [0.0, em_body + 0.35, MAT_ENTITY, 0.0],
                    );
                    // Forward energy shield
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (1.10 * scale) + fwd * (0.68 * scale),
                        right,
                        up,
                        fwd,
                        Vec3::new(0.95, 1.25, 0.20) * scale,
                        [accent_col[0], accent_col[1], accent_col[2], 0.18],
                        [0.0, em_body + 0.55, MAT_ENTITY, 0.0],
                    );
                }
                CLASS_TIDEKIN => {
                    // Agile amphibious Tidekin with sleek dorsal fin & twin spears
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (0.95 * scale),
                        right,
                        up,
                        fwd,
                        Vec3::new(0.75, 1.20, 0.72) * scale,
                        [team_col[0], team_col[1], team_col[2], 0.24],
                        [0.0, em_body + 0.15, MAT_ENTITY, 0.0],
                    );
                    push_oriented_pyramid(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (1.55 * scale),
                        right,
                        up,
                        fwd,
                        0.45 * scale,
                        1.05 * scale,
                        [accent_col[0], accent_col[1], accent_col[2], 0.20],
                        [0.0, em_body + 0.55, MAT_ENTITY, 0.0],
                    );
                }
                CLASS_EMBERSTALKER => {
                    // Solo Emberstalker predator with blazing twin blades
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (0.90 * scale),
                        right,
                        up,
                        fwd,
                        Vec3::new(0.72, 1.10, 0.90) * scale,
                        [1.0, 0.44, 0.18, 0.30],
                        [0.0, em_body + 0.40, MAT_ENTITY, 0.0],
                    );
                    for side in [-0.62f32, 0.62] {
                        push_oriented_box(
                            &mut self.dynamic_solid_verts,
                            &mut self.dynamic_solid_indices,
                            base_pos + up * (1.05 * scale) + right * (side * scale) + fwd * (0.55 * scale),
                            right,
                            up,
                            fwd,
                            Vec3::new(0.16, 0.22, 1.25) * scale,
                            [1.0, 0.78, 0.26, 0.16],
                            [0.0, 1.45, MAT_ENTITY, 0.0],
                        );
                    }
                }
                CLASS_BURROWER | _ => {
                    // Heavy crystal-armored Burrower golem
                    push_oriented_box(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (0.85 * scale),
                        right,
                        up,
                        fwd,
                        Vec3::new(1.18, 0.95, 1.28) * scale,
                        [team_col[0], team_col[1], team_col[2], 0.34],
                        [0.0, em_body + 0.15, MAT_ENTITY, 0.0],
                    );
                    push_oriented_pyramid(
                        &mut self.dynamic_solid_verts,
                        &mut self.dynamic_solid_indices,
                        base_pos + up * (1.35 * scale),
                        right,
                        up,
                        fwd,
                        0.68 * scale,
                        0.95 * scale,
                        [0.44, 0.94, 0.88, 0.18],
                        [0.0, em_body + 0.65, MAT_ENTITY, 0.0],
                    );
                }
            }

            // Glowing ground aura ring beneath every hero so all 60 bots read unmistakably from BOTH orbit & surface!
            push_spherical_ring(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                base_pos + up * 0.18,
                up,
                right,
                fwd,
                if is_self { 1.45 } else { 1.05 },
                if is_self { 2.10 } else { 1.55 },
                12,
                [team_col[0], team_col[1], team_col[2], if is_self { 0.86 } else { 0.64 }],
                [0.0, if is_self { 1.8 } else { 1.2 }, MAT_BEAM, 0.0],
            );

            // Compact orbital beacon crest (crisp from orbit without overpowering close-up hero silhouettes!)
            if is_self {
                push_radial_pillar(
                    &mut self.dynamic_fx_verts,
                    &mut self.dynamic_fx_indices,
                    base_pos + up * (2.4 * scale),
                    up,
                    right,
                    fwd,
                    0.26,
                    3.2,
                    [team_col[0], team_col[1], team_col[2], 0.65],
                    [0.0, 1.8, MAT_BEAM, 0.0],
                );
            } else {
                push_oriented_pyramid(
                    &mut self.dynamic_solid_verts,
                    &mut self.dynamic_solid_indices,
                    base_pos + up * (2.6 * scale),
                    right,
                    up,
                    fwd,
                    0.24,
                    0.55,
                    [team_col[0], team_col[1], team_col[2], 0.16],
                    [0.0, 1.35, MAT_ENTITY, 0.0],
                );
            }
        }

        // 2. All 6 Live Planetary Shrines (Crystalline Beacon Spires + Capture Rings + Orbital Light Pillars!)
        for s in &ent.shrines {
            let (base_pos, up, right, fwd) =
                self.entity_world_frame(s.face, s.x, s.y, 0.0, 1.0, 0.0);
            let shrine_col = if s.owner < 0 {
                [0.22, 0.80, 1.00]
            } else if s.owner > 0 {
                [1.00, 0.38, 0.18]
            } else {
                [1.00, 0.86, 0.34]
            };

            // Pedestal base
            push_oriented_box(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                base_pos + up * 0.55,
                right,
                up,
                fwd,
                Vec3::new(2.4, 1.1, 2.4),
                [0.34, 0.38, 0.48, 0.45],
                [0.0, 0.18, MAT_ENTITY, 0.0],
            );
            // Tall glowing crystalline obelisk spire
            push_oriented_pyramid(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                base_pos + up * 1.1,
                right,
                up,
                fwd,
                1.05,
                4.8,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.16],
                [0.0, 1.85, MAT_ENTITY, 0.0],
            );
            // Rotating floating crystal crown
            let spin = sim_time * 1.6 + s.id as f32;
            let r_rot = right * spin.cos() + fwd * spin.sin();
            let f_rot = -right * spin.sin() + fwd * spin.cos();
            push_oriented_box(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                base_pos + up * (6.2 + (sim_time * 2.5 + s.id as f32).sin() * 0.35),
                r_rot,
                up,
                f_rot,
                Vec3::new(0.95, 1.35, 0.95),
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.12],
                [0.0, 2.5, MAT_ENTITY, 0.0],
            );
            // Dual concentric capture rings tangent to the sphere
            push_spherical_ring(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                base_pos + up * 0.26,
                up,
                right,
                fwd,
                2.8,
                3.6,
                20,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.80],
                [0.0, 1.8, MAT_BEAM, 0.0],
            );
            push_spherical_ring(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                base_pos + up * 0.32,
                up,
                right,
                fwd,
                4.4,
                4.9,
                24,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.48],
                [0.0, 1.4, MAT_BEAM, 0.0],
            );
            // Tall luminous vertical light pillar clearly visible from full-globe orbit!
            push_radial_pillar(
                &mut self.dynamic_fx_verts,
                &mut self.dynamic_fx_indices,
                base_pos + up * 1.5,
                up,
                right,
                fwd,
                0.78,
                14.5,
                [shrine_col[0], shrine_col[1], shrine_col[2], 0.52],
                [0.0, 2.6, MAT_BEAM, 0.0],
            );
        }

        // 3. Live 3D Elemental Actuator Beams curving along the sphere!
        for pulse in &ent.actuator_pulses {
            let (p0_base, up0, _, _) =
                self.entity_world_frame(pulse.face, pulse.x0, pulse.y0, pulse.z0, 1.0, 0.0);
            let (p1_base, up1, r1, f1) =
                self.entity_world_frame(pulse.face, pulse.x1, pulse.y1, pulse.z1, 1.0, 0.0);
            let p0 = p0_base + up0 * 1.35;
            let p1 = p1_base + up1 * 1.15;

            let (beam_col, em_val) = if pulse.thermal_flux > 0.2 {
                ([1.00, 0.56, 0.14], 3.2) // Fiery orange-gold heat beam
            } else if pulse.thermal_flux < -0.2 {
                ([0.32, 0.94, 1.00], 3.0) // Cryo ice-cyan ray
            } else if pulse.sp_oil > 0.2 {
                ([0.78, 0.28, 1.00], 2.4) // Ultraviolet oil arc
            } else if pulse.sp_water > 0.2 {
                ([0.24, 0.78, 1.00], 2.2) // Azure water torrent
            } else {
                ([0.72, 0.78, 0.90], 1.2) // Smoke/kinetic wave
            };

            // Curved spherical ribbon arc (8 segments slerped over the curved planet!)
            let segs = 8usize;
            let r0 = p0.length();
            let r1_len = p1.length();
            let d0 = p0 / r0.max(1e-4);
            let d1 = p1 / r1_len.max(1e-4);
            let arc_lift = (p0.distance(p1) * 0.10).clamp(0.4, 2.4);
            let beam_w = (0.14 + pulse.focus_radius * 0.025).clamp(0.14, 0.32);

            let mut pts = Vec::with_capacity(segs + 1);
            for s in 0..=segs {
                let t = s as f32 / segs as f32;
                let dir = d0.lerp(d1, t).normalize_or(d0);
                let r = r0 + (r1_len - r0) * t + (t * std::f32::consts::PI).sin() * arc_lift;
                pts.push(dir * r);
            }
            for s in 0..segs {
                let a = pts[s];
                let b = pts[s + 1];
                let seg_dir = (b - a).normalize_or(Vec3::X);
                let rad_mid = (a + b).normalize_or(Vec3::Y);
                let side = seg_dir.cross(rad_mid).normalize_or(Vec3::Z);
                // Two orthogonal ribbon planes so the glowing beam is thick from orbit AND surface
                for perp in [side, rad_mid] {
                    let base = self.dynamic_fx_verts.len() as u32;
                    let c = [beam_col[0], beam_col[1], beam_col[2], 0.88];
                    let fl = [0.0, em_val, MAT_BEAM, 0.0];
                    for pos in [
                        a - perp * beam_w,
                        a + perp * beam_w,
                        b + perp * beam_w,
                        b - perp * beam_w,
                    ] {
                        self.dynamic_fx_verts.push(PlanetVertex {
                            pos: pos.to_array(),
                            normal: rad_mid.to_array(),
                            color: c,
                            flags: fl,
                        });
                    }
                    self.dynamic_fx_indices
                        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            // Incandescent impact flare at beam endpoint
            push_oriented_box(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                p1,
                r1,
                up1,
                f1,
                Vec3::splat((0.42 + pulse.focus_radius * 0.08).clamp(0.42, 0.95)),
                [beam_col[0], beam_col[1], beam_col[2], 0.15],
                [0.0, em_val * 1.15, MAT_BEAM, 0.0],
            );
        }

        // 4. Live 3D Elemental Projectiles orbiting/streaking above the planet surface!
        for pr in &ent.projectiles {
            let (base_pos, up, right, fwd) =
                self.entity_world_frame(pr.face, pr.x, pr.y, pr.z, pr.vx, pr.vy);
            let p_pos = base_pos + up * 1.25;
            let (pcol, em_v) = match pr.element {
                0 => ([1.00, 0.54, 0.14], 3.4), // Fire
                1 => ([0.36, 0.94, 1.00], 3.2), // Frost
                2 => ([0.82, 0.34, 1.00], 2.8), // Oil
                _ => ([1.00, 0.86, 0.32], 3.8), // Meteor/Nova
            };
            let sz = (pr.radius * 0.72).clamp(0.42, 1.45);
            push_oriented_box(
                &mut self.dynamic_solid_verts,
                &mut self.dynamic_solid_indices,
                p_pos,
                right,
                up,
                fwd,
                Vec3::new(sz, sz, sz * 2.1),
                [pcol[0], pcol[1], pcol[2], 0.12],
                [0.0, em_v, MAT_BEAM, 0.0],
            );
        }
    }

    /// Find the densest active multi-bot combat cluster on the sphere for the Surface Horizon Action shot!
    /// Returns `(focus_world_pos, focus_tangent_fwd)` preferring sunlit clusters near continents/coastlines.
    pub fn find_best_action_cluster(
        &self,
        entities: Option<&LiveEntityState>,
        sun_dir: Vec3,
    ) -> (Vec3, Vec3) {
        let sun = sun_dir.normalize_or(Vec3::new(-0.78, 0.28, 0.56).normalize());
        if let Some(ent) = entities {
            let mut positions: Vec<(Vec3, Vec3, u8)> = Vec::with_capacity(ent.players.len());
            for p in &ent.players {
                if p.hp <= 0 {
                    continue;
                }
                let (wp, _, _, fwd) = self.entity_world_frame(p.face, p.x, p.y, p.z, p.aim_x, p.aim_y);
                positions.push((wp, fwd, p.team));
            }
            let mut best_score = f32::NEG_INFINITY;
            let mut best_pos = Vec3::ZERO;
            let mut best_fwd = Vec3::X;
            for (i, &(pi, fwdi, ti)) in positions.iter().enumerate() {
                let dir_i = pi.normalize_or(Vec3::Z);
                let sun_dot = dir_i.dot(sun);
                let mut neighbors = 0.0f32;
                let mut has_enemy = false;
                let mut centroid = pi;
                let mut w_sum = 1.0f32;
                for (j, &(pj, _, tj)) in positions.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let d = pi.distance(pj);
                    if d < 22.0 {
                        let w = (1.0 - d / 22.0).max(0.1);
                        neighbors += w;
                        centroid += pj * w;
                        w_sum += w;
                        if tj != ti {
                            has_enemy = true;
                        }
                    }
                }
                let samp = self.sample_surface_smooth(dir_i.to_array());
                // Avoid placing the Surface Horizon camera directly underneath a low Orbital Sky-Ring canopy
                let (fi, ui, vi) = sphere_dir_to_face_uv(dir_i.to_array());
                let xi = (((ui + 1.0) * 0.5 * self.face_size as f32).floor() as isize)
                    .clamp(0, (self.face_size - 1) as isize) as usize;
                let yi = (((vi + 1.0) * 0.5 * self.face_size as f32).floor() as isize)
                    .clamp(0, (self.face_size - 1) as isize) as usize;
                let mut under_sky_island = false;
                'sky_chk: for dy in -3isize..=3 {
                    for dx in -3isize..=3 {
                        for z in self.ring_z_floor()..self.depth {
                            if self.faces[fi].solid_at(xi as isize + dx, yi as isize + dy, z as isize) >= 100 {
                                under_sky_island = true;
                                break 'sky_chk;
                            }
                        }
                    }
                }
                let coast_bonus = if samp.surf_z > self.z_sea() as f32 + 0.3 { 2.2 } else { 0.6 };
                let lava_bonus = if samp.emissive > 0.3 { 1.5 } else { 0.0 };
                let sky_clear_penalty = if under_sky_island { -18.0 } else { 0.0 };
                let sun_bonus = if sun_dot > 0.20 {
                    9.5 + sun_dot * 4.0
                } else if sun_dot > -0.05 {
                    3.0 + sun_dot * 6.0
                } else {
                    sun_dot * 6.0
                };
                let score = neighbors * 1.8
                    + if has_enemy { 3.5 } else { 0.0 }
                    + sun_bonus
                    + coast_bonus
                    + lava_bonus
                    + sky_clear_penalty;
                if score > best_score {
                    best_score = score;
                    let c = centroid / w_sum;
                    best_pos = c.normalize_or(dir_i) * self.sample_surface_radius(c);
                    best_fwd = fwdi;
                }
            }
            if best_pos.length_squared() > 1.0 {
                return (best_pos, best_fwd);
            }
        }
        // Fallback: scenic sunlit equatorial/mid-latitude coast
        let site_dir = (sun * 0.72 + Vec3::new(0.25, 0.38, 0.65)).normalize();
        let r = self.sample_surface_radius(site_dir);
        let tang = Vec3::Y.cross(site_dir).normalize_or(Vec3::X);
        (site_dir * r, tang)
    }
}

// =============================================================================
// Lightweight 3D Oriented Geometry Primitives (Spherical World Space)
// =============================================================================

fn push_oriented_box(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    right: Vec3,
    up: Vec3,
    fwd: Vec3,
    size: Vec3,
    color: [f32; 4],
    flags: [f32; 4],
) {
    let hx = right * (size.x * 0.5);
    let hy = up * (size.y * 0.5);
    let hz = fwd * (size.z * 0.5);

    let faces: [(Vec3, [Vec3; 4]); 6] = [
        // +Y (top)
        (
            up,
            [
                center - hx + hy - hz,
                center - hx + hy + hz,
                center + hx + hy + hz,
                center + hx + hy - hz,
            ],
        ),
        // -Y (bottom)
        (
            -up,
            [
                center - hx - hy + hz,
                center - hx - hy - hz,
                center + hx - hy - hz,
                center + hx - hy + hz,
            ],
        ),
        // +X (right)
        (
            right,
            [
                center + hx - hy + hz,
                center + hx - hy - hz,
                center + hx + hy - hz,
                center + hx + hy + hz,
            ],
        ),
        // -X (left)
        (
            -right,
            [
                center - hx - hy - hz,
                center - hx - hy + hz,
                center - hx + hy + hz,
                center - hx + hy - hz,
            ],
        ),
        // +Z (front)
        (
            fwd,
            [
                center - hx - hy + hz,
                center + hx - hy + hz,
                center + hx + hy + hz,
                center - hx + hy + hz,
            ],
        ),
        // -Z (back)
        (
            -fwd,
            [
                center + hx - hy - hz,
                center - hx - hy - hz,
                center - hx + hy - hz,
                center + hx + hy - hz,
            ],
        ),
    ];

    for (n, corners) in faces {
        let base = verts.len() as u32;
        let mut norm = (corners[1] - corners[0]).cross(corners[2] - corners[0]).normalize_or(n);
        if norm.dot(n) < 0.0 {
            norm = -norm;
            for p in [corners[0], corners[3], corners[2], corners[1]] {
                verts.push(PlanetVertex {
                    pos: p.to_array(),
                    normal: norm.to_array(),
                    color,
                    flags,
                });
            }
        } else {
            for p in corners {
                verts.push(PlanetVertex {
                    pos: p.to_array(),
                    normal: norm.to_array(),
                    color,
                    flags,
                });
            }
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

fn push_oriented_pyramid(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    base_center: Vec3,
    right: Vec3,
    up: Vec3,
    fwd: Vec3,
    half_w: f32,
    height: f32,
    color: [f32; 4],
    flags: [f32; 4],
) {
    let apex = base_center + up * height;
    let c0 = base_center - right * half_w - fwd * half_w;
    let c1 = base_center + right * half_w - fwd * half_w;
    let c2 = base_center + right * half_w + fwd * half_w;
    let c3 = base_center - right * half_w + fwd * half_w;

    let tris = [(c0, c1, apex), (c1, c2, apex), (c2, c3, apex), (c3, c0, apex)];
    for (a, b, c) in tris {
        let mid = (a + b + c) / 3.0;
        let out_hint = (mid - base_center).normalize_or(up);
        let mut n = (b - a).cross(c - a).normalize_or(out_hint);
        let base = verts.len() as u32;
        if n.dot(out_hint) < 0.0 {
            n = -n;
            for p in [a, c, b] {
                verts.push(PlanetVertex { pos: p.to_array(), normal: n.to_array(), color, flags });
            }
        } else {
            for p in [a, b, c] {
                verts.push(PlanetVertex { pos: p.to_array(), normal: n.to_array(), color, flags });
            }
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

fn push_spherical_ring(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    center: Vec3,
    up: Vec3,
    right: Vec3,
    fwd: Vec3,
    r_inner: f32,
    r_outer: f32,
    segs: usize,
    color: [f32; 4],
    flags: [f32; 4],
) {
    let base = verts.len() as u32;
    let r_rad = center.length();
    for i in 0..=segs {
        let ang = (i as f32 / segs as f32) * std::f32::consts::TAU;
        let dir_t = right * ang.cos() + fwd * ang.sin();
        let p_in = (center + dir_t * r_inner).normalize_or(up) * r_rad;
        let p_out = (center + dir_t * r_outer).normalize_or(up) * r_rad;
        verts.push(PlanetVertex {
            pos: p_in.to_array(),
            normal: up.to_array(),
            color,
            flags,
        });
        verts.push(PlanetVertex {
            pos: p_out.to_array(),
            normal: up.to_array(),
            color,
            flags,
        });
    }
    for i in 0..segs {
        let a = base + (i * 2) as u32;
        let b = a + 1;
        let c = a + 2;
        let d = a + 3;
        indices.extend_from_slice(&[a, b, d, a, d, c]);
    }
}

fn push_radial_pillar(
    verts: &mut Vec<PlanetVertex>,
    indices: &mut Vec<u32>,
    base_pos: Vec3,
    up: Vec3,
    right: Vec3,
    fwd: Vec3,
    radius: f32,
    height: f32,
    color: [f32; 4],
    flags: [f32; 4],
) {
    let top_pos = base_pos + up * height;
    let top_col = [color[0], color[1], color[2], color[3] * 0.12];
    for axis in [right, fwd] {
        let base = verts.len() as u32;
        let n = up.cross(axis).normalize_or(right);
        verts.push(PlanetVertex {
            pos: (base_pos - axis * radius).to_array(),
            normal: n.to_array(),
            color,
            flags,
        });
        verts.push(PlanetVertex {
            pos: (base_pos + axis * radius).to_array(),
            normal: n.to_array(),
            color,
            flags,
        });
        verts.push(PlanetVertex {
            pos: (top_pos + axis * (radius * 0.35)).to_array(),
            normal: n.to_array(),
            color: top_col,
            flags,
        });
        verts.push(PlanetVertex {
            pos: (top_pos - axis * (radius * 0.35)).to_array(),
            normal: n.to_array(),
            color: top_col,
            flags,
        });
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{PlanetCamMode, PlanetCamera};
    use frostmarsh_proto::{
        decode_brick3d_delta_voxels, decode_subbrick3d_lz4, encode_brick3d_delta,
        encode_subbrick3d_lz4,
    };

    #[test]
    fn test_wave31b_planet_world_mesh_and_6dof_camera() {
        let face_size = 32usize;
        let depth = 32usize;
        let r_core = 48.0f32;
        let mut planet = PlanetWorld::new(face_size, depth, r_core);

        // Feed full PlanetFaceBrick3D sub-bricks across all 6 faces (terrain + ocean + volcanic rift + Sky-Ring!)
        let sub_d = 16usize;
        let sub_bytes = CHUNK_SIZE * CHUNK_SIZE * sub_d * 4;
        for face in 0..FACE_COUNT as u8 {
            for bz in 0..2u16 {
                for by in 0..2u16 {
                    for bx in 0..2u16 {
                        let mut dm = vec![0u8; sub_bytes];
                        let mut at = vec![0u8; sub_bytes];
                        for dz in 0..sub_d {
                            let gz = (bz as usize) * 16 + dz;
                            for dy in 0..CHUNK_SIZE {
                                for dx in 0..CHUNK_SIZE {
                                    let o = ((dz * CHUNK_SIZE + dy) * CHUNK_SIZE + dx) * 4;
                                    if gz <= 14 {
                                        // Solid continent / seafloor
                                        dm[o] = 240;
                                        dm[o + 3] = 200;
                                        at[o] = 48;
                                        at[o + 1] = 110;
                                        at[o + 2] = 56;
                                        // Volcanic rift hotspot on face 0
                                        if face == 0 && dy == 8 && gz == 14 {
                                            at[o] = 255;
                                            at[o + 1] = 120;
                                            at[o + 2] = 24;
                                            at[o + 3] = 230;
                                        }
                                    } else if gz == 15 && bx == 1 {
                                        // Curved sapphire ocean
                                        dm[o + 1] = 220;
                                        dm[o + 3] = 20;
                                        at[o] = 30;
                                        at[o + 1] = 112;
                                        at[o + 2] = 168;
                                    } else if gz >= 23 && gz <= 25 && bx == 0 && by == 0 && dx >= 5 && dx <= 10 && dy >= 5 && dy <= 10 {
                                        // Orbital Sky-Ring Floating Island + Buoyant Crystal Core underside
                                        dm[o] = 235;
                                        dm[o + 3] = if gz == 23 { 50 } else { 190 };
                                        if gz == 23 {
                                            at[o] = 80;
                                            at[o + 1] = 220;
                                            at[o + 2] = 255;
                                        } else {
                                            at[o] = 42;
                                            at[o + 1] = 125;
                                            at[o + 2] = 58;
                                        }
                                    }
                                }
                            }
                        }
                        let (lz4_dm, lz4_at) = encode_subbrick3d_lz4(&dm, &at);
                        let (dec_dm, dec_at) = decode_subbrick3d_lz4(&lz4_dm, &lz4_at).unwrap();
                        planet.apply_update(PlanetBrickUpdate::FullSubBrick {
                            face,
                            bx,
                            by,
                            bz,
                            total_depth: depth as u16,
                            sub_depth: sub_d as u8,
                            tick: 1,
                            density_mat: dec_dm.clone(),
                            albedo_thermal: dec_at.clone(),
                        });

                        // Also feed a PlanetFaceBrick3DDelta on each face's top sub-brick
                        if bx == 0 && by == 0 && bz == 1 {
                            let mut dm2 = dec_dm.clone();
                            let mut at2 = dec_at.clone();
                            // Add volcanic steam plume voxel via sparse delta
                            let idx = (2 * CHUNK_SIZE + 4) * CHUNK_SIZE + 4;
                            dm2[idx * 4 + 2] = 180;
                            at2[idx * 4] = 220;
                            at2[idx * 4 + 1] = 232;
                            at2[idx * 4 + 2] = 248;
                            let lz4_delta = encode_brick3d_delta(&dec_dm, &dec_at, &dm2, &at2, 4)
                                .expect("delta encoded");
                            let patches = decode_brick3d_delta_voxels(&lz4_delta).unwrap();
                            planet.apply_update(PlanetBrickUpdate::DeltaSubBrick {
                                face,
                                bx,
                                by,
                                bz,
                                total_depth: depth as u16,
                                sub_depth: sub_d as u8,
                                tick: 2,
                                patches,
                            });
                        }
                    }
                }
            }
        }

        assert_eq!(planet.populated_face_count(), 6);
        assert_eq!(planet.received_brick_count(), 6 * 2 * 2 * 2);

        let rebuilt = planet.rebuild_dirty_faces(6);
        assert_eq!(rebuilt, 6);
        assert!(!planet.combined_terrain_verts.is_empty());
        assert!(!planet.combined_terrain_indices.is_empty());
        assert!(!planet.combined_ocean_verts.is_empty());
        assert!(!planet.combined_ocean_indices.is_empty());
        assert!(!planet.combined_gas_verts.is_empty());

        for v in &planet.combined_terrain_verts {
            assert!(v.pos.iter().all(|c| c.is_finite()), "non-finite terrain pos");
            assert!(v.normal.iter().all(|c| c.is_finite()), "non-finite terrain normal");
            let nlen = Vec3::from_array(v.normal).length();
            assert!((nlen - 1.0).abs() < 0.05, "terrain normal must be unit length, got {}", nlen);
            let r = Vec3::from_array(v.pos).length();
            assert!(r >= r_core && r <= r_core + depth as f32 + 2.0, "vertex radius {} out of shell", r);
        }

        // Verify 6-DOF geodesic spherical camera preserves orthonormal UP = normalize(pos) across cube seams & poles!
        let mut cam = PlanetCamera::new(r_core, depth as f32);
        cam.mode = PlanetCamMode::SurfaceHorizon;
        let sun = Vec3::new(-0.78, 0.28, 0.56).normalize();
        // Walk hero smoothly across +Z -> +Y (North Pole) -> -Z cube seams
        for step in 0..60 {
            let theta = (step as f32 / 59.0) * std::f32::consts::PI;
            let dir = Vec3::new(0.0, theta.sin(), theta.cos()).normalize();
            let hero_pos = dir * (r_core + 16.0);
            let tang_fwd = Vec3::new(0.0, theta.cos(), -theta.sin()).normalize();
            for _ in 0..8 {
                cam.update_follow(hero_pos, tang_fwd, sun, r_core, depth as f32, 0.05);
            }
            let mats = cam.compute_matrices(1920.0, 1080.0);
            assert!(mats.eye.is_finite() && mats.target.is_finite() && mats.up.is_finite());
            assert!((mats.up.length() - 1.0).abs() < 1e-3, "camera UP must remain unit length");
            let radial = hero_pos.normalize();
            assert!(
                mats.up.dot(radial) > 0.85,
                "camera UP must track radial normal across seams/poles (dot={})",
                mats.up.dot(radial)
            );
            let view_dir = (mats.target - mats.eye).normalize();
            assert!(
                view_dir.cross(mats.up).length() > 0.15,
                "camera look vector and UP must stay non-degenerate at poles"
            );
        }

        // Verify live Metal/WGSL `planet_live.wgsl` pipeline compilation & headless planetary render!
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let meshes = crate::mesh_gen::StaticMeshBank::build(1);
        let surfels = crate::surfel_gen::SurfelProtoBank::build();
        if let Ok(mut renderer) =
            rt.block_on(crate::renderer::Approach4Renderer::new_headless(&meshes, &surfels))
        {
            renderer.upload_planet_world(&mut planet);
            let orb_cam = PlanetCamera::orbital_terminator_shot(sun, planet.r_outer());
            let orb_mats = orb_cam.compute_matrices(320.0, 180.0);
            let (rgba, _gpu_ms) = renderer.render_planet_to_rgba(
                320,
                180,
                &orb_mats,
                sun,
                orb_cam.exposure,
                1.0,
                &planet,
                None,
            );
            assert_eq!(rgba.len(), 320 * 180 * 4);
            let non_dark = rgba
                .chunks_exact(4)
                .filter(|px| px[0] > 18 || px[1] > 18 || px[2] > 18)
                .count();
            assert!(
                non_dark > 1000,
                "rendered curved planet frame must contain visible illuminated globe pixels, got {}",
                non_dark
            );
        }
    }
}
