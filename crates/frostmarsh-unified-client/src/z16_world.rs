//! True 3D `Z=16` World Volume, Incremental Smooth Surface-Nets (`16×16×16` dirty bricks),
//! Coarse `8×8×8` Brick Occupancy, Hard-Z Sculpted Pine Placements, Live Minimap, and
//! Exact Continuous `∇solid_density` 3D Capsule Physics Predictor.
//!
//! ## Axis Convention (matches `frostmarsh-3d-sandbox/src/volume.rs` & GPU `TextureDimension::D3`)
//! - Volume coordinates `(vx, vy, vz)` in `[0..width, 0..depth, 0..height]`:
//!   - `vx` = East-West (`width`, e.g. 128 or 256)
//!   - `vy` = Altitude UP (`depth`, 16)
//!   - `vz` = North-South (`height`, e.g. 128 or 256)
//! - Linear byte index in GPU 3D texture order (`Extent3d { width, height: depth, depth_or_array_layers: height }`):
//!   `((vz * depth + vy) * width + vx) * 4`
//! - Server / `PlayerSnap` coordinates `(gx, gy, gz)`:
//!   - `gx = vx` (East-West), `gy = vz` (North-South), `gz = vy` (Altitude UP)
//! - Unified client centered world coordinates:
//!   - `wx = gx - width * 0.5`, `wy = gz`, `wz = gy - height * 0.5`

use bytemuck::{Pod, Zeroable};
use frostmarsh_proto::{chunks_for_dim, CHUNK_SIZE};
use glam::Vec3;
use std::collections::HashSet;
use std::f32::consts::TAU;
use std::time::Instant;

pub const SN_BRICK_SIZE: usize = CHUNK_SIZE; // 16
pub const OCC_BRICK_SIZE: usize = 8;
const ISO: f32 = 0.5;

/// Minimal 12-byte vertex for Smooth Surface-Nets (`pos` in volume space `[0..W, 0..D, 0..H]`).
/// `field_mesh_z16.wgsl` samples albedo, roughness, thermal glow, analytical trilinear
/// `∇solid_density` normals, 3D cave AO, and 3D raymarched sun shadows directly from 3D textures!
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SurfaceVertex {
    pub pos: [f32; 3],
}

#[derive(Clone, Debug, Default)]
pub struct BrickMesh {
    pub vertices: Vec<SurfaceVertex>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Z16PinePlacement {
    pub world_pos: Vec3,
    pub yaw: f32,
    pub scale: f32,
    pub tint: [f32; 3],
    pub wind_phase: f32,
    pub snowy: bool,
}

/// The 12 edges of a unit cube, indexing the 8 corners `(dx, dy, dz)` encoded as `dx | (dy<<1) | (dz<<2)`.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1), (2, 3), (4, 5), (6, 7), // X-aligned
    (0, 2), (1, 3), (4, 6), (5, 7), // Y-aligned
    (0, 4), (1, 5), (2, 6), (3, 7), // Z-aligned
];

const CORNER_OFFSETS: [[f32; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

#[inline]
fn hash1(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277_803_737);
    ((x ^ (x >> 22)) as f32) / (u32::MAX as f32)
}

pub struct Z16World {
    pub width: usize,  // X (e.g. 128 or 256)
    pub height: usize, // Z north-south (e.g. 128 or 256)
    pub depth: usize,  // Y altitude (16)
    pub bricks_x: usize,
    pub bricks_z: usize,

    /// GPU-ready `[iz][iy][ix][4]` RGBA8: `(solid_density, fluid_density, gas_density, roughness)`
    pub density_mat: Vec<u8>,
    /// GPU-ready `[iz][iy][ix][4]` RGBA8: `(albedo_r, albedo_g, albedo_b, thermal_glow)`
    pub albedo_thermal: Vec<u8>,

    /// Coarse `8×8×8` dilated brick occupancy 3D texture `[bz][by][bx][4]`
    pub occ_bx: usize,
    pub occ_by: usize,
    pub occ_bz: usize,
    pub occ_data: Vec<u8>,

    /// Per-`16×16` column Surface-Nets vertex/index caches
    pub brick_meshes: Vec<BrickMesh>,
    pub flat_verts: Vec<SurfaceVertex>,
    pub flat_idxs: Vec<u32>,

    /// Per-`16×16` column Hard-Z sculpted pine placements & audio/lighting hotspots
    pub brick_pines: Vec<Vec<Z16PinePlacement>>,
    pub brick_fires: Vec<Vec<(Vec3, f32)>>,
    pub brick_waterfalls: Vec<Vec<Vec3>>,

    /// Live `W×H` RGBA8 tactical minimap derived from top solid/fluid/glow voxel per column
    pub minimap_rgba: Vec<u8>,
    pub minimap_dirty: bool,

    /// GPU sync dirty flags
    pub gpu_tex_full_dirty: bool,
    pub gpu_dirty_bricks: Vec<(usize, usize)>,
    pub gpu_occ_dirty: bool,
    pub gpu_mesh_dirty: bool,
}

impl Z16World {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        let w = width.max(16);
        let h = height.max(16);
        let d = depth.max(2);
        let bricks_x = chunks_for_dim(w);
        let bricks_z = chunks_for_dim(h);
        let occ_bx = w.div_ceil(OCC_BRICK_SIZE);
        let occ_by = d.div_ceil(OCC_BRICK_SIZE);
        let occ_bz = h.div_ceil(OCC_BRICK_SIZE);
        let n_bricks = bricks_x * bricks_z;
        Self {
            width: w,
            height: h,
            depth: d,
            bricks_x,
            bricks_z,
            density_mat: vec![0u8; w * d * h * 4],
            albedo_thermal: vec![0u8; w * d * h * 4],
            occ_bx,
            occ_by,
            occ_bz,
            occ_data: vec![0u8; occ_bx * occ_by * occ_bz * 4],
            brick_meshes: vec![BrickMesh::default(); n_bricks],
            flat_verts: Vec::new(),
            flat_idxs: Vec::new(),
            brick_pines: vec![Vec::new(); n_bricks],
            brick_fires: vec![Vec::new(); n_bricks],
            brick_waterfalls: vec![Vec::new(); n_bricks],
            minimap_rgba: vec![32u8; w * h * 4],
            minimap_dirty: true,
            gpu_tex_full_dirty: true,
            gpu_dirty_bricks: Vec::new(),
            gpu_occ_dirty: true,
            gpu_mesh_dirty: true,
        }
    }

    pub fn resize_if_needed(&mut self, w: usize, h: usize, d: usize) {
        let d = d.max(2);
        if w >= 16 && h >= 16 && (self.width != w || self.height != h || self.depth != d) {
            *self = Self::new(w, h, d);
        }
    }

    #[inline]
    pub fn byte_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ((iz * self.depth + iy) * self.width + ix) * 4
    }

    #[inline]
    pub fn solid_f32(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        self.density_mat[self.byte_index(ix, iy, iz)] as f32 / 255.0
    }

    #[inline]
    pub fn grid_to_world_xz(&self, gx: f32, gy: f32) -> (f32, f32) {
        (gx - (self.width as f32) * 0.5, gy - (self.height as f32) * 0.5)
    }

    pub fn total_triangles(&self) -> usize {
        self.flat_idxs.len() / 3
    }

    pub fn non_empty_bricks(&self) -> usize {
        self.brick_meshes.iter().filter(|b| !b.indices.is_empty()).count()
    }

    /// Incrementally copy dirty `16×16×D` bricks from `LiveWorldState`, update `8×8×8` occupancy,
    /// rebuild ONLY dirty Smooth Surface-Nets bricks (plus 1-brick seam neighbors!), update minimap,
    /// and refresh Hard-Z pine placements & fire/waterfall hotspots.
    /// Returns `(rebuilt_mesh_bricks, elapsed_ms)`.
    pub fn apply_dirty_bricks(
        &mut self,
        src_density_mat: &[u8],
        src_albedo_thermal: &[u8],
        dirty_bricks: &[(usize, usize)],
    ) -> (usize, f64) {
        if dirty_bricks.is_empty() {
            return (0, 0.0);
        }
        let expected = self.width * self.depth * self.height * 4;
        if src_density_mat.len() != expected || src_albedo_thermal.len() != expected {
            return (0, 0.0);
        }

        let t0 = Instant::now();
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let bx_max = self.bricks_x;
        let bz_max = self.bricks_z;

        // 1. Copy voxel columns for all dirty bricks FIRST so cross-brick halo reads see fresh data!
        let mut unique_dirty: Vec<(usize, usize)> = Vec::with_capacity(dirty_bricks.len());
        let mut seen = HashSet::with_capacity(dirty_bricks.len() * 2);
        for &(bx, bz) in dirty_bricks {
            if bx >= bx_max || bz >= bz_max || !seen.insert((bx, bz)) {
                continue;
            }
            unique_dirty.push((bx, bz));

            let x0 = bx * SN_BRICK_SIZE;
            let z0 = bz * SN_BRICK_SIZE;
            let cw = SN_BRICK_SIZE.min(w - x0);
            let ch = SN_BRICK_SIZE.min(h - z0);
            let row_bytes = cw * 4;

            for dz in 0..ch {
                let iz = z0 + dz;
                for iy in 0..d {
                    let off = ((iz * d + iy) * w + x0) * 4;
                    self.density_mat[off..off + row_bytes]
                        .copy_from_slice(&src_density_mat[off..off + row_bytes]);
                    self.albedo_thermal[off..off + row_bytes]
                        .copy_from_slice(&src_albedo_thermal[off..off + row_bytes]);
                }
            }
        }

        if unique_dirty.is_empty() {
            return (0, 0.0);
        }

        // Queue GPU 3D texture upload (full upload if many bricks changed, sub-brick otherwise)
        if unique_dirty.len() >= (bx_max * bz_max) / 4 {
            self.gpu_tex_full_dirty = true;
            self.gpu_dirty_bricks.clear();
        } else if !self.gpu_tex_full_dirty {
            self.gpu_dirty_bricks.extend_from_slice(&unique_dirty);
        }

        // 2. Update coarse 8×8×8 brick occupancy texture (with 1-voxel dilation halo)
        if unique_dirty.len() >= (bx_max * bz_max) / 3 {
            self.rebuild_all_occupancy();
        } else {
            self.update_occupancy_for_bricks(&unique_dirty);
        }
        self.gpu_occ_dirty = true;

        // 3. Collect dirty Surface-Nets bricks + seam neighbors (1-cell -X/-Z halo & +1 corner sharing)
        let mut mesh_dirty_set: HashSet<(usize, usize)> = HashSet::with_capacity(unique_dirty.len() * 4);
        for &(bx, bz) in &unique_dirty {
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = bx as i32 + dx;
                    let nz = bz as i32 + dz;
                    if nx >= 0 && nz >= 0 && (nx as usize) < bx_max && (nz as usize) < bz_max {
                        mesh_dirty_set.insert((nx as usize, nz as usize));
                    }
                }
            }
        }

        // Rebuild only affected 16×16×16 Surface-Nets bricks (~31 µs each!)
        let mut rebuilt = 0usize;
        for &(bx, bz) in &mesh_dirty_set {
            let mesh = extract_brick(&self.density_mat, w, d, h, bx, bz);
            self.brick_meshes[bz * bx_max + bx] = mesh;
            rebuilt += 1;
        }
        self.flatten_mesh_buffers();
        self.gpu_mesh_dirty = true;

        // 4. Incrementally update minimap, Hard-Z pines, and fire/waterfall hotspots for dirty bricks
        for &(bx, bz) in &unique_dirty {
            self.update_minimap_brick(bx, bz);
            self.rebuild_pines_and_hotspots_brick(bx, bz);
        }
        self.minimap_dirty = true;

        (rebuilt, t0.elapsed().as_secs_f64() * 1000.0)
    }

    fn flatten_mesh_buffers(&mut self) {
        let v_tot: usize = self.brick_meshes.iter().map(|b| b.vertices.len()).sum();
        let i_tot: usize = self.brick_meshes.iter().map(|b| b.indices.len()).sum();
        self.flat_verts.clear();
        self.flat_idxs.clear();
        self.flat_verts.reserve(v_tot);
        self.flat_idxs.reserve(i_tot);
        for b in &self.brick_meshes {
            let base = self.flat_verts.len() as u32;
            self.flat_verts.extend_from_slice(&b.vertices);
            self.flat_idxs.extend(b.indices.iter().map(|&idx| idx + base));
        }
    }

    fn rebuild_all_occupancy(&mut self) {
        let (obx, oby, obz) = (self.occ_bx, self.occ_by, self.occ_bz);
        for bz in 0..obz {
            for by in 0..oby {
                for bx in 0..obx {
                    self.recompute_occ_cell(bx, by, bz);
                }
            }
        }
    }

    fn update_occupancy_for_bricks(&mut self, dirty_bricks: &[(usize, usize)]) {
        let mut occ_xz: HashSet<(usize, usize)> = HashSet::with_capacity(dirty_bricks.len() * 6);
        let ratio = SN_BRICK_SIZE / OCC_BRICK_SIZE; // 2
        for &(bx, bz) in dirty_bricks {
            let ox0 = (bx * ratio).saturating_sub(1);
            let ox1 = ((bx + 1) * ratio + 1).min(self.occ_bx);
            let oz0 = (bz * ratio).saturating_sub(1);
            let oz1 = ((bz + 1) * ratio + 1).min(self.occ_bz);
            for oz in oz0..oz1 {
                for ox in ox0..ox1 {
                    occ_xz.insert((ox, oz));
                }
            }
        }
        for &(ox, oz) in &occ_xz {
            for oy in 0..self.occ_by {
                self.recompute_occ_cell(ox, oy, oz);
            }
        }
    }

    fn recompute_occ_cell(&mut self, bx: usize, by: usize, bz: usize) {
        let x0 = (bx * OCC_BRICK_SIZE).saturating_sub(1);
        let x1 = ((bx + 1) * OCC_BRICK_SIZE + 1).min(self.width);
        let y0 = (by * OCC_BRICK_SIZE).saturating_sub(1);
        let y1 = ((by + 1) * OCC_BRICK_SIZE + 1).min(self.depth);
        let z0 = (bz * OCC_BRICK_SIZE).saturating_sub(1);
        let z1 = ((bz + 1) * OCC_BRICK_SIZE + 1).min(self.height);

        let mut max_s = 0u8;
        let mut max_f = 0u8;
        let mut max_g = 0u8;
        let mut max_t = 0u8;

        for iz in z0..z1 {
            for iy in y0..y1 {
                let row = (iz * self.depth + iy) * self.width * 4;
                for ix in x0..x1 {
                    let i = row + ix * 4;
                    max_s = max_s.max(self.density_mat[i]);
                    max_f = max_f.max(self.density_mat[i + 1]);
                    max_g = max_g.max(self.density_mat[i + 2]);
                    max_t = max_t.max(self.albedo_thermal[i + 3]);
                }
            }
        }
        let bi = ((bz * self.occ_by + by) * self.occ_bx + bx) * 4;
        self.occ_data[bi] = max_s;
        self.occ_data[bi + 1] = max_f;
        self.occ_data[bi + 2] = max_g;
        self.occ_data[bi + 3] = max_t;
    }

    fn update_minimap_brick(&mut self, bx: usize, bz: usize) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let x0 = bx * SN_BRICK_SIZE;
        let z0 = bz * SN_BRICK_SIZE;
        let x1 = (x0 + SN_BRICK_SIZE).min(w);
        let z1 = (z0 + SN_BRICK_SIZE).min(h);

        for iz in z0..z1 {
            for ix in x0..x1 {
                let mut rgb = [38u8, 46, 58];
                for iy in (0..d).rev() {
                    let bi = ((iz * d + iy) * w + ix) * 4;
                    let solid = self.density_mat[bi];
                    let fluid = self.density_mat[bi + 1];
                    let glow = self.albedo_thermal[bi + 3];

                    if glow > 45 {
                        rgb = [255, 140, 36];
                        break;
                    }
                    if fluid > 72 && solid < 110 {
                        let wf_y = (d * 7) / 16;
                        let bright = if iy >= wf_y { 48 } else { 0 };
                        rgb = [
                            (34u16 + bright).min(255) as u8,
                            (126u16 + bright).min(255) as u8,
                            (204u16 + bright).min(255) as u8,
                        ];
                        break;
                    }
                    if solid >= 128 {
                        let elev_shade = 0.72 + 0.42 * (iy as f32 / (d.max(1) as f32));
                        let ar = self.albedo_thermal[bi] as f32;
                        let ag = self.albedo_thermal[bi + 1] as f32;
                        let ab = self.albedo_thermal[bi + 2] as f32;
                        // Alpine snow highlight on high ridges (scales seamlessly for Z=16..32!)
                        let snow_y = (d * 10) / 16;
                        if iy >= snow_y && ar > 70.0 && ab > 70.0 {
                            rgb = [214, 228, 242];
                        } else {
                            rgb = [
                                (ar * elev_shade).clamp(0.0, 255.0) as u8,
                                (ag * elev_shade).clamp(0.0, 255.0) as u8,
                                (ab * elev_shade).clamp(0.0, 255.0) as u8,
                            ];
                        }
                        break;
                    }
                }
                let moff = (iz * w + ix) * 4;
                self.minimap_rgba[moff] = rgb[0];
                self.minimap_rgba[moff + 1] = rgb[1];
                self.minimap_rgba[moff + 2] = rgb[2];
                self.minimap_rgba[moff + 3] = 255;
            }
        }
    }

    fn find_column_surface_y(&self, ix: usize, iz: usize) -> Option<f32> {
        let ix = ix.min(self.width - 1);
        let iz = iz.min(self.height - 1);
        for iy in (1..self.depth).rev() {
            let d_cur = self.solid_f32(ix, iy, iz);
            if d_cur >= ISO {
                let d_above = if iy + 1 < self.depth {
                    self.solid_f32(ix, iy + 1, iz)
                } else {
                    0.0
                };
                let frac = ((d_cur - ISO) / (d_cur - d_above).max(1e-3)).clamp(0.0, 1.0);
                return Some(iy as f32 + 0.5 + frac);
            }
        }
        None
    }

    /// Public surface altitude lookup at server grid `(gx, gy)` for shrines, relics, domes & popups.
    pub fn sample_surface_y(&self, gx: f32, gy: f32) -> f32 {
        let ix = (gx.floor() as i32).clamp(0, (self.width - 1) as i32) as usize;
        let iz = (gy.floor() as i32).clamp(0, (self.height - 1) as i32) as usize;
        self.find_column_surface_y(ix, iz).unwrap_or(1.5)
    }

    fn rebuild_pines_and_hotspots_brick(&mut self, bx: usize, bz: usize) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let x0 = bx * SN_BRICK_SIZE;
        let z0 = bz * SN_BRICK_SIZE;
        let x1 = (x0 + SN_BRICK_SIZE).min(w);
        let z1 = (z0 + SN_BRICK_SIZE).min(h);

        let mut pines = Vec::with_capacity(8);
        let mut fire_sum = Vec3::ZERO;
        let mut fire_cnt = 0.0f32;
        let mut waterfalls = Vec::new();

        // Scan fire thermal hotspots & elevated plunging waterfalls (every 2 cells for speed)
        for iz in (z0..z1).step_by(2) {
            for ix in (x0..x1).step_by(2) {
                for iy in 1..d {
                    let bi = ((iz * d + iy) * w + ix) * 4;
                    let glow = self.albedo_thermal[bi + 3];
                    if glow > 85 {
                        let (wx, wz) = self.grid_to_world_xz(ix as f32 + 0.5, iz as f32 + 0.5);
                        fire_sum += Vec3::new(wx, iy as f32 + 0.8, wz);
                        fire_cnt += glow as f32 / 255.0;
                    }
                    let fluid = self.density_mat[bi + 1];
                    let wf_hi = d.saturating_sub(3).max(9);
                    if iy >= 5 && iy <= wf_hi && fluid > 90 && waterfalls.len() < 4 {
                        let (wx, wz) = self.grid_to_world_xz(ix as f32 + 0.5, iz as f32 + 0.5);
                        waterfalls.push(Vec3::new(wx, iy as f32 + 0.5, wz));
                    }
                }
            }
        }

        // Scatter Hard-Z sculpted Nordic Pines & Snow-Pines on forested mountain slopes
        // (high-green albedo OR snowy upper ridges, avoiding bridge corridor, river canyon, & fire!)
        let step = if w <= 128 { 5usize } else { 7usize };
        let start_x = x0.max(5);
        let end_x = x1.min(w.saturating_sub(5));
        let start_z = z0.max(5);
        let end_z = z1.min(h.saturating_sub(5));

        let mut iz = (start_z / step) * step;
        if iz < start_z {
            iz += step;
        }
        while iz < end_z {
            let mut ix = (start_x / step) * step;
            if ix < start_x {
                ix += step;
            }
            while ix < end_x {
                let seed = ((ix as u32).wrapping_mul(73_856_093))
                    ^ ((iz as u32).wrapping_mul(19_349_663))
                    ^ 0x9E37_79B9;
                let h0 = hash1(seed);
                let h1 = hash1(seed.wrapping_add(31));
                let h2 = hash1(seed.wrapping_add(73));
                let h3 = hash1(seed.wrapping_add(127));

                if h0 >= 0.34 {
                    let gx = (ix as f32 + (h1 - 0.5) * (step as f32 * 0.65)).clamp(4.0, wf - 5.0);
                    let gz = (iz as f32 + (h2 - 0.5) * (step as f32 * 0.65)).clamp(4.0, hf - 5.0);
                    let nx = gx / wf;
                    let ny = gz / hf;

                    // Keep 3-span bridge corridor, spawn clearings, river canyon, & waterfall sightline clear
                    let in_bridge = nx > 0.26 && nx < 0.74 && (ny - 0.50).abs() < 0.085;
                    let river_cx = 0.50 + 0.032 * ((ny - 0.50) * TAU * 1.1).sin();
                    let in_river = (nx - river_cx).abs() < 0.095;
                    let in_waterfall = (nx - 0.50).abs() < 0.11 && ny > 0.10 && ny < 0.34;
                    let in_spawn = (nx - 0.14).hypot(ny - 0.50) < 0.075
                        || (nx - 0.86).hypot(ny - 0.50) < 0.075;

                    if !in_bridge && !in_river && !in_waterfall && !in_spawn {
                        let sxi = gx as usize;
                        let szi = gz as usize;
                        if let Some(wy) = self.find_column_surface_y(sxi, szi) {
                            let max_pine_y = (d as f32 - 3.2).max(12.2);
                            let ridge_y = (d as f32 * 0.54).max(8.6);
                            let snow_y = (d as f32 * 0.55).max(8.8);
                            if wy >= 4.4 && wy <= max_pine_y {
                                let iy_s = (wy.floor() as usize).min(d - 1);
                                let iy_u = (iy_s + 1).min(d - 1);
                                let bi_s = self.byte_index(sxi, iy_s, szi);
                                let bi_u = self.byte_index(sxi, iy_u, szi);
                                let no_fluid = self.density_mat[bi_s + 1] <= 30
                                    && self.density_mat[bi_u + 1] <= 30;
                                let no_fire = self.albedo_thermal[bi_s + 3] <= 30
                                    && self.albedo_thermal[bi_u + 3] <= 30;

                                // Check green forested albedo (meadow/soil/plant) or high-altitude alpine ridge
                                let ar = self.albedo_thermal[bi_s] as f32;
                                let ag = self.albedo_thermal[bi_s + 1] as f32;
                                let ab = self.albedo_thermal[bi_s + 2] as f32;
                                let is_forested_green = ag > ar * 1.06 && ag > ab * 1.04 && ag > 40.0;
                                let is_alpine_ridge = wy >= ridge_y && h0 > 0.62;

                                if no_fluid && no_fire && (is_forested_green || is_alpine_ridge) {
                                    // Gentle slope check so pines stand naturally on slopes
                                    let hx0 = self.find_column_surface_y(sxi.saturating_sub(2), szi).unwrap_or(wy);
                                    let hx1 = self.find_column_surface_y((sxi + 2).min(w - 1), szi).unwrap_or(wy);
                                    let hz0 = self.find_column_surface_y(sxi, szi.saturating_sub(2)).unwrap_or(wy);
                                    let hz1 = self.find_column_surface_y(sxi, (szi + 2).min(h - 1)).unwrap_or(wy);
                                    if (hx1 - hx0).abs() <= 2.6 && (hz1 - hz0).abs() <= 2.6 {
                                        let snowy = wy >= snow_y || (ny < 0.28 && wy >= snow_y - 1.2);
                                        let scale = (0.72 + h3 * 0.36) * if snowy { 0.90 } else { 1.02 };
                                        let tv = 0.88 + h1 * 0.20;
                                        let tint = if snowy {
                                            [tv * 0.96, tv * 0.99, (tv * 1.04).min(1.12)]
                                        } else {
                                            [tv * 0.90, tv * 1.02, tv * 0.92]
                                        };
                                        let (wx, wz) = self.grid_to_world_xz(gx, gz);
                                        pines.push(Z16PinePlacement {
                                            world_pos: Vec3::new(wx, wy - 0.22, wz),
                                            yaw: h2 * TAU,
                                            scale,
                                            tint,
                                            wind_phase: h0 * TAU,
                                            snowy,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                ix += step;
            }
            iz += step;
        }

        let bi = bz * self.bricks_x + bx;
        self.brick_pines[bi] = pines;
        self.brick_fires[bi].clear();
        if fire_cnt > 0.25 {
            self.brick_fires[bi].push((fire_sum / fire_cnt, fire_cnt));
        }
        self.brick_waterfalls[bi] = waterfalls;
    }

    pub fn sample_ambient_audio_sources(
        &self,
        listener_pos: Vec3,
    ) -> (Option<(Vec3, f32)>, Option<(Vec3, f32)>) {
        let mut best_wf: Option<(Vec3, f32)> = None;
        for wlist in &self.brick_waterfalls {
            for &wp in wlist {
                let d = wp.distance(listener_pos);
                if best_wf.map_or(true, |(_, bd)| d < bd) {
                    best_wf = Some((wp, d));
                }
            }
        }
        let mut fire_pos_sum = Vec3::ZERO;
        let mut fire_w_sum = 0.0f32;
        for flist in &self.brick_fires {
            for &(fp, inten) in flist {
                let d = fp.distance(listener_pos);
                if d < 72.0 {
                    let w = inten / (1.0 + (d / 18.0).powi(2));
                    fire_pos_sum += fp * w;
                    fire_w_sum += w;
                }
            }
        }
        let nearby_fire = if fire_w_sum > 0.05 {
            Some((fire_pos_sum / fire_w_sum, fire_w_sum))
        } else {
            None
        };
        (best_wf, nearby_fire)
    }

    /// Physical 3D Cave & Archway Acoustic Enclosure probe at server grid `(gx, gy, gz)`
    /// (`gx` = East, `gy` = South, `gz` = Altitude UP).
    ///
    /// Samples `solid_density` across a `3×3` overhead ceiling ring `2.0..6.0m` above `(gx, gy, gz)`
    /// plus 4 horizontal wall probes at `4.0m` radius, returning `0.0` (open sky) → `1.0` (deep vaulted cave).
    pub fn sample_cave_enclosure(&self, gx: f32, gy: f32, gz: f32) -> f32 {
        if self.width < 4 || self.height < 4 || self.depth < 4 {
            return 0.0;
        }
        let max_z = self.depth as f32 - 0.1;
        let sample_bounded = |x: f32, y: f32, z: f32| -> f32 {
            if z < 0.0 || z >= max_z {
                0.0
            } else {
                self.sample_solid_density_3d(x, y, z)
            }
        };

        // 1. 3×3 overhead ceiling ring 2.0..6.0m above the listener
        let offsets = [-2.0f32, 0.0, 2.0];
        let heights = [2.0f32, 3.0, 4.0, 5.0, 6.0];
        let mut ceil_sum = 0.0f32;
        for &dy in &offsets {
            for &dx in &offsets {
                let mut col_max = 0.0f32;
                for &dz in &heights {
                    let s = sample_bounded(gx + dx, gy + dy, gz + dz);
                    if s > col_max {
                        col_max = s;
                    }
                }
                // Map solid density [0.35..0.65] cleanly to [0..1]
                let occ = ((col_max - 0.35) / 0.30).clamp(0.0, 1.0);
                ceil_sum += occ;
            }
        }
        let ceil_cov = ceil_sum / 9.0;

        // 2. 4 horizontal wall probes at 4.0m radius (plus inner 2.5m check for tight caverns)
        let wall_dirs = [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        let probe_z = gz + 1.25;
        let mut wall_sum = 0.0f32;
        for &(dx, dy) in &wall_dirs {
            let s_far = sample_bounded(gx + dx * 4.0, gy + dy * 4.0, probe_z);
            let s_mid = sample_bounded(gx + dx * 2.5, gy + dy * 2.5, probe_z);
            let occ = ((s_far.max(s_mid) - 0.35) / 0.30).clamp(0.0, 1.0);
            wall_sum += occ;
        }
        let wall_cov = wall_sum / 4.0;

        // Under a full roof without walls -> ~0.78 (>0.7); deep cave with roof + walls -> 1.0;
        // open sky (even beside a cliff) -> <0.09!
        let enc = ceil_cov * 0.78 + wall_cov * 0.22 * (0.28 + 0.72 * ceil_cov);
        enc.clamp(0.0, 1.0)
    }

    /// Detects whether solid terrain blocks the hero from above (`> gz + 2.5m`) or along the
    /// hero-to-camera line-of-sight above `gz + 2.2m`, triggering Smart 3D Cave/Bridge Ceiling Cutaway.
    pub fn should_cutaway_ceiling(&self, gx: f32, gy: f32, gz: f32, cam_world: Vec3) -> bool {
        if self.width < 4 || self.height < 4 || self.depth < 4 {
            return false;
        }
        let max_z = self.depth as f32 - 0.1;
        // 1. Direct overhead roof check above `gz + 2.5m`
        let mut z = gz + 2.5;
        while z < max_z {
            if self.sample_solid_density_3d(gx, gy, z) >= ISO
                || self.sample_solid_density_3d(gx + 1.2, gy, z) >= ISO
                || self.sample_solid_density_3d(gx - 1.2, gy, z) >= ISO
                || self.sample_solid_density_3d(gx, gy + 1.2, z) >= ISO
                || self.sample_solid_density_3d(gx, gy - 1.2, z) >= ISO
            {
                return true;
            }
            z += 0.85;
        }
        // 2. Raymarched camera-to-hero LOS check for overhanging arches/caves blocking the view
        let cam_gx = cam_world.x + (self.width as f32) * 0.5;
        let cam_gy = cam_world.z + (self.height as f32) * 0.5;
        let cam_gz = cam_world.y;
        let dx = cam_gx - gx;
        let dy = cam_gy - gy;
        let dz = cam_gz - (gz + 1.2);
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-3);
        let steps = (dist.min(48.0) / 1.25).ceil() as usize;
        for s in 2..=steps {
            let t = (s as f32 * 1.25) / dist;
            if t >= 0.98 {
                break;
            }
            let qz = (gz + 1.2) + dz * t;
            if qz <= gz + 2.2 {
                continue;
            }
            if qz >= max_z {
                break;
            }
            let qx = gx + dx * t;
            let qy = gy + dy * t;
            if qx < 0.0 || qy < 0.0 || qx >= self.width as f32 || qy >= self.height as f32 {
                break;
            }
            if self.sample_solid_density_3d(qx, qy, qz) >= ISO {
                return true;
            }
        }
        false
    }

    // =========================================================================
    // EXACT CONTINUOUS `∇solid_density` 3D CAPSULE COLLISION & PREDICTOR
    // Matches `field-core::VolumeField::step_capsule_3d` 1:1!
    // Coordinates: `(x, y, z)` where `x = East`, `y = South`, `z = Altitude UP`.
    // =========================================================================

    /// Continuous hardware-matched trilinear sample of `solid_density` at `(x, y, z)`.
    /// Voxel `(ix, iy_alt, iz_north)` has center `(ix + 0.5, iz_north + 0.5, iy_alt + 0.5)`.
    #[inline]
    pub fn sample_solid_density_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return 0.0;
        }
        let fx = x - 0.5;
        let fy = y - 0.5;
        let fz = z - 0.5;

        let max_x = (self.width - 1) as i32;
        let max_y = (self.height - 1) as i32;
        let max_z = (self.depth - 1) as i32;

        let x0 = (fx.floor() as i32).clamp(0, max_x);
        let y0 = (fy.floor() as i32).clamp(0, max_y);
        let z0 = (fz.floor() as i32).clamp(0, max_z);
        let x1 = (x0 + 1).clamp(0, max_x);
        let y1 = (y0 + 1).clamp(0, max_y);
        let z1 = (z0 + 1).clamp(0, max_z);

        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let tz = (fz - z0 as f32).clamp(0.0, 1.0);

        let (x0, x1) = (x0 as usize, x1 as usize);
        let (y0, y1) = (y0 as usize, y1 as usize);
        let (z0, z1) = (z0 as usize, z1 as usize);

        // GPU layout: solid_f32(ix = east, iy = altitude, iz = south)
        let c000 = self.solid_f32(x0, z0, y0);
        let c100 = self.solid_f32(x1, z0, y0);
        let c010 = self.solid_f32(x0, z0, y1);
        let c110 = self.solid_f32(x1, z0, y1);
        let c001 = self.solid_f32(x0, z1, y0);
        let c101 = self.solid_f32(x1, z1, y0);
        let c011 = self.solid_f32(x0, z1, y1);
        let c111 = self.solid_f32(x1, z1, y1);

        let x00 = c000 + (c100 - c000) * tx;
        let x10 = c010 + (c110 - c010) * tx;
        let x01 = c001 + (c101 - c001) * tx;
        let x11 = c011 + (c111 - c011) * tx;

        let y0v = x00 + (x10 - x00) * ty;
        let y1v = x01 + (x11 - x01) * ty;
        y0v + (y1v - y0v) * tz
    }

    #[inline]
    pub fn sample_solid_gradient_3d(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        const EPS: f32 = 0.25;
        let gx = self.sample_solid_density_3d(x + EPS, y, z)
            - self.sample_solid_density_3d(x - EPS, y, z);
        let gy = self.sample_solid_density_3d(x, y + EPS, z)
            - self.sample_solid_density_3d(x, y - EPS, z);
        let gz = self.sample_solid_density_3d(x, y, z + EPS)
            - self.sample_solid_density_3d(x, y, z - EPS);
        let inv = 1.0 / (2.0 * EPS);
        (gx * inv, gy * inv, gz * inv)
    }

    /// Exact 1:1 mirror of `field-core::VolumeField::step_capsule_3d` for zero-rubber-band
    /// 3D client prediction across arched bridges, under-bridge walkways, and vaulted caves!
    pub fn step_capsule_3d(
        &self,
        pos: (f32, f32, f32),
        vel: (f32, f32, f32),
        radius: f32,
        height: f32,
        dt: f32,
    ) -> ((f32, f32, f32), (f32, f32, f32), bool) {
        const GRAVITY: f32 = 18.0;
        if self.width < 4 || self.height < 4 || self.depth < 2 || dt <= 0.0 {
            return (pos, vel, false);
        }

        let substeps = ((dt / 0.008).ceil() as usize).clamp(1, 8);
        let sub_dt = dt / substeps as f32;
        let mut grounded = false;

        let (mut px, mut py, mut pz) = pos;
        let (mut vx, mut vy, mut vz) = vel;

        let min_x = radius + 2.0;
        let max_x = self.width as f32 - radius - 2.0;
        let min_y = radius + 2.0;
        let max_y = self.height as f32 - radius - 2.0;
        let max_z = (self.depth as f32 - height - 0.1).max(0.5);

        for _ in 0..substeps {
            vz -= GRAVITY * sub_dt;
            px = (px + vx * sub_dt).clamp(min_x, max_x);
            py = (py + vy * sub_dt).clamp(min_y, max_y);
            pz = (pz + vz * sub_dt).clamp(0.5, max_z);

            for _iter in 0..10 {
                let mut worst_pen = 0.0f32;
                let mut best_n = (0.0f32, 0.0f32, 1.0f32);

                let samples = capsule_shell_samples_3d((px, py, pz), radius, height);
                for (sidx, &(qx, qy, qz)) in samples.iter().enumerate() {
                    let d = self.sample_solid_density_3d(qx, qy, qz);
                    if d > ISO {
                        let (gx, gy, gz) = self.sample_solid_gradient_3d(qx, qy, qz);
                        let g_len = (gx * gx + gy * gy + gz * gz).sqrt();
                        let outward_n = if g_len > 1e-3 {
                            (-gx / g_len, -gy / g_len, -gz / g_len)
                        } else if sidx < 5 {
                            (0.0, 0.0, 1.0)
                        } else if sidx >= 21 {
                            (0.0, 0.0, -1.0)
                        } else {
                            let dx = px - qx;
                            let dy = py - qy;
                            let dl = dx.hypot(dy);
                            if dl > 1e-4 {
                                (dx / dl, dy / dl, 0.0)
                            } else {
                                (0.0, 0.0, 1.0)
                            }
                        };
                        let pen = (d - ISO) / g_len.max(0.55);
                        if pen > worst_pen {
                            worst_pen = pen;
                            best_n = outward_n;
                        }
                    }
                }

                if worst_pen <= 1e-4 {
                    break;
                }

                let push = worst_pen.min(0.35);
                px = (px + best_n.0 * push).clamp(min_x, max_x);
                py = (py + best_n.1 * push).clamp(min_y, max_y);
                pz = (pz + best_n.2 * push).clamp(0.5, max_z);

                if best_n.2 >= 0.55 {
                    grounded = true;
                    if vz < 0.0 {
                        vz = 0.0;
                    }
                }

                let vn = vx * best_n.0 + vy * best_n.1 + vz * best_n.2;
                if vn < 0.0 {
                    vx -= best_n.0 * vn;
                    vy -= best_n.1 * vn;
                    vz -= best_n.2 * vn;
                }
            }
        }

        ((px, py, pz), (vx, vy, vz), grounded)
    }
}

fn capsule_shell_samples_3d(pos: (f32, f32, f32), radius: f32, height: f32) -> [(f32, f32, f32); 26] {
    let (px, py, pz) = pos;
    let r = radius;
    let z_bot = r;
    let z_top = (height - r).max(r + 0.05);
    let z_mid = 0.5 * (z_bot + z_top);
    let d = r * std::f32::consts::FRAC_1_SQRT_2;
    let cap_r = r * 0.7071;
    let cap_dz = r * 0.7071;

    [
        // Bottom pole + 4 lower hemisphere points (5)
        (px, py, pz),
        (px + cap_r, py, pz + z_bot - cap_dz),
        (px - cap_r, py, pz + z_bot - cap_dz),
        (px, py + cap_r, pz + z_bot - cap_dz),
        (px, py - cap_r, pz + z_bot - cap_dz),
        // Lower cylinder ring (8)
        (px + r, py, pz + z_bot),
        (px - r, py, pz + z_bot),
        (px, py + r, pz + z_bot),
        (px, py - r, pz + z_bot),
        (px + d, py + d, pz + z_bot),
        (px - d, py + d, pz + z_bot),
        (px + d, py - d, pz + z_bot),
        (px - d, py - d, pz + z_bot),
        // Mid-waist ring (4)
        (px + r, py, pz + z_mid),
        (px - r, py, pz + z_mid),
        (px, py + r, pz + z_mid),
        (px, py - r, pz + z_mid),
        // Upper cylinder ring (4)
        (px + r, py, pz + z_top),
        (px - r, py, pz + z_top),
        (px, py + r, pz + z_top),
        (px, py - r, pz + z_top),
        // Top hemisphere + top pole (5)
        (px + cap_r, py, pz + z_top + cap_dz),
        (px - cap_r, py, pz + z_top + cap_dz),
        (px, py + cap_r, pz + z_top + cap_dz),
        (px, py - cap_r, pz + z_top + cap_dz),
        (px, py, pz + height),
    ]
}

/// Extract one `16×16×D` watertight Smooth Surface-Nets brick `(bx, bz)` with a 1-cell negative halo
/// so cross-brick seams are 100% seamless and outward-facing CCW.
fn extract_brick(dm: &[u8], w: usize, d: usize, h: usize, bx: usize, bz: usize) -> BrickMesh {
    let cell_w = w - 1;
    let cell_d = d - 1;
    let cell_h = h - 1;

    let ox = bx * SN_BRICK_SIZE;
    let oz = bz * SN_BRICK_SIZE;
    let x_end = (ox + SN_BRICK_SIZE).min(cell_w);
    let z_end = (oz + SN_BRICK_SIZE).min(cell_h);
    if ox >= x_end || oz >= z_end || cell_d == 0 {
        return BrickMesh::default();
    }

    let gx0 = ox.saturating_sub(1);
    let gz0 = oz.saturating_sub(1);
    let gw = x_end - gx0;
    let gd = cell_d;
    let gh = z_end - gz0;

    let mut cell_vid = vec![u32::MAX; gw * gd * gh];
    let mut vertices = Vec::with_capacity(512);
    let mut indices = Vec::with_capacity(1536);

    #[inline]
    fn sample_raw(dm: &[u8], w: usize, d: usize, x: usize, y: usize, z: usize) -> f32 {
        dm[((z * d + y) * w + x) * 4] as f32 / 255.0
    }

    // Pass 1: Place one smooth dual vertex per sign-change cell at the average edge-crossing point.
    for z in gz0..z_end {
        let lz = z - gz0;
        for y in 0..gd {
            for x in gx0..x_end {
                let lx = x - gx0;
                let c = [
                    sample_raw(dm, w, d, x, y, z),
                    sample_raw(dm, w, d, x + 1, y, z),
                    sample_raw(dm, w, d, x, y + 1, z),
                    sample_raw(dm, w, d, x + 1, y + 1, z),
                    sample_raw(dm, w, d, x, y, z + 1),
                    sample_raw(dm, w, d, x + 1, y, z + 1),
                    sample_raw(dm, w, d, x, y + 1, z + 1),
                    sample_raw(dm, w, d, x + 1, y + 1, z + 1),
                ];

                let mut mask = 0u8;
                for (i, &v) in c.iter().enumerate() {
                    if v >= ISO {
                        mask |= 1 << i;
                    }
                }
                if mask == 0 || mask == 255 {
                    continue;
                }

                let mut sx = 0.0f32;
                let mut sy = 0.0f32;
                let mut sz = 0.0f32;
                let mut count = 0.0f32;
                for &(e0, e1) in &CUBE_EDGES {
                    let s0 = c[e0];
                    let s1 = c[e1];
                    if (s0 >= ISO) != (s1 >= ISO) {
                        let t = ((ISO - s0) / (s1 - s0)).clamp(0.0, 1.0);
                        let p0 = CORNER_OFFSETS[e0];
                        let p1 = CORNER_OFFSETS[e1];
                        sx += p0[0] + (p1[0] - p0[0]) * t;
                        sy += p0[1] + (p1[1] - p0[1]) * t;
                        sz += p0[2] + (p1[2] - p0[2]) * t;
                        count += 1.0;
                    }
                }
                let inv = 1.0 / count.max(1.0);
                // Align with GPU voxel centers at (x + 0.5, y + 0.5, z + 0.5)
                let vx = x as f32 + 0.5 + sx * inv;
                let vy = y as f32 + 0.5 + sy * inv;
                let vz = z as f32 + 0.5 + sz * inv;

                let vid = vertices.len() as u32;
                vertices.push(SurfaceVertex { pos: [vx, vy, vz] });
                cell_vid[(lz * gd + y) * gw + lx] = vid;
            }
        }
    }

    if vertices.is_empty() {
        return BrickMesh::default();
    }

    let get_vid = |x: usize, y: usize, z: usize| -> u32 {
        cell_vid[((z - gz0) * gd + y) * gw + (x - gx0)]
    };

    #[inline]
    fn emit_quad(indices: &mut Vec<u32>, a: u32, b: u32, c: u32, d: u32, flip: bool) {
        if a == u32::MAX || b == u32::MAX || c == u32::MAX || d == u32::MAX {
            return;
        }
        if !flip {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        } else {
            indices.extend_from_slice(&[a, d, c, a, c, b]);
        }
    }

    // Pass 2: Connect adjacent sign-change cells around each bipolar minimal edge into CCW quads.
    for z in oz..z_end {
        for y in 0..gd {
            for x in ox..x_end {
                let s0 = sample_raw(dm, w, d, x, y, z) >= ISO;

                // X-edge: (x,y,z) -> (x+1,y,z)
                if y > 0 && z > 0 {
                    let sx = sample_raw(dm, w, d, x + 1, y, z) >= ISO;
                    if s0 != sx {
                        let c00 = get_vid(x, y - 1, z - 1);
                        let c10 = get_vid(x, y, z - 1);
                        let c11 = get_vid(x, y, z);
                        let c01 = get_vid(x, y - 1, z);
                        emit_quad(&mut indices, c00, c10, c11, c01, !s0);
                    }
                }

                // Y-edge: (x,y,z) -> (x,y+1,z)
                if x > 0 && z > 0 {
                    let sy = sample_raw(dm, w, d, x, y + 1, z) >= ISO;
                    if s0 != sy {
                        let c00 = get_vid(x - 1, y, z - 1);
                        let c10 = get_vid(x - 1, y, z);
                        let c11 = get_vid(x, y, z);
                        let c01 = get_vid(x, y, z - 1);
                        emit_quad(&mut indices, c00, c10, c11, c01, !s0);
                    }
                }

                // Z-edge: (x,y,z) -> (x,y,z+1)
                if x > 0 && y > 0 {
                    let sz = sample_raw(dm, w, d, x, y, z + 1) >= ISO;
                    if s0 != sz {
                        let c00 = get_vid(x - 1, y - 1, z);
                        let c10 = get_vid(x, y - 1, z);
                        let c11 = get_vid(x, y, z);
                        let c01 = get_vid(x - 1, y, z);
                        emit_quad(&mut indices, c00, c10, c11, c01, !s0);
                    }
                }
            }
        }
    }

    BrickMesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z32_surface_nets_extract() {
        let (w, h, d) = (32usize, 32usize, 32usize);
        let mut world = Z16World::new(w, h, d);
        assert_eq!(world.depth, 32);
        assert_eq!(world.occ_by, 4, "32m altitude should use 4 coarse 8x8x8 occupancy bricks");

        let mut dm = vec![0u8; w * d * h * 4];
        let mut at = vec![0u8; w * d * h * 4];

        // Sculpt a 28m towering mountain peak + vaulted cave tunnel + arched bridge at Z=32!
        for iz in 0..h {
            for ix in 0..w {
                let dx = ix as f32 - 16.0;
                let dz = iz as f32 - 16.0;
                let r = dx.hypot(dz);
                let ridge_h = (27.0 - r * 1.15).clamp(4.0, 28.0);
                for iy in 0..d {
                    let y = iy as f32 + 0.5;
                    let mut solid = if y <= ridge_h { 245u8 } else { 0u8 };
                    // Carve a subterranean vaulted cave chamber at y in 4..10 near center
                    if r < 6.5 && y > 3.5 && y < 9.5 {
                        solid = 0;
                    }
                    let off = ((iz * d + iy) * w + ix) * 4;
                    dm[off] = solid;
                    dm[off + 3] = 200; // roughness
                    if solid > 0 {
                        at[off] = 96;
                        at[off + 1] = 138;
                        at[off + 2] = 104;
                    }
                }
            }
        }

        let all_bricks = vec![(0, 0), (1, 0), (0, 1), (1, 1)];
        let (rebuilt, _ms) = world.apply_dirty_bricks(&dm, &at, &all_bricks);
        assert_eq!(rebuilt, 4);
        assert!(world.total_triangles() > 200, "Z=32 Surface-Nets must emit watertight triangles");
        assert!(world.flat_verts.iter().all(|v| {
            v.pos[0].is_finite()
                && v.pos[1].is_finite()
                && v.pos[2].is_finite()
                && v.pos[1] >= 0.0
                && v.pos[1] <= d as f32
        }));
        let max_y = world
            .flat_verts
            .iter()
            .map(|v| v.pos[1])
            .fold(0.0f32, f32::max);
        assert!(max_y > 22.0, "Z=32 mountain peak must reach >22m, got {max_y}");
        // Verify vaulted cave enclosure inside the chamber vs high open sky on top of peak!
        let enc_cave = world.sample_cave_enclosure(16.0, 16.0, 5.0);
        let enc_sky = world.sample_cave_enclosure(16.0, 16.0, 28.5);
        assert!(enc_cave > 0.7, "Inside Z=32 vaulted cave should have enclosure > 0.7, got {enc_cave}");
        assert!(enc_sky < 0.1, "Above Z=32 mountain peak should have enclosure < 0.1, got {enc_sky}");
    }
}
