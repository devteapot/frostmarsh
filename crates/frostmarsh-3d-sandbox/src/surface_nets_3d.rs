//! Smooth 3D Surface-Nets Isosurface Mesh Extractor (`solid_density == 0.5`)
//! with `16×16×16` Incremental Dirty-Brick Rebuild Support.
//!
//! Aligns cell corners `(cx, cy, cz)` with world-space voxel centers `(cx + 0.5, cy + 0.5, cz + 0.5)`
//! so the extracted mesh matches GPU hardware trilinear raymarching (`world_pos / dims`) with
//! zero half-voxel shift!

use crate::volume::WorldVolume3D;
use bytemuck::{Pod, Zeroable};
use std::time::Instant;

pub const SN_BRICK_SIZE: usize = 16;
const ISO: f32 = 0.5;

/// Minimal vertex format: world position only (`12 bytes`).
/// `field_mesh.wgsl` samples albedo, roughness, thermal glow, analytical trilinear `∇solid_density`
/// normals, 3D-texture cave AO, and 3D-texture raymarched shadows directly from the 3D textures!
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

pub struct SurfaceNetsWorld {
    pub bricks_x: usize,
    pub bricks_z: usize,
    pub bricks: Vec<BrickMesh>,
}

/// The 12 edges of a unit cube, indexing the 8 corners `(dx, dy, dz)` encoded as `dx | (dy<<1) | (dz<<2)`.
const CUBE_EDGES: [(usize, usize); 12] = [
    // X-aligned edges
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    // Y-aligned edges
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    // Z-aligned edges
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
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

impl SurfaceNetsWorld {
    /// Extract the full 3D Surface-Nets mesh across all `16×16×16` bricks.
    /// Returns `(mesh_world, build_ms)`.
    pub fn build_all(volume: &WorldVolume3D) -> (Self, f64) {
        let t0 = Instant::now();
        let bricks_x = volume.width.div_ceil(SN_BRICK_SIZE);
        let bricks_z = volume.height.div_ceil(SN_BRICK_SIZE);
        let mut bricks = Vec::with_capacity(bricks_x * bricks_z);
        for bz in 0..bricks_z {
            for bx in 0..bricks_x {
                bricks.push(extract_brick(volume, bx, bz));
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        (
            Self {
                bricks_x,
                bricks_z,
                bricks,
            },
            ms,
        )
    }

    /// Incrementally rebuild a single `16×16×16` dirty brick `(bx, bz)` in place.
    /// Returns rebuild time in milliseconds.
    pub fn rebuild_brick(&mut self, volume: &WorldVolume3D, bx: usize, bz: usize) -> f64 {
        let t0 = Instant::now();
        let mesh = extract_brick(volume, bx, bz);
        self.bricks[bz * self.bricks_x + bx] = mesh;
        t0.elapsed().as_secs_f64() * 1000.0
    }

    pub fn total_vertices(&self) -> usize {
        self.bricks.iter().map(|b| b.vertices.len()).sum()
    }

    pub fn total_triangles(&self) -> usize {
        self.bricks.iter().map(|b| b.indices.len() / 3).sum()
    }

    /// Flatten all bricks into contiguous GPU vertex & index buffers.
    pub fn flatten_buffers(&self) -> (Vec<SurfaceVertex>, Vec<u32>) {
        let v_tot = self.total_vertices();
        let i_tot: usize = self.bricks.iter().map(|b| b.indices.len()).sum();
        let mut verts = Vec::with_capacity(v_tot);
        let mut idxs = Vec::with_capacity(i_tot);
        for b in &self.bricks {
            let base = verts.len() as u32;
            verts.extend_from_slice(&b.vertices);
            idxs.extend(b.indices.iter().map(|&idx| idx + base));
        }
        (verts, idxs)
    }
}

/// Extract one `16×16×16` brick `(bx, bz)` with a 1-cell negative halo so seams across bricks
/// are 100% watertight and outward-facing CCW.
pub fn extract_brick(volume: &WorldVolume3D, bx: usize, bz: usize) -> BrickMesh {
    let cell_w = volume.width - 1;
    let cell_d = volume.depth - 1;
    let cell_h = volume.height - 1;

    let ox = bx * SN_BRICK_SIZE;
    let oz = bz * SN_BRICK_SIZE;
    let x_end = (ox + SN_BRICK_SIZE).min(cell_w);
    let z_end = (oz + SN_BRICK_SIZE).min(cell_h);
    if ox >= x_end || oz >= z_end {
        return BrickMesh::default();
    }

    // Local grid includes -1 halo on X and Z (and full Y = 0..cell_d) so every quad owned by
    // cell (x, y, z) in [ox..x_end, 0..cell_d, oz..z_end] has its 4 neighboring dual vertices.
    let gx0 = ox.saturating_sub(1);
    let gz0 = oz.saturating_sub(1);
    let gw = x_end - gx0;
    let gd = cell_d;
    let gh = z_end - gz0;

    let mut cell_vid = vec![u32::MAX; gw * gd * gh];
    let mut vertices = Vec::with_capacity(512);
    let mut indices = Vec::with_capacity(1536);

    let w = volume.width;
    let d = volume.depth;
    let dm = &volume.density_mat;

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

                // X-edge: (x,y,z) -> (x+1,y,z); surrounded by (x, y-1..y, z-1..z)
                if y > 0 && z > 0 {
                    let sx = sample_raw(dm, w, d, x + 1, y, z) >= ISO;
                    if s0 != sx {
                        let c00 = get_vid(x, y - 1, z - 1);
                        let c10 = get_vid(x, y, z - 1);
                        let c11 = get_vid(x, y, z);
                        let c01 = get_vid(x, y - 1, z);
                        // s0=solid, sx=air -> outward +X -> (c00, c10, c11, c01)
                        emit_quad(&mut indices, c00, c10, c11, c01, !s0);
                    }
                }

                // Y-edge: (x,y,z) -> (x,y+1,z); surrounded by (x-1..x, y, z-1..z)
                if x > 0 && z > 0 {
                    let sy = sample_raw(dm, w, d, x, y + 1, z) >= ISO;
                    if s0 != sy {
                        let c00 = get_vid(x - 1, y, z - 1);
                        let c10 = get_vid(x - 1, y, z);
                        let c11 = get_vid(x, y, z);
                        let c01 = get_vid(x, y, z - 1);
                        // s0=solid, sy=air -> outward +Y -> (c00, c10, c11, c01)
                        emit_quad(&mut indices, c00, c10, c11, c01, !s0);
                    }
                }

                // Z-edge: (x,y,z) -> (x,y,z+1); surrounded by (x-1..x, y-1..y, z)
                if x > 0 && y > 0 {
                    let sz = sample_raw(dm, w, d, x, y, z + 1) >= ISO;
                    if s0 != sz {
                        let c00 = get_vid(x - 1, y - 1, z);
                        let c10 = get_vid(x, y - 1, z);
                        let c11 = get_vid(x, y, z);
                        let c01 = get_vid(x - 1, y, z);
                        // s0=solid, sz=air -> outward +Z -> (c00, c10, c11, c01)
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
    fn test_surface_nets_extract_and_dirty_brick() {
        let vol = WorldVolume3D::generate_showcase();
        let (mut sn, build_ms) = SurfaceNetsWorld::build_all(&vol);
        assert!(sn.total_triangles() > 50_000, "Expected rich 3D world mesh");
        assert!(build_ms < 2000.0);
        // Dirty rebuild of a bridge brick (bx=8, bz=8 covers X=128..144, Z=128..144)
        let dirty_ms = sn.rebuild_brick(&vol, 8, 8);
        assert!(dirty_ms < 50.0, "Single 16x16x16 brick rebuild too slow: {dirty_ms} ms");
    }
}
