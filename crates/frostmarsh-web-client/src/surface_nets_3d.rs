//! Smooth 3D Surface-Nets Isosurface Mesh Extractor (`solid_density == 0.5`)
//! with `16×16×16` Brick Partitioning for WASM & WebGPU.

use crate::volume::WorldVolume3D;
use bytemuck::{Pod, Zeroable};

pub const SN_BRICK_SIZE: usize = 16;
const ISO: f32 = 0.5;

/// Minimal vertex format: world position only (`12 bytes`).
/// `field_mesh.wgsl` samples albedo, roughness, thermal glow, analytical trilinear `∇solid_density`
/// normals, 3D-texture cave AO, and 3D-texture raymarched shadows directly from the 3D textures.
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
    pub bricks: Vec<BrickMesh>,
}

const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
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
    pub fn build_all(volume: &WorldVolume3D) -> Self {
        let bricks_x = volume.width.div_ceil(SN_BRICK_SIZE);
        let bricks_z = volume.height.div_ceil(SN_BRICK_SIZE);
        let mut bricks = Vec::with_capacity(bricks_x * bricks_z);
        for bz in 0..bricks_z {
            for bx in 0..bricks_x {
                bricks.push(extract_brick(volume, bx, bz));
            }
        }
        Self { bricks }
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

    for z in oz..z_end {
        for y in 0..gd {
            for x in ox..x_end {
                let s0 = sample_raw(dm, w, d, x, y, z) >= ISO;

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
