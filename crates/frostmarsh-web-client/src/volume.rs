//! True 3D `256×256×D` (`D = 16..=32`) Isotropic (`1m × 1m × 1m`) World Volume Format,
//! Procedural Alpine/Bridge/Cave/Waterfall/Wildfire Scene Generator,
//! `FM3D` Binary Parser (in-memory for WASM/HTTP), Incremental `16×16×D` Brick Patching,
//! and `8×8×8` Sparse Brick Occupancy Builder.

use glam::Vec3;

pub const DEFAULT_WIDTH: usize = 256;
pub const DEFAULT_HEIGHT: usize = 256;
pub const DEFAULT_DEPTH: usize = 16;
pub const MIN_SHOWCASE_DEPTH: usize = 16;
pub const MAX_SHOWCASE_DEPTH: usize = 32;
pub const BRICK_SIZE: usize = 8;
pub const STREAM_BRICK_XY: usize = 16;

/// Sub-region descriptor returned by [`WorldVolume3D::apply_brick3d_raw`] for incremental GPU uploads.
#[derive(Clone, Copy, Debug)]
pub struct BrickPatchResult {
    pub bx: usize,
    pub by: usize,
    pub x0: usize,
    pub z0: usize,
    pub cw: usize,
    pub ch: usize,
    pub depth: usize,
    pub resized_depth: bool,
}

/// Affected coarse `8×8×8` occupancy sub-region for incremental GPU texture upload.
#[derive(Clone, Copy, Debug)]
pub struct OccupancySubregion {
    pub x0: usize,
    pub z0: usize,
    pub cw: usize,
    pub ch: usize,
}

/// World volume storing two `RGBA8Unorm` 3D textures in GPU-ready `(X=width, Y=depth, Z=height)` order.
pub struct WorldVolume3D {
    pub width: usize,  // X (256)
    pub depth: usize,  // Y altitude (16..=32+)
    pub height: usize, // Z (256)
    /// `[Z][Y][X][4]` -> R: solid_density, G: fluid_density, B: gas_optical_density, A: roughness_wetness
    pub density_mat: Vec<u8>,
    /// `[Z][Y][X][4]` -> RGB: albedo, A: blackbody_temp_glow
    pub albedo_thermal: Vec<u8>,
}

/// Coarse `8×8×8` brick occupancy 3D texture (`W/8 × ceil(D/8) × H/8`).
pub struct BrickOccupancy3D {
    pub bx_count: usize,
    pub by_count: usize,
    pub bz_count: usize,
    /// `[bz][by][bx][4]` -> R: max_solid, G: max_fluid, B: max_gas, A: max_glow
    pub data: Vec<u8>,
    pub non_empty_bricks: usize,
    pub total_bricks: usize,
}

#[inline]
fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(y.wrapping_mul(668265263))
        .wrapping_add(z.wrapping_mul(2147483647));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h ^ (h >> 16)) as u32 as f32) / (u32::MAX as f32)
}

fn value_noise3(p: Vec3) -> f32 {
    let ix = p.x.floor() as i32;
    let iy = p.y.floor() as i32;
    let iz = p.z.floor() as i32;
    let fx = p.x - ix as f32;
    let fy = p.y - iy as f32;
    let fz = p.z - iz as f32;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let c000 = hash3(ix, iy, iz);
    let c100 = hash3(ix + 1, iy, iz);
    let c010 = hash3(ix, iy + 1, iz);
    let c110 = hash3(ix + 1, iy + 1, iz);
    let c001 = hash3(ix, iy, iz + 1);
    let c101 = hash3(ix + 1, iy, iz + 1);
    let c011 = hash3(ix, iy + 1, iz + 1);
    let c111 = hash3(ix + 1, iy + 1, iz + 1);

    let x00 = c000 + (c100 - c000) * ux;
    let x10 = c010 + (c110 - c010) * ux;
    let x01 = c001 + (c101 - c001) * ux;
    let x11 = c011 + (c111 - c011) * ux;
    let y0 = x00 + (x10 - x00) * uy;
    let y1 = x01 + (x11 - x01) * uy;
    y0 + (y1 - y0) * uz
}

fn fbm2(x: f32, z: f32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..4 {
        sum += amp * (value_noise3(Vec3::new(x * freq, 0.37 * freq, z * freq)) * 2.0 - 1.0);
        freq *= 2.07;
        amp *= 0.48;
    }
    sum
}

fn fbm3(p: Vec3) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut freq = 1.0;
    for _ in 0..3 {
        sum += amp * (value_noise3(p * freq) * 2.0 - 1.0);
        freq *= 2.13;
        amp *= 0.5;
    }
    sum
}

#[inline]
fn sdf_to_density(sdf: f32, band: f32) -> f32 {
    let t = (0.5 - sdf / (2.0 * band)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn dist_to_segment(p: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let len2 = ab.length_squared().max(1e-6);
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    let proj = a + ab * t;
    ((p - proj).length(), t)
}

#[inline]
pub fn river_center_x(z: f32) -> f32 {
    129.5 + 8.5 * ((z - 128.0) * 0.028).sin()
}

#[inline]
pub fn bridge_deck_top_y(x: f32) -> f32 {
    let u = ((x - 130.0) / 30.0).clamp(-1.0, 1.0);
    9.05 - 0.55 * u * u
}

pub fn cave_centerline_points() -> [(Vec3, f32, f32); 6] {
    [
        (Vec3::new(103.5, 4.25, 150.5), 5.6, 6.0),
        (Vec3::new(97.0, 4.25, 152.5), 5.2, 5.9),
        (Vec3::new(89.5, 4.25, 155.0), 5.4, 5.8),
        (Vec3::new(81.5, 4.25, 159.5), 6.2, 6.2),
        (Vec3::new(76.0, 4.25, 167.5), 5.2, 5.6),
        (Vec3::new(74.0, 4.25, 179.0), 4.6, 5.2),
    ]
}

impl WorldVolume3D {
    pub fn new_empty(width: usize, depth: usize, height: usize) -> Self {
        let len = width * depth * height * 4;
        Self {
            width,
            depth,
            height,
            density_mat: vec![0u8; len],
            albedo_thermal: vec![0u8; len],
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

    /// Hardware-matched trilinear sample of `solid_density` in world coordinates `(x, y, z)`.
    #[inline]
    pub fn sample_solid(&self, p: Vec3) -> f32 {
        let fx = p.x - 0.5;
        let fy = p.y - 0.5;
        let fz = p.z - 0.5;

        let max_x = (self.width - 1) as i32;
        let max_y = (self.depth - 1) as i32;
        let max_z = (self.height - 1) as i32;

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

        let c000 = self.solid_f32(x0, y0, z0);
        let c100 = self.solid_f32(x1, y0, z0);
        let c010 = self.solid_f32(x0, y1, z0);
        let c110 = self.solid_f32(x1, y1, z0);
        let c001 = self.solid_f32(x0, y0, z1);
        let c101 = self.solid_f32(x1, y0, z1);
        let c011 = self.solid_f32(x0, y1, z1);
        let c111 = self.solid_f32(x1, y1, z1);

        let x00 = c000 + (c100 - c000) * tx;
        let x10 = c010 + (c110 - c010) * tx;
        let x01 = c001 + (c101 - c001) * tx;
        let x11 = c011 + (c111 - c011) * tx;

        let y0v = x00 + (x10 - x00) * ty;
        let y1v = x01 + (x11 - x01) * ty;
        y0v + (y1v - y0v) * tz
    }

    /// Analytical 3D central-difference gradient `∇solid_density(p)`.
    #[inline]
    pub fn sample_solid_grad(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 0.25;
        let gx = self.sample_solid(p + Vec3::X * EPS) - self.sample_solid(p - Vec3::X * EPS);
        let gy = self.sample_solid(p + Vec3::Y * EPS) - self.sample_solid(p - Vec3::Y * EPS);
        let gz = self.sample_solid(p + Vec3::Z * EPS) - self.sample_solid(p - Vec3::Z * EPS);
        Vec3::new(gx, gy, gz) / (2.0 * EPS)
    }

    /// Parse an in-memory `FM3D` binary buffer fetched over HTTP in the browser:
    /// `magic = b"FM3D"`, `u32 width, height, depth` (LE),
    /// `density_mat [D*H*W*4] u8` (Z-major `[alt][north][east][rgba]`),
    /// `albedo_thermal [D*H*W*4] u8` (Z-major `[alt][north][east][rgba]`).
    pub fn from_fm3d_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 16 {
            return Err(format!("FM3D buffer too short: {} bytes", bytes.len()));
        }
        if &bytes[0..4] != b"FM3D" {
            return Err(format!("Invalid FM3D magic: {:?}", &bytes[0..4]));
        }
        let width = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let height = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let depth = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
        if width == 0 || height == 0 || depth == 0 || width > 512 || height > 512 || depth > 128 {
            return Err(format!("Unsupported FM3D dimensions: {width}x{height}x{depth}"));
        }

        let n = width
            .checked_mul(height)
            .and_then(|v| v.checked_mul(depth))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| "FM3D dimensions overflow".to_string())?;
        let expected = 16 + n * 2;
        if bytes.len() < expected {
            return Err(format!(
                "Truncated FM3D payload: got {} bytes, expected {}",
                bytes.len(),
                expected
            ));
        }

        let raw_dm = &bytes[16..16 + n];
        let raw_at = &bytes[16 + n..16 + n * 2];

        let mut vol = Self::new_empty(width, depth, height);
        // Transpose Z-major [alt=z][north=y][east=x] -> GPU [iz=north][iy=alt][ix=east]
        for alt in 0..depth {
            for north in 0..height {
                let src_row = (alt * height + north) * width * 4;
                let dst_row = (north * depth + alt) * width * 4;
                vol.density_mat[dst_row..dst_row + width * 4]
                    .copy_from_slice(&raw_dm[src_row..src_row + width * 4]);
                vol.albedo_thermal[dst_row..dst_row + width * 4]
                    .copy_from_slice(&raw_at[src_row..src_row + width * 4]);
            }
        }
        Ok(vol)
    }

    /// Build the coarse `8×8×8` occupancy brick 3D texture (`32×2×32`) with a 1-voxel halo.
    pub fn build_brick_occupancy(&self) -> BrickOccupancy3D {
        let bx_count = self.width.div_ceil(BRICK_SIZE);
        let by_count = self.depth.div_ceil(BRICK_SIZE);
        let bz_count = self.height.div_ceil(BRICK_SIZE);
        let mut data = vec![0u8; bx_count * by_count * bz_count * 4];
        let mut non_empty = 0usize;

        for bz in 0..bz_count {
            let z0 = (bz * BRICK_SIZE).saturating_sub(1);
            let z1 = ((bz + 1) * BRICK_SIZE + 1).min(self.height);
            for by in 0..by_count {
                let y0 = (by * BRICK_SIZE).saturating_sub(1);
                let y1 = ((by + 1) * BRICK_SIZE + 1).min(self.depth);
                for bx in 0..bx_count {
                    let x0 = (bx * BRICK_SIZE).saturating_sub(1);
                    let x1 = ((bx + 1) * BRICK_SIZE + 1).min(self.width);

                    let mut max_s = 0u8;
                    let mut max_f = 0u8;
                    let mut max_g = 0u8;
                    let mut max_t = 0u8;

                    for iz in z0..z1 {
                        for iy in y0..y1 {
                            let row = ((iz * self.depth + iy) * self.width) * 4;
                            for ix in x0..x1 {
                                let i = row + ix * 4;
                                max_s = max_s.max(self.density_mat[i]);
                                max_f = max_f.max(self.density_mat[i + 1]);
                                max_g = max_g.max(self.density_mat[i + 2]);
                                max_t = max_t.max(self.albedo_thermal[i + 3]);
                            }
                        }
                    }

                    let bi = ((bz * by_count + by) * bx_count + bx) * 4;
                    data[bi] = max_s;
                    data[bi + 1] = max_f;
                    data[bi + 2] = max_g;
                    data[bi + 3] = max_t;
                    if max_s > 2 || max_f > 2 || max_g > 2 {
                        non_empty += 1;
                    }
                }
            }
        }

        BrickOccupancy3D {
            bx_count,
            by_count,
            bz_count,
            data,
            non_empty_bricks: non_empty,
            total_bricks: bx_count * by_count * bz_count,
        }
    }

    /// Incrementally patch one live `16×16×D` (`D = 16..=32`, up to `128`) brick `(bx, by)`
    /// from raw Z-major `[alt][dy][dx][4]` RGBA8 `density_rgba` and `albedo_rgba` slices.
    ///
    /// Automatically resizes the volume altitude if `brick_depth != self.depth` (preserving
    /// overlapping altitude layers) so live `Z=16..32` upgrades work seamlessly!
    pub fn apply_brick3d_raw(
        &mut self,
        bx: usize,
        by: usize,
        brick_depth: usize,
        density_rgba: &[u8],
        albedo_rgba: &[u8],
    ) -> Result<BrickPatchResult, String> {
        if !(2..=128).contains(&brick_depth) {
            return Err(format!("Unsupported brick depth: {brick_depth} (expected 2..=128)"));
        }
        let x0 = bx.saturating_mul(STREAM_BRICK_XY);
        let z0 = by.saturating_mul(STREAM_BRICK_XY);
        if x0 >= self.width || z0 >= self.height {
            return Err(format!(
                "Brick ({bx}, {by}) out of bounds for {}x{} volume",
                self.width, self.height
            ));
        }

        let cw = STREAM_BRICK_XY.min(self.width - x0);
        let ch = STREAM_BRICK_XY.min(self.height - z0);
        let full_len = brick_depth * STREAM_BRICK_XY * STREAM_BRICK_XY * 4;
        let clipped_len = brick_depth * ch * cw * 4;

        let (src_w, src_h) = if density_rgba.len() == full_len && albedo_rgba.len() == full_len {
            (STREAM_BRICK_XY, STREAM_BRICK_XY)
        } else if density_rgba.len() == clipped_len && albedo_rgba.len() == clipped_len {
            (cw, ch)
        } else {
            return Err(format!(
                "Brick ({bx},{by}) payload size mismatch: got den={} alb={}, expected {} (full) or {} (clipped)",
                density_rgba.len(),
                albedo_rgba.len(),
                full_len,
                clipped_len
            ));
        };

        let mut resized_depth = false;
        if brick_depth != self.depth {
            let old_d = self.depth;
            let copy_d = old_d.min(brick_depth);
            let mut new_dm = vec![0u8; self.width * brick_depth * self.height * 4];
            let mut new_at = vec![0u8; self.width * brick_depth * self.height * 4];
            let row_bytes = self.width * 4;
            for iz in 0..self.height {
                for iy in 0..copy_d {
                    let src_off = (iz * old_d + iy) * row_bytes;
                    let dst_off = (iz * brick_depth + iy) * row_bytes;
                    new_dm[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&self.density_mat[src_off..src_off + row_bytes]);
                    new_at[dst_off..dst_off + row_bytes]
                        .copy_from_slice(&self.albedo_thermal[src_off..src_off + row_bytes]);
                }
            }
            self.depth = brick_depth;
            self.density_mat = new_dm;
            self.albedo_thermal = new_at;
            resized_depth = true;
        }

        let d = self.depth;
        let w = self.width;
        let copy_row = cw * 4;
        for alt in 0..d {
            for dy in 0..ch {
                let iz = z0 + dy;
                let src_off = ((alt * src_h + dy) * src_w) * 4;
                let dst_off = ((iz * d + alt) * w + x0) * 4;
                self.density_mat[dst_off..dst_off + copy_row]
                    .copy_from_slice(&density_rgba[src_off..src_off + copy_row]);
                self.albedo_thermal[dst_off..dst_off + copy_row]
                    .copy_from_slice(&albedo_rgba[src_off..src_off + copy_row]);
            }
        }

        Ok(BrickPatchResult {
            bx,
            by,
            x0,
            z0,
            cw,
            ch,
            depth: d,
            resized_depth,
        })
    }

    /// Pack a contiguous GPU sub-brick `[dz=0..ch][iy=0..depth][dx=0..cw][4]` ready for
    /// `queue.write_texture` at `Origin3d { x: x0, y: 0, z: z0 }`.
    pub fn pack_brick_subregion(&self, x0: usize, z0: usize, cw: usize, ch: usize) -> (Vec<u8>, Vec<u8>) {
        let d = self.depth;
        let w = self.width;
        let row_bytes = cw * 4;
        let total = cw * d * ch * 4;
        let mut dm = vec![0u8; total];
        let mut at = vec![0u8; total];
        for dz in 0..ch {
            let iz = z0 + dz;
            for iy in 0..d {
                let src_off = ((iz * d + iy) * w + x0) * 4;
                let dst_off = (dz * d + iy) * row_bytes;
                dm[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&self.density_mat[src_off..src_off + row_bytes]);
                at[dst_off..dst_off + row_bytes]
                    .copy_from_slice(&self.albedo_thermal[src_off..src_off + row_bytes]);
            }
        }
        (dm, at)
    }

    /// Incrementally update the coarse `8×8×8` occupancy bricks affected by a `16×16×D` brick
    /// `(bx16, by16)` (including its 1-voxel dilation halo).
    pub fn update_brick_occupancy_region(
        &self,
        occ: &mut BrickOccupancy3D,
        bx16: usize,
        by16: usize,
    ) -> OccupancySubregion {
        let ox0 = (bx16 * STREAM_BRICK_XY / BRICK_SIZE).saturating_sub(1);
        let ox1 = ((bx16 + 1) * STREAM_BRICK_XY)
            .div_ceil(BRICK_SIZE)
            .saturating_add(1)
            .min(occ.bx_count);
        let oz0 = (by16 * STREAM_BRICK_XY / BRICK_SIZE).saturating_sub(1);
        let oz1 = ((by16 + 1) * STREAM_BRICK_XY)
            .div_ceil(BRICK_SIZE)
            .saturating_add(1)
            .min(occ.bz_count);

        for bz in oz0..oz1 {
            let z0 = (bz * BRICK_SIZE).saturating_sub(1);
            let z1 = ((bz + 1) * BRICK_SIZE + 1).min(self.height);
            for by in 0..occ.by_count {
                let y0 = (by * BRICK_SIZE).saturating_sub(1);
                let y1 = ((by + 1) * BRICK_SIZE + 1).min(self.depth);
                for bx in ox0..ox1 {
                    let x0 = (bx * BRICK_SIZE).saturating_sub(1);
                    let x1 = ((bx + 1) * BRICK_SIZE + 1).min(self.width);

                    let mut max_s = 0u8;
                    let mut max_f = 0u8;
                    let mut max_g = 0u8;
                    let mut max_t = 0u8;
                    for iz in z0..z1 {
                        for iy in y0..y1 {
                            let row = ((iz * self.depth + iy) * self.width) * 4;
                            for ix in x0..x1 {
                                let i = row + ix * 4;
                                max_s = max_s.max(self.density_mat[i]);
                                max_f = max_f.max(self.density_mat[i + 1]);
                                max_g = max_g.max(self.density_mat[i + 2]);
                                max_t = max_t.max(self.albedo_thermal[i + 3]);
                            }
                        }
                    }

                    let bi = ((bz * occ.by_count + by) * occ.bx_count + bx) * 4;
                    let was_active = occ.data[bi] > 2 || occ.data[bi + 1] > 2 || occ.data[bi + 2] > 2;
                    let now_active = max_s > 2 || max_f > 2 || max_g > 2;
                    occ.data[bi] = max_s;
                    occ.data[bi + 1] = max_f;
                    occ.data[bi + 2] = max_g;
                    occ.data[bi + 3] = max_t;
                    if now_active && !was_active {
                        occ.non_empty_bricks += 1;
                    } else if !now_active && was_active {
                        occ.non_empty_bricks = occ.non_empty_bricks.saturating_sub(1);
                    }
                }
            }
        }

        OccupancySubregion {
            x0: ox0,
            z0: oz0,
            cw: ox1.saturating_sub(ox0),
            ch: oz1.saturating_sub(oz0),
        }
    }

    /// Generate the rich procedural isotropic `256×256×16` Frostmarsh True-3D showcase world.
    pub fn generate_showcase() -> Self {
        Self::generate_showcase_with_depth(DEFAULT_DEPTH)
    }

    /// Generate the procedural isotropic `256×256×D` (`D = 16..=32`) Frostmarsh True-3D showcase world.
    /// Higher depths (`Z > 16`) extend alpine mountain peaks, towering conifer spires, and cave massifs!
    pub fn generate_showcase_with_depth(depth: usize) -> Self {
        let w = DEFAULT_WIDTH;
        let d = depth.clamp(MIN_SHOWCASE_DEPTH, MAX_SHOWCASE_DEPTH);
        let h = DEFAULT_HEIGHT;
        let mut vol = Self::new_empty(w, d, h);
        let extra_y = (d as f32 - 16.0).max(0.0);
        let peak_boost = 1.0 + extra_y * 0.095;
        let max_ground = (d as f32 - 0.9).max(15.1);
        let max_tree_y = (d as f32 - 1.2).max(14.8);

        let cave_pts = cave_centerline_points();

        let conifers: &[(f32, f32, f32, f32)] = &[
            (163.0, 136.0, 6.6, 2.5),
            (168.0, 124.0, 6.8, 2.6),
            (161.0, 104.0, 6.4, 2.4),
            (153.0, 98.0, 6.2, 2.3),
            (124.0, 104.0, 6.6, 2.5),
            (116.0, 108.0, 6.4, 2.4),
            (126.0, 96.0, 6.0, 2.3),
            (96.0, 118.0, 6.5, 2.5),
            (89.0, 126.0, 6.6, 2.6),
            (92.0, 162.0, 5.2, 2.2),
            (84.0, 166.0, 5.4, 2.2),
            (148.0, 84.0, 6.2, 2.4),
            (120.0, 86.0, 6.2, 2.4),
            (172.0, 148.0, 6.6, 2.5),
            (156.0, 152.0, 6.4, 2.4),
        ];

        let stalactites: &[(f32, f32, f32, f32, f32)] = &[
            (98.5, 150.2, 10.1, 7.8, 0.85),
            (96.2, 154.6, 10.0, 7.6, 0.90),
            (93.0, 151.0, 10.0, 7.9, 0.78),
            (90.5, 156.2, 9.9, 7.5, 0.95),
            (86.2, 153.5, 10.1, 7.7, 0.88),
            (83.5, 161.2, 10.3, 7.6, 1.00),
            (80.2, 157.8, 10.2, 7.8, 0.85),
            (78.0, 163.5, 10.0, 7.5, 0.92),
        ];

        let pillars: &[(f32, f32, f32)] = &[
            (90.8, 151.6, 1.05),
            (84.6, 157.8, 1.15),
            (78.8, 161.5, 1.05),
        ];

        let crystals: &[(Vec3, f32, [f32; 3], f32)] = &[
            (Vec3::new(93.5, 5.1, 155.8), 1.35, [1.00, 0.55, 0.16], 0.92),
            (Vec3::new(87.2, 5.2, 157.6), 1.55, [0.42, 0.88, 1.00], 0.88),
            (Vec3::new(83.8, 4.9, 155.2), 1.40, [1.00, 0.44, 0.12], 0.96),
            (Vec3::new(79.5, 5.4, 160.8), 1.80, [1.00, 0.62, 0.18], 1.00),
            (Vec3::new(85.5, 6.2, 161.0), 1.35, [0.76, 0.46, 1.00], 0.85),
        ];

        let lanterns: &[Vec3] = &[
            Vec3::new(121.6, 10.35, 124.5),
            Vec3::new(121.6, 10.35, 131.5),
            Vec3::new(138.4, 10.35, 124.5),
            Vec3::new(138.4, 10.35, 131.5),
            Vec3::new(105.5, 9.85, 124.5),
            Vec3::new(154.5, 9.85, 124.5),
        ];

        let mut base_h = vec![0.0f32; w * h];
        for iz in 0..h {
            let wz = iz as f32 + 0.5;
            let rcx = river_center_x(wz);
            for ix in 0..w {
                let wx = ix as f32 + 0.5;
                let n_big = fbm2(wx * 0.018, wz * 0.018);
                let n_med = fbm2(wx * 0.055 + 17.3, wz * 0.055 - 9.1);
                let dist_r = (wx - rcx).abs();

                let river_t = smoothstep(5.5, 16.5, dist_r);
                let mut ground = 1.65 + river_t * 6.85 + n_big * 1.45 + n_med * 0.45;

                let under_path_x = 1.0 - smoothstep(2.5, 6.0, (wx - 113.5).abs());
                let under_path_z =
                    smoothstep(104.0, 112.0, wz) * (1.0 - smoothstep(148.0, 156.0, wz));
                let under_w = under_path_x * under_path_z;
                ground = ground * (1.0 - under_w) + 4.02 * under_w;

                let east_path_x = 1.0 - smoothstep(2.5, 5.8, (wx - 147.0).abs());
                let east_path_z =
                    smoothstep(115.0, 122.0, wz) * (1.0 - smoothstep(134.0, 142.0, wz));
                let east_w = east_path_x * east_path_z;
                ground = ground * (1.0 - east_w) + 4.10 * east_w;

                let z_bridge_env = 1.0 - smoothstep(4.5, 13.0, (wz - 128.0).abs());
                let west_abut = (1.0 - smoothstep(4.0, 10.0, (wx - 101.0).abs())) * z_bridge_env;
                let east_abut = (1.0 - smoothstep(4.0, 10.0, (wx - 159.0).abs())) * z_bridge_env;
                ground = ground * (1.0 - west_abut) + 8.45 * west_abut;
                ground = ground * (1.0 - east_abut) + 8.45 * east_abut;

                let mtn_z = smoothstep(66.0, 82.0, wz) * (1.0 - smoothstep(97.0, 109.0, wz));
                let mtn_ridge = (1.0 - smoothstep(18.0, 58.0, (wx - 137.0).abs())) * mtn_z;
                let peak_left = (1.0
                    - smoothstep(
                        0.0,
                        24.0,
                        ((wx - 118.0).powi(2) + (wz - 92.0).powi(2)).sqrt(),
                    ))
                    * 6.8;
                let peak_right = (1.0
                    - smoothstep(
                        0.0,
                        25.0,
                        ((wx - 155.0).powi(2) + (wz - 90.0).powi(2)).sqrt(),
                    ))
                    * 7.2;
                let peak_mid = (1.0
                    - smoothstep(
                        0.0,
                        20.0,
                        ((wx - 137.0).powi(2) + (wz - 82.0).powi(2)).sqrt(),
                    ))
                    * 6.6;
                ground += (mtn_ridge * 5.0 + peak_left + peak_right + peak_mid) * peak_boost;

                let basin_dx = (wx - 138.0) / 10.5;
                let basin_dz = (wz - 93.0) / 7.5;
                let basin_r = (basin_dx * basin_dx + basin_dz * basin_dz).sqrt();
                let basin_w = 1.0 - smoothstep(0.50, 1.12, basin_r);
                ground = ground * (1.0 - basin_w) + 9.5 * basin_w;

                let notch_x = 1.0 - smoothstep(3.0, 5.5, (wx - 138.0).abs());
                let notch_z = smoothstep(96.5, 99.5, wz) * (1.0 - smoothstep(101.8, 104.0, wz));
                let notch_w = notch_x * notch_z;
                ground = ground * (1.0 - notch_w) + 10.65 * notch_w;

                let pool_r =
                    (((wx - 138.0) / 9.5).powi(2) + ((wz - 107.5) / 6.2).powi(2)).sqrt();
                let pool_w = 1.0 - smoothstep(0.45, 1.10, pool_r);
                ground = ground * (1.0 - pool_w) + 1.5 * pool_w;

                let cave_mtn_dx = (wx - 84.0) / 21.5;
                let cave_mtn_dz = (wz - 159.0) / 22.0;
                let cave_mtn_r = (cave_mtn_dx * cave_mtn_dx + cave_mtn_dz * cave_mtn_dz).sqrt();
                let cave_mtn_w = 1.0 - smoothstep(0.45, 1.08, cave_mtn_r);
                let cave_mtn_h = 13.6 + extra_y * 0.42 + n_med * 0.95 + (1.0 - cave_mtn_r.min(1.0)) * (1.4 + extra_y * 0.35);
                ground = ground.max(ground * (1.0 - cave_mtn_w) + cave_mtn_h * cave_mtn_w);

                let apron_dx = (wx - 106.0) / 7.5;
                let apron_dz = (wz - 150.5) / 6.5;
                let apron_r = (apron_dx * apron_dx + apron_dz * apron_dz).sqrt();
                let apron_w = 1.0 - smoothstep(0.55, 1.10, apron_r);
                ground = ground * (1.0 - apron_w) + 4.25 * apron_w;

                let fire_r =
                    (((wx - 154.0) / 7.5).powi(2) + ((wz - 115.0) / 7.5).powi(2)).sqrt();
                let fire_w = 1.0 - smoothstep(0.5, 1.1, fire_r);
                ground = ground * (1.0 - fire_w) + 8.35 * fire_w;

                let outer_fade = smoothstep(58.0, 88.0, (wx - 128.0).hypot(wz - 128.0));
                ground =
                    ground * (1.0 - outer_fade * 0.45) + (4.5 + n_big * 1.2) * (outer_fade * 0.45);
                ground = ground.clamp(1.2, max_ground);

                base_h[iz * w + ix] = ground;
            }
        }

        for iz in 0..h {
            let wz = iz as f32 + 0.5;
            let rcx = river_center_x(wz);
            for ix in 0..w {
                let wx = ix as f32 + 0.5;
                let gh = base_h[iz * w + ix];
                let n_col = fbm2(wx * 0.14, wz * 0.14);

                for iy in 0..d {
                    let wy = iy as f32 + 0.5;
                    let p = Vec3::new(wx, wy, wz);
                    let n3 = fbm3(p * 0.18);

                    let mut sdf = wy - gh;
                    let mut mat_kind = 0u8;

                    let wf_recess = (((wx - 138.0) / 4.8).powi(2)
                        + ((wy - 6.8) / 3.2).powi(2)
                        + ((wz - 101.6) / 2.4).powi(2))
                    .sqrt()
                        - 1.0;
                    if wf_recess < 0.0 {
                        sdf = sdf.max(-wf_recess * 2.4);
                    }

                    if (98.0..=116.0).contains(&wx)
                        && (95.0..=114.0).contains(&wz)
                        && (3.5..=13.2).contains(&wy)
                    {
                        let plane_d =
                            ((wx - 107.0) + (wz - 105.0)) * std::f32::consts::FRAC_1_SQRT_2;
                        let box_y = (wy - 12.4).max(3.8 - wy);
                        let box_diag = (((wx - 107.0) - (wz - 105.0)).abs()
                            * std::f32::consts::FRAC_1_SQRT_2)
                            - 7.5;
                        let wall_sdf = plane_d.max(box_y).max(box_diag);
                        if wall_sdf < sdf {
                            sdf = wall_sdf;
                            mat_kind = 3;
                        }
                    }

                    if (97.0..=163.0).contains(&wx) && (122.0..=134.0).contains(&wz) && wy <= 11.2 {
                        let top_y = bridge_deck_top_y(wx);
                        let half_w = 3.8;
                        let dz = (wz - 128.0).abs();

                        let prism_x = (wx - 130.0).abs() - 30.5;
                        let prism_z = dz - half_w;
                        let prism_y = wy - top_y;
                        let mut bridge_sdf = prism_x.max(prism_z).max(prism_y);

                        let arches: [(f32, f32, f32, f32); 3] = [
                            (113.0, 6.8, 3.3, 4.45),
                            (130.0, 7.4, 3.0, 5.15),
                            (147.0, 6.8, 3.3, 4.45),
                        ];
                        for &(ax, rx, spring_y, ry) in &arches {
                            let dx = (wx - ax) / rx;
                            let dy = ((wy - spring_y).max(0.0)) / ry;
                            let arch_r = (dx * dx + dy * dy).sqrt();
                            let open_sdf = (arch_r - 1.0) * rx.min(ry);
                            let rect_open = (wx - ax).abs() - rx;
                            let hole_sdf = if wy < spring_y { rect_open } else { open_sdf };
                            bridge_sdf = bridge_sdf.max(-hole_sdf);
                        }

                        for &px in &[121.6f32, 138.4] {
                            let butt_x = (wx - px).abs() - 1.35;
                            let butt_z = dz - 4.75;
                            let butt_y = wy - (top_y - 0.5);
                            let butt_sdf = butt_x.max(butt_z).max(butt_y);
                            bridge_sdf = bridge_sdf.min(butt_sdf);
                        }

                        let para_z = (dz - 3.45).abs() - 0.42;
                        let para_x = (wx - 130.0).abs() - 29.5;
                        let merlon = if ((wx + 1.5).rem_euclid(5.0)) < 3.6 {
                            0.85
                        } else {
                            0.45
                        };
                        let para_y = (wy - (top_y + merlon)).max(top_y - 0.3 - wy);
                        let para_sdf = para_x.max(para_z).max(para_y);
                        bridge_sdf = bridge_sdf.min(para_sdf);

                        if bridge_sdf < sdf {
                            sdf = bridge_sdf;
                            if dz < 2.6 && wy > top_y - 0.45 && wy <= top_y + 0.15 {
                                mat_kind = 2;
                            } else {
                                mat_kind = 1;
                            }
                        }
                    }

                    for &lp in lanterns {
                        let d_post = ((wx - lp.x).hypot(wz - lp.z) - 0.28)
                            .max(wy - lp.y)
                            .max((lp.y - 1.2) - wy);
                        if d_post < sdf {
                            sdf = d_post;
                            mat_kind = 1;
                        }
                        let d_lamp = (p - lp).length() - 0.42;
                        if d_lamp < sdf {
                            sdf = d_lamp;
                            mat_kind = 7;
                        }
                    }

                    let mut inside_cave_zone = false;
                    let mut cave_wall_prox = 0.0f32;
                    if (64.0..=112.0).contains(&wx) && (138.0..=185.0).contains(&wz) {
                        let mut best_tunnel_sdf = 999.0f32;
                        for seg in 0..(cave_pts.len() - 1) {
                            let (a, hw_a, vh_a) = cave_pts[seg];
                            let (b, hw_b, vh_b) = cave_pts[seg + 1];
                            let ab = b - a;
                            let len2 = ab.length_squared().max(1e-6);
                            let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
                            let center = a + ab * t;
                            let hw = hw_a + (hw_b - hw_a) * t;
                            let vh = vh_a + (vh_b - vh_a) * t;

                            let horiz_d = (wx - center.x).hypot(wz - center.z);
                            let y_rel = wy - center.y;
                            let norm_x = horiz_d / hw;
                            let norm_y = if y_rel < 1.2 {
                                ((1.2 - y_rel) / 1.2).max(0.0)
                            } else {
                                (y_rel - 1.2) / (vh - 1.2)
                            };
                            let ripple = if y_rel > 0.6 { n3 * 0.14 } else { 0.0 };
                            let vault_r = (norm_x * norm_x + norm_y * norm_y).sqrt() + ripple;
                            let floor_cut = -y_rel;
                            let tunnel_d = ((vault_r - 1.0) * hw.min(vh)).max(floor_cut);
                            best_tunnel_sdf = best_tunnel_sdf.min(tunnel_d);
                        }

                        if best_tunnel_sdf < 2.2 {
                            cave_wall_prox =
                                (1.0 - (best_tunnel_sdf / 2.2).clamp(0.0, 1.0)).max(0.0);
                        }
                        if best_tunnel_sdf < 0.0 {
                            inside_cave_zone = true;
                            sdf = sdf.max(-best_tunnel_sdf);
                        }
                        if cave_wall_prox > 0.15 && sdf < 0.8 && mat_kind == 0 {
                            mat_kind = 3;
                        }

                        for &(sx, sz, ceil_y, tip_y, base_r) in stalactites {
                            if wy >= tip_y - 0.5 && wy <= ceil_y + 0.8 {
                                let t = ((wy - tip_y) / (ceil_y - tip_y).max(0.1)).clamp(0.0, 1.0);
                                let rad = base_r * (0.12 + 0.88 * t * t);
                                let d_st = ((wx - sx).hypot(wz - sz) - rad)
                                    .max(tip_y - wy)
                                    .max(wy - (ceil_y + 0.4));
                                if d_st < sdf {
                                    sdf = d_st;
                                    mat_kind = 3;
                                }
                            }
                        }

                        for &(px, pz, rad) in pillars {
                            if (3.8..=10.6).contains(&wy) {
                                let waist = 1.0 + 0.25 * ((wy - 7.0) / 3.0).powi(2);
                                let d_pil = (wx - px).hypot(wz - pz) - rad * waist + n3 * 0.18;
                                if d_pil < sdf {
                                    sdf = d_pil;
                                    mat_kind = 3;
                                }
                            }
                        }

                        for &(cp, cr, _, _) in crystals {
                            let d_cr = (p - cp).length() - cr * (0.85 + 0.25 * n3.abs());
                            if d_cr < sdf {
                                sdf = d_cr;
                                mat_kind = 6;
                            }
                        }

                        if (3.6..=4.35).contains(&wy) && !inside_cave_zone {
                            let (dm, _) = dist_to_segment(
                                Vec3::new(wx, 4.2, wz),
                                Vec3::new(93.5, 4.2, 154.8),
                                Vec3::new(78.5, 4.2, 162.5),
                            );
                            if dm < 1.35 && wy > 3.75 && wy < 4.32 {
                                mat_kind = 8;
                            }
                        }
                    }

                    for &(tx, tz, tree_h, max_r) in conifers {
                        let dx = wx - tx;
                        let dz = wz - tz;
                        let dist2 = dx * dx + dz * dz;
                        if dist2 < (max_r + 1.5) * (max_r + 1.5) {
                            let t_base =
                                base_h[(tz as usize).min(h - 1) * w + (tx as usize).min(w - 1)]
                                    - 0.6;
                            let top_y = (t_base + tree_h + extra_y * 0.14).min(max_tree_y);
                            let rel_y = wy - t_base;
                            if rel_y >= -0.8 && wy <= top_y + 0.4 {
                                let r_xz = dist2.sqrt();
                                let trunk_r = (0.86
                                    * (1.0 - (rel_y / tree_h).clamp(0.0, 1.0) * 0.28))
                                    .max(0.62);
                                let trunk_sdf =
                                    (r_xz - trunk_r).max(wy - (top_y - 1.0)).max(-0.8 - rel_y);
                                if trunk_sdf < sdf {
                                    sdf = trunk_sdf;
                                    mat_kind = 4;
                                }

                                if rel_y >= 1.3 && wy <= top_y + 0.2 {
                                    let f_t = ((wy - (t_base + 1.4))
                                        / (top_y - (t_base + 1.4)).max(0.5))
                                    .clamp(0.0, 1.0);
                                    let tier = ((rel_y * 1.10).fract() * 0.35 + 0.76).min(1.08);
                                    let canopy_r =
                                        (max_r * (1.0 - f_t * 0.78) * tier + n3 * 0.22).max(0.65);
                                    let foliage_sdf = (r_xz - canopy_r)
                                        .max(wy - top_y)
                                        .max((t_base + 1.4) - wy);
                                    if foliage_sdf < sdf {
                                        sdf = foliage_sdf;
                                        mat_kind = 5;
                                    }
                                }
                            }
                        }
                    }

                    for &(fx, fy, fz, scale) in &[
                        (154.0f32, 8.45f32, 115.0f32, 1.48f32),
                        (96.0, 4.40, 149.2, 0.50),
                    ] {
                        let d_log1 = dist_to_segment(
                            p,
                            Vec3::new(fx - 1.6 * scale, fy, fz - 0.9 * scale),
                            Vec3::new(fx + 1.6 * scale, fy + 0.55 * scale, fz + 0.9 * scale),
                        )
                        .0 - 0.42 * scale;
                        let d_log2 = dist_to_segment(
                            p,
                            Vec3::new(fx - 1.3 * scale, fy + 0.45 * scale, fz + 1.1 * scale),
                            Vec3::new(fx + 1.3 * scale, fy, fz - 1.1 * scale),
                        )
                        .0 - 0.40 * scale;
                        let d_logs = d_log1.min(d_log2);
                        if d_logs < sdf {
                            sdf = d_logs;
                            mat_kind = 9;
                        }
                    }

                    let band = if mat_kind == 1 || mat_kind == 2 {
                        0.46
                    } else {
                        0.64
                    };
                    let solid_d = sdf_to_density(sdf, band);

                    let mut fluid_d = 0.0f32;
                    if wy < 4.05 && (wx - rcx).abs() < 16.0 && solid_d < 0.75 {
                        let fd = sdf_to_density(wy - 3.80, 0.32);
                        fluid_d = fluid_d.max(fd * (1.0 - smoothstep(0.45, 0.75, solid_d)));
                    }
                    let b_dx = (wx - 138.0) / 9.2;
                    let b_dz = (wz - 93.0) / 6.8;
                    if b_dx * b_dx + b_dz * b_dz < 1.15 && wy < 11.6 && wy > 8.6 && solid_d < 0.75 {
                        let fd = sdf_to_density(wy - 11.40, 0.28);
                        fluid_d = fluid_d.max(fd * (1.0 - smoothstep(0.45, 0.75, solid_d)));
                    }
                    if (132.0..=144.0).contains(&wx)
                        && (99.0..=108.5).contains(&wz)
                        && (3.5..=11.5).contains(&wy)
                    {
                        let drop_t = ((11.3 - wy) / 7.5).clamp(0.0, 1.0);
                        let wf_z = 100.2 + 5.6 * drop_t.sqrt();
                        let wf_half_w = 3.2 + drop_t * 1.3;
                        let wf_thick = 1.05 + drop_t * 0.45;
                        let d_x = (wx - 138.0).abs() - wf_half_w;
                        let d_z = (wz - wf_z).abs() - wf_thick;
                        let wf_sdf = d_x.max(d_z);
                        let wf_f = sdf_to_density(wf_sdf, 0.50) * (0.90 + 0.12 * n3);
                        fluid_d = fluid_d.max(wf_f * (1.0 - smoothstep(0.5, 0.8, solid_d)));
                    }

                    let mut gas_d = 0.0f32;
                    let mut gas_albedo = [0.88f32, 0.94, 1.00];
                    let mut gas_glow = 0.0f32;

                    let mist_r = (((wx - 138.0) / 7.0).powi(2)
                        + ((wy - 5.1) / 2.5).powi(2)
                        + ((wz - 106.5) / 4.5).powi(2))
                    .sqrt();
                    if mist_r < 1.35 && solid_d < 0.55 {
                        let m =
                            (1.0 - smoothstep(0.22, 1.25, mist_r)) * (0.56 + 0.35 * n3.max(-0.2));
                        if m > gas_d {
                            gas_d = m.clamp(0.0, 0.78);
                            gas_albedo = [0.90, 0.96, 1.00];
                            gas_glow = 0.0;
                        }
                    }

                    for &(fx, fy, fz, scale) in &[
                        (154.0f32, 8.5f32, 115.0f32, 1.48f32),
                        (96.0, 4.5, 149.2, 0.45),
                    ] {
                        let dy = wy - fy;
                        if dy >= -0.2 && dy <= 7.0 * scale {
                            let h_t = (dy / (6.8 * scale)).clamp(0.0, 1.0);
                            let cx = fx
                                + h_t * h_t * 3.8 * scale
                                + (dy * 0.9 + wz * 0.25).sin() * 0.45 * scale;
                            let cz = fz
                                + h_t * 1.8 * scale
                                + (dy * 0.75 + wx * 0.25).cos() * 0.40 * scale;
                            let plume_r = (1.65 + h_t * 2.1) * scale;
                            let r = (wx - cx).hypot(wz - cz) / plume_r;
                            if r < 1.25 && solid_d < 0.55 {
                                let density = (1.0 - smoothstep(0.15, 1.15, r))
                                    * (1.0 - h_t * 0.55)
                                    * (0.68 + 0.38 * n3);
                                if density > gas_d {
                                    gas_d = density.clamp(0.0, 0.92);
                                    if h_t < 0.36 {
                                        let flame_heat = (1.0 - h_t / 0.36) * (1.0 - r * 0.5);
                                        gas_albedo = [1.0, 0.58 + 0.32 * flame_heat, 0.16];
                                        gas_glow = (0.72 + 0.28 * flame_heat).clamp(0.0, 1.0);
                                    } else {
                                        let smoke_v = 0.20 + 0.14 * h_t;
                                        let ember = if n3 > 0.38 && h_t < 0.68 { 0.85 } else { 0.0 };
                                        gas_albedo = if ember > 0.1 {
                                            [1.0, 0.48, 0.10]
                                        } else {
                                            [smoke_v * 1.08, smoke_v * 1.02, smoke_v * 0.98]
                                        };
                                        gas_glow = ember;
                                    }
                                }
                            }
                        }
                    }

                    let mut albedo = [0.36f32, 0.40, 0.44];
                    let mut rough = 0.82f32;
                    let mut glow = 0.0f32;

                    if solid_d > 0.04 {
                        match mat_kind {
                            0 => {
                                let strata = (wy * 2.2 + n_col * 2.5).sin() * 0.06;
                                if wy < 4.35 {
                                    albedo = [0.27 + strata, 0.29 + strata, 0.31 + strata];
                                    rough = 0.28;
                                } else if wy > 11.4 + n_col * 0.9 {
                                    albedo = [0.86, 0.90, 0.95];
                                    rough = 0.55;
                                } else if wy > 8.6 + n_col * 0.7 {
                                    albedo = [0.42 + strata, 0.44 + strata, 0.48 + strata];
                                    rough = 0.84;
                                } else {
                                    let moss = smoothstep(4.2, 6.2, wy);
                                    let g_var = n_col * 0.08;
                                    albedo = [
                                        0.20 + 0.07 * (1.0 - moss) + g_var,
                                        0.38 + 0.06 * moss + g_var,
                                        0.20 + 0.04 * (1.0 - moss),
                                    ];
                                    rough = 0.88;
                                }
                                let d_wf = (wx - 138.0).hypot(wz - 94.0);
                                if d_wf < 7.5 && wy < 11.5 {
                                    rough *= 0.35;
                                    albedo[0] *= 0.72;
                                    albedo[1] *= 0.76;
                                    albedo[2] *= 0.82;
                                }
                            }
                            1 => {
                                let block = ((((wx * 0.75).floor() as i32)
                                    ^ ((wy * 1.4).floor() as i32)
                                    ^ ((wz * 0.8).floor() as i32))
                                    & 3) as f32
                                    * 0.035;
                                albedo = [0.54 + block, 0.52 + block, 0.49 + block];
                                rough = if wy < 4.5 { 0.32 } else { 0.76 };
                            }
                            2 => {
                                let plank = ((wx * 1.6).sin() * 0.045) + n3 * 0.04;
                                albedo = [0.46 + plank, 0.30 + plank * 0.7, 0.18 + plank * 0.5];
                                rough = 0.74;
                            }
                            3 => {
                                let v = n3 * 0.06;
                                albedo = [0.22 + v, 0.23 + v, 0.27 + v];
                                rough = if wy < 5.2 { 0.42 } else { 0.78 };
                            }
                            4 => {
                                albedo = [0.30 + n3 * 0.05, 0.20 + n3 * 0.04, 0.14 + n3 * 0.03];
                                rough = 0.90;
                            }
                            5 => {
                                let snow_tip = smoothstep(12.2, 14.4, wy) * 0.45;
                                let v = n3 * 0.06;
                                albedo = [
                                    (0.11 + v) * (1.0 - snow_tip) + 0.82 * snow_tip,
                                    (0.28 + v) * (1.0 - snow_tip) + 0.88 * snow_tip,
                                    (0.16 + v) * (1.0 - snow_tip) + 0.94 * snow_tip,
                                ];
                                rough = 0.86;
                            }
                            6 => {
                                let mut best_c = [1.0, 0.56, 0.18];
                                let mut best_g = 0.92;
                                let mut best_d = 999.0;
                                for &(cp, _, rgb, gl) in crystals {
                                    let dist = (p - cp).length();
                                    if dist < best_d {
                                        best_d = dist;
                                        best_c = rgb;
                                        best_g = gl;
                                    }
                                }
                                albedo = best_c;
                                rough = 0.14;
                                glow = best_g;
                            }
                            7 => {
                                albedo = [1.0, 0.78, 0.32];
                                rough = 0.18;
                                glow = 0.96;
                            }
                            8 => {
                                let pulse = 0.82 + 0.18 * n3.abs();
                                albedo = [1.0, 0.46 * pulse, 0.08];
                                rough = 0.22;
                                glow = 0.98;
                            }
                            9 => {
                                let hot = smoothstep(-0.1, 0.35, n3);
                                albedo = [0.16 + 0.84 * hot, 0.10 + 0.35 * hot, 0.08];
                                rough = 0.80;
                                glow = 0.75 * hot + 0.18;
                            }
                            _ => {}
                        }
                    } else if gas_d > 0.02 {
                        albedo = gas_albedo;
                        glow = gas_glow;
                        rough = 0.90;
                    } else if fluid_d > 0.02 {
                        let is_wf = (132.0..=144.0).contains(&wx)
                            && (99.0..=108.5).contains(&wz)
                            && wy > 4.1;
                        if is_wf {
                            albedo = [0.86, 0.95, 1.00];
                            rough = 0.10;
                        } else {
                            albedo = [0.10, 0.44, 0.66];
                            rough = 0.05;
                        }
                    }

                    let idx = vol.byte_index(ix, iy, iz);
                    vol.density_mat[idx] = to_u8(solid_d);
                    vol.density_mat[idx + 1] = to_u8(fluid_d);
                    vol.density_mat[idx + 2] = to_u8(gas_d);
                    vol.density_mat[idx + 3] = to_u8(rough);

                    vol.albedo_thermal[idx] = to_u8(albedo[0]);
                    vol.albedo_thermal[idx + 1] = to_u8(albedo[1]);
                    vol.albedo_thermal[idx + 2] = to_u8(albedo[2]);
                    vol.albedo_thermal[idx + 3] = to_u8(glow);
                }
            }
        }

        vol
    }
}
