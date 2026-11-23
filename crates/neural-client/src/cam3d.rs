//! Tier 2 Client-Side 3D Free-Camera Heightfield Ray-Marcher & G-Buffer Projector.
//!
//! Projects the server's live `128×128` compositional `A+` world into a screen-space
//! `[1, 18, 256, 256]` 3D G-buffer (top-2 species, temp, burning, critters, depth, 3D normals, height)
//! matching `tier2_deferred_3d.onnx`, plus exact depth-tested 3D entity projection for 60 Hz combat!

use frostmarsh_proto::{GRID_H, GRID_W};

pub const GBUF_CHANNELS: usize = 18;
pub const W3D: usize = 256;
pub const H3D: usize = 256;

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target_x: f32,
    pub target_z: f32,
    pub yaw: f32,   // radians around vertical Y axis
    pub pitch: f32, // radians above horizon (0.35..1.35)
    pub dist: f32,  // distance in grid cells
    pub fov_y: f32, // vertical FOV in radians
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target_x: GRID_W as f32 * 0.5,
            target_z: GRID_H as f32 * 0.5,
            yaw: -0.65,
            pitch: 0.72,
            dist: 92.0,
            fov_y: 0.82,
        }
    }
}

#[derive(Clone, Copy)]
pub struct CameraBasis {
    pub pos: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub forward: [f32; 3],
    pub tan_half_fov: f32,
}

impl OrbitCamera {
    pub fn basis(&self) -> CameraBasis {
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let px = self.target_x + self.dist * cp * sy;
        let py = 3.0 + self.dist * sp;
        let pz = self.target_z + self.dist * cp * cy;

        let f = norm3([self.target_x - px, 2.0 - py, self.target_z - pz]);
        let r = norm3(cross3(f, [0.0, 1.0, 0.0]));
        let u = cross3(r, f);
        CameraBasis {
            pos: [px, py, pz],
            right: r,
            up: u,
            forward: f,
            tan_half_fov: (self.fov_y * 0.5).tan(),
        }
    }

    /// Project a 3D world point `(wx, wy, wz)` into screen `(sx, sy, linear_depth_norm)`.
    pub fn project_point(&self, wx: f32, wy: f32, wz: f32) -> Option<(f32, f32, f32)> {
        let b = self.basis();
        let d = [wx - b.pos[0], wy - b.pos[1], wz - b.pos[2]];
        let z_cam = dot3(d, b.forward);
        if z_cam <= 1.0 {
            return None;
        }
        let x_cam = dot3(d, b.right);
        let y_cam = dot3(d, b.up);
        let ndc_x = x_cam / (z_cam * b.tan_half_fov);
        let ndc_y = y_cam / (z_cam * b.tan_half_fov);
        let sx = (ndc_x * 0.5 + 0.5) * (W3D as f32);
        let sy = (0.5 - ndc_y * 0.5) * (H3D as f32);
        let depth_norm = (z_cam / 220.0).clamp(0.0, 1.0);
        Some((sx, sy, depth_norm))
    }
}

#[inline]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm3(a: [f32; 3]) -> [f32; 3] {
    let l = dot3(a, a).sqrt().max(1e-6);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Height (in world cell units, 0..14) derived cleanly from `A+` species fractions.
#[inline]
pub fn cell_height_norm(aplus: &[u8], gx: usize, gy: usize) -> f32 {
    let idx = (gy * GRID_W + gx) * 4;
    let b0 = aplus[idx];
    let b1 = aplus[idx + 1];
    let s1 = ((b0 >> 4) & 0x0f) as usize;
    let s2 = (b0 & 0x0f) as usize;
    let frac = ((b1 >> 4) & 0x0f) as f32 / 15.0;

    // Matching Tier 2 contract heights:
    // water=0.08, ice=0.14, steam=0.32, soil=0.26, rock=0.78, wood=0.58, ash=0.22, smoke=0.40, oil=0.10, plant=0.34
    const H_SP: [f32; 10] = [0.08, 0.14, 0.32, 0.26, 0.78, 0.58, 0.22, 0.40, 0.10, 0.34];
    let mut h = if s1 < 10 { H_SP[s1] * (1.0 - frac) } else { 0.08 };
    if s2 < 10 {
        h += H_SP[s2] * frac;
    }
    h
}

/// `O(N)` separable 2-axis running-sum box blur + top-weighted (`2x +Y`) normal smoothing
/// ported from Approach 4 (`moba-world-poc/src/parametric.rs`) to eliminate blocky staircase
/// normals on the live 128×128 `A+` heightfield in <0.1 ms before Tier 2 neural deferred shading.
fn compute_separable_smooth_normals(hmap: &[f32], h_scale: f32) -> Vec<[f32; 3]> {
    let len = GRID_W * GRID_H;
    let mut raw = vec![[0.0f32; 3]; len];
    for z in 0..GRID_H {
        let zd = z.saturating_sub(1);
        let zu = (z + 1).min(GRID_H - 1);
        for x in 0..GRID_W {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(GRID_W - 1);
            let dhdx = (hmap[z * GRID_W + xr] - hmap[z * GRID_W + xl]) * h_scale * 0.5;
            let dhdz = (hmap[zu * GRID_W + x] - hmap[zd * GRID_W + x]) * h_scale * 0.5;
            // Top-weighted (+2.0 Y bias) for crisp MOBA/isometric readability
            raw[z * GRID_W + x] = norm3([-dhdx, 2.0, -dhdz]);
        }
    }
    // Horizontal running-sum blur (radius 1 -> 3-tap)
    let mut tmp = vec![[0.0f32; 3]; len];
    for z in 0..GRID_H {
        let row = z * GRID_W;
        for x in 0..GRID_W {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(GRID_W - 1);
            let a = raw[row + xl];
            let b = raw[row + x];
            let c = raw[row + xr];
            tmp[row + x] = [
                (a[0] + b[0] * 2.0 + c[0]) * 0.25,
                (a[1] + b[1] * 2.0 + c[1]) * 0.25,
                (a[2] + b[2] * 2.0 + c[2]) * 0.25,
            ];
        }
    }
    // Vertical running-sum blur + re-normalize
    let mut out = vec![[0.0f32; 3]; len];
    for z in 0..GRID_H {
        let zd = z.saturating_sub(1) * GRID_W;
        let zc = z * GRID_W;
        let zu = (z + 1).min(GRID_H - 1) * GRID_W;
        for x in 0..GRID_W {
            let a = tmp[zd + x];
            let b = tmp[zc + x];
            let c = tmp[zu + x];
            out[zc + x] = norm3([
                a[0] + b[0] * 2.0 + c[0],
                a[1] + b[1] * 2.0 + c[1],
                a[2] + b[2] * 2.0 + c[2],
            ]);
        }
    }
    out
}

/// Optional 9-step Cool-Indigo (`#261e48`) / Warm-Cream (`#fff4d6`) hue-shifted tone ramp
/// from Approach 4, toggleable live via `T` on top of the Tier 2 Neural Deferred Shader.
pub fn apply_9step_tone_ramp(fb: &mut [u32]) {
    const SHADOW: [f32; 3] = [38.0 / 255.0, 30.0 / 255.0, 72.0 / 255.0]; // #261e48
    const HIGHLIGHT: [f32; 3] = [1.0, 244.0 / 255.0, 214.0 / 255.0]; // #fff4d6
    for px in fb.iter_mut() {
        let r = ((*px >> 16) & 0xff) as f32 / 255.0;
        let g = ((*px >> 8) & 0xff) as f32 / 255.0;
        let b = (*px & 0xff) as f32 / 255.0;
        let lum = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0);
        let q_lum = (lum * 8.0).round() / 8.0;
        let scale = if lum > 1e-4 { q_lum / lum } else { 0.0 };
        let mut nr = (r * scale).clamp(0.0, 1.0);
        let mut ng = (g * scale).clamp(0.0, 1.0);
        let mut nb = (b * scale).clamp(0.0, 1.0);
        // Pull shadows toward cool indigo, sunlit peaks toward warm cream
        let sh_w = ((0.35 - q_lum).max(0.0) / 0.35) * 0.28;
        let hi_w = ((q_lum - 0.72).max(0.0) / 0.28) * 0.22;
        nr = nr * (1.0 - sh_w - hi_w) + SHADOW[0] * sh_w + HIGHLIGHT[0] * hi_w;
        ng = ng * (1.0 - sh_w - hi_w) + SHADOW[1] * sh_w + HIGHLIGHT[1] * hi_w;
        nb = nb * (1.0 - sh_w - hi_w) + SHADOW[2] * sh_w + HIGHLIGHT[2] * hi_w;
        *px = ((nr.clamp(0.0, 1.0) * 255.0) as u32) << 16
            | ((ng.clamp(0.0, 1.0) * 255.0) as u32) << 8
            | ((nb.clamp(0.0, 1.0) * 255.0) as u32);
    }
}

/// Ray-march the 128×128 `A+` heightfield into a `[1, 18, 256, 256]` G-buffer + depth buffer.
pub fn render_gbuffer_3d(
    aplus: &[u8],
    cam: &OrbitCamera,
    gbuf: &mut [f32],
    depth_buf: &mut [f32],
) {
    let plane = W3D * H3D;
    gbuf.fill(0.0);
    // Precompute heightfield + O(N) separable smooth normals on the 128x128 grid (~0.2ms!)
    let mut hmap = [0.08f32; GRID_W * GRID_H];
    for y in 0..GRID_H {
        for x in 0..GRID_W {
            hmap[y * GRID_W + x] = cell_height_norm(aplus, x, y);
        }
    }
    let h_scale = 14.0f32; // world-space max cliff height in cell units
    let smooth_normals = compute_separable_smooth_normals(&hmap, h_scale);

    let b = cam.basis();
    let max_t = 220.0f32;

    for py in 0..H3D {
        let ndc_y = 1.0 - 2.0 * ((py as f32 + 0.5) / H3D as f32);
        for px in 0..W3D {
            let ndc_x = 2.0 * ((px as f32 + 0.5) / W3D as f32) - 1.0;
            let ray = norm3([
                b.forward[0] + b.right[0] * ndc_x * b.tan_half_fov + b.up[0] * ndc_y * b.tan_half_fov,
                b.forward[1] + b.right[1] * ndc_x * b.tan_half_fov + b.up[1] * ndc_y * b.tan_half_fov,
                b.forward[2] + b.right[2] * ndc_x * b.tan_half_fov + b.up[2] * ndc_y * b.tan_half_fov,
            ]);

            let out_idx = py * W3D + px;

            // Intersect ray with slab y ∈ [0, h_scale]
            if ray[1] >= -1e-4 && b.pos[1] > h_scale {
                gbuf[13 * plane + out_idx] = 1.0; // sky
                depth_buf[out_idx] = 1.0;
                continue;
            }
            let t_start = if b.pos[1] > h_scale {
                ((h_scale - b.pos[1]) / ray[1]).max(1.0)
            } else {
                1.0
            };
            let t_end = if ray[1] < -1e-4 {
                ((-b.pos[1]) / ray[1]).min(max_t)
            } else {
                max_t
            };

            let mut t = t_start;
            let mut hit = None;
            let step = 0.85f32;
            while t < t_end {
                let wx = b.pos[0] + ray[0] * t;
                let wy = b.pos[1] + ray[1] * t;
                let wz = b.pos[2] + ray[2] * t;
                if wx >= 0.0 && wz >= 0.0 && wx < (GRID_W - 1) as f32 && wz < (GRID_H - 1) as f32 {
                    let gx = wx as usize;
                    let gz = wz as usize;
                    let gh = hmap[gz * GRID_W + gx] * h_scale;
                    if wy <= gh {
                        // Binary refine 3 steps for crisp edges
                        let mut lo = (t - step).max(t_start);
                        let mut hi = t;
                        for _ in 0..3 {
                            let mid = 0.5 * (lo + hi);
                            let mx = (b.pos[0] + ray[0] * mid).clamp(0.0, (GRID_W - 1) as f32) as usize;
                            let mz = (b.pos[2] + ray[2] * mid).clamp(0.0, (GRID_H - 1) as f32) as usize;
                            let my = b.pos[1] + ray[1] * mid;
                            if my <= hmap[mz * GRID_W + mx] * h_scale {
                                hi = mid;
                            } else {
                                lo = mid;
                            }
                        }
                        let hx = (b.pos[0] + ray[0] * hi).clamp(0.0, (GRID_W - 1) as f32) as usize;
                        let hz = (b.pos[2] + ray[2] * hi).clamp(0.0, (GRID_H - 1) as f32) as usize;
                        hit = Some((hx, hz, hi));
                        break;
                    }
                }
                t += step;
            }

            if let Some((gx, gz, t_hit)) = hit {
                let cidx = (gz * GRID_W + gx) * 4;
                let b0 = aplus[cidx];
                let b1 = aplus[cidx + 1];
                let b2 = aplus[cidx + 2];
                let b3 = aplus[cidx + 3];
                let s1 = ((b0 >> 4) & 0x0f) as usize;
                let s2 = (b0 & 0x0f) as usize;
                let frac = ((b1 >> 4) & 0x0f) as f32 / 15.0;

                if s1 < 10 {
                    gbuf[s1 * plane + out_idx] += 1.0 - frac;
                }
                if s2 < 10 {
                    gbuf[s2 * plane + out_idx] += frac;
                }
                gbuf[10 * plane + out_idx] = b2 as f32 / 255.0;
                gbuf[11 * plane + out_idx] = if (b3 & (1 << 2)) != 0 { 1.0 } else { 0.0 };
                gbuf[12 * plane + out_idx] = ((b3 >> 5) & 0x07) as f32 / 7.0;

                let d_norm = (t_hit / max_t).clamp(0.0, 1.0);
                gbuf[13 * plane + out_idx] = d_norm;
                depth_buf[out_idx] = d_norm;

                // O(N) separable top-weighted smooth 3D surface normal
                let n = smooth_normals[gz * GRID_W + gx];
                gbuf[14 * plane + out_idx] = n[0];
                gbuf[15 * plane + out_idx] = n[1];
                gbuf[16 * plane + out_idx] = n[2];
                gbuf[17 * plane + out_idx] = hmap[gz * GRID_W + gx];
            } else {
                gbuf[13 * plane + out_idx] = 1.0;
                depth_buf[out_idx] = 1.0;
            }
        }
    }
}

/// Fast fallback 3D shading from the G-buffer ONLY used if `tier2_deferred_3d.onnx` isn't loaded yet.
pub fn shade_gbuffer_fallback(gbuf: &[f32], fb: &mut [u32]) {
    let plane = W3D * H3D;
    let sp_rgb: [[f32; 3]; 10] = [
        [24.0, 108.0, 198.0],  // water
        [180.0, 236.0, 255.0], // ice
        [210.0, 224.0, 235.0], // steam
        [96.0, 76.0, 56.0],    // soil
        [98.0, 102.0, 112.0],  // rock
        [52.0, 96.0, 44.0],    // wood
        [62.0, 62.0, 68.0],    // ash
        [44.0, 46.0, 52.0],    // smoke
        [40.0, 22.0, 58.0],    // oil
        [66.0, 146.0, 56.0],   // plant
    ];
    let sun = norm3([0.55, 0.75, 0.35]);

    for py in 0..H3D {
        let sky_t = py as f32 / H3D as f32;
        for px in 0..W3D {
            let idx = py * W3D + px;
            let depth = gbuf[13 * plane + idx];
            if depth >= 0.999 {
                let r = (18.0 + 30.0 * sky_t) as u32;
                let g = (26.0 + 44.0 * sky_t) as u32;
                let b = (48.0 + 62.0 * sky_t) as u32;
                fb[idx] = (r << 16) | (g << 8) | b;
                continue;
            }
            let mut rgb = [0.0f32; 3];
            for s in 0..10 {
                let w = gbuf[s * plane + idx];
                if w > 0.0 {
                    rgb[0] += sp_rgb[s][0] * w;
                    rgb[1] += sp_rgb[s][1] * w;
                    rgb[2] += sp_rgb[s][2] * w;
                }
            }
            let n = [
                gbuf[14 * plane + idx],
                gbuf[15 * plane + idx],
                gbuf[16 * plane + idx],
            ];
            let ndl = dot3(n, sun).clamp(0.0, 1.0);
            let lit = 0.38 + 0.68 * ndl;
            for c in 0..3 {
                rgb[c] *= lit;
            }
            if gbuf[11 * plane + idx] > 0.5 {
                rgb = [255.0, 145.0, 35.0];
            }
            let crit = gbuf[12 * plane + idx];
            if crit > 0.25 {
                rgb[0] = rgb[0] * 0.4 + 250.0 * 0.6;
                rgb[1] = rgb[1] * 0.4 + 215.0 * 0.6;
                rgb[2] = rgb[2] * 0.4 + 75.0 * 0.6;
            }
            // Atmospheric depth fog
            let fog = (depth * 0.55).clamp(0.0, 0.55);
            let r = (rgb[0] * (1.0 - fog) + 32.0 * fog).clamp(0.0, 255.0) as u32;
            let g = (rgb[1] * (1.0 - fog) + 46.0 * fog).clamp(0.0, 255.0) as u32;
            let b = (rgb[2] * (1.0 - fog) + 74.0 * fog).clamp(0.0, 255.0) as u32;
            fb[idx] = (r << 16) | (g << 8) | b;
        }
    }
}
