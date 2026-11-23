//! Parametric Continuous-Space CSG/SDF Asset Engine -> Approach 4 Compiler
//!
//! Compiles continuous-unit parametric volumes into:
//!   1. Greedy-merged 3D triangle meshes with 4-level baked voxel AO & O(N) top-weighted smooth normals
//!   2. Surface-sampled hard-Z GPU-instanced Gaussian Surfel prototypes from smooth-min organic SDFs

use crate::mesh_gen::MeshVertex;
use crate::palette::hex_to_rgb;
use crate::surfel_gen::SplatProto;
use glam::Vec3;
use std::collections::HashMap;

pub fn hash2i(a: i32, b: i32) -> f32 {
    let mut h = a
        .wrapping_mul(374761393)
        .wrapping_add(b.wrapping_mul(668265263));
    h = (h ^ (h >> 13)).wrapping_mul(1274126177);
    ((h ^ (h >> 16)) as u32 as f32) / (u32::MAX as f32)
}

#[inline]
pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k.max(1e-5);
    a.min(b) - h * h * k * 0.25
}

#[derive(Clone, Copy, Debug)]
pub struct ParamColor {
    pub rgb: [f32; 3],
    pub mat_id: f32,
    pub glow: f32,
}

pub struct ParamVolume {
    pub detail: u32,
    pub w: usize,
    pub h: usize,
    pub d: usize,
    pub data: Vec<u8>,
    pub palette: Vec<ParamColor>,
    lookup: HashMap<(u32, i32), u8>,
}

impl ParamVolume {
    pub fn new(units_x: f32, units_y: f32, units_z: f32, detail: u32) -> Self {
        let s = detail.clamp(1, 4);
        let w = ((units_x * s as f32).round() as usize).max(1);
        let h = ((units_y * s as f32).round() as usize).max(1);
        let d = ((units_z * s as f32).round() as usize).max(1);
        let mut palette = Vec::with_capacity(64);
        palette.push(ParamColor {
            rgb: [0.0, 0.0, 0.0],
            mat_id: 0.0,
            glow: 0.0,
        });
        Self {
            detail: s,
            w,
            h,
            d,
            data: vec![0u8; w * h * d],
            palette,
            lookup: HashMap::new(),
        }
    }

    pub fn color_mat(&mut self, hex: u32, mat_id: f32) -> u8 {
        let key = (hex, (mat_id * 10.0).round() as i32);
        if let Some(&idx) = self.lookup.get(&key) {
            return idx;
        }
        if self.palette.len() >= 255 {
            return 254;
        }
        let idx = self.palette.len() as u8;
        let glow = if (mat_id - 5.0).abs() < 0.2 { 0.95 } else { 0.0 };
        self.palette.push(ParamColor {
            rgb: hex_to_rgb(hex),
            mat_id,
            glow,
        });
        self.lookup.insert(key, idx);
        idx
    }

    pub fn color(&mut self, hex: u32) -> u8 {
        self.color_mat(hex, 0.0)
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        (y * self.d + z) * self.w + x
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> u8 {
        if x < 0
            || y < 0
            || z < 0
            || x >= self.w as i32
            || y >= self.h as i32
            || z >= self.d as i32
        {
            return 0;
        }
        self.data[self.idx(x as usize, y as usize, z as usize)]
    }

    #[inline]
    pub fn is_surface(&self, x: usize, y: usize, z: usize) -> bool {
        let xi = x as i32;
        let yi = y as i32;
        let zi = z as i32;
        if self.get(xi, yi, zi) == 0 {
            return false;
        }
        self.get(xi + 1, yi, zi) == 0
            || self.get(xi - 1, yi, zi) == 0
            || self.get(xi, yi + 1, zi) == 0
            || self.get(xi, yi - 1, zi) == 0
            || self.get(xi, yi, zi + 1) == 0
            || self.get(xi, yi, zi - 1) == 0
    }

    pub fn region<F>(
        &mut self,
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        c: u8,
        mut test: F,
    ) where
        F: FnMut(f32, f32, f32) -> bool,
    {
        let k = self.detail as f32;
        let ax = (x0 * k + 1e-5).floor().max(0.0) as usize;
        let ay = (y0 * k + 1e-5).floor().max(0.0) as usize;
        let az = (z0 * k + 1e-5).floor().max(0.0) as usize;
        let bx = ((x1 * k - 1e-5).ceil().max(0.0) as usize).min(self.w);
        let by = ((y1 * k - 1e-5).ceil().max(0.0) as usize).min(self.h);
        let bz = ((z1 * k - 1e-5).ceil().max(0.0) as usize).min(self.d);
        for y in ay..by {
            let uy = (y as f32 + 0.5) / k;
            for z in az..bz {
                let uz = (z as f32 + 0.5) / k;
                let row = (y * self.d + z) * self.w;
                for x in ax..bx {
                    let ux = (x as f32 + 0.5) / k;
                    if test(ux, uy, uz) {
                        self.data[row + x] = c;
                    }
                }
            }
        }
    }

    pub fn sculpt<F>(
        &mut self,
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        mut paint: F,
    ) where
        F: FnMut(f32, f32, f32) -> u8,
    {
        let k = self.detail as f32;
        let ax = (x0 * k + 1e-5).floor().max(0.0) as usize;
        let ay = (y0 * k + 1e-5).floor().max(0.0) as usize;
        let az = (z0 * k + 1e-5).floor().max(0.0) as usize;
        let bx = ((x1 * k - 1e-5).ceil().max(0.0) as usize).min(self.w);
        let by = ((y1 * k - 1e-5).ceil().max(0.0) as usize).min(self.h);
        let bz = ((z1 * k - 1e-5).ceil().max(0.0) as usize).min(self.d);
        for y in ay..by {
            let uy = (y as f32 + 0.5) / k;
            for z in az..bz {
                let uz = (z as f32 + 0.5) / k;
                let row = (y * self.d + z) * self.w;
                for x in ax..bx {
                    let ux = (x as f32 + 0.5) / k;
                    let c = paint(ux, uy, uz);
                    if c != 0 {
                        self.data[row + x] = c;
                    }
                }
            }
        }
    }

    pub fn carve<F>(
        &mut self,
        x0: f32,
        y0: f32,
        z0: f32,
        x1: f32,
        y1: f32,
        z1: f32,
        test: F,
    ) where
        F: FnMut(f32, f32, f32) -> bool,
    {
        self.region(x0, y0, z0, x1, y1, z1, 0, test);
    }

    pub fn box_f(&mut self, x: f32, y: f32, z: f32, sx: f32, sy: f32, sz: f32, c: u8) {
        self.region(x, y, z, x + sx, y + sy, z + sz, c, |_, _, _| true);
    }

    pub fn rounded_box(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        sx: f32,
        sy: f32,
        sz: f32,
        radius: f32,
        c: u8,
    ) {
        let r = radius.clamp(0.0, (sx * 0.5).min(sz * 0.5));
        let cx = x + sx * 0.5;
        let cz = z + sz * 0.5;
        let hx = sx * 0.5 - r;
        let hz = sz * 0.5 - r;
        self.region(x, y, z, x + sx, y + sy, z + sz, c, |px, _, pz| {
            let dx = ((px - cx).abs() - hx).max(0.0);
            let dz = ((pz - cz).abs() - hz).max(0.0);
            dx * dx + dz * dz <= r * r
        });
    }

    pub fn ellipsoid(&mut self, cx: f32, cy: f32, cz: f32, rx: f32, ry: f32, rz: f32, c: u8) {
        self.region(
            cx - rx,
            cy - ry,
            cz - rz,
            cx + rx,
            cy + ry,
            cz + rz,
            c,
            |x, y, z| {
                let dx = (x - cx) / rx.max(1e-4);
                let dy = (y - cy) / ry.max(1e-4);
                let dz = (z - cz) / rz.max(1e-4);
                dx * dx + dy * dy + dz * dz <= 1.0
            },
        );
    }

    pub fn column(&mut self, cx: f32, y0: f32, cz: f32, height: f32, r0: f32, r1: f32, c: u8) {
        let r_max = r0.max(r1);
        self.region(
            cx - r_max,
            y0,
            cz - r_max,
            cx + r_max,
            y0 + height,
            cz + r_max,
            c,
            |x, y, z| {
                let t = if height <= 1e-4 {
                    0.0
                } else {
                    ((y - y0) / height).clamp(0.0, 1.0)
                };
                let r = r0 + (r1 - r0) * t;
                let dx = x - cx;
                let dz = z - cz;
                dx * dx + dz * dz <= r * r
            },
        );
    }

    /// O(N) 3-Axis Separable Running-Sum Normal Smoother with 2x top-face weighting for MOBA clarity.
    pub fn compute_smooth_normals(&self) -> Vec<[f32; 3]> {
        let total = self.w * self.h * self.d;
        let mut field = vec![[0.0f32; 3]; total];
        for y in 0..self.h {
            for z in 0..self.d {
                for x in 0..self.w {
                    if !self.is_surface(x, y, z) {
                        continue;
                    }
                    let xi = x as i32;
                    let yi = y as i32;
                    let zi = z as i32;
                    let open = |dx, dy, dz| if self.get(xi + dx, yi + dy, zi + dz) != 0 { 0.0 } else { 1.0 };
                    let nx = open(1, 0, 0) - open(-1, 0, 0);
                    let ny = 2.0 * open(0, 1, 0) - 0.5 * open(0, -1, 0);
                    let nz = open(0, 0, 1) - open(0, 0, -1);
                    field[self.idx(x, y, z)] = [nx, ny, nz];
                }
            }
        }

        let radius = self.detail.max(1) as usize;
        let mut tmp = vec![[0.0f32; 3]; total];
        let dims = [self.w, self.d, self.h];
        let strides = [1usize, self.w, self.w * self.d];

        let pass = |src: &[[f32; 3]], dst: &mut [[f32; 3]], axis: usize| {
            let size = dims[axis];
            let st = strides[axis];
            let (o1, o2) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            for a1 in 0..dims[o1] {
                for a2 in 0..dims[o2] {
                    let base = a1 * strides[o1] + a2 * strides[o2];
                    let mut sx = 0.0f32;
                    let mut sy = 0.0f32;
                    let mut sz = 0.0f32;
                    for q in 0..=radius.min(size - 1) {
                        let v = src[base + q * st];
                        sx += v[0];
                        sy += v[1];
                        sz += v[2];
                    }
                    for q in 0..size {
                        dst[base + q * st] = [sx, sy, sz];
                        let add = q + radius + 1;
                        if add < size {
                            let v = src[base + add * st];
                            sx += v[0];
                            sy += v[1];
                            sz += v[2];
                        }
                        if q >= radius {
                            let v = src[base + (q - radius) * st];
                            sx -= v[0];
                            sy -= v[1];
                            sz -= v[2];
                        }
                    }
                }
            }
        };

        pass(&field, &mut tmp, 0);
        pass(&tmp, &mut field, 1);
        pass(&field, &mut tmp, 2);

        for v in &mut tmp {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if len < 1e-4 {
                *v = [0.0, 1.0, 0.0];
            } else {
                v[0] /= len;
                v[1] /= len;
                v[2] /= len;
            }
        }
        tmp
    }

    /// Compile ParamVolume into a Greedy-Merged 3D Triangle Mesh with baked 4-level voxel AO + smooth normals!
    pub fn compile_to_mesh(
        &self,
        verts: &mut Vec<MeshVertex>,
        idxs: &mut Vec<u32>,
        unit_scale: f32,
        center_xz: bool,
        smooth_blend: f32,
    ) {
        let normals = self.compute_smooth_normals();
        let vs = unit_scale / self.detail as f32;
        let ox = if center_xz { -(self.w as f32 * 0.5) * vs } else { 0.0 };
        let oz = if center_xz { -(self.d as f32 * 0.5) * vs } else { 0.0 };

        #[derive(Clone, Copy, PartialEq, Eq)]
        struct FaceKey {
            col_id: u8,
            ao_packed: u8,
            n_q: [i8; 3],
        }

        let dirs: [(usize, i32, usize, usize); 6] = [
            (0, 1, 2, 1),
            (0, -1, 2, 1),
            (1, 1, 0, 2),
            (1, -1, 0, 2),
            (2, 1, 0, 1),
            (2, -1, 0, 1),
        ];
        let dims = [self.w, self.h, self.d];

        for &(axis, sign, u_ax, v_ax) in &dirs {
            let n_slices = dims[axis];
            let nu = dims[u_ax];
            let nv = dims[v_ax];
            let mut mask: Vec<Option<FaceKey>> = vec![None; nu * nv];
            let mut face_n = Vec3::ZERO;
            face_n[axis] = sign as f32;

            for slice in 0..n_slices {
                mask.fill(None);
                for v in 0..nv {
                    for u in 0..nu {
                        let mut c = [0i32; 3];
                        c[axis] = slice as i32;
                        c[u_ax] = u as i32;
                        c[v_ax] = v as i32;
                        let col_id = self.get(c[0], c[1], c[2]);
                        if col_id == 0 {
                            continue;
                        }
                        let mut nb = c;
                        nb[axis] += sign;
                        if self.get(nb[0], nb[1], nb[2]) != 0 {
                            continue;
                        }

                        let occ = |du: i32, dv: i32| -> bool {
                            let mut q = nb;
                            q[u_ax] += du;
                            q[v_ax] += dv;
                            self.get(q[0], q[1], q[2]) != 0
                        };
                        let ao_lvl = |du: i32, dv: i32| -> u8 {
                            let s1 = occ(du, 0);
                            let s2 = occ(0, dv);
                            let cr = occ(du, dv);
                            if s1 && s2 {
                                3
                            } else {
                                (s1 as u8) + (s2 as u8) + (cr as u8)
                            }
                        };
                        let ao_packed = ao_lvl(-1, -1)
                            | (ao_lvl(1, -1) << 2)
                            | (ao_lvl(1, 1) << 4)
                            | (ao_lvl(-1, 1) << 6);

                        let sn = normals[self.idx(c[0] as usize, c[1] as usize, c[2] as usize)];
                        let bn = (face_n * (1.0 - smooth_blend)
                            + Vec3::from_array(sn) * smooth_blend)
                            .normalize_or(face_n);
                        let n_q = [
                            (bn.x * 6.0).round() as i8,
                            (bn.y * 6.0).round() as i8,
                            (bn.z * 6.0).round() as i8,
                        ];

                        mask[v * nu + u] = Some(FaceKey {
                            col_id,
                            ao_packed,
                            n_q,
                        });
                    }
                }

                for v in 0..nv {
                    let mut u = 0;
                    while u < nu {
                        let Some(key) = mask[v * nu + u] else {
                            u += 1;
                            continue;
                        };
                        let mut w_span = 1usize;
                        while u + w_span < nu && mask[v * nu + u + w_span] == Some(key) {
                            w_span += 1;
                        }
                        let mut h_span = 1usize;
                        'v_grow: while v + h_span < nv {
                            for du in 0..w_span {
                                if mask[(v + h_span) * nu + u + du] != Some(key) {
                                    break 'v_grow;
                                }
                            }
                            h_span += 1;
                        }
                        for dv in 0..h_span {
                            for du in 0..w_span {
                                mask[(v + dv) * nu + u + du] = None;
                            }
                        }

                        let pc = self.palette[key.col_id as usize];
                        let n = Vec3::new(key.n_q[0] as f32, key.n_q[1] as f32, key.n_q[2] as f32)
                            .normalize_or(face_n);
                        let ao_val = |lvl: u8| match lvl {
                            0 => 1.0f32,
                            1 => 0.87,
                            2 => 0.74,
                            _ => 0.60,
                        };
                        let a00 = ao_val(key.ao_packed & 3);
                        let a10 = ao_val((key.ao_packed >> 2) & 3);
                        let a11 = ao_val((key.ao_packed >> 4) & 3);
                        let a01 = ao_val((key.ao_packed >> 6) & 3);

                        let mk = |du: usize, dv: usize| -> Vec3 {
                            let mut p = [0.0f32; 3];
                            p[axis] = (slice as f32 + if sign > 0 { 1.0 } else { 0.0 }) * vs;
                            p[u_ax] = (u + du) as f32 * vs;
                            p[v_ax] = (v + dv) as f32 * vs;
                            Vec3::new(p[0] + ox, p[1], p[2] + oz)
                        };

                        let p00 = mk(0, 0);
                        let p10 = mk(w_span, 0);
                        let p11 = mk(w_span, h_span);
                        let p01 = mk(0, h_span);

                        let cross = (p10 - p00).cross(p11 - p00);
                        let flip = cross.dot(face_n) < 0.0;

                        let base = verts.len() as u32;
                        let corners = if !flip {
                            [(p00, a00), (p10, a10), (p11, a11), (p01, a01)]
                        } else {
                            [(p00, a00), (p01, a01), (p11, a11), (p10, a10)]
                        };
                        for (pos, ao) in corners {
                            verts.push(MeshVertex {
                                pos: pos.to_array(),
                                normal: n.to_array(),
                                color_mat: [
                                    pc.rgb[0] * ao,
                                    pc.rgb[1] * ao,
                                    pc.rgb[2] * ao,
                                    pc.mat_id,
                                ],
                                extra: [0.0; 4],
                            });
                        }
                        if corners[0].1 + corners[2].1 >= corners[1].1 + corners[3].1 {
                            idxs.extend_from_slice(&[
                                base,
                                base + 1,
                                base + 2,
                                base,
                                base + 2,
                                base + 3,
                            ]);
                        } else {
                            idxs.extend_from_slice(&[
                                base + 1,
                                base + 2,
                                base + 3,
                                base + 1,
                                base + 3,
                                base,
                            ]);
                        }

                        u += w_span;
                    }
                }
            }
        }
    }

    /// Compile smooth-min organic ParamVolume surface voxels directly into GPU-instanced Gaussian Surfels!
    pub fn compile_to_surfels(
        &self,
        out: &mut Vec<SplatProto>,
        unit_scale: f32,
        y_offset: f32,
        stride: usize,
        radius_mul: f32,
        flags: f32,
    ) {
        let normals = self.compute_smooth_normals();
        let vs = unit_scale / self.detail as f32;
        let ox = -(self.w as f32 * 0.5) * vs;
        let oz = -(self.d as f32 * 0.5) * vs;
        let sr_base = vs * 0.88 * (stride as f32).sqrt() * radius_mul;

        let mut counter = 0usize;
        for y in 0..self.h {
            for z in 0..self.d {
                for x in 0..self.w {
                    if !self.is_surface(x, y, z) {
                        continue;
                    }
                    let c_id = self.data[self.idx(x, y, z)];
                    let pc = self.palette[c_id as usize];
                    let keep = pc.glow > 0.05 || (counter % stride.max(1) == 0);
                    counter += 1;
                    if !keep {
                        continue;
                    }

                    let n = normals[self.idx(x, y, z)];
                    let j0 = hash2i(x as i32 * 13 + y as i32 * 7, z as i32 * 19 + 3) - 0.5;
                    let j1 = hash2i(x as i32 * 23 - z as i32 * 5, y as i32 * 11 + 7) - 0.5;
                    let px = (x as f32 + 0.5) * vs + ox + j0 * vs * 0.35;
                    let py = (y as f32 + 0.5) * vs + y_offset + j1 * vs * 0.30;
                    let pz = (z as f32 + 0.5) * vs + oz - j0 * vs * 0.35;
                    let sr = if pc.glow > 0.05 {
                        vs * 0.78
                    } else {
                        sr_base * (0.90 + 0.22 * (j1 + 0.5))
                    };

                    out.push(SplatProto {
                        local_pos_rad: [px, py, pz, sr],
                        local_norm_flags: [n[0], n[1], n[2], flags],
                        color_glow: [pc.rgb[0], pc.rgb[1], pc.rgb[2], pc.glow],
                    });
                }
            }
        }
    }
}
