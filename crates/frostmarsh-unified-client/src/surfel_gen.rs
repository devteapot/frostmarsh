//! Smooth-Min SDF Hard-Z GPU-Instanced Foliage & Wildfire Surfel Prototypes (`compile_to_surfels()`)

use crate::palette::hex_to_rgb;
use crate::parametric::{smooth_min, ParamVolume};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::PI;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatProto {
    pub local_pos_rad: [f32; 4],    // xyz = local offset, w = radius
    pub local_norm_flags: [f32; 4], // xyz = unit normal, w = flags (1=two_sided, 2=wind_sway, 3=both)
    pub color_glow: [f32; 4],       // rgb = albedo, w = glow
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatPlacement {
    pub pos_yaw: [f32; 4],    // camera-relative xyz, w = yaw
    pub tint_scale: [f32; 4], // rgb = tint, w = scale
    pub lod_phase: [f32; 4],  // x = lod_radius_mul, y = wind_phase, z = glow_mul, w = unused
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatRef {
    pub placement_idx: u32,
    pub proto_idx: u32,
}

pub const SURFEL_PROTO_COUNT: usize = 7;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SurfelProtoKind {
    PineCanopy = 0,
    SnowPine = 1,
    MarshBrush = 2,
    RiverReeds = 3,
    WildfireFlame = 4,
    SmokePlume = 5,
    ArcaneWisp = 6,
}

#[derive(Clone, Copy, Debug)]
pub struct ProtoRange {
    pub start: u32,
    pub count: u32,
}

pub struct SurfelProtoBank {
    pub protos: Vec<SplatProto>,
    pub ranges: [ProtoRange; SURFEL_PROTO_COUNT],
}

fn hash1(n: usize) -> f32 {
    let mut x = (n as u32)
        .wrapping_mul(747796405)
        .wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    ((x ^ (x >> 22)) as f32) / (u32::MAX as f32)
}

fn add_lobe(
    out: &mut Vec<SplatProto>,
    center: Vec3,
    rad_xz: f32,
    rad_y: f32,
    count: usize,
    shadow_col: [f32; 3],
    mid_col: [f32; 3],
    hi_col: [f32; 3],
    glow: f32,
    flags: f32,
    seed: usize,
) {
    let avg_r = (rad_xz + rad_y) * 0.5;
    let area = 4.0 * PI * avg_r * avg_r;
    let sr_base = (area / count.max(1) as f32).sqrt() * 0.78;
    let golden = PI * (3.0 - 5.0_f32.sqrt());
    for i in 0..count {
        let y = 1.0 - (i as f32 / (count - 1).max(1) as f32) * 2.0;
        let r_ring = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f32 + seed as f32 * 0.7;
        let h0 = hash1(seed * 997 + i * 13 + 7);
        let h1 = hash1(seed * 613 + i * 29 + 19);
        let dir = Vec3::new(theta.cos() * r_ring, y, theta.sin() * r_ring);
        let p = center
            + Vec3::new(
                dir.x * rad_xz * (0.86 + 0.28 * h0),
                dir.y * rad_y * (0.88 + 0.22 * h1),
                dir.z * rad_xz * (0.86 + 0.28 * h0),
            );
        let n = (dir + Vec3::Y * 0.25).normalize_or(Vec3::Y);
        let top_t = ((y * 0.5 + 0.5) * 0.85 + h1 * 0.20).clamp(0.0, 1.0);
        let c = if top_t < 0.5 {
            let t = top_t * 2.0;
            [
                shadow_col[0] + (mid_col[0] - shadow_col[0]) * t,
                shadow_col[1] + (mid_col[1] - shadow_col[1]) * t,
                shadow_col[2] + (mid_col[2] - shadow_col[2]) * t,
            ]
        } else {
            let t = (top_t - 0.5) * 2.0;
            [
                mid_col[0] + (hi_col[0] - mid_col[0]) * t,
                mid_col[1] + (hi_col[1] - mid_col[1]) * t,
                mid_col[2] + (hi_col[2] - mid_col[2]) * t,
            ]
        };
        out.push(SplatProto {
            local_pos_rad: [p.x, p.y, p.z, sr_base * (0.84 + 0.30 * h1)],
            local_norm_flags: [n.x, n.y, n.z, flags],
            color_glow: [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0), glow],
        });
    }
}

/// Scalloped-Tier Conifer Pine compiled via smooth-min SDF ParamVolume -> Hard-Z Surfels!
fn build_scalloped_pine(out: &mut Vec<SplatProto>, snowy: bool) {
    let mut v = ParamVolume::new(12.0, 16.0, 12.0, 1);
    let (dark, mid, hi) = if snowy {
        (v.color(0x1e3c52), v.color(0x6c9eb8), v.color(0xf0f8ff))
    } else {
        (v.color(0x122d29), v.color(0x286442), v.color(0x7cc956))
    };
    let c = 6.0f32;
    let tiers: [(f32, f32, f32); 4] = [
        (0.8, 4.4, 5.0),
        (3.8, 4.2, 4.2),
        (6.8, 4.0, 3.2),
        (9.6, 4.2, 2.0),
    ];
    for (ti, &(y0, h, r_max)) in tiers.iter().enumerate() {
        v.sculpt(0.0, y0, 0.0, 12.0, y0 + h, 12.0, |x, y, z| {
            let dx = x - c;
            let dz = z - c;
            let t = ((y - y0) / h).clamp(0.0, 1.0);
            let a = dz.atan2(dx);
            let d = dx.hypot(dz);
            let wave = (a * 7.0 + ti as f32 * 1.3).sin();
            let rr = r_max * (1.0 + 0.14 * wave) * (1.0 - t).powf(1.12);
            if d > rr {
                return 0;
            }
            if t < 0.18 {
                return dark;
            }
            if snowy {
                if t > 0.34 || wave > 0.18 {
                    return hi;
                }
                return mid;
            }
            if wave > 0.42 && d > rr - 1.1 {
                hi
            } else {
                mid
            }
        });
    }
    v.compile_to_surfels(out, 0.30, 1.05, 2, 0.80, 3.0);
}

/// Smooth-Min SDF Marsh Brush Clump compiled via `ParamVolume::compile_to_surfels()`!
fn build_smoothmin_marsh_brush(out: &mut Vec<SplatProto>) {
    let mut v = ParamVolume::new(10.0, 7.0, 10.0, 1);
    let leaf_dark = v.color(0x193f2e);
    let leaf_mid = v.color(0x3a8245);
    let leaf_hi = v.color(0x88cc58);
    let berry = v.color_mat(0xffe978, 5.0);

    let (cx, cz) = (5.0f32, 5.0f32);
    let blobs: [(f32, f32, f32, f32, f32); 4] = [
        (0.0, 2.5, 0.0, 2.7, 2.1),
        (1.8, 2.1, 1.1, 1.9, 1.6),
        (-1.7, 2.0, 0.9, 1.8, 1.5),
        (-0.5, 2.2, -1.7, 1.9, 1.6),
    ];
    let field = |x: f32, y: f32, z: f32| -> f32 {
        let mut f = f32::INFINITY;
        for &(bx, by, bz, rx, ry) in &blobs {
            let e = (((x - cx - bx) / rx).powi(2)
                + ((y - by) / ry).powi(2)
                + ((z - cz - bz) / rx).powi(2))
            .sqrt();
            let dist = (e - 1.0) * rx.min(ry);
            f = if f.is_infinite() {
                dist
            } else {
                smooth_min(f, dist, 0.95)
            };
        }
        f
    };

    v.sculpt(0.0, 0.0, 0.0, 10.0, 7.0, 10.0, |x, y, z| {
        if field(x, y, z) > 0.0 {
            return 0;
        }
        if field(x, y - 0.85, z) > 0.0 {
            return leaf_dark;
        }
        if field(x, y + 0.85, z) > 0.0 && y > 2.6 {
            return leaf_hi;
        }
        leaf_mid
    });
    v.ellipsoid(cx + 1.1, 3.6, cz + 0.6, 0.45, 0.45, 0.45, berry);
    v.ellipsoid(cx - 1.2, 3.3, cz - 0.5, 0.42, 0.42, 0.42, berry);

    v.compile_to_surfels(out, 0.24, 0.05, 2, 0.82, 3.0);
}

impl SurfelProtoBank {
    pub fn build() -> Self {
        let mut protos = Vec::with_capacity(4_000);
        let mut make = |f: &mut dyn FnMut(&mut Vec<SplatProto>)| -> ProtoRange {
            let start = protos.len() as u32;
            f(&mut protos);
            ProtoRange {
                start,
                count: protos.len() as u32 - start,
            }
        };

        let pine = make(&mut |o| build_scalloped_pine(o, false));
        let snow_pine = make(&mut |o| build_scalloped_pine(o, true));
        let brush = make(&mut |o| build_smoothmin_marsh_brush(o));

        let reeds = make(&mut |o| {
            let reed_d = hex_to_rgb(0x275b36);
            let reed_l = hex_to_rgb(0x64a84f);
            let cattail = hex_to_rgb(0x633722);
            for k in 0..10 {
                let a = k as f32 * 2.3999;
                let r = 0.12 + hash1(700 + k) * 0.72;
                let bx = a.cos() * r;
                let bz = a.sin() * r;
                let h = 0.75 + hash1(730 + k) * 0.55;
                for s in 0..3 {
                    let t = (s as f32 + 0.5) / 3.0;
                    let col = if s == 2 && k % 3 == 0 {
                        cattail
                    } else if s >= 1 {
                        reed_l
                    } else {
                        reed_d
                    };
                    o.push(SplatProto {
                        local_pos_rad: [
                            bx + t * 0.10 * a.cos(),
                            t * h,
                            bz + t * 0.10 * a.sin(),
                            0.14,
                        ],
                        local_norm_flags: [a.cos() * 0.3, 0.95, a.sin() * 0.3, 3.0],
                        color_glow: [col[0], col[1], col[2], 0.0],
                    });
                }
            }
        });

        let wildfire = make(&mut |o| {
            // Sculpted blazing wildfire tongues (rich crimson base -> vibrant orange -> incandescent gold tip!)
            add_lobe(
                o,
                Vec3::new(0.0, 0.72, 0.0),
                0.62,
                0.76,
                22,
                hex_to_rgb(0xb8241c),
                hex_to_rgb(0xf26424),
                hex_to_rgb(0xf7a428),
                0.96,
                3.0,
                51,
            );
            add_lobe(
                o,
                Vec3::new(0.0, 1.55, 0.0),
                0.38,
                0.88,
                18,
                hex_to_rgb(0xf47e28),
                hex_to_rgb(0xffd242),
                hex_to_rgb(0xfff8b8),
                1.0,
                3.0,
                52,
            );
            // Tapered flame crown + rising glowing ember sparks
            for k in 0..12 {
                let a = k as f32 * 2.1;
                let h = 2.0 + (k as f32) * 0.24;
                let r = 0.12 + hash1(500 + k) * 0.42;
                let col = if k % 3 == 0 {
                    hex_to_rgb(0xfff6a8)
                } else if k % 3 == 1 {
                    hex_to_rgb(0xff9a2e)
                } else {
                    hex_to_rgb(0xe84228)
                };
                o.push(SplatProto {
                    local_pos_rad: [a.cos() * r, h, a.sin() * r, 0.10 + hash1(550 + k) * 0.06],
                    local_norm_flags: [0.0, 1.0, 0.0, 3.0],
                    color_glow: [col[0], col[1], col[2], 1.0],
                });
            }
        });

        // Wispy, sparse ember-smoke motes (never opaque marshmallow spheres!)
        let smoke = make(&mut |o| {
            for k in 0..10 {
                let a = k as f32 * 2.3999;
                let t = k as f32 / 10.0;
                let r = 0.14 + t * 0.48;
                let h = 1.6 + t * 2.2;
                let col = if k % 3 == 0 {
                    hex_to_rgb(0x52485c)
                } else {
                    hex_to_rgb(0x322b3e)
                };
                o.push(SplatProto {
                    local_pos_rad: [a.cos() * r, h, a.sin() * r, 0.16 + hash1(630 + k) * 0.10],
                    local_norm_flags: [0.0, 1.0, 0.0, 3.0],
                    color_glow: [col[0], col[1], col[2], 0.0],
                });
            }
        });

        let wisp = make(&mut |o| {
            add_lobe(
                o,
                Vec3::ZERO,
                0.24,
                0.24,
                14,
                hex_to_rgb(0x4fb8de),
                hex_to_rgb(0x8be8ff),
                hex_to_rgb(0xd8fbff),
                1.0,
                3.0,
                61,
            );
        });

        Self {
            protos,
            ranges: [pine, snow_pine, brush, reeds, wildfire, smoke, wisp],
        }
    }

    pub fn range(&self, kind: SurfelProtoKind) -> ProtoRange {
        self.ranges[kind as usize]
    }
}
