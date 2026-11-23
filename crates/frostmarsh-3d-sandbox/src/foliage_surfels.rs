//! Iteration 9A: Hard-Z GPU-Instanced Sculpted Pine Canopy & Snow-Pine Surfel Generator
//!
//! Adapted directly from Frostmarsh's signature Approach 4 scalloped-tier conifer SDF surfels!
//! Places lush towering Nordic Pines (`PineCanopy`) and high-elevation alpine Snow Pines (`SnowPine`)
//! cleanly along true 3D `solid_density` mountain slopes & ridges, replacing smooth voxel tree bumps!

use crate::volume::WorldVolume3D;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::TAU;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatProto {
    pub local_pos_rad: [f32; 4],    // xyz = local offset (meters), w = surfel radius
    pub local_norm_flags: [f32; 4], // xyz = unit normal, w = flags (bit0=two_sided, bit1=wind, bit2=snow, bit3=bark)
    pub color_glow: [f32; 4],       // rgb = base Nordic sRGB albedo, w = glow
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatPlacement {
    pub pos_yaw: [f32; 4],    // xyz = world root base pos, w = yaw radians
    pub tint_scale: [f32; 4], // rgb = per-tree subtle tint, w = uniform scale
    pub lod_phase: [f32; 4],  // x = rad_mul, y = wind_phase, z = snow_factor (0..1), w = unused
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SplatRef {
    pub placement_idx: u32,
    pub proto_idx: u32,
}

#[derive(Clone, Copy)]
struct ProtoRange {
    start: u32,
    count: u32,
}

pub struct PineSurfelScene {
    pub protos: Vec<SplatProto>,
    pub placements: Vec<SplatPlacement>,
    pub draw_refs: Vec<SplatRef>,
    pub pine_count: usize,
    pub snow_pine_count: usize,
}

#[inline]
fn hex_to_rgb(hex: u32) -> [f32; 3] {
    [
        ((hex >> 16) & 0xff) as f32 / 255.0,
        ((hex >> 8) & 0xff) as f32 / 255.0,
        (hex & 0xff) as f32 / 255.0,
    ]
}

#[inline]
fn hash1(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747796405).wrapping_add(2891336453);
    x = ((x >> ((x >> 28).wrapping_add(4))) ^ x).wrapping_mul(277803737);
    ((x ^ (x >> 22)) as f32) / (u32::MAX as f32)
}

/// Build a scalloped 4-tier Nordic Conifer Pine (or Snow-Pine) + tapered bark trunk directly as
/// crisp Hard-Z GPU-instanced surfels (matching Frostmarsh's signature silhouette!).
fn build_sculpted_pine_proto(out: &mut Vec<SplatProto>, snowy: bool) -> ProtoRange {
    let start = out.len() as u32;

    let bark_dark = hex_to_rgb(0x3a241c);
    let bark_mid = hex_to_rgb(0x5a3828);
    // Tapered sturdy bark trunk from y=0.0 to y=1.55m
    for k in 0..14u32 {
        let t = k as f32 / 13.0;
        let y = 0.12 + t * 1.45;
        let a = (k as f32) * 2.3999;
        let r = 0.24 * (1.0 - 0.35 * t);
        let col = if k % 2 == 0 { bark_dark } else { bark_mid };
        out.push(SplatProto {
            local_pos_rad: [a.cos() * r * 0.45, y, a.sin() * r * 0.45, 0.26 * (1.0 - 0.2 * t)],
            local_norm_flags: [a.cos(), 0.12, a.sin(), 9.0], // bit0=two_sided | bit3=bark
            color_glow: [col[0], col[1], col[2], 0.0],
        });
    }

    let (dark, mid, hi) = if snowy {
        (hex_to_rgb(0x1b3a50), hex_to_rgb(0x6c9eb8), hex_to_rgb(0xf0f8ff))
    } else {
        (hex_to_rgb(0x112a26), hex_to_rgb(0x286442), hex_to_rgb(0x7cc956))
    };

    // 4 overlapping scalloped conical bough tiers (total height ~5.2m at scale=1.0)
    let tiers: [(f32, f32, f32, usize); 4] = [
        (1.05, 1.50, 1.65, 5),
        (2.00, 1.45, 1.35, 5),
        (2.95, 1.35, 1.02, 4),
        (3.85, 1.35, 0.64, 4),
    ];

    let base_flags = if snowy { 7.0 } else { 3.0 }; // two_sided | wind | (snow)

    for (ti, &(y0, h, r_max, rings)) in tiers.iter().enumerate() {
        for ri in 0..rings {
            let t = (ri as f32 + 0.35) / rings as f32;
            let y = y0 + t * h;
            let ring_r_base = r_max * (1.0 - t * 0.86).powf(1.08);
            let segs = ((ring_r_base * 9.0).round() as usize).clamp(5, 14);
            for si in 0..segs {
                let seed = (ti as u32) * 997 + (ri as u32) * 131 + (si as u32) * 17 + if snowy { 5000 } else { 0 };
                let j0 = hash1(seed) - 0.5;
                let j1 = hash1(seed + 71) - 0.5;
                let a = (si as f32 / segs as f32) * TAU + (ri as f32) * 0.48 + (ti as f32) * 0.85;
                let wave = (a * 7.0 + ti as f32 * 1.3).sin();
                let r = ring_r_base * (1.0 + 0.16 * wave) * (0.94 + 0.12 * j0);
                let px = a.cos() * r;
                let pz = a.sin() * r;
                let py = y + j1 * 0.14 - 0.08 * (1.0 - t);

                let n = Vec3::new(a.cos() * 0.82, 0.52 + 0.25 * t, a.sin() * 0.82).normalize();
                let col = if t < 0.22 {
                    dark
                } else if snowy {
                    if t > 0.36 || wave > 0.12 { hi } else { mid }
                } else if wave > 0.32 || t > 0.72 {
                    hi
                } else {
                    mid
                };
                let sr = (0.36 + 0.14 * (1.0 - t)) * (0.92 + 0.16 * (j0 + 0.5));
                out.push(SplatProto {
                    local_pos_rad: [px, py, pz, sr],
                    local_norm_flags: [n.x, n.y, n.z, base_flags],
                    color_glow: [col[0], col[1], col[2], 0.0],
                });
            }
        }
    }
    // Crisp conical spire tip
    let tip_col = if snowy { hi } else { hi };
    out.push(SplatProto {
        local_pos_rad: [0.0, 5.25, 0.0, 0.30],
        local_norm_flags: [0.0, 1.0, 0.0, base_flags],
        color_glow: [tip_col[0], tip_col[1], tip_col[2], 0.0],
    });

    ProtoRange {
        start,
        count: out.len() as u32 - start,
    }
}

/// Top-down terrain solid_density height lookup at `(ix, iz)` (ignoring overhang ceilings if near caves,
/// returning the highest solid surface where `solid_density >= 0.5`).
fn find_surface_y(vol: &WorldVolume3D, ix: usize, iz: usize) -> Option<f32> {
    for iy in (2..vol.depth).rev() {
        let d_cur = vol.solid_f32(ix, iy, iz);
        if d_cur >= 0.5 {
            let d_above = if iy + 1 < vol.depth {
                vol.solid_f32(ix, iy + 1, iz)
            } else {
                0.0
            };
            let frac = ((d_cur - 0.5) / (d_cur - d_above).max(1e-3)).clamp(0.0, 1.0);
            return Some(iy as f32 + 0.5 + frac);
        }
    }
    None
}

impl PineSurfelScene {
    /// Scan the true 3D `WorldVolume3D` (`solid_density`, `fluid_density`, `glow`) and scatter
    /// lush sculpted Nordic Pines & Snow Pines cleanly along the alpine slopes & ridges!
    pub fn build_for_volume(vol: &WorldVolume3D, is_latent_world: bool) -> Self {
        let mut protos = Vec::with_capacity(640);
        let range_pine = build_sculpted_pine_proto(&mut protos, false);
        let range_snow = build_sculpted_pine_proto(&mut protos, true);

        let mut placements = Vec::with_capacity(140);
        let mut draw_refs = Vec::with_capacity(28_000);
        let mut pine_count = 0usize;
        let mut snow_pine_count = 0usize;

        // Deterministic jittered grid across forested mountain slopes (excluding river canyon,
        // bridge deck, cave mouth apron, camera lenses, and active wildfire core!)
        let step = 11usize;
        let mut idx_seed = 101u32;

        for gz in (18..vol.height.saturating_sub(18)).step_by(step) {
            for gx in (18..vol.width.saturating_sub(18)).step_by(step) {
                idx_seed = idx_seed.wrapping_add(1);
                let h0 = hash1(idx_seed * 31 + 7);
                let h1 = hash1(idx_seed * 53 + 13);
                let h2 = hash1(idx_seed * 97 + 29);
                let h3 = hash1(idx_seed * 149 + 41);

                // Keep natural grove clustering (skip ~28% of grid cells)
                if h0 < 0.28 {
                    continue;
                }

                let wx = (gx as f32 + (h1 - 0.5) * (step as f32 * 0.72)).clamp(12.0, (vol.width - 13) as f32);
                let wz = (gz as f32 + (h2 - 0.5) * (step as f32 * 0.72)).clamp(12.0, (vol.height - 13) as f32);
                let ix = wx as usize;
                let iz = wz as usize;

                // Keep bridge corridor, river canyon center, waterfall splash, and camera foreground clear!
                if wx > 86.0 && wx < 170.0 && wz > 116.0 && wz < 138.0 {
                    continue; // Arched bridge span + approach pads + cave mouth
                }
                if wx > 116.0 && wx < 142.0 {
                    continue; // Main flowing river canyon channel
                }
                if is_latent_world {
                    // Keep plunging 3D waterfall sightline (X=135..178, Z=54..116) and bridge camera clear!
                    if wx > 135.0 && wx < 178.0 && wz > 54.0 && wz < 116.0 {
                        continue;
                    }
                    if (wx - 102.5).hypot(wz - 163.5) < 16.0 {
                        continue; // Bridge camera clearance
                    }
                } else {
                    if (wx - 107.5).hypot(wz - 153.5) < 16.0 || (wx - 111.5).hypot(wz - 147.2) < 14.0 {
                        continue;
                    }
                }

                let Some(wy) = find_surface_y(vol, ix, iz) else {
                    continue;
                };
                // Valid forested slope elevation band (above riverbank, below vertical ceiling clamp)
                if wy < 4.7 || wy > 11.4 {
                    continue;
                }

                // Ensure not underwater and not inside wildfire/magma heat
                let iy_surf = (wy.floor() as usize).min(vol.depth - 1);
                let iy_up = (iy_surf + 1).min(vol.depth - 1);
                let bi_s = vol.byte_index(ix, iy_surf, iz);
                let bi_u = vol.byte_index(ix, iy_up, iz);
                if vol.density_mat[bi_s + 1] > 35 || vol.density_mat[bi_u + 1] > 35 {
                    continue; // Fluid/river/lake
                }
                if vol.albedo_thermal[bi_s + 3] > 35 || vol.albedo_thermal[bi_u + 3] > 35 {
                    continue; // Wildfire / magma
                }

                // Slope check so pines stand upright on real mountain slopes rather than sheer vertical cliffs
                let hx0 = find_surface_y(vol, ix.saturating_sub(2), iz).unwrap_or(wy);
                let hx1 = find_surface_y(vol, (ix + 2).min(vol.width - 1), iz).unwrap_or(wy);
                let hz0 = find_surface_y(vol, ix, iz.saturating_sub(2)).unwrap_or(wy);
                let hz1 = find_surface_y(vol, ix, (iz + 2).min(vol.height - 1)).unwrap_or(wy);
                if (hx1 - hx0).abs() > 2.4 || (hz1 - hz0).abs() > 2.4 {
                    continue;
                }

                let is_snowy = wy >= 8.75 || (wy >= 8.15 && h3 > 0.55);
                let scale = (0.78 + h3 * 0.44) * if is_snowy { 0.92 } else { 1.04 };
                // Ensure tree apex fits nicely inside camera silhouette
                let tint_v = 0.88 + h1 * 0.22;
                let tint = if is_snowy {
                    [tint_v * 0.96, tint_v * 0.99, (tint_v * 1.04).min(1.12)]
                } else {
                    [tint_v * 0.94, tint_v * 1.02, tint_v * 0.96]
                };

                let pl_idx = placements.len() as u32;
                placements.push(SplatPlacement {
                    pos_yaw: [wx, wy - 0.30, wz, h2 * TAU],
                    tint_scale: [tint[0], tint[1], tint[2], scale],
                    lod_phase: [1.0, h0 * TAU, if is_snowy { 1.0 } else { 0.0 }, 0.0],
                });

                let range = if is_snowy {
                    snow_pine_count += 1;
                    range_snow
                } else {
                    pine_count += 1;
                    range_pine
                };
                for pi in range.start..(range.start + range.count) {
                    draw_refs.push(SplatRef {
                        placement_idx: pl_idx,
                        proto_idx: pi,
                    });
                }
            }
        }

        Self {
            protos,
            placements,
            draw_refs,
            pine_count,
            snow_pine_count,
        }
    }
}
