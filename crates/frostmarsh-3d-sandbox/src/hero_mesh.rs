//! Compact Parametric Armored 3D Hero Knight Figurine at True `1.90m` Isotropic Height.
//!
//! Placed directly into the `256×256×16` scene to prove true physical scale and pixel-exact
//! depth-tested occlusion inside vaulted caves and underneath arched bridge spans.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::TAU;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HeroVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// rgb = base albedo, a = roughness (0.15 polished steel/gold .. 0.85 cloth/leather)
    pub color_rough: [f32; 4],
    /// x = faction_tint_weight (1.0 = tinted by instance faction color), y = emissive
    pub flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HeroInstance {
    /// xyz = world foot position, w = yaw radians
    pub pos_yaw: [f32; 4],
    /// rgb = faction cloak/tabard/plume color, w = scale (1.0 => exact 1.90m height)
    pub faction_scale: [f32; 4],
}

pub struct HeroMesh {
    pub vertices: Vec<HeroVertex>,
    pub indices: Vec<u32>,
}

fn push_quad(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    col: [f32; 3],
    rough: f32,
    tint_w: f32,
) {
    let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Y);
    let base = verts.len() as u32;
    for p in [p0, p1, p2, p3] {
        verts.push(HeroVertex {
            pos: p.to_array(),
            normal: n.to_array(),
            color_rough: [col[0], col[1], col[2], rough],
            flags: [tint_w, 0.0, 0.0, 0.0],
        });
    }
    idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_box(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    min: Vec3,
    max: Vec3,
    col: [f32; 3],
    rough: f32,
    tint_w: f32,
) {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);
    // +Z front
    push_quad(verts, idxs, Vec3::new(x0, y0, z1), Vec3::new(x1, y0, z1), Vec3::new(x1, y1, z1), Vec3::new(x0, y1, z1), col, rough, tint_w);
    // -Z back
    push_quad(verts, idxs, Vec3::new(x1, y0, z0), Vec3::new(x0, y0, z0), Vec3::new(x0, y1, z0), Vec3::new(x1, y1, z0), col, rough, tint_w);
    // +X right
    push_quad(verts, idxs, Vec3::new(x1, y0, z1), Vec3::new(x1, y0, z0), Vec3::new(x1, y1, z0), Vec3::new(x1, y1, z1), col, rough, tint_w);
    // -X left
    push_quad(verts, idxs, Vec3::new(x0, y0, z0), Vec3::new(x0, y0, z1), Vec3::new(x0, y1, z1), Vec3::new(x0, y1, z0), col, rough, tint_w);
    // +Y top
    push_quad(verts, idxs, Vec3::new(x0, y1, z1), Vec3::new(x1, y1, z1), Vec3::new(x1, y1, z0), Vec3::new(x0, y1, z0), col, rough, tint_w);
    // -Y bottom
    push_quad(verts, idxs, Vec3::new(x0, y0, z0), Vec3::new(x1, y0, z0), Vec3::new(x1, y0, z1), Vec3::new(x0, y0, z1), col, rough, tint_w);
}

fn push_cylinder(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    r0: f32,
    r1: f32,
    segs: usize,
    col: [f32; 3],
    rough: f32,
    tint_w: f32,
) {
    let axis = (b - a).normalize_or(Vec3::Y);
    let up = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let t = axis.cross(up).normalize_or(Vec3::X);
    let bit = axis.cross(t);
    for i in 0..segs {
        let a0 = (i as f32 / segs as f32) * TAU;
        let a1 = ((i + 1) as f32 / segs as f32) * TAU;
        let d0 = t * a0.cos() + bit * a0.sin();
        let d1 = t * a1.cos() + bit * a1.sin();
        push_quad(
            verts,
            idxs,
            a + d0 * r0,
            a + d1 * r0,
            b + d1 * r1,
            b + d0 * r1,
            col,
            rough,
            tint_w,
        );
    }
}

/// Build an armored Hero Knight whose feet rest at `Y = 0.0` and plume crest reaches `Y = 1.90m`.
pub fn build_hero_knight_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(512);
    let mut idxs = Vec::with_capacity(1024);

    let steel = [0.76, 0.82, 0.90];
    let steel_dark = [0.40, 0.46, 0.54];
    let gold = [0.95, 0.76, 0.28];
    let leather = [0.34, 0.21, 0.14];
    let blade = [0.88, 0.95, 1.00];
    let white = [1.0, 1.0, 1.0];

    // Armored greaves & boots (Y = 0.00..0.68)
    for s in [-1.0f32, 1.0] {
        let lx = s * 0.15;
        push_box(&mut verts, &mut idxs, Vec3::new(lx - 0.10, 0.0, -0.10), Vec3::new(lx + 0.10, 0.15, 0.18), steel_dark, 0.30, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(lx, 0.14, 0.0), Vec3::new(lx, 0.68, 0.0), 0.10, 0.115, 8, steel, 0.25, 0.0);
    }

    // Faction tabard & steel cuirass (Y = 0.62..1.32)
    push_box(&mut verts, &mut idxs, Vec3::new(-0.26, 0.62, -0.16), Vec3::new(0.26, 1.28, 0.16), white, 0.78, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.25, 0.84, -0.18), Vec3::new(0.25, 1.26, 0.19), steel, 0.22, 0.0);
    // Leather belt + gold buckle
    push_box(&mut verts, &mut idxs, Vec3::new(-0.27, 0.72, -0.18), Vec3::new(0.27, 0.84, 0.18), leather, 0.75, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.08, 0.71, 0.16), Vec3::new(0.08, 0.85, 0.21), gold, 0.18, 0.0);

    // Sculpted layered pauldrons + gauntleted arms
    for s in [-1.0f32, 1.0] {
        let sx = s * 0.34;
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.13, 1.12, -0.15), Vec3::new(sx + 0.13, 1.32, 0.15), steel, 0.20, 0.0);
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.14, 1.26, -0.16), Vec3::new(sx + 0.14, 1.34, 0.16), gold, 0.18, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(sx, 1.12, 0.0), Vec3::new(sx * 1.04, 0.74, 0.08), 0.095, 0.085, 8, steel_dark, 0.28, 0.0);
    }

    // Flowing faction cape sweeping back (-Z) with gold hem
    push_quad(
        &mut verts,
        &mut idxs,
        Vec3::new(-0.26, 1.26, -0.17),
        Vec3::new(0.26, 1.26, -0.17),
        Vec3::new(0.38, 0.24, -0.44),
        Vec3::new(-0.38, 0.24, -0.44),
        white,
        0.80,
        1.0,
    );
    push_quad(
        &mut verts,
        &mut idxs,
        Vec3::new(0.26, 1.26, -0.18),
        Vec3::new(-0.26, 1.26, -0.18),
        Vec3::new(-0.38, 0.24, -0.45),
        Vec3::new(0.38, 0.24, -0.45),
        white,
        0.80,
        1.0,
    );
    push_box(&mut verts, &mut idxs, Vec3::new(-0.38, 0.20, -0.46), Vec3::new(0.38, 0.28, -0.40), gold, 0.22, 0.0);

    // Visored Steel Knight Helm (Y = 1.30..1.68) + Sweeping Faction Crest Plume (Y = 1.66..1.90)
    push_cylinder(&mut verts, &mut idxs, Vec3::new(0.0, 1.30, 0.0), Vec3::new(0.0, 1.64, 0.0), 0.19, 0.17, 10, steel, 0.20, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.18, 1.44, 0.13), Vec3::new(0.18, 1.50, 0.22), gold, 0.18, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.13, 1.39, 0.14), Vec3::new(0.13, 1.44, 0.20), [0.08, 0.09, 0.14], 0.40, 0.0);
    // Sweeping plume crest reaching exact Y = 1.90m!
    push_box(&mut verts, &mut idxs, Vec3::new(-0.05, 1.64, -0.26), Vec3::new(0.05, 1.90, 0.10), white, 0.72, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.04, 1.54, -0.36), Vec3::new(0.04, 1.76, -0.22), white, 0.72, 1.0);

    // Heraldic Kite Shield on left arm (-X) with embossed gold cross
    push_box(&mut verts, &mut idxs, Vec3::new(-0.48, 0.56, -0.05), Vec3::new(-0.40, 1.18, 0.36), white, 0.60, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.50, 0.62, 0.12), Vec3::new(-0.47, 1.14, 0.19), gold, 0.20, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.50, 0.86, -0.02), Vec3::new(-0.47, 0.93, 0.33), gold, 0.20, 0.0);

    // Crossguard Broadsword held forward on right arm (+X)
    push_box(&mut verts, &mut idxs, Vec3::new(0.32, 0.72, 0.05), Vec3::new(0.40, 0.80, 0.28), leather, 0.75, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(0.24, 0.71, 0.26), Vec3::new(0.48, 0.81, 0.33), gold, 0.18, 0.0);
    push_quad(
        &mut verts,
        &mut idxs,
        Vec3::new(0.33, 0.74, 0.33),
        Vec3::new(0.39, 0.74, 0.33),
        Vec3::new(0.37, 1.24, 0.96),
        Vec3::new(0.35, 1.24, 0.96),
        blade,
        0.12,
        0.0,
    );
    push_quad(
        &mut verts,
        &mut idxs,
        Vec3::new(0.39, 0.78, 0.33),
        Vec3::new(0.33, 0.78, 0.33),
        Vec3::new(0.35, 1.24, 0.96),
        Vec3::new(0.37, 1.24, 0.96),
        blade,
        0.12,
        0.0,
    );

    HeroMesh { vertices: verts, indices: idxs }
}

/// Default 4 hero knight placements showcasing true 3D multi-level scale & occlusion:
/// 1. Frost Paladin walking across the arched bridge deck (Y=9.05)
/// 2. Crimson Vanguard standing cleanly underneath the West bridge archway (Y=4.02, 3.7m headroom!)
/// 3. Emerald Warden standing at the vaulted basalt cave entrance (Y=4.25)
/// 4. Amber Scout deeper inside the glowing basalt cave mouth (Y=4.25)
pub fn showcase_hero_instances() -> Vec<HeroInstance> {
    vec![
        // (a) Walking across the arched bridge center
        HeroInstance {
            pos_yaw: [126.5, 9.04, 128.8, 1.15],
            faction_scale: [0.16, 0.52, 0.98, 1.0], // Frost Cobalt
        },
        // (b) Standing cleanly underneath the West bridge arch on dry riverbank!
        HeroInstance {
            pos_yaw: [113.5, 4.02, 128.0, 0.45],
            faction_scale: [0.92, 0.22, 0.24, 1.0], // Pyro Crimson
        },
        // (c) Standing at the vaulted basalt cave entrance!
        HeroInstance {
            pos_yaw: [100.2, 4.25, 151.4, -1.95],
            faction_scale: [0.18, 0.80, 0.48, 1.0], // Emerald Warden
        },
        // (d) Standing inside the vaulted cave bathed in magma/crystal glow!
        HeroInstance {
            pos_yaw: [93.2, 4.25, 153.6, -1.75],
            faction_scale: [0.96, 0.62, 0.16, 1.0], // Solar Amber
        },
    ]
}
