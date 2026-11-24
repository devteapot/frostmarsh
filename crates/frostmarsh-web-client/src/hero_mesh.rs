//! Compact Parametric 3D Player Archetype Meshes (5 Distinct Silhouettes) at True `~1.90m` Scale.
//!
//! Archetypes (`class_id` 0..4):
//! - `0` **Strider**: Heavy armored knight with kite shield, broadsword, and sweeping plume.
//! - `1` **Skywing**: Winged aerial flyer with wide animated flapping wings, glowing wingtips, and sky-glaive.
//! - `2` **Tidekin**: Gleaming aqua amphibious swimmer with swept dorsal/arm fins and glowing hydro-trident.
//! - `3` **Emberstalker**: Crouched obsidian/ember prowler with horned hood, shadow-ember cloak, and twin molten daggers.
//! - `4` **Burrower**: Stocky subterranean delver with crystal-amber back spires and a spinning glowing bore-drill.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::TAU;

pub const NUM_ARCHETYPES: usize = 5;
pub const ARCHETYPE_STRIDER: u8 = 0;
pub const ARCHETYPE_SKYWING: u8 = 1;
pub const ARCHETYPE_TIDEKIN: u8 = 2;
pub const ARCHETYPE_EMBERSTALKER: u8 = 3;
pub const ARCHETYPE_BURROWER: u8 = 4;

pub const ARCHETYPE_NAMES: [&str; NUM_ARCHETYPES] = [
    "Strider",
    "Skywing",
    "Tidekin",
    "Emberstalker",
    "Burrower",
];

pub fn archetype_name(class_id: u8) -> &'static str {
    ARCHETYPE_NAMES[(class_id as usize) % NUM_ARCHETYPES]
}

pub fn archetype_default_color(class_id: u8) -> [f32; 3] {
    match class_id % (NUM_ARCHETYPES as u8) {
        ARCHETYPE_STRIDER => [0.22, 0.56, 0.98],
        ARCHETYPE_SKYWING => [0.96, 0.78, 0.24],
        ARCHETYPE_TIDEKIN => [0.10, 0.88, 0.92],
        ARCHETYPE_EMBERSTALKER => [1.00, 0.34, 0.10],
        _ => [1.00, 0.68, 0.14],
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HeroVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color_rough: [f32; 4],
    /// x = faction_tint_weight (0..1), y = emissive (0..1), z = wing_flap_weight, w = secondary anim (1=hover, 2=drill spin)
    pub flags: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct HeroInstance {
    pub pos_yaw: [f32; 4],
    pub faction_scale: [f32; 4],
}

pub struct HeroMesh {
    pub vertices: Vec<HeroVertex>,
    pub indices: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn push_quad_ex(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    col: [f32; 3],
    rough: f32,
    flags: [f32; 4],
) {
    let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Y);
    let base = verts.len() as u32;
    for p in [p0, p1, p2, p3] {
        verts.push(HeroVertex {
            pos: p.to_array(),
            normal: n.to_array(),
            color_rough: [col[0], col[1], col[2], rough],
            flags,
        });
    }
    idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

#[allow(clippy::too_many_arguments)]
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
    push_quad_ex(verts, idxs, p0, p1, p2, p3, col, rough, [tint_w, 0.0, 0.0, 0.0]);
}

#[allow(clippy::too_many_arguments)]
fn push_tri_ex(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    col: [f32; 3],
    rough: f32,
    flags: [f32; 4],
) {
    let n = (b - a).cross(c - a).normalize_or(Vec3::Y);
    let base = verts.len() as u32;
    for p in [a, b, c] {
        verts.push(HeroVertex {
            pos: p.to_array(),
            normal: n.to_array(),
            color_rough: [col[0], col[1], col[2], rough],
            flags,
        });
    }
    idxs.extend_from_slice(&[base, base + 1, base + 2]);
}

#[allow(clippy::too_many_arguments)]
fn push_box_ex(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    min: Vec3,
    max: Vec3,
    col: [f32; 3],
    rough: f32,
    flags: [f32; 4],
) {
    let (x0, y0, z0) = (min.x, min.y, min.z);
    let (x1, y1, z1) = (max.x, max.y, max.z);
    push_quad_ex(verts, idxs, Vec3::new(x0, y0, z1), Vec3::new(x1, y0, z1), Vec3::new(x1, y1, z1), Vec3::new(x0, y1, z1), col, rough, flags);
    push_quad_ex(verts, idxs, Vec3::new(x1, y0, z0), Vec3::new(x0, y0, z0), Vec3::new(x0, y1, z0), Vec3::new(x1, y1, z0), col, rough, flags);
    push_quad_ex(verts, idxs, Vec3::new(x1, y0, z1), Vec3::new(x1, y0, z0), Vec3::new(x1, y1, z0), Vec3::new(x1, y1, z1), col, rough, flags);
    push_quad_ex(verts, idxs, Vec3::new(x0, y0, z0), Vec3::new(x0, y0, z1), Vec3::new(x0, y1, z1), Vec3::new(x0, y1, z0), col, rough, flags);
    push_quad_ex(verts, idxs, Vec3::new(x0, y1, z1), Vec3::new(x1, y1, z1), Vec3::new(x1, y1, z0), Vec3::new(x0, y1, z0), col, rough, flags);
    push_quad_ex(verts, idxs, Vec3::new(x0, y0, z0), Vec3::new(x1, y0, z0), Vec3::new(x1, y0, z1), Vec3::new(x0, y0, z1), col, rough, flags);
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
    push_box_ex(verts, idxs, min, max, col, rough, [tint_w, 0.0, 0.0, 0.0]);
}

#[allow(clippy::too_many_arguments)]
fn push_cylinder_ex(
    verts: &mut Vec<HeroVertex>,
    idxs: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    r0: f32,
    r1: f32,
    segs: usize,
    col: [f32; 3],
    rough: f32,
    flags: [f32; 4],
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
        push_quad_ex(
            verts,
            idxs,
            a + d0 * r0,
            a + d1 * r0,
            b + d1 * r1,
            b + d0 * r1,
            col,
            rough,
            flags,
        );
    }
}

#[allow(clippy::too_many_arguments)]
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
    push_cylinder_ex(verts, idxs, a, b, r0, r1, segs, col, rough, [tint_w, 0.0, 0.0, 0.0]);
}

/// Archetype 0 (`Strider`): Armored Hero Knight whose feet rest at `Y = 0.0` and plume crest reaches `Y = 1.90m`.
pub fn build_hero_knight_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(512);
    let mut idxs = Vec::with_capacity(1024);

    let steel = [0.76, 0.82, 0.90];
    let steel_dark = [0.40, 0.46, 0.54];
    let gold = [0.95, 0.76, 0.28];
    let leather = [0.34, 0.21, 0.14];
    let blade = [0.88, 0.95, 1.00];
    let white = [1.0, 1.0, 1.0];

    for s in [-1.0f32, 1.0] {
        let lx = s * 0.15;
        push_box(&mut verts, &mut idxs, Vec3::new(lx - 0.10, 0.0, -0.10), Vec3::new(lx + 0.10, 0.15, 0.18), steel_dark, 0.30, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(lx, 0.14, 0.0), Vec3::new(lx, 0.68, 0.0), 0.10, 0.115, 8, steel, 0.25, 0.0);
    }

    push_box(&mut verts, &mut idxs, Vec3::new(-0.26, 0.62, -0.16), Vec3::new(0.26, 1.28, 0.16), white, 0.78, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.25, 0.84, -0.18), Vec3::new(0.25, 1.26, 0.19), steel, 0.22, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.27, 0.72, -0.18), Vec3::new(0.27, 0.84, 0.18), leather, 0.75, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.08, 0.71, 0.16), Vec3::new(0.08, 0.85, 0.21), gold, 0.18, 0.0);

    for s in [-1.0f32, 1.0] {
        let sx = s * 0.34;
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.13, 1.12, -0.15), Vec3::new(sx + 0.13, 1.32, 0.15), steel, 0.20, 0.0);
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.14, 1.26, -0.16), Vec3::new(sx + 0.14, 1.34, 0.16), gold, 0.18, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(sx, 1.12, 0.0), Vec3::new(sx * 1.04, 0.74, 0.08), 0.095, 0.085, 8, steel_dark, 0.28, 0.0);
    }

    push_quad(&mut verts, &mut idxs, Vec3::new(-0.26, 1.26, -0.17), Vec3::new(0.26, 1.26, -0.17), Vec3::new(0.38, 0.24, -0.44), Vec3::new(-0.38, 0.24, -0.44), white, 0.80, 1.0);
    push_quad(&mut verts, &mut idxs, Vec3::new(0.26, 1.26, -0.18), Vec3::new(-0.26, 1.26, -0.18), Vec3::new(-0.38, 0.24, -0.45), Vec3::new(0.38, 0.24, -0.45), white, 0.80, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.38, 0.20, -0.46), Vec3::new(0.38, 0.28, -0.40), gold, 0.22, 0.0);

    push_cylinder(&mut verts, &mut idxs, Vec3::new(0.0, 1.30, 0.0), Vec3::new(0.0, 1.64, 0.0), 0.19, 0.17, 10, steel, 0.20, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.18, 1.44, 0.13), Vec3::new(0.18, 1.50, 0.22), gold, 0.18, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.13, 1.39, 0.14), Vec3::new(0.13, 1.44, 0.20), [0.08, 0.09, 0.14], 0.40, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.05, 1.64, -0.26), Vec3::new(0.05, 1.90, 0.10), white, 0.72, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.04, 1.54, -0.36), Vec3::new(0.04, 1.76, -0.22), white, 0.72, 1.0);

    push_box(&mut verts, &mut idxs, Vec3::new(-0.48, 0.56, -0.05), Vec3::new(-0.40, 1.18, 0.36), white, 0.60, 1.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.50, 0.62, 0.12), Vec3::new(-0.47, 1.14, 0.19), gold, 0.20, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.50, 0.86, -0.02), Vec3::new(-0.47, 0.93, 0.33), gold, 0.20, 0.0);

    push_box(&mut verts, &mut idxs, Vec3::new(0.32, 0.72, 0.05), Vec3::new(0.40, 0.80, 0.28), leather, 0.75, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(0.24, 0.71, 0.26), Vec3::new(0.48, 0.81, 0.33), gold, 0.18, 0.0);
    push_quad(&mut verts, &mut idxs, Vec3::new(0.33, 0.74, 0.33), Vec3::new(0.39, 0.74, 0.33), Vec3::new(0.37, 1.24, 0.96), Vec3::new(0.35, 1.24, 0.96), blade, 0.12, 0.0);
    push_quad(&mut verts, &mut idxs, Vec3::new(0.39, 0.78, 0.33), Vec3::new(0.33, 0.78, 0.33), Vec3::new(0.35, 1.24, 0.96), Vec3::new(0.37, 1.24, 0.96), blade, 0.12, 0.0);

    HeroMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Archetype 1 (`Skywing`): Winged 3D aerial flyer with wide GPU-animated flapping wings (`4.1m` span),
/// luminous sky-gold/cyan pinion tips, falcon crest helm, and aerial wind-glaive.
pub fn build_skywing_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(640);
    let mut idxs = Vec::with_capacity(1280);

    let sky_silver = [0.84, 0.90, 0.98];
    let sky_gold = [0.98, 0.80, 0.30];
    let wing_ivory = [0.95, 0.97, 1.00];
    let wing_glow = [0.46, 0.94, 1.00];
    let solar_tip = [1.00, 0.82, 0.32];
    let hover = [0.0, 0.0, 0.0, 1.0];
    let hover_tint = [0.85, 0.0, 0.0, 1.0];

    // Swept aerodynamic talon-greaves (slightly trailed back in flight)
    for s in [-1.0f32, 1.0] {
        let lx = s * 0.13;
        push_cylinder_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(lx, 0.12, -0.18),
            Vec3::new(lx, 0.72, -0.02),
            0.075,
            0.095,
            8,
            sky_silver,
            0.20,
            hover,
        );
        push_box_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(lx - 0.07, 0.02, -0.26),
            Vec3::new(lx + 0.07, 0.15, -0.04),
            sky_gold,
            0.18,
            hover,
        );
    }

    // Sleek tapered aerial cuirass & faction flight tunic
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.21, 0.68, -0.14), Vec3::new(0.21, 1.32, 0.15), wing_ivory, 0.60, hover_tint);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.22, 0.92, -0.16), Vec3::new(0.22, 1.30, 0.18), sky_silver, 0.16, hover);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.10, 0.96, 0.16), Vec3::new(0.10, 1.24, 0.21), wing_glow, 0.12, [0.25, 0.62, 0.0, 1.0]);

    // Swept pauldrons & slender arms
    for s in [-1.0f32, 1.0] {
        let sx = s * 0.29;
        push_box_ex(&mut verts, &mut idxs, Vec3::new(sx - 0.10, 1.16, -0.13), Vec3::new(sx + 0.10, 1.33, 0.13), sky_gold, 0.18, hover);
        push_cylinder_ex(&mut verts, &mut idxs, Vec3::new(sx, 1.16, 0.0), Vec3::new(sx * 1.08, 0.78, 0.12), 0.075, 0.068, 7, sky_silver, 0.22, hover);
    }

    // Falcon-crested aerodynamic helm + glowing sky-visor
    push_cylinder_ex(&mut verts, &mut idxs, Vec3::new(0.0, 1.34, 0.0), Vec3::new(0.0, 1.66, 0.02), 0.16, 0.14, 10, sky_silver, 0.16, hover);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.13, 1.46, 0.12), Vec3::new(0.13, 1.54, 0.20), wing_glow, 0.10, [0.15, 0.72, 0.0, 1.0]);
    for s in [-1.0f32, 1.0] {
        push_quad_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 0.12, 1.54, 0.04),
            Vec3::new(s * 0.15, 1.64, -0.02),
            Vec3::new(s * 0.34, 1.88, -0.34),
            Vec3::new(s * 0.24, 1.74, -0.26),
            sky_gold,
            0.22,
            hover,
        );
    }

    // Wide GPU-animated articulated soaring wings (span X = -2.05 .. +2.05m!)
    for s in [-1.0f32, 1.0] {
        let flap_inner = [0.35, 0.0, 0.65, 1.0];
        let flap_mid = [0.45, 0.18, 0.90, 1.0];
        let flap_tip = [0.25, 0.60, 1.00, 1.0];

        // Leading golden wing spar (shoulder -> elbow -> tip)
        push_cylinder_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 0.20, 1.24, -0.15),
            Vec3::new(s * 1.08, 1.56, -0.28),
            0.055,
            0.042,
            6,
            sky_gold,
            0.18,
            flap_inner,
        );
        push_cylinder_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 1.08, 1.56, -0.28),
            Vec3::new(s * 2.04, 1.44, -0.58),
            0.042,
            0.022,
            6,
            solar_tip,
            0.16,
            flap_tip,
        );

        // Inner wing membrane (thick double-sided airfoil quads)
        let p0 = Vec3::new(s * 0.22, 1.24, -0.16);
        let p1 = Vec3::new(s * 1.10, 1.56, -0.29);
        let p2 = Vec3::new(s * 1.00, 0.96, -0.50);
        let p3 = Vec3::new(s * 0.22, 0.88, -0.30);
        push_quad_ex(&mut verts, &mut idxs, p0, p1, p2, p3, wing_ivory, 0.32, flap_inner);
        push_quad_ex(&mut verts, &mut idxs, p1, p0, p3, p2, wing_ivory, 0.32, flap_inner);

        // Outer swept pinion sail
        let q0 = Vec3::new(s * 1.06, 1.56, -0.28);
        let q1 = Vec3::new(s * 2.06, 1.46, -0.58);
        let q2 = Vec3::new(s * 1.78, 0.98, -0.78);
        let q3 = Vec3::new(s * 0.98, 0.98, -0.50);
        push_quad_ex(&mut verts, &mut idxs, q0, q1, q2, q3, wing_ivory, 0.26, flap_mid);
        push_quad_ex(&mut verts, &mut idxs, q1, q0, q3, q2, wing_ivory, 0.26, flap_mid);

        // Three glowing sky-energy primary pinion feathers fanning from the outer wingtip
        for f in 0..3 {
            let t0 = f as f32 * 0.22;
            let root = Vec3::new(s * (1.28 + t0 * 0.95), 1.30 - t0 * 0.28, -0.44 - t0 * 0.20);
            let tip = Vec3::new(s * (2.10 - t0 * 0.36), 1.48 - t0 * 0.68, -0.66 - t0 * 0.44);
            let side_v = Vec3::new(0.0, -0.16, -0.08);
            let fcol = if f == 0 { solar_tip } else { wing_glow };
            push_tri_ex(&mut verts, &mut idxs, root, tip, root + side_v, fcol, 0.14, flap_tip);
            push_tri_ex(&mut verts, &mut idxs, root, root + side_v, tip, fcol, 0.14, flap_tip);
        }
    }

    // Swept tail plume stabilizer
    push_tri_ex(
        &mut verts,
        &mut idxs,
        Vec3::new(-0.18, 0.72, -0.18),
        Vec3::new(0.18, 0.72, -0.18),
        Vec3::new(0.0, 0.38, -0.74),
        wing_ivory,
        0.45,
        hover_tint,
    );

    // Aerial Wind-Glaive held forward on right hand (+X)
    push_cylinder_ex(
        &mut verts,
        &mut idxs,
        Vec3::new(0.31, 0.60, -0.25),
        Vec3::new(0.35, 1.08, 0.78),
        0.032,
        0.032,
        6,
        sky_gold,
        0.20,
        hover,
    );
    push_tri_ex(
        &mut verts,
        &mut idxs,
        Vec3::new(0.35, 1.04, 0.74),
        Vec3::new(0.35, 1.24, 1.22),
        Vec3::new(0.35, 1.18, 0.74),
        wing_glow,
        0.10,
        [0.20, 0.72, 0.0, 1.0],
    );

    HeroMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Archetype 2 (`Tidekin`): Gleaming aqua amphibious swimmer/diver with iridescent hydro-scales,
/// tall swept dorsal crest fin, forearm fins, and a glowing aqua hydro-trident.
pub fn build_tidekin_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(560);
    let mut idxs = Vec::with_capacity(1120);

    let aqua_scale = [0.10, 0.74, 0.84];
    let deep_teal = [0.05, 0.36, 0.48];
    let pearl_fin = [0.44, 0.96, 0.96];
    let aqua_glow = [0.18, 0.98, 1.00];
    let silver = [0.76, 0.92, 0.96];

    // Sleek hydrodynamic legs + webbed flippers
    for s in [-1.0f32, 1.0] {
        let lx = s * 0.14;
        push_box(&mut verts, &mut idxs, Vec3::new(lx - 0.10, 0.0, -0.10), Vec3::new(lx + 0.10, 0.10, 0.24), deep_teal, 0.12, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(lx, 0.10, 0.0), Vec3::new(lx, 0.70, 0.0), 0.088, 0.105, 8, aqua_scale, 0.10, 0.0);
        // Calf swim-fin
        push_tri_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(lx, 0.20, -0.08),
            Vec3::new(lx, 0.56, -0.08),
            Vec3::new(lx + s * 0.12, 0.32, -0.34),
            pearl_fin,
            0.14,
            [0.35, 0.32, 0.0, 0.0],
        );
    }

    // Iridescent streamlined torso + faction sash
    push_box(&mut verts, &mut idxs, Vec3::new(-0.23, 0.64, -0.15), Vec3::new(0.23, 1.30, 0.16), aqua_scale, 0.10, 0.25);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.24, 0.72, -0.16), Vec3::new(0.24, 0.86, 0.17), pearl_fin, 0.22, 0.75);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.08, 0.98, 0.15), Vec3::new(0.08, 1.22, 0.20), aqua_glow, 0.08, [0.20, 0.68, 0.0, 0.0]);

    // Swept arms + glowing forearm hydro-fins
    for s in [-1.0f32, 1.0] {
        let sx = s * 0.30;
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.10, 1.14, -0.12), Vec3::new(sx + 0.10, 1.32, 0.12), deep_teal, 0.12, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(sx, 1.16, 0.0), Vec3::new(sx * 1.06, 0.76, 0.10), 0.08, 0.072, 8, aqua_scale, 0.10, 0.0);
        push_quad_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(sx * 1.05, 1.06, -0.04),
            Vec3::new(sx * 1.08, 0.78, 0.06),
            Vec3::new(sx * 1.56, 0.74, -0.26),
            Vec3::new(sx * 1.48, 1.12, -0.28),
            pearl_fin,
            0.14,
            [0.35, 0.36, 0.0, 0.0],
        );
    }

    // Prominent swept dorsal crest fin running down the spine & helmet
    push_quad_ex(
        &mut verts,
        &mut idxs,
        Vec3::new(0.0, 1.32, -0.14),
        Vec3::new(0.0, 0.68, -0.16),
        Vec3::new(0.0, 0.56, -0.62),
        Vec3::new(0.0, 1.46, -0.58),
        aqua_glow,
        0.12,
        [0.35, 0.54, 0.0, 0.0],
    );
    // Twin caudal hip fins
    for s in [-1.0f32, 1.0] {
        push_tri_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 0.12, 0.74, -0.16),
            Vec3::new(s * 0.42, 0.44, -0.58),
            Vec3::new(s * 0.06, 0.54, -0.26),
            pearl_fin,
            0.14,
            [0.45, 0.25, 0.0, 0.0],
        );
    }

    // Streamlined aquatic helm + high dorsal head-crest + glowing eyes
    push_cylinder(&mut verts, &mut idxs, Vec3::new(0.0, 1.32, 0.0), Vec3::new(0.0, 1.66, 0.02), 0.17, 0.14, 10, aqua_scale, 0.10, 0.0);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.12, 1.46, 0.13), Vec3::new(0.12, 1.53, 0.20), aqua_glow, 0.08, [0.10, 0.76, 0.0, 0.0]);
    push_tri_ex(
        &mut verts,
        &mut idxs,
        Vec3::new(0.0, 1.64, 0.10),
        Vec3::new(0.0, 2.02, -0.34),
        Vec3::new(0.0, 1.44, -0.18),
        aqua_glow,
        0.12,
        [0.30, 0.58, 0.0, 0.0],
    );

    // Glowing Aqua Hydro-Trident held forward on right side (+X)
    push_cylinder(&mut verts, &mut idxs, Vec3::new(0.36, 0.34, -0.18), Vec3::new(0.38, 1.38, 0.62), 0.036, 0.036, 6, silver, 0.15, 0.0);
    // Crossbar
    push_box(&mut verts, &mut idxs, Vec3::new(0.22, 1.32, 0.56), Vec3::new(0.54, 1.40, 0.66), silver, 0.15, 0.0);
    // 3 glowing aqua trident prongs
    for px in [0.24f32, 0.38, 0.52] {
        let tip_y = if (px - 0.38).abs() < 0.05 { 1.88 } else { 1.72 };
        let tip_z = if (px - 0.38).abs() < 0.05 { 0.96 } else { 0.84 };
        push_cylinder_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(px, 1.36, 0.61),
            Vec3::new(px, tip_y, tip_z),
            0.032,
            0.008,
            6,
            aqua_glow,
            0.08,
            [0.15, 0.78, 0.0, 0.0],
        );
    }

    HeroMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Archetype 3 (`Emberstalker`): Crouched dark obsidian/ember assassin with horned shadow hood,
/// tattered glowing-hem cloak, burning visor eyes, and twin molten-ember daggers.
pub fn build_emberstalker_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(560);
    let mut idxs = Vec::with_capacity(1120);

    let obsidian = [0.09, 0.08, 0.13];
    let char_slate = [0.18, 0.16, 0.22];
    let ember_hot = [1.00, 0.36, 0.06];
    let ember_core = [1.00, 0.66, 0.16];

    // Agile prowling crouched legs
    for s in [-1.0f32, 1.0] {
        let lx = s * 0.17;
        push_box(&mut verts, &mut idxs, Vec3::new(lx - 0.09, 0.0, -0.14), Vec3::new(lx + 0.09, 0.13, 0.16), obsidian, 0.26, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(lx, 0.12, -0.06), Vec3::new(lx, 0.42, 0.12), 0.085, 0.092, 7, obsidian, 0.22, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(lx, 0.40, 0.12), Vec3::new(lx * 0.85, 0.70, -0.02), 0.092, 0.105, 7, char_slate, 0.28, 0.0);
    }

    // Forward-leaning obsidian torso + glowing magma rune slits
    push_box(&mut verts, &mut idxs, Vec3::new(-0.24, 0.64, -0.15), Vec3::new(0.24, 1.24, 0.18), obsidian, 0.24, 0.15);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.14, 0.86, 0.16), Vec3::new(0.14, 1.12, 0.21), ember_hot, 0.15, [0.15, 0.70, 0.0, 0.0]);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.25, 0.70, -0.16), Vec3::new(0.25, 0.80, 0.19), char_slate, 0.55, 0.55);

    // Jagged obsidian shoulder spikes & crouched arms
    for s in [-1.0f32, 1.0] {
        let sx = s * 0.32;
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.11, 1.08, -0.14), Vec3::new(sx + 0.11, 1.28, 0.16), obsidian, 0.20, 0.0);
        push_tri_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(sx, 1.26, -0.10),
            Vec3::new(sx, 1.26, 0.12),
            Vec3::new(sx + s * 0.22, 1.54, -0.08),
            obsidian,
            0.18,
            [0.0, 0.0, 0.0, 0.0],
        );
        push_cylinder(&mut verts, &mut idxs, Vec3::new(sx, 1.12, 0.04), Vec3::new(sx * 1.08, 0.74, 0.24), 0.082, 0.072, 7, char_slate, 0.24, 0.0);
    }

    // Split tattered shadow-ember cloak sweeping back (-Z)
    for s in [-1.0f32, 1.0] {
        push_quad_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 0.04, 1.22, -0.14),
            Vec3::new(s * 0.26, 1.22, -0.14),
            Vec3::new(s * 0.44, 0.22, -0.58),
            Vec3::new(s * 0.12, 0.30, -0.52),
            char_slate,
            0.70,
            [0.45, 0.0, 0.0, 0.0],
        );
        push_tri_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 0.12, 0.30, -0.52),
            Vec3::new(s * 0.44, 0.22, -0.58),
            Vec3::new(s * 0.34, 0.08, -0.68),
            ember_hot,
            0.22,
            [0.10, 0.65, 0.0, 0.0],
        );
    }

    // Deep horned obsidian assassin hood + glowing molten-ember eyes
    push_box(&mut verts, &mut idxs, Vec3::new(-0.20, 1.24, -0.16), Vec3::new(0.20, 1.62, 0.22), obsidian, 0.28, 0.10);
    push_tri_ex(
        &mut verts,
        &mut idxs,
        Vec3::new(-0.18, 1.60, 0.18),
        Vec3::new(0.18, 1.60, 0.18),
        Vec3::new(0.0, 1.78, -0.26),
        obsidian,
        0.30,
        [0.0, 0.0, 0.0, 0.0],
    );
    // Twin curved obsidian horns
    for s in [-1.0f32, 1.0] {
        push_cylinder(&mut verts, &mut idxs, Vec3::new(s * 0.16, 1.56, 0.02), Vec3::new(s * 0.30, 1.88, -0.14), 0.042, 0.012, 6, obsidian, 0.16, 0.0);
        // Glowing ember eyes
        push_box_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(s * 0.04, 1.40, 0.19),
            Vec3::new(s * 0.13, 1.46, 0.24),
            ember_core,
            0.10,
            [0.0, 0.85, 0.0, 0.0],
        );
    }

    // Twin curved glowing molten-ember daggers (left & right hands)
    for s in [-1.0f32, 1.0] {
        let hx = s * 0.36;
        push_box(&mut verts, &mut idxs, Vec3::new(hx - 0.04, 0.70, 0.18), Vec3::new(hx + 0.04, 0.78, 0.34), obsidian, 0.30, 0.0);
        push_quad_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(hx - 0.02, 0.72, 0.34),
            Vec3::new(hx + 0.02, 0.72, 0.34),
            Vec3::new(hx + s * 0.10, 0.94, 0.94),
            Vec3::new(hx + s * 0.04, 0.96, 0.94),
            ember_hot,
            0.12,
            [0.08, 0.80, 0.0, 0.0],
        );
        push_quad_ex(
            &mut verts,
            &mut idxs,
            Vec3::new(hx + 0.02, 0.78, 0.34),
            Vec3::new(hx - 0.02, 0.78, 0.34),
            Vec3::new(hx + s * 0.04, 0.96, 0.94),
            Vec3::new(hx + s * 0.10, 0.94, 0.94),
            ember_core,
            0.12,
            [0.08, 0.80, 0.0, 0.0],
        );
    }

    HeroMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Archetype 4 (`Burrower`): Stocky heavy subterranean delver with glowing crystal-amber back spires,
/// armored miner helm with glowing headlamp visor, and a GPU-spinning glowing crystal-amber bore-drill.
pub fn build_burrower_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(600);
    let mut idxs = Vec::with_capacity(1200);

    let bronze_iron = [0.36, 0.28, 0.21];
    let dark_iron = [0.22, 0.19, 0.18];
    let amber_crystal = [1.00, 0.68, 0.12];
    let amber_core = [1.00, 0.86, 0.34];
    let copper_trim = [0.76, 0.46, 0.20];

    // Wide stocky heavy armored boots & greaves
    for s in [-1.0f32, 1.0] {
        let lx = s * 0.21;
        push_box(&mut verts, &mut idxs, Vec3::new(lx - 0.13, 0.0, -0.14), Vec3::new(lx + 0.13, 0.18, 0.22), dark_iron, 0.34, 0.0);
        push_cylinder(&mut verts, &mut idxs, Vec3::new(lx, 0.16, 0.0), Vec3::new(lx, 0.64, 0.0), 0.13, 0.145, 8, bronze_iron, 0.30, 0.0);
    }

    // Broad armored miner chassis + faction chest plate
    push_box(&mut verts, &mut idxs, Vec3::new(-0.35, 0.58, -0.22), Vec3::new(0.35, 1.32, 0.22), bronze_iron, 0.28, 0.20);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.30, 0.72, -0.24), Vec3::new(0.30, 1.24, 0.25), copper_trim, 0.24, 0.50);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.14, 0.86, 0.23), Vec3::new(0.14, 1.12, 0.28), amber_crystal, 0.12, [0.12, 0.74, 0.0, 0.0]);

    // Heavy wide sloped pauldrons
    for s in [-1.0f32, 1.0] {
        let sx = s * 0.44;
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.16, 1.06, -0.20), Vec3::new(sx + 0.16, 1.38, 0.20), dark_iron, 0.26, 0.0);
        push_box(&mut verts, &mut idxs, Vec3::new(sx - 0.17, 1.28, -0.21), Vec3::new(sx + 0.17, 1.38, 0.21), copper_trim, 0.22, 0.45);
    }

    // Left heavy arm + crystal-studded buckler
    push_cylinder(&mut verts, &mut idxs, Vec3::new(-0.44, 1.10, 0.0), Vec3::new(-0.48, 0.72, 0.14), 0.115, 0.105, 8, bronze_iron, 0.30, 0.0);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.62, 0.56, -0.02), Vec3::new(-0.52, 1.08, 0.36), dark_iron, 0.28, 0.0);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.65, 0.68, 0.08), Vec3::new(-0.60, 0.96, 0.26), amber_crystal, 0.12, [0.10, 0.72, 0.0, 0.0]);

    // Right arm housing + GPU-spinning glowing Crystal-Amber Bore-Drill! (flags.w = 2.0)
    push_cylinder(&mut verts, &mut idxs, Vec3::new(0.42, 1.10, 0.0), Vec3::new(0.38, 0.88, 0.24), 0.12, 0.13, 8, dark_iron, 0.26, 0.0);
    let drill_flags = [0.10, 0.74, 0.0, 2.0];
    let drill_base = Vec3::new(0.38, 0.88, 0.24);
    let drill_mid = Vec3::new(0.38, 0.88, 0.64);
    let drill_tip = Vec3::new(0.38, 0.88, 1.12);
    push_cylinder_ex(&mut verts, &mut idxs, drill_base, drill_mid, 0.20, 0.12, 6, amber_crystal, 0.10, drill_flags);
    push_cylinder_ex(&mut verts, &mut idxs, drill_mid, drill_tip, 0.12, 0.015, 6, amber_core, 0.08, drill_flags);
    // Fluted auger fins on the spinning drill
    for k in 0..3 {
        let ang = (k as f32 / 3.0) * TAU;
        let d = Vec3::new(ang.cos(), ang.sin(), 0.0);
        push_tri_ex(
            &mut verts,
            &mut idxs,
            drill_base + d * 0.24,
            drill_mid + d * 0.17,
            drill_tip,
            amber_core,
            0.10,
            drill_flags,
        );
    }

    // Armored subterranean backpack + 4 glowing crystal-amber back spires
    push_box(&mut verts, &mut idxs, Vec3::new(-0.28, 0.72, -0.44), Vec3::new(0.28, 1.28, -0.22), dark_iron, 0.30, 0.0);
    let crystals = [
        (Vec3::new(-0.16, 1.24, -0.32), Vec3::new(-0.34, 1.84, -0.48), 0.095),
        (Vec3::new(0.16, 1.24, -0.32), Vec3::new(0.34, 1.86, -0.46), 0.095),
        (Vec3::new(0.0, 1.26, -0.34), Vec3::new(0.0, 1.98, -0.52), 0.11),
        (Vec3::new(0.0, 1.00, -0.42), Vec3::new(0.0, 1.38, -0.76), 0.085),
    ];
    for (c_a, c_b, r) in crystals {
        push_cylinder_ex(
            &mut verts,
            &mut idxs,
            c_a,
            c_b,
            r,
            0.015,
            5,
            amber_crystal,
            0.10,
            [0.10, 0.76, 0.0, 0.0],
        );
    }

    // Domed heavy delver helm + glowing crystal-amber headlamp visor
    push_cylinder(&mut verts, &mut idxs, Vec3::new(0.0, 1.30, 0.0), Vec3::new(0.0, 1.62, 0.0), 0.22, 0.19, 10, bronze_iron, 0.24, 0.0);
    push_box_ex(&mut verts, &mut idxs, Vec3::new(-0.16, 1.42, 0.17), Vec3::new(0.16, 1.54, 0.25), amber_core, 0.08, [0.05, 0.84, 0.0, 0.0]);
    push_box(&mut verts, &mut idxs, Vec3::new(-0.20, 1.56, -0.14), Vec3::new(0.20, 1.66, 0.20), copper_trim, 0.22, 0.0);

    HeroMesh {
        vertices: verts,
        indices: idxs,
    }
}

/// Build all 5 distinct 3D player archetype meshes indexed by `class_id` (`0..4`).
pub fn build_archetype_meshes() -> [HeroMesh; NUM_ARCHETYPES] {
    [
        build_hero_knight_mesh(),
        build_skywing_mesh(),
        build_tidekin_mesh(),
        build_emberstalker_mesh(),
        build_burrower_mesh(),
    ]
}

/// Build a compact glowing 3D orb mesh (octahedral crystal) for live multiplayer projectiles & beam streams.
/// Uses `flags.x = 1.0` (full faction/spell color tint) and `flags.y = 1.0` (emissive HDR glow).
pub fn build_projectile_orb_mesh() -> HeroMesh {
    let mut verts = Vec::with_capacity(32);
    let mut idxs = Vec::with_capacity(48);
    let white = [1.0, 1.0, 1.0];

    let top = Vec3::new(0.0, 1.0, 0.0);
    let bot = Vec3::new(0.0, -1.0, 0.0);
    let ring = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];

    let mut push_tri = |a: Vec3, b: Vec3, c: Vec3| {
        let n = (b - a).cross(c - a).normalize_or(Vec3::Y);
        let base = verts.len() as u32;
        for p in [a, b, c] {
            verts.push(HeroVertex {
                pos: p.to_array(),
                normal: n.to_array(),
                color_rough: [white[0], white[1], white[2], 0.12],
                flags: [1.0, 1.0, 0.0, 0.0],
            });
        }
        idxs.extend_from_slice(&[base, base + 1, base + 2]);
    };

    for i in 0..4 {
        let p0 = ring[i];
        let p1 = ring[(i + 1) % 4];
        push_tri(top, p1, p0);
        push_tri(bot, p0, p1);
    }

    HeroMesh {
        vertices: verts,
        indices: idxs,
    }
}

pub fn showcase_hero_instances() -> Vec<HeroInstance> {
    vec![
        HeroInstance {
            pos_yaw: [126.5, 9.04, 128.8, 1.15],
            faction_scale: [0.16, 0.52, 0.98, 1.0],
        },
        HeroInstance {
            pos_yaw: [113.5, 4.02, 128.0, 0.45],
            faction_scale: [0.92, 0.22, 0.24, 1.0],
        },
        HeroInstance {
            pos_yaw: [100.2, 4.25, 151.4, -1.95],
            faction_scale: [0.18, 0.80, 0.48, 1.0],
        },
        HeroInstance {
            pos_yaw: [93.2, 4.25, 153.6, -1.75],
            faction_scale: [0.96, 0.62, 0.16, 1.0],
        },
    ]
}

/// Default 5-archetype offline showcase featuring all 5 distinct silhouettes, including a flying winged Skywing
/// soaring above the canyon bridge, a gleaming aqua Tidekin in the river, an Emberstalker at the basalt cave mouth,
/// and a crystal-amber Burrower deeper inside the cavern.
pub fn showcase_archetype_heroes() -> Vec<(HeroInstance, u8)> {
    vec![
        (
            HeroInstance {
                pos_yaw: [126.5, 9.04, 128.8, 1.15],
                faction_scale: [0.18, 0.54, 0.98, 1.0],
            },
            ARCHETYPE_STRIDER,
        ),
        (
            HeroInstance {
                pos_yaw: [120.5, 13.4, 124.2, 0.82],
                faction_scale: [0.98, 0.80, 0.26, 1.08],
            },
            ARCHETYPE_SKYWING,
        ),
        (
            HeroInstance {
                pos_yaw: [115.4, 3.18, 128.6, 0.42],
                faction_scale: [0.10, 0.90, 0.94, 1.0],
            },
            ARCHETYPE_TIDEKIN,
        ),
        (
            HeroInstance {
                pos_yaw: [100.2, 4.25, 151.4, -1.95],
                faction_scale: [0.96, 0.26, 0.14, 1.0],
            },
            ARCHETYPE_EMBERSTALKER,
        ),
        (
            HeroInstance {
                pos_yaw: [93.2, 4.25, 153.6, -1.75],
                faction_scale: [0.98, 0.68, 0.16, 1.02],
            },
            ARCHETYPE_BURROWER,
        ),
    ]
}
