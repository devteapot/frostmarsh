//! Approach 4 Parametric CSG/SDF Greedy-Meshed Baked-AO 3D Characters & Props for Live Frostmarsh.
//!
//! 5 Distinct 3D Player Archetypes (`class_id` 0..=4):
//! - `HeroStrider` (0 — Heavy Walker / Squad Anchor: broad pauldrons, heavy shield crest, banner plume)
//! - `HeroSkywing` + `SkywingWingL`/`SkywingWingR` (1 — True 3D Flyer / Thermal Glider: articulated glowing wings)
//! - `HeroTidekin` (2 — Amphibious Swimmer / Diver: sleek dorsal fin, hydro-fins, aqua bioluminescent trim)
//! - `HeroStalker` (3 — Emberstalker Solo Cavern Climber / Raider: dark hooded cloak + glowing twin ember-blades)
//! - `HeroBurrower` (4 — Subterranean Crystal-Plated Golem: twin conical drill-gauntlets, glowing amethyst/amber dorsal crystal spine, heavy subterranean treads/claws)
//!
//! Plus Props & FX:
//! - IceRaft, RelicPedestal, GoldenRelic, DashHalo, MeleeArc, ProjectileOrb,
//!   TreeTrunk, RockOutcrop, MarshDeer, GroundRing, ShrineObelisk

use crate::parametric::{hash2i, ParamVolume};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::{PI, TAU};

pub const CLASS_STRIDER: u8 = 0;
pub const CLASS_SKYWING: u8 = 1;
pub const CLASS_TIDEKIN: u8 = 2;
pub const CLASS_EMBERSTALKER: u8 = 3;
pub const CLASS_BURROWER: u8 = 4;

#[inline]
pub fn archetype_tag(class_id: u8) -> &'static str {
    match class_id {
        CLASS_STRIDER => "STRIDER",
        CLASS_SKYWING => "SKYWING",
        CLASS_TIDEKIN => "TIDEKIN",
        CLASS_EMBERSTALKER => "STALKER",
        CLASS_BURROWER => "BURROWER",
        _ => "STALKER",
    }
}

#[inline]
pub fn archetype_color(class_id: u8) -> [f32; 4] {
    match class_id {
        CLASS_STRIDER => [0.46, 0.90, 1.00, 1.0],
        CLASS_SKYWING => [1.00, 0.74, 0.28, 1.0],
        CLASS_TIDEKIN => [0.22, 0.96, 0.86, 1.0],
        CLASS_EMBERSTALKER => [1.00, 0.36, 0.22, 1.0],
        CLASS_BURROWER => [0.94, 0.58, 0.34, 1.0],
        _ => [1.00, 0.36, 0.22, 1.0],
    }
}

#[inline]
pub fn archetype_is_solo(class_id: u8) -> bool {
    class_id == CLASS_EMBERSTALKER
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// rgb = albedo, a = mat_id:
    /// 0=plain/cloth, 1=stone, 3=river water, 4=wood, 5=emissive, 6=ice,
    /// 9=A+ compositional terrain, 10=polished metal/gold, 12=faction-tinted cloth
    pub color_mat: [f32; 4],
    /// x=temp_norm, y=burn_intensity, z=flow_x, w=flow_z (used on terrain & water)
    pub extra: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct MeshInstance {
    pub pos_yaw: [f32; 4],
    pub scale_pitch: [f32; 4],
    pub tint_glow: [f32; 4],
    /// x = mat_override (<0 keep vertex mat), y = selective_tint (>0.5 only mat 12), z = roll (rad), w = reserved
    pub flags: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct MeshSlice {
    pub first_index: u32,
    pub index_count: u32,
}

pub const MESH_KIND_COUNT: usize = 18;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MeshKind {
    HeroStrider = 0,
    HeroSkywing = 1,
    IceRaft = 2,
    RelicPedestal = 3,
    GoldenRelic = 4,
    DashHalo = 5,
    MeleeArc = 6,
    ProjectileOrb = 7,
    TreeTrunk = 8,
    RockOutcrop = 9,
    MarshDeer = 10,
    GroundRing = 11,
    HeroTidekin = 12,
    ShrineObelisk = 13,
    HeroStalker = 14,
    SkywingWingL = 15,
    SkywingWingR = 16,
    HeroBurrower = 17,
}

pub struct StaticMeshBank {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub slices: [MeshSlice; MESH_KIND_COUNT],
}

fn push_quad(
    verts: &mut Vec<MeshVertex>,
    idxs: &mut Vec<u32>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    col: [f32; 3],
    mat: f32,
) {
    let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Y);
    let base = verts.len() as u32;
    let cm = [col[0], col[1], col[2], mat];
    for p in [p0, p1, p2, p3] {
        verts.push(MeshVertex {
            pos: p.to_array(),
            normal: n.to_array(),
            color_mat: cm,
            extra: [0.0; 4],
        });
    }
    idxs.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn push_limb(
    verts: &mut Vec<MeshVertex>,
    idxs: &mut Vec<u32>,
    a: Vec3,
    b: Vec3,
    r0: f32,
    r1: f32,
    segs: usize,
    col: [f32; 3],
    mat: f32,
) {
    let axis = (b - a).normalize_or(Vec3::Y);
    let ref_v = if axis.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let t = axis.cross(ref_v).normalize_or(Vec3::X);
    let bit = axis.cross(t);
    for i in 0..segs {
        let a0 = i as f32 / segs as f32 * TAU;
        let a1 = (i + 1) as f32 / segs as f32 * TAU;
        let d0 = t * a0.cos() + bit * a0.sin();
        let d1 = t * a1.cos() + bit * a1.sin();
        push_quad(
            verts,
            idxs,
            a + d0 * r0,
            b + d0 * r1,
            b + d1 * r1,
            a + d1 * r0,
            col,
            mat,
        );
    }
}

/// Archetype 0: STRIDER — Heavy Walker / Squad Anchor
/// Broad double-tiered armored pauldrons, heavy fortress shield with golden crest & anchor rune core,
/// steel cuirass, visored greathelm, and squad-anchor crest plume.
fn build_parametric_strider(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(18.0, 22.0, 16.0, det);
    let (cx, cz) = (9.0f32, 7.6f32);

    let team = vol.color_mat(0xffffff, 12.0);
    let team_d = vol.color_mat(0xc8c8c8, 12.0);
    let steel = vol.color_mat(0xc8d8eb, 10.0);
    let steel_d = vol.color_mat(0x63748c, 10.0);
    let gold = vol.color_mat(0xf5c84c, 10.0);
    let leather = vol.color_mat(0x5c3828, 0.0);
    let skin = vol.color_mat(0xf0c29e, 0.0);
    let eye = vol.color_mat(0x262038, 0.0);
    let blade = vol.color_mat(0xdcf5ff, 10.0);
    let anchor_rune = vol.color_mat(0x68eeff, 5.0);

    // Wide-stance heavy armored greaves & sabatons
    for s in [-1.0f32, 1.0] {
        let lx = cx + s * 1.65;
        vol.rounded_box(lx - 1.05, 0.0, cz - 1.25, 2.1, 5.4, 2.5, 0.5, steel_d);
        vol.box_f(lx - 1.15, 0.0, cz - 1.1, 2.3, 1.5, 3.0, steel);
        vol.box_f(lx - 0.9, 2.8, cz + 1.0, 1.8, 1.1, 0.45, gold);
    }

    // Broad fortified steel cuirass + faction tabard + heavy waist belt
    vol.rounded_box(cx - 3.0, 5.2, cz - 2.1, 6.0, 6.4, 4.2, 0.9, team);
    vol.rounded_box(cx - 2.8, 7.2, cz - 1.9, 5.6, 4.2, 4.2, 0.8, steel);
    vol.rounded_box(cx - 3.1, 5.2, cz - 2.2, 6.2, 1.3, 4.4, 0.7, leather);
    vol.box_f(cx - 1.0, 5.0, cz + 2.0, 2.0, 1.5, 0.65, gold);
    vol.box_f(cx - 0.6, 8.6, cz + 2.1, 1.2, 1.4, 0.45, anchor_rune);

    // Unmistakable broad double-tiered armored pauldrons + gauntleted arms
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 4.15;
        vol.ellipsoid(sx, 10.9, cz, 2.25, 1.65, 2.35, steel);
        vol.ellipsoid(sx + s * 0.35, 11.7, cz, 1.95, 0.72, 2.15, gold);
        vol.ellipsoid(sx + s * 0.65, 10.3, cz, 1.55, 0.85, 1.95, team);
        vol.rounded_box(sx - 1.0, 6.2, cz - 0.9, 2.0, 4.0, 2.1, 0.55, steel_d);
    }

    // Flowing heavy squad-anchor mantle sweeping behind back (-Z)
    vol.sculpt(cx - 4.0, 2.0, cz - 4.6, cx + 4.0, 11.8, cz - 1.8, |wx, wy, wz| {
        let t = ((11.8 - wy) / 9.8).clamp(0.0, 1.0);
        let half_w = 2.9 + t * 1.1;
        let cz_back = cz - 2.0 - t * 2.0;
        if (wx - cx).abs() > half_w || (wz - cz_back).abs() > 0.6 {
            return 0;
        }
        if wy < 3.0 { gold } else { team_d }
    });

    // Head + visored steel greathelm + towering Squad Anchor crest plume
    vol.rounded_box(cx - 2.1, 11.6, cz - 2.0, 4.2, 4.2, 4.0, 0.9, skin);
    vol.box_f(cx - 1.0, 13.2, cz + 1.85, 0.65, 0.85, 0.35, eye);
    vol.box_f(cx + 0.35, 13.2, cz + 1.85, 0.65, 0.85, 0.35, eye);
    vol.sculpt(cx - 2.6, 12.8, cz - 2.6, cx + 2.6, 16.9, cz + 2.6, |wx, wy, wz| {
        let dx = (wx - cx) / 2.55;
        let dz = (wz - cz) / 2.55;
        let dy = ((wy - 14.2) / 2.5).max(0.0);
        if dx * dx + dz * dz + dy * dy > 1.0 {
            return 0;
        }
        if wy < 14.2 && wz > cz + 1.15 && (wx - cx).abs() < 1.65 {
            return 0;
        }
        steel
    });
    vol.box_f(cx - 2.4, 14.1, cz + 1.7, 4.8, 0.7, 1.0, gold);
    vol.sculpt(cx - 0.85, 15.6, cz - 4.0, cx + 0.85, 21.2, cz + 1.5, |wx, wy, wz| {
        let u = ((cz + 1.3 - wz) / 5.1).clamp(0.0, 1.0);
        let y_bot = 16.4 - u * u * 1.4;
        let y_top = 20.4 - u * 1.0 + (u * PI).sin() * 0.85;
        if (wx - cx).abs() > 0.72 || wy < y_bot || wy > y_top {
            return 0;
        }
        if wy > y_top - 0.6 { gold } else { team }
    });

    // Heavy Fortress Tower Shield on left arm (-X) with embossed gold crest + anchor rune gem
    vol.sculpt(cx - 6.4, 4.2, cz - 1.1, cx - 4.3, 12.4, cz + 4.8, |wx, wy, wz| {
        let dz = (wz - (cz + 1.85)).abs();
        let t = ((12.4 - wy) / 8.2).clamp(0.0, 1.0);
        let half_z = 2.85 * (1.0 - t * t * 0.55);
        if dz > half_z || (wx - (cx - 5.25)).abs() > 0.65 {
            return 0;
        }
        if wx < cx - 5.45 && dz < 0.62 && (wy - 9.0).abs() < 0.72 {
            return anchor_rune;
        }
        if wx < cx - 5.45 && (dz < 0.45 || (wy - 9.0).abs() < 0.48) {
            return gold;
        }
        if dz > half_z - 0.42 || wy > 11.9 || wy < 4.7 {
            return steel;
        }
        team
    });

    // Heavy Crossguard Broadsword held forward on right arm (+X)
    vol.box_f(cx + 4.2, 7.0, cz + 0.8, 1.0, 1.0, 2.2, leather);
    vol.box_f(cx + 3.2, 7.0, cz + 2.8, 3.0, 1.0, 0.85, gold);
    vol.sculpt(cx + 4.1, 7.2, cz + 3.5, cx + 5.3, 11.2, cz + 8.6, |wx, wy, wz| {
        let u = ((wz - (cz + 3.5)) / 5.0).clamp(0.0, 1.0);
        let yc = 7.7 + u * 2.5;
        let half_h = 0.72 * (1.0 - u * 0.52);
        if (wx - (cx + 4.7)).abs() > 0.42 || (wy - yc).abs() > half_h {
            return 0;
        }
        blade
    });

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.42);
}

/// Archetype 1: SKYWING — True 3D Flyer / Thermal Glider Body
/// Streamlined aerodynamic flight harness, swept raptor crest helm with glowing visor,
/// glowing thermal flight core, twin dorsal wing mounts, and forked aero-tail.
/// Paired with articulated flapping `SkywingWingL` & `SkywingWingR` instances!
fn build_parametric_skywing(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(14.0, 21.0, 15.0, det);
    let (cx, cz) = (7.0f32, 7.6f32);

    let team = vol.color_mat(0xffffff, 12.0);
    let team_d = vol.color_mat(0xc2c2c2, 12.0);
    let aero_metal = vol.color_mat(0xdce6f5, 10.0);
    let gold = vol.color_mat(0xffcc48, 10.0);
    let slate = vol.color_mat(0x3a435a, 10.0);
    let thermal_glow = vol.color_mat(0xffc248, 5.0);
    let visor_glow = vol.color_mat(0x64f2ff, 5.0);

    // Tapered aerodynamic flight greaves (swept slightly back in glide stance)
    for s in [-1.0f32, 1.0] {
        let lx = cx + s * 1.25;
        vol.sculpt(lx - 0.9, 0.2, cz - 2.6, lx + 0.9, 6.2, cz + 1.4, |wx, wy, wz| {
            let t = ((wy - 0.2) / 6.0).clamp(0.0, 1.0);
            let zc = cz - 0.95 + t * 0.95;
            let r = 0.62 + t * 0.24;
            if (wx - lx).hypot(wz - zc) > r {
                return 0;
            }
            if wy < 2.2 { gold } else { slate }
        });
        // Small ankle aero-vanes
        vol.box_f(lx + s * 0.55 - 0.25, 1.6, cz - 1.8, 0.5, 2.2, 1.4, thermal_glow);
    }

    // Streamlined chevron flight torso + glowing thermal chest core
    vol.rounded_box(cx - 2.1, 6.0, cz - 1.5, 4.2, 5.8, 3.2, 0.85, team);
    vol.rounded_box(cx - 2.3, 8.2, cz - 1.7, 4.6, 3.2, 3.6, 0.75, aero_metal);
    vol.ellipsoid(cx, 9.8, cz + 1.7, 0.85, 1.15, 0.55, thermal_glow);
    vol.box_f(cx - 1.6, 6.1, cz - 1.6, 3.2, 0.9, 3.4, gold);

    // Sleek swept arms with glowing energy bracers
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 2.85;
        vol.ellipsoid(sx, 10.6, cz - 0.1, 1.15, 0.95, 1.25, gold);
        vol.rounded_box(sx - 0.65, 6.6, cz - 0.7, 1.3, 3.8, 1.5, 0.45, team_d);
        vol.box_f(sx - 0.7, 6.6, cz - 0.8, 1.4, 1.0, 1.6, thermal_glow);
    }

    // Dorsal wing-harness pylons on upper back (-Z) where articulated wings socket cleanly
    for s in [-1.0f32, 1.0] {
        vol.rounded_box(cx + s * 1.9 - 0.65, 9.6, cz - 2.8, 1.3, 2.6, 1.6, 0.4, gold);
        vol.ellipsoid(cx + s * 2.0, 11.0, cz - 2.6, 0.6, 0.7, 0.6, thermal_glow);
    }

    // Forked aerodynamic tail-fin mantle trailing behind (-Z)
    vol.sculpt(cx - 2.8, 2.6, cz - 5.4, cx + 2.8, 9.6, cz - 1.4, |wx, wy, wz| {
        let t = ((9.6 - wy) / 7.0).clamp(0.0, 1.0);
        let zc = cz - 1.6 - t * 3.1;
        let dx = (wx - cx).abs();
        let half_w = 1.6 + t * 1.05;
        let fork_gap = if t > 0.42 { (t - 0.42) * 1.6 } else { 0.0 };
        if dx > half_w || dx < fork_gap || (wz - zc).abs() > 0.45 {
            return 0;
        }
        if t > 0.80 { thermal_glow } else { team_d }
    });

    // Swept Falcon Aero-Helm + glowing visor + twin swept crest-fins
    vol.ellipsoid(cx, 13.8, cz + 0.1, 2.1, 2.3, 2.4, aero_metal);
    vol.box_f(cx - 1.45, 13.4, cz + 1.85, 2.9, 0.75, 0.75, visor_glow);
    vol.sculpt(cx - 2.6, 14.4, cz - 4.4, cx + 2.6, 19.2, cz + 1.6, |wx, wy, wz| {
        let u = ((cz + 1.2 - wz) / 5.2).clamp(0.0, 1.0);
        let yc = 15.2 + u * 2.8;
        let dx = (wx - cx).abs();
        let in_center = dx < 0.48 && (wy - yc).abs() < 0.72 * (1.0 - u * 0.45);
        let wing_x = 1.45 + u * 0.85;
        let in_ear = (dx - wing_x).abs() < 0.36 && (wy - (14.8 + u * 2.4)).abs() < 0.55;
        if in_center {
            if wy > yc + 0.2 { thermal_glow } else { gold }
        } else if in_ear {
            team
        } else {
            0
        }
    });

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.44);
}

/// Articulated Swept Glowing Elemental Wing for `HeroSkywing` (`side = -1.0` left, `+1.0` right).
/// Root shoulder hinge is translated to local `(0.0, 0.0, 0.0)` so `flags.z` (roll) flaps cleanly!
fn build_skywing_wing(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32, side: f32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(13.0, 12.0, 10.0, det);

    let team = vol.color_mat(0xffffff, 12.0);
    let gold_spar = vol.color_mat(0xffd254, 10.0);
    let glow_edge = vol.color_mat(0xffe066, 5.0);
    let glow_tip = vol.color_mat(0x5ef6ff, 5.0);

    let root_x = if side > 0.0 { 0.8f32 } else { 12.2f32 };
    let root_y = 4.6f32;
    let root_z = 7.8f32;

    vol.sculpt(0.0, 0.0, 0.0, 13.0, 12.0, 10.0, |wx, wy, wz| {
        let span = (wx - root_x) * side; // 0 at root .. ~11.6 at tip
        if span < 0.0 || span > 11.4 {
            return 0;
        }
        let u = (span / 11.4).clamp(0.0, 1.0);
        // Swept back and gracefully arched upward
        let z_lead = root_z - u * 3.8 - u * u * 1.4;
        let chord = 4.2 * (1.0 - u * 0.55) + (u * PI).sin() * 1.1;
        let z_trail = z_lead - chord;
        if wz > z_lead || wz < z_trail {
            return 0;
        }
        let chord_t = ((z_lead - wz) / chord.max(0.1)).clamp(0.0, 1.0);

        // Sculpted feather notches along trailing edge
        let feather_notch = ((u * 5.0 * PI).sin().abs()) * 0.22;
        if chord_t > 1.0 - feather_notch * u {
            return 0;
        }

        let y_camber = root_y + u * 3.4 + (u * PI).sin() * 1.4 - chord_t * (1.1 + u * 0.7);
        let thick = 0.62 * (1.0 - chord_t * 0.55) * (1.0 - u * 0.35);
        if (wy - y_camber).abs() > thick {
            return 0;
        }

        if chord_t < 0.19 {
            gold_spar
        } else if u > 0.76 || chord_t > 0.78 {
            if u > 0.82 { glow_tip } else { glow_edge }
        } else if (u * 5.0).fract() < 0.14 && chord_t > 0.35 {
            glow_edge
        } else {
            team
        }
    });

    let base_v = v_out.len();
    let unit_scale = 0.148;
    vol.compile_to_mesh(v_out, i_out, unit_scale, false, 0.36);

    let shift_x = root_x * unit_scale;
    let shift_y = root_y * unit_scale;
    let shift_z = root_z * unit_scale;
    for mv in &mut v_out[base_v..] {
        mv.pos[0] -= shift_x;
        mv.pos[1] -= shift_y;
        mv.pos[2] -= shift_z;
    }
}

/// Archetype 2: TIDEKIN — Amphibious Swimmer / Diver
/// Sleek hydrodynamic carapace, prominent tall swept dorsal crest/fin with aqua bioluminescent trim,
/// forearm & calf hydro-fins, glowing gill vents, and sculpted hydro-trident.
fn build_parametric_tidekin(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(16.0, 21.0, 17.0, det);
    let (cx, cz) = (8.0f32, 8.2f32);

    let team = vol.color_mat(0xffffff, 12.0);
    let deep_sea = vol.color_mat(0x163854, 10.0);
    let teal_shell = vol.color_mat(0x26748e, 10.0);
    let aqua_glow = vol.color_mat(0x3cf4ff, 5.0);
    let pearl = vol.color_mat(0xb8f0ff, 10.0);

    // Streamlined diver legs + flared webbed flippers + calf hydro-fins
    for s in [-1.0f32, 1.0] {
        let lx = cx + s * 1.35;
        vol.rounded_box(lx - 0.8, 0.0, cz - 1.0, 1.6, 5.6, 2.0, 0.55, deep_sea);
        // Flipper foot extending forward/back
        vol.box_f(lx - 1.0, 0.0, cz - 1.2, 2.0, 0.8, 3.2, teal_shell);
        // Swept calf fin
        vol.sculpt(lx + s * 0.6 - 0.6, 1.4, cz - 2.6, lx + s * 0.6 + 0.6, 4.6, cz - 0.6, |wx, wy, wz| {
            let t = ((wy - 1.4) / 3.2).clamp(0.0, 1.0);
            let z_edge = cz - 0.8 - (1.0 - t) * 1.5;
            if (wx - (lx + s * 0.65)).abs() > 0.35 || wz < z_edge {
                return 0;
            }
            if wz < z_edge + 0.45 { aqua_glow } else { teal_shell }
        });
    }

    // Sleek hydrodynamic torso with bioluminescent aqua gill stripes
    vol.rounded_box(cx - 2.3, 5.4, cz - 1.8, 4.6, 6.2, 3.6, 1.0, deep_sea);
    vol.rounded_box(cx - 2.4, 7.6, cz - 1.9, 4.8, 3.4, 3.9, 0.9, team);
    // Glowing aqua bioluminescent ribs & chest core
    vol.box_f(cx - 0.55, 7.8, cz + 1.85, 1.1, 2.4, 0.45, aqua_glow);
    for s in [-1.0f32, 1.0] {
        vol.box_f(cx + s * 1.8 - 0.35, 6.6, cz + 1.5, 0.7, 0.45, 0.5, aqua_glow);
        vol.box_f(cx + s * 1.9 - 0.35, 7.6, cz + 1.5, 0.7, 0.45, 0.5, aqua_glow);
    }

    // Sleek arms + forearm hydro-fins with aqua glow edge
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 3.1;
        vol.ellipsoid(sx, 10.6, cz, 1.25, 1.05, 1.45, teal_shell);
        vol.rounded_box(sx - 0.75, 6.4, cz - 0.85, 1.5, 3.8, 1.8, 0.5, deep_sea);
        vol.box_f(sx + s * 0.55 - 0.3, 6.6, cz - 2.1, 0.6, 2.8, 1.5, aqua_glow);
    }

    // Streamlined aquatic helm + glowing aqua visor
    vol.ellipsoid(cx, 13.7, cz + 0.15, 2.15, 2.25, 2.65, teal_shell);
    vol.box_f(cx - 1.55, 13.3, cz + 2.0, 3.1, 0.85, 0.75, aqua_glow);

    // Tall Swept Hydrodynamic Crest + Dorsal Fin running from crown down spine (-Z) with bioluminescent rim!
    vol.sculpt(cx - 0.85, 6.2, cz - 6.0, cx + 0.85, 20.6, cz + 1.8, |wx, wy, wz| {
        if (wx - cx).abs() > 0.48 {
            return 0;
        }
        // Upper cranial fin sweeping back from forehead
        let u_head = ((cz + 1.6 - wz) / 6.2).clamp(0.0, 1.0);
        let head_top = 16.0 + (u_head * PI * 0.85).sin() * 4.4;
        let head_bot = 14.2 - u_head * 2.2;
        let in_head_fin = wz >= cz - 4.6 && wy >= head_bot && wy <= head_top;

        // Tall dorsal spine fin projecting back from shoulders/mid-back
        let v_back = ((13.8 - wy) / 7.6).clamp(0.0, 1.0);
        let back_far_z = cz - 1.8 - (v_back * PI).sin() * 4.1;
        let in_dorsal = wy >= 6.2 && wy <= 13.8 && wz <= cz - 1.5 && wz >= back_far_z;

        if in_head_fin {
            if wy > head_top - 0.75 { aqua_glow } else { team }
        } else if in_dorsal {
            if wz < back_far_z + 0.75 { aqua_glow } else { teal_shell }
        } else {
            0
        }
    });

    // Sculpted Aqua-Tipped Hydro-Trident held forward on right side (+X)
    vol.column(cx + 3.9, 1.2, cz + 1.8, 15.0, 0.42, 0.36, pearl);
    vol.box_f(cx + 2.7, 15.8, cz + 1.35, 2.4, 0.75, 0.9, pearl);
    for px in [-0.95f32, 0.0, 0.95] {
        let h = if px.abs() < 0.1 { 3.6 } else { 2.7 };
        vol.column(cx + 3.9 + px, 16.4, cz + 1.8, h, 0.34, 0.10, aqua_glow);
    }

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.45);
}

/// Archetype 3: EMBERSTALKER — Solo Cavern Climber / Raider
/// Sleek dark hooded cloak silhouette, deep pointed cowl with glowing twin ember eyes,
/// climbing harness, and glowing twin curved ember-blades in both hands!
fn build_parametric_stalker(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(16.0, 20.0, 16.0, det);
    let (cx, cz) = (8.0f32, 7.8f32);

    let obsidian = vol.color_mat(0x1b1726, 0.0);
    let charcoal = vol.color_mat(0x312b40, 0.0);
    let dark_steel = vol.color_mat(0x524c64, 10.0);
    let team_trim = vol.color_mat(0xdce0ea, 12.0);
    let ember_core = vol.color_mat(0xff521b, 5.0);
    let ember_hot = vol.color_mat(0xffd24c, 5.0);

    // Agile crouched climber legs + metal piton/boot claws
    for s in [-1.0f32, 1.0] {
        let lx = cx + s * 1.45;
        vol.rounded_box(lx - 0.85, 0.0, cz - 0.95, 1.7, 5.4, 2.0, 0.48, obsidian);
        vol.box_f(lx - 0.95, 0.0, cz - 0.8, 1.9, 1.1, 2.5, dark_steel);
    }

    // Lithe hooded-raider tunic + crossed climbing harness + subtle team sash
    vol.rounded_box(cx - 2.3, 5.2, cz - 1.7, 4.6, 6.0, 3.4, 0.8, charcoal);
    vol.rounded_box(cx - 2.4, 5.4, cz - 1.8, 4.8, 1.1, 3.6, 0.6, team_trim);
    vol.box_f(cx - 0.55, 5.3, cz + 1.75, 1.1, 1.2, 0.45, ember_core);
    // Shoulder mantle
    vol.ellipsoid(cx, 11.1, cz - 0.1, 3.55, 1.25, 2.25, obsidian);

    // Flowing jagged shadow cloak sweeping down behind back (-Z) with ember-lit hem
    vol.sculpt(cx - 4.1, 1.4, cz - 5.2, cx + 4.1, 11.8, cz - 1.3, |wx, wy, wz| {
        let t = ((11.8 - wy) / 10.4).clamp(0.0, 1.0);
        let half_w = 2.6 + t * 1.4;
        let cz_back = cz - 1.6 - t * 2.8;
        let dx = (wx - cx).abs();
        let jag = ((dx * 1.9).sin().abs()) * 1.1;
        if dx > half_w || (wz - cz_back).abs() > 0.55 || wy < 1.6 + jag {
            return 0;
        }
        if wy < 2.6 + jag {
            ember_core
        } else if dx > half_w - 0.45 {
            team_trim
        } else {
            obsidian
        }
    });

    // Deep pointed hooded cowl overhanging shadowed face + glowing twin ember eyes!
    vol.sculpt(cx - 2.7, 11.6, cz - 4.2, cx + 2.7, 18.8, cz + 2.8, |wx, wy, wz| {
        let t = ((wy - 11.6) / 7.0).clamp(0.0, 1.0);
        let r = 2.65 * (1.0 - t * 0.72);
        let zc = cz + 0.15 - t * t * 2.9;
        let dx = wx - cx;
        let dz = wz - zc;
        if dx * dx + dz * dz > r * r {
            return 0;
        }
        // Carve shadowed cowl interior opening at front (+Z)
        if wy > 12.1 && wy < 14.9 && wz > cz + 0.95 && dx.abs() < 1.45 {
            return 0;
        }
        if wy < 12.4 && wz > cz + 0.4 {
            team_trim
        } else {
            obsidian
        }
    });
    // Shadowed inner head + blazing twin ember eyes
    vol.ellipsoid(cx, 13.4, cz + 0.1, 1.45, 1.55, 1.45, obsidian);
    vol.box_f(cx - 0.95, 13.3, cz + 1.35, 0.60, 0.50, 0.45, ember_hot);
    vol.box_f(cx + 0.35, 13.3, cz + 1.35, 0.60, 0.50, 0.45, ember_hot);

    // Arms + GLOWING TWIN EMBER-BLADES held in BOTH hands (-X and +X)!
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 3.15;
        vol.rounded_box(sx - 0.75, 6.4, cz - 0.7, 1.5, 4.0, 1.8, 0.45, charcoal);
        vol.box_f(sx - 0.55, 6.5, cz + 0.8, 1.1, 1.0, 1.6, dark_steel);
        // Glowing curved twin ember-blade sweeping forward & slightly outward
        vol.sculpt(sx - 1.4, 5.8, cz + 1.8, sx + 1.4, 10.6, cz + 7.6, |wx, wy, wz| {
            let u = ((wz - (cz + 1.8)) / 5.6).clamp(0.0, 1.0);
            let bx = sx + s * (0.25 + u * 0.65);
            let by = 7.0 + u * 2.2 - (1.0 - u) * 0.4;
            let half_h = 0.68 * (1.0 - u * 0.62);
            if (wx - bx).abs() > 0.32 || (wy - by).abs() > half_h {
                return 0;
            }
            if u > 0.55 || wy > by + half_h * 0.25 {
                ember_hot
            } else {
                ember_core
            }
        });
    }

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.44);
}

/// Archetype 4: BURROWER — Subterranean Crystal-Plated Golem
/// Stocky armored terracotta-amber chassis, heavy subterranean treads/claws,
/// twin conical spiral drill-gauntlets with glowing amber tips, and a blazing
/// amethyst/amber crystal dorsal spine erupting along the back and shoulders!
fn build_parametric_burrower(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(19.0, 20.0, 19.0, det);
    let (cx, cz) = (9.5f32, 8.2f32);

    let team = vol.color_mat(0xffffff, 12.0);
    let terracotta = vol.color_mat(0xcc6c3c, 1.0);
    let terracotta_d = vol.color_mat(0x884224, 1.0);
    let basalt = vol.color_mat(0x362f3c, 1.0);
    let bronze = vol.color_mat(0xdc9844, 10.0);
    let drill_steel = vol.color_mat(0xc2cedc, 10.0);
    let amber_glow = vol.color_mat(0xffaa30, 5.0);
    let amethyst_glow = vol.color_mat(0xc95cff, 5.0);

    // Wide heavy subterranean treads & digging claws
    for s in [-1.0f32, 1.0] {
        let lx = cx + s * 2.45;
        // Armored basalt tread housing
        vol.rounded_box(lx - 1.45, 0.0, cz - 2.35, 2.9, 2.3, 4.9, 0.5, basalt);
        // Stocky hydraulic piston legs
        vol.rounded_box(lx - 1.25, 2.0, cz - 1.35, 2.5, 3.0, 2.8, 0.55, terracotta_d);
        // Bronze tread cleats & twin forward subterranean digging claws
        vol.box_f(lx - 1.55, 0.25, cz - 1.9, 3.1, 0.75, 0.8, bronze);
        vol.box_f(lx - 1.55, 0.25, cz + 0.1, 3.1, 0.75, 0.8, bronze);
        for c_off in [-0.75f32, 0.75] {
            vol.column(lx + c_off, 0.2, cz + 2.65, 1.3, 0.45, 0.14, bronze);
        }
    }

    // Stocky broad crystal-plated terracotta golem torso + faction chest carapace + amber furnace core
    vol.rounded_box(cx - 3.65, 4.4, cz - 2.65, 7.3, 6.2, 5.3, 1.0, terracotta);
    vol.rounded_box(cx - 3.10, 5.8, cz - 2.25, 6.2, 4.2, 4.9, 0.85, team);
    vol.rounded_box(cx - 3.85, 4.3, cz - 2.75, 7.7, 1.4, 5.5, 0.65, basalt);
    vol.box_f(cx - 1.4, 4.4, cz + 2.55, 2.8, 1.2, 0.6, bronze);
    // Glowing amber core furnace nestled in armored chest
    vol.ellipsoid(cx, 7.8, cz + 2.55, 1.2, 1.05, 0.6, amber_glow);
    vol.box_f(cx - 1.8, 7.3, cz + 2.35, 3.6, 0.35, 0.55, bronze);

    // Massive hunched boulder-plated pauldrons
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 4.45;
        vol.ellipsoid(sx, 9.6, cz - 0.15, 2.25, 1.85, 2.45, terracotta_d);
        vol.ellipsoid(sx + s * 0.2, 10.6, cz - 0.15, 1.75, 0.85, 2.0, bronze);
        vol.ellipsoid(sx + s * 0.55, 9.2, cz - 0.1, 1.45, 0.95, 1.8, team);
    }

    // Low-set armored subterranean golem head / brow nestled between shoulders
    vol.rounded_box(cx - 2.35, 10.2, cz - 1.75, 4.7, 3.5, 4.1, 0.85, terracotta);
    vol.rounded_box(cx - 2.55, 12.2, cz - 1.85, 5.1, 1.5, 4.4, 0.65, terracotta_d);
    vol.box_f(cx - 1.55, 11.1, cz + 2.05, 3.1, 0.9, 0.6, basalt);
    // Twin glowing amber visor eyes + crown crystal ridge
    vol.box_f(cx - 1.25, 11.25, cz + 2.35, 0.9, 0.55, 0.45, amber_glow);
    vol.box_f(cx + 0.35, 11.25, cz + 2.35, 0.9, 0.55, 0.45, amber_glow);
    vol.column(cx, 13.4, cz - 0.3, 2.4, 0.72, 0.18, amethyst_glow);

    // Glowing Amethyst & Amber Crystal Dorsal Spine erupting along the back & shoulders (-Z)
    vol.sculpt(cx - 4.6, 6.2, cz - 6.6, cx + 4.6, 18.8, cz - 1.0, |wx, wy, wz| {
        // 5 sculpted angled crystal spires: (base_x, base_y, base_z, dir_x, dir_y, dir_z, len, rad, is_amethyst)
        let spires: [(f32, f32, f32, f32, f32, f32, f32, f32, bool); 5] = [
            (cx, 10.6, cz - 2.0, 0.0, 0.82, -0.57, 7.6, 1.15, true),
            (cx - 2.0, 10.0, cz - 2.2, -0.26, 0.78, -0.56, 6.0, 0.95, false),
            (cx + 2.0, 10.0, cz - 2.2, 0.26, 0.78, -0.56, 6.0, 0.95, false),
            (cx - 1.2, 7.4, cz - 2.4, -0.18, 0.48, -0.86, 4.6, 0.85, true),
            (cx + 1.2, 7.4, cz - 2.4, 0.18, 0.48, -0.86, 4.6, 0.85, true),
        ];
        for (bx, by, bz, dx, dy, dz, len, rad, is_amethyst) in spires {
            let rx = wx - bx;
            let ry = wy - by;
            let rz = wz - bz;
            let proj = rx * dx + ry * dy + rz * dz;
            if proj < 0.0 || proj > len {
                continue;
            }
            let t = proj / len;
            let px = rx - dx * proj;
            let py = ry - dy * proj;
            let pz = rz - dz * proj;
            let d = (px * px + py * py + pz * pz).sqrt();
            let max_r = rad * (1.0 - t * 0.84) * (0.88 + 0.14 * (t * PI).sin());
            if d <= max_r {
                return if t > 0.72 {
                    if is_amethyst { amber_glow } else { amethyst_glow }
                } else if is_amethyst {
                    amethyst_glow
                } else {
                    amber_glow
                };
            }
        }
        0
    });

    // Twin Conical Spiral Drill-Gauntlets extending forward (+Z) on both arms (-X and +X)!
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 4.75;
        let arm_y = 6.9f32;
        // Heavy forearm housing + bronze rotary collar
        vol.rounded_box(sx - 1.25, arm_y - 1.35, cz - 1.1, 2.5, 2.7, 3.6, 0.55, basalt);
        vol.rounded_box(sx - 1.45, arm_y - 1.55, cz + 2.1, 2.9, 3.1, 1.3, 0.45, bronze);
        // Sculpted conical spiral auger drill bit (+Z)
        vol.sculpt(sx - 1.85, arm_y - 1.85, cz + 3.2, sx + 1.85, arm_y + 1.85, cz + 9.4, |wx, wy, wz| {
            let u = ((wz - (cz + 3.2)) / 6.0).clamp(0.0, 1.0);
            let dx = wx - sx;
            let dy = wy - arm_y;
            let r = dx.hypot(dy);
            let ang = dy.atan2(dx);
            let flute = (u * 16.0 + ang * 2.0 * s).sin();
            let cone_r = (1.52 * (1.0 - u * 0.88)).max(0.16) + flute * 0.20 * (1.0 - u * 0.5);
            if r > cone_r || wz > cz + 9.2 {
                return 0;
            }
            if u > 0.76 {
                amber_glow
            } else if flute > 0.45 {
                if u > 0.45 { amethyst_glow } else { bronze }
            } else {
                drill_steel
            }
        });
    }

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.42);
}

/// Parametric Sculpted Carved Ice Raft (`CellBody` beveled glacial floe with frosted rim & cyan veins!)
fn build_parametric_ice_raft(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(14.0, 5.0, 14.0, det);
    let (cx, cz) = (7.0f32, 7.0f32);

    let ice_deep = vol.color_mat(0x3ca0d8, 6.0);
    let ice_mid = vol.color_mat(0x88dcf8, 6.0);
    let ice_snow = vol.color_mat(0xe6f8ff, 6.0);
    let ice_core = vol.color_mat(0x6ce8ff, 5.0);

    vol.sculpt(0.5, 0.0, 0.5, 13.5, 4.6, 13.5, |wx, wy, wz| {
        let dx = wx - cx;
        let dz = wz - cz;
        let ang = dz.atan2(dx);
        let r = dx.hypot(dz);
        // Sculpted 7-faceted beveled ice floe shape
        let facet = 1.0 + 0.11 * (ang * 7.0 + 0.4).cos() + 0.05 * (ang * 3.0).sin();
        let top_bevel = if wy > 2.4 { 1.0 - (wy - 2.4) * 0.14 } else { 0.82 + (wy / 2.4) * 0.18 };
        let max_r = 5.6 * facet * top_bevel;
        if r > max_r || wy > 3.8 {
            return 0;
        }
        // Raised frosted outer ice rim around walking deck
        if wy > 3.1 && r < max_r - 1.2 {
            return 0;
        }
        if wy < 1.3 {
            ice_deep
        } else if wy > 2.85 {
            ice_snow
        } else if r < 1.6 && wy > 2.1 {
            ice_core
        } else {
            ice_mid
        }
    });

    // Compile at unit radius ~1.0 so instance scale = raft.radius!
    vol.compile_to_mesh(v_out, i_out, 1.0 / 5.4, true, 0.35);
}

/// Sculpted Stepped Rune Pedestal for the Golden Relic
fn build_relic_pedestal(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(14.0, 10.0, 14.0, det);
    let (cx, cz) = (7.0f32, 7.0f32);

    let stone_d = vol.color_mat(0x3a4864, 1.0);
    let stone_l = vol.color_mat(0x768aa8, 1.0);
    let gold = vol.color_mat(0xebb93d, 10.0);
    let rune = vol.color_mat(0x74ecff, 5.0);

    vol.column(cx, 0.0, cz, 2.0, 6.2, 5.6, stone_d);
    vol.column(cx, 2.0, cz, 2.0, 4.8, 4.3, stone_l);
    vol.column(cx, 4.0, cz, 1.4, 3.6, 3.3, gold);
    vol.column(cx, 5.4, cz, 0.7, 3.1, 3.0, rune);
    // 4 carved corner braziers/pillars
    for k in 0..4 {
        let a = (k as f32 + 0.5) * PI * 0.5;
        let px = cx + a.cos() * 4.5;
        let pz = cz + a.sin() * 4.5;
        vol.column(px, 1.6, pz, 4.6, 0.85, 0.68, stone_l);
        vol.column(px, 6.2, pz, 1.2, 0.72, 0.15, gold);
    }

    vol.compile_to_mesh(v_out, i_out, 0.36, true, 0.32);
}

/// Sculpted Floating Golden Relic Crown / Chalice
fn build_golden_relic(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(10.0, 14.0, 10.0, det);
    let (cx, cz) = (5.0f32, 5.0f32);

    let bronze = vol.color_mat(0xb87e22, 10.0);
    let gold = vol.color_mat(0xf5c842, 10.0);
    let sun_core = vol.color_mat(0xfff48a, 5.0);

    vol.column(cx, 0.5, cz, 2.4, 2.2, 1.0, bronze);
    vol.column(cx, 2.9, cz, 2.6, 1.0, 2.8, gold);
    vol.column(cx, 5.5, cz, 2.2, 2.8, 3.2, gold);
    // Carve hollow chalice cup & place blazing golden sun gem inside
    vol.carve(cx - 2.2, 6.2, cz - 2.2, cx + 2.2, 8.2, cz + 2.2, |wx, _, wz| {
        (wx - cx).hypot(wz - cz) <= 2.0
    });
    vol.ellipsoid(cx, 8.6, cz, 1.8, 2.4, 1.8, sun_core);
    // 4 sculpted crown prongs around sun gem
    for k in 0..4 {
        let a = k as f32 * PI * 0.5;
        let px = cx + a.cos() * 2.5;
        let pz = cz + a.sin() * 2.5;
        vol.column(px, 6.5, pz, 4.2, 0.55, 0.18, gold);
    }

    vol.compile_to_mesh(v_out, i_out, 0.24, true, 0.40);
}

/// Sculpted Parametric Frostmarsh Wildlife Elk/Deer (spawned from live `ecosystem_model.pt` herds!)
fn build_marsh_deer(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(10.0, 12.0, 14.0, det);
    let (cx, cz) = (5.0f32, 7.0f32);

    let coat = vol.color_mat(0xc88d56, 0.0);
    let coat_hi = vol.color_mat(0xe6c394, 0.0);
    let hoof = vol.color_mat(0x3a261e, 0.0);
    let antler = vol.color_mat(0xefe4cc, 0.0);

    // 4 slender legs
    for sx in [-1.1f32, 1.1] {
        for sz in [-2.2f32, 2.0] {
            vol.column(cx + sx, 0.0, cz + sz, 4.4, 0.45, 0.52, hoof);
        }
    }
    // Rounded body & white rump tail
    vol.ellipsoid(cx, 5.2, cz, 1.9, 1.7, 3.4, coat);
    vol.ellipsoid(cx, 5.6, cz - 3.3, 0.85, 0.75, 0.85, coat_hi);
    // Neck & head
    vol.column(cx, 5.6, cz + 2.6, 3.0, 1.1, 0.85, coat);
    vol.ellipsoid(cx, 8.6, cz + 3.4, 1.0, 0.95, 1.65, coat_hi);
    // Sweeping white antlers!
    for s in [-1.0f32, 1.0] {
        vol.sculpt(cx + s * 0.4 - 2.6, 9.2, cz + 1.8, cx + s * 0.4 + 2.6, 11.8, cz + 4.4, |wx, wy, wz| {
            let t = ((wy - 9.2) / 2.4).clamp(0.0, 1.0);
            let ax = cx + s * (0.6 + t * 1.9);
            let az = cz + 2.9 - t * 0.7;
            if (wx - ax).hypot(wz - az) <= 0.34 {
                antler
            } else {
                0
            }
        });
    }

    vol.compile_to_mesh(v_out, i_out, 0.145, true, 0.45);
}

/// Sculpted Rock Outcrop / Marsh Boulder
fn build_rock_outcrop(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(12.0, 12.0, 12.0, det);
    let (cx, cz) = (6.0f32, 6.0f32);

    let dark = vol.color_mat(0x424f68, 1.0);
    let mid = vol.color_mat(0x667896, 1.0);
    let snow = vol.color_mat(0xd4e6f5, 6.0);

    vol.sculpt(0.5, 0.0, 0.5, 11.5, 11.5, 11.5, |wx, wy, wz| {
        let dx = wx - cx;
        let dz = wz - cz;
        let a = dz.atan2(dx);
        let r = dx.hypot(dz);
        let h_norm = wy / 10.5;
        let rock_r = 4.9 * (1.0 - h_norm * 0.62) * (1.0 + 0.16 * (a * 5.0 + wy * 0.4).sin());
        if r > rock_r || wy > 10.2 {
            return 0;
        }
        if wy > 8.1 && hash2i(wx as i32, wz as i32) > 0.28 {
            snow
        } else if wy < 3.2 {
            dark
        } else {
            mid
        }
    });

    vol.compile_to_mesh(v_out, i_out, 0.24, true, 0.30);
}

fn build_ring_mesh(
    verts: &mut Vec<MeshVertex>,
    idxs: &mut Vec<u32>,
    r_in: f32,
    r_out: f32,
    y0: f32,
    y1: f32,
    segs: usize,
    arc_frac: f32,
    col: [f32; 3],
    mat: f32,
) {
    let total_ang = TAU * arc_frac;
    let start_ang = -total_ang * 0.5;
    for i in 0..segs {
        let a0 = start_ang + (i as f32 / segs as f32) * total_ang;
        let a1 = start_ang + ((i + 1) as f32 / segs as f32) * total_ang;
        let (c0, s0) = (a0.cos(), a0.sin());
        let (c1, s1) = (a1.cos(), a1.sin());
        let i0_t = Vec3::new(c0 * r_in, y1, s0 * r_in);
        let o0_t = Vec3::new(c0 * r_out, y1, s0 * r_out);
        let o1_t = Vec3::new(c1 * r_out, y1, s1 * r_out);
        let i1_t = Vec3::new(c1 * r_in, y1, s1 * r_in);
        let o0_b = Vec3::new(c0 * r_out, y0, s0 * r_out);
        let o1_b = Vec3::new(c1 * r_out, y0, s1 * r_out);
        // Top band + outer wall
        push_quad(verts, idxs, i0_t, i1_t, o1_t, o0_t, col, mat);
        push_quad(verts, idxs, o0_b, o0_t, o1_t, o1_b, col, mat);
    }
}

fn build_octahedron(
    verts: &mut Vec<MeshVertex>,
    idxs: &mut Vec<u32>,
    rx: f32,
    ry: f32,
    rz: f32,
    col: [f32; 3],
    mat: f32,
) {
    let top = Vec3::new(0.0, ry, 0.0);
    let bot = Vec3::new(0.0, -ry, 0.0);
    let ring = [
        Vec3::new(rx, 0.0, 0.0),
        Vec3::new(0.0, 0.0, rz),
        Vec3::new(-rx, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -rz),
    ];
    for k in 0..4 {
        let a = ring[k];
        let b = ring[(k + 1) % 4];
        let mut push_t = |p0: Vec3, p1: Vec3, p2: Vec3| {
            let n = (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Y);
            let base = verts.len() as u32;
            let cm = [col[0], col[1], col[2], mat];
            for p in [p0, p1, p2] {
                verts.push(MeshVertex {
                    pos: p.to_array(),
                    normal: n.to_array(),
                    color_mat: cm,
                    extra: [0.0; 4],
                });
            }
            idxs.extend_from_slice(&[base, base + 1, base + 2]);
        };
        push_t(a, top, b);
        push_t(b, bot, a);
    }
}

impl StaticMeshBank {
    pub fn build(detail: u32) -> Self {
        let mut verts = Vec::with_capacity(48_000);
        let mut idxs = Vec::with_capacity(96_000);

        let mut make_slice = |f: &mut dyn FnMut(&mut Vec<MeshVertex>, &mut Vec<u32>)| -> MeshSlice {
            let first = idxs.len() as u32;
            f(&mut verts, &mut idxs);
            MeshSlice {
                first_index: first,
                index_count: idxs.len() as u32 - first,
            }
        };

        let strider = make_slice(&mut |v, i| build_parametric_strider(v, i, detail));
        let skywing = make_slice(&mut |v, i| build_parametric_skywing(v, i, detail));
        let ice_raft = make_slice(&mut |v, i| build_parametric_ice_raft(v, i, detail));
        let relic_pedestal = make_slice(&mut |v, i| build_relic_pedestal(v, i, detail));
        let golden_relic = make_slice(&mut |v, i| build_golden_relic(v, i, detail));

        // 5: Dash i-frame glowing arcane halo ring
        let dash_halo = make_slice(&mut |v, i| {
            build_ring_mesh(v, i, 1.35, 1.85, 0.25, 0.72, 16, 1.0, [0.55, 0.94, 1.0], 5.0);
            build_ring_mesh(v, i, 1.05, 1.28, 1.45, 1.72, 16, 1.0, [0.85, 0.98, 1.0], 5.0);
        });

        // 6: Melee slash arc
        let melee_arc = make_slice(&mut |v, i| {
            build_ring_mesh(v, i, 1.3, 2.9, 1.15, 1.48, 10, 0.35, [1.0, 0.95, 0.72], 5.0);
        });

        // 7: Faceted elemental projectile orb (tinted per element: Fire/Frost/Oil)
        let projectile_orb = make_slice(&mut |v, i| {
            build_octahedron(v, i, 0.58, 0.58, 0.95, [1.0, 1.0, 1.0], 5.0);
        });

        // 8: Sculpted Forked Pine Trunk with buttress roots
        let tree_trunk = make_slice(&mut |v, i| {
            let bark = [0.46, 0.29, 0.21];
            let bark_d = [0.32, 0.20, 0.15];
            push_limb(v, i, Vec3::new(0.0, -0.15, 0.0), Vec3::new(0.04, 1.25, 0.02), 0.36, 0.22, 6, bark_d, 4.0);
            push_limb(v, i, Vec3::new(0.04, 1.25, 0.02), Vec3::new(-0.02, 2.50, -0.02), 0.22, 0.14, 6, bark, 4.0);
            for k in 0..4 {
                let a = k as f32 * PI * 0.5 + 0.35;
                push_limb(
                    v,
                    i,
                    Vec3::new(a.cos() * 0.14, 0.42, a.sin() * 0.14),
                    Vec3::new(a.cos() * 0.62, -0.10, a.sin() * 0.62),
                    0.15,
                    0.07,
                    4,
                    bark_d,
                    4.0,
                );
            }
        });

        // 9: Rock Outcrop
        let rock_outcrop = make_slice(&mut |v, i| build_rock_outcrop(v, i, detail));

        // 10: Marsh Deer (live ecosystem wildlife!)
        let marsh_deer = make_slice(&mut |v, i| build_marsh_deer(v, i, detail));

        // 11: Crisp Faction Ground Ring
        let ground_ring = make_slice(&mut |v, i| {
            build_ring_mesh(v, i, 1.05, 1.40, 0.06, 0.16, 16, 1.0, [1.0, 1.0, 1.0], 5.0);
        });

        // 12: Tidekin (Amphibious Swimmer / Diver with hydrodynamic dorsal crest & bioluminescent trim!)
        let hero_tidekin = make_slice(&mut |v, i| build_parametric_tidekin(v, i, detail));

        // 13: Capturable Elemental Shrine Obelisk (stepped stone plinth + glowing floating crystal!)
        let shrine_obelisk = make_slice(&mut |v, i| {
            let stone = [0.36, 0.40, 0.52];
            let stone_hi = [0.54, 0.58, 0.70];
            // Stepped octagonal base
            push_limb(v, i, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.55, 0.0), 2.35, 2.10, 8, stone, 1.0);
            push_limb(v, i, Vec3::new(0.0, 0.55, 0.0), Vec3::new(0.0, 1.05, 0.0), 1.65, 1.40, 8, stone_hi, 1.0);
            // Four corner rune pillars
            for k in 0..4 {
                let a = k as f32 * PI * 0.5 + PI * 0.25;
                let px = a.cos() * 1.45;
                let pz = a.sin() * 1.45;
                push_limb(v, i, Vec3::new(px, 0.5, pz), Vec3::new(px, 2.35, pz), 0.26, 0.18, 6, stone_hi, 1.0);
            }
            // Floating faction-colored crystal spire (mat 5 emissive + tinted by instance!)
            let base = v.len();
            build_octahedron(v, i, 0.85, 0.85, 2.15, [1.0, 1.0, 1.0], 5.0);
            for mv in &mut v[base..] {
                mv.pos[1] += 3.55;
            }
        });

        // 14: Emberstalker (Solo Cavern Climber / Raider — dark hooded cloak + glowing twin ember-blades!)
        let hero_stalker = make_slice(&mut |v, i| build_parametric_stalker(v, i, detail));

        // 15 & 16: Articulated Swept Glowing Elemental Wings for Skywing (shoulder hinge at origin!)
        let skywing_wing_l = make_slice(&mut |v, i| build_skywing_wing(v, i, detail, -1.0));
        let skywing_wing_r = make_slice(&mut |v, i| build_skywing_wing(v, i, detail, 1.0));

        // 17: Burrower (Subterranean Crystal-Plated Golem — twin conical drill-gauntlets + crystal dorsal spine!)
        let hero_burrower = make_slice(&mut |v, i| build_parametric_burrower(v, i, detail));

        Self {
            vertices: verts,
            indices: idxs,
            slices: [
                strider,
                skywing,
                ice_raft,
                relic_pedestal,
                golden_relic,
                dash_halo,
                melee_arc,
                projectile_orb,
                tree_trunk,
                rock_outcrop,
                marsh_deer,
                ground_ring,
                hero_tidekin,
                shrine_obelisk,
                hero_stalker,
                skywing_wing_l,
                skywing_wing_r,
                hero_burrower,
            ],
        }
    }
}
