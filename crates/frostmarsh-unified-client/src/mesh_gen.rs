//! Approach 4 Parametric CSG/SDF Greedy-Meshed Baked-AO 3D Characters & Props for Live Frostmarsh.
//!
//! Includes:
//! - HeroKnight (FrostBots — blue-tabard paladin with steel cuirass, visor helm, shield & broadsword)
//! - HeroMage (PyroBots — crimson wizard with curved pointed hat, gold stole & glowing crystal staff)
//! - IceRaft (beveled carved glacial CellBody raft with frosted rim & translucent cyan core)
//! - RelicPedestal & GoldenRelic (stepped rune altar & sculpted golden crown/chalice)
//! - DashHalo & MeleeArc (dash i-frame shield halo & combat slash arc)
//! - ProjectileOrb, TreeTrunk, RockOutcrop, MarshDeer (ecosystem herds!), GroundRing

use crate::parametric::{hash2i, ParamVolume};
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::f32::consts::{PI, TAU};

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
    pub flags: [f32; 4], // x = mat_override (<0 keep vertex mat), y = selective_tint (>0.5 only mat 12)
}

#[derive(Clone, Copy, Debug)]
pub struct MeshSlice {
    pub first_index: u32,
    pub index_count: u32,
}

pub const MESH_KIND_COUNT: usize = 14;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MeshKind {
    HeroKnight = 0,
    HeroMage = 1,
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
    HeroRanger = 12,
    ShrineObelisk = 13,
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

/// Parametric Sculpted Knight / FrostBot Paladin (continuous CSG -> greedy-merged 3D mesh with baked AO!)
fn build_parametric_knight(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(16.0, 21.0, 16.0, det);
    let (cx, cz) = (8.0f32, 7.6f32);

    let team = vol.color_mat(0xffffff, 12.0);
    let team_d = vol.color_mat(0xc8c8c8, 12.0);
    let steel = vol.color_mat(0xc8d8eb, 10.0);
    let steel_d = vol.color_mat(0x6e7e96, 10.0);
    let gold = vol.color_mat(0xf5c84c, 10.0);
    let leather = vol.color_mat(0x5c3828, 0.0);
    let skin = vol.color_mat(0xf0c29e, 0.0);
    let eye = vol.color_mat(0x262038, 0.0);
    let blade = vol.color_mat(0xdcf5ff, 10.0);

    // Armored boots & greaves
    for s in [-1.0f32, 1.0] {
        let lx = cx + s * 1.4;
        vol.rounded_box(lx - 0.9, 0.0, cz - 1.1, 1.8, 5.2, 2.2, 0.45, steel_d);
        vol.box_f(lx - 1.0, 0.0, cz - 1.0, 2.0, 1.4, 2.8, steel);
    }

    // Waist belt + steel cuirass + faction tabard
    vol.rounded_box(cx - 2.6, 5.2, cz - 1.9, 5.2, 6.2, 3.8, 0.85, team);
    vol.rounded_box(cx - 2.4, 7.2, cz - 1.6, 4.8, 3.8, 3.8, 0.75, steel);
    vol.rounded_box(cx - 2.7, 5.2, cz - 2.0, 5.4, 1.2, 4.0, 0.65, leather);
    vol.box_f(cx - 0.8, 5.1, cz + 1.8, 1.6, 1.4, 0.55, gold);

    // Layered rounded pauldrons + gauntleted arms
    for s in [-1.0f32, 1.0] {
        let sx = cx + s * 3.4;
        vol.ellipsoid(sx, 10.8, cz, 1.65, 1.35, 1.95, steel);
        vol.ellipsoid(sx + s * 0.25, 11.4, cz, 1.45, 0.55, 1.85, gold);
        vol.rounded_box(sx - 0.9, 6.4, cz - 0.8, 1.8, 3.8, 2.0, 0.55, steel_d);
    }

    // Flowing curved cape sweeping down behind back (-Z)
    vol.sculpt(cx - 3.6, 2.0, cz - 4.4, cx + 3.6, 11.6, cz - 1.6, |wx, wy, wz| {
        let t = ((11.6 - wy) / 9.6).clamp(0.0, 1.0);
        let half_w = 2.6 + t * 1.0;
        let cz_back = cz - 1.9 - t * 1.9;
        if (wx - cx).abs() > half_w || (wz - cz_back).abs() > 0.55 {
            return 0;
        }
        if wy < 2.9 { gold } else { team_d }
    });

    // Head + visored steel knight helmet + sweeping crest plume
    vol.rounded_box(cx - 2.1, 11.4, cz - 2.0, 4.2, 4.2, 4.0, 0.9, skin);
    vol.box_f(cx - 1.0, 13.0, cz + 1.85, 0.65, 0.85, 0.35, eye);
    vol.box_f(cx + 0.35, 13.0, cz + 1.85, 0.65, 0.85, 0.35, eye);
    vol.sculpt(cx - 2.5, 12.6, cz - 2.5, cx + 2.5, 16.6, cz + 2.5, |wx, wy, wz| {
        let dx = (wx - cx) / 2.45;
        let dz = (wz - cz) / 2.45;
        let dy = ((wy - 14.0) / 2.4).max(0.0);
        if dx * dx + dz * dz + dy * dy > 1.0 {
            return 0;
        }
        if wy < 14.0 && wz > cz + 1.1 && (wx - cx).abs() < 1.65 {
            return 0;
        }
        steel
    });
    vol.box_f(cx - 2.3, 13.9, cz + 1.65, 4.6, 0.65, 0.95, gold);
    vol.sculpt(cx - 0.75, 15.2, cz - 3.6, cx + 0.75, 19.6, cz + 1.4, |wx, wy, wz| {
        let u = ((cz + 1.2 - wz) / 4.6).clamp(0.0, 1.0);
        let y_bot = 16.0 - u * u * 1.4;
        let y_top = 18.8 - u * 0.9 + (u * PI).sin() * 0.7;
        if (wx - cx).abs() > 0.65 || wy < y_bot || wy > y_top {
            return 0;
        }
        team
    });

    // Curved Heraldic Kite Shield on left arm (-X) with embossed gold cross
    vol.sculpt(cx - 5.4, 4.8, cz - 0.8, cx - 3.6, 11.4, cz + 4.4, |wx, wy, wz| {
        let dz = (wz - (cz + 1.8)).abs();
        let t = ((11.4 - wy) / 6.6).clamp(0.0, 1.0);
        let half_z = 2.35 * (1.0 - t * t * 0.75);
        if dz > half_z || (wx - (cx - 4.4)).abs() > 0.55 {
            return 0;
        }
        if wx < cx - 4.55 && (dz < 0.42 || (wy - 8.8).abs() < 0.42) {
            return gold;
        }
        if dz > half_z - 0.35 || wy > 11.0 {
            return steel;
        }
        team
    });

    // Crossguard Broadsword held forward on right arm (+X)
    vol.box_f(cx + 3.6, 7.0, cz + 0.8, 0.9, 0.9, 2.2, leather);
    vol.box_f(cx + 2.7, 7.0, cz + 2.8, 2.7, 0.9, 0.8, gold);
    vol.sculpt(cx + 3.5, 7.2, cz + 3.5, cx + 4.6, 10.6, cz + 8.4, |wx, wy, wz| {
        let u = ((wz - (cz + 3.5)) / 4.8).clamp(0.0, 1.0);
        let yc = 7.6 + u * 2.2;
        let half_h = 0.65 * (1.0 - u * 0.55);
        if (wx - (cx + 4.05)).abs() > 0.38 || (wy - yc).abs() > half_h {
            return 0;
        }
        blade
    });

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.42);
}

/// Parametric Sculpted PyroBot Arcane Mage (curved wide-brimmed wizard hat, layered robe, glowing fire orb staff!)
fn build_parametric_mage(v_out: &mut Vec<MeshVertex>, i_out: &mut Vec<u32>, detail: u32) {
    let det = detail.clamp(1, 2);
    let mut vol = ParamVolume::new(15.0, 22.0, 15.0, det);
    let (cx, cz) = (7.5f32, 7.5f32);

    let team = vol.color_mat(0xffffff, 12.0);
    let team_d = vol.color_mat(0xc2c2c2, 12.0);
    let gold = vol.color_mat(0xf5c84c, 10.0);
    let skin = vol.color_mat(0xf0c29e, 0.0);
    let eye = vol.color_mat(0x262038, 0.0);
    let wood = vol.color_mat(0x5c3220, 0.0);
    let orb = vol.color_mat(0xff9c38, 5.0);

    // Flared arcane robe skirt + upper tunic + gold stole
    vol.column(cx, 0.2, cz, 7.6, 3.8, 2.4, team_d);
    vol.rounded_box(cx - 2.4, 7.6, cz - 1.8, 4.8, 4.0, 3.6, 0.8, team);
    vol.box_f(cx - 0.85, 1.4, cz + 1.5, 1.7, 9.8, 1.2, gold);
    for s in [-1.0f32, 1.0] {
        vol.rounded_box(cx + s * 3.1 - 0.9, 6.6, cz - 1.0, 1.8, 4.2, 2.2, 0.55, team);
    }

    // Expressive rounded head
    vol.rounded_box(cx - 2.0, 11.4, cz - 1.9, 4.0, 3.8, 3.8, 0.9, skin);
    vol.box_f(cx - 0.95, 12.9, cz + 1.75, 0.60, 0.80, 0.35, eye);
    vol.box_f(cx + 0.35, 12.9, cz + 1.75, 0.60, 0.80, 0.35, eye);

    // Wide-brimmed curved pointed Wizard Hat + gold hatband
    vol.column(cx, 14.4, cz, 0.85, 4.5, 4.0, team_d);
    vol.column(cx, 15.1, cz, 0.85, 2.7, 2.5, gold);
    vol.sculpt(cx - 3.2, 15.8, cz - 4.0, cx + 3.2, 21.8, cz + 3.0, |wx, wy, wz| {
        let t = ((wy - 15.8) / 5.8).clamp(0.0, 1.0);
        let r = 2.55 * (1.0 - t).powf(0.85);
        let bend_z = cz - t * t * 1.9;
        let dx = wx - cx;
        let dz = wz - bend_z;
        if dx * dx + dz * dz <= r * r { team } else { 0 }
    });

    // Gnarled Fire Staff + Golden Claw + Glowing Ember Orb (+X)
    vol.column(cx + 4.1, 0.6, cz + 2.2, 16.2, 0.52, 0.45, wood);
    vol.column(cx + 4.1, 16.5, cz + 2.2, 1.1, 0.95, 1.15, gold);
    vol.ellipsoid(cx + 4.1, 18.7, cz + 2.2, 1.25, 1.55, 1.25, orb);

    vol.compile_to_mesh(v_out, i_out, 0.14, true, 0.45);
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

        let knight = make_slice(&mut |v, i| build_parametric_knight(v, i, detail));
        let mage = make_slice(&mut |v, i| build_parametric_mage(v, i, detail));
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

        // 12: Marsh Ranger (Hooded emerald/faction scout with cloak, quiver, and recurve bow!)
        let hero_ranger = make_slice(&mut |v, i| {
            let cloak = [0.22, 0.68, 0.42];
            let leather = [0.46, 0.31, 0.20];
            let bow = [0.64, 0.44, 0.24];
            let skin = [0.92, 0.78, 0.64];
            // Boots & agile stance legs
            push_limb(v, i, Vec3::new(-0.24, 0.0, 0.06), Vec3::new(-0.20, 0.92, 0.02), 0.18, 0.14, 6, leather, 0.0);
            push_limb(v, i, Vec3::new(0.24, 0.0, -0.06), Vec3::new(0.20, 0.92, -0.02), 0.18, 0.14, 6, leather, 0.0);
            // Faction tunic + shoulder mantle (mat 12 = selective team tint)
            push_limb(v, i, Vec3::new(0.0, 0.85, 0.0), Vec3::new(0.0, 1.88, 0.0), 0.34, 0.42, 8, cloak, 12.0);
            // Flowing rear cloak
            push_limb(v, i, Vec3::new(0.0, 1.78, -0.22), Vec3::new(0.0, 0.45, -0.58), 0.36, 0.56, 6, cloak, 12.0);
            // Head + pointed scout hood
            push_limb(v, i, Vec3::new(0.0, 1.92, 0.04), Vec3::new(0.0, 2.38, 0.04), 0.24, 0.22, 7, skin, 0.0);
            push_limb(v, i, Vec3::new(0.0, 2.10, -0.06), Vec3::new(0.0, 2.66, -0.22), 0.34, 0.10, 7, cloak, 12.0);
            // Quiver on back
            push_limb(v, i, Vec3::new(0.22, 1.15, -0.34), Vec3::new(0.32, 2.08, -0.28), 0.11, 0.13, 6, leather, 4.0);
            // Recurve longbow held forward (+Z)
            push_limb(v, i, Vec3::new(-0.42, 1.48, 0.12), Vec3::new(-0.36, 1.45, 0.72), 0.10, 0.09, 5, leather, 0.0);
            push_limb(v, i, Vec3::new(-0.36, 1.45, 0.76), Vec3::new(-0.34, 2.28, 0.48), 0.06, 0.04, 5, bow, 4.0);
            push_limb(v, i, Vec3::new(-0.36, 1.45, 0.76), Vec3::new(-0.34, 0.62, 0.48), 0.06, 0.04, 5, bow, 4.0);
        });

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

        Self {
            vertices: verts,
            indices: idxs,
            slices: [
                knight,
                mage,
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
                hero_ranger,
                shrine_obelisk,
            ],
        }
    }
}
