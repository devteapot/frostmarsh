//! Native GPU 2D Action-MMO HUD + Live Tactical Minimap + Bitmap Font Renderer.
//!
//! Renders razor-sharp at native output resolution (1080p / 4K / Retina) right on top of the
//! Approach 4 + Tier 3 Hybrid 3D world:
//! - Top-Center Match Score Banner, 3 Shrine Capture Pips, Weather Badge & Match Clock
//! - Top-Right Live 256×256 A+ Tactical Minimap (terrain, ice, wildfires, shrines, Relic, bots, player)
//! - Bottom-Center Hero Plate (Class, Level, HP/Mana/Stamina/XP bars, Status Pills, 5-Slot Spell Hotbar)
//! - Top-Left Compact GB10 Neural Telemetry + Live Kill Feed & Elemental Combo Log
//! - World-Space Overhead HP Bars & Floating Damage/Combo Popups

use crate::camera::CameraMatrices;
use crate::mesh_gen::{
    archetype_color, archetype_is_solo, archetype_tag, CLASS_BURROWER, CLASS_EMBERSTALKER,
    CLASS_SKYWING, CLASS_STRIDER, CLASS_TIDEKIN,
};
use crate::net::LiveEntityState;
use crate::planet_world::{face_grid_to_world_pos, PlanetWorld};
use crate::world_bridge::LiveApproach4World;
use crate::z16_world::Z16World;
use bytemuck::{Pod, Zeroable};
use frostmarsh_proto::*;
use glam::{Vec3, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HudVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub mode: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct HudConfig<'a> {
    pub show_hud: bool,
    pub show_telemetry: bool,
    pub show_brain_inspector: bool,
    pub brain_selected_idx: Option<usize>,
    pub selected_element: u8,
    pub focus_radius: f32,
    pub style_label: &'static str,
    pub z16_active: bool,
    pub z16_style_label: &'static str,
    pub fps: f32,
    pub rtt_ms: f32,
    pub chat_open: bool,
    pub chat_input: &'a str,
    pub chat_shout: bool,
    pub show_scoreboard: bool,
    pub audio_muted: bool,
    pub audio_available: bool,
    pub cave_enclosure: f32,
    pub show_squad_links: bool,
    pub camera_preset_label: &'static str,
}

pub struct HudBatch {
    pub vertices: Vec<HudVertex>,
}

impl HudBatch {
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(16_384),
        }
    }

    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        if w <= 0.0 || h <= 0.0 || color[3] <= 0.001 {
            return;
        }
        let v0 = HudVertex { pos: [x, y], uv: [0.0, 0.0], color, mode: 0.0 };
        let v1 = HudVertex { pos: [x + w, y], uv: [1.0, 0.0], color, mode: 0.0 };
        let v2 = HudVertex { pos: [x + w, y + h], uv: [1.0, 1.0], color, mode: 0.0 };
        let v3 = HudVertex { pos: [x, y + h], uv: [0.0, 1.0], color, mode: 0.0 };
        self.vertices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
    }

    pub fn rect_outline(&mut self, x: f32, y: f32, w: f32, h: f32, t: f32, color: [f32; 4]) {
        self.rect(x, y, w, t, color);
        self.rect(x, y + h - t, w, t, color);
        self.rect(x, y + t, t, (h - 2.0 * t).max(0.0), color);
        self.rect(x + w - t, y + t, t, (h - 2.0 * t).max(0.0), color);
    }

    pub fn textured_rect(&mut self, x: f32, y: f32, w: f32, h: f32, tint: [f32; 4]) {
        let v0 = HudVertex { pos: [x, y], uv: [0.0, 0.0], color: tint, mode: 1.0 };
        let v1 = HudVertex { pos: [x + w, y], uv: [1.0, 0.0], color: tint, mode: 1.0 };
        let v2 = HudVertex { pos: [x + w, y + h], uv: [1.0, 1.0], color: tint, mode: 1.0 };
        let v3 = HudVertex { pos: [x, y + h], uv: [0.0, 1.0], color: tint, mode: 1.0 };
        self.vertices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
    }

    pub fn text(&mut self, x: f32, y: f32, scale: f32, color: [f32; 4], s: &str) {
        let mut cx = x;
        for ch in s.chars() {
            let g = glyph_5x7(ch);
            for (row, &bits) in g.iter().enumerate() {
                if bits == 0 {
                    continue;
                }
                for col in 0..5 {
                    if (bits & (1 << (4 - col))) != 0 {
                        self.rect(
                            cx + col as f32 * scale,
                            y + row as f32 * scale,
                            scale,
                            scale,
                            color,
                        );
                    }
                }
            }
            cx += 6.0 * scale;
        }
    }

    pub fn text_shadow(&mut self, x: f32, y: f32, scale: f32, color: [f32; 4], s: &str) {
        self.text(x + scale * 0.7, y + scale * 0.7, scale, [0.04, 0.03, 0.09, color[3] * 0.85], s);
        self.text(x, y, scale, color, s);
    }

    pub fn text_width(s: &str, scale: f32) -> f32 {
        (s.chars().count() as f32) * 6.0 * scale - scale
    }
}

/// Maps normalized 4×4 `Z_world` saliency (`0..255`) through Indigo → Cyan → Gold → Flame Orange.
fn saliency_color(v: u8) -> [f32; 4] {
    let t = (v as f32 / 255.0).clamp(0.0, 1.0);
    let lerp3 = |a: [f32; 3], b: [f32; 3], u: f32| -> [f32; 4] {
        [
            a[0] + (b[0] - a[0]) * u,
            a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u,
            0.96,
        ]
    };
    let c0 = [0.16, 0.13, 0.38]; // Cool Indigo
    let c1 = [0.14, 0.78, 0.94]; // Electric Cyan
    let c2 = [1.00, 0.84, 0.24]; // Radiant Gold
    let c3 = [1.00, 0.36, 0.12]; // Flame Orange
    if t < 0.35 {
        lerp3(c0, c1, t / 0.35)
    } else if t < 0.70 {
        lerp3(c1, c2, (t - 0.35) / 0.35)
    } else {
        lerp3(c2, c3, (t - 0.70) / 0.30)
    }
}

fn project_world(world_rel: Vec3, mats: &CameraMatrices, sw: f32, sh: f32) -> Option<(f32, f32)> {
    let clip: Vec4 = mats.vp * Vec4::new(world_rel.x, world_rel.y, world_rel.z, 1.0);
    if clip.w <= 0.1 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.x < -1.1 || ndc.x > 1.1 || ndc.y < -1.1 || ndc.y > 1.1 || ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }
    let sx = (ndc.x * 0.5 + 0.5) * sw;
    let sy = (0.5 - ndc.y * 0.5) * sh;
    Some((sx, sy))
}

fn snr_badge(snr_db: f32) -> (String, [f32; 4]) {
    if snr_db >= 10.0 {
        (format!("+{:.0}DB", snr_db), [0.22, 0.94, 0.56, 1.0])
    } else if snr_db >= 0.0 {
        (format!("+{:.0}DB", snr_db), [1.00, 0.84, 0.28, 1.0])
    } else {
        (format!("{:.0}DB MUFFLED", snr_db), [0.94, 0.36, 0.38, 1.0])
    }
}

pub fn build_hud(
    sw: f32,
    sh: f32,
    world: &LiveApproach4World,
    z16: Option<&Z16World>,
    entities: Option<&LiveEntityState>,
    mats: &CameraMatrices,
    cfg: &HudConfig<'_>,
) -> HudBatch {
    let mut b = HudBatch::new();
    if !cfg.show_hud {
        return b;
    }
    let grid_to_xz = |gx: f32, gy: f32| -> (f32, f32) {
        if let Some(z16w) = z16 {
            z16w.grid_to_world_xz(gx, gy)
        } else {
            world.grid_to_world_xz(gx, gy)
        }
    };
    let sample_g_y = |gx: f32, gy: f32| -> f32 {
        if let Some(z16w) = z16 {
            z16w.sample_surface_y(gx, gy)
        } else {
            world.sample_ground_y(gx, gy)
        }
    };
    let sample_e_y = |gx: f32, gy: f32, gz: f32| -> f32 {
        if z16.is_some() {
            gz
        } else {
            world.sample_3d_y(gx, gy, gz)
        }
    };
    let ui = (sh / 1080.0).min(sw / 1920.0).clamp(0.68, 2.0);
    let panel_bg = [0.08, 0.07, 0.16, 0.82];
    let panel_border = [0.36, 0.34, 0.58, 0.85];
    let text_hi = [0.98, 0.96, 0.88, 1.0];
    let text_dim = [0.70, 0.74, 0.86, 0.92];
    let west_col = [0.28, 0.72, 1.0, 1.0];
    let east_col = [1.0, 0.36, 0.28, 1.0];
    let gold_col = [1.0, 0.84, 0.28, 1.0];

    let planet_mode = entities.map(|e| e.telemetry.planet_mode).unwrap_or(false);
    let planet_r_core = entities
        .map(|e| if e.telemetry.planet_r_core > 1.0 { e.telemetry.planet_r_core } else { 48.0 })
        .unwrap_or(48.0);
    let planet_face_n = entities
        .map(|e| (e.telemetry.grid_w as usize).max(16))
        .unwrap_or(64);
    let planet_depth_d = entities
        .map(|e| (e.telemetry.grid_d as usize).max(16))
        .unwrap_or(32);

    // ------------------------------------------------------------------
    // 1. WORLD-SPACE OVERHEAD HP BARS + FLOATING COMBAT POPUPS
    //    40–64 Player LOD: when players.len() > 16, render full nameplates
    //    for the closest 14 players + local player + relic carrier, and crisp
    //    compact team-colored HP pips for distant players!
    // ------------------------------------------------------------------
    if let Some(ent) = entities {
        if planet_mode {
            // Spherical front-hemisphere world labels for planetary mode (zero back-side bleed!)
            let cam_dist = mats.rel_eye.length();
            let is_orbital_far = cam_dist > planet_r_core * 2.4;
            let z_surf_est = ((15.0 * (planet_depth_d as f32 / 32.0)).round()).max(8.0);

            // Show nearby player pips/nameplates when in Tactical/Surface views, or local hero in Orbit
            for p in &ent.players {
                if p.hp <= 0 {
                    continue;
                }
                let is_self = p.id == ent.your_player_id;
                if is_orbital_far && !is_self {
                    continue;
                }
                let wpos = face_grid_to_world_pos(
                    p.face,
                    p.x,
                    p.y,
                    p.z.max(z_surf_est) + 2.8,
                    planet_face_n,
                    planet_r_core,
                );
                // Front-hemisphere visibility check against camera
                if wpos.normalize_or(Vec3::Y).dot(mats.rel_eye.normalize_or(Vec3::Z)) < 0.05 {
                    continue;
                }
                let dist = wpos.distance(mats.rel_eye);
                if !is_self && dist > 95.0 {
                    continue;
                }
                if let Some((sx, sy)) = project_world(wpos, mats, sw, sh) {
                    let hp_frac = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
                    let bar_c = if is_self {
                        [0.32, 0.96, 0.52, 0.96]
                    } else if p.team == 0 {
                        west_col
                    } else {
                        east_col
                    };
                    if is_self || dist < 48.0 {
                        let bw = 54.0 * ui;
                        let bh = 5.5 * ui;
                        let bx = sx - bw * 0.5;
                        let by = sy - bh;
                        b.rect(bx - 1.2 * ui, by - 1.2 * ui, bw + 2.4 * ui, bh + 2.4 * ui, if is_self { gold_col } else { [0.05, 0.04, 0.12, 0.85] });
                        b.rect(bx, by, bw, bh, [0.16, 0.14, 0.24, 0.92]);
                        b.rect(bx, by, bw * hp_frac, bh, bar_c);
                        let short_name: String = p.name.chars().take(8).collect();
                        let tag = format!("{} {}", archetype_tag(p.class_id), short_name);
                        let sc = 1.25 * ui;
                        let tw = HudBatch::text_width(&tag, sc);
                        b.text_shadow(sx - tw * 0.5, by - 11.5 * ui, sc, if is_self { gold_col } else { text_hi }, &tag);
                    } else {
                        let pw = 16.0 * ui;
                        let ph = 3.5 * ui;
                        b.rect(sx - pw * 0.5, sy - ph, pw, ph, [0.12, 0.11, 0.20, 0.85]);
                        b.rect(sx - pw * 0.5, sy - ph, pw * hp_frac, ph, bar_c);
                    }
                }
            }

            // Front-hemisphere Planetary Shrine Beacon labels
            for s in &ent.shrines {
                let wpos = face_grid_to_world_pos(
                    s.face,
                    s.x,
                    s.y,
                    z_surf_est + 8.5,
                    planet_face_n,
                    planet_r_core,
                );
                if wpos.normalize_or(Vec3::Y).dot(mats.rel_eye.normalize_or(Vec3::Z)) < 0.08 {
                    continue;
                }
                if let Some((sx, sy)) = project_world(wpos, mats, sw, sh) {
                    let bw = 64.0 * ui;
                    let bh = 4.5 * ui;
                    let bx = sx - bw * 0.5;
                    b.rect(bx - 1.0, sy - 1.0, bw + 2.0, bh + 2.0, [0.05, 0.04, 0.12, 0.82]);
                    let p01 = (s.progress * 0.5 + 0.5).clamp(0.0, 1.0);
                    b.rect(bx, sy, bw * (1.0 - p01), bh, west_col);
                    b.rect(bx + bw * (1.0 - p01), sy, bw * p01, bh, east_col);
                    let sc = 1.2 * ui;
                    let tw = HudBatch::text_width(&s.name, sc);
                    b.text_shadow(sx - tw * 0.5, sy - 10.5 * ui, sc, gold_col, &s.name);
                }
            }
        } else {
        let mut proj_players: Vec<(usize, f32, f32, f32, bool)> =
            Vec::with_capacity(ent.players.len());
        for (idx, p) in ent.players.iter().enumerate() {
            let (wx, wz) = grid_to_xz(p.x, p.y);
            let gy = sample_e_y(p.x, p.y, p.z);
            let rel = Vec3::new(wx, gy + 3.65, wz) - mats.origin;
            if let Some((sx, sy)) = project_world(rel, mats, sw, sh) {
                let d2 = rel.distance_squared(mats.rel_eye);
                let is_priority = p.id == ent.your_player_id || p.carrying_relic;
                proj_players.push((idx, sx, sy, d2, is_priority));
            }
        }

        let use_lod = ent.players.len() > 16;
        let mut full_mask = vec![!use_lod; ent.players.len()];
        if use_lod {
            let mut non_prio: Vec<(usize, f32)> = proj_players
                .iter()
                .filter_map(|&(idx, _, _, d2, is_prio)| {
                    if is_prio {
                        full_mask[idx] = true;
                        None
                    } else {
                        Some((idx, d2))
                    }
                })
                .collect();
            non_prio.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            for &(idx, _) in non_prio.iter().take(14) {
                full_mask[idx] = true;
            }
        }

        // Pass 1A: Crisp compact team-colored HP pips for distant players (zero text clutter!)
        if use_lod {
            for &(idx, sx, sy, _, _) in &proj_players {
                if full_mask[idx] {
                    continue;
                }
                let p = &ent.players[idx];
                let hp_frac = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
                let bar_c = if p.team == 0 { west_col } else { east_col };
                let arch_c = archetype_color(p.class_id);
                let pw = 18.0 * ui;
                let ph = 4.0 * ui;
                let px = sx - pw * 0.5;
                let py = sy - ph;
                b.rect(px - 1.0 * ui, py - 1.0 * ui, pw + 2.0 * ui, ph + 2.0 * ui, [0.05, 0.04, 0.10, 0.80]);
                b.rect(px, py, pw, ph, [0.16, 0.14, 0.24, 0.85]);
                b.rect(px, py, pw * hp_frac, ph, bar_c);
                // Crisp 2.5px archetype/solo accent pip on left edge
                b.rect(px - 3.5 * ui, py, 2.5 * ui, ph, arch_c);
            }
        }

        // Pass 1B: Full nameplates (far-to-near so closest heroes & local player stay crisp on top!)
        let mut full_list: Vec<(usize, f32, f32, f32)> = proj_players
            .into_iter()
            .filter(|&(idx, _, _, _, _)| full_mask[idx])
            .map(|(idx, sx, sy, d2, _)| (idx, sx, sy, d2))
            .collect();
        full_list.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        for (idx, sx, sy, _) in full_list {
            let p = &ent.players[idx];
            let bw = 58.0 * ui;
            let bh = 6.0 * ui;
            let bx = sx - bw * 0.5;
            let by = sy - bh;
            let hp_frac = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
            let is_self = p.id == ent.your_player_id;
            let is_solo = archetype_is_solo(p.class_id);
            let arch = archetype_tag(p.class_id);
            let arch_c = archetype_color(p.class_id);
            let bar_c = if is_self {
                [0.32, 0.96, 0.52, 0.96]
            } else if p.team == 0 {
                west_col
            } else {
                east_col
            };
            let border_c = if is_self || p.carrying_relic {
                gold_col
            } else if is_solo {
                [1.0, 0.42, 0.18, 0.92]
            } else {
                [0.05, 0.04, 0.10, 0.85]
            };
            b.rect(bx - 1.5 * ui, by - 1.5 * ui, bw + 3.0 * ui, bh + 3.0 * ui, border_c);
            b.rect(bx, by, bw, bh, [0.16, 0.14, 0.24, 0.92]);
            b.rect(bx, by, bw * hp_frac, bh, bar_c);
            // Archetype indicator swatch on left edge of HP bar
            b.rect(bx - 5.5 * ui, by, 3.2 * ui, bh, arch_c);

            let short_name: String = p.name.chars().take(8).collect();
            let tag = format!("L{} {} {}", p.level, arch, short_name);
            let sc = 1.32 * ui;
            let tw = HudBatch::text_width(&tag, sc);
            let name_col = if is_self {
                gold_col
            } else if is_solo {
                [1.0, 0.80, 0.66, 1.0]
            } else {
                text_hi
            };
            b.text_shadow(sx - tw * 0.5, by - 12.5 * ui, sc, name_col, &tag);
        }

        // Shrine world labels
        for s in &ent.shrines {
            let (wx, wz) = grid_to_xz(s.x, s.y);
            let gy = if let Some(z16w) = z16 {
                if z16w.depth >= 48 {
                    if s.name.contains("ABYSSAL") {
                        z16w.sample_surface_y_near(s.x, s.y, 10.0)
                    } else if s.name.contains("VIADUCT") {
                        z16w.sample_surface_y_near(s.x, s.y, 35.0)
                    } else {
                        z16w.sample_surface_y(s.x, s.y)
                    }
                } else {
                    z16w.sample_surface_y(s.x, s.y)
                }
            } else {
                sample_g_y(s.x, s.y)
            };
            let rel = Vec3::new(wx, gy + 5.8, wz) - mats.origin;
            if let Some((sx, sy)) = project_world(rel, mats, sw, sh) {
                let bw = 72.0 * ui;
                let bh = 5.0 * ui;
                let bx = sx - bw * 0.5;
                b.rect(bx - 1.0, sy - 1.0, bw + 2.0, bh + 2.0, [0.05, 0.04, 0.12, 0.85]);
                let p01 = (s.progress * 0.5 + 0.5).clamp(0.0, 1.0);
                b.rect(bx, sy, bw * (1.0 - p01), bh, west_col);
                b.rect(bx + bw * (1.0 - p01), sy, bw * p01, bh, east_col);
                let sc = 1.3 * ui;
                let tw = HudBatch::text_width(&s.name, sc);
                b.text_shadow(sx - tw * 0.5, sy - 11.0 * ui, sc, gold_col, &s.name);
            }
        }

        // Active Spatial Time-Dilation Dome world-space badges
        for d in &ent.time_domes {
            let (wx, wz) = grid_to_xz(d.x, d.y);
            let gy = sample_g_y(d.x, d.y);
            let rel = Vec3::new(wx, gy + d.radius * 0.62 + 1.8, wz) - mats.origin;
            if let Some((sx, sy)) = project_world(rel, mats, sw, sh) {
                let is_stasis = d.dt_scale < 0.8;
                let lbl = if is_stasis {
                    format!("STASIS 0.15X {:.0}S", d.ttl_s.max(0.0))
                } else {
                    format!("HASTE 3.0X {:.0}S", d.ttl_s.max(0.0))
                };
                let col = if is_stasis {
                    [0.36, 0.94, 1.0, 1.0]
                } else {
                    [1.0, 0.82, 0.28, 1.0]
                };
                let sc = 1.35 * ui;
                let tw = HudBatch::text_width(&lbl, sc);
                let pw = tw + 12.0 * ui;
                let ph = 18.0 * ui;
                b.rect(sx - pw * 0.5, sy - ph * 0.5, pw, ph, [0.05, 0.05, 0.14, 0.82]);
                b.rect_outline(sx - pw * 0.5, sy - ph * 0.5, pw, ph, 1.2 * ui, col);
                b.text_shadow(sx - tw * 0.5, sy - 4.5 * ui, sc, col, &lbl);
            }
        }

        // Floating damage & elemental combo callouts
        for ev in ent.events.iter().rev().take(12) {
            if ev.kind == 0 || ev.kind == 2 {
                let (wx, wz) = grid_to_xz(ev.x, ev.y);
                let gy = sample_g_y(ev.x, ev.y);
                let rel = Vec3::new(wx, gy + 4.9, wz) - mats.origin;
                if let Some((sx, sy)) = project_world(rel, mats, sw, sh) {
                    let sc = if ev.kind == 2 { 2.0 * ui } else { 1.6 * ui };
                    let col = if ev.kind == 2 {
                        gold_col
                    } else {
                        [1.0, 0.52, 0.36, 1.0]
                    };
                    let tw = HudBatch::text_width(&ev.label, sc);
                    b.text_shadow(sx - tw * 0.5, sy - 14.0 * ui, sc, col, &ev.label);
                }
            }
        }

        // 3D Projected In-World Acoustic Speech Bubbles (with team border, SNR pill, & HUD avoidance!)
        if !ent.speech_bubbles.is_empty() {
            let mut placed: Vec<(f32, f32, f32, f32)> = Vec::new();
            let safe_min_x = 16.0 * ui;
            let safe_max_x = sw - 16.0 * ui;
            let safe_min_y = 78.0 * ui;
            let safe_max_y = sh - 144.0 * ui;
            let hdr_sc = 1.15 * ui;
            let msg_sc = 1.25 * ui;

            for sb in ent.speech_bubbles.iter().rev().take(5) {
                let (wx, wz) = grid_to_xz(sb.x, sb.y);
                let gy = sample_e_y(sb.x, sb.y, sb.z);
                let rel = Vec3::new(wx, gy + 5.2, wz) - mats.origin;
                let Some((sx, sy)) = project_world(rel, mats, sw, sh) else {
                    continue;
                };

                let is_self_bubble = sb.speaker_id == ent.your_player_id || sb.speaker_name == "YOU";
                let border_col = if is_self_bubble {
                    gold_col
                } else if sb.team == 0 {
                    west_col
                } else {
                    east_col
                };
                let (snr_lbl, snr_col) = snr_badge(sb.snr_db);
                let spk_short: String = sb.speaker_name.chars().take(16).collect();
                let msg_short: String = sb.text.chars().take(52).collect();

                let spk_w = HudBatch::text_width(&spk_short, hdr_sc);
                let snr_w = HudBatch::text_width(&snr_lbl, 1.05 * ui) + 8.0 * ui;
                let hdr_w = spk_w + 10.0 * ui + snr_w;
                let msg_w = HudBatch::text_width(&msg_short, msg_sc);
                let card_w = (hdr_w.max(msg_w) + 18.0 * ui).clamp(150.0 * ui, 430.0 * ui);
                let card_h = 38.0 * ui;

                let mut cx = (sx - card_w * 0.5).clamp(safe_min_x, (safe_max_x - card_w).max(safe_min_x));
                let mut cy = (sy - card_h - 10.0 * ui).clamp(safe_min_y, (safe_max_y - card_h).max(safe_min_y));

                // Avoid top-left telemetry & top-right minimap / brain inspector
                if cfg.show_telemetry && cx < 444.0 * ui && cy < 224.0 * ui {
                    cx = 446.0 * ui;
                }
                let right_guard_h = if cfg.show_brain_inspector { 442.0 * ui } else { 226.0 * ui };
                if cx + card_w > sw - 256.0 * ui && cy < right_guard_h {
                    cx = (sw - 258.0 * ui - card_w).max(safe_min_x);
                }
                // Avoid bottom-left acoustic transcript log
                if cx < 458.0 * ui && cy + card_h > sh - 266.0 * ui {
                    cy = (sh - 270.0 * ui - card_h).max(safe_min_y);
                }

                // De-overlap against previously placed bubbles
                for _ in 0..4 {
                    let mut bumped = false;
                    for &(px, py, pw, ph) in &placed {
                        let overlap_x = cx < px + pw + 6.0 * ui && cx + card_w + 6.0 * ui > px;
                        let overlap_y = cy < py + ph + 5.0 * ui && cy + card_h + 5.0 * ui > py;
                        if overlap_x && overlap_y {
                            if cy - (card_h + 6.0 * ui) >= safe_min_y {
                                cy = py - card_h - 6.0 * ui;
                            } else {
                                cy = (py + ph + 6.0 * ui).min(safe_max_y - card_h);
                            }
                            bumped = true;
                        }
                    }
                    if !bumped {
                        break;
                    }
                }
                placed.push((cx, cy, card_w, card_h));

                // Crisp dark rounded-style card (beveled 2px inset corners + team border)
                let bg = [0.06, 0.05, 0.14, 0.90];
                let bev = 2.5 * ui;
                b.rect(cx + bev, cy, card_w - 2.0 * bev, card_h, bg);
                b.rect(cx, cy + bev, card_w, card_h - 2.0 * bev, bg);
                b.rect_outline(cx + bev * 0.5, cy + bev * 0.5, card_w - bev, card_h - bev, 1.8 * ui, border_col);
                // Top accent bar
                b.rect(cx + bev, cy + 1.5 * ui, card_w - 2.0 * bev, 2.2 * ui, border_col);
                // Subtle downward pointer notch if bubble is near speaker
                if (sx - (cx + card_w * 0.5)).abs() < card_w * 0.65 && sy > cy + card_h {
                    let tx = sx.clamp(cx + 12.0 * ui, cx + card_w - 12.0 * ui);
                    b.rect(tx - 4.0 * ui, cy + card_h - 1.0 * ui, 8.0 * ui, 4.0 * ui, border_col);
                    b.rect(tx - 2.0 * ui, cy + card_h + 2.5 * ui, 4.0 * ui, 3.0 * ui, border_col);
                }

                // Header: speaker name + SNR pill
                b.text_shadow(cx + 9.0 * ui, cy + 7.0 * ui, hdr_sc, border_col, &spk_short);
                let pill_x = cx + card_w - snr_w - 7.0 * ui;
                let pill_y = cy + 5.5 * ui;
                let pill_h = 12.5 * ui;
                b.rect(
                    pill_x,
                    pill_y,
                    snr_w,
                    pill_h,
                    [snr_col[0] * 0.22, snr_col[1] * 0.22, snr_col[2] * 0.22, 0.94],
                );
                b.rect_outline(pill_x, pill_y, snr_w, pill_h, 1.0 * ui, snr_col);
                b.text(pill_x + 4.0 * ui, pill_y + 2.8 * ui, 1.05 * ui, snr_col, &snr_lbl);

                // Message line
                b.text_shadow(cx + 9.0 * ui, cy + 22.0 * ui, msg_sc, text_hi, &msg_short);
            }
        }
        }
    }

    // ------------------------------------------------------------------
    // 2. TOP-CENTER SCORE, SHRINE OWNERSHIP & WEATHER BANNER
    // ------------------------------------------------------------------
    let banner_w = 460.0 * ui;
    let banner_h = 56.0 * ui;
    let bx = (sw - banner_w) * 0.5;
    let by = 12.0 * ui;
    b.rect(bx, by, banner_w, banner_h, panel_bg);
    b.rect_outline(bx, by, banner_w, banner_h, 2.0 * ui, panel_border);

    if let Some(ent) = entities {
        let tel = &ent.telemetry;
        let w_str = format!("WEST {:03}", tel.score_west);
        let e_str = format!("{:03} EAST", tel.score_east);
        let sc = 2.2 * ui;
        b.text_shadow(bx + 16.0 * ui, by + 10.0 * ui, sc, west_col, &w_str);
        let ew = HudBatch::text_width(&e_str, sc);
        b.text_shadow(bx + banner_w - ew - 16.0 * ui, by + 10.0 * ui, sc, east_col, &e_str);

        // Match countdown timer + weather badge in center
        let w_label = match tel.weather_kind {
            WEATHER_BLIZZARD => "BLIZZARD",
            WEATHER_SQUALL => "SQUALL",
            WEATHER_EMBER => "EMBER DUSK",
            _ => "CLEAR FROST",
        };
        let (mid_str, mid_col) = if ent.match_phase != MATCH_PHASE_ACTIVE {
            (format!("NEXT {:02}S  {}", ent.match_time_left_s, w_label), gold_col)
        } else {
            let secs_disp = if ent.match_time_left_s > 0 || ent.sim_tick > 60 {
                ent.match_time_left_s as u32
            } else {
                ent.match_time_secs
            };
            let mins = secs_disp / 60;
            let secs = secs_disp % 60;
            let col = if secs_disp <= 30 && ent.match_time_left_s > 0 {
                [1.0, 0.42, 0.24, 1.0]
            } else {
                gold_col
            };
            (format!("{:02}:{:02} {}", mins, secs, w_label), col)
        };
        let msc = 1.45 * ui;
        let mw = HudBatch::text_width(&mid_str, msc);
        b.text_shadow(bx + (banner_w - mw) * 0.5, by + 9.0 * ui, msc, mid_col, &mid_str);

        // Shrine ownership pips (3 on flat arena, up to 6 on full spherical planet!)
        let pip_count = if planet_mode {
            ent.shrines.len().clamp(3, 6)
        } else {
            ent.shrines.len().clamp(1, 3)
        };
        let pip_sz = if pip_count > 3 { 10.5 * ui } else { 12.0 * ui };
        let pip_gap = if pip_count > 3 { 11.0 * ui } else { 18.0 * ui };
        let total_pips_w = pip_count as f32 * pip_sz + (pip_count.saturating_sub(1) as f32) * pip_gap;
        let px0 = bx + (banner_w - total_pips_w) * 0.5;
        let py0 = by + 32.0 * ui;
        for (i, s) in ent.shrines.iter().take(pip_count).enumerate() {
            let px = px0 + i as f32 * (pip_sz + pip_gap);
            let col = if s.owner < 0 {
                west_col
            } else if s.owner > 0 {
                east_col
            } else {
                [0.58, 0.58, 0.70, 0.9]
            };
            b.rect(px, py0, pip_sz, pip_sz, col);
            b.rect_outline(px, py0, pip_sz, pip_sz, 1.5 * ui, text_hi);
        }

        // Diurnal World Time + Time-Warp Pill directly below top-center banner
        let total_mins = ((tel.day_time.rem_euclid(1.0) * 1440.0).floor() as u32) % 1440;
        let hh = total_mins / 60;
        let mm = total_mins % 60;
        let day_num = tel.world_day.max(1);
        let ts = if tel.time_scale > 0.01 { tel.time_scale } else { 1.0 };
        let ts_str = if ts < 0.5 {
            "0.25X".to_string()
        } else {
            format!("{:.0}X", ts)
        };
        let clock_str = format!("DAY {} {:02}:{:02} [K]{}", day_num, hh, mm, ts_str);
        let solar = (tel.day_time.rem_euclid(1.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2).sin();
        let clock_col = if solar < -0.2 {
            [0.58, 0.74, 1.00, 1.0] // Moonlit night indigo
        } else if solar > 0.3 {
            [1.00, 0.96, 0.78, 1.0] // Bright midday sun
        } else {
            [1.00, 0.70, 0.38, 1.0] // Golden dawn / crimson dusk
        };
        let csc = 1.25 * ui;
        let cw = HudBatch::text_width(&clock_str, csc) + 18.0 * ui;
        let ch = 20.0 * ui;
        let cx = bx + (banner_w - cw) * 0.5;
        let cy = by + banner_h + 4.0 * ui;
        b.rect(cx, cy, cw, ch, [0.07, 0.06, 0.14, 0.84]);
        b.rect_outline(cx, cy, cw, ch, 1.2 * ui, clock_col);
        b.text_shadow(cx + 9.0 * ui, cy + 5.0 * ui, csc, clock_col, &clock_str);
    } else {
        b.text_shadow(bx + 90.0 * ui, by + 20.0 * ui, 2.0 * ui, text_hi, "CONNECTING TO SPARK GB10...");
    }

    // ------------------------------------------------------------------
    // 3. TOP-RIGHT LIVE 256x256 TACTICAL MINIMAP
    // ------------------------------------------------------------------
    let mm_sz = 188.0 * ui;
    let mm_x = sw - mm_sz - 16.0 * ui;
    let mm_y = 14.0 * ui;
    b.rect(mm_x - 4.0 * ui, mm_y - 4.0 * ui, mm_sz + 8.0 * ui, mm_sz + 24.0 * ui, panel_bg);
    b.rect_outline(mm_x - 4.0 * ui, mm_y - 4.0 * ui, mm_sz + 8.0 * ui, mm_sz + 24.0 * ui, 2.0 * ui, panel_border);
    b.textured_rect(mm_x, mm_y, mm_sz, mm_sz, [1.0, 1.0, 1.0, 0.96]);

    let (gw_u, gh_u, gd_u) = if let Some(z16w) = z16 {
        (z16w.width, z16w.height, z16w.depth)
    } else {
        (world.grid_w, world.grid_h, world.grid_d)
    };
    let gw = gw_u.max(1) as f32;
    let gh = gh_u.max(1) as f32;
    if let Some(ent) = entities {
        // Shrines on minimap
        for s in &ent.shrines {
            let (sx, sy) = if planet_mode {
                let wp = face_grid_to_world_pos(s.face, s.x, s.y, 16.0, planet_face_n, planet_r_core);
                let (u, v) = PlanetWorld::sphere_pos_to_minimap_uv(wp);
                (mm_x + u * mm_sz, mm_y + v * mm_sz)
            } else {
                (mm_x + (s.x / gw) * mm_sz, mm_y + (s.y / gh) * mm_sz)
            };
            let col = if s.owner < 0 { west_col } else if s.owner > 0 { east_col } else { gold_col };
            b.rect(sx - 4.0 * ui, sy - 4.0 * ui, 8.0 * ui, 8.0 * ui, col);
            b.rect_outline(sx - 4.0 * ui, sy - 4.0 * ui, 8.0 * ui, 8.0 * ui, 1.2 * ui, text_hi);
        }
        // Active Time-Dilation Domes on minimap
        if !planet_mode {
            for d in &ent.time_domes {
                let dx = mm_x + (d.x / gw) * mm_sz;
                let dy = mm_y + (d.y / gh) * mm_sz;
                let dr = ((d.radius / gw) * mm_sz).clamp(4.0 * ui, 28.0 * ui);
                let dcol = if d.dt_scale < 0.8 {
                    [0.32, 0.94, 1.0, 0.92]
                } else {
                    [1.0, 0.82, 0.26, 0.92]
                };
                b.rect(dx - dr, dy - dr, dr * 2.0, dr * 2.0, [dcol[0], dcol[1], dcol[2], 0.20]);
                b.rect_outline(dx - dr, dy - dr, dr * 2.0, dr * 2.0, 1.4 * ui, dcol);
            }
            // Relic star
            let rx = mm_x + (ent.relic_x / gw) * mm_sz;
            let ry = mm_y + (ent.relic_y / gh) * mm_sz;
            b.rect(rx - 3.5 * ui, ry - 3.5 * ui, 7.0 * ui, 7.0 * ui, gold_col);
        }

        // Heroes / Bots on minimap (scales cleanly up to 64 players; self & relic carrier drawn on top!)
        let crowded = ent.players.len() > 24;
        let base_dot = if crowded { 3.1 * ui } else { 4.2 * ui };
        for pass in 0..2 {
            for p in &ent.players {
                let is_self = p.id == ent.your_player_id;
                let is_top = is_self || p.carrying_relic;
                if (pass == 0 && is_top) || (pass == 1 && !is_top) {
                    continue;
                }
                let sz = if is_self {
                    6.4 * ui
                } else if p.carrying_relic {
                    5.4 * ui
                } else {
                    base_dot
                };
                let half = sz * 0.5;
                let (ux, uy) = if planet_mode {
                    let wp = face_grid_to_world_pos(p.face, p.x, p.y, p.z, planet_face_n, planet_r_core);
                    PlanetWorld::sphere_pos_to_minimap_uv(wp)
                } else {
                    ((p.x / gw).clamp(0.0, 1.0), (p.y / gh).clamp(0.0, 1.0))
                };
                let px = (mm_x + ux * mm_sz).clamp(mm_x + half, mm_x + mm_sz - half);
                let py = (mm_y + uy * mm_sz).clamp(mm_y + half, mm_y + mm_sz - half);
                let col = if is_self {
                    [0.35, 1.0, 0.55, 1.0]
                } else if p.team == 0 {
                    west_col
                } else {
                    east_col
                };
                if archetype_is_solo(p.class_id) && !is_self {
                    b.rect(px - half - 0.8 * ui, py - half - 0.8 * ui, sz + 1.6 * ui, sz + 1.6 * ui, [1.0, 0.42, 0.14, 0.92]);
                }
                b.rect(px - half, py - half, sz, sz, col);
                if is_self {
                    b.rect_outline(px - half, py - half, sz, sz, 1.0 * ui, text_hi);
                } else if p.carrying_relic {
                    b.rect_outline(px - half - 1.0 * ui, py - half - 1.0 * ui, sz + 2.0 * ui, sz + 2.0 * ui, 1.1 * ui, gold_col);
                }
            }
        }
    }
    let player_cnt = entities.map(|e| e.players.len()).unwrap_or(0);
    let mm_lbl = if planet_mode {
        format!("PLANET 6×{}²×{} ({}P)", planet_face_n, planet_depth_d, player_cnt)
    } else if cfg.z16_active {
        format!("3D-Z{} {}X{} ({}P)", gd_u, gw_u, gh_u, player_cnt)
    } else {
        format!("REACH {}X{}X{} ({}P)", gw_u, gh_u, gd_u, player_cnt)
    };
    b.text_shadow(mm_x + 4.0 * ui, mm_y + mm_sz + 5.0 * ui, 1.20 * ui, if planet_mode || cfg.z16_active { [0.38, 0.94, 1.0, 1.0] } else { text_dim }, &mm_lbl);

    // ------------------------------------------------------------------
    // 3B. LIVE NEURAL-BRAIN INSPECTOR CARD (TOP-RIGHT BELOW MINIMAP)
    // ------------------------------------------------------------------
    let mut right_cursor_y = mm_y + mm_sz + 28.0 * ui;
    if cfg.show_brain_inspector {
        if let Some(ent) = entities {
            if !ent.bot_brains.is_empty() {
                let ref_pos = ent
                    .players
                    .iter()
                    .find(|p| p.id == ent.your_player_id)
                    .map(|p| (p.x, p.y))
                    .unwrap_or((gw * 0.5, gh * 0.5));

                let (sel_idx, brain) = if let Some(idx) = cfg.brain_selected_idx {
                    let i = idx % ent.bot_brains.len();
                    (i, &ent.bot_brains[i])
                } else {
                    // Auto-select the most active nearby neural bot
                    let mut best_i = 0usize;
                    let mut best_score = f32::NEG_INFINITY;
                    for (i, br) in ent.bot_brains.iter().enumerate() {
                        let ab_sum: f32 = br.ability_probs.iter().map(|&v| v as f32 / 255.0).sum();
                        let sal_peak = br.z_heatmap_4x4.iter().copied().max().unwrap_or(0) as f32 / 255.0;
                        let dist_pen = ent
                            .players
                            .iter()
                            .find(|p| p.id == br.bot_id)
                            .map(|p| ((p.x - ref_pos.0).hypot(p.y - ref_pos.1) / gw.max(1.0)) * 2.2)
                            .unwrap_or(1.0);
                        let score = ab_sum + sal_peak * 0.65 - dist_pen;
                        if score > best_score {
                            best_score = score;
                            best_i = i;
                        }
                    }
                    (best_i, &ent.bot_brains[best_i])
                };

                let bot_snap = ent.players.iter().find(|p| p.id == brain.bot_id);
                let team_accent = if brain.team == 0 { west_col } else { east_col };
                let layer_tag = bot_snap
                    .map(|p| {
                        if cfg.z16_active {
                            format!("Z{:.0}", p.z)
                        } else if p.z >= 1.55 {
                            "L2".to_string()
                        } else if p.z < 0.55 {
                            "L0".to_string()
                        } else {
                            "L1".to_string()
                        }
                    })
                    .unwrap_or_else(|| "L1".to_string());
                let raw_name = bot_snap
                    .map(|p| p.name.as_str())
                    .unwrap_or_else(|| archetype_tag(brain.class_id));
                let short_name: String = raw_name.chars().take(12).collect();
                let title = format!("BRAIN: {} {}", layer_tag, short_name);

                let bi_w = 236.0 * ui;
                let bi_h = 204.0 * ui;
                let bi_x = sw - bi_w - 12.0 * ui;
                let bi_y = right_cursor_y;

                b.rect(bi_x, bi_y, bi_w, bi_h, panel_bg);
                b.rect(bi_x, bi_y, bi_w, 3.0 * ui, team_accent);
                b.rect_outline(bi_x, bi_y, bi_w, bi_h, 1.5 * ui, panel_border);

                // Header + cycle indicator
                b.text_shadow(bi_x + 8.0 * ui, bi_y + 8.0 * ui, 1.30 * ui, team_accent, &title);
                let idx_tag = if cfg.brain_selected_idx.is_some() {
                    format!("{}/{} [B]", sel_idx + 1, ent.bot_brains.len())
                } else {
                    "AUTO [B]".to_string()
                };
                let idx_w = HudBatch::text_width(&idx_tag, 1.05 * ui);
                b.text(bi_x + bi_w - idx_w - 8.0 * ui, bi_y + 23.0 * ui, 1.05 * ui, text_dim, &idx_tag);
                b.text(bi_x + 8.0 * ui, bi_y + 23.0 * ui, 1.05 * ui, text_dim, "Z_WORLD 4X4");

                // Left: 4×4 Z_world Spatial Saliency Heatmap
                let hm_x = bi_x + 10.0 * ui;
                let hm_y = bi_y + 37.0 * ui;
                let cell_sz = 20.5 * ui;
                let cell_gap = 2.0 * ui;
                let hm_tot = 4.0 * cell_sz + 3.0 * cell_gap;
                b.rect(hm_x - 2.0 * ui, hm_y - 2.0 * ui, hm_tot + 4.0 * ui, hm_tot + 4.0 * ui, [0.05, 0.04, 0.11, 0.92]);
                for r in 0..4 {
                    for c in 0..4 {
                        let v = brain.z_heatmap_4x4[r * 4 + c];
                        let cx = hm_x + c as f32 * (cell_sz + cell_gap);
                        let cy = hm_y + r as f32 * (cell_sz + cell_gap);
                        b.rect(cx, cy, cell_sz, cell_sz, saliency_color(v));
                    }
                }
                // Subtle center ego crosshair on the 4×4 grid
                let mid_x = hm_x + hm_tot * 0.5;
                let mid_y = hm_y + hm_tot * 0.5;
                b.rect(mid_x - 4.0 * ui, mid_y - 0.75 * ui, 8.0 * ui, 1.5 * ui, [1.0, 1.0, 1.0, 0.75]);
                b.rect(mid_x - 0.75 * ui, mid_y - 4.0 * ui, 1.5 * ui, 8.0 * ui, [1.0, 1.0, 1.0, 0.75]);

                // Right: Element / Physical Actuator Intent Bars
                let ex = bi_x + 108.0 * ui;
                let mut ey = bi_y + 38.0 * ui;
                let bar_w = 84.0 * ui;
                let bar_h = 9.0 * ui;
                let dom_elem = if brain.elem_probs[0] >= brain.elem_probs[1]
                    && brain.elem_probs[0] >= brain.elem_probs[2]
                {
                    0
                } else if brain.elem_probs[1] >= brain.elem_probs[2] {
                    1
                } else {
                    2
                };
                let elems: [(&str, u8, [f32; 4]); 3] = if ent.telemetry.physics_native {
                    [
                        ("+DE", brain.elem_probs[0], [1.00, 0.44, 0.16, 1.0]),
                        ("-DE", brain.elem_probs[1], [0.34, 0.88, 1.00, 1.0]),
                        ("MAT", brain.elem_probs[2], [0.78, 0.36, 0.96, 1.0]),
                    ]
                } else {
                    [
                        ("FIR", brain.elem_probs[0], [1.00, 0.44, 0.16, 1.0]),
                        ("FRS", brain.elem_probs[1], [0.34, 0.88, 1.00, 1.0]),
                        ("OIL", brain.elem_probs[2], [0.78, 0.36, 0.96, 1.0]),
                    ]
                };
                for (ei, (lbl, prob, col)) in elems.iter().enumerate() {
                    let frac = (*prob as f32 / 255.0).clamp(0.0, 1.0);
                    b.text(ex, ey + 1.0 * ui, 1.05 * ui, if ei == dom_elem { *col } else { text_dim }, lbl);
                    let bx0 = ex + 32.0 * ui;
                    b.rect(bx0, ey, bar_w, bar_h, [0.13, 0.12, 0.22, 0.95]);
                    b.rect(bx0, ey, bar_w * frac, bar_h, *col);
                    if ei == dom_elem {
                        b.rect_outline(bx0, ey, bar_w, bar_h, 1.0 * ui, *col);
                    }
                    ey += 16.0 * ui;
                }

                // Target / class readout below element bars
                let tgt_str = if brain.target_id != 0 {
                    ent.players
                        .iter()
                        .find(|p| p.id == brain.target_id)
                        .map(|p| format!("TGT:{}", p.name.chars().take(10).collect::<String>()))
                        .unwrap_or_else(|| format!("TGT:#{}", brain.target_id))
                } else {
                    "TGT:ROAM".to_string()
                };
                b.text(ex, ey + 4.0 * ui, 1.00 * ui, gold_col, &tgt_str);
                let arch_tag = archetype_tag(brain.class_id);
                let role_tag = if archetype_is_solo(brain.class_id) { "SOLO" } else { "SQD" };
                let cls_tag = format!("{} [{}]", arch_tag, role_tag);
                b.text(ex, ey + 18.0 * ui, 1.00 * ui, archetype_color(brain.class_id), &cls_tag);

                // Bottom: 5 Ability / Actuator Intent Bars
                let ab_y = bi_y + 136.0 * ui;
                let ab_hdr = if ent.telemetry.physics_native {
                    "ACTUATOR INTENT"
                } else {
                    "ABILITY INTENT"
                };
                b.text(bi_x + 10.0 * ui, ab_y, 1.00 * ui, text_dim, ab_hdr);
                let abilities: [(&str, u8, [f32; 4]); 5] = if ent.telemetry.physics_native {
                    [
                        ("FLX", brain.ability_probs[2], [1.00, 0.52, 0.18, 1.0]),
                        ("IMP", brain.ability_probs[1], [0.96, 0.86, 0.52, 1.0]),
                        ("SMK", brain.ability_probs[0], [0.68, 0.78, 0.92, 1.0]),
                        ("OIL", brain.ability_probs[4], [0.78, 0.36, 0.96, 1.0]),
                        ("RAD", brain.ability_probs[5], [0.32, 0.96, 0.76, 1.0]),
                    ]
                } else {
                    [
                        ("BLT", brain.ability_probs[2], [1.00, 0.58, 0.20, 1.0]),
                        ("MLE", brain.ability_probs[1], [0.96, 0.86, 0.52, 1.0]),
                        ("DSH", brain.ability_probs[0], [0.36, 0.92, 1.00, 1.0]),
                        ("E", brain.ability_probs[4], [0.40, 0.96, 0.58, 1.0]),
                        ("R", brain.ability_probs[5], [1.00, 0.36, 0.44, 1.0]),
                    ]
                };
                let col_w = 38.0 * ui;
                let col_gap = 6.5 * ui;
                let meter_h = 34.0 * ui;
                let meter_y = ab_y + 13.0 * ui;
                let ax0 = bi_x + 10.0 * ui;
                for (ai, (lbl, prob, col)) in abilities.iter().enumerate() {
                    let cx = ax0 + ai as f32 * (col_w + col_gap);
                    let frac = (*prob as f32 / 255.0).clamp(0.0, 1.0);
                    let fill_h = meter_h * frac;
                    b.rect(cx, meter_y, col_w, meter_h, [0.13, 0.12, 0.22, 0.95]);
                    b.rect(cx, meter_y + meter_h - fill_h, col_w, fill_h, *col);
                    // 50% trigger threshold tick line
                    b.rect(cx, meter_y + meter_h * 0.5, col_w, 1.0 * ui, [0.95, 0.95, 1.0, 0.35]);
                    if frac >= 0.50 {
                        b.rect_outline(cx, meter_y, col_w, meter_h, 1.2 * ui, *col);
                    }
                    let lw = HudBatch::text_width(lbl, 1.05 * ui);
                    b.text(cx + (col_w - lw) * 0.5, meter_y + meter_h + 4.0 * ui, 1.05 * ui, if frac >= 0.50 { *col } else { text_hi }, lbl);
                }

                right_cursor_y += bi_h + 10.0 * ui;
            }
        }
    }

    // ------------------------------------------------------------------
    // 3C. LIVE KILL FEED (RIGHT EDGE BELOW BRAIN INSPECTOR)
    // ------------------------------------------------------------------
    if let Some(ent) = entities {
        if !ent.kill_feed.is_empty() {
            let mut ky = right_cursor_y;
            let row_h = 20.0 * ui;
            let ksc = 1.15 * ui;
            for kf in ent.kill_feed.iter().rev().take(5) {
                let (el_lbl, el_col) = match kf.elem {
                    0 => ("FIR", [1.00, 0.46, 0.16, 1.0]),
                    1 => ("FRS", [0.38, 0.90, 1.00, 1.0]),
                    2 => ("OIL", [0.78, 0.36, 0.96, 1.0]),
                    _ => ("MLE", [0.96, 0.86, 0.56, 1.0]),
                };
                let k_col = if kf.killer_team == 0 { west_col } else { east_col };
                let v_col = if kf.killer_team == 0 { east_col } else { west_col };
                let k_short: String = kf.killer_name.chars().take(11).collect();
                let v_short: String = kf.victim_name.chars().take(11).collect();
                let sep = format!(" {} ", el_lbl);
                let kw = HudBatch::text_width(&k_short, ksc);
                let sw_sep = HudBatch::text_width(&sep, ksc);
                let vw = HudBatch::text_width(&v_short, ksc);
                let pill_w = (kw + sw_sep + vw + 18.0 * ui).min(264.0 * ui);
                let kx = sw - pill_w - 12.0 * ui;

                b.rect(kx, ky, pill_w, row_h, [0.07, 0.06, 0.14, 0.80]);
                b.rect(kx, ky, 3.0 * ui, row_h, k_col);
                b.text_shadow(kx + 7.0 * ui, ky + 6.0 * ui, ksc, k_col, &k_short);
                b.text_shadow(kx + 7.0 * ui + kw, ky + 6.0 * ui, ksc, el_col, &sep);
                b.text_shadow(kx + 7.0 * ui + kw + sw_sep, ky + 6.0 * ui, ksc, v_col, &v_short);
                ky += row_h + 4.0 * ui;
            }
        }
    }

    // ------------------------------------------------------------------
    // 4. TOP-LEFT NEURAL TELEMETRY + LIVE KILL/COMBO FEED
    // ------------------------------------------------------------------
    if cfg.show_telemetry {
        let tx = 14.0 * ui;
        let ty = 14.0 * ui;
        let tw = 424.0 * ui;
        let th = 194.0 * ui;
        b.rect(tx, ty, tw, th, panel_bg);
        b.rect_outline(tx, ty, tw, th, 1.5 * ui, panel_border);

        let sc = 1.40 * ui;
        if let Some(ent) = entities {
            let t = &ent.telemetry;
            let l1 = if t.planet_mode {
                format!(
                    "PLANET 6×{}²×{}  {} T:{}",
                    t.grid_w,
                    t.grid_d,
                    t.backend.to_uppercase(),
                    t.field_tick
                )
            } else {
                format!(
                    "{} {} {}X{}X{} T:{}",
                    t.mode.to_uppercase(),
                    t.backend.to_uppercase(),
                    t.grid_w,
                    t.grid_h,
                    t.grid_d,
                    t.field_tick
                )
            };
            let l2 = if t.planet_mode {
                format!(
                    "SEAM {}  SKY-CORES {}/8  CAM:{}",
                    t.seam_mass_crossed, t.active_sky_cores, cfg.camera_preset_label
                )
            } else {
                format!(
                    "WORLD {:.2}MS ECO {:.2}MS PROJ {:.1}%",
                    t.infer_ms_last, t.eco_infer_ms, t.proj_active_pct
                )
            };
            let l3 = if t.planet_mode {
                format!(
                    "M_ERR:{} E_ERR:{} WORLD {:.2}MS",
                    t.mass_error, t.energy_error, t.infer_ms_last
                )
            } else {
                format!(
                    "MASS_ERR:{} E_ERR:{} CHUNKS:{}/{}",
                    t.mass_error, t.energy_error, t.aoi_chunks_sent, t.total_chunks
                )
            };
            let l4 = format!(
                "FPS:{:.0} RTT:{:.0}MS WIRE:{:.0}KB/S {}",
                cfg.fps, cfg.rtt_ms, t.wire_kbps, cfg.style_label
            );
            let l5 = if t.neural_bots_on {
                format!("AI: NEURAL-Z SHARED LATENT ({:.2}MS, {} BOTS)", t.bot_infer_ms, t.active_bots)
            } else {
                format!("AI: SCRIPTED FALLBACK ({} BOTS) [N TOGGLE]", t.active_bots)
            };
            let sparse_disp = if t.world_sparse_pct > 0.01 { t.world_sparse_pct } else { 100.0 };
            let l6 = if t.neural_3d_vertical {
                format!(
                    "ACOUSTIC {:.2}MS SPARSE {:.0}% 3D-NEURAL-Z",
                    t.acoustic_ms, sparse_disp
                )
            } else {
                format!(
                    "ACOUSTIC {:.2}MS  SPARSE {:.0}%",
                    t.acoustic_ms, sparse_disp
                )
            };
            let (l7, l7_col) = if t.physics_native {
                (
                    format!("PHYSICS-NATIVE [P]  FOCUS R={:.1}", cfg.focus_radius),
                    [0.24, 0.98, 0.72, 1.0],
                )
            } else {
                ("CLASSIC SPELLS [P]".to_string(), text_dim)
            };
            let dome_cnt = (t.active_time_domes as usize).max(ent.time_domes.len());
            let slm_tag = if t.system2_slm_active { "SLM:ON" } else { "SLM:OFF" };
            let l8 = if cfg.z16_active {
                if cfg.cave_enclosure >= 0.12 {
                    format!(
                        "[Y] STYLE:{}  CAVE ECHO {:.0}%",
                        cfg.z16_style_label,
                        (cfg.cave_enclosure * 100.0).clamp(0.0, 100.0)
                    )
                } else {
                    format!("[Y] STYLE:{}  3D-Z{}  DOME:{}/4", cfg.z16_style_label, gd_u, dome_cnt)
                }
            } else {
                format!("[J] DOME:{}/4 (SHIFT+J CLR)  {}", dome_cnt, slm_tag)
            };
            let l8_col = if cfg.z16_active {
                match cfg.z16_style_label {
                    "VOLCANIC" => [1.00, 0.52, 0.22, 1.0],
                    "ARCANE" => [0.58, 0.88, 1.00, 1.0],
                    "EMERALD" => [0.24, 0.98, 0.54, 1.0],
                    _ => [0.34, 0.98, 0.86, 1.0],
                }
            } else if t.system2_slm_active {
                [0.34, 0.98, 0.76, 1.0]
            } else if dome_cnt > 0 {
                [0.38, 0.94, 1.0, 1.0]
            } else {
                [0.72, 0.80, 0.94, 1.0]
            };
            b.text_shadow(tx + 10.0 * ui, ty + 10.0 * ui, sc, gold_col, &l1);
            b.text_shadow(tx + 10.0 * ui, ty + 30.0 * ui, sc, text_hi, &l2);
            b.text_shadow(
                tx + 10.0 * ui,
                ty + 50.0 * ui,
                sc,
                if t.mass_error == 0 && t.energy_error == 0 {
                    [0.36, 0.95, 0.56, 1.0]
                } else {
                    east_col
                },
                &l3,
            );
            b.text_shadow(tx + 10.0 * ui, ty + 70.0 * ui, sc, text_dim, &l4);
            b.text_shadow(
                tx + 10.0 * ui,
                ty + 90.0 * ui,
                sc,
                if t.neural_bots_on { [0.32, 0.96, 0.88, 1.0] } else { text_dim },
                &l5,
            );
            b.text_shadow(
                tx + 10.0 * ui,
                ty + 110.0 * ui,
                sc,
                if sparse_disp < 95.0 { [0.28, 0.96, 0.62, 1.0] } else { [0.76, 0.86, 1.0, 1.0] },
                &l6,
            );
            b.text_shadow(tx + 10.0 * ui, ty + 130.0 * ui, sc, l7_col, &l7);
            let mut arch_counts = [0usize; 5];
            for p in &ent.players {
                arch_counts[(p.class_id as usize).min(4)] += 1;
            }
            let l9 = format!(
                "P:{:02} STR:{} SKY:{} TID:{} STK:{} BRW:{} [F4]",
                ent.players.len(),
                arch_counts[0],
                arch_counts[1],
                arch_counts[2],
                arch_counts[3],
                arch_counts[4],
            );
            b.text_shadow(tx + 10.0 * ui, ty + 150.0 * ui, sc, l8_col, &l8);
            b.text_shadow(tx + 10.0 * ui, ty + 170.0 * ui, sc, [0.76, 0.88, 1.0, 1.0], &l9);
        }

        // Compact Shrine / Combo / Weather Feed right below telemetry (kills live in dedicated right-edge Kill Feed)
        if let Some(ent) = entities {
            let min_kind = if ent.kill_feed.is_empty() { 1 } else { 2 };
            let mut fy = ty + th + 10.0 * ui;
            for ev in ent
                .events
                .iter()
                .rev()
                .filter(|e| e.kind >= min_kind && e.kind <= 5)
                .take(5)
            {
                let fcol = match ev.kind {
                    1 => if ev.team == 0 { west_col } else { east_col },
                    3 | 4 => gold_col,
                    5 => [0.68, 0.92, 1.0, 1.0],
                    _ => [1.0, 0.72, 0.38, 1.0],
                };
                let fw = (HudBatch::text_width(&ev.label, 1.4 * ui) + 16.0 * ui).min(340.0 * ui);
                b.rect(tx, fy, fw, 20.0 * ui, [0.07, 0.06, 0.14, 0.74]);
                b.text_shadow(tx + 8.0 * ui, fy + 5.0 * ui, 1.4 * ui, fcol, &ev.label);
                fy += 24.0 * ui;
            }
        }
    }

    // ------------------------------------------------------------------
    // 5. BOTTOM-CENTER ACTION-MMO HERO PLATE & 5-SLOT SPELL HOTBAR
    // ------------------------------------------------------------------
    let hud_w = 584.0 * ui;
    let hud_h = 114.0 * ui;
    let hx = (sw - hud_w) * 0.5;
    let hy = sh - hud_h - 14.0 * ui;
    b.rect(hx, hy, hud_w, hud_h, panel_bg);
    b.rect_outline(hx, hy, hud_w, hud_h, 2.0 * ui, panel_border);

    let self_p = entities.and_then(|e| {
        e.players
            .iter()
            .find(|p| p.id == e.your_player_id)
            .or_else(|| e.players.first())
    });

    if let Some(p) = self_p {
        let arch_name = archetype_tag(p.class_id);
        let arch_col = archetype_color(p.class_id);
        let is_solo = archetype_is_solo(p.class_id);
        let role_lbl = if is_solo { "SOLO" } else { "SQUAD" };
        let role_col = if is_solo {
            [1.00, 0.42, 0.18, 1.0]
        } else {
            [0.36, 0.86, 1.00, 1.0]
        };
        let alt_tag = if cfg.z16_active {
            format!("Z:{:.1}M", p.z)
        } else if p.class_id == CLASS_SKYWING && p.z >= 1.35 {
            "AIR".to_string()
        } else if p.class_id == CLASS_BURROWER && p.z < 0.75 {
            "DEEP".to_string()
        } else if p.z >= 1.55 {
            "L2".to_string()
        } else {
            "L1".to_string()
        };
        let header = format!("LV.{} {} [{}] K:{} D:{}", p.level, arch_name, alt_tag, p.kills, p.deaths);
        b.text_shadow(hx + 14.0 * ui, hy + 10.0 * ui, 1.60 * ui, arch_col, &header);

        // Squad / Solo role indicator pill right after the archetype header
        let hdr_w = HudBatch::text_width(&header, 1.60 * ui);
        let rp_x = hx + 22.0 * ui + hdr_w;
        let rp_w = HudBatch::text_width(role_lbl, 1.18 * ui) + 10.0 * ui;
        b.rect(rp_x, hy + 8.5 * ui, rp_w, 15.0 * ui, [role_col[0] * 0.24, role_col[1] * 0.24, role_col[2] * 0.24, 0.92]);
        b.rect_outline(rp_x, hy + 8.5 * ui, rp_w, 15.0 * ui, 1.1 * ui, role_col);
        b.text(rp_x + 5.0 * ui, hy + 11.5 * ui, 1.18 * ui, role_col, role_lbl);

        // Active status pills on top-right of hero plate
        let mut sx = hx + hud_w - 14.0 * ui;
        let pills = [
            (STATUS_BURNING, "BURN", [1.0, 0.42, 0.16, 1.0]),
            (STATUS_CHILLED, "CHILL", [0.46, 0.90, 1.0, 1.0]),
            (STATUS_OILED, "OIL", [0.76, 0.36, 0.92, 1.0]),
            (STATUS_SHIELDED, "SHIELD", [0.38, 0.78, 1.0, 1.0]),
            (STATUS_HASTE, "HASTE", [0.38, 0.98, 0.56, 1.0]),
        ];
        for (bit, lbl, col) in pills {
            if (p.status_flags & bit) != 0 {
                let pw = HudBatch::text_width(lbl, 1.3 * ui) + 10.0 * ui;
                sx -= pw + 6.0 * ui;
                b.rect(sx, hy + 8.0 * ui, pw, 16.0 * ui, [col[0] * 0.28, col[1] * 0.28, col[2] * 0.28, 0.92]);
                b.rect_outline(sx, hy + 8.0 * ui, pw, 16.0 * ui, 1.0 * ui, col);
                b.text(sx + 5.0 * ui, hy + 11.0 * ui, 1.3 * ui, col, lbl);
            }
        }

        // HP, Mana, Stamina bars (left column)
        let bar_x = hx + 14.0 * ui;
        let bar_w = 216.0 * ui;
        let hp_f = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
        let mp_f = (p.mana as f32 / p.max_mana.max(1) as f32).clamp(0.0, 1.0);
        let st_f = (p.stamina as f32 / p.max_stamina.max(1) as f32).clamp(0.0, 1.0);

        // HP
        b.rect(bar_x, hy + 32.0 * ui, bar_w, 16.0 * ui, [0.16, 0.10, 0.14, 0.95]);
        b.rect(bar_x, hy + 32.0 * ui, bar_w * hp_f, 16.0 * ui, [0.22, 0.86, 0.44, 0.98]);
        b.text_shadow(bar_x + 6.0 * ui, hy + 35.0 * ui, 1.4 * ui, text_hi, &format!("HP {}/{}", p.hp, p.max_hp));

        // Mana
        b.rect(bar_x, hy + 53.0 * ui, bar_w, 14.0 * ui, [0.10, 0.12, 0.22, 0.95]);
        b.rect(bar_x, hy + 53.0 * ui, bar_w * mp_f, 14.0 * ui, [0.24, 0.62, 1.0, 0.98]);
        b.text_shadow(bar_x + 6.0 * ui, hy + 55.5 * ui, 1.3 * ui, text_hi, &format!("MP {}/{}", p.mana, p.max_mana));

        // Stamina
        b.rect(bar_x, hy + 72.0 * ui, bar_w, 11.0 * ui, [0.16, 0.15, 0.10, 0.95]);
        b.rect(bar_x, hy + 72.0 * ui, bar_w * st_f, 11.0 * ui, [0.96, 0.78, 0.24, 0.98]);
        b.text_shadow(bar_x + 6.0 * ui, hy + 73.5 * ui, 1.15 * ui, text_hi, &format!("STA {}", p.stamina));

        let phys_native = entities.map(|e| e.telemetry.physics_native).unwrap_or(false);

        // Controls hint bar at bottom
        let hint_bar = if cfg.z16_active {
            if phys_native {
                format!("1-3 MAT [/] R={:.1}  Y STYLE  G:SQD  F1-F3/F8/F10 ARCH  F4", cfg.focus_radius)
            } else {
                "1-3 ELEM  Y STYLE  G:SQD  F1-F3/F8/F10 ARCH  F4 TELEM  F7 BIOME".to_string()
            }
        } else if phys_native {
            format!("1-3 MAT [/] R={:.1}  J DOME  K TIME  G:SQD  F1-F3/F8/F10 ARCH", cfg.focus_radius)
        } else {
            "1-3 ELEM  J DOME  K TIME  G:SQD  F1-F3/F8/F10 ARCH  F4 TELEM".to_string()
        };
        b.text(
            bar_x,
            hy + 92.0 * ui,
            1.02 * ui,
            text_dim,
            &hint_bar,
        );

        // 5-Slot Action Hotbar (right side of hero plate: Continuous Actuators or Classic Spells)
        let slot_sz = 54.0 * ui;
        let slot_gap = 8.0 * ui;
        let sx0 = hx + 266.0 * ui;
        let sy0 = hy + 32.0 * ui;
        let slots: [(&str, &str, [f32; 4], f32); 5] = if phys_native {
            let matter_lbl = match cfg.selected_element % 3 {
                0 => ("OIL", [0.78, 0.36, 0.96, 1.0]),
                1 => ("WATER", [0.26, 0.66, 1.0, 1.0]),
                _ => ("SMOKE", [0.74, 0.80, 0.90, 1.0]),
            };
            [
                ("LMB", "+HEAT", [1.0, 0.46, 0.16, 1.0], 0.0),
                ("RMB", "-CRYO", [0.36, 0.90, 1.0, 1.0], 0.0),
                ("SPC", "DASH", [0.48, 0.92, 1.0, 1.0], p.dash_cd_frac),
                ("E", matter_lbl.0, matter_lbl.1, 0.0),
                ("R", "PULSE", [1.0, 0.72, 0.24, 1.0], 0.0),
            ]
        } else {
            let elem_lbl = match cfg.selected_element % 3 {
                0 => ("FIRE", [1.0, 0.50, 0.16, 1.0]),
                1 => ("FROST", [0.42, 0.92, 1.0, 1.0]),
                _ => ("OIL", [0.78, 0.36, 0.96, 1.0]),
            };
            let e_lbl = match p.class_id {
                CLASS_STRIDER => "BRIDGE",
                CLASS_SKYWING => "UPDRAFT",
                CLASS_TIDEKIN => "SURGE",
                CLASS_EMBERSTALKER => "SHADOW",
                CLASS_BURROWER => "TUNNEL",
                _ => "SPRINT",
            };
            let r_lbl = match p.class_id {
                CLASS_STRIDER => "NOVA",
                CLASS_SKYWING => "METEOR",
                CLASS_TIDEKIN => "TIDAL",
                CLASS_EMBERSTALKER => "BLADES",
                CLASS_BURROWER => "QUAKE",
                _ => "SURGE",
            };
            [
                ("LMB", elem_lbl.0, elem_lbl.1, 0.0),
                ("RMB", "MELEE", [0.96, 0.86, 0.56, 1.0], 0.0),
                ("SPC", "DASH", [0.48, 0.92, 1.0, 1.0], p.dash_cd_frac),
                ("E", e_lbl, [0.46, 0.96, 0.62, 1.0], p.ability_e_cd_frac),
                ("R", r_lbl, [1.0, 0.68, 0.24, 1.0], p.ult_r_cd_frac),
            ]
        };

        for (i, (key, name, col, cd)) in slots.iter().enumerate() {
            let x = sx0 + i as f32 * (slot_sz + slot_gap);
            b.rect(x, sy0, slot_sz, slot_sz, [0.13, 0.12, 0.23, 0.94]);
            b.rect(x + 3.0 * ui, sy0 + 3.0 * ui, slot_sz - 6.0 * ui, 6.0 * ui, *col);
            b.rect_outline(x, sy0, slot_sz, slot_sz, 1.8 * ui, if *cd <= 0.01 { *col } else { panel_border });
            b.text_shadow(x + 6.0 * ui, sy0 + 14.0 * ui, 1.35 * ui, text_hi, key);
            b.text(x + 5.0 * ui, sy0 + 34.0 * ui, 1.15 * ui, *col, name);
            if *cd > 0.01 {
                let h = slot_sz * cd.clamp(0.0, 1.0);
                b.rect(x, sy0 + slot_sz - h, slot_sz, h, [0.04, 0.03, 0.10, 0.72]);
            }
        }
    }

    // ------------------------------------------------------------------
    // 6. BOTTOM-LEFT ACOUSTIC TRANSCRIPT LOG (3 LINES MAX) + CHAT BOX
    //    Anchored strictly above the action bar height so it never
    //    overlaps top-left telemetry or the bottom-center hero plate!
    // ------------------------------------------------------------------
    {
        let chat_x = 14.0 * ui;
        let chat_w = 436.0 * ui;
        let mut cursor_bottom = hy - 14.0 * ui;

        if cfg.chat_open {
            let inp_h = 30.0 * ui;
            let inp_y = cursor_bottom - inp_h;
            let mode_col = if cfg.chat_shout {
                [1.00, 0.76, 0.24, 1.0]
            } else {
                [0.36, 0.88, 1.00, 1.0]
            };
            let mode_lbl = if cfg.chat_shout { "[SHOUT 1.0]" } else { "[WHISPER 0.25]" };

            b.rect(chat_x, inp_y, chat_w, inp_h, [0.06, 0.05, 0.15, 0.94]);
            b.rect_outline(chat_x, inp_y, chat_w, inp_h, 2.0 * ui, mode_col);

            let msc = 1.20 * ui;
            let mw = HudBatch::text_width(mode_lbl, msc) + 8.0 * ui;
            b.rect(
                chat_x + 5.0 * ui,
                inp_y + 5.0 * ui,
                mw,
                inp_h - 10.0 * ui,
                [mode_col[0] * 0.22, mode_col[1] * 0.22, mode_col[2] * 0.22, 0.95],
            );
            b.text(chat_x + 9.0 * ui, inp_y + 10.0 * ui, msc, mode_col, mode_lbl);

            let max_chars = 32usize;
            let shown_inp: String = if cfg.chat_input.chars().count() > max_chars {
                cfg.chat_input
                    .chars()
                    .rev()
                    .take(max_chars)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect()
            } else {
                cfg.chat_input.to_string()
            };
            let line = format!("{}_", shown_inp);
            b.text_shadow(
                chat_x + mw + 12.0 * ui,
                inp_y + 10.0 * ui,
                1.30 * ui,
                text_hi,
                &line,
            );
            let hint = "[TAB] VOL  [ESC]";
            let hw = HudBatch::text_width(hint, 1.00 * ui);
            b.text(
                chat_x + chat_w - hw - 6.0 * ui,
                inp_y - 11.0 * ui,
                1.00 * ui,
                text_dim,
                hint,
            );
            cursor_bottom = inp_y - 16.0 * ui;
        } else {
            let hint_h = 18.0 * ui;
            let hint_y = cursor_bottom - hint_h;
            let hint_txt = "[ENTER] ACOUSTIC CHAT (SHOUT / WHISPER)";
            let hw = HudBatch::text_width(hint_txt, 1.10 * ui) + 14.0 * ui;
            b.rect(chat_x, hint_y, hw, hint_h, [0.06, 0.05, 0.14, 0.68]);
            b.rect_outline(chat_x, hint_y, hw, hint_h, 1.0 * ui, panel_border);
            b.text(chat_x + 7.0 * ui, hint_y + 5.0 * ui, 1.10 * ui, text_dim, hint_txt);
            cursor_bottom = hint_y - 6.0 * ui;
        }

        if let Some(ent) = entities {
            if !ent.speech_bubbles.is_empty() {
                let row_h = 22.0 * ui;
                let row_gap = 4.0 * ui;
                let recent: Vec<&SpeechBubbleSnap> = ent.speech_bubbles.iter().rev().take(3).collect();
                for sb in recent {
                    let ry = cursor_bottom - row_h;
                    // Never climb into top-left telemetry zone
                    if ry < 275.0 * ui {
                        break;
                    }
                    let (snr_lbl, snr_col) = snr_badge(sb.snr_db);
                    let is_self = sb.speaker_id == ent.your_player_id || sb.speaker_name == "YOU";
                    let spk_col = if is_self {
                        gold_col
                    } else if sb.team == 0 {
                        west_col
                    } else {
                        east_col
                    };
                    let spk_short: String = sb.speaker_name.chars().take(13).collect();
                    let msg_short: String = sb.text.chars().take(44).collect();

                    b.rect(chat_x, ry, chat_w, row_h, [0.06, 0.05, 0.14, 0.82]);
                    b.rect(chat_x, ry, 3.0 * ui, row_h, spk_col);

                    let bsc = 1.05 * ui;
                    let bw = HudBatch::text_width(&snr_lbl, bsc) + 7.0 * ui;
                    b.rect(
                        chat_x + 7.0 * ui,
                        ry + 4.0 * ui,
                        bw,
                        row_h - 8.0 * ui,
                        [snr_col[0] * 0.22, snr_col[1] * 0.22, snr_col[2] * 0.22, 0.94],
                    );
                    b.rect_outline(chat_x + 7.0 * ui, ry + 4.0 * ui, bw, row_h - 8.0 * ui, 1.0 * ui, snr_col);
                    b.text(chat_x + 10.5 * ui, ry + 7.0 * ui, bsc, snr_col, &snr_lbl);

                    let mut tx_x = chat_x + 12.0 * ui + bw;
                    let tsc = 1.15 * ui;
                    if !is_self {
                        let spk_tag = format!("{}: ", spk_short);
                        b.text_shadow(tx_x, ry + 6.5 * ui, tsc, spk_col, &spk_tag);
                        tx_x += HudBatch::text_width(&spk_tag, tsc) + 2.0 * ui;
                    }
                    b.text_shadow(tx_x, ry + 6.5 * ui, tsc, text_hi, &msg_short);
                    cursor_bottom = ry - row_gap;
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // 6B. BOTTOM-RIGHT COMPACT SFX & SCOREBOARD STATUS PILLS
    // ------------------------------------------------------------------
    {
        let pill_h = 20.0 * ui;
        let py = sh - pill_h - 14.0 * ui;
        let sc = 1.15 * ui;

        let (sfx_lbl, sfx_col) = if !cfg.audio_available {
            ("SFX:OFF", text_dim)
        } else if cfg.audio_muted {
            ("[U] SFX:MUTE", [1.0, 0.44, 0.34, 1.0])
        } else {
            ("[U] SFX:ON", [0.30, 0.96, 0.62, 1.0])
        };
        let sfx_w = HudBatch::text_width(sfx_lbl, sc) + 14.0 * ui;
        let sfx_x = sw - sfx_w - 14.0 * ui;
        b.rect(sfx_x, py, sfx_w, pill_h, [0.06, 0.05, 0.14, 0.82]);
        b.rect_outline(sfx_x, py, sfx_w, pill_h, 1.2 * ui, sfx_col);
        b.text_shadow(sfx_x + 7.0 * ui, py + 6.0 * ui, sc, sfx_col, sfx_lbl);

        // Live [F11] World-Law preset pill directly above bottom-right status row (zero overlap!)
        let law_id = crate::net::current_world_law_preset();
        let law_name = world_law_preset_name(law_id);
        let law_col = match law_id % 4 {
            1 => [0.42, 0.92, 1.00, 1.0], // FREEZE-OUT
            2 => [0.24, 0.76, 1.00, 1.0], // RISING FLOOD
            3 => [1.00, 0.48, 0.18, 1.0], // LOW-G MAGMA
            _ => [0.76, 0.86, 0.98, 1.0], // STANDARD
        };
        let law_lbl = format!("[F11] LAW:{}", law_name);
        let law_w = HudBatch::text_width(&law_lbl, sc) + 14.0 * ui;
        let law_x = sw - law_w - 14.0 * ui;
        let law_y = py - pill_h - 6.0 * ui;
        if law_x > hx + hud_w + 8.0 * ui {
            b.rect(law_x, law_y, law_w, pill_h, [0.06, 0.05, 0.14, 0.86]);
            b.rect_outline(law_x, law_y, law_w, pill_h, 1.2 * ui, law_col);
            b.text_shadow(law_x + 7.0 * ui, law_y + 6.0 * ui, sc, law_col, &law_lbl);
        }

        let tab_lbl = "[TAB] SCORE";
        let tab_w = HudBatch::text_width(tab_lbl, sc) + 14.0 * ui;
        let mut cursor_x = sfx_x - tab_w - 8.0 * ui;
        if cursor_x > hx + hud_w + 8.0 * ui {
            b.rect(cursor_x, py, tab_w, pill_h, [0.06, 0.05, 0.14, 0.72]);
            b.rect_outline(cursor_x, py, tab_w, pill_h, 1.0 * ui, panel_border);
            b.text(cursor_x + 7.0 * ui, py + 6.0 * ui, sc, text_dim, tab_lbl);
        }

        let (sqd_lbl, sqd_col) = if cfg.show_squad_links {
            ("G:SQD ON", [0.36, 0.88, 1.00, 1.0])
        } else {
            ("G:SQD OFF", text_dim)
        };
        let sqd_w = HudBatch::text_width(sqd_lbl, sc) + 14.0 * ui;
        let sqd_x = cursor_x - sqd_w - 8.0 * ui;
        if sqd_x > hx + hud_w + 8.0 * ui {
            b.rect(sqd_x, py, sqd_w, pill_h, [0.06, 0.05, 0.14, 0.82]);
            b.rect_outline(
                sqd_x,
                py,
                sqd_w,
                pill_h,
                if cfg.show_squad_links { 1.2 * ui } else { 1.0 * ui },
                if cfg.show_squad_links { sqd_col } else { panel_border },
            );
            b.text_shadow(sqd_x + 7.0 * ui, py + 6.0 * ui, sc, sqd_col, sqd_lbl);
            cursor_x = sqd_x;
        }

        // Subtle [F5] Camera Preset indicator pill (`ORBITAL`/`TACTICAL`/`SURFACE` or `TACTICAL`/`ACTION`/`OVERVIEW`)
        let cam_col = match cfg.camera_preset_label {
            "ORBITAL" => [0.42, 0.88, 1.00, 1.0],
            "SURFACE" | "ACTION" => [1.00, 0.78, 0.28, 1.0],
            "OVERVIEW" => [0.64, 0.84, 1.00, 1.0],
            _ => [0.76, 0.86, 0.98, 1.0],
        };
        let cam_lbl = format!("[F5] CAM:{}", cfg.camera_preset_label);
        let cam_w = HudBatch::text_width(&cam_lbl, sc) + 14.0 * ui;
        let cam_x = cursor_x - cam_w - 8.0 * ui;
        if cam_x > hx + hud_w + 8.0 * ui {
            b.rect(cam_x, py, cam_w, pill_h, [0.06, 0.05, 0.14, 0.84]);
            b.rect_outline(cam_x, py, cam_w, pill_h, 1.1 * ui, cam_col);
            b.text_shadow(cam_x + 7.0 * ui, py + 6.0 * ui, sc, cam_col, &cam_lbl);
            cursor_x = cam_x;
        }

        if cfg.z16_active {
            let style_col = match cfg.z16_style_label {
                "VOLCANIC" => [1.00, 0.48, 0.18, 1.0],
                "ARCANE" => [0.56, 0.88, 1.00, 1.0],
                "EMERALD" => [0.24, 0.98, 0.54, 1.0],
                _ => [0.34, 0.96, 0.84, 1.0],
            };
            let st_lbl = format!("[Y] STYLE:{}", cfg.z16_style_label);
            let st_w = HudBatch::text_width(&st_lbl, sc) + 14.0 * ui;
            let st_x = cursor_x - st_w - 8.0 * ui;
            if st_x > hx + hud_w + 8.0 * ui {
                b.rect(st_x, py, st_w, pill_h, [0.06, 0.05, 0.14, 0.86]);
                b.rect_outline(st_x, py, st_w, pill_h, 1.3 * ui, style_col);
                b.text_shadow(st_x + 7.0 * ui, py + 6.0 * ui, sc, style_col, &st_lbl);
                cursor_x = st_x;
            }
            if cfg.cave_enclosure >= 0.12 {
                let echo_lbl = format!("CAVE ECHO {:.0}%", (cfg.cave_enclosure * 100.0).clamp(0.0, 100.0));
                let echo_w = HudBatch::text_width(&echo_lbl, sc) + 14.0 * ui;
                let echo_x = cursor_x - echo_w - 8.0 * ui;
                if echo_x > hx + hud_w + 8.0 * ui {
                    let echo_col = [0.46, 0.88, 1.00, 1.0];
                    b.rect(echo_x, py, echo_w, pill_h, [0.06, 0.10, 0.20, 0.90]);
                    b.rect_outline(echo_x, py, echo_w, pill_h, 1.3 * ui, echo_col);
                    b.text_shadow(echo_x + 7.0 * ui, py + 6.0 * ui, sc, echo_col, &echo_lbl);
                    cursor_x = echo_x;
                }
            }
            let z_lbl = format!("3D-Z{}", gd_u);
            let z_w = HudBatch::text_width(&z_lbl, sc) + 14.0 * ui;
            let z_x = cursor_x - z_w - 8.0 * ui;
            if z_x > hx + hud_w + 8.0 * ui {
                let z_col = [0.38, 0.94, 1.0, 1.0];
                b.rect(z_x, py, z_w, pill_h, [0.06, 0.10, 0.18, 0.88]);
                b.rect_outline(z_x, py, z_w, pill_h, 1.3 * ui, z_col);
                b.text_shadow(z_x + 7.0 * ui, py + 6.0 * ui, sc, z_col, &z_lbl);
            }
        }
    }

    // ------------------------------------------------------------------
    // 7. CENTERED VICTORY / MATCH SUMMARY BANNER (`match_phase != 0`)
    // ------------------------------------------------------------------
    if let Some(ent) = entities {
        if ent.match_phase != MATCH_PHASE_ACTIVE && !cfg.show_scoreboard {
            let (title, accent) = match ent.match_phase {
                MATCH_PHASE_WEST_VICTORY => ("WEST VICTORY!", west_col),
                MATCH_PHASE_EAST_VICTORY => ("EAST VICTORY!", east_col),
                _ => ("MATCH DRAW!", gold_col),
            };
            let vb_w = (540.0 * ui).min(sw - 40.0 * ui);
            let vb_h = 192.0 * ui;
            let vx = (sw - vb_w) * 0.5;
            let vy = ((sh - vb_h) * 0.5 - 24.0 * ui).max(82.0 * ui);

            b.rect(vx, vy, vb_w, vb_h, [0.05, 0.04, 0.12, 0.94]);
            b.rect(vx, vy, vb_w, 5.0 * ui, accent);
            b.rect(vx, vy + vb_h - 3.0 * ui, vb_w, 3.0 * ui, accent);
            b.rect_outline(vx, vy, vb_w, vb_h, 2.2 * ui, accent);

            // Title
            let tsc = 3.1 * ui;
            let tw = HudBatch::text_width(title, tsc);
            b.text_shadow(vx + (vb_w - tw) * 0.5, vy + 22.0 * ui, tsc, accent, title);

            // Final Score
            let score_line = format!(
                "WEST {:03}   -   {:03} EAST",
                ent.telemetry.score_west, ent.telemetry.score_east
            );
            let ssc = 2.0 * ui;
            let sw_txt = HudBatch::text_width(&score_line, ssc);
            b.text_shadow(vx + (vb_w - sw_txt) * 0.5, vy + 66.0 * ui, ssc, text_hi, &score_line);

            // MVP Callout Strip
            let mvp_box_x = vx + 26.0 * ui;
            let mvp_box_y = vy + 102.0 * ui;
            let mvp_box_w = vb_w - 52.0 * ui;
            let mvp_box_h = 36.0 * ui;
            b.rect(mvp_box_x, mvp_box_y, mvp_box_w, mvp_box_h, [0.13, 0.11, 0.24, 0.94]);
            b.rect_outline(mvp_box_x, mvp_box_y, mvp_box_w, mvp_box_h, 1.5 * ui, gold_col);
            let mvp_disp = if ent.mvp_name.is_empty() { "NONE" } else { ent.mvp_name.as_str() };
            let mvp_line = format!("MATCH MVP: {}  ({} PTS)", mvp_disp, ent.mvp_score);
            let msc = 1.55 * ui;
            let mw = HudBatch::text_width(&mvp_line, msc);
            b.text_shadow(
                mvp_box_x + (mvp_box_w - mw) * 0.5,
                mvp_box_y + 12.0 * ui,
                msc,
                gold_col,
                &mvp_line,
            );

            // Next-match countdown
            let cd_line = format!(
                "NEXT MATCH IN {:02}S   |   HOLD [TAB] FOR SCOREBOARD",
                ent.match_time_left_s
            );
            let csc = 1.3 * ui;
            let cw = HudBatch::text_width(&cd_line, csc);
            b.text_shadow(vx + (vb_w - cw) * 0.5, vy + 156.0 * ui, csc, text_dim, &cd_line);
        }
    }

    // ------------------------------------------------------------------
    // 8. LIVE [TAB] MATCH SCOREBOARD OVERLAY (SCALES CLEANLY TO 64 PLAYERS!)
    // ------------------------------------------------------------------
    if cfg.show_scoreboard {
        if let Some(ent) = entities {
            let mut west_team: Vec<&PlayerSnap> =
                ent.players.iter().filter(|p| p.team == 0).collect();
            let mut east_team: Vec<&PlayerSnap> =
                ent.players.iter().filter(|p| p.team != 0).collect();
            let sort_players = |a: &&PlayerSnap, b: &&PlayerSnap| {
                b.score
                    .cmp(&a.score)
                    .then_with(|| b.kills.cmp(&a.kills))
                    .then_with(|| a.id.cmp(&b.id))
            };
            west_team.sort_by(sort_players);
            east_team.sort_by(sort_players);

            let team_max = west_team.len().max(east_team.len()).clamp(1, 32);
            let top_guard = by + banner_h + 30.0 * ui;
            let bot_guard = hy - 12.0 * ui;
            let avail_h = (bot_guard - top_guard).max(220.0 * ui);
            let chrome_h = 104.0 * ui;
            let rows_avail_h = (avail_h - chrome_h).max(110.0 * ui);

            let (row_h, rsc) = if team_max <= 8 {
                (24.0 * ui, 1.12 * ui)
            } else if team_max <= 16 {
                (19.5 * ui, 1.00 * ui)
            } else if team_max <= 24 {
                (17.0 * ui, 0.92 * ui)
            } else {
                (15.2 * ui, 0.85 * ui)
            };
            let visible_rows = ((rows_avail_h / row_h).floor() as usize)
                .clamp(6, 32)
                .min(team_max);
            let sb_w = (880.0 * ui).min(sw - 32.0 * ui);
            let sb_h = (chrome_h + visible_rows as f32 * row_h).min(avail_h);
            let sb_x = (sw - sb_w) * 0.5;
            let sb_y = (top_guard + (avail_h - sb_h) * 0.5).max(top_guard);

            // Sleek centered modal card
            b.rect(sb_x, sb_y, sb_w, sb_h, [0.05, 0.04, 0.13, 0.94]);
            b.rect(sb_x, sb_y, sb_w, 3.5 * ui, gold_col);
            b.rect_outline(sb_x, sb_y, sb_w, sb_h, 2.0 * ui, panel_border);

            let west_shrines = ent.shrines.iter().filter(|s| s.owner < 0).count();
            let east_shrines = ent.shrines.iter().filter(|s| s.owner > 0).count();

            // Header bar with live team counts
            let hdr_y = sb_y + 11.0 * ui;
            let w_hdr = format!(
                "WEST [{}] {:03} ({} SHR)",
                west_team.len(),
                ent.telemetry.score_west,
                west_shrines
            );
            let e_hdr = format!(
                "({} SHR) {:03} [{}] EAST",
                east_shrines,
                ent.telemetry.score_east,
                east_team.len()
            );
            let hsc = 1.55 * ui;
            b.text_shadow(sb_x + 16.0 * ui, hdr_y, hsc, west_col, &w_hdr);
            let ew = HudBatch::text_width(&e_hdr, hsc);
            b.text_shadow(sb_x + sb_w - ew - 16.0 * ui, hdr_y, hsc, east_col, &e_hdr);

            let clk_str = if ent.match_phase != MATCH_PHASE_ACTIVE {
                format!("NEXT MATCH {:02}S", ent.match_time_left_s)
            } else {
                format!(
                    "{:02}:{:02} LEFT ({}P)",
                    ent.match_time_left_s / 60,
                    ent.match_time_left_s % 60,
                    ent.players.len()
                )
            };
            let cw = HudBatch::text_width(&clk_str, 1.45 * ui);
            b.text_shadow(sb_x + (sb_w - cw) * 0.5, hdr_y + 1.0 * ui, 1.45 * ui, gold_col, &clk_str);

            // Divider below header
            let table_y = sb_y + 36.0 * ui;
            b.rect(sb_x + 12.0 * ui, table_y - 4.0 * ui, sb_w - 24.0 * ui, 1.5 * ui, panel_border);

            let col_gap = 14.0 * ui;
            let half_w = (sb_w - 24.0 * ui - col_gap) * 0.5;
            let left_x = sb_x + 12.0 * ui;
            let right_x = left_x + half_w + col_gap;
            // Center vertical divider
            b.rect(
                left_x + half_w + col_gap * 0.5 - 0.75 * ui,
                table_y,
                1.5 * ui,
                (sb_h - 68.0 * ui).max(40.0 * ui),
                panel_border,
            );

            let teams_draw: [(&[&PlayerSnap], f32, [f32; 4]); 2] = [
                (&west_team, left_x, west_col),
                (&east_team, right_x, east_col),
            ];
            let neural_on = ent.telemetry.neural_bots_on;

            for (team_list, tx, tcol) in teams_draw {
                // Column labels
                let csc = 1.00 * ui;
                let off_arch = half_w * 0.34;
                let off_lv = half_w * 0.54;
                let off_kd = half_w * 0.62;
                let off_scr = half_w * 0.74;
                let off_tag = half_w * 0.85;

                b.text(tx + 8.0 * ui, table_y + 2.0 * ui, csc, tcol, "NAME");
                b.text(tx + off_arch, table_y + 2.0 * ui, csc, text_dim, "ARCH");
                b.text(tx + off_lv, table_y + 2.0 * ui, csc, text_dim, "LV");
                b.text(tx + off_kd, table_y + 2.0 * ui, csc, text_dim, "K/D");
                b.text(tx + off_scr, table_y + 2.0 * ui, csc, text_dim, "SCR");
                b.text(tx + off_tag, table_y + 2.0 * ui, csc, text_dim, "ROLE");

                // Pin local player into last visible slot if viewport is small and player is below cutoff
                let mut rows_to_show: Vec<&PlayerSnap> =
                    team_list.iter().take(visible_rows).copied().collect();
                if team_list.len() > visible_rows && visible_rows >= 2 {
                    let self_in_shown = rows_to_show.iter().any(|p| p.id == ent.your_player_id);
                    if !self_in_shown {
                        if let Some(self_row) =
                            team_list.iter().find(|p| p.id == ent.your_player_id)
                        {
                            rows_to_show[visible_rows - 1] = self_row;
                        }
                    }
                }

                let mut ry = table_y + 17.0 * ui;
                let txt_off_y = ((row_h - 7.0 * rsc) * 0.48).max(2.0 * ui);
                for (ri, p) in rows_to_show.iter().enumerate() {
                    let is_self = p.id == ent.your_player_id;
                    let is_solo = archetype_is_solo(p.class_id);
                    let row_bg = if is_self {
                        [0.24, 0.20, 0.10, 0.88]
                    } else if ri % 2 == 0 {
                        [0.09, 0.08, 0.19, 0.78]
                    } else {
                        [0.07, 0.06, 0.15, 0.68]
                    };
                    let inner_h = (row_h - 1.5 * ui).max(8.0 * ui);
                    b.rect(tx, ry, half_w, inner_h, row_bg);
                    if is_self {
                        b.rect(tx, ry, 3.0 * ui, inner_h, gold_col);
                        b.rect_outline(tx, ry, half_w, inner_h, 1.0 * ui, gold_col);
                    } else if is_solo {
                        b.rect(tx, ry, 2.5 * ui, inner_h, [1.0, 0.42, 0.16, 1.0]);
                    } else {
                        b.rect(tx, ry, 2.5 * ui, inner_h, tcol);
                    }

                    let name_s: String = p.name.chars().take(10).collect();
                    let arch_s = archetype_tag(p.class_id);
                    let arch_c = archetype_color(p.class_id);
                    let lv_s = format!("L{}", p.level);
                    let kd_s = format!("{}/{}", p.kills, p.deaths);
                    let scr_s = format!("{}", p.score);
                    let (tag_s, tag_c) = if !p.is_bot {
                        ("HUMAN", gold_col)
                    } else if is_solo {
                        ("SOLO", [1.0, 0.44, 0.18, 1.0])
                    } else if neural_on {
                        ("RL-Z", [0.24, 0.94, 0.92, 1.0])
                    } else {
                        ("BOT", text_dim)
                    };

                    let ty_txt = ry + txt_off_y;
                    b.text_shadow(tx + 7.0 * ui, ty_txt, rsc, if is_self { gold_col } else { text_hi }, &name_s);
                    b.text(tx + off_arch, ty_txt, rsc, arch_c, arch_s);
                    b.text(tx + off_lv, ty_txt, rsc, text_dim, &lv_s);
                    b.text(tx + off_kd, ty_txt, rsc, text_hi, &kd_s);
                    b.text_shadow(tx + off_scr, ty_txt, rsc, gold_col, &scr_s);

                    // Compact role badge pill
                    let bsc = (rsc * 0.84).max(0.72 * ui);
                    let bw = HudBatch::text_width(tag_s, bsc) + 5.0 * ui;
                    let bx_p = (tx + off_tag - 2.0 * ui).min(tx + half_w - bw - 3.0 * ui);
                    let bh_p = (inner_h - 3.0 * ui).clamp(9.5 * ui, 13.5 * ui);
                    let by_p = ry + (inner_h - bh_p) * 0.5;
                    b.rect(bx_p, by_p, bw, bh_p, [tag_c[0] * 0.22, tag_c[1] * 0.22, tag_c[2] * 0.22, 0.94]);
                    b.rect_outline(bx_p, by_p, bw, bh_p, 1.0 * ui, tag_c);
                    b.text(bx_p + 2.5 * ui, by_p + ((bh_p - 7.0 * bsc) * 0.5).max(1.0 * ui), bsc, tag_c, tag_s);

                    ry += row_h;
                }
            }

            // Footer: Shrine control + hidden-row note (if viewport constrained) + Match target
            let foot_y = sb_y + sb_h - 24.0 * ui;
            b.rect(sb_x + 12.0 * ui, foot_y - 5.0 * ui, sb_w - 24.0 * ui, 1.2 * ui, panel_border);
            let mut sx_f = sb_x + 16.0 * ui;
            for s in ent.shrines.iter().take(3) {
                let scol = if s.owner < 0 { west_col } else if s.owner > 0 { east_col } else { [0.60, 0.60, 0.72, 1.0] };
                b.rect(sx_f, foot_y + 2.0 * ui, 10.0 * ui, 10.0 * ui, scol);
                b.rect_outline(sx_f, foot_y + 2.0 * ui, 10.0 * ui, 10.0 * ui, 1.0 * ui, text_hi);
                sx_f += 14.0 * ui;
                let sn: String = s.name.chars().take(14).collect();
                b.text(sx_f, foot_y + 3.5 * ui, 1.05 * ui, scol, &sn);
                sx_f += HudBatch::text_width(&sn, 1.05 * ui) + 18.0 * ui;
            }
            let hidden_cnt = west_team.len().saturating_sub(visible_rows)
                + east_team.len().saturating_sub(visible_rows);
            let goal_txt = if hidden_cnt > 0 {
                format!("+{} MORE | {} PLAYERS | FIRST TO 500", hidden_cnt, ent.players.len())
            } else {
                format!("{} PLAYERS | FIRST TO 500 PTS OR 5:00", ent.players.len())
            };
            let gw_t = HudBatch::text_width(&goal_txt, 1.02 * ui);
            b.text(sb_x + sb_w - gw_t - 16.0 * ui, foot_y + 3.5 * ui, 1.02 * ui, text_dim, &goal_txt);
        }
    }

    b
}

/// Minimal crisp 5×7 uppercase bitmap font (zero external font files needed!).
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10011, 0b10101, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '!' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        '?' => [0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100, 0b01000],
        '\'' => [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '"' => [0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '<' => [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '@' => [0b01110, 0b10001, 0b10111, 0b10101, 0b10110, 0b10000, 0b01110],
        '×' => [0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b00000],
        '²' => [0b01100, 0b10010, 0b00100, 0b01000, 0b11110, 0b00000, 0b00000],
        '^' => [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000],
        _ => [0; 7],
    }
}
