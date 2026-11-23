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
use crate::net::LiveEntityState;
use crate::world_bridge::LiveApproach4World;
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
pub struct HudConfig {
    pub show_hud: bool,
    pub show_telemetry: bool,
    pub selected_element: u8,
    pub style_label: &'static str,
    pub fps: f32,
    pub rtt_ms: f32,
}

pub struct HudBatch {
    pub vertices: Vec<HudVertex>,
}

impl HudBatch {
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(8192),
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

pub fn build_hud(
    sw: f32,
    sh: f32,
    world: &LiveApproach4World,
    entities: Option<&LiveEntityState>,
    mats: &CameraMatrices,
    cfg: &HudConfig,
) -> HudBatch {
    let mut b = HudBatch::new();
    if !cfg.show_hud {
        return b;
    }
    let ui = (sh / 1080.0).clamp(0.75, 2.0);
    let panel_bg = [0.08, 0.07, 0.16, 0.82];
    let panel_border = [0.36, 0.34, 0.58, 0.85];
    let text_hi = [0.98, 0.96, 0.88, 1.0];
    let text_dim = [0.70, 0.74, 0.86, 0.92];
    let west_col = [0.28, 0.72, 1.0, 1.0];
    let east_col = [1.0, 0.36, 0.28, 1.0];
    let gold_col = [1.0, 0.84, 0.28, 1.0];

    // ------------------------------------------------------------------
    // 1. WORLD-SPACE OVERHEAD HP BARS + FLOATING COMBAT POPUPS
    // ------------------------------------------------------------------
    if let Some(ent) = entities {
        for p in &ent.players {
            let (wx, wz) = world.grid_to_world_xz(p.x, p.y);
            let gy = world.sample_3d_y(p.x, p.y, p.z);
            let rel = Vec3::new(wx, gy + 3.65, wz) - mats.origin;
            if let Some((sx, sy)) = project_world(rel, mats, sw, sh) {
                let bw = 54.0 * ui;
                let bh = 6.0 * ui;
                let bx = sx - bw * 0.5;
                let by = sy - bh;
                let hp_frac = (p.hp as f32 / p.max_hp.max(1) as f32).clamp(0.0, 1.0);
                let is_self = p.id == ent.your_player_id;
                let bar_c = if is_self {
                    [0.32, 0.96, 0.52, 0.96]
                } else if p.team == 0 {
                    west_col
                } else {
                    east_col
                };
                b.rect(bx - 1.5 * ui, by - 1.5 * ui, bw + 3.0 * ui, bh + 3.0 * ui, [0.05, 0.04, 0.10, 0.85]);
                b.rect(bx, by, bw, bh, [0.16, 0.14, 0.24, 0.9]);
                b.rect(bx, by, bw * hp_frac, bh, bar_c);
                let tag = format!("L{} {}", p.level, &p.name[..p.name.len().min(10)]);
                let sc = 1.45 * ui;
                let tw = HudBatch::text_width(&tag, sc);
                b.text_shadow(sx - tw * 0.5, by - 13.0 * ui, sc, if is_self { gold_col } else { text_hi }, &tag);
            }
        }

        // Shrine world labels
        for s in &ent.shrines {
            let (wx, wz) = world.grid_to_world_xz(s.x, s.y);
            let gy = world.sample_ground_y(s.x, s.y);
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

        // Floating damage & elemental combo callouts
        for ev in ent.events.iter().rev().take(12) {
            if ev.kind == 0 || ev.kind == 2 {
                let (wx, wz) = world.grid_to_world_xz(ev.x, ev.y);
                let gy = world.sample_ground_y(ev.x, ev.y);
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

        // Match timer + weather badge in center
        let mins = ent.match_time_secs / 60;
        let secs = ent.match_time_secs % 60;
        let w_label = match tel.weather_kind {
            WEATHER_BLIZZARD => "BLIZZARD",
            WEATHER_SQUALL => "SQUALL",
            WEATHER_EMBER => "EMBER DUSK",
            _ => "CLEAR FROST",
        };
        let mid_str = format!("{:02}:{:02} {}", mins, secs, w_label);
        let msc = 1.45 * ui;
        let mw = HudBatch::text_width(&mid_str, msc);
        b.text_shadow(bx + (banner_w - mw) * 0.5, by + 9.0 * ui, msc, gold_col, &mid_str);

        // 3 Shrine ownership pips
        let pip_sz = 12.0 * ui;
        let pip_gap = 18.0 * ui;
        let total_pips_w = 3.0 * pip_sz + 2.0 * pip_gap;
        let px0 = bx + (banner_w - total_pips_w) * 0.5;
        let py0 = by + 32.0 * ui;
        for (i, s) in ent.shrines.iter().take(3).enumerate() {
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

    let gw = world.grid_w.max(1) as f32;
    let gh = world.grid_h.max(1) as f32;
    if let Some(ent) = entities {
        // Shrines on minimap
        for s in &ent.shrines {
            let sx = mm_x + (s.x / gw) * mm_sz;
            let sy = mm_y + (s.y / gh) * mm_sz;
            let col = if s.owner < 0 { west_col } else if s.owner > 0 { east_col } else { gold_col };
            b.rect(sx - 4.0 * ui, sy - 4.0 * ui, 8.0 * ui, 8.0 * ui, col);
            b.rect_outline(sx - 4.0 * ui, sy - 4.0 * ui, 8.0 * ui, 8.0 * ui, 1.2 * ui, text_hi);
        }
        // Relic star
        let rx = mm_x + (ent.relic_x / gw) * mm_sz;
        let ry = mm_y + (ent.relic_y / gh) * mm_sz;
        b.rect(rx - 3.5 * ui, ry - 3.5 * ui, 7.0 * ui, 7.0 * ui, gold_col);

        // Heroes / Bots
        for p in &ent.players {
            let px = mm_x + (p.x / gw) * mm_sz;
            let py = mm_y + (p.y / gh) * mm_sz;
            let is_self = p.id == ent.your_player_id;
            let sz = if is_self { 6.5 * ui } else { 4.2 * ui };
            let col = if is_self {
                [0.35, 1.0, 0.55, 1.0]
            } else if p.team == 0 {
                west_col
            } else {
                east_col
            };
            b.rect(px - sz * 0.5, py - sz * 0.5, sz, sz, col);
        }
    }
    let mm_lbl = format!("3D REACH {}X{}X{}", world.grid_w, world.grid_h, world.grid_d);
    b.text_shadow(mm_x + 4.0 * ui, mm_y + mm_sz + 5.0 * ui, 1.25 * ui, text_dim, &mm_lbl);

    // ------------------------------------------------------------------
    // 4. TOP-LEFT NEURAL TELEMETRY + LIVE KILL/COMBO FEED
    // ------------------------------------------------------------------
    if cfg.show_telemetry {
        let tx = 14.0 * ui;
        let ty = 14.0 * ui;
        let tw = 382.0 * ui;
        let th = 114.0 * ui;
        b.rect(tx, ty, tw, th, panel_bg);
        b.rect_outline(tx, ty, tw, th, 1.5 * ui, panel_border);

        let sc = 1.45 * ui;
        if let Some(ent) = entities {
            let t = &ent.telemetry;
            let l1 = format!(
                "{} {} {}X{}X{} T:{}",
                t.mode.to_uppercase(),
                t.backend.to_uppercase(),
                t.grid_w,
                t.grid_h,
                t.grid_d,
                t.field_tick
            );
            let l2 = format!(
                "WORLD {:.2}MS ECO {:.2}MS PROJ {:.1}%",
                t.infer_ms_last, t.eco_infer_ms, t.proj_active_pct
            );
            let l3 = format!(
                "MASS_ERR:{} E_ERR:{} CHUNKS:{}/{}",
                t.mass_error, t.energy_error, t.aoi_chunks_sent, t.total_chunks
            );
            let l4 = format!(
                "FPS:{:.0} RTT:{:.0}MS WIRE:{:.0}KB/S {}",
                cfg.fps, cfg.rtt_ms, t.wire_kbps, cfg.style_label
            );
            let l5 = if t.neural_bots_on {
                format!("AI: NEURAL-Z SHARED LATENT ({:.2}MS, {} BOTS)", t.bot_infer_ms, t.active_bots)
            } else {
                format!("AI: SCRIPTED FALLBACK ({} BOTS) [N TOGGLE]", t.active_bots)
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
        }

        // Compact Kill / Shrine / Combo Feed right below telemetry
        if let Some(ent) = entities {
            let mut fy = ty + th + 10.0 * ui;
            for ev in ent
                .events
                .iter()
                .rev()
                .filter(|e| e.kind >= 1 && e.kind <= 5)
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
    let hud_w = 560.0 * ui;
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
        let cls_name = match p.class_id {
            CLASS_WARDEN => "FROST WARDEN",
            CLASS_ALCHEMIST => "PYRO ALCHEMIST",
            _ => "MARSH RANGER",
        };
        let alt_tag = if p.z >= 1.55 { "L2 HIGH" } else { "L1 GND" };
        let header = format!(
            "LV.{} {} [{}] K:{} D:{}",
            p.level, cls_name, alt_tag, p.kills, p.deaths
        );
        b.text_shadow(hx + 14.0 * ui, hy + 10.0 * ui, 1.65 * ui, gold_col, &header);

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

        // Controls hint bar at bottom
        b.text(
            bar_x,
            hy + 92.0 * ui,
            1.15 * ui,
            text_dim,
            "1/2/3 ELEM  F1-F3 CLASS  T STYLE  M SIM",
        );

        // 5-Slot Spell Hotbar (right side of hero plate)
        let slot_sz = 54.0 * ui;
        let slot_gap = 8.0 * ui;
        let sx0 = hx + 244.0 * ui;
        let sy0 = hy + 32.0 * ui;
        let elem_lbl = match cfg.selected_element % 3 {
            0 => ("FIRE", [1.0, 0.50, 0.16, 1.0]),
            1 => ("FROST", [0.42, 0.92, 1.0, 1.0]),
            _ => ("OIL", [0.78, 0.36, 0.96, 1.0]),
        };
        let e_lbl = match p.class_id {
            CLASS_WARDEN => "BRIDGE",
            CLASS_ALCHEMIST => "OILFAN",
            _ => "SPRINT",
        };
        let r_lbl = match p.class_id {
            CLASS_WARDEN => "NOVA",
            CLASS_ALCHEMIST => "METEOR",
            _ => "SURGE",
        };
        let slots: [(&str, &str, [f32; 4], f32); 5] = [
            ("LMB", elem_lbl.0, elem_lbl.1, 0.0),
            ("RMB", "MELEE", [0.96, 0.86, 0.56, 1.0], 0.0),
            ("SPC", "DASH", [0.48, 0.92, 1.0, 1.0], p.dash_cd_frac),
            ("E", e_lbl, [0.46, 0.96, 0.62, 1.0], p.ability_e_cd_frac),
            ("R", r_lbl, [1.0, 0.68, 0.24, 1.0], p.ult_r_cd_frac),
        ];

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
        _ => [0; 7],
    }
}
