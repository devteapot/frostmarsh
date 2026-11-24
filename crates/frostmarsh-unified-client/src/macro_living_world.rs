//! Wave 37: The Living Nordic Planet (`macro_living_world.rs`).
//!
//! Modular floating-origin `f64` builders for all 5 living-world pillars:
//! 1. **Carved Rivers & Cascading Mountain Waterfalls** (steep-drop whitewater cascades + plunge mist)
//! 2. **Nordic Villages, Stone Ruins & Arched River Bridges** (timber longhouses with glowing windows
//!    & chimney smoke, hearth campfires, coastal wooden jetties, ancient mossy rune arches, arched
//!    stone-and-timber bridges spanning real carved river channels)
//! 3. **Explorable Glowing Underground Cavern Grottos** (vaulted basalt archways, hanging stalactites,
//!    radiant cyan-amethyst crystal clusters, bioluminescent mushrooms & glowing subterranean pools)
//! 4. **Living Ambient Wildlife** (animated grazing Nordic elk/stag herds, wheeling coastal seabird
//!    flocks, breaching ocean whales with misty blowhole spouts, and bobbing moored Nordic longboats)
//! 5. **Dynamic Live Climate Weather Particles** (Coriolis wind-driven snow & rain streaks, meadow
//!    fireflies/pollen motes, hearth embers, and rising bioluminescent cavern spores)

use crate::macro_planet_world::{
    add_f64, cross_f64, push_contact_shadow_disc_f64, push_oriented_box_f64,
    push_oriented_pyramid_f64, scale_f64, sub_f64, tangent_basis_f64, MacroPlanetVertex,
    MAT_MACRO_BEAM, MAT_MACRO_CRYSTAL, MAT_MACRO_ENTITY, MAT_MACRO_GAS, MAT_MACRO_PROP,
};
use field_core::planet_macro::{
    dot3_f64, face_uv_to_sphere_dir_f64, face_virtual_uv_f64, normalize3_f64,
    sample_macro_planet_column, world_f64_to_camera_rel_f32,
};

/// Lightweight per-column descriptor passed from `MacroShardData` into the living-world builder.
#[derive(Clone, Copy, Debug)]
pub struct LivingShardCell {
    pub surf_z: f64,
    pub water_depth: f32,
    pub river_depth: f32,
    pub cavern_depth: f32,
    pub has_wood: bool,
    pub is_snowy: bool,
}

/// Persistent world anchors discovered per hydrated shard so per-frame animated wildlife (herds,
/// seabirds, whales, moored longboats) and cavern spores spawn organically in the exact right biomes!
#[derive(Clone, Debug, Default)]
pub struct ShardLivingAnchors {
    /// Scenic coastal meadow anchor for the Nordic village & campfire.
    pub village_dir: Option<([f64; 3], f64)>,
    /// Deep coastal bay anchors `(sphere_dir, water_depth)` for swimming & breaching ocean whales.
    pub whale_bays: Vec<([f64; 3], f64)>,
    /// Shallow turquoise beach pier anchor `(pier_dir, sea_out_dir, surf_z)` for moored longboats.
    pub pier_anchor: Option<([f64; 3], [f64; 3], f64)>,
    /// Sunny meadow pasture anchors `(sphere_dir, surf_z)` for grazing Nordic elk & caribou herds.
    pub meadow_pastures: Vec<([f64; 3], f64)>,
    /// Glowing cavern grotto centers `(sphere_dir, floor_z)` for rising bioluminescent spores.
    pub cavern_grottos: Vec<([f64; 3], f64)>,
    /// River/waterfall landmarks for circling seabirds.
    pub waterfall_sites: Vec<([f64; 3], f64)>,
}

/// Populate one hydrated `256×256` shard with waterfalls, arched river bridges, a Nordic coastal
/// frontier village (longhouses, hearth campfire, coastal pier), ancient hilltop stone ruins, and
/// explorable glowing underground cavern grottos. Returns discovered wildlife & spore anchors.
#[allow(clippy::too_many_arguments)]
pub fn populate_shard_living_world(
    face: usize,
    sx: usize,
    sy: usize,
    s: usize,
    n_virt: usize,
    r_core: f64,
    z_sea_surf: f64,
    seed: u64,
    origin_f64: [f64; 3],
    cells: &[LivingShardCell],
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
) -> ShardLivingAnchors {
    let mut anchors = ShardLivingAnchors::default();
    if cells.len() < s * s || s < 32 {
        return anchors;
    }
    let gx_base = sx * s;
    let gy_base = sy * s;

    let cell_dir = |lx: usize, ly: usize| -> [f64; 3] {
        let (u, v) = face_virtual_uv_f64(
            (gx_base + lx) as f64 + 0.5,
            (gy_base + ly) as f64 + 0.5,
            n_virt,
        );
        face_uv_to_sphere_dir_f64(face, u, v)
    };

    // -------------------------------------------------------------------------
    // 1. Cascading Mountain Waterfalls along steep carved river drops!
    // -------------------------------------------------------------------------
    let mut placed_waterfalls: Vec<(usize, usize)> = Vec::new();
    for ly in (6..s - 6).step_by(3) {
        for lx in (6..s - 6).step_by(3) {
            let c = cells[ly * s + lx];
            if c.river_depth < 0.14 || c.surf_z < z_sea_surf + 1.6 {
                continue;
            }
            // Find steepest downhill neighbor within 3m along the river valley
            let mut best_drop = 0.0f64;
            let mut best_nb = (lx, ly);
            for &(dx, dy) in &[(-3isize, 0isize), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)] {
                let nx = (lx as isize + dx) as usize;
                let ny = (ly as isize + dy) as usize;
                let nc = cells[ny * s + nx];
                let drop = c.surf_z - nc.surf_z;
                if drop > best_drop && (nc.river_depth > 0.04 || nc.water_depth > 0.05) {
                    best_drop = drop;
                    best_nb = (nx, ny);
                }
            }
            if best_drop >= 1.05 {
                let too_close = placed_waterfalls.iter().any(|&(px, py)| {
                    let dx = px as isize - lx as isize;
                    let dy = py as isize - ly as isize;
                    dx * dx + dy * dy < 144
                });
                if !too_close && placed_waterfalls.len() < 10 {
                    placed_waterfalls.push((lx, ly));
                    let d_top = cell_dir(lx, ly);
                    let d_bot = cell_dir(best_nb.0, best_nb.1);
                    let z_top = c.surf_z + c.river_depth as f64 * 0.75 + 0.25;
                    let z_bot = cells[best_nb.1 * s + best_nb.0].surf_z.max(z_sea_surf - 0.1);
                    push_mountain_waterfall(
                        terrain_verts,
                        terrain_indices,
                        gas_verts,
                        gas_indices,
                        d_top,
                        d_bot,
                        z_top,
                        z_bot,
                        r_core,
                        origin_f64,
                    );
                    anchors.waterfall_sites.push((d_top, z_top));
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 2. Arched Stone & Timber Bridges spanning carved river valleys!
    // -------------------------------------------------------------------------
    let mut placed_bridges: Vec<(usize, usize)> = Vec::new();
    for ly in (14..s - 14).step_by(6) {
        for lx in (14..s - 14).step_by(6) {
            if placed_bridges.len() >= 4 {
                break;
            }
            let c = cells[ly * s + lx];
            if c.river_depth < 0.25 || c.surf_z < z_sea_surf + 0.5 || c.cavern_depth > 0.5 {
                continue;
            }
            let too_close = placed_bridges.iter().any(|&(px, py)| {
                let dx = px as isize - lx as isize;
                let dy = py as isize - ly as isize;
                dx * dx + dy * dy < 38 * 38
            });
            if too_close {
                continue;
            }
            // Check either X-axis or Y-axis dry banks ~6..8m across the river
            let x_ok = cells[ly * s + (lx - 7)].river_depth < 0.04
                && cells[ly * s + (lx + 7)].river_depth < 0.04
                && cells[ly * s + (lx - 7)].water_depth < 0.05
                && cells[ly * s + (lx + 7)].water_depth < 0.05
                && (cells[ly * s + (lx - 7)].surf_z - cells[ly * s + (lx + 7)].surf_z).abs() < 2.2;
            let y_ok = cells[(ly - 7) * s + lx].river_depth < 0.04
                && cells[(ly + 7) * s + lx].river_depth < 0.04
                && cells[(ly - 7) * s + lx].water_depth < 0.05
                && cells[(ly + 7) * s + lx].water_depth < 0.05
                && (cells[(ly - 7) * s + lx].surf_z - cells[(ly + 7) * s + lx].surf_z).abs() < 2.2;

            if x_ok || y_ok {
                let (a_xy, b_xy) = if x_ok {
                    ((lx - 7, ly), (lx + 7, ly))
                } else {
                    ((lx, ly - 7), (lx, ly + 7))
                };
                let d_a = cell_dir(a_xy.0, a_xy.1);
                let d_b = cell_dir(b_xy.0, b_xy.1);
                let z_a = cells[a_xy.1 * s + a_xy.0].surf_z;
                let z_b = cells[b_xy.1 * s + b_xy.0].surf_z;
                let deck_z = z_a.max(z_b) + 0.85;
                push_arched_river_bridge(
                    terrain_verts,
                    terrain_indices,
                    gas_verts,
                    gas_indices,
                    d_a,
                    d_b,
                    deck_z,
                    c.surf_z,
                    r_core,
                    origin_f64,
                );
                placed_bridges.push((lx, ly));
            }
        }
    }

    // -------------------------------------------------------------------------
    // 3. Explorable Glowing Underground Cavern Grottos!
    // -------------------------------------------------------------------------
    let mut placed_caverns: Vec<(usize, usize)> = Vec::new();
    for ly in (12..s - 12).step_by(4) {
        for lx in (12..s - 12).step_by(4) {
            let c = cells[ly * s + lx];
            if c.cavern_depth < 2.4 {
                continue;
            }
            // Local maximum of cavern_depth (grotto center)
            let is_peak = [(-4isize, 0isize), (4, 0), (0, -4), (0, 4)].iter().all(|&(dx, dy)| {
                let nx = (lx as isize + dx) as usize;
                let ny = (ly as isize + dy) as usize;
                c.cavern_depth >= cells[ny * s + nx].cavern_depth - 0.05
            });
            if !is_peak {
                continue;
            }
            let too_close = placed_caverns.iter().any(|&(px, py)| {
                let dx = px as isize - lx as isize;
                let dy = py as isize - ly as isize;
                dx * dx + dy * dy < 26 * 26
            });
            if !too_close && placed_caverns.len() < 4 {
                placed_caverns.push((lx, ly));
                let dir = cell_dir(lx, ly);
                let rim_z = (c.surf_z + c.cavern_depth as f64 * 0.72).min(38.0);
                push_glowing_cavern_grotto(
                    terrain_verts,
                    terrain_indices,
                    gas_verts,
                    gas_indices,
                    dir,
                    c.surf_z,
                    rim_z,
                    r_core,
                    origin_f64,
                    seed.wrapping_add((lx * 31 + ly * 17) as u64),
                );
                anchors.cavern_grottos.push((dir, c.surf_z));
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. Nordic Coastal Frontier Village (Longhouses, Hearth Campfire & Pier)!
    // -------------------------------------------------------------------------
    let mut best_village: Option<(f64, usize, usize, (isize, isize))> = None;
    for ly in (20..s - 20).step_by(8) {
        for lx in (20..s - 20).step_by(8) {
            let c = cells[ly * s + lx];
            if c.surf_z < z_sea_surf + 1.4
                || c.surf_z > z_sea_surf + 5.8
                || c.water_depth > 0.05
                || c.river_depth > 0.05
                || c.cavern_depth > 0.2
            {
                continue;
            }
            // Check flatness across a 12m radius
            let mut max_dz = 0.0f64;
            let mut bad = false;
            for &(dx, dy) in &[(-8isize, 0isize), (8, 0), (0, -8), (0, 8), (-6, -6), (6, 6)] {
                let nc = cells[((ly as isize + dy) as usize) * s + ((lx as isize + dx) as usize)];
                if nc.water_depth > 0.1 || nc.cavern_depth > 0.4 || nc.river_depth > 0.1 {
                    bad = true;
                    break;
                }
                max_dz = max_dz.max((nc.surf_z - c.surf_z).abs());
            }
            if bad || max_dz > 2.2 {
                continue;
            }
            // Prefer scenic flat meadow within 14..32m of a coastal bay!
            let mut sea_dir = (0isize, 0isize);
            let mut sea_bonus = 0.0f64;
            for &(dx, dy) in &[(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                for dist in [14isize, 22, 30] {
                    let nx = lx as isize + dx * dist;
                    let ny = ly as isize + dy * dist;
                    if nx >= 4 && ny >= 4 && nx < (s as isize - 4) && ny < (s as isize - 4) {
                        if cells[ny as usize * s + nx as usize].water_depth > 0.45 {
                            sea_bonus = 6.0 - (dist as f64) * 0.1;
                            sea_dir = (dx, dy);
                            break;
                        }
                    }
                }
                if sea_bonus > 0.0 {
                    break;
                }
            }
            let score = (3.0 - max_dz) * 2.0 + sea_bonus + if !c.is_snowy { 1.5 } else { 0.0 };
            if best_village.map_or(true, |(bs, _, _, _)| score > bs) {
                best_village = Some((score, lx, ly, sea_dir));
            }
        }
    }

    if let Some((_, vx, vy, sea_step)) = best_village {
        let v_dir = cell_dir(vx, vy);
        let v_z = cells[vy * s + vx].surf_z;
        let v_snow = cells[vy * s + vx].is_snowy;
        anchors.village_dir = Some((v_dir, v_z));

        // Central Nordic Hearth / Campfire Pit
        let hearth_pos = scale_f64(v_dir, r_core + v_z);
        push_village_hearth(
            terrain_verts,
            terrain_indices,
            gas_verts,
            gas_indices,
            hearth_pos,
            v_dir,
            origin_f64,
        );

        // 3–4 Timber Longhouses arranged organically around the hearth square
        let house_offsets: [(isize, isize, f64, f64, f64); 4] = [
            (-10, -7, 0.4, 7.6, 4.6),
            (10, -5, 2.3, 6.8, 4.4),
            (-7, 10, -1.1, 6.4, 4.2),
            (9, 9, -2.5, 5.8, 4.0),
        ];
        for (hi, &(dx, dy, yaw, len, wid)) in house_offsets.iter().enumerate() {
            let hx = (vx as isize + dx).clamp(6, s as isize - 7) as usize;
            let hy = (vy as isize + dy).clamp(6, s as isize - 7) as usize;
            let hc = cells[hy * s + hx];
            if hc.water_depth > 0.05 || hc.river_depth > 0.08 || hc.cavern_depth > 0.4 {
                continue;
            }
            let h_dir = cell_dir(hx, hy);
            let h_pos = scale_f64(h_dir, r_core + hc.surf_z);
            push_nordic_longhouse(
                terrain_verts,
                terrain_indices,
                gas_verts,
                gas_indices,
                h_pos,
                h_dir,
                origin_f64,
                yaw,
                len,
                wid,
                3.4 + (hi % 2) as f64 * 0.55,
                hc.is_snowy || v_snow,
            );
        }

        // Coastal Wooden Jetty / Pier extending into the turquoise bay from the nearest beach!
        if sea_step != (0, 0) {
            for step_d in 6isize..38 {
                let px = vx as isize + sea_step.0 * step_d;
                let py = vy as isize + sea_step.1 * step_d;
                if px < 8 || py < 8 || px >= s as isize - 8 || py >= s as isize - 8 {
                    break;
                }
                let pc = cells[py as usize * s + px as usize];
                if pc.water_depth > 0.06 && pc.water_depth < 0.75 {
                    let p_dir = cell_dir(px as usize, py as usize);
                    let p_next = cell_dir(
                        (px + sea_step.0 * 4).clamp(2, s as isize - 3) as usize,
                        (py + sea_step.1 * 4).clamp(2, s as isize - 3) as usize,
                    );
                    let out_raw = sub_f64(p_next, p_dir);
                    let out_t = normalize3_f64(sub_f64(out_raw, scale_f64(p_dir, dot3_f64(out_raw, p_dir))));
                    push_coastal_wooden_pier(
                        terrain_verts,
                        terrain_indices,
                        p_dir,
                        out_t,
                        r_core,
                        z_sea_surf,
                        origin_f64,
                    );
                    anchors.pier_anchor = Some((p_dir, out_t, z_sea_surf));
                    break;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // 5. Ancient Mossy Stone Ruins & Standing Rune Arches on scenic hilltops!
    // -------------------------------------------------------------------------
    let mut ruins_placed: Vec<(usize, usize)> = Vec::new();
    for ly in (16..s - 16).step_by(14) {
        for lx in (16..s - 16).step_by(14) {
            if ruins_placed.len() >= 2 {
                break;
            }
            let c = cells[ly * s + lx];
            if c.surf_z < z_sea_surf + 5.0
                || c.surf_z > 35.5
                || c.water_depth > 0.05
                || c.river_depth > 0.08
                || c.cavern_depth > 0.4
            {
                continue;
            }
            if let Some((_, vx, vy, _)) = best_village {
                let d2 = (lx as isize - vx as isize).pow(2) + (ly as isize - vy as isize).pow(2);
                if d2 < 28 * 28 {
                    continue;
                }
            }
            let too_close = ruins_placed.iter().any(|&(rx, ry)| {
                (lx as isize - rx as isize).pow(2) + (ly as isize - ry as isize).pow(2) < 48 * 48
            });
            if too_close {
                continue;
            }
            let r_dir = cell_dir(lx, ly);
            let r_pos = scale_f64(r_dir, r_core + c.surf_z);
            push_ancient_stone_ruins(
                terrain_verts,
                terrain_indices,
                gas_verts,
                gas_indices,
                r_pos,
                r_dir,
                origin_f64,
                ((lx * 13 + ly * 29) as f64) * 0.4,
            );
            ruins_placed.push((lx, ly));
        }
    }

    // -------------------------------------------------------------------------
    // 6. Discover Meadow Pastures (Elk Herds) & Deep Coastal Bays (Whales)!
    // -------------------------------------------------------------------------
    for ly in (18..s - 18).step_by(18) {
        for lx in (18..s - 18).step_by(18) {
            let c = cells[ly * s + lx];
            if c.water_depth > 2.2 && anchors.whale_bays.len() < 4 {
                anchors.whale_bays.push((cell_dir(lx, ly), c.water_depth as f64));
            } else if c.surf_z > z_sea_surf + 1.5
                && c.surf_z < z_sea_surf + 8.5
                && c.water_depth < 0.05
                && c.river_depth < 0.05
                && c.cavern_depth < 0.3
                && !c.has_wood
                && anchors.meadow_pastures.len() < 4
            {
                anchors.meadow_pastures.push((cell_dir(lx, ly), c.surf_z));
            }
        }
    }

    anchors
}

// =============================================================================
// Per-Frame Animated Living Wildlife & Live Climate Weather Particles
// =============================================================================

/// Emit animated Nordic elk/stag herds, wheeling coastal seabird flocks, breaching ocean whales,
/// bobbing moored Nordic longboats, and live `PlanetaryClimateNet` weather particles (snow, rain,
/// meadow fireflies, hearth embers, and rising cavern spores)!
#[allow(clippy::too_many_arguments)]
pub fn emit_living_wildlife_and_weather(
    anchors: &[&ShardLivingAnchors],
    focus_pos_f64: [f64; 3],
    r_core: f64,
    z_sea_surf: f64,
    seed: u64,
    origin_f64: [f64; 3],
    sim_time: f32,
    local_wind_u: f32,
    local_wind_v: f32,
    local_temp_c: f32,
    local_cloud_cover: f32,
    solid_verts: &mut Vec<MacroPlanetVertex>,
    solid_indices: &mut Vec<u32>,
    fx_verts: &mut Vec<MacroPlanetVertex>,
    fx_indices: &mut Vec<u32>,
) {
    let t = sim_time as f64;
    let r_surf = r_core + z_sea_surf;
    let focus_dir = normalize3_f64(focus_pos_f64);

    // 1. Grazing Nordic Elk / Stag & Caribou Herds roaming sunny meadow pastures!
    let mut elk_emitted = 0usize;
    for anc in anchors {
        for (pi, &(p_dir, base_z)) in anc.meadow_pastures.iter().enumerate() {
            if elk_emitted >= 10 {
                break;
            }
            let (t1, t2) = tangent_basis_f64(p_dir);
            let herd_sz = 3usize.min(10 - elk_emitted);
            for ei in 0..herd_sz {
                let phase = (pi as f64) * 2.1 + (ei as f64) * 1.65;
                let roam_ang = t * 0.14 + phase;
                let roam_r = 4.5 + (ei as f64) * 2.6;
                let ox = roam_ang.cos() * roam_r;
                let oy = (roam_ang * 0.72 + phase).sin() * roam_r * 0.8;
                let vx = -roam_ang.sin() * roam_r * 0.14;
                let vy = (roam_ang * 0.72 + phase).cos() * roam_r * 0.8 * 0.10;
                let e_dir = normalize3_f64(add_f64(
                    p_dir,
                    add_f64(scale_f64(t1, ox / r_surf), scale_f64(t2, oy / r_surf)),
                ));
                let col = sample_macro_planet_column(e_dir, seed);
                let ez = if col.water_depth > 0.05 { base_z } else { col.surf_z };
                let e_pos = scale_f64(e_dir, r_core + ez);
                let fwd_raw = add_f64(scale_f64(t1, vx), scale_f64(t2, vy));
                let fwd = normalize3_f64(sub_f64(fwd_raw, scale_f64(e_dir, dot3_f64(fwd_raw, e_dir))));
                let right = normalize3_f64(cross_f64(fwd, e_dir));
                let is_stag = ei == 0;
                push_nordic_elk(
                    solid_verts,
                    solid_indices,
                    fx_verts,
                    fx_indices,
                    e_pos,
                    e_dir,
                    right,
                    fwd,
                    origin_f64,
                    t * 4.2 + phase,
                    is_stag,
                );
                elk_emitted += 1;
            }
        }
    }

    // 2. Breaching & Swimming Ocean Whales in deep coastal bays!
    let mut whales_emitted = 0usize;
    for anc in anchors {
        for (wi, &(b_dir, _wd)) in anc.whale_bays.iter().enumerate() {
            if whales_emitted >= 3 {
                break;
            }
            let (t1, t2) = tangent_basis_f64(b_dir);
            let phase = (wi as f64) * 2.4 + 0.7;
            let swim_ang = t * 0.16 + phase;
            let orbit_r = 14.0 + (wi as f64) * 5.5;
            let wx = swim_ang.cos() * orbit_r;
            let wy = swim_ang.sin() * orbit_r * 0.75;
            let w_dir = normalize3_f64(add_f64(
                b_dir,
                add_f64(scale_f64(t1, wx / r_surf), scale_f64(t2, wy / r_surf)),
            ));
            let fwd_raw = add_f64(
                scale_f64(t1, -swim_ang.sin() * orbit_r),
                scale_f64(t2, swim_ang.cos() * orbit_r * 0.75),
            );
            let fwd = normalize3_f64(sub_f64(fwd_raw, scale_f64(w_dir, dot3_f64(fwd_raw, w_dir))));
            let right = normalize3_f64(cross_f64(fwd, w_dir));

            // Graceful rhythmic whale breach & dive cycle every ~14s
            let breach_wave = (t * 0.45 + phase).sin();
            let breach_h = if breach_wave > 0.55 {
                let bt = ((breach_wave - 0.55) / 0.45).clamp(0.0, 1.0);
                bt * (1.0 - bt * 0.35) * 3.8
            } else {
                -0.35 + 0.32 * (t * 1.1 + phase).sin()
            };
            let pitch = if breach_wave > 0.55 {
                ((t * 0.45 + phase).cos() * 0.42).clamp(-0.45, 0.45)
            } else {
                0.06 * (t * 1.1 + phase).cos()
            };
            let w_pos = scale_f64(w_dir, r_core + z_sea_surf + breach_h);
            push_ocean_whale(
                solid_verts,
                solid_indices,
                fx_verts,
                fx_indices,
                w_pos,
                w_dir,
                right,
                fwd,
                origin_f64,
                pitch,
                t * 3.2 + phase,
                breach_wave > 0.62,
            );
            whales_emitted += 1;
        }
    }

    // 3. Moored Bobbing Nordic Longboats beside the coastal village pier!
    for anc in anchors {
        if let Some((p_dir, out_t, sea_z)) = anc.pier_anchor {
            let side_t = normalize3_f64(cross_f64(p_dir, out_t));
            for bi in 0..2usize {
                let side_s = if bi == 0 { 3.2 } else { -3.5 };
                let out_s = 7.5 + bi as f64 * 4.2;
                let b_dir = normalize3_f64(add_f64(
                    p_dir,
                    add_f64(scale_f64(out_t, out_s / r_surf), scale_f64(side_t, side_s / r_surf)),
                ));
                let bob = (t * 1.9 + bi as f64 * 1.7).sin() * 0.16;
                let roll = (t * 1.5 + bi as f64 * 2.2).cos() * 0.10;
                let b_up = normalize3_f64(add_f64(b_dir, scale_f64(side_t, roll)));
                let b_right = normalize3_f64(cross_f64(out_t, b_up));
                let b_fwd = normalize3_f64(cross_f64(b_up, b_right));
                let b_pos = scale_f64(b_dir, r_core + sea_z + 0.18 + bob);
                push_nordic_longboat(
                    solid_verts,
                    solid_indices,
                    b_pos,
                    b_up,
                    b_right,
                    b_fwd,
                    origin_f64,
                );
            }
            break;
        }
    }

    // 4. Wheeling Flocks of Coastal Seabirds & Alpine Eagles soaring above bay & meadows!
    let (ft1, ft2) = tangent_basis_f64(focus_dir);
    for bi in 0..20usize {
        let f_i = bi as f64;
        let flock_id = bi / 5;
        let center_ang = flock_id as f64 * 1.57 + 0.35;
        let center_dist = 24.0 + flock_id as f64 * 18.0;
        let orbit_r = 11.0 + (bi % 5) as f64 * 3.8;
        let speed = 0.42 + (bi % 4) as f64 * 0.08;
        let ang = t * speed + f_i * 1.37;
        let bx = center_ang.cos() * center_dist + ang.cos() * orbit_r;
        let by = center_ang.sin() * center_dist + ang.sin() * orbit_r;
        let b_dir = normalize3_f64(add_f64(
            focus_dir,
            add_f64(scale_f64(ft1, bx / r_surf), scale_f64(ft2, by / r_surf)),
        ));
        let alt = z_sea_surf + 14.0 + (bi % 5) as f64 * 4.2 + (t * 0.9 + f_i).sin() * 1.8;
        let b_pos = scale_f64(b_dir, r_core + alt);
        let fwd_raw = add_f64(scale_f64(ft1, -ang.sin()), scale_f64(ft2, ang.cos()));
        let fwd = normalize3_f64(sub_f64(fwd_raw, scale_f64(b_dir, dot3_f64(fwd_raw, b_dir))));
        let right = normalize3_f64(cross_f64(fwd, b_dir));
        let flap = (t * 8.5 + f_i * 2.1).sin() * 0.52;
        let is_eagle = flock_id == 3;
        push_soaring_bird(
            solid_verts,
            solid_indices,
            b_pos,
            b_dir,
            right,
            fwd,
            origin_f64,
            flap,
            is_eagle,
        );
    }

    // 5. Dynamic Live Climate Weather Particles & Ambient Motes around the camera/hero!
    //    Driven by live PlanetaryClimateNet (temp_c, cloud_cover, wind_u, wind_v)!
    let is_snow_weather = local_temp_c < 7.5;
    let is_rain_weather = !is_snow_weather && local_cloud_cover > 0.30;
    let wu = (local_wind_u as f64).clamp(-6.0, 6.0);
    let wv = (local_wind_v as f64).clamp(-6.0, 6.0);
    let focus_alt = dot3_f64(focus_pos_f64, focus_pos_f64).sqrt() - r_core;

    // 5A. Live Precipitation (Snowflakes or Slanted Rain) + Golden Meadow Fireflies & Pollen
    let weather_count = 240usize;
    for i in 0..weather_count {
        let h1 = ((i.wrapping_mul(2_654_435_761)) ^ (seed as usize)) & 0xFFFF;
        let h2 = ((i.wrapping_mul(1_597_334_677)) ^ (seed as usize >> 8)) & 0xFFFF;
        let h3 = ((i.wrapping_mul(3_266_489_917)) ^ (seed as usize >> 16)) & 0xFFFF;
        let rx0 = (h1 as f64 / 65535.0) - 0.5;
        let ry0 = (h2 as f64 / 65535.0) - 0.5;
        let rz0 = h3 as f64 / 65535.0;

        // Mix 55% atmospheric precipitation/snowflakes + 45% golden-emerald meadow fireflies/pollen!
        let is_firefly = (i % 9) < 4;
        let span_xy = if is_firefly { 58.0 } else { 76.0 };
        let span_z = if is_firefly { 9.5 } else { 28.0 };

        let (px_l, py_l, pz_l, w_sz, h_sz, col, em) = if is_firefly {
            let drift_x = ((rx0 * span_xy + (t * 0.45 + i as f64).sin() * 2.2) + span_xy * 0.5).rem_euclid(span_xy) - span_xy * 0.5;
            let drift_y = ((ry0 * span_xy + (t * 0.38 + i as f64 * 1.3).cos() * 2.2) + span_xy * 0.5).rem_euclid(span_xy) - span_xy * 0.5;
            let bob_z = 0.8 + rz0 * span_z + (t * 1.6 + i as f64 * 0.9).sin() * 0.65;
            let pulse = (0.55 + 0.45 * (t * 3.8 + i as f64 * 2.1).sin()) as f32;
            let warm_gold = i % 2 == 0;
            let c = if warm_gold {
                [1.00, 0.88, 0.36, 0.72 * pulse]
            } else {
                [0.45, 0.98, 0.72, 0.64 * pulse]
            };
            let e = [c[0] * 2.2 * pulse, c[1] * 2.2 * pulse, c[2] * 2.2 * pulse, MAT_MACRO_BEAM];
            (drift_x, drift_y, bob_z, 0.072, 0.072, c, e)
        } else if is_rain_weather {
            let fall_spd = 17.5 + (i % 5) as f64 * 1.4;
            let drift_x = ((rx0 * span_xy + t * (wu * 1.2 + 1.4)) + span_xy * 0.5).rem_euclid(span_xy) - span_xy * 0.5;
            let drift_y = ((ry0 * span_xy + t * (wv * 1.2 - 0.8)) + span_xy * 0.5).rem_euclid(span_xy) - span_xy * 0.5;
            let z_cur = ((rz0 * span_z - t * fall_spd).rem_euclid(span_z)) + 0.5;
            let c = [0.70, 0.86, 1.00, 0.44];
            let e = [0.22, 0.36, 0.52, MAT_MACRO_BEAM];
            (drift_x, drift_y, z_cur, 0.032, 0.82, c, e)
        } else {
            // Crisp drifting alpine snowflakes (light flurries even on cool clear Nordic days!)
            let fall_spd = if is_snow_weather { 3.2 } else { 1.9 } + (i % 5) as f64 * 0.35;
            let sway = (t * 1.4 + i as f64 * 0.7).sin() * 1.6;
            let drift_x = ((rx0 * span_xy + t * (wu * 0.85 + 0.9) + sway) + span_xy * 0.5).rem_euclid(span_xy) - span_xy * 0.5;
            let drift_y = ((ry0 * span_xy + t * (wv * 0.85 - 0.6)) + span_xy * 0.5).rem_euclid(span_xy) - span_xy * 0.5;
            let z_cur = ((rz0 * span_z - t * fall_spd).rem_euclid(span_z)) + 0.6;
            let alpha = if is_snow_weather { 0.78 } else { 0.52 };
            let c = [0.94, 0.98, 1.00, alpha];
            let e = [0.55, 0.68, 0.85, MAT_MACRO_BEAM];
            (drift_x, drift_y, z_cur, 0.068, 0.068, c, e)
        };

        let p_dir = normalize3_f64(add_f64(
            focus_dir,
            add_f64(scale_f64(ft1, px_l / r_surf), scale_f64(ft2, py_l / r_surf)),
        ));
        let p_pos = scale_f64(p_dir, r_core + focus_alt.max(z_sea_surf) + pz_l);
        push_cross_particle_f64(
            fx_verts,
            fx_indices,
            p_pos,
            p_dir,
            ft1,
            ft2,
            w_sz,
            h_sz,
            origin_f64,
            col,
            em,
        );
    }

    // 5B. Rising Bioluminescent Cavern Spores drifting out of glowing cavern grottos!
    for anc in anchors {
        for (gi, &(g_dir, g_floor)) in anc.cavern_grottos.iter().enumerate() {
            let (gt1, gt2) = tangent_basis_f64(g_dir);
            for si in 0..16usize {
                let ang = (si as f64) * 1.9 + t * 0.35 + gi as f64;
                let r_sp = 1.5 + ((si % 5) as f64) * 1.4;
                let z_rise = ((si as f64 * 0.85 + t * 1.35).rem_euclid(12.0)) + 0.8;
                let s_dir = normalize3_f64(add_f64(
                    g_dir,
                    add_f64(
                        scale_f64(gt1, (ang.cos() * r_sp) / r_surf),
                        scale_f64(gt2, (ang.sin() * r_sp) / r_surf),
                    ),
                ));
                let s_pos = scale_f64(s_dir, r_core + g_floor + z_rise);
                let pulse = (0.55 + 0.45 * (t * 3.2 + si as f64).sin()) as f32;
                let is_amethyst = si % 3 == 0;
                let (c, e) = if is_amethyst {
                    (
                        [0.72, 0.36, 1.00, 0.78 * pulse],
                        [1.45 * pulse, 0.65 * pulse, 2.60 * pulse, MAT_MACRO_BEAM],
                    )
                } else {
                    (
                        [0.18, 0.96, 0.88, 0.80 * pulse],
                        [0.35 * pulse, 2.20 * pulse, 2.40 * pulse, MAT_MACRO_BEAM],
                    )
                };
                push_cross_particle_f64(
                    fx_verts,
                    fx_indices,
                    s_pos,
                    s_dir,
                    gt1,
                    gt2,
                    0.20,
                    0.20,
                    origin_f64,
                    c,
                    e,
                );
            }
        }
    }
}

// =============================================================================
// Architectural, Geological & Wildlife Sculpting Helpers (`f64` Floating Origin)
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn push_mountain_waterfall(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
    d_top: [f64; 3],
    d_bot: [f64; 3],
    z_top: f64,
    z_bot: f64,
    r_core: f64,
    origin_f64: [f64; 3],
) {
    let dir_mid = normalize3_f64(add_f64(d_top, d_bot));
    let flow = normalize3_f64(sub_f64(d_bot, d_top));
    let side = normalize3_f64(cross_f64(dir_mid, flow));
    let segs = 6usize;
    let cols = 3usize;
    let base = terrain_verts.len() as u32;
    let stride = cols + 1;

    for ri in 0..=segs {
        let s = ri as f64 / segs as f64;
        let d_cur = normalize3_f64([
            d_top[0] + (d_bot[0] - d_top[0]) * s,
            d_top[1] + (d_bot[1] - d_top[1]) * s,
            d_top[2] + (d_bot[2] - d_top[2]) * s,
        ]);
        let z_cur = z_top + (z_bot - z_top) * (s * s * 0.55 + s * 0.45) + (s * std::f64::consts::PI).sin() * 0.35;
        let half_w = 1.35 + 0.85 * s;
        for ci in 0..=cols {
            let u_w = (ci as f64 / cols as f64) * 2.0 - 1.0;
            let p_c = add_f64(scale_f64(d_cur, r_core + z_cur), scale_f64(side, u_w * half_w));
            let foam = (0.45 + 0.45 * s as f32 + 0.15 * (u_w * 5.0).sin().abs() as f32).clamp(0.0, 1.0);
            terrain_verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p_c, origin_f64),
                normal: [dir_mid[0] as f32, dir_mid[1] as f32, dir_mid[2] as f32],
                color: [0.30 + 0.62 * foam, 0.72 + 0.24 * foam, 0.92 + 0.08 * foam, 0.12],
                emissive_mat: [0.08 * foam, 0.24 * foam, 0.36 * foam, MAT_MACRO_PROP],
                water_ice: [0.0, 0.0, 0.0, 1.0],
            });
        }
    }
    for ri in 0..segs {
        for ci in 0..cols {
            let i00 = base + (ri * stride + ci) as u32;
            let i10 = i00 + 1;
            let i01 = base + ((ri + 1) * stride + ci) as u32;
            let i11 = i01 + 1;
            terrain_indices.extend_from_slice(&[i00, i10, i11, i00, i11, i01, i00, i11, i10, i00, i01, i11]);
        }
    }

    // Frothy white plunge-pool mist puff at the base of the waterfall!
    let mist_pos = scale_f64(d_bot, r_core + z_bot + 1.1);
    push_cross_particle_f64(
        gas_verts,
        gas_indices,
        mist_pos,
        d_bot,
        flow,
        side,
        2.6,
        2.1,
        origin_f64,
        [0.86, 0.95, 1.00, 0.34],
        [0.08, 0.14, 0.22, MAT_MACRO_GAS],
    );
}

#[allow(clippy::too_many_arguments)]
fn push_arched_river_bridge(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
    d_a: [f64; 3],
    d_b: [f64; 3],
    deck_z: f64,
    bed_z: f64,
    r_core: f64,
    origin_f64: [f64; 3],
) {
    let up = normalize3_f64(add_f64(d_a, d_b));
    let span_raw = sub_f64(scale_f64(d_b, r_core + deck_z), scale_f64(d_a, r_core + deck_z));
    let span_len = dot3_f64(span_raw, span_raw).sqrt().clamp(8.0, 22.0);
    let fwd = normalize3_f64(sub_f64(span_raw, scale_f64(up, dot3_f64(span_raw, up))));
    let right = normalize3_f64(cross_f64(fwd, up));
    let mid_p = scale_f64(up, r_core + deck_z);

    // Arched timber-and-stone segmented deck spanning bank to bank
    let segs = 7usize;
    let seg_len = span_len / segs as f64 + 0.35;
    let bridge_w = 3.4;
    for si in 0..segs {
        let t = (si as f64 + 0.5) / segs as f64;
        let arch = (t * std::f64::consts::PI).sin() * 1.45;
        let off_f = (t - 0.5) * span_len;
        let seg_c = add_f64(
            add_f64(mid_p, scale_f64(fwd, off_f)),
            scale_f64(up, arch),
        );
        // Stone arch understructure
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            sub_f64(seg_c, scale_f64(up, 0.45)),
            right,
            up,
            fwd,
            [bridge_w + 0.3, 0.72, seg_len],
            origin_f64,
            [0.34, 0.35, 0.39, 0.78],
            [0.0, 0.0, 0.0, MAT_MACRO_PROP],
        );
        // Warm weathered timber plank deck
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            seg_c,
            right,
            up,
            fwd,
            [bridge_w, 0.24, seg_len],
            origin_f64,
            [0.32, 0.21, 0.12, 0.86],
            [0.0, 0.0, 0.0, MAT_MACRO_PROP],
        );
        // Timber side railings + posts
        for side_s in [-1.0f64, 1.0] {
            let rail_p = add_f64(
                add_f64(seg_c, scale_f64(right, side_s * (bridge_w * 0.48))),
                scale_f64(up, 0.62),
            );
            push_oriented_box_f64(
                terrain_verts,
                terrain_indices,
                rail_p,
                right,
                up,
                fwd,
                [0.20, 0.18, seg_len],
                origin_f64,
                [0.26, 0.17, 0.10, 0.88],
                [0.0, 0.0, 0.0, MAT_MACRO_PROP],
            );
        }
    }

    // Two carved stone abutment piers + glowing warm bridge-entry lanterns on both banks!
    let pier_h = (deck_z - bed_z + 1.8).clamp(2.4, 7.5);
    for end_s in [-1.0f64, 1.0] {
        let end_p = add_f64(mid_p, scale_f64(fwd, end_s * span_len * 0.46));
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            sub_f64(end_p, scale_f64(up, pier_h * 0.42)),
            right,
            up,
            fwd,
            [bridge_w + 0.8, pier_h, 2.1],
            origin_f64,
            [0.30, 0.32, 0.37, 0.78],
            [0.0, 0.0, 0.0, MAT_MACRO_PROP],
        );
        for side_s in [-1.0f64, 1.0] {
            let post_p = add_f64(end_p, scale_f64(right, side_s * (bridge_w * 0.52)));
            push_oriented_box_f64(
                terrain_verts,
                terrain_indices,
                add_f64(post_p, scale_f64(up, 1.05)),
                right,
                up,
                fwd,
                [0.38, 2.1, 0.38],
                origin_f64,
                [0.28, 0.30, 0.35, 0.76],
                [0.0, 0.0, 0.0, MAT_MACRO_PROP],
            );
            push_oriented_box_f64(
                terrain_verts,
                terrain_indices,
                add_f64(post_p, scale_f64(up, 2.25)),
                right,
                up,
                fwd,
                [0.44, 0.46, 0.44],
                origin_f64,
                [1.00, 0.78, 0.34, 0.16],
                [2.4, 1.25, 0.30, MAT_MACRO_BEAM],
            );
        }
    }
    push_contact_shadow_disc_f64(
        gas_verts,
        gas_indices,
        scale_f64(up, r_core + bed_z + 0.18),
        up,
        3.2,
        10,
        origin_f64,
        0.26,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_nordic_longhouse(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up: [f64; 3],
    origin_f64: [f64; 3],
    yaw: f64,
    length: f64,
    width: f64,
    wall_h: f64,
    snowy_roof: bool,
) {
    let (t1, t2) = tangent_basis_f64(up);
    let fwd = normalize3_f64(add_f64(scale_f64(t1, yaw.cos()), scale_f64(t2, yaw.sin())));
    let right = normalize3_f64(cross_f64(fwd, up));

    push_contact_shadow_disc_f64(
        gas_verts,
        gas_indices,
        add_f64(base_world, scale_f64(up, 0.12)),
        up,
        length.max(width) * 0.72,
        10,
        origin_f64,
        0.34,
    );

    // Carved stone foundation plinth
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        add_f64(base_world, scale_f64(up, 0.25)),
        right,
        up,
        fwd,
        [width + 0.6, 1.1, length + 0.6],
        origin_f64,
        [0.30, 0.32, 0.36, 0.80],
        [0.0, 0.0, 0.0, MAT_MACRO_PROP],
    );
    // Dark weathered Nordic timber log walls
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        add_f64(base_world, scale_f64(up, 0.75 + wall_h * 0.5)),
        right,
        up,
        fwd,
        [width, wall_h, length],
        origin_f64,
        [0.24, 0.15, 0.09, 0.88],
        [0.0, 0.0, 0.0, MAT_MACRO_PROP],
    );
    // Warm glowing amber leaded windows on both long sides & front gable!
    for side_s in [-1.0f64, 1.0] {
        for &z_off in &[-length * 0.24, length * 0.24] {
            let win_p = add_f64(
                add_f64(base_world, scale_f64(up, 0.9 + wall_h * 0.48)),
                add_f64(scale_f64(right, side_s * (width * 0.505)), scale_f64(fwd, z_off)),
            );
            push_oriented_box_f64(
                terrain_verts,
                terrain_indices,
                win_p,
                right,
                up,
                fwd,
                [0.14, 0.82, 0.92],
                origin_f64,
                [1.00, 0.78, 0.32, 0.16],
                [2.5, 1.35, 0.30, MAT_MACRO_BEAM],
            );
        }
    }
    // Warm timber door on front gable
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        add_f64(
            add_f64(base_world, scale_f64(up, 1.65)),
            scale_f64(fwd, length * 0.505),
        ),
        right,
        up,
        fwd,
        [1.15, 1.85, 0.16],
        origin_f64,
        [0.34, 0.22, 0.12, 0.84],
        [0.25, 0.12, 0.03, MAT_MACRO_PROP],
    );

    // Steep pitched Nordic slate roof (3 stepped gabled tiers with optional snow-dusted eaves!)
    let roof_col = if snowy_roof {
        [0.80, 0.86, 0.93, 0.34]
    } else {
        [0.16, 0.20, 0.26, 0.68]
    };
    let roof_base_z = 0.72 + wall_h;
    for (ti, &(frac_w, h_step, z_step)) in [(1.18f64, 0.95f64, 0.42f64), (0.82, 0.95, 1.25), (0.44, 0.85, 2.02)].iter().enumerate() {
        let tier_col = if ti == 2 && snowy_roof {
            [0.88, 0.93, 0.98, 0.28]
        } else {
            roof_col
        };
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            add_f64(base_world, scale_f64(up, roof_base_z + z_step)),
            right,
            up,
            fwd,
            [width * frac_w, h_step, length + 0.85 - ti as f64 * 0.12],
            origin_f64,
            tier_col,
            [0.0, 0.0, 0.0, MAT_MACRO_PROP],
        );
    }
    // Stone chimney + gentle translucent hearth smoke wisp!
    let chim_p = add_f64(
        add_f64(base_world, scale_f64(up, roof_base_z + 2.2)),
        add_f64(scale_f64(right, width * 0.18), scale_f64(fwd, -length * 0.22)),
    );
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        chim_p,
        right,
        up,
        fwd,
        [0.72, 2.2, 0.72],
        origin_f64,
        [0.32, 0.33, 0.36, 0.80],
        [0.0, 0.0, 0.0, MAT_MACRO_PROP],
    );
    push_cross_particle_f64(
        gas_verts,
        gas_indices,
        add_f64(chim_p, scale_f64(up, 2.3)),
        up,
        right,
        fwd,
        1.45,
        2.2,
        origin_f64,
        [0.78, 0.82, 0.88, 0.26],
        [0.04, 0.04, 0.05, MAT_MACRO_GAS],
    );
}

#[allow(clippy::too_many_arguments)]
fn push_village_hearth(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up: [f64; 3],
    origin_f64: [f64; 3],
) {
    let (t1, t2) = tangent_basis_f64(up);
    // Stone fire-ring
    for i in 0..8usize {
        let ang = (i as f64 / 8.0) * std::f64::consts::TAU;
        let sp = add_f64(
            add_f64(base_world, scale_f64(up, 0.28)),
            add_f64(scale_f64(t1, ang.cos() * 1.35), scale_f64(t2, ang.sin() * 1.35)),
        );
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            sp,
            t1,
            up,
            t2,
            [0.62, 0.52, 0.62],
            origin_f64,
            [0.32, 0.33, 0.36, 0.82],
            [0.0, 0.0, 0.0, MAT_MACRO_PROP],
        );
    }
    // Glowing hearth fire core + rising warm smoke puff
    push_oriented_pyramid_f64(
        terrain_verts,
        terrain_indices,
        add_f64(base_world, scale_f64(up, 0.25)),
        t1,
        up,
        t2,
        0.78,
        1.75,
        origin_f64,
        [1.00, 0.62, 0.16, 0.14],
        [3.4, 1.55, 0.24, MAT_MACRO_BEAM],
    );
    push_cross_particle_f64(
        gas_verts,
        gas_indices,
        add_f64(base_world, scale_f64(up, 2.8)),
        up,
        t1,
        t2,
        1.6,
        2.5,
        origin_f64,
        [0.82, 0.78, 0.74, 0.24],
        [0.35, 0.14, 0.03, MAT_MACRO_GAS],
    );
}

#[allow(clippy::too_many_arguments)]
fn push_coastal_wooden_pier(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    start_dir: [f64; 3],
    out_dir: [f64; 3],
    r_core: f64,
    z_sea_surf: f64,
    origin_f64: [f64; 3],
) {
    let side = normalize3_f64(cross_f64(start_dir, out_dir));
    let pier_len = 13.5;
    let pier_w = 2.8;
    let deck_z = z_sea_surf + 0.85;
    let start_p = scale_f64(start_dir, r_core + deck_z);
    let mid_p = add_f64(start_p, scale_f64(out_dir, pier_len * 0.5));

    // Weathered timber jetty deck
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        mid_p,
        side,
        start_dir,
        out_dir,
        [pier_w, 0.24, pier_len],
        origin_f64,
        [0.31, 0.21, 0.12, 0.86],
        [0.0, 0.0, 0.0, MAT_MACRO_PROP],
    );
    // Support pilings & end lantern post
    for pi in 0..5usize {
        let d = (pi as f64 / 4.0) * pier_len;
        for s_sign in [-1.0f64, 1.0] {
            let p_top = add_f64(
                add_f64(start_p, scale_f64(out_dir, d)),
                scale_f64(side, s_sign * (pier_w * 0.46)),
            );
            let h = if pi == 4 { 3.8 } else { 2.8 };
            push_oriented_box_f64(
                terrain_verts,
                terrain_indices,
                sub_f64(p_top, scale_f64(start_dir, if pi == 4 { 0.4 } else { 1.1 })),
                side,
                start_dir,
                out_dir,
                [0.36, h, 0.36],
                origin_f64,
                [0.24, 0.16, 0.09, 0.88],
                [0.0, 0.0, 0.0, MAT_MACRO_PROP],
            );
            if pi == 4 && s_sign > 0.0 {
                push_oriented_box_f64(
                    terrain_verts,
                    terrain_indices,
                    add_f64(p_top, scale_f64(start_dir, 1.65)),
                    side,
                    start_dir,
                    out_dir,
                    [0.45, 0.48, 0.45],
                    origin_f64,
                    [1.00, 0.80, 0.36, 0.15],
                    [2.6, 1.35, 0.32, MAT_MACRO_BEAM],
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_ancient_stone_ruins(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
    base_world: [f64; 3],
    up: [f64; 3],
    origin_f64: [f64; 3],
    yaw: f64,
) {
    let (t1, t2) = tangent_basis_f64(up);
    let fwd = normalize3_f64(add_f64(scale_f64(t1, yaw.cos()), scale_f64(t2, yaw.sin())));
    let right = normalize3_f64(cross_f64(fwd, up));

    push_contact_shadow_disc_f64(
        gas_verts,
        gas_indices,
        add_f64(base_world, scale_f64(up, 0.12)),
        up,
        6.2,
        12,
        origin_f64,
        0.30,
    );

    // Ancient standing Nordic Rune Archway (2 mossy carved granite monoliths + stone lintel!)
    for s_sign in [-1.0f64, 1.0] {
        let p_c = add_f64(
            add_f64(base_world, scale_f64(right, s_sign * 2.5)),
            scale_f64(up, 2.7),
        );
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            p_c,
            right,
            up,
            fwd,
            [1.15, 5.8, 1.15],
            origin_f64,
            [0.32, 0.36, 0.38, 0.76],
            [0.06, 0.24, 0.36, MAT_MACRO_PROP],
        );
    }
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        add_f64(base_world, scale_f64(up, 5.85)),
        right,
        up,
        fwd,
        [6.8, 1.05, 1.35],
        origin_f64,
        [0.30, 0.35, 0.37, 0.76],
        [0.08, 0.28, 0.42, MAT_MACRO_PROP],
    );

    // 4 broken surrounding granite colonnade pillars + 1 fallen mossy lintel
    for ci in 0..4usize {
        let ang = (ci as f64 / 4.0) * std::f64::consts::TAU + 0.65;
        let cp = add_f64(
            base_world,
            add_f64(scale_f64(right, ang.cos() * 5.2), scale_f64(fwd, ang.sin() * 5.2)),
        );
        let h = 1.8 + (ci % 3) as f64 * 1.15;
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            add_f64(cp, scale_f64(up, h * 0.5 - 0.2)),
            right,
            up,
            fwd,
            [0.95, h, 0.95],
            origin_f64,
            [0.30, 0.34, 0.35, 0.78],
            [0.0, 0.0, 0.0, MAT_MACRO_PROP],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_glowing_cavern_grotto(
    terrain_verts: &mut Vec<MacroPlanetVertex>,
    terrain_indices: &mut Vec<u32>,
    gas_verts: &mut Vec<MacroPlanetVertex>,
    gas_indices: &mut Vec<u32>,
    dir: [f64; 3],
    floor_z: f64,
    rim_z: f64,
    r_core: f64,
    origin_f64: [f64; 3],
    seed: u64,
) {
    let (t1, t2) = tangent_basis_f64(dir);
    let floor_pos = scale_f64(dir, r_core + floor_z);
    let arch_h = (rim_z - floor_z).clamp(4.0, 10.0);

    // Vaulted dark basalt cavern archways framing the grotto entrance!
    for ai in 0..2usize {
        let ang = (ai as f64) * 1.15 + (seed % 7) as f64 * 0.4;
        let a_right = normalize3_f64(add_f64(scale_f64(t1, ang.cos()), scale_f64(t2, ang.sin())));
        let a_fwd = normalize3_f64(cross_f64(dir, a_right));
        for side_s in [-1.0f64, 1.0] {
            let pil_p = add_f64(
                add_f64(floor_pos, scale_f64(a_right, side_s * 5.2)),
                scale_f64(dir, arch_h * 0.52),
            );
            push_oriented_box_f64(
                terrain_verts,
                terrain_indices,
                pil_p,
                a_right,
                dir,
                a_fwd,
                [2.2, arch_h * 1.05, 2.4],
                origin_f64,
                [0.14, 0.16, 0.21, 0.78],
                [0.04, 0.16, 0.24, MAT_MACRO_PROP],
            );
        }
        // Vaulted basalt crown + hanging stalactites!
        let crown_p = add_f64(floor_pos, scale_f64(dir, arch_h + 0.6));
        push_oriented_box_f64(
            terrain_verts,
            terrain_indices,
            crown_p,
            a_right,
            dir,
            a_fwd,
            [11.8, 2.1, 3.0],
            origin_f64,
            [0.16, 0.18, 0.23, 0.80],
            [0.03, 0.12, 0.18, MAT_MACRO_PROP],
        );
        for si in -2isize..=2 {
            let st_p = add_f64(
                sub_f64(crown_p, scale_f64(dir, 1.0)),
                scale_f64(a_right, si as f64 * 1.9),
            );
            push_oriented_pyramid_f64(
                terrain_verts,
                terrain_indices,
                st_p,
                a_right,
                scale_f64(dir, -1.0),
                a_fwd,
                0.55,
                2.2 + ((si.unsigned_abs() + ai) % 2) as f64 * 0.9,
                origin_f64,
                [0.18, 0.22, 0.28, 0.65],
                [0.08, 0.42, 0.58, MAT_MACRO_PROP],
            );
        }
    }

    // Radiant cyan-emerald & amethyst glowing crystal spire clusters lining the grotto floor!
    for ci in 0..8usize {
        let ang = (ci as f64 / 8.0) * std::f64::consts::TAU + 0.3;
        let rad = 2.0 + (ci % 3) as f64 * 1.45;
        let c_base = add_f64(
            add_f64(floor_pos, scale_f64(t1, ang.cos() * rad)),
            scale_f64(t2, ang.sin() * rad),
        );
        let is_amethyst = ci % 3 == 0;
        let (col, em) = if is_amethyst {
            ([0.66, 0.28, 1.00, 0.14], [1.45, 0.48, 2.85, MAT_MACRO_CRYSTAL])
        } else {
            ([0.14, 0.92, 0.86, 0.14], [0.28, 2.15, 2.65, MAT_MACRO_CRYSTAL])
        };
        push_oriented_pyramid_f64(
            terrain_verts,
            terrain_indices,
            c_base,
            t1,
            dir,
            t2,
            0.55 + (ci % 2) as f64 * 0.22,
            2.6 + (ci % 4) as f64 * 0.85,
            origin_f64,
            col,
            em,
        );
    }

    // Glowing subterranean bioluminescent mineral pool + mist veil in the grotto heart!
    push_oriented_box_f64(
        terrain_verts,
        terrain_indices,
        add_f64(floor_pos, scale_f64(dir, 0.18)),
        t1,
        dir,
        t2,
        [5.4, 0.24, 5.4],
        origin_f64,
        [0.12, 0.84, 0.96, 0.12],
        [0.24, 1.65, 2.25, MAT_MACRO_CRYSTAL],
    );
    push_cross_particle_f64(
        gas_verts,
        gas_indices,
        add_f64(floor_pos, scale_f64(dir, 2.4)),
        dir,
        t1,
        t2,
        4.6,
        3.2,
        origin_f64,
        [0.18, 0.90, 0.95, 0.22],
        [0.18, 0.85, 1.05, MAT_MACRO_GAS],
    );
}

#[allow(clippy::too_many_arguments)]
fn push_nordic_elk(
    solid_verts: &mut Vec<MacroPlanetVertex>,
    solid_indices: &mut Vec<u32>,
    fx_verts: &mut Vec<MacroPlanetVertex>,
    fx_indices: &mut Vec<u32>,
    base_pos: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    fwd: [f64; 3],
    origin_f64: [f64; 3],
    stride_phase: f64,
    is_stag: bool,
) {
    let sc = if is_stag { 1.15 } else { 0.94 };
    let coat = if is_stag {
        [0.36, 0.24, 0.15, 0.78]
    } else {
        [0.44, 0.34, 0.24, 0.80]
    };
    let em = [0.02, 0.015, 0.01, MAT_MACRO_ENTITY];

    push_contact_shadow_disc_f64(
        fx_verts,
        fx_indices,
        add_f64(base_pos, scale_f64(up, 0.10)),
        up,
        1.15 * sc,
        8,
        origin_f64,
        0.30,
    );

    // Torso
    let body_c = add_f64(base_pos, scale_f64(up, 1.12 * sc));
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        body_c,
        right,
        up,
        fwd,
        [0.64 * sc, 0.66 * sc, 1.55 * sc],
        origin_f64,
        coat,
        em,
    );
    // 4 animated striding legs
    let swing = stride_phase.sin() * 0.22 * sc;
    for &(sx, sz, s_dir) in &[(-0.22, 0.52, 1.0), (0.22, 0.52, -1.0), (-0.22, -0.52, -1.0), (0.22, -0.52, 1.0)] {
        let leg_c = add_f64(
            add_f64(base_pos, scale_f64(up, 0.45 * sc)),
            add_f64(scale_f64(right, sx * sc), scale_f64(fwd, sz * sc + swing * s_dir)),
        );
        push_oriented_box_f64(
            solid_verts,
            solid_indices,
            leg_c,
            right,
            up,
            fwd,
            [0.16 * sc, 0.88 * sc, 0.18 * sc],
            origin_f64,
            [0.24, 0.16, 0.10, 0.82],
            em,
        );
    }
    // Neck & head (gentle grazing bob)
    let bob = (stride_phase * 0.5).sin() * 0.10 * sc;
    let head_c = add_f64(
        add_f64(base_pos, scale_f64(up, 1.72 * sc + bob)),
        scale_f64(fwd, 0.88 * sc),
    );
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        add_f64(add_f64(base_pos, scale_f64(up, 1.48 * sc + bob * 0.5)), scale_f64(fwd, 0.68 * sc)),
        right,
        up,
        fwd,
        [0.32 * sc, 0.62 * sc, 0.36 * sc],
        origin_f64,
        coat,
        em,
    );
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        head_c,
        right,
        up,
        fwd,
        [0.32 * sc, 0.34 * sc, 0.58 * sc],
        origin_f64,
        [0.42, 0.30, 0.19, 0.78],
        em,
    );
    // Proud branching antlers on stags!
    if is_stag {
        for side_s in [-1.0f64, 1.0] {
            let ant_c = add_f64(
                add_f64(head_c, scale_f64(up, 0.48 * sc)),
                scale_f64(right, side_s * 0.44 * sc),
            );
            push_oriented_box_f64(
                solid_verts,
                solid_indices,
                ant_c,
                right,
                up,
                fwd,
                [0.62 * sc, 0.52 * sc, 0.14 * sc],
                origin_f64,
                [0.78, 0.72, 0.58, 0.65],
                [0.06, 0.05, 0.03, MAT_MACRO_ENTITY],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_ocean_whale(
    solid_verts: &mut Vec<MacroPlanetVertex>,
    solid_indices: &mut Vec<u32>,
    fx_verts: &mut Vec<MacroPlanetVertex>,
    fx_indices: &mut Vec<u32>,
    pos: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    fwd: [f64; 3],
    origin_f64: [f64; 3],
    pitch: f64,
    tail_phase: f64,
    spout_active: bool,
) {
    let w_fwd = normalize3_f64(add_f64(scale_f64(fwd, pitch.cos()), scale_f64(up, pitch.sin())));
    let w_up = normalize3_f64(cross_f64(right, w_fwd));
    let whale_top = [0.10, 0.20, 0.32, 0.24];
    let whale_belly = [0.58, 0.68, 0.78, 0.30];
    let em = [0.01, 0.03, 0.06, MAT_MACRO_ENTITY];

    // Main streamlined whale torso & pale belly
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        pos,
        right,
        w_up,
        w_fwd,
        [2.9, 1.95, 7.6],
        origin_f64,
        whale_top,
        em,
    );
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        sub_f64(pos, scale_f64(w_up, 0.78)),
        right,
        w_up,
        w_fwd,
        [2.5, 0.62, 6.4],
        origin_f64,
        whale_belly,
        em,
    );
    // Sweeping pectoral fins
    for side_s in [-1.0f64, 1.0] {
        let fin_p = add_f64(
            add_f64(pos, scale_f64(right, side_s * 2.2)),
            scale_f64(w_fwd, 0.8),
        );
        push_oriented_box_f64(
            solid_verts,
            solid_indices,
            fin_p,
            right,
            w_up,
            w_fwd,
            [2.4, 0.28, 1.45],
            origin_f64,
            whale_top,
            em,
        );
    }
    // Undulating tail stock & wide horizontal tail fluke!
    let tail_und = tail_phase.sin() * 0.95;
    let tail_p = add_f64(
        sub_f64(pos, scale_f64(w_fwd, 5.2)),
        scale_f64(w_up, tail_und),
    );
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        tail_p,
        right,
        w_up,
        w_fwd,
        [1.15, 0.92, 3.2],
        origin_f64,
        whale_top,
        em,
    );
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        sub_f64(tail_p, scale_f64(w_fwd, 1.7)),
        right,
        w_up,
        w_fwd,
        [4.2, 0.24, 1.5],
        origin_f64,
        whale_top,
        em,
    );

    // Misty white blowhole water spout when surfacing/breaching!
    if spout_active {
        let spout_p = add_f64(add_f64(pos, scale_f64(w_fwd, 2.2)), scale_f64(up, 3.4));
        push_cross_particle_f64(
            fx_verts,
            fx_indices,
            spout_p,
            up,
            right,
            fwd,
            1.6,
            3.8,
            origin_f64,
            [0.88, 0.96, 1.00, 0.56],
            [0.24, 0.36, 0.48, MAT_MACRO_GAS],
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_nordic_longboat(
    solid_verts: &mut Vec<MacroPlanetVertex>,
    solid_indices: &mut Vec<u32>,
    pos: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    fwd: [f64; 3],
    origin_f64: [f64; 3],
) {
    let wood = [0.30, 0.18, 0.10, 0.82];
    let em = [0.01, 0.01, 0.01, MAT_MACRO_ENTITY];
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        pos,
        right,
        up,
        fwd,
        [1.95, 0.72, 5.4],
        origin_f64,
        wood,
        em,
    );
    // Upturned Nordic prow & stern posts + mast
    for end_s in [-1.0f64, 1.0] {
        push_oriented_box_f64(
            solid_verts,
            solid_indices,
            add_f64(add_f64(pos, scale_f64(fwd, end_s * 2.85)), scale_f64(up, 0.62)),
            right,
            up,
            fwd,
            [0.36, 1.45, 0.52],
            origin_f64,
            [0.24, 0.14, 0.08, 0.84],
            em,
        );
    }
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        add_f64(pos, scale_f64(up, 2.15)),
        right,
        up,
        fwd,
        [0.18, 3.8, 0.18],
        origin_f64,
        [0.36, 0.24, 0.14, 0.82],
        em,
    );
    // Furled cream-crimson striped Nordic sail yard
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        add_f64(pos, scale_f64(up, 2.85)),
        right,
        up,
        fwd,
        [2.6, 0.42, 0.36],
        origin_f64,
        [0.86, 0.26, 0.22, 0.72],
        em,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_soaring_bird(
    solid_verts: &mut Vec<MacroPlanetVertex>,
    solid_indices: &mut Vec<u32>,
    pos: [f64; 3],
    up: [f64; 3],
    right: [f64; 3],
    fwd: [f64; 3],
    origin_f64: [f64; 3],
    flap: f64,
    is_eagle: bool,
) {
    let sc = if is_eagle { 1.45 } else { 0.92 };
    let col = if is_eagle {
        [0.24, 0.16, 0.11, 0.72]
    } else {
        [0.92, 0.95, 0.98, 0.65]
    };
    let em = [0.06, 0.08, 0.10, MAT_MACRO_ENTITY];
    push_oriented_box_f64(
        solid_verts,
        solid_indices,
        pos,
        right,
        up,
        fwd,
        [0.32 * sc, 0.24 * sc, 1.05 * sc],
        origin_f64,
        col,
        em,
    );
    for side_s in [-1.0f64, 1.0] {
        let w_dir = normalize3_f64(add_f64(scale_f64(right, side_s), scale_f64(up, flap)));
        let w_up = scale_f64(normalize3_f64(cross_f64(fwd, w_dir)), side_s);
        let w_c = add_f64(pos, scale_f64(w_dir, 0.85 * sc));
        push_oriented_box_f64(
            solid_verts,
            solid_indices,
            w_c,
            w_dir,
            w_up,
            fwd,
            [1.45 * sc, 0.09 * sc, 0.44 * sc],
            origin_f64,
            col,
            em,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_cross_particle_f64(
    verts: &mut Vec<MacroPlanetVertex>,
    indices: &mut Vec<u32>,
    center: [f64; 3],
    up: [f64; 3],
    t1: [f64; 3],
    t2: [f64; 3],
    half_w: f64,
    half_h: f64,
    origin_f64: [f64; 3],
    color: [f32; 4],
    emissive_mat: [f32; 4],
) {
    let n_f = [up[0] as f32, up[1] as f32, up[2] as f32];
    for axis in [t1, t2] {
        let base = verts.len() as u32;
        let p0 = sub_f64(sub_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h));
        let p1 = sub_f64(add_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h));
        let p2 = add_f64(add_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h));
        let p3 = add_f64(sub_f64(center, scale_f64(axis, half_w)), scale_f64(up, half_h));
        for p in [p0, p1, p2, p3] {
            verts.push(MacroPlanetVertex {
                pos: world_f64_to_camera_rel_f32(p, origin_f64),
                normal: n_f,
                color,
                emissive_mat,
                water_ice: [0.0; 4],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}
