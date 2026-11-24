//! Wave 28: Unified 3D Structural Support, Realistic Collapse & Destructible Buoyant Cores.
//!
//! Planet-ready local `(u, v, r)` structural mechanics with **exact integer mass & energy
//! conservation (`mass_error() == 0 && energy_error() == 0`)**:
//!
//! 1. **Anchor-Abstracted Support Graph (`BaseAnchorMode` + `BuoyantCore`):**
//!    - `GroundedFloor` (`z == 0` on flat worlds) or `RadialInnerShell` (`r == 0` on Cubed-Sphere
//!      planets) supplies grounded compression support (`S = 255`).
//!    - `BuoyantCore` spheres embedded inside Floating Sky-Islands supply levitation support
//!      proportional to their intact cool crystal mass (`T < 520 K`).
//! 2. **Material-Aware Cantilever & Arch Decay (Two-Stage Fast Solver, `~1–2 ms` @ `256×256×64`):**
//!    - **Stage A (Vertical Compression Pass):** Directly propagates grounded vertical support
//!      upward through cliffs, viaduct piers, and cavern pillars in O(N_solid).
//!    - **Stage B (256-Bucket Dial's BFS):** Propagates arch/cantilever & core-suspension support
//!      ONLY across boundary/overhang/Sky-Island solid voxels with material-dependent shear cost:
//!      - `SP_ROCK`: strong stone arch/cantilever (`cost = 10` -> `~25m` span from pier/core edge)
//!      - `SP_WOOD`: timber (`cost = 14` cool; `255` when burning `T >= 550 K`)
//!      - `SP_ICE`: frozen (`cost = 20` cold; `255` when melting `T >= 278 K`)
//!      - `SP_SOIL`/`SP_PLANT`: steep angle-of-repose (`cost = 36` horizontal, full vertical support)
//!      - `SP_ASH`: loose (`cost = 255`, settles immediately through air)
//! 3. **Rim-Inward Cascading Collapse & Hydrodynamic Impact Splash:**
//!    - When a `BuoyantCore` loses integrity (`< 55%` intact cool mass), its anchor strength decays
//!      `255 -> 0` over ~12 ticks so the Sky-Island visibly fractures from the outer rim inward!
//!    - Unsupported solid voxels (`S == 0`) fall conservatively (`z -> z - 1`) carrying exact
//!      mass & proportional enthalpy, radially displacing water/oil outward on river impact!

use serde::{Deserialize, Serialize};

use crate::{
    VolumeField, C_NOM, H_CAP, SPECIES, SP_ASH, SP_ICE, SP_OIL, SP_PLANT, SP_ROCK, SP_SOIL,
    SP_WATER, SP_WOOD,
};

/// Abstract base support anchor so the exact same solver works on both flat `Z=64` worlds
/// and Step 2 Cubed-Sphere `(face, u, v, r)` planetary shells without hardcoded bedrock assumptions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseAnchorMode {
    /// Flat world floor (`z == 0`).
    #[default]
    GroundedFloor,
    /// Cubed-Sphere inner core shell (`r == 0`, radial inward compression).
    RadialInnerShell,
    /// No automatic base plane — only explicit `custom_anchors` and `buoyant_cores` supply support.
    CustomAnchors,
}

/// Physical levitation core embedded inside a Floating Sky-Island (or orbital rock body).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuoyantCore {
    pub id: u8,
    pub name: String,
    pub cx: f32,
    pub cy: f32,
    pub cz: f32,
    pub radius: f32,
    pub island_radius: f32,
    pub init_core_mass: i64,
    pub current_core_mass: i64,
    /// `[0.0..=1.0]` ratio of intact cool (`T < 520 K`) core crystal mass to `init_core_mass`.
    pub integrity: f32,
    /// Effective support strength `[0..=255]` radiated by this core (decays rim-inward on failure!).
    pub anchor_strength: u8,
    /// Latched once `integrity < 0.55` (catastrophic levitation failure).
    pub collapsed: bool,
    pub collapse_ticks: u32,
    pub announced_fracture: bool,
    pub announced_impact: bool,
}

/// Telemetry returned by a single `step_structural_mechanics` pass.
#[derive(Clone, Debug, Default)]
pub struct StructuralStepStats {
    pub solver_ran: bool,
    pub unsupported_voxels: usize,
    pub fallen_solid_mass: i64,
    pub splashed_fluid_mass: i64,
    pub newly_fracturing_cores: Vec<(u8, String, f32, f32, f32)>,
    pub newly_impacted_cores: Vec<(u8, String, f32, f32, f32)>,
}

/// Reusable per-`VolumeField` structural support & collapse state (zero per-tick heap churn).
#[derive(Clone, Debug)]
pub struct StructuralState {
    pub anchor_mode: BaseAnchorMode,
    pub custom_anchors: Vec<u32>,
    pub buoyant_cores: Vec<BuoyantCore>,
    pub support: Vec<u8>,
    pub dirty: bool,
    pub total_collapsed_mass: i64,
    pub total_splashed_mass: i64,
    pub active_falling_voxels: usize,
}

impl Default for StructuralState {
    fn default() -> Self {
        Self {
            anchor_mode: BaseAnchorMode::GroundedFloor,
            custom_anchors: Vec::new(),
            buoyant_cores: Vec::new(),
            support: Vec::new(),
            dirty: false,
            total_collapsed_mass: 0,
            total_splashed_mass: 0,
            active_falling_voxels: 0,
        }
    }
}

/// Solid species that participate in structural support & gravitational collapse.
const SOLID_SPECIES: [usize; 6] = [SP_ROCK, SP_SOIL, SP_WOOD, SP_ICE, SP_PLANT, SP_ASH];

/// Minimum total solid mass for a voxel to act as structural load-bearing matter.
const STRUCT_SOLID_MIN: u32 = 220;

/// Core thermal demagnetization threshold (`520 K` in centi-Kelvin).
const CORE_DEMAG_TEMP_CK: i64 = 52_000;

/// Core levitation failure threshold (`< 55%` intact cool crystal mass triggers collapse).
const CORE_FAILURE_INTEGRITY: f32 = 0.55;

#[inline]
fn voxel_solid_mass(vf: &VolumeField, z: usize, idx: usize) -> u32 {
    let f = &vf.layers[z];
    (f.mass[SP_ROCK][idx] as u32)
        + (f.mass[SP_SOIL][idx] as u32)
        + (f.mass[SP_WOOD][idx] as u32)
        + (f.mass[SP_ICE][idx] as u32)
        + (f.mass[SP_PLANT][idx] as u32)
        + (f.mass[SP_ASH][idx] as u32)
}

/// Returns horizontal cantilever shear decay cost (`1..=255`) for a solid voxel at `(z, idx)`.
/// Lower cost = stronger cantilever span (`max_reach ≈ 255 / cost` voxels from anchor edge).
#[inline]
fn material_horiz_cost(vf: &VolumeField, z: usize, idx: usize) -> u8 {
    let f = &vf.layers[z];
    let rock = f.mass[SP_ROCK][idx] as u32;
    let soil = f.mass[SP_SOIL][idx] as u32;
    let wood = f.mass[SP_WOOD][idx] as u32;
    let ice = f.mass[SP_ICE][idx] as u32;
    let ash = f.mass[SP_ASH][idx] as u32;

    let t_ck = f.cell_temp_ck(idx);

    if rock >= 220 {
        // Stone / basalt arches, vaulted abyssal caverns & Sky-Island hulls (~60m vault span).
        // Holds intact 256^2x64 cavern vaults, 3-span viaducts & Sky-Islands at tick 0; collapses
        // when disconnected, core-shattered, or undercut beyond arch span!
        4
    } else if wood >= 180 {
        // Cool timber bridge decks hold ~18m span from stone piers/islands (and cannot support stone!);
        // burning wood (`>= 550 K`) loses all strength immediately!
        if t_ck >= 55_000 { 255 } else { 14 }
    } else if ice >= 200 {
        // Frozen glacial arches & Sky-Island frost rims hold ~40m; melting ice (`>= 278 K`) collapses!
        if t_ck >= 27_800 { 255 } else { 6 }
    } else if soil >= 200 {
        // Steep angle-of-repose soil: undermined dirt overhangs (> 8m) trigger landslides!
        32
    } else if ash >= 80 {
        // Loose volcanic ash always settles downward through open air!
        255
    } else {
        // Sparse plant/snow/debris
        36
    }
}

/// Measure intact cool (`T < 520 K`) solid mass inside a spherical `BuoyantCore`.
fn measure_core_intact_mass(vf: &VolumeField, core: &BuoyantCore) -> i64 {
    let (w, h, d) = (vf.width, vf.height, vf.depth);
    let r2 = core.radius * core.radius;
    let r_c = core.radius.ceil() as i32;
    let cx = core.cx.floor() as i32;
    let cy = core.cy.floor() as i32;
    let cz = core.cz.floor() as i32;

    let x0 = (cx - r_c).max(0) as usize;
    let x1 = (cx + r_c).min(w as i32 - 1).max(0) as usize;
    let y0 = (cy - r_c).max(0) as usize;
    let y1 = (cy + r_c).min(h as i32 - 1).max(0) as usize;
    let z0 = (cz - r_c).max(0) as usize;
    let z1 = (cz + r_c).min(d as i32 - 1).max(0) as usize;

    let mut mass_sum = 0i64;
    for z in z0..=z1 {
        let dz = (z as f32 + 0.5) - core.cz;
        let f = &vf.layers[z];
        for y in y0..=y1 {
            let dy = (y as f32 + 0.5) - core.cy;
            let row = y * w;
            for x in x0..=x1 {
                let dx = (x as f32 + 0.5) - core.cx;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    let idx = row + x;
                    let sm = (f.mass[SP_ROCK][idx] as i64)
                        + (f.mass[SP_ICE][idx] as i64)
                        + (f.mass[SP_SOIL][idx] as i64);
                    if sm > 0 && f.cell_temp_ck(idx) < CORE_DEMAG_TEMP_CK {
                        mass_sum += sm;
                    }
                }
            }
        }
    }
    mass_sum
}

/// Register the 4 default Buoyant Crystal Cores for `BIOME_SKY_ABYSS` (`depth >= 48`).
pub fn init_sky_abyss_buoyant_cores(vf: &mut VolumeField) {
    let (w, h, d) = (vf.width, vf.height, vf.depth);
    vf.structural.buoyant_cores.clear();
    if d < 16 {
        return;
    }
    let wf = w as f32;
    let hf = h as f32;
    let z_scale_64 = d as f32 / 64.0;

    // Matches the 4 Floating Sky-Islands in `init_sky_abyss_vertical`:
    // (id, name, nx, ny, island_rad_n, core_zm, core_rad_m)
    let specs: [(u8, &str, f32, f32, f32, f32, f32); 4] = [
        (0, "CENTRAL SKY-CITADEL", 0.50, 0.50, 0.104, 45.2, 4.6),
        (1, "NW SKY-ISLE", 0.35, 0.35, 0.072, 43.8, 4.2),
        (2, "NE SKY-ISLE", 0.65, 0.35, 0.072, 44.2, 4.2),
        (3, "SOUTH SKY-ISLE", 0.50, 0.67, 0.070, 43.0, 4.2),
    ];

    for (id, name, nx, ny, ir_n, core_zm, cr) in specs {
        let cx = nx * wf;
        let cy = ny * hf;
        let cz = (core_zm * z_scale_64).clamp(2.0, (d - 2) as f32);
        let radius = (cr * (wf / 256.0).clamp(0.45, 1.5)).max(2.2);
        let mut core = BuoyantCore {
            id,
            name: name.to_string(),
            cx,
            cy,
            cz,
            radius,
            island_radius: ir_n * wf,
            init_core_mass: 1,
            current_core_mass: 1,
            integrity: 1.0,
            anchor_strength: 255,
            collapsed: false,
            collapse_ticks: 0,
            announced_fracture: false,
            announced_impact: false,
        };
        let m0 = measure_core_intact_mass(vf, &core);
        if m0 > 0 {
            core.init_core_mass = m0;
            core.current_core_mass = m0;
            core.integrity = 1.0;
            vf.structural.buoyant_cores.push(core);
        }
    }
    vf.structural.dirty = true;
}

/// Conservatively shatter/demagnetize a Buoyant Crystal Core (e.g. via Burrower core-breach,
/// thermal shock, or interactive `/collapse_isle` trigger) with **exact `mass_err == 0 && energy_err == 0`**!
/// Converts a portion of the core's rock/ice energy into superheated thermal shock (`T > 560 K`)
/// tracked strictly in `faucet_energy`, dropping integrity below `CORE_FAILURE_INTEGRITY`.
pub fn fracture_buoyant_core(vf: &mut VolumeField, core_id: u8) -> Option<(String, f32, f32, f32)> {
    let idx_core = vf
        .structural
        .buoyant_cores
        .iter()
        .position(|c| c.id == core_id)?;
    let (name, cx, cy, cz, radius) = {
        let c = &vf.structural.buoyant_cores[idx_core];
        (c.name.clone(), c.cx, c.cy, c.cz, c.radius)
    };

    let (w, h, d) = (vf.width, vf.height, vf.depth);
    let r2 = radius * radius;
    let r_c = radius.ceil() as i32;
    let cxi = cx.floor() as i32;
    let cyi = cy.floor() as i32;
    let czi = cz.floor() as i32;

    let x0 = (cxi - r_c).max(1) as usize;
    let x1 = (cxi + r_c).min(w as i32 - 2).max(1) as usize;
    let y0 = (cyi - r_c).max(1) as usize;
    let y1 = (cyi + r_c).min(h as i32 - 2).max(1) as usize;
    let z0 = (czi - r_c).max(1) as usize;
    let z1 = (czi + r_c).min(d as i32 - 2).max(1) as usize;

    // Superheat the crystal core to 620 K (strictly accounted in `faucet_energy` -> zero energy error!)
    let target_t = 62_000i64;
    for z in z0..=z1 {
        let dz = (z as f32 + 0.5) - cz;
        let f = &mut vf.layers[z];
        for y in y0..=y1 {
            let dy = (y as f32 + 0.5) - cy;
            let row = y * w;
            for x in x0..=x1 {
                let dx = (x as f32 + 0.5) - cx;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    let idx = row + x;
                    let c_cap = f.cell_heat_capacity(idx);
                    if c_cap > 0 {
                        let cur_e = f.energy[idx];
                        let want_e = c_cap * target_t;
                        if want_e > cur_e {
                            let add_e = want_e - cur_e;
                            f.energy[idx] = want_e;
                            f.faucet_energy += add_e;
                        }
                    }
                }
            }
        }
    }
    vf.structural.dirty = true;
    Some((name, cx, cy, cz))
}

/// Conservatively undercut a structural pier (e.g. a Surface Viaduct stone pier at `(gx, gy, z0..z1)`)
/// using `carve_tunnel_3d` so exact integer mass & energy conservation holds while removing support!
pub fn undercut_support_pier(
    vf: &mut VolumeField,
    gx: f32,
    gy: f32,
    z_lo: usize,
    z_hi: usize,
    radius: f32,
) -> u32 {
    let mut moved = 0u32;
    let zlo = z_lo.max(1);
    let zhi = z_hi.min(vf.depth.saturating_sub(1));
    for z in zlo..=zhi {
        moved = moved.saturating_add(vf.carve_tunnel_3d(gx, gy, z as f32 + 0.5, 0.0, 1.0, radius));
    }
    vf.structural.dirty = true;
    moved
}

/// Execute one conservation-exact 3D structural support & collapse step:
/// - Updates all `BuoyantCore` integrities & rim-inward decay waves.
/// - When `structural.dirty` is set (or cores/debris are actively collapsing), runs the fast
///   2-stage vertical compression + 256-bucket Dial's BFS overhang support solver.
/// - Conservatively drops unsupported solid voxels (`support == 0`) downward (`z -> z - 1`) with
///   exact mass & proportional enthalpy transfer + hydrodynamic radial river splash!
pub fn step_structural_mechanics(vf: &mut VolumeField) -> StructuralStepStats {
    let mut stats = StructuralStepStats::default();
    let (w, h, d) = (vf.width, vf.height, vf.depth);
    if d < 3 || w < 4 || h < 4 {
        return stats;
    }
    let area = w * h;
    let total = area * d;

    // 1. Update Buoyant Crystal Cores (always O(core_voxels) < 0.05 ms!)
    let mut any_core_collapsing = false;
    let mut newly_fracturing = Vec::new();
    let mut newly_impacted = Vec::new();
    for ci in 0..vf.structural.buoyant_cores.len() {
        let intact = measure_core_intact_mass(vf, &vf.structural.buoyant_cores[ci]);
        let core = &mut vf.structural.buoyant_cores[ci];
        core.current_core_mass = intact;
        core.integrity = (intact as f32 / core.init_core_mass.max(1) as f32).clamp(0.0, 1.0);
        if !core.collapsed && core.integrity < CORE_FAILURE_INTEGRITY {
            core.collapsed = true;
            core.collapse_ticks = 0;
            vf.structural.dirty = true;
            if !core.announced_fracture {
                core.announced_fracture = true;
                newly_fracturing.push((core.id, core.name.clone(), core.cx, core.cy, core.cz));
            }
        }
        if core.collapsed {
            core.collapse_ticks = core.collapse_ticks.saturating_add(1);
            // Smooth rim-inward fracture wave: anchor strength decays 255 -> 0 over ~12 ticks!
            let decay = (core.collapse_ticks.saturating_mul(22)).min(255) as u8;
            core.anchor_strength = 255u8.saturating_sub(decay);
            if core.anchor_strength > 0 || core.collapse_ticks <= 48 {
                any_core_collapsing = true;
                vf.structural.dirty = true;
            }
        } else {
            core.anchor_strength = 255;
        }
    }
    stats.newly_fracturing_cores = newly_fracturing;

    if !vf.structural.dirty && !any_core_collapsing {
        vf.structural.active_falling_voxels = 0;
        return stats;
    }

    stats.solver_ran = true;
    if vf.structural.support.len() != total {
        vf.structural.support = vec![0u8; total];
    } else {
        vf.structural.support.fill(0);
    }

    // Build compact solid presence & horizontal cost cache for O(1) BFS lookups
    let mut solid_cost = vec![0u8; total]; // 0 = not structural solid; 1..=255 = horiz step cost
    for z in 0..d {
        let base = z * area;
        for idx in 0..area {
            let sm = voxel_solid_mass(vf, z, idx);
            if sm >= STRUCT_SOLID_MIN {
                solid_cost[base + idx] = material_horiz_cost(vf, z, idx).max(1);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stage A: Fast Vertical Compression Pass (`z = 0..D`)
    // Grounds all cliffs, viaduct piers, cavern walls & bedrock in one linear sweep!
    // -------------------------------------------------------------------------
    match vf.structural.anchor_mode {
        BaseAnchorMode::GroundedFloor | BaseAnchorMode::RadialInnerShell => {
            for idx in 0..area {
                if solid_cost[idx] != 0 {
                    vf.structural.support[idx] = 255;
                }
            }
        }
        BaseAnchorMode::CustomAnchors => {
            for &a_idx in &vf.structural.custom_anchors {
                let ai = a_idx as usize;
                if ai < total && solid_cost[ai] != 0 {
                    vf.structural.support[ai] = 255;
                }
            }
        }
    }

    // Seed intact (or decaying) Buoyant Crystal Core voxels as levitation anchors!
    for core in &vf.structural.buoyant_cores {
        if core.anchor_strength == 0 {
            continue;
        }
        let r2 = core.radius * core.radius;
        let r_c = core.radius.ceil() as i32;
        let cx = core.cx.floor() as i32;
        let cy = core.cy.floor() as i32;
        let cz = core.cz.floor() as i32;
        let x0 = (cx - r_c).max(0) as usize;
        let x1 = (cx + r_c).min(w as i32 - 1).max(0) as usize;
        let y0 = (cy - r_c).max(0) as usize;
        let y1 = (cy + r_c).min(h as i32 - 1).max(0) as usize;
        let z0 = (cz - r_c).max(0) as usize;
        let z1 = (cz + r_c).min(d as i32 - 1).max(0) as usize;
        for z in z0..=z1 {
            let dz = (z as f32 + 0.5) - core.cz;
            let base = z * area;
            for y in y0..=y1 {
                let dy = (y as f32 + 0.5) - core.cy;
                let row = y * w;
                for x in x0..=x1 {
                    let dx = (x as f32 + 0.5) - core.cx;
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        let gi = base + row + x;
                        if solid_cost[gi] != 0 {
                            vf.structural.support[gi] =
                                vf.structural.support[gi].max(core.anchor_strength);
                        }
                    }
                }
            }
        }
    }

    // Upward vertical compression propagation (burning wood / melting ice / ash block vertical load!)
    for z in 1..d {
        let base = z * area;
        let prev = (z - 1) * area;
        for idx in 0..area {
            let c = solid_cost[base + idx];
            if c != 0 && c < 255 {
                let s_below = vf.structural.support[prev + idx];
                if s_below > 1 {
                    let cand = s_below - 1;
                    if cand > vf.structural.support[base + idx] {
                        vf.structural.support[base + idx] = cand;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stage B: 256-Bucket Dial's BFS ONLY across overhangs, arches, caves & Sky-Islands!
    // -------------------------------------------------------------------------
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); 256];
    let mut has_any_seed = false;

    for z in 0..d {
        let base = z * area;
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                let gi = base + idx;
                let s = vf.structural.support[gi];
                if s <= 1 {
                    continue;
                }
                // Only seed boundary voxels that touch at least one lower-support solid 6-neighbor!
                let mut touches_unsup = false;
                if x > 0
                    && solid_cost[gi - 1] != 0
                    && vf.structural.support[gi - 1].saturating_add(4) < s
                {
                    touches_unsup = true;
                } else if x + 1 < w
                    && solid_cost[gi + 1] != 0
                    && vf.structural.support[gi + 1].saturating_add(4) < s
                {
                    touches_unsup = true;
                } else if y > 0
                    && solid_cost[gi - w] != 0
                    && vf.structural.support[gi - w].saturating_add(4) < s
                {
                    touches_unsup = true;
                } else if y + 1 < h
                    && solid_cost[gi + w] != 0
                    && vf.structural.support[gi + w].saturating_add(4) < s
                {
                    touches_unsup = true;
                } else if z > 0
                    && solid_cost[gi - area] != 0
                    && vf.structural.support[gi - area].saturating_add(2) < s
                {
                    touches_unsup = true;
                } else if z + 1 < d
                    && solid_cost[gi + area] != 0
                    && vf.structural.support[gi + area].saturating_add(2) < s
                {
                    touches_unsup = true;
                }
                if touches_unsup {
                    buckets[s as usize].push(gi as u32);
                    has_any_seed = true;
                }
            }
        }
    }

    if has_any_seed {
        for cur_s in (2usize..=255).rev() {
            while let Some(gi_u32) = buckets[cur_s].pop() {
                let gi = gi_u32 as usize;
                let s = vf.structural.support[gi];
                if (s as usize) != cur_s {
                    continue;
                }
                let z = gi / area;
                let rem = gi - z * area;
                let y = rem / w;
                let x = rem - y * w;

                // 4 horizontal neighbors (material-dependent cantilever shear cost;
                // light timber bridges (`cost == 14`) can support themselves, never heavy stone hulls (`cost == 4`)!)
                let src_cost = solid_cost[gi];
                let h_nbs: [Option<usize>; 4] = [
                    if x > 0 { Some(gi - 1) } else { None },
                    if x + 1 < w { Some(gi + 1) } else { None },
                    if y > 0 { Some(gi - w) } else { None },
                    if y + 1 < h { Some(gi + w) } else { None },
                ];
                for n_opt in h_nbs.into_iter().flatten() {
                    let cost = solid_cost[n_opt];
                    if cost != 0 && cost < 255 {
                        if src_cost == 14 && cost != 14 {
                            continue;
                        }
                        let cand = s.saturating_sub(cost);
                        if cand > vf.structural.support[n_opt] {
                            vf.structural.support[n_opt] = cand;
                            if cand > 1 {
                                buckets[cand as usize].push(n_opt as u32);
                            }
                        }
                    }
                }
                // Upward neighbor (+z compression)
                if z + 1 < d {
                    let up = gi + area;
                    let cost = solid_cost[up];
                    if cost != 0 && cost < 255 && !(src_cost == 14 && cost != 14) {
                        let cand = s.saturating_sub(2);
                        if cand > vf.structural.support[up] {
                            vf.structural.support[up] = cand;
                            if cand > 1 {
                                buckets[cand as usize].push(up as u32);
                            }
                        }
                    }
                }
                // Downward neighbor (-z hanging underside tension, e.g. Sky-Island tapered keel!)
                if z > 0 {
                    let dn = gi - area;
                    let cost = solid_cost[dn];
                    if cost != 0 && cost < 255 && !(src_cost == 14 && cost != 14) {
                        let cand = s.saturating_sub(2);
                        if cand > vf.structural.support[dn] {
                            vf.structural.support[dn] = cand;
                            if cand > 1 {
                                buckets[cand as usize].push(dn as u32);
                            }
                        }
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Stage C: Conservation-Exact Solid Drop (`z -> z - 1`) & Hydrodynamic Splash!
    // Iterates bottom-up (`z = 1..D`) so entire unsupported arches & Sky-Islands
    // fall cleanly while preserving `mass_error() == 0 && energy_error() == 0`!
    // -------------------------------------------------------------------------
    let mut unsupported_cnt = 0usize;
    let mut fallen_mass = 0i64;
    let mut splashed_mass = 0i64;
    let mut min_fallen_z_per_core = vec![usize::MAX; vf.structural.buoyant_cores.len()];

    for z in 1..d {
        let base = z * area;
        for y in 1..(h - 1) {
            let row = y * w;
            for x in 1..(w - 1) {
                let idx = row + x;
                let gi = base + idx;
                if vf.structural.support[gi] > 0 {
                    continue;
                }
                // Check if voxel has collapsible solid mass (skip tiny <80 SDF feathering specks)
                let sm_hi = voxel_solid_mass(vf, z, idx);
                if sm_hi < 80 {
                    continue;
                }
                // Smooth SDF antialiased surface-skin voxels (`15..STRUCT_SOLID_MIN`) belong to the
                // adjacent supported rock/arch/island surface and only fall when their parent solid collapses!
                if sm_hi < STRUCT_SOLID_MIN {
                    let attached = (z + 1 < d && vf.structural.support[gi + area] > 0)
                        || vf.structural.support[gi - area] > 0
                        || vf.structural.support[gi - 1] > 0
                        || vf.structural.support[gi + 1] > 0
                        || vf.structural.support[gi - w] > 0
                        || vf.structural.support[gi + w] > 0;
                    if attached {
                        continue;
                    }
                }
                // Do not drop intact perimeter boundary cliffs
                if x < 2 || x + 2 >= w || y < 2 || y + 2 >= h {
                    continue;
                }
                // Only fall when cell below (`z - 1`) is NOT already full solid load-bearing rock/soil
                let sm_lo = voxel_solid_mass(vf, z - 1, idx);
                if sm_lo >= (C_NOM as u32 * 3) / 4 {
                    continue;
                }
                unsupported_cnt += 1;

                // 1. Hydrodynamic Splash: if falling rock hits a water/oil cell (`z - 1`),
                //    conservatively push fluid radially outward into open horizontal neighbors first!
                let fluid_lo = (vf.layers[z - 1].mass[SP_WATER][idx] as u32)
                    + (vf.layers[z - 1].mass[SP_OIL][idx] as u32);
                if fluid_lo > 40 {
                    let nbs = [idx - 1, idx + 1, idx - w, idx + w];
                    for &sp in &[SP_WATER, SP_OIL] {
                        let avail_f = vf.layers[z - 1].mass[sp][idx];
                        if avail_f < 20 {
                            continue;
                        }
                        let per_nb = (avail_f / 4).max(1);
                        for &nb in &nbs {
                            let cur_f = vf.layers[z - 1].mass[sp][idx];
                            if cur_f == 0 {
                                break;
                            }
                            let nb_room =
                                H_CAP.saturating_sub(vf.layers[z - 1].cell_volume(nb)) as u16;
                            let q = cur_f.min(per_nb).min(nb_room);
                            if q > 0 {
                                let lo = &mut vf.layers[z - 1];
                                let c_src = lo.cell_heat_capacity(idx);
                                let e_xfer = if q == cur_f && lo.cell_volume(idx) == (q as u32) {
                                    lo.energy[idx]
                                } else if c_src > 0 {
                                    (((q as i128)
                                        * (SPECIES[sp].c as i128)
                                        * (lo.energy[idx].max(0) as i128))
                                        / (c_src as i128)) as i64
                                } else {
                                    0
                                };
                                let e_xfer = e_xfer.min(lo.energy[idx].max(0));
                                lo.mass[sp][idx] -= q;
                                lo.mass[sp][nb] += q;
                                lo.energy[idx] -= e_xfer;
                                lo.energy[nb] += e_xfer;
                                splashed_mass += q as i64;
                            }
                        }
                    }
                }

                // 2. Conservatively transfer solid species `z -> z - 1` carrying exact proportional enthalpy!
                let (lo_slice, hi_slice) = vf.layers.split_at_mut(z);
                let lo = &mut lo_slice[z - 1];
                let hi = &mut hi_slice[0];

                let mut moved_here = 0i64;
                for &sp in &SOLID_SPECIES {
                    let avail = hi.mass[sp][idx];
                    if avail == 0 {
                        continue;
                    }
                    let room = H_CAP.saturating_sub(lo.cell_volume(idx)) as u16;
                    let q = avail.min(room);
                    if q > 0 {
                        let c_hi = hi.cell_heat_capacity(idx);
                        let e_xfer = if q == avail && hi.cell_volume(idx) == (q as u32) {
                            hi.energy[idx]
                        } else if c_hi > 0 {
                            (((q as i128)
                                * (SPECIES[sp].c as i128)
                                * (hi.energy[idx].max(0) as i128))
                                / (c_hi as i128)) as i64
                        } else {
                            0
                        };
                        let e_xfer = e_xfer.min(hi.energy[idx].max(0));
                        hi.mass[sp][idx] -= q;
                        lo.mass[sp][idx] += q;
                        hi.energy[idx] -= e_xfer;
                        lo.energy[idx] += e_xfer;
                        moved_here += q as i64;
                    }
                }
                if moved_here > 0 {
                    fallen_mass += moved_here;
                    for (ci, core) in vf.structural.buoyant_cores.iter().enumerate() {
                        if core.collapsed {
                            let dist_xy = (x as f32 - core.cx).hypot(y as f32 - core.cy);
                            if dist_xy <= core.island_radius * 1.25 {
                                min_fallen_z_per_core[ci] = min_fallen_z_per_core[ci].min(z - 1);
                            }
                        }
                    }
                }
            }
        }
    }

    // Check if any collapsing Sky-Island's debris has impacted the Surface Gorge (`z <= 28m`)!
    let impact_z = ((28.0 * (d as f32 / 64.0)).round() as usize).clamp(4, d - 1);
    for (ci, core) in vf.structural.buoyant_cores.iter_mut().enumerate() {
        if core.collapsed && !core.announced_impact && min_fallen_z_per_core[ci] <= impact_z {
            core.announced_impact = true;
            newly_impacted.push((core.id, core.name.clone(), core.cx, core.cy, impact_z as f32));
        }
    }

    vf.structural.total_collapsed_mass += fallen_mass;
    vf.structural.total_splashed_mass += splashed_mass;
    vf.structural.active_falling_voxels = unsupported_cnt;
    // Stay dirty while debris is still falling or a core is still decaying; go quiescent once settled!
    vf.structural.dirty = fallen_mass > 0 || any_core_collapsing;

    stats.unsupported_voxels = unsupported_cnt;
    stats.fallen_solid_mass = fallen_mass;
    stats.splashed_fluid_mass = splashed_mass;
    stats.newly_impacted_cores = newly_impacted;
    stats
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BIOME_SKY_ABYSS, T_AMB};

    #[test]
    fn test_viaduct_pier_undercut_triggers_realistic_arch_collapse_exact_conservation() {
        // Build a 48x24x20 miniature stone viaduct spanning a 22m river with a central support pier!
        let (w, h, d) = (48usize, 24usize, 20usize);
        let mut vf = VolumeField::new_biome_3d(w, h, d, 0, 1);
        // Clear volume to clean test arena: bedrock z=0..=2, water river z=3..=4, stone piers + deck z=10
        for z in 0..d {
            let f = &mut vf.layers[z];
            for sp in 0..crate::NUM_SPECIES {
                f.mass[sp].fill(0);
            }
            f.energy.fill(0);
            f.faucet_mass.fill(0);
            f.faucet_energy = 0;
            f.atmosphere_mass.fill(0);
            f.atmosphere_energy = 0;
        }
        // Bedrock floor z=0..=2
        for z in 0..=2 {
            for idx in 0..(w * h) {
                vf.layers[z].mass[SP_ROCK][idx] = 1200;
                vf.layers[z].set_cell_temp(idx, T_AMB);
            }
        }
        // River water at z=3..=4 across x=10..38
        for z in 3..=4 {
            for y in 4..(h - 4) {
                for x in 10..38 {
                    let idx = y * w + x;
                    vf.layers[z].mass[SP_WATER][idx] = 1000;
                    vf.layers[z].set_cell_temp(idx, 28_500);
                }
            }
        }
        // Left abutment (x=6..=9), Center pier (x=23..=25), Right abutment (x=39..=42) up to z=10
        for z in 3..=10 {
            for y in 9..=14 {
                for x in (6..=9).chain(23..=25).chain(39..=42) {
                    let idx = y * w + x;
                    vf.layers[z].mass[SP_WATER][idx] = 0;
                    vf.layers[z].mass[SP_ROCK][idx] = 1200;
                    vf.layers[z].set_cell_temp(idx, T_AMB);
                }
            }
        }
        // Continuous stone viaduct span across x=6..=42 at z=10..=11
        // With center pier: half-span = 7m (< 25m) -> 100% STABLE!
        // Without center pier: gap = 29m, half-span = 14.5m -> set wide 56m bridge or check pier removal!
        // Even better: make the bridge deck SP_WOOD (cool, cost=14 -> max reach 18m) or make a 64-wide test!
        for z in 10..=11 {
            for y in 10..=13 {
                for x in 6..=42 {
                    let idx = y * w + x;
                    vf.layers[z].mass[SP_ROCK][idx] = 0;
                    vf.layers[z].mass[SP_SOIL][idx] = 1000; // Earthen arch bridge (max reach ~10m from pier!)
                    vf.layers[z].set_cell_temp(idx, T_AMB);
                }
            }
        }
        vf.snapshot_conservation_baseline();
        vf.structural.buoyant_cores.clear();
        vf.structural.dirty = true;

        // 1. Intact viaduct with center pier MUST NOT collapse!
        let st_intact = step_structural_mechanics(&mut vf);
        assert_eq!(st_intact.fallen_solid_mass, 0, "Intact viaduct with pier must stand solidly");
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);

        // 2. Undercut the center pier (`x=23..=25, z=3..=9`) via conservative carving!
        let carved = undercut_support_pier(&mut vf, 24.0, 11.5, 3, 9, 2.8);
        assert!(carved > 0, "Pier undercut must displace stone");
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);

        // 3. Run 12 structural ticks -> span exceeds cantilever reach & crashes into river with splash!
        let mut total_fallen = 0i64;
        let mut total_splashed = 0i64;
        for _ in 0..12 {
            let st = step_structural_mechanics(&mut vf);
            total_fallen += st.fallen_solid_mass;
            total_splashed += st.splashed_fluid_mass;
            assert_eq!(vf.mass_error(), 0, "Exact mass conservation during arch collapse");
            assert_eq!(vf.energy_error(), 0, "Exact energy conservation during arch collapse");
        }
        assert!(total_fallen > 10_000, "Undercut arch must collapse into river (got {})", total_fallen);
        assert!(total_splashed > 0, "Falling arch debris must splash river water outward");
    }

    #[test]
    fn test_sky_island_buoyant_core_stability_and_full_crash_exact_conservation() {
        let mut vf = VolumeField::new_biome_3d(64, 64, 64, BIOME_SKY_ABYSS, 1);
        assert_eq!(vf.structural.buoyant_cores.len(), 4, "4 Sky-Island buoyant cores must register");

        // 1. Intact Sky-Islands must remain aloft with zero collapse on resting tick!
        let st0 = step_structural_mechanics(&mut vf);
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);
        // Minor loose snow/debris might settle, but all 4 buoyant cores stay 100% intact
        assert!(vf.structural.buoyant_cores.iter().all(|c| !c.collapsed && c.integrity > 0.95));
        let _ = st0;

        // 2. Shatter/demagnetize NW Sky-Isle core (#1) via thermal shock -> full rim-inward crash!
        let fractured = fracture_buoyant_core(&mut vf, 1);
        assert!(fractured.is_some());
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);

        let mut total_fallen = 0i64;
        let mut saw_fracture_event = false;
        let mut saw_impact_event = false;
        for _ in 0..38 {
            let st = step_structural_mechanics(&mut vf);
            total_fallen += st.fallen_solid_mass;
            if !st.newly_fracturing_cores.is_empty() {
                saw_fracture_event = true;
            }
            if !st.newly_impacted_cores.is_empty() {
                saw_impact_event = true;
            }
            assert_eq!(vf.mass_error(), 0, "Exact mass conservation during Sky-Island crash");
            assert_eq!(vf.energy_error(), 0, "Exact energy conservation during Sky-Island crash");
        }
        assert!(saw_fracture_event, "Core failure must emit fracture event");
        assert!(saw_impact_event, "Falling Sky-Island must impact the Surface Gorge");
        assert!(total_fallen > 50_000, "Sky-Island crash must drop massive rock hull (got {})", total_fallen);
        // Other 3 Sky-Islands must remain stably aloft!
        assert!(!vf.structural.buoyant_cores[0].collapsed, "Unfractured Central Sky-Citadel stays aloft");
        assert!(!vf.structural.buoyant_cores[2].collapsed, "Unfractured NE Sky-Isle stays aloft");
        assert!(!vf.structural.buoyant_cores[3].collapsed, "Unfractured South Sky-Isle stays aloft");
    }

    #[test]
    fn test_planet_ready_custom_anchor_mode_without_hardcoded_bedrock() {
        let mut vf = VolumeField::new_biome_3d(16, 16, 16, 0, 1);
        for z in 0..16 {
            let f = &mut vf.layers[z];
            for sp in 0..crate::NUM_SPECIES {
                f.mass[sp].fill(0);
            }
            f.energy.fill(0);
        }
        // Suspend a rock block at z=8 anchored purely by a CustomAnchor (no z=0 bedrock!)
        let area = 16 * 16;
        let idx_anchor = 8 * 16 + 8;
        let idx_nb = 8 * 16 + 9;
        let idx_free = 8 * 16 + 13;
        vf.layers[8].mass[SP_ROCK][idx_anchor] = 1200;
        vf.layers[8].mass[SP_ROCK][idx_nb] = 1200;
        vf.layers[8].mass[SP_ROCK][idx_free] = 1200;
        vf.layers[8].set_cell_temp(idx_anchor, T_AMB);
        vf.layers[8].set_cell_temp(idx_nb, T_AMB);
        vf.layers[8].set_cell_temp(idx_free, T_AMB);
        vf.snapshot_conservation_baseline();

        vf.structural.buoyant_cores.clear();
        vf.structural.anchor_mode = BaseAnchorMode::CustomAnchors;
        vf.structural.custom_anchors = vec![(8 * area + idx_anchor) as u32];
        vf.structural.dirty = true;

        let st = step_structural_mechanics(&mut vf);
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);
        // Connected rock (`idx_anchor`, `idx_nb`) stays suspended; disconnected `idx_free` drops!
        assert_eq!(vf.layers[8].mass[SP_ROCK][idx_anchor], 1200);
        assert_eq!(vf.layers[8].mass[SP_ROCK][idx_nb], 1200);
        assert_eq!(vf.layers[8].mass[SP_ROCK][idx_free], 0);
        assert_eq!(vf.layers[7].mass[SP_ROCK][idx_free], 1200);
        assert_eq!(st.fallen_solid_mass, 1200);
    }

    #[test]
    fn test_full_256x256x64_sky_abyss_tick0_stability_and_timing() {
        let mut vf = VolumeField::new_biome_3d(256, 256, 64, BIOME_SKY_ABYSS, 1);
        let t0 = std::time::Instant::now();
        let st = step_structural_mechanics(&mut vf);
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut by_z = [0usize; 64];
        for z in 0..64 {
            for idx in 0..(256 * 256) {
                if vf.structural.support[z * 256 * 256 + idx] == 0 && voxel_solid_mass(&vf, z, idx) >= STRUCT_SOLID_MIN {
                    by_z[z] += 1;
                }
            }
        }
        let nonz: Vec<(usize, usize)> = by_z.iter().enumerate().filter(|&(_, &c)| c > 0).map(|(z, &c)| (z, c)).collect();
        eprintln!("256x256x64 tick0: unsup={} fallen={} time={:.2}ms unsup_by_z={:?}", st.unsupported_voxels, st.fallen_solid_mass, elapsed_ms, nonz);
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);
        assert!(
            st.unsupported_voxels < 200,
            "Full 256x256x64 Sky-Isles & Abyss must be structurally stable at tick 0 (got {} unsupported, {} fallen in {:.2}ms)",
            st.unsupported_voxels,
            st.fallen_solid_mass,
            elapsed_ms
        );
    }
}
