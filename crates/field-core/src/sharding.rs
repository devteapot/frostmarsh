//! Wave 24B (MMO Phase 2B): Seamless `2×2` Multi-Region Ghost-Halo Sharding,
//! Exact Cross-Shard Mass & Energy Conservation, Async Multi-Rate `BorderFluxEscrow`,
//! `16m` Border Hysteresis Authority Handoff, and Toroidal `RollingClipmap256`.
//!
//! Zero new crate dependencies. Guarantees `global_mass_error == 0 && global_energy_error == 0`
//! bit-for-bit across both synchronous (`10 Hz / 10 Hz`) and asynchronous multi-rate
//! (`10 Hz / 2 Hz`) region seams.

use crate::{
    C_NOM, Field, H_CAP, MOBILE_SPECIES, NUM_MOBILE, NUM_SPECIES, RawStepOutput, SPECIES, SP_OIL,
    SP_ROCK, SP_SMOKE, SP_STEAM, SP_WATER, T_AMB, T_MAX, T_MIN, VolumeField,
};

/// Single-cell snapshot extracted into a read-only ghost halo strip across a shard seam.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BorderCellSnapshot {
    pub mass: [u16; NUM_SPECIES],
    pub energy: i64,
    pub flow_e: i16,
    pub flow_s: i16,
    pub elevation: u8,
}

impl BorderCellSnapshot {
    pub fn from_field(field: &Field, idx: usize) -> Self {
        let mut mass = [0u16; NUM_SPECIES];
        for s in 0..NUM_SPECIES {
            mass[s] = field.mass[s][idx];
        }
        Self {
            mass,
            energy: field.energy[idx],
            flow_e: field.flow_e[idx],
            flow_s: field.flow_s[idx],
            elevation: field.elevation[idx],
        }
    }

    #[inline]
    pub fn volume(&self) -> u32 {
        let mut sum = 0u32;
        for &m in &self.mass {
            sum += m as u32;
        }
        sum
    }

    #[inline]
    pub fn heat_capacity(&self) -> i64 {
        let mut cap = 0i64;
        for s in 0..NUM_SPECIES {
            cap += (self.mass[s] as i64) * SPECIES[s].c;
        }
        cap
    }

    #[inline]
    pub fn temp_ck(&self) -> i64 {
        let c = self.heat_capacity();
        if c <= 0 {
            T_AMB
        } else {
            (self.energy / c).clamp(T_MIN, T_MAX)
        }
    }

    #[inline]
    pub fn is_rock_wall(&self) -> bool {
        self.mass[SP_ROCK] >= 500
    }
}

/// Face-centered antisymmetric integer border flux between owner cell `A` (West or North)
/// and neighbor cell `B` (East or South).
///
/// Convention:
/// - Positive `q_mobile[m]` moves mass of `MOBILE_SPECIES[m]` from `A -> B`.
/// - Positive `delta_e_total` (= `advected_e + conducted_e`) moves thermal energy from `A -> B`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectedBorderFlux {
    pub q_mobile: [i32; NUM_MOBILE],
    pub advected_e: i64,
    pub conducted_e: i64,
}

impl ProjectedBorderFlux {
    #[inline]
    pub fn total_thermal_e(&self) -> i64 {
        self.advected_e + self.conducted_e
    }

    #[inline]
    pub fn total_mass_moved_abs(&self) -> i64 {
        self.q_mobile.iter().map(|&q| (q as i64).abs()).sum()
    }
}

/// Raw floating-point border flux proposal (matching `RawStepOutput` units:
/// `q_raw` in `1000` mass units, `heat_raw` in `1e6` thermal energy units).
#[derive(Clone, Copy, Debug, Default)]
pub struct RawBorderProposal {
    pub q_raw: [f32; NUM_MOBILE],
    pub heat_raw: f32,
}

#[inline]
fn sanitize_border_f32(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
fn round_ties_even_border_i32(v: f32) -> i32 {
    v.round_ties_even()
        .clamp(i32::MIN as f32 / 2.0, i32::MAX as f32 / 2.0) as i32
}

/// Compute analytic RefSim-consistent raw flux proposal across a seam face `(A -> B)`.
pub fn propose_border_flux_analytic(
    a: &BorderCellSnapshot,
    b: &BorderCellSnapshot,
    is_east: bool,
) -> RawBorderProposal {
    if a.is_rock_wall() || b.is_rock_wall() {
        // Still allow heat conduction across rock walls!
        let mut prop = RawBorderProposal::default();
        prop.heat_raw = analytic_heat_raw(a, b);
        return prop;
    }

    let water_a = a.mass[SP_WATER] as f32;
    let water_b = b.mass[SP_WATER] as f32;
    let oil_a = a.mass[SP_OIL] as f32;
    let oil_b = b.mass[SP_OIL] as f32;
    let liq_a = water_a + oil_a;
    let liq_b = water_b + oil_b;

    let head_a = (a.elevation as f32) * 120.0 + liq_a;
    let head_b = (b.elevation as f32) * 120.0 + liq_b;
    let mom = if is_east {
        a.flow_e as f32
    } else {
        a.flow_s as f32
    };
    let drive = 0.20 * (head_a - head_b) + 0.35 * mom;

    let (frac_w, frac_o, has_liq) = if drive >= 0.0 {
        (
            water_a / (liq_a + 1e-6),
            oil_a / (liq_a + 1e-6),
            liq_a > 1e-4,
        )
    } else {
        (
            water_b / (liq_b + 1e-6),
            oil_b / (liq_b + 1e-6),
            liq_b > 1e-4,
        )
    };

    let q_water = if has_liq { drive * frac_w } else { 0.0 };
    let q_oil = (if has_liq { 0.55 * drive * frac_o } else { 0.0 }) + 0.06 * (oil_a - oil_b);

    let steam_a = a.mass[SP_STEAM] as f32;
    let steam_b = b.mass[SP_STEAM] as f32;
    let smoke_a = a.mass[SP_SMOKE] as f32;
    let smoke_b = b.mass[SP_SMOKE] as f32;

    let (q_steam, q_smoke) = if is_east {
        (
            0.16 * (steam_a - steam_b) + 0.035 * steam_a,
            0.15 * (smoke_a - smoke_b) + 0.035 * smoke_a,
        )
    } else {
        (
            0.16 * (steam_a - steam_b) - 0.020 * steam_b,
            0.15 * (smoke_a - smoke_b) - 0.020 * smoke_b,
        )
    };

    RawBorderProposal {
        q_raw: [
            q_water / (C_NOM as f32),
            q_steam / (C_NOM as f32),
            q_smoke / (C_NOM as f32),
            q_oil / (C_NOM as f32),
        ],
        heat_raw: analytic_heat_raw(a, b),
    }
}

fn analytic_heat_raw(a: &BorderCellSnapshot, b: &BorderCellSnapshot) -> f32 {
    let ca = a.heat_capacity() as f64;
    let cb = b.heat_capacity() as f64;
    if ca <= 0.0 || cb <= 0.0 {
        return 0.0;
    }
    let ka = weighted_k(a);
    let kb = weighted_k(b);
    let k_eff = 2.0 * ka * kb / (ka + kb + 1e-6);
    let c_eff = (ca * cb) / (ca + cb + 1e-6);
    let dt = (a.temp_ck() - b.temp_ck()) as f64;
    let heat = 0.008 * k_eff * c_eff * dt;
    (heat / 1e6) as f32
}

fn weighted_k(cell: &BorderCellSnapshot) -> f64 {
    let mut tot = 0.0f64;
    let mut sum_k = 0.0f64;
    for s in 0..NUM_SPECIES {
        let m = cell.mass[s] as f64;
        tot += m;
        sum_k += m * (SPECIES[s].k as f64);
    }
    if tot > 1e-4 { sum_k / tot } else { 2.0 }
}

/// Exact integer face-centered antisymmetric border flux projection (`project_border_flux`).
///
/// Mirrors `Field::project_and_apply` Steps 2–6 on a cross-shard seam face `(A -> B)`:
/// - Strictly preserves non-negativity (`m_a' >= 0`, `m_b' >= 0`)
/// - Strictly enforces `H_CAP` volume ceiling on both sides
/// - Strictly enforces discrete 2nd-law heat conduction & `[T_MIN, T_MAX]` bounds
/// - Guarantees `Δm_A + Δm_B == 0` and `ΔE_A + ΔE_B == 0` bit-exactly!
pub fn project_border_flux(
    a: &BorderCellSnapshot,
    b: &BorderCellSnapshot,
    proposal: &RawBorderProposal,
) -> ProjectedBorderFlux {
    let vol_a = a.volume();
    let vol_b = b.volume();
    let cap_a = a.heat_capacity();
    let cap_b = b.heat_capacity();
    let temp_a = a.temp_ck();
    let temp_b = b.temp_ck();

    let mut q = [0i32; NUM_MOBILE];
    if !a.is_rock_wall() && !b.is_rock_wall() {
        for m in 0..NUM_MOBILE {
            let s = MOBILE_SPECIES[m];
            let v = sanitize_border_f32(proposal.q_raw[m]);
            let q_raw = round_ties_even_border_i32(v * (C_NOM as f32));
            q[m] = if q_raw > 0 {
                q_raw.min((a.mass[s] / 4) as i32)
            } else if q_raw < 0 {
                q_raw.max(-((b.mass[s] / 4) as i32))
            } else {
                0
            };
        }

        // Volume ceiling on receiver B (positive flows)
        let pos_tot: i64 = q.iter().filter(|&&v| v > 0).map(|&v| v as i64).sum();
        let cap_room_b = (H_CAP.saturating_sub(vol_b) / 4) as i64;
        if pos_tot > cap_room_b && pos_tot > 0 {
            for v in &mut q {
                if *v > 0 {
                    *v = ((*v as i64) * cap_room_b / pos_tot) as i32;
                }
            }
        }

        // Volume ceiling on receiver A (negative flows)
        let neg_tot: i64 = q.iter().filter(|&&v| v < 0).map(|&v| (-v) as i64).sum();
        let cap_room_a = (H_CAP.saturating_sub(vol_a) / 4) as i64;
        if neg_tot > cap_room_a && neg_tot > 0 {
            for v in &mut q {
                if *v < 0 {
                    *v = -(((-*v) as i64) * cap_room_a / neg_tot) as i32;
                }
            }
        }
    }

    // Exact deterministic integer advected thermal energy (`A -> B` positive)
    let mut advected_e = 0i64;
    let mut delta_c_a = 0i64;
    let mut delta_c_b = 0i64;
    for m in 0..NUM_MOBILE {
        let qv = q[m] as i64;
        if qv == 0 {
            continue;
        }
        let s = MOBILE_SPECIES[m];
        let cs = SPECIES[s].c;
        delta_c_a -= qv * cs;
        delta_c_b += qv * cs;
        if qv > 0 && cap_a > 0 {
            let e_adv = ((qv as i128) * (cs as i128) * (a.energy as i128) / (cap_a as i128)) as i64;
            advected_e += e_adv;
        } else if qv < 0 && cap_b > 0 {
            let q_abs = -qv;
            let e_adv =
                ((q_abs as i128) * (cs as i128) * (b.energy as i128) / (cap_b as i128)) as i64;
            advected_e -= e_adv;
        }
    }

    // Discrete 2nd-law heat conduction with post-advection [T_MIN, T_MAX] safety guard
    let g_raw = (sanitize_border_f32(proposal.heat_raw) as f64 * 1e6).round() as i64;
    let dt = temp_a - temp_b;
    let mut conducted_e = if cap_a > 0 && cap_b > 0 && dt != 0 {
        let max_g = ((dt.unsigned_abs() as u128) * (cap_a as u128) * (cap_b as u128)
            / (4 * ((cap_a + cap_b) as u128))) as i64;
        if dt > 0 {
            g_raw.clamp(0, max_g)
        } else {
            g_raw.clamp(-max_g, 0)
        }
    } else {
        0
    };

    // Guard both sides against leaving [T_MIN * C_post, T_MAX * C_post]
    let c_a_post = (cap_a + delta_c_a).max(0);
    let c_b_post = (cap_b + delta_c_b).max(0);
    let e_a_adv = a.energy - advected_e;
    let e_b_adv = b.energy + advected_e;

    if conducted_e > 0 {
        let max_give_a = (e_a_adv - c_a_post * T_MIN).max(0);
        let max_take_b = (c_b_post * T_MAX - e_b_adv).max(0);
        conducted_e = conducted_e.min(max_give_a).min(max_take_b);
    } else if conducted_e < 0 {
        let max_give_b = (e_b_adv - c_b_post * T_MIN).max(0);
        let max_take_a = (c_a_post * T_MAX - e_a_adv).max(0);
        conducted_e = conducted_e.max(-max_give_b).max(-max_take_a);
    }

    ProjectedBorderFlux {
        q_mobile: q,
        advected_e,
        conducted_e,
    }
}

/// Apply a projected antisymmetric border flux symmetrically to owner cell `A` and neighbor cell `B`.
pub fn apply_border_flux_symmetric(
    field_a: &mut Field,
    idx_a: usize,
    field_b: &mut Field,
    idx_b: usize,
    flux: &ProjectedBorderFlux,
    is_east: bool,
) {
    for m in 0..NUM_MOBILE {
        let qv = flux.q_mobile[m];
        if qv != 0 {
            let s = MOBILE_SPECIES[m];
            field_a.mass[s][idx_a] = (field_a.mass[s][idx_a] as i32 - qv) as u16;
            field_b.mass[s][idx_b] = (field_b.mass[s][idx_b] as i32 + qv) as u16;
        }
    }
    let de = flux.total_thermal_e();
    field_a.energy[idx_a] -= de;
    field_b.energy[idx_b] += de;

    let liq_flow = (flux.q_mobile[0] + flux.q_mobile[3]).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    if is_east {
        field_a.flow_e[idx_a] = liq_flow;
    } else {
        field_a.flow_s[idx_a] = liq_flow;
    }
}

/// Apply ONLY the owner (`-flux`) side of a projected border flux to cell `A` (used during
/// async multi-rate fast ticks when the neighbor's half is deposited into `BorderFluxEscrow`).
fn apply_border_flux_owner_only(
    field_a: &mut Field,
    idx_a: usize,
    flux: &ProjectedBorderFlux,
    is_east: bool,
) {
    for m in 0..NUM_MOBILE {
        let qv = flux.q_mobile[m];
        if qv != 0 {
            let s = MOBILE_SPECIES[m];
            field_a.mass[s][idx_a] = (field_a.mass[s][idx_a] as i32 - qv) as u16;
        }
    }
    field_a.energy[idx_a] -= flux.total_thermal_e();
    let liq_flow = (flux.q_mobile[0] + flux.q_mobile[3]).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    if is_east {
        field_a.flow_e[idx_a] = liq_flow;
    } else {
        field_a.flow_s[idx_a] = liq_flow;
    }
}

/// Statistics returned by `RegionShardGrid::step_conserved_sync`.
#[derive(Clone, Debug, Default)]
pub struct ShardSyncStepStats {
    pub tick: u64,
    pub global_mass_error: i64,
    pub global_energy_error: i64,
    pub seam_mass_crossed: i64,
    pub seam_energy_crossed: i64,
}

/// Read-only exchanged 1-cell border halo strips across the 4 internal seams of a `2×2` grid.
#[derive(Clone, Debug, Default)]
pub struct ExchangedGhostHalos {
    /// West->East seams: `(0,0)<->(0,1)` and `(1,0)<->(1,1)`, indexed `[seam_idx][z * height + y]`
    pub east_owner_cells: [Vec<BorderCellSnapshot>; 2],
    pub east_neighbor_cells: [Vec<BorderCellSnapshot>; 2],
    /// North->South seams: `(0,0)<->(1,0)` and `(0,1)<->(1,1)`, indexed `[seam_idx][z * width + x]`
    pub south_owner_cells: [Vec<BorderCellSnapshot>; 2],
    pub south_neighbor_cells: [Vec<BorderCellSnapshot>; 2],
}

/// `2×2` Grid of authoritative 3D `VolumeField` region shards:
/// - `shards[0]` = `(0,0)` NW
/// - `shards[1]` = `(0,1)` NE
/// - `shards[2]` = `(1,0)` SW
/// - `shards[3]` = `(1,1)` SE
///
/// Enforces bit-exact global mass & energy conservation across all 4 internal seams!
#[derive(Clone, Debug)]
pub struct RegionShardGrid {
    pub region_width: usize,
    pub region_height: usize,
    pub depth: usize,
    pub shards: [VolumeField; 4],
    pub ghost_halos: ExchangedGhostHalos,
    pub initial_global_mass: i64,
    pub initial_global_energy: i64,
    pub tick: u64,
}

impl RegionShardGrid {
    /// Wrap 4 equal-dimension `VolumeField` regions into a `2×2` sharded world.
    pub fn from_quadrants(mut shards: [VolumeField; 4]) -> Self {
        let w = shards[0].width;
        let h = shards[0].height;
        let d = shards[0].depth;
        for s in &shards {
            assert_eq!(s.width, w, "all shards must share region_width");
            assert_eq!(s.height, h, "all shards must share region_height");
            assert_eq!(s.depth, d, "all shards must share depth");
        }
        for s in &mut shards {
            s.snapshot_conservation_baseline();
        }
        let initial_global_mass: i64 = shards.iter().map(|s| s.current_accounted_mass()).sum();
        let initial_global_energy: i64 = shards.iter().map(|s| s.current_accounted_energy()).sum();
        let mut grid = Self {
            region_width: w,
            region_height: h,
            depth: d,
            shards,
            ghost_halos: ExchangedGhostHalos::default(),
            initial_global_mass,
            initial_global_energy,
            tick: 0,
        };
        grid.exchange_ghost_halos();
        grid
    }

    #[inline]
    pub fn shard_index(sy: usize, sx: usize) -> usize {
        assert!(sy < 2 && sx < 2, "2x2 shard index out of bounds");
        sy * 2 + sx
    }

    #[inline]
    pub fn shard(&self, sy: usize, sx: usize) -> &VolumeField {
        &self.shards[Self::shard_index(sy, sx)]
    }

    #[inline]
    pub fn shard_mut(&mut self, sy: usize, sx: usize) -> &mut VolumeField {
        &mut self.shards[Self::shard_index(sy, sx)]
    }

    /// Re-snapshot global conservation baselines across all 4 shards.
    pub fn snapshot_conservation_baseline(&mut self) {
        for s in &mut self.shards {
            s.snapshot_conservation_baseline();
        }
        self.initial_global_mass = self.current_global_mass();
        self.initial_global_energy = self.current_global_energy();
    }

    pub fn current_global_mass(&self) -> i64 {
        self.shards.iter().map(|s| s.current_accounted_mass()).sum()
    }

    pub fn current_global_energy(&self) -> i64 {
        self.shards.iter().map(|s| s.current_accounted_energy()).sum()
    }

    pub fn global_mass_error(&self) -> i64 {
        self.current_global_mass() - self.initial_global_mass
    }

    pub fn global_energy_error(&self) -> i64 {
        self.current_global_energy() - self.initial_global_energy
    }

    /// Exchange border-cell ghost halos across all 4 internal seams (`2` E/W + `2` N/S).
    pub fn exchange_ghost_halos(&mut self) {
        let w = self.region_width;
        let h = self.region_height;
        let d = self.depth;

        let ew_pairs = [(0usize, 1usize), (2usize, 3usize)];
        for (seam_i, &(west, east)) in ew_pairs.iter().enumerate() {
            let mut own = Vec::with_capacity(d * h);
            let mut nbr = Vec::with_capacity(d * h);
            for z in 0..d {
                let fa = &self.shards[west].layers[z];
                let fb = &self.shards[east].layers[z];
                for y in 0..h {
                    own.push(BorderCellSnapshot::from_field(fa, y * w + (w - 1)));
                    nbr.push(BorderCellSnapshot::from_field(fb, y * w));
                }
            }
            self.ghost_halos.east_owner_cells[seam_i] = own;
            self.ghost_halos.east_neighbor_cells[seam_i] = nbr;
        }

        let ns_pairs = [(0usize, 2usize), (1usize, 3usize)];
        for (seam_i, &(north, south)) in ns_pairs.iter().enumerate() {
            let mut own = Vec::with_capacity(d * w);
            let mut nbr = Vec::with_capacity(d * w);
            for z in 0..d {
                let fa = &self.shards[north].layers[z];
                let fb = &self.shards[south].layers[z];
                for x in 0..w {
                    own.push(BorderCellSnapshot::from_field(fa, (h - 1) * w + x));
                    nbr.push(BorderCellSnapshot::from_field(fb, x));
                }
            }
            self.ghost_halos.south_owner_cells[seam_i] = own;
            self.ghost_halos.south_neighbor_cells[seam_i] = nbr;
        }
    }

    /// Synchronous conserved `2×2` tick:
    /// 1. Exchange ghost halos across all 4 seams.
    /// 2. Compute & project face-centered antisymmetric border fluxes (`flux_e` owned by West,
    ///    `flux_s` owned by North) and apply symmetrically to BOTH sides.
    /// 3. Step each shard's 3D interior via `VolumeField::step_3d`.
    /// 4. Verify & return exact `global_mass_error == 0 && global_energy_error == 0`.
    pub fn step_conserved_sync(
        &mut self,
        raw_per_shard: Option<&[Vec<RawStepOutput>; 4]>,
    ) -> ShardSyncStepStats {
        self.exchange_ghost_halos();
        let w = self.region_width;
        let h = self.region_height;
        let d = self.depth;

        let mut seam_mass_crossed = 0i64;
        let mut seam_energy_crossed = 0i64;

        // 1. East-West seams: (0,0)<->(0,1) and (1,0)<->(1,1) — owned by West shard
        let ew_pairs = [(0usize, 1usize), (2usize, 3usize)];
        for (seam_i, &(west, east)) in ew_pairs.iter().enumerate() {
            let mut fluxes = Vec::with_capacity(d * h);
            for z in 0..d {
                for y in 0..h {
                    let k = z * h + y;
                    let a = &self.ghost_halos.east_owner_cells[seam_i][k];
                    let b = &self.ghost_halos.east_neighbor_cells[seam_i][k];
                    let prop = if let Some(raws) = raw_per_shard {
                        let raw_l = &raws[west][z];
                        let idx_a = y * w + (w - 1);
                        RawBorderProposal {
                            q_raw: [
                                raw_l.get(0, idx_a),
                                raw_l.get(1, idx_a),
                                raw_l.get(2, idx_a),
                                raw_l.get(3, idx_a),
                            ],
                            heat_raw: raw_l.get(8, idx_a),
                        }
                    } else {
                        propose_border_flux_analytic(a, b, true)
                    };
                    fluxes.push(project_border_flux(a, b, &prop));
                }
            }

            let (left_slice, right_slice) = self.shards.split_at_mut(east);
            let shard_w = &mut left_slice[west];
            let shard_e = &mut right_slice[0];
            for z in 0..d {
                let fa = &mut shard_w.layers[z];
                let fb = &mut shard_e.layers[z];
                for y in 0..h {
                    let fl = &fluxes[z * h + y];
                    seam_mass_crossed += fl.total_mass_moved_abs();
                    seam_energy_crossed += fl.total_thermal_e().abs();
                    apply_border_flux_symmetric(fa, y * w + (w - 1), fb, y * w, fl, true);
                }
            }
        }

        // Refresh ghost halos after E/W border update so N/S corners see exact post-E/W state
        self.exchange_ghost_halos();

        // 2. North-South seams: (0,0)<->(1,0) and (0,1)<->(1,1) — owned by North shard
        let ns_pairs = [(0usize, 2usize), (1usize, 3usize)];
        for (seam_i, &(north, south)) in ns_pairs.iter().enumerate() {
            let mut fluxes = Vec::with_capacity(d * w);
            for z in 0..d {
                for x in 0..w {
                    let k = z * w + x;
                    let a = &self.ghost_halos.south_owner_cells[seam_i][k];
                    let b = &self.ghost_halos.south_neighbor_cells[seam_i][k];
                    let prop = if let Some(raws) = raw_per_shard {
                        let raw_l = &raws[north][z];
                        let idx_a = (h - 1) * w + x;
                        RawBorderProposal {
                            q_raw: [
                                raw_l.get(4, idx_a),
                                raw_l.get(5, idx_a),
                                raw_l.get(6, idx_a),
                                raw_l.get(7, idx_a),
                            ],
                            heat_raw: raw_l.get(9, idx_a),
                        }
                    } else {
                        propose_border_flux_analytic(a, b, false)
                    };
                    fluxes.push(project_border_flux(a, b, &prop));
                }
            }

            let (top_slice, bot_slice) = self.shards.split_at_mut(south);
            let shard_n = &mut top_slice[north];
            let shard_s = &mut bot_slice[0];
            for z in 0..d {
                let fa = &mut shard_n.layers[z];
                let fb = &mut shard_s.layers[z];
                for x in 0..w {
                    let fl = &fluxes[z * w + x];
                    seam_mass_crossed += fl.total_mass_moved_abs();
                    seam_energy_crossed += fl.total_thermal_e().abs();
                    apply_border_flux_symmetric(fa, (h - 1) * w + x, fb, x, fl, false);
                }
            }
        }

        // 3. Step all 4 shard interiors independently (exact per-shard conservation)
        for s_idx in 0..4 {
            let raws_owned;
            let raws_ref = if let Some(r) = raw_per_shard {
                &r[s_idx]
            } else {
                raws_owned = self.shards[s_idx].refsim_raw_outputs_all();
                &raws_owned
            };
            self.shards[s_idx].step_3d(raws_ref);
        }

        self.tick += 1;
        ShardSyncStepStats {
            tick: self.tick,
            global_mass_error: self.global_mass_error(),
            global_energy_error: self.global_energy_error(),
            seam_mass_crossed,
            seam_energy_crossed,
        }
    }
}

// =============================================================================
// Async Multi-Rate `BorderFluxEscrow` (e.g. 10 Hz Fast Shard <-> 2 Hz Quiet Shard)
// =============================================================================

/// Seam orientation for `BorderFluxEscrow`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamAxis {
    /// Fast shard is West (`x = w - 1`), coarse neighbor is East (`x = 0`).
    EastWest,
    /// Fast shard is North (`y = h - 1`), coarse neighbor is South (`y = 0`).
    NorthSouth,
}

/// Asynchronous multi-rate border flux escrow allowing a fast shard (`10 Hz`) to execute
/// multiple substeps adjacent to a coarse shard (`2 Hz`) with **ZERO lockstep stalls**
/// and **BIT-EXACT mass & energy conservation** at EVERY fast substep and on coarse commit!
///
/// Invariant at every substep `k = 0..5` (even before coarse commit!):
/// - `fast.current_accounted_mass() + slow.current_accounted_mass() + escrow.escrowed_total_mass() == initial_mass`
/// - `fast.current_accounted_energy() + slow.current_accounted_energy() + escrow.escrowed_total_energy() == initial_energy`
#[derive(Clone, Debug)]
pub struct BorderFluxEscrow {
    pub axis: SeamAxis,
    pub depth: usize,
    pub span: usize,
    /// Virtual shadow of the coarse neighbor's seam cells, updated on every fast substep
    /// so accumulated fluxes NEVER violate `m >= 0`, `V <= H_CAP`, or `[T_MIN, T_MAX]` on commit!
    virtual_neighbor: Vec<BorderCellSnapshot>,
    /// Accumulated signed mobile mass deltas queued for the coarse neighbor (`+q` enters neighbor).
    pub queued_mass: Vec<[i64; NUM_MOBILE]>,
    /// Accumulated signed thermal energy deltas queued for the coarse neighbor (`+e` enters neighbor).
    pub queued_thermal_energy: Vec<i64>,
    pub fast_substeps_accumulated: u32,
}

impl BorderFluxEscrow {
    /// Capture a read-only seam snapshot from `coarse_neighbor` to open a new multi-rate window.
    pub fn begin_window(coarse_neighbor: &VolumeField, axis: SeamAxis) -> Self {
        let depth = coarse_neighbor.depth;
        let span = match axis {
            SeamAxis::EastWest => coarse_neighbor.height,
            SeamAxis::NorthSouth => coarse_neighbor.width,
        };
        let mut virtual_neighbor = Vec::with_capacity(depth * span);
        for z in 0..depth {
            let f = &coarse_neighbor.layers[z];
            for i in 0..span {
                let idx = match axis {
                    SeamAxis::EastWest => i * coarse_neighbor.width,
                    SeamAxis::NorthSouth => i,
                };
                virtual_neighbor.push(BorderCellSnapshot::from_field(f, idx));
            }
        }
        let n = depth * span;
        Self {
            axis,
            depth,
            span,
            virtual_neighbor,
            queued_mass: vec![[0i64; NUM_MOBILE]; n],
            queued_thermal_energy: vec![0i64; n],
            fast_substeps_accumulated: 0,
        }
    }

    /// Execute ONE fast-shard (`10 Hz`) border exchange substep against the virtual coarse halo,
    /// applying `-flux` immediately to `fast_shard` and escrowing `+flux` for `coarse_neighbor`
    /// without ever touching or stalling on `coarse_neighbor`!
    pub fn step_fast_border(&mut self, fast_shard: &mut VolumeField) -> (i64, i64) {
        assert_eq!(fast_shard.depth, self.depth);
        let w = fast_shard.width;
        let h = fast_shard.height;
        let is_east = self.axis == SeamAxis::EastWest;

        let mut mass_moved = 0i64;
        let mut energy_moved = 0i64;

        for z in 0..self.depth {
            let fa = &mut fast_shard.layers[z];
            for i in 0..self.span {
                let k = z * self.span + i;
                let idx_a = if is_east { i * w + (w - 1) } else { (h - 1) * w + i };
                let snap_a = BorderCellSnapshot::from_field(fa, idx_a);
                let snap_b = self.virtual_neighbor[k];

                let prop = propose_border_flux_analytic(&snap_a, &snap_b, is_east);
                let fl = project_border_flux(&snap_a, &snap_b, &prop);

                // 1. Apply owner half immediately to fast_shard
                apply_border_flux_owner_only(fa, idx_a, &fl, is_east);

                // 2. Deposit neighbor half into escrow & advance virtual_neighbor state
                let vnb = &mut self.virtual_neighbor[k];
                for m in 0..NUM_MOBILE {
                    let qv = fl.q_mobile[m];
                    if qv != 0 {
                        let s = MOBILE_SPECIES[m];
                        vnb.mass[s] = (vnb.mass[s] as i32 + qv) as u16;
                        self.queued_mass[k][m] += qv as i64;
                    }
                }
                let de = fl.total_thermal_e();
                vnb.energy += de;
                self.queued_thermal_energy[k] += de;

                mass_moved += fl.total_mass_moved_abs();
                energy_moved += de.abs();
            }
        }

        self.fast_substeps_accumulated += 1;
        (mass_moved, energy_moved)
    }

    /// Total mass currently held in-flight inside this escrow.
    pub fn escrowed_total_mass(&self) -> i64 {
        let mut sum = 0i64;
        for q in &self.queued_mass {
            for &v in q {
                sum += v;
            }
        }
        sum
    }

    /// Total energy (thermal + chemical `u0`) currently held in-flight inside this escrow.
    pub fn escrowed_total_energy(&self) -> i64 {
        let mut sum: i64 = self.queued_thermal_energy.iter().sum();
        for q in &self.queued_mass {
            for m in 0..NUM_MOBILE {
                let s = MOBILE_SPECIES[m];
                sum += q[m] * SPECIES[s].u0;
            }
        }
        sum
    }

    /// True if all queued mass and thermal energy have been committed & cleared.
    pub fn is_empty(&self) -> bool {
        self.queued_mass.iter().all(|q| q.iter().all(|&v| v == 0))
            && self.queued_thermal_energy.iter().all(|&e| e == 0)
    }

    /// Commit all accumulated escrow fluxes atomically onto `coarse_neighbor` on its `2 Hz` sync tick,
    /// then reset the escrow to zero and refresh the virtual halo snapshot for the next window!
    pub fn commit_to_neighbor(&mut self, coarse_neighbor: &mut VolumeField) {
        assert_eq!(coarse_neighbor.depth, self.depth);
        let w = coarse_neighbor.width;
        let is_east = self.axis == SeamAxis::EastWest;

        for z in 0..self.depth {
            let fb = &mut coarse_neighbor.layers[z];
            for i in 0..self.span {
                let k = z * self.span + i;
                let idx_b = if is_east { i * w } else { i };
                for m in 0..NUM_MOBILE {
                    let q_acc = self.queued_mass[k][m];
                    if q_acc != 0 {
                        let s = MOBILE_SPECIES[m];
                        let next_m = (fb.mass[s][idx_b] as i64) + q_acc;
                        debug_assert!(
                            next_m >= 0 && next_m <= u16::MAX as i64,
                            "escrow commit mass out of bounds: {next_m}"
                        );
                        fb.mass[s][idx_b] = next_m as u16;
                        self.queued_mass[k][m] = 0;
                    }
                }
                let e_acc = self.queued_thermal_energy[k];
                if e_acc != 0 {
                    fb.energy[idx_b] += e_acc;
                    self.queued_thermal_energy[k] = 0;
                }
                debug_assert!(fb.cell_volume(idx_b) <= H_CAP);
                self.virtual_neighbor[k] = BorderCellSnapshot::from_field(fb, idx_b);
            }
        }
        self.fast_substeps_accumulated = 0;
    }
}

// =============================================================================
// `16m` Hysteresis Authority Handoff & Toroidal `RollingClipmap256`
// =============================================================================

/// Border hysteresis authority handoff manager preventing entity ownership ping-pong
/// when players or bots oscillate near shard seams (e.g., `x = 256.0`).
///
/// With `region_size = 256.0` and `hysteresis_band_m = 16.0`, the deadband straddling
/// `x = 256.0` is `[248.0, 264.0]` (exact `16m` width):
/// - Owner `(0, sy)` retains authority until `x >= 264.0`.
/// - Owner `(1, sy)` retains authority until `x < 248.0`.
#[derive(Clone, Copy, Debug)]
pub struct HysteresisAuthority {
    pub region_size: f32,
    pub grid_x: i32,
    pub grid_y: i32,
    /// Full hysteresis deadband width centered on each seam (default `16.0` m -> `±8.0` m).
    pub hysteresis_band_m: f32,
}

impl HysteresisAuthority {
    pub const fn new(region_size: f32, grid_x: i32, grid_y: i32, hysteresis_band_m: f32) -> Self {
        Self {
            region_size,
            grid_x,
            grid_y,
            hysteresis_band_m,
        }
    }

    /// Initial authority shard `(sx, sy)` without hysteresis (for freshly spawned entities).
    pub fn shard_for_position(&self, x: f32, y: f32) -> (i32, i32) {
        let sx = (x / self.region_size).floor() as i32;
        let sy = (y / self.region_size).floor() as i32;
        (sx.clamp(0, self.grid_x - 1), sy.clamp(0, self.grid_y - 1))
    }

    /// Evaluate updated authority `(new_sx, new_sy, handoff_occurred)` given current owner
    /// `current_shard` and entity position `(x, y)`.
    pub fn update_authority(&self, current_shard: (i32, i32), x: f32, y: f32) -> ((i32, i32), bool) {
        let half = 0.5 * self.hysteresis_band_m.max(0.0);
        let new_sx = Self::update_axis(current_shard.0, x, self.region_size, self.grid_x, half);
        let new_sy = Self::update_axis(current_shard.1, y, self.region_size, self.grid_y, half);
        let next = (new_sx, new_sy);
        (next, next != current_shard)
    }

    fn update_axis(cur: i32, pos: f32, region_size: f32, grid_n: i32, half_band: f32) -> i32 {
        let cur_clamped = cur.clamp(0, grid_n - 1);
        let min_keep = (cur_clamped as f32) * region_size - half_band;
        let max_keep = ((cur_clamped + 1) as f32) * region_size + half_band;
        if pos >= min_keep && pos < max_keep {
            return cur_clamped;
        }
        let target = (pos / region_size).floor() as i32;
        target.clamp(0, grid_n - 1)
    }
}

/// Brick slot descriptor returned by `RollingClipmap256::recenter`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrickSlotTransfer {
    pub global_bx: i32,
    pub global_by: i32,
    pub slot_x: usize,
    pub slot_y: usize,
    pub slot_idx: usize,
}

/// Incremental eviction/load report emitted when the camera moves across brick or shard boundaries.
#[derive(Clone, Debug, Default)]
pub struct ClipmapUpdate {
    pub origin_bx: i32,
    pub origin_by: i32,
    pub evicted: Vec<BrickSlotTransfer>,
    pub loaded: Vec<BrickSlotTransfer>,
    pub unchanged_count: usize,
}

/// Fixed `16×16` (`256×256` voxel) toroidal GPU brick slot mapper (`RollingClipmap256`).
///
/// Maps unbounded global brick coordinates `(gbx: i32, gby: i32)` (each brick `16×16` voxels)
/// into fixed `[0..16) × [0..16)` toroidal GPU texture/buffer slots via `rem_euclid(16)`.
/// Crossing a shard border streams ONLY the newly exposed `16`-brick edge strip while keeping
/// the other `240` bricks (`93.75%`) resident in-place with ZERO GPU buffer reallocations!
#[derive(Clone, Debug)]
pub struct RollingClipmap256 {
    pub origin_bx: i32,
    pub origin_by: i32,
    resident: [Option<(i32, i32)>; Self::TOTAL_SLOTS],
}

impl Default for RollingClipmap256 {
    fn default() -> Self {
        Self::new()
    }
}

impl RollingClipmap256 {
    pub const BRICK_VOXELS: i32 = 16;
    pub const SLOTS_PER_AXIS: usize = 16;
    pub const TOTAL_SLOTS: usize = Self::SLOTS_PER_AXIS * Self::SLOTS_PER_AXIS;

    pub fn new() -> Self {
        Self {
            origin_bx: 0,
            origin_by: 0,
            resident: [None; Self::TOTAL_SLOTS],
        }
    }

    /// Map any unbounded global brick `(gbx, gby)` (including negative coordinates!) to its
    /// deterministic toroidal GPU slot `(slot_x, slot_y, slot_idx)` in `[0..16) × [0..16)`.
    #[inline]
    pub fn toroidal_slot(gbx: i32, gby: i32) -> (usize, usize, usize) {
        let sx = gbx.rem_euclid(Self::SLOTS_PER_AXIS as i32) as usize;
        let sy = gby.rem_euclid(Self::SLOTS_PER_AXIS as i32) as usize;
        (sx, sy, sy * Self::SLOTS_PER_AXIS + sx)
    }

    /// Query which global brick `(gbx, gby)` is currently resident in toroidal slot `(slot_x, slot_y)`.
    #[inline]
    pub fn resident_at_slot(&self, slot_x: usize, slot_y: usize) -> Option<(i32, i32)> {
        if slot_x < Self::SLOTS_PER_AXIS && slot_y < Self::SLOTS_PER_AXIS {
            self.resident[slot_y * Self::SLOTS_PER_AXIS + slot_x]
        } else {
            None
        }
    }

    /// Check if global brick `(gbx, gby)` is currently resident in the clipmap.
    #[inline]
    pub fn is_resident(&self, gbx: i32, gby: i32) -> bool {
        let (_, _, idx) = Self::toroidal_slot(gbx, gby);
        self.resident[idx] == Some((gbx, gby))
    }

    /// Recenter the `256×256`-voxel toroidal clipmap around world camera `(cam_x, cam_y)`
    /// and return exact evicted / newly-resident brick slot transfers.
    pub fn recenter(&mut self, cam_x: f32, cam_y: f32) -> ClipmapUpdate {
        let cam_bx = (cam_x / Self::BRICK_VOXELS as f32).floor() as i32;
        let cam_by = (cam_y / Self::BRICK_VOXELS as f32).floor() as i32;
        let half = (Self::SLOTS_PER_AXIS as i32) / 2;
        let origin_bx = cam_bx - half;
        let origin_by = cam_by - half;

        self.origin_bx = origin_bx;
        self.origin_by = origin_by;

        let mut evicted = Vec::new();
        let mut loaded = Vec::new();
        let mut unchanged_count = 0usize;

        for dy in 0..(Self::SLOTS_PER_AXIS as i32) {
            for dx in 0..(Self::SLOTS_PER_AXIS as i32) {
                let gbx = origin_bx + dx;
                let gby = origin_by + dy;
                let (slot_x, slot_y, slot_idx) = Self::toroidal_slot(gbx, gby);
                match self.resident[slot_idx] {
                    Some((cur_x, cur_y)) if cur_x == gbx && cur_y == gby => {
                        unchanged_count += 1;
                    }
                    prev => {
                        if let Some((old_x, old_y)) = prev {
                            evicted.push(BrickSlotTransfer {
                                global_bx: old_x,
                                global_by: old_y,
                                slot_x,
                                slot_y,
                                slot_idx,
                            });
                        }
                        self.resident[slot_idx] = Some((gbx, gby));
                        loaded.push(BrickSlotTransfer {
                            global_bx: gbx,
                            global_by: gby,
                            slot_x,
                            slot_y,
                            slot_idx,
                        });
                    }
                }
            }
        }

        ClipmapUpdate {
            origin_bx,
            origin_by,
            evicted,
            loaded,
            unchanged_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SP_SOIL;

    fn make_clean_open_shard(w: usize, h: usize, depth: usize) -> VolumeField {
        let mut vf = VolumeField::new_shattered_reach_3d(w, h, depth);
        // Clear all cells to flat open soil floor at z=0 and air above so rivers & fires
        // can flow unobstructed right across shard seams!
        for z in 0..depth {
            let f = &mut vf.layers[z];
            for idx in 0..(w * h) {
                for s in 0..NUM_SPECIES {
                    f.mass[s][idx] = 0;
                }
                f.flow_e[idx] = 0;
                f.flow_s[idx] = 0;
                f.elevation[idx] = if z == 0 { 10 } else { 10 + (z as u8) * 6 };
                if z == 0 {
                    f.mass[SP_SOIL][idx] = 600;
                    f.set_cell_temp(idx, T_AMB);
                } else {
                    f.energy[idx] = 0;
                }
            }
        }
        vf.snapshot_conservation_baseline();
        vf
    }

    #[test]
    fn test_2x2_cross_shard_river_and_heat_exact_conservation() {
        let w = 24usize;
        let h = 24usize;
        let d = 4usize;

        let mut s00 = make_clean_open_shard(w, h, d);
        let s01 = make_clean_open_shard(w, h, d);
        let s10 = make_clean_open_shard(w, h, d);
        let s11 = make_clean_open_shard(w, h, d);

        // Seed a high-head hot river & fire right up against the East (x=w-1) and South (y=h-1)
        // seams of Shard (0,0) so water AND heat physically cross into Shards (0,1) and (1,0)!
        let f0 = &mut s00.layers[0];
        for y in 6..18 {
            for x in (w - 5)..w {
                let idx = f0.idx(x, y);
                f0.mass[SP_WATER][idx] = 1600;
                f0.flow_e[idx] = 280;
                f0.set_cell_temp(idx, 35_500); // hot thermal front!
            }
        }
        for y in (h - 5)..h {
            for x in 6..18 {
                let idx = f0.idx(x, y);
                f0.mass[SP_WATER][idx] = 1500;
                f0.flow_s[idx] = 260;
                f0.set_cell_temp(idx, 34_800);
            }
        }

        let mut grid = RegionShardGrid::from_quadrants([s00, s01, s10, s11]);
        let east_water_before: u64 = grid.shard(0, 1).layers[0].mass[SP_WATER]
            .iter()
            .map(|&v| v as u64)
            .sum();
        let south_water_before: u64 = grid.shard(1, 0).layers[0].mass[SP_WATER]
            .iter()
            .map(|&v| v as u64)
            .sum();
        let east_max_temp_before = (0..h)
            .map(|y| grid.shard(0, 1).layers[0].cell_temp_ck(y * w))
            .max()
            .unwrap();

        let mut total_seam_mass = 0i64;
        let mut total_seam_energy = 0i64;
        for tick in 0..12 {
            let st = grid.step_conserved_sync(None);
            assert_eq!(st.global_mass_error, 0, "global_mass_error != 0 at tick {tick}");
            assert_eq!(
                st.global_energy_error, 0,
                "global_energy_error != 0 at tick {tick}"
            );
            total_seam_mass += st.seam_mass_crossed;
            total_seam_energy += st.seam_energy_crossed;
        }

        let east_water_after: u64 = grid.shard(0, 1).layers[0].mass[SP_WATER]
            .iter()
            .map(|&v| v as u64)
            .sum();
        let south_water_after: u64 = grid.shard(1, 0).layers[0].mass[SP_WATER]
            .iter()
            .map(|&v| v as u64)
            .sum();
        let east_max_temp_after = (0..h)
            .map(|y| grid.shard(0, 1).layers[0].cell_temp_ck(y * w))
            .max()
            .unwrap();

        assert!(
            east_water_after > east_water_before + 500,
            "River must physically cross West->East seam into Shard (0,1): before={east_water_before}, after={east_water_after}"
        );
        assert!(
            south_water_after > south_water_before + 500,
            "River must physically cross North->South seam into Shard (1,0): before={south_water_before}, after={south_water_after}"
        );
        assert!(
            east_max_temp_after > east_max_temp_before + 500,
            "Thermal energy must cross seam into Shard (0,1): before={east_max_temp_before}, after={east_max_temp_after}"
        );
        assert!(total_seam_mass > 0 && total_seam_energy > 0);
        assert_eq!(grid.global_mass_error(), 0);
        assert_eq!(grid.global_energy_error(), 0);
    }

    #[test]
    fn test_async_multirate_10hz_2hz_border_escrow_exact_conservation() {
        let w = 24usize;
        let h = 24usize;
        let d = 4usize;

        // Fast Shard (0,0) ticking at 10 Hz adjacent to quiet Shard (0,1) ticking at 2 Hz
        let mut fast = make_clean_open_shard(w, h, d);
        let mut slow = make_clean_open_shard(w, h, d);

        // Seed rushing hot water & oil right along East seam of fast shard (0,0)
        {
            let f0 = &mut fast.layers[0];
            for y in 4..20 {
                for x in (w - 4)..w {
                    let idx = f0.idx(x, y);
                    f0.mass[SP_WATER][idx] = 1700;
                    f0.mass[SP_OIL][idx] = 300;
                    f0.flow_e[idx] = 260;
                    f0.set_cell_temp(idx, 36_000);
                }
            }
        }
        fast.snapshot_conservation_baseline();
        slow.snapshot_conservation_baseline();

        let initial_mass = fast.current_accounted_mass() + slow.current_accounted_mass();
        let initial_energy = fast.current_accounted_energy() + slow.current_accounted_energy();

        let mut escrow = BorderFluxEscrow::begin_window(&slow, SeamAxis::EastWest);

        // Run 3 full 2 Hz coarse windows = 15 fast (10 Hz) steps + 3 coarse (2 Hz) steps!
        for coarse_window in 0..3 {
            // 5 fast ticks (10 Hz) WITHOUT stalling on `slow`!
            for sub in 0..5 {
                let (dm, de) = escrow.step_fast_border(&mut fast);
                assert!(dm > 0 && de > 0, "expected active cross-seam flux on sub {sub}");
                let raws_fast = fast.refsim_raw_outputs_all();
                fast.step_3d(&raws_fast);

                // Even MID-CYCLE before `slow` commits, global mass & energy incl. escrow are EXACT!
                let live_mass = fast.current_accounted_mass()
                    + slow.current_accounted_mass()
                    + escrow.escrowed_total_mass();
                let live_energy = fast.current_accounted_energy()
                    + slow.current_accounted_energy()
                    + escrow.escrowed_total_energy();
                assert_eq!(
                    live_mass - initial_mass,
                    0,
                    "mid-cycle mass error != 0 at window={coarse_window}, sub={sub}"
                );
                assert_eq!(
                    live_energy - initial_energy,
                    0,
                    "mid-cycle energy error != 0 at window={coarse_window}, sub={sub}"
                );
            }
            assert_eq!(escrow.fast_substeps_accumulated, 5);
            assert!(!escrow.is_empty());

            // Coarse sync tick (2 Hz): drain escrow atomically into `slow` & step `slow`
            escrow.commit_to_neighbor(&mut slow);
            assert!(escrow.is_empty());
            let raws_slow = slow.refsim_raw_outputs_all();
            slow.step_3d(&raws_slow);

            let post_mass = fast.current_accounted_mass() + slow.current_accounted_mass();
            let post_energy = fast.current_accounted_energy() + slow.current_accounted_energy();
            assert_eq!(post_mass - initial_mass, 0, "post-commit global_mass_error != 0");
            assert_eq!(
                post_energy - initial_energy,
                0,
                "post-commit global_energy_error != 0"
            );
        }

        // Verify water & oil actually arrived inside the 2 Hz slow shard!
        let slow_water: u64 = slow.layers[0].mass[SP_WATER].iter().map(|&v| v as u64).sum();
        let slow_oil: u64 = slow.layers[0].mass[SP_OIL].iter().map(|&v| v as u64).sum();
        assert!(
            slow_water > 1000 && slow_oil > 100,
            "Expected water & oil transferred via async escrow: water={slow_water}, oil={slow_oil}"
        );
    }

    #[test]
    fn test_hysteresis_handoff_and_rolling_clipmap_toroidal_wrap() {
        // 1. 16m Hysteresis Authority Handoff around seam x = 256.0 (deadband [248.0, 264.0])
        let auth = HysteresisAuthority::new(256.0, 2, 2, 16.0);
        let mut owner = auth.shard_for_position(240.0, 128.0);
        assert_eq!(owner, (0, 0));

        // Oscillate aggressively across x = 256.0 inside [249.0, 263.0] -> ZERO ping-pong handoffs!
        let mut handoffs = 0usize;
        for &x in &[255.0, 257.0, 254.0, 260.0, 250.0, 263.5, 248.5, 256.0] {
            let (next, changed) = auth.update_authority(owner, x, 128.0);
            if changed {
                handoffs += 1;
            }
            owner = next;
        }
        assert_eq!(handoffs, 0, "Entity oscillating inside 16m band must NOT ping-pong");
        assert_eq!(owner, (0, 0));

        // Cross decisively past 256 + 8 = 264.0 -> clean handoff to Shard (1, 0)
        let (next, changed) = auth.update_authority(owner, 264.5, 128.0);
        assert!(changed);
        assert_eq!(next, (1, 0));
        owner = next;

        // Stepping back to x = 250.0 (still > 248.0) must STAY on Shard (1, 0)
        let (next, changed) = auth.update_authority(owner, 250.0, 128.0);
        assert!(!changed);
        assert_eq!(next, (1, 0));

        // Crossing decisively below 256 - 8 = 248.0 -> clean handoff back to Shard (0, 0)
        let (next, changed) = auth.update_authority(owner, 247.5, 128.0);
        assert!(changed);
        assert_eq!(next, (0, 0));

        // 2. RollingClipmap256 toroidal brick slot mapper across shard borders & negative coords
        let mut clip = RollingClipmap256::new();
        let u0 = clip.recenter(248.0, 248.0); // cam_brick = (15, 15) -> window [7..23) x [7..23)
        assert_eq!(u0.loaded.len(), 256);
        assert_eq!(u0.evicted.len(), 0);
        assert_eq!(u0.unchanged_count, 0);
        assert!(clip.is_resident(15, 15));
        assert!(clip.is_resident(16, 16)); // already straddling the x=256, y=256 shard corner!

        // Move camera 1 brick east across the x = 256 shard seam -> ONLY 16 bricks evicted & loaded!
        let u1 = clip.recenter(264.0, 248.0); // cam_brick = (16, 15) -> window [8..24) x [7..23)
        assert_eq!(u1.evicted.len(), 16, "Expected exactly 1 column (16 bricks) evicted");
        assert_eq!(u1.loaded.len(), 16, "Expected exactly 1 column (16 bricks) loaded");
        assert_eq!(u1.unchanged_count, 240, "240 bricks (93.75%) must stay resident in-place");
        assert!(!clip.is_resident(7, 15));
        assert!(clip.is_resident(23, 15));
        // Toroidal wrap verification: global brick 23 wraps cleanly into slot 7 (replacing brick 7!)
        assert_eq!(RollingClipmap256::toroidal_slot(23, 15), RollingClipmap256::toroidal_slot(7, 15));
        assert_eq!(clip.resident_at_slot(7, 15), Some((23, 15)));

        // Verify negative global brick coordinates wrap cleanly via rem_euclid
        let u_neg = clip.recenter(-8.0, -8.0); // cam_brick = (-1, -1) -> window [-9..7) x [-9..7)
        assert_eq!(u_neg.loaded.len() + u_neg.unchanged_count, 256);
        assert!(clip.is_resident(-9, -9));
        assert_eq!(RollingClipmap256::toroidal_slot(-1, -1), (15, 15, 255));
        assert_eq!(clip.resident_at_slot(15, 15), Some((-1, -1)));
    }
}
