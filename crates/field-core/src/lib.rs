//! `field-core`: Authoritative multi-species conserved physical world field,
//! 10-step integer Projection Firewall, exact integer RefSim, and Frostmarsh scenes.
//!
//! Matches `FIELD_CONTRACT.md` bit-for-bit in species, reactions, and ONNX tensor layout.

pub const NUM_SPECIES: usize = 10;
pub const NUM_MOBILE: usize = 4;
pub const NUM_REACTIONS: usize = 7;
pub const INPUT_CHANNELS: usize = 18;
pub const OUTPUT_CHANNELS: usize = 17;
pub const CHUNK_SIZE: usize = 16;

pub const C_NOM: u16 = 1000;
pub const H_CAP: u32 = 4000;
pub const T_AMB: i64 = 29_000; // 290.00 K
pub const T_MIN: i64 = 15_000; // 150.00 K
pub const T_MAX: i64 = 150_000; // 1500.00 K

pub const SP_WATER: usize = 0;
pub const SP_ICE: usize = 1;
pub const SP_STEAM: usize = 2;
pub const SP_SOIL: usize = 3;
pub const SP_ROCK: usize = 4;
pub const SP_WOOD: usize = 5;
pub const SP_ASH: usize = 6;
pub const SP_SMOKE: usize = 7;
pub const SP_OIL: usize = 8;
pub const SP_PLANT: usize = 9;

pub const MOBILE_SPECIES: [usize; NUM_MOBILE] = [SP_WATER, SP_STEAM, SP_SMOKE, SP_OIL];

#[derive(Clone, Copy, Debug)]
pub struct SpeciesProps {
    pub name: &'static str,
    pub mobile: bool,
    pub is_liquid: bool,
    pub is_gas: bool,
    pub c: i64,  // heat capacity per mass unit
    pub k: i64,  // thermal conductivity weight
    pub u0: i64, // chemical/latent energy per unit (cK * c)
}

pub const SPECIES: [SpeciesProps; NUM_SPECIES] = [
    SpeciesProps { name: "water", mobile: true,  is_liquid: true,  is_gas: false, c: 42, k: 6,  u0: 0 },
    SpeciesProps { name: "ice",   mobile: false, is_liquid: false, is_gas: false, c: 21, k: 22, u0: -336_000 },
    SpeciesProps { name: "steam", mobile: true,  is_liquid: false, is_gas: true,  c: 20, k: 2,  u0: 840_000 },
    SpeciesProps { name: "soil",  mobile: false, is_liquid: false, is_gas: false, c: 15, k: 10, u0: 0 },
    SpeciesProps { name: "rock",  mobile: false, is_liquid: false, is_gas: false, c: 10, k: 20, u0: 0 },
    SpeciesProps { name: "wood",  mobile: false, is_liquid: false, is_gas: false, c: 17, k: 3,  u0: 450_000 },
    SpeciesProps { name: "ash",   mobile: false, is_liquid: false, is_gas: false, c: 10, k: 2,  u0: 0 },
    SpeciesProps { name: "smoke", mobile: true,  is_liquid: false, is_gas: true,  c: 12, k: 2,  u0: 0 },
    SpeciesProps { name: "oil",   mobile: true,  is_liquid: true,  is_gas: false, c: 20, k: 3,  u0: 750_000 },
    SpeciesProps { name: "plant", mobile: false, is_liquid: false, is_gas: false, c: 25, k: 4,  u0: 300_000 },
];

#[derive(Clone, Copy, Debug)]
pub struct ReactionSpec {
    pub name: &'static str,
    pub nu: [i16; NUM_SPECIES],
    pub t_lo: i64, // inclusive lower temp bound (cK)
    pub t_hi: i64, // inclusive upper temp bound (cK)
}

impl ReactionSpec {
    /// Exact thermal energy released per integer reaction unit `xi = 1`:
    /// `ΔE_th = -Σ_s (nu[s] * u0[s])`
    pub const fn delta_e_th(&self) -> i64 {
        let mut sum = 0i64;
        let mut s = 0;
        while s < NUM_SPECIES {
            sum -= (self.nu[s] as i64) * SPECIES[s].u0;
            s += 1;
        }
        sum
    }
}

pub const REACTIONS: [ReactionSpec; NUM_REACTIONS] = [
    // 0: Freeze (water -> ice, T < 27315)
    ReactionSpec {
        name: "freeze",
        nu: [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        t_lo: T_MIN,
        t_hi: 27_314,
    },
    // 1: Melt (ice -> water, T > 27315)
    ReactionSpec {
        name: "melt",
        nu: [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        t_lo: 27_316,
        t_hi: T_MAX,
    },
    // 2: Boil (water -> steam, T > 37315)
    ReactionSpec {
        name: "boil",
        nu: [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        t_lo: 37_316,
        t_hi: T_MAX,
    },
    // 3: Condense (steam -> water, T < 37000)
    ReactionSpec {
        name: "condense",
        nu: [1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        t_lo: T_MIN,
        t_hi: 36_999,
    },
    // 4: WoodBurn (2 wood -> 1 ash + 1 smoke, T > 52000)
    ReactionSpec {
        name: "wood_burn",
        nu: [0, 0, 0, 0, 0, -2, 1, 1, 0, 0],
        t_lo: 52_001,
        t_hi: T_MAX,
    },
    // 5: OilBurn (1 oil -> 1 smoke, T > 44000)
    ReactionSpec {
        name: "oil_burn",
        nu: [0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
        t_lo: 44_001,
        t_hi: T_MAX,
    },
    // 6: PlantBurn (2 plant -> 1 ash + 1 smoke, T > 46000)
    ReactionSpec {
        name: "plant_burn",
        nu: [0, 0, 0, 0, 0, 0, 1, 1, 0, -2],
        t_lo: 46_001,
        t_hi: T_MAX,
    },
];

/// Raw continuous predictions matching ONNX output `[1, 17, H, W]` in CHW layout.
#[derive(Clone, Debug)]
pub struct RawStepOutput {
    pub width: usize,
    pub height: usize,
    /// Flat CHW tensor of length `17 * height * width`.
    pub data: Vec<f32>,
}

impl RawStepOutput {
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; OUTPUT_CHANNELS * width * height],
        }
    }

    #[inline]
    pub fn get(&self, c: usize, idx: usize) -> f32 {
        self.data[c * self.width * self.height + idx]
    }

    #[inline]
    pub fn set(&mut self, c: usize, idx: usize, v: f32) {
        self.data[c * self.width * self.height + idx] = v;
    }
}

#[derive(Clone, Debug, Default)]
pub struct StepStats {
    pub tick: u64,
    pub edges_evaluated: u64,
    pub edges_projected: u64,
    pub proj_active_pct: f32,
    pub mass_error: i64,
    pub energy_error: i64,
    pub dirty_chunks: Vec<(u16, u16)>,
}

#[derive(Clone, Debug)]
pub struct ChunkPayload {
    pub chunk_id: u32,
    pub cx: u16,
    pub cy: u16,
    /// 2 bytes per cell (512 bytes for 16x16): [display_class, packed_temp_fill]
    pub display_bytes: Vec<u8>,
    /// 1 byte per cell (256 bytes for 16x16):
    /// bit0=walkable, bit1=wall, bit2=burning, bit3=freezing, bit4=water_slow, bits5..7=critter(0..7)
    pub hazard_bytes: Vec<u8>,
}

/// Compositional `A+` wire chunk (4 bytes/cell = 1024 bytes raw per 16x16 chunk).
/// Preserves top-2 species mixtures, mass fraction, flow direction, 8-bit temperature,
/// and exact gameplay hazard/walkability/critter bits without collapsing mixtures to 1 class.
#[derive(Clone, Debug)]
pub struct APlusChunkPayload {
    pub chunk_id: u32,
    pub cx: u16,
    pub cy: u16,
    /// 4 bytes per cell (1024 bytes for 16x16):
    /// [0] = (s1 << 4) | s2          (top-2 species 0..9, 15=empty)
    /// [1] = (s2_frac_4b << 4) | flow_dir_4b
    /// [2] = temp_8b                 (20000..110000 cK mapped to 0..255)
    /// [3] = hazard_critter          (bit0=walkable, bit1=wall, bit2=burning, bit3=freezing, bit4=water_slow, bits5..7=critter)
    pub cells: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct CarvedIceRaft {
    pub center_x: f32,
    pub center_y: f32,
    pub radius: f32,
    pub ice_mass: u32,
    pub cell_count: u32,
}

#[derive(Clone, Debug)]
pub struct Field {
    pub width: usize,
    pub height: usize,
    pub mass: [Vec<u16>; NUM_SPECIES],
    pub energy: Vec<i64>,
    pub flow_e: Vec<i16>,
    pub flow_s: Vec<i16>,
    pub elevation: Vec<u8>,
    pub forcing_energy: Vec<i64>,
    pub forcing_mass: Vec<i32>,
    /// Optional 0..7 critter density overlay populated by Phase 3 neural ecosystem model.
    pub critter_density: Vec<u8>,
    /// Explicit conservation accounts.
    pub atmosphere_mass: [i64; NUM_SPECIES],
    pub atmosphere_energy: i64,
    pub faucet_mass: [i64; NUM_SPECIES],
    pub faucet_energy: i64,
    pub initial_total_mass: i64,
    pub initial_total_energy: i64,
    pub tick: u64,
    prev_display_hash: Vec<u64>,
}

impl Field {
    pub fn new_empty(width: usize, height: usize) -> Self {
        let len = width * height;
        let chunks_x = width.div_ceil(CHUNK_SIZE);
        let chunks_y = height.div_ceil(CHUNK_SIZE);
        let mass = std::array::from_fn(|_| vec![0u16; len]);
        Self {
            width,
            height,
            mass,
            energy: vec![0i64; len],
            flow_e: vec![0i16; len],
            flow_s: vec![0i16; len],
            elevation: vec![10u8; len],
            forcing_energy: vec![0i64; len],
            forcing_mass: vec![0i32; len],
            critter_density: vec![0u8; len],
            atmosphere_mass: [0; NUM_SPECIES],
            atmosphere_energy: 0,
            faucet_mass: [0; NUM_SPECIES],
            faucet_energy: 0,
            initial_total_mass: 0,
            initial_total_energy: 0,
            tick: 0,
            prev_display_hash: vec![u64::MAX; chunks_x * chunks_y],
        }
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    #[inline]
    pub fn cell_volume(&self, idx: usize) -> u32 {
        let mut sum = 0u32;
        for s in 0..NUM_SPECIES {
            sum += self.mass[s][idx] as u32;
        }
        sum
    }

    #[inline]
    pub fn cell_heat_capacity(&self, idx: usize) -> i64 {
        let mut cap = 0i64;
        for s in 0..NUM_SPECIES {
            cap += (self.mass[s][idx] as i64) * SPECIES[s].c;
        }
        cap
    }

    #[inline]
    pub fn cell_temp_ck(&self, idx: usize) -> i64 {
        let c = self.cell_heat_capacity(idx);
        if c <= 0 {
            T_AMB
        } else {
            (self.energy[idx] / c).clamp(T_MIN, T_MAX)
        }
    }

    pub fn set_cell_temp(&mut self, idx: usize, temp_ck: i64) {
        let c = self.cell_heat_capacity(idx);
        self.energy[idx] = c * temp_ck.clamp(T_MIN, T_MAX);
    }

    pub fn snapshot_conservation_baseline(&mut self) {
        self.atmosphere_mass = [0; NUM_SPECIES];
        self.atmosphere_energy = 0;
        self.faucet_mass = [0; NUM_SPECIES];
        self.faucet_energy = 0;
        self.initial_total_mass = self.current_accounted_mass();
        self.initial_total_energy = self.current_accounted_energy();
    }

    pub fn current_accounted_mass(&self) -> i64 {
        let mut total = 0i64;
        for s in 0..NUM_SPECIES {
            let grid_s: i64 = self.mass[s].iter().map(|&v| v as i64).sum();
            total += grid_s + self.atmosphere_mass[s] - self.faucet_mass[s];
        }
        total
    }

    pub fn current_accounted_energy(&self) -> i64 {
        let mut total: i64 = self.energy.iter().sum();
        for s in 0..NUM_SPECIES {
            let grid_s: i64 = self.mass[s].iter().map(|&v| v as i64).sum();
            total += grid_s * SPECIES[s].u0;
        }
        total + self.atmosphere_energy - self.faucet_energy
    }

    /// Build the playable Frostmarsh arena: a N->S river splitting West and East banks,
    /// pine woods, oil marsh pools, rocky chokepoints, and a fragile ice northern crossing.
    pub fn new_frostmarsh(width: usize, height: usize) -> Self {
        let mut f = Self::new_empty(width, height);
        let wf = width as f32;
        let hf = height as f32;

        for y in 0..height {
            for x in 0..width {
                let idx = f.idx(x, y);
                let nx = x as f32 / wf;
                let ny = y as f32 / hf;

                // Winding river center down x ≈ 0.5
                let river_cx = 0.50 + 0.06 * (ny * std::f32::consts::TAU * 1.2).sin();
                let river_half_w = 0.085 + 0.02 * (ny * std::f32::consts::TAU * 2.0).cos();
                let dist_river = (nx - river_cx).abs();

                // Gentle north->south slope so river flows southward
                let slope = ((1.0 - ny) * 8.0) as u8;

                // Perimeter rock walls
                if x < 2 || x + 2 >= width || y < 2 || y + 2 >= height {
                    f.mass[SP_ROCK][idx] = 1200;
                    f.elevation[idx] = 28;
                    f.set_cell_temp(idx, T_AMB);
                    continue;
                }

                if dist_river < river_half_w {
                    // Riverbed
                    f.elevation[idx] = 2 + slope / 2;
                    // Northern fragile ice crossing around y in [0.16, 0.23]
                    if ny > 0.16 && ny < 0.23 {
                        f.mass[SP_ICE][idx] = 950;
                        f.mass[SP_WATER][idx] = 200;
                        f.set_cell_temp(idx, 26_600);
                    } else {
                        f.mass[SP_WATER][idx] = 1100;
                        f.flow_s[idx] = 120;
                        f.set_cell_temp(idx, 28_600);
                    }
                } else {
                    // Banks
                    let bank_lift = ((dist_river - river_half_w) * 35.0).min(12.0) as u8;
                    f.elevation[idx] = 10 + bank_lift + slope / 3;
                    f.mass[SP_SOIL][idx] = 900;

                    // Deterministic pseudo-hash for trees/plants/oil pools
                    let h = ((x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663))) % 1000;

                    // Oil marsh pockets on southern banks
                    let oil_west = ((nx - 0.34).powi(2) + (ny - 0.72).powi(2)).sqrt() < 0.065;
                    let oil_east = ((nx - 0.67).powi(2) + (ny - 0.68).powi(2)).sqrt() < 0.065;
                    // Rock pillars for combat cover
                    let rock_w = ((nx - 0.30).powi(2) + (ny - 0.42).powi(2)).sqrt() < 0.032;
                    let rock_e = ((nx - 0.71).powi(2) + (ny - 0.45).powi(2)).sqrt() < 0.032;
                    // Keep West & East spawn pads clear
                    let spawn_w = ((nx - 0.15).powi(2) + (ny - 0.50).powi(2)).sqrt() < 0.09;
                    let spawn_e = ((nx - 0.85).powi(2) + (ny - 0.50).powi(2)).sqrt() < 0.09;

                    if rock_w || rock_e {
                        f.mass[SP_ROCK][idx] = 1100;
                    } else if (oil_west || oil_east) && !spawn_w && !spawn_e {
                        f.mass[SP_OIL][idx] = 750;
                        f.mass[SP_WATER][idx] = 200;
                    } else if !spawn_w && !spawn_e {
                        // Wood groves near the riverbanks (for burning bridges / forest fires!)
                        let near_bank = dist_river < river_half_w + 0.14;
                        if near_bank && h < 230 {
                            f.mass[SP_WOOD][idx] = 900;
                            f.mass[SP_PLANT][idx] = 250;
                        } else if h < 420 {
                            f.mass[SP_PLANT][idx] = 450;
                        }
                    }

                    f.set_cell_temp(idx, T_AMB);
                }
            }
        }

        f.snapshot_conservation_baseline();
        f
    }

    /// Build the scaled 6-biome **Shattered Reach** arena (designed for 256×256 / 512×512):
    /// - Twin northern glacial cataracts converging into a wide central fjord
    /// - Central Relic Spire island flanked by rock arches
    /// - Two destructible timber bridges (y≈0.30 & y≈0.72) + northern fragile ice crossing (y≈0.15)
    /// - East Obsidian Caldera & elevated volatile oil basins
    /// - West Sunken Briar Marsh (reeds, pine groves, wildlife sanctuary)
    /// - Three Shrine plazas (North Glacier, Central Spire, South Caldera)
    pub fn new_shattered_reach(width: usize, height: usize) -> Self {
        let mut f = Self::new_empty(width, height);
        let wf = width as f32;
        let hf = height as f32;

        for y in 0..height {
            for x in 0..width {
                let idx = f.idx(x, y);
                let nx = x as f32 / wf;
                let ny = y as f32 / hf;

                // Perimeter cliffs
                if x < 3 || x + 3 >= width || y < 3 || y + 3 >= height {
                    f.mass[SP_ROCK][idx] = 1200;
                    f.elevation[idx] = 29;
                    f.set_cell_temp(idx, T_AMB);
                    continue;
                }

                let slope = ((1.0 - ny) * 10.0) as u8;

                // Main winding fjord down x≈0.50 plus a northern tributary merging from x≈0.36
                let main_cx = 0.50 + 0.055 * (ny * std::f32::consts::TAU * 1.4).sin();
                let main_hw = 0.072 + 0.022 * (ny * std::f32::consts::TAU * 2.1).cos();
                let d_main = (nx - main_cx).abs();

                let trib_cx = 0.36 + 0.14 * (ny / 0.38).min(1.0);
                let trib_hw = if ny < 0.38 { 0.036 * (1.0 - ny * 0.4) } else { 0.0 };
                let d_trib = (nx - trib_cx).abs();

                // Central Relic Spire island right in the fjord center
                let d_center_island = ((nx - 0.50).powi(2) + (ny - 0.50).powi(2)).sqrt();
                let on_center_island = d_center_island < 0.058;

                // Three shrine plazas
                let d_shrine_n = ((nx - 0.50).powi(2) + (ny - 0.18).powi(2)).sqrt();
                let d_shrine_s = ((nx - 0.50).powi(2) + (ny - 0.82).powi(2)).sqrt();

                let in_water = !on_center_island && (d_main < main_hw || (ny < 0.38 && d_trib < trib_hw));

                // Destructible timber bridges across the fjord at y≈0.31 and y≈0.71
                let on_bridge_n = (ny - 0.31).abs() < 0.018 && d_main < main_hw + 0.025;
                let on_bridge_s = (ny - 0.71).abs() < 0.018 && d_main < main_hw + 0.025;

                if in_water && (on_bridge_n || on_bridge_s) {
                    f.elevation[idx] = 8 + slope / 3;
                    f.mass[SP_WOOD][idx] = 520; // Walkable timber deck (<650 wall threshold) that burns down!
                    f.mass[SP_WATER][idx] = 350;
                    f.set_cell_temp(idx, 28_600);
                } else if in_water {
                    f.elevation[idx] = 2 + slope / 2;
                    // Northern glacial ice crossing around y in [0.13, 0.21]
                    if ny > 0.13 && ny < 0.21 {
                        f.mass[SP_ICE][idx] = 980;
                        f.mass[SP_WATER][idx] = 180;
                        f.set_cell_temp(idx, 26_400);
                    } else if ny < 0.28 && ((x + y) % 13 == 0) {
                        // Drifting icebergs / brash ice in northern rapids
                        f.mass[SP_ICE][idx] = 520;
                        f.mass[SP_WATER][idx] = 600;
                        f.flow_s[idx] = 110;
                        f.set_cell_temp(idx, 27_100);
                    } else {
                        f.mass[SP_WATER][idx] = 1120;
                        f.flow_s[idx] = 135;
                        f.set_cell_temp(idx, 28_500);
                    }
                } else {
                    // Land / Island / Banks
                    let bank_dist = d_main.min(if ny < 0.38 { d_trib } else { 1.0 });
                    let bank_lift = ((bank_dist - main_hw).max(0.0) * 34.0).min(14.0) as u8;
                    f.elevation[idx] = if on_center_island {
                        12 + ((0.058 - d_center_island) * 90.0) as u8
                    } else {
                        10 + bank_lift + slope / 3
                    };
                    f.mass[SP_SOIL][idx] = 920;

                    let h = ((x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663)) ^ (x * y)) % 1000;

                    // Spawn sanctums kept clear
                    let spawn_w = ((nx - 0.12).powi(2) + (ny - 0.50).powi(2)).sqrt() < 0.075;
                    let spawn_e = ((nx - 0.88).powi(2) + (ny - 0.50).powi(2)).sqrt() < 0.075;
                    let near_shrine = d_shrine_n < 0.045 || d_shrine_s < 0.045 || d_center_island < 0.042;

                    // Obsidian Caldera rock ridges & elevated oil basins (East & South-West)
                    let caldera_ring = {
                        let d = ((nx - 0.74).powi(2) + (ny - 0.64).powi(2)).sqrt();
                        d > 0.068 && d < 0.092 && ny > 0.57
                    };
                    let oil_caldera = ((nx - 0.74).powi(2) + (ny - 0.64).powi(2)).sqrt() < 0.064;
                    let oil_south_w = ((nx - 0.30).powi(2) + (ny - 0.74).powi(2)).sqrt() < 0.055;
                    let oil_north_e = ((nx - 0.70).powi(2) + (ny - 0.28).powi(2)).sqrt() < 0.044;

                    // Strategic tactical granite outcrops for cover
                    let rock_nw = ((nx - 0.26).powi(2) + (ny - 0.32).powi(2)).sqrt() < 0.026;
                    let rock_sw = ((nx - 0.28).powi(2) + (ny - 0.58).powi(2)).sqrt() < 0.024;
                    let rock_ne = ((nx - 0.73).powi(2) + (ny - 0.42).powi(2)).sqrt() < 0.026;
                    let rock_arch_w = ((nx - 0.40).powi(2) + (ny - 0.48).powi(2)).sqrt() < 0.018;
                    let rock_arch_e = ((nx - 0.61).powi(2) + (ny - 0.52).powi(2)).sqrt() < 0.018;

                    if (caldera_ring || rock_nw || rock_sw || rock_ne || rock_arch_w || rock_arch_e)
                        && !spawn_w
                        && !spawn_e
                        && !near_shrine
                    {
                        f.mass[SP_ROCK][idx] = 1150;
                        f.elevation[idx] = (f.elevation[idx] + 8).min(30);
                    } else if (oil_caldera || oil_south_w || oil_north_e)
                        && !spawn_w
                        && !spawn_e
                        && !near_shrine
                    {
                        f.mass[SP_OIL][idx] = 820;
                        f.mass[SP_WATER][idx] = 180;
                    } else if !spawn_w && !spawn_e && !near_shrine {
                        // Alpine pine forests (North) & Briar Marsh groves (West/South)
                        let in_briar_marsh = nx < 0.42 && ny > 0.22 && ny < 0.85;
                        let near_bank = d_main < main_hw + 0.16;
                        if ny < 0.34 && h < 210 {
                            f.mass[SP_WOOD][idx] = 920;
                            f.mass[SP_PLANT][idx] = 220;
                        } else if near_bank && h < 200 {
                            f.mass[SP_WOOD][idx] = 880;
                            f.mass[SP_PLANT][idx] = 260;
                        } else if in_briar_marsh && h < 540 {
                            f.mass[SP_PLANT][idx] = 520;
                            if h < 110 {
                                f.mass[SP_WATER][idx] = 280; // Shallow marsh puddles
                            }
                        } else if h < 360 {
                            f.mass[SP_PLANT][idx] = 420;
                        }
                    }

                    // Northern biome slightly colder ambient
                    let t_init = if ny < 0.25 { 27_800 } else { T_AMB };
                    f.set_cell_temp(idx, t_init);
                }
            }
        }

        f.snapshot_conservation_baseline();
        f
    }

    /// Zero-violation authoritative weather pass (tracked in `atmosphere_*` / `faucet_*`).
    /// `weather_kind`: 0=Clear Frost, 1=Blizzard Front, 2=Ash/Rain Squall, 3=Ember Dusk.
    pub fn apply_weather_tick(&mut self, weather_kind: u8, tick: u64) {
        let len = self.width * self.height;
        // Stride-sample cells each 10 Hz tick for lightweight organic weather drift
        let stride = 7usize;
        let phase = (tick as usize) % stride;
        let mut idx = phase;
        while idx < len {
            let y = idx / self.width;
            let ny = y as f32 / self.height.max(1) as f32;
            let c = self.cell_heat_capacity(idx);
            if c > 0 {
                match weather_kind {
                    1 => {
                        // Blizzard: chills northern/exposed cells & helps river freeze
                        if ny < 0.65 && self.mass[SP_ROCK][idx] < 500 {
                            let cool = ((650.0 * (1.1 - ny)) as i64) * c / 10;
                            let min_e = c * 25_800;
                            let delta = cool.min((self.energy[idx] - min_e).max(0));
                            self.energy[idx] -= delta;
                            self.atmosphere_energy += delta;
                        }
                    }
                    2 => {
                        // Rain / Thaw Squall: gently cools hot wildfires & dampens smoke
                        let t = self.cell_temp_ck(idx);
                        if t > 38_000 {
                            let cool = ((t - 32_000) / 8) * c;
                            let min_e = c * T_AMB;
                            let delta = cool.min((self.energy[idx] - min_e).max(0));
                            self.energy[idx] -= delta;
                            self.atmosphere_energy += delta;
                        }
                    }
                    3 => {
                        // Ember Dusk: warms southern volcanic caldera & oil pools slightly
                        if ny > 0.55 && self.mass[SP_OIL][idx] > 100 {
                            let heat = 320 * c / 10;
                            let cap_e = c * 42_000;
                            let delta = heat.min((cap_e - self.energy[idx]).max(0));
                            self.energy[idx] += delta;
                            self.faucet_energy += delta;
                        }
                    }
                    _ => {}
                }
            }
            idx += stride;
        }
    }

    /// Elemental Combo: Fire hitting Water/Ice triggers an instant billowing Steam Burst
    /// (exact mass & energy conservation via `faucet_energy` for the latent boiling kick).
    pub fn apply_steam_burst(&mut self, cx: i32, cy: i32, radius: i32) -> u32 {
        let r2 = (radius * radius).max(1);
        let mut boiled_cells = 0u32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let x = cx + dx;
                let y = cy + dy;
                if x < 2 || y < 2 || x + 2 >= self.width as i32 || y + 2 >= self.height as i32 {
                    continue;
                }
                let idx = self.idx(x as usize, y as usize);
                let avail = self.mass[SP_WATER][idx].min(320);
                if avail >= 60 {
                    // Convert water -> steam using exact reaction stoichiometry & inject heat to match
                    self.mass[SP_WATER][idx] -= avail;
                    self.mass[SP_STEAM][idx] = self.mass[SP_STEAM][idx].saturating_add(avail);
                    let delta_u0 = (avail as i64) * (SPECIES[SP_STEAM].u0 - SPECIES[SP_WATER].u0);
                    // Account chemical energy jump into faucet_energy so total accounted energy stays 0
                    self.faucet_energy += delta_u0;
                    // Also warm the steam cloud cleanly to ~400K
                    let c = self.cell_heat_capacity(idx).max(1);
                    let target_e = c * 40_000;
                    if target_e > self.energy[idx] {
                        let add_e = target_e - self.energy[idx];
                        self.energy[idx] += add_e;
                        self.faucet_energy += add_e;
                    }
                    boiled_cells += 1;
                }
            }
        }
        boiled_cells
    }

    /// Frost Warden Ultimate (`R`): Glacial Nova freezes a wide radius into walkable ice.
    pub fn apply_glacial_nova(&mut self, cx: i32, cy: i32, radius: i32) {
        self.apply_frost_bolt(cx, cy, radius, 2.8);
    }

    /// Pyro Alchemist Ultimate (`R`): Meteor Impact drops volatile oil + intense wildfire heat.
    pub fn apply_meteor_impact(&mut self, cx: i32, cy: i32, radius: i32) {
        self.apply_oil_splash(cx, cy, radius, 1.35);
        self.apply_fire_bolt(cx, cy, radius + 1, 2.4);
    }

    /// Inject a player Fire Bolt heat burst into the grid (tracked in faucet_energy).
    pub fn apply_fire_bolt(&mut self, cx: i32, cy: i32, radius: i32, strength: f32) {
        let r2 = (radius * radius).max(1) as f32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 1 || y < 1 || x + 1 >= self.width as i32 || y + 1 >= self.height as i32 {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > r2 {
                    continue;
                }
                let falloff = 1.0 - (d2 / r2).sqrt() * 0.65;
                let idx = self.idx(x as usize, y as usize);
                let c = self.cell_heat_capacity(idx).max(15_000);
                // Push temperature well past wood/oil ignition (~750K) at center
                let add_e = ((42_000.0 * falloff * strength) as i64) * c;
                let cap_e = c * T_MAX - self.energy[idx];
                let delta = add_e.min(cap_e.max(0));
                self.energy[idx] += delta;
                self.faucet_energy += delta;
                self.forcing_energy[idx] = self.forcing_energy[idx].saturating_add(delta);
            }
        }
    }

    /// Inject a player Frost Bolt cold burst into the grid (cools water below freezing).
    pub fn apply_frost_bolt(&mut self, cx: i32, cy: i32, radius: i32, strength: f32) {
        let r2 = (radius * radius).max(1) as f32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 1 || y < 1 || x + 1 >= self.width as i32 || y + 1 >= self.height as i32 {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > r2 {
                    continue;
                }
                let falloff = 1.0 - (d2 / r2).sqrt() * 0.55;
                let idx = self.idx(x as usize, y as usize);
                let c = self.cell_heat_capacity(idx).max(15_000);
                // Pull temperature well below 27315 cK (down toward ~22000 cK)
                let sub_e = ((9_500.0 * falloff * strength) as i64) * c;
                let min_e = c * T_MIN;
                let delta = -(sub_e.min((self.energy[idx] - min_e).max(0)));
                self.energy[idx] += delta;
                self.faucet_energy += delta;
                self.forcing_energy[idx] = self.forcing_energy[idx].saturating_add(delta);
            }
        }
    }

    /// Splash flammable oil onto the ground/water (tracked in faucet accounts).
    pub fn apply_oil_splash(&mut self, cx: i32, cy: i32, radius: i32, strength: f32) {
        let r2 = (radius * radius).max(1) as f32;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = cx + dx;
                let y = cy + dy;
                if x < 2 || y < 2 || x + 2 >= self.width as i32 || y + 2 >= self.height as i32 {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > r2 {
                    continue;
                }
                let idx = self.idx(x as usize, y as usize);
                if self.mass[SP_ROCK][idx] >= 500 {
                    continue;
                }
                let room = H_CAP.saturating_sub(self.cell_volume(idx));
                let add = ((450.0 * strength) as u32).min(room) as u16;
                if add > 0 {
                    self.mass[SP_OIL][idx] += add;
                    let e_add = (add as i64) * SPECIES[SP_OIL].c * T_AMB;
                    self.energy[idx] += e_add;
                    self.faucet_mass[SP_OIL] += add as i64;
                    self.faucet_energy += e_add + (add as i64) * SPECIES[SP_OIL].u0;
                    self.forcing_mass[idx] += add as i32;
                }
            }
        }
    }

    /// Atomically carve solid ice cells out of the grid into a floating `CellBody` raft.
    /// Exact conservation: debited ice + energy moved into `atmosphere_*` (acting as body escrow).
    pub fn carve_ice_raft(&mut self, cx: i32, cy: i32, radius: i32) -> Option<CarvedIceRaft> {
        let mut total_ice = 0u32;
        let mut cells = 0u32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let r2 = radius * radius;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > r2 {
                    continue;
                }
                let x = cx + dx;
                let y = cy + dy;
                if x < 2 || y < 2 || x + 2 >= self.width as i32 || y + 2 >= self.height as i32 {
                    continue;
                }
                let idx = self.idx(x as usize, y as usize);
                let ice = self.mass[SP_ICE][idx];
                if ice >= 350 {
                    let c_cell = self.cell_heat_capacity(idx).max(1);
                    let e_ice = (ice as i64) * SPECIES[SP_ICE].c * (self.energy[idx] / c_cell);
                    self.mass[SP_ICE][idx] = 0;
                    self.energy[idx] -= e_ice;
                    // Escrow into ledger body account
                    self.atmosphere_mass[SP_ICE] += ice as i64;
                    self.atmosphere_energy += e_ice + (ice as i64) * SPECIES[SP_ICE].u0;
                    total_ice += ice as u32;
                    cells += 1;
                    sum_x += x as f32;
                    sum_y += y as f32;
                }
            }
        }

        if cells >= 4 {
            Some(CarvedIceRaft {
                center_x: sum_x / cells as f32,
                center_y: sum_y / cells as f32,
                radius: (cells as f32 / std::f32::consts::PI).sqrt().max(1.5),
                ice_mass: total_ice,
                cell_count: cells,
            })
        } else {
            None
        }
    }

    /// Encode current state into `[1, 18, H, W]` CHW float32 tensor matching `FIELD_CONTRACT.md`.
    pub fn encode_input_tensor(&self) -> Vec<f32> {
        let len = self.width * self.height;
        let mut out = vec![0.0f32; INPUT_CHANNELS * len];
        for idx in 0..len {
            let mut tot = 0u32;
            for s in 0..NUM_SPECIES {
                let m = self.mass[s][idx];
                tot += m as u32;
                out[s * len + idx] = (m as f32) / (C_NOM as f32);
            }
            let t = self.cell_temp_ck(idx);
            out[10 * len + idx] = ((t - T_AMB) as f32) / 10_000.0;
            out[11 * len + idx] = (1.0 + (tot as f32) / (C_NOM as f32)).ln();
            out[12 * len + idx] = (self.flow_e[idx] as f32) / (C_NOM as f32);
            out[13 * len + idx] = (self.flow_s[idx] as f32) / (C_NOM as f32);
            out[14 * len + idx] = (self.elevation[idx] as f32) / 31.0;
            out[15 * len + idx] = if self.mass[SP_ROCK][idx] >= 500 { 1.0 } else { 0.0 };
            out[16 * len + idx] = ((self.forcing_energy[idx] as f32) / 1e8).clamp(-10.0, 10.0);
            out[17 * len + idx] = ((self.forcing_mass[idx] as f32) / (C_NOM as f32)).clamp(-4.0, 4.0);
        }
        out
    }

    /// Exact deterministic `RefSim` producing raw outputs in the identical format as the ONNX model.
    pub fn refsim_raw_outputs(&self) -> RawStepOutput {
        let w = self.width;
        let h = self.height;
        let mut out = RawStepOutput::zeros(w, h);

        // Precompute temperatures & liquid heads
        let len = w * h;
        let mut temps = vec![T_AMB; len];
        let mut heads = vec![0.0f32; len];
        let mut blocked = vec![false; len];
        let mut k_eff_cell = vec![1.0f32; len];

        for idx in 0..len {
            temps[idx] = self.cell_temp_ck(idx);
            let liq = (self.mass[SP_WATER][idx] as f32) + (self.mass[SP_OIL][idx] as f32);
            heads[idx] = (self.elevation[idx] as f32) * 120.0 + liq;
            blocked[idx] = self.mass[SP_ROCK][idx] >= 500 || self.mass[SP_ICE][idx] >= 700;

            let mut k_sum = 0.0f32;
            let mut m_sum = 0.0f32;
            for s in 0..NUM_SPECIES {
                let m = self.mass[s][idx] as f32;
                k_sum += m * (SPECIES[s].k as f32);
                m_sum += m;
            }
            k_eff_cell[idx] = if m_sum > 0.0 { k_sum / m_sum } else { 4.0 };
        }

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;

                // East edge (x -> x+1)
                if x + 1 < w {
                    let nb = idx + 1;
                    self.compute_edge_refsim(
                        idx,
                        nb,
                        self.flow_e[idx] as f32,
                        &temps,
                        &heads,
                        &blocked,
                        &k_eff_cell,
                        true,
                        &mut out,
                    );
                }

                // South edge (y -> y+1)
                if y + 1 < h {
                    let nb = idx + w;
                    self.compute_edge_refsim(
                        idx,
                        nb,
                        self.flow_s[idx] as f32,
                        &temps,
                        &heads,
                        &blocked,
                        &k_eff_cell,
                        false,
                        &mut out,
                    );
                }

                // Reactions (in 100.0 reaction units)
                let t = temps[idx];
                // 0: Freeze
                if t < REACTIONS[0].t_hi && self.mass[SP_WATER][idx] > 0 {
                    let rate = ((27_315 - t) as f32 * 0.045).min(self.mass[SP_WATER][idx] as f32 * 0.35);
                    out.set(10, idx, rate / 100.0);
                }
                // 1: Melt
                if t > REACTIONS[1].t_lo && self.mass[SP_ICE][idx] > 0 {
                    let rate = ((t - 27_315) as f32 * 0.030).min(self.mass[SP_ICE][idx] as f32 * 0.30);
                    out.set(11, idx, rate / 100.0);
                }
                // 2: Boil
                if t > REACTIONS[2].t_lo && self.mass[SP_WATER][idx] > 0 {
                    let rate = ((t - 37_315) as f32 * 0.015).min(self.mass[SP_WATER][idx] as f32 * 0.20);
                    out.set(12, idx, rate / 100.0);
                }
                // 3: Condense
                if t < REACTIONS[3].t_hi && self.mass[SP_STEAM][idx] > 0 {
                    let rate = ((37_000 - t) as f32 * 0.012).min(self.mass[SP_STEAM][idx] as f32 * 0.25);
                    out.set(13, idx, rate / 100.0);
                }
                // 4: WoodBurn (2 wood -> 1 ash + 1 smoke)
                if t > REACTIONS[4].t_lo && self.mass[SP_WOOD][idx] >= 2 {
                    let rate = ((t - 52_000) as f32 * 0.004).clamp(4.0, 45.0);
                    out.set(14, idx, rate / 100.0);
                }
                // 5: OilBurn (1 oil -> 1 smoke)
                if t > REACTIONS[5].t_lo && self.mass[SP_OIL][idx] >= 1 {
                    let rate = ((t - 44_000) as f32 * 0.008).clamp(8.0, 85.0);
                    out.set(15, idx, rate / 100.0);
                }
                // 6: PlantBurn (2 plant -> 1 ash + 1 smoke)
                if t > REACTIONS[6].t_lo && self.mass[SP_PLANT][idx] >= 2 {
                    let rate = ((t - 46_000) as f32 * 0.006).clamp(6.0, 60.0);
                    out.set(16, idx, rate / 100.0);
                }
            }
        }

        out
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_edge_refsim(
        &self,
        a: usize,
        b: usize,
        prev_flow: f32,
        temps: &[i64],
        heads: &[f32],
        blocked: &[bool],
        k_eff_cell: &[f32],
        is_east: bool,
        out: &mut RawStepOutput,
    ) {
        let base_c = if is_east { 0 } else { 4 };
        let heat_c = if is_east { 8 } else { 9 };

        // Transport of mobile species
        if !blocked[a] && !blocked[b] {
            let dh = heads[a] - heads[b];
            // Water (mobile slot 0)
            let w_flux = 0.16 * dh + 0.25 * prev_flow;
            let w_clamped = if w_flux > 0.0 {
                w_flux.min(self.mass[SP_WATER][a] as f32 * 0.22)
            } else {
                w_flux.max(-(self.mass[SP_WATER][b] as f32 * 0.22))
            };
            out.set(base_c, a, w_clamped / (C_NOM as f32));

            // Steam (mobile slot 1) — gas diffusion
            let st_flux = 0.18 * (self.mass[SP_STEAM][a] as f32 - self.mass[SP_STEAM][b] as f32);
            out.set(base_c + 1, a, st_flux / (C_NOM as f32));

            // Smoke (mobile slot 2) — gas diffusion + gentle northward/eastward drift
            let drift = if is_east { 4.0 } else { -3.0 };
            let sm_a = self.mass[SP_SMOKE][a] as f32;
            let sm_b = self.mass[SP_SMOKE][b] as f32;
            let sm_flux = 0.18 * (sm_a - sm_b) + if sm_a + sm_b > 20.0 { drift } else { 0.0 };
            out.set(base_c + 2, a, sm_flux / (C_NOM as f32));

            // Oil (mobile slot 3) — viscous liquid spreading on head gradient
            let o_flux = 0.09 * dh + 0.10 * (self.mass[SP_OIL][a] as f32 - self.mass[SP_OIL][b] as f32);
            let o_clamped = if o_flux > 0.0 {
                o_flux.min(self.mass[SP_OIL][a] as f32 * 0.18)
            } else {
                o_flux.max(-(self.mass[SP_OIL][b] as f32 * 0.18))
            };
            out.set(base_c + 3, a, o_clamped / (C_NOM as f32));
        }

        // Heat conduction along edge (works through solids/ice/wood too so fires spread!)
        let dt = (temps[a] - temps[b]) as f32;
        if dt.abs() > 1.0 {
            let k_edge = 0.5 * (k_eff_cell[a] + k_eff_cell[b]);
            let ca = self.cell_heat_capacity(a) as f32;
            let cb = self.cell_heat_capacity(b) as f32;
            if ca > 0.0 && cb > 0.0 {
                let c_harm = (ca * cb) / (ca + cb);
                // Smooth conduction factor bounded by 0.22 (< 0.25 max-principle cap)
                let alpha = (0.012 * k_edge).min(0.22);
                let g = alpha * dt * c_harm;
                out.set(heat_c, a, g / 1e6);
            }
        }
    }

    /// 10-step exact integer Projection Firewall.
    /// Guarantees: exact per-species & energy conservation, m >= 0, V <= H_CAP,
    /// discrete 2nd-law heat conduction, and immunity to NaN/Inf/adversarial neural outputs.
    pub fn project_and_apply(&mut self, raw: &RawStepOutput) -> StepStats {
        let w = self.width;
        let h = self.height;
        let len = w * h;
        assert_eq!(raw.width, w);
        assert_eq!(raw.height, h);

        // Snapshot tick-start state for Jacobi edge-local bounds
        let mut temps = vec![T_AMB; len];
        let mut caps = vec![0i64; len];
        let mut vols = vec![0u32; len];
        let mut rock_wall = vec![false; len];

        for idx in 0..len {
            caps[idx] = self.cell_heat_capacity(idx);
            vols[idx] = self.cell_volume(idx);
            temps[idx] = if caps[idx] > 0 {
                (self.energy[idx] / caps[idx]).clamp(T_MIN, T_MAX)
            } else {
                T_AMB
            };
            rock_wall[idx] = self.mass[SP_ROCK][idx] >= 500;
        }

        let mut edges_evaluated = 0u64;
        let mut edges_projected = 0u64;

        // Projected integer fluxes stored per owned edge
        let mut qe = vec![[0i32; NUM_MOBILE]; len];
        let mut qs = vec![[0i32; NUM_MOBILE]; len];
        let mut ge = vec![0i64; len];
        let mut gs = vec![0i64; len];

        // Steps 1-7: Edge-local sanitization, quantization, positivity, ceiling, 2nd-law heat, advection
        let mut adv_out_e = vec![0i64; len];
        let mut adv_in_e = vec![0i64; len];

        for y in 0..h {
            for x in 0..w {
                let a = y * w + x;
                for dir in 0..2 {
                    let (b, is_east) = if dir == 0 {
                        if x + 1 >= w {
                            continue;
                        }
                        (a + 1, true)
                    } else {
                        if y + 1 >= h {
                            continue;
                        }
                        (a + w, false)
                    };
                    edges_evaluated += 1;

                    let base_c = if is_east { 0 } else { 4 };
                    let heat_c = if is_east { 8 } else { 9 };
                    let wall_edge = rock_wall[a] || rock_wall[b];
                    let mut edge_clamped = false;

                    let mut q = [0i32; NUM_MOBILE];
                    if !wall_edge {
                        // Step 2-3: Sanitize, quantize, edge-local positivity (<= floor(m_donor / 4))
                        for m in 0..NUM_MOBILE {
                            let s = MOBILE_SPECIES[m];
                            let v = sanitize_f32(raw.get(base_c + m, a));
                            let q_raw = round_ties_even_i32(v * (C_NOM as f32));
                            let clamped = if q_raw > 0 {
                                let lim = (self.mass[s][a] / 4) as i32;
                                if q_raw > lim {
                                    if q_raw - lim > 10 {
                                        edge_clamped = true;
                                    }
                                    lim
                                } else {
                                    q_raw
                                }
                            } else if q_raw < 0 {
                                let lim = (self.mass[s][b] / 4) as i32;
                                if -q_raw > lim {
                                    if (-q_raw) - lim > 10 {
                                        edge_clamped = true;
                                    }
                                    -lim
                                } else {
                                    q_raw
                                }
                            } else {
                                0
                            };
                            q[m] = clamped;
                        }

                        // Step 4: Edge-local volume ceiling on both directions
                        // Positive flows enter `b`; negative flows enter `a`.
                        let pos_tot: i64 = q.iter().filter(|&&v| v > 0).map(|&v| v as i64).sum();
                        let cap_b = (H_CAP.saturating_sub(vols[b]) / 4) as i64;
                        if pos_tot > cap_b {
                            edge_clamped = true;
                            for v in &mut q {
                                if *v > 0 {
                                    *v = ((*v as i64) * cap_b / pos_tot) as i32;
                                }
                            }
                        }

                        let neg_tot: i64 = q.iter().filter(|&&v| v < 0).map(|&v| (-v) as i64).sum();
                        let cap_a = (H_CAP.saturating_sub(vols[a]) / 4) as i64;
                        if neg_tot > cap_a {
                            edge_clamped = true;
                            for v in &mut q {
                                if *v < 0 {
                                    *v = -(((-*v) as i64) * cap_a / neg_tot) as i32;
                                }
                            }
                        }
                    } else {
                        // Any nonzero proposal across a wall is clamped
                        for m in 0..NUM_MOBILE {
                            if sanitize_f32(raw.get(base_c + m, a)).abs() > 1e-3 {
                                edge_clamped = true;
                            }
                        }
                    }

                    // Step 5: Discrete 2nd-law heat conduction
                    let g_raw_f = sanitize_f32(raw.get(heat_c, a));
                    let g_raw = (g_raw_f as f64 * 1e6).round() as i64;
                    let dt = temps[a] - temps[b];
                    let g_clamped = if caps[a] > 0 && caps[b] > 0 && dt != 0 {
                        let max_g = ((dt.unsigned_abs() as u128)
                            * (caps[a] as u128)
                            * (caps[b] as u128)
                            / (4 * ((caps[a] + caps[b]) as u128)))
                            as i64;
                        if dt > 0 {
                            let c = g_raw.clamp(0, max_g);
                            if (g_raw - c).abs() > 250_000 {
                                edge_clamped = true;
                            }
                            c
                        } else {
                            let c = g_raw.clamp(-max_g, 0);
                            if (g_raw - c).abs() > 250_000 {
                                edge_clamped = true;
                            }
                            c
                        }
                    } else {
                        if g_raw.abs() > 250_000 {
                            edge_clamped = true;
                        }
                        0
                    };

                    // Step 6: Deterministic integer advected thermal energy
                    for m in 0..NUM_MOBILE {
                        let qv = q[m] as i64;
                        if qv == 0 {
                            continue;
                        }
                        let s = MOBILE_SPECIES[m];
                        if qv > 0 && caps[a] > 0 {
                            let e_adv = ((qv as i128) * (SPECIES[s].c as i128) * (self.energy[a] as i128)
                                / (caps[a] as i128)) as i64;
                            adv_out_e[a] += e_adv;
                            adv_in_e[b] += e_adv;
                        } else if qv < 0 && caps[b] > 0 {
                            let q_abs = -qv;
                            let e_adv = ((q_abs as i128) * (SPECIES[s].c as i128) * (self.energy[b] as i128)
                                / (caps[b] as i128)) as i64;
                            adv_out_e[b] += e_adv;
                            adv_in_e[a] += e_adv;
                        }
                    }

                    if edge_clamped {
                        edges_projected += 1;
                    }

                    if is_east {
                        qe[a] = q;
                        ge[a] = g_clamped;
                    } else {
                        qs[a] = q;
                        gs[a] = g_clamped;
                    }
                }
            }
        }

        // Step 7-8: Apply mass transport, advected energy, and conduction (with final positivity guard)
        for y in 0..h {
            for x in 0..w {
                let a = y * w + x;
                // East edge
                if x + 1 < w {
                    let b = a + 1;
                    for m in 0..NUM_MOBILE {
                        let qv = qe[a][m];
                        if qv != 0 {
                            let s = MOBILE_SPECIES[m];
                            self.mass[s][a] = (self.mass[s][a] as i32 - qv) as u16;
                            self.mass[s][b] = (self.mass[s][b] as i32 + qv) as u16;
                        }
                    }
                    // Guard conduction so donor never goes below T_MIN * remaining_C
                    let mut g = ge[a];
                    if g > 0 {
                        let avail = (self.energy[a] - adv_out_e[a]).max(0) / 4;
                        g = g.min(avail);
                    } else if g < 0 {
                        let avail = (self.energy[b] - adv_out_e[b]).max(0) / 4;
                        g = g.max(-avail);
                    }
                    self.energy[a] -= g;
                    self.energy[b] += g;
                    self.flow_e[a] = (qe[a][0] + qe[a][3]).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                } else {
                    self.flow_e[a] = 0;
                }

                // South edge
                if y + 1 < h {
                    let b = a + w;
                    for m in 0..NUM_MOBILE {
                        let qv = qs[a][m];
                        if qv != 0 {
                            let s = MOBILE_SPECIES[m];
                            self.mass[s][a] = (self.mass[s][a] as i32 - qv) as u16;
                            self.mass[s][b] = (self.mass[s][b] as i32 + qv) as u16;
                        }
                    }
                    let mut g = gs[a];
                    if g > 0 {
                        let avail = (self.energy[a] - adv_out_e[a]).max(0) / 4;
                        g = g.min(avail);
                    } else if g < 0 {
                        let avail = (self.energy[b] - adv_out_e[b]).max(0) / 4;
                        g = g.max(-avail);
                    }
                    self.energy[a] -= g;
                    self.energy[b] += g;
                    self.flow_s[a] = (qs[a][0] + qs[a][3]).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                } else {
                    self.flow_s[a] = 0;
                }
            }
        }
        // Apply exact advected energy deltas (sum(adv_in) == sum(adv_out) identically)
        for idx in 0..len {
            self.energy[idx] = self.energy[idx] - adv_out_e[idx] + adv_in_e[idx];
        }

        // Step 9: Stoichiometric reactions on post-transport masses
        for idx in 0..len {
            let t_now = self.cell_temp_ck(idx);
            let mut xi = [0i32; NUM_REACTIONS];
            for r in 0..NUM_REACTIONS {
                let spec = &REACTIONS[r];
                if t_now < spec.t_lo || t_now > spec.t_hi {
                    continue;
                }
                let raw_xi = sanitize_f32(raw.get(10 + r, idx)).max(0.0);
                let mut x_int = (raw_xi * 100.0).floor().min(2000.0) as i32;
                if x_int <= 0 {
                    continue;
                }
                // Endothermic energy guard: don't drop energy below T_MIN * C
                let de = spec.delta_e_th();
                if de < 0 {
                    let c_now = self.cell_heat_capacity(idx);
                    let avail = (self.energy[idx] - c_now * T_MIN).max(0);
                    let max_xi = (avail / (-de)) as i32;
                    x_int = x_int.min(max_xi);
                } else if de > 0 {
                    // Exothermic upper temp cap guard
                    let c_now = self.cell_heat_capacity(idx).max(10_000);
                    let room = (c_now * T_MAX - self.energy[idx]).max(0);
                    let max_xi = (room / de) as i32;
                    x_int = x_int.min(max_xi);
                }
                xi[r] = x_int.max(0);
            }

            // Limiting-reagent scaling across species
            for s in 0..NUM_SPECIES {
                let demand: i64 = (0..NUM_REACTIONS)
                    .filter(|&r| REACTIONS[r].nu[s] < 0)
                    .map(|r| (-REACTIONS[r].nu[s] as i64) * (xi[r] as i64))
                    .sum();
                let avail = self.mass[s][idx] as i64;
                if demand > avail && demand > 0 {
                    for r in 0..NUM_REACTIONS {
                        if REACTIONS[r].nu[s] < 0 {
                            xi[r] = ((xi[r] as i64) * avail / demand) as i32;
                        }
                    }
                }
            }

            // Apply integer reactions (mass + exact chemical/latent energy)
            for r in 0..NUM_REACTIONS {
                let x_int = xi[r];
                if x_int <= 0 {
                    continue;
                }
                let spec = &REACTIONS[r];
                for s in 0..NUM_SPECIES {
                    let dm = (spec.nu[s] as i32) * x_int;
                    if dm != 0 {
                        self.mass[s][idx] = (self.mass[s][idx] as i32 + dm) as u16;
                    }
                }
                self.energy[idx] += spec.delta_e_th() * (x_int as i64);
            }
        }

        // Gentle ambient thermal exchange + smoke/steam dissipation into explicit atmosphere accounts
        for idx in 0..len {
            let c_now = self.cell_heat_capacity(idx);
            if c_now > 0 {
                let target_e = c_now * T_AMB;
                let diff = self.energy[idx] - target_e;
                // 2.5% per tick relaxation toward ambient (so fires cool down & ice slowly warms)
                let cool = diff / 40;
                self.energy[idx] -= cool;
                self.atmosphere_energy += cool;
            }
            // Smoke dissipates upward into atmosphere (10% per tick)
            let sm = self.mass[SP_SMOKE][idx];
            if sm > 0 {
                let diss = ((sm / 10).max(1)).min(sm);
                let c_cell = self.cell_heat_capacity(idx).max(1);
                let e_diss = (diss as i64) * SPECIES[SP_SMOKE].c * (self.energy[idx] / c_cell);
                self.mass[SP_SMOKE][idx] -= diss;
                self.energy[idx] -= e_diss;
                self.atmosphere_mass[SP_SMOKE] += diss as i64;
                self.atmosphere_energy += e_diss + (diss as i64) * SPECIES[SP_SMOKE].u0;
            }
            // Clear per-tick forcing channels
            self.forcing_energy[idx] = 0;
            self.forcing_mass[idx] = 0;
        }

        // Maintain river faucet at north edge (y=2) and drain at south edge (y=h-3)
        // explicitly tracked in faucet_mass/faucet_energy for exact accounting!
        self.apply_river_boundary_faucets();

        self.tick += 1;

        let dirty_chunks = self.compute_dirty_chunks();
        let proj_active_pct = if edges_evaluated > 0 {
            (edges_projected as f32) * 100.0 / (edges_evaluated as f32)
        } else {
            0.0
        };

        StepStats {
            tick: self.tick,
            edges_evaluated,
            edges_projected,
            proj_active_pct,
            mass_error: self.current_accounted_mass() - self.initial_total_mass,
            energy_error: self.current_accounted_energy() - self.initial_total_energy,
            dirty_chunks,
        }
    }

    fn apply_river_boundary_faucets(&mut self) {
        if self.height < 8 || self.width < 16 {
            return;
        }
        let wf = self.width as f32;
        let y_top = 2usize;
        let y_bot = self.height - 3;

        for x in 2..(self.width - 2) {
            let nx = x as f32 / wf;
            // North river inflow (main fjord + western glacial tributary in Shattered Reach)
            let in_north_inflow = (nx - 0.50).abs() < 0.085 || (nx - 0.36).abs() < 0.036;
            if in_north_inflow {
                let idx = self.idx(x, y_top);
                if self.mass[SP_ROCK][idx] < 500 && self.mass[SP_ICE][idx] < 500 {
                    let cur = self.mass[SP_WATER][idx];
                    if cur < 1100 {
                        let add = (1100 - cur).min(120);
                        let room = H_CAP.saturating_sub(self.cell_volume(idx)) as u16;
                        let actual = add.min(room);
                        if actual > 0 {
                            self.mass[SP_WATER][idx] += actual;
                            let e_add = (actual as i64) * SPECIES[SP_WATER].c * 28_600;
                            self.energy[idx] += e_add;
                            self.faucet_mass[SP_WATER] += actual as i64;
                            self.faucet_energy += e_add;
                        }
                    }
                }
            }
            // South river outflow
            if (nx - 0.50).abs() < 0.12 {
                let idx = self.idx(x, y_bot);
                let cur = self.mass[SP_WATER][idx];
                if cur > 1100 {
                    let rem = (cur - 1100).min(140);
                    let c_cell = self.cell_heat_capacity(idx).max(1);
                    let e_rem = (rem as i64) * SPECIES[SP_WATER].c * (self.energy[idx] / c_cell);
                    self.mass[SP_WATER][idx] -= rem;
                    self.energy[idx] -= e_rem;
                    self.atmosphere_mass[SP_WATER] += rem as i64;
                    self.atmosphere_energy += e_rem;
                }
            }
        }
    }

    /// Encode a single cell into 2 display bytes + 1 hazard/walkability byte.
    pub fn encode_cell_display_and_hazard(&self, idx: usize) -> ([u8; 2], u8) {
        let t = self.cell_temp_ck(idx);
        let water = self.mass[SP_WATER][idx];
        let ice = self.mass[SP_ICE][idx];
        let steam = self.mass[SP_STEAM][idx];
        let soil = self.mass[SP_SOIL][idx];
        let rock = self.mass[SP_ROCK][idx];
        let wood = self.mass[SP_WOOD][idx];
        let ash = self.mass[SP_ASH][idx];
        let smoke = self.mass[SP_SMOKE][idx];
        let oil = self.mass[SP_OIL][idx];
        let plant = self.mass[SP_PLANT][idx];

        // Visual display class (0..15):
        // 0: empty/dark, 1: water, 2: ice, 3: steam, 4: soil, 5: mud (soil+water),
        // 6: rock, 7: wood, 8: ash, 9: smoke, 10: oil, 11: plant, 12: fire (hot burning!)
        let is_burning = t > 44_000 && (wood > 20 || oil > 15 || plant > 20 || smoke > 20);
        let disp_class: u8 = if rock >= 500 {
            6
        } else if is_burning {
            12
        } else if ice >= 350 {
            2
        } else if oil >= 150 {
            10
        } else if wood >= 200 {
            7
        } else if water >= 250 && soil >= 250 {
            5 // mud!
        } else if water >= 180 {
            1
        } else if plant >= 150 {
            11
        } else if steam >= 120 {
            3
        } else if smoke >= 80 {
            9
        } else if ash >= 80 {
            8
        } else if soil >= 100 {
            4
        } else {
            0
        };

        // Pack temperature bucket (high 4 bits, 0..15) + fill intensity (low 4 bits, 0..15)
        let temp_bucket = (((t - 22_000).clamp(0, 60_000) * 15) / 60_000) as u8;
        let fill_bucket = ((self.cell_volume(idx).min(2000) * 15) / 2000) as u8;
        let packed = (temp_bucket << 4) | (fill_bucket & 0x0f);

        // Hazard/walkability byte:
        // bit0 = walkable, bit1 = wall, bit2 = burning, bit3 = freezing, bit4 = water_slow, bits5..7 = critter(0..7)
        let is_wall = rock >= 500 || wood >= 650;
        let is_deep_water = water >= 450 && ice < 350 && soil < 250;
        let walkable = !is_wall && !is_deep_water;
        let is_freezing = t < 26_500;
        let water_slow = water >= 200 && ice < 350;
        let critter = self.critter_density[idx].min(7);

        let mut hz = 0u8;
        if walkable {
            hz |= 1 << 0;
        }
        if is_wall {
            hz |= 1 << 1;
        }
        if is_burning {
            hz |= 1 << 2;
        }
        if is_freezing {
            hz |= 1 << 3;
        }
        if water_slow {
            hz |= 1 << 4;
        }
        hz |= critter << 5;

        ([disp_class, packed], hz)
    }

    pub fn encode_chunk(&self, cx: u16, cy: u16) -> ChunkPayload {
        let chunks_x = self.width.div_ceil(CHUNK_SIZE) as u32;
        let chunk_id = (cy as u32) * chunks_x + (cx as u32);
        let mut display_bytes = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE * 2);
        let mut hazard_bytes = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE);

        let x0 = (cx as usize) * CHUNK_SIZE;
        let y0 = (cy as usize) * CHUNK_SIZE;
        for dy in 0..CHUNK_SIZE {
            for dx in 0..CHUNK_SIZE {
                let x = (x0 + dx).min(self.width - 1);
                let y = (y0 + dy).min(self.height - 1);
                let (d, hz) = self.encode_cell_display_and_hazard(self.idx(x, y));
                display_bytes.push(d[0]);
                display_bytes.push(d[1]);
                hazard_bytes.push(hz);
            }
        }
        ChunkPayload {
            chunk_id,
            cx,
            cy,
            display_bytes,
            hazard_bytes,
        }
    }

    /// Encode a single cell into the 4-byte compositional `A+` wire format.
    pub fn encode_cell_aplus(&self, idx: usize) -> [u8; 4] {
        // Find top-2 species by mass
        let mut s1: u8 = 15;
        let mut m1: u16 = 0;
        let mut s2: u8 = 15;
        let mut m2: u16 = 0;
        for s in 0..NUM_SPECIES {
            let m = self.mass[s][idx];
            if m > m1 {
                s2 = s1;
                m2 = m1;
                s1 = s as u8;
                m1 = m;
            } else if m > m2 {
                s2 = s as u8;
                m2 = m;
            }
        }
        let s2_frac_4b: u8 = if m1 + m2 > 0 && s2 < 15 {
            (((m2 as u32) * 15 + ((m1 + m2) as u32) / 2) / ((m1 + m2) as u32)).min(15) as u8
        } else {
            0
        };

        // Quantize flow vector (flow_e, flow_s) into 4-bit direction+speed (0=still, 1..8 slow, 9..15 fast)
        let fx = self.flow_e[idx] as f32;
        let fy = self.flow_s[idx] as f32;
        let fmag = (fx * fx + fy * fy).sqrt();
        let flow_dir_4b: u8 = if fmag < 12.0 {
            0
        } else {
            let ang = fy.atan2(fx); // [-pi, pi]
            let oct = (((ang + std::f32::consts::PI) / (std::f32::consts::TAU / 8.0) + 0.5) as i32
                + 4)
                .rem_euclid(8) as u8;
            let fast = if fmag >= 80.0 { 8u8 } else { 0u8 };
            if fast != 0 && oct == 7 {
                15
            } else {
                (oct + 1 + fast).min(15)
            }
        };

        let t = self.cell_temp_ck(idx);
        let temp_8b = (((t - 20_000).clamp(0, 90_000) * 255) / 90_000) as u8;

        let (_, hz) = self.encode_cell_display_and_hazard(idx);
        [((s1 & 0x0f) << 4) | (s2 & 0x0f), ((s2_frac_4b & 0x0f) << 4) | (flow_dir_4b & 0x0f), temp_8b, hz]
    }

    pub fn encode_aplus_chunk(&self, cx: u16, cy: u16) -> APlusChunkPayload {
        let chunks_x = self.width.div_ceil(CHUNK_SIZE) as u32;
        let chunk_id = (cy as u32) * chunks_x + (cx as u32);
        let mut cells = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE * 4);
        let x0 = (cx as usize) * CHUNK_SIZE;
        let y0 = (cy as usize) * CHUNK_SIZE;
        for dy in 0..CHUNK_SIZE {
            for dx in 0..CHUNK_SIZE {
                let x = (x0 + dx).min(self.width - 1);
                let y = (y0 + dy).min(self.height - 1);
                cells.extend_from_slice(&self.encode_cell_aplus(self.idx(x, y)));
            }
        }
        APlusChunkPayload {
            chunk_id,
            cx,
            cy,
            cells,
        }
    }

    pub fn mark_all_chunks_dirty(&mut self) {
        self.prev_display_hash.fill(u64::MAX);
    }

    fn compute_dirty_chunks(&mut self) -> Vec<(u16, u16)> {
        let chunks_x = self.width.div_ceil(CHUNK_SIZE);
        let chunks_y = self.height.div_ceil(CHUNK_SIZE);
        let mut dirty = Vec::new();
        for cy in 0..chunks_y {
            for cx in 0..chunks_x {
                let mut h: u64 = 0xcbf2_9ce4_8422_2325;
                let x0 = cx * CHUNK_SIZE;
                let y0 = cy * CHUNK_SIZE;
                for dy in 0..CHUNK_SIZE {
                    for dx in 0..CHUNK_SIZE {
                        let x = (x0 + dx).min(self.width - 1);
                        let y = (y0 + dy).min(self.height - 1);
                        let (d, hz) = self.encode_cell_display_and_hazard(self.idx(x, y));
                        let v = ((d[0] as u64) << 16) | ((d[1] as u64) << 8) | (hz as u64);
                        h ^= v;
                        h = h.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                let ci = cy * chunks_x + cx;
                if self.prev_display_hash[ci] != h {
                    self.prev_display_hash[ci] = h;
                    dirty.push((cx as u16, cy as u16));
                }
            }
        }
        dirty
    }

    pub fn state_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for s in 0..NUM_SPECIES {
            for &m in &self.mass[s] {
                h ^= m as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        for &e in &self.energy {
            h ^= e as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

// =========================================================================
// TRUE 3D VOLUMETRIC MULTI-STRATA ENGINE (`VolumeField`: W × H × L)
// =========================================================================

#[derive(Clone, Debug, Default)]
pub struct VolumeStepStats {
    pub tick: u64,
    pub proj_active_pct: f32,
    pub mass_error: i64,
    pub energy_error: i64,
    /// `(layer, cx, cy)` dirty volumetric chunks
    pub dirty_chunks: Vec<(u8, u16, u16)>,
    pub vertical_gravity_mass: i64,
    pub vertical_buoyancy_mass: i64,
}

/// Physically coupled 3D Volumetric Field (`W × H × depth`, default `256 × 256 × 4 = 262,144` voxels):
/// - Layer 0: Subterranean / Deep Riverbed (under-bridge waterways, sunken oil seams)
/// - Layer 1: Surface / Ground (marshes, lower banks, shallow streams, shrines, lower forest)
/// - Layer 2: Highland / Overpasses / Waterfalls (elevated basalt plateaus, alpine glacial lakes
///            that cascade down into L1/L0 via 3D gravity flux, walkable timber & stone Skybridges
///            directly over flowing river water, and upper pine cliffs)
/// - Layer 3: Upper Atmosphere / Plumes (buoyant rising 3D steam & wildfire smoke clouds + rain)
///
/// Maintains **exact integer 3D mass & energy conservation** (`mass_error == 0`, `energy_error == 0`)
/// across ALL vertical layers simultaneously via antisymmetric integer vertical fluxes `q_z`.
#[derive(Clone, Debug)]
pub struct VolumeField {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub layers: Vec<Field>,
    pub initial_total_mass: i64,
    pub initial_total_energy: i64,
    pub tick: u64,
}

impl VolumeField {
    /// Build **Shattered Reach 3D** (`W × H × depth`, `depth >= 4` recommended):
    /// true multi-level 3D world with under-bridge rivers, overhanging timber & stone skybridges,
    /// highland alpine glacial lakes feeding live 3D waterfalls, and an upper atmospheric plume layer!
    pub fn new_shattered_reach_3d(width: usize, height: usize, depth: usize) -> Self {
        let d = depth.max(1);
        let mut layers: Vec<Field> = (0..d).map(|_| Field::new_empty(width, height)).collect();
        let wf = width as f32;
        let hf = height as f32;

        // Layer 1 (or Layer 0 if depth == 1): Main Shattered Reach surface
        let surf_l = if d >= 2 { 1 } else { 0 };
        layers[surf_l] = Field::new_shattered_reach(width, height);

        // Layer 0 (when d >= 2): Deep Subterranean Riverbed + Under-Bridge Fjord Channel
        if d >= 2 {
            let l0 = &mut layers[0];
            for y in 0..height {
                for x in 0..width {
                    let idx = l0.idx(x, y);
                    let nx = x as f32 / wf;
                    let ny = y as f32 / hf;
                    let main_cx = 0.50 + 0.055 * (ny * std::f32::consts::TAU * 1.4).sin();
                    let main_hw = 0.072 + 0.022 * (ny * std::f32::consts::TAU * 2.1).cos();
                    let d_main = (nx - main_cx).abs();
                    let d_island = ((nx - 0.50).powi(2) + (ny - 0.50).powi(2)).sqrt();
                    if d_main < main_hw && d_island >= 0.058 {
                        l0.elevation[idx] = 1;
                        l0.mass[SP_WATER][idx] = 1150;
                        l0.flow_s[idx] = 140;
                        l0.set_cell_temp(idx, 28_400);
                    } else {
                        l0.elevation[idx] = 6;
                        l0.mass[SP_ROCK][idx] = 1200;
                        l0.set_cell_temp(idx, T_AMB);
                    }
                }
            }
        }

        // Layer 2 (when d >= 3): Highland Basalt Plateaus, Alpine Glacial Lakes (Waterfall Sources!),
        // True 3D Overhanging Skybridges & Stone Arches over the Fjord, and Highland Pine Ridges!
        if d >= 3 {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let nx = x as f32 / wf;
                    let ny = y as f32 / hf;
                    let main_cx = 0.50 + 0.055 * (ny * std::f32::consts::TAU * 1.4).sin();
                    let main_hw = 0.072 + 0.022 * (ny * std::f32::consts::TAU * 2.1).cos();
                    let d_main = (nx - main_cx).abs();

                    let h = ((x.wrapping_mul(73856093)) ^ (y.wrapping_mul(19349663)) ^ (x * y)) % 1000;

                    // 1. NW & NE Highland Basalt Plateaus perched above the fjord!
                    let nw_plateau = nx > 0.06 && nx < 0.36 && ny > 0.06 && ny < 0.33;
                    let ne_plateau = nx > 0.64 && nx < 0.94 && ny > 0.06 && ny < 0.33;
                    // Highland Alpine Lakes that spill over the cliff rim into live 3D waterfalls!
                    let nw_lake = ((nx - 0.25).powi(2) + (ny - 0.19).powi(2)).sqrt() < 0.075
                        || (nx >= 0.25 && nx < 0.37 && (ny - 0.22).abs() < 0.024);
                    let ne_lake = ((nx - 0.75).powi(2) + (ny - 0.19).powi(2)).sqrt() < 0.075
                        || (nx > 0.63 && nx <= 0.75 && (ny - 0.22).abs() < 0.024);

                    // 2. SE Obsidian High-Overlook Cliff with elevated Oil Basin (cascades fiery oil down!)
                    let se_overlook = ((nx - 0.76).powi(2) + (ny - 0.66).powi(2)).sqrt() < 0.105;
                    let se_oil_pool = ((nx - 0.76).powi(2) + (ny - 0.66).powi(2)).sqrt() < 0.045;

                    // 3. True 3D Overhanging Skybridges & Stone Arches directly OVER flowing L1/L0 river!
                    //    - North Stone Sky-Arch (y≈0.29): rock-bordered walkable stone overpass
                    //    - South Timber Skybridge (y≈0.67): destructible high wooden overpass (burns!)
                    let on_stone_arch = (ny - 0.29).abs() < 0.020 && nx > 0.34 && nx < 0.66;
                    let on_timber_skybridge = (ny - 0.67).abs() < 0.018 && nx > 0.35 && nx < 0.65;

                    let l2 = &mut layers[2];
                    l2.elevation[idx] = 20;

                    if nw_lake || ne_lake {
                        // Alpine water source spilling over cliff edge!
                        l2.mass[SP_WATER][idx] = 1050;
                        l2.flow_e[idx] = if nw_lake { 110 } else { -110 };
                        l2.set_cell_temp(idx, 28_200);
                    } else if on_stone_arch {
                        l2.mass[SP_SOIL][idx] = 950; // Walkable stone/flagstone overpass deck
                        l2.set_cell_temp(idx, T_AMB);
                    } else if on_timber_skybridge {
                        l2.mass[SP_WOOD][idx] = 520; // Walkable destructible timber Skybridge (<650 wall)
                        l2.set_cell_temp(idx, T_AMB);
                    } else if se_oil_pool {
                        l2.mass[SP_OIL][idx] = 760;
                        l2.mass[SP_SOIL][idx] = 400;
                        l2.set_cell_temp(idx, T_AMB);
                    } else if nw_plateau || ne_plateau || se_overlook {
                        l2.mass[SP_SOIL][idx] = 950;
                        if (nw_plateau || ne_plateau) && h < 190 && d_main > main_hw + 0.04 {
                            l2.mass[SP_WOOD][idx] = 900; // Highland snow-capped alpine pines
                        } else if h < 260 {
                            l2.mass[SP_PLANT][idx] = 380;
                        }
                        l2.set_cell_temp(idx, if ny < 0.35 { 27_400 } else { T_AMB });
                    }
                    // Otherwise Layer 2 is open air (all masses 0)!
                }
            }
        }

        // Layer 3 (when d >= 4): Upper Atmosphere / Plumes (starts as clean open air + light high mist)
        if d >= 4 {
            let l3 = &mut layers[3];
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    l3.elevation[idx] = 30;
                    // Seed light high-altitude glacial mist over the northern waterfalls
                    let nx = x as f32 / wf;
                    let ny = y as f32 / hf;
                    if ny > 0.16 && ny < 0.28 && ((nx - 0.37).abs() < 0.035 || (nx - 0.63).abs() < 0.035) {
                        l3.mass[SP_STEAM][idx] = 180;
                        l3.set_cell_temp(idx, 37_600);
                    }
                }
            }
        }

        let mut vf = Self {
            width,
            height,
            depth: d,
            layers,
            initial_total_mass: 0,
            initial_total_energy: 0,
            tick: 0,
        };
        vf.snapshot_conservation_baseline();
        vf
    }

    pub fn snapshot_conservation_baseline(&mut self) {
        for l in &mut self.layers {
            l.snapshot_conservation_baseline();
        }
        self.initial_total_mass = self.current_accounted_mass();
        self.initial_total_energy = self.current_accounted_energy();
    }

    pub fn current_accounted_mass(&self) -> i64 {
        self.layers.iter().map(|l| l.current_accounted_mass()).sum()
    }

    pub fn current_accounted_energy(&self) -> i64 {
        self.layers.iter().map(|l| l.current_accounted_energy()).sum()
    }

    /// Encode all `L` vertical layers concatenated into `[L, 18, H, W]` CHW f32 for batched GPU inference.
    pub fn encode_batched_input_tensor(&self) -> Vec<f32> {
        let layer_len = INPUT_CHANNELS * self.width * self.height;
        let mut out = Vec::with_capacity(self.depth * layer_len);
        for l in &self.layers {
            out.extend_from_slice(&l.encode_input_tensor());
        }
        out
    }

    pub fn refsim_raw_outputs_all(&self) -> Vec<RawStepOutput> {
        self.layers.iter().map(|l| l.refsim_raw_outputs()).collect()
    }

    /// Apply zero-violation weather across the 3D volume (and maintain Layer 2 highland waterfall springs).
    pub fn apply_weather_and_springs_3d(&mut self, weather_kind: u8, tick: u64) {
        let surf_l = if self.depth >= 2 { 1 } else { 0 };
        self.layers[surf_l].apply_weather_tick(weather_kind, tick);
        if self.depth >= 3 {
            // Replenish the NW & NE Layer 2 alpine lakes so the 3D waterfalls cascade continuously
            // (tracked explicitly in `faucet_mass` / `faucet_energy` -> zero conservation error!)
            let w = self.width;
            let h = self.height;
            let wf = w as f32;
            let hf = h as f32;
            let l2 = &mut self.layers[2];
            for y in (h / 8)..(h / 3) {
                let ny = y as f32 / hf;
                if (ny - 0.22).abs() > 0.022 {
                    continue;
                }
                let spring_hw = (w / 128).clamp(2, 6);
                for &nx_src in &[0.26f32, 0.74f32] {
                    let cx = (nx_src * wf) as usize;
                    for x in cx.saturating_sub(spring_hw)..=(cx + spring_hw).min(w - 3) {
                        let idx = y * w + x;
                        let cur = l2.mass[SP_WATER][idx];
                        if cur < 1000 && l2.mass[SP_ROCK][idx] < 500 {
                            let room = H_CAP.saturating_sub(l2.cell_volume(idx)) as u16;
                            let add = (1000 - cur).min(140).min(room);
                            if add > 0 {
                                l2.mass[SP_WATER][idx] += add;
                                let e_add = (add as i64) * SPECIES[SP_WATER].c * 28_200;
                                l2.energy[idx] += e_add;
                                l2.faucet_mass[SP_WATER] += add as i64;
                                l2.faucet_energy += e_add;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Exact integer 3D vertical exchange between adjacent strata (`z <-> z+1`):
    /// - **Gravity (`z+1 -> z`):** Liquid `water` & `oil` fall down through open air (3D Waterfalls & Oil Cascades!)
    /// - **Buoyancy (`z -> z+1`):** Hot `steam` & `smoke` rise upward into L2/L3 (3D Plume Columns!)
    /// - **Vertical Heat Conduction (`z <-> z+1`):** Rising wildfire heat ignites overhanging timber Skybridges!
    /// Guarantees **exact 0 mass error and 0 energy error** across all 3D layers.
    fn step_vertical_exchange_3d(&mut self) -> (i64, i64) {
        if self.depth < 2 {
            return (0, 0);
        }
        let len = self.width * self.height;
        let mut grav_moved = 0i64;
        let mut buoy_moved = 0i64;

        for z in 0..(self.depth - 1) {
            let (lo_slice, hi_slice) = self.layers.split_at_mut(z + 1);
            let lo = &mut lo_slice[z];
            let hi = &mut hi_slice[0];

            for idx in 0..len {
                let lo_blocked_solid =
                    lo.mass[SP_ROCK][idx] >= 500 || lo.mass[SP_SOIL][idx] >= 500 || lo.mass[SP_ICE][idx] >= 650;
                let hi_blocked_solid =
                    hi.mass[SP_ROCK][idx] >= 500 || hi.mass[SP_SOIL][idx] >= 500 || hi.mass[SP_ICE][idx] >= 650;

                // 1. GRAVITY: Water & Oil fall from `hi` (z+1) -> `lo` (z) when `lo` isn't solid rock/soil!
                //    (And `hi` isn't held up by a solid soil/rock plateau floor underneath!)
                let hi_has_floor = hi.mass[SP_SOIL][idx] >= 350 || hi.mass[SP_ROCK][idx] >= 350 || hi.mass[SP_ICE][idx] >= 450;
                if !lo_blocked_solid && !hi_has_floor {
                    for &sp in &[SP_WATER, SP_OIL] {
                        let avail = hi.mass[sp][idx];
                        if avail > 20 {
                            let room = H_CAP.saturating_sub(lo.cell_volume(idx)) as u16;
                            let q = ((avail / 3).max(1)).min(room);
                            if q > 0 {
                                let c_hi = hi.cell_heat_capacity(idx).max(1);
                                let t_hi = (hi.energy[idx] / c_hi).clamp(T_MIN, T_MAX);
                                let e_xfer = (q as i64) * SPECIES[sp].c * t_hi;
                                let e_clamped = e_xfer.min(hi.energy[idx].max(0));
                                hi.mass[sp][idx] -= q;
                                lo.mass[sp][idx] += q;
                                hi.energy[idx] -= e_clamped;
                                lo.energy[idx] += e_clamped;
                                grav_moved += q as i64;
                            }
                        }
                    }
                }

                // 2. BUOYANCY: Steam & Smoke rise from `lo` (z) -> `hi` (z+1) when `hi` is open air!
                if !hi_blocked_solid {
                    for &sp in &[SP_STEAM, SP_SMOKE] {
                        let avail = lo.mass[sp][idx];
                        if avail > 12 {
                            let room = H_CAP.saturating_sub(hi.cell_volume(idx)) as u16;
                            let q = ((avail / 3).max(1)).min(room);
                            if q > 0 {
                                let c_lo = lo.cell_heat_capacity(idx).max(1);
                                let t_lo = (lo.energy[idx] / c_lo).clamp(T_MIN, T_MAX);
                                let e_xfer = (q as i64) * SPECIES[sp].c * t_lo;
                                let e_clamped = e_xfer.min(lo.energy[idx].max(0));
                                lo.mass[sp][idx] -= q;
                                hi.mass[sp][idx] += q;
                                lo.energy[idx] -= e_clamped;
                                hi.energy[idx] += e_clamped;
                                buoy_moved += q as i64;
                            }
                        }
                    }
                }

                // 3. VERTICAL THERMAL CONDUCTION (hot fires beneath rise into skybridges/plumes!)
                let c_lo = lo.cell_heat_capacity(idx);
                let c_hi = hi.cell_heat_capacity(idx);
                if c_lo > 0 && c_hi > 0 {
                    let t_lo = lo.energy[idx] / c_lo;
                    let t_hi = hi.energy[idx] / c_hi;
                    let dt = t_lo - t_hi;
                    if dt.abs() > 1500 {
                        // Rising heat conducts faster upward than downward (convective bias)
                        let rate = if dt > 0 { 22 } else { 64 };
                        let g_raw = (dt * c_lo.min(c_hi)) / rate;
                        let g = if g_raw > 0 {
                            let avail = (lo.energy[idx] - c_lo * T_MIN).max(0) / 4;
                            let room = (c_hi * T_MAX - hi.energy[idx]).max(0) / 4;
                            g_raw.min(avail).min(room)
                        } else {
                            let avail = (hi.energy[idx] - c_hi * T_MIN).max(0) / 4;
                            let room = (c_lo * T_MAX - lo.energy[idx]).max(0) / 4;
                            g_raw.max(-avail).max(-room)
                        };
                        if g != 0 {
                            lo.energy[idx] -= g;
                            hi.energy[idx] += g;
                        }
                    }
                }
            }
        }
        (grav_moved, buoy_moved)
    }

    /// Apply all `L` horizontal neural/refsim layer steps + exact 3D vertical exchange + dirty chunk diffing.
    pub fn step_3d(&mut self, raw_layers: &[RawStepOutput]) -> VolumeStepStats {
        let mut proj_sum = 0.0f32;
        for (z, l) in self.layers.iter_mut().enumerate() {
            if let Some(raw) = raw_layers.get(z) {
                let st = l.project_and_apply(raw);
                proj_sum += st.proj_active_pct;
            }
        }
        let (grav, buoy) = self.step_vertical_exchange_3d();
        self.tick += 1;

        let mut dirty_chunks = Vec::new();
        for (z, l) in self.layers.iter_mut().enumerate() {
            // Re-hash after vertical exchange so waterfall/plume changes mark chunks dirty!
            let dc = l.compute_dirty_chunks();
            for (cx, cy) in dc {
                dirty_chunks.push((z as u8, cx, cy));
            }
        }

        VolumeStepStats {
            tick: self.tick,
            proj_active_pct: proj_sum / self.depth.max(1) as f32,
            mass_error: self.current_accounted_mass() - self.initial_total_mass,
            energy_error: self.current_accounted_energy() - self.initial_total_energy,
            dirty_chunks,
            vertical_gravity_mass: grav,
            vertical_buoyancy_mass: buoy,
        }
    }

    /// Check whether a 3D cell `(x, y, layer)` has non-empty matter (used to skip empty air chunks on L2/L3).
    pub fn is_chunk_non_empty(&self, layer: usize, cx: u16, cy: u16) -> bool {
        let Some(f) = self.layers.get(layer) else {
            return false;
        };
        let x0 = (cx as usize) * CHUNK_SIZE;
        let y0 = (cy as usize) * CHUNK_SIZE;
        for dy in 0..CHUNK_SIZE {
            let y = (y0 + dy).min(f.height - 1);
            for dx in 0..CHUNK_SIZE {
                let x = (x0 + dx).min(f.width - 1);
                if f.cell_volume(f.idx(x, y)) > 0 {
                    return true;
                }
            }
        }
        false
    }

    /// Helper: apply a spell burst at `(cx, cy, layer)` with vertical splash onto adjacent layer.
    pub fn apply_fire_3d(&mut self, cx: i32, cy: i32, layer: usize, radius: i32, strength: f32) {
        let l = layer.min(self.depth - 1);
        self.layers[l].apply_fire_bolt(cx, cy, radius, strength);
    }

    pub fn apply_frost_3d(&mut self, cx: i32, cy: i32, layer: usize, radius: i32, strength: f32) {
        let l = layer.min(self.depth - 1);
        self.layers[l].apply_frost_bolt(cx, cy, radius, strength);
    }

    pub fn apply_oil_3d(&mut self, cx: i32, cy: i32, layer: usize, radius: i32, strength: f32) {
        let l = layer.min(self.depth - 1);
        self.layers[l].apply_oil_splash(cx, cy, radius, strength);
    }

    pub fn apply_steam_burst_3d(&mut self, cx: i32, cy: i32, layer: usize, radius: i32) -> u32 {
        let l = layer.min(self.depth - 1);
        self.layers[l].apply_steam_burst(cx, cy, radius)
    }
}

#[inline]
fn sanitize_f32(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

#[inline]
fn round_ties_even_i32(v: f32) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    v.round_ties_even().clamp(-100_000.0, 100_000.0) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reaction_mass_conservation_by_construction() {
        for r in &REACTIONS {
            let sum: i32 = r.nu.iter().map(|&v| v as i32).sum();
            assert_eq!(sum, 0, "Reaction {} violates mass conservation", r.name);
        }
    }

    #[test]
    fn test_g0_bit_exact_replay_and_g1_refsim_conservation() {
        let mut f1 = Field::new_frostmarsh(64, 64);
        let mut f2 = Field::new_frostmarsh(64, 64);

        for tick in 0..300 {
            if tick == 20 {
                f1.apply_frost_bolt(32, 32, 5, 1.5);
                f2.apply_frost_bolt(32, 32, 5, 1.5);
            }
            if tick == 60 {
                f1.apply_fire_bolt(24, 32, 4, 1.5);
                f2.apply_fire_bolt(24, 32, 4, 1.5);
            }
            let r1 = f1.refsim_raw_outputs();
            let s1 = f1.project_and_apply(&r1);
            let r2 = f2.refsim_raw_outputs();
            let _s2 = f2.project_and_apply(&r2);

            assert_eq!(s1.mass_error, 0, "mass conservation error at tick {}", tick);
            assert_eq!(s1.energy_error, 0, "energy conservation error at tick {}", tick);
            for idx in 0..(64 * 64) {
                assert!(f1.cell_volume(idx) <= H_CAP);
                let t = f1.cell_temp_ck(idx);
                assert!((T_MIN..=T_MAX).contains(&t));
            }
        }
        assert_eq!(f1.state_hash(), f2.state_hash(), "replay hash mismatch");
    }

    #[test]
    fn test_g1_projection_firewall_under_garbage_nan_inf_adversarial_inputs() {
        let mut f = Field::new_frostmarsh(48, 48);
        let mut raw = RawStepOutput::zeros(48, 48);

        for step in 0..50 {
            for (i, v) in raw.data.iter_mut().enumerate() {
                *v = match (i + step) % 7 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => 1e9,
                    4 => -1e9,
                    5 => 9999.5,
                    _ => -9999.5,
                };
            }
            let stats = f.project_and_apply(&raw);
            assert_eq!(stats.mass_error, 0, "garbage mass violation at step {}", step);
            assert_eq!(stats.energy_error, 0, "garbage energy violation at step {}", step);
            for idx in 0..(48 * 48) {
                assert!(f.cell_volume(idx) <= H_CAP, "volume cap violated");
            }
        }
    }

    #[test]
    fn test_frost_bolt_freezes_river_and_fire_bolt_ignites_wood() {
        let mut f = Field::new_frostmarsh(64, 64);
        let river_idx = f.idx(32, 32);
        assert!(f.mass[SP_WATER][river_idx] > 500);
        let ice_before = f.mass[SP_ICE][river_idx];

        f.apply_frost_bolt(32, 32, 4, 2.0);
        for _ in 0..15 {
            let raw = f.refsim_raw_outputs();
            let stats = f.project_and_apply(&raw);
            assert_eq!(stats.mass_error, 0);
            assert_eq!(stats.energy_error, 0);
        }
        assert!(
            f.mass[SP_ICE][river_idx] > ice_before + 100,
            "expected frost bolt to freeze river water into ice"
        );

        // Find a wood cell and ignite it
        let wood_cell = (0..(64 * 64))
            .find(|&i| f.mass[SP_WOOD][i] >= 500)
            .expect("frostmarsh should contain wood");
        let wx = (wood_cell % 64) as i32;
        let wy = (wood_cell / 64) as i32;
        let wood_before = f.mass[SP_WOOD][wood_cell];

        f.apply_fire_bolt(wx, wy, 3, 2.0);
        for _ in 0..15 {
            let raw = f.refsim_raw_outputs();
            let stats = f.project_and_apply(&raw);
            assert_eq!(stats.mass_error, 0);
            assert_eq!(stats.energy_error, 0);
        }
        assert!(
            f.mass[SP_WOOD][wood_cell] < wood_before,
            "expected fire bolt to burn wood into ash+smoke"
        );
    }

    #[test]
    fn test_shattered_reach_256_conservation_and_weather() {
        let mut f = Field::new_shattered_reach(128, 128);
        f.apply_glacial_nova(64, 48, 6);
        f.apply_meteor_impact(92, 82, 5);
        let boiled = f.apply_steam_burst(64, 34, 4);
        assert!(boiled > 0, "steam burst should boil fjord water");
        for tick in 0..20 {
            f.apply_weather_tick((tick % 4) as u8, tick);
            let raw = f.refsim_raw_outputs();
            let stats = f.project_and_apply(&raw);
            assert_eq!(stats.mass_error, 0, "mass err at tick {}", tick);
            assert_eq!(stats.energy_error, 0, "energy err at tick {}", tick);
        }
    }

    #[test]
    fn test_volume_3d_conservation_waterfalls_buoyancy_skybridge() {
        let mut vf = VolumeField::new_shattered_reach_3d(64, 64, 4);
        // Trigger a steam burst & wildfire on L1 (ground) to verify buoyant plumes rise L1->L2->L3
        vf.apply_steam_burst_3d(32, 32, 1, 4);
        vf.apply_fire_3d(24, 42, 1, 4, 1.8);
        let mut total_grav = 0i64;
        let mut total_buoy = 0i64;
        for tick in 0..16 {
            vf.apply_weather_and_springs_3d((tick % 4) as u8, tick);
            let raws = vf.refsim_raw_outputs_all();
            let stats = vf.step_3d(&raws);
            assert_eq!(stats.mass_error, 0, "3D mass error at tick {}", tick);
            assert_eq!(stats.energy_error, 0, "3D energy error at tick {}", tick);
            total_grav += stats.vertical_gravity_mass;
            total_buoy += stats.vertical_buoyancy_mass;
        }
        assert!(total_grav > 0, "expected 3D waterfall gravity transport L2->L1->L0");
        assert!(total_buoy > 0, "expected 3D buoyant steam/smoke rise L1->L2->L3");
    }

    #[test]
    fn test_volume_512_init_and_step_conservation() {
        let mut vf = VolumeField::new_shattered_reach_3d(512, 512, 4);
        assert_eq!(vf.width, 512);
        assert_eq!(vf.height, 512);
        assert_eq!(vf.depth, 4);
        assert_eq!(vf.layers.len(), 4);
        assert_eq!(vf.layers[1].mass[SP_WATER].len(), 512 * 512);

        // Verify proportional geometry at 512x512x4:
        // L2 stone sky-arch around y=0.29, x=0.50; timber skybridge around y=0.67, x=0.50
        let arch_idx = vf.layers[2].idx(256, (512.0 * 0.29) as usize);
        let bridge_idx = vf.layers[2].idx(256, (512.0 * 0.67) as usize);
        assert!(vf.layers[2].mass[SP_SOIL][arch_idx] > 0, "L2 stone sky-arch missing at 512x512");
        assert!(vf.layers[2].mass[SP_WOOD][bridge_idx] > 0, "L2 timber skybridge missing at 512x512");
        // L1 fjord water directly beneath L2 skybridge
        assert!(vf.layers[1].mass[SP_WATER][bridge_idx] > 0, "L1 river under skybridge missing at 512x512");

        // Trigger 3D spell + 2 full conservation steps on 1,048,576 voxels
        vf.apply_steam_burst_3d(256, 256, 1, 4);
        let mut total_grav = 0i64;
        for tick in 0..2 {
            vf.apply_weather_and_springs_3d((tick % 4) as u8, tick);
            let raws = vf.refsim_raw_outputs_all();
            let stats = vf.step_3d(&raws);
            assert_eq!(stats.mass_error, 0, "512x512x4 mass error at tick {}", tick);
            assert_eq!(stats.energy_error, 0, "512x512x4 energy error at tick {}", tick);
            total_grav += stats.vertical_gravity_mass;
        }
        assert!(total_grav > 0, "expected 3D waterfall gravity transport at 512x512x4");
    }
}
