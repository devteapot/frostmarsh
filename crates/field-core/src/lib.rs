//! `field-core`: Authoritative multi-species conserved physical world field,
//! 10-step integer Projection Firewall, exact integer RefSim, and Frostmarsh scenes.
//!
//! Matches `FIELD_CONTRACT.md` bit-for-bit in species, reactions, and ONNX tensor layout.

use serde::{Deserialize, Serialize};

pub mod sharding;
pub mod structural;

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

pub const BIOME_NORDIC_VIADUCT: u8 = 0;
pub const BIOME_VOLCANIC_CALDERA: u8 = 1;
pub const BIOME_SINKHOLE_CAVERNS: u8 = 2;
pub const BIOME_GLACIER_GORGE: u8 = 3;
pub const BIOME_GEOTHERMAL_BASIN: u8 = 4;
pub const BIOME_SKY_ABYSS: u8 = 5;
pub const BIOME_COUNT: u8 = 6;
pub const NUM_BIOMES: u8 = BIOME_COUNT;

pub const fn biome_name(id: u8) -> &'static str {
    match id % NUM_BIOMES {
        BIOME_NORDIC_VIADUCT => "Nordic Viaduct",
        BIOME_VOLCANIC_CALDERA => "Volcanic Caldera",
        BIOME_SINKHOLE_CAVERNS => "Sinkhole Caverns",
        BIOME_GLACIER_GORGE => "Glacier Gorge",
        BIOME_GEOTHERMAL_BASIN => "Geothermal Mist Basin",
        BIOME_SKY_ABYSS => "Sky-Isles & Abyss",
        _ => "Nordic Viaduct",
    }
}

pub const WORLD_LAW_STANDARD: u8 = 0;
pub const WORLD_LAW_FREEZE_OUT: u8 = 1;
pub const WORLD_LAW_RISING_FLOOD: u8 = 2;
pub const WORLD_LAW_LOW_G_MAGMA: u8 = 3;
pub const NUM_WORLD_LAW_PRESETS: u8 = 4;

/// Live 6-float world-law parameter vector governing gravity, ambient temperature shift,
/// combustion volatility, fluid viscosity, hydrologic precipitation, and thermal conductivity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldLaws {
    pub gravity_g: f32,
    pub ambient_temp_shift: f32,
    pub volatility: f32,
    pub viscosity: f32,
    pub precipitation: f32,
    pub thermal_cond: f32,
}

impl Default for WorldLaws {
    fn default() -> Self {
        Self::from_preset(WORLD_LAW_STANDARD)
    }
}

impl WorldLaws {
    pub const fn from_preset(preset_id: u8) -> Self {
        match preset_id % NUM_WORLD_LAW_PRESETS {
            WORLD_LAW_FREEZE_OUT => Self {
                gravity_g: 1.0,
                ambient_temp_shift: -5500.0,
                volatility: 1.2,
                viscosity: 1.1,
                precipitation: 1.4,
                thermal_cond: 1.3,
            },
            WORLD_LAW_RISING_FLOOD => Self {
                gravity_g: 1.15,
                ambient_temp_shift: -600.0,
                volatility: 0.6,
                viscosity: 0.55,
                precipitation: 2.6,
                thermal_cond: 1.0,
            },
            WORLD_LAW_LOW_G_MAGMA => Self {
                gravity_g: 0.35,
                ambient_temp_shift: 5500.0,
                volatility: 2.2,
                viscosity: 1.6,
                precipitation: 0.25,
                thermal_cond: 1.5,
            },
            _ => Self {
                gravity_g: 1.0,
                ambient_temp_shift: 0.0,
                volatility: 1.0,
                viscosity: 1.0,
                precipitation: 1.0,
                thermal_cond: 1.0,
            },
        }
    }

    pub const fn preset_name(preset_id: u8) -> &'static str {
        match preset_id % NUM_WORLD_LAW_PRESETS {
            WORLD_LAW_FREEZE_OUT => "FREEZE-OUT",
            WORLD_LAW_RISING_FLOOD => "RISING FLOOD",
            WORLD_LAW_LOW_G_MAGMA => "LOW-G MAGMA",
            _ => "STANDARD",
        }
    }

    pub fn is_standard(&self) -> bool {
        (self.gravity_g - 1.0).abs() <= 1e-4
            && self.ambient_temp_shift.abs() <= 1e-2
            && (self.volatility - 1.0).abs() <= 1e-4
            && (self.viscosity - 1.0).abs() <= 1e-4
            && (self.precipitation - 1.0).abs() <= 1e-4
            && (self.thermal_cond - 1.0).abs() <= 1e-4
    }
}

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

    pub fn mass_error(&self) -> i64 {
        self.current_accounted_mass() - self.initial_total_mass
    }

    pub fn energy_error(&self) -> i64 {
        self.current_accounted_energy() - self.initial_total_energy
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

#[derive(Clone, Copy, Debug, Default)]
pub struct EntityVoxelPhysics {
    pub temp_c: f32,            // local voxel temperature
    pub delta_temp_spike: f32,  // positive sharp thermal rise since last tick (combustion/ignition!)
    pub delta_steam_burst: f32, // sharp steam surge since last tick (flash-boil!)
    pub ice_frac: f32,          // low friction / glide
    pub oil_frac: f32,          // slip / flammable substrate
    pub water_frac: f32,        // drag / buoyancy
    pub smoke_opacity: f32,     // optical smoke+steam concealment [0..1]
    pub support_solid: bool,    // whether solid ground/bridge exists at or directly below entity
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldEvolutionStats {
    pub basalt_forged: u64,
    pub saplings_grown: u64,
    pub glaciers_formed: u64,
    pub erosion_events: u64,
    pub world_epoch: u64,
}

/// Lossless serializable snapshot of a single 2D stratum `Field`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldSnapshot {
    pub width: usize,
    pub height: usize,
    pub mass: [Vec<u16>; NUM_SPECIES],
    pub energy: Vec<i64>,
    pub flow_e: Vec<i16>,
    pub flow_s: Vec<i16>,
    pub elevation: Vec<u8>,
    pub forcing_energy: Vec<i64>,
    pub forcing_mass: Vec<i32>,
    pub critter_density: Vec<u8>,
    pub atmosphere_mass: [i64; NUM_SPECIES],
    pub atmosphere_energy: i64,
    pub faucet_mass: [i64; NUM_SPECIES],
    pub faucet_energy: i64,
    pub initial_total_mass: i64,
    pub initial_total_energy: i64,
    pub tick: u64,
}

/// Lossless serializable snapshot of an entire 3D `VolumeField` preserving exact integer
/// conservation (`mass_error() == 0 && energy_error() == 0`) immediately upon reload.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VolumeFieldSnapshot {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub biome_id: u8,
    pub seed: u64,
    pub layers: Vec<FieldSnapshot>,
    pub prev_temp: Vec<i32>,
    pub prev_steam: Vec<i32>,
    pub initial_total_mass: i64,
    pub initial_total_energy: i64,
    pub tick: u64,
    pub world_time_secs: f64,
    pub evolution_stats: WorldEvolutionStats,
    pub world_laws: WorldLaws,
    pub world_law_preset: u8,
}

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
    pub biome_id: u8,
    pub seed: u64,
    pub layers: Vec<Field>,
    pub prev_temp: Vec<i32>,
    pub prev_steam: Vec<i32>,
    pub initial_total_mass: i64,
    pub initial_total_energy: i64,
    pub tick: u64,
    pub world_time_secs: f64,
    pub evolution_stats: WorldEvolutionStats,
    pub world_laws: WorldLaws,
    pub world_law_preset: u8,
    pub structural: structural::StructuralState,
}

impl VolumeField {
    /// Set the active live `WorldLaws` preset (`0=STANDARD, 1=FREEZE-OUT, 2=RISING FLOOD, 3=LOW-G MAGMA`).
    pub fn set_world_law_preset(&mut self, preset_id: u8) {
        let p = preset_id % NUM_WORLD_LAW_PRESETS;
        self.world_law_preset = p;
        self.world_laws = WorldLaws::from_preset(p);
    }

    /// Build **Shattered Reach 3D** (`W × H × depth`, `depth >= 4` recommended):
    /// true multi-level 3D world with under-bridge rivers, overhanging timber & stone skybridges,
    /// highland alpine glacial lakes feeding live 3D waterfalls, and an upper atmospheric plume layer!
    pub fn new_shattered_reach_3d(width: usize, height: usize, depth: usize) -> Self {
        Self::new_biome_3d(width, height, depth, BIOME_NORDIC_VIADUCT, 1)
    }

    /// Build a 3D volumetric world for any of the 5 true-3D biomes (`0..=4`) and deterministic `seed`.
    /// When `depth >= 16`, delegates directly to `init_biome_3d(biome_id, seed)` (supporting `Z = 16..=32`).
    pub fn new_biome_3d(
        width: usize,
        height: usize,
        depth: usize,
        biome_id: u8,
        seed: u64,
    ) -> Self {
        let d = depth.max(1);
        let b_id = biome_id % NUM_BIOMES;
        let mut layers: Vec<Field> = (0..d).map(|_| Field::new_empty(width, height)).collect();
        if d >= 16 {
            let vlen = d * width * height;
            let mut vf = Self {
                width,
                height,
                depth: d,
                biome_id: b_id,
                seed,
                layers,
                prev_temp: vec![T_AMB as i32; vlen],
                prev_steam: vec![0i32; vlen],
                initial_total_mass: 0,
                initial_total_energy: 0,
                tick: 0,
                world_time_secs: 0.0,
                evolution_stats: WorldEvolutionStats::default(),
                world_laws: WorldLaws::default(),
                world_law_preset: WORLD_LAW_STANDARD,
                structural: structural::StructuralState::default(),
            };
            vf.init_biome_3d(b_id, seed);
            return vf;
        }
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

        let vlen = d * width * height;
        let mut vf = Self {
            width,
            height,
            depth: d,
            biome_id: b_id,
            seed,
            layers,
            prev_temp: vec![T_AMB as i32; vlen],
            prev_steam: vec![0i32; vlen],
            initial_total_mass: 0,
            initial_total_energy: 0,
            tick: 0,
            world_time_secs: 0.0,
            evolution_stats: WorldEvolutionStats::default(),
            world_laws: WorldLaws::default(),
            world_law_preset: WORLD_LAW_STANDARD,
            structural: structural::StructuralState::default(),
        };
        vf.snapshot_conservation_baseline();
        vf.snapshot_prev_sensors();
        vf
    }

    pub fn snapshot_prev_sensors(&mut self) {
        let len = self.width * self.height;
        let total = self.depth * len;
        if self.prev_temp.len() != total {
            self.prev_temp.resize(total, T_AMB as i32);
        }
        if self.prev_steam.len() != total {
            self.prev_steam.resize(total, 0);
        }
        for (z, f) in self.layers.iter().enumerate() {
            let base = z * len;
            for idx in 0..len {
                self.prev_temp[base + idx] = f.cell_temp_ck(idx) as i32;
                self.prev_steam[base + idx] = f.mass[SP_STEAM][idx] as i32;
            }
        }
    }

    pub fn snapshot_layer_prev_sensors(&mut self, layer: usize) {
        if layer >= self.depth {
            return;
        }
        let len = self.width * self.height;
        let total = self.depth * len;
        if self.prev_temp.len() != total {
            self.prev_temp.resize(total, T_AMB as i32);
        }
        if self.prev_steam.len() != total {
            self.prev_steam.resize(total, 0);
        }
        let f = &self.layers[layer];
        let base = layer * len;
        for idx in 0..len {
            self.prev_temp[base + idx] = f.cell_temp_ck(idx) as i32;
            self.prev_steam[base + idx] = f.mass[SP_STEAM][idx] as i32;
        }
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

    pub fn mass_error(&self) -> i64 {
        self.current_accounted_mass() - self.initial_total_mass
    }

    pub fn energy_error(&self) -> i64 {
        self.current_accounted_energy() - self.initial_total_energy
    }

    /// Capture an exact lossless snapshot of the entire 3D `VolumeField`.
    pub fn to_snapshot(&self) -> VolumeFieldSnapshot {
        let layers = self
            .layers
            .iter()
            .map(|l| FieldSnapshot {
                width: l.width,
                height: l.height,
                mass: l.mass.clone(),
                energy: l.energy.clone(),
                flow_e: l.flow_e.clone(),
                flow_s: l.flow_s.clone(),
                elevation: l.elevation.clone(),
                forcing_energy: l.forcing_energy.clone(),
                forcing_mass: l.forcing_mass.clone(),
                critter_density: l.critter_density.clone(),
                atmosphere_mass: l.atmosphere_mass,
                atmosphere_energy: l.atmosphere_energy,
                faucet_mass: l.faucet_mass,
                faucet_energy: l.faucet_energy,
                initial_total_mass: l.initial_total_mass,
                initial_total_energy: l.initial_total_energy,
                tick: l.tick,
            })
            .collect();
        VolumeFieldSnapshot {
            width: self.width,
            height: self.height,
            depth: self.depth,
            biome_id: self.biome_id,
            seed: self.seed,
            layers,
            prev_temp: self.prev_temp.clone(),
            prev_steam: self.prev_steam.clone(),
            initial_total_mass: self.initial_total_mass,
            initial_total_energy: self.initial_total_energy,
            tick: self.tick,
            world_time_secs: self.world_time_secs,
            evolution_stats: self.evolution_stats,
            world_laws: self.world_laws,
            world_law_preset: self.world_law_preset,
        }
    }

    /// Restore a `VolumeField` losslessly from a `VolumeFieldSnapshot`, validating dimensions
    /// and guaranteeing exact conservation (`mass_error() == 0 && energy_error() == 0`).
    pub fn from_snapshot(snap: VolumeFieldSnapshot) -> Result<Self, String> {
        if snap.width == 0 || snap.height == 0 || snap.depth == 0 {
            return Err("VolumeFieldSnapshot has zero dimension".to_string());
        }
        if snap.layers.len() != snap.depth {
            return Err(format!(
                "VolumeFieldSnapshot layer count {} != depth {}",
                snap.layers.len(),
                snap.depth
            ));
        }
        let area = snap.width * snap.height;
        let chunks_x = snap.width.div_ceil(CHUNK_SIZE);
        let chunks_y = snap.height.div_ceil(CHUNK_SIZE);
        let mut layers = Vec::with_capacity(snap.depth);
        for (z, ls) in snap.layers.into_iter().enumerate() {
            if ls.width != snap.width || ls.height != snap.height {
                return Err(format!("layer {} dimensions mismatch", z));
            }
            for s in 0..NUM_SPECIES {
                if ls.mass[s].len() != area {
                    return Err(format!("layer {} species {} mass len mismatch", z, s));
                }
            }
            if ls.energy.len() != area
                || ls.flow_e.len() != area
                || ls.flow_s.len() != area
                || ls.elevation.len() != area
                || ls.forcing_energy.len() != area
                || ls.forcing_mass.len() != area
                || ls.critter_density.len() != area
            {
                return Err(format!("layer {} field vector length mismatch", z));
            }
            layers.push(Field {
                width: ls.width,
                height: ls.height,
                mass: ls.mass,
                energy: ls.energy,
                flow_e: ls.flow_e,
                flow_s: ls.flow_s,
                elevation: ls.elevation,
                forcing_energy: ls.forcing_energy,
                forcing_mass: ls.forcing_mass,
                critter_density: ls.critter_density,
                atmosphere_mass: ls.atmosphere_mass,
                atmosphere_energy: ls.atmosphere_energy,
                faucet_mass: ls.faucet_mass,
                faucet_energy: ls.faucet_energy,
                initial_total_mass: ls.initial_total_mass,
                initial_total_energy: ls.initial_total_energy,
                tick: ls.tick,
                prev_display_hash: vec![u64::MAX; chunks_x * chunks_y],
            });
        }
        let vlen = snap.depth * area;
        let mut vf = Self {
            width: snap.width,
            height: snap.height,
            depth: snap.depth,
            biome_id: snap.biome_id % NUM_BIOMES,
            seed: snap.seed,
            layers,
            prev_temp: if snap.prev_temp.len() == vlen {
                snap.prev_temp
            } else {
                vec![T_AMB as i32; vlen]
            },
            prev_steam: if snap.prev_steam.len() == vlen {
                snap.prev_steam
            } else {
                vec![0i32; vlen]
            },
            initial_total_mass: snap.initial_total_mass,
            initial_total_energy: snap.initial_total_energy,
            tick: snap.tick,
            world_time_secs: if snap.world_time_secs.is_finite() && snap.world_time_secs >= 0.0 {
                snap.world_time_secs
            } else {
                snap.tick as f64 * 0.1
            },
            evolution_stats: snap.evolution_stats,
            world_laws: snap.world_laws,
            world_law_preset: snap.world_law_preset % NUM_WORLD_LAW_PRESETS,
            structural: structural::StructuralState::default(),
        };
        if vf.biome_id == BIOME_SKY_ABYSS && vf.depth >= 16 {
            structural::init_sky_abyss_buoyant_cores(&mut vf);
        }
        if vf.prev_temp.len() != vlen || vf.prev_steam.len() != vlen {
            vf.snapshot_prev_sensors();
        }
        Ok(vf)
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

    /// Apply zero-violation weather across the 3D volume (and maintain highland waterfall/geothermal springs).
    pub fn apply_weather_and_springs_3d(&mut self, weather_kind: u8, tick: u64) {
        if self.depth >= 16 {
            let z_scale = (self.depth as f32 / 16.0).clamp(1.0, 2.0);
            let surf_l = ((4.0 * z_scale).round() as usize).min(self.depth - 1);
            self.layers[surf_l].apply_weather_tick(weather_kind, tick);
            let w = self.width;
            let h = self.height;
            let wf = w as f32;
            let hf = h as f32;
            match self.biome_id % NUM_BIOMES {
                BIOME_NORDIC_VIADUCT => {
                    // Replenish the upper highland glacial basin so the plunging 3D waterfall cascades continuously
                    // (tracked explicitly in `faucet_mass` / `faucet_energy` -> zero conservation error!)
                    let z_basin = ((10.0 * z_scale).round() as usize).min(self.depth - 2);
                    let lb = &mut self.layers[z_basin];
                    let cy = (0.17 * hf) as usize;
                    let cx = (0.50 * wf) as usize;
                    let rx = (w / 48).clamp(2, 6);
                    let ry = (h / 64).clamp(1, 4);
                    for y in cy.saturating_sub(ry)..=(cy + ry).min(h - 3) {
                        for x in cx.saturating_sub(rx)..=(cx + rx).min(w - 3) {
                            let idx = y * w + x;
                            let cur = lb.mass[SP_WATER][idx];
                            if cur < 1050 && lb.mass[SP_ROCK][idx] < 500 && lb.mass[SP_ICE][idx] < 500 {
                                let room = H_CAP.saturating_sub(lb.cell_volume(idx)) as u16;
                                let add = (1050 - cur).min(140).min(room);
                                if add > 0 {
                                    lb.mass[SP_WATER][idx] += add;
                                    let e_add = (add as i64) * SPECIES[SP_WATER].c * 28_000;
                                    lb.energy[idx] += e_add;
                                    lb.faucet_mass[SP_WATER] += add as i64;
                                    lb.faucet_energy += e_add;
                                }
                            }
                        }
                    }
                }
                BIOME_VOLCANIC_CALDERA => {
                    // Volcanic fumaroles continuously emit warm buoyant steam + smoke into the caldera sky
                    let z_vent = ((3.0 * z_scale).round() as usize).clamp(2, self.depth - 3);
                    let lv = &mut self.layers[z_vent];
                    for &(vx_n, vy_n, sp, target_t) in &[
                        (0.38f32, 0.36f32, SP_STEAM, 44_000i64),
                        (0.62f32, 0.64f32, SP_SMOKE, 58_000i64),
                    ] {
                        let cx = (vx_n * wf) as usize;
                        let cy = (vy_n * hf) as usize;
                        let r = (w / 48).clamp(1, 3);
                        for y in cy.saturating_sub(r)..=(cy + r).min(h - 3) {
                            for x in cx.saturating_sub(r)..=(cx + r).min(w - 3) {
                                let idx = y * w + x;
                                if lv.mass[SP_ROCK][idx] < 400 && lv.mass[sp][idx] < 360 {
                                    let room = H_CAP.saturating_sub(lv.cell_volume(idx)) as u16;
                                    let add = (360 - lv.mass[sp][idx]).min(75).min(room);
                                    if add > 0 {
                                        lv.mass[sp][idx] += add;
                                        let e_th = (add as i64) * SPECIES[sp].c * target_t;
                                        let e_chem = (add as i64) * SPECIES[sp].u0;
                                        lv.energy[idx] += e_th;
                                        lv.faucet_mass[sp] += add as i64;
                                        lv.faucet_energy += e_th + e_chem;
                                    }
                                }
                            }
                        }
                    }
                }
                BIOME_SINKHOLE_CAVERNS => {
                    // Replenish a karst waterfall pouring over the northern sinkhole lip into the subterranean lake
                    let z_rim = ((11.0 * z_scale).round() as usize).clamp(4, self.depth - 2);
                    let lr = &mut self.layers[z_rim];
                    let cx = (0.50 * wf) as usize;
                    let cy = (0.34 * hf) as usize;
                    let r = (w / 48).clamp(1, 3);
                    for y in cy.saturating_sub(r)..=(cy + r).min(h - 3) {
                        for x in cx.saturating_sub(r)..=(cx + r).min(w - 3) {
                            let idx = y * w + x;
                            let cur = lr.mass[SP_WATER][idx];
                            if cur < 920 && lr.mass[SP_ROCK][idx] < 500 {
                                let room = H_CAP.saturating_sub(lr.cell_volume(idx)) as u16;
                                let add = (920 - cur).min(110).min(room);
                                if add > 0 {
                                    lr.mass[SP_WATER][idx] += add;
                                    let e_add = (add as i64) * SPECIES[SP_WATER].c * 28_500;
                                    lr.energy[idx] += e_add;
                                    lr.faucet_mass[SP_WATER] += add as i64;
                                    lr.faucet_energy += e_add;
                                }
                            }
                        }
                    }
                }
                BIOME_GLACIER_GORGE => {
                    // Replenish the cold glacial cataract cascading down the canyon wall
                    let z_src = ((9.0 * z_scale).round() as usize).clamp(4, self.depth - 2);
                    let lg = &mut self.layers[z_src];
                    let cx = (0.37 * wf) as usize;
                    let cy = (0.24 * hf) as usize;
                    let r = (w / 48).clamp(1, 3);
                    for y in cy.saturating_sub(r)..=(cy + r).min(h - 3) {
                        for x in cx.saturating_sub(r)..=(cx + r).min(w - 3) {
                            let idx = y * w + x;
                            let cur = lg.mass[SP_WATER][idx];
                            if cur < 900 && lg.mass[SP_ROCK][idx] < 500 && lg.mass[SP_ICE][idx] < 500 {
                                let room = H_CAP.saturating_sub(lg.cell_volume(idx)) as u16;
                                let add = (900 - cur).min(100).min(room);
                                if add > 0 {
                                    lg.mass[SP_WATER][idx] += add;
                                    let e_add = (add as i64) * SPECIES[SP_WATER].c * 27_600;
                                    lg.energy[idx] += e_add;
                                    lg.faucet_mass[SP_WATER] += add as i64;
                                    lg.faucet_energy += e_add;
                                }
                            }
                        }
                    }
                }
                BIOME_SKY_ABYSS => {
                    // Replenish the Floating Sky-Island plunging waterfall springs (Stratum 3)
                    // + Abyssal geothermal steam vents (Stratum 1), strictly faucet-accounted!
                    let z_scale_64 = self.depth as f32 / 64.0;
                    let z_sky_spring = ((49.0 * z_scale_64).round() as usize).clamp(4, self.depth - 2);
                    let ls_top = &mut self.layers[z_sky_spring];
                    for &(sx_n, sy_n) in &[(0.50f32, 0.405f32), (0.385f32, 0.375f32), (0.615f32, 0.375f32)] {
                        let cx = (sx_n * wf) as usize;
                        let cy = (sy_n * hf) as usize;
                        let r = (w / 48).clamp(1, 3);
                        for y in cy.saturating_sub(r)..=(cy + r).min(h - 3) {
                            for x in cx.saturating_sub(r)..=(cx + r).min(w - 3) {
                                let idx = y * w + x;
                                let cur = ls_top.mass[SP_WATER][idx];
                                if cur < 980 && ls_top.mass[SP_ROCK][idx] < 500 {
                                    let room = H_CAP.saturating_sub(ls_top.cell_volume(idx)) as u16;
                                    let add = (980 - cur).min(120).min(room);
                                    if add > 0 {
                                        ls_top.mass[SP_WATER][idx] += add;
                                        let e_add = (add as i64) * SPECIES[SP_WATER].c * 28_000;
                                        ls_top.energy[idx] += e_add;
                                        ls_top.faucet_mass[SP_WATER] += add as i64;
                                        ls_top.faucet_energy += e_add;
                                    }
                                }
                            }
                        }
                    }
                    let z_abyss_vent = ((4.0 * z_scale_64).round() as usize).clamp(2, self.depth - 3);
                    let ls_bot = &mut self.layers[z_abyss_vent];
                    for &(vx_n, vy_n) in &[(0.42f32, 0.56f32), (0.58f32, 0.56f32)] {
                        let cx = (vx_n * wf) as usize;
                        let cy = (vy_n * hf) as usize;
                        let r = (w / 64).clamp(1, 2);
                        for y in cy.saturating_sub(r)..=(cy + r).min(h - 3) {
                            for x in cx.saturating_sub(r)..=(cx + r).min(w - 3) {
                                let idx = y * w + x;
                                if ls_bot.mass[SP_ROCK][idx] < 400 && ls_bot.mass[SP_STEAM][idx] < 320 {
                                    let room = H_CAP.saturating_sub(ls_bot.cell_volume(idx)) as u16;
                                    let add = (320 - ls_bot.mass[SP_STEAM][idx]).min(65).min(room);
                                    if add > 0 {
                                        ls_bot.mass[SP_STEAM][idx] += add;
                                        let e_th = (add as i64) * SPECIES[SP_STEAM].c * 43_000;
                                        let e_chem = (add as i64) * SPECIES[SP_STEAM].u0;
                                        ls_bot.energy[idx] += e_th;
                                        ls_bot.faucet_mass[SP_STEAM] += add as i64;
                                        ls_bot.faucet_energy += e_th + e_chem;
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {
                    // BIOME_GEOTHERMAL_BASIN: boiling hot spring feeds dense rising steam -> closed 3D hydrologic rain cycle!
                    let z_steam = ((4.0 * z_scale).round() as usize).clamp(2, self.depth - 3);
                    let ls = &mut self.layers[z_steam];
                    let cx = (0.50 * wf) as usize;
                    let cy = (0.50 * hf) as usize;
                    let r = (w / 32).clamp(2, 5);
                    for y in cy.saturating_sub(r)..=(cy + r).min(h - 3) {
                        for x in cx.saturating_sub(r)..=(cx + r).min(w - 3) {
                            let idx = y * w + x;
                            if ls.mass[SP_ROCK][idx] < 400 && ls.mass[SP_STEAM][idx] < 420 {
                                let room = H_CAP.saturating_sub(ls.cell_volume(idx)) as u16;
                                let add = (420 - ls.mass[SP_STEAM][idx]).min(90).min(room);
                                if add > 0 {
                                    ls.mass[SP_STEAM][idx] += add;
                                    let e_th = (add as i64) * SPECIES[SP_STEAM].c * 41_000;
                                    let e_chem = (add as i64) * SPECIES[SP_STEAM].u0;
                                    ls.energy[idx] += e_th;
                                    ls.faucet_mass[SP_STEAM] += add as i64;
                                    ls.faucet_energy += e_th + e_chem;
                                }
                            }
                        }
                    }
                }
            }
            return;
        }
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

    /// Apply physics-coupled diurnal solar heating/cooling across exposed surface & highland cells.
    /// Computes `solar = (day_time * TAU - FRAC_PI_2).sin()` (`-1` midnight, `+1` noon):
    /// - Night (`solar < -0.2`): gently cools exposed surface water/air toward freezing (midnight ice
    ///   crossings & damped wildfires).
    /// - Midday (`solar > 0.3`): warms exposed terrain & thaws thin ice (fiercer midday wildfires).
    /// Strictly accounts for all added/removed thermal energy in `faucet_energy` so `energy_error == 0`.
    pub fn apply_solar_cycle_3d(&mut self, day_time: f32, tick: u64) {
        if self.depth == 0 || self.width == 0 || self.height == 0 {
            return;
        }
        let solar = (day_time * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2).sin();
        if (-0.2..=0.3).contains(&solar) {
            return;
        }

        let len = self.width * self.height;
        let surf_l = if self.depth >= 2 { 1 } else { 0 };
        let has_overhang_layer = self.depth >= 3;
        let stride = 5usize;
        let phase = (tick as usize) % stride;

        if solar < -0.2 {
            let night_strength = ((-solar - 0.2) / 0.8).clamp(0.0, 1.0);
            let mut idx = phase;
            while idx < len {
                // Surface layer (skip cells sheltered underneath solid L2 plateaus/bridges)
                let surf_sheltered = has_overhang_layer
                    && (self.layers[2].mass[SP_ROCK][idx] >= 400
                        || self.layers[2].mass[SP_SOIL][idx] >= 400
                        || self.layers[2].mass[SP_WOOD][idx] >= 350);
                if !surf_sheltered {
                    let f = &mut self.layers[surf_l];
                    let c = f.cell_heat_capacity(idx);
                    if c > 0 && f.mass[SP_ROCK][idx] < 500 {
                        let t = f.cell_temp_ck(idx);
                        if t > 38_000 {
                            let cool_ck = (((t - 30_000) / 14).max(180) as f32 * night_strength) as i64;
                            let min_e = c * T_AMB;
                            let delta = (cool_ck * c).min((f.energy[idx] - min_e).max(0));
                            if delta > 0 {
                                f.energy[idx] -= delta;
                                f.faucet_energy -= delta;
                            }
                        } else if t > 26_600
                            && (f.mass[SP_WATER][idx] > 40
                                || f.mass[SP_STEAM][idx] > 20
                                || f.mass[SP_SMOKE][idx] > 20
                                || f.mass[SP_SOIL][idx] > 40)
                        {
                            let cool_ck = (240.0 * night_strength) as i64;
                            let min_e = c * 26_400;
                            let delta = (cool_ck * c).min((f.energy[idx] - min_e).max(0));
                            if delta > 0 {
                                f.energy[idx] -= delta;
                                f.faucet_energy -= delta;
                            }
                        }
                    }
                }
                // Exposed highland layer (L2) also cools at night
                if has_overhang_layer {
                    let f2 = &mut self.layers[2];
                    let c2 = f2.cell_heat_capacity(idx);
                    if c2 > 0 && f2.mass[SP_ROCK][idx] < 500 {
                        let t2 = f2.cell_temp_ck(idx);
                        if t2 > 26_600 {
                            let cool_ck = (200.0 * night_strength) as i64;
                            let min_e = c2 * 26_400;
                            let delta = (cool_ck * c2).min((f2.energy[idx] - min_e).max(0));
                            if delta > 0 {
                                f2.energy[idx] -= delta;
                                f2.faucet_energy -= delta;
                            }
                        }
                    }
                }
                idx += stride;
            }
        } else {
            // Midday warming (solar > 0.3): warm exposed terrain, thaw thin ice, intensify midday fires
            let sun_strength = ((solar - 0.3) / 0.7).clamp(0.0, 1.0);
            let mut idx = phase;
            while idx < len {
                let surf_sheltered = has_overhang_layer
                    && (self.layers[2].mass[SP_ROCK][idx] >= 400
                        || self.layers[2].mass[SP_SOIL][idx] >= 400
                        || self.layers[2].mass[SP_WOOD][idx] >= 350);
                if !surf_sheltered {
                    let f = &mut self.layers[surf_l];
                    let c = f.cell_heat_capacity(idx);
                    if c > 0 && f.mass[SP_ROCK][idx] < 800 {
                        let t = f.cell_temp_ck(idx);
                        let is_fuel = f.mass[SP_WOOD][idx] > 100
                            || f.mass[SP_PLANT][idx] > 100
                            || f.mass[SP_OIL][idx] > 80;
                        let is_thin_ice = f.mass[SP_ICE][idx] > 0 && f.mass[SP_ICE][idx] < 650;
                        if t > 44_000 {
                            let heat_ck = (320.0 * sun_strength) as i64;
                            let cap_e = c * (t + 500).min(T_MAX);
                            let delta = (heat_ck * c).min((cap_e - f.energy[idx]).max(0));
                            if delta > 0 {
                                f.energy[idx] += delta;
                                f.faucet_energy += delta;
                            }
                        } else if is_fuel || is_thin_ice || f.mass[SP_SOIL][idx] > 40 || f.mass[SP_WATER][idx] > 40 {
                            let target_t = if is_fuel { 32_500 } else { 30_200 };
                            if t < target_t {
                                let heat_ck = (220.0 * sun_strength) as i64;
                                let cap_e = c * target_t;
                                let delta = (heat_ck * c).min((cap_e - f.energy[idx]).max(0));
                                if delta > 0 {
                                    f.energy[idx] += delta;
                                    f.faucet_energy += delta;
                                }
                            }
                        }
                    }
                }
                if has_overhang_layer {
                    let f2 = &mut self.layers[2];
                    let c2 = f2.cell_heat_capacity(idx);
                    if c2 > 0 && f2.mass[SP_ROCK][idx] < 800 {
                        let t2 = f2.cell_temp_ck(idx);
                        if t2 < 30_200 {
                            let heat_ck = (180.0 * sun_strength) as i64;
                            let cap_e = c2 * 30_200;
                            let delta = (heat_ck * c2).min((cap_e - f2.energy[idx]).max(0));
                            if delta > 0 {
                                f2.energy[idx] += delta;
                                f2.faucet_energy += delta;
                            }
                        }
                    }
                }
                idx += stride;
            }
        }
    }

    /// Apply `[depth - 1, 5, H, W]` neural vertical interface predictions from `WorldModel3D` v4
    /// (`flux_z_water, flux_z_oil, flux_z_steam, flux_z_smoke, heat_z`) through an **exact
    /// antisymmetric integer conservation firewall**:
    /// - Donor clamp across all interfaces so no cell mass ever goes negative (`m >= 0`)
    /// - Receiver `H_CAP` headroom clamp so no cell ever exceeds `H_CAP`
    /// - Exact `+dm / -dm` across layer `z <-> z+1`
    /// - Exact `+dE / -dE` for both advected thermal enthalpy and 2nd-law vertical heat conduction `heat_z`
    /// Strictly preserves **`mass_error == 0 && energy_error == 0`** even under adversarial/NaN/Inf inputs!
    /// Returns `(gravity_mass_moved, buoyancy_mass_moved)`.
    pub fn apply_neural_3d_vertical_fluxes(&mut self, z_flux_f32: &[f32]) -> (i64, i64) {
        if self.depth < 2 || self.width == 0 || self.height == 0 {
            return (0, 0);
        }
        let d = self.depth;
        let n_if = d - 1;
        let len = self.width * self.height;
        let stride_z = 5 * len;
        if z_flux_f32.len() < n_if * stride_z {
            return (0, 0);
        }

        const Z_SPECIES: [usize; 4] = [SP_WATER, SP_OIL, SP_STEAM, SP_SMOKE];
        let mut q_if = vec![[0i32; 4]; n_if];
        let mut caps_pre = vec![0i64; d];
        let mut energy_pre = vec![0i64; d];
        let mut grav_moved = 0i64;
        let mut buoy_moved = 0i64;

        for idx in 0..len {
            // 1. Decode & sanitize raw neural vertical mass fluxes [n_if, 4] (positive = z -> z+1)
            let mut any_mass_flux = false;
            for z in 0..n_if {
                let base = z * stride_z;
                for k in 0..4 {
                    let raw_f = sanitize_f32(z_flux_f32[base + k * len + idx]) * 1000.0;
                    let qv = round_ties_even_i32(raw_f);
                    q_if[z][k] = qv;
                    if qv != 0 {
                        any_mass_flux = true;
                    }
                }
            }

            if any_mass_flux {
                // 2. Exact integer donor clamping per layer & species (guarantees non-negative masses)
                for l in 0..d {
                    for (k, &sp) in Z_SPECIES.iter().enumerate() {
                        let out_up = if l < n_if { q_if[l][k].max(0) as i64 } else { 0 };
                        let out_dn = if l > 0 { (-q_if[l - 1][k]).max(0) as i64 } else { 0 };
                        let out_tot = out_up + out_dn;
                        if out_tot > 0 {
                            let avail = self.layers[l].mass[sp][idx] as i64;
                            if out_tot > avail {
                                if l < n_if && q_if[l][k] > 0 {
                                    q_if[l][k] = ((q_if[l][k] as i64) * avail / out_tot) as i32;
                                }
                                if l > 0 && q_if[l - 1][k] < 0 {
                                    q_if[l - 1][k] = -(((-q_if[l - 1][k]) as i64) * avail / out_tot) as i32;
                                }
                            }
                        }
                    }
                }

                // 3. Exact integer receiver H_CAP headroom clamping per layer
                for l in 0..d {
                    let mut in_tot = 0i64;
                    if l > 0 {
                        for k in 0..4 {
                            in_tot += q_if[l - 1][k].max(0) as i64;
                        }
                    }
                    if l < n_if {
                        for k in 0..4 {
                            in_tot += (-q_if[l][k]).max(0) as i64;
                        }
                    }
                    if in_tot > 0 {
                        let headroom = H_CAP.saturating_sub(self.layers[l].cell_volume(idx)) as i64;
                        if in_tot > headroom {
                            if l > 0 {
                                for k in 0..4 {
                                    if q_if[l - 1][k] > 0 {
                                        q_if[l - 1][k] = ((q_if[l - 1][k] as i64) * headroom / in_tot) as i32;
                                    }
                                }
                            }
                            if l < n_if {
                                for k in 0..4 {
                                    if q_if[l][k] < 0 {
                                        q_if[l][k] = -(((-q_if[l][k]) as i64) * headroom / in_tot) as i32;
                                    }
                                }
                            }
                        }
                    }
                }

                // 4. Snapshot pre-transport heat capacities & energies for exact proportional enthalpy advection
                for l in 0..d {
                    caps_pre[l] = self.layers[l].cell_heat_capacity(idx);
                    energy_pre[l] = self.layers[l].energy[idx].max(0);
                }

                // 5. Apply exact antisymmetric integer mass + advected thermal energy across each interface
                for z in 0..n_if {
                    let mut e_adv_up = 0i64;
                    for (k, &sp) in Z_SPECIES.iter().enumerate() {
                        let qv = q_if[z][k];
                        if qv == 0 {
                            continue;
                        }
                        if qv > 0 {
                            let q_u16 = qv as u16;
                            self.layers[z].mass[sp][idx] -= q_u16;
                            self.layers[z + 1].mass[sp][idx] += q_u16;
                            if caps_pre[z] > 0 {
                                let e_xfer = ((qv as i128)
                                    * (SPECIES[sp].c as i128)
                                    * (energy_pre[z] as i128)
                                    / (caps_pre[z] as i128)) as i64;
                                e_adv_up += e_xfer;
                            }
                            if k == 2 || k == 3 {
                                buoy_moved += qv as i64;
                            }
                        } else {
                            let q_abs = (-qv) as u16;
                            self.layers[z + 1].mass[sp][idx] -= q_abs;
                            self.layers[z].mass[sp][idx] += q_abs;
                            if caps_pre[z + 1] > 0 {
                                let e_xfer = ((q_abs as i128)
                                    * (SPECIES[sp].c as i128)
                                    * (energy_pre[z + 1] as i128)
                                    / (caps_pre[z + 1] as i128)) as i64;
                                e_adv_up -= e_xfer;
                            }
                            if k == 0 || k == 1 {
                                grav_moved += q_abs as i64;
                            }
                        }
                    }
                    if e_adv_up != 0 {
                        self.layers[z].energy[idx] -= e_adv_up;
                        self.layers[z + 1].energy[idx] += e_adv_up;
                    }
                }
            }

            // 6. Apply exact antisymmetric 2nd-law vertical heat conduction (`heat_z`, channel 4)
            for z in 0..n_if {
                let g_raw_f = sanitize_f32(z_flux_f32[z * stride_z + 4 * len + idx]);
                if g_raw_f == 0.0 {
                    continue;
                }
                let c_lo = self.layers[z].cell_heat_capacity(idx);
                let c_hi = self.layers[z + 1].cell_heat_capacity(idx);
                if c_lo <= 0 || c_hi <= 0 {
                    continue;
                }
                let g_raw = (g_raw_f as f64 * 1e6).clamp(-1e15, 1e15).round() as i64;
                let t_lo = self.layers[z].energy[idx] / c_lo;
                let t_hi = self.layers[z + 1].energy[idx] / c_hi;
                let dt = t_lo - t_hi;
                if dt == 0 {
                    continue;
                }
                let max_g = ((dt.unsigned_abs() as u128)
                    * (c_lo as u128)
                    * (c_hi as u128)
                    / (4 * ((c_lo + c_hi) as u128))) as i64;
                let g = if dt > 0 && g_raw > 0 {
                    let avail = (self.layers[z].energy[idx] - c_lo * T_MIN).max(0) / 2;
                    let room = (c_hi * T_MAX - self.layers[z + 1].energy[idx]).max(0) / 2;
                    g_raw.min(max_g).min(avail).min(room)
                } else if dt < 0 && g_raw < 0 {
                    let avail = (self.layers[z + 1].energy[idx] - c_hi * T_MIN).max(0) / 2;
                    let room = (c_lo * T_MAX - self.layers[z].energy[idx]).max(0) / 2;
                    g_raw.max(-max_g).max(-avail).max(-room)
                } else {
                    0
                };
                if g != 0 {
                    self.layers[z].energy[idx] -= g;
                    self.layers[z + 1].energy[idx] += g;
                }
            }
        }

        if self.depth >= 12 {
            grav_moved += self.step_hydrologic_and_collapse_3d();
        }

        (grav_moved, buoy_moved)
    }

    /// Closed 3D Hydrologic Condensation (`z >= depth - 3`) + 3D Destructible Bridge/Arch Structural Collapse (`z >= 2`).
    /// Called by BOTH `step_vertical_exchange_3d` and `apply_neural_3d_vertical_fluxes` when `self.depth >= 12`.
    /// Preserves **exact integer mass & energy conservation** (`mass_error == 0 && energy_error == 0`).
    fn step_hydrologic_and_collapse_3d(&mut self) -> i64 {
        let d = self.depth;
        let w = self.width;
        let h = self.height;
        if d < 12 || w == 0 || h == 0 {
            return 0;
        }
        let len = w * h;
        let mut collapse_moved = 0i64;

        // 1. Closed 3D Hydrologic Condensation in upper troposphere (`z >= depth - 3`, or deeper under high precipitation):
        //    High-altitude SP_STEAM radiatively cools (faucet_energy accounted cleanly -> energy_error == 0)
        //    and condenses into SP_WATER (or SP_ICE when T < 273.15 K) with exact dm_steam == dm_condensed!
        let precip = self.world_laws.precipitation.clamp(0.1, 4.0);
        let amb_shift_sky = (self.world_laws.ambient_temp_shift * 0.5).round() as i64;
        let cloud_layers = if precip > 1.5 { 5usize } else { 3usize };
        let z_cloud_lo = d.saturating_sub(cloud_layers);
        let is_glacial = self.biome_id == BIOME_GLACIER_GORGE;
        for z in z_cloud_lo..d {
            let z_from_top = (d - 1) - z;
            let base_sky: i64 = if is_glacial {
                match z_from_top {
                    0 => 25_600,
                    1 => 26_400,
                    _ => 27_000,
                }
            } else {
                match z_from_top {
                    0 => 26_800,
                    1 => 29_800,
                    _ => 32_800,
                }
            };
            let t_sky = if precip > 1.5 && self.world_laws.ambient_temp_shift > -2000.0 {
                // High-precipitation temperate deluge: clouds condense into heavy liquid rain
                (base_sky + amb_shift_sky).max(28_200).clamp(T_MIN, T_MAX)
            } else {
                (base_sky + amb_shift_sky).clamp(T_MIN, T_MAX)
            };
            let f = &mut self.layers[z];
            for idx in 0..len {
                let steam = f.mass[SP_STEAM][idx];
                if steam < 8 {
                    continue;
                }
                // 1a. Radiative cooling toward upper-atmosphere sky temperature
                let c = f.cell_heat_capacity(idx);
                if c > 0 {
                    let t = f.cell_temp_ck(idx);
                    if t > t_sky {
                        let cool_ck = ((t - t_sky) / 2).max(1_200).min(t - t_sky);
                        let min_e = c * T_MIN;
                        let delta_cool = (cool_ck * c).min((f.energy[idx] - min_e).max(0));
                        if delta_cool > 0 {
                            f.energy[idx] -= delta_cool;
                            f.faucet_energy -= delta_cool;
                        }
                    }
                }

                // 1b. Exact mass-preserving condensation (SP_STEAM -> SP_WATER or SP_ICE)
                let t_now = f.cell_temp_ck(idx);
                if t_now < 37_000 {
                    let target_sp = if t_now < 27_315 { SP_ICE } else { SP_WATER };
                    let dm = if (precip - 1.0).abs() <= 1e-4 {
                        ((steam / 2).max(1)).min(steam)
                    } else {
                        let frac = (0.50 * precip).clamp(0.05, 0.95);
                        ((steam as f32 * frac).round() as u16).max(1).min(steam)
                    };
                    if dm > 0 {
                        let e_before = f.energy[idx];
                        f.mass[SP_STEAM][idx] -= dm;
                        f.mass[target_sp][idx] += dm;
                        let c_new = f.cell_heat_capacity(idx);
                        let e_new = c_new * t_now.clamp(T_MIN, T_MAX);
                        let delta_e_th = e_new - e_before;
                        let delta_e_chem = (dm as i64) * (SPECIES[target_sp].u0 - SPECIES[SP_STEAM].u0);
                        f.energy[idx] = e_new;
                        // Radiative release of exact latent + heat-capacity adjustment -> energy_error == 0!
                        f.faucet_energy += delta_e_th + delta_e_chem;
                    }
                }
            }
        }

        // 2. 3D Destructible Bridge/Arch Structural Collapse (`z >= 2`):
        //    When an overhanging SP_WOOD (burning T > 550 K) or SP_ICE (melting T > 278 K, or loose
        //    high-altitude snow precipitate) voxel has open air below (solid_below < 0.25) AND fewer
        //    than 2 solid horizontal 4-neighbors, conservatively drop its mass & enthalpy into `z - 1`!
        for z in 2..d {
            let prev_base = z * len;
            let prev_t_slice = self.prev_temp.get(prev_base..prev_base + len);
            let (lo_slice, hi_slice) = self.layers.split_at_mut(z);
            let lo = &mut lo_slice[z - 1];
            let hi = &mut hi_slice[0];

            for y in 0..h {
                let row = y * w;
                for x in 0..w {
                    let idx = row + x;
                    let wood = hi.mass[SP_WOOD][idx];
                    let ice = hi.mass[SP_ICE][idx];
                    if wood < 15 && ice < 15 {
                        continue;
                    }
                    // Must have open air directly below (`solid_below < 0.25`)
                    if field_solid_density(lo, idx) >= 0.25 {
                        continue;
                    }
                    let t_ck = hi.cell_temp_ck(idx);
                    let prev_t_ck = prev_t_slice.map(|s| s[idx] as i64).unwrap_or(t_ck);
                    let t_peak = t_ck.max(prev_t_ck);
                    let wood_weak = wood >= 15 && t_peak > 55_000;
                    let ice_weak = ice >= 15 && t_peak > 27_800;
                    let snow_fall = z >= z_cloud_lo && ice >= 15 && ice < 400 && t_ck < 27_315;
                    if !wood_weak && !ice_weak && !snow_fall {
                        continue;
                    }
                    // Count solid 4-neighbors at same elevation `z`
                    let mut solid_nb = 0u8;
                    if x > 0 && field_solid_density(hi, idx - 1) >= 0.25 {
                        solid_nb += 1;
                    }
                    if x + 1 < w && field_solid_density(hi, idx + 1) >= 0.25 {
                        solid_nb += 1;
                    }
                    if y > 0 && field_solid_density(hi, idx - w) >= 0.25 {
                        solid_nb += 1;
                    }
                    if y + 1 < h && field_solid_density(hi, idx + w) >= 0.25 {
                        solid_nb += 1;
                    }
                    if solid_nb >= 2 {
                        continue;
                    }

                    // Conservatively drop weakened overhanging mass & enthalpy downward into `z - 1`
                    for &(sp, active) in &[(SP_WOOD, wood_weak), (SP_ICE, ice_weak || snow_fall)] {
                        if !active {
                            continue;
                        }
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
                                let e_prop = ((q as i128)
                                    * (SPECIES[sp].c as i128)
                                    * (hi.energy[idx].max(0) as i128)
                                    / (c_hi as i128)) as i64;
                                e_prop.min(hi.energy[idx].max(0))
                            } else {
                                0
                            };
                            hi.mass[sp][idx] -= q;
                            lo.mass[sp][idx] += q;
                            hi.energy[idx] -= e_xfer;
                            lo.energy[idx] += e_xfer;
                            collapse_moved += q as i64;
                        }
                    }
                }
            }
        }

        collapse_moved
    }

    /// Exact integer 3D vertical exchange between adjacent strata (`z <-> z+1`):
    /// - **Hydrologic Condensation & Structural Collapse (`depth >= 12`):** Cloud-top condensation + weakened arch collapse!
    /// - **Gravity (`z+1 -> z`):** Liquid `water` & `oil` fall down through open air (3D Waterfalls, Rain & Oil Cascades!)
    /// - **Buoyancy (`z -> z+1`):** Hot `steam` & `smoke` rise upward into upper strata (3D Plume Columns & Clouds!)
    /// - **Vertical Heat Conduction (`z <-> z+1`):** Rising wildfire heat ignites overhanging timber Skybridges!
    /// Guarantees **exact 0 mass error and 0 energy error** across all 3D layers.
    fn step_vertical_exchange_3d(&mut self) -> (i64, i64) {
        if self.depth < 2 {
            return (0, 0);
        }
        let len = self.width * self.height;
        let mut grav_moved = 0i64;
        let mut buoy_moved = 0i64;

        if self.depth >= 12 {
            grav_moved += self.step_hydrologic_and_collapse_3d();
        }
        let grav_min_mass: u16 = if self.depth >= 12 { 8 } else { 20 };
        let laws = self.world_laws;
        let standard_laws = laws.is_standard();
        let grav_fall_frac =
            (0.3333 * (laws.gravity_g.clamp(0.15, 2.5) / laws.viscosity.clamp(0.25, 4.0)))
                .clamp(0.08, 0.85);
        let buoy_rise_frac = (0.3333
            * (1.0 / laws.gravity_g.clamp(0.25, 2.5)).sqrt()
            * (1.0 + (laws.ambient_temp_shift.max(0.0) / 11_000.0)))
        .clamp(0.10, 0.85);
        let cond_scale = laws.thermal_cond.clamp(0.25, 4.0);

        for z in 0..(self.depth - 1) {
            // 3D Altitude-Coupled Gravity (`depth >= 48`): softly scale effective gravity to 0.45g
            // above `0.65 * depth` so high-altitude Sky-Island waterfalls & plumes hang buoyantly!
            let high_alt_buoyant = self.depth >= 48 && (z as f32) >= 0.65 * (self.depth as f32);
            let z_grav_fall_frac = if high_alt_buoyant {
                (grav_fall_frac * 0.45).clamp(0.05, 0.85)
            } else {
                grav_fall_frac
            };
            let z_buoy_rise_frac = if high_alt_buoyant {
                (buoy_rise_frac * 1.25).clamp(0.10, 0.85)
            } else {
                buoy_rise_frac
            };
            let use_std_grav = standard_laws && !high_alt_buoyant;
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
                        if avail > grav_min_mass {
                            let room = H_CAP.saturating_sub(lo.cell_volume(idx)) as u16;
                            let q = if use_std_grav {
                                ((avail / 3).max(1)).min(room)
                            } else {
                                ((avail as f32 * z_grav_fall_frac).round() as u16)
                                    .max(1)
                                    .min(avail)
                                    .min(room)
                            };
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
                            let q = if use_std_grav {
                                ((avail / 3).max(1)).min(room)
                            } else {
                                ((avail as f32 * z_buoy_rise_frac).round() as u16)
                                    .max(1)
                                    .min(avail)
                                    .min(room)
                            };
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
                        let base_rate = if dt > 0 { 22.0 } else { 64.0 };
                        let rate = if standard_laws {
                            if dt > 0 { 22 } else { 64 }
                        } else {
                            ((base_rate / cond_scale).round() as i64).max(6)
                        };
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

    /// Evolving Persistent 3D World pass (called periodically when `depth >= 12`, scaled by `evo_speed`):
    /// 1. **Lava / Superheated Magma Quenching → Permanent Basalt Landforms**: Hot `SP_OIL` or hot
    ///    `SP_ROCK`-adjacent magma (`T > 650 K`) meeting `SP_WATER` or `SP_ICE` conservatively forges
    ///    solid `SP_ROCK` + `SP_STEAM` (`dm_in == dm_out`, exact thermal enthalpy balance).
    /// 2. **Ash + Rain/Moisture → Organic Regrowth (Living Forest Succession)**: Sunlit soil/ash
    ///    surfaces (`SP_ASH > 0.05` or `SP_SOIL > 0.25`, `SP_WATER > 0.08`, mild `T in 278..330 K`)
    ///    conservatively sprout `SP_PLANT`, and dense `SP_PLANT` matures upward into `SP_WOOD` trees.
    /// 3. **Hydraulic Erosion**: Fast water (`SP_WATER > 0.4`) adjacent to soil redistributes small
    ///    `SP_SOIL` mass & enthalpy conservatively downstream.
    /// 4. **Glacial Expansion**: Cold high-moisture voxels (`T < 268 K`, `SP_WATER`/`SP_STEAM`)
    ///    crystallize into permanent glacial `SP_ICE` caps.
    ///
    /// Preserves **exact integer 3D mass & energy conservation** (`mass_error == 0 && energy_error == 0`).
    pub fn step_world_evolution_3d(&mut self, evo_speed: f32) {
        let (w, h, d) = (self.width, self.height, self.depth);
        if d < 12 || w < 4 || h < 4 || !evo_speed.is_finite() || evo_speed <= 0.0 {
            return;
        }
        let speed = evo_speed.clamp(0.1, 32.0);
        self.evolution_stats.world_epoch = self.evolution_stats.world_epoch.saturating_add(1);
        let epoch = self.evolution_stats.world_epoch;
        let len = w * h;
        let stride = if speed >= 4.0 { 1usize } else { 2usize };
        let parity = (epoch as usize) % stride;

        let quench_rate = (20.0 * speed).round().clamp(8.0, 240.0) as u16;
        let grow_rate = (14.0 * speed).round().clamp(6.0, 160.0) as u16;
        let erode_rate = (8.0 * speed).round().clamp(3.0, 80.0) as u16;
        let glacier_rate = (18.0 * speed).round().clamp(6.0, 180.0) as u16;

        // Exact antisymmetric mass + proportional enthalpy transfer from `(za, ia)` -> `(zb, ib)`.
        #[inline]
        fn transfer_species_3d(
            layers: &mut [Field],
            za: usize,
            ia: usize,
            zb: usize,
            ib: usize,
            sp: usize,
            q: u16,
        ) {
            if q == 0 || (za == zb && ia == ib) {
                return;
            }
            let c_a = layers[za].cell_heat_capacity(ia);
            let e_a = layers[za].energy[ia].max(0);
            let e_xfer = if q == layers[za].mass[sp][ia] && layers[za].cell_volume(ia) == q as u32 {
                layers[za].energy[ia]
            } else if c_a > 0 {
                let prop = ((q as i128) * (SPECIES[sp].c as i128) * (e_a as i128) / (c_a as i128)) as i64;
                prop.min(e_a)
            } else {
                0
            };
            layers[za].mass[sp][ia] -= q;
            layers[zb].mass[sp][ib] += q;
            layers[za].energy[ia] -= e_xfer;
            layers[zb].energy[ib] += e_xfer;
        }

        // Exact single-cell conserved conversion (`sum(deltas) == 0`), maintaining `mass_error == 0 && energy_error == 0`.
        #[inline]
        fn convert_cell_conserved(
            f: &mut Field,
            idx: usize,
            deltas: &[(usize, i32)],
            keep_temp_ck: Option<i64>,
        ) {
            let e_before = f.energy[idx];
            let mut delta_u0 = 0i64;
            for &(sp, dm) in deltas {
                if dm > 0 {
                    f.mass[sp][idx] = f.mass[sp][idx].saturating_add(dm as u16);
                } else if dm < 0 {
                    f.mass[sp][idx] -= (-dm) as u16;
                }
                delta_u0 += (dm as i64) * SPECIES[sp].u0;
            }
            let c_new = f.cell_heat_capacity(idx);
            let e_min = c_new * T_MIN;
            let e_max = c_new * T_MAX;
            let e_new = match keep_temp_ck {
                Some(t_ck) => (c_new * t_ck.clamp(T_MIN, T_MAX)).clamp(e_min, e_max),
                None => (e_before - delta_u0).clamp(e_min, e_max),
            };
            f.energy[idx] = e_new;
            let residual = (e_new - e_before) + delta_u0;
            if residual != 0 {
                f.faucet_energy += residual;
            }
        }

        // Precompute highest solid surface per `(x, y)` column for open-sky sunlit regrowth checks
        let mut sunlit_z = vec![0u8; len];
        for idx in 0..len {
            let mut top = 0u8;
            for z in (1..d).rev() {
                if field_solid_density(&self.layers[z], idx) >= 0.35 {
                    top = z as u8;
                    break;
                }
            }
            sunlit_z[idx] = top;
        }

        // 1. LAVA / SUPERHEATED MAGMA QUENCHING -> PERMANENT BASALT LANDFORMS (`SP_ROCK + SP_STEAM`)
        let mut basalt_forged = 0u64;
        for z in 1..(d - 1) {
            for y in 1..(h - 1) {
                let row = y * w;
                for x in 1..(w - 1) {
                    if (x + y) % stride != parity {
                        continue;
                    }
                    let idx = row + x;
                    let water = self.layers[z].mass[SP_WATER][idx];
                    let ice = self.layers[z].mass[SP_ICE][idx];
                    let rock = self.layers[z].mass[SP_ROCK][idx];
                    if (water < 12 && ice < 12) || rock >= 980 {
                        continue;
                    }
                    let w_sp = if water >= ice { SP_WATER } else { SP_ICE };
                    let w_avail = self.layers[z].mass[w_sp][idx];
                    let t_here = self.layers[z].cell_temp_ck(idx);
                    let oil_here = self.layers[z].mass[SP_OIL][idx];

                    // 1a. Same-cell hot oil or hot magma rock (> 650 K) meeting water/ice
                    if t_here > 65_000 && oil_here >= 12 {
                        let q = oil_here.min(w_avail).min(quench_rate).min(980 - rock);
                        if q >= 2 {
                            convert_cell_conserved(
                                &mut self.layers[z],
                                idx,
                                &[
                                    (SP_OIL, -(q as i32)),
                                    (w_sp, -(q as i32)),
                                    (SP_ROCK, q as i32),
                                    (SP_STEAM, q as i32),
                                ],
                                None,
                            );
                            basalt_forged += 1;
                            continue;
                        }
                    } else if t_here > 65_000 && rock >= 180 {
                        let q = (w_avail / 2).min(quench_rate).min(980 - rock);
                        if q >= 2 {
                            convert_cell_conserved(
                                &mut self.layers[z],
                                idx,
                                &[
                                    (w_sp, -((2 * q) as i32)),
                                    (SP_ROCK, q as i32),
                                    (SP_STEAM, q as i32),
                                ],
                                None,
                            );
                            basalt_forged += 1;
                            continue;
                        }
                    }

                    // 1b. Adjacent hot magma rock or superheated oil (> 650 K) in 6-neighborhood
                    let neighbors: [(usize, usize); 6] = [
                        (z, idx - 1),
                        (z, idx + 1),
                        (z, idx - w),
                        (z, idx + w),
                        (z - 1, idx),
                        (z + 1, idx),
                    ];
                    let mut hot_nb: Option<(usize, usize, bool)> = None;
                    for &(nz, nidx) in &neighbors {
                        let nf = &self.layers[nz];
                        let n_oil = nf.mass[SP_OIL][nidx];
                        let n_rock = nf.mass[SP_ROCK][nidx];
                        if n_oil < 12 && n_rock < 200 {
                            continue;
                        }
                        if nf.cell_temp_ck(nidx) > 65_000 {
                            hot_nb = Some((nz, nidx, n_oil >= 12));
                            break;
                        }
                    }
                    if let Some((nz, nidx, has_oil)) = hot_nb {
                        if has_oil {
                            let n_oil = self.layers[nz].mass[SP_OIL][nidx];
                            let room = H_CAP.saturating_sub(self.layers[z].cell_volume(idx)) as u16;
                            let q = n_oil.min(w_avail).min(quench_rate).min(room).min(980 - rock);
                            if q >= 2 {
                                transfer_species_3d(&mut self.layers, nz, nidx, z, idx, SP_OIL, q);
                                convert_cell_conserved(
                                    &mut self.layers[z],
                                    idx,
                                    &[
                                        (SP_OIL, -(q as i32)),
                                        (w_sp, -(q as i32)),
                                        (SP_ROCK, q as i32),
                                        (SP_STEAM, q as i32),
                                    ],
                                    None,
                                );
                                basalt_forged += 1;
                            }
                        } else {
                            let q = (w_avail / 2).min(quench_rate).min(980 - rock);
                            if q >= 2 {
                                // Pull quenching heat from hot magma neighbor antisymmetrically
                                let c_nb = self.layers[nz].cell_heat_capacity(nidx);
                                if c_nb > 0 {
                                    let avail_e = (self.layers[nz].energy[nidx] - c_nb * 42_000).max(0);
                                    let heat_xfer = avail_e.min((q as i64) * 900_000);
                                    if heat_xfer > 0 {
                                        self.layers[nz].energy[nidx] -= heat_xfer;
                                        self.layers[z].energy[idx] += heat_xfer;
                                    }
                                }
                                convert_cell_conserved(
                                    &mut self.layers[z],
                                    idx,
                                    &[
                                        (w_sp, -((2 * q) as i32)),
                                        (SP_ROCK, q as i32),
                                        (SP_STEAM, q as i32),
                                    ],
                                    None,
                                );
                                basalt_forged += 1;
                            }
                        }
                    }
                }
            }
        }
        self.evolution_stats.basalt_forged = self.evolution_stats.basalt_forged.saturating_add(basalt_forged);

        // 2. ASH + RAIN/MOISTURE -> ORGANIC REGROWTH (LIVING FOREST SUCCESSION)
        let mut saplings_grown = 0u64;
        for y in 1..(h - 1) {
            let row = y * w;
            for x in 1..(w - 1) {
                if (x + y) % stride != parity {
                    continue;
                }
                let idx = row + x;
                let z = sunlit_z[idx] as usize;
                if z == 0 || z >= d - 1 {
                    continue;
                }
                let t_ck = self.layers[z].cell_temp_ck(idx);
                if !(27_800..=33_000).contains(&t_ck) {
                    continue;
                }
                let ash = self.layers[z].mass[SP_ASH][idx];
                let soil = self.layers[z].mass[SP_SOIL][idx];
                let water = self.layers[z].mass[SP_WATER][idx];
                let plant = self.layers[z].mass[SP_PLANT][idx];

                // 2a. Sprouting on sunlit ash/soil with moisture (`SP_WATER + SP_ASH/SP_SOIL -> SP_PLANT`)
                if (ash > 50 || soil > 250) && plant < 850 {
                    let src_sp = if ash > 50 { SP_ASH } else { SP_SOIL };
                    let avail_s = if src_sp == SP_ASH { ash } else { soil.saturating_sub(200) };
                    if water > 80 && avail_s >= 4 {
                        let q = (water / 3).min(avail_s / 3).min(grow_rate).min((850 - plant) / 2);
                        if q >= 2 {
                            convert_cell_conserved(
                                &mut self.layers[z],
                                idx,
                                &[
                                    (SP_WATER, -(q as i32)),
                                    (src_sp, -(q as i32)),
                                    (SP_PLANT, (2 * q) as i32),
                                ],
                                Some(t_ck),
                            );
                            saplings_grown += 1;
                        }
                    } else if avail_s >= 6 {
                        // Moisture wicking from direct rain film above (`z+1`) or riverbank neighbor
                        let room = H_CAP.saturating_sub(self.layers[z].cell_volume(idx)) as u16;
                        if room >= 4 {
                            let wet_nbs: [(usize, usize); 5] = [
                                (z + 1, idx),
                                (z, idx - 1),
                                (z, idx + 1),
                                (z, idx - w),
                                (z, idx + w),
                            ];
                            for &(wz, widx) in &wet_nbs {
                                let nw = self.layers[wz].mass[SP_WATER][widx];
                                if nw > 180 {
                                    let q = ((nw - 100) / 5)
                                        .min(avail_s / 4)
                                        .min(room)
                                        .min(grow_rate / 2 + 1)
                                        .min((850 - plant) / 2);
                                    if q >= 2 {
                                        transfer_species_3d(&mut self.layers, wz, widx, z, idx, SP_WATER, q);
                                        convert_cell_conserved(
                                            &mut self.layers[z],
                                            idx,
                                            &[
                                                (SP_WATER, -(q as i32)),
                                                (src_sp, -(q as i32)),
                                                (SP_PLANT, (2 * q) as i32),
                                            ],
                                            Some(t_ck),
                                        );
                                        saplings_grown += 1;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }

                // 2b. Mature dense `SP_PLANT` upward into `SP_WOOD` trees
                let plant_now = self.layers[z].mass[SP_PLANT][idx];
                if plant_now >= 450 && z + 1 < d - 1 && self.layers[z + 1].mass[SP_ROCK][idx] < 200 {
                    let room_up = H_CAP.saturating_sub(self.layers[z + 1].cell_volume(idx)) as u16;
                    let wood_up = self.layers[z + 1].mass[SP_WOOD][idx];
                    let q_up = (plant_now / 8)
                        .min(grow_rate)
                        .min(room_up)
                        .min(720u16.saturating_sub(wood_up));
                    if q_up >= 4 {
                        transfer_species_3d(&mut self.layers, z, idx, z + 1, idx, SP_PLANT, q_up);
                        let t_up = self.layers[z + 1].cell_temp_ck(idx);
                        convert_cell_conserved(
                            &mut self.layers[z + 1],
                            idx,
                            &[(SP_PLANT, -(q_up as i32)), (SP_WOOD, q_up as i32)],
                            Some(t_up),
                        );
                        saplings_grown += 1;
                    }
                }
            }
        }
        self.evolution_stats.saplings_grown = self.evolution_stats.saplings_grown.saturating_add(saplings_grown);

        // 3. HYDRAULIC EROSION: Fast water (`SP_WATER > 400`) adjacent to soil moves small `SP_SOIL` downstream
        let mut erosion_events = 0u64;
        for z in 1..(d - 1) {
            for y in 2..(h - 2) {
                let row = y * w;
                for x in 2..(w - 2) {
                    if (x + y) % stride != parity {
                        continue;
                    }
                    let idx = row + x;
                    if self.layers[z].mass[SP_WATER][idx] <= 400 {
                        continue;
                    }
                    let fe = self.layers[z].flow_e[idx];
                    let fs = self.layers[z].flow_s[idx];
                    let (dx, dy): (isize, isize) = if fe.abs() > fs.abs() && fe.abs() > 15 {
                        (if fe > 0 { 1 } else { -1 }, 0)
                    } else if fs.abs() > 15 {
                        (0, if fs > 0 { 1 } else { -1 })
                    } else {
                        (0, 1)
                    };
                    let dst_idx = ((y as isize + dy) as usize) * w + ((x as isize + dx) as usize);
                    if self.layers[z].mass[SP_ROCK][dst_idx] >= 700 {
                        continue;
                    }
                    let room = H_CAP.saturating_sub(self.layers[z].cell_volume(dst_idx)) as u16;
                    if room < 4 {
                        continue;
                    }
                    // Bank neighbors perpendicular/upstream to water cell
                    let bank_cands = [idx - 1, idx + 1, idx - w, idx + w];
                    for &src_idx in &bank_cands {
                        if src_idx == dst_idx {
                            continue;
                        }
                        let soil = self.layers[z].mass[SP_SOIL][src_idx];
                        if soil >= 220 {
                            let q = ((soil - 160) / 10).min(erode_rate).min(room);
                            if q >= 2 {
                                transfer_species_3d(&mut self.layers, z, src_idx, z, dst_idx, SP_SOIL, q);
                                erosion_events += 1;
                                break;
                            }
                        }
                    }
                }
            }
        }
        self.evolution_stats.erosion_events = self.evolution_stats.erosion_events.saturating_add(erosion_events);

        // 4. GLACIAL EXPANSION: Cold high-moisture voxels (`T < 268 K`, `SP_WATER`/`SP_STEAM`) -> glacial `SP_ICE` caps
        let mut glaciers_formed = 0u64;
        for z in 1..d {
            let f = &mut self.layers[z];
            for y in 1..(h - 1) {
                let row = y * w;
                for x in 1..(w - 1) {
                    if (x + y) % stride != parity {
                        continue;
                    }
                    let idx = row + x;
                    let water = f.mass[SP_WATER][idx];
                    let steam = f.mass[SP_STEAM][idx];
                    let ice = f.mass[SP_ICE][idx];
                    if (water < 30 && steam < 20) || ice >= 980 {
                        continue;
                    }
                    let t_ck = f.cell_temp_ck(idx);
                    if t_ck < 26_800 {
                        let src_sp = if water >= steam { SP_WATER } else { SP_STEAM };
                        let avail = f.mass[src_sp][idx];
                        let q = (avail / 3).max(1).min(glacier_rate).min(980 - ice);
                        if q > 0 {
                            convert_cell_conserved(
                                f,
                                idx,
                                &[(src_sp, -(q as i32)), (SP_ICE, q as i32)],
                                Some(t_ck),
                            );
                            glaciers_formed += 1;
                        }
                    }
                }
            }
        }
        self.evolution_stats.glaciers_formed = self.evolution_stats.glaciers_formed.saturating_add(glaciers_formed);
    }

    /// Living World Epoch classifier driven by `evolution_stats` (`basalt_forged`, `saplings_grown`,
    /// `glaciers_formed`, `world_epoch`).
    pub fn evolution_epoch_name(&self) -> &'static str {
        let ev = self.evolution_stats;
        let total_events = ev
            .basalt_forged
            .saturating_add(ev.saplings_grown)
            .saturating_add(ev.glaciers_formed);
        if total_events < 15 && ev.world_epoch < 15 {
            return "EPOCH I: PRISTINE";
        }
        if ev.glaciers_formed >= 15
            && ev.glaciers_formed > ev.basalt_forged
            && ev.glaciers_formed > ev.saplings_grown
        {
            "EPOCH IV: GLACIAL AGE"
        } else if ev.saplings_grown >= 15 && ev.saplings_grown >= ev.basalt_forged {
            "EPOCH III: VERDANT REGROWTH"
        } else if ev.basalt_forged >= 15 {
            "EPOCH II: VOLCANIC FORGE"
        } else if ev.world_epoch >= 180 {
            "EPOCH IV: GLACIAL AGE"
        } else if ev.world_epoch >= 90 {
            "EPOCH III: VERDANT REGROWTH"
        } else if ev.world_epoch >= 15 {
            "EPOCH II: VOLCANIC FORGE"
        } else {
            "EPOCH I: PRISTINE"
        }
    }

    /// Conservative subterranean tunnel carving for `CLASS_BURROWER` (`Terramole`):
    /// Displaces `SP_SOIL` and soft `SP_ROCK` (up to 60% per step) in a small sphere around `(gx, gy, gz)`
    /// (strictly respecting the `z >= 1` bedrock floor and outer perimeter cliffs) into adjacent
    /// perimeter / behind-the-burrower cells via exact antisymmetric integer mass & proportional
    /// enthalpy transfer so **`mass_error == 0 && energy_error == 0` ALWAYS**!
    /// Returns total mass units conservatively displaced.
    pub fn carve_tunnel_3d(
        &mut self,
        gx: f32,
        gy: f32,
        gz: f32,
        dir_x: f32,
        dir_y: f32,
        radius: f32,
    ) -> u32 {
        let (w, h, d) = (self.width, self.height, self.depth);
        if d < 2 || w < 6 || h < 6 {
            return 0;
        }
        let gx = sanitize_f32(gx).clamp(2.0, (w - 3) as f32);
        let gy = sanitize_f32(gy).clamp(2.0, (h - 3) as f32);
        let gz = sanitize_f32(gz).clamp(1.0, (d - 1) as f32);
        let rf = sanitize_f32(radius).clamp(0.9, 4.5);
        let r2 = rf * rf;

        let dx_raw = sanitize_f32(dir_x);
        let dy_raw = sanitize_f32(dir_y);
        let dmag = dx_raw.hypot(dy_raw);
        let (fwd_x, fwd_y) = if dmag > 1e-4 {
            (dx_raw / dmag, dy_raw / dmag)
        } else {
            (1.0, 0.0)
        };

        let r_ceil = rf.ceil() as i32;
        let cx = gx.floor() as i32;
        let cy = gy.floor() as i32;
        let cz = gz.floor() as i32;

        let x_min = (cx - r_ceil).max(2);
        let x_max = (cx + r_ceil).min((w - 3) as i32);
        let y_min = (cy - r_ceil).max(2);
        let y_max = (cy + r_ceil).min((h - 3) as i32);
        let z_min = (cz - r_ceil).max(1); // Preserve z == 0 bedrock floor!
        let z_max = (cz + r_ceil).min((d - 1) as i32);

        let mut src_voxels: Vec<(usize, usize)> = Vec::new();
        for z in z_min..=z_max {
            let dz = (z as f32 + 0.5) - gz;
            let zu = z as usize;
            for y in y_min..=y_max {
                let dy = (y as f32 + 0.5) - gy;
                let row = (y as usize) * w;
                for x in x_min..=x_max {
                    let dx = (x as f32 + 0.5) - gx;
                    if dx * dx + dy * dy + dz * dz <= r2 {
                        let idx = row + (x as usize);
                        if self.layers[zu].mass[SP_SOIL][idx] > 0
                            || self.layers[zu].mass[SP_ROCK][idx] > 0
                        {
                            src_voxels.push((zu, idx));
                        }
                    }
                }
            }
        }
        if src_voxels.is_empty() {
            return 0;
        }

        let shell_r = (rf + 2.5).ceil() as i32;
        let shell_r2_max = (rf + 2.6) * (rf + 2.6);
        let sx_min = (cx - shell_r).max(1);
        let sx_max = (cx + shell_r).min((w - 2) as i32);
        let sy_min = (cy - shell_r).max(1);
        let sy_max = (cy + shell_r).min((h - 2) as i32);
        let sz_min = (cz - shell_r).max(0);
        let sz_max = (cz + shell_r).min((d - 1) as i32);

        let mut dst_voxels: Vec<(f32, usize, usize)> = Vec::new();
        for z in sz_min..=sz_max {
            let dz = (z as f32 + 0.5) - gz;
            let zu = z as usize;
            for y in sy_min..=sy_max {
                let dy = (y as f32 + 0.5) - gy;
                let row = (y as usize) * w;
                for x in sx_min..=sx_max {
                    let dx = (x as f32 + 0.5) - gx;
                    let dist2 = dx * dx + dy * dy + dz * dz;
                    if dist2 <= r2 || dist2 > shell_r2_max {
                        continue;
                    }
                    let idx = row + (x as usize);
                    if self.layers[zu].cell_volume(idx) >= H_CAP {
                        continue;
                    }
                    let dist = dist2.sqrt().max(1e-3);
                    let forward_dot = (dx * fwd_x + dy * fwd_y) / dist;
                    let bedrock_penalty = if zu == 0 { 2.5 } else { 0.0 };
                    let ahead_penalty = if forward_dot > 0.35 { 3.5 } else { 0.0 };
                    let score = forward_dot * 2.0 + (dist - rf) * 0.6 + bedrock_penalty + ahead_penalty;
                    dst_voxels.push((score, zu, idx));
                }
            }
        }
        if dst_voxels.is_empty() {
            return 0;
        }
        dst_voxels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut total_displaced = 0u32;
        let mut dst_cursor = 0usize;
        let dst_len = dst_voxels.len();

        for (sz, sidx) in src_voxels {
            for &sp in &[SP_SOIL, SP_ROCK] {
                let avail = self.layers[sz].mass[sp][sidx];
                if avail == 0 {
                    continue;
                }
                let mut to_move = (((avail as u32) * 60 + 99) / 100).min(avail as u32) as u16;
                if to_move == 0 {
                    continue;
                }

                for pass in 0..2 {
                    if to_move == 0 {
                        break;
                    }
                    let per_cell_cap = if pass == 0 { 600u16 } else { H_CAP as u16 };
                    for _ in 0..dst_len {
                        if to_move == 0 {
                            break;
                        }
                        let (_, dz, didx) = dst_voxels[dst_cursor];
                        dst_cursor = (dst_cursor + 1) % dst_len;
                        if dz == sz && didx == sidx {
                            continue;
                        }
                        let room = H_CAP.saturating_sub(self.layers[dz].cell_volume(didx)) as u16;
                        if room == 0 {
                            continue;
                        }
                        let q = to_move.min(room).min(per_cell_cap);
                        if q == 0 {
                            continue;
                        }
                        let c_src = self.layers[sz].cell_heat_capacity(sidx);
                        let e_src = self.layers[sz].energy[sidx];
                        let e_xfer = if q == self.layers[sz].mass[sp][sidx]
                            && self.layers[sz].cell_volume(sidx) == q as u32
                        {
                            e_src
                        } else if c_src > 0 && e_src > 0 {
                            let prop = ((q as i128)
                                * (SPECIES[sp].c as i128)
                                * (e_src as i128)
                                / (c_src as i128)) as i64;
                            prop.clamp(0, e_src)
                        } else {
                            0
                        };

                        self.layers[sz].mass[sp][sidx] -= q;
                        self.layers[dz].mass[sp][didx] += q;
                        self.layers[sz].energy[sidx] -= e_xfer;
                        self.layers[dz].energy[didx] += e_xfer;

                        to_move -= q;
                        total_displaced = total_displaced.saturating_add(q as u32);
                    }
                }
            }
        }

        if total_displaced > 0 {
            self.structural.dirty = true;
        }
        total_displaced
    }

    /// Run one conservation-exact 3D structural support, realistic arch/cave collapse,
    /// and Buoyant Crystal Core stability step (`mass_error() == 0 && energy_error() == 0`).
    pub fn step_structural_mechanics(&mut self) -> structural::StructuralStepStats {
        structural::step_structural_mechanics(self)
    }

    /// Mark the 3D structural support graph dirty after terrain/thermal mutations.
    pub fn mark_structural_dirty(&mut self) {
        self.structural.dirty = true;
    }

    /// Apply a single layer step while snapshotting `prev_temp` / `prev_steam` first.
    pub fn step_layer(&mut self, layer: usize, raw: &RawStepOutput) -> StepStats {
        let l = layer.min(self.depth.saturating_sub(1));
        self.snapshot_layer_prev_sensors(l);
        self.layers[l].project_and_apply(raw)
    }

    /// Apply all `L` horizontal neural/refsim layer steps + exact 3D vertical exchange + dirty chunk diffing.
    pub fn step_3d(&mut self, raw_layers: &[RawStepOutput]) -> VolumeStepStats {
        self.step_3d_with_neural_z(raw_layers, None)
    }

    /// Apply gentle conservation-exact ambient climate relaxation (`world_laws.ambient_temp_shift`),
    /// updating `faucet_energy` so `mass_error == 0 && energy_error == 0` holds strictly on every tick.
    fn apply_world_law_ambient_relaxation_3d(&mut self) {
        let shift = self.world_laws.ambient_temp_shift;
        if !shift.is_finite() || shift.abs() <= 1.0 {
            return;
        }
        let t_target = (T_AMB + shift.round() as i64).clamp(T_MIN, T_MAX);
        let len = self.width * self.height;
        for f in &mut self.layers {
            for idx in 0..len {
                let c = f.cell_heat_capacity(idx);
                if c <= 0 {
                    continue;
                }
                let t = f.cell_temp_ck(idx);
                let step_ck = (t_target - t) / 8;
                if step_ck != 0 {
                    let e_cur = f.energy[idx];
                    let e_next = (e_cur + step_ck * c).clamp(c * T_MIN, c * T_MAX);
                    let delta = e_next - e_cur;
                    if delta != 0 {
                        f.energy[idx] = e_next;
                        f.faucet_energy += delta;
                    }
                }
            }
        }
    }

    /// 3D Altitude-Coupled Conservation-Exact Ambient Relaxation (`depth >= 48`):
    /// - Below `0.25 * depth` (geothermal abyss): gently warms cool subterranean rock/fluids toward `330 K`.
    /// - Above `0.72 * depth` (freezing stratosphere): gently chills high-altitude sky/cloud layers toward `266 K`.
    /// Strictly accounted in `faucet_energy` so `mass_error() == 0 && energy_error() == 0` ALWAYS holds!
    fn apply_altitude_ambient_relaxation_3d(&mut self) {
        if self.depth < 48 {
            return;
        }
        let d = self.depth;
        let z_abyss_hi = ((d as f32) * 0.25).floor() as usize;
        let z_strato_lo = ((d as f32) * 0.72).ceil() as usize;
        let len = self.width * self.height;
        let stride = 4usize;
        let phase = (self.tick as usize) % stride;

        // Warm geothermal abyss (`z < 0.25 * depth`)
        for z in 1..z_abyss_hi.min(d) {
            let f = &mut self.layers[z];
            let mut idx = phase;
            while idx < len {
                let c = f.cell_heat_capacity(idx);
                if c > 0 {
                    let t = f.cell_temp_ck(idx);
                    if t < 33_000 {
                        let step_ck = ((33_000 - t) / 16).max(60).min(33_000 - t);
                        let e_cur = f.energy[idx];
                        let e_next = (e_cur + step_ck * c).clamp(c * T_MIN, c * T_MAX);
                        let delta = e_next - e_cur;
                        if delta != 0 {
                            f.energy[idx] = e_next;
                            f.faucet_energy += delta;
                        }
                    }
                }
                idx += stride;
            }
        }

        // Crisp freezing stratosphere (`z >= 0.72 * depth`)
        for z in z_strato_lo..d {
            let f = &mut self.layers[z];
            let mut idx = phase;
            while idx < len {
                let c = f.cell_heat_capacity(idx);
                if c > 0 {
                    let t = f.cell_temp_ck(idx);
                    if t > 26_600 {
                        let step_ck = ((t - 26_600) / 16).max(60).min(t - 26_600);
                        let e_cur = f.energy[idx];
                        let e_next = (e_cur - step_ck * c).clamp(c * T_MIN, c * T_MAX);
                        let delta = e_next - e_cur;
                        if delta != 0 {
                            f.energy[idx] = e_next;
                            f.faucet_energy += delta;
                        }
                    }
                }
                idx += stride;
            }
        }
    }

    /// Apply all `L` horizontal layer steps + either neural `WorldModel3D` v4 vertical interface fluxes
    /// (`Some(z_flux_f32)` of layout `[depth - 1, 5, H, W]`) or analytic `step_vertical_exchange_3d` (`None`),
    /// then re-hash dirty chunks and report exact 3D mass & energy conservation errors.
    pub fn step_3d_with_neural_z(
        &mut self,
        raw_layers: &[RawStepOutput],
        z_flux_f32: Option<&[f32]>,
    ) -> VolumeStepStats {
        self.snapshot_prev_sensors();
        let laws = self.world_laws;
        let scale_raw = (laws.viscosity - 1.0).abs() > 1e-3
            || (laws.volatility - 1.0).abs() > 1e-3
            || (laws.thermal_cond - 1.0).abs() > 1e-3
            || (laws.precipitation - 1.0).abs() > 1e-3;
        let liq_mul = (1.0 / laws.viscosity.clamp(0.25, 4.0)).clamp(0.25, 4.0);
        let cond_mul = laws.thermal_cond.clamp(0.25, 4.0);
        let vol_mul = laws.volatility.clamp(0.2, 4.0);
        let precip_mul = laws.precipitation.clamp(0.1, 4.0);

        let mut proj_sum = 0.0f32;
        for (z, l) in self.layers.iter_mut().enumerate() {
            if let Some(raw) = raw_layers.get(z) {
                let st = if scale_raw {
                    let mut scaled = raw.clone();
                    let area = scaled.width * scaled.height;
                    for &c in &[0usize, 3, 4, 7] {
                        let base = c * area;
                        for v in &mut scaled.data[base..base + area] {
                            *v *= liq_mul;
                        }
                    }
                    for &c in &[8usize, 9] {
                        let base = c * area;
                        for v in &mut scaled.data[base..base + area] {
                            *v *= cond_mul;
                        }
                    }
                    {
                        let base = 13 * area;
                        for v in &mut scaled.data[base..base + area] {
                            *v *= precip_mul;
                        }
                    }
                    for &c in &[14usize, 15, 16] {
                        let base = c * area;
                        for v in &mut scaled.data[base..base + area] {
                            *v *= vol_mul;
                        }
                    }
                    l.project_and_apply(&scaled)
                } else {
                    l.project_and_apply(raw)
                };
                proj_sum += st.proj_active_pct;
            }
        }
        self.apply_world_law_ambient_relaxation_3d();
        self.apply_altitude_ambient_relaxation_3d();
        let (grav, buoy) = if let Some(zf) = z_flux_f32 {
            self.apply_neural_3d_vertical_fluxes(zf)
        } else {
            self.step_vertical_exchange_3d()
        };
        self.tick += 1;
        self.world_time_secs += 0.1;

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

    /// Sample continuous voxel-coupled physical state at `(x, y, layer)` for physics-native entities.
    pub fn sample_entity_physics(&self, x: f32, y: f32, layer: usize) -> EntityVoxelPhysics {
        if self.depth == 0 || self.width == 0 || self.height == 0 {
            return EntityVoxelPhysics::default();
        }
        let l = layer.min(self.depth - 1);
        let xi = (x.floor() as i32).clamp(0, (self.width - 1) as i32) as usize;
        let yi = (y.floor() as i32).clamp(0, (self.height - 1) as i32) as usize;
        let f = &self.layers[l];
        let idx = f.idx(xi, yi);
        let len = self.width * self.height;
        let vidx = l * len + idx;

        let t_ck = f.cell_temp_ck(idx) as i32;
        let prev_t = self.prev_temp.get(vidx).copied().unwrap_or(t_ck);
        let temp_c = (t_ck as f32 - 27_315.0) * 0.01;
        let delta_temp_spike = ((t_ck - prev_t).max(0) as f32) * 0.01;

        let steam_now = f.mass[SP_STEAM][idx] as i32;
        let prev_st = self.prev_steam.get(vidx).copied().unwrap_or(steam_now);
        let delta_steam_burst = ((steam_now - prev_st).max(0) as f32) / (C_NOM as f32);

        let ice_frac = (f.mass[SP_ICE][idx] as f32 / (C_NOM as f32)).clamp(0.0, 1.0);
        let oil_frac = (f.mass[SP_OIL][idx] as f32 / (C_NOM as f32)).clamp(0.0, 1.0);
        let water_frac = (f.mass[SP_WATER][idx] as f32 / (C_NOM as f32)).clamp(0.0, 1.0);
        let smoke_opacity = (((f.mass[SP_SMOKE][idx] as f32) + (f.mass[SP_STEAM][idx] as f32))
            / 600.0)
            .clamp(0.0, 1.0);

        let support_solid = f.mass[SP_SOIL][idx] >= 250
            || f.mass[SP_ROCK][idx] >= 250
            || f.mass[SP_WOOD][idx] >= 200
            || f.mass[SP_ICE][idx] >= 300;

        EntityVoxelPhysics {
            temp_c,
            delta_temp_spike,
            delta_steam_burst,
            ice_frac,
            oil_frac,
            water_frac,
            smoke_opacity,
            support_solid,
        }
    }

    /// Conservation-guarded continuous physical actuator: injects bounded thermal energy (`±ΔE`)
    /// and/or matter (`water`, `oil`, `smoke`) into `(x, y, layer)` while keeping exact integer
    /// accounting (`mass_error == 0 && energy_error == 0`). Returns `(delta_mass, delta_energy)`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_physical_actuator(
        &mut self,
        x: f32,
        y: f32,
        layer: usize,
        thermal_flux: f32,
        sp_water: f32,
        sp_oil: f32,
        sp_smoke: f32,
        focus_radius: f32,
    ) -> (i64, i64) {
        if self.depth == 0 || self.width < 4 || self.height < 4 {
            return (0, 0);
        }
        let l = layer.min(self.depth - 1);
        let tf = sanitize_f32(thermal_flux).clamp(-1.0, 1.0);
        let mw = sanitize_f32(sp_water).clamp(0.0, 1.0);
        let mo = sanitize_f32(sp_oil).clamp(0.0, 1.0);
        let ms = sanitize_f32(sp_smoke).clamp(0.0, 1.0);
        let rf = sanitize_f32(focus_radius).clamp(1.0, 10.0);
        let r_i = rf.ceil() as i32;
        let r2 = (rf * rf).max(1.0);
        let focus_gain = (4.0 / rf).clamp(0.65, 1.6);

        let cx = sanitize_f32(x).round() as i32;
        let cy = sanitize_f32(y).round() as i32;

        let f = &mut self.layers[l];
        let w_i = f.width as i32;
        let h_i = f.height as i32;
        let mut total_dm = 0i64;
        let mut total_de = 0i64;

        let matter_specs = [
            (SP_WATER, mw, 420.0f32),
            (SP_OIL, mo, 380.0f32),
            (SP_SMOKE, ms, 320.0f32),
        ];

        for dy in -r_i..=r_i {
            for dx in -r_i..=r_i {
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > r2 {
                    continue;
                }
                let gx = cx + dx;
                let gy = cy + dy;
                if gx < 2 || gy < 2 || gx + 2 >= w_i || gy + 2 >= h_i {
                    continue;
                }
                let falloff = (1.0 - (d2 / r2).sqrt() * 0.65).clamp(0.15, 1.0);
                let idx = f.idx(gx as usize, gy as usize);

                // 1. Continuous matter injection (bounded by H_CAP, skips solid rock walls)
                if f.mass[SP_ROCK][idx] < 500 {
                    for &(sp, amt, max_units) in &matter_specs {
                        if amt > 0.01 {
                            let room = H_CAP.saturating_sub(f.cell_volume(idx));
                            let add = ((max_units * amt * falloff) as u32).min(room) as u16;
                            if add > 0 {
                                f.mass[sp][idx] += add;
                                let e_th = (add as i64) * SPECIES[sp].c * T_AMB;
                                let e_chem = (add as i64) * SPECIES[sp].u0;
                                f.energy[idx] += e_th;
                                f.faucet_mass[sp] += add as i64;
                                f.faucet_energy += e_th + e_chem;
                                f.forcing_mass[idx] = f.forcing_mass[idx].saturating_add(add as i32);
                                total_dm += add as i64;
                                total_de += e_th + e_chem;
                            }
                        }
                    }
                }

                // 2. Continuous bounded thermal flux (+ΔE heat / -ΔE cryo)
                let c = f.cell_heat_capacity(idx);
                if c > 0 {
                    if tf > 0.01 {
                        let add_e = ((36_000.0 * tf * falloff * focus_gain) as i64) * c;
                        let cap_e = (c * T_MAX - f.energy[idx]).max(0);
                        let delta = add_e.min(cap_e);
                        if delta > 0 {
                            f.energy[idx] += delta;
                            f.faucet_energy += delta;
                            f.forcing_energy[idx] = f.forcing_energy[idx].saturating_add(delta);
                            total_de += delta;
                        }
                    } else if tf < -0.01 {
                        let sub_e = ((12_000.0 * (-tf) * falloff * focus_gain) as i64) * c;
                        let min_e = c * T_MIN;
                        let delta = -(sub_e.min((f.energy[idx] - min_e).max(0)));
                        if delta < 0 {
                            f.energy[idx] += delta;
                            f.faucet_energy += delta;
                            f.forcing_energy[idx] = f.forcing_energy[idx].saturating_add(delta);
                            total_de += delta;
                        }
                    }
                }
            }
        }

        if total_de.abs() > 250_000 {
            self.structural.dirty = true;
        }
        (total_dm, total_de)
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

    // =========================================================================
    // TRUE ISOTROPIC 3D Z=16 WORLD, FM3D BRICK ENCODER & CAPSULE LOCOMOTION
    // =========================================================================

    /// Initialize a true isotropic 3D `W × H × D` (`1m × 1m × 1m`) **Shattered Reach** world
    /// (delegates to `init_biome_3d(BIOME_NORDIC_VIADUCT, 1)` so all existing Z=16 tests stay 100% green).
    pub fn init_shattered_reach_z16(&mut self) {
        self.init_biome_3d(BIOME_NORDIC_VIADUCT, 1);
    }

    /// Initialize any of the **5 distinct true-3D biomes** (`0..=4`) at any vertical depth `16..=32`
    /// with deterministic seeded diversity:
    /// - `0` (`BIOME_NORDIC_VIADUCT`): 3-span arched stone/timber bridge, vaulted basalt cave (+ upper gallery at Z>=24), alpine snow peaks, plunging waterfall, Nordic pines
    /// - `1` (`BIOME_VOLCANIC_CALDERA`): Steep obsidian caldera rim, glowing superheated basalt magma channels (`950 K`), natural arched lava-tube bridges, ash-draped terraces, billowing steam/smoke vents
    /// - `2` (`BIOME_SINKHOLE_CAVERNS`): Dramatic karst sinkhole opening into a massive multi-pillar vaulted underground cavern (2-level stacked caverns at Z>=24!) with subterranean azure pools, hanging basalt arches, and vertical light shafts
    /// - `3` (`BIOME_GLACIER_GORGE`): Towering translucent glacial ice spires, natural carved ice bridges over a deep canyon, sub-zero snowdrifts, and frozen waterfalls
    /// - `4` (`BIOME_GEOTHERMAL_BASIN`): Stepped travertine mineral terraces, boiling hot springs (`380–450 K`), dense buoyant rising steam clouds feeding the 3D hydrologic cycle, and lush warm-valley flora
    pub fn init_biome_3d(&mut self, biome_id: u8, seed: u64) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        if w == 0 || h == 0 || d == 0 {
            return;
        }
        let b_id = biome_id % NUM_BIOMES;
        self.biome_id = b_id;
        self.seed = seed;
        self.layers = (0..d).map(|_| Field::new_empty(w, h)).collect();
        self.tick = 0;
        self.evolution_stats = WorldEvolutionStats::default();

        let (seed_ox, seed_oy, seed_phase) = if seed == 1 {
            (0.0f32, 0.0f32, 0.0f32)
        } else {
            let h1 = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let h2 = (h1 ^ (h1 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            let ox = ((h1 & 0xFFFF) as f32 / 65535.0) * 31.0 + 2.7;
            let oy = ((h2 & 0xFFFF) as f32 / 65535.0) * 31.0 - 3.9;
            let ph = (((h1 >> 16) & 0xFFFF) as f32 / 65535.0) * std::f32::consts::TAU;
            (ox, oy, ph)
        };

        match b_id {
            BIOME_NORDIC_VIADUCT => self.init_biome_nordic_viaduct(seed_ox, seed_oy, seed_phase),
            BIOME_VOLCANIC_CALDERA => self.init_biome_volcanic_caldera(seed_ox, seed_oy, seed_phase),
            BIOME_SINKHOLE_CAVERNS => self.init_biome_sinkhole_caverns(seed_ox, seed_oy, seed_phase),
            BIOME_GLACIER_GORGE => self.init_biome_glacier_gorge(seed_ox, seed_oy, seed_phase),
            BIOME_SKY_ABYSS => self.init_sky_abyss_vertical(seed_ox, seed_oy, seed_phase),
            _ => self.init_biome_geothermal_basin(seed_ox, seed_oy, seed_phase),
        }

        self.snapshot_conservation_baseline();
        self.snapshot_prev_sensors();
        if b_id == BIOME_SKY_ABYSS && d >= 16 {
            structural::init_sky_abyss_buoyant_cores(self);
        } else {
            self.structural = structural::StructuralState::default();
        }
    }

    fn init_biome_nordic_viaduct(&mut self, seed_ox: f32, seed_oy: f32, seed_phase: f32) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let z_scale = (d as f32 / 16.0).clamp(1.0, 2.0);
        let river_top_z = (2.0 * z_scale).round() as usize;
        let basin_z = ((10.0 * z_scale).round() as usize).min(d.saturating_sub(2));
        let fire_z = ((7.0 * z_scale).round() as usize).min(d.saturating_sub(3));
        let perim_z = ((9.0 * z_scale).round() as usize).min(d.saturating_sub(2));
        let cave_stream_z = (2.0 * z_scale).round() as usize;
        let magma_z = (1.0 * z_scale).round().max(1.0) as usize;
        let has_stacked_gallery = d >= 24;

        // Bridge geometry parameters (proportional in X/Y, scaled smoothly across Z=16..32!)
        let bridge_cy = 0.50 * hf;
        let bridge_hw_y = (hf * 0.032).clamp(3.8, 5.6);
        let bridge_x0 = 0.29 * wf;
        let bridge_x1 = 0.71 * wf;
        let bridge_cx = 0.50 * wf;
        let bridge_half_span = 0.21 * wf;

        // Vaulted basalt cave tunnel spline in SW mountain (world X, world Y, norm_floor_z, half_w_m, norm_vault_h)
        let cave_hw = (wf * 0.042).max(4.6);
        let cave_pts: [(f32, f32, f32, f32, f32); 4] = [
            (0.375 * wf, 0.700 * hf, 1.55, cave_hw * 1.05, 4.30),
            (0.305 * wf, 0.710 * hf, 1.55, cave_hw, 4.25),
            (0.235 * wf, 0.720 * hf, 1.55, cave_hw * 1.15, 4.40),
            (0.155 * wf, 0.745 * hf, 1.55, cave_hw * 0.92, 4.00),
        ];
        // Cave basalt pillars offset cleanly off the central walkway so capsule traversal stays clear
        let cave_pillars: [(f32, f32, f32); 2] = [
            (0.275 * wf, 0.715 * hf + 3.2, 0.85),
            (0.220 * wf, 0.722 * hf - 3.2, 0.90),
        ];

        // Deterministic biomass-scaled conifer placements (nx, ny, height_m_norm, canopy_r_m)
        let conifers: &[(f32, f32, f32, f32)] = &[
            (0.76, 0.24, 5.8, 2.3),
            (0.82, 0.30, 6.0, 2.4),
            (0.70, 0.20, 5.4, 2.2),
            (0.78, 0.64, 5.8, 2.4),
            (0.72, 0.72, 5.6, 2.3),
            (0.84, 0.70, 5.5, 2.2),
            (0.22, 0.24, 5.6, 2.3),
            (0.16, 0.32, 5.8, 2.4),
            (0.28, 0.20, 5.4, 2.2),
            (0.20, 0.64, 4.2, 2.0),
            (0.26, 0.80, 4.4, 2.0),
            (0.14, 0.84, 5.2, 2.2),
            (0.36, 0.14, 4.2, 2.0),
            (0.64, 0.14, 4.2, 2.0),
        ];

        // 1. Precompute 2D sculpted alpine & canyon base heightfield (in normalized Z units 0..14)
        let mut base_h = vec![0.0f32; w * h];
        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            let river_cx_n = 0.50 + 0.032 * ((ny - 0.50) * std::f32::consts::TAU * 1.1 + seed_phase * 0.25).sin();
            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;

                let n_big = z16_fbm2(nx * 5.2 + seed_ox, ny * 5.2 + seed_oy);
                let n_med = z16_fbm2(nx * 14.0 + 11.3 + seed_ox, ny * 14.0 - 7.1 + seed_oy);
                let dist_r = (nx - river_cx_n).abs();

                let canyon_t = z16_smoothstep(0.042, 0.145, dist_r);
                let mut gh = 0.95 + canyon_t * 5.35 + n_big * 1.15 + n_med * 0.35;

                // Dry West under-bridge riverside path
                let under_w_x = 1.0 - z16_smoothstep(0.020, 0.046, (nx - 0.388).abs());
                let under_w_y = z16_smoothstep(0.32, 0.37, ny) * (1.0 - z16_smoothstep(0.63, 0.68, ny));
                let under_w = under_w_x * under_w_y;
                gh = gh * (1.0 - under_w) + 2.85 * under_w;

                // Dry East under-bridge riverside terrace
                let under_e_x = 1.0 - z16_smoothstep(0.018, 0.044, (nx - 0.612).abs());
                let under_e_y = z16_smoothstep(0.36, 0.41, ny) * (1.0 - z16_smoothstep(0.59, 0.64, ny));
                let under_e = under_e_x * under_e_y;
                gh = gh * (1.0 - under_e) + 2.95 * under_e;

                // Smooth bridge approach ramps & abutments
                let y_bridge_env = 1.0 - z16_smoothstep(0.030, 0.085, (ny - 0.50).abs());
                let w_abut = (1.0 - z16_smoothstep(0.025, 0.085, (nx - 0.285).abs())) * y_bridge_env;
                let e_abut = (1.0 - z16_smoothstep(0.025, 0.085, (nx - 0.715).abs())) * y_bridge_env;
                gh = gh * (1.0 - w_abut) + 8.05 * w_abut;
                gh = gh * (1.0 - e_abut) + 8.05 * e_abut;

                // North Alpine Mountain Massif
                let mtn_n_y = z16_smoothstep(0.02, 0.09, ny) * (1.0 - z16_smoothstep(0.21, 0.30, ny));
                let mtn_n_x = 1.0 - z16_smoothstep(0.14, 0.42, (nx - 0.50).abs());
                let peak_nw = (1.0 - z16_smoothstep(0.0, 0.16, (nx - 0.36).hypot(ny - 0.15))) * 5.6;
                let peak_ne = (1.0 - z16_smoothstep(0.0, 0.16, (nx - 0.64).hypot(ny - 0.15))) * 5.8;
                gh += mtn_n_x * mtn_n_y * 4.8 + peak_nw + peak_ne;

                // Upper Highland Glacial Basin
                let basin_r = (((nx - 0.50) / 0.095).powi(2) + ((ny - 0.145) / 0.060).powi(2)).sqrt();
                let basin_w = 1.0 - z16_smoothstep(0.48, 1.05, basin_r);
                gh = gh * (1.0 - basin_w) + 9.25 * basin_w;

                // Waterfall spillway notch & plunge pool
                let notch_x = 1.0 - z16_smoothstep(0.020, 0.040, (nx - 0.50).abs());
                let notch_y = z16_smoothstep(0.185, 0.205, ny) * (1.0 - z16_smoothstep(0.218, 0.236, ny));
                let notch_w = notch_x * notch_y;
                gh = gh * (1.0 - notch_w) + 9.65 * notch_w;

                let pool_r = (((nx - 0.50) / 0.065).powi(2) + ((ny - 0.265) / 0.042).powi(2)).sqrt();
                let pool_w = 1.0 - z16_smoothstep(0.42, 1.05, pool_r);
                gh = gh * (1.0 - pool_w) + 0.90 * pool_w;

                // SW Basalt Mountain Massif housing the vaulted cave tunnel
                let cave_mtn_r = (((nx - 0.235) / 0.135).powi(2) + ((ny - 0.725) / 0.135).powi(2)).sqrt();
                let cave_mtn_w = 1.0 - z16_smoothstep(0.45, 1.08, cave_mtn_r);
                let cave_mtn_h = 11.4 + n_med * 0.75 + (1.0 - cave_mtn_r.min(1.0)) * 1.3;
                gh = gh.max(gh * (1.0 - cave_mtn_w) + cave_mtn_h * cave_mtn_w);

                // Smooth gravel apron right outside East cave mouth
                let apron_r = (((nx - 0.385) / 0.048).powi(2) + ((ny - 0.695) / 0.052).powi(2)).sqrt();
                let apron_w = 1.0 - z16_smoothstep(0.50, 1.05, apron_r);
                gh = gh * (1.0 - apron_w) + 1.60 * apron_w;

                // East Wildfire Bluff terrace
                let fire_r = (((nx - 0.74) / 0.065).powi(2) + ((ny - 0.38) / 0.065).powi(2)).sqrt();
                let fire_w = 1.0 - z16_smoothstep(0.45, 1.05, fire_r);
                gh = gh * (1.0 - fire_w) + 6.60 * fire_w;

                // West & East team spawn clearings gently leveled
                let sp_w_r = ((nx - 0.14).powi(2) + (ny - 0.50).powi(2)).sqrt();
                let sp_w = 1.0 - z16_smoothstep(0.04, 0.09, sp_w_r);
                gh = gh * (1.0 - sp_w) + 5.80 * sp_w;

                let sp_e_r = ((nx - 0.86).powi(2) + (ny - 0.50).powi(2)).sqrt();
                let sp_e = 1.0 - z16_smoothstep(0.04, 0.09, sp_e_r);
                gh = gh * (1.0 - sp_e) + 5.80 * sp_e;

                base_h[y * w + x] = gh.clamp(0.85, 13.2);
            }
        }

        // 2. Populate all Z=0..depth layers
        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            let river_cx_n = 0.50 + 0.032 * ((ny - 0.50) * std::f32::consts::TAU * 1.1 + seed_phase * 0.25).sin();
            let river_slope = ((1.0 - ny) * 5.0) as u8;

            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;
                let idx = y * w + x;
                let gh = base_h[idx];
                let dist_r = (nx - river_cx_n).abs();
                let is_perimeter = x < 2 || x + 2 >= w || y < 2 || y + 2 >= h;

                for z in 0..d {
                    let wz_raw = z as f32 + 0.5;
                    let wz = wz_raw / z_scale; // normalized 0..16 height
                    let f = &mut self.layers[z];
                    f.elevation[idx] = ((z * 26 / d.max(1)).min(26) as u8)
                        .saturating_add(if z <= river_top_z + 1 { river_slope } else { 0 });

                    if z == 0 {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, T_AMB);
                        continue;
                    }
                    if is_perimeter && z <= perim_z {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, T_AMB);
                        continue;
                    }

                    let mut sdf = wz - gh;
                    let mut mat_kind = 0u8;

                    // --- True 3-Span Arched Stone & Timber Bridge ---
                    let dy_b = (wy - bridge_cy).abs();
                    if wx >= bridge_x0 - 1.5
                        && wx <= bridge_x1 + 1.5
                        && dy_b <= bridge_hw_y + 1.5
                        && wz <= 10.6
                    {
                        let u = ((wx - bridge_cx) / bridge_half_span).clamp(-1.0, 1.0);
                        let deck_top = 8.80 - 0.72 * u * u;
                        let prism_x = (wx - bridge_cx).abs() - bridge_half_span;
                        let prism_y = dy_b - bridge_hw_y;
                        let prism_z = wz - deck_top;
                        let mut bridge_sdf = prism_x.max(prism_y).max(prism_z);

                        let arches: [(f32, f32, f32, f32); 3] = [
                            (0.388 * wf, 0.045 * wf, 3.0, 3.80),
                            (0.500 * wf, 0.047 * wf, 2.8, 4.25),
                            (0.612 * wf, 0.045 * wf, 3.0, 3.80),
                        ];
                        for &(ax, rx, spring_z, rz) in &arches {
                            let adx = (wx - ax) / rx.max(1.0);
                            let adz = (wz - spring_z).max(0.0) / rz;
                            let arch_r = (adx * adx + adz * adz).sqrt();
                            let open_sdf = (arch_r - 1.0) * rx.min(rz);
                            let rect_open = (wx - ax).abs() - rx;
                            let hole_sdf = if wz < spring_z { rect_open } else { open_sdf };
                            bridge_sdf = bridge_sdf.max(-hole_sdf);
                        }

                        for &pier_nx in &[0.443f32, 0.557f32] {
                            let px = pier_nx * wf;
                            let butt_x = (wx - px).abs() - (wf * 0.011).max(1.25);
                            let butt_y = dy_b - (bridge_hw_y + 0.8);
                            let butt_z = wz - (deck_top - 0.6);
                            bridge_sdf = bridge_sdf.min(butt_x.max(butt_y).max(butt_z));
                        }

                        let para_y = (dy_b - (bridge_hw_y - 0.35)).abs() - 0.38;
                        let para_x = (wx - bridge_cx).abs() - (bridge_half_span - 0.6);
                        let para_z = (wz - (deck_top + 0.65)).max((deck_top - 0.3) - wz);
                        bridge_sdf = bridge_sdf.min(para_x.max(para_y).max(para_z));

                        if bridge_sdf < sdf {
                            sdf = bridge_sdf;
                            if dy_b < bridge_hw_y - 1.0 && wz > deck_top - 1.1 && wz <= deck_top + 0.25 {
                                mat_kind = 2;
                            } else {
                                mat_kind = 1;
                            }
                        }
                    }

                    // --- Vaulted Basalt Cave Tunnel (+ 2nd-level stacked upper gallery when depth >= 24!) ---
                    let mut inside_cave_air = false;
                    let mut cave_stream_zone = false;
                    let mut magma_vent_zone = false;
                    if nx >= 0.10 && nx <= 0.43 && ny >= 0.60 && ny <= 0.86 && wz <= 10.2 {
                        let mut best_tunnel_sdf = 999.0f32;
                        let mut best_lat_y = 0.0f32;
                        for seg in 0..(cave_pts.len() - 1) {
                            let (ax, ay, afz, ahw, avh) = cave_pts[seg];
                            let (bx, by, bfz, bhw, bvh) = cave_pts[seg + 1];
                            let abx = bx - ax;
                            let aby = by - ay;
                            let len2 = (abx * abx + aby * aby).max(1e-6);
                            let t = (((wx - ax) * abx + (wy - ay) * aby) / len2).clamp(0.0, 1.0);
                            let cx = ax + abx * t;
                            let cy = ay + aby * t;
                            let fz = afz + (bfz - afz) * t;
                            let hw = ahw + (bhw - ahw) * t;
                            let vh = avh + (bvh - avh) * t;

                            let horiz_d = (wx - cx).hypot(wy - cy);
                            let z_rel = wz - fz;
                            let norm_x = horiz_d / hw;
                            let norm_z = if z_rel < 1.1 {
                                ((1.1 - z_rel) / 1.1).max(0.0)
                            } else {
                                (z_rel - 1.1) / (vh - 1.1)
                            };
                            let vault_r = (norm_x * norm_x + norm_z * norm_z).sqrt();
                            let floor_cut = -z_rel;
                            let tunnel_d = ((vault_r - 1.0) * hw.min(vh)).max(floor_cut);
                            if tunnel_d < best_tunnel_sdf {
                                best_tunnel_sdf = tunnel_d;
                                best_lat_y = wy - cy;
                            }
                        }

                        // At Z>=24 (e.g. 32m deep), carve a 2nd-level stacked upper basalt gallery right above the lower cave!
                        if has_stacked_gallery && wz >= 6.8 && wz <= 9.6 {
                            let g_r = (((nx - 0.24) / 0.095).powi(2) + ((ny - 0.725) / 0.070).powi(2)).sqrt();
                            let g_z = ((wz - 8.2) / 1.25).abs();
                            let upper_d = ((g_r * g_r + g_z * g_z).sqrt() - 1.0) * 1.4;
                            if upper_d < best_tunnel_sdf {
                                best_tunnel_sdf = upper_d;
                            }
                        }

                        if best_tunnel_sdf < 0.0 {
                            inside_cave_air = true;
                            sdf = sdf.max(-best_tunnel_sdf);
                            if z == cave_stream_z && best_lat_y < -2.3 && best_lat_y > -4.2 && nx < 0.36 && nx > 0.18 {
                                cave_stream_zone = true;
                            }
                        }
                        if best_tunnel_sdf < 1.8 && sdf < 0.8 && mat_kind == 0 {
                            mat_kind = 3;
                        }

                        for &(px, py, prad) in &cave_pillars {
                            if wz >= 1.2 && wz <= 6.2 {
                                let d_pil = (wx - px).hypot(wy - py) - prad;
                                if d_pil < sdf {
                                    sdf = d_pil;
                                    mat_kind = 3;
                                    inside_cave_air = false;
                                }
                            }
                        }

                        if z == magma_z && !inside_cave_air {
                            let d_vent1 = (nx - 0.245).hypot(ny - 0.742);
                            let d_vent2 = (nx - 0.205).hypot(ny - 0.748);
                            if d_vent1 < 0.018 || d_vent2 < 0.016 {
                                magma_vent_zone = true;
                                mat_kind = 6;
                            }
                        }
                    }

                    // --- Biomass-Scaled Conifers ---
                    for &(cnx, cny, tree_h, max_r) in conifers {
                        let tx = cnx * wf;
                        let ty = cny * hf;
                        let d_xy = (wx - tx).hypot(wy - ty);
                        if d_xy < max_r + 1.5 {
                            let txi = (tx as usize).min(w - 1);
                            let tyi = (ty as usize).min(h - 1);
                            let t_base = base_h[tyi * w + txi] - 0.3;
                            let top_z = (t_base + tree_h).min(14.8);
                            let rel_z = wz - t_base;
                            if rel_z >= -0.4 && wz <= top_z + 0.5 {
                                let trunk_r = 0.72;
                                let trunk_sdf = (d_xy - trunk_r).max(wz - (top_z - 0.8)).max(-0.4 - rel_z);
                                if trunk_sdf < sdf {
                                    sdf = trunk_sdf;
                                    mat_kind = 4;
                                }
                                if rel_z >= 1.4 && wz <= top_z + 0.3 {
                                    let ft = ((wz - (t_base + 1.4)) / (top_z - (t_base + 1.4)).max(0.5)).clamp(0.0, 1.0);
                                    let canopy_r = (max_r * (1.0 - ft * 0.75)).max(0.65);
                                    let foli_sdf = (d_xy - canopy_r).max(wz - top_z).max((t_base + 1.4) - wz);
                                    if foli_sdf < sdf {
                                        sdf = foli_sdf;
                                        mat_kind = 5;
                                    }
                                }
                            }
                        }
                    }

                    // Glacial ice rim around the highland basin
                    if z >= basin_z && z <= (basin_z + 1).min(d - 1) && mat_kind == 0 && sdf <= 0.4 {
                        let br = (((nx - 0.50) / 0.115).powi(2) + ((ny - 0.145) / 0.080).powi(2)).sqrt();
                        if br > 0.62 && br < 1.18 && ny < 0.205 {
                            mat_kind = 7;
                        }
                    }

                    let band = (if mat_kind == 1 || mat_kind == 2 { 0.48 } else { 0.60 }) / z_scale;
                    let solid_d = z16_sdf_to_density(sdf, band);
                    let solid_m = (solid_d * (C_NOM as f32)).round() as u16;
                    let mut cell_temp = if ny < 0.24 && wz >= 9.0 { 27_100 } else { T_AMB };

                    if solid_m >= 15 {
                        match mat_kind {
                            1 | 3 => f.mass[SP_ROCK][idx] = solid_m,
                            2 | 4 => f.mass[SP_WOOD][idx] = solid_m,
                            5 => f.mass[SP_PLANT][idx] = ((solid_m as u32 * 4) / 3).min(1100) as u16,
                            6 => {
                                f.mass[SP_ROCK][idx] = solid_m.max(950);
                                cell_temp = 105_000;
                            }
                            7 => {
                                f.mass[SP_ICE][idx] = solid_m;
                                cell_temp = 26_600;
                            }
                            _ => {
                                if wz < gh - 1.3 || wz > 8.6 || wz < 2.2 {
                                    f.mass[SP_ROCK][idx] = solid_m;
                                } else {
                                    f.mass[SP_SOIL][idx] = solid_m;
                                }
                            }
                        }
                    }

                    // --- Fluids: Canyon River, Cave Stream, Glacial Basin & 3D Waterfall ---
                    if solid_d < 0.45 {
                        if z <= river_top_z && dist_r < 0.058 && !inside_cave_air {
                            f.mass[SP_WATER][idx] = 1100;
                            f.flow_s[idx] = 130;
                            cell_temp = 28_400;
                        } else if cave_stream_zone {
                            f.mass[SP_WATER][idx] = 680;
                            f.flow_e[idx] = 85;
                            cell_temp = 28_500;
                        } else if z == basin_z {
                            let br = (((nx - 0.50) / 0.088).powi(2) + ((ny - 0.145) / 0.056).powi(2)).sqrt();
                            let in_spill = (nx - 0.50).abs() < 0.028 && ny >= 0.18 && ny <= 0.228;
                            if br < 0.95 || in_spill {
                                f.mass[SP_WATER][idx] = 1080;
                                f.flow_s[idx] = 120;
                                cell_temp = 28_000;
                            }
                        } else if z >= river_top_z && z < basin_z {
                            let drop_t = ((basin_z as f32 - z as f32) / (basin_z - river_top_z).max(1) as f32).clamp(0.0, 1.0);
                            let wf_ny = 0.220 + 0.040 * drop_t.sqrt();
                            let wf_hw_x = 0.024 + 0.008 * drop_t;
                            if (nx - 0.50).abs() < wf_hw_x && (ny - wf_ny).abs() < 0.015 {
                                f.mass[SP_WATER][idx] = 860;
                                f.flow_s[idx] = 90;
                                cell_temp = 28_100;
                            }
                        }
                    }

                    // --- Plunge Pool Mist ---
                    let mist_lo = river_top_z + 1;
                    let mist_hi = (river_top_z + (2.0 * z_scale).round() as usize).min(d - 2);
                    if (mist_lo..=mist_hi).contains(&z) && solid_d < 0.30 {
                        let mr = (((nx - 0.50) / 0.048).powi(2) + ((ny - 0.265) / 0.036).powi(2)).sqrt();
                        if mr < 1.0 {
                            f.mass[SP_STEAM][idx] = (240.0 * (1.0 - mr)) as u16;
                            cell_temp = 37_600;
                        }
                    }

                    // --- Active Wildfire on East Pine Bluff ---
                    let fire_d = (nx - 0.74).hypot(ny - 0.38);
                    let plume_hi = (fire_z + (5.0 * z_scale).round() as usize).min(d - 2);
                    if fire_d < 0.036 && z == fire_z && solid_d < 0.50 {
                        let fall = (1.0 - fire_d / 0.036).clamp(0.2, 1.0);
                        f.mass[SP_WOOD][idx] = f.mass[SP_WOOD][idx].max((720.0 * fall) as u16);
                        f.mass[SP_OIL][idx] = (380.0 * fall) as u16;
                        f.mass[SP_PLANT][idx] = (320.0 * fall) as u16;
                        cell_temp = 78_000;
                    } else if z > fire_z && z <= plume_hi && solid_d < 0.35 {
                        let dz_n = (z - fire_z) as f32 / z_scale;
                        let drift_x = 0.74 + dz_n * 0.004;
                        let drift_y = 0.38 + dz_n * 0.002;
                        let plume_r = 0.025 + dz_n * 0.005;
                        let pd = (nx - drift_x).hypot(ny - drift_y) / plume_r;
                        if pd < 1.0 {
                            f.mass[SP_SMOKE][idx] = (360.0 * (1.0 - pd)) as u16;
                            cell_temp = (54_000 - ((dz_n - 1.0).max(0.0) * 3_500.0) as i64).clamp(T_MIN, T_MAX);
                        }
                    }

                    if magma_vent_zone {
                        cell_temp = 105_000;
                    }

                    f.set_cell_temp(idx, cell_temp);
                }
            }
        }
    }

    /// Biome 1 (`Volcanic Caldera`): Steep obsidian caldera rim, glowing superheated basalt magma channels (`950 K`),
    /// natural arched lava-tube bridges, ash-draped terraces, billowing steam/smoke vents (+ stacked lava-tube gallery at Z>=24).
    fn init_biome_volcanic_caldera(&mut self, seed_ox: f32, seed_oy: f32, seed_phase: f32) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let z_scale = (d as f32 / 16.0).clamp(1.0, 2.0);
        let perim_z = ((10.0 * z_scale).round() as usize).min(d.saturating_sub(2));
        let magma_top_z = (2.0 * z_scale).round() as usize;
        let has_stacked_tube = d >= 24;

        let mut base_h = vec![0.0f32; w * h];
        for y in 0..h {
            let ny = (y as f32 + 0.5) / hf;
            for x in 0..w {
                let nx = (x as f32 + 0.5) / wf;
                let dx = nx - 0.50;
                let dy = ny - 0.50;
                let r = dx.hypot(dy);
                let ang = dy.atan2(dx) + seed_phase;
                let n_big = z16_fbm2(nx * 5.8 + seed_ox + 4.1, ny * 5.8 + seed_oy - 2.3);
                let n_med = z16_fbm2(nx * 15.0 + seed_ox, ny * 15.0 + seed_oy);

                // Steep jagged obsidian caldera rim around r=0.33..0.46, sunken volcanic floor inside
                let rim_r = 0.37 + 0.03 * (ang * 3.0).sin() + 0.015 * (ang * 5.0).cos();
                let rim_band = (1.0 - ((r - rim_r).abs() / 0.12).clamp(0.0, 1.0)).powf(1.5);
                let crater_floor = 2.0 + z16_smoothstep(0.18, 0.34, r) * 4.4;
                // Stepped ash terraces inside the caldera
                let terraced = (crater_floor * 0.75).floor() / 0.75 * 0.45 + crater_floor * 0.55;
                let mut gh = terraced + rim_band * (7.2 + n_big * 1.4) + n_med * 0.35;

                // Central obsidian volcanic island cone (safe fighting spire)
                let spire_w = 1.0 - z16_smoothstep(0.06, 0.13, r);
                gh = gh * (1.0 - spire_w) + (5.2 + (1.0 - (r / 0.13).min(1.0)) * 2.0) * spire_w;

                // Carve glowing superheated magma ring channel (r ∈ [0.145..0.22]) & central N-S rift
                let ring_d = (r - 0.182).abs();
                let magma_ring = 1.0 - z16_smoothstep(0.018, 0.044, ring_d);
                let rift_d = (nx - (0.50 + 0.025 * (ny * std::f32::consts::TAU + seed_phase).sin())).abs();
                let magma_rift = (1.0 - z16_smoothstep(0.016, 0.040, rift_d)) * z16_smoothstep(0.12, 0.22, r);
                let magma_cut = magma_ring.max(magma_rift * 0.85);
                gh = gh * (1.0 - magma_cut) + 1.15 * magma_cut;

                // Safe walkable West & East spawn terraces
                let sp_w = 1.0 - z16_smoothstep(0.045, 0.095, (nx - 0.22).hypot(ny - 0.50));
                let sp_e = 1.0 - z16_smoothstep(0.045, 0.095, (nx - 0.80).hypot(ny - 0.50));
                gh = gh * (1.0 - sp_w) + 6.4 * sp_w;
                gh = gh * (1.0 - sp_e) + 6.4 * sp_e;

                base_h[y * w + x] = gh.clamp(1.0, 13.5);
            }
        }

        let bridge_hw_y = (hf * 0.030).clamp(3.2, 5.0);
        let bridge_hw_x = (wf * 0.028).clamp(3.0, 4.6);

        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;
                let idx = y * w + x;
                let gh = base_h[idx];
                let r = (nx - 0.50).hypot(ny - 0.50);
                let is_perimeter = x < 2 || x + 2 >= w || y < 2 || y + 2 >= h;

                for z in 0..d {
                    let wz = (z as f32 + 0.5) / z_scale;
                    let f = &mut self.layers[z];
                    f.elevation[idx] = (z * 28 / d.max(1)).min(28) as u8;

                    if z == 0 || (is_perimeter && z <= perim_z) {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, 31_000);
                        continue;
                    }

                    let mut sdf = wz - gh;
                    // 0=obsidian/ash terrain, 1=lava_tube_arch_rock, 2=scorched_timber_deck, 3=magma_channel
                    let mut mat_kind = 0u8;

                    // Natural arched lava-tube bridge E-W across the caldera (ny ≈ 0.50, nx ∈ [0.22..0.78])
                    let dy_b = (wy - 0.50 * hf).abs();
                    if nx >= 0.21 && nx <= 0.79 && dy_b <= bridge_hw_y + 1.2 && wz <= 9.0 {
                        let u = ((nx - 0.50) / 0.28).clamp(-1.0, 1.0);
                        let deck_top = 7.40 - 0.85 * u * u;
                        let mut b_sdf = ((nx - 0.50).abs() - 0.28)
                            .max(dy_b / wf - bridge_hw_y / wf)
                            .max((wz - deck_top) / wf);
                        b_sdf *= wf;
                        // Two bold elliptical lava-tube arches over the glowing magma moat!
                        for &ax_n in &[0.32f32, 0.68f32] {
                            let adx = (wx - ax_n * wf) / (0.068 * wf).max(1.0);
                            let adz = (wz - 2.0).max(0.0) / 3.9;
                            let hole_r = (adx * adx + adz * adz).sqrt();
                            let open = (hole_r - 1.0) * 3.2;
                            b_sdf = b_sdf.max(-open);
                        }
                        if b_sdf < sdf {
                            sdf = b_sdf;
                            mat_kind = if dy_b < bridge_hw_y - 0.9 && wz > deck_top - 0.95 {
                                2
                            } else {
                                1
                            };
                        }
                    }

                    // Cross N-S natural obsidian arch spanning the north/south magma moat
                    let dx_b = (wx - 0.50 * wf).abs();
                    if ny >= 0.22 && ny <= 0.78 && dx_b <= bridge_hw_x + 1.0 && wz <= 8.6 {
                        let u = ((ny - 0.50) / 0.27).clamp(-1.0, 1.0);
                        let deck_top = 7.05 - 0.75 * u * u;
                        let mut b_sdf = ((ny - 0.50).abs() - 0.27)
                            .max(dx_b / hf - bridge_hw_x / hf)
                            .max((wz - deck_top) / hf);
                        b_sdf *= hf;
                        for &ay_n in &[0.32f32, 0.68f32] {
                            let ady = (wy - ay_n * hf) / (0.064 * hf).max(1.0);
                            let adz = (wz - 2.0).max(0.0) / 3.7;
                            let hole_r = (ady * ady + adz * adz).sqrt();
                            b_sdf = b_sdf.max(-((hole_r - 1.0) * 3.0));
                        }
                        if b_sdf < sdf {
                            sdf = b_sdf;
                            mat_kind = 1;
                        }
                    }

                    // Stacked upper lava-tube cavern through the towering north caldera wall when depth >= 24!
                    if has_stacked_tube && ny < 0.26 && nx > 0.28 && nx < 0.72 && wz >= 6.8 && wz <= 9.8 {
                        let tube_r = (((ny - 0.14) / 0.045).powi(2) + ((wz - 8.2) / 1.25).powi(2)).sqrt();
                        let tube_sdf = (tube_r - 1.0) * 1.4;
                        if tube_sdf < 0.0 {
                            sdf = sdf.max(-tube_sdf);
                        }
                    }

                    let band = (if mat_kind == 1 || mat_kind == 2 { 0.48 } else { 0.58 }) / z_scale;
                    let solid_d = z16_sdf_to_density(sdf, band);
                    let solid_m = (solid_d * (C_NOM as f32)).round() as u16;
                    let mut cell_temp = 31_500i64;

                    let ring_d = (r - 0.182).abs();
                    let in_magma_channel = z <= magma_top_z && gh < 1.85 && (ring_d < 0.036 || (nx - 0.50).abs() < 0.028);

                    if in_magma_channel && mat_kind == 0 {
                        // Glowing superheated basalt magma channel (`950 K` = `95_000 cK`)!
                        f.mass[SP_ROCK][idx] = 1050;
                        cell_temp = 95_000;
                    } else if solid_m >= 15 {
                        match mat_kind {
                            1 => {
                                f.mass[SP_ROCK][idx] = solid_m;
                                cell_temp = 34_000;
                            }
                            2 => {
                                f.mass[SP_WOOD][idx] = solid_m;
                                cell_temp = 33_000;
                            }
                            _ => {
                                // Ash-draped caldera terraces over dark obsidian bedrock
                                if wz >= gh - 0.9 && wz >= 2.2 && wz <= 8.4 && r < 0.36 {
                                    f.mass[SP_ASH][idx] = (solid_m / 2).max(200);
                                    f.mass[SP_ROCK][idx] = solid_m.saturating_sub(solid_m / 2);
                                    cell_temp = 33_500;
                                } else {
                                    f.mass[SP_ROCK][idx] = solid_m;
                                    cell_temp = if wz < 2.5 { 46_000 } else { 31_500 };
                                }
                            }
                        }
                    }

                    // Volatile volcanic oil seepage pools on outer terraces
                    if solid_d < 0.35 && z == magma_top_z + 1 {
                        let d_oil1 = (nx - 0.34).hypot(ny - 0.64);
                        let d_oil2 = (nx - 0.66).hypot(ny - 0.36);
                        if d_oil1 < 0.036 || d_oil2 < 0.036 {
                            f.mass[SP_OIL][idx] = 520;
                            cell_temp = 39_500;
                        }
                    }

                    // Billowing steam & smoke vents rising all the way to upper atmosphere (`z = 3..d-2`)
                    if solid_d < 0.30 && z >= magma_top_z + 1 && z < d - 1 {
                        let rise_frac = (z as f32 / d as f32).clamp(0.0, 1.0);
                        let r_vent = 0.028 + rise_frac * 0.042;
                        let d_steam_vent = (nx - (0.38 + rise_frac * 0.02)).hypot(ny - 0.36);
                        let d_smoke_vent = (nx - (0.62 - rise_frac * 0.02)).hypot(ny - 0.64);
                        if d_steam_vent < r_vent {
                            let fall = 1.0 - d_steam_vent / r_vent;
                            f.mass[SP_STEAM][idx] = (340.0 * fall) as u16;
                            cell_temp = 41_000 - (rise_frac * 3_200.0) as i64;
                        } else if d_smoke_vent < r_vent {
                            let fall = 1.0 - d_smoke_vent / r_vent;
                            f.mass[SP_SMOKE][idx] = (360.0 * fall) as u16;
                            cell_temp = 54_000 - (rise_frac * 14_000.0) as i64;
                        }
                    }

                    f.set_cell_temp(idx, cell_temp.clamp(T_MIN, T_MAX));
                }
            }
        }
    }

    /// Biome 2 (`Sinkhole Caverns`): Dramatic karst sinkhole opening straight into a massive multi-pillar
    /// vaulted underground cavern (2-level stacked caverns at Z>=24!) with subterranean azure pools,
    /// hanging basalt arches, and vertical light shafts!
    fn init_biome_sinkhole_caverns(&mut self, seed_ox: f32, seed_oy: f32, seed_phase: f32) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let z_scale = (d as f32 / 16.0).clamp(1.0, 2.0);
        let perim_z = ((12.0 * z_scale).round() as usize).min(d.saturating_sub(2));
        let pool_top_z = (2.0 * z_scale).round() as usize;
        let has_stacked_caverns = d >= 24;

        // 6 sculpted monumental karst/basalt support pillars encircling the grand cavern
        let mut pillars = [(0.0f32, 0.0f32, 0.0f32); 6];
        for (i, slot) in pillars.iter_mut().enumerate() {
            let a = (i as f32) * (std::f32::consts::TAU / 6.0) + 0.35 + seed_phase * 0.2;
            let pr = 0.225;
            *slot = ((0.50 + pr * a.cos()) * wf, (0.50 + pr * a.sin()) * hf, (wf * 0.022).max(1.5));
        }

        // 3 secondary vertical skylight oculus shafts punching cleanly through the karst ceiling!
        let skylights: [(f32, f32, f32); 3] = [
            (0.32, 0.33, 0.052),
            (0.68, 0.33, 0.052),
            (0.50, 0.73, 0.055),
        ];

        // Surface karst rim trees around the sinkhole
        let rim_trees: &[(f32, f32, f32, f32)] = &[
            (0.24, 0.24, 3.8, 1.9),
            (0.76, 0.24, 3.8, 1.9),
            (0.22, 0.74, 3.6, 1.8),
            (0.78, 0.74, 3.6, 1.8),
        ];

        let mut roof_h = vec![0.0f32; w * h];
        for y in 0..h {
            let ny = (y as f32 + 0.5) / hf;
            for x in 0..w {
                let nx = (x as f32 + 0.5) / wf;
                let n_big = z16_fbm2(nx * 5.0 + seed_ox + 9.2, ny * 5.0 + seed_oy + 1.7);
                let n_med = z16_fbm2(nx * 13.0 + seed_ox, ny * 13.0 + seed_oy);
                let r = (nx - 0.50).hypot(ny - 0.50);
                // High rolling karst plateau (z ≈ 11.2..13.2 normalized) dipping gently near the sinkhole lip
                let rim_dip = 1.0 - z16_smoothstep(0.14, 0.24, r);
                let mut gh = 11.8 + n_big * 1.1 + n_med * 0.35 - rim_dip * 1.3;
                // Level West & East highland spawn pads cleanly
                let sp_w = 1.0 - z16_smoothstep(0.04, 0.09, (nx - 0.22).hypot(ny - 0.50));
                let sp_e = 1.0 - z16_smoothstep(0.04, 0.09, (nx - 0.80).hypot(ny - 0.50));
                gh = gh * (1.0 - sp_w) + 11.4 * sp_w;
                gh = gh * (1.0 - sp_e) + 11.4 * sp_e;
                roof_h[y * w + x] = gh.clamp(9.8, 13.4);
            }
        }

        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;
                let idx = y * w + x;
                let gh = roof_h[idx];
                let r = (nx - 0.50).hypot(ny - 0.50);
                let is_perimeter = x < 2 || x + 2 >= w || y < 2 || y + 2 >= h;

                for z in 0..d {
                    let wz = (z as f32 + 0.5) / z_scale;
                    let f = &mut self.layers[z];
                    f.elevation[idx] = (z * 28 / d.max(1)).min(28) as u8;

                    if z == 0 || (is_perimeter && z <= perim_z) {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, T_AMB);
                        continue;
                    }

                    // Base solid karst massif from bedrock up to plateau surface
                    let mut sdf = wz - gh;
                    // 0=karst_terrain, 1=pillar_arch_basalt, 2=hanging_timber_bridge, 3=tree_wood, 4=tree_plant
                    let mut mat_kind = 0u8;

                    // 1. Massive vaulted underground cavern(s)!
                    let mut inside_cavern = false;
                    if has_stacked_caverns {
                        // True 2-level stacked cavern system for Z>=24 (e.g. Z=32)!
                        // Lower abyssal grotto: wz ∈ [1.4..5.2] (z=2.8..10.4m)
                        // Upper cathedral cavern: wz ∈ [6.6..10.4] (z=13.2..20.8m)
                        let r_lo = (r / 0.36).powi(2) + ((wz - 3.3) / 1.95).powi(2);
                        let cav_lo = (r_lo.sqrt() - 1.0) * 2.2;
                        let r_hi = (r / 0.34).powi(2) + ((wz - 8.5) / 1.90).powi(2);
                        let cav_hi = (r_hi.sqrt() - 1.0) * 2.2;
                        let cav_sdf = cav_lo.min(cav_hi);
                        if cav_sdf < 0.0 && wz >= 1.35 {
                            inside_cavern = true;
                            sdf = sdf.max(-cav_sdf);
                        }
                    } else {
                        // Grand single vaulted cathedral cavern at Z=16: wz ∈ [1.5..8.4]
                        let r_cav = (r / 0.37).powi(2) + ((wz - 4.9) / 3.55).powi(2);
                        let cav_sdf = (r_cav.sqrt() - 1.0) * 3.2;
                        if cav_sdf < 0.0 && wz >= 1.45 {
                            inside_cavern = true;
                            sdf = sdf.max(-cav_sdf);
                        }
                    }

                    // 2. Central dramatic karst sinkhole opening straight through the roof(s) into the cavern!
                    let sink_r = 0.145 + 0.015 * ((wz - 4.0).max(0.0) / 8.0);
                    let sink_sdf = (r - sink_r) * wf.min(hf) * 0.5;
                    if sink_sdf < 0.0 && wz >= 1.35 {
                        inside_cavern = true;
                        sdf = sdf.max(-sink_sdf);
                    }

                    // 3. Secondary vertical light shafts piercing the karst roof cleanly
                    for &(sx, sy, sr) in &skylights {
                        let d_sky = ((nx - sx).hypot(ny - sy) - sr) * wf.min(hf) * 0.5;
                        if d_sky < 0.0 && wz >= 2.2 {
                            inside_cavern = true;
                            sdf = sdf.max(-d_sky);
                        }
                    }

                    // 4. Central dry sanctuary islet in the subterranean azure pool
                    if r < 0.078 && wz <= 2.9 {
                        let islet_sdf = (r - 0.072).max((wz - 2.6) / 14.0) * 14.0;
                        if islet_sdf < sdf {
                            sdf = islet_sdf;
                            mat_kind = 1;
                            inside_cavern = false;
                        }
                    }

                    // 5. 6 Monumental sculpted cavern pillars supporting the vaulted ceiling
                    for &(px, py, prad) in &pillars {
                        if wz >= 1.0 && wz <= gh - 0.5 {
                            let d_pil = (wx - px).hypot(wy - py) - prad;
                            if d_pil < sdf {
                                sdf = d_pil;
                                mat_kind = 1;
                                inside_cavern = false;
                            }
                        }
                    }

                    // 6. Hanging natural basalt & timber arch bridge spanning across the sinkhole (ny ≈ 0.50)
                    let dy_arch = (wy - 0.50 * hf).abs();
                    let arch_hw = (hf * 0.028).clamp(2.8, 4.4);
                    if nx >= 0.30 && nx <= 0.70 && dy_arch <= arch_hw + 1.0 && wz >= 6.8 && wz <= 11.4 {
                        let u = ((nx - 0.50) / 0.19).clamp(-1.0, 1.0);
                        let deck_top = 10.45 - 0.65 * u * u;
                        let arch_bot = 8.95 - 1.35 * (1.0 - u * u);
                        let box_sdf = ((nx - 0.50).abs() - 0.19)
                            .max((dy_arch - arch_hw) / wf)
                            .max((wz - deck_top) / wf)
                            .max((arch_bot - wz) / wf)
                            * wf;
                        if box_sdf < sdf {
                            sdf = box_sdf;
                            mat_kind = if dy_arch < arch_hw - 0.7 && wz > deck_top - 0.85 { 2 } else { 1 };
                            inside_cavern = false;
                        }
                    }

                    // 7. Surface highland trees flanking the sinkhole rim
                    for &(cnx, cny, tree_h, max_r) in rim_trees {
                        let tx = cnx * wf;
                        let ty = cny * hf;
                        let d_xy = (wx - tx).hypot(wy - ty);
                        if d_xy < max_r + 1.0 {
                            let t_base = roof_h[(ty as usize).min(h - 1) * w + (tx as usize).min(w - 1)] - 0.2;
                            let top_z = (t_base + tree_h).min(15.2);
                            let rel_z = wz - t_base;
                            if rel_z >= -0.3 && wz <= top_z + 0.3 {
                                let trunk_sdf = (d_xy - 0.65).max(wz - (top_z - 0.6)).max(-0.3 - rel_z);
                                if trunk_sdf < sdf {
                                    sdf = trunk_sdf;
                                    mat_kind = 3;
                                }
                                if rel_z >= 1.0 {
                                    let ft = ((wz - (t_base + 1.0)) / (top_z - (t_base + 1.0)).max(0.5)).clamp(0.0, 1.0);
                                    let foli_sdf = (d_xy - (max_r * (1.0 - ft * 0.7)).max(0.55))
                                        .max(wz - top_z)
                                        .max((t_base + 1.0) - wz);
                                    if foli_sdf < sdf {
                                        sdf = foli_sdf;
                                        mat_kind = 4;
                                    }
                                }
                            }
                        }
                    }

                    let band = (if mat_kind == 1 || mat_kind == 2 { 0.48 } else { 0.58 }) / z_scale;
                    let solid_d = z16_sdf_to_density(sdf, band);
                    let solid_m = (solid_d * (C_NOM as f32)).round() as u16;
                    let mut cell_temp = 28_600i64;

                    if solid_m >= 15 {
                        match mat_kind {
                            1 => f.mass[SP_ROCK][idx] = solid_m,
                            2 | 3 => f.mass[SP_WOOD][idx] = solid_m,
                            4 => f.mass[SP_PLANT][idx] = ((solid_m as u32 * 4) / 3).min(1100) as u16,
                            _ => {
                                if wz >= gh - 1.2 {
                                    f.mass[SP_SOIL][idx] = solid_m;
                                } else {
                                    f.mass[SP_ROCK][idx] = solid_m;
                                }
                            }
                        }
                    }

                    // Subterranean azure pools filling the cavern floor + waterfall pouring from North sinkhole lip!
                    if solid_d < 0.45 {
                        if z <= pool_top_z && inside_cavern && r < 0.34 && r > 0.072 {
                            f.mass[SP_WATER][idx] = 1080;
                            f.flow_e[idx] = 45;
                            cell_temp = 28_500;
                        } else if (pool_top_z..((11.0 * z_scale).round() as usize).min(d - 1)).contains(&z)
                            && (nx - 0.50).abs() < 0.022
                            && (ny - 0.36).abs() < 0.016
                        {
                            f.mass[SP_WATER][idx] = 760;
                            f.flow_s[idx] = 70;
                            cell_temp = 28_400;
                        } else if z == pool_top_z + 1 && r < 0.14 {
                            // Soft cavern mist rising in the central light shaft
                            f.mass[SP_STEAM][idx] = 140;
                            cell_temp = 37_400;
                        }
                    }

                    f.set_cell_temp(idx, cell_temp);
                }
            }
        }
    }

    /// Biome 3 (`Glacier Gorge`): Towering translucent glacial ice spires, natural carved ice bridges
    /// spanning a deep frozen/flowing canyon, sub-zero snowdrifts, and frozen waterfalls!
    fn init_biome_glacier_gorge(&mut self, seed_ox: f32, seed_oy: f32, seed_phase: f32) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let z_scale = (d as f32 / 16.0).clamp(1.0, 2.0);
        let perim_z = ((10.0 * z_scale).round() as usize).min(d.saturating_sub(2));
        let river_top_z = (2.0 * z_scale).round() as usize;
        let has_stacked_ice_cave = d >= 24;

        // Towering glacial ice spires flanking the gorge (nx, ny, spire_top_norm_z, base_radius_m)
        let spires: &[(f32, f32, f32, f32)] = &[
            (0.28, 0.22, 14.2, 4.2),
            (0.72, 0.24, 14.5, 4.4),
            (0.24, 0.76, 13.9, 4.0),
            (0.76, 0.78, 14.4, 4.3),
            (0.35, 0.50, 12.6, 3.0),
            (0.65, 0.50, 12.8, 3.0),
            (0.18, 0.36, 13.4, 3.6),
            (0.82, 0.64, 13.6, 3.6),
        ];

        let mut base_h = vec![0.0f32; w * h];
        for y in 0..h {
            let ny = (y as f32 + 0.5) / hf;
            let gorge_cx = 0.50 + 0.045 * ((ny - 0.50) * std::f32::consts::TAU * 1.2 + seed_phase * 0.3).sin();
            for x in 0..w {
                let nx = (x as f32 + 0.5) / wf;
                let n_big = z16_fbm2(nx * 5.6 + seed_ox - 3.4, ny * 5.6 + seed_oy + 6.2);
                let n_med = z16_fbm2(nx * 14.0 + seed_ox, ny * 14.0 + seed_oy);
                let dist_g = (nx - gorge_cx).abs();

                // Deep sheer frozen canyon down the center, high snow-covered glacial shelves on either side
                let wall_t = z16_smoothstep(0.055, 0.145, dist_g);
                let mut gh = 1.05 + wall_t * 8.2 + n_big * 1.35 + n_med * 0.40;

                // Level safe West & East spawn ridges
                let sp_w = 1.0 - z16_smoothstep(0.045, 0.095, (nx - 0.22).hypot(ny - 0.50));
                let sp_e = 1.0 - z16_smoothstep(0.045, 0.095, (nx - 0.80).hypot(ny - 0.50));
                gh = gh * (1.0 - sp_w) + 7.4 * sp_w;
                gh = gh * (1.0 - sp_e) + 7.4 * sp_e;

                base_h[y * w + x] = gh.clamp(0.95, 11.6);
            }
        }

        let bridge_hw_y = (hf * 0.030).clamp(3.0, 4.8);

        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            let gorge_cx = 0.50 + 0.045 * ((ny - 0.50) * std::f32::consts::TAU * 1.2 + seed_phase * 0.3).sin();
            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;
                let idx = y * w + x;
                let gh = base_h[idx];
                let dist_g = (nx - gorge_cx).abs();
                let is_perimeter = x < 2 || x + 2 >= w || y < 2 || y + 2 >= h;

                for z in 0..d {
                    let wz = (z as f32 + 0.5) / z_scale;
                    let f = &mut self.layers[z];
                    f.elevation[idx] = (z * 28 / d.max(1)).min(28) as u8;

                    if z == 0 || (is_perimeter && z <= perim_z) {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, 26_200);
                        continue;
                    }

                    let mut sdf = wz - gh;
                    // 0=bedrock+snowdrift, 1=glacial_ice_spire, 2=natural_ice_bridge, 3=frozen_waterfall_ice
                    let mut mat_kind = 0u8;

                    // 1. Towering translucent glacial ice spires!
                    for &(snx, sny, stop_z, srad) in spires {
                        let sx = snx * wf;
                        let sy = sny * hf;
                        let d_xy = (wx - sx).hypot(wy - sy);
                        if d_xy < srad + 1.2 && wz <= stop_z + 0.8 {
                            let base_z = base_h[(sy as usize).min(h - 1) * w + (sx as usize).min(w - 1)] - 0.6;
                            let h_t = ((wz - base_z) / (stop_z - base_z).max(1.0)).clamp(0.0, 1.0);
                            let cone_r = (srad * (1.0 - h_t * 0.86)).max(0.55);
                            let spire_sdf = (d_xy - cone_r).max(wz - stop_z);
                            if spire_sdf < sdf {
                                sdf = spire_sdf;
                                mat_kind = 1;
                            }
                        }
                    }

                    // 2. Two natural carved glacial ice bridges spanning cleanly over the deep gorge!
                    for &(by_n, deck_h) in &[(0.36f32, 7.60f32), (0.64f32, 6.80f32)] {
                        let dy_b = (wy - by_n * hf).abs();
                        if nx >= 0.30 && nx <= 0.70 && dy_b <= bridge_hw_y + 1.0 && wz <= deck_h + 0.8 {
                            let u = ((nx - 0.50) / 0.19).clamp(-1.0, 1.0);
                            let deck_top = deck_h - 0.65 * u * u;
                            let arch_bot = (deck_h - 1.65) - 1.35 * (1.0 - u * u);
                            let b_sdf = ((nx - 0.50).abs() - 0.19)
                                .max((dy_b - bridge_hw_y) / wf)
                                .max((wz - deck_top) / wf)
                                .max((arch_bot - wz) / wf)
                                * wf;
                            if b_sdf < sdf {
                                sdf = b_sdf;
                                mat_kind = 2;
                            }
                        }
                    }

                    // 3. Frozen waterfall cascade clinging to the west canyon cliff around (nx≈0.39, ny≈0.24)
                    if (nx - 0.39).abs() < 0.032 && (ny - 0.24).abs() < 0.032 && wz >= 2.0 && wz <= 9.2 {
                        let wf_sdf = ((nx - 0.39).hypot(ny - 0.24) - 0.026) * wf;
                        if wf_sdf < sdf {
                            sdf = wf_sdf;
                            mat_kind = 3;
                        }
                    }

                    // 4. Stacked glacial ice caves inside the western cliff when depth >= 24!
                    if has_stacked_ice_cave && nx > 0.14 && nx < 0.36 && ny > 0.54 && ny < 0.78 {
                        let c1 = (((nx - 0.25) / 0.08).powi(2) + ((wz - 3.2) / 1.4).powi(2)).sqrt() - 1.0;
                        let c2 = (((nx - 0.25) / 0.08).powi(2) + ((wz - 7.8) / 1.4).powi(2)).sqrt() - 1.0;
                        let ice_cav = c1.min(c2) * 1.4;
                        if ice_cav < 0.0 {
                            sdf = sdf.max(-ice_cav);
                        }
                    }

                    let band = (if mat_kind == 2 { 0.48 } else { 0.58 }) / z_scale;
                    let solid_d = z16_sdf_to_density(sdf, band);
                    let solid_m = (solid_d * (C_NOM as f32)).round() as u16;
                    let mut cell_temp = 25_600i64; // Sub-zero alpine glacial air/ice!

                    if solid_m >= 15 {
                        match mat_kind {
                            1 | 2 | 3 => {
                                f.mass[SP_ICE][idx] = solid_m;
                                cell_temp = if mat_kind == 2 { 26_400 } else { 25_400 };
                            }
                            _ => {
                                // Thick sub-zero snowdrifts blanketing the upper gorge ledges over dark granite
                                if wz >= gh - 1.8 && wz >= 4.2 {
                                    f.mass[SP_ICE][idx] = solid_m;
                                    cell_temp = 25_500;
                                } else {
                                    f.mass[SP_ROCK][idx] = solid_m;
                                    cell_temp = 26_400;
                                }
                            }
                        }
                    }

                    // Deep canyon river (flowing icy water + brash ice) + liquid waterfall core
                    if solid_d < 0.45 {
                        if z <= river_top_z && dist_g < 0.068 {
                            f.mass[SP_WATER][idx] = 1050;
                            if (x + y) % 9 == 0 {
                                f.mass[SP_ICE][idx] = 320;
                                cell_temp = 27_100;
                            } else {
                                cell_temp = 27_600;
                            }
                            f.flow_s[idx] = 125;
                        } else if z > river_top_z
                            && z <= ((9.0 * z_scale).round() as usize).min(d - 2)
                            && (nx - 0.415).abs() < 0.018
                            && (ny - 0.24).abs() < 0.018
                        {
                            f.mass[SP_WATER][idx] = 720;
                            cell_temp = 27_600;
                        }
                    }

                    f.set_cell_temp(idx, cell_temp);
                }
            }
        }
    }

    /// Biome 4 (`Geothermal Mist Basin`): Stepped travertine mineral terraces, boiling hot springs (`380–450 K`),
    /// dense buoyant rising steam clouds feeding the closed 3D hydrologic cycle, and lush warm-valley flora!
    fn init_biome_geothermal_basin(&mut self, seed_ox: f32, seed_oy: f32, seed_phase: f32) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let z_scale = (d as f32 / 16.0).clamp(1.0, 2.0);
        let perim_z = ((10.0 * z_scale).round() as usize).min(d.saturating_sub(2));
        let has_steam_chamber = d >= 24;

        // Satellite stepped boiling terrace pools (nx, ny, radius, pool_z_norm, temp_ck)
        let hot_pools: [(f32, f32, f32, f32, i64); 5] = [
            (0.50, 0.50, 0.105, 2.5, 44_200), // Central boiling geyser caldera (442 K!)
            (0.34, 0.36, 0.062, 4.4, 41_500), // NW travertine terrace spring (415 K)
            (0.66, 0.36, 0.062, 4.4, 40_500), // NE travertine terrace spring (405 K)
            (0.35, 0.66, 0.064, 4.8, 39_500), // SW travertine terrace spring (395 K)
            (0.65, 0.66, 0.064, 4.8, 42_000), // SE travertine terrace spring (420 K)
        ];

        // Lush warm-valley broadleaf/cedar groves thriving in the humid geothermal mist!
        let valley_trees: &[(f32, f32, f32, f32)] = &[
            (0.18, 0.22, 4.8, 2.4),
            (0.28, 0.18, 5.2, 2.6),
            (0.72, 0.18, 5.0, 2.5),
            (0.82, 0.24, 4.8, 2.4),
            (0.18, 0.78, 5.0, 2.5),
            (0.28, 0.84, 4.6, 2.3),
            (0.72, 0.84, 5.2, 2.6),
            (0.84, 0.76, 4.8, 2.4),
            (0.16, 0.40, 4.4, 2.2),
            (0.84, 0.60, 4.6, 2.3),
        ];

        let mut base_h = vec![0.0f32; w * h];
        for y in 0..h {
            let ny = (y as f32 + 0.5) / hf;
            for x in 0..w {
                let nx = (x as f32 + 0.5) / wf;
                let r = (nx - 0.50).hypot(ny - 0.50);
                let ang = (ny - 0.50).atan2(nx - 0.50) + seed_phase;
                let n_big = z16_fbm2(nx * 5.0 + seed_ox + 6.4, ny * 5.0 + seed_oy - 5.1);
                let n_med = z16_fbm2(nx * 12.0 + seed_ox, ny * 12.0 + seed_oy);

                // Stepped travertine amphitheater terraces descending smoothly into the central basin!
                let radial = 2.0 + z16_smoothstep(0.09, 0.44, r) * 8.4 + 0.35 * (ang * 3.0).sin() + n_big * 0.8;
                let step_size = 1.65;
                let step_idx = (radial / step_size).floor();
                let step_frac = (radial / step_size) - step_idx;
                let stepped = (step_idx + z16_smoothstep(0.72, 0.96, step_frac)) * step_size;
                let mut gh = stepped + n_med * 0.22;

                // Carve each stepped hot-spring basin cleanly
                for &(px, py, pr, pz, _) in &hot_pools {
                    let dp = (nx - px).hypot(ny - py) / pr;
                    let pw = 1.0 - z16_smoothstep(0.55, 1.05, dp);
                    gh = gh * (1.0 - pw) + (pz - 0.85) * pw;
                }

                // Safe West & East warm meadow spawn clearings
                let sp_w = 1.0 - z16_smoothstep(0.045, 0.095, (nx - 0.20).hypot(ny - 0.50));
                let sp_e = 1.0 - z16_smoothstep(0.045, 0.095, (nx - 0.80).hypot(ny - 0.50));
                gh = gh * (1.0 - sp_w) + 6.6 * sp_w;
                gh = gh * (1.0 - sp_e) + 6.6 * sp_e;

                base_h[y * w + x] = gh.clamp(1.2, 12.4);
            }
        }

        let bridge_hw_y = (hf * 0.026).clamp(2.8, 4.2);

        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;
                let idx = y * w + x;
                let gh = base_h[idx];
                let r = (nx - 0.50).hypot(ny - 0.50);
                let is_perimeter = x < 2 || x + 2 >= w || y < 2 || y + 2 >= h;

                for z in 0..d {
                    let wz = (z as f32 + 0.5) / z_scale;
                    let f = &mut self.layers[z];
                    f.elevation[idx] = (z * 28 / d.max(1)).min(28) as u8;

                    if z == 0 || (is_perimeter && z <= perim_z) {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, 30_500);
                        continue;
                    }

                    let mut sdf = wz - gh;
                    // 0=travertine_terrace, 1=timber_footbridge, 2=tree_wood, 3=tree_plant, 4=travertine_arch
                    let mut mat_kind = 0u8;

                    // Natural travertine & timber footbridge spanning the central steaming geyser caldera
                    let dy_b = (wy - 0.50 * hf).abs();
                    if nx >= 0.36 && nx <= 0.64 && dy_b <= bridge_hw_y + 0.9 && wz >= 3.2 && wz <= 6.2 {
                        let u = ((nx - 0.50) / 0.135).clamp(-1.0, 1.0);
                        let deck_top = 5.45 - 0.55 * u * u;
                        let arch_bot = 4.35 - 0.95 * (1.0 - u * u);
                        let b_sdf = ((nx - 0.50).abs() - 0.135)
                            .max((dy_b - bridge_hw_y) / wf)
                            .max((wz - deck_top) / wf)
                            .max((arch_bot - wz) / wf)
                            * wf;
                        if b_sdf < sdf {
                            sdf = b_sdf;
                            mat_kind = if dy_b < bridge_hw_y - 0.6 && wz > deck_top - 0.75 { 1 } else { 4 };
                        }
                    }

                    // Deep geothermal steam-vent cavern stacked under the amphitheater when depth >= 24!
                    if has_steam_chamber && r < 0.24 && wz >= 1.2 && wz <= 3.6 {
                        let ch_r = (r / 0.22).powi(2) + ((wz - 2.2) / 1.1).powi(2);
                        let ch_sdf = (ch_r.sqrt() - 1.0) * 1.4;
                        if ch_sdf < 0.0 {
                            sdf = sdf.max(-ch_sdf);
                        }
                    }

                    // Lush warm-valley trees & foliage
                    for &(cnx, cny, tree_h, max_r) in valley_trees {
                        let tx = cnx * wf;
                        let ty = cny * hf;
                        let d_xy = (wx - tx).hypot(wy - ty);
                        if d_xy < max_r + 1.2 {
                            let t_base = base_h[(ty as usize).min(h - 1) * w + (tx as usize).min(w - 1)] - 0.3;
                            let top_z = (t_base + tree_h).min(14.8);
                            let rel_z = wz - t_base;
                            if rel_z >= -0.3 && wz <= top_z + 0.4 {
                                let trunk_sdf = (d_xy - 0.68).max(wz - (top_z - 0.9)).max(-0.3 - rel_z);
                                if trunk_sdf < sdf {
                                    sdf = trunk_sdf;
                                    mat_kind = 2;
                                }
                                if rel_z >= 1.2 {
                                    let ft = ((wz - (t_base + 1.2)) / (top_z - (t_base + 1.2)).max(0.5)).clamp(0.0, 1.0);
                                    let canopy_r = (max_r * (1.0 - (ft - 0.35).abs() * 0.75)).max(0.65);
                                    let foli_sdf = (d_xy - canopy_r).max(wz - top_z).max((t_base + 1.2) - wz);
                                    if foli_sdf < sdf {
                                        sdf = foli_sdf;
                                        mat_kind = 3;
                                    }
                                }
                            }
                        }
                    }

                    let band = (if mat_kind == 1 || mat_kind == 4 { 0.48 } else { 0.58 }) / z_scale;
                    let solid_d = z16_sdf_to_density(sdf, band);
                    let solid_m = (solid_d * (C_NOM as f32)).round() as u16;
                    let mut cell_temp = 30_200i64; // Warm humid geothermal valley microclimate

                    if solid_m >= 15 {
                        match mat_kind {
                            1 | 2 => f.mass[SP_WOOD][idx] = solid_m,
                            3 => f.mass[SP_PLANT][idx] = ((solid_m as u32 * 4) / 3).min(1150) as u16,
                            4 => f.mass[SP_ROCK][idx] = solid_m,
                            _ => {
                                // Stepped mineral travertine rock/ash rims with lush mossy valley soil on outer banks
                                if r < 0.28 {
                                    f.mass[SP_ROCK][idx] = (solid_m * 3) / 4;
                                    f.mass[SP_ASH][idx] = solid_m - (solid_m * 3) / 4;
                                    cell_temp = if wz < 3.0 { 45_500 } else { 33_000 };
                                } else if wz >= gh - 1.4 {
                                    f.mass[SP_SOIL][idx] = solid_m;
                                    if (x.wrapping_mul(31) ^ y.wrapping_mul(17)) % 5 == 0 {
                                        f.mass[SP_PLANT][idx] = 280;
                                    }
                                } else {
                                    f.mass[SP_ROCK][idx] = solid_m;
                                }
                            }
                        }
                    }

                    // Boiling hot springs (380–450 K) + dense rising steam columns feeding the 3D hydrologic cycle!
                    if solid_d < 0.45 {
                        for &(px, py, pr, pz, ptemp) in &hot_pools {
                            let dp = (nx - px).hypot(ny - py);
                            if dp < pr * 0.95 && wz >= pz - 1.0 && wz <= pz + 0.55 {
                                f.mass[SP_WATER][idx] = 1050;
                                cell_temp = ptemp; // 395..442 K boiling hot springs!
                                break;
                            } else if wz > pz + 0.55 && z < d - 1 {
                                let plume_h = (wz - pz) / 12.0;
                                let plume_r = pr * (0.85 + plume_h * 0.85);
                                if dp < plume_r {
                                    let fall = (1.0 - dp / plume_r).clamp(0.0, 1.0);
                                    let steam_m = (380.0 * fall * (1.0 - plume_h * 0.25).max(0.35)) as u16;
                                    if steam_m > f.mass[SP_STEAM][idx] {
                                        f.mass[SP_STEAM][idx] = steam_m;
                                        cell_temp = (40_200 - (plume_h * 2_400.0) as i64).clamp(37_400, 42_000);
                                    }
                                }
                            }
                        }
                    }

                    f.set_cell_temp(idx, cell_temp.clamp(T_MIN, T_MAX));
                }
            }
        }
    }

    /// Biome 5 (`BIOME_SKY_ABYSS` — "Sky-Isles & Abyss"): 3-Strata Stacked `Z=64` Vertical World
    /// (designed natively for `depth >= 48`, e.g., `256×256×64`, and gracefully scaling to `Z=16/32`):
    /// - **Stratum 1 (`z = 0..18m`):** Bedrock floor (`z = 0..2`), sprawling vaulted basalt caverns
    ///   (`z = 3..15m`) with glowing subterranean lava/oil rivers (`T ~ 660 K`), steam vents, basalt
    ///   pillars, Abyssal Heart shrine island (`z ~ 6m`), and a deep central sinkhole shaft!
    /// - **Stratum 2 (`z = 18..38m`):** Dramatic cliff-walled alpine canyon with a turquoise river
    ///   (`z = 21..25m`) pouring into the central abyssal sinkhole, a 3-span carved stone/timber
    ///   viaduct crossing at `z = 29m`, and lush pine-forested highlands (`z = 26..31m`).
    /// - **Stratum 3 (`z = 38..63m`):** 4 dramatic suspended rocky Floating Sky-Islands (`z = 42..53m` —
    ///   central Sky-Citadel + 3 flanking islands with tapering undersides and overhanging sky-bridges!),
    ///   giant `28m` plunging waterfalls pouring off the island rims straight down into the gorge & chasm,
    ///   and a living high-altitude Cloud Condensation Deck (`z = 56..62m`)!
    fn init_sky_abyss_vertical(&mut self, seed_ox: f32, seed_oy: f32, seed_phase: f32) {
        let w = self.width;
        let h = self.height;
        let d = self.depth;
        let wf = w as f32;
        let hf = h as f32;
        let z_scale_64 = d as f32 / 64.0;
        let band_scale = (1.0 / z_scale_64.max(0.25)).min(2.5);

        // Precompute 2D Surface Gorge + Pine Highlands heightfield (in Z=64 metric altitude `zm`)
        let mut surf_h = vec![28.0f32; w * h];
        let mut cav_noise = vec![0.0f32; w * h];
        for y in 0..h {
            let ny = (y as f32 + 0.5) / hf;
            let river_cx = 0.50 + 0.032 * ((ny - 0.50) * std::f32::consts::TAU * 1.1 + seed_phase * 0.25).sin();
            for x in 0..w {
                let nx = (x as f32 + 0.5) / wf;
                let idx = y * w + x;
                let n_big = z16_fbm2(nx * 5.0 + seed_ox, ny * 5.0 + seed_oy);
                let n_med = z16_fbm2(nx * 12.0 - 4.3 + seed_ox, ny * 12.0 + 7.7 + seed_oy);
                cav_noise[idx] = n_big;

                let dist_r = (nx - river_cx).abs();
                let canyon_t = z16_smoothstep(0.052, 0.145, dist_r);
                // Riverbed ~20.4m, steep canyon walls climbing to lush Pine Highlands at ~26.5..30.6m
                let mut gh = 20.4 + canyon_t * 8.4 + n_big * 1.45 + n_med * 0.45;

                // Level smooth West & East viaduct abutments at ny ~ 0.50
                let y_bridge = 1.0 - z16_smoothstep(0.030, 0.085, (ny - 0.50).abs());
                let w_abut = (1.0 - z16_smoothstep(0.025, 0.085, (nx - 0.275).abs())) * y_bridge;
                let e_abut = (1.0 - z16_smoothstep(0.025, 0.085, (nx - 0.725).abs())) * y_bridge;
                gh = gh * (1.0 - w_abut) + 28.6 * w_abut;
                gh = gh * (1.0 - e_abut) + 28.6 * e_abut;

                // Spawn clearings on West & East highlands
                let sp_w = 1.0 - z16_smoothstep(0.04, 0.09, (nx - 0.16).hypot(ny - 0.50));
                let sp_e = 1.0 - z16_smoothstep(0.04, 0.09, (nx - 0.84).hypot(ny - 0.50));
                gh = gh * (1.0 - sp_w) + 27.6 * sp_w;
                gh = gh * (1.0 - sp_e) + 27.6 * sp_e;

                // Keep highlands strictly below z=31.2m so bz=2 above highlands is pure empty sky!
                surf_h[idx] = gh.clamp(20.0, 30.8);
            }
        }

        // 4 Floating Sky-Islands concentrated in the dramatic central amphitheater (so outer sky cubes stay 100% empty sky!):
        // (nx, ny, radius_n, top_zm, taper_depth_m)
        let islands: [(f32, f32, f32, f32, f32); 4] = [
            (0.50, 0.50, 0.104, 50.2, 8.8), // Central Sky-Citadel (Shrine 2 at z ~ 50m!)
            (0.35, 0.35, 0.072, 48.0, 7.6), // NW Sky-Isle
            (0.65, 0.35, 0.072, 48.5, 7.8), // NE Sky-Isle
            (0.50, 0.67, 0.070, 47.0, 7.2), // South Sky-Isle
        ];
        // Overhanging Sky-Bridges linking Central Sky-Citadel to the 3 flanking islands
        let sky_bridges: [(f32, f32, f32, f32, f32); 3] = [
            (0.50, 0.50, 0.35, 0.35, 48.4),
            (0.50, 0.50, 0.65, 0.35, 48.7),
            (0.50, 0.50, 0.50, 0.67, 47.8),
        ];
        // Giant 28m plunging waterfall spill points: (nx, ny, top_zm, bot_zm, spill_dx, spill_dy)
        let waterfalls: [(f32, f32, f32, f32, f32, f32); 3] = [
            (0.50, 0.405, 49.4, 21.0, 0.0, -0.014), // Central Citadel plunging straight into abyssal sinkhole!
            (0.385, 0.375, 47.4, 21.2, 0.014, 0.008), // NW Sky-Isle cascade into canyon!
            (0.615, 0.375, 47.8, 21.2, -0.014, 0.008), // NE Sky-Isle cascade into canyon!
        ];
        // Abyssal cavern basalt support pillars
        let cav_pillars: [(f32, f32, f32); 4] = [
            (0.38, 0.39, 0.024),
            (0.62, 0.39, 0.024),
            (0.38, 0.61, 0.024),
            (0.62, 0.61, 0.024),
        ];
        // High-altitude Cloud Condensation Deck clusters (z = 56..62m), hovering above the floating archipelago
        let clouds: [(f32, f32, f32, f32); 4] = [
            (0.36, 0.34, 0.068, 59.0),
            (0.64, 0.34, 0.068, 59.5),
            (0.36, 0.66, 0.066, 58.5),
            (0.64, 0.66, 0.066, 59.0),
        ];

        // Bridge geometry (Stratum 2 Surface Viaduct at ny = 0.50, zm ~ 29.2m)
        let bridge_cy = 0.50 * hf;
        let bridge_hw_y = (hf * 0.032).clamp(3.2, 6.0);
        let bridge_x0 = 0.275 * wf;
        let bridge_x1 = 0.725 * wf;
        let bridge_cx = 0.50 * wf;
        let bridge_half_span = 0.225 * wf;

        for y in 0..h {
            let wy = y as f32 + 0.5;
            let ny = wy / hf;
            let river_cx = 0.50 + 0.032 * ((ny - 0.50) * std::f32::consts::TAU * 1.1 + seed_phase * 0.25).sin();
            let lava_cx = 0.50 + 0.065 * ((ny - 0.50) * std::f32::consts::TAU * 1.3 - seed_phase * 0.3).sin();
            let river_slope = ((1.0 - ny) * 5.0) as u8;

            for x in 0..w {
                let wx = x as f32 + 0.5;
                let nx = wx / wf;
                let idx = y * w + x;
                let gh = surf_h[idx];
                let n_big = cav_noise[idx];
                let dist_r = (nx - river_cx).abs();
                let dist_lava = (nx - lava_cx).abs();
                let is_perimeter = x < 2 || x + 2 >= w || y < 2 || y + 2 >= h;

                // Abyssal cavern radial field (interior vault `nx, ny in ~[0.27..0.73]`, leaving outer border pure uniform bedrock!)
                let cav_r = (((nx - 0.50) / 0.205).powi(2) + ((ny - 0.50) / 0.205).powi(2)).sqrt()
                    + n_big * 0.03;
                let cav_ceil = (15.2 - 5.0 * cav_r * cav_r).clamp(2.6, 15.4);
                let cav_floor = 2.6 + (n_big * 0.5 + 0.5) * 0.55;
                // Raised dry basalt sanctuary platform for Shrine 0 (ABYSSAL HEART) at (0.50, 0.64)
                let d_abyss_shrine = (nx - 0.50).hypot(ny - 0.64);
                let on_abyss_shrine_pad = d_abyss_shrine < 0.062;
                let shrine_floor = if on_abyss_shrine_pad {
                    let t = (1.0 - d_abyss_shrine / 0.062).clamp(0.0, 1.0);
                    cav_floor.max(5.5 * (0.65 + 0.35 * t))
                } else {
                    cav_floor
                };

                // Central Abyssal Sinkhole shaft at (0.50, 0.38) connecting Surface Gorge to Abyssal Cavern!
                let d_sink = (nx - 0.50).hypot(ny - 0.38);
                let sink_r = 0.056 + n_big * 0.008;

                let h_tree = ((x.wrapping_mul(73_856_093)) ^ (y.wrapping_mul(19_349_663))) % 1000;

                for z in 0..d {
                    let zm = (z as f32 + 0.5) / z_scale_64;
                    let f = &mut self.layers[z];
                    f.elevation[idx] = ((z * 28 / d.max(1)).min(28) as u8)
                        .saturating_add(if (20.0..=25.0).contains(&zm) { river_slope } else { 0 });

                    // Solid indestructible bedrock base (`z = 0..=2` at Z=64)
                    if z == 0 || zm <= 2.5 {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, T_AMB);
                        continue;
                    }
                    // Outer bedrock cliffs up to Stratum 2 height (`zm <= 30.8`), keeping Stratum 3 pure empty sky!
                    if is_perimeter && zm <= 30.8 {
                        f.mass[SP_ROCK][idx] = 1200;
                        f.set_cell_temp(idx, T_AMB);
                        continue;
                    }

                    let mut sdf = zm - gh; // < 0 inside solid ground below surface highlands/gorge
                    let mut mat_kind = 0u8; // 0=terrain, 1=viaduct rock, 2=viaduct wood, 3=sky rock, 4=sky grass/top, 5=sky snow, 6=lava bed, 7=pine wood/plant

                    // --- Stratum 1: Vaulted Abyssal Cavern & Central Sinkhole Shaft ---
                    let mut inside_abyss_air = false;
                    if zm < 25.5 {
                        // Carve vaulted abyssal cavern (`cav_r < 1.0`, `zm in [shrine_floor .. cav_ceil]`)
                        if cav_r < 1.0 && zm > shrine_floor && zm < cav_ceil {
                            let d_floor = shrine_floor - zm;
                            let d_ceil = zm - cav_ceil;
                            let d_wall = (cav_r - 1.0) * 14.0;
                            let cav_sdf = d_floor.max(d_ceil).max(d_wall);
                            if cav_sdf < 0.0 {
                                sdf = sdf.max(-cav_sdf);
                                inside_abyss_air = true;
                            }
                        }
                        // Carve Central Sinkhole Shaft linking Surface Gorge (`zm ~ 24m`) -> Abyssal Cavern (`zm ~ 4m`)!
                        if d_sink < sink_r + 0.015 && zm > 3.6 && zm < 25.0 {
                            let sink_sdf = (d_sink - sink_r) * wf;
                            if sink_sdf < 0.0 {
                                sdf = sdf.max(-sink_sdf);
                                inside_abyss_air = true;
                            }
                        }
                        // Add back the 4 vaulted basalt pillars inside the abyssal cavern
                        if zm <= cav_ceil + 1.5 {
                            for &(px, py, pr) in &cav_pillars {
                                let dp = ((nx - px).hypot(ny - py) - pr) * wf;
                                if dp < sdf {
                                    sdf = dp;
                                    mat_kind = 1;
                                    inside_abyss_air = false;
                                }
                            }
                        }
                    }

                    // --- Stratum 2: Monumental 3-Span Arched Stone & Timber Viaduct (`zm ~ 29.2m`) ---
                    let dy_b = (wy - bridge_cy).abs();
                    if wx >= bridge_x0 - 1.5
                        && wx <= bridge_x1 + 1.5
                        && dy_b <= bridge_hw_y + 1.5
                        && (20.0..=31.2).contains(&zm)
                    {
                        let u = ((wx - bridge_cx) / bridge_half_span).clamp(-1.0, 1.0);
                        let deck_top = 29.3 - 0.65 * u * u;
                        let prism_x = (wx - bridge_cx).abs() - bridge_half_span;
                        let prism_y = dy_b - bridge_hw_y;
                        let prism_z = (zm - deck_top).max(20.0 - zm);
                        let mut bridge_sdf = prism_x.max(prism_y).max(prism_z);

                        // 3 tall open arches spanning the alpine river gorge
                        let arches: [(f32, f32, f32, f32); 3] = [
                            (0.382 * wf, 0.048 * wf, 23.2, 4.4),
                            (0.500 * wf, 0.052 * wf, 23.0, 4.8),
                            (0.618 * wf, 0.048 * wf, 23.2, 4.4),
                        ];
                        for &(ax, rx, spring_z, rz) in &arches {
                            let adx = (wx - ax) / rx.max(1.0);
                            let adz = (zm - spring_z).max(0.0) / rz;
                            let arch_r = (adx * adx + adz * adz).sqrt();
                            let open_sdf = (arch_r - 1.0) * rx.min(rz);
                            let rect_open = (wx - ax).abs() - rx;
                            let hole_sdf = if zm < spring_z { rect_open } else { open_sdf };
                            bridge_sdf = bridge_sdf.max(-hole_sdf);
                        }

                        // Stone parapets along north & south edges of viaduct
                        let para_y = (dy_b - (bridge_hw_y - 0.45)).abs() - 0.45;
                        let para_x = (wx - bridge_cx).abs() - (bridge_half_span - 0.6);
                        let para_z = (zm - (deck_top + 0.85)).max((deck_top - 0.3) - zm);
                        bridge_sdf = bridge_sdf.min(para_x.max(para_y).max(para_z));

                        if bridge_sdf < sdf {
                            sdf = bridge_sdf;
                            if dy_b < bridge_hw_y - 1.1 && zm > deck_top - 1.3 && zm <= deck_top + 0.25 {
                                mat_kind = 2; // timber deck
                            } else {
                                mat_kind = 1; // carved stone arches & parapets
                            }
                        }
                    }

                    // --- Stratum 3: 4 Floating Sky-Islands + Overhanging Sky-Bridges (`zm = 38..54m`) ---
                    if (38.0..=54.5).contains(&zm) {
                        for &(ix_n, iy_n, rad_n, top_zm, taper_m) in &islands {
                            let rn = (nx - ix_n).hypot(ny - iy_n) / rad_n;
                            if rn < 1.12 {
                                let top_cap = top_zm - 1.15 * rn * rn + n_big * 0.55;
                                let bot_taper = (top_zm - taper_m)
                                    + taper_m * rn.powf(1.30)
                                    + n_big * 0.85;
                                if top_cap > bot_taper - 0.6 {
                                    let d_top = zm - top_cap;
                                    let d_bot = bot_taper - zm;
                                    let d_rad = (rn - 1.0) * (rad_n * wf);
                                    let isl_sdf = d_top.max(d_bot).max(d_rad);
                                    if isl_sdf < sdf {
                                        sdf = isl_sdf;
                                        if zm >= top_cap - 1.35 {
                                            mat_kind = if rn > 0.72 { 5 } else { 4 };
                                        } else {
                                            mat_kind = 3;
                                        }
                                    }
                                }
                            }
                        }

                        // Overhanging arched Sky-Bridges linking Central Citadel to flanking Sky-Islands
                        for &(ax_n, ay_n, bx_n, by_n, mid_zm) in &sky_bridges {
                            let abx = bx_n - ax_n;
                            let aby = by_n - ay_n;
                            let len2 = (abx * abx + aby * aby).max(1e-6);
                            let t = (((nx - ax_n) * abx + (ny - ay_n) * aby) / len2).clamp(0.0, 1.0);
                            if (0.18..=0.82).contains(&t) {
                                let cx = ax_n + abx * t;
                                let cy = ay_n + aby * t;
                                let lat = (nx - cx).hypot(ny - cy) * wf;
                                let arch_z = mid_zm + (t * std::f32::consts::PI).sin() * 1.1;
                                let sb_sdf = (lat - 2.2)
                                    .max(zm - (arch_z + 0.75))
                                    .max((arch_z - 0.85) - zm);
                                if sb_sdf < sdf {
                                    sdf = sb_sdf;
                                    // Timber suspension Sky-Bridges: hold cleanly between intact islands,
                                    // burnable by wildfire, and snap cleanly if either island's core fails!
                                    mat_kind = 2;
                                }
                            }
                        }
                    }

                    // Convert SDF to exact conserved solid mass
                    let band = 0.62 * band_scale;
                    let solid_d = z16_sdf_to_density(sdf, band);
                    let solid_m = (solid_d * (C_NOM as f32)).round() as u16;
                    let mut cell_temp = if zm > 46.0 { 27_100 } else { T_AMB };

                    // Geothermal glowing cavern bed right under the subterranean lava river
                    if zm >= 2.5 && zm <= 3.6 && cav_r < 0.86 && dist_lava < 0.048 && !on_abyss_shrine_pad {
                        mat_kind = 6;
                    }

                    if solid_m >= 15 {
                        match mat_kind {
                            1 | 3 => f.mass[SP_ROCK][idx] = solid_m,
                            2 => f.mass[SP_WOOD][idx] = solid_m,
                            4 => {
                                // Lush mossy/pine Sky-Island crown
                                f.mass[SP_SOIL][idx] = solid_m;
                                if h_tree < 180 {
                                    f.mass[SP_PLANT][idx] = 280;
                                }
                            }
                            5 => {
                                // Frost-rimmed alpine ledge on Sky-Island edges
                                f.mass[SP_ICE][idx] = solid_m;
                                cell_temp = 26_600;
                            }
                            6 => {
                                f.mass[SP_ROCK][idx] = solid_m.max(960);
                                cell_temp = 66_000; // Glowing abyssal magma bed (~660 K)
                            }
                            _ => {
                                if zm < gh - 1.6 || zm < 19.5 {
                                    f.mass[SP_ROCK][idx] = solid_m;
                                } else {
                                    f.mass[SP_SOIL][idx] = solid_m;
                                    // Lush pine-forested surface highlands (z = 26..31m)
                                    if zm >= 26.0 && dist_r > 0.12 && h_tree < 260 {
                                        if h_tree < 90 {
                                            f.mass[SP_WOOD][idx] = 320;
                                        }
                                        f.mass[SP_PLANT][idx] = 300;
                                    }
                                }
                            }
                        }
                    }

                    // --- Fluids & Gases Across All 3 Strata ---
                    if solid_d < 0.45 {
                        // 1. Abyssal Subterranean Lava/Oil Rivers (`zm = 2.8..4.6m`, T ~ 660 K) + Steam Vents
                        if inside_abyss_air && zm <= 4.7 && cav_r < 0.86 && !on_abyss_shrine_pad && dist_lava < 0.042 {
                            f.mass[SP_OIL][idx] = 560;
                            f.flow_s[idx] = 70;
                            cell_temp = 66_000;
                        } else if inside_abyss_air && (4.6..=8.6).contains(&zm) {
                            let v1 = (nx - 0.42).hypot(ny - 0.56);
                            let v2 = (nx - 0.58).hypot(ny - 0.56);
                            let vs = (nx - 0.50).hypot(ny - 0.38);
                            if v1 < 0.024 || v2 < 0.024 || vs < 0.048 {
                                f.mass[SP_STEAM][idx] = 240;
                                cell_temp = 41_000;
                            }
                        }

                        // 2. Surface Canyon Turquoise River (`zm = 20.8..24.4m`) pouring into Central Abyssal Sinkhole!
                        if (20.6..=24.4).contains(&zm) && dist_r < 0.056 {
                            f.mass[SP_WATER][idx] = 1080;
                            f.flow_s[idx] = 130;
                            cell_temp = 28_400;
                        } else if (4.8..21.0).contains(&zm) && d_sink < 0.036 {
                            // Water cascading straight down the central sinkhole into the abyssal chasm!
                            f.mass[SP_WATER][idx] = 740;
                            f.flow_s[idx] = 85;
                            cell_temp = 28_500;
                        }

                        // 3. Giant 28m Plunging Waterfalls pouring off the Floating Sky-Island rims!
                        for &(wx_n, wy_n, top_zm, bot_zm, sdx, sdy) in &waterfalls {
                            if zm >= bot_zm && zm <= top_zm + 0.6 {
                                let drop_t = ((top_zm - zm) / (top_zm - bot_zm).max(1.0)).clamp(0.0, 1.0);
                                let cur_x = wx_n + sdx * drop_t.sqrt();
                                let cur_y = wy_n + sdy * drop_t.sqrt();
                                let wf_r = 0.017 + 0.008 * drop_t;
                                if (nx - cur_x).hypot(ny - cur_y) < wf_r {
                                    f.mass[SP_WATER][idx] = 880;
                                    f.flow_s[idx] = 95;
                                    cell_temp = 28_000;
                                }
                            } else if zm >= bot_zm - 0.5 && zm <= bot_zm + 3.8 {
                                // Plunge-pool waterfall mist at base of 28m fall
                                let mist_d = (nx - (wx_n + sdx)).hypot(ny - (wy_n + sdy));
                                if mist_d < 0.036 {
                                    f.mass[SP_STEAM][idx] = (220.0 * (1.0 - mist_d / 0.036)) as u16;
                                    cell_temp = 37_500;
                                }
                            }
                        }

                        // 4. High-Altitude Cloud Condensation Deck (`zm = 56..62m`)
                        if (56.0..=62.5).contains(&zm) {
                            for &(cx_n, cy_n, cr_n, cz_m) in &clouds {
                                let h_d = (nx - cx_n).hypot(ny - cy_n) / cr_n;
                                let v_d = (zm - cz_m).abs() / 2.2;
                                let c_r = (h_d * h_d + v_d * v_d).sqrt() + n_big * 0.14;
                                if c_r < 1.0 {
                                    let steam_m = (145.0 * (1.0 - c_r)).round() as u16;
                                    if steam_m >= 24 {
                                        f.mass[SP_STEAM][idx] = f.mass[SP_STEAM][idx].max(steam_m);
                                        cell_temp = 38_500;
                                    }
                                }
                            }
                        }
                    }

                    f.set_cell_temp(idx, cell_temp.clamp(T_MIN, T_MAX));
                }
            }
        }
    }

    /// Physical per-voxel solid density in `[0.0, 1.0]` computed directly from conserved solid species masses.
    /// Shared 1:1 by `encode_3d_brick_16` (GPU rendering) and `sample_solid_density_3d` (3D capsule physics).
    #[inline]
    pub fn voxel_solid_density(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        let f = &self.layers[iz];
        let idx = iy * self.width + ix;
        let solid_m = (f.mass[SP_ROCK][idx] as u32)
            + (f.mass[SP_SOIL][idx] as u32)
            + (f.mass[SP_WOOD][idx] as u32)
            + (f.mass[SP_ICE][idx] as u32)
            + ((f.mass[SP_ASH][idx] as u32) / 2)
            + ((f.mass[SP_PLANT][idx] as u32 * 3) / 4);
        (solid_m as f32 / (C_NOM as f32)).clamp(0.0, 1.0)
    }

    /// Continuous hardware-matched trilinear sample of `solid_density` in `(x, y, z)` world coordinates
    /// (voxel `(ix, iy, iz)` centered at `(ix + 0.5, iy + 0.5, iz + 0.5)`).
    #[inline]
    pub fn sample_solid_density_3d(&self, x: f32, y: f32, z: f32) -> f32 {
        if self.width == 0 || self.height == 0 || self.depth == 0 {
            return 0.0;
        }
        let fx = x - 0.5;
        let fy = y - 0.5;
        let fz = z - 0.5;

        let max_x = (self.width - 1) as i32;
        let max_y = (self.height - 1) as i32;
        let max_z = (self.depth - 1) as i32;

        let x0 = (fx.floor() as i32).clamp(0, max_x);
        let y0 = (fy.floor() as i32).clamp(0, max_y);
        let z0 = (fz.floor() as i32).clamp(0, max_z);
        let x1 = (x0 + 1).clamp(0, max_x);
        let y1 = (y0 + 1).clamp(0, max_y);
        let z1 = (z0 + 1).clamp(0, max_z);

        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let tz = (fz - z0 as f32).clamp(0.0, 1.0);

        let (x0, x1) = (x0 as usize, x1 as usize);
        let (y0, y1) = (y0 as usize, y1 as usize);
        let (z0, z1) = (z0 as usize, z1 as usize);

        let c000 = self.voxel_solid_density(x0, y0, z0);
        let c100 = self.voxel_solid_density(x1, y0, z0);
        let c010 = self.voxel_solid_density(x0, y1, z0);
        let c110 = self.voxel_solid_density(x1, y1, z0);
        let c001 = self.voxel_solid_density(x0, y0, z1);
        let c101 = self.voxel_solid_density(x1, y0, z1);
        let c011 = self.voxel_solid_density(x0, y1, z1);
        let c111 = self.voxel_solid_density(x1, y1, z1);

        let x00 = c000 + (c100 - c000) * tx;
        let x10 = c010 + (c110 - c010) * tx;
        let x01 = c001 + (c101 - c001) * tx;
        let x11 = c011 + (c111 - c011) * tx;

        let y0v = x00 + (x10 - x00) * ty;
        let y1v = x01 + (x11 - x01) * ty;
        y0v + (y1v - y0v) * tz
    }

    /// Analytical 3D central-difference gradient `∇solid_density(x, y, z)`.
    /// Points INTO solid (increasing density); negate for outward surface normal.
    #[inline]
    pub fn sample_solid_gradient_3d(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        const EPS: f32 = 0.25;
        let gx = self.sample_solid_density_3d(x + EPS, y, z)
            - self.sample_solid_density_3d(x - EPS, y, z);
        let gy = self.sample_solid_density_3d(x, y + EPS, z)
            - self.sample_solid_density_3d(x, y - EPS, z);
        let gz = self.sample_solid_density_3d(x, y, z + EPS)
            - self.sample_solid_density_3d(x, y, z - EPS);
        let inv = 1.0 / (2.0 * EPS);
        (gx * inv, gy * inv, gz * inv)
    }

    /// Directly encode a `16 × 16 × depth` column brick `(bx, by)` into the two compact `FM3D` `RGBA8`
    /// 3D GPU fields (Z-major `[z][dy][dx][4]`, length `depth * 16 * 16 * 4` bytes each):
    /// - `density_mat`: `[solid_density, fluid_density, gas_optical_density, roughness_wetness]`
    /// - `albedo_thermal`: `[albedo_r, albedo_g, albedo_b, blackbody_temp_glow]`
    /// Zero client species heuristics: derived purely from exact physical masses & temperatures.
    pub fn encode_3d_brick_16(&self, bx: usize, by: usize) -> (Vec<u8>, Vec<u8>) {
        let (dm, at, _) = self.encode_3d_brick_z_range(bx, by, 0, self.depth);
        (dm, at)
    }

    /// Encode a true `16 × 16 × sub_depth` cubic 3D sub-brick `(bx, by, bz)` covering vertical
    /// layers `[bz * 16 .. (bz * 16 + 16).min(depth)]`, returning `(density_mat, albedo_thermal, sub_depth)`.
    pub fn encode_3d_subbrick_16(&self, bx: usize, by: usize, bz: usize) -> (Vec<u8>, Vec<u8>, u8) {
        let z0 = bz * CHUNK_SIZE;
        let z1 = (z0 + CHUNK_SIZE).min(self.depth);
        self.encode_3d_brick_z_range(bx, by, z0, z1)
    }

    fn encode_3d_brick_z_range(
        &self,
        bx: usize,
        by: usize,
        z0: usize,
        z1: usize,
    ) -> (Vec<u8>, Vec<u8>, u8) {
        let sub_d = z1.saturating_sub(z0);
        let total_bytes = sub_d * CHUNK_SIZE * CHUNK_SIZE * 4;
        let mut density_mat = vec![0u8; total_bytes];
        let mut albedo_thermal = vec![0u8; total_bytes];
        if self.width == 0 || self.height == 0 || sub_d == 0 {
            return (density_mat, albedo_thermal, 0);
        }

        // Physical species base linear albedos & matte roughness values
        const SP_ALBEDO: [[f32; 3]; NUM_SPECIES] = [
            [0.12, 0.44, 0.66], // water (glacial azure)
            [0.76, 0.89, 0.96], // ice (frost cyan-white)
            [0.88, 0.94, 1.00], // steam (bright mist)
            [0.24, 0.36, 0.21], // soil (boreal meadow/moss)
            [0.35, 0.37, 0.41], // rock (granite/basalt)
            [0.42, 0.27, 0.16], // wood (weathered timber)
            [0.22, 0.21, 0.21], // ash
            [0.23, 0.22, 0.21], // smoke
            [0.11, 0.09, 0.13], // oil
            [0.13, 0.31, 0.17], // plant (alpine pine)
        ];
        const SP_ROUGH: [f32; NUM_SPECIES] = [
            0.06, // water
            0.22, // ice
            0.90, // steam
            0.86, // soil
            0.80, // rock
            0.78, // wood
            0.92, // ash
            0.92, // smoke
            0.12, // oil
            0.85, // plant
        ];

        let x0 = bx * CHUNK_SIZE;
        let y0 = by * CHUNK_SIZE;
        let mut out_i = 0usize;

        for z in z0..z1 {
            let f = &self.layers[z];
            for dy in 0..CHUNK_SIZE {
                let gy = (y0 + dy).min(self.height - 1);
                let row = gy * self.width;
                for dx in 0..CHUNK_SIZE {
                    let gx = (x0 + dx).min(self.width - 1);
                    let idx = row + gx;

                    let solid_d = self.voxel_solid_density(gx, gy, z);
                    let water = f.mass[SP_WATER][idx] as f32;
                    let oil = f.mass[SP_OIL][idx] as f32;
                    let steam = f.mass[SP_STEAM][idx] as f32;
                    let smoke = f.mass[SP_SMOKE][idx] as f32;

                    let fluid_d = ((water + oil) / (C_NOM as f32)).clamp(0.0, 1.0);
                    let gas_d = ((steam + smoke) / 650.0).clamp(0.0, 1.0);

                    let mut r_sum = 0.0f32;
                    let mut g_sum = 0.0f32;
                    let mut b_sum = 0.0f32;
                    let mut rough_sum = 0.0f32;
                    let mut w_sum = 0.0f32;

                    for s in 0..NUM_SPECIES {
                        let m = f.mass[s][idx] as f32;
                        if m > 0.0 {
                            r_sum += m * SP_ALBEDO[s][0];
                            g_sum += m * SP_ALBEDO[s][1];
                            b_sum += m * SP_ALBEDO[s][2];
                            rough_sum += m * SP_ROUGH[s];
                            w_sum += m;
                        }
                    }

                    let (mut ar, mut ag, mut ab, mut rough) = if w_sum > 0.0 {
                        let inv = 1.0 / w_sum;
                        (r_sum * inv, g_sum * inv, b_sum * inv, rough_sum * inv)
                    } else {
                        (0.36, 0.40, 0.44, 0.85)
                    };

                    // Wetness gloss reduction when liquid coats solid
                    if solid_d > 0.1 && fluid_d > 0.05 {
                        rough *= (1.0 - 0.65 * fluid_d).clamp(0.15, 1.0);
                    }

                    // Physical blackbody thermal glow directly from voxel temperature (cK)
                    let t_ck = f.cell_temp_ck(idx);
                    let glow = if w_sum > 0.0 && t_ck > 42_000 {
                        ((t_ck - 42_000) as f32 / 65_000.0).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if glow > 0.02 {
                        // Blend incandescent blackbody orange-amber into albedo
                        let k = (glow * 0.85).clamp(0.0, 0.92);
                        ar = ar * (1.0 - k) + 1.0 * k;
                        ag = ag * (1.0 - k) + (0.42 + 0.34 * glow) * k;
                        ab = ab * (1.0 - k) + (0.08 + 0.12 * glow) * k;
                    }

                    density_mat[out_i] = z16_to_u8(solid_d);
                    density_mat[out_i + 1] = z16_to_u8(fluid_d);
                    density_mat[out_i + 2] = z16_to_u8(gas_d);
                    density_mat[out_i + 3] = z16_to_u8(rough);

                    albedo_thermal[out_i] = z16_to_u8(ar);
                    albedo_thermal[out_i + 1] = z16_to_u8(ag);
                    albedo_thermal[out_i + 2] = z16_to_u8(ab);
                    albedo_thermal[out_i + 3] = z16_to_u8(glow);
                    out_i += 4;
                }
            }
        }

        (density_mat, albedo_thermal, sub_d as u8)
    }

    /// Measure maximum solid penetration (in meters) across a vertical 3D capsule shell at foot base `pos`.
    pub fn measure_capsule_penetration_3d(
        &self,
        pos: (f32, f32, f32),
        radius: f32,
        height: f32,
    ) -> f32 {
        let mut max_pen = 0.0f32;
        for (qx, qy, qz) in z16_capsule_shell_samples(pos, radius, height) {
            let d = self.sample_solid_density_3d(qx, qy, qz);
            if d > 0.5 {
                let (gx, gy, gz) = self.sample_solid_gradient_3d(qx, qy, qz);
                let g_len = (gx * gx + gy * gy + gz * gz).sqrt().max(0.55);
                let pen = (d - 0.5) / g_len;
                max_pen = max_pen.max(pen);
            }
        }
        max_pen
    }

    /// Measure vertical open-air headroom clearance (in meters) directly above foot base `pos = (x, y, z)`.
    pub fn measure_vertical_clearance_3d(&self, pos: (f32, f32, f32)) -> f32 {
        let step = 0.1f32;
        let max_z = self.depth as f32 - 0.25;
        let mut z = pos.2 + 0.15;
        while z < max_z {
            if self.sample_solid_density_3d(pos.0, pos.1, z) >= 0.5 {
                return (z - pos.2).max(0.0);
            }
            z += step;
        }
        (self.depth as f32 - pos.2).max(0.0)
    }

    /// Continuous trilinear `∇solid_density` 3D capsule collision & locomotion solver.
    /// - `pos`: `(x, y, z)` foot base center in metric world coordinates (`z` is altitude UP).
    /// - `vel`: `(vx, vy, vz)` velocity in meters/sec.
    /// - `radius`, `height`: capsule radius & total height in meters (e.g. `0.55`, `1.80`).
    /// - `dt`: timestep in seconds.
    /// Returns `(new_pos, new_vel, grounded)` with smooth wall sliding, floor settling,
    /// cave ceiling clearance, and 3-span arched bridge traversal!
    pub fn step_capsule_3d(
        &self,
        pos: (f32, f32, f32),
        vel: (f32, f32, f32),
        radius: f32,
        height: f32,
        dt: f32,
    ) -> ((f32, f32, f32), (f32, f32, f32), bool) {
        let gravity: f32 = 18.0 * self.world_laws.gravity_g.clamp(0.15, 2.5);
        if self.width < 4 || self.height < 4 || self.depth < 2 || dt <= 0.0 {
            return (pos, vel, false);
        }

        let substeps = ((dt / 0.008).ceil() as usize).clamp(1, 8);
        let sub_dt = dt / substeps as f32;
        let mut grounded = false;

        let (mut px, mut py, mut pz) = pos;
        let (mut vx, mut vy, mut vz) = vel;

        let min_x = radius + 2.0;
        let max_x = self.width as f32 - radius - 2.0;
        let min_y = radius + 2.0;
        let max_y = self.height as f32 - radius - 2.0;
        let max_z = (self.depth as f32 - height - 0.1).max(0.5);

        for _ in 0..substeps {
            let alt_grav_scale = if self.depth >= 48 && pz >= 0.65 * (self.depth as f32) {
                0.45
            } else {
                1.0
            };
            vz -= gravity * alt_grav_scale * sub_dt;
            px = (px + vx * sub_dt).clamp(min_x, max_x);
            py = (py + vy * sub_dt).clamp(min_y, max_y);
            pz = (pz + vz * sub_dt).clamp(0.5, max_z);

            // Iterative Gauss-Seidel pushout along analytical -∇solid_density
            for _iter in 0..10 {
                let mut worst_pen = 0.0f32;
                let mut best_n = (0.0f32, 0.0f32, 1.0f32);

                let samples = z16_capsule_shell_samples((px, py, pz), radius, height);
                for (sidx, &(qx, qy, qz)) in samples.iter().enumerate() {
                    let d = self.sample_solid_density_3d(qx, qy, qz);
                    if d > 0.5 {
                        let (gx, gy, gz) = self.sample_solid_gradient_3d(qx, qy, qz);
                        let g_len = (gx * gx + gy * gy + gz * gz).sqrt();
                        let outward_n = if g_len > 1e-3 {
                            (-gx / g_len, -gy / g_len, -gz / g_len)
                        } else if sidx < 5 {
                            (0.0, 0.0, 1.0)
                        } else if sidx >= 21 {
                            (0.0, 0.0, -1.0)
                        } else {
                            let dx = px - qx;
                            let dy = py - qy;
                            let dl = dx.hypot(dy);
                            if dl > 1e-4 { (dx / dl, dy / dl, 0.0) } else { (0.0, 0.0, 1.0) }
                        };
                        let pen = (d - 0.5) / g_len.max(0.55);
                        if pen > worst_pen {
                            worst_pen = pen;
                            best_n = outward_n;
                        }
                    }
                }

                if worst_pen <= 1e-4 {
                    break;
                }

                let push = worst_pen.min(0.35);
                px = (px + best_n.0 * push).clamp(min_x, max_x);
                py = (py + best_n.1 * push).clamp(min_y, max_y);
                pz = (pz + best_n.2 * push).clamp(0.5, max_z);

                if best_n.2 >= 0.55 {
                    grounded = true;
                    if vz < 0.0 {
                        vz = 0.0;
                    }
                }

                // Project velocity onto tangent plane for smooth wall sliding & ceiling deflection
                let vn = vx * best_n.0 + vy * best_n.1 + vz * best_n.2;
                if vn < 0.0 {
                    vx -= best_n.0 * vn;
                    vy -= best_n.1 * vn;
                    vz -= best_n.2 * vn;
                }
            }
        }

        ((px, py, pz), (vx, vy, vz), grounded)
    }
}

#[inline]
fn field_solid_density(f: &Field, idx: usize) -> f32 {
    let solid_m = (f.mass[SP_ROCK][idx] as u32)
        + (f.mass[SP_SOIL][idx] as u32)
        + (f.mass[SP_WOOD][idx] as u32)
        + (f.mass[SP_ICE][idx] as u32)
        + ((f.mass[SP_ASH][idx] as u32) / 2)
        + ((f.mass[SP_PLANT][idx] as u32 * 3) / 4);
    (solid_m as f32 / (C_NOM as f32)).clamp(0.0, 1.0)
}

#[inline]
fn z16_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[inline]
fn z16_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn z16_sdf_to_density(sdf: f32, band: f32) -> f32 {
    let t = (0.5 - sdf / (2.0 * band)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn z16_hash2(ix: i32, iy: i32) -> f32 {
    let mut h = ix
        .wrapping_mul(374_761_393)
        .wrapping_add(iy.wrapping_mul(668_265_263));
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    ((h ^ (h >> 16)) as u32 as f32) / (u32::MAX as f32)
}

fn z16_value_noise2(x: f32, y: f32) -> f32 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let fx = x - ix as f32;
    let fy = y - iy as f32;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let c00 = z16_hash2(ix, iy);
    let c10 = z16_hash2(ix + 1, iy);
    let c01 = z16_hash2(ix, iy + 1);
    let c11 = z16_hash2(ix + 1, iy + 1);
    let x0 = c00 + (c10 - c00) * ux;
    let x1 = c01 + (c11 - c01) * ux;
    x0 + (x1 - x0) * uy
}

fn z16_fbm2(x: f32, y: f32) -> f32 {
    let mut sum = 0.0f32;
    let mut amp = 0.5f32;
    let mut freq = 1.0f32;
    for _ in 0..4 {
        sum += amp * (z16_value_noise2(x * freq, y * freq) * 2.0 - 1.0);
        freq *= 2.07;
        amp *= 0.48;
    }
    sum
}

/// 26 shell sample points on a vertical capsule with foot base `pos = (x, y, z)` (`z` up).
fn z16_capsule_shell_samples(pos: (f32, f32, f32), radius: f32, height: f32) -> [(f32, f32, f32); 26] {
    let (px, py, pz) = pos;
    let r = radius;
    let z_bot = r;
    let z_top = (height - r).max(r + 0.05);
    let z_mid = 0.5 * (z_bot + z_top);
    let d = r * std::f32::consts::FRAC_1_SQRT_2;
    let cap_r = r * 0.7071;
    let cap_dz = r * 0.7071;

    [
        // Bottom pole + 4 lower hemisphere points (5)
        (px, py, pz),
        (px + cap_r, py, pz + z_bot - cap_dz),
        (px - cap_r, py, pz + z_bot - cap_dz),
        (px, py + cap_r, pz + z_bot - cap_dz),
        (px, py - cap_r, pz + z_bot - cap_dz),
        // Lower cylinder ring (8)
        (px + r, py, pz + z_bot),
        (px - r, py, pz + z_bot),
        (px, py + r, pz + z_bot),
        (px, py - r, pz + z_bot),
        (px + d, py + d, pz + z_bot),
        (px - d, py + d, pz + z_bot),
        (px + d, py - d, pz + z_bot),
        (px - d, py - d, pz + z_bot),
        // Mid-waist ring (4)
        (px + r, py, pz + z_mid),
        (px - r, py, pz + z_mid),
        (px, py + r, pz + z_mid),
        (px, py - r, pz + z_mid),
        // Upper cylinder ring (4)
        (px + r, py, pz + z_top),
        (px - r, py, pz + z_top),
        (px, py + r, pz + z_top),
        (px, py - r, pz + z_top),
        // Top hemisphere + top pole (5)
        (px + cap_r, py, pz + z_top + cap_dz),
        (px - cap_r, py, pz + z_top + cap_dz),
        (px, py + cap_r, pz + z_top + cap_dz),
        (px, py - cap_r, pz + z_top + cap_dz),
        (px, py, pz + height),
    ]
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

    #[test]
    fn test_physical_actuator_conservation_and_entity_physics() {
        let mut vf = VolumeField::new_shattered_reach_3d(64, 64, 4);

        // 1. Oil + Heat ignition on dry west bank (L1): verify exact conservation & positive delta_temp_spike
        let (ox, oy) = (12.0f32, 32.0f32);
        let (dm_oil, de_oil) = vf.apply_physical_actuator(ox, oy, 1, 0.0, 0.0, 1.0, 0.0, 3.5);
        assert!(dm_oil > 0 && de_oil > 0, "oil injection should add mass & chemical+thermal energy");
        assert_eq!(vf.current_accounted_mass() - vf.initial_total_mass, 0);
        assert_eq!(vf.current_accounted_energy() - vf.initial_total_energy, 0);

        let (_dm_heat, de_heat) = vf.apply_physical_actuator(ox, oy, 1, 0.75, 0.0, 0.0, 0.0, 3.5);
        assert!(de_heat > 0, "positive thermal flux should inject energy");
        assert_eq!(vf.current_accounted_mass() - vf.initial_total_mass, 0);
        assert_eq!(vf.current_accounted_energy() - vf.initial_total_energy, 0);

        let mut max_temp_spike = 0.0f32;
        for tick in 0..8 {
            let raws = vf.refsim_raw_outputs_all();
            let stats = vf.step_3d(&raws);
            assert_eq!(stats.mass_error, 0, "mass error during oil ignition at tick {}", tick);
            assert_eq!(stats.energy_error, 0, "energy error during oil ignition at tick {}", tick);
            let phys = vf.sample_entity_physics(ox, oy, 1);
            max_temp_spike = max_temp_spike.max(phys.delta_temp_spike);
        }
        assert!(
            max_temp_spike > 0.0,
            "expected positive delta_temp_spike from exothermic oil combustion, got {}",
            max_temp_spike
        );

        // 2. Water + Heat flash-boil in central fjord (L1): verify positive delta_steam_burst & conservation
        let water_cell = (0..(64 * 64))
            .find(|&i| {
                let x = i % 64;
                let y = i / 64;
                x > 6 && x < 58 && y > 6 && y < 58 && vf.layers[1].mass[SP_WATER][i] >= 800
            })
            .expect("shattered reach should have fjord water");
        let (wx, wy) = ((water_cell % 64) as f32, (water_cell / 64) as f32);
        vf.apply_physical_actuator(wx, wy, 1, 1.0, 0.0, 0.0, 0.0, 3.5);
        assert_eq!(vf.current_accounted_mass() - vf.initial_total_mass, 0);
        assert_eq!(vf.current_accounted_energy() - vf.initial_total_energy, 0);

        let mut max_steam_burst = 0.0f32;
        for tick in 0..6 {
            let raws = vf.refsim_raw_outputs_all();
            let stats = vf.step_3d(&raws);
            assert_eq!(stats.mass_error, 0, "mass error during flash-boil at tick {}", tick);
            assert_eq!(stats.energy_error, 0, "energy error during flash-boil at tick {}", tick);
            let phys = vf.sample_entity_physics(wx, wy, 1);
            max_steam_burst = max_steam_burst.max(phys.delta_steam_burst);
        }
        assert!(
            max_steam_burst > 0.0,
            "expected positive delta_steam_burst from water+heat flash-boil, got {}",
            max_steam_burst
        );

        // 3. Cryo extractor + smoke + L2 structural support check
        let (_dm_s, _de_s) = vf.apply_physical_actuator(ox, oy, 1, -0.8, 0.4, 0.0, 0.9, 3.0);
        assert_eq!(vf.current_accounted_mass() - vf.initial_total_mass, 0);
        assert_eq!(vf.current_accounted_energy() - vf.initial_total_energy, 0);
        let phys_smk = vf.sample_entity_physics(ox, oy, 1);
        assert!(phys_smk.smoke_opacity > 0.0, "smoke injection should increase smoke_opacity");
        assert!(phys_smk.support_solid, "west bank soil should provide solid support");

        let phys_air = vf.sample_entity_physics(32.0, 26.0, 2);
        assert!(!phys_air.support_solid, "open air over fjord on L2 should have !support_solid");
    }

    #[test]
    fn test_neural_3d_vertical_flux_conservation() {
        let mut vf = VolumeField::new_shattered_reach_3d(64, 64, 4);
        let len = 64 * 64;
        let n_if = 3usize;
        let stride_z = 5 * len;

        // Seed steam/smoke & fire on L1 so both gravity (L2->L1->L0) and buoyancy (L1->L2->L3) are active
        vf.apply_steam_burst_3d(32, 32, 1, 5);
        vf.apply_fire_3d(24, 42, 1, 4, 1.8);

        // 1. Realistic neural vertical fluxes (downward water/oil, upward steam/smoke, vertical heat)
        let mut z_flux = vec![0.0f32; n_if * stride_z];
        for z in 0..n_if {
            let base = z * stride_z;
            for idx in 0..len {
                z_flux[base + 0 * len + idx] = -0.35; // downward water plunge
                z_flux[base + 1 * len + idx] = -0.25; // downward oil cascade
                z_flux[base + 2 * len + idx] = 0.40;  // upward steam buoyancy
                z_flux[base + 3 * len + idx] = 0.35;  // upward smoke buoyancy
                z_flux[base + 4 * len + idx] = 2.5;   // upward heat conduction/convection
            }
        }
        let raws = vf.refsim_raw_outputs_all();
        let st = vf.step_3d_with_neural_z(&raws, Some(&z_flux));
        assert_eq!(st.mass_error, 0, "realistic neural z_flux mass error");
        assert_eq!(st.energy_error, 0, "realistic neural z_flux energy error");
        assert!(st.vertical_gravity_mass > 0, "expected downward gravity mass transfer");
        assert!(st.vertical_buoyancy_mass > 0, "expected upward buoyancy mass transfer");

        // 2. Adversarial / extreme neural z_flux values (NaN, +/-Inf, +/-1e9, huge alternating signs)
        for step in 0..20 {
            for (i, v) in z_flux.iter_mut().enumerate() {
                *v = match (i + step) % 9 {
                    0 => f32::NAN,
                    1 => f32::INFINITY,
                    2 => f32::NEG_INFINITY,
                    3 => 1e9,
                    4 => -1e9,
                    5 => 9999.5,
                    6 => -9999.5,
                    7 => 500.0,
                    _ => -500.0,
                };
            }
            // Also verify diurnal solar cycle coupling stays exactly conserved!
            let day_t = if step % 2 == 0 { 0.0 } else { 0.5 }; // midnight & noon
            vf.apply_solar_cycle_3d(day_t, step as u64);
            let (_g, _b) = vf.apply_neural_3d_vertical_fluxes(&z_flux);
            assert_eq!(
                vf.current_accounted_mass() - vf.initial_total_mass,
                0,
                "adversarial neural z_flux mass violation at step {}",
                step
            );
            assert_eq!(
                vf.current_accounted_energy() - vf.initial_total_energy,
                0,
                "adversarial neural z_flux energy violation at step {}",
                step
            );
            for l in &vf.layers {
                for idx in 0..len {
                    assert!(l.cell_volume(idx) <= H_CAP, "H_CAP violated at step {}", step);
                    if l.cell_heat_capacity(idx) > 0 {
                        let t = l.cell_temp_ck(idx);
                        assert!(
                            (T_MIN..=T_MAX).contains(&t),
                            "temp out of bounds ({}) at step {}",
                            t,
                            step
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_volume_z16_conservation_cave_bridge() {
        let mut vf = VolumeField::new_shattered_reach_3d(128, 128, 16);
        assert_eq!((vf.width, vf.height, vf.depth), (128, 128, 16));
        assert_eq!(vf.layers.len(), 16);

        // Verify key isotropic 3D landmarks exist:
        // 1. Arched bridge deck at (x=64, y=64, z=7..8) with open air (z=5) & flowing river (z=1..2) directly underneath!
        assert!(
            vf.sample_solid_density_3d(64.0, 64.0, 7.8) >= 0.75,
            "expected solid bridge deck at crown"
        );
        assert!(
            vf.sample_solid_density_3d(64.0, 64.0, 5.0) < 0.15,
            "expected open archway air directly under bridge center"
        );
        assert!(
            vf.layers[1].mass[SP_WATER][vf.layers[1].idx(64, 64)] >= 500,
            "expected flowing river water beneath center bridge arch"
        );

        // 2. Vaulted basalt cave tunnel (z=1..5) with solid mountain roof overhead (z=8..10)
        let cave_x = 0.24 * 128.0;
        let cave_y = 0.72 * 128.0;
        assert!(
            vf.sample_solid_density_3d(cave_x, cave_y, 3.5) < 0.15,
            "expected hollow vaulted cave interior at z=3.5"
        );
        assert!(
            vf.sample_solid_density_3d(cave_x, cave_y, 8.5) >= 0.85,
            "expected solid basalt mountain roof above cave at z=8.5"
        );

        // 3. Upper highland glacial basin at z=10 & plunging waterfall
        let basin_idx = vf.layers[10].idx(64, (0.15 * 128.0) as usize);
        assert!(
            vf.layers[10].mass[SP_WATER][basin_idx] >= 500,
            "expected glacial basin water at z=10"
        );

        // 4. Verify 16x16x16 FM3D brick encoding dimensions and non-trivial channels (bridge + wildfire)
        let (dm_br, at_br) = vf.encode_3d_brick_16(4, 4);
        assert_eq!(dm_br.len(), 16 * 16 * 16 * 4);
        assert_eq!(at_br.len(), 16 * 16 * 16 * 4);
        let (_, at_fire) = vf.encode_3d_brick_16(5, 3); // contains East wildfire bluff (x≈94, y≈48)
        let max_glow = at_fire.chunks_exact(4).map(|c| c[3]).max().unwrap_or(0);
        assert!(max_glow > 40, "expected active wildfire thermal glow in FM3D brick, got {}", max_glow);

        // 5. Apply physical actuators across multiple layers (cave, bridge, waterfall) & step_3d with vertical exchange
        let (dm1, de1) = vf.apply_physical_actuator(64.0, 64.0, 8, 0.85, 0.0, 0.6, 0.0, 3.5);
        let (_dm2, _de2) = vf.apply_physical_actuator(cave_x, cave_y, 2, -0.75, 0.5, 0.0, 0.4, 3.0);
        assert!(dm1 > 0 && de1 > 0);
        assert_eq!(vf.current_accounted_mass() - vf.initial_total_mass, 0);
        assert_eq!(vf.current_accounted_energy() - vf.initial_total_energy, 0);

        let mut total_grav = 0i64;
        let mut total_buoy = 0i64;
        for tick in 0..8 {
            vf.apply_weather_and_springs_3d((tick % 4) as u8, tick);
            vf.apply_solar_cycle_3d(if tick % 2 == 0 { 0.1 } else { 0.5 }, tick);
            let raws = vf.refsim_raw_outputs_all();
            let stats = vf.step_3d(&raws);
            assert_eq!(stats.mass_error, 0, "Z=16 mass_error != 0 at tick {}", tick);
            assert_eq!(stats.energy_error, 0, "Z=16 energy_error != 0 at tick {}", tick);
            total_grav += stats.vertical_gravity_mass;
            total_buoy += stats.vertical_buoyancy_mass;
        }
        assert!(total_grav > 0, "expected 3D waterfall gravity transport in Z=16");
        assert!(total_buoy > 0, "expected 3D wildfire/mist buoyancy rise in Z=16");
    }

    #[test]
    fn test_volume_z16_capsule_cave_and_bridge() {
        let vf = VolumeField::new_shattered_reach_3d(128, 128, 16);
        let r = 0.55f32;
        let h = 1.80f32;
        let dt = 1.0 / 60.0f32;

        // 1. Walk smoothly across the 3-span arched stone & timber bridge (x: 38 -> 90 along y=64)
        let mut pos = (38.0f32, 64.0f32, 8.4f32);
        let mut vel = (0.0f32, 0.0f32, 0.0f32);
        for _ in 0..30 {
            let (p, v, _) = vf.step_capsule_3d(pos, vel, r, h, dt);
            pos = p;
            vel = v;
        }
        let mut bridge_min_z = f32::MAX;
        let mut bridge_max_z = f32::MIN;
        let mut bridge_max_pen = 0.0f32;
        for _ in 0..700 {
            vel.0 = 5.2;
            vel.1 = (64.0 - pos.1) * 2.0;
            let (p, v, _) = vf.step_capsule_3d(pos, vel, r, h, dt);
            pos = p;
            vel = v;
            bridge_min_z = bridge_min_z.min(pos.2);
            bridge_max_z = bridge_max_z.max(pos.2);
            bridge_max_pen = bridge_max_pen.max(vf.measure_capsule_penetration_3d(pos, r, h));
            if pos.0 >= 89.5 {
                break;
            }
        }
        assert!(
            pos.0 >= 88.5 && bridge_min_z >= 7.5 && bridge_max_z <= 9.4 && bridge_max_pen < 0.05,
            "Bridge walk failed: end_x={:.2}, z=[{:.2}..{:.2}], max_pen={:.4}",
            pos.0, bridge_min_z, bridge_max_z, bridge_max_pen
        );

        // 2. Walk cleanly underneath the West bridge archway along x = 0.388*128 ≈ 49.66, y: 48 -> 80
        let arch_x = 0.388 * 128.0;
        let mut pos = (arch_x, 48.0f32, 3.1f32);
        let mut vel = (0.0f32, 0.0f32, 0.0f32);
        for _ in 0..30 {
            let (p, v, _) = vf.step_capsule_3d(pos, vel, r, h, dt);
            pos = p;
            vel = v;
        }
        let mut under_min_clear = f32::MAX;
        let mut under_max_pen = 0.0f32;
        for _ in 0..500 {
            vel.1 = 4.8;
            vel.0 = (arch_x - pos.0) * 2.0;
            let (p, v, _) = vf.step_capsule_3d(pos, vel, r, h, dt);
            pos = p;
            vel = v;
            if (pos.1 - 64.0).abs() <= 3.5 {
                under_min_clear = under_min_clear.min(vf.measure_vertical_clearance_3d(pos));
            }
            under_max_pen = under_max_pen.max(vf.measure_capsule_penetration_3d(pos, r, h));
            if pos.1 >= 79.0 {
                break;
            }
        }
        assert!(
            pos.1 >= 78.0 && pos.2 < 4.0 && under_min_clear >= 2.5 && under_max_pen < 0.05,
            "Under-bridge walk failed: end_y={:.2}, z={:.2}, min_clear={:.2}, max_pen={:.4}",
            pos.1, pos.2, under_min_clear, under_max_pen
        );

        // 3. Enter and traverse the vaulted basalt cave (east mouth -> grand inner chamber)
        let waypoints = [
            (0.305 * 128.0, 0.710 * 128.0),
            (0.235 * 128.0, 0.720 * 128.0),
        ];
        let mut pos = (0.375 * 128.0, 0.700 * 128.0, 1.9f32);
        let mut vel = (0.0f32, 0.0f32, 0.0f32);
        for _ in 0..30 {
            let (p, v, _) = vf.step_capsule_3d(pos, vel, r, h, dt);
            pos = p;
            vel = v;
        }
        let mut wp_idx = 0usize;
        let mut cave_min_clear = f32::MAX;
        let mut cave_max_pen = 0.0f32;
        for _ in 0..500 {
            let (tx, ty) = waypoints[wp_idx];
            let dx = tx - pos.0;
            let dy = ty - pos.1;
            let dist = dx.hypot(dy);
            if dist < 1.2 {
                if wp_idx + 1 < waypoints.len() {
                    wp_idx += 1;
                    continue;
                } else {
                    break;
                }
            }
            vel.0 = (dx / dist) * 4.5;
            vel.1 = (dy / dist) * 4.5;
            let (p, v, _) = vf.step_capsule_3d(pos, vel, r, h, dt);
            pos = p;
            vel = v;
            cave_max_pen = cave_max_pen.max(vf.measure_capsule_penetration_3d(pos, r, h));
            if pos.0 <= 0.34 * 128.0 {
                cave_min_clear = cave_min_clear.min(vf.measure_vertical_clearance_3d(pos));
            }
        }
        let (goal_x, goal_y) = waypoints[waypoints.len() - 1];
        let dist_goal = (pos.0 - goal_x).hypot(pos.1 - goal_y);
        assert!(
            wp_idx == waypoints.len() - 1 && dist_goal < 2.2 && cave_min_clear >= 2.5 && cave_max_pen < 0.05,
            "Cave traversal failed: dist_goal={:.2}, pos=({:.2},{:.2},{:.2}), min_clear={:.2}, max_pen={:.4}",
            dist_goal, pos.0, pos.1, pos.2, cave_min_clear, cave_max_pen
        );
    }

    #[test]
    fn test_multibiome_z16_and_z32_conservation_and_hydrologic_cycle() {
        // 1. Test ALL 5 biomes at both 64x64x16 and 64x64x32 (2.1M-scale depth!)
        for &depth in &[16usize, 32usize] {
            for biome in 0..NUM_BIOMES {
                assert!(!biome_name(biome).is_empty());
                let seed = 100 + biome as u64 * 17 + depth as u64;
                let mut vf = VolumeField::new_biome_3d(64, 64, depth, biome, seed);
                assert_eq!((vf.width, vf.height, vf.depth, vf.biome_id), (64, 64, depth, biome));

                // Verify FM3D brick encoding matches exact depth * 16 * 16 * 4 bytes
                let (dm, at) = vf.encode_3d_brick_16(2, 2);
                assert_eq!(dm.len(), depth * 16 * 16 * 4);
                assert_eq!(at.len(), depth * 16 * 16 * 4);

                // Verify distinct signature landmarks for each biome
                match biome {
                    BIOME_VOLCANIC_CALDERA => {
                        let max_t = (1..=4.min(depth - 1))
                            .flat_map(|z| {
                                let l = &vf.layers[z];
                                (0..(64 * 64)).map(move |i| l.cell_temp_ck(i))
                            })
                            .max()
                            .unwrap_or(0);
                        assert!(max_t >= 90_000, "Volcanic Caldera Z={} should have >=900K magma, got {}", depth, max_t);
                    }
                    BIOME_SINKHOLE_CAVERNS => {
                        // Central sinkhole skylight beside the hanging arch should be open air,
                        // and the hanging arch bridge at center (32, 32) should have open cavern air underneath!
                        let z_sc = (depth as f32 / 16.0).clamp(1.0, 2.0);
                        assert!(
                            vf.sample_solid_density_3d(32.0, 27.0, 8.5 * z_sc) < 0.20,
                            "Sinkhole Caverns Z={} central skylight shaft should be open air",
                            depth
                        );
                        assert!(
                            vf.sample_solid_density_3d(32.0, 32.0, 9.2 * z_sc) >= 0.50,
                            "Sinkhole Caverns Z={} hanging arch bridge should span sinkhole",
                            depth
                        );
                        assert!(
                            vf.sample_solid_density_3d(32.0, 32.0, 5.0 * z_sc) < 0.20,
                            "Sinkhole Caverns Z={} vaulted cavern underneath arch should be open air",
                            depth
                        );
                        let sub_water: u64 = (1..=4.min(depth - 1))
                            .map(|z| vf.layers[z].mass[SP_WATER].iter().map(|&m| m as u64).sum::<u64>())
                            .sum();
                        assert!(sub_water > 10_000, "Sinkhole Caverns Z={} should have subterranean azure pools", depth);
                    }
                    BIOME_GLACIER_GORGE => {
                        let total_ice: u64 = vf.layers.iter()
                            .map(|l| l.mass[SP_ICE].iter().map(|&m| m as u64).sum::<u64>())
                            .sum();
                        assert!(total_ice > 100_000, "Glacier Gorge Z={} should have glacial spires & ice bridges", depth);
                    }
                    BIOME_GEOTHERMAL_BASIN => {
                        let has_boiling_spring = vf.layers.iter().any(|l| {
                            (0..(64 * 64)).any(|i| l.mass[SP_WATER][i] >= 500 && l.cell_temp_ck(i) >= 38_000)
                        });
                        let total_steam: u64 = vf.layers.iter()
                            .map(|l| l.mass[SP_STEAM].iter().map(|&m| m as u64).sum::<u64>())
                            .sum();
                        assert!(has_boiling_spring, "Geothermal Basin Z={} should have boiling hot springs (>=380K)", depth);
                        assert!(total_steam > 10_000, "Geothermal Basin Z={} should have rising steam clouds", depth);
                    }
                    _ => {}
                }

                // Step multiple ticks and assert exact zero conservation error (mass_error == 0 && energy_error == 0)
                for tick in 0..3 {
                    vf.apply_weather_and_springs_3d((tick % 4) as u8, tick);
                    let raws = vf.refsim_raw_outputs_all();
                    let st = vf.step_3d(&raws);
                    assert_eq!(st.mass_error, 0, "biome={} Z={} mass_error at tick {}", biome, depth, tick);
                    assert_eq!(st.energy_error, 0, "biome={} Z={} energy_error at tick {}", biome, depth, tick);
                }
            }
        }

        // 2. Verify Closed 3D Hydrologic Condensation: high-altitude SP_STEAM cools & condenses into precipitating SP_WATER / SP_ICE!
        {
            let mut vf = VolumeField::new_biome_3d(64, 64, 16, BIOME_GEOTHERMAL_BASIN, 1);
            // Clear upper layers 11..15 and seed a dense warm steam cloud at z=14 (depth - 2) and z=15 (depth - 1)
            let idx_cloud = vf.layers[14].idx(32, 32);
            let idx_snow = vf.layers[15].idx(36, 32);
            for z in 11..16 {
                for &idx in &[idx_cloud, idx_snow] {
                    for s in 0..NUM_SPECIES {
                        vf.layers[z].mass[s][idx] = 0;
                    }
                    vf.layers[z].energy[idx] = 0;
                }
            }
            vf.layers[14].mass[SP_STEAM][idx_cloud] = 600;
            vf.layers[14].set_cell_temp(idx_cloud, 37_800);
            vf.layers[15].mass[SP_STEAM][idx_snow] = 600;
            vf.layers[15].set_cell_temp(idx_snow, 37_200);
            vf.snapshot_conservation_baseline();

            let mut condensed_water = 0u32;
            let mut condensed_ice = 0u32;
            for tick in 0..6 {
                let raws = vf.refsim_raw_outputs_all();
                let st = vf.step_3d(&raws);
                assert_eq!(st.mass_error, 0, "hydrologic mass_error at tick {}", tick);
                assert_eq!(st.energy_error, 0, "hydrologic energy_error at tick {}", tick);
                condensed_water = condensed_water.max(
                    (10..16).map(|z| vf.layers[z].mass[SP_WATER][idx_cloud] as u32).sum(),
                );
                condensed_ice = condensed_ice.max(
                    (10..16).map(|z| (vf.layers[z].mass[SP_ICE][idx_snow] + vf.layers[z].mass[SP_WATER][idx_snow]) as u32).sum(),
                );
            }
            assert!(
                condensed_water > 100,
                "expected high-altitude steam at z=14 to condense into precipitating water, got {}",
                condensed_water
            );
            assert!(
                condensed_ice > 100,
                "expected top-layer steam at z=15 to condense into precipitating ice/water, got {}",
                condensed_ice
            );
        }

        // 3. Verify 3D Destructible Bridge/Arch Structural Collapse on BOTH analytic and neural vertical paths!
        {
            let mut vf = VolumeField::new_biome_3d(64, 64, 16, BIOME_NORDIC_VIADUCT, 1);
            let wood_idx = vf.layers[8].idx(32, 32);
            let ice_idx = vf.layers[8].idx(36, 32);
            // Isolate two overhanging voxels at z=8 (open air below at z=7 and empty 4-neighbors at z=8)
            for &(cx, cy) in &[(32usize, 32usize), (36usize, 32usize)] {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let i = vf.layers[8].idx((cx as i32 + dx) as usize, (cy as i32 + dy) as usize);
                        for z in [7usize, 8usize] {
                            for s in 0..NUM_SPECIES {
                                vf.layers[z].mass[s][i] = 0;
                            }
                            vf.layers[z].energy[i] = 0;
                        }
                    }
                }
            }
            // Place burning unsupported wood voxel (>550 K) & melting unsupported ice voxel (>278 K)
            vf.layers[8].mass[SP_WOOD][wood_idx] = 600;
            vf.layers[8].set_cell_temp(wood_idx, 64_000);
            vf.layers[8].mass[SP_ICE][ice_idx] = 600;
            vf.layers[8].set_cell_temp(ice_idx, 29_000);
            vf.snapshot_conservation_baseline();

            let raws = vf.refsim_raw_outputs_all();
            let st = vf.step_3d(&raws);
            assert_eq!(st.mass_error, 0, "structural collapse mass_error");
            assert_eq!(st.energy_error, 0, "structural collapse energy_error");
            assert!(
                vf.layers[7].mass[SP_WOOD][wood_idx] > 0 && vf.layers[8].mass[SP_WOOD][wood_idx] < 600,
                "expected burning unsupported bridge wood to collapse downward from z=8 -> z=7"
            );
            assert!(
                vf.layers[7].mass[SP_ICE][ice_idx] > 0 && vf.layers[8].mass[SP_ICE][ice_idx] < 600,
                "expected melting unsupported ice arch voxel to collapse downward from z=8 -> z=7"
            );

            // Also verify neural 3D vertical flux path runs hydrologic + collapse with exact zero error!
            let z_flux = vec![0.0f32; (vf.depth - 1) * 5 * 64 * 64];
            let st_n = vf.step_3d_with_neural_z(&raws, Some(&z_flux));
            assert_eq!(st_n.mass_error, 0, "neural z_flux Z=16 mass_error");
            assert_eq!(st_n.energy_error, 0, "neural z_flux Z=16 energy_error");
        }
    }

    #[test]
    fn test_world_evolution_basalt_regrowth_and_conservation() {
        let mut vf = VolumeField::new_biome_3d(64, 64, 16, BIOME_NORDIC_VIADUCT, 1);

        // Seed an explicit lava/magma-water contact zone (hot SP_OIL + SP_WATER > 650 K)
        for dy in 0..3 {
            for dx in 0..3 {
                let idx = vf.layers[3].idx(30 + dx, 30 + dy);
                vf.layers[3].mass[SP_ROCK][idx] = 100;
                vf.layers[3].mass[SP_OIL][idx] = 420;
                vf.layers[3].mass[SP_WATER][idx] = 420;
                vf.layers[3].set_cell_temp(idx, 82_000);
            }
        }
        // Seed a sunlit ash + moisture clearing at z=6 with open sky above (`z=7..16` clear)
        for dy in 0..3 {
            for dx in 0..3 {
                let idx = vf.layers[6].idx(14 + dx, 14 + dy);
                for z in 7..16 {
                    for s in 0..NUM_SPECIES {
                        vf.layers[z].mass[s][idx] = 0;
                    }
                    vf.layers[z].energy[idx] = 0;
                }
                vf.layers[6].mass[SP_SOIL][idx] = 450;
                vf.layers[6].mass[SP_ASH][idx] = 280;
                vf.layers[6].mass[SP_WATER][idx] = 320;
                vf.layers[6].mass[SP_PLANT][idx] = 100;
                vf.layers[6].set_cell_temp(idx, 29_200);
            }
        }
        // Seed cold high-moisture glacial cap zone (`T < 268 K`)
        for dy in 0..2 {
            for dx in 0..2 {
                let idx = vf.layers[12].idx(32 + dx, 10 + dy);
                vf.layers[12].mass[SP_WATER][idx] = 420;
                vf.layers[12].mass[SP_ICE][idx] = 120;
                vf.layers[12].set_cell_temp(idx, 26_100);
            }
        }
        vf.snapshot_conservation_baseline();

        let basalt_before: u64 = vf.layers[3].mass[SP_ROCK].iter().map(|&m| m as u64).sum();
        let plant_wood_before: u64 = vf
            .layers
            .iter()
            .map(|l| {
                l.mass[SP_PLANT].iter().map(|&m| m as u64).sum::<u64>()
                    + l.mass[SP_WOOD].iter().map(|&m| m as u64).sum::<u64>()
            })
            .sum();

        for step in 0..30 {
            vf.step_world_evolution_3d(8.0);
            let mass_err = vf.current_accounted_mass() - vf.initial_total_mass;
            let energy_err = vf.current_accounted_energy() - vf.initial_total_energy;
            assert_eq!(mass_err, 0, "world evolution mass_error != 0 at step {}", step);
            assert_eq!(energy_err, 0, "world evolution energy_error != 0 at step {}", step);
        }

        let ev = vf.evolution_stats;
        assert_eq!(ev.world_epoch, 30);
        assert!(ev.basalt_forged > 0, "expected real basalt forged > 0, got {:?}", ev);
        assert!(ev.saplings_grown > 0, "expected real flora saplings_grown > 0, got {:?}", ev);
        assert!(ev.glaciers_formed > 0, "expected glacial crystallization > 0, got {:?}", ev);
        assert!(ev.erosion_events > 0, "expected hydraulic erosion > 0, got {:?}", ev);

        let basalt_after: u64 = vf.layers[3].mass[SP_ROCK].iter().map(|&m| m as u64).sum();
        let plant_wood_after: u64 = vf
            .layers
            .iter()
            .map(|l| {
                l.mass[SP_PLANT].iter().map(|&m| m as u64).sum::<u64>()
                    + l.mass[SP_WOOD].iter().map(|&m| m as u64).sum::<u64>()
            })
            .sum();
        assert!(
            basalt_after > basalt_before,
            "expected solid SP_ROCK mass to increase from basalt quenching ({} -> {})",
            basalt_before,
            basalt_after
        );
        assert!(
            plant_wood_after > plant_wood_before,
            "expected total organic SP_PLANT/SP_WOOD mass to increase ({} -> {})",
            plant_wood_before,
            plant_wood_after
        );
        // Also verify living world epoch transitions cleanly from PRISTINE
        let epoch_name = vf.evolution_epoch_name();
        assert_ne!(
            epoch_name, "EPOCH I: PRISTINE",
            "expected living epoch to advance after 30 evolution steps, got {}",
            epoch_name
        );
        vf.evolution_stats = WorldEvolutionStats::default();
        assert_eq!(vf.evolution_epoch_name(), "EPOCH I: PRISTINE");
        vf.evolution_stats = WorldEvolutionStats {
            basalt_forged: 80,
            saplings_grown: 10,
            glaciers_formed: 5,
            erosion_events: 0,
            world_epoch: 20,
        };
        assert_eq!(vf.evolution_epoch_name(), "EPOCH II: VOLCANIC FORGE");
        vf.evolution_stats.saplings_grown = 120;
        assert_eq!(vf.evolution_epoch_name(), "EPOCH III: VERDANT REGROWTH");
        vf.evolution_stats.glaciers_formed = 200;
        assert_eq!(vf.evolution_epoch_name(), "EPOCH IV: GLACIAL AGE");
    }

    #[test]
    fn test_burrower_conservative_tunnel_carving() {
        let mut vf = VolumeField::new_biome_3d(64, 64, 16, BIOME_NORDIC_VIADUCT, 1);

        // Build a solid subterranean hillside block (soil + soft rock) at x=20..36, y=28..36, z=1..7
        // above the z=0 bedrock floor
        for z in 1..7 {
            for y in 28..36 {
                for x in 20..36 {
                    let idx = vf.layers[z].idx(x, y);
                    for s in 0..NUM_SPECIES {
                        vf.layers[z].mass[s][idx] = 0;
                    }
                    if z <= 3 {
                        vf.layers[z].mass[SP_ROCK][idx] = 950;
                    } else {
                        vf.layers[z].mass[SP_SOIL][idx] = 950;
                    }
                    vf.layers[z].set_cell_temp(idx, T_AMB);
                }
            }
        }
        // Record z=0 bedrock mass along the tunnel line to verify bedrock is never breached
        let bedrock_before: u64 = (20..36)
            .map(|x| vf.layers[0].mass[SP_ROCK][vf.layers[0].idx(x, 32)] as u64)
            .sum();
        vf.snapshot_conservation_baseline();

        let density_before = vf.sample_solid_density_3d(26.5, 32.5, 3.5);
        assert!(
            density_before >= 0.85,
            "expected solid hillside before carving, got {}",
            density_before
        );

        // Burrower digs a horizontal tunnel straight through the solid rock/soil hillside!
        let mut total_displaced = 0u32;
        for step in 0..8 {
            let gx = 22.5 + (step as f32) * 1.25;
            for _ in 0..3 {
                let moved = vf.carve_tunnel_3d(gx, 32.5, 3.5, 1.0, 0.0, 1.85);
                total_displaced = total_displaced.saturating_add(moved);
                assert_eq!(
                    vf.current_accounted_mass() - vf.initial_total_mass,
                    0,
                    "burrower tunnel carving broke exact mass conservation!"
                );
                assert_eq!(
                    vf.current_accounted_energy() - vf.initial_total_energy,
                    0,
                    "burrower tunnel carving broke exact energy conservation!"
                );
            }
        }
        // Also attempt to carve right at z=0.5 to verify z==0 bedrock floor stays intact
        let _ = vf.carve_tunnel_3d(26.5, 32.5, 0.5, 1.0, 0.0, 1.85);
        assert_eq!(vf.current_accounted_mass() - vf.initial_total_mass, 0);
        assert_eq!(vf.current_accounted_energy() - vf.initial_total_energy, 0);

        let bedrock_after: u64 = (20..36)
            .map(|x| vf.layers[0].mass[SP_ROCK][vf.layers[0].idx(x, 32)] as u64)
            .sum();
        assert!(
            bedrock_after >= bedrock_before,
            "z=0 bedrock floor must never lose rock mass ({} -> {})",
            bedrock_before,
            bedrock_after
        );
        assert!(
            total_displaced > 5000,
            "expected significant conservative mass displacement, got {}",
            total_displaced
        );

        // Verify a real traversable air cavity opened along the tunnel core!
        let density_after = vf.sample_solid_density_3d(26.5, 32.5, 3.5);
        assert!(
            density_after < 0.25,
            "expected traversable carved air cavity (<0.25 solid density), got {} (before={})",
            density_after,
            density_before
        );

        // Step the 3D physics after carving and verify conservation remains strictly 0!
        let raws = vf.refsim_raw_outputs_all();
        let st = vf.step_3d(&raws);
        assert_eq!(st.mass_error, 0, "post-carve step_3d mass_error != 0");
        assert_eq!(st.energy_error, 0, "post-carve step_3d energy_error != 0");
    }

    #[test]
    fn test_world_law_presets_exact_conservation() {
        assert_eq!(WorldLaws::preset_name(WORLD_LAW_STANDARD), "STANDARD");
        assert_eq!(WorldLaws::preset_name(WORLD_LAW_FREEZE_OUT), "FREEZE-OUT");
        assert_eq!(WorldLaws::preset_name(WORLD_LAW_RISING_FLOOD), "RISING FLOOD");
        assert_eq!(WorldLaws::preset_name(WORLD_LAW_LOW_G_MAGMA), "LOW-G MAGMA");

        let std_laws = WorldLaws::from_preset(0);
        let frz_laws = WorldLaws::from_preset(1);
        let fld_laws = WorldLaws::from_preset(2);
        let mag_laws = WorldLaws::from_preset(3);
        assert!((std_laws.gravity_g - 1.0).abs() < 1e-6 && std_laws.ambient_temp_shift == 0.0);
        assert!((frz_laws.ambient_temp_shift + 5500.0).abs() < 1e-3 && (frz_laws.precipitation - 1.4).abs() < 1e-4);
        assert!((fld_laws.gravity_g - 1.15).abs() < 1e-4 && (fld_laws.precipitation - 2.6).abs() < 1e-4);
        assert!((mag_laws.gravity_g - 0.35).abs() < 1e-4 && (mag_laws.ambient_temp_shift - 5500.0).abs() < 1e-3);

        // Run all 4 presets from the exact same 32x32x16 seeded state and compare physical outcomes
        let mut final_ice = [0u64; 4];
        let mut final_water = [0u64; 4];
        let mut mean_temp_ck = [0i64; 4];
        let mut total_buoy = [0i64; 4];

        for preset in 0..4u8 {
            let mut vf = VolumeField::new_biome_3d(32, 32, 16, BIOME_NORDIC_VIADUCT, 7);
            // Seed uniform water pool at z=5 and hot steam cloud at z=11..14 so all 4 presets diverge cleanly
            for y in 10..22 {
                for x in 10..22 {
                    let idx = vf.layers[5].idx(x, y);
                    for s in 0..NUM_SPECIES {
                        vf.layers[5].mass[s][idx] = 0;
                    }
                    vf.layers[5].mass[SP_WATER][idx] = 900;
                    vf.layers[5].set_cell_temp(idx, 28_000);

                    for z in 11..=14 {
                        let cidx = vf.layers[z].idx(x, y);
                        for s in 0..NUM_SPECIES {
                            vf.layers[z].mass[s][cidx] = 0;
                        }
                        vf.layers[z].mass[SP_STEAM][cidx] = 600;
                        vf.layers[z].set_cell_temp(cidx, 38_200);
                    }
                }
            }
            vf.snapshot_conservation_baseline();
            vf.set_world_law_preset(preset);
            assert_eq!(vf.world_law_preset, preset);

            for tick in 0..10 {
                let raws = vf.refsim_raw_outputs_all();
                let st = vf.step_3d(&raws);
                assert_eq!(st.mass_error, 0, "preset {} mass_error != 0 at tick {}", preset, tick);
                assert_eq!(st.energy_error, 0, "preset {} energy_error != 0 at tick {}", preset, tick);
                total_buoy[preset as usize] += st.vertical_buoyancy_mass;
            }

            let mut ice_sum = 0u64;
            let mut water_sum = 0u64;
            let mut t_sum = 0i64;
            let mut t_cnt = 0i64;
            for z in 0..16 {
                for &m in &vf.layers[z].mass[SP_WATER] {
                    water_sum += m as u64;
                }
                for &m in &vf.layers[z].mass[SP_ICE] {
                    ice_sum += m as u64;
                }
            }
            for y in 10..22 {
                for x in 10..22 {
                    let idx = vf.layers[5].idx(x, y);
                    t_sum += vf.layers[5].cell_temp_ck(idx);
                    t_cnt += 1;
                }
            }
            final_ice[preset as usize] = ice_sum;
            final_water[preset as usize] = water_sum;
            mean_temp_ck[preset as usize] = t_sum / t_cnt.max(1);
        }

        assert!(
            final_ice[1] > final_ice[0] && final_ice[1] > final_ice[3],
            "FREEZE-OUT must produce strictly more SP_ICE than STANDARD and LOW-G MAGMA: {:?}",
            final_ice
        );
        assert!(
            final_water[2] > final_water[0] && final_water[2] > final_water[3],
            "RISING FLOOD must condense strictly more SP_WATER than STANDARD and LOW-G MAGMA: {:?}",
            final_water
        );
        assert!(
            mean_temp_ck[3] > mean_temp_ck[0] && mean_temp_ck[0] > mean_temp_ck[1],
            "LOW-G MAGMA must stay hotter than STANDARD, and STANDARD hotter than FREEZE-OUT: {:?}",
            mean_temp_ck
        );
        assert!(
            total_buoy[3] > total_buoy[0],
            "LOW-G MAGMA must exhibit higher vertical buoyancy transport than STANDARD: {:?}",
            total_buoy
        );
    }

    #[test]
    fn test_volume_snapshot_exact_conservation_roundtrip() {
        let mut vf = VolumeField::new_biome_3d(32, 32, 16, BIOME_VOLCANIC_CALDERA, 42);
        vf.set_world_law_preset(WORLD_LAW_LOW_G_MAGMA);
        vf.apply_weather_and_springs_3d(3, 1);
        vf.layers[4].apply_fire_bolt(16, 16, 3, 1.5);
        vf.layers[4].apply_oil_splash(12, 12, 2, 1.0);
        vf.carve_tunnel_3d(16.0, 16.0, 4.0, 1.0, 0.0, 2.0);
        for _ in 0..4 {
            vf.step_world_evolution_3d(4.0);
            let raws = vf.refsim_raw_outputs_all();
            let st = vf.step_3d(&raws);
            assert_eq!(st.mass_error, 0);
            assert_eq!(st.energy_error, 0);
        }
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);
        assert!(vf.tick == 4 && vf.world_time_secs >= 0.39);

        let snap = vf.to_snapshot();
        let mut restored = VolumeField::from_snapshot(snap).expect("restore snapshot");
        assert_eq!(restored.width, vf.width);
        assert_eq!(restored.height, vf.height);
        assert_eq!(restored.depth, vf.depth);
        assert_eq!(restored.biome_id, BIOME_VOLCANIC_CALDERA);
        assert_eq!(restored.seed, 42);
        assert_eq!(restored.tick, vf.tick);
        assert_eq!(restored.world_law_preset, WORLD_LAW_LOW_G_MAGMA);
        assert_eq!(restored.world_laws, vf.world_laws);
        assert_eq!(restored.evolution_stats, vf.evolution_stats);
        assert!((restored.world_time_secs - vf.world_time_secs).abs() < 1e-9);
        assert_eq!(restored.mass_error(), 0, "restored mass_error must be 0");
        assert_eq!(restored.energy_error(), 0, "restored energy_error must be 0");

        // Verify post-reload stepping preserves exact conservation and identical encoded bricks
        let (dm_orig, at_orig) = vf.encode_3d_brick_16(1, 1);
        let (dm_rest, at_rest) = restored.encode_3d_brick_16(1, 1);
        assert_eq!(dm_orig, dm_rest);
        assert_eq!(at_orig, at_rest);

        for tick in 0..3 {
            let raws = restored.refsim_raw_outputs_all();
            let st = restored.step_3d(&raws);
            assert_eq!(st.mass_error, 0, "post-reload mass_error at tick {}", tick);
            assert_eq!(st.energy_error, 0, "post-reload energy_error at tick {}", tick);
        }
    }

    #[test]
    fn test_vertical_z64_three_strata_conservation_and_cubic_bricks() {
        assert_eq!(BIOME_SKY_ABYSS, 5);
        assert_eq!(BIOME_COUNT, 6);
        assert_eq!(NUM_BIOMES, 6);
        assert_eq!(biome_name(BIOME_SKY_ABYSS), "Sky-Isles & Abyss");

        // Build a 64×64×64 3-strata Sky-Isles & Abyss world (4×4×4 = 64 cubic 16×16×16 sub-bricks)
        let mut vf = VolumeField::new_biome_3d(64, 64, 64, BIOME_SKY_ABYSS, 1);
        assert_eq!((vf.width, vf.height, vf.depth, vf.biome_id), (64, 64, 64, BIOME_SKY_ABYSS));
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);

        // 1. Verify Stratum 3: Real Floating Sky-Island solid voxels at z >= 44 (Central Sky-Citadel!)
        let sky_island_solid = (44..52).any(|z| vf.sample_solid_density_3d(32.0, 32.0, z as f32 + 0.5) >= 0.5);
        assert!(
            sky_island_solid,
            "Z=64 Sky-Isles & Abyss must have real Floating Sky-Island solid voxels at z >= 44"
        );
        // Open sky gap between Surface Viaduct (z~29) and Sky-Citadel underside (z~41)
        assert!(
            vf.sample_solid_density_3d(32.0, 32.0, 35.0) < 0.2,
            "Z=64 must have open sky gap at z=35m between Viaduct and Sky-Citadel"
        );

        // 2. Verify Stratum 2: Real Surface Viaduct solid deck at z ~ 28..29m with open gorge air underneath!
        let viaduct_solid = (27..=30).any(|z| vf.sample_solid_density_3d(32.0, 32.0, z as f32 + 0.5) >= 0.5);
        assert!(
            viaduct_solid,
            "Z=64 Sky-Isles & Abyss must have real Surface Viaduct solid deck at z ~ 28m"
        );
        assert!(
            vf.sample_solid_density_3d(32.0, 32.0, 24.0) < 0.2,
            "Z=64 Surface Viaduct must have open arch air underneath at z ~ 24m"
        );

        // 3. Verify Stratum 1: Real Abyssal Cavern void at z ~ 8m with bedrock floor below!
        assert!(
            vf.sample_solid_density_3d(32.0, 32.0, 8.0) < 0.2,
            "Z=64 Sky-Isles & Abyss must have real Abyssal Cavern void at z ~ 8m"
        );
        assert!(
            vf.sample_solid_density_3d(32.0, 32.0, 1.0) >= 0.5,
            "Z=64 Abyssal Cavern must rest on solid bedrock at z=1m"
        );

        // 4. Measure cubic 16×16×16 sub-brick sky/bedrock skip rate (> 40%!)
        let is_skippable = |dm: &[u8], at: &[u8]| -> bool {
            let empty_sky = dm
                .chunks_exact(4)
                .zip(at.chunks_exact(4))
                .all(|(d, a)| d[0] == 0 && d[1] == 0 && d[2] == 0 && a[3] == 0);
            if empty_sky {
                return true;
            }
            let (mut min_r, mut max_r) = (u8::MAX, 0u8);
            let (mut min_g, mut max_g) = (u8::MAX, 0u8);
            let (mut min_b, mut max_b) = (u8::MAX, 0u8);
            let (mut min_t, mut max_t) = (u8::MAX, 0u8);
            let (mut min_rg, mut max_rg) = (u8::MAX, 0u8);
            for (d, a) in dm.chunks_exact(4).zip(at.chunks_exact(4)) {
                if d[0] < 245 || d[1] != 0 || d[2] != 0 {
                    return false;
                }
                min_rg = min_rg.min(d[3]);
                max_rg = max_rg.max(d[3]);
                min_r = min_r.min(a[0]);
                max_r = max_r.max(a[0]);
                min_g = min_g.min(a[1]);
                max_g = max_g.max(a[1]);
                min_b = min_b.min(a[2]);
                max_b = max_b.max(a[2]);
                min_t = min_t.min(a[3]);
                max_t = max_t.max(a[3]);
                if max_rg - min_rg > 4
                    || max_r - min_r > 4
                    || max_g - min_g > 4
                    || max_b - min_b > 4
                    || max_t - min_t > 4
                {
                    return false;
                }
            }
            true
        };

        let bx_n = vf.width / CHUNK_SIZE;
        let by_n = vf.height / CHUNK_SIZE;
        let bz_n = vf.depth.div_ceil(CHUNK_SIZE);
        let mut total_sub = 0usize;
        let mut skipped_sub = 0usize;
        for bz in 0..bz_n {
            for by in 0..by_n {
                for bx in 0..bx_n {
                    let (dm, at, sd) = vf.encode_3d_subbrick_16(bx, by, bz);
                    assert_eq!(sd, 16);
                    assert_eq!(dm.len(), 16 * 16 * 16 * 4);
                    total_sub += 1;
                    if is_skippable(&dm, &at) {
                        skipped_sub += 1;
                    }
                }
            }
        }
        let skip_pct = (skipped_sub as f64 / total_sub as f64) * 100.0;
        assert!(
            skip_pct > 40.0,
            "Z=64 cubic sub-brick sky/bedrock skip rate must be > 40%, got {:.1}% ({}/{})",
            skip_pct,
            skipped_sub,
            total_sub
        );

        // 5. Run 10 full 3D ticks with altitude-coupled gravity & ambient relaxation -> exact 0 error!
        for tick in 0..10 {
            vf.apply_weather_and_springs_3d((tick % 4) as u8, tick);
            let raws = vf.refsim_raw_outputs_all();
            let st = vf.step_3d(&raws);
            assert_eq!(st.mass_error, 0, "Z=64 mass_error != 0 at tick {}", tick);
            assert_eq!(st.energy_error, 0, "Z=64 energy_error != 0 at tick {}", tick);
        }
        assert_eq!(vf.mass_error(), 0);
        assert_eq!(vf.energy_error(), 0);
    }
}
